# Tushare data center

Interface coverage claims are governed by
[`TUSHARE_COVERAGE.md`](TUSHARE_COVERAGE.md). Downloader implementation,
production materialization, and Qlib readiness must always be reported
separately.

The data center stores immutable market files in Parquet and mutable task state in
PostgreSQL. A download job never writes research tables into PostgreSQL. Each provider
request is a resumable work unit with an atomic output file, row count and SHA-256.

## Download bundles

| Task | Interfaces |
| --- | --- |
| `cn_extended_daily` | historical ST state, SW industry bars, holders, surveys, block trades, northbound holdings, professional factors, audit opinions, business composition and IPOs |
| `cn_funds` | ETF-specific masters/tracking indices plus exchange-traded and OTC fund masters, companies, managers, NAV, shares/scale, dividends and portfolios |
| `cn_macro` | GDP, CPI, PPI, PMI, money supply, social financing, Shibor, LPR and official publication schedules |
| `cn_institutional` | broker research forecasts, Shanghai/Shenzhen ETF creation-redemption baskets, CITIC industry daily indices, Shibor contributor quotes and long-form financial news |
| `cn_futures` | contract masters, exchange calendars, daily bars, continuous mappings, holdings, warehouse receipts, settlement parameters, price limits and minimum margin ratios |
| `cn_options_bonds` | option masters/bars, convertible-bond issue, redemption, coupon, conversion-price/share lifecycle, ratings, top holders, daily bars, repos and yield curves |
| `hk_market` | masters, calendars, daily/adjusted prices and four financial-statement/indicator interfaces |
| `us_market` | masters, calendars, daily/adjusted prices and four financial-statement/indicator interfaces |
| `global_markets` | international indices, overseas FX masters/daily prices and the US Treasury yield curve |
| `cn_governance_risk` | corporate/governance masters, manager rewards, abnormal volatility, chip distribution, CCASS, broker recommendations, margin and securities lending |
| `cn_capital_flow` | northbound/southbound, THS and Eastmoney security, concept, industry and market-wide capital flow |
| `cn_fund_index_enhanced` | ETF scale, index announcements/members/factors, exchange statistics, benchmark libraries and fund factors |
| `cn_derivatives_enhanced` | futures indices/weekly statistics, Shanghai gold, convertible-bond factors, OTC bond quotes, block trades and economic calendar |
| `global_rates_enhanced` | Hong Kong/US adjustment factors plus LIBOR, HIBOR and US rate curves |
| `research_corpus` | policy, research-report, financial-news, exchange Q&A and public-account text corpora |
| `strategy_specialty` | optional limit-board, theme, hot-list, active-capital and vendor index/member feeds |
| `strategy_specialty_minutes` | explicitly selected SW-index and Hong Kong 5-minute histories |

### Goal interface contract

The production task catalog and downloader are tested against the same logical dataset
set. The exact Tushare-facing APIs required by this delivery are:

| Domain | Tushare APIs |
| --- | --- |
| A-share core | `stock_basic`, `trade_cal`, `daily`, `adj_factor`, `daily_basic`, `suspend_d`, `stk_limit`, `limit_list_d`, `stock_st`, financial statements, corporate events and research flows |
| Domestic indices | complete paged `index_basic`, `index_daily`, `index_dailybasic` and `index_weight` for SSE Composite, SSE 50, CSI 300, STAR 50, CSI 500, CSI 1000, Shenzhen Component, ChiNext and Beijing 50 |
| Industry risk | `index_classify`, `index_member_all`, `sw_daily` for point-in-time SW2021 membership and industry returns/valuation |
| Funds | `fund_basic` (`E`/`O`, split by status), `fund_company`, `fund_manager`, `fund_nav`, `fund_share`, `fund_div`, `fund_portfolio`, `etf_basic`, `etf_index`, plus ETF `fund_daily` and `fund_adj` |
| Futures | `fut_basic`, `fut_trade_cal`, `fut_mapping`, `fut_daily`, `fut_holding`, `fut_wsr`, `fut_settle`, `ft_limit` |
| Options and bonds | `opt_basic`, `opt_daily`, `cb_basic`, `cb_issue`, `cb_redeem`, `cb_rate`, `cb_price_chg`, `cb_share`, `cb_rating`, `top10_cb_holders`, `cb_daily`, `repo_daily`, `yc_cb` |
| Hong Kong | `hk_basic`, `hk_tradecal`, `hk_daily`, `hk_daily_adj`, `hk_income`, `hk_balancesheet`, `hk_cashflow`, `hk_fina_indicator` |
| United States | `us_basic`, `us_tradecal`, `us_daily`, `us_daily_adj`, `us_income`, `us_balancesheet`, `us_cashflow`, `us_fina_indicator` |
| Macro timing | `cn_schedule` alongside the domestic macro value interfaces, so point-in-time research can use publication dates |
| Institutional research | `report_rc`, `etf_sh_cons`, `etf_sz_cons`, `ci_daily`, `shibor_quote`, `major_news` |
| Global signals | `index_global`, `fx_obasic`, `fx_daily`, `us_tycr` |
| Execution data | `margin_secs`, `idx_mins`, `etf_mins`, `ft_mins`, `opt_mins`, selected-stock `stk_mins` at 1 minute and all-listed-A-share `stk_mins` at 5 minutes |

The reviewed offline historical target now has formal downloaders for all 211 selected
Tushare interfaces. This is not a promise that future provider additions are implicitly
enabled: the official catalog is reconciled as a dated release gate. Exchange Tick,
transaction-level and Level-2 feeds remain external products with their own storage,
permissions and cost controls.

Weekly and monthly bars are derived locally from daily data instead of being downloaded
again.

Hong Kong and US downloads first fetch the market master and official calendar, then
plan daily and adjusted-price pages only for rows where `is_open=1`. Hong Kong masters
request `L`, `D` and `P` statuses, and the automatic financial universe keeps only
symbols whose listing lifecycle intersects the requested range. Explicit `--symbols`
values are used unchanged. US income, balance-sheet and cash-flow history uses reporting-
period cross sections through the available VIP interfaces. The US financial-indicator
interface and all Hong Kong financial interfaces require exactly one `ts_code`;
`--symbols` (or the Web “financial universe” field) therefore bounds those unavoidable
fan-outs, while an empty value selects every downloaded market symbol:

```powershell
python -m quant_data.cli supplemental-download `
  --bundle hk_market --start 2026-07-10 --end 2026-07-10 `
  --symbols 00700.HK
```

## Independent pipeline stages

The durable pipeline is deliberately split into separate jobs:

1. Download provider work units.
2. Verify completion, file hashes, empty-result policy and primary-key uniqueness.
3. Compact verified units into a new immutable Parquet snapshot.
4. Build Qlib binary data from that exact snapshot.
5. Run the Qlib Alpha158 acceptance baseline.

Each stage has its own job, log, error and retry operation. Retrying a failed job keeps
the original parameters and does not repeat already successful provider work units.
As-of reference requests use reviewed daily, weekly or monthly refresh buckets. A new
bucket creates a successor work unit without deleting the earlier file. Immutable old
snapshots retain their original source manifest; a new snapshot selects the newest
successful version per reference partition and records its bucket, cadence and source
hash. Dated datasets also record `date_min` and `date_max`, and the snapshot-level
coverage audit enumerates every dataset with observations before 2024.

Paged Tushare interfaces are expanded lazily: the planner creates only page zero and
adds another page only when the previous response is exactly full. A short non-empty
page is accepted as the terminal page; a full final allowed page fails closed as a
possible provider truncation. This avoids the old fixed-page fan-out and its empty calls.
The page ceiling is only a runaway-request guard. It is deliberately above the dense
partitions seen in production (fund portfolios, company holders, futures holdings and
option masters/daily bars), while still requiring a short terminal page before a task
can be reported as complete.

Date and time partitions use one recovery contract: request the largest reviewed
window, paginate it, and bisect it only when a provider row/offset ceiling is reached.
Child windows are adjacent and carry their parent/supersession identity in `FetchSpec.scope`.
A single day or second that still reaches the limit fails explicitly. Successful legacy
units remain immutable; only old pending/failed units are marked `superseded`.

The all-A-share 5-minute task budgets at most 150 actual exchange sessions per symbol,
so a 130-session range normally starts with one request per stock instead of monthly
fan-out. News starts with one source/day request and splits by seconds only for a capped
source/day. Existing successful monthly minute units and paired half-day news units are
reused when they are fully contained in the requested range.

Sparse interfaces use wider initial partitions: `fund_share` and THS/Eastmoney concept
or industry flows use month ranges; northbound/southbound and market-wide Eastmoney
flows use year ranges. Dense fund NAV, stock-level money flow and CCASS detail remain
daily. Fund dividends and portfolios use calendar days so weekend announcements are not
omitted. ETF creation/redemption baskets are planned per ETF over the full requested
history and bisected by date only if pagination cannot prove termination.

Provider limits are part of the download contract. For example, overseas FX daily
requests use the documented 1,000-row page, fund masters separately request the `E`
and `O` markets by listing status, US financial statements use 1,000-row reporting-
period pages, and the single-symbol US-indicator/Hong Kong financial limits fail closed
instead of being silently accepted as complete responses.

Institutional reference additions follow the same fail-closed rules: `stock_st`,
`sw_daily` and `ft_limit` are partitioned by trading day; ETF masters are split by
listing status; `cn_schedule` is partitioned by month; and `us_tycr` is partitioned by
calendar year. A provider response at its documented row limit is rejected rather than
silently treated as complete.

Convertible-bond endpoints that require `ts_code` are a second planning stage: the
downloader first materializes `cb_basic`, then batches its symbols for coupons,
conversion-price changes, conversion results, ratings and top-holder history. This
prevents an empty-parameter call from being mistaken for a real lifecycle download.

A-share audit opinions and main-business composition use the cross-sectional VIP
interfaces. `fina_audit_vip` is partitioned by reporting period, while
`fina_mainbz_vip` is partitioned by reporting period and product, region or industry
view. Each partition is paginated in 1,000-row batches. The ordinary interfaces are
not used for history initialization because they require one `ts_code` and do not
interpret comma-separated stock codes as a batch.

The full A-share bootstrap follows the same rule. Income statements, balance sheets,
cash flows, indicators, forecasts and express reports use their VIP reporting-period
interfaces. Name changes, dividends, repurchases, unlock schedules, pledge statistics,
pledge details, holder trades and announcements use monthly or exact-date full-market
partitions. The former per-stock fundamental and corporate-event planners have been
removed, so there is no parallel slow path.

## Automatic minute universe

`core-intraday --auto-universe` resolves a reproducible universe from downloaded
masters before planning minute requests:

- SSE 50, CSI 300, CSI 500 and CSI 1000;
- broad, industry, gold and bond ETF groups;
- every IF/IC/IM/IH mapped deliverable contract active in the requested historical
  range; synthetic continuous codes are reconstructed locally from `fut_mapping`
  because the Tushare minute endpoint only accepts real contracts;
- the configurable top liquid A-share pool by mean amount over the latest 20 available
  trading days, excluding ST names;
- configurable top active SSE/SZSE option contracts by amount and volume.

Manual codes are merged with, not replaced by, the automatic selection. The selected
symbols, rule names, source dates and counts are written into the immutable execution
snapshot manifest. Minute calls are partitioned by symbol and month (two-week windows
for futures) and reject possible 8,000-row truncation.

Example:

```powershell
python -m quant_data.cli core-intraday `
  --start 2024-01-01 --end latest --auto-universe `
  --max-stocks 100 --max-options 100
```

The Web data center separates catalog readiness from live download execution. Catalog
readiness describes which research capabilities are available; it is not a percentage
for the currently running job. Active jobs publish their target range, request strategy,
adaptive-partition phase and checkpoint counters through `jobs.progress_json`. The UI
therefore distinguishes queued/running work, shared rate-limit cooldown, scheduled retry,
blocked prerequisites, recoverable failure and terminal failure. Successful, pending,
running, retrying and superseded work units remain visible for resume and audit, and retry
actions preserve the original parameters. Tick, transactions and Level-2 remain outside
the Tushare-only beginner scope and require a separately licensed source.
