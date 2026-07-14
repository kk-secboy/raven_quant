# Tushare data center

The data center stores immutable market files in Parquet and mutable task state in
PostgreSQL. A download job never writes research tables into PostgreSQL. Each provider
request is a resumable work unit with an atomic output file, row count and SHA-256.

## Download bundles

| Task | Interfaces |
| --- | --- |
| `cn_extended_daily` | historical ST state, SW industry bars, holders, surveys, block trades, northbound holdings, professional factors, audit opinions, business composition and IPOs |
| `cn_funds` | ETF-specific masters/tracking indices plus exchange-traded and OTC fund masters, companies, managers, NAV, shares/scale, dividends and portfolios |
| `cn_macro` | GDP, CPI, PPI, PMI, money supply, social financing, Shibor, LPR and official publication schedules |
| `cn_futures` | contract masters, exchange calendars, daily bars, continuous mappings, holdings, warehouse receipts, settlement parameters, price limits and minimum margin ratios |
| `cn_options_bonds` | option masters/bars, convertible-bond issue, redemption, coupon, conversion-price/share lifecycle, ratings, top holders, daily bars, repos and yield curves |
| `hk_market` | masters, calendars, daily/adjusted prices and four financial-statement/indicator interfaces |
| `us_market` | masters, calendars, daily/adjusted prices and four financial-statement/indicator interfaces |
| `global_markets` | international indices, overseas FX masters/daily prices and the US Treasury yield curve |

### Goal interface contract

The production task catalog and downloader are tested against the same logical dataset
set. The exact Tushare-facing APIs required by this delivery are:

| Domain | Tushare APIs |
| --- | --- |
| A-share core | `stock_basic`, `trade_cal`, `daily`, `adj_factor`, `daily_basic`, `suspend_d`, `stk_limit`, `limit_list_d`, `stock_st`, financial statements, corporate events and research flows |
| Domestic indices | `index_daily`, `index_weight` for SSE 50, CSI 300, CSI 500 and CSI 1000 |
| Industry risk | `index_classify`, `index_member_all`, `sw_daily` for point-in-time SW2021 membership and industry returns/valuation |
| Funds | `fund_basic` (`E`/`O`, split by status), `fund_company`, `fund_manager`, `fund_nav`, `fund_share`, `fund_div`, `fund_portfolio`, `etf_basic`, `etf_index`, plus ETF `fund_daily` and `fund_adj` |
| Futures | `fut_basic`, `fut_trade_cal`, `fut_mapping`, `fut_daily`, `fut_holding`, `fut_wsr`, `fut_settle`, `ft_limit` |
| Options and bonds | `opt_basic`, `opt_daily`, `cb_basic`, `cb_issue`, `cb_call`, `cb_rate`, `cb_price_chg`, `cb_share`, `cb_rating`, `top10_cb_holders`, `cb_daily`, `repo_daily`, `yc_cb` |
| Hong Kong | `hk_basic`, `hk_tradecal`, `hk_daily`, `hk_daily_adj`, `hk_income`, `hk_balancesheet`, `hk_cashflow`, `hk_fina_indicator` |
| United States | `us_basic`, `us_tradecal`, `us_daily`, `us_daily_adj`, `us_income`, `us_balancesheet`, `us_cashflow`, `us_fina_indicator` |
| Macro timing | `cn_schedule` alongside the domestic macro value interfaces, so point-in-time research can use publication dates |
| Global signals | `index_global`, `fx_obasic`, `fx_daily`, `us_tycr` |
| Execution data | `margin_secs`, `idx_mins`, `etf_mins`, `ft_mins`, `opt_mins`, `stk_mins` |

This is the required beginner-to-research platform scope, not a claim that every
Tushare product is downloaded. Newly added premium specialty feeds, real-time feeds,
all-market tick data and Level-2 remain separate opt-in products with their own storage,
permissions and cost controls.

The domestic index bootstrap includes SSE 50, CSI 300, CSI 500 and CSI 1000. Weekly
and monthly bars are derived locally from daily data instead of being downloaded again.

Hong Kong and US downloads always fetch the market master, calendar and requested
daily price window. US income, balance-sheet and cash-flow history uses reporting-
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
Paged Tushare interfaces are expanded lazily: the planner creates only page zero and
adds another page only when the previous response is exactly full. A short non-empty
page is accepted as the terminal page; a full final allowed page fails closed as a
possible provider truncation. This avoids the old fixed-page fan-out and its empty calls.
The page ceiling is only a runaway-request guard. It is deliberately above the dense
partitions seen in production (fund portfolios, company holders, futures holdings and
option masters/daily bars), while still requiring a short terminal page before a task
can be reported as complete.

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
- every IF/IC/IM/IH continuous code and mapped deliverable contract active in the
  requested historical range;
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

The Web data center exposes the same configuration, task coverage, provider row counts,
failure reason and retry-by-original-parameters action. Tick, transactions and Level-2
remain outside the Tushare-only beginner scope and require a separately licensed source.
