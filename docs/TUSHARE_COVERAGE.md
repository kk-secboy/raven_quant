# Tushare interface coverage contract

This document is the release gate for claims about data coverage. QuantLab must
not describe Tushare coverage as "complete" unless all three levels below are
reported separately and the target interface inventory has no open rows.

## Coverage levels

1. **Downloader implemented**: a resumable planner, provider call, pagination or
   row-cap proof, validation, Parquet storage, and regression test exist.
2. **Remote data materialized**: the production job completed for the requested
   date range and every planned work unit reached a successful terminal state.
3. **Research-ready**: an immutable snapshot exists and, where applicable, a
   frequency-correct Qlib dataset has passed its validation gate.

These levels are not interchangeable. A downloader can exist without the server
having downloaded the data, and a Parquet snapshot can exist without a Qlib
conversion.

## Official catalog audit snapshot

The 2026-07-15 audit parsed the current Tushare data-interface navigation and
its leaf documentation pages, then compared the documented API names with the
provider calls reachable from this repository:

- 243 documented APIs were identified;
- 211 are semantically covered by local downloaders (including the bulk VIP
  equivalents of the same financial statements and the 85 interfaces added in
  this coverage release);
- 32 are intentionally outside the offline research warehouse:
  12 real-time feeds, 7 locally derived weekly/monthly feeds, 5 stopped/legacy
  feeds, 4 cloud portfolio CRUD APIs, and 4 redundant alternatives;
- no interface remains in the reviewed 211-interface offline historical target.

The 85 interfaces added by this release are **implemented and regression-tested
locally**, but they are not described as remotely materialized until the
production queue is idle, the release is deployed, and the corresponding jobs
finish. Provider permission failures are recorded per interface and do not
block unrelated bundles.

These counts are a dated audit snapshot, not a permanent property of Tushare.
The official catalog must be reconciled again before any future claim of full
provider coverage.

### Default broad institutional coverage added locally (59)

- A-share reference, governance and risk: `st`, `stock_hsgt`, `stock_company`,
  `stk_managers`, `stk_rewards`, `bse_mapping`, `ggt_top10`, `ggt_daily`,
  `stk_shock`, `stk_high_shock`, `stk_alert`, `cyq_perf`, `cyq_chips`,
  `ccass_hold`, `ccass_hold_detail`, `broker_recommend`, `margin`, `slb_len`.
- Capital flow: `moneyflow_hsgt`, `moneyflow_ths`, `moneyflow_dc`,
  `moneyflow_cnt_ths`, `moneyflow_ind_ths`, `moneyflow_ind_dc`,
  `moneyflow_mkt_dc`.
- ETF, index and fund research: `etf_share_size`, `idx_anns`,
  `ci_index_member`, `idx_factor_pro`, `daily_info`, `sz_daily_info`,
  `mkt_idx_bmk`, `fund_factor_pro`.
- Futures, gold and bonds: `fut_index_daily`, `fut_weekly_detail`, `sge_basic`,
  `sge_daily`, `cb_factor_pro`, `bc_otcqt`, `bc_bestotcqt`, `bond_blk`,
  `bond_blk_detail`, `eco_cal`.
- Overseas and rates: `hk_adjfactor`, `us_adjfactor`, `libor`, `hibor`,
  `us_trycr`, `us_tbr`, `us_tltr`, `us_trltr`.
- Research corpus: `npr`, `research_report`, `monetary_policy`, `cctv_news`,
  `irm_qa_sh`, `irm_qa_sz`, `wc_list`, `wc_cnt`.

### Strategy-specific optional coverage added locally (26)

`stk_nineturn`, `stk_ah_comparison`, `limit_list_ths`, `limit_step`,
`limit_cpt_list`, `ths_index`, `ths_daily`, `ths_member`, `dc_index`,
`dc_member`, `dc_daily`, `hm_list`, `hm_detail`, `ths_hot`, `dc_hot`,
`tdx_index`, `tdx_member`, `tdx_daily`, `kpl_list`, `kpl_concept_cons`,
`dc_concept`, `dc_concept_cons`, `sw_mins`, `hk_mins`, `wz_index`, `gz_index`.

## First priority capital-flow gap

The seven interfaces below are the first capital-flow subset of the 59-interface
default backlog. They are not the complete remaining gap.

| Interface | Required use | Downloader status |
| --- | --- | --- |
| `moneyflow_hsgt` | northbound/southbound aggregate capital flow | implemented locally; production pending |
| `moneyflow_ths` | THS security capital flow | implemented locally; production pending |
| `moneyflow_dc` | Eastmoney security capital flow | implemented locally; production pending |
| `moneyflow_cnt_ths` | THS market-wide capital-flow statistics | implemented locally; production pending |
| `moneyflow_ind_ths` | THS industry capital flow | implemented locally; production pending |
| `moneyflow_ind_dc` | Eastmoney industry capital flow | implemented locally; production pending |
| `moneyflow_mkt_dc` | Eastmoney market capital flow | implemented locally; production pending |

Existing `moneyflow`, `top_list`, `top_inst`, and `hk_hold` datasets do not
substitute for the interfaces above.

## Reference refresh and historical lineage

The 29 datasets identified by the production audit as lacking a reliable
snapshot time axis now have an explicit daily, weekly, or monthly refresh
policy. Full/as-of requests receive a version bucket in the work-unit identity;
dated request windows keep their natural identity. Work units are append-only,
old immutable snapshots remain unchanged, and a successor snapshot selects only
the latest successful version of each reference partition.

Every new snapshot records per-dataset `date_min`, `date_max`, source-unit hashes,
selected reference refresh buckets, and a `coverage_audit` section listing all
datasets with observations before 2024. This replaces an informal count with
traceable evidence; the previously observed 22-dataset count must be confirmed
again from the first post-deployment snapshot.

The six pre-existing institutional interfaces `report_rc`, `etf_sh_cons`,
`etf_sz_cons`, `ci_daily`, `shibor_quote`, and `major_news` retain separate
pagination and permission gates. ETF basket history is initialized as one full-range
partition per listed ETF instead of one full-market request per ETF/date; capped
partitions are bisected by date. `major_news` explicitly requests article
content in addition to title, source, and publication time. Their production
probe remains pending until the active production queue is idle.

## Reporting rule

Every future coverage report must state:

- the named target inventory or bundle;
- implemented count and explicit missing interfaces;
- production materialization range and failed/incomplete work units;
- snapshot and Qlib readiness separately.

If the official Tushare catalog has not been reconciled during the same review,
the report must say so and may not claim full-provider coverage.
