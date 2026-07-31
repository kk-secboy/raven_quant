# Pre-2016 market-history policy

The configured Tushare-compatible production gateway rejects requests before
2016-01-01. QuantLab therefore uses a second, explicitly labelled source only
for the market history needed to extend stress tests through 2008–2015.

## Admitted data

BaoStock 0.9.3 is pinned and may populate only:

- `trade_cal`, including closed dates and the previous open date;
- unadjusted `daily` OHLC, previous close, return, volume, and amount;
- the subset of `daily_basic` that BaoStock publishes (`close`, turnover,
  `pe_ttm`, `pb`, and `ps_ttm`);
- `adj_factor`, derived as BaoStock back-adjusted close divided by raw close.

BaoStock volume is converted from shares to Tushare's 100-share lots. Amount
is converted from yuan to thousand yuan. Raw upstream rows are retained in the
normal immutable raw store.

No pre-2016 fundamentals, corporate events, news, share counts, market values,
dividend yields, volume ratios, or free-float turnover are invented. Those
columns remain unavailable when the source does not publish them.

## Admission gate

Production import is refused until `validate-baostock-overlap` passes on the
full 2016 calendar year for the audited ten-stock sample. The report compares:

- common session count per instrument;
- raw OHLC and previous close;
- percentage return;
- volume and amount after unit conversion;
- the *relative path* of the adjustment factor (source levels may use different
  bases).

The gate uses both percentile and maximum-error bounds and writes the full
per-symbol evidence to JSON. `bootstrap-legacy-market` requires that report and
also refuses any import ending on or after 2016-01-01, so the two production
sources cannot overlap silently.

## Durable jobs

The API exposes two sequential background jobs:

1. `POST /api/jobs/baostock-overlap-validation`
2. `POST /api/jobs/legacy-market-backfill`

The second endpoint remains unavailable until the first report passes. Both
jobs are PostgreSQL-backed and visible in the normal job history. The legacy
runner claims only BaoStock API contracts for the planned datasets, preventing
it from executing dormant Tushare units with the BaoStock adapter.

After the legacy job completes, the ordinary full quality gate, immutable
snapshot, Qlib build, and long-horizon formal validation must still run. Passing
the source-overlap gate is necessary data admission evidence; it is not proof
that any factor or strategy is profitable.
