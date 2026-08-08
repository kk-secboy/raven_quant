# Weekly structured information-factor refresh

`information_factor_refresh` is separate from the bounded daily NLP pipeline.
On its configured weekday it recomputes the selected deterministic factors from
their real source boundaries, registers content-addressed immutable artifacts,
and evaluates the exact registered hashes against one pinned daily Qlib dataset.

It never invents pre-source history:

- `report_rc` starts at 2010-01-01.
- `major_news_mentions` starts at 2018-11-20.
- `news_flash` starts at 2018-11-20.

```json
{
  "name": "weekly structured information factors",
  "kind": "information_factor_refresh",
  "timezone": "Asia/Shanghai",
  "run_time": "22:30:00",
  "trading_days_only": true,
  "payload": {
    "weekday": 4,
    "sources": ["report_rc", "major_news_mentions", "news_flash"],
    "factor_evaluation": {
      "dataset": "cn-full-20260803",
      "periods": {
        "train_start": "2010-01-01",
        "train_end": "2018-12-31",
        "valid_start": "2019-01-01",
        "valid_end": "2021-12-31",
        "test_start": "2022-01-10",
        "test_end": "2026-08-03"
      },
      "universe": "cn_all",
      "benchmark": "SH000300"
    }
  },
  "misfire_grace_seconds": 1800,
  "actor": "quantlab-operator"
}
```

The refresh is skipped while any download, snapshot, Qlib build, NLP, factor
registration, or information evaluation job is queued/running. A successful
evaluation updates research-candidate evidence only; it does not publish a
strategy, recommendation, or trade.
