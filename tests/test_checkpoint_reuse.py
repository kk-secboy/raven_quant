from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from quant_data.checkpoint import CheckpointStore
from quant_data.cli import _reconcile_range_plan
from quant_data.execution_data import MINUTE_FIELDS, NEWS_FIELDS
from quant_data.models import FetchSpec, UnitResult
from quant_data.planner import BootstrapPlanner, ExecutionDataPlanner
from quant_data.supplemental_data import supplemental_specs


def _succeed(store: CheckpointStore, spec: FetchSpec, *, rows: int = 1) -> None:
    store.succeed(
        spec.unit_key,
        UnitResult(output_path=f"units/{spec.unit_key}.parquet", row_count=rows, sha256="abc"),
    )


def test_a_share_five_minute_reuses_legacy_success_and_supersedes_unfinished(
    database_url: str,
) -> None:
    store = CheckpointStore(database_url)
    succeeded = FetchSpec(
        "ashare_5m",
        "stk_mins",
        {
            "ts_code": "600000.SH",
            "start": "2024-01-02 00:00:00",
            "end": "2024-01-03 23:59:59",
            "freq": "5min",
        },
        {
            "ts_code": "600000.SH",
            "start_date": "2024-01-02 00:00:00",
            "end_date": "2024-01-03 23:59:59",
            "freq": "5min",
        },
        fields=MINUTE_FIELDS,
        allow_empty=True,
    )
    unfinished = FetchSpec(
        "ashare_5m",
        "stk_mins",
        {"ts_code": "600000.SH", "month": "202401"},
        {
            "ts_code": "600000.SH",
            "start_date": "2024-01-01 00:00:00",
            "end_date": "2024-01-31 23:59:59",
            "freq": "5min",
        },
        allow_empty=True,
    )
    store.add([succeeded, unfinished])
    _succeed(store, succeeded)

    specs = ExecutionDataPlanner(store).plan_minutes(
        {"ashare_5m": ["600000.SH"]},
        date(2024, 1, 2),
        date(2024, 1, 4),
        3,
        freq="5min",
        trading_dates=["20240102", "20240103", "20240104"],
    )

    assert succeeded.unit_key in {spec.unit_key for spec in specs}
    missing = [spec for spec in specs if spec.unit_key != succeeded.unit_key]
    assert len(missing) == 1
    assert missing[0].params["start_date"] == "2024-01-04 00:00:00"
    assert store.unit_rows([unfinished.unit_key])[0]["status"] == "superseded"


def test_one_minute_planner_reuses_legacy_exact_window(database_url: str) -> None:
    store = CheckpointStore(database_url)
    legacy = FetchSpec(
        "etf_1m",
        "etf_mins",
        {
            "ts_code": "510300.SH",
            "start": "2024-01-02 00:00:00",
            "end": "2024-01-31 23:59:59",
            "freq": "1min",
        },
        {
            "ts_code": "510300.SH",
            "start_date": "2024-01-02 00:00:00",
            "end_date": "2024-01-31 23:59:59",
            "freq": "1min",
        },
        fields=MINUTE_FIELDS,
        allow_empty=True,
    )
    store.add([legacy])
    _succeed(store, legacy)

    specs = ExecutionDataPlanner(store).plan_minutes(
        {"etf_1m": ["510300.SH"]},
        date(2024, 1, 2),
        date(2024, 1, 31),
        3,
    )

    assert [spec.unit_key for spec in specs] == [legacy.unit_key]


def test_news_reuses_one_legacy_half_and_plans_only_the_other(database_url: str) -> None:
    store = CheckpointStore(database_url)
    first_half = FetchSpec(
        "news",
        "news",
        {
            "date": "2024-01-02",
            "source": "sina",
            "start": "2024-01-02 00:00:00",
            "end": "2024-01-02 11:59:59",
            "row_limit": 1_500,
        },
        {
            "src": "sina",
            "start_date": "2024-01-02 00:00:00",
            "end_date": "2024-01-02 11:59:59",
        },
        fields=NEWS_FIELDS,
        allow_empty=True,
    )
    store.add([first_half])
    _succeed(store, first_half)

    specs = BootstrapPlanner(store, None).news_specs(
        date(2024, 1, 2), date(2024, 1, 2), 3
    )
    sina = [spec for spec in specs if spec.params["src"] == "sina"]

    assert len(sina) == 2
    assert first_half.unit_key in {spec.unit_key for spec in sina}
    missing = next(spec for spec in sina if spec.unit_key != first_half.unit_key)
    assert missing.params["start_date"] == "2024-01-02 12:00:00"
    assert missing.params["end_date"] == "2024-01-02 23:59:59"


def test_range_plan_reuses_complete_legacy_day_and_plans_only_gap(
    database_url: str,
) -> None:
    store = CheckpointStore(database_url)
    legacy = FetchSpec(
        "fund_share",
        "fund_share",
        {
            "trade_date": "20240102",
            "page_group": "fund_share:20240102",
            "page_size": 2_000,
            "offset": 0,
        },
        {"trade_date": "20240102", "limit": 2_000, "offset": 0},
        allow_empty=True,
    )
    store.add([legacy])
    _succeed(store, legacy, rows=0)
    target = next(
        spec
        for spec in supplemental_specs(
            "cn_funds",
            start=date(2024, 1, 2),
            end=date(2024, 1, 3),
            trading_dates=["20240102", "20240103"],
            max_attempts=3,
        )
        if spec.dataset == "fund_share"
    )

    reconciled = _reconcile_range_plan(
        SimpleNamespace(checkpoint=store), [target]
    )

    assert legacy.unit_key in {spec.unit_key for spec in reconciled}
    gap = next(spec for spec in reconciled if spec.unit_key != legacy.unit_key)
    assert gap.params["start_date"] == "20240103"
    assert gap.params["end_date"] == "20240103"
