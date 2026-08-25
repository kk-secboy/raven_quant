from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from quant_data.checkpoint import CheckpointStore
from quant_data.cli import (
    _activate_stk_surv_plan,
    _rehydrate_durable_pagination_specs,
    _reconcile_range_plan,
    _reconcile_stk_surv_plan,
)
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


def test_a_share_five_minute_extensions_reuse_success_and_only_plan_suffix(
    database_url: str,
) -> None:
    store = CheckpointStore(database_url)
    planner = ExecutionDataPlanner(store)
    calendar_start = date(2024, 1, 2)
    calendar_end = date(2024, 12, 31)
    trading_dates = [
        value.strftime("%Y%m%d")
        for offset in range((calendar_end - calendar_start).days + 1)
        if (value := calendar_start + timedelta(days=offset)).weekday() < 5
    ]
    initial = planner.plan_minutes(
        {"ashare_5m": ["600000.SH"]},
        calendar_start,
        date(2024, 8, 20),
        3,
        freq="5min",
        trading_dates=trading_dates,
    )
    for spec in initial:
        _succeed(store, spec)

    within_quarter = planner.plan_minutes(
        {"ashare_5m": ["600000.SH"]},
        calendar_start,
        date(2024, 8, 30),
        3,
        freq="5min",
        trading_dates=trading_dates,
    )
    initial_keys = {spec.unit_key for spec in initial}
    within_quarter_keys = {spec.unit_key for spec in within_quarter}
    assert initial_keys <= within_quarter_keys
    within_quarter_suffix = [
        spec for spec in within_quarter if spec.unit_key not in initial_keys
    ]
    assert len(within_quarter_suffix) == 1
    assert within_quarter_suffix[0].params["start_date"][:10] == "2024-08-21"
    assert within_quarter_suffix[0].params["end_date"][:10] == "2024-08-30"
    for spec in within_quarter_suffix:
        _succeed(store, spec)

    cross_quarter = planner.plan_minutes(
        {"ashare_5m": ["600000.SH"]},
        calendar_start,
        calendar_end,
        3,
        freq="5min",
        trading_dates=trading_dates,
    )

    assert within_quarter_keys <= {spec.unit_key for spec in cross_quarter}
    new_specs = [
        spec for spec in cross_quarter if spec.unit_key not in within_quarter_keys
    ]
    assert len(new_specs) == 2
    assert new_specs[0].params["start_date"][:10] == "2024-09-02"
    assert new_specs[0].params["end_date"][:10] == "2024-09-30"
    assert new_specs[1].params["start_date"][:10] == "2024-10-01"
    assert new_specs[1].params["end_date"][:10] == "2024-12-31"
    for spec in within_quarter:
        assert store.unit_rows([spec.unit_key])[0]["status"] == "succeeded"
    assert all(
        sum(
            spec.params["start_date"][:10] <= session[:4] + "-" + session[4:6] + "-" + session[6:]
            <= spec.params["end_date"][:10]
            for session in trading_dates
        )
        <= 150
        for spec in cross_quarter
    )


def test_a_share_five_minute_fourteen_day_gap_is_one_request_and_next_day_is_suffix(
    database_url: str,
) -> None:
    store = CheckpointStore(database_url)
    planner = ExecutionDataPlanner(store)
    start = date(2026, 7, 1)
    initial_end = date(2026, 8, 3)
    gap_end = date(2026, 8, 21)
    next_end = date(2026, 8, 24)
    trading_dates = [
        value.strftime("%Y%m%d")
        for offset in range((next_end - start).days + 1)
        if (value := start + timedelta(days=offset)).weekday() < 5
    ]

    initial = planner.plan_minutes(
        {"ashare_5m": ["600000.SH"]},
        start,
        initial_end,
        3,
        freq="5min",
        trading_dates=trading_dates,
    )
    for spec in initial:
        _succeed(store, spec)
    initial_keys = {spec.unit_key for spec in initial}

    gap_plan = planner.plan_minutes(
        {"ashare_5m": ["600000.SH"]},
        start,
        gap_end,
        3,
        freq="5min",
        trading_dates=trading_dates,
    )
    gap_keys = {spec.unit_key for spec in gap_plan}
    assert initial_keys <= gap_keys
    gap_specs = [spec for spec in gap_plan if spec.unit_key not in initial_keys]
    assert len(gap_specs) == 1
    assert gap_specs[0].params["start_date"][:10] == "2026-08-04"
    assert gap_specs[0].params["end_date"][:10] == "2026-08-21"
    assert sum(
        "20260804" <= session <= "20260821" for session in trading_dates
    ) == 14
    for key in initial_keys:
        assert store.unit_rows([key])[0]["status"] == "succeeded"

    _succeed(store, gap_specs[0])
    extended = planner.plan_minutes(
        {"ashare_5m": ["600000.SH"]},
        start,
        next_end,
        3,
        freq="5min",
        trading_dates=trading_dates,
    )
    extended_keys = {spec.unit_key for spec in extended}
    assert gap_keys <= extended_keys
    next_specs = [spec for spec in extended if spec.unit_key not in gap_keys]
    assert len(next_specs) == 1
    assert next_specs[0].params["start_date"][:10] == "2026-08-24"
    assert next_specs[0].params["end_date"][:10] == "2026-08-24"


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


def test_pagination_restart_rehydrates_all_durable_siblings(database_url: str) -> None:
    store = CheckpointStore(database_url)
    base_scope = {
        "ann_date": "20240102",
        "page_group": "report:20240102",
        "page_size": 100,
    }
    first = FetchSpec(
        "report",
        "report",
        {**base_scope, "offset": 0},
        {"ann_date": "20240102", "limit": 100, "offset": 0},
        allow_empty=True,
        max_attempts=3,
    )
    second = FetchSpec(
        "report",
        "report",
        {**base_scope, "offset": 100},
        {"ann_date": "20240102", "limit": 100, "offset": 100},
        allow_empty=True,
        max_attempts=3,
    )
    unrelated = FetchSpec(
        "report",
        "report",
        {**base_scope, "page_group": "report:20240103", "offset": 0},
        {"ann_date": "20240103", "limit": 100, "offset": 0},
        allow_empty=True,
        max_attempts=3,
    )
    stale_generation = FetchSpec(
        "report",
        "report",
        {
            **base_scope,
            "offset": 0,
            "reference_refresh_bucket": "2023-12-25",
        },
        {"ann_date": "20240102", "limit": 100, "offset": 0},
        allow_empty=True,
        max_attempts=3,
    )
    store.add([first, second, unrelated, stale_generation])
    _succeed(store, first, rows=100)
    _succeed(store, second, rows=17)
    _succeed(store, unrelated, rows=1)
    _succeed(store, stale_generation, rows=9)

    restored = _rehydrate_durable_pagination_specs(
        SimpleNamespace(checkpoint=store), [first]
    )

    assert [spec.unit_key for spec in restored] == [first.unit_key, second.unit_key]
    durable_keys = {
        row["unit_key"]
        for row in store.pagination_group_units({("report", "report:20240102")})
    }
    assert durable_keys == {
        first.unit_key,
        second.unit_key,
        stale_generation.unit_key,
    }


@pytest.mark.no_database
def test_exact_range_restart_skips_full_historical_scan() -> None:
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

    class Checkpoint:
        @staticmethod
        def unit_rows(_unit_keys) -> list[dict]:
            return [{"unit_key": target.unit_key, "status": "succeeded"}]

        @staticmethod
        def successful(_dataset: str) -> list[dict]:
            raise AssertionError("unchanged range must not scan all historical rows")

    assert _reconcile_range_plan(
        SimpleNamespace(checkpoint=Checkpoint()), [target]
    ) == [target]


def test_stk_surv_migration_reuses_short_success_and_supersedes_legacy_failure(
    database_url: str,
) -> None:
    store = CheckpointStore(database_url)

    def legacy(day: str) -> FetchSpec:
        params = {"start_date": day, "end_date": day}
        return FetchSpec(
            "stk_surv",
            "stk_surv",
            {**params, "row_limit": 400},
            params,
            allow_empty=True,
            max_attempts=3,
        )

    short = legacy("20240102")
    capped = legacy("20240103")
    failed = legacy("20240104")
    outside = legacy("20240105")
    store.add([short, capped, failed, outside])
    _succeed(store, short, rows=399)
    _succeed(store, capped, rows=400)
    store.fail(failed.unit_key, "legacy row cap", terminal=True)
    store.fail(outside.unit_key, "outside requested range", terminal=True)

    current = [
        spec
        for spec in supplemental_specs(
            "cn_extended_daily",
            start=date(2024, 1, 2),
            end=date(2024, 1, 4),
            trading_dates=["20240102", "20240103", "20240104"],
            max_attempts=3,
        )
        if spec.dataset == "stk_surv"
    ]
    reconciled, _inserted = _activate_stk_surv_plan(
        SimpleNamespace(checkpoint=store), current
    )
    by_day = {spec.params["start_date"]: spec for spec in reconciled}

    assert by_day["20240102"].unit_key == short.unit_key
    assert by_day["20240103"].unit_key != capped.unit_key
    assert by_day["20240103"].params["offset"] == 0
    assert by_day["20240104"].unit_key != failed.unit_key
    assert store.unit_rows([failed.unit_key])[0]["status"] == "superseded"
    assert store.unit_rows([outside.unit_key])[0]["status"] == "failed"
    assert store.unit_rows([capped.unit_key])[0]["status"] == "succeeded"


@pytest.mark.no_database
def test_stk_surv_migration_does_not_retire_legacy_unit_when_add_crashes() -> None:
    day = "20240103"
    legacy = FetchSpec(
        "stk_surv",
        "stk_surv",
        {"start_date": day, "end_date": day, "row_limit": 400},
        {"start_date": day, "end_date": day},
        allow_empty=True,
        max_attempts=3,
    )
    current = next(
        spec
        for spec in supplemental_specs(
            "cn_extended_daily",
            start=date(2024, 1, 3),
            end=date(2024, 1, 3),
            trading_dates=[day],
            max_attempts=3,
        )
        if spec.dataset == "stk_surv"
    )

    class CrashingCheckpoint:
        superseded: list[str] = []

        @staticmethod
        def unit_rows(_keys) -> list[dict]:
            return []

        @staticmethod
        def successful(_dataset: str) -> list[dict]:
            return []

        @staticmethod
        def unfinished_units(_dataset: str) -> list[dict]:
            return [
                {
                    "unit_key": legacy.unit_key,
                    "dataset": legacy.dataset,
                    "api_name": legacy.api_name,
                    "scope_json": legacy.scope,
                    "params_json": legacy.params,
                    "fields_json": [],
                    "allow_empty": True,
                    "max_attempts": 3,
                }
            ]

        @staticmethod
        def add(_specs) -> int:
            raise RuntimeError("simulated checkpoint add crash")

        def supersede_units(self, unit_keys, _reason: str) -> int:
            self.superseded.extend(unit_keys)
            return len(self.superseded)

    checkpoint = CrashingCheckpoint()
    with pytest.raises(RuntimeError, match="add crash"):
        _activate_stk_surv_plan(
            SimpleNamespace(checkpoint=checkpoint), [current]
        )

    assert checkpoint.superseded == []


@pytest.mark.no_database
def test_stk_surv_migration_does_not_reuse_success_without_a_row_count() -> None:
    day = "20240103"
    legacy = FetchSpec(
        "stk_surv",
        "stk_surv",
        {"start_date": day, "end_date": day, "row_limit": 400},
        {"start_date": day, "end_date": day},
        allow_empty=True,
        max_attempts=3,
    )
    current = next(
        spec
        for spec in supplemental_specs(
            "cn_extended_daily",
            start=date(2024, 1, 3),
            end=date(2024, 1, 3),
            trading_dates=[day],
            max_attempts=3,
        )
        if spec.dataset == "stk_surv"
    )
    legacy_row = {
        "unit_key": legacy.unit_key,
        "dataset": legacy.dataset,
        "api_name": legacy.api_name,
        "scope_json": legacy.scope,
        "params_json": legacy.params,
        "fields_json": [],
        "allow_empty": True,
        "max_attempts": 3,
        "row_count": None,
    }

    class Checkpoint:
        @staticmethod
        def unit_rows(_keys) -> list[dict]:
            return []

        @staticmethod
        def successful(_dataset: str) -> list[dict]:
            return [legacy_row]

        @staticmethod
        def unfinished_units(_dataset: str) -> list[dict]:
            return []

    reconciled, obsolete = _reconcile_stk_surv_plan(
        SimpleNamespace(checkpoint=Checkpoint()), [current]
    )

    assert reconciled == [current]
    assert obsolete == {}
