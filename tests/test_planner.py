from datetime import date
from pathlib import Path

from quant_data.catalog import CORE_DAILY, INDEX_CATALOG_MARKETS, RESEARCH_DAILY
from quant_data.checkpoint import CheckpointStore
from quant_data.models import FetchSpec, ProviderResult
from quant_data.planner import (
    PIT_CALENDAR_FORWARD_BUFFER_DAYS,
    BootstrapPlanner,
    _month_ranges,
    _quarter_ranges,
    _report_periods,
)
from quant_data.reference_data import INDEX_MEMBER_ALL_WEEKLY_COHORT
from quant_data.storage import ParquetStore


def persist_reference(
    checkpoint: CheckpointStore,
    storage: ParquetStore,
    spec: FetchSpec,
    columns: list[str],
    rows: list[dict[str, object]],
) -> None:
    checkpoint.add([spec])
    result = storage.write_unit(
        spec.dataset,
        spec.unit_key,
        ProviderResult(spec.api_name, columns, rows, b"{}"),
    )
    checkpoint.succeed(spec.unit_key, result)


def test_plans_full_market_calls_by_trade_date(tmp_path: Path, database_url: str) -> None:
    checkpoint = CheckpointStore(database_url)
    storage = ParquetStore(tmp_path)
    trade_cal = FetchSpec(
        dataset="trade_cal",
        api_name="trade_cal",
        scope={"range": "test"},
        params={},
    )
    persist_reference(
        checkpoint,
        storage,
        trade_cal,
        ["exchange", "cal_date", "is_open"],
        [
            {"exchange": "SSE", "cal_date": "20240102", "is_open": 1},
            {"exchange": "SSE", "cal_date": "20240103", "is_open": 1},
            {"exchange": "SSE", "cal_date": "20240104", "is_open": 0},
        ],
    )
    planner = BootstrapPlanner(checkpoint, storage)
    dates = planner.trading_dates(date(2024, 1, 1), date(2024, 1, 4))
    assert dates == ["20240102", "20240103"]
    assert planner.plan_daily(dates, CORE_DAILY, 5) == len(CORE_DAILY) * 2

    rows = [row["params_json"] for row in checkpoint.successful("daily")]
    assert rows == []
    planned = []
    while unit := checkpoint.claim({"daily"}):
        planned.append(unit.spec.params)
        checkpoint.fail(unit.unit_key, "test inspection", terminal=True)
    assert planned == [{"trade_date": "20240102"}, {"trade_date": "20240103"}]
    assert all("ts_code" not in row for row in planned)

    for dataset, row_limit in (
        ("stk_premarket", 8_000),
        ("stk_auction_o", 10_000),
        ("stk_auction_c", 10_000),
    ):
        units = []
        while unit := checkpoint.claim({dataset}):
            units.append(unit.spec)
            checkpoint.fail(unit.unit_key, "test inspection", terminal=True)
        assert sorted((unit.params for unit in units), key=lambda item: item["trade_date"]) == [
            {"trade_date": "20240102"},
            {"trade_date": "20240103"},
        ]
        assert {unit.scope["row_limit"] for unit in units} == {row_limit}
        assert {unit.scope["expected_date_field"] for unit in units} == {"trade_date"}

    planner.plan_index_context(date(2024, 1, 1), date(2024, 1, 4), 5)
    index_codes = []
    while unit := checkpoint.claim({"index_daily"}):
        index_codes.append(unit.spec.params["ts_code"])
        checkpoint.fail(unit.unit_key, "test inspection", terminal=True)
    assert sorted(index_codes) == [
        "000001.SH",
        "000016.SH",
        "000300.SH",
        "000688.SH",
        "000852.SH",
        "000905.SH",
        "399001.SZ",
        "399006.SZ",
        "899050.BJ",
    ]

    daily_basic = []
    while unit := checkpoint.claim({"index_dailybasic"}):
        daily_basic.append(unit.spec.params)
        checkpoint.fail(unit.unit_key, "test inspection", terminal=True)
    assert sorted(daily_basic, key=lambda item: item["trade_date"]) == [
        {"trade_date": "20240102"},
        {"trade_date": "20240103"},
    ]


def test_reference_calendar_extends_only_the_pit_schedule_horizon(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    planner = BootstrapPlanner(checkpoint, ParquetStore(tmp_path))
    publication_end = date(2026, 8, 21)

    planner.plan_reference(date(2026, 8, 1), publication_end, 5)

    rows = checkpoint.dataset_units("trade_cal")
    assert len(rows) == 1
    spec = rows[0]
    assert spec["params_json"] == {
        "exchange": "SSE",
        "start_date": "20260801",
        "end_date": "20260921",
    }
    assert spec["scope_json"] == {
        "exchange": "SSE",
        "start": "20260801",
        "end": "20260921",
        "publication_end": "20260821",
        "purpose": "pit_availability_horizon",
    }
    assert PIT_CALENDAR_FORWARD_BUFFER_DAYS == 31


def test_daily_planning_skips_dates_before_documented_provider_history(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    planner = BootstrapPlanner(checkpoint, ParquetStore(tmp_path))
    definitions = (
        next(item for item in RESEARCH_DAILY if item.name == "moneyflow"),
        next(item for item in RESEARCH_DAILY if item.name == "margin_detail"),
        next(item for item in CORE_DAILY if item.name == "limit_list_d"),
    )

    assert planner.plan_daily(
        ["20091231", "20100104", "20191231", "20200102"],
        definitions,
        3,
    ) == 10

    planned: dict[str, list[str]] = {}
    for dataset in ("moneyflow", "margin_detail", "limit_list_d"):
        planned[dataset] = []
        while unit := checkpoint.claim({dataset}):
            planned[dataset].append(str(unit.spec.params["trade_date"]))
            checkpoint.fail(unit.unit_key, "test inspection", terminal=True)

    assert {dataset: sorted(values) for dataset, values in planned.items()} == {
        "moneyflow": ["20100104", "20191231", "20200102"],
        "margin_detail": ["20100104", "20191231", "20200102"],
        "limit_list_d": ["20091231", "20100104", "20191231", "20200102"],
    }


def test_primary_market_plan_does_not_overlap_baostock_legacy_history(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    planner = BootstrapPlanner(checkpoint, ParquetStore(tmp_path))
    definitions = tuple(
        item
        for item in CORE_DAILY
        if item.name in {"daily", "daily_basic", "adj_factor"}
    )

    assert planner.plan_daily(["20151231", "20160104"], definitions, 3) == 3

    for dataset in ("daily", "daily_basic", "adj_factor"):
        unit = checkpoint.claim({dataset})
        assert unit is not None
        assert unit.spec.params == {"trade_date": "20160104"}
        checkpoint.fail(unit.unit_key, "test inspection", terminal=True)
        assert checkpoint.claim({dataset}) is None


def test_complete_index_catalog_starts_one_paginated_partition_per_market(
    tmp_path: Path, database_url: str
) -> None:
    planner = BootstrapPlanner(CheckpointStore(database_url), ParquetStore(tmp_path))
    specs = planner.index_catalog_specs(5)
    assert [spec.params["market"] for spec in specs] == list(INDEX_CATALOG_MARKETS)
    assert all(spec.params["limit"] == 1_000 and spec.params["offset"] == 0 for spec in specs)
    assert all(spec.scope["page_group"].startswith("index_basic:") for spec in specs)
    assert all(spec.allow_empty for spec in specs)


def test_index_context_respects_downloaded_index_inception_dates(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    storage = ParquetStore(tmp_path)
    index_basic = FetchSpec(
        dataset="index_basic",
        api_name="index_basic",
        scope={"market": "SSE"},
        params={"market": "SSE"},
    )
    persist_reference(
        checkpoint,
        storage,
        index_basic,
        ["ts_code", "list_date"],
        [
            {"ts_code": "000300.SH", "list_date": "20050408"},
            {"ts_code": "000688.SH", "list_date": "20200123"},
            {"ts_code": "899050.BJ", "list_date": "20221121"},
        ],
    )
    trade_cal = FetchSpec(
        dataset="trade_cal",
        api_name="trade_cal",
        scope={"range": "test"},
        params={},
    )
    persist_reference(
        checkpoint,
        storage,
        trade_cal,
        ["exchange", "cal_date", "is_open"],
        [{"exchange": "SSE", "cal_date": "20160104", "is_open": 1}],
    )
    planner = BootstrapPlanner(checkpoint, storage)

    planner.plan_index_context(date(2016, 1, 1), date(2017, 12, 31), 5)
    planned: dict[str, tuple[str, str]] = {}
    while unit := checkpoint.claim({"index_daily"}):
        planned[str(unit.spec.params["ts_code"])] = (
            str(unit.spec.params["start_date"]),
            str(unit.spec.params["end_date"]),
        )
        checkpoint.fail(unit.unit_key, "test inspection", terminal=True)

    assert "000688.SH" not in planned
    assert "899050.BJ" not in planned
    assert planned["000300.SH"] == ("20160101", "20171231")


def test_research_reference_skips_unsupported_pre_2016_disclosure_periods(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    planner = BootstrapPlanner(checkpoint, ParquetStore(tmp_path))

    planner.plan_research_reference(date(2016, 1, 1), date(2017, 12, 31), 5)
    periods: list[str] = []
    while unit := checkpoint.claim({"disclosure_date"}):
        periods.append(str(unit.spec.params["end_date"]))
        checkpoint.fail(unit.unit_key, "test inspection", terminal=True)

    assert sorted(periods) == [
        "20160331",
        "20160630",
        "20160930",
        "20161231",
        "20170331",
        "20170630",
        "20170930",
        "20171231",
    ]
    classify = checkpoint.dataset_units("index_classify")
    assert {
        (str(row["params_json"]["src"]), str(row["params_json"]["level"]))
        for row in classify
    } == {
        (source, level)
        for source in ("SW2014", "SW2021")
        for level in ("L1", "L2", "L3")
    }
    fund_basic = checkpoint.dataset_units("fund_basic")
    assert {
        (str(row["params_json"]["market"]), str(row["params_json"]["status"]))
        for row in fund_basic
    } == {("E", status) for status in ("L", "I", "D")}


def test_industry_members_use_supported_l3_current_and_historical_partitions(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    storage = ParquetStore(tmp_path)
    for source, rows in (
        (
            "SW2014",
            [
                {"index_code": "850111.SI", "level": "L3"},
                {"index_code": "850412.SI", "level": "L3"},
            ],
        ),
        (
            "SW2021",
            [
                {"index_code": "801010.SI", "level": "L1"},
                {"index_code": "850111.SI", "level": "L3"},
                {"index_code": "850112.SI", "level": "L3"},
            ],
        ),
    ):
        classify = FetchSpec(
            dataset="index_classify",
            api_name="index_classify",
            scope={"src": source},
            params={"src": source},
        )
        persist_reference(
            checkpoint,
            storage,
            classify,
            ["index_code", "level"],
            rows,
        )
    planner = BootstrapPlanner(checkpoint, storage)

    assert planner.plan_industry_members(4, as_of=date(2026, 7, 29)) == 6
    specs = []
    while unit := checkpoint.claim({"index_member_all"}):
        specs.append(unit.spec)
        checkpoint.fail(unit.unit_key, "test inspection", terminal=True)

    assert {tuple(sorted(spec.params.items())) for spec in specs} == {
        (("is_new", "N"), ("l3_code", "850111.SI")),
        (("is_new", "Y"), ("l3_code", "850111.SI")),
        (("is_new", "N"), ("l3_code", "850112.SI")),
        (("is_new", "Y"), ("l3_code", "850112.SI")),
        (("is_new", "N"), ("l3_code", "850412.SI")),
        (("is_new", "Y"), ("l3_code", "850412.SI")),
    }
    assert all(spec.scope["row_limit"] == 2_000 for spec in specs)
    assert all(spec.scope["reference_refresh_bucket"] == "2026-07-27" for spec in specs)
    assert all(spec.scope["reference_refresh_cadence"] == "weekly" for spec in specs)
    assert all(
        spec.scope["membership_cohort"] == INDEX_MEMBER_ALL_WEEKLY_COHORT
        for spec in specs
    )
    assert all(spec.allow_empty for spec in specs)


def test_plans_idempotent_ts_code_repairs_for_benchmark_pit_residuals(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    storage = ParquetStore(tmp_path)
    weights = FetchSpec(
        dataset="index_weight",
        api_name="index_weight",
        scope={"index_code": "000300.SH", "month": "2024-01"},
        params={"index_code": "000300.SH"},
    )
    persist_reference(
        checkpoint,
        storage,
        weights,
        ["index_code", "con_code", "trade_date", "weight"],
        [
            {
                "index_code": "000300.SH",
                "con_code": "000001.SZ",
                "trade_date": "20240131",
                "weight": 50.0,
            },
            {
                "index_code": "000300.SH",
                "con_code": "000002.SZ",
                "trade_date": "20240131",
                "weight": 30.0,
            },
            {
                "index_code": "000300.SH",
                "con_code": "000003.SZ",
                "trade_date": "20240131",
                "weight": 20.0,
            },
        ],
    )
    members = FetchSpec(
        dataset="index_member_all",
        api_name="index_member_all",
        scope={
            "l3_code": "850111.SI",
            "is_new": "Y",
            "row_limit": 2_000,
            "membership_cohort": INDEX_MEMBER_ALL_WEEKLY_COHORT,
            "reference_refresh_bucket": "2024-01-29",
            "reference_refresh_cadence": "weekly",
        },
        params={"l3_code": "850111.SI", "is_new": "Y"},
        allow_empty=True,
    )
    persist_reference(
        checkpoint,
        storage,
        members,
        ["ts_code", "l1_code", "in_date", "out_date"],
        [
            {
                "ts_code": "000001.SZ",
                "l1_code": "801010.SI",
                "in_date": "20200101",
                "out_date": None,
            },
            {
                "ts_code": "000002.SZ",
                "l1_code": "801020.SI",
                "in_date": "20240201",
                "out_date": None,
            },
        ],
    )
    planner = BootstrapPlanner(checkpoint, storage)

    assert planner.benchmark_industry_residual_symbols(
        date(2024, 1, 1), date(2024, 1, 31)
    ) == ["000002.SZ", "000003.SZ"]
    assert planner.plan_benchmark_industry_residual_members(
        date(2024, 1, 1),
        date(2024, 1, 31),
        4,
        as_of=date(2024, 1, 31),
    ) == 4
    assert planner.plan_benchmark_industry_residual_members(
        date(2024, 1, 1),
        date(2024, 1, 31),
        4,
        as_of=date(2024, 1, 31),
    ) == 0

    specs = []
    while unit := checkpoint.claim({"index_member_all"}):
        if unit.spec.params.get("ts_code"):
            specs.append(unit.spec)
        checkpoint.fail(unit.unit_key, "test inspection", terminal=True)
    assert {tuple(sorted(spec.params.items())) for spec in specs} == {
        (("is_new", "N"), ("ts_code", "000002.SZ")),
        (("is_new", "Y"), ("ts_code", "000002.SZ")),
        (("is_new", "N"), ("ts_code", "000003.SZ")),
        (("is_new", "Y"), ("ts_code", "000003.SZ")),
    }
    assert all(spec.scope["residual_benchmark"] == "000300.SH" for spec in specs)
    assert all(spec.scope["reference_refresh_bucket"] == "2024-01-29" for spec in specs)
    assert all(
        spec.scope["membership_cohort"] == INDEX_MEMBER_ALL_WEEKLY_COHORT
        for spec in specs
    )

    next_week = FetchSpec(
        dataset="index_member_all",
        api_name="index_member_all",
        scope={
            "l3_code": "850111.SI",
            "is_new": "Y",
            "row_limit": 2_000,
            "membership_cohort": INDEX_MEMBER_ALL_WEEKLY_COHORT,
            "reference_refresh_bucket": "2024-02-05",
            "reference_refresh_cadence": "weekly",
        },
        params={"l3_code": "850111.SI", "is_new": "Y"},
        allow_empty=True,
    )
    persist_reference(
        checkpoint,
        storage,
        next_week,
        ["ts_code", "l1_code", "in_date", "out_date"],
        [
            {
                "ts_code": symbol,
                "l1_code": "801010.SI",
                "in_date": "20200101",
                "out_date": None,
            }
            for symbol in ("000001.SZ", "000002.SZ", "000003.SZ")
        ],
    )
    assert planner.benchmark_industry_residual_symbols(
        date(2024, 1, 1), date(2024, 2, 7)
    ) == []
    assert planner.plan_benchmark_industry_residual_members(
        date(2024, 1, 1),
        date(2024, 2, 7),
        4,
        as_of=date(2024, 2, 7),
    ) == 4
    refreshed = [
        row
        for row in checkpoint.dataset_units("index_member_all")
        if dict(row["scope_json"]).get("reference_refresh_bucket") == "2024-02-05"
        and dict(row["params_json"]).get("ts_code")
    ]
    assert {
        (row["params_json"]["ts_code"], row["params_json"]["is_new"])
        for row in refreshed
    } == {
        (symbol, status)
        for symbol in ("000002.SZ", "000003.SZ")
        for status in ("Y", "N")
    }


def test_quarter_ranges_clip_to_requested_window() -> None:
    assert _quarter_ranges(date(2024, 2, 10), date(2024, 8, 5)) == [
        (date(2024, 2, 10), date(2024, 3, 31)),
        (date(2024, 4, 1), date(2024, 6, 30)),
        (date(2024, 7, 1), date(2024, 8, 5)),
    ]


def test_month_ranges_clip_to_requested_window() -> None:
    assert _month_ranges(date(2024, 2, 10), date(2024, 4, 5)) == [
        (date(2024, 2, 10), date(2024, 2, 29)),
        (date(2024, 3, 1), date(2024, 3, 31)),
        (date(2024, 4, 1), date(2024, 4, 5)),
    ]


def test_report_periods_include_prior_annual_report() -> None:
    periods = _report_periods(date(2024, 1, 1), date(2024, 7, 1))
    assert periods == ["20230331", "20230630", "20230930", "20231231", "20240331", "20240630"]
