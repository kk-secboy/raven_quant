from datetime import date
from pathlib import Path

from quant_data.catalog import CORE_DAILY, INDEX_CATALOG_MARKETS, RESEARCH_DAILY
from quant_data.checkpoint import CheckpointStore
from quant_data.models import FetchSpec, ProviderResult
from quant_data.planner import BootstrapPlanner, _month_ranges, _quarter_ranges, _report_periods
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


def test_complete_index_catalog_starts_one_paginated_partition_per_market(
    tmp_path: Path, database_url: str
) -> None:
    planner = BootstrapPlanner(CheckpointStore(database_url), ParquetStore(tmp_path))
    specs = planner.index_catalog_specs(5)
    assert [spec.params["market"] for spec in specs] == list(INDEX_CATALOG_MARKETS)
    assert all(spec.params["limit"] == 1_000 and spec.params["offset"] == 0 for spec in specs)
    assert all(spec.scope["page_group"].startswith("index_basic:") for spec in specs)
    assert all(spec.allow_empty for spec in specs)


def test_industry_members_use_supported_l3_current_and_historical_partitions(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    storage = ParquetStore(tmp_path)
    classify = FetchSpec(
        dataset="index_classify",
        api_name="index_classify",
        scope={"src": "SW2021"},
        params={"src": "SW2021"},
    )
    persist_reference(
        checkpoint,
        storage,
        classify,
        ["index_code", "level"],
        [
            {"index_code": "801010.SI", "level": "L1"},
            {"index_code": "850111.SI", "level": "L3"},
            {"index_code": "850112.SI", "level": "L3"},
        ],
    )
    planner = BootstrapPlanner(checkpoint, storage)

    assert planner.plan_industry_members(4, as_of=date(2026, 7, 29)) == 4
    specs = []
    while unit := checkpoint.claim({"index_member_all"}):
        specs.append(unit.spec)
        checkpoint.fail(unit.unit_key, "test inspection", terminal=True)

    assert {tuple(sorted(spec.params.items())) for spec in specs} == {
        (("is_new", "N"), ("l3_code", "850111.SI")),
        (("is_new", "Y"), ("l3_code", "850111.SI")),
        (("is_new", "N"), ("l3_code", "850112.SI")),
        (("is_new", "Y"), ("l3_code", "850112.SI")),
    }
    assert all(spec.scope["row_limit"] == 2_000 for spec in specs)
    assert all(spec.allow_empty for spec in specs)


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
