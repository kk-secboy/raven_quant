from datetime import date
from pathlib import Path

from quant_data.catalog import CORE_DAILY
from quant_data.checkpoint import CheckpointStore
from quant_data.models import FetchSpec, ProviderResult
from quant_data.planner import BootstrapPlanner, _quarter_ranges, _report_periods
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

    planner.plan_index_context(date(2024, 1, 1), date(2024, 1, 4), 5)
    index_codes = []
    while unit := checkpoint.claim({"index_daily"}):
        index_codes.append(unit.spec.params["ts_code"])
        checkpoint.fail(unit.unit_key, "test inspection", terminal=True)
    assert sorted(index_codes) == ["000016.SH", "000300.SH", "000852.SH", "000905.SH"]


def test_quarter_ranges_clip_to_requested_window() -> None:
    assert _quarter_ranges(date(2024, 2, 10), date(2024, 8, 5)) == [
        (date(2024, 2, 10), date(2024, 3, 31)),
        (date(2024, 4, 1), date(2024, 6, 30)),
        (date(2024, 7, 1), date(2024, 8, 5)),
    ]


def test_report_periods_include_prior_annual_report() -> None:
    periods = _report_periods(date(2024, 1, 1), date(2024, 7, 1))
    assert periods == ["20230331", "20230630", "20230930", "20231231", "20240331", "20240630"]
