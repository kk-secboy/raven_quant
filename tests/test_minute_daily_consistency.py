from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from quant_data.execution_contract import SIMULATION_MINUTE_SOURCE_DATASETS
from quant_data.verify import _verify_minute_daily_consistency

pytestmark = pytest.mark.no_database

SNAPSHOT_END = date(2024, 1, 31)


def _write_units(root: Path, dataset: str, rows: list[dict]) -> list[dict]:
    target = root / "units" / dataset
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target / "fixture.parquet", index=False)
    return [{"output_path": f"units/{dataset}/fixture.parquet"}]


def _daily_row(**overrides) -> dict:
    row = {
        "ts_code": "000001.SZ",
        "trade_date": "2024-01-02",
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        # 日线口径：vol 为手、amount 为千元。
        "vol": 100.0,
        "amount": 105.0,
    }
    row.update(overrides)
    return row


def _minute_rows(*, close: float = 10.5, vols=(4000, 3000, 3000)) -> list[dict]:
    # 分钟口径：vol 为股、amount 为 CNY。聚合后开=10.0、高=11.0、低=9.0、
    # 收=close、量=Σvol、额=105000 CNY = 105 千元。
    return [
        {
            "ts_code": "000001.SZ",
            "trade_time": "2024-01-02 09:35:00",
            "open": 10.0,
            "high": 10.8,
            "low": 9.9,
            "close": 10.6,
            "vol": vols[0],
            "amount": 40_000.0,
        },
        {
            "ts_code": "000001.SZ",
            "trade_time": "2024-01-02 10:05:00",
            "open": 10.6,
            "high": 11.0,
            "low": 10.2,
            "close": 10.3,
            "vol": vols[1],
            "amount": 30_000.0,
        },
        {
            "ts_code": "000001.SZ",
            "trade_time": "2024-01-02 14:55:00",
            "open": 10.3,
            "high": 10.6,
            "low": 9.0,
            "close": close,
            "vol": vols[2],
            "amount": 35_000.0,
        },
    ]


def _run(tmp_path: Path, daily: list[dict], minute: list[dict] | None, dataset="ashare_5m"):
    selected = {"daily": _write_units(tmp_path, "daily", daily)}
    if minute is not None:
        selected[dataset] = _write_units(tmp_path, dataset, minute)
    connection = duckdb.connect()
    try:
        return _verify_minute_daily_consistency(
            connection, selected, tmp_path, snapshot_end=SNAPSHOT_END
        )
    finally:
        connection.close()


def test_consistent_minute_aggregation_passes(tmp_path: Path) -> None:
    errors, warnings, checks = _run(tmp_path, [_daily_row()], _minute_rows())
    assert errors == []
    assert warnings == []
    assert checks["minute_daily_ashare_5m_compared_keys"] == 1
    assert checks["minute_daily_ashare_5m_mismatched_keys"] == 0
    assert checks["minute_daily_ashare_5m_daily_keys_without_minute_coverage"] == 0
    assert checks["minute_daily_ashare_5m_minute_keys_without_daily"] == 0


@pytest.mark.parametrize("dataset", sorted(SIMULATION_MINUTE_SOURCE_DATASETS))
def test_all_share_volume_minute_datasets_are_checked(tmp_path: Path, dataset: str) -> None:
    errors, _, checks = _run(tmp_path, [_daily_row()], _minute_rows(), dataset=dataset)
    assert errors == []
    assert checks[f"minute_daily_{dataset}_compared_keys"] == 1


def test_price_mismatch_is_an_error(tmp_path: Path) -> None:
    errors, _, checks = _run(tmp_path, [_daily_row()], _minute_rows(close=11.0))
    assert checks["minute_daily_ashare_5m_mismatched_keys"] == 1
    assert any("disagree" in error for error in errors)


def test_price_tolerance_boundary(tmp_path: Path) -> None:
    # 相对容差 1e-4：日内收盘 10.5，容差 0.00105。
    errors, _, checks = _run(tmp_path / "in", [_daily_row()], _minute_rows(close=10.5005))
    assert errors == []
    assert checks["minute_daily_ashare_5m_mismatched_keys"] == 0

    errors, _, checks = _run(tmp_path / "out", [_daily_row()], _minute_rows(close=10.5021))
    assert checks["minute_daily_ashare_5m_mismatched_keys"] == 1
    assert errors


def test_volume_amount_unit_conversion_and_mismatch(tmp_path: Path) -> None:
    # 量额按换算契约折算：分钟 10000 股=100 手、105000 CNY=105 千元（见一致案例）。
    # 量放大到 14000 股=140 手，相对偏差 40% 超容差。
    errors, _, checks = _run(
        tmp_path, [_daily_row()], _minute_rows(vols=(8000, 3000, 3000))
    )
    assert checks["minute_daily_ashare_5m_mismatched_keys"] == 1
    assert errors


def test_missing_minute_coverage_is_not_a_hard_failure(tmp_path: Path) -> None:
    daily = [
        _daily_row(),
        _daily_row(ts_code="000002.SZ"),  # 分钟是子集覆盖，该股票无分钟数据
    ]
    errors, warnings, checks = _run(tmp_path, daily, _minute_rows())
    assert errors == []
    assert checks["minute_daily_ashare_5m_compared_keys"] == 1
    assert checks["minute_daily_ashare_5m_daily_keys_without_minute_coverage"] == 1
    assert not any("000002" in warning for warning in warnings)


def test_minute_key_without_daily_row_is_counted_not_blocking(tmp_path: Path) -> None:
    minute = _minute_rows() + [
        {
            "ts_code": "000001.SZ",
            "trade_time": "2024-01-03 09:35:00",
            "open": 10.5,
            "high": 10.6,
            "low": 10.4,
            "close": 10.5,
            "vol": 100,
            "amount": 1050.0,
        }
    ]
    errors, _, checks = _run(tmp_path, [_daily_row()], minute)
    assert errors == []
    assert checks["minute_daily_ashare_5m_minute_keys_without_daily"] == 1


def test_no_minute_units_means_no_checks(tmp_path: Path) -> None:
    errors, warnings, checks = _run(tmp_path, [_daily_row()], None)
    assert errors == []
    assert warnings == []
    assert checks == {}
