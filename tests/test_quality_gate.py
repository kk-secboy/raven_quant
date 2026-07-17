from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from quant_data.cli import _require_snapshot_quality_gate
from quant_data.verify import _verify_daily_ohlc, quality_gate_payload

pytestmark = pytest.mark.no_database


def _write_units(
    root: Path, dataset: str, rows: list[dict], *, name: str = "fixture.parquet"
) -> list[dict]:
    target = root / "units" / dataset
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target / name, index=False)
    return [{"output_path": f"units/{dataset}/{name}"}]


def _daily_rows() -> list[dict]:
    return [
        {
            "ts_code": "000001.SZ",
            "trade_date": "2024-01-02",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "vol": 100.0,
            "amount": 105.0,
            "pct_chg": 2.0,
        },
        {
            "ts_code": "000001.SZ",
            "trade_date": "2024-01-03",
            "open": 10.5,
            "high": 11.5,
            "low": 10.0,
            "close": 11.0,
            "vol": 120.0,
            "amount": 126.0,
            "pct_chg": 4.76,
        },
    ]


def _adj_rows() -> list[dict]:
    return [
        {"ts_code": "000001.SZ", "trade_date": "2024-01-02", "adj_factor": 1.0},
        {"ts_code": "000001.SZ", "trade_date": "2024-01-03", "adj_factor": 1.0},
    ]


def _run_ohlc(tmp_path: Path, daily: list[dict], adj: list[dict] | None):
    selected = {"daily": _write_units(tmp_path, "daily", daily)}
    if adj is not None:
        selected["adj_factor"] = _write_units(tmp_path, "adj_factor", adj)
    connection = duckdb.connect()
    try:
        return _verify_daily_ohlc(
            connection, selected, tmp_path, snapshot_end=date(2024, 1, 31)
        )
    finally:
        connection.close()


def test_ohlc_checks_pass_on_consistent_rows(tmp_path: Path) -> None:
    errors, warnings, checks = _run_ohlc(tmp_path, _daily_rows(), _adj_rows())
    assert errors == []
    assert warnings == []
    assert checks["daily_ohlc_rows"] == 2
    assert checks["daily_missing_adj_factor_keys"] == 0


def test_ohlc_relationship_violations_are_errors(tmp_path: Path) -> None:
    daily = _daily_rows()
    daily[0]["close"] = -1.0  # non-positive price
    daily[1]["high"] = 5.0  # high below low
    daily.append(
        {
            "ts_code": "000002.SZ",
            "trade_date": "2024-01-03",
            "open": 20.0,  # open above high
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
            "vol": 50.0,
            "amount": 50.0,
            "pct_chg": 1.0,
        }
    )
    errors, _, checks = _run_ohlc(tmp_path, daily, _adj_rows())
    assert checks["daily_nonpositive_price_rows"] == 1
    assert checks["daily_high_below_low_rows"] == 1
    # all three corrupted rows also have open or close outside [low, high]
    assert checks["daily_open_close_outside_range_rows"] == 3
    assert any("non-positive" in error for error in errors)
    assert any("high below low" in error for error in errors)
    assert any("outside the [low, high] range" in error for error in errors)


def test_missing_adj_factor_is_an_error(tmp_path: Path) -> None:
    errors, _, checks = _run_ohlc(tmp_path, _daily_rows(), _adj_rows()[:1])
    assert checks["daily_missing_adj_factor_keys"] == 1
    assert any("no adjustment factor" in error for error in errors)

    errors, _, checks = _run_ohlc(tmp_path, _daily_rows(), None)
    assert checks["daily_missing_adj_factor_keys"] == 2
    assert any("no adjustment factor" in error for error in errors)


def test_large_pct_chg_is_a_warning_not_an_error(tmp_path: Path) -> None:
    daily = _daily_rows()
    daily[0]["pct_chg"] = 40.0
    errors, warnings, checks = _run_ohlc(tmp_path, daily, _adj_rows())
    assert checks["daily_large_pct_chg_rows"] == 1
    assert errors == []
    assert any("35%" in warning for warning in warnings)


def test_quality_gate_payload_captures_report_state() -> None:
    payload = quality_gate_payload(
        {"ok": False, "checked_at": "2026-07-17T00:00:00+00:00", "errors": ["boom"]}
    )
    assert payload == {
        "ok": False,
        "verified_at": "2026-07-17T00:00:00+00:00",
        "errors": ["boom"],
    }


def _snapshot_with_manifest(tmp_path: Path, manifest: dict) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return snapshot


def test_build_refuses_snapshot_without_passing_quality_gate(tmp_path: Path) -> None:
    missing = _snapshot_with_manifest(tmp_path, {"name": "snapshot"})
    with pytest.raises(ValueError, match="quality gate"):
        _require_snapshot_quality_gate(missing)

    failed = _snapshot_with_manifest(
        tmp_path / "other", {"name": "other", "quality_gate": {"ok": False, "errors": ["x"]}}
    )
    with pytest.raises(ValueError, match="quality gate"):
        _require_snapshot_quality_gate(failed)

    passed = _snapshot_with_manifest(
        tmp_path / "good",
        {
            "name": "good",
            "quality_gate": {"ok": True, "verified_at": "2026-07-17T00:00:00+00:00", "errors": []},
        },
    )
    _require_snapshot_quality_gate(passed)


def test_skip_quality_gate_overrides_the_refusal(tmp_path: Path) -> None:
    missing = _snapshot_with_manifest(tmp_path, {"name": "snapshot"})
    _require_snapshot_quality_gate(missing, skip=True)
