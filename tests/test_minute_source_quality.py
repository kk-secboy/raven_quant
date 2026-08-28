from __future__ import annotations

import hashlib
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_data.cli import _explicit_execution_quality_gate
from quant_data.execution_contract import MINUTE_CANONICALIZATION_POLICY_VERSION
from quant_data.verify import verify_ashare_5m_source_files

pytestmark = pytest.mark.no_database


def _session_endpoints(day: date) -> list[datetime]:
    values: list[datetime] = [datetime.combine(day, time(9, 30))]
    for start, end in ((time(9, 35), time(11, 30)), (time(13, 5), time(15, 0))):
        current = datetime.combine(day, start)
        boundary = datetime.combine(day, end)
        while current <= boundary:
            values.append(current)
            current += timedelta(minutes=5)
    assert len(values) == 49
    return values


def _rows(*, conflicting_duplicate: bool = False) -> list[dict[str, object]]:
    rows = [
        {
            "ts_code": "600000.SH",
            "trade_time": stamp,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "vol": 1_000.0,
            "amount": 10_000.0,
        }
        for stamp in _session_endpoints(date(2024, 10, 30))
    ]
    # The provider occasionally reports a one-share fill as raw vol=100.  The
    # direct implied price is wrong while the governed /100 interpretation is exact.
    rows[0]["vol"] = 100.0
    rows[0]["amount"] = 10.0
    # A canonical timestamp with irreconcilable amount is retained as a price
    # mark but must be made non-tradable by the builder.
    rows[1]["amount"] = 0.0
    # Accepted content fallbacks never publish an implied VWAP outside OHLC;
    # the builder uses close and records the exact fallback category instead.
    rows[2]["amount"] = 10_400.0  # within the relative 5% band only
    rows[3]["vol"] = 1.0
    rows[3]["amount"] = 11.0  # exactly one CNY outside the OHLC amount envelope
    for field in ("open", "high", "low", "close"):
        rows[4][field] = 0.01
    rows[4]["amount"] = 19.0  # implied 0.019: price-tick fallback only
    # Known raw provider additions: never aggregate either into the existing
    # canonical 5-minute bar.
    rows.extend(
        [
            {**rows[2], "trade_time": datetime(2024, 10, 30, 9, 36)},
            {**rows[3], "trade_time": datetime(2024, 10, 30, 15, 30)},
        ]
    )
    if conflicting_duplicate:
        rows.append(
            {
                **rows[4],
                "high": 10.2,
                "close": 10.1,
            }
        )
    return rows


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_ashare_5m_audit_records_exclusions_normalization_and_nontradable(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path / "ashare.parquet", _rows())

    errors, warnings, audit = verify_ashare_5m_source_files(
        [path],
        snapshot_start=date(2024, 10, 30),
        snapshot_end=date(2024, 10, 30),
    )
    repeated = verify_ashare_5m_source_files(
        [path],
        snapshot_start=date(2024, 10, 30),
        snapshot_end=date(2024, 10, 30),
    )[2]

    assert errors == []
    assert audit["policy"]["version"] == MINUTE_CANONICALIZATION_POLICY_VERSION
    assert audit["source_rows"] == 51
    assert audit["canonical_rows"] == 49
    assert audit["outside_session_rows"] == 1
    assert audit["off_cadence_rows"] == 1
    assert audit["excluded_rows"] == 2
    assert audit["hand_normalized_rows"] == 1
    assert audit["volume_normalized_rows"] == 1
    assert audit["relative_fallback_rows"] == 1
    assert audit["amount_rounding_fallback_rows"] == 1
    assert audit["price_tick_fallback_rows"] == 1
    assert audit["nontradable_rows"] == 1
    assert audit["nontradable_symbol_days"] == 1
    assert audit["incomplete_symbol_days"] == 0
    assert audit["audit_status"] == "pass_with_canonicalization"
    assert len(audit["audit_sha256"]) == 64
    assert len(audit["event_sha256"]) == 64
    assert len(audit["summary_sha256"]) == 64
    assert repeated["audit_sha256"] == audit["audit_sha256"]
    assert any("deterministic session/cadence" in warning for warning in warnings)
    assert any("non-tradable" in warning for warning in warnings)


def test_ashare_5m_audit_blocks_conflicting_canonical_timestamp(tmp_path: Path) -> None:
    path = _write(tmp_path / "conflict.parquet", _rows(conflicting_duplicate=True))

    errors, _, audit = verify_ashare_5m_source_files(
        [path],
        snapshot_start=date(2024, 10, 30),
        snapshot_end=date(2024, 10, 30),
    )

    assert audit["conflicting_canonical_keys"] == 1
    assert audit["audit_status"] == "block"
    assert any("conflicting bar content" in error for error in errors)


def test_ashare_5m_audit_blocks_missing_contract_columns(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"
    pd.DataFrame([{"ts_code": "600000.SH", "trade_time": "2024-10-30 09:35:00"}]).to_parquet(
        path, index=False
    )

    errors, _, audit = verify_ashare_5m_source_files(
        [path],
        snapshot_start=date(2024, 10, 30),
        snapshot_end=date(2024, 10, 30),
    )

    assert audit["audit_status"] == "block"
    assert "amount" in audit["missing_columns"]
    assert errors


def test_scoped_execution_gate_seals_minute_source_audit(tmp_path: Path) -> None:
    relative = Path("units") / "ashare_5m" / "unit.parquet"
    path = _write(tmp_path / relative, _rows())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row = {
        "unit_key": "ashare-5m-20241030",
        "dataset": "ashare_5m",
        "scope_json": {"start_date": "20241030", "end_date": "20241030"},
        "params_json": {},
        "status": "succeeded",
        "output_path": str(relative).replace("\\", "/"),
        "sha256": digest,
        "row_count": len(_rows()),
    }
    context = SimpleNamespace(settings=SimpleNamespace(data_root=tmp_path))

    gate = _explicit_execution_quality_gate(
        context,  # type: ignore[arg-type]
        selected={"ashare_5m": [row]},
        start_date=date(2024, 10, 30),
        end_date=date(2024, 10, 30),
        profile="ashare_intraday",
    )

    assert gate["ok"] is True
    assert gate["minute_source_audits"]["ashare_5m"]["excluded_rows"] == 2
    assert gate["minute_source_audits"]["ashare_5m"]["nontradable_rows"] == 1
    assert gate["minute_source_warnings"]


def test_scoped_execution_gate_keeps_empty_units_out_of_parquet_audit(
    tmp_path: Path,
) -> None:
    parquet_relative = Path("units") / "ashare_5m" / "unit.parquet"
    parquet_path = _write(tmp_path / parquet_relative, _rows())
    empty_relative = Path("units") / "ashare_5m" / "empty.empty.json"
    empty_path = tmp_path / empty_relative
    empty_path.write_text('{"allow_empty":true}', encoding="utf-8")
    selected = [
        {
            "unit_key": "ashare-5m-20241030-data",
            "dataset": "ashare_5m",
            "scope_json": {"start_date": "20241030", "end_date": "20241030"},
            "params_json": {},
            "status": "succeeded",
            "output_path": str(parquet_relative).replace("\\", "/"),
            "sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
            "row_count": len(_rows()),
        },
        {
            "unit_key": "ashare-5m-20241030-empty",
            "dataset": "ashare_5m",
            "scope_json": {"start_date": "20241030", "end_date": "20241030"},
            "params_json": {},
            "status": "succeeded",
            "output_path": str(empty_relative).replace("\\", "/"),
            "sha256": hashlib.sha256(empty_path.read_bytes()).hexdigest(),
            "row_count": 0,
        },
    ]
    context = SimpleNamespace(settings=SimpleNamespace(data_root=tmp_path))

    gate = _explicit_execution_quality_gate(
        context,  # type: ignore[arg-type]
        selected={"ashare_5m": selected},
        start_date=date(2024, 10, 30),
        end_date=date(2024, 10, 30),
        profile="ashare_intraday",
    )

    assert gate["ok"] is True
    assert gate["minute_source_audits"]["ashare_5m"]["source_rows"] == 51
