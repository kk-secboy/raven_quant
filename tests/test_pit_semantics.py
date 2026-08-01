import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from quant_data.availability import (
    AVAILABILITY_POLICY_VERSION,
    CURRENT_ONLY,
    EVIDENCE_RECOVERABILITY_LEVELS,
    METADATA_AVAILABILITY_LAG_DAYS,
    NATIVE_HISTORY,
    RECONSTRUCTED,
    UNAVAILABLE,
    AvailabilityPolicyError,
    availability_contract_label,
    filter_available,
    recoverability_level,
)
from quant_data.models import ProviderResult
from quant_data.storage import ParquetStore
from quant_data.verify import _verify_disclosure_reconciliation
from quant_platform.strategy_backtest import _industries_at, _snapshot

pytestmark = pytest.mark.no_database


def _unit_row(unit_key: str, output_path: str, sha256: str = "0" * 64) -> dict:
    return {
        "unit_key": unit_key,
        "output_path": output_path,
        "row_count": 1,
        "sha256": sha256,
    }


# --- P1: row-level ingested_at persisted in unit files and snapshots --------


def test_write_unit_records_tz_aware_row_level_ingested_at(tmp_path: Path) -> None:
    storage = ParquetStore(
        tmp_path, clock=lambda: datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    )
    written = storage.write_unit(
        "daily",
        "unit-a",
        ProviderResult(
            "daily",
            ["ts_code", "trade_date", "close"],
            [{"ts_code": "000001.SZ", "trade_date": "20240102", "close": 10.0}],
            b"{}",
        ),
    )

    frame = pd.read_parquet(tmp_path / written.output_path)

    # pyarrow persists timestamps at microsecond precision; what matters is the
    # value stays tz-aware UTC.
    assert frame["ingested_at"].dt.tz is not None
    assert frame["ingested_at"].tolist() == [pd.Timestamp("2026-07-18T12:00:00Z")]


def test_snapshot_preserves_ingested_at_and_null_fills_legacy_units(tmp_path: Path) -> None:
    clock_time = datetime(2026, 7, 18, 8, 30, tzinfo=UTC)
    storage = ParquetStore(tmp_path, clock=lambda: clock_time)
    fresh = storage.write_unit(
        "daily",
        "fresh",
        ProviderResult(
            "daily",
            ["ts_code", "trade_date", "close"],
            [{"ts_code": "000001.SZ", "trade_date": "20240102", "close": 10.0}],
            b"{}",
        ),
    )
    # Legacy unit written before the ingested_at column existed.
    legacy_dir = tmp_path / "units" / "daily"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"ts_code": "000001.SZ", "trade_date": "2024-02-03", "close": 11.0}]
    ).to_parquet(legacy_dir / "legacy.parquet", index=False)

    snapshot = storage.build_snapshot(
        name="pit",
        successful_units={
            "daily": [
                _unit_row("fresh", fresh.output_path, fresh.sha256),
                _unit_row("legacy", "units/daily/legacy.parquet"),
            ]
        },
        manifest_extra={"profile": "test"},
    )

    frame = pd.concat(
        [pd.read_parquet(path) for path in (snapshot / "parquet" / "daily").rglob("*.parquet")],
        ignore_index=True,
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values("trade_date")
    assert frame["ingested_at"].iloc[0] == pd.Timestamp(clock_time)
    assert pd.isna(frame["ingested_at"].iloc[1])

    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    daily = manifest["datasets"]["daily"]
    assert daily["ingested_at_min"] is not None
    assert daily["ingested_at_max"] is not None
    assert daily["ingested_at_min"] <= daily["ingested_at_max"]
    # P4/P2: snapshot manifest carries the recoverability level and policy label.
    assert daily["recoverability"] == NATIVE_HISTORY
    assert daily["availability_policy"] == "same_trade_date_after_close"
    assert manifest["created_at"]


# --- P2/P6: availability registry and the read-side guard -------------------


def test_availability_registry_declares_all_policy_families() -> None:
    assert availability_contract_label("fina_indicator") == "strictly_after_announcement_date"
    assert availability_contract_label("daily_basic") == "same_trade_date_after_close"
    assert availability_contract_label("index_weight") == (
        f"effective_date_with_lag(days={METADATA_AVAILABILITY_LAG_DAYS})"
    )
    assert availability_contract_label("index_member_all") == (
        f"effective_date_with_lag(days={METADATA_AVAILABILITY_LAG_DAYS})"
    )
    assert AVAILABILITY_POLICY_VERSION >= 1


def test_financial_fields_are_invisible_before_the_announcement_date() -> None:
    frame = pd.DataFrame(
        [{"ts_code": "000001.SZ", "ann_date": "2024-04-30", "roe": 10.0}]
    )
    # P6(b): the field must not be readable on the announcement date itself;
    # the ASOF build rule is trade_date > ann_date, so the guard matches it.
    assert filter_available("income", frame, "2024-04-30").empty
    assert len(filter_available("income", frame, "2024-05-01")) == 1


def test_daily_basic_is_available_on_its_own_trade_date_only() -> None:
    frame = pd.DataFrame(
        [{"ts_code": "000001.SZ", "trade_date": "2024-01-02", "total_mv": 100.0}]
    )
    assert filter_available("daily_basic", frame, "2024-01-01").empty
    assert len(filter_available("daily_basic", frame, "2024-01-02")) == 1


def test_index_weight_applies_the_conservative_publication_lag() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "000001.SZ",
                "trade_date": "2024-01-31",
                "weight": 4.5,
            }
        ]
    )
    lag = METADATA_AVAILABILITY_LAG_DAYS
    before = pd.Timestamp("2024-01-31") + timedelta(days=lag - 1)
    at = pd.Timestamp("2024-01-31") + timedelta(days=lag)
    assert filter_available("index_weight", frame, before).empty
    assert len(filter_available("index_weight", frame, at)) == 1
    # The normalized qlib metadata frame renames the date column to datetime.
    normalized = frame.rename(columns={"trade_date": "datetime"})
    assert len(filter_available("index_weight", normalized, at)) == 1


def test_membership_guard_lags_both_entry_and_exit() -> None:
    frame = pd.DataFrame(
        [
            {
                "instrument": "SZ000001",
                "industry": "bank",
                "in_date": "2024-01-10",
                "out_date": "2024-06-10",
            }
        ]
    )
    lag = METADATA_AVAILABILITY_LAG_DAYS
    entry = pd.Timestamp("2024-01-10") + timedelta(days=lag)
    exit_ = pd.Timestamp("2024-06-10") + timedelta(days=lag)
    assert filter_available("index_member_all", frame, entry - timedelta(days=1)).empty
    assert len(filter_available("index_member_all", frame, entry)) == 1
    assert len(filter_available("index_member_all", frame, exit_)) == 1
    assert filter_available("index_member_all", frame, exit_ + timedelta(days=1)).empty


def test_guard_fails_closed_for_unregistered_dataset() -> None:
    # P6(c): a dataset without a declared policy must be refused, not trusted.
    with pytest.raises(AvailabilityPolicyError, match="no declared availability policy"):
        filter_available(
            "never_registered", pd.DataFrame({"trade_date": ["2024-01-02"]}), "2024-01-03"
        )


def test_guard_fails_closed_when_policy_columns_are_missing() -> None:
    with pytest.raises(AvailabilityPolicyError, match="lacks availability"):
        filter_available("income", pd.DataFrame({"ts_code": ["000001.SZ"]}), "2024-05-01")


def test_guard_drops_rows_with_unparseable_dates() -> None:
    frame = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "ann_date": "not-a-date", "roe": 1.0},
            {"ts_code": "000002.SZ", "ann_date": "2024-04-30", "roe": 2.0},
        ]
    )
    available = filter_available("income", frame, "2024-05-01")
    assert available["ts_code"].tolist() == ["000002.SZ"]


# --- P2 consumer side: conservative lag inside build_governed_signal ---------


def test_snapshot_helper_applies_publication_lag() -> None:
    values = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2024-01-31")],
            "instrument": ["SH000300"],
            "weight": [0.5],
        }
    )
    lag = METADATA_AVAILABILITY_LAG_DAYS
    assert _snapshot(
        values, pd.Timestamp("2024-01-31") + timedelta(days=lag - 1), "weight", lag_days=lag
    ).empty
    visible = _snapshot(
        values, pd.Timestamp("2024-01-31") + timedelta(days=lag), "weight", lag_days=lag
    )
    assert visible.loc["SH000300"] == pytest.approx(0.5)
    # Explicit lag 0 preserves the legacy immediate-effective behavior.
    assert not _snapshot(
        values, pd.Timestamp("2024-01-31"), "weight", lag_days=0
    ).empty


def test_industries_at_helper_applies_publication_lag() -> None:
    values = pd.DataFrame(
        {
            "instrument": ["SZ000001"],
            "industry": ["bank"],
            "in_date": [pd.Timestamp("2024-01-10")],
            "out_date": [pd.Timestamp("2024-06-10")],
        }
    )
    lag = METADATA_AVAILABILITY_LAG_DAYS
    entry = pd.Timestamp("2024-01-10") + timedelta(days=lag)
    exit_ = pd.Timestamp("2024-06-10") + timedelta(days=lag)
    assert _industries_at(values, entry - timedelta(days=1), lag_days=lag).empty
    assert _industries_at(values, entry, lag_days=lag).loc["SZ000001"] == "bank"
    assert _industries_at(values, exit_, lag_days=lag).loc["SZ000001"] == "bank"
    assert _industries_at(values, exit_ + timedelta(days=1), lag_days=lag).empty


# --- P4: recoverability levels ------------------------------------------------


def test_recoverability_levels_cover_the_audited_groups() -> None:
    assert recoverability_level("daily") == NATIVE_HISTORY
    assert recoverability_level("fina_indicator") == NATIVE_HISTORY
    assert recoverability_level("index_weight") == NATIVE_HISTORY
    assert recoverability_level("index_member_all") == RECONSTRUCTED
    assert recoverability_level("stock_basic") == CURRENT_ONLY
    assert recoverability_level("index_classify") == CURRENT_ONLY
    assert recoverability_level("never_seen_dataset") == UNAVAILABLE


def test_evidence_grade_levels_are_exactly_native_and_reconstructed() -> None:
    assert EVIDENCE_RECOVERABILITY_LEVELS == frozenset({NATIVE_HISTORY, RECONSTRUCTED})


# --- P5: disclosure_date reconciliation (flag-only) ---------------------------


def _write_units(root: Path, dataset: str, rows: list[dict]) -> str:
    directory = root / "units" / dataset
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "data.parquet"
    pd.DataFrame(rows).to_parquet(target, index=False)
    return f"units/{dataset}/data.parquet"


def test_disclosure_reconciliation_flags_ann_date_mismatch(tmp_path: Path) -> None:
    calendar_path = _write_units(
        tmp_path,
        "disclosure_date",
        [
            {
                "ts_code": "000001.SZ",
                "end_date": "20231231",
                "ann_date": "20240415",
                "actual_date": "20240420",
                "pre_date": None,
                "modify_date": None,
            }
        ],
    )
    income_path = _write_units(
        tmp_path,
        "income",
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240419",
                "end_date": "20231231",
                "revenue": 1.0,
            }
        ],
    )
    selected = {
        "disclosure_date": [{"output_path": calendar_path}],
        "income": [{"output_path": income_path}],
    }
    connection = duckdb.connect()
    try:
        warnings, checks = _verify_disclosure_reconciliation(connection, selected, tmp_path)
    finally:
        connection.close()

    assert checks["disclosure_calendar_rows"] == 1
    assert checks["compared_rows"] == 1
    assert checks["mismatched_ann_date_rows"] == 1
    assert any("income" in warning for warning in warnings)


def test_disclosure_reconciliation_is_silent_when_dates_agree(tmp_path: Path) -> None:
    calendar_path = _write_units(
        tmp_path,
        "disclosure_date",
        [
            {
                "ts_code": "000001.SZ",
                "end_date": "20231231",
                "ann_date": "20240415",
                "actual_date": "20240420",
                "pre_date": None,
                "modify_date": None,
            }
        ],
    )
    income_path = _write_units(
        tmp_path,
        "income",
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240420",
                "end_date": "20231231",
                "revenue": 1.0,
            }
        ],
    )
    selected = {
        "disclosure_date": [{"output_path": calendar_path}],
        "income": [{"output_path": income_path}],
    }
    connection = duckdb.connect()
    try:
        warnings, checks = _verify_disclosure_reconciliation(connection, selected, tmp_path)
    finally:
        connection.close()

    assert checks["compared_rows"] == 1
    assert checks["mismatched_ann_date_rows"] == 0
    assert warnings == []


def test_disclosure_reconciliation_skips_when_calendar_is_absent(tmp_path: Path) -> None:
    income_path = _write_units(
        tmp_path,
        "income",
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240420",
                "end_date": "20231231",
                "revenue": 1.0,
            }
        ],
    )
    connection = duckdb.connect()
    try:
        warnings, checks = _verify_disclosure_reconciliation(
            connection, {"income": [{"output_path": income_path}]}, tmp_path
        )
    finally:
        connection.close()

    assert checks["compared_rows"] == 0
    assert warnings == []
