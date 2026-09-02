"""Peripheral global-reference data: PIT registry, snapshot profile, verify."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quant_data.availability import (
    CURRENT_ONLY,
    FOREIGN_CLOSE_NEXT_CALENDAR_DAY,
    NATIVE_HISTORY,
    AvailabilityPolicyError,
    availability_policy,
    filter_available,
    recoverability_level,
)
from quant_data.catalog import ALL_DEFINITIONS, GLOBAL_REFERENCE_DATASETS
from quant_data.cli import (
    GLOBAL_REFERENCE_SNAPSHOT_PROFILE,
    SNAPSHOT_PROFILES,
    _profile_datasets,
    _required_profile_datasets,
)
from quant_data.verify import _verify_global_reference_frames

pytestmark = pytest.mark.no_database

PERIPHERAL_DAILY = (
    "us_daily",
    "us_daily_adj",
    "hk_daily",
    "hk_daily_adj",
    "index_global",
    "us_tycr",
    "us_tbr",
    "us_tltr",
    "us_trltr",
    "us_trycr",
)


def test_peripheral_datasets_register_the_foreign_close_policy() -> None:
    for dataset in (*PERIPHERAL_DAILY, "us_tradecal", "hk_tradecal"):
        policy = availability_policy(dataset)
        assert policy is not None and policy.kind == FOREIGN_CLOSE_NEXT_CALENDAR_DAY
    assert availability_policy("us_daily").date_columns == ("trade_date",)
    assert availability_policy("us_tycr").date_columns == ("date",)
    assert availability_policy("us_tradecal").date_columns == ("cal_date",)


def test_foreign_close_rows_are_visible_from_the_next_calendar_day() -> None:
    # A US session dated 2026-08-31 closes after the A-share close of that
    # date, so the row is invisible to A-share research on 08-31 and visible
    # from 09-01 (the next pre-open) onward.
    frame = pd.DataFrame(
        [{"ts_code": "NVDA", "trade_date": "2026-08-31", "close": 100.0}]
    )
    assert filter_available("us_daily", frame, "2026-08-31").empty
    assert len(filter_available("us_daily", frame, "2026-09-01")) == 1
    assert len(filter_available("hk_daily", frame, "2026-09-01")) == 1


def test_peripheral_policy_fails_closed_on_missing_or_bad_dates() -> None:
    frame = pd.DataFrame([{"ts_code": "NVDA", "close": 100.0}])
    with pytest.raises(AvailabilityPolicyError, match="lacks availability"):
        filter_available("us_daily", frame, "2026-09-01")
    bad = pd.DataFrame(
        [
            {"ts_code": "NVDA", "trade_date": "not-a-date", "close": 1.0},
            {"ts_code": "TSM", "trade_date": "2026-08-31", "close": 2.0},
        ]
    )
    available = filter_available("us_daily", bad, "2026-09-01")
    assert available["ts_code"].tolist() == ["TSM"]


def test_peripheral_recoverability_levels() -> None:
    for dataset in (*PERIPHERAL_DAILY, "us_tradecal", "hk_tradecal"):
        assert recoverability_level(dataset) == NATIVE_HISTORY
    assert recoverability_level("us_basic") == CURRENT_ONLY
    assert recoverability_level("hk_basic") == CURRENT_ONLY


def test_global_reference_profile_dataset_contract() -> None:
    assert GLOBAL_REFERENCE_SNAPSHOT_PROFILE in SNAPSHOT_PROFILES
    expected = set(GLOBAL_REFERENCE_DATASETS)
    assert _profile_datasets(GLOBAL_REFERENCE_SNAPSHOT_PROFILE) == expected
    assert _required_profile_datasets(GLOBAL_REFERENCE_SNAPSHOT_PROFILE) == (
        GLOBAL_REFERENCE_DATASETS
    )
    # The A-share profiles are untouched by the peripheral profile.
    assert not expected & set(_profile_datasets("full"))
    for dataset in sorted(expected):
        assert dataset in ALL_DEFINITIONS
        assert ALL_DEFINITIONS[dataset].primary_key


def _write_unit(data_root: Path, dataset: str, frame: pd.DataFrame, name: str) -> dict:
    unit_dir = data_root / "units" / dataset
    unit_dir.mkdir(parents=True, exist_ok=True)
    path = unit_dir / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return {"output_path": f"units/{dataset}/{name}.parquet"}


def _run_checks(data_root: Path, selected: dict[str, list[dict]]) -> tuple[list, list, dict]:
    import duckdb

    connection = duckdb.connect()
    try:
        return _verify_global_reference_frames(
            connection, selected, data_root, snapshot_end=date(2026, 8, 31)
        )
    finally:
        connection.close()


def _ohlc_frame(**overrides: object) -> pd.DataFrame:
    row = {
        "ts_code": "NVDA",
        "trade_date": "20260831",
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 104.0,
        "pct_chg": 3.9,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_verify_global_reference_accepts_clean_units(tmp_path: Path) -> None:
    selected = {
        "us_daily": [_write_unit(tmp_path, "us_daily", _ohlc_frame(), "a")],
        "us_tradecal": [
            _write_unit(
                tmp_path,
                "us_tradecal",
                # Calendars legitimately carry future sessions.
                pd.DataFrame([{"cal_date": "20261225", "is_open": 0}]),
                "cal",
            )
        ],
    }
    errors, warnings, checks = _run_checks(tmp_path, selected)
    assert not errors and not warnings
    assert checks["global_reference_rows"] == 2


def test_verify_global_reference_rejects_bad_dates_and_ohlc(tmp_path: Path) -> None:
    selected = {
        # One unparseable date row and one dated row with high < low.
        "us_daily": [
            _write_unit(
                tmp_path,
                "us_daily",
                pd.concat(
                    [
                        _ohlc_frame(trade_date="bogus"),
                        _ohlc_frame(high=90.0),
                    ],
                    ignore_index=True,
                ),
                "bad",
            )
        ],
        # Market data dated after the snapshot end.
        "index_global": [
            _write_unit(
                tmp_path,
                "index_global",
                _ohlc_frame(ts_code="SPX", trade_date="20260915"),
                "future",
            )
        ],
    }
    errors, warnings, checks = _run_checks(tmp_path, selected)
    assert any("us_daily: 1 rows have a missing or unparseable" in e for e in errors)
    # Row-quality noise (high < low) is quarantined as a warning, not an error.
    assert any("us_daily: 1 rows have high below low" in w for w in warnings)
    assert any("index_global: 1 rows are dated after the snapshot end" in e for e in errors)
    assert checks["global_reference_bad_date_rows"] == 1
    assert checks["global_reference_future_rows"] == 1


def test_verify_global_reference_flags_missing_ohlc_and_big_moves(
    tmp_path: Path,
) -> None:
    selected = {
        # Provider schema drift: OHLC columns absent.
        "hk_daily": [
            _write_unit(
                tmp_path,
                "hk_daily",
                pd.DataFrame([{"ts_code": "00700", "trade_date": "20260831"}]),
                "sparse",
            )
        ],
        "index_global": [
            _write_unit(
                tmp_path,
                "index_global",
                _ohlc_frame(ts_code="SPX", pct_chg=41.0),
                "jump",
            )
        ],
    }
    errors, warnings, checks = _run_checks(tmp_path, selected)
    assert any("hk_daily: provider columns lack the expected OHLC" in e for e in errors)
    assert any("index_global: 1 rows move more than 35%" in w for w in warnings)
    assert checks["global_reference_missing_ohlc_datasets"] == 1
    assert checks["global_reference_large_pct_chg_rows"] == 1


# ---------------------------------------------------------------------------
# Latest-generation duplicate adjudication (verify layer)
# ---------------------------------------------------------------------------


def _unresolved_keys(data_root: Path, dataset: str) -> int:
    import duckdb

    from quant_data.verify import _latest_generation_unresolved_keys

    unit_dir = data_root / "units" / dataset
    glob = str((unit_dir / "*.parquet").resolve()).replace("'", "''")
    connection = duckdb.connect()
    try:
        columns = {
            str(row[0])
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{glob}', union_by_name=true)"
            ).fetchall()
        }
        connection.execute(
            "CREATE TEMP TABLE selected_unit_files(path VARCHAR PRIMARY KEY)"
        )
        for path in sorted(unit_dir.glob("*.parquet")):
            connection.execute(
                "INSERT INTO selected_unit_files VALUES (?)", (str(path.resolve()),)
            )
        return _latest_generation_unresolved_keys(connection, dataset, glob, columns)
    finally:
        connection.close()


def _generation_rows(close: float, ingested: str, pct_change=None) -> dict:
    return {
        "ts_code": "NVDA",
        "trade_date": "20260819",
        "close": close,
        "pct_change": pct_change,
        "ingested_at": pd.Timestamp(ingested),
    }


def test_latest_generation_resolver_counts_only_true_conflicts(tmp_path: Path) -> None:
    # One key revised across generations (resolvable), one key torn inside a
    # single generation (true conflict), one key with identical re-pulls.
    rows = [
        _generation_rows(100.0, "2026-08-20T08:00:00Z", 0.5),
        _generation_rows(105.0, "2026-08-21T08:00:00Z", 0.5),  # newer generation wins
        {**_generation_rows(50.0, "2026-08-21T08:00:00Z", 0.5), "ts_code": "TSM"},
        {**_generation_rows(51.0, "2026-08-21T08:00:00Z", 0.5), "ts_code": "TSM"},
        # same generation, same completeness, different value: true conflict
        {**_generation_rows(60.0, "2026-08-21T08:00:00Z", 0.5), "ts_code": "AVGO"},
        {**_generation_rows(60.0, "2026-08-21T08:00:00Z", 0.5), "ts_code": "AVGO"},
    ]
    _write_unit(tmp_path, "us_daily_adj", pd.DataFrame(rows), "a")
    assert _unresolved_keys(tmp_path, "us_daily_adj") == 1


def test_latest_generation_resolver_accepts_completeness_tiebreak(
    tmp_path: Path,
) -> None:
    # Same generation: a NULL-derived-field row plus its filled twin resolve to
    # the filled row, so the key is not a true conflict.
    rows = [
        _generation_rows(21.34, "2026-08-22T18:11:42Z", None),
        _generation_rows(21.34, "2026-08-22T18:11:42Z", 3.39),
    ]
    _write_unit(tmp_path, "us_daily_adj", pd.DataFrame(rows), "b")
    assert _unresolved_keys(tmp_path, "us_daily_adj") == 0
