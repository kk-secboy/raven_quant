"""Global-reference structured factors: build, PIT, registration and CLI."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import quant_platform.db_cli as db_cli
from quant_data.cli import app as data_app
from quant_platform import global_reference_factors as m
from quant_platform.global_reference_sector_map import (
    SECTOR_LEADER_MAP_VERSION,
    SECTOR_LEADERS,
    sector_factor_name,
    sector_leader_map_identity,
)
from quant_platform.research_store import ResearchStore

NOW = datetime(2026, 8, 31, 12, 0, 0, tzinfo=UTC)
# Lazy engine only: validation must fail before any database round-trip.
DUMMY_URL = "postgresql+psycopg://quantlab:quantlab@127.0.0.1:55433/quantlab_test"

OPEN_DAYS = list(pd.bdate_range("2026-08-03", periods=30).date)


def _write_parquet(directory: Path, rows: list[dict], name: str = "data.parquet") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / name, index=False)


def _seed_trade_cal(data_root: Path) -> None:
    rows = [{"cal_date": day.strftime("%Y%m%d"), "is_open": 1} for day in OPEN_DAYS]
    _write_parquet(data_root / "units" / "trade_cal", rows)


def _index_row(code: str, day: date, pct_chg: float) -> dict:
    return {
        "ts_code": code,
        "trade_date": day.strftime("%Y%m%d"),
        "close": 1000.0,
        "pct_chg": pct_chg,
    }


def _us_row(code: str, day: date, pct_chg: float) -> dict:
    return {
        "ts_code": code,
        "trade_date": day.strftime("%Y%m%d"),
        "close": 100.0,
        "pct_chg": pct_chg,
    }


def _seed_full(data_root: Path) -> None:
    """Trade calendar plus a complete peripheral window (Mon-Wed)."""

    _seed_trade_cal(data_root)
    days = [date(2026, 8, 24), date(2026, 8, 25), date(2026, 8, 26)]
    _write_parquet(
        data_root / "units" / "index_global",
        [
            _index_row(code, day, pct)
            for day, pct in zip(days, (0.5, -0.4, 0.3), strict=True)
            for code in ("SPX", "IXIC", "HSI")
        ],
    )
    _write_parquet(
        data_root / "units" / "index_global",
        [
            _index_row(code, day, pct)
            for day, pct in zip(days, (0.5, -0.4, 0.3), strict=True)
            for code in ("SPX", "IXIC", "HSI")
        ],
        name="dup.parquet",
    )  # exact duplicates across unit files collapse onto one row
    members = sorted({code for _label, codes in SECTOR_LEADERS.values() for code in codes})
    _write_parquet(
        data_root / "units" / "us_daily",
        [_us_row(code, day, 1.0) for day in days for code in members],
    )
    _write_parquet(
        data_root / "units" / "us_tycr",
        [
            {"date": day.isoformat(), "y10": yield_}
            for day, yield_ in zip(days, (4.10, 4.15, 4.12), strict=True)
        ],
    )


def _unused_store() -> ResearchStore:
    return ResearchStore(DUMMY_URL)


# ---------------------------------------------------------------------------
# Factor building
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_process_builds_all_factors_on_the_pit_grid(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_global_reference(tmp_path, now=lambda: NOW)

    assert set(summary.factors) == set(m.FACTOR_NAMES)
    spx = summary.factors[m.SPX_OVERNIGHT_FACTOR_NAME]
    assert spx["manifest"]["rows"] == 3
    assert spx["manifest"]["sha256"]
    frame = pd.read_parquet(spx["artifact_path"])
    # Source 2026-08-24 (Monday) lands on the next A-share open day 08-25.
    assert frame["datetime"].tolist()[0] == pd.Timestamp("2026-08-25")
    assert frame["instrument"].unique().tolist() == ["MARKET"]
    # Provider pct_chg is a percent; the factor stores fractional returns.
    assert frame[m.SPX_OVERNIGHT_FACTOR_NAME].tolist()[0] == pytest.approx(0.005)


@pytest.mark.no_database
def test_process_is_deterministic_across_reruns(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    first = m.process_global_reference(tmp_path, now=lambda: NOW)
    later = NOW + timedelta(hours=6)
    second = m.process_global_reference(tmp_path, now=lambda: later)

    for name in m.FACTOR_NAMES:
        assert first.factors[name]["manifest"]["sha256"] == (
            second.factors[name]["manifest"]["sha256"]
        )


@pytest.mark.no_database
def test_process_respects_the_requested_window(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_global_reference(
        tmp_path, start=date(2026, 8, 25), end=date(2026, 8, 26), now=lambda: NOW
    )
    assert summary.factors[m.SPX_OVERNIGHT_FACTOR_NAME]["manifest"]["rows"] == 2
    # A two-row treasury window yields exactly one first difference.
    assert summary.factors[m.US10Y_CHANGE_FACTOR_NAME]["manifest"]["rows"] == 1


@pytest.mark.no_database
def test_process_fails_closed_without_source_data(tmp_path: Path) -> None:
    _seed_trade_cal(tmp_path)
    with pytest.raises(RuntimeError, match="index_global parquet is unavailable"):
        m.process_global_reference(tmp_path, now=lambda: NOW)


@pytest.mark.no_database
def test_process_fails_closed_without_trade_calendar(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    import shutil

    shutil.rmtree(tmp_path / "units" / "trade_cal")
    with pytest.raises(RuntimeError, match="trade_cal trading calendar is unavailable"):
        m.process_global_reference(tmp_path, now=lambda: NOW)


@pytest.mark.no_database
def test_process_fails_closed_when_a_sector_member_is_missing(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    us_path = tmp_path / "units" / "us_daily" / "data.parquet"
    frame = pd.read_parquet(us_path)
    frame = frame[frame["ts_code"] != "XOM"]
    frame.to_parquet(us_path, index=False)
    with pytest.raises(RuntimeError, match="no rows for 能源 sector basket members"):
        m.process_global_reference(tmp_path, now=lambda: NOW)


@pytest.mark.no_database
def test_process_fails_closed_without_the_us10y_column(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    tycr_path = tmp_path / "units" / "us_tycr" / "data.parquet"
    frame = pd.read_parquet(tycr_path).rename(columns={"y10": "y30"})
    frame.to_parquet(tycr_path, index=False)
    with pytest.raises(RuntimeError, match="lacks the y10 10-year yield column"):
        m.process_global_reference(tmp_path, now=lambda: NOW)


@pytest.mark.no_database
def test_sector_basket_skips_dates_without_all_members(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    us_path = tmp_path / "units" / "us_daily" / "data.parquet"
    frame = pd.read_parquet(us_path)
    # TSM did not trade on the first source date: that date drops out entirely.
    frame = frame[
        ~((frame["ts_code"] == "TSM") & (frame["trade_date"] == "20260824"))
    ]
    frame.to_parquet(us_path, index=False)
    summary = m.process_global_reference(tmp_path, now=lambda: NOW)
    name = sector_factor_name("information_technology")
    basket = summary.factors[name]
    assert basket["manifest"]["rows"] == 2
    values = pd.read_parquet(basket["artifact_path"])
    # Uniform +1% members: the surviving dates hold the fractional mean 0.01.
    assert values[name].tolist() == [pytest.approx(0.01)] * 2


@pytest.mark.no_database
def test_all_eleven_gics_sector_factors_are_produced(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_global_reference(tmp_path, now=lambda: NOW)
    sector_names = {sector_factor_name(slug) for slug in SECTOR_LEADERS}
    assert len(sector_names) == 11
    assert sector_names <= set(summary.factors)
    for name in sector_names:
        manifest = summary.factors[name]["manifest"]
        identity = sector_leader_map_identity()
        assert manifest["source"]["sector_leader_map_version"] == identity["version"]
        assert manifest["source"]["sector_leader_map_sha256"] == identity["sha256"]
        assert manifest["rows"] == 3


@pytest.mark.no_database
def test_sector_map_identity_detects_tampering() -> None:
    identity = sector_leader_map_identity()
    assert identity["version"] == SECTOR_LEADER_MAP_VERSION
    tampered = dict(SECTOR_LEADERS)
    label, members = tampered["energy"]
    tampered["energy"] = (label, (*members, "BP"))
    canonical = {
        "version": SECTOR_LEADER_MAP_VERSION,
        "sectors": {
            slug: {"label": lbl, "members": list(m)}
            for slug, (lbl, m) in tampered.items()
        },
    }
    raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(raw.encode("utf-8")).hexdigest() != identity["sha256"]


@pytest.mark.no_database
def test_pit_boundary_matches_the_registered_policy(tmp_path: Path) -> None:
    """US Monday rows never land on the same-calendar-date A-share session."""
    _seed_full(tmp_path)
    summary = m.process_global_reference(tmp_path, now=lambda: NOW)
    frame = pd.read_parquet(
        summary.factors[m.HSI_OVERNIGHT_FACTOR_NAME]["artifact_path"]
    )
    # Source dates Mon/Tue/Wed land on Tue/Wed/Thu — never on themselves.
    assert frame["datetime"].tolist() == [
        pd.Timestamp(day) for day in ("2026-08-25", "2026-08-26", "2026-08-27")
    ]


# ---------------------------------------------------------------------------
# Registration: fail-closed validation (no database)
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_register_rejects_unknown_factor_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown global-reference factor"):
        m.register_global_reference_factor(_unused_store(), tmp_path, factor_name="nope")


@pytest.mark.no_database
def test_register_missing_manifest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest is missing"):
        m.register_global_reference_factor(
            _unused_store(), tmp_path, factor_name=m.SPX_OVERNIGHT_FACTOR_NAME
        )


@pytest.mark.no_database
def test_register_tampered_artifact_fails_closed(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_global_reference(tmp_path, now=lambda: NOW)
    entry = summary.factors[m.US10Y_CHANGE_FACTOR_NAME]
    tampered = pd.read_parquet(entry["artifact_path"])
    tampered[m.US10Y_CHANGE_FACTOR_NAME] = tampered[m.US10Y_CHANGE_FACTOR_NAME] + 1
    tampered.to_parquet(entry["artifact_path"], index=False)
    with pytest.raises(ValueError, match="does not match the manifest sha256"):
        m.register_global_reference_factor(
            _unused_store(),
            m.global_reference_factors_dir(tmp_path),
            factor_name=m.US10Y_CHANGE_FACTOR_NAME,
        )


@pytest.mark.no_database
def test_register_unexpected_source_identity_fails_closed(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_global_reference(tmp_path, now=lambda: NOW)
    entry = summary.factors[m.SPX_OVERNIGHT_FACTOR_NAME]
    manifest = {**entry["manifest"], "source": {"dataset": "somewhere_else"}}
    entry["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source identity"):
        m.register_global_reference_factor(
            _unused_store(),
            m.global_reference_factors_dir(tmp_path),
            factor_name=m.SPX_OVERNIGHT_FACTOR_NAME,
        )


@pytest.mark.no_database
def test_code_artifact_recomputes_registered_values(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_global_reference(tmp_path, now=lambda: NOW)
    intermediate = pd.read_parquet(summary.series_path)
    series = m.build_global_reference_series(tmp_path)
    for name in m.FACTOR_NAMES:
        manifest = summary.factors[name]["manifest"]
        source = m._code_artifact_source(
            factor_name=name, manifest=manifest, values_sha256=manifest["sha256"]
        )
        assert m.PRODUCER_VERSION in source
        assert manifest["sha256"] in source
        namespace: dict = {}
        exec(compile(source, "<code-artifact>", "exec"), namespace)
        recomputed = namespace["compute_factor"](intermediate)
        assert recomputed.equals(series[name])


# ---------------------------------------------------------------------------
# CLI (no database: production path fails before any round-trip)
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_quant_data_cli_builds_factors_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_full(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    runner = CliRunner()
    for _ in range(2):
        result = runner.invoke(
            data_app,
            ["global-reference-factors", "--start", "2026-08-24", "--end", "2026-08-26"],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["status"] == "succeeded"
        assert set(payload["factors"]) == set(m.FACTOR_NAMES)


@pytest.mark.no_database
def test_quant_db_register_rejects_unknown_factor_name(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        db_cli.app, ["register-global-reference-factor", "--factor-name", "nope"]
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Registration into factor_candidates (real database)
# ---------------------------------------------------------------------------


def test_register_success(database_url: str, tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_global_reference(tmp_path, now=lambda: NOW)
    store = ResearchStore(database_url)
    factors_dir = m.global_reference_factors_dir(tmp_path)

    result = m.register_global_reference_factor(
        store, factors_dir, factor_name=m.SPX_OVERNIGHT_FACTOR_NAME
    )

    assert result["created"] is True
    candidate = store.get_candidate(result["candidate_id"])
    assert candidate["name"] == m.SPX_OVERNIGHT_FACTOR_NAME
    assert candidate["status"] == "awaiting_evaluation"
    assert candidate["values_sha256"] == result["values_sha256"]
    assert candidate["variables"]["source"]["producer_version"] == m.PRODUCER_VERSION
    assert candidate["variables"]["source"]["dataset"] == "global_reference"
    assert candidate["variables"]["source"]["source_dataset"] == "index_global"
    run = store.get_run(result["run_id"])
    assert run["kind"] == m.IMPORT_RUN_KIND
    assert run["status"] == "succeeded"
    manifest_rows = summary.factors[m.SPX_OVERNIGHT_FACTOR_NAME]["manifest"]["rows"]
    assert candidate["variables"]["rows"] == manifest_rows


def test_register_is_idempotent_for_same_sha256(database_url: str, tmp_path: Path) -> None:
    _seed_full(tmp_path)
    m.process_global_reference(tmp_path, now=lambda: NOW)
    store = ResearchStore(database_url)
    factors_dir = m.global_reference_factors_dir(tmp_path)

    first = m.register_global_reference_factor(
        store, factors_dir, factor_name=m.HSI_OVERNIGHT_FACTOR_NAME
    )
    second = m.register_global_reference_factor(
        store, factors_dir, factor_name=m.HSI_OVERNIGHT_FACTOR_NAME
    )

    assert first["created"] is True
    assert second["created"] is False
    assert second["candidate_id"] == first["candidate_id"]
    assert second["run_id"] == first["run_id"]


# ---------------------------------------------------------------------------
# Dual-layer / generational deduplication
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_dual_layer_and_generational_duplicates_collapse(tmp_path: Path) -> None:
    """The same provider row in units and snapshots plus an older revised row.

    The units layer holds the current generation, the snapshots layer an exact
    copy of it, and a second unit file carries an older generation of the same
    business key with a different value.  The output must contain one row per
    (datetime, instrument) and keep the latest generation's value.
    """

    _seed_full(tmp_path)
    index_dir = tmp_path / "units" / "index_global"
    for stale in index_dir.glob("*.parquet"):
        stale.unlink()
    older = {
        "ts_code": "SPX",
        "trade_date": "20260824",
        "close": 1000.0,
        "pct_chg": 0.5,
        "ingested_at": pd.Timestamp("2026-08-25T08:00:00Z"),
    }
    newer = {**older, "pct_chg": 0.9, "ingested_at": pd.Timestamp("2026-08-26T08:00:00Z")}
    peers = [
        {
            "ts_code": code,
            "trade_date": "20260824",
            "close": 1000.0,
            "pct_chg": 0.5,
            "ingested_at": pd.Timestamp("2026-08-26T08:00:00Z"),
        }
        for code in ("IXIC", "HSI")
    ]
    pd.DataFrame([older]).to_parquet(index_dir / "older.parquet", index=False)
    pd.DataFrame([newer, *peers]).to_parquet(index_dir / "newer.parquet", index=False)
    # The exact same current-generation row also lives in the snapshot layer.
    snapshot_dir = tmp_path / "snapshots" / "snap1" / "parquet" / "index_global"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([newer]).to_parquet(snapshot_dir / "snap.parquet", index=False)

    series = m.build_global_reference_series(tmp_path)
    spx = series[m.SPX_OVERNIGHT_FACTOR_NAME]
    assert len(spx) == 1
    assert spx.iloc[0] == pytest.approx(0.009)


@pytest.mark.no_database
def test_holiday_compression_keeps_the_latest_source_session(tmp_path: Path) -> None:
    """A-share holidays compress several foreign sessions onto one open day.

    trade_cal opens only Mon 2026-08-24 and Thu 2026-08-27; index_global has
    US sessions Mon-Wed.  All three map to Thursday; the newest source
    session's value wins and no duplicate factor dates survive.
    """

    open_days = [date(2026, 8, 24), date(2026, 8, 27)]
    _write_parquet(
        tmp_path / "units" / "trade_cal",
        [{"cal_date": day.strftime("%Y%m%d"), "is_open": 1} for day in open_days],
    )
    days = [date(2026, 8, 24), date(2026, 8, 25), date(2026, 8, 26)]
    pcts = {"SPX": (0.1, 0.2, 0.3), "IXIC": (0.4, 0.5, 0.6), "HSI": (0.7, 0.8, 0.9)}
    _write_parquet(
        tmp_path / "units" / "index_global",
        [
            _index_row(code, day, pct)
            for day, _ in zip(days, (0, 0, 0), strict=True)
            for code in ("SPX", "IXIC", "HSI")
            for pct in (pcts[code][days.index(day)],)
        ],
    )
    members = sorted(
        {code for _label, codes in SECTOR_LEADERS.values() for code in codes}
    )
    _write_parquet(
        tmp_path / "units" / "us_daily",
        [_us_row(code, days[0], 1.0) for code in members],
    )
    _write_parquet(
        tmp_path / "units" / "us_tycr",
        [{"date": days[0].isoformat(), "y10": 4.10}, {"date": days[1].isoformat(), "y10": 4.15}],
    )

    series = m.build_global_reference_series(tmp_path)
    spx = series[m.SPX_OVERNIGHT_FACTOR_NAME]
    assert len(spx) == 1
    assert spx.index.get_level_values("datetime").tolist() == [pd.Timestamp("2026-08-27")]
    # Wednesday's session (the latest knowable one) wins the compressed slot.
    assert spx.iloc[0] == pytest.approx(0.003)
