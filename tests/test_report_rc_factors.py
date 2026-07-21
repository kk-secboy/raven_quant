from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import quant_platform.db_cli as db_cli
from quant_data.availability import (
    NATIVE_HISTORY,
    availability_contract_label,
    filter_available,
    recoverability_level,
)
from quant_platform import external_factor_evaluation as ext
from quant_platform import report_rc_factors as m
from quant_platform.research_store import ResearchStore

NOW = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
# Lazy engine only: validation must fail before any database round-trip.
DUMMY_URL = "postgresql+psycopg://quantlab:quantlab@127.0.0.1:55433/quantlab_test"

OPEN_DAYS = list(pd.bdate_range("2024-01-01", periods=120).date)


def _write_parquet(directory: Path, rows: list[dict], name: str = "data.parquet") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / name, index=False)


def _seed_trade_cal(data_root: Path, days: list[date] | None = None) -> None:
    open_days = days or OPEN_DAYS
    rows = [{"cal_date": day.strftime("%Y%m%d"), "is_open": 1} for day in open_days]
    # A closed weekday must never be treated as a trading day.
    closed = open_days[-1] + timedelta(days=1)
    while closed.weekday() >= 5:
        closed += timedelta(days=1)
    rows.append({"cal_date": closed.strftime("%Y%m%d"), "is_open": 0})
    _write_parquet(data_root / "units" / "trade_cal", rows)


def _report_row(
    ts_code: str,
    org: str,
    report_date: str,
    rating: str,
    *,
    quarter: str = "2024Q4",
    eps: float | None = 1.0,
    title: str = "公司点评",
) -> dict:
    return {
        "ts_code": ts_code,
        "name": "测试股",
        "report_date": report_date,
        "report_title": title,
        "report_type": "一般报告",
        "classify": "一般报告",
        "org_name": org,
        "author_name": "分析师",
        "quarter": quarter,
        "eps": eps,
        "np": 1000.0,
        "max_price": 20.0,
        "min_price": 15.0,
        "rating": rating,
    }


def _seed_report_rc(
    data_root: Path, rows: list[dict], *, duplicate_in_snapshot: bool = True
) -> None:
    _write_parquet(data_root / "units" / "report_rc", rows)
    if duplicate_in_snapshot:
        # Exact duplicates living in both layouts must collapse onto one row.
        _write_parquet(data_root / "snapshots" / "snap1" / "parquet" / "report_rc", rows)


def _fields(rows: list[dict], days: list[date] | None = None) -> pd.DataFrame:
    reports = pd.DataFrame(rows)
    reports["report_date"] = pd.to_datetime(reports["report_date"])
    return m._fields_frame(reports, days or OPEN_DAYS, ingested_at=NOW)


def _unused_store() -> ResearchStore:
    return ResearchStore(DUMMY_URL)


# ---------------------------------------------------------------------------
# Rating ladder mapping
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_rating_ladder_maps_known_vocabularies() -> None:
    assert m.rating_level("买入") == 5
    assert m.rating_level("强烈推荐") == 5
    assert m.rating_level("增持") == 4
    assert m.rating_level("推荐") == 4
    assert m.rating_level("优于大市") == 4
    assert m.rating_level("中性") == 3
    assert m.rating_level("持有") == 3
    assert m.rating_level("减持") == 2
    assert m.rating_level("卖出") == 1
    assert m.rating_level("回避") == 1
    # Surrounding whitespace from the provider is tolerated.
    assert m.rating_level("  买入 ") == 5


@pytest.mark.no_database
def test_rating_ladder_unknown_ratings_fail_closed() -> None:
    # Unknown vocabularies are never guessed onto the ladder.
    assert m.rating_level("积极配置") is None
    assert m.rating_level("超强烈推荐") is None
    assert m.rating_level("") is None
    assert m.rating_level(None) is None


# ---------------------------------------------------------------------------
# Rating change events (intra-org chains)
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_rating_change_events_are_intra_org_with_signed_deltas() -> None:
    fields = _fields(
        [
            _report_row("600519.SH", "安信证券", "20240102", "增持", eps=1.0),
            _report_row("600519.SH", "安信证券", "20240109", "买入", eps=1.2),
            _report_row("600519.SH", "华西证券", "20240103", "买入", eps=1.1),
            _report_row("600519.SH", "华西证券", "20240110", "中性", eps=0.9),
            # First-ever rating of another org: no predecessor, no event.
            _report_row("600519.SH", "信达证券", "20240110", "买入", eps=1.0),
            # Repeated identical rating: no change, no event.
            _report_row("600519.SH", "安信证券", "20240116", "买入", eps=1.2),
        ]
    )
    events = m.build_rating_change_events(fields)
    assert len(events) == 2
    by_org = {row.org_name: row for row in events.itertuples()}
    upgrade = by_org["安信证券"]
    assert upgrade.prev_rating == "增持"
    assert upgrade.new_rating == "买入"
    assert (upgrade.prev_level, upgrade.new_level) == (4, 5)
    assert upgrade.rating_delta == 1.0
    downgrade = by_org["华西证券"]
    assert (downgrade.prev_level, downgrade.new_level) == (5, 3)
    assert downgrade.rating_delta == -2.0


@pytest.mark.no_database
def test_unknown_and_anonymous_ratings_are_excluded_from_events() -> None:
    fields = _fields(
        [
            _report_row("600519.SH", "安信证券", "20240102", "买入"),
            # Unknown rating in the middle of the chain: skipped, the chain
            # compares the bracketing known ratings instead of guessing.
            _report_row("600519.SH", "安信证券", "20240109", "积极配置"),
            _report_row("600519.SH", "安信证券", "20240116", "减持"),
            # Anonymous row can never form an intra-org chain.
            _report_row("600519.SH", "", "20240109", "买入"),
            _report_row("600519.SH", "", "20240116", "卖出"),
        ]
    )
    events = m.build_rating_change_events(fields)
    assert len(events) == 1
    event = events.iloc[0]
    assert event["prev_rating"] == "买入"
    assert event["new_rating"] == "减持"
    assert event["rating_delta"] == -3.0


@pytest.mark.no_database
def test_unknown_ratings_still_count_for_coverage() -> None:
    fields = _fields(
        [
            _report_row("600519.SH", "安信证券", "20240102", "积极配置"),
            _report_row("600519.SH", "华西证券", "20240103", "买入"),
        ]
    )
    coverage = m.build_coverage_frame(fields, OPEN_DAYS)
    first_grid = coverage[coverage["factor_date"] == coverage["factor_date"].min()]
    row = first_grid[first_grid["ts_code"] == "600519.SH"]
    assert row["report_count"].iloc[0] == 2


# ---------------------------------------------------------------------------
# PIT: availability policy and weekly grid
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_report_rc_availability_policy_is_registered() -> None:
    assert availability_contract_label("report_rc") == "strictly_after_announcement_date"
    assert recoverability_level("report_rc") == NATIVE_HISTORY


@pytest.mark.no_database
def test_report_rc_rows_are_invisible_on_the_report_date_itself() -> None:
    frame = pd.DataFrame(
        [{"ts_code": "600519.SH", "report_date": "2024-01-05", "rating": "买入"}]
    )
    # strictly_after_announcement_date: usable from the day after publication.
    assert filter_available("report_rc", frame, "2024-01-05").empty
    assert len(filter_available("report_rc", frame, "2024-01-06")) == 1


@pytest.mark.no_database
def test_available_at_is_the_next_trading_day_strictly_after_report_date() -> None:
    days = list(pd.bdate_range("2024-01-01", periods=20).date)  # Jan 1 is a Monday
    fields = _fields(
        [
            # Friday 2024-01-05 report -> Monday 2024-01-08.
            _report_row("600519.SH", "安信证券", "20240105", "买入"),
            # Saturday 2024-01-06 report -> Monday 2024-01-08.
            _report_row("000001.SZ", "安信证券", "20240106", "增持"),
        ],
        days,
    )
    by_code = fields.set_index("ts_code")
    assert by_code.loc["600519.SH", "available_at"] == pd.Timestamp("2024-01-08")
    assert by_code.loc["000001.SZ", "available_at"] == pd.Timestamp("2024-01-08")


@pytest.mark.no_database
def test_weekly_grid_lands_on_the_last_open_day_of_the_iso_week() -> None:
    days = list(pd.bdate_range("2024-01-01", periods=20).date)
    # Report Monday 2024-01-01 -> available Tuesday 2024-01-02 -> grid Friday.
    fields = _fields([_report_row("600519.SH", "安信证券", "20240101", "买入")], days)
    row = fields.iloc[0]
    assert row["available_at"] == pd.Timestamp("2024-01-02")
    assert row["factor_date"] == pd.Timestamp("2024-01-05")
    assert row["factor_date"] >= row["available_at"]
    grid = m.weekly_grid_days(days)
    # Every open day maps to a grid day within the same ISO week, never earlier.
    for day in days:
        assert grid[day] >= day
        assert grid[day].isocalendar()[:2] == day.isocalendar()[:2]


@pytest.mark.no_database
def test_factor_values_never_precede_availability() -> None:
    rows = []
    day = date(2024, 1, 2)
    for week in range(6):
        report_day = day + timedelta(days=7 * week)
        rows.append(
            _report_row(
                "600519.SH", "安信证券", report_day.strftime("%Y%m%d"),
                "买入" if week % 2 else "增持",
            )
        )
    fields = _fields(rows)
    events = m.build_rating_change_events(fields)
    series = m.build_rating_change_series(events)
    assert not series.empty
    # Factor datetimes are exactly the event grid days.
    assert set(series.index.get_level_values("datetime")) <= set(events["factor_date"])
    # Every aggregated row was available no later than its grid day.
    availability = fields.set_index(["ts_code", "org_name", "report_date"])["available_at"]
    for event in events.itertuples():
        available = availability.loc[(event.ts_code, event.org_name, event.report_date)]
        assert event.factor_date >= available


# ---------------------------------------------------------------------------
# Coverage factor
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_coverage_counts_reports_in_the_trailing_window_and_expires() -> None:
    fields = _fields(
        [
            _report_row("600519.SH", "安信证券", "20240102", "买入"),
            _report_row("600519.SH", "华西证券", "20240103", "增持"),
            _report_row("600519.SH", "安信证券", "20240109", "买入"),
            _report_row("000001.SZ", "信达证券", "20240110", "中性"),
            # A late report keeps 600519 covered so the frame extends far
            # enough for 000001.SZ to expire out of the trailing window.
            _report_row("600519.SH", "安信证券", "20240220", "买入"),
        ]
    )
    coverage = m.build_coverage_frame(fields, OPEN_DAYS, window_days=20)
    wide = coverage.set_index(["factor_date", "ts_code"])["report_count"].unstack("ts_code")
    first_grid = wide.index.min()
    assert wide.loc[first_grid, "600519.SH"] == 2  # 安信 01-02 + 华西 01-03
    assert pd.isna(wide.loc[first_grid, "000001.SZ"])
    second_grid = wide.index[1]
    assert wide.loc[second_grid, "600519.SH"] == 3
    assert wide.loc[second_grid, "000001.SZ"] == 1
    # Once every report is older than 20 open days the instrument disappears.
    last_grid = wide.index.max()
    assert pd.isna(wide.loc[last_grid, "000001.SZ"])
    assert wide.loc[last_grid, "600519.SH"] >= 1


@pytest.mark.no_database
def test_coverage_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window_days"):
        m.build_coverage_frame(pd.DataFrame(), OPEN_DAYS, window_days=0)


# ---------------------------------------------------------------------------
# EPS revision events
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_eps_revision_uses_nearest_quarter_per_report_intra_org() -> None:
    fields = _fields(
        [
            _report_row("600519.SH", "安信证券", "20240102", "增持", quarter="2024Q4", eps=1.0),
            # One report, two forecast quarters: only the nearest one revises.
            _report_row("600519.SH", "安信证券", "20240109", "买入", quarter="2024Q4", eps=1.2),
            _report_row("600519.SH", "安信证券", "20240109", "买入", quarter="2025Q4", eps=2.0),
            _report_row("600519.SH", "华西证券", "20240103", "买入", quarter="2024Q4", eps=1.1),
            _report_row("600519.SH", "华西证券", "20240110", "中性", quarter="2024Q4", eps=0.9),
        ]
    )
    events = m.build_eps_revision_events(fields)
    assert len(events) == 2
    by_org = {row.org_name: row for row in events.itertuples()}
    assert by_org["安信证券"].quarter == "2024Q4"
    assert by_org["安信证券"].eps_revision == pytest.approx(0.2)
    assert by_org["华西证券"].eps_revision == pytest.approx(-0.2 / 1.1)


@pytest.mark.no_database
def test_eps_revision_denominator_is_floored_near_zero() -> None:
    fields = _fields(
        [
            _report_row("600519.SH", "安信证券", "20240102", "增持", eps=0.005),
            _report_row("600519.SH", "安信证券", "20240109", "买入", eps=0.02),
        ]
    )
    events = m.build_eps_revision_events(fields)
    assert len(events) == 1
    # (0.02 - 0.005) / max(0.005, 0.01) = 1.5, not 3.0.
    assert events.iloc[0]["eps_revision"] == pytest.approx(1.5)


@pytest.mark.no_database
def test_eps_revision_skips_reports_without_forecast_columns() -> None:
    row = _report_row("600519.SH", "安信证券", "20240102", "买入", eps=None)
    row["eps"] = None
    fields = _fields([row])
    events = m.build_eps_revision_events(fields)
    assert events.empty


# ---------------------------------------------------------------------------
# Producer end to end (artifacts, manifests, determinism)
# ---------------------------------------------------------------------------


def _seed_full(data_root: Path) -> None:
    _seed_trade_cal(data_root)
    rows = []
    base = date(2024, 1, 2)
    instruments = [f"6000{i:02d}.SH" for i in range(12)]
    for week in range(14):
        report_day = base + timedelta(days=7 * week)
        for index, instrument in enumerate(instruments):
            rating = "买入" if (week + index) % 3 else "增持"
            rows.append(
                _report_row(
                    instrument,
                    f"券商{index % 4}",
                    report_day.strftime("%Y%m%d"),
                    rating,
                    eps=1.0 + 0.05 * week,
                )
            )
    _seed_report_rc(data_root, rows)


@pytest.mark.no_database
def test_process_report_rc_writes_artifacts_and_manifests(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_report_rc(tmp_path, now=lambda: NOW)
    assert summary.reports > 0
    assert summary.rating_events > 0
    assert summary.eps_events > 0
    assert summary.coverage_rows > 0
    for path in (
        summary.fields_path,
        summary.rating_events_path,
        summary.eps_events_path,
        summary.coverage_path,
    ):
        assert path.is_file()
    fields = pd.read_parquet(summary.fields_path)
    assert set(m.FIELDS_COLUMNS) <= set(fields.columns)
    # Raw forecast columns survive as the extension slot for future factors.
    assert {"eps", "np", "max_price", "min_price", "quarter"} <= set(fields.columns)
    for name in m.FACTOR_NAMES:
        entry = summary.factors[name]
        artifact_path = entry["artifact_path"]
        manifest = entry["manifest"]
        assert artifact_path.is_file()
        assert manifest["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert manifest["rows"] > 0
        assert manifest["source"]["dataset"] == "report_rc"
        assert manifest["source"]["producer_version"] == m.PRODUCER_VERSION
        assert "strictly after report_date" in manifest["availability_policy"][name]
        values = pd.read_parquet(artifact_path)
        assert {"datetime", "instrument", name} <= set(values.columns)
    # Deterministic rerun: identical inputs reproduce identical artifact hashes.
    again = m.process_report_rc(tmp_path, now=lambda: NOW)
    for name in m.FACTOR_NAMES:
        new_sha = again.factors[name]["manifest"]["sha256"]
        assert new_sha == summary.factors[name]["manifest"]["sha256"]


@pytest.mark.no_database
def test_process_missing_report_rc_fails_closed(tmp_path: Path) -> None:
    _seed_trade_cal(tmp_path)
    with pytest.raises(RuntimeError, match="report_rc parquet is unavailable"):
        m.process_report_rc(tmp_path, now=lambda: NOW)


@pytest.mark.no_database
def test_process_missing_trade_cal_fails_closed(tmp_path: Path) -> None:
    _seed_report_rc(tmp_path, [_report_row("600519.SH", "安信证券", "20240102", "买入")])
    with pytest.raises(RuntimeError, match="trade_cal"):
        m.process_report_rc(tmp_path, now=lambda: NOW)


@pytest.mark.no_database
def test_process_missing_required_column_fails_closed(tmp_path: Path) -> None:
    _seed_trade_cal(tmp_path)
    _write_parquet(
        tmp_path / "units" / "report_rc",
        [{"ts_code": "600519.SH", "report_date": "20240102"}],
    )
    with pytest.raises(RuntimeError, match="misses required columns"):
        m.process_report_rc(tmp_path, now=lambda: NOW)


# ---------------------------------------------------------------------------
# Sparse-event evaluation channel compatibility
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_produced_factor_fits_the_sparse_event_evaluation_shape(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_report_rc(tmp_path, now=lambda: NOW)
    factor = pd.read_parquet(summary.factors[m.RATING_CHANGE_FACTOR_NAME]["artifact_path"])
    factor = factor.set_index(["datetime", "instrument"])[m.RATING_CHANGE_FACTOR_NAME]

    instruments = [f"6000{i:02d}.SH" for i in range(12)]
    days = pd.bdate_range("2024-01-01", periods=100)
    rng = np.random.default_rng(7)
    label_rows = []
    for day in days:
        for instrument in instruments:
            label_rows.append((day, instrument, 0.01 * float(rng.standard_normal())))
    labels = pd.DataFrame(label_rows, columns=["datetime", "instrument", "label"]).set_index(
        ["datetime", "instrument"]
    )["label"]
    # Inject the event effect so the direction/selection split is well defined.
    for (stamp, instrument), value in factor.items():
        labels.loc[(stamp, instrument)] += 0.05 * float(value)

    valid_start = days[0].date()
    valid_end = days[-1].date()
    shape = ext.detect_external_factor_shape(
        factor,
        labels,
        valid_start=valid_start,
        valid_end=valid_end,
        shape_hint=ext.SHAPE_SPARSE_EVENT,
    )
    assert shape == ext.SHAPE_SPARSE_EVENT
    outcome = ext.evaluate_sparse_event_factor(
        factor,
        labels,
        valid_start=valid_start,
        valid_end=valid_end,
        test_start=valid_end + timedelta(days=10),
        test_end=valid_end + timedelta(days=200),
    )
    assert outcome["status"] == "ok"
    assert outcome["metrics"]["evaluation_shape"] == ext.SHAPE_SPARSE_EVENT
    assert outcome["metrics"]["event_days"] >= 10


# ---------------------------------------------------------------------------
# Registration: fail-closed validation (no database)
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_register_rejects_unknown_factor_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown report_rc factor"):
        m.register_report_rc_factor(_unused_store(), tmp_path, factor_name="nope")


@pytest.mark.no_database
def test_register_missing_manifest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest is missing"):
        m.register_report_rc_factor(
            _unused_store(), tmp_path, factor_name=m.RATING_CHANGE_FACTOR_NAME
        )


@pytest.mark.no_database
def test_register_tampered_artifact_fails_closed(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_report_rc(tmp_path, now=lambda: NOW)
    entry = summary.factors[m.COVERAGE_FACTOR_NAME]
    tampered = pd.read_parquet(entry["artifact_path"])
    tampered[m.COVERAGE_FACTOR_NAME] = tampered[m.COVERAGE_FACTOR_NAME] + 1
    tampered.to_parquet(entry["artifact_path"], index=False)
    with pytest.raises(ValueError, match="does not match the manifest sha256"):
        m.register_report_rc_factor(
            _unused_store(),
            m.default_factors_dir(tmp_path),
            factor_name=m.COVERAGE_FACTOR_NAME,
        )


@pytest.mark.no_database
def test_register_unexpected_source_identity_fails_closed(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_report_rc(tmp_path, now=lambda: NOW)
    entry = summary.factors[m.RATING_CHANGE_FACTOR_NAME]
    manifest = {**entry["manifest"], "source": {"dataset": "somewhere_else"}}
    entry["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source identity"):
        m.register_report_rc_factor(
            _unused_store(),
            m.default_factors_dir(tmp_path),
            factor_name=m.RATING_CHANGE_FACTOR_NAME,
        )


@pytest.mark.no_database
def test_code_artifact_recomputes_registered_values(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_report_rc(tmp_path, now=lambda: NOW)
    intermediates = {
        m.RATING_CHANGE_FACTOR_NAME: (
            pd.read_parquet(summary.rating_events_path), m.build_rating_change_series
        ),
        m.COVERAGE_FACTOR_NAME: (pd.read_parquet(summary.coverage_path), m.build_coverage_series),
        m.EPS_REVISION_FACTOR_NAME: (
            pd.read_parquet(summary.eps_events_path), m.build_eps_revision_series
        ),
    }
    for name, (frame, builder) in intermediates.items():
        manifest = summary.factors[name]["manifest"]
        source = m._code_artifact_source(
            factor_name=name, manifest=manifest, values_sha256=manifest["sha256"]
        )
        assert m.PRODUCER_VERSION in source
        assert manifest["sha256"] in source
        namespace: dict = {}
        exec(compile(source, "<code-artifact>", "exec"), namespace)
        recomputed = namespace["compute_factor"](frame)
        assert recomputed.equals(builder(frame))


# ---------------------------------------------------------------------------
# Registration into factor_candidates (real database)
# ---------------------------------------------------------------------------


def test_register_success(database_url: str, tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_report_rc(tmp_path, now=lambda: NOW)
    store = ResearchStore(database_url)
    factors_dir = m.default_factors_dir(tmp_path)

    result = m.register_report_rc_factor(
        store, factors_dir, factor_name=m.RATING_CHANGE_FACTOR_NAME
    )

    assert result["created"] is True
    candidate = store.get_candidate(result["candidate_id"])
    assert candidate["name"] == m.RATING_CHANGE_FACTOR_NAME
    assert candidate["status"] == "awaiting_evaluation"
    assert candidate["values_sha256"] == result["values_sha256"]
    code_path = Path(candidate["code_path"])
    assert code_path.is_file()
    assert candidate["code_sha256"] == hashlib.sha256(code_path.read_bytes()).hexdigest()
    variables = candidate["variables"]
    assert variables["source"]["dataset"] == "report_rc"
    assert variables["source"]["producer_version"] == m.PRODUCER_VERSION
    assert variables["rating_ladder"] == m.RATING_LEVELS
    assert "strictly after report_date" in candidate["description"]
    run = store.get_run(result["run_id"])
    assert run["kind"] == m.IMPORT_RUN_KIND
    assert run["status"] == "succeeded"
    assert run["dataset"] == "report_rc"
    events = {event["event_type"] for event in store.list_events(run["id"])}
    assert {"run.created", "candidate.imported", "run.succeeded"} <= events
    manifest_rows = summary.factors[m.RATING_CHANGE_FACTOR_NAME]["manifest"]["rows"]
    assert variables["rows"] == manifest_rows


def test_register_is_idempotent_for_same_sha256(database_url: str, tmp_path: Path) -> None:
    _seed_full(tmp_path)
    m.process_report_rc(tmp_path, now=lambda: NOW)
    store = ResearchStore(database_url)
    factors_dir = m.default_factors_dir(tmp_path)

    first = m.register_report_rc_factor(
        store, factors_dir, factor_name=m.COVERAGE_FACTOR_NAME
    )
    second = m.register_report_rc_factor(
        store, factors_dir, factor_name=m.COVERAGE_FACTOR_NAME
    )

    assert first["created"] is True
    assert second["created"] is False
    assert second["candidate_id"] == first["candidate_id"]
    assert second["run_id"] == first["run_id"]
    candidates = [
        item
        for item in store.list_candidates(limit=100)
        if item["name"] == m.COVERAGE_FACTOR_NAME
    ]
    assert len(candidates) == 1


def test_register_checksum_mismatch_writes_nothing(database_url: str, tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_report_rc(tmp_path, now=lambda: NOW)
    entry = summary.factors[m.EPS_REVISION_FACTOR_NAME]
    tampered = pd.read_parquet(entry["artifact_path"])
    tampered[m.EPS_REVISION_FACTOR_NAME] = tampered[m.EPS_REVISION_FACTOR_NAME] + 1
    tampered.to_parquet(entry["artifact_path"], index=False)
    store = ResearchStore(database_url)

    with pytest.raises(ValueError, match="does not match the manifest sha256"):
        m.register_report_rc_factor(
            store,
            m.default_factors_dir(tmp_path),
            factor_name=m.EPS_REVISION_FACTOR_NAME,
        )

    assert store.list_candidates(limit=100) == []
    assert store.list_runs(limit=100) == []


def test_cli_registers_all_factors_idempotently(
    database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_full(tmp_path)
    m.process_report_rc(tmp_path, now=lambda: NOW)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    runner = CliRunner()
    first = runner.invoke(db_cli.app, ["register-report-rc-factor"])
    assert first.exit_code == 0, first.output
    payload = json.loads(first.output)
    assert {item["factor_name"] for item in payload["factors"]} == set(m.FACTOR_NAMES)
    assert all(item["created"] is True for item in payload["factors"])

    second = runner.invoke(db_cli.app, ["register-report-rc-factor"])
    assert second.exit_code == 0, second.output
    repeated = json.loads(second.output)
    assert all(item["created"] is False for item in repeated["factors"])
    first_ids = {item["factor_name"]: item["candidate_id"] for item in payload["factors"]}
    for item in repeated["factors"]:
        assert item["candidate_id"] == first_ids[item["factor_name"]]
