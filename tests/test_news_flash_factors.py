from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import quant_platform.db_cli as db_cli
from quant_data.availability import availability_policy, filter_available
from quant_platform import news_flash_factors as m
from quant_platform.external_factor_evaluation import (
    SHAPE_MARKET_TIMESERIES,
    detect_external_factor_shape,
)
from quant_platform.research_store import ResearchStore

NOW = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
DUMMY_URL = "postgresql+psycopg://quantlab:quantlab@127.0.0.1:55433/quantlab_test"

# One mid-week holiday (2024-01-03) so tests prove the persisted trade_cal —
# not weekday rules — drives factor dates and the trailing grid.
CALENDAR_DAYS = [
    (date(2024, 1, 2), 1),
    (date(2024, 1, 3), 0),
    (date(2024, 1, 4), 1),
    (date(2024, 1, 5), 1),
    (date(2024, 1, 6), 0),
    (date(2024, 1, 7), 0),
    (date(2024, 1, 8), 1),
    (date(2024, 1, 9), 1),
    (date(2024, 1, 10), 1),
    (date(2024, 1, 11), 1),
]
OPEN_DAYS = [day for day, flag in CALENDAR_DAYS if flag]


def _write_parquet(directory: Path, rows: list[dict], name: str = "data.parquet") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / name, index=False)


def _seed_trade_cal(data_root: Path) -> None:
    _write_parquet(
        data_root / "units" / "trade_cal",
        [
            {"cal_date": day.strftime("%Y%m%d"), "is_open": flag, "pretrade_date": ""}
            for day, flag in CALENDAR_DAYS
        ],
    )


def _flash_rows() -> list[dict]:
    """Flash datetimes per day; the 16:00 flash on 01-02 maps to 01-04."""

    rows = []

    def add(day: str, times: list[str]) -> None:
        for value in times:
            rows.append(
                {"datetime": f"{day} {value}", "title": "快讯", "content": "正文"}
            )

    add("2024-01-02", ["09:00:00", "12:00:00", "16:00:00"])
    add("2024-01-04", ["08:30:00", "10:00:00", "11:00:00", "14:00:00"])
    add("2024-01-05", ["09:00:00"] * 5)
    add("2024-01-08", ["09:00:00"] * 6)
    add("2024-01-09", ["09:00:00"] * 7)
    add("2024-01-10", ["09:00:00"] * 8)
    add("2024-01-11", ["09:00:00"] * 9)
    return rows


def _seed_news(data_root: Path) -> None:
    _write_parquet(data_root / "units" / "news", _flash_rows())


def _seed_full(data_root: Path) -> None:
    _seed_trade_cal(data_root)
    _seed_news(data_root)


def _unused_store() -> ResearchStore:
    return ResearchStore(DUMMY_URL)


# --- loading and daily counts ---------------------------------------------------


@pytest.mark.no_database
def test_load_news_missing_dataset_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="news parquet is unavailable"):
        m.load_news_flash_datetimes(tmp_path)


@pytest.mark.no_database
def test_load_news_missing_datetime_column_fail_closed(tmp_path: Path) -> None:
    _write_parquet(tmp_path / "units" / "news", [{"title": "t", "content": "c"}])
    with pytest.raises(RuntimeError, match="datetime"):
        m.load_news_flash_datetimes(tmp_path)


@pytest.mark.no_database
def test_load_news_drops_unparseable_timestamps(tmp_path: Path) -> None:
    _write_parquet(
        tmp_path / "units" / "news",
        [
            {"datetime": "2024-01-02 09:00:00", "title": "a", "content": "b"},
            {"datetime": "not-a-timestamp", "title": "a", "content": "b"},
        ],
    )
    moments = m.load_news_flash_datetimes(tmp_path)
    assert len(moments) == 1
    assert moments.iloc[0] == pd.Timestamp("2024-01-02 09:00:00")


@pytest.mark.no_database
def test_daily_counts_use_the_1500_cutoff_and_zero_fill(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    moments = m.load_news_flash_datetimes(tmp_path)

    counts = m.build_daily_counts(moments, OPEN_DAYS)

    keyed = counts.set_index("factor_date")["flash_count"]
    # 01-02 keeps the two pre-15:00 flashes; the 16:00 one rolls past the
    # Wednesday holiday to 01-04. Every open day in range is present.
    assert keyed.to_dict() == {
        pd.Timestamp(date(2024, 1, 2)): 2,
        pd.Timestamp(date(2024, 1, 4)): 5,
        pd.Timestamp(date(2024, 1, 5)): 5,
        pd.Timestamp(date(2024, 1, 8)): 6,
        pd.Timestamp(date(2024, 1, 9)): 7,
        pd.Timestamp(date(2024, 1, 10)): 8,
        pd.Timestamp(date(2024, 1, 11)): 9,
    }


@pytest.mark.no_database
def test_daily_counts_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window_days must be positive"):
        m.build_daily_counts(pd.Series(dtype="datetime64[ns]"), OPEN_DAYS, window_days=0)


# --- intensity series -----------------------------------------------------------


def _counts_frame() -> pd.DataFrame:
    days = [
        date(2024, 1, 2),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8),
        date(2024, 1, 9),
        date(2024, 1, 10),
        date(2024, 1, 11),
    ]
    return pd.DataFrame(
        {"factor_date": pd.to_datetime(days), "flash_count": [2, 5, 5, 6, 7, 8, 9]}
    )


@pytest.mark.no_database
def test_intensity_uses_strictly_prior_trailing_mean() -> None:
    series = m.build_intensity_series(_counts_frame(), window_days=3, min_history_days=2)

    keyed = series.droplevel("instrument")
    # First two days lack the minimum history.
    assert pd.Timestamp(date(2024, 1, 2)) not in keyed.index
    assert pd.Timestamp(date(2024, 1, 4)) not in keyed.index
    assert keyed[pd.Timestamp(date(2024, 1, 5))] == pytest.approx(5 / 3.5)
    assert keyed[pd.Timestamp(date(2024, 1, 8))] == pytest.approx(6 / 4.0)
    assert keyed[pd.Timestamp(date(2024, 1, 9))] == pytest.approx(7 / (16 / 3))
    assert keyed[pd.Timestamp(date(2024, 1, 10))] == pytest.approx(8 / 6.0)
    assert keyed[pd.Timestamp(date(2024, 1, 11))] == pytest.approx(9 / 7.0)
    assert set(series.index.get_level_values("instrument")) == {"MARKET"}


@pytest.mark.no_database
def test_intensity_drops_zero_flash_days() -> None:
    frame = _counts_frame()
    frame.loc[frame["factor_date"] == pd.Timestamp(date(2024, 1, 9)), "flash_count"] = 0

    series = m.build_intensity_series(frame, window_days=3, min_history_days=2)

    assert pd.Timestamp(date(2024, 1, 9)) not in series.droplevel("instrument").index
    # The zero still enters later denominators (it is a real zero on the grid).
    keyed = series.droplevel("instrument")
    assert keyed[pd.Timestamp(date(2024, 1, 10))] == pytest.approx(8 / ((5 + 6 + 0) / 3))


@pytest.mark.no_database
def test_intensity_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        m.build_intensity_series(_counts_frame(), window_days=0)
    with pytest.raises(ValueError, match="must be positive"):
        m.build_intensity_series(_counts_frame(), min_history_days=0)


# --- end-to-end -----------------------------------------------------------------


@pytest.mark.no_database
def test_process_writes_artifacts_deterministically(tmp_path: Path) -> None:
    _seed_full(tmp_path)

    summary = m.process_news_flash(
        tmp_path, window_days=3, min_history_days=2, now=lambda: NOW
    )

    assert summary.flashes == len(_flash_rows())
    counts = pd.read_parquet(summary.counts_path)
    assert list(counts.columns) == list(m.COUNTS_COLUMNS)
    assert summary.count_days == 7

    factors_dir = m.default_factors_dir(tmp_path)
    factor = pd.read_parquet(factors_dir / f"{m.INTENSITY_FACTOR_NAME}.parquet")
    assert list(factor.columns) == ["datetime", "instrument", m.INTENSITY_FACTOR_NAME]
    assert set(factor["instrument"]) == {"MARKET"}
    assert len(factor) == 5  # two leading days lack the minimum history

    manifest = json.loads(
        (factors_dir / f"{m.INTENSITY_FACTOR_NAME}.json").read_text(encoding="utf-8")
    )
    assert manifest["factor"] == m.INTENSITY_FACTOR_NAME
    assert manifest["rows"] == 5
    assert manifest["source"]["dataset"] == "news"
    assert manifest["source"]["producer_version"] == m.PRODUCER_VERSION
    assert "no look-ahead" in manifest["availability_policy"][m.INTENSITY_FACTOR_NAME]
    assert "MARKET" in manifest["instrument_convention"]
    assert manifest["sha256"] == hashlib.sha256(
        (factors_dir / f"{m.INTENSITY_FACTOR_NAME}.parquet").read_bytes()
    ).hexdigest()

    rerun = m.process_news_flash(
        tmp_path, window_days=3, min_history_days=2, now=lambda: NOW
    )
    assert (
        rerun.factors[m.INTENSITY_FACTOR_NAME]["manifest"]["sha256"]
        == summary.factors[m.INTENSITY_FACTOR_NAME]["manifest"]["sha256"]
    )


@pytest.mark.no_database
def test_process_missing_trade_cal_fail_closed(tmp_path: Path) -> None:
    _seed_news(tmp_path)
    with pytest.raises(RuntimeError, match="trade_cal"):
        m.process_news_flash(tmp_path, now=lambda: NOW)
    assert not (tmp_path / "news_flash" / "daily_counts.parquet").exists()


@pytest.mark.no_database
def test_produced_factor_fits_the_market_timeseries_shape(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    m.process_news_flash(tmp_path, window_days=3, min_history_days=2, now=lambda: NOW)
    factor = pd.read_parquet(
        m.default_factors_dir(tmp_path) / f"{m.INTENSITY_FACTOR_NAME}.parquet"
    )
    factor_series = factor.set_index(["datetime", "instrument"])[m.INTENSITY_FACTOR_NAME]

    shape = detect_external_factor_shape(
        factor_series,
        pd.Series(dtype="float64"),
        valid_start=date(2024, 1, 2),
        valid_end=date(2024, 1, 31),
    )

    assert shape == SHAPE_MARKET_TIMESERIES


@pytest.mark.no_database
def test_news_availability_policy_is_registered_and_enforced() -> None:
    policy = availability_policy("news")
    assert policy is not None and policy.kind == "same_trade_date_after_close"
    frame = pd.DataFrame(
        {"datetime": ["2024-01-02 16:00:00", "2024-01-05 09:00:00"], "content": ["a", "b"]}
    )
    visible = filter_available("news", frame, date(2024, 1, 4))
    assert visible["content"].tolist() == ["a"]


# --- registration (fail closed, no database) -------------------------------------


@pytest.mark.no_database
def test_register_rejects_unknown_factor_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown news flash factor"):
        m.register_news_flash_factor(_unused_store(), tmp_path, factor_name="nope")


@pytest.mark.no_database
def test_register_missing_manifest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest is missing"):
        m.register_news_flash_factor(_unused_store(), tmp_path)


@pytest.mark.no_database
def test_register_tampered_artifact_fails_closed(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_news_flash(
        tmp_path, window_days=3, min_history_days=2, now=lambda: NOW
    )
    entry = summary.factors[m.INTENSITY_FACTOR_NAME]
    tampered = pd.read_parquet(entry["artifact_path"])
    tampered[m.INTENSITY_FACTOR_NAME] = tampered[m.INTENSITY_FACTOR_NAME] + 1
    tampered.to_parquet(entry["artifact_path"], index=False)
    with pytest.raises(ValueError, match="does not match the manifest sha256"):
        m.register_news_flash_factor(_unused_store(), m.default_factors_dir(tmp_path))


@pytest.mark.no_database
def test_code_artifact_recomputes_registered_values(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_news_flash(tmp_path, now=lambda: NOW)
    entry = summary.factors[m.INTENSITY_FACTOR_NAME]
    manifest = entry["manifest"]
    source = m._code_artifact_source(manifest=manifest, values_sha256=manifest["sha256"])
    assert m.PRODUCER_VERSION in source
    assert manifest["sha256"] in source
    namespace: dict = {}
    exec(compile(source, "<code-artifact>", "exec"), namespace)
    counts = pd.read_parquet(summary.counts_path)
    recomputed = namespace["compute_factor"](counts)
    artifact = pd.read_parquet(entry["artifact_path"])
    registered = artifact.set_index(["datetime", "instrument"])[m.INTENSITY_FACTOR_NAME]
    assert recomputed.equals(registered)


# --- registration into factor_candidates (real database) ------------------------


def test_register_success(database_url: str, tmp_path: Path) -> None:
    _seed_full(tmp_path)
    m.process_news_flash(tmp_path, now=lambda: NOW)
    store = ResearchStore(database_url)
    factors_dir = m.default_factors_dir(tmp_path)

    result = m.register_news_flash_factor(store, factors_dir)

    assert result["created"] is True
    candidate = store.get_candidate(result["candidate_id"])
    assert candidate["name"] == m.INTENSITY_FACTOR_NAME
    assert candidate["status"] == "awaiting_evaluation"
    variables = candidate["variables"]
    assert variables["source"]["dataset"] == "news"
    assert variables["source"]["producer_version"] == m.PRODUCER_VERSION
    assert "trailing" in candidate["description"]
    run = store.get_run(result["run_id"])
    assert run["kind"] == m.IMPORT_RUN_KIND
    assert run["status"] == "succeeded"
    assert run["dataset"] == "news"
    events = {event["event_type"] for event in store.list_events(run["id"])}
    assert {"run.created", "candidate.imported", "run.succeeded"} <= events


def test_register_is_idempotent_for_same_sha256(database_url: str, tmp_path: Path) -> None:
    _seed_full(tmp_path)
    m.process_news_flash(tmp_path, now=lambda: NOW)
    store = ResearchStore(database_url)
    factors_dir = m.default_factors_dir(tmp_path)

    first = m.register_news_flash_factor(store, factors_dir)
    second = m.register_news_flash_factor(store, factors_dir)

    assert first["created"] is True
    assert second["created"] is False
    assert second["candidate_id"] == first["candidate_id"]
    candidates = [
        item
        for item in store.list_candidates(limit=100)
        if item["name"] == m.INTENSITY_FACTOR_NAME
    ]
    assert len(candidates) == 1


def test_cli_registers_idempotently(
    database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_full(tmp_path)
    m.process_news_flash(tmp_path, now=lambda: NOW)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    runner = CliRunner()
    first = runner.invoke(db_cli.app, ["register-news-flash-factor"])
    assert first.exit_code == 0, first.output
    payload = json.loads(first.output)
    assert payload["created"] is True

    second = runner.invoke(db_cli.app, ["register-news-flash-factor"])
    assert second.exit_code == 0, second.output
    repeated = json.loads(second.output)
    assert repeated["created"] is False
    assert repeated["candidate_id"] == payload["candidate_id"]
