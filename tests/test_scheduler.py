import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from governance_fixtures import governed_etf_ready_evidence

from quant_data.config import Settings
from quant_data.coverage_data import DEFAULT_COVERAGE_BUNDLES, OPTIONAL_COVERAGE_BUNDLES
from quant_data.execution_contract import DAILY_QLIB_FIELD_CONTRACT_VERSION
from quant_data.history_bounds import GOVERNED_DAILY_STOCK_SCOPE_VERSION
from quant_data.qlib_builder import build_qlib_output_manifest
from quant_platform.alert_store import AlertStore
from quant_platform.factor_library import compile_qlib_expression
from quant_platform.feature_set_registry import get_feature_set
from quant_platform.job_store import JobStore
from quant_platform.research_horizon import canonical_sha256
from quant_platform.research_store import ResearchStore
from quant_platform.schedule_store import ScheduleStore
from quant_platform.scheduler import AUTOMATED_DATA_BUNDLES, SchedulerEngine


@pytest.mark.no_database
def test_research_report_backfill_waits_for_missing_data_volume(tmp_path: Path) -> None:
    class BackfillStub:
        reconciled = False

        def reconcile(self) -> None:
            self.reconciled = True

    backfill = BackfillStub()
    scheduler = object.__new__(SchedulerEngine)
    scheduler.settings = SimpleNamespace(
        data_root=tmp_path / "not-mounted",
        research_asset_auto_enabled=True,
        research_asset_auto_hour=0,
        research_asset_auto_minute=0,
    )
    scheduler.research_report_backfill = backfill

    assert scheduler._enqueue_research_report_backfill(datetime.now(UTC)) == 0
    assert backfill.reconciled is True


@pytest.mark.no_database
def test_automatic_pipeline_includes_default_coverage_but_not_optional_specialties() -> None:
    assert DEFAULT_COVERAGE_BUNDLES <= set(AUTOMATED_DATA_BUNDLES)
    assert OPTIONAL_COVERAGE_BUNDLES.isdisjoint(AUTOMATED_DATA_BUNDLES)


def _settings(database_url: str, tmp_path: Path) -> Settings:
    return Settings(
        api_url="https://relay.example/api/v1/query",
        token="test-token",
        data_root=tmp_path / "data",
        database_url=database_url,
        embedded_worker=False,
        research_asset_auto_enabled=False,
    )


def _write_qlib_dataset(
    data_root: Path,
    *,
    name: str,
    frequency: str,
    start: str,
    end: str,
    source_lineage_id: str,
) -> None:
    root = data_root / "qlib" / name
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / f"{frequency}.txt").write_text(
        f"{start}\n{end}\n",
        encoding="utf-8",
    )
    (root / "instruments" / "cn_all.txt").write_text(
        "SH600000\t2024-01-01\t2025-01-02\n",
        encoding="utf-8",
    )
    (root / "metadata").mkdir()
    (root / "metadata" / "provenance.json").write_text(
        json.dumps(
            {
                "frequency": frequency,
                "dataset_identity_sha256": "b" * 64,
                "dataset_lineage_id": "c" * 64,
                "source_lineage_id": source_lineage_id,
                "snapshot_manifest_sha256": "d" * 64,
                "lineage_verified": True,
                "output_manifest": build_qlib_output_manifest(root),
                **(
                    {
                        "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
                        "source_volume_unit": "hand",
                        "qlib_volume_unit": "share",
                        "source_amount_unit": "thousand_cny",
                        "qlib_amount_unit": "cny",
                        "source_hand_size": 100,
                        "index_volume_policy": "excluded_non_tradable_benchmark",
                        "governed_etf_whitelist": governed_etf_ready_evidence(),
                    }
                    if frequency == "day"
                    else {}
                ),
            }
        ),
        encoding="utf-8",
    )


def _write_trade_calendar(
    data_root: Path,
    *,
    open_days: list[date],
    closed_days: list[date] | None = None,
) -> None:
    target = data_root / "units" / "trade_cal" / "trade_cal.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            *({"cal_date": day, "is_open": 1} for day in open_days),
            *({"cal_date": day, "is_open": 0} for day in (closed_days or [])),
        ]
    )
    frame.to_parquet(target, index=False, compression="zstd", engine="pyarrow")


def test_scheduler_automatically_enqueues_factor_library_materialization(
    database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(database_url, tmp_path)
    _write_qlib_dataset(
        settings.data_root,
        name="cn-daily-20250102",
        frequency="day",
        start="2024-01-02",
        end="2025-01-02",
        source_lineage_id="a" * 64,
    )
    monkeypatch.setattr(
        "quant_platform.scheduler.FACTOR_LIBRARY_MIN_FREE_BYTES", 0
    )
    engine = SchedulerEngine(settings)

    assert engine._enqueue_due_factor_library_materialization(
        datetime(2025, 1, 2, 10, tzinfo=UTC)
    ) == 1
    assert engine._enqueue_due_factor_library_materialization(
        datetime(2025, 1, 2, 10, 1, tzinfo=UTC)
    ) == 0

    queued = JobStore(database_url).list(
        kinds=("factor_library_materialize",), limit=10
    )
    assert len(queued) == 1
    payload = queued[0]["payload"]
    assert payload["dataset"] == "cn-daily-20250102"
    assert payload["feature_set_id"] == "unified-research-v1"
    assert payload["universe"] == "cn_all"
    assert payload["start"] == "2024-01-02"
    assert payload["end"] == "2025-01-02"


def test_scheduler_materializes_once_and_enqueues_incremental_job(
    database_url: str, tmp_path: Path
) -> None:
    current = datetime(2025, 1, 2, 7, 29, tzinfo=UTC)
    store = ScheduleStore(database_url)
    schedule = store.create(
        name="daily incremental sync",
        kind="incremental_sync",
        timezone="Asia/Shanghai",
        run_time=time(15, 30),
        trading_days_only=True,
        payload={"profile": "core", "lookback_days": 7, "build_qlib": False},
        misfire_grace_seconds=1800,
        actor="operator",
        now=current,
    )
    assert schedule["next_run_at"] == "2025-01-02T07:30:00+00:00"
    engine = SchedulerEngine(_settings(database_url, tmp_path))
    first = engine.tick(current + timedelta(minutes=1))
    second = engine.tick(current + timedelta(minutes=1))
    assert first["materialized"] == 1
    assert first["processed"] == 1
    assert second["materialized"] == 0
    assert second["processed"] == 0
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0]["status"] == "enqueued"
    job = JobStore(database_url).get(runs[0]["job_id"])
    assert job["payload"]["start"] == "2024-12-26"
    assert job["payload"]["end"] == "latest"
    assert job["payload"]["build_qlib"] is False
    assert job["payload"]["incremental"] is True
    assert job["payload"]["finalize_after_download"] is False
    assert job["payload"]["snapshot_start"] == "2008-01-01"


def test_scheduler_creates_recoverable_full_data_pipeline(
    database_url: str, tmp_path: Path
) -> None:
    current = datetime(2025, 1, 2, 7, 29, tzinfo=UTC)
    store = ScheduleStore(database_url)
    store.create(
        name="weekly complete data pipeline",
        kind="data_pipeline",
        timezone="Asia/Shanghai",
        run_time=time(15, 30),
        trading_days_only=True,
        payload={
            "profile": "full",
            "lookback_days": 14,
            "snapshot_start": "2024-01-01",
            "bundles": ["cn_extended_daily", "cn_macro", "global_markets"],
        },
        misfire_grace_seconds=1800,
        actor="operator",
        now=current,
    )

    result = SchedulerEngine(_settings(database_url, tmp_path)).tick(current + timedelta(minutes=1))

    assert result["processed"] == 1
    run = store.list_runs()[0]
    job = JobStore(database_url).get(run["job_id"])
    assert job["kind"] == "bootstrap"
    assert job["payload"]["finalize_after_download"] is False
    assert job["payload"]["start"] == "2024-12-19"
    assert job["payload"]["snapshot_start"] == "2024-01-01"
    assert job["payload"]["snapshot_end"] == "2025-01-02"
    assert [step["kind"] for step in job["payload"]["pipeline_steps"]] == [
        "data_verify",
        "data_snapshot",
        "data_qlib",
        "qlib_baseline",
        "supplemental_cn_extended_daily",
        "supplemental_cn_macro",
        "supplemental_global_markets",
    ]
    assert {
        step["payload"]["end"]
        for step in job["payload"]["pipeline_steps"]
        if step["kind"].startswith("supplemental_")
    } == {"2025-01-02"}


def test_daily_publication_precedes_every_optional_supplemental_bundle(
    database_url: str, tmp_path: Path
) -> None:
    current = datetime(2025, 1, 2, 7, 29, tzinfo=UTC)
    store = ScheduleStore(database_url)
    store.create(
        name="daily publication priority fixture",
        kind="data_pipeline",
        timezone="Asia/Shanghai",
        run_time=time(15, 30),
        trading_days_only=True,
        payload={
            "profile": "full",
            "lookback_days": 7,
            "snapshot_start": "2008-01-01",
            "bundles": list(AUTOMATED_DATA_BUNDLES),
        },
        misfire_grace_seconds=1800,
        actor="operator",
        now=current,
    )

    SchedulerEngine(_settings(database_url, tmp_path)).tick(current + timedelta(minutes=1))

    run = store.list_runs()[0]
    job = JobStore(database_url).get(run["job_id"])
    kinds = [step["kind"] for step in job["payload"]["pipeline_steps"]]
    assert kinds[:4] == [
        "data_verify",
        "data_snapshot",
        "data_qlib",
        "qlib_baseline",
    ]
    assert all(kind.startswith("supplemental_") for kind in kinds[4:])


@pytest.mark.parametrize("kind", ["incremental_sync", "data_pipeline"])
@pytest.mark.parametrize("local_day", [date(2025, 1, 4), date(2025, 1, 5)])
def test_trading_day_data_schedules_skip_shanghai_weekends_without_jobs(
    database_url: str,
    tmp_path: Path,
    kind: str,
    local_day: date,
) -> None:
    zone = ZoneInfo("Asia/Shanghai")
    current = datetime.combine(local_day, time(15, 29), tzinfo=zone).astimezone(UTC)
    payload = (
        {"profile": "full", "lookback_days": 30, "build_qlib": True}
        if kind == "incremental_sync"
        else {
            "profile": "full",
            "lookback_days": 30,
            "snapshot_start": "2008-01-01",
            "bundles": ["cn_extended_daily"],
        }
    )
    store = ScheduleStore(database_url)
    store.create(
        name=f"{kind}-{local_day.isoformat()}",
        kind=kind,
        timezone="Asia/Shanghai",
        run_time=time(15, 30),
        trading_days_only=True,
        payload=payload,
        misfire_grace_seconds=1800,
        actor="operator",
        now=current,
    )

    result = SchedulerEngine(_settings(database_url, tmp_path)).tick(
        current + timedelta(minutes=1)
    )

    assert result["processed"] == 1
    run = store.list_runs()[0]
    assert run["status"] == "skipped"
    assert run["job_id"] is None
    assert "weekend" in run["message"]
    assert local_day.isoformat() in run["message"]
    assert JobStore(database_url).count(kinds=("bootstrap",)) == 0


def test_full_data_pipeline_recovery_coalesces_to_latest_due_weekday(
    database_url: str, tmp_path: Path
) -> None:
    zone = ZoneInfo("Asia/Shanghai")
    before_friday_slot = datetime(2025, 1, 3, 15, 29, tzinfo=zone).astimezone(UTC)
    recovered_on_monday = datetime(2025, 1, 6, 16, 0, tzinfo=zone).astimezone(UTC)
    store = ScheduleStore(database_url)
    store.create(
        name="recover full daily market data",
        kind="data_pipeline",
        timezone="Asia/Shanghai",
        run_time=time(15, 30),
        trading_days_only=True,
        payload={
            "profile": "full",
            "lookback_days": 7,
            "snapshot_start": "2008-01-01",
            "bundles": ["cn_extended_daily"],
        },
        misfire_grace_seconds=1800,
        actor="operator",
        now=before_friday_slot,
    )

    engine = SchedulerEngine(_settings(database_url, tmp_path))
    for _ in range(4):
        engine.tick(recovered_on_monday)

    runs = sorted(store.list_runs(), key=lambda item: item["scheduled_for"])
    assert [item["status"] for item in runs] == [
        "skipped",
        "skipped",
        "skipped",
        "enqueued",
    ]
    assert "superseded by a newer due" in runs[0]["message"]
    assert "weekend" in runs[1]["message"]
    assert "weekend" in runs[2]["message"]
    job = JobStore(database_url).get(runs[3]["job_id"])
    assert job["payload"]["snapshot_end"] == "2025-01-06"
    assert JobStore(database_url).count(kinds=("bootstrap",)) == 1


def test_non_full_data_pipeline_recovery_keeps_misfire_fail_closed(
    database_url: str, tmp_path: Path
) -> None:
    zone = ZoneInfo("Asia/Shanghai")
    before_slot = datetime(2025, 1, 3, 15, 29, tzinfo=zone).astimezone(UTC)
    recovered = datetime(2025, 1, 3, 17, 0, tzinfo=zone).astimezone(UTC)
    store = ScheduleStore(database_url)
    store.create(
        name="research assets still fail closed",
        kind="data_pipeline",
        timezone="Asia/Shanghai",
        run_time=time(15, 30),
        trading_days_only=True,
        payload={
            "profile": "research-assets",
            "lookback_days": 7,
            "snapshot_start": "2023-01-01",
            "bundles": ["research_corpus"],
        },
        misfire_grace_seconds=1800,
        actor="operator",
        now=before_slot,
    )

    SchedulerEngine(_settings(database_url, tmp_path)).tick(recovered)

    run = store.list_runs()[0]
    assert run["status"] == "missed"
    assert "failed closed" in run["message"]
    assert JobStore(database_url).count(kinds=("bootstrap",)) == 0


def test_scheduler_creates_bounded_recoverable_information_pipeline(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    current = datetime(2025, 1, 2, 13, 29, tzinfo=UTC)
    snapshot = tmp_path / "data" / "snapshots" / "cn-verified"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "name": "cn-verified",
                "start_date": "2008-01-01",
                "end_date": "2025-01-02",
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "verification.json").write_text(
        json.dumps({"ok": True, "errors": []}), encoding="utf-8"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "quant_platform.scheduler.resolve_information_evaluation_dataset",
        lambda _root, _evaluation: {
            "name": "qlib-frozen",
            "path": str(tmp_path / "data" / "qlib" / "qlib-frozen"),
            "provenance": {
                "dataset_identity_sha256": "a" * 64,
                "snapshot_name": "snapshot-frozen",
            },
        },
    )
    store = ScheduleStore(database_url)
    store.create(
        name="daily governed information pipeline",
        kind="information_pipeline",
        timezone="Asia/Shanghai",
        run_time=time(21, 30),
        trading_days_only=True,
        payload={
            "lookback_days": 3,
            "enable_nlp": True,
            "announcement_nlp_limit": 125,
            "corpus_nlp_limit": 175,
            "corpus_datasets": ["major_news", "irm_qa_sh", "irm_qa_sz"],
            "include_factor_evaluation": True,
            "factor_evaluation": {
                "dataset": "qlib-frozen",
                "periods": {
                    "train_start": "2020-01-01",
                    "train_end": "2021-12-31",
                    "valid_start": "2022-01-01",
                    "valid_end": "2023-12-31",
                    "test_start": "2024-01-08",
                    "test_end": "2025-01-02",
                },
                "universe": "cn_all",
                "benchmark": "SH000300",
            },
        },
        misfire_grace_seconds=1800,
        actor="operator",
        now=current,
    )

    result = SchedulerEngine(_settings(database_url, tmp_path)).tick(current + timedelta(minutes=2))

    assert result["processed"] == 1
    run = store.list_runs()[0]
    assert run["status"] == "enqueued"
    job = JobStore(database_url).get(run["job_id"])
    assert job["kind"] == "cninfo_announcements_download"
    assert job["payload"]["start"] == "2024-12-30"
    assert job["payload"]["end"] == "2025-01-02"
    assert job["payload"]["regulatory_only"] is True
    steps = job["payload"]["pipeline_steps"]
    assert [step["kind"] for step in steps] == [
        "announcement_nlp",
        "announcement_factor_register",
        "corpus_nlp",
        "corpus_factor_register",
        "event_market_response",
        "information_factor_evaluate",
        "multiface_audit",
    ]
    assert steps[0]["payload"]["limit"] == 125
    assert steps[2]["payload"]["limit"] == 175
    assert steps[4]["payload"]["snapshot_name"] == "cn-verified"
    assert steps[5]["payload"]["dataset"] == "qlib-frozen"
    assert steps[5]["payload"]["dataset_identity_sha256"] == "a" * 64
    assert steps[5]["payload"]["factor_names"] == [
        "announcement_logic_score",
        "announcement_tone",
        "irm_qa_sentiment_daily",
        "news_sentiment_daily",
    ]
    assert steps[6]["payload"] == {
        "dataset": "qlib-frozen",
        "snapshot_name": "cn-verified",
        "require_ready": True,
    }


def test_information_schedule_skips_when_a_conflicting_job_is_active(
    database_url: str, tmp_path: Path
) -> None:
    current = datetime(2025, 1, 2, 13, 29, tzinfo=UTC)
    store = ScheduleStore(database_url)
    store.create(
        name="daily raw information update",
        kind="information_pipeline",
        timezone="Asia/Shanghai",
        run_time=time(21, 30),
        trading_days_only=True,
        payload={},
        misfire_grace_seconds=1800,
        actor="operator",
        now=current,
    )
    active = JobStore(database_url).create(
        "cninfo_announcements_download",
        {"start": "2016-01-01", "end": "2025-01-02"},
        tmp_path / "active-cninfo.log",
    )

    result = SchedulerEngine(_settings(database_url, tmp_path)).tick(current + timedelta(minutes=2))

    assert result["processed"] == 1
    run = store.list_runs()[0]
    assert run["status"] == "skipped"
    assert "already active" in run["message"]
    assert JobStore(database_url).list()[0]["id"] == active["id"]


def test_scheduler_creates_weekly_structured_information_factor_refresh(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    current = datetime(2025, 1, 2, 13, 29, tzinfo=UTC)  # Thursday in Shanghai
    monkeypatch.setattr(
        "quant_platform.scheduler.resolve_information_evaluation_dataset",
        lambda _root, _evaluation: {
            "name": "qlib-frozen",
            "path": str(tmp_path / "data" / "qlib" / "qlib-frozen"),
            "provenance": {
                "dataset_identity_sha256": "a" * 64,
                "snapshot_name": "snapshot-frozen",
            },
        },
    )
    evaluation = {
        "dataset": "qlib-frozen",
        "periods": {
            "train_start": "2010-01-01",
            "train_end": "2019-12-31",
            "valid_start": "2020-01-01",
            "valid_end": "2022-12-31",
            "test_start": "2023-01-09",
            "test_end": "2025-01-02",
        },
        "universe": "cn_all",
        "benchmark": "SH000300",
    }
    store = ScheduleStore(database_url)
    store.create(
        name="weekly structured information factors",
        kind="information_factor_refresh",
        timezone="Asia/Shanghai",
        run_time=time(21, 30),
        trading_days_only=True,
        payload={
            "weekday": 3,
            "sources": ["report_rc", "major_news_mentions", "news_flash"],
            "factor_evaluation": evaluation,
        },
        misfire_grace_seconds=1800,
        actor="operator",
        now=current,
    )

    result = SchedulerEngine(_settings(database_url, tmp_path)).tick(
        current + timedelta(minutes=2)
    )

    assert result["processed"] == 1
    run = store.list_runs()[0]
    assert run["status"] == "enqueued"
    job = JobStore(database_url).get(run["job_id"])
    assert job["kind"] == "announcement_factor_register"
    assert job["payload"]["start"] == "2010-01-01"
    assert job["payload"]["end"] == "2025-01-02"
    steps = job["payload"]["pipeline_steps"]
    assert [step["kind"] for step in steps] == [
        "corpus_factor_register",
        "report_rc_factors",
        "report_rc_factor_register",
        "major_news_mentions",
        "major_news_mentions_factor_register",
        "news_flash_factors",
        "news_flash_factor_register",
        "event_market_response",
        "information_factor_evaluate",
        "multiface_audit",
    ]
    assert steps[1]["payload"]["start"] == "2010-01-01"
    assert steps[3]["payload"]["start"] == "2018-11-20"
    assert steps[5]["payload"]["start"] == "2018-11-20"
    assert steps[-3]["payload"]["snapshot_name"] == "snapshot-frozen"
    assert steps[-2]["payload"]["factor_names"] == [
        "announcement_logic_score",
        "announcement_tone",
        "irm_qa_sentiment_daily",
        "major_news_mention_count_daily",
        "major_news_mention_sentiment_daily",
        "news_flash_intensity_daily",
        "news_sentiment_daily",
        "policy_sentiment_daily",
        "report_rc_coverage_20d",
        "report_rc_eps_revision",
        "report_rc_rating_change",
    ]
    assert steps[-1]["payload"] == {
        "dataset": "qlib-frozen",
        "snapshot_name": "snapshot-frozen",
        "require_ready": True,
    }


def test_structured_information_refresh_skips_nonmatching_weekday(
    database_url: str, tmp_path: Path
) -> None:
    current = datetime(2025, 1, 2, 13, 29, tzinfo=UTC)
    store = ScheduleStore(database_url)
    store.create(
        name="friday structured information factors",
        kind="information_factor_refresh",
        timezone="Asia/Shanghai",
        run_time=time(21, 30),
        trading_days_only=True,
        payload={
            "weekday": 4,
            "factor_evaluation": {
                "dataset": "qlib-frozen",
                "periods": {
                    "train_start": "2010-01-01",
                    "train_end": "2019-12-31",
                    "valid_start": "2020-01-01",
                    "valid_end": "2022-12-31",
                    "test_start": "2023-01-09",
                    "test_end": "2025-01-02",
                },
            },
        },
        misfire_grace_seconds=1800,
        actor="operator",
        now=current,
    )

    SchedulerEngine(_settings(database_url, tmp_path)).tick(
        current + timedelta(minutes=2)
    )

    run = store.list_runs()[0]
    assert run["status"] == "skipped"
    assert "weekday 4" in run["message"]
    assert JobStore(database_url).list() == []


def test_scheduler_creates_daily_full_a_share_five_minute_increment(
    database_url: str, tmp_path: Path
) -> None:
    current = datetime(2025, 1, 2, 12, 59, tzinfo=UTC)
    store = ScheduleStore(database_url)
    store.create(
        name="daily A-share five-minute sync",
        kind="ashare_5m_sync",
        timezone="Asia/Shanghai",
        run_time=time(21, 0),
        trading_days_only=True,
        payload={"history_start": "2024-01-01", "daily_dataset": "daily-fixture"},
        misfire_grace_seconds=1800,
        actor="operator",
        now=current,
    )

    settings = _settings(database_url, tmp_path)
    _write_qlib_dataset(
        settings.data_root,
        name="daily-fixture",
        frequency="day",
        start="2024-01-01",
        end="2025-01-02",
        source_lineage_id="a" * 64,
    )
    _write_trade_calendar(
        settings.data_root,
        open_days=[date(2025, 1, 2), date(2025, 1, 3)],
    )
    result = SchedulerEngine(settings).tick(
        datetime(2025, 1, 2, 13, 1, tzinfo=UTC)
    )

    assert result["processed"] == 1
    run = store.list_runs()[0]
    job = JobStore(database_url).get(run["job_id"])
    assert job["kind"] == "ashare_5m_download"
    assert job["payload"]["start"] == "2024-01-01"
    assert job["payload"]["end"] == "2025-01-02"
    assert job["payload"]["snapshot_name"] == "ashare-5m-incremental-20250102"
    assert job["payload"]["source_lineage_id"] == "a" * 64
    assert job["payload"]["pipeline_steps"] == [
        {
            "kind": "minute_qlib",
            "payload": {
                "output_name": "ashare-5m-incremental-20250102-5min",
                "target_frequency": "5min",
            },
        }
    ]


def test_scheduler_skips_five_minute_sync_on_persisted_exchange_closure(
    database_url: str, tmp_path: Path
) -> None:
    current = datetime(2025, 1, 4, 12, 59, tzinfo=UTC)  # Saturday 20:59 Shanghai
    store = ScheduleStore(database_url)
    store.create(
        name="daily A-share five-minute sync",
        kind="ashare_5m_sync",
        timezone="Asia/Shanghai",
        run_time=time(21, 0),
        trading_days_only=True,
        payload={"history_start": "2024-01-01"},
        misfire_grace_seconds=1800,
        actor="operator",
        now=current,
    )
    settings = _settings(database_url, tmp_path)
    _write_trade_calendar(
        settings.data_root,
        open_days=[date(2025, 1, 3), date(2025, 1, 6)],
        closed_days=[date(2025, 1, 4), date(2025, 1, 5)],
    )

    result = SchedulerEngine(settings).tick(current + timedelta(minutes=2))

    assert result["processed"] == 1
    run = store.list_runs()[0]
    assert run["status"] == "skipped"
    assert "non-trading day 2025-01-04" in run["message"]
    assert JobStore(database_url).list() == []


def test_scheduler_fails_closed_when_daily_publication_misses_open_session(
    database_url: str, tmp_path: Path
) -> None:
    current = datetime(2025, 1, 3, 12, 59, tzinfo=UTC)
    store = ScheduleStore(database_url)
    store.create(
        name="daily A-share five-minute sync",
        kind="ashare_5m_sync",
        timezone="Asia/Shanghai",
        run_time=time(21, 0),
        trading_days_only=True,
        payload={"history_start": "2024-01-01", "daily_dataset": "daily-stale"},
        misfire_grace_seconds=1800,
        actor="operator",
        now=current,
    )
    settings = _settings(database_url, tmp_path)
    _write_trade_calendar(
        settings.data_root,
        open_days=[date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6)],
    )
    _write_qlib_dataset(
        settings.data_root,
        name="daily-stale",
        frequency="day",
        start="2024-01-01",
        end="2025-01-02",
        source_lineage_id="a" * 64,
    )

    result = SchedulerEngine(settings).tick(current + timedelta(minutes=2))

    assert result["processed"] == 1
    run = store.list_runs()[0]
    assert run["status"] == "failed"
    assert "daily Qlib dataset is required" in run["message"]
    assert JobStore(database_url).list() == []
    alerts = AlertStore(database_url).list()
    assert any(
        alert["category"] == "schedule_failure" and alert["source_id"] == run["id"]
        for alert in alerts
    )


def test_scheduler_rejects_five_minute_sync_before_market_is_fully_closed(
    database_url: str, tmp_path: Path
) -> None:
    current = datetime(2025, 1, 2, 6, 29, tzinfo=UTC)  # 14:29 Shanghai
    store = ScheduleStore(database_url)
    store.create(
        name="unsafe pre-close five-minute sync",
        kind="ashare_5m_sync",
        timezone="Asia/Shanghai",
        run_time=time(14, 30),
        trading_days_only=True,
        payload={"history_start": "2024-01-01"},
        misfire_grace_seconds=1800,
        actor="operator",
        now=current,
    )

    result = SchedulerEngine(_settings(database_url, tmp_path)).tick(
        current + timedelta(minutes=2)
    )

    assert result["processed"] == 1
    run = store.list_runs()[0]
    assert run["status"] == "failed"
    assert "after the market has fully closed" in run["message"]
    assert JobStore(database_url).list() == []


def test_scheduler_enqueues_bounded_rdagent_research_with_qlib_provenance(
    database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = datetime(2025, 1, 2, 12, 29, tzinfo=UTC)
    settings = _settings(database_url, tmp_path)
    dataset = settings.data_root / "qlib" / "cn-research"
    (dataset / "calendars").mkdir(parents=True)
    (dataset / "instruments").mkdir()
    (dataset / "features").mkdir()
    (dataset / "metadata").mkdir()
    calendar_start = date(2010, 1, 1)
    calendar_end = date(2025, 1, 2)
    calendar_days: list[str] = []
    cursor = calendar_start
    while cursor <= calendar_end:
        if cursor.weekday() < 5:
            calendar_days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    (dataset / "calendars" / "day.txt").write_text(
        "\n".join(calendar_days) + "\n", encoding="utf-8"
    )
    (dataset / "instruments" / "cn_all.txt").write_text(
        "SH600000\t2010-01-01\t2025-01-02\n", encoding="utf-8"
    )
    feature_set = get_feature_set("governed-baseline")
    required_fields = sorted(
        {
            field
            for expression in feature_set["features"].values()
            for field in compile_qlib_expression(str(expression)).required_fields
        }
    )
    field_year_coverage = {
        "version": "qlib-field-year-source-coverage-v1",
        "source_attribution_policy": "normalized-staging-and-snapshot-contracts-v1",
        "legacy_overlap_policy_version": "overlap-v1",
        "primary_market_history_start": "2010-01-01",
        "fields": {
            field: {
                "source_family": "test",
                "available_from": "2010-01-01",
                "available_to": "2025-01-02",
                "continuous_from": "2010-01-01",
                "research_available_from": "2010-01-01",
                "years": [
                    {
                        "year": 2010,
                        "observed_rows": 1,
                        "non_null_rows": 1,
                        "coverage_ratio": 1.0,
                        "first_session": "2010-01-01",
                        "last_session": "2025-01-02",
                        "source_contracts": ["test-source"],
                    }
                ],
            }
            for field in required_fields
        },
    }
    field_coverage_sha256 = canonical_sha256(field_year_coverage)
    field_year_coverage = {
        **field_year_coverage,
        "coverage_sha256": field_coverage_sha256,
    }
    (dataset / "metadata" / "provenance.json").write_text(
        json.dumps(
            {
                "frequency": "day",
                "dataset_identity_sha256": "a" * 64,
                "snapshot_manifest_sha256": "b" * 64,
                "qlib_builder_sha256": "c" * 64,
                "dataset_lineage_id": "c" * 64,
                "source_lineage_id": "d" * 64,
                "dataset_contract_sha256": "f" * 64,
                "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
                "fields": required_fields,
                "field_units": {field: "normalized" for field in required_fields},
                "research_features": {"version": "pit-research-features-v1"},
                "field_coverage_sha256": field_coverage_sha256,
                "field_year_coverage": field_year_coverage,
                "source_start_date": "2010-01-01",
                "source_end_date": "2025-01-02",
                "source_volume_unit": "hand",
                "qlib_volume_unit": "share",
                "source_amount_unit": "thousand_cny",
                "qlib_amount_unit": "cny",
                "source_hand_size": 100,
                "index_volume_policy": "excluded_non_tradable_benchmark",
                "governed_etf_whitelist": governed_etf_ready_evidence(),
                "lineage_verified": True,
                "output_manifest": build_qlib_output_manifest(dataset),
                "execution_controls": {
                    "native_complete_from": "2010-01-01",
                    "formal_execution_requires_native_controls": True,
                    "scope_version": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "quant_platform.scheduler.probe_rdagent",
        lambda _settings, _root: {"status": "ok"},
    )
    monkeypatch.setattr(
        "quant_platform.scheduler.require_ready_scenario",
        lambda _runtime, _settings, _scenario: None,
    )
    monkeypatch.setattr(
        "quant_platform.scheduler.expected_rdagent_runtime_identity",
        lambda _runtime, _scenario: {
            "production_reproducible": True,
            "source_tree_sha256": "e" * 64,
        },
    )
    store = ScheduleStore(database_url)
    research_payload = {
        "objective": "Research a low-turnover quality factor for CSI 300 enhancement.",
        "scenario": "fin_factor",
        "dataset": "cn-research",
        "feature_set_id": "governed-baseline",
        "horizon": "short",
        "loop_n": 2,
        "duration": "1h",
        "requested_by": "research-scheduler",
        "period_mode": "rolling",
    }
    for name in ("daily bounded factor research", "overlapping research guard"):
        store.create(
            name=name,
            kind="rdagent_research",
            timezone="Asia/Shanghai",
            run_time=time(20, 30),
            trading_days_only=True,
            payload=research_payload,
            misfire_grace_seconds=1800,
            actor="operator",
            now=current,
        )

    result = SchedulerEngine(settings).tick(current + timedelta(minutes=1))
    assert result["materialized"] == 2
    assert result["processed"] == 2
    schedule_runs = store.list_runs()
    assert sorted(item["status"] for item in schedule_runs) == ["enqueued", "skipped"]
    skipped = next(item for item in schedule_runs if item["status"] == "skipped")
    assert "already active" in skipped["message"]
    schedule_run = next(item for item in schedule_runs if item["status"] == "enqueued")
    assert schedule_run["status"] == "enqueued"
    job = JobStore(database_url).get(schedule_run["job_id"])
    assert job["kind"] == "rdagent_factor"
    assert job["payload"]["loop_n"] == 2
    research_run = next(
        item
        for item in ResearchStore(database_url).list_runs()
        if item["id"] == job["payload"]["research_run_id"]
    )
    assert research_run["status"] == "queued"
    assert research_run["job_id"] == job["id"]


def test_expired_schedule_run_lease_is_reclaimed(database_url: str) -> None:
    current = datetime(2025, 1, 2, 7, 29, tzinfo=UTC)
    store = ScheduleStore(database_url)
    store.create(
        name="lease recovery",
        kind="incremental_sync",
        timezone="Asia/Shanghai",
        run_time=time(15, 30),
        trading_days_only=True,
        payload={"profile": "core"},
        misfire_grace_seconds=1800,
        actor="operator",
        now=current,
    )
    store.materialize_due(current + timedelta(minutes=1))
    claimed = store.claim_run(now=current + timedelta(minutes=1), lease_seconds=60)
    assert claimed and claimed["attempts"] == 1
    assert store.claim_run(now=current + timedelta(minutes=1, seconds=30)) is None
    reclaimed = store.claim_run(now=current + timedelta(minutes=2, seconds=1))
    assert reclaimed and reclaimed["id"] == claimed["id"]
    assert reclaimed["attempts"] == 2


def test_waiting_schedule_run_is_persisted_and_reclaimed_after_retry_at(
    database_url: str,
) -> None:
    current = datetime(2025, 1, 2, 7, 29, tzinfo=UTC)
    store = ScheduleStore(database_url)
    store.create(
        name="dependency recovery",
        kind="incremental_sync",
        timezone="Asia/Shanghai",
        run_time=time(15, 30),
        trading_days_only=True,
        payload={"profile": "core"},
        misfire_grace_seconds=1800,
        actor="operator",
        now=current,
    )
    store.materialize_due(current + timedelta(minutes=1))
    claimed = store.claim_run(now=current + timedelta(minutes=1), lease_seconds=60)
    assert claimed is not None
    retry_at = current + timedelta(minutes=3)
    store.wait_run(claimed["id"], message="waiting for data", retry_at=retry_at)

    waiting = store.get_run(claimed["id"])
    assert waiting["status"] == "waiting"
    assert waiting["message"] == "waiting for data"
    assert waiting["finished_at"] is None
    assert store.claim_run(now=retry_at - timedelta(seconds=1)) is None
    reclaimed = store.claim_run(now=retry_at + timedelta(seconds=1))
    assert reclaimed is not None
    assert reclaimed["id"] == claimed["id"]
    assert reclaimed["attempts"] == 2


def test_job_idempotency_allows_multiple_scheduled_recommendation_jobs(
    database_url: str, tmp_path: Path
) -> None:
    jobs = JobStore(database_url)
    first = jobs.create(
        "recommendation_refresh",
        {"recommendation_portfolio": "one"},
        tmp_path / "one.log",
        dedupe_active_kind=False,
        idempotency_key="slot-one",
    )
    second = jobs.create(
        "recommendation_refresh",
        {"recommendation_portfolio": "two"},
        tmp_path / "two.log",
        dedupe_active_kind=False,
        idempotency_key="slot-two",
    )
    with pytest.raises(ValueError, match="different job payload"):
        jobs.create(
            "recommendation_refresh",
            {"recommendation_portfolio": "different-payload"},
            tmp_path / "duplicate.log",
            dedupe_active_kind=False,
            idempotency_key="slot-one",
        )
    assert first["id"] != second["id"]


def test_alerts_are_idempotent_deliverable_and_acknowledgeable(
    database_url: str, monkeypatch
) -> None:
    alerts = AlertStore(database_url)
    first = alerts.create(
        source_type="job",
        source_id="job-1",
        severity="critical",
        category="job_failure",
        title="job failed",
        message="provider timeout",
        dedupe_key="job:job-1:failed",
    )
    duplicate = alerts.create(
        source_type="job",
        source_id="job-1",
        severity="critical",
        category="job_failure",
        title="job failed again",
        message="same occurrence",
        dedupe_key="job:job-1:failed",
    )
    assert duplicate["id"] == first["id"]
    assert alerts.deliver_pending("") == 0
    assert alerts.get(first["id"])["delivery_status"] == "not_configured"
    acknowledged = alerts.acknowledge(first["id"], actor="risk-owner")
    assert acknowledged["status"] == "acknowledged"
    assert acknowledged["acknowledged_by"] == "risk-owner"

    delivery = alerts.create(
        source_type="risk_event",
        source_id="risk-2",
        severity="critical",
        category="portfolio_risk",
        title="risk threshold exceeded",
        message="portfolio paused",
        dedupe_key="risk:risk-2",
    )

    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

    monkeypatch.setattr(
        "quant_platform.alert_store.requests.post",
        lambda *_args, **_kwargs: Response(),
    )
    assert alerts.deliver_pending("https://alerts.internal/hook") == 1
    delivered = alerts.get(delivery["id"])
    assert delivered["delivery_status"] == "delivered"
    assert delivered["delivery_attempts"] == 1


@pytest.mark.no_database
def test_scheduler_projects_unified_account_actions_into_existing_alert_path() -> None:
    class Result:
        @staticmethod
        def all() -> list[object]:
            return []

    class Connection:
        def __enter__(self) -> object:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def execute(_statement: object) -> Result:
            return Result()

    scheduler = object.__new__(SchedulerEngine)
    scheduler.safe_mode = SimpleNamespace(
        check_persistent_nav_anomalies=lambda: None
    )
    scheduler.jobs = SimpleNamespace(engine=SimpleNamespace(connect=Connection))
    calls: list[int] = []

    def project(*, limit: int = 500) -> int:
        calls.append(limit)
        return 3

    scheduler.advice_alerts = SimpleNamespace(project=project)

    assert scheduler.project_alerts() == 3
    assert calls == [500]


def test_empty_alert_webhook_advances_past_not_configured_rows(database_url: str) -> None:
    alerts = AlertStore(database_url)
    created = [
        alerts.create(
            source_type="job",
            source_id=f"job-{index}",
            severity="critical",
            category="job_failure",
            title=f"job {index} failed",
            message="provider timeout",
            dedupe_key=f"job:job-{index}:failed",
        )
        for index in range(3)
    ]

    for index in range(3):
        assert alerts.deliver_pending("", limit=1) == 0
        statuses = [alerts.get(item["id"])["delivery_status"] for item in created]
        assert statuses[: index + 1] == ["not_configured"] * (index + 1)
        assert statuses[index + 1 :] == ["pending"] * (2 - index)

    assert alerts.deliver_pending("", limit=1) == 0
    assert [alerts.get(item["id"])["delivery_status"] for item in created] == [
        "not_configured",
        "not_configured",
        "not_configured",
    ]
