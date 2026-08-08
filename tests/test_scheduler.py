import json
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

from quant_data.config import Settings
from quant_data.coverage_data import DEFAULT_COVERAGE_BUNDLES, OPTIONAL_COVERAGE_BUNDLES
from quant_platform.alert_store import AlertStore
from quant_platform.job_store import JobStore
from quant_platform.research_store import ResearchStore
from quant_platform.schedule_store import ScheduleStore
from quant_platform.scheduler import AUTOMATED_DATA_BUNDLES, SchedulerEngine


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
            }
        ),
        encoding="utf-8",
    )


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
    assert job["payload"]["finalize_after_download"] is False
    assert job["payload"]["snapshot_start"] == "2018-01-01"


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
    assert [step["kind"] for step in job["payload"]["pipeline_steps"]] == [
        "supplemental_cn_extended_daily",
        "supplemental_cn_macro",
        "supplemental_global_markets",
        "data_verify",
        "data_snapshot",
        "data_qlib",
        "qlib_baseline",
    ]


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
            "provenance": {"dataset_identity_sha256": "a" * 64},
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


def test_scheduler_enqueues_bounded_rdagent_research_with_qlib_provenance(
    database_url: str, tmp_path: Path
) -> None:
    current = datetime(2025, 1, 2, 12, 29, tzinfo=UTC)
    settings = _settings(database_url, tmp_path)
    dataset = settings.data_root / "qlib" / "cn-research"
    (dataset / "calendars").mkdir(parents=True)
    (dataset / "instruments").mkdir()
    (dataset / "features").mkdir()
    (dataset / "metadata").mkdir()
    (dataset / "calendars" / "day.txt").write_text("2018-01-01\n2025-01-02\n", encoding="utf-8")
    (dataset / "instruments" / "cn_all.txt").write_text(
        "SH600000\t2018-01-01\t2025-01-02\n", encoding="utf-8"
    )
    (dataset / "metadata" / "provenance.json").write_text(
        json.dumps(
            {
                "dataset_identity_sha256": "a" * 64,
                "snapshot_manifest_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    store = ScheduleStore(database_url)
    research_payload = {
        "objective": "Research a low-turnover quality factor for CSI 300 enhancement.",
        "dataset": "cn-research",
        "loop_n": 2,
        "duration": "1h",
        "requested_by": "research-scheduler",
        "periods": {
            "train_start": "2018-01-01",
            "train_end": "2021-12-31",
            "valid_start": "2022-01-01",
            "valid_end": "2023-12-31",
            "test_start": "2024-01-01",
            "test_end": "2025-01-02",
        },
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
    research_run = ResearchStore(database_url).list_runs()[0]
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
    duplicate = jobs.create(
        "recommendation_refresh",
        {"recommendation_portfolio": "different-payload"},
        tmp_path / "duplicate.log",
        dedupe_active_kind=False,
        idempotency_key="slot-one",
    )
    assert first["id"] != second["id"]
    assert duplicate["id"] == first["id"]


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
