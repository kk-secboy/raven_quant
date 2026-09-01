from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from threading import Event, Thread

import pytest
from sqlalchemy import insert, select, update

from quant_data.config import Settings
from quant_data.database import (
    jobs,
    open_database,
    research_campaign_events,
    research_campaigns,
    research_program_events,
    research_programs,
    schedule_runs,
    schedules,
)
from quant_platform.job_store import JobStore
from quant_platform.legacy_research_retirement import LegacyResearchRetirement
from quant_platform.schedule_store import (
    LEGACY_RESEARCH_SCHEDULE_ERROR,
    ScheduleStore,
)
from quant_platform.scheduler import SchedulerEngine


def _scheduler_settings(database_url: str, tmp_path: Path) -> Settings:
    return Settings(
        api_url="https://relay.example/api/v1/query",
        token="test-token",
        data_root=tmp_path / "data",
        database_url=database_url,
        embedded_worker=False,
        research_asset_auto_enabled=False,
    )


def _seed_legacy_campaign(database_url: str) -> str:
    campaign_id = "legacy-campaign-history"
    current = datetime(2020, 1, 1, tzinfo=UTC)
    with open_database(database_url).begin() as connection:
        connection.execute(
            insert(research_campaigns).values(
                id=campaign_id,
                name="retired campaign history",
                status="cancelled",
                stage="research",
                objective="Preserve historical research evidence.",
                dataset="cn-legacy",
                benchmark="SH000300",
                universe="cn_all",
                recipe_id="legacy_recipe",
                config_json={},
                state_json={"retired": True},
                attempts=1,
                next_action_at=current,
                created_by="legacy-migration",
                created_at=current,
                updated_at=current,
                finished_at=current,
            )
        )
        connection.execute(
            insert(research_campaign_events).values(
                campaign_id=campaign_id,
                event_type="campaign.cancelled",
                actor="legacy-migration",
                payload_json={"reason": "single mainline cutover"},
                created_at=current,
            )
        )
    return campaign_id


def test_one_time_retirement_uses_the_only_privileged_legacy_write_path(
    database_url: str,
) -> None:
    current = datetime(2020, 1, 1, tzinfo=UTC)
    program_id = "active-legacy-program"
    campaign_id = "active-legacy-campaign"
    with open_database(database_url).begin() as connection:
        connection.execute(
            insert(research_programs).values(
                id=program_id,
                name="active legacy program",
                status="active",
                recipe_id="legacy_recipe",
                objective="retire this legacy control path",
                benchmark="SH000300",
                universe="cn_all",
                dataset_lineage_id="a" * 64,
                config_json={},
                min_new_trading_days=252,
                max_active_campaigns=1,
                next_check_at=current,
                created_by="legacy-system",
                created_at=current,
                updated_at=current,
            )
        )
        connection.execute(
            insert(research_campaigns).values(
                id=campaign_id,
                name="active legacy campaign",
                status="running",
                stage="research",
                objective="retire this legacy campaign",
                dataset="cn-legacy",
                benchmark="SH000300",
                universe="cn_all",
                recipe_id="legacy_recipe",
                research_program_id=program_id,
                dataset_identity_sha256="b" * 64,
                config_json={},
                state_json={},
                attempts=1,
                next_action_at=current,
                created_by="legacy-system",
                created_at=current,
                updated_at=current,
            )
        )

    retirement = LegacyResearchRetirement(database_url)
    result = retirement.apply(actor="cutover-test")
    repeated = retirement.apply(actor="cutover-test")

    assert result["retired_campaign_ids"] == [campaign_id]
    assert result["retired_program_ids"] == [program_id]
    assert repeated["retired_campaign_ids"] == []
    assert repeated["retired_program_ids"] == []
    with open_database(database_url).connect() as connection:
        campaign_status = connection.scalar(
            select(research_campaigns.c.status).where(
                research_campaigns.c.id == campaign_id
            )
        )
        program_status = connection.scalar(
            select(research_programs.c.status).where(
                research_programs.c.id == program_id
            )
        )
    assert campaign_status == "cancelled"
    assert program_status == "cancelled"
    with open_database(database_url).connect() as connection:
        campaign_events = connection.execute(
            select(research_campaign_events.c.id).where(
                research_campaign_events.c.campaign_id == campaign_id,
                research_campaign_events.c.event_type == "campaign.cancelled",
            )
        ).all()
        program_events = connection.execute(
            select(research_program_events.c.id).where(
                research_program_events.c.program_id == program_id,
                research_program_events.c.event_type == "program.cancelled",
            )
        ).all()
    assert len(campaign_events) == 1
    assert len(program_events) == 1


def test_succeeded_campaign_schedule_is_disabled_but_history_and_oos_remain(
    database_url: str,
) -> None:
    current = datetime(2020, 1, 2, tzinfo=UTC)
    program_id = "completed-legacy-program"
    campaign_id = "completed-legacy-campaign"
    payload_schedule_id = "completed-campaign-refresh"
    actor_schedule_id = "orphan-campaign-schedule"
    with open_database(database_url).begin() as connection:
        connection.execute(
            insert(research_programs).values(
                id=program_id,
                name="completed legacy program",
                status="cancelled",
                recipe_id="legacy_recipe",
                objective="preserve consumed OOS evidence",
                benchmark="SH000300",
                universe="cn_all",
                dataset_lineage_id="c" * 64,
                config_json={},
                min_new_trading_days=252,
                max_active_campaigns=1,
                next_check_at=current,
                created_by="legacy-system",
                created_at=current,
                updated_at=current,
            )
        )
        connection.execute(
            insert(schedules),
            [
                {
                    "id": payload_schedule_id,
                    "name": "completed campaign recommendation refresh",
                    "kind": "recommendation_refresh",
                    "status": "active",
                    "desired_status": "active",
                    "suspension_reason": None,
                    "timezone": "Asia/Shanghai",
                    "run_time": time(16, 30),
                    "trading_days_only": True,
                    "payload_json": {
                        "recommendation_portfolio_id": "legacy-portfolio",
                    },
                    "misfire_grace_seconds": 3600,
                    "next_run_at": current - timedelta(minutes=1),
                    "created_by": "legacy-controller",
                    "created_at": current,
                    "updated_at": current,
                },
                {
                    "id": actor_schedule_id,
                    "name": "orphan campaign actor schedule",
                    "kind": "rdagent_research",
                    "status": "active",
                    "desired_status": "active",
                    "suspension_reason": None,
                    "timezone": "Asia/Shanghai",
                    "run_time": time(17, 0),
                    "trading_days_only": True,
                    "payload_json": {},
                    "misfire_grace_seconds": 3600,
                    "next_run_at": current - timedelta(minutes=1),
                    "created_by": f"research-campaign:{campaign_id}",
                    "created_at": current,
                    "updated_at": current,
                },
            ],
        )
        connection.execute(
            insert(research_campaigns).values(
                id=campaign_id,
                name="completed legacy campaign",
                status="succeeded",
                stage="completed",
                objective="preserve completed research",
                dataset="cn-legacy",
                benchmark="SH000300",
                universe="cn_all",
                recipe_id="legacy_recipe",
                research_program_id=program_id,
                dataset_identity_sha256="d" * 64,
                config_json={
                    "backtest_periods": {
                        "start": "2019-01-02",
                        "end": "2019-12-31",
                    }
                },
                state_json={},
                paper_schedule_id=payload_schedule_id,
                attempts=1,
                next_action_at=current,
                created_by="legacy-system",
                created_at=current,
                updated_at=current,
                finished_at=current,
            )
        )
        connection.execute(
            insert(schedule_runs),
            [
                {
                    "id": f"run-{payload_schedule_id}",
                    "schedule_id": payload_schedule_id,
                    "scheduled_for": current - timedelta(minutes=1),
                    "status": "pending",
                    "attempts": 0,
                    "dedupe_key": f"legacy:{payload_schedule_id}",
                    "created_at": current,
                },
                {
                    "id": f"run-{actor_schedule_id}",
                    "schedule_id": actor_schedule_id,
                    "scheduled_for": current - timedelta(minutes=1),
                    "status": "pending",
                    "attempts": 0,
                    "dedupe_key": f"legacy:{actor_schedule_id}",
                    "created_at": current,
                },
            ],
        )

    retirement = LegacyResearchRetirement(database_url)
    result = retirement.apply(actor="cutover-test")
    repeated = retirement.apply(actor="cutover-test")

    assert result["retired_campaign_ids"] == []
    assert result["disabled_schedule_ids"] == sorted(
        [payload_schedule_id, actor_schedule_id]
    )
    assert repeated["disabled_schedule_ids"] == []
    store = ScheduleStore(database_url)
    for schedule_id in (payload_schedule_id, actor_schedule_id):
        schedule = store.get(schedule_id)
        assert schedule["status"] == "paused"
        assert schedule["desired_status"] == "paused"
        with pytest.raises(ValueError, match=LEGACY_RESEARCH_SCHEDULE_ERROR):
            store.set_status(schedule_id, "active", now=current)
        with pytest.raises(ValueError, match=LEGACY_RESEARCH_SCHEDULE_ERROR):
            store.trigger_now(schedule_id, actor="operator", now=current)

    # Even direct database drift cannot make the generic materialize/claim
    # paths execute a schedule carrying either legacy ownership marker.
    with open_database(database_url).begin() as connection:
        connection.execute(
            update(schedules)
            .where(schedules.c.id.in_((payload_schedule_id, actor_schedule_id)))
            .values(status="active", desired_status="active")
        )
    assert store.materialize_due(now=current) == 0
    assert store.claim_run(now=current) is None
    assert store.get_run(f"run-{payload_schedule_id}")["status"] == "pending"
    assert {item["id"] for item in store.list()} == {
        payload_schedule_id,
        actor_schedule_id,
    }


def test_schedule_run_owned_orphan_running_job_is_cancelled_idempotently(
    database_url: str,
) -> None:
    current = datetime(2020, 1, 2, tzinfo=UTC)
    campaign_id = "completed-campaign-with-orphan-job"
    schedule_id = "completed-campaign-orphan-job-schedule"
    schedule_run_id = "completed-campaign-orphan-schedule-run"
    job_id = "orphan-legacy-campaign-job"
    with open_database(database_url).begin() as connection:
        connection.execute(
            insert(research_campaigns).values(
                id=campaign_id,
                name="completed campaign with orphan job",
                status="succeeded",
                stage="completed",
                objective="find payload-owned orphan jobs",
                dataset="cn-legacy",
                benchmark="SH000300",
                universe="cn_all",
                recipe_id="legacy_recipe",
                dataset_identity_sha256="e" * 64,
                config_json={
                    "backtest_periods": {
                        "start": "2019-01-02",
                        "end": "2019-12-31",
                    }
                },
                state_json={},
                attempts=1,
                next_action_at=current,
                created_by="legacy-system",
                created_at=current,
                updated_at=current,
                finished_at=current,
            )
        )
        connection.execute(
            insert(schedules).values(
                id=schedule_id,
                name="completed campaign orphan job schedule",
                kind="recommendation_refresh",
                status="active",
                desired_status="active",
                suspension_reason=None,
                timezone="Asia/Shanghai",
                run_time=time(16, 30),
                trading_days_only=True,
                payload_json={
                    "recommendation_portfolio_id": "legacy-portfolio"
                },
                misfire_grace_seconds=3600,
                next_run_at=current,
                created_by="legacy-controller",
                created_at=current,
                updated_at=current,
            )
        )
        connection.execute(
            update(research_campaigns)
            .where(research_campaigns.c.id == campaign_id)
            .values(paper_schedule_id=schedule_id)
        )
        # Model a scheduler process dying after JobStore.create committed but
        # before schedule_runs.job_id was attached. Retirement must recover
        # ownership from the durable schedule-run idempotency contract.
        connection.execute(
            insert(jobs).values(
                id=job_id,
                kind="recommendation_refresh",
                idempotency_key=f"schedule-run:{schedule_run_id}",
                status="running",
                payload_json={
                    "recommendation_portfolio_id": "legacy-portfolio",
                    "schedule_run_id": schedule_run_id,
                },
                attempts=1,
                max_attempts=1,
                created_at=current,
                started_at=current,
            )
        )
        connection.execute(
            insert(schedule_runs).values(
                id=schedule_run_id,
                schedule_id=schedule_id,
                scheduled_for=current,
                status="running",
                attempts=1,
                dedupe_key="legacy:orphan-job",
                created_at=current,
            )
        )

    retirement = LegacyResearchRetirement(database_url)
    first = retirement.apply(actor="cutover-test")
    with open_database(database_url).connect() as connection:
        after_first = connection.execute(
            select(jobs.c.status, jobs.c.cancel_requested_at).where(jobs.c.id == job_id)
        ).one()
    second = retirement.apply(actor="cutover-test")
    with open_database(database_url).connect() as connection:
        after_second = connection.execute(
            select(jobs.c.status, jobs.c.cancel_requested_at).where(jobs.c.id == job_id)
        ).one()

    assert first["exclusive_job_ids"] == [job_id]
    assert first["cancelled_job_ids"] == [job_id]
    assert second["cancelled_job_ids"] == []
    assert after_first.status == "running"
    assert after_first.cancel_requested_at is not None
    assert after_second.cancel_requested_at == after_first.cancel_requested_at
    with open_database(database_url).connect() as connection:
        assert connection.scalar(
            select(research_campaigns.c.status).where(
                research_campaigns.c.id == campaign_id
            )
        ) == "succeeded"


def _seed_claimed_schedule(
    database_url: str,
    *,
    current: datetime,
) -> tuple[str, str]:
    schedule_id = "claimed-schedule"
    run_id = "claimed-schedule-run"
    with open_database(database_url).begin() as connection:
        connection.execute(
            insert(schedules).values(
                id=schedule_id,
                name="claimed schedule",
                kind="incremental_sync",
                status="active",
                desired_status="active",
                suspension_reason=None,
                timezone="Asia/Shanghai",
                run_time=time(15, 30),
                trading_days_only=False,
                payload_json={"profile": "core"},
                misfire_grace_seconds=3600,
                next_run_at=current + timedelta(days=1),
                created_by="legacy-controller",
                created_at=current,
                updated_at=current,
            )
        )
        connection.execute(
            insert(schedule_runs).values(
                id=run_id,
                schedule_id=schedule_id,
                scheduled_for=current,
                status="running",
                attempts=1,
                lease_until=current + timedelta(minutes=2),
                dedupe_key=f"legacy:{run_id}",
                created_at=current,
            )
        )
    return schedule_id, run_id


def _link_completed_campaign_to_schedule(
    database_url: str,
    *,
    schedule_id: str,
    current: datetime,
) -> None:
    with open_database(database_url).begin() as connection:
        connection.execute(
            insert(research_campaigns).values(
                id="completed-campaign-with-claimed-run",
                name="completed campaign with claimed run",
                status="succeeded",
                stage="completed",
                objective="exercise the claimed-run retirement interleaving",
                dataset="cn-legacy",
                benchmark="SH000300",
                universe="cn_all",
                recipe_id="legacy_recipe",
                dataset_identity_sha256="f" * 64,
                config_json={},
                state_json={},
                paper_schedule_id=schedule_id,
                attempts=1,
                next_action_at=current,
                created_by="legacy-system",
                created_at=current,
                updated_at=current,
                finished_at=current,
            )
        )


def test_dispatch_and_retirement_share_lock_and_leave_no_live_legacy_job(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = datetime(2025, 1, 2, 7, 30, tzinfo=UTC)
    schedule_id, run_id = _seed_claimed_schedule(database_url, current=current)
    store = ScheduleStore(database_url)
    engine = SchedulerEngine(_scheduler_settings(database_url, tmp_path))
    dispatch_entered = Event()
    allow_enqueue = Event()
    retirement_started = Event()
    retirement_finished = Event()
    scheduler_errors: list[BaseException] = []
    retirement_errors: list[BaseException] = []
    retirement_result: dict[str, object] = {}

    def enqueue_after_interleaving(
        run: dict[str, object], scheduled_for: datetime
    ) -> dict[str, object]:
        del scheduled_for
        dispatch_entered.set()
        if not allow_enqueue.wait(timeout=10):
            raise TimeoutError("test did not release the guarded enqueue")
        return engine.jobs.create(
            "recommendation_refresh",
            {"schedule_run_id": str(run["id"])},
            tmp_path / "legacy-dispatch.log",
            dedupe_active_kind=False,
            idempotency_key=f"schedule-run:{run['id']}",
        )

    monkeypatch.setattr(engine, "_enqueue_incremental", enqueue_after_interleaving)

    def process_claimed_run() -> None:
        try:
            engine._process_run(store.get_run(run_id), current)
        except BaseException as exc:  # pragma: no cover - asserted below
            scheduler_errors.append(exc)

    def retire_legacy_control_plane() -> None:
        retirement_started.set()
        try:
            retirement_result.update(
                LegacyResearchRetirement(database_url).apply(actor="cutover-test")
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            retirement_errors.append(exc)
        finally:
            retirement_finished.set()

    scheduler_thread = Thread(target=process_claimed_run, daemon=True)
    scheduler_thread.start()
    assert dispatch_entered.wait(timeout=10)
    # Freeze the adversarial interleaving: dispatch validated the row while it
    # was ordinary, then historical ownership becomes visible before the job
    # is created. Retirement must wait for dispatch and cancel the linked job.
    _link_completed_campaign_to_schedule(
        database_url, schedule_id=schedule_id, current=current
    )
    retirement_thread = Thread(target=retire_legacy_control_plane, daemon=True)
    retirement_thread.start()
    assert retirement_started.wait(timeout=10)
    assert not retirement_finished.wait(timeout=0.5)

    allow_enqueue.set()
    scheduler_thread.join(timeout=10)
    retirement_thread.join(timeout=10)

    assert not scheduler_thread.is_alive()
    assert not retirement_thread.is_alive()
    assert scheduler_errors == []
    assert retirement_errors == []
    run = store.get_run(run_id)
    assert run["status"] == "enqueued"
    job = JobStore(database_url).get(str(run["job_id"]))
    assert job["status"] == "cancelled"
    assert retirement_result["disabled_schedule_ids"] == [schedule_id]
    assert retirement_result["cancelled_job_ids"] == [job["id"]]


def test_retirement_wins_before_dispatch_and_normal_schedule_still_dispatches(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = datetime(2025, 1, 2, 7, 30, tzinfo=UTC)
    legacy_schedule_id, legacy_run_id = _seed_claimed_schedule(
        database_url, current=current
    )
    _link_completed_campaign_to_schedule(
        database_url, schedule_id=legacy_schedule_id, current=current
    )
    store = ScheduleStore(database_url)
    engine = SchedulerEngine(_scheduler_settings(database_url, tmp_path))
    LegacyResearchRetirement(database_url).apply(actor="cutover-test")
    enqueue_calls: list[str] = []

    def enqueue(
        run: dict[str, object], scheduled_for: datetime
    ) -> dict[str, object]:
        del scheduled_for
        enqueue_calls.append(str(run["id"]))
        return engine.jobs.create(
            "recommendation_refresh",
            {"schedule_run_id": str(run["id"])},
            tmp_path / f"{run['id']}.log",
            dedupe_active_kind=False,
            idempotency_key=f"schedule-run:{run['id']}",
        )

    monkeypatch.setattr(engine, "_enqueue_incremental", enqueue)
    engine._process_run(store.get_run(legacy_run_id), current)

    legacy_run = store.get_run(legacy_run_id)
    assert legacy_run["status"] == "skipped"
    assert "retired before dispatch" in str(legacy_run["message"])
    assert enqueue_calls == []
    assert JobStore(database_url).list() == []

    normal_schedule = store.create(
        name="ordinary incremental schedule",
        kind="incremental_sync",
        timezone="Asia/Shanghai",
        run_time=time(15, 30),
        trading_days_only=False,
        payload={"profile": "core"},
        misfire_grace_seconds=3600,
        actor="operator",
        now=current,
    )
    pending = store.trigger_now(str(normal_schedule["id"]), actor="operator", now=current)
    claimed = store.claim_run(now=current)
    assert claimed is not None and claimed["id"] == pending["id"]
    engine._process_run(claimed, current)

    ordinary_run = store.get_run(str(claimed["id"]))
    assert ordinary_run["status"] == "enqueued"
    assert enqueue_calls == [str(claimed["id"])]
    assert JobStore(database_url).get(str(ordinary_run["job_id"]))["status"] == "queued"
