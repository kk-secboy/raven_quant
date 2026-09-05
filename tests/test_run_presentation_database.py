from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import event, insert, select, update

from quant_data.database import jobs, open_database, research_runs
from quant_platform.api import _public_rdagent_run
from quant_platform.research_store import ResearchStore
from quant_platform.run_presentation import RunPresentationStore


def test_research_page_and_exact_successor_projection_preserve_audit(database_url):
    engine = open_database(database_url)
    store = RunPresentationStore(engine)
    now = datetime(2026, 9, 6, tzinfo=UTC)
    active_id, history_id = "a" * 32, "b" * 32
    proposal_id, parameter_id, cancelled_id = "c" * 32, "d" * 32, "e" * 32

    def job(job_id, kind, status, minute, payload):
        return {
            "id": job_id, "kind": kind, "status": status, "payload_json": payload,
            "created_at": now + timedelta(minutes=minute), "attempts": 1, "max_attempts": 1,
        }

    with engine.begin() as connection:
        connection.execute(insert(jobs), [
            job(proposal_id, "rdagent_run", "succeeded", 0, {"research_run_id": active_id}),
            job(parameter_id, "parameter_experiment", "running", 1, {
                "research_run_id": active_id, "strategy_competition_stage": "policy_only",
                "parameter_experiment_id": "f" * 32,
            }),
            job(cancelled_id, "parameter_experiment", "cancelled", 0, {}),
            # Neither a newer terminal sibling nor an unrelated active run may hide the owner.
            job("1" * 32, "parameter_experiment", "failed", 2, {"research_run_id": active_id}),
            job("2" * 32, "parameter_experiment", "running", 3, {"research_run_id": "3" * 32}),
        ])
        for run_id, status, job_id, minute in (
            (active_id, "running", proposal_id, 0),
            (history_id, "failed", cancelled_id, 1),
        ):
            connection.execute(insert(research_runs).values(
                id=run_id, job_id=job_id, kind="fin_strategy", objective="Exact owner only",
                dataset="test-daily", status=status, requested_by="test",
                budget_json={}, config_json={}, runtime_json={},
                error="old private diagnostic" if status == "failed" else None,
                created_at=now + timedelta(minutes=minute), updated_at=now,
            ))

    active, total = store.list_runs(limit=1, offset=0, status_group="active")
    assert total == 1 and [row["id"] for row in active] == [active_id]
    assert "budget_json" not in active[0] and active[0]["budget"] == {}
    canonical = ResearchStore(database_url)
    assert active[0] == canonical.get_run(active_id)
    canonical.engine.dispose()
    linked = store.linked_executions(active)
    assert linked[active_id]["id"] == parameter_id
    public = _public_rdagent_run(active[0], linked_job=linked[active_id])
    assert public["job_id"] == proposal_id
    assert public["status"] == "running"
    assert public["presentation"]["execution_phase"] == "policy_only"
    history, total = store.list_runs(limit=20, offset=0, status_group="history")
    assert total == 1 and history[0]["id"] == history_id
    public_history = _public_rdagent_run(
        history[0], linked_job=store.linked_executions(history)[history_id]
    )
    assert public_history["presentation"]["display_status"] == "interrupted"
    assert public_history["status"] == "failed"
    all_page, total = store.list_runs(limit=1, offset=1, status_group=None)
    assert total == 2 and all_page[0]["id"] == active_id
    assert store.list_runs(limit=1, offset=99, status_group="history") == ([], 1)
    # A formal successor uses a different, equally exact payload owner field.
    with engine.begin() as connection:
        connection.execute(update(jobs).where(jobs.c.id == parameter_id).values(status="succeeded"))
        connection.execute(insert(jobs).values(**job(
            "4" * 32, "strategy_backtest", "queued", 4,
            {"fin_strategy_research_run_id": active_id},
        )))
    formal = store.linked_executions(active)[active_id]
    assert formal["id"] == "4" * 32
    assert _public_rdagent_run(active[0], linked_job=formal)["presentation"][
        "execution_phase"
    ] == "formal_final_oos"
    with engine.connect() as connection:
        persisted = connection.execute(
            select(research_runs.c.status, research_runs.c.error)
            .where(research_runs.c.id == history_id)
        ).one()
        assert tuple(persisted) == ("failed", "old private diagnostic")
    engine.dispose()


def test_research_count_and_page_share_repeatable_snapshot_during_transition(database_url):
    engine = open_database(database_url)
    store = RunPresentationStore(engine)
    now = datetime(2026, 9, 6, tzinfo=UTC)
    run_id = "a" * 32
    with engine.begin() as connection:
        connection.execute(insert(research_runs).values(
            id=run_id, kind="fin_strategy", objective="Snapshot test", dataset="test-daily",
            status="running", requested_by="test", budget_json={}, config_json={},
            runtime_json={}, created_at=now, updated_at=now,
        ))
    transitioned = False

    def transition_after_count(_conn, _cursor, statement, _parameters, _context, _executemany):
        nonlocal transitioned
        if not transitioned and statement.startswith("SELECT count(*)"):
            transitioned = True
            with engine.begin() as other:
                other.execute(
                    update(research_runs)
                    .where(research_runs.c.id == run_id)
                    .values(status="failed")
                )

    event.listen(engine, "after_cursor_execute", transition_after_count)
    try:
        rows, total = store.list_runs(limit=20, offset=0, status_group="active")
    finally:
        event.remove(engine, "after_cursor_execute", transition_after_count)
    assert transitioned
    assert total == 1 and len(rows) == 1
    assert rows[0]["status"] == "running"
    with engine.connect() as connection:
        assert connection.execute(
            select(research_runs.c.status).where(research_runs.c.id == run_id)
        ).scalar_one() == "failed"
    engine.dispose()


def test_recent_history_sorts_by_completion_time_before_creation_time(database_url):
    engine = open_database(database_url)
    now = datetime(2026, 9, 6, tzinfo=UTC)
    with engine.begin() as connection:
        for run_id, created_at, finished_at in (
            ("a" * 32, now - timedelta(days=2), now),
            ("b" * 32, now - timedelta(days=1), now - timedelta(hours=1)),
            ("c" * 32, now - timedelta(hours=2), None),
        ):
            connection.execute(insert(research_runs).values(
                id=run_id, kind="fin_strategy", objective="Recent result test", dataset="test-day",
                status="failed", requested_by="test", budget_json={}, config_json={},
                runtime_json={}, created_at=created_at, updated_at=now, finished_at=finished_at,
            ))
    store = RunPresentationStore(engine)
    recent, total = store.list_runs(limit=2, offset=0, status_group="history")
    assert total == 3 and [row["id"] for row in recent] == ["a" * 32, "b" * 32]
    tail, total = store.list_runs(limit=2, offset=2, status_group="history")
    assert total == 3 and [row["id"] for row in tail] == ["c" * 32]
    all_runs, _ = store.list_runs(limit=3, offset=0, status_group=None)
    assert [row["id"] for row in all_runs] == ["c" * 32, "b" * 32, "a" * 32]
    engine.dispose()
