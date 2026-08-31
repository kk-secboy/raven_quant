from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select, update

from quant_data.database import (
    autopilot_branches,
    research_events,
    research_runs,
)
from quant_data.database import (
    jobs as job_rows,
)
from quant_platform.autopilot import (
    AutopilotController,
    AutopilotStore,
    _derived_branch_status,
)
from quant_platform.job_store import JobStore
from quant_platform.research_store import ResearchStore


@pytest.mark.no_database
def test_controller_delegates_failed_branch_retry_to_atomic_store() -> None:
    calls: list[tuple[str, str]] = []
    controller = AutopilotController.__new__(AutopilotController)
    controller.store = SimpleNamespace(
        retry_failed_branch=lambda branch_id, actor: calls.append((branch_id, actor)) or True
    )

    assert controller._retry_failed_branch(
        {
            "id": "factor-branch",
            "status": "failed",
            "research_run_id": "factor-run",
            "job_id": "stale-generation-job",
            "details": {},
        }
    )
    assert calls == [("factor-branch", "autopilot")]


@pytest.mark.no_database
@pytest.mark.parametrize(
    "branch",
    [
        {"id": "not-failed", "status": "queued", "details": {}},
        {"id": "spent", "status": "failed", "details": {"retry_count": 1}},
        {"id": "", "status": "failed", "details": {}},
    ],
)
def test_controller_does_not_open_atomic_retry_for_ineligible_branch(
    branch: dict[str, object],
) -> None:
    controller = AutopilotController.__new__(AutopilotController)
    controller.store = SimpleNamespace(
        retry_failed_branch=lambda *_args, **_kwargs: pytest.fail(
            "ineligible branch reached retry transaction"
        )
    )

    assert controller._retry_failed_branch(branch) is False


@pytest.mark.no_database
def test_controller_does_not_rewrite_run_when_atomic_retry_raises() -> None:
    controller = AutopilotController.__new__(AutopilotController)

    def fail_retry(_branch_id: str, *, actor: str) -> bool:
        assert actor == "autopilot"
        raise RuntimeError("transaction rolled back")

    controller.store = SimpleNamespace(retry_failed_branch=fail_retry)
    # The old compensation path called research.mark_run(..., failed) here.
    # Deliberately omit a research store: propagating the transaction error is
    # the only permitted behavior.
    with pytest.raises(RuntimeError, match="rolled back"):
        controller._retry_failed_branch(
            {"id": "branch", "status": "failed", "details": {}}
        )


@pytest.mark.no_database
def test_current_failed_job_cannot_be_hidden_by_stale_queued_run() -> None:
    assert _derived_branch_status("queued", "failed") == "failed"
    assert _derived_branch_status("running", "cancelled") == "failed"
    assert _derived_branch_status("succeeded", "failed") == "succeeded"


def _seed_failed_branch(
    database_url: str,
    tmp_path: Path,
    *,
    run_status: str = "failed",
    current_job_status: str = "failed",
    current_job_kind: str = "factor_evaluate",
) -> dict[str, str]:
    autopilot = AutopilotStore(database_url)
    cycle = autopilot.ensure_cycle(
        {
            "name": "retry-dataset",
            "lineage_id": "b" * 64,
            "end_date": "2026-08-31",
            "provenance": {"dataset_identity_sha256": "a" * 64},
        },
        config_revision=1,
    )
    jobs = JobStore(database_url)
    parent = jobs.create(
        "rdagent",
        {"stage": "generation"},
        tmp_path / "parent.log",
        dedupe_active_kind=False,
    )
    jobs.finish(parent["id"], exit_code=0, result={"generated": True})
    current = jobs.create(
        current_job_kind,
        {"stage": "independent_evaluation"},
        tmp_path / "current.log",
        dedupe_active_kind=False,
    )
    if current_job_status == "failed":
        jobs.finish(current["id"], exit_code=1, error="evaluation failed")
    elif current_job_status == "cancelled":
        jobs.request_cancel(current["id"])
    elif current_job_status != "queued":
        with jobs.engine.begin() as connection:
            connection.execute(
                update(job_rows)
                .where(job_rows.c.id == current["id"])
                .values(status=current_job_status)
            )

    research = ResearchStore(database_url)
    run = research.create_run(
        kind="factor",
        objective="verify atomic retry",
        dataset="retry-dataset",
        requested_by="test",
        budget={},
        config={},
        artifact_path=tmp_path / "run",
    )
    research.attach_job(run["id"], current["id"])
    if run_status != "queued":
        research.mark_run(run["id"], run_status, actor="test", error="run failed")

    branch = autopilot.create_branch(
        cycle["id"],
        scenario="fin_factor",
        scope_key="daily",
        research_run_id=run["id"],
        job_id=parent["id"],
        details={"branch_kind": "factor"},
    )
    with autopilot.engine.begin() as connection:
        connection.execute(
            update(autopilot_branches)
            .where(autopilot_branches.c.id == branch["id"])
            .values(status="failed", error="branch failed")
        )
    return {
        "branch_id": str(branch["id"]),
        "run_id": str(run["id"]),
        "parent_job_id": str(parent["id"]),
        "current_job_id": str(current["id"]),
    }


def test_atomic_retry_uses_current_evaluator_instead_of_stale_parent_job(
    tmp_path: Path, database_url: str
) -> None:
    ids = _seed_failed_branch(database_url, tmp_path)
    store = AutopilotStore(database_url)

    assert store.retry_failed_branch(ids["branch_id"])

    branch = store.get_branch(ids["branch_id"])
    run = ResearchStore(database_url).get_run(ids["run_id"])
    jobs = JobStore(database_url)
    assert branch["status"] == "queued"
    assert branch["details"]["retry_count"] == 1
    assert branch["details"]["retried_job_id"] == ids["current_job_id"]
    assert branch["details"]["retry_job_binding"] == "research_run_current"
    assert run["status"] == "queued"
    assert jobs.get(ids["current_job_id"])["status"] == "queued"
    assert jobs.get(ids["parent_job_id"])["status"] == "succeeded"
    with store.engine.connect() as connection:
        requeued = connection.scalar(
            select(func.count())
            .select_from(research_events)
            .where(
                research_events.c.research_run_id == ids["run_id"],
                research_events.c.event_type == "run.requeued",
            )
        )
    assert int(requeued or 0) == 1


def test_reconcile_exposes_failed_current_evaluator_for_atomic_retry(
    tmp_path: Path, database_url: str
) -> None:
    ids = _seed_failed_branch(database_url, tmp_path, run_status="queued")
    store = AutopilotStore(database_url)
    with store.engine.begin() as connection:
        connection.execute(
            update(autopilot_branches)
            .where(autopilot_branches.c.id == ids["branch_id"])
            .values(status="queued", error=None)
        )

    assert store.reconcile() >= 1
    assert store.get_branch(ids["branch_id"])["status"] == "failed"
    assert store.retry_failed_branch(ids["branch_id"])
    assert store.get_branch(ids["branch_id"])["status"] == "queued"
    assert JobStore(database_url).get(ids["current_job_id"])["status"] == "queued"


def test_atomic_retry_falls_back_to_branch_job_for_legacy_unattached_run(
    tmp_path: Path, database_url: str
) -> None:
    ids = _seed_failed_branch(database_url, tmp_path)
    store = AutopilotStore(database_url)
    with store.engine.begin() as connection:
        connection.execute(
            update(research_runs)
            .where(research_runs.c.id == ids["run_id"])
            .values(job_id=None)
        )
        connection.execute(
            update(job_rows)
            .where(job_rows.c.id == ids["parent_job_id"])
            .values(status="failed", error="legacy generation failed")
        )

    assert store.retry_failed_branch(ids["branch_id"])

    branch = store.get_branch(ids["branch_id"])
    jobs = JobStore(database_url)
    assert branch["details"]["retried_job_id"] == ids["parent_job_id"]
    assert branch["details"]["retry_job_binding"] == "branch_provenance"
    assert jobs.get(ids["parent_job_id"])["status"] == "queued"
    assert jobs.get(ids["current_job_id"])["status"] == "failed"


def test_concurrent_atomic_retry_has_exactly_one_winner(
    tmp_path: Path, database_url: str
) -> None:
    ids = _seed_failed_branch(database_url, tmp_path)
    contenders = Barrier(2)

    def retry() -> bool:
        contender = AutopilotStore(database_url)
        contenders.wait(timeout=10)
        return contender.retry_failed_branch(ids["branch_id"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = [future.result() for future in [executor.submit(retry) for _ in range(2)]]

    assert sorted(outcomes) == [False, True]
    store = AutopilotStore(database_url)
    branch = store.get_branch(ids["branch_id"])
    assert branch["status"] == "queued"
    assert branch["details"]["retry_count"] == 1
    with store.engine.connect() as connection:
        requeued = connection.scalar(
            select(func.count())
            .select_from(research_events)
            .where(
                research_events.c.research_run_id == ids["run_id"],
                research_events.c.event_type == "run.requeued",
            )
        )
    assert int(requeued or 0) == 1


def test_atomic_retry_state_mismatch_changes_nothing(
    tmp_path: Path, database_url: str
) -> None:
    ids = _seed_failed_branch(
        database_url,
        tmp_path,
        run_status="queued",
        current_job_status="running",
    )
    store = AutopilotStore(database_url)

    assert store.retry_failed_branch(ids["branch_id"]) is False

    assert store.get_branch(ids["branch_id"])["status"] == "failed"
    assert ResearchStore(database_url).get_run(ids["run_id"])["status"] == "queued"
    assert JobStore(database_url).get(ids["current_job_id"])["status"] == "running"


def test_atomic_retry_error_rolls_back_without_marking_queued_run_failed(
    tmp_path: Path, database_url: str
) -> None:
    ids = _seed_failed_branch(
        database_url,
        tmp_path,
        run_status="queued",
        current_job_kind="strategy_backtest",
    )
    store = AutopilotStore(database_url)

    with pytest.raises(ValueError, match="formal final-test"):
        store.retry_failed_branch(ids["branch_id"])

    assert store.get_branch(ids["branch_id"])["status"] == "failed"
    assert ResearchStore(database_url).get_run(ids["run_id"])["status"] == "queued"
    assert JobStore(database_url).get(ids["current_job_id"])["status"] == "failed"
