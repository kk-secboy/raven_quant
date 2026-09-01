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
from quant_platform.platform_model_tournament import PlatformModelTournamentService
from quant_platform.research_store import ResearchStore
from quant_platform.research_tournament import ResearchTournamentStore


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
def test_active_model_tournament_falls_back_for_legacy_cycle_without_pointer() -> None:
    calls: list[str] = []
    expected = {"id": "legacy-primary-tournament"}
    controller = AutopilotController.__new__(AutopilotController)
    controller.tournaments = SimpleNamespace(
        get_for_cycle=lambda cycle_id: calls.append(cycle_id) or expected,
        get_tournament=lambda _tournament_id: pytest.fail(
            "legacy cycle must use the primary tournament lookup"
        ),
    )

    assert controller._active_model_tournament({"id": "cycle-1", "state": {}}) is expected
    assert calls == ["cycle-1"]


@pytest.mark.no_database
def test_operational_successor_discards_old_model_results_only() -> None:
    source = {
        "dataset_end_date": "2026-08-31",
        "factor_sota_status": "ready",
        "screen_selected_feature_set_ids": ["alpha158", "alpha360"],
        "feature_screen_evidence": {"old": True},
        "model_champions": [{"old": True}],
        "model_ensemble_status": "complete",
        "prediction_champion": {"old": True},
        "fin_quant_status": "ready",
        "research_tournament_status": "succeeded",
    }

    clean = AutopilotController._without_model_tournament_results(source)

    assert clean == {
        "dataset_end_date": "2026-08-31",
        "factor_sota_status": "ready",
    }


@pytest.mark.no_database
@pytest.mark.parametrize(
    ("source_status", "branch_status", "run_status", "job_status", "expected"),
    [
        ("running", "blocked", "blocked", "failed", True),
        ("running", "running", "blocked", "failed", False),
        ("running", "blocked", "evaluating", "failed", False),
        ("running", "blocked", "blocked", "queued", False),
        ("succeeded", "blocked", "blocked", "failed", False),
    ],
)
def test_operational_successor_waits_for_every_durable_source_owner(
    source_status: str,
    branch_status: str,
    run_status: str,
    job_status: str,
    expected: bool,
) -> None:
    controller = AutopilotController.__new__(AutopilotController)
    controller.research = SimpleNamespace(
        get_run=lambda run_id: {
            "id": run_id,
            "status": run_status,
            "job_id": "current-job",
        }
    )
    controller.jobs = SimpleNamespace(
        get=lambda job_id: {"id": job_id, "status": job_status}
    )
    cycle = {
        "branches": [
            {
                "id": "source-branch",
                "status": branch_status,
                "research_run_id": "source-run",
                "job_id": "generation-job",
                "details": {"tournament_id": "source-tournament"},
            }
        ]
    }
    source = {"id": "source-tournament", "status": source_status}

    assert controller._operational_source_owners_terminal(cycle, source) is expected


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
    requested_by: str = "test",
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
        requested_by=requested_by,
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
    ids = _seed_failed_branch(
        database_url,
        tmp_path,
        run_status="queued",
        requested_by="autopilot",
    )
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


def test_scheduler_projects_terminal_current_job_before_atomic_retry(
    tmp_path: Path, database_url: str
) -> None:
    ids = _seed_failed_branch(
        database_url,
        tmp_path,
        run_status="queued",
        requested_by="autopilot",
    )
    research = ResearchStore(database_url)
    with research.engine.begin() as connection:
        connection.execute(
            update(job_rows)
            .where(job_rows.c.id == ids["current_job_id"])
            .values(
                payload_json={
                    "stage": "independent_evaluation",
                    "research_run_id": ids["run_id"],
                }
            )
        )

    assert research.reconcile_terminal_autopilot_jobs() == 1
    assert research.reconcile_terminal_autopilot_jobs() == 0
    assert research.get_run(ids["run_id"])["status"] == "failed"

    store = AutopilotStore(database_url)
    store.reconcile()
    assert store.get_branch(ids["branch_id"])["status"] == "failed"
    assert store.retry_failed_branch(ids["branch_id"])
    assert research.get_run(ids["run_id"])["status"] == "queued"
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


@pytest.mark.parametrize(
    ("terminal_mode", "expected_owner"),
    [
        ("blocked", "blocked/dataset_unavailable"),
        ("superseded", "paused/superseded"),
    ],
)
def test_terminal_cycle_reconciliation_releases_setup_only_model_run(
    tmp_path: Path,
    database_url: str,
    terminal_mode: str,
    expected_owner: str,
) -> None:
    autopilot = AutopilotStore(database_url)
    cycle = autopilot.ensure_cycle(
        {
            "name": "retired-model-dataset",
            "lineage_id": "b" * 64,
            "end_date": "2026-08-31",
            "provenance": {"dataset_identity_sha256": "a" * 64},
        },
        config_revision=1,
    )
    research = ResearchStore(database_url)
    kind = "platform_model_feature_screen_deadbeef_short_1_5d"
    run = research.create_run(
        kind=kind,
        objective="recover interrupted platform setup",
        dataset="retired-model-dataset",
        requested_by=f"autopilot:{cycle['id']}",
        budget={},
        config={
            "contract_version": "platform-model-tournament-run-v2-horizon",
            "autopilot_cycle_id": cycle["id"],
        },
        artifact_path=tmp_path / "model-run",
    )

    # An active owner is resumable. The scheduler must not guess that the
    # create-before-job gap is a terminal failure.
    assert research.reconcile_terminal_autopilot_initializations() == 0
    assert research.get_run(run["id"])["status"] == "queued"

    if terminal_mode == "superseded":
        autopilot.supersede_research_cycle(
            cycle["id"],
            replacement_dataset={
                "name": "successor-model-dataset",
                "provenance": {"dataset_identity_sha256": "c" * 64},
            },
        )
    else:
        autopilot.set_cycle_state(
            cycle["id"],
            state={**cycle["state"], "result": "bound_dataset_unavailable"},
            stage="dataset_unavailable",
            status="blocked",
            error="bound Qlib dataset is unavailable",
            finished=True,
        )
    assert research.reconcile_terminal_autopilot_initializations() == 1
    assert research.reconcile_terminal_autopilot_initializations() == 0
    terminal = research.get_run(run["id"])
    assert terminal["status"] == "failed"
    assert f"owning autopilot cycle is {expected_owner}" in terminal["error"]
    with research.engine.connect() as connection:
        events = list(
            connection.scalars(
                select(research_events.c.event_type).where(
                    research_events.c.research_run_id == run["id"]
                )
            )
        )
    assert events == ["run.created", "run.failed"]

    # History is retained, while the partial unique index no longer prevents
    # the next immutable cycle from using the same governed lane kind.
    replacement = research.create_run(
        kind=kind,
        objective="new immutable model lane",
        dataset="successor-model-dataset",
        requested_by="autopilot:successor-cycle",
        budget={},
        config={},
        artifact_path=tmp_path / "replacement-model-run",
    )
    assert replacement["status"] == "queued"
    assert research.get_run(run["id"])["status"] == "failed"


def test_terminal_cycle_cancels_unattached_payload_job_before_reconciling_run(
    tmp_path: Path, database_url: str
) -> None:
    autopilot = AutopilotStore(database_url)
    cycle = autopilot.ensure_cycle(
        {
            "name": "job-gap-dataset",
            "lineage_id": "b" * 64,
            "end_date": "2026-08-31",
            "provenance": {"dataset_identity_sha256": "c" * 64},
        },
        config_revision=1,
    )
    research = ResearchStore(database_url)
    run = research.create_run(
        kind="platform_model_feature_screen_payload_short_1_5d",
        objective="preserve durable evaluator authority",
        dataset="job-gap-dataset",
        requested_by=f"autopilot:{cycle['id']}",
        budget={},
        config={
            "contract_version": "platform-model-tournament-run-v2-horizon",
            "autopilot_cycle_id": cycle["id"],
        },
        artifact_path=tmp_path / "payload-run",
    )
    payload = {"research_run_id": run["id"], "contract": "exact"}
    job = JobStore(database_url).create(
        "model_evaluate",
        payload,
        tmp_path / "model.log",
        dedupe_active_kind=False,
        idempotency_key="platform-model:payload-gap",
    )
    autopilot.set_cycle_state(
        cycle["id"],
        state={**cycle["state"], "result": "blocked"},
        stage="research_blocked",
        status="blocked",
        error="test terminal",
        finished=True,
    )

    assert research.reconcile_terminal_autopilot_initializations() == 0
    assert research.get_run(run["id"])["status"] == "queued"

    service = PlatformModelTournamentService.__new__(
        PlatformModelTournamentService
    )
    service.engine = research.engine
    service.research = research
    service.jobs = JobStore(database_url)
    with pytest.raises(ValueError, match="no longer has an active owner"):
        service._adopt_exact_unattached_job(
            cycle_id=cycle["id"],
            research_run_id=run["id"],
            kind="model_evaluate",
            payload=payload,
        )
    assert JobStore(database_url).get(job["id"])["status"] == "cancelled"
    assert research.get_run(run["id"])["job_id"] is None
    assert research.reconcile_terminal_autopilot_initializations() == 1
    assert research.get_run(run["id"])["status"] == "failed"
    with research.engine.connect() as connection:
        assert int(
            connection.scalar(
                select(func.count())
                .select_from(job_rows)
                .where(
                    job_rows.c.payload_json["research_run_id"].as_string()
                    == run["id"]
                )
            )
            or 0
        ) == 1


def test_active_cycle_atomically_adopts_exact_unattached_payload_job(
    tmp_path: Path, database_url: str
) -> None:
    autopilot = AutopilotStore(database_url)
    cycle = autopilot.ensure_cycle(
        {
            "name": "active-job-gap-dataset",
            "lineage_id": "d" * 64,
            "end_date": "2026-08-31",
            "provenance": {"dataset_identity_sha256": "e" * 64},
        },
        config_revision=1,
    )
    research = ResearchStore(database_url)
    run = research.create_run(
        kind="platform_model_feature_screen_active_short_1_5d",
        objective="adopt exact interrupted evaluator",
        dataset="active-job-gap-dataset",
        requested_by=f"autopilot:{cycle['id']}",
        budget={},
        config={
            "contract_version": "platform-model-tournament-run-v2-horizon",
            "autopilot_cycle_id": cycle["id"],
        },
        artifact_path=tmp_path / "active-model-run",
    )
    payload = {"research_run_id": run["id"], "contract": "exact-active"}
    job = JobStore(database_url).create(
        "model_evaluate",
        payload,
        tmp_path / "active-model.log",
        dedupe_active_kind=False,
        idempotency_key="platform-model:active-payload-gap",
    )
    service = PlatformModelTournamentService.__new__(PlatformModelTournamentService)
    service.engine = research.engine
    service.research = research
    service.jobs = JobStore(database_url)

    adopted = service._adopt_exact_unattached_job(
        cycle_id=cycle["id"],
        research_run_id=run["id"],
        kind="model_evaluate",
        payload=payload,
    )

    assert adopted["id"] == job["id"]
    assert research.get_run(run["id"])["job_id"] == job["id"]
    assert JobStore(database_url).get(job["id"])["status"] == "queued"


def _seed_operational_remediation_source(
    database_url: str,
    tmp_path: Path,
) -> tuple[dict, dict, dict, dict, dict[str, str]]:
    autopilot = AutopilotStore(database_url)
    cycle = autopilot.ensure_cycle(
        {
            "name": "resource-remediation-dataset",
            "lineage_id": "d" * 64,
            "end_date": "2026-08-31",
            "provenance": {"dataset_identity_sha256": "e" * 64},
        },
        config_revision=1,
    )
    tournaments = ResearchTournamentStore(database_url)
    source = tournaments.ensure_preregistered(
        cycle_id=str(cycle["id"]),
        dataset_identity_sha256="e" * 64,
    )
    screen = [
        item
        for item in source["trials"]
        if (item.get("spec") or {}).get("round") == "feature_screen"
    ]
    resource_trial, operational_trial, inherited_trial = screen[:3]
    tournaments.transition_trial(str(resource_trial["id"]), "queued")
    tournaments.transition_trial(str(resource_trial["id"]), "running")
    resource_trial = tournaments.transition_trial(
        str(resource_trial["id"]),
        "failed",
        candidate_id="source-resource-candidate",
        metrics={"reason_code": "resource_blocked"},
        evidence={
            "contract_version": "model-feature-screen-failure-v1",
            "investment_hypothesis_rejected": False,
            "error": "memory budget exceeded",
        },
    )
    tournaments.transition_trial(str(operational_trial["id"]), "queued")
    tournaments.transition_trial(str(operational_trial["id"]), "running")
    operational_trial = tournaments.transition_trial(
        str(operational_trial["id"]),
        "failed",
        candidate_id="source-operational-candidate",
        metrics={"reason_code": "screen_execution_failed"},
        # Historical PortAna failures carried this incorrect marker.  The
        # absence of computed performance metrics still makes them operational.
        evidence={
            "contract_version": "model-feature-screen-failure-v1",
            "investment_hypothesis_rejected": True,
            "error": "PortAnaRecord execution failed before metrics",
        },
    )
    tournaments.transition_trial(str(inherited_trial["id"]), "queued")
    tournaments.transition_trial(str(inherited_trial["id"]), "running")
    inherited_trial = tournaments.transition_trial(
        str(inherited_trial["id"]),
        "passed",
        candidate_id="source-passed-candidate",
        metrics={"cells": [{"gate": "passed"}]},
        evidence={"contract_version": "source-passed-evidence-v1"},
    )
    research = ResearchStore(database_url)
    run = research.create_run(
        kind="platform_model_feature_screen_operational_settlement_short_1_5d",
        objective="freeze the quiescent source before operational remediation",
        dataset="resource-remediation-dataset",
        requested_by=f"autopilot:{cycle['id']}",
        budget={},
        config={
            "contract_version": "platform-model-tournament-run-v2-horizon",
            "autopilot_cycle_id": cycle["id"],
            "research_tournament_id": source["id"],
        },
        artifact_path=tmp_path / "operational-source-run",
    )
    jobs = JobStore(database_url)
    job = jobs.create(
        "model_evaluate",
        {
            "research_run_id": run["id"],
            "research_tournament_id": source["id"],
        },
        tmp_path / "operational-source.log",
        dedupe_active_kind=False,
        idempotency_key=f"operational-source:{source['id']}",
    )
    research.attach_job(str(run["id"]), str(job["id"]))
    branch = autopilot.create_branch(
        str(cycle["id"]),
        scenario="fin_model",
        scope_key="platform:feature_screen:operational-source",
        research_run_id=str(run["id"]),
        job_id=str(job["id"]),
        details={
            "branch_kind": "platform_model_feature_screen",
            "tournament_id": str(source["id"]),
            "model_recompute_executor_version": "model-recompute-docker-v6",
            "candidate_bindings": [
                {"trial_id": str(resource_trial["id"])},
                {"trial_id": str(operational_trial["id"])},
                {"trial_id": str(inherited_trial["id"])},
            ],
        },
    )
    jobs.finish(str(job["id"]), exit_code=1, error="operational source settled")
    research.mark_run(
        str(run["id"]),
        "blocked",
        error="operational source settled",
    )
    assert autopilot.reconcile() == 1
    tournaments.mark_running(str(source["id"]))
    return (
        tournaments.get_tournament(str(source["id"])),
        resource_trial,
        operational_trial,
        inherited_trial,
        {
            "cycle_id": str(cycle["id"]),
            "branch_id": str(branch["id"]),
            "run_id": str(run["id"]),
            "job_id": str(job["id"]),
        },
    )


def test_operational_remediation_preserves_mixed_failures_and_source_history(
    tmp_path: Path,
    database_url: str,
) -> None:
    source, resource_trial, operational_trial, inherited_trial, _owners = (
        _seed_operational_remediation_source(database_url, tmp_path)
    )
    store = ResearchTournamentStore(database_url)

    successor = store.ensure_operational_remediation_preregistered(
        source_tournament_id=str(source["id"]),
        source_executor_version="model-recompute-docker-v6",
        target_executor_version="model-recompute-docker-v7",
    )

    unchanged = store.get_tournament(str(source["id"]))
    assert unchanged["status"] == "blocked"
    assert unchanged["multiple_testing"]["contract_version"] == (
        "model-tournament-operational-settlement-v1"
    )
    assert unchanged["multiple_testing"][
        "all_bound_branches_runs_and_jobs_terminal"
    ] is True
    unchanged_resource = next(
        item for item in unchanged["trials"] if item["id"] == resource_trial["id"]
    )
    assert unchanged_resource["status"] == "failed"
    assert unchanged_resource["candidate_id"] == "source-resource-candidate"
    assert unchanged_resource["evidence"] == resource_trial["evidence"]
    assert successor["stage"] == "model_full"
    remediation = successor["manifest"]["operational_remediation"]
    assert remediation["source_tournament_id"] == source["id"]
    assert remediation["performance_triggered"] is False
    assert remediation["contract_upgrade"] is True
    assert remediation["same_frozen_evaluation_contract"] is False
    assert remediation["all_trials_re_preregistered"] is True
    assert remediation["new_multiple_testing_family"] is True
    assert successor["manifest"]["feature_set_window_policy"][
        "identical_periods_within_horizon"
    ] is True
    replacement = next(
        item for item in successor["trials"] if item["name"] == resource_trial["name"]
    )
    operational_replacement = next(
        item
        for item in successor["trials"]
        if item["name"] == operational_trial["name"]
    )
    inherited = next(
        item for item in successor["trials"] if item["name"] == inherited_trial["name"]
    )
    assert replacement["status"] == "preregistered"
    assert replacement["candidate_id"] is None
    assert replacement["spec_sha256"] == resource_trial["spec_sha256"]
    assert operational_replacement["status"] == "preregistered"
    assert operational_replacement["candidate_id"] is None
    assert operational_replacement["spec_sha256"] == operational_trial["spec_sha256"]
    assert inherited["status"] == "preregistered"
    assert inherited["candidate_id"] is None
    assert inherited["metrics"] is None
    assert inherited["evidence"] is None
    unchanged_inherited = next(
        item for item in unchanged["trials"] if item["id"] == inherited_trial["id"]
    )
    assert unchanged_inherited["status"] == "passed"
    assert unchanged_inherited["candidate_id"] == "source-passed-candidate"
    assert unchanged_inherited["evidence"] == inherited_trial["evidence"]


@pytest.mark.parametrize(
    ("owner", "expected_error"),
    [
        ("branch", "source branch is not terminal"),
        ("run", "source research run is not terminal"),
        ("job", "source job is not terminal"),
    ],
)
def test_operational_remediation_refuses_nonterminal_source_owner(
    tmp_path: Path,
    database_url: str,
    owner: str,
    expected_error: str,
) -> None:
    source, _resource, _operational, _inherited, owners = (
        _seed_operational_remediation_source(database_url, tmp_path)
    )
    store = ResearchTournamentStore(database_url)
    with store.engine.begin() as connection:
        if owner == "branch":
            connection.execute(
                update(autopilot_branches)
                .where(autopilot_branches.c.id == owners["branch_id"])
                .values(status="running", finished_at=None)
            )
        elif owner == "run":
            connection.execute(
                update(research_runs)
                .where(research_runs.c.id == owners["run_id"])
                .values(status="evaluating", finished_at=None)
            )
        else:
            connection.execute(
                update(job_rows)
                .where(job_rows.c.id == owners["job_id"])
                .values(status="queued", finished_at=None)
            )

    with pytest.raises(ValueError, match=expected_error):
        store.ensure_operational_remediation_preregistered(
            source_tournament_id=str(source["id"]),
            source_executor_version="model-recompute-docker-v6",
            target_executor_version="model-recompute-docker-v7",
        )
    with pytest.raises(KeyError):
        store.get_for_cycle_stage(owners["cycle_id"], "model_full")


def test_concurrent_operational_remediation_settles_source_once(
    tmp_path: Path,
    database_url: str,
) -> None:
    source, _resource, _operational, _inherited, owners = (
        _seed_operational_remediation_source(database_url, tmp_path)
    )
    barrier = Barrier(2)

    def create_successor() -> str:
        barrier.wait()
        successor = ResearchTournamentStore(
            database_url
        ).ensure_operational_remediation_preregistered(
            source_tournament_id=str(source["id"]),
            source_executor_version="model-recompute-docker-v6",
            target_executor_version="model-recompute-docker-v7",
        )
        return str(successor["id"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        successor_ids = [
            future.result()
            for future in [executor.submit(create_successor) for _ in range(2)]
        ]

    assert len(set(successor_ids)) == 1
    store = ResearchTournamentStore(database_url)
    assert store.get_tournament(str(source["id"]))["status"] == "blocked"
    assert (
        store.get_for_cycle_stage(owners["cycle_id"], "model_full")["id"]
        == successor_ids[0]
    )


def test_real_model_failure_cannot_open_operational_remediation(database_url: str) -> None:
    autopilot = AutopilotStore(database_url)
    cycle = autopilot.ensure_cycle(
        {
            "name": "real-gate-failure-dataset",
            "lineage_id": "a" * 64,
            "end_date": "2026-08-31",
            "provenance": {"dataset_identity_sha256": "b" * 64},
        },
        config_revision=1,
    )
    store = ResearchTournamentStore(database_url)
    source = store.ensure_preregistered(
        cycle_id=str(cycle["id"]), dataset_identity_sha256="b" * 64
    )
    trial = next(
        item
        for item in source["trials"]
        if (item.get("spec") or {}).get("round") == "feature_screen"
    )
    store.transition_trial(str(trial["id"]), "queued")
    store.transition_trial(str(trial["id"]), "running")
    store.transition_trial(
        str(trial["id"]),
        "failed",
        candidate_id="real-failed-candidate",
        metrics={
            "reason_code": "performance_gate_failed",
            "rank_ic": -0.05,
            "annualized_excess_return_with_cost": -0.20,
        },
        evidence={
            "contract_version": "model-feature-screen-gate-v1",
            "performance_metrics_produced": True,
            "investment_hypothesis_rejected": True,
        },
    )

    with pytest.raises(ValueError, match="no operational trial"):
        store.ensure_operational_remediation_preregistered(
            source_tournament_id=str(source["id"]),
            source_executor_version="executor-v1",
            target_executor_version="executor-v2",
        )
    with pytest.raises(KeyError):
        store.get_for_cycle_stage(str(cycle["id"]), "model_full")
