from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier
from types import SimpleNamespace

import pytest
from sqlalchemy import func, insert, select
from test_autopilot_research_events import _dataset, _registered
from test_fin_quant_handoff_recovery import _fixture

from quant_data.database import (
    audit_events,
    autopilot_cycles,
    jobs,
    research_runs,
    research_tournaments,
    row_dict,
)
from quant_platform.autopilot import AutopilotStore, normalize_autopilot_config
from quant_platform.fin_quant_handoff_recovery import (
    HANDOFF_ERROR,
    RECOVERY_KEY,
    FinQuantHandoffRecovery,
)
from quant_platform.job_store import JobStore
from quant_platform.platform_config_store import PlatformConfigStore
from quant_platform.research_store import ResearchStore
from quant_platform.research_tournament import ResearchTournamentStore, canonical_sha256


def _service(database_url, tmp_path, monkeypatch):
    """Real transactions/owners; file and statistical checks have separate pure coverage."""
    store = AutopilotStore(database_url)
    dataset = _dataset(tmp_path)
    source = _registered(store, dataset, completion_mode="managed_fin_strategy")
    _, sample, tournament, _, _, _, _ = _fixture(tmp_path)
    source["state"].update({key: deepcopy(value) for key, value in sample["state"].items()
                            if key != "research_event"})
    source = store.set_cycle_state(source["id"], state=source["state"], stage="joint_optimization")
    research = ResearchStore(database_url)
    run = research.create_run(
        kind="rdagent_quant:short_1_5d", objective="handoff", dataset=dataset["name"],
        requested_by="autopilot", budget={"loop_n": 10, "duration": "1h"},
        config={"autopilot_cycle_id": source["id"]}, artifact_path=tmp_path / "artifacts",
    )
    job_store = JobStore(database_url)
    job = job_store.create("rdagent_quant", {"research_run_id": run["id"]}, tmp_path / "old.log")
    research.attach_job(run["id"], job["id"])
    store.create_branch(source["id"], scenario="fin_quant", scope_key="joint",
                        research_run_id=run["id"], job_id=job["id"], details={})
    research.mark_run(run["id"], "failed", error=HANDOFF_ERROR)
    job_store.finish(job["id"], exit_code=2, error=HANDOFF_ERROR)
    store.reconcile()
    source = store.get_cycle(source["id"])
    assert source["status"] == "blocked"
    tournament["cycle_id"] = source["id"]
    configs = PlatformConfigStore(database_url)
    configs.put("autopilot", normalize_autopilot_config(), actor="test", reason="recovery fixture")
    controller = SimpleNamespace(
        engine=store.engine, settings=SimpleNamespace(
            data_root=tmp_path, rdagent_enabled=True, rdagent_max_loops=10,
            rdagent_max_duration="2h", quantlab_release_id="repaired-release",
            quantlab_config_digest="c" * 64,
        ),
    )
    service = FinQuantHandoffRecovery(controller)

    def read(_service, connection, source_id):
        joined_store = AutopilotStore(database_url)
        # This read shares the caller's transaction, including rollback tests.
        from quant_platform.fin_quant_handoff_recovery import _JoinedEngine
        joined_store.engine = _JoinedEngine(connection)
        current = joined_store.get_cycle(source_id)
        return {"source": current, "dataset": dataset, "tournament": tournament,
                "run": research.get_run(run["id"]), "job": job_store.get(job["id"]),
                "trials_sha256": "d" * 64, "output_proof": {"files": []},
                "publication_proof": {"numeric_outputs_revalidated": False}}
    monkeypatch.setattr(FinQuantHandoffRecovery, "_read", read)
    return service, source, store, configs, research, job_store, run, job


def _plan(service, source, key="recovery"):
    return service.plan(source["id"], key, "operator", "Recover the failed handoff")


def test_database_successor_keeps_source_owners_unchanged_and_is_idempotent(
    database_url, tmp_path, monkeypatch,
):
    service, source, store, _, research, job_store, run, job = _service(
        database_url, tmp_path, monkeypatch,
    )
    before = deepcopy((store.get_cycle(source["id"]), research.get_run(run["id"]),
                       job_store.get(job["id"])))
    plan = _plan(service, source)
    successor = service.execute(plan, expected_sha256=canonical_sha256(plan))
    assert successor["id"] != source["id"]
    assert successor["state"][RECOVERY_KEY]["source_cycle_id"] == source["id"]
    assert not successor["branches"]
    assert service.execute(plan, expected_sha256=canonical_sha256(plan))["id"] == successor["id"]
    assert (store.get_cycle(source["id"]), research.get_run(run["id"]),
            job_store.get(job["id"])) == before
    with store.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(jobs)) == 1
        assert connection.scalar(select(func.count()).select_from(research_runs)) == 1
        assert connection.scalar(select(func.count()).select_from(audit_events).where(
            audit_events.c.action == "autopilot.fin_quant_handoff_recovery.register")) == 1


def test_database_audit_insert_failure_rolls_back_the_entire_successor(
    database_url, tmp_path, monkeypatch,
):
    service, source, store, *_ = _service(database_url, tmp_path, monkeypatch)
    plan = _plan(service, source)
    from sqlalchemy import event

    def reject_audit(_connection, _cursor, statement, _params, _context, _many):
        if "INSERT INTO quantlab.audit_events" in statement:
            raise RuntimeError("injected audit write failure")
    event.listen(store.engine, "before_cursor_execute", reject_audit)
    try:
        with pytest.raises(RuntimeError, match="audit write failure"):
            service.execute(plan, expected_sha256=canonical_sha256(plan))
    finally:
        event.remove(store.engine, "before_cursor_execute", reject_audit)
    assert store.get_research_event("recovery") is None
    assert store.get_cycle(source["id"]) == source
    with store.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(autopilot_cycles)) == 1


def test_database_config_revision_drift_rejects_before_any_successor_write(
    database_url, tmp_path, monkeypatch,
):
    service, source, store, configs, *_ = _service(database_url, tmp_path, monkeypatch)
    plan = _plan(service, source)
    configs.put("autopilot", normalize_autopilot_config({"enabled": False}),
                actor="another-operator", reason="operator changed config")
    with pytest.raises(ValueError, match="changed after planning"):
        service.execute(plan, expected_sha256=canonical_sha256(plan))
    assert store.get_research_event("recovery") is None
    assert store.get_cycle(source["id"]) == source


@pytest.mark.parametrize("same_key", [True, False])
def test_database_concurrent_recovery_requests_create_exactly_one_successor(
    database_url, tmp_path, monkeypatch, same_key,
):
    service, source, store, *_ = _service(database_url, tmp_path, monkeypatch)
    plans = [_plan(service, source, "recovery-a"),
             _plan(service, source, "recovery-a" if same_key else "recovery-b")]
    barrier = Barrier(2)

    def execute(plan):
        barrier.wait(timeout=10)
        try:
            return service.execute(plan, expected_sha256=canonical_sha256(plan))["id"]
        except ValueError as exc:
            return str(exc)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, plans))
    with store.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(autopilot_cycles)) == 2
        assert connection.scalar(select(func.count()).select_from(audit_events).where(
            audit_events.c.action == "autopilot.fin_quant_handoff_recovery.register")) == 1
    if same_key:
        assert results[0] == results[1]
    else:
        assert any("already owns a recovery successor" in result for result in results)
    assert store.get_cycle(source["id"]) == source


def test_database_new_quant_family_belongs_to_successor_and_preserves_source_tournament(
    database_url, tmp_path, monkeypatch,
):
    from datetime import UTC, datetime

    from test_research_tournament_ledger import _baseline, _candidate

    service, source, store, _, research, *_ = _service(database_url, tmp_path, monkeypatch)
    plan = _plan(service, source)
    cycle = service.execute(plan, expected_sha256=canonical_sha256(plan))
    with store.engine.begin() as connection:
        connection.execute(insert(research_tournaments).values(
            id=source["state"]["research_tournament_id"], cycle_id=source["id"],
            stage="feature_screen", status="succeeded", dataset_identity_sha256="a" * 64,
            manifest_json={}, manifest_sha256="b" * 64, max_trials=1,
            selected_trial_ids_json=[], multiple_testing_json={},
            multiple_testing_sha256="c" * 64, created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC), finished_at=datetime.now(UTC),
        ))
    tournaments = ResearchTournamentStore(database_url)
    parent = tournaments.get_tournament(source["state"]["research_tournament_id"])
    run = research.create_run(
        kind="rdagent_quant:short_1_5d", objective="recovered research", dataset=source["dataset"],
        requested_by="autopilot", budget={"loop_n": 10, "duration": "1h"},
        config={"autopilot_cycle_id": cycle["id"]}, artifact_path=tmp_path / "new-run",
    )
    checks = []

    def verify_owner(_connection, parent_value, run_id, candidates):
        checks.append(run_id)
        assert parent_value["id"] == parent["id"] and run_id == run["id"]
        assert candidates[0]["candidate_id"] == "new-joint"
        return cycle["id"], {"cycle_id": cycle["id"], "research_run_id": run_id,
                              "source_cycle_id": source["id"],
                              "source_tournament_id": parent["id"], "lineage_sha256": "e" * 64}
    monkeypatch.setattr(tournaments, "_quant_recovery_owner", verify_owner)
    result = tournaments.ensure_quant_preregistered(
        parent_tournament_id=parent["id"], dataset_identity_sha256="a" * 64,
        baseline_prediction_champion=_baseline(), candidates=[_candidate("new-joint", "1")],
        research_run_id=run["id"],
    )
    assert result["cycle_id"] == cycle["id"]
    assert result["manifest"]["handoff_recovery"]["source_cycle_id"] == source["id"]
    assert checks == [run["id"], run["id"]]
    assert tournaments.get_tournament(parent["id"]) == parent
    with pytest.raises(KeyError):
        tournaments.get_for_cycle_stage(source["id"], "quant")
    with store.engine.connect() as connection:
        rows = [row_dict(row) for row in connection.execute(select(research_tournaments)
            .where(research_tournaments.c.stage == "quant"))]
    assert len(rows) == 1 and rows[0]["cycle_id"] == cycle["id"]
