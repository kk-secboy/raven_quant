from __future__ import annotations

from copy import deepcopy

import pytest
from sqlalchemy import event, select, update
from test_fin_quant_handoff_recovery_database import _service

from quant_data.database import audit_events, jobs
from quant_platform.fin_quant_handoff_recovery import RECOVERY_KEY
from quant_platform.fin_quant_runtime_restart import remaining_restart_attempts
from quant_platform.research_tournament import canonical_sha256


def _interrupted(database_url, tmp_path, monkeypatch):
    service, source, store, _, research, job_store, *_ = _service(
        database_url, tmp_path, monkeypatch)
    plan = service.plan(source["id"], "first-handoff", "operator", "Original handoff repair")
    prior = service.execute(plan, expected_sha256=canonical_sha256(plan))
    artifact_base = tmp_path / "artifacts" / "rdagent"
    runtime = {"runtime_image_digest": "sha256:" + "c" * 64}
    payload = {
        "research_tournament_id": source["state"]["research_tournament_id"],
        "dataset_identity_sha256": source["dataset_identity_sha256"],
        "expected_rdagent_runtime": runtime, "loop_n": 10, "duration": "1h",
        **{key: deepcopy(prior["state"][key]) for key in
           ("prediction_champion", "prediction_champion_evidence")},
    }
    run = research.create_run(
        kind="rdagent_quant:short_1_5d", objective="joint research", dataset=prior["dataset"],
        requested_by="autopilot", budget={"loop_n": 10, "duration": "1h"},
        config={**payload, "autopilot_cycle_id": prior["id"]}, artifact_path=artifact_base)
    output = artifact_base / run["id"]
    output.mkdir(parents=True)
    (output / "trace").mkdir()
    (output / "trace" / "experiment.txt").write_text("old unsuccessful experiment")
    log = tmp_path / "platform" / "logs" / "interrupted.log"
    log.parent.mkdir(parents=True)
    log.write_text("PermissionError: restored result.h5 cannot be written")
    job = job_store.create("rdagent_quant", {**payload, "research_run_id": run["id"]},
                           log, max_attempts=3)
    research.attach_job(run["id"], job["id"])
    store.create_branch(prior["id"], scenario="fin_quant", scope_key="joint",
                        research_run_id=run["id"], job_id=job["id"], details={})
    job_store.request_cancel(job["id"])
    with store.engine.begin() as c:
        c.execute(update(jobs).where(jobs.c.id == job["id"]).values(attempts=2))
    research.mark_run(run["id"], "cancelled", error="Stopped for runtime repair")
    prior = store.set_cycle_state(prior["id"], state=prior["state"],
                                  stage="joint_optimization", status="paused")
    service.controller.settings.quantlab_release_id = "new-runtime-release"
    service.controller.settings.quantlab_config_digest = "e" * 64
    return service, source, prior, store, research, job_store, run, job, output, log


def _plan(service, prior, key="runtime-restart"):
    return service.plan_runtime_restart(prior["id"], key, "operator", "Use repaired runtime")


def test_runtime_restart_preserves_trials_files_and_consumed_attempts(
    database_url, tmp_path, monkeypatch,
):
    service, source, prior, store, research, job_store, run, job, output, log = _interrupted(
        database_url, tmp_path, monkeypatch)
    before = deepcopy((store.get_cycle(source["id"]), research.get_run(run["id"]),
                       job_store.get(job["id"])))
    old_files = ((output / "trace" / "experiment.txt").read_bytes(), log.read_bytes())
    plan = _plan(service, prior)
    successor = service.execute(plan, expected_sha256=canonical_sha256(plan))
    assert successor["id"] != prior["id"]
    assert successor["stage"] == "joint_optimization" and successor["status"] == "active"
    assert not successor["branches"]
    assert remaining_restart_attempts(successor) == 1
    assert store.get_cycle(prior["id"])["finished_at"] is not None
    assert store.get_cycle(prior["id"])["state"] == prior["state"]
    assert (store.get_cycle(source["id"]), research.get_run(run["id"]),
            job_store.get(job["id"])) == before
    assert ((output / "trace" / "experiment.txt").read_bytes(), log.read_bytes()) == old_files
    assert service.verify(successor)["source"]["id"] == source["id"]
    assert service.execute(plan, expected_sha256=canonical_sha256(plan))["id"] == successor["id"]
    with pytest.raises(ValueError, match="original handoff"):
        _plan(service, successor, "forbidden-chain")


def test_runtime_restart_audit_failure_restores_original_horizon_owner(
    database_url, tmp_path, monkeypatch,
):
    service, _, prior, store, *_ = _interrupted(database_url, tmp_path, monkeypatch)
    plan = _plan(service, prior)

    def reject_audit(_connection, _cursor, statement, _params, _context, _many):
        if "INSERT INTO quantlab.audit_events" in statement:
            raise RuntimeError("injected audit failure")

    event.listen(store.engine, "before_cursor_execute", reject_audit)
    try:
        with pytest.raises(RuntimeError, match="injected audit failure"):
            service.execute(plan, expected_sha256=canonical_sha256(plan))
    finally:
        event.remove(store.engine, "before_cursor_execute", reject_audit)
    assert store.get_cycle(prior["id"]) == prior
    assert store.get_research_event("runtime-restart") is None


@pytest.mark.parametrize("change", ["exhausted", "active", "exported", "oos", "log"])
def test_runtime_restart_rejects_drift_without_closing_predecessor(
    database_url, tmp_path, monkeypatch, change,
):
    service, _, prior, store, _, _, _, job, output, log = _interrupted(
        database_url, tmp_path, monkeypatch)
    plan = _plan(service, prior)
    if change in {"exhausted", "active", "oos"}:
        with store.engine.begin() as c:
            current = c.execute(select(jobs).where(jobs.c.id == job["id"])).one()
            values = {"attempts": 3} if change == "exhausted" else {"status": "running"}
            if change == "oos":
                values = {"payload_json": {**current.payload_json, "final_oos_opened": True}}
            c.execute(update(jobs).where(jobs.c.id == job["id"]).values(**values))
    elif change == "exported":
        (output / "result.json").write_text("{}")
    else:
        log.write_text("unexpected history change")
    with pytest.raises(ValueError):
        service.execute(plan, expected_sha256=canonical_sha256(plan))
    assert store.get_cycle(prior["id"]) == prior
    assert store.get_research_event("runtime-restart") is None


def test_runtime_restart_predecessor_drift_blocks_downstream_verification(
    database_url, tmp_path, monkeypatch,
):
    service, _, prior, store, _, _, _, _, _, log = _interrupted(
        database_url, tmp_path, monkeypatch)
    plan = _plan(service, prior)
    successor = service.execute(plan, expected_sha256=canonical_sha256(plan))
    log.write_text("modified after restart")
    with pytest.raises(ValueError, match="predecessor evidence changed"):
        service.verify(successor)
    stored = store.get_cycle(successor["id"])
    assert stored["state"][RECOVERY_KEY] == successor["state"][RECOVERY_KEY]
    with store.engine.connect() as c:
        assert len(c.execute(select(audit_events.c.id)).all()) == 2


def _failed_runtime_restart(database_url, tmp_path, monkeypatch):
    service, source, prior, store, research, job_store, run, _, _, _ = _interrupted(
        database_url, tmp_path, monkeypatch)
    plan = _plan(service, prior)
    failed = service.execute(plan, expected_sha256=canonical_sha256(plan))
    payload = {**run["config"], "autopilot_cycle_id": failed["id"]}
    payload["expected_rdagent_runtime"] = {"runtime_image_digest": "sha256:" + "f" * 64}
    run = research.create_run(
        kind="rdagent_quant:short_1_5d", objective="joint research", dataset=failed["dataset"],
        requested_by="autopilot", budget={"loop_n": 10, "duration": "1h"}, config=payload,
        artifact_path=tmp_path / "artifacts" / "rdagent")
    output = tmp_path / "artifacts" / "rdagent" / run["id"]
    output.mkdir()
    log = tmp_path / "platform" / "logs" / "exhausted.log"
    log.write_text("pandarallel pickle.dump: OSError: [Errno 28] No space left on device")
    job = job_store.create("rdagent_quant", {**payload, "research_run_id": run["id"]},
                           log, max_attempts=1)
    research.attach_job(run["id"], job["id"])
    store.create_branch(failed["id"], scenario="fin_quant", scope_key="joint",
                        research_run_id=run["id"], job_id=job["id"], details={})
    assert job_store.claim_next(("rdagent_quant",))["id"] == job["id"]
    job_store.finish(job["id"], exit_code=1, error="OSError: [Errno 28]")
    research.mark_run(run["id"], "failed", error="OSError: [Errno 28]")
    failed = store.set_cycle_state(failed["id"], state=failed["state"],
                                  stage="joint_optimization_blocked", status="blocked",
                                  finished=True)
    service.controller.settings.quantlab_release_id = "pipe-transport-repaired-release"
    service.controller.settings.quantlab_config_digest = "f" * 64
    return service, source, failed, store, research, job_store, run, job, log


def test_explicit_operator_repair_preserves_exhausted_history_and_records_one_new_attempt(
    database_url, tmp_path, monkeypatch,
):
    service, source, failed, store, research, job_store, run, job, log = _failed_runtime_restart(
        database_url, tmp_path, monkeypatch)
    before = deepcopy((store.get_cycle(source["id"]), failed,
                       research.get_run(run["id"]), job_store.get(job["id"]), log.read_bytes()))
    with pytest.raises(ValueError, match="original handoff"):
        _plan(service, failed)
    plan = service.plan_runtime_restart(failed["id"], "operator-repair", "operator",
                                        "Fix exhausted infrastructure failure", repair_attempts=1)
    assert plan["runtime_restart"]["remaining_attempts"] == 0
    assert plan["operator_retry_attempts"] == 1
    successor = service.execute(plan, expected_sha256=canonical_sha256(plan))
    assert remaining_restart_attempts(successor) == 1
    assert service.verify(successor)["source"]["id"] == source["id"]
    assert (store.get_cycle(source["id"]), store.get_cycle(failed["id"]),
            research.get_run(run["id"]), job_store.get(job["id"]), log.read_bytes()) == before
    assert service.execute(plan, expected_sha256=canonical_sha256(plan))["id"] == successor["id"]
    duplicate = service.plan_runtime_restart(failed["id"], "another-event", "operator",
                                             "Duplicate repair", repair_attempts=1)
    with pytest.raises(ValueError, match="already owns"):
        service.execute(duplicate, expected_sha256=canonical_sha256(duplicate))
    with store.engine.connect() as c:
        audits = c.execute(select(audit_events.c.details_json)).scalars().all()
        assert any(audit["cycle_id"] == successor["id"]
                   and audit["plan"].get("operator_retry_attempts") == 1 for audit in audits)
    log.write_text("changed historical predecessor")
    with pytest.raises(ValueError, match="predecessor evidence changed"):
        service.verify(successor)


def test_operator_repair_requires_new_release_valid_budget_and_atomic_audit(
    database_url, tmp_path, monkeypatch,
):
    service, _, failed, store, *_ = _failed_runtime_restart(database_url, tmp_path, monkeypatch)
    for budget in (-1, 2, True):
        with pytest.raises(ValueError, match="budget"):
            service.plan_runtime_restart(failed["id"], "repair", "operator", "Repair",
                                         repair_attempts=budget)
    repaired = service.controller.settings.quantlab_release_id
    old_target = failed["state"][RECOVERY_KEY]["target_release"]
    service.controller.settings.quantlab_release_id = old_target["release_id"]
    service.controller.settings.quantlab_config_digest = old_target["config_digest"]
    with pytest.raises(ValueError, match="not repaired"):
        service.plan_runtime_restart(failed["id"], "repair", "operator", "Repair",
                                     repair_attempts=1)
    service.controller.settings.quantlab_release_id = repaired
    service.controller.settings.quantlab_config_digest = "f" * 64
    plan = service.plan_runtime_restart(failed["id"], "repair", "operator", "Repair",
                                         repair_attempts=1)

    def reject_audit(_connection, _cursor, statement, _params, _context, _many):
        if "INSERT INTO quantlab.audit_events" in statement:
            raise RuntimeError("injected repair audit failure")

    event.listen(store.engine, "before_cursor_execute", reject_audit)
    try:
        with pytest.raises(RuntimeError, match="injected repair audit failure"):
            service.execute(plan, expected_sha256=canonical_sha256(plan))
    finally:
        event.remove(store.engine, "before_cursor_execute", reject_audit)
    assert store.get_cycle(failed["id"]) == failed
    assert store.get_research_event("repair") is None
