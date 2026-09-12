from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from quant_platform.autopilot import AutopilotStore
from quant_platform.fin_quant_handoff_recovery import RECOVERY_KEY, FinQuantHandoffRecovery
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


def owner_fixture(monkeypatch, *, recovery=True):
    champion = {"candidate_id": "original-model", "kind": "model"}
    selection = {"evidence_sha256": "a" * 64}
    payload = {
        "research_run_id": "new-run", "research_tournament_id": "old-model-tournament",
        "dataset_identity_sha256": "d" * 64,
        "prediction_champion": champion, "prediction_champion_evidence": selection,
    }
    run = {
        "id": "new-run", "job_id": "new-rdagent-job",
        "config": {**deepcopy(payload), "autopilot_cycle_id": "new-cycle"},
    }
    parent = {"id": "old-model-tournament", "cycle_id": "old-cycle" if recovery else "new-cycle"}
    cycle = {
        "id": "new-cycle", "status": "active", "finished_at": None,
        "dataset_identity_sha256": "d" * 64,
        "state": {
            "prediction_champion": deepcopy(champion),
            "prediction_champion_evidence": deepcopy(selection),
            RECOVERY_KEY: {"source_tournament_id": parent["id"]},
        },
        "branches": [{"scenario": "fin_quant", "scope_key": "joint",
                      "research_run_id": "new-run", "job_id": "new-rdagent-job"}],
    }
    worker = object.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace()
    worker.research = SimpleNamespace(get_run=lambda _id: run, engine=object())
    worker.research_tournaments = SimpleNamespace(get_tournament=lambda _id: parent)
    monkeypatch.setattr(AutopilotStore, "get_cycle", lambda _self, _id: cycle)
    verified = []

    def verify(_service, observed):
        verified.append(observed)
        return {"tournament": parent}

    monkeypatch.setattr(FinQuantHandoffRecovery, "verify", verify)
    job = {"id": "new-rdagent-job", "payload": payload}
    return worker, job, run, parent, cycle, verified


def test_recovery_owner_comes_from_real_run_config_without_payload_cycle_id(monkeypatch):
    worker, job, _run, _parent, cycle, verified = owner_fixture(monkeypatch)
    assert "autopilot_cycle_id" not in job["payload"]
    assert worker._quant_preregistration_research_run_id(job) == "new-run"
    assert verified == [cycle]


def test_normal_quant_research_keeps_same_cycle_parent_without_recovery(monkeypatch):
    worker, job, _run, _parent, _cycle, verified = owner_fixture(monkeypatch, recovery=False)
    assert worker._quant_preregistration_research_run_id(job) == "new-run"
    assert verified == []


@pytest.mark.parametrize("change", ["job", "missing_owner", "parent", "dataset"])
def test_real_run_owner_drift_rejected_before_source_lookup(monkeypatch, change):
    worker, job, run, _parent, _cycle, verified = owner_fixture(monkeypatch)
    if change == "job":
        run["job_id"] = "independent-evaluator-job"
    elif change == "missing_owner":
        run["config"].pop("autopilot_cycle_id")
        job["payload"]["autopilot_cycle_id"] = "new-cycle"
    elif change == "parent":
        run["config"]["research_tournament_id"] = "other-parent"
    else:
        run["config"]["dataset_identity_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="executing job or frozen owner differs"):
        worker._quant_preregistration_research_run_id(job)
    assert verified == []


@pytest.mark.parametrize("change", ["branch", "branch_job", "inactive", "parent", "champion"])
def test_cross_cycle_quant_requires_actual_successor_branch_and_frozen_inputs(monkeypatch, change):
    worker, job, run, _parent, cycle, _verified = owner_fixture(monkeypatch)
    if change == "branch":
        cycle["branches"][0]["research_run_id"] = "other-run"
    elif change == "branch_job":
        cycle["branches"][0]["job_id"] = "other-job"
    elif change == "inactive":
        cycle["status"] = "paused"
    elif change == "parent":
        cycle["state"][RECOVERY_KEY]["source_tournament_id"] = "other-model-tournament"
    else:
        run["config"]["prediction_champion"] = {"candidate_id": "another-model"}
    with pytest.raises(ValueError, match="verified successor"):
        worker._quant_preregistration_research_run_id(job)


def test_source_lineage_verification_failure_cannot_register_quant_outputs(monkeypatch):
    worker, job, _run, _parent, _cycle, _verified = owner_fixture(monkeypatch)
    from quant_platform import worker as worker_module

    monkeypatch.setattr(worker_module, "_validate_fin_quant_research_result", lambda _result: None)
    worker._freeze_fin_quant_baseline = lambda _payload: pytest.fail(
        "unverified owner reached candidate preparation")

    def verify(_service, _cycle):
        raise ValueError("source activity changed")

    monkeypatch.setattr(FinQuantHandoffRecovery, "verify", verify)
    with pytest.raises(ValueError, match="source activity changed"):
        worker._queue_quant_bundle_evaluation(job, {"quant_bundles": []})
