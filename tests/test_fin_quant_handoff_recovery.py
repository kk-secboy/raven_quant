from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from test_autopilot_research_events import _cycle, _tick_fixture
from test_autopilot_terminal_resolution import _terminal_fixture

from quant_platform.autopilot import (
    AutopilotController,
    _joint_completion_verified,
    normalize_autopilot_config,
)
from quant_platform.fin_quant_handoff_recovery import (
    EVIDENCE_KEYS,
    HANDOFF_ERROR,
    RECOVERY_KEY,
    RECOVERY_VERSION,
    FinQuantHandoffRecovery,
    plan_file,
    preparation_outputs,
    require_recovery_evidence,
    require_recovery_lineage,
    source_identity,
    tournament_identity,
    validate_source,
)
from quant_platform.research_tournament import canonical_sha256

pytestmark = pytest.mark.no_database


def _fixture(tmp_path):
    dataset, source, tournament = _terminal_fixture(tmp_path, resource=False)
    source.update(status="blocked", stage="joint_optimization_blocked",
                  finished_at="2026-09-10T04:35:32+00:00")
    source["state"]["prediction_champion"]["primary_feature_set_id"] = "platform-seed-v1"
    branch = source["branches"][-1]
    branch.update(status="failed", research_run_id="failed-run", job_id="failed-job")
    run = {"id": "failed-run", "job_id": "failed-job", "status": "failed",
           "config_json": {"autopilot_cycle_id": source["id"]},
           "finished_at": "2026-09-10T04:35:16+00:00"}
    job = {"id": "failed-job", "kind": "rdagent_quant", "status": "failed",
           "error": HANDOFF_ERROR, "finished_at": "2026-09-10T04:35:16+00:00",
           "payload_json": {
               "research_run_id": run["id"], "research_tournament_id": tournament["id"],
               "dataset_identity_sha256": source["dataset_identity_sha256"],
               "prediction_champion": deepcopy(source["state"]["prediction_champion"]),
               "prediction_champion_evidence": deepcopy(
                   source["state"]["prediction_champion_evidence"]),
               "loop_n": 10, "duration": "1h",
           }}
    successor = _cycle(
        dataset, event_key="recovered-handoff", completion_mode="managed_fin_strategy",
    )
    successor["id"] = "recovery-cycle"
    successor["state"].update({key: deepcopy(source["state"][key])
                               for key in EVIDENCE_KEYS if key in source["state"]})
    successor["state"].update(research_tournament_id=tournament["id"],
                               active_research_tournament_id=tournament["id"])
    lineage = {
        "contract_version": RECOVERY_VERSION, "source_cycle_id": source["id"],
        "source_cycle_sha256": canonical_sha256(source_identity(source)),
        "source_event_sha256": source["state"]["research_event"]["sha256"],
        "source_tournament_id": tournament["id"],
        "source_tournament_sha256": canonical_sha256(tournament_identity(tournament)),
        "recovery_event_sha256": successor["state"]["research_event"]["sha256"],
    }
    successor["state"][RECOVERY_KEY] = {**lineage, "sha256": canonical_sha256(lineage)}
    return dataset, source, tournament, branch, run, job, successor


def test_original_manual_cycle_and_sealed_models_are_unchanged(tmp_path):
    dataset, source, tournament, branch, run, job, successor = _fixture(tmp_path)
    before = deepcopy((source, tournament, branch, run, job))
    validate_source(source, tournament, branch, run, job, dataset)
    assert _joint_completion_verified(successor, tournament, source)
    assert not _joint_completion_verified(successor, tournament)
    assert (source, tournament, branch, run, job) == before


@pytest.mark.parametrize("change", ["dataset", "label", "source", "tournament", "event", "budget"])
def test_recovery_rejects_changed_provenance_even_with_valid_lineage_hash(tmp_path, change):
    _, source, tournament, _, _, _, successor = _fixture(tmp_path)
    if change == "dataset":
        successor["dataset_identity_sha256"] = "b" * 64
    elif change == "label":
        successor["state"]["prediction_champion_evidence"]["label_horizon_sessions"] = 1
    elif change == "source":
        source["state"]["source_changed"] = True
    elif change == "tournament":
        tournament["status"] = "failed"
    elif change == "event":
        successor["state"]["research_event"]["dataset"]["name"] = "another-dataset"
    else:
        successor["state"]["research_event"]["request"]["quant_loop_n"] = 20
    assert not _joint_completion_verified(successor, tournament, source)


@pytest.mark.parametrize("change", ["error", "opened_oos", "active", "job", "dataset", "budget"])
def test_only_the_pre_experiment_handoff_failure_is_recoverable(tmp_path, change):
    dataset, source, tournament, branch, run, job, _ = _fixture(tmp_path)
    if change == "error":
        job["error"] = "economic validation failed"
    elif change == "opened_oos":
        source["state"]["final_oos_opened"] = True
    elif change == "active":
        source["status"] = "active"
    elif change == "job":
        run["job_id"] = "independent-evaluator-job"
    elif change == "dataset":
        dataset["provenance"]["dataset_identity_sha256"] = "b" * 64
    else:
        job["payload_json"]["duration"] = "2h"
    with pytest.raises(ValueError, match="fin_quant recovery"):
        validate_source(source, tournament, branch, run, job, dataset)


def test_recovery_tick_enqueues_only_new_fin_quant_without_model_scheduling(tmp_path, monkeypatch):
    dataset, source, tournament, _, _, _, successor = _fixture(tmp_path)
    controller, captured, _ = _tick_fixture(monkeypatch, dataset, successor)
    controller.store.get_cycle = lambda cid: source if cid == source["id"] else successor
    controller.store.branch_for_scope = lambda *_: None
    controller.tournaments.get_tournament = lambda tid: tournament
    monkeypatch.setattr(FinQuantHandoffRecovery, "verify", lambda *_: {"tournament": tournament})
    controller._quant_input_sha256 = lambda *_: "a" * 64
    controller._quant_due = lambda *_args, **_kwargs: True
    calls = []
    controller._enqueue = lambda *args, **kwargs: calls.append((args, kwargs))
    controller._ensure_model_selection_schedule = lambda *_args, **_kwargs: pytest.fail(
        "recovery must not select models again")
    controller._ensure_horizon_factor_bundle = lambda *_args, **_kwargs: pytest.fail(
        "recovery must preserve the source factor inputs")
    result = controller._tick_horizon(
        current=datetime(2026, 9, 12, tzinfo=UTC), config=normalize_autopilot_config(),
        revision=7, dataset=dataset, horizon_profile=successor["horizon_profile"],
    )
    assert result == {"cycles": 1, "branches": 1, "failed": 0}
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0]["id"] == successor["id"]
    assert args[2:4] == ("fin_quant", "joint")
    assert kwargs["tournament_id"] == tournament["id"]
    assert kwargs["config"]["quant_loop_n"] == 10
    assert not captured["platform"] and not captured["preregistrations"]


def test_active_recovery_tournament_requires_live_source_verification(tmp_path, monkeypatch):
    _, source, tournament, _, _, _, successor = _fixture(tmp_path)
    controller = object.__new__(AutopilotController)
    controller.store = SimpleNamespace(get_cycle=lambda _id: source)
    controller.tournaments = SimpleNamespace(get_tournament=lambda _id: tournament)
    def verify(_service, cycle):
        require_recovery_lineage(cycle, tournament, source)
        return {"tournament": tournament}
    monkeypatch.setattr(FinQuantHandoffRecovery, "verify", verify)
    assert controller._active_model_tournament(successor) is tournament
    source["state"]["changed"] = True
    with pytest.raises(ValueError, match="source activity changed"):
        controller._active_model_tournament(successor)


def test_plan_byte_hash_and_execution_hash_are_mandatory(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"contract_version": RECOVERY_VERSION}), encoding="utf-8")
    raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = plan_file(path, raw_hash)
    path.write_text(json.dumps({"contract_version": "changed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="plan file bytes changed"):
        plan_file(path, raw_hash)
    with pytest.raises(ValueError, match="plan hash changed"):
        FinQuantHandoffRecovery(SimpleNamespace()).execute(plan, expected_sha256="0" * 64)


def test_plain_cycles_cannot_borrow_another_cycles_model_tournament(tmp_path):
    _, source, tournament, _, _, _, successor = _fixture(tmp_path)
    successor["state"].pop(RECOVERY_KEY)
    assert not _joint_completion_verified(successor, tournament, source)
    with pytest.raises(ValueError, match="lineage is missing"):
        require_recovery_lineage(successor, tournament, source)


def test_completed_recovery_hands_off_before_newer_data_starts_another_tournament(
    tmp_path, monkeypatch,
):
    from quant_platform.fin_strategy_research_completion import ManagedResearchCompletion

    dataset, _, _, _, _, _, successor = _fixture(tmp_path)
    successor.update(status="succeeded", stage="research_complete",
                     finished_at="2026-09-12T00:00:00+00:00")
    successor["branches"] = [{"scenario": "fin_quant", "status": "succeeded"}]
    newer = {**dataset, "name": "newer-publication", "end_date": "2026-09-11"}
    controller, captured, _ = _tick_fixture(monkeypatch, newer, successor)
    controller.store.ensure_cycle = lambda *_args, **_kwargs: pytest.fail(
        "pending strategy completion must get the horizon before a new model tournament")
    monkeypatch.setattr(ManagedResearchCompletion, "pending_cycle_ids",
                        lambda _self: [successor["id"]])
    result = controller._tick_horizon(
        current=datetime(2026, 9, 12, tzinfo=UTC), config=normalize_autopilot_config(),
        revision=7, dataset=newer, horizon_profile=successor["horizon_profile"],
    )
    assert result == {"cycles": 0, "branches": 0, "failed": 0}
    assert not captured["platform"] and not captured["preregistrations"]
    monkeypatch.setattr(ManagedResearchCompletion, "pending_cycle_ids", lambda _self: [])
    controller._handoff_recovery_dispatch_inflight = lambda _ids: False
    assert not controller._handoff_recovery_completion_pending(
        [successor], successor["horizon_profile"],
    )


def test_queued_completion_dispatch_keeps_recovery_priority_until_it_is_claimed(tmp_path):
    from test_autopilot_research_events import _Connection

    controller = object.__new__(AutopilotController)
    queries = []
    controller.engine = SimpleNamespace(connect=lambda: _Connection(["queued-dispatch"], queries))
    assert controller._handoff_recovery_dispatch_inflight({"recovered-cycle"})
    assert "queued" in str(queries[0].compile(compile_kwargs={"literal_binds": True}))


def test_input_preparation_is_allowed_but_real_experiment_outputs_are_not(tmp_path):
    feature_set = {"id": "frozen", "features": {"factor": "$close"}}
    target = tmp_path / "scenario-inputs" / "base-features"
    target.mkdir(parents=True)
    (target / "definition.json").write_text(json.dumps(feature_set), encoding="utf-8")
    (target / "base_factors.json").write_text(json.dumps(feature_set["features"]), encoding="utf-8")
    (tmp_path / "docker-runtime").mkdir()
    (tmp_path / "qlib-home" / "qlib_data" / "cn_data").mkdir(parents=True)
    assert len(preparation_outputs(tmp_path, feature_set)) == 2
    (tmp_path / "trace").mkdir()
    with pytest.raises(ValueError, match="experiment directory"):
        preparation_outputs(tmp_path, feature_set)


@pytest.mark.parametrize("changed", ["job", "run", "trials", "branches", "target", "outputs"])
def test_each_tick_checks_source_owners_trials_outputs_and_target_release(tmp_path, changed):
    _, source, tournament, _, run, job, successor = _fixture(tmp_path)
    evidence = {"source": source, "tournament": tournament, "run": run, "job": job,
                "trials_sha256": "f" * 64, "output_proof": {}, "publication_proof": {}}
    settings = SimpleNamespace(quantlab_release_id="release", quantlab_config_digest="c" * 64)
    lineage = successor["state"][RECOVERY_KEY]
    lineage.update({
        "source_branches_sha256": canonical_sha256(
            sorted(source["branches"], key=lambda x: x["id"])),
        "source_run_sha256": canonical_sha256(run), "source_job_sha256": canonical_sha256(job),
        "source_trials_sha256": evidence["trials_sha256"],
        "source_output_proof_sha256": canonical_sha256({}),
        "source_publication_proof_sha256": canonical_sha256({}),
        "target_release": {"release_id": "release", "config_digest": "c" * 64},
    })
    lineage["sha256"] = canonical_sha256({k: v for k, v in lineage.items() if k != "sha256"})
    require_recovery_evidence(successor, evidence, settings)
    if changed in {"job", "run"}:
        evidence[changed]["status"] = "queued"
    elif changed == "trials":
        evidence["trials_sha256"] = "1" * 64
    elif changed == "branches":
        source["branches"][0]["status"] = "running"
    elif changed == "target":
        settings.quantlab_release_id = "another-release"
    else:
        evidence["output_proof"]["experiment"] = "new"
    with pytest.raises(ValueError, match="changed"):
        require_recovery_evidence(successor, evidence, settings)
