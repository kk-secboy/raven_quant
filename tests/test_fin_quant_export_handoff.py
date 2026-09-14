from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from quant_platform import worker as worker_module
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


def exported_result(tmp_path):
    path = Path(__file__).parents[1] / "scripts" / "rdagent_bridge.py"
    spec = importlib.util.spec_from_file_location("quant_export_handoff_bridge", path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    factor_code = "def calculate(data):\n    return data\n"
    model_code = "class Net:\n    pass\n"
    rounds = {
        0: {
            "feedback": {"decision": True},
            "runner_snapshot": {
                "kind": "factor",
                "artifacts": [{"name": "close_factor", "formulation": "$close",
                               "code": factor_code,
                               "code_sha256": bridge._sha256(factor_code)}],
            },
        },
        1: {
            "feedback": {"decision": True},
            "runner_snapshot": {
                "kind": "model",
                "artifacts": [{"name": "model", "model_type": "Tabular",
                               "code": model_code,
                               "code_sha256": bridge._sha256(model_code)}],
            },
        },
    }
    code_root, values_root = tmp_path / "code", tmp_path / "values"
    code_root.mkdir()
    values_root.mkdir()
    bundles = bridge._materialize_quant_bundles(
        rounds, code_root=code_root, values_root=values_root,
        feature_set_id="base", feature_set_sha256="a" * 64,
    )
    # Production exports on Linux; retain those bytes when this test runs on Windows.
    for code_file in code_root.iterdir():
        code_file.write_bytes(code_file.read_bytes().replace(b"\r\n", b"\n"))
    coverage = bridge._fin_quant_arm_coverage(rounds)
    return {"quant_bundles": bundles, "fin_quant_coverage": coverage,
            "fin_quant_outcome": bridge._fin_quant_research_outcome(coverage, bundles)}


def handoff_worker(monkeypatch, tmp_path, result):
    worker = object.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root=tmp_path)
    worker._quant_preregistration_research_run_id = Mock(return_value="run")
    worker._freeze_fin_quant_baseline = Mock(return_value={"evidence_sha256": "b" * 64})
    monkeypatch.setattr(worker_module, "resolve_research_label_binding", lambda _: None)
    worker.factor_library = SimpleNamespace(register_expression_definition=Mock(
        return_value={"id": "definition", "expression": "$close"}))
    factor = result["quant_bundles"][0]["factors"][0]
    worker.rdagent_candidates = SimpleNamespace(
        register_run_artifact=Mock(return_value={"id": "artifact"}),
        create_model_candidate=Mock(return_value={"id": "governed-model"}),
        create_joint_quant_bundle_candidate=Mock(return_value={
            "id": "governed-bundle", "bundle_manifest_sha256": "c" * 64,
            "bundle_manifest_json": {
                "factors": [{"code_sha256": factor["code_sha256"],
                             "candidate_id": "governed-factor"}],
                "model": {"recipe_sha256": "d" * 64},
            },
        }),
    )
    worker.research_tournaments = SimpleNamespace(ensure_quant_preregistered=Mock(
        return_value={"id": "quant-tournament", "manifest_sha256": "e" * 64,
                      "trials": [{"candidate_id": "governed-bundle", "id": "trial"}]}))
    worker.store = SimpleNamespace(create=Mock(return_value={"id": "evaluation-job"}))
    worker.research = SimpleNamespace(attach_job=Mock())
    job = {"payload": {
        "research_run_id": "run", "research_tournament_id": "model-tournament",
        "feature_set": {"id": "base", "definition_sha256": "a" * 64},
        "dataset": "dataset", "dataset_path": "dataset-path",
        "dataset_identity_sha256": "f" * 64,
        "periods": {"valid_end": "2024-12-31", "test_start": "2025-01-01",
                    "test_end": "2025-12-31"},
    }}
    return worker, job


@pytest.mark.parametrize("missing_name", [False, True])
def test_actual_exported_factor_identity_reaches_independent_handoff(
    monkeypatch, tmp_path, missing_name,
):
    result = exported_result(tmp_path)
    factor = result["quant_bundles"][0]["factors"][0]
    assert "id" not in factor
    if missing_name:
        factor.pop("name")
    worker, job = handoff_worker(monkeypatch, tmp_path, result)

    assert worker._queue_quant_bundle_evaluation(job, result) == 1
    registration = worker.factor_library.register_expression_definition.call_args.kwargs
    assert registration["alias"] == f"rdagent-quant:{factor['candidate_id']}"
    assert registration["name"] == (factor["candidate_id"] if missing_name else "close_factor")
    queued = worker.store.create.call_args.args
    assert queued[0] == "quant_bundle_evaluate"
    assert queued[1]["candidates"][0]["factors"][0]["candidate_id"] == "governed-factor"
    assert queued[1]["research_trial_ids"] == {"governed-bundle": "trial"}
    assert queued[1]["final_oos_opened"] is False
    worker.research.attach_job.assert_called_once_with("run", "evaluation-job")


def test_missing_export_identity_is_rejected_before_candidate_registration(monkeypatch, tmp_path):
    result = exported_result(tmp_path)
    result["quant_bundles"][0]["factors"][0].pop("candidate_id")
    worker, job = handoff_worker(monkeypatch, tmp_path, result)
    with pytest.raises(ValueError, match="factor candidate identity is missing"):
        worker._queue_quant_bundle_evaluation(job, result)
    worker.factor_library.register_expression_definition.assert_not_called()
    worker.rdagent_candidates.create_model_candidate.assert_not_called()
    worker.store.create.assert_not_called()


@pytest.mark.parametrize("changed", [None, "result", "code", "inventory", "failed_result"])
def test_result_recovery_uses_original_sealed_export_bytes(tmp_path, changed):
    script = Path(__file__).parents[1] / "scripts" / "recover_fin_quant_result.py"
    spec = importlib.util.spec_from_file_location("quant_result_recovery", script)
    recovery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recovery)
    result = exported_result(tmp_path)
    result.update(status="ok", scenario="fin_quant")
    if changed == "failed_result":
        result["status"] = "failed"
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    files = [result_path, *sorted((tmp_path / "code").iterdir())]
    inventory = {"files": [
        {"relative_path": p.relative_to(tmp_path).as_posix(), "size_bytes": p.stat().st_size,
         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files
    ]}
    inventory_path = tmp_path / "trace-inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    artifacts = [{"id": "inventory", "artifact_type": "fin_quant_trace_inventory",
                  "status": "recorded", "storage_path": str(inventory_path),
                  "content_sha256": recovery.file_hash(inventory_path)}]
    if changed == "result":
        result_path.write_text("{}", encoding="utf-8")
    elif changed == "code":
        files[1].write_text("changed", encoding="utf-8")
    elif changed == "inventory":
        inventory_path.write_text("{}", encoding="utf-8")
    if changed:
        with pytest.raises(ValueError):
            recovery.load_sealed_result(tmp_path, artifacts)
    else:
        actual, proof = recovery.load_sealed_result(tmp_path, artifacts)
        assert actual == result
        assert proof["files"]["result.json"] == recovery.file_hash(result_path)
