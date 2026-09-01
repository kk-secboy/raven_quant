from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import quant_platform.worker as worker_module
from quant_platform.rdagent_candidate_store import RDAGentCandidateStore
from quant_platform.research_execution_cadence import (
    build_research_execution_cadence_contract,
)
from quant_platform.research_store import ResearchStore
from quant_platform.research_tournament import canonical_sha256
from quant_platform.worker import LocalJobWorker


class _CapturingArtifactStore:
    def __init__(self) -> None:
        self.storage_paths: list[Path] = []

    def register_or_reuse_run_artifact(self, **values: object) -> dict[str, str]:
        storage_path = Path(str(values["storage_path"]))
        self.storage_paths.append(storage_path)
        return {
            "id": f"artifact-{len(self.storage_paths)}",
            "content_sha256": hashlib.sha256(storage_path.read_bytes()).hexdigest(),
        }


class _CapturingTournamentStore:
    def __init__(self) -> None:
        self.completed: list[tuple[str, dict[str, object]]] = []

    def complete_quant_screening(self, tournament_id: str, **values: object) -> None:
        self.completed.append((tournament_id, values))


def _model_job(run_id: str, *, attempts: int = 1) -> dict:
    feature_set = {
        "id": "qlib-alpha158",
        "definition_sha256": "f" * 64,
        "features": {"close": {"expression": "$close"}},
    }
    return {
        "id": "model-evaluate-job-1",
        "kind": "model_evaluate",
        "attempts": attempts,
        "payload": {
            "research_run_id": run_id,
            "dataset": "sealed-dataset",
            "dataset_path": "qlib/daily",
            "dataset_identity_sha256": "d" * 64,
            "feature_set_id": feature_set["id"],
            "feature_set_definition_sha256": feature_set["definition_sha256"],
            "feature_set": feature_set,
            "evaluation_profiles": [],
            "evaluation_stage": "model_full",
            "research_window_contract": {"contract_version": "research-window-v1"},
            "research_window_contract_sha256": "w" * 64,
            "label_horizon_sessions": 2,
            "candidates": [],
        },
    }


def _quant_binding() -> dict:
    return {
        "horizon_profile": "swing_1_6m",
        "periods": {
            "train": ["2018-01-01", "2022-12-31"],
            "valid": ["2023-01-01", "2023-12-31"],
            "test": ["2024-01-01", "2024-12-31"],
        },
        "research_window_contract": {"contract_version": "research-window-v1"},
        "research_window_contract_sha256": "w" * 64,
        "label_horizon_sessions": 20,
        "binding_sha256": "b" * 64,
    }


def _quant_job(
    run_id: str,
    root: Path,
    binding: dict,
    *,
    attempts: int = 1,
) -> dict:
    feature_set = {
        "id": "qlib-alpha158",
        "definition_sha256": "f" * 64,
        "features": {"close": {"expression": "$close"}},
    }
    return {
        "id": "quant-bundle-evaluate-job-1",
        "kind": "quant_bundle_evaluate",
        "attempts": attempts,
        "payload": {
            "research_run_id": run_id,
            "dataset": "sealed-dataset",
            "dataset_path": "qlib/daily",
            "dataset_identity_sha256": "d" * 64,
            "feature_set_id": feature_set["id"],
            "feature_set_definition_sha256": feature_set["definition_sha256"],
            "feature_set": feature_set,
            "evaluation_profiles": [],
            "baseline_prediction_champion": {},
            "research_tournament_id": "quant-tournament-1",
            "parent_research_tournament_id": "model-tournament-1",
            "research_tournament_manifest_sha256": "m" * 64,
            "research_trial_ids": {"quant-candidate-1": "trial-1"},
            "research_screening_only": True,
            "not_capital_confirmation": True,
            "cross_cycle_fwer_claimed": False,
            "final_oos_opened": False,
            "candidates": [
                {
                    "id": "quant-candidate-1",
                    "factors": [],
                    "model": {"code_path": str(root / "model.py")},
                    "research_label_binding": binding,
                    "research_label_binding_sha256": binding["binding_sha256"],
                }
            ],
        },
    }


def _quant_result(binding: dict, *, attempt: str) -> dict:
    cadence = build_research_execution_cadence_contract(
        str(binding["horizon_profile"])
    )
    receipt = {
        "contract_version": "fin-quant-research-ledger-receipt-v1",
        "research_tournament_id": "quant-tournament-1",
        "parent_research_tournament_id": "model-tournament-1",
        "research_tournament_manifest_sha256": "m" * 64,
        "research_trial_ids": {"quant-candidate-1": "trial-1"},
        "candidate_statuses": {"quant-candidate-1": "resource_blocked"},
        "research_screening_only": True,
        "not_capital_confirmation": True,
        "cross_cycle_fwer_claimed": False,
        "final_oos_opened": False,
        "research_label_binding_sha256": binding["binding_sha256"],
        "research_execution_cadence_sha256": cadence["evidence_sha256"],
    }
    receipt["evidence_sha256"] = canonical_sha256(receipt)
    return {
        "status": "ok",
        "attempt": attempt,
        "research_label_binding": binding,
        "research_label_binding_sha256": binding["binding_sha256"],
        "research_execution_cadence": cadence,
        "research_execution_cadence_sha256": cadence["evidence_sha256"],
        "research_trial_ledger_receipt": receipt,
        "evaluations": [
            {
                "candidate_id": "quant-candidate-1",
                "status": "resource_blocked",
                "reason_code": "test_resource_limit",
            }
        ],
    }


def _worker(tmp_path: Path, artifact_store: object) -> LocalJobWorker:
    worker = object.__new__(LocalJobWorker)
    worker.project_root = tmp_path
    worker.settings = SimpleNamespace(
        data_root=tmp_path,
        qlib_python="python",
        qlib_wsl_distro="Ubuntu-22.04",
    )
    worker.rdagent_candidates = artifact_store
    return worker


def _write_result(path: Path, result: dict) -> bytes:
    content = json.dumps(result, sort_keys=True).encode("utf-8")
    path.write_bytes(content)
    return content


@pytest.mark.no_database
def test_model_evaluation_registers_content_archive_not_retry_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_SANDBOX_IMAGE", "model-sandbox:test")
    artifacts = _CapturingArtifactStore()
    worker = _worker(tmp_path, artifacts)
    job = _model_job("run-1")
    evaluation_root = (
        tmp_path
        / "artifacts"
        / "model-evaluations"
        / "run-1"
        / job["id"]
    )
    evaluation_root.mkdir(parents=True)
    legacy_result = evaluation_root / "result.json"
    legacy_result.write_bytes(b"legacy registered bytes")

    command, working_result, _environment = worker._command(job)

    assert working_result is not None
    assert working_result != legacy_result
    assert working_result.parent.parent.name == "attempts"
    assert str(working_result) in command
    result = {"status": "ok", "evaluations": [], "attempt": "new"}
    content = _write_result(working_result, result)

    worker._import_model_evaluations(job, result, working_result)

    assert legacy_result.read_bytes() == b"legacy registered bytes"
    assert len(artifacts.storage_paths) == 1
    registered = artifacts.storage_paths[0]
    digest = hashlib.sha256(content).hexdigest()
    assert registered == (
        evaluation_root / "immutable" / "sha256" / digest / "result.json"
    )
    assert registered.read_bytes() == content
    working_result.write_bytes(b"retry workspace may change")
    assert registered.read_bytes() == content


@pytest.mark.no_database
def test_quant_bundle_evaluation_retry_registers_distinct_content_archives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_SANDBOX_IMAGE", "model-sandbox:test")
    binding = _quant_binding()
    monkeypatch.setattr(
        worker_module,
        "resolve_research_label_binding",
        lambda _payload: binding,
    )
    artifacts = _CapturingArtifactStore()
    worker = _worker(tmp_path, artifacts)
    worker.research_tournaments = _CapturingTournamentStore()
    job = _quant_job("run-quant-1", tmp_path, binding)
    evaluation_root = (
        tmp_path
        / "artifacts"
        / "quant-bundle-evaluations"
        / "run-quant-1"
        / job["id"]
    )
    evaluation_root.mkdir(parents=True)
    legacy_result = evaluation_root / "result.json"
    legacy_content = b'{"status":"historical-quant"}\n'
    legacy_result.write_bytes(legacy_content)

    working_paths: list[Path] = []
    contents: list[bytes] = []
    for attempt in ("first", "retry"):
        command, working_result, _environment = worker._command(job)
        assert working_result is not None
        assert working_result.parent.parent.name == "attempts"
        assert str(working_result) in command
        working_paths.append(working_result)
        result = _quant_result(binding, attempt=attempt)
        contents.append(_write_result(working_result, result))
        worker._import_quant_bundle_evaluation_artifact(
            job,
            result,
            working_result,
        )

    assert working_paths[0] != working_paths[1]
    for path in working_paths:
        path.write_bytes(b"discarded quant attempt workspace")
    assert legacy_result.read_bytes() == legacy_content
    assert len(artifacts.storage_paths) == 2
    for content, working, registered in zip(
        contents,
        working_paths,
        artifacts.storage_paths,
        strict=True,
    ):
        digest = hashlib.sha256(content).hexdigest()
        assert registered != working.resolve()
        assert registered == (
            evaluation_root.resolve()
            / "immutable"
            / "sha256"
            / digest
            / "result.json"
        )
        assert registered.read_bytes() == content


def test_model_evaluation_retry_preserves_registered_and_legacy_artifacts(
    tmp_path: Path,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_SANDBOX_IMAGE", "model-sandbox:test")
    run = ResearchStore(database_url).create_run(
        kind="platform_model_artifact_immutability_test",
        objective="preserve model evaluation evidence across retry",
        dataset="sealed-dataset",
        requested_by="test",
        budget={},
        config={},
        artifact_path=tmp_path / "run",
    )
    artifact_store = RDAGentCandidateStore(database_url)
    worker = _worker(tmp_path, artifact_store)
    job = _model_job(str(run["id"]), attempts=1)
    evaluation_root = (
        tmp_path
        / "artifacts"
        / "model-evaluations"
        / str(run["id"])
        / job["id"]
    )
    evaluation_root.mkdir(parents=True)
    provenance = {
        "dataset_identity_sha256": job["payload"]["dataset_identity_sha256"],
        "feature_set_id": job["payload"]["feature_set_id"],
    }

    legacy_result = evaluation_root / "result.json"
    legacy_content = b'{"status":"historical"}\n'
    legacy_result.write_bytes(legacy_content)
    legacy_artifact = artifact_store.register_or_reuse_run_artifact(
        research_run_id=str(run["id"]),
        artifact_type="model_independent_evaluation",
        storage_path=legacy_result,
        producer="quantlab_independent_evaluator",
        actor="test",
        contract_version="model-independent-evaluation-v1",
        metadata=provenance,
    )

    _command, first_working, _environment = worker._command(job)
    assert first_working is not None
    first_result = {"status": "ok", "evaluations": [], "attempt": "first"}
    first_content = _write_result(first_working, first_result)
    worker._import_model_evaluations(job, first_result, first_working)

    # Administrative resubmission may reset attempts to one.  Its execution
    # token must still allocate another workspace and another content archive.
    _command, retry_working, _environment = worker._command(job)
    assert retry_working is not None
    assert retry_working != first_working
    retry_result = {"status": "ok", "evaluations": [], "attempt": "retry"}
    retry_content = _write_result(retry_working, retry_result)
    worker._import_model_evaluations(job, retry_result, retry_working)

    first_working.write_bytes(b"discarded workspace")
    retry_working.write_bytes(b"discarded retry workspace")
    assert legacy_result.read_bytes() == legacy_content
    assert artifact_store.get_run_artifact(
        str(legacy_artifact["id"]), verify=True
    )["storage_path"] == str(legacy_result.resolve())

    artifacts = artifact_store.list_run_artifacts(str(run["id"]), verify=True)
    assert len(artifacts) == 3
    paths_by_digest = {
        str(item["content_sha256"]): Path(str(item["storage_path"]))
        for item in artifacts
    }
    for content, working in (
        (first_content, first_working),
        (retry_content, retry_working),
    ):
        digest = hashlib.sha256(content).hexdigest()
        registered = paths_by_digest[digest]
        assert registered != working.resolve()
        assert registered == (
            evaluation_root.resolve()
            / "immutable"
            / "sha256"
            / digest
            / "result.json"
        )
        assert registered.read_bytes() == content


def test_quant_bundle_retry_preserves_registered_and_legacy_artifacts(
    tmp_path: Path,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_SANDBOX_IMAGE", "model-sandbox:test")
    binding = _quant_binding()
    monkeypatch.setattr(
        worker_module,
        "resolve_research_label_binding",
        lambda _payload: binding,
    )
    run = ResearchStore(database_url).create_run(
        kind="quant_bundle_artifact_immutability_test",
        objective="preserve quant bundle evaluation evidence across retry",
        dataset="sealed-dataset",
        requested_by="test",
        budget={},
        config={},
        artifact_path=tmp_path / "run",
    )
    artifact_store = RDAGentCandidateStore(database_url)
    worker = _worker(tmp_path, artifact_store)
    worker.research_tournaments = _CapturingTournamentStore()
    job = _quant_job(str(run["id"]), tmp_path, binding)
    evaluation_root = (
        tmp_path
        / "artifacts"
        / "quant-bundle-evaluations"
        / str(run["id"])
        / job["id"]
    )
    evaluation_root.mkdir(parents=True)
    provenance = {
        "dataset_identity_sha256": job["payload"]["dataset_identity_sha256"],
        "feature_set_id": job["payload"]["feature_set_id"],
        "passed": 0,
    }

    legacy_result = evaluation_root / "result.json"
    legacy_content = b'{"status":"historical-quant"}\n'
    legacy_result.write_bytes(legacy_content)
    legacy_artifact = artifact_store.register_or_reuse_run_artifact(
        research_run_id=str(run["id"]),
        artifact_type="quant_bundle_independent_evaluation",
        storage_path=legacy_result,
        producer="quantlab_independent_evaluator",
        actor="test",
        contract_version="quant-bundle-independent-evaluation-v1",
        metadata=provenance,
    )

    working_paths: list[Path] = []
    contents: list[bytes] = []
    for attempt in ("first", "retry"):
        _command, working_result, _environment = worker._command(job)
        assert working_result is not None
        working_paths.append(working_result)
        result = _quant_result(binding, attempt=attempt)
        contents.append(_write_result(working_result, result))
        worker._import_quant_bundle_evaluation_artifact(
            job,
            result,
            working_result,
        )

    for path in working_paths:
        path.write_bytes(b"discarded quant attempt workspace")
    assert legacy_result.read_bytes() == legacy_content
    assert artifact_store.get_run_artifact(
        str(legacy_artifact["id"]), verify=True
    )["storage_path"] == str(legacy_result.resolve())

    artifacts = artifact_store.list_run_artifacts(str(run["id"]), verify=True)
    assert len(artifacts) == 3
    paths_by_digest = {
        str(item["content_sha256"]): Path(str(item["storage_path"]))
        for item in artifacts
    }
    for content, working in zip(contents, working_paths, strict=True):
        digest = hashlib.sha256(content).hexdigest()
        registered = paths_by_digest[digest]
        assert registered != working.resolve()
        assert registered == (
            evaluation_root.resolve()
            / "immutable"
            / "sha256"
            / digest
            / "result.json"
        )
        assert registered.read_bytes() == content
