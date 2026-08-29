from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import quant_platform.worker as worker_module
from quant_platform.strategy_research_evaluation import (
    STRATEGY_FULL_STACK_MODE,
    STRATEGY_POLICY_ONLY_MODE,
)
from quant_platform.worker import Worker

pytestmark = pytest.mark.no_database


class _ArtifactStore:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.register_calls: list[dict[str, Any]] = []

    def seed_json(
        self,
        *,
        artifact_id: str,
        run_id: str,
        artifact_type: str,
        path: Path,
        value: dict[str, Any],
        source_iteration: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        path.write_bytes(encoded)
        row = {
            "id": artifact_id,
            "research_run_id": run_id,
            "artifact_type": artifact_type,
            "storage_path": str(path.resolve()),
            "content_sha256": hashlib.sha256(encoded).hexdigest(),
            "manifest_json": {
                "source_iteration": source_iteration,
                "metadata": deepcopy(metadata or {}),
            },
        }
        self.rows[artifact_id] = row
        return deepcopy(row)

    def get_run_artifact(self, artifact_id: str, *, verify: bool) -> dict[str, Any]:
        assert verify is True
        row = deepcopy(self.rows[artifact_id])
        path = Path(row["storage_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["content_sha256"]:
            raise ValueError("registered artifact content changed")
        return row

    def find_run_artifact(
        self,
        *,
        research_run_id: str,
        artifact_type: str,
        content_sha256: str,
        verify: bool,
    ) -> dict[str, Any] | None:
        assert verify is True
        for row in self.rows.values():
            if (
                row["research_run_id"] == research_run_id
                and row["artifact_type"] == artifact_type
                and row["content_sha256"] == content_sha256
            ):
                return self.get_run_artifact(str(row["id"]), verify=True)
        return None

    def register_run_artifact(self, **kwargs: Any) -> dict[str, Any]:
        path = Path(kwargs["storage_path"]).resolve()
        row = {
            "id": f"registered-{len(self.register_calls) + 1}",
            "research_run_id": str(kwargs["research_run_id"]),
            "artifact_type": str(kwargs["artifact_type"]),
            "storage_path": str(path),
            "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "manifest_json": {
                "source_iteration": kwargs.get("source_iteration"),
                "metadata": deepcopy(kwargs.get("metadata") or {}),
            },
        }
        self.register_calls.append(deepcopy(kwargs))
        self.rows[str(row["id"])] = row
        return deepcopy(row)

    def list_run_artifacts(
        self,
        research_run_id: str,
        *,
        artifact_types: tuple[str, ...],
        verify: bool,
    ) -> list[dict[str, Any]]:
        assert verify is True
        return [
            self.get_run_artifact(str(row["id"]), verify=True)
            for row in self.rows.values()
            if row["research_run_id"] == research_run_id
            and row["artifact_type"] in artifact_types
        ]


class _ParameterExperiments:
    def __init__(self, experiment: dict[str, Any]) -> None:
        self.experiment = experiment
        self.ensure_calls: list[dict[str, Any]] = []
        self.attach_calls: list[tuple[str, str]] = []

    def get(self, experiment_id: str) -> dict[str, Any]:
        if experiment_id != self.experiment["id"]:
            raise KeyError(experiment_id)
        return deepcopy(self.experiment)

    def ensure_strategy_research_competition(self, **kwargs: Any) -> dict[str, Any]:
        self.ensure_calls.append(deepcopy(kwargs))
        return {
            "experiment": {
                "id": "full-experiment",
                "status": "queued",
                "job_id": None,
            },
            "job_payload": {
                "parameter_experiment_id": "full-experiment",
                "strategy_version_id": kwargs["strategy_version"]["id"],
                "dataset": kwargs["dataset"]["name"],
                "dataset_path": kwargs["dataset"]["path"],
                "dataset_identity_sha256": kwargs["dataset"]["provenance"][
                    "dataset_identity_sha256"
                ],
                "strategy_evaluation_mode": STRATEGY_FULL_STACK_MODE,
                "strategy_competition_plan_sha256": kwargs["plan"]["plan_sha256"],
                "strategy_competition_stage": "full_stack",
            },
        }

    def attach_job(self, experiment_id: str, job_id: str) -> None:
        self.attach_calls.append((experiment_id, job_id))


class _JobStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        kind: str,
        payload: dict[str, Any],
        log_path: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "kind": kind,
                "payload": deepcopy(payload),
                "log_path": Path(log_path),
                **deepcopy(kwargs),
            }
        )
        return {"id": "full-job"}


class _ResearchStore:
    def __init__(self) -> None:
        self.runtime: dict[str, Any] = {"existing": "kept"}
        self.status = "evaluating"
        self.mark_calls: list[dict[str, Any]] = []

    def get_run(self, run_id: str) -> dict[str, Any]:
        assert run_id == "run-1"
        return {"status": self.status, "runtime": deepcopy(self.runtime)}

    def mark_run(self, run_id: str, status: str, **kwargs: Any) -> None:
        self.mark_calls.append(
            {"run_id": run_id, "status": status, **deepcopy(kwargs)}
        )
        self.status = status
        if kwargs.get("runtime") is not None:
            self.runtime = deepcopy(kwargs["runtime"])


def _fake_stage_builder(
    plan: dict[str, Any],
    *,
    stage_name: str,
    experiment_result: dict[str, Any],
    artifact_root: Path,
    prerequisite_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert artifact_root.name == "experiment-1"
    if stage_name == "full_stack" and (
        not isinstance(prerequisite_evidence, dict)
        or prerequisite_evidence.get("stage") != "policy_only"
        or prerequisite_evidence.get("gate_passed") is not True
        or prerequisite_evidence.get("plan_sha256") != plan["plan_sha256"]
    ):
        raise ValueError("full-stack evaluation requires passed policy-only evidence")
    gate_passed = experiment_result.get("gate_passed") is True
    evidence = {
        "stage": stage_name,
        "plan_sha256": plan["plan_sha256"],
        "gate_passed": gate_passed,
        "evidence_sha256": ("e" if gate_passed else "f") * 64,
    }
    return {
        "parameter_experiment_id": experiment_result.get("experiment_id"),
        "artifact_sha256": ("c" if gate_passed else "d") * 64,
        "evidence": evidence,
    }


def _settlement_fixture(
    tmp_path: Path,
    *,
    stage: str = "policy_only",
    policy_gate_passed: bool = True,
) -> tuple[Worker, dict[str, Any], _ArtifactStore, _ParameterExperiments, _JobStore]:
    run_id = "run-1"
    version_id = "version-1"
    experiment_id = "experiment-1"
    plan = {
        "research_run_id": run_id,
        "plan_sha256": "a" * 64,
    }
    artifacts = _ArtifactStore()
    plan_row = artifacts.seed_json(
        artifact_id="plan-artifact",
        run_id=run_id,
        artifact_type="fin_strategy_competition_plan",
        path=tmp_path / "registered" / "plan.json",
        value=plan,
        source_iteration=7,
    )
    experiment_root = tmp_path / "parameter-experiments" / experiment_id
    experiment_root.mkdir(parents=True)
    mode = (
        STRATEGY_POLICY_ONLY_MODE
        if stage == "policy_only"
        else STRATEGY_FULL_STACK_MODE
    )
    governance = {
        "mode": mode,
        "stage": stage,
        "plan_sha256": plan["plan_sha256"],
        "research_run_id": run_id,
        "dataset_identity_sha256": "b" * 64,
    }
    parameter_experiments = _ParameterExperiments(
        {
            "id": experiment_id,
            "strategy_version_id": version_id,
            "dataset": "daily-snapshot",
            "artifact_path": str(experiment_root),
            "periods": {"governance": governance},
        }
    )
    jobs = _JobStore()
    research = _ResearchStore()
    worker = object.__new__(Worker)
    worker.settings = SimpleNamespace(data_root=tmp_path)
    worker.rdagent_candidates = artifacts
    worker.parameter_experiments = parameter_experiments
    worker.store = jobs
    worker.research = research
    worker.strategies = SimpleNamespace(
        get_version=lambda requested: {
            "id": requested,
            "status": "draft",
            "config": {},
        }
    )
    worker.notify = lambda: None
    payload = {
        "strategy_evaluation_mode": mode,
        "strategy_competition_stage": stage,
        "research_run_id": run_id,
        "strategy_version_id": version_id,
        "parameter_experiment_id": experiment_id,
        "dataset": "daily-snapshot",
        "dataset_path": str(tmp_path / "qlib"),
        "dataset_identity_sha256": "b" * 64,
        "strategy_competition_plan_artifact_id": plan_row["id"],
        "strategy_competition_plan_path": plan_row["storage_path"],
        "strategy_competition_plan_content_sha256": plan_row["content_sha256"],
        "strategy_competition_plan_sha256": plan["plan_sha256"],
    }
    if stage == "full_stack":
        policy_evidence = {
            "parameter_experiment_id": "policy-experiment",
            "artifact_sha256": "3" * 64,
            "evidence": {
                "stage": "policy_only",
                "plan_sha256": plan["plan_sha256"],
                "gate_passed": policy_gate_passed,
                "evidence_sha256": "4" * 64,
            },
        }
        policy_row = artifacts.seed_json(
            artifact_id="policy-artifact",
            run_id=run_id,
            artifact_type="fin_strategy_policy_only_evaluation",
            path=tmp_path / "registered" / "policy.json",
            value=policy_evidence,
        )
        payload.update(
            {
                "strategy_policy_evidence_artifact_id": policy_row["id"],
                "strategy_policy_evidence_path": policy_row["storage_path"],
                "strategy_policy_evidence_content_sha256": policy_row[
                    "content_sha256"
                ],
            }
        )
    return worker, {"payload": payload}, artifacts, parameter_experiments, jobs


def test_non_strategy_parameter_experiment_has_no_settlement_side_effects() -> None:
    worker = object.__new__(Worker)

    assert (
        worker._settle_fin_strategy_experiment(
            {"payload": {"strategy_evaluation_mode": "ordinary"}}, {}
        )
        is None
    )


def test_policy_pass_queues_only_full_stack_with_same_plan_and_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, job, artifacts, experiments, jobs = _settlement_fixture(tmp_path)
    monkeypatch.setattr(
        worker_module,
        "build_strategy_stage_artifact_from_parameter_experiment",
        _fake_stage_builder,
    )

    settlement = worker._settle_fin_strategy_experiment(
        job, {"experiment_id": "experiment-1", "gate_passed": True}
    )

    assert settlement is not None
    assert settlement["stage"] == "policy_only"
    assert settlement["gate_passed"] is True
    assert settlement["next_job"]["stage"] == "full_stack"
    assert len(experiments.ensure_calls) == 1
    prepared = experiments.ensure_calls[0]
    assert prepared["stage"] == "full_stack"
    assert prepared["plan"]["plan_sha256"] == "a" * 64
    assert prepared["dataset"]["provenance"]["dataset_identity_sha256"] == "b" * 64
    assert len(jobs.calls) == 1
    assert jobs.calls[0]["kind"] == "parameter_experiment"
    assert jobs.calls[0]["idempotency_key"] == f"fin-strategy:{'a' * 64}:full_stack"
    queued_payload = jobs.calls[0]["payload"]
    assert queued_payload["strategy_competition_plan_artifact_id"] == "plan-artifact"
    assert queued_payload["strategy_policy_evidence_artifact_id"].startswith(
        "registered-"
    )
    policy_row = artifacts.rows[queued_payload["strategy_policy_evidence_artifact_id"]]
    assert queued_payload["strategy_policy_evidence_path"] == policy_row["storage_path"]
    assert queued_payload["strategy_policy_evidence_content_sha256"] == policy_row[
        "content_sha256"
    ]
    assert experiments.attach_calls == [("full-experiment", "full-job")]


def test_policy_reject_registers_evidence_without_queueing_next_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, job, artifacts, experiments, jobs = _settlement_fixture(tmp_path)
    monkeypatch.setattr(
        worker_module,
        "build_strategy_stage_artifact_from_parameter_experiment",
        _fake_stage_builder,
    )

    settlement = worker._settle_fin_strategy_experiment(
        job, {"experiment_id": "experiment-1", "gate_passed": False}
    )

    assert settlement is not None
    assert settlement["gate_passed"] is False
    assert settlement["next_gate"] == "research_rejected"
    assert "next_job" not in settlement
    assert experiments.ensure_calls == []
    assert jobs.calls == []
    assert [call["artifact_type"] for call in artifacts.register_calls] == [
        "fin_strategy_policy_only_evaluation"
    ]


def test_full_stack_rejects_policy_artifact_that_did_not_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, job, _, _, _ = _settlement_fixture(
        tmp_path, stage="full_stack", policy_gate_passed=False
    )
    monkeypatch.setattr(
        worker_module,
        "build_strategy_stage_artifact_from_parameter_experiment",
        _fake_stage_builder,
    )

    with pytest.raises(
        ValueError, match="requires passed policy-only evidence"
    ):
        worker._settle_fin_strategy_experiment(
            job, {"experiment_id": "experiment-1", "gate_passed": True}
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "strategy_competition_plan_sha256",
            "9" * 64,
            "plan binding changed",
        ),
        (
            "strategy_competition_plan_path",
            "missing-plan.json",
            "artifact identity changed",
        ),
        (
            "strategy_competition_plan_content_sha256",
            "8" * 64,
            "artifact identity changed",
        ),
    ],
)
def test_plan_path_and_hash_drift_fail_closed_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
    message: str,
) -> None:
    worker, job, _, _, _ = _settlement_fixture(tmp_path)

    def should_not_evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("tampered identity reached evaluation")

    monkeypatch.setattr(
        worker_module,
        "build_strategy_stage_artifact_from_parameter_experiment",
        should_not_evaluate,
    )
    job["payload"][field] = replacement

    with pytest.raises(ValueError, match=message):
        worker._settle_fin_strategy_experiment(job, {})


@pytest.mark.parametrize(
    ("experiment_field", "replacement"),
    [
        ("strategy_version_id", "other-version"),
        ("dataset", "other-dataset"),
    ],
)
def test_parameter_experiment_identity_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiment_field: str,
    replacement: str,
) -> None:
    worker, job, _, experiments, _ = _settlement_fixture(tmp_path)
    experiments.experiment[experiment_field] = replacement

    def should_not_evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("wrong experiment reached evaluation")

    monkeypatch.setattr(
        worker_module,
        "build_strategy_stage_artifact_from_parameter_experiment",
        should_not_evaluate,
    )

    with pytest.raises(ValueError, match="experiment binding changed"):
        worker._settle_fin_strategy_experiment(job, {})


def test_evaluation_mode_must_match_preregistered_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, job, _, _, _ = _settlement_fixture(tmp_path)
    job["payload"]["strategy_evaluation_mode"] = STRATEGY_FULL_STACK_MODE

    def should_not_evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("wrong mode reached evaluation")

    monkeypatch.setattr(
        worker_module,
        "build_strategy_stage_artifact_from_parameter_experiment",
        should_not_evaluate,
    )

    with pytest.raises(ValueError, match="stage.*mode|mode.*stage"):
        worker._settle_fin_strategy_experiment(job, {})


def test_experiment_governance_must_bind_same_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, job, _, experiments, _ = _settlement_fixture(tmp_path)
    experiments.experiment["periods"]["governance"]["plan_sha256"] = "7" * 64

    def should_not_evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("wrong governed plan reached evaluation")

    monkeypatch.setattr(
        worker_module,
        "build_strategy_stage_artifact_from_parameter_experiment",
        should_not_evaluate,
    )

    with pytest.raises(ValueError, match="experiment.*binding|governance.*binding"):
        worker._settle_fin_strategy_experiment(job, {})


def test_stage_evaluation_artifact_registration_is_idempotent(tmp_path: Path) -> None:
    worker, _, artifacts, _, _ = _settlement_fixture(tmp_path)
    value = _fake_stage_builder(
        {"research_run_id": "run-1", "plan_sha256": "a" * 64},
        stage_name="policy_only",
        experiment_result={"experiment_id": "experiment-1", "gate_passed": True},
        artifact_root=tmp_path / "experiment-1",
    )

    first = worker._write_fin_strategy_stage_artifact(
        run_id="run-1",
        stage="policy_only",
        strategy_version_id="version-1",
        source_iteration=7,
        value=value,
    )
    second = worker._write_fin_strategy_stage_artifact(
        run_id="run-1",
        stage="policy_only",
        strategy_version_id="version-1",
        source_iteration=7,
        value=value,
    )

    assert first == second
    assert len(artifacts.register_calls) == 1
    assert Path(first["artifact_path"]).read_bytes() == Path(
        second["artifact_path"]
    ).read_bytes()


def test_run_reconciliation_queues_only_the_deterministic_full_stack_winner(
    tmp_path: Path,
) -> None:
    worker, _, artifacts, _, _ = _settlement_fixture(tmp_path)
    versions = ["version-a", "version-b"]
    worker.research.runtime = {
        "strategy_policy_evaluation_jobs": [
            {"strategy_version_id": version_id} for version_id in versions
        ]
    }
    for index, version_id in enumerate(versions, start=1):
        plan_sha = str(index) * 64
        policy_sha = str(index + 2) * 64
        full_sha = str(index + 4) * 64
        policy = {
            "evidence": {
                "plan_sha256": plan_sha,
                "gate_passed": True,
                "evidence_sha256": policy_sha,
            }
        }
        full = {
            "evidence": {
                "plan_sha256": plan_sha,
                "gate_passed": True,
                "prerequisite_evidence_sha256": policy_sha,
                "evidence_sha256": full_sha,
                "paired_block_bootstrap": {
                    "observed_mean_difference": 0.003 if index == 1 else 0.002,
                },
                "alpha_spending": {
                    "holm_equivalent_adjusted_p_value": 0.01,
                },
                "pbo": {"pbo": 0.10},
            }
        }
        artifacts.seed_json(
            artifact_id=f"policy-{version_id}",
            run_id="run-1",
            artifact_type="fin_strategy_policy_only_evaluation",
            path=tmp_path / "registered" / f"policy-{version_id}.json",
            value=policy,
            metadata={"strategy_version_id": version_id},
        )
        artifacts.seed_json(
            artifact_id=f"full-{version_id}",
            run_id="run-1",
            artifact_type="fin_strategy_full_stack_evaluation",
            path=tmp_path / "registered" / f"full-{version_id}.json",
            value=full,
            metadata={"strategy_version_id": version_id},
        )
    queued: list[dict[str, Any]] = []
    worker._queue_fin_strategy_formal_oos = lambda **kwargs: (
        queued.append(deepcopy(kwargs))
        or {"job_id": "formal-job", "backtest_id": "formal-backtest"}
    )

    result = worker._reconcile_fin_strategy_competition(
        run_id="run-1",
        payload={
            "dataset_path": str(tmp_path / "qlib"),
            "dataset_lineage_id": "a" * 64,
            "dataset_identity_sha256": "b" * 64,
        },
    )

    assert result["status"] == "winner_selected"
    assert result["winner_strategy_version_id"] == "version-a"
    assert len(queued) == 1
    assert queued[0]["version_id"] == "version-a"
    winner_rows = [
        call
        for call in artifacts.register_calls
        if call["artifact_type"] == "fin_strategy_governed_winner"
    ]
    assert len(winner_rows) == 1
    assert winner_rows[0]["metadata"]["all_branches_settled"] is True
