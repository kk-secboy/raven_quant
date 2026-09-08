from __future__ import annotations

import copy
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_platform.worker as worker_module
from quant_platform.model_recompute import ModelResourceLimitError
from quant_platform.model_research_governance import (
    canonical_sha256,
    file_sha256,
    model_metric_gate_failures,
    require_model_metric_gate,
)
from quant_platform.research_execution_cadence import build_research_execution_cadence_contract
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


def _metrics() -> dict[str, float]:
    return {
        "ic": 0.03,
        "icir": 0.6,
        "rank_ic": 0.04,
        "rank_icir": 0.7,
        "information_ratio": 0.3,
        "annualized_excess_return_with_cost": 0.03,
        "max_drawdown": -0.1,
        "total_cost": 0.01,
        "average_turnover": 0.02,
    }


@pytest.fixture
def screen(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_model_batch.py"
    spec = importlib.util.spec_from_file_location("feature_screen_outcome_batch", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider = tmp_path / "provider"
    (provider / "metadata").mkdir(parents=True)
    (provider / "metadata" / "provenance.json").write_text(
        json.dumps({"dataset_identity_sha256": "d" * 64}), encoding="utf-8"
    )
    (provider.parent / "receipt.json").write_text("{}", encoding="utf-8")
    periods = {
        "train_start": "2018-01-02", "train_end": "2022-12-30",
        "valid_start": "2023-01-03", "valid_end": "2023-12-29",
        "test_start": "2024-07-02", "test_end": "2026-08-20",
    }
    cadence = build_research_execution_cadence_contract("short_1_5d")
    binding = {
        "binding_sha256": "b" * 64,
        "horizon_profile": "short_1_5d",
        "research_window_contract": {},
        "research_window_contract_sha256": "w" * 64,
        "dataset_identity_sha256": "d" * 64,
        "label_horizon_sessions": 2,
        "periods": periods,
    }
    feature_set = {"id": "test-feature", "definition_sha256": "f" * 64}
    manifest = {
        "research_run_id": "run-1",
        "research_label_binding": binding,
        "research_label_binding_sha256": binding["binding_sha256"],
        "research_execution_cadence": cadence,
        "research_window_contract": {},
        "research_window_contract_sha256": "w" * 64,
        "dataset_identity_sha256": "d" * 64,
        "label_horizon_sessions": 2,
        "feature_set_id": feature_set["id"],
        "feature_set_definition_sha256": feature_set["definition_sha256"],
        "evaluation_stage": "feature_screen",
        "evaluation_profiles": [{"id": "recent_3y", "periods": periods}],
        "research_tournament_id": "tournament-1",
        "candidate_bindings": [{"candidate_id": "candidate-1", "trial_id": "trial-1"}],
        "candidates": [{
            "id": "candidate-1", "code_path": "stub.py", "code_sha256": "c" * 64,
            "model_type": "Tabular", "model_engine": "lightgbm_baseline",
        }],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_path = (
        tmp_path / "data" / "artifacts" / "model-evaluations" / "run-1" / "job-1"
        / "attempts" / ("attempt-0001-" + "a" * 32) / "result.json"
    )
    monkeypatch.setattr(module, "validate_research_label_binding", lambda _: binding)
    monkeypatch.setattr(worker_module, "validate_research_label_binding", lambda _: binding)
    monkeypatch.setattr(module, "verify_qlib_output_manifest", lambda *_: None)
    monkeypatch.setattr(module, "prepare_model_dataset_view", lambda *_a, **_k: provider)
    monkeypatch.setattr(module, "resolve_feature_set", lambda *_: feature_set)
    monkeypatch.setattr(module, "calendar_between", lambda *_: ["2023-01-03", "2023-12-29"])
    monkeypatch.setattr(
        module, "verify_model_prediction_artifact",
        lambda *_a, **_k: {"coverage_gate_passed": True},
    )
    state = SimpleNamespace(metrics=_metrics(), execution_error=None, calls=[], batches=[])

    def execute(**kwargs):
        state.calls.append(kwargs)
        if state.execution_error:
            raise state.execution_error
        output = kwargs["workspace"] / "output"
        output.mkdir(parents=True)
        paths = {}
        for key, filename in (
            ("predictions", "predictions.parquet"),
            ("checkpoint", "checkpoint.txt"),
            ("portfolio_report", "portfolio_report.parquet"),
        ):
            path = output / filename
            path.write_bytes(filename.encode())
            paths[key + "_sha256"] = file_sha256(path)
        return {
            **paths, "metrics": state.metrics, "checkpoint_format": "lightgbm_text",
            "resource_policy": {"stage": kwargs["manifest"]["resource_stage"]},
            "model_label_contract": {}, "latest_prediction_date": periods["valid_end"],
            "model_label_contract_sha256": "l" * 64,
            "research_execution_cadence_sha256": cadence["evidence_sha256"],
        }, {"evidence_sha256": "e" * 64, "execution_environment_sha256": "n" * 64}

    class DirectCells:
        def run_many(self, calls):
            state.batches.append(len(calls))
            outcomes = []
            for call in calls:
                call = dict(call)
                cell_manifest = call["manifest"]
                call.setdefault("workspace", output_path.parent / "test-cells"
                                / cell_manifest["evaluation_profile_id"]
                                / f"seed-{cell_manifest['seed']}")
                try:
                    result, evidence = execute(**call)
                    outcomes.append({
                        "status": "completed", "result": result,
                        "execution_evidence": evidence, "workspace": str(call["workspace"]),
                        "receipt_sha256": "r" * 64, "reused": False,
                    })
                except ModelResourceLimitError as exc:
                    outcomes.append({"status": "resource_blocked", "error": str(exc)})
                except Exception as exc:
                    outcomes.append({"status": "failed", "error": str(exc)})
            return outcomes

    monkeypatch.setattr(module, "_new_cell_executor", lambda *_: DirectCells())
    monkeypatch.setattr(module, "arm_model_batch_owner", lambda *_: nullcontext())
    monkeypatch.setattr(module, "governed_checkpoint_filename", lambda _: "checkpoint.txt")

    def run():
        monkeypatch.setattr(sys, "argv", [
            "evaluate_model_batch", "--provider-uri", str(provider),
            "--manifest", str(manifest_path), "--output", str(output_path),
        ])
        module.main()
        return json.loads(output_path.read_text(encoding="utf-8"))

    transitions = []
    candidate_transitions = []
    worker = LocalJobWorker.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root=tmp_path / "data")
    worker.research_tournaments = SimpleNamespace(
        get_trial=lambda _: {
            "tournament_id": "tournament-1", "candidate_id": "candidate-1", "status": "queued",
        },
        transition_trial=lambda trial_id, status, **values: transitions.append((status, values)),
    )
    worker.rdagent_candidates = SimpleNamespace(
        register_or_reuse_run_artifact=lambda **_: {"id": "artifact-1"},
        transition_candidate=lambda *_, **values: candidate_transitions.append(values),
    )
    job = {"id": "job-1", "kind": "model_evaluate", "payload": manifest}
    return SimpleNamespace(
        run=run, state=state, worker=worker, job=job, output_path=output_path,
        transitions=transitions, candidate_transitions=candidate_transitions,
        module=module, manifest=manifest, manifest_path=manifest_path,
    )


@pytest.fixture
def full_model_grid(screen, monkeypatch):
    manifest = screen.manifest
    manifest["evaluation_stage"] = "model_full"
    periods = manifest["evaluation_profiles"][0]["periods"]
    manifest["evaluation_profiles"] = [
        {"id": name, "periods": {**periods, "train_start": start}}
        for name, start in (
            ("recent_3y", periods["train_start"]),
            ("balanced_5y", "2016-01-04"),
            ("robust_10y", "2011-01-04"),
        )
    ]
    screen.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(screen.module.pd, "read_parquet", lambda *_: pd.DataFrame(
        {"return": 0.001, "bench": 0.0, "cost": 0.0001},
        index=pd.date_range("2023-01-03", periods=60, freq="B"),
    ))
    monkeypatch.setattr(screen.module, "build_run_multiple_testing_evidence", lambda **_: {})
    monkeypatch.setattr(
        screen.module, "validate_independent_model_evidence", lambda *_a, **_k: None,
    )
    return screen


def test_full_model_grid_reuses_first_full_cell_and_preserves_all_nine(full_model_grid):
    grid = full_model_grid
    grid.state.metrics["information_ratio"] = -1.0
    item = grid.run()["evaluations"][0]
    assert item["status"] == "passed"
    assert grid.state.batches == [1, 8]
    calls = [call["manifest"] for call in grid.state.calls]
    assert (calls[0]["evaluation_profile_id"], calls[0]["seed"]) == ("balanced_5y", 11)
    expected = {(profile["id"], seed)
                for profile in grid.manifest["evaluation_profiles"] for seed in (11, 29, 47)}
    assert len(calls) == len(expected) == 9
    assert {(call["evaluation_profile_id"], call["seed"]) for call in calls} == expected
    assert {call["resource_stage"] for call in calls} == {"full_validation"}
    assert all(call["final_oos_opened"] is False for call in calls)
    assert all(call["periods"] == next(
        profile["periods"] for profile in grid.manifest["evaluation_profiles"]
        if profile["id"] == call["evaluation_profile_id"]
    ) for call in calls)
    evidence = item["evidence"]
    first = evidence["profiles"]["balanced_5y"]["seeds"]["11"]
    feasibility = evidence["resource_screen"]
    assert feasibility["mode"] == "reuse_first_full_cell"
    assert feasibility["included_in_full_validation_grid"] is True
    assert feasibility["cell_execution"] == first["cell_execution"]
    assert feasibility["execution_evidence_sha256"] == first["execution_evidence_sha256"]


@pytest.mark.parametrize("failure", [
    "resource", "runtime", "invalid_metrics", "coverage", "old_screen",
])
def test_first_full_cell_failure_stops_remaining_grid(full_model_grid, monkeypatch, failure):
    grid = full_model_grid
    if failure == "resource":
        grid.state.execution_error = ModelResourceLimitError("memory exhausted")
    elif failure == "runtime":
        grid.state.execution_error = ValueError("invalid output contract")
    elif failure == "invalid_metrics":
        grid.state.metrics["ic"] = float("nan")
    elif failure == "coverage":
        def invalid_coverage(*_a, **_k):
            raise ValueError("prediction coverage is incomplete")
        monkeypatch.setattr(grid.module, "verify_model_prediction_artifact", invalid_coverage)
    else:
        execute = grid.module._execute_cell

        def old_screen_result(*args, **kwargs):
            result = execute(*args, **kwargs)
            result[0]["resource_policy"]["stage"] = "screening"
            return result

        monkeypatch.setattr(grid.module, "_execute_cell", old_screen_result)
    item = grid.run()["evaluations"][0]
    assert grid.state.batches == [1]
    assert len(grid.state.calls) == 1
    assert item["status"] == ("resource_blocked" if failure == "resource" else "failed")
    if failure == "resource":
        assert item["reason_code"] == "full_validation_resource_limit"


def test_weak_effects_keep_evidence_and_settle_completed_screen(screen):
    screen.state.metrics["information_ratio"] = -0.65128008
    screen.state.metrics["annualized_excess_return_with_cost"] = -0.079300529
    screen.state.metrics["max_drawdown"] = -0.48789213
    result = screen.run()
    item = result["evaluations"][0]
    assert item["status"] == "passed"
    assert item["reason_code"] == "model_metrics_report_only"
    cell = item["evidence"]["cells"][0]
    assert cell["metrics"] == screen.state.metrics
    assert len(cell["gate_reasons"]) == 3
    assert cell["metric_report"]["gate_passed"] is False
    assert cell["metric_report"]["statistical_evidence_role"] == "report_only"
    assert cell["coverage"]["coverage_gate_passed"]
    screen.worker._import_model_evaluations(screen.job, result, screen.output_path)
    assert [status for status, _ in screen.transitions] == ["running", "passed"]
    assert screen.transitions[-1][1]["evidence"] == item["evidence"]
    assert screen.candidate_transitions[0]["status"] == "invalidated"
    assert not any(x["status"] in {"failed", "resource_blocked"} for x in result["evaluations"])


def test_passed_screen_retains_screening_only_boundary(screen):
    result = screen.run()
    assert result["evaluations"][0]["status"] == "passed"
    screen.worker._import_model_evaluations(screen.job, result, screen.output_path)
    assert [status for status, _ in screen.transitions] == ["running", "passed"]
    assert screen.candidate_transitions[0]["status"] == "invalidated"


@pytest.mark.parametrize("value", [float("nan"), None, "missing"])
def test_malformed_metrics_remain_operational_failure(screen, value):
    if value == "missing":
        screen.state.metrics.pop("ic")
    else:
        screen.state.metrics["ic"] = value
    result = screen.run()
    assert result["evaluations"][0]["status"] == "failed"
    assert "evidence" not in result["evaluations"][0]


def test_resource_exhaustion_stays_distinct_from_economic_report(screen):
    screen.state.execution_error = ModelResourceLimitError("governed memory limit")
    result = screen.run()
    assert result["evaluations"][0]["status"] == "resource_blocked"
    screen.worker._import_model_evaluations(screen.job, result, screen.output_path)
    assert screen.transitions[-1][0] == "failed"
    assert screen.transitions[-1][1]["evidence"]["investment_hypothesis_rejected"] is False


def test_worker_recomputes_gate_instead_of_trusting_claimed_pass(screen):
    screen.state.metrics["ic"] = -0.1
    result = screen.run()
    forged = copy.deepcopy(result)
    item = forged["evaluations"][0]
    item["status"] = "passed"
    cell = item["evidence"]["cells"][0]
    cell["gate_status"] = "passed"
    cell["gate_reasons"] = []
    evidence = item["evidence"]
    evidence.pop("evidence_sha256")
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    item["evidence_sha256"] = evidence["evidence_sha256"]
    screen.output_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with its metric gate"):
        screen.worker._import_model_evaluations(screen.job, forged, screen.output_path)
    assert not screen.candidate_transitions


def test_report_only_still_requires_unchanged_artifacts(screen):
    screen.state.metrics["ic"] = -0.1
    result = screen.run()
    cell = result["evaluations"][0]["evidence"]["cells"][0]
    Path(cell["predictions_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="predictions_path artifact changed"):
        screen.worker._import_model_evaluations(screen.job, result, screen.output_path)
    assert not screen.candidate_transitions
    assert [status for status, _ in screen.transitions] == ["running"]


def test_shared_failure_list_preserves_existing_fail_closed_gate():
    metrics = _metrics()
    metrics["ic"] = 0.019
    failures = model_metric_gate_failures(metrics)
    assert failures == ["ic=0.019 < 0.02"]
    with pytest.raises(ValueError, match="screen failed model-metric-gate-v1: ic=0.019 < 0.02"):
        require_model_metric_gate(metrics, context="screen")
