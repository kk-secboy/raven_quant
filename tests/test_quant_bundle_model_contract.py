from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from quant_platform.model_recompute import ModelResourceLimitError
from quant_platform.model_research_governance import REQUIRED_MODEL_METRICS, canonical_sha256
from quant_platform.research_execution_cadence import (
    build_research_execution_cadence_contract,
)
from quant_platform.research_horizon import SHORT_1_5D
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


VALID_MODEL = """
import torch
from torch import nn

class SafeModel(nn.Module):
    def __init__(self, num_features=20):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x)

model_cls = SafeModel
"""


def _module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_quant_bundle.py"
    spec = importlib.util.spec_from_file_location("evaluate_quant_bundle_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quant_bundle_materializes_qlib_signal_record_dependencies() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "evaluate_quant_bundle.py"
    ).read_text(encoding="utf-8")
    save_index = source.index('"pred.pkl": predictions[["score"]]')
    boundary_index = source.index("resolve_qlib_portfolio_calendar_boundary(")
    portfolio_index = source.index("record = PortAnaRecord(")
    assert '"label.pkl": labels.to_frame("label")' in source
    assert '"signal": "<PRED>"' in source
    assert boundary_index < save_index < portfolio_index
    assert '"class": "GovernedDPlusOneTopkDropoutStrategy"' in source
    assert '"module_path": "quant_platform.qlib_research_strategy"' in source
    assert '"research_execution_cadence": manifest[' in source
    assert '"end_time": periods["valid_end"]' in source
    assert "Qlib quant-bundle portfolio record generation was skipped" in source


def test_fin_quant_rejects_legacy_recent_only_prediction_selection() -> None:
    worker = object.__new__(LocalJobWorker)
    selection = {
        "contract_version": "prediction-champion-selection-v1",
        "selection_data": "pre_final_only",
        "final_oos_opened": False,
        "selected_kind": "model",
        "selected_candidate_id": "model-1",
    }
    selection["evidence_sha256"] = canonical_sha256(selection)
    with pytest.raises(ValueError, match="prediction champion evidence is invalid"):
        worker._freeze_fin_quant_baseline(  # noqa: SLF001
            {
                "prediction_champion": {
                    "kind": "model",
                    "candidate_id": "model-1",
                },
                "prediction_champion_evidence": selection,
            }
        )


def test_candidate_specific_holm_failure_does_not_poison_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    index = pd.date_range("2024-01-02", periods=60, freq="B")
    incumbent = pd.Series(0.001, index=index)
    candidate_returns = {("good", "joint"): incumbent + 0.002}
    evaluations = [
        {
            "candidate_id": "bad",
            "status": "passed",
            "evidence": {
                "baseline_prediction_champion": {"candidate_id": "incumbent"}
            },
        },
        {
            "candidate_id": "good",
            "status": "passed",
            "evidence": {
                "baseline_prediction_champion": {"candidate_id": "incumbent"}
            },
        },
    ]
    multiple = {
        "trial_names": [
            "bad:joint_vs_incumbent",
            "good:joint_vs_incumbent",
        ],
        "raw_p_values": [1.0, 0.01],
        "holm_adjusted_p_values": [1.0, 0.02],
        "trial_daily_means": [0.0, 0.002],
        "eligible_trial_names": ["good:joint_vs_incumbent"],
    }
    validated: list[str] = []
    monkeypatch.setattr(
        module,
        "validate_quant_bundle_evidence",
        lambda evidence, **_: validated.append(str(evidence["id"])),
    )
    evaluations[1]["evidence"]["id"] = "good"
    module._finalize_candidate_multiple_testing_batch(
        evaluations=evaluations,
        multiple=multiple,
        candidate_returns=candidate_returns,
        incumbent_returns={"incumbent": incumbent},
        dataset_identity_sha256="d" * 64,
    )
    assert evaluations[0]["status"] == "failed"
    assert "candidate-specific" in evaluations[0]["error"]
    assert evaluations[1]["status"] == "passed"
    assert validated == ["good"]


def _bundle(module: ModuleType, tmp_path: Path) -> dict[str, Any]:
    code_path = tmp_path / "candidate_model.py"
    code_path.write_text(VALID_MODEL, encoding="utf-8")
    bundle = {
        "id": "bundle-1",
        "experiment_family_id": "family-1",
        "factors": [{"candidate_id": "factor-1"}],
        "model": {
            "code_path": str(code_path),
            "code_sha256": module.file_sha256(code_path),
            "recipe_sha256": "a" * 64,
            "model_type": "TimeSeries",
            "model_engine": "platform_gru",
            "model_hyperparameters": {"model_engine": "platform_gru"},
            "training_hyperparameters": {"n_epochs": 100},
        },
    }
    baseline_code_path = tmp_path / "incumbent_model.py"
    baseline_code_path.write_text(VALID_MODEL, encoding="utf-8")
    frozen_baseline = {
        "contract_version": "fin-quant-baseline-prediction-v1",
        "kind": "model",
        "candidate_id": "incumbent-model-1",
        "candidate_manifest_sha256": "2" * 64,
        "admission_evidence_sha256": "3" * 64,
        "dataset": "cn-test",
        "dataset_identity_sha256": "b" * 64,
        "pre_final_end": "2023-12-29",
        "final_oos_start": "2024-01-02",
        "final_oos_end": "2024-12-31",
        "feature_set_id": "governed-baseline",
        "feature_set_definition_sha256": "1" * 64,
        "model": {
            "candidate_id": "incumbent-model-1",
            "code_path": str(baseline_code_path),
            "code_sha256": module.file_sha256(baseline_code_path),
            "model_type": "Tabular",
            "model_engine": "ridge_baseline",
            "architecture": {},
            "model_hyperparameters": {"model_engine": "ridge_baseline"},
            "training_hyperparameters": {"alpha": 1.0},
            "recipe_sha256": "4" * 64,
        },
        "profiles": {},
        "quant_retraining_supported": True,
        "selection_evidence_sha256": "5" * 64,
    }
    frozen_baseline["evidence_sha256"] = module.canonical_sha256(
        frozen_baseline
    )
    bundle["baseline_prediction_champion"] = frozen_baseline
    bundle["baseline_prediction_runtime"] = json.loads(
        json.dumps(frozen_baseline)
    )
    return bundle


def _profiles() -> list[dict[str, Any]]:
    return [
        {
            "id": profile_id,
            "periods": {
                "train_start": "2018-01-02",
                "train_end": "2022-12-30",
                "valid_start": "2023-01-03",
                "valid_end": "2023-12-29",
                "test_start": "2024-01-02",
                "test_end": "2024-12-31",
            },
        }
        for profile_id in ("recent_3y", "balanced_5y", "robust_10y")
    ]


def _manifest() -> dict[str, Any]:
    return {
        "dataset_identity_sha256": "b" * 64,
        "universe": "cn_all",
        "benchmark": "SH000300",
        "model_timeout_seconds": 7200,
        "research_execution_cadence": (
            build_research_execution_cadence_contract(SHORT_1_5D)
        ),
    }


def test_quant_candidate_lane_uses_challenger_and_exact_frozen_incumbent(
    tmp_path: Path,
) -> None:
    module = _module()
    bundle = _bundle(module, tmp_path)
    assert module._ablation_model_config("model_only", bundle)["model_engine"] == (
        "platform_gru"
    )
    assert module._ablation_model_config("joint", bundle)["model_engine"] == "platform_gru"
    baseline = module._ablation_model_config("factor_only", bundle)
    assert baseline["model_engine"] == "ridge_baseline"
    assert baseline["model_type"] == "Tabular"

    nested = dict(bundle["model"])
    nested.pop("model_engine")
    assert module._frozen_candidate_model_engine(nested) == "platform_gru"
    with pytest.raises(ValueError, match="changed inside the frozen recipe"):
        module._frozen_candidate_model_engine(
            {
                **bundle["model"],
                "model_hyperparameters": {"model_engine": "platform_transformer"},
            }
        )
    with pytest.raises(ValueError, match="no frozen model_engine"):
        module._frozen_candidate_model_engine({"model_hyperparameters": {}})


def _ensemble_bundle(module: ModuleType, tmp_path: Path) -> dict[str, Any]:
    bundle = _bundle(module, tmp_path)
    feature_set = module.resolve_feature_set("governed-baseline")
    components = []
    for index, (candidate_id, model_engine) in enumerate(
        (("model-a", "ridge_baseline"), ("model-b", "lightgbm_baseline")),
        start=1,
    ):
        code_path = tmp_path / f"{candidate_id}.py"
        code_path.write_text(VALID_MODEL, encoding="utf-8")
        components.append(
            {
                "model_candidate_id": candidate_id,
                "model_family": model_engine,
                "weight": 0.5,
                "model_manifest_sha256": str(index) * 64,
                "model_admission_evidence_sha256": str(index + 2) * 64,
                "prediction_grid_sha256": str(index + 4) * 64,
                "candidate_manifest_sha256": str(index) * 64,
                "admission_evidence_sha256": str(index + 2) * 64,
                "feature_set_id": feature_set["id"],
                "feature_set_definition_sha256": feature_set[
                    "definition_sha256"
                ],
                "feature_set": feature_set,
                "model": {
                    "candidate_id": candidate_id,
                    "code_path": str(code_path),
                    "code_sha256": module.file_sha256(code_path),
                    "model_type": "Tabular",
                    "model_engine": model_engine,
                    "model_hyperparameters": {"model_engine": model_engine},
                    "training_hyperparameters": {},
                    "recipe_sha256": str(index + 6) * 64,
                },
                "profiles": {},
            }
        )
    frozen = {
        key: value
        for key, value in bundle["baseline_prediction_champion"].items()
        if key not in {"model", "profiles", "evidence_sha256"}
    }
    frozen.update(
        {
            "kind": "ensemble",
            "candidate_id": "ensemble-1",
            "feature_set_id": None,
            "feature_set_definition_sha256": None,
            "combiner": "equal_rank",
            "stacking": False,
            "components": components,
            "profiles": {},
            "member_retraining_contract_version": (
                "fin-quant-ensemble-member-retraining-v1"
            ),
            "quant_retraining_supported": True,
        }
    )
    frozen["evidence_sha256"] = module.canonical_sha256(frozen)
    bundle["baseline_prediction_champion"] = frozen
    bundle["baseline_prediction_runtime"] = json.loads(json.dumps(frozen))
    return bundle


def test_quant_ensemble_incumbent_requires_member_dispatch(
    tmp_path: Path,
) -> None:
    module = _module()
    bundle = _ensemble_bundle(module, tmp_path)

    with pytest.raises(
        module.UnsupportedQuantBaseline,
        match="ensemble_factor_only_requires_member_dispatch",
    ):
        module._ablation_model_config("factor_only", bundle)

    bundle["baseline_prediction_runtime"]["candidate_id"] = "ensemble-substitute"
    with pytest.raises(ValueError, match="runtime identity changed"):
        module._runtime_baseline(bundle)


def test_quant_ensemble_factor_only_retrains_every_member_and_equal_ranks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    bundle = _ensemble_bundle(module, tmp_path)
    instruments = [f"S{index:03d}" for index in range(60)]
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2023-12-29")], instruments],
        names=["datetime", "instrument"],
    )
    observed: list[dict[str, Any]] = []

    def fake_member(**kwargs: Any) -> tuple[dict[str, Any], str]:
        observed.append(kwargs)
        workspace = Path(kwargs["workspace"])
        output = workspace / "output"
        output.mkdir(parents=True)
        values = list(range(60))
        if "model-b" in kwargs["execution_candidate_id"]:
            values = [value * 7 % 60 for value in values]
        predictions = pd.DataFrame({"score": values}, index=index)
        predictions_path = output / "predictions.parquet"
        predictions.to_parquet(predictions_path)
        checkpoint_path = output / "checkpoint.bin"
        checkpoint_path.write_bytes(b"checkpoint")
        report_path = output / "portfolio_report.parquet"
        report_path.write_bytes(b"member-report")
        return (
            {
                "status": "passed",
                "metrics": {name: 0.03 for name in REQUIRED_MODEL_METRICS},
                "latest_prediction_date": "2023-12-29",
                "predictions_path": str(predictions_path),
                "predictions_sha256": module.file_sha256(predictions_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": module.file_sha256(checkpoint_path),
                "checkpoint_format": "test",
                "portfolio_report_path": str(report_path),
                "portfolio_report_sha256": module.file_sha256(report_path),
                "execution_evidence": {"evidence_sha256": "a" * 64},
                "execution_evidence_sha256": "a" * 64,
                "execution_environment_sha256": "b" * 64,
                "coverage": {"coverage_gate_passed": True},
                "model_engine": kwargs["config"]["model_engine"],
                "resource_stage": "full_validation",
                "resource_policy": {},
            },
            "b" * 64,
        )

    def fake_portfolio(**kwargs: Any) -> tuple[dict[str, Any], Path, dict[str, Any]]:
        output = Path(kwargs["workspace"]) / "output"
        output.mkdir(parents=True, exist_ok=False)
        report_path = output / "portfolio_report.parquet"
        report_path.write_bytes(b"ensemble-report")
        return (
            {name: 0.03 for name in REQUIRED_MODEL_METRICS},
            report_path,
            {
                "contract_version": "qlib-portfolio-calendar-boundary-v1",
                "backtest_end": "2023-12-29",
                "interval_end": "2024-01-02",
                "interval_end_has_market_data": False,
            },
        )

    monkeypatch.setattr(module, "_execute_single_model_cell", fake_member)
    monkeypatch.setattr(module, "_evaluate_equal_rank_portfolio", fake_portfolio)
    monkeypatch.setattr(
        module,
        "verify_model_prediction_artifact",
        lambda *_args, **_kwargs: {"coverage_gate_passed": True},
    )
    monkeypatch.setattr(module, "_calendar_between", lambda *_args: ["2023-12-29"])
    result, environment = module._execute_ensemble_factor_only_cell(
        bundle=bundle,
        view=tmp_path / "view",
        factor_values_path=tmp_path / "factors.parquet",
        periods=_profiles()[0]["periods"],
        seed=11,
        resource_stage="full_validation",
        workspace=tmp_path / "ensemble-cell",
        manifest=_manifest(),
    )

    assert environment == "b" * 64
    assert result["status"] == "passed"
    assert result["prediction_component_kind"] == "ensemble"
    assert result["combiner"] == "equal_rank"
    assert result["stacking"] is False
    assert len(result["member_artifacts"]) == 2
    assert len(observed) == 2
    assert all(item["factor_values_path"] == tmp_path / "factors.parquet" for item in observed)
    combined = pd.read_parquet(result["predictions_path"])
    assert combined["score"].between(0.0, 1.0).all()
    assert result["execution_evidence"]["execution"] == "sequential_cpu_only"
    assert (
        result["execution_evidence"]["portfolio_calendar_boundary"][
            "interval_end_has_market_data"
        ]
        is False
    )


def test_quant_incumbent_allows_only_runtime_path_translation(tmp_path: Path) -> None:
    module = _module()
    bundle = _bundle(module, tmp_path)
    frozen = json.loads(json.dumps(bundle["baseline_prediction_champion"]))
    runtime = json.loads(json.dumps(frozen))
    runtime_code_path = runtime["model"]["code_path"]
    frozen["model"]["code_path"] = r"E:\\sealed-host\\incumbent_model.py"
    frozen["evidence_sha256"] = module.canonical_sha256(
        {key: value for key, value in frozen.items() if key != "evidence_sha256"}
    )
    runtime = json.loads(json.dumps(frozen))
    runtime["model"]["code_path"] = runtime_code_path

    validated = module._validate_baseline_prediction(
        baseline=frozen,
        runtime_baseline=runtime,
        manifest_baseline=frozen,
        dataset_identity_sha256="b" * 64,
        feature_set_definition_sha256="1" * 64,
    )
    assert validated["candidate_id"] == "incumbent-model-1"

    runtime["model"]["training_hyperparameters"] = {"alpha": 99.0}
    with pytest.raises(ValueError, match="runtime model recipe changed"):
        module._validate_baseline_prediction(
            baseline=frozen,
            runtime_baseline=runtime,
            manifest_baseline=frozen,
            dataset_identity_sha256="b" * 64,
            feature_set_definition_sha256="1" * 64,
        )


def test_quant_full_grid_preserves_policy_and_execution_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    bundle = _bundle(module, tmp_path)
    observed: list[dict[str, Any]] = []

    def fake_execute_model_candidate(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest = dict(kwargs["manifest"])
        observed.append(manifest)
        workspace = Path(kwargs["workspace"])
        output = workspace / "output"
        output.mkdir(parents=True)
        (output / "predictions.parquet").write_bytes(b"predictions")
        checkpoint_name = module.governed_checkpoint_filename(
            str(manifest["model_engine"])
        )
        (output / checkpoint_name).write_bytes(b"checkpoint")
        report_path = output / "portfolio_report.parquet"
        report_path.write_bytes(b"portfolio")
        policy = {
            "contract_version": "policy-v-test",
            "stage": manifest["resource_stage"],
            "model_engine": manifest["model_engine"],
        }
        execution = {
            "evidence_sha256": "c" * 64,
            "execution_environment_sha256": "d" * 64,
            "resource_policy": policy,
        }
        return (
            {
                "metrics": {name: 0.03 for name in REQUIRED_MODEL_METRICS},
                "latest_prediction_date": manifest["periods"]["valid_end"],
                "predictions_sha256": "e" * 64,
                "checkpoint_sha256": "f" * 64,
                "checkpoint_format": {
                    "ridge_baseline": "ridge_numeric_json",
                    "lightgbm_baseline": "lightgbm_text",
                    "platform_gru": "pytorch_state_dict",
                    "platform_transformer": "pytorch_state_dict",
                    "rdagent_pytorch": "pytorch_state_dict",
                }[manifest["model_engine"]],
                "portfolio_report_sha256": module.file_sha256(report_path),
                "resource_policy": policy,
                "research_execution_cadence_sha256": manifest[
                    "research_execution_cadence"
                ]["evidence_sha256"],
            },
            execution,
        )

    monkeypatch.setattr(module, "execute_model_candidate", fake_execute_model_candidate)
    monkeypatch.setattr(module, "_calendar_between", lambda *_args: ["2023-01-03"])
    monkeypatch.setattr(
        module,
        "verify_model_prediction_artifact",
        lambda *_args, **_kwargs: {"passed": True},
    )
    common = {
        "bundle": bundle,
        "feature_set": {"definition_sha256": "1" * 64},
        "profiles": _profiles(),
        "view": tmp_path / "view",
        "factor_values_path": tmp_path / "factors.parquet",
        "output": tmp_path / "candidate",
        "manifest": _manifest(),
    }
    candidate_result = module._run_ablation(name="model_only", **common)
    baseline_result = module._run_ablation(name="factor_only", **common)

    assert candidate_result["status"] == "passed"
    assert baseline_result["status"] == "passed"
    assert len(observed) == 18
    assert {item["model_engine"] for item in observed[:9]} == {"platform_gru"}
    assert {item["model_engine"] for item in observed[9:]} == {"ridge_baseline"}
    assert {item["resource_stage"] for item in observed} == {"full_validation"}
    for result in (candidate_result, baseline_result):
        assert set(result["profiles"]) == {
            "recent_3y",
            "balanced_5y",
            "robust_10y",
        }
        for profile in result["profiles"].values():
            assert set(profile["seeds"]) == {"11", "29", "47"}
            for cell in profile["seeds"].values():
                assert cell["resource_policy"]["stage"] == "full_validation"
                assert cell["execution_evidence"]["evidence_sha256"] == "c" * 64


def test_quant_resource_limits_are_audited_as_resource_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    bundle = _bundle(module, tmp_path)
    observed: list[dict[str, Any]] = []

    def resource_limited(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest = dict(kwargs["manifest"])
        observed.append(manifest)
        workspace = Path(kwargs["workspace"])
        workspace.mkdir(parents=True)
        environment = {"sandbox_image": "model@sha256:" + "a" * 64}
        runtime_manifest = {
            **manifest,
            "resource_policy": {
                "contract_version": "policy-v-test",
                "stage": manifest["resource_stage"],
                "model_engine": manifest["model_engine"],
            },
            "execution_environment": environment,
            "execution_environment_sha256": canonical_sha256(environment),
        }
        (workspace / "manifest.json").write_text(
            json.dumps(runtime_manifest), encoding="utf-8"
        )
        raise ModelResourceLimitError("governed CPU budget exceeded")

    monkeypatch.setattr(module, "execute_model_candidate", resource_limited)
    common = {
        "bundle": bundle,
        "feature_set": {"definition_sha256": "1" * 64},
        "profiles": _profiles(),
        "view": tmp_path / "view",
        "factor_values_path": tmp_path / "factors.parquet",
        "output": tmp_path / "candidate",
        "manifest": _manifest(),
    }
    screen = module._run_resource_screen(**common)
    assert screen["status"] == "resource_blocked"
    assert screen["profile_id"] == "balanced_5y"
    assert screen["seed"] == 11
    assert screen["model_engine"] == "platform_gru"
    assert screen["resource_policy"]["stage"] == "screening"
    assert screen["execution_evidence"]["status"] == "resource_blocked"

    full = module._run_ablation(
        name="joint",
        **{**common, "output": tmp_path / "full-candidate"},
    )
    assert full["status"] == "resource_blocked"
    assert full["reason_code"] == "full_validation_resource_limit"
    cell = full["profiles"]["recent_3y"]["seeds"]["11"]
    assert cell["status"] == "resource_blocked"
    assert cell["resource_policy"]["stage"] == "full_validation"
    assert cell["execution_evidence_sha256"] == cell["execution_evidence"][
        "evidence_sha256"
    ]
    assert [item["resource_stage"] for item in observed] == [
        "screening",
        "full_validation",
    ]


def test_quant_run_level_failure_summary_keeps_the_cell_root_cause() -> None:
    module = _module()
    summary = module._failed_evaluation_summary(
        [
            {
                "candidate_id": "bundle-7",
                "status": "failed",
                "error": "joint/robust_10y/seed-29: model data contract mismatch",
            },
            {
                "candidate_id": "bundle-8",
                "status": "resource_blocked",
                "error": "CPU budget exceeded",
            },
        ]
    )
    assert summary == (
        "bundle-7: joint/robust_10y/seed-29: model data contract mismatch"
    )
