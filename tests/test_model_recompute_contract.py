import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from quant_platform.model_recompute import (
    MODEL_DATA_CONTRACT_VERSION,
    MODEL_RESOURCE_POLICY_VERSION,
    execute_model_candidate,
    governed_checkpoint_filename,
    governed_model_resource_policy,
    validate_model_code,
    verify_governed_checkpoint,
)

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


def test_model_code_is_limited_to_pure_torch_definition() -> None:
    validate_model_code(VALID_MODEL)
    with pytest.raises(ValueError, match="forbidden module"):
        validate_model_code("import os\nclass X: pass\nmodel_cls = X\n")
    with pytest.raises(ValueError, match="forbidden builtin"):
        validate_model_code(
            "class X:\n    def forward(self):\n        return open('/etc/passwd')\nmodel_cls = X\n"
        )


def test_model_recompute_fails_closed_without_immutable_inputs(tmp_path: Path) -> None:
    code = tmp_path / "model.py"
    code.write_text(VALID_MODEL, encoding="utf-8")
    provider = tmp_path / "qlib"
    provider.mkdir()
    runner = tmp_path / "runner.py"
    runner.write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="code hash"):
        execute_model_candidate(
            code_path=code,
            provider_path=provider,
            manifest={
                "candidate_id": "model-1",
                "code_sha256": "0" * 64,
                "final_oos_opened": False,
            },
            workspace=tmp_path / "work",
            runner_path=runner,
        )


def test_model_sandbox_runner_uses_the_executor_contract_versions() -> None:
    runner_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "model_sandbox_runner.py"
    )
    spec = importlib.util.spec_from_file_location("model_sandbox_runner_contract", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.MODEL_RESOURCE_POLICY_VERSION == MODEL_RESOURCE_POLICY_VERSION
    assert module.MODEL_DATA_CONTRACT_VERSION == MODEL_DATA_CONTRACT_VERSION


def test_live_inference_requires_an_immutable_checkpoint_before_docker(
    tmp_path: Path,
) -> None:
    code = tmp_path / "model.py"
    code.write_text(VALID_MODEL, encoding="utf-8")
    provider = tmp_path / "qlib"
    provider.mkdir()
    runner = tmp_path / "runner.py"
    runner.write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable fitted checkpoint"):
        execute_model_candidate(
            code_path=code,
            provider_path=provider,
            manifest={
                "candidate_id": "model-1",
                "code_sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
                "model_engine": "ridge_baseline",
                "final_oos_opened": False,
                "inference_only": True,
            },
            workspace=tmp_path / "work",
            runner_path=runner,
            allow_inference=True,
        )


def test_ridge_checkpoint_is_numeric_and_not_a_pickle(tmp_path: Path) -> None:
    checkpoint = tmp_path / governed_checkpoint_filename("ridge_baseline")
    checkpoint.write_text(
        json.dumps(
            {
                "coefficients": [0.25, -0.5],
                "feature_count": 2,
                "format": "ridge-numeric-v1",
                "intercept": 0.0,
                "model_engine": "ridge_baseline",
                "model_spec_sha256": "a" * 64,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    verify_governed_checkpoint(
        checkpoint,
        model_engine="ridge_baseline",
        checkpoint_format="ridge_numeric_json",
        expected_sha256=digest,
    )
    checkpoint.write_bytes(b"\x80\x04pickle")
    with pytest.raises(ValueError, match="immutable verification"):
        verify_governed_checkpoint(
            checkpoint,
            model_engine="ridge_baseline",
            checkpoint_format="ridge_numeric_json",
            expected_sha256=digest,
        )


def test_sandbox_inference_branch_never_calls_fit() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "model_sandbox_runner.py"
    ).read_text(encoding="utf-8")
    inference_branch = source.split(
        "    if inference_only:\n        checkpoint = _require_inference_checkpoint",
        1,
    )[1].split("    else:\n        checkpoint = output / checkpoint_filename", 1)[0]
    assert "model.fit(" not in inference_branch
    assert "weights_only=True" in inference_branch


def test_governed_model_resource_policy_caps_compute_not_research_data() -> None:
    screening = governed_model_resource_policy(
        model_type="TimeSeries",
        model_engine="platform_gru",
        requested_hyperparameters={
            "n_epochs": 500,
            "early_stop": 100,
            "batch_size": 8192,
            "lr": 2e-4,
        },
        stage="screening",
        requested_timeout_seconds=99_999,
        seed=11,
    )
    assert screening["effective_training_hyperparameters"] == {
        "n_epochs": 4,
        "early_stop": 3,
        "batch_size": 2048,
        "lr": 2e-4,
        "weight_decay": 1e-4,
    }
    assert screening["limits"]["timeout_seconds"] == 1800
    assert screening["limits"]["date_segments_modified"] is False
    assert screening["limits"]["universe_modified"] is False
    assert screening["limits"]["qlib_evaluation_concurrency_cap"] == 3
    assert screening["limits"]["reserved_service_resource_fraction"] == 0.25

    full = governed_model_resource_policy(
        model_type="TimeSeries",
        model_engine="platform_transformer",
        requested_hyperparameters={"n_epochs": 500, "early_stop": 100},
        stage="full_validation",
        requested_timeout_seconds=99_999,
        seed=29,
    )
    assert full["effective_training_hyperparameters"]["n_epochs"] == 12
    assert full["effective_training_hyperparameters"]["early_stop"] == 3
    assert full["limits"]["timeout_seconds"] == 7200
    assert full["limits"]["cpu_only_timeseries_cap"] is True
    assert full["limits"]["exclusive_concurrency"] == 1

    ridge = governed_model_resource_policy(
        model_type="Tabular",
        model_engine="ridge_baseline",
        requested_hyperparameters={"n_epochs": 500, "early_stop": 100},
        stage="screening",
        requested_timeout_seconds=99_999,
        seed=11,
    )
    assert ridge["effective_training_hyperparameters"]["alpha"] == 1.0
    assert ridge["effective_training_hyperparameters"]["fit_intercept"] is False
    assert "train_only" in ridge["effective_training_hyperparameters"]["normalization"]
    assert ridge["seed_policy"]["allowed_seeds"] == [11]
    assert ridge["limits"]["cpu_only_timeseries_cap"] is False


def test_governed_model_resource_policy_freezes_lanes_and_seed_rules() -> None:
    lightgbm = governed_model_resource_policy(
        model_type="Tabular",
        model_engine="lightgbm_baseline",
        requested_hyperparameters={"subsample": 1.0},
        stage="full_validation",
        requested_timeout_seconds=7200,
        seed=47,
    )
    assert lightgbm["effective_training_hyperparameters"]["random_sampling"] is True
    assert lightgbm["seed_policy"]["allowed_seeds"] == [11, 29, 47]
    with pytest.raises(ValueError, match="tabular lane"):
        governed_model_resource_policy(
            model_type="TimeSeries",
            model_engine="ridge_baseline",
            requested_hyperparameters={},
            stage="screening",
            requested_timeout_seconds=1800,
            seed=11,
        )
    with pytest.raises(ValueError, match="seed is outside"):
        governed_model_resource_policy(
            model_type="TimeSeries",
            model_engine="platform_gru",
            requested_hyperparameters={},
            stage="full_validation",
            requested_timeout_seconds=7200,
            seed=13,
        )


def test_governed_gru_template_uses_explicit_dropout() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "quant_platform"
        / "model_templates.py"
    ).read_text(encoding="utf-8")
    compile(source, "model_templates.py", "exec")
    gru_source = source.split("class GovernedGRU", 1)[1].split(
        "class _SinusoidalPosition", 1
    )[0]
    assert "num_layers=1" in gru_source
    assert "dropout=0.0" in gru_source
    assert "self.output_dropout = nn.Dropout(p=0.1)" in gru_source
    assert "num_timesteps: int = SEQUENCE_LENGTH" in source
    transformer_source = source.split("class GovernedTransformer", 1)[1]
    assert "d_model=32" in transformer_source
    assert "nhead=4" in transformer_source
    assert "num_layers=2" in transformer_source
    assert "dropout=0.1" in transformer_source


def test_governed_model_resource_policy_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        governed_model_resource_policy(
            model_type="Tabular",
            requested_hyperparameters={"lr": float("nan")},
            stage="screening",
            requested_timeout_seconds=1800,
        )
