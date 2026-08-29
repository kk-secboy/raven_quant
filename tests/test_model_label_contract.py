from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from quant_platform.model_recompute import HORIZON_MODEL_DATA_CONTRACT_VERSION
from quant_platform.model_research_governance import (
    LEGACY_MODEL_PREDICTION_HORIZON_SESSIONS,
    canonical_sha256,
    resolve_model_label_contract,
)
from quant_platform.research_horizon import (
    LONG_1_3Y,
    SHORT_1_5D,
    SWING_1_6M,
    research_horizon_contract,
)
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


def _window(profile: str) -> tuple[dict[str, Any], str]:
    horizon = research_horizon_contract(profile)
    value = {
        "contract_version": "research-window-v1",
        "horizon_profile": profile,
        "horizon_contract_sha256": horizon.sha256,
        "label_horizons_sessions": list(horizon.label_horizons_sessions),
        "purge_sessions": horizon.purge_sessions,
        "embargo_sessions": horizon.embargo_sessions,
        "label_maturity_enforced": True,
    }
    return value, canonical_sha256(value)


@pytest.mark.parametrize(
    ("profile", "selected", "expression", "purge", "embargo"),
    (
        (SHORT_1_5D, 5, "Ref($close,-6)/Ref($close,-1)-1", 6, 6),
        (SWING_1_6M, 126, "Ref($close,-127)/Ref($close,-1)-1", 127, 127),
        (LONG_1_3Y, 252, "Ref($close,-253)/Ref($close,-1)-1", 253, 253),
    ),
)
def test_active_model_runs_default_to_the_longest_allowed_prediction_label(
    profile: str,
    selected: int,
    expression: str,
    purge: int,
    embargo: int,
) -> None:
    window, digest = _window(profile)

    contract = resolve_model_label_contract(
        research_window_contract=window,
        research_window_contract_sha256=digest,
    )

    assert contract["legacy"] is False
    assert contract["label_horizon_sessions"] == selected
    assert contract["label_reference_offset_sessions"] == selected + 1
    assert contract["label_expression"] == expression
    assert contract["purge_sessions"] == purge
    assert contract["embargo_sessions"] == embargo


def test_model_run_can_select_another_preregistered_horizon_but_not_invent_one() -> None:
    window, digest = _window(SWING_1_6M)

    selected = resolve_model_label_contract(
        research_window_contract=window,
        research_window_contract_sha256=digest,
        label_horizon_sessions=63,
    )

    assert selected["label_horizon_sessions"] == 63
    assert selected["label_expression"] == "Ref($close,-64)/Ref($close,-1)-1"
    with pytest.raises(ValueError, match="not allowed"):
        resolve_model_label_contract(
            research_window_contract=window,
            research_window_contract_sha256=digest,
            label_horizon_sessions=60,
        )


def test_legacy_model_label_keeps_the_historical_expression_and_marks_it() -> None:
    contract = resolve_model_label_contract()

    assert contract["legacy"] is True
    assert contract["horizon_profile"] == "legacy_ambiguous"
    assert contract["label_horizon_sessions"] == LEGACY_MODEL_PREDICTION_HORIZON_SESSIONS
    assert contract["label_reference_offset_sessions"] == 2
    assert contract["label_expression"] == "Ref($close,-2)/Ref($close,-1)-1"


def test_isolated_runner_reverifies_the_platform_label_contract() -> None:
    runner_path = Path(__file__).resolve().parents[1] / "scripts" / "model_sandbox_runner.py"
    spec = importlib.util.spec_from_file_location("model_sandbox_label_contract", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    window, window_sha256 = _window(SHORT_1_5D)
    label = resolve_model_label_contract(
        research_window_contract=window,
        research_window_contract_sha256=window_sha256,
    )
    manifest = {
        "research_window_contract": window,
        "research_window_contract_sha256": window_sha256,
        "model_label_contract": label,
        "model_label_contract_sha256": canonical_sha256(label),
    }

    assert module.resolve_manifest_label_contract(manifest) == label
    assert module.HORIZON_MODEL_DATA_CONTRACT_VERSION == HORIZON_MODEL_DATA_CONTRACT_VERSION
    manifest["model_label_contract"] = {**label, "label_horizon_sessions": 3}
    with pytest.raises(ValueError, match="digest is invalid"):
        module.resolve_manifest_label_contract(manifest)


def test_active_model_label_requires_the_exact_window_digest() -> None:
    window, _digest = _window(LONG_1_3Y)

    with pytest.raises(ValueError, match="digest is missing or invalid"):
        resolve_model_label_contract(
            research_window_contract=window,
            research_window_contract_sha256="0" * 64,
        )


def test_independent_model_batch_propagates_the_window_and_selected_label() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "evaluate_model_batch.py"
    ).read_text(encoding="utf-8")

    assert '"research_window_contract": manifest.get("research_window_contract")' in source
    assert '"research_window_contract_sha256": manifest.get(' in source
    assert '"label_horizon_sessions": _candidate.get("label_horizon_sessions")' in source
    assert '"model_label_contract_sha256": result[' in source


def test_worker_freezes_research_window_and_selected_label_in_model_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, window_sha256 = _window(SWING_1_6M)
    feature_set = {
        "id": "features-1",
        "definition_sha256": "f" * 64,
        "features": {"alpha": {"expression": "$close"}},
    }
    monkeypatch.setenv("MODEL_SANDBOX_IMAGE", "model-sandbox:test")
    worker = object.__new__(LocalJobWorker)
    worker.project_root = tmp_path
    worker.settings = SimpleNamespace(
        data_root=tmp_path,
        qlib_python="python",
        qlib_wsl_distro="Ubuntu-22.04",
        mlflow_tracking_uri="sqlite:///tracking.db",
    )
    job = {
        "id": "model-evaluate-1",
        "kind": "model_evaluate",
        "payload": {
            "research_run_id": "run-1",
            "dataset_path": str(tmp_path / "qlib" / "daily"),
            "dataset_identity_sha256": "d" * 64,
            "feature_set_id": feature_set["id"],
            "feature_set_definition_sha256": feature_set["definition_sha256"],
            "feature_set": feature_set,
            "candidates": [
                {
                    "id": "candidate-1",
                    "code_path": str(tmp_path / "candidate.py"),
                    "label_horizon_sessions": 63,
                }
            ],
            "research_window_contract": window,
            "research_window_contract_sha256": window_sha256,
            "label_horizon_sessions": 63,
        },
    }

    worker._command(job)

    manifest = json.loads(
        (
            tmp_path
            / "artifacts"
            / "model-evaluations"
            / "run-1"
            / "model-evaluate-1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["research_window_contract"] == window
    assert manifest["research_window_contract_sha256"] == window_sha256
    assert manifest["label_horizon_sessions"] == 63
