from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_model_research_governance import _model_evidence, _quant_incumbent
from test_quant_bundle_model_contract import _module

from quant_platform.model_research_governance import (
    QUANT_BUNDLE_CONTRACT_VERSION,
    build_run_multiple_testing_evidence,
    canonical_sha256,
    model_metric_report,
    require_model_metric_report,
    validate_independent_model_evidence,
    validate_quant_bundle_evidence,
    validate_run_multiple_testing_evidence,
)
from quant_platform.model_strategy_contract import (
    build_model_formal_admission_binding,
    validate_model_formal_admission_binding,
)

pytestmark = pytest.mark.no_database


def _seal(value: dict, field: str = "evidence_sha256") -> dict:
    value[field] = canonical_sha256({k: v for k, v in value.items() if k != field})
    return value


def _multiple(tmp_path: Path, names: list[str]) -> dict:
    index = pd.bdate_range("2024-01-02", periods=64)
    return build_run_multiple_testing_evidence(
        research_run_id="synthetic-report-only",
        trial_series=[
            ({"name": name, "candidate_id": name, "kind": "model"},
             pd.Series(-0.001 + np.sin(np.arange(64) + i) * 0.0001, index=index))
            for i, name in enumerate(names)
        ],
        output=tmp_path,
    )


def _weak_model(tmp_path: Path) -> dict:
    value = _model_evidence()
    value["multiple_testing"] = _multiple(tmp_path, ["candidate-1"])
    for profile in value["profiles"].values():
        for cell in profile["seeds"].values():
            cell["metrics"].update({
                "ic": -0.2, "icir": -1.0, "rank_ic": -0.2, "rank_icir": -1.0,
                "information_ratio": -2.0, "annualized_excess_return_with_cost": -0.4,
                "max_drawdown": -0.8,
            })
            cell["metric_report"] = model_metric_report(cell["metrics"])
    return _seal(value)


def _validate_model(value: dict) -> dict:
    return validate_independent_model_evidence(
        value, candidate_id="candidate-1", dataset_identity_sha256="c" * 64,
        pre_final_end="2025-12-31",
    )


def test_full_model_weak_metrics_remain_exact_and_admissible(tmp_path):
    value = _weak_model(tmp_path)
    before = copy.deepcopy(value)
    assert value["multiple_testing"]["gate_passed"] is False
    assert value["multiple_testing"]["eligible_trial_names"] == []
    assert _validate_model(value) == before
    assert value == before
    for profile in value["profiles"].values():
        for cell in profile["seeds"].values():
            assert cell["metric_report"]["gate_passed"] is False
            assert len(cell["metric_report"]["failure_reasons"]) == 7


@pytest.mark.parametrize("mutation", ["missing", "nan", "inf", "false_report", "pit", "hash"])
def test_report_only_keeps_structural_model_rejections(tmp_path, mutation):
    value = _weak_model(tmp_path)
    cell = value["profiles"]["recent_3y"]["seeds"]["11"]
    if mutation == "missing":
        cell["metrics"].pop("total_cost")
    elif mutation in {"nan", "inf"}:
        cell["metrics"]["ic"] = float(mutation)
    elif mutation == "false_report":
        cell["metric_report"]["gate_passed"] = True
    elif mutation == "pit":
        cell["latest_prediction_date"] = "2026-01-01"
    else:
        cell["predictions_sha256"] = "missing"
    _seal(value)
    with pytest.raises(ValueError):
        _validate_model(value)


def test_high_pbo_is_reported_not_falsely_passed(tmp_path):
    value = _multiple(tmp_path, ["one", "two"])
    value["pbo"]["pbo"] = 0.99
    value["eligible_trial_names"] = []
    value["gate_passed"] = False
    _seal(value)
    assert validate_run_multiple_testing_evidence(value, selected_trial_name="one") == value
    assert value["pbo"]["pbo"] == 0.99


@pytest.mark.parametrize("mutation", ["pbo_nan", "pbo_range", "missing", "wrong_role",
                                     "gate", "eligible", "holm", "hash", "unknown_trial"])
def test_report_role_cannot_hide_invalid_statistical_proof(tmp_path, mutation):
    value = _multiple(tmp_path, ["one", "two"])
    selected = "one"
    if mutation == "pbo_nan":
        value["pbo"]["pbo"] = float("nan")
    elif mutation == "pbo_range":
        value["pbo"]["pbo"] = 1.1
    elif mutation == "missing":
        value.pop("raw_p_values")
    elif mutation == "wrong_role":
        value["statistical_evidence_role"] = "skip"
    elif mutation == "gate":
        value["gate_passed"] = True
    elif mutation == "eligible":
        value["eligible_trial_names"] = ["one"]
    elif mutation == "holm":
        value["holm_adjusted_p_values"][0] = 0.5
    elif mutation == "unknown_trial":
        selected = "other"
    _seal(value)
    if mutation == "hash":
        value["evidence_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_run_multiple_testing_evidence(value, selected_trial_name=selected)


def test_forced_failure_placeholder_is_never_admissible(tmp_path):
    value = _multiple(tmp_path, ["failed"])
    value["forced_raw_p_values"] = {"failed": 1.0}
    _seal(value)
    with pytest.raises(ValueError):
        validate_run_multiple_testing_evidence(value, selected_trial_name="failed")


def test_false_statistics_survive_real_formal_binding_round_trip(tmp_path):
    value = _weak_model(tmp_path)
    evidence_sha = value["evidence_sha256"]
    config = {
        "signal_source": "model_prediction", "model_candidate_id": "candidate-1",
        "model_evaluation_id": "evaluation-1", "model_code_sha256": "d" * 64,
        "model_recipe_sha256": "f" * 64, "model_evidence_sha256": evidence_sha,
        "feature_set_id": "governed-baseline", "feature_set_definition_sha256": "1" * 64,
    }
    binding = build_model_formal_admission_binding(
        config=config, candidate_manifest_sha256="2" * 64,
        dataset_identity_sha256="c" * 64, pre_final_end="2025-12-31",
        model_admission_evidence=value, model_admission_evidence_sha256=evidence_sha,
    )
    assert binding["model_grid"]["multiple_testing"] == value["multiple_testing"]
    kwargs = dict(config=config, dataset_identity_sha256="c" * 64, pre_final_end="2025-12-31")
    assert validate_model_formal_admission_binding(binding, **kwargs) == binding
    binding["model_grid"]["multiple_testing_evidence_sha256"] = "0" * 64
    _seal(binding, "binding_sha256")
    with pytest.raises(ValueError, match="binding"):
        validate_model_formal_admission_binding(binding, **kwargs)


def test_joint_comparison_producer_to_consumer_preserves_negative_outcome(tmp_path):
    module = _module()
    model = _weak_model(tmp_path / "model")
    baseline = _quant_incumbent()
    ablations = {}
    for name in ("factor_only", "model_only", "joint"):
        ablations[name] = _seal({
            "status": "passed", "source": "independent_qlib_recompute",
            "experiment_family_id": "family-1", "dataset_identity_sha256": "c" * 64,
            "execution_environment_sha256": "e" * 64, "final_oos_opened": False,
            "profiles": copy.deepcopy(model["profiles"]),
        })
    bundle = {
        "contract_version": QUANT_BUNDLE_CONTRACT_VERSION, "id": "bundle-1",
        "dataset_identity_sha256": "c" * 64, "experiment_family_id": "family-1",
        "factors": [{"candidate_id": "factor-1", "code_sha256": "a" * 64}],
        "model": {"code_sha256": "b" * 64, "recipe_sha256": "d" * 64},
        "baseline_prediction_champion": baseline,
        "baseline_prediction_champion_sha256": baseline["evidence_sha256"],
        "execution_environment_sha256": "e" * 64, "ablations": ablations,
    }
    definitions = [{"name": "incumbent:incumbent-model-1", "candidate_id": "incumbent-model-1",
                    "kind": "fin_quant_incumbent", "ablation": "incumbent"}]
    definitions.extend({"name": f"bundle-1:{name}", "candidate_id": "bundle-1",
                        "kind": "quant_bundle", "ablation": name} for name in ablations)
    definitions.append({"name": "bundle-1:joint_vs_incumbent", "candidate_id": "bundle-1",
                        "kind": "fin_quant_joint_delta", "ablation": "joint_vs_incumbent"})
    index = pd.bdate_range("2024-01-02", periods=64)
    incumbent = pd.Series(0.001 + np.sin(np.arange(64)) * 0.0001, index=index)
    joint = incumbent - 0.002
    multiple = build_run_multiple_testing_evidence(
        research_run_id="quant-synthetic",
        trial_series=[(definition, joint - incumbent if definition["ablation"]
                       == "joint_vs_incumbent" else incumbent if definition["ablation"]
                       == "incumbent" else joint) for definition in definitions],
        output=tmp_path / "quant-multiple",
    )
    item = {"candidate_id": "bundle-1", "status": "passed", "evidence": bundle}
    module._finalize_candidate_multiple_testing(
        item=item, multiple=multiple, candidate_returns={("bundle-1", "joint"): joint},
        incumbent_returns={"incumbent-model-1": incumbent}, dataset_identity_sha256="c" * 64,
    )
    assert item["status"] == "passed"
    comparison = bundle["incumbent_comparison"]
    assert comparison["statistical_evidence_role"] == "report_only"
    assert comparison["passed"] is False
    assert comparison["family_observed_mean_difference"] < 0
    assert validate_quant_bundle_evidence(bundle, dataset_identity_sha256="c" * 64) == bundle
    comparison["passed"] = True
    _seal(comparison)
    _seal(bundle, "bundle_sha256")
    with pytest.raises(ValueError, match="incumbent comparison"):
        validate_quant_bundle_evidence(bundle, dataset_identity_sha256="c" * 64)


def test_ensemble_cells_report_weak_effects_but_reject_nonfinite_data():
    cell = _model_evidence()["profiles"]["recent_3y"]["seeds"]["11"]
    cell["metrics"]["ic"] = -0.2
    cell["metric_report"] = model_metric_report(cell["metrics"])
    require_model_metric_report(cell, context="ensemble")
    assert cell["metric_report"]["gate_passed"] is False
    cell["metrics"]["total_cost"] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        require_model_metric_report(cell, context="ensemble")


def _ensemble_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_model_ensemble.py"
    spec = importlib.util.spec_from_file_location("report_only_ensemble", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ensemble_actual_finalizer_keeps_false_cell_and_statistics_reports(tmp_path):
    module = _ensemble_module()
    multiple = _multiple(tmp_path, ["ensemble-1"])
    item = {"ensemble_id": "ensemble-1", "status": "pending_multiple_testing",
            "all_metric_cells_passed": False, "evidence": {"final_oos_opened": False}}
    module._finalize_ensemble_evaluation(item, multiple)
    assert item["status"] == "passed"
    evidence = item["evidence"]
    assert evidence["all_metric_cells_passed"] is False
    assert evidence["multiple_testing"]["gate_passed"] is False
    assert evidence["multiple_testing"]["eligible_trial_names"] == []
    assert evidence["evidence_sha256"] == canonical_sha256(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )
    assert evidence["multiple_testing"] == multiple


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "invalid"])
@pytest.mark.parametrize("factory", [_module, _ensemble_module])
def test_portfolio_producer_does_not_replace_invalid_metrics_with_zero(factory, value):
    module = factory()
    with pytest.raises(ValueError, match="not (numeric|finite)"):
        module._finite(value)
