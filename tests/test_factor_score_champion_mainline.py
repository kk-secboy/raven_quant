from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import date

import pytest

from quant_platform.autopilot_champion_selection import (
    AutopilotResearchChampionSelector,
    _validate_factor_sota_member_evaluation,
)
from quant_platform.factor_score_champion import (
    build_factor_score_champion_contract,
    factor_score_champion_feature_set,
    factor_score_champion_signal_config,
    validate_factor_score_champion_contract,
)
from quant_platform.qlib_factor_baseline import bind_factor_source_config
from quant_platform.research_store import FactorGatePolicy
from quant_platform.strategy_recipes import get_strategy_recipe
from quant_platform.strategy_research_evaluation import (
    build_public_strategy_control_config,
    build_transparent_full_stack_control_config,
)
from quant_platform.strategy_research_signal_binding import (
    build_strategy_research_signal_binding,
    research_feature_set_for_champion_selection,
    validate_strategy_research_signal_binding,
)
from quant_platform.strategy_rule_compiler import (
    compile_strategy_rule_policy,
    validate_strategy_rule_ir,
)
from quant_platform.strategy_rule_ir import canonical_sha256

pytestmark = pytest.mark.no_database


def _raw_factor_member(*, weight: float | None = None) -> dict:
    return {
        "member_rank": 0,
        "factor_candidate_id": "factor-1",
        "factor_definition_id": "definition-00000001",
        "factor_definition_sha256": "d" * 64,
        "factor_evaluation_id": "evaluation-1",
        "factor_evaluation_evidence_sha256": "e" * 64,
        "factor_evaluation_dataset_identity_sha256": "a" * 64,
        "incremental_evidence_sha256": "f" * 64,
        "candidate_code_sha256": "1" * 64,
        "expression": "Ref($close, 1) / $close - 1",
        "direction": -1,
        "weight": weight,
    }


def _passing_factor_metrics() -> dict:
    return {
        "ic": 0.035,
        "icir": 0.80,
        "rank_ic": 0.041,
        "rank_icir": 0.76,
        "turnover": 0.32,
        "max_correlation": 0.44,
        "cost_adjusted_return": 0.052,
        "raw_valid_ic": 0.031,
        "raw_selection_ic": 0.035,
        "selection_days": 400,
        "coverage_pass_rate": 0.99,
        "mean_coverage_ratio": 0.95,
        "constant_day_rate": 0.0,
        "direction": "direct",
        "hac_p_value": 0.01,
        "bh_q_value": 0.02,
    }


def _sealed_factor_evaluation(metrics: dict) -> tuple[dict, dict]:
    policy = FactorGatePolicy()
    gate_status, gate_reasons = policy.evaluate(metrics)
    periods = {
        "train_start": date(2018, 1, 1),
        "train_end": date(2021, 12, 31),
        "valid_start": date(2022, 1, 1),
        "valid_end": date(2023, 12, 31),
        "test_start": date(2024, 1, 8),
        "test_end": date(2026, 7, 10),
    }
    candidate = {
        "id": "factor-1",
        "status": "promoted",
        "admission_path": "standalone",
        "code_sha256": "1" * 64,
        "values_sha256": "2" * 64,
    }
    recompute_evidence = {"sealed": True}
    policy_json = asdict(policy)
    evaluation = {
        "id": "3" * 32,
        "factor_candidate_id": candidate["id"],
        "dataset": "daily-v1",
        "dataset_identity_sha256": "a" * 64,
        "is_legacy": False,
        **periods,
        "metrics": metrics,
        "metrics_sha256": canonical_sha256(metrics),
        "gate_status": gate_status,
        "gate_reasons": gate_reasons,
        "evaluator_version": policy.version,
        "candidate_code_sha256": candidate["code_sha256"],
        "candidate_values_sha256": candidate["values_sha256"],
        "recomputed_values_sha256": candidate["values_sha256"],
        "submitted_values_sha256": "4" * 64,
        "recompute_evidence": recompute_evidence,
        "artifact_sha256": "5" * 64,
        "policy_json": policy_json,
        "policy_sha256": canonical_sha256(policy_json),
    }
    evidence = {
        "candidate_id": candidate["id"],
        "dataset": evaluation["dataset"],
        "dataset_identity_sha256": evaluation["dataset_identity_sha256"],
        "periods": {key: value.isoformat() for key, value in periods.items()},
        "gate_status": gate_status,
        "gate_reasons": gate_reasons,
        "evaluator_version": policy.version,
        "candidate_code_sha256": candidate["code_sha256"],
        "candidate_values_sha256": candidate["values_sha256"],
        "submitted_values_sha256": evaluation["submitted_values_sha256"],
        "recompute_evidence_sha256": canonical_sha256(recompute_evidence),
        "artifact_sha256": evaluation["artifact_sha256"],
        "metrics_sha256": evaluation["metrics_sha256"],
        "policy_sha256": evaluation["policy_sha256"],
    }
    evaluation["evidence_sha256"] = canonical_sha256(evidence)
    evaluation["execution_contract_hash"] = evaluation["evidence_sha256"]
    return candidate, evaluation


def _selection() -> tuple[dict, dict, dict]:
    identity = "a" * 64
    contract = build_factor_score_champion_contract(
        dataset="daily-v1",
        dataset_identity_sha256=identity,
        horizon_profile="short_1_5d",
        sota_version_id="sota-exact",
        sota_evidence_sha256="b" * 64,
        sota_policy_sha256="c" * 64,
        members=[_raw_factor_member()],
    )
    feature_set = factor_score_champion_feature_set(contract)
    signal_config = factor_score_champion_signal_config(contract)
    evidence = {
        "contract_version": "autopilot-factor-score-champion-selection-v1",
        "selection_policy_version": "exact-dataset-horizon-sota-v1",
        "dataset": "daily-v1",
        "dataset_identity_sha256": identity,
        "horizon_profile": "short_1_5d",
        "selected_kind": "factor",
        "selected_candidate_id": "sota-exact",
        "selected_strategy_config": signal_config,
        "selected_research_feature_set": feature_set,
        "factor_score_champion_contract_sha256": contract["contract_sha256"],
        "final_oos_opened": False,
        "research_screening_only": True,
        "not_capital_confirmation": True,
        "cross_cycle_fwer_claimed": False,
    }
    selection = {
        "champion_selection_evidence": evidence,
        "champion_selection_evidence_sha256": canonical_sha256(evidence),
    }
    return selection, contract, feature_set


def _candidate_config(contract: dict) -> dict:
    recipe = get_strategy_recipe("short_relative_strength")
    config = {
        **deepcopy(recipe["config_overrides"]),
        "recipe_id": recipe["id"],
        "recipe_version": recipe["version"],
        "horizon_profile": "short_1_5d",
        "source_research_artifact_id": "compiled-artifact-1",
        **factor_score_champion_signal_config(contract),
    }
    rule_ir = deepcopy(config["strategy_rule_ir"])
    slots = deepcopy(rule_ir["slots"])
    weights = {
        str(item["feature_id"]): float(item["weight"])
        for item in contract["members"]
    }
    slots["alpha_rank"]["components"][0]["parameters"]["weights"] = weights
    rules = validate_strategy_rule_ir(
        "short_1_5d", slots, allowed_factor_ids=set(weights)
    )
    policy = compile_strategy_rule_policy(
        "short_1_5d", rules, allowed_factor_ids=set(weights)
    )
    config["strategy_rule_ir"] = rules
    config["strategy_rules_sha256"] = rules["rules_sha256"]
    config["strategy_rule_policy_sha256"] = policy["policy_sha256"]
    for field, value in policy.items():
        if field in config and value is not None:
            config[field] = value
    return config


def test_factor_champion_freezes_effective_fin_strategy_score_grid() -> None:
    selection, contract, feature_set = _selection()
    public = {
        "id": "transparent-control",
        "features": {"public": "$close"},
        "definition_sha256": "9" * 64,
    }

    effective = research_feature_set_for_champion_selection(public, selection)
    assert effective == feature_set
    binding = build_strategy_research_signal_binding(
        horizon_profile="short_1_5d",
        dataset="daily-v1",
        dataset_identity_sha256="a" * 64,
        research_feature_set=effective,
        champion_selection=selection,
    )
    assert validate_strategy_research_signal_binding(binding) == binding
    assert binding["champion_kind"] == "factor"
    assert binding["signal_config"]["factor_score_champion_contract"] == contract

    tampered = deepcopy(selection)
    tampered["champion_selection_evidence"]["dataset_identity_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="selection evidence"):
        research_feature_set_for_champion_selection(public, tampered)


def test_factor_champion_remains_on_existing_competition_and_oos_path() -> None:
    _, contract, _ = _selection()
    config = _candidate_config(contract)
    bound = bind_factor_source_config(config, factor_count=0, creating_family=True)
    assert bound["factor_source_mode"] == "qlib_baseline"
    assert bound["factor_score_champion_contract_sha256"] == contract[
        "contract_sha256"
    ]

    policy_control = build_public_strategy_control_config(config)
    assert policy_control["factor_score_champion_contract_sha256"] == contract[
        "contract_sha256"
    ]
    assert set(
        policy_control["strategy_rule_ir"]["slots"]["alpha_rank"]["components"][0][
            "parameters"
        ]["weights"]
    ) == {contract["members"][0]["feature_id"]}

    full_stack_control = build_transparent_full_stack_control_config(config)
    assert "factor_score_champion_contract" not in full_stack_control
    assert "factor_score_champion_contract_sha256" not in full_stack_control
    assert full_stack_control["factor_source_mode"] == "qlib_baseline"
    assert set(
        full_stack_control["strategy_rule_ir"]["slots"]["alpha_rank"][
            "components"
        ][0]["parameters"]["weights"]
    ) == {
        "amount_expansion_5d",
        "close_location_5d",
        "extension_penalty_5d",
        "relative_strength_5d",
    }


def test_selector_uses_factor_only_when_existing_admitted_family_is_empty() -> None:
    selection, _, _ = _selection()

    class EmptyModelSelection:
        @staticmethod
        def select_champion(**_: object) -> dict:
            raise ValueError(
                "no independently admitted signal matches this dataset identity"
            )

    class FactorSelection:
        @staticmethod
        def select(**_: object) -> dict:
            return selection

    selector = object.__new__(AutopilotResearchChampionSelector)
    selector._selection = EmptyModelSelection()
    selector._factor_selection = FactorSelection()
    assert selector.select_champion(
        dataset="daily-v1",
        dataset_identity_sha256="a" * 64,
        horizon_profile="short_1_5d",
    ) == selection

    class ExistingModelSelection:
        @staticmethod
        def select_champion(**_: object) -> dict:
            return {"selected": "model"}

    selector._selection = ExistingModelSelection()
    assert selector.select_champion(
        dataset="daily-v1",
        dataset_identity_sha256="a" * 64,
        horizon_profile="short_1_5d",
    ) == {"selected": "model"}


def test_factor_sota_member_evaluation_revalidates_dataset_gate_and_seal() -> None:
    candidate, evaluation = _sealed_factor_evaluation(_passing_factor_metrics())
    _validate_factor_sota_member_evaluation(
        candidate=candidate,
        evaluation=evaluation,
        dataset="daily-v1",
        dataset_identity_sha256="a" * 64,
    )

    stale_dataset = deepcopy(evaluation)
    stale_dataset["dataset_identity_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="evaluation binding"):
        _validate_factor_sota_member_evaluation(
            candidate=candidate,
            evaluation=stale_dataset,
            dataset="daily-v1",
            dataset_identity_sha256="a" * 64,
        )

    stale_gate = deepcopy(evaluation)
    stale_gate["gate_status"] = "failed"
    with pytest.raises(ValueError, match="evaluation gate"):
        _validate_factor_sota_member_evaluation(
            candidate=candidate,
            evaluation=stale_gate,
            dataset="daily-v1",
            dataset_identity_sha256="a" * 64,
        )

    stale_seal = deepcopy(evaluation)
    stale_seal["evidence_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="evaluation evidence"):
        _validate_factor_sota_member_evaluation(
            candidate=candidate,
            evaluation=stale_seal,
            dataset="daily-v1",
            dataset_identity_sha256="a" * 64,
        )


def test_factor_sota_member_admission_archives_effect_without_vetoing() -> None:
    metrics = _passing_factor_metrics()
    metrics["ic"] = 0.0
    candidate, evaluation = _sealed_factor_evaluation(metrics)
    candidate["admission_path"] = "incremental"
    _validate_factor_sota_member_evaluation(
        candidate=candidate,
        evaluation=evaluation,
        dataset="daily-v1",
        dataset_identity_sha256="a" * 64,
    )

    # 宽进严出:效应/显著性只入档,standalone 准入只看硬门(完整性/覆盖率/冗余)。
    candidate["admission_path"] = "standalone"
    _validate_factor_sota_member_evaluation(
        candidate=candidate,
        evaluation=evaluation,
        dataset="daily-v1",
        dataset_identity_sha256="a" * 64,
    )


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf")])
def test_factor_score_contract_builder_rejects_nonfinite_weights(weight: float) -> None:
    with pytest.raises(ValueError, match="weights"):
        build_factor_score_champion_contract(
            dataset="daily-v1",
            dataset_identity_sha256="a" * 64,
            horizon_profile="short_1_5d",
            sota_version_id="sota-exact",
            sota_evidence_sha256="b" * 64,
            sota_policy_sha256="c" * 64,
            members=[_raw_factor_member(weight=weight)],
        )


@pytest.mark.parametrize("weight", ["nan", "inf", "-inf"])
def test_factor_score_contract_validator_rejects_nonfinite_weights(weight: str) -> None:
    _, contract, _ = _selection()
    tampered = deepcopy(contract)
    tampered["members"][0]["weight"] = weight
    tampered.pop("contract_sha256")
    tampered["contract_sha256"] = canonical_sha256(tampered)
    with pytest.raises(ValueError, match="weights"):
        validate_factor_score_champion_contract(tampered)
