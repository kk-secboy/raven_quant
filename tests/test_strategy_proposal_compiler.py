from __future__ import annotations

from copy import deepcopy

import pytest

from quant_platform.cost_model import COST_SCHEDULE_VERSION
from quant_platform.strategy_proposal import (
    STRATEGY_PROPOSAL_VERSION,
    parse_strategy_proposal_json,
    validate_strategy_proposal,
)
from quant_platform.strategy_recipes import get_strategy_recipe
from quant_platform.strategy_rule_compiler import (
    compile_strategy_proposal,
    materialize_strategy_candidate_config,
    validate_compiled_strategy_artifact,
)

pytestmark = pytest.mark.no_database


def _proposal(recipe_id: str) -> dict:
    recipe = get_strategy_recipe(recipe_id)
    horizon = recipe["horizon"]
    slots = deepcopy(recipe["strategy_rule_ir"]["slots"])
    weights = slots["alpha_rank"]["components"][0]["parameters"]["weights"]
    first, second = list(weights)[:2]
    weights[first] += 0.05
    weights[second] -= 0.05
    return {
        "contract_version": STRATEGY_PROPOSAL_VERSION,
        "delivery_status": "research_only",
        "name": f"{recipe_id} challenger",
        "description": "A falsifiable research-only structured-rule challenger.",
        "horizon": horizon,
        "economic_hypothesis": "The changed ranking may improve after-cost robustness.",
        "baseline_recipe_id": recipe_id,
        "baseline_recipe_version": recipe["version"],
        "baseline_rules_sha256": recipe["strategy_rule_ir"]["rules_sha256"],
        "parent_strategy_version_id": None,
        "changed_slots": ["alpha_rank"],
        "data_contract": {
            "dataset_snapshot_id": "b" * 64,
            "feature_set_id": "governed-baseline",
            "feature_set_definition_sha256": "a" * 64,
            "research_periods": {
                "train_start": "2008-01-02",
                "train_end": "2018-12-28",
                "valid_start": "2019-01-02",
                "valid_end": "2021-12-31",
                "test_start": "2022-01-04",
                "test_end": "2024-12-31",
            },
            "decision_frequency": {
                "short_1_5d": "day",
                "swing_1_6m": "week",
                "long_1_3y": "month",
            }[horizon],
            "label_horizon_trading_days": {
                "short_1_5d": 5,
                "swing_1_6m": 63,
                "long_1_3y": 252,
            }[horizon],
        },
        "evaluation_contract": {
            "benchmark": "SH000300",
            "primary_metric": "after_cost_information_ratio",
            "cost_schedule_version": COST_SCHEDULE_VERSION,
            "rolling_folds": 5,
            "minimum_oos_observations": {
                "short_1_5d": 252,
                "swing_1_6m": 504,
                "long_1_3y": 756,
            }[horizon],
            "final_oos_visible_during_selection": False,
        },
        "slots": slots,
    }


def _factor_ids(proposal: dict) -> set[str]:
    alpha = proposal["slots"]["alpha_rank"]["components"][0]
    return set(alpha["parameters"]["weights"])


def test_strategy_proposal_compiles_deterministically_but_remains_research_only() -> None:
    proposal = _proposal("short_relative_strength")
    allowed = _factor_ids(proposal)

    first = compile_strategy_proposal(proposal, allowed_factor_ids=allowed)
    second = compile_strategy_proposal(deepcopy(proposal), allowed_factor_ids=allowed)

    assert first == second
    assert first["delivery_status"] == "research_only"
    assert first["strategy_spec_candidate"]["capital_eligible"] is False
    assert first["strategy_spec_candidate"]["simulation_eligible"] is False
    assert first["strategy_spec_candidate"]["required_next_gate"] == (
        "formal_rolling_oos_backtest"
    )
    execution = first["strategy_spec_candidate"]["execution_policy"]
    assert execution["contract_version"] == "strategy-rule-policy-v1"
    assert execution["max_holding_sessions"] == 5
    assert execution["entry_score_min_percentile"] == pytest.approx(0.80)
    assert execution["strategy_rules_sha256"] == first["rules_sha256"]
    assert validate_compiled_strategy_artifact(first, allowed_factor_ids=allowed) == first

    reordered = deepcopy(proposal)
    reordered["slots"] = dict(reversed(list(reordered["slots"].items())))
    assert compile_strategy_proposal(reordered, allowed_factor_ids=allowed) == first


def test_compiled_proposal_materializes_only_an_inert_rule_bound_candidate() -> None:
    from quant_platform.api import StrategyConfigRequest
    from quant_platform.strategy_store import _normalize_multifactor_contract
    from quant_platform.transparent_baseline_runner import (
        STRATEGY_RESEARCH_TARGET_RUNNER_SHA256,
        STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256,
    )

    proposal = _proposal("swing_trend")
    allowed = _factor_ids(proposal)
    artifact = compile_strategy_proposal(proposal, allowed_factor_ids=allowed)
    config = materialize_strategy_candidate_config(
        artifact,
        source_research_artifact_id="artifact-1",
        allowed_factor_ids=allowed,
    )
    validated = StrategyConfigRequest.model_validate(config)

    assert validated.horizon_profile == "swing_1_6m"
    assert validated.source_research_artifact_id == "artifact-1"
    assert validated.strategy_rules_sha256 == artifact["rules_sha256"]
    assert validated.strategy_rule_policy_sha256 == artifact[
        "strategy_spec_candidate"
    ]["execution_policy"]["policy_sha256"]
    assert validated.max_holding_sessions == 126
    assert validated.trend_break_lookback_sessions == 20
    assert len(validated.execution_contract_hash or "") == 64

    normalized = _normalize_multifactor_contract(
        config,
        factor_count=0,
        creating_family=True,
    )
    runtime = normalized["transparent_baseline_bootstrap"]
    assert runtime["target_runner_sha256"] == (
        STRATEGY_RESEARCH_TARGET_RUNNER_SHA256
    )
    assert runtime["target_runtime_bundle_sha256"] == (
        STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256
    )
    assert runtime["target_worker_runtime_image_digest"] == "sha256:" + "d" * 64


def test_strategy_proposal_rejects_code_unknown_parameters_and_final_oos_visibility() -> None:
    proposal = _proposal("swing_trend")
    proposal["python_code"] = "place_order()"
    with pytest.raises(ValueError, match="top-level contract drifted"):
        validate_strategy_proposal(proposal)

    proposal = _proposal("swing_trend")
    proposal["slots"]["entry_timing"]["components"][0]["parameters"]["python_code"] = (
        "place_order()"
    )
    with pytest.raises(ValueError, match="parameter contract drifted"):
        compile_strategy_proposal(proposal, allowed_factor_ids=_factor_ids(proposal))

    proposal = _proposal("swing_trend")
    proposal["evaluation_contract"]["final_oos_visible_during_selection"] = True
    with pytest.raises(ValueError, match="cannot expose final OOS"):
        validate_strategy_proposal(proposal)

    proposal = _proposal("swing_trend")
    proposal["evaluation_contract"]["cost_schedule_version"] = "latest"
    with pytest.raises(ValueError, match="cost schedule is unsupported"):
        validate_strategy_proposal(proposal)

    proposal = _proposal("swing_trend")
    proposal["data_contract"]["research_periods"]["test_start"] = "2020-01-02"
    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        validate_strategy_proposal(proposal)


def test_strategy_proposal_json_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        parse_strategy_proposal_json('{"contract_version":"a","contract_version":"b"}')


def test_horizon_specific_exit_contracts_fail_closed() -> None:
    short = _proposal("short_relative_strength")
    short["slots"]["exit_state"]["components"][0]["parameters"]["days"] = 6
    with pytest.raises(ValueError, match="no greater than 5"):
        compile_strategy_proposal(short, allowed_factor_ids=_factor_ids(short))

    long = _proposal("long_quality_value")
    long["slots"]["exit_state"]["components"] = []
    with pytest.raises(ValueError, match="required strategy slot"):
        compile_strategy_proposal(long, allowed_factor_ids=_factor_ids(long))

    swing = _proposal("swing_trend")
    swing["slots"]["eligibility_gate"]["components"][0]["parameters"][
        "min_listing_days"
    ] = 60
    with pytest.raises(ValueError, match="requires at least 252 listing days"):
        compile_strategy_proposal(swing, allowed_factor_ids=_factor_ids(swing))

    long = _proposal("long_quality_value")
    for component in long["slots"]["entry_timing"]["components"]:
        if component["component"] == "rebalance_calendar":
            component["parameters"]["frequency"] = "quarter"
    with pytest.raises(ValueError, match="review/rebalance frequency"):
        compile_strategy_proposal(long, allowed_factor_ids=_factor_ids(long))

    long = _proposal("long_quality_value")
    thesis = next(
        item
        for item in long["slots"]["exit_state"]["components"]
        if item["component"] == "thesis_break"
    )
    thesis["parameters"]["review_frequency"] = "quarter"
    with pytest.raises(ValueError, match="monthly thesis review"):
        compile_strategy_proposal(long, allowed_factor_ids=_factor_ids(long))


def test_compiled_strategy_artifact_detects_tampering() -> None:
    proposal = _proposal("long_quality_value")
    allowed = _factor_ids(proposal)
    artifact = compile_strategy_proposal(proposal, allowed_factor_ids=allowed)
    artifact["strategy_spec_candidate"]["name"] = "tampered"

    with pytest.raises(ValueError, match="spec digest disagrees"):
        validate_compiled_strategy_artifact(artifact, allowed_factor_ids=allowed)

    artifact = compile_strategy_proposal(proposal, allowed_factor_ids=allowed)
    artifact["strategy_proposal"]["description"] = "tampered proposal"
    with pytest.raises(ValueError, match="proposal digest disagrees"):
        validate_compiled_strategy_artifact(artifact, allowed_factor_ids=allowed)


def test_strategy_proposal_rejects_false_baseline_diff_and_missing_a_share_guard() -> None:
    proposal = _proposal("short_relative_strength")
    proposal["slots"] = deepcopy(
        get_strategy_recipe("short_relative_strength")["strategy_rule_ir"]["slots"]
    )
    with pytest.raises(ValueError, match="changed_slots disagrees"):
        compile_strategy_proposal(proposal, allowed_factor_ids=_factor_ids(proposal))

    proposal = _proposal("swing_trend")
    proposal["slots"]["execution_requirement"]["components"] = [
        item
        for item in proposal["slots"]["execution_requirement"]["components"]
        if item["component"] != "a_share_t_plus_one"
    ]
    with pytest.raises(ValueError, match="omits mandatory components"):
        compile_strategy_proposal(proposal, allowed_factor_ids=_factor_ids(proposal))

    proposal = _proposal("short_relative_strength")
    with pytest.raises(ValueError, match="requires a governed factor allowlist"):
        compile_strategy_proposal(proposal)

    proposal = _proposal("short_relative_strength")
    proposal["slots"]["eligibility_gate"]["empty_behavior"] = "pass_through"
    with pytest.raises(ValueError, match="not fail-closed"):
        compile_strategy_proposal(proposal, allowed_factor_ids=_factor_ids(proposal))

    proposal = _proposal("short_relative_strength")
    proposal["baseline_rules_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="baseline binding disagrees"):
        compile_strategy_proposal(proposal, allowed_factor_ids=_factor_ids(proposal))
