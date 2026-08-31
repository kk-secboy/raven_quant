from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from quant_platform.research_horizon import research_horizon_contract
from quant_platform.strategy_proposal import validate_strategy_proposal
from quant_platform.strategy_research_signal_binding import (
    signal_config_from_strategy_research_binding,
)
from quant_platform.strategy_rule_ir import (
    STRATEGY_RULE_IR_VERSION,
    canonical_sha256,
    validate_strategy_rule_ir,
)

STRATEGY_COMPILER_VERSION = "strategy-rule-compiler-v1"

_POLICY_CONFIG_FIELDS = (
    "topk",
    "max_position_weight",
    "max_industry_weight",
    "max_daily_turnover",
    "portfolio_construction",
    "min_average_daily_amount",
    "liquidity_lookback_days",
    "min_listing_days",
    "entry_score_min_percentile",
    "score_drop_exit_percentile",
    "score_deterioration_reduce_percentile",
    "score_deterioration_reduce_fraction",
    "extension_guard_max_return_5d",
    "holding_min_sessions",
    "max_holding_sessions",
    "min_rebalance_weight_change",
    "market_trend_lookback_sessions",
    "market_trend_benchmark",
    "valuation_regime_max_percentile",
    "valuation_reduce_percentile",
    "valuation_reduce_fraction",
    "trend_break_lookback_sessions",
    "thesis_min_holding_sessions",
    "thesis_review_frequency",
    "thesis_break_score_percentile",
    "hard_risk_target_fraction",
    "stop_loss",
    "rebalance_frequency",
    "lot_size",
    "max_volume_participation",
    "execution_method",
    "cash_when_no_edge",
)


def _component_parameters(
    slots: Mapping[str, Any], slot: str, component: str
) -> dict[str, Any] | None:
    slot_value = slots.get(slot)
    if not isinstance(slot_value, Mapping):
        return None
    for item in slot_value.get("components") or []:
        if isinstance(item, Mapping) and item.get("component") == component:
            parameters = item.get("parameters")
            return dict(parameters) if isinstance(parameters, Mapping) else {}
    return None


def compile_strategy_rule_policy(
    horizon: str,
    rule_ir: Mapping[str, Any],
    *,
    allowed_factor_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Compile the allowlisted rule IR into the shared portfolio-policy contract.

    The result contains only deterministic data consumed by both the formal Qlib
    backtest and the daily recommendation refresh.  Keeping this projection in
    one function prevents an accepted proposal from becoming a decorative JSON
    artifact whose entry/exit rules are silently ignored at execution time.
    """

    slots_value = rule_ir.get("slots")
    rules = validate_strategy_rule_ir(
        horizon,
        slots_value if isinstance(slots_value, Mapping) else None,
        allowed_factor_ids=allowed_factor_ids,
    )
    slots = rules["slots"]

    alpha = _component_parameters(slots, "alpha_rank", "weighted_factor_rank")
    score_threshold = _component_parameters(slots, "entry_timing", "score_threshold")
    extension_guard = _component_parameters(slots, "entry_timing", "extension_guard")
    rebalance = _component_parameters(slots, "entry_timing", "rebalance_calendar")
    max_holding = _component_parameters(slots, "exit_state", "max_holding_days")
    score_drop = _component_parameters(slots, "exit_state", "score_drop_exit")
    score_reduce = _component_parameters(
        slots, "exit_state", "score_deterioration_reduce"
    )
    valuation_reduce = _component_parameters(slots, "exit_state", "valuation_reduce")
    stop_loss = _component_parameters(slots, "exit_state", "stop_loss")
    trend_break = _component_parameters(slots, "exit_state", "trend_break")
    thesis_break = _component_parameters(slots, "exit_state", "thesis_break")
    topk = _component_parameters(slots, "portfolio_risk", "topk_equal_weight")
    industry = _component_parameters(slots, "portfolio_risk", "max_industry_weight")
    turnover = _component_parameters(slots, "portfolio_risk", "max_daily_turnover")
    minimum_trade_band = _component_parameters(
        slots, "portfolio_risk", "minimum_trade_band"
    )
    liquidity = _component_parameters(slots, "eligibility_gate", "liquidity_floor")
    listing = _component_parameters(slots, "eligibility_gate", "tradable_ashare")
    market_trend = _component_parameters(
        slots, "direction_regime_gate", "market_trend_filter"
    )
    valuation = _component_parameters(
        slots, "direction_regime_gate", "valuation_regime_filter"
    )
    board_lot = _component_parameters(slots, "execution_requirement", "board_lot")
    participation = _component_parameters(
        slots, "execution_requirement", "liquidity_participation"
    )

    if alpha is None or score_threshold is None or rebalance is None or topk is None:
        raise ValueError("strategy rule IR cannot be projected into the portfolio policy")
    if industry is None or turnover is None or liquidity is None or listing is None:
        raise ValueError("strategy rule IR is missing governed portfolio or eligibility limits")
    if board_lot is None or participation is None:
        raise ValueError("strategy rule IR is missing governed execution limits")

    horizon_contract = research_horizon_contract(horizon)
    policy = {
        "contract_version": "strategy-rule-policy-v1",
        "horizon_profile": horizon,
        "strategy_rules_sha256": rules["rules_sha256"],
        "alpha_factor_weights": dict(alpha["weights"]),
        "topk": int(topk["topk"]),
        "max_position_weight": float(topk["max_position_weight"]),
        "max_industry_weight": float(industry["fraction"]),
        "max_daily_turnover": float(turnover["fraction"]),
        "portfolio_construction": "topk_equal_weight",
        "min_average_daily_amount": float(liquidity["min_average_daily_amount"]),
        "liquidity_lookback_days": int(liquidity["lookback_days"]),
        "min_listing_days": int(listing["min_listing_days"]),
        "entry_score_min_percentile": float(score_threshold["minimum_percentile"]),
        "score_drop_exit_percentile": (
            float(score_drop["below_percentile"]) if score_drop is not None else None
        ),
        "score_deterioration_reduce_percentile": (
            float(score_reduce["below_percentile"])
            if score_reduce is not None
            else None
        ),
        "score_deterioration_reduce_fraction": (
            float(score_reduce["reduce_fraction"])
            if score_reduce is not None
            else 0.50
        ),
        "extension_guard_max_return_5d": (
            float(extension_guard["max_return_5d"])
            if extension_guard is not None
            else None
        ),
        "max_holding_sessions": (
            int(max_holding["days"]) if max_holding is not None else None
        ),
        "holding_min_sessions": horizon_contract.holding_min_sessions,
        "min_rebalance_weight_change": (
            float(minimum_trade_band["fraction"])
            if minimum_trade_band is not None
            else 0.0
        ),
        "market_trend_lookback_sessions": (
            int(market_trend["lookback_days"]) if market_trend is not None else None
        ),
        "market_trend_benchmark": (
            str(market_trend["benchmark"]) if market_trend is not None else None
        ),
        "valuation_regime_max_percentile": (
            float(valuation["max_percentile"]) if valuation is not None else None
        ),
        "valuation_reduce_percentile": (
            float(valuation_reduce["above_percentile"])
            if valuation_reduce is not None
            else None
        ),
        "valuation_reduce_fraction": (
            float(valuation_reduce["reduce_fraction"])
            if valuation_reduce is not None
            else 0.50
        ),
        "trend_break_lookback_sessions": (
            int(trend_break["lookback_days"]) if trend_break is not None else None
        ),
        "thesis_min_holding_sessions": (
            int(thesis_break["minimum_holding_days"])
            if thesis_break is not None
            else None
        ),
        "thesis_review_frequency": (
            str(thesis_break["review_frequency"]) if thesis_break is not None else None
        ),
        # The quality/value strategy uses a deterministic below-median composite
        # score as the quantifiable thesis-break proxy after its minimum holding
        # period.  This is deliberately described as a proxy, not a moat verdict.
        "thesis_break_score_percentile": 0.50 if thesis_break is not None else None,
        "stop_loss": float(stop_loss["fraction"]) if stop_loss is not None else None,
        "hard_risk_target_fraction": 0.50,
        "rebalance_frequency": str(rebalance["frequency"]),
        "lot_size": int(board_lot["shares"]),
        "max_volume_participation": float(participation["max_fraction"]),
        "execution_method": "open",
        "execution_lag_sessions": 1,
        "cash_when_no_edge": (
            _component_parameters(slots, "portfolio_risk", "cash_when_no_edge")
            is not None
        ),
    }
    policy["policy_sha256"] = canonical_sha256(policy)
    return policy


def validate_strategy_rule_binding(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Recompute and validate the rule identity embedded in a StrategySpec.

    Legacy strategies intentionally have no declared long/medium/short rule
    contract.  Every explicit product horizon must bind the exact allowlisted
    IR and its deterministic policy projection; otherwise database columns and
    UI labels could claim one policy while the runner executes another.
    """

    horizon = str(config.get("horizon_profile") or "legacy_ambiguous")
    if horizon == "legacy_ambiguous":
        return None
    rule_ir = config.get("strategy_rule_ir")
    if not isinstance(rule_ir, Mapping):
        raise ValueError("explicit horizon strategies require strategy_rule_ir")
    alpha = _component_parameters(
        rule_ir.get("slots") if isinstance(rule_ir.get("slots"), Mapping) else {},
        "alpha_rank",
        "weighted_factor_rank",
    )
    allowed_factor_ids = (
        {str(item) for item in (alpha or {}).get("weights", {})}
        if isinstance((alpha or {}).get("weights"), Mapping)
        else None
    )
    policy = compile_strategy_rule_policy(
        horizon,
        rule_ir,
        allowed_factor_ids=allowed_factor_ids,
    )
    if rule_ir.get("rules_sha256") != policy["strategy_rules_sha256"]:
        raise ValueError("strategy_rule_ir carries a stale rules digest")
    if config.get("strategy_rules_sha256") != policy["strategy_rules_sha256"]:
        raise ValueError("strategy_rules_sha256 does not match strategy_rule_ir")
    if config.get("strategy_rule_policy_sha256") != policy["policy_sha256"]:
        raise ValueError("strategy_rule_policy_sha256 does not match compiled policy")

    recipe_id = str(config.get("recipe_id") or "custom")
    if recipe_id != "custom":
        from quant_platform.strategy_recipes import get_strategy_recipe

        try:
            recipe = get_strategy_recipe(recipe_id, include_execution_policy=False)
        except KeyError as exc:
            raise ValueError("strategy recipe has no governed rule baseline") from exc
        baseline_rules = recipe.get("strategy_rule_ir")
        if recipe.get("research_baseline") is not True or not isinstance(
            baseline_rules, Mapping
        ):
            raise ValueError("explicit horizon recipe has no governed rule baseline")
        if (
            config.get("source_research_artifact_id") is None
            and dict(rule_ir) != dict(baseline_rules)
        ):
            raise ValueError("public baseline strategy rules differ from the recipe")

    for field in _POLICY_CONFIG_FIELDS:
        expected = policy.get(field)
        if expected is None:
            continue
        actual = config.get(field)
        if isinstance(expected, float):
            try:
                matches = abs(float(actual) - expected) <= 1e-12
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f"strategy config field {field} differs from compiled rules")
    if int(config.get("execution_lag_bars") or 0) != int(
        policy["execution_lag_sessions"]
    ):
        raise ValueError("strategy execution lag differs from compiled rules")
    return policy


def compile_strategy_proposal(
    proposal: Mapping[str, Any],
    *,
    allowed_factor_ids: set[str] | None = None,
) -> dict[str, Any]:
    proposal_input = dict(proposal)
    supplied_proposal_sha256 = proposal_input.pop("proposal_sha256", None)
    normalized = validate_strategy_proposal(proposal_input)
    proposal_sha256 = normalized.pop("proposal_sha256")
    if supplied_proposal_sha256 not in {None, proposal_sha256}:
        raise ValueError("strategy proposal digest disagrees")
    rules = validate_strategy_rule_ir(
        str(normalized["horizon"]),
        normalized["slots"],
        allowed_factor_ids=allowed_factor_ids,
    )
    execution_policy = compile_strategy_rule_policy(
        str(normalized["horizon"]),
        rules,
        allowed_factor_ids=allowed_factor_ids,
    )
    # Resolve the existing authoritative recipe instead of inventing a second
    # baseline registry for strategy research.
    from quant_platform.strategy_recipes import get_strategy_recipe

    try:
        baseline = get_strategy_recipe(
            str(normalized["baseline_recipe_id"]), include_execution_policy=False
        )
    except KeyError as exc:
        raise ValueError("strategy proposal baseline recipe is unknown") from exc
    baseline_rules = baseline.get("strategy_rule_ir")
    if (
        baseline.get("research_baseline") is not True
        or baseline.get("horizon") != normalized["horizon"]
        or baseline.get("version") != normalized["baseline_recipe_version"]
        or not isinstance(baseline_rules, Mapping)
        or baseline_rules.get("rules_sha256") != normalized["baseline_rules_sha256"]
    ):
        raise ValueError("strategy proposal baseline binding disagrees")
    baseline_slots = baseline_rules.get("slots")
    if not isinstance(baseline_slots, Mapping):
        raise ValueError("strategy proposal baseline rules are incomplete")
    actual_changed_slots = [
        slot
        for slot in rules["control_order"]
        if rules["slots"][slot] != baseline_slots.get(slot)
    ]
    if actual_changed_slots != normalized["changed_slots"]:
        raise ValueError("strategy proposal changed_slots disagrees with the baseline diff")
    compiled_spec = {
        "contract_version": "strategy-spec-candidate-v1",
        "delivery_status": "research_only",
        "capital_eligible": False,
        "simulation_eligible": False,
        "name": normalized["name"],
        "description": normalized["description"],
        "horizon": normalized["horizon"],
        "horizon_contract": rules["horizon_contract"],
        "economic_hypothesis": normalized["economic_hypothesis"],
        "baseline_recipe_id": normalized["baseline_recipe_id"],
        "baseline_recipe_version": normalized["baseline_recipe_version"],
        "baseline_rules_sha256": normalized["baseline_rules_sha256"],
        "parent_strategy_version_id": normalized["parent_strategy_version_id"],
        "changed_slots": normalized["changed_slots"],
        "data_contract": deepcopy(normalized["data_contract"]),
        "evaluation_contract": deepcopy(normalized["evaluation_contract"]),
        "rule_ir": rules,
        "execution_policy": execution_policy,
        "required_next_gate": "formal_rolling_oos_backtest",
    }
    compiled_spec["strategy_spec_sha256"] = canonical_sha256(compiled_spec)
    artifact = {
        "contract_version": "compiled-strategy-proposal-v1",
        "compiler_version": STRATEGY_COMPILER_VERSION,
        "rule_ir_version": STRATEGY_RULE_IR_VERSION,
        "delivery_status": "research_only",
        "proposal_sha256": proposal_sha256,
        "strategy_proposal": deepcopy(normalized),
        "rules_sha256": rules["rules_sha256"],
        "strategy_spec_candidate": compiled_spec,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def validate_compiled_strategy_artifact(
    artifact: Any,
    *,
    allowed_factor_ids: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise ValueError("compiled strategy artifact must be an object")
    expected_fields = {
        "contract_version",
        "compiler_version",
        "rule_ir_version",
        "delivery_status",
        "proposal_sha256",
        "strategy_proposal",
        "rules_sha256",
        "strategy_spec_candidate",
        "artifact_sha256",
    }
    if set(artifact) != expected_fields:
        raise ValueError("compiled strategy artifact contract drifted")
    if (
        artifact["contract_version"] != "compiled-strategy-proposal-v1"
        or artifact["compiler_version"] != STRATEGY_COMPILER_VERSION
        or artifact["rule_ir_version"] != STRATEGY_RULE_IR_VERSION
        or artifact["delivery_status"] != "research_only"
    ):
        raise ValueError("compiled strategy artifact identity is unsupported")
    proposal = validate_strategy_proposal(artifact["strategy_proposal"])
    if proposal.pop("proposal_sha256") != artifact["proposal_sha256"]:
        raise ValueError("compiled strategy proposal digest disagrees")
    candidate = artifact["strategy_spec_candidate"]
    if not isinstance(candidate, Mapping):
        raise ValueError("compiled strategy candidate must be an object")
    rule_ir = candidate.get("rule_ir")
    rules = validate_strategy_rule_ir(
        str(candidate.get("horizon") or ""),
        rule_ir.get("slots") if isinstance(rule_ir, Mapping) else None,
        allowed_factor_ids=allowed_factor_ids,
    )
    if rules["rules_sha256"] != artifact["rules_sha256"]:
        raise ValueError("compiled strategy rules digest disagrees")
    candidate_without_digest = dict(candidate)
    candidate_digest = candidate_without_digest.pop("strategy_spec_sha256", None)
    if candidate_digest != canonical_sha256(candidate_without_digest):
        raise ValueError("compiled strategy spec digest disagrees")
    artifact_without_digest = dict(artifact)
    artifact_digest = artifact_without_digest.pop("artifact_sha256", None)
    if artifact_digest != canonical_sha256(artifact_without_digest):
        raise ValueError("compiled strategy artifact digest disagrees")
    expected = compile_strategy_proposal(
        artifact["strategy_proposal"],
        allowed_factor_ids=allowed_factor_ids,
    )
    if dict(artifact) != expected:
        raise ValueError("compiled strategy artifact is not deterministic")
    return expected


def materialize_strategy_candidate_config(
    artifact: Mapping[str, Any],
    *,
    source_research_artifact_id: str,
    allowed_factor_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Build the inert StrategyVersion config for an archived proposal.

    This does not approve, backtest, simulate, or recommend.  It only projects
    a verified compiled artifact into the existing StrategySpec schema so the
    normal independent Qlib gates can evaluate it.
    """

    normalized = validate_compiled_strategy_artifact(
        artifact,
        allowed_factor_ids=allowed_factor_ids,
    )
    proposal = normalized["strategy_proposal"]
    candidate = normalized["strategy_spec_candidate"]
    from quant_platform.cost_model import CostModelConfig
    from quant_platform.research_horizon import research_horizon_contract
    from quant_platform.strategy_recipes import get_strategy_recipe

    baseline = get_strategy_recipe(str(proposal["baseline_recipe_id"]))
    if baseline["version"] != proposal["baseline_recipe_version"]:
        raise ValueError("strategy proposal baseline recipe version changed")
    policy = dict(candidate["execution_policy"])
    config = dict(baseline["config_overrides"])
    config.update(
        {
            key: value
            for key, value in policy.items()
            if key in _POLICY_CONFIG_FIELDS and value is not None
        }
    )
    cost = CostModelConfig().to_dict()
    cost["cost_schedule_version"] = cost.pop("version")
    config.update(cost)
    horizon = research_horizon_contract(str(candidate["horizon"]))
    research_signal_binding = proposal["data_contract"].get(
        "research_signal_binding"
    )
    if research_signal_binding is not None:
        config.update(
            signal_config_from_strategy_research_binding(
                research_signal_binding
            )
        )
    config.update(
        {
            "recipe_id": proposal["baseline_recipe_id"],
            "recipe_version": proposal["baseline_recipe_version"],
            "horizon_profile": candidate["horizon"],
            "outer_purge_days": int(horizon.purge_sessions or 0),
            "outer_embargo_days": int(horizon.embargo_sessions or 0),
            "min_backtest_days": int(
                proposal["evaluation_contract"]["minimum_oos_observations"]
            ),
            "min_pre_final_history_days": 2520,
            "rolling_window_days": 252,
            "rolling_step_days": 63,
            "min_rolling_windows": int(
                proposal["evaluation_contract"]["rolling_folds"]
            ),
            "min_rolling_pass_rate": 0.60,
            "minimum_outer_test_pass_rate": 0.60,
            "min_robustness_pass_rate": 1.0,
            "capacity_curve_notionals": [5_000_000, 20_000_000, 100_000_000],
            "source_research_artifact_id": str(source_research_artifact_id),
            "strategy_research_proposal_sha256": normalized["proposal_sha256"],
            "strategy_research_artifact_sha256": normalized["artifact_sha256"],
            "parent_strategy_version_id": proposal["parent_strategy_version_id"],
            "strategy_research_data_contract": deepcopy(proposal["data_contract"]),
            **(
                {
                    "strategy_research_signal_binding": deepcopy(
                        research_signal_binding
                    )
                }
                if research_signal_binding is not None
                else {}
            ),
            "strategy_evaluation_contract": deepcopy(
                proposal["evaluation_contract"]
            ),
            "strategy_rule_ir": deepcopy(candidate["rule_ir"]),
            "strategy_rules_sha256": normalized["rules_sha256"],
            "strategy_rule_policy_sha256": policy["policy_sha256"],
        }
    )
    return config
