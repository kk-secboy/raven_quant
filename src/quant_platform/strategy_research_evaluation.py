from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_data.execution_contract import strategy_execution_contract_hash

from .cost_model import CN_COST_SCHEDULE_BOOK
from .statistical_validation import (
    paired_moving_block_bootstrap,
    probability_of_backtest_overfitting,
)
from .strategy_rule_compiler import (
    compile_strategy_rule_policy,
    validate_strategy_rule_binding,
)
from .strategy_rule_ir import canonical_sha256, validate_strategy_rule_ir

STRATEGY_POLICY_ONLY_MODE = "strategy_policy_only_pre_final"
STRATEGY_FULL_STACK_MODE = "strategy_full_stack_pre_final"
STRATEGY_RESEARCH_EVALUATION_MODES = frozenset(
    {STRATEGY_POLICY_ONLY_MODE, STRATEGY_FULL_STACK_MODE}
)
STRATEGY_RESEARCH_COMPETITION_VERSION = "fin-strategy-fair-competition-v1"

_SHA256_FIELDS = (
    "baseline_definition_sha256",
    "feature_set_definition_sha256",
    "factor_score_champion_contract_sha256",
    "model_code_sha256",
    "model_recipe_sha256",
    "model_evidence_sha256",
    "quant_bundle_sha256",
    "quant_bundle_factor_contract_sha256",
)
_SCORE_ID_FIELDS = (
    "signal_source",
    "factor_source_mode",
    "challenger_weight",
    "feature_set_id",
    "model_candidate_id",
    "model_evaluation_id",
    "model_ensemble_candidate_id",
    "model_ensemble_evaluation_id",
    "quant_bundle_candidate_id",
    "quant_bundle_evaluation_id",
    *_SHA256_FIELDS,
)
_COST_FIELDS = (
    "cost_schedule_version",
    "buy_commission_rate",
    "sell_commission_rate",
    "stock_sell_stamp_duty_rate",
    "etf_sell_stamp_duty_rate",
    "transfer_fee_rate",
    "annual_borrow_rate",
    "fixed_slippage_rate",
    "impact_at_max_participation",
    "min_commission",
)
_POLICY_FIELDS = (
    "topk",
    "max_position_weight",
    "max_industry_weight",
    "max_daily_turnover",
    "min_average_daily_amount",
    "liquidity_lookback_days",
    "min_listing_days",
    "entry_score_min_percentile",
    "score_drop_exit_percentile",
    "extension_guard_max_return_5d",
    "max_holding_sessions",
    "market_trend_lookback_sessions",
    "market_trend_benchmark",
    "valuation_regime_max_percentile",
    "trend_break_lookback_sessions",
    "thesis_min_holding_sessions",
    "thesis_review_frequency",
    "thesis_break_score_percentile",
    "stop_loss",
    "rebalance_frequency",
    "lot_size",
    "max_volume_participation",
    "execution_method",
    "cash_when_no_edge",
)


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strategy_score_grid_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Identify everything allowed to influence the cross-sectional score grid."""

    policy = validate_strategy_rule_binding(config)
    alpha_weights = (
        dict(policy["alpha_factor_weights"])
        if isinstance(policy, Mapping)
        else None
    )
    contract = {
        "contract_version": "strategy-score-grid-v1",
        "horizon_profile": config.get("horizon_profile"),
        "alpha_factor_weights": alpha_weights,
        "score_inputs": {field: config.get(field) for field in _SCORE_ID_FIELDS},
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def build_public_strategy_control_config(
    candidate_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Recreate the public recipe control under the candidate's frozen costs."""

    from .strategy_recipes import get_strategy_recipe

    recipe = get_strategy_recipe(str(candidate_config.get("recipe_id") or ""))
    if recipe.get("version") != candidate_config.get("recipe_version"):
        raise ValueError("strategy candidate public recipe version changed")
    result = deepcopy(dict(candidate_config))
    for field in (
        "source_research_artifact_id",
        "strategy_research_proposal_sha256",
        "strategy_research_artifact_sha256",
        "parent_strategy_version_id",
        "strategy_research_data_contract",
    ):
        result.pop(field, None)
    for field in _POLICY_FIELDS:
        result.pop(field, None)
    result.update(deepcopy(dict(recipe["config_overrides"])))
    result["recipe_id"] = recipe["id"]
    result["recipe_version"] = recipe["version"]
    champion = candidate_config.get("factor_score_champion_contract")
    if isinstance(champion, Mapping):
        source_artifact_id = str(
            candidate_config.get("source_research_artifact_id") or ""
        ).strip()
        if not source_artifact_id:
            raise ValueError(
                "factor champion control requires its fin_strategy source artifact"
            )
        result["source_research_artifact_id"] = source_artifact_id
        members = champion.get("members")
        if not isinstance(members, list) or not members:
            raise ValueError("factor champion public control has no frozen score weights")
        weights = {
            str(item["feature_id"]): float(item["weight"])
            for item in members
            if isinstance(item, Mapping)
        }
        if len(weights) != len(members):
            raise ValueError("factor champion public control score grid is incomplete")
        public_rule_ir = deepcopy(dict(result["strategy_rule_ir"]))
        slots = deepcopy(dict(public_rule_ir["slots"]))
        alpha_components = slots["alpha_rank"].get("components") or []
        weighted = next(
            (
                item
                for item in alpha_components
                if isinstance(item, dict)
                and item.get("component") == "weighted_factor_rank"
            ),
            None,
        )
        if weighted is None:
            raise ValueError("factor champion public control has no alpha-rank slot")
        weighted["parameters"] = {"weights": weights}
        rules = validate_strategy_rule_ir(
            str(result.get("horizon_profile") or ""),
            slots,
            allowed_factor_ids=set(weights),
        )
        policy = compile_strategy_rule_policy(
            str(result.get("horizon_profile") or ""),
            rules,
            allowed_factor_ids=set(weights),
        )
        result["strategy_rule_ir"] = rules
        result["strategy_rules_sha256"] = rules["rules_sha256"]
        result["strategy_rule_policy_sha256"] = policy["policy_sha256"]
        for field in _POLICY_FIELDS:
            if field in policy and policy[field] is not None:
                result[field] = policy[field]
            elif field in result and policy.get(field) is None:
                result.pop(field)
    result["execution_contract_hash"] = strategy_execution_contract_hash(result)
    validate_strategy_rule_binding(result)
    return result


def build_transparent_full_stack_control_config(
    candidate_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the transparent factor+public-policy control for full-stack comparison.

    ``build_public_strategy_control_config`` deliberately preserves the
    candidate score source for the policy-only ablation. This second helper
    removes every model/ensemble/fin_quant identity and rebinds the public Qlib
    factor baseline, so the later full-stack stage actually tests the complete
    champion-signal + researched-policy stack.
    """

    factor_champion = isinstance(
        candidate_config.get("factor_score_champion_contract"), Mapping
    )
    result = build_public_strategy_control_config(candidate_config)
    for field in {
        *_SCORE_ID_FIELDS,
        "model_signal_contract_version",
        "model_primary_profile_id",
        "model_primary_seed",
        "model_refit_policy",
        "model_refit_policy_sha256",
        "quant_bundle_factor_contract",
        "model_ensemble_manifest_sha256",
        "model_ensemble_evidence_sha256",
        "model_ensemble_combiner",
        "model_ensemble_stacking",
        "model_component_candidate_ids",
        "model_component_families",
        "strategy_research_signal_binding",
        "baseline_definition",
        "factor_score_champion_contract",
    }:
        result.pop(field, None)
    if factor_champion:
        from .strategy_recipes import get_strategy_recipe

        recipe = get_strategy_recipe(str(result.get("recipe_id") or ""))
        public_config = dict(recipe["config_overrides"])
        for field in (*_POLICY_FIELDS, "strategy_rule_ir", "strategy_rules_sha256"):
            if field in public_config:
                result[field] = deepcopy(public_config[field])
            else:
                result.pop(field, None)
        result["strategy_rule_policy_sha256"] = public_config[
            "strategy_rule_policy_sha256"
        ]
        result.pop("source_research_artifact_id", None)
    result.update(
        {
            "signal_source": "factor_score",
            "factor_source_mode": "qlib_baseline",
            "challenger_weight": 0.0,
        }
    )
    # Use the same authoritative normalizer as StrategyStore. It restores the
    # exact public baseline definition, horizon/runtime identity and execution
    # hash without creating another StrategyVersion or state machine.
    from .strategy_store import _normalize_multifactor_contract

    return _normalize_multifactor_contract(
        result,
        factor_count=0,
        creating_family=True,
    )


def _cost_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "contract_version": "strategy-research-cost-v1",
        "cost": {field: config.get(field) for field in _COST_FIELDS},
        "execution": {
            "signal_frequency": config.get("signal_frequency"),
            "execution_frequency": config.get("execution_frequency"),
            "execution_method": config.get("execution_method"),
            "execution_lag_bars": config.get("execution_lag_bars"),
        },
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def _alpha_slot(rule_ir: Mapping[str, Any]) -> dict[str, Any]:
    slots = rule_ir.get("slots")
    if not isinstance(slots, Mapping) or not isinstance(slots.get("alpha_rank"), Mapping):
        raise ValueError("strategy rule IR has no alpha-rank slot")
    return deepcopy(dict(slots["alpha_rank"]))


def _policy_only_challenger(
    baseline_config: Mapping[str, Any], candidate_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Use candidate rules with the baseline alpha slot for a true policy ablation."""

    baseline_rule_ir = baseline_config.get("strategy_rule_ir")
    candidate_rule_ir = candidate_config.get("strategy_rule_ir")
    if not isinstance(baseline_rule_ir, Mapping) or not isinstance(
        candidate_rule_ir, Mapping
    ):
        raise ValueError("strategy comparison requires governed rule IRs")
    derived_rule_ir = deepcopy(dict(candidate_rule_ir))
    slots = deepcopy(dict(derived_rule_ir.get("slots") or {}))
    slots["alpha_rank"] = _alpha_slot(baseline_rule_ir)
    alpha_components = slots["alpha_rank"].get("components") or []
    alpha_weights = next(
        (
            dict(item.get("parameters", {}).get("weights") or {})
            for item in alpha_components
            if isinstance(item, Mapping) and item.get("component") == "weighted_factor_rank"
        ),
        {},
    )
    rules = validate_strategy_rule_ir(
        str(candidate_config.get("horizon_profile") or ""),
        slots,
        allowed_factor_ids=set(alpha_weights),
    )
    policy = compile_strategy_rule_policy(
        str(candidate_config.get("horizon_profile") or ""),
        rules,
        allowed_factor_ids=set(alpha_weights),
    )
    result = deepcopy(dict(candidate_config))
    result["strategy_rule_ir"] = rules
    result["strategy_rules_sha256"] = rules["rules_sha256"]
    result["strategy_rule_policy_sha256"] = policy["policy_sha256"]
    for field in _POLICY_FIELDS:
        if field in policy and policy[field] is not None:
            result[field] = policy[field]
        elif field in result and policy.get(field) is None:
            result.pop(field)
    result["execution_lag_bars"] = int(policy["execution_lag_sessions"])
    result["execution_contract_hash"] = strategy_execution_contract_hash(result)
    validate_strategy_rule_binding(result)
    return result


def _require_periods(periods: Mapping[str, Any]) -> dict[str, Any]:
    if set(periods) != {"in_sample", "out_of_sample", "governance"}:
        raise ValueError("strategy comparison periods must be preregistered")
    normalized = _json_clone(periods)
    governance = normalized["governance"]
    historical = (
        governance.get("historical_validation_periods")
        if isinstance(governance, dict)
        else None
    )
    if (
        not isinstance(governance, dict)
        or governance.get("final_oos_opened") is not False
        or not str(governance.get("pre_final_cutoff") or "")
        or not isinstance(historical, dict)
        or set(historical) != {"start", "end"}
        or not str(historical["start"]) <= str(historical["end"])
    ):
        raise ValueError("strategy comparison must remain before the sealed final OOS")
    cutoff = str(governance["pre_final_cutoff"])
    for segment in ("in_sample", "out_of_sample"):
        value = normalized[segment]
        if (
            not isinstance(value, dict)
            or set(value) != {"start", "end"}
            or not str(value["start"]) <= str(value["end"]) <= cutoff
        ):
            raise ValueError("strategy comparison period crosses the pre-final cutoff")
    if normalized["in_sample"]["end"] >= normalized["out_of_sample"]["start"]:
        raise ValueError("strategy comparison segments overlap")
    if str(historical["end"]) >= str(normalized["in_sample"]["start"]):
        raise ValueError("strategy comparison history overlaps the selection segments")
    return normalized


def derive_strategy_research_competition_periods(
    research_periods: Mapping[str, Any],
    *,
    dataset_path: Path,
    purge_sessions: int,
    minimum_oos_observations: int,
) -> dict[str, Any]:
    """Derive two pre-final comparison segments from the frozen Qlib calendar."""

    calendar_path = Path(dataset_path) / "calendars" / "day.txt"
    if not calendar_path.is_file():
        raise ValueError("strategy comparison requires the frozen Qlib daily calendar")
    calendar = pd.DatetimeIndex(
        pd.to_datetime(
            [
                line.strip()
                for line in calendar_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ],
            errors="coerce",
        )
    ).dropna().sort_values().unique()
    return derive_strategy_research_competition_periods_from_calendar(
        research_periods,
        calendar,
        purge_sessions=purge_sessions,
        minimum_oos_observations=minimum_oos_observations,
    )


def derive_strategy_research_competition_periods_from_calendar(
    research_periods: Mapping[str, Any],
    calendar: pd.DatetimeIndex,
    *,
    purge_sessions: int,
    minimum_oos_observations: int,
) -> dict[str, Any]:
    """Derive the comparison segments against an already loaded calendar."""

    required = {
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
        "test_start",
        "test_end",
    }
    if set(research_periods) != required:
        raise ValueError("strategy research periods are incomplete")
    calendar = pd.DatetimeIndex(calendar).dropna().sort_values().unique()
    train = calendar[
        (calendar >= pd.Timestamp(str(research_periods["train_start"])))
        & (calendar <= pd.Timestamp(str(research_periods["train_end"])))
    ]
    valid = calendar[
        (calendar >= pd.Timestamp(str(research_periods["valid_start"])))
        & (calendar <= pd.Timestamp(str(research_periods["valid_end"])))
    ]
    # Training history may predate the first effective-dated cost record, but
    # policy/full-stack trials place simulated orders and therefore must never
    # trade in that unsupported period.  Clamp both selection segments to the
    # first authoritative cost date while retaining the earlier ``train`` rows
    # for the separate historical context below.
    first_cost_date = pd.Timestamp(
        CN_COST_SCHEDULE_BOOK.versions[0].effective_from
    )
    valid = valid[valid >= first_cost_date]
    purge = int(purge_sessions)
    minimum_oos = int(minimum_oos_observations)
    minimum_history = 252
    minimum_in_sample = max(252, minimum_oos // 2)
    if purge < 1 or minimum_oos < 40:
        raise ValueError("strategy comparison purge or OOS requirement is invalid")
    if len(valid) < minimum_oos:
        raise ValueError("strategy validation window is shorter than its preregistered OOS")
    split = max(minimum_history - 1, len(train) // 2 - 1)
    first_cost_train_index = int(train.searchsorted(first_cost_date, side="left"))
    in_start_index = max(split + purge + 1, first_cost_train_index)
    in_end_index = len(train) - purge - 1
    if (
        len(train) < minimum_history + minimum_in_sample + (2 * purge)
        or in_start_index > in_end_index
        or in_end_index - in_start_index + 1 < minimum_in_sample
    ):
        raise ValueError(
            "cost-covered strategy training window cannot isolate "
            "fair-comparison history"
        )
    historical = {
        "start": train[0].date().isoformat(),
        "end": train[split].date().isoformat(),
    }
    return _require_periods(
        {
            "in_sample": {
                "start": train[in_start_index].date().isoformat(),
                "end": train[in_end_index].date().isoformat(),
            },
            "out_of_sample": {
                "start": valid[0].date().isoformat(),
                "end": valid[-1].date().isoformat(),
            },
            "governance": {
                "final_oos_opened": False,
                "pre_final_cutoff": valid[-1].date().isoformat(),
                "historical_validation_periods": historical,
                "purge_sessions": purge,
            },
        }
    )


def build_strategy_research_competition_plan(
    *,
    research_run_id: str,
    compiled_artifact_id: str,
    compiled_artifact_sha256: str,
    baseline_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    full_stack_control_config: Mapping[str, Any] | None = None,
    dataset: str,
    dataset_identity_sha256: str,
    score_inputs_sha256: str,
    periods: Mapping[str, Any],
    benchmark: str = "SH000300",
    universe: str = "cn_all",
    seed: int = 0,
    preregistered_candidate_count: int = 2,
) -> dict[str, Any]:
    """Pre-register the two-stage fin_strategy competition on the existing DAG."""

    if not research_run_id or not compiled_artifact_id:
        raise ValueError("strategy comparison requires immutable research identities")
    if not all(
        _is_sha256(value)
        for value in (
            compiled_artifact_sha256,
            dataset_identity_sha256,
            score_inputs_sha256,
        )
    ):
        raise ValueError("strategy comparison contains an invalid content digest")
    baseline_policy = validate_strategy_rule_binding(baseline_config)
    candidate_policy = validate_strategy_rule_binding(candidate_config)
    if baseline_policy is None or candidate_policy is None:
        raise ValueError("strategy comparison does not accept legacy ambiguous strategies")
    if (
        baseline_config.get("horizon_profile") != candidate_config.get("horizon_profile")
        or baseline_config.get("recipe_id") != candidate_config.get("recipe_id")
        or baseline_config.get("recipe_version") != candidate_config.get("recipe_version")
    ):
        raise ValueError("strategy candidate and baseline do not share one public control")
    full_stack_control = (
        dict(full_stack_control_config)
        if full_stack_control_config is not None
        else dict(baseline_config)
    )
    full_stack_control_policy = validate_strategy_rule_binding(full_stack_control)
    if (
        full_stack_control_policy is None
        or full_stack_control.get("horizon_profile")
        != candidate_config.get("horizon_profile")
    ):
        raise ValueError("full-stack control belongs to another strategy horizon")
    baseline_cost = _cost_contract(baseline_config)
    candidate_cost = _cost_contract(candidate_config)
    full_stack_control_cost = _cost_contract(full_stack_control)
    if baseline_cost != candidate_cost or full_stack_control_cost != candidate_cost:
        raise ValueError("strategy candidate changed the frozen cost/execution contract")
    normalized_periods = _require_periods(periods)
    policy_challenger = _policy_only_challenger(baseline_config, candidate_config)
    baseline_score = strategy_score_grid_contract(baseline_config)
    policy_score = strategy_score_grid_contract(policy_challenger)
    if baseline_score != policy_score:
        raise ValueError("policy-only challenger changed the score grid")
    if score_inputs_sha256 != baseline_score["contract_sha256"]:
        raise ValueError("strategy comparison score-input digest changed")
    evaluation_contract = candidate_config.get("strategy_evaluation_contract")
    if not isinstance(evaluation_contract, Mapping):
        raise ValueError("strategy candidate has no frozen evaluation contract")
    minimum_oos = int(evaluation_contract.get("minimum_oos_observations") or 0)
    if minimum_oos < 40:
        raise ValueError("strategy comparison has no valid minimum OOS requirement")
    family_size = int(preregistered_candidate_count)
    if family_size < 2 or family_size % 2:
        raise ValueError(
            "strategy comparison family must preregister both trials for every proposal"
        )

    def trial(index: int, role: str, config: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "trial_index": index,
            "role": role,
            "parameters": {"strategy_comparison_role": role},
            "config": _json_clone(config),
            "config_sha256": canonical_sha256(config),
        }

    common = {
        "dataset": dataset,
        "dataset_identity_sha256": dataset_identity_sha256,
        "periods": normalized_periods,
        "benchmark": benchmark,
        "universe": universe,
        "seed": int(seed),
        "cost_contract": baseline_cost,
        "minimum_oos_observations": minimum_oos,
        "job_kind": "parameter_experiment",
        "capital_eligible": False,
        "final_oos_opened": False,
    }
    policy_stage = {
        **common,
        "score_inputs_sha256": score_inputs_sha256,
        "stage": "policy_only",
        "evaluation_mode": STRATEGY_POLICY_ONLY_MODE,
        "parameter_grid": {
            "strategy_comparison_role": ["public_baseline", "policy_challenger"]
        },
        "score_grid_contract": baseline_score,
        "trials": [
            trial(0, "public_baseline", baseline_config),
            trial(1, "policy_challenger", policy_challenger),
        ],
    }
    full_stage = {
        **common,
        "score_inputs_sha256": canonical_sha256(
            {
                "contract_version": "full-stack-score-pair-v1",
                "control": strategy_score_grid_contract(full_stack_control)[
                    "contract_sha256"
                ],
                "challenger": strategy_score_grid_contract(candidate_config)[
                    "contract_sha256"
                ],
            }
        ),
        "stage": "full_stack",
        "evaluation_mode": STRATEGY_FULL_STACK_MODE,
        "parameter_grid": {
            "strategy_comparison_role": [
                "public_baseline",
                "full_stack_challenger",
            ]
        },
        "requires_stage": "policy_only",
        "trials": [
            trial(0, "public_baseline", full_stack_control),
            trial(1, "full_stack_challenger", candidate_config),
        ],
    }
    plan = {
        "contract_version": STRATEGY_RESEARCH_COMPETITION_VERSION,
        "delivery_status": "research_only",
        "capital_eligible": False,
        "simulation_eligible": False,
        "research_run_id": research_run_id,
        "compiled_artifact_id": compiled_artifact_id,
        "compiled_artifact_sha256": compiled_artifact_sha256,
        "horizon": candidate_config["horizon_profile"],
        "preregistered_candidate_count": family_size,
        "family_alpha": 0.05,
        "stages": [policy_stage, full_stage],
        "next_gate_after_success": "formal_final_oos_once",
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def _metric_passes(metrics: Mapping[str, Any]) -> bool:
    # The deflated Sharpe probability is computed and archived as report-only
    # evidence; per the wide-entry gate policy it never vetoes a research gate.
    return (
        metrics.get("robustness_passed") is True
        and metrics.get("component_cost_stress_passed") is True
        and metrics.get("rolling_passed") is True
        and metrics.get("event_stress_passed") is True
        and metrics.get("capacity_curve_passed") is True
    )


def build_strategy_stage_evidence(
    plan: Mapping[str, Any],
    *,
    stage_name: str,
    trial_results: list[Mapping[str, Any]],
    daily_returns: Mapping[str, pd.Series],
    governed_score_sha256: Mapping[str, str],
    prerequisite_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one completed stage and emit an immutable research-only artifact."""

    plan_without_digest = dict(plan)
    plan_digest = plan_without_digest.pop("plan_sha256", None)
    if (
        plan.get("contract_version") != STRATEGY_RESEARCH_COMPETITION_VERSION
        or plan_digest != canonical_sha256(plan_without_digest)
    ):
        raise ValueError("strategy comparison plan is invalid or changed")
    stages = [item for item in plan.get("stages") or [] if item.get("stage") == stage_name]
    if len(stages) != 1:
        raise ValueError("strategy comparison stage is not preregistered")
    stage = stages[0]
    expected_roles = [item["role"] for item in stage["trials"]]
    observed = {
        str(
            item.get("role")
            or (item.get("parameters") or {}).get("strategy_comparison_role")
            or ""
        ): item
        for item in trial_results
    }
    if set(observed) != set(expected_roles) or len(observed) != 2:
        raise ValueError("strategy comparison results changed the frozen trial family")
    if any(item.get("status") != "succeeded" for item in observed.values()):
        raise ValueError("strategy comparison has an incomplete trial")
    if set(daily_returns) != set(expected_roles) or set(governed_score_sha256) != set(
        expected_roles
    ):
        raise ValueError("strategy comparison artifacts are incomplete")
    if not all(_is_sha256(value) for value in governed_score_sha256.values()):
        raise ValueError("strategy comparison score-grid hashes are invalid")
    baseline_role = "public_baseline"
    challenger_role = next(role for role in expected_roles if role != baseline_role)
    baseline = pd.to_numeric(daily_returns[baseline_role], errors="coerce")
    challenger = pd.to_numeric(daily_returns[challenger_role], errors="coerce")
    if (
        not baseline.index.equals(challenger.index)
        or len(baseline) < int(stage.get("minimum_oos_observations") or 0)
        or baseline.isna().any()
        or challenger.isna().any()
        or not np.isfinite(baseline.to_numpy(dtype=float)).all()
        or not np.isfinite(challenger.to_numpy(dtype=float)).all()
    ):
        raise ValueError("strategy comparison returns are not a paired complete grid")
    if stage_name == "policy_only" and len(set(governed_score_sha256.values())) != 1:
        raise ValueError("policy-only trials did not consume the same score grid")
    if stage_name == "full_stack":
        prerequisite = dict(prerequisite_evidence or {})
        if (
            prerequisite.get("stage") != "policy_only"
            or prerequisite.get("gate_passed") is not True
            or not _is_sha256(prerequisite.get("evidence_sha256"))
        ):
            raise ValueError("full-stack evaluation requires passed policy-only evidence")
    else:
        prerequisite = None

    bootstrap = paired_moving_block_bootstrap(
        challenger,
        baseline,
        block_size=min(20, len(challenger)),
        samples=2000,
        seed=int(stage["seed"]),
    )
    family_size = int(plan["preregistered_candidate_count"])
    alpha_threshold = float(plan["family_alpha"]) / family_size
    adjusted_p_value = min(1.0, float(bootstrap["one_sided_p_value"]) * family_size)
    pbo = probability_of_backtest_overfitting(
        pd.concat(
            [
                baseline.rename(baseline_role),
                challenger.rename(challenger_role),
            ],
            axis=1,
        ),
        blocks=8,
    )
    challenger_metrics = dict(
        (observed[challenger_role].get("metrics") or {}).get("out_of_sample") or {}
    )
    # Gate recalibration (wide-in, strict-out): the research stage only
    # fail-closes on data integrity (paired complete grid, finite returns —
    # enforced above) and on the challenger's stress metrics.  The bootstrap
    # mean difference, confidence interval, alpha-spending p-values and PBO
    # are still computed and sealed below, but they are a report-only health
    # check: the single life-or-death gate is the forward paper performance.
    gate_passed = _metric_passes(challenger_metrics)
    evidence = {
        "contract_version": "fin-strategy-stage-evidence-v1",
        "delivery_status": "research_only",
        "capital_eligible": False,
        "final_oos_opened": False,
        "research_run_id": plan["research_run_id"],
        "plan_sha256": plan["plan_sha256"],
        "stage": stage_name,
        "evaluation_mode": stage["evaluation_mode"],
        "trial_roles": expected_roles,
        "governed_score_sha256": dict(governed_score_sha256),
        "paired_block_bootstrap": bootstrap,
        "alpha_spending": {
            "family_alpha": plan["family_alpha"],
            "preregistered_candidate_count": family_size,
            "stage_alpha_threshold": alpha_threshold,
            "holm_equivalent_adjusted_p_value": adjusted_p_value,
        },
        "pbo": pbo,
        # Bootstrap/alpha-spending/PBO above are archived as a health report,
        # not a verdict: they never flip ``gate_passed``.
        "statistical_evidence_role": "report_only",
        "challenger_stress_gates_passed": _metric_passes(challenger_metrics),
        "prerequisite_evidence_sha256": (
            prerequisite.get("evidence_sha256") if prerequisite is not None else None
        ),
        "gate_passed": gate_passed,
        "next_gate": (
            "full_stack_pre_final"
            if stage_name == "policy_only" and gate_passed
            else "formal_final_oos_once"
            if stage_name == "full_stack" and gate_passed
            else "research_rejected"
        ),
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def build_strategy_stage_artifact_from_parameter_experiment(
    plan: Mapping[str, Any],
    *,
    stage_name: str,
    experiment_result: Mapping[str, Any],
    artifact_root: Path,
    prerequisite_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Consume the existing parameter-experiment output without another runner."""

    stages = [item for item in plan.get("stages") or [] if item.get("stage") == stage_name]
    if len(stages) != 1:
        raise ValueError("strategy comparison stage is not preregistered")
    stage = stages[0]
    if (
        experiment_result.get("status") != "ok"
        or experiment_result.get("evaluation_mode") != stage["evaluation_mode"]
        or experiment_result.get("final_oos_opened") is not False
        or experiment_result.get("periods") != stage["periods"]
    ):
        raise ValueError("parameter experiment does not match the strategy stage")
    results = experiment_result.get("trials")
    if not isinstance(results, list):
        raise ValueError("parameter experiment has no trial results")
    plan_trials = {int(item["trial_index"]): item for item in stage["trials"]}
    normalized_results: list[dict[str, Any]] = []
    daily_returns: dict[str, pd.Series] = {}
    score_hashes: dict[str, str] = {}
    for raw in results:
        index = int(raw.get("trial_index", -1))
        trial = plan_trials.get(index)
        if trial is None or raw.get("parameters") != trial["parameters"]:
            raise ValueError("parameter experiment changed a preregistered trial")
        role = str(trial["role"])
        trial_root = artifact_root / f"trial-{index:03d}" / "out_of_sample"
        returns_path = trial_root / "daily_returns.parquet"
        # The governed signal already contains policy filtering (for example
        # TopK), so it is expected to differ in a policy-only ablation.  Compare
        # the raw cross-sectional score grid produced before policy application.
        score_path = trial_root / "score_grid.parquet"
        if not returns_path.is_file() or not score_path.is_file():
            raise ValueError("parameter experiment trial artifacts are missing")
        frame = pd.read_parquet(returns_path)
        if "return" not in frame:
            raise ValueError("parameter experiment return artifact is invalid")
        returns = pd.to_numeric(frame["return"], errors="coerce") - pd.to_numeric(
            frame["cost"] if "cost" in frame else 0.0,
            errors="coerce",
        )
        if "datetime" in frame:
            returns.index = pd.to_datetime(frame["datetime"], errors="coerce")
        elif "date" in frame:
            returns.index = pd.to_datetime(frame["date"], errors="coerce")
        daily_returns[role] = returns
        score_hashes[role] = _sha256_file(score_path)
        normalized_results.append({**dict(raw), "role": role})
    evidence = build_strategy_stage_evidence(
        plan,
        stage_name=stage_name,
        trial_results=normalized_results,
        daily_returns=daily_returns,
        governed_score_sha256=score_hashes,
        prerequisite_evidence=prerequisite_evidence,
    )
    artifact = {
        "contract_version": "fin-strategy-evaluation-artifact-v1",
        "delivery_status": "research_only",
        "capital_eligible": False,
        "research_run_id": plan["research_run_id"],
        "artifact_type": f"fin_strategy_{stage_name}_evaluation",
        "parameter_experiment_id": experiment_result.get("experiment_id"),
        "parameter_experiment_result_sha256": canonical_sha256(experiment_result),
        "evidence": evidence,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact
