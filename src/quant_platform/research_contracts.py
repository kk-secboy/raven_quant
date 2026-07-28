from __future__ import annotations

import hashlib
import json
from typing import Any

RESEARCH_CONTRACT_VERSION = "research-contracts-v1"

RESEARCH_MODES: dict[str, dict[str, Any]] = {
    "template_extension": {
        "artifact": "strategy_spec_candidate",
        "capital_eligible": False,
        "runner": "factor_component_pipeline",
    },
    "component_discovery": {
        "artifact": "versioned_component_evidence",
        "capital_eligible": False,
        "runner": "factor_component_pipeline",
    },
    "model_challenger": {
        "artifact": "model_candidate_calibration",
        "capital_eligible": False,
        "runner": "model_challenger_pipeline",
    },
    "portfolio_execution_challenger": {
        "artifact": "policy_candidate_paired_comparison",
        "capital_eligible": False,
        "runner": "policy_challenger_pipeline",
    },
    "new_strategy_proposal": {
        "artifact": "new_strategy_proposal",
        "capital_eligible": False,
        "runner": "proposal_review_pipeline",
    },
    "falsification_or_retirement": {
        "artifact": "falsification_report",
        "capital_eligible": False,
        "runner": "falsification_pipeline",
    },
}

STRATEGY_SLOT_ORDER = (
    "eligibility_gate",
    "universe_dedup",
    "direction_regime_gate",
    "alpha_rank",
    "entry_timing",
    "exit_state",
    "portfolio_risk",
    "execution_requirement",
)

_BRIEF_REQUIRED_FIELDS = (
    "research_question",
    "economic_gap",
    "allowed_strategy_families",
    "allowed_assets",
    "allowed_frequencies",
    "allowed_data",
    "prohibited_actions",
    "personal_constraints",
    "delivery_status",
    "dataset_snapshot",
    "pit_rules",
    "periods",
    "baseline",
    "primary_metric",
    "failure_conditions",
    "cost_capacity_risk",
    "experiment_budget",
    "runtime_budget",
    "llm_disclosure",
    "final_oos_rule",
)


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def research_mode_catalog() -> list[dict[str, Any]]:
    return [
        {
            "research_mode": mode,
            **details,
            "status": (
                "runnable"
                if details["runner"] == "factor_component_pipeline"
                else "registered_requires_specialized_runner"
            ),
        }
        for mode, details in RESEARCH_MODES.items()
    ]


def freeze_research_brief(
    research_mode: str,
    brief: dict[str, Any],
) -> dict[str, Any]:
    mode = str(research_mode)
    if mode not in RESEARCH_MODES:
        raise ValueError(f"unknown research mode: {mode}")
    if not isinstance(brief, dict):
        raise ValueError("ResearchBrief must be an object")
    missing = [
        field
        for field in _BRIEF_REQUIRED_FIELDS
        if field not in brief
        or brief[field] is None
        or brief[field] == ""
        or brief[field] == []
        or brief[field] == {}
    ]
    if missing:
        raise ValueError("ResearchBrief is incomplete: " + ", ".join(missing))
    delivery = str(brief["delivery_status"])
    if delivery not in {
        "research_only",
        "simulation_only",
        "eligible_for_formal_review",
    }:
        raise ValueError("ResearchBrief delivery_status is invalid")
    if mode == "new_strategy_proposal" and delivery != "research_only":
        raise ValueError("new strategy proposals must remain research_only")
    prohibited = {str(item) for item in brief["prohibited_actions"]}
    mandatory_prohibitions = {
        "production_database_write",
        "broker_order_submission",
        "final_oos_visibility_during_selection",
        "secret_access",
    }
    if not mandatory_prohibitions <= prohibited:
        raise ValueError(
            "ResearchBrief must explicitly prohibit production writes, broker "
            "orders, secret access and final-OOS visibility"
        )
    frozen = {
        "contract_version": RESEARCH_CONTRACT_VERSION,
        "research_mode": mode,
        "mode_contract": RESEARCH_MODES[mode],
        **brief,
    }
    frozen["brief_sha256"] = _canonical_sha256(frozen)
    return frozen


def validate_strategy_spec_slots(
    slots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate the fixed eight-slot order and explicit empty-slot behavior."""

    if not isinstance(slots, dict) or tuple(slots) != STRATEGY_SLOT_ORDER:
        raise ValueError(
            "StrategySpec slots must use the frozen eight-slot order: "
            + ", ".join(STRATEGY_SLOT_ORDER)
        )
    normalized: dict[str, dict[str, Any]] = {}
    for name in STRATEGY_SLOT_ORDER:
        slot = slots[name]
        if not isinstance(slot, dict):
            raise ValueError(f"StrategySpec slot {name} must be an object")
        required = bool(slot.get("required"))
        components = slot.get("components")
        if not isinstance(components, list):
            raise ValueError(f"StrategySpec slot {name} components must be a list")
        empty_behavior = str(slot.get("empty_behavior") or "")
        if not components and not empty_behavior:
            raise ValueError(
                f"StrategySpec slot {name} must declare empty_behavior"
            )
        if required and not components:
            raise ValueError(f"required StrategySpec slot {name} cannot be empty")
        normalized[name] = {
            "required": required,
            "components": [dict(item) for item in components],
            "empty_behavior": empty_behavior,
        }
    return {
        "contract_version": RESEARCH_CONTRACT_VERSION,
        "control_order": list(STRATEGY_SLOT_ORDER),
        "slots": normalized,
        "slots_sha256": _canonical_sha256(normalized),
    }


def default_campaign_research_brief(
    *,
    objective: str,
    dataset: str,
    benchmark: str,
    universe: str,
    recipe_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    research = dict(config.get("research") or {})
    strategy = dict(config.get("strategy_config") or {})
    evidence = dict(config.get("dataset_evidence") or {})
    periods = dict(research.get("periods") or {})
    budget = {
        "max_candidates": int(config.get("max_factors", 1)),
        "max_parameter_trials": len(config.get("experiment_trials") or []),
        "random_seeds": [0],
    }
    raw = {
        "research_question": objective,
        "economic_gap": f"test whether {recipe_id} adds robust evidence versus its baseline",
        "allowed_strategy_families": [recipe_id],
        "allowed_assets": [universe],
        "allowed_frequencies": [str(strategy.get("signal_frequency") or "day")],
        "allowed_data": [dataset],
        "prohibited_actions": [
            "production_database_write",
            "broker_order_submission",
            "final_oos_visibility_during_selection",
            "secret_access",
            "automatic_capital_allocation",
        ],
        "personal_constraints": {
            "benchmark": benchmark,
            "capacity_notional": strategy.get("capacity_notional"),
            "topk": strategy.get("topk"),
        },
        "delivery_status": "eligible_for_formal_review",
        "dataset_snapshot": {
            "name": evidence.get("name") or dataset,
            "identity": (evidence.get("provenance") or {}).get(
                "dataset_identity_sha256"
            ),
            "path": evidence.get("path"),
        },
        "pit_rules": {
            "lineage": (evidence.get("provenance") or {}).get(
                "dataset_lineage_id"
            ),
            "decision_time_rule": "available_at_before_decision",
        },
        "periods": periods,
        "baseline": strategy.get("baseline_definition")
        or {"kind": "benchmark", "instrument": benchmark},
        "primary_metric": "information_ratio",
        "failure_conditions": {
            "max_drawdown": strategy.get("max_drawdown"),
            "minimum_robustness_pass_rate": strategy.get(
                "min_robustness_pass_rate"
            ),
            "non_finite_result": "fail",
        },
        "cost_capacity_risk": {
            "cost_schedule_version": strategy.get("cost_schedule_version"),
            "capacity_curve_notionals": strategy.get(
                "capacity_curve_notionals"
            ),
            "max_position_weight": strategy.get("max_position_weight"),
        },
        "experiment_budget": budget,
        "runtime_budget": {
            "loop_n": (research.get("budget") or {}).get("loop_n")
            or research.get("loop_n")
            or 1,
            "duration": (research.get("budget") or {}).get("duration")
            or research.get("duration")
            or "bounded",
            "concurrency": 1,
        },
        "llm_disclosure": {
            "provider": research.get("llm_provider") or "runtime_configured",
            "model": research.get("llm_model") or "runtime_configured",
            "knowledge_cutoff": research.get("llm_knowledge_cutoff")
            or "not_disclosed",
            "network_access": bool(research.get("network_access", False)),
        },
        "final_oos_rule": {
            "visible_during_selection": False,
            "one_time_consumption": True,
        },
    }
    return freeze_research_brief("template_extension", raw)
