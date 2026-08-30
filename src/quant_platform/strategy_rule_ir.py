from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from quant_platform.research_contracts import STRATEGY_SLOT_ORDER

STRATEGY_RULE_IR_VERSION = "strategy-rule-ir-v1"

HORIZON_CONTRACTS: dict[str, dict[str, Any]] = {
    "short_1_5d": {
        "label_horizon_trading_days": 5,
        "holding_period_trading_days": [1, 5],
        "decision_frequency": "day",
        "review_frequency": "day",
    },
    "swing_1_6m": {
        "label_horizon_trading_days": 63,
        "holding_period_trading_days": [21, 126],
        "decision_frequency": "week",
        "review_frequency": "week",
    },
    "long_1_3y": {
        "label_horizon_trading_days": 252,
        "holding_period_trading_days": [252, 756],
        "decision_frequency": "month",
        "review_frequency": "month",
    },
}

_ALLOWED_EMPTY_BEHAVIOURS = frozenset(
    {
        "pass_through",
        "no_new_entries",
        "hold_existing",
        "remain_in_cash",
        "use_baseline",
    }
)

_SLOT_COMPONENTS: dict[str, frozenset[str]] = {
    "eligibility_gate": frozenset(
        {"tradable_ashare", "liquidity_floor", "regulatory_exclusion"}
    ),
    "universe_dedup": frozenset({"instrument_unique"}),
    "direction_regime_gate": frozenset(
        {"long_only", "market_trend_filter", "valuation_regime_filter"}
    ),
    # The first contract accepts only factors from the staged immutable
    # feature set. Model artifacts need their own immutable input binding and
    # are deliberately not represented by a free-form string id here.
    "alpha_rank": frozenset({"weighted_factor_rank"}),
    "entry_timing": frozenset(
        {"next_open", "score_threshold", "extension_guard", "rebalance_calendar"}
    ),
    "exit_state": frozenset(
        {
            "max_holding_days",
            "score_drop_exit",
            "score_deterioration_reduce",
            "valuation_reduce",
            "stop_loss",
            "trend_break",
            "thesis_break",
        }
    ),
    "portfolio_risk": frozenset(
        {
            "topk_equal_weight",
            "max_industry_weight",
            "max_daily_turnover",
            "minimum_trade_band",
            "cash_when_no_edge",
        }
    ),
    "execution_requirement": frozenset(
        {
            "a_share_t_plus_one",
            "board_lot",
            "price_limit_guard",
            "liquidity_participation",
        }
    ),
}

_REQUIRED_COMPONENTS: dict[str, frozenset[str]] = {
    "eligibility_gate": frozenset(
        {"tradable_ashare", "liquidity_floor", "regulatory_exclusion"}
    ),
    "universe_dedup": frozenset({"instrument_unique"}),
    "direction_regime_gate": frozenset({"long_only"}),
    "alpha_rank": frozenset(),
    "entry_timing": frozenset({"next_open", "score_threshold", "rebalance_calendar"}),
    "exit_state": frozenset(),
    "portfolio_risk": frozenset(
        {
            "topk_equal_weight",
            "max_industry_weight",
            "max_daily_turnover",
            "cash_when_no_edge",
        }
    ),
    "execution_requirement": frozenset(
        {
            "a_share_t_plus_one",
            "board_lot",
            "price_limit_guard",
            "liquidity_participation",
        }
    ),
}

_SLOT_EMPTY_BEHAVIOUR = {
    "eligibility_gate": "no_new_entries",
    "universe_dedup": "no_new_entries",
    "direction_regime_gate": "remain_in_cash",
    "alpha_rank": "remain_in_cash",
    "entry_timing": "no_new_entries",
    "exit_state": "hold_existing",
    "portfolio_risk": "remain_in_cash",
    "execution_requirement": "no_new_entries",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _plain_number(value: Any, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    normalized = float(value)
    if not minimum <= normalized <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return normalized


def _plain_integer(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _exact_parameters(
    component: str,
    parameters: Any,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(parameters, Mapping):
        raise ValueError(f"strategy component {component} parameters must be an object")
    actual = {str(key) for key in parameters}
    missing = required - actual
    unexpected = actual - required - optional
    if missing or unexpected:
        raise ValueError(
            f"strategy component {component} parameter contract drifted: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return {str(key): value for key, value in parameters.items()}


def _validate_component_parameters(
    component: str,
    parameters: Any,
    *,
    allowed_factor_ids: set[str] | None,
) -> dict[str, Any]:
    if component == "tradable_ashare":
        values = _exact_parameters(component, parameters, frozenset({"min_listing_days"}))
        return {
            "min_listing_days": _plain_integer(
                values["min_listing_days"], name="min_listing_days", minimum=1, maximum=2520
            )
        }
    if component == "liquidity_floor":
        values = _exact_parameters(
            component, parameters, frozenset({"min_average_daily_amount", "lookback_days"})
        )
        return {
            "min_average_daily_amount": _plain_number(
                values["min_average_daily_amount"],
                name="min_average_daily_amount",
                minimum=100_000_000,
                maximum=1e13,
            ),
            "lookback_days": _plain_integer(
                values["lookback_days"], name="lookback_days", minimum=1, maximum=252
            ),
        }
    if component == "regulatory_exclusion":
        values = _exact_parameters(component, parameters, frozenset({"exclude"}))
        exclude = values["exclude"]
        allowed = {"st", "suspended", "delisting_risk", "regulatory_investigation"}
        if (
            not isinstance(exclude, list)
            or not exclude
            or any(not isinstance(item, str) or item not in allowed for item in exclude)
        ):
            raise ValueError("regulatory_exclusion.exclude contains an unsupported status")
        if not {"st", "suspended", "delisting_risk"}.issubset(exclude):
            raise ValueError(
                "regulatory_exclusion.exclude must cover st, suspended and delisting_risk"
            )
        return {"exclude": sorted(set(exclude))}
    if component == "instrument_unique":
        values = _exact_parameters(component, parameters, frozenset({"identity"}))
        if values["identity"] != "canonical_instrument_id":
            raise ValueError("instrument_unique.identity must be canonical_instrument_id")
        return values
    if component == "long_only":
        return _exact_parameters(component, parameters, frozenset())
    if component == "market_trend_filter":
        values = _exact_parameters(component, parameters, frozenset({"benchmark", "lookback_days"}))
        benchmark = str(values["benchmark"])
        if benchmark not in {"SH000300", "SH000905", "SH000001"}:
            raise ValueError("market_trend_filter benchmark is not allowlisted")
        return {
            "benchmark": benchmark,
            "lookback_days": _plain_integer(
                values["lookback_days"], name="lookback_days", minimum=5, maximum=504
            ),
        }
    if component == "valuation_regime_filter":
        values = _exact_parameters(component, parameters, frozenset({"max_percentile"}))
        return {
            "max_percentile": _plain_number(
                values["max_percentile"], name="max_percentile", minimum=0.05, maximum=1.0
            )
        }
    if component == "weighted_factor_rank":
        values = _exact_parameters(component, parameters, frozenset({"weights"}))
        weights = values["weights"]
        if not isinstance(weights, Mapping) or not weights:
            raise ValueError("weighted_factor_rank.weights must be a non-empty object")
        normalized: dict[str, float] = {}
        for raw_name, raw_weight in weights.items():
            name = str(raw_name)
            if allowed_factor_ids is None:
                raise ValueError("weighted_factor_rank requires a governed factor allowlist")
            if not name or name not in allowed_factor_ids:
                raise ValueError(f"weighted_factor_rank references an ungoverned factor: {name}")
            normalized[name] = _plain_number(
                raw_weight, name=f"factor weight {name}", minimum=0.0, maximum=1.0
            )
        if abs(sum(normalized.values()) - 1.0) > 1e-9:
            raise ValueError("weighted_factor_rank weights must sum to 1")
        return {"weights": dict(sorted(normalized.items()))}
    if component == "next_open":
        values = _exact_parameters(component, parameters, frozenset({"max_signal_age_days"}))
        return {
            "max_signal_age_days": _plain_integer(
                values["max_signal_age_days"], name="max_signal_age_days", minimum=1, maximum=5
            )
        }
    if component == "score_threshold":
        values = _exact_parameters(component, parameters, frozenset({"minimum_percentile"}))
        return {
            "minimum_percentile": _plain_number(
                values["minimum_percentile"], name="minimum_percentile", minimum=0.5, maximum=0.999
            )
        }
    if component == "extension_guard":
        values = _exact_parameters(component, parameters, frozenset({"max_return_5d"}))
        return {
            "max_return_5d": _plain_number(
                values["max_return_5d"], name="max_return_5d", minimum=0.01, maximum=0.5
            )
        }
    if component == "rebalance_calendar":
        values = _exact_parameters(component, parameters, frozenset({"frequency"}))
        frequency = str(values["frequency"])
        if frequency not in {"day", "week", "month", "quarter"}:
            raise ValueError("rebalance_calendar.frequency is invalid")
        return {"frequency": frequency}
    if component == "max_holding_days":
        values = _exact_parameters(component, parameters, frozenset({"days"}))
        return {
            "days": _plain_integer(values["days"], name="days", minimum=1, maximum=1260)
        }
    if component == "score_drop_exit":
        values = _exact_parameters(component, parameters, frozenset({"below_percentile"}))
        return {
            "below_percentile": _plain_number(
                values["below_percentile"], name="below_percentile", minimum=0.0, maximum=0.9
            )
        }
    if component == "score_deterioration_reduce":
        values = _exact_parameters(
            component,
            parameters,
            frozenset({"below_percentile", "reduce_fraction"}),
        )
        return {
            "below_percentile": _plain_number(
                values["below_percentile"],
                name="below_percentile",
                minimum=0.0,
                maximum=0.9,
            ),
            "reduce_fraction": _plain_number(
                values["reduce_fraction"],
                name="reduce_fraction",
                minimum=0.05,
                maximum=0.95,
            ),
        }
    if component == "valuation_reduce":
        values = _exact_parameters(
            component,
            parameters,
            frozenset({"above_percentile", "reduce_fraction"}),
        )
        return {
            "above_percentile": _plain_number(
                values["above_percentile"],
                name="above_percentile",
                minimum=0.5,
                maximum=1.0,
            ),
            "reduce_fraction": _plain_number(
                values["reduce_fraction"],
                name="reduce_fraction",
                minimum=0.05,
                maximum=0.95,
            ),
        }
    if component == "stop_loss":
        values = _exact_parameters(component, parameters, frozenset({"fraction"}))
        return {
            "fraction": _plain_number(
                values["fraction"], name="fraction", minimum=0.01, maximum=0.5
            )
        }
    if component == "trend_break":
        values = _exact_parameters(component, parameters, frozenset({"lookback_days"}))
        return {
            "lookback_days": _plain_integer(
                values["lookback_days"], name="lookback_days", minimum=5, maximum=252
            )
        }
    if component == "thesis_break":
        values = _exact_parameters(
            component, parameters, frozenset({"minimum_holding_days", "review_frequency"})
        )
        frequency = str(values["review_frequency"])
        if frequency not in {"month", "quarter"}:
            raise ValueError("thesis_break.review_frequency must be month or quarter")
        return {
            "minimum_holding_days": _plain_integer(
                values["minimum_holding_days"],
                name="minimum_holding_days",
                minimum=21,
                maximum=756,
            ),
            "review_frequency": frequency,
        }
    if component == "topk_equal_weight":
        values = _exact_parameters(
            component, parameters, frozenset({"topk", "max_position_weight"})
        )
        topk = _plain_integer(values["topk"], name="topk", minimum=1, maximum=500)
        weight = _plain_number(
            values["max_position_weight"], name="max_position_weight", minimum=0.001, maximum=1.0
        )
        if topk * weight < 1.0 - 1e-9:
            raise ValueError("topk * max_position_weight cannot fund a fully invested portfolio")
        return {"topk": topk, "max_position_weight": weight}
    if component == "max_industry_weight":
        values = _exact_parameters(component, parameters, frozenset({"fraction"}))
        return {
            "fraction": _plain_number(
                values["fraction"], name="fraction", minimum=0.01, maximum=1.0
            )
        }
    if component == "max_daily_turnover":
        values = _exact_parameters(component, parameters, frozenset({"fraction"}))
        return {
            "fraction": _plain_number(
                values["fraction"], name="fraction", minimum=0.0, maximum=1.0
            )
        }
    if component == "minimum_trade_band":
        values = _exact_parameters(component, parameters, frozenset({"fraction"}))
        return {
            "fraction": _plain_number(
                values["fraction"], name="fraction", minimum=0.0, maximum=0.10
            )
        }
    if component == "cash_when_no_edge":
        return _exact_parameters(component, parameters, frozenset())
    if component == "a_share_t_plus_one":
        return _exact_parameters(component, parameters, frozenset())
    if component == "board_lot":
        values = _exact_parameters(component, parameters, frozenset({"shares"}))
        shares = _plain_integer(values["shares"], name="shares", minimum=100, maximum=1000)
        if shares % 100:
            raise ValueError("board_lot.shares must be a multiple of 100")
        return {"shares": shares}
    if component == "price_limit_guard":
        return _exact_parameters(component, parameters, frozenset())
    if component == "liquidity_participation":
        values = _exact_parameters(component, parameters, frozenset({"max_fraction"}))
        return {
            "max_fraction": _plain_number(
                values["max_fraction"], name="max_fraction", minimum=0.0001, maximum=0.1
            )
        }
    raise ValueError(f"unsupported strategy component: {component}")


def _component_names(slots: Mapping[str, Any], slot: str) -> list[str]:
    return [str(item["component"]) for item in slots[slot]["components"]]


def _enforce_horizon_contract(horizon: str, slots: Mapping[str, Any]) -> None:
    exits = slots["exit_state"]["components"]
    holding = next(
        (item["parameters"]["days"] for item in exits if item["component"] == "max_holding_days"),
        None,
    )
    entries = _component_names(slots, "entry_timing")
    if "next_open" not in entries:
        raise ValueError("all daily recommendations must execute no earlier than next open")
    rebalance_frequency = next(
        (
            item["parameters"]["frequency"]
            for item in slots["entry_timing"]["components"]
            if item["component"] == "rebalance_calendar"
        ),
        None,
    )
    if rebalance_frequency != HORIZON_CONTRACTS[horizon]["review_frequency"]:
        raise ValueError(f"{horizon} requires its governed review/rebalance frequency")
    min_listing_days = next(
        item["parameters"]["min_listing_days"]
        for item in slots["eligibility_gate"]["components"]
        if item["component"] == "tradable_ashare"
    )
    required_history = {"short_1_5d": 60, "swing_1_6m": 252, "long_1_3y": 756}[
        horizon
    ]
    if min_listing_days < required_history:
        raise ValueError(f"{horizon} requires at least {required_history} listing days")
    if horizon == "short_1_5d" and (holding is None or holding > 5):
        raise ValueError("short_1_5d requires max_holding_days no greater than 5")
    if horizon == "swing_1_6m" and (holding is None or not 21 <= holding <= 126):
        raise ValueError("swing_1_6m requires max_holding_days between 21 and 126")
    if horizon == "long_1_3y":
        thesis = next(
            (
                item
                for item in slots["exit_state"]["components"]
                if item["component"] == "thesis_break"
            ),
            None,
        )
        if thesis is None:
            raise ValueError("long_1_3y requires a thesis_break exit")
        if thesis["parameters"]["review_frequency"] != "month":
            raise ValueError("long_1_3y requires monthly thesis review")
        if thesis["parameters"]["minimum_holding_days"] < 252:
            raise ValueError("long_1_3y thesis review cannot force an exit before 252 days")
        if holding is not None and holding < 252:
            raise ValueError("long_1_3y cannot force an exit before 252 trading days")


def validate_strategy_rule_ir(
    horizon: str,
    slots: Any,
    *,
    allowed_factor_ids: set[str] | None = None,
) -> dict[str, Any]:
    if horizon not in HORIZON_CONTRACTS:
        raise ValueError(f"unsupported strategy horizon: {horizon}")
    if not isinstance(slots, Mapping) or set(slots) != set(STRATEGY_SLOT_ORDER):
        raise ValueError(
            "strategy rules must contain the frozen slots: " + ", ".join(STRATEGY_SLOT_ORDER)
        )
    normalized_slots: dict[str, dict[str, Any]] = {}
    for slot_name in STRATEGY_SLOT_ORDER:
        raw_slot = slots[slot_name]
        if not isinstance(raw_slot, Mapping):
            raise ValueError(f"strategy slot {slot_name} must be an object")
        if set(raw_slot) != {"required", "components", "empty_behavior"}:
            raise ValueError(f"strategy slot {slot_name} contract drifted")
        required = raw_slot["required"]
        components = raw_slot["components"]
        empty_behavior = raw_slot["empty_behavior"]
        if required is not True or not isinstance(components, list):
            raise ValueError(f"strategy slot {slot_name} has invalid types")
        if not isinstance(empty_behavior, str) or empty_behavior not in _ALLOWED_EMPTY_BEHAVIOURS:
            raise ValueError(f"strategy slot {slot_name} empty_behavior is invalid")
        if empty_behavior != _SLOT_EMPTY_BEHAVIOUR[slot_name]:
            raise ValueError(f"strategy slot {slot_name} empty_behavior is not fail-closed")
        if required and not components:
            raise ValueError(f"required strategy slot {slot_name} cannot be empty")
        normalized_components: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_component in components:
            if not isinstance(raw_component, Mapping) or set(raw_component) != {
                "component",
                "parameters",
            }:
                raise ValueError(f"strategy slot {slot_name} component contract drifted")
            component = str(raw_component["component"])
            if component not in _SLOT_COMPONENTS[slot_name]:
                raise ValueError(f"component {component} is not allowed in slot {slot_name}")
            if component in seen:
                raise ValueError(f"component {component} is duplicated in slot {slot_name}")
            seen.add(component)
            normalized_components.append(
                {
                    "component": component,
                    "parameters": _validate_component_parameters(
                        component,
                        raw_component["parameters"],
                        allowed_factor_ids=allowed_factor_ids,
                    ),
                }
            )
        missing_components = _REQUIRED_COMPONENTS[slot_name] - seen
        if missing_components:
            raise ValueError(
                f"strategy slot {slot_name} omits mandatory components: "
                f"{sorted(missing_components)}"
            )
        normalized_slots[slot_name] = {
            "required": required,
            "components": normalized_components,
            "empty_behavior": empty_behavior,
        }
    _enforce_horizon_contract(horizon, normalized_slots)
    result = {
        "contract_version": STRATEGY_RULE_IR_VERSION,
        "horizon": horizon,
        "horizon_contract": dict(HORIZON_CONTRACTS[horizon]),
        "control_order": list(STRATEGY_SLOT_ORDER),
        "slots": normalized_slots,
    }
    result["rules_sha256"] = canonical_sha256(result)
    return result


def strategy_component_catalog() -> dict[str, list[str]]:
    return {slot: sorted(components) for slot, components in _SLOT_COMPONENTS.items()}
