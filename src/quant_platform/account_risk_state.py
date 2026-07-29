"""Selected-account risk state and fail-closed recommendation overlay.

The design deliberately keeps account risk simple: ``normal``, ``caution`` or
``risk_off``.  This module is pure and deterministic.  It never claims to
measure household wealth: every assessment is explicitly scoped to
``selected_account_only``.

Two risk-off modes are kept distinct:

* trustworthy account/market data plus a hard portfolio breach may create
  reduction/exit targets;
* stale or incomplete data may only block new risk and delay ordinary
  execution.  It must not manufacture a confident exit from untrusted prices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from math import isfinite
from typing import Any

RISK_NORMAL = "normal"
RISK_CAUTION = "caution"
RISK_OFF = "risk_off"
RISK_STATES = (RISK_NORMAL, RISK_CAUTION, RISK_OFF)
RISK_SCOPE = "selected_account_only"
RISK_MODEL_VERSION = "selected-account-risk-v1"
LEDGER_POLICY_RISK_VERSION = "ledger-policy-risk-v1"


@dataclass(frozen=True, slots=True)
class AccountRiskThresholds:
    concentration_caution: float = 0.25
    concentration_risk_off: float = 0.45
    drawdown_caution: float = 0.10
    drawdown_risk_off: float = 0.20
    annualized_volatility_caution: float = 0.30
    annualized_volatility_risk_off: float = 0.50
    illiquid_weight_caution: float = 0.20
    illiquid_weight_risk_off: float = 0.40
    caution_new_risk_multiplier: float = 0.50

    def validate(self) -> None:
        pairs = (
            ("concentration", self.concentration_caution, self.concentration_risk_off),
            ("drawdown", self.drawdown_caution, self.drawdown_risk_off),
            (
                "annualized_volatility",
                self.annualized_volatility_caution,
                self.annualized_volatility_risk_off,
            ),
            ("illiquid_weight", self.illiquid_weight_caution, self.illiquid_weight_risk_off),
        )
        for label, caution, risk_off in pairs:
            if not 0 <= caution < risk_off <= 1:
                raise ValueError(f"{label} thresholds must satisfy 0 <= caution < risk_off <= 1")
        if not 0 <= self.caution_new_risk_multiplier <= 1:
            raise ValueError("caution_new_risk_multiplier must stay within [0, 1]")


def _ratio(name: str, value: float) -> float:
    normalized = float(value)
    if not isfinite(normalized) or not 0 <= normalized <= 1:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return normalized


def _non_negative(name: str, value: float) -> float:
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def ledger_policy_risk_inputs(
    *,
    portfolio_id: str,
    portfolio_status: str,
    latest_nav: dict[str, Any] | None,
    position_count: int,
    required_nav_date: date | None = None,
) -> dict[str, Any]:
    """Translate certified account-ledger facts into PortfolioPolicy inputs.

    A brand-new empty account may create its first target. Once positions
    exist, missing, stale or uncertified evidence blocks new risk and supplies
    neutral return inputs so untrusted prices cannot manufacture a liquidation.
    """

    count = int(position_count)
    if count < 0:
        raise ValueError("position_count must be non-negative")
    base = {
        "contract_version": LEDGER_POLICY_RISK_VERSION,
        "risk_scope": RISK_SCOPE,
        "portfolio_id": str(portfolio_id),
        "portfolio_drawdown": 0.0,
        "daily_return": 0.0,
        "allow_new_risk": False,
    }
    if str(portfolio_status) != "active":
        return {
            **base,
            "status": "blocked_inactive_account",
            "reasons": ["simulation_account_not_active"],
        }
    if latest_nav is None:
        if count == 0:
            return {
                **base,
                "status": "initial_empty_account",
                "allow_new_risk": True,
                "reasons": [],
            }
        return {
            **base,
            "status": "blocked_missing_nav",
            "reasons": ["positions_or_orders_exist_without_nav"],
        }
    reasons: list[str] = []
    nav_trade_date: date | None = None
    raw_nav_trade_date = latest_nav.get("trade_date")
    try:
        nav_trade_date = (
            raw_nav_trade_date
            if isinstance(raw_nav_trade_date, date)
            else date.fromisoformat(str(raw_nav_trade_date))
        )
    except (TypeError, ValueError):
        reasons.append("nav_trade_date_invalid")
    if required_nav_date is not None:
        base["required_nav_date"] = required_nav_date.isoformat()
        if nav_trade_date is None:
            reasons.append("nav_trade_date_missing")
        elif nav_trade_date < required_nav_date:
            reasons.append("nav_lags_required_trade_date")
        elif nav_trade_date > required_nav_date:
            reasons.append("nav_is_after_required_trade_date")
    if not bool(latest_nav.get("performance_certified")):
        reasons.append("performance_not_certified")
    if bool(latest_nav.get("has_stale_prices")):
        reasons.append("stale_prices")
    if str(latest_nav.get("status") or "") != "healthy":
        reasons.append("nav_not_healthy")
    if str(latest_nav.get("twr_status") or "") != "ok":
        reasons.append("twr_chain_not_ok")
    try:
        drawdown = float(latest_nav["twr_drawdown"])
        daily_return = float(latest_nav["twr_daily_return"])
    except (KeyError, TypeError, ValueError):
        reasons.append("unitized_risk_values_missing")
        drawdown = 0.0
        daily_return = 0.0
    if (
        not isfinite(drawdown)
        or not -1.0 <= drawdown <= 0.0
        or not isfinite(daily_return)
        or daily_return < -1.0
    ):
        reasons.append("unitized_risk_values_invalid")
        drawdown = 0.0
        daily_return = 0.0
    if reasons:
        return {
            **base,
            "status": "blocked_untrusted_ledger",
            "reasons": sorted(set(reasons)),
        }
    return {
        **base,
        "status": "certified",
        "portfolio_drawdown": drawdown,
        "daily_return": daily_return,
        "allow_new_risk": True,
        "reasons": [],
        "nav_trade_date": nav_trade_date.isoformat() if nav_trade_date else "",
    }


def assess_account_risk(
    *,
    cash_shortfall: float = 0.0,
    max_position_weight: float = 0.0,
    investment_wealth_drawdown: float = 0.0,
    annualized_volatility: float = 0.0,
    illiquid_weight: float = 0.0,
    unresolved_order_count: int = 0,
    data_stale: bool = False,
    account_state_stale: bool = False,
    market_data_trusted: bool = True,
    thresholds: AccountRiskThresholds | None = None,
) -> dict[str, Any]:
    """Return a frozen, auditable selected-account risk assessment."""

    policy = thresholds or AccountRiskThresholds()
    policy.validate()
    shortfall = float(cash_shortfall)
    if not isfinite(shortfall) or shortfall < 0:
        raise ValueError("cash_shortfall must be finite and non-negative")
    unresolved = int(unresolved_order_count)
    if unresolved < 0:
        raise ValueError("unresolved_order_count must be non-negative")
    metrics = {
        "cash_shortfall": shortfall,
        "max_position_weight": _ratio("max_position_weight", max_position_weight),
        "investment_wealth_drawdown": _ratio(
            "investment_wealth_drawdown", investment_wealth_drawdown
        ),
        "annualized_volatility": _non_negative(
            "annualized_volatility", annualized_volatility
        ),
        "illiquid_weight": _ratio("illiquid_weight", illiquid_weight),
        "unresolved_order_count": unresolved,
        "data_stale": bool(data_stale),
        "account_state_stale": bool(account_state_stale),
        "market_data_trusted": bool(market_data_trusted),
    }
    caution: list[str] = []
    hard: list[str] = []
    untrusted: list[str] = []
    if metrics["data_stale"]:
        untrusted.append("data_stale")
    if metrics["account_state_stale"]:
        untrusted.append("account_state_stale")
    if not metrics["market_data_trusted"]:
        untrusted.append("market_data_untrusted")
    if shortfall > 0:
        hard.append("cash_shortfall")
    for name, value, caution_limit, hard_limit in (
        (
            "concentration",
            metrics["max_position_weight"],
            policy.concentration_caution,
            policy.concentration_risk_off,
        ),
        (
            "drawdown",
            metrics["investment_wealth_drawdown"],
            policy.drawdown_caution,
            policy.drawdown_risk_off,
        ),
        (
            "annualized_volatility",
            metrics["annualized_volatility"],
            policy.annualized_volatility_caution,
            policy.annualized_volatility_risk_off,
        ),
        (
            "illiquid_weight",
            metrics["illiquid_weight"],
            policy.illiquid_weight_caution,
            policy.illiquid_weight_risk_off,
        ),
    ):
        if value >= hard_limit:
            hard.append(f"{name}_risk_off")
        elif value >= caution_limit:
            caution.append(f"{name}_caution")
    if unresolved:
        caution.append("unresolved_orders_require_review")

    if hard or untrusted:
        state = RISK_OFF
    elif caution:
        state = RISK_CAUTION
    else:
        state = RISK_NORMAL
    # Reduction/exit targets require trustworthy market and account state.
    exit_existing_risk = bool(hard) and not untrusted
    return {
        "model_version": RISK_MODEL_VERSION,
        "risk_state": state,
        "risk_scope": RISK_SCOPE,
        "reasons": sorted(set(hard + untrusted + caution)),
        "hard_reasons": sorted(set(hard)),
        "untrusted_reasons": sorted(set(untrusted)),
        "metrics": metrics,
        "thresholds": asdict(policy),
        "new_risk_multiplier": (
            1.0
            if state == RISK_NORMAL
            else policy.caution_new_risk_multiplier
            if state == RISK_CAUTION
            else 0.0
        ),
        "manual_review_required": state != RISK_NORMAL,
        "exit_existing_risk": exit_existing_risk,
    }


def apply_risk_overlay(
    instrument: dict[str, Any],
    assessment: dict[str, Any],
) -> dict[str, Any]:
    """Overlay account risk without allowing it to increase ordinary Alpha risk."""

    state = str(assessment.get("risk_state"))
    scope = str(assessment.get("risk_scope"))
    if state not in RISK_STATES:
        raise ValueError(f"unknown risk_state: {state}")
    if scope != RISK_SCOPE:
        raise ValueError(f"risk_scope must be {RISK_SCOPE}")
    item = dict(instrument)
    target = item.get("target_quantity")
    filled = int(item.get("filled_position", 0))
    item["original_target_quantity"] = target
    if target is not None:
        target = int(target)
    reasons = list(assessment.get("reasons") or [])
    reason_text = ",".join(reasons) or state

    if state == RISK_CAUTION and target is not None and target > filled:
        multiplier = float(assessment.get("new_risk_multiplier", 0.5))
        increment = int((target - filled) * multiplier)
        lot = max(1, int(item.get("lot_increment", 1)))
        increment = increment // lot * lot
        item["target_quantity"] = filled + increment
        item.setdefault(
            "not_executable_reason",
            f"account_risk_caution_manual_review:{reason_text}",
        )
    elif state == RISK_OFF:
        if bool(assessment.get("exit_existing_risk")):
            item["target_quantity"] = 0
        elif target is not None and target > filled:
            # Stale/untrusted data must preserve the BUY target and show it as
            # blocked; replacing it with HOLD would erase the still-valid action.
            item.setdefault(
                "hard_blocked_reason",
                f"account_risk_off_new_risk_blocked:{reason_text}",
            )
        elif target is not None and target < filled:
            # An existing reduction target is preserved, but untrusted market
            # state cannot claim immediate executability.
            item.setdefault(
                "not_executable_reason",
                f"account_risk_off_reduction_wait:{reason_text}",
            )
    item["_risk_metadata"] = {
        "risk_state": state,
        "risk_scope": scope,
        "risk_reasons": reasons,
        "original_target_quantity": item["original_target_quantity"],
    }
    return item
