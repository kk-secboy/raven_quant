"""Two-path transition cost comparison and cost-aware no-trade band.

Design draft 8.1 step 5 and 8.2: before accepting a new candidate target the
portfolio decision layer builds two executable paths — "keep the current
effective target" and "switch to the candidate target" — and prices the full
transition (two-sided commission, stamp duty, transfer fee, slippage and
market impact via :class:`quant_platform.cost_model.CostModelConfig`).

The alpha no-trade band then compares a *conservative* incremental expected
benefit against the *full* transition cost:

- With out-of-sample calibrated expected returns, the incremental benefit is
  the frozen-weight change times those returns, times portfolio value, times a
  conservative haircut. Raw rank scores are never compared against CNY costs:
  they must first pass the explicit, frozen :func:`expected_returns_from_scores`
  mapping (cross-sectional z-score x frozen slope x horizon, hard-capped).
- Without calibrated expected returns, a pre-frozen drift band applies
  (minimum per-instrument weight change, minimum turnover, minimum legal
  order value); no CNY benefit comparison happens at all in that mode.

Hard constraints — cash shortfall, instrument invalidation, permission
tightening, risk reduction/exit — bypass the alpha band (they remain subject
to sellable-quantity and market-tradability limits enforced downstream).

The module is a pure, deterministic layer: same inputs, same decision, and
the full evidence (both path costs, benefit, mode, reasons) is returned for
the snapshot/evidence record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from math import isfinite
from typing import Any

from .cost_model import CostModelConfig, infer_cn_asset_type

POLICY_VERSION = "transition-decision-v1"

DECISION_HOLD = "hold"
DECISION_SWITCH = "switch"
DECISION_FORCED = "forced"

BENEFIT_MODE_CALIBRATED = "calibrated_expected_returns"
BENEFIT_MODE_FROZEN_BAND = "frozen_drift_band"
BENEFIT_MODE_BYPASS = "hard_constraint_bypass"


@dataclass(frozen=True)
class NoTradeBandConfig:
    """Pre-frozen band parameters (design 8.2).

    Cost components (commission, tax, slippage, impact) live in the cost
    model and must not be duplicated here; the band only encodes the frozen
    drift tolerance and the conservative benefit haircut.
    """

    min_weight_change: float = 0.002
    min_turnover: float = 0.005
    min_order_value: float = 1000.0
    benefit_haircut: float = 0.5
    max_expected_return: float = 0.20

    def __post_init__(self) -> None:
        if not 0 < self.min_weight_change <= 1:
            raise ValueError("min_weight_change must be in (0, 1]")
        if not 0 < self.min_turnover <= 1:
            raise ValueError("min_turnover must be in (0, 1]")
        if self.min_order_value < 0:
            raise ValueError("min_order_value must be non-negative")
        if not 0 < self.benefit_haircut <= 1:
            raise ValueError("benefit_haircut must be in (0, 1]")
        if not 0 < self.max_expected_return <= 1:
            raise ValueError("max_expected_return must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionDecision:
    decision: str
    hold_path: dict[str, Any]
    switch_path: dict[str, Any]
    incremental_benefit_cny: float | None
    benefit_mode: str
    reasons: list[str]
    band: dict[str, Any]
    policy_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized_weights(weights: Mapping[str, float] | None, name: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for instrument, value in (weights or {}).items():
        weight = float(value)
        if not isfinite(weight) or weight < 0:
            raise ValueError(f"{name} weights must be finite and non-negative")
        if weight > 0:
            result[str(instrument).upper()] = weight
    return result


def estimate_transition_cost(
    previous_weights: Mapping[str, float] | None,
    target_weights: Mapping[str, float] | None,
    *,
    prices: Mapping[str, float],
    portfolio_value: float,
    cost_model: CostModelConfig | None = None,
    average_daily_values: Mapping[str, float] | None = None,
    trade_date: date | None = None,
) -> dict[str, Any]:
    """Full two-sided CNY cost of moving ``previous_weights`` to ``target_weights``.

    Each changed instrument is one leg priced by the shared cost model
    (commission, stamp duty on stock sells, transfer fee, slippage, market
    impact). Participation is the leg value over the instrument's average
    daily value when that evidence is supplied, else the conservative
    ``max_volume_participation`` assumption.
    """

    model = cost_model or CostModelConfig()
    value = float(portfolio_value)
    if not isfinite(value) or value <= 0:
        raise ValueError("portfolio_value must be positive")
    previous = _normalized_weights(previous_weights, "previous")
    target = _normalized_weights(target_weights, "target")
    instruments = sorted(set(previous) | set(target))
    price_map = {str(key).upper(): float(item) for key, item in prices.items()}
    adv_map = {
        str(key).upper(): float(item) for key, item in (average_daily_values or {}).items()
    }
    legs: list[dict[str, Any]] = []
    total_cny = 0.0
    stock_change = 0.0
    for instrument in instruments:
        price = price_map.get(instrument)
        if price is None or not isfinite(price) or price <= 0:
            raise ValueError(f"prices must cover every traded instrument: {instrument}")
        delta = target.get(instrument, 0.0) - previous.get(instrument, 0.0)
        if abs(delta) <= 1e-12:
            continue
        stock_change += abs(delta)
        gross_value = abs(delta) * value
        side = "buy" if delta > 0 else "sell"
        adv = adv_map.get(instrument)
        participation = (
            gross_value / adv
            if adv is not None and adv > 0
            else model.max_volume_participation
        )
        breakdown = model.estimate_breakdown(
            side=side,
            gross_value=gross_value,
            participation=participation,
            asset_type=infer_cn_asset_type(instrument),
            trade_date=trade_date,
        )
        total_cny += float(breakdown["total"])
        legs.append(
            {
                "instrument": instrument,
                "side": side,
                "weight_change": delta,
                "gross_value": gross_value,
                "participation": participation,
                "cost_cny": float(breakdown["total"]),
                "breakdown": breakdown,
            }
        )
    cash_change = abs(
        (1.0 - sum(target.values())) - (1.0 - sum(previous.values()))
    )
    turnover = 0.5 * (stock_change + cash_change)
    return {
        "total_cny": total_cny,
        "turnover": turnover,
        "legs": legs,
        "portfolio_value": value,
        "cost_version": model.version,
    }


def expected_returns_from_scores(
    scores: Mapping[str, float],
    *,
    score_to_return_slope: float,
    horizon_days: int,
    max_expected_return: float = 0.20,
) -> dict[str, float]:
    """Explicit frozen mapping from frozen rank scores to expected returns.

    The only sanctioned bridge between scores and money: cross-sectional
    z-scores multiplied by a frozen, out-of-sample calibrated slope (expected
    daily return per score standard deviation) and the frozen horizon, hard
    capped at ``max_expected_return`` per leg. Raw scores themselves are never
    a CNY quantity and must not be compared with costs directly.
    """

    slope = float(score_to_return_slope)
    if not isfinite(slope) or slope <= 0:
        raise ValueError("score_to_return_slope must be a positive calibrated constant")
    if int(horizon_days) != horizon_days or int(horizon_days) < 1:
        raise ValueError("horizon_days must be a positive integer")
    cap = float(max_expected_return)
    if not 0 < cap <= 1:
        raise ValueError("max_expected_return must be in (0, 1]")
    values = {str(key).upper(): float(item) for key, item in scores.items()}
    if not values or any(not isfinite(item) for item in values.values()):
        raise ValueError("scores must be finite and non-empty")
    mean = sum(values.values()) / len(values)
    variance = sum((item - mean) ** 2 for item in values.values()) / len(values)
    std = variance**0.5
    if std == 0:
        return {instrument: 0.0 for instrument in values}
    mapped: dict[str, float] = {}
    for instrument in sorted(values):
        zscore = (values[instrument] - mean) / std
        expected = slope * zscore * int(horizon_days)
        mapped[instrument] = max(-cap, min(cap, expected))
    return mapped


def compare_transition_paths(
    previous_weights: Mapping[str, float] | None,
    candidate_weights: Mapping[str, float] | None,
    *,
    prices: Mapping[str, float],
    portfolio_value: float,
    cost_model: CostModelConfig | None = None,
    expected_returns: Mapping[str, float] | None = None,
    band: NoTradeBandConfig | None = None,
    hard_constraints: list[str] | tuple[str, ...] = (),
    average_daily_values: Mapping[str, float] | None = None,
    trade_date: date | None = None,
) -> TransitionDecision:
    """Build both paths, price them, and apply the cost-aware no-trade band."""

    config = band or NoTradeBandConfig()
    switch_path = estimate_transition_cost(
        previous_weights,
        candidate_weights,
        prices=prices,
        portfolio_value=portfolio_value,
        cost_model=cost_model,
        average_daily_values=average_daily_values,
        trade_date=trade_date,
    )
    hold_path = {
        "action": "keep_current_target",
        "total_cny": 0.0,
        "turnover": 0.0,
        "legs": [],
        "portfolio_value": switch_path["portfolio_value"],
        "cost_version": switch_path["cost_version"],
    }
    reasons: list[str] = []
    constraints = [str(item) for item in hard_constraints]
    if constraints:
        reasons.extend(
            f"hard constraint bypasses alpha no-trade band: {item}" for item in constraints
        )
        reasons.append(
            "bypass remains subject to sellable-quantity and market-tradability limits"
        )
        return TransitionDecision(
            decision=DECISION_FORCED,
            hold_path=hold_path,
            switch_path=switch_path,
            incremental_benefit_cny=None,
            benefit_mode=BENEFIT_MODE_BYPASS,
            reasons=reasons,
            band=config.to_dict(),
            policy_version=POLICY_VERSION,
        )
    if not switch_path["legs"]:
        reasons.append("candidate target equals the current target; no trade needed")
        return TransitionDecision(
            decision=DECISION_HOLD,
            hold_path=hold_path,
            switch_path=switch_path,
            incremental_benefit_cny=0.0,
            benefit_mode=BENEFIT_MODE_FROZEN_BAND,
            reasons=reasons,
            band=config.to_dict(),
            policy_version=POLICY_VERSION,
        )

    if expected_returns is not None:
        previous = _normalized_weights(previous_weights, "previous")
        candidate = _normalized_weights(candidate_weights, "candidate")
        returns = {
            str(key).upper(): float(item) for key, item in expected_returns.items()
        }
        for expected in returns.values():
            if not isfinite(expected):
                raise ValueError("expected returns must be finite")
            if abs(expected) > config.max_expected_return:
                raise ValueError(
                    "expected returns exceed the frozen sanity cap; raw rank scores "
                    "are not CNY-mappable returns — map them explicitly via "
                    "expected_returns_from_scores with a frozen slope and horizon"
                )
        raw_benefit = 0.0
        for instrument in sorted(set(previous) | set(candidate)):
            delta = candidate.get(instrument, 0.0) - previous.get(instrument, 0.0)
            raw_benefit += delta * returns.get(instrument, 0.0)
        raw_benefit *= switch_path["portfolio_value"]
        benefit = config.benefit_haircut * raw_benefit
        decision = DECISION_SWITCH if benefit > switch_path["total_cny"] else DECISION_HOLD
        reasons.append(
            "conservative incremental benefit "
            f"{benefit:.2f} CNY (haircut {config.benefit_haircut} of {raw_benefit:.2f}) "
            f"vs full transition cost {switch_path['total_cny']:.2f} CNY"
        )
        reasons.append(
            "switch to candidate target"
            if decision == DECISION_SWITCH
            else "incremental benefit does not cover the full transition cost; keep current target"
        )
        return TransitionDecision(
            decision=decision,
            hold_path=hold_path,
            switch_path=switch_path,
            incremental_benefit_cny=benefit,
            benefit_mode=BENEFIT_MODE_CALIBRATED,
            reasons=reasons,
            band=config.to_dict(),
            policy_version=POLICY_VERSION,
        )

    # Frozen drift band: no CNY benefit comparison is permitted without
    # calibrated expected returns (design 8.2).
    max_change = max(abs(leg["weight_change"]) for leg in switch_path["legs"])
    within_band = (
        switch_path["turnover"] <= config.min_turnover
        and max_change <= config.min_weight_change
    )
    below_min_order = all(
        leg["gross_value"] < config.min_order_value for leg in switch_path["legs"]
    )
    decision = DECISION_HOLD if within_band or below_min_order else DECISION_SWITCH
    reasons.append(
        f"frozen drift band: turnover {switch_path['turnover']:.4f} "
        f"(min {config.min_turnover}), max weight change {max_change:.4f} "
        f"(min {config.min_weight_change}), "
        + (
            "all legs below the minimum legal order value"
            if below_min_order
            else "legs above the minimum order value"
        )
    )
    reasons.append(
        "inside the frozen no-trade band; keep current target"
        if decision == DECISION_HOLD
        else "outside the frozen no-trade band; switch to candidate target"
    )
    return TransitionDecision(
        decision=decision,
        hold_path=hold_path,
        switch_path=switch_path,
        incremental_benefit_cny=None,
        benefit_mode=BENEFIT_MODE_FROZEN_BAND,
        reasons=reasons,
        band=config.to_dict(),
        policy_version=POLICY_VERSION,
    )
