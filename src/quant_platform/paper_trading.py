from __future__ import annotations

from datetime import datetime
from math import floor
from typing import Any

import pandas as pd


def _risk_event(
    rule: str,
    *,
    severity: str = "warning",
    event_type: str = "pre_trade_control",
    observed: float | None = None,
    limit: float | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "rule": rule,
        "severity": severity,
        "event_type": event_type,
        "observed": observed,
        "limit_value": limit,
        "status": "open",
        "details": details or {},
    }


def build_rebalance_plan(
    scores: pd.Series,
    market: pd.DataFrame,
    current_positions: list[dict[str, Any]],
    *,
    nav: float,
    cash: float,
    high_water_mark: float | None = None,
    portfolio_status: str = "active",
    topk: int,
    n_drop: int,
    max_position_weight: float,
    max_daily_turnover: float,
    max_daily_loss: float = 0.03,
    stop_loss: float = 0.07,
    take_profit_partial: float = 0.12,
    take_profit_partial_fraction: float = 0.50,
    take_profit: float = 0.20,
    max_drawdown_reduce: float = 0.10,
    max_drawdown_liquidate: float = 0.15,
    drawdown_reduction_exposure: float = 0.50,
    max_industry_weight: float = 0.30,
    min_average_daily_amount: float = 500_000_000,
    max_volume_participation: float = 0.01,
    open_cost: float,
    close_cost: float,
    slippage: float,
    lot_size: int = 100,
    fill_time: datetime,
) -> dict[str, Any]:
    """Create a deterministic, long-only, next-open paper execution plan.

    ``amount`` and ``average_amount`` are CNY notionals. Industry values must be
    point-in-time classifications for the execution date.
    """
    if nav <= 0 or cash < 0:
        raise ValueError("NAV must be positive and cash must not be negative")
    if topk < 1 or n_drop < 0 or n_drop > topk:
        raise ValueError("topk and n_drop are invalid")
    limits = (
        lot_size,
        max_position_weight,
        max_daily_turnover,
        max_daily_loss,
        stop_loss,
        take_profit_partial,
        take_profit_partial_fraction,
        take_profit,
        max_drawdown_reduce,
        max_drawdown_liquidate,
        drawdown_reduction_exposure,
        max_industry_weight,
        min_average_daily_amount,
        max_volume_participation,
    )
    if any(value <= 0 for value in limits):
        raise ValueError("execution and risk limits must be positive")
    if take_profit_partial >= take_profit:
        raise ValueError("partial take-profit threshold must be below full take-profit")
    if max_drawdown_reduce >= max_drawdown_liquidate:
        raise ValueError("drawdown reduction threshold must be below liquidation threshold")
    if take_profit_partial_fraction >= 1 or drawdown_reduction_exposure >= 1:
        raise ValueError("reduction fractions must be below one")
    score = pd.to_numeric(scores, errors="coerce").dropna().sort_values(ascending=False)
    if score.empty:
        raise ValueError("no factor scores are available for the signal date")
    required = {
        "open",
        "close",
        "paused",
        "volume",
        "amount",
        "average_amount",
        "industry",
        "up_limit",
        "down_limit",
    }
    if not required.issubset(market.columns):
        raise ValueError(f"market data is missing fields: {sorted(required - set(market.columns))}")

    current_items = {
        str(item["instrument"]): item for item in current_positions if float(item["quantity"]) > 0
    }
    current = {key: float(value["quantity"]) for key, value in current_items.items()}
    risk_events: list[dict[str, Any]] = []
    selection_rejections: dict[str, str] = {}

    opening_value = cash
    for instrument, quantity in current.items():
        row = market.loc[instrument] if instrument in market.index else None
        mark = (
            float(row["open"])
            if row is not None and pd.notna(row["open"]) and float(row["open"]) > 0
            else float(current_items[instrument].get("market_price") or 0)
        )
        opening_value += quantity * mark
    opening_return = opening_value / nav - 1.0
    reference_high = max(float(high_water_mark or nav), opening_value)
    opening_drawdown = opening_value / reference_high - 1.0
    daily_loss_breached = opening_return < -max_daily_loss
    liquidation_required = (
        portfolio_status == "liquidation_pending" or opening_drawdown <= -max_drawdown_liquidate
    )
    reduction_required = not liquidation_required and (
        portfolio_status == "risk_reduction_pending" or opening_drawdown <= -max_drawdown_reduce
    )
    risk_action = "liquidate" if liquidation_required else "reduce" if reduction_required else None
    if liquidation_required:
        risk_events.append(
            _risk_event(
                "max_drawdown_liquidate",
                severity="critical",
                event_type="circuit_breaker",
                observed=abs(opening_drawdown),
                limit=max_drawdown_liquidate,
                details={"action": "liquidate_at_next_open", "opening_drawdown": opening_drawdown},
            )
        )
    elif reduction_required:
        risk_events.append(
            _risk_event(
                "max_drawdown_reduce",
                severity="critical",
                event_type="circuit_breaker",
                observed=abs(opening_drawdown),
                limit=max_drawdown_reduce,
                details={
                    "action": "reduce_exposure_then_pause",
                    "target_exposure": drawdown_reduction_exposure,
                    "opening_drawdown": opening_drawdown,
                },
            )
        )
    if daily_loss_breached:
        if risk_action is None:
            risk_action = "pause"
        risk_events.append(
            _risk_event(
                "max_daily_loss",
                severity="critical",
                event_type="circuit_breaker",
                observed=abs(opening_return),
                limit=max_daily_loss,
                details={
                    "action": "portfolio_paused_no_new_buys",
                    "opening_return": opening_return,
                },
            )
        )

    target_weight = min(1.0 / topk, max_position_weight)
    if reduction_required:
        target_weight *= drawdown_reduction_exposure
    buffer = set(score.head(topk + n_drop).index.astype(str))
    ranked = [item for item in current if item in buffer]
    ranked.extend(item for item in score.index.astype(str) if item not in ranked)
    ranked = ranked[: max(topk + n_drop, topk * 5)]
    selected: list[str] = []
    industry_weights: dict[str, float] = {}
    unknown_existing: set[str] = set()
    for instrument in ranked:
        if len(selected) >= topk:
            break
        row = market.loc[instrument] if instrument in market.index else None
        industry = "" if row is None or pd.isna(row["industry"]) else str(row["industry"]).strip()
        if not industry:
            if instrument in current:
                unknown_existing.add(instrument)
            else:
                selection_rejections[instrument] = "missing point-in-time industry classification"
            continue
        next_weight = industry_weights.get(industry, 0.0) + target_weight
        if next_weight > max_industry_weight + 1e-12:
            if instrument not in current:
                selection_rejections[instrument] = "industry concentration limit"
            continue
        selected.append(instrument)
        industry_weights[industry] = next_weight
    if unknown_existing and not liquidation_required:
        risk_events.append(
            _risk_event(
                "industry_data_missing",
                severity="critical",
                observed=float(len(unknown_existing)),
                details={
                    "action": "portfolio_paused_existing_positions_held",
                    "instruments": sorted(unknown_existing),
                },
            )
        )
    targets = {} if liquidation_required else {instrument: target_weight for instrument in selected}

    orders: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    forced_exit_reasons: dict[str, str] = {}
    forced_target_quantities: dict[str, float] = {}
    take_profit_stages: dict[str, int] = {}
    for instrument, item in current_items.items():
        row = market.loc[instrument] if instrument in market.index else None
        if row is None or pd.isna(row["open"]) or float(row["open"]) <= 0:
            continue
        avg_cost = float(item.get("avg_cost") or 0)
        if avg_cost <= 0:
            continue
        position_return = float(row["open"]) / avg_cost - 1.0
        if liquidation_required:
            forced_exit_reasons[instrument] = "max_drawdown_liquidation"
            forced_target_quantities[instrument] = 0.0
        elif position_return <= -stop_loss:
            forced_exit_reasons[instrument] = "stop_loss"
            forced_target_quantities[instrument] = 0.0
            risk_events.append(
                _risk_event(
                    "stop_loss",
                    event_type="risk_exit",
                    observed=abs(position_return),
                    limit=stop_loss,
                    details={"instrument": instrument, "position_return": position_return},
                )
            )
        elif position_return >= take_profit:
            forced_exit_reasons[instrument] = "take_profit"
            forced_target_quantities[instrument] = 0.0
            risk_events.append(
                _risk_event(
                    "take_profit",
                    event_type="risk_exit",
                    observed=position_return,
                    limit=take_profit,
                    details={"instrument": instrument, "position_return": position_return},
                )
            )
        elif position_return >= take_profit_partial and int(item.get("take_profit_stage") or 0) < 1:
            remaining = (
                floor((float(item["quantity"]) * (1.0 - take_profit_partial_fraction)) / lot_size)
                * lot_size
            )
            forced_exit_reasons[instrument] = "take_profit_partial"
            forced_target_quantities[instrument] = float(remaining)
            take_profit_stages[instrument] = 1
            risk_events.append(
                _risk_event(
                    "take_profit_partial",
                    event_type="risk_exit",
                    observed=position_return,
                    limit=take_profit_partial,
                    details={
                        "instrument": instrument,
                        "position_return": position_return,
                        "sell_fraction": take_profit_partial_fraction,
                    },
                )
            )
        elif position_return >= take_profit_partial:
            # Do not let the ranking target immediately buy back a staged take-profit.
            forced_target_quantities[instrument] = float(item["quantity"])

    universe = set(current) | set(targets)
    for instrument in sorted(universe):
        row = market.loc[instrument] if instrument in market.index else None
        tradable = (
            row is not None
            and pd.notna(row["open"])
            and float(row["open"]) > 0
            and float(row["paused"]) < 0.5
            and float(row["volume"]) > 0
        )
        target = targets.get(instrument, 0.0)
        quantity = current.get(instrument, 0.0)
        if instrument in unknown_existing and instrument not in forced_exit_reasons:
            target_quantity = quantity
            target = quantity * float(current_items[instrument].get("market_price") or 0) / nav
        elif row is not None and pd.notna(row["open"]) and float(row["open"]) > 0:
            target_quantity = floor((target * nav / float(row["open"])) / lot_size) * lot_size
        else:
            target_quantity = quantity
        if instrument in forced_target_quantities:
            target_quantity = min(target_quantity, forced_target_quantities[instrument])
            target = target_quantity * float(row["open"]) / nav if row is not None else 0.0
        delta = float(target_quantity) - quantity
        if abs(delta) < 1e-9:
            continue
        side = "buy" if delta > 0 else "sell"
        reason = forced_exit_reasons.get(instrument)
        if (daily_loss_breached or liquidation_required) and side == "buy":
            orders.append(
                {
                    "instrument": instrument,
                    "side": side,
                    "order_type": "market",
                    "target_weight": target,
                    "quantity": abs(delta),
                    "reason": "daily loss circuit breaker",
                }
            )
            continue
        if not tradable:
            orders.append(
                {
                    "instrument": instrument,
                    "side": side,
                    "order_type": "market",
                    "target_weight": target,
                    "quantity": abs(delta),
                    "reason": "missing price, zero volume, or suspended",
                }
            )
            continue
        if pd.isna(row["up_limit"]) or pd.isna(row["down_limit"]):
            orders.append(
                {
                    "instrument": instrument,
                    "side": side,
                    "order_type": "market",
                    "target_weight": target,
                    "quantity": abs(delta),
                    "reason": "missing A-share price-limit data",
                }
            )
            risk_events.append(
                _risk_event(
                    "price_limit_data_missing",
                    severity="critical",
                    observed=None,
                    details={"instrument": instrument, "side": side, "action": "order_rejected"},
                )
            )
            if risk_action is None:
                risk_action = "pause"
            continue
        open_price = float(row["open"])
        at_up_limit = open_price >= float(row["up_limit"]) * (1.0 - 1e-6)
        at_down_limit = open_price <= float(row["down_limit"]) * (1.0 + 1e-6)
        if (side == "buy" and at_up_limit) or (side == "sell" and at_down_limit):
            rule = "limit_up_buy_blocked" if side == "buy" else "limit_down_sell_blocked"
            orders.append(
                {
                    "instrument": instrument,
                    "side": side,
                    "order_type": "market",
                    "target_weight": target,
                    "quantity": abs(delta),
                    "reason": rule,
                }
            )
            risk_events.append(
                _risk_event(
                    rule,
                    event_type="execution_block",
                    observed=open_price,
                    limit=float(row["up_limit"] if side == "buy" else row["down_limit"]),
                    details={
                        "instrument": instrument,
                        "side": side,
                        "forced_reason": reason,
                        "action": "retry_next_batch" if side == "sell" else "order_rejected",
                    },
                )
            )
            continue
        if side == "buy" and float(row["average_amount"]) < min_average_daily_amount:
            orders.append(
                {
                    "instrument": instrument,
                    "side": side,
                    "order_type": "market",
                    "target_weight": target,
                    "quantity": abs(delta),
                    "reason": "minimum average daily amount",
                }
            )
            risk_events.append(
                _risk_event(
                    "min_average_daily_amount",
                    observed=float(row["average_amount"]),
                    limit=min_average_daily_amount,
                    details={"instrument": instrument, "action": "buy_rejected"},
                )
            )
            continue
        candidates.append(
            {
                "instrument": instrument,
                "side": side,
                "order_type": "market",
                "target_weight": target,
                "quantity": abs(delta),
                "open": open_price,
                "market_amount": max(0.0, float(row["amount"])),
                "reason": reason,
            }
        )

    for instrument, reason in selection_rejections.items():
        if instrument in universe:
            continue
        row = market.loc[instrument] if instrument in market.index else None
        if row is None or pd.isna(row["open"]) or float(row["open"]) <= 0:
            continue
        quantity = floor((target_weight * nav / float(row["open"])) / lot_size) * lot_size
        if quantity <= 0:
            continue
        orders.append(
            {
                "instrument": instrument,
                "side": "buy",
                "order_type": "market",
                "target_weight": target_weight,
                "quantity": float(quantity),
                "reason": reason,
            }
        )
        risk_events.append(
            _risk_event(
                "max_industry_weight" if "concentration" in reason else "industry_data_missing",
                severity="warning" if "concentration" in reason else "critical",
                observed=None,
                limit=max_industry_weight if "concentration" in reason else None,
                details={"instrument": instrument, "action": "buy_rejected", "reason": reason},
            )
        )

    turnover_budget = nav * max_daily_turnover
    used = 0.0
    fills: list[dict[str, Any]] = []
    available_cash = cash
    for item in sorted(candidates, key=lambda value: 0 if value["side"] == "sell" else 1):
        budget_left = max(0.0, turnover_budget - used)
        participation_budget = item["market_amount"] * max_volume_participation
        executable_budget = min(budget_left, participation_budget)
        max_quantity = floor((executable_budget / item["open"]) / lot_size) * lot_size
        quantity = min(float(item["quantity"]), float(max_quantity))
        if item["side"] == "sell" and quantity < item["quantity"] and item["target_weight"] == 0:
            odd_remainder = item["quantity"] - quantity
            if (
                0 < odd_remainder < lot_size
                and (quantity + odd_remainder) * item["open"] <= executable_budget
            ):
                quantity += odd_remainder
        if item["side"] == "buy":
            fill_price = item["open"] * (1.0 + slippage)
            affordable = (
                floor((available_cash / (fill_price * (1.0 + open_cost))) / lot_size) * lot_size
            )
            quantity = min(quantity, float(affordable))
            fee_rate = open_cost
        else:
            fill_price = item["open"] * (1.0 - slippage)
            fee_rate = close_cost
        order = {key: value for key, value in item.items() if key not in {"open", "market_amount"}}
        order["quantity"] = float(item["quantity"])
        if quantity <= 0:
            order["reason"] = "turnover, cash, or market participation limit"
            orders.append(order)
            continue
        if quantity + 1e-9 < float(item["quantity"]):
            risk_events.append(
                _risk_event(
                    "max_volume_participation",
                    observed=quantity * item["open"] / item["market_amount"]
                    if item["market_amount"] > 0
                    else None,
                    limit=max_volume_participation,
                    details={
                        "instrument": item["instrument"],
                        "requested_quantity": item["quantity"],
                        "filled_quantity": quantity,
                    },
                )
            )
        gross = quantity * fill_price
        fee = gross * fee_rate
        if item["side"] == "buy":
            available_cash -= gross + fee
        else:
            available_cash += gross - fee
        used += quantity * item["open"]
        orders.append(order)
        fills.append(
            {
                "instrument": item["instrument"],
                "quantity": quantity,
                "price": fill_price,
                "fee": fee,
                "slippage": slippage,
                "fill_time": fill_time.isoformat(),
            }
        )
    closing_prices = {
        str(instrument): float(row["close"])
        for instrument, row in market.iterrows()
        if pd.notna(row["close"]) and float(row["close"]) > 0
    }
    industries = {
        str(instrument): str(row["industry"]).strip()
        for instrument, row in market.iterrows()
        if pd.notna(row["industry"]) and str(row["industry"]).strip()
    }
    return {
        "orders": orders,
        "fills": fills,
        "closing_prices": closing_prices,
        "industries": industries,
        "take_profit_stages": take_profit_stages,
        "risk_events": risk_events,
        "risk_action": risk_action,
        "opening_return": opening_return,
        "opening_drawdown": opening_drawdown,
        "estimated_turnover": used / nav,
    }
