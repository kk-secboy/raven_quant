from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time
from math import floor, isfinite
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .cost_model import CostModelConfig, infer_cn_asset_type
from .execution_algorithms import build_execution_slices, normalize_execution_policy

SIMULATION_ENGINE_VERSION = "ashare-minute-simulation-v1"
_SHANGHAI = ZoneInfo("Asia/Shanghai")


def execute_simulation_day(
    *,
    trade_date: date,
    cash: float,
    prior_nav: float,
    high_water_mark: float,
    positions: dict[str, dict[str, Any]],
    target_weights: dict[str, float],
    minute_bars: pd.DataFrame,
    closing_prices: dict[str, dict[str, Any]],
    cost_model: CostModelConfig,
    execution_policy: dict[str, Any],
) -> dict[str, Any]:
    """Execute one close-to-next-day A-share/ETF rebalance with an auditable ledger."""

    if not isfinite(cash) or cash < 0 or prior_nav <= 0 or high_water_mark <= 0:
        raise ValueError("simulation account balances are invalid")
    policy = normalize_execution_policy(execution_policy)
    targets = {str(key): float(value) for key, value in target_weights.items()}
    if any(not isfinite(value) or value < 0 for value in targets.values()):
        raise ValueError("simulation target weights must be finite and non-negative")
    if sum(targets.values()) > 1.0 + 1e-8:
        raise ValueError("simulation target weights exceed one")
    bars = _normalize_bars(minute_bars, trade_date)
    state = deepcopy(positions)
    for instrument, position in state.items():
        quantity = int(position.get("quantity", 0))
        available = int(position.get("available_quantity", 0))
        last_trade = position.get("last_trade_date")
        if quantity < 0 or available < 0 or available > quantity:
            raise ValueError(f"invalid position state for {instrument}")
        if last_trade is None or _as_date(last_trade) < trade_date:
            position["available_quantity"] = quantity

    reference_prices = _execution_reference_prices(bars)
    instruments = sorted(set(state) | set(targets))
    desired: dict[str, int] = {}
    for instrument in instruments:
        price = reference_prices.get(instrument)
        if price is None:
            desired[instrument] = int(state.get(instrument, {}).get("quantity", 0))
            continue
        desired[instrument] = (
            floor(targets.get(instrument, 0.0) * prior_nav / price / 100.0) * 100
        )

    order_specs: list[dict[str, Any]] = []
    for instrument in instruments:
        current = int(state.get(instrument, {}).get("quantity", 0))
        delta = desired[instrument] - current
        if delta == 0:
            continue
        side = "buy" if delta > 0 else "sell"
        requested = abs(delta)
        if side == "sell":
            available = int(state[instrument].get("available_quantity", 0))
            requested = min(requested, available)
            if desired[instrument] > 0:
                requested = requested // 100 * 100
        if requested <= 0:
            order_specs.append(
                _rejected_order(
                    instrument,
                    side,
                    targets.get(instrument, 0.0),
                    abs(delta),
                    "t_plus_one_unavailable",
                    reference_prices.get(instrument, 0.0),
                )
            )
            continue
        order_specs.append(
            {
                "instrument": instrument,
                "side": side,
                "target_weight": targets.get(instrument, 0.0),
                "requested_quantity": requested,
                "requested_value": requested * reference_prices.get(instrument, 0.0),
            }
        )

    # A-share sell proceeds are available for buys on the same trading day.
    order_specs.sort(key=lambda item: 0 if item["side"] == "sell" else 1)
    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    cash_flows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    starting_cash = cash
    for spec in order_specs:
        if "status" in spec:
            orders.append(spec)
            events.append(_rejection_event(spec))
            continue
        instrument = spec["instrument"]
        side = spec["side"]
        requested = int(spec["requested_quantity"])
        instrument_bars = bars[bars["instrument"] == instrument].set_index("datetime")
        if instrument_bars.empty:
            order = {**spec, **_empty_execution("missing_minute_bars")}
            orders.append(order)
            events.append(_rejection_event(order))
            continue
        slices = build_execution_slices(
            quantity=requested,
            side=side,
            trade_date=trade_date,
            policy=policy,
        )
        remaining = requested
        order_fills: list[dict[str, Any]] = []
        rejection_reasons: list[str] = []
        for execution_slice in slices:
            if remaining <= 0:
                break
            scheduled = pd.Timestamp(execution_slice["scheduled_for"])
            if scheduled.tzinfo is not None:
                scheduled = scheduled.tz_convert(_SHANGHAI).tz_localize(None)
            else:
                scheduled = scheduled.tz_localize(None)
            if scheduled not in instrument_bars.index:
                rejection_reasons.append("missing_minute_bar")
                continue
            bar = instrument_bars.loc[scheduled]
            if isinstance(bar, pd.DataFrame):
                raise ValueError("minute execution bars contain duplicate timestamps")
            reason = _bar_rejection_reason(bar, side)
            if reason:
                rejection_reasons.append(reason)
                continue
            price = float(bar["vwap"])
            minute_volume = int(floor(float(bar["volume"])))
            capacity = int(floor(minute_volume * float(policy["max_participation"])))
            slice_request = min(remaining, int(execution_slice["quantity"]))
            fill_quantity = min(slice_request, capacity)
            if side == "buy":
                fill_quantity = fill_quantity // 100 * 100
                fill_quantity = _affordable_buy_quantity(
                    fill_quantity,
                    cash=cash,
                    price=price,
                    participation=(fill_quantity / minute_volume if minute_volume else 0.0),
                    asset_type=infer_cn_asset_type(instrument),
                    trade_date=trade_date,
                    costs=cost_model,
                )
            if side == "sell":
                fill_quantity = min(
                    fill_quantity,
                    int(state.get(instrument, {}).get("available_quantity", 0)),
                )
            if fill_quantity <= 0:
                rejection_reasons.append(
                    "insufficient_cash" if side == "buy" and cash <= price * 100 else "capacity"
                )
                continue
            participation = fill_quantity / minute_volume
            breakdown = cost_model.estimate_breakdown(
                side=side,
                gross_value=fill_quantity * price,
                participation=participation,
                asset_type=infer_cn_asset_type(instrument),
                trade_date=trade_date,
            )
            gross = fill_quantity * price
            fee = float(breakdown["total"])
            cash_delta = -(gross + fee) if side == "buy" else gross - fee
            if cash + cash_delta < -1e-6:
                raise RuntimeError("simulation execution would create negative cash")
            cash += cash_delta
            fill = {
                "instrument": instrument,
                "side": side,
                "executed_at": scheduled.to_pydatetime().replace(tzinfo=_SHANGHAI),
                "quantity": fill_quantity,
                "price": price,
                "gross_value": gross,
                "fee": fee,
                "cost_breakdown": breakdown,
                "minute_volume": minute_volume,
                "capacity_quantity": capacity,
            }
            order_fills.append(fill)
            fills.append(fill)
            cash_flows.append(
                {
                    "trade_date": trade_date,
                    "flow_type": "buy_settlement" if side == "buy" else "sell_settlement",
                    "amount": cash_delta,
                    "balance_after": cash,
                }
            )
            _apply_fill(state, fill, trade_date)
            remaining -= fill_quantity
        filled = requested - remaining
        status = (
            "filled"
            if remaining == 0
            else ("partial_filled_expired" if filled else "rejected")
        )
        reject_reason = None
        if remaining:
            reject_reason = ",".join(sorted(set(rejection_reasons or ["capacity"])))
        order = {
            **spec,
            "filled_quantity": filled,
            "filled_value": sum(item["gross_value"] for item in order_fills),
            "capacity_fill_ratio": filled / requested,
            "status": status,
            "reject_reason": reject_reason,
            "expires_at": datetime.combine(trade_date, time(15, 0), _SHANGHAI),
        }
        orders.append(order)
        if remaining:
            events.append(_rejection_event(order))

    if cash < -1e-6:
        raise RuntimeError("simulation ledger cash conservation failed")
    if abs(cash - (starting_cash + sum(item["amount"] for item in cash_flows))) > 1e-6:
        raise RuntimeError("simulation ledger cash flows do not reconcile")
    valuation = _value_positions(state, closing_prices, trade_date)
    events.extend(valuation["events"])
    nav = cash + valuation["market_value"]
    new_peak = max(high_water_mark, nav)
    has_stale = valuation["has_stale_prices"]
    nav_row = {
        "trade_date": trade_date,
        "cash": cash,
        "market_value": valuation["market_value"],
        "nav": nav,
        "daily_return": nav / prior_nav - 1.0,
        "drawdown": nav / new_peak - 1.0,
        "market_date": valuation["market_date"],
        "has_stale_prices": has_stale,
        "status": "degraded" if has_stale else "healthy",
        "performance_certified": not has_stale,
    }
    return {
        "engine_version": SIMULATION_ENGINE_VERSION,
        "trade_date": trade_date,
        "cash": cash,
        "nav": nav,
        "high_water_mark": new_peak,
        "positions": state,
        "orders": orders,
        "fills": fills,
        "cash_flows": cash_flows,
        "nav_row": nav_row,
        "events": events,
        "conservation": {
            "cash_difference": cash
            - (starting_cash + sum(item["amount"] for item in cash_flows)),
            "negative_positions": sum(
                1 for item in state.values() if int(item.get("quantity", 0)) < 0
            ),
        },
    }


def _normalize_bars(values: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    required = {
        "datetime",
        "instrument",
        "close",
        "vwap",
        "volume",
        "paused",
        "up_limit",
        "down_limit",
    }
    if not required.issubset(values.columns):
        raise ValueError("minute bars are missing execution fields")
    result = values.loc[:, sorted(required)].copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    if result["datetime"].dt.tz is not None:
        result["datetime"] = result["datetime"].dt.tz_convert(_SHANGHAI).dt.tz_localize(None)
    result["instrument"] = result["instrument"].astype(str).str.upper()
    for column in ("close", "vwap", "volume", "paused", "up_limit", "down_limit"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result[result["datetime"].dt.date == trade_date]
    if result.duplicated(["datetime", "instrument"]).any():
        raise ValueError("minute bars contain duplicate instrument timestamps")
    if result[["datetime", "instrument", "close", "volume"]].isna().any().any():
        raise ValueError("minute bars contain invalid values")
    return result.sort_values(["instrument", "datetime"])


def _execution_reference_prices(bars: pd.DataFrame) -> dict[str, float]:
    if bars.empty:
        return {}
    first = bars.groupby("instrument", sort=False).first()
    return {str(index): float(row["close"]) for index, row in first.iterrows()}


def _bar_rejection_reason(bar: pd.Series, side: str) -> str | None:
    if float(bar["paused"]) > 0 or float(bar["volume"]) <= 0:
        return "suspended"
    price = float(bar["close"])
    if not isfinite(price) or price <= 0:
        return "invalid_price"
    if side == "buy" and isfinite(float(bar["up_limit"])) and price >= float(bar["up_limit"]):
        return "limit_up"
    if side == "sell" and isfinite(float(bar["down_limit"])) and price <= float(bar["down_limit"]):
        return "limit_down"
    return None


def _affordable_buy_quantity(
    quantity: int,
    *,
    cash: float,
    price: float,
    participation: float,
    asset_type: str,
    trade_date: date,
    costs: CostModelConfig,
) -> int:
    result = quantity // 100 * 100
    while result > 0:
        gross = result * price
        fee = costs.estimate(
            side="buy",
            gross_value=gross,
            participation=participation,
            asset_type=asset_type,
            trade_date=trade_date,
        )
        if gross + fee <= cash + 1e-9:
            return result
        result -= 100
    return 0


def _apply_fill(state: dict[str, dict[str, Any]], fill: dict[str, Any], trade_date: date) -> None:
    instrument = fill["instrument"]
    position = state.setdefault(
        instrument,
        {"quantity": 0, "available_quantity": 0, "average_cost": 0.0},
    )
    quantity = int(position["quantity"])
    filled = int(fill["quantity"])
    if fill["side"] == "buy":
        total_cost = quantity * float(position.get("average_cost", 0.0))
        total_cost += float(fill["gross_value"]) + float(fill["fee"])
        position["quantity"] = quantity + filled
        position["average_cost"] = total_cost / position["quantity"]
        # New buys stay unavailable until the next trading day.
    else:
        available = int(position.get("available_quantity", 0))
        if filled > available or filled > quantity:
            raise RuntimeError("simulation fill creates a short position")
        position["quantity"] = quantity - filled
        position["available_quantity"] = available - filled
        if position["quantity"] == 0:
            position["average_cost"] = 0.0
    position["last_trade_date"] = trade_date


def _value_positions(
    state: dict[str, dict[str, Any]],
    closing_prices: dict[str, dict[str, Any]],
    trade_date: date,
) -> dict[str, Any]:
    market_value = 0.0
    market_dates: list[date] = []
    events: list[dict[str, Any]] = []
    stale = False
    for instrument in list(state):
        position = state[instrument]
        quantity = int(position.get("quantity", 0))
        if quantity == 0:
            del state[instrument]
            continue
        quote = closing_prices.get(instrument)
        if quote and bool(quote.get("delisted")) and not bool(quote.get("cash_liquidated")):
            price = 0.0
            quote_date = trade_date
            is_stale = False
            events.append(
                {
                    "severity": "critical",
                    "event_type": "delisted_zero_valuation",
                    "instrument": instrument,
                    "reason": "delisted_without_cash_liquidation",
                    "details": {"quantity": quantity},
                }
            )
        elif quote:
            price = float(quote["price"])
            quote_date = _as_date(quote["market_date"])
            is_stale = quote_date != trade_date
        else:
            previous_price = position.get("market_price")
            previous_date = position.get("market_date")
            if previous_price is None or previous_date is None:
                price = 0.0
                quote_date = None
            else:
                price = float(previous_price)
                quote_date = _as_date(previous_date)
            is_stale = True
        if not isfinite(price) or price < 0:
            raise ValueError(f"invalid closing valuation for {instrument}")
        value = quantity * price
        position.update(
            {
                "market_price": price,
                "market_date": quote_date,
                "stale": is_stale,
                "market_value": value,
            }
        )
        market_value += value
        if quote_date is not None:
            market_dates.append(quote_date)
        if is_stale:
            stale = True
            events.append(
                {
                    "severity": "warning",
                    "event_type": "stale_valuation",
                    "instrument": instrument,
                    "reason": "closing_price_not_current",
                    "details": {"market_date": quote_date.isoformat() if quote_date else None},
                }
            )
    return {
        "market_value": market_value,
        "market_date": min(market_dates) if market_dates else None,
        "has_stale_prices": stale,
        "events": events,
    }


def _empty_execution(reason: str) -> dict[str, Any]:
    return {
        "filled_quantity": 0,
        "filled_value": 0.0,
        "capacity_fill_ratio": 0.0,
        "status": "rejected",
        "reject_reason": reason,
        "expires_at": None,
    }


def _rejected_order(
    instrument: str,
    side: str,
    target_weight: float,
    requested: int,
    reason: str,
    price: float,
) -> dict[str, Any]:
    return {
        "instrument": instrument,
        "side": side,
        "target_weight": target_weight,
        "requested_quantity": requested,
        "requested_value": requested * price,
        **_empty_execution(reason),
    }


def _rejection_event(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "severity": "warning",
        "event_type": "order_rejected" if order["status"] == "rejected" else "order_expired",
        "instrument": order["instrument"],
        "reason": order.get("reject_reason") or "unfilled",
        "details": {
            "requested_quantity": order["requested_quantity"],
            "filled_quantity": order.get("filled_quantity", 0),
        },
    }


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()
