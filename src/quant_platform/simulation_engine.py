from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time
from math import floor, isfinite
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .cost_model import CostModelConfig, CostScheduleBook, infer_cn_asset_type
from .execution_algorithms import build_execution_slices, normalize_execution_policy

SIMULATION_ENGINE_VERSION = "ashare-minute-simulation-v2"
_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _resolve_cost_schedule(
    cost_model: CostModelConfig | None,
    cost_schedule: CostScheduleBook | None,
) -> CostScheduleBook:
    if (cost_model is None) == (cost_schedule is None):
        raise ValueError("exactly one of cost_model or cost_schedule is required")
    return cost_schedule or CostScheduleBook.from_versions([cost_model])


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
    cost_model: CostModelConfig | None = None,
    cost_schedule: CostScheduleBook | None = None,
    execution_policy: dict[str, Any],
    signal_at: datetime | None = None,
) -> dict[str, Any]:
    """Execute one next-eligible-bar A-share/ETF rebalance with an auditable ledger."""

    if not isfinite(cash) or cash < 0 or prior_nav <= 0 or high_water_mark <= 0:
        raise ValueError("simulation account balances are invalid")
    cost_model = _resolve_cost_schedule(cost_model, cost_schedule).as_of(trade_date)
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
            signal_at=signal_at,
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


def execute_atomic_pair_day(
    *,
    trade_date: date,
    cash: float,
    prior_nav: float,
    high_water_mark: float,
    positions: dict[str, dict[str, Any]],
    target_payload: dict[str, Any],
    minute_bars: pd.DataFrame,
    closing_prices: dict[str, dict[str, Any]],
    shortability: dict[str, bool],
    cost_model: CostModelConfig | None = None,
    cost_schedule: CostScheduleBook | None = None,
    execution_policy: dict[str, Any],
) -> dict[str, Any]:
    """Execute a governed pair target as one all-filled or all-rejected atomic group."""

    if not isfinite(cash) or cash < 0 or prior_nav <= 0 or high_water_mark <= 0:
        raise ValueError("simulation account balances are invalid")
    cost_model = _resolve_cost_schedule(cost_model, cost_schedule).as_of(trade_date)
    policy = normalize_execution_policy(execution_policy)
    group_id = str(target_payload.get("atomic_group_id") or "").strip()
    legs = [dict(item) for item in (target_payload.get("legs") or [])]
    if not group_id or len(legs) != 2:
        raise ValueError("pair execution requires exactly one governed atomic group")
    if {int(item.get("leg_no") or 0) for item in legs} != {1, 2}:
        raise ValueError("pair execution leg numbers must be 1 and 2")
    if {str(item.get("position_side")) for item in legs} != {"long", "short"}:
        raise ValueError("pair execution requires one long and one short leg")
    state = deepcopy(positions)
    bars = _normalize_bars(minute_bars, trade_date)
    specs: list[dict[str, Any]] = []
    rejection: str | None = None
    for leg in sorted(legs, key=lambda item: int(item["leg_no"])):
        instrument = str(leg["instrument"]).upper()
        position_side = str(leg["position_side"])
        target = int(leg["target_quantity"])
        if target < 0 or target % 100:
            raise ValueError("pair target quantities must be non-negative board lots")
        current_row = state.get(instrument) or {}
        current = int(current_row.get("quantity", 0))
        current_side = str(current_row.get("position_side") or position_side)
        if current and current_side != position_side:
            raise ValueError("pair target cannot reverse an existing leg in one batch")
        if (
            position_side == "long"
            and current
            and (
                current_row.get("last_trade_date") is None
                or _as_date(current_row["last_trade_date"]) < trade_date
            )
        ):
            current_row["available_quantity"] = current
        delta = target - current
        side = (
            ("buy" if delta > 0 else "sell")
            if position_side == "long"
            else ("sell_short" if delta > 0 else "buy_to_cover")
        )
        specs.append(
            {
                "instrument": instrument,
                "side": side,
                "cost_side": "buy" if side in {"buy", "buy_to_cover"} else "sell",
                "atomic_group_id": group_id,
                "leg_no": int(leg["leg_no"]),
                "position_side": position_side,
                "annual_borrow_rate": float(leg.get("annual_borrow_rate") or 0.0),
                "target_quantity": target,
                "starting_quantity": current,
                "target_weight": 0.0,
                "requested_quantity": abs(delta),
            }
        )
    active = [item for item in specs if item["requested_quantity"] > 0]
    if len(active) == 1:
        rejection = "unbalanced_pair_target"
    short_leg = next(item for item in specs if item["position_side"] == "short")
    starting_cash = cash
    cash_flows: list[dict[str, Any]] = []
    carry_borrow_cost = 0.0
    starting_short = int(short_leg["starting_quantity"])
    if starting_short:
        if not 0 < short_leg["annual_borrow_rate"] <= 1:
            raise ValueError("an existing short leg requires a governed borrow rate")
        reference = _execution_reference_prices(bars).get(short_leg["instrument"])
        closing_quote = closing_prices.get(short_leg["instrument"])
        if reference is None and closing_quote is not None:
            market_date = closing_quote.get("market_date")
            closing_date = _as_date(market_date) if market_date is not None else None
            closing_value = float(closing_quote.get("price") or 0.0)
            if closing_date == trade_date and isfinite(closing_value) and closing_value > 0:
                reference = closing_value
        if reference is None or not isfinite(float(reference)) or float(reference) <= 0:
            raise ValueError("existing pair short borrow carry cannot be priced for the trade date")
        carry_borrow_cost = (
            starting_short
            * float(reference)
            * float(short_leg["annual_borrow_rate"])
            / 252.0
        )
        if cash < carry_borrow_cost:
            raise RuntimeError("pair borrow cost would create negative cash")
        cash -= carry_borrow_cost
        state[short_leg["instrument"]]["borrow_cost"] = float(
            state[short_leg["instrument"]].get("borrow_cost", 0.0)
        ) + carry_borrow_cost
        cash_flows.append(
            {
                "trade_date": trade_date,
                "flow_type": "pair_borrow_carry",
                "amount": -carry_borrow_cost,
                "balance_after": cash,
            }
        )
    if (
        short_leg["target_quantity"] > 0
        and shortability.get(short_leg["instrument"]) is not True
    ):
        rejection = "short_borrow_not_authorized"
    if short_leg["target_quantity"] > 0 and not 0 < short_leg["annual_borrow_rate"] <= 1:
        rejection = "borrow_cost_not_governed"
    for spec in active:
        if spec["side"] == "sell":
            available = int(state.get(spec["instrument"], {}).get("available_quantity", 0))
            if available < spec["requested_quantity"]:
                rejection = "t_plus_one_unavailable"

    execution_rows: dict[str, pd.Series] = {}
    executed_at: pd.Timestamp | None = None
    if active and rejection is None:
        leg_frames: dict[str, pd.DataFrame] = {}
        for spec in active:
            frame = bars[bars["instrument"] == spec["instrument"]].set_index("datetime")
            leg_frames[spec["instrument"]] = frame
        common = set.intersection(*(set(frame.index) for frame in leg_frames.values()))
        allowed = sorted(
            value
            for value in common
            if time(10, 0) <= value.time() <= time(11, 20)
            or time(13, 30) <= value.time() <= time(14, 50)
        )
        if not allowed:
            rejection = "missing_common_execution_bar"
        else:
            executed_at = pd.Timestamp(allowed[0])
            for spec in active:
                row = leg_frames[spec["instrument"]].loc[executed_at]
                if isinstance(row, pd.DataFrame):
                    raise ValueError("pair minute bars contain duplicate timestamps")
                reason = _bar_rejection_reason(row, spec["cost_side"])
                if reason:
                    rejection = reason
                    break
                capacity = int(
                    floor(float(row["volume"]) * float(policy["max_participation"]))
                )
                if capacity < spec["requested_quantity"]:
                    rejection = "atomic_capacity"
                    break
                execution_rows[spec["instrument"]] = row

    prepared: list[tuple[dict[str, Any], dict[str, Any], float, float]] = []
    cash_delta_total = 0.0
    if active and rejection is None and executed_at is not None:
        for spec in active:
            row = execution_rows[spec["instrument"]]
            quantity = int(spec["requested_quantity"])
            price = float(row["vwap"])
            minute_volume = int(floor(float(row["volume"])))
            participation = quantity / minute_volume
            costs = cost_model
            borrow_days = 0
            if spec["side"] == "sell_short":
                costs = CostModelConfig(
                    **{
                        **cost_model.to_dict(),
                        "annual_borrow_rate": spec["annual_borrow_rate"],
                    }
                )
                borrow_days = 1
            breakdown = costs.estimate_breakdown(
                side=spec["cost_side"],
                gross_value=quantity * price,
                participation=participation,
                asset_type=infer_cn_asset_type(spec["instrument"]),
                trade_date=trade_date,
                borrow_days=borrow_days,
            )
            gross = quantity * price
            fee = float(breakdown["total"])
            delta = -(gross + fee) if spec["cost_side"] == "buy" else gross - fee
            cash_delta_total += delta
            prepared.append((spec, breakdown, gross, delta))
        if cash + cash_delta_total < -1e-6:
            rejection = "insufficient_cash"

    if rejection is not None:
        orders = [
            {
                **spec,
                "requested_value": float(spec["requested_quantity"])
                * float(_execution_reference_prices(bars).get(spec["instrument"], 0.0)),
                "filled_quantity": 0,
                "filled_value": 0.0,
                "capacity_fill_ratio": 0.0,
                "status": "rejected",
                "reject_reason": rejection,
                "expires_at": datetime.combine(trade_date, time(15, 0), _SHANGHAI),
                "borrow_cost": 0.0,
            }
            for spec in specs
        ]
        events = [
            {
                "severity": "critical",
                "event_type": "atomic_pair_rejected",
                "instrument": None,
                "reason": rejection,
                "details": {"atomic_group_id": group_id},
            }
        ]
        return _pair_result(
            trade_date=trade_date,
            starting_cash=starting_cash,
            cash=cash,
            prior_nav=prior_nav,
            high_water_mark=high_water_mark,
            state=state,
            closing_prices=closing_prices,
            orders=orders,
            fills=[],
            cash_flows=cash_flows,
            events=events,
            certified=False,
        )

    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    if not active:
        return _pair_result(
            trade_date=trade_date,
            starting_cash=starting_cash,
            cash=cash,
            prior_nav=prior_nav,
            high_water_mark=high_water_mark,
            state=state,
            closing_prices=closing_prices,
            orders=[],
            fills=[],
            cash_flows=cash_flows,
            events=[],
            certified=True,
        )
    assert executed_at is not None
    for spec, breakdown, gross, cash_delta in prepared:
        quantity = int(spec["requested_quantity"])
        row = execution_rows[spec["instrument"]]
        price = float(row["vwap"])
        borrow_cost = float(breakdown["borrow_cost"])
        fill = {
            **{
                key: spec[key]
                for key in (
                    "instrument",
                    "side",
                    "atomic_group_id",
                    "leg_no",
                    "position_side",
                )
            },
            "executed_at": executed_at.to_pydatetime().replace(tzinfo=_SHANGHAI),
            "quantity": quantity,
            "price": price,
            "gross_value": gross,
            "fee": float(breakdown["total"]),
            "borrow_cost": borrow_cost,
            "cost_breakdown": breakdown,
            "minute_volume": int(floor(float(row["volume"]))),
            "capacity_quantity": int(
                floor(float(row["volume"]) * float(policy["max_participation"]))
            ),
        }
        fills.append(fill)
        orders.append(
            {
                **spec,
                "requested_value": gross,
                "filled_quantity": quantity,
                "filled_value": gross,
                "capacity_fill_ratio": 1.0,
                "status": "filled",
                "reject_reason": None,
                "expires_at": datetime.combine(trade_date, time(15, 0), _SHANGHAI),
                "borrow_cost": borrow_cost,
            }
        )
        cash += cash_delta
        cash_flows.append(
            {
                "trade_date": trade_date,
                "flow_type": f"pair_{spec['side']}",
                "amount": cash_delta,
                "balance_after": cash,
            }
        )
        _apply_pair_fill(state, fill, trade_date)
    return _pair_result(
        trade_date=trade_date,
        starting_cash=starting_cash,
        cash=cash,
        prior_nav=prior_nav,
        high_water_mark=high_water_mark,
        state=state,
        closing_prices=closing_prices,
        orders=orders,
        fills=fills,
        cash_flows=cash_flows,
        events=[],
        certified=True,
    )


def _apply_pair_fill(
    state: dict[str, dict[str, Any]], fill: dict[str, Any], trade_date: date
) -> None:
    instrument = str(fill["instrument"])
    side = str(fill["side"])
    filled = int(fill["quantity"])
    position = state.setdefault(
        instrument,
        {
            "quantity": 0,
            "available_quantity": 0,
            "average_cost": 0.0,
            "position_side": fill["position_side"],
            "borrow_cost": 0.0,
        },
    )
    quantity = int(position.get("quantity", 0))
    if side in {"buy", "sell_short"}:
        prior_cost = quantity * float(position.get("average_cost", 0.0))
        position["quantity"] = quantity + filled
        position["average_cost"] = (prior_cost + float(fill["gross_value"])) / position[
            "quantity"
        ]
        if side == "sell_short":
            position["available_quantity"] = position["quantity"]
    else:
        if filled > quantity:
            raise RuntimeError("pair close quantity exceeds its governed leg")
        position["quantity"] = quantity - filled
        if side == "sell":
            available = int(position.get("available_quantity", 0))
            if filled > available:
                raise RuntimeError("pair long leg violates T+1")
            position["available_quantity"] = available - filled
        else:
            position["available_quantity"] = position["quantity"]
        if position["quantity"] == 0:
            position["average_cost"] = 0.0
    position.update(
        atomic_group_id=fill["atomic_group_id"],
        leg_no=int(fill["leg_no"]),
        position_side=fill["position_side"],
        borrow_cost=float(position.get("borrow_cost", 0.0))
        + float(fill.get("borrow_cost", 0.0)),
        last_trade_date=trade_date,
    )


def _pair_result(
    *,
    trade_date: date,
    starting_cash: float,
    cash: float,
    prior_nav: float,
    high_water_mark: float,
    state: dict[str, dict[str, Any]],
    closing_prices: dict[str, dict[str, Any]],
    orders: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    cash_flows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    certified: bool,
) -> dict[str, Any]:
    market_value = 0.0
    market_dates: list[date] = []
    stale = False
    for instrument in list(state):
        position = state[instrument]
        quantity = int(position.get("quantity", 0))
        if quantity == 0:
            del state[instrument]
            continue
        quote = closing_prices.get(instrument)
        if quote:
            price = float(quote["price"])
            market_date = _as_date(quote["market_date"])
            is_stale = market_date != trade_date
        else:
            price = float(position.get("market_price") or 0.0)
            previous_date = position.get("market_date")
            market_date = _as_date(previous_date) if previous_date else None
            is_stale = True
        sign = -1.0 if position.get("position_side") == "short" else 1.0
        value = sign * quantity * price
        position.update(
            market_price=price,
            market_date=market_date,
            stale=is_stale,
            market_value=value,
        )
        market_value += value
        stale = stale or is_stale
        if market_date:
            market_dates.append(market_date)
    nav = cash + market_value
    peak = max(high_water_mark, nav)
    certified = certified and not stale
    cash_difference = cash - (starting_cash + sum(item["amount"] for item in cash_flows))
    if abs(cash_difference) < 1e-9:
        cash_difference = 0.0
    return {
        "engine_version": SIMULATION_ENGINE_VERSION,
        "trade_date": trade_date,
        "cash": cash,
        "nav": nav,
        "high_water_mark": peak,
        "positions": state,
        "orders": orders,
        "fills": fills,
        "cash_flows": cash_flows,
        "nav_row": {
            "trade_date": trade_date,
            "cash": cash,
            "market_value": market_value,
            "nav": nav,
            "daily_return": nav / prior_nav - 1.0,
            "drawdown": nav / peak - 1.0,
            "market_date": min(market_dates) if market_dates else None,
            "has_stale_prices": stale,
            "status": "healthy" if certified else "degraded",
            "performance_certified": certified,
        },
        "events": events,
        "conservation": {
            "cash_difference": cash_difference,
            "negative_positions": 0,
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
