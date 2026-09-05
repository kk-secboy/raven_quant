from __future__ import annotations

import math
import random
from typing import Any

import pandas as pd

from .cost_model import infer_cn_asset_type
from .discrete_constraints import ExecutionLotContext
from .market_rules import order_unit_rules
from .portfolio_policy import PortfolioPolicy, is_rebalance_due


def qlib_execution_lot_context(
    current: Any,
    *,
    instruments: pd.Index,
    factors: pd.Series,
    prices: pd.Series,
    current_prices: pd.Series,
    trade_date: Any,
    locked_amounts: dict[str, float],
) -> ExecutionLotContext:
    """Bridge Qlib adjusted amounts to physical shares using signal-time evidence.

    A position's stored price marks the same NAV as its weight.  Reconstructing
    its shares using the signal open (or the next execution price) is invalid.
    New entries use the known signal close; the execution reference remains the
    known signal open/VWAP and does not inspect tomorrow's quote.
    """
    index = instruments.astype(str)
    if index.has_duplicates or factors.index.astype(str).has_duplicates:
        raise ValueError("Qlib quantity evidence has duplicate instruments")
    factor = pd.to_numeric(factors, errors="coerce").reindex(index)
    marks = pd.to_numeric(current_prices, errors="coerce").reindex(index).copy()
    execution = pd.to_numeric(prices, errors="coerce").reindex(index)
    quantities = pd.Series(0.0, index=index)
    held = {str(key) for key in current.get_stock_list()}
    if not held.issubset(set(index)):
        raise ValueError("Qlib quantity evidence omits an actual holding")
    required = ((execution > 0) & (execution < float("inf"))) | index.isin(held)
    invalid_factor = ~(factor.gt(0) & factor.lt(float("inf")))
    if (required & invalid_factor).any():
        instrument = str(index[required & invalid_factor][0])
        raise ValueError(f"Qlib position has no valid signal factor for {instrument}")
    for instrument in held:
        unit_factor = float(factor[instrument])
        amount = float(current.get_stock_amount(instrument))
        mark = float(current.get_stock_price(instrument))
        if not math.isfinite(amount) or amount < 0 or not math.isfinite(mark) or mark <= 0:
            raise ValueError(f"Qlib position has invalid amount or valuation for {instrument}")
        quantities[instrument] = amount * unit_factor
        marks[instrument] = mark / unit_factor
    # Missing/unlisted candidates remain blocked by the existing gate, without
    # asking for a not-yet-effective board rule. Batch the cross-section fields
    # rather than performing multiple pandas scalar writes for every candidate.
    rules = {instrument: order_unit_rules(instrument, trade_date)
             for instrument in index[required]}
    minimum = pd.Series({key: rule.min_lot for key, rule in rules.items()}, dtype=float)
    increment = pd.Series({key: rule.lot_increment for key, rule in rules.items()}, dtype=float)
    locked = pd.Series(locked_amounts, dtype=float).reindex(index, fill_value=0.0)
    locked = (locked * factor).where(locked > 0, 0.0)
    return ExecutionLotContext(
        previous_quantities=quantities,
        valuation_prices=marks,
        execution_prices=execution,
        buy_minimum=minimum.reindex(index, fill_value=100.0),
        buy_increment=increment.reindex(index, fill_value=100.0),
        sell_increment=pd.Series(1.0, index=index),
        locked_quantities=locked,
        # The pinned Exchange already supports exact full-position disposal
        # without rounding a dividend-adjusted share-equivalent remainder.
        allow_full_liquidation=True,
    )


class _AuditedQuantityOrderGenerator:
    """Submit the policy's audited quantities without a second weight conversion."""

    def __init__(self) -> None:
        self.plan: dict[str, Any] | None = None

    def generate_order_list_from_target_weight_position(
        self, current: Any, trade_exchange: Any, target_weight_position: dict[str, float],
        risk_degree: float, pred_start_time: Any, pred_end_time: Any,
        trade_start_time: Any, trade_end_time: Any,
    ) -> list[Any]:
        from qlib.backtest.decision import Order

        del pred_start_time, pred_end_time
        plan, self.plan = self.plan, None
        if plan is None:
            raise ValueError("Qlib order generation has no audited quantity plan")
        current_amounts = {str(key): float(value)
                           for key, value in current.get_stock_amount_dict().items()}
        if (
            risk_degree != 1.0
            or target_weight_position != plan["weights"]
            or current_amounts != plan["current_amounts"]
            or float(current.calculate_value()) != plan["portfolio_value"]
            or (trade_start_time, trade_end_time) != plan["trade_window"]
        ):
            raise ValueError("Qlib order generation differs from the audited quantity plan")
        # Preserve the upstream deterministic submission order and sell-first
        # funding.  Board lot/minimum and actual volume/cash clipping happen
        # once in SquareRootImpactExchange, not in Qlib's flat-100 pre-round.
        instruments = sorted(set(current_amounts) | set(plan["target_amounts"]))
        random.Random(0).shuffle(instruments)
        buys, sells = [], []
        for instrument in instruments:
            delta = plan["target_amounts"].get(instrument, 0.0) - current_amounts.get(
                instrument, 0.0
            )
            if delta == 0 or not trade_exchange.is_stock_tradable(
                stock_id=instrument, start_time=trade_start_time, end_time=trade_end_time
            ):
                continue
            order = Order(
                stock_id=instrument, amount=abs(delta),
                direction=Order.BUY if delta > 0 else Order.SELL,
                start_time=trade_start_time, end_time=trade_end_time,
            )
            (buys if delta > 0 else sells).append(order)
        return sells + buys


def apply_t1_target_floor(
    target_weights: dict[str, float],
    *,
    locked_quantities: dict[str, float],
    current_prices: pd.Series,
    portfolio_value: float,
) -> dict[str, float]:
    """Keep stock bought today in the target until the next trading day."""

    if portfolio_value <= 0:
        raise ValueError("portfolio value must be positive for T+1 enforcement")
    result = {str(key): max(0.0, float(value)) for key, value in target_weights.items()}
    prices = pd.to_numeric(current_prices, errors="coerce")
    prices.index = prices.index.astype(str)
    for instrument, raw_quantity in locked_quantities.items():
        quantity = max(0.0, float(raw_quantity))
        if quantity <= 0 or infer_cn_asset_type(instrument) != "stock":
            continue
        price = float(prices.get(instrument, float("nan")))
        if not pd.notna(price) or price <= 0:
            raise ValueError(f"T+1 enforcement has no current price for {instrument}")
        floor_weight = quantity * price / portfolio_value
        result[instrument] = max(result.get(instrument, 0.0), floor_weight)
    return result


def create_qlib_policy_strategy(
    *, signal: pd.Series | pd.DataFrame, policy: PortfolioPolicy, metadata_provider: Any = None
) -> Any:
    """Create the only promotable Qlib strategy without importing Qlib at web startup."""

    try:
        from qlib.contrib.strategy.signal_strategy import WeightStrategyBase
    except ImportError as exc:  # pragma: no cover - Qlib runs in the configured WSL runtime
        raise RuntimeError("the formal backtest runtime does not contain Qlib") from exc

    class QlibPortfolioPolicyStrategy(WeightStrategyBase):
        policy_version = policy.version

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # PortfolioPolicy weights already include cash/risk limits against
            # the complete NAV.  Qlib's default 0.95 would rescale them again.
            super().__init__(*args, risk_degree=1.0, **kwargs)
            self.order_generator = _AuditedQuantityOrderGenerator()
            self._cost_basis: dict[str, float] = {}
            self._amounts: dict[str, float] = {}
            self._take_profit_stages: dict[str, int] = {}
            self._high_water_mark: float | None = None
            self._day_open_value: float | None = None
            self._value_date: Any = None
            self._execution_state: dict[str, Any] = {}
            self._holding_age_sessions: dict[str, int] = {}
            self._t1_locked: dict[str, dict[Any, float]] = {}
            self._last_rebalance_signal_time: Any = None

        def generate_trade_decision(self, execute_result: Any = None) -> Any:
            from qlib.backtest.decision import Order

            for order, trade_value, _cost, trade_price in execute_result or []:
                instrument = str(order.stock_id)
                amount = float(trade_value) / float(trade_price) if trade_price else 0.0
                if order.direction == Order.BUY and amount > 0:
                    old_amount = self._amounts.get(instrument, 0.0)
                    old_basis = self._cost_basis.get(instrument, float(trade_price))
                    new_amount = old_amount + amount
                    self._cost_basis[instrument] = (
                        old_amount * old_basis + amount * float(trade_price)
                    ) / new_amount
                    self._amounts[instrument] = new_amount
                    if infer_cn_asset_type(instrument) == "stock":
                        trade_date = order.start_time.date()
                        by_date = self._t1_locked.setdefault(instrument, {})
                        by_date[trade_date] = by_date.get(trade_date, 0.0) + amount
                elif amount > 0:
                    remaining = max(0.0, self._amounts.get(instrument, amount) - amount)
                    if remaining <= 1e-10:
                        self._amounts.pop(instrument, None)
                        self._cost_basis.pop(instrument, None)
                        self._take_profit_stages.pop(instrument, None)
                    else:
                        self._amounts[instrument] = remaining
            return super().generate_trade_decision(execute_result)

        def generate_target_weight_position(
            self,
            score: pd.Series,
            current: Any,
            trade_start_time: Any,
            trade_end_time: Any,
        ) -> dict[str, float]:
            self.order_generator.plan = None
            trade_step = self.trade_calendar.get_trade_step()
            signal_start_time, _ = self.trade_calendar.get_step_time(trade_step, shift=1)
            current_weights = {
                str(instrument): float(current.get_stock_weight(instrument))
                for instrument in current.get_stock_list()
            }
            current_instruments = {
                instrument for instrument, weight in current_weights.items() if weight > 0
            }
            # The policy emits intended target state, while Qlib reports the
            # actually filled portfolio on the next decision.  A sell can be
            # rejected by a suspension, price limit, lot rule, or liquidity
            # control, so an intended exit must not erase the holding age of a
            # position that is still present in the executor account.  Prune
            # completed exits here, then retain rejected/partial exits below.
            self._holding_age_sessions = {
                instrument: age
                for instrument, age in self._holding_age_sessions.items()
                if instrument in current_instruments
            }
            metadata = (
                metadata_provider(
                    signal_start_time,
                    score.index.union(pd.Index(current_weights, dtype=str)),
                )
                if metadata_provider is not None
                else {}
            )
            portfolio_value = float(current.calculate_value())
            trade_date = trade_start_time.date()
            self._t1_locked = {
                instrument: {
                    day: quantity for day, quantity in by_date.items() if day >= trade_date
                }
                for instrument, by_date in self._t1_locked.items()
                if any(day >= trade_date for day in by_date)
            }
            locked_amounts = {
                instrument: sum(by_date.values())
                for instrument, by_date in self._t1_locked.items()
            }
            factors = metadata.pop("qlib_factors", None)
            lot_context = None
            policy_cost_basis = self._cost_basis
            if factors is not None:
                lot_context = qlib_execution_lot_context(
                    current,
                    instruments=score.index.union(pd.Index(current_weights, dtype=str)),
                    factors=factors,
                    prices=metadata["prices"],
                    current_prices=metadata["current_prices"],
                    trade_date=trade_date,
                    locked_amounts=locked_amounts,
                )
                metadata["execution_lot_context"] = lot_context
                # Fill bookkeeping remains in Qlib's adjusted price/amount
                # space.  Risk comparisons use today's matching raw units.
                policy_cost_basis = {
                    instrument: basis / float(factors[instrument])
                    for instrument, basis in self._cost_basis.items()
                    if instrument in current_weights
                }
            peak = max(self._high_water_mark or portfolio_value, portfolio_value)
            drawdown = portfolio_value / peak - 1.0 if peak > 0 else 0.0
            decision_date = trade_start_time.date()
            if self._value_date != decision_date:
                self._value_date = decision_date
                self._day_open_value = portfolio_value
            daily_return = (
                portfolio_value / self._day_open_value - 1.0
                if self._day_open_value and self._day_open_value > 0
                else 0.0
            )
            metadata.update(
                {
                    "portfolio_value": portfolio_value,
                    "cost_basis": policy_cost_basis,
                    "take_profit_stages": self._take_profit_stages,
                    "execution_state": self._execution_state,
                    "holding_age_sessions": self._holding_age_sessions,
                    "portfolio_drawdown": drawdown,
                    "daily_return": daily_return,
                    "rebalance_due": is_rebalance_due(
                        signal_start_time,
                        self._last_rebalance_signal_time,
                        policy.config.rebalance_frequency,
                    ),
                }
            )
            decision = policy.decide(score, current_weights, **metadata)
            if metadata["rebalance_due"]:
                self._last_rebalance_signal_time = signal_start_time
            self._take_profit_stages = dict(
                decision.position_state.get("take_profit_stages") or {}
            )
            self._execution_state = dict(decision.position_state.get("execution") or {})
            next_holding_age_sessions = {
                str(key): int(value)
                for key, value in (
                    decision.position_state.get("holding_age_sessions") or {}
                ).items()
            }
            for instrument in current_instruments - set(next_holding_age_sessions):
                # ``policy.decide`` already required complete age evidence for
                # every actual holding.  Increment the age just as it does for
                # retained targets so a failed exit remains governed on the
                # following trading session instead of becoming untracked.
                next_holding_age_sessions[instrument] = (
                    int(self._holding_age_sessions[instrument]) + 1
                )
            self._holding_age_sessions = next_holding_age_sessions
            self._high_water_mark = peak
            if lot_context is not None:
                current_amounts = {
                    str(key): float(value)
                    for key, value in current.get_stock_amount_dict().items()
                }
                targets = decision.position_state["execution_lot_target_quantities"]
                target_amounts = {}
                for instrument, quantity in targets.items():
                    if float(quantity) == 0.0:
                        # Qlib's existing full-liquidation exception must
                        # receive exactly zero, including adjusted positions
                        # whose physical equivalent contains a dividend tail.
                        continue
                    original_quantity = float(lot_context.previous_quantities[instrument])
                    delta = float(quantity) - original_quantity
                    # Exact HOLD must preserve the original internal amount,
                    # including corporate-action/floating residuals.
                    amount = current_amounts.get(instrument, 0.0)
                    if delta != 0:
                        amount += delta / float(factors[instrument])
                    if amount > 0:
                        target_amounts[instrument] = amount
                self.order_generator.plan = {
                    "weights": dict(decision.target_weights),
                    "current_amounts": current_amounts,
                    "target_amounts": target_amounts,
                    "portfolio_value": portfolio_value,
                    "trade_window": (trade_start_time, trade_end_time),
                }
                return dict(decision.target_weights)
            return apply_t1_target_floor(
                decision.target_weights,
                locked_quantities=locked_amounts,
                current_prices=metadata.get("current_prices", pd.Series(dtype=float)),
                portfolio_value=portfolio_value,
            )

    return QlibPortfolioPolicyStrategy(signal=signal)
