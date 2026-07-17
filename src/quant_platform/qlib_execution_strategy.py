from __future__ import annotations

from math import isfinite
from typing import Any

from .execution_algorithms import build_execution_slices, normalize_execution_policy


def create_qlib_execution_strategy(policy: dict[str, Any]) -> Any:
    """Create the intraday order-slicing strategy without importing Qlib at web startup."""

    normalized = normalize_execution_policy(policy)
    try:
        from qlib.backtest.decision import Order, TradeDecisionWO
        from qlib.strategy.base import BaseStrategy
    except ImportError as exc:  # pragma: no cover - executed in configured Qlib runtime
        raise RuntimeError("the formal backtest runtime does not contain Qlib") from exc

    class QlibExecutionSliceStrategy(BaseStrategy):
        execution_algorithm = normalized["execution_algorithm"]

        def __init__(self) -> None:
            self._parents: dict[str, Any] = {}
            self._filled_amounts: dict[str, float] = {}
            self._cumulative_plans: dict[str, dict[str, int]] = {}
            self._last_slots: dict[str, str] = {}
            self._planned_parent_amount = 0.0
            self._requested_value = 0.0
            self._filled_value = 0.0
            self._scheduled_slice_count = 0
            self._submitted_slice_count = 0
            self._next_bar_submitted = False
            super().__init__()

        def reset(self, *args: Any, **kwargs: Any) -> None:
            super().reset(*args, **kwargs)
            self._parents = {}
            self._filled_amounts = {}
            self._cumulative_plans = {}
            self._last_slots = {}
            self._next_bar_submitted = False
            decision = getattr(self, "outer_trade_decision", None)
            if decision is None:
                return
            for parent in decision.get_decision():
                amount = float(parent.amount)
                integer_amount = int(round(amount))
                if not isfinite(amount) or amount <= 0 or abs(amount - integer_amount) > 1e-6:
                    raise ValueError("minute execution requires positive whole-share parent orders")
                side = "buy" if parent.direction == Order.BUY else "sell"
                instrument = str(parent.stock_id)
                self._parents[instrument] = parent
                self._filled_amounts[instrument] = 0.0
                self._planned_parent_amount += amount
                if self.execution_algorithm == "next_bar":
                    self._scheduled_slice_count += 1
                    continue
                slices = build_execution_slices(
                    quantity=integer_amount,
                    side=side,
                    trade_date=parent.start_time.date(),
                    policy=normalized,
                    instrument=instrument,
                )
                cumulative = 0
                plan: dict[str, int] = {}
                for item in slices:
                    cumulative += int(item["quantity"])
                    slot = str(item["scheduled_for"])[11:16]
                    plan[slot] = cumulative
                if not plan:
                    raise ValueError(f"minute execution produced no slices for {parent.stock_id}")
                self._cumulative_plans[instrument] = plan
                self._last_slots[instrument] = next(reversed(plan))
                self._scheduled_slice_count += len(plan)

        def post_exe_step(self, execute_result: list | None) -> None:
            for order, trade_value, _cost, trade_price in execute_result or []:
                instrument = str(order.stock_id)
                filled = max(0.0, float(order.deal_amount))
                self._filled_amounts[instrument] = (
                    self._filled_amounts.get(instrument, 0.0) + filled
                )
                price = float(trade_price or 0.0)
                if isfinite(price) and price > 0:
                    self._requested_value += max(0.0, float(order.amount)) * price
                    self._filled_value += max(0.0, float(trade_value))

        def generate_trade_decision(self, execute_result: list | None = None) -> Any:
            del execute_result
            trade_step = self.trade_calendar.get_trade_step()
            trade_start, trade_end = self.trade_calendar.get_step_time(trade_step)
            if self.execution_algorithm == "next_bar":
                if self._next_bar_submitted:
                    return TradeDecisionWO([], self)
                orders = []
                for instrument, parent in self._parents.items():
                    amount = float(parent.amount) - self._filled_amounts.get(instrument, 0.0)
                    if amount <= 1e-5:
                        continue
                    orders.append(
                        Order(
                            stock_id=instrument,
                            amount=amount,
                            start_time=trade_start,
                            end_time=trade_end,
                            direction=parent.direction,
                        )
                    )
                    self._submitted_slice_count += 1
                self._next_bar_submitted = True
                return TradeDecisionWO(orders, self)
            slot = trade_start.strftime("%H:%M")
            orders = []
            for instrument, parent in self._parents.items():
                plan = self._cumulative_plans[instrument]
                if slot not in plan:
                    continue
                parent_amount = float(parent.amount)
                filled = self._filled_amounts.get(instrument, 0.0)
                planned = (
                    parent_amount
                    if slot == self._last_slots[instrument]
                    else float(plan[slot])
                )
                amount = min(parent_amount - filled, max(0.0, planned - filled))
                if amount <= 1e-5:
                    continue
                orders.append(
                    Order(
                        stock_id=instrument,
                        amount=amount,
                        start_time=trade_start,
                        end_time=trade_end,
                        direction=parent.direction,
                    )
                )
                self._submitted_slice_count += 1
            return TradeDecisionWO(orders, self)

        def statistics(self) -> dict[str, Any]:
            fill_ratio = (
                min(1.0, self._filled_value / self._requested_value)
                if self._requested_value > 0
                else (0.0 if self._planned_parent_amount > 0 else 1.0)
            )
            return {
                "execution_algorithm": self.execution_algorithm,
                "planned_parent_amount": self._planned_parent_amount,
                "requested_slice_value": self._requested_value,
                "filled_slice_value": self._filled_value,
                "capacity_fill_ratio": fill_ratio,
                "scheduled_slice_count": self._scheduled_slice_count,
                "submitted_slice_count": self._submitted_slice_count,
            }

    return QlibExecutionSliceStrategy()
