from __future__ import annotations

from typing import Any

import pandas as pd

from .portfolio_policy import PortfolioPolicy


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
            super().__init__(*args, **kwargs)
            self._cost_basis: dict[str, float] = {}
            self._amounts: dict[str, float] = {}
            self._take_profit_stages: dict[str, int] = {}
            self._high_water_mark: float | None = None
            self._previous_value: float | None = None
            self._execution_state: dict[str, Any] = {}

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
            del trade_start_time, trade_end_time
            trade_step = self.trade_calendar.get_trade_step()
            signal_start_time, _ = self.trade_calendar.get_step_time(trade_step, shift=1)
            current_weights = {
                str(instrument): float(current.get_stock_weight(instrument))
                for instrument in current.get_stock_list()
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
            peak = max(self._high_water_mark or portfolio_value, portfolio_value)
            drawdown = portfolio_value / peak - 1.0 if peak > 0 else 0.0
            daily_return = (
                portfolio_value / self._previous_value - 1.0
                if self._previous_value and self._previous_value > 0
                else 0.0
            )
            metadata.update(
                {
                    "portfolio_value": portfolio_value,
                    "cost_basis": self._cost_basis,
                    "take_profit_stages": self._take_profit_stages,
                    "execution_state": self._execution_state,
                    "portfolio_drawdown": drawdown,
                    "daily_return": daily_return,
                }
            )
            decision = policy.decide(score, current_weights, **metadata)
            self._take_profit_stages = dict(
                decision.position_state.get("take_profit_stages") or {}
            )
            self._execution_state = dict(decision.position_state.get("execution") or {})
            self._high_water_mark = peak
            self._previous_value = portfolio_value
            return decision.target_weights

    return QlibPortfolioPolicyStrategy(signal=signal)
