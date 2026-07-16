from __future__ import annotations

from typing import Any

import pandas as pd

from .cost_model import infer_cn_asset_type
from .portfolio_policy import PortfolioPolicy, is_rebalance_due


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
            super().__init__(*args, **kwargs)
            self._cost_basis: dict[str, float] = {}
            self._amounts: dict[str, float] = {}
            self._take_profit_stages: dict[str, int] = {}
            self._high_water_mark: float | None = None
            self._day_open_value: float | None = None
            self._value_date: Any = None
            self._execution_state: dict[str, Any] = {}
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
            del trade_end_time
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
                    "cost_basis": self._cost_basis,
                    "take_profit_stages": self._take_profit_stages,
                    "execution_state": self._execution_state,
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
            self._high_water_mark = peak
            trade_date = trade_start_time.date()
            for instrument in list(self._t1_locked):
                current = {
                    locked_date: quantity
                    for locked_date, quantity in self._t1_locked[instrument].items()
                    if locked_date >= trade_date
                }
                if current:
                    self._t1_locked[instrument] = current
                else:
                    self._t1_locked.pop(instrument, None)
            locked_quantities = {
                instrument: sum(by_date.values())
                for instrument, by_date in self._t1_locked.items()
            }
            return apply_t1_target_floor(
                decision.target_weights,
                locked_quantities=locked_quantities,
                current_prices=metadata.get("current_prices", pd.Series(dtype=float)),
                portfolio_value=portfolio_value,
            )

    return QlibPortfolioPolicyStrategy(signal=signal)
