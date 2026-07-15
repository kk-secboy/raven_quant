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
            metadata["portfolio_value"] = float(current.calculate_value())
            return policy.decide(score, current_weights, **metadata).target_weights

    return QlibPortfolioPolicyStrategy(signal=signal)
