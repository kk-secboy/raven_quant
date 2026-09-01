from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

from .qlib_portfolio_calendar import prove_d_plus_one_signal_execution
from .research_execution_cadence import (
    is_research_decision_step,
    validate_research_execution_cadence_contract,
)


class GovernedDPlusOneTopkDropoutStrategy(TopkDropoutStrategy):
    """Pinned TopK control whose signal-to-execution lag is checked at runtime.

    Qlib's upstream ``TopkDropoutStrategy`` reads the signal interval with
    ``get_step_time(trade_step, shift=1)`` and trades on ``trade_step``.  The
    platform pins that implementation, but this subclass additionally proves
    the relation on every decision so an upstream semantic change fails closed.
    """

    SIGNAL_LAG_SESSIONS = 1

    def __init__(
        self,
        *args: Any,
        research_execution_cadence: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        cadence = validate_research_execution_cadence_contract(
            research_execution_cadence
        )
        if int(cadence["signal_lag_sessions"]) != self.SIGNAL_LAG_SESSIONS:
            raise ValueError("research strategy signal lag differs from its cadence")
        self.research_execution_cadence = cadence
        self.decision_interval_sessions = int(
            cadence["decision_interval_sessions"]
        )
        super().__init__(*args, **kwargs)

    def generate_trade_decision(self, execute_result: Any = None) -> Any:
        trade_step = self.trade_calendar.get_trade_step()
        if not is_research_decision_step(
            trade_step,
            decision_interval_sessions=self.decision_interval_sessions,
        ):
            from qlib.backtest.decision import TradeDecisionWO

            return TradeDecisionWO([], self)
        execution_start_time, _ = self.trade_calendar.get_step_time(trade_step)
        _, signal_end_time = self.trade_calendar.get_step_time(
            trade_step,
            shift=self.SIGNAL_LAG_SESSIONS,
        )
        prove_d_plus_one_signal_execution(
            signal_end_time=signal_end_time,
            execution_start_time=execution_start_time,
            signal_lag_sessions=self.SIGNAL_LAG_SESSIONS,
        )
        return super().generate_trade_decision(execute_result)
