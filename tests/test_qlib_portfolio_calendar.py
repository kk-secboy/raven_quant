from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from quant_platform.qlib_portfolio_calendar import (
    prove_d_plus_one_signal_execution,
    resolve_qlib_portfolio_calendar_boundary,
)
from quant_platform.research_execution_cadence import (
    build_research_execution_cadence_contract,
)
from quant_platform.research_horizon import SWING_1_6M

pytestmark = pytest.mark.no_database


def _write_calendars(
    root: Path,
    *,
    market: str,
    future: str | None = None,
) -> Path:
    calendars = root / "calendars"
    calendars.mkdir(parents=True)
    (calendars / "day.txt").write_text(market, encoding="utf-8")
    if future is not None:
        (calendars / "day_future.txt").write_text(future, encoding="utf-8")
    return root


def test_pre_final_view_uses_boundary_without_future_market_data(tmp_path: Path) -> None:
    provider = _write_calendars(
        tmp_path / "view",
        market="2024-01-02\n2024-01-03\n",
        future="2024-01-02\n2024-01-03\n2024-01-04\n",
    )

    evidence = resolve_qlib_portfolio_calendar_boundary(
        provider,
        backtest_end="2024-01-03",
    )

    assert evidence["interval_end"] == "2024-01-04"
    assert evidence["boundary_source"] == "calendars/day_future.txt"
    assert evidence["market_data_calendar_end"] == "2024-01-03"
    assert evidence["interval_end_has_market_data"] is False


def test_full_provider_may_use_next_market_data_session_as_boundary(tmp_path: Path) -> None:
    provider = _write_calendars(
        tmp_path / "provider",
        market="2024-01-02\n2024-01-03\n2024-01-04\n",
    )

    evidence = resolve_qlib_portfolio_calendar_boundary(
        provider,
        backtest_end="2024-01-03",
    )

    assert evidence["interval_end"] == "2024-01-04"
    assert evidence["boundary_source"] == "calendars/day.txt"
    assert evidence["interval_end_has_market_data"] is True


def test_portfolio_boundary_fails_closed_for_missing_or_changed_calendar(
    tmp_path: Path,
) -> None:
    missing = _write_calendars(
        tmp_path / "missing",
        market="2024-01-02\n2024-01-03\n",
    )
    with pytest.raises(ValueError, match="no later calendar interval boundary"):
        resolve_qlib_portfolio_calendar_boundary(
            missing,
            backtest_end="2024-01-03",
        )

    changed = _write_calendars(
        tmp_path / "changed",
        market="2024-01-02\n2024-01-03\n",
        future="2024-01-02\n2024-01-04\n",
    )
    with pytest.raises(ValueError, match="changed the market-data calendar prefix"):
        resolve_qlib_portfolio_calendar_boundary(
            changed,
            backtest_end="2024-01-03",
        )


def test_signal_execution_contract_requires_previous_completed_session() -> None:
    evidence = prove_d_plus_one_signal_execution(
        signal_end_time=pd.Timestamp("2024-01-03 23:59:59.999999"),
        execution_start_time=pd.Timestamp("2024-01-04"),
        signal_lag_sessions=1,
    )
    assert evidence["signal_lag_sessions"] == 1
    assert evidence["no_same_day_or_future_signal"] is True

    with pytest.raises(ValueError, match="same-day information"):
        prove_d_plus_one_signal_execution(
            signal_end_time=pd.Timestamp("2024-01-04"),
            execution_start_time=pd.Timestamp("2024-01-04"),
            signal_lag_sessions=1,
        )
    with pytest.raises(ValueError, match="exactly one session"):
        prove_d_plus_one_signal_execution(
            signal_end_time=pd.Timestamp("2024-01-02"),
            execution_start_time=pd.Timestamp("2024-01-04"),
            signal_lag_sessions=2,
        )


def test_governed_strategy_explicitly_requests_previous_session_signal() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "quant_platform"
        / "qlib_research_strategy.py"
    ).read_text(encoding="utf-8")

    assert "SIGNAL_LAG_SESSIONS = 1" in source
    assert "shift=self.SIGNAL_LAG_SESSIONS" in source
    assert "is_research_decision_step(" in source
    assert "return TradeDecisionWO([], self)" in source
    assert "prove_d_plus_one_signal_execution(" in source
    assert "return super().generate_trade_decision(execute_result)" in source


def test_governed_strategy_holds_between_due_steps_and_keeps_d_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDecision:
        def __init__(self, orders: list[Any], strategy: Any) -> None:
            self.orders = orders
            self.strategy = strategy

    class FakeTopkDropoutStrategy:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.parent_calls = 0

        def generate_trade_decision(self, execute_result: Any = None) -> Any:
            self.parent_calls += 1
            return ("parent", execute_result)

    signal_strategy = ModuleType("qlib.contrib.strategy.signal_strategy")
    signal_strategy.TopkDropoutStrategy = FakeTopkDropoutStrategy
    decision = ModuleType("qlib.backtest.decision")
    decision.TradeDecisionWO = FakeDecision
    for name, module in {
        "qlib": ModuleType("qlib"),
        "qlib.contrib": ModuleType("qlib.contrib"),
        "qlib.contrib.strategy": ModuleType("qlib.contrib.strategy"),
        "qlib.contrib.strategy.signal_strategy": signal_strategy,
        "qlib.backtest": ModuleType("qlib.backtest"),
        "qlib.backtest.decision": decision,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "quant_platform"
        / "qlib_research_strategy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "quant_platform._cadence_strategy_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strategy = module.GovernedDPlusOneTopkDropoutStrategy(
        research_execution_cadence=(
            build_research_execution_cadence_contract(SWING_1_6M)
        )
    )

    class Calendar:
        step = 1

        def get_trade_step(self) -> int:
            return self.step

        def get_step_time(
            self, trade_step: int, shift: int = 0
        ) -> tuple[pd.Timestamp, pd.Timestamp]:
            del trade_step
            if shift == 1:
                signal = pd.Timestamp("2024-01-03 23:59:59")
                return signal, signal
            execution = pd.Timestamp("2024-01-04")
            return execution, execution

    calendar = Calendar()
    strategy.trade_calendar = calendar

    skipped = strategy.generate_trade_decision("skipped")
    assert skipped.orders == []
    assert strategy.parent_calls == 0

    calendar.step = 5
    assert strategy.generate_trade_decision("due") == ("parent", "due")
    assert strategy.parent_calls == 1
