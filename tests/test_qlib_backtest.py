from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.cost_model import CostModelConfig
from quant_platform.qlib_backtest import (
    QlibBacktestResult,
    calculate_trade_metrics,
    run_qlib_validation_suites,
)

pytestmark = pytest.mark.no_database


def _result(report: pd.DataFrame, excess: float = 0.10) -> QlibBacktestResult:
    return QlibBacktestResult(
        metrics={"annualized_excess_return": excess, "max_drawdown": -0.10},
        report=report,
        positions=None,
    )


def test_event_stress_requires_results_to_pass_not_only_event_count() -> None:
    dates = pd.bdate_range("2024-01-02", periods=100)
    full_report = pd.DataFrame(
        {
            "return": 0.0,
            "cost": 0.0,
            "bench": [-0.03 if index in {30, 70} else 0.001 for index in range(len(dates))],
            "turnover": 0.0,
        },
        index=dates,
    )

    def runner(start: str, end: str, _costs: CostModelConfig) -> QlibBacktestResult:
        selected = pd.bdate_range(start, end)
        report = pd.DataFrame(
            {"return": -0.01, "cost": 0.0, "bench": 0.0, "turnover": 0.0},
            index=selected,
        )
        return _result(report, excess=-0.10)

    validation = run_qlib_validation_suites(
        runner=runner,
        full_result=_result(full_report),
        start_time="2024-01-02",
        end_time=dates[-1].date().isoformat(),
        cost_model=CostModelConfig(),
        config={
            "rolling_window_days": 60,
            "rolling_step_days": 60,
            "min_rolling_windows": 1,
            "min_rolling_pass_rate": 0.0,
            "event_window_days": 20,
            "event_count": 2,
            "max_event_underperformance": 0.05,
            "min_event_stress_pass_rate": 0.60,
        },
    )
    event_stress = validation["event_stress"]
    assert event_stress["event_count"] == 2
    assert event_stress["pass_rate"] == 0.0
    assert event_stress["passed"] is False
    assert {item["status"] for item in event_stress["events"]} == {"failed"}


def test_trade_metrics_report_win_rate_and_profit_loss_ratio() -> None:
    metrics = calculate_trade_metrics(
        [
            {
                "instrument": "one",
                "side": "buy",
                "amount": 100,
                "trade_value": 1000,
                "trade_price": 10,
                "cost": 0,
            },
            {
                "instrument": "one",
                "side": "sell",
                "amount": 100,
                "trade_value": 1200,
                "trade_price": 12,
                "cost": 0,
            },
            {
                "instrument": "two",
                "side": "buy",
                "amount": 100,
                "trade_value": 1000,
                "trade_price": 10,
                "cost": 0,
            },
            {
                "instrument": "two",
                "side": "sell",
                "amount": 100,
                "trade_value": 900,
                "trade_price": 9,
                "cost": 0,
            },
        ]
    )
    assert metrics["closed_trade_count"] == 2
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["average_win"] == pytest.approx(200.0)
    assert metrics["average_loss"] == pytest.approx(-100.0)
    assert metrics["profit_loss_ratio"] == pytest.approx(2.0)


def test_capacity_curve_repeats_formal_runner_at_three_notionals() -> None:
    dates = pd.bdate_range("2024-01-02", periods=80)
    report = pd.DataFrame(
        {"return": 0.001, "cost": 0.0, "bench": 0.0, "turnover": 0.0}, index=dates
    )
    notionals: list[float] = []

    def runner(start: str, end: str, _costs: CostModelConfig) -> QlibBacktestResult:
        return _result(report.loc[start:end])

    def capacity_runner(notional: float) -> QlibBacktestResult:
        notionals.append(notional)
        return _result(report, excess=0.01)

    validation = run_qlib_validation_suites(
        runner=runner,
        full_result=_result(report),
        start_time=dates[0].date().isoformat(),
        end_time=dates[-1].date().isoformat(),
        cost_model=CostModelConfig(),
        config={"event_count": 0, "min_capacity_excess_return": 0.0},
        capacity_runner=capacity_runner,
    )
    assert notionals == [5_000_000.0, 20_000_000.0, 100_000_000.0]
    assert validation["capacity"]["passed"] is True


def test_capacity_curve_accepts_zero_when_zero_is_the_configured_floor() -> None:
    dates = pd.bdate_range("2024-01-02", periods=80)
    report = pd.DataFrame(
        {"return": 0.0, "cost": 0.0, "bench": 0.0, "turnover": 0.0}, index=dates
    )

    def runner(start: str, end: str, _costs: CostModelConfig) -> QlibBacktestResult:
        return _result(report.loc[start:end])

    validation = run_qlib_validation_suites(
        runner=runner,
        full_result=_result(report),
        start_time=dates[0].date().isoformat(),
        end_time=dates[-1].date().isoformat(),
        cost_model=CostModelConfig(),
        config={"event_count": 0, "min_capacity_excess_return": 0.0},
        capacity_runner=lambda _notional: _result(report, excess=0.0),
    )
    assert validation["capacity"]["passed"] is True
