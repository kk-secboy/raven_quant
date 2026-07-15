from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.cost_model import CostModelConfig
from quant_platform.qlib_backtest import QlibBacktestResult, run_qlib_validation_suites

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
