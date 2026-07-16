from __future__ import annotations

import pandas as pd
import pytest
from qlib_test_doubles import risk_analysis

from quant_platform.cost_model import CostModelConfig
from quant_platform.qlib_backtest import (
    QlibBacktestResult,
    aggregate_intraday_report,
    calculate_qlib_metrics,
    calculate_trade_metrics,
    run_qlib_validation_suites,
)

pytestmark = pytest.mark.no_database


@pytest.fixture(autouse=True)
def _qlib_analysis_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "quant_platform.qlib_backtest._load_qlib_risk_analysis",
        lambda: risk_analysis,
    )


def _result(
    report: pd.DataFrame,
    excess: float = 0.10,
    fills: list[dict] | None = None,
) -> QlibBacktestResult:
    return QlibBacktestResult(
        metrics={
            "annualized_excess_return": excess,
            "max_drawdown": -0.10,
            "capacity_fill_ratio": 1.0,
        },
        report=report,
        positions=None,
        fills=fills or [],
    )


def test_event_stress_requires_results_to_pass_not_only_event_count() -> None:
    dates = pd.bdate_range("2024-01-02", periods=100)
    full_report = pd.DataFrame(
        {
            "return": -0.01,
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
        full_result=_result(
            full_report,
            fills=[
                {
                    "instrument": "SH600000",
                    "date": "2024-01-03 10:00:00",
                    "side": "buy",
                    "amount": 100.0,
                }
            ],
        ),
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
    assert {item["state_source"] for item in event_stress["events"]} == {
        "full_backtest_carried_positions"
    }
    assert event_stress["position_state_method"] == "formal_fill_ledger_v1"
    assert all(item["start_holdings"] == {"SH600000": 100.0} for item in event_stress["events"])


def test_sortino_uses_target_shortfall_not_std_of_negative_observations() -> None:
    dates = pd.bdate_range("2024-01-02", periods=40)
    report = pd.DataFrame(
        {"return": -0.01, "cost": 0.0, "bench": 0.0, "turnover": 0.0},
        index=dates,
    )

    metrics = calculate_qlib_metrics(report)

    assert metrics["analysis_engine"] == "qlib.contrib.evaluate.risk_analysis"
    assert metrics["annualization_periods"] == 252
    assert metrics["return_accumulation"] == "geometric"
    assert metrics["cumulative_return"] == pytest.approx((0.99**40) - 1.0)
    assert metrics["sortino_status"] == "ok"
    assert metrics["annualized_downside_deviation"] == pytest.approx(0.01 * 252**0.5)
    assert metrics["sortino_ratio"] == pytest.approx(-252**0.5)


def test_intraday_report_is_geometrically_aggregated_to_daily_metrics() -> None:
    index = pd.to_datetime(
        [
            "2026-07-10 09:35",
            "2026-07-10 09:40",
            "2026-07-13 09:35",
        ]
    )
    report = pd.DataFrame(
        {
            "return": [0.01, -0.005, 0.02],
            "cost": [0.001, 0.002, 0.001],
            "bench": [0.002, 0.003, -0.001],
            "turnover": [0.1, 0.2, 0.1],
            "account": [101.0, 100.495, 102.5049],
        },
        index=index,
    )

    daily = aggregate_intraday_report(report)

    assert daily.index.tolist() == [
        pd.Timestamp("2026-07-10"),
        pd.Timestamp("2026-07-13"),
    ]
    assert daily.loc["2026-07-10", "return"] == pytest.approx(
        (1.01 * 0.995) - 1.0
    )
    assert daily.loc["2026-07-10", "bench"] == pytest.approx(
        (1.002 * 1.003) - 1.0
    )
    assert daily.loc["2026-07-10", "cost"] == pytest.approx(0.003)
    assert daily.loc["2026-07-10", "turnover"] == pytest.approx(0.3)
    assert daily.loc["2026-07-10", "account"] == pytest.approx(100.495)


def test_robustness_runs_all_four_configured_scenarios() -> None:
    dates = pd.bdate_range("2024-01-02", periods=80)
    report = pd.DataFrame(
        {"return": 0.001, "cost": 0.0, "bench": 0.0, "turnover": 0.0}, index=dates
    )
    seen: list[dict] = []

    def runner(start: str, end: str, _costs: CostModelConfig) -> QlibBacktestResult:
        return _result(report.loc[start:end])

    def robustness(overrides: dict, _costs: CostModelConfig) -> QlibBacktestResult:
        seen.append(overrides)
        return _result(report)

    validation = run_qlib_validation_suites(
        runner=runner,
        full_result=_result(report),
        start_time=dates[0].date().isoformat(),
        end_time=dates[-1].date().isoformat(),
        cost_model=CostModelConfig(),
        config={
            "topk": 50,
            "n_drop": 5,
            "max_daily_turnover": 0.20,
            "event_count": 0,
            "min_robustness_pass_rate": 1.0,
        },
        robustness_runner=robustness,
    )

    assert seen == [
        {},
        {"max_daily_turnover": pytest.approx(0.15)},
        {"topk": 40, "n_drop": 5},
        {"n_drop": 0},
    ]
    assert validation["robustness"]["scenario_count"] == 4
    assert validation["robustness"]["pass_rate"] == 1.0
    assert validation["robustness"]["passed"] is True


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


def test_capacity_curve_fails_when_minute_fill_ratio_is_below_gate() -> None:
    dates = pd.bdate_range("2024-01-02", periods=80)
    report = pd.DataFrame(
        {"return": 0.001, "cost": 0.0, "bench": 0.0, "turnover": 0.0}, index=dates
    )

    def runner(start: str, end: str, _costs: CostModelConfig) -> QlibBacktestResult:
        return _result(report.loc[start:end])

    def capacity_runner(_notional: float) -> QlibBacktestResult:
        result = _result(report, excess=0.05)
        result.metrics["capacity_fill_ratio"] = 0.80
        return result

    validation = run_qlib_validation_suites(
        runner=runner,
        full_result=_result(report),
        start_time=dates[0].date().isoformat(),
        end_time=dates[-1].date().isoformat(),
        cost_model=CostModelConfig(),
        config={"event_count": 0, "min_capacity_fill_ratio": 0.95},
        capacity_runner=capacity_runner,
    )
    assert validation["capacity"]["passed"] is False
    assert validation["capacity"]["minimum_fill_ratio"] == pytest.approx(0.95)


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
