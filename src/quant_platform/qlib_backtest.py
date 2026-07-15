from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .cost_model import CostModelConfig

QLIB_ENGINE_VERSION = "qlib-policy-engine-v2"


@dataclass(frozen=True)
class QlibBacktestResult:
    metrics: dict[str, Any]
    report: pd.DataFrame
    positions: Any
    fills: list[dict[str, Any]] = field(default_factory=list)


def calculate_trade_metrics(fills: list[dict[str, Any]]) -> dict[str, Any]:
    positions: dict[str, dict[str, float]] = {}
    closed_pnl: list[float] = []
    for fill in fills:
        instrument = str(fill["instrument"])
        amount = max(0.0, float(fill["amount"]))
        value = max(0.0, float(fill["trade_value"]))
        cost = max(0.0, float(fill["cost"]))
        state = positions.setdefault(instrument, {"amount": 0.0, "basis": 0.0})
        if fill["side"] == "buy":
            new_amount = state["amount"] + amount
            if new_amount > 0:
                state["basis"] = (
                    state["amount"] * state["basis"] + value + cost
                ) / new_amount
                state["amount"] = new_amount
            continue
        sold = min(amount, state["amount"])
        if sold <= 0:
            continue
        net_proceeds = sold * float(fill["trade_price"]) - cost * (sold / amount if amount else 0.0)
        closed_pnl.append(net_proceeds - sold * state["basis"])
        state["amount"] -= sold
        if state["amount"] <= 1e-10:
            state["amount"] = 0.0
            state["basis"] = 0.0
    wins = [value for value in closed_pnl if value > 0]
    losses = [value for value in closed_pnl if value < 0]
    average_win = float(np.mean(wins)) if wins else 0.0
    average_loss = float(np.mean(losses)) if losses else 0.0
    return {
        "closed_trade_count": len(closed_pnl),
        "win_rate": float(len(wins) / len(closed_pnl)) if closed_pnl else None,
        "average_win": average_win,
        "average_loss": average_loss,
        "profit_loss_ratio": (
            float(average_win / abs(average_loss)) if wins and losses and average_loss else None
        ),
        "gross_realized_pnl": float(sum(closed_pnl)),
    }


def calculate_qlib_metrics(report: pd.DataFrame) -> dict[str, Any]:
    required = {"return", "cost", "bench", "turnover"}
    if not required.issubset(report.columns) or report.empty:
        raise ValueError("Qlib portfolio report is incomplete")
    net = pd.to_numeric(report["return"], errors="coerce") - pd.to_numeric(
        report["cost"], errors="coerce"
    )
    benchmark = pd.to_numeric(report["bench"], errors="coerce")
    excess = net - benchmark
    nav = (1.0 + net).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    downside = net[net < 0].std(ddof=1)
    excess_std = excess.std(ddof=1)
    net_std = net.std(ddof=1)
    return {
        "backtest_engine": "qlib",
        "backtest_engine_version": QLIB_ENGINE_VERSION,
        "qlib_native_backtest": True,
        "annualized_return": float(nav.iloc[-1] ** (252 / len(net)) - 1.0),
        "annualized_excess_return": float(excess.mean() * 252),
        "tracking_error": float(excess_std * np.sqrt(252)),
        "information_ratio": float(excess.mean() / excess_std * np.sqrt(252))
        if excess_std and np.isfinite(excess_std)
        else None,
        "sharpe_ratio": float(net.mean() / net_std * np.sqrt(252))
        if net_std and np.isfinite(net_std)
        else None,
        "sortino_ratio": float(net.mean() / downside * np.sqrt(252))
        if downside and np.isfinite(downside)
        else None,
        "max_drawdown": float(drawdown.min()),
        "average_turnover": float(pd.to_numeric(report["turnover"]).mean()),
        "total_cost": float(pd.to_numeric(report["cost"]).sum()),
        "trading_days": int(len(report)),
    }


def run_formal_qlib_backtest(
    *,
    strategy: Any,
    start_time: str,
    end_time: str,
    account: float,
    benchmark: str,
    cost_model: CostModelConfig,
    execution_method: str = "open",
) -> QlibBacktestResult:
    try:
        from qlib.contrib.evaluate import backtest_daily
    except ImportError as exc:  # pragma: no cover - executed in configured Qlib runtime
        raise RuntimeError("the formal backtest runtime does not contain Qlib") from exc
    from .qlib_exchange import SquareRootImpactExchange

    deal_prices = {
        "open": "$open",
        "vwap": "$vwap",
        "twap": (
            "($open+$high+$low+$close)/4",
            "($open+$high+$low+$close)/4",
        ),
    }
    if execution_method not in deal_prices:
        raise ValueError("unsupported formal execution method")
    exchange = SquareRootImpactExchange(
        cost_model=cost_model,
        freq="day",
        start_time=start_time,
        end_time=end_time,
        deal_price=deal_prices[execution_method],
        limit_threshold=(
            "Or($paused, Ge($open, $up_limit))",
            "Or($paused, Le($open, $down_limit))",
        ),
    )
    report, positions = backtest_daily(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy,
        account=account,
        benchmark=benchmark,
        exchange_kwargs={"exchange": exchange},
    )
    metrics = {**calculate_qlib_metrics(report), **calculate_trade_metrics(exchange.fill_log)}
    return QlibBacktestResult(metrics, report, positions, exchange.fill_log)


def run_qlib_validation_suites(
    *,
    runner: Callable[[str, str, CostModelConfig], QlibBacktestResult],
    full_result: QlibBacktestResult,
    start_time: str,
    end_time: str,
    cost_model: CostModelConfig,
    config: dict[str, Any],
    capacity_runner: Callable[[float], QlibBacktestResult] | None = None,
) -> dict[str, Any]:
    """Repeat the same Qlib runner for cost, rolling and event validation."""

    double_cost = runner(start_time, end_time, cost_model.doubled())
    cost_passed = (double_cost.metrics.get("annualized_excess_return") or -np.inf) > 0
    dates = pd.DatetimeIndex(full_result.report.index).tz_localize(None)
    window = int(config.get("rolling_window_days", 252))
    step = int(config.get("rolling_step_days", 63))
    rolling: list[dict[str, Any]] = []
    for offset in range(0, max(0, len(dates) - window + 1), step):
        selected = dates[offset : offset + window]
        if len(selected) != window:
            continue
        result = runner(selected[0].date().isoformat(), selected[-1].date().isoformat(), cost_model)
        rolling.append(
            {
                "start": selected[0].date().isoformat(),
                "end": selected[-1].date().isoformat(),
                "metrics": result.metrics,
                "status": "passed"
                if (result.metrics.get("annualized_excess_return") or -np.inf) > 0
                else "failed",
            }
        )
    event_window = int(config.get("event_window_days", 20))
    event_count = int(config.get("event_count", 5))
    max_event_underperformance = float(config.get("max_event_underperformance", 0.05))
    min_event_stress_pass_rate = float(config.get("min_event_stress_pass_rate", 0.60))
    bench = pd.to_numeric(full_result.report["bench"], errors="coerce")
    losses = ((1.0 + bench).rolling(event_window).apply(np.prod, raw=True) - 1.0).sort_values()
    selected_ends: list[pd.Timestamp] = []
    for value in losses.dropna().index:
        end = pd.Timestamp(value).tz_localize(None)
        if all(
            abs(dates.get_loc(end) - dates.get_loc(item)) >= event_window for item in selected_ends
        ):
            selected_ends.append(end)
        if len(selected_ends) == event_count:
            break
    events: list[dict[str, Any]] = []
    for end in selected_ends:
        end_index = int(dates.get_loc(end))
        start = dates[end_index - event_window + 1]
        result = runner(start.date().isoformat(), end.date().isoformat(), cost_model)
        event_net = pd.to_numeric(result.report["return"], errors="coerce") - pd.to_numeric(
            result.report["cost"], errors="coerce"
        )
        event_benchmark = pd.to_numeric(result.report["bench"], errors="coerce")
        cumulative_return = float((1.0 + event_net).prod() - 1.0)
        cumulative_benchmark_return = float((1.0 + event_benchmark).prod() - 1.0)
        underperformance = cumulative_benchmark_return - cumulative_return
        passed = underperformance <= max_event_underperformance
        events.append(
            {
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "metrics": result.metrics,
                "cumulative_return": cumulative_return,
                "cumulative_benchmark_return": cumulative_benchmark_return,
                "underperformance": underperformance,
                "status": "passed" if passed else "failed",
            }
        )
    rolling_pass_rate = (
        sum(item["status"] == "passed" for item in rolling) / len(rolling) if rolling else 0.0
    )
    event_pass_rate = (
        sum(item["status"] == "passed" for item in events) / len(events) if events else 0.0
    )
    capacity_curve: list[dict[str, Any]] = []
    if capacity_runner is not None:
        notionals = sorted(
            {
                float(value)
                for value in config.get(
                    "capacity_curve_notionals", [5_000_000, 20_000_000, 100_000_000]
                )
            }
        )
        for notional in notionals:
            result = capacity_runner(notional)
            capacity_curve.append(
                {
                    "notional": notional,
                    "annualized_return": result.metrics.get("annualized_return"),
                    "annualized_excess_return": result.metrics.get("annualized_excess_return"),
                    "sharpe_ratio": result.metrics.get("sharpe_ratio"),
                    "total_cost": result.metrics.get("total_cost"),
                }
            )
    minimum_capacity_return = float(config.get("min_capacity_excess_return", 0.0))
    capacity_passed = len(capacity_curve) >= 3 and all(
        item.get("annualized_excess_return") is not None
        and np.isfinite(float(item["annualized_excess_return"]))
        and float(item["annualized_excess_return"]) >= minimum_capacity_return
        for item in capacity_curve
    )
    return {
        "double_cost": {"passed": cost_passed, "metrics": double_cost.metrics},
        "rolling": {
            "window_count": len(rolling),
            "pass_rate": rolling_pass_rate,
            "passed": len(rolling) >= int(config.get("min_rolling_windows", 3))
            and rolling_pass_rate >= float(config.get("min_rolling_pass_rate", 0.60)),
            "windows": rolling,
        },
        "event_stress": {
            "event_count": len(events),
            "pass_rate": event_pass_rate,
            "passed": len(events) >= event_count and event_pass_rate >= min_event_stress_pass_rate,
            "max_event_underperformance": max_event_underperformance,
            "min_pass_rate": min_event_stress_pass_rate,
            "events": events,
        },
        "capacity": {
            "passed": capacity_passed,
            "minimum_excess_return": minimum_capacity_return,
            "points": capacity_curve,
        },
    }
