from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .cost_model import CostModelConfig
from .qlib_execution_strategy import create_qlib_execution_strategy

QLIB_ENGINE_VERSION = "qlib-policy-engine-v4-financial-correctness"


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


def calculate_qlib_metrics(
    report: pd.DataFrame, *, annual_minimum_acceptable_return: float = 0.0
) -> dict[str, Any]:
    required = {"return", "cost", "bench", "turnover"}
    if not required.issubset(report.columns) or report.empty:
        raise ValueError("Qlib portfolio report is incomplete")
    net = pd.to_numeric(report["return"], errors="coerce") - pd.to_numeric(
        report["cost"], errors="coerce"
    )
    benchmark = pd.to_numeric(report["bench"], errors="coerce")
    if (
        net.isna().any()
        or benchmark.isna().any()
        or not np.isfinite(net.to_numpy(dtype=float)).all()
        or not np.isfinite(benchmark.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Qlib portfolio report contains non-finite returns")
    if annual_minimum_acceptable_return <= -1:
        raise ValueError("annual minimum acceptable return must be greater than -100%")
    excess = net - benchmark
    nav = (1.0 + net).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    daily_mar = (1.0 + annual_minimum_acceptable_return) ** (1.0 / 252.0) - 1.0
    downside_shortfall = np.minimum(net.to_numpy(dtype=float) - daily_mar, 0.0)
    annualized_downside = float(np.sqrt(np.mean(np.square(downside_shortfall))) * np.sqrt(252))
    annualized_mar_excess = float((net.mean() - daily_mar) * 252)
    sortino_ok = annualized_downside > 0 and np.isfinite(annualized_downside)
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
        "sortino_ratio": float(annualized_mar_excess / annualized_downside)
        if sortino_ok
        else None,
        "sortino_status": "ok" if sortino_ok else "undefined_no_downside",
        "annual_minimum_acceptable_return": float(annual_minimum_acceptable_return),
        "annualized_downside_deviation": annualized_downside,
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
    execution_frequency: str | None = None,
    execution_policy: dict[str, Any] | None = None,
    instruments: list[str] | None = None,
    annual_minimum_acceptable_return: float = 0.0,
) -> QlibBacktestResult:
    try:
        from qlib.contrib.evaluate import backtest_daily
    except ImportError as exc:  # pragma: no cover - executed in configured Qlib runtime
        raise RuntimeError("the formal backtest runtime does not contain Qlib") from exc
    from .qlib_exchange import SquareRootImpactExchange

    if execution_method not in {"open", "twap", "vwap"}:
        raise ValueError("unsupported formal execution method")
    if execution_method in {"twap", "vwap"}:
        return _run_formal_minute_backtest(
            strategy=strategy,
            start_time=start_time,
            end_time=end_time,
            account=account,
            benchmark=benchmark,
            cost_model=cost_model,
            execution_method=execution_method,
            execution_frequency=execution_frequency,
            execution_policy=execution_policy,
            instruments=instruments,
            annual_minimum_acceptable_return=annual_minimum_acceptable_return,
        )
    exchange = SquareRootImpactExchange(
        cost_model=cost_model,
        freq="day",
        start_time=start_time,
        end_time=end_time,
        deal_price="$open",
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
    metrics = {
        **calculate_qlib_metrics(
            report,
            annual_minimum_acceptable_return=annual_minimum_acceptable_return,
        ),
        **calculate_trade_metrics(exchange.fill_log),
    }
    metrics.update(
        {
            "minute_execution_enforced": False,
            "execution_frequency": "day",
            "capacity_fill_ratio": 1.0,
        }
    )
    return QlibBacktestResult(metrics, report, positions, exchange.fill_log)


def _run_formal_minute_backtest(
    *,
    strategy: Any,
    start_time: str,
    end_time: str,
    account: float,
    benchmark: str,
    cost_model: CostModelConfig,
    execution_method: str,
    execution_frequency: str | None,
    execution_policy: dict[str, Any] | None,
    instruments: list[str] | None,
    annual_minimum_acceptable_return: float,
) -> QlibBacktestResult:
    if not execution_frequency or execution_frequency == "day":
        raise ValueError("TWAP/VWAP formal backtests require a minute execution frequency")
    if not execution_policy:
        raise ValueError("TWAP/VWAP formal backtests require an execution policy")
    if str(execution_policy.get("execution_algorithm") or "").lower() != execution_method:
        raise ValueError("execution policy algorithm does not match the formal execution method")
    if not instruments:
        raise ValueError("minute execution requires an explicit instrument universe")

    try:
        from qlib.backtest import backtest as qlib_backtest
        from qlib.backtest.executor import NestedExecutor, SimulatorExecutor
    except ImportError as exc:  # pragma: no cover - executed in configured Qlib runtime
        raise RuntimeError("the formal backtest runtime does not contain Qlib") from exc

    from .qlib_exchange import SquareRootImpactExchange

    slice_strategy = create_qlib_execution_strategy(execution_policy)
    exchange = SquareRootImpactExchange(
        cost_model=cost_model,
        freq=execution_frequency,
        start_time=start_time,
        end_time=end_time,
        codes=instruments,
        deal_price="$vwap",
        limit_threshold=(
            "Or($paused, Ge($vwap, $up_limit))",
            "Or($paused, Le($vwap, $down_limit))",
        ),
    )
    executor = NestedExecutor(
        time_per_step="day",
        inner_executor=SimulatorExecutor(
            time_per_step=execution_frequency,
            generate_portfolio_metrics=False,
        ),
        inner_strategy=slice_strategy,
        generate_portfolio_metrics=True,
    )
    portfolio_metrics, _indicators = qlib_backtest(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy,
        executor=executor,
        account=account,
        benchmark=benchmark,
        exchange_kwargs={"exchange": exchange},
    )
    daily = portfolio_metrics.get("1day")
    if daily is None:
        raise RuntimeError("nested Qlib backtest did not produce daily portfolio metrics")
    report, positions = daily
    execution_stats = slice_strategy.statistics()
    metrics = {
        **calculate_qlib_metrics(
            report,
            annual_minimum_acceptable_return=annual_minimum_acceptable_return,
        ),
        **calculate_trade_metrics(exchange.fill_log),
        **execution_stats,
        "minute_execution_enforced": True,
        "execution_frequency": execution_frequency,
    }
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
    robustness_runner: Callable[[dict[str, Any], CostModelConfig], QlibBacktestResult]
    | None = None,
    robustness_artifact_writer: Callable[[str, QlibBacktestResult], dict[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Repeat the same Qlib runner for cost, rolling and event validation."""

    def scenario_result(
        overrides: dict[str, Any], costs: CostModelConfig
    ) -> QlibBacktestResult:
        if robustness_runner is not None:
            return robustness_runner(overrides, costs)
        return runner(start_time, end_time, costs)

    topk = int(config.get("topk", 50))
    robustness_specs = {
        "double_cost": ({}, cost_model.doubled()),
        "turnover_75pct": (
            {"max_daily_turnover": float(config.get("max_daily_turnover", 0.15)) * 0.75},
            cost_model,
        ),
        "topk_80pct": (
            {
                "topk": max(5, int(np.floor(topk * 0.80))),
                "n_drop": min(
                    int(config.get("n_drop", 0)), max(5, int(np.floor(topk * 0.80)))
                ),
            },
            cost_model,
        ),
        "zero_retention_buffer": ({"n_drop": 0}, cost_model),
    }
    robustness: dict[str, dict[str, Any]] = {}
    for name, (overrides, costs) in robustness_specs.items():
        result = scenario_result(overrides, costs)
        excess_return = result.metrics.get("annualized_excess_return")
        drawdown = result.metrics.get("max_drawdown")
        passed = (
            excess_return is not None
            and np.isfinite(float(excess_return))
            and float(excess_return) > 0
            and drawdown is not None
            and np.isfinite(float(drawdown))
            and float(drawdown) >= -float(config.get("max_drawdown", 0.25))
        )
        robustness[name] = {
            "passed": passed,
            "overrides": overrides,
            "cost_model": costs.to_dict(),
            "metrics": result.metrics,
            "artifacts": (
                robustness_artifact_writer(name, result)
                if robustness_artifact_writer is not None
                else {}
            ),
        }
    robustness_pass_rate = sum(item["passed"] for item in robustness.values()) / len(robustness)
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
        event_report = full_result.report.loc[start:end]
        event_net = pd.to_numeric(event_report["return"], errors="coerce") - pd.to_numeric(
            event_report["cost"], errors="coerce"
        )
        event_benchmark = pd.to_numeric(event_report["bench"], errors="coerce")
        cumulative_return = float((1.0 + event_net).prod() - 1.0)
        cumulative_benchmark_return = float((1.0 + event_benchmark).prod() - 1.0)
        underperformance = cumulative_benchmark_return - cumulative_return
        passed = underperformance <= max_event_underperformance
        start_holdings, state_fill_count = _holdings_before(
            full_result.fills, start
        )
        events.append(
            {
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "state_source": "full_backtest_carried_positions",
                "position_state_method": "formal_fill_ledger_v1",
                "start_holdings": start_holdings,
                "state_fill_count": state_fill_count,
                "return_state_source": "full_backtest_report_slice",
                "metrics": calculate_qlib_metrics(
                    event_report,
                    annual_minimum_acceptable_return=float(
                        config.get("annual_minimum_acceptable_return", 0.0)
                    ),
                ),
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
                    "capacity_fill_ratio": result.metrics.get("capacity_fill_ratio"),
                }
            )
    minimum_capacity_return = float(config.get("min_capacity_excess_return", 0.0))
    minimum_fill_ratio = float(config.get("min_capacity_fill_ratio", 0.95))
    capacity_passed = len(capacity_curve) >= 3 and all(
        item.get("annualized_excess_return") is not None
        and np.isfinite(float(item["annualized_excess_return"]))
        and float(item["annualized_excess_return"]) >= minimum_capacity_return
        and item.get("capacity_fill_ratio") is not None
        and np.isfinite(float(item["capacity_fill_ratio"]))
        and float(item["capacity_fill_ratio"]) >= minimum_fill_ratio
        for item in capacity_curve
    )
    return {
        "robustness": {
            "scenario_count": len(robustness),
            "pass_rate": robustness_pass_rate,
            "passed": len(robustness) == 4 and robustness_pass_rate >= float(
                config.get("min_robustness_pass_rate", 1.0)
            ),
            "minimum_pass_rate": float(config.get("min_robustness_pass_rate", 1.0)),
            "scenarios": robustness,
        },
        "double_cost": robustness["double_cost"],
        "rolling": {
            "window_count": len(rolling),
            "pass_rate": rolling_pass_rate,
            "passed": len(rolling) >= int(config.get("min_rolling_windows", 3))
            and rolling_pass_rate >= float(config.get("min_rolling_pass_rate", 0.60)),
            "windows": rolling,
        },
        "event_stress": {
            "state_source": "full_backtest_carried_positions",
            "position_state_method": "formal_fill_ledger_v1",
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
            "minimum_fill_ratio": minimum_fill_ratio,
            "points": capacity_curve,
        },
    }


def _holdings_before(
    fills: list[dict[str, Any]], start: pd.Timestamp
) -> tuple[dict[str, float], int]:
    holdings: dict[str, float] = {}
    applied = 0
    boundary = pd.Timestamp(start).tz_localize(None)
    parsed: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    for fill in fills:
        timestamp = pd.to_datetime(fill.get("date"), errors="coerce")
        if pd.isna(timestamp):
            raise ValueError("formal fill ledger contains an invalid timestamp")
        timestamp = pd.Timestamp(timestamp)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(None)
        parsed.append((timestamp, fill))
    for timestamp, fill in sorted(parsed, key=lambda item: item[0]):
        if timestamp >= boundary:
            break
        instrument = str(fill.get("instrument") or "")
        amount = float(fill.get("amount") or 0.0)
        side = str(fill.get("side") or "")
        if not instrument or not np.isfinite(amount) or amount <= 0 or side not in {"buy", "sell"}:
            raise ValueError("formal fill ledger contains invalid position evidence")
        signed = amount if side == "buy" else -amount
        holdings[instrument] = holdings.get(instrument, 0.0) + signed
        if holdings[instrument] < -1e-8:
            raise ValueError("formal fill ledger reconstructs a negative long-only position")
        applied += 1
    return (
        {
            instrument: float(amount)
            for instrument, amount in sorted(holdings.items())
            if amount > 1e-8
        },
        applied,
    )
