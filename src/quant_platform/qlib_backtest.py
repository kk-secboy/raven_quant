from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .cost_model import CostModelConfig, CostScheduleBook
from .qlib_execution_strategy import create_qlib_execution_strategy

QLIB_ENGINE_VERSION = "qlib-policy-engine-v5-single-mainline"

# Per-component cost stress scenarios (design draft 7.3): each entry degrades
# exactly the named CostModelConfig fields via CostScheduleBook.scaled, so a
# scenario can be attributed to one cost channel. ``fill_rate_75pct`` tightens
# the volume-participation cap, stressing the achievable fill rate rather than
# a fee. ``double_cost`` (all components x2) stays as the aggregate scenario.
COMPONENT_COST_STRESS_MULTIPLIERS: dict[str, dict[str, float]] = {
    "commission_2x": {
        "buy_commission_rate": 2.0,
        "sell_commission_rate": 2.0,
        "min_commission": 2.0,
    },
    "slippage_2x": {"fixed_slippage_rate": 2.0},
    "impact_2x": {"impact_at_max_participation": 2.0},
    "fill_rate_75pct": {"max_volume_participation": 0.75},
}


def _resolve_cost_schedule(
    cost_model: CostModelConfig | None,
    cost_schedule: CostScheduleBook | None,
) -> CostScheduleBook:
    if (cost_model is None) == (cost_schedule is None):
        raise ValueError("exactly one of cost_model or cost_schedule is required")
    return cost_schedule or CostScheduleBook.from_versions([cost_model])


def _load_qlib_risk_analysis() -> Callable[..., pd.DataFrame]:
    try:
        from qlib.contrib.evaluate import risk_analysis
    except ImportError as exc:  # pragma: no cover - configured runtime assertion
        raise RuntimeError("Qlib analysis runtime is unavailable") from exc
    return risk_analysis


def _risk_value(analysis: pd.DataFrame, name: str) -> float:
    if name not in analysis.index or analysis.shape[1] != 1:
        raise ValueError(f"Qlib risk analysis does not contain {name}")
    value = float(analysis.loc[name].iloc[0])
    if not np.isfinite(value):
        raise ValueError(f"Qlib risk analysis returned a non-finite {name}")
    return value


def _optional_risk_value(analysis: pd.DataFrame, name: str) -> float | None:
    if name not in analysis.index or analysis.shape[1] != 1:
        raise ValueError(f"Qlib risk analysis does not contain {name}")
    value = float(analysis.loc[name].iloc[0])
    return value if np.isfinite(value) else None


@dataclass(frozen=True)
class QlibBacktestResult:
    metrics: dict[str, Any]
    report: pd.DataFrame
    positions: Any
    fills: list[dict[str, Any]] = field(default_factory=list)


def calculate_trade_metrics(fills: list[dict[str, Any]]) -> dict[str, Any]:
    positions: dict[str, dict[str, float]] = {}
    closed_net_pnl: list[float] = []
    closed_gross_pnl: list[float] = []
    for fill in fills:
        instrument = str(fill["instrument"])
        amount = max(0.0, float(fill["amount"]))
        value = max(0.0, float(fill["trade_value"]))
        cost = max(0.0, float(fill["cost"]))
        state = positions.setdefault(
            instrument,
            {
                "amount": 0.0,
                "gross_basis": 0.0,
                "net_basis": 0.0,
                "cycle_gross_pnl": 0.0,
                "cycle_net_pnl": 0.0,
            },
        )
        if fill["side"] == "buy":
            new_amount = state["amount"] + amount
            if new_amount > 0:
                state["gross_basis"] = (
                    state["amount"] * state["gross_basis"] + value
                ) / new_amount
                state["net_basis"] = (
                    state["amount"] * state["net_basis"] + value + cost
                ) / new_amount
                state["amount"] = new_amount
            continue
        sold = min(amount, state["amount"])
        if sold <= 0:
            continue
        gross_proceeds = sold * float(fill["trade_price"])
        allocated_cost = cost * (sold / amount if amount else 0.0)
        state["cycle_gross_pnl"] += gross_proceeds - sold * state["gross_basis"]
        state["cycle_net_pnl"] += (
            gross_proceeds - allocated_cost - sold * state["net_basis"]
        )
        state["amount"] -= sold
        if state["amount"] <= 1e-10:
            closed_gross_pnl.append(state["cycle_gross_pnl"])
            closed_net_pnl.append(state["cycle_net_pnl"])
            state["amount"] = 0.0
            state["gross_basis"] = 0.0
            state["net_basis"] = 0.0
            state["cycle_gross_pnl"] = 0.0
            state["cycle_net_pnl"] = 0.0
    wins = [value for value in closed_net_pnl if value > 0]
    losses = [value for value in closed_net_pnl if value < 0]
    average_win = float(np.mean(wins)) if wins else 0.0
    average_loss = float(np.mean(losses)) if losses else 0.0
    return {
        "closed_trade_count": len(closed_net_pnl),
        "win_rate": (
            float(len(wins) / len(closed_net_pnl))
            if closed_net_pnl
            else None
        ),
        "average_win": average_win,
        "average_loss": average_loss,
        "profit_loss_ratio": (
            float(average_win / abs(average_loss)) if wins and losses and average_loss else None
        ),
        "gross_realized_pnl": float(sum(closed_gross_pnl)),
        "net_realized_pnl": float(sum(closed_net_pnl)),
    }


def calculate_capacity_fill_ratio(fills: list[dict[str, Any]]) -> float:
    """Return requested-notional-weighted execution completion.

    Share counts cannot be added across instruments with different prices.
    Weighting each order by ``requested_amount * trade_price`` makes the
    capacity statistic an economic fraction of requested notional.  Missing
    prices on a non-empty request fail closed because its notional cannot be
    established.
    """

    if not fills:
        return 1.0
    requested_value = 0.0
    filled_value = 0.0
    for fill in fills:
        requested_amount = float(fill.get("requested_amount") or 0.0)
        amount = float(fill.get("amount") or 0.0)
        price = float(fill.get("trade_price") or 0.0)
        trade_value = float(fill.get("trade_value") or 0.0)
        values = (requested_amount, amount, price, trade_value)
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError("formal fill ledger contains invalid capacity evidence")
        if amount > requested_amount + 1e-8:
            raise ValueError("formal fill ledger executes more than the requested amount")
        if requested_amount <= 0:
            continue
        if price <= 0:
            return 0.0
        requested_value += requested_amount * price
        filled_value += min(trade_value, requested_amount * price)
    if requested_value <= 0:
        return 1.0
    return float(min(1.0, filled_value / requested_value))


def _max_drawdown_recovery(nav: pd.Series) -> tuple[int | None, str]:
    """Trading days from the max-drawdown trough back to the prior peak.

    Returns ``(None, "ongoing")`` when the NAV has not recovered by the end of
    the report and ``(0, "no_drawdown")`` when the NAV never trades below its
    running peak.
    """

    peaks = nav.cummax()
    drawdown = nav / peaks - 1.0
    if not bool((drawdown < 0).any()):
        return 0, "no_drawdown"
    trough_position = int(drawdown.to_numpy(dtype=float).argmin())
    peak_level = float(peaks.iloc[trough_position])
    after = nav.iloc[trough_position + 1 :].to_numpy(dtype=float)
    hits = np.flatnonzero(after >= peak_level * (1.0 - 1e-12))
    if len(hits) == 0:
        return None, "ongoing"
    return int(hits[0]) + 1, "recovered"


def calculate_qlib_metrics(
    report: pd.DataFrame,
    *,
    annual_minimum_acceptable_return: float = 0.0,
    risk_analysis_fn: Callable[..., pd.DataFrame] | None = None,
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
    analyzer = risk_analysis_fn or _load_qlib_risk_analysis()
    net_geometric = analyzer(net, N=252, freq=None, mode="product")
    net_standard = analyzer(net, N=252, freq=None, mode="sum")
    excess_standard = analyzer(excess, N=252, freq=None, mode="sum")
    daily_mar = (1.0 + annual_minimum_acceptable_return) ** (1.0 / 252.0) - 1.0
    downside_shortfall = np.minimum(net.to_numpy(dtype=float) - daily_mar, 0.0)
    annualized_downside = float(np.sqrt(np.mean(np.square(downside_shortfall))) * np.sqrt(252))
    annualized_mar_excess = float((net.mean() - daily_mar) * 252)
    sortino_ok = annualized_downside > 0 and np.isfinite(annualized_downside)
    excess_std = _risk_value(excess_standard, "std")
    recovery_days, recovery_status = _max_drawdown_recovery(nav)
    return {
        "backtest_engine": "qlib",
        "backtest_engine_version": QLIB_ENGINE_VERSION,
        "qlib_native_backtest": True,
        "analysis_engine": "qlib.contrib.evaluate.risk_analysis",
        "annualization_periods": 252,
        "return_accumulation": "geometric",
        "cumulative_return": float(nav.iloc[-1] - 1.0),
        "annualized_return": _risk_value(net_geometric, "annualized_return"),
        "annualized_excess_return": _risk_value(excess_standard, "annualized_return"),
        "tracking_error": float(excess_std * np.sqrt(252)),
        "information_ratio": _optional_risk_value(excess_standard, "information_ratio"),
        "sharpe_ratio": _optional_risk_value(net_standard, "information_ratio"),
        "sortino_ratio": float(annualized_mar_excess / annualized_downside) if sortino_ok else None,
        "sortino_status": "ok" if sortino_ok else "undefined_no_downside",
        "annual_minimum_acceptable_return": float(annual_minimum_acceptable_return),
        "annualized_downside_deviation": annualized_downside,
        "max_drawdown": _risk_value(net_geometric, "max_drawdown"),
        "max_drawdown_recovery_days": recovery_days,
        "max_drawdown_recovery_status": recovery_status,
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
    cost_model: CostModelConfig | None = None,
    cost_schedule: CostScheduleBook | None = None,
    execution_method: str = "open",
    signal_frequency: str = "day",
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

    schedule = _resolve_cost_schedule(cost_model, cost_schedule)
    if execution_method not in {"open", "twap", "vwap", "next_bar"}:
        raise ValueError("unsupported formal execution method")
    if execution_method in {"twap", "vwap", "next_bar"}:
        return _run_formal_minute_backtest(
            strategy=strategy,
            start_time=start_time,
            end_time=end_time,
            account=account,
            benchmark=benchmark,
            cost_schedule=schedule,
            execution_method=execution_method,
            signal_frequency=signal_frequency,
            execution_frequency=execution_frequency,
            execution_policy=execution_policy,
            instruments=instruments,
            annual_minimum_acceptable_return=annual_minimum_acceptable_return,
        )
    exchange = SquareRootImpactExchange(
        cost_schedule=schedule,
        freq="day",
        start_time=start_time,
        end_time=end_time,
        # Restrict the quote universe to the strategy instruments: the provider
        # also carries index quotes (e.g. BJ899050) that have no price-limit
        # channels, and Qlib's default codes=None would load them and crash
        # evaluating the limit-threshold expressions.
        **({"codes": instruments} if instruments else {}),
        deal_price="$open",
        limit_threshold=(
            "Or(Gt($paused, 0), Ge($open, $up_limit))",
            "Or(Gt($paused, 0), Le($open, $down_limit))",
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
            "capacity_fill_ratio": calculate_capacity_fill_ratio(exchange.fill_log),
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
    cost_schedule: CostScheduleBook,
    execution_method: str,
    signal_frequency: str,
    execution_frequency: str | None,
    execution_policy: dict[str, Any] | None,
    instruments: list[str] | None,
    annual_minimum_acceptable_return: float,
) -> QlibBacktestResult:
    if not execution_frequency or execution_frequency == "day":
        raise ValueError("minute formal backtests require a minute execution frequency")
    if not execution_policy:
        raise ValueError("minute formal backtests require an execution policy")
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
        cost_schedule=cost_schedule,
        freq=execution_frequency,
        start_time=start_time,
        end_time=end_time,
        codes=instruments,
        deal_price="$vwap",
        limit_threshold=(
            "Or(Gt($paused, 0), Ge($vwap, $up_limit))",
            "Or(Gt($paused, 0), Le($vwap, $down_limit))",
        ),
    )
    executor = NestedExecutor(
        time_per_step=signal_frequency,
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
    result = portfolio_metrics.get(signal_frequency)
    if result is None and signal_frequency == "day":
        result = portfolio_metrics.get("1day")
    if result is None:
        raise RuntimeError(
            "nested Qlib backtest did not produce the configured signal-frequency metrics"
        )
    report, positions = result
    if signal_frequency != "day":
        report = aggregate_intraday_report(report)
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
        "signal_frequency": signal_frequency,
    }
    return QlibBacktestResult(metrics, report, positions, exchange.fill_log)


def aggregate_intraday_report(report: pd.DataFrame) -> pd.DataFrame:
    """Convert Qlib intraday portfolio metrics to the governed daily metric contract."""

    if report.empty or not isinstance(report.index, pd.DatetimeIndex):
        raise ValueError("intraday Qlib portfolio report is empty or has no datetime index")
    required = {"return", "cost", "bench", "turnover"}
    if not required.issubset(report.columns):
        raise ValueError("intraday Qlib portfolio report is incomplete")
    values = report.copy()
    if values.index.tz is not None:
        values.index = values.index.tz_localize(None)
    dates = values.index.normalize()
    numeric_values: dict[str, pd.Series] = {}
    for column in values.columns:
        numeric = pd.to_numeric(values[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("intraday Qlib portfolio report contains non-numeric values")
        numeric_values[column] = numeric

    gross_factors = 1.0 + numeric_values["return"]
    net_factors = 1.0 + numeric_values["return"] - numeric_values["cost"]
    benchmark_factors = 1.0 + numeric_values["bench"]
    invalid_factor = (
        (gross_factors <= 0).any()
        or (net_factors <= 0).any()
        or (benchmark_factors <= 0).any()
    )
    if bool(invalid_factor):
        raise ValueError("intraday Qlib report contains a return at or below -100%")
    gross_daily = gross_factors.groupby(dates).prod() - 1.0
    net_daily = net_factors.groupby(dates).prod() - 1.0
    result: dict[str, pd.Series] = {
        "return": gross_daily,
        # Preserve Qlib's report contract (net = return - cost) while making
        # the daily net return exactly equal to the compounded intraday net
        # path.  A plain sum of intraday costs drops return/cost cross terms.
        "cost": gross_daily - net_daily,
        "bench": benchmark_factors.groupby(dates).prod() - 1.0,
    }
    for column, numeric in numeric_values.items():
        if column in {"return", "cost", "bench"}:
            continue
        grouped = numeric.groupby(dates)
        if column == "turnover":
            result[column] = grouped.sum()
        else:
            result[column] = grouped.last()
    daily = pd.DataFrame(result)
    daily.index.name = report.index.name or "datetime"
    return daily.sort_index()


def run_qlib_validation_suites(
    *,
    runner: Callable[[str, str, CostScheduleBook], QlibBacktestResult],
    full_result: QlibBacktestResult,
    start_time: str,
    end_time: str,
    cost_model: CostModelConfig | None = None,
    cost_schedule: CostScheduleBook | None = None,
    config: dict[str, Any],
    capacity_runner: Callable[[float], QlibBacktestResult] | None = None,
    robustness_runner: Callable[[dict[str, Any], CostScheduleBook], QlibBacktestResult]
    | None = None,
    robustness_artifact_writer: Callable[[str, QlibBacktestResult], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Repeat the same Qlib runner for cost, rolling and event validation."""

    schedule = _resolve_cost_schedule(cost_model, cost_schedule)

    def scenario_result(overrides: dict[str, Any], costs: CostScheduleBook) -> QlibBacktestResult:
        if robustness_runner is not None:
            return robustness_runner(overrides, costs)
        return runner(start_time, end_time, costs)

    topk = int(config.get("topk", 50))
    robustness_specs = {
        "double_cost": ({}, schedule.doubled()),
        "turnover_75pct": (
            {"max_daily_turnover": float(config.get("max_daily_turnover", 0.15)) * 0.75},
            schedule,
        ),
        "topk_80pct": (
            {
                "topk": max(5, int(np.floor(topk * 0.80))),
                "n_drop": min(int(config.get("n_drop", 0)), max(5, int(np.floor(topk * 0.80)))),
            },
            schedule,
        ),
        "zero_retention_buffer": ({"n_drop": 0}, schedule),
    }

    def scenario_entry(
        name: str, overrides: dict[str, Any], costs: CostScheduleBook
    ) -> dict[str, Any]:
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
        return {
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

    robustness = {
        name: scenario_entry(name, overrides, costs)
        for name, (overrides, costs) in robustness_specs.items()
    }
    robustness_pass_rate = sum(item["passed"] for item in robustness.values()) / len(robustness)
    component_cost_stress = {
        name: scenario_entry(name, {}, schedule.scaled(**multipliers))
        for name, multipliers in COMPONENT_COST_STRESS_MULTIPLIERS.items()
    }
    component_pass_rate = sum(item["passed"] for item in component_cost_stress.values()) / len(
        component_cost_stress
    )
    dates = pd.DatetimeIndex(full_result.report.index).tz_localize(None)
    window = int(config.get("rolling_window_days", 252))
    step = int(config.get("rolling_step_days", 63))
    rolling: list[dict[str, Any]] = []
    for offset in range(0, max(0, len(dates) - window + 1), step):
        selected = dates[offset : offset + window]
        if len(selected) != window:
            continue
        result = runner(selected[0].date().isoformat(), selected[-1].date().isoformat(), schedule)
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
        start_holdings, state_fill_count = _holdings_before(full_result.fills, start)
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
            "passed": len(robustness) == 4
            and robustness_pass_rate >= float(config.get("min_robustness_pass_rate", 1.0)),
            "minimum_pass_rate": float(config.get("min_robustness_pass_rate", 1.0)),
            "scenarios": robustness,
        },
        "double_cost": robustness["double_cost"],
        "component_cost_stress": {
            "scenario_count": len(component_cost_stress),
            "pass_rate": component_pass_rate,
            "passed": len(component_cost_stress) == len(COMPONENT_COST_STRESS_MULTIPLIERS)
            and component_pass_rate >= float(config.get("min_component_stress_pass_rate", 1.0)),
            "minimum_pass_rate": float(config.get("min_component_stress_pass_rate", 1.0)),
            "scenarios": component_cost_stress,
        },
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
