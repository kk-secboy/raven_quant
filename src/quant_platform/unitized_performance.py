"""Unitized TWR performance curve and money-weighted XIRR (design 4.4/8.3/12.1).

External cash flows (actual deposits/withdrawals) never manufacture return or
drawdown: the daily time-weighted return strips them out via

    r_t = (V_t - F_t_close) / (V_{t-1} + F_t_open) - 1

where ``F_t_open`` is confirmed before the open (investable the same day) and
``F_t_close`` is confirmed after the open (investable from the next day).
The unitized curve ``investment_wealth_t = investment_wealth_{t-1} * (1 + r_t)``
starts at 1.0; drawdown and recovery time are measured on this curve, not on
the CNY balance, which stays the ledger reconciliation view.

A non-positive base (``V_{t-1} + F_t_open <= 0``) or a missing prior chain
state makes the curve *unavailable from that day on* — the design forbids
skipping the day and silently continuing the compounding, so the broken
state propagates until the ledger is rebuilt from the last verified state.

XIRR is the money-weighted companion metric only (personal cash experience,
not strategy alpha evidence). It reports an explicit status instead of
pseudo-precise numbers when the equation is degenerate.
"""

from __future__ import annotations

from datetime import date
from math import isfinite, sqrt
from statistics import fmean, stdev
from typing import Any

UNITIZED_PERFORMANCE_VERSION = "unitized-twr-v3-inception-baseline"

_XIRR_YEAR_DAYS = 365.2425
_TRADING_DAYS_PER_YEAR = 252


def _undefined_day(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "daily_return": None,
        "investment_wealth": None,
        "drawdown": None,
        "high_water_mark": None,
    }


def chain_unitized_day(
    *,
    prior_nav: float,
    nav: float,
    flow_open: float,
    flow_close: float,
    prior_wealth: float | None,
    prior_high_water_mark: float | None,
) -> dict[str, Any]:
    """Chain one day of the unitized TWR curve (design 4.4 F_t_open/F_t_close)."""

    if prior_wealth is None or prior_high_water_mark is None:
        # 链已断裂：不得跳过当日继续连乘，必须从最后已验证状态重建。
        return _undefined_day("unavailable_broken_chain")
    values = (
        prior_nav,
        nav,
        flow_open,
        flow_close,
        prior_wealth,
        prior_high_water_mark,
    )
    if any(not isfinite(float(value)) for value in values):
        raise ValueError("unitized performance inputs must be finite")
    if (
        prior_nav < 0
        or nav < 0
        or prior_wealth < 0
        or prior_high_water_mark <= 0
        or prior_high_water_mark + 1e-12 < prior_wealth
    ):
        raise ValueError("unitized NAV and wealth state is invalid")
    base = prior_nav + flow_open
    if base <= 0:
        return _undefined_day("undefined_nonpositive_base")
    daily_return = (nav - flow_close) / base - 1.0
    wealth = prior_wealth * (1.0 + daily_return)
    if not isfinite(daily_return) or not isfinite(wealth) or wealth < -1e-12:
        raise ValueError("unitized performance would create invalid negative wealth")
    wealth = max(0.0, wealth)
    high_water_mark = max(prior_high_water_mark, wealth)
    return {
        "status": "ok",
        "daily_return": daily_return,
        "investment_wealth": wealth,
        "drawdown": wealth / high_water_mark - 1.0,
        "high_water_mark": high_water_mark,
    }


def unitized_drawdown_recovery(
    points: list[tuple[date, float]],
) -> dict[str, Any]:
    """Max drawdown and recovery time on the unitized wealth curve.

    ``points`` are end-of-day ``(trade_date, investment_wealth)`` observations
    in date order.  The unitized account is defined to start at 1.0 immediately
    before the first observation, so total return and first-day drawdown must
    retain that inception baseline.  Recovery time counts trading days from
    the max-drawdown trough until the curve reaches its prior high again;
    ``None`` with status ``ongoing`` when the curve has not recovered.
    """

    if not points:
        return {"status": "insufficient_evidence"}
    normalized = [(day, float(wealth)) for day, wealth in points]
    if (
        any(not isfinite(wealth) or wealth < 0 for _, wealth in normalized)
        or any(
            previous_wealth == 0 and current_wealth > 0
            for (_, previous_wealth), (_, current_wealth) in zip(
                normalized, normalized[1:], strict=False
            )
        )
        or any(
            current_day <= previous_day
            for (previous_day, _), (current_day, _) in zip(
                normalized, normalized[1:], strict=False
            )
        )
    ):
        raise ValueError(
            "unitized wealth points must be finite, non-negative and strictly ordered"
        )
    points = normalized
    # Every persisted investment_wealth value is chained from the 1.0
    # inception unit.  Omitting that point would drop the first day's return
    # from TWR and hide a first-day loss from max drawdown.
    peak_value = 1.0
    peak_date: date | None = None
    max_drawdown = 0.0
    drawdown_peak_date: date | None = None
    trough_date: date | None = None
    trough_index: int | None = None
    trough_prior_peak: float | None = None
    for index, (day, wealth) in enumerate(points):
        if wealth >= peak_value:
            peak_value = wealth
            peak_date = day
        drawdown = wealth / peak_value - 1.0
        if drawdown < max_drawdown:
            max_drawdown = drawdown
            drawdown_peak_date = peak_date
            trough_date = day
            trough_index = index
            trough_prior_peak = peak_value
    result: dict[str, Any] = {
        "observations": len(points),
        "start_date": points[0][0].isoformat(),
        "end_date": points[-1][0].isoformat(),
        "twr": points[-1][1] - 1.0,
        "max_drawdown": max_drawdown,
        "peak_date": (
            drawdown_peak_date.isoformat()
            if drawdown_peak_date
            else (peak_date.isoformat() if trough_index is None and peak_date else None)
        ),
        "trough_date": trough_date.isoformat() if trough_date else None,
        "recovery_date": None,
        "recovery_trading_days": None,
        "status": "ok",
        "contract_version": UNITIZED_PERFORMANCE_VERSION,
    }
    if trough_index is None or trough_prior_peak is None:
        # 曲线从未跌破前高：无回撤，恢复期为零。
        result["recovery_trading_days"] = 0
        return result
    for index in range(trough_index + 1, len(points)):
        day, wealth = points[index]
        if wealth >= trough_prior_peak:
            result["recovery_date"] = day.isoformat()
            result["recovery_trading_days"] = index - trough_index
            return result
    result["status"] = "ongoing"
    return result


def unitized_return_statistics(
    points: list[tuple[date, float, float]],
    *,
    cash_return_annual: float = 0.0,
    minimum_acceptable_return_annual: float = 0.0,
    annualization_factor: int = _TRADING_DAYS_PER_YEAR,
) -> dict[str, Any]:
    """Compute account return/risk statistics from the certified TWR chain.

    Each point is ``(trade_date, investment_wealth, twr_daily_return)``.  The
    chain is independently reconciled from the 1.0 inception unit before any
    statistic is reported.  Undefined denominators remain ``None`` with an
    explicit per-metric status; they are never rendered as zero or infinity.
    """

    if not points:
        return {
            "status": "insufficient_evidence",
            "observations": 0,
            "contract_version": UNITIZED_PERFORMANCE_VERSION,
        }
    if annualization_factor <= 1:
        raise ValueError("annualization_factor must be greater than one")
    if (
        not isfinite(float(cash_return_annual))
        or float(cash_return_annual) <= -1.0
        or not isfinite(float(minimum_acceptable_return_annual))
        or float(minimum_acceptable_return_annual) <= -1.0
    ):
        raise ValueError("annual return assumptions must be finite and greater than -1")
    normalized = [
        (day, float(wealth), float(daily_return))
        for day, wealth, daily_return in points
    ]
    if any(
        not isfinite(wealth)
        or wealth < 0
        or not isfinite(daily_return)
        or daily_return < -1.0
        for _, wealth, daily_return in normalized
    ) or any(
        current_day <= previous_day
        for (previous_day, _, _), (current_day, _, _) in zip(
            normalized, normalized[1:], strict=False
        )
    ):
        raise ValueError(
            "unitized return points must be finite, valid and strictly ordered"
        )
    prior_wealth = 1.0
    for day, wealth, daily_return in normalized:
        expected = prior_wealth * (1.0 + daily_return)
        tolerance = 1e-8 * max(1.0, abs(expected), abs(wealth))
        if abs(wealth - expected) > tolerance:
            raise ValueError(
                f"unitized return chain does not reconcile on {day.isoformat()}"
            )
        prior_wealth = wealth

    observations = len(normalized)
    daily_returns = [daily_return for _, _, daily_return in normalized]
    elapsed_days = (normalized[-1][0] - normalized[0][0]).days
    total_return = normalized[-1][1] - 1.0
    metric_status: dict[str, str] = {"twr": "ok"}
    cagr: float | None
    if observations < 2 or elapsed_days <= 0:
        cagr = None
        metric_status["cagr"] = "insufficient_evidence"
    elif normalized[-1][1] == 0.0:
        cagr = -1.0
        metric_status["cagr"] = "ok"
    else:
        cagr = normalized[-1][1] ** (_XIRR_YEAR_DAYS / elapsed_days) - 1.0
        metric_status["cagr"] = "ok"

    annualized_volatility: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    if observations < 2:
        metric_status.update(
            {
                "annualized_volatility": "insufficient_evidence",
                "sharpe_ratio": "insufficient_evidence",
                "sortino_ratio": "insufficient_evidence",
            }
        )
    else:
        daily_std = stdev(daily_returns)
        annualized_volatility = daily_std * sqrt(float(annualization_factor))
        metric_status["annualized_volatility"] = "ok"
        cash_daily = (1.0 + float(cash_return_annual)) ** (
            1.0 / annualization_factor
        ) - 1.0
        excess = [value - cash_daily for value in daily_returns]
        if daily_std <= 1e-15:
            metric_status["sharpe_ratio"] = "undefined_zero_variance"
        else:
            sharpe_ratio = (
                fmean(excess) / daily_std * sqrt(float(annualization_factor))
            )
            metric_status["sharpe_ratio"] = "ok"
        minimum_daily = (1.0 + float(minimum_acceptable_return_annual)) ** (
            1.0 / annualization_factor
        ) - 1.0
        differences = [value - minimum_daily for value in daily_returns]
        downside_deviation = sqrt(
            fmean(min(value, 0.0) ** 2 for value in differences)
        )
        if downside_deviation <= 1e-15:
            metric_status["sortino_ratio"] = "undefined_zero_downside_deviation"
        else:
            sortino_ratio = (
                fmean(differences)
                / downside_deviation
                * sqrt(float(annualization_factor))
            )
            metric_status["sortino_ratio"] = "ok"

    return {
        "status": "ok" if observations >= 2 else "insufficient_evidence",
        "observations": observations,
        "start_date": normalized[0][0].isoformat(),
        "end_date": normalized[-1][0].isoformat(),
        "elapsed_calendar_days": elapsed_days,
        "twr": total_return,
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "metric_status": metric_status,
        "assumptions": {
            "annualization_factor": annualization_factor,
            "cash_return_annual": float(cash_return_annual),
            "minimum_acceptable_return_annual": float(
                minimum_acceptable_return_annual
            ),
            "benchmark_status": "not_configured",
        },
        "contract_version": UNITIZED_PERFORMANCE_VERSION,
    }


def xirr(
    flows: list[tuple[date, float]],
    *,
    terminal: tuple[date, float],
) -> dict[str, Any]:
    """Money-weighted annualized return (XIRR) with explicit degenerate states.

    Amounts are signed from the investor's perspective: deposits into the
    account are negative, withdrawals and the terminal liquidation value are
    positive. Solves ``sum(a_i / (1 + r) ** t_i) = 0`` for ``r > -1`` by grid
    bracketing plus bisection; reports ``undefined_*``/``multiple_roots``
    statuses instead of pseudo-precise numbers.
    """

    points = [(day, float(amount)) for day, amount in flows]
    terminal_value = float(terminal[1])
    if not isfinite(terminal_value) or terminal_value < 0:
        raise ValueError("XIRR terminal value must be finite and non-negative")
    if points and terminal[0] < max(day for day, _ in points):
        raise ValueError("XIRR terminal date cannot precede a cash flow")
    points.append((terminal[0], terminal_value))
    if any(not isfinite(amount) for _, amount in points):
        raise ValueError("XIRR cash flows must be finite")
    if len(points) < 2:
        return {"status": "insufficient_evidence", "rate": None}
    has_positive = any(amount > 0 for _, amount in points)
    has_negative = any(amount < 0 for _, amount in points)
    if not (has_positive and has_negative):
        return {"status": "undefined_single_sign", "rate": None}
    ordered = sorted(points, key=lambda item: item[0])
    origin = ordered[0][0]
    terms = [
        ((day - origin).days / _XIRR_YEAR_DAYS, amount) for day, amount in ordered
    ]
    if all(t == 0.0 for t, _ in terms):
        # With no elapsed time the equation is rate-independent. Even when
        # amounts net to zero there are infinitely many roots, not a meaningful
        # annualized return.
        return {
            "status": "undefined_no_elapsed_time",
            "rate": None,
            "observations": len(points),
            "contract_version": UNITIZED_PERFORMANCE_VERSION,
        }

    def npv(rate: float) -> float:
        base = 1.0 + rate
        return sum(amount / base**t for t, amount in terms)

    # Exact grid roots must be retained: zero-return cash flows have r=0
    # exactly, and a strict sign-change-only search would incorrectly report
    # that ordinary case as having no solution.
    grid = [-0.9999] + [
        -1.0 + 10 ** (-4 + i * 0.05) for i in range(0, 141)
    ]
    scale = max(1.0, sum(abs(amount) for _, amount in points))
    zero_tolerance = 1e-12 * scale
    brackets: list[tuple[float, float]] = []
    roots: list[float] = []
    finite_grid: list[tuple[float, float]] = []
    for rate in grid:
        value = npv(rate)
        if isfinite(value):
            finite_grid.append((rate, value))
            if abs(value) <= zero_tolerance:
                roots.append(rate)
    for (previous_rate, previous_value), (rate, value) in zip(
        finite_grid, finite_grid[1:], strict=False
    ):
        if (
            abs(previous_value) > zero_tolerance
            and abs(value) > zero_tolerance
            and previous_value * value < 0
        ):
            brackets.append((previous_rate, rate))
    if not brackets and not roots:
        return {
            "status": "undefined_no_root",
            "rate": None,
            "observations": len(points),
            "contract_version": UNITIZED_PERFORMANCE_VERSION,
        }
    for low, high in brackets:
        for _ in range(200):
            middle = (low + high) / 2.0
            value = npv(middle)
            if value == 0.0 or high - low < 1e-12:
                break
            if npv(low) * value < 0:
                high = middle
            else:
                low = middle
        roots.append((low + high) / 2.0)
    roots = sorted(roots)
    unique_roots: list[float] = []
    for root in roots:
        if not unique_roots or abs(root - unique_roots[-1]) > 1e-9:
            unique_roots.append(root)
    # 多解退化：取经济上最相关（最接近零）的根并如实标注。
    rate = min(unique_roots, key=abs)
    return {
        "status": "ok" if len(unique_roots) == 1 else "multiple_roots",
        "rate": rate,
        "root_count": len(unique_roots),
        "observations": len(points),
        "contract_version": UNITIZED_PERFORMANCE_VERSION,
    }
