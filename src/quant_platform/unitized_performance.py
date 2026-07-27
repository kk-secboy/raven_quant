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
from math import isfinite
from typing import Any

UNITIZED_PERFORMANCE_VERSION = "unitized-twr-v1"

_XIRR_YEAR_DAYS = 365.2425


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
    base = prior_nav + flow_open
    if base <= 0:
        return _undefined_day("undefined_nonpositive_base")
    daily_return = (nav - flow_close) / base - 1.0
    wealth = prior_wealth * (1.0 + daily_return)
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

    ``points`` are ``(trade_date, investment_wealth)`` in date order. Recovery
    time counts trading days from the max-drawdown trough until the curve
    reaches its prior high again; ``None`` with status ``ongoing`` when the
    curve has not recovered.
    """

    if not points:
        return {"status": "insufficient_evidence"}
    peak_value = float("-inf")
    peak_date: date | None = None
    max_drawdown = 0.0
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
            trough_date = day
            trough_index = index
            trough_prior_peak = peak_value
    result: dict[str, Any] = {
        "observations": len(points),
        "start_date": points[0][0].isoformat(),
        "end_date": points[-1][0].isoformat(),
        "twr": points[-1][1] / points[0][1] - 1.0 if len(points) > 1 else 0.0,
        "max_drawdown": max_drawdown,
        "peak_date": peak_date.isoformat() if peak_date else None,
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
    points.append((terminal[0], float(terminal[1])))
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

    def npv(rate: float) -> float:
        base = 1.0 + rate
        return sum(amount / base**t for t, amount in terms)

    # log-spaced grid over (-1, 1000%]: finds every sign change bracket.
    grid = [-0.9999] + [
        -1.0 + 10 ** (-4 + i * 0.05) for i in range(0, 141)
    ]
    brackets: list[tuple[float, float]] = []
    previous_rate = grid[0]
    previous_value = npv(previous_rate)
    for rate in grid[1:]:
        value = npv(rate)
        if isfinite(value) and previous_value * value < 0:
            brackets.append((previous_rate, rate))
        if isfinite(value):
            previous_rate, previous_value = rate, value
    if not brackets:
        return {
            "status": "undefined_no_root",
            "rate": None,
            "observations": len(points),
            "contract_version": UNITIZED_PERFORMANCE_VERSION,
        }
    roots: list[float] = []
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
    # 多解退化：取经济上最相关（最接近零）的根并如实标注。
    rate = min(roots, key=abs)
    return {
        "status": "ok" if len(roots) == 1 else "multiple_roots",
        "rate": rate,
        "root_count": len(roots),
        "observations": len(points),
        "contract_version": UNITIZED_PERFORMANCE_VERSION,
    }
