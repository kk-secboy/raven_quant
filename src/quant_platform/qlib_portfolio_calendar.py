from __future__ import annotations

from pathlib import Path
from typing import Any

QLIB_PORTFOLIO_CALENDAR_CONTRACT_VERSION = (
    "qlib-portfolio-calendar-boundary-v1"
)
QLIB_SIGNAL_EXECUTION_CONTRACT_VERSION = "qlib-signal-d-plus-one-v1"


def _read_calendar(path: Path, *, label: str) -> list[str]:
    try:
        values = [
            value.strip()
            for value in path.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
    except OSError as exc:
        raise ValueError(f"Qlib {label} calendar is unavailable") from exc
    if not values or values != sorted(set(values)):
        raise ValueError(f"Qlib {label} calendar must contain ordered unique sessions")
    return values


def resolve_qlib_portfolio_calendar_boundary(
    provider_uri: str | Path,
    *,
    backtest_end: str,
) -> dict[str, Any]:
    """Prove Qlib can represent the closed final daily execution interval.

    Qlib's daily ``TradeCalendarManager`` represents one closed execution bar as
    ``[calendar[i], calendar[i + 1])``.  Consequently a backtest ending on the
    final market-data session still needs one later *calendar boundary*.  That
    boundary is timing metadata only; it must not be confused with another day
    of features or prices.

    A normal provider may satisfy the boundary with a later row in ``day.txt``.
    A pre-final research view instead uses Qlib's official ``day_future.txt``
    mechanism, whose historical prefix must exactly match ``day.txt``.
    """

    provider = Path(provider_uri).resolve()
    calendar_root = provider / "calendars"
    market_calendar = _read_calendar(
        calendar_root / "day.txt",
        label="daily market-data",
    )
    future_path = calendar_root / "day_future.txt"
    if future_path.is_file():
        interval_calendar = _read_calendar(
            future_path,
            label="daily future-boundary",
        )
        boundary_source = "calendars/day_future.txt"
        if interval_calendar[: len(market_calendar)] != market_calendar:
            raise ValueError(
                "Qlib future-boundary calendar changed the market-data calendar prefix"
            )
    else:
        interval_calendar = market_calendar
        boundary_source = "calendars/day.txt"

    if backtest_end not in market_calendar:
        raise ValueError("Qlib portfolio backtest end is outside the market-data calendar")
    try:
        boundary = interval_calendar[interval_calendar.index(backtest_end) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(
            "Qlib portfolio backtest end has no later calendar interval boundary"
        ) from exc
    if boundary <= backtest_end:
        raise ValueError("Qlib portfolio calendar interval boundary is not later")

    return {
        "contract_version": QLIB_PORTFOLIO_CALENDAR_CONTRACT_VERSION,
        "backtest_end": backtest_end,
        "interval_end": boundary,
        "boundary_source": boundary_source,
        "market_data_calendar_end": market_calendar[-1],
        "interval_end_has_market_data": boundary in market_calendar,
    }


def prove_d_plus_one_signal_execution(
    *,
    signal_end_time: Any,
    execution_start_time: Any,
    signal_lag_sessions: int,
) -> dict[str, Any]:
    """Fail closed unless a completed signal interval precedes execution."""

    if signal_lag_sessions != 1:
        raise ValueError("Qlib research portfolio signal lag must be exactly one session")
    if signal_end_time >= execution_start_time:
        raise ValueError("Qlib research portfolio would execute with same-day information")
    return {
        "contract_version": QLIB_SIGNAL_EXECUTION_CONTRACT_VERSION,
        "signal_lag_sessions": signal_lag_sessions,
        "signal_end_time": str(signal_end_time),
        "execution_start_time": str(execution_start_time),
        "no_same_day_or_future_signal": True,
    }
