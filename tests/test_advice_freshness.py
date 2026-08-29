from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from quant_platform.advice_freshness import (
    advice_session_requirement,
    assess_netting_plan_freshness,
    assess_signal_freshness,
    latest_closed_trading_day,
)

pytestmark = pytest.mark.no_database

_SHANGHAI = ZoneInfo("Asia/Shanghai")


def test_latest_closed_trading_day_excludes_open_session_before_close() -> None:
    days = [date(2026, 8, 28), date(2026, 8, 31)]

    assert latest_closed_trading_day(
        days,
        now=datetime(2026, 8, 31, 14, 59, tzinfo=_SHANGHAI),
    ) == date(2026, 8, 28)
    assert latest_closed_trading_day(
        days,
        now=datetime(2026, 8, 31, 15, 0, tzinfo=_SHANGHAI),
    ) == date(2026, 8, 31)


def test_calendar_unavailable_blocks_formal_advice(tmp_path) -> None:
    requirement = advice_session_requirement(
        tmp_path,
        now=datetime(2026, 8, 31, 16, 0, tzinfo=_SHANGHAI),
    )

    assert requirement["status"] == "blocked"
    assert requirement["required_signal_date"] is None
    assert "交易日历不可用" in requirement["reason"]


def test_signal_freshness_requires_exact_latest_closed_session() -> None:
    stale = assess_signal_freshness(
        required_signal_date=date(2026, 8, 28),
        observed_signal_date="2026-08-27",
    )
    current = assess_signal_freshness(
        required_signal_date=date(2026, 8, 28),
        observed_signal_date="2026-08-28",
    )

    assert stale["status"] == "stale"
    assert stale["passed"] is False
    assert current["status"] == "current"
    assert current["passed"] is True


def test_netting_plan_rejects_stale_member_hidden_by_current_max_date() -> None:
    freshness = assess_netting_plan_freshness(
        required_signal_date=date(2026, 8, 28),
        inputs_as_of=date(2026, 8, 28),
        plan={
            "input_evidence": {
                "member_snapshots": {
                    "short": {"as_of_date": "2026-08-28"},
                    "swing": {"as_of_date": "2026-08-27"},
                }
            }
        },
    )

    assert freshness["status"] == "member_snapshots_stale"
    assert freshness["passed"] is False
    assert freshness["member_signal_dates"] == {
        "short": "2026-08-28",
        "swing": "2026-08-27",
    }


def test_netting_plan_accepts_only_current_member_evidence() -> None:
    freshness = assess_netting_plan_freshness(
        required_signal_date=date(2026, 8, 28),
        inputs_as_of="2026-08-28",
        plan={
            "input_evidence": {
                "member_snapshots": {
                    "short": {"as_of_date": "2026-08-28"},
                    "swing": {"as_of_date": "2026-08-28"},
                    "long": {"as_of_date": "2026-08-28"},
                }
            }
        },
    )

    assert freshness["status"] == "current"
    assert freshness["passed"] is True
