from datetime import UTC, datetime, time

import pytest

from quant_platform.schedule_store import (
    intraday_occurrence_after,
    next_intraday_occurrence,
    validate_intraday_run_time,
)

pytestmark = pytest.mark.no_database


def test_intraday_schedule_created_mid_session_uses_next_completed_bar_today() -> None:
    current = datetime(2026, 7, 29, 2, 2, tzinfo=UTC)  # 10:02 Asia/Shanghai
    result = next_intraday_occurrence(
        current,
        "Asia/Shanghai",
        time(9, 35),
        5,
    )
    assert result == datetime(2026, 7, 29, 2, 5, tzinfo=UTC)


def test_intraday_schedule_created_during_lunch_resumes_at_first_afternoon_close() -> None:
    current = datetime(2026, 7, 29, 4, 0, tzinfo=UTC)  # 12:00 Asia/Shanghai
    result = next_intraday_occurrence(
        current,
        "Asia/Shanghai",
        time(9, 35),
        5,
    )
    assert result == datetime(2026, 7, 29, 5, 5, tzinfo=UTC)


def test_intraday_schedule_created_after_close_starts_next_calendar_day() -> None:
    current = datetime(2026, 7, 29, 7, 1, tzinfo=UTC)  # 15:01 Asia/Shanghai
    result = next_intraday_occurrence(
        current,
        "Asia/Shanghai",
        time(9, 35),
        5,
    )
    assert result == datetime(2026, 7, 30, 1, 35, tzinfo=UTC)


def test_intraday_occurrence_resumes_after_lunch_without_incomplete_bar() -> None:
    scheduled = datetime(2026, 7, 29, 3, 30, tzinfo=UTC)  # 11:30 Asia/Shanghai
    result = intraday_occurrence_after(
        scheduled,
        "Asia/Shanghai",
        time(9, 35),
        5,
    )
    assert result == datetime(2026, 7, 29, 5, 5, tzinfo=UTC)


@pytest.mark.parametrize("run_time", [time(9, 31), time(9, 36), time(13, 1)])
def test_five_minute_schedule_rejects_misaligned_or_incomplete_bars(
    run_time: time,
) -> None:
    with pytest.raises(ValueError, match="align to a completed bar"):
        validate_intraday_run_time(run_time, 5)
