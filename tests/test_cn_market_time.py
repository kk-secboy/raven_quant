from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from quant_data.planner import CN_MARKET_TIMEZONE, parse_date, today_cn

pytestmark = pytest.mark.no_database


def test_today_cn_uses_shanghai_semantics() -> None:
    # 2024-01-01 16:30 UTC is already 2024-01-02 00:30 in Shanghai.
    assert today_cn(datetime(2024, 1, 1, 16, 30, tzinfo=UTC)) == date(2024, 1, 2)
    assert today_cn(datetime(2024, 1, 1, 15, 59, tzinfo=UTC)) == date(2024, 1, 1)
    # Naive datetimes are interpreted as UTC.
    assert today_cn(datetime(2024, 1, 1, 16, 30)) == date(2024, 1, 2)


def test_today_cn_matches_shanghai_now() -> None:
    assert today_cn() == datetime.now(CN_MARKET_TIMEZONE).date()


def test_parse_date_latest_falls_back_to_market_today() -> None:
    assert parse_date("latest") == today_cn()
    assert parse_date("latest", latest=date(2024, 2, 1)) == date(2024, 2, 1)
