from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from quant_data.execution_data import (
    MINUTE_FIELDS,
    NEWS_FIELDS,
    minute_specs,
    news_specs,
    validate_and_normalize,
)
from quant_data.models import ProviderResult
from quant_data.partitioning import split_partition_spec
from quant_data.provider import ProviderError

pytestmark = pytest.mark.no_database


def _trading_dates(count: int) -> list[str]:
    values = []
    cursor = date(2024, 1, 2)
    while len(values) < count:
        if cursor.weekday() < 5:
            values.append(cursor.strftime("%Y%m%d"))
        cursor += timedelta(days=1)
    return values


def test_a_share_five_minute_uses_a_150_session_budget() -> None:
    dates = _trading_dates(151)

    first = minute_specs(
        {"ashare_5m": ["600000.SH"]},
        start=datetime.strptime(dates[0], "%Y%m%d").date(),
        end=datetime.strptime(dates[129], "%Y%m%d").date(),
        trading_dates=dates[:130],
        max_attempts=3,
        freq="5min",
    )
    second = minute_specs(
        {"ashare_5m": ["600000.SH"]},
        start=datetime.strptime(dates[0], "%Y%m%d").date(),
        end=datetime.strptime(dates[-1], "%Y%m%d").date(),
        trading_dates=dates,
        max_attempts=3,
        freq="5min",
    )

    assert len(first) == 1
    assert len(second) == 2
    assert second[0].params["end_date"].startswith(dates[149][:4] + "-")


def test_minute_cap_bisects_dates_without_overlap() -> None:
    spec = minute_specs(
        {"ashare_5m": ["600000.SH"]},
        start=date(2024, 1, 2),
        end=date(2024, 1, 5),
        trading_dates=["20240102", "20240103", "20240104", "20240105"],
        max_attempts=3,
        freq="5min",
    )[0]
    row = {
        "ts_code": "600000.SH",
        "trade_time": "2024-01-02 09:35:00",
        "open": 10,
        "close": 10,
        "high": 10,
        "low": 10,
        "vol": 1,
        "amount": 1,
    }

    uncapped_rows = [
        {
            **row,
            "trade_time": (
                datetime(2024, 1, 2) + timedelta(seconds=index)
            ).strftime("%Y-%m-%d %H:%M:%S"),
        }
        for index in range(7_999)
    ]
    normalized = validate_and_normalize(
        spec,
        ProviderResult("stk_mins", list(MINUTE_FIELDS), uncapped_rows, b"{}"),
    )
    assert len(normalized.rows) == 7_999

    with pytest.raises(ProviderError, match="8000-row limit"):
        validate_and_normalize(
            spec,
            ProviderResult("stk_mins", list(MINUTE_FIELDS), [row] * 8_000, b"{}"),
        )

    left, right = split_partition_spec(spec)
    assert left.params["start_date"] == "2024-01-02 00:00:00"
    assert left.params["end_date"] == "2024-01-03 23:59:59"
    assert right.params["start_date"] == "2024-01-04 00:00:00"
    assert right.params["end_date"] == "2024-01-05 23:59:59"


def test_single_day_minute_cap_fails_explicitly() -> None:
    spec = minute_specs(
        {"ashare_5m": ["600000.SH"]},
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
        freq="5min",
    )[0]

    with pytest.raises(RuntimeError, match="single-day"):
        split_partition_spec(spec)


def test_news_cap_bisects_to_adjacent_seconds() -> None:
    spec = news_specs(
        date(2024, 1, 2), date(2024, 1, 2), sources=["sina"], max_attempts=3
    )[0]
    row = {
        "datetime": "2024-01-02 10:00:00",
        "content": "update",
        "title": "headline",
        "channels": "finance",
    }

    normalized = validate_and_normalize(
        spec,
        ProviderResult("news", list(NEWS_FIELDS), [row] * 1_499, b"{}"),
    )
    assert len(normalized.rows) == 1
    with pytest.raises(ProviderError, match="1500-row limit"):
        validate_and_normalize(
            spec,
            ProviderResult("news", list(NEWS_FIELDS), [row] * 1_500, b"{}"),
        )

    left, right = split_partition_spec(spec)
    assert left.params["end_date"] == "2024-01-02 11:59:59"
    assert right.params["start_date"] == "2024-01-02 12:00:00"


def test_single_second_news_cap_fails_explicitly() -> None:
    parent = news_specs(
        date(2024, 1, 2), date(2024, 1, 2), sources=["sina"], max_attempts=3
    )[0]
    instant = datetime(2024, 1, 2, 10, 0, 0)
    from quant_data.partitioning import resize_partition_spec

    spec = resize_partition_spec(parent, instant, instant)
    with pytest.raises(RuntimeError, match="single-second"):
        split_partition_spec(spec)
