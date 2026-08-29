from __future__ import annotations

from datetime import date

import pytest

from quant_data.history_bounds import (
    PRIMARY_MARKET_HISTORY_DATASETS,
    PRIMARY_MARKET_HISTORY_START,
    clip_history_range,
    history_start_date,
    is_governed_mainland_a_share_code,
    is_mainland_b_share_code,
)

pytestmark = pytest.mark.no_database


def test_report_rc_uses_documented_2010_history_boundary() -> None:
    assert history_start_date("report_rc") == date(2010, 1, 1)
    assert clip_history_range(
        "report_rc", date(2008, 1, 1), date(2026, 8, 3)
    ) == (date(2010, 1, 1), date(2026, 8, 3))


def test_primary_market_planning_starts_after_the_baostock_legacy_period() -> None:
    assert PRIMARY_MARKET_HISTORY_START == date(2016, 1, 1)
    assert PRIMARY_MARKET_HISTORY_DATASETS == {
        "daily",
        "daily_basic",
        "adj_factor",
    }
    for dataset in PRIMARY_MARKET_HISTORY_DATASETS:
        assert history_start_date(dataset) == date(2016, 1, 1)
        assert clip_history_range(
            dataset,
            date(2008, 1, 1),
            date(2026, 8, 3),
        ) == (date(2016, 1, 1), date(2026, 8, 3))


@pytest.mark.parametrize(
    "ts_code",
    [
        "000043.SZ",
        "300114.SZ",
        "600849.SH",
        "688001.SH",
        "920001.BJ",
        "430001.BJ",
    ],
)
def test_governed_a_share_code_families_accept_historical_and_bse_codes(
    ts_code: str,
) -> None:
    assert is_governed_mainland_a_share_code(ts_code) is True


@pytest.mark.parametrize("ts_code", ["200001.SZ", "201872.SZ", "900901.SH"])
def test_governed_a_share_code_families_reject_b_shares(ts_code: str) -> None:
    assert is_mainland_b_share_code(ts_code) is True
    assert is_governed_mainland_a_share_code(ts_code) is False


@pytest.mark.parametrize(
    "ts_code",
    ["600123.HK", "60012.SH", "ABC123.SH", "999999.SH", "700001.SZ"],
)
def test_governed_a_share_code_families_reject_invalid_or_unsupported_codes(
    ts_code: str,
) -> None:
    assert is_governed_mainland_a_share_code(ts_code) is False
