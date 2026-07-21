from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_data.style_exposure_panel import (
    LIQUIDITY_LOOKBACK_DAYS,
    MOMENTUM_LOOKBACK_DAYS,
    MOMENTUM_SKIP_DAYS,
    build_adjusted_close,
    build_raw_style_panel,
    parse_trade_dates,
)

pytestmark = pytest.mark.no_database


def test_parse_trade_dates_mixed_formats() -> None:
    parsed = parse_trade_dates(
        pd.Series(["20240131", "2024-02-29", 20240329, pd.Timestamp("2024-04-30")])
    )
    assert parsed.tolist() == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-29"),
        pd.Timestamp("2024-03-29"),
        pd.Timestamp("2024-04-30"),
    ]


def _daily_basic(days: int, *, start: str = "2022-01-03") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=days)
    return pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "trade_date": dates,
            "total_mv": 1_000_000.0,
            "circ_mv": 800_000.0,
            "pb": 2.0,
            "pe_ttm": 10.0,
            "turnover_rate": 2.0,
        }
    )


def _geometric_close(days: int, growth: float = 1.02) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=days)
    return pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "trade_date": dates,
            "adj_close": growth ** np.arange(days),
        }
    )


def test_value_and_size_descriptors() -> None:
    panel = build_raw_style_panel(_daily_basic(5))
    row = panel.iloc[-1]
    assert row["log_market_cap"] == pytest.approx(np.log(1_000_000.0))
    assert row["float_market_cap"] == pytest.approx(800_000.0)
    assert row["book_to_price"] == pytest.approx(0.5)
    assert row["earnings_to_price"] == pytest.approx(0.1)


def test_momentum_skips_recent_month() -> None:
    days = MOMENTUM_LOOKBACK_DAYS + 10
    panel = build_raw_style_panel(_daily_basic(days), _geometric_close(days))
    momentum = panel["momentum"].iloc[-1]
    expected = 1.02 ** (MOMENTUM_LOOKBACK_DAYS - MOMENTUM_SKIP_DAYS) - 1.0
    assert momentum == pytest.approx(expected, rel=1e-9)
    # Not enough history: the first rows cannot have a momentum value.
    assert panel["momentum"].iloc[: MOMENTUM_LOOKBACK_DAYS].isna().all()


def test_volatility_of_constant_returns_is_zero() -> None:
    days = 130
    panel = build_raw_style_panel(_daily_basic(days), _geometric_close(days))
    assert panel["volatility"].iloc[-1] == pytest.approx(0.0, abs=1e-12)
    # Warm-up: fewer than the minimum observations stays NaN.
    assert panel["volatility"].iloc[:60].isna().all()


def test_liquidity_is_log_mean_turnover() -> None:
    days = LIQUIDITY_LOOKBACK_DAYS + 5
    panel = build_raw_style_panel(_daily_basic(days), _geometric_close(days))
    assert panel["liquidity"].iloc[-1] == pytest.approx(np.log(0.02))


def test_adjusted_close_uses_adjustment_factor() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    daily = pd.DataFrame(
        {"ts_code": "000001.SZ", "trade_date": dates, "close": [10.0, 10.0, 10.0]}
    )
    factors = pd.DataFrame(
        {"ts_code": "000001.SZ", "trade_date": dates, "adj_factor": [1.0, 1.0, 2.0]}
    )
    adjusted = build_adjusted_close(daily, factors)
    assert adjusted["adj_close"].tolist() == pytest.approx([10.0, 10.0, 20.0])


def test_fundamentals_respect_strictly_after_announcement() -> None:
    days = 12
    basic = _daily_basic(days, start="2024-03-01")
    dates = pd.bdate_range("2024-03-01", periods=days)
    fina = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": dates[3].strftime("%Y%m%d"),
                "roe": 10.0,
                "or_yoy": 20.0,
                "netprofit_yoy": 15.0,
                "debt_to_assets": 55.0,
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": dates[8].strftime("%Y-%m-%d"),
                "roe": 99.0,  # revised value, not visible before its announcement
                "or_yoy": 21.0,
                "netprofit_yoy": 16.0,
                "debt_to_assets": 56.0,
            },
        ]
    )
    panel = build_raw_style_panel(basic, fina_indicator=fina)
    by_date = panel.set_index("trade_date")
    # Same-day announcement is NOT available (strictly after ann_date).
    assert pd.isna(by_date.loc[dates[3], "roe"])
    assert by_date.loc[dates[4], "roe"] == pytest.approx(10.0)
    assert by_date.loc[dates[8], "roe"] == pytest.approx(10.0)
    assert by_date.loc[dates[9], "roe"] == pytest.approx(99.0)
    assert by_date.loc[dates[4], "revenue_yoy"] == pytest.approx(20.0)
    assert by_date.loc[dates[4], "netprofit_yoy"] == pytest.approx(15.0)
    assert by_date.loc[dates[4], "debt_to_assets"] == pytest.approx(55.0)


def test_missing_optional_sources_leave_nan() -> None:
    panel = build_raw_style_panel(_daily_basic(30))
    assert panel[["momentum", "volatility", "liquidity"]].isna().all().all()
    assert panel[["revenue_yoy", "netprofit_yoy", "roe", "debt_to_assets"]].isna().all().all()
