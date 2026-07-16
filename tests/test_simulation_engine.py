from datetime import date

import pandas as pd
import pytest

from quant_platform.cost_model import CostModelConfig
from quant_platform.simulation_engine import execute_simulation_day

pytestmark = pytest.mark.no_database


def _bars(
    *,
    instrument: str = "SH600000",
    price: float = 10.0,
    volume: int = 1_000_000,
    paused: int = 0,
    up_limit: float = 11.0,
    down_limit: float = 9.0,
    day: str = "2025-01-03",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "datetime": f"{day} 13:30:00",
                "instrument": instrument,
                "close": price,
                "vwap": price,
                "volume": volume,
                "paused": paused,
                "up_limit": up_limit,
                "down_limit": down_limit,
            }
        ]
    )


def _run(**overrides):
    defaults = {
        "trade_date": date(2025, 1, 3),
        "cash": 100_000.0,
        "prior_nav": 100_000.0,
        "high_water_mark": 100_000.0,
        "positions": {},
        "target_weights": {"SH600000": 0.50},
        "minute_bars": _bars(),
        "closing_prices": {
            "SH600000": {"price": 10.0, "market_date": date(2025, 1, 3)}
        },
        "cost_model": CostModelConfig(),
        "execution_policy": {
            "execution_algorithm": "twap",
            "slice_minutes": 20,
            "max_slices": 1,
            "max_participation": 0.01,
        },
    }
    defaults.update(overrides)
    return execute_simulation_day(**defaults)


def test_buy_is_round_lot_and_locked_until_next_day() -> None:
    result = _run()
    assert result["orders"][0]["requested_quantity"] == 5_000
    assert result["orders"][0]["filled_quantity"] == 5_000
    assert result["positions"]["SH600000"]["quantity"] == 5_000
    assert result["positions"]["SH600000"]["available_quantity"] == 0
    assert result["cash"] >= 0
    assert result["conservation"]["cash_difference"] == pytest.approx(0.0)


def test_fill_uses_auditable_bar_vwap_not_bar_close() -> None:
    bars = _bars(price=10.0)
    bars["vwap"] = 10.05
    result = _run(minute_bars=bars)

    assert result["fills"][0]["price"] == pytest.approx(10.05)


def test_t_plus_one_unlocks_and_full_liquidation_may_sell_odd_lot() -> None:
    result = _run(
        positions={
            "SH600000": {
                "quantity": 105,
                "available_quantity": 0,
                "average_cost": 9.0,
                "last_trade_date": date(2025, 1, 2),
            }
        },
        target_weights={},
    )
    assert result["orders"][0]["side"] == "sell"
    assert result["orders"][0]["filled_quantity"] == 105
    assert "SH600000" not in result["positions"]


def test_same_day_buy_cannot_be_sold_and_creates_auditable_rejection() -> None:
    result = _run(
        positions={
            "SH600000": {
                "quantity": 100,
                "available_quantity": 0,
                "average_cost": 10.0,
                "last_trade_date": date(2025, 1, 3),
            }
        },
        target_weights={},
    )
    assert result["orders"][0]["status"] == "rejected"
    assert result["orders"][0]["reject_reason"] == "t_plus_one_unavailable"
    assert result["positions"]["SH600000"]["quantity"] == 100


def test_cash_guard_reduces_buy_without_negative_balance() -> None:
    result = _run(cash=1_050.0, target_weights={"SH600000": 1.0})
    assert result["orders"][0]["filled_quantity"] == 100
    assert result["cash"] >= 0
    assert result["orders"][0]["status"] == "partial_filled_expired"


@pytest.mark.parametrize(
    ("bars", "reason"),
    [
        (_bars(paused=1), "suspended"),
        (_bars(price=11.0), "limit_up"),
    ],
)
def test_market_controls_reject_unexecutable_buy(bars: pd.DataFrame, reason: str) -> None:
    result = _run(minute_bars=bars)
    assert result["orders"][0]["status"] == "rejected"
    assert reason in result["orders"][0]["reject_reason"]
    assert result["fills"] == []


def test_capacity_shortfall_partially_fills_and_expires_at_day_end() -> None:
    result = _run(minute_bars=_bars(volume=50_000))
    order = result["orders"][0]
    assert order["requested_quantity"] == 5_000
    assert order["filled_quantity"] == 500
    assert order["capacity_fill_ratio"] == pytest.approx(0.10)
    assert order["status"] == "partial_filled_expired"


def test_stale_price_degrades_nav_and_prevents_performance_certification() -> None:
    result = _run(
        target_weights={"SH600000": 0.001},
        positions={
            "SH600000": {
                "quantity": 100,
                "available_quantity": 100,
                "average_cost": 9.0,
                "last_trade_date": date(2025, 1, 2),
                "market_price": 9.5,
                "market_date": date(2025, 1, 2),
            }
        },
        minute_bars=_bars().iloc[0:0],
        closing_prices={},
    )
    assert result["nav_row"]["status"] == "degraded"
    assert result["nav_row"]["performance_certified"] is False
    assert result["positions"]["SH600000"]["stale"] is True


def test_delisted_position_without_cash_liquidation_is_conservatively_zero_valued() -> None:
    result = _run(
        target_weights={"SH600000": 0.001},
        positions={
            "SH600000": {
                "quantity": 100,
                "available_quantity": 100,
                "average_cost": 9.0,
                "last_trade_date": date(2025, 1, 2),
            }
        },
        minute_bars=_bars().iloc[0:0],
        closing_prices={
            "SH600000": {
                "price": 8.0,
                "market_date": date(2025, 1, 3),
                "delisted": True,
                "cash_liquidated": False,
            }
        },
    )
    assert result["positions"]["SH600000"]["market_value"] == 0.0
    assert any(event["event_type"] == "delisted_zero_valuation" for event in result["events"])
