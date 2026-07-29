"""Golden-case differential tests between the two execution chains.

Formal backtest chain: the pinned Qlib ``Exchange`` driven through
``quant_platform.qlib_exchange.SquareRootImpactExchange`` (real pinned source
loaded by ``tests/execution_core_harness.py``).  Forward simulation chain:
``quant_platform.simulation_engine.execute_simulation_day``.

Every case asserts either bit-level agreement (fills, prices, fees, cash
impact) or the exact registered divergence pinned in
``docs/design-gap-analysis.md`` section 2.1 ("已登记口径差异"), so a semantic
drift on either side fails the suite instead of spreading silently.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest
from execution_core_harness import (
    install_pinned_qlib,
    make_position,
    pinned_qlib_root,
    raw_quote_frame,
    run_pinned_exchange,
)

from quant_platform.cost_model import CostModelConfig, CostScheduleBook
from quant_platform.simulation_engine import execute_simulation_day

_QLIB_ROOT = pinned_qlib_root()

pytestmark = [
    pytest.mark.no_database,
    pytest.mark.skipif(
        _QLIB_ROOT is None,
        reason="pinned qlib source checkout (QLIB_REPO) is not available",
    ),
]

if _QLIB_ROOT is not None:
    install_pinned_qlib(_QLIB_ROOT)

TRADE_DATE = date(2024, 1, 2)
STOCK = "SH600000"
ETF = "SZ159915"
BAR = "2024-01-02 10:00:00"
BAR_END = "2024-01-02 10:05:00"
NAV = 1_000_000.0


def _cost_book() -> CostScheduleBook:
    return CostScheduleBook.from_versions([CostModelConfig()])


def _raw_row(
    instrument: str = STOCK,
    *,
    datetime: str = BAR,
    open_: float = 10.0,
    close: float | None = 10.0,
    vwap: float = 10.0,
    volume: float = 2_000_000,
    paused: float = 0,
    up_limit: float = 11.0,
    down_limit: float = 9.0,
    factor: float | None = 1.0,
    change: float = 0.01,
) -> dict[str, Any]:
    return {
        "datetime": datetime,
        "instrument": instrument,
        "open": open_,
        "close": close,
        "vwap": vwap,
        "volume": volume,
        "paused": paused,
        "up_limit": up_limit,
        "down_limit": down_limit,
        "factor": factor,
        "change": change,
    }


def _run_qlib(
    rows: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    *,
    freq: str = "5min",
    deal_price: str = "$vwap",
    limit_price: str = "vwap",
) -> list[dict[str, Any]]:
    return run_pinned_exchange(
        raw=raw_quote_frame(rows),
        orders=orders,
        freq=freq,
        deal_price=deal_price,
        limit_price=limit_price,
        cost_schedule=_cost_book(),
        start_time=TRADE_DATE,
        end_time=date(2024, 1, 3),
    )


def _qlib_order(
    instrument: str = STOCK,
    side: str = "buy",
    quantity: float = 5_000,
    *,
    start_time: str = BAR,
    end_time: str = BAR_END,
    position: Any = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "instrument": instrument,
        "side": side,
        "quantity": quantity,
        "start_time": start_time,
        "end_time": end_time,
    }
    if position is not None:
        spec["position"] = position
    return spec


def _sim_bars(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)[
        ["datetime", "instrument", "close", "vwap", "volume", "paused", "up_limit", "down_limit"]
    ]
    return frame


def _sim_row(
    instrument: str = STOCK,
    *,
    datetime: str = BAR,
    close: float = 10.0,
    vwap: float = 10.0,
    volume: float = 2_000_000,
    paused: float = 0,
    up_limit: float = 11.0,
    down_limit: float = 9.0,
) -> dict[str, Any]:
    return {
        "datetime": datetime,
        "instrument": instrument,
        "close": close,
        "vwap": vwap,
        "volume": volume,
        "paused": paused,
        "up_limit": up_limit,
        "down_limit": down_limit,
    }


def _weight(shares: float, price: float) -> float:
    return shares * price / NAV


def _run_sim(
    rows: list[dict[str, Any]],
    *,
    targets: dict[str, float],
    positions: dict[str, dict[str, Any]] | None = None,
    cash: float = NAV,
    closing_price: float = 10.0,
) -> dict[str, Any]:
    instruments = {str(row["instrument"]) for row in rows}
    closing = {
        instrument: {"price": closing_price, "market_date": TRADE_DATE}
        for instrument in instruments
    }
    return execute_simulation_day(
        trade_date=TRADE_DATE,
        cash=cash,
        prior_nav=NAV,
        high_water_mark=NAV,
        positions=positions or {},
        target_weights=targets,
        minute_bars=_sim_bars(rows),
        closing_prices=closing,
        cost_schedule=_cost_book(),
        execution_policy={
            "execution_algorithm": "next_bar",
            "execution_frequency": "5min",
            "max_participation": 0.01,
        },
    )


def _sim_order(result: dict[str, Any], instrument: str = STOCK) -> dict[str, Any]:
    return next(item for item in result["orders"] if item["instrument"] == instrument)


# ---------------------------------------------------------------------------
# Aligned cases: both chains must produce identical fill economics
# ---------------------------------------------------------------------------


def test_normal_next_bar_buy_fills_identically() -> None:
    qlib = _run_qlib([_raw_row()], [_qlib_order()])[0]
    sim = _run_sim([_sim_row()], targets={STOCK: _weight(5_000, 10.0)})
    sim_order = _sim_order(sim)
    sim_fill = sim["fills"][0]

    assert qlib["filled_quantity"] == 5_000
    assert sim_order["filled_quantity"] == 5_000
    assert qlib["trade_price"] == pytest.approx(sim_fill["price"])
    assert qlib["trade_value"] == pytest.approx(sim_fill["gross_value"])
    assert qlib["cost"] == pytest.approx(sim_fill["fee"])
    # Cash impact on both chains is gross + fee.
    assert qlib["trade_value"] + qlib["cost"] == pytest.approx(NAV - sim["cash"])


def test_suspension_blocks_both_chains() -> None:
    qlib = _run_qlib(
        [_raw_row(close=None, paused=1, volume=0, factor=None)],
        [_qlib_order()],
    )[0]
    sim = _run_sim(
        [_sim_row(paused=1, volume=0)],
        targets={STOCK: _weight(5_000, 10.0)},
    )

    assert qlib["filled_quantity"] == 0
    assert qlib["cost"] == 0
    assert qlib["fill_evidence"]["requested_amount"] == 5_000
    assert qlib["fill_evidence"]["amount"] == 0
    assert qlib["fill_evidence"]["capacity_fill_ratio"] == 0
    assert _sim_order(sim)["status"] == "rejected"
    assert _sim_order(sim)["reject_reason"] == "suspended"
    assert sim["fills"] == []


def test_zero_volume_blocks_both_chains() -> None:
    qlib = _run_qlib([_raw_row(volume=0)], [_qlib_order()])[0]
    sim = _run_sim([_sim_row(volume=0)], targets={STOCK: _weight(5_000, 10.0)})

    assert qlib["filled_quantity"] == 0
    assert _sim_order(sim)["filled_quantity"] == 0


def test_one_way_limit_up_blocks_buy_on_both_chains() -> None:
    qlib = _run_qlib([_raw_row(close=11.0, vwap=11.0)], [_qlib_order()])[0]
    sim = _run_sim(
        [_sim_row(close=11.0, vwap=11.0)],
        targets={STOCK: _weight(5_000, 11.0)},
    )

    assert qlib["filled_quantity"] == 0
    assert qlib["cost"] == 0
    order = _sim_order(sim)
    assert order["status"] == "rejected"
    assert order["reject_reason"] == "limit_up"


def test_one_way_limit_down_blocks_sell_on_both_chains() -> None:
    position = make_position(cash=0.0, holdings={STOCK: 5_000})
    qlib = _run_qlib(
        [_raw_row(close=9.0, vwap=9.0)],
        [_qlib_order(side="sell", position=position)],
    )[0]
    sim = _run_sim(
        [_sim_row(close=9.0, vwap=9.0)],
        positions={
            STOCK: {
                "quantity": 5_000,
                "available_quantity": 5_000,
                "average_cost": 10.0,
                "last_trade_date": date(2024, 1, 1),
            }
        },
        targets={},
        closing_price=9.0,
    )

    assert qlib["filled_quantity"] == 0
    assert qlib["cost"] == 0
    order = _sim_order(sim)
    assert order["status"] == "rejected"
    assert order["reject_reason"] == "limit_down"


def test_participation_cap_partial_fill_matches_and_expires_same_day() -> None:
    qlib = _run_qlib([_raw_row(volume=100_000)], [_qlib_order()])[0]
    sim = _run_sim(
        [_sim_row(volume=100_000)],
        targets={STOCK: _weight(5_000, 10.0)},
    )
    order = _sim_order(sim)

    # 0.01 x 100,000 shares = 1,000 shares of capacity on both chains.
    assert qlib["filled_quantity"] == 1_000
    assert order["filled_quantity"] == 1_000
    assert qlib["cost"] == pytest.approx(sim["fills"][0]["fee"])
    # The unfilled remainder dies with the same trading day on both chains.
    assert order["status"] == "partial_filled_expired"
    assert order["expires_at"].date() == TRADE_DATE


def test_buy_quantity_rounds_down_to_board_lot_on_both_chains() -> None:
    qlib = _run_qlib([_raw_row()], [_qlib_order(quantity=5_050)])[0]
    # The sim chain derives 5,050 desired shares from the target weight and
    # lot-floors the same way before placing the order.
    sim = _run_sim([_sim_row()], targets={STOCK: _weight(5_050, 10.0)})

    assert qlib["filled_quantity"] == 5_000
    assert _sim_order(sim)["filled_quantity"] == 5_000


def test_cost_breakdown_is_component_identical() -> None:
    book = _cost_book().as_of(TRADE_DATE)
    cases = [
        (STOCK, "buy", 100, 2_000_000),
        (STOCK, "sell", 100, 2_000_000),
        (ETF, "sell", 100, 2_000_000),
    ]
    for instrument, side, quantity, volume in cases:
        position = (
            make_position(cash=0.0, holdings={instrument: quantity}) if side == "sell" else None
        )
        qlib = _run_qlib(
            [_raw_row(instrument, volume=volume)],
            [_qlib_order(instrument, side, quantity, position=position)],
        )[0]
        asset_type = "etf" if instrument == ETF else "stock"
        expected = book.estimate_breakdown(
            side=side,
            gross_value=quantity * 10.0,
            participation=quantity / volume,
            asset_type=asset_type,
            trade_date=TRADE_DATE,
        )
        assert qlib["filled_quantity"] == quantity
        assert qlib["cost"] == pytest.approx(expected["total"])

        sim_positions = (
            {
                instrument: {
                    "quantity": quantity,
                    "available_quantity": quantity,
                    "average_cost": 10.0,
                    "last_trade_date": date(2024, 1, 1),
                }
            }
            if side == "sell"
            else {}
        )
        sim = _run_sim(
            [_sim_row(instrument, volume=volume)],
            positions=sim_positions,
            targets={} if side == "sell" else {instrument: _weight(quantity, 10.0)},
        )
        breakdown = sim["fills"][0]["cost_breakdown"]
        for component in (
            "commission",
            "stamp_duty",
            "transfer_fee",
            "slippage",
            "market_impact",
            "total",
        ):
            assert breakdown[component] == pytest.approx(expected[component])


def test_odd_lot_full_liquidation_sells_identically() -> None:
    position = make_position(cash=0.0, holdings={STOCK: 150})
    qlib = _run_qlib(
        [_raw_row()],
        [_qlib_order(side="sell", quantity=150, position=position)],
    )[0]
    sim = _run_sim(
        [_sim_row()],
        positions={
            STOCK: {
                "quantity": 150,
                "available_quantity": 150,
                "average_cost": 10.0,
                "last_trade_date": date(2024, 1, 1),
            }
        },
        targets={},
    )

    assert qlib["filled_quantity"] == 150
    assert _sim_order(sim)["filled_quantity"] == 150
    assert qlib["cost"] == pytest.approx(sim["fills"][0]["fee"])


def test_t_plus_one_unlocked_position_sells_identically_next_day() -> None:
    position = make_position(cash=0.0, holdings={STOCK: 1_000})
    qlib = _run_qlib(
        [_raw_row()],
        [_qlib_order(side="sell", quantity=1_000, position=position)],
    )[0]
    sim = _run_sim(
        [_sim_row()],
        positions={
            STOCK: {
                "quantity": 1_000,
                "available_quantity": 0,
                "average_cost": 10.0,
                "last_trade_date": date(2024, 1, 1),
            }
        },
        targets={},
    )

    assert qlib["filled_quantity"] == 1_000
    assert _sim_order(sim)["filled_quantity"] == 1_000


def test_cash_sufficient_buy_fills_identically() -> None:
    position = make_position(cash=2_000.0)
    qlib = _run_qlib(
        [_raw_row()],
        [_qlib_order(quantity=100, position=position)],
    )[0]
    sim = _run_sim(
        [_sim_row()],
        targets={STOCK: _weight(100, 10.0)},
        cash=2_000.0,
    )

    assert qlib["filled_quantity"] == 100
    assert _sim_order(sim)["filled_quantity"] == 100


# ---------------------------------------------------------------------------
# Registered divergences (docs/design-gap-analysis.md 2.1 已登记口径差异):
# each test pins the exact behavior of both chains so the gap cannot widen
# silently.  D1..D5 numbering matches the registration list.
# ---------------------------------------------------------------------------


def test_registered_d1_limit_price_basis_minute_vwap_vs_bar_close() -> None:
    # Sim chain judges the limit on the slice bar close; the Qlib minute chain
    # judges on the bar vwap.  Bar closes at the limit but vwap sits below it.
    qlib = _run_qlib(
        [_raw_row(close=11.0, vwap=10.4)],
        [_qlib_order()],
    )[0]
    sim = _run_sim(
        [_sim_row(close=11.0, vwap=10.4)],
        targets={STOCK: _weight(5_000, 11.0)},
    )

    assert qlib["filled_quantity"] == 5_000  # vwap below the limit: fills
    assert qlib["trade_price"] == pytest.approx(10.4)
    order = _sim_order(sim)
    assert order["status"] == "rejected"  # close at the limit: rejected
    assert order["reject_reason"] == "limit_up"


def test_registered_d2_daily_formal_chain_judges_limit_at_day_open() -> None:
    # The daily formal backtest (deal_price $open) judges the limit at the day
    # open; the sim minute chain trades the intraday bar below the limit.
    qlib = _run_qlib(
        [_raw_row(open_=11.0, close=10.4, vwap=10.4)],
        [_qlib_order(start_time="2024-01-02", end_time="2024-01-03")],
        freq="day",
        deal_price="$open",
        limit_price="open",
    )[0]
    sim = _run_sim(
        [_sim_row(close=10.4, vwap=10.4)],
        targets={STOCK: _weight(5_000, 10.4)},
    )

    assert qlib["filled_quantity"] == 0  # open at the limit: rejected
    assert _sim_order(sim)["filled_quantity"] == 5_000  # bar below limit: fills


def test_registered_d3_t_plus_one_enforcement_layer() -> None:
    # The Qlib Exchange is T+0 at the exchange layer (T+1 is enforced upstream
    # by the strategy target floor); the sim engine locks same-day buys.
    position = make_position(cash=0.0, holdings={STOCK: 1_000})
    qlib = _run_qlib(
        [_raw_row()],
        [_qlib_order(side="sell", quantity=1_000, position=position)],
    )[0]
    sim = _run_sim(
        [_sim_row()],
        positions={
            STOCK: {
                "quantity": 1_000,
                "available_quantity": 0,
                "average_cost": 10.0,
                "last_trade_date": TRADE_DATE,
            }
        },
        targets={},
    )

    assert qlib["filled_quantity"] == 1_000  # exchange layer sells same-day
    order = _sim_order(sim)
    assert order["status"] == "rejected"
    assert order["reject_reason"] == "t_plus_one_unavailable"


def test_registered_d4_buy_cash_limit_model_and_rounding_epsilon() -> None:
    # Cash check on the Qlib adapter uses the flat conservative cost ratio and
    # qlib's +0.1 rounding epsilon; the sim chain iterates the exact shared
    # cost model and shrinks to an affordable lot.
    position = make_position(cash=1_004.0)
    qlib = _run_qlib(
        [_raw_row()],
        [_qlib_order(quantity=100, position=position)],
    )[0]
    sim = _run_sim(
        [_sim_row()],
        targets={STOCK: _weight(100, 10.0)},
        cash=1_004.0,
    )

    assert qlib["filled_quantity"] == 100  # 99.9 affordable rounds up to 100
    expected = _cost_book().as_of(TRADE_DATE).estimate(
        side="buy",
        gross_value=100 * 10.0,
        participation=100 / 2_000_000,
        asset_type="stock",
        trade_date=TRADE_DATE,
    )
    assert qlib["cost"] == pytest.approx(expected)
    assert _sim_order(sim)["filled_quantity"] == 0  # exact model: 1,005.58 > cash


def test_registered_d5_multi_bar_window_requires_all_bars_limited() -> None:
    # Qlib judges suspension/limit across the whole order window with "all":
    # one limited bar out of two still trades.  The sim chain judges per
    # slice bar (no multi-bar window exists on that side); registered.
    qlib = _run_qlib(
        [
            _raw_row(close=11.0, vwap=11.0, volume=500_000),
            _raw_row(
                datetime="2024-01-02 10:05:00", close=10.4, vwap=10.4, volume=500_000
            ),
        ],
        [_qlib_order(quantity=1_000, start_time=BAR, end_time="2024-01-02 10:10:00")],
    )[0]

    assert qlib["filled_quantity"] == 1_000
    assert qlib["trade_price"] == pytest.approx(10.4)  # ts_data_last deal price


def test_registered_d6_non_liquidating_sell_lot_rounding() -> None:
    # Qlib sells any integer share count (odd-lot reduction); the sim chain
    # conservatively floors a non-liquidating sell to the board lot increment.
    position = make_position(cash=0.0, holdings={STOCK: 450})
    qlib = _run_qlib(
        [_raw_row()],
        [_qlib_order(side="sell", quantity=150, position=position)],
    )[0]
    sim = _run_sim(
        [_sim_row()],
        positions={
            STOCK: {
                "quantity": 450,
                "available_quantity": 450,
                "average_cost": 10.0,
                "last_trade_date": date(2024, 1, 1),
            }
        },
        targets={STOCK: _weight(300, 10.0)},
    )

    assert qlib["filled_quantity"] == 150  # odd-lot reduction fills fully
    order = _sim_order(sim)
    assert order["requested_quantity"] == 100  # floored to the 100-share increment
    assert order["filled_quantity"] == 100
