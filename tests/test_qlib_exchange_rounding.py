"""Actual pinned Exchange clipping/rounding and Position updates, with synthetic quotes."""
from __future__ import annotations

import math
import sys
from collections import defaultdict

import pandas as pd
import pytest
from execution_core_harness import (
    HarnessQuote,
    install_pinned_qlib,
    make_position,
    pinned_qlib_root,
)

from quant_platform.cost_model import CostModelConfig
from quant_platform.qlib_backtest import calculate_capacity_fill_ratio

_ROOT = pinned_qlib_root()
pytestmark = [
    pytest.mark.no_database,
    pytest.mark.skipif(_ROOT is None, reason="pinned Qlib absent"),
]


@pytest.fixture
def exchange_case():
    if "qlib.backtest.exchange" not in sys.modules:
        install_pinned_qlib(_ROOT)
    from qlib.backtest.decision import Order
    from test_qlib_physical_orders import _exchange

    def run(
        *, request, factor=1.0, holdings=0.0, side="BUY", volume=1_000_000.0,
        cash=100_000.0, instrument="SZ000001", cost_model=None, freq="day", raw_price=10.0,
    ):
        exchange = _exchange(
            instrument, factor=factor, volume=volume, cost_model=cost_model, raw_price=raw_price,
        )
        if freq != "day":
            # Reuse only the synthetic quote data; construct a genuine minute Exchange.
            exchange = type(exchange)(
                cost_model=cost_model or CostModelConfig(), freq=freq, codes=[instrument],
                start_time=pd.Timestamp("2024-01-03"), end_time=pd.Timestamp("2024-01-04"),
                deal_price="$open", quote_cls=HarnessQuote,
            )
        current = make_position(
            cash=cash, holdings={instrument: holdings} if holdings else {},
        )
        if holdings:
            current.update_stock_price(instrument, raw_price * factor)
        order = Order(
            stock_id=instrument, amount=request, direction=getattr(Order, side),
            start_time=pd.Timestamp("2024-01-03"), end_time=pd.Timestamp("2024-01-04"),
        )
        value, cost, price = exchange.deal_order(
            order, position=current, dealt_order_amount=defaultdict(float),
        )
        assert order.deal_amount <= request
        assert order.deal_amount * factor <= volume * 0.01
        assert value == order.deal_amount * price
        fill = exchange.fill_log[-1]
        assert fill["amount"] == pytest.approx(order.deal_amount)
        assert calculate_capacity_fill_ratio(exchange.fill_log) <= 1.0
        return exchange, current, order, value, cost

    return run


@pytest.mark.parametrize("factor", [1.0, 0.9995])
@pytest.mark.parametrize("side,requested,expected_raw", [
    ("BUY", 100.0, 100.0),
    ("SELL", 99.95, 99.0),
])
def test_requested_amount_is_not_increased_to_next_unit(
    exchange_case, factor, side, requested, expected_raw,
):
    holdings = 1000.0 if side == "SELL" else 0.0
    _exchange, position, order, _value, _cost = exchange_case(
        request=requested, factor=factor, holdings=holdings, side=side,
    )
    if side == "BUY" and factor != 1.0:
        expected_raw = 0.0  # 99.95 execution-day shares cannot form a 100-share order.
    assert order.deal_amount * factor == pytest.approx(expected_raw)
    expected = holdings + order.deal_amount * (1 if side == "BUY" else -1)
    assert position.get_stock_amount("SZ000001") == pytest.approx(expected)


@pytest.mark.parametrize("factor", [1.0, 0.9995])
def test_volume_clip_is_not_rounded_back_up(exchange_case, factor):
    _exchange, position, order, _value, _cost = exchange_case(
        request=200.0 / factor, factor=factor, volume=19_995.0,
    )
    assert order.deal_amount * factor == pytest.approx(100.0)
    assert position.get_stock_amount("SZ000001") == order.deal_amount


def test_cash_clip_is_not_rounded_back_up(exchange_case):
    cost_model = CostModelConfig()
    # Upstream's conservative commission/impact cash bound is the binding bound.
    conservative_rate = (
        cost_model.buy_commission_rate + cost_model.stock_buy_stamp_duty_rate
        + cost_model.conservative_transfer_value_rate() + cost_model.fixed_slippage_rate
        + cost_model.impact_at_max_participation
    )
    gross_bound = 199.95 * 10.0
    cash = gross_bound + max(gross_bound * conservative_rate, cost_model.min_commission)
    exchange, position, order, value, cost = exchange_case(
        request=200.0, cash=cash, cost_model=cost_model,
    )
    assert exchange._get_buy_amount_by_cash_limit(10.0, cash, exchange.open_cost) < 200.0
    assert order.deal_amount == 100.0
    assert value + cost <= cash
    assert position.get_cash() == pytest.approx(cash - value - cost)


@pytest.mark.parametrize("factor", [1.0, 0.8, 0.1260743886232376])
def test_exact_lot_roundtrip_and_adjacent_float_stay_bounded(exchange_case, factor):
    amount = 100.0 / factor
    lower = math.nextafter(amount, 0.0)
    for request in (amount, lower):
        _exchange, _position, order, _value, _cost = exchange_case(
            request=request, factor=factor,
        )
        assert order.deal_amount == request
        assert abs(order.deal_amount * factor - 100.0) <= 4 * math.ulp(100.0)


def test_material_subminimum_is_not_representation_noise(exchange_case):
    _exchange, _position, order, _value, _cost = exchange_case(request=99.999999)
    assert order.deal_amount == 0.0


@pytest.mark.parametrize("instrument,requested,expected", [
    ("SH688001", 201.0, 201.0), ("SH688001", 200.95, 200.0),
    ("SH688001", 199.95, 0.0), ("BJ430047", 100.95, 100.0),
])
def test_board_minimum_and_increment_remain_in_force(
    exchange_case, instrument, requested, expected,
):
    _exchange, _position, order, _value, _cost = exchange_case(
        request=requested, instrument=instrument,
    )
    assert order.deal_amount == expected


def test_fractional_full_liquidation_remains_exact(exchange_case):
    _exchange, position, order, _value, _cost = exchange_case(
        request=10_000.0, holdings=10_000.0, factor=0.100037, side="SELL",
    )
    assert order.deal_amount == 10_000.0
    assert order.deal_amount * order.factor == 1000.37
    assert position.get_stock_amount("SZ000001") == 0.0


@pytest.mark.parametrize("side,requested,volume,expected", [
    ("BUY", 100.0, 1_000_000.0, 100.0),
    ("BUY", 100.0, 9_995.0, 0.0),
    ("SELL", 99.95, 1_000_000.0, 99.0),
])
def test_raw_minute_factor_one_uses_the_same_bounded_contract(
    exchange_case, side, requested, volume, expected,
):
    exchange, _position, order, _value, _cost = exchange_case(
        request=requested, volume=volume, side=side, freq="1min",
        holdings=1000.0 if side == "SELL" else 0.0,
    )
    assert exchange.freq == "1min"
    assert order.factor == 1.0
    assert order.deal_amount == expected


def test_captured_rolling_order_does_not_fill_an_extra_physical_fraction(exchange_case):
    # Read-only c835 replay, SH600276 / 2018-08-28: 3399.981634113734
    # equivalent shares were incorrectly rounded up to 3400 by upstream +0.1.
    factor = 0.1489630490541458
    requested = 22824.32895743086
    adjusted_price = 10.498915672302246
    _exchange, position, order, value, _cost = exchange_case(
        instrument="SH600276", request=requested, factor=factor, cash=1_000_000.0,
        raw_price=adjusted_price / factor,
    )
    assert requested * factor == 3399.981634113734
    assert order.deal_amount == 3300.0 / factor
    assert order.deal_amount < requested
    assert value <= requested * adjusted_price
    assert position.get_stock_amount("SH600276") == order.deal_amount
