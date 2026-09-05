"""Physical-share bridge with the pinned Position, Order and Exchange source.

Only signal/calendar/data-provider plumbing is replaced. PortfolioPolicy and
the quantity order generator run in full; no test-side rounding oracle exists.
"""
from __future__ import annotations

import sys
import types
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
from execution_core_harness import (
    _PROVIDER,
    HarnessQuote,
    install_pinned_qlib,
    make_position,
    pinned_qlib_root,
    raw_quote_frame,
)

from quant_platform.cost_model import CostModelConfig
from quant_platform.portfolio_policy import PortfolioPolicy, PortfolioPolicyConfig

_ROOT = pinned_qlib_root()
pytestmark = [
    pytest.mark.no_database, pytest.mark.skipif(_ROOT is None, reason="pinned Qlib absent"),
]


@pytest.fixture
def bridge(monkeypatch):
    # Keep the genuine Exchange and its SingleData type from the same load.
    # Reinstalling per test would leave the cached platform subclass referring
    # to an older Qlib class while quote plumbing used a new class identity.
    if "qlib.backtest.exchange" not in sys.modules:
        install_pinned_qlib(_ROOT)

    class WeightStrategyBase:
        def __init__(self, *, signal, risk_degree):
            self.signal, self.risk_degree = signal, risk_degree

    module = types.ModuleType("qlib.contrib.strategy.signal_strategy")
    module.WeightStrategyBase = WeightStrategyBase
    monkeypatch.setitem(sys.modules, module.__name__, module)
    from quant_platform.qlib_policy_strategy import create_qlib_policy_strategy

    def create(
        *, instrument="SZ000001", quantity=0.0, factor=0.8, mark=10.0, execution=10.0,
        nav=100_000.0, hold=False, max_weight=0.10, capacity=1_000_000.0,
        signal_date="2024-01-02", **config,
    ):
        amount = quantity / factor
        current = make_position(cash=nav - quantity * mark,
                                holdings={instrument: amount} if quantity else {})
        if quantity:
            current.update_stock_price(instrument, mark * factor)
            current.update_stock_weight(instrument, quantity * mark / current.calculate_value())
        day = pd.Timestamp(signal_date)
        trade_day = day + pd.offsets.BDay()
        trade_end = trade_day + pd.Timedelta(days=1)
        metadata = {
            "prices": pd.Series({instrument: execution}),
            "current_prices": pd.Series({instrument: mark}),
            "qlib_factors": pd.Series({instrument: factor}),
            "average_daily_values": pd.Series({instrument: capacity / 0.01}),
        }
        policy = PortfolioPolicy(PortfolioPolicyConfig(
            topk=1, n_drop=0, max_position_weight=max_weight, max_daily_turnover=1.0,
            rebalance_frequency="month" if hold else "day", **config,
        ))
        strategy = create_qlib_policy_strategy(
            signal=pd.Series({instrument: 1.0}), policy=policy,
            metadata_provider=lambda _when, _instruments: dict(metadata),
        )
        strategy.trade_calendar = types.SimpleNamespace(
            get_trade_step=lambda: 7, get_step_time=lambda _step, shift: (day, day),
        )
        if hold:
            strategy._last_rebalance_signal_time = day
        strategy._holding_age_sessions = {instrument: 1} if quantity else {}

        def decide():
            return strategy.generate_target_weight_position(
                pd.Series({instrument: 1.0}), current, trade_day, trade_end,
            )

        def orders(weights, exchange=None, **overrides):
            kwargs = dict(
                current=current,
                trade_exchange=(
                    exchange or types.SimpleNamespace(is_stock_tradable=lambda **_: True)
                ),
                target_weight_position=weights, risk_degree=strategy.risk_degree,
                pred_start_time=day, pred_end_time=day,
                trade_start_time=trade_day, trade_end_time=trade_end,
            )
            kwargs.update(overrides)
            return strategy.order_generator.generate_order_list_from_target_weight_position(
                **kwargs
            )

        return strategy, current, metadata, decide, orders

    return create


def _exchange(
    instrument, *, factor, raw_price=10.0, day="2024-01-03", volume=1_000_000.0,
    cost_model=None,
):
    from quant_platform.qlib_exchange import SquareRootImpactExchange

    timestamp = pd.Timestamp(day)
    _PROVIDER.raw = raw_quote_frame([{
        "datetime": timestamp, "instrument": instrument,
        "open": raw_price * factor, "close": raw_price * factor,
        "vwap": raw_price * factor, "volume": volume / factor,
        "paused": 0.0, "up_limit": raw_price * factor * 1.1,
        "down_limit": raw_price * factor * 0.9, "factor": factor, "change": 0.0,
    }])
    return SquareRootImpactExchange(
        cost_model=cost_model or CostModelConfig(), freq="day", start_time=timestamp,
        end_time=timestamp + pd.Timedelta(days=1),
        codes=[instrument], deal_price="$open", quote_cls=HarnessQuote,
        limit_threshold=("Or(Gt($paused, 0), Ge($open, $up_limit))",
                         "Or(Gt($paused, 0), Le($open, $down_limit))"),
    )


@pytest.mark.parametrize("quantity,factor,mark,execution", [
    (950.0, 1.0, 10.0, 10.0), (1000.0, 0.8, 10.0, 10.0),
    (1000.0, 1.0, 10.1, 10.0), (950.5, 0.0733, 10.1, 9.8),
])
def test_legal_hold_keeps_exact_internal_amount_and_submits_no_order(
    bridge, quantity, factor, mark, execution,
):
    strategy, current, _metadata, decide, orders = bridge(
        quantity=quantity, factor=factor, mark=mark, execution=execution, hold=True,
        max_weight=0.60,
    )
    weights = decide()
    assert strategy.risk_degree == 1.0
    assert strategy.order_generator.plan["target_amounts"] == current.get_stock_amount_dict()
    assert orders(weights) == []


def test_captured_day_131_position_retains_exact_amount_and_no_order(bridge):
    factor = 0.1260743886232376
    amount = 122150.10652180569
    strategy, current, _metadata, decide, orders = bridge(
        instrument="SZ002308", quantity=amount * factor, factor=factor,
        mark=1.8848121166229248 / factor, execution=1.878508448600769 / factor,
        nav=4_993_121.283154555, max_weight=0.05, min_rebalance_weight_change=0.01,
        signal_date="2015-10-09", capacity=6_924_073.6,
    )
    weights = decide()
    assert current.get_stock_amount("SZ002308") == amount
    assert strategy.order_generator.plan["target_amounts"]["SZ002308"] == amount
    assert orders(weights) == []


@pytest.mark.parametrize("instrument,quantity,weight,direction,raw_amount", [
    ("SZ000001", 950.0, 0.0949, "SELL", 1.0),
    ("SH688001", 0.0, 0.0201, "BUY", 201.0),
    ("BJ430047", 0.0, 0.0101, "BUY", 101.0),
])
def test_board_order_increment_survives_generation_and_real_exchange(
    bridge, instrument, quantity, weight, direction, raw_amount,
):
    factor = 0.8
    _strategy, current, _metadata, decide, orders = bridge(
        instrument=instrument, quantity=quantity, max_weight=weight, factor=factor,
    )
    exchange = _exchange(instrument, factor=factor)
    generated = orders(decide(), exchange)
    assert len(generated) == 1
    order = generated[0]
    assert order.direction == getattr(type(order), direction)
    assert order.amount * factor == pytest.approx(raw_amount)
    _value, _cost, price = exchange.deal_order(
        order, position=current, dealt_order_amount=defaultdict(float),
    )
    assert order.deal_amount * factor == pytest.approx(raw_amount)
    assert price == pytest.approx(10.0 * factor)
    assert exchange.fill_log[0]["raw_amount"] == pytest.approx(raw_amount)
    assert exchange.fill_log[0]["raw_trade_price"] == pytest.approx(10.0)


def test_subminimum_buy_is_no_trade_instead_of_illegal_target(bridge):
    _strategy, _current, _metadata, decide, orders = bridge(max_weight=0.0095)
    assert orders(decide()) == []


def test_trade_date_factor_changes_only_exchange_physical_conversion(bridge):
    _strategy, current, _metadata, decide, orders = bridge(max_weight=0.01, factor=0.8)
    # A 2-for-1 adjustment halves the raw price. The Qlib adjusted price and
    # amount plan remain fixed; no future factor is provided to the policy.
    exchange = _exchange("SZ000001", factor=1.6, raw_price=5.0)
    generated = orders(decide(), exchange)
    assert len(generated) == 1
    assert generated[0].amount == 125.0
    value, _cost, price = exchange.deal_order(
        generated[0], position=current, dealt_order_amount=defaultdict(float),
    )
    assert value == 1000.0
    assert price == 8.0
    assert exchange.fill_log[0]["raw_amount"] == 200.0


def test_dividend_factor_remainder_can_liquidate_exact_original_position(bridge):
    factor = 0.100037
    original_amount = 10_000.0
    _strategy, current, _metadata, decide, orders = bridge(
        quantity=original_amount * factor, factor=factor, max_weight=0.60,
        max_holding_sessions=1,
    )
    exchange = _exchange("SZ000001", factor=factor)
    generated = orders(decide(), exchange)
    assert len(generated) == 1
    assert generated[0].amount == original_amount
    _value, _cost, _price = exchange.deal_order(
        generated[0], position=current, dealt_order_amount=defaultdict(float),
    )
    assert generated[0].deal_amount == original_amount
    assert current.get_stock_list() == []


def test_share_based_transfer_fee_uses_actual_share_quantity_and_preserves_qlib_units(bridge):
    instrument = "SH600000"
    factor = 0.5
    _strategy, current, _metadata, decide, orders = bridge(
        instrument=instrument, factor=factor, max_weight=0.10,
    )
    model = CostModelConfig(sh_transfer_fee_par_rate=0.001)
    exchange = _exchange(instrument, factor=factor, cost_model=model)
    generated = orders(decide(), exchange)
    value, cost, price = exchange.deal_order(
        generated[0], position=current, dealt_order_amount=defaultdict(float),
    )
    assert price == 5.0
    assert generated[0].deal_amount == 2000.0
    expected = model.estimate(
        side="buy", gross_value=value, participation=value / 10_000_000.0,
        asset_type="stock", trade_date=pd.Timestamp("2024-01-03").date(),
        instrument=instrument, quantity=1000.0,
    )
    assert cost == expected
    assert exchange.fill_log[0]["amount"] == 2000.0
    assert exchange.fill_log[0]["raw_amount"] == 1000.0


def test_t1_locked_quantity_is_part_of_the_audited_target(bridge):
    strategy, _current, _metadata, decide, orders = bridge(
        quantity=950.0, factor=0.8, hold=True, max_weight=0.60,
    )
    strategy._t1_locked = {"SZ000001": {pd.Timestamp("2024-01-03").date(): 950.0 / 0.8}}
    assert orders(decide()) == []


@pytest.mark.parametrize("changed", ["weights", "risk", "window", "position"])
def test_quantity_plan_rejects_second_scale_or_changed_binding(bridge, changed):
    _strategy, current, _metadata, decide, orders = bridge(max_weight=0.01)
    weights = decide()
    kwargs = {}
    if changed == "weights":
        weights = {key: value * 0.95 for key, value in weights.items()}
    elif changed == "risk":
        kwargs["risk_degree"] = 0.95
    elif changed == "window":
        kwargs["trade_start_time"] = pd.Timestamp("2024-01-04")
    else:
        current.position["cash"] -= 1
    with pytest.raises(ValueError, match="differs from the audited"):
        orders(weights, **kwargs)
    with pytest.raises(ValueError, match="no audited quantity plan"):
        orders(weights)


def test_unheld_unlisted_has_no_factor_or_board_rule_requirement_but_held_does(bridge):
    from quant_platform.qlib_policy_strategy import qlib_execution_lot_context

    _strategy, current, _metadata, _decide, _orders = bridge()
    index = pd.Index(["BJ430047"])
    context = qlib_execution_lot_context(
        current, instruments=index, factors=pd.Series(np.nan, index=index),
        prices=pd.Series(np.nan, index=index), current_prices=pd.Series(np.nan, index=index),
        trade_date=pd.Timestamp("2015-10-12").date(), locked_amounts={},
    )
    context.prepare(pd.Series(0.0, index=index), portfolio_value=100_000,
                    frozen_instruments={"BJ430047"})
    current.position["BJ430047"] = {"amount": 100.0, "price": 10.0, "weight": 0.01}
    with pytest.raises(ValueError, match="no valid signal factor"):
        qlib_execution_lot_context(
            current, instruments=index, factors=pd.Series(np.nan, index=index),
            prices=pd.Series(np.nan, index=index), current_prices=pd.Series(np.nan, index=index),
            trade_date=pd.Timestamp("2015-10-12").date(), locked_amounts={},
        )
