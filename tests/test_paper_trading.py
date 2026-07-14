import inspect
from datetime import UTC, datetime

import pandas as pd

from quant_platform.paper_trading import build_rebalance_plan


def test_docx_default_single_position_stop_loss_is_seven_percent() -> None:
    parameter = inspect.signature(build_rebalance_plan).parameters["stop_loss"]
    assert parameter.default == 0.07


def test_rebalance_plan_respects_turnover_cash_and_suspension() -> None:
    scores = pd.Series({"SH600000": 3.0, "SH600001": 2.0, "SH600002": 1.0})
    market = pd.DataFrame(
        {
            "open": [10.0, 20.0, 15.0],
            "close": [10.5, 20.5, 15.0],
            "paused": [0.0, 0.0, 1.0],
            "volume": [1_000_000.0, 1_000_000.0, 0.0],
            "amount": [1_000_000_000.0, 1_000_000_000.0, 0.0],
            "average_amount": [1_000_000_000.0, 1_000_000_000.0, 0.0],
            "industry": ["Bank", "Technology", "Industrial"],
            "up_limit": [11.0, 22.0, 16.5],
            "down_limit": [9.0, 18.0, 13.5],
        },
        index=scores.index,
    )
    result = build_rebalance_plan(
        scores,
        market,
        [],
        nav=1_000_000,
        cash=1_000_000,
        topk=3,
        n_drop=1,
        max_position_weight=0.20,
        max_daily_turnover=0.10,
        open_cost=0.0005,
        close_cost=0.0015,
        slippage=0.0005,
        fill_time=datetime(2025, 1, 3, 1, 30, tzinfo=UTC),
    )
    assert result["estimated_turnover"] <= 0.10 + 1e-12
    assert (
        sum(item["quantity"] * item["price"] + item["fee"] for item in result["fills"]) <= 1_000_000
    )
    suspended = next(item for item in result["orders"] if item["instrument"] == "SH600002")
    assert "suspended" in suspended["reason"]


def test_stop_loss_forces_auditable_sell() -> None:
    scores = pd.Series({"SH600000": 3.0})
    market = pd.DataFrame(
        {
            "open": [9.0],
            "close": [9.1],
            "paused": [0.0],
            "volume": [1_000_000.0],
            "amount": [1_000_000_000.0],
            "average_amount": [1_000_000_000.0],
            "industry": ["Bank"],
            "up_limit": [11.0],
            "down_limit": [8.0],
        },
        index=scores.index,
    )
    result = build_rebalance_plan(
        scores,
        market,
        [{"instrument": "SH600000", "quantity": 1000, "avg_cost": 10, "market_price": 10}],
        nav=100_000,
        cash=90_000,
        topk=1,
        n_drop=0,
        max_position_weight=0.20,
        max_daily_turnover=0.50,
        stop_loss=0.05,
        open_cost=0.0005,
        close_cost=0.0015,
        slippage=0.0005,
        fill_time=datetime(2025, 1, 3, 1, 30, tzinfo=UTC),
    )
    order = next(item for item in result["orders"] if item["instrument"] == "SH600000")
    assert order["side"] == "sell"
    assert order["reason"] == "stop_loss"
    assert result["fills"][0]["quantity"] == 1000
    assert any(item["rule"] == "stop_loss" for item in result["risk_events"])


def test_daily_loss_breaker_blocks_new_buys() -> None:
    scores = pd.Series({"SH600001": 3.0, "SH600000": 2.0})
    market = pd.DataFrame(
        {
            "open": [20.0, 5.0],
            "close": [20.0, 5.0],
            "paused": [0.0, 0.0],
            "volume": [1_000_000.0, 1_000_000.0],
            "amount": [1_000_000_000.0, 1_000_000_000.0],
            "average_amount": [1_000_000_000.0, 1_000_000_000.0],
            "industry": ["Technology", "Bank"],
            "up_limit": [22.0, 5.5],
            "down_limit": [18.0, 4.5],
        },
        index=scores.index,
    )
    result = build_rebalance_plan(
        scores,
        market,
        [{"instrument": "SH600000", "quantity": 10_000, "avg_cost": 10, "market_price": 10}],
        nav=1_000_000,
        cash=900_000,
        topk=2,
        n_drop=0,
        max_position_weight=0.20,
        max_daily_turnover=0.50,
        max_daily_loss=0.03,
        stop_loss=0.90,
        open_cost=0.0005,
        close_cost=0.0015,
        slippage=0.0005,
        fill_time=datetime(2025, 1, 3, 1, 30, tzinfo=UTC),
    )
    buy = next(item for item in result["orders"] if item["side"] == "buy")
    assert buy["reason"] == "daily loss circuit breaker"
    breaker = next(item for item in result["risk_events"] if item["rule"] == "max_daily_loss")
    assert breaker["severity"] == "critical"


def test_industry_and_liquidity_controls_reject_buys() -> None:
    scores = pd.Series({"SH600000": 3.0, "SH600001": 2.0, "SH600002": 1.0})
    market = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0],
            "paused": [0.0, 0.0, 0.0],
            "volume": [1_000_000.0] * 3,
            "amount": [1_000_000_000.0] * 3,
            "average_amount": [1_000_000_000.0, 1_000_000_000.0, 100_000_000.0],
            "industry": ["Bank", "Bank", "Technology"],
            "up_limit": [11.0, 11.0, 11.0],
            "down_limit": [9.0, 9.0, 9.0],
        },
        index=scores.index,
    )
    result = build_rebalance_plan(
        scores,
        market,
        [],
        nav=1_000_000,
        cash=1_000_000,
        topk=3,
        n_drop=0,
        max_position_weight=0.20,
        max_daily_turnover=1.0,
        max_industry_weight=0.20,
        min_average_daily_amount=500_000_000,
        open_cost=0.0005,
        close_cost=0.0015,
        slippage=0.0005,
        fill_time=datetime(2025, 1, 3, 1, 30, tzinfo=UTC),
    )
    reasons = {item["instrument"]: item.get("reason") for item in result["orders"]}
    assert reasons["SH600001"] == "industry concentration limit"
    assert reasons["SH600002"] == "minimum average daily amount"


def test_price_limits_block_unexecutable_orders() -> None:
    scores = pd.Series({"SH600001": 2.0, "SH600000": 1.0})
    market = pd.DataFrame(
        {
            "open": [11.0, 8.0],
            "close": [11.0, 8.0],
            "paused": [0.0, 0.0],
            "volume": [1_000_000.0, 1_000_000.0],
            "amount": [1_000_000_000.0, 1_000_000_000.0],
            "average_amount": [1_000_000_000.0, 1_000_000_000.0],
            "industry": ["Technology", "Bank"],
            "up_limit": [11.0, 12.0],
            "down_limit": [9.0, 8.0],
        },
        index=scores.index,
    )
    result = build_rebalance_plan(
        scores,
        market,
        [{"instrument": "SH600000", "quantity": 1000, "avg_cost": 10, "market_price": 10}],
        nav=100_000,
        cash=90_000,
        topk=1,
        n_drop=0,
        max_position_weight=0.20,
        max_daily_turnover=1.0,
        stop_loss=0.10,
        open_cost=0.0005,
        close_cost=0.0015,
        slippage=0.0005,
        fill_time=datetime(2025, 1, 3, 1, 30, tzinfo=UTC),
    )
    reasons = {item["instrument"]: item["reason"] for item in result["orders"]}
    assert reasons["SH600000"] == "limit_down_sell_blocked"
    assert reasons["SH600001"] == "limit_up_buy_blocked"
    assert result["fills"] == []


def test_partial_take_profit_is_staged_once() -> None:
    scores = pd.Series({"SH600000": 1.0})
    market = pd.DataFrame(
        {
            "open": [11.3],
            "close": [11.4],
            "paused": [0.0],
            "volume": [1_000_000.0],
            "amount": [1_000_000_000.0],
            "average_amount": [1_000_000_000.0],
            "industry": ["Bank"],
            "up_limit": [12.0],
            "down_limit": [8.0],
        },
        index=scores.index,
    )
    position = {
        "instrument": "SH600000",
        "quantity": 1000,
        "avg_cost": 10,
        "market_price": 10,
        "take_profit_stage": 0,
    }
    result = build_rebalance_plan(
        scores,
        market,
        [position],
        nav=100_000,
        cash=90_000,
        topk=1,
        n_drop=0,
        max_position_weight=0.20,
        max_daily_turnover=1.0,
        open_cost=0.0005,
        close_cost=0.0015,
        slippage=0.0005,
        fill_time=datetime(2025, 1, 3, 1, 30, tzinfo=UTC),
    )
    order = next(item for item in result["orders"] if item["instrument"] == "SH600000")
    assert order["reason"] == "take_profit_partial"
    assert result["fills"][0]["quantity"] == 500
    assert result["take_profit_stages"] == {"SH600000": 1}

    position["take_profit_stage"] = 1
    repeated = build_rebalance_plan(
        scores,
        market,
        [position],
        nav=100_000,
        cash=90_000,
        topk=1,
        n_drop=0,
        max_position_weight=0.20,
        max_daily_turnover=1.0,
        open_cost=0.0005,
        close_cost=0.0015,
        slippage=0.0005,
        fill_time=datetime(2025, 1, 3, 1, 30, tzinfo=UTC),
    )
    assert repeated["orders"] == []


def test_drawdown_state_forces_liquidation() -> None:
    scores = pd.Series({"SH600000": 1.0})
    market = pd.DataFrame(
        {
            "open": [10.0],
            "close": [10.0],
            "paused": [0.0],
            "volume": [1_000_000.0],
            "amount": [1_000_000_000.0],
            "average_amount": [1_000_000_000.0],
            "industry": ["Bank"],
            "up_limit": [11.0],
            "down_limit": [9.0],
        },
        index=scores.index,
    )
    result = build_rebalance_plan(
        scores,
        market,
        [{"instrument": "SH600000", "quantity": 10_000, "avg_cost": 10, "market_price": 10}],
        nav=900_000,
        cash=800_000,
        high_water_mark=1_100_000,
        portfolio_status="liquidation_pending",
        topk=1,
        n_drop=0,
        max_position_weight=0.20,
        max_daily_turnover=1.0,
        open_cost=0.0005,
        close_cost=0.0015,
        slippage=0.0005,
        fill_time=datetime(2025, 1, 3, 1, 30, tzinfo=UTC),
    )
    assert result["risk_action"] == "liquidate"
    assert result["orders"][0]["reason"] == "max_drawdown_liquidation"
    assert result["fills"][0]["quantity"] == 10_000
