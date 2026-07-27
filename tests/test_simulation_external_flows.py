"""Engine-level tests for external cash flows and the unitized TWR chain."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant_platform.cost_model import CostModelConfig
from quant_platform.simulation_engine import (
    execute_atomic_pair_day,
    execute_simulation_day,
)

pytestmark = pytest.mark.no_database


def _bars(price: float = 10.0, day: str = "2025-01-03") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "datetime": f"{day} 13:30:00",
                "instrument": "SH600000",
                "close": price,
                "vwap": price,
                "volume": 1_000_000,
                "paused": 0,
                "up_limit": 11.0,
                "down_limit": 9.0,
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
        "target_weights": {},
        "minute_bars": _bars(),
        "closing_prices": {},
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


def test_open_deposit_raises_cny_nav_but_not_twr() -> None:
    result = _run(external_flow_open=50_000.0)
    assert result["cash"] == pytest.approx(150_000.0)
    assert result["nav"] == pytest.approx(150_000.0)
    # 人民币口径日收益 +50%（跳升），单位化 TWR 收益为零（不制造收益）。
    assert result["nav_row"]["daily_return"] == pytest.approx(0.50)
    assert result["nav_row"]["twr_daily_return"] == pytest.approx(0.0)
    assert result["nav_row"]["investment_wealth"] == pytest.approx(1.0)
    assert result["nav_row"]["twr_status"] == "ok"
    assert result["nav_row"]["external_flow_open"] == pytest.approx(50_000.0)
    assert result["conservation"]["cash_difference"] == pytest.approx(0.0)
    flows = [item["flow_type"] for item in result["cash_flows"]]
    assert flows == ["external_deposit_open"]


def test_close_deposit_enters_nav_but_twr_strips_it() -> None:
    result = _run(external_flow_close=50_000.0)
    assert result["nav"] == pytest.approx(150_000.0)
    # r_t = (150_000 − 50_000) / 100_000 − 1 = 0。
    assert result["nav_row"]["twr_daily_return"] == pytest.approx(0.0)
    assert result["nav_row"]["external_flow_close"] == pytest.approx(50_000.0)
    assert result["conservation"]["cash_difference"] == pytest.approx(0.0)
    flows = [item["flow_type"] for item in result["cash_flows"]]
    assert flows == ["external_deposit_close"]


def test_open_deposit_is_investable_the_same_day() -> None:
    # 开盘前入金当日可投资：相对无入金基线，可买数量随入金增加。
    baseline = _run(target_weights={"SH600000": 1.0})
    result = _run(
        target_weights={"SH600000": 1.0},
        external_flow_open=50_000.0,
    )
    assert result["fills"][0]["quantity"] > baseline["fills"][0]["quantity"]
    assert result["cash"] >= 0
    assert result["conservation"]["cash_difference"] == pytest.approx(0.0)


def test_close_deposit_is_not_investable_the_same_day() -> None:
    # 盘后确认的入金当日不可交易：买入仍受盘前 10 万现金约束，与基线一致。
    baseline = _run(target_weights={"SH600000": 1.0})
    result = _run(
        target_weights={"SH600000": 1.0},
        external_flow_close=50_000.0,
    )
    assert result["fills"][0]["quantity"] == baseline["fills"][0]["quantity"]
    assert result["cash"] >= 50_000.0
    assert result["conservation"]["cash_difference"] == pytest.approx(0.0)


def test_withdrawal_is_symmetric_and_keeps_conservation() -> None:
    result = _run(external_flow_open=-60_000.0)
    assert result["cash"] == pytest.approx(40_000.0)
    assert result["nav"] == pytest.approx(40_000.0)
    assert result["nav_row"]["twr_daily_return"] == pytest.approx(0.0)
    assert result["conservation"]["cash_difference"] == pytest.approx(0.0)
    flows = [item["flow_type"] for item in result["cash_flows"]]
    assert flows == ["external_withdrawal_open"]


def test_withdrawal_beyond_cash_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="would create negative cash"):
        _run(external_flow_open=-100_001.0)
    with pytest.raises(RuntimeError, match="would create negative cash"):
        _run(external_flow_close=-100_001.0)


def test_non_finite_flows_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        _run(external_flow_open=float("nan"))


def test_nonpositive_twr_base_marks_day_unavailable() -> None:
    result = _run(external_flow_open=-100_000.0)
    nav_row = result["nav_row"]
    assert nav_row["twr_status"] == "undefined_nonpositive_base"
    assert nav_row["twr_daily_return"] is None
    assert nav_row["investment_wealth"] is None
    assert result["investment_wealth"] is None


def test_broken_chain_does_not_skip_and_continue() -> None:
    # 设计 4.4：链断裂后不得跳过当日继续连乘。
    result = _run(prior_investment_wealth=None, twr_high_water_mark=None)
    assert result["nav_row"]["twr_status"] == "unavailable_broken_chain"
    assert result["investment_wealth"] is None
    assert result["twr_high_water_mark"] is None


def test_pair_ledger_rejects_external_flows() -> None:
    bars = pd.DataFrame(
        [
            {
                "datetime": "2025-01-03 13:30:00",
                "instrument": instrument,
                "close": 10.0,
                "vwap": 10.0,
                "volume": 1_000_000,
                "paused": 0,
                "up_limit": 11.0,
                "down_limit": 9.0,
            }
            for instrument in ("SH600000", "SH600001")
        ]
    )
    with pytest.raises(ValueError, match="external cash flows"):
        execute_atomic_pair_day(
            trade_date=date(2025, 1, 3),
            cash=100_000.0,
            prior_nav=100_000.0,
            high_water_mark=100_000.0,
            positions={},
            target_payload={
                "atomic_group_id": "g1",
                "legs": [
                    {
                        "leg_no": 1,
                        "instrument": "SH600000",
                        "position_side": "long",
                        "target_quantity": 0,
                    },
                    {
                        "leg_no": 2,
                        "instrument": "SH600001",
                        "position_side": "short",
                        "target_quantity": 0,
                    },
                ],
            },
            minute_bars=bars,
            closing_prices={},
            shortability={"SH600001": True},
            cost_model=CostModelConfig(),
            execution_policy={
                "execution_algorithm": "twap",
                "slice_minutes": 20,
                "max_slices": 1,
                "max_participation": 0.01,
            },
            external_flow_open=10_000.0,
        )
