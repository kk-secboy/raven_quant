from datetime import date

import pandas as pd
import pytest

from quant_platform.cost_model import CostModelConfig
from quant_platform.etf_subtypes import (
    ETF_SUBTYPE_GATE_VERSION,
    etf_trading_gate,
    fund_subtype,
    list_etf_subtype_registry,
    validate_etf_subtype_registry,
)
from quant_platform.simulation_engine import (
    execute_atomic_pair_day,
    execute_simulation_day,
)

pytestmark = pytest.mark.no_database


def test_registry_contract_and_version() -> None:
    validate_etf_subtype_registry()
    assert ETF_SUBTYPE_GATE_VERSION
    registry = {entry.subtype: entry for entry in list_etf_subtype_registry()}
    assert registry["equity"].accepted is True
    assert registry["equity"].accepted_on and registry["equity"].evidence
    for subtype in ("cross_border", "bond", "gold", "commodity", "money"):
        assert registry[subtype].accepted is False
        assert registry[subtype].pending_acceptance


@pytest.mark.parametrize(
    ("instrument", "expected"),
    [
        ("SH600000", None),  # 股票不是基金
        ("000001.SZ", None),
        ("510300.SH", "equity"),
        ("159915.SZ", "equity"),
        ("588000.SH", "equity"),
        ("518880.SH", "gold"),
        ("159934.SZ", "gold"),
        ("511010.SH", "bond"),
        ("511360.SH", "bond"),
        ("511880.SH", "money"),
        ("513100.SH", "cross_border"),
        ("159920.SZ", "cross_border"),
        ("159980.SZ", "commodity"),
        ("160105.SZ", "unclassified"),  # LOF 不在白名单分类内
        ("501018.SH", "unclassified"),
    ],
)
def test_fund_subtype_classification(instrument: str, expected: str | None) -> None:
    assert fund_subtype(instrument) == expected


def test_gate_allows_non_funds_and_accepted_subtypes_only() -> None:
    assert etf_trading_gate("SH600000") is None
    assert etf_trading_gate("510300.SH") is None
    assert etf_trading_gate("518880.SH") == "etf_subtype_not_accepted:gold"
    assert etf_trading_gate("511010.SH") == "etf_subtype_not_accepted:bond"
    assert etf_trading_gate("513100.SH") == "etf_subtype_not_accepted:cross_border"
    assert etf_trading_gate("511990.SH") == "etf_subtype_not_accepted:money"
    assert etf_trading_gate("159980.SZ") == "etf_subtype_not_accepted:commodity"
    assert etf_trading_gate("160105.SZ") == "etf_subtype_unclassified"


def _bars(*instruments: str, price: float = 10.0, day: str = "2025-01-03") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "datetime": f"{day} 13:30:00",
                "instrument": instrument,
                "close": price,
                "vwap": price,
                "volume": 1_000_000,
                "paused": 0,
                "up_limit": 11.0,
                "down_limit": 9.0,
            }
            for instrument in instruments
        ]
    )


def _run(instrument: str, **overrides):
    defaults = {
        "trade_date": date(2025, 1, 3),
        "cash": 100_000.0,
        "prior_nav": 100_000.0,
        "high_water_mark": 100_000.0,
        "positions": {},
        "target_weights": {instrument: 0.50},
        "minute_bars": _bars(instrument),
        "closing_prices": {instrument: {"price": 10.0, "market_date": date(2025, 1, 3)}},
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


def test_unaccepted_etf_subtype_order_is_rejected_fail_closed() -> None:
    result = _run("SH518880")  # 黄金 ETF：子类型未验收
    order = result["orders"][0]
    assert order["status"] == "rejected"
    assert order["reject_reason"] == "etf_subtype_not_accepted:gold"
    assert result["fills"] == []
    assert "SH518880" not in result["positions"]
    event = next(item for item in result["events"] if item["event_type"] == "order_rejected")
    assert event["reason"] == "etf_subtype_not_accepted:gold"


def test_persistent_order_spec_for_unaccepted_subtype_is_rejected() -> None:
    result = _run(
        "SH511010",  # 债券 ETF：子类型未验收
        target_weights={},
        order_specs_override=[
            {"instrument": "SH511010", "side": "buy", "requested_quantity": 100}
        ],
    )
    order = result["orders"][0]
    assert order["status"] == "rejected"
    assert order["reject_reason"] == "etf_subtype_not_accepted:bond"
    assert result["fills"] == []


def test_sell_of_unaccepted_subtype_is_also_rejected() -> None:
    result = _run(
        "SH518880",
        positions={
            "SH518880": {
                "quantity": 100,
                "available_quantity": 100,
                "average_cost": 10.0,
                "last_trade_date": date(2025, 1, 2),
            }
        },
        target_weights={},
    )
    order = result["orders"][0]
    assert order["side"] == "sell"
    assert order["status"] == "rejected"
    assert order["reject_reason"] == "etf_subtype_not_accepted:gold"
    assert result["positions"]["SH518880"]["quantity"] == 100


def test_accepted_equity_etf_and_stocks_still_trade() -> None:
    etf = _run("SH510300")  # 已验收的股票型 ETF
    assert etf["orders"][0]["status"] == "filled"
    assert etf["positions"]["SH510300"]["quantity"] == 5_000

    stock = _run("SH600000")  # 股票不受门禁影响
    assert stock["orders"][0]["status"] == "filled"


def test_research_pair_ledger_is_not_gated() -> None:
    # 配对台账是永久离线研究路径（research_only），不挂 ETF 交易门禁。
    result = execute_atomic_pair_day(
        trade_date=date(2025, 1, 3),
        cash=100_000.0,
        prior_nav=100_000.0,
        high_water_mark=100_000.0,
        positions={},
        target_payload={
            "atomic_group_id": "pair-gold-1",
            "legs": [
                {
                    "instrument": "SH518880",
                    "leg_no": 1,
                    "position_side": "long",
                    "target_quantity": 100,
                    "annual_borrow_rate": 0.0,
                },
                {
                    "instrument": "SH600001",
                    "leg_no": 2,
                    "position_side": "short",
                    "target_quantity": 100,
                    "annual_borrow_rate": 0.08,
                },
            ],
        },
        minute_bars=_bars("SH518880", "SH600001"),
        closing_prices={
            "SH518880": {"price": 10.0, "market_date": date(2025, 1, 3)},
            "SH600001": {"price": 10.0, "market_date": date(2025, 1, 3)},
        },
        shortability={"SH600001": True},
        cost_model=CostModelConfig(),
        execution_policy={
            "execution_algorithm": "vwap",
            "slice_minutes": 5,
            "max_slices": 1,
            "max_participation": 0.01,
        },
    )
    assert [item["status"] for item in result["orders"]] == ["filled", "filled"]
    assert result["positions"]["SH518880"]["quantity"] == 100
