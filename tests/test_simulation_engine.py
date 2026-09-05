import hashlib
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.cost_model import CostModelConfig
from quant_platform.simulation_engine import execute_atomic_pair_day, execute_simulation_day
from quant_platform.simulation_store import (
    SimulationStore,
    build_settlement_calendar_binding,
    build_settlement_calendar_evidence,
    validate_settlement_calendar_binding,
    validate_settlement_calendar_evidence,
)
from quant_platform.worker import _bind_daily_simulation_settlement_calendar

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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cash", float("nan")),
        ("prior_nav", float("nan")),
        ("high_water_mark", float("inf")),
    ],
)
def test_non_finite_account_balances_fail_closed(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="account balances"):
        _run(**{field: value})


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
    assert result["nav_row"]["market_date"] == date(2025, 1, 3)
    assert result["nav_row"]["performance_certified"] is True


def test_explicit_empty_target_books_a_certified_all_cash_decision() -> None:
    result = _run(
        target_weights={},
        positions={},
        minute_bars=_bars().iloc[0:0],
        closing_prices={},
    )

    assert result["orders"] == []
    assert result["fills"] == []
    assert result["cash_flows"] == []
    assert result["cash"] == pytest.approx(100_000.0)
    assert result["nav"] == pytest.approx(100_000.0)
    assert result["conservation"]["cash_difference"] == pytest.approx(0.0)
    assert result["nav_row"]["market_date"] == date(2025, 1, 3)
    assert result["nav_row"]["status"] == "healthy"
    assert result["nav_row"]["performance_certified"] is True


def test_long_only_target_contract_distinguishes_empty_from_missing() -> None:
    assert SimulationStore._normalize_target_payload(
        {"target_weights": {}}, adapter="long_only"
    ) == {"target_weights": {}}
    with pytest.raises(ValueError, match="requires target_weights"):
        SimulationStore._normalize_target_payload({}, adapter="long_only")


def test_settlement_calendar_evidence_is_bound_to_execution_lineage() -> None:
    evidence = build_settlement_calendar_evidence(
        trade_date=date(2025, 1, 3),
        next_trade_date=date(2025, 1, 6),
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
        calendar_file_sha256="c" * 64,
    )

    assert validate_settlement_calendar_evidence(
        evidence,
        trade_date=date(2025, 1, 3),
        next_trade_date=date(2025, 1, 6),
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
        calendar_file_sha256="c" * 64,
    ) == evidence
    with pytest.raises(ValueError, match="does not match the bound dataset"):
        validate_settlement_calendar_evidence(
            evidence,
            trade_date=date(2025, 1, 3),
            next_trade_date=date(2025, 1, 6),
            dataset_identity_sha256="d" * 64,
            dataset_lineage_id="b" * 64,
            calendar_file_sha256="c" * 64,
        )
    self_signed_tamper = build_settlement_calendar_evidence(
        trade_date=date(2025, 1, 3),
        next_trade_date=date(2025, 1, 6),
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
        calendar_file_sha256="e" * 64,
    )
    with pytest.raises(ValueError, match="does not match the bound dataset"):
        validate_settlement_calendar_evidence(
            self_signed_tamper,
            trade_date=date(2025, 1, 3),
            next_trade_date=date(2025, 1, 6),
            dataset_identity_sha256="a" * 64,
            dataset_lineage_id="b" * 64,
            calendar_file_sha256="c" * 64,
        )


def test_settlement_calendar_binding_is_derived_from_sealed_dataset(
    tmp_path: Path,
) -> None:
    provider = tmp_path / "qlib"
    calendar = provider / "metadata" / "known_trading_calendar.parquet"
    calendar.parent.mkdir(parents=True)
    pd.DataFrame({"date": pd.to_datetime(["2025-01-03", "2025-01-06"])}).to_parquet(
        calendar,
        index=False,
    )
    calendar_hash = hashlib.sha256(calendar.read_bytes()).hexdigest()
    dataset = {
        "path": str(provider),
        "output_files_verified": True,
        "provenance": {
            "dataset_identity_sha256": "a" * 64,
            "dataset_lineage_id": "b" * 64,
            "output_manifest": {
                "version": "qlib-output-files-v1",
                "files": [
                    {
                        "path": "metadata/known_trading_calendar.parquet",
                        "bytes": calendar.stat().st_size,
                        "sha256": calendar_hash,
                    }
                ],
            },
        },
    }

    binding = build_settlement_calendar_binding(
        dataset,
        trade_date=date(2025, 1, 3),
    )

    assert binding["calendar_file_sha256"] == calendar_hash
    assert binding["next_trade_date"] == "2025-01-06"
    assert validate_settlement_calendar_binding(
        binding,
        trade_date=date(2025, 1, 3),
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
    ) == binding
    worker_manifest = _bind_daily_simulation_settlement_calendar(
        {
            "execution_frequency": "day",
            "trade_date": "2025-01-03",
            "settlement_calendar_binding": binding,
        },
        dataset,
    )
    assert worker_manifest["settlement_calendar_binding"] == binding
    calendar.write_bytes(calendar.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="sealed output manifest"):
        build_settlement_calendar_binding(
            dataset,
            trade_date=date(2025, 1, 3),
        )


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


def test_later_sell_cannot_fund_earlier_buy_but_can_fund_same_bar_buy() -> None:
    sell_bars = pd.concat(
        [_bars(price=20.0, up_limit=22.0, down_limit=18.0)] * 2, ignore_index=True
    )
    buy_bars = pd.concat([_bars(instrument="SH600001")] * 2, ignore_index=True)
    for frame in (sell_bars, buy_bars):
        frame["datetime"] = ["2025-01-03 10:00:00", "2025-01-03 14:50:00"]
    sell_bars.loc[0, "paused"] = 1
    result = _run(
        cash=0.0,
        prior_nav=20_000.0,
        high_water_mark=20_000.0,
        positions={
            "SH600000": {
                "quantity": 1000,
                "available_quantity": 1000,
                "average_cost": 20.0,
                "last_trade_date": date(2025, 1, 2),
            }
        },
        target_weights={"SH600001": 0.5},
        minute_bars=pd.concat([buy_bars, sell_bars], ignore_index=True),
        closing_prices={
            "SH600000": {"price": 20.0, "market_date": date(2025, 1, 3)},
            "SH600001": {"price": 10.0, "market_date": date(2025, 1, 3)},
        },
        execution_policy={
            "execution_algorithm": "twap",
            "slice_minutes": 20,
            "max_slices": 2,
            "max_participation": 0.01,
        },
    )

    assert [(fill["side"], fill["quantity"]) for fill in result["fills"]] == [
        ("sell", 500),
        ("buy", 500),
    ]
    assert all(fill["executed_at"].strftime("%H:%M") == "14:50" for fill in result["fills"])
    balance = 0.0
    for fill in sorted(result["fills"], key=lambda item: item["executed_at"]):
        balance += (
            fill["gross_value"] - fill["fee"]
            if fill["side"] == "sell"
            else -fill["gross_value"] - fill["fee"]
        )
        assert balance >= 0.0
    assert result["cash"] == pytest.approx(balance)
    assert all(flow["balance_after"] >= 0.0 for flow in result["cash_flows"])
    assert result["positions"]["SH600001"]["available_quantity"] == 0
    assert result["orders"][1]["status"] == "partial_filled_expired"
    assert result["orders"][1]["reject_reason"] == "insufficient_cash"


def test_multiple_orders_share_one_instrument_bar_participation_capacity() -> None:
    result = _run(
        minute_bars=_bars(volume=100_000),
        order_specs_override=[
            {
                "instrument": "SH600000",
                "side": "buy",
                "requested_quantity": 700,
                "order_ref": reference,
            }
            for reference in ("first", "second")
        ],
    )
    assert [(fill["order_ref"], fill["quantity"]) for fill in result["fills"]] == [
        ("first", 700),
        ("second", 300),
    ]
    assert sum(fill["quantity"] for fill in result["fills"]) == 1000
    assert [order["status"] for order in result["orders"]] == [
        "filled", "partial_filled_expired"
    ]
    assert result["orders"][1]["reject_reason"] == "capacity"


def test_order_cash_reservation_is_retained_between_chronological_slices() -> None:
    bars = pd.concat([_bars()] * 2, ignore_index=True)
    bars["datetime"] = ["2025-01-03 10:00:00", "2025-01-03 14:50:00"]
    result = _run(
        minute_bars=bars,
        order_specs_override=[
            {
                "instrument": "SH600000",
                "side": "buy",
                "requested_quantity": 1000,
                "order_ref": "reserved",
                "reserved_cash": 6020.0,
            }
        ],
        execution_policy={
            "execution_algorithm": "twap",
            "slice_minutes": 20,
            "max_slices": 2,
            "max_participation": 0.01,
        },
    )
    assert [fill["quantity"] for fill in result["fills"]] == [500, 100]
    assert sum(fill["gross_value"] + fill["fee"] for fill in result["fills"]) <= 6020.0
    assert result["orders"][0]["filled_quantity"] == 600
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


def _pair_run(**overrides):
    bars = pd.concat(
        [
            _bars(instrument="SH600000", price=10.0),
            _bars(instrument="SH600001", price=20.0),
        ],
        ignore_index=True,
    )
    defaults = {
        "trade_date": date(2025, 1, 3),
        "cash": 100_000.0,
        "prior_nav": 100_000.0,
        "high_water_mark": 100_000.0,
        "positions": {},
        "target_payload": {
            "atomic_group_id": "pair-group-1",
            "legs": [
                {
                    "instrument": "SH600000",
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
        "minute_bars": bars,
        "closing_prices": {
            "SH600000": {"price": 10.0, "market_date": date(2025, 1, 3)},
            "SH600001": {"price": 20.0, "market_date": date(2025, 1, 3)},
        },
        "shortability": {"SH600001": True},
        "cost_model": CostModelConfig(),
        "execution_policy": {
            "execution_algorithm": "vwap",
            "slice_minutes": 5,
            "max_slices": 1,
            "max_participation": 0.01,
        },
    }
    defaults.update(overrides)
    return execute_atomic_pair_day(**defaults)


def test_pair_execution_books_both_legs_as_one_atomic_group() -> None:
    result = _pair_run()

    assert [item["status"] for item in result["orders"]] == ["filled", "filled"]
    assert {item["atomic_group_id"] for item in result["fills"]} == {"pair-group-1"}
    assert {item["leg_no"] for item in result["fills"]} == {1, 2}
    assert result["positions"]["SH600000"]["position_side"] == "long"
    assert result["positions"]["SH600001"]["position_side"] == "short"
    short_fill = next(item for item in result["fills"] if item["position_side"] == "short")
    assert short_fill["borrow_cost"] > 0
    assert result["conservation"]["cash_difference"] == pytest.approx(0.0)


def test_pair_execution_uses_one_coordinated_vwap_window_for_both_legs() -> None:
    first = pd.concat(
        [
            _bars(instrument="SH600000", price=10.0, volume=100_000),
            _bars(instrument="SH600001", price=20.0, volume=100_000),
        ],
        ignore_index=True,
    )
    second = pd.concat(
        [
            _bars(
                instrument="SH600000", price=12.0, volume=300_000,
                up_limit=13.0, down_limit=9.0,
            ),
            _bars(
                instrument="SH600001", price=24.0, volume=300_000,
                up_limit=26.0, down_limit=18.0,
            ),
        ],
        ignore_index=True,
    )
    second["datetime"] = "2025-01-03 14:00:00"
    result = _pair_run(minute_bars=pd.concat([first, second], ignore_index=True))

    fills = {item["instrument"]: item for item in result["fills"]}
    assert fills["SH600000"]["price"] == pytest.approx(11.5)
    assert fills["SH600001"]["price"] == pytest.approx(23.0)
    assert all(item["minute_volume"] == 400_000 for item in fills.values())
    assert all(item["executed_at"].hour == 14 for item in fills.values())


def test_pair_simulation_accrues_borrow_cost_on_an_unchanged_short_leg() -> None:
    opened = _pair_run()
    next_day = date(2025, 1, 6)
    bars = pd.concat(
        [
            _bars(instrument="SH600000", price=10.0, day=next_day.isoformat()),
            _bars(instrument="SH600001", price=20.0, day=next_day.isoformat()),
        ],
        ignore_index=True,
    )
    carried = _pair_run(
        trade_date=next_day,
        cash=opened["cash"],
        prior_nav=opened["nav"],
        high_water_mark=opened["high_water_mark"],
        positions=opened["positions"],
        minute_bars=bars,
        closing_prices={
            "SH600000": {"price": 10.0, "market_date": next_day},
            "SH600001": {"price": 20.0, "market_date": next_day},
        },
    )

    assert carried["orders"] == []
    assert carried["fills"] == []
    assert carried["cash_flows"][0]["flow_type"] == "pair_borrow_carry"
    assert carried["cash_flows"][0]["amount"] < 0
    assert carried["positions"]["SH600001"]["borrow_cost"] > opened["positions"][
        "SH600001"
    ]["borrow_cost"]


def test_pair_rebalance_rejection_still_books_existing_short_carry() -> None:
    opened = _pair_run()
    next_day = date(2025, 1, 6)
    bars = pd.concat(
        [
            _bars(instrument="SH600000", price=10.0, day=next_day.isoformat()),
            _bars(instrument="SH600001", price=20.0, day=next_day.isoformat()),
        ],
        ignore_index=True,
    )
    target = {
        "atomic_group_id": "pair-group-2",
        "legs": [
            {
                "instrument": "SH600000",
                "leg_no": 1,
                "position_side": "long",
                "target_quantity": 200,
                "annual_borrow_rate": 0.0,
            },
            {
                "instrument": "SH600001",
                "leg_no": 2,
                "position_side": "short",
                "target_quantity": 200,
                "annual_borrow_rate": 0.08,
            },
        ],
    }
    rejected = _pair_run(
        trade_date=next_day,
        cash=opened["cash"],
        prior_nav=opened["nav"],
        high_water_mark=opened["high_water_mark"],
        positions=opened["positions"],
        target_payload=target,
        minute_bars=bars,
        closing_prices={
            "SH600000": {"price": 10.0, "market_date": next_day},
            "SH600001": {"price": 20.0, "market_date": next_day},
        },
        shortability={"SH600001": False},
    )

    assert {item["status"] for item in rejected["orders"]} == {"rejected"}
    assert rejected["cash_flows"][0]["flow_type"] == "pair_borrow_carry"
    assert rejected["cash"] < opened["cash"]
    assert rejected["positions"]["SH600001"]["quantity"] == 100
    assert rejected["positions"]["SH600001"]["borrow_cost"] > opened["positions"][
        "SH600001"
    ]["borrow_cost"]
    assert rejected["conservation"]["cash_difference"] == pytest.approx(0.0)


def test_pair_day_does_not_advance_when_existing_short_carry_is_unpriceable() -> None:
    opened = _pair_run()
    next_day = date(2025, 1, 6)
    with pytest.raises(ValueError, match="borrow carry cannot be priced"):
        _pair_run(
            trade_date=next_day,
            cash=opened["cash"],
            prior_nav=opened["nav"],
            high_water_mark=opened["high_water_mark"],
            positions=opened["positions"],
            minute_bars=_bars(
                instrument="SH600000", price=10.0, day=next_day.isoformat()
            ),
            closing_prices={
                "SH600000": {"price": 10.0, "market_date": next_day},
                "SH600001": {"price": 20.0, "market_date": date(2025, 1, 3)},
            },
        )


@pytest.mark.parametrize("reason", ["short_borrow_not_authorized", "atomic_capacity"])
def test_pair_execution_rejects_both_legs_when_either_leg_is_ineligible(reason: str) -> None:
    overrides = {"shortability": {"SH600001": False}}
    if reason == "atomic_capacity":
        overrides = {
            "minute_bars": pd.concat(
                [
                    _bars(instrument="SH600000", price=10.0),
                    _bars(instrument="SH600001", price=20.0, volume=5_000),
                ],
                ignore_index=True,
            )
        }
    result = _pair_run(**overrides)

    assert len(result["orders"]) == 2
    assert {item["status"] for item in result["orders"]} == {"rejected"}
    assert {item["reject_reason"] for item in result["orders"]} == {reason}
    assert result["fills"] == []
    assert result["positions"] == {}
    assert result["nav_row"]["performance_certified"] is False


def test_non_liquidating_sell_uses_board_lot_increment() -> None:
    # STAR board: minimum declaration 200 shares, increments of 1 above it.
    # Selling 150 of a 450-share position toward a 300-share target must keep
    # 150 (board increment 1), not be floored to the main-board 100.
    result = _run(
        positions={
            "SH688001": {
                "quantity": 450,
                "available_quantity": 450,
                "average_cost": 10.0,
                "last_trade_date": date(2025, 1, 2),
            }
        },
        target_weights={"SH688001": 0.03},
        minute_bars=_bars(instrument="SH688001"),
        closing_prices={"SH688001": {"price": 10.0, "market_date": date(2025, 1, 3)}},
    )
    order = result["orders"][0]
    assert order["side"] == "sell"
    assert order["requested_quantity"] == 150
    assert order["filled_quantity"] == 150
    assert result["positions"]["SH688001"]["quantity"] == 300
