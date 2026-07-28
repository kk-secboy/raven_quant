"""Persistent simulation order states and order-plan consumption (design 8.1/9.2/12.4)."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from governance_fixtures import create_strategy_version
from sqlalchemy import insert, select, update
from test_simulation_store import (
    COST_SCHEDULE_VERSION,
    TRADE_DATE,
    _daily_dataset,
    _execution_dataset,
    _execution_evidence,
)

from quant_data.database import (
    open_database,
    simulation_batches,
    strategy_allocation_artifacts,
    strategy_allocations,
    strategy_versions,
)
from quant_platform.account_netting import AccountNettingStore
from quant_platform.cost_model import CostModelConfig
from quant_platform.recommendation_actions import plan_instrument_action
from quant_platform.recommendation_store import RecommendationStore
from quant_platform.simulation_engine import execute_simulation_day
from quant_platform.simulation_order_state import (
    STATUS_CANCELLED,
    STATUS_FILLED,
    STATUS_OPEN,
    STATUS_PLANNED,
    apply_order_plan,
    assert_transition,
)
from quant_platform.simulation_store import SimulationStore

SHANGHAI = ZoneInfo("Asia/Shanghai")
DAY2 = date(2026, 7, 14)


# ---------------------------------------------------------------------------
# Pure state machine
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_state_machine_legal_and_illegal_transitions() -> None:
    assert_transition("planned", "open")
    assert_transition("planned", "cancelled")
    assert_transition("open", "open")  # 工作单跨日留存
    assert_transition("open", "filled")
    assert_transition("open", "partial_filled_expired")
    assert_transition("open", "expired")
    assert_transition("open", "cancelled")
    for terminal in ("filled", "partial_filled_expired", "rejected", "expired", "cancelled"):
        with pytest.raises(ValueError, match="illegal simulation order transition"):
            assert_transition(terminal, "open")
    with pytest.raises(ValueError, match="illegal simulation order transition"):
        assert_transition("filled", "cancelled")


def _open_order(order_id: str, **overrides) -> dict:
    row = {
        "id": order_id,
        "instrument": "SH600000",
        "side": "buy",
        "requested_quantity": 200,
        "filled_quantity": 0,
        "status": "open",
        "limit_price": None,
    }
    row.update(overrides)
    return row


@pytest.mark.no_database
def test_apply_order_plan_maps_all_four_ops() -> None:
    outcome = apply_order_plan(
        open_orders=[
            _open_order("o-keep"),
            _open_order("o-cancel"),
            _open_order("o-replace", filled_quantity=40, requested_quantity=200),
        ],
        plan_entries=[
            {"op": "keep", "order_id": "o-keep"},
            {"op": "cancel", "order_id": "o-cancel", "quantity": 200, "reason": "opposite"},
            {"op": "replace", "order_id": "o-replace", "quantity": 60, "reason": "trim"},
            {"op": "new", "instrument": "SH600001", "side": "sell", "quantity": 100},
        ],
    )
    assert outcome["keeps"] == [{"op": "keep", "order_id": "o-keep"}]
    assert outcome["cancels"][0]["released_quantity"] == 200
    replace = outcome["replaces"][0]
    # 计划量是调整后的余量：新请求量 = 已成交 40 + 余量 60；释放 100。
    assert replace["new_requested_quantity"] == 100
    assert replace["released_quantity"] == 100
    assert outcome["news"] == [
        {
            "op": "new",
            "instrument": "SH600001",
            "side": "sell",
            "quantity": 100,
            "limit_price": None,
            "reason": "",
        }
    ]


@pytest.mark.no_database
def test_apply_order_plan_cancel_on_terminal_is_skipped_not_double_release() -> None:
    outcome = apply_order_plan(
        open_orders=[_open_order("o-1", status="cancelled")],
        plan_entries=[{"op": "cancel", "order_id": "o-1", "quantity": 200}],
    )
    assert outcome["cancels"] == []
    assert outcome["skipped"] == [{"op": "cancel", "order_id": "o-1", "status": "cancelled"}]


@pytest.mark.no_database
def test_apply_order_plan_replace_may_only_trim() -> None:
    with pytest.raises(ValueError, match="only trim"):
        apply_order_plan(
            open_orders=[_open_order("o-1", requested_quantity=200, filled_quantity=40)],
            # 余量 160 全部保留 → 新请求量 = 40 + 160 = 原值，并非 trim。
            plan_entries=[{"op": "replace", "order_id": "o-1", "quantity": 160}],
        )


@pytest.mark.no_database
def test_apply_order_plan_unknown_order_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown order"):
        apply_order_plan(
            open_orders=[_open_order("o-1")],
            plan_entries=[{"op": "cancel", "order_id": "ghost", "quantity": 100}],
        )


# ---------------------------------------------------------------------------
# Engine: price protection and execution window (pure)
# ---------------------------------------------------------------------------


def _engine_bars(*, price: float = 10.0, day: str = "2025-01-03") -> pd.DataFrame:
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


def _run_override(**spec_overrides):
    spec = {
        "instrument": "SH600000",
        "side": "buy",
        "requested_quantity": 1_000,
        "order_ref": "order-1",
    }
    spec.update(spec_overrides)
    return execute_simulation_day(
        trade_date=date(2025, 1, 3),
        cash=100_000.0,
        prior_nav=100_000.0,
        high_water_mark=100_000.0,
        positions={},
        target_weights={},
        minute_bars=_engine_bars(),
        closing_prices={"SH600000": {"price": 10.0, "market_date": date(2025, 1, 3)}},
        cost_model=CostModelConfig(),
        execution_policy={
            "execution_algorithm": "twap",
            "slice_minutes": 20,
            "max_slices": 1,
            "max_participation": 0.01,
        },
        order_specs_override=[spec],
    )


@pytest.mark.no_database
def test_engine_limit_price_blocks_fills_outside_protection() -> None:
    result = _run_override(limit_price=9.5)
    order = result["orders"][0]
    assert order["filled_quantity"] == 0
    assert order["reject_reason"] == "price_protection"
    assert result["fills"] == []
    filled = _run_override(limit_price=10.5)
    assert filled["orders"][0]["filled_quantity"] == 1_000
    assert filled["fills"][0]["order_ref"] == "order-1"


@pytest.mark.no_database
def test_engine_execution_window_skips_slices() -> None:
    early = _run_override(not_before=datetime(2025, 1, 3, 14, 0, tzinfo=SHANGHAI))
    assert early["orders"][0]["reject_reason"] == "before_execution_window"
    assert early["fills"] == []
    lapsed = _run_override(not_after=datetime(2025, 1, 3, 9, 30, tzinfo=SHANGHAI))
    assert lapsed["orders"][0]["reject_reason"] == "execution_window_elapsed"
    assert lapsed["fills"] == []


# ---------------------------------------------------------------------------
# DB fixtures
# ---------------------------------------------------------------------------


def _create_simulation(database_url: str, tmp_path) -> tuple[SimulationStore, dict]:
    version_id = create_strategy_version(
        database_url,
        tmp_path,
        config_overrides={"execution_frequency": "5min", "execution_method": "twap"},
    )
    recommendations = RecommendationStore(database_url)
    with recommendations.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved")
        )
    recommendation = recommendations.create(
        name="order-plan target",
        strategy_version_id=version_id,
        dataset="snapshot",
        hypothetical_initial_value=1_000_000,
        actor="test",
    )
    store = SimulationStore(database_url)
    simulation = store.create(
        name="order-plan simulation",
        recommendation_portfolio_id=recommendation["id"],
        daily_dataset=_daily_dataset(),
        execution_dataset=_execution_dataset(),
        initial_cash=1_000_000,
        execution_policy={"execution_algorithm": "twap"},
        cost_schedule_version=COST_SCHEDULE_VERSION,
        actor="test",
    )
    store.set_status(simulation["id"], "active")
    return store, simulation


def _bars(
    *, price: float = 10.0, day: date = TRADE_DATE, instruments: tuple[str, ...] = ("SH600000",)
) -> pd.DataFrame:
    # 覆盖受管 TWAP 政策的全部 10 个时间片（execution_algorithms.ASHARE_SESSIONS
    # 以 20 分钟间隔取片），否则只有落在有 Bar 的片上的那部分数量成交。
    slots = (
        "10:00", "10:20", "10:40", "11:00", "11:20",
        "13:30", "13:50", "14:10", "14:30", "14:50",
    )
    return pd.DataFrame(
        [
            {
                "datetime": f"{day.isoformat()} {slot}:00",
                "instrument": instrument,
                "close": price,
                "vwap": price,
                "volume": 1_000_000,
                "paused": 0,
                "up_limit": price * 1.1,
                "down_limit": price * 0.9,
            }
            for slot in slots
            for instrument in instruments
        ]
    )


def _process(
    store: SimulationStore,
    simulation: dict,
    batch: dict,
    *,
    price=10.0,
    day=TRADE_DATE,
    instruments: tuple[str, ...] = ("SH600000",),
    industry_snapshot: dict[str, str] | None = None,
):
    next_day = day + pd.offsets.BusinessDay(1)
    evidence = _execution_evidence(
        batch["id"],
        simulation["execution_contract_hash"],
        simulation["execution_policy"]["simulation_semantics_sha256"],
    )
    evidence["next_trade_date"] = next_day.date().isoformat()
    if industry_snapshot is not None:
        normalized = {
            str(instrument).strip().upper(): str(industry).strip()
            for instrument, industry in industry_snapshot.items()
        }
        evidence["industry_snapshot_sha256"] = hashlib.sha256(
            json.dumps(
                {
                    "trade_date": day.isoformat(),
                    "values": dict(sorted(normalized.items())),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    return store.process_batch(
        batch["id"],
        minute_bars=_bars(price=price, day=day, instruments=instruments),
        closing_prices={
            instrument: {"price": price, "market_date": day.isoformat()}
            for instrument in instruments
        },
        execution_evidence=evidence,
        industry_snapshot=industry_snapshot,
    )


def _orders(store: SimulationStore, simulation: dict) -> list[dict]:
    return store.rows(simulation["id"], "orders")


def _events(store: SimulationStore, simulation: dict, event_type: str) -> list[dict]:
    return [
        row for row in store.rows(simulation["id"], "events") if row["event_type"] == event_type
    ]


def _buy_action(instrument: str, quantity: int) -> dict:
    return {
        "instrument": instrument,
        "action": "BUY",
        "order_plan": [
            {
                "op": "new",
                "instrument": instrument,
                "side": "buy",
                "quantity": quantity,
                "limit_price": 10.0,
            }
        ],
    }


# ---------------------------------------------------------------------------
# DB: plan commit and execution wiring
# ---------------------------------------------------------------------------


def test_order_plan_batch_persists_planned_then_executes(database_url: str, tmp_path) -> None:
    store, simulation = _create_simulation(database_url, tmp_path)
    batch, created = store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[_buy_action("SH600000", 1_000)],
        target_version="target-v1",
        actor="test",
    )
    assert created is True
    (order,) = _orders(store, simulation)
    assert order["status"] == STATUS_PLANNED
    assert order["plan_op"] == "new"
    assert order["target_version"] == "target-v1"
    assert order["portfolio_id"] == simulation["id"]

    completed = _process(store, simulation, batch)
    assert completed["status"] == "succeeded"
    (order,) = _orders(store, simulation)
    assert order["status"] == STATUS_FILLED
    assert order["filled_quantity"] == 1_000
    fills = store.rows(simulation["id"], "fills")
    assert sum(int(fill["quantity"]) for fill in fills) == 1_000
    assert {fill["order_id"] for fill in fills} == {order["id"]}
    assert abs(completed["summary"]["conservation"]["cash_difference"]) < 1e-6
    cash = store.cash_view(simulation["id"])
    assert cash["total_cash"] == pytest.approx(float(completed["summary"]["cash"]))
    assert cash["frozen_cash"] == pytest.approx(0.0)
    cash_events = store.rows(simulation["id"], "cash_events")
    assert [item["event_type"] for item in cash_events].count("freeze") == 1
    assert [item["event_type"] for item in cash_events].count(
        "consume_frozen"
    ) == len(fills)
    # Limit-price and per-slice fee headroom is released after the terminal
    # fill instead of being charged or left frozen.
    assert [item["event_type"] for item in cash_events].count("release") == 1
    (attribution,) = store.rows(simulation["id"], "day_attributions")
    assert attribution["coverage_status"] == "partial"
    assert attribution["blocker_reasons_json"] == [
        "blocked_missing_bound_industry_snapshot"
    ]
    assert attribution["industry_json"]["status"] == (
        "blocked_missing_bound_industry_snapshot"
    )
    assert attribution["strategy_json"]["status"] == (
        "source_level_fallback_no_frozen_member_contributions"
    )
    execution = attribution["execution_json"]["instruments"]["SH600000"]
    assert int(execution["filled_quantity"]) == 1_000
    assert execution["fill_ratio"] == pytest.approx(1.0)
    assert attribution["cost_json"]["total_fee"] == pytest.approx(
        sum(float(fill["fee"]) for fill in fills)
    )


def test_buy_order_freeze_and_cancel_only_reclassify_cash_once(
    database_url: str, tmp_path
) -> None:
    store, simulation = _create_simulation(database_url, tmp_path)
    initial = store.get(simulation["id"])
    batch, _ = store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[_buy_action("SH600000", 1_000)],
        target_version="target-v1",
        actor="test",
    )
    (order,) = _orders(store, simulation)
    frozen = store.cash_view(simulation["id"])
    assert frozen["total_cash"] == pytest.approx(float(initial["cash"]))
    assert frozen["frozen_cash"] > 10_000
    assert frozen["free_cash"] + frozen["frozen_cash"] == pytest.approx(
        frozen["total_cash"]
    )
    # A reservation is a classification only: neither scalar cash nor NAV
    # changes when the order is committed.
    after_plan = store.get(simulation["id"])
    assert float(after_plan["cash"]) == pytest.approx(float(initial["cash"]))
    assert float(after_plan["nav"]) == pytest.approx(float(initial["nav"]))

    cancel = {
        "op": "cancel",
        "order_id": order["id"],
        "quantity": 1_000,
        "reason": "target_already_filled",
    }
    store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[
            {
                "instrument": "SH600000",
                "action": "HOLD",
                "order_plan": [cancel],
            }
        ],
        target_version="target-v2",
        actor="test",
    )
    released = store.cash_view(simulation["id"])
    assert released["total_cash"] == pytest.approx(frozen["total_cash"])
    assert released["frozen_cash"] == pytest.approx(0.0)
    assert released["free_cash"] == pytest.approx(released["total_cash"])

    # Replaying the cancellation against its terminal order creates no second
    # release and cannot manufacture free cash.
    store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[
            {
                "instrument": "SH600000",
                "action": "HOLD",
                "order_plan": [cancel],
            }
        ],
        target_version="target-v3",
        actor="test",
    )
    replayed = store.cash_view(simulation["id"])
    for field in (
        "total_cash",
        "free_cash",
        "frozen_cash",
        "tradable_cash",
        "withdrawable_cash",
    ):
        assert replayed[field] == pytest.approx(released[field])
    release_events = [
        row
        for row in store.rows(simulation["id"], "cash_events")
        if row["event_type"] == "release"
    ]
    assert len(release_events) == 1
    assert release_events[0]["order_id"] == order["id"]
    assert batch["id"] != release_events[0]["batch_id"]


def test_bound_industry_snapshot_completes_day_attribution(
    database_url: str, tmp_path
) -> None:
    store, simulation = _create_simulation(database_url, tmp_path)
    batch, _ = store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[_buy_action("SH600000", 1_000)],
        target_version="target-v1",
        actor="test",
    )
    _process(
        store,
        simulation,
        batch,
        industry_snapshot={"SH600000": "银行"},
    )
    (attribution,) = store.rows(simulation["id"], "day_attributions")
    assert attribution["coverage_status"] == "complete"
    assert attribution["blocker_reasons_json"] == []
    assert attribution["industry_json"]["status"] == "available"
    assert attribution["industry_json"]["unclassified_instruments"] == []
    assert attribution["industry_json"]["groups"]["银行"][
        "closing_market_value"
    ] == pytest.approx(10_000.0)


def test_sell_proceeds_are_tradable_same_day_but_withdrawable_next_business_day(
    database_url: str, tmp_path
) -> None:
    store, simulation = _create_simulation(database_url, tmp_path)
    buy_batch, _ = store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[_buy_action("SH600000", 1_000)],
        target_version="target-v1",
        actor="test",
    )
    _process(store, simulation, buy_batch)
    sell_batch, _ = store.create_order_plan_batch(
        simulation["id"],
        trade_date=DAY2,
        actions=[
            {
                "instrument": "SH600000",
                "action": "SELL",
                "order_plan": [
                    {
                        "op": "new",
                        "instrument": "SH600000",
                        "side": "sell",
                        "quantity": 1_000,
                        "limit_price": 9.0,
                    }
                ],
            }
        ],
        target_version="target-v2",
        actor="test",
    )
    (position_before_fill,) = store.rows(simulation["id"], "positions")
    assert int(position_before_fill["frozen_quantity"]) == 1_000
    assert int(position_before_fill["free_sellable_quantity"]) == 0
    _process(store, simulation, sell_batch, day=DAY2)
    assert store.rows(simulation["id"], "positions") == []
    (sell_reservation,) = [
        item
        for item in store.rows(simulation["id"], "position_reservations")
        if item["order_id"]
        == next(
            order["id"]
            for order in _orders(store, simulation)
            if order["side"] == "sell"
        )
    ]
    assert int(sell_reservation["remaining_quantity"]) == 0
    sell_lots = [
        row
        for row in store.rows(simulation["id"], "cash_lots")
        if row["source_type"] == "sell_settlement"
    ]
    assert len(sell_lots) == len(
        [
            fill
            for fill in store.rows(simulation["id"], "fills")
            if fill["side"] == "sell"
        ]
    )
    for proceeds in sell_lots:
        tradable_at = datetime.fromisoformat(str(proceeds["tradable_at"]))
        withdrawable_at = datetime.fromisoformat(str(proceeds["withdrawable_at"]))
        assert tradable_at.astimezone(SHANGHAI).date() == DAY2
        assert withdrawable_at.astimezone(SHANGHAI).date() == date(2026, 7, 15)
        assert withdrawable_at > tradable_at


def test_sell_orders_cannot_double_reserve_and_cancel_releases_once(
    database_url: str, tmp_path
) -> None:
    store, simulation = _create_simulation(database_url, tmp_path)
    buy_batch, _ = store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[_buy_action("SH600000", 1_000)],
        target_version="target-v1",
        actor="test",
    )
    _process(store, simulation, buy_batch)
    sell_batch, _ = store.create_order_plan_batch(
        simulation["id"],
        trade_date=DAY2,
        actions=[
            {
                "instrument": "SH600000",
                "action": "SELL",
                "order_plan": [
                    {
                        "op": "new",
                        "instrument": "SH600000",
                        "side": "sell",
                        "quantity": 600,
                        "limit_price": 9.0,
                    }
                ],
            }
        ],
        target_version="target-v2",
        actor="test",
    )
    sell_order = next(
        order for order in _orders(store, simulation) if order["side"] == "sell"
    )
    with pytest.raises(
        RuntimeError,
        match="exceeds free sellable quantity after existing freezes",
    ):
        store.create_order_plan_batch(
            simulation["id"],
            trade_date=DAY2,
            actions=[
                {
                    "instrument": "SH600000",
                    "action": "SELL",
                    "order_plan": [
                        {
                            "op": "new",
                            "instrument": "SH600000",
                            "side": "sell",
                            "quantity": 500,
                            "limit_price": 9.0,
                        }
                    ],
                }
            ],
            target_version="target-v3",
            actor="test",
        )
    cancel_kwargs = {
        "trade_date": DAY2,
        "actions": [
            {
                "instrument": "SH600000",
                "action": "HOLD",
                "order_plan": [
                    {
                        "op": "cancel",
                        "order_id": sell_order["id"],
                        "quantity": 600,
                        "reason": "target_already_filled",
                    }
                ],
            }
        ],
        "target_version": "target-v4",
        "actor": "test",
    }
    _, created = store.create_order_plan_batch(simulation["id"], **cancel_kwargs)
    _, replay_created = store.create_order_plan_batch(
        simulation["id"], **cancel_kwargs
    )
    assert created is True
    assert replay_created is False
    (position,) = store.rows(simulation["id"], "positions")
    assert int(position["frozen_quantity"]) == 0
    reservation = next(
        item
        for item in store.rows(simulation["id"], "position_reservations")
        if item["order_id"] == sell_order["id"]
    )
    assert int(reservation["remaining_quantity"]) == 0
    release_events = [
        event
        for event in store.rows(simulation["id"], "security_events")
        if event["event_type"] == "release"
        and event["order_id"] == sell_order["id"]
    ]
    assert len(release_events) == 1
    assert sell_batch["id"] != release_events[0]["batch_id"]


def test_plan_consumption_cancel_replace_new_end_to_end(database_url: str, tmp_path) -> None:
    store, simulation = _create_simulation(database_url, tmp_path)
    batch_a, _ = store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[_buy_action("SH600000", 200), _buy_action("SH600001", 300)],
        target_version="target-v1",
        actor="test",
    )
    orders_a = {row["instrument"]: row for row in _orders(store, simulation)}

    batch_b, created_b = store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[
            {
                "instrument": "SH600000",
                "action": "BUY",
                "order_plan": [
                    {
                        "op": "replace",
                        "order_id": orders_a["SH600000"]["id"],
                        "quantity": 100,
                        "reason": "trim_to_final_target",
                    },
                    {
                        "op": "new",
                        "instrument": "SH600002",
                        "side": "buy",
                        "quantity": 100,
                        "limit_price": 10.0,
                    },
                ],
            },
            {
                "instrument": "SH600001",
                "action": "HOLD",
                "order_plan": [
                    {
                        "op": "cancel",
                        "order_id": orders_a["SH600001"]["id"],
                        "quantity": 300,
                        "reason": "target_already_filled",
                    }
                ],
            },
        ],
        target_version="target-v2",
        actor="test",
    )
    assert created_b is True
    by_instrument = {row["instrument"]: row for row in _orders(store, simulation)}
    cancelled = by_instrument["SH600001"]
    assert cancelled["status"] == STATUS_CANCELLED
    assert cancelled["cancel_reason"] == "target_already_filled"
    replaced = by_instrument["SH600000"]
    assert replaced["requested_quantity"] == 100
    assert replaced["plan_op"] == "replace"
    new_order = by_instrument["SH600002"]
    assert new_order["status"] == STATUS_PLANNED
    assert new_order["batch_id"] == batch_b["id"]
    assert len(_events(store, simulation, "order_cancelled")) == 1
    assert len(_events(store, simulation, "order_replaced")) == 1

    # 被取代的计划批次 A 不再持有可执行单；批次 B 执行被替换的订单与新单。
    completed = _process(
        store,
        simulation,
        batch_b,
        instruments=("SH600000", "SH600001", "SH600002"),
    )
    assert completed["status"] == "succeeded"
    by_instrument = {row["instrument"]: row for row in _orders(store, simulation)}
    assert by_instrument["SH600000"]["status"] == STATUS_FILLED
    assert by_instrument["SH600000"]["filled_quantity"] == 100
    assert by_instrument["SH600002"]["status"] == STATUS_FILLED
    assert by_instrument["SH600001"]["status"] == STATUS_CANCELLED
    fills = store.rows(simulation["id"], "fills")
    assert sorted(int(fill["quantity"]) for fill in fills) == [100, 100]
    positions = store.rows(simulation["id"], "positions")
    assert {row["instrument"]: int(row["quantity"]) for row in positions} == {
        "SH600000": 100,
        "SH600002": 100,
    }
    assert batch_a["id"] != batch_b["id"]


def test_plan_batch_retry_is_idempotent(database_url: str, tmp_path) -> None:
    store, simulation = _create_simulation(database_url, tmp_path)
    kwargs = {
        "trade_date": TRADE_DATE,
        "actions": [_buy_action("SH600000", 500)],
        "target_version": "target-v1",
        "actor": "test",
    }
    first, created_first = store.create_order_plan_batch(simulation["id"], **kwargs)
    second, created_second = store.create_order_plan_batch(simulation["id"], **kwargs)
    assert created_first is True
    assert created_second is False
    assert second["id"] == first["id"]
    assert len(_orders(store, simulation)) == 1
    engine = open_database(database_url)
    with engine.connect() as connection:
        planned_batches = connection.execute(
            select(simulation_batches).where(
                simulation_batches.c.idempotency_key.like("order-plan:%")
            )
        ).all()
    assert len(planned_batches) == 1


def test_cancel_of_already_cancelled_order_is_skipped(database_url: str, tmp_path) -> None:
    store, simulation = _create_simulation(database_url, tmp_path)
    store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[_buy_action("SH600000", 200)],
        target_version="target-v1",
        actor="test",
    )
    (order,) = _orders(store, simulation)
    cancel_entry = {
        "op": "cancel",
        "order_id": order["id"],
        "quantity": 200,
        "reason": "opposite_of_final_target",
    }
    store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[{"instrument": "SH600000", "action": "SELL", "order_plan": [cancel_entry]}],
        target_version="target-v2",
        actor="test",
    )
    assert _orders(store, simulation)[0]["status"] == STATUS_CANCELLED
    assert len(_events(store, simulation, "order_cancelled")) == 1
    # 幂等重放同一取消：终态订单跳过，不重复释放、不重复记事件。
    store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[{"instrument": "SH600000", "action": "SELL", "order_plan": [cancel_entry]}],
        target_version="target-v3",
        actor="test",
    )
    assert _orders(store, simulation)[0]["status"] == STATUS_CANCELLED
    assert len(_events(store, simulation, "order_cancelled")) == 1


def test_hold_plan_cancels_excess_open_orders(database_url: str, tmp_path) -> None:
    store, simulation = _create_simulation(database_url, tmp_path)
    store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[_buy_action("SH600000", 100)],
        target_version="target-v1",
        actor="test",
    )
    (order,) = _orders(store, simulation)
    # recommendation_actions 语义：持仓 100、目标 100、仍有买单 100 → HOLD 并取消。
    action = plan_instrument_action(
        instrument="SH600000",
        target_quantity=100,
        filled_position=100,
        open_orders=[
            {
                "order_id": order["id"],
                "side": "buy",
                "requested_quantity": 100,
                "filled_quantity": 0,
                "expires_at": None,
                "created_at": None,
            }
        ],
        now=datetime(2026, 7, 13, 9, 0, tzinfo=SHANGHAI),
    )
    assert action["action"] == "HOLD"
    assert action["order_plan"][0]["op"] == "cancel"
    store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[action],
        target_version="target-v2",
        actor="test",
    )
    (cancelled,) = _orders(store, simulation)
    assert cancelled["status"] == STATUS_CANCELLED
    assert cancelled["cancel_reason"] == "target_already_filled"


def test_multi_day_limit_order_carries_over_and_fills_next_day(
    database_url: str, tmp_path
) -> None:
    store, simulation = _create_simulation(database_url, tmp_path)
    window_end = datetime.combine(DAY2, time(15, 0), SHANGHAI)
    batch, _ = store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[_buy_action("SH600000", 1_000)],
        target_version="target-v1",
        actor="test",
        limit_prices={"SH600000": 9.5},
        not_after=window_end,
    )
    # 第 1 天 vwap=10.0 超出 9.5 价格保护：不成交，窗口未结束，订单保持 open。
    _process(store, simulation, batch, price=10.0, day=TRADE_DATE)
    (order,) = _orders(store, simulation)
    assert order["status"] == STATUS_OPEN
    assert order["filled_quantity"] == 0
    assert order["reject_reason"] == "price_protection"

    batch2, created2 = store.create_order_plan_batch(
        simulation["id"],
        trade_date=DAY2,
        actions=[
            {
                "instrument": "SH600000",
                "action": "BUY",
                "order_plan": [
                    {
                        "op": "keep",
                        "order_id": order["id"],
                        "side": "buy",
                        "quantity": 1_000,
                    }
                ],
            }
        ],
        target_version="target-v1",
        actor="test",
    )
    assert created2 is True
    # 第 2 天 vwap=9.0 进入价格保护区间：同一订单行累计成交。
    _process(store, simulation, batch2, price=9.0, day=DAY2)
    rows = _orders(store, simulation)
    assert len(rows) == 1
    (order,) = rows
    assert order["status"] == STATUS_FILLED
    assert order["filled_quantity"] == 1_000
    assert order["batch_id"] == batch["id"]  # 创建批次不变
    fills = store.rows(simulation["id"], "fills")
    assert sum(int(fill["quantity"]) for fill in fills) == 1_000
    assert {fill["batch_id"] for fill in fills} == {batch2["id"]}  # 成交记在执行批次
    assert all(float(fill["price"]) == pytest.approx(9.0) for fill in fills)


def test_order_without_window_expires_end_of_day(database_url: str, tmp_path) -> None:
    store, simulation = _create_simulation(database_url, tmp_path)
    batch, _ = store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[_buy_action("SH600000", 1_000)],
        target_version="target-v1",
        actor="test",
        limit_prices={"SH600000": 9.5},
    )
    _process(store, simulation, batch, price=10.0)
    (order,) = _orders(store, simulation)
    # 无跨日窗口：当日未成交即过期（本项目订单默认当日有效）。
    assert order["status"] == "expired"
    assert order["filled_quantity"] == 0


# ---------------------------------------------------------------------------
# DB: account netting plan binding
# ---------------------------------------------------------------------------


def _seed_active_allocation(database_url: str) -> tuple[str, str]:
    engine = open_database(database_url)
    allocation_id = uuid.uuid4().hex
    artifact_id = uuid.uuid4().hex
    now = datetime.now(UTC)
    with engine.begin() as connection:
        connection.execute(
            insert(strategy_allocations).values(
                id=allocation_id,
                name=f"netting-sim-{allocation_id[:8]}",
                dataset="snapshot",
                status="active",
                is_legacy=False,
                allocation_method="fixed",
                decision_frequency="monthly",
                lookback_days=120,
                target_volatility=0.2,
                max_pairwise_correlation=0.8,
                max_strategy_weight=0.7,
                max_member_drawdown=0.08,
                max_drawdown_reduce=0.10,
                max_drawdown_liquidate=0.15,
                total_capital=1_000_000,
                cash_reserve=0,
                nav=1_000_000,
                high_water_mark=1_000_000,
                analysis_json={},
                created_by="test",
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            insert(strategy_allocation_artifacts).values(
                id=artifact_id,
                allocation_id=allocation_id,
                decision_date=date(2026, 7, 10),
                inputs_as_of=date(2026, 7, 9),
                valid_until=date(2026, 8, 20),
                member_weights_json={"m1": 0.5, "m2": 0.5},
                analysis_json={},
                artifact_hash="a" * 64,
                created_at=now,
            )
        )
    return allocation_id, artifact_id


def _netting_plan(database_url: str, allocation_id: str, artifact_id: str) -> dict:
    store = AccountNettingStore(database_url)
    return store.create_plan(
        actor="netting-operator",
        account_id=allocation_id,
        allocation_artifact_id=artifact_id,
        decision_date=date(2026, 7, 10),
        inputs_as_of=date(2026, 7, 9),
        policy_version="allocation:fixed/monthly",
        member_budgets={"m1": 0.5, "m2": 0.5},
        member_targets={"m1": {"SH600000": 0.2}, "m2": {"SH600000": 0.1}},
        total_capital=1_000_000.0,
    )


def _create_allocation_simulation(
    database_url: str, allocation_id: str
) -> tuple[SimulationStore, dict]:
    store = SimulationStore(database_url)
    simulation = store.create(
        name="netting-bound simulation",
        source_type="allocation",
        source_id=allocation_id,
        daily_dataset=_daily_dataset(),
        execution_dataset=_execution_dataset(),
        initial_cash=1_000_000,
        execution_policy={"execution_algorithm": "twap"},
        cost_schedule_version=COST_SCHEDULE_VERSION,
        actor="test",
    )
    store.set_status(simulation["id"], "active")
    return store, simulation


def test_netting_plan_binding_records_contributions(database_url: str, tmp_path) -> None:
    allocation_id, artifact_id = _seed_active_allocation(database_url)
    plan = _netting_plan(database_url, allocation_id, artifact_id)
    store, simulation = _create_allocation_simulation(database_url, allocation_id)

    batch, created = store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[_buy_action("SH600000", 1_000)],
        target_version=f"netting-plan:{plan['id']}",
        actor="test",
        account_netting_plan_id=plan["id"],
    )
    assert created is True
    assert batch["account_netting_plan_id"] == plan["id"]
    (order,) = _orders(store, simulation)
    assert order["account_netting_plan_id"] == plan["id"]
    contributions = order["strategy_contributions_json"]
    assert contributions is not None
    assert contributions["net_delta"] == pytest.approx(0.15)
    members = contributions["members"]
    assert members["m1"]["net_contribution"] == pytest.approx(0.10)
    assert members["m2"]["net_contribution"] == pytest.approx(0.05)

    completed = _process(store, simulation, batch)
    assert completed["status"] == "succeeded"
    assert _orders(store, simulation)[0]["status"] == STATUS_FILLED


def test_netting_plan_binding_requires_matching_account(database_url: str, tmp_path) -> None:
    allocation_id, artifact_id = _seed_active_allocation(database_url)
    plan = _netting_plan(database_url, allocation_id, artifact_id)
    # 非 allocation 来源的账户不得绑定净额计划（fail closed）。
    store, simulation = _create_simulation(database_url, tmp_path)
    with pytest.raises(ValueError, match="does not match the simulation account"):
        store.create_order_plan_batch(
            simulation["id"],
            trade_date=TRADE_DATE,
            actions=[_buy_action("SH600000", 100)],
            target_version="target-v1",
            actor="test",
            account_netting_plan_id=plan["id"],
        )
    with pytest.raises(KeyError):
        store.create_order_plan_batch(
            simulation["id"],
            trade_date=TRADE_DATE,
            actions=[_buy_action("SH600000", 100)],
            target_version="target-v1",
            actor="test",
            account_netting_plan_id="missing-plan",
        )


def test_stale_open_order_expires_before_execution(database_url: str, tmp_path) -> None:
    store, simulation = _create_simulation(database_url, tmp_path)
    batch, _ = store.create_order_plan_batch(
        simulation["id"],
        trade_date=TRADE_DATE,
        actions=[_buy_action("SH600000", 1_000)],
        target_version="target-v1",
        actor="test",
        limit_prices={"SH600000": 9.5},
        not_after=datetime.combine(TRADE_DATE, time(15, 0), SHANGHAI),
    )
    # 第 1 天价格保护不成交，窗口当日结束 → expired 而非 open。
    _process(store, simulation, batch, price=10.0, day=TRADE_DATE)
    (order,) = _orders(store, simulation)
    assert order["status"] == "expired"
    # 之后的批次不得再执行该订单（fail closed）。
    batch2, _ = store.create_order_plan_batch(
        simulation["id"],
        trade_date=DAY2,
        actions=[_buy_action("SH600000", 100)],
        target_version="target-v2",
        actor="test",
    )
    _process(store, simulation, batch2, price=9.0, day=DAY2)
    fills = store.rows(simulation["id"], "fills")
    assert all(int(fill["quantity"]) != 1_000 for fill in fills)


def test_order_plan_batch_requires_nonempty_plan_and_active_portfolio(
    database_url: str, tmp_path
) -> None:
    store, simulation = _create_simulation(database_url, tmp_path)
    with pytest.raises(ValueError, match="no keep/cancel/replace/new entry"):
        store.create_order_plan_batch(
            simulation["id"],
            trade_date=TRADE_DATE,
            actions=[{"instrument": "SH600000", "order_plan": []}],
            target_version="target-v1",
            actor="test",
        )
    store.set_status(simulation["id"], "paused")
    with pytest.raises(ValueError, match="not active"):
        store.create_order_plan_batch(
            simulation["id"],
            trade_date=TRADE_DATE,
            actions=[_buy_action("SH600000", 100)],
            target_version="target-v1",
            actor="test",
        )
