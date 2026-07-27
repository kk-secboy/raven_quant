"""Two-dimension recommendation action model tests (design draft 8.4)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from governance_fixtures import DATASET_IDENTITY, create_strategy_version
from sqlalchemy import update

from quant_data.database import strategy_versions
from quant_platform.portfolio_policy import POLICY_VERSION
from quant_platform.qlib_backtest import QLIB_ENGINE_VERSION
from quant_platform.recommendation_actions import (
    RECOMMENDATION_ACTION_MODEL_VERSION,
    normalize_open_order,
    plan_account_actions,
    plan_instrument_action,
    projected_position,
)
from quant_platform.recommendation_store import RecommendationStore

NOW = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
FUTURE = NOW + timedelta(hours=2)
PAST = NOW - timedelta(hours=1)


def _order(side: str, requested: int, filled: int = 0, *, expired: bool = False, oid: str = ""):
    return {
        "order_id": oid or f"{side}-{requested}-{filled}",
        "side": side,
        "requested_quantity": requested,
        "filled_quantity": filled,
        "expires_at": PAST if expired else FUTURE,
        "created_at": PAST - timedelta(hours=1),
    }


def _plan(**overrides):
    defaults = {
        "instrument": "SH600000",
        "target_quantity": 100,
        "filled_position": 0,
        "now": NOW,
    }
    defaults.update(overrides)
    return plan_instrument_action(**defaults)


no_database = pytest.mark.no_database


# ---------------------------------------------------------------------------
# 三个典型语义（设计 8.4 行 834）
# ---------------------------------------------------------------------------


@no_database
def test_typical_buy_covered_by_open_order_adds_nothing() -> None:
    # 持仓 0、目标 100、未成交买单 100 → BUY，新单 0，projected=100。
    result = _plan(open_orders=[_order("buy", 100)])
    assert result["action"] == "BUY"
    assert result["execution_state"] == "READY"
    assert result["projected_position"] == 100
    assert result["order_plan"] == [
        {
            "op": "keep",
            "order_id": "buy-100-0",
            "side": "buy",
            "quantity": 100,
            "reason": "covers_final_target",
        }
    ]


@no_database
def test_typical_buy_partially_filled_is_partial() -> None:
    # 已成交 40 计入 filled_position，有效剩余 60 → projected=100，状态 PARTIAL。
    result = _plan(filled_position=40, open_orders=[_order("buy", 100, 40)])
    assert result["action"] == "BUY"
    assert result["execution_state"] == "PARTIAL"
    assert result["projected_position"] == 100
    assert [entry["op"] for entry in result["order_plan"]] == ["keep"]


@no_database
def test_typical_hold_cancels_deviating_buy() -> None:
    # 持仓 100、目标 100、仍有买单 20 → HOLD 并取消多余买单。
    result = _plan(
        target_quantity=100,
        filled_position=100,
        open_orders=[_order("buy", 20)],
    )
    assert result["action"] == "HOLD"
    assert result["execution_state"] == "CANCELLED"
    assert result["order_plan"][0]["op"] == "cancel"
    assert result["order_plan"][0]["reason"] == "target_already_filled"


@no_database
def test_typical_exit_reports_remaining_sellable() -> None:
    # 持仓 100、目标 0、未成交卖单 60 → EXIT，尚需卖出 40。
    result = _plan(
        target_quantity=0,
        filled_position=100,
        open_orders=[_order("sell", 60)],
    )
    assert result["action"] == "EXIT"
    assert result["execution_state"] == "READY"
    assert result["projected_position"] == 40
    assert [entry["op"] for entry in result["order_plan"]] == ["keep", "new"]
    assert result["order_plan"][1]["quantity"] == 40
    assert "remaining_sellable_quantity=100" in result["notes"]


# ---------------------------------------------------------------------------
# projected_position 与动作/状态矩阵
# ---------------------------------------------------------------------------


@no_database
def test_projected_position_counts_only_valid_orders() -> None:
    orders = [
        {"side": "buy", "requested_quantity": 100, "filled_quantity": 40, "expires_at": FUTURE},
        {"side": "sell", "requested_quantity": 30, "filled_quantity": 0, "expires_at": FUTURE},
        {"side": "buy", "requested_quantity": 50, "filled_quantity": 0, "expires_at": PAST},
    ]
    normalized = [normalize_open_order(order, now=NOW) for order in orders]
    # 有效买单剩 60、有效卖单剩 30；过期买单不计。
    assert projected_position(200, normalized) == 230


@no_database
def test_buy_without_coverage_creates_new_order() -> None:
    result = _plan()
    assert result["action"] == "BUY"
    assert result["execution_state"] == "READY"
    assert result["order_plan"][-1]["op"] == "new"
    assert result["order_plan"][-1]["quantity"] == 100


@no_database
def test_sell_partial_target() -> None:
    result = _plan(target_quantity=30, filled_position=100)
    assert result["action"] == "SELL"
    assert result["execution_state"] == "READY"
    assert result["order_plan"][-1] == {
        "op": "new",
        "order_id": "",
        "side": "sell",
        "quantity": 70,
        "reason": "target_minus_projected_position",
    }


@no_database
def test_sell_odd_lot_is_not_rounded() -> None:
    result = _plan(target_quantity=0, filled_position=105)
    assert result["action"] == "EXIT"
    assert result["order_plan"][-1]["quantity"] == 105


@no_database
def test_buy_rounds_down_to_lot_increment() -> None:
    result = _plan(target_quantity=250)
    assert result["order_plan"][-1]["quantity"] == 200
    assert "new_buy_rounded_down_to_lot" in result["notes"]


@no_database
def test_opposite_side_order_is_cancelled() -> None:
    result = _plan(open_orders=[_order("sell", 50)])
    assert result["projected_position"] == -50
    ops = [(entry["op"], entry["side"]) for entry in result["order_plan"]]
    assert ops == [("cancel", "sell"), ("new", "buy")]


@no_database
def test_overshooting_order_is_replaced_not_topped_up() -> None:
    result = _plan(open_orders=[_order("buy", 120)])
    assert result["order_plan"] == [
        {
            "op": "replace",
            "order_id": "buy-120-0",
            "side": "buy",
            "quantity": 100,
            "previous_quantity": 120,
            "reason": "trim_to_final_target",
        }
    ]


@no_database
def test_excess_same_side_orders_are_cancelled() -> None:
    result = _plan(open_orders=[_order("buy", 100), _order("buy", 50)])
    ops = [entry["op"] for entry in result["order_plan"]]
    assert ops == ["keep", "cancel"]


@no_database
def test_expired_coverage_replans_as_expired_state() -> None:
    # 模拟订单当日 15:00 过期：过期覆盖不能当作仍有效，目标未达 → EXPIRED 并重报新单。
    result = _plan(open_orders=[_order("buy", 100, expired=True)])
    assert result["action"] == "BUY"
    assert result["projected_position"] == 0
    assert result["execution_state"] == "EXPIRED"
    assert result["order_plan"][-1]["op"] == "new"
    assert result["order_plan"][-1]["quantity"] == 100


@no_database
def test_expired_with_valid_coverage_is_not_expired_state() -> None:
    result = _plan(
        open_orders=[_order("buy", 100, expired=True), _order("buy", 100, oid="live")]
    )
    assert result["execution_state"] == "READY"
    assert result["order_plan"][0]["order_id"] == "live"


@no_database
def test_hard_gate_keeps_action_and_marks_blocked() -> None:
    result = _plan(hard_blocked_reason="data_stale")
    assert result["action"] == "BUY"
    assert result["execution_state"] == "BLOCKED"
    assert result["blocked_reason"] == "data_stale"
    # 阻断不删除目标与计划，只阻止释放。
    assert result["order_plan"][-1]["op"] == "new"


@no_database
def test_soft_unexecutable_marks_wait() -> None:
    result = _plan(not_executable_reason="limit_up")
    assert result["action"] == "BUY"
    assert result["execution_state"] == "WAIT"
    assert result["wait_reason"] == "limit_up"


@no_database
def test_exit_without_sellable_waits() -> None:
    result = _plan(target_quantity=0, filled_position=100, sellable_quantity=0)
    assert result["action"] == "EXIT"
    assert result["execution_state"] == "WAIT"
    assert result["wait_reason"] == "sellable_quantity_unavailable"
    assert "sell_clamped_by_sellable_quantity" in result["notes"]


@no_database
def test_no_valid_target_is_no_action_and_keeps_previous() -> None:
    result = _plan(target_quantity=None, filled_position=100)
    assert result["action"] == "NO_ACTION"
    assert result["order_plan"] == []
    assert "no_valid_target_previous_retained" in result["notes"]


@no_database
def test_zero_target_zero_position_is_no_action() -> None:
    result = _plan(target_quantity=0, filled_position=0)
    assert result["action"] == "NO_ACTION"
    assert result["execution_state"] == "READY"


@no_database
def test_zero_target_zero_position_cancels_deviating_order_as_hold() -> None:
    result = _plan(target_quantity=0, filled_position=0, open_orders=[_order("buy", 100)])
    assert result["action"] == "HOLD"
    assert result["execution_state"] == "CANCELLED"


@no_database
def test_sell_new_order_respects_sellable_headroom() -> None:
    # 可卖 100，有效卖单已占 60：新卖单最多再报 40。
    result = _plan(
        target_quantity=0,
        filled_position=100,
        sellable_quantity=100,
        open_orders=[_order("sell", 60)],
    )
    assert result["order_plan"][-1]["quantity"] == 40
    assert result["execution_state"] == "READY"


@no_database
def test_partial_state_requires_carried_order() -> None:
    # 部分成交的订单被计划取消（已达目标）→ 不是 PARTIAL，是 CANCELLED。
    result = _plan(
        target_quantity=100,
        filled_position=100,
        open_orders=[_order("buy", 40, 20)],
    )
    assert result["execution_state"] == "CANCELLED"


@no_database
def test_naive_expiry_is_interpreted_in_now_timezone() -> None:
    naive = datetime(2026, 7, 15, 15, 0)
    result = _plan(
        open_orders=[
            {
                "side": "buy",
                "requested_quantity": 100,
                "filled_quantity": 0,
                "expires_at": naive,
            }
        ],
        now=datetime(2026, 7, 15, 16, 0, tzinfo=UTC),
    )
    assert result["execution_state"] == "EXPIRED"


@no_database
def test_input_validation_failures() -> None:
    with pytest.raises(ValueError, match="filled position"):
        _plan(filled_position=-1)
    with pytest.raises(ValueError, match="sellable"):
        _plan(filled_position=100, sellable_quantity=200)
    with pytest.raises(ValueError, match="target quantity"):
        _plan(target_quantity=-5)
    with pytest.raises(ValueError, match="side"):
        _plan(open_orders=[{"side": "hold", "requested_quantity": 1}])


@no_database
def test_plan_account_actions_sorts_and_vectors() -> None:
    items = plan_account_actions(
        [
            {"instrument": "SZ000001", "target_quantity": 0, "filled_position": 100},
            {"instrument": "SH600000", "target_quantity": 100, "filled_position": 0},
        ],
        now=NOW,
    )
    assert [item["instrument"] for item in items] == ["SH600000", "SZ000001"]
    assert items[0]["action"] == "BUY"
    assert items[1]["action"] == "EXIT"
    assert all(item["model_version"] == RECOMMENDATION_ACTION_MODEL_VERSION for item in items)


# ---------------------------------------------------------------------------
# 快照集成端到端（DB）
# ---------------------------------------------------------------------------


def _make_succeeded_snapshot(database_url: str, tmp_path):
    version_id = create_strategy_version(database_url, tmp_path)
    recommendations = RecommendationStore(database_url)
    with recommendations.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved")
        )
    portfolio = recommendations.create(
        name="two-dim action target",
        strategy_version_id=version_id,
        dataset="snapshot",
        hypothetical_initial_value=1_000_000,
        actor="test",
    )
    snapshot, created = recommendations.create_snapshot(
        portfolio_id=portfolio["id"],
        as_of_date=date(2026, 7, 10),
        dataset="snapshot",
        dataset_identity_sha256=DATASET_IDENTITY,
    )
    assert created is True
    recommendations.apply_result(
        snapshot["id"],
        {
            "status": "ok",
            "portfolio_id": portfolio["id"],
            "strategy_version_id": version_id,
            "dataset": "snapshot",
            "dataset_identity_sha256": DATASET_IDENTITY,
            "as_of_date": "2026-07-10",
            "effective_date": "2026-07-13",
            "policy_version": POLICY_VERSION,
            "backtest_engine_version": QLIB_ENGINE_VERSION,
            "cost_model": snapshot["cost_model"],
            "cash_weight": 0.999,
            "holdings": [
                {
                    "instrument": "SH600000",
                    "weight": 0.001,
                    "previous_weight": 0.0,
                    "weight_change": 0.001,
                    "action": "increase",
                    "reason": "governed target",
                }
            ],
        },
    )
    return recommendations, snapshot["id"]


def test_attach_account_actions_end_to_end(database_url: str, tmp_path) -> None:
    recommendations, snapshot_id = _make_succeeded_snapshot(database_url, tmp_path)
    updated = recommendations.attach_account_actions(
        snapshot_id,
        account_state={
            # 权重 0.001 × 1,000,000 ÷ 10 = 100 股目标；有效买单已覆盖 → BUY/READY 不补单。
            "SH600000": {
                "reference_price": 10.0,
                "filled_position": 0,
                "open_orders": [
                    {
                        "order_id": "sim-order-1",
                        "side": "buy",
                        "requested_quantity": 100,
                        "filled_quantity": 0,
                        "expires_at": datetime(2026, 7, 16, 15, 0, tzinfo=UTC),
                    }
                ],
            },
            # 不在目标内但持仓 200：目标 0 → EXIT。
            "SZ000001": {"filled_position": 200, "sellable_quantity": 200},
        },
        now=datetime(2026, 7, 15, 10, 0, tzinfo=UTC),
    )
    plan = updated["account_actions"]
    assert plan["model_version"] == RECOMMENDATION_ACTION_MODEL_VERSION
    items = {item["instrument"]: item for item in plan["items"]}
    assert items["SH600000"]["action"] == "BUY"
    assert items["SH600000"]["execution_state"] == "READY"
    assert items["SH600000"]["target_quantity"] == 100
    assert items["SH600000"]["projected_position"] == 100
    assert [entry["op"] for entry in items["SH600000"]["order_plan"]] == ["keep"]
    assert items["SZ000001"]["action"] == "EXIT"
    assert items["SZ000001"]["order_plan"][-1]["quantity"] == 200
    # 既有 increase/decrease 导出保持兼容，不被两维模型改写。
    assert updated["holdings"][0]["action"] == "increase"


def test_attach_account_actions_requires_succeeded_snapshot(
    database_url: str, tmp_path
) -> None:
    version_id = create_strategy_version(database_url, tmp_path)
    recommendations = RecommendationStore(database_url)
    with recommendations.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved")
        )
    portfolio = recommendations.create(
        name="two-dim action gate",
        strategy_version_id=version_id,
        dataset="snapshot",
        hypothetical_initial_value=1_000_000,
        actor="test",
    )
    snapshot, _ = recommendations.create_snapshot(
        portfolio_id=portfolio["id"],
        as_of_date=date(2026, 7, 10),
        dataset="snapshot",
        dataset_identity_sha256=DATASET_IDENTITY,
    )
    with pytest.raises(ValueError, match="succeeded"):
        recommendations.attach_account_actions(snapshot["id"], account_state={})


def test_attach_account_actions_requires_price_for_target_quantity(
    database_url: str, tmp_path
) -> None:
    recommendations, snapshot_id = _make_succeeded_snapshot(database_url, tmp_path)
    with pytest.raises(ValueError, match="reference price"):
        recommendations.attach_account_actions(
            snapshot_id, account_state={"SH600000": {"filled_position": 0}}
        )
