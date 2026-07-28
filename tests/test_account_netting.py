from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from governance_fixtures import DATASET_IDENTITY
from sqlalchemy import select, update
from test_allocation_policy_guards_db import _create_allocation, _qlib_doubles, _two_versions

from quant_data.database import (
    account_netting_plans,
    open_database,
    strategy_allocation_artifacts,
    strategy_allocation_members,
    strategy_allocations,
    strategy_versions,
)
from quant_platform.account_netting import (
    NETTING_PLAN_VERSION,
    AccountNettingStore,
    build_account_netting_plan,
    net_member_demands,
    plan_idempotency_key,
)
from quant_platform.allocation_store import AllocationStore
from quant_platform.portfolio_policy import POLICY_VERSION
from quant_platform.qlib_backtest import QLIB_ENGINE_VERSION
from quant_platform.recommendation_store import RecommendationStore

ARTIFACT = "artifact-1"
DECISION = date(2026, 7, 20)
AS_OF = date(2026, 7, 17)


def _plan(**overrides) -> dict:
    options = {
        "account_id": "account-1",
        "allocation_artifact_id": ARTIFACT,
        "decision_date": DECISION,
        "inputs_as_of": AS_OF,
        "policy_version": "allocation:inverse_volatility/monthly",
        "member_budgets": {"m1": 0.5, "m2": 0.5},
        "member_targets": {"m1": {}, "m2": {}},
        "total_capital": 1_000_000.0,
    }
    options.update(overrides)
    return build_account_netting_plan(**options)


# ---------------------------------------------------------------------------
# Netting algebra (pure)
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_same_direction_demands_add_up() -> None:
    plan = _plan(
        member_targets={
            "m1": {"SH600000": 0.20},
            "m2": {"SH600000": 0.10, "SH600001": 0.05},
        }
    )
    # 0.5×0.20 + 0.5×0.10 = 0.15；SH600001 = 0.5×0.05 = 0.025
    assert plan["net_targets"]["SH600000"]["weight"] == pytest.approx(0.15)
    assert plan["net_targets"]["SH600001"]["weight"] == pytest.approx(0.025)
    assert plan["net_trades"]["SH600000"]["side"] == "buy"
    assert plan["cash_weight"] == pytest.approx(1.0 - 0.175)
    contributions = plan["strategy_contributions"]["SH600000"]["members"]
    assert contributions["m1"]["gross_delta"] == pytest.approx(0.10)
    assert contributions["m2"]["gross_delta"] == pytest.approx(0.05)
    # 同向无抵消：净贡献等于毛需求
    assert contributions["m1"]["net_contribution"] == pytest.approx(0.10)
    assert contributions["m2"]["net_contribution"] == pytest.approx(0.05)


@pytest.mark.no_database
def test_opposite_demands_partially_offset_and_attribute_pro_rata() -> None:
    plan = _plan(
        member_targets={"m1": {"SH600000": 0.20}, "m2": {}},
        member_current_weights={"m1": {}, "m2": {"SH600000": 0.12}},
    )
    # 毛需求：m1 +0.10（买），m2 −0.06（卖）→ 净 +0.04；净目标 0.10，账户现仓 0.06
    assert plan["net_targets"]["SH600000"]["weight"] == pytest.approx(0.10)
    assert plan["net_trades"]["SH600000"]["delta_weight"] == pytest.approx(0.04)
    contributions = plan["strategy_contributions"]["SH600000"]["members"]
    assert contributions["m1"]["gross_delta"] == pytest.approx(0.10)
    assert contributions["m2"]["gross_delta"] == pytest.approx(-0.06)
    # 同向比例分配：买方独占净额，被抵消方净贡献为零
    assert contributions["m1"]["net_contribution"] == pytest.approx(0.04)
    assert contributions["m2"]["net_contribution"] == pytest.approx(0.0)


@pytest.mark.no_database
def test_winning_side_shares_net_pro_rata() -> None:
    net, contributions = net_member_demands(
        {
            "a1": {"X": 0.10},
            "a2": {"X": 0.05},
            "b": {"X": -0.06},
        }
    )
    assert net["X"] == pytest.approx(0.09)
    members = contributions["X"]["members"]
    assert members["a1"]["net_contribution"] == pytest.approx(0.09 * 0.10 / 0.15)
    assert members["a2"]["net_contribution"] == pytest.approx(0.09 * 0.05 / 0.15)
    assert members["b"]["net_contribution"] == pytest.approx(0.0)


@pytest.mark.no_database
def test_fully_offsetting_demands_net_to_zero() -> None:
    plan = _plan(
        member_targets={"m1": {"SH600000": 0.10}, "m2": {}},
        member_current_weights={"m1": {}, "m2": {"SH600000": 0.10}},
    )
    # m1 +0.05 与 m2 −0.05 完全抵消：净目标两侧相等，无净交易
    assert plan["net_targets"]["SH600000"]["weight"] == pytest.approx(0.05)
    assert "SH600000" not in plan["net_trades"]
    members = plan["strategy_contributions"]["SH600000"]["members"]
    assert members["m1"]["net_contribution"] == pytest.approx(0.0)
    assert members["m2"]["net_contribution"] == pytest.approx(0.0)


@pytest.mark.no_database
def test_account_hard_constraint_clamps_net_target_into_cash() -> None:
    plan = _plan(
        member_targets={"m1": {"SH600000": 0.30}, "m2": {"SH600000": 0.10}},
        max_instrument_weight=0.15,
    )
    # 净目标 0.20 被账户硬约束压到 0.15，溢出 0.05 转现金
    assert plan["net_targets"]["SH600000"]["weight"] == pytest.approx(0.15)
    assert plan["constraint_clamps"]["SH600000"]["raw_weight"] == pytest.approx(0.20)
    assert plan["cash_weight"] == pytest.approx(0.85)
    contributions = plan["strategy_contributions"]["SH600000"]["members"]
    total = sum(item["net_contribution"] for item in contributions.values())
    assert total == pytest.approx(0.15)


@pytest.mark.no_database
def test_plan_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="exceed investable capital"):
        _plan(member_budgets={"m1": 0.7, "m2": 0.7})
    with pytest.raises(ValueError, match="non-negative"):
        _plan(member_targets={"m1": {"SH600000": -0.1}, "m2": {}})
    with pytest.raises(ValueError, match="execution policy"):
        _plan(execution_policy="market_on_close")
    with pytest.raises(ValueError, match="member sleeve"):
        _plan(member_targets={"m1": {"SH600000": 0.6, "SH600001": 0.6}, "m2": {}})
    with pytest.raises(ValueError, match="finite and non-negative"):
        _plan(member_budgets={"m1": float("nan"), "m2": 0.5})
    with pytest.raises(ValueError, match="finite and non-negative"):
        _plan(member_targets={"m1": {"SH600000": float("nan")}, "m2": {}})
    with pytest.raises(ValueError, match="current weights"):
        _plan(member_current_weights={"m1": {"SH600000": float("inf")}})
    with pytest.raises(ValueError, match="matching budget"):
        _plan(member_targets={"m1": {}, "m2": {}, "unknown": {"SH600000": 1.0}})
    with pytest.raises(ValueError, match="finite values"):
        net_member_demands({"m1": {"SH600000": float("nan")}})


@pytest.mark.no_database
def test_idempotency_key_semantics() -> None:
    base = {
        "account_id": "account-1",
        "allocation_artifact_id": ARTIFACT,
        "decision_date": DECISION,
        "inputs_as_of": AS_OF,
        "policy_version": "allocation:fixed/monthly",
        "tranche_index": 0,
    }
    key = plan_idempotency_key(**base)
    assert key == plan_idempotency_key(**base)
    assert key != plan_idempotency_key(**{**base, "tranche_index": 1})
    assert key != plan_idempotency_key(**{**base, "decision_date": date(2026, 7, 21)})
    # strategy_id 不是幂等键成分：成员集合变化不改变键（内容哈希才变）
    first = _plan(member_targets={"m1": {"SH600000": 0.1}, "m2": {}})
    second = _plan(member_targets={"m1": {"SH600000": 0.2}, "m2": {}})
    assert first["plan_key"] == second["plan_key"]
    assert first["plan_hash"] != second["plan_hash"]
    assert first["plan_version"] == NETTING_PLAN_VERSION


# ---------------------------------------------------------------------------
# DB persistence: idempotent create/replay
# ---------------------------------------------------------------------------


def test_create_plan_is_idempotent_and_fail_closed_on_conflict(database_url: str) -> None:
    store = AccountNettingStore(database_url)
    artifact_id = _seed_artifact(database_url)
    kwargs = {**_plan_kwargs(), "allocation_artifact_id": artifact_id}
    plan = store.create_plan(actor="netting-operator", **kwargs)
    assert plan["idempotent_replay"] is False
    replay = store.create_plan(actor="netting-operator", **kwargs)
    assert replay["idempotent_replay"] is True
    assert replay["id"] == plan["id"]
    with store.engine.connect() as connection:
        rows = connection.execute(select(account_netting_plans)).all()
    assert len(rows) == 1
    with pytest.raises(ValueError, match="idempotency key conflict"):
        store.create_plan(
            actor="netting-operator",
            **{**kwargs, "member_targets": {"m1": {"SH600000": 0.3}, "m2": {}}},
        )


def _plan_kwargs() -> dict:
    return {
        "account_id": "account-db",
        "allocation_artifact_id": None,  # filled by _seed_artifact
        "decision_date": DECISION,
        "inputs_as_of": AS_OF,
        "policy_version": "allocation:fixed/monthly",
        "member_budgets": {"m1": 0.5, "m2": 0.5},
        "member_targets": {"m1": {"SH600000": 0.2}, "m2": {"SH600000": 0.1}},
        "total_capital": 1_000_000.0,
    }


def _seed_artifact(database_url: str) -> str:
    """A minimal allocation + artifact row to satisfy the plan foreign key."""

    import uuid

    engine = open_database(database_url)
    allocation_id = uuid.uuid4().hex
    artifact_id = uuid.uuid4().hex
    now = datetime.now(UTC)
    with engine.begin() as connection:
        connection.execute(
            strategy_allocations.insert().values(
                id=allocation_id,
                name=f"netting-test-{allocation_id[:8]}",
                dataset="snapshot",
                status="paused",
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
            strategy_allocation_artifacts.insert().values(
                id=artifact_id,
                allocation_id=allocation_id,
                decision_date=DECISION,
                inputs_as_of=AS_OF,
                valid_until=date(2026, 8, 20),
                member_weights_json={"m1": 0.5, "m2": 0.5},
                analysis_json={},
                artifact_hash="a" * 64,
                created_at=now,
            )
        )
    return artifact_id


# ---------------------------------------------------------------------------
# DB assembly from allocation ledger
# ---------------------------------------------------------------------------


def test_build_plan_for_allocation(database_url: str, tmp_path: Path, monkeypatch) -> None:
    _qlib_doubles(monkeypatch)
    version_ids = _two_versions(database_url, tmp_path)
    store = AllocationStore(database_url)
    allocation = _create_allocation(store, version_ids, "netting allocation")
    # Fixture shortcut: real approval sets promotion_stage="paper"; this test
    # exercises netting, not the forward gate, so it marks the versions
    # enabled directly (production must pass PromotionStore.promote).
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id.in_(version_ids))
            .values(promotion_stage="recommendation_enabled")
        )
    recommendations = RecommendationStore(database_url)
    engine = open_database(database_url)
    targets = [
        {"SH600000": 0.20, "SH600001": 0.10},
        {"SH600000": 0.10},
    ]
    for version_id, holdings in zip(version_ids, targets, strict=True):
        portfolio = recommendations.create(
            name=f"member {version_id[:8]}",
            strategy_version_id=version_id,
            dataset="allocation-data",
            hypothetical_initial_value=500_000,
            actor="operator-a",
            recommendation_scope="allocation_member",
        )
        snapshot, _ = recommendations.create_snapshot(
            portfolio_id=portfolio["id"],
            as_of_date=date(2026, 7, 20),
            dataset="allocation-data",
            dataset_identity_sha256=DATASET_IDENTITY,
        )
        recommendations.apply_result(
            snapshot["id"],
            {
                "status": "ok",
                "portfolio_id": portfolio["id"],
                "strategy_version_id": version_id,
                "dataset": "allocation-data",
                "dataset_identity_sha256": DATASET_IDENTITY,
                "as_of_date": "2026-07-20",
                "effective_date": "2026-07-21",
                "policy_version": POLICY_VERSION,
                "backtest_engine_version": QLIB_ENGINE_VERSION,
                "cost_model": snapshot["cost_model"],
                "cash_weight": 1.0 - sum(holdings.values()),
                "holdings": [
                    {
                        "instrument": instrument,
                        "weight": weight,
                        "previous_weight": 0.0,
                        "weight_change": weight,
                        "action": "buy",
                        "reason": "governed target",
                    }
                    for instrument, weight in holdings.items()
                ],
            },
        )
        with engine.begin() as connection:
            connection.execute(
                update(strategy_allocation_members)
                .where(
                    strategy_allocation_members.c.allocation_id == allocation["id"],
                    strategy_allocation_members.c.strategy_version_id == version_id,
                )
                .values(recommendation_portfolio_id=portfolio["id"])
            )

    netting = AccountNettingStore(database_url)
    plan = netting.build_plan_for_allocation(allocation["id"], actor="netting-operator")

    budgets = {
        member["strategy_version_id"]: member["target_weight"]
        for member in store.get(allocation["id"])["members"]
    }
    expected = (
        budgets[version_ids[0]] * 0.20 + budgets[version_ids[1]] * 0.10
    )
    assert plan["net_targets"]["SH600000"]["weight"] == pytest.approx(expected)
    assert plan["net_targets"]["SH600001"]["weight"] == pytest.approx(
        budgets[version_ids[0]] * 0.10
    )
    assert plan["execution_policy"] == "next_bar"
    assert plan["total_capital"] == pytest.approx(1_000_000.0)
    assert plan["cash_weight"] == pytest.approx(
        1.0 - expected - budgets[version_ids[0]] * 0.10
    )
    contributions = plan["strategy_contributions"]["SH600000"]["members"]
    assert contributions[version_ids[0]]["gross_delta"] == pytest.approx(
        budgets[version_ids[0]] * 0.20
    )
    # 幂等重放：同一决策日重试返回同一行
    replay = netting.build_plan_for_allocation(allocation["id"], actor="netting-operator")
    assert replay["idempotent_replay"] is True
    assert replay["id"] == plan["id"]
