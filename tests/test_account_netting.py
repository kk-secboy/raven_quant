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
from quant_platform.three_horizon_account import (
    HORIZON_WEIGHTS,
    _active_horizon_fixed_weights,
)

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
def test_progressive_horizon_plan_attributes_inherited_reduction_to_old_sleeve() -> None:
    plan = _plan(
        member_budgets={"short": 0.20, "swing": 0.40},
        member_targets={
            "short": {"SH600000": 0.50},
            "swing": {"SH600001": 1.0},
        },
        member_current_weights={
            "short": {"SH600000": 1.0},
            "swing": {},
        },
        total_capital=800_000.0,
    )

    assert plan["net_trades"]["SH600000"]["side"] == "sell"
    assert plan["strategy_contributions"]["SH600000"]["members"]["short"][
        "net_contribution"
    ] == pytest.approx(-0.10)
    assert plan["net_trades"]["SH600001"]["side"] == "buy"


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
def test_account_industry_cap_is_applied_after_cross_strategy_netting() -> None:
    plan = _plan(
        member_targets={
            "m1": {"A": 0.40, "B": 0.20},
            "m2": {"A": 0.20, "C": 0.20},
        },
        industry_memberships={"A": "bank", "B": "bank", "C": "technology"},
        max_industry_weight=0.25,
    )

    bank_weight = sum(
        target["weight"]
        for instrument, target in plan["net_targets"].items()
        if plan["industry_memberships"].get(instrument) == "bank"
    )
    assert bank_weight == pytest.approx(0.25)
    assert plan["industry_exposure"]["bank"] == pytest.approx(0.25)
    assert plan["industry_exposure"]["technology"] == pytest.approx(0.10)
    assert plan["cash_weight"] == pytest.approx(0.65)
    assert {
        item["reason"] for item in plan["industry_constraint_clamps"].values()
    } == {"account_industry_weight_cap"}
    for instrument, trade in plan["net_trades"].items():
        attributed = sum(
            item["net_contribution"]
            for item in plan["strategy_contributions"][instrument]["members"].values()
        )
        assert attributed == pytest.approx(trade["delta_weight"])


@pytest.mark.no_database
def test_missing_industry_metadata_fails_closed_with_explanation() -> None:
    plan = _plan(
        member_targets={"m1": {"UNKNOWN": 0.20}, "m2": {}},
        industry_memberships={},
        max_industry_weight=0.25,
    )

    assert plan["net_targets"] == {}
    assert plan["cash_weight"] == pytest.approx(1.0)
    assert plan["industry_constraint_clamps"]["UNKNOWN"] == {
        "industry": None,
        "raw_weight": pytest.approx(0.10),
        "clamped_weight": 0.0,
        "reason": "missing_industry_membership_blocks_target",
    }


@pytest.mark.no_database
def test_three_horizon_default_is_20_40_40_of_investable_capital() -> None:
    assert HORIZON_WEIGHTS == {
        "short_1_5d": 0.20,
        "swing_1_6m": 0.40,
        "long_1_3y": 0.40,
    }
    # Balanced account: sleeves consume 90% gross and preserve 10% cash.
    budgets = {key: value * 0.90 for key, value in HORIZON_WEIGHTS.items()}
    plan = build_account_netting_plan(
        account_id="primary",
        allocation_artifact_id=ARTIFACT,
        decision_date=DECISION,
        inputs_as_of=AS_OF,
        policy_version="three-horizon-balanced",
        member_budgets=budgets,
        member_targets={key: {f"{index:06d}.SZ": 1.0} for index, key in enumerate(budgets)},
        total_capital=100_000,
        execution_policy="open",
    )
    assert sum(item["weight"] for item in plan["net_targets"].values()) == pytest.approx(0.90)
    assert plan["cash_weight"] == pytest.approx(0.10)
    assert plan["execution_policy"] == "open"
    assert plan["member_targets"] == {
        key: {f"{index:06d}.SZ": 1.0}
        for index, key in enumerate(budgets)
    }


@pytest.mark.no_database
def test_verified_horizons_do_not_redistribute_missing_sleeve_budgets() -> None:
    short_only = _active_horizon_fixed_weights(
        {"short_1_5d": "short-version"}, max_gross_exposure=0.90
    )
    short_and_swing = _active_horizon_fixed_weights(
        {
            "short_1_5d": "short-version",
            "swing_1_6m": "swing-version",
        },
        max_gross_exposure=0.90,
    )

    assert short_only == {"short-version": pytest.approx(0.18)}
    assert short_and_swing == {
        "short-version": pytest.approx(0.18),
        "swing-version": pytest.approx(0.36),
    }
    assert 1.0 - sum(short_only.values()) == pytest.approx(0.82)
    assert 1.0 - sum(short_and_swing.values()) == pytest.approx(0.46)


@pytest.mark.no_database
def test_member_target_transition_preserves_exit_attribution_conservation() -> None:
    budgets = {"short": 0.18, "swing": 0.36, "long": 0.36}
    previous = {
        "short": {"SH600000": 1.0},
        "swing": {"SH600000": 0.5},
        "long": {"SH600001": 1.0},
    }
    targets = {
        "short": {},
        "swing": {"SH600000": 1.0},
        "long": {"SH600001": 1.0},
    }
    plan = build_account_netting_plan(
        account_id="primary",
        allocation_artifact_id=ARTIFACT,
        decision_date=DECISION,
        inputs_as_of=AS_OF,
        policy_version="three-horizon-transition",
        member_budgets=budgets,
        member_targets=targets,
        member_current_weights=previous,
        total_capital=100_000,
        execution_policy="open",
    )

    contribution = plan["strategy_contributions"]["SH600000"]
    assert contribution["members"]["short"]["gross_delta"] == pytest.approx(-0.18)
    assert contribution["members"]["swing"]["gross_delta"] == pytest.approx(0.18)
    assert contribution["net_delta"] == pytest.approx(0.0)
    assert sum(
        item["net_contribution"] for item in contribution["members"].values()
    ) == pytest.approx(plan["net_trades"].get("SH600000", {}).get("delta_weight", 0.0))


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
    # This test owns account-level netting arithmetic, not the production
    # health-evidence chain.  Supply an explicit authoritative gate double;
    # arbitrary fixture snapshots must never weaken the real fail-closed gate.
    monkeypatch.setattr(
        "quant_platform.account_netting.load_production_health_gate",
        lambda _connection, version_id: {
            "strategy_version_id": version_id,
            "health_status": "healthy",
            "allow_new_risk": True,
            "ready": True,
            "reasons": [],
            "snapshot_id": f"health-{version_id}",
        },
    )
    version_ids = _two_versions(database_url, tmp_path)
    store = AllocationStore(database_url)
    allocation = _create_allocation(store, version_ids, "netting allocation")
    # _two_versions advances both the version projection and its durable
    # promotion-stage evidence; production reaches this only through the gate.
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
                "reference_prices": {
                    instrument: 10.0 for instrument in holdings
                },
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
    assert plan["execution_policy"] == "open"
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

    live_nav = 800_000.0
    primary = {
        "portfolio_id": "persistent-primary-ledger",
        "source_id": allocation["id"],
        "nav": live_nav,
        "updated_at": "2026-07-21T16:00:00+08:00",
    }
    live_plan = netting.build_plan_for_allocation(
        allocation["id"],
        actor="netting-operator",
        primary_account=primary,
        max_gross_exposure=0.90,
    )
    assert live_plan["total_capital"] == pytest.approx(live_nav)
    assert live_plan["input_evidence"]["primary_account"]["portfolio_id"] == (
        "persistent-primary-ledger"
    )
    assert live_plan["net_targets"]["SH600000"]["target_value"] == pytest.approx(
        live_nav * live_plan["net_targets"]["SH600000"]["weight"]
    )
    assert sum(
        item["target_value"] for item in live_plan["net_targets"].values()
    ) <= live_nav * 0.90 + 1e-6
