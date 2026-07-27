"""Suspended-member budget renormalization (design 6.10 cash_fallback_policy).

Only frozen-decision-day artifact resolution re-solves budgets; suspending a
member mid-cycle changes nothing until the artifact expires, and then the
policy method re-solves on the active set while the suspended budget moves to
cash — never leverage, never ad-hoc re-estimation.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from governance_fixtures import PERIODS
from sqlalchemy import update
from test_allocation_policy_guards_db import (
    _create_allocation,
    _qlib_doubles,
    _two_versions,
)
from test_strategy_allocation_recommendations import _approve_version

from quant_data.database import (
    open_database,
    strategy_allocation_artifacts,
    strategy_allocation_members,
    strategy_allocations,
    strategy_versions,
)
from quant_platform.allocation_store import AllocationStore
from quant_platform.recommendation_store import RecommendationStore
from quant_platform.strategy_allocation import renormalize_budgets_for_suspended


def _analysis(members: dict[str, float]) -> dict:
    return {
        "members": {
            version_id: {
                "unscaled_weight": weight,
                "target_weight": weight,
                "annualized_volatility": 0.2,
                "risk_contribution": 1.0 / len(members),
                "risk_budget": 1.0 / len(members),
            }
            for version_id, weight in members.items()
        },
        "cash_weight": 1.0 - sum(members.values()),
    }


@pytest.mark.no_database
def test_renormalize_scales_solved_weights_to_active_mass() -> None:
    result = renormalize_budgets_for_suspended(
        _analysis({"a": 0.6, "b": 0.4}),
        previous_weights={"a": 0.3, "b": 0.2, "c": 0.3},
        suspended={"c": 0.3},
    )
    # 活跃质量 0.5 保持：求解分布 (0.6, 0.4) 归一到 0.5 → (0.3, 0.2)
    assert result["members"]["a"]["target_weight"] == pytest.approx(0.3)
    assert result["members"]["b"]["target_weight"] == pytest.approx(0.2)
    # 暂停成员清零，其 0.3 预算转现金
    assert result["members"]["c"]["target_weight"] == 0.0
    assert result["members"]["c"]["suspended"] is True
    assert result["cash_weight"] == pytest.approx(0.5)
    total = sum(item["target_weight"] for item in result["members"].values())
    assert total + result["cash_weight"] == pytest.approx(1.0)
    assert result["renormalization"]["rule"] == "suspended_budget_to_cash_v1"
    assert result["renormalization"]["suspended_share"] == pytest.approx(0.3)
    assert result["suspended_members"] == {"c": pytest.approx(0.3)}


@pytest.mark.no_database
def test_renormalize_without_suspended_members_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one suspended"):
        renormalize_budgets_for_suspended(
            _analysis({"a": 1.0}), previous_weights={"a": 1.0}, suspended={}
        )


@pytest.mark.no_database
def test_renormalize_zero_active_mass_goes_all_cash() -> None:
    result = renormalize_budgets_for_suspended(
        _analysis({}),
        previous_weights={"a": 0.6, "b": 0.4},
        suspended={"a": 0.6, "b": 0.4},
    )
    assert all(item["target_weight"] == 0.0 for item in result["members"].values())
    assert result["cash_weight"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DB integration through AllocationStore.refresh
# ---------------------------------------------------------------------------


def _three_versions(database_url: str, tmp_path: Path) -> list[str]:
    dates = pd.bdate_range("2024-01-02", periods=160)
    series = [
        pd.Series(np.sin(np.arange(len(dates)) / 5) * 0.01, index=dates),
        pd.Series(np.cos(np.arange(len(dates)) / 7) * 0.012, index=dates),
        pd.Series(np.sin(np.arange(len(dates)) / 9 + 1.0) * 0.008, index=dates),
    ]
    # OOS vintage seal: each candidate needs a distinct final test window.
    windows = [
        PERIODS,
        {**PERIODS, "test_start": date(2024, 2, 8)},
        {**PERIODS, "test_start": date(2024, 2, 22)},
    ]
    return [
        _approve_version(
            database_url, tmp_path, suffix=f"m{index}", returns=returns, periods=window
        )
        for index, (returns, window) in enumerate(zip(series, windows, strict=True))
    ]


def _pause_member(
    database_url: str,
    allocation_id: str,
    version_id: str,
    *,
    dataset: str = "allocation-data",
) -> None:
    recommendations = RecommendationStore(database_url)
    portfolio = recommendations.create(
        name=f"suspended member {version_id[:8]}",
        strategy_version_id=version_id,
        dataset=dataset,
        hypothetical_initial_value=500_000,
        actor="operator-a",
    )
    recommendations.set_status(portfolio["id"], "paused", actor="operator-b")
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(strategy_allocation_members)
            .where(
                strategy_allocation_members.c.allocation_id == allocation_id,
                strategy_allocation_members.c.strategy_version_id == version_id,
            )
            .values(recommendation_portfolio_id=portfolio["id"])
        )


def _enable_recommendations(database_url: str, version_ids: list[str]) -> None:
    """Fixture shortcut: real approval sets promotion_stage="paper"; these
    tests exercise suspension, not the forward gate (production must pass
    PromotionStore.promote)."""

    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id.in_(version_ids))
            .values(promotion_stage="recommendation_enabled")
        )


def _activate(database_url: str, allocation_id: str) -> None:
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(strategy_allocations)
            .where(strategy_allocations.c.id == allocation_id)
            .values(status="active")
        )


def _expire_artifact(database_url: str, allocation_id: str) -> None:
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(strategy_allocation_artifacts)
            .where(strategy_allocation_artifacts.c.allocation_id == allocation_id)
            .values(valid_until=date.today() - timedelta(days=1))
        )


def _weights(store: AllocationStore, allocation_id: str) -> dict[str, float]:
    return {
        member["strategy_version_id"]: member["target_weight"]
        for member in store.get(allocation_id)["members"]
    }


def test_suspension_waits_for_decision_day_then_renormalizes(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    _qlib_doubles(monkeypatch)
    version_ids = _three_versions(database_url, tmp_path)
    _enable_recommendations(database_url, version_ids)
    store = AllocationStore(database_url)
    allocation = _create_allocation(store, version_ids, "suspension renormalization")
    _activate(database_url, allocation["id"])
    before = _weights(store, allocation["id"])
    cash_before = float(store.get(allocation["id"])["cash_reserve"])
    paused = version_ids[0]
    _pause_member(database_url, allocation["id"], paused)

    # 制品仍有效：暂停不重估，预算原样（冻结决策日语义）
    store.refresh(allocation["id"])
    assert _weights(store, allocation["id"]) == pytest.approx(before)
    assert float(store.get(allocation["id"])["cash_reserve"]) == pytest.approx(cash_before)

    # 决策日到达（制品过期）：暂停成员清零、其预算转现金、活跃成员按政策归一
    _expire_artifact(database_url, allocation["id"])
    store.refresh(allocation["id"])
    after = _weights(store, allocation["id"])
    active_mass = sum(weight for key, weight in before.items() if key != paused)
    assert after[paused] == 0.0
    assert sum(weight for key, weight in after.items() if key != paused) == pytest.approx(
        active_mass
    )
    updated = store.get(allocation["id"])
    total_capital = float(updated["total_capital"])
    assert float(updated["cash_reserve"]) == pytest.approx(
        total_capital * (1.0 - active_mass)
    )
    assert sum(after.values()) + float(updated["cash_reserve"]) / total_capital == (
        pytest.approx(1.0)
    )
    latest = updated["artifacts"][0]
    analysis = latest["analysis"]
    assert analysis["suspended_members"] == {paused: pytest.approx(before[paused])}
    assert analysis["renormalization"]["rule"] == "suspended_budget_to_cash_v1"
    assert latest["member_weights"][paused] == 0.0


def test_single_active_member_keeps_frozen_budget(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    _qlib_doubles(monkeypatch)
    version_ids = _two_versions(database_url, tmp_path)
    _enable_recommendations(database_url, version_ids)
    store = AllocationStore(database_url)
    allocation = _create_allocation(store, version_ids, "single active fallback")
    _activate(database_url, allocation["id"])
    before = _weights(store, allocation["id"])
    paused = version_ids[0]
    _pause_member(database_url, allocation["id"], paused)
    _expire_artifact(database_url, allocation["id"])

    store.refresh(allocation["id"])

    after = _weights(store, allocation["id"])
    # 单活跃成员无法重估（<2 收益序列）：保留冻结预算作为简单基线，暂停转现金
    assert after[paused] == 0.0
    assert after[version_ids[1]] == pytest.approx(before[version_ids[1]])
    updated = store.get(allocation["id"])
    total_capital = float(updated["total_capital"])
    assert float(updated["cash_reserve"]) == pytest.approx(
        total_capital * (1.0 - before[version_ids[1]])
    )
    latest = updated["artifacts"][0]
    assert latest["analysis"]["fallback_reason"] == "single_active_member_keeps_frozen_budget"
