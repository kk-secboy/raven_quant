from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from governance_fixtures import PERIODS, create_strategy_version
from qlib_test_doubles import (
    QlibPortfolioOptimizer,
    QlibRiskEstimator,
    qlib_runtime_identity,
)
from sqlalchemy import update
from test_strategy_allocation_recommendations import _approve_version

import quant_platform.allocation_store as allocation_store
import quant_platform.risk_math as risk_math
import quant_platform.strategy_allocation as strategy_allocation
from quant_data.database import (
    open_database,
    recommendation_portfolios,
    strategy_allocation_artifacts,
    strategy_allocations,
    strategy_versions,
)
from quant_platform.allocation_store import AllocationStore, next_decision_date
from quant_platform.recommendation_store import RecommendationStore


def _qlib_doubles(monkeypatch) -> None:
    monkeypatch.setattr(risk_math, "_load_qlib_risk_model", lambda: QlibRiskEstimator)
    monkeypatch.setattr(risk_math, "upstream_runtime_identity", qlib_runtime_identity)
    monkeypatch.setattr(
        strategy_allocation,
        "_load_qlib_portfolio_optimizer",
        lambda: QlibPortfolioOptimizer,
    )
    monkeypatch.setattr(
        strategy_allocation, "upstream_runtime_identity", qlib_runtime_identity
    )


def _two_versions(database_url: str, tmp_path: Path) -> list[str]:
    dates = pd.bdate_range("2024-01-02", periods=160)
    first = pd.Series(np.sin(np.arange(len(dates)) / 5) * 0.01, index=dates)
    second = pd.Series(np.cos(np.arange(len(dates)) / 7) * 0.012, index=dates)
    # The OOS vintage seal rejects a second candidate consuming the same final
    # test window; shift the second version's window (same pattern as
    # tests/test_strategy_allocation_recommendations.py).
    second_window = {**PERIODS, "test_start": date(2024, 2, 8)}
    return [
        _approve_version(database_url, tmp_path, suffix="one", returns=first),
        _approve_version(
            database_url, tmp_path, suffix="two", returns=second, periods=second_window
        ),
    ]


def _create_allocation(
    store: AllocationStore, version_ids: list[str], name: str, **overrides
) -> dict:
    options = {
        "name": name,
        "strategy_version_ids": version_ids,
        "dataset": "allocation-data",
        "total_capital": 1_000_000,
        "allocation_method": "inverse_volatility",
        "lookback_days": 120,
        "target_volatility": 0.20,
        "max_pairwise_correlation": 0.80,
        "max_strategy_weight": 0.70,
        "max_member_drawdown": 0.08,
        "max_drawdown_reduce": 0.10,
        "max_drawdown_liquidate": 0.15,
        "fixed_weights": None,
        "actor": "allocation-author",
    }
    options.update(overrides)
    return store.create(**options)


def test_next_decision_date_calendar() -> None:
    assert next_decision_date(date(2026, 7, 21), "weekly") == date(2026, 7, 28)
    assert next_decision_date(date(2026, 1, 31), "monthly") == date(2026, 2, 28)
    assert next_decision_date(date(2026, 12, 15), "monthly") == date(2027, 1, 15)
    with pytest.raises(ValueError, match="unknown decision frequency"):
        next_decision_date(date(2026, 7, 21), "daily")


def test_single_active_allocation_guard(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    _qlib_doubles(monkeypatch)
    version_ids = _two_versions(database_url, tmp_path)
    store = AllocationStore(database_url)
    first = _create_allocation(store, version_ids, "guard allocation one")
    second = _create_allocation(store, version_ids, "guard allocation two")
    engine = open_database(database_url)

    # Seed one active allocation directly; every activation path must now fail
    # closed while another allocation is active.
    with engine.begin() as connection:
        connection.execute(
            update(strategy_allocations)
            .where(strategy_allocations.c.id == first["id"])
            .values(status="active")
        )
    with pytest.raises(ValueError, match="already active"):
        store.approve(
            second["id"],
            actor="allocation-approver",
            reason="Second allocation approval must fail while another is active.",
        )
    with engine.begin() as connection:
        connection.execute(
            update(strategy_allocations)
            .where(strategy_allocations.c.id == second["id"])
            .values(status="paused")
        )
    with pytest.raises(ValueError, match="already active"):
        store.set_status(second["id"], "active", actor="allocation-risk-owner")

    store.set_status(first["id"], "paused", actor="allocation-risk-owner")
    store.set_status(second["id"], "active", actor="allocation-risk-owner")
    with pytest.raises(ValueError, match="already active"):
        store.set_status(first["id"], "active", actor="allocation-risk-owner")

    # The database layer carries the same contract as a partial unique index.
    with engine.begin() as connection:
        with pytest.raises(Exception, match="uq_strategy_allocations_single_active"):
            connection.execute(
                update(strategy_allocations)
                .where(strategy_allocations.c.id == first["id"])
                .values(status="active")
            )


def test_single_active_recommendation_sender_guard(
    database_url: str, tmp_path: Path
) -> None:
    version_id = create_strategy_version(database_url, tmp_path)
    store = RecommendationStore(database_url)
    with store.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved")
        )
    first = store.create(
        name="sender one",
        strategy_version_id=version_id,
        dataset="snapshot",
        hypothetical_initial_value=5_000_000,
        actor="operator-a",
    )
    with pytest.raises(ValueError, match="already the active sender"):
        store.create(
            name="sender two",
            strategy_version_id=version_id,
            dataset="snapshot",
            hypothetical_initial_value=5_000_000,
            actor="operator-a",
        )

    store.set_status(first["id"], "paused", actor="operator-b")
    second = store.create(
        name="sender two",
        strategy_version_id=version_id,
        dataset="snapshot",
        hypothetical_initial_value=5_000_000,
        actor="operator-a",
    )
    assert second["status"] == "active"
    with pytest.raises(ValueError, match="already the active sender"):
        store.set_status(first["id"], "active", actor="operator-b")

    # Allocation-owned member portfolios are outside the sender uniqueness.
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(recommendation_portfolios)
            .where(recommendation_portfolios.c.id == second["id"])
            .values(recommendation_scope="allocation_member")
        )
    third = store.create(
        name="sender three",
        strategy_version_id=version_id,
        dataset="snapshot",
        hypothetical_initial_value=5_000_000,
        actor="operator-a",
    )
    assert third["status"] == "active"


def test_allocation_artifact_freezes_decision_day(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    _qlib_doubles(monkeypatch)
    version_ids = _two_versions(database_url, tmp_path)
    store = AllocationStore(database_url)
    allocation = _create_allocation(
        store, version_ids, "artifact allocation", decision_frequency="weekly"
    )
    assert allocation["decision_frequency"] == "weekly"
    today = datetime.now(UTC).date()
    assert len(allocation["artifacts"]) == 1
    artifact = allocation["artifacts"][0]
    assert artifact["decision_date"] == today
    assert artifact["valid_until"] == today + timedelta(days=7)
    assert artifact["inputs_as_of"] <= artifact["decision_date"]
    member_weights = artifact["member_weights"]
    assert set(member_weights) == set(version_ids)
    for member in allocation["members"]:
        assert member_weights[member["strategy_version_id"]] == pytest.approx(
            member["target_weight"]
        )
    assert len(artifact["artifact_hash"]) == 64

    calls: list[int] = []
    real_analyze = allocation_store.analyze_strategy_allocation

    def counting_analyze(*args, **kwargs):
        calls.append(1)
        return real_analyze(*args, **kwargs)

    monkeypatch.setattr(
        allocation_store, "analyze_strategy_allocation", counting_analyze
    )
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(strategy_allocations)
            .where(strategy_allocations.c.id == allocation["id"])
            .values(status="active")
        )

    # A still-valid artifact is reused: refresh never re-estimates budgets.
    result = store.refresh(allocation["id"])
    assert result["refresh_status"] == "waiting_for_simulation_portfolio"
    assert calls == []

    # Expire the artifact: the next refresh re-solves exactly once and applies
    # the new budgets exactly once.
    with engine.begin() as connection:
        connection.execute(
            update(strategy_allocation_artifacts)
            .where(
                strategy_allocation_artifacts.c.allocation_id == allocation["id"]
            )
            .values(valid_until=today - timedelta(days=1))
        )
    store.refresh(allocation["id"])
    assert len(calls) == 1
    refreshed = store.get(allocation["id"])
    assert len(refreshed["artifacts"]) == 2
    latest = refreshed["artifacts"][0]
    assert latest["decision_date"] == today
    assert latest["valid_until"] == today + timedelta(days=7)
    for member in refreshed["members"]:
        assert latest["member_weights"][member["strategy_version_id"]] == pytest.approx(
            member["target_weight"]
        )
    resolved_events = [
        event
        for event in refreshed["events"]
        if event["event_type"] == "allocation.artifact_resolved"
    ]
    assert len(resolved_events) == 1

    # The artifact is valid again: subsequent refreshes reuse it.
    store.refresh(allocation["id"])
    assert len(calls) == 1
    assert len(store.get(allocation["id"])["artifacts"]) == 2


def test_allocation_artifact_reschedule_failure_keeps_budgets(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    _qlib_doubles(monkeypatch)
    version_ids = _two_versions(database_url, tmp_path)
    store = AllocationStore(database_url)
    allocation = _create_allocation(store, version_ids, "failing artifact allocation")
    today = datetime.now(UTC).date()
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(strategy_allocations)
            .where(strategy_allocations.c.id == allocation["id"])
            .values(status="active")
        )
        connection.execute(
            update(strategy_allocation_artifacts)
            .where(
                strategy_allocation_artifacts.c.allocation_id == allocation["id"]
            )
            .values(valid_until=today - timedelta(days=1))
        )
    original_weights = {
        member["strategy_version_id"]: member["target_weight"]
        for member in allocation["members"]
    }

    def broken_analyze(*args, **kwargs):
        raise ValueError("synthetic solver outage")

    monkeypatch.setattr(
        allocation_store, "analyze_strategy_allocation", broken_analyze
    )
    result = store.refresh(allocation["id"])
    assert result["refresh_status"] == "waiting_for_simulation_portfolio"
    refreshed = store.get(allocation["id"])
    # The previous budgets stay in force and the failure is recorded.
    assert {
        member["strategy_version_id"]: member["target_weight"]
        for member in refreshed["members"]
    } == original_weights
    assert len(refreshed["artifacts"]) == 1
    assert any(
        event["event_type"] == "allocation.artifact_reschedule_failed"
        for event in refreshed["events"]
    )
