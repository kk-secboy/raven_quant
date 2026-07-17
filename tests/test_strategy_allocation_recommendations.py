from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from governance_fixtures import (
    PERIODS,
    create_strategy_version,
    formal_backtest_metrics,
)
from qlib_test_doubles import (
    QlibPortfolioOptimizer,
    QlibRiskEstimator,
    qlib_runtime_identity,
)
from sqlalchemy import func, insert, select

import quant_platform.risk_math as risk_math
import quant_platform.strategy_allocation as strategy_allocation
from quant_data.database import (
    open_database,
    paper_portfolios,
    recommendation_portfolios,
    simulation_nav,
    simulation_portfolios,
)
from quant_data.execution_contract import (
    DAILY_QLIB_FIELD_CONTRACT_VERSION,
    MINUTE_EXECUTION_CONTRACT_VERSION,
    MINUTE_SOURCE_UNIT_CONTRACTS,
)
from quant_platform.allocation_store import AllocationStore
from quant_platform.cost_model import COST_SCHEDULE_VERSION
from quant_platform.simulation_engine import SIMULATION_ENGINE_VERSION
from quant_platform.simulation_store import SimulationStore
from quant_platform.strategy_store import StrategyStore


def _approve_version(
    database_url: str, tmp_path: Path, *, suffix: str, returns: pd.Series
) -> str:
    version_id = create_strategy_version(database_url, tmp_path, dataset="allocation-data")
    strategies = StrategyStore(database_url)
    version = strategies.get_version(version_id)
    artifact = tmp_path / f"backtest-{suffix}"
    artifact.mkdir()
    periods = {
        "start": PERIODS["test_start"].isoformat(),
        "end": PERIODS["test_end"].isoformat(),
    }
    backtest = strategies.create_backtest(
        version_id=version_id,
        dataset="allocation-data",
        periods=periods,
        artifact_path=artifact,
    )
    factor = version["factors"][0]
    manifest = artifact / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy_version_id": version_id,
                "dataset": "allocation-data",
                "benchmark": version["benchmark"],
                "universe": version["universe"],
                "periods": periods,
                "config": version["config"],
                "factors": [
                    {
                        "candidate_id": factor["factor_candidate_id"],
                        "values_path": factor["values_path"],
                        "code_sha256": factor["code_sha256"],
                        "weight": factor["weight"],
                        "direction": factor["direction"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"datetime": returns.index, "return": returns.values, "cost": 0.0}
    ).to_parquet(artifact / "daily_returns.parquet", index=False)
    metrics = formal_backtest_metrics(version, manifest)
    strategies.validate_backtest_artifacts(backtest["id"], metrics)
    strategies.mark_backtest(backtest["id"], "succeeded", metrics=metrics)
    strategies.approve(
        version_id,
        actor="allocation-risk-owner",
        reason="Approved independently for recommendation allocation testing.",
    )
    return version_id


def _daily_dataset() -> dict:
    return {
        "name": "allocation-data",
        "provenance": {
            "frequency": "day",
            "dataset_identity_sha256": "a" * 64,
            "dataset_lineage_id": "b" * 64,
            "source_lineage_id": "9" * 64,
            "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
            "source_volume_unit": "hand",
            "qlib_volume_unit": "share",
            "source_amount_unit": "thousand_cny",
            "qlib_amount_unit": "cny",
            "source_hand_size": 100,
            "index_volume_policy": "excluded_non_tradable_benchmark",
            "lineage_verified": True,
        },
    }


def _minute_dataset() -> dict:
    return {
        "name": "allocation-5m",
        "provenance": {
            "frequency": "5min",
            "dataset_identity_sha256": "c" * 64,
            "dataset_lineage_id": "d" * 64,
            "source_lineage_id": "9" * 64,
            "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
            "fields": ["vwap", "volume", "paused", "up_limit", "down_limit"],
            "source_datasets": ["ashare_5m"],
            "source_unit_contracts": {
                "ashare_5m": MINUTE_SOURCE_UNIT_CONTRACTS["ashare_5m"]
            },
            "lineage_verified": True,
        },
    }


def test_allocation_uses_recommendation_ledgers_and_propagates_risk(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(risk_math, "_load_qlib_risk_model", lambda: QlibRiskEstimator)
    monkeypatch.setattr(
        risk_math, "upstream_runtime_identity", qlib_runtime_identity
    )
    monkeypatch.setattr(
        strategy_allocation,
        "_load_qlib_portfolio_optimizer",
        lambda: QlibPortfolioOptimizer,
    )
    monkeypatch.setattr(
        strategy_allocation, "upstream_runtime_identity", qlib_runtime_identity
    )
    dates = pd.bdate_range("2024-01-02", periods=160)
    first = pd.Series(np.sin(np.arange(len(dates)) / 5) * 0.01, index=dates)
    second = pd.Series(np.cos(np.arange(len(dates)) / 7) * 0.012, index=dates)
    version_ids = [
        _approve_version(database_url, tmp_path, suffix="one", returns=first),
        _approve_version(database_url, tmp_path, suffix="two", returns=second),
    ]
    store = AllocationStore(database_url)
    with pytest.raises(ValueError, match="core/satellite member cap"):
        store.create(
            name="invalid satellite allocation",
            strategy_version_ids=version_ids,
            dataset="allocation-data",
            total_capital=1_000_000,
            allocation_method="risk_parity",
            lookback_days=120,
            target_volatility=0.20,
            max_pairwise_correlation=0.80,
            max_strategy_weight=0.70,
            max_member_drawdown=0.08,
            max_drawdown_reduce=0.10,
            max_drawdown_liquidate=0.15,
            fixed_weights=None,
            actor="allocation-author",
            member_specs=[
                {
                    "strategy_version_id": version_ids[0],
                    "role": "core",
                    "risk_budget": 0.85,
                    "member_cap": 0.70,
                },
                {
                    "strategy_version_id": version_ids[1],
                    "role": "satellite",
                    "risk_budget": 0.15,
                    "member_cap": 0.15,
                },
            ],
        )
    allocation = store.create(
        name="recommendation risk parity",
        strategy_version_ids=version_ids,
        dataset="allocation-data",
        total_capital=1_000_000,
        allocation_method="risk_parity",
        lookback_days=120,
        target_volatility=0.20,
        max_pairwise_correlation=0.80,
        max_strategy_weight=0.70,
        max_member_drawdown=0.08,
        max_drawdown_reduce=0.10,
        max_drawdown_liquidate=0.15,
        fixed_weights=None,
        actor="allocation-author",
    )
    engine = open_database(database_url)
    now = datetime.now(UTC)
    simulation_ids: list[str] = []
    with engine.begin() as connection:
        for index, version_id in enumerate(version_ids):
            simulation_id = uuid.uuid4().hex
            simulation_ids.append(simulation_id)
            simulation_initial_cash = Decimal("500000") if index == 0 else Decimal("2000000")
            simulation_value = simulation_initial_cash * Decimal("0.80")
            connection.execute(
                insert(simulation_portfolios).values(
                    id=simulation_id,
                    name=f"strategy-simulation-{index}",
                    recommendation_portfolio_id=None,
                    source_type="strategy_version",
                    source_id=version_id,
                    status="active",
                    base_currency="CNY",
                    initial_cash=simulation_initial_cash,
                    cash=simulation_value,
                    nav=simulation_value,
                    high_water_mark=simulation_initial_cash,
                    execution_algorithm="twap",
                    execution_adapter="long_only",
                    execution_frequency="5min",
                    execution_contract_hash="f" * 64,
                    execution_dataset="allocation-5m",
                    daily_dataset="allocation-data",
                    daily_dataset_identity_sha256="a" * 64,
                    daily_dataset_lineage_id="b" * 64,
                    daily_field_contract_version="daily-qlib-field-v3-cny-amount",
                    execution_dataset_identity_sha256="c" * 64,
                    execution_dataset_lineage_id="d" * 64,
                    execution_field_contract_version=(
                        "minute-qlib-execution-v4-source-units"
                    ),
                    execution_engine_version=SIMULATION_ENGINE_VERSION,
                    cost_schedule_version=COST_SCHEDULE_VERSION,
                    execution_policy_json={"execution_algorithm": "twap"},
                    created_by="test",
                    created_at=now,
                    updated_at=now,
                )
            )
            for trade_date in pd.bdate_range("2026-07-06", periods=5):
                connection.execute(
                    insert(simulation_nav).values(
                        portfolio_id=simulation_id,
                        trade_date=trade_date.date(),
                        cash=simulation_value,
                        market_value=Decimal("0"),
                        nav=simulation_value,
                        daily_return=-0.20 if trade_date.date() == date(2026, 7, 10) else 0.0,
                        drawdown=-0.20,
                        market_date=trade_date.date(),
                        has_stale_prices=False,
                        status="ok",
                        performance_certified=True,
                        nav_scope="member_ledger",
                        produced_by=f"simulation-producer-{index}",
                        created_at=now,
                    )
                )
    with pytest.raises(ValueError, match="independently reviewed"):
        store.approve(
            allocation["id"],
            actor="allocation-approver",
            reason="Review evidence is intentionally missing for this attempt.",
        )
    simulations = SimulationStore(database_url)
    for simulation_index, simulation_id in enumerate(simulation_ids):
        for day_index, trade_date in enumerate(pd.bdate_range("2026-07-06", periods=5)):
            simulations.review_nav(
                simulation_id,
                trade_date.date(),
                actor="simulation-risk-reviewer",
                evidence_sha256=f"{simulation_index * 5 + day_index + 1:064x}",
                note="Reviewed cash, positions, fills, data lineage, and certified NAV.",
            )
    assert all(
        simulations.get(simulation_id)["review_readiness"]["ready"]
        for simulation_id in simulation_ids
    )
    approved = store.approve(
        allocation["id"],
        actor="allocation-approver",
        reason="Approved low-correlation recommendation allocation.",
    )
    portfolio_ids = [item["recommendation_portfolio_id"] for item in approved["members"]]
    assert all(portfolio_ids)
    assert {item["role"] for item in approved["members"]} == {"core"}
    assert approved["analysis"]["core_satellite"]["core_weight"] == pytest.approx(1.0)
    assert {
        item["reviewed_days"]
        for item in approved["analysis"]["approval_simulation_evidence"].values()
    } == {5}
    assert all(
        item["review_audit_sha256"]
        for item in approved["analysis"]["approval_simulation_evidence"].values()
    )
    assert all(
        row["reviewed_by"] == "simulation-risk-reviewer"
        for item in approved["analysis"]["approval_simulation_evidence"].values()
        for row in item["nav_rows"]
    )
    assert approved["analysis"]["approval_simulation_nav"]["performance_certified"] is True
    assert approved["analysis"]["approval_simulation_nav"]["reviewed_days"] == 5
    expected_certified_nav = float(approved["cash_reserve"]) + sum(
        1_000_000 * float(member["target_weight"]) * 0.80
        for member in approved["members"]
    )
    assert approved["analysis"]["approval_simulation_nav"]["latest_nav"] == pytest.approx(
        expected_certified_nav
    )
    allocation_simulation = simulations.create(
        name="approved allocation simulation",
        source_type="allocation",
        source_id=allocation["id"],
        daily_dataset=_daily_dataset(),
        execution_dataset=_minute_dataset(),
        initial_cash=1_000_000,
        execution_policy={"execution_algorithm": "twap"},
        cost_schedule_version=COST_SCHEDULE_VERSION,
        actor="test",
    )
    assert allocation_simulation["execution_adapter"] == "long_only"
    allocation_simulation_id = allocation_simulation["id"]
    simulations.set_status(allocation_simulation_id, "active")
    with pytest.raises(ValueError, match="direct simulation target payloads are forbidden"):
        simulations.create_batch_for_targets(
            allocation_simulation_id,
            source_snapshot_id="allocation-target-1",
            signal_date=date(2026, 7, 9),
            trade_date=date(2026, 7, 10),
            target_payload={"target_weights": {"SH600000": 1.0}},
            execution_contract_hash=allocation_simulation["execution_contract_hash"],
            idempotency_key="allocation-target:1",
        )
    refreshed = store.refresh(allocation["id"], actor="allocation-nav-producer")
    assert refreshed["status"] == "liquidation_pending"
    with engine.connect() as connection:
        overrides = connection.execute(
            select(recommendation_portfolios.c.risk_exposure_override).where(
                recommendation_portfolios.c.id.in_(portfolio_ids)
            )
        ).scalars()
        assert list(overrides) == [0.0, 0.0]
        assert connection.scalar(select(func.count()).select_from(paper_portfolios)) == 0
        virtual_nav = connection.execute(
            select(simulation_nav).where(
                simulation_nav.c.portfolio_id == allocation_simulation_id
            )
        ).one()
        assert virtual_nav.performance_certified is True
        assert virtual_nav.nav_scope == "aggregate_view"
        assert virtual_nav.produced_by == "allocation-nav-producer"
        assert float(virtual_nav.nav) == pytest.approx(float(refreshed["nav"]))
    aggregate_review = simulations.review_nav(
        allocation_simulation_id,
        virtual_nav.trade_date,
        actor="allocation-nav-reviewer",
        evidence_sha256="f" * 64,
        note="Reviewed the allocation aggregate view against all member NAV references.",
    )
    assert aggregate_review["review_subject"] == "aggregate_simulation_view"
    assert refreshed["analysis"]["members"].keys() == set(version_ids)
    member_state = store.member_risk_state(version_ids[0])
    assert member_state["state"] == "pause_new_risk"
    assert member_state["allow_new_risk"] is False
    source_simulation = next(
        item
        for item in simulations.list()
        if item["source_type"] == "strategy_version"
        and item["source_id"] == version_ids[0]
    )
    unified_state = simulations.source_risk_state(source_simulation["id"])
    assert unified_state["state"] == "liquidation"
    assert unified_state["member_risk_state"] == member_state["state"]
    assert unified_state["allocation_risk_state"] == "liquidation"
    assert unified_state["allow_new_risk"] is False
    assert unified_state["risk_exposure_override"] == 0.0
    assert member_state["event_ids"][0] in unified_state["member_event_ids"]
    member_event = next(
        item
        for item in refreshed["events"]
        if item["rule"] == "max_member_drawdown"
        and item["details"].get("strategy_version_id") == version_ids[0]
    )
    store.resolve_event(
        allocation["id"],
        member_event["id"],
        actor="allocation-risk-owner",
        reason="Member risk was reviewed and the gate may be reopened.",
    )
    assert store.member_risk_state(version_ids[0])["state"] == "active"
