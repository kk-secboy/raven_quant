from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from governance_fixtures import create_strategy_version
from sqlalchemy import insert

from quant_data.database import (
    open_database,
    simulation_portfolios,
    strategy_allocation_events,
    strategy_allocations,
)
from quant_platform.allocation_store import AllocationStore
from quant_platform.cost_model import COST_SCHEDULE_VERSION
from quant_platform.simulation_engine import SIMULATION_ENGINE_VERSION
from quant_platform.simulation_store import SimulationStore
from quant_platform.strategy_store import StrategyStore


def test_unresolved_member_event_gates_long_and_pair_sources_until_resolution(
    database_url: str,
    tmp_path: Path,
) -> None:
    version_id = create_strategy_version(database_url, tmp_path, dataset="risk-gate-data")
    version = StrategyStore(database_url).get_version(version_id)
    allocation_id = uuid.uuid4().hex
    simulation_id = uuid.uuid4().hex
    now = datetime.now(UTC)
    contract_hash = str(version["execution_contract_hash"])
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            insert(strategy_allocations).values(
                id=allocation_id,
                name=f"risk-gate-{allocation_id}",
                dataset="risk-gate-data",
                status="active",
                allocation_method="fixed",
                lookback_days=60,
                target_volatility=0.15,
                max_pairwise_correlation=0.70,
                max_strategy_weight=0.70,
                max_member_drawdown=0.08,
                max_drawdown_reduce=0.10,
                max_drawdown_liquidate=0.15,
                total_capital=Decimal("500000"),
                cash_reserve=Decimal("0"),
                nav=Decimal("500000"),
                high_water_mark=Decimal("500000"),
                analysis_json={},
                created_by="risk-owner",
                created_at=now,
                updated_at=now,
            )
        )
        event_id = connection.scalar(
            insert(strategy_allocation_events)
            .values(
                allocation_id=allocation_id,
                recommendation_portfolio_id=None,
                severity="critical",
                event_type="member_circuit_breaker",
                rule="max_member_drawdown",
                observed=0.09,
                limit_value=0.08,
                status="open",
                details_json={
                    "strategy_version_id": version_id,
                    "risk_state": "pause_new_risk",
                },
                created_at=now,
            )
            .returning(strategy_allocation_events.c.id)
        )
        connection.execute(
            insert(simulation_portfolios).values(
                id=simulation_id,
                name=f"risk-gate-simulation-{simulation_id}",
                recommendation_portfolio_id=None,
                source_type="strategy_version",
                source_id=version_id,
                status="paused",
                base_currency="CNY",
                initial_cash=Decimal("500000"),
                cash=Decimal("500000"),
                nav=Decimal("500000"),
                high_water_mark=Decimal("500000"),
                execution_algorithm="twap",
                execution_adapter="pair",
                execution_frequency="5min",
                execution_contract_hash=contract_hash,
                execution_dataset="risk-gate-5m",
                daily_dataset="risk-gate-data",
                daily_dataset_identity_sha256="a" * 64,
                daily_dataset_lineage_id="b" * 64,
                daily_field_contract_version="daily-qlib-field-v2-share-volume",
                execution_dataset_identity_sha256="c" * 64,
                execution_dataset_lineage_id="d" * 64,
                execution_field_contract_version="minute-qlib-execution-v4-source-units",
                execution_engine_version=SIMULATION_ENGINE_VERSION,
                cost_schedule_version=COST_SCHEDULE_VERSION,
                execution_policy_json={"execution_algorithm": "twap"},
                created_by="risk-owner",
                created_at=now,
                updated_at=now,
            )
        )

    allocations = AllocationStore(database_url)
    simulations = SimulationStore(database_url)
    expected = allocations.member_risk_state(version_id)
    assert expected["state"] == "pause_new_risk"
    assert expected["allow_new_risk"] is False
    unified = simulations.source_risk_state(simulation_id)
    assert unified["state"] == expected["state"]
    assert unified["allow_new_risk"] is False
    assert unified["risk_exposure_override"] == 1.0
    assert unified["member_event_ids"] == expected["event_ids"]

    allocations.resolve_event(
        allocation_id,
        int(event_id),
        actor="risk-reviewer",
        reason="Drawdown evidence reviewed; controlled reopening is approved.",
    )
    assert allocations.member_risk_state(version_id)["state"] == "active"
    assert simulations.source_risk_state(simulation_id)["allow_new_risk"] is True
