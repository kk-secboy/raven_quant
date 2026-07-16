from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import insert, select, update

from quant_data.database import (
    open_database,
    strategy_allocation_events,
    strategy_allocation_members,
    strategy_allocations,
)
from quant_platform.allocation_store import AllocationStore
from quant_platform.member_risk_gate import (
    ALLOCATION_LIQUIDATION_RULE,
    ALLOCATION_REDUCTION_RULE,
)
from quant_platform.pair_trading import PairTradingConfig
from quant_platform.strategy_store import StrategyStore


def test_pair_member_reads_allocation_reduction_and_liquidation_state(
    database_url: str,
    tmp_path: Path,
) -> None:
    strategies = StrategyStore(database_url)
    pair = strategies.create_pair(
        name=f"pair-risk-state-{uuid.uuid4().hex}",
        description="Pair member without a recommendation portfolio.",
        leg_y="SH510300",
        leg_x="SZ159919",
        asset_class="etf",
        shorting_mode="margin_borrow",
        config=asdict(PairTradingConfig()),
        actor="pair-researcher",
    )
    version_id = str(pair["versions"][0]["id"])
    backtest = strategies.create_backtest(
        version_id=version_id,
        dataset="pair-risk-daily",
        execution_dataset="pair-risk-5min",
        periods={"start": "2024-01-01", "end": "2026-07-10"},
        artifact_path=tmp_path,
    )
    allocation_id = uuid.uuid4().hex
    now = datetime.now(UTC)
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            insert(strategy_allocations).values(
                id=allocation_id,
                name=f"pair-allocation-risk-{allocation_id}",
                dataset="pair-risk-daily",
                status="risk_reduction_pending",
                allocation_method="fixed",
                lookback_days=120,
                target_volatility=0.15,
                max_pairwise_correlation=0.70,
                max_strategy_weight=0.70,
                max_member_drawdown=0.08,
                max_drawdown_reduce=0.10,
                max_drawdown_liquidate=0.15,
                total_capital=Decimal("1000000"),
                cash_reserve=Decimal("850000"),
                nav=Decimal("900000"),
                high_water_mark=Decimal("1000000"),
                analysis_json={},
                created_by="allocation-owner",
                approved_by="allocation-approver",
                approval_reason="Approved pair risk state fixture.",
                created_at=now,
                approved_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            insert(strategy_allocation_members).values(
                allocation_id=allocation_id,
                strategy_version_id=version_id,
                backtest_id=backtest["id"],
                recommendation_portfolio_id=None,
                target_weight=0.15,
                role="satellite",
                risk_budget=0.15,
                member_cap=0.15,
                annualized_volatility=0.12,
                risk_contribution=0.15,
                created_at=now,
            )
        )
        reduction_event_id = connection.scalar(
            insert(strategy_allocation_events)
            .values(
                allocation_id=allocation_id,
                recommendation_portfolio_id=None,
                severity="critical",
                event_type="allocation_circuit_breaker",
                rule=ALLOCATION_REDUCTION_RULE,
                observed=0.10,
                limit_value=0.10,
                status="open",
                details_json={
                    "risk_state": "risk_reduction",
                    "risk_exposure_override": 0.5,
                },
                created_at=now,
            )
            .returning(strategy_allocation_events.c.id)
        )
        assert connection.scalar(
            select(
                strategy_allocation_members.c.recommendation_portfolio_id
            ).where(
                strategy_allocation_members.c.allocation_id == allocation_id,
                strategy_allocation_members.c.strategy_version_id == version_id,
            )
        ) is None

    allocations = AllocationStore(database_url)
    reduction = allocations.strategy_risk_state(version_id)
    assert reduction["state"] == "risk_reduction"
    assert reduction["risk_exposure_override"] == pytest.approx(0.5)
    assert reduction["allow_new_risk"] is False
    assert reduction["allocation_event_ids"] == [int(reduction_event_id)]

    allocations.acknowledge_event(
        allocation_id,
        int(reduction_event_id),
        actor="allocation-risk-reviewer",
    )
    assert allocations.strategy_risk_state(version_id)[
        "risk_exposure_override"
    ] == pytest.approx(0.5)

    allocations.resolve_event(
        allocation_id,
        int(reduction_event_id),
        actor="allocation-risk-reviewer",
        reason="Reduction evidence is resolved; explicit reactivation is still required.",
    )
    resolved_reduction = allocations.strategy_risk_state(version_id)
    assert resolved_reduction["allocation_event_ids"] == []
    assert resolved_reduction["risk_exposure_override"] == pytest.approx(0.5)
    assert resolved_reduction["recovery"][
        "allocation_ids_requiring_reactivation"
    ] == [allocation_id]

    allocations.set_status(allocation_id, "active", actor="allocation-risk-owner")
    assert allocations.strategy_risk_state(version_id)[
        "risk_exposure_override"
    ] == 1.0

    with engine.begin() as connection:
        connection.execute(
            update(strategy_allocations)
            .where(strategy_allocations.c.id == allocation_id)
            .values(status="liquidation_pending", updated_at=now)
        )
        liquidation_event_id = connection.scalar(
            insert(strategy_allocation_events)
            .values(
                allocation_id=allocation_id,
                recommendation_portfolio_id=None,
                severity="critical",
                event_type="allocation_circuit_breaker",
                rule=ALLOCATION_LIQUIDATION_RULE,
                observed=0.16,
                limit_value=0.15,
                status="open",
                details_json={
                    "risk_state": "liquidation",
                    "risk_exposure_override": 0.0,
                },
                created_at=now,
            )
            .returning(strategy_allocation_events.c.id)
        )

    liquidation = allocations.strategy_risk_state(version_id)
    assert liquidation["state"] == "liquidation"
    assert liquidation["risk_exposure_override"] == 0.0
    assert liquidation["allow_new_risk"] is False
    assert liquidation["allocation_event_ids"] == [int(liquidation_event_id)]

    allocations.resolve_event(
        allocation_id,
        int(liquidation_event_id),
        actor="allocation-risk-reviewer",
        reason="Liquidation evidence is resolved; explicit reactivation is still required.",
    )
    assert allocations.strategy_risk_state(version_id)[
        "risk_exposure_override"
    ] == 0.0
    allocations.set_status(allocation_id, "active", actor="allocation-risk-owner")
    restored = allocations.strategy_risk_state(version_id)
    assert restored["state"] == "active"
    assert restored["allow_new_risk"] is True
    assert restored["risk_exposure_override"] == 1.0
