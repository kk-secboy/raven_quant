from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import delete, insert, select, update

from quant_data.database import (
    backtest_runs,
    open_database,
    paper_portfolios,
    paper_positions,
    portfolio_nav,
    schedules,
    strategies,
    strategy_allocation_events,
    strategy_versions,
)
from quant_platform.allocation_store import AllocationStore
from quant_platform.schedule_store import ScheduleStore, synchronize_portfolio_schedules
from quant_platform.strategy_allocation import analyze_strategy_allocation


def _strategy_evidence(
    database_url: str,
    root: Path,
    name: str,
    returns: np.ndarray,
) -> str:
    engine = open_database(database_url)
    strategy_id = uuid.uuid4().hex
    version_id = uuid.uuid4().hex
    backtest_id = uuid.uuid4().hex
    now = datetime.now(UTC)
    with engine.begin() as connection:
        connection.execute(
            insert(strategies).values(
                id=strategy_id,
                name=name,
                description=f"Governed {name} strategy for allocation testing.",
                status="approved",
                created_by="research-owner",
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            insert(strategy_versions).values(
                id=version_id,
                strategy_id=strategy_id,
                version=1,
                status="approved",
                benchmark="SH000300",
                universe="cn_all",
                config_json={},
                created_by="research-owner",
                approved_by="risk-owner",
                approval_reason="Independent evidence passed all required controls.",
                created_at=now,
                approved_at=now,
            )
        )
        connection.execute(
            insert(backtest_runs).values(
                id=backtest_id,
                strategy_version_id=version_id,
                dataset="snapshot",
                status="succeeded",
                periods_json={"start": "2024-01-01", "end": "2026-07-10"},
                metrics_json={"backtest_engine": "qlib", "qlib_native_backtest": True},
                artifact_path=str(root),
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
    output = root / backtest_id
    output.mkdir(parents=True)
    pd.DataFrame(
        {
            "datetime": pd.bdate_range("2025-01-01", periods=len(returns)),
            "net_return": returns,
        }
    ).to_parquet(output / "daily_returns.parquet", index=False)
    return version_id


def test_strategy_allocation_rejects_highly_correlated_members() -> None:
    dates = pd.bdate_range("2025-01-01", periods=100)
    values = np.sin(np.arange(100) / 5) * 0.01
    frame = pd.DataFrame({"a": values, "b": values * 1.01}, index=dates)

    with pytest.raises(ValueError, match="correlation"):
        analyze_strategy_allocation(
            frame,
            method="risk_parity",
            lookback_days=60,
            target_volatility=0.15,
            max_pairwise_correlation=0.70,
            max_strategy_weight=0.70,
        )


def test_governed_allocation_provisions_children_and_enforces_group_drawdown(
    tmp_path: Path,
    database_url: str,
) -> None:
    observations = np.arange(180)
    first = np.sin(observations / 4.0) * 0.012 + 0.0003
    second = np.cos(observations / 7.0) * 0.009 + 0.0002
    version_one = _strategy_evidence(database_url, tmp_path, "Trend core", first)
    version_two = _strategy_evidence(database_url, tmp_path, "Neutral satellite", second)
    store = AllocationStore(database_url)

    allocation = store.create(
        name="Low correlation core satellite",
        strategy_version_ids=[version_one, version_two],
        dataset="snapshot",
        total_capital=5_000_000,
        allocation_method="risk_parity",
        lookback_days=120,
        target_volatility=0.15,
        max_pairwise_correlation=0.70,
        max_strategy_weight=0.70,
        max_member_drawdown=0.08,
        max_drawdown_reduce=0.10,
        max_drawdown_liquidate=0.15,
        fixed_weights=None,
        actor="allocation-owner",
    )

    assert allocation["status"] == "draft"
    assert len(allocation["members"]) == 2
    assert allocation["analysis"]["highest_pairwise_correlation"] < 0.70
    assert sum(item["target_weight"] for item in allocation["members"]) <= 1.0 + 1e-9
    with pytest.raises(ValueError, match="second operator"):
        store.approve(
            allocation["id"],
            actor="allocation-owner",
            reason="The creator must not approve this allocation.",
        )

    approved = store.approve(
        allocation["id"],
        actor="risk-approver",
        reason="Correlation, volatility, capital, and risk budgets passed review.",
    )
    assert approved["status"] == "active"
    assert all(item["portfolio_id"] for item in approved["members"])
    engine = open_database(database_url)
    with engine.connect() as connection:
        children = connection.execute(
            select(paper_portfolios).where(
                paper_portfolios.c.id.in_([item["portfolio_id"] for item in approved["members"]])
            )
        ).all()
    allocated = sum(float(item.initial_cash) for item in children)
    assert allocated + approved["cash_reserve"] == pytest.approx(5_000_000)

    schedule_store = ScheduleStore(database_url)
    automation = schedule_store.create_allocation_group(
        approved["id"],
        timezone="Asia/Shanghai",
        run_time=time(15, 30),
        trading_days_only=True,
        slippage=0.0005,
        misfire_grace_seconds=1800,
        actor="portfolio-operator",
        now=datetime(2026, 7, 13, 6, 0, tzinfo=UTC),
    )
    assert automation["effective_status"] == "active"
    assert len(automation["members"]) == 2
    schedule_ids = {item["schedule_id"] for item in automation["members"]}
    with pytest.raises(ValueError, match="controlled through their group"):
        schedule_store.set_status(next(iter(schedule_ids)), "paused")
    with pytest.raises(ValueError, match="already has a non-retired paper schedule"):
        schedule_store.create(
            name="conflicting child schedule",
            kind="paper_rebalance",
            timezone="Asia/Shanghai",
            run_time=time(15, 45),
            trading_days_only=True,
            payload={"portfolio_id": str(children[0].id), "slippage": 0.0005},
            misfire_grace_seconds=1800,
            actor="portfolio-operator",
            now=datetime(2026, 7, 13, 6, 2, tzinfo=UTC),
        )
    updated_automation = schedule_store.create_allocation_group(
        approved["id"],
        timezone="Asia/Shanghai",
        run_time=time(15, 40),
        trading_days_only=True,
        slippage=0.001,
        misfire_grace_seconds=1800,
        actor="portfolio-operator",
        now=datetime(2026, 7, 13, 6, 5, tzinfo=UTC),
    )
    assert {item["schedule_id"] for item in updated_automation["members"]} == schedule_ids
    assert schedule_store.set_allocation_group_status(approved["id"], "paused")[
        "effective_status"
    ] == "paused"
    assert schedule_store.set_allocation_group_status(approved["id"], "active")[
        "effective_status"
    ] == "active"

    trade_date = date(2026, 7, 13)
    now = datetime.now(UTC)
    with engine.begin() as connection:
        for child in children:
            stressed_nav = float(child.initial_cash) * 0.70
            connection.execute(
                insert(portfolio_nav).values(
                    portfolio_id=child.id,
                    trade_date=trade_date,
                    cash=0,
                    market_value=stressed_nav,
                    nav=stressed_nav,
                    daily_return=-0.30,
                    benchmark_return=-0.01,
                    drawdown=-0.30,
                    exposure=1.0,
                    turnover=0.0,
                    fees=0,
                    created_at=now,
                )
            )
            connection.execute(
                insert(paper_positions).values(
                    portfolio_id=child.id,
                    instrument=f"SH{child.id[:6]}",
                    industry="test",
                    take_profit_stage=0,
                    quantity=100,
                    avg_cost=10,
                    market_price=7,
                    market_value=700,
                    weight=0.01,
                    realized_pnl=0,
                    unrealized_pnl=-300,
                    updated_at=now,
                )
            )

    refreshed = store.refresh(approved["id"])

    assert refreshed["refresh_status"] == "recorded"
    assert refreshed["status"] == "liquidation_pending"
    assert refreshed["nav_history"][0]["drawdown"] < -0.15
    with engine.connect() as connection:
        statuses = set(
            connection.execute(
                select(paper_portfolios.c.status).where(
                    paper_portfolios.c.id.in_([item.id for item in children])
                )
            ).scalars()
        )
    assert statuses == {"liquidation_pending"}
    risk_automation = schedule_store.get_allocation_group(approved["id"])
    assert {item["status"] for item in risk_automation["members"]} == {"active"}
    with pytest.raises(ValueError, match="risk-execution"):
        schedule_store.set_allocation_group_status(approved["id"], "paused")

    critical_events = [item for item in refreshed["events"] if item["severity"] == "critical"]
    liquidation_event = next(
        item for item in critical_events if item["rule"] == "max_drawdown_liquidate"
    )
    acknowledged = store.acknowledge_event(
        approved["id"], liquidation_event["id"], actor="risk-owner"
    )
    assert acknowledged["status"] == "acknowledged"
    with pytest.raises(ValueError, match="must finish"):
        store.resolve_event(
            approved["id"],
            liquidation_event["id"],
            actor="risk-owner",
            reason="The liquidation execution has been reviewed and reconciled.",
        )

    with engine.begin() as connection:
        connection.execute(
            delete(paper_positions).where(
                paper_positions.c.portfolio_id.in_([item.id for item in children])
            )
        )
        connection.execute(
            update(paper_portfolios)
            .where(paper_portfolios.c.id.in_([item.id for item in children]))
            .values(status="paused", updated_at=now)
        )
        for child in children:
            synchronize_portfolio_schedules(
                connection,
                str(child.id),
                "paused",
                now=now,
            )

    for event in critical_events:
        store.resolve_event(
            approved["id"],
            event["id"],
            actor="risk-owner",
            reason="Required risk execution completed and child ledgers were reconciled.",
        )
    assert store.get(approved["id"])["status"] == "paused"
    resumed = store.set_status(approved["id"], "active", actor="portfolio-operator")
    assert resumed["status"] == "active"
    with engine.connect() as connection:
        assert set(
            connection.execute(
                select(strategy_allocation_events.c.status).where(
                    strategy_allocation_events.c.id.in_([item["id"] for item in critical_events])
                )
            ).scalars()
        ) == {"resolved"}
        assert set(
            connection.execute(
                select(paper_portfolios.c.status).where(
                    paper_portfolios.c.id.in_([item.id for item in children])
                )
            ).scalars()
        ) == {"active"}
        assert set(
            connection.execute(
                select(schedules.c.status).where(schedules.c.id.in_(schedule_ids))
            ).scalars()
        ) == {"active"}
