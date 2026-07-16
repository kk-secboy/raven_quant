from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from governance_fixtures import (
    PERIODS,
    create_strategy_version,
    formal_backtest_metrics,
)
from sqlalchemy import func, insert, select

from quant_data.database import (
    open_database,
    paper_portfolios,
    recommendation_portfolios,
    simulation_nav,
    simulation_portfolios,
)
from quant_platform.allocation_store import AllocationStore
from quant_platform.cost_model import COST_SCHEDULE_VERSION
from quant_platform.simulation_engine import SIMULATION_ENGINE_VERSION
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


def test_allocation_uses_recommendation_ledgers_and_propagates_risk(
    database_url: str, tmp_path: Path
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=160)
    first = pd.Series(np.sin(np.arange(len(dates)) / 5) * 0.01, index=dates)
    second = pd.Series(np.cos(np.arange(len(dates)) / 7) * 0.012, index=dates)
    version_ids = [
        _approve_version(database_url, tmp_path, suffix="one", returns=first),
        _approve_version(database_url, tmp_path, suffix="two", returns=second),
    ]
    store = AllocationStore(database_url)
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
    approved = store.approve(
        allocation["id"],
        actor="allocation-approver",
        reason="Approved low-correlation recommendation allocation.",
    )
    portfolio_ids = [item["recommendation_portfolio_id"] for item in approved["members"]]
    assert all(portfolio_ids)
    engine = open_database(database_url)
    now = datetime.now(UTC)
    with engine.begin() as connection:
        for portfolio_id in portfolio_ids:
            initial = connection.execute(
                select(recommendation_portfolios.c.hypothetical_initial_value).where(
                    recommendation_portfolios.c.id == portfolio_id
                )
            ).scalar_one()
            simulation_id = uuid.uuid4().hex
            simulation_value = Decimal(initial) * Decimal("0.80")
            connection.execute(
                insert(simulation_portfolios).values(
                    id=simulation_id,
                    name=f"simulation-{portfolio_id}",
                    recommendation_portfolio_id=portfolio_id,
                    status="active",
                    base_currency="CNY",
                    initial_cash=Decimal(initial),
                    cash=simulation_value,
                    nav=simulation_value,
                    high_water_mark=Decimal(initial),
                    execution_algorithm="twap",
                    execution_dataset="allocation-5m",
                    daily_dataset="allocation-data",
                    daily_dataset_identity_sha256="a" * 64,
                    daily_dataset_lineage_id="b" * 64,
                    daily_field_contract_version="daily-qlib-field-v2-share-volume",
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
            connection.execute(
                insert(simulation_nav).values(
                    portfolio_id=simulation_id,
                    trade_date=date(2026, 7, 10),
                    cash=simulation_value,
                    market_value=Decimal("0"),
                    nav=simulation_value,
                    daily_return=-0.20,
                    drawdown=-0.20,
                    market_date=date(2026, 7, 10),
                    has_stale_prices=False,
                    status="ok",
                    performance_certified=True,
                    created_at=now,
                )
            )
    refreshed = store.refresh(allocation["id"])
    assert refreshed["status"] == "liquidation_pending"
    with engine.connect() as connection:
        overrides = connection.execute(
            select(recommendation_portfolios.c.risk_exposure_override).where(
                recommendation_portfolios.c.id.in_(portfolio_ids)
            )
        ).scalars()
        assert list(overrides) == [0.0, 0.0]
        assert connection.scalar(select(func.count()).select_from(paper_portfolios)) == 0
    assert refreshed["analysis"]["members"].keys() == set(version_ids)
