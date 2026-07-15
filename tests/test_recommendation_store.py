from __future__ import annotations

from copy import deepcopy
from datetime import date

import pytest
from governance_fixtures import DATASET_IDENTITY, create_strategy_version
from sqlalchemy import func, select, update

from quant_data.database import paper_fills, paper_orders, strategy_versions
from quant_platform.cost_model import CostModelConfig
from quant_platform.portfolio_policy import POLICY_VERSION
from quant_platform.qlib_backtest import QLIB_ENGINE_VERSION
from quant_platform.recommendation_store import RecommendationStore


def test_recommendation_snapshot_is_independent_of_paper_orders_and_fills(
    tmp_path, database_url: str
) -> None:
    version_id = create_strategy_version(database_url, tmp_path)
    store = RecommendationStore(database_url)
    with store.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved")
        )
        before = (
            connection.scalar(select(func.count()).select_from(paper_orders)),
            connection.scalar(select(func.count()).select_from(paper_fills)),
        )
    portfolio = store.create(
        name="governed recommendations",
        strategy_version_id=version_id,
        dataset="snapshot",
        hypothetical_initial_value=5_000_000,
        actor="test",
    )
    snapshot, created = store.create_snapshot(
        portfolio_id=portfolio["id"],
        as_of_date=date(2026, 7, 10),
        dataset="snapshot",
        dataset_identity_sha256=DATASET_IDENTITY,
    )
    assert created is True
    result = {
        "status": "ok",
        "portfolio_id": portfolio["id"],
        "strategy_version_id": version_id,
        "dataset": "snapshot",
        "dataset_identity_sha256": DATASET_IDENTITY,
        "as_of_date": "2026-07-10",
        "policy_version": POLICY_VERSION,
        "backtest_engine_version": QLIB_ENGINE_VERSION,
        "effective_date": "2026-07-13",
        "cost_model": CostModelConfig().to_dict(),
        "cash_weight": 0.98,
        "holdings": [
            {
                "instrument": "SH600000",
                "weight": 0.02,
                "previous_weight": 0.0,
                "weight_change": 0.02,
                "action": "increase",
                "reason": "ranked signal and constraints",
            }
        ],
        "hypothetical_observation": {
            "trade_date": "2026-07-10",
            "hypothetical_value": 4_999_000,
            "daily_return": 0.0,
            "benchmark_return": 0.0,
            "drawdown": 0.0,
            "turnover": 0.02,
            "estimated_cost": 1_000,
        },
    }
    completed = store.apply_result(snapshot["id"], result)
    assert completed["status"] == "succeeded"
    assert completed["holdings"][0]["instrument"] == "SH600000"
    tracked = store.get(portfolio["id"])
    assert float(tracked["hypothetical_performance"][0]["hypothetical_value"]) == 4_999_000
    with store.engine.connect() as connection:
        after = (
            connection.scalar(select(func.count()).select_from(paper_orders)),
            connection.scalar(select(func.count()).select_from(paper_fills)),
        )
    assert after == before


def test_recommendation_result_identity_is_bound_and_cash_only_is_valid(
    tmp_path, database_url: str
) -> None:
    version_id = create_strategy_version(database_url, tmp_path)
    store = RecommendationStore(database_url)
    with store.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved")
        )
    portfolio = store.create(
        name="cash recommendation",
        strategy_version_id=version_id,
        dataset="snapshot",
        hypothetical_initial_value=5_000_000,
        actor="test",
    )
    snapshot, _ = store.create_snapshot(
        portfolio_id=portfolio["id"],
        as_of_date=date(2026, 7, 10),
        dataset="snapshot",
        dataset_identity_sha256=DATASET_IDENTITY,
    )
    result = {
        "status": "ok",
        "portfolio_id": portfolio["id"],
        "strategy_version_id": version_id,
        "dataset": "snapshot",
        "dataset_identity_sha256": DATASET_IDENTITY,
        "as_of_date": "2026-07-10",
        "effective_date": "2026-07-13",
        "policy_version": POLICY_VERSION,
        "backtest_engine_version": QLIB_ENGINE_VERSION,
        "cost_model": CostModelConfig().to_dict(),
        "cash_weight": 1.0,
        "holdings": [],
        "changes": [{"instrument": "SH600000", "action": "sell", "target_weight": 0.0}],
    }
    for field, bad_value in (
        ("portfolio_id", "wrong-portfolio"),
        ("strategy_version_id", "wrong-version"),
        ("dataset", "wrong-dataset"),
        ("dataset_identity_sha256", "b" * 64),
        ("as_of_date", "2026-07-09"),
    ):
        tampered = deepcopy(result)
        tampered[field] = bad_value
        with pytest.raises(ValueError, match="identity does not match"):
            store.apply_result(snapshot["id"], tampered)

    completed = store.apply_result(snapshot["id"], result)
    assert completed["status"] == "succeeded"
    assert completed["holdings"] == []
    assert completed["snapshot"]["cash_weight"] == 1.0
