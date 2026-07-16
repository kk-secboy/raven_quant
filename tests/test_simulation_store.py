from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from governance_fixtures import DATASET_IDENTITY, create_strategy_version
from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    recommendation_snapshots,
    simulation_nav,
    simulation_orders,
    strategy_versions,
)
from quant_data.execution_contract import (
    MINUTE_EXECUTION_CONTRACT_VERSION,
    MINUTE_SOURCE_UNIT_CONTRACTS,
)
from quant_platform.api import create_app
from quant_platform.cost_model import COST_SCHEDULE_VERSION
from quant_platform.portfolio_policy import POLICY_VERSION
from quant_platform.qlib_backtest import QLIB_ENGINE_VERSION
from quant_platform.recommendation_store import RecommendationStore
from quant_platform.simulation_store import SimulationStore

TRADE_DATE = date(2026, 7, 13)
SOURCE_LINEAGE = "9" * 64
EXECUTION_IDENTITY = "d" * 64
EXECUTION_LINEAGE = "e" * 64


def _daily_dataset() -> dict:
    return {
        "name": "snapshot",
        "provenance": {
            "frequency": "day",
            "dataset_identity_sha256": DATASET_IDENTITY,
            "dataset_lineage_id": "b" * 64,
            "source_lineage_id": SOURCE_LINEAGE,
            "field_contract_version": "daily-qlib-field-v2-share-volume",
            "source_volume_unit": "hand",
            "qlib_volume_unit": "share",
            "source_hand_size": 100,
            "index_volume_policy": "excluded_non_tradable_benchmark",
            "lineage_verified": True,
        },
    }


def _execution_dataset() -> dict:
    return {
        "name": "snapshot-5m",
        "provenance": {
            "frequency": "5min",
            "dataset_identity_sha256": EXECUTION_IDENTITY,
            "dataset_lineage_id": EXECUTION_LINEAGE,
            "source_lineage_id": SOURCE_LINEAGE,
            "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
            "fields": ["vwap", "volume", "paused", "up_limit", "down_limit"],
            "source_datasets": ["ashare_5m"],
            "source_unit_contracts": {
                "ashare_5m": MINUTE_SOURCE_UNIT_CONTRACTS["ashare_5m"]
            },
            "lineage_verified": True,
        },
    }


def _execution_evidence(batch_id: str) -> dict:
    return {
        "batch_id": batch_id,
        "dataset_identity_sha256": EXECUTION_IDENTITY,
        "dataset_lineage_id": EXECUTION_LINEAGE,
        "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
    }


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "datetime": "2026-07-13 10:00:00",
                "instrument": "SH600000",
                "close": 10.0,
                "vwap": 10.0,
                "volume": 1_000_000,
                "paused": 0,
                "up_limit": 11.0,
                "down_limit": 9.0,
            }
        ]
    )


def _create_batch(database_url: str, tmp_path) -> tuple[SimulationStore, dict, dict]:
    version_id = create_strategy_version(database_url, tmp_path)
    recommendations = RecommendationStore(database_url)
    with recommendations.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved")
        )
    recommendation = recommendations.create(
        name="simulation target",
        strategy_version_id=version_id,
        dataset="snapshot",
        hypothetical_initial_value=1_000_000,
        actor="test",
    )
    simulation_store = SimulationStore(database_url)
    simulation = simulation_store.create(
        name="transactional simulation",
        recommendation_portfolio_id=recommendation["id"],
        daily_dataset=_daily_dataset(),
        execution_dataset=_execution_dataset(),
        initial_cash=1_000_000,
        execution_policy={"execution_algorithm": "twap"},
        cost_schedule_version=COST_SCHEDULE_VERSION,
        actor="test",
    )
    simulation_store.set_status(simulation["id"], "active")
    snapshot, _ = recommendations.create_snapshot(
        portfolio_id=recommendation["id"],
        as_of_date=date(2026, 7, 10),
        dataset="snapshot",
        dataset_identity_sha256=DATASET_IDENTITY,
    )
    recommendations.apply_result(
        snapshot["id"],
        {
            "status": "ok",
            "portfolio_id": recommendation["id"],
            "strategy_version_id": version_id,
            "dataset": "snapshot",
            "dataset_identity_sha256": DATASET_IDENTITY,
            "as_of_date": "2026-07-10",
            "effective_date": TRADE_DATE.isoformat(),
            "policy_version": POLICY_VERSION,
            "backtest_engine_version": QLIB_ENGINE_VERSION,
            "cost_model": snapshot["cost_model"],
            "cash_weight": 0.999,
            "holdings": [
                {
                    "instrument": "SH600000",
                    "weight": 0.001,
                    "previous_weight": 0.0,
                    "weight_change": 0.001,
                    "action": "increase",
                    "reason": "governed target",
                }
            ],
        },
    )
    batch, created = simulation_store.create_batch_for_snapshot(snapshot["id"])
    assert created is True
    assert batch is not None
    return simulation_store, simulation, batch


def test_simulation_batch_is_idempotent_and_books_auditable_nav(
    database_url: str, tmp_path
) -> None:
    store, simulation, batch = _create_batch(database_url, tmp_path)
    repeated, created = store.create_batch_for_snapshot(batch["recommendation_snapshot_id"])
    assert created is False
    assert repeated["id"] == batch["id"]

    completed = store.process_batch(
        batch["id"],
        minute_bars=_bars(),
        closing_prices={
            "SH600000": {"price": 10.0, "market_date": TRADE_DATE.isoformat()}
        },
        execution_evidence=_execution_evidence(batch["id"]),
    )
    assert completed["status"] == "succeeded"
    assert completed["summary"]["conservation"] == {
        "cash_difference": pytest.approx(0.0),
        "negative_positions": 0,
    }
    assert store.process_batch(
        batch["id"],
        minute_bars=_bars(),
        closing_prices={},
        execution_evidence={},
    )["status"] == "succeeded"
    nav = store.rows(simulation["id"], "nav")
    assert len(nav) == 1
    assert nav[0]["performance_certified"] is True
    assert len(store.rows(simulation["id"], "fills")) == 1


def test_simulation_rejects_mismatched_execution_lineage_before_booking(
    database_url: str, tmp_path
) -> None:
    store, simulation, batch = _create_batch(database_url, tmp_path)
    evidence = _execution_evidence(batch["id"])
    evidence["dataset_identity_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="bound dataset"):
        store.process_batch(
            batch["id"],
            minute_bars=_bars(),
            closing_prices={},
            execution_evidence=evidence,
        )
    assert store.get_batch(batch["id"])["status"] == "queued"
    assert store.rows(simulation["id"], "orders") == []


def test_simulation_booking_rolls_back_all_ledger_writes_on_nav_conflict(
    database_url: str, tmp_path
) -> None:
    store, simulation, batch = _create_batch(database_url, tmp_path)
    with store.engine.begin() as connection:
        connection.execute(
            insert(simulation_nav).values(
                portfolio_id=simulation["id"],
                trade_date=TRADE_DATE,
                cash=Decimal("1000000"),
                market_value=Decimal("0"),
                nav=Decimal("1000000"),
                daily_return=0.0,
                drawdown=0.0,
                market_date=TRADE_DATE,
                has_stale_prices=False,
                status="healthy",
                performance_certified=True,
                created_at=datetime.now(UTC),
            )
        )
    with pytest.raises(IntegrityError):
        store.process_batch(
            batch["id"],
            minute_bars=_bars(),
            closing_prices={
                "SH600000": {"price": 10.0, "market_date": TRADE_DATE.isoformat()}
            },
            execution_evidence=_execution_evidence(batch["id"]),
        )
    assert store.get_batch(batch["id"])["status"] == "queued"
    with store.engine.connect() as connection:
        assert (
            connection.scalar(
                select(func.count()).select_from(simulation_orders).where(
                    simulation_orders.c.batch_id == batch["id"]
                )
            )
            == 0
        )
        assert connection.scalar(
            select(func.count()).select_from(recommendation_snapshots).where(
                recommendation_snapshots.c.id == batch["recommendation_snapshot_id"]
            )
        ) == 1


def test_simulation_api_exposes_ledger_and_retires_hypothetical_performance(
    database_url: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, simulation, _batch = _create_batch(database_url, tmp_path)
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    with TestClient(create_app(tmp_path)) as client:
        listed = client.get("/api/simulation-portfolios")
        detail = client.get(f"/api/simulation-portfolios/{simulation['id']}")
        positions = client.get(
            f"/api/simulation-portfolios/{simulation['id']}/positions"
        )
        paused = client.post(f"/api/simulation-portfolios/{simulation['id']}/pause")
        activated = client.post(f"/api/simulation-portfolios/{simulation['id']}/activate")
        retired = client.get(
            "/api/recommendation-portfolios/"
            f"{simulation['recommendation_portfolio_id']}/hypothetical-performance"
        )
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == simulation["id"]
    assert detail.status_code == 200
    assert positions.status_code == 200
    assert paused.json()["status"] == "paused"
    assert activated.json()["status"] == "active"
    assert retired.status_code == 410
    assert store.get(simulation["id"])["status"] == "active"
