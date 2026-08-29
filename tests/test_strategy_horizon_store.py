from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from governance_fixtures import create_strategy_version
from sqlalchemy import delete, text, update
from sqlalchemy.exc import DBAPIError

from quant_data.database import (
    open_database,
    strategy_forward_gates,
    strategy_health_snapshots,
    strategy_versions,
)
from quant_platform.promotion import ForwardGateThresholds, PromotionStore
from quant_platform.research_horizon import SHORT_1_5D, canonical_sha256
from quant_platform.strategy_recipes import get_strategy_recipe
from quant_platform.strategy_store import StrategyStore


def _short_version(database_url: str, tmp_path: Path) -> str:
    base_id = create_strategy_version(database_url, tmp_path)
    store = StrategyStore(database_url)
    base = store.get_version(base_id)
    config = dict(base["config"])
    config.pop("horizon_contract", None)
    config.pop("horizon_contract_sha256", None)
    config.update(
        deepcopy(get_strategy_recipe("short_relative_strength")["config_overrides"])
    )
    config["factor_source_mode"] = "promoted_only"
    config["challenger_weight"] = 1.0
    config["outer_purge_days"] = 6
    config["outer_embargo_days"] = 6
    created = store.create_version(
        str(base["strategy_id"]),
        benchmark=str(base["benchmark"]),
        universe=str(base["universe"]),
        factors=[
            {
                "candidate_id": str(base["factors"][0]["factor_candidate_id"]),
                "weight": 1.0,
            }
        ],
        config=config,
        actor="horizon-test",
    )
    return str(created["id"])


def test_strategy_version_persists_one_canonical_horizon_contract(
    database_url: str, tmp_path: Path
) -> None:
    version_id = _short_version(database_url, tmp_path)
    version = StrategyStore(database_url).get_version(version_id)

    assert version["horizon_profile"] == SHORT_1_5D
    assert version["label_horizons_json"] == [1, 2, 3, 5]
    assert version["decision_interval_sessions"] == 1
    assert version["review_interval_sessions"] == 1
    assert version["holding_min_sessions"] == 1
    assert version["holding_target_sessions"] == 3
    assert version["holding_max_sessions"] == 5
    assert version["execution_lag_sessions"] == 1
    assert version["purge_sessions"] == 6
    assert version["embargo_sessions"] == 6
    assert version["sealed_oos_required"] is True
    assert version["sealed_oos_sessions"] == 252
    assert canonical_sha256(version["horizon_contract_json"]) == version[
        "horizon_contract_sha256"
    ]

    with pytest.raises(DBAPIError):
        with open_database(database_url).begin() as connection:
            connection.execute(
                update(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .values(decision_interval_sessions=None)
            )


def test_forward_gate_persists_and_checks_horizon_bound_criteria(
    database_url: str, tmp_path: Path
) -> None:
    version_id = _short_version(database_url, tmp_path)
    promotion = PromotionStore(database_url)
    gate = promotion.register_forward_gate(
        version_id,
        actor="horizon-test",
        thresholds=ForwardGateThresholds(
            min_forward_calendar_days=0,
            min_forward_trading_days=90,
            min_decision_batches=60,
            min_closed_round_trips=30,
        ),
    )

    assert gate["criteria_json"]["horizon_profile"] == SHORT_1_5D
    assert gate["criteria_json"]["thresholds"]["min_forward_trading_days"] == 90
    assert gate["criteria_json"]["thresholds"]["min_decision_batches"] == 60
    assert canonical_sha256(gate["criteria_json"]) == gate["criteria_sha256"]

    engine = open_database(database_url)
    with pytest.raises(DBAPIError):
        with engine.begin() as connection:
            connection.execute(
                update(strategy_forward_gates)
                .where(strategy_forward_gates.c.strategy_version_id == version_id)
                .values(min_forward_trading_days=91)
            )
    with engine.begin() as connection:
        connection.execute(
            update(strategy_forward_gates)
            .where(strategy_forward_gates.c.strategy_version_id == version_id)
            .values(criteria_sha256="0" * 64)
        )
    evaluated = promotion.evaluate_forward_gate(version_id)
    assert evaluated["passed"] is False
    assert evaluated["reasons"] == ["forward evidence gate criteria seal is invalid"]


def test_strategy_health_snapshots_are_sealed_idempotent_and_append_only(
    database_url: str, tmp_path: Path
) -> None:
    version_id = _short_version(database_url, tmp_path)
    store = StrategyStore(database_url)
    values = {
        "version_id": version_id,
        "as_of": datetime(2026, 8, 28, 8, 0, tzinfo=UTC),
        "health_status": "healthy",
        "criteria": {"max_drawdown": 0.15, "minimum_batches": 20},
        "evidence": {"drawdown": 0.03, "batches": 24},
        "actor": "health-owner",
    }
    first = store.record_health_snapshot(**values)
    retry = store.record_health_snapshot(**values)

    assert retry["id"] == first["id"]
    assert store.list_health_snapshots(version_id)[0]["health_status"] == "healthy"
    assert canonical_sha256(first["criteria_json"]) == first["criteria_sha256"]
    assert canonical_sha256(first["evidence_json"]) == first["evidence_sha256"]

    engine = open_database(database_url)
    with pytest.raises(DBAPIError, match="append-only"):
        with engine.begin() as connection:
            connection.execute(
                update(strategy_health_snapshots)
                .where(strategy_health_snapshots.c.id == first["id"])
                .values(health_status="watch")
            )
    with pytest.raises(DBAPIError, match="append-only"):
        with engine.begin() as connection:
            connection.execute(
                delete(strategy_health_snapshots).where(
                    strategy_health_snapshots.c.id == first["id"]
                )
            )
    with pytest.raises(DBAPIError, match="append-only"):
        with engine.begin() as connection:
            connection.execute(text("TRUNCATE quantlab.strategy_health_snapshots"))
