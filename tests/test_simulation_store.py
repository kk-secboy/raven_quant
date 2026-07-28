from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from governance_fixtures import DATASET_IDENTITY, create_strategy_version
from qlib_test_doubles import qlib_workflow_identity
from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import DBAPIError, IntegrityError

from quant_data.database import (
    backtest_runs,
    recommendation_snapshots,
    simulation_batches,
    simulation_nav,
    simulation_orders,
    simulation_portfolios,
    strategy_versions,
)
from quant_data.execution_contract import (
    MINUTE_EXECUTION_CONTRACT_VERSION,
    MINUTE_SOURCE_UNIT_CONTRACTS,
)
from quant_platform.api import create_app
from quant_platform.cost_model import COST_SCHEDULE_VERSION
from quant_platform.pair_trading import PairTradingConfig
from quant_platform.portfolio_policy import POLICY_VERSION
from quant_platform.qlib_backtest import QLIB_ENGINE_VERSION
from quant_platform.recommendation_store import RecommendationStore
from quant_platform.simulation_store import SimulationStore
from quant_platform.strategy_store import StrategyStore

TRADE_DATE = date(2026, 7, 13)
SOURCE_LINEAGE = "9" * 64
EXECUTION_IDENTITY = "d" * 64
EXECUTION_LINEAGE = "e" * 64


def test_forward_batch_binds_immutable_daily_and_execution_descendants(
    tmp_path, monkeypatch
) -> None:
    daily_lineage = "b" * 64
    execution_lineage = "d" * 64
    source_lineage = "9" * 64
    daily = {
        "name": "daily-descendant",
        "ready": True,
        "provenance": {
            "dataset_identity_sha256": "a" * 64,
            "dataset_lineage_id": daily_lineage,
            "source_lineage_id": source_lineage,
        },
    }
    execution = {
        "name": "minute-descendant",
        "ready": True,
        "provenance": {
            "dataset_identity_sha256": "c" * 64,
            "dataset_lineage_id": execution_lineage,
            "source_lineage_id": source_lineage,
        },
    }
    monkeypatch.setattr(
        "quant_platform.services.list_qlib_datasets", lambda _: [daily]
    )
    monkeypatch.setattr(
        "quant_platform.data_rollover.select_qlib_dataset",
        lambda *args, **kwargs: execution,
    )
    portfolio = SimpleNamespace(
        daily_dataset="daily-anchor",
        daily_dataset_identity_sha256="e" * 64,
        daily_dataset_lineage_id=daily_lineage,
        daily_roll_policy="latest_compatible",
        execution_dataset="minute-anchor",
        execution_dataset_identity_sha256="f" * 64,
        execution_dataset_lineage_id=execution_lineage,
        execution_roll_policy="latest_compatible",
    )
    snapshot = SimpleNamespace(
        dataset="daily-descendant",
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id=daily_lineage,
    )

    bindings = SimulationStore.__new__(
        SimulationStore
    )._snapshot_batch_dataset_bindings(
        portfolio=portfolio,
        snapshot=snapshot,
        trade_date=date(2026, 7, 28),
        data_root=tmp_path,
    )

    assert bindings["daily_dataset"] == "daily-descendant"
    assert bindings["execution_dataset"] == "minute-descendant"
    assert bindings["daily_dataset_identity_sha256"] == "a" * 64
    assert bindings["execution_dataset_identity_sha256"] == "c" * 64


def _daily_dataset() -> dict:
    return {
        "name": "snapshot",
        "provenance": {
            "frequency": "day",
            "dataset_identity_sha256": DATASET_IDENTITY,
            "dataset_lineage_id": "b" * 64,
            "source_lineage_id": SOURCE_LINEAGE,
            "field_contract_version": "daily-qlib-field-v3-cny-amount",
            "source_volume_unit": "hand",
            "qlib_volume_unit": "share",
            "source_amount_unit": "thousand_cny",
            "qlib_amount_unit": "cny",
            "source_hand_size": 100,
            "index_volume_policy": "excluded_non_tradable_benchmark",
            "lineage_verified": True,
        },
    }


def _execution_dataset(frequency: str = "5min") -> dict:
    source_dataset = "ashare_5m" if frequency == "5min" else "liquid_stocks_1m"
    return {
        "name": f"snapshot-{frequency}",
        "provenance": {
            "frequency": frequency,
            "dataset_identity_sha256": EXECUTION_IDENTITY,
            "dataset_lineage_id": EXECUTION_LINEAGE,
            "source_lineage_id": SOURCE_LINEAGE,
            "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
            "fields": ["vwap", "volume", "paused", "up_limit", "down_limit"],
            "source_datasets": [source_dataset],
            "source_unit_contracts": {source_dataset: MINUTE_SOURCE_UNIT_CONTRACTS[source_dataset]},
            "lineage_verified": True,
        },
    }


def _execution_evidence(
    batch_id: str,
    contract_hash: str,
    simulation_semantics_sha256: str,
) -> dict:
    return {
        "batch_id": batch_id,
        "dataset_identity_sha256": EXECUTION_IDENTITY,
        "dataset_lineage_id": EXECUTION_LINEAGE,
        "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
        "execution_contract_hash": contract_hash,
        "simulation_semantics_sha256": simulation_semantics_sha256,
        "next_trade_date": "2026-07-14",
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
    version_id = create_strategy_version(
        database_url,
        tmp_path,
        config_overrides={
            "execution_frequency": "5min",
            "execution_method": "twap",
        },
    )
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


def _approved_source_version(
    database_url: str,
    tmp_path,
    *,
    pair: bool = False,
    frequency: str = "5min",
    config_overrides: dict | None = None,
) -> str:
    execution_frequency = "1min" if pair else frequency
    overrides = {
        "execution_frequency": execution_frequency,
        "execution_method": "vwap" if pair else "twap",
    }
    overrides.update(config_overrides or {})
    version_id = create_strategy_version(
        database_url,
        tmp_path,
        config_overrides=overrides,
    )
    store = SimulationStore(database_url)
    now = datetime.now(UTC)
    with store.engine.begin() as connection:
        version = connection.execute(
            select(strategy_versions).where(strategy_versions.c.id == version_id)
        ).one()
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved", strategy_type="pair" if pair else "multifactor")
        )
        connection.execute(
            insert(backtest_runs).values(
                id=f"formal-{version_id}",
                strategy_version_id=version_id,
                dataset="snapshot",
                execution_dataset=f"snapshot-{execution_frequency}",
                signal_frequency=version.signal_frequency,
                execution_frequency=execution_frequency,
                execution_contract_hash=version.execution_contract_hash,
                qlib_version=version.qlib_version,
                qlib_commit=version.qlib_commit,
                rdagent_version=version.rdagent_version,
                rdagent_commit=version.rdagent_commit,
                status="succeeded",
                periods_json={"start": "2024-01-01", "end": "2026-07-10"},
                artifact_path=str(tmp_path),
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
    return version_id


def _write_qlib_order_plan(
    tmp_path,
    *,
    version_id: str,
    execution_contract_hash: str,
    target_weights: dict[str, float] | None = None,
    signal_date: date = date(2026, 7, 10),
    trade_date: date = TRADE_DATE,
    signal_at: datetime | None = None,
    execution_not_before: datetime | None = None,
) -> str:
    normalized_weights = dict(
        sorted((target_weights or {"SH600000": 0.001}).items())
    )
    targets = {"target_weights": normalized_weights}
    target_bytes = json.dumps(
        targets, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest = {
        "format_version": "qlib-order-plan-v1",
        "produced_by": "qlib-workflow-recorder",
        "source_type": "strategy_version",
        "source_id": version_id,
        "formal_backtest_id": f"formal-{version_id}",
        "execution_contract_hash": execution_contract_hash,
        "daily_dataset": "snapshot",
        "signal_date": signal_date.isoformat(),
        "trade_date": trade_date.isoformat(),
        "source_snapshot": {
            "id": DATASET_IDENTITY,
            "dataset_identity_sha256": DATASET_IDENTITY,
            "dataset_lineage_id": "b" * 64,
        },
        "target_weights_file_sha256": hashlib.sha256(target_bytes).hexdigest(),
        "target_weights_sha256": hashlib.sha256(target_bytes).hexdigest(),
        "qlib_workflow": qlib_workflow_identity(),
    }
    if signal_at is not None or execution_not_before is not None:
        manifest["signal_at"] = signal_at.isoformat() if signal_at else None
        manifest["execution_not_before"] = (
            execution_not_before.isoformat() if execution_not_before else None
        )
        manifest["signal_snapshot"] = {
            "name": "snapshot-5min",
            "dataset_identity_sha256": EXECUTION_IDENTITY,
            "dataset_lineage_id": EXECUTION_LINEAGE,
            "source_lineage_id": SOURCE_LINEAGE,
            "frequency": "5min",
        }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    artifact = (
        tmp_path
        / "data"
        / "artifacts"
        / "order-plans"
        / manifest_sha256
    )
    artifact.mkdir(parents=True)
    (artifact / "manifest.json").write_bytes(manifest_bytes)
    (artifact / "target_weights.json").write_bytes(target_bytes)
    return manifest_sha256


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
        closing_prices={"SH600000": {"price": 10.0, "market_date": TRADE_DATE.isoformat()}},
        execution_evidence=_execution_evidence(
            batch["id"],
            simulation["execution_contract_hash"],
            simulation["execution_policy"]["simulation_semantics_sha256"],
        ),
    )
    assert completed["status"] == "succeeded"
    assert completed["summary"]["conservation"] == {
        "cash_difference": pytest.approx(0.0),
        "negative_positions": 0,
    }
    assert (
        store.process_batch(
            batch["id"],
            minute_bars=_bars(),
            closing_prices={},
            execution_evidence={},
        )["status"]
        == "succeeded"
    )
    nav = store.rows(simulation["id"], "nav")
    assert len(nav) == 1
    assert nav[0]["performance_certified"] is True
    assert len(store.rows(simulation["id"], "fills")) == 1


def test_certified_nav_review_is_four_eyes_and_database_immutable(
    database_url: str, tmp_path
) -> None:
    store, simulation, batch = _create_batch(database_url, tmp_path)
    store.process_batch(
        batch["id"],
        minute_bars=_bars(),
        closing_prices={"SH600000": {"price": 10.0, "market_date": TRADE_DATE.isoformat()}},
        execution_evidence=_execution_evidence(
            batch["id"],
            simulation["execution_contract_hash"],
            simulation["execution_policy"]["simulation_semantics_sha256"],
        ),
    )
    nav = store.rows(simulation["id"], "nav")[0]
    assert nav["nav_scope"] == "member_ledger"
    assert nav["produced_by"] == "recommendation-worker"
    assert nav["reviewed_at"] is None

    with pytest.raises(ValueError, match="must differ"):
        store.review_nav(
            simulation["id"],
            TRADE_DATE,
            actor="recommendation-worker",
            evidence_sha256="a" * 64,
            note="Producer cannot approve the NAV that it created.",
        )
    reviewed = store.review_nav(
        simulation["id"],
        TRADE_DATE,
        actor="risk-reviewer",
        evidence_sha256="b" * 64,
        note="Reconciled cash, fills, positions, lineage, and daily NAV.",
    )
    assert reviewed["review_subject"] == "member_simulation_ledger"
    assert reviewed["reviewed_by"] == "risk-reviewer"
    assert reviewed["review_evidence_sha256"] == "b" * 64
    readiness = store.get(simulation["id"])["review_readiness"]
    assert readiness["nav_scope"] == "member_ledger"
    assert readiness["reviewed_days"] == 1
    assert readiness["ready"] is False

    with pytest.raises(ValueError, match="immutable"):
        store.review_nav(
            simulation["id"],
            TRADE_DATE,
            actor="second-reviewer",
            evidence_sha256="c" * 64,
            note="A later review cannot replace the original immutable review.",
        )
    with pytest.raises(DBAPIError, match="simulation NAV review is immutable"):
        with store.engine.begin() as connection:
            connection.execute(
                update(simulation_nav)
                .where(
                    simulation_nav.c.portfolio_id == simulation["id"],
                    simulation_nav.c.trade_date == TRADE_DATE,
                )
                .values(review_note="Attempted mutation of accepted review evidence.")
            )
    events = store.rows(simulation["id"], "events")
    assert any(item["event_type"] == "simulation_nav_reviewed" for item in events)


def test_uncertified_nav_cannot_be_reviewed(database_url: str, tmp_path) -> None:
    store, simulation, _batch = _create_batch(database_url, tmp_path)
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
                has_stale_prices=True,
                status="degraded",
                performance_certified=False,
                nav_scope="member_ledger",
                produced_by="simulation-engine",
                created_at=datetime.now(UTC),
            )
        )
    with pytest.raises(ValueError, match="performance-certified"):
        store.review_nav(
            simulation["id"],
            TRADE_DATE,
            actor="risk-reviewer",
            evidence_sha256="d" * 64,
            note="This row is intentionally uncertified and must remain blocked.",
        )


def test_recommendation_snapshot_queues_every_active_simulation_account(
    database_url: str, tmp_path
) -> None:
    store, first, first_batch = _create_batch(database_url, tmp_path)
    second_execution = _execution_dataset()
    second_execution["name"] = "snapshot-5min-secondary"
    second = store.create(
        name="second execution account",
        recommendation_portfolio_id=first["source_id"],
        daily_dataset=_daily_dataset(),
        execution_dataset=second_execution,
        initial_cash=2_000_000,
        execution_policy={"execution_algorithm": "twap"},
        cost_schedule_version=COST_SCHEDULE_VERSION,
        actor="test",
    )
    store.set_status(second["id"], "active")

    batches = store.create_batches_for_snapshot(first_batch["recommendation_snapshot_id"])

    assert len(batches) == 2
    assert {item[0]["portfolio_id"] for item in batches} == {first["id"], second["id"]}
    assert {item[0]["id"] for item in batches if not item[1]} == {first_batch["id"]}
    assert len([item for item in batches if item[1]]) == 1


def test_simulation_rejects_mismatched_execution_lineage_before_booking(
    database_url: str, tmp_path
) -> None:
    store, simulation, batch = _create_batch(database_url, tmp_path)
    evidence = _execution_evidence(
        batch["id"],
        simulation["execution_contract_hash"],
        simulation["execution_policy"]["simulation_semantics_sha256"],
    )
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
            closing_prices={"SH600000": {"price": 10.0, "market_date": TRADE_DATE.isoformat()}},
            execution_evidence=_execution_evidence(
                batch["id"],
                simulation["execution_contract_hash"],
                simulation["execution_policy"]["simulation_semantics_sha256"],
            ),
        )
    assert store.get_batch(batch["id"])["status"] == "queued"
    with store.engine.connect() as connection:
        assert (
            connection.scalar(
                select(func.count())
                .select_from(simulation_orders)
                .where(simulation_orders.c.batch_id == batch["id"])
            )
            == 0
        )
        assert (
            connection.scalar(
                select(func.count())
                .select_from(recommendation_snapshots)
                .where(recommendation_snapshots.c.id == batch["recommendation_snapshot_id"])
            )
            == 1
        )


def test_approved_strategy_source_accepts_only_immutable_qlib_order_plans(
    database_url: str, tmp_path
) -> None:
    version_id = _approved_source_version(database_url, tmp_path, frequency="1min")
    store = SimulationStore(database_url)
    simulation = store.create(
        name="approved strategy 1min simulation",
        source_type="strategy_version",
        source_id=version_id,
        daily_dataset=_daily_dataset(),
        execution_dataset=_execution_dataset("1min"),
        initial_cash=1_000_000,
        execution_policy={"execution_algorithm": "twap"},
        cost_schedule_version=COST_SCHEDULE_VERSION,
        actor="test",
    )
    assert simulation["source_type"] == "strategy_version"
    assert simulation["source_id"] == version_id
    assert simulation["execution_frequency"] == "1min"
    with store.engine.connect() as connection:
        assert simulation["execution_contract_hash"] == connection.scalar(
            select(strategy_versions.c.execution_contract_hash).where(
                strategy_versions.c.id == version_id
            )
        )
    store.set_status(simulation["id"], "active")
    manifest_sha256 = _write_qlib_order_plan(
        tmp_path,
        version_id=version_id,
        execution_contract_hash=simulation["execution_contract_hash"],
    )
    batch, created = store.create_batch_from_order_plan(
        simulation["id"],
        order_plan_manifest_sha256=manifest_sha256,
        data_root=tmp_path / "data",
        actor="simulation-operator",
    )
    assert created is True
    repeated, created = store.create_batch_from_order_plan(
        simulation["id"],
        order_plan_manifest_sha256=manifest_sha256,
        data_root=tmp_path / "data",
        actor="simulation-operator",
    )
    assert created is False
    assert repeated["id"] == batch["id"]
    with pytest.raises(ValueError, match="direct simulation target payloads are forbidden"):
        store.create_batch_for_targets(
            simulation["id"],
            source_snapshot_id=DATASET_IDENTITY,
            signal_date=date(2026, 7, 10),
            trade_date=TRADE_DATE,
            target_payload={"target_weights": {"SH600000": 0.002}},
            execution_contract_hash=simulation["execution_contract_hash"],
            idempotency_key="strategy-target:qlib-order-plan-1",
        )
    completed = store.process_batch(
        batch["id"],
        minute_bars=_bars(),
        closing_prices={"SH600000": {"price": 10.0, "market_date": TRADE_DATE.isoformat()}},
        execution_evidence=_execution_evidence(
            batch["id"],
            simulation["execution_contract_hash"],
            simulation["execution_policy"]["simulation_semantics_sha256"],
        ),
    )
    assert completed["status"] == "succeeded"


def test_minute_order_plan_persists_and_executes_the_strict_next_bar(
    database_url: str, tmp_path
) -> None:
    version_id = _approved_source_version(
        database_url,
        tmp_path,
        config_overrides={
            "signal_frequency": "5min",
            "signal_period": 12,
            "execution_frequency": "5min",
            "execution_method": "next_bar",
            "rebalance_frequency": "bar",
        },
    )
    store = SimulationStore(database_url)
    simulation = store.create(
        name="strict minute next-bar simulation",
        source_type="strategy_version",
        source_id=version_id,
        daily_dataset=_daily_dataset(),
        execution_dataset=_execution_dataset(),
        initial_cash=1_000_000,
        execution_policy={},
        cost_schedule_version=COST_SCHEDULE_VERSION,
        actor="test",
    )
    store.set_status(simulation["id"], "active")
    signal_at = datetime.fromisoformat("2026-07-13T10:05:00+08:00")
    next_bar = datetime.fromisoformat("2026-07-13T10:10:00+08:00")
    same_bar_manifest = _write_qlib_order_plan(
        tmp_path,
        version_id=version_id,
        execution_contract_hash=simulation["execution_contract_hash"],
        signal_date=TRADE_DATE,
        trade_date=TRADE_DATE,
        signal_at=signal_at,
        execution_not_before=signal_at,
    )
    with pytest.raises(ValueError, match="same-bar execution is forbidden"):
        store.create_batch_from_order_plan(
            simulation["id"],
            order_plan_manifest_sha256=same_bar_manifest,
            data_root=tmp_path / "data",
            actor="simulation-operator",
        )

    manifest_sha256 = _write_qlib_order_plan(
        tmp_path,
        version_id=version_id,
        execution_contract_hash=simulation["execution_contract_hash"],
        signal_date=TRADE_DATE,
        trade_date=TRADE_DATE,
        signal_at=signal_at,
        execution_not_before=next_bar,
    )
    batch, created = store.create_batch_from_order_plan(
        simulation["id"],
        order_plan_manifest_sha256=manifest_sha256,
        data_root=tmp_path / "data",
        actor="simulation-operator",
    )
    assert created is True
    assert datetime.fromisoformat(batch["signal_at"]).astimezone(
        UTC
    ) == signal_at.astimezone(UTC)
    assert datetime.fromisoformat(batch["execution_not_before"]).astimezone(
        UTC
    ) == next_bar.astimezone(UTC)
    manifest = store.execution_manifest(batch["id"])
    assert datetime.fromisoformat(manifest["signal_at"]).astimezone(
        UTC
    ) == signal_at.astimezone(UTC)
    assert datetime.fromisoformat(manifest["execution_not_before"]).astimezone(
        UTC
    ) == next_bar.astimezone(UTC)
    bars = _bars().copy()
    bars["datetime"] = "2026-07-13 10:10:00"
    evidence = _execution_evidence(
        batch["id"],
        simulation["execution_contract_hash"],
        simulation["execution_policy"]["simulation_semantics_sha256"],
    )
    evidence.update(
        {
            "signal_at": manifest["signal_at"],
            "execution_not_before": manifest["execution_not_before"],
        }
    )
    completed = store.process_batch(
        batch["id"],
        minute_bars=bars,
        closing_prices={
            "SH600000": {"price": 10.0, "market_date": TRADE_DATE.isoformat()}
        },
        execution_evidence=evidence,
    )
    assert completed["status"] == "succeeded"
    assert datetime.fromisoformat(
        store.rows(simulation["id"], "fills")[0]["executed_at"]
    ).astimezone(UTC) == next_bar.astimezone(UTC)


def test_long_only_order_plan_and_execution_semantics_fail_closed_on_tampering(
    database_url: str, tmp_path
) -> None:
    version_id = _approved_source_version(database_url, tmp_path)
    store = SimulationStore(database_url)
    simulation = store.create(
        name="tamper guarded strategy simulation",
        source_type="strategy_version",
        source_id=version_id,
        daily_dataset=_daily_dataset(),
        execution_dataset=_execution_dataset(),
        initial_cash=1_000_000,
        execution_policy={},
        cost_schedule_version=COST_SCHEDULE_VERSION,
        actor="test",
    )
    store.set_status(simulation["id"], "active")
    manifest_sha256 = _write_qlib_order_plan(
        tmp_path,
        version_id=version_id,
        execution_contract_hash=simulation["execution_contract_hash"],
    )
    batch, _ = store.create_batch_from_order_plan(
        simulation["id"],
        order_plan_manifest_sha256=manifest_sha256,
        data_root=tmp_path / "data",
        actor="simulation-operator",
    )
    with store.engine.begin() as connection:
        stored = connection.execute(
            select(simulation_batches.c.target_payload_json).where(
                simulation_batches.c.id == batch["id"]
            )
        ).scalar_one()
        connection.execute(
            update(simulation_batches)
            .where(simulation_batches.c.id == batch["id"])
            .values(
                target_payload_json={
                    **dict(stored),
                    "target_weights": {"SH600000": 0.002},
                }
            )
        )
    with pytest.raises(ValueError, match="order-plan failed batch-time verification"):
        store.process_batch(
            batch["id"],
            minute_bars=_bars(),
            closing_prices={},
            execution_evidence=_execution_evidence(
                batch["id"],
                simulation["execution_contract_hash"],
                simulation["execution_policy"]["simulation_semantics_sha256"],
            ),
        )
    assert store.get_batch(batch["id"])["status"] == "queued"

    with store.engine.begin() as connection:
        policy = connection.execute(
            select(simulation_portfolios.c.execution_policy_json).where(
                simulation_portfolios.c.id == simulation["id"]
            )
        ).scalar_one()
        connection.execute(
            update(simulation_portfolios)
            .where(simulation_portfolios.c.id == simulation["id"])
            .values(
                execution_policy_json={
                    **dict(policy),
                    "max_participation": 0.20,
                }
            )
        )
    with pytest.raises(ValueError, match="approved source contract"):
        store.set_status(simulation["id"], "active")


def test_process_batch_uses_full_cost_parameters_from_approved_source_contract(
    database_url: str, tmp_path
) -> None:
    version_id = _approved_source_version(
        database_url,
        tmp_path,
        config_overrides={
            "buy_commission_rate": 0.01,
            "sell_commission_rate": 0.01,
            "stock_sell_stamp_duty_rate": 0.0,
            "transfer_fee_rate": 0.0,
            "fixed_slippage_rate": 0.0,
            "impact_at_max_participation": 0.0,
            "min_commission": 0.0,
        },
    )
    store = SimulationStore(database_url)
    simulation = store.create(
        name="governed cost strategy simulation",
        source_type="strategy_version",
        source_id=version_id,
        daily_dataset=_daily_dataset(),
        execution_dataset=_execution_dataset(),
        initial_cash=1_000_000,
        execution_policy={},
        cost_schedule_version=COST_SCHEDULE_VERSION,
        actor="test",
    )
    assert simulation["execution_policy"]["cost_model"]["buy_commission_rate"] == 0.01
    store.set_status(simulation["id"], "active")
    manifest_sha256 = _write_qlib_order_plan(
        tmp_path,
        version_id=version_id,
        execution_contract_hash=simulation["execution_contract_hash"],
    )
    batch, _ = store.create_batch_from_order_plan(
        simulation["id"],
        order_plan_manifest_sha256=manifest_sha256,
        data_root=tmp_path / "data",
        actor="simulation-operator",
    )
    store.process_batch(
        batch["id"],
        minute_bars=_bars(),
        closing_prices={
            "SH600000": {"price": 10.0, "market_date": TRADE_DATE.isoformat()}
        },
        execution_evidence=_execution_evidence(
            batch["id"],
            simulation["execution_contract_hash"],
            simulation["execution_policy"]["simulation_semantics_sha256"],
        ),
    )
    fill = store.rows(simulation["id"], "fills")[0]
    assert float(fill["gross_value"]) == pytest.approx(1_000.0)
    assert float(fill["fee"]) == pytest.approx(10.0)
    assert float(fill["cost_breakdown_json"]["commission"]) == pytest.approx(10.0)


def test_order_plan_artifact_tampering_is_rejected_before_batch_creation(
    database_url: str, tmp_path
) -> None:
    version_id = _approved_source_version(database_url, tmp_path)
    store = SimulationStore(database_url)
    simulation = store.create(
        name="artifact guarded strategy simulation",
        source_type="strategy_version",
        source_id=version_id,
        daily_dataset=_daily_dataset(),
        execution_dataset=_execution_dataset(),
        initial_cash=1_000_000,
        execution_policy={},
        cost_schedule_version=COST_SCHEDULE_VERSION,
        actor="test",
    )
    store.set_status(simulation["id"], "active")
    manifest_sha256 = _write_qlib_order_plan(
        tmp_path,
        version_id=version_id,
        execution_contract_hash=simulation["execution_contract_hash"],
    )
    target_path = (
        tmp_path
        / "data"
        / "artifacts"
        / "order-plans"
        / manifest_sha256
        / "target_weights.json"
    )
    target_path.write_text(
        json.dumps({"target_weights": {"SH600000": 0.50}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="failed immutable verification"):
        store.create_batch_from_order_plan(
            simulation["id"],
            order_plan_manifest_sha256=manifest_sha256,
            data_root=tmp_path / "data",
            actor="simulation-operator",
        )


def test_api_queues_qlib_order_plan_generation_without_client_targets(
    database_url: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version_id = _approved_source_version(database_url, tmp_path)
    store = SimulationStore(database_url)
    simulation = store.create(
        name="API order plan generation simulation",
        source_type="strategy_version",
        source_id=version_id,
        daily_dataset=_daily_dataset(),
        execution_dataset=_execution_dataset(),
        initial_cash=1_000_000,
        execution_policy={},
        cost_schedule_version=COST_SCHEDULE_VERSION,
        actor="test",
    )
    store.set_status(simulation["id"], "active")
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            f"/api/simulation-portfolios/{simulation['id']}/order-plans",
            json={"signal_date": "2026-07-10"},
        )
        forbidden = client.post(
            f"/api/simulation-portfolios/{simulation['id']}/order-plans",
            json={
                "signal_date": "2026-07-10",
                "target_weights": {"SH600000": 1.0},
            },
        )
    assert response.status_code == 202
    assert response.json()["kind"] == "simulation_order_plan"
    assert response.json()["payload"]["simulation_portfolio_id"] == simulation["id"]
    assert forbidden.status_code == 422


def test_legacy_or_changed_source_contract_cannot_be_activated(database_url: str, tmp_path) -> None:
    version_id = _approved_source_version(database_url, tmp_path)
    store = SimulationStore(database_url)
    simulation = store.create(
        name="source contract guarded simulation",
        source_type="strategy_version",
        source_id=version_id,
        daily_dataset=_daily_dataset(),
        execution_dataset=_execution_dataset(),
        initial_cash=1_000_000,
        execution_policy={"execution_algorithm": "twap"},
        cost_schedule_version=COST_SCHEDULE_VERSION,
        actor="test",
    )
    with store.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(execution_contract_hash="legacy-unversioned")
        )
    with pytest.raises(ValueError, match="missing or inconsistent"):
        store.set_status(simulation["id"], "active")


def test_pair_simulation_creation_is_research_only(database_url: str, tmp_path) -> None:
    """Persistent pair simulation ledgers are research-only rejects (design 6.4.3/13).

    The offline pair backtest path stays available, but the capitalized
    forward ledger can no longer be opened even for an approved pair version.
    """

    strategies = StrategyStore(database_url)
    created = strategies.create_pair(
        name="research only pair simulation",
        description="pair strategies keep offline backtests but no persistent ledger",
        leg_y="SH600000",
        leg_x="SH600001",
        asset_class="stock",
        shorting_mode="margin_borrow",
        config=asdict(PairTradingConfig()),
        actor="researcher-a",
    )
    version = created["versions"][0]
    backtest = strategies.create_backtest(
        version_id=version["id"],
        dataset="snapshot",
        execution_dataset="execution-snapshot/liquid_stocks_1m+margin_eligibility",
        periods={"start": "2024-01-01", "end": "2026-07-13"},
        artifact_path=tmp_path / "data" / "artifacts" / "backtests",
    )
    strategies.mark_backtest(backtest["id"], "succeeded", metrics={"provenance": {}})
    with strategies.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version["id"])
            .values(status="approved")
        )
    store = SimulationStore(database_url)
    with pytest.raises(ValueError, match="research_only"):
        store.create(
            name="research only pair simulation ledger",
            source_type="strategy_version",
            source_id=version["id"],
            daily_dataset=_daily_dataset(),
            execution_dataset=_execution_dataset("1min"),
            initial_cash=PairTradingConfig().initial_capital,
            execution_policy={
                "execution_algorithm": "vwap",
                "slice_minutes": 5,
                "max_slices": 1,
                "max_participation": 0.01,
                "volume_profile": [{"time": "10:00", "weight": 1.0}],
            },
            cost_schedule_version=COST_SCHEDULE_VERSION,
            actor="simulation-operator",
            execution_adapter="pair",
        )


def test_simulation_api_exposes_ledger_and_retires_hypothetical_performance(
    database_url: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, simulation, batch = _create_batch(database_url, tmp_path)
    store.process_batch(
        batch["id"],
        minute_bars=_bars(),
        closing_prices={"SH600000": {"price": 10.0, "market_date": TRADE_DATE.isoformat()}},
        execution_evidence=_execution_evidence(
            batch["id"],
            simulation["execution_contract_hash"],
            simulation["execution_policy"]["simulation_semantics_sha256"],
        ),
    )
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    with TestClient(create_app(tmp_path)) as client:
        listed = client.get("/api/simulation-portfolios")
        detail = client.get(f"/api/simulation-portfolios/{simulation['id']}")
        positions = client.get(f"/api/simulation-portfolios/{simulation['id']}/positions")
        reviewed = client.post(
            f"/api/simulation-portfolios/{simulation['id']}/nav/{TRADE_DATE.isoformat()}/review",
            json={
                "actor": "ignored-when-authenticated",
                "evidence_sha256": "e" * 64,
                "note": "API reviewer reconciled the certified member simulation ledger.",
            },
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
    assert reviewed.status_code == 200
    assert reviewed.json()["reviewed_by"] == "local-admin"
    assert reviewed.json()["review_subject"] == "member_simulation_ledger"
    assert paused.json()["status"] == "paused"
    assert activated.json()["status"] == "active"
    assert retired.status_code == 410
    assert store.get(simulation["id"])["status"] == "active"
