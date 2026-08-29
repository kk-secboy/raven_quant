from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from governance_fixtures import (
    DATASET_IDENTITY,
    create_strategy_version,
    governed_etf_ready_evidence,
)
from sqlalchemy import insert, select, update

from quant_data.database import (
    backtest_runs,
    simulation_batches,
    simulation_nav,
    strategy_versions,
)
from quant_data.execution_contract import DAILY_QLIB_FIELD_CONTRACT_VERSION
from quant_data.history_bounds import GOVERNED_DAILY_STOCK_SCOPE_VERSION
from quant_platform.cost_model import COST_SCHEDULE_VERSION
from quant_platform.job_store import (
    ORDER_PLAN_AWAITING_EXECUTION_DATA,
    ORDER_PLAN_EXECUTION_TRADE_DATE_KEY,
    ORDER_PLAN_FAILED,
    ORDER_PLAN_MATERIALIZATION_STATUS_KEY,
    ORDER_PLAN_MATERIALIZED,
    ORDER_PLAN_SUPERSEDED,
    JobStore,
)
from quant_platform.scheduler import SchedulerEngine
from quant_platform.simulation_store import SimulationStore


def _daily_dataset() -> dict:
    return {
        "name": "snapshot",
        "start_date": "2008-01-01",
        "end_date": "2026-08-31",
        "provenance": {
            "frequency": "day",
            "dataset_identity_sha256": DATASET_IDENTITY,
            "dataset_lineage_id": "b" * 64,
            "source_lineage_id": "c" * 64,
            "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
            "source_volume_unit": "hand",
            "qlib_volume_unit": "share",
            "source_amount_unit": "thousand_cny",
            "qlib_amount_unit": "cny",
            "source_hand_size": 100,
            "index_volume_policy": "excluded_non_tradable_benchmark",
            "governed_etf_whitelist": governed_etf_ready_evidence(),
            "lineage_verified": True,
            "execution_controls": {
                "formal_execution_requires_native_controls": True,
                "scope_version": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
                "native_complete_from": "2008-01-01",
            },
        },
    }


def _paper_account(database_url: str, tmp_path: Path, *, name: str) -> dict:
    version_id = create_strategy_version(database_url, tmp_path)
    store = SimulationStore(database_url)
    now = datetime.now(UTC)
    with store.engine.begin() as connection:
        version = connection.execute(
            select(strategy_versions).where(strategy_versions.c.id == version_id)
        ).one()
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved")
        )
        connection.execute(
            insert(backtest_runs).values(
                id=f"formal-{version_id}",
                strategy_version_id=version_id,
                dataset="snapshot",
                execution_dataset="snapshot",
                signal_frequency=version.signal_frequency,
                execution_frequency="day",
                execution_contract_hash=version.execution_contract_hash,
                qlib_version=version.qlib_version,
                qlib_commit=version.qlib_commit,
                rdagent_version=version.rdagent_version,
                rdagent_commit=version.rdagent_commit,
                status="succeeded",
                periods_json={"start": "2024-01-01", "end": "2026-08-28"},
                artifact_path=str(tmp_path),
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
    account = store.create(
        name=name,
        source_type="strategy_version",
        source_id=version_id,
        daily_dataset=_daily_dataset(),
        execution_dataset=_daily_dataset(),
        initial_cash=100_000,
        execution_policy={},
        cost_schedule_version=COST_SCHEDULE_VERSION,
        actor="test",
        daily_roll_policy="latest_compatible",
        execution_roll_policy="latest_compatible",
    )
    return store.set_status(account["id"], "active")


def test_predecessor_gate_recovers_same_frozen_job_and_isolates_accounts(
    database_url: str,
    tmp_path: Path,
) -> None:
    simulations = SimulationStore(database_url)
    jobs = JobStore(database_url)
    first = _paper_account(database_url, tmp_path, name="first paper account")
    second = _paper_account(database_url, tmp_path, name="independent paper account")
    signal_date = date(2026, 8, 28)
    trade_date = date(2026, 8, 31)
    next_signal_date = trade_date
    manifest_sha256 = "a" * 64
    payload = {
        "simulation_portfolio_id": first["id"],
        "signal_date": signal_date.isoformat(),
        "signal_at": None,
        "dataset_identity_sha256": DATASET_IDENTITY,
        "model_artifact_binding": None,
        "actor": "autopilot",
    }
    plan = jobs.create(
        "simulation_order_plan",
        payload,
        tmp_path / "plan.log",
        dedupe_active_kind=False,
        idempotency_key=f"simulation-order-plan-v2:{first['id']}:{signal_date}",
    )
    claimed = jobs.claim_next(("simulation_order_plan",))
    assert claimed is not None and claimed["id"] == plan["id"]
    waiting_result = {
        "order_plan_manifest_sha256": manifest_sha256,
        ORDER_PLAN_MATERIALIZATION_STATUS_KEY: ORDER_PLAN_AWAITING_EXECUTION_DATA,
        ORDER_PLAN_EXECUTION_TRADE_DATE_KEY: trade_date.isoformat(),
        "simulation_batch_id": None,
    }
    jobs.finish(claimed["id"], exit_code=0, result=waiting_result)

    waiting = simulations.order_plan_predecessor_state(
        first["id"], signal_date=next_signal_date
    )
    assert waiting["ready"] is False
    assert waiting["status"] == ORDER_PLAN_AWAITING_EXECUTION_DATA
    assert simulations.order_plan_predecessor_state(
        second["id"], signal_date=next_signal_date
    )["ready"] is True

    jobs.terminate_simulation_order_plan_materialization(
        plan["id"],
        order_plan_manifest_sha256=manifest_sha256,
        materialization_status=ORDER_PLAN_FAILED,
        reason="temporary artifact mount failure",
    )
    failed = simulations.order_plan_predecessor_state(
        first["id"], signal_date=next_signal_date
    )
    assert failed["predecessor_job_status"] == "failed"
    assert failed["recovery"] == "retry_frozen_simulation_order_plan"

    failed_progress = jobs.get(plan["id"])["progress"]
    retried = jobs.retry(plan["id"])
    assert retried["payload"] == payload
    assert retried["status"] == "succeeded"
    assert retried["progress"] == {
        **failed_progress,
        ORDER_PLAN_MATERIALIZATION_STATUS_KEY: ORDER_PLAN_AWAITING_EXECUTION_DATA,
    }
    # LocalJobWorker claims only queued jobs.  A sealed-plan retry therefore
    # cannot re-enter signal generation after a newer daily publication.
    assert jobs.claim_next(("simulation_order_plan",)) is None

    materialization_calls: list[dict] = []
    now = datetime.now(UTC)
    batch_id = "batch-deferred-1"
    class MaterializingSimulations:
        @staticmethod
        def create_batch_from_order_plan(
            portfolio_id: str,
            *,
            order_plan_manifest_sha256: str,
            data_root: Path,
            actor: str,
        ) -> tuple[dict, bool]:
            materialization_calls.append(
                {
                    "portfolio_id": portfolio_id,
                    "manifest_sha256": order_plan_manifest_sha256,
                    "data_root": data_root,
                    "actor": actor,
                }
            )
            with simulations.engine.begin() as connection:
                connection.execute(
                    insert(simulation_batches).values(
                        id=batch_id,
                        portfolio_id=first["id"],
                        recommendation_snapshot_id=None,
                        source_snapshot_id=DATASET_IDENTITY,
                        target_payload_json={"target_weights": {}},
                        execution_adapter="long_only",
                        execution_contract_hash=first["execution_contract_hash"],
                        daily_dataset=first["daily_dataset"],
                        daily_dataset_identity_sha256=first[
                            "daily_dataset_identity_sha256"
                        ],
                        daily_dataset_lineage_id=first[
                            "daily_dataset_lineage_id"
                        ],
                        execution_dataset=first["execution_dataset"],
                        execution_dataset_identity_sha256=first[
                            "execution_dataset_identity_sha256"
                        ],
                        execution_dataset_lineage_id=first[
                            "execution_dataset_lineage_id"
                        ],
                        simulation_semantics_sha256=first["execution_policy"][
                            "simulation_semantics_sha256"
                        ],
                        signal_date=signal_date,
                        trade_date=trade_date,
                        signal_at=None,
                        execution_not_before=None,
                        status="queued",
                        idempotency_key=(
                            f"qlib-order-plan:{first['id']}:{manifest_sha256}"
                        ),
                        created_by="simulation-order-plan-scheduler",
                        created_at=now,
                    )
                )
            return {"id": batch_id, "trade_date": trade_date.isoformat()}, True

    scheduler = object.__new__(SchedulerEngine)
    scheduler.settings = SimpleNamespace(data_root=tmp_path)
    scheduler.jobs = jobs
    scheduler.simulations = MaterializingSimulations()
    scheduler.alerts = SimpleNamespace(create=lambda **_kwargs: None)
    # 2026-09-01 represents a newer published daily snapshot.  Recovery still
    # consumes the original 2026-08-28 manifest and never regenerates a signal.
    assert scheduler._materialize_awaiting_simulation_order_plans(date(2026, 9, 1)) == 1
    assert materialization_calls == [
        {
            "portfolio_id": first["id"],
            "manifest_sha256": manifest_sha256,
            "data_root": tmp_path,
            "actor": "autopilot",
        }
    ]
    recovered = jobs.get(plan["id"])
    assert recovered["progress"][ORDER_PLAN_MATERIALIZATION_STATUS_KEY] == (
        ORDER_PLAN_MATERIALIZED
    )
    assert recovered["progress"]["simulation_batch_id"] == batch_id
    unsettled = simulations.order_plan_predecessor_state(
        first["id"], signal_date=next_signal_date
    )
    assert unsettled["status"] == "predecessor_batch_not_succeeded"

    with simulations.engine.begin() as connection:
        connection.execute(
            update(simulation_batches)
            .where(simulation_batches.c.id == batch_id)
            .values(status="succeeded", started_at=now, finished_at=now)
        )
        connection.execute(
            insert(simulation_nav).values(
                portfolio_id=first["id"],
                trade_date=trade_date,
                cash=Decimal("100000"),
                market_value=Decimal("0"),
                nav=Decimal("100000"),
                daily_return=0.0,
                drawdown=0.0,
                market_date=trade_date,
                has_stale_prices=False,
                status="healthy",
                performance_certified=True,
                produced_by="simulation-order-plan-scheduler",
                created_at=now,
            )
        )

    settled = simulations.require_order_plan_predecessor_settled(
        first["id"], signal_date=next_signal_date
    )
    assert settled["status"] == "predecessor_settled"
    assert settled["predecessor_batch_id"] == batch_id


def test_order_plan_retry_boundaries_do_not_change_other_jobs(
    database_url: str,
    tmp_path: Path,
) -> None:
    jobs = JobStore(database_url)

    ordinary = jobs.create(
        "qlib_baseline",
        {"dataset": "snapshot"},
        tmp_path / "ordinary.log",
        dedupe_active_kind=False,
    )
    claimed = jobs.claim_next(("qlib_baseline",))
    assert claimed is not None and claimed["id"] == ordinary["id"]
    jobs.finish(claimed["id"], exit_code=1, result={"partial": "discarded"})
    ordinary_retry = jobs.retry(ordinary["id"])
    assert ordinary_retry["status"] == "queued"
    assert ordinary_retry["progress"] is None

    unsealed = jobs.create(
        "simulation_order_plan",
        {"simulation_portfolio_id": "pre-artifact", "signal_date": "2026-08-28"},
        tmp_path / "unsealed.log",
        dedupe_active_kind=False,
    )
    claimed = jobs.claim_next(("simulation_order_plan",))
    assert claimed is not None and claimed["id"] == unsealed["id"]
    jobs.finish(claimed["id"], exit_code=1, result={"phase": "before_artifact"})
    unsealed_retry = jobs.retry(unsealed["id"])
    assert unsealed_retry["status"] == "queued"
    assert unsealed_retry["progress"] is None

    superseded = jobs.create(
        "simulation_order_plan",
        {"simulation_portfolio_id": "old-stage", "signal_date": "2026-08-28"},
        tmp_path / "superseded.log",
        dedupe_active_kind=False,
    )
    claimed = jobs.claim_next(("simulation_order_plan",))
    assert claimed is not None and claimed["id"] == unsealed["id"]
    jobs.finish(claimed["id"], exit_code=1)
    claimed = jobs.claim_next(("simulation_order_plan",))
    assert claimed is not None and claimed["id"] == superseded["id"]
    sealed = {
        "order_plan_manifest_sha256": "d" * 64,
        ORDER_PLAN_MATERIALIZATION_STATUS_KEY: ORDER_PLAN_AWAITING_EXECUTION_DATA,
        ORDER_PLAN_EXECUTION_TRADE_DATE_KEY: "2026-08-31",
    }
    jobs.finish(claimed["id"], exit_code=0, result=sealed)
    jobs.terminate_simulation_order_plan_materialization(
        superseded["id"],
        order_plan_manifest_sha256="d" * 64,
        materialization_status=ORDER_PLAN_SUPERSEDED,
        reason="promotion stage was replaced",
    )
    with pytest.raises(ValueError, match="superseded.*cannot be retried"):
        jobs.retry(superseded["id"])
