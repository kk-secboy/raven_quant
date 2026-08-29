from __future__ import annotations

import hashlib
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from governance_fixtures import governed_etf_ready_evidence
from sqlalchemy import func, insert, select, update
from test_simulation_store import _daily_dataset

import quant_platform.three_horizon_account as three_horizon_account
from quant_data.database import (
    open_database,
    simulation_batches,
    simulation_events,
    simulation_nav,
    simulation_orders,
    simulation_portfolios,
    simulation_positions,
    strategy_allocations,
)
from quant_data.execution_contract import DAILY_QLIB_FIELD_CONTRACT_VERSION
from quant_platform.allocation_store import AllocationStore
from quant_platform.cost_model import COST_SCHEDULE_VERSION, CostModelConfig
from quant_platform.simulation_store import (
    SimulationStore,
    build_allocation_source_rebind_event,
    validate_allocation_source_rebind_event,
)
from quant_platform.three_horizon_account import (
    THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
    ThreeHorizonAccountService,
)


def _contract_hash(source_id: str) -> str:
    return hashlib.sha256(f"allocation-contract:{source_id}".encode()).hexdigest()


def _allocation_source(connection, source_type: str, source_id: str) -> dict:
    if source_type != "allocation":
        raise ValueError("test source supports allocations only")
    row = connection.execute(
        select(strategy_allocations).where(strategy_allocations.c.id == source_id)
    ).one()
    if str(row.status) != "active":
        raise ValueError("simulation requires an approved active allocation")
    return {
        "dataset": str(row.dataset),
        "execution_adapter": "long_only",
        "execution_contract_hash": _contract_hash(source_id),
        "signal_frequency": "day",
        "signal_horizon": "1d",
        "execution_frequency": "",
        "execution_method": "",
        "benchmark": "SH000300",
        "config": CostModelConfig().to_dict(),
        "policy_mode": "self_contained",
    }


def _insert_allocation(
    connection,
    *,
    allocation_id: str,
    status: str,
    lineage: str = "b" * 64,
) -> None:
    now = datetime.now(UTC)
    connection.execute(
        insert(strategy_allocations).values(
            id=allocation_id,
            name=f"three-horizon-{allocation_id}",
            dataset="snapshot",
            status=status,
            is_legacy=False,
            allocation_method="fixed",
            decision_frequency="monthly",
            lookback_days=252,
            target_volatility=0.50,
            max_pairwise_correlation=0.99,
            max_strategy_weight=0.40,
            max_member_drawdown=0.08,
            max_drawdown_reduce=0.10,
            max_drawdown_liquidate=0.15,
            total_capital=Decimal("1000000"),
            cash_reserve=Decimal("800000"),
            nav=Decimal("1000000"),
            high_water_mark=Decimal("1000000"),
            analysis_json={
                "allocation_dataset_lineage_id": lineage,
                "approval_simulation_evidence": {"sealed": allocation_id},
            },
            created_by="system:auto-promotion",
            approved_by=("system:auto-promotion" if status == "active" else None),
            approval_reason=("governed test activation" if status == "active" else None),
            created_at=now,
            approved_at=(now if status == "active" else None),
            updated_at=now,
        )
    )


def _governed_daily_dataset() -> dict:
    dataset = _daily_dataset()
    dataset["end_date"] = "2026-08-28"
    dataset["provenance"]["field_contract_version"] = (
        DAILY_QLIB_FIELD_CONTRACT_VERSION
    )
    dataset["provenance"]["execution_controls"] = {
        "formal_execution_requires_native_controls": True,
        "native_complete_from": "2008-01-01",
    }
    dataset["provenance"]["governed_etf_whitelist"] = (
        governed_etf_ready_evidence()
    )
    return dataset


def _rolled_daily_dataset(*, name: str, identity: str, lineage: str) -> dict:
    dataset = _governed_daily_dataset()
    dataset["name"] = name
    dataset["ready"] = True
    dataset["reproducible"] = True
    dataset["lineage_id"] = lineage
    dataset["start_date"] = "2008-01-01"
    dataset["end_date"] = "2026-08-29"
    dataset["provenance"] = {
        **dataset["provenance"],
        "dataset_identity_sha256": identity,
        "dataset_lineage_id": lineage,
    }
    return dataset


def test_three_horizon_tick_serializes_the_multi_transaction_saga(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = open_database(database_url)
    services = []
    for _index in range(2):
        service = object.__new__(ThreeHorizonAccountService)
        service.engine = engine
        services.append(service)

    active = 0
    maximum_active = 0
    state_lock = threading.Lock()

    def _tick_locked(
        _service: ThreeHorizonAccountService, *, now: datetime | None = None
    ) -> dict:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.10)
        with state_lock:
            active -= 1
        return {"status": "tested", "now": now}

    monkeypatch.setattr(ThreeHorizonAccountService, "_tick_locked", _tick_locked)
    ready = threading.Barrier(2)

    def _run(service: ThreeHorizonAccountService) -> dict:
        ready.wait(timeout=2)
        return service.tick(now=datetime(2026, 8, 29, tzinfo=UTC))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(_run, services))

    assert [item["status"] for item in results] == ["tested", "tested"]
    assert maximum_active == 1


@pytest.mark.no_database
def test_allocation_source_rebind_event_is_deterministic_and_tamper_evident() -> None:
    kwargs = {
        "portfolio_id": "primary-ledger",
        "old_allocation_id": "short-only",
        "new_allocation_id": "short-swing",
        "actor": THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
        "old_execution_contract_hash": "a" * 64,
        "new_execution_contract_hash": "b" * 64,
        "old_simulation_semantics_sha256": "c" * 64,
        "new_simulation_semantics_sha256": "d" * 64,
    }
    first = build_allocation_source_rebind_event(**kwargs)
    second = build_allocation_source_rebind_event(**kwargs)

    assert first == second
    assert first["ledger_action"] == "in_place_source_rebind"
    assert first["ledger_rows_copied"] is False
    assert validate_allocation_source_rebind_event(
        first,
        portfolio_id="primary-ledger",
        old_allocation_id="short-only",
        new_allocation_id="short-swing",
    ) == first
    changed = dict(first)
    changed["new_allocation_id"] = "another-allocation"
    with pytest.raises(ValueError, match="seal"):
        validate_allocation_source_rebind_event(
            changed,
            portfolio_id="primary-ledger",
            old_allocation_id="short-only",
            new_allocation_id="another-allocation",
        )


@pytest.mark.no_database
def test_latest_compatible_account_batch_binds_the_verified_descendant(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descendant = _rolled_daily_dataset(
        name="snapshot-descendant",
        identity="c" * 64,
        lineage="b" * 64,
    )
    monkeypatch.setattr(
        "quant_platform.data_rollover.select_qlib_dataset",
        lambda *args, **kwargs: descendant,
    )
    monkeypatch.setattr(
        "quant_platform.data_rollover.next_qlib_trading_date",
        lambda _dataset, _trade_date: date(2026, 9, 1),
    )
    portfolio = SimpleNamespace(
        daily_dataset="snapshot",
        daily_dataset_identity_sha256="a" * 64,
        daily_dataset_lineage_id="b" * 64,
        daily_roll_policy="latest_compatible",
        execution_dataset="snapshot",
        execution_dataset_identity_sha256="a" * 64,
        execution_dataset_lineage_id="b" * 64,
        execution_roll_policy="latest_compatible",
        execution_frequency="day",
    )

    bindings = SimulationStore._account_order_plan_dataset_bindings(
        portfolio=portfolio,
        signal_date=date(2026, 8, 28),
        trade_date=date(2026, 8, 29),
        data_root=tmp_path,
    )

    assert bindings["daily_dataset"] == "snapshot-descendant"
    assert bindings["execution_dataset"] == "snapshot-descendant"
    assert bindings["daily_dataset_identity_sha256"] == "c" * 64
    assert bindings["execution_dataset_identity_sha256"] == "c" * 64


def _seed_primary_ledger(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[SimulationStore, dict, str, str]:
    monkeypatch.setattr(
        SimulationStore, "_resolve_source", staticmethod(_allocation_source)
    )
    store = SimulationStore(database_url)
    old_id = f"old-{uuid.uuid4().hex}"
    new_id = f"new-{uuid.uuid4().hex}"
    with store.engine.begin() as connection:
        _insert_allocation(connection, allocation_id=old_id, status="active")
        _insert_allocation(connection, allocation_id=new_id, status="draft")
    dataset = _governed_daily_dataset()
    simulation = store.create(
        name=f"three-horizon-primary-{uuid.uuid4().hex}",
        source_type="allocation",
        source_id=old_id,
        daily_dataset=dataset,
        execution_dataset=dataset,
        initial_cash=1_000_000,
        execution_policy={
            "execution_algorithm": "open",
            "execution_frequency": "day",
            "max_participation": 0.01,
        },
        cost_schedule_version=COST_SCHEDULE_VERSION,
        actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
        daily_roll_policy="latest_compatible",
        execution_roll_policy="latest_compatible",
    )
    simulation = store.set_status(str(simulation["id"]), "active")
    now = datetime.now(UTC)
    historical_batch_id = uuid.uuid4().hex
    with store.engine.begin() as connection:
        connection.execute(
            insert(simulation_positions).values(
                portfolio_id=simulation["id"],
                instrument="SH600000",
                position_side="long",
                quantity=100,
                available_quantity=100,
                frozen_quantity=0,
                average_cost=Decimal("10"),
                market_price=Decimal("11"),
                market_date=date(2026, 8, 28),
                stale=False,
                market_value=Decimal("1100"),
                updated_at=now,
            )
        )
        connection.execute(
            insert(simulation_nav).values(
                portfolio_id=simulation["id"],
                trade_date=date(2026, 8, 28),
                cash=Decimal("1000000"),
                market_value=Decimal("0"),
                nav=Decimal("1000000"),
                daily_return=0.0,
                drawdown=0.0,
                has_stale_prices=False,
                status="certified",
                performance_certified=True,
                nav_scope="aggregate_view",
                produced_by=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
                created_at=now,
            )
        )
        connection.execute(
            insert(simulation_batches).values(
                id=historical_batch_id,
                portfolio_id=simulation["id"],
                source_snapshot_id="e" * 64,
                target_payload_json={},
                execution_adapter="long_only",
                execution_contract_hash=simulation["execution_contract_hash"],
                daily_dataset=simulation["daily_dataset"],
                daily_dataset_identity_sha256=simulation[
                    "daily_dataset_identity_sha256"
                ],
                daily_dataset_lineage_id=simulation["daily_dataset_lineage_id"],
                execution_dataset=simulation["execution_dataset"],
                execution_dataset_identity_sha256=simulation[
                    "execution_dataset_identity_sha256"
                ],
                execution_dataset_lineage_id=simulation[
                    "execution_dataset_lineage_id"
                ],
                simulation_semantics_sha256=simulation["execution_policy"][
                    "simulation_semantics_sha256"
                ],
                signal_date=date(2026, 8, 27),
                trade_date=date(2026, 8, 28),
                status="succeeded",
                idempotency_key=f"historical-{historical_batch_id}",
                created_by=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
                summary_json={"conservation": {"cash_difference": 0.0}},
                created_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(simulation_orders).values(
                id=uuid.uuid4().hex,
                batch_id=historical_batch_id,
                portfolio_id=simulation["id"],
                instrument="SH600000",
                side="buy",
                position_side="long",
                target_weight=0.01,
                requested_quantity=100,
                filled_quantity=100,
                status="filled",
                requested_value=Decimal("1000"),
                filled_value=Decimal("1000"),
                capacity_fill_ratio=1.0,
                expires_at=now,
                created_at=now,
            )
        )
    return store, simulation, old_id, new_id


def test_progressive_horizon_rebind_preserves_one_ledger_and_is_restart_idempotent(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, simulation, old_id, new_id = _seed_primary_ledger(
        database_url, monkeypatch
    )
    before = store.get(str(simulation["id"]))
    before_semantics = before["execution_policy"]["simulation_semantics_sha256"]

    prepared = store.prepare_allocation_source_replacement(
        str(simulation["id"]),
        old_allocation_id=old_id,
        new_allocation_id=new_id,
        actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
    )
    assert prepared["status"] == "paused"
    restored = store.abort_allocation_source_replacement(
        str(simulation["id"]),
        old_allocation_id=old_id,
        new_allocation_id=new_id,
    )
    assert restored["status"] == "active"
    prepared = store.prepare_allocation_source_replacement(
        str(simulation["id"]),
        old_allocation_id=old_id,
        new_allocation_id=new_id,
        actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
    )
    assert prepared["status"] == "paused"
    with store.engine.begin() as connection:
        connection.execute(
            update(strategy_allocations)
            .where(strategy_allocations.c.id == old_id)
            .values(status="paused")
        )
        connection.execute(
            update(strategy_allocations)
            .where(strategy_allocations.c.id == new_id)
            .values(
                status="active",
                approved_by="system:auto-promotion",
                approved_at=datetime.now(UTC),
            )
        )

    rebound = store.replace_allocation_source(
        str(simulation["id"]),
        old_allocation_id=old_id,
        new_allocation_id=new_id,
        actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
    )
    assert rebound["id"] == simulation["id"]
    assert rebound["source_id"] == new_id
    assert rebound["status"] == "active"
    assert rebound["initial_cash"] == before["initial_cash"]
    assert rebound["cash"] == before["cash"]
    assert rebound["nav"] == before["nav"]
    assert rebound["execution_contract_hash"] == _contract_hash(new_id)
    assert rebound["execution_policy"]["source_execution_contract_hash"] == (
        _contract_hash(new_id)
    )
    assert (
        rebound["execution_policy"]["simulation_semantics_sha256"]
        != before_semantics
    )
    with store.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(simulation_portfolios)) == 1
        assert connection.scalar(select(func.count()).select_from(simulation_positions)) == 1
        assert connection.scalar(select(func.count()).select_from(simulation_nav)) == 1
        assert connection.scalar(select(func.count()).select_from(simulation_batches)) == 1
        assert connection.scalar(select(func.count()).select_from(simulation_orders)) == 1
        assert (
            connection.scalar(
                select(func.count())
                .select_from(simulation_events)
                .where(simulation_events.c.event_type == "allocation_source_rebound")
            )
            == 1
        )

    restarted = SimulationStore(database_url)
    replay = restarted.replace_allocation_source(
        str(simulation["id"]),
        old_allocation_id=old_id,
        new_allocation_id=new_id,
        actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
    )
    assert replay["id"] == simulation["id"]
    service = object.__new__(ThreeHorizonAccountService)
    service.engine = restarted.engine
    service.simulations = restarted
    monkeypatch.setattr(
        three_horizon_account,
        "list_qlib_datasets",
        lambda _root: [
            {
                **_governed_daily_dataset(),
                "ready": True,
                "reproducible": True,
                "lineage_id": "b" * 64,
            }
        ],
    )
    service.settings = type("Settings", (), {"data_root": None})()
    reused = service._ensure_simulation(
        allocation={"id": new_id, "name": "three-horizon-new"},
        dataset_name="snapshot",
        initial_cash=1_000_000,
    )
    assert reused["id"] == simulation["id"]
    with restarted.engine.connect() as connection:
        assert (
            connection.scalar(
                select(func.count())
                .select_from(simulation_events)
                .where(simulation_events.c.event_type == "allocation_source_rebound")
            )
            == 1
        )


def test_cutover_waits_without_pausing_serving_account_when_batch_is_unsettled(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, simulation, old_id, new_id = _seed_primary_ledger(
        database_url, monkeypatch
    )
    policy = simulation["execution_policy"]
    now = datetime.now(UTC)
    batch_id = uuid.uuid4().hex
    with store.engine.begin() as connection:
        connection.execute(
            insert(simulation_batches).values(
                id=batch_id,
                portfolio_id=simulation["id"],
                source_snapshot_id="f" * 64,
                target_payload_json={},
                execution_adapter="long_only",
                execution_contract_hash=simulation["execution_contract_hash"],
                daily_dataset=simulation["daily_dataset"],
                daily_dataset_identity_sha256=simulation[
                    "daily_dataset_identity_sha256"
                ],
                daily_dataset_lineage_id=simulation["daily_dataset_lineage_id"],
                execution_dataset=simulation["execution_dataset"],
                execution_dataset_identity_sha256=simulation[
                    "execution_dataset_identity_sha256"
                ],
                execution_dataset_lineage_id=simulation[
                    "execution_dataset_lineage_id"
                ],
                simulation_semantics_sha256=policy[
                    "simulation_semantics_sha256"
                ],
                signal_date=date(2026, 8, 28),
                trade_date=date(2026, 8, 29),
                status="queued",
                idempotency_key=f"cutover-block-{uuid.uuid4().hex}",
                created_by=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
                created_at=now,
            )
        )

    with pytest.raises(ValueError, match="queued batches"):
        store.prepare_allocation_source_replacement(
            str(simulation["id"]),
            old_allocation_id=old_id,
            new_allocation_id=new_id,
            actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
        )
    assert store.get(str(simulation["id"]))["status"] == "active"
    with store.engine.begin() as connection:
        connection.execute(
            update(simulation_batches)
            .where(simulation_batches.c.id == batch_id)
            .values(status="failed", finished_at=now)
        )
        connection.execute(
            insert(simulation_orders).values(
                id=uuid.uuid4().hex,
                batch_id=batch_id,
                portfolio_id=simulation["id"],
                instrument="SH600000",
                side="buy",
                position_side="long",
                target_weight=0.01,
                requested_quantity=100,
                filled_quantity=0,
                status="planned",
                requested_value=Decimal("1000"),
                filled_value=Decimal("0"),
                capacity_fill_ratio=0.0,
                expires_at=now,
                created_at=now,
            )
        )
    with pytest.raises(ValueError, match="working orders"):
        store.prepare_allocation_source_replacement(
            str(simulation["id"]),
            old_allocation_id=old_id,
            new_allocation_id=new_id,
            actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
        )
    assert store.get(str(simulation["id"]))["status"] == "active"
    with store.engine.connect() as connection:
        assert connection.scalar(
            select(strategy_allocations.c.status).where(
                strategy_allocations.c.id == old_id
            )
        ) == "active"
        assert connection.scalar(
            select(strategy_allocations.c.status).where(
                strategy_allocations.c.id == new_id
            )
        ) == "draft"


def test_cross_lineage_cutover_rolls_primary_dataset_without_resetting_ledger(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, simulation, old_id, new_id = _seed_primary_ledger(
        database_url, monkeypatch
    )
    replacement = _rolled_daily_dataset(
        name="snapshot-next-contract",
        identity="c" * 64,
        lineage="d" * 64,
    )
    before = store.get(str(simulation["id"]))
    with store.engine.begin() as connection:
        connection.execute(
            update(strategy_allocations)
            .where(strategy_allocations.c.id == new_id)
            .values(
                dataset=replacement["name"],
                analysis_json={
                    "allocation_dataset_lineage_id": "d" * 64,
                    "approval_simulation_evidence": {"sealed": new_id},
                },
            )
        )

    with pytest.raises(ValueError, match="requires a verified replacement"):
        store.prepare_allocation_source_replacement(
            str(simulation["id"]),
            old_allocation_id=old_id,
            new_allocation_id=new_id,
            actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
        )
    assert store.get(str(simulation["id"]))["status"] == "active"

    prepared = store.prepare_allocation_source_replacement(
        str(simulation["id"]),
        old_allocation_id=old_id,
        new_allocation_id=new_id,
        actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
        daily_dataset=replacement,
        execution_dataset=replacement,
    )
    assert prepared["status"] == "paused"
    with store.engine.begin() as connection:
        connection.execute(
            update(strategy_allocations)
            .where(strategy_allocations.c.id == old_id)
            .values(status="paused")
        )
        connection.execute(
            update(strategy_allocations)
            .where(strategy_allocations.c.id == new_id)
            .values(status="active", approved_at=datetime.now(UTC))
        )
    rebound = store.replace_allocation_source(
        str(simulation["id"]),
        old_allocation_id=old_id,
        new_allocation_id=new_id,
        actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
        daily_dataset=replacement,
        execution_dataset=replacement,
    )

    assert rebound["id"] == before["id"]
    assert rebound["cash"] == before["cash"]
    assert rebound["nav"] == before["nav"]
    assert rebound["daily_dataset"] == replacement["name"]
    assert rebound["execution_dataset"] == replacement["name"]
    assert rebound["daily_dataset_identity_sha256"] == "c" * 64
    assert rebound["daily_dataset_lineage_id"] == "d" * 64
    assert rebound["execution_dataset_lineage_id"] == "d" * 64


def test_allocation_aggregate_nav_never_overwrites_physical_primary_ledger(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, simulation, old_id, _new_id = _seed_primary_ledger(
        database_url, monkeypatch
    )
    before = store.get(str(simulation["id"]))
    now = datetime.now(UTC)
    with store.engine.begin() as connection:
        allocation = connection.execute(
            select(strategy_allocations).where(strategy_allocations.c.id == old_id)
        ).one()
        AllocationStore._sync_allocation_simulation_nav(
            connection,
            allocation=allocation,
            trade_date=date(2026, 8, 29),
            allocation_nav=Decimal("900000"),
            daily_return=-0.10,
            drawdown=-0.10,
            produced_by="member-paper-aggregate",
            now=now,
        )
    after = store.get(str(simulation["id"]))
    assert after["cash"] == before["cash"]
    assert after["nav"] == before["nav"]
    with store.engine.connect() as connection:
        assert connection.scalar(
            select(func.count())
            .select_from(simulation_nav)
            .where(simulation_nav.c.trade_date == date(2026, 8, 29))
        ) == 0
