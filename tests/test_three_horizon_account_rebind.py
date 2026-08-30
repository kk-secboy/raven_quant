from __future__ import annotations

import hashlib
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from governance_fixtures import (
    governed_etf_ready_evidence,
    write_governed_daily_qlib_dataset,
)
from sqlalchemy import delete, func, insert, select, update
from test_simulation_store import _daily_dataset

import quant_platform.three_horizon_account as three_horizon_account
from quant_data.database import (
    account_netting_plans,
    open_database,
    simulation_batches,
    simulation_events,
    simulation_nav,
    simulation_orders,
    simulation_portfolios,
    simulation_positions,
    strategy_allocation_artifacts,
    strategy_allocations,
)
from quant_data.execution_contract import DAILY_QLIB_FIELD_CONTRACT_VERSION
from quant_data.history_bounds import GOVERNED_DAILY_STOCK_SCOPE_VERSION
from quant_platform.account_netting import (
    AccountNettingStore,
    allocate_actual_sleeve_inventory,
)
from quant_platform.advice_service import AdviceService
from quant_platform.allocation_store import AllocationStore
from quant_platform.cost_model import COST_SCHEDULE_VERSION, CostModelConfig
from quant_platform.simulation_store import (
    SimulationStore,
    build_account_order_decision_event,
    build_allocation_source_rebind_event,
    build_settlement_calendar_evidence,
    validate_account_order_decision_event,
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
        "scope_version": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
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
def test_decision_only_event_is_deterministic_and_cannot_hide_orders() -> None:
    kwargs = {
        "portfolio_id": "primary-ledger",
        "account_netting_plan_id": "netting-plan",
        "target_version": "three-horizon-netting:" + "a" * 64,
        "actor": THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
        "signal_date": date(2026, 8, 28),
        "trade_date": date(2026, 8, 31),
        "actions": [
            {
                "instrument": "SH600001",
                "action": "NO_ACTION",
                "execution_state": "READY",
                "target_quantity": 0,
                "filled_position": 0,
                "projected_position": 0,
                "order_plan": [],
                "notes": ["new_buy_below_min_lot"],
            }
        ],
        "limit_prices": {"SH600001": 10.0},
    }

    first = build_account_order_decision_event(**kwargs)
    second = build_account_order_decision_event(**kwargs)

    assert first == second
    assert first["decision_outcome"] == "no_new_order_plan_operations"
    assert first["execution_order_plan_batch_created"] is False
    assert first["simulation_orders_created"] == 0
    assert first["prior_plan_continuation_allowed"] is False
    continuing = build_account_order_decision_event(
        **{
            **kwargs,
            "actions": [
                {
                    **kwargs["actions"][0],
                    "notes": ["no_valid_target_previous_retained"],
                }
            ],
        }
    )
    assert continuing["prior_plan_continuation_allowed"] is True
    assert validate_account_order_decision_event(
        first,
        portfolio_id="primary-ledger",
        account_netting_plan_id="netting-plan",
    ) == first
    with pytest.raises(ValueError, match="cannot contain order operations"):
        build_account_order_decision_event(
            **{
                **kwargs,
                "actions": [
                    {
                        **kwargs["actions"][0],
                        "action": "BUY",
                        "order_plan": [
                            {
                                "op": "new",
                                "side": "buy",
                                "quantity": 100,
                            }
                        ],
                    }
                ],
            }
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
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    initial_cash: int = 1_000_000,
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
        initial_cash=initial_cash,
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
                cash=Decimal(str(initial_cash)),
                market_value=Decimal("0"),
                nav=Decimal(str(initial_cash)),
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


def test_valued_decision_only_plan_is_the_durable_sleeve_continuity(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero-order A->B transfer must not be replayed on every later day."""

    store, simulation, allocation_id, _new_id = _seed_primary_ledger(
        database_url, monkeypatch
    )
    netting = AccountNettingStore(database_url)
    now = datetime.now(UTC)
    account_weight = 1_100.0 / 1_000_000.0

    def artifact(
        *, decision_day: date, inputs_day: date, suffix: str
    ) -> str:
        artifact_id = f"continuity-{suffix}-{uuid.uuid4().hex}"
        with store.engine.begin() as connection:
            connection.execute(
                insert(strategy_allocation_artifacts).values(
                    id=artifact_id,
                    allocation_id=allocation_id,
                    decision_date=decision_day,
                    inputs_as_of=inputs_day,
                    valid_until=date(2026, 9, 30),
                    member_weights_json={"A": 0.5, "B": 0.5},
                    analysis_json={},
                    artifact_hash=hashlib.sha256(artifact_id.encode()).hexdigest(),
                    created_at=now,
                )
            )
        return artifact_id

    def evidence(*, inputs_day: date) -> dict[str, Any]:
        return {
            "allocation_artifact_inputs_as_of": inputs_day.isoformat(),
            "primary_account": {
                "portfolio_id": str(simulation["id"]),
                "source_id": allocation_id,
                "nav": 1_000_000.0,
            },
        }

    initial_inputs = date(2026, 8, 27)
    initial = netting.create_plan(
        actor="three-horizon-netting",
        account_id=allocation_id,
        allocation_artifact_id=artifact(
            decision_day=date(2026, 8, 28),
            inputs_day=initial_inputs,
            suffix="initial",
        ),
        decision_date=date(2026, 8, 28),
        inputs_as_of=initial_inputs,
        policy_version="decision-continuity-initial-v1",
        member_budgets={"A": 0.5, "B": 0.5},
        member_targets={"A": {"SH600000": account_weight / 0.5}, "B": {}},
        total_capital=1_000_000,
        input_evidence=evidence(inputs_day=initial_inputs),
    )
    current = store.get(str(simulation["id"]))
    with store.engine.begin() as connection:
        connection.execute(
            insert(simulation_batches).values(
                id=f"initial-plan-{uuid.uuid4().hex}",
                portfolio_id=simulation["id"],
                source_snapshot_id=None,
                target_payload_json={
                    "order_plan": {
                        "target_version": (
                            f"three-horizon-netting:{initial['plan_hash']}"
                        ),
                        "account_netting_plan_id": initial["id"],
                        "actions": [],
                    }
                },
                execution_adapter="long_only",
                execution_contract_hash=current["execution_contract_hash"],
                daily_dataset=current["daily_dataset"],
                daily_dataset_identity_sha256=current[
                    "daily_dataset_identity_sha256"
                ],
                daily_dataset_lineage_id=current["daily_dataset_lineage_id"],
                execution_dataset=current["execution_dataset"],
                execution_dataset_identity_sha256=current[
                    "execution_dataset_identity_sha256"
                ],
                execution_dataset_lineage_id=current[
                    "execution_dataset_lineage_id"
                ],
                simulation_semantics_sha256=current["execution_policy"][
                    "simulation_semantics_sha256"
                ],
                signal_date=initial_inputs,
                trade_date=date(2026, 8, 28),
                status="succeeded",
                idempotency_key=f"initial-plan:{initial['id']}",
                account_netting_plan_id=initial["id"],
                created_by=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
                summary_json={"conservation": {"cash_difference": 0.0}},
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )

    transfer_inputs = date(2026, 8, 28)
    transfer = netting.create_plan(
        actor="three-horizon-netting",
        account_id=allocation_id,
        allocation_artifact_id=artifact(
            decision_day=date(2026, 8, 31),
            inputs_day=transfer_inputs,
            suffix="transfer",
        ),
        decision_date=date(2026, 8, 31),
        inputs_as_of=transfer_inputs,
        policy_version="decision-continuity-transfer-v1",
        member_budgets={"A": 0.5, "B": 0.5},
        member_targets={"A": {}, "B": {"SH600000": account_weight / 0.5}},
        member_current_account_weights={"A": {"SH600000": account_weight}},
        total_capital=1_000_000,
        input_evidence=evidence(inputs_day=transfer_inputs),
    )
    assert transfer["net_trades"] == {}

    def seal_valued_decision(plan: dict, *, signal_day: date, trade_day: date) -> str:
        event = build_account_order_decision_event(
            portfolio_id=str(simulation["id"]),
            account_netting_plan_id=str(plan["id"]),
            target_version=f"three-horizon-netting:{plan['plan_hash']}",
            actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
            signal_date=signal_day,
            trade_date=trade_day,
            actions=[
                {
                    "instrument": "SH600000",
                    "action": "HOLD",
                    "order_plan": [],
                }
            ],
        )
        batch_id = f"decision-valuation-{uuid.uuid4().hex}"
        with store.engine.begin() as connection:
            connection.execute(
                insert(simulation_events).values(
                    id=event["event_sha256"],
                    portfolio_id=simulation["id"],
                    batch_id=None,
                    trade_date=trade_day,
                    severity="info",
                    event_type="account_order_plan_decision_only",
                    instrument=None,
                    reason="account_netting_plan_created_no_new_order_operations",
                    details_json=event,
                    created_at=now,
                )
            )
            connection.execute(
                insert(simulation_batches).values(
                    id=batch_id,
                    portfolio_id=simulation["id"],
                    source_snapshot_id=event["event_sha256"],
                    target_payload_json={
                        "order_plan": {
                            "target_version": (
                                f"three-horizon-netting:{plan['plan_hash']}"
                            ),
                            "account_netting_plan_id": None,
                            "actions": [],
                            "decision_valuation_replay": True,
                            "decision_event_sha256": event["event_sha256"],
                        },
                        "decision_actions": event["actions"],
                        "governed_order_plan": {
                            "format_version": (
                                "simulation-account-decision-valuation-v1"
                            ),
                            "decision_event_sha256": event["event_sha256"],
                            "settlement_calendar_binding": {},
                        },
                    },
                    execution_adapter="long_only",
                    execution_contract_hash=current["execution_contract_hash"],
                    daily_dataset=current["daily_dataset"],
                    daily_dataset_identity_sha256=current[
                        "daily_dataset_identity_sha256"
                    ],
                    daily_dataset_lineage_id=current["daily_dataset_lineage_id"],
                    execution_dataset=current["execution_dataset"],
                    execution_dataset_identity_sha256=current[
                        "execution_dataset_identity_sha256"
                    ],
                    execution_dataset_lineage_id=current[
                        "execution_dataset_lineage_id"
                    ],
                    simulation_semantics_sha256=current["execution_policy"][
                        "simulation_semantics_sha256"
                    ],
                    signal_date=signal_day,
                    trade_date=trade_day,
                    status="succeeded",
                    idempotency_key=f"decision-valuation:{event['event_sha256']}",
                    account_netting_plan_id=None,
                    created_by=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
                    summary_json={"conservation": {"cash_difference": 0.0}},
                    created_at=now,
                    started_at=now,
                    finished_at=now,
                )
            )
            connection.execute(
                insert(simulation_nav).values(
                    portfolio_id=simulation["id"],
                    trade_date=trade_day,
                    cash=Decimal("998900"),
                    market_value=Decimal("1100"),
                    nav=Decimal("1000000"),
                    daily_return=0.0,
                    drawdown=0.0,
                    market_date=trade_day,
                    has_stale_prices=False,
                    status="healthy",
                    performance_certified=True,
                    nav_scope="aggregate_view",
                    produced_by=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
                    created_at=now,
                )
            )
        return batch_id

    transfer_batch_id = seal_valued_decision(
        transfer,
        signal_day=transfer_inputs,
        trade_day=date(2026, 8, 31),
    )
    with store.engine.connect() as connection:
        allocation = connection.execute(
            select(strategy_allocations).where(
                strategy_allocations.c.id == allocation_id
            )
        ).one()
        selected = netting._authoritative_prior_plan(
            connection,
            portfolio_id=str(simulation["id"]),
            current_allocation=allocation,
            decision_date=date(2026, 9, 1),
            inputs_as_of=date(2026, 8, 31),
        )
    assert selected is not None
    assert selected["kind"] == "decision_only"
    assert selected["plan_id"] == transfer["id"]
    assert selected["batch_id"] == transfer_batch_id

    # A decision event is not authoritative merely because it exists.  Until
    # its valuation succeeds, continuity stays on the last executed plan.
    with store.engine.begin() as connection:
        connection.execute(
            update(simulation_batches)
            .where(simulation_batches.c.id == transfer_batch_id)
            .values(status="failed", error="valuation failed")
        )
        allocation = connection.execute(
            select(strategy_allocations).where(
                strategy_allocations.c.id == allocation_id
            )
        ).one()
        failed_selection = netting._authoritative_prior_plan(
            connection,
            portfolio_id=str(simulation["id"]),
            current_allocation=allocation,
            decision_date=date(2026, 9, 1),
            inputs_as_of=date(2026, 8, 31),
        )
        connection.execute(
            update(simulation_batches)
            .where(simulation_batches.c.id == transfer_batch_id)
            .values(status="succeeded", error=None)
        )
    assert failed_selection is not None
    assert failed_selection["plan_id"] == initial["id"]

    # The event->plan pointer is insufficient by itself: changing a hashed
    # economic input makes the otherwise succeeded decision unusable.
    with store.engine.begin() as connection:
        stored_payload = dict(
            connection.scalar(
                select(account_netting_plans.c.plan_json).where(
                    account_netting_plans.c.id == transfer["id"]
                )
            )
        )
        tampered_payload = {
            **stored_payload,
            "member_targets": {
                **dict(stored_payload["member_targets"]),
                "B": {"SH600000": account_weight},
            },
        }
        connection.execute(
            update(account_netting_plans)
            .where(account_netting_plans.c.id == transfer["id"])
            .values(plan_json=tampered_payload)
        )
        allocation = connection.execute(
            select(strategy_allocations).where(
                strategy_allocations.c.id == allocation_id
            )
        ).one()
        with pytest.raises(ValueError, match="input hash is invalid"):
            netting._authoritative_prior_plan(
                connection,
                portfolio_id=str(simulation["id"]),
                current_allocation=allocation,
                decision_date=date(2026, 9, 1),
                inputs_as_of=date(2026, 8, 31),
            )
        connection.execute(
            update(account_netting_plans)
            .where(account_netting_plans.c.id == transfer["id"])
            .values(plan_json=stored_payload)
        )

    inventory, _inventory_evidence = allocate_actual_sleeve_inventory(
        actual_account_weights={"SH600000": account_weight},
        prior_plan=selected["plan"],
    )
    assert inventory == {"B": {"SH600000": pytest.approx(account_weight)}}
    next_inputs = date(2026, 8, 31)
    next_day = netting.create_plan(
        actor="three-horizon-netting",
        account_id=allocation_id,
        allocation_artifact_id=artifact(
            decision_day=date(2026, 9, 1),
            inputs_day=next_inputs,
            suffix="next-day",
        ),
        decision_date=date(2026, 9, 1),
        inputs_as_of=next_inputs,
        policy_version="decision-continuity-next-v1",
        member_budgets={"A": 0.5, "B": 0.5},
        member_targets={"A": {}, "B": {"SH600000": account_weight / 0.5}},
        member_current_account_weights=inventory,
        total_capital=1_000_000,
        input_evidence=evidence(inputs_day=next_inputs),
    )
    assert next_day["net_trades"] == {}
    seal_valued_decision(
        next_day,
        signal_day=next_inputs,
        trade_day=date(2026, 9, 1),
    )

    with store.engine.connect() as connection:
        allocation = connection.execute(
            select(strategy_allocations).where(
                strategy_allocations.c.id == allocation_id
            )
        ).one()
        selected_next = netting._authoritative_prior_plan(
            connection,
            portfolio_id=str(simulation["id"]),
            current_allocation=allocation,
            decision_date=date(2026, 9, 2),
            inputs_as_of=date(2026, 9, 1),
        )
    assert selected_next is not None
    assert selected_next["plan_id"] == next_day["id"]
    next_inventory, _ = allocate_actual_sleeve_inventory(
        actual_account_weights={"SH600000": account_weight},
        prior_plan=selected_next["plan"],
    )
    partial_inputs = date(2026, 9, 1)
    partial_kwargs = {
        "account_id": allocation_id,
        "allocation_artifact_id": artifact(
            decision_day=date(2026, 9, 2),
            inputs_day=partial_inputs,
            suffix="partial",
        ),
        "decision_date": date(2026, 9, 2),
        "inputs_as_of": partial_inputs,
        "policy_version": "decision-continuity-partial-v1",
        "member_budgets": {"A": 0.5, "B": 0.5},
        "member_targets": {
            "A": {},
            "B": {"SH600000": account_weight / 0.5 / 2.0},
        },
        "member_current_account_weights": next_inventory,
        "total_capital": 1_000_000,
        "input_evidence": evidence(inputs_day=partial_inputs),
    }
    partial = netting.create_plan(actor="three-horizon-netting", **partial_kwargs)
    retry = netting.create_plan(actor="three-horizon-netting", **partial_kwargs)
    assert retry["id"] == partial["id"]
    assert retry["idempotent_replay"] is True
    assert partial["net_trades"]["SH600000"]["delta_weight"] == pytest.approx(
        -account_weight / 2.0
    )
    assert partial["strategy_contributions"]["SH600000"]["members"]["B"][
        "net_contribution"
    ] == pytest.approx(-account_weight / 2.0)


def test_zero_order_account_decision_survives_restart_and_projects_to_advice(
    database_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    store, simulation, allocation_id, _new_id = _seed_primary_ledger(
        database_url, monkeypatch, initial_cash=100_000
    )
    artifact_id = uuid.uuid4().hex
    now = datetime.now(UTC)
    with store.engine.begin() as connection:
        connection.execute(
            delete(simulation_orders).where(
                simulation_orders.c.portfolio_id == simulation["id"]
            )
        )
        connection.execute(
            delete(simulation_positions).where(
                simulation_positions.c.portfolio_id == simulation["id"]
            )
        )
        # This regression is about the account decision/valuation boundary,
        # not dataset rollover. Keep the seeded ledger on its already-sealed
        # datasets so the minimal Qlib fixture only supplies the settlement
        # calendar used by the valuation replay.
        connection.execute(
            update(simulation_portfolios)
            .where(simulation_portfolios.c.id == simulation["id"])
            .values(
                daily_roll_policy="pinned",
                execution_roll_policy="pinned",
            )
        )
        connection.execute(
            insert(strategy_allocation_artifacts).values(
                id=artifact_id,
                allocation_id=allocation_id,
                decision_date=date(2026, 8, 31),
                inputs_as_of=date(2026, 8, 28),
                valid_until=date(2026, 9, 30),
                member_weights_json={},
                analysis_json={},
                artifact_hash=hashlib.sha256(artifact_id.encode()).hexdigest(),
                created_at=now,
            )
        )
    netting_store = AccountNettingStore(database_url)
    plan = netting_store.create_plan(
        actor="three-horizon-netting",
        account_id=allocation_id,
        allocation_artifact_id=artifact_id,
        decision_date=date(2026, 8, 31),
        inputs_as_of=date(2026, 8, 28),
        policy_version="decision-only-test-v1",
        member_budgets={"short": 0.20},
        member_targets={
            "short": {
                # On a CNY 100,000 account these account targets are CNY 1,000,
                # CNY 2,000 and CNY 2,000. They deliberately exercise a
                # high-priced main-board lot, STAR's 200-share minimum, and an
                # explicitly disabled ChiNext permission without re-ranking.
                "SH600001": 0.05,
                "SH688001": 0.10,
                "SZ300001": 0.10,
            }
        },
        total_capital=100_000,
        input_evidence={
            "primary_account": {
                "portfolio_id": simulation["id"],
                "source_id": allocation_id,
                "nav": 100_000,
            }
        },
    )
    service = object.__new__(ThreeHorizonAccountService)
    service.engine = store.engine
    service.simulations = store
    data_root = write_governed_daily_qlib_dataset(
        tmp_path / "decision-valuation-data",
        sessions=[
            date(2026, 8, 28),
            date(2026, 8, 31),
            date(2026, 9, 1),
        ],
    )
    service.settings = SimpleNamespace(data_root=data_root)
    service._latest_snapshot_payloads = lambda _allocation_id, **_kwargs: (
        {"SH600001": 20.0, "SH688001": 11.0, "SZ300001": 10.0},
        {},
    )
    investor_profile = {
        "market_permissions": {
            "main_board": True,
            "star_market": True,
            "chi_next": False,
            "beijing_exchange": False,
            "etf": False,
        }
    }
    first = service._materialize_order_plan(
        allocation={"id": allocation_id},
        simulation=store.get(str(simulation["id"])),
        plan=plan,
        investor_profile=investor_profile,
        now=datetime(2026, 8, 30, 8, tzinfo=UTC),
    )
    restarted = SimulationStore(database_url)
    service.simulations = restarted
    second = service._materialize_order_plan(
        allocation={"id": allocation_id},
        simulation=restarted.get(str(simulation["id"])),
        plan=plan,
        investor_profile=investor_profile,
        now=datetime(2026, 8, 30, 8, tzinfo=UTC),
    )

    assert first["kind"] == "decision_only"
    assert first["created"] is True
    assert second["created"] is False
    assert first["valuation_batch"]["status"] == "queued"
    assert first["valuation_batch"]["account_netting_plan_id"] is None
    assert second["valuation_batch"]["id"] == first["valuation_batch"]["id"]
    with pytest.raises(ValueError, match="bound to a decision-only record"):
        restarted.create_order_plan_batch(
            str(simulation["id"]),
            trade_date=date(2026, 8, 31),
            signal_date=date(2026, 8, 28),
            actions=[
                {
                    "instrument": "SH600001",
                    "action": "BUY",
                    "execution_state": "READY",
                    "target_quantity": 100,
                    "filled_position": 0,
                    "projected_position": 0,
                    "sellable_quantity": 0,
                    "order_plan": [
                        {
                            "op": "new",
                            "order_id": "",
                            "side": "buy",
                            "quantity": 100,
                            "reason": "cross-kind-regression",
                        }
                    ],
                    "blocked_reason": None,
                    "wait_reason": None,
                    "notes": [],
                }
            ],
            target_version=f"three-horizon-netting:{plan['plan_hash']}",
            actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
            account_netting_plan_id=str(plan["id"]),
            limit_prices={"SH600001": 10.0},
        )
    decision_actions = {
        item["instrument"]: item
        for item in first["decision_event"]["actions"]
    }
    assert decision_actions["SH600001"]["raw_target_quantity"] == 50
    assert decision_actions["SH600001"]["lot_adjusted_target_quantity"] == 0
    assert "new_buy_below_min_lot" in decision_actions["SH600001"]["notes"]
    assert decision_actions["SH688001"]["raw_target_quantity"] == 181
    assert decision_actions["SH688001"]["minimum_order_quantity"] == 200
    assert decision_actions["SH688001"]["lot_adjusted_target_quantity"] == 0
    assert "new_buy_below_min_lot" in decision_actions["SH688001"]["notes"]
    assert decision_actions["SZ300001"]["new_risk_blocked"] is True
    assert decision_actions["SZ300001"]["new_risk_blocked_reason"] == (
        "investor_permission_disabled:chi_next"
    )
    with restarted.engine.connect() as connection:
        assert connection.scalar(
            select(func.count())
            .select_from(simulation_batches)
            .where(simulation_batches.c.account_netting_plan_id == plan["id"])
        ) == 0
        assert connection.scalar(
            select(func.count())
            .select_from(simulation_batches)
            .where(
                simulation_batches.c.portfolio_id == simulation["id"],
                simulation_batches.c.source_snapshot_id
                == first["decision_event"]["event_sha256"],
                simulation_batches.c.account_netting_plan_id.is_(None),
            )
        ) == 1
        assert connection.scalar(
            select(func.count())
            .select_from(simulation_orders)
            .where(simulation_orders.c.portfolio_id == simulation["id"])
        ) == 0
        assert connection.scalar(
            select(func.count())
            .select_from(simulation_events)
            .where(
                simulation_events.c.portfolio_id == simulation["id"],
                simulation_events.c.event_type
                == "account_order_plan_decision_only",
            )
        ) == 1
        assert connection.scalar(
            select(func.count())
            .select_from(account_netting_plans)
            .where(account_netting_plans.c.id == plan["id"])
        ) == 1

    advice = object.__new__(AdviceService)
    advice.engine = restarted.engine
    advice.simulations = restarted
    advice.data_root = None
    projection = advice._unified_execution_facts(
        str(plan["id"]),
        account_id=allocation_id,
        portfolio_id=str(simulation["id"]),
    )

    assert projection["status"] == "ready"
    assert projection["batch_id"] is None
    assert projection["decision_event_id"] == first["decision_event"]["event_sha256"]
    assert projection["items"]["SH600001"]["trade_quantity"] == 0
    assert projection["items"]["SH600001"]["action"] == "NO_ACTION"
    assert projection["items"]["SH600001"]["quantity_source"] == (
        "unified_account_decision_only"
    )
    assert projection["items"]["SH600001"]["cannot_buy_reasons"] == [
        "模拟本金不足以买入最小整手，资金保留为现金"
    ]
    assert projection["items"]["SH688001"]["action"] == "NO_ACTION"
    assert projection["items"]["SH688001"]["cannot_buy_reasons"] == [
        "模拟本金不足以买入最小整手，资金保留为现金"
    ]
    assert projection["items"]["SZ300001"]["action"] == "NO_ACTION"
    assert projection["items"]["SZ300001"]["cannot_buy_reasons"] == [
        "你尚未确认创业板交易权限，本次不新增买入"
    ]

    valuation_batch = first["valuation_batch"]
    settlement_binding = restarted.execution_manifest(
        str(valuation_batch["id"])
    )["settlement_calendar_binding"]
    next_trade_date = date.fromisoformat(settlement_binding["next_trade_date"])
    completed_valuation = restarted.process_batch(
        str(valuation_batch["id"]),
        minute_bars=pd.DataFrame(
            columns=[
                "datetime",
                "instrument",
                "close",
                "vwap",
                "volume",
                "paused",
                "up_limit",
                "down_limit",
            ]
        ),
        closing_prices={},
        execution_evidence={
            "batch_id": valuation_batch["id"],
            "dataset_identity_sha256": valuation_batch[
                "execution_dataset_identity_sha256"
            ],
            "dataset_lineage_id": valuation_batch[
                "execution_dataset_lineage_id"
            ],
            "execution_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
            "execution_contract_hash": simulation["execution_contract_hash"],
            "simulation_semantics_sha256": valuation_batch[
                "simulation_semantics_sha256"
            ],
            "next_trade_date": next_trade_date.isoformat(),
            "settlement_calendar_evidence": build_settlement_calendar_evidence(
                trade_date=date(2026, 8, 31),
                next_trade_date=next_trade_date,
                dataset_identity_sha256=valuation_batch[
                    "execution_dataset_identity_sha256"
                ],
                dataset_lineage_id=valuation_batch[
                    "execution_dataset_lineage_id"
                ],
                calendar_file_sha256=settlement_binding[
                    "calendar_file_sha256"
                ],
            ),
        },
    )
    assert completed_valuation["status"] == "succeeded"
    assert restarted.rows(str(simulation["id"]), "orders") == []
    nav_days = restarted.rows(str(simulation["id"]), "nav")
    assert [item["trade_date"] for item in nav_days].count(date(2026, 8, 31)) == 1
    assert next(
        item for item in nav_days if item["trade_date"] == date(2026, 8, 31)
    )["nav"] == pytest.approx(100_000)
    post_settlement_restart = SimulationStore(database_url)
    service.simulations = post_settlement_restart
    third = service._materialize_order_plan(
        allocation={"id": allocation_id},
        simulation=post_settlement_restart.get(str(simulation["id"])),
        plan=plan,
        investor_profile=investor_profile,
        now=datetime(2026, 8, 30, 8, tzinfo=UTC),
    )
    assert third["created"] is False
    assert third["valuation_batch"]["id"] == valuation_batch["id"]
    assert third["valuation_batch"]["status"] == "succeeded"
    assert post_settlement_restart.rows(str(simulation["id"]), "orders") == []
    assert [
        item["trade_date"]
        for item in post_settlement_restart.rows(str(simulation["id"]), "nav")
    ].count(date(2026, 8, 31)) == 1

    execution_plan = netting_store.create_plan(
        actor="three-horizon-netting",
        account_id=allocation_id,
        allocation_artifact_id=artifact_id,
        decision_date=date(2026, 8, 31),
        inputs_as_of=date(2026, 8, 28),
        policy_version="decision-only-test-v1",
        tranche_index=1,
        member_budgets={"short": 0.20},
        member_targets={"short": {}},
        total_capital=100_000,
        input_evidence={
            "primary_account": {
                "portfolio_id": simulation["id"],
                "source_id": allocation_id,
                "nav": 100_000,
            }
        },
    )
    current = restarted.get(str(simulation["id"]))
    execution_batch_id = uuid.uuid4().hex
    with restarted.engine.begin() as connection:
        connection.execute(
            insert(simulation_batches).values(
                id=execution_batch_id,
                portfolio_id=simulation["id"],
                source_snapshot_id=None,
                target_payload_json={"order_plan": {"actions": []}},
                execution_adapter="long_only",
                execution_contract_hash=current["execution_contract_hash"],
                daily_dataset=current["daily_dataset"],
                daily_dataset_identity_sha256=current[
                    "daily_dataset_identity_sha256"
                ],
                daily_dataset_lineage_id=current["daily_dataset_lineage_id"],
                execution_dataset=current["execution_dataset"],
                execution_dataset_identity_sha256=current[
                    "execution_dataset_identity_sha256"
                ],
                execution_dataset_lineage_id=current[
                    "execution_dataset_lineage_id"
                ],
                simulation_semantics_sha256=current["execution_policy"][
                    "simulation_semantics_sha256"
                ],
                signal_date=date(2026, 8, 28),
                trade_date=date(2026, 8, 31),
                status="queued",
                idempotency_key=f"cross-kind-{execution_batch_id}",
                account_netting_plan_id=execution_plan["id"],
                created_by=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
                created_at=now,
            )
        )
    with pytest.raises(ValueError, match="bound to an execution batch"):
        restarted.record_account_order_plan_decision(
            str(simulation["id"]),
            trade_date=date(2026, 8, 31),
            signal_date=date(2026, 8, 28),
            actions=[],
            target_version=(
                f"three-horizon-netting:{execution_plan['plan_hash']}"
            ),
            actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
            account_netting_plan_id=str(execution_plan["id"]),
        )


def test_decision_valuation_continues_a_previously_frozen_open_order(
    database_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    store, simulation, allocation_id, _new_id = _seed_primary_ledger(
        database_url, monkeypatch
    )
    with store.engine.begin() as connection:
        connection.execute(
            delete(simulation_orders).where(
                simulation_orders.c.portfolio_id == simulation["id"]
            )
        )
        connection.execute(
            delete(simulation_positions).where(
                simulation_positions.c.portfolio_id == simulation["id"]
            )
        )
        connection.execute(
            update(simulation_portfolios)
            .where(simulation_portfolios.c.id == simulation["id"])
            .values(
                daily_roll_policy="pinned",
                execution_roll_policy="pinned",
            )
        )

    data_root = write_governed_daily_qlib_dataset(
        tmp_path / "decision-continuation-data",
        sessions=[
            date(2026, 8, 28),
            date(2026, 8, 31),
            date(2026, 9, 1),
            date(2026, 9, 2),
        ],
    )

    def bars(day: date, *, volume: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "datetime": datetime(day.year, day.month, day.day, 9, 30),
                    "instrument": "SH600001",
                    "close": 10.0,
                    "vwap": 10.0,
                    "volume": volume,
                    "paused": 0,
                    "up_limit": 11.0,
                    "down_limit": 9.0,
                }
            ]
        )

    def execution_evidence(batch: dict[str, Any]) -> dict[str, Any]:
        settlement = store.execution_manifest(str(batch["id"]))[
            "settlement_calendar_binding"
        ]
        next_date = date.fromisoformat(settlement["next_trade_date"])
        return {
            "batch_id": batch["id"],
            "dataset_identity_sha256": batch[
                "execution_dataset_identity_sha256"
            ],
            "dataset_lineage_id": batch["execution_dataset_lineage_id"],
            "execution_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
            "execution_contract_hash": simulation["execution_contract_hash"],
            "simulation_semantics_sha256": batch[
                "simulation_semantics_sha256"
            ],
            "next_trade_date": next_date.isoformat(),
            "settlement_calendar_evidence": build_settlement_calendar_evidence(
                trade_date=date.fromisoformat(str(batch["trade_date"])),
                next_trade_date=next_date,
                dataset_identity_sha256=batch[
                    "execution_dataset_identity_sha256"
                ],
                dataset_lineage_id=batch["execution_dataset_lineage_id"],
                calendar_file_sha256=settlement["calendar_file_sha256"],
            ),
        }

    working_batch, created = store.create_order_plan_batch(
        str(simulation["id"]),
        trade_date=date(2026, 8, 31),
        signal_date=date(2026, 8, 28),
        actions=[
            {
                "instrument": "SH600001",
                "action": "BUY",
                "execution_state": "READY",
                "target_quantity": 100,
                "filled_position": 0,
                "projected_position": 0,
                "sellable_quantity": 0,
                "order_plan": [
                    {
                        "op": "new",
                        "order_id": "",
                        "side": "buy",
                        "quantity": 100,
                        "reason": "prior-authorized-target",
                    }
                ],
                "blocked_reason": None,
                "wait_reason": None,
                "notes": [],
            }
        ],
        target_version="prior-authorized-target-v1",
        actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
        limit_prices={"SH600001": 10.0},
        not_before=datetime(2026, 8, 31, 9, 30, tzinfo=ZoneInfo("Asia/Shanghai")),
        not_after=datetime(2026, 9, 2, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        data_root=data_root,
    )
    assert created is True
    # The preceding frozen plan is intentionally left unprocessed here. Mark
    # its reserved order working to model a carry-over order from an earlier
    # execution attempt; the new decision must neither duplicate nor cancel it.
    with store.engine.begin() as connection:
        connection.execute(
            update(simulation_orders)
            .where(simulation_orders.c.batch_id == working_batch["id"])
            .values(status="open")
        )
    existing_orders = store.rows(str(simulation["id"]), "orders")
    assert len(existing_orders) == 1
    assert existing_orders[0]["status"] == "open"

    artifact_id = uuid.uuid4().hex
    now = datetime.now(UTC)
    with store.engine.begin() as connection:
        connection.execute(
            insert(strategy_allocation_artifacts).values(
                id=artifact_id,
                allocation_id=allocation_id,
                decision_date=date(2026, 9, 1),
                inputs_as_of=date(2026, 8, 31),
                valid_until=date(2026, 9, 30),
                member_weights_json={},
                analysis_json={},
                artifact_hash=hashlib.sha256(artifact_id.encode()).hexdigest(),
                created_at=now,
            )
        )
    netting_store = AccountNettingStore(database_url)
    blocked_plan = netting_store.create_plan(
        actor="three-horizon-netting",
        account_id=allocation_id,
        allocation_artifact_id=artifact_id,
        decision_date=date(2026, 9, 1),
        inputs_as_of=date(2026, 8, 31),
        policy_version="decision-continuation-test-v1",
        member_budgets={"short": 0.20},
        member_targets={"short": {}},
        total_capital=1_000_000,
        input_evidence={
            "primary_account": {
                "portfolio_id": simulation["id"],
                "source_id": allocation_id,
                "nav": 1_000_000,
            }
        },
    )
    blocked_decision, _ = store.record_account_order_plan_decision(
        str(simulation["id"]),
        trade_date=date(2026, 9, 1),
        signal_date=date(2026, 8, 31),
        actions=[
            {
                "instrument": "SH600001",
                "action": "NO_ACTION",
                "execution_state": "READY",
                "target_quantity": 0,
                "filled_position": 0,
                "projected_position": 0,
                "sellable_quantity": 0,
                "order_plan": [],
                "blocked_reason": None,
                "wait_reason": None,
                "notes": [],
            }
        ],
        target_version=f"three-horizon-netting:{blocked_plan['plan_hash']}",
        actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
        account_netting_plan_id=str(blocked_plan["id"]),
    )
    blocked_batch, _ = store.create_account_decision_valuation_batch(
        str(simulation["id"]),
        decision_event_sha256=str(blocked_decision["event_sha256"]),
        actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
        data_root=data_root,
    )
    tampered_decision = {
        **blocked_decision,
        "actions": [
            {
                **blocked_decision["actions"][0],
                "notes": ["tampered-after-batch-creation"],
            }
        ],
    }
    with store.engine.begin() as connection:
        connection.execute(
            update(simulation_events)
            .where(simulation_events.c.id == blocked_decision["event_sha256"])
            .values(details_json=tampered_decision)
        )
    with pytest.raises(ValueError, match="evidence seal is invalid"):
        store.process_batch(
            str(blocked_batch["id"]),
            minute_bars=bars(date(2026, 9, 1), volume=1_000_000),
            closing_prices={
                "SH600001": {"price": 10.0, "market_date": "2026-09-01"}
            },
            execution_evidence=execution_evidence(blocked_batch),
        )
    with store.engine.begin() as connection:
        connection.execute(
            update(simulation_events)
            .where(simulation_events.c.id == blocked_decision["event_sha256"])
            .values(details_json=blocked_decision)
        )
    with pytest.raises(ValueError, match="forbids unapproved working-order"):
        store.process_batch(
            str(blocked_batch["id"]),
            minute_bars=bars(date(2026, 9, 1), volume=1_000_000),
            closing_prices={
                "SH600001": {"price": 10.0, "market_date": "2026-09-01"}
            },
            execution_evidence=execution_evidence(blocked_batch),
        )
    assert store.get_batch(str(blocked_batch["id"]))["status"] == "queued"
    assert store.rows(str(simulation["id"]), "orders")[0]["status"] == "open"

    plan = netting_store.create_plan(
        actor="three-horizon-netting",
        account_id=allocation_id,
        allocation_artifact_id=artifact_id,
        decision_date=date(2026, 9, 1),
        inputs_as_of=date(2026, 8, 31),
        policy_version="decision-continuation-test-v1",
        tranche_index=1,
        member_budgets={"short": 0.20},
        member_targets={"short": {}},
        total_capital=1_000_000,
        input_evidence={
            "primary_account": {
                "portfolio_id": simulation["id"],
                "source_id": allocation_id,
                "nav": 1_000_000,
            }
        },
    )
    decision, decision_created = store.record_account_order_plan_decision(
        str(simulation["id"]),
        trade_date=date(2026, 9, 1),
        signal_date=date(2026, 8, 31),
        actions=[
            {
                "instrument": "SH600001",
                "action": "NO_ACTION",
                "execution_state": "BLOCKED",
                "target_quantity": None,
                "filled_position": 0,
                "projected_position": 100,
                "sellable_quantity": 0,
                "order_plan": [],
                "blocked_reason": "fresh_reference_price_unavailable",
                "wait_reason": None,
                "notes": ["no_valid_target_previous_retained"],
            }
        ],
        target_version=f"three-horizon-netting:{plan['plan_hash']}",
        actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
        account_netting_plan_id=str(plan["id"]),
    )
    assert decision_created is True
    assert decision["prior_plan_continuation_allowed"] is True
    order_count_before = len(store.rows(str(simulation["id"]), "orders"))
    valuation_batch, valuation_created = store.create_account_decision_valuation_batch(
        str(simulation["id"]),
        decision_event_sha256=str(decision["event_sha256"]),
        actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
        data_root=data_root,
    )
    assert valuation_created is True
    store.process_batch(
        str(valuation_batch["id"]),
        minute_bars=bars(date(2026, 9, 1), volume=1_000_000),
        closing_prices={
            "SH600001": {"price": 10.0, "market_date": "2026-09-01"}
        },
        execution_evidence=execution_evidence(valuation_batch),
    )

    orders = store.rows(str(simulation["id"]), "orders")
    assert len(orders) == order_count_before == 1
    assert orders[0]["status"] == "filled"
    assert len(store.rows(str(simulation["id"]), "fills")) == 1
    nav_dates = [
        item["trade_date"] for item in store.rows(str(simulation["id"]), "nav")
    ]
    assert nav_dates.count(date(2026, 9, 1)) == 1


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
