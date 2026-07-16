from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
from sqlalchemy import delete, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    open_database,
    recommendation_holdings,
    recommendation_portfolios,
    recommendation_snapshots,
    row_dict,
    simulation_batches,
    simulation_cash_flows,
    simulation_events,
    simulation_fills,
    simulation_nav,
    simulation_orders,
    simulation_portfolios,
    simulation_positions,
)
from quant_data.execution_contract import (
    require_daily_qlib_contract,
    require_minute_execution_contract,
)

from .cost_model import COST_SCHEDULE_VERSION, CostModelConfig
from .execution_algorithms import normalize_execution_policy
from .simulation_engine import SIMULATION_ENGINE_VERSION, execute_simulation_day


def _now() -> datetime:
    return datetime.now(UTC)


class SimulationStore:
    """Transactional T+1 simulation ledger driven only by recommendation targets."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def create(
        self,
        *,
        name: str,
        recommendation_portfolio_id: str,
        daily_dataset: dict[str, Any],
        execution_dataset: dict[str, Any],
        initial_cash: float,
        execution_policy: dict[str, Any],
        cost_schedule_version: str,
        actor: str,
    ) -> dict[str, Any]:
        if initial_cash < 100_000:
            raise ValueError("simulation initial cash must be at least 100000")
        if cost_schedule_version != COST_SCHEDULE_VERSION:
            raise ValueError("simulation cost schedule version is unavailable")
        policy = normalize_execution_policy(execution_policy)
        daily_provenance = dict(daily_dataset.get("provenance") or {})
        execution_provenance = dict(execution_dataset.get("provenance") or {})
        require_daily_qlib_contract(daily_provenance)
        require_minute_execution_contract(
            execution_provenance, frequency="5min", simulation_eligible=True
        )
        daily_source = str(daily_provenance.get("source_lineage_id") or "")
        execution_source = str(execution_provenance.get("source_lineage_id") or "")
        if len(daily_source) != 64 or daily_source != execution_source:
            raise ValueError("daily and 5-minute datasets must share one verified source lineage")
        required_hashes = {
            "daily identity": daily_provenance.get("dataset_identity_sha256"),
            "daily lineage": daily_provenance.get("dataset_lineage_id"),
            "execution identity": execution_provenance.get("dataset_identity_sha256"),
            "execution lineage": execution_provenance.get("dataset_lineage_id"),
        }
        if any(len(str(value or "")) != 64 for value in required_hashes.values()):
            raise ValueError("simulation datasets require immutable identities and lineages")
        with self.engine.connect() as connection:
            recommendation = connection.execute(
                select(recommendation_portfolios).where(
                    recommendation_portfolios.c.id == recommendation_portfolio_id
                )
            ).first()
        if recommendation is None:
            raise KeyError(recommendation_portfolio_id)
        if recommendation.status != "active":
            raise ValueError("simulation requires an active recommendation portfolio")
        if str(recommendation.dataset) != str(daily_dataset.get("name") or ""):
            raise ValueError("simulation daily dataset must match the recommendation portfolio")
        portfolio_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(simulation_portfolios).values(
                        id=portfolio_id,
                        name=name.strip(),
                        recommendation_portfolio_id=recommendation_portfolio_id,
                        status="paused",
                        base_currency="CNY",
                        initial_cash=Decimal(str(initial_cash)),
                        cash=Decimal(str(initial_cash)),
                        nav=Decimal(str(initial_cash)),
                        high_water_mark=Decimal(str(initial_cash)),
                        execution_algorithm=policy["execution_algorithm"],
                        execution_dataset=str(execution_dataset["name"]),
                        daily_dataset=str(daily_dataset["name"]),
                        daily_dataset_identity_sha256=daily_provenance[
                            "dataset_identity_sha256"
                        ],
                        daily_dataset_lineage_id=daily_provenance["dataset_lineage_id"],
                        daily_field_contract_version=daily_provenance["field_contract_version"],
                        execution_dataset_identity_sha256=execution_provenance[
                            "dataset_identity_sha256"
                        ],
                        execution_dataset_lineage_id=execution_provenance["dataset_lineage_id"],
                        execution_field_contract_version=execution_provenance[
                            "execution_contract_version"
                        ],
                        execution_engine_version=SIMULATION_ENGINE_VERSION,
                        cost_schedule_version=cost_schedule_version,
                        execution_policy_json=policy,
                        created_by=actor.strip(),
                        created_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    insert(simulation_cash_flows).values(
                        id=uuid.uuid4().hex,
                        portfolio_id=portfolio_id,
                        batch_id=None,
                        trade_date=now.date(),
                        flow_type="initial_deposit",
                        amount=Decimal(str(initial_cash)),
                        balance_after=Decimal(str(initial_cash)),
                        created_at=now,
                    )
                )
        except IntegrityError as exc:
            raise ValueError(
                "simulation name or recommendation portfolio is already in use"
            ) from exc
        return self.get(portfolio_id)

    def set_status(self, portfolio_id: str, status: str) -> dict[str, Any]:
        if status not in {"active", "paused"}:
            raise ValueError("simulation status must be active or paused")
        with self.engine.begin() as connection:
            result = connection.execute(
                update(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .values(status=status, updated_at=_now())
            )
            if not result.rowcount:
                raise KeyError(portfolio_id)
        return self.get(portfolio_id)

    def create_batch_for_snapshot(self, snapshot_id: str) -> tuple[dict[str, Any] | None, bool]:
        now = _now()
        with self.engine.begin() as connection:
            snapshot = connection.execute(
                select(recommendation_snapshots).where(
                    recommendation_snapshots.c.id == snapshot_id
                )
            ).first()
            if snapshot is None:
                raise KeyError(snapshot_id)
            if snapshot.status != "succeeded" or snapshot.effective_date is None:
                raise ValueError("simulation batch requires a successful recommendation snapshot")
            portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.recommendation_portfolio_id
                    == snapshot.portfolio_id
                )
            ).first()
            if portfolio is None:
                return None, False
            if portfolio.status != "active":
                return None, False
            if (
                str(snapshot.dataset) != str(portfolio.daily_dataset)
                or str(snapshot.dataset_identity_sha256)
                != str(portfolio.daily_dataset_identity_sha256)
            ):
                raise ValueError("recommendation snapshot lineage does not match simulation")
            batch_id = uuid.uuid4().hex
            inserted_id = connection.scalar(
                pg_insert(simulation_batches)
                .values(
                        id=batch_id,
                        portfolio_id=portfolio.id,
                        recommendation_snapshot_id=snapshot_id,
                        signal_date=snapshot.as_of_date,
                        trade_date=snapshot.effective_date,
                        status="queued",
                        idempotency_key=f"simulation:{snapshot_id}",
                        created_at=now,
                    )
                .on_conflict_do_nothing(
                    index_elements=[simulation_batches.c.recommendation_snapshot_id]
                )
                .returning(simulation_batches.c.id)
            )
            if inserted_id is None:
                existing = connection.execute(
                    select(simulation_batches).where(
                        simulation_batches.c.recommendation_snapshot_id == snapshot_id
                    )
                ).one()
                return self._batch_dict(existing), False
        return self.get_batch(batch_id), True

    def process_batch(
        self,
        batch_id: str,
        *,
        minute_bars: pd.DataFrame,
        closing_prices: dict[str, dict[str, Any]],
        execution_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute, book and value a simulation day in one database transaction."""

        now = _now()
        with self.engine.begin() as connection:
            batch = connection.execute(
                select(simulation_batches)
                .where(simulation_batches.c.id == batch_id)
                .with_for_update()
            ).first()
            if batch is None:
                raise KeyError(batch_id)
            if batch.status == "succeeded":
                return self._batch_dict(batch)
            if batch.status != "queued":
                raise ValueError("only queued simulation batches may execute")
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == batch.portfolio_id)
                .with_for_update()
            ).one()
            if portfolio.status != "active":
                raise ValueError("simulation portfolio is not active")
            expected_execution = {
                "dataset_identity_sha256": str(
                    portfolio.execution_dataset_identity_sha256
                ),
                "dataset_lineage_id": str(portfolio.execution_dataset_lineage_id),
                "execution_contract_version": str(
                    portfolio.execution_field_contract_version
                ),
            }
            mismatches = [
                field
                for field, expected in expected_execution.items()
                if str(execution_evidence.get(field) or "") != expected
            ]
            if mismatches:
                raise ValueError(
                    "simulation execution evidence does not match the bound dataset: "
                    + ", ".join(mismatches)
                )
            if str(execution_evidence.get("batch_id") or "") != str(batch.id):
                raise ValueError("simulation execution evidence does not match the batch")
            snapshot = connection.execute(
                select(recommendation_snapshots).where(
                    recommendation_snapshots.c.id == batch.recommendation_snapshot_id
                )
            ).one()
            if snapshot.status != "succeeded":
                raise ValueError("simulation recommendation snapshot is no longer valid")
            target_weights = {
                str(item.instrument): float(item.weight)
                for item in connection.execute(
                    select(recommendation_holdings).where(
                        recommendation_holdings.c.snapshot_id == snapshot.id
                    )
                )
            }
            position_state = {
                str(item.instrument): row_dict(item)
                for item in connection.execute(
                    select(simulation_positions).where(
                        simulation_positions.c.portfolio_id == portfolio.id
                    )
                )
            }
            result = execute_simulation_day(
                trade_date=batch.trade_date,
                cash=float(portfolio.cash),
                prior_nav=float(portfolio.nav),
                high_water_mark=float(portfolio.high_water_mark),
                positions=position_state,
                target_weights=target_weights,
                minute_bars=minute_bars,
                closing_prices=closing_prices,
                cost_model=CostModelConfig(version=str(portfolio.cost_schedule_version)),
                execution_policy=dict(portfolio.execution_policy_json),
            )
            self._persist_result(connection, batch, portfolio, result, now)
        return self.get_batch(batch_id)

    def mark_batch_failed(self, batch_id: str, error: str) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(simulation_batches)
                .where(
                    simulation_batches.c.id == batch_id,
                    simulation_batches.c.status == "queued",
                )
                .values(status="failed", error=error, finished_at=_now())
            )
            if not result.rowcount:
                existing = connection.execute(
                    select(simulation_batches.c.id).where(
                        simulation_batches.c.id == batch_id
                    )
                ).first()
                if existing is None:
                    raise KeyError(batch_id)

    def _persist_result(
        self, connection: Any, batch: Any, portfolio: Any, result: dict[str, Any], now: datetime
    ) -> None:
        order_ids: dict[tuple[str, str], str] = {}
        for order in result["orders"]:
            order_id = uuid.uuid4().hex
            key = (str(order["instrument"]), str(order["side"]))
            order_ids[key] = order_id
            expires_at = order.get("expires_at") or datetime.combine(
                batch.trade_date, datetime.max.time(), tzinfo=UTC
            )
            connection.execute(
                insert(simulation_orders).values(
                    id=order_id,
                    batch_id=batch.id,
                    instrument=order["instrument"],
                    side=order["side"],
                    target_weight=float(order["target_weight"]),
                    requested_quantity=int(order["requested_quantity"]),
                    filled_quantity=int(order["filled_quantity"]),
                    status=order["status"],
                    reject_reason=order.get("reject_reason"),
                    requested_value=Decimal(str(order["requested_value"])),
                    filled_value=Decimal(str(order["filled_value"])),
                    capacity_fill_ratio=float(order["capacity_fill_ratio"]),
                    expires_at=expires_at,
                    created_at=now,
                )
            )
        fill_ids: list[str] = []
        for fill in result["fills"]:
            fill_id = uuid.uuid4().hex
            fill_ids.append(fill_id)
            connection.execute(
                insert(simulation_fills).values(
                    id=fill_id,
                    order_id=order_ids[(str(fill["instrument"]), str(fill["side"]))],
                    batch_id=batch.id,
                    instrument=fill["instrument"],
                    side=fill["side"],
                    executed_at=fill["executed_at"],
                    quantity=int(fill["quantity"]),
                    price=Decimal(str(fill["price"])),
                    gross_value=Decimal(str(fill["gross_value"])),
                    fee=Decimal(str(fill["fee"])),
                    cost_breakdown_json=fill["cost_breakdown"],
                    minute_volume=int(fill["minute_volume"]),
                    capacity_quantity=int(fill["capacity_quantity"]),
                )
            )
        for index, flow in enumerate(result["cash_flows"]):
            connection.execute(
                insert(simulation_cash_flows).values(
                    id=uuid.uuid4().hex,
                    portfolio_id=portfolio.id,
                    batch_id=batch.id,
                    trade_date=batch.trade_date,
                    flow_type=flow["flow_type"],
                    amount=Decimal(str(flow["amount"])),
                    balance_after=Decimal(str(flow["balance_after"])),
                    reference_id=fill_ids[index] if index < len(fill_ids) else None,
                    created_at=now,
                )
            )
        connection.execute(
            delete(simulation_positions).where(
                simulation_positions.c.portfolio_id == portfolio.id
            )
        )
        for instrument, position in result["positions"].items():
            connection.execute(
                insert(simulation_positions).values(
                    portfolio_id=portfolio.id,
                    instrument=instrument,
                    quantity=int(position["quantity"]),
                    available_quantity=int(position["available_quantity"]),
                    average_cost=Decimal(str(position["average_cost"])),
                    last_trade_date=position.get("last_trade_date"),
                    market_price=(
                        Decimal(str(position["market_price"]))
                        if position.get("market_price") is not None
                        else None
                    ),
                    market_date=position.get("market_date"),
                    stale=bool(position.get("stale", True)),
                    market_value=Decimal(str(position.get("market_value", 0.0))),
                    updated_at=now,
                )
            )
        nav = result["nav_row"]
        connection.execute(
            insert(simulation_nav).values(
                portfolio_id=portfolio.id,
                trade_date=batch.trade_date,
                cash=Decimal(str(nav["cash"])),
                market_value=Decimal(str(nav["market_value"])),
                nav=Decimal(str(nav["nav"])),
                daily_return=float(nav["daily_return"]),
                drawdown=float(nav["drawdown"]),
                market_date=nav["market_date"],
                has_stale_prices=bool(nav["has_stale_prices"]),
                status=nav["status"],
                performance_certified=bool(nav["performance_certified"]),
                created_at=now,
            )
        )
        for event in result["events"]:
            connection.execute(
                insert(simulation_events).values(
                    id=uuid.uuid4().hex,
                    portfolio_id=portfolio.id,
                    batch_id=batch.id,
                    trade_date=batch.trade_date,
                    severity=event["severity"],
                    event_type=event["event_type"],
                    instrument=event.get("instrument"),
                    reason=event["reason"],
                    details_json=event.get("details") or {},
                    created_at=now,
                )
            )
        summary = {
            "engine_version": result["engine_version"],
            "orders": len(result["orders"]),
            "fills": len(result["fills"]),
            "rejections": sum(order["status"] != "filled" for order in result["orders"]),
            "cash": result["cash"],
            "nav": result["nav"],
            "conservation": result["conservation"],
        }
        connection.execute(
            update(simulation_portfolios)
            .where(simulation_portfolios.c.id == portfolio.id)
            .values(
                cash=Decimal(str(result["cash"])),
                nav=Decimal(str(result["nav"])),
                high_water_mark=Decimal(str(result["high_water_mark"])),
                updated_at=now,
            )
        )
        connection.execute(
            update(simulation_batches)
            .where(simulation_batches.c.id == batch.id)
            .values(
                status="succeeded",
                summary_json=summary,
                started_at=now,
                finished_at=now,
                error=None,
            )
        )

    def get(self, portfolio_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == portfolio_id
                )
            ).first()
            if row is None:
                raise KeyError(portfolio_id)
            result = self._portfolio_dict(row)
            result["latest_nav"] = self._first_dict(
                connection.execute(
                    select(simulation_nav)
                    .where(simulation_nav.c.portfolio_id == portfolio_id)
                    .order_by(simulation_nav.c.trade_date.desc())
                    .limit(1)
                ).first()
            )
        return result

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(simulation_portfolios)
                .order_by(simulation_portfolios.c.updated_at.desc())
                .limit(limit)
            )
            return [self._portfolio_dict(row) for row in rows]

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(simulation_batches).where(simulation_batches.c.id == batch_id)
            ).first()
            if row is None:
                raise KeyError(batch_id)
            return self._batch_dict(row)

    def execution_manifest(self, batch_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            batch = connection.execute(
                select(simulation_batches).where(simulation_batches.c.id == batch_id)
            ).first()
            if batch is None:
                raise KeyError(batch_id)
            portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == batch.portfolio_id
                )
            ).one()
            snapshot = connection.execute(
                select(recommendation_snapshots).where(
                    recommendation_snapshots.c.id == batch.recommendation_snapshot_id
                )
            ).one()
            target_instruments = set(
                str(value)
                for value in connection.scalars(
                    select(recommendation_holdings.c.instrument).where(
                        recommendation_holdings.c.snapshot_id == snapshot.id
                    )
                )
            )
            held_instruments = set(
                str(value)
                for value in connection.scalars(
                    select(simulation_positions.c.instrument).where(
                        simulation_positions.c.portfolio_id == portfolio.id
                    )
                )
            )
        return {
            "batch_id": str(batch.id),
            "portfolio_id": str(portfolio.id),
            "recommendation_portfolio_id": str(portfolio.recommendation_portfolio_id),
            "recommendation_snapshot_id": str(snapshot.id),
            "trade_date": batch.trade_date.isoformat(),
            "daily_dataset": str(portfolio.daily_dataset),
            "execution_dataset": str(portfolio.execution_dataset),
            "execution_algorithm": str(portfolio.execution_algorithm),
            "instruments": sorted(target_instruments | held_instruments),
        }

    def rows(self, portfolio_id: str, resource: str) -> list[dict[str, Any]]:
        resources = {
            "orders": (
                simulation_orders.join(
                    simulation_batches,
                    simulation_orders.c.batch_id == simulation_batches.c.id,
                ),
                simulation_orders,
                simulation_orders.c.created_at,
            ),
            "fills": (
                simulation_fills.join(
                    simulation_batches,
                    simulation_fills.c.batch_id == simulation_batches.c.id,
                ),
                simulation_fills,
                simulation_fills.c.executed_at,
            ),
            "positions": (
                simulation_positions,
                simulation_positions,
                simulation_positions.c.instrument,
            ),
            "nav": (simulation_nav, simulation_nav, simulation_nav.c.trade_date),
            "events": (simulation_events, simulation_events, simulation_events.c.created_at),
        }
        if resource not in resources:
            raise ValueError("unknown simulation resource")
        source, table, ordering = resources[resource]
        portfolio_column = (
            simulation_batches.c.portfolio_id
            if resource in {"orders", "fills"}
            else table.c.portfolio_id
        )
        with self.engine.connect() as connection:
            return [
                row_dict(row)
                for row in connection.execute(
                    select(table)
                    .select_from(source)
                    .where(portfolio_column == portfolio_id)
                    .order_by(ordering)
                )
            ]

    @staticmethod
    def _portfolio_dict(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["execution_policy"] = result.pop("execution_policy_json")
        result["provenance"] = {
            "daily": {
                "dataset": result["daily_dataset"],
                "dataset_identity_sha256": result["daily_dataset_identity_sha256"],
                "dataset_lineage_id": result["daily_dataset_lineage_id"],
                "field_contract_version": result["daily_field_contract_version"],
            },
            "minute": {
                "dataset": result["execution_dataset"],
                "dataset_identity_sha256": result[
                    "execution_dataset_identity_sha256"
                ],
                "dataset_lineage_id": result["execution_dataset_lineage_id"],
                "field_contract_version": result["execution_field_contract_version"],
            },
            "execution_engine_version": result["execution_engine_version"],
            "cost_schedule_version": result["cost_schedule_version"],
        }
        return result

    @staticmethod
    def _batch_dict(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["summary"] = result.pop("summary_json")
        return result

    @staticmethod
    def _first_dict(row: Any | None) -> dict[str, Any] | None:
        return row_dict(row) if row is not None else None
