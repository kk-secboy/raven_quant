from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    open_database,
    pair_paper_fills,
    pair_paper_orders,
    pair_paper_portfolios,
    pair_portfolio_batches,
    pair_portfolio_nav,
    pair_portfolio_reviews,
    pair_portfolio_risk_events,
    row_dict,
)

from .schedule_store import synchronize_pair_portfolio_schedules
from .strategy_store import StrategyStore

_ROLL_POLICIES = {"pinned", "latest_compatible"}


def _now() -> datetime:
    return datetime.now(UTC)


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _state_payload(row: Any) -> dict[str, Any]:
    return {
        "status": str(row.status),
        "cash": str(row.cash),
        "nav": str(row.nav),
        "high_water_mark": str(row.high_water_mark),
        "position_direction": int(row.position_direction),
        "quantity_y": int(row.quantity_y),
        "quantity_x": int(row.quantity_x),
        "entry_nav": str(row.entry_nav) if row.entry_nav is not None else None,
        "holding_days": int(row.holding_days),
        "last_signal_date": row.last_signal_date.isoformat() if row.last_signal_date else None,
        "last_trade_date": row.last_trade_date.isoformat() if row.last_trade_date else None,
    }


def _state_sha256(row: Any) -> str:
    payload = json.dumps(_state_payload(row), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_position(direction: int, quantity_y: int, quantity_x: int) -> None:
    if direction not in {-1, 0, 1}:
        raise ValueError("pair portfolio direction must be -1, 0, or 1")
    if direction == 0 and (quantity_y or quantity_x):
        raise ValueError("flat pair portfolio cannot retain leg quantities")
    if direction == 1 and not (quantity_y > 0 and quantity_x < 0):
        raise ValueError("positive pair direction requires long Y and short X")
    if direction == -1 and not (quantity_y < 0 and quantity_x > 0):
        raise ValueError("negative pair direction requires short Y and long X")


def _pair_batch_evidence(
    portfolio: Any,
    dataset_evidence: dict[str, Any] | None,
    execution_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    if dataset_evidence is None:
        if portfolio.dataset_roll_policy == "latest_compatible":
            raise ValueError("latest-compatible pair batch requires Qlib dataset evidence")
        dataset = {
            "name": str(portfolio.dataset),
            "identity": None,
            "lineage_id": None,
        }
    else:
        provenance = dict(dataset_evidence.get("provenance") or {})
        dataset = {
            "name": str(dataset_evidence.get("name") or ""),
            "identity": provenance.get("dataset_identity_sha256"),
            "lineage_id": dataset_evidence.get("lineage_id"),
        }
        if not dataset["name"] or not _is_sha256(dataset["identity"]):
            raise ValueError("pair batch requires immutable Qlib dataset evidence")
        if portfolio.dataset_roll_policy == "pinned" and dataset["name"] != portfolio.dataset:
            raise ValueError("pinned pair portfolio cannot change its Qlib dataset")
        if portfolio.dataset_roll_policy == "latest_compatible" and (
            dataset["lineage_id"] != portfolio.dataset_lineage_id
            or not _is_sha256(dataset["lineage_id"])
        ):
            raise ValueError("resolved Qlib dataset is outside the pair portfolio lineage")

    if execution_evidence is None:
        if portfolio.execution_roll_policy == "latest_compatible":
            raise ValueError("latest-compatible pair batch requires execution evidence")
        execution = {
            "name": str(portfolio.execution_snapshot),
            "manifest_sha256": None,
            "lineage_id": None,
        }
    else:
        minute = dict(execution_evidence.get("minute") or {})
        shortability = dict(execution_evidence.get("shortability") or {})
        snapshot = dict(execution_evidence.get("snapshot") or {})
        if minute.get("manifest_sha256") != shortability.get("manifest_sha256"):
            raise ValueError("minute and shortability evidence must share one snapshot")
        execution = {
            "name": str(snapshot.get("name") or minute.get("snapshot_name") or ""),
            "manifest_sha256": minute.get("manifest_sha256"),
            "lineage_id": snapshot.get("lineage_id") or minute.get("snapshot_lineage_id"),
        }
        if not execution["name"] or not _is_sha256(execution["manifest_sha256"]):
            raise ValueError("pair batch requires immutable execution-snapshot evidence")
        if (
            portfolio.execution_roll_policy == "pinned"
            and execution["name"] != portfolio.execution_snapshot
        ):
            raise ValueError("pinned pair portfolio cannot change its execution snapshot")
        if portfolio.execution_roll_policy == "latest_compatible" and (
            execution["lineage_id"] != portfolio.execution_lineage_id
            or not _is_sha256(execution["lineage_id"])
        ):
            raise ValueError("resolved execution snapshot is outside the portfolio lineage")
    return {"dataset": dataset, "execution": execution}


class PairPortfolioStore:
    """Atomic PostgreSQL ledger for two-leg statistical-arbitrage paper portfolios."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)
        self.strategies = StrategyStore(database_url)

    def create(
        self,
        *,
        name: str,
        strategy_version_id: str,
        dataset: str,
        execution_snapshot: str,
        minute_dataset: str,
        shortability_dataset: str,
        initial_cash: float,
        actor: str,
        dataset_roll_policy: str = "pinned",
        dataset_lineage_id: str | None = None,
        execution_roll_policy: str = "pinned",
        execution_lineage_id: str | None = None,
    ) -> dict[str, Any]:
        if not name.strip() or not actor.strip():
            raise ValueError("pair portfolio name and actor are required")
        if initial_cash < 100_000:
            raise ValueError("pair portfolio initial cash must be at least 100000")
        if minute_dataset == shortability_dataset:
            raise ValueError("minute and shortability evidence must be separate datasets")
        if dataset_roll_policy not in _ROLL_POLICIES or execution_roll_policy not in _ROLL_POLICIES:
            raise ValueError("unsupported pair data roll policy")
        if dataset_roll_policy == "latest_compatible" and not _is_sha256(dataset_lineage_id):
            raise ValueError("latest-compatible pair portfolios require a Qlib lineage")
        if execution_roll_policy == "latest_compatible" and not _is_sha256(
            execution_lineage_id
        ):
            raise ValueError("latest-compatible pair portfolios require an execution lineage")
        version = self.strategies.get_version(strategy_version_id)
        if version.get("strategy_type") != "pair" or not version.get("pair"):
            raise ValueError("pair paper portfolios require a pair strategy version")
        if version["status"] != "approved":
            raise ValueError("pair paper portfolios require an approved pair strategy version")
        approved_capacity = float(version["config"].get("initial_capital") or 0)
        if initial_cash > approved_capacity + 1e-6:
            raise ValueError("pair paper capital exceeds the approved backtest capital")
        expected_execution = f"{execution_snapshot}/{minute_dataset}+{shortability_dataset}"
        backtests = self.strategies.list_backtests(version_id=strategy_version_id, limit=1)
        if (
            not backtests
            or backtests[0]["status"] != "succeeded"
            or backtests[0]["dataset"] != dataset
            or backtests[0].get("execution_dataset") != expected_execution
        ):
            raise ValueError("pair paper inputs must match the approved backtest evidence")
        approved_provenance = dict(backtests[0]["metrics"].get("provenance") or {})
        if dataset_roll_policy == "latest_compatible" and (
            approved_provenance.get("daily_dataset_lineage_id") != dataset_lineage_id
        ):
            raise ValueError("approved pair backtest does not pin the requested Qlib lineage")
        if execution_roll_policy == "latest_compatible" and (
            approved_provenance.get("execution_snapshot_lineage_id")
            != execution_lineage_id
        ):
            raise ValueError("approved pair backtest does not pin the execution lineage")
        portfolio_id = uuid.uuid4().hex
        now = _now()
        cash = _decimal(initial_cash)
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(pair_paper_portfolios).values(
                        id=portfolio_id,
                        name=name.strip(),
                        strategy_version_id=strategy_version_id,
                        dataset=dataset,
                        execution_snapshot=execution_snapshot,
                        minute_dataset=minute_dataset,
                        shortability_dataset=shortability_dataset,
                        dataset_roll_policy=dataset_roll_policy,
                        dataset_lineage_id=dataset_lineage_id,
                        execution_roll_policy=execution_roll_policy,
                        execution_lineage_id=execution_lineage_id,
                        status="active",
                        base_currency="CNY",
                        initial_cash=cash,
                        cash=cash,
                        nav=cash,
                        high_water_mark=cash,
                        position_direction=0,
                        quantity_y=0,
                        quantity_x=0,
                        entry_nav=None,
                        holding_days=0,
                        created_by=actor.strip(),
                        created_at=now,
                        updated_at=now,
                    )
                )
        except IntegrityError as exc:
            raise ValueError(f"pair portfolio name {name!r} already exists") from exc
        return self.get(portfolio_id)

    def get(self, portfolio_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(pair_paper_portfolios).where(pair_paper_portfolios.c.id == portfolio_id)
            ).first()
            if row is None:
                raise KeyError(portfolio_id)
            result = row_dict(row)
            result["version"] = self.strategies.get_version(str(row.strategy_version_id))
            result["nav_history"] = [
                row_dict(item)
                for item in connection.execute(
                    select(pair_portfolio_nav)
                    .where(pair_portfolio_nav.c.portfolio_id == portfolio_id)
                    .order_by(pair_portfolio_nav.c.trade_date.desc())
                    .limit(500)
                )
            ]
            result["batches"] = [
                self._batch_row(item)
                for item in connection.execute(
                    select(pair_portfolio_batches)
                    .where(pair_portfolio_batches.c.portfolio_id == portfolio_id)
                    .order_by(pair_portfolio_batches.c.created_at.desc())
                    .limit(100)
                )
            ]
            result["orders"] = [
                row_dict(item)
                for item in connection.execute(
                    select(pair_paper_orders)
                    .where(pair_paper_orders.c.portfolio_id == portfolio_id)
                    .order_by(pair_paper_orders.c.created_at.desc())
                    .limit(200)
                )
            ]
            result["risk_events"] = [
                self._risk_row(item)
                for item in connection.execute(
                    select(pair_portfolio_risk_events)
                    .where(pair_portfolio_risk_events.c.portfolio_id == portfolio_id)
                    .order_by(pair_portfolio_risk_events.c.created_at.desc())
                    .limit(100)
                )
            ]
            result["reviews"] = [
                self._review_row(item)
                for item in connection.execute(
                    select(pair_portfolio_reviews)
                    .where(pair_portfolio_reviews.c.portfolio_id == portfolio_id)
                    .order_by(pair_portfolio_reviews.c.trade_date.desc())
                    .limit(100)
                )
            ]
        return result

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            ids = [
                str(item.id)
                for item in connection.execute(
                    select(pair_paper_portfolios.c.id)
                    .order_by(pair_paper_portfolios.c.updated_at.desc())
                    .limit(limit)
                )
            ]
        return [self.get(portfolio_id) for portfolio_id in ids]

    def set_status(self, portfolio_id: str, status: str) -> dict[str, Any]:
        if status not in {"active", "paused", "closed"}:
            raise ValueError("pair portfolio status must be active, paused, or closed")
        now = _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(pair_paper_portfolios)
                .where(pair_paper_portfolios.c.id == portfolio_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(portfolio_id)
            if row.status == "closed":
                raise ValueError("closed pair portfolios cannot be reopened")
            if status == "paused" and row.status == "liquidation_pending":
                raise ValueError("liquidation-pending pair portfolios cannot be paused")
            if status == "closed" and (row.position_direction or row.quantity_y or row.quantity_x):
                raise ValueError("pair portfolio must be flat before closing")
            if status == "active":
                unresolved = connection.scalar(
                    select(func.count())
                    .select_from(pair_portfolio_risk_events)
                    .where(
                        pair_portfolio_risk_events.c.portfolio_id == portfolio_id,
                        pair_portfolio_risk_events.c.severity == "critical",
                        pair_portfolio_risk_events.c.status.in_(["open", "acknowledged"]),
                    )
                )
                if unresolved:
                    raise ValueError("critical pair risk events must be resolved before activation")
            connection.execute(
                update(pair_paper_portfolios)
                .where(pair_paper_portfolios.c.id == portfolio_id)
                .values(status=status, updated_at=now)
            )
            synchronize_pair_portfolio_schedules(connection, portfolio_id, status, now=now)
        return self.get(portfolio_id)

    def create_batch(
        self,
        *,
        portfolio_id: str,
        as_of_date: date,
        artifact_path: Path,
        dataset_evidence: dict[str, Any] | None = None,
        execution_evidence: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        batch_id = uuid.uuid4().hex
        key = f"pair-paper-rebalance:{portfolio_id}:{as_of_date.isoformat()}"
        created = True
        try:
            with self.engine.begin() as connection:
                portfolio = connection.execute(
                    select(pair_paper_portfolios)
                    .where(pair_paper_portfolios.c.id == portfolio_id)
                    .with_for_update()
                ).first()
                if portfolio is None:
                    raise KeyError(portfolio_id)
                if portfolio.status not in {"active", "liquidation_pending"}:
                    raise ValueError("pair portfolio must be active or liquidation-pending")
                if portfolio.last_signal_date and as_of_date <= portfolio.last_signal_date:
                    raise ValueError("pair portfolio signal dates must increase strictly")
                active = connection.scalar(
                    select(func.count())
                    .select_from(pair_portfolio_batches)
                    .where(
                        pair_portfolio_batches.c.portfolio_id == portfolio_id,
                        pair_portfolio_batches.c.status.in_(["queued", "running"]),
                    )
                )
                if active:
                    raise ValueError("pair portfolio already has an active batch")
                evidence = _pair_batch_evidence(
                    portfolio,
                    dataset_evidence,
                    execution_evidence,
                )
                connection.execute(
                    insert(pair_portfolio_batches).values(
                        id=batch_id,
                        portfolio_id=portfolio_id,
                        as_of_date=as_of_date,
                        status="queued",
                        idempotency_key=key,
                        starting_state_sha256=_state_sha256(portfolio),
                        dataset=evidence["dataset"]["name"],
                        dataset_identity_sha256=evidence["dataset"].get("identity"),
                        dataset_lineage_id=evidence["dataset"].get("lineage_id"),
                        execution_snapshot=evidence["execution"]["name"],
                        execution_manifest_sha256=evidence["execution"].get(
                            "manifest_sha256"
                        ),
                        execution_lineage_id=evidence["execution"].get("lineage_id"),
                        artifact_path=str(artifact_path / batch_id),
                        created_at=_now(),
                    )
                )
        except IntegrityError:
            created = False
            with self.engine.connect() as connection:
                existing = connection.execute(
                    select(pair_portfolio_batches).where(
                        pair_portfolio_batches.c.idempotency_key == key
                    )
                ).first()
            if existing is None:
                raise
            batch_id = str(existing.id)
        return self.get_batch(batch_id), created

    def attach_job(self, batch_id: str, job_id: str) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(pair_portfolio_batches)
                .where(pair_portfolio_batches.c.id == batch_id)
                .values(job_id=job_id)
            )
            if not result.rowcount:
                raise KeyError(batch_id)

    def mark_batch(self, batch_id: str, status: str, *, error: str | None = None) -> None:
        if status not in {"running", "failed", "cancelled"}:
            raise ValueError("unsupported pair batch status transition")
        values: dict[str, Any] = {"status": status, "error": error}
        if status == "running":
            values["started_at"] = _now()
        else:
            values["finished_at"] = _now()
        with self.engine.begin() as connection:
            result = connection.execute(
                update(pair_portfolio_batches)
                .where(pair_portfolio_batches.c.id == batch_id)
                .values(**values)
            )
            if not result.rowcount:
                raise KeyError(batch_id)

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(pair_portfolio_batches).where(pair_portfolio_batches.c.id == batch_id)
            ).first()
        if row is None:
            raise KeyError(batch_id)
        return self._batch_row(row)

    def apply_batch(self, batch_id: str, result: dict[str, Any]) -> dict[str, Any]:
        now = _now()
        with self.engine.begin() as connection:
            batch = connection.execute(
                select(pair_portfolio_batches)
                .where(pair_portfolio_batches.c.id == batch_id)
                .with_for_update()
            ).first()
            if batch is None:
                raise KeyError(batch_id)
            if batch.status == "succeeded":
                return self._batch_row(batch)
            if batch.status not in {"queued", "running"}:
                raise ValueError(f"cannot apply a {batch.status} pair batch")
            portfolio = connection.execute(
                select(pair_paper_portfolios)
                .where(pair_paper_portfolios.c.id == batch.portfolio_id)
                .with_for_update()
            ).first()
            if portfolio is None:
                raise KeyError(batch.portfolio_id)
            if _state_sha256(portfolio) != batch.starting_state_sha256:
                raise ValueError("pair portfolio state changed after the batch was queued")
            if result.get("status") != "ok":
                raise ValueError(str(result.get("error") or "pair paper step did not succeed"))
            if result.get("as_of_date") != batch.as_of_date.isoformat():
                raise ValueError("pair result as_of_date does not match the batch")
            pair = self.strategies.get_version(str(portfolio.strategy_version_id))["pair"]
            if result.get("leg_y") != pair["leg_y"] or result.get("leg_x") != pair["leg_x"]:
                raise ValueError("pair result instruments do not match the portfolio")
            provenance = result.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError("pair paper result requires immutable provenance")
            backtests = self.strategies.list_backtests(
                version_id=str(portfolio.strategy_version_id), limit=1
            )
            approved_provenance = dict(backtests[0]["metrics"].get("provenance") or {})
            evidence_fields = (
                "daily_dataset_identity_sha256",
                "daily_snapshot_manifest_sha256",
                "minute_snapshot_manifest_sha256",
                "shortability_evidence_sha256",
                "strategy_config_sha256",
                "pair_engine_sha256",
            )
            for field in (*evidence_fields, "execution_manifest_sha256"):
                if not _is_sha256(provenance.get(field)):
                    raise ValueError(f"pair paper provenance {field} must be a SHA-256 digest")
            for field in ("strategy_config_sha256", "pair_engine_sha256"):
                if provenance[field] != approved_provenance.get(field):
                    raise ValueError(f"pair paper provenance {field} changed since approval")
            if portfolio.dataset_roll_policy == "pinned":
                for field in (
                    "daily_dataset_identity_sha256",
                    "daily_snapshot_manifest_sha256",
                ):
                    if provenance[field] != approved_provenance.get(field):
                        raise ValueError(f"pair paper provenance {field} changed since approval")
            else:
                daily_lineage = provenance.get("daily_dataset_lineage_id")
                if (
                    not _is_sha256(daily_lineage)
                    or daily_lineage != portfolio.dataset_lineage_id
                    or daily_lineage != approved_provenance.get("daily_dataset_lineage_id")
                ):
                    raise ValueError("pair paper daily evidence left the approved lineage")
            if portfolio.execution_roll_policy == "pinned":
                for field in (
                    "minute_snapshot_manifest_sha256",
                    "shortability_evidence_sha256",
                ):
                    if provenance[field] != approved_provenance.get(field):
                        raise ValueError(f"pair paper provenance {field} changed since approval")
            else:
                execution_lineage = provenance.get("execution_snapshot_lineage_id")
                if (
                    not _is_sha256(execution_lineage)
                    or execution_lineage != portfolio.execution_lineage_id
                    or execution_lineage
                    != approved_provenance.get("execution_snapshot_lineage_id")
                ):
                    raise ValueError("pair paper execution evidence left the approved lineage")
            if batch.dataset_identity_sha256 and (
                provenance["daily_dataset_identity_sha256"]
                != batch.dataset_identity_sha256
            ):
                raise ValueError("pair paper result changed the batch-pinned Qlib dataset")
            if batch.execution_manifest_sha256 and (
                provenance["minute_snapshot_manifest_sha256"]
                != batch.execution_manifest_sha256
            ):
                raise ValueError("pair paper result changed the batch-pinned execution snapshot")
            trade_date = date.fromisoformat(str(result["trade_date"]))
            if trade_date <= batch.as_of_date:
                raise ValueError("pair trade date must be after the signal date")
            if portfolio.last_trade_date and trade_date <= portfolio.last_trade_date:
                raise ValueError("pair trade dates must increase strictly")

            orders = list(result.get("orders") or [])
            fills = list(result.get("fills") or [])
            action = str(result.get("action") or "hold")
            if action not in {"hold", "entry", "exit"}:
                raise ValueError("pair result action is invalid")
            if action == "hold" and (orders or fills):
                raise ValueError("pair hold results cannot contain orders or fills")
            if action in {"entry", "exit"} and len(orders) != 2:
                raise ValueError("pair entry/exit must retain exactly two leg orders")
            if fills and len(fills) != 2:
                raise ValueError("pair fills must be atomic across both legs")
            if result.get("rejection") and fills:
                raise ValueError("rejected pair actions cannot contain fills")
            order_instruments = {str(item.get("instrument")) for item in orders}
            order_by_instrument = {str(item.get("instrument")): item for item in orders}
            if len(order_by_instrument) != len(orders):
                raise ValueError("pair orders must reference unique instruments")
            fill_by_instrument = {str(item.get("instrument")): item for item in fills}
            if len(fill_by_instrument) != len(fills) or not set(fill_by_instrument).issubset(
                order_instruments
            ):
                raise ValueError("pair fills must reference unique batch leg orders")

            order_ids: dict[str, str] = {}
            for item in orders:
                instrument = str(item["instrument"])
                leg = str(item["leg"])
                side = str(item["side"])
                quantity = int(item["requested_quantity"])
                target = int(item["target_quantity"])
                status = str(item["status"])
                if leg not in {"y", "x"} or side not in {"buy", "sell"}:
                    raise ValueError("pair order leg or side is invalid")
                if quantity <= 0 or status not in {"filled", "rejected"}:
                    raise ValueError("pair order quantity or status is invalid")
                if (instrument in fill_by_instrument) != (status == "filled"):
                    raise ValueError("pair order and fill statuses do not agree")
                order_id = uuid.uuid4().hex
                order_ids[instrument] = order_id
                connection.execute(
                    insert(pair_paper_orders).values(
                        id=order_id,
                        batch_id=batch_id,
                        portfolio_id=portfolio.id,
                        leg=leg,
                        instrument=instrument,
                        side=side,
                        requested_quantity=quantity,
                        target_quantity=target,
                        status=status,
                        reason=item.get("reason"),
                        created_at=now,
                    )
                )
            for instrument, item in fill_by_instrument.items():
                quantity = int(item["quantity"])
                price = _decimal(item["price"])
                gross = _decimal(item["gross_value"])
                fee = _decimal(item["fee"])
                if quantity <= 0 or min(price, gross) <= 0 or fee < 0:
                    raise ValueError("pair fill values are invalid")
                if abs(gross - price * quantity) > Decimal("0.02"):
                    raise ValueError("pair fill gross value does not reconcile")
                connection.execute(
                    insert(pair_paper_fills).values(
                        id=uuid.uuid4().hex,
                        order_id=order_ids[instrument],
                        fill_time=datetime.fromisoformat(str(item["fill_time"])),
                        quantity=quantity,
                        price=price,
                        gross_value=gross,
                        fee=fee,
                        slippage=float(item.get("slippage") or 0),
                        created_at=now,
                    )
                )

            state = dict(result.get("state") or {})
            metrics = dict(result.get("metrics") or {})
            prices = dict(result.get("closing_prices") or {})
            direction = int(state["position_direction"])
            quantity_y = int(state["quantity_y"])
            quantity_x = int(state["quantity_x"])
            _validate_position(direction, quantity_y, quantity_x)
            cash = _decimal(state["cash"])
            nav = _decimal(state["nav"])
            price_y = _decimal(prices[pair["leg_y"]])
            price_x = _decimal(prices[pair["leg_x"]])
            reconciled_nav = cash + quantity_y * price_y + quantity_x * price_x
            if min(cash, nav, price_y, price_x) <= 0:
                raise ValueError("pair state cash, NAV, and marks must be positive")
            if abs(nav - reconciled_nav) > Decimal("0.02"):
                raise ValueError("pair NAV does not reconcile to cash and signed legs")
            expected_cash = _decimal(portfolio.cash)
            expected_fees = Decimal("0")
            expected_gross_traded = Decimal("0")
            for instrument, fill in fill_by_instrument.items():
                order = order_by_instrument[instrument]
                gross = _decimal(fill["gross_value"])
                fee = _decimal(fill["fee"])
                expected_fees += fee
                expected_gross_traded += gross
                expected_cash += gross - fee if order["side"] == "sell" else -gross - fee
            expected_cash -= _decimal(metrics.get("borrow_cost") or 0)
            if abs(cash - expected_cash) > Decimal("0.02"):
                raise ValueError("pair cash does not reconcile to fills, fees, and borrow cost")
            next_status = str(state["status"])
            if next_status not in {"active", "paused", "liquidation_pending"}:
                raise ValueError("pair result portfolio status is invalid")
            high_water_mark = _decimal(state["high_water_mark"])
            entry_nav = _decimal(state["entry_nav"]) if state.get("entry_nav") else None
            if high_water_mark < nav:
                raise ValueError("pair high-water mark cannot be below NAV")
            required_metrics = (
                "daily_return",
                "drawdown",
                "long_value",
                "short_value",
                "gross_exposure",
                "net_exposure",
                "turnover",
                "fees",
                "borrow_cost",
                "zscore",
                "correlation",
                "cointegration_pvalue",
            )
            if any(name not in metrics for name in required_metrics):
                raise ValueError("pair result is missing ledger metrics")
            expected_long = sum(
                quantity * price
                for quantity, price in ((quantity_y, price_y), (quantity_x, price_x))
                if quantity > 0
            )
            expected_short = sum(
                abs(quantity) * price
                for quantity, price in ((quantity_y, price_y), (quantity_x, price_x))
                if quantity < 0
            )
            expected_net = quantity_y * price_y + quantity_x * price_x
            expected_daily_return = float(nav / _decimal(portfolio.nav) - 1)
            expected_drawdown = float(nav / high_water_mark - 1)
            expected_gross_exposure = float((expected_long + expected_short) / nav)
            expected_net_exposure = float(expected_net / nav)
            expected_turnover = float(expected_gross_traded / _decimal(portfolio.nav))
            reconciliations = {
                "long_value": (float(metrics["long_value"]), float(expected_long)),
                "short_value": (float(metrics["short_value"]), float(expected_short)),
                "daily_return": (float(metrics["daily_return"]), expected_daily_return),
                "drawdown": (float(metrics["drawdown"]), expected_drawdown),
                "gross_exposure": (
                    float(metrics["gross_exposure"]),
                    expected_gross_exposure,
                ),
                "net_exposure": (float(metrics["net_exposure"]), expected_net_exposure),
                "turnover": (float(metrics["turnover"]), expected_turnover),
                "fees": (float(metrics["fees"]), float(expected_fees)),
            }
            for field, (observed, expected) in reconciliations.items():
                if abs(observed - expected) > 1e-8:
                    raise ValueError(f"pair {field} does not reconcile")
            if fills:
                targets = {str(item["leg"]): int(item["target_quantity"]) for item in orders}
                if targets != {"y": quantity_y, "x": quantity_x}:
                    raise ValueError("pair filled order targets do not match final quantities")
            elif orders and (
                quantity_y != int(portfolio.quantity_y) or quantity_x != int(portfolio.quantity_x)
            ):
                raise ValueError("rejected pair orders cannot change final quantities")
            for enforcement in (
                "atomic_pair_execution_enforced",
                "shortability_enforced",
                "minute_execution_enforced",
            ):
                if metrics.get(enforcement) is not True:
                    raise ValueError(f"{enforcement} is required for pair paper execution")

            connection.execute(
                insert(pair_portfolio_nav).values(
                    portfolio_id=portfolio.id,
                    trade_date=trade_date,
                    cash=cash,
                    long_value=_decimal(metrics["long_value"]),
                    short_value=_decimal(metrics["short_value"]),
                    nav=nav,
                    daily_return=float(metrics["daily_return"]),
                    drawdown=float(metrics["drawdown"]),
                    gross_exposure=float(metrics["gross_exposure"]),
                    net_exposure=float(metrics["net_exposure"]),
                    turnover=float(metrics["turnover"]),
                    fees=_decimal(metrics["fees"]),
                    borrow_cost=_decimal(metrics["borrow_cost"]),
                    zscore=float(metrics["zscore"]),
                    correlation=float(metrics["correlation"]),
                    cointegration_pvalue=float(metrics["cointegration_pvalue"]),
                    position_direction=direction,
                    quantity_y=quantity_y,
                    quantity_x=quantity_x,
                    price_y=price_y,
                    price_x=price_x,
                    created_at=now,
                )
            )
            for event in result.get("risk_events") or []:
                connection.execute(
                    insert(pair_portfolio_risk_events).values(
                        portfolio_id=portfolio.id,
                        batch_id=batch_id,
                        severity=str(event.get("severity") or "warning"),
                        event_type=str(event.get("event_type") or "pair_risk"),
                        rule=str(event["rule"]),
                        observed=event.get("observed"),
                        limit_value=event.get("limit_value"),
                        status="open",
                        details_json=dict(event.get("details") or {}),
                        created_at=now,
                    )
                )
            summary = {
                "action": action,
                "reason": result.get("reason"),
                "rejection": result.get("rejection"),
                "orders": len(orders),
                "fills": len(fills),
                "fees": float(metrics["fees"]),
                "borrow_cost": float(metrics["borrow_cost"]),
                "daily_return": float(metrics["daily_return"]),
                "drawdown": float(metrics["drawdown"]),
                "gross_exposure": float(metrics["gross_exposure"]),
                "net_exposure": float(metrics["net_exposure"]),
                "zscore": float(metrics["zscore"]),
                "correlation": float(metrics["correlation"]),
                "cointegration_pvalue": float(metrics["cointegration_pvalue"]),
            }
            connection.execute(
                insert(pair_portfolio_reviews).values(
                    id=uuid.uuid4().hex,
                    portfolio_id=portfolio.id,
                    batch_id=batch_id,
                    trade_date=trade_date,
                    status="completed",
                    summary_json=summary,
                    created_at=now,
                )
            )
            connection.execute(
                update(pair_paper_portfolios)
                .where(pair_paper_portfolios.c.id == portfolio.id)
                .values(
                    status=next_status,
                    cash=cash,
                    nav=nav,
                    high_water_mark=high_water_mark,
                    position_direction=direction,
                    quantity_y=quantity_y,
                    quantity_x=quantity_x,
                    entry_nav=entry_nav,
                    holding_days=int(state["holding_days"]),
                    last_signal_date=batch.as_of_date,
                    last_trade_date=trade_date,
                    updated_at=now,
                )
            )
            synchronize_pair_portfolio_schedules(
                connection, str(portfolio.id), next_status, now=now
            )
            connection.execute(
                update(pair_portfolio_batches)
                .where(pair_portfolio_batches.c.id == batch_id)
                .values(status="succeeded", trade_date=trade_date, error=None, finished_at=now)
            )
        return self.get_batch(batch_id)

    def acknowledge_risk_event(
        self, portfolio_id: str, event_id: int, *, actor: str
    ) -> dict[str, Any]:
        if not actor.strip():
            raise ValueError("risk acknowledgement actor is required")
        with self.engine.begin() as connection:
            row = connection.execute(
                select(pair_portfolio_risk_events)
                .where(
                    pair_portfolio_risk_events.c.id == event_id,
                    pair_portfolio_risk_events.c.portfolio_id == portfolio_id,
                )
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(event_id)
            if row.status != "open":
                raise ValueError("only open pair risk events can be acknowledged")
            connection.execute(
                update(pair_portfolio_risk_events)
                .where(pair_portfolio_risk_events.c.id == event_id)
                .values(
                    status="acknowledged", acknowledged_by=actor.strip(), acknowledged_at=_now()
                )
            )
        return self.get_risk_event(portfolio_id, event_id)

    def resolve_risk_event(
        self, portfolio_id: str, event_id: int, *, actor: str, reason: str
    ) -> dict[str, Any]:
        if not actor.strip() or len(reason.strip()) < 10:
            raise ValueError("risk resolution requires an actor and meaningful reason")
        with self.engine.begin() as connection:
            event = connection.execute(
                select(pair_portfolio_risk_events)
                .where(
                    pair_portfolio_risk_events.c.id == event_id,
                    pair_portfolio_risk_events.c.portfolio_id == portfolio_id,
                )
                .with_for_update()
            ).first()
            if event is None:
                raise KeyError(event_id)
            if event.status != "acknowledged":
                raise ValueError("pair risk event must be acknowledged before resolution")
            portfolio = connection.execute(
                select(pair_paper_portfolios).where(pair_paper_portfolios.c.id == portfolio_id)
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            if portfolio.position_direction or portfolio.status == "liquidation_pending":
                raise ValueError("pair risk event cannot resolve before the spread is flat")
            active_batches = connection.scalar(
                select(func.count())
                .select_from(pair_portfolio_batches)
                .where(
                    pair_portfolio_batches.c.portfolio_id == portfolio_id,
                    pair_portfolio_batches.c.status.in_(["queued", "running"]),
                )
            )
            if active_batches:
                raise ValueError("pair risk event cannot resolve while a batch is active")
            connection.execute(
                update(pair_portfolio_risk_events)
                .where(pair_portfolio_risk_events.c.id == event_id)
                .values(
                    status="resolved",
                    resolved_by=actor.strip(),
                    resolved_at=_now(),
                    resolution_reason=reason.strip(),
                )
            )
        return self.get_risk_event(portfolio_id, event_id)

    def get_risk_event(self, portfolio_id: str, event_id: int) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(pair_portfolio_risk_events).where(
                    pair_portfolio_risk_events.c.id == event_id,
                    pair_portfolio_risk_events.c.portfolio_id == portfolio_id,
                )
            ).first()
        if row is None:
            raise KeyError(event_id)
        return self._risk_row(row)

    @staticmethod
    def _batch_row(row: Any) -> dict[str, Any]:
        return row_dict(row)

    @staticmethod
    def _risk_row(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["details"] = result.pop("details_json")
        return result

    @staticmethod
    def _review_row(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["summary"] = result.pop("summary_json")
        return result
