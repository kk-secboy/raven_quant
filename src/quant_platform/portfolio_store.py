from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import delete, insert, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    paper_fills,
    paper_orders,
    paper_portfolios,
    paper_positions,
    portfolio_batches,
    portfolio_nav,
    portfolio_reviews,
    risk_events,
    row_dict,
    strategy_versions,
)

from .schedule_store import synchronize_portfolio_schedules
from .strategy_store import StrategyStore

_ROLL_POLICIES = {"pinned", "latest_compatible"}


def _now() -> datetime:
    return datetime.now(UTC)


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_dataset_evidence(
    portfolio: dict[str, Any],
    evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    if evidence is None:
        if portfolio.get("dataset_roll_policy") == "latest_compatible":
            raise ValueError("latest-compatible batch requires resolved dataset evidence")
        return {"name": portfolio["dataset"], "identity": None, "lineage_id": None}
    provenance = dict(evidence.get("provenance") or {})
    resolved = {
        "name": str(evidence.get("name") or ""),
        "identity": provenance.get("dataset_identity_sha256"),
        "lineage_id": evidence.get("lineage_id"),
    }
    if not resolved["name"] or not _is_sha256(resolved["identity"]):
        raise ValueError("paper batch requires immutable Qlib dataset evidence")
    if portfolio.get("dataset_roll_policy") == "pinned":
        if resolved["name"] != portfolio["dataset"]:
            raise ValueError("pinned portfolio cannot change its Qlib dataset")
    elif not _is_sha256(resolved["lineage_id"]) or resolved["lineage_id"] != portfolio.get(
        "dataset_lineage_id"
    ):
        raise ValueError("resolved Qlib dataset is outside the portfolio lineage")
    return resolved


def _pnl_contributors(
    starting: dict[str, dict[str, Any]],
    ending: dict[str, dict[str, Any]],
    orders: dict[str, dict[str, Any]],
    fills: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    contributors = []
    for instrument in sorted(set(starting) | set(ending) | set(fills)):
        before = starting.get(instrument, {})
        after = ending.get(instrument, {})
        start_quantity = _decimal(before.get("quantity", 0))
        start_mark = _decimal(before.get("market_price", 0))
        end_mark = _decimal(after.get("market_price", start_mark))
        fill = fills.get(instrument)
        contribution = start_quantity * (end_mark - start_mark)
        if fill:
            fill_quantity = _decimal(fill["quantity"])
            fill_price = _decimal(fill["price"])
            if str(orders[instrument]["side"]).lower() == "buy":
                contribution += fill_quantity * (end_mark - fill_price)
            else:
                sold = min(start_quantity, fill_quantity)
                remaining = max(Decimal("0"), start_quantity - sold)
                contribution = sold * (fill_price - start_mark) + remaining * (
                    end_mark - start_mark
                )
        if start_quantity or fill:
            contributors.append({"instrument": instrument, "pnl": float(contribution)})
    return contributors


class PortfolioStore:
    """Atomic PostgreSQL ledger for governed paper portfolios."""

    def __init__(self, database_url: str) -> None:
        self.strategies = StrategyStore(database_url)
        self.engine = self.strategies.engine

    def create(
        self,
        *,
        name: str,
        strategy_version_id: str,
        dataset: str,
        initial_cash: float,
        actor: str,
        dataset_roll_policy: str = "pinned",
        dataset_lineage_id: str | None = None,
    ) -> dict[str, Any]:
        version = self.strategies.get_version(strategy_version_id)
        if version["status"] != "approved":
            raise ValueError("paper portfolios require an approved strategy version")
        if version.get("strategy_type") != "multifactor":
            raise ValueError("pair strategy paper portfolios require the dedicated spread ledger")
        if initial_cash < 100_000:
            raise ValueError("initial cash must be at least 100000")
        if not name.strip() or not dataset.strip() or not actor.strip():
            raise ValueError("name, dataset, and actor are required")
        if dataset_roll_policy not in _ROLL_POLICIES:
            raise ValueError("unsupported dataset roll policy")
        if dataset_roll_policy == "latest_compatible" and not _is_sha256(dataset_lineage_id):
            raise ValueError("latest-compatible portfolios require a verified dataset lineage")
        portfolio_id = uuid.uuid4().hex
        now = _now()
        cash = _decimal(initial_cash)
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(paper_portfolios).values(
                        id=portfolio_id,
                        name=name.strip(),
                        strategy_version_id=strategy_version_id,
                        dataset=dataset,
                        dataset_roll_policy=dataset_roll_policy,
                        dataset_lineage_id=dataset_lineage_id,
                        status="active",
                        base_currency="CNY",
                        initial_cash=cash,
                        cash=cash,
                        nav=cash,
                        high_water_mark=cash,
                        created_by=actor.strip(),
                        created_at=now,
                        updated_at=now,
                    )
                )
        except IntegrityError as exc:
            raise ValueError(f"portfolio name {name!r} already exists") from exc
        return self.get(portfolio_id)

    def get(self, portfolio_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(paper_portfolios).where(paper_portfolios.c.id == portfolio_id)
            ).first()
            if row is None:
                raise KeyError(portfolio_id)
            result = row_dict(row)
            result["positions"] = [
                row_dict(item)
                for item in connection.execute(
                    select(paper_positions)
                    .where(paper_positions.c.portfolio_id == portfolio_id)
                    .order_by(paper_positions.c.market_value.desc())
                )
            ]
            result["nav_history"] = [
                row_dict(item)
                for item in connection.execute(
                    select(portfolio_nav)
                    .where(portfolio_nav.c.portfolio_id == portfolio_id)
                    .order_by(portfolio_nav.c.trade_date.desc())
                    .limit(500)
                )
            ]
            result["batches"] = [
                self._batch_row(item)
                for item in connection.execute(
                    select(portfolio_batches)
                    .where(portfolio_batches.c.portfolio_id == portfolio_id)
                    .order_by(portfolio_batches.c.created_at.desc())
                    .limit(100)
                )
            ]
            result["orders"] = [
                row_dict(item)
                for item in connection.execute(
                    select(paper_orders)
                    .where(paper_orders.c.portfolio_id == portfolio_id)
                    .order_by(paper_orders.c.created_at.desc())
                    .limit(500)
                )
            ]
            result["risk_events"] = [
                self._risk_row(item)
                for item in connection.execute(
                    select(risk_events)
                    .where(risk_events.c.portfolio_id == portfolio_id)
                    .order_by(risk_events.c.created_at.desc())
                    .limit(200)
                )
            ]
            result["reviews"] = [
                self._review_row(item)
                for item in connection.execute(
                    select(portfolio_reviews)
                    .where(portfolio_reviews.c.portfolio_id == portfolio_id)
                    .order_by(portfolio_reviews.c.trade_date.desc())
                    .limit(200)
                )
            ]
        return result

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            ids = [
                str(row.id)
                for row in connection.execute(
                    select(paper_portfolios.c.id)
                    .order_by(paper_portfolios.c.updated_at.desc())
                    .limit(limit)
                )
            ]
        return [self.get(portfolio_id) for portfolio_id in ids]

    def get_by_name(self, name: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            portfolio_id = connection.scalar(
                select(paper_portfolios.c.id).where(paper_portfolios.c.name == name)
            )
        return self.get(str(portfolio_id)) if portfolio_id else None

    def set_status(self, portfolio_id: str, status: str) -> dict[str, Any]:
        if status not in {"active", "paused", "closed"}:
            raise ValueError("portfolio status must be active, paused, or closed")
        now = _now()
        with self.engine.begin() as connection:
            current = connection.execute(
                select(paper_portfolios.c.status).where(paper_portfolios.c.id == portfolio_id)
            ).scalar_one_or_none()
            if current is None:
                raise KeyError(portfolio_id)
            if current in {"liquidation_pending", "risk_reduction_pending"}:
                raise ValueError("pending risk execution cannot be overridden manually")
            if status == "active":
                unresolved = connection.execute(
                    select(risk_events.c.id)
                    .where(
                        risk_events.c.portfolio_id == portfolio_id,
                        risk_events.c.severity.in_(["critical", "hard"]),
                        risk_events.c.status.in_(["open", "acknowledged"]),
                    )
                    .limit(1)
                ).first()
                if unresolved is not None:
                    raise ValueError("critical risk events must be resolved before reactivation")
            result = connection.execute(
                update(paper_portfolios)
                .where(paper_portfolios.c.id == portfolio_id)
                .values(status=status, updated_at=now)
            )
            if not result.rowcount:
                raise KeyError(portfolio_id)
            synchronize_portfolio_schedules(
                connection,
                portfolio_id,
                status,
                now=now,
            )
        return self.get(portfolio_id)

    def acknowledge_risk_event(
        self,
        portfolio_id: str,
        event_id: int,
        *,
        actor: str,
    ) -> dict[str, Any]:
        if len(actor.strip()) < 2:
            raise ValueError("a responsible actor is required")
        now = _now()
        with self.engine.begin() as connection:
            event = connection.execute(
                select(risk_events)
                .where(
                    risk_events.c.id == event_id,
                    risk_events.c.portfolio_id == portfolio_id,
                )
                .with_for_update()
            ).first()
            if event is None:
                raise KeyError(event_id)
            if event.status == "resolved":
                raise ValueError("resolved risk events cannot be acknowledged again")
            if event.status not in {"open", "acknowledged"}:
                raise ValueError("only open risk events may be acknowledged")
            if event.status == "open":
                connection.execute(
                    update(risk_events)
                    .where(risk_events.c.id == event_id)
                    .values(
                        status="acknowledged",
                        acknowledged_by=actor.strip(),
                        acknowledged_at=now,
                    )
                )
        return self.get_risk_event(portfolio_id, event_id)

    def resolve_risk_event(
        self,
        portfolio_id: str,
        event_id: int,
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        if len(actor.strip()) < 2 or len(reason.strip()) < 10:
            raise ValueError("a responsible actor and meaningful resolution reason are required")
        now = _now()
        with self.engine.begin() as connection:
            event = connection.execute(
                select(risk_events)
                .where(
                    risk_events.c.id == event_id,
                    risk_events.c.portfolio_id == portfolio_id,
                )
                .with_for_update()
            ).first()
            if event is None:
                raise KeyError(event_id)
            if event.status == "resolved":
                return self._risk_row(event)
            if event.status not in {"open", "acknowledged"}:
                raise ValueError("only open or acknowledged risk events may be resolved")
            portfolio = connection.execute(
                select(paper_portfolios)
                .where(paper_portfolios.c.id == portfolio_id)
                .with_for_update()
            ).one()
            if portfolio.status in {"liquidation_pending", "risk_reduction_pending"}:
                raise ValueError("pending risk execution must finish before event resolution")
            if event.severity in {"critical", "hard"} and portfolio.status == "active":
                raise ValueError("pause the portfolio before resolving a critical risk event")
            active_batch = connection.execute(
                select(portfolio_batches.c.id)
                .where(
                    portfolio_batches.c.portfolio_id == portfolio_id,
                    portfolio_batches.c.status.in_(["queued", "running"]),
                )
                .limit(1)
            ).first()
            if active_batch is not None:
                raise ValueError("an active portfolio batch must finish before event resolution")

            details = dict(event.details_json or {})
            action = str(details.get("action") or "")
            liquidation = event.rule == "max_drawdown_liquidate" or "liquidat" in action
            reduction = event.rule == "max_drawdown_reduce" or "reduction" in action
            if liquidation:
                position = connection.execute(
                    select(paper_positions.c.instrument)
                    .where(paper_positions.c.portfolio_id == portfolio_id)
                    .limit(1)
                ).first()
                if position is not None:
                    raise ValueError("liquidation risk cannot be resolved while positions remain")
            if reduction:
                version = connection.execute(
                    select(strategy_versions.c.config_json).where(
                        strategy_versions.c.id == portfolio.strategy_version_id
                    )
                ).scalar_one()
                exposure_limit = float(dict(version or {}).get("drawdown_reduction_exposure", 0.50))
                latest_exposure = connection.execute(
                    select(portfolio_nav.c.exposure)
                    .where(portfolio_nav.c.portfolio_id == portfolio_id)
                    .order_by(portfolio_nav.c.trade_date.desc())
                    .limit(1)
                ).scalar_one_or_none()
                if latest_exposure is None or float(latest_exposure) > exposure_limit + 0.01:
                    raise ValueError(
                        "risk reduction cannot be resolved until exposure is within limit"
                    )
            if event.rule == "max_position_weight" and event.limit_value is not None:
                weights = connection.execute(
                    select(paper_positions.c.weight).where(
                        paper_positions.c.portfolio_id == portfolio_id
                    )
                ).scalars()
                if (
                    max((float(item) for item in weights), default=0.0)
                    > float(event.limit_value) + 1e-9
                ):
                    raise ValueError("position concentration remains above the risk-event limit")
            if event.rule == "max_industry_weight" and event.limit_value is not None:
                industry_weights: dict[str, float] = {}
                for row in connection.execute(
                    select(paper_positions.c.industry, paper_positions.c.weight).where(
                        paper_positions.c.portfolio_id == portfolio_id
                    )
                ):
                    industry = str(row.industry or "").strip()
                    industry_weights[industry] = industry_weights.get(industry, 0.0) + float(
                        row.weight
                    )
                if max(industry_weights.values(), default=0.0) > float(event.limit_value) + 1e-9:
                    raise ValueError("industry concentration remains above the risk-event limit")
            if event.rule == "industry_data_missing":
                unclassified = connection.execute(
                    select(paper_positions.c.instrument)
                    .where(
                        paper_positions.c.portfolio_id == portfolio_id,
                        paper_positions.c.industry.is_(None),
                    )
                    .limit(1)
                ).first()
                if unclassified is not None:
                    raise ValueError(
                        "industry-data risk cannot be resolved while holdings are unclassified"
                    )

            values: dict[str, Any] = {
                "status": "resolved",
                "resolved_by": actor.strip(),
                "resolved_at": now,
                "resolution_reason": reason.strip(),
            }
            if event.acknowledged_at is None:
                values.update(acknowledged_by=actor.strip(), acknowledged_at=now)
            connection.execute(
                update(risk_events).where(risk_events.c.id == event_id).values(**values)
            )
        return self.get_risk_event(portfolio_id, event_id)

    def get_risk_event(self, portfolio_id: str, event_id: int) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(risk_events).where(
                    risk_events.c.id == event_id,
                    risk_events.c.portfolio_id == portfolio_id,
                )
            ).first()
        if row is None:
            raise KeyError(event_id)
        return self._risk_row(row)

    def create_batch(
        self,
        *,
        portfolio_id: str,
        as_of_date: date,
        artifact_path: Path,
        dataset_evidence: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        portfolio = self.get(portfolio_id)
        allowed = {"active", "liquidation_pending", "risk_reduction_pending"}
        if portfolio["status"] not in allowed:
            raise ValueError("portfolio must be active or pending risk execution")
        key = f"paper-rebalance:{portfolio_id}:{as_of_date.isoformat()}"
        evidence = _validate_dataset_evidence(portfolio, dataset_evidence)
        batch_id = uuid.uuid4().hex
        created = True
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(portfolio_batches).values(
                        id=batch_id,
                        portfolio_id=portfolio_id,
                        as_of_date=as_of_date,
                        status="queued",
                        idempotency_key=key,
                        artifact_path=str(artifact_path),
                        dataset=evidence["name"],
                        dataset_identity_sha256=evidence.get("identity"),
                        dataset_lineage_id=evidence.get("lineage_id"),
                        created_at=_now(),
                    )
                )
        except IntegrityError:
            created = False
            with self.engine.connect() as connection:
                existing = connection.execute(
                    select(portfolio_batches).where(portfolio_batches.c.idempotency_key == key)
                ).first()
            if existing is None:
                raise
            batch_id = str(existing.id)
        return self.get_batch(batch_id), created

    def attach_job(self, batch_id: str, job_id: str) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(portfolio_batches)
                .where(portfolio_batches.c.id == batch_id)
                .values(job_id=job_id)
            )
            if not result.rowcount:
                raise KeyError(batch_id)

    def mark_batch(self, batch_id: str, status: str, *, error: str | None = None) -> None:
        if status not in {"running", "failed", "cancelled"}:
            raise ValueError("unsupported batch status transition")
        values: dict[str, Any] = {"status": status, "error": error}
        if status == "running":
            values["started_at"] = _now()
        else:
            values["finished_at"] = _now()
        with self.engine.begin() as connection:
            result = connection.execute(
                update(portfolio_batches).where(portfolio_batches.c.id == batch_id).values(**values)
            )
            if not result.rowcount:
                raise KeyError(batch_id)

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(portfolio_batches).where(portfolio_batches.c.id == batch_id)
            ).first()
        if row is None:
            raise KeyError(batch_id)
        return self._batch_row(row)

    def apply_batch(self, batch_id: str, result: dict[str, Any]) -> dict[str, Any]:
        now = _now()
        with self.engine.begin() as connection:
            batch = connection.execute(
                select(portfolio_batches)
                .where(portfolio_batches.c.id == batch_id)
                .with_for_update()
            ).first()
            if batch is None:
                raise KeyError(batch_id)
            if batch.status == "succeeded":
                return self._batch_row(batch)
            if batch.status not in {"queued", "running"}:
                raise ValueError(f"cannot apply a {batch.status} portfolio batch")
            if result.get("status") != "ok":
                raise ValueError(result.get("error") or "paper rebalance did not succeed")
            if result.get("as_of_date") != batch.as_of_date.isoformat():
                raise ValueError("result as_of_date does not match the batch")
            trade_date = date.fromisoformat(str(result["trade_date"]))
            if trade_date <= batch.as_of_date:
                raise ValueError("trade date must be after the signal date")

            portfolio = connection.execute(
                select(paper_portfolios)
                .where(paper_portfolios.c.id == batch.portfolio_id)
                .with_for_update()
            ).one()
            version = connection.execute(
                select(strategy_versions).where(
                    strategy_versions.c.id == portfolio.strategy_version_id
                )
            ).one()
            if result.get("strategy_version_id") != str(portfolio.strategy_version_id):
                raise ValueError("rebalance result strategy version does not match the portfolio")
            if result.get("signal_engine") != "qlib_governed_signal":
                raise ValueError("paper rebalance requires the governed Qlib signal engine")
            if batch.dataset_identity_sha256:
                provenance = dict(result.get("provenance") or {})
                if provenance.get("daily_dataset_identity_sha256") != batch.dataset_identity_sha256:
                    raise ValueError("paper result changed the batch-pinned Qlib dataset")
                if batch.dataset_lineage_id and (
                    provenance.get("daily_dataset_lineage_id") != batch.dataset_lineage_id
                ):
                    raise ValueError("paper result left the batch-pinned Qlib lineage")
            config = dict(version.config_json)
            cash = _decimal(portfolio.cash)
            starting_nav = _decimal(portfolio.nav)
            positions = {
                str(row.instrument): {
                    "quantity": _decimal(row.quantity),
                    "avg_cost": _decimal(row.avg_cost),
                    "market_price": _decimal(row.market_price),
                    "realized_pnl": _decimal(row.realized_pnl),
                    "industry": row.industry,
                    "take_profit_stage": int(row.take_profit_stage),
                }
                for row in connection.execute(
                    select(paper_positions)
                    .where(paper_positions.c.portfolio_id == portfolio.id)
                    .with_for_update()
                )
            }
            starting_positions = {instrument: dict(item) for instrument, item in positions.items()}
            orders = {str(item["instrument"]): item for item in result.get("orders", [])}
            if len(orders) != len(result.get("orders", [])):
                raise ValueError("a batch may contain only one order per instrument")
            fills = {str(item["instrument"]): item for item in result.get("fills", [])}
            if not set(fills).issubset(orders):
                raise ValueError("every fill must reference a batch order")
            order_ids: dict[str, str] = {}
            for instrument, item in orders.items():
                side = str(item["side"]).lower()
                quantity = _decimal(item["quantity"])
                if side not in {"buy", "sell"} or quantity <= 0:
                    raise ValueError(f"invalid order for {instrument}")
                order_id = uuid.uuid4().hex
                order_ids[instrument] = order_id
                filled = instrument in fills
                connection.execute(
                    insert(paper_orders).values(
                        id=order_id,
                        batch_id=batch_id,
                        portfolio_id=portfolio.id,
                        instrument=instrument,
                        side=side,
                        order_type=str(item.get("order_type") or "market"),
                        target_weight=float(item.get("target_weight", 0.0)),
                        requested_quantity=quantity,
                        status="filled" if filled else "rejected",
                        reason=(
                            str(item["reason"])
                            if item.get("reason")
                            else None
                            if filled
                            else "not executable"
                        ),
                        created_at=now,
                        updated_at=now,
                    )
                )

            gross_traded = Decimal("0")
            fees = Decimal("0")
            for instrument, fill in fills.items():
                order = orders[instrument]
                side = str(order["side"]).lower()
                quantity = _decimal(fill["quantity"])
                price = _decimal(fill["price"])
                fee = _decimal(fill.get("fee", 0))
                if quantity <= 0 or price <= 0 or fee < 0:
                    raise ValueError(f"invalid fill for {instrument}")
                gross = quantity * price
                current = positions.get(
                    instrument,
                    {
                        "quantity": Decimal("0"),
                        "avg_cost": Decimal("0"),
                        "market_price": price,
                        "realized_pnl": Decimal("0"),
                        "industry": None,
                        "take_profit_stage": 0,
                    },
                )
                if side == "buy":
                    required = gross + fee
                    if required > cash + Decimal("0.000001"):
                        raise ValueError(f"insufficient cash for {instrument}")
                    previous_cost = current["quantity"] * current["avg_cost"]
                    current["quantity"] += quantity
                    current["avg_cost"] = (previous_cost + gross + fee) / current["quantity"]
                    current["take_profit_stage"] = 0
                    cash -= required
                else:
                    if quantity > current["quantity"] + Decimal("0.000001"):
                        raise ValueError(f"sell quantity exceeds position for {instrument}")
                    current["quantity"] -= quantity
                    current["realized_pnl"] += (price - current["avg_cost"]) * quantity - fee
                    cash += gross - fee
                current["market_price"] = price
                positions[instrument] = current
                gross_traded += gross
                fees += fee
                fill_time = datetime.fromisoformat(str(fill["fill_time"]))
                connection.execute(
                    insert(paper_fills).values(
                        id=uuid.uuid4().hex,
                        order_id=order_ids[instrument],
                        fill_time=fill_time,
                        quantity=quantity,
                        price=price,
                        gross_value=gross,
                        fee=fee,
                        slippage=float(fill.get("slippage", 0.0)),
                        created_at=now,
                    )
                )

            marks = {
                key: _decimal(value) for key, value in result.get("closing_prices", {}).items()
            }
            industries = {
                str(key): str(value)
                for key, value in result.get("industries", {}).items()
                if str(value).strip()
            }
            take_profit_stages = {
                str(key): int(value) for key, value in result.get("take_profit_stages", {}).items()
            }
            for instrument, mark in marks.items():
                if instrument in positions and mark > 0:
                    positions[instrument]["market_price"] = mark
            for instrument, industry in industries.items():
                if instrument in positions:
                    positions[instrument]["industry"] = industry
            for instrument, stage in take_profit_stages.items():
                if instrument in positions and instrument in fills:
                    positions[instrument]["take_profit_stage"] = stage
            positions = {
                instrument: item
                for instrument, item in positions.items()
                if item["quantity"] > Decimal("0.000001")
            }
            market_value = sum(
                (item["quantity"] * item["market_price"] for item in positions.values()),
                Decimal("0"),
            )
            nav = cash + market_value
            if nav <= 0:
                raise ValueError("portfolio NAV must remain positive")
            high_water_mark = max(_decimal(portfolio.high_water_mark), nav)
            daily_return = float(nav / starting_nav - 1) if starting_nav > 0 else 0.0
            drawdown = float(nav / high_water_mark - 1)
            exposure = float(market_value / nav)
            turnover = float(gross_traded / starting_nav) if starting_nav > 0 else 0.0

            connection.execute(
                delete(paper_positions).where(paper_positions.c.portfolio_id == portfolio.id)
            )
            max_weight = 0.0
            industry_weights: dict[str, float] = {}
            for instrument, item in positions.items():
                value = item["quantity"] * item["market_price"]
                weight = float(value / nav)
                max_weight = max(max_weight, weight)
                industry = item.get("industry")
                if industry:
                    industry_weights[str(industry)] = (
                        industry_weights.get(str(industry), 0.0) + weight
                    )
                connection.execute(
                    insert(paper_positions).values(
                        portfolio_id=portfolio.id,
                        instrument=instrument,
                        industry=industry,
                        take_profit_stage=int(item.get("take_profit_stage") or 0),
                        quantity=item["quantity"],
                        avg_cost=item["avg_cost"],
                        market_price=item["market_price"],
                        market_value=value,
                        weight=weight,
                        realized_pnl=item["realized_pnl"],
                        unrealized_pnl=(item["market_price"] - item["avg_cost"]) * item["quantity"],
                        updated_at=now,
                    )
                )
            connection.execute(
                insert(portfolio_nav).values(
                    portfolio_id=portfolio.id,
                    trade_date=trade_date,
                    cash=cash,
                    market_value=market_value,
                    nav=nav,
                    daily_return=daily_return,
                    benchmark_return=result.get("benchmark_return"),
                    drawdown=drawdown,
                    exposure=exposure,
                    turnover=turnover,
                    fees=fees,
                    created_at=now,
                )
            )

            breaches = [
                ("max_position_weight", max_weight, float(config["max_position_weight"])),
                ("max_daily_turnover", turnover, float(config["max_daily_turnover"])),
                (
                    "max_daily_loss",
                    abs(min(daily_return, 0.0)),
                    float(config.get("max_daily_loss", 0.03)),
                ),
            ]
            max_industry = max(industry_weights.values(), default=0.0)
            breaches.append(
                (
                    "max_industry_weight",
                    max_industry,
                    float(config.get("max_industry_weight", 0.30)),
                )
            )
            hard_breach = False
            risk_action = str(result.get("risk_action") or "")
            for event in result.get("risk_events", []):
                severity = str(event.get("severity") or "warning")
                if severity in {"critical", "hard"}:
                    hard_breach = True
                connection.execute(
                    insert(risk_events).values(
                        portfolio_id=portfolio.id,
                        batch_id=batch_id,
                        severity=severity,
                        event_type=str(event.get("event_type") or "pre_trade_control"),
                        rule=str(event["rule"]),
                        observed=event.get("observed"),
                        limit_value=event.get("limit_value"),
                        status=str(event.get("status") or "open"),
                        details_json=dict(event.get("details") or {}),
                        created_at=now,
                    )
                )
            for rule, observed, limit in breaches:
                if observed > limit + 1e-9:
                    hard_breach = True
                    connection.execute(
                        insert(risk_events).values(
                            portfolio_id=portfolio.id,
                            batch_id=batch_id,
                            severity="critical",
                            event_type="limit_breach",
                            rule=rule,
                            observed=observed,
                            limit_value=limit,
                            status="open",
                            details_json={"action": "portfolio_paused"},
                            created_at=now,
                        )
                    )
            posttrade_action = ""
            drawdown_loss = abs(min(drawdown, 0.0))
            if not risk_action and positions:
                liquidation_limit = float(config.get("max_drawdown_liquidate", 0.15))
                reduction_limit = float(config.get("max_drawdown_reduce", 0.10))
                if drawdown_loss > liquidation_limit + 1e-9:
                    posttrade_action = "liquidate"
                    connection.execute(
                        insert(risk_events).values(
                            portfolio_id=portfolio.id,
                            batch_id=batch_id,
                            severity="critical",
                            event_type="circuit_breaker",
                            rule="max_drawdown_liquidate",
                            observed=drawdown_loss,
                            limit_value=liquidation_limit,
                            status="open",
                            details_json={"action": "liquidation_scheduled_next_open"},
                            created_at=now,
                        )
                    )
                elif drawdown_loss > reduction_limit + 1e-9:
                    posttrade_action = "reduce"
                    connection.execute(
                        insert(risk_events).values(
                            portfolio_id=portfolio.id,
                            batch_id=batch_id,
                            severity="critical",
                            event_type="circuit_breaker",
                            rule="max_drawdown_reduce",
                            observed=drawdown_loss,
                            limit_value=reduction_limit,
                            status="open",
                            details_json={"action": "risk_reduction_scheduled_next_open"},
                            created_at=now,
                        )
                    )
            if risk_action == "liquidate":
                next_status = "liquidation_pending" if positions else "paused"
            elif risk_action == "reduce":
                exposure_limit = float(config.get("drawdown_reduction_exposure", 0.50))
                next_status = (
                    "risk_reduction_pending"
                    if positions and exposure > exposure_limit + 0.01
                    else "paused"
                )
            elif posttrade_action == "liquidate":
                next_status = "liquidation_pending"
            elif posttrade_action == "reduce":
                next_status = "risk_reduction_pending"
            elif hard_breach or portfolio.status in {
                "liquidation_pending",
                "risk_reduction_pending",
            }:
                next_status = "paused"
            else:
                next_status = portfolio.status
            risk_rows = connection.execute(
                select(risk_events.c.severity, risk_events.c.rule).where(
                    risk_events.c.batch_id == batch_id
                )
            ).all()
            critical_risks = sum(str(item.severity) in {"critical", "hard"} for item in risk_rows)
            contributors = _pnl_contributors(
                starting_positions,
                positions,
                orders,
                fills,
            )
            best = sorted(contributors, key=lambda item: item["pnl"], reverse=True)[:5]
            worst = sorted(contributors, key=lambda item: item["pnl"])[:5]
            net_pnl = nav - starting_nav
            market_pnl = net_pnl + fees
            attributed_pnl = sum(_decimal(item["pnl"]) for item in contributors)
            rejected = [
                {
                    "instrument": instrument,
                    "reason": str(item.get("reason") or "not executable"),
                }
                for instrument, item in orders.items()
                if instrument not in fills
            ]
            review_status = (
                "action_required"
                if next_status != "active" or critical_risks
                else "attention"
                if rejected or risk_rows
                else "ok"
            )
            benchmark_return = result.get("benchmark_return")
            summary = {
                "starting_nav": float(starting_nav),
                "ending_nav": float(nav),
                "net_pnl": float(net_pnl),
                "market_pnl_before_fees": float(market_pnl),
                "attributed_market_pnl": float(attributed_pnl),
                "attribution_gap": float(market_pnl - attributed_pnl),
                "daily_return": daily_return,
                "benchmark_return": benchmark_return,
                "active_return": daily_return - float(benchmark_return)
                if benchmark_return is not None
                else None,
                "drawdown": drawdown,
                "exposure": exposure,
                "turnover": turnover,
                "fees": float(fees),
                "max_position_weight": max_weight,
                "max_industry_weight": max_industry,
                "requested_orders": len(orders),
                "filled_orders": len(fills),
                "rejected_orders": len(rejected),
                "fill_rate": len(fills) / len(orders) if orders else 1.0,
                "risk_event_count": len(risk_rows),
                "critical_risk_count": critical_risks,
                "next_portfolio_status": next_status,
                "best_contributors": best,
                "worst_contributors": worst,
                "rejections": rejected[:20],
                "risk_rules": sorted({str(item.rule) for item in risk_rows}),
            }
            connection.execute(
                insert(portfolio_reviews).values(
                    id=uuid.uuid4().hex,
                    portfolio_id=portfolio.id,
                    batch_id=batch_id,
                    trade_date=trade_date,
                    status=review_status,
                    summary_json=summary,
                    created_at=now,
                )
            )
            connection.execute(
                update(paper_portfolios)
                .where(paper_portfolios.c.id == portfolio.id)
                .values(
                    status=next_status,
                    cash=cash,
                    nav=nav,
                    high_water_mark=high_water_mark,
                    updated_at=now,
                )
            )
            synchronize_portfolio_schedules(
                connection,
                str(portfolio.id),
                next_status,
                now=now,
            )
            connection.execute(
                update(portfolio_batches)
                .where(portfolio_batches.c.id == batch_id)
                .values(
                    status="succeeded",
                    trade_date=trade_date,
                    error=None,
                    finished_at=now,
                )
            )
        return self.get_batch(batch_id)

    @staticmethod
    def _batch_row(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        for key in ("as_of_date", "trade_date"):
            if result.get(key) is not None:
                result[key] = result[key].isoformat()
        return result

    @staticmethod
    def _risk_row(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["details"] = result.pop("details_json")
        return result

    @staticmethod
    def _review_row(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["summary"] = result.pop("summary_json")
        if result.get("trade_date") is not None:
            result["trade_date"] = result["trade_date"].isoformat()
        return result
