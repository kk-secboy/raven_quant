from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    backtest_runs,
    recommendation_nav,
    recommendation_portfolios,
    row_dict,
    strategies,
    strategy_allocation_events,
    strategy_allocation_members,
    strategy_allocation_nav,
    strategy_allocations,
)

from .strategy_allocation import analyze_strategy_allocation
from .strategy_store import StrategyStore


def _now() -> datetime:
    return datetime.now(UTC)


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


class AllocationStore:
    """Low-correlation strategy allocations backed only by recommendation ledgers."""

    def __init__(self, database_url: str) -> None:
        self.strategies = StrategyStore(database_url)
        self.engine = self.strategies.engine

    @staticmethod
    def _daily_returns(backtest: dict[str, Any]) -> pd.Series:
        root = Path(str(backtest["artifact_path"]))
        candidates = (
            root / str(backtest["id"]) / "daily_returns.parquet",
            root / "daily_returns.parquet",
        )
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise ValueError(f"backtest {backtest['id']} has no daily return artifact")
        frame = pd.read_parquet(source)
        date_column = next(
            (name for name in ("datetime", "trade_date", "date") if name in frame.columns),
            None,
        )
        if date_column is None:
            raise ValueError(f"backtest {backtest['id']} daily artifact has no date")
        if "net_return" in frame:
            values = pd.to_numeric(frame["net_return"], errors="coerce")
        elif {"return", "cost"}.issubset(frame.columns):
            values = pd.to_numeric(frame["return"], errors="coerce") - pd.to_numeric(
                frame["cost"], errors="coerce"
            )
        else:
            raise ValueError(f"backtest {backtest['id']} daily artifact is incomplete")
        timestamps = pd.to_datetime(frame[date_column], errors="coerce")
        series = pd.Series(values.to_numpy(), index=timestamps).dropna().sort_index()
        if series.index.duplicated().any():
            raise ValueError(f"backtest {backtest['id']} has duplicate return dates")
        return series

    def create(
        self,
        *,
        name: str,
        strategy_version_ids: list[str],
        dataset: str,
        total_capital: float,
        allocation_method: str,
        lookback_days: int,
        target_volatility: float,
        max_pairwise_correlation: float,
        max_strategy_weight: float,
        max_member_drawdown: float,
        max_drawdown_reduce: float,
        max_drawdown_liquidate: float,
        fixed_weights: dict[str, float] | None,
        actor: str,
    ) -> dict[str, Any]:
        version_ids = list(
            dict.fromkeys(item.strip() for item in strategy_version_ids if item.strip())
        )
        if len(version_ids) < 2 or len(version_ids) > 10:
            raise ValueError("a strategy allocation requires 2 to 10 unique strategy versions")
        if not name.strip() or not dataset.strip() or not actor.strip():
            raise ValueError("name, dataset and actor are required")
        if total_capital < 500_000:
            raise ValueError("strategy allocation capital must be at least 500000")
        if not 0 < max_member_drawdown < max_drawdown_reduce < max_drawdown_liquidate <= 0.50:
            raise ValueError("member, reduction and liquidation drawdowns must be increasing")

        returns: dict[str, pd.Series] = {}
        members: list[dict[str, Any]] = []
        with self.engine.connect() as connection:
            for version_id in version_ids:
                version = self.strategies.get_version(version_id)
                if version["status"] != "approved" or version.get("is_legacy"):
                    raise ValueError("allocations require approved non-legacy strategy versions")
                backtest_row = connection.execute(
                    select(backtest_runs)
                    .where(
                        backtest_runs.c.strategy_version_id == version_id,
                        backtest_runs.c.status == "succeeded",
                        backtest_runs.c.dataset == dataset,
                        backtest_runs.c.is_legacy.is_(False),
                    )
                    .order_by(backtest_runs.c.finished_at.desc())
                    .limit(1)
                ).first()
                if backtest_row is None:
                    raise ValueError(
                        f"strategy version {version_id} has no formal {dataset} backtest"
                    )
                backtest = row_dict(backtest_row)
                strategy_name = connection.execute(
                    select(strategies.c.name).where(strategies.c.id == version["strategy_id"])
                ).scalar_one()
                returns[version_id] = self._daily_returns(backtest)
                members.append(
                    {
                        "strategy_version_id": version_id,
                        "strategy_name": str(strategy_name),
                        "strategy_version": version["version"],
                        "backtest_id": backtest["id"],
                    }
                )
        analysis = analyze_strategy_allocation(
            pd.concat(returns, axis=1, join="inner"),
            method=allocation_method,
            lookback_days=lookback_days,
            target_volatility=target_volatility,
            max_pairwise_correlation=max_pairwise_correlation,
            max_strategy_weight=max_strategy_weight,
            fixed_weights=fixed_weights,
        )
        allocation_id = uuid.uuid4().hex
        now = _now()
        capital = _decimal(total_capital)
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(strategy_allocations).values(
                        id=allocation_id,
                        name=name.strip(),
                        dataset=dataset.strip(),
                        status="draft",
                        is_legacy=False,
                        allocation_method=allocation_method,
                        lookback_days=lookback_days,
                        target_volatility=target_volatility,
                        max_pairwise_correlation=max_pairwise_correlation,
                        max_strategy_weight=max_strategy_weight,
                        max_member_drawdown=max_member_drawdown,
                        max_drawdown_reduce=max_drawdown_reduce,
                        max_drawdown_liquidate=max_drawdown_liquidate,
                        total_capital=capital,
                        cash_reserve=capital * _decimal(analysis["cash_weight"]),
                        nav=capital,
                        high_water_mark=capital,
                        analysis_json=analysis,
                        created_by=actor.strip(),
                        created_at=now,
                        updated_at=now,
                    )
                )
                for member in members:
                    evidence = analysis["members"][member["strategy_version_id"]]
                    connection.execute(
                        insert(strategy_allocation_members).values(
                            allocation_id=allocation_id,
                            strategy_version_id=member["strategy_version_id"],
                            backtest_id=member["backtest_id"],
                            target_weight=evidence["target_weight"],
                            annualized_volatility=evidence["annualized_volatility"],
                            risk_contribution=evidence["risk_contribution"],
                            created_at=now,
                        )
                    )
                self._event(
                    connection,
                    allocation_id,
                    event_type="allocation.created",
                    severity="info",
                    rule="low_correlation_risk_budget",
                    observed=analysis["highest_pairwise_correlation"],
                    limit=max_pairwise_correlation,
                    details={"actor": actor.strip(), "members": members},
                )
        except IntegrityError as exc:
            raise ValueError(f"strategy allocation name {name!r} already exists") from exc
        return self.get(allocation_id)

    def approve(self, allocation_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        if not actor.strip() or len(reason.strip()) < 10:
            raise ValueError("actor and a meaningful approval reason are required")
        now = _now()
        with self.engine.begin() as connection:
            allocation = connection.execute(
                select(strategy_allocations)
                .where(strategy_allocations.c.id == allocation_id)
                .with_for_update()
            ).first()
            if allocation is None:
                raise KeyError(allocation_id)
            if allocation.is_legacy:
                raise ValueError("legacy paper-backed allocations are read-only")
            if allocation.status != "draft":
                raise ValueError("only draft strategy allocations may be approved")
            if actor.strip() == allocation.created_by:
                raise ValueError("strategy allocation approval requires a second operator")
            members = connection.execute(
                select(strategy_allocation_members).where(
                    strategy_allocation_members.c.allocation_id == allocation_id
                )
            ).all()
            for member in members:
                version = self.strategies.get_version(str(member.strategy_version_id))
                strategy_name = connection.execute(
                    select(strategies.c.name).where(strategies.c.id == version["strategy_id"])
                ).scalar_one()
                initial_value = _decimal(allocation.total_capital) * _decimal(member.target_weight)
                if initial_value < Decimal("100000"):
                    raise ValueError(f"allocated value for {strategy_name} is below 100000")
                portfolio_id = uuid.uuid4().hex
                connection.execute(
                    insert(recommendation_portfolios).values(
                        id=portfolio_id,
                        name=f"{allocation.name} / {strategy_name} v{version['version']}"[:150],
                        strategy_version_id=member.strategy_version_id,
                        dataset=allocation.dataset,
                        status="active",
                        base_currency="CNY",
                        hypothetical_initial_value=initial_value,
                        risk_exposure_override=1.0,
                        created_by=allocation.created_by,
                        created_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    update(strategy_allocation_members)
                    .where(
                        strategy_allocation_members.c.allocation_id == allocation_id,
                        strategy_allocation_members.c.strategy_version_id
                        == member.strategy_version_id,
                    )
                    .values(recommendation_portfolio_id=portfolio_id)
                )
            connection.execute(
                update(strategy_allocations)
                .where(strategy_allocations.c.id == allocation_id)
                .values(
                    status="active",
                    approved_by=actor.strip(),
                    approval_reason=reason.strip(),
                    approved_at=now,
                    updated_at=now,
                )
            )
            self._event(
                connection,
                allocation_id,
                event_type="allocation.approved",
                severity="info",
                rule="four_eyes_approval",
                details={"actor": actor.strip(), "reason": reason.strip()},
            )
        return self.get(allocation_id)

    def set_status(self, allocation_id: str, status: str, *, actor: str) -> dict[str, Any]:
        if status not in {"active", "paused"}:
            raise ValueError("strategy allocation status must be active or paused")
        if len(actor.strip()) < 2:
            raise ValueError("a responsible actor is required")
        now = _now()
        with self.engine.begin() as connection:
            allocation = connection.execute(
                select(strategy_allocations).where(strategy_allocations.c.id == allocation_id)
            ).first()
            if allocation is None:
                raise KeyError(allocation_id)
            if allocation.is_legacy:
                raise ValueError("legacy paper-backed allocations are read-only")
            if allocation.status == "draft":
                raise ValueError("draft strategy allocations cannot be operated")
            if status == "active":
                unresolved = connection.execute(
                    select(strategy_allocation_events.c.id)
                    .where(
                        strategy_allocation_events.c.allocation_id == allocation_id,
                        strategy_allocation_events.c.severity == "critical",
                        strategy_allocation_events.c.status.in_(["open", "acknowledged"]),
                    )
                    .limit(1)
                ).first()
                if unresolved is not None:
                    raise ValueError("critical allocation events must be resolved first")
            portfolio_ids = self._portfolio_ids(connection, allocation_id)
            connection.execute(
                update(recommendation_portfolios)
                .where(recommendation_portfolios.c.id.in_(portfolio_ids))
                .values(
                    status=status,
                    risk_exposure_override=1.0 if status == "active" else 0.0,
                    updated_at=now,
                )
            )
            connection.execute(
                update(strategy_allocations)
                .where(strategy_allocations.c.id == allocation_id)
                .values(status=status, updated_at=now)
            )
            self._event(
                connection,
                allocation_id,
                event_type=f"allocation.{status}",
                severity="info",
                rule="operator_state_change",
                details={"actor": actor.strip()},
            )
        return self.get(allocation_id)

    def refresh(self, allocation_id: str) -> dict[str, Any]:
        now = _now()
        with self.engine.begin() as connection:
            allocation = connection.execute(
                select(strategy_allocations)
                .where(strategy_allocations.c.id == allocation_id)
                .with_for_update()
            ).first()
            if allocation is None:
                raise KeyError(allocation_id)
            if allocation.is_legacy:
                raise ValueError("legacy paper-backed allocations are read-only")
            if allocation.status not in {"active", "risk_reduction_pending", "liquidation_pending"}:
                raise ValueError("strategy allocation is not refreshable")
            members = connection.execute(
                select(strategy_allocation_members).where(
                    strategy_allocation_members.c.allocation_id == allocation_id
                )
            ).all()
            latest: dict[str, Any] = {}
            for member in members:
                portfolio_id = member.recommendation_portfolio_id
                if not portfolio_id:
                    raise ValueError(
                        "strategy allocation has an unprovisioned recommendation member"
                    )
                nav_row = connection.execute(
                    select(recommendation_nav)
                    .where(recommendation_nav.c.portfolio_id == portfolio_id)
                    .order_by(recommendation_nav.c.trade_date.desc())
                    .limit(1)
                ).first()
                if nav_row is None:
                    return {
                        "id": allocation_id,
                        "status": allocation.status,
                        "refresh_status": "waiting_for_member_nav",
                    }
                latest[str(portfolio_id)] = nav_row
            dates = {item.trade_date for item in latest.values()}
            if len(dates) != 1:
                return {
                    "id": allocation_id,
                    "status": allocation.status,
                    "refresh_status": "waiting_for_aligned_trade_date",
                }
            trade_date = next(iter(dates))
            if connection.execute(
                select(strategy_allocation_nav.c.trade_date).where(
                    strategy_allocation_nav.c.allocation_id == allocation_id,
                    strategy_allocation_nav.c.trade_date == trade_date,
                )
            ).first():
                return {
                    "id": allocation_id,
                    "status": allocation.status,
                    "refresh_status": "already_recorded",
                }
            member_nav = {
                portfolio_id: float(item.hypothetical_value)
                for portfolio_id, item in latest.items()
            }
            nav = _decimal(allocation.cash_reserve) + sum(
                (_decimal(value) for value in member_nav.values()), Decimal("0")
            )
            previous = connection.execute(
                select(strategy_allocation_nav.c.nav)
                .where(strategy_allocation_nav.c.allocation_id == allocation_id)
                .order_by(strategy_allocation_nav.c.trade_date.desc())
                .limit(1)
            ).scalar_one_or_none()
            previous_nav = _decimal(previous or allocation.total_capital)
            daily_return = float(nav / previous_nav - 1) if previous_nav else 0.0
            high_water_mark = max(_decimal(allocation.high_water_mark), nav)
            drawdown = float(nav / high_water_mark - 1) if high_water_mark else 0.0
            history = list(
                connection.execute(
                    select(strategy_allocation_nav.c.daily_return)
                    .where(strategy_allocation_nav.c.allocation_id == allocation_id)
                    .order_by(strategy_allocation_nav.c.trade_date.desc())
                    .limit(59)
                ).scalars()
            )
            history.append(daily_return)
            volatility = (
                float(np.std(history, ddof=1) * np.sqrt(252.0))
                if len(history) > 1
                else 0.0
            )
            member_weights = {
                key: float(_decimal(value) / nav) if nav else 0.0
                for key, value in member_nav.items()
            }
            drawdown_loss = abs(min(drawdown, 0.0))
            next_status = str(allocation.status)
            override = 1.0
            if drawdown_loss >= float(allocation.max_drawdown_liquidate):
                next_status, override = "liquidation_pending", 0.0
                self._event(
                    connection,
                    allocation_id,
                    event_type="allocation_circuit_breaker",
                    severity="critical",
                    rule="max_drawdown_liquidate",
                    observed=drawdown_loss,
                    limit=float(allocation.max_drawdown_liquidate),
                    details={"trade_date": trade_date.isoformat()},
                )
            elif drawdown_loss >= float(allocation.max_drawdown_reduce):
                next_status, override = "risk_reduction_pending", 0.5
                self._event(
                    connection,
                    allocation_id,
                    event_type="allocation_circuit_breaker",
                    severity="critical",
                    rule="max_drawdown_reduce",
                    observed=drawdown_loss,
                    limit=float(allocation.max_drawdown_reduce),
                    details={"trade_date": trade_date.isoformat()},
                )
            for member in members:
                portfolio_id = str(member.recommendation_portfolio_id)
                member_loss = abs(min(float(latest[portfolio_id].drawdown), 0.0))
                member_override = override
                if member_loss >= float(allocation.max_member_drawdown):
                    member_override = min(member_override, 0.5)
                    next_status = (
                        next_status
                        if next_status == "liquidation_pending"
                        else "risk_reduction_pending"
                    )
                    self._event(
                        connection,
                        allocation_id,
                        recommendation_portfolio_id=portfolio_id,
                        event_type="member_circuit_breaker",
                        severity="critical",
                        rule="max_member_drawdown",
                        observed=member_loss,
                        limit=float(allocation.max_member_drawdown),
                        details={"trade_date": trade_date.isoformat()},
                    )
                connection.execute(
                    update(recommendation_portfolios)
                    .where(recommendation_portfolios.c.id == portfolio_id)
                    .values(risk_exposure_override=member_override, updated_at=now)
                )
            connection.execute(
                insert(strategy_allocation_nav).values(
                    allocation_id=allocation_id,
                    trade_date=trade_date,
                    nav=nav,
                    daily_return=daily_return,
                    annualized_volatility=volatility,
                    drawdown=drawdown,
                    member_nav_json=member_nav,
                    member_weights_json=member_weights,
                    created_at=now,
                )
            )
            connection.execute(
                update(strategy_allocations)
                .where(strategy_allocations.c.id == allocation_id)
                .values(
                    status=next_status,
                    nav=nav,
                    high_water_mark=high_water_mark,
                    updated_at=now,
                )
            )
        result = self.get(allocation_id)
        result["refresh_status"] = "recorded"
        return result

    def acknowledge_event(self, allocation_id: str, event_id: int, *, actor: str) -> dict[str, Any]:
        return self._update_event(allocation_id, event_id, actor=actor, resolution=None)

    def resolve_event(
        self, allocation_id: str, event_id: int, *, actor: str, reason: str
    ) -> dict[str, Any]:
        if len(reason.strip()) < 10:
            raise ValueError("risk event resolution reason must be meaningful")
        return self._update_event(allocation_id, event_id, actor=actor, resolution=reason)

    def _update_event(
        self,
        allocation_id: str,
        event_id: int,
        *,
        actor: str,
        resolution: str | None,
    ) -> dict[str, Any]:
        if len(actor.strip()) < 2:
            raise ValueError("a responsible actor is required")
        now = _now()
        with self.engine.begin() as connection:
            event = connection.execute(
                select(strategy_allocation_events)
                .where(
                    strategy_allocation_events.c.id == event_id,
                    strategy_allocation_events.c.allocation_id == allocation_id,
                )
                .with_for_update()
            ).first()
            if event is None:
                raise KeyError(event_id)
            if event.status == "resolved":
                raise ValueError("risk event is already resolved")
            values: dict[str, Any] = {
                "status": "resolved" if resolution is not None else "acknowledged",
                "acknowledged_by": actor.strip(),
                "acknowledged_at": now,
            }
            if resolution is not None:
                values.update(
                    resolved_by=actor.strip(), resolved_at=now, resolution_reason=resolution.strip()
                )
            connection.execute(
                update(strategy_allocation_events)
                .where(strategy_allocation_events.c.id == event_id)
                .values(**values)
            )
        return self.get_event(allocation_id, event_id)

    def get_event(self, allocation_id: str, event_id: int) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(strategy_allocation_events).where(
                    strategy_allocation_events.c.id == event_id,
                    strategy_allocation_events.c.allocation_id == allocation_id,
                )
            ).first()
        if row is None:
            raise KeyError(event_id)
        return self._event_row(row)

    def get(self, allocation_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            allocation = connection.execute(
                select(strategy_allocations).where(strategy_allocations.c.id == allocation_id)
            ).first()
            if allocation is None:
                raise KeyError(allocation_id)
            result = row_dict(allocation)
            result["analysis"] = result.pop("analysis_json")
            result["members"] = [
                row_dict(item)
                for item in connection.execute(
                    select(strategy_allocation_members).where(
                        strategy_allocation_members.c.allocation_id == allocation_id
                    )
                )
            ]
            result["nav_history"] = [
                row_dict(item)
                for item in connection.execute(
                    select(strategy_allocation_nav)
                    .where(strategy_allocation_nav.c.allocation_id == allocation_id)
                    .order_by(strategy_allocation_nav.c.trade_date.desc())
                    .limit(500)
                )
            ]
            result["events"] = [
                self._event_row(item)
                for item in connection.execute(
                    select(strategy_allocation_events)
                    .where(strategy_allocation_events.c.allocation_id == allocation_id)
                    .order_by(strategy_allocation_events.c.created_at.desc())
                    .limit(500)
                )
            ]
        return result

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            ids = [
                str(item)
                for item in connection.execute(
                    select(strategy_allocations.c.id)
                    .order_by(strategy_allocations.c.updated_at.desc())
                    .limit(limit)
                ).scalars()
            ]
        return [self.get(item) for item in ids]

    def refresh_for_portfolio(self, portfolio_id: str) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            allocation_ids = list(
                connection.execute(
                    select(strategy_allocation_members.c.allocation_id).where(
                        strategy_allocation_members.c.recommendation_portfolio_id == portfolio_id
                    )
                ).scalars()
            )
        return [self.refresh(str(item)) for item in allocation_ids]

    @staticmethod
    def _portfolio_ids(connection: Any, allocation_id: str) -> list[str]:
        rows = connection.execute(
            select(strategy_allocation_members.c.recommendation_portfolio_id).where(
                strategy_allocation_members.c.allocation_id == allocation_id
            )
        ).scalars()
        result = [str(item) for item in rows if item]
        if not result:
            raise ValueError("strategy allocation has no recommendation portfolios")
        return result

    @staticmethod
    def _event(
        connection: Any,
        allocation_id: str,
        *,
        event_type: str,
        severity: str,
        rule: str,
        observed: float | None = None,
        limit: float | None = None,
        recommendation_portfolio_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        connection.execute(
            insert(strategy_allocation_events).values(
                allocation_id=allocation_id,
                recommendation_portfolio_id=recommendation_portfolio_id,
                severity=severity,
                event_type=event_type,
                rule=rule,
                observed=observed,
                limit_value=limit,
                status="open" if severity == "critical" else "resolved",
                details_json=details or {},
                created_at=_now(),
            )
        )

    @staticmethod
    def _event_row(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["details"] = result.pop("details_json")
        return result
