from __future__ import annotations

import uuid
from datetime import UTC, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func, insert, or_, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    allocation_schedule_groups,
    allocation_schedule_members,
    open_database,
    pair_paper_portfolios,
    paper_portfolios,
    row_dict,
    schedule_runs,
    schedules,
    strategy_allocation_members,
    strategy_allocations,
)


def _now() -> datetime:
    return datetime.now(UTC)


def next_occurrence(now: datetime, timezone: str, run_time: time) -> datetime:
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown timezone: {timezone}") from exc
    local = now.astimezone(zone)
    candidate = datetime.combine(local.date(), run_time, tzinfo=zone)
    if candidate <= local:
        candidate += timedelta(days=1)
    return candidate.astimezone(UTC)


def occurrence_after(scheduled_for: datetime, timezone: str, run_time: time) -> datetime:
    zone = ZoneInfo(timezone)
    next_date = scheduled_for.astimezone(zone).date() + timedelta(days=1)
    return datetime.combine(next_date, run_time, tzinfo=zone).astimezone(UTC)


RISK_EXECUTION_STATUSES = {"liquidation_pending", "risk_reduction_pending"}
RUNNABLE_PORTFOLIO_STATUSES = {"active", *RISK_EXECUTION_STATUSES}
RUNNABLE_PAIR_PORTFOLIO_STATUSES = {"active", "liquidation_pending"}


def _lock_portfolio_schedule(connection: Any, portfolio_id: str) -> None:
    connection.execute(
        select(func.pg_advisory_xact_lock(func.hashtext(f"quantlab:paper-schedule:{portfolio_id}")))
    )


def _assert_no_conflicting_portfolio_schedule(
    connection: Any,
    portfolio_id: str,
    *,
    exclude_schedule_id: str | None = None,
) -> None:
    statement = select(schedules.c.id).where(
        schedules.c.kind == "paper_rebalance",
        schedules.c.payload_json["portfolio_id"].as_string() == portfolio_id,
        schedules.c.status != "retired",
    )
    if exclude_schedule_id is not None:
        statement = statement.where(schedules.c.id != exclude_schedule_id)
    if connection.execute(statement.limit(1)).first() is not None:
        raise ValueError("portfolio already has a non-retired paper schedule")


def synchronize_portfolio_schedules(
    connection: Any,
    portfolio_id: str,
    portfolio_status: str,
    *,
    now: datetime,
) -> None:
    """Apply portfolio safety state without overwriting operator schedule intent."""
    rows = connection.execute(
        select(schedules).where(
            schedules.c.kind == "paper_rebalance",
            schedules.c.payload_json["portfolio_id"].as_string() == portfolio_id,
        )
    ).all()
    suspension = (
        None if portfolio_status in RUNNABLE_PORTFOLIO_STATUSES else f"portfolio:{portfolio_status}"
    )
    for row in rows:
        desired = str(row.desired_status)
        effective = desired if desired == "retired" or suspension is None else "paused"
        values: dict[str, Any] = {
            "status": effective,
            "suspension_reason": suspension,
            "updated_at": now,
        }
        if effective == "active":
            values["next_run_at"] = next_occurrence(now, row.timezone, row.run_time)
        connection.execute(update(schedules).where(schedules.c.id == row.id).values(**values))


def synchronize_pair_portfolio_schedules(
    connection: Any,
    portfolio_id: str,
    portfolio_status: str,
    *,
    now: datetime,
) -> None:
    rows = connection.execute(
        select(schedules).where(
            schedules.c.kind == "pair_paper_rebalance",
            schedules.c.payload_json["pair_portfolio_id"].as_string() == portfolio_id,
        )
    ).all()
    suspension = (
        None
        if portfolio_status in RUNNABLE_PAIR_PORTFOLIO_STATUSES
        else f"pair_portfolio:{portfolio_status}"
    )
    for row in rows:
        desired = str(row.desired_status)
        effective = desired if desired == "retired" or suspension is None else "paused"
        values: dict[str, Any] = {
            "status": effective,
            "suspension_reason": suspension,
            "updated_at": now,
        }
        if effective == "active":
            values["next_run_at"] = next_occurrence(now, row.timezone, row.run_time)
        connection.execute(update(schedules).where(schedules.c.id == row.id).values(**values))


class ScheduleStore:
    """PostgreSQL schedule repository with leased, restart-safe run claiming."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def create(
        self,
        *,
        name: str,
        kind: str,
        timezone: str,
        run_time: time,
        trading_days_only: bool,
        payload: dict[str, Any],
        misfire_grace_seconds: int,
        actor: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if kind not in {
            "incremental_sync",
            "data_pipeline",
            "ashare_5m_sync",
            "rdagent_research",
            "paper_rebalance",
            "pair_paper_rebalance",
            "broker_reconcile",
        }:
            raise ValueError("unsupported schedule kind")
        if not name.strip() or not actor.strip():
            raise ValueError("schedule name and actor are required")
        if misfire_grace_seconds < 60:
            raise ValueError("misfire grace must be at least 60 seconds")
        portfolio_id = str(payload.get("portfolio_id") or "") if kind == "paper_rebalance" else None
        pair_portfolio_id = (
            str(payload.get("pair_portfolio_id") or "") if kind == "pair_paper_rebalance" else None
        )
        if kind == "paper_rebalance" and not portfolio_id:
            raise ValueError("paper_rebalance requires portfolio_id")
        if kind == "pair_paper_rebalance" and not pair_portfolio_id:
            raise ValueError("pair_paper_rebalance requires pair_portfolio_id")
        current = now or _now()
        schedule_id = uuid.uuid4().hex
        try:
            with self.engine.begin() as connection:
                if portfolio_id:
                    _lock_portfolio_schedule(connection, portfolio_id)
                    _assert_no_conflicting_portfolio_schedule(connection, portfolio_id)
                if pair_portfolio_id:
                    connection.execute(
                        select(
                            func.pg_advisory_xact_lock(
                                func.hashtext(f"quantlab:pair-paper-schedule:{pair_portfolio_id}")
                            )
                        )
                    )
                    conflict = connection.execute(
                        select(schedules.c.id)
                        .where(
                            schedules.c.kind == "pair_paper_rebalance",
                            schedules.c.payload_json["pair_portfolio_id"].as_string()
                            == pair_portfolio_id,
                            schedules.c.status != "retired",
                        )
                        .limit(1)
                    ).first()
                    if conflict is not None:
                        raise ValueError("pair portfolio already has a non-retired paper schedule")
                connection.execute(
                    insert(schedules).values(
                        id=schedule_id,
                        name=name.strip(),
                        kind=kind,
                        status="active",
                        desired_status="active",
                        suspension_reason=None,
                        timezone=timezone,
                        run_time=run_time,
                        trading_days_only=trading_days_only,
                        payload_json=payload,
                        misfire_grace_seconds=misfire_grace_seconds,
                        next_run_at=next_occurrence(current, timezone, run_time),
                        created_by=actor.strip(),
                        created_at=current,
                        updated_at=current,
                    )
                )
        except IntegrityError as exc:
            raise ValueError(f"schedule name {name!r} already exists") from exc
        return self.get(schedule_id)

    def get(self, schedule_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(select(schedules).where(schedules.c.id == schedule_id)).first()
        if row is None:
            raise KeyError(schedule_id)
        return self._schedule_row(row)

    def get_by_name(self, name: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(schedules).where(schedules.c.name == name)
            ).first()
        return self._schedule_row(row) if row else None

    def list(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(schedules).order_by(schedules.c.created_at.desc()).limit(limit)
            )
            return [self._schedule_row(row) for row in rows]

    def set_status(
        self,
        schedule_id: str,
        status: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if status not in {"active", "paused"}:
            raise ValueError("schedule status must be active or paused")
        current = now or _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(schedules).where(schedules.c.id == schedule_id).with_for_update()
            ).first()
            if row is None:
                raise KeyError(schedule_id)
            allocation_id = connection.execute(
                select(allocation_schedule_members.c.allocation_id).where(
                    allocation_schedule_members.c.schedule_id == schedule_id
                )
            ).scalar_one_or_none()
            if allocation_id is not None:
                raise ValueError(
                    "allocation-managed schedules must be controlled through their group"
                )
            suspension = row.suspension_reason
            if row.kind == "paper_rebalance":
                portfolio_id = str(dict(row.payload_json).get("portfolio_id") or "")
                portfolio_status = connection.execute(
                    select(paper_portfolios.c.status).where(paper_portfolios.c.id == portfolio_id)
                ).scalar_one_or_none()
                if status == "paused" and portfolio_status in RISK_EXECUTION_STATUSES:
                    raise ValueError("a pending risk-execution schedule cannot be paused")
                suspension = (
                    None
                    if portfolio_status in RUNNABLE_PORTFOLIO_STATUSES
                    else f"portfolio:{portfolio_status or 'missing'}"
                )
            elif row.kind == "pair_paper_rebalance":
                pair_portfolio_id = str(dict(row.payload_json).get("pair_portfolio_id") or "")
                portfolio_status = connection.execute(
                    select(pair_paper_portfolios.c.status).where(
                        pair_paper_portfolios.c.id == pair_portfolio_id
                    )
                ).scalar_one_or_none()
                if status == "paused" and portfolio_status == "liquidation_pending":
                    raise ValueError("a pending pair liquidation schedule cannot be paused")
                suspension = (
                    None
                    if portfolio_status in RUNNABLE_PAIR_PORTFOLIO_STATUSES
                    else f"pair_portfolio:{portfolio_status or 'missing'}"
                )
            effective = status if suspension is None else "paused"
            values: dict[str, Any] = {
                "desired_status": status,
                "status": effective,
                "suspension_reason": suspension,
                "updated_at": current,
            }
            if effective == "active":
                values["next_run_at"] = next_occurrence(
                    current,
                    row.timezone,
                    row.run_time,
                )
            connection.execute(
                update(schedules).where(schedules.c.id == schedule_id).values(**values)
            )
        return self.get(schedule_id)

    def create_allocation_group(
        self,
        allocation_id: str,
        *,
        timezone: str,
        run_time: time,
        trading_days_only: bool,
        slippage: float,
        misfire_grace_seconds: int,
        actor: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if not actor.strip():
            raise ValueError("schedule actor is required")
        if run_time < time(15, 10):
            raise ValueError("allocation rebalances must run after the A-share close")
        if not 0 <= slippage <= 0.02:
            raise ValueError("slippage must be between 0 and 0.02")
        if misfire_grace_seconds < 60:
            raise ValueError("misfire grace must be at least 60 seconds")
        current = now or _now()
        next_run = next_occurrence(current, timezone, run_time)
        with self.engine.begin() as connection:
            allocation = connection.execute(
                select(strategy_allocations)
                .where(strategy_allocations.c.id == allocation_id)
                .with_for_update()
            ).first()
            if allocation is None:
                raise KeyError(allocation_id)
            if allocation.status == "draft":
                raise ValueError("approve the strategy allocation before scheduling it")
            members = connection.execute(
                select(
                    strategy_allocation_members.c.portfolio_id,
                    paper_portfolios.c.name,
                    paper_portfolios.c.status,
                )
                .join(
                    paper_portfolios,
                    paper_portfolios.c.id == strategy_allocation_members.c.portfolio_id,
                )
                .where(strategy_allocation_members.c.allocation_id == allocation_id)
            ).all()
            if len(members) < 2 or any(not item.portfolio_id for item in members):
                raise ValueError("every allocation member must have a child portfolio")
            for portfolio_id in sorted(str(item.portfolio_id) for item in members):
                _lock_portfolio_schedule(connection, portfolio_id)
            existing = connection.execute(
                select(allocation_schedule_groups)
                .where(allocation_schedule_groups.c.allocation_id == allocation_id)
                .with_for_update()
            ).first()
            if existing is None:
                connection.execute(
                    insert(allocation_schedule_groups).values(
                        allocation_id=allocation_id,
                        status="active",
                        timezone=timezone,
                        run_time=run_time,
                        trading_days_only=trading_days_only,
                        slippage=slippage,
                        misfire_grace_seconds=misfire_grace_seconds,
                        created_by=actor.strip(),
                        created_at=current,
                        updated_at=current,
                    )
                )
            else:
                self._assert_group_idle(connection, allocation_id)
                connection.execute(
                    update(allocation_schedule_groups)
                    .where(allocation_schedule_groups.c.allocation_id == allocation_id)
                    .values(
                        status="active",
                        timezone=timezone,
                        run_time=run_time,
                        trading_days_only=trading_days_only,
                        slippage=slippage,
                        misfire_grace_seconds=misfire_grace_seconds,
                        updated_at=current,
                    )
                )

            links = {
                str(item.portfolio_id): str(item.schedule_id)
                for item in connection.execute(
                    select(allocation_schedule_members).where(
                        allocation_schedule_members.c.allocation_id == allocation_id
                    )
                )
            }
            for member in members:
                portfolio_id = str(member.portfolio_id)
                _assert_no_conflicting_portfolio_schedule(
                    connection,
                    portfolio_id,
                    exclude_schedule_id=links.get(portfolio_id),
                )
            for member in members:
                portfolio_id = str(member.portfolio_id)
                suspension = (
                    None
                    if member.status in RUNNABLE_PORTFOLIO_STATUSES
                    else f"portfolio:{member.status}"
                )
                effective = "active" if suspension is None else "paused"
                payload = {
                    "portfolio_id": portfolio_id,
                    "allocation_id": allocation_id,
                    "managed_by": "allocation_schedule_group",
                    "slippage": slippage,
                }
                schedule_id = links.get(portfolio_id)
                if schedule_id:
                    connection.execute(
                        update(schedules)
                        .where(schedules.c.id == schedule_id)
                        .values(
                            status=effective,
                            desired_status="active",
                            suspension_reason=suspension,
                            timezone=timezone,
                            run_time=run_time,
                            trading_days_only=trading_days_only,
                            payload_json=payload,
                            misfire_grace_seconds=misfire_grace_seconds,
                            next_run_at=next_run,
                            updated_at=current,
                        )
                    )
                    continue
                schedule_id = uuid.uuid4().hex
                name = f"{allocation.name} · {member.name} · {portfolio_id[:8]}"[:150]
                connection.execute(
                    insert(schedules).values(
                        id=schedule_id,
                        name=name,
                        kind="paper_rebalance",
                        status=effective,
                        desired_status="active",
                        suspension_reason=suspension,
                        timezone=timezone,
                        run_time=run_time,
                        trading_days_only=trading_days_only,
                        payload_json=payload,
                        misfire_grace_seconds=misfire_grace_seconds,
                        next_run_at=next_run,
                        created_by=actor.strip(),
                        created_at=current,
                        updated_at=current,
                    )
                )
                connection.execute(
                    insert(allocation_schedule_members).values(
                        allocation_id=allocation_id,
                        portfolio_id=portfolio_id,
                        schedule_id=schedule_id,
                        created_at=current,
                    )
                )
        return self.get_allocation_group(allocation_id)

    def set_allocation_group_status(
        self,
        allocation_id: str,
        status: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if status not in {"active", "paused", "retired"}:
            raise ValueError("allocation schedule status must be active, paused, or retired")
        current = now or _now()
        with self.engine.begin() as connection:
            group = connection.execute(
                select(allocation_schedule_groups)
                .where(allocation_schedule_groups.c.allocation_id == allocation_id)
                .with_for_update()
            ).first()
            if group is None:
                raise KeyError(allocation_id)
            self._assert_group_idle(connection, allocation_id)
            rows = connection.execute(
                select(
                    allocation_schedule_members.c.portfolio_id,
                    allocation_schedule_members.c.schedule_id,
                    paper_portfolios.c.status.label("portfolio_status"),
                    schedules.c.timezone,
                    schedules.c.run_time,
                )
                .join(
                    paper_portfolios,
                    paper_portfolios.c.id == allocation_schedule_members.c.portfolio_id,
                )
                .join(schedules, schedules.c.id == allocation_schedule_members.c.schedule_id)
                .where(allocation_schedule_members.c.allocation_id == allocation_id)
            ).all()
            if status in {"paused", "retired"} and any(
                item.portfolio_status in RISK_EXECUTION_STATUSES for item in rows
            ):
                raise ValueError("pending risk-execution schedules cannot be paused or retired")
            if status == "active":
                for row in sorted(rows, key=lambda item: str(item.portfolio_id)):
                    portfolio_id = str(row.portfolio_id)
                    _lock_portfolio_schedule(connection, portfolio_id)
                    _assert_no_conflicting_portfolio_schedule(
                        connection,
                        portfolio_id,
                        exclude_schedule_id=str(row.schedule_id),
                    )
            connection.execute(
                update(allocation_schedule_groups)
                .where(allocation_schedule_groups.c.allocation_id == allocation_id)
                .values(status=status, updated_at=current)
            )
            for row in rows:
                suspension = (
                    None
                    if row.portfolio_status in RUNNABLE_PORTFOLIO_STATUSES
                    else f"portfolio:{row.portfolio_status}"
                )
                effective = (
                    "retired"
                    if status == "retired"
                    else "active"
                    if status == "active" and suspension is None
                    else "paused"
                )
                values: dict[str, Any] = {
                    "status": effective,
                    "desired_status": status,
                    "suspension_reason": suspension,
                    "updated_at": current,
                }
                if effective == "active":
                    values["next_run_at"] = next_occurrence(current, row.timezone, row.run_time)
                connection.execute(
                    update(schedules).where(schedules.c.id == row.schedule_id).values(**values)
                )
        return self.get_allocation_group(allocation_id)

    def get_allocation_group(self, allocation_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            group = connection.execute(
                select(allocation_schedule_groups).where(
                    allocation_schedule_groups.c.allocation_id == allocation_id
                )
            ).first()
            if group is None:
                raise KeyError(allocation_id)
            members = connection.execute(
                select(
                    allocation_schedule_members.c.portfolio_id,
                    allocation_schedule_members.c.schedule_id,
                    schedules.c.name,
                    schedules.c.status,
                    schedules.c.desired_status,
                    schedules.c.suspension_reason,
                    schedules.c.next_run_at,
                    schedules.c.last_run_at,
                )
                .join(schedules, schedules.c.id == allocation_schedule_members.c.schedule_id)
                .where(allocation_schedule_members.c.allocation_id == allocation_id)
                .order_by(schedules.c.name)
            ).all()
        result = row_dict(group)
        result["run_time"] = result["run_time"].isoformat(timespec="minutes")
        result["members"] = [row_dict(item) for item in members]
        statuses = {str(item.status) for item in members}
        result["effective_status"] = (
            "retired"
            if result["status"] == "retired"
            else "active"
            if statuses == {"active"}
            else "suspended"
            if result["status"] == "active"
            else "paused"
        )
        return result

    def get_allocation_group_optional(self, allocation_id: str) -> dict[str, Any] | None:
        try:
            return self.get_allocation_group(allocation_id)
        except KeyError:
            return None

    @staticmethod
    def _assert_group_idle(connection: Any, allocation_id: str) -> None:
        active = connection.execute(
            select(schedule_runs.c.id)
            .join(
                allocation_schedule_members,
                allocation_schedule_members.c.schedule_id == schedule_runs.c.schedule_id,
            )
            .where(
                allocation_schedule_members.c.allocation_id == allocation_id,
                schedule_runs.c.status.in_(["pending", "running"]),
            )
            .limit(1)
        ).first()
        if active is not None:
            raise ValueError("allocation schedule configuration is locked by an active run")

    def materialize_due(self, now: datetime | None = None, limit: int = 50) -> int:
        current = now or _now()
        created = 0
        with self.engine.begin() as connection:
            rows = connection.execute(
                select(schedules)
                .where(schedules.c.status == "active", schedules.c.next_run_at <= current)
                .order_by(schedules.c.next_run_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            ).all()
            for row in rows:
                scheduled_for = row.next_run_at
                connection.execute(
                    insert(schedule_runs).values(
                        id=uuid.uuid4().hex,
                        schedule_id=row.id,
                        scheduled_for=scheduled_for,
                        status="pending",
                        attempts=0,
                        dedupe_key=f"{row.id}:{scheduled_for.isoformat()}",
                        created_at=current,
                    )
                )
                connection.execute(
                    update(schedules)
                    .where(schedules.c.id == row.id)
                    .values(
                        last_run_at=scheduled_for,
                        next_run_at=occurrence_after(
                            scheduled_for,
                            row.timezone,
                            row.run_time,
                        ),
                        updated_at=current,
                    )
                )
                created += 1
        return created

    def claim_run(
        self,
        *,
        now: datetime | None = None,
        lease_seconds: int = 120,
    ) -> dict[str, Any] | None:
        current = now or _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(schedule_runs)
                .where(
                    or_(
                        schedule_runs.c.status == "pending",
                        (
                            (schedule_runs.c.status == "running")
                            & (schedule_runs.c.lease_until < current)
                        ),
                    )
                )
                .order_by(schedule_runs.c.scheduled_for)
                .limit(1)
                .with_for_update(skip_locked=True)
            ).first()
            if row is None:
                return None
            connection.execute(
                update(schedule_runs)
                .where(schedule_runs.c.id == row.id)
                .values(
                    status="running",
                    attempts=int(row.attempts) + 1,
                    lease_until=current + timedelta(seconds=lease_seconds),
                    message=None,
                )
            )
            run_id = str(row.id)
        return self.get_run(run_id)

    def finish_run(
        self,
        run_id: str,
        status: str,
        *,
        job_id: str | None = None,
        message: str | None = None,
        now: datetime | None = None,
    ) -> None:
        if status not in {"enqueued", "succeeded", "skipped", "missed", "failed"}:
            raise ValueError("unsupported schedule run status")
        with self.engine.begin() as connection:
            result = connection.execute(
                update(schedule_runs)
                .where(schedule_runs.c.id == run_id)
                .values(
                    status=status,
                    job_id=job_id,
                    message=message,
                    lease_until=None,
                    finished_at=now or _now(),
                )
            )
            if not result.rowcount:
                raise KeyError(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    schedule_runs,
                    schedules.c.kind,
                    schedules.c.timezone,
                    schedules.c.run_time,
                    schedules.c.trading_days_only,
                    schedules.c.payload_json,
                    schedules.c.misfire_grace_seconds,
                    schedules.c.name.label("schedule_name"),
                )
                .join(schedules, schedules.c.id == schedule_runs.c.schedule_id)
                .where(schedule_runs.c.id == run_id)
            ).first()
        if row is None:
            raise KeyError(run_id)
        return self._run_row(row)

    def list_runs(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(schedule_runs, schedules.c.name.label("schedule_name"), schedules.c.kind)
                .join(schedules, schedules.c.id == schedule_runs.c.schedule_id)
                .order_by(schedule_runs.c.created_at.desc())
                .limit(limit)
            )
            return [self._run_row(row) for row in rows]

    @staticmethod
    def _schedule_row(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["payload"] = result.pop("payload_json")
        result["run_time"] = result["run_time"].isoformat(timespec="minutes")
        return result

    @staticmethod
    def _run_row(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        if "payload_json" in result:
            result["payload"] = result.pop("payload_json")
        if result.get("run_time") is not None:
            result["run_time"] = result["run_time"].isoformat(timespec="minutes")
        return result
