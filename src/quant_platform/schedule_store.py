from __future__ import annotations

import uuid
from datetime import UTC, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import insert, or_, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    allocation_schedule_groups,
    open_database,
    recommendation_portfolios,
    row_dict,
    schedule_runs,
    schedules,
    strategy_allocation_members,
    strategy_allocations,
)

ACTIVE_SCHEDULE_KINDS = (
    "incremental_sync",
    "data_pipeline",
    "ashare_5m_sync",
    "rdagent_research",
    "recommendation_refresh",
    "weekly_report",
    "monthly_decision_day",
    "preopen_check",
    "intraday_execution_check",
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


def validate_intraday_run_time(run_time: time, interval_minutes: int) -> None:
    """Validate that a run time is the close of a complete A-share bar."""

    if interval_minutes not in {1, 5}:
        raise ValueError("intraday interval must be 1 or 5 minutes")
    minute_of_day = run_time.hour * 60 + run_time.minute
    morning_open = 9 * 60 + 30
    morning_end = 11 * 60 + 30
    afternoon_open = 13 * 60
    afternoon_end = 15 * 60
    morning_complete = (
        morning_open < minute_of_day <= morning_end
        and (minute_of_day - morning_open) % interval_minutes == 0
    )
    afternoon_complete = (
        afternoon_open < minute_of_day <= afternoon_end
        and (minute_of_day - afternoon_open) % interval_minutes == 0
    )
    if run_time.second or run_time.microsecond or not (
        morning_complete or afternoon_complete
    ):
        raise ValueError(
            "intraday run_time must align to a completed bar inside A-share sessions"
        )


def next_intraday_occurrence(
    now: datetime,
    timezone: str,
    run_time: time,
    interval_minutes: int,
) -> datetime:
    """Return the next intraday slot, including the remaining session today."""

    validate_intraday_run_time(run_time, interval_minutes)
    zone = ZoneInfo(timezone)
    local = now.astimezone(zone)
    local_date = local.date()
    start = datetime.combine(local_date, run_time, tzinfo=zone)
    morning_end = datetime.combine(local_date, time(11, 30), tzinfo=zone)
    afternoon_start = datetime.combine(
        local_date, time(13, interval_minutes), tzinfo=zone
    )
    afternoon_end = datetime.combine(local_date, time(15, 0), tzinfo=zone)

    candidates: list[datetime] = []
    if start <= morning_end:
        candidate = start
        while candidate <= morning_end:
            candidates.append(candidate)
            candidate += timedelta(minutes=interval_minutes)
        candidate = afternoon_start
    else:
        candidate = start
    while candidate <= afternoon_end:
        candidates.append(candidate)
        candidate += timedelta(minutes=interval_minutes)
    for candidate in candidates:
        if candidate > local:
            return candidate.astimezone(UTC)
    next_date = local_date + timedelta(days=1)
    return datetime.combine(next_date, run_time, tzinfo=zone).astimezone(UTC)


def intraday_occurrence_after(
    scheduled_for: datetime,
    timezone: str,
    run_time: time,
    interval_minutes: int,
) -> datetime:
    """Return the next completed A-share minute-bar check slot.

    The schedule itself does not guess trading days; the scheduler's persisted
    Qlib calendar gate skips non-trading dates.  Within a date we only advance
    through the two continuous-auction sessions and resume at the configured
    first completed-bar time on the next calendar day.
    """

    validate_intraday_run_time(run_time, interval_minutes)
    zone = ZoneInfo(timezone)
    local = scheduled_for.astimezone(zone)
    candidate = local + timedelta(minutes=interval_minutes)
    morning_end = datetime.combine(local.date(), time(11, 30), tzinfo=zone)
    afternoon_start = datetime.combine(
        local.date(), time(13, interval_minutes), tzinfo=zone
    )
    afternoon_end = datetime.combine(local.date(), time(15, 0), tzinfo=zone)
    if candidate <= morning_end:
        return candidate.astimezone(UTC)
    if local <= morning_end:
        return afternoon_start.astimezone(UTC)
    if candidate <= afternoon_end:
        return candidate.astimezone(UTC)
    next_date = local.date() + timedelta(days=1)
    return datetime.combine(next_date, run_time, tzinfo=zone).astimezone(UTC)


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
        if kind not in ACTIVE_SCHEDULE_KINDS:
            raise ValueError("unsupported schedule kind")
        if not name.strip() or not actor.strip():
            raise ValueError("schedule name and actor are required")
        if misfire_grace_seconds < 60:
            raise ValueError("misfire grace must be at least 60 seconds")
        recommendation_portfolio_id = (
            str(payload.get("recommendation_portfolio_id") or "")
            if kind == "recommendation_refresh"
            else None
        )
        if kind == "recommendation_refresh" and not recommendation_portfolio_id:
            raise ValueError("recommendation_refresh requires recommendation_portfolio_id")
        current = now or _now()
        interval_minutes = 5
        if kind == "intraday_execution_check":
            interval_minutes = int(payload.get("interval_minutes", 5))
            validate_intraday_run_time(run_time, interval_minutes)
        schedule_id = uuid.uuid4().hex
        try:
            with self.engine.begin() as connection:
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
                        next_run_at=(
                            next_intraday_occurrence(
                                current,
                                timezone,
                                run_time,
                                interval_minutes,
                            )
                            if kind == "intraday_execution_check"
                            else next_occurrence(current, timezone, run_time)
                        ),
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
            row = connection.execute(select(schedules).where(schedules.c.name == name)).first()
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
            if row.kind not in ACTIVE_SCHEDULE_KINDS:
                raise ValueError("legacy schedules are read-only and cannot be reactivated")
            suspension = row.suspension_reason
            effective = status if suspension is None else "paused"
            values: dict[str, Any] = {
                "desired_status": status,
                "status": effective,
                "suspension_reason": suspension,
                "updated_at": current,
            }
            if effective == "active":
                values["next_run_at"] = (
                    next_intraday_occurrence(
                        current,
                        row.timezone,
                        row.run_time,
                        int((row.payload_json or {}).get("interval_minutes", 5)),
                    )
                    if row.kind == "intraday_execution_check"
                    else next_occurrence(
                        current,
                        row.timezone,
                        row.run_time,
                    )
                )
            connection.execute(
                update(schedules).where(schedules.c.id == schedule_id).values(**values)
            )
        return self.get(schedule_id)

    def create_recommendation_allocation_group(
        self,
        allocation_id: str,
        *,
        timezone: str,
        run_time: time,
        trading_days_only: bool,
        misfire_grace_seconds: int,
        actor: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or _now()
        with self.engine.begin() as connection:
            allocation = connection.execute(
                select(strategy_allocations).where(strategy_allocations.c.id == allocation_id)
            ).first()
            if allocation is None:
                raise KeyError(allocation_id)
            if allocation.status == "draft":
                raise ValueError("approve the strategy allocation before scheduling it")
            members = connection.execute(
                select(
                    strategy_allocation_members.c.recommendation_portfolio_id,
                    recommendation_portfolios.c.name,
                )
                .join(
                    recommendation_portfolios,
                    recommendation_portfolios.c.id
                    == strategy_allocation_members.c.recommendation_portfolio_id,
                )
                .where(strategy_allocation_members.c.allocation_id == allocation_id)
            ).all()
            if len(members) < 2:
                raise ValueError("allocation requires at least two recommendation portfolios")
            group = connection.execute(
                select(allocation_schedule_groups).where(
                    allocation_schedule_groups.c.allocation_id == allocation_id
                )
            ).first()
            values = {
                "status": "active",
                "timezone": timezone,
                "run_time": run_time,
                "trading_days_only": trading_days_only,
                "slippage": 0.0,
                "misfire_grace_seconds": misfire_grace_seconds,
                "updated_at": current,
            }
            if group is None:
                connection.execute(
                    insert(allocation_schedule_groups).values(
                        allocation_id=allocation_id,
                        created_by=actor.strip(),
                        created_at=current,
                        **values,
                    )
                )
            else:
                connection.execute(
                    update(allocation_schedule_groups)
                    .where(allocation_schedule_groups.c.allocation_id == allocation_id)
                    .values(**values)
                )
        existing = {
            str(item["payload"].get("recommendation_portfolio_id")): item
            for item in self.list(1000)
            if item["kind"] == "recommendation_refresh"
            and item["payload"].get("allocation_id") == allocation_id
        }
        for member in members:
            portfolio_id = str(member.recommendation_portfolio_id)
            item = existing.get(portfolio_id)
            if item is None:
                self.create(
                    name=f"{allocation.name} / {member.name}"[:150],
                    kind="recommendation_refresh",
                    timezone=timezone,
                    run_time=run_time,
                    trading_days_only=trading_days_only,
                    payload={
                        "recommendation_portfolio_id": portfolio_id,
                        "allocation_id": allocation_id,
                    },
                    misfire_grace_seconds=misfire_grace_seconds,
                    actor=actor,
                    now=current,
                )
            else:
                self.set_status(str(item["id"]), "active", now=current)
        return self.get_recommendation_allocation_group(allocation_id)

    def get_recommendation_allocation_group(self, allocation_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            group = connection.execute(
                select(allocation_schedule_groups).where(
                    allocation_schedule_groups.c.allocation_id == allocation_id
                )
            ).first()
        if group is None:
            raise KeyError(allocation_id)
        result = row_dict(group)
        result["run_time"] = result["run_time"].isoformat(timespec="minutes")
        result["members"] = [
            item
            for item in self.list(1000)
            if item["kind"] == "recommendation_refresh"
            and item["payload"].get("allocation_id") == allocation_id
        ]
        return result

    def get_recommendation_allocation_group_optional(
        self, allocation_id: str
    ) -> dict[str, Any] | None:
        try:
            return self.get_recommendation_allocation_group(allocation_id)
        except KeyError:
            return None

    def set_recommendation_allocation_group_status(
        self, allocation_id: str, status: str
    ) -> dict[str, Any]:
        if status not in {"active", "paused", "retired"}:
            raise ValueError("allocation schedule status must be active, paused or retired")
        group = self.get_recommendation_allocation_group(allocation_id)
        with self.engine.begin() as connection:
            allocation = connection.execute(
                select(strategy_allocations.c.status, strategy_allocations.c.is_legacy).where(
                    strategy_allocations.c.id == allocation_id
                )
            ).first()
            if allocation is None:
                raise KeyError(allocation_id)
            if allocation.is_legacy:
                raise ValueError("legacy paper-backed allocations are read-only")
            if status == "active" and allocation.status != "active":
                raise ValueError("a paused or risk-blocked allocation cannot activate schedules")
            for member in group["members"]:
                values = {
                    "status": status,
                    "desired_status": status,
                    "updated_at": _now(),
                }
                connection.execute(
                    update(schedules).where(schedules.c.id == member["id"]).values(**values)
                )
            connection.execute(
                update(allocation_schedule_groups)
                .where(allocation_schedule_groups.c.allocation_id == allocation_id)
                .values(status=status, updated_at=_now())
            )
        return self.get_recommendation_allocation_group(allocation_id)

    def materialize_due(self, now: datetime | None = None, limit: int = 50) -> int:
        current = now or _now()
        created = 0
        with self.engine.begin() as connection:
            rows = connection.execute(
                select(schedules)
                .where(
                    schedules.c.status == "active",
                    schedules.c.kind.in_(ACTIVE_SCHEDULE_KINDS),
                    schedules.c.next_run_at <= current,
                )
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
                        next_run_at=(
                            intraday_occurrence_after(
                                scheduled_for,
                                row.timezone,
                                row.run_time,
                                int((row.payload_json or {}).get("interval_minutes", 5)),
                            )
                            if row.kind == "intraday_execution_check"
                            else occurrence_after(
                                scheduled_for,
                                row.timezone,
                                row.run_time,
                            )
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
                .join(schedules, schedules.c.id == schedule_runs.c.schedule_id)
                .where(
                    schedules.c.kind.in_(ACTIVE_SCHEDULE_KINDS),
                    or_(
                        schedule_runs.c.status == "pending",
                        (
                            (schedule_runs.c.status == "running")
                            & (schedule_runs.c.lease_until < current)
                        ),
                    ),
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
