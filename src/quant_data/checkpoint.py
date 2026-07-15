from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import case, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .database import open_database, row_dict, work_units
from .models import FetchSpec, UnitResult, WorkUnit

_INSERT_BATCH_SIZE = 500
_SELECT_BATCH_SIZE = 5_000


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CheckpointStore:
    """PostgreSQL-backed work-unit checkpoint store with safe concurrent claiming."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def add(self, specs: Iterable[FetchSpec]) -> int:
        now = _utc_now()
        rows = [
            {
                "unit_key": spec.unit_key,
                "dataset": spec.dataset,
                "api_name": spec.api_name,
                "scope_json": spec.scope,
                "params_json": spec.params,
                "fields_json": list(spec.fields),
                "allow_empty": spec.allow_empty,
                "status": "pending",
                "attempts": 0,
                "max_attempts": spec.max_attempts,
                "created_at": now,
                "updated_at": now,
            }
            for spec in specs
        ]
        if not rows:
            return 0
        inserted = 0
        with self.engine.begin() as connection:
            for offset in range(0, len(rows), _INSERT_BATCH_SIZE):
                statement = (
                    pg_insert(work_units)
                    .values(rows[offset : offset + _INSERT_BATCH_SIZE])
                    .on_conflict_do_nothing(index_elements=[work_units.c.unit_key])
                    .returning(work_units.c.unit_key)
                )
                inserted += len(connection.execute(statement).all())
        return inserted

    def reset_stale(self) -> int:
        now = _utc_now()
        statement = (
            update(work_units)
            .where(work_units.c.status == "running", work_units.c.lease_until < now)
            .values(status="pending", lease_until=None, updated_at=now)
        )
        with self.engine.begin() as connection:
            return int(connection.execute(statement).rowcount or 0)

    def claim(self, datasets: set[str] | None = None, lease_seconds: int = 300) -> WorkUnit | None:
        now = _utc_now()
        conditions = [
            work_units.c.status.in_(("pending", "failed")),
            work_units.c.attempts < work_units.c.max_attempts,
            or_(work_units.c.next_retry_at.is_(None), work_units.c.next_retry_at <= now),
        ]
        if datasets:
            conditions.append(work_units.c.dataset.in_(sorted(datasets)))
        statement = (
            select(work_units)
            .where(*conditions)
            .order_by(work_units.c.created_at, work_units.c.dataset, work_units.c.unit_key)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        with self.engine.begin() as connection:
            row = connection.execute(statement).first()
            if row is None:
                return None
            attempts = int(row.attempts) + 1
            connection.execute(
                update(work_units)
                .where(work_units.c.unit_key == row.unit_key)
                .values(
                    status="running",
                    attempts=attempts,
                    lease_until=now + timedelta(seconds=lease_seconds),
                    last_error=None,
                    updated_at=now,
                )
            )
        spec = FetchSpec(
            dataset=row.dataset,
            api_name=row.api_name,
            scope=dict(row.scope_json),
            params=dict(row.params_json),
            fields=tuple(row.fields_json),
            allow_empty=bool(row.allow_empty),
            max_attempts=int(row.max_attempts),
        )
        return WorkUnit(unit_key=row.unit_key, spec=spec, attempts=attempts)

    def succeed(self, unit_key: str, result: UnitResult) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(work_units)
                .where(work_units.c.unit_key == unit_key)
                .values(
                    status="succeeded",
                    lease_until=None,
                    next_retry_at=None,
                    output_path=result.output_path,
                    row_count=result.row_count,
                    sha256=result.sha256,
                    last_error=None,
                    updated_at=_utc_now(),
                )
            )

    def fail(
        self,
        unit_key: str,
        error: str,
        retry_after_seconds: int = 0,
        *,
        terminal: bool = False,
    ) -> None:
        now = _utc_now()
        values: dict[str, Any] = {
            "status": "failed",
            "lease_until": None,
            "next_retry_at": now + timedelta(seconds=retry_after_seconds),
            "last_error": error[:2000],
            "updated_at": now,
        }
        if terminal:
            values["attempts"] = work_units.c.max_attempts
        with self.engine.begin() as connection:
            connection.execute(
                update(work_units).where(work_units.c.unit_key == unit_key).values(**values)
            )

    def retry_failed(self) -> int:
        statement = (
            update(work_units)
            .where(work_units.c.status == "failed")
            .values(
                status="pending",
                attempts=0,
                next_retry_at=None,
                lease_until=None,
                updated_at=_utc_now(),
            )
        )
        with self.engine.begin() as connection:
            return int(connection.execute(statement).rowcount or 0)

    def retry_failed_units(self, unit_keys: Iterable[str]) -> int:
        keys = sorted(set(unit_keys))
        if not keys:
            return 0
        statement = (
            update(work_units)
            .where(work_units.c.status == "failed", work_units.c.unit_key.in_(keys))
            .values(
                status="pending",
                attempts=0,
                next_retry_at=None,
                lease_until=None,
                updated_at=_utc_now(),
            )
        )
        with self.engine.begin() as connection:
            return int(connection.execute(statement).rowcount or 0)

    def counts(self) -> list[dict[str, Any]]:
        statement = (
            select(
                work_units.c.dataset,
                work_units.c.status,
                func.count().label("units"),
                func.coalesce(func.sum(work_units.c.row_count), 0).label("rows"),
            )
            .group_by(work_units.c.dataset, work_units.c.status)
            .order_by(work_units.c.dataset, work_units.c.status)
        )
        with self.engine.connect() as connection:
            return [row_dict(row) for row in connection.execute(statement)]

    def successful(self, dataset: str | None = None) -> list[dict[str, Any]]:
        statement = select(work_units).where(work_units.c.status == "succeeded")
        if dataset:
            statement = statement.where(work_units.c.dataset == dataset)
        statement = statement.order_by(work_units.c.dataset, work_units.c.unit_key)
        with self.engine.connect() as connection:
            return [row_dict(row) for row in connection.execute(statement)]

    def successful_units(self, unit_keys: Iterable[str]) -> list[dict[str, Any]]:
        """Return successful rows for an explicit plan without scanning all history."""

        keys = sorted(set(unit_keys))
        if not keys:
            return []
        rows: list[dict[str, Any]] = []
        with self.engine.connect() as connection:
            for offset in range(0, len(keys), _SELECT_BATCH_SIZE):
                batch = keys[offset : offset + _SELECT_BATCH_SIZE]
                statement = select(work_units).where(
                    work_units.c.status == "succeeded",
                    work_units.c.unit_key.in_(batch),
                )
                rows.extend(row_dict(row) for row in connection.execute(statement))
        return sorted(rows, key=lambda row: (str(row["dataset"]), str(row["unit_key"])))

    def datasets(self) -> list[str]:
        statement = select(work_units.c.dataset).distinct().order_by(work_units.c.dataset)
        with self.engine.connect() as connection:
            return [str(row.dataset) for row in connection.execute(statement)]

    def failures(self, limit: int = 20) -> list[dict[str, Any]]:
        statement = (
            select(
                work_units.c.dataset,
                work_units.c.scope_json,
                work_units.c.attempts,
                work_units.c.last_error,
                work_units.c.updated_at,
            )
            .where(work_units.c.status == "failed")
            .order_by(work_units.c.updated_at.desc())
            .limit(limit)
        )
        with self.engine.connect() as connection:
            return [row_dict(row) for row in connection.execute(statement)]

    def remaining_count(self, datasets: set[str] | None = None) -> int:
        conditions = [
            work_units.c.status != "succeeded",
            work_units.c.attempts < work_units.c.max_attempts,
        ]
        if datasets:
            conditions.append(work_units.c.dataset.in_(sorted(datasets)))
        statement = select(func.count()).select_from(work_units).where(*conditions)
        with self.engine.connect() as connection:
            return int(connection.execute(statement).scalar_one())

    def verification_rows(self) -> list[dict[str, Any]]:
        statement = (
            select(
                work_units.c.dataset,
                func.count().label("planned"),
                func.sum(case((work_units.c.status == "succeeded", 1), else_=0)).label("succeeded"),
                func.sum(case((work_units.c.status == "failed", 1), else_=0)).label("failed"),
                func.sum(case((work_units.c.status == "running", 1), else_=0)).label("running"),
                func.sum(
                    case(
                        (
                            (work_units.c.status == "succeeded") & (work_units.c.row_count == 0),
                            1,
                        ),
                        else_=0,
                    )
                ).label("empty"),
                func.coalesce(
                    func.sum(
                        case(
                            (work_units.c.status == "succeeded", work_units.c.row_count),
                            else_=0,
                        )
                    ),
                    0,
                ).label("rows"),
            )
            .group_by(work_units.c.dataset)
            .order_by(work_units.c.dataset)
        )
        with self.engine.connect() as connection:
            return [row_dict(row) for row in connection.execute(statement)]
