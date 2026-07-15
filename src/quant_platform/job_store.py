from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from quant_data.database import jobs, open_database, row_dict

AUTO_RETRY_ATTEMPTS = {
    "rdagent_factor": 3,
    "recommendation_refresh": 3,
}


def _now() -> datetime:
    return datetime.now(UTC)


class JobStore:
    """PostgreSQL-backed durable job repository for API and worker processes."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def recover_interrupted(self, allowed_kinds: tuple[str, ...] = ()) -> int:
        statement = update(jobs).where(jobs.c.status == "running")
        if allowed_kinds:
            statement = statement.where(jobs.c.kind.in_(allowed_kinds))
        statement = statement.values(
            status="queued",
            started_at=None,
            next_attempt_at=None,
            error="Worker restarted; job safely requeued",
        )
        with self.engine.begin() as connection:
            return int(connection.execute(statement).rowcount or 0)

    def create(
        self,
        kind: str,
        payload: dict[str, Any],
        log_path: Path,
        *,
        dedupe_active_kind: bool = True,
        idempotency_key: str | None = None,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        existing_id: str | None = None
        try:
            with self.engine.begin() as connection:
                if idempotency_key:
                    existing = connection.execute(
                        select(jobs.c.id).where(jobs.c.idempotency_key == idempotency_key)
                    ).first()
                    if existing:
                        existing_id = str(existing.id)
                if existing_id is None:
                    if dedupe_active_kind:
                        connection.execute(
                            text("SELECT pg_advisory_xact_lock(hashtext(:kind))"),
                            {"kind": kind},
                        )
                        active = connection.execute(
                            select(jobs.c.id)
                            .where(
                                jobs.c.kind == kind,
                                jobs.c.status.in_(("queued", "running")),
                            )
                            .limit(1)
                        ).first()
                        if active:
                            raise ValueError(f"an active {kind} job already exists: {active.id}")
                    connection.execute(
                        pg_insert(jobs).values(
                            id=job_id,
                            kind=kind,
                            idempotency_key=idempotency_key,
                            status="queued",
                            payload_json=payload,
                            log_path=str(log_path),
                            attempts=0,
                            max_attempts=(
                                max_attempts
                                if max_attempts is not None
                                else AUTO_RETRY_ATTEMPTS.get(kind, 1)
                            ),
                            created_at=_now(),
                        )
                    )
        except IntegrityError as exc:
            if idempotency_key:
                with self.engine.connect() as connection:
                    existing = connection.execute(
                        select(jobs.c.id).where(jobs.c.idempotency_key == idempotency_key)
                    ).first()
                if existing:
                    return self.get(str(existing.id))
            raise ValueError(f"could not create {kind} job") from exc
        return self.get(existing_id or job_id)

    def claim_next(self, allowed_kinds: tuple[str, ...] = ()) -> dict[str, Any] | None:
        statement = select(jobs).where(
            jobs.c.status == "queued",
            (jobs.c.next_attempt_at.is_(None)) | (jobs.c.next_attempt_at <= _now()),
        )
        if allowed_kinds:
            statement = statement.where(jobs.c.kind.in_(allowed_kinds))
        statement = statement.order_by(jobs.c.created_at).limit(1).with_for_update(skip_locked=True)
        with self.engine.begin() as connection:
            row = connection.execute(statement).first()
            if row is None:
                return None
            connection.execute(
                update(jobs)
                .where(jobs.c.id == row.id)
                .values(
                    status="running",
                    started_at=_now(),
                    attempts=jobs.c.attempts + 1,
                    next_attempt_at=None,
                    error=None,
                )
            )
            job_id = str(row.id)
        return self.get(job_id)

    def finish(
        self,
        job_id: str,
        *,
        exit_code: int,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        status = "succeeded" if exit_code == 0 else "failed"
        with self.engine.begin() as connection:
            connection.execute(
                update(jobs)
                .where(jobs.c.id == job_id)
                .values(
                    status=status,
                    exit_code=exit_code,
                    error=error,
                    progress_json=result,
                    cancel_requested_at=None,
                    next_attempt_at=None,
                    finished_at=_now(),
                )
            )

    def finish_or_retry(
        self,
        job_id: str,
        *,
        exit_code: int,
        error: str,
        result: dict[str, Any] | None = None,
        retryable: bool,
    ) -> bool:
        """Finish a job or queue a bounded transient retry.

        Returns True only when the same durable job was requeued. Attempts are
        incremented on claim, so a max_attempts value of three means at most
        three actual process executions.
        """
        with self.engine.begin() as connection:
            row = connection.execute(
                select(jobs.c.status, jobs.c.attempts, jobs.c.max_attempts)
                .where(jobs.c.id == job_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(job_id)
            if (
                retryable
                and row.status == "running"
                and int(row.attempts) < int(row.max_attempts)
            ):
                delay_seconds = min(900, 30 * (2 ** max(0, int(row.attempts) - 1)))
                connection.execute(
                    update(jobs)
                    .where(jobs.c.id == job_id)
                    .values(
                        status="queued",
                        exit_code=exit_code,
                        error=error,
                        progress_json=result,
                        started_at=None,
                        finished_at=None,
                        cancel_requested_at=None,
                        next_attempt_at=_now() + timedelta(seconds=delay_seconds),
                    )
                )
                return True
            connection.execute(
                update(jobs)
                .where(jobs.c.id == job_id)
                .values(
                    status="failed",
                    exit_code=exit_code,
                    error=error,
                    progress_json=result,
                    cancel_requested_at=None,
                    next_attempt_at=None,
                    finished_at=_now(),
                )
            )
        return False

    def retry(self, job_id: str) -> dict[str, Any]:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(jobs.c.status).where(jobs.c.id == job_id).with_for_update()
            ).first()
            if row is None:
                raise KeyError(job_id)
            if row.status not in {"failed", "cancelled"}:
                raise ValueError("only failed or cancelled jobs may be retried")
            connection.execute(
                update(jobs)
                .where(jobs.c.id == job_id)
                .values(
                    status="queued",
                    attempts=0,
                    progress_json=None,
                    exit_code=None,
                    error=None,
                    started_at=None,
                    finished_at=None,
                    cancel_requested_at=None,
                    next_attempt_at=None,
                )
            )
        return self.get(job_id)

    def request_cancel(self, job_id: str) -> dict[str, Any]:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(jobs.c.status).where(jobs.c.id == job_id).with_for_update()
            ).first()
            if row is None:
                raise KeyError(job_id)
            if row.status == "queued":
                connection.execute(
                    update(jobs)
                    .where(jobs.c.id == job_id)
                    .values(
                        status="cancelled",
                        error="Cancelled before execution",
                        cancel_requested_at=_now(),
                        finished_at=_now(),
                    )
                )
            elif row.status == "running":
                connection.execute(
                    update(jobs).where(jobs.c.id == job_id).values(cancel_requested_at=_now())
                )
            else:
                raise ValueError("only queued or running jobs may be cancelled")
        return self.get(job_id)

    def cancellation_requested(self, job_id: str) -> bool:
        with self.engine.connect() as connection:
            value = connection.execute(
                select(jobs.c.cancel_requested_at).where(jobs.c.id == job_id)
            ).scalar_one_or_none()
        return value is not None

    def mark_cancelled(self, job_id: str, error: str = "Cancelled by operator") -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(jobs)
                .where(jobs.c.id == job_id)
                .values(
                    status="cancelled",
                    exit_code=None,
                    error=error,
                    finished_at=_now(),
                )
            )

    def list(
        self,
        limit: int = 50,
        *,
        offset: int = 0,
        statuses: tuple[str, ...] = (),
        kinds: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        statement = select(jobs)
        if statuses:
            statement = statement.where(jobs.c.status.in_(statuses))
        if kinds:
            statement = statement.where(jobs.c.kind.in_(kinds))
        statement = statement.order_by(jobs.c.created_at.desc()).offset(offset).limit(limit)
        with self.engine.connect() as connection:
            return [self._decode(row_dict(row)) for row in connection.execute(statement)]

    def count(
        self,
        *,
        statuses: tuple[str, ...] = (),
        kinds: tuple[str, ...] = (),
    ) -> int:
        statement = select(text("count(*)")).select_from(jobs)
        if statuses:
            statement = statement.where(jobs.c.status.in_(statuses))
        if kinds:
            statement = statement.where(jobs.c.kind.in_(kinds))
        with self.engine.connect() as connection:
            return int(connection.execute(statement).scalar_one())

    def get(self, job_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(select(jobs).where(jobs.c.id == job_id)).first()
        if row is None:
            raise KeyError(job_id)
        return self._decode(row_dict(row))

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        row["payload"] = row.pop("payload_json")
        row["progress"] = row.pop("progress_json")
        return row
