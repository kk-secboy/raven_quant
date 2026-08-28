from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    jobs,
    open_database,
    research_report_backfill_days,
    row_dict,
)


def _now() -> datetime:
    return datetime.now(UTC)


class ResearchReportBackfillStore:
    """Durable newest-first cursor for selected Tushare report PDFs."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def seed(self, dates: list[date], *, snapshot_name: str) -> int:
        created = 0
        now = _now()
        for report_date in sorted(set(dates), reverse=True):
            try:
                with self.engine.begin() as connection:
                    connection.execute(
                        insert(research_report_backfill_days).values(
                            report_date=report_date,
                            snapshot_name=snapshot_name,
                            status="pending",
                            attempts=0,
                            selected_count=0,
                            published_count=0,
                            blocked_count=0,
                            bytes_downloaded=0,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                created += 1
            except IntegrityError:
                with self.engine.begin() as connection:
                    connection.execute(
                        update(research_report_backfill_days)
                        .where(
                            research_report_backfill_days.c.report_date == report_date,
                            research_report_backfill_days.c.status == "pending",
                        )
                        .values(snapshot_name=snapshot_name, updated_at=now)
                    )
        return created

    def reconcile(self) -> int:
        changed = 0
        now = _now()
        with self.engine.begin() as connection:
            rows = connection.execute(
                select(
                    research_report_backfill_days,
                    jobs.c.status.label("job_status"),
                    jobs.c.progress_json.label("job_result"),
                    jobs.c.error.label("job_error"),
                )
                .join(jobs, jobs.c.id == research_report_backfill_days.c.job_id)
                .where(research_report_backfill_days.c.status.in_(("queued", "running")))
                .with_for_update()
            ).all()
            for row in rows:
                job_status = str(row.job_status)
                if job_status in {"queued", "running"}:
                    if job_status != str(row.status):
                        connection.execute(
                            update(research_report_backfill_days)
                            .where(research_report_backfill_days.c.report_date == row.report_date)
                            .values(status=job_status, updated_at=now)
                        )
                        changed += 1
                    continue
                result = dict(row.job_result or {})
                assets = list(result.get("assets") or [])
                connection.execute(
                    update(research_report_backfill_days)
                    .where(research_report_backfill_days.c.report_date == row.report_date)
                    .values(
                        status="succeeded" if job_status == "succeeded" else "blocked",
                        selected_count=int(result.get("tushare_selected") or 0),
                        published_count=int(result.get("published") or 0),
                        blocked_count=int(result.get("blocked") or 0),
                        bytes_downloaded=sum(
                            int(item.get("size_bytes") or 0)
                            for item in assets
                            if isinstance(item, dict)
                        ),
                        last_error=(
                            None
                            if job_status == "succeeded"
                            else str(row.job_error or "research report acquisition blocked")
                        ),
                        updated_at=now,
                        finished_at=now,
                    )
                )
                changed += 1
        return changed

    def pending(self, *, limit: int) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            return [
                row_dict(row)
                for row in connection.execute(
                    select(research_report_backfill_days)
                    .where(research_report_backfill_days.c.status == "pending")
                    .order_by(research_report_backfill_days.c.report_date.desc())
                    .limit(max(0, limit))
                )
            ]

    def active_count(self) -> int:
        with self.engine.connect() as connection:
            return int(
                connection.scalar(
                    select(func.count())
                    .select_from(research_report_backfill_days)
                    .where(research_report_backfill_days.c.status.in_(("queued", "running")))
                )
                or 0
            )

    def attempts_started_since(self, since: datetime) -> int:
        with self.engine.connect() as connection:
            return int(
                connection.scalar(
                    select(
                        func.coalesce(func.sum(research_report_backfill_days.c.attempts), 0)
                    ).where(research_report_backfill_days.c.updated_at >= since)
                )
                or 0
            )

    def bytes_finished_since(self, since: datetime) -> int:
        with self.engine.connect() as connection:
            return int(
                connection.scalar(
                    select(
                        func.coalesce(func.sum(research_report_backfill_days.c.bytes_downloaded), 0)
                    ).where(research_report_backfill_days.c.finished_at >= since)
                )
                or 0
            )

    def mark_queued(self, report_date: date, *, job_id: str) -> None:
        now = _now()
        with self.engine.begin() as connection:
            result = connection.execute(
                update(research_report_backfill_days)
                .where(
                    research_report_backfill_days.c.report_date == report_date,
                    research_report_backfill_days.c.status == "pending",
                )
                .values(
                    status="queued",
                    attempts=research_report_backfill_days.c.attempts + 1,
                    job_id=job_id,
                    updated_at=now,
                )
            )
            if result.rowcount != 1:
                raise ValueError("research-report backfill day is no longer pending")

    def list(self, *, limit: int = 500) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            return [
                row_dict(row)
                for row in connection.execute(
                    select(research_report_backfill_days)
                    .order_by(research_report_backfill_days.c.report_date.desc())
                    .limit(min(max(limit, 1), 2000))
                )
            ]

    def summary(self) -> dict[str, Any]:
        with self.engine.connect() as connection:
            counts = {
                str(status): int(count)
                for status, count in connection.execute(
                    select(research_report_backfill_days.c.status, func.count()).group_by(
                        research_report_backfill_days.c.status
                    )
                )
            }
            totals = connection.execute(
                select(
                    func.coalesce(func.sum(research_report_backfill_days.c.selected_count), 0),
                    func.coalesce(func.sum(research_report_backfill_days.c.published_count), 0),
                    func.coalesce(func.sum(research_report_backfill_days.c.bytes_downloaded), 0),
                )
            ).one()
        return {
            "counts": counts,
            "selected": int(totals[0]),
            "published": int(totals[1]),
            "bytes_downloaded": int(totals[2]),
            "complete": counts.get("pending", 0)
            + counts.get("queued", 0)
            + counts.get("running", 0)
            == 0,
        }
