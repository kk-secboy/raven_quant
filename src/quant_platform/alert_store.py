from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import requests
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from quant_data.database import alerts, open_database, row_dict


def _now() -> datetime:
    return datetime.now(UTC)


class AlertStore:
    """Durable alert inbox with idempotent projection and optional webhook delivery."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def create(
        self,
        *,
        source_type: str,
        source_id: str,
        severity: str,
        category: str,
        title: str,
        message: str,
        dedupe_key: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        alert_id = uuid.uuid4().hex
        with self.engine.begin() as connection:
            inserted = connection.execute(
                pg_insert(alerts)
                .values(
                    id=alert_id,
                    source_type=source_type,
                    source_id=source_id,
                    severity=severity,
                    category=category,
                    title=title,
                    message=message,
                    status="open",
                    dedupe_key=dedupe_key,
                    details_json=details or {},
                    delivery_status="pending",
                    delivery_attempts=0,
                    created_at=_now(),
                )
                .on_conflict_do_nothing(index_elements=[alerts.c.dedupe_key])
                .returning(alerts.c.id)
            ).scalar_one_or_none()
            if inserted is None:
                alert_id = str(
                    connection.execute(
                        select(alerts.c.id).where(alerts.c.dedupe_key == dedupe_key)
                    ).scalar_one()
                )
        return self.get(alert_id)

    def get(self, alert_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(select(alerts).where(alerts.c.id == alert_id)).first()
        if row is None:
            raise KeyError(alert_id)
        return self._row(row)

    def list(self, *, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        statement = select(alerts)
        if status:
            statement = statement.where(alerts.c.status == status)
        statement = statement.order_by(alerts.c.created_at.desc()).limit(limit)
        with self.engine.connect() as connection:
            return [self._row(row) for row in connection.execute(statement)]

    def acknowledge(self, alert_id: str, *, actor: str) -> dict[str, Any]:
        if not actor.strip():
            raise ValueError("actor is required")
        with self.engine.begin() as connection:
            result = connection.execute(
                update(alerts)
                .where(alerts.c.id == alert_id, alerts.c.status == "open")
                .values(
                    status="acknowledged",
                    acknowledged_by=actor.strip(),
                    acknowledged_at=_now(),
                )
            )
            if not result.rowcount:
                existing = connection.execute(
                    select(alerts.c.id).where(alerts.c.id == alert_id)
                ).first()
                if existing is None:
                    raise KeyError(alert_id)
        return self.get(alert_id)

    def resolve(self, alert_id: str, *, actor: str) -> dict[str, Any]:
        if not actor.strip():
            raise ValueError("actor is required")
        with self.engine.begin() as connection:
            result = connection.execute(
                update(alerts)
                .where(alerts.c.id == alert_id, alerts.c.status != "resolved")
                .values(status="resolved", resolved_by=actor.strip(), resolved_at=_now())
            )
            if not result.rowcount:
                existing = connection.execute(
                    select(alerts.c.id).where(alerts.c.id == alert_id)
                ).first()
                if existing is None:
                    raise KeyError(alert_id)
        return self.get(alert_id)

    def deliver_pending(self, webhook_url: str, *, limit: int = 20) -> int:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(alerts)
                .where(
                    alerts.c.status == "open",
                    alerts.c.delivery_status.in_(("pending", "failed", "not_configured")),
                    alerts.c.delivery_attempts < 10,
                )
                .order_by(alerts.c.created_at)
                .limit(limit)
            ).all()
        if not webhook_url:
            if rows:
                with self.engine.begin() as connection:
                    connection.execute(
                        update(alerts)
                        .where(alerts.c.id.in_([row.id for row in rows]))
                        .values(delivery_status="not_configured")
                    )
            return 0
        delivered = 0
        for row in rows:
            payload = self._row(row)
            try:
                response = requests.post(webhook_url, json=payload, timeout=8)
                response.raise_for_status()
            except requests.RequestException as exc:
                values = {
                    "delivery_status": "failed",
                    "delivery_attempts": int(row.delivery_attempts) + 1,
                    "last_delivery_error": str(exc)[:2000],
                }
            else:
                delivered += 1
                values = {
                    "delivery_status": "delivered",
                    "delivery_attempts": int(row.delivery_attempts) + 1,
                    "delivered_at": _now(),
                    "last_delivery_error": None,
                }
            with self.engine.begin() as connection:
                connection.execute(update(alerts).where(alerts.c.id == row.id).values(**values))
        return delivered

    @staticmethod
    def _row(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["details"] = result.pop("details_json")
        return result
