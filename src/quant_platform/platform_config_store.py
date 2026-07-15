from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import insert, select, update

from quant_data.database import (
    open_database,
    platform_config_revisions,
    platform_configs,
    row_dict,
)


def _now() -> datetime:
    return datetime.now(UTC)


class PlatformConfigStore:
    """Versioned non-secret configuration managed through the Web control plane."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def get(self, key: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(platform_configs).where(platform_configs.c.key == key)
            ).first()
        if row is None:
            return None
        result = row_dict(row)
        result["value"] = result.pop("value_json")
        return result

    def list_revisions(self, key: str, limit: int = 50) -> list[dict[str, Any]]:
        statement = (
            select(platform_config_revisions)
            .where(platform_config_revisions.c.key == key)
            .order_by(platform_config_revisions.c.revision.desc())
            .limit(limit)
        )
        with self.engine.connect() as connection:
            rows = [row_dict(row) for row in connection.execute(statement)]
        for row in rows:
            row["value"] = row.pop("value_json")
        return rows

    def put(
        self,
        key: str,
        value: dict[str, Any],
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        actor = actor.strip()
        reason = reason.strip()
        if not key.strip() or not actor:
            raise ValueError("configuration key and actor are required")
        if len(reason) < 10:
            raise ValueError("configuration reason must contain at least 10 characters")
        now = _now()
        with self.engine.begin() as connection:
            current = connection.execute(
                select(platform_configs).where(platform_configs.c.key == key).with_for_update()
            ).first()
            revision = int(current.revision) + 1 if current else 1
            connection.execute(
                insert(platform_config_revisions).values(
                    key=key,
                    revision=revision,
                    value_json=value,
                    reason=reason,
                    updated_by=actor,
                    created_at=now,
                )
            )
            if current:
                connection.execute(
                    update(platform_configs)
                    .where(platform_configs.c.key == key)
                    .values(
                        revision=revision,
                        value_json=value,
                        updated_by=actor,
                        updated_at=now,
                    )
                )
            else:
                connection.execute(
                    insert(platform_configs).values(
                        key=key,
                        revision=revision,
                        value_json=value,
                        updated_by=actor,
                        updated_at=now,
                    )
                )
        result = self.get(key)
        if result is None:  # pragma: no cover - transaction invariant
            raise RuntimeError("configuration write did not persist")
        return result
