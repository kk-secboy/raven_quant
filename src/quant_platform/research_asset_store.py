from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    open_database,
    research_asset_consumptions,
    research_assets,
    row_dict,
)

from .model_research_governance import is_sha256


def _now() -> datetime:
    return datetime.now(UTC)


def _actor(value: str) -> str:
    normalized = str(value or "").strip()
    if len(normalized) < 2 or len(normalized) > 100:
        raise ValueError("a responsible research-asset actor is required")
    return normalized


class ResearchAssetStore:
    """Safe DB projection plus one-time reservations for automatic inputs."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def unavailable_asset_ids(self) -> set[str]:
        with self.engine.connect() as connection:
            return {
                str(value)
                for value in connection.scalars(
                    select(research_asset_consumptions.c.asset_id)
                ).all()
            }

    def reserve_automatic(
        self,
        *,
        research_run_id: str,
        scenario: str,
        asset_manifest_sha256: Mapping[str, str],
        actor: str,
    ) -> list[dict[str, Any]]:
        """Atomically reserve never-before-used assets for one research run."""

        if not asset_manifest_sha256:
            return []
        responsible_actor = _actor(actor)
        now = _now()
        try:
            with self.engine.begin() as connection:
                for asset_id, expected_materialized_sha256 in sorted(
                    asset_manifest_sha256.items()
                ):
                    expected = str(expected_materialized_sha256).lower()
                    if not is_sha256(expected):
                        raise ValueError("research asset manifest SHA-256 is invalid")
                    asset = connection.execute(
                        select(research_assets)
                        .where(research_assets.c.id == str(asset_id))
                        .with_for_update()
                    ).first()
                    if asset is None or str(asset.status) != "registered":
                        raise ValueError(f"research asset is not registered: {asset_id}")
                    db_manifest = dict(asset.manifest_json or {})
                    metadata = dict(db_manifest.get("metadata") or {})
                    actual = str(metadata.get("materialized_manifest_sha256") or "").lower()
                    if actual != expected:
                        raise ValueError(
                            f"research asset manifest identity disagrees: {asset_id}"
                        )
                    connection.execute(
                        insert(research_asset_consumptions).values(
                            id=uuid.uuid4().hex,
                            asset_id=str(asset_id),
                            research_run_id=research_run_id,
                            scenario=scenario,
                            selection_mode="automatic",
                            asset_manifest_sha256=expected,
                            # Assignment itself consumes an automatically
                            # selected document.  A failed experiment must not
                            # make the same evidence look like a fresh trial.
                            status="consumed",
                            reserved_by=responsible_actor,
                            reserved_at=now,
                            completed_at=now,
                            details_json={
                                "contract_version": "research-asset-consumption-v1",
                                "auto_selected": True,
                            },
                        )
                    )
        except IntegrityError as exc:
            raise ValueError(
                "one or more automatically selected research assets were already consumed"
            ) from exc
        return self.list_consumptions(research_run_id=research_run_id)

    def finish_run(self, research_run_id: str, *, succeeded: bool) -> int:
        status = "consumed" if succeeded else "failed"
        with self.engine.begin() as connection:
            result = connection.execute(
                update(research_asset_consumptions)
                .where(
                    research_asset_consumptions.c.research_run_id == research_run_id,
                    research_asset_consumptions.c.status == "reserved",
                )
                .values(status=status, completed_at=_now())
            )
        return int(result.rowcount or 0)

    def list_consumptions(
        self, *, research_run_id: str | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        statement = select(research_asset_consumptions)
        if research_run_id is not None:
            statement = statement.where(
                research_asset_consumptions.c.research_run_id == research_run_id
            )
        statement = statement.order_by(
            research_asset_consumptions.c.reserved_at.desc()
        ).limit(min(max(int(limit), 1), 1000))
        with self.engine.connect() as connection:
            return [row_dict(row) for row in connection.execute(statement)]

    def list_assets(self, *, limit: int = 500) -> list[dict[str, Any]]:
        bounded = min(max(int(limit), 1), 1000)
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(research_assets)
                .order_by(research_assets.c.created_at.desc())
                .limit(bounded)
            ).all()
            consumptions = {
                str(row.asset_id): row
                for row in connection.execute(select(research_asset_consumptions)).all()
            }
        results: list[dict[str, Any]] = []
        for row in rows:
            manifest = dict(row.manifest_json or {})
            metadata = dict(manifest.get("metadata") or {})
            consumption = consumptions.get(str(row.id))
            results.append(
                {
                    "id": str(row.id),
                    "asset_key": str(row.asset_key),
                    "asset_type": str(row.asset_type),
                    "asset_kind": str(metadata.get("asset_kind") or ""),
                    "media_type": str(row.media_type),
                    "source_kind": str(row.publisher or ""),
                    "title": str(metadata.get("title") or ""),
                    "authors": list(metadata.get("authors") or []),
                    "categories": list(metadata.get("categories") or []),
                    "published_at": row.published_at,
                    "retrieved_at": row.retrieved_at,
                    "content_sha256": str(row.content_sha256),
                    "manifest_sha256": str(row.manifest_sha256),
                    "materialized_manifest_sha256": str(
                        metadata.get("materialized_manifest_sha256") or ""
                    ),
                    "size_bytes": int(row.size_bytes),
                    "status": str(row.status),
                    "consumption": (
                        {
                            "status": str(consumption.status),
                            "scenario": str(consumption.scenario),
                            "research_run_id": str(consumption.research_run_id),
                            "reserved_at": consumption.reserved_at,
                            "completed_at": consumption.completed_at,
                        }
                        if consumption is not None
                        else None
                    ),
                }
            )
        return results
