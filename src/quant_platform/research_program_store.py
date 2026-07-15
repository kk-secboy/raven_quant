from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import insert, or_, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    open_database,
    research_program_events,
    research_programs,
    row_dict,
)


def _now() -> datetime:
    return datetime.now(UTC)


class ResearchProgramStore:
    """Durable policies that launch campaigns when a compatible Qlib dataset advances."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def create(
        self,
        *,
        name: str,
        recipe_id: str,
        objective: str,
        benchmark: str,
        universe: str,
        dataset_lineage_id: str,
        config: dict[str, Any],
        min_new_trading_days: int,
        max_active_campaigns: int,
        actor: str,
    ) -> dict[str, Any]:
        if min_new_trading_days < 1:
            raise ValueError("min_new_trading_days must be positive")
        if max_active_campaigns < 1:
            raise ValueError("max_active_campaigns must be positive")
        program_id = uuid.uuid4().hex
        current = _now()
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(research_programs).values(
                        id=program_id,
                        name=name.strip(),
                        status="active",
                        recipe_id=recipe_id.strip(),
                        objective=objective.strip(),
                        benchmark=benchmark.strip(),
                        universe=universe.strip(),
                        dataset_lineage_id=dataset_lineage_id.strip(),
                        config_json=config,
                        min_new_trading_days=min_new_trading_days,
                        max_active_campaigns=max_active_campaigns,
                        next_check_at=current,
                        created_by=actor.strip(),
                        created_at=current,
                        updated_at=current,
                    )
                )
                self._event(
                    connection,
                    program_id=program_id,
                    event_type="program.created",
                    actor=actor,
                    payload={
                        "dataset_lineage_id": dataset_lineage_id,
                        "recipe_id": recipe_id,
                    },
                )
        except IntegrityError as exc:
            raise ValueError(f"research program name {name!r} already exists") from exc
        return self.get(program_id)

    def get(self, program_id: str, *, include_events: bool = True) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(research_programs).where(research_programs.c.id == program_id)
            ).first()
            if row is None:
                raise KeyError(program_id)
            events: list[dict[str, Any]] = []
            if include_events:
                events = [
                    self._decode_event(row_dict(item))
                    for item in connection.execute(
                        select(research_program_events)
                        .where(research_program_events.c.program_id == program_id)
                        .order_by(research_program_events.c.created_at.desc())
                        .limit(200)
                    )
                ]
        result = self._decode(row_dict(row))
        if include_events:
            result["events"] = events
        return result

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        statement = (
            select(research_programs).order_by(research_programs.c.created_at.desc()).limit(limit)
        )
        with self.engine.connect() as connection:
            return [self._decode(row_dict(row)) for row in connection.execute(statement)]

    def claim_due(
        self,
        *,
        now: datetime | None = None,
        lease_seconds: int = 120,
    ) -> dict[str, Any] | None:
        current = now or _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(research_programs)
                .where(
                    research_programs.c.status == "active",
                    research_programs.c.next_check_at <= current,
                    or_(
                        research_programs.c.lease_until.is_(None),
                        research_programs.c.lease_until < current,
                    ),
                )
                .order_by(research_programs.c.next_check_at, research_programs.c.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            ).first()
            if row is None:
                return None
            connection.execute(
                update(research_programs)
                .where(research_programs.c.id == row.id)
                .values(
                    lease_until=current + timedelta(seconds=lease_seconds),
                    updated_at=current,
                )
            )
            program_id = str(row.id)
        return self.get(program_id, include_events=False)

    def checked(
        self,
        program_id: str,
        *,
        message: str,
        delay_seconds: int,
    ) -> dict[str, Any]:
        current = _now()
        values: dict[str, Any] = {
            "last_message": message,
            "last_checked_at": current,
            "next_check_at": current + timedelta(seconds=max(1, delay_seconds)),
            "lease_until": None,
            "updated_at": current,
        }
        with self.engine.begin() as connection:
            result = connection.execute(
                update(research_programs)
                .where(research_programs.c.id == program_id)
                .values(**values)
            )
            if not result.rowcount:
                raise KeyError(program_id)
        return self.get(program_id, include_events=False)

    def failed_check(
        self,
        program_id: str,
        *,
        error: str,
        delay_seconds: int,
    ) -> dict[str, Any]:
        """Release the lease and retain an immutable controller-failure record."""

        current = _now()
        message = f"自动检查失败：{error}"
        with self.engine.begin() as connection:
            result = connection.execute(
                update(research_programs)
                .where(research_programs.c.id == program_id)
                .values(
                    last_message=message,
                    last_checked_at=current,
                    next_check_at=current + timedelta(seconds=max(1, delay_seconds)),
                    lease_until=None,
                    updated_at=current,
                )
            )
            if not result.rowcount:
                raise KeyError(program_id)
            self._event(
                connection,
                program_id=program_id,
                event_type="program.check_failed",
                actor="research-program-controller",
                payload={"error": error},
            )
        return self.get(program_id)

    def triggered(
        self,
        program_id: str,
        *,
        campaign_id: str,
        dataset: dict[str, Any],
    ) -> dict[str, Any]:
        current = _now()
        identity = str((dataset.get("provenance") or {})["dataset_identity_sha256"])
        with self.engine.begin() as connection:
            result = connection.execute(
                update(research_programs)
                .where(research_programs.c.id == program_id)
                .values(
                    last_dataset_name=dataset["name"],
                    last_dataset_identity_sha256=identity,
                    last_dataset_end_date=dataset.get("end_date"),
                    last_message=f"已为 {dataset['name']} 创建自动研究活动",
                    last_checked_at=current,
                    last_triggered_at=current,
                    next_check_at=current + timedelta(minutes=5),
                    lease_until=None,
                    updated_at=current,
                )
            )
            if not result.rowcount:
                raise KeyError(program_id)
            self._event(
                connection,
                program_id=program_id,
                event_type="program.triggered",
                actor="research-program-controller",
                payload={
                    "campaign_id": campaign_id,
                    "dataset": dataset["name"],
                    "dataset_identity_sha256": identity,
                },
            )
        return self.get(program_id)

    def record_campaign_outcome(
        self,
        program_id: str,
        *,
        campaign: dict[str, Any],
    ) -> dict[str, Any]:
        if campaign.get("status") != "succeeded":
            raise ValueError("only succeeded campaigns can be recorded")
        version_id = str((campaign.get("state") or {}).get("preferred_version_id") or "")
        if not version_id:
            raise ValueError("campaign has no frozen strategy version")

        current = _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(research_programs)
                .where(research_programs.c.id == program_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(program_id)
            if str(row.last_evaluated_campaign_id or "") == str(campaign["id"]):
                return self.get(program_id)
            connection.execute(
                update(research_programs)
                .where(research_programs.c.id == program_id)
                .values(
                    last_evaluated_campaign_id=str(campaign["id"]),
                    decay_status="legacy",
                    decay_message="final test recorded; never used for cross-campaign selection",
                    updated_at=current,
                )
            )
            self._event(
                connection,
                program_id=program_id,
                event_type="program.final_test_recorded",
                actor="research-program-controller",
                payload={
                    "campaign_id": campaign["id"],
                    "frozen_strategy_version_id": version_id,
                    "used_for_selection": False,
                },
            )
        return self.get(program_id)

    def set_status(self, program_id: str, status: str, *, actor: str) -> dict[str, Any]:
        if status not in {"active", "paused", "cancelled"}:
            raise ValueError("program status must be active, paused, or cancelled")
        current = _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(research_programs)
                .where(research_programs.c.id == program_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(program_id)
            if row.status == "cancelled":
                raise ValueError("cancelled research programs are immutable")
            connection.execute(
                update(research_programs)
                .where(research_programs.c.id == program_id)
                .values(
                    status=status,
                    lease_until=None,
                    next_check_at=current,
                    updated_at=current,
                )
            )
            self._event(
                connection,
                program_id=program_id,
                event_type=f"program.{status}",
                actor=actor,
                payload={"previous_status": str(row.status)},
            )
        return self.get(program_id)

    def check_now(self, program_id: str, *, actor: str) -> dict[str, Any]:
        current = _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(research_programs)
                .where(research_programs.c.id == program_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(program_id)
            if row.status != "active":
                raise ValueError("only active research programs can be checked")
            connection.execute(
                update(research_programs)
                .where(research_programs.c.id == program_id)
                .values(next_check_at=current, lease_until=None, updated_at=current)
            )
            self._event(
                connection,
                program_id=program_id,
                event_type="program.check_requested",
                actor=actor,
                payload={},
            )
        return self.get(program_id)

    @staticmethod
    def _event(
        connection: Any,
        *,
        program_id: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            insert(research_program_events).values(
                program_id=program_id,
                event_type=event_type,
                actor=actor,
                payload_json=payload,
                created_at=_now(),
            )
        )

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        row["config"] = row.pop("config_json")
        return row

    @staticmethod
    def _decode_event(row: dict[str, Any]) -> dict[str, Any]:
        row["payload"] = row.pop("payload_json")
        return row
