from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, insert, or_, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    open_database,
    research_campaign_events,
    research_campaigns,
    row_dict,
)

ACTIVE_CAMPAIGN_STATUSES = ("queued", "running", "awaiting_approval")
TERMINAL_CAMPAIGN_STATUSES = ("succeeded", "failed", "cancelled")


def _now() -> datetime:
    return datetime.now(UTC)


class ResearchCampaignStore:
    """Durable control-plane state for restart-safe autonomous research."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def create(
        self,
        *,
        name: str,
        objective: str,
        dataset: str,
        benchmark: str,
        universe: str,
        recipe_id: str,
        config: dict[str, Any],
        actor: str,
        research_program_id: str | None = None,
        dataset_identity_sha256: str | None = None,
    ) -> dict[str, Any]:
        campaign_id = uuid.uuid4().hex
        current = _now()
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(research_campaigns).values(
                        id=campaign_id,
                        name=name.strip(),
                        status="queued",
                        stage="research",
                        objective=objective.strip(),
                        dataset=dataset.strip(),
                        benchmark=benchmark.strip(),
                        universe=universe.strip(),
                        recipe_id=recipe_id.strip(),
                        research_program_id=research_program_id,
                        dataset_identity_sha256=dataset_identity_sha256,
                        config_json=config,
                        state_json={},
                        attempts=0,
                        next_action_at=current,
                        created_by=actor.strip(),
                        created_at=current,
                        updated_at=current,
                    )
                )
                self._event(
                    connection,
                    campaign_id=campaign_id,
                    event_type="campaign.created",
                    actor=actor,
                    payload={"stage": "research", "dataset": dataset},
                )
        except IntegrityError as exc:
            raise ValueError(f"research campaign name {name!r} already exists") from exc
        return self.get(campaign_id)

    def active_count(self, *, research_program_id: str) -> int:
        with self.engine.connect() as connection:
            return int(
                connection.scalar(
                    select(func.count())
                    .select_from(research_campaigns)
                    .where(
                        research_campaigns.c.research_program_id == research_program_id,
                        research_campaigns.c.status.in_(ACTIVE_CAMPAIGN_STATUSES),
                    )
                )
                or 0
            )

    def for_program_dataset(
        self,
        *,
        research_program_id: str,
        dataset_identity_sha256: str,
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(research_campaigns).where(
                    research_campaigns.c.research_program_id == research_program_id,
                    research_campaigns.c.dataset_identity_sha256
                    == dataset_identity_sha256,
                )
            ).first()
        return self._decode(row_dict(row)) if row is not None else None

    def latest_completed_for_program(
        self,
        research_program_id: str,
        *,
        exclude_campaign_id: str | None = None,
    ) -> dict[str, Any] | None:
        statement = (
            select(research_campaigns)
            .where(
                research_campaigns.c.research_program_id == research_program_id,
                research_campaigns.c.status == "succeeded",
            )
            .order_by(
                research_campaigns.c.finished_at.desc(),
                research_campaigns.c.created_at.desc(),
            )
            .limit(1)
        )
        if exclude_campaign_id:
            statement = statement.where(research_campaigns.c.id != exclude_campaign_id)
        with self.engine.connect() as connection:
            row = connection.execute(statement).first()
        return self._decode(row_dict(row)) if row is not None else None

    def get(self, campaign_id: str, *, include_events: bool = True) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(research_campaigns).where(research_campaigns.c.id == campaign_id)
            ).first()
            if row is None:
                raise KeyError(campaign_id)
            events = []
            if include_events:
                events = [
                    self._decode_event(row_dict(item))
                    for item in connection.execute(
                        select(research_campaign_events)
                        .where(research_campaign_events.c.campaign_id == campaign_id)
                        .order_by(research_campaign_events.c.created_at.desc())
                        .limit(200)
                    )
                ]
        result = self._decode(row_dict(row))
        if include_events:
            result["events"] = events
        return result

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        statement = (
            select(research_campaigns)
            .order_by(research_campaigns.c.created_at.desc())
            .limit(limit)
        )
        with self.engine.connect() as connection:
            return [
                self._decode(row_dict(row)) for row in connection.execute(statement)
            ]

    def claim_due(
        self,
        *,
        now: datetime | None = None,
        lease_seconds: int = 120,
    ) -> dict[str, Any] | None:
        current = now or _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(research_campaigns)
                .where(
                    research_campaigns.c.status.in_(ACTIVE_CAMPAIGN_STATUSES),
                    research_campaigns.c.next_action_at <= current,
                    or_(
                        research_campaigns.c.lease_until.is_(None),
                        research_campaigns.c.lease_until < current,
                    ),
                )
                .order_by(research_campaigns.c.next_action_at, research_campaigns.c.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            ).first()
            if row is None:
                return None
            values: dict[str, Any] = {
                "lease_until": current + timedelta(seconds=lease_seconds),
                "attempts": int(row.attempts) + 1,
                "updated_at": current,
            }
            if row.status == "queued":
                values["status"] = "running"
            connection.execute(
                update(research_campaigns)
                .where(research_campaigns.c.id == row.id)
                .values(**values)
            )
            campaign_id = str(row.id)
        return self.get(campaign_id, include_events=False)

    def transition(
        self,
        campaign_id: str,
        *,
        stage: str | None = None,
        status: str | None = None,
        actor: str = "research-orchestrator",
        event_type: str,
        payload: dict[str, Any] | None = None,
        state_patch: dict[str, Any] | None = None,
        links: dict[str, str | None] | None = None,
        delay_seconds: int = 0,
        error: str | None = None,
    ) -> dict[str, Any]:
        current = _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(research_campaigns)
                .where(research_campaigns.c.id == campaign_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(campaign_id)
            state = dict(row.state_json or {})
            state.update(state_patch or {})
            values: dict[str, Any] = {
                "state_json": state,
                "lease_until": None,
                "next_action_at": current + timedelta(seconds=max(0, delay_seconds)),
                "updated_at": current,
                "error": error,
            }
            if stage is not None:
                values["stage"] = stage
            if status is not None:
                values["status"] = status
                if status in TERMINAL_CAMPAIGN_STATUSES:
                    values["finished_at"] = current
            allowed_links = {
                "research_run_id",
                "strategy_id",
                "strategy_version_id",
                "parameter_experiment_id",
                "backtest_id",
                "paper_portfolio_id",
                "paper_schedule_id",
            }
            for key, value in (links or {}).items():
                if key not in allowed_links:
                    raise ValueError(f"unsupported campaign link: {key}")
                values[key] = value
            connection.execute(
                update(research_campaigns)
                .where(research_campaigns.c.id == campaign_id)
                .values(**values)
            )
            self._event(
                connection,
                campaign_id=campaign_id,
                event_type=event_type,
                actor=actor,
                payload=payload or {},
            )
        return self.get(campaign_id)

    def defer(
        self,
        campaign_id: str,
        *,
        seconds: int = 30,
        reason: str | None = None,
    ) -> None:
        current = _now()
        with self.engine.begin() as connection:
            result = connection.execute(
                update(research_campaigns)
                .where(research_campaigns.c.id == campaign_id)
                .values(
                    lease_until=None,
                    next_action_at=current + timedelta(seconds=max(1, seconds)),
                    updated_at=current,
                )
            )
            if not result.rowcount:
                raise KeyError(campaign_id)
            if reason:
                self._event(
                    connection,
                    campaign_id=campaign_id,
                    event_type="campaign.deferred",
                    actor="research-orchestrator",
                    payload={"reason": reason, "retry_in_seconds": seconds},
                )

    def fail(self, campaign_id: str, error: str) -> dict[str, Any]:
        return self.transition(
            campaign_id,
            status="failed",
            event_type="campaign.failed",
            payload={"error": error},
            error=error,
        )

    def retry(self, campaign_id: str, *, actor: str) -> dict[str, Any]:
        current = _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(research_campaigns)
                .where(research_campaigns.c.id == campaign_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(campaign_id)
            if row.status != "failed":
                raise ValueError("only failed research campaigns may be retried")
            connection.execute(
                update(research_campaigns)
                .where(research_campaigns.c.id == campaign_id)
                .values(
                    status="running",
                    error=None,
                    lease_until=None,
                    next_action_at=current,
                    finished_at=None,
                    updated_at=current,
                )
            )
            self._event(
                connection,
                campaign_id=campaign_id,
                event_type="campaign.retried",
                actor=actor,
                payload={"stage": str(row.stage)},
            )
        return self.get(campaign_id)

    def set_status(
        self, campaign_id: str, status: str, *, actor: str
    ) -> dict[str, Any]:
        if status not in {"paused", "running", "cancelled"}:
            raise ValueError("campaign status must be paused, running, or cancelled")
        current = _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(research_campaigns)
                .where(research_campaigns.c.id == campaign_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(campaign_id)
            if row.status in TERMINAL_CAMPAIGN_STATUSES:
                raise ValueError(f"campaign is already {row.status}")
            connection.execute(
                update(research_campaigns)
                .where(research_campaigns.c.id == campaign_id)
                .values(
                    status=status,
                    lease_until=None,
                    next_action_at=current,
                    updated_at=current,
                    finished_at=current if status == "cancelled" else None,
                )
            )
            self._event(
                connection,
                campaign_id=campaign_id,
                event_type=f"campaign.{status}",
                actor=actor,
                payload={"previous_status": str(row.status)},
            )
        return self.get(campaign_id)

    @staticmethod
    def _event(
        connection: Any,
        *,
        campaign_id: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            insert(research_campaign_events).values(
                campaign_id=campaign_id,
                event_type=event_type,
                actor=actor,
                payload_json=payload,
                created_at=_now(),
            )
        )

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        row["config"] = row.pop("config_json")
        row["state"] = row.pop("state_json")
        return row

    @staticmethod
    def _decode_event(row: dict[str, Any]) -> dict[str, Any]:
        row["payload"] = row.pop("payload_json")
        return row
