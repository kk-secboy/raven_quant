from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import insert, or_, select, text, update

from quant_data.database import (
    autopilot_branches,
    backtest_runs,
    jobs,
    open_database,
    parameter_experiments,
    research_campaign_events,
    research_campaigns,
    research_program_events,
    research_programs,
    research_runs,
    schedule_runs,
    schedules,
)

from .schedule_store import (
    LEGACY_RESEARCH_RETIREMENT_LOCK_KEY,
    LEGACY_RESEARCH_SCHEDULE_SUSPENSION,
)

_ACTIVE_JOB_STATUSES = frozenset({"queued", "running"})
_PROTECTED_JOB_KINDS = frozenset(
    {
        "bootstrap",
        "baostock_overlap_validation",
        "core_intraday_download",
        "data_qlib",
        "data_snapshot",
        "data_verify",
        "factor_library_materialize",
        "factor_library_cluster",
        "legacy_market_backfill",
        "margin_eligibility_download",
        "minute_qlib",
        "qlib_baseline",
        "research_asset_acquire",
        "research_report_download",
        "research_report_backfill",
    }
)
_PROTECTED_JOB_PREFIXES = (
    "ashare_",
    "data_",
    "supplemental_",
    "tushare_",
)
def _now() -> datetime:
    return datetime.now(UTC)


def _is_protected_job_kind(kind: str) -> bool:
    normalized = str(kind or "").strip()
    return bool(
        normalized in _PROTECTED_JOB_KINDS
        or normalized.startswith(_PROTECTED_JOB_PREFIXES)
        or normalized.endswith("_download")
    )


def _classify_jobs(
    active_jobs: dict[str, dict[str, str]],
    *,
    autopilot_job_ids: set[str],
) -> tuple[list[str], list[str], list[str]]:
    """Return exclusive, shared, and protected IDs without mutating history."""

    shared = sorted(set(active_jobs) & autopilot_job_ids)
    protected = sorted(
        job_id
        for job_id, item in active_jobs.items()
        if _is_protected_job_kind(item["kind"])
    )
    exclusive = sorted(set(active_jobs) - set(shared) - set(protected))
    return exclusive, shared, protected


def _job_ids(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("job_id") and isinstance(item, str) and item:
                result.add(item)
            result.update(_job_ids(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_job_ids(item))
    return result


class LegacyResearchRetirement:
    """Fail-closed one-time retirement of jobs exclusively owned by old orchestration."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    @staticmethod
    def _plan(connection: Any) -> dict[str, Any]:
        # Scan every historical campaign for durable ownership links.  A
        # succeeded campaign is immutable history, but a schedule or orphaned
        # job it created must not remain a live write path after the cutover.
        all_campaigns = connection.execute(select(research_campaigns)).all()
        campaign_ids = {str(item.id) for item in all_campaigns}
        active_campaigns = [
            item
            for item in all_campaigns
            if str(item.status) in {"queued", "running", "awaiting_approval"}
        ]
        programs = connection.execute(
            select(research_programs).where(
                research_programs.c.status.in_(("active", "paused"))
            )
        ).all()
        autopilot_job_ids = {
            str(item)
            for item in connection.scalars(
                select(autopilot_branches.c.job_id).where(
                    autopilot_branches.c.job_id.is_not(None)
                )
            )
        }
        candidate_job_ids: set[str] = set()
        experiment_ids: set[str] = set()
        backtest_ids: set[str] = set()
        research_run_ids: set[str] = set()
        paper_schedule_ids: set[str] = set()
        for campaign in all_campaigns:
            candidate_job_ids.update(_job_ids(campaign.state_json or {}))
            if campaign.parameter_experiment_id:
                experiment_ids.add(str(campaign.parameter_experiment_id))
            if campaign.backtest_id:
                backtest_ids.add(str(campaign.backtest_id))
            if campaign.research_run_id:
                research_run_ids.add(str(campaign.research_run_id))
            if campaign.paper_schedule_id:
                paper_schedule_ids.add(str(campaign.paper_schedule_id))
        if research_run_ids:
            candidate_job_ids.update(
                str(item)
                for item in connection.scalars(
                    select(research_runs.c.job_id).where(
                        research_runs.c.id.in_(sorted(research_run_ids)),
                        research_runs.c.job_id.is_not(None),
                    )
                )
            )
        if experiment_ids:
            candidate_job_ids.update(
                str(item)
                for item in connection.scalars(
                    select(parameter_experiments.c.job_id).where(
                        parameter_experiments.c.id.in_(sorted(experiment_ids)),
                        parameter_experiments.c.job_id.is_not(None),
                    )
                )
            )
        if backtest_ids:
            candidate_job_ids.update(
                str(item)
                for item in connection.scalars(
                    select(backtest_runs.c.job_id).where(
                        backtest_runs.c.id.in_(sorted(backtest_ids)),
                        backtest_runs.c.job_id.is_not(None),
                    )
                )
            )
        if campaign_ids:
            candidate_job_ids.update(
                str(item)
                for item in connection.scalars(
                    select(jobs.c.id).where(
                        jobs.c.payload_json["research_campaign_id"]
                        .as_string()
                        .in_(sorted(campaign_ids))
                    )
                )
            )
        schedule_ownership = [
            schedules.c.created_by.like("research-campaign:%"),
            schedules.c.payload_json.op("?")("research_campaign_id").is_(True),
        ]
        if paper_schedule_ids:
            schedule_ownership.append(schedules.c.id.in_(sorted(paper_schedule_ids)))
        legacy_schedule_ids = sorted(
            str(item)
            for item in connection.scalars(
                select(schedules.c.id).where(or_(*schedule_ownership))
            )
        )
        legacy_schedule_run_ids: list[str] = []
        if legacy_schedule_ids:
            legacy_schedule_run_ids = sorted(
                str(item)
                for item in connection.scalars(
                    select(schedule_runs.c.id).where(
                        schedule_runs.c.schedule_id.in_(legacy_schedule_ids)
                    )
                )
            )
            candidate_job_ids.update(
                str(item)
                for item in connection.scalars(
                    select(schedule_runs.c.job_id).where(
                        schedule_runs.c.schedule_id.in_(legacy_schedule_ids),
                        schedule_runs.c.job_id.is_not(None),
                    )
                )
            )
        if legacy_schedule_run_ids:
            schedule_idempotency_keys = [
                f"schedule-run:{item}" for item in legacy_schedule_run_ids
            ]
            candidate_job_ids.update(
                str(item)
                for item in connection.scalars(
                    select(jobs.c.id).where(
                        or_(
                            jobs.c.idempotency_key.in_(schedule_idempotency_keys),
                            jobs.c.payload_json["schedule_run_id"]
                            .as_string()
                            .in_(legacy_schedule_run_ids),
                        )
                    )
                )
            )
        active_jobs = {
            str(row.id): {
                "id": str(row.id),
                "kind": str(row.kind),
                "status": str(row.status),
            }
            for row in connection.execute(
                select(jobs).where(
                    jobs.c.id.in_(sorted(candidate_job_ids))
                    if candidate_job_ids
                    else False,
                    jobs.c.status.in_(tuple(_ACTIVE_JOB_STATUSES)),
                )
            )
        }
        exclusive, shared, protected = _classify_jobs(
            active_jobs,
            autopilot_job_ids=autopilot_job_ids,
        )
        return {
            "campaign_ids": [str(item.id) for item in active_campaigns],
            "program_ids": [str(item.id) for item in programs],
            "legacy_schedule_ids": legacy_schedule_ids,
            "exclusive_job_ids": exclusive,
            "shared_autopilot_job_ids": shared,
            "protected_job_ids": protected,
            "jobs": active_jobs,
            # Protected data/materialisation jobs are deliberately left alive;
            # only a job shared with the new Autopilot proves ambiguous
            # ownership and must block the migration.
            "safe_to_apply": not shared,
        }

    def plan(self) -> dict[str, Any]:
        with self.engine.connect() as connection:
            return self._plan(connection)

    @staticmethod
    def _cancel_exclusive_job(connection: Any, job_id: str, *, current: datetime) -> bool:
        """Atomically recheck Autopilot ownership before requesting cancel."""

        row = connection.execute(
            select(jobs.c.kind, jobs.c.status, jobs.c.cancel_requested_at)
            .where(jobs.c.id == job_id)
            .with_for_update()
        ).first()
        if row is None or str(row.status) not in _ACTIVE_JOB_STATUSES:
            return False
        if _is_protected_job_kind(str(row.kind)):
            return False
        shared = connection.execute(
            select(autopilot_branches.c.id)
            .where(autopilot_branches.c.job_id == job_id)
            .limit(1)
        ).first()
        if shared is not None:
            raise ValueError(
                "legacy retirement blocked: a job is also owned by the new Autopilot"
            )
        if str(row.status) == "running" and row.cancel_requested_at is not None:
            return False
        values: dict[str, Any] = {
            "cancel_requested_at": row.cancel_requested_at or current
        }
        if str(row.status) == "queued":
            values.update(
                status="cancelled",
                error="Cancelled by single-Autopilot legacy retirement",
                finished_at=current,
            )
        connection.execute(update(jobs).where(jobs.c.id == job_id).values(**values))
        return True

    @staticmethod
    def _disable_legacy_schedule(
        connection: Any, schedule_id: str, *, current: datetime
    ) -> bool:
        row = connection.execute(
            select(
                schedules.c.status,
                schedules.c.desired_status,
                schedules.c.suspension_reason,
            )
            .where(schedules.c.id == schedule_id)
            .with_for_update()
        ).first()
        if row is None:
            return False
        if (
            str(row.status) == "paused"
            and str(row.desired_status) == "paused"
            and row.suspension_reason == LEGACY_RESEARCH_SCHEDULE_SUSPENSION
        ):
            return False
        connection.execute(
            update(schedules)
            .where(schedules.c.id == schedule_id)
            .values(
                status="paused",
                desired_status="paused",
                suspension_reason=LEGACY_RESEARCH_SCHEDULE_SUSPENSION,
                updated_at=current,
            )
        )
        return True

    @staticmethod
    def _retire_campaign(
        connection: Any, campaign_id: str, *, actor: str, current: datetime
    ) -> bool:
        """Privileged one-time retirement outside the public read-only Store."""

        row = connection.execute(
            select(research_campaigns.c.status)
            .where(research_campaigns.c.id == campaign_id)
            .with_for_update()
        ).first()
        if row is None or str(row.status) not in {
            "queued",
            "running",
            "awaiting_approval",
        }:
            return False
        connection.execute(
            update(research_campaigns)
            .where(research_campaigns.c.id == campaign_id)
            .values(
                status="cancelled",
                lease_until=None,
                next_action_at=current,
                updated_at=current,
                finished_at=current,
            )
        )
        connection.execute(
            insert(research_campaign_events).values(
                campaign_id=campaign_id,
                event_type="campaign.cancelled",
                actor=actor,
                payload_json={"previous_status": str(row.status)},
                created_at=current,
            )
        )
        return True

    @staticmethod
    def _retire_program(
        connection: Any, program_id: str, *, actor: str, current: datetime
    ) -> bool:
        """Privileged one-time retirement outside the public read-only Store."""

        row = connection.execute(
            select(research_programs.c.status)
            .where(research_programs.c.id == program_id)
            .with_for_update()
        ).first()
        if row is None or str(row.status) not in {"active", "paused"}:
            return False
        connection.execute(
            update(research_programs)
            .where(research_programs.c.id == program_id)
            .values(
                status="cancelled",
                lease_until=None,
                next_check_at=current,
                updated_at=current,
            )
        )
        connection.execute(
            insert(research_program_events).values(
                program_id=program_id,
                event_type="program.cancelled",
                actor=actor,
                payload_json={"previous_status": str(row.status)},
                created_at=current,
            )
        )
        return True

    def apply(self, *, actor: str = "single-autopilot-migration") -> dict[str, Any]:
        with self.engine.begin() as connection:
            connection.execute(
                text("SELECT pg_advisory_xact_lock(:lock_key)"),
                {"lock_key": LEGACY_RESEARCH_RETIREMENT_LOCK_KEY},
            )
            plan = self._plan(connection)
            if not plan["safe_to_apply"]:
                raise ValueError(
                    "legacy retirement blocked: a job is also owned by the new Autopilot"
                )
            current = _now()
            cancelled_jobs = [
                job_id
                for job_id in plan["exclusive_job_ids"]
                if self._cancel_exclusive_job(connection, job_id, current=current)
            ]
            disabled_schedules = [
                schedule_id
                for schedule_id in plan["legacy_schedule_ids"]
                if self._disable_legacy_schedule(
                    connection, schedule_id, current=current
                )
            ]
            retired_campaigns = [
                campaign_id
                for campaign_id in plan["campaign_ids"]
                if self._retire_campaign(
                    connection, campaign_id, actor=actor, current=current
                )
            ]
            retired_programs = [
                program_id
                for program_id in plan["program_ids"]
                if self._retire_program(
                    connection, program_id, actor=actor, current=current
                )
            ]
        return {
            **plan,
            "cancelled_job_ids": cancelled_jobs,
            "disabled_schedule_ids": disabled_schedules,
            "retired_campaign_ids": retired_campaigns,
            "retired_program_ids": retired_programs,
            "applied": True,
        }
