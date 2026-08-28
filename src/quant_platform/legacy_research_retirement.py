from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update

from quant_data.database import (
    autopilot_branches,
    backtest_runs,
    jobs,
    open_database,
    parameter_experiments,
    research_campaigns,
    research_programs,
    research_runs,
)

from .research_campaign_store import ResearchCampaignStore
from .research_program_store import ResearchProgramStore

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
        self.campaigns = ResearchCampaignStore(database_url)
        self.programs = ResearchProgramStore(database_url)

    def plan(self) -> dict[str, Any]:
        with self.engine.connect() as connection:
            campaigns = connection.execute(
                select(research_campaigns).where(
                    research_campaigns.c.status.in_(("queued", "running", "awaiting_approval"))
                )
            ).all()
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
            for campaign in campaigns:
                candidate_job_ids.update(_job_ids(campaign.state_json or {}))
                if campaign.parameter_experiment_id:
                    experiment_ids.add(str(campaign.parameter_experiment_id))
                if campaign.backtest_id:
                    backtest_ids.add(str(campaign.backtest_id))
                if campaign.research_run_id:
                    run = connection.execute(
                        select(research_runs.c.job_id).where(
                            research_runs.c.id == campaign.research_run_id
                        )
                    ).first()
                    if run and run.job_id:
                        candidate_job_ids.add(str(run.job_id))
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
            active_jobs = {
                str(row.id): {"id": str(row.id), "kind": str(row.kind), "status": str(row.status)}
                for row in connection.execute(
                    select(jobs).where(
                        jobs.c.id.in_(sorted(candidate_job_ids)) if candidate_job_ids else False,
                        jobs.c.status.in_(tuple(_ACTIVE_JOB_STATUSES)),
                    )
                )
            }
        exclusive, shared, protected = _classify_jobs(
            active_jobs,
            autopilot_job_ids=autopilot_job_ids,
        )
        return {
            "campaign_ids": [str(item.id) for item in campaigns],
            "program_ids": [str(item.id) for item in programs],
            "exclusive_job_ids": exclusive,
            "shared_autopilot_job_ids": shared,
            "protected_job_ids": protected,
            "jobs": active_jobs,
            # Protected data/materialisation jobs are deliberately left alive;
            # only a job shared with the new Autopilot proves ambiguous
            # ownership and must block the migration.
            "safe_to_apply": not shared,
        }

    def _cancel_exclusive_job(self, job_id: str) -> bool:
        """Atomically recheck Autopilot ownership before requesting cancel."""

        with self.engine.begin() as connection:
            row = connection.execute(
                select(jobs.c.kind, jobs.c.status)
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
            current = _now()
            values: dict[str, Any] = {"cancel_requested_at": current}
            if str(row.status) == "queued":
                values.update(
                    status="cancelled",
                    error="Cancelled by single-Autopilot legacy retirement",
                    finished_at=current,
                )
            connection.execute(
                update(jobs).where(jobs.c.id == job_id).values(**values)
            )
        return True

    def apply(self, *, actor: str = "single-autopilot-migration") -> dict[str, Any]:
        plan = self.plan()
        if not plan["safe_to_apply"]:
            raise ValueError(
                "legacy retirement blocked: a job is also owned by the new Autopilot"
            )
        cancelled_jobs: list[str] = []
        for job_id in plan["exclusive_job_ids"]:
            if self._cancel_exclusive_job(job_id):
                cancelled_jobs.append(job_id)
        retired_campaigns: list[str] = []
        for campaign_id in plan["campaign_ids"]:
            self.campaigns.set_status(campaign_id, "cancelled", actor=actor)
            retired_campaigns.append(campaign_id)
        retired_programs: list[str] = []
        for program_id in plan["program_ids"]:
            self.programs.set_status(program_id, "cancelled", actor=actor)
            retired_programs.append(program_id)
        return {
            **plan,
            "cancelled_job_ids": cancelled_jobs,
            "retired_campaign_ids": retired_campaigns,
            "retired_program_ids": retired_programs,
            "applied": True,
        }
