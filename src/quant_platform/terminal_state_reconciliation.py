from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, insert, select, update

from quant_data.database import (
    audit_events,
    autopilot_branches,
    jobs,
    model_candidates,
    open_database,
    parameter_experiment_trials,
    parameter_experiments,
    research_events,
    research_run_artifacts,
    research_runs,
)

PARAMETER_PROGRESS_REPAIR_ACTION = "parameter_trial_progress_terminal_reconciled"
RESEARCH_TERMINAL_JOB_REPAIR_ACTION = "research_run_terminal_job_reconciled"
RESEARCH_ORPHAN_REPAIR_ACTION = "platform_model_initialization_orphan_reconciled"


def _now() -> datetime:
    return datetime.now(UTC)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class ReconciliationDecision:
    subject_type: str
    subject_id: str
    classification: str
    repairable: bool
    reason: str
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_complete_failed_progress(
    progress: Mapping[str, Any], *, expected_indexes: Sequence[int]
) -> list[dict[str, Any]]:
    """Return exact failed trial terminals, or reject incomplete evidence.

    Historical progress files do not contain successful trial metrics.  They
    are therefore sufficient only when every preregistered trial failed and
    retained a non-empty execution error.
    """

    indexes = sorted(int(value) for value in expected_indexes)
    trials = progress.get("trials")
    if not isinstance(trials, list) or len(trials) != len(indexes):
        raise ValueError("progress does not contain every preregistered trial")
    if (
        int(progress.get("trial_count", -1)) != len(indexes)
        or int(progress.get("completed_count", -1)) != len(indexes)
        or int(progress.get("succeeded_count", -1)) != 0
        or int(progress.get("failed_count", -1)) != len(indexes)
    ):
        raise ValueError("progress terminal counters are inconsistent")
    by_index: dict[int, dict[str, Any]] = {}
    for raw in trials:
        if not isinstance(raw, Mapping):
            raise ValueError("progress trial entry is invalid")
        trial = dict(raw)
        index = int(trial.get("trial_index", -1))
        warnings = trial.get("warnings")
        error = str(trial.get("error") or "").strip()
        if (
            index in by_index
            or trial.get("status") != "failed"
            or trial.get("score") is not None
            or not isinstance(warnings, list)
            or not error
        ):
            raise ValueError("progress trial is not an exact failed terminal")
        by_index[index] = {
            "trial_index": index,
            "status": "failed",
            "score": None,
            "warnings": list(warnings),
            "error": error,
        }
    if sorted(by_index) != indexes:
        raise ValueError("progress trial indexes differ from preregistration")
    return [by_index[index] for index in indexes]


class TerminalStateReconciler:
    """Repair only historical state that has an exact terminal authority.

    This service deliberately has no age-based generic cleanup.  Unknown
    active rows remain visible for operator review instead of being guessed
    into a terminal state.
    """

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    @staticmethod
    def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
        try:
            payload = path.read_bytes()
            value = json.loads(payload.decode("utf-8"))
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None, None
        return (dict(value), _sha256_bytes(payload)) if isinstance(value, dict) else (None, None)

    def inspect_parameter_experiment(self, experiment_id: str) -> ReconciliationDecision:
        with self.engine.connect() as connection:
            experiment = connection.execute(
                select(parameter_experiments).where(
                    parameter_experiments.c.id == experiment_id
                )
            ).first()
            if experiment is None:
                raise KeyError(experiment_id)
            trials = connection.execute(
                select(parameter_experiment_trials)
                .where(parameter_experiment_trials.c.experiment_id == experiment_id)
                .order_by(parameter_experiment_trials.c.trial_index)
            ).all()
        nonterminal = [
            int(trial.trial_index)
            for trial in trials
            if str(trial.status) in {"queued", "running"}
        ]
        evidence: dict[str, Any] = {
            "experiment_status": str(experiment.status),
            "experiment_finished_at": (
                experiment.finished_at.isoformat() if experiment.finished_at else None
            ),
            "trial_statuses": {
                str(int(trial.trial_index)): str(trial.status) for trial in trials
            },
        }
        if not nonterminal:
            return ReconciliationDecision(
                "parameter_experiment",
                experiment_id,
                "consistent",
                False,
                "all trial rows are already terminal",
                evidence,
            )
        if str(experiment.status) != "failed" or experiment.finished_at is None:
            return ReconciliationDecision(
                "parameter_experiment",
                experiment_id,
                "unknown",
                False,
                "the parent experiment is not a finished failure",
                evidence,
            )
        artifact = Path(str(experiment.artifact_path))
        result_path = artifact / "result.json"
        result, result_sha256 = self._read_json(result_path)
        if result is not None:
            evidence["result_sha256"] = result_sha256
            return ReconciliationDecision(
                "parameter_experiment",
                experiment_id,
                "complete_result_available",
                False,
                "use ParameterExperimentStore.apply_result with the complete result",
                evidence,
            )
        if result_path.exists():
            return ReconciliationDecision(
                "parameter_experiment",
                experiment_id,
                "invalid_result_artifact",
                False,
                "a result artifact exists but cannot be validated as JSON",
                evidence,
            )
        progress_path = artifact / "progress.json"
        progress, progress_sha256 = self._read_json(progress_path)
        if progress is None:
            return ReconciliationDecision(
                "parameter_experiment",
                experiment_id,
                "not_run_or_unknown",
                False,
                (
                    "progress evidence is absent or invalid; trial states cannot be inferred"
                    if progress_path.exists()
                    else (
                        "no complete result or progress ledger exists; "
                        "trial states cannot be inferred"
                    )
                ),
                evidence,
            )
        evidence["progress_sha256"] = progress_sha256
        try:
            failed = validate_complete_failed_progress(
                progress,
                expected_indexes=[int(trial.trial_index) for trial in trials],
            )
        except (TypeError, ValueError) as exc:
            return ReconciliationDecision(
                "parameter_experiment",
                experiment_id,
                "incomplete_progress",
                False,
                str(exc),
                evidence,
            )
        evidence["failed_trials"] = failed
        return ReconciliationDecision(
            "parameter_experiment",
            experiment_id,
            "complete_failed_progress",
            True,
            "every preregistered trial has an exact failed progress terminal",
            evidence,
        )

    def reconcile_parameter_experiment(
        self, experiment_id: str, *, actor: str, apply: bool = False
    ) -> ReconciliationDecision:
        decision = self.inspect_parameter_experiment(experiment_id)
        if not apply or not decision.repairable:
            return decision
        if decision.classification != "complete_failed_progress":
            raise ValueError("parameter experiment has no supported repair authority")
        progress_path = Path(
            self._experiment_artifact_path(experiment_id)
        ) / "progress.json"
        if progress_path.with_name("result.json").exists():
            raise ValueError("a complete result appeared; use the normal result importer")
        payload = progress_path.read_bytes()
        if _sha256_bytes(payload) != decision.evidence["progress_sha256"]:
            raise ValueError("parameter progress evidence changed during reconciliation")
        progress = json.loads(payload.decode("utf-8"))
        now = _now()
        with self.engine.begin() as connection:
            experiment = connection.execute(
                select(parameter_experiments)
                .where(parameter_experiments.c.id == experiment_id)
                .with_for_update()
            ).first()
            trials = connection.execute(
                select(parameter_experiment_trials)
                .where(parameter_experiment_trials.c.experiment_id == experiment_id)
                .order_by(parameter_experiment_trials.c.trial_index)
                .with_for_update()
            ).all()
            if (
                experiment is None
                or str(experiment.status) != "failed"
                or experiment.finished_at is None
            ):
                raise ValueError("parameter experiment changed during reconciliation")
            failed = validate_complete_failed_progress(
                progress,
                expected_indexes=[int(trial.trial_index) for trial in trials],
            )
            by_index = {int(item["trial_index"]): item for item in failed}
            nonterminal_count = 0
            for trial in trials:
                terminal = by_index[int(trial.trial_index)]
                if str(trial.status) == "failed":
                    if (
                        trial.score is not None
                        or trial.metrics_json is not None
                        or list(trial.warnings_json or []) != terminal["warnings"]
                        or str(trial.error or "").strip() != terminal["error"]
                    ):
                        raise ValueError(
                            "existing failed trial differs from progress evidence"
                        )
                    continue
                if str(trial.status) not in {"queued", "running"}:
                    raise ValueError("parameter trial changed during reconciliation")
                nonterminal_count += 1
                connection.execute(
                    update(parameter_experiment_trials)
                    .where(
                        parameter_experiment_trials.c.id == trial.id,
                        parameter_experiment_trials.c.status == trial.status,
                    )
                    .values(
                        status="failed",
                        score=None,
                        metrics_json=None,
                        warnings_json=terminal["warnings"],
                        error=terminal["error"],
                        started_at=trial.started_at or experiment.started_at,
                        finished_at=experiment.finished_at,
                    )
                )
            if not nonterminal_count:
                raise ValueError("parameter trials were already reconciled")
            self._audit(
                connection,
                actor=actor,
                action=PARAMETER_PROGRESS_REPAIR_ACTION,
                path=f"/internal/parameter-experiments/{experiment_id}/reconcile",
                details={
                    "experiment_id": experiment_id,
                    "progress_sha256": decision.evidence["progress_sha256"],
                    "trial_indexes": sorted(by_index),
                },
                created_at=now,
            )
        return self.inspect_parameter_experiment(experiment_id)

    def _experiment_artifact_path(self, experiment_id: str) -> str:
        with self.engine.connect() as connection:
            value = connection.scalar(
                select(parameter_experiments.c.artifact_path).where(
                    parameter_experiments.c.id == experiment_id
                )
            )
        if value is None:
            raise KeyError(experiment_id)
        return str(value)

    def inspect_research_run(
        self,
        run_id: str,
        *,
        orphan_minimum_age: timedelta = timedelta(hours=1),
    ) -> ReconciliationDecision:
        with self.engine.connect() as connection:
            run = connection.execute(
                select(research_runs).where(research_runs.c.id == run_id)
            ).first()
            if run is None:
                raise KeyError(run_id)
            current_job = (
                connection.execute(select(jobs).where(jobs.c.id == run.job_id)).first()
                if run.job_id
                else None
            )
            payload_jobs = connection.execute(
                select(jobs.c.id, jobs.c.status).where(
                    jobs.c.payload_json["research_run_id"].as_string() == run_id
                )
            ).all()
            branches = connection.scalar(
                select(func.count())
                .select_from(autopilot_branches)
                .where(autopilot_branches.c.research_run_id == run_id)
            )
            artifacts = connection.scalar(
                select(func.count())
                .select_from(research_run_artifacts)
                .where(research_run_artifacts.c.research_run_id == run_id)
            )
            candidates = connection.scalar(
                select(func.count())
                .select_from(model_candidates)
                .where(model_candidates.c.research_run_id == run_id)
            )
            event_types = list(
                connection.scalars(
                    select(research_events.c.event_type)
                    .where(research_events.c.research_run_id == run_id)
                    .order_by(research_events.c.created_at)
                )
            )
        evidence = {
            "run_status": str(run.status),
            "run_kind": str(run.kind),
            "job_id": str(run.job_id) if run.job_id else None,
            "job_status": str(current_job.status) if current_job is not None else None,
            "payload_jobs": [
                {"id": str(item.id), "status": str(item.status)} for item in payload_jobs
            ],
            "branch_count": int(branches or 0),
            "artifact_count": int(artifacts or 0),
            "model_candidate_count": int(candidates or 0),
            "event_types": event_types,
        }
        if str(run.status) not in {"queued", "running", "evaluating"}:
            return ReconciliationDecision(
                "research_run", run_id, "consistent", False, "research run is terminal", evidence
            )
        if (
            current_job is not None
            and str(current_job.status) in {"failed", "cancelled"}
            and str((current_job.payload_json or {}).get("research_run_id") or "") == run_id
            and current_job.finished_at is not None
        ):
            evidence["job_error"] = str(current_job.error or "")
            return ReconciliationDecision(
                "research_run",
                run_id,
                "terminal_current_job",
                True,
                "the exact currently attached job is terminal",
                evidence,
            )
        config = dict(run.config_json or {})
        old_enough = _now() - run.created_at >= orphan_minimum_age
        orphaned = (
            str(run.status) == "queued"
            and run.job_id is None
            and str(run.kind).startswith("platform_model_")
            and config.get("contract_version") == "platform-model-tournament-run-v2-horizon"
            and run.created_at == run.updated_at
            and run.started_at is None
            and run.finished_at is None
            and run.error is None
            and old_enough
            and not payload_jobs
            and not branches
            and not artifacts
            and not candidates
            and event_types == ["run.created"]
        )
        if orphaned:
            return ReconciliationDecision(
                "research_run",
                run_id,
                "unstarted_platform_initialization",
                True,
                "no job, branch, artifact, candidate, or post-create event exists",
                evidence,
            )
        return ReconciliationDecision(
            "research_run",
            run_id,
            "unknown_or_active",
            False,
            "no exact terminal authority exists; leave the run unchanged",
            evidence,
        )

    def reconcile_research_run(
        self,
        run_id: str,
        *,
        actor: str,
        apply: bool = False,
        orphan_minimum_age: timedelta = timedelta(hours=1),
    ) -> ReconciliationDecision:
        decision = self.inspect_research_run(
            run_id, orphan_minimum_age=orphan_minimum_age
        )
        if not apply or not decision.repairable:
            return decision
        if decision.classification == "terminal_current_job":
            error = str(decision.evidence.get("job_error") or "terminal research job failed")
            action = RESEARCH_TERMINAL_JOB_REPAIR_ACTION
        elif decision.classification == "unstarted_platform_initialization":
            error = (
                "platform model lane initialization was abandoned before any job, branch, "
                "artifact, candidate, or post-create event was recorded"
            )
            action = RESEARCH_ORPHAN_REPAIR_ACTION
        else:
            raise ValueError("research run has no supported repair authority")
        now = _now()
        with self.engine.begin() as connection:
            run = connection.execute(
                select(research_runs)
                .where(research_runs.c.id == run_id)
                .with_for_update()
            ).first()
            if run is None or str(run.status) not in {"queued", "running", "evaluating"}:
                raise ValueError("research run changed during reconciliation")
            expected_job_id = decision.evidence.get("job_id")
            if decision.classification == "terminal_current_job":
                current_job = connection.execute(
                    select(jobs)
                    .where(jobs.c.id == expected_job_id)
                    .with_for_update()
                ).first()
                if (
                    current_job is None
                    or str(current_job.status) not in {"failed", "cancelled"}
                    or str(
                        (current_job.payload_json or {}).get("research_run_id") or ""
                    )
                    != run_id
                    or current_job.finished_at is None
                ):
                    raise ValueError("research job changed during reconciliation")
            else:
                config = dict(run.config_json or {})
                payload_job_count = connection.scalar(
                    select(func.count())
                    .select_from(jobs)
                    .where(jobs.c.payload_json["research_run_id"].as_string() == run_id)
                )
                branch_count = connection.scalar(
                    select(func.count())
                    .select_from(autopilot_branches)
                    .where(autopilot_branches.c.research_run_id == run_id)
                )
                artifact_count = connection.scalar(
                    select(func.count())
                    .select_from(research_run_artifacts)
                    .where(research_run_artifacts.c.research_run_id == run_id)
                )
                candidate_count = connection.scalar(
                    select(func.count())
                    .select_from(model_candidates)
                    .where(model_candidates.c.research_run_id == run_id)
                )
                event_types = list(
                    connection.scalars(
                        select(research_events.c.event_type)
                        .where(research_events.c.research_run_id == run_id)
                        .order_by(research_events.c.created_at)
                    )
                )
                if not (
                    str(run.status) == "queued"
                    and run.job_id is None
                    and str(run.kind).startswith("platform_model_")
                    and config.get("contract_version")
                    == "platform-model-tournament-run-v2-horizon"
                    and run.created_at == run.updated_at
                    and run.started_at is None
                    and run.finished_at is None
                    and run.error is None
                    and _now() - run.created_at >= orphan_minimum_age
                    and not payload_job_count
                    and not branch_count
                    and not artifact_count
                    and not candidate_count
                    and event_types == ["run.created"]
                ):
                    raise ValueError(
                        "platform model initialization evidence changed during reconciliation"
                    )
            result = connection.execute(
                update(research_runs)
                .where(
                    research_runs.c.id == run_id,
                    research_runs.c.status == run.status,
                    research_runs.c.job_id.is_(None)
                    if expected_job_id is None
                    else research_runs.c.job_id == expected_job_id,
                    research_runs.c.updated_at == run.updated_at,
                )
                .values(status="failed", error=error, finished_at=now, updated_at=now)
            )
            if int(result.rowcount or 0) != 1:
                raise ValueError("research run changed during reconciliation")
            connection.execute(
                insert(research_events).values(
                    research_run_id=run_id,
                    factor_candidate_id=None,
                    event_type="run.failed",
                    actor=actor,
                    payload_json={"error": error, "reconciliation_action": action},
                    created_at=now,
                )
            )
            self._audit(
                connection,
                actor=actor,
                action=action,
                path=f"/internal/research-runs/{run_id}/reconcile",
                details={"research_run_id": run_id, **decision.evidence},
                created_at=now,
            )
        return self.inspect_research_run(run_id, orphan_minimum_age=orphan_minimum_age)

    @staticmethod
    def _audit(
        connection: Any,
        *,
        actor: str,
        action: str,
        path: str,
        details: Mapping[str, Any],
        created_at: datetime,
    ) -> None:
        connection.execute(
            insert(audit_events).values(
                user_id=None,
                username=actor,
                action=action,
                method="INTERNAL",
                path=path,
                status_code=200,
                ip_hash=None,
                user_agent="quantlab-terminal-state-reconciler",
                details_json=dict(details),
                created_at=created_at,
            )
        )
