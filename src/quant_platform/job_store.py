from __future__ import annotations

import hashlib
import json
import uuid
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import case, cast, func, literal, select, text, update
from sqlalchemy.dialects.postgresql import JSONPATH
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from quant_data.database import backtest_runs, jobs, open_database, row_dict
from quant_platform.jsonb_safety import normalize_jsonb_document

EVALUATION_STATUS_COUNTS_KEY = "_quantlab_evaluation_status_counts"
MAX_NUMERICAL_THREADS_PER_JOB = 8
ORDER_PLAN_MATERIALIZATION_STATUS_KEY = "simulation_batch_materialization_status"
ORDER_PLAN_EXECUTION_TRADE_DATE_KEY = "simulation_batch_execution_trade_date"
ORDER_PLAN_AWAITING_EXECUTION_DATA = "awaiting_execution_data"
ORDER_PLAN_MATERIALIZED = "materialized"
ORDER_PLAN_CANCELLED = "cancelled"
ORDER_PLAN_SUPERSEDED = "superseded"
ORDER_PLAN_FAILED = "failed"
INTERRUPTED_ATTEMPT_EXHAUSTED_ERROR = (
    "Worker restarted after the bounded attempt limit; operator review is required"
)
_PENDING_V17_INTERRUPTION_RECOVERY_JOB_ID = "858a75a6f1994c359fa9c3567ed09f57"

# Global heavy-work CPU tokens.  This is deliberately independent of Docker's
# per-container ceiling: several individually capped containers can still
# overcommit one host when they run together.  Full daily/minute Qlib
# publication participates in the same ledger after an unbounded data_qlib
# attempt starved the API and SSH on a 32-core host.  The live paper lifecycle
# remains outside this ledger and therefore retains its service reserve.
RESEARCH_JOB_CPU_COST = {
    "data_qlib": 16,
    "minute_qlib": 16,
    "rdagent_run": 8,
    "rdagent_factor_report": 4,
    "rdagent_quant": 12,
    "minute_research": 4,
    "qlib_baseline": 8,
    "external_factor_evaluate": 4,
    "information_factor_evaluate": 4,
    "multiface_audit": 4,
    "factor_library_materialize": 8,
    "factor_library_cluster": 8,
    "factor_sota_evaluate": 8,
    "factor_evaluate": 8,
    "model_evaluate": 8,
    "model_ensemble_evaluate": 8,
    "quant_bundle_evaluate": 12,
    "strategy_backtest": 8,
    "strategy_health_collect": 4,
    "parameter_experiment": 8,
}

RESEARCH_JOB_MEMORY_GB = {
    "data_qlib": 24,
    "minute_qlib": 24,
    "rdagent_run": 12,
    "rdagent_factor_report": 8,
    "rdagent_quant": 20,
    "minute_research": 8,
    # The full Alpha158/LightGBM baseline reached about 36.4 GiB RSS on the
    # production 32-core/64-GiB host.  Charge the full 40-GiB research budget
    # so this memory-heavy baseline is exclusive among governed research jobs.
    "qlib_baseline": 40,
    "external_factor_evaluate": 8,
    "information_factor_evaluate": 8,
    "multiface_audit": 8,
    "factor_library_materialize": 12,
    "factor_library_cluster": 12,
    "factor_sota_evaluate": 40,
    "factor_evaluate": 12,
    "model_evaluate": 40,
    "model_ensemble_evaluate": 16,
    "quant_bundle_evaluate": 40,
    "strategy_backtest": 40,
    "strategy_health_collect": 8,
    "parameter_experiment": 12,
}


def research_job_cpu_cost(kind: str) -> int:
    return int(RESEARCH_JOB_CPU_COST.get(str(kind), 0))


def research_job_memory_gb(kind: str) -> int:
    return int(RESEARCH_JOB_MEMORY_GB.get(str(kind), 0))


def _with_evaluation_status_counts(
    document: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Persist bounded evaluation counters beside potentially large evidence.

    The complete evaluation array remains the immutable worker result, but
    operational job polling must not transfer and deserialize that array merely
    to render a one-line outcome. New writes therefore carry small counters;
    read projections retain a DB-side fallback for historical rows.
    """

    if document is None:
        return None
    evaluations = document.get("evaluations")
    if not isinstance(evaluations, list):
        return document
    counts = Counter(
        str(item.get("status"))
        for item in evaluations
        if isinstance(item, dict) and isinstance(item.get("status"), str)
    )
    document[EVALUATION_STATUS_COUNTS_KEY] = dict(sorted(counts.items()))
    return document


FORMAL_DATA_AUTO_RETRY_KINDS = frozenset(
    {
        # Formal data, information-factor, and Qlib publication jobs can all
        # be submitted as a durable pipeline root. Successors already inherit
        # the root's bound max_attempts value in LocalJobWorker; keeping every
        # possible root here prevents an otherwise identical chain from
        # silently dropping to one execution when it starts at a later stage.
        "announcement_factor_register",
        "announcement_nlp",
        "ashare_5m_download",
        "baostock_overlap_validation",
        "bootstrap",
        "cninfo_announcements_download",
        "core_intraday_download",
        "corpus_factor_register",
        "corpus_nlp",
        "data_qlib",
        "data_snapshot",
        "data_verify",
        "event_market_response",
        "information_factor_evaluate",
        "legacy_market_backfill",
        "major_news_mentions",
        "major_news_mentions_factor_register",
        "margin_eligibility_download",
        "minute_qlib",
        "minute_research",
        "multiface_audit",
        "news_flash_factor_register",
        "news_flash_factors",
        "qlib_baseline",
        "report_rc_factor_register",
        "report_rc_factors",
        "supplemental_cn_capital_flow",
        "supplemental_cn_derivatives_enhanced",
        "supplemental_cn_extended_daily",
        "supplemental_cn_fund_index_enhanced",
        "supplemental_cn_funds",
        "supplemental_cn_futures",
        "supplemental_cn_governance_risk",
        "supplemental_cn_institutional",
        "supplemental_cn_macro",
        "supplemental_cn_options_bonds",
        "supplemental_download",
        "supplemental_global_markets",
        "supplemental_global_rates_enhanced",
        "supplemental_hk_market",
        "supplemental_research_corpus",
        "supplemental_strategy_specialty",
        "supplemental_strategy_specialty_minutes",
        "supplemental_us_market",
    }
)

AUTO_RETRY_ATTEMPTS = {
    **{kind: 3 for kind in FORMAL_DATA_AUTO_RETRY_KINDS},
    "research_asset_acquire": 3,
    "rdagent_run": 3,
    "rdagent_quant": 3,
    "rdagent_factor_report": 3,
    "factor_sota_evaluate": 2,
    "factor_library_materialize": 2,
    "factor_library_cluster": 2,
    "model_refit": 3,
    "recommendation_refresh": 3,
    "simulation_order_plan": 3,
    "simulation_replay": 3,
    "strategy_health_collect": 2,
}


def _now() -> datetime:
    return datetime.now(UTC)


def research_asset_acquisition_idempotency_key(
    *,
    research_day: str,
    snapshot_name: str,
    include_tushare: bool,
    include_arxiv: bool,
    report_date: str | None = None,
) -> str:
    """Bind one automatic acquisition identity to its complete source contract."""

    contract = {
        "contract_version": "research-asset-acquisition-job-v1",
        "as_of": str(research_day),
        "snapshot_name": str(snapshot_name),
        "include_tushare": bool(include_tushare),
        "include_arxiv": bool(include_arxiv),
        "report_date": str(report_date or ""),
    }
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"research-assets:auto:{research_day}:{digest}"


class JobStore:
    """PostgreSQL-backed durable job repository for API and worker processes."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def recover_interrupted(self, allowed_kinds: tuple[str, ...] = ()) -> int:
        return self._recover_interrupted(allowed_kinds)

    def interrupted_dependency_failures(
        self, allowed_kinds: tuple[str, ...] = ()
    ) -> list[dict[str, Any]]:
        """Return exact restart terminals whose domain state must be projected.

        This query intentionally includes terminals created by an earlier
        process.  Domain projection and capital-OOS failure settlement are
        idempotent, so a crash immediately after queue recovery cannot strand
        them.  The one known v17 source remains untouched at attempts 1/1 until
        its dedicated append-only recovery transaction seals the running row.
        """

        predicate = [
            jobs.c.status == "failed",
            jobs.c.exit_code == 143,
            jobs.c.error == INTERRUPTED_ATTEMPT_EXHAUSTED_ERROR,
            ~(
                (jobs.c.id == _PENDING_V17_INTERRUPTION_RECOVERY_JOB_ID)
                & (jobs.c.attempts == 1)
                & (jobs.c.max_attempts == 1)
            ),
        ]
        if allowed_kinds:
            predicate.append(jobs.c.kind.in_(allowed_kinds))
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(jobs).where(*predicate).order_by(jobs.c.finished_at, jobs.c.id)
            ).all()
        return [self._decode(row_dict(row)) for row in rows]

    def _recover_interrupted(
        self, allowed_kinds: tuple[str, ...]
    ) -> int:
        predicate = [jobs.c.status == "running"]
        if allowed_kinds:
            predicate.append(jobs.c.kind.in_(allowed_kinds))
        now = _now()
        with self.engine.begin() as connection:
            # A formal backtest is one-shot after its first claim.  Requeueing
            # one with attempts < max_attempts would strand it forever because
            # claim_next deliberately excludes every repeated formal attempt.
            exhausted_rows = connection.execute(
                update(jobs)
                .where(
                    *predicate,
                    (jobs.c.kind == "strategy_backtest")
                    | (jobs.c.attempts >= jobs.c.max_attempts),
                )
                .values(
                    status="failed",
                    exit_code=143,
                    error=INTERRUPTED_ATTEMPT_EXHAUSTED_ERROR,
                    cancel_requested_at=None,
                    next_attempt_at=None,
                    finished_at=now,
                )
                .returning(*jobs.c)
            ).all()
            exhausted_ids = [str(row.id) for row in exhausted_rows]
            if exhausted_ids:
                connection.execute(
                    update(backtest_runs)
                    .where(
                        backtest_runs.c.job_id.in_(exhausted_ids),
                        backtest_runs.c.status.in_(("queued", "running")),
                    )
                    .values(
                        status="failed",
                        error=INTERRUPTED_ATTEMPT_EXHAUSTED_ERROR,
                        finished_at=now,
                    )
                )
            recoverable = connection.execute(
                update(jobs)
                .where(
                    *predicate,
                    jobs.c.kind != "strategy_backtest",
                    jobs.c.attempts < jobs.c.max_attempts,
                )
                .values(
                    status="queued",
                    started_at=None,
                    # Let PostgreSQL, the API and readiness probes settle before
                    # a full-market build can reclaim its bounded resources.
                    next_attempt_at=_now() + timedelta(seconds=120),
                    error="Worker restarted; job safely requeued after warm-up",
                )
            )
        return len(exhausted_rows) + int(recoverable.rowcount or 0)

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
                        select(jobs.c.id, jobs.c.kind, jobs.c.payload_json).where(
                            jobs.c.idempotency_key == idempotency_key
                        )
                    ).first()
                    if existing:
                        if str(existing.kind) != kind or dict(existing.payload_json) != payload:
                            raise ValueError(
                                "idempotency key is already bound to a different job payload"
                            )
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
                        select(jobs.c.id, jobs.c.kind, jobs.c.payload_json).where(
                            jobs.c.idempotency_key == idempotency_key
                        )
                    ).first()
                if existing:
                    if str(existing.kind) != kind or dict(existing.payload_json) != payload:
                        raise ValueError(
                            "idempotency key is already bound to a different job payload"
                        ) from exc
                    return self.get(str(existing.id))
            raise ValueError(f"could not create {kind} job") from exc
        return self.get(existing_id or job_id)

    def claim_next(
        self,
        allowed_kinds: tuple[str, ...] = (),
        *,
        research_cpu_budget: int = 0,
        research_memory_budget_gb: int = 0,
    ) -> dict[str, Any] | None:
        statement = select(jobs).where(
            jobs.c.status == "queued",
            (jobs.c.next_attempt_at.is_(None)) | (jobs.c.next_attempt_at <= _now()),
            ~((jobs.c.kind == "strategy_backtest") & (jobs.c.attempts >= 1)),
        )
        if allowed_kinds:
            statement = statement.where(jobs.c.kind.in_(allowed_kinds))
        governed_resources = research_cpu_budget > 0 or research_memory_budget_gb > 0
        candidate_limit = 100 if governed_resources else 1
        statement = (
            statement.order_by(jobs.c.created_at)
            .limit(candidate_limit)
            .with_for_update(skip_locked=True)
        )
        with self.engine.begin() as connection:
            if governed_resources:
                connection.execute(
                    text(
                        "SELECT pg_advisory_xact_lock("
                        "hashtext('quantlab-research-cpu-budget'))"
                    )
                )
                running_kinds = connection.scalars(
                    select(jobs.c.kind).where(jobs.c.status == "running")
                ).all()
                used_cpu = sum(research_job_cpu_cost(kind) for kind in running_kinds)
                used_memory_gb = sum(
                    research_job_memory_gb(kind) for kind in running_kinds
                )
            else:
                used_cpu = 0
                used_memory_gb = 0
            rows = connection.execute(statement).all()
            row = next(
                (
                    item
                    for item in rows
                    if (
                        (
                            research_cpu_budget <= 0
                            or research_job_cpu_cost(str(item.kind)) == 0
                            or used_cpu + research_job_cpu_cost(str(item.kind))
                            <= research_cpu_budget
                        )
                        and (
                            research_memory_budget_gb <= 0
                            or research_job_memory_gb(str(item.kind)) == 0
                            or used_memory_gb + research_job_memory_gb(str(item.kind))
                            <= research_memory_budget_gb
                        )
                    )
                ),
                None,
            )
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
        persisted_result = _with_evaluation_status_counts(normalize_jsonb_document(result))
        with self.engine.begin() as connection:
            connection.execute(
                update(jobs)
                .where(jobs.c.id == job_id)
                .values(
                    status=status,
                    exit_code=exit_code,
                    error=error,
                    progress_json=persisted_result,
                    cancel_requested_at=None,
                    next_attempt_at=None,
                    finished_at=_now(),
                )
            )

    def update_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        """Persist a live subprocess progress snapshot without changing job state."""

        persisted_progress = _with_evaluation_status_counts(normalize_jsonb_document(progress))
        with self.engine.begin() as connection:
            updated = connection.execute(
                update(jobs)
                .where(jobs.c.id == job_id, jobs.c.status == "running")
                .values(progress_json=persisted_progress)
            )
        if not int(updated.rowcount or 0):
            current = self.get(job_id)
            if current["status"] != "running":
                return
            raise KeyError(job_id)

    def awaiting_simulation_order_plans(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return successful immutable plans still waiting for their D+1 dataset."""

        statement = (
            select(
                jobs.c.id,
                jobs.c.payload_json,
                jobs.c.progress_json,
                jobs.c.created_at,
            )
            .where(
                jobs.c.kind == "simulation_order_plan",
                jobs.c.status == "succeeded",
                jobs.c.progress_json[ORDER_PLAN_MATERIALIZATION_STATUS_KEY].as_string()
                == ORDER_PLAN_AWAITING_EXECUTION_DATA,
            )
            .order_by(jobs.c.created_at)
            .limit(limit)
        )
        with self.engine.connect() as connection:
            return [self._decode(row_dict(row)) for row in connection.execute(statement)]

    def complete_simulation_order_plan_materialization(
        self,
        job_id: str,
        *,
        order_plan_manifest_sha256: str,
        simulation_batch_id: str,
        batch_created: bool,
    ) -> bool:
        """Atomically settle one deferred plan after its idempotent batch exists."""

        with self.engine.begin() as connection:
            row = connection.execute(
                select(jobs)
                .where(jobs.c.id == job_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(job_id)
            if str(row.kind) != "simulation_order_plan" or str(row.status) != "succeeded":
                raise ValueError(
                    "deferred order-plan materialization requires a successful plan job"
                )
            progress = dict(row.progress_json or {})
            if (
                str(progress.get("order_plan_manifest_sha256") or "")
                != order_plan_manifest_sha256
            ):
                raise ValueError(
                    "deferred order-plan materialization changed its immutable manifest"
                )
            current_status = str(
                progress.get(ORDER_PLAN_MATERIALIZATION_STATUS_KEY) or ""
            )
            if current_status == ORDER_PLAN_MATERIALIZED:
                if str(progress.get("simulation_batch_id") or "") != simulation_batch_id:
                    raise ValueError(
                        "deferred order-plan materialization is bound to another batch"
                    )
                return False
            if current_status != ORDER_PLAN_AWAITING_EXECUTION_DATA:
                raise ValueError("order-plan job is not awaiting execution data")
            progress.update(
                {
                    ORDER_PLAN_MATERIALIZATION_STATUS_KEY: ORDER_PLAN_MATERIALIZED,
                    "simulation_batch_id": simulation_batch_id,
                    "simulation_batch_created": bool(batch_created),
                }
            )
            connection.execute(
                update(jobs)
                .where(jobs.c.id == job_id)
                .values(progress_json=normalize_jsonb_document(progress))
            )
        return True

    def terminate_simulation_order_plan_materialization(
        self,
        job_id: str,
        *,
        order_plan_manifest_sha256: str,
        materialization_status: str,
        reason: str,
    ) -> bool:
        """Persist a domain terminal without leaving a poison awaiting row."""

        if materialization_status not in {
            ORDER_PLAN_CANCELLED,
            ORDER_PLAN_SUPERSEDED,
            ORDER_PLAN_FAILED,
        }:
            raise ValueError("order-plan materialization terminal status is invalid")
        normalized_reason = str(reason or "")[:2_000]
        with self.engine.begin() as connection:
            row = connection.execute(
                select(jobs).where(jobs.c.id == job_id).with_for_update()
            ).first()
            if row is None:
                raise KeyError(job_id)
            progress = dict(row.progress_json or {})
            if (
                str(row.kind) != "simulation_order_plan"
                or str(progress.get("order_plan_manifest_sha256") or "")
                != order_plan_manifest_sha256
            ):
                raise ValueError(
                    "order-plan materialization terminal changed its immutable identity"
                )
            current = str(progress.get(ORDER_PLAN_MATERIALIZATION_STATUS_KEY) or "")
            if current == materialization_status:
                return False
            if current != ORDER_PLAN_AWAITING_EXECUTION_DATA:
                raise ValueError("order-plan materialization is no longer awaiting")
            progress.update(
                {
                    ORDER_PLAN_MATERIALIZATION_STATUS_KEY: materialization_status,
                    "simulation_batch_materialization_reason": normalized_reason,
                }
            )
            job_status = (
                materialization_status
                if materialization_status in {ORDER_PLAN_CANCELLED, ORDER_PLAN_FAILED}
                else "succeeded"
            )
            connection.execute(
                update(jobs)
                .where(jobs.c.id == job_id)
                .values(
                    status=job_status,
                    progress_json=normalize_jsonb_document(progress),
                    error=(
                        normalized_reason
                        if materialization_status == ORDER_PLAN_FAILED
                        else None
                    ),
                )
            )
        return True

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
        persisted_result = _with_evaluation_status_counts(normalize_jsonb_document(result))
        with self.engine.begin() as connection:
            row = connection.execute(
                select(
                    jobs.c.kind,
                    jobs.c.status,
                    jobs.c.attempts,
                    jobs.c.max_attempts,
                )
                .where(jobs.c.id == job_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(job_id)
            if (
                retryable
                and str(row.kind) != "strategy_backtest"
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
                        progress_json=persisted_result,
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
                    progress_json=persisted_result,
                    cancel_requested_at=None,
                    next_attempt_at=None,
                    finished_at=_now(),
                )
            )
        return False

    def retry(self, job_id: str) -> dict[str, Any]:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(jobs).where(jobs.c.id == job_id).with_for_update()
            ).first()
            if row is None:
                raise KeyError(job_id)
            if str(row.kind) == "strategy_backtest":
                raise ValueError("formal final-test jobs cannot be retried")
            progress = dict(row.progress_json or {})
            if str(row.kind) == "simulation_order_plan":
                materialization_status = str(
                    progress.get(ORDER_PLAN_MATERIALIZATION_STATUS_KEY) or ""
                )
                if materialization_status == ORDER_PLAN_SUPERSEDED:
                    raise ValueError("superseded simulation order plans cannot be retried")
                raw_manifest_sha256 = progress.get("order_plan_manifest_sha256")
                if raw_manifest_sha256 is not None:
                    manifest_sha256 = str(raw_manifest_sha256).strip().lower()
                    if len(manifest_sha256) != 64 or any(
                        character not in "0123456789abcdef"
                        for character in manifest_sha256
                    ):
                        raise ValueError(
                            "sealed simulation order-plan manifest identity is invalid"
                        )
                    if materialization_status not in {
                        ORDER_PLAN_FAILED,
                        ORDER_PLAN_CANCELLED,
                        ORDER_PLAN_AWAITING_EXECUTION_DATA,
                    }:
                        raise ValueError(
                            "sealed simulation order plan has no recoverable "
                            "materialization terminal"
                        )
                    try:
                        date.fromisoformat(
                            str(progress[ORDER_PLAN_EXECUTION_TRADE_DATE_KEY])
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            "sealed simulation order plan has no recoverable execution date"
                        ) from exc
                    if row.status not in {"failed", "cancelled"}:
                        raise ValueError(
                            "only failed or cancelled jobs may be retried"
                        )
                    # The governed signal artifact already exists.  Requeueing
                    # the subprocess would recompute an old D signal against a
                    # newer publication and can never be a frozen recovery.
                    # Preserve every result/evidence field and hand the exact
                    # manifest back to Scheduler's D+1 materializer instead.
                    progress[ORDER_PLAN_MATERIALIZATION_STATUS_KEY] = (
                        ORDER_PLAN_AWAITING_EXECUTION_DATA
                    )
                    connection.execute(
                        update(jobs)
                        .where(jobs.c.id == job_id)
                        .values(
                            status="succeeded",
                            progress_json=normalize_jsonb_document(progress),
                            exit_code=0,
                            error=None,
                            cancel_requested_at=None,
                            next_attempt_at=None,
                        )
                    )
                    return self._decode(
                        row_dict(
                            connection.execute(
                                select(jobs).where(jobs.c.id == job_id)
                            ).one()
                        )
                    )
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
        payload_keys: tuple[str, ...] | None = None,
        progress_keys: tuple[str, ...] | None = None,
        progress_evaluation_statuses: tuple[str, ...] = (),
        progress_evaluation_kind: str | None = None,
    ) -> list[dict[str, Any]]:
        statement = select(
            *self._projected_columns(
                payload_keys=payload_keys,
                progress_keys=progress_keys,
                progress_evaluation_statuses=progress_evaluation_statuses,
                progress_evaluation_kind=progress_evaluation_kind,
            )
        )
        if statuses:
            statement = statement.where(jobs.c.status.in_(statuses))
        if kinds:
            statement = statement.where(jobs.c.kind.in_(kinds))
        statement = statement.order_by(jobs.c.created_at.desc()).offset(offset).limit(limit)
        with self.engine.connect() as connection:
            rows = [self._decode(row_dict(row)) for row in connection.execute(statement)]
            return self._annotate_retry_successors(connection, rows)

    def status_summaries(
        self,
        *,
        statuses: tuple[str, ...],
        limit: int = 1_000,
    ) -> list[dict[str, Any]]:
        """Return the bounded fields needed by operational counters.

        A completed research job may keep megabytes of immutable evidence in
        ``progress_json``.  Operational badges need only kind and status; do
        not deserialize that evidence on every overview poll.
        """

        statement = (
            select(jobs.c.id, jobs.c.kind, jobs.c.status)
            .where(jobs.c.status.in_(statuses))
            .order_by(jobs.c.created_at.desc())
            .limit(limit)
        )
        with self.engine.connect() as connection:
            return [row_dict(row) for row in connection.execute(statement)]

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

    def get(
        self,
        job_id: str,
        *,
        payload_keys: tuple[str, ...] | None = None,
        progress_keys: tuple[str, ...] | None = None,
        progress_evaluation_statuses: tuple[str, ...] = (),
        progress_evaluation_kind: str | None = None,
    ) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    *self._projected_columns(
                        payload_keys=payload_keys,
                        progress_keys=progress_keys,
                        progress_evaluation_statuses=progress_evaluation_statuses,
                        progress_evaluation_kind=progress_evaluation_kind,
                    )
                ).where(jobs.c.id == job_id)
            ).first()
            if row is None:
                raise KeyError(job_id)
            decoded = self._decode(row_dict(row))
            return self._annotate_retry_successors(connection, [decoded])[0]

    @staticmethod
    def _evaluation_status_count(column, status: str, *, job_kind: str | None = None):
        persisted = column[EVALUATION_STATUS_COUNTS_KEY][status].as_integer()
        json_path = cast(
            literal(f'$.evaluations[*] ? (@.status == "{status}")'),
            JSONPATH,
        )
        historical = func.jsonb_array_length(func.jsonb_path_query_array(column, json_path))
        count = func.coalesce(persisted, historical, 0)
        if job_kind is not None:
            return case((jobs.c.kind == job_kind, count), else_=0)
        return count

    @classmethod
    def _jsonb_projection(
        cls,
        column,
        keys: tuple[str, ...],
        label: str,
        *,
        evaluation_statuses: tuple[str, ...] = (),
        evaluation_kind: str | None = None,
    ):
        arguments: list[Any] = []
        for key in keys:
            arguments.extend((key, column[key]))
        if evaluation_statuses:
            count_arguments: list[Any] = []
            for status in evaluation_statuses:
                count_arguments.extend(
                    (
                        status,
                        cls._evaluation_status_count(
                            column,
                            status,
                            job_kind=evaluation_kind,
                        ),
                    )
                )
            arguments.extend(
                (
                    EVALUATION_STATUS_COUNTS_KEY,
                    func.jsonb_build_object(*count_arguments),
                )
            )
        return func.jsonb_strip_nulls(func.jsonb_build_object(*arguments)).label(label)

    @classmethod
    def _projected_columns(
        cls,
        *,
        payload_keys: tuple[str, ...] | None,
        progress_keys: tuple[str, ...] | None,
        progress_evaluation_statuses: tuple[str, ...] = (),
        progress_evaluation_kind: str | None = None,
    ) -> list[Any]:
        projected = [
            column
            for column in jobs.c
            if not (
                (column.name == "payload_json" and payload_keys is not None)
                or (column.name == "progress_json" and progress_keys is not None)
            )
        ]
        if payload_keys is not None:
            projected.append(
                cls._jsonb_projection(jobs.c.payload_json, payload_keys, "payload_json")
            )
        if progress_keys is not None:
            projected.append(
                cls._jsonb_projection(
                    jobs.c.progress_json,
                    progress_keys,
                    "progress_json",
                    evaluation_statuses=progress_evaluation_statuses,
                    evaluation_kind=progress_evaluation_kind,
                )
            )
        return projected

    @staticmethod
    def _lineage_identity(job: dict[str, Any]) -> tuple[str, str, str] | None:
        payload = job.get("payload") or {}
        pipeline_id = payload.get("pipeline_id")
        if isinstance(pipeline_id, str) and pipeline_id:
            return str(job["kind"]), "pipeline", pipeline_id
        snapshot_name = payload.get("snapshot_name")
        if isinstance(snapshot_name, str) and snapshot_name:
            return str(job["kind"]), "snapshot", snapshot_name
        return None

    @classmethod
    def _annotate_retry_successors(
        cls,
        connection,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        failed = [
            row
            for row in rows
            if row.get("status") == "failed" and cls._lineage_identity(row) is not None
        ]
        if not failed:
            return rows
        # row_dict intentionally serializes datetimes for API responses and
        # truncates them to seconds. Retry attempts can be created within the
        # same second, so ordering must use the database timestamps rather
        # than those serialized display values.
        failed_created_at = {
            str(row.id): row.created_at
            for row in connection.execute(
                select(jobs.c.id, jobs.c.created_at).where(
                    jobs.c.id.in_([str(item["id"]) for item in failed])
                )
            )
        }
        earliest = min(failed_created_at.values())
        failed_kinds = sorted({str(row["kind"]) for row in failed})
        successor_payload = cls._jsonb_projection(
            jobs.c.payload_json,
            ("pipeline_id", "snapshot_name"),
            "payload_json",
        )
        candidates = []
        for raw in connection.execute(
            select(
                jobs.c.id,
                jobs.c.kind,
                jobs.c.status,
                jobs.c.created_at,
                successor_payload,
            )
            .where(
                jobs.c.created_at > earliest,
                jobs.c.kind.in_(failed_kinds),
                jobs.c.status.in_(("queued", "running", "succeeded")),
            )
            .order_by(jobs.c.created_at.desc())
        ):
            candidate = row_dict(raw)
            candidate["payload"] = candidate.pop("payload_json") or {}
            candidates.append((raw.created_at, candidate))
        for row in failed:
            identity = cls._lineage_identity(row)
            successor = next(
                (
                    candidate
                    for candidate_created_at, candidate in candidates
                    if candidate_created_at > failed_created_at[str(row["id"])]
                    and cls._lineage_identity(candidate) == identity
                ),
                None,
            )
            if successor is not None:
                row["retry_successor"] = {
                    "id": successor["id"],
                    "status": successor["status"],
                }
        return rows

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        row["payload"] = row.pop("payload_json")
        row["progress"] = row.pop("progress_json")
        return row
