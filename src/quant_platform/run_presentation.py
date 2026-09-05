"""Read-only, path-free presentation of persisted execution evidence.

These labels never change a run's audit status or decide an economic outcome.
Unstructured diagnostics are only inputs to fixed messages, never public text.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import case, func, or_, select, text, true
from sqlalchemy.engine import Engine

from quant_data.database import jobs, research_runs, row_dict

from .research_store import ResearchStore
from .run_progress import COUNTERS, attach_parameter_progress, unavailable_progress

ACTIVE_RESEARCH_STATUSES = ("queued", "running", "evaluating", "exporting")
PRESENTATION_PAYLOAD_KEYS = frozenset(
    {"strategy_competition_stage", "fin_strategy_research_run_id"}
)
_STATUS_LABELS = {
    "queued": "等待执行",
    "running": "运行中",
    "succeeded": "已完成",
    "failed": "执行失败",
    "interrupted": "已中止",
    "stopping": "正在中止",
    "blocked": "未通过",
    "unknown": "状态待确认",
}
_PHASE_LABELS = {
    "proposal": "研究提案",
    "evaluation": "独立评估",
    "parameter_search": "参数回测与验证",
    "policy_only": "策略规则竞赛（回测与验证）",
    "full_stack": "完整策略竞赛（回测与验证）",
    "formal_final_oos": "正式样本外验证",
    "settlement": "研究结果结算",
    "unknown": "",
}
_REASONS = {
    "operator_cancelled": "本次执行已按中止请求结束，历史记录保留。",
    "cancel_requested": "已收到中止请求，正在等待执行任务退出。",
    "worker_interrupted": "执行服务曾中断，本次任务未完成。",
    "resource_exhausted": "执行时计算资源不足，本次任务未完成。",
    "identity_conflict": "任务身份与已有记录冲突，未创建重复执行。",
    "data_validation_failed": "输入数据或回测窗口未通过校验，本次执行未完成。",
    "capacity_validation_failed": "交易容量约束校验未通过，本次执行未完成。",
    "runtime_validation_failed": "运行环境或代码身份未通过校验，本次执行未完成。",
    "execution_failed": "本次执行未完成；详细诊断保留在受限审计记录中。",
    "gate_not_passed": "本次研究未通过准入检查，具体结论以审计记录为准。",
}


def _failure_code(error: Any) -> str:
    # Match known diagnostics without forwarding any user/provider-controlled text.
    value = str(error or "").lower()
    if "cancelled by operator" in value or "cancelled before execution" in value:
        return "operator_cancelled"
    if "worker restarted" in value:
        return "worker_interrupted"
    if any(part in value for part in ("out of memory", "oom_kill", "cannot allocate memory")):
        return "resource_exhausted"
    if "idempotency key is already bound to a different job payload" in value:
        return "identity_conflict"
    if re.search(
        r"\bpost-discretization hard constraint violation:\s*capacity_trade_value\b", value
    ):
        return "capacity_validation_failed"
    if any(
        part in value
        for part in (
            "style exposure missing rate exceeds",
            "point-in-time styles have no valid",
            "style metadata is missing",
            "style panel is missing required columns",
            "validation window is shorter than its preregistered oos",
            "governed signal has no eligible trading dates",
            "fin_strategy dataset or research periods differ from the plan",
            "qlib baseline requires dataset provenance metadata",
            "dataset identity mismatch",
            "dataset identity does not match",
        )
    ) or re.search(r"point-in-time[^\r\n]{0,160}missing rate exceeds\b", value):
        return "data_validation_failed"
    if any(
        part in value
        for part in (
            "runtime identity mismatch", "source closure mismatch",
            "source closure sha256 mismatch",
            "rdagent runtime identity changed after enqueue",
        )
    ) or re.search(
        r"transparent v[0-9]+ runner bytes differ from the repair authorization\b", value
    ):
        return "runtime_validation_failed"
    return "execution_failed"


def execution_phase(job: dict[str, Any]) -> str:
    """Expose the registered stage, not inferred trial/IS/OOS log progress."""

    payload = job.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    stage = payload.get("strategy_competition_stage")
    kind = job.get("kind")
    if kind == "parameter_experiment":
        return stage if isinstance(stage, str) and stage in {
            "policy_only", "full_stack"
        } else "parameter_search"
    if kind == "strategy_backtest" and payload.get("fin_strategy_research_run_id"):
        return "formal_final_oos"
    if kind in {"factor_evaluate", "model_evaluate", "model_ensemble_evaluate"}:
        return "evaluation"
    if kind in {
        "rdagent_run", "rdagent_factor", "rdagent_quant", "rdagent_model",
        "rdagent_factor_report", "rdagent_data_science", "rdagent_llm_finetune",
    }:
        return "proposal"
    return "unknown"


def _public_progress(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("state") != "available":
        return unavailable_progress()
    counts = {key: value.get(key) for key in COUNTERS}
    if (
        any(type(count) is not int or count < 0 for count in counts.values())
        or not 0 < counts["trial_count"] <= 10000
        or counts["completed_count"] > counts["trial_count"]
        or counts["completed_count"] != counts["succeeded_count"] + counts["failed_count"]
    ):
        return unavailable_progress()
    try:
        updated = datetime.fromisoformat(value["updated_at"])
        if updated.tzinfo is None:
            return unavailable_progress()
    except (KeyError, TypeError, ValueError):
        return unavailable_progress()
    return {"state": "available", **counts, "updated_at": updated.isoformat()}


def run_presentation(
    record: dict[str, Any], *, linked_job: dict[str, Any] | None = None, research: bool = False
) -> dict[str, Any]:
    """Keep persisted status authoritative while explaining current execution."""

    audit_status = str(record.get("status") or "")
    active = audit_status in ACTIVE_RESEARCH_STATUSES
    current = linked_job if research and linked_job else record
    phase = execution_phase(current)
    if research and phase == "unknown" and audit_status == "evaluating":
        phase = "evaluation"
    if research and active and current.get("status") == "succeeded":
        # A successful proposal/parameter subprocess is not completed research.
        phase = "settlement"
    status = "running" if audit_status in {"evaluating", "exporting"} else audit_status
    reason_code: str | None = None
    if active:
        if record.get("cancel_requested_at") or current.get("cancel_requested_at"):
            status, reason_code = "stopping", "cancel_requested"
        elif research and linked_job and linked_job.get("status") != "succeeded":
            linked_presentation = run_presentation(linked_job)
            status = linked_presentation["display_status"]
            reason_code = linked_presentation["reason_code"]
    elif audit_status == "cancelled":
        status, reason_code = "interrupted", "operator_cancelled"
    elif audit_status == "failed":
        reason_code = _failure_code(record.get("error"))
        # A linked cancelled owner explains legacy research.status=failed.
        if research and linked_job and linked_job.get("status") == "cancelled":
            reason_code = "operator_cancelled"
        elif research and linked_job and reason_code == "execution_failed":
            reason_code = _failure_code(linked_job.get("error"))
        if reason_code == "operator_cancelled":
            status = "interrupted"
    elif audit_status == "blocked":
        reason_code = "gate_not_passed"
    if status not in _STATUS_LABELS:
        status = "unknown"
    progress = _public_progress(current.get("_parameter_progress"))
    safe_reason = _REASONS.get(reason_code)
    label = _STATUS_LABELS[status]
    if (
        status == "running" and isinstance(progress, dict)
        and progress.get("state") == "available" and progress.get("failed_count", 0) > 0
    ):
        reason_code = "partial_trial_failure"
        label = "运行中（已有试验失败）"
        safe_reason = (
            f"仍在执行；已完成 {progress['completed_count']}/{progress['trial_count']} 个试验，"
            f"其中 {progress['failed_count']} 个失败。"
        )
    return {
        "display_status": status,
        "label": label,
        "reason_code": reason_code,
        "safe_reason": safe_reason,
        "execution_phase": phase,
        "phase_label": _PHASE_LABELS[phase],
        "progress": progress,
    }


def public_linked_execution(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if job is None:
        return None
    return {
        "job_id": job["id"],
        "kind": job["kind"],
        "status": job["status"],
        "attempts": job.get("attempts"),
        "max_attempts": job.get("max_attempts"),
        "parameter_experiment_id": (
            job["parameter_experiment_id"]
            if isinstance(job.get("parameter_experiment_id"), str)
            and re.fullmatch(r"[0-9a-f]{32}", job["parameter_experiment_id"])
            else None
        ),
        "presentation": run_presentation(job),
        # Jobs have no updated_at; do not invent a heartbeat/progress timestamp.
        "updated_at": next(
            (job[key] for key in ("finished_at", "started_at", "created_at") if job.get(key)),
            None,
        ),
    }


def research_page_statements(
    *, limit: int, offset: int, status_group: Literal["active", "history"] | None
) -> tuple[Any, Any]:
    condition = true()
    if status_group == "active":
        condition = research_runs.c.status.in_(ACTIVE_RESEARCH_STATUSES)
    elif status_group == "history":
        condition = research_runs.c.status.not_in(ACTIVE_RESEARCH_STATUSES)
    order_time = (
        func.coalesce(research_runs.c.finished_at, research_runs.c.created_at)
        if status_group == "history" else research_runs.c.created_at
    )
    return (
        select(func.count()).select_from(research_runs).where(condition),
        select(research_runs)
        .where(condition)
        .order_by(order_time.desc(), research_runs.c.id.desc())
        .limit(limit)
        .offset(offset),
    )


def linked_execution_statement(run_ids: list[str]) -> Any:
    """One bounded lateral lookup per displayed run, including successor jobs.

    Only exact stored owner IDs establish linkage. A newer active successor
    takes precedence over historical terminal jobs. No objective/name matching,
    artifact traversal, full payload, trial metrics or returns are needed.
    """

    payload = jobs.c.payload_json
    current = (
        select(
            jobs.c.id,
            jobs.c.kind,
            jobs.c.status,
            jobs.c.attempts,
            jobs.c.max_attempts,
            jobs.c.cancel_requested_at,
            jobs.c.created_at,
            jobs.c.started_at,
            jobs.c.finished_at,
            func.left(jobs.c.error, 4096).label("error"),
            payload["parameter_experiment_id"].as_string().label("parameter_experiment_id"),
            payload["strategy_competition_stage"].as_string().label("strategy_competition_stage"),
            payload["fin_strategy_research_run_id"].as_string().label("fin_strategy_research_run_id"),
        )
        .where(
            or_(
                jobs.c.id == research_runs.c.job_id,
                payload["research_run_id"].as_string() == research_runs.c.id,
                payload["fin_strategy_research_run_id"].as_string() == research_runs.c.id,
            )
        )
        .order_by(
            case((jobs.c.status.in_(("queued", "running")), 0), else_=1),
            jobs.c.created_at.desc(),
            jobs.c.id.desc(),
        )
        .limit(1)
        .correlate(research_runs)
        .lateral("linked_execution")
    )
    return (
        select(research_runs.c.id.label("research_run_id"), *current.c)
        .select_from(research_runs.join(current, true()))
        .where(research_runs.c.id.in_(run_ids))
    )


class RunPresentationStore:
    """Read-only API projections; never repair or settle persisted runs."""

    def __init__(self, engine: Engine, data_root: Path | None = None):
        self.engine = engine
        self.data_root = data_root

    @staticmethod
    def _read_only(connection: Any) -> None:
        connection.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        connection.execute(text("SET LOCAL statement_timeout = '5s'"))

    def list_runs(
        self, *, limit: int, offset: int, status_group: Literal["active", "history"] | None
    ) -> tuple[list[dict[str, Any]], int]:
        count, page = research_page_statements(
            limit=limit, offset=offset, status_group=status_group
        )
        with self.engine.connect() as connection:
            self._read_only(connection)
            total = int(connection.execute(count).scalar_one())
            rows = [ResearchStore._decode_run(row_dict(row)) for row in connection.execute(page)]
        return rows, total

    def linked_executions(self, runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if not runs:
            return {}
        with self.engine.connect() as connection:
            self._read_only(connection)
            rows = connection.execute(linked_execution_statement([run["id"] for run in runs]))
            result = {}
            for row in rows:
                job = dict(row._mapping)
                for key, value in job.items():
                    if isinstance(value, datetime):
                        job[key] = value.isoformat()
                job["payload"] = {key: job.pop(key) for key in PRESENTATION_PAYLOAD_KEYS}
                result[job.pop("research_run_id")] = job
            if self.data_root is not None:
                annotated = attach_parameter_progress(
                    connection, list(result.values()), self.data_root
                )
                by_id = {job["id"]: job for job in annotated}
                result = {run_id: by_id[job["id"]] for run_id, job in result.items()}
        return result

    def with_progress(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.data_root is None or not any(
            row.get("kind") == "parameter_experiment" and row.get("status") == "running"
            for row in records
        ):
            return records
        with self.engine.connect() as connection:
            self._read_only(connection)
            return attach_parameter_progress(connection, records, self.data_root)
