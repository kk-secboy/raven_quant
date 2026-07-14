from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import and_, insert, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    factor_candidates,
    factor_evaluations,
    open_database,
    research_events,
    research_runs,
    row_dict,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hash(path_value: str | None, label: str, *, required: bool) -> str | None:
    if not path_value:
        if required:
            raise ValueError(f"{label} artifact path is required")
        return None
    path = Path(path_value)
    if not path.is_file():
        if required:
            raise ValueError(f"{label} artifact is missing: {path}")
        return None
    return _sha256_file(path)


def _evaluation_artifact_metrics(path_value: str, candidate_id: str) -> dict[str, Any]:
    path = Path(path_value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Qlib evaluation artifact is unreadable") from exc
    evaluations = payload.get("evaluations") if isinstance(payload, dict) else None
    if not isinstance(evaluations, list):
        raise ValueError("Qlib evaluation artifact has no evaluations list")
    matches = [
        item
        for item in evaluations
        if isinstance(item, dict) and str(item.get("candidate_id")) == candidate_id
    ]
    if len(matches) != 1 or matches[0].get("status") != "ok":
        raise ValueError("Qlib evaluation artifact has no unique successful candidate result")
    metrics = matches[0].get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Qlib evaluation artifact candidate metrics are missing")
    return metrics


def _evaluation_evidence(
    *,
    candidate_id: str,
    dataset: str,
    periods: dict[str, str],
    gate_status: str,
    gate_reasons: list[str],
    evaluator_version: str,
    candidate_code_sha256: str,
    candidate_values_sha256: str,
    artifact_sha256: str,
    metrics_sha256: str,
    policy_sha256: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "dataset": dataset,
        "periods": periods,
        "gate_status": gate_status,
        "gate_reasons": gate_reasons,
        "evaluator_version": evaluator_version,
        "candidate_code_sha256": candidate_code_sha256,
        "candidate_values_sha256": candidate_values_sha256,
        "artifact_sha256": artifact_sha256,
        "metrics_sha256": metrics_sha256,
        "policy_sha256": policy_sha256,
    }


@dataclass(frozen=True, slots=True)
class FactorGatePolicy:
    """Versioned, deterministic admission policy for production factor candidates."""

    version: str = "factor-gate-v1"
    min_abs_ic: float = 0.02
    min_abs_icir: float = 0.50
    min_abs_rank_ic: float = 0.025
    min_abs_rank_icir: float = 0.50
    max_turnover: float = 0.60
    max_correlation: float = 0.75
    min_cost_adjusted_return: float = 0.0
    min_test_days: int = 252

    def evaluate(self, metrics: dict[str, float | None]) -> tuple[str, list[str]]:
        checks = (
            ("ic", lambda value: abs(value) >= self.min_abs_ic, f"|IC| >= {self.min_abs_ic}"),
            (
                "icir",
                lambda value: abs(value) >= self.min_abs_icir,
                f"|ICIR| >= {self.min_abs_icir}",
            ),
            (
                "rank_ic",
                lambda value: abs(value) >= self.min_abs_rank_ic,
                f"|RankIC| >= {self.min_abs_rank_ic}",
            ),
            (
                "rank_icir",
                lambda value: abs(value) >= self.min_abs_rank_icir,
                f"|RankICIR| >= {self.min_abs_rank_icir}",
            ),
            (
                "turnover",
                lambda value: value <= self.max_turnover,
                f"turnover <= {self.max_turnover}",
            ),
            (
                "max_correlation",
                lambda value: abs(value) <= self.max_correlation,
                f"|correlation| <= {self.max_correlation}",
            ),
            (
                "cost_adjusted_return",
                lambda value: value > self.min_cost_adjusted_return,
                f"cost-adjusted return > {self.min_cost_adjusted_return}",
            ),
            (
                "test_days",
                lambda value: value >= self.min_test_days,
                f"test days >= {self.min_test_days}",
            ),
        )
        reasons: list[str] = []
        for name, predicate, expectation in checks:
            value = metrics.get(name)
            if value is None:
                reasons.append(f"{name} is missing; expected {expectation}")
            elif not predicate(float(value)):
                reasons.append(f"{name}={value:g} failed; expected {expectation}")
        valid_ic = metrics.get("valid_ic")
        ic = metrics.get("ic")
        rank_ic = metrics.get("rank_ic")
        if valid_ic is None:
            reasons.append("valid_ic is missing; validation direction cannot be verified")
        elif ic is not None and float(valid_ic) * float(ic) <= 0:
            reasons.append("validation IC and out-of-sample IC must have the same direction")
        if ic is not None and rank_ic is not None and float(ic) * float(rank_ic) <= 0:
            reasons.append("IC and RankIC must have the same direction")
        return ("passed" if not reasons else "failed", reasons)


class ResearchStore:
    """PostgreSQL repository for RD-Agent runs and governed factor promotion."""

    def __init__(self, database_url: str, policy: FactorGatePolicy | None = None) -> None:
        self.engine = open_database(database_url)
        self.policy = policy or FactorGatePolicy()

    def create_run(
        self,
        *,
        kind: str,
        objective: str,
        dataset: str,
        requested_by: str,
        budget: dict[str, Any],
        config: dict[str, Any],
        artifact_path: Path,
    ) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(research_runs).values(
                        id=run_id,
                        kind=kind,
                        objective=objective,
                        dataset=dataset,
                        status="queued",
                        requested_by=requested_by,
                        budget_json=budget,
                        config_json=config,
                        artifact_path=str(artifact_path),
                        created_at=now,
                        updated_at=now,
                    )
                )
                self._event(
                    connection,
                    run_id=run_id,
                    event_type="run.created",
                    actor=requested_by,
                    payload={"kind": kind, "dataset": dataset, "budget": budget},
                )
        except IntegrityError as exc:
            raise ValueError(f"an active {kind} research run already exists") from exc
        return self.get_run(run_id)

    def attach_job(self, run_id: str, job_id: str) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(research_runs)
                .where(research_runs.c.id == run_id)
                .values(job_id=job_id, updated_at=_now())
            )
            if not result.rowcount:
                raise KeyError(run_id)

    def mark_run(
        self,
        run_id: str,
        status: str,
        *,
        actor: str = "worker",
        runtime: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        now = _now()
        values: dict[str, Any] = {"status": status, "updated_at": now, "error": error}
        if runtime is not None:
            values["runtime_json"] = runtime
        if status == "running":
            values["started_at"] = now
        if status in {"succeeded", "failed", "cancelled"}:
            values["finished_at"] = now
        with self.engine.begin() as connection:
            result = connection.execute(
                update(research_runs).where(research_runs.c.id == run_id).values(**values)
            )
            if not result.rowcount:
                raise KeyError(run_id)
            self._event(
                connection,
                run_id=run_id,
                event_type=f"run.{status}",
                actor=actor,
                payload={"error": error} if error else {},
            )

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(research_runs).where(research_runs.c.id == run_id)
            ).first()
        if row is None:
            raise KeyError(run_id)
        return self._decode_run(row_dict(row))

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        statement = select(research_runs).order_by(research_runs.c.created_at.desc()).limit(limit)
        with self.engine.connect() as connection:
            return [self._decode_run(row_dict(row)) for row in connection.execute(statement)]

    def add_candidate(
        self,
        run_id: str,
        *,
        name: str,
        description: str,
        formulation: str | None,
        variables: dict[str, Any],
        source_iteration: int | None,
        code_path: str | None,
        values_path: str | None,
        code_sha256: str | None,
        rdagent_decision: bool | None,
        rdagent_feedback: str | None,
        actor: str = "rdagent-importer",
    ) -> dict[str, Any]:
        candidate_id = uuid.uuid4().hex
        now = _now()
        status = "awaiting_evaluation" if rdagent_decision is not False else "rejected_by_rdagent"
        requires_artifacts = rdagent_decision is not False
        actual_code_sha256 = _artifact_hash(
            code_path, "factor code", required=requires_artifacts
        )
        values_sha256 = _artifact_hash(
            values_path, "factor values", required=requires_artifacts
        )
        if code_sha256 and actual_code_sha256 and code_sha256 != actual_code_sha256:
            raise ValueError("factor code artifact does not match RD-Agent SHA-256")
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(factor_candidates).values(
                        id=candidate_id,
                        research_run_id=run_id,
                        name=name,
                        description=description,
                        formulation=formulation,
                        variables_json=variables,
                        status=status,
                        source_iteration=source_iteration,
                        code_path=code_path,
                        values_path=values_path,
                        code_sha256=actual_code_sha256,
                        values_sha256=values_sha256,
                        rdagent_decision=rdagent_decision,
                        rdagent_feedback=rdagent_feedback,
                        created_at=now,
                        updated_at=now,
                    )
                )
                self._event(
                    connection,
                    run_id=run_id,
                    candidate_id=candidate_id,
                    event_type="candidate.imported",
                    actor=actor,
                    payload={
                        "name": name,
                        "status": status,
                        "code_sha256": actual_code_sha256,
                        "values_sha256": values_sha256,
                    },
                )
        except IntegrityError as exc:
            raise ValueError(f"candidate {name!r} already exists in research run") from exc
        return self.get_candidate(candidate_id)

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(factor_candidates).where(factor_candidates.c.id == candidate_id)
            ).first()
        if row is None:
            raise KeyError(candidate_id)
        candidate = self._decode_candidate(row_dict(row))
        candidate["latest_evaluation"] = self.latest_evaluation(candidate_id)
        return candidate

    def list_candidates(
        self, *, run_id: str | None = None, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        statement = select(factor_candidates)
        predicates = []
        if run_id:
            predicates.append(factor_candidates.c.research_run_id == run_id)
        if status:
            predicates.append(factor_candidates.c.status == status)
        if predicates:
            statement = statement.where(and_(*predicates))
        statement = statement.order_by(factor_candidates.c.updated_at.desc()).limit(limit)
        with self.engine.connect() as connection:
            rows = [self._decode_candidate(row_dict(row)) for row in connection.execute(statement)]
        for candidate in rows:
            candidate["latest_evaluation"] = self.latest_evaluation(candidate["id"])
        return rows

    def record_evaluation(
        self,
        candidate_id: str,
        *,
        dataset: str,
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
        test_start: date,
        test_end: date,
        metrics: dict[str, float | None],
        artifact_path: str | None,
        actor: str = "qlib-evaluator",
    ) -> dict[str, Any]:
        if not (train_start <= train_end < valid_start <= valid_end < test_start <= test_end):
            raise ValueError(
                "train, validation, and test windows must be ordered and non-overlapping"
            )
        candidate = self.get_candidate(candidate_id)
        if candidate["status"] in {"promoted", "retired"}:
            raise ValueError(f"cannot evaluate candidate in {candidate['status']} state")
        gate_status, reasons = self.policy.evaluate(metrics)
        current_code_sha256 = _artifact_hash(
            candidate.get("code_path"), "factor code", required=True
        )
        current_values_sha256 = _artifact_hash(
            candidate.get("values_path"), "factor values", required=True
        )
        if current_code_sha256 != candidate.get("code_sha256"):
            raise ValueError("factor code artifact changed after RD-Agent import")
        if current_values_sha256 != candidate.get("values_sha256"):
            raise ValueError("factor values artifact changed after RD-Agent import")
        if gate_status == "passed" and not artifact_path:
            raise ValueError("passed Qlib evaluation requires a durable result artifact")
        artifact_sha256 = _artifact_hash(
            artifact_path, "Qlib evaluation", required=gate_status == "passed"
        )
        if artifact_path:
            artifact_metrics = _evaluation_artifact_metrics(artifact_path, candidate_id)
            if _canonical_sha256(artifact_metrics) != _canonical_sha256(metrics):
                raise ValueError("Qlib evaluation artifact metrics do not match imported metrics")
        metrics_sha256 = _canonical_sha256(metrics)
        policy = asdict(self.policy)
        policy_sha256 = _canonical_sha256(policy)
        periods = {
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "valid_start": valid_start.isoformat(),
            "valid_end": valid_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
        }
        evidence = _evaluation_evidence(
            candidate_id=candidate_id,
            dataset=dataset,
            periods=periods,
            gate_status=gate_status,
            gate_reasons=reasons,
            evaluator_version=self.policy.version,
            candidate_code_sha256=current_code_sha256,
            candidate_values_sha256=current_values_sha256,
            artifact_sha256=artifact_sha256 or "",
            metrics_sha256=metrics_sha256,
            policy_sha256=policy_sha256,
        )
        evidence_sha256 = _canonical_sha256(evidence)
        evaluation_id = uuid.uuid4().hex
        now = _now()
        scalars = {
            key: metrics.get(key)
            for key in (
                "ic",
                "icir",
                "rank_ic",
                "rank_icir",
                "turnover",
                "max_correlation",
                "cost_adjusted_return",
            )
        }
        candidate_status = "gate_passed" if gate_status == "passed" else "gate_failed"
        with self.engine.begin() as connection:
            connection.execute(
                insert(factor_evaluations).values(
                    id=evaluation_id,
                    factor_candidate_id=candidate_id,
                    dataset=dataset,
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                    test_start=test_start,
                    test_end=test_end,
                    **scalars,
                    metrics_json=metrics,
                    gate_status=gate_status,
                    gate_reasons_json=reasons,
                    evaluator_version=self.policy.version,
                    artifact_path=artifact_path,
                    artifact_sha256=artifact_sha256,
                    candidate_code_sha256=current_code_sha256,
                    candidate_values_sha256=current_values_sha256,
                    metrics_sha256=metrics_sha256,
                    policy_json=policy,
                    policy_sha256=policy_sha256,
                    evidence_sha256=evidence_sha256,
                    created_at=now,
                )
            )
            connection.execute(
                update(factor_candidates)
                .where(factor_candidates.c.id == candidate_id)
                .values(status=candidate_status, updated_at=now)
            )
            self._event(
                connection,
                run_id=candidate["research_run_id"],
                candidate_id=candidate_id,
                event_type=f"candidate.{candidate_status}",
                actor=actor,
                payload={
                    "evaluation_id": evaluation_id,
                    "reasons": reasons,
                    "evidence_sha256": evidence_sha256,
                },
            )
        return self.get_evaluation(evaluation_id)

    def promote(self, candidate_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        actor = actor.strip()
        reason = reason.strip()
        if not actor or not reason:
            raise ValueError("actor and promotion reason are required")
        if len(reason) < 10:
            raise ValueError("promotion reason must contain at least 10 characters")
        candidate = self.get_candidate(candidate_id)
        evaluation = candidate["latest_evaluation"]
        if not evaluation or evaluation["gate_status"] != "passed":
            raise ValueError("candidate must pass the latest Qlib gate before promotion")
        if candidate["status"] != "gate_passed":
            raise ValueError(f"candidate cannot be promoted from {candidate['status']} state")
        current_code_sha256 = _artifact_hash(
            candidate.get("code_path"), "factor code", required=True
        )
        current_values_sha256 = _artifact_hash(
            candidate.get("values_path"), "factor values", required=True
        )
        if current_code_sha256 != candidate.get("code_sha256"):
            raise ValueError("factor code artifact changed after evaluation")
        if current_values_sha256 != candidate.get("values_sha256"):
            raise ValueError("factor values artifact changed after evaluation")
        if evaluation.get("candidate_code_sha256") != current_code_sha256:
            raise ValueError("Qlib evaluation is not bound to the current factor code")
        if evaluation.get("candidate_values_sha256") != current_values_sha256:
            raise ValueError("Qlib evaluation is not bound to the current factor values")
        metrics = evaluation.get("metrics")
        policy = evaluation.get("policy_json")
        if not isinstance(metrics, dict) or evaluation.get("metrics_sha256") != _canonical_sha256(
            metrics
        ):
            raise ValueError("Qlib evaluation metrics provenance is invalid")
        expected_policy = asdict(self.policy)
        if policy != expected_policy or evaluation.get("policy_sha256") != _canonical_sha256(
            expected_policy
        ):
            raise ValueError(
                "Qlib evaluation policy is stale or invalid; re-evaluation is required"
            )
        repeated_status, repeated_reasons = self.policy.evaluate(metrics)
        if repeated_status != "passed" or repeated_reasons != evaluation.get("gate_reasons"):
            raise ValueError("Qlib evaluation no longer passes the recorded factor gate")
        artifact_path = evaluation.get("artifact_path")
        artifact_sha256 = _artifact_hash(
            artifact_path, "Qlib evaluation", required=True
        )
        if artifact_sha256 != evaluation.get("artifact_sha256"):
            raise ValueError("Qlib evaluation artifact changed after evaluation")
        artifact_metrics = _evaluation_artifact_metrics(str(artifact_path), candidate_id)
        if _canonical_sha256(artifact_metrics) != evaluation.get("metrics_sha256"):
            raise ValueError("Qlib evaluation artifact no longer matches recorded metrics")
        evidence = _evaluation_evidence(
            candidate_id=candidate_id,
            dataset=evaluation["dataset"],
            periods={
                key: evaluation[key].isoformat()
                for key in (
                    "train_start",
                    "train_end",
                    "valid_start",
                    "valid_end",
                    "test_start",
                    "test_end",
                )
            },
            gate_status=evaluation["gate_status"],
            gate_reasons=evaluation["gate_reasons"],
            evaluator_version=evaluation["evaluator_version"],
            candidate_code_sha256=current_code_sha256,
            candidate_values_sha256=current_values_sha256,
            artifact_sha256=artifact_sha256,
            metrics_sha256=evaluation["metrics_sha256"],
            policy_sha256=evaluation["policy_sha256"],
        )
        evidence_sha256 = _canonical_sha256(evidence)
        if evaluation.get("evidence_sha256") != evidence_sha256:
            raise ValueError("Qlib evaluation evidence provenance is invalid")
        now = _now()
        with self.engine.begin() as connection:
            result = connection.execute(
                update(factor_candidates)
                .where(
                    factor_candidates.c.id == candidate_id,
                    factor_candidates.c.status == "gate_passed",
                )
                .values(
                    status="promoted",
                    promoted_evaluation_id=evaluation["id"],
                    promotion_evidence_sha256=evidence_sha256,
                    promoted_by=actor,
                    promoted_at=now,
                    updated_at=now,
                )
            )
            if result.rowcount != 1:
                raise ValueError("candidate state changed during promotion")
            self._event(
                connection,
                run_id=candidate["research_run_id"],
                candidate_id=candidate_id,
                event_type="candidate.promoted",
                actor=actor,
                payload={
                    "reason": reason,
                    "evaluation_id": evaluation["id"],
                    "evidence_sha256": evidence_sha256,
                    "code_sha256": current_code_sha256,
                    "values_sha256": current_values_sha256,
                },
            )
        return self.get_candidate(candidate_id)

    def latest_evaluation(self, candidate_id: str) -> dict[str, Any] | None:
        statement = (
            select(factor_evaluations)
            .where(factor_evaluations.c.factor_candidate_id == candidate_id)
            .order_by(factor_evaluations.c.created_at.desc())
            .limit(1)
        )
        with self.engine.connect() as connection:
            row = connection.execute(statement).first()
        return self._decode_evaluation(row_dict(row)) if row else None

    def get_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(factor_evaluations).where(factor_evaluations.c.id == evaluation_id)
            ).first()
        if row is None:
            raise KeyError(evaluation_id)
        return self._decode_evaluation(row_dict(row))

    def list_events(self, run_id: str, limit: int = 200) -> list[dict[str, Any]]:
        statement = (
            select(research_events)
            .where(research_events.c.research_run_id == run_id)
            .order_by(research_events.c.created_at.desc())
            .limit(limit)
        )
        with self.engine.connect() as connection:
            rows = [row_dict(row) for row in connection.execute(statement)]
        for row in rows:
            row["payload"] = row.pop("payload_json")
        return rows

    def policy_summary(self) -> dict[str, Any]:
        return asdict(self.policy)

    @staticmethod
    def _event(
        connection: Any,
        *,
        run_id: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
        candidate_id: str | None = None,
    ) -> None:
        connection.execute(
            insert(research_events).values(
                research_run_id=run_id,
                factor_candidate_id=candidate_id,
                event_type=event_type,
                actor=actor,
                payload_json=payload,
                created_at=_now(),
            )
        )

    @staticmethod
    def _decode_run(row: dict[str, Any]) -> dict[str, Any]:
        row["budget"] = row.pop("budget_json")
        row["config"] = row.pop("config_json")
        row["runtime"] = row.pop("runtime_json")
        return row

    @staticmethod
    def _decode_candidate(row: dict[str, Any]) -> dict[str, Any]:
        row["variables"] = row.pop("variables_json")
        return row

    @staticmethod
    def _decode_evaluation(row: dict[str, Any]) -> dict[str, Any]:
        row["metrics"] = row.pop("metrics_json")
        row["gate_reasons"] = row.pop("gate_reasons_json")
        return row
