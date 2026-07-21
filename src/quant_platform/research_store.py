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
from quant_platform.qlib_workflow import require_qlib_workflow_identity
from quant_platform.upstream_versions import QLIB_COMMIT, RDAGENT_COMMIT


def _now() -> datetime:
    return datetime.now(UTC)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


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
    require_qlib_workflow_identity(
        payload.get("qlib_workflow") if isinstance(payload, dict) else None
    )
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
    dataset_identity_sha256: str,
    periods: dict[str, str],
    gate_status: str,
    gate_reasons: list[str],
    evaluator_version: str,
    candidate_code_sha256: str,
    candidate_values_sha256: str,
    submitted_values_sha256: str,
    recompute_evidence_sha256: str,
    artifact_sha256: str,
    metrics_sha256: str,
    policy_sha256: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "dataset": dataset,
        "dataset_identity_sha256": dataset_identity_sha256,
        "periods": periods,
        "gate_status": gate_status,
        "gate_reasons": gate_reasons,
        "evaluator_version": evaluator_version,
        "candidate_code_sha256": candidate_code_sha256,
        "candidate_values_sha256": candidate_values_sha256,
        "submitted_values_sha256": submitted_values_sha256,
        "recompute_evidence_sha256": recompute_evidence_sha256,
        "artifact_sha256": artifact_sha256,
        "metrics_sha256": metrics_sha256,
        "policy_sha256": policy_sha256,
    }


@dataclass(frozen=True, slots=True)
class FactorGatePolicy:
    """Versioned, deterministic admission policy for production factor candidates."""

    version: str = "factor-gate-v3-hac-bh"
    min_abs_ic: float = 0.02
    min_abs_icir: float = 0.50
    min_abs_rank_ic: float = 0.025
    min_abs_rank_icir: float = 0.50
    max_turnover: float = 0.60
    max_correlation: float = 0.75
    min_cost_adjusted_return: float = 0.0
    min_selection_days: int = 100
    max_bh_q_value: float = 0.10

    def evaluate(self, metrics: dict[str, float | None]) -> tuple[str, list[str]]:
        checks = (
            ("ic", lambda value: value >= self.min_abs_ic, f"directed IC >= {self.min_abs_ic}"),
            (
                "icir",
                lambda value: value >= self.min_abs_icir,
                f"directed ICIR >= {self.min_abs_icir}",
            ),
            (
                "rank_ic",
                lambda value: value >= self.min_abs_rank_ic,
                f"directed RankIC >= {self.min_abs_rank_ic}",
            ),
            (
                "rank_icir",
                lambda value: value >= self.min_abs_rank_icir,
                f"directed RankICIR >= {self.min_abs_rank_icir}",
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
                "selection_days",
                lambda value: value >= self.min_selection_days,
                f"validation selection days >= {self.min_selection_days}",
            ),
            (
                "coverage_pass_rate",
                lambda value: value >= 0.95,
                "coverage pass rate >= 0.95",
            ),
            (
                "mean_coverage_ratio",
                lambda value: value >= 0.80,
                "mean universe coverage >= 0.80",
            ),
            (
                "constant_day_rate",
                lambda value: value <= 0.05,
                "constant factor days <= 0.05",
            ),
            (
                "hac_p_value",
                lambda value: 0 <= value <= self.max_bh_q_value,
                f"HAC p-value <= {self.max_bh_q_value}",
            ),
            (
                "bh_q_value",
                lambda value: 0 <= value <= self.max_bh_q_value,
                f"BH-FDR q-value <= {self.max_bh_q_value}",
            ),
        )
        reasons: list[str] = []
        for name, predicate, expectation in checks:
            value = metrics.get(name)
            if value is None:
                reasons.append(f"{name} is missing; expected {expectation}")
            elif not predicate(float(value)):
                reasons.append(f"{name}={value:g} failed; expected {expectation}")
        raw_valid_ic = metrics.get("raw_valid_ic")
        raw_selection_ic = metrics.get("raw_selection_ic")
        ic = metrics.get("ic")
        rank_ic = metrics.get("rank_ic")
        if raw_valid_ic is None or raw_selection_ic is None:
            reasons.append("raw direction and selection IC are required")
        elif float(raw_valid_ic) * float(raw_selection_ic) <= 0:
            reasons.append("raw direction and selection IC must have the same sign")
        if ic is not None and rank_ic is not None and float(ic) * float(rank_ic) <= 0:
            reasons.append("IC and RankIC must have the same direction")
        return ("passed" if not reasons else "failed", reasons)


# Executor version bound into external factor evaluation evidence. Mirrors the
# hardcoded "factor-recompute-v1" check in record_evaluation: the version lives
# here because this module validates the evidence chain.
EXTERNAL_EVALUATOR_VERSION = "external-factor-eval-v2"

# Gate status produced when an external factor lacks enough independent events
# or signal days; the candidate keeps full records and may be re-evaluated.
EXTERNAL_GATE_INSUFFICIENT = "insufficient_evidence"


def _shared_sign_check_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    raw_valid_ic = metrics.get("raw_valid_ic")
    raw_selection_ic = metrics.get("raw_selection_ic")
    ic = metrics.get("ic")
    rank_ic = metrics.get("rank_ic")
    if raw_valid_ic is None or raw_selection_ic is None:
        reasons.append("raw direction and selection IC are required")
    elif float(raw_valid_ic) * float(raw_selection_ic) <= 0:
        reasons.append("raw direction and selection IC must have the same sign")
    if ic is not None and rank_ic is not None and float(ic) * float(rank_ic) <= 0:
        reasons.append("IC and RankIC must have the same direction")
    return reasons


@dataclass(frozen=True, slots=True)
class ExternalEventGatePolicy:
    """Admission gate for sparse event-driven external NLP factors.

    Sparse event factors (announcement_tone, irm_qa_sentiment_daily) only carry
    values on a few event days/instruments, so the cross-sectional coverage
    gates of FactorGatePolicy can never apply (design draft 4.3/6.9). Evidence
    is gated on the number of independent event days (effective decisions)
    instead of daily universe coverage; when that count is too low the gate
    reports "insufficient_evidence" rather than relaxing thresholds.

    候选参数：以下阈值均为保守默认值，需预注册评审后冻结，不作为已评审依据。
    """

    version: str = "external-event-gate-v1"
    min_abs_ic: float = 0.02
    min_abs_icir: float = 0.50
    min_abs_rank_ic: float = 0.025
    min_abs_rank_icir: float = 0.50
    max_turnover: float = 0.60
    max_correlation: float = 0.75
    min_cost_adjusted_return: float = 0.0
    min_event_days: int = 30
    max_bh_q_value: float = 0.10

    def evaluate(self, metrics: dict[str, float | None]) -> tuple[str, list[str]]:
        checks = (
            ("ic", lambda value: value >= self.min_abs_ic, f"directed IC >= {self.min_abs_ic}"),
            (
                "icir",
                lambda value: value >= self.min_abs_icir,
                f"directed ICIR >= {self.min_abs_icir}",
            ),
            (
                "rank_ic",
                lambda value: value >= self.min_abs_rank_ic,
                f"directed RankIC >= {self.min_abs_rank_ic}",
            ),
            (
                "rank_icir",
                lambda value: value >= self.min_abs_rank_icir,
                f"directed RankICIR >= {self.min_abs_rank_icir}",
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
                "hac_p_value",
                lambda value: 0 <= value <= self.max_bh_q_value,
                f"HAC p-value <= {self.max_bh_q_value}",
            ),
            (
                "bh_q_value",
                lambda value: 0 <= value <= self.max_bh_q_value,
                f"BH-FDR q-value <= {self.max_bh_q_value}",
            ),
        )
        reasons: list[str] = []
        for name, predicate, expectation in checks:
            value = metrics.get(name)
            if value is None:
                reasons.append(f"{name} is missing; expected {expectation}")
            elif not predicate(float(value)):
                reasons.append(f"{name}={value:g} failed; expected {expectation}")
        reasons.extend(_shared_sign_check_reasons(metrics))
        selection_days = metrics.get("selection_days")
        if selection_days is None:
            reasons.append(
                "selection_days is missing; expected "
                f"independent event days >= {self.min_event_days}"
            )
        elif float(selection_days) < self.min_event_days:
            return (
                EXTERNAL_GATE_INSUFFICIENT,
                [
                    f"independent event days={selection_days:g} below {self.min_event_days}; "
                    "evidence is insufficient and gate thresholds were not relaxed",
                    *reasons,
                ],
            )
        return ("passed" if not reasons else "failed", reasons)


@dataclass(frozen=True, slots=True)
class MarketTimeseriesGatePolicy:
    """Admission gate for market-level timeseries external NLP factors.

    news_sentiment_daily carries a single MARKET pseudo-instrument, so there is
    no cross-section: the signal is evaluated as a timeseries against benchmark
    forward returns and gated on the number of independent signal days (design
    draft 4.3 effective decisions), never on cross-sectional coverage.

    候选参数：以下阈值均为保守默认值，需预注册评审后冻结，不作为已评审依据。
    """

    version: str = "external-market-gate-v1"
    min_abs_ic: float = 0.02
    min_abs_rank_ic: float = 0.025
    max_turnover: float = 1.0
    min_cost_adjusted_return: float = 0.0
    min_signal_days: int = 60
    max_bh_q_value: float = 0.10

    def evaluate(self, metrics: dict[str, float | None]) -> tuple[str, list[str]]:
        checks = (
            ("ic", lambda value: value >= self.min_abs_ic, f"directed IC >= {self.min_abs_ic}"),
            (
                "rank_ic",
                lambda value: value >= self.min_abs_rank_ic,
                f"directed RankIC >= {self.min_abs_rank_ic}",
            ),
            (
                "turnover",
                lambda value: value <= self.max_turnover,
                f"turnover <= {self.max_turnover}",
            ),
            (
                "cost_adjusted_return",
                lambda value: value > self.min_cost_adjusted_return,
                f"cost-adjusted return > {self.min_cost_adjusted_return}",
            ),
            (
                "hac_p_value",
                lambda value: 0 <= value <= self.max_bh_q_value,
                f"HAC p-value <= {self.max_bh_q_value}",
            ),
            (
                "bh_q_value",
                lambda value: 0 <= value <= self.max_bh_q_value,
                f"BH-FDR q-value <= {self.max_bh_q_value}",
            ),
        )
        reasons: list[str] = []
        for name, predicate, expectation in checks:
            value = metrics.get(name)
            if value is None:
                reasons.append(f"{name} is missing; expected {expectation}")
            elif not predicate(float(value)):
                reasons.append(f"{name}={value:g} failed; expected {expectation}")
        reasons.extend(_shared_sign_check_reasons(metrics))
        selection_days = metrics.get("selection_days")
        if selection_days is None:
            reasons.append(
                "selection_days is missing; expected "
                f"independent signal days >= {self.min_signal_days}"
            )
        elif float(selection_days) < self.min_signal_days:
            return (
                EXTERNAL_GATE_INSUFFICIENT,
                [
                    f"independent signal days={selection_days:g} below {self.min_signal_days}; "
                    "evidence is insufficient and gate thresholds were not relaxed",
                    *reasons,
                ],
            )
        return ("passed" if not reasons else "failed", reasons)


ExternalGatePolicy = ExternalEventGatePolicy | MarketTimeseriesGatePolicy


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

    def requeue_run(self, run_id: str, *, actor: str = "operator") -> None:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(research_runs.c.status).where(research_runs.c.id == run_id).with_for_update()
            ).first()
            if row is None:
                raise KeyError(run_id)
            if row.status not in {"failed", "cancelled"}:
                raise ValueError("only failed or cancelled research runs may be requeued")
            connection.execute(
                update(research_runs)
                .where(research_runs.c.id == run_id)
                .values(
                    status="queued",
                    error=None,
                    started_at=None,
                    finished_at=None,
                    updated_at=_now(),
                )
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="run.requeued",
                actor=actor,
                payload={},
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
        experiment_family_id: str | None = None,
        label_horizon_days: int = 1,
        experiment_count: int = 1,
        actor: str = "rdagent-importer",
    ) -> dict[str, Any]:
        candidate_id = uuid.uuid4().hex
        now = _now()
        status = "awaiting_evaluation" if rdagent_decision is not False else "rejected_by_rdagent"
        requires_artifacts = rdagent_decision is not False
        actual_code_sha256 = _artifact_hash(code_path, "factor code", required=requires_artifacts)
        values_sha256 = _artifact_hash(values_path, "factor values", required=requires_artifacts)
        family_id = str(experiment_family_id or run_id).strip()
        if not family_id or label_horizon_days < 1 or experiment_count < 1:
            raise ValueError("factor experiment family, label horizon and count are required")
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
                        experiment_family_id=family_id,
                        label_horizon_days=label_horizon_days,
                        experiment_count=experiment_count,
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

    def find_candidate(self, *, name: str, values_sha256: str) -> dict[str, Any] | None:
        statement = (
            select(factor_candidates)
            .where(
                factor_candidates.c.name == name,
                factor_candidates.c.values_sha256 == values_sha256,
            )
            .order_by(factor_candidates.c.created_at.asc())
            .limit(1)
        )
        with self.engine.connect() as connection:
            row = connection.execute(statement).first()
        if row is None:
            return None
        candidate = self._decode_candidate(row_dict(row))
        candidate["latest_evaluation"] = self.latest_evaluation(candidate["id"])
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
        dataset_identity_sha256: str,
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
        test_start: date,
        test_end: date,
        metrics: dict[str, float | None],
        artifact_path: str | None,
        recomputed_values_path: str,
        recomputed_values_sha256: str,
        recompute_evidence: dict[str, Any],
        actor: str = "qlib-evaluator",
    ) -> dict[str, Any]:
        if not (train_start <= train_end < valid_start <= valid_end < test_start <= test_end):
            raise ValueError(
                "train, validation, and test windows must be ordered and non-overlapping"
            )
        if not _is_sha256(dataset_identity_sha256):
            raise ValueError("factor evaluation requires immutable dataset identity")
        candidate = self.get_candidate(candidate_id)
        embargo_days = max(5, int(candidate["label_horizon_days"]))
        if (test_start - valid_end).days <= embargo_days:
            raise ValueError(
                f"reserved final test requires purge/embargo gap greater than {embargo_days} days"
            )
        if candidate["status"] in {"promoted", "retired"}:
            raise ValueError(f"cannot evaluate candidate in {candidate['status']} state")
        gate_status, reasons = self.policy.evaluate(metrics)
        if metrics.get("statistical_contract_version") not in {
            None,
            "research-statistics-v1-hac-bh-dsr",
        }:
            raise ValueError("factor statistical contract is obsolete")
        current_code_sha256 = _artifact_hash(
            candidate.get("code_path"), "factor code", required=True
        )
        submitted_values_sha256 = _artifact_hash(
            candidate.get("values_path"), "factor values", required=True
        )
        if current_code_sha256 != candidate.get("code_sha256"):
            raise ValueError("factor code artifact changed after RD-Agent import")
        if submitted_values_sha256 != candidate.get("values_sha256"):
            raise ValueError("factor values artifact changed after RD-Agent import")
        actual_recomputed_sha256 = _artifact_hash(
            recomputed_values_path, "independently recomputed factor values", required=True
        )
        if actual_recomputed_sha256 != recomputed_values_sha256:
            raise ValueError("recomputed factor values do not match evaluator SHA-256")
        if recompute_evidence.get("code_sha256") != current_code_sha256:
            raise ValueError("factor recomputation evidence is not bound to candidate code")
        if recompute_evidence.get("dataset_identity_sha256") != dataset_identity_sha256:
            raise ValueError("factor recomputation evidence is not bound to the Qlib dataset")
        if recompute_evidence.get("authoritative_values_sha256") != actual_recomputed_sha256:
            raise ValueError("factor recomputation evidence is not bound to authoritative values")
        if recompute_evidence.get("executor_version") != "factor-recompute-v1":
            raise ValueError("factor recomputation evidence has an unsupported executor version")
        if int(recompute_evidence.get("label_horizon_days") or 0) != int(
            candidate["label_horizon_days"]
        ):
            raise ValueError("factor recomputation evidence has the wrong label horizon")
        if not recompute_evidence.get("provider_input_sha256"):
            raise ValueError("factor recomputation evidence is not bound to provider input")
        expected_periods = {
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "valid_start": valid_start.isoformat(),
            "valid_end": valid_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
        }
        if recompute_evidence.get("periods") != expected_periods:
            raise ValueError("factor recomputation evidence is not bound to evaluation periods")
        submitted_comparison = recompute_evidence.get("submitted_comparison")
        if not isinstance(submitted_comparison, dict) or not (
            submitted_comparison.get("available") is True
            and submitted_comparison.get("exact_match") is True
            and submitted_comparison.get("submitted_sha256") == submitted_values_sha256
        ):
            raise ValueError("submitted factor values do not match independent recomputation")
        recompute_evidence_sha256 = _canonical_sha256(recompute_evidence)
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
            dataset_identity_sha256=dataset_identity_sha256,
            periods=periods,
            gate_status=gate_status,
            gate_reasons=reasons,
            evaluator_version=self.policy.version,
            candidate_code_sha256=current_code_sha256,
            candidate_values_sha256=actual_recomputed_sha256,
            submitted_values_sha256=submitted_values_sha256,
            recompute_evidence_sha256=recompute_evidence_sha256,
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
                    dataset_identity_sha256=dataset_identity_sha256,
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
                    candidate_values_sha256=actual_recomputed_sha256,
                    submitted_values_sha256=submitted_values_sha256,
                    recomputed_values_sha256=actual_recomputed_sha256,
                    recompute_evidence_json=recompute_evidence,
                    hac_p_value=metrics.get("hac_p_value"),
                    bh_q_value=metrics.get("bh_q_value"),
                    statistical_contract_version="research-statistics-v1-hac-bh-dsr",
                    signal_frequency="day",
                    signal_horizon=f"{int(candidate.get('label_horizon_days') or 1)}d",
                    execution_frequency="day",
                    execution_contract_hash=evidence_sha256,
                    qlib_version=f"0.0.dev0+g{QLIB_COMMIT}",
                    qlib_commit=QLIB_COMMIT,
                    rdagent_version=f"0.0.dev0+g{RDAGENT_COMMIT}",
                    rdagent_commit=RDAGENT_COMMIT,
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
                .values(
                    status=candidate_status,
                    values_path=recomputed_values_path,
                    values_sha256=actual_recomputed_sha256,
                    updated_at=now,
                )
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

    def record_external_evaluation(
        self,
        candidate_id: str,
        *,
        policy: ExternalGatePolicy,
        dataset: str,
        dataset_identity_sha256: str,
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
        test_start: date,
        test_end: date,
        evaluation_shape: str,
        metrics: dict[str, Any] | None,
        insufficient_reasons: list[str] | None,
        external_evidence: dict[str, Any],
        artifact_path: str | None,
        actor: str = "external-factor-evaluator",
    ) -> dict[str, Any]:
        """Record an external NLP factor evaluation with the same anchoring chain.

        External factors (announcement/corpus NLP artifacts) cannot be
        recomputed from market data, so the registered values artifact is the
        authoritative values: ``recomputed_values_sha256`` and
        ``submitted_values_sha256`` are both anchored to the candidate values
        sha256, and ``recompute_evidence_json`` carries the external evaluator
        evidence instead of a factor-recompute proof. Gate outcomes advance the
        candidate through the same state machine, plus the explicit
        ``insufficient_evidence`` state when independent events/signal days are
        too few (fail-closed, thresholds are not relaxed).
        """

        if not (train_start <= train_end < valid_start <= valid_end < test_start <= test_end):
            raise ValueError(
                "train, validation, and test windows must be ordered and non-overlapping"
            )
        if not _is_sha256(dataset_identity_sha256):
            raise ValueError("factor evaluation requires immutable dataset identity")
        candidate = self.get_candidate(candidate_id)
        embargo_days = max(5, int(candidate["label_horizon_days"]))
        if (test_start - valid_end).days <= embargo_days:
            raise ValueError(
                f"reserved final test requires purge/embargo gap greater than {embargo_days} days"
            )
        if candidate["status"] in {"promoted", "retired"}:
            raise ValueError(f"cannot evaluate candidate in {candidate['status']} state")
        expected_policy = {
            "sparse_event": ExternalEventGatePolicy,
            "market_timeseries": MarketTimeseriesGatePolicy,
        }.get(evaluation_shape)
        if expected_policy is None:
            raise ValueError(f"unknown external evaluation shape: {evaluation_shape!r}")
        if not isinstance(policy, expected_policy):
            raise ValueError(
                f"external evaluation shape {evaluation_shape!r} requires "
                f"{expected_policy.__name__}"
            )
        if metrics is not None and metrics.get("statistical_contract_version") not in {
            None,
            "research-statistics-v1-hac-bh-dsr",
        }:
            raise ValueError("factor statistical contract is obsolete")
        current_code_sha256 = _artifact_hash(
            candidate.get("code_path"), "factor code", required=True
        )
        current_values_sha256 = _artifact_hash(
            candidate.get("values_path"), "factor values", required=True
        )
        if current_code_sha256 != candidate.get("code_sha256"):
            raise ValueError("factor code artifact changed after import")
        if current_values_sha256 != candidate.get("values_sha256"):
            raise ValueError("factor values artifact changed after import")
        if external_evidence.get("executor_version") != EXTERNAL_EVALUATOR_VERSION:
            raise ValueError("external evaluation evidence has an unsupported executor version")
        if external_evidence.get("evaluation_shape") != evaluation_shape:
            raise ValueError("external evaluation evidence is not bound to the evaluation shape")
        if external_evidence.get("candidate_code_sha256") != current_code_sha256:
            raise ValueError("external evaluation evidence is not bound to candidate code")
        if external_evidence.get("candidate_values_sha256") != current_values_sha256:
            raise ValueError("external evaluation evidence is not bound to candidate values")
        if external_evidence.get("authoritative_values_sha256") != current_values_sha256:
            raise ValueError("external evaluation evidence is not bound to authoritative values")
        if external_evidence.get("dataset_identity_sha256") != dataset_identity_sha256:
            raise ValueError("external evaluation evidence is not bound to the Qlib dataset")
        if external_evidence.get("policy") != asdict(policy):
            raise ValueError("external evaluation evidence is not bound to the gate policy")
        if int(external_evidence.get("label_horizon_days") or 0) != int(
            candidate["label_horizon_days"]
        ):
            raise ValueError("external evaluation evidence has the wrong label horizon")
        expected_periods = {
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "valid_start": valid_start.isoformat(),
            "valid_end": valid_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
        }
        if external_evidence.get("periods") != expected_periods:
            raise ValueError("external evaluation evidence is not bound to evaluation periods")
        if not _is_sha256(str(external_evidence.get("input_data_sha256") or "")):
            raise ValueError("external evaluation evidence is not bound to the price input")
        metrics = dict(metrics) if metrics is not None else None
        if metrics is None:
            gate_status = EXTERNAL_GATE_INSUFFICIENT
            reasons = [str(reason) for reason in (insufficient_reasons or []) if str(reason)]
            if not reasons:
                raise ValueError("insufficient-evidence evaluations require explicit reasons")
        else:
            gate_status, reasons = policy.evaluate(metrics)
        if gate_status == "passed" and not artifact_path:
            raise ValueError("passed external evaluation requires a durable result artifact")
        artifact_sha256 = _artifact_hash(
            artifact_path, "external evaluation", required=gate_status == "passed"
        )
        if artifact_path and metrics is not None:
            artifact_metrics = _evaluation_artifact_metrics(artifact_path, candidate_id)
            if _canonical_sha256(artifact_metrics) != _canonical_sha256(metrics):
                raise ValueError(
                    "external evaluation artifact metrics do not match imported metrics"
                )
        metrics_payload = metrics if metrics is not None else {}
        metrics_sha256 = _canonical_sha256(metrics_payload)
        policy_dict = asdict(policy)
        policy_sha256 = _canonical_sha256(policy_dict)
        evidence = _evaluation_evidence(
            candidate_id=candidate_id,
            dataset=dataset,
            dataset_identity_sha256=dataset_identity_sha256,
            periods=expected_periods,
            gate_status=gate_status,
            gate_reasons=reasons,
            evaluator_version=policy.version,
            candidate_code_sha256=current_code_sha256,
            candidate_values_sha256=current_values_sha256,
            submitted_values_sha256=current_values_sha256,
            recompute_evidence_sha256=_canonical_sha256(external_evidence),
            artifact_sha256=artifact_sha256 or "",
            metrics_sha256=metrics_sha256,
            policy_sha256=policy_sha256,
        )
        evidence_sha256 = _canonical_sha256(evidence)
        evaluation_id = uuid.uuid4().hex
        now = _now()
        scalars = {
            key: metrics_payload.get(key)
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
        candidate_status = {
            "passed": "gate_passed",
            "failed": "gate_failed",
            EXTERNAL_GATE_INSUFFICIENT: EXTERNAL_GATE_INSUFFICIENT,
        }[gate_status]
        with self.engine.begin() as connection:
            connection.execute(
                insert(factor_evaluations).values(
                    id=evaluation_id,
                    factor_candidate_id=candidate_id,
                    dataset=dataset,
                    dataset_identity_sha256=dataset_identity_sha256,
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                    test_start=test_start,
                    test_end=test_end,
                    **scalars,
                    metrics_json=metrics_payload,
                    gate_status=gate_status,
                    gate_reasons_json=reasons,
                    evaluator_version=policy.version,
                    artifact_path=artifact_path,
                    artifact_sha256=artifact_sha256,
                    candidate_code_sha256=current_code_sha256,
                    candidate_values_sha256=current_values_sha256,
                    submitted_values_sha256=current_values_sha256,
                    recomputed_values_sha256=current_values_sha256,
                    recompute_evidence_json=external_evidence,
                    hac_p_value=metrics_payload.get("hac_p_value"),
                    bh_q_value=metrics_payload.get("bh_q_value"),
                    statistical_contract_version="research-statistics-v1-hac-bh-dsr",
                    signal_frequency="day",
                    signal_horizon=f"{int(candidate.get('label_horizon_days') or 1)}d",
                    execution_frequency="day",
                    execution_contract_hash=evidence_sha256,
                    qlib_version=f"0.0.dev0+g{QLIB_COMMIT}",
                    qlib_commit=QLIB_COMMIT,
                    rdagent_version=f"0.0.dev0+g{RDAGENT_COMMIT}",
                    rdagent_commit=RDAGENT_COMMIT,
                    metrics_sha256=metrics_sha256,
                    policy_json=policy_dict,
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
                    "evaluation_shape": evaluation_shape,
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
        artifact_sha256 = _artifact_hash(artifact_path, "Qlib evaluation", required=True)
        if artifact_sha256 != evaluation.get("artifact_sha256"):
            raise ValueError("Qlib evaluation artifact changed after evaluation")
        artifact_metrics = _evaluation_artifact_metrics(str(artifact_path), candidate_id)
        if _canonical_sha256(artifact_metrics) != evaluation.get("metrics_sha256"):
            raise ValueError("Qlib evaluation artifact no longer matches recorded metrics")
        evidence = _evaluation_evidence(
            candidate_id=candidate_id,
            dataset=evaluation["dataset"],
            dataset_identity_sha256=evaluation["dataset_identity_sha256"],
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
            submitted_values_sha256=evaluation["submitted_values_sha256"],
            recompute_evidence_sha256=_canonical_sha256(evaluation["recompute_evidence"]),
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
        row["recompute_evidence"] = row.pop("recompute_evidence_json")
        return row
