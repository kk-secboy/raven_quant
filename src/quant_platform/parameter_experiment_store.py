from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, insert, select, update

from quant_data.database import (
    factor_evaluations,
    open_database,
    parameter_experiment_trials,
    parameter_experiments,
    research_run_artifacts,
    row_dict,
    strategy_factors,
    strategy_versions,
)
from quant_data.execution_contract import build_strategy_execution_contract

from .parameter_experiments import (
    PORTFOLIO_CONSTRUCTION_CANDIDATES,
    ParameterValue,
    build_portfolio_construction_trials,
    merge_admitted_trial_ledgers,
    portfolio_trial_comparability_evidence,
    select_frozen_portfolio_config,
    split_model_portfolio_period,
)
from .strategy_research_evaluation import (
    STRATEGY_FULL_STACK_MODE,
    STRATEGY_POLICY_ONLY_MODE,
    STRATEGY_RESEARCH_COMPETITION_VERSION,
)
from .strategy_rule_compiler import (
    validate_compiled_strategy_artifact,
    validate_strategy_rule_binding,
)
from .strategy_store import StrategyStore
from .transparent_baseline_runner import bind_transparent_baseline_job_identity


def _now() -> datetime:
    return datetime.now(UTC)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _terminal_result_state(
    result: Mapping[str, Any],
    trial_results: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> tuple[str, str | None]:
    """Validate an explicit terminal trial ledger and map it to DB state.

    Results created before the terminal-result contract omitted the execution
    counters; retain their historical behavior.  New explicit results
    distinguish executable trials rejected by statistics from trials that did
    not execute.
    """

    explicit_status = result.get("status")
    if explicit_status is None:
        return "succeeded", None
    status = str(explicit_status)
    if status not in {"ok", "failed"}:
        raise ValueError("parameter experiment terminal status is invalid")
    failed = [item for item in trial_results if item.get("status") == "failed"]
    missing_errors = [
        int(item.get("trial_index", -1))
        for item in failed
        if not str(item.get("error") or "").strip()
    ]
    if missing_errors:
        raise ValueError("failed parameter trials must retain their execution errors")
    expected_failed = len(failed)
    expected_succeeded = len(trial_results) - expected_failed
    has_failed_count = "execution_failed_count" in summary
    has_succeeded_count = "execution_succeeded_count" in summary
    if status == "ok" and not has_failed_count and not has_succeeded_count:
        # Historical successful artifacts predate the explicit split between
        # execution failure and statistical rejection.
        return "succeeded", None
    if (
        has_failed_count is not has_succeeded_count
        or not has_failed_count
        or int(summary.get("execution_failed_count", -1)) != expected_failed
        or int(summary.get("execution_succeeded_count", -1)) != expected_succeeded
    ):
        raise ValueError("parameter experiment execution counts are inconsistent")
    if status == "ok":
        if failed:
            raise ValueError("successful parameter experiment contains failed executions")
        return "succeeded", None
    error = str(result.get("error") or "").strip()
    if (
        result.get("failure_kind") != "trial_execution_error"
        or not failed
        or not error
    ):
        raise ValueError("failed parameter experiment terminal evidence is incomplete")
    return "failed", error


def _portfolio_competition_spec_sha256(
    *,
    strategy_version_id: str,
    dataset: str,
    dataset_identity_sha256: str | None,
    periods: dict[str, Any],
    parameter_grid: dict[str, Any],
    baseline_config: dict[str, Any],
) -> str:
    return _canonical_sha256(
        {
            "contract_version": "model-portfolio-competition-v1",
            "strategy_version_id": strategy_version_id,
            "dataset": dataset,
            "dataset_identity_sha256": dataset_identity_sha256,
            "periods": periods,
            "parameter_grid": parameter_grid,
            "baseline_config": baseline_config,
        }
    )


class ParameterExperimentStore:
    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def create(
        self,
        *,
        strategy_version_id: str,
        dataset: str,
        periods: dict[str, dict[str, str]],
        parameter_grid: dict[str, list[ParameterValue]],
        baseline_config: dict[str, Any],
        trials: list[dict[str, Any]],
        artifact_root: Path,
        created_by: str,
        dataset_identity_sha256: str | None = None,
    ) -> dict[str, Any]:
        experiment_id = uuid.uuid4().hex
        artifact_path = artifact_root / experiment_id
        now = _now()
        with self.engine.begin() as connection:
            model_evidence = StrategyStore._model_signal_evidence(
                connection, baseline_config
            )
            evidence = connection.execute(
                select(
                    factor_evaluations.c.dataset,
                    factor_evaluations.c.valid_end,
                    factor_evaluations.c.metrics_json,
                    factor_evaluations.c.evaluator_version,
                    factor_evaluations.c.is_legacy,
                )
                .join(
                    strategy_factors,
                    strategy_factors.c.factor_evaluation_id == factor_evaluations.c.id,
                )
                .where(strategy_factors.c.strategy_version_id == strategy_version_id)
            ).all()
            windows = [
                section
                for section in periods.values()
                if isinstance(section, dict) and {"start", "end"}.issubset(section)
            ]
            if len(windows) < 2:
                raise ValueError("parameter experiments require separated research windows")
            experiment_start = min(date.fromisoformat(section["start"]) for section in windows)
            experiment_end = max(date.fromisoformat(section["end"]) for section in windows)
            governance: dict[str, Any]
            if model_evidence is None:
                if not evidence or any(
                    item.dataset != dataset
                    or item.is_legacy
                    or str(item.evaluator_version) != "factor-gate-v3-hac-bh"
                    for item in evidence
                ):
                    raise ValueError(
                        "parameter experiments require matching factor evaluation v3 evidence"
                    )
                valid_start = max(
                    date.fromisoformat(str(dict(item.metrics_json)["selection_start"]))
                    for item in evidence
                )
                valid_end = min(item.valid_end for item in evidence)
                governance = {
                    "mode": "factor_pre_final_parameter_search",
                    "final_oos_opened": False,
                    "selection_start": valid_start.isoformat(),
                    "pre_final_cutoff": valid_end.isoformat(),
                }
            else:
                candidate = model_evidence["candidate"]
                primary_evaluation = model_evidence["evaluation"]
                if (
                    str(candidate.dataset) != dataset
                    or str(primary_evaluation.dataset) != dataset
                    or str(primary_evaluation.dataset_identity_sha256)
                    != str(candidate.dataset_identity_sha256)
                    or (
                        dataset_identity_sha256 is not None
                        and dataset_identity_sha256
                        != str(candidate.dataset_identity_sha256)
                    )
                ):
                    raise ValueError(
                        "model portfolio experiment dataset does not match admission evidence"
                    )
                valid_start = primary_evaluation.valid_start
                valid_end = primary_evaluation.valid_end
                expected_identity = model_evidence["identity"]
                baseline_contract = build_strategy_execution_contract(baseline_config)
                allowed_trial_fields = set(parameter_grid)
                if allowed_trial_fields != {"portfolio_construction"} or len(trials) != 2:
                    raise ValueError(
                        "model portfolio competition is exactly one TopK and one QP trial"
                    )
                baseline_frozen = {
                    key: value
                    for key, value in baseline_config.items()
                    if key not in allowed_trial_fields
                }
                constructions: set[str] = set()
                for trial in trials:
                    trial_config = dict(trial.get("config") or {})
                    rebound = StrategyStore._model_signal_evidence(connection, trial_config)
                    if rebound is None or rebound["identity"] != expected_identity:
                        raise ValueError(
                            "model portfolio trials must keep one immutable admitted signal"
                        )
                    if build_strategy_execution_contract(trial_config) != baseline_contract:
                        raise ValueError(
                            "model portfolio trials must share one cost/execution contract"
                        )
                    trial_frozen = {
                        key: value
                        for key, value in trial_config.items()
                        if key not in allowed_trial_fields
                    }
                    if trial_frozen != baseline_frozen:
                        raise ValueError(
                            "model portfolio trial changed an unregistered strategy field"
                        )
                    construction = str(trial_config.get("portfolio_construction") or "")
                    constructions.add(construction)
                if constructions != set(PORTFOLIO_CONSTRUCTION_CANDIDATES):
                    raise ValueError(
                        "model portfolio experiment must preregister TopK and industry-neutral QP"
                    )
                admitted_multiple_testing = merge_admitted_trial_ledgers(
                    model_evidence["formal_admission_binding"]
                )
                governance = {
                    "mode": "model_portfolio_pre_final",
                    "final_oos_opened": False,
                    "dataset_identity_sha256": str(candidate.dataset_identity_sha256),
                    "selection_start": valid_start.isoformat(),
                    "pre_final_cutoff": candidate.pre_final_end.isoformat(),
                    "portfolio_validation_end": valid_end.isoformat(),
                    "model_signal_identity_sha256": expected_identity["identity_sha256"],
                    "model_admission_evidence_sha256": str(
                        candidate.admission_evidence_sha256
                    ),
                    "formal_admission_binding_sha256": model_evidence[
                        "formal_admission_binding"
                    ]["binding_sha256"],
                    "quant_bundle_admission_evidence_sha256": (
                        str(model_evidence["bundle"].admission_evidence_sha256)
                        if model_evidence.get("bundle") is not None
                        else None
                    ),
                    "primary_profile_id": str(primary_evaluation.profile_id),
                    "primary_seed": int(primary_evaluation.seed),
                    "prior_admitted_trial_count": int(
                        admitted_multiple_testing.get("trial_count") or 0
                    ),
                    "competition_spec_sha256": _portfolio_competition_spec_sha256(
                        strategy_version_id=strategy_version_id,
                        dataset=dataset,
                        dataset_identity_sha256=str(candidate.dataset_identity_sha256),
                        periods=periods,
                        parameter_grid=parameter_grid,
                        baseline_config=baseline_config,
                    ),
                }
            if experiment_start < valid_start or experiment_end > valid_end:
                raise ValueError(
                    "parameter experiments must stay inside the validation selection window"
                )
            governed_periods = {**periods, "governance": governance}
            connection.execute(
                insert(parameter_experiments).values(
                    id=experiment_id,
                    strategy_version_id=strategy_version_id,
                    dataset=dataset,
                    status="queued",
                    periods_json=governed_periods,
                    parameter_grid_json=parameter_grid,
                    baseline_config_json=baseline_config,
                    artifact_path=str(artifact_path),
                    created_by=created_by,
                    created_at=now,
                )
            )
            connection.execute(
                insert(parameter_experiment_trials),
                [
                    {
                        "id": uuid.uuid4().hex,
                        "experiment_id": experiment_id,
                        "trial_index": index,
                        "parameters_json": item["parameters"],
                        "config_json": item["config"],
                        "status": "queued",
                        "created_at": now,
                    }
                    for index, item in enumerate(trials)
                ],
            )
        return self.get(experiment_id)

    def ensure_model_portfolio_competition(
        self,
        *,
        strategy_version: Mapping[str, Any],
        dataset: Mapping[str, Any],
        candidate_valid_start: date | str,
        candidate_valid_end: date | str,
        artifact_root: Path,
        created_by: str,
    ) -> dict[str, Any]:
        """Idempotently prepare the governed TopK/QP experiment and job payload."""

        version_id = str(strategy_version.get("id") or "")
        config = dict(strategy_version.get("config") or {})
        dataset_name = str(dataset.get("name") or dataset.get("id") or "")
        dataset_path = str(dataset.get("path") or "")
        dataset_identity = str(
            ((dataset.get("provenance") or {}).get("dataset_identity_sha256")) or ""
        )
        if (
            not version_id
            or not dataset_name
            or not dataset_path
            or len(dataset_identity) != 64
            or any(character not in "0123456789abcdef" for character in dataset_identity)
        ):
            raise ValueError("portfolio competition requires a frozen version and dataset path")
        execution_dataset = dataset.get("execution_dataset")
        execution_method = str(config.get("execution_method") or "open")
        if execution_method in {"twap", "vwap", "next_bar"} and not isinstance(
            execution_dataset, Mapping
        ):
            raise ValueError("minute portfolio competition requires execution_dataset")
        if execution_method == "open" and execution_dataset is not None:
            raise ValueError("daily portfolio competition cannot use a minute dataset")
        valid_start = (
            candidate_valid_start
            if isinstance(candidate_valid_start, date)
            else date.fromisoformat(str(candidate_valid_start))
        )
        valid_end = (
            candidate_valid_end
            if isinstance(candidate_valid_end, date)
            else date.fromisoformat(str(candidate_valid_end))
        )
        periods = split_model_portfolio_period(valid_start, valid_end)
        parameter_grid, trials = build_portfolio_construction_trials(config)
        spec_sha256 = _portfolio_competition_spec_sha256(
            strategy_version_id=version_id,
            dataset=dataset_name,
            dataset_identity_sha256=dataset_identity,
            periods=periods,
            parameter_grid=parameter_grid,
            baseline_config=config,
        )
        # StrategyVersion is immutable.  Reuse its one governed competition
        # regardless of which trusted scheduler actor is reconciling it.
        existing = self.latest_for_version(version_id)
        created = False
        if existing is not None:
            governance = (existing.get("periods") or {}).get("governance") or {}
            if governance.get("competition_spec_sha256") != spec_sha256:
                raise ValueError(
                    "another portfolio competition already exists for this frozen version"
                )
            experiment = existing
        else:
            experiment = self.create(
                strategy_version_id=version_id,
                dataset=dataset_name,
                periods=periods,
                parameter_grid=parameter_grid,
                baseline_config=config,
                trials=trials,
                artifact_root=artifact_root,
                created_by=created_by,
                dataset_identity_sha256=dataset_identity,
            )
            created = True
        job_payload = bind_transparent_baseline_job_identity(
            config=config,
            job_payload={
                "parameter_experiment_id": experiment["id"],
                "strategy_version_id": version_id,
                "dataset": dataset_name,
                "dataset_identity_sha256": dataset_identity,
                "dataset_path": dataset_path,
                "execution_dataset": (
                    dict(execution_dataset)
                    if isinstance(execution_dataset, Mapping)
                    else None
                ),
            },
        )
        return {
            "experiment": experiment,
            "job_payload": job_payload,
            "created": created,
            "needs_job": created or (
                experiment.get("status") == "queued" and not experiment.get("job_id")
            ),
        }

    @staticmethod
    def _prepare_strategy_research_competition(
        *,
        strategy_version: Mapping[str, Any],
        plan: Mapping[str, Any],
        stage: str,
        dataset: Mapping[str, Any],
        artifact_root: Path,
        created_by: str,
    ) -> dict[str, Any]:
        plan_value = dict(plan)
        plan_sha256 = str(plan_value.pop("plan_sha256", ""))
        if (
            plan.get("contract_version") != STRATEGY_RESEARCH_COMPETITION_VERSION
            or plan.get("delivery_status") != "research_only"
            or plan.get("capital_eligible") is not False
            or plan.get("simulation_eligible") is not False
            or plan_sha256 != _canonical_sha256(plan_value)
        ):
            raise ValueError("fin_strategy competition plan is invalid or changed")
        if stage not in {"policy_only", "full_stack"}:
            raise ValueError("fin_strategy competition stage is unsupported")
        stages = [item for item in plan.get("stages") or [] if item.get("stage") == stage]
        if len(stages) != 1:
            raise ValueError("fin_strategy competition stage is not preregistered")
        stage_spec = dict(stages[0])
        expected_mode = (
            STRATEGY_POLICY_ONLY_MODE
            if stage == "policy_only"
            else STRATEGY_FULL_STACK_MODE
        )
        periods = stage_spec.get("periods")
        governance = periods.get("governance") if isinstance(periods, Mapping) else None
        period_segments_valid = isinstance(periods, Mapping) and all(
            isinstance(periods.get(segment), Mapping)
            and set(periods[segment]) == {"start", "end"}
            for segment in ("in_sample", "out_of_sample")
        )
        if (
            stage_spec.get("evaluation_mode") != expected_mode
            or stage_spec.get("job_kind") != "parameter_experiment"
            or stage_spec.get("capital_eligible") is not False
            or stage_spec.get("final_oos_opened") is not False
            or not isinstance(periods, Mapping)
            or not isinstance(governance, Mapping)
            or not period_segments_valid
            or governance.get("final_oos_opened") is not False
        ):
            raise ValueError("fin_strategy stage is not pre-final research-only evidence")

        version_id = str(strategy_version.get("id") or "")
        config = dict(strategy_version.get("config") or {})
        source_artifact_id = str(
            strategy_version.get("source_research_artifact_id")
            or config.get("source_research_artifact_id")
            or ""
        )
        compiled_artifact_id = str(plan.get("compiled_artifact_id") or "")
        compiled_artifact_sha256 = str(plan.get("compiled_artifact_sha256") or "")
        research_run_id = str(plan.get("research_run_id") or "")
        if (
            not version_id
            or (
                strategy_version.get("status") is not None
                and strategy_version.get("status") != "draft"
            )
            or not source_artifact_id
            or not research_run_id
            or source_artifact_id != compiled_artifact_id
            or len(compiled_artifact_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in compiled_artifact_sha256
            )
            or not str(created_by).strip()
            or config.get("source_research_artifact_id") != compiled_artifact_id
            or config.get("strategy_research_artifact_sha256")
            != compiled_artifact_sha256
            or strategy_version.get("horizon_profile", config.get("horizon_profile"))
            != plan.get("horizon")
            or strategy_version.get("strategy_rules_sha256", config.get("strategy_rules_sha256"))
            != config.get("strategy_rules_sha256")
        ):
            raise ValueError("candidate StrategyVersion is not bound to the compiled artifact")
        validate_strategy_rule_binding(config)
        full_stages = [
            item for item in plan.get("stages") or [] if item.get("stage") == "full_stack"
        ]
        if len(full_stages) != 1:
            raise ValueError("fin_strategy plan has no unique full-stack stage")
        full_candidates = [
            item
            for item in full_stages[0].get("trials") or []
            if item.get("role") == "full_stack_challenger"
        ]
        if (
            len(full_candidates) != 1
            or full_candidates[0].get("config") != config
            or full_candidates[0].get("config_sha256") != _canonical_sha256(config)
        ):
            raise ValueError("candidate StrategyVersion config differs from the frozen plan")

        dataset_name = str(dataset.get("name") or dataset.get("id") or "")
        dataset_path = str(dataset.get("path") or "")
        dataset_identity = str(
            ((dataset.get("provenance") or {}).get("dataset_identity_sha256")) or ""
        )
        research_data = config.get("strategy_research_data_contract") or {}
        research_periods = research_data.get("research_periods") or {}
        selection_start = str(research_periods.get("train_start") or "")
        selection_end = str(research_periods.get("valid_end") or "")
        historical = governance.get("historical_validation_periods")
        score_inputs_sha256 = str(stage_spec.get("score_inputs_sha256") or "")
        if (
            not dataset_name
            or not dataset_path
            or len(dataset_identity) != 64
            or any(character not in "0123456789abcdef" for character in dataset_identity)
            or stage_spec.get("dataset") != dataset_name
            or stage_spec.get("dataset_identity_sha256") != dataset_identity
            or research_data.get("dataset_snapshot_id") != dataset_identity
            or not selection_start
            or not selection_end
            or not isinstance(historical, Mapping)
            or set(historical) != {"start", "end"}
            or historical["start"] < selection_start
            or historical["end"] >= periods["in_sample"]["start"]
            or periods["in_sample"]["start"] < selection_start
            or periods["out_of_sample"]["end"] > selection_end
            or governance.get("pre_final_cutoff") != selection_end
            or len(score_inputs_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in score_inputs_sha256
            )
        ):
            raise ValueError("fin_strategy dataset or research periods differ from the plan")
        trials = [dict(item) for item in stage_spec.get("trials") or []]
        if (
            len(trials) != 2
            or {int(item.get("trial_index", -1)) for item in trials} != {0, 1}
            or any(
                item.get("config_sha256") != _canonical_sha256(item.get("config") or {})
                for item in trials
            )
        ):
            raise ValueError("fin_strategy stage trial family is invalid")
        parameter_grid = dict(stage_spec.get("parameter_grid") or {})
        expected_roles = [
            str(item["role"])
            for item in sorted(trials, key=lambda value: int(value["trial_index"]))
        ]
        required_roles = {
            "public_baseline",
            "policy_challenger" if stage == "policy_only" else "full_stack_challenger",
        }
        if (
            set(expected_roles) != required_roles
            or parameter_grid != {"strategy_comparison_role": expected_roles}
        ):
            raise ValueError("fin_strategy stage parameter grid changed")
        baseline_config = dict(
            next(item for item in trials if item["role"] == "public_baseline")["config"]
        )
        stage_periods = {
            "in_sample": dict(periods["in_sample"]),
            "out_of_sample": dict(periods["out_of_sample"]),
        }
        governed_periods = {
            **stage_periods,
            "governance": {
                **dict(governance),
                "mode": expected_mode,
                "final_oos_opened": False,
                "plan_sha256": plan_sha256,
                "stage": stage,
                "compiled_artifact_id": compiled_artifact_id,
                "compiled_artifact_sha256": compiled_artifact_sha256,
                "research_run_id": research_run_id,
                "dataset_identity_sha256": dataset_identity,
                "score_inputs_sha256": score_inputs_sha256,
                "strategy_version_config_sha256": _canonical_sha256(config),
            },
        }
        competition_spec = {
            "contract_version": STRATEGY_RESEARCH_COMPETITION_VERSION,
            "strategy_version_id": version_id,
            "dataset": dataset_name,
            "dataset_identity_sha256": dataset_identity,
            "periods": governed_periods,
            "parameter_grid": parameter_grid,
            "trials": trials,
            "plan_sha256": plan_sha256,
            "stage": stage,
        }
        competition_spec_sha256 = _canonical_sha256(competition_spec)
        governed_periods["governance"][
            "competition_spec_sha256"
        ] = competition_spec_sha256
        return {
            "version_id": version_id,
            "version_config": config,
            "source_artifact_id": compiled_artifact_id,
            "source_artifact_sha256": compiled_artifact_sha256,
            "research_run_id": research_run_id,
            "plan_sha256": plan_sha256,
            "stage": stage,
            "evaluation_mode": expected_mode,
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "dataset_identity_sha256": dataset_identity,
            "periods": governed_periods,
            "parameter_grid": parameter_grid,
            "baseline_config": baseline_config,
            "trials": sorted(trials, key=lambda value: int(value["trial_index"])),
            "artifact_root": Path(artifact_root),
            "created_by": str(created_by),
            "competition_spec_sha256": competition_spec_sha256,
        }

    def ensure_strategy_research_competition(
        self,
        *,
        strategy_version: Mapping[str, Any],
        plan: Mapping[str, Any],
        stage: str,
        dataset: Mapping[str, Any],
        artifact_root: Path,
        created_by: str,
    ) -> dict[str, Any]:
        """Persist one fin_strategy stage in the existing parameter experiment DAG."""

        prepared = self._prepare_strategy_research_competition(
            strategy_version=strategy_version,
            plan=plan,
            stage=stage,
            dataset=dataset,
            artifact_root=artifact_root,
            created_by=created_by,
        )
        experiment, created = self._persist_strategy_research_competition(prepared)
        job_payload = bind_transparent_baseline_job_identity(
            config=dict(strategy_version.get("config") or {}),
            job_payload={
                "parameter_experiment_id": experiment["id"],
                "strategy_version_id": prepared["version_id"],
                "dataset": prepared["dataset_name"],
                "dataset_identity_sha256": prepared["dataset_identity_sha256"],
                "dataset_path": prepared["dataset_path"],
                "strategy_evaluation_mode": prepared["evaluation_mode"],
                "strategy_competition_plan_sha256": prepared["plan_sha256"],
                "strategy_competition_stage": prepared["stage"],
                "execution_dataset": None,
            },
        )
        return {
            "experiment": experiment,
            "job_payload": job_payload,
            "created": created,
            "needs_job": created
            or (experiment.get("status") == "queued" and not experiment.get("job_id")),
        }

    def _persist_strategy_research_competition(
        self, prepared: Mapping[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        """Serialize creation on the immutable StrategyVersion row."""

        existing_id: str | None = None
        created_id: str | None = None
        with self.engine.begin() as connection:
            version = connection.execute(
                select(strategy_versions)
                .where(strategy_versions.c.id == prepared["version_id"])
                .with_for_update()
            ).first()
            if version is None:
                raise KeyError(str(prepared["version_id"]))
            if (
                str(version.source_research_artifact_id or "")
                != prepared["source_artifact_id"]
                or str(version.status) != "draft"
                or version.promotion_stage is not None
                or dict(version.config_json or {}) != prepared["version_config"]
                or str(version.strategy_rules_sha256 or "")
                != str(prepared["version_config"].get("strategy_rules_sha256") or "")
            ):
                raise ValueError("candidate StrategyVersion changed after plan registration")
            artifact = connection.execute(
                select(research_run_artifacts).where(
                    research_run_artifacts.c.id == prepared["source_artifact_id"]
                )
            ).first()
            artifact_path = Path(str(artifact.storage_path)) if artifact is not None else None
            if (
                artifact is None
                or str(artifact.research_run_id) != prepared["research_run_id"]
                or str(artifact.artifact_type) != "fin_strategy_compiled_artifact"
                or str(artifact.contract_version) != "compiled-strategy-proposal-v1"
                or str(artifact.status) != "recorded"
                or bool(artifact.capital_eligible)
                or artifact_path is None
                or not artifact_path.is_file()
                or _sha256_file(artifact_path) != str(artifact.content_sha256)
                or int(artifact_path.stat().st_size) != int(artifact.size_bytes)
            ):
                raise ValueError("compiled strategy source artifact is unavailable or changed")
            try:
                compiled = json.loads(artifact_path.read_text(encoding="utf-8"))
                policy = validate_strategy_rule_binding(prepared["version_config"])
                normalized = validate_compiled_strategy_artifact(
                    compiled,
                    allowed_factor_ids=set((policy or {}).get("alpha_factor_weights") or {}),
                )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError("compiled strategy source artifact is invalid") from exc
            manifest = dict(artifact.manifest_json or {})
            metadata = dict(manifest.get("metadata") or {})
            if (
                normalized.get("artifact_sha256") != prepared["source_artifact_sha256"]
                or _canonical_sha256(manifest) != str(artifact.manifest_sha256)
                or metadata.get("artifact_sha256")
                != prepared["source_artifact_sha256"]
                or metadata.get("scenario") != "fin_strategy"
                or metadata.get("delivery_status") != "research_only"
            ):
                raise ValueError("compiled strategy source identity differs from the plan")

            candidates = connection.execute(
                select(parameter_experiments).where(
                    parameter_experiments.c.strategy_version_id == prepared["version_id"]
                )
            ).all()
            for candidate in candidates:
                governance = dict(candidate.periods_json or {}).get("governance") or {}
                if (
                    governance.get("plan_sha256") == prepared["plan_sha256"]
                    and governance.get("stage") == prepared["stage"]
                ):
                    if (
                        governance.get("competition_spec_sha256")
                        != prepared["competition_spec_sha256"]
                        or str(candidate.dataset) != prepared["dataset_name"]
                        or dict(candidate.periods_json or {}) != prepared["periods"]
                        or dict(candidate.parameter_grid_json or {})
                        != prepared["parameter_grid"]
                        or dict(candidate.baseline_config_json or {})
                        != prepared["baseline_config"]
                    ):
                        raise ValueError(
                            "fin_strategy competition stage already exists with another spec"
                        )
                    existing_trials = connection.execute(
                        select(parameter_experiment_trials).where(
                            parameter_experiment_trials.c.experiment_id == candidate.id
                        )
                    ).all()
                    expected_trials = {
                        int(item["trial_index"]): item for item in prepared["trials"]
                    }
                    if len(existing_trials) != 2 or any(
                        int(item.trial_index) not in expected_trials
                        or dict(item.parameters_json or {})
                        != expected_trials[int(item.trial_index)]["parameters"]
                        or dict(item.config_json or {})
                        != expected_trials[int(item.trial_index)]["config"]
                        for item in existing_trials
                    ):
                        raise ValueError(
                            "fin_strategy competition trial ledger changed after registration"
                        )
                    existing_id = str(candidate.id)
                    break
            if existing_id is None:
                experiment_id = uuid.uuid4().hex
                now = _now()
                connection.execute(
                    insert(parameter_experiments).values(
                        id=experiment_id,
                        strategy_version_id=prepared["version_id"],
                        dataset=prepared["dataset_name"],
                        status="queued",
                        periods_json=prepared["periods"],
                        parameter_grid_json=prepared["parameter_grid"],
                        baseline_config_json=prepared["baseline_config"],
                        artifact_path=str(prepared["artifact_root"] / experiment_id),
                        created_by=prepared["created_by"],
                        created_at=now,
                    )
                )
                connection.execute(
                    insert(parameter_experiment_trials),
                    [
                        {
                            "id": uuid.uuid4().hex,
                            "experiment_id": experiment_id,
                            "trial_index": int(item["trial_index"]),
                            "parameters_json": item["parameters"],
                            "config_json": item["config"],
                            "status": "queued",
                            "created_at": now,
                        }
                        for item in prepared["trials"]
                    ],
                )
                created_id = experiment_id
        identity = existing_id or created_id
        if identity is None:
            raise RuntimeError("fin_strategy competition persistence lost its identity")
        return self.get(identity), created_id is not None

    def attach_job(self, experiment_id: str, job_id: str) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(parameter_experiments)
                .where(parameter_experiments.c.id == experiment_id)
                .values(job_id=job_id)
            )
            if not result.rowcount:
                raise KeyError(experiment_id)

    def mark(
        self,
        experiment_id: str,
        status: str,
        *,
        summary: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        now = _now()
        values: dict[str, Any] = {"status": status, "error": error}
        if summary is not None:
            values["summary_json"] = summary
        if status == "running":
            values["started_at"] = now
        if status in {"succeeded", "failed", "cancelled"}:
            values["finished_at"] = now
        with self.engine.begin() as connection:
            result = connection.execute(
                update(parameter_experiments)
                .where(parameter_experiments.c.id == experiment_id)
                .values(**values)
            )
            if not result.rowcount:
                raise KeyError(experiment_id)

    def requeue(self, experiment_id: str) -> None:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(parameter_experiments.c.status)
                .where(parameter_experiments.c.id == experiment_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(experiment_id)
            if row.status not in {"failed", "cancelled"}:
                raise ValueError("only failed or cancelled parameter experiments may be requeued")
            connection.execute(
                update(parameter_experiments)
                .where(parameter_experiments.c.id == experiment_id)
                .values(
                    status="queued",
                    summary_json=None,
                    error=None,
                    started_at=None,
                    finished_at=None,
                )
            )

    @staticmethod
    def _validate_strategy_research_result(
        *,
        experiment_id: str,
        experiment_row: Any,
        periods: Mapping[str, Any],
        governance: Mapping[str, Any],
        result: Mapping[str, Any],
        trial_results: list[dict[str, Any]],
        summary: Mapping[str, Any],
        expected_trials: list[Any],
    ) -> None:
        mode = str(governance.get("mode") or "")
        if mode not in {STRATEGY_POLICY_ONLY_MODE, STRATEGY_FULL_STACK_MODE}:
            return
        if (
            result.get("status") not in {"ok", "failed"}
            or result.get("experiment_id") != experiment_id
            or result.get("strategy_version_id")
            != str(experiment_row.strategy_version_id)
            or result.get("dataset") != str(experiment_row.dataset)
            or result.get("evaluation_mode") != mode
            or result.get("final_oos_opened") is not False
            or result.get("periods") != dict(periods)
            or governance.get("final_oos_opened") is not False
            or summary.get("final_oos_opened") is not False
            or int(summary.get("trial_count") or 0) != len(expected_trials)
            or int(summary.get("governed_trial_count") or 0)
            != len(expected_trials)
            or int(summary.get("prior_admitted_trial_count") or 0) != 0
            or int(summary.get("succeeded_count") or 0)
            + int(summary.get("failed_count") or 0)
            != len(expected_trials)
        ):
            raise ValueError(
                "fin_strategy parameter experiment result is not pre-final evidence"
            )
        required_identity = {
            "dataset_identity_sha256": governance.get("dataset_identity_sha256"),
            "pre_final_cutoff": governance.get("pre_final_cutoff"),
        }
        if not all(
            isinstance(value, str) and value
            for value in (
                governance.get("plan_sha256"),
                governance.get("competition_spec_sha256"),
                governance.get("compiled_artifact_id"),
                governance.get("compiled_artifact_sha256"),
                governance.get("research_run_id"),
                governance.get("score_inputs_sha256"),
                governance.get("strategy_version_config_sha256"),
                *required_identity.values(),
            )
        ):
            raise ValueError("fin_strategy experiment governance identity is incomplete")
        registered = {int(item.trial_index): item for item in expected_trials}
        for item in trial_results:
            trial_index = int(item.get("trial_index", -1))
            frozen = registered.get(trial_index)
            if (
                frozen is None
                or item.get("parameters") != dict(frozen.parameters_json or {})
            ):
                raise ValueError("fin_strategy result changed a preregistered trial")
            if item.get("status") != "succeeded":
                continue
            metrics = item.get("metrics")
            if not isinstance(metrics, Mapping):
                raise ValueError("fin_strategy succeeded trial has no metrics")
            expected_config_sha256 = _canonical_sha256(
                dict(frozen.config_json or {})
            )
            for segment in ("in_sample", "out_of_sample"):
                segment_metrics = metrics.get(segment)
                provenance = (
                    segment_metrics.get("provenance")
                    if isinstance(segment_metrics, Mapping)
                    else None
                )
                if (
                    not isinstance(segment_metrics, Mapping)
                    or segment_metrics.get("evaluation_mode") != mode
                    or segment_metrics.get("evaluation_scope") != "pre_final_only"
                    or segment_metrics.get("final_oos_opened") is not False
                    or segment_metrics.get("capital_eligible") is not False
                    or not isinstance(provenance, Mapping)
                    or provenance.get("evaluation_mode") != mode
                    or provenance.get("evaluation_scope") != "pre_final_only"
                    or provenance.get("final_oos_opened") is not False
                    or provenance.get("strategy_config_sha256")
                    != expected_config_sha256
                    or any(
                        provenance.get(key) != value
                        for key, value in required_identity.items()
                    )
                ):
                    raise ValueError(
                        "fin_strategy result provenance differs from governance"
                    )

    def apply_result(self, experiment_id: str, result: dict[str, Any]) -> None:
        trial_results = result.get("trials")
        summary = result.get("summary")
        if not isinstance(trial_results, list) or not isinstance(summary, dict):
            raise ValueError("parameter experiment result is incomplete")
        now = _now()
        with self.engine.begin() as connection:
            experiment_row = connection.execute(
                select(parameter_experiments).where(
                    parameter_experiments.c.id == experiment_id
                )
            ).first()
            if experiment_row is None:
                raise KeyError(experiment_id)

            periods = dict(experiment_row.periods_json or {})
            governance = periods.get("governance") or {}
            if governance.get("mode") == "model_portfolio_pre_final":
                expected_trial_count = int(
                    governance.get("prior_admitted_trial_count") or 0
                ) + len(trial_results)
                if (
                    result.get("experiment_id") != experiment_id
                    or result.get("strategy_version_id")
                    != str(experiment_row.strategy_version_id)
                    or result.get("dataset") != str(experiment_row.dataset)
                    or result.get("evaluation_mode") != "pre_final_portfolio_trial"
                    or result.get("final_oos_opened") is not False
                    or result.get("periods") != periods
                    or summary.get("final_oos_opened") is not False
                    or int(summary.get("governed_trial_count") or 0)
                    != expected_trial_count
                    or any(
                        (item.get("metrics") or {})
                        and any(
                            (
                                ((item.get("metrics") or {}).get(segment) or {}).get(
                                    "provenance"
                                )
                                or {}
                            ).get("evaluation_scope")
                            != "pre_final_only"
                            or (
                                ((item.get("metrics") or {}).get(segment) or {}).get(
                                    "provenance"
                                )
                                or {}
                            ).get("final_oos_opened")
                            is not False
                            for segment in ("in_sample", "out_of_sample")
                        )
                        for item in trial_results
                        if item.get("status") == "succeeded"
                    )
                ):
                    raise ValueError(
                        "model portfolio experiment result is not pre-final evidence"
                    )
            expected = connection.execute(
                select(
                    parameter_experiment_trials.c.trial_index,
                    parameter_experiment_trials.c.parameters_json,
                    parameter_experiment_trials.c.config_json,
                ).where(
                    parameter_experiment_trials.c.experiment_id == experiment_id
                )
            ).all()
            expected_indexes = {int(row.trial_index) for row in expected}
            result_indexes = {int(item["trial_index"]) for item in trial_results}
            if expected_indexes != result_indexes:
                raise ValueError("parameter experiment result does not cover every trial")
            experiment_status, experiment_error = _terminal_result_state(
                result,
                trial_results,
                summary,
            )
            self._validate_strategy_research_result(
                experiment_id=experiment_id,
                experiment_row=experiment_row,
                periods=periods,
                governance=governance,
                result=result,
                trial_results=trial_results,
                summary=summary,
                expected_trials=list(expected),
            )
            if (
                governance.get("mode") == "model_portfolio_pre_final"
                and experiment_status == "succeeded"
            ):
                registered = {int(row.trial_index): row for row in expected}
                combined = [
                    {
                        **item,
                        "parameters": dict(registered[int(item["trial_index"])].parameters_json),
                        "config": dict(registered[int(item["trial_index"])].config_json),
                    }
                    for item in trial_results
                ]
                comparability = portfolio_trial_comparability_evidence(combined)
                if summary.get("comparability") != comparability or any(
                    any(
                        (
                            ((item.get("metrics") or {}).get(segment) or {}).get(
                                "provenance"
                            )
                            or {}
                        ).get(key)
                        != expected_value
                        for key, expected_value in (
                            (
                                "dataset_identity_sha256",
                                governance.get("dataset_identity_sha256"),
                            ),
                            (
                                "pre_final_cutoff",
                                governance.get("pre_final_cutoff"),
                            ),
                            (
                                "model_signal_identity_sha256",
                                governance.get("model_signal_identity_sha256"),
                            ),
                            (
                                "formal_model_admission_binding_sha256",
                                governance.get("formal_admission_binding_sha256"),
                            ),
                        )
                    )
                    for item in combined
                    for segment in ("in_sample", "out_of_sample")
                ):
                    raise ValueError(
                        "model portfolio result changed dataset or admission evidence"
                    )
            for item in trial_results:
                status = str(item.get("status"))
                if status not in {"succeeded", "failed"}:
                    raise ValueError("parameter experiment trial has an invalid terminal status")
                connection.execute(
                    update(parameter_experiment_trials)
                    .where(
                        parameter_experiment_trials.c.experiment_id == experiment_id,
                        parameter_experiment_trials.c.trial_index == int(item["trial_index"]),
                    )
                    .values(
                        status=status,
                        score=item.get("score"),
                        metrics_json=item.get("metrics"),
                        warnings_json=item.get("warnings", []),
                        error=item.get("error"),
                        started_at=now,
                        finished_at=now,
                    )
                )
            connection.execute(
                update(parameter_experiments)
                .where(parameter_experiments.c.id == experiment_id)
                .values(
                    status=experiment_status,
                    summary_json=summary,
                    error=experiment_error,
                    finished_at=now,
                )
            )

    def frozen_portfolio_config(self, experiment_id: str) -> dict[str, Any]:
        return select_frozen_portfolio_config(self.get(experiment_id))

    def discard(self, experiment_id: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                delete(parameter_experiments).where(parameter_experiments.c.id == experiment_id)
            )

    def get(self, experiment_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(parameter_experiments).where(parameter_experiments.c.id == experiment_id)
            ).first()
            if row is None:
                raise KeyError(experiment_id)
            trials = connection.execute(
                select(parameter_experiment_trials)
                .where(parameter_experiment_trials.c.experiment_id == experiment_id)
                .order_by(parameter_experiment_trials.c.trial_index)
            ).all()
        result = self._decode_experiment(row_dict(row))
        result["trials"] = [self._decode_trial(row_dict(item)) for item in trials]
        result["trial_count"] = len(result["trials"])
        progress_path = Path(result["artifact_path"]) / "progress.json"
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            result["progress"] = progress if isinstance(progress, dict) else None
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            result["progress"] = None
        return result

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        statement = (
            select(parameter_experiments)
            .order_by(parameter_experiments.c.created_at.desc())
            .limit(limit)
        )
        with self.engine.connect() as connection:
            rows = [self._decode_experiment(row_dict(row)) for row in connection.execute(statement)]
        return rows

    def latest_for_version(
        self, strategy_version_id: str, *, created_by: str | None = None
    ) -> dict[str, Any] | None:
        statement = select(parameter_experiments.c.id).where(
            parameter_experiments.c.strategy_version_id == strategy_version_id
        )
        if created_by is not None:
            statement = statement.where(parameter_experiments.c.created_by == created_by)
        statement = statement.order_by(parameter_experiments.c.created_at.desc()).limit(1)
        with self.engine.connect() as connection:
            experiment_id = connection.scalar(statement)
        return self.get(str(experiment_id)) if experiment_id else None

    @staticmethod
    def _decode_experiment(row: dict[str, Any]) -> dict[str, Any]:
        row["periods"] = row.pop("periods_json")
        row["parameter_grid"] = row.pop("parameter_grid_json")
        row["baseline_config"] = row.pop("baseline_config_json")
        row["summary"] = row.pop("summary_json")
        return row

    @staticmethod
    def _decode_trial(row: dict[str, Any]) -> dict[str, Any]:
        row["parameters"] = row.pop("parameters_json")
        row["config"] = row.pop("config_json")
        row["metrics"] = row.pop("metrics_json")
        row["warnings"] = row.pop("warnings_json")
        return row
