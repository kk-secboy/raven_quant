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
    row_dict,
    strategy_factors,
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
from .strategy_store import StrategyStore


def _now() -> datetime:
    return datetime.now(UTC)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        job_payload = {
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
        }
        return {
            "experiment": experiment,
            "job_payload": job_payload,
            "created": created,
            "needs_job": created or (
                experiment.get("status") == "queued" and not experiment.get("job_id")
            ),
        }

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
            if governance.get("mode") == "model_portfolio_pre_final":
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
                    status="succeeded",
                    summary_json=summary,
                    error=None,
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
