from __future__ import annotations

import json
import uuid
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


def _now() -> datetime:
    return datetime.now(UTC)


class ParameterExperimentStore:
    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def create(
        self,
        *,
        strategy_version_id: str,
        dataset: str,
        periods: dict[str, dict[str, str]],
        parameter_grid: dict[str, list[int | float]],
        baseline_config: dict[str, Any],
        trials: list[dict[str, Any]],
        artifact_root: Path,
        created_by: str,
    ) -> dict[str, Any]:
        experiment_id = uuid.uuid4().hex
        artifact_path = artifact_root / experiment_id
        now = _now()
        with self.engine.begin() as connection:
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
            if not evidence or any(
                item.dataset != dataset
                or item.is_legacy
                or str(item.evaluator_version) != "factor-gate-v3-hac-bh"
                for item in evidence
            ):
                raise ValueError(
                    "parameter experiments require matching factor evaluation v2 evidence"
                )
            windows = [
                section
                for section in periods.values()
                if isinstance(section, dict) and {"start", "end"}.issubset(section)
            ]
            if len(windows) < 2:
                raise ValueError("parameter experiments require separated research windows")
            experiment_start = min(date.fromisoformat(section["start"]) for section in windows)
            experiment_end = max(date.fromisoformat(section["end"]) for section in windows)
            valid_start = max(
                date.fromisoformat(str(dict(item.metrics_json)["selection_start"]))
                for item in evidence
            )
            valid_end = min(item.valid_end for item in evidence)
            if experiment_start < valid_start or experiment_end > valid_end:
                raise ValueError(
                    "parameter experiments must stay inside the validation selection window"
                )
            connection.execute(
                insert(parameter_experiments).values(
                    id=experiment_id,
                    strategy_version_id=strategy_version_id,
                    dataset=dataset,
                    status="queued",
                    periods_json=periods,
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
            expected = connection.execute(
                select(parameter_experiment_trials.c.trial_index).where(
                    parameter_experiment_trials.c.experiment_id == experiment_id
                )
            ).all()
            expected_indexes = {int(row.trial_index) for row in expected}
            result_indexes = {int(item["trial_index"]) for item in trial_results}
            if expected_indexes != result_indexes:
                raise ValueError("parameter experiment result does not cover every trial")
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
