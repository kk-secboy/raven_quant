"""Bounded read-only counters for an actively owned parameter experiment.

The progress file is a checkpoint, not a heartbeat or a settlement result.
Only its four counters are published; scores, warnings and errors stay private.
"""

from __future__ import annotations

import json
import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from quant_data.database import jobs, parameter_experiment_trials, parameter_experiments

COUNTERS = ("completed_count", "succeeded_count", "failed_count", "trial_count")
_MAX_JSON_BYTES = 2 * 1024 * 1024
_IDENTITY_FIELDS = {
    "dataset_identity_sha256": "dataset_identity_sha256",
    "stage": "strategy_competition_stage",
    "plan_sha256": "strategy_competition_plan_sha256",
    "mode": "strategy_evaluation_mode",
}


def unavailable_progress() -> dict[str, Any]:
    return {"state": "unavailable", **dict.fromkeys(COUNTERS), "updated_at": None}


def parameter_progress_statement(job_ids: list[str]):
    experiment = parameter_experiments.c
    payload = jobs.c.payload_json
    governance = experiment.periods_json["governance"]
    trial_count = (
        select(func.count()).select_from(parameter_experiment_trials)
        .where(parameter_experiment_trials.c.experiment_id == experiment.id)
        .correlate(parameter_experiments).scalar_subquery()
    )
    return select(
        jobs.c.id.label("job_id"), jobs.c.status.label("job_status"),
        jobs.c.attempts, jobs.c.started_at.label("job_started_at"),
        experiment.id.label("experiment_id"), experiment.status.label("experiment_status"),
        experiment.strategy_version_id, experiment.dataset, experiment.artifact_path,
        trial_count.label("trial_count"),
        payload["strategy_version_id"].as_string().label("payload_strategy_version_id"),
        payload["dataset"].as_string().label("payload_dataset"),
        *(payload[value].as_string().label(f"payload_{key}")
          for key, value in _IDENTITY_FIELDS.items()),
        *(governance[key].as_string().label(f"governance_{key}") for key in _IDENTITY_FIELDS),
    ).select_from(jobs.join(parameter_experiments, (
        (experiment.job_id == jobs.c.id)
        & (experiment.id == payload["parameter_experiment_id"].as_string())
    ))).where(jobs.c.id.in_(job_ids), jobs.c.kind == "parameter_experiment")


def _stamp(value: Any) -> datetime:
    result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if result.tzinfo is None:
        raise ValueError("progress timestamps require a timezone")
    return result.astimezone(UTC)


def _signature(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _read_json(path: Path) -> tuple[dict[str, Any], os.stat_result]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_JSON_BYTES:
        raise ValueError("progress metadata is not a bounded regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        if _signature(os.fstat(handle.fileno())) != _signature(before):
            raise ValueError("progress metadata changed before reading")
        raw = handle.read(_MAX_JSON_BYTES + 1)
        after = os.fstat(handle.fileno())
    if (
        len(raw) != before.st_size
        or _signature(before) != _signature(after)
        or _signature(before) != _signature(path.lstat())
    ):
        raise ValueError("progress metadata changed while reading")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("progress metadata requires an object")
    return result, before


class ParameterProgressReader:
    def __init__(self, data_root: Path):
        self.data_root = data_root.resolve()

    def read(self, job: dict[str, Any], binding: dict[str, Any] | None) -> dict[str, Any]:
        try:
            return self._read(job, binding)
        except (KeyError, ValueError, TypeError, OSError, OverflowError, RecursionError):
            return unavailable_progress()

    def _read(self, job: dict[str, Any], binding: dict[str, Any] | None) -> dict[str, Any]:
        if (
            not binding or job.get("kind") != "parameter_experiment"
            or job.get("status") != "running" or binding["job_status"] != "running"
            or binding["experiment_status"] != "running" or job["id"] != binding["job_id"]
            or job.get("attempts") != binding["attempts"] or binding["attempts"] < 1
            or binding["strategy_version_id"] != binding["payload_strategy_version_id"]
            or binding["dataset"] != binding["payload_dataset"]
        ):
            raise ValueError("progress owner is not the displayed active attempt")
        experiment_id = str(binding["experiment_id"])
        if not re.fullmatch(r"[0-9a-f]{32}", experiment_id):
            raise ValueError("invalid progress owner identifier")
        allowed = (self.data_root / "artifacts" / "parameter-experiments").resolve()
        expected = allowed / experiment_id
        if (
            not allowed.is_relative_to(self.data_root)
            or Path(binding["artifact_path"]).resolve() != expected
            or expected.resolve() != expected
        ):
            raise ValueError("progress metadata is outside its exact artifact owner")
        manifest_path = expected / "manifest.json"
        manifest, manifest_stat = _read_json(manifest_path)
        started_at = _stamp(binding["job_started_at"]).timestamp()
        # Public jobs may truncate timestamps to seconds; the DB timestamp remains authoritative.
        if abs(_stamp(job["started_at"]).timestamp() - started_at) >= 1:
            raise ValueError("displayed job belongs to another attempt")
        if manifest_stat.st_mtime < started_at:
            raise ValueError("manifest predates the active attempt")
        for key in ("experiment_id", "strategy_version_id", "dataset"):
            if manifest.get(key) != binding[key]:
                raise ValueError("manifest owner differs from its database binding")
        periods = manifest.get("periods")
        if not isinstance(periods, dict) or not isinstance(periods.get("governance"), dict):
            raise ValueError("manifest provenance is unavailable")
        governance = periods["governance"]
        if binding["payload_mode"] is not None and any(
            not binding[f"payload_{key}"] for key in _IDENTITY_FIELDS
        ):
            raise ValueError("governed progress provenance is incomplete")
        for key in _IDENTITY_FIELDS:
            value = binding[f"payload_{key}"]
            if value is not None and (
                value != binding[f"governance_{key}"] or value != governance.get(key)
            ):
                raise ValueError("manifest provenance differs from its active job")
        if binding["payload_mode"] is not None and manifest.get("evaluation_mode") != binding[
            "payload_mode"
        ]:
            raise ValueError("manifest evaluation mode differs from its active job")
        document, progress_stat = _read_json(expected / "progress.json")
        if (
            progress_stat.st_mtime_ns < manifest_stat.st_mtime_ns
            or progress_stat.st_mtime > datetime.now(UTC).timestamp() + 5
            or _signature(manifest_path.lstat()) != _signature(manifest_stat)
        ):
            raise ValueError("progress file does not belong to the current manifest")
        counts = {key: document.get(key) for key in COUNTERS}
        if (
            any(type(value) is not int or value < 0 for value in counts.values())
            or not 0 < counts["trial_count"] == binding["trial_count"] <= 10000
            or counts["completed_count"] > counts["trial_count"]
            or counts["completed_count"] != counts["succeeded_count"] + counts["failed_count"]
        ):
            raise ValueError("progress counters do not match registered trials")
        return {
            "state": "available", **counts,
            "updated_at": datetime.fromtimestamp(progress_stat.st_mtime, UTC).isoformat(),
        }


def attach_parameter_progress(connection, records: list[dict[str, Any]], data_root: Path):
    """One narrow DB read and at most two small files per active experiment."""
    active = [row for row in records if row.get("kind") == "parameter_experiment"
              and row.get("status") == "running"]
    if not active:
        return records
    statement = parameter_progress_statement(sorted({row["id"] for row in active}))
    bindings = {
        row.job_id: dict(row._mapping)
        for row in connection.execute(statement)
    }
    reader = ParameterProgressReader(data_root)
    unique = {row["id"]: row for row in active}
    projected = {job_id: reader.read(row, bindings.get(job_id)) for job_id, row in unique.items()}
    return [{**row, "_parameter_progress": projected[row["id"]]} if row["id"] in projected else row
            for row in records]
