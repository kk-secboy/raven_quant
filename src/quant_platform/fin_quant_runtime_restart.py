"""Bounded runtime restarts of unexported joint research, keeping sealed models."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sqlalchemy import Text, cast, or_, select

from quant_data.database import (
    factor_candidates,
    jobs,
    model_candidates,
    quant_bundle_candidates,
    research_run_artifacts,
    research_runs,
    research_tournaments,
    row_dict,
    schedules,
    strategy_versions,
)

from .fin_quant_handoff_recovery import (
    RECOVERY_KEY,
    RECOVERY_VERSION,
    _has_opened_oos,
    require,
    require_recovery_evidence,
    source_identity,
)
from .research_tournament import canonical_sha256

RUNTIME_RESTART_VERSION = "fin-quant-runtime-restart-v1"


def remaining_restart_attempts(cycle: dict[str, Any]) -> int | None:
    lineage = (cycle.get("state") or {}).get(RECOVERY_KEY) or {}
    if lineage.get("contract_version") != RUNTIME_RESTART_VERSION:
        return None
    restart = lineage["runtime_restart"]
    remaining = restart["max_attempts"] - restart["consumed_attempts"]
    require(0 < remaining == restart["remaining_attempts"], "restart attempt budget changed")
    return remaining


def read_runtime_predecessor(service: Any, connection: Any, cycle_id: str,
                             source: dict[str, Any]) -> dict[str, Any]:
    from .autopilot import AutopilotStore
    from .fin_quant_handoff_recovery import _JoinedEngine

    store = object.__new__(AutopilotStore)
    store.engine = _JoinedEngine(connection)
    prior = store.get_cycle(cycle_id)
    lineage = prior["state"].get(RECOVERY_KEY) or {}
    require(lineage.get("contract_version") == RECOVERY_VERSION,
            "only the original handoff activity may consume a runtime restart")
    # Historical verification uses its recorded release. The new activity is
    # separately pinned to the actual accepted release by ordinary verification.
    target = lineage.get("target_release") or {}
    require_recovery_evidence(prior, source, SimpleNamespace(
        quantlab_release_id=target.get("release_id"),
        quantlab_config_digest=target.get("config_digest"),
    ))
    require(prior["status"] in {"paused", "blocked"}, "predecessor is not stopped")
    branches = prior["branches"]
    require(len(branches) == 1 and branches[0]["scenario"] == "fin_quant"
            and branches[0]["scope_key"] == "joint", "predecessor has other research branches")
    branch = branches[0]
    run_row = connection.execute(select(research_runs).where(
        research_runs.c.id == branch["research_run_id"])).first()
    job_row = connection.execute(select(jobs).where(jobs.c.id == branch["job_id"])).first()
    require(run_row is not None and job_row is not None, "predecessor run/job is missing")
    run, job = row_dict(run_row), row_dict(job_row)
    payload = job["payload_json"]
    require(run["status"] in {"failed", "cancelled"}
            and job["status"] == "cancelled" and job["kind"] == "rdagent_quant"
            and run["finished_at"] and job["finished_at"]
            and run["job_id"] == job["id"] == branch["job_id"]
            and payload["research_run_id"] == run["id"]
            and run["config_json"]["autopilot_cycle_id"] == prior["id"],
            "predecessor cancellation has not settled")
    require(0 < job["attempts"] < job["max_attempts"], "predecessor retry budget exhausted")
    require(payload["expected_rdagent_runtime"] == run["config_json"]["expected_rdagent_runtime"],
            "predecessor runtime ownership differs")
    event = prior["state"]["research_event"]
    require(payload.get("loop_n") == event["request"]["quant_loop_n"]
            and payload.get("duration") == event["request"]["quant_duration"]
            and payload.get("dataset_identity_sha256") == prior["dataset_identity_sha256"]
            and payload.get("research_tournament_id") == source["tournament"]["id"],
            "predecessor frozen budget or dataset differs")
    for key in ("prediction_champion", "prediction_champion_evidence"):
        require(payload.get(key) == run["config_json"].get(key) == prior["state"].get(key),
                "predecessor frozen model evidence differs")
    require(not _has_opened_oos(prior["state"]) and not _has_opened_oos(run["config_json"])
            and not _has_opened_oos(payload), "predecessor has opened OOS")
    for table in (factor_candidates, model_candidates, quant_bundle_candidates,
                  research_run_artifacts):
        require(connection.scalar(select(table.c.id).where(
            table.c.research_run_id == run["id"]).limit(1)) is None,
            "predecessor already exported research outputs")
    require(connection.scalar(select(research_tournaments.c.id).where(
        research_tournaments.c.cycle_id == prior["id"],
        research_tournaments.c.stage == "quant").limit(1)) is None,
        "predecessor already registered independent validation")
    for table, column in ((strategy_versions, strategy_versions.c.config_json),
                          (schedules, schedules.c.payload_json)):
        require(connection.scalar(select(table.c.id).where(or_(
            cast(column, Text).contains(prior["id"]),
            cast(column, Text).contains(run["id"]),
        )).limit(1)) is None, "predecessor has downstream strategy dispatch")
    root = Path(service.controller.settings.data_root).resolve()
    artifacts = root / "artifacts" / "rdagent"
    output = artifacts / run["id"]
    require(Path(run["artifact_path"]).resolve() == artifacts
            and not output.is_symlink() and output.resolve().is_relative_to(artifacts)
            and output.is_dir() and not (output / "result.json").exists(),
            "predecessor output root is missing, escaped or already exported")
    log = Path(job["log_path"])
    require(log.is_file() and not log.is_symlink()
            and log.resolve().is_relative_to(root / "platform" / "logs"),
            "predecessor log is unavailable")
    with log.open("rb") as stream:
        log_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    proof = {
        "cycle_id": prior["id"], "cycle_sha256": canonical_sha256(source_identity(prior)),
        "branches_sha256": canonical_sha256(branches),
        "run_id": run["id"], "run_sha256": canonical_sha256(run),
        "job_id": job["id"], "job_sha256": canonical_sha256(job),
        "consumed_attempts": job["attempts"], "max_attempts": job["max_attempts"],
        "remaining_attempts": job["max_attempts"] - job["attempts"],
        "old_runtime": payload["expected_rdagent_runtime"],
        "old_release": target, "artifact_root": str(output),
        "log_path": str(log), "log_sha256": log_sha,
        "old_artifacts_reused": False, "registered_outputs": 0,
        "final_oos_opened": False,
    }
    return {"cycle": prior, "proof": proof}


def verify_runtime_predecessor(service: Any, connection: Any, cycle: dict[str, Any],
                               source: dict[str, Any]) -> None:
    lineage = cycle["state"][RECOVERY_KEY]
    frozen = lineage["runtime_restart"]
    current = read_runtime_predecessor(service, connection, frozen["cycle_id"], source)
    require(current["cycle"]["finished_at"] is not None, "predecessor still owns the horizon")
    require(current["proof"] == frozen, "runtime predecessor evidence changed")
    remaining_restart_attempts(cycle)
