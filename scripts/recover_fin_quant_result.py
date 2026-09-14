"""Audit and complete a failed fin_quant result handoff without rerunning research.

Only the candidate_id/id consumer defect is supported. The failed generation job
and its sealed outputs remain unchanged. All database handoff writes share one
transaction; ordinary independent validation retains every admission gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import Text, cast, insert, select, update

from quant_data.config import Settings
from quant_data.database import (
    audit_events,
    autopilot_branches,
    autopilot_cycles,
    factor_candidates,
    jobs,
    model_candidates,
    open_database,
    quant_bundle_candidates,
    research_run_artifacts,
    research_runs,
    research_tournaments,
    row_dict,
)
from quant_platform.autopilot import AutopilotStore
from quant_platform.fin_quant_handoff_recovery import (
    RECOVERY_KEY,
    FinQuantHandoffRecovery,
    _has_opened_oos,
    _JoinedEngine,
    require,
)
from quant_platform.job_store import JobStore
from quant_platform.rdagent_runtime import require_matching_rdagent_runtime_identity
from quant_platform.research_tournament import canonical_sha256
from quant_platform.worker import LocalJobWorker, _validate_fin_quant_research_result

KEY = "fin_quant_result_handoff_recovery"
VERSION = "fin-quant-result-handoff-recovery-v1"


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_sealed_result(root: Path, artifacts: list[dict]) -> tuple[dict, dict]:
    """Validate the original worker inventory and every input to the next gate."""
    inventory_rows = [a for a in artifacts if a["artifact_type"] == "fin_quant_trace_inventory"]
    require(len(inventory_rows) == 1, "exactly one sealed trace inventory is required")
    artifact = inventory_rows[0]
    inventory_path = root / "trace-inventory.json"
    require(Path(artifact["storage_path"]) == inventory_path
            and artifact["status"] == "recorded" and not inventory_path.is_symlink()
            and file_hash(inventory_path) == artifact["content_sha256"],
            "original trace inventory changed")
    inventory = json.loads(inventory_path.read_bytes())
    entries = {item["relative_path"]: item for item in inventory["files"]}
    require(len(entries) == len(inventory["files"]), "duplicate inventory paths")
    proofs = {}

    def verify(path: Path):
        require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
                "result input escaped the original run")
        relative = path.relative_to(root).as_posix()
        entry = entries.get(relative) or {}
        digest = file_hash(path)
        require(entry.get("sha256") == digest and entry.get("size_bytes") == path.stat().st_size,
                "sealed research input changed: " + relative)
        proofs[relative] = digest

    result_path = root / "result.json"
    verify(result_path)
    result = json.loads(result_path.read_bytes())
    require(result.get("status") == "ok" and result.get("scenario") == "fin_quant",
            "research process did not export a successful fin_quant result")
    _validate_fin_quant_research_result(result)
    require(result.get("quant_bundles"), "result contains no joint proposals")
    for bundle in result["quant_bundles"]:
        for component in [bundle["model"], *bundle["factors"]]:
            code_path = Path(component["code_path"])
            verify(code_path)
            require(proofs[code_path.relative_to(root).as_posix()] == component["code_sha256"],
                    "result code differs from its inventory")
            if component.get("submitted_values_path"):
                verify(Path(component["submitted_values_path"]))
        require(all(f.get("candidate_id") for f in bundle["factors"]),
                "exported factor identity is missing")
    return result, {"inventory_artifact_id": artifact["id"],
                    "inventory_sha256": artifact["content_sha256"], "files": proofs}


def snapshot(connection, settings, cycle_id):
    store = object.__new__(AutopilotStore)
    store.engine = _JoinedEngine(connection)
    cycle = store.get_cycle(cycle_id)
    require(cycle["status"] == "blocked" and cycle["stage"] == "joint_optimization_blocked"
            and cycle["finished_at"] and KEY not in cycle["state"],
            "activity is not an unrecovered failed handoff")
    require(len(cycle["branches"]) == 1, "activity has another branch")
    branch = cycle["branches"][0]
    require(branch["scenario"] == "fin_quant" and branch["scope_key"] == "joint"
            and branch["status"] == "failed", "branch is not a failed joint research owner")
    job = row_dict(connection.execute(select(jobs).where(jobs.c.id == branch["job_id"])).one())
    run = row_dict(connection.execute(select(research_runs).where(
        research_runs.c.id == branch["research_run_id"])).one())
    require(job["kind"] == "rdagent_quant" and job["status"] == run["status"] == "failed"
            and job["error"] == run["error"] == "'id'" and job["finished_at"]
            and run["finished_at"] and job["id"] == run["job_id"]
            and job["payload_json"]["research_run_id"] == run["id"]
            and run["config_json"]["autopilot_cycle_id"] == cycle_id,
            "failure is not the supported result consumer defect")
    require(not _has_opened_oos(cycle["state"]) and not _has_opened_oos(run["config_json"]),
            "activity already opened OOS")
    for table in (factor_candidates, model_candidates, quant_bundle_candidates):
        require(connection.scalar(select(table.c.id).where(
            table.c.research_run_id == run["id"]).limit(1)) is None,
            "candidate registration already started")
    require(connection.scalar(select(research_tournaments.c.id).where(
        research_tournaments.c.cycle_id == cycle_id,
        research_tournaments.c.stage == "quant").limit(1)) is None,
        "independent validation already registered")
    require(connection.scalar(select(autopilot_cycles.c.id).where(
        autopilot_cycles.c.id != cycle_id,
        cast(autopilot_cycles.c.state_json, Text).contains(cycle_id)).limit(1)) is None,
        "activity already has a successor")
    lineage = cycle["state"][RECOVERY_KEY]
    historical_settings = replace(settings,
        quantlab_release_id=lineage["target_release"]["release_id"],
        quantlab_config_digest=lineage["target_release"]["config_digest"])
    FinQuantHandoffRecovery(SimpleNamespace(settings=historical_settings)).verify(
        cycle, connection=connection)
    artifacts = [row_dict(r) for r in connection.execute(
        select(research_run_artifacts).where(research_run_artifacts.c.research_run_id == run["id"])
        .order_by(research_run_artifacts.c.id))]
    root = settings.data_root / "artifacts" / "rdagent" / run["id"]
    require(not root.is_symlink(), "research root is a symlink")
    result, proof = load_sealed_result(root, artifacts)
    require_matching_rdagent_runtime_identity(
        job["payload_json"]["expected_rdagent_runtime"], result["rdagent_runtime"])
    return cycle, job, run, artifacts, result, {
        "cycle_sha256": canonical_sha256(cycle), "job_sha256": canonical_sha256(job),
        "run_sha256": canonical_sha256(run), "artifacts_sha256": canonical_sha256(artifacts),
        "output_proof": proof,
    }


def plan_recovery(connection, settings, cycle_id, actor, reason):
    cycle, job, run, _, _, proof = snapshot(connection, settings, cycle_id)
    require(actor.strip() and reason.strip(), "actor and repair reason are required")
    require(settings.quantlab_release_id and settings.quantlab_config_digest,
            "accepted target release is missing")
    return {"contract_version": VERSION, "cycle_id": cycle_id, "run_id": run["id"],
            "horizon_profile": cycle["horizon_profile"],
            "job_id": job["id"], "actor": actor, "reason": reason,
            "source_release": cycle["state"][RECOVERY_KEY]["target_release"],
            "target_release": {"release_id": settings.quantlab_release_id,
                               "config_digest": settings.quantlab_config_digest}, **proof}


def recover(connection, settings, plan, *, expected_sha256):
    require(plan.get("contract_version") == VERSION and canonical_sha256(plan) == expected_sha256,
            "reviewed result recovery plan changed")
    existing = connection.execute(select(audit_events.c.details_json).where(
        audit_events.c.action == KEY,
        audit_events.c.details_json["plan_sha256"].as_string() == expected_sha256)).scalar()
    if existing:
        return existing["receipt"]
    AutopilotStore._lock_horizon(connection, plan["horizon_profile"])
    for table, clause in (
        (autopilot_cycles, autopilot_cycles.c.id == plan["cycle_id"]),
        (autopilot_branches, autopilot_branches.c.cycle_id == plan["cycle_id"]),
        (research_runs, research_runs.c.id == plan["run_id"]),
        (jobs, jobs.c.id == plan["job_id"]),
    ):
        connection.execute(select(table.c.id).where(clause).with_for_update()).all()
    require(plan_recovery(connection, settings, plan["cycle_id"], plan["actor"], plan["reason"])
            == plan, "research evidence or target release changed after planning")
    cycle, job, run, artifacts, result, _ = snapshot(connection, settings, plan["cycle_id"])
    now = datetime.now(UTC)
    state = deepcopy(cycle["state"])
    previous_lineage = deepcopy(state[RECOVERY_KEY])
    state[RECOVERY_KEY]["target_release"] = plan["target_release"]
    state[RECOVERY_KEY].pop("sha256")
    state[RECOVERY_KEY]["sha256"] = canonical_sha256(state[RECOVERY_KEY])
    state[KEY] = {"plan_sha256": expected_sha256, "source_release": plan["source_release"],
                  "target_release": plan["target_release"], "source_job_id": job["id"],
                  "result_sha256": plan["output_proof"]["files"]["result.json"],
                  "research_reexecuted": False, "original_lineage": previous_lineage}
    connection.execute(update(autopilot_cycles).where(autopilot_cycles.c.id == cycle["id"]).values(
        status="active", stage="joint_optimization", state_json=state,
        error=None, finished_at=None, updated_at=now))
    connection.execute(update(autopilot_branches).where(
        autopilot_branches.c.id == cycle["branches"][0]["id"]).values(
            status="evaluating", error=None, finished_at=None, updated_at=now))
    worker = LocalJobWorker(JobStore(settings.database_url), Path("/app"), settings,
                            initialize_queue=False)
    for value in vars(worker).values():
        if hasattr(value, "engine"):
            value.engine = _JoinedEngine(connection)
    worker.research.requeue_run(run["id"], actor=plan["actor"])
    # Only the result delivery is retried; retain when the original research began.
    connection.execute(update(research_runs).where(research_runs.c.id == run["id"]).values(
        started_at=run["started_at"]))
    decoded_job = worker.store.get(job["id"])
    count = worker._queue_quant_bundle_evaluation(decoded_job, result)
    runtime = {"scenario": "fin_quant", "rdagent_runtime": result["rdagent_runtime"],
               "trace_path": result.get("trace_path"), "trace_summary": result.get("trace_summary"),
               "rounds": result.get("rounds", 0), "quant_bundles": count,
               "fin_quant_outcome": result["fin_quant_outcome"], KEY: state[KEY]}
    for artifact in artifacts:
        if artifact["artifact_type"] == "fin_quant_sanitized_result":
            runtime["sanitized_result_artifact_id"] = artifact["id"]
            runtime["sanitized_result_sha256"] = artifact["content_sha256"]
        elif artifact["artifact_type"] == "fin_quant_trace_inventory":
            runtime["trace_inventory_artifact_id"] = artifact["id"]
    worker.research.mark_run(run["id"], "evaluating", runtime=runtime)
    restored = worker.research.get_run(run["id"])
    receipt = {"cycle_id": cycle["id"], "run_id": run["id"],
               "evaluation_job_id": restored["job_id"], "bundles": count,
               "status": "evaluating", "research_reexecuted": False}
    require(canonical_sha256(row_dict(connection.execute(select(jobs).where(
        jobs.c.id == job["id"])).one())) == plan["job_sha256"],
        "original failed generation job was changed")
    connection.execute(insert(audit_events).values(
        user_id=None, username=plan["actor"], action=KEY, method="SERVICE",
        path="scripts/recover_fin_quant_result.py", status_code=202, created_at=now,
        details_json=json.loads(json.dumps({
            "plan": plan, "plan_sha256": expected_sha256, "receipt": receipt,
            "original_cycle": cycle, "original_run": run, "original_job": job}, default=str))))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle")
    parser.add_argument("--actor", default="codex-operator")
    parser.add_argument("--reason")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Exercise the handoff, then roll back")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    with open_database(settings.database_url).connect() as connection:
        transaction = connection.begin()
        try:
            if args.check or args.execute:
                require(args.plan is not None and args.plan_sha256, "reviewed plan is required")
                raw = args.plan.read_bytes()
                require(hashlib.sha256(raw).hexdigest() == args.plan_sha256, "plan file changed")
                plan = json.loads(raw)
                output = recover(connection, settings, plan,
                                 expected_sha256=canonical_sha256(plan))
            else:
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                output = plan_recovery(connection, settings, args.cycle, args.actor, args.reason)
            if args.execute:
                transaction.commit()
            else:
                transaction.rollback()
            print(json.dumps(output, default=str, ensure_ascii=False, indent=2))
        except BaseException:
            transaction.rollback()
            raise


if __name__ == "__main__":
    main()
