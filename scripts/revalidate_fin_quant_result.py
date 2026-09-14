"""Revalidate an unchanged sealed fin_quant export after an input-history repair.

Preserve the settled failed trial and register a successor research screening
family. This command imports existing research; it does not execute new RD-Agent
rounds or inherit any candidate admission, statistical result, or OOS access.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
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
    research_tournament_trials,
    research_tournaments,
    row_dict,
    schedules,
    strategy_versions,
)
from quant_platform.autopilot import AutopilotStore
from quant_platform.fin_quant_handoff_recovery import (
    EVIDENCE_KEYS,
    RECOVERY_KEY,
    FinQuantHandoffRecovery,
    _has_opened_oos,
    _JoinedEngine,
    require,
)
from quant_platform.job_store import JobStore
from quant_platform.research_tournament import canonical_sha256
from quant_platform.worker import LocalJobWorker

KEY = "fin_quant_history_revalidation"
VERSION = "fin-quant-history-revalidation-v1"
CONSUMERS = ("scripts/evaluate_quant_bundle.py", "src/quant_platform/factor_recompute.py")


def load_export(root, artifacts):
    path = Path(__file__).with_name("recover_fin_quant_result.py")
    spec = importlib.util.spec_from_file_location("sealed_fin_quant_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_sealed_result(root, artifacts)


def verify_preflight(proof, payload, source_root):
    require(proof.get("status") == "passed", "factor preflight did not pass")
    expected = {f["candidate_id"]: f for b in payload["candidates"] for f in b["factors"]}
    observed = {f["candidate_id"]: f for f in proof["factors"]}
    require(len(observed) == len(proof["factors"]) and observed.keys() == expected.keys(),
            "preflight factor ownership differs")
    for factor_id, item in observed.items():
        source = expected[factor_id]
        comparison = item["submitted_comparison"]
        pit = item["pit_invariance"]
        require(item["code_sha256"] == source["code_sha256"]
                and comparison.get("exact_match") is True
                and comparison.get("index_exact_match") is True
                and comparison.get("warmup_prefix_rows") == 0
                and pit.get("status") == "passed" and pit.get("cutpoint_count", 0) >= 3,
                "factor preflight lacks exact-domain or PIT evidence")
        with Path(source["submitted_values_path"]).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        require(comparison.get("submitted_sha256") == digest, "preflight values changed")
    require(proof.get("consumer_sources") == {
        name: hashlib.sha256((source_root / name).read_bytes().replace(b"\r\n", b"\n"))
        .hexdigest() for name in CONSUMERS
    }, "preflight used another evaluator implementation")


def read_source(connection, settings, cycle_id):
    store = object.__new__(AutopilotStore)
    store.engine = _JoinedEngine(connection)
    cycle = store.get_cycle(cycle_id)
    require(cycle["status"] == "succeeded" and cycle["stage"] == "research_complete"
            and cycle["finished_at"] and len(cycle["branches"]) == 1,
            "source research is not terminal")
    branch = cycle["branches"][0]
    require(branch["scenario"] == "fin_quant" and branch["scope_key"] == "joint"
            and branch["status"] == "succeeded", "source branch differs")
    run = row_dict(connection.execute(select(research_runs).where(
        research_runs.c.id == branch["research_run_id"])).one())
    generation = row_dict(connection.execute(select(jobs).where(
        jobs.c.id == branch["job_id"])).one())
    evaluation = row_dict(connection.execute(select(jobs).where(
        jobs.c.id == run["job_id"])).one())
    require(run["status"] == evaluation["status"] == "succeeded" and run["finished_at"]
            and evaluation["finished_at"] and evaluation["kind"] == "quant_bundle_evaluate"
            and evaluation["payload_json"]["research_run_id"] == run["id"]
            and run["config_json"]["autopilot_cycle_id"] == cycle_id
            and generation["payload_json"]["research_run_id"] == run["id"],
            "source execution ownership differs")
    outcomes = evaluation["progress_json"].get("evaluations") or []
    require(outcomes and all(item["status"] == "failed" and re.fullmatch(
        r"quant factor [0-9a-f]+ submitted values do not recompute", item.get("error", ""))
        for item in outcomes), "source failure is not the input-history defect")
    tid = evaluation["payload_json"]["research_tournament_id"]
    tournament = row_dict(connection.execute(select(research_tournaments).where(
        research_tournaments.c.id == tid)).one())
    trials = [row_dict(r) for r in connection.execute(select(research_tournament_trials).where(
        research_tournament_trials.c.tournament_id == tid)
        .order_by(research_tournament_trials.c.id))]
    require(tournament["cycle_id"] == cycle_id and tournament["stage"] == "quant"
            and tournament["finished_at"] and len(trials) == len(outcomes)
            and all(t["status"] == "failed" and not t["metrics_json"] for t in trials)
            and {t["candidate_id"] for t in trials} == {e["candidate_id"] for e in outcomes},
            "settled source screening ledger differs")
    require(not any(_has_opened_oos(x) for x in (
        cycle["state"], run["config_json"], evaluation["progress_json"])), "source opened OOS")
    for table in (strategy_versions, schedules):
        column = table.c.config_json if table is strategy_versions else table.c.payload_json
        require(connection.scalar(select(table.c.id).where(
            cast(column, Text).contains(cycle_id) | cast(column, Text).contains(run["id"])
        ).limit(1)) is None, "source already dispatched downstream strategy work")
    lineage = cycle["state"][RECOVERY_KEY]
    historical = replace(settings,
        quantlab_release_id=lineage["target_release"]["release_id"],
        quantlab_config_digest=lineage["target_release"]["config_digest"])
    evidence = FinQuantHandoffRecovery(SimpleNamespace(settings=historical)).verify(
        cycle, connection=connection)
    artifacts = [row_dict(r) for r in connection.execute(select(research_run_artifacts).where(
        research_run_artifacts.c.research_run_id == run["id"])
        .order_by(research_run_artifacts.c.id))]
    result, export_proof = load_export(settings.data_root / "artifacts/rdagent" / run["id"],
                                      artifacts)
    candidates = {}
    for table in (factor_candidates, model_candidates, quant_bundle_candidates):
        candidates[table.name] = [row_dict(r) for r in connection.execute(select(table).where(
            table.c.research_run_id == run["id"]).order_by(table.c.id))]
    frozen = {"cycle": cycle, "run": run, "generation": generation, "evaluation": evaluation,
              "tournament": tournament, "trials": trials, "artifacts": artifacts,
              "candidates": candidates, "export_proof": export_proof}
    return json.loads(json.dumps(frozen, default=str)), result, evidence


def plan_revalidation(connection, settings, cycle_id, event_key, actor, reason, proof_path):
    source, _, _ = read_source(connection, settings, cycle_id)
    require(actor.strip() and reason.strip() and event_key.strip(), "operator reason is required")
    require(settings.quantlab_release_id and settings.quantlab_config_digest,
            "accepted release is missing")
    raw = proof_path.read_bytes()
    verify_preflight(json.loads(raw), source["evaluation"]["payload_json"],
                     Path(__file__).resolve().parents[1])
    require(connection.scalar(select(autopilot_cycles.c.id).where(
        autopilot_cycles.c.state_json[KEY]["source_cycle_id"].as_string() == cycle_id
    ).limit(1)) is None, "source already has a revalidation successor")
    return {"contract_version": VERSION, "source_cycle_id": cycle_id,
            "source_run_id": source["run"]["id"], "source_sha256": canonical_sha256(source),
            "event_key": event_key, "actor": actor, "reason": reason,
            "preflight_path": str(proof_path), "preflight_sha256": hashlib.sha256(raw).hexdigest(),
            "target_release": {"release_id": settings.quantlab_release_id,
                               "config_digest": settings.quantlab_config_digest}}


def revalidate(connection, settings, plan, *, expected_sha256):
    require(plan.get("contract_version") == VERSION and canonical_sha256(plan) == expected_sha256,
            "reviewed revalidation plan changed")
    existing = connection.scalar(select(audit_events.c.details_json).where(
        audit_events.c.action == KEY,
        audit_events.c.details_json["plan_sha256"].as_string() == expected_sha256))
    if existing:
        return existing["receipt"]
    source, result, model = read_source(connection, settings, plan["source_cycle_id"])
    AutopilotStore._lock_horizon(connection, source["cycle"]["horizon_profile"])
    for table, clause in (
        (autopilot_cycles, autopilot_cycles.c.id == source["cycle"]["id"]),
        (autopilot_branches, autopilot_branches.c.cycle_id == source["cycle"]["id"]),
        (research_runs, research_runs.c.id == source["run"]["id"]),
        (jobs, jobs.c.id.in_([source["generation"]["id"], source["evaluation"]["id"]])),
        (research_tournaments, research_tournaments.c.id == source["tournament"]["id"]),
        (research_tournament_trials,
         research_tournament_trials.c.tournament_id == source["tournament"]["id"]),
        *((table, table.c.research_run_id == source["run"]["id"]) for table in (
            factor_candidates, model_candidates, quant_bundle_candidates, research_run_artifacts)),
    ):
        connection.execute(select(table.c.id).where(clause).order_by(table.c.id)
                           .with_for_update()).all()
    require(plan_revalidation(connection, settings, plan["source_cycle_id"], plan["event_key"],
            plan["actor"], plan["reason"], Path(plan["preflight_path"])) == plan,
            "source, preflight or accepted release changed")
    worker = LocalJobWorker(JobStore(settings.database_url), Path("/app"), settings,
                            initialize_queue=False)
    for value in vars(worker).values():
        if hasattr(value, "engine"):
            value.engine = _JoinedEngine(connection)
    store = object.__new__(AutopilotStore)
    store.engine = _JoinedEngine(connection)
    event = source["cycle"]["state"]["research_event"]
    request = {**event["request"], "event_key": plan["event_key"],
               "actor": plan["actor"], "reason": plan["reason"]}
    cycle = store.create_research_event(request=request, dataset=model["dataset"],
        config=event["config"], config_revision=event["config_revision"])
    lineage = deepcopy(source["cycle"]["state"][RECOVERY_KEY])
    lineage.update(target_release=plan["target_release"], plan_sha256=expected_sha256,
                   recovery_event_sha256=cycle["state"]["research_event"]["sha256"])
    lineage.pop("sha256")
    lineage["sha256"] = canonical_sha256(lineage)
    provenance = {"source_cycle_id": source["cycle"]["id"], "source_run_id": source["run"]["id"],
                  "source_evaluation_job_id": source["evaluation"]["id"],
                  "source_tournament_id": source["tournament"]["id"],
                  "source_sha256": plan["source_sha256"], "plan_sha256": expected_sha256,
                  "research_reexecuted": False, "old_failed_trials_retained": True}
    state = {**cycle["state"], **{key: deepcopy(source["cycle"]["state"][key])
             for key in EVIDENCE_KEYS if key in source["cycle"]["state"]},
             RECOVERY_KEY: lineage, KEY: provenance,
             "research_tournament_id": model["tournament"]["id"],
             "active_research_tournament_id": model["tournament"]["id"],
             "fin_quant_status": "imported_result_awaiting_validation", "final_oos_opened": False}
    cycle = store.set_cycle_state(cycle["id"], state=state, stage="joint_optimization")
    config = {**source["run"]["config_json"], "autopilot_cycle_id": cycle["id"], KEY: provenance}
    run = worker.research.create_run(kind=source["run"]["kind"],
        objective=source["run"]["objective"], dataset=source["run"]["dataset"],
        requested_by=plan["actor"], config=config,
        artifact_path=settings.data_root / "artifacts/rdagent",
        budget={"loop_n": 0, "research_reexecuted": False,
                "source_budget": source["run"]["budget_json"]})
    payload = {**source["generation"]["payload_json"], "research_run_id": run["id"],
               "autopilot_cycle_id": cycle["id"], KEY: provenance}
    # This operator command performs the import inside this transaction. Its job
    # is completed before commit, so no generic worker can claim it as RD research.
    imported = worker.store.create("quant_result_import", payload,
        settings.data_root / "platform/logs" / f"quant-result-import-{run['id']}.log",
        idempotency_key=KEY + ":" + expected_sha256, max_attempts=1)
    now = datetime.now(UTC)
    connection.execute(update(jobs).where(jobs.c.id == imported["id"]).values(
        status="running", started_at=now, attempts=1))
    worker.research.attach_job(run["id"], imported["id"])
    store.create_branch(cycle_id=cycle["id"], scenario="fin_quant", scope_key="joint",
        research_run_id=run["id"], job_id=imported["id"], details={KEY: provenance})
    worker.rdagent_candidates.register_run_artifact(research_run_id=run["id"],
        artifact_type="imported_fin_quant_result",
        storage_path=settings.data_root / "artifacts/rdagent" / source["run"]["id"] / "result.json",
        producer="operator_sealed_result_import", actor=plan["actor"],
        contract_version=VERSION, metadata=provenance)
    count = worker._queue_quant_bundle_evaluation(worker.store.get(imported["id"]), result)
    worker.store.finish(imported["id"], exit_code=0, result={**provenance, "bundles": count})
    worker.research.mark_run(run["id"], "evaluating", runtime={
        "scenario": "fin_quant", "rounds": 0, "source_rounds": result.get("rounds", 0),
        "quant_bundles": count, "fin_quant_outcome": result["fin_quant_outcome"], KEY: provenance})
    updated = worker.research.get_run(run["id"])
    receipt = {"cycle_id": cycle["id"], "run_id": run["id"], "import_job_id": imported["id"],
               "evaluation_job_id": updated["job_id"], "bundles": count,
               "status": "evaluating", **provenance}
    after, _, _ = read_source(connection, settings, source["cycle"]["id"])
    require(canonical_sha256(after) == plan["source_sha256"], "original settled evidence changed")
    connection.execute(insert(audit_events).values(user_id=None, username=plan["actor"],
        action=KEY, method="SERVICE", path="scripts/revalidate_fin_quant_result.py",
        status_code=202,
        created_at=now, details_json={"plan": plan, "plan_sha256": expected_sha256,
                                     "receipt": receipt}))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle")
    parser.add_argument("--event-key")
    parser.add_argument("--actor", default="codex-operator")
    parser.add_argument("--reason")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Exercise import then roll back")
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
                output = revalidate(connection, settings, plan,
                                    expected_sha256=canonical_sha256(plan))
            else:
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                output = plan_revalidation(connection, settings, args.cycle, args.event_key,
                                          args.actor, args.reason, args.preflight)
            transaction.commit() if args.execute else transaction.rollback()
            print(json.dumps(output, default=str, ensure_ascii=False, indent=2))
        except BaseException:
            transaction.rollback()
            raise


if __name__ == "__main__":
    main()
