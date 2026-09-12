"""Audited successor activities for a pre-experiment fin_quant handoff defect.

Completed manual activities and their model evidence are never edited.  This
control-plane entry point appends one activity on the exact same publication;
ordinary workers still validate labels and perform all new research/validation.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from sqlalchemy import Text, cast, insert, or_, select

from quant_data.database import (
    audit_events,
    autopilot_branches,
    autopilot_cycles,
    factor_candidates,
    jobs,
    model_candidates,
    platform_configs,
    quant_bundle_candidates,
    research_run_artifacts,
    research_runs,
    research_tournament_trials,
    research_tournaments,
    row_dict,
    schedules,
    strategy_versions,
)

from .research_tournament import canonical_sha256

RECOVERY_KEY = "fin_quant_handoff_recovery"
RECOVERY_VERSION = "fin-quant-handoff-recovery-v1"
HANDOFF_ERROR = "fin_quant incumbent prediction uses another label horizon"
EVIDENCE_KEYS = (
    "prediction_champion", "prediction_champion_evidence", "model_champion_evidence",
    "horizon_factor_bundle", "horizon_factor_bundle_sha256",
    "horizon_factor_bundle_feature_set_id",
    "horizon_factor_bundle_research_label_binding_sha256",
)


def require(value: Any, message: str) -> None:
    if not value:
        raise ValueError("fin_quant recovery: " + message)


def source_identity(cycle: dict[str, Any]) -> dict[str, Any]:
    """Stable complete source row, excluding branches loaded through another query."""
    return {key: value for key, value in cycle.items() if key != "branches"}


def tournament_identity(tournament: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in tournament.items() if key != "trials"}


def _has_opened_oos(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in {"final_oos_opened", "final_oos_consumed", "capital_eligible"}
            and item is True or _has_opened_oos(item)
            for key, item in value.items()
        )
    return isinstance(value, list) and any(_has_opened_oos(item) for item in value)


def _source_publication(data_root: Path, binding: dict[str, Any]) -> tuple[dict, dict]:
    """Read only the registered publication identity, never a global catalog scan.

    This admission record is not a numeric-data validation receipt. Ordinary
    Autopilot/Worker consumers retain their full sealed-output verification.
    """
    root = data_root / "qlib" / str(binding["name"])
    require(root.resolve() == Path(binding["path"]).resolve()
            and root.resolve().is_relative_to((data_root / "qlib").resolve()),
            "registered publication escaped the Qlib root")
    provenance_path = root / "metadata" / "provenance.json"
    raw = provenance_path.read_bytes()
    provenance = json.loads(raw.decode("utf-8"))
    require(provenance.get("dataset_identity_sha256") == binding["dataset_identity_sha256"]
            and provenance.get("dataset_lineage_id") == binding["dataset_lineage_id"]
            and provenance.get("lineage_verified") is True
            and provenance.get("snapshot_manifest_sha256")
            and (provenance.get("output_manifest") or {}).get("files"),
            "source publication provenance is unsealed or changed")
    calendar_path = root / "calendars" / "day.txt"
    raw_calendar = calendar_path.read_bytes()
    days = raw_calendar.decode("utf-8").splitlines()
    instruments = next((root / "instruments" / name for name in
                        ("cn_all.txt", "liquid_all.txt", "all.txt")
                        if (root / "instruments" / name).is_file()), None)
    require(days and instruments is not None and (root / "features").is_dir(),
            "source publication calendar, universe or features are missing")
    raw_instruments = instruments.read_bytes()
    require(raw_instruments.strip() and days[-1] == binding["end_date"],
            "source publication calendar or universe changed")
    record = {"name": root.name, "path": str(root), "end_date": days[-1],
              "start_date": days[0], "provenance": provenance,
              "lineage_id": provenance["dataset_lineage_id"],
              "frequency": str(provenance.get("frequency") or "day"),
              "ready": True, "reproducible": True, "lineage_verified": True}
    proof = {"provenance_sha256": hashlib.sha256(raw).hexdigest(),
             "calendar_sha256": hashlib.sha256(raw_calendar).hexdigest(),
             "instruments_sha256": hashlib.sha256(raw_instruments).hexdigest(),
             "numeric_outputs_revalidated": False,
             "metadata_only_current_check": True,
             "worker_full_verification_required": True,
             "historical_admission_attestation": "validated_immutable_manual_event",
             "numeric_output_validation_owner": "ordinary_autopilot_and_worker_admission"}
    return record, proof


def preparation_outputs(run_root: Path, feature_set: dict[str, Any]) -> dict[str, str]:
    """Allow only files materialized by command construction before Popen."""
    allowed_directories = {
        "docker-runtime", "qlib-home", "qlib-home/qlib_data", "qlib-home/qlib_data/cn_data",
        "scenario-inputs", "scenario-inputs/base-features",
    }
    expected = {
        "scenario-inputs/base-features/definition.json": feature_set,
        "scenario-inputs/base-features/base_factors.json": feature_set.get("features"),
    }
    hashes = {}
    pending = list(run_root.iterdir()) if run_root.exists() else []
    while pending:
        path = pending.pop()
        relative = path.relative_to(run_root).as_posix()
        if relative == "research-dataset":
            continue  # Its exact publication and pre-final view are checked separately.
        require(not path.is_symlink(), "source setup path contains a symlink")
        if path.is_dir():
            require(relative in allowed_directories, "source already wrote an experiment directory")
            pending.extend(path.iterdir())
        else:
            require(relative in expected, "source already wrote experiment or result files")
            raw = path.read_bytes()
            require(json.loads(raw.decode("utf-8")) == expected[relative],
                    "staged base-feature input differs from the original job")
            hashes[relative] = hashlib.sha256(raw).hexdigest()
    return hashes


def validate_source(
    source: dict[str, Any], tournament: dict[str, Any], branch: dict[str, Any],
    run: dict[str, Any], job: dict[str, Any], dataset: dict[str, Any],
) -> None:
    from .autopilot import (
        _event_dataset_binding,
        _joint_completion_verified,
        _manual_research_event,
    )

    event = _manual_research_event(source)
    require(event is not None and event["request"].get("completion_mode")
            == "managed_fin_strategy", "source is not a managed manual activity")
    require(source.get("status") == "blocked" and source.get("finished_at")
            and source.get("stage") == "joint_optimization_blocked",
            "source is not a terminal handoff failure")
    require(not source["state"].get(RECOVERY_KEY), "recovery chains are not permitted")
    require(_event_dataset_binding(dataset) == event["dataset"]
            and all(dataset.get(flag) is True
                    for flag in ("ready", "reproducible", "lineage_verified"))
            and dataset.get("frequency") == "day", "source dataset is unavailable or changed")
    require(_joint_completion_verified(source, tournament), "model selection is not sealed")
    require(not _has_opened_oos(source["state"]), "source has opened OOS or capital evidence")
    require(branch.get("scenario") == "fin_quant" and branch.get("status") == "failed"
            and branch.get("research_run_id") == run.get("id")
            and branch.get("job_id") == job.get("id") == run.get("job_id"),
            "failed branch/run/job ownership differs")
    require(run.get("status") == "failed" and job.get("status") == "failed"
            and job.get("kind") == "rdagent_quant" and job.get("error") == HANDOFF_ERROR,
            "failure is not the supported pre-experiment handoff defect")
    require(run.get("finished_at") and job.get("finished_at"), "source owner is not terminal")
    payload = dict(job.get("payload_json") or job.get("payload") or {})
    require(payload.get("research_run_id") == run["id"]
            and (run.get("config_json") or {}).get("autopilot_cycle_id") == source["id"]
            and payload.get("research_tournament_id") == tournament["id"]
            and payload.get("dataset_identity_sha256") == source["dataset_identity_sha256"],
            "failed job frozen ownership differs")
    require(payload.get("prediction_champion") == source["state"]["prediction_champion"]
            and payload.get("prediction_champion_evidence")
            == source["state"]["prediction_champion_evidence"], "failed job champion differs")
    require(not _has_opened_oos(payload), "failed job contains opened OOS evidence")
    require(payload.get("loop_n") == event["request"]["quant_loop_n"]
            and payload.get("duration") == event["request"]["quant_duration"],
            "source job did not use the original managed research budget")
    for item in source.get("branches", []):
        require(item["id"] == branch["id"] or item.get("scenario") == "fin_model"
                and item.get("status") == "succeeded", "another source branch is unresolved")


def require_recovery_lineage(
    cycle: dict[str, Any], tournament: dict[str, Any], source: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate the only permitted cross-cycle model-tournament reference."""
    from .autopilot import _manual_research_event

    lineage = deepcopy((cycle.get("state") or {}).get(RECOVERY_KEY))
    require(isinstance(lineage, dict), "lineage is missing")
    digest = lineage.pop("sha256", None)
    require(digest == canonical_sha256(lineage), "lineage hash changed")
    require(lineage.get("contract_version") == RECOVERY_VERSION
            and source is not None and source.get("id") == lineage.get("source_cycle_id")
            and source.get("id") != cycle.get("id"), "source ownership differs")
    require(canonical_sha256(source_identity(source)) == lineage.get("source_cycle_sha256"),
            "source activity changed")
    require(canonical_sha256(tournament_identity(tournament))
            == lineage.get("source_tournament_sha256")
            and tournament.get("id") == lineage.get("source_tournament_id")
            and tournament.get("cycle_id") == source["id"]
            and tournament.get("status") == "succeeded" and tournament.get("finished_at"),
            "source tournament changed")
    event = _manual_research_event(cycle)
    source_event = _manual_research_event(source)
    require(event is not None and source_event is not None
            and event["sha256"] == lineage.get("recovery_event_sha256")
            and source_event["sha256"] == lineage.get("source_event_sha256")
            and event["dataset"] == source_event["dataset"]
            and event["config"] == source_event["config"], "frozen activity inputs differ")
    for key in ("quant_loop_n", "quant_duration", "completion_mode", "horizon_profile"):
        require(event["request"].get(key) == source_event["request"].get(key),
                "source research budget changed")
    require(cycle.get("dataset_identity_sha256") == source.get("dataset_identity_sha256")
            and cycle.get("dataset_lineage_id") == source.get("dataset_lineage_id")
            and cycle.get("horizon_profile") == source.get("horizon_profile"),
            "source dataset identity differs")
    for key in EVIDENCE_KEYS:
        require(cycle["state"].get(key) == source["state"].get(key),
                "frozen model evidence changed: " + key)
    require(cycle["state"].get("research_tournament_id") == tournament["id"]
            and cycle["state"].get("active_research_tournament_id") == tournament["id"],
            "active source tournament differs")
    return {**lineage, "sha256": digest}


def require_recovery_evidence(
    cycle: dict[str, Any], evidence: dict[str, Any], settings: Any,
) -> dict[str, Any]:
    lineage = require_recovery_lineage(cycle, evidence["tournament"], evidence["source"])
    observed = {
        "source_trials_sha256": evidence["trials_sha256"],
        "source_branches_sha256": canonical_sha256(sorted(
            evidence["source"]["branches"], key=lambda item: item["id"])),
        "source_run_sha256": canonical_sha256(evidence["run"]),
        "source_job_sha256": canonical_sha256(evidence["job"]),
        "source_output_proof_sha256": canonical_sha256(evidence["output_proof"]),
        "source_publication_proof_sha256": canonical_sha256(evidence["publication_proof"]),
    }
    require(all(lineage.get(key) == value for key, value in observed.items()),
            "source trials, owners or output evidence changed")
    require(lineage.get("target_release") == {
        "release_id": settings.quantlab_release_id,
        "config_digest": settings.quantlab_config_digest,
    }, "recovery target release changed")
    return lineage


class _JoinedEngine:
    def __init__(self, connection: Any) -> None:
        self.connection = connection

    @contextmanager
    def begin(self):
        yield self.connection

    @contextmanager
    def connect(self):
        yield self.connection


class FinQuantHandoffRecovery:
    def __init__(self, controller: Any) -> None:
        self.controller = controller

    def _read(self, connection: Any, source_cycle_id: str) -> dict[str, Any]:
        from .autopilot import AutopilotStore, _manual_research_event
        from .research_tournament import ResearchTournamentStore

        store = object.__new__(AutopilotStore)
        store.engine = _JoinedEngine(connection)
        source = store.get_cycle(source_cycle_id)
        event = _manual_research_event(source)
        require(event is not None, "source is not a manual event")
        dataset, publication_proof = _source_publication(
            Path(self.controller.settings.data_root), event["dataset"],
        )
        quant = [item for item in source["branches"] if item["scenario"] == "fin_quant"]
        require(len(quant) == 1, "source must have exactly one fin_quant branch")
        branch = quant[0]
        run = connection.execute(select(research_runs)
            .where(research_runs.c.id == branch["research_run_id"])).first()
        job = connection.execute(select(jobs).where(jobs.c.id == branch["job_id"])).first()
        require(run is not None and job is not None, "source run/job is missing")
        run, job = row_dict(run), row_dict(job)
        tid = source["state"].get("research_tournament_id")
        row = connection.execute(select(research_tournaments)
            .where(research_tournaments.c.id == tid)).first()
        require(row is not None, "source tournament is missing")
        tournament = ResearchTournamentStore._decode_tournament(row_dict(row))
        validate_source(source, tournament, branch, run, job, dataset)
        # A handoff repair is pre-experiment only. Existing outputs, candidate
        # registrations, independent evaluation or capital dispatch fail closed.
        for table in (factor_candidates, model_candidates, quant_bundle_candidates,
                      research_run_artifacts):
            require(connection.scalar(select(table.c.id).where(
                table.c.research_run_id == run["id"]).limit(1)) is None,
                "source fin_quant already registered research outputs")
        artifact_base = Path(self.controller.settings.data_root) / "artifacts" / "rdagent"
        require(Path(str(run.get("artifact_path") or "")).resolve() == artifact_base.resolve(),
                "source run artifact root differs from its registered command")
        run_root = artifact_base / run["id"]
        require(not run_root.is_symlink() and run_root.resolve().is_relative_to(
            artifact_base.resolve()), "source run output escaped its governed root")
        # rdagent_command prepares its truncated input provider before baseline
        # validation. Permit only that exact, pre-final view; it is not research
        # output. A trace/checkpoint/result anywhere else is ineligible.
        setup_files = preparation_outputs(run_root, job["payload_json"]["feature_set"])
        view_sha256 = None
        if (run_root / "research-dataset").exists():
            view_root = run_root / "research-dataset"
            require(view_root.is_dir() and not view_root.is_symlink(), "input view is invalid")
            view_path = view_root / "quantlab-rdagent-dataset-view.json"
            require(view_path.is_file() and not view_path.is_symlink(), "input view is unsealed")
            raw_view = view_path.read_bytes()
            view = json.loads(raw_view.decode("utf-8"))
            payload = job["payload_json"]
            require(view.get("schema_version") == 3
                    and Path(str(view.get("source"))).resolve() == Path(dataset["path"]).resolve()
                    and view.get("requested_cutoff") == payload["periods"]["valid_end"]
                    and view.get("effective_cutoff") <= payload["periods"]["valid_end"]
                    and view.get("future_calendar_contains_market_data") is False,
                    "prepared input view crosses the source pre-final boundary")
            require(all(path.name in {"features", "calendars", "instruments", view_path.name}
                        for path in view_root.iterdir()), "input view contains unexpected outputs")
            view_sha256 = hashlib.sha256(raw_view).hexdigest()
        output_proof = {"registered_artifact_base": str(artifact_base),
                        "command_output_root": str(run_root), "experiment_files": [],
                        "verified_pre_subprocess_setup_files": setup_files,
                        "prepared_input_view_manifest_sha256": view_sha256,
                        "registered_research_outputs": 0, "pre_subprocess_exit_code": 2}
        require(job.get("exit_code") == 2, "source did not fail before subprocess execution")
        for table, column in ((strategy_versions, strategy_versions.c.config_json),
                              (schedules, schedules.c.payload_json)):
            require(connection.scalar(select(table.c.id).where(or_(
                cast(column, Text).contains(source["id"]),
                cast(column, Text).contains(run["id"]),
            )).limit(1)) is None, "source already has downstream capital/strategy dispatch")
        require(not _has_opened_oos(run.get("config_json")), "source run has opened OOS")
        trial_rows = [row_dict(row) for row in connection.execute(
            select(research_tournament_trials).where(
                research_tournament_trials.c.tournament_id == tid)
            .order_by(research_tournament_trials.c.id))]
        return {"source": source, "tournament": tournament, "branch": branch,
                "run": run, "job": job, "dataset": dataset,
                "trials_sha256": canonical_sha256(trial_rows), "output_proof": output_proof,
                "publication_proof": publication_proof}

    def verify(self, cycle: dict[str, Any], *, connection: Any = None) -> dict[str, Any]:
        source_id = (cycle.get("state") or {}).get(RECOVERY_KEY, {}).get("source_cycle_id")
        if connection is None:
            with self.controller.engine.connect() as reader:
                evidence = self._read(reader, source_id)
        else:
            evidence = self._read(connection, source_id)
        require_recovery_evidence(cycle, evidence, self.controller.settings)
        return evidence

    def _plan(self, connection: Any, source_cycle_id: str, event_key: str,
              actor: str, reason: str) -> dict[str, Any]:
        from .autopilot import _manual_research_event, _research_event_request
        from .rdagent_runtime import validate_duration_limit

        evidence = self._read(connection, source_cycle_id)
        event = _manual_research_event(evidence["source"])
        request = _research_event_request(**{
            **event["request"], "event_key": event_key, "actor": actor, "reason": reason,
        })
        require(request["event_key"] != event["request"]["event_key"], "new event key required")
        settings = self.controller.settings
        require(settings.rdagent_enabled and request["quant_loop_n"] <= settings.rdagent_max_loops,
                "research runtime is unavailable or budget exceeds its limit")
        validate_duration_limit(request["quant_duration"], settings.rdagent_max_duration)
        config = connection.execute(select(platform_configs)
            .where(platform_configs.c.key == "autopilot")).first()
        require(config is not None, "autopilot configuration is missing")
        target = {"release_id": settings.quantlab_release_id,
                  "config_digest": settings.quantlab_config_digest}
        require(target["release_id"] and len(target["config_digest"]) == 64,
                "target release identity is absent")
        return {
            "contract_version": RECOVERY_VERSION, "request": request,
            "source_cycle_id": source_cycle_id,
            "source_cycle_sha256": canonical_sha256(source_identity(evidence["source"])),
            "source_event_sha256": event["sha256"],
            "source_tournament_id": evidence["tournament"]["id"],
            "source_tournament_sha256": canonical_sha256(
                tournament_identity(evidence["tournament"])),
            "source_trials_sha256": evidence["trials_sha256"],
            "source_branches_sha256": canonical_sha256(sorted(
                evidence["source"]["branches"], key=lambda item: item["id"])),
            "source_run_id": evidence["run"]["id"],
            "source_job_id": evidence["job"]["id"],
            "source_run_sha256": canonical_sha256(evidence["run"]),
            "source_job_sha256": canonical_sha256(evidence["job"]),
            "source_output_proof": evidence["output_proof"],
            "source_output_proof_sha256": canonical_sha256(evidence["output_proof"]),
            "source_publication_proof": evidence["publication_proof"],
            "source_publication_proof_sha256": canonical_sha256(evidence["publication_proof"]),
            "source_failure": HANDOFF_ERROR, "target_release": target,
            "current_config_revision": config.revision,
            "current_config_sha256": canonical_sha256(config.value_json),
            "dataset": deepcopy(event["dataset"]),
            "model_selection_reused": True, "new_model_experiments": 0,
            "final_oos_opened": False, "source_outputs_absent": True,
        }

    def plan(self, source_cycle_id: str, event_key: str, actor: str,
             reason: str) -> dict[str, Any]:
        with self.controller.engine.connect() as connection:
            connection.exec_driver_sql("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            return self._plan(connection, source_cycle_id, event_key, actor, reason)

    def execute(self, plan: dict[str, Any], *, expected_sha256: str) -> dict[str, Any]:
        from .autopilot import AutopilotStore, _manual_research_event, _now

        require(canonical_sha256(plan) == expected_sha256, "approved plan hash changed")
        require(plan.get("contract_version") == RECOVERY_VERSION, "unsupported recovery plan")
        request = plan["request"]
        with self.controller.engine.begin() as connection:
            AutopilotStore._lock_horizon(connection, request["horizon_profile"])
            connection.execute(select(platform_configs.c.key)
                .where(platform_configs.c.key == "autopilot").with_for_update()).all()
            connection.execute(select(autopilot_cycles.c.id)
                .where(autopilot_cycles.c.id == plan["source_cycle_id"]).with_for_update()).all()
            for table, clause in (
                (autopilot_branches, autopilot_branches.c.cycle_id == plan["source_cycle_id"]),
                (research_runs, research_runs.c.id == plan["source_run_id"]),
                (jobs, jobs.c.id == plan["source_job_id"]),
                (research_tournaments, research_tournaments.c.id == plan["source_tournament_id"]),
                (research_tournament_trials,
                 research_tournament_trials.c.tournament_id == plan["source_tournament_id"]),
            ):
                connection.execute(select(table.c.id).where(clause)
                    .order_by(table.c.id).with_for_update()).all()
            store = object.__new__(AutopilotStore)
            store.engine = _JoinedEngine(connection)
            existing = store.get_research_event(request["event_key"])
            if existing is not None:
                require((existing["state"].get(RECOVERY_KEY) or {}).get("plan_sha256")
                        == expected_sha256, "event key belongs to another recovery")
                return existing
            actual = self._plan(connection, plan["source_cycle_id"], request["event_key"],
                                request["actor"], request["reason"])
            require(actual == plan, "live state or target release changed after planning")
            # Prevent duplicate operator recovery with a different event key.
            require(connection.scalar(select(autopilot_cycles.c.id).where(
                autopilot_cycles.c.state_json[RECOVERY_KEY]["source_cycle_id"].as_string()
                == plan["source_cycle_id"]).limit(1)) is None,
                "source already owns a recovery successor")
            evidence = self._read(connection, plan["source_cycle_id"])
            source = evidence["source"]
            source_event = _manual_research_event(source)
            cycle = store.create_research_event(
                request=request, dataset=evidence["dataset"], config=source_event["config"],
                config_revision=source_event["config_revision"],
            )
            lineage = {**deepcopy(plan), "plan_sha256": expected_sha256,
                       "recovery_event_sha256": cycle["state"]["research_event"]["sha256"]}
            lineage["sha256"] = canonical_sha256(lineage)
            state = {**cycle["state"], **{
                key: deepcopy(source["state"][key])
                for key in EVIDENCE_KEYS if key in source["state"]
            }, RECOVERY_KEY: lineage,
                "research_tournament_id": plan["source_tournament_id"],
                "active_research_tournament_id": plan["source_tournament_id"],
                "prediction_champion_status": "validated_source_handoff_recovery",
                "fin_quant_status": "pending_handoff_recovery", "final_oos_opened": False}
            cycle = store.set_cycle_state(cycle["id"], state=state, stage="joint_optimization")
            require_recovery_lineage(cycle, evidence["tournament"], source)
            connection.execute(insert(audit_events).values(
                user_id=None, username=request["actor"],
                action="autopilot.fin_quant_handoff_recovery.register", method="SERVICE",
                path="scripts/recover_fin_quant_handoff.py", status_code=202,
                details_json={"cycle_id": cycle["id"], "plan": plan,
                              "plan_sha256": expected_sha256}, created_at=_now(),
            ))
            return cycle


def plan_file(path: Path, expected_bytes_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_bytes_sha256, "plan file bytes changed")
    return json.loads(raw.decode("utf-8"))
