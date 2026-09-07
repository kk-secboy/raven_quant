"""Bind completed manual research to the existing managed strategy dispatcher.

This service only appends a schedule and its ordinary run. Autopilot records,
calendar cadence, StrategyVersions, final OOS and paper state are not modified.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from sqlalchemy import exists, select, text

from quant_data.database import (
    autopilot_branches,
    autopilot_cycles,
    jobs,
    quant_bundle_candidates,
    research_runs,
    schedule_runs,
    schedules,
)

from .autopilot import _manual_research_event
from .fin_strategy_schedule import (
    build_managed_fin_strategy_schedule_specs,
    canonical_sha256,
    select_latest_reproducible_daily_dataset,
    validate_managed_fin_strategy_payload,
)

COMPLETION_KEY = "managed_fin_strategy_completion"
COMPLETION_VERSION = "managed-fin-strategy-research-completion-v1"
COMPLETION_ACTOR = "system:fin-strategy-research-completion"
COMPLETION_NAME_PREFIX = "QuantLab / fin_strategy / research completion / "
_TERMINAL_BRANCHES = {"succeeded", "failed", "blocked", "skipped"}


def completed_event(cycle: dict[str, Any]) -> dict[str, Any] | None:
    event = _manual_research_event(cycle)
    if (
        event is None or event["request"].get("completion_mode") != "managed_fin_strategy"
        or cycle.get("status") != "succeeded"
        or cycle.get("stage") != "research_complete" or not cycle.get("finished_at")
        or any(b.get("status") not in _TERMINAL_BRANCHES for b in cycle.get("branches", []))
        or not any(b.get("scenario") == "fin_quant" and b.get("status") == "succeeded"
                   for b in cycle.get("branches", []))
    ):
        return None
    return event


def build_completion_binding(
    cycle: dict[str, Any], candidates: list[dict[str, Any]], selection: dict[str, Any],
    managed: dict[str, Any],
) -> dict[str, Any]:
    event = completed_event(cycle)
    if event is None or not candidates:
        raise ValueError("strategy completion requires completed manual joint research")
    evidence = selection.get("champion_selection_evidence", {})
    if (
        evidence.get("dataset") != event["dataset"]["name"]
        or evidence.get("dataset_identity_sha256") != cycle["dataset_identity_sha256"]
        or evidence.get("horizon_profile") != cycle["horizon_profile"]
        or managed["horizon_profile"] != cycle["horizon_profile"]
        or evidence.get("final_oos_opened") is not False
        or evidence.get("research_screening_only") is not True
        or evidence.get("not_capital_confirmation") is not True
        or selection.get("champion_selection_evidence_sha256") != canonical_sha256(evidence)
    ):
        raise ValueError("strategy completion champion evidence binding differs")
    candidate_ids = {item["id"] for item in candidates}
    admitted_joint = {item["candidate_id"] for item in evidence.get("eligible_candidates", [])
                      if item.get("kind") == "joint"}
    if not candidate_ids or not candidate_ids <= admitted_joint:
        raise ValueError("strategy completion has no verified independent joint evidence")
    binding = {
        "contract_version": COMPLETION_VERSION,
        "source_cycle_id": str(cycle["id"]),
        "research_event_key": cycle["research_event_key"],
        "research_event_sha256": event["sha256"],
        "dataset": deepcopy(event["dataset"]),
        "horizon_profile": cycle["horizon_profile"],
        "joint_candidates": sorted(deepcopy(candidates), key=lambda item: item["id"]),
        "champion_selection": deepcopy(selection),
        "managed_contract_sha256": managed["contract_sha256"],
        "final_oos_opened": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return binding


def require_completion_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    raw = payload.get(COMPLETION_KEY)
    if raw is None:
        return None
    managed = validate_managed_fin_strategy_payload(payload)
    if not isinstance(raw, dict) or managed is None:
        raise ValueError("strategy completion requires a managed payload")
    expected_payload = next(
        spec["payload"] for spec in build_managed_fin_strategy_schedule_specs()
        if spec["payload"]["horizon_profile"] == managed["horizon_profile"]
    )
    if {key: value for key, value in payload.items() if key != COMPLETION_KEY} != expected_payload:
        raise ValueError("strategy completion schedule definition changed")
    binding = deepcopy(raw)
    supplied = binding.pop("binding_sha256", None)
    if (
        set(binding) != {"contract_version", "source_cycle_id", "research_event_key",
                         "research_event_sha256", "dataset", "horizon_profile",
                         "joint_candidates", "champion_selection", "managed_contract_sha256",
                         "final_oos_opened"}
        or binding.get("contract_version") != COMPLETION_VERSION
        or supplied != canonical_sha256(binding)
        or binding.get("managed_contract_sha256") != managed["contract_sha256"]
        or binding.get("horizon_profile") != managed["horizon_profile"]
        or binding.get("final_oos_opened") is not False
    ):
        raise ValueError("strategy completion frozen contract differs")
    return {**binding, "binding_sha256": supplied}


def select_completion_dataset(
    binding: dict[str, Any], available: list[dict[str, Any]],
) -> dict[str, Any]:
    frozen = binding["dataset"]
    # Run the existing strict daily-publication verifier on exactly the frozen
    # vintage. A later publication cannot silently replace this research input.
    dataset = select_latest_reproducible_daily_dataset(
        [item for item in available if item.get("name") == frozen["name"]]
    )
    actual = {
        "name": dataset["name"],
        "path": str(dataset["path"]),
        "dataset_identity_sha256": dataset["dataset_identity_sha256"],
        "dataset_lineage_id": dataset["dataset_lineage_id"],
        "end_date": str(dataset.get("end_date") or ""),
    }
    if any(actual[key] != frozen[key] for key in actual):
        raise ValueError("strategy completion dataset changed")
    return dataset


class ManagedResearchCompletion:
    def __init__(self, controller, schedule_store):
        self.controller = controller
        self.schedules = schedule_store
        self.engine = schedule_store.engine

    def pending_cycle_ids(self) -> list[str]:
        with self.engine.connect() as connection:
            return list(connection.scalars(
                select(autopilot_cycles.c.id).where(
                    autopilot_cycles.c.research_event_key.like("manual:%"),
                    autopilot_cycles.c.state_json["research_event"]["request"]["completion_mode"]
                    .as_string() == "managed_fin_strategy",
                    autopilot_cycles.c.status == "succeeded",
                    autopilot_cycles.c.stage == "research_complete",
                    autopilot_cycles.c.finished_at.is_not(None),
                    exists(select(quant_bundle_candidates.c.id).join(
                        autopilot_branches,
                        autopilot_branches.c.research_run_id
                        == quant_bundle_candidates.c.research_run_id,
                    ).where(
                        autopilot_branches.c.cycle_id == autopilot_cycles.c.id,
                        autopilot_branches.c.scenario == "fin_quant",
                        autopilot_branches.c.status == "succeeded",
                        quant_bundle_candidates.c.status == "research_admitted",
                    )),
                    ~exists(select(schedule_runs.c.id).join(schedules).where(
                        schedules.c.name == COMPLETION_NAME_PREFIX + autopilot_cycles.c.id
                    )),
                ).order_by(autopilot_cycles.c.created_at).limit(20)
            ))

    def _candidates(self, cycle: dict[str, Any]) -> list[dict[str, Any]]:
        run_ids = [b["research_run_id"] for b in cycle.get("branches", [])
                   if b.get("scenario") == "fin_quant" and b.get("status") == "succeeded"]
        with self.engine.connect() as connection:
            # Only independently admitted bundles produced by this actual activity
            # may expand its candidate family; same-vintage neighbours are excluded.
            rows = connection.execute(select(
                quant_bundle_candidates.c.id, quant_bundle_candidates.c.research_run_id,
                quant_bundle_candidates.c.bundle_manifest_sha256,
                quant_bundle_candidates.c.admission_evidence_sha256,
            ).join(research_runs).where(
                quant_bundle_candidates.c.research_run_id.in_(run_ids),
                research_runs.c.status == "succeeded",
                quant_bundle_candidates.c.status == "research_admitted",
                quant_bundle_candidates.c.dataset == cycle["dataset"],
                quant_bundle_candidates.c.dataset_identity_sha256
                == cycle["dataset_identity_sha256"],
            ).order_by(quant_bundle_candidates.c.id)).mappings()
            return [dict(row) for row in rows]

    def enabled_and_idle(self, horizon: str) -> bool:
        config, _ = self.controller.config()
        if not self.controller.settings.rdagent_enabled or not config["enabled"]:
            return False
        with self.engine.connect() as connection:
            active_cycle = connection.scalar(select(autopilot_cycles.c.id).where(
                autopilot_cycles.c.horizon_profile == horizon,
                autopilot_cycles.c.status.in_(("active", "paused")),
                autopilot_cycles.c.finished_at.is_(None),
                autopilot_cycles.c.stage.not_in(("superseded", "legacy_readonly")),
                autopilot_cycles.c.state_json["historical_results_only"].as_boolean().is_not(True),
            ).limit(1))
            active_branch = connection.scalar(select(autopilot_branches.c.id)
                .join(autopilot_cycles).join(
                    research_runs, research_runs.c.id == autopilot_branches.c.research_run_id,
                ).outerjoin(jobs, jobs.c.id == research_runs.c.job_id).where(
                    autopilot_cycles.c.horizon_profile == horizon,
                    (research_runs.c.status.in_(("queued", "running", "evaluating")))
                    | (jobs.c.status.in_(("queued", "running", "cancel_requested"))),
                ).limit(1))
        return active_cycle is None and active_branch is None

    def bind(self, cycle_id: str, managed: dict[str, Any]) -> dict[str, Any] | None:
        cycle = self.controller.store.get_cycle(cycle_id)
        if completed_event(cycle) is None:
            return None
        candidates = self._candidates(cycle)
        if not candidates:
            return None
        selection = self.controller.champion_selector.select_champion(
            dataset=cycle["dataset"], dataset_identity_sha256=cycle["dataset_identity_sha256"],
            horizon_profile=cycle["horizon_profile"],
            allowed_candidate_ids=[item["id"] for item in candidates],
        )
        return build_completion_binding(cycle, candidates, selection, managed)

    def register(self, cycle_id: str, now: datetime) -> dict[str, Any] | None:
        cycle = self.controller.store.get_cycle(cycle_id)
        if completed_event(cycle) is None or not self.enabled_and_idle(cycle["horizon_profile"]):
            return None
        name = COMPLETION_NAME_PREFIX + cycle_id
        with self.engine.begin() as connection:
            connection.execute(text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
                               {"key": name})
            schedule = self.schedules.get_by_name(name)
            if schedule is None:
                spec = next(item for item in build_managed_fin_strategy_schedule_specs()
                            if item["payload"]["horizon_profile"] == cycle["horizon_profile"])
                binding = self.bind(cycle_id, spec["payload"]["managed_fin_strategy"])
                if binding is None:
                    return None
                payload = {**spec["payload"], COMPLETION_KEY: binding}
                schedule = self.schedules.create(
                    **{**spec, "name": name, "payload": payload},
                    actor=COMPLETION_ACTOR, now=now, status="paused",
                )
            prior = connection.scalar(select(schedule_runs.c.id).where(
                schedule_runs.c.schedule_id == schedule["id"]
            ).limit(1))
            if prior is not None:
                return self.schedules.get_run(str(prior))
            if not self.dispatch_allowed(schedule["id"]):
                return None
            return self.schedules.trigger_now(schedule["id"], actor=COMPLETION_ACTOR, now=now)

    def dispatch_allowed(
        self, schedule_id: str, *, connection=None, expected_payload=None,
    ) -> bool:
        statement = select(schedules).where(
                schedules.c.id == schedule_id,
                schedules.c.created_by == COMPLETION_ACTOR,
                schedules.c.status == "paused", schedules.c.desired_status == "paused",
                schedules.c.suspension_reason.is_(None),
                schedules.c.created_at == schedules.c.updated_at,
            )
        if connection is None:
            with self.engine.connect() as reader:
                row = reader.execute(statement).mappings().one_or_none()
        else:
            # Serialize the actual enqueue with a concurrent operator set_status.
            row = connection.execute(statement.with_for_update()).mappings().one_or_none()
        if row is None:
            return False
        if expected_payload is not None and row["payload_json"] != expected_payload:
            raise ValueError("strategy completion payload changed after claim")
        binding = require_completion_payload(row["payload_json"])
        if binding is None or row["name"] != COMPLETION_NAME_PREFIX + binding["source_cycle_id"]:
            return False
        return self.enabled_and_idle(binding["horizon_profile"])

    def verify(self, payload: dict[str, Any]) -> dict[str, Any]:
        binding = require_completion_payload(payload)
        if binding is None:
            raise ValueError("strategy completion binding is missing")
        actual = self.bind(binding["source_cycle_id"], payload["managed_fin_strategy"])
        if actual != binding:
            raise ValueError("strategy completion source or independent evidence changed")
        return binding
