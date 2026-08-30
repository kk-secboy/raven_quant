from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.config import Settings
from quant_data.database import (
    autopilot_branches,
    autopilot_cycles,
    jobs,
    model_candidates,
    model_ensemble_candidates,
    model_evaluations,
    open_database,
    research_runs,
    row_dict,
)

from .autopilot_capital_pipeline import (
    CAPITAL_PIPELINE_STATE_KEY,
    AutopilotCapitalBlocked,
    AutopilotCapitalPipeline,
)
from .autopilot_completion import aggregate_pre_final_grid
from .factor_autopilot import FactorAutopilotService
from .factor_library_store import FactorLibraryStore
from .feature_set_registry import get_feature_set
from .horizon_factor_bundle import (
    build_horizon_factor_bundle,
    validate_horizon_factor_bundle,
)
from .job_store import JobStore
from .model_ensemble import (
    EnsemblePredictionsPending,
    pairwise_grid_correlation,
    prediction_grid_from_admission,
)
from .model_ensemble_pipeline import ModelEnsemblePipelineService
from .model_research_governance import (
    REQUIRED_MODEL_SEEDS,
    REQUIRED_RESEARCH_PROFILES,
    build_run_multiple_testing_evidence,
    file_sha256,
)
from .platform_config_store import PlatformConfigStore
from .platform_model_tournament import PlatformModelTournamentService
from .rdagent_runtime import expected_rdagent_runtime_identity, probe_rdagent
from .rdagent_scenarios import (
    get_rdagent_scenario,
    require_ready_scenario,
    resolve_rdagent_assets,
)
from .research_asset_store import ResearchAssetStore
from .research_automation import resolve_research_periods, resolve_research_window_contract
from .research_horizon import (
    LEGACY_AMBIGUOUS,
    LONG_1_3Y,
    SHORT_1_5D,
    SWING_1_6M,
    primary_label_horizon_sessions,
    primary_label_policy_contract,
    primary_label_policy_sha256,
)
from .research_horizon import (
    research_cadence_bucket as horizon_research_cadence_bucket,
)
from .research_label_binding import resolve_research_label_binding
from .research_store import ResearchStore
from .research_tournament import (
    FEATURE_SCREEN_IDS,
    FULL_PROFILES,
    FULL_SEEDS,
    MODEL_FAMILIES,
    ResearchTournamentStore,
    canonical_sha256,
)
from .services import list_qlib_datasets
from .statistical_validation import holm_bonferroni

AUTOPILOT_CONFIG_KEY = "autopilot"
AUTOPILOT_CONTRACT_VERSION = "autopilot-v1"
RDAGENT_INTEGRATION_CONTRACT_VERSION = "rdagent-integration-v4"
AUTOPILOT_RESEARCH_HORIZONS = (SHORT_1_5D, SWING_1_6M, LONG_1_3Y)

DEFAULT_AUTOPILOT_CONFIG: dict[str, Any] = {
    "contract_version": AUTOPILOT_CONTRACT_VERSION,
    "enabled": True,
    "factor_loop_n": 2,
    "factor_duration": "1h",
    "factor_research_interval_days": 1,
    "model_loop_n": 1,
    "model_duration": "1h",
    "model_research_interval_days": 28,
    "model_feature_set_id": "qlib-alpha158",
    "model_feature_set_ids": list(FEATURE_SCREEN_IDS),
    "model_families": list(MODEL_FAMILIES),
    "qlib_evaluation_concurrency": 3,
    "transformer_concurrency": 1,
    "service_resource_reserve": 0.25,
    "report_loop_n": 1,
    "report_duration": "30m",
    "report_daily_limit": 20,
    "quant_loop_n": 1,
    "quant_duration": "1h",
    "quant_feature_set_id": "qlib-alpha158",
    "quant_cooldown_days": 7,
    "paper_min_calendar_days": 183,
    "paper_min_trading_days": 126,
}


def normalize_autopilot_config(value: Any = None) -> dict[str, Any]:
    raw = deepcopy(DEFAULT_AUTOPILOT_CONFIG)
    if value is not None:
        if not isinstance(value, dict):
            raise ValueError("autopilot configuration must be an object")
        unknown = sorted(set(value) - set(raw))
        if unknown:
            raise ValueError(f"unsupported autopilot settings: {unknown}")
        raw.update(value)
    if raw["contract_version"] != AUTOPILOT_CONTRACT_VERSION:
        raise ValueError("autopilot contract version is invalid")
    if not isinstance(raw["enabled"], bool):
        raise ValueError("autopilot enabled must be boolean")
    for key, minimum, maximum in (
        ("factor_loop_n", 1, 20),
        ("model_loop_n", 1, 20),
        ("report_loop_n", 1, 20),
        ("report_daily_limit", 1, 20),
        ("quant_loop_n", 1, 20),
        ("quant_cooldown_days", 1, 90),
        ("paper_min_calendar_days", 183, 3650),
        ("paper_min_trading_days", 126, 2520),
        ("factor_research_interval_days", 1, 90),
        ("model_research_interval_days", 28, 180),
        ("qlib_evaluation_concurrency", 1, 3),
        ("transformer_concurrency", 1, 1),
    ):
        candidate = raw[key]
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise ValueError(f"{key} must be an integer")
        if not minimum <= candidate <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
    for key in ("factor_duration", "model_duration", "report_duration", "quant_duration"):
        candidate = str(raw[key]).strip()
        if not candidate or len(candidate) > 20:
            raise ValueError(f"{key} is invalid")
        raw[key] = candidate
    for key in ("model_feature_set_id", "quant_feature_set_id"):
        raw[key] = get_feature_set(str(raw[key]))["id"]
    feature_set_ids = raw["model_feature_set_ids"]
    if not isinstance(feature_set_ids, list) or not feature_set_ids:
        raise ValueError("model_feature_set_ids must be a non-empty list")
    raw["model_feature_set_ids"] = [
        get_feature_set(str(item))["id"] for item in dict.fromkeys(feature_set_ids)
    ]
    families = raw["model_families"]
    if not isinstance(families, list) or set(families) != set(MODEL_FAMILIES):
        raise ValueError("model_families must contain ridge, lightgbm, gru and transformer")
    raw["model_families"] = list(MODEL_FAMILIES)
    reserve = raw["service_resource_reserve"]
    if isinstance(reserve, bool) or not isinstance(reserve, (int, float)):
        raise ValueError("service_resource_reserve must be numeric")
    if not 0.25 <= float(reserve) <= 0.75:
        raise ValueError("service_resource_reserve must be between 0.25 and 0.75")
    raw["service_resource_reserve"] = float(reserve)
    return raw


def _now() -> datetime:
    return datetime.now(UTC)


def _branch_created_at(branch: dict[str, Any]) -> datetime:
    """Return one persisted branch timestamp as an aware UTC datetime.

    ``row_dict`` deliberately serializes database datetimes for API-safe store
    results.  Autopilot cadence checks are internal datetime arithmetic, so
    comparing that ISO string directly with ``datetime`` raises at runtime.
    Keep the serialization boundary intact and normalize only at the cadence
    decision point.
    """

    value = branch.get("created_at")
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("autopilot branch created_at is invalid")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _profile_family_multiple_testing(
    *,
    research_run_id: str,
    trial_series_by_profile: dict[
        str, list[tuple[dict[str, str], pd.Series]]
    ],
    family_definitions: list[dict[str, str]],
    output: Path,
    contract_version: str,
) -> dict[str, Any]:
    """Apply one Holm family across every candidate/window hypothesis.

    The three research windows overlap in calendar time, so concatenating them
    and pretending they are independent observations would overstate sample
    size.  We instead compute each candidate's p-value and PBO inside each
    window, then apply Holm once across the complete candidate x window family.
    A candidate is eligible only when all three governed windows survive.
    """

    family_names = [str(item["name"]) for item in family_definitions]
    if len(family_names) != len(set(family_names)):
        raise ValueError("multiple-testing family contains duplicate trials")
    per_profile: dict[str, dict[str, Any]] = {}
    raw_by_profile: dict[str, dict[str, float]] = {}
    completed_by_profile: dict[str, set[str]] = {}
    pbo_passed_by_profile: dict[str, bool] = {}
    for profile_id in REQUIRED_RESEARCH_PROFILES:
        series = list(trial_series_by_profile.get(profile_id) or [])
        if not series:
            per_profile[profile_id] = {
                "status": "no_completed_trials",
                "trial_names": [],
                "raw_p_values": [],
                "pbo": {"status": "not_evaluable", "pbo": None},
            }
            raw_by_profile[profile_id] = {}
            completed_by_profile[profile_id] = set()
            pbo_passed_by_profile[profile_id] = False
            continue
        evidence = build_run_multiple_testing_evidence(
            research_run_id=f"{research_run_id}:{profile_id}",
            trial_series=series,
            output=output / profile_id,
        )
        names = [str(item[0]["name"]) for item in series]
        raw = dict(zip(names, evidence["raw_p_values"], strict=True))
        pbo = dict(evidence["pbo"])
        pbo_passed = pbo.get("status") == "not_applicable_single_trial" or (
            pbo.get("status") == "ok"
            and pbo.get("pbo") is not None
            and float(pbo["pbo"]) <= 0.50
        )
        profile_evidence = {
            **evidence,
            "profile_id": profile_id,
            "pbo_passed": pbo_passed,
        }
        profile_evidence["evidence_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in profile_evidence.items()
                if key != "evidence_sha256"
            }
        )
        per_profile[profile_id] = profile_evidence
        raw_by_profile[profile_id] = {name: float(value) for name, value in raw.items()}
        completed_by_profile[profile_id] = set(names)
        pbo_passed_by_profile[profile_id] = pbo_passed

    hypothesis_names = [
        f"{profile_id}:{name}"
        for profile_id in REQUIRED_RESEARCH_PROFILES
        for name in family_names
    ]
    raw_p_values = [
        raw_by_profile[profile_id].get(name, 1.0)
        for profile_id in REQUIRED_RESEARCH_PROFILES
        for name in family_names
    ]
    adjusted = holm_bonferroni(raw_p_values)
    adjusted_by_hypothesis = dict(
        zip(hypothesis_names, adjusted, strict=True)
    )
    eligible_trial_names = [
        name
        for name in family_names
        if all(
            name in completed_by_profile[profile_id]
            and pbo_passed_by_profile[profile_id]
            and float(adjusted_by_hypothesis[f"{profile_id}:{name}"]) <= 0.05
            for profile_id in REQUIRED_RESEARCH_PROFILES
        )
    ]
    definition_by_name = {str(item["name"]): dict(item) for item in family_definitions}
    evidence = {
        "contract_version": contract_version,
        "source": "independent_qlib_recompute",
        "research_run_id": research_run_id,
        "profile_aggregation": "equal_weight_with_worst_window_robustness",
        "overlapping_profiles_are_not_pooled_as_independent_rows": True,
        "profiles": list(REQUIRED_RESEARCH_PROFILES),
        "per_profile": per_profile,
        "trial_definitions": [
            {
                **definition_by_name[name],
                "profile_status": {
                    profile_id: (
                        "completed"
                        if name in completed_by_profile[profile_id]
                        else "failed"
                    )
                    for profile_id in REQUIRED_RESEARCH_PROFILES
                },
            }
            for name in family_names
        ],
        "trial_names": family_names,
        "trial_count": len(family_names),
        "hypothesis_names": hypothesis_names,
        "hypothesis_count": len(hypothesis_names),
        "raw_p_values": raw_p_values,
        "holm_adjusted_p_values": adjusted,
        "maximum_adjusted_p_value": 0.05,
        "maximum_pbo": 0.50,
        "eligible_trial_names": eligible_trial_names,
        "failed_trials_count_as_p_one": True,
        "gate_passed": bool(eligible_trial_names),
        "final_oos_opened": False,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def _asset_available_on(asset: dict[str, Any]) -> date | None:
    raw = str(asset.get("available_at") or "").strip()
    if not raw:
        return None
    try:
        available_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if available_at.tzinfo is None or available_at.utcoffset() is None:
        return None
    return available_at.astimezone(ZoneInfo("Asia/Shanghai")).date()


_TERMINAL_BRANCH_STATUSES = frozenset({"succeeded", "failed", "blocked", "skipped"})


def _derived_branch_status(run_status: str, job_status: str) -> str:
    """Project execution and research state without calling a blocked study successful."""

    projected = {
        "queued": "queued",
        "running": "running",
        "evaluating": "evaluating",
        "succeeded": "succeeded",
        "blocked": "blocked",
        "failed": "failed",
        "cancelled": "failed",
    }.get(run_status)
    if projected is not None:
        # A research run can be re-attached to a newer evaluation job.  Its
        # state is therefore authoritative over the branch's original job.
        return projected
    if job_status in {"failed", "cancelled"}:
        return "failed"
    return "running"


def _cycle_terminal_resolution(
    branches: list[tuple[str, str]],
) -> tuple[str, str] | None:
    """Return the honest terminal cycle state, or None while work can continue."""

    if not branches or not all(status in _TERMINAL_BRANCH_STATUSES for _, status in branches):
        return None
    quant_statuses = [status for scenario, status in branches if scenario == "fin_quant"]
    # A successful fin_quant branch is not the end of the capital pipeline.
    # Portfolio selection, the one-shot final OOS, formal approval and paper
    # creation still have to run.  Ordinary failed research hypotheses are
    # retained in the trial ledger and must not block a cycle that still has a
    # valid champion.  Only a terminal joint-optimization failure is terminal
    # at the branch layer; later stages are resolved by the completion service.
    if quant_statuses and all(status in {"failed", "blocked"} for status in quant_statuses):
        return "blocked", "joint_optimization_blocked"
    return None


def _prediction_champion_identity_error(
    cycle: dict[str, Any], dataset: dict[str, Any]
) -> str | None:
    """Return why a frozen prediction champion is not current-vintage evidence.

    A predecessor's recipe remains useful research lineage, but its selection
    and admission cannot be relabelled as if they were evaluated on a newly
    published immutable dataset.  Joint optimisation is allowed only after the
    champion selection and its model-family evidence were produced on exactly
    the dataset identity owned by this cycle.
    """

    state = dict(cycle.get("state") or {})
    champion = state.get("prediction_champion")
    selection = state.get("prediction_champion_evidence")
    model_selection = state.get("model_champion_evidence")
    if not isinstance(champion, dict) or not champion.get("candidate_id"):
        return "prediction champion is not frozen"
    if not isinstance(selection, dict):
        return "prediction champion selection evidence is missing"
    if not isinstance(model_selection, dict):
        return "model-family selection evidence is missing"
    dataset_identity = str(
        (dataset.get("provenance") or {}).get("dataset_identity_sha256") or ""
    )
    cycle_identity = str(cycle.get("dataset_identity_sha256") or "")
    if len(dataset_identity) != 64 or cycle_identity != dataset_identity:
        return "cycle and current Qlib dataset identities do not match"
    if str(selection.get("dataset_identity_sha256") or "") != dataset_identity:
        return "prediction champion awaits current dataset identity revalidation"
    if str(model_selection.get("dataset_identity_sha256") or "") != dataset_identity:
        return "model-family evidence awaits current dataset identity revalidation"
    return None


def _cycle_has_capital_commitment(cycle: dict[str, Any]) -> bool:
    """Return whether a cycle has crossed from research into capital validation.

    A publication from a later cadence bucket may supersede ordinary research,
    but it must never orphan a joint winner after the formal capital pipeline
    has started.  The old immutable dataset remains available to finish that
    one-shot OOS attempt while a new research cycle consumes the latest
    publication.
    """

    branches = list(cycle.get("branches") or [])
    quant_succeeded = any(
        str(item.get("scenario") or "") == "fin_quant"
        and str(item.get("status") or "") == "succeeded"
        for item in branches
    )
    capital_state = (cycle.get("state") or {}).get(CAPITAL_PIPELINE_STATE_KEY)
    return quant_succeeded or isinstance(capital_state, dict)


class AutopilotStore:
    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def ensure_cycle(
        self,
        dataset: dict[str, Any],
        *,
        config_revision: int,
        horizon_profile: str = SHORT_1_5D,
    ) -> dict[str, Any]:
        provenance = dict(dataset.get("provenance") or {})
        identity = str(provenance.get("dataset_identity_sha256") or "")
        lineage = str(dataset.get("lineage_id") or provenance.get("dataset_lineage_id") or "")
        if len(identity) != 64 or len(lineage) != 64:
            raise ValueError("autopilot requires verified dataset identity and lineage")
        primary_label = primary_label_horizon_sessions(horizon_profile)
        policy_sha256 = primary_label_policy_sha256()
        now = _now()
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(autopilot_cycles).values(
                        id=uuid.uuid4().hex,
                        dataset=str(dataset["name"]),
                        dataset_identity_sha256=identity,
                        dataset_lineage_id=lineage,
                        horizon_profile=horizon_profile,
                        primary_label_policy_sha256=policy_sha256,
                        status="active",
                        stage="parallel_research",
                        config_revision=config_revision,
                        state_json={
                            "dataset_end_date": dataset.get("end_date"),
                            "horizon_profile": horizon_profile,
                            "label_horizon_sessions": primary_label,
                            "primary_label_policy": primary_label_policy_contract(),
                            "research_cadence_bucket": horizon_research_cadence_bucket(
                                horizon_profile, str(dataset.get("end_date") or "")
                            ),
                        },
                        created_at=now,
                        updated_at=now,
                    )
                )
        except IntegrityError:
            pass
        return self.get_cycle_by_identity(identity, horizon_profile=horizon_profile)

    def get_cycle_by_identity(
        self, identity: str, *, horizon_profile: str = SHORT_1_5D
    ) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(autopilot_cycles).where(
                    autopilot_cycles.c.dataset_identity_sha256 == identity,
                    autopilot_cycles.c.horizon_profile == horizon_profile,
                )
            ).first()
        if row is None:
            raise KeyError(identity)
        return self._decode_cycle(row_dict(row))

    def latest_active_cycle(
        self, *, horizon_profile: str | None = None
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            statement = select(autopilot_cycles).where(
                autopilot_cycles.c.status == "active"
            )
            if horizon_profile is not None:
                statement = statement.where(
                    autopilot_cycles.c.horizon_profile == horizon_profile
                )
            row = connection.execute(
                statement
                .order_by(autopilot_cycles.c.created_at.desc())
                .limit(1)
            ).first()
        return self.get_cycle(str(row.id)) if row is not None else None

    def supersede_research_cycle(
        self,
        cycle_id: str,
        *,
        replacement_dataset: dict[str, Any],
    ) -> dict[str, Any]:
        """Freeze a stale research cycle without cancelling its running jobs."""

        replacement_identity = str(
            (replacement_dataset.get("provenance") or {}).get(
                "dataset_identity_sha256"
            )
            or ""
        )
        if len(replacement_identity) != 64:
            raise ValueError("replacement Qlib dataset identity is invalid")
        now = _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(autopilot_cycles)
                .where(autopilot_cycles.c.id == cycle_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(cycle_id)
            state = dict(row.state_json or {})
            if state.get("superseded_by_dataset_identity_sha256"):
                return self.get_cycle(cycle_id)
            state.update(
                {
                    "superseded_by_dataset": str(replacement_dataset["name"]),
                    "superseded_by_dataset_identity_sha256": replacement_identity,
                    "superseded_at": now.isoformat(),
                    "historical_results_only": True,
                    "capital_eligible": False,
                    "final_oos_must_not_open": True,
                    "running_jobs_cancelled": False,
                }
            )
            connection.execute(
                update(autopilot_cycles)
                .where(autopilot_cycles.c.id == cycle_id)
                .values(
                    state_json=state,
                    stage="superseded",
                    status="paused",
                    error=None,
                    updated_at=now,
                    finished_at=now,
                )
            )
        return self.get_cycle(cycle_id)

    def set_cycle_state(
        self,
        cycle_id: str,
        *,
        state: dict[str, Any],
        stage: str,
        status: str = "active",
        error: str | None = None,
        finished: bool = False,
    ) -> dict[str, Any]:
        if status not in {"active", "blocked", "succeeded", "paused"}:
            raise ValueError("autopilot cycle status is invalid")
        stage_value = str(stage).strip()
        if not stage_value:
            raise ValueError("autopilot cycle stage is required")
        now = _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(autopilot_cycles)
                .where(autopilot_cycles.c.id == cycle_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(cycle_id)
            connection.execute(
                update(autopilot_cycles)
                .where(autopilot_cycles.c.id == cycle_id)
                .values(
                    state_json=state,
                    stage=stage_value,
                    status=status,
                    error=error,
                    updated_at=now,
                    finished_at=now if finished else None,
                )
            )
        return self.get_cycle(str(cycle_id))

    def patch_cycle_state(
        self,
        cycle_id: str,
        *,
        state_patch: dict[str, Any],
        stage: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_cycle(cycle_id)
        state = {**dict(current.get("state") or {}), **state_patch}
        return self.set_cycle_state(
            cycle_id,
            state=state,
            stage=stage or str(current["stage"]),
            status="active",
        )

    def create_branch(
        self,
        cycle_id: str,
        *,
        scenario: str,
        scope_key: str,
        research_run_id: str,
        job_id: str,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        now = _now()
        branch_id = uuid.uuid4().hex
        try:
            with self.engine.begin() as connection:
                cycle = connection.execute(
                    select(
                        autopilot_cycles.c.status,
                        autopilot_cycles.c.state_json,
                    )
                    .where(autopilot_cycles.c.id == cycle_id)
                    .with_for_update()
                ).first()
                if cycle is None:
                    raise KeyError(cycle_id)
                cycle_state = dict(cycle.state_json or {})
                if (
                    str(cycle.status) != "active"
                    or cycle_state.get("historical_results_only") is True
                    or cycle_state.get("final_oos_must_not_open") is True
                ):
                    raise ValueError(
                        "autopilot branch cannot be created on an inactive or "
                        "superseded cycle"
                    )
                connection.execute(
                    insert(autopilot_branches).values(
                        id=branch_id,
                        cycle_id=cycle_id,
                        scenario=scenario,
                        scope_key=scope_key,
                        status="queued",
                        research_run_id=research_run_id,
                        job_id=job_id,
                        details_json=details,
                        created_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    update(autopilot_cycles)
                    .where(autopilot_cycles.c.id == cycle_id)
                    .values(
                        status="active",
                        stage=(
                            "joint_optimization"
                            if scenario == "fin_quant"
                            else "parallel_research"
                        ),
                        updated_at=now,
                        finished_at=None,
                    )
                )
        except IntegrityError as exc:
            raise ValueError("autopilot branch already exists") from exc
        return self.get_branch(branch_id)

    def get_branch(self, branch_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(autopilot_branches).where(autopilot_branches.c.id == branch_id)
            ).first()
        if row is None:
            raise KeyError(branch_id)
        return self._decode_branch(row_dict(row))

    def branch_for_scope(
        self, cycle_id: str, scenario: str, scope_key: str
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(autopilot_branches).where(
                    autopilot_branches.c.cycle_id == cycle_id,
                    autopilot_branches.c.scenario == scenario,
                    autopilot_branches.c.scope_key == scope_key,
                )
            ).first()
        return self._decode_branch(row_dict(row)) if row else None

    def mark_branch_retried(self, branch_id: str) -> dict[str, Any]:
        now = _now()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(autopilot_branches)
                .where(autopilot_branches.c.id == branch_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(branch_id)
            if str(row.status) != "failed":
                raise ValueError("only failed autopilot branches may be retried")
            details = dict(row.details_json or {})
            retry_count = int(details.get("retry_count") or 0)
            if retry_count >= 1:
                raise ValueError("autopilot branch retry budget is exhausted")
            details["retry_count"] = retry_count + 1
            details["retried_at"] = now.isoformat()
            connection.execute(
                update(autopilot_branches)
                .where(autopilot_branches.c.id == branch_id)
                .values(
                    status="queued",
                    details_json=details,
                    error=None,
                    updated_at=now,
                    finished_at=None,
                )
            )
        return self.get_branch(branch_id)

    def latest_branch(
        self, scenario: str, *, horizon_profile: str | None = None
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            statement = select(autopilot_branches)
            if horizon_profile is not None:
                statement = statement.join(
                    autopilot_cycles,
                    autopilot_cycles.c.id == autopilot_branches.c.cycle_id,
                ).where(autopilot_cycles.c.horizon_profile == horizon_profile)
            row = connection.execute(
                statement.where(autopilot_branches.c.scenario == scenario)
                .order_by(autopilot_branches.c.created_at.desc())
                .limit(1)
            ).first()
        return self._decode_branch(row_dict(row)) if row else None

    def count_since(self, scenario: str, since: datetime) -> int:
        with self.engine.connect() as connection:
            return int(
                connection.scalar(
                    select(func.count())
                    .select_from(autopilot_branches)
                    .where(
                        autopilot_branches.c.scenario == scenario,
                        autopilot_branches.c.created_at >= since,
                    )
                )
                or 0
            )

    def reconcile(self) -> int:
        changed = 0
        now = _now()
        with self.engine.begin() as connection:
            rows = connection.execute(
                select(
                    autopilot_branches,
                    research_runs.c.status.label("run_status"),
                    research_runs.c.error.label("run_error"),
                    jobs.c.status.label("job_status"),
                    jobs.c.error.label("job_error"),
                )
                .join(research_runs, research_runs.c.id == autopilot_branches.c.research_run_id)
                .join(jobs, jobs.c.id == autopilot_branches.c.job_id)
                .with_for_update()
            ).all()
            for row in rows:
                run_status = str(row.run_status)
                job_status = str(row.job_status)
                status = _derived_branch_status(run_status, job_status)
                if status != str(row.status):
                    error = str(row.run_error or row.job_error or "") or None
                    connection.execute(
                        update(autopilot_branches)
                        .where(autopilot_branches.c.id == row.id)
                        .values(
                            status=status,
                            error=error,
                            updated_at=now,
                            finished_at=(
                                now if status in _TERMINAL_BRANCH_STATUSES else None
                            ),
                        )
                    )
                    changed += 1
                if status not in _TERMINAL_BRANCH_STATUSES:
                    connection.execute(
                        update(autopilot_cycles)
                        .where(
                            autopilot_cycles.c.id == row.cycle_id,
                            autopilot_cycles.c.status != "paused",
                        )
                        .values(
                            status="active",
                            stage=(
                                "joint_optimization"
                                if str(row.scenario) == "fin_quant"
                                else "parallel_research"
                            ),
                            updated_at=now,
                            finished_at=None,
                        )
                    )
        self._refresh_cycles()
        return changed

    def _refresh_cycles(self) -> None:
        now = _now()
        with self.engine.begin() as connection:
            cycles = connection.execute(
                select(autopilot_cycles).where(autopilot_cycles.c.status == "active")
            ).all()
            for cycle in cycles:
                branches = list(
                    connection.execute(
                        select(
                            autopilot_branches.c.scenario,
                            autopilot_branches.c.status,
                        ).where(autopilot_branches.c.cycle_id == cycle.id)
                    )
                )
                if not branches:
                    continue
                stage = "parallel_research"
                branch_states = [(str(row.scenario), str(row.status)) for row in branches]
                has_quant = any(scenario == "fin_quant" for scenario, _ in branch_states)
                if has_quant:
                    stage = (
                        str(cycle.stage)
                        if str(cycle.stage)
                        not in {"parallel_research", "joint_optimization"}
                        else "joint_optimization"
                    )
                values: dict[str, Any] = {"stage": stage, "updated_at": now}
                resolution = _cycle_terminal_resolution(branch_states)
                if resolution is not None:
                    status, terminal_stage = resolution
                    values.update(status=status, stage=terminal_stage, finished_at=now)
                connection.execute(
                    update(autopilot_cycles)
                    .where(autopilot_cycles.c.id == cycle.id)
                    .values(**values)
                )

    def list_cycles(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(autopilot_cycles)
                .order_by(autopilot_cycles.c.created_at.desc())
                .limit(min(max(limit, 1), 500))
            ).all()
            result = []
            for row in rows:
                cycle = self._decode_cycle(row_dict(row))
                cycle["branches"] = [
                    self._decode_branch(row_dict(item))
                    for item in connection.execute(
                        select(autopilot_branches)
                        .where(autopilot_branches.c.cycle_id == row.id)
                        .order_by(autopilot_branches.c.created_at)
                    )
                ]
                result.append(cycle)
        return result

    def get_cycle(self, cycle_id: str) -> dict[str, Any]:
        cycles = [item for item in self.list_cycles(limit=500) if item["id"] == cycle_id]
        if not cycles:
            raise KeyError(cycle_id)
        return cycles[0]

    @staticmethod
    def _decode_cycle(row: dict[str, Any]) -> dict[str, Any]:
        row["state"] = row.pop("state_json")
        profile = str(row.get("horizon_profile") or "")
        policy = primary_label_policy_contract()
        if profile == LEGACY_AMBIGUOUS:
            state = dict(row["state"] or {})
            if (
                row.get("primary_label_policy_sha256")
                != policy["policy_sha256"]
                or state.get("primary_label_policy") != policy
                or state.get("label_horizon_sessions") is not None
                or state.get("historical_results_only") is not True
                or state.get("migrated_from_unbound_autopilot_cycle") is not True
            ):
                raise ValueError("legacy autopilot cycle was reinterpreted as executable")
            return row
        if (
            primary_label_horizon_sessions(profile)
            != int((row["state"] or {}).get("label_horizon_sessions") or 0)
            or row.get("primary_label_policy_sha256") != policy["policy_sha256"]
            or (row["state"] or {}).get("primary_label_policy") != policy
        ):
            raise ValueError("autopilot cycle horizon or primary-label policy drifted")
        return row

    @staticmethod
    def _decode_branch(row: dict[str, Any]) -> dict[str, Any]:
        row["details"] = row.pop("details_json")
        return row


class AutopilotController:
    """Create bounded RD-Agent research branches from immutable Qlib vintages."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = AutopilotStore(settings.database_url)
        self.configs = PlatformConfigStore(settings.database_url)
        self.jobs = JobStore(settings.database_url)
        self.research = ResearchStore(settings.database_url)
        self.assets = ResearchAssetStore(settings.database_url)
        self.engine = open_database(settings.database_url)
        self.factor_library = FactorLibraryStore(self.engine)
        self.factor_autopilot = FactorAutopilotService(settings)
        self.tournaments = ResearchTournamentStore(settings.database_url)
        self.platform_models = PlatformModelTournamentService(settings)
        self.capital_pipeline = AutopilotCapitalPipeline(settings)
        self.model_ensembles = ModelEnsemblePipelineService(settings)

    def config(self) -> tuple[dict[str, Any], int]:
        record = self.configs.get(AUTOPILOT_CONFIG_KEY)
        return (
            normalize_autopilot_config(record["value"] if record else None),
            int(record["revision"]) if record else 0,
        )

    def tick(self, now: datetime | None = None) -> dict[str, int]:
        current = now or _now()
        self.store.reconcile()
        config, revision = self.config()
        if not config["enabled"]:
            return {"cycles": 0, "branches": 0}
        dataset = self._latest_dataset()
        if dataset is None:
            return {"cycles": 0, "branches": 0}
        totals = {"cycles": 0, "branches": 0, "failed": 0}
        for horizon_profile in AUTOPILOT_RESEARCH_HORIZONS:
            result = self._tick_horizon(
                current=current,
                config=config,
                revision=revision,
                dataset=dataset,
                horizon_profile=horizon_profile,
            )
            for key in totals:
                totals[key] += int(result.get(key) or 0)
        return totals

    def _tick_horizon(
        self,
        *,
        current: datetime,
        config: dict[str, Any],
        revision: int,
        dataset: dict[str, Any],
        horizon_profile: str,
    ) -> dict[str, int]:
        latest_identity = str(
            (dataset.get("provenance") or {}).get("dataset_identity_sha256") or ""
        )
        available_by_name = {
            str(item.get("name")): item
            for item in list_qlib_datasets(self.settings.data_root)
        }
        listed_cycles = self.store.list_cycles(limit=500)
        latest_cycle_is_active = any(
            item.get("status") == "active"
            and item.get("horizon_profile") == horizon_profile
            and str(item.get("dataset_identity_sha256") or "") == latest_identity
            for item in listed_cycles
        )
        cadence_bucket = horizon_research_cadence_bucket(
            horizon_profile, str(dataset.get("end_date") or "")
        )
        continuing_cycle: dict[str, Any] | None = None
        continuing_dataset: dict[str, Any] | None = None
        created = 0
        failed = 0
        # Each weekly/monthly/quarterly event stays on the immutable publication
        # that started it.  A newer daily snapshot in that same bucket waits for
        # the in-flight event to finish; a later bucket supersedes ordinary work.
        # A cycle that already produced a joint winner is different: its one-shot
        # capital OOS remains bound to the old vintage and advances independently.
        for stale_cycle in listed_cycles:
            if (
                stale_cycle.get("status") != "active"
                or stale_cycle.get("horizon_profile") != horizon_profile
                or str(stale_cycle.get("dataset_identity_sha256") or "")
                == latest_identity
            ):
                continue
            stale_dataset = available_by_name.get(str(stale_cycle.get("dataset") or ""))
            if stale_dataset is None:
                self.store.set_cycle_state(
                    str(stale_cycle["id"]),
                    state={
                        **dict(stale_cycle.get("state") or {}),
                        "blockers": [
                            "bound Qlib dataset is no longer available; recovery requires review"
                        ],
                    },
                    stage="dataset_unavailable",
                    status="blocked",
                    error="bound Qlib dataset is unavailable",
                    finished=True,
                )
                failed += 1
                continue
            if _cycle_has_capital_commitment(stale_cycle):
                capital_created, capital_failed = self._advance_capital_cycle(
                    stale_cycle,
                    stale_dataset,
                )
                created += capital_created
                failed += capital_failed
            elif (
                not latest_cycle_is_active
                and continuing_cycle is None
                and horizon_research_cadence_bucket(
                    horizon_profile,
                    str(
                        (stale_cycle.get("state") or {}).get("dataset_end_date")
                        or stale_dataset.get("end_date")
                        or ""
                    ),
                )
                == cadence_bucket
            ):
                # One immutable publication owns the complete research event.
                # A newer daily snapshot inside the same week/month/quarter
                # cannot pause it and then suppress its replacement via the
                # cadence gate.  Finish this cycle; the latest publication is
                # consumed after the owner reaches a terminal state.
                continuing_cycle = stale_cycle
                continuing_dataset = stale_dataset
            else:
                self.store.supersede_research_cycle(
                    str(stale_cycle["id"]),
                    replacement_dataset=dataset,
                )
        if continuing_cycle is not None and continuing_dataset is not None:
            cycle = continuing_cycle
            dataset = continuing_dataset
        else:
            cycle = self.store.ensure_cycle(
                dataset,
                config_revision=revision,
                horizon_profile=horizon_profile,
            )
        if cycle.get("status") != "active":
            return {
                "cycles": 1,
                "branches": created,
                "failed": failed + int(cycle.get("status") == "blocked"),
            }
        factor_due = self._factor_due(
            dataset,
            current,
            config,
            horizon_profile=horizon_profile,
        )
        model_due = self._model_due(
            dataset,
            current,
            config,
            horizon_profile=horizon_profile,
        )
        cycle = self.store.get_cycle(str(cycle["id"]))
        research_already_started = bool(cycle.get("branches")) or bool(
            (cycle.get("state") or {}).get("research_tournament_id")
        )
        if not factor_due and not model_due and not research_already_started:
            self.store.set_cycle_state(
                str(cycle["id"]),
                state={
                    **dict(cycle.get("state") or {}),
                    "result": "research_cadence_not_due",
                    "research_cadence_bucket": horizon_research_cadence_bucket(
                        horizon_profile, str(dataset.get("end_date") or "")
                    ),
                },
                stage="complete",
                status="succeeded",
                finished=True,
            )
            return {"cycles": 1, "branches": created, "failed": failed}
        cycle = self._roll_forward_prediction_champion(cycle, dataset)
        if cycle.get("status") == "succeeded":
            return {"cycles": 1, "branches": created, "failed": failed}
        sota_feature_set_id = self._active_sota_feature_set_id(
            dataset, horizon_profile=horizon_profile
        )
        cycle = self._ensure_horizon_factor_bundle(
            cycle,
            dataset,
            feature_set_id=(
                sota_feature_set_id or str(config["model_feature_set_id"])
            ),
        )
        revalidation_pending = self._current_identity_revalidation_pending(cycle, dataset)
        try:
            existing_model_tournament = self.tournaments.get_for_cycle(
                str(cycle["id"])
            )
        except KeyError:
            existing_model_tournament = None
        if (
            revalidation_pending
            and not model_due
            and existing_model_tournament is None
        ):
            revalidation_created, revalidation_failed = (
                self._ensure_current_identity_revalidation(cycle, dataset)
            )
            created += revalidation_created
            failed += revalidation_failed
            self._reconcile_current_identity_revalidation(
                self.store.get_cycle(str(cycle["id"])), dataset
            )
        elif model_due:
            self.tournaments.ensure_preregistered(
                cycle_id=str(cycle["id"]),
                dataset_identity_sha256=str(
                    dataset["provenance"]["dataset_identity_sha256"]
                ),
                active_sota_feature_set_id=sota_feature_set_id,
            )
        try:
            tournament = (
                None
                if revalidation_pending
                and not model_due
                and existing_model_tournament is None
                else self.tournaments.get_for_cycle(str(cycle["id"]))
            )
        except KeyError:
            tournament = None
        factor_branch = self.store.branch_for_scope(cycle["id"], "fin_factor", "daily")
        if factor_due and factor_branch is None:
            try:
                self._enqueue(cycle, dataset, "fin_factor", "daily", config=config)
                created += 1
            except ValueError:
                failed += 1
        elif self._retry_failed_branch(factor_branch):
            created += 1
        if tournament is not None:
            platform_created, platform_failed = self._ensure_platform_model_branches(
                cycle,
                dataset,
                tournament,
                stage="feature_screen",
            )
            created += platform_created
            failed += platform_failed
            self._reconcile_model_tournament(cycle, tournament)
            selected_feature_set_ids = self._screen_selected_feature_sets(
                cycle, tournament
            )
            if selected_feature_set_ids:
                full_created, full_failed = self._ensure_platform_model_branches(
                    self.store.get_cycle(str(cycle["id"])),
                    dataset,
                    self.tournaments.get_for_cycle(str(cycle["id"])),
                    stage="model_full",
                    eligible_feature_set_ids=set(selected_feature_set_ids),
                )
                created += full_created
                failed += full_failed
                self._reconcile_model_tournament(
                    self.store.get_cycle(str(cycle["id"])),
                    self.tournaments.get_for_cycle(str(cycle["id"])),
                )
                created += self._enqueue_next_rdagent_model_challenger(
                    self.store.get_cycle(str(cycle["id"])),
                    dataset,
                    selected_feature_set_ids,
                    config=config,
                )
                self._reconcile_dynamic_model_trials(
                    self.store.get_cycle(str(cycle["id"])),
                    self.tournaments.get_for_cycle(str(cycle["id"])),
                )
                champions = self._model_champions(
                    self.store.get_cycle(str(cycle["id"])),
                    self.tournaments.get_for_cycle(str(cycle["id"])),
                    selected_feature_set_ids,
                )
                ensemble_created, ensemble_failed = self._reconcile_model_ensembles(
                    self.store.get_cycle(str(cycle["id"])),
                    dataset,
                    self.tournaments.get_for_cycle(str(cycle["id"])),
                    champions,
                )
                created += ensemble_created
                failed += ensemble_failed
        cycle_branches = self.store.get_cycle(cycle["id"])["branches"]
        report_active = any(
            branch["scenario"] == "fin_factor_report"
            and branch["status"] in {"queued", "running", "evaluating"}
            for branch in cycle_branches
        )
        report_retry = next(
            (
                branch
                for branch in cycle_branches
                if branch["scenario"] == "fin_factor_report"
                and branch["status"] == "failed"
                and int((branch.get("details") or {}).get("retry_count") or 0) < 1
            ),
            None,
        )
        if horizon_profile == SHORT_1_5D:
            try:
                if report_active:
                    pass
                elif report_retry is not None:
                    created += int(self._retry_failed_branch(report_retry))
                else:
                    created += self._enqueue_reports(
                        cycle, dataset, config=config, now=current
                    )
            except ValueError:
                failed += 1
        factor_sota_branches = [
            branch
            for branch in self.store.get_cycle(cycle["id"])["branches"]
            if branch["scenario"] == "factor_sota"
        ]
        active_factor_sota = next(
            (
                branch
                for branch in factor_sota_branches
                if branch["status"] in {"queued", "running", "evaluating"}
            ),
            None,
        )
        if active_factor_sota is None:
            retried = False
            for branch in factor_sota_branches:
                if branch["status"] == "failed" and self._retry_failed_branch(branch):
                    created += 1
                    retried = True
                    break
            if not retried:
                try:
                    lane = self.factor_autopilot.ensure_incremental_lane(
                        cycle=cycle,
                        dataset=dataset,
                        horizon_profile=horizon_profile,
                    )
                    if lane is not None:
                        self.store.create_branch(
                            str(cycle["id"]),
                            scenario="factor_sota",
                            scope_key=str(lane["scope_key"]),
                            research_run_id=str(lane["run"]["id"]),
                            job_id=str(lane["job"]["id"]),
                            details={
                                "branch_kind": "factor_incremental_ablation",
                                "candidate_ids": list(lane["candidate_ids"]),
                                "frozen_model_sha256": lane["frozen_model_sha256"],
                                "final_oos_opened": False,
                            },
                        )
                        created += 1
                except ValueError:
                    failed += 1
        quant_branch = self.store.branch_for_scope(cycle["id"], "fin_quant", "joint")
        quant_input_sha256 = self._quant_input_sha256(
            self.store.get_cycle(str(cycle["id"])), dataset
        )
        if (
            self._quant_due(
                self.store.get_cycle(str(cycle["id"])),
                dataset,
                current,
                config,
                input_sha256=quant_input_sha256,
            )
            and quant_branch is None
        ):
            try:
                quant_cycle = self.store.get_cycle(str(cycle["id"]))
                identity_error = _prediction_champion_identity_error(
                    quant_cycle, dataset
                )
                if identity_error is not None:
                    self.store.patch_cycle_state(
                        str(cycle["id"]),
                        state_patch={
                            "prediction_champion_status": (
                                "pending_current_identity_revalidation"
                            ),
                            "fin_quant_status": (
                                "pending_current_identity_revalidation"
                            ),
                            "fin_quant_blocker": identity_error,
                        },
                    )
                    raise ValueError(identity_error)
                prediction_champion = dict(
                    (quant_cycle.get("state") or {}).get(
                        "prediction_champion"
                    )
                    or {}
                )
                prediction_champion_evidence = dict(
                    (quant_cycle.get("state") or {}).get(
                        "prediction_champion_evidence"
                    )
                    or {}
                )
                parent_tournament_id = str(
                    (quant_cycle.get("state") or {}).get(
                        "research_tournament_id"
                    )
                    or ""
                )
                if not parent_tournament_id:
                    raise ValueError(
                        "fin_quant cannot start before the model tournament is sealed"
                    )
                champion_feature_set_id = str(
                    prediction_champion.get("primary_feature_set_id")
                    or config["quant_feature_set_id"]
                )
                self._enqueue(
                    cycle,
                    dataset,
                    "fin_quant",
                    "joint",
                    config=config,
                    feature_set_id=champion_feature_set_id,
                    tournament_id=parent_tournament_id,
                    branch_details={
                        "quant_input_sha256": quant_input_sha256,
                        "model_champion_feature_set_id": champion_feature_set_id,
                        "prediction_champion_kind": prediction_champion.get("kind"),
                        "prediction_champion_id": prediction_champion.get("candidate_id"),
                        "prediction_champion_primary_model_candidate_id": (
                            prediction_champion.get("primary_model_candidate_id")
                        ),
                        "prediction_champion": prediction_champion,
                        "prediction_champion_evidence": (
                            prediction_champion_evidence
                        ),
                    },
                )
                created += 1
            except ValueError:
                failed += 1
        elif quant_branch is not None and self._retry_failed_branch(quant_branch):
            created += 1
        refreshed_cycle = self.store.get_cycle(str(cycle["id"]))
        refreshed_quant = self.store.branch_for_scope(
            str(cycle["id"]), "fin_quant", "joint"
        )
        if refreshed_quant is not None and refreshed_quant["status"] == "succeeded":
            capital_created, capital_failed = self._advance_capital_cycle(
                refreshed_cycle,
                dataset,
            )
            created += capital_created
            failed += capital_failed
            refreshed_cycle = self.store.get_cycle(str(cycle["id"]))
        # A daily data publication must not remain the global active cycle
        # merely because no monthly model tournament was due.  Once every
        # research branch for this immutable vintage is terminal and no new
        # work was created on this tick, close the research-only cycle so the
        # next publication can be consumed.  A genuine execution failure is
        # surfaced as blocked; a clean run with no new champion is a valid
        # no-op, not a reason to reopen final OOS.
        current_cycle = self.store.get_cycle(str(cycle["id"]))
        current_branches = list(current_cycle.get("branches") or [])
        no_active_branch = all(
            str(item.get("status") or "") in _TERMINAL_BRANCH_STATUSES
            for item in current_branches
        )
        current_quant = next(
            (item for item in current_branches if item.get("scenario") == "fin_quant"),
            None,
        )
        try:
            terminal_tournament = self.tournaments.get_for_cycle(str(cycle["id"]))
        except KeyError:
            revalidation = dict(
                (current_cycle.get("state") or {}).get(
                    "current_identity_revalidation"
                )
                or {}
            )
            revalidation_id = str(revalidation.get("tournament_id") or "")
            try:
                terminal_tournament = (
                    self.tournaments.get_tournament(revalidation_id)
                    if revalidation_id
                    else None
                )
            except KeyError:
                terminal_tournament = None
        tournament_is_terminal = terminal_tournament is None or str(
            terminal_tournament.get("status") or ""
        ) in {"succeeded", "blocked", "failed", "cancelled"}
        if (
            current_cycle.get("status") == "active"
            and no_active_branch
            and current_quant is None
            and tournament_is_terminal
            and created == 0
        ):
            branch_failures = [
                item
                for item in current_branches
                if item.get("status") in {"failed", "blocked"}
            ]
            tournament_blocked = bool(
                terminal_tournament is not None
                and str(terminal_tournament.get("status") or "")
                in {"blocked", "failed", "cancelled"}
            )
            blocked = bool(branch_failures or failed or tournament_blocked)
            state = {
                **dict(current_cycle.get("state") or {}),
                "result": (
                    "research_execution_blocked"
                    if blocked
                    else "research_complete_no_joint_update"
                ),
                "runner_up_allowed": False,
            }
            if branch_failures:
                state["blockers"] = sorted(
                    {
                        str(item.get("error") or item.get("scenario") or "research failed")
                        for item in branch_failures
                    }
                )
            elif tournament_blocked:
                state["blockers"] = [
                    str(
                        terminal_tournament.get("blocked_reason")
                        or terminal_tournament.get("error")
                        or "model tournament did not produce a valid champion"
                    )
                ]
            self.store.set_cycle_state(
                str(cycle["id"]),
                state=state,
                stage="research_blocked" if blocked else "complete",
                status="blocked" if blocked else "succeeded",
                error=("one or more research branches failed" if blocked else None),
                finished=True,
            )
        return {"cycles": 1, "branches": created, "failed": failed}

    def _advance_capital_cycle(
        self,
        cycle: dict[str, Any],
        dataset: dict[str, Any],
    ) -> tuple[int, int]:
        """Advance one immutable capital attempt without reopening research."""

        if str(cycle.get("dataset_identity_sha256") or "") != str(
            (dataset.get("provenance") or {}).get("dataset_identity_sha256") or ""
        ):
            raise ValueError("capital cycle and bound Qlib dataset identities differ")
        quant_branch = next(
            (
                item
                for item in list(cycle.get("branches") or [])
                if item.get("scenario") == "fin_quant"
                and item.get("status") == "succeeded"
            ),
            None,
        )
        if quant_branch is None or cycle.get("status") == "succeeded":
            return 0, 0
        try:
            capital_progress = self.capital_pipeline.advance(
                cycle=cycle,
                dataset=dataset,
                quant_branch=quant_branch,
            )
        except (AutopilotCapitalBlocked, ValueError) as exc:
            root_state = dict(cycle.get("state") or {})
            blockers = list(root_state.get("blockers") or [])
            message = str(exc)
            if message not in blockers:
                blockers.append(message)
            root_state.update(
                {
                    "blockers": blockers,
                    "capital_pipeline_error": message,
                    "runner_up_allowed": False,
                }
            )
            self.store.set_cycle_state(
                str(cycle["id"]),
                state=root_state,
                stage="capital_gate_blocked",
                status="blocked",
                error=message,
                finished=True,
            )
            return 0, 1
        root_state = {
            **dict(cycle.get("state") or {}),
            CAPITAL_PIPELINE_STATE_KEY: capital_progress.state,
        }
        self.store.set_cycle_state(
            str(cycle["id"]),
            state=root_state,
            stage=capital_progress.stage,
            status="succeeded" if capital_progress.complete else "active",
            finished=capital_progress.complete,
        )
        return int(capital_progress.created_jobs), 0

    def _retry_failed_branch(self, branch: dict[str, Any]) -> bool:
        if branch.get("status") != "failed":
            return False
        if int((branch.get("details") or {}).get("retry_count") or 0) >= 1:
            return False
        run_id = str(branch.get("research_run_id") or "")
        job_id = str(branch.get("job_id") or "")
        if not run_id or not job_id:
            return False
        run = self.research.get_run(run_id)
        job = self.jobs.get(job_id)
        try:
            if run["status"] in {"failed", "cancelled"}:
                self.research.requeue_run(run_id, actor="autopilot")
            elif run["status"] != "queued":
                return False
            if job["status"] in {"failed", "cancelled"}:
                self.jobs.retry(job_id)
            elif job["status"] != "queued":
                return False
            self.store.mark_branch_retried(str(branch["id"]))
        except Exception as exc:
            self.research.mark_run(
                run_id,
                "failed",
                actor="autopilot",
                error=f"autopilot retry failed: {exc}",
            )
            raise
        return True

    def _latest_dataset(self) -> dict[str, Any] | None:
        candidates = [
            item
            for item in list_qlib_datasets(self.settings.data_root)
            if item.get("ready")
            and item.get("reproducible")
            and item.get("lineage_verified")
            and item.get("lineage_id")
            and item.get("frequency") == "day"
        ]
        return max(
            candidates,
            key=lambda item: (str(item.get("end_date") or ""), str(item["name"])),
            default=None,
        )

    def _factor_due(
        self,
        dataset: dict[str, Any],
        now: datetime,
        config: dict[str, Any],
        *,
        horizon_profile: str = SHORT_1_5D,
    ) -> bool:
        del now, config
        return self._horizon_branch_due(
            "fin_factor", dataset, horizon_profile=horizon_profile
        )

    def _model_due(
        self,
        dataset: dict[str, Any],
        now: datetime,
        config: dict[str, Any],
        *,
        horizon_profile: str = SHORT_1_5D,
    ) -> bool:
        del now, config
        return self._horizon_branch_due(
            "fin_model", dataset, horizon_profile=horizon_profile
        )

    def _horizon_branch_due(
        self,
        scenario: str,
        dataset: dict[str, Any],
        *,
        horizon_profile: str,
    ) -> bool:
        latest = self.store.latest_branch(
            scenario, horizon_profile=horizon_profile
        )
        if latest is None:
            return True
        try:
            source_cycle = self.store.get_cycle(str(latest["cycle_id"]))
            source_end = str((source_cycle.get("state") or {}).get("dataset_end_date") or "")
        except (KeyError, TypeError, ValueError):
            return True
        if source_cycle.get("horizon_profile") != horizon_profile:
            raise ValueError("automatic research branch belongs to another horizon")
        return horizon_research_cadence_bucket(
            horizon_profile, source_end
        ) != horizon_research_cadence_bucket(
            horizon_profile, str(dataset.get("end_date") or "")
        )

    def _ensure_platform_model_branches(
        self,
        cycle: dict[str, Any],
        dataset: dict[str, Any],
        tournament: dict[str, Any],
        *,
        stage: str,
        eligible_feature_set_ids: set[str] | None = None,
    ) -> tuple[int, int]:
        if stage not in {"feature_screen", "model_full"}:
            raise ValueError("platform model stage is invalid")
        stage_trials = [
            item
            for item in tournament["trials"]
            if (item.get("spec") or {}).get("round") == stage
            and (
                eligible_feature_set_ids is None
                or str(item.get("feature_set_id")) in eligible_feature_set_ids
            )
        ]
        by_feature: dict[str, list[dict[str, Any]]] = {}
        for trial in stage_trials:
            by_feature.setdefault(str(trial["feature_set_id"]), []).append(trial)
        calendar = (
            (Path(str(dataset["path"])) / "calendars" / "day.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        horizon_profile = str(cycle.get("horizon_profile") or "")
        primary_label = primary_label_horizon_sessions(horizon_profile)
        policy = primary_label_policy_contract()
        created = 0
        failed = 0
        for feature_set_id, trials in by_feature.items():
            feature_set = get_feature_set(feature_set_id)
            periods, resolution = resolve_research_window_contract(
                dataset,
                calendar,
                horizon_profile=horizon_profile,
                feature_set=feature_set,
            )
            scope = f"platform:{stage}:{feature_set_id}"
            branch = self.store.branch_for_scope(str(cycle["id"]), "fin_model", scope)
            if branch is not None:
                if branch.get("status") == "failed":
                    created += int(self._retry_failed_branch(branch))
                continue
            try:
                lane = self.platform_models.ensure_lane(
                    cycle_id=str(cycle["id"]),
                    tournament_id=str(tournament["id"]),
                    dataset=dataset,
                    stage=stage,
                    feature_set_id=feature_set_id,
                    periods=periods,
                    evaluation_profiles=resolution["evaluation_profiles"],
                    horizon_profile=horizon_profile,
                    label_horizon_sessions=primary_label,
                    primary_label_policy=policy,
                    research_window_contract=resolution[
                        "research_window_contract"
                    ],
                    research_window_contract_sha256=resolution[
                        "research_window_contract_sha256"
                    ],
                    trials=trials,
                )
                bindings: list[dict[str, str]] = []
                for binding in lane["bindings"]:
                    self._advance_tournament_trial(
                        binding["trial_id"],
                        "queued",
                        candidate_id=binding["candidate_id"],
                    )
                    bindings.append(dict(binding))
                self.tournaments.mark_running(str(tournament["id"]))
                self.store.create_branch(
                    str(cycle["id"]),
                    scenario="fin_model",
                    scope_key=scope,
                    research_run_id=str(lane["run"]["id"]),
                    job_id=str(lane["job"]["id"]),
                    details={
                        "branch_kind": f"platform_model_{stage}",
                        "tournament_stage": stage,
                        "feature_set_id": feature_set_id,
                        "tournament_id": str(tournament["id"]),
                        "horizon_profile": horizon_profile,
                        "label_horizon_sessions": primary_label,
                        "primary_label_policy_sha256": policy[
                            "policy_sha256"
                        ],
                        "candidate_bindings": bindings,
                        "final_oos_opened": False,
                    },
                )
                created += 1
            except ValueError:
                failed += 1
        return created, failed

    def _current_identity_revalidation_pending(
        self, cycle: dict[str, Any], dataset: dict[str, Any]
    ) -> bool:
        """Whether this daily vintage needs the frozen incumbent retrained.

        This deliberately keys off the rollover state, rather than the month
        clock: a new dataset must receive a current prediction champion even
        when the next full `fin_model` contest is weeks away.
        """

        state = dict(cycle.get("state") or {})
        return (
            state.get("prediction_champion_status")
            == "pending_current_identity_revalidation"
            and _prediction_champion_identity_error(cycle, dataset) is not None
            and isinstance(state.get("prior_prediction_champion"), dict)
        )

    def _champion_revalidation_source(
        self, cycle: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        """Resolve the *old* immutable winner into fixed component recipes."""

        state = dict(cycle.get("state") or {})
        champion = dict(state.get("prior_prediction_champion") or {})
        selection = dict(state.get("prior_prediction_champion_evidence") or {})
        model_selection = dict(state.get("prior_model_champion_evidence") or {})
        if (
            champion.get("kind") not in {"model", "ensemble"}
            or not champion.get("candidate_id")
            or canonical_sha256(
                {key: value for key, value in selection.items() if key != "evidence_sha256"}
            )
            != str(selection.get("evidence_sha256") or "")
            or canonical_sha256(
                {
                    key: value
                    for key, value in model_selection.items()
                    if key != "evidence_sha256"
                }
            )
            != str(model_selection.get("evidence_sha256") or "")
        ):
            raise ValueError("prior prediction champion evidence is unavailable")

        source_rows: list[dict[str, Any]]
        if champion["kind"] == "model":
            trial_family = ""
            if champion.get("trial_id"):
                try:
                    trial_family = str(
                        self.tournaments.get_trial(str(champion["trial_id"])).get(
                            "model_family"
                        )
                        or ""
                    )
                except KeyError:
                    trial_family = ""
            source_rows = [
                {
                    "model_candidate_id": str(champion["candidate_id"]),
                    "model_family": str(champion.get("model_family") or trial_family),
                    "weight": 1.0,
                    "model_manifest_sha256": str(champion.get("manifest_sha256") or ""),
                    "model_admission_evidence_sha256": str(
                        champion.get("admission_evidence_sha256") or ""
                    ),
                }
            ]
        else:
            ensemble = self.tournaments.get_ensemble(
                str(champion["candidate_id"]), verify=True
            )
            if (
                str(ensemble.get("manifest_sha256") or "")
                != str(champion.get("manifest_sha256") or "")
                or str(ensemble.get("admission_evidence_sha256") or "")
                != str(champion.get("admission_evidence_sha256") or "")
            ):
                raise ValueError("prior ensemble champion changed after selection")
            source_rows = [dict(item) for item in ensemble.get("components") or []]

        components: list[dict[str, Any]] = []
        for raw in source_rows:
            candidate_id = str(raw.get("model_candidate_id") or "")
            candidate = self.platform_models.candidates.get_model_candidate(
                candidate_id, verify=True
            )
            manifest = dict(candidate.get("manifest_json") or {})
            base_features = dict(candidate.get("base_features_manifest_json") or {})
            feature_set_id = str(base_features.get("feature_set_id") or "")
            feature_set = get_feature_set(feature_set_id)
            if (
                str(candidate.get("status") or "") != "research_admitted"
                or str(candidate.get("manifest_sha256") or "")
                != str(raw.get("model_manifest_sha256") or "")
                or str(candidate.get("admission_evidence_sha256") or "")
                != str(raw.get("model_admission_evidence_sha256") or "")
                or not str(manifest.get("recipe_sha256") or "")
            ):
                raise ValueError("prior champion component is no longer immutable")
            family = str(raw.get("model_family") or champion.get("model_family") or "")
            if not family:
                raise ValueError("prior champion component family is missing")
            components.append(
                {
                    "source_model_candidate_id": candidate_id,
                    "model_family": family,
                    "feature_set_id": feature_set_id,
                    "feature_set_definition_sha256": str(feature_set["definition_sha256"]),
                    "source_model_manifest_sha256": str(candidate["manifest_sha256"]),
                    "source_recipe_sha256": str(manifest["recipe_sha256"]),
                    "source_code_sha256": str(candidate["code_sha256"]),
                    "weight": float(raw.get("weight") or 0.0),
                }
            )
        return champion, selection, model_selection, components

    def _block_current_identity_revalidation(
        self, cycle: dict[str, Any], tournament_id: str | None, reason: str
    ) -> None:
        if tournament_id:
            tournament = self.tournaments.get_tournament(tournament_id)
            if str(tournament.get("status") or "") not in {"succeeded", "blocked"}:
                self.tournaments.block(tournament_id, reason=reason)
        self.store.patch_cycle_state(
            str(cycle["id"]),
            state_patch={
                "prediction_champion_status": "blocked_current_identity_revalidation",
                "fin_quant_status": "blocked_current_identity_revalidation",
                "fin_quant_blocker": reason,
                "current_identity_revalidation": {
                    "status": "blocked",
                    "reason": reason,
                    "final_oos_opened": False,
                    "research_screening_only": True,
                },
            },
            stage="model_revalidation_blocked",
        )

    def _ensure_current_identity_revalidation(
        self, cycle: dict[str, Any], dataset: dict[str, Any]
    ) -> tuple[int, int]:
        """Create fixed-recipe current-vintage jobs, grouped only by feature set."""

        try:
            champion, selection, model_selection, components = (
                self._champion_revalidation_source(cycle)
            )
            identity = str(dataset["provenance"]["dataset_identity_sha256"])
            tournament = self.tournaments.ensure_champion_revalidation_preregistered(
                cycle_id=str(cycle["id"]),
                dataset_identity_sha256=identity,
                source_champion=champion,
                source_selection_evidence_sha256=str(selection["evidence_sha256"]),
                source_model_evidence_sha256=str(model_selection["evidence_sha256"]),
                components=components,
            )
            calendar = (
                (Path(str(dataset["path"])) / "calendars" / "day.txt")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            horizon_profile = str(cycle.get("horizon_profile") or "")
            primary_label = primary_label_horizon_sessions(horizon_profile)
            policy = primary_label_policy_contract()
            trials_by_source = {
                str((item.get("spec") or {}).get("source_model_candidate_id") or ""): item
                for item in tournament["trials"]
            }
            groups: dict[str, list[dict[str, Any]]] = {}
            for component in components:
                groups.setdefault(str(component["feature_set_id"]), []).append(component)
            created = 0
            for feature_set_id, group in sorted(groups.items()):
                feature_set = get_feature_set(feature_set_id)
                periods, resolution = resolve_research_window_contract(
                    dataset,
                    calendar,
                    horizon_profile=horizon_profile,
                    feature_set=feature_set,
                )
                scope = f"revalidation:{feature_set_id}"
                branch = self.store.branch_for_scope(cycle["id"], "fin_model", scope)
                if branch is not None:
                    continue
                trials = [
                    trials_by_source[str(item["source_model_candidate_id"])]
                    for item in group
                ]
                lane = self.platform_models.ensure_champion_revalidation_lane(
                    cycle_id=str(cycle["id"]),
                    tournament_id=str(tournament["id"]),
                    dataset=dataset,
                    periods=periods,
                    evaluation_profiles=resolution["evaluation_profiles"],
                    horizon_profile=horizon_profile,
                    label_horizon_sessions=primary_label,
                    primary_label_policy=policy,
                    research_window_contract=resolution[
                        "research_window_contract"
                    ],
                    research_window_contract_sha256=resolution[
                        "research_window_contract_sha256"
                    ],
                    source_components=group,
                    trials=trials,
                    source_manifest_sha256=str(tournament["manifest_sha256"]),
                )
                bindings = [dict(item) for item in lane["bindings"]]
                for binding in bindings:
                    self._advance_tournament_trial(
                        str(binding["trial_id"]),
                        "queued",
                        candidate_id=str(binding["candidate_id"]),
                    )
                self.store.create_branch(
                    str(cycle["id"]),
                    scenario="fin_model",
                    scope_key=scope,
                    research_run_id=str(lane["run"]["id"]),
                    job_id=str(lane["job"]["id"]),
                    details={
                        "branch_kind": "champion_current_identity_revalidation",
                        "tournament_id": str(tournament["id"]),
                        "feature_set_id": feature_set_id,
                        "horizon_profile": horizon_profile,
                        "label_horizon_sessions": primary_label,
                        "primary_label_policy_sha256": policy[
                            "policy_sha256"
                        ],
                        "candidate_bindings": bindings,
                        "final_oos_opened": False,
                        "research_screening_only": True,
                        "not_capital_confirmation": True,
                    },
                )
                created += 1
            self.tournaments.mark_running(str(tournament["id"]))
            self.store.patch_cycle_state(
                str(cycle["id"]),
                state_patch={
                    "current_identity_revalidation": {
                        "status": "running",
                        "tournament_id": str(tournament["id"]),
                        "source_prediction_champion_id": str(champion["candidate_id"]),
                        "source_prediction_champion_kind": str(champion["kind"]),
                        "component_count": len(components),
                        "profiles": list(FULL_PROFILES),
                        "seeds": list(FULL_SEEDS),
                        "final_oos_opened": False,
                        "research_screening_only": True,
                    }
                },
                stage="current_identity_revalidation",
            )
            return created, 0
        except (KeyError, OSError, ValueError) as exc:
            reason = f"current-identity champion revalidation blocked: {exc}"
            self._block_current_identity_revalidation(cycle, None, reason)
            return 0, 1

    def _reconcile_current_identity_revalidation(
        self, cycle: dict[str, Any], dataset: dict[str, Any]
    ) -> None:
        state = dict(cycle.get("state") or {})
        progress = dict(state.get("current_identity_revalidation") or {})
        tournament_id = str(progress.get("tournament_id") or "")
        if not tournament_id:
            return
        tournament = self.tournaments.get_tournament(tournament_id)
        if str(tournament.get("status") or "") in {"blocked", "failed", "cancelled"}:
            self._block_current_identity_revalidation(
                cycle, tournament_id, str(tournament.get("blocked_reason") or "revalidation failed")
            )
            return
        bindings: list[dict[str, Any]] = []
        terminal_failure: str | None = None
        for branch in self.store.get_cycle(str(cycle["id"]))["branches"]:
            details = dict(branch.get("details") or {})
            if details.get("branch_kind") != "champion_current_identity_revalidation":
                continue
            run = self.research.get_run(str(branch["research_run_id"]))
            for binding in details.get("candidate_bindings") or []:
                entry = dict(binding)
                bindings.append(entry)
                evidence = self._candidate_tournament_evidence(str(entry["candidate_id"]))
                candidate_status = str(evidence["candidate_status"])
                if candidate_status == "research_admitted":
                    self._advance_tournament_trial(
                        str(entry["trial_id"]), "passed", candidate_id=str(entry["candidate_id"]),
                        metrics={"cells": evidence["cells"]}, evidence=evidence
                    )
                elif candidate_status in {"rejected", "invalidated"}:
                    self._advance_tournament_trial(
                        str(entry["trial_id"]), "rejected", candidate_id=str(entry["candidate_id"]),
                        metrics={"cells": evidence["cells"]}, evidence=evidence
                    )
                    terminal_failure = "frozen champion component failed current-identity gates"
                elif str(run["status"]) == "blocked":
                    terminal_failure = "current-identity model evaluation was resource blocked"
                elif str(run["status"]) in {"failed", "cancelled"} and int(
                    details.get("retry_count") or 0
                ) >= 1:
                    terminal_failure = "current-identity model evaluation failed after retry"
                elif str(run["status"]) in {"running", "evaluating"}:
                    self._advance_tournament_trial(
                        str(entry["trial_id"]), "running", candidate_id=str(entry["candidate_id"])
                    )
        if terminal_failure:
            self._block_current_identity_revalidation(cycle, tournament_id, terminal_failure)
            return
        if not bindings:
            return
        refreshed = self.tournaments.get_tournament(tournament_id)
        trials = [
            item for item in refreshed["trials"] if item.get("trial_kind") == "model"
        ]
        if any(str(item.get("status") or "") not in {"passed", "selected"} for item in trials):
            return
        try:
            self._finalize_current_identity_revalidation(cycle, dataset, refreshed, bindings)
        except (KeyError, OSError, ValueError, EnsemblePredictionsPending) as exc:
            self._block_current_identity_revalidation(
                cycle, tournament_id, f"current-identity revalidation evidence failed closed: {exc}"
            )

    def _candidate_return_grid(
        self, candidate_id: str, definition: dict[str, str]
    ) -> dict[str, list[tuple[dict[str, str], pd.Series]]]:
        candidate = self.platform_models.candidates.get_model_candidate(candidate_id, verify=True)
        admission = dict(candidate.get("admission_evidence_json") or {})
        grids: dict[str, list[tuple[dict[str, str], pd.Series]]] = {
            profile_id: [] for profile_id in REQUIRED_RESEARCH_PROFILES
        }
        for profile_id in REQUIRED_RESEARCH_PROFILES:
            profile = dict((admission.get("profiles") or {}).get(profile_id) or {})
            returns: list[pd.Series] = []
            for seed in REQUIRED_MODEL_SEEDS:
                cell = dict((profile.get("seeds") or {}).get(str(seed)) or {})
                path = Path(
                    str(cell.get("portfolio_report_path") or "")
                ).resolve()
                if not path.is_file() or file_sha256(path) != str(
                    cell.get("portfolio_report_sha256") or ""
                ):
                    raise ValueError("current-identity model report evidence changed")
                report = pd.read_parquet(path)
                if not {"return", "bench", "cost"}.issubset(report.columns):
                    raise ValueError("current-identity model report is incomplete")
                returns.append(
                    pd.to_numeric(report["return"], errors="coerce")
                    - pd.to_numeric(report["bench"], errors="coerce")
                    - pd.to_numeric(report["cost"], errors="coerce")
                )
            grids[profile_id].append(
                (definition, pd.concat(returns, axis=1, join="inner").mean(axis=1))
            )
        return grids

    def _finalize_current_identity_revalidation(
        self,
        cycle: dict[str, Any],
        dataset: dict[str, Any],
        tournament: dict[str, Any],
        bindings: list[dict[str, Any]],
    ) -> None:
        """Settle the fixed revalidation ledger without opening final OOS."""

        manifest = dict(tournament.get("manifest") or {})
        source = dict(manifest["source_champion"])
        bindings_by_source = {
            str(item["source_model_candidate_id"]): dict(item) for item in bindings
        }
        model_rows: list[dict[str, Any]] = []
        series: dict[str, list[tuple[dict[str, str], pd.Series]]] = {
            profile_id: [] for profile_id in REQUIRED_RESEARCH_PROFILES
        }
        for trial in tournament["trials"]:
            if trial.get("trial_kind") != "model":
                continue
            source_id = str((trial.get("spec") or {}).get("source_model_candidate_id") or "")
            binding = bindings_by_source[source_id]
            candidate_id = str(binding["candidate_id"])
            evidence = self._candidate_tournament_evidence(candidate_id)
            definition = {
                "name": str(trial["name"]),
                "candidate_id": candidate_id,
                "trial_id": str(trial["id"]),
                "feature_set_id": str(trial["feature_set_id"]),
                "model_family": str(trial["model_family"]),
            }
            grid = self._candidate_return_grid(candidate_id, definition)
            for profile_id, values in grid.items():
                series[profile_id].extend(values)
            model_rows.append(
                {
                    **definition,
                    "score": list(self._model_tournament_score(evidence)),
                    "candidate_manifest_sha256": str(evidence["candidate_manifest_sha256"]),
                    "admission_evidence_sha256": str(evidence["admission_evidence_sha256"]),
                }
            )
        multiple = _profile_family_multiple_testing(
            research_run_id=f"tournament:{tournament['id']}:champion_revalidation",
            trial_series_by_profile=series,
            family_definitions=[
                {
                    key: str(item[key])
                    for key in (
                        "name",
                        "candidate_id",
                        "trial_id",
                        "feature_set_id",
                        "model_family",
                    )
                }
                for item in model_rows
            ],
            output=(
                self.settings.data_root
                / "artifacts"
                / "model-tournaments"
                / str(tournament["id"])
                / "champion-revalidation-multiple-testing"
            ),
            contract_version="model-full-multiple-testing-v2",
        )
        if set(multiple["eligible_trial_names"]) != {str(item["name"]) for item in model_rows}:
            raise ValueError("frozen champion did not pass every current-identity model gate")
        model_evidence = {
            "contract_version": "model-family-champions-v2",
            "dataset_identity_sha256": str(tournament["dataset_identity_sha256"]),
            "champions": model_rows,
            "multiple_testing": multiple,
            "multiple_testing_evidence_sha256": str(multiple["evidence_sha256"]),
            "family_selection": "fixed_prior_champion_components",
            "seed_cells_are_robustness_repeats": True,
            "window_cells_are_not_independent_votes": True,
            "failed_and_rejected_trials_retained": True,
            "research_screening_only": True,
            "not_capital_confirmation": True,
            "final_oos_opened": False,
        }
        model_evidence["evidence_sha256"] = canonical_sha256(model_evidence)

        if source["kind"] == "model":
            winner = model_rows[0]
            champion = {
                "kind": "model",
                "candidate_id": winner["candidate_id"],
                "trial_id": winner["trial_id"],
                "score": winner["score"],
                "primary_model_candidate_id": winner["candidate_id"],
                "primary_feature_set_id": winner["feature_set_id"],
                "component_model_candidate_ids": [winner["candidate_id"]],
                "component_feature_set_ids": [winner["feature_set_id"]],
                "manifest_sha256": winner["candidate_manifest_sha256"],
                "admission_evidence_sha256": winner["admission_evidence_sha256"],
            }
        else:
            champion = self._current_identity_revalidation_ensemble(
                cycle, dataset, tournament, manifest, model_rows
            )
            if champion is None:
                return
        final_definition = {
            "name": f"prediction-champion:{champion['candidate_id']}",
            "candidate_id": str(champion["candidate_id"]),
            "trial_id": str(champion["trial_id"]),
            "kind": str(champion["kind"]),
        }
        final_series = self._winner_return_grid(champion, final_definition)
        final_multiple = _profile_family_multiple_testing(
            research_run_id=f"tournament:{tournament['id']}:prediction_finalist",
            trial_series_by_profile=final_series,
            family_definitions=[final_definition],
            output=(
                self.settings.data_root
                / "artifacts"
                / "model-tournaments"
                / str(tournament["id"])
                / "champion-revalidation-finalist"
            ),
            contract_version="prediction-finalist-multiple-testing-v2",
        )
        if final_definition["name"] not in set(final_multiple["eligible_trial_names"]):
            raise ValueError("current-identity prediction champion failed final gate")
        selection = {
            "contract_version": "prediction-champion-selection-v1",
            "dataset_identity_sha256": str(tournament["dataset_identity_sha256"]),
            "selection_data": "pre_final_only",
            "final_oos_opened": False,
            "selected_kind": champion["kind"],
            "selected_candidate_id": champion["candidate_id"],
            "selected_trial_ids": [champion["trial_id"]],
            "global_multiple_testing": final_multiple,
            "global_multiple_testing_evidence_sha256": final_multiple["evidence_sha256"],
            "models_and_ensembles_share_one_finalist_family": True,
            "fixed_prior_champion_current_identity_revalidation": True,
            "research_screening_only": True,
            "not_capital_confirmation": True,
        }
        selection["evidence_sha256"] = canonical_sha256(selection)
        self.tournaments.complete_selection(
            str(tournament["id"]),
            selected_trial_ids=[str(champion["trial_id"])],
            multiple_testing=final_multiple,
        )
        current_state = dict(
            self.store.get_cycle(str(cycle["id"])).get("state") or {}
        )
        old = dict(current_state.get("prediction_champion_roll_forward") or {})
        old.update(
            {
                "eligible_for_fin_quant": True,
                "revalidated_tournament_id": str(tournament["id"]),
                "revalidated_selection_evidence_sha256": selection["evidence_sha256"],
            }
        )
        self.store.patch_cycle_state(
            str(cycle["id"]),
            state_patch={
                "research_tournament_id": str(tournament["id"]),
                "model_champions": model_rows,
                "model_champion_evidence": model_evidence,
                "prediction_champion": champion,
                "prediction_champion_evidence": selection,
                "prediction_champion_status": "validated_current_identity_revalidation",
                "fin_quant_status": "ready",
                "fin_quant_blocker": None,
                "prediction_champion_roll_forward": old,
                "current_identity_revalidation": {
                    "status": "succeeded",
                    "tournament_id": str(tournament["id"]),
                    "final_oos_opened": False,
                    "research_screening_only": True,
                },
            },
            stage="joint_optimization",
        )

    def _current_identity_revalidation_ensemble(
        self,
        cycle: dict[str, Any],
        dataset: dict[str, Any],
        tournament: dict[str, Any],
        manifest: dict[str, Any],
        model_rows: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Rebuild exactly the frozen equal-rank ensemble on current predictions."""

        source_by_id = {
            str(item["source_model_candidate_id"]): dict(item)
            for item in manifest["components"]
        }
        by_family = {str(item["model_family"]): item for item in model_rows}
        components: list[dict[str, Any]] = []
        grids: list[dict[str, Any]] = []
        for source in sorted(
            source_by_id.values(), key=lambda item: str(item["model_family"])
        ):
            row = by_family[str(source["model_family"])]
            candidate = self.platform_models.candidates.get_model_candidate(
                str(row["candidate_id"]), verify=True
            )
            grid = prediction_grid_from_admission(candidate)
            grids.append(grid)
            components.append({
                "model_candidate_id": str(row["candidate_id"]),
                "model_family": str(row["model_family"]),
                "weight": float(source["weight"]),
                "model_manifest_sha256": str(row["candidate_manifest_sha256"]),
                "model_admission_evidence_sha256": str(row["admission_evidence_sha256"]),
                "prediction_grid_sha256": str(grid["prediction_grid_sha256"]),
            })
        correlations = [
            pairwise_grid_correlation(grids[left], grids[right])
            for left in range(len(grids)) for right in range(left + 1, len(grids))
        ]
        source_id = str(manifest["source_champion"]["candidate_id"])
        ensemble = self.tournaments.create_ensemble(
            tournament_id=str(tournament["id"]),
            name=f"champion-revalidation-{source_id[:16]}",
            dataset=str(dataset["name"]),
            dataset_identity_sha256=str(tournament["dataset_identity_sha256"]),
            components=components,
            prediction_correlations=[
                float(item["maximum_mean_absolute_daily_rank_correlation"])
                for item in correlations
            ],
            correlation_evidence=correlations,
        )
        trial = next(
            item
            for item in self.tournaments.get_tournament(str(tournament["id"]))[
                "trials"
            ]
            if str(item.get("candidate_id") or "") == str(ensemble["id"])
        )
        status = str(ensemble["status"])
        if status in {"awaiting_evaluation", "evaluating"}:
            calendar = (
                (Path(str(dataset["path"])) / "calendars" / "day.txt")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            _periods, resolution = resolve_research_periods(calendar)
            self.model_ensembles.queue_evaluation(
                tournament_id=str(tournament["id"]),
                dataset=dataset,
                evaluation_profiles=resolution["evaluation_profiles"],
            )
            return None
        if status != "research_admitted":
            raise ValueError("frozen current-identity ensemble did not pass independent gates")
        score = self._model_tournament_score(
            {
                "cells": [
                    {
                        "profile_id": item["profile_id"],
                        "seed": item["seed"],
                        "gate_status": item["gate_status"],
                        "metrics": item["metrics"],
                    }
                    for item in ensemble["evaluations"]
                ]
            }
        )
        self._advance_tournament_trial(
            str(trial["id"]),
            "passed",
            candidate_id=str(ensemble["id"]),
            metrics={"score": list(score)},
            evidence=dict(ensemble.get("admission_evidence_json") or {}),
        )
        primary = max(
            model_rows,
            key=lambda item: (tuple(item["score"]), str(item["candidate_id"])),
        )
        return {
            "kind": "ensemble",
            "candidate_id": str(ensemble["id"]),
            "trial_id": str(trial["id"]),
            "score": list(score),
            "primary_model_candidate_id": str(primary["candidate_id"]),
            "primary_feature_set_id": str(primary["feature_set_id"]),
            "component_model_candidate_ids": [
                str(item["model_candidate_id"]) for item in components
            ],
            "component_feature_set_ids": sorted(
                {str(item["feature_set_id"]) for item in model_rows}
            ),
            "manifest_sha256": str(ensemble["manifest_sha256"]),
            "admission_evidence_sha256": str(
                ensemble["admission_evidence_sha256"]
            ),
        }

    def _winner_return_grid(
        self, champion: dict[str, Any], definition: dict[str, str]
    ) -> dict[str, list[tuple[dict[str, str], pd.Series]]]:
        if champion["kind"] == "model":
            return self._candidate_return_grid(str(champion["candidate_id"]), definition)
        ensemble = self.tournaments.get_ensemble(
            str(champion["candidate_id"]), verify=True
        )
        result: dict[str, list[tuple[dict[str, str], pd.Series]]] = {
            profile_id: [] for profile_id in REQUIRED_RESEARCH_PROFILES
        }
        for profile_id in REQUIRED_RESEARCH_PROFILES:
            values: list[pd.Series] = []
            rows = [
                item
                for item in ensemble["evaluations"]
                if str(item["profile_id"]) == profile_id
                and int(item["seed"]) in REQUIRED_MODEL_SEEDS
            ]
            if len(rows) != len(REQUIRED_MODEL_SEEDS):
                raise ValueError("current-identity ensemble return grid is incomplete")
            for row in rows:
                evidence = dict(row.get("evidence") or {})
                path = Path(
                    str(evidence.get("portfolio_report_path") or "")
                ).resolve()
                if not path.is_file() or file_sha256(path) != str(
                    evidence.get("portfolio_report_sha256") or ""
                ):
                    raise ValueError("current-identity ensemble report evidence changed")
                report = pd.read_parquet(path)
                values.append(
                    pd.to_numeric(report["return"], errors="coerce")
                    - pd.to_numeric(report["bench"], errors="coerce")
                    - pd.to_numeric(report["cost"], errors="coerce")
                )
            result[profile_id].append(
                (definition, pd.concat(values, axis=1, join="inner").mean(axis=1))
            )
        return result

    def _advance_tournament_trial(
        self,
        trial_id: str,
        target: str,
        *,
        candidate_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        paths = {
            "queued": ("queued",),
            "running": ("queued", "running"),
            "passed": ("queued", "running", "passed"),
            "failed": ("queued", "running", "failed"),
            "rejected": ("queued", "running", "rejected"),
            "selected": ("queued", "running", "passed", "selected"),
        }
        if target not in paths:
            raise ValueError(f"unsupported tournament target status {target!r}")
        path = paths[target]
        current = self.tournaments.get_trial(trial_id)
        current_status = str(current["status"])
        if current_status == target:
            return
        if current_status == "passed" and target in {"failed", "rejected"}:
            # The trial result is immutable historical evidence.  A later
            # candidate invalidation is recorded on the candidate, not by
            # rewriting the tournament outcome.
            return
        if current_status in {"failed", "rejected", "selected"}:
            raise ValueError("tournament trial terminal result cannot change")
        start = path.index(current_status) + 1 if current_status in path else 0
        for status in path[start:]:
            self.tournaments.transition_trial(
                trial_id,
                status,
                candidate_id=candidate_id,
                metrics=metrics if status in {"passed", "failed", "rejected"} else None,
                evidence=evidence if status in {"passed", "failed", "rejected"} else None,
            )

    def _candidate_tournament_evidence(self, candidate_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            candidate = connection.execute(
                select(model_candidates).where(model_candidates.c.id == candidate_id)
            ).first()
            rows = connection.execute(
                select(model_evaluations).where(
                    model_evaluations.c.model_candidate_id == candidate_id,
                    model_evaluations.c.evidence_role == "independent_gate",
                )
            ).all()
        if candidate is None:
            raise KeyError(candidate_id)
        cells = [
            {
                "profile_id": str(row.profile_id),
                "seed": int(row.seed),
                "gate_status": str(row.gate_status),
                "metrics": dict(row.metrics_json or {}),
                "metrics_sha256": str(row.metrics_sha256),
                "evidence_sha256": str(row.evidence_sha256),
            }
            for row in rows
        ]
        payload = {
            "contract_version": "model-tournament-result-v1",
            "candidate_id": candidate_id,
            "candidate_status": str(candidate.status),
            "candidate_manifest_sha256": str(candidate.manifest_sha256),
            "dataset_identity_sha256": str(candidate.dataset_identity_sha256),
            "admission_evidence_sha256": candidate.admission_evidence_sha256,
            "final_oos_opened": False,
            "cells": cells,
        }
        payload["evidence_sha256"] = canonical_sha256(payload)
        return payload

    def _reconcile_model_tournament(
        self, cycle: dict[str, Any], tournament: dict[str, Any]
    ) -> None:
        branches = self.store.get_cycle(str(cycle["id"]))["branches"]
        for branch in branches:
            details = dict(branch.get("details") or {})
            if details.get("branch_kind") not in {
                "platform_model_feature_screen",
                "platform_model_model_full",
            }:
                continue
            run = self.research.get_run(str(branch["research_run_id"]))
            for binding in details.get("candidate_bindings") or []:
                candidate_id = str(binding["candidate_id"])
                evidence = self._candidate_tournament_evidence(candidate_id)
                candidate_status = str(evidence["candidate_status"])
                trial_ids = (str(binding["trial_id"]),)
                if candidate_status == "research_admitted":
                    metrics = {"cells": evidence["cells"]}
                    for trial_id in trial_ids:
                        self._advance_tournament_trial(
                            trial_id,
                            "passed",
                            candidate_id=candidate_id,
                            metrics=metrics,
                            evidence=evidence,
                        )
                elif candidate_status in {"rejected", "invalidated"}:
                    for trial_id in trial_ids:
                        self._advance_tournament_trial(
                            trial_id,
                            "rejected",
                            candidate_id=candidate_id,
                            metrics={"cells": evidence["cells"]},
                            evidence=evidence,
                        )
                elif str(run["status"]) == "blocked":
                    for trial_id in trial_ids:
                        self._advance_tournament_trial(
                            trial_id,
                            "failed",
                            candidate_id=candidate_id,
                            metrics={"reason_code": "resource_blocked"},
                            evidence=evidence,
                        )
                elif str(run["status"]) in {"running", "evaluating"}:
                    for trial_id in trial_ids:
                        self._advance_tournament_trial(
                            trial_id, "running", candidate_id=candidate_id
                        )
                elif str(run["status"]) in {"failed", "cancelled"} and int(
                    details.get("retry_count") or 0
                ) >= 1:
                    for trial_id in trial_ids:
                        self._advance_tournament_trial(
                            trial_id,
                            "failed",
                            candidate_id=candidate_id,
                            metrics={"reason_code": "execution_failed_after_retry"},
                            evidence=evidence,
                        )

    def _reconcile_dynamic_model_trials(
        self, cycle: dict[str, Any], tournament: dict[str, Any]
    ) -> None:
        del cycle
        refreshed = self.tournaments.get_for_cycle(str(tournament["cycle_id"]))
        for trial in refreshed["trials"]:
            if (
                (trial.get("spec") or {}).get("source") != "rdagent_fin_model"
                or not trial.get("candidate_id")
                or trial["status"] in {"failed", "rejected", "selected"}
            ):
                continue
            candidate_id = str(trial["candidate_id"])
            evidence = self._candidate_tournament_evidence(candidate_id)
            status = str(evidence["candidate_status"])
            if status == "research_admitted":
                self._advance_tournament_trial(
                    str(trial["id"]),
                    "passed",
                    candidate_id=candidate_id,
                    metrics={"cells": evidence["cells"]},
                    evidence=evidence,
                )
            elif status in {"rejected", "invalidated"}:
                self._advance_tournament_trial(
                    str(trial["id"]),
                    "rejected",
                    candidate_id=candidate_id,
                    metrics={"cells": evidence["cells"]},
                    evidence=evidence,
                )
            elif status in {"evaluating", "awaiting_independent_evaluation"}:
                self._advance_tournament_trial(
                    str(trial["id"]), "running", candidate_id=candidate_id
                )

    @staticmethod
    def _model_tournament_score(
        evidence: dict[str, Any], *, seed: int | None = None
    ) -> tuple[float, ...]:
        cells = list(evidence.get("cells") or [])
        if seed is not None:
            # The first feature-set screen is deliberately cheap and uses only
            # recent_3y/seed 11.  It is not the full model championship.
            recent = [
                item
                for item in cells
                if item.get("profile_id") == "recent_3y"
                and item.get("gate_status") == "passed"
                and int(item.get("seed")) == seed
            ]
            if len(recent) != 1:
                return float("-inf"), float("-inf")
            metrics = dict(recent[0].get("metrics") or {})
            return (
                float(metrics["annualized_excess_return_with_cost"]),
                float(metrics["rank_ic"]),
            )

        # The complete round uses all three windows and all three fixed seeds.
        # Reuse the same equal-profile/worst-window rule that freezes the final
        # capital champion; recent_3y is never allowed to dominate this stage.
        grid = aggregate_pre_final_grid(
            [
                {
                    **dict(item),
                    "evidence_role": "independent_gate",
                    "oos_vintage_id": None,
                }
                for item in cells
            ]
        )
        return tuple(float(value) for value in grid["score_vector"])

    def _screen_selected_feature_sets(
        self, cycle: dict[str, Any], tournament: dict[str, Any]
    ) -> list[str]:
        state = dict(cycle.get("state") or {})
        frozen = state.get("screen_selected_feature_set_ids")
        if isinstance(frozen, list) and len(frozen) == 2:
            frozen_evidence = state.get("feature_screen_evidence")
            multiple = (
                frozen_evidence.get("multiple_testing")
                if isinstance(frozen_evidence, dict)
                else None
            )
            if (
                not isinstance(frozen_evidence, dict)
                or not isinstance(multiple, dict)
                or multiple.get("contract_version")
                != "feature-screen-multiple-testing-v1"
                or multiple.get("evidence_sha256")
                != canonical_sha256(
                    {
                        key: value
                        for key, value in multiple.items()
                        if key != "evidence_sha256"
                    }
                )
                or frozen_evidence.get("evidence_sha256")
                != canonical_sha256(
                    {
                        key: value
                        for key, value in frozen_evidence.items()
                        if key != "evidence_sha256"
                    }
                )
            ):
                self.tournaments.block(
                    str(tournament["id"]),
                    reason="legacy feature screen lacks the global Holm/PBO gate",
                )
                return []
            return [str(item) for item in frozen]
        refreshed = self.tournaments.get_for_cycle(str(cycle["id"]))
        screen_trials = [
            item
            for item in refreshed["trials"]
            if (item.get("spec") or {}).get("round") == "feature_screen"
        ]
        if not screen_trials or any(
            item["status"] in {"preregistered", "queued", "running"}
            for item in screen_trials
        ):
            return []
        candidates: list[dict[str, Any]] = []
        trial_series: list[tuple[dict[str, str], pd.Series]] = []
        failed_trial_names: list[str] = []
        for trial in screen_trials:
            trial_id = str(trial["id"])
            candidate_id = str(trial.get("candidate_id") or "")
            trial_name = f"feature-screen:{trial_id}"
            if trial["status"] != "passed" or not candidate_id:
                failed_trial_names.append(trial_name)
                continue
            evidence = dict(trial.get("evidence") or {})
            cells = list(evidence.get("cells") or [])
            if (
                evidence.get("contract_version") != "model-feature-screen-v1"
                or evidence.get("candidate_id") != candidate_id
                or evidence.get("dataset_identity_sha256")
                != tournament["dataset_identity_sha256"]
                or evidence.get("selection_profile") != "recent_3y"
                or evidence.get("selection_seed") != 11
                or evidence.get("final_oos_opened") is not False
                or len(cells) != 1
            ):
                raise ValueError("feature-screen trial evidence is incomplete")
            cell = dict(cells[0])
            report_path = Path(str(cell.get("portfolio_report_path") or "")).resolve()
            if (
                not report_path.is_file()
                or file_sha256(report_path)
                != str(cell.get("portfolio_report_sha256") or "")
            ):
                raise ValueError("feature-screen portfolio evidence changed")
            report = pd.read_parquet(report_path)
            if not {"return", "bench", "cost"}.issubset(report.columns):
                raise ValueError("feature-screen portfolio report is incomplete")
            excess = (
                pd.to_numeric(report["return"], errors="coerce")
                - pd.to_numeric(report["bench"], errors="coerce")
                - pd.to_numeric(report["cost"], errors="coerce")
            )
            trial_series.append(
                (
                    {
                        "name": trial_name,
                        "candidate_id": candidate_id,
                        "trial_id": trial_id,
                        "feature_set_id": str(trial["feature_set_id"]),
                    },
                    excess,
                )
            )
            score = self._model_tournament_score(
                {"cells": [cell]},
                seed=11,
            )
            candidates.append(
                {
                    "feature_set_id": str(trial["feature_set_id"]),
                    "trial_id": trial_id,
                    "trial_name": trial_name,
                    "candidate_id": candidate_id,
                    "score": score,
                }
            )
        if len(trial_series) < 2:
            self.tournaments.block(
                str(tournament["id"]),
                reason="fewer than two feature-screen trials produced valid evidence",
            )
            return []
        try:
            multiple = build_run_multiple_testing_evidence(
                research_run_id=f"tournament:{tournament['id']}:feature_screen",
                trial_series=trial_series,
                output=(
                    self.settings.data_root
                    / "artifacts"
                    / "model-tournaments"
                    / str(tournament["id"])
                    / "feature-screen-multiple-testing"
                ),
                seeds=(11,),
                seed_aggregation="fixed_feature_screen_seed",
            )
        except (OSError, ValueError) as exc:
            self.tournaments.block(
                str(tournament["id"]),
                reason=f"feature-screen shared statistics failed closed: {exc}",
            )
            return []
        completed_names = [str(item[0]["name"]) for item in trial_series]
        raw_by_name = dict(zip(completed_names, multiple["raw_p_values"], strict=True))
        family_names = [f"feature-screen:{item['id']}" for item in screen_trials]
        trial_metadata = {
            f"feature-screen:{item['id']}": item for item in screen_trials
        }
        family_raw = [float(raw_by_name.get(name, 1.0)) for name in family_names]
        family_adjusted = holm_bonferroni(family_raw)
        completed_definitions = {
            str(item["name"]): dict(item)
            for item in multiple["trial_definitions"]
        }
        completed_sharpes = dict(
            zip(
                completed_names,
                multiple["trial_daily_sharpes"],
                strict=True,
            )
        )
        pbo = dict(multiple["pbo"])
        pbo_passed = (
            pbo.get("status") == "ok"
            and pbo.get("pbo") is not None
            and float(pbo["pbo"]) <= 0.50
        )
        eligible_trial_names = [
            name
            for name, adjusted in zip(family_names, family_adjusted, strict=True)
            if name in raw_by_name and float(adjusted) <= 0.05 and pbo_passed
        ]
        multiple = {
            **multiple,
            "contract_version": "feature-screen-multiple-testing-v1",
            "trial_definitions": [
                {
                    **completed_definitions.get(
                        name,
                        {
                            "name": name,
                            "candidate_id": str(
                                trial_metadata[name].get("candidate_id")
                                or "not_registered"
                            ),
                            "trial_id": name.removeprefix("feature-screen:"),
                            "feature_set_id": str(
                                trial_metadata[name]["feature_set_id"]
                            ),
                        },
                    ),
                    "evaluation_status": (
                        "completed" if name in completed_definitions else "failed"
                    ),
                }
                for name in family_names
            ],
            "trial_names": family_names,
            "trial_count": len(family_names),
            "completed_trial_names": completed_names,
            "failed_trial_names": failed_trial_names,
            "raw_p_values": family_raw,
            "holm_adjusted_p_values": family_adjusted,
            "trial_daily_sharpes": [
                completed_sharpes.get(name) for name in family_names
            ],
            "eligible_trial_names": eligible_trial_names,
            "gate_passed": bool(eligible_trial_names),
        }
        multiple["evidence_sha256"] = canonical_sha256(
            {key: value for key, value in multiple.items() if key != "evidence_sha256"}
        )
        eligible = set(eligible_trial_names)
        candidates = [item for item in candidates if item["trial_name"] in eligible]
        best_by_feature: dict[str, dict[str, Any]] = {}
        for item in candidates:
            current = best_by_feature.get(item["feature_set_id"])
            if current is None or (item["score"], item["candidate_id"]) > (
                current["score"],
                current["candidate_id"],
            ):
                best_by_feature[item["feature_set_id"]] = item
        selected = sorted(
            best_by_feature.values(),
            key=lambda item: (item["score"], item["candidate_id"]),
            reverse=True,
        )[:2]
        if len(selected) != 2:
            self.tournaments.block(
                str(tournament["id"]),
                reason=(
                    "fewer than two feature sets passed the fixed-LightGBM "
                    "independent screen"
                ),
            )
            return []
        selected_trial_ids = [str(item["trial_id"]) for item in selected]
        for trial_id in selected_trial_ids:
            self._advance_tournament_trial(trial_id, "selected")
        selected_features = [str(item["feature_set_id"]) for item in selected]
        selected_set = set(selected_features)
        screen_evidence = {
            "contract_version": "feature-screen-selection-v1",
            "dataset_identity_sha256": tournament["dataset_identity_sha256"],
            "selection_profile": "recent_3y",
            "selection_seed": 11,
            "selected_feature_set_ids": selected_features,
            "multiple_testing": multiple,
            "multiple_testing_evidence_sha256": multiple["evidence_sha256"],
            "ranked": [
                {
                    "feature_set_id": item["feature_set_id"],
                    "candidate_id": item["candidate_id"],
                    "score": list(item["score"]),
                }
                for item in sorted(
                    best_by_feature.values(),
                    key=lambda value: (value["score"], value["candidate_id"]),
                    reverse=True,
                )
            ],
            "failed_trials_retained": True,
            "other_profile_and_seed_cells_are_not_extra_votes": True,
            "all_preregistered_screen_trials_counted_in_holm_family": True,
        }
        screen_evidence["evidence_sha256"] = canonical_sha256(screen_evidence)
        # Conditional full trials for losing feature sets remain explicit
        # rejected hypotheses instead of silently disappearing.
        for trial in refreshed["trials"]:
            if (
                (trial.get("spec") or {}).get("round") == "model_full"
                and str(trial.get("feature_set_id")) not in selected_set
                and trial["status"] == "preregistered"
            ):
                self.tournaments.transition_trial(
                    str(trial["id"]),
                    "rejected",
                    metrics={"reason_code": "feature_set_not_in_screen_top_two"},
                    evidence=screen_evidence,
                )
        # Screen artifacts are never capital candidates.  Selected feature
        # sets are evaluated again in the full model round.
        for item in candidates:
            try:
                self.platform_models.candidates.transition_candidate(
                    "model",
                    str(item["candidate_id"]),
                    status="invalidated",
                    reason="screening-only model; full-round candidate required",
                    actor="autopilot",
                )
            except ValueError:
                pass
        self.store.patch_cycle_state(
            str(cycle["id"]),
            state_patch={
                "screen_selected_feature_set_ids": selected_features,
                "feature_screen_evidence": screen_evidence,
            },
            stage="model_full",
        )
        return selected_features

    def _enqueue_next_rdagent_model_challenger(
        self,
        cycle: dict[str, Any],
        dataset: dict[str, Any],
        feature_set_ids: list[str],
        *,
        config: dict[str, Any],
    ) -> int:
        if not feature_set_ids:
            return 0
        branches = self.store.get_cycle(str(cycle["id"]))["branches"]
        rdagent_branches = [
            item
            for item in branches
            if item["scenario"] == "fin_model"
            and not str((item.get("details") or {}).get("branch_kind") or "").startswith(
                "platform_model_"
            )
        ]
        active = next(
            (
                item
                for item in rdagent_branches
                if item["status"] in {"queued", "running", "evaluating"}
            ),
            None,
        )
        if active is not None:
            return 0
        for branch in rdagent_branches:
            if branch["status"] == "failed" and self._retry_failed_branch(branch):
                return 1
        existing_scopes = {str(item["scope_key"]) for item in rdagent_branches}
        for feature_set_id in feature_set_ids:
            scope = f"monthly:{feature_set_id}"
            if scope in existing_scopes:
                continue
            self._enqueue(
                cycle,
                dataset,
                "fin_model",
                scope,
                config=config,
                feature_set_id=feature_set_id,
                tournament_id=str(
                    self.tournaments.get_for_cycle(str(cycle["id"]))["id"]
                ),
            )
            return 1
        return 0

    def _model_champions(
        self,
        cycle: dict[str, Any],
        tournament: dict[str, Any],
        selected_feature_set_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Freeze one pre-final champion per model family.

        Platform models and RD-Agent challengers share this comparison.  The
        function waits for both platform full lanes and the two bounded
        RD-Agent lanes; failures remain counted trials and do not create an
        opportunity to substitute an unregistered experiment.
        """

        state = dict(cycle.get("state") or {})
        frozen = state.get("model_champions")
        if isinstance(frozen, list) and frozen:
            frozen_values = [dict(item) for item in frozen]
            frozen_evidence = state.get("model_champion_evidence")
            if not isinstance(frozen_evidence, dict) or frozen_evidence.get(
                "contract_version"
            ) != "model-family-champions-v2":
                self.tournaments.block(
                    str(tournament["id"]),
                    reason="legacy model selection lacks the global Holm/PBO gate",
                )
                return []
            frozen_multiple = frozen_evidence.get("multiple_testing")
            if (
                not isinstance(frozen_multiple, dict)
                or frozen_multiple.get("contract_version")
                != "model-full-multiple-testing-v2"
            ):
                self.tournaments.block(
                    str(tournament["id"]),
                    reason="legacy model selection used recent-only statistics",
                )
                return []
            if (
                frozen_evidence.get("dataset_identity_sha256")
                != tournament["dataset_identity_sha256"]
                or frozen_evidence.get("champions") != frozen_values
                or frozen_evidence.get("evidence_sha256")
                != canonical_sha256(
                    {
                        key: value
                        for key, value in frozen_evidence.items()
                        if key != "evidence_sha256"
                    }
                )
            ):
                raise ValueError("frozen model champion evidence changed")
            with self.engine.connect() as connection:
                candidate_rows = {
                    str(item.id): item
                    for item in connection.execute(
                        select(model_candidates).where(
                            model_candidates.c.id.in_(
                                [
                                    str(item.get("candidate_id") or "")
                                    for item in frozen_values
                                ]
                            )
                        )
                    ).all()
                }
            if set(candidate_rows) != {
                str(item.get("candidate_id") or "") for item in frozen_values
            }:
                raise ValueError("frozen model champion no longer exists")
            for item in frozen_values:
                row = candidate_rows[str(item["candidate_id"])]
                if (
                    str(item.get("candidate_manifest_sha256") or "")
                    != str(row.manifest_sha256)
                    or str(item.get("admission_evidence_sha256") or "")
                    != str(row.admission_evidence_sha256)
                ):
                    raise ValueError("frozen model champion component changed")
            return frozen_values
        selected_set = set(selected_feature_set_ids)
        branches = self.store.get_cycle(str(cycle["id"]))["branches"]
        relevant = [
            item
            for item in branches
            if item["scenario"] == "fin_model"
            and (
                (item.get("details") or {}).get("branch_kind")
                == "platform_model_model_full"
                or str(item.get("scope_key") or "").startswith("monthly:")
            )
            and str((item.get("details") or {}).get("feature_set_id") or "")
            in selected_set
        ]
        expected_scopes = {
            *(f"platform:model_full:{item}" for item in selected_set),
            *(f"monthly:{item}" for item in selected_set),
        }
        if {str(item["scope_key"]) for item in relevant} != expected_scopes:
            return []
        if any(
            item["status"] in {"queued", "running", "evaluating"}
            or (
                item["status"] == "failed"
                and int((item.get("details") or {}).get("retry_count") or 0) < 1
            )
            for item in relevant
        ):
            return []

        refreshed = self.tournaments.get_for_cycle(str(cycle["id"]))
        candidates: list[dict[str, Any]] = []
        trial_series_by_profile: dict[
            str, list[tuple[dict[str, str], pd.Series]]
        ] = {profile_id: [] for profile_id in REQUIRED_RESEARCH_PROFILES}
        full_trials = [
            trial
            for trial in refreshed["trials"]
            if trial["trial_kind"] == "model"
            and str(trial.get("feature_set_id") or "") in selected_set
            and (
                (trial.get("spec") or {}).get("round") == "model_full"
                or (trial.get("spec") or {}).get("source") == "rdagent_fin_model"
            )
        ]
        for trial in refreshed["trials"]:
            if (
                trial["trial_kind"] != "model"
                or trial["status"] != "passed"
                or str(trial.get("feature_set_id") or "") not in selected_set
                or not trial.get("candidate_id")
            ):
                continue
            evidence = self._candidate_tournament_evidence(str(trial["candidate_id"]))
            if evidence["candidate_status"] != "research_admitted":
                continue
            candidate_record = self.platform_models.candidates.get_model_candidate(
                str(trial["candidate_id"]), verify=True
            )
            admission = dict(candidate_record.get("admission_evidence_json") or {})
            if (
                canonical_sha256(
                    {key: value for key, value in admission.items() if key != "evidence_sha256"}
                )
                != str(candidate_record.get("admission_evidence_sha256") or "")
            ):
                raise ValueError("model candidate admission evidence changed")
            trial_name = f"model-full:{trial['id']}"
            definition = {
                "name": trial_name,
                "candidate_id": str(trial["candidate_id"]),
                "trial_id": str(trial["id"]),
                "feature_set_id": str(trial["feature_set_id"]),
                "model_family": str(trial.get("model_family") or "unknown"),
            }
            for profile_id in REQUIRED_RESEARCH_PROFILES:
                profile = dict(
                    (admission.get("profiles") or {}).get(profile_id) or {}
                )
                seed_reports: list[pd.Series] = []
                for seed in REQUIRED_MODEL_SEEDS:
                    cell = dict((profile.get("seeds") or {}).get(str(seed)) or {})
                    report_path = Path(
                        str(cell.get("portfolio_report_path") or "")
                    ).resolve()
                    if (
                        not report_path.is_file()
                        or file_sha256(report_path)
                        != str(cell.get("portfolio_report_sha256") or "")
                    ):
                        raise ValueError(
                            "model champion portfolio evidence changed"
                        )
                    report = pd.read_parquet(report_path)
                    if not {"return", "bench", "cost"}.issubset(report.columns):
                        raise ValueError(
                            "model champion portfolio report is incomplete"
                        )
                    seed_reports.append(
                        (
                            pd.to_numeric(report["return"], errors="coerce")
                            - pd.to_numeric(report["bench"], errors="coerce")
                            - pd.to_numeric(report["cost"], errors="coerce")
                        ).rename(str(seed))
                    )
                trial_series_by_profile[profile_id].append(
                    (
                        definition,
                        pd.concat(seed_reports, axis=1, join="inner").mean(
                            axis=1
                        ),
                    )
                )
            candidates.append(
                {
                    "trial_id": str(trial["id"]),
                    "candidate_id": str(trial["candidate_id"]),
                    "model_family": str(trial.get("model_family") or "unknown"),
                    "feature_set_id": str(trial["feature_set_id"]),
                    "score": self._model_tournament_score(evidence),
                    "evidence_sha256": str(evidence["evidence_sha256"]),
                    "candidate_manifest_sha256": str(
                        evidence["candidate_manifest_sha256"]
                    ),
                    "admission_evidence_sha256": str(
                        evidence["admission_evidence_sha256"]
                    ),
                    "trial_name": trial_name,
                }
            )
        if not any(trial_series_by_profile.values()):
            self.tournaments.block(
                str(tournament["id"]),
                reason="no full-round model produced immutable return evidence",
            )
            return []
        family_definitions = [
            {
                "name": f"model-full:{item['id']}",
                "candidate_id": str(item.get("candidate_id") or "not_registered"),
                "trial_id": str(item["id"]),
                "feature_set_id": str(item.get("feature_set_id") or "unknown"),
                "model_family": str(item.get("model_family") or "unknown"),
            }
            for item in full_trials
        ]
        try:
            multiple = _profile_family_multiple_testing(
                research_run_id=f"tournament:{tournament['id']}:model_full",
                trial_series_by_profile=trial_series_by_profile,
                family_definitions=family_definitions,
                output=(
                    self.settings.data_root
                    / "artifacts"
                    / "model-tournaments"
                    / str(tournament["id"])
                    / "model-full-multiple-testing"
                ),
                contract_version="model-full-multiple-testing-v2",
            )
        except (OSError, ValueError) as exc:
            self.tournaments.block(
                str(tournament["id"]),
                reason=f"full-model shared statistics failed closed: {exc}",
            )
            return []
        eligible = set(multiple["eligible_trial_names"])
        candidates = [item for item in candidates if item["trial_name"] in eligible]
        best_by_family: dict[str, dict[str, Any]] = {}
        for item in candidates:
            current = best_by_family.get(item["model_family"])
            if current is None or (item["score"], item["candidate_id"]) > (
                current["score"],
                current["candidate_id"],
            ):
                best_by_family[item["model_family"]] = item
        champions = sorted(
            best_by_family.values(),
            key=lambda item: (item["score"], item["candidate_id"]),
            reverse=True,
        )[:4]
        if not champions:
            self.tournaments.block(
                str(tournament["id"]),
                reason="no full-round model passed the independent three-window gate",
            )
            return []
        for item in champions:
            self._advance_tournament_trial(str(item["trial_id"]), "selected")
        frozen_champions = [
            {**item, "score": list(item["score"])} for item in champions
        ]
        evidence = {
            "contract_version": "model-family-champions-v2",
            "dataset_identity_sha256": tournament["dataset_identity_sha256"],
            "champions": frozen_champions,
            "multiple_testing": multiple,
            "multiple_testing_evidence_sha256": multiple["evidence_sha256"],
            "family_selection": "one_champion_per_distinct_model_family",
            "seed_cells_are_robustness_repeats": True,
            "window_cells_are_not_independent_votes": True,
            "failed_and_rejected_trials_retained": True,
        }
        evidence["evidence_sha256"] = canonical_sha256(evidence)
        self.store.patch_cycle_state(
            str(cycle["id"]),
            state_patch={
                "model_champions": frozen_champions,
                "model_champion_evidence": evidence,
            },
            stage="ensemble",
        )
        return frozen_champions

    def _reconcile_model_ensembles(
        self,
        cycle: dict[str, Any],
        dataset: dict[str, Any],
        tournament: dict[str, Any],
        champions: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """Run the bounded equal-rank round and close the model tournament.

        Correlations are computed only from the sealed per-date, per-security
        prediction artifacts owned by the admitted model candidates.  Missing
        artifacts leave the cycle waiting; corrupt or inconsistent immutable
        evidence blocks the tournament instead of falling back to IC or model
        metrics as a proxy for prediction correlation.
        """

        if not champions or str(tournament.get("status") or "") in {
            "failed",
            "blocked",
        }:
            return 0, 0
        state = dict(cycle.get("state") or {})
        frozen = state.get("prediction_champion")
        if isinstance(frozen, dict) and frozen.get("candidate_id"):
            return 0, 0

        identity = str(dataset["provenance"]["dataset_identity_sha256"])
        candidate_result: dict[str, Any] = {
            "status": "not_applicable_single_family",
            "candidate_count": 0,
            "candidates": [],
            "pairwise_correlation_evidence": [],
        }
        preregistered_ids = state.get("model_ensemble_candidate_ids")
        if isinstance(preregistered_ids, list):
            candidate_result["status"] = str(
                state.get("model_ensemble_preregistration_status")
                or "ready"
            )
        if len(champions) >= 2 and not isinstance(preregistered_ids, list):
            try:
                candidate_result = self.model_ensembles.ensure_candidates(
                    tournament_id=str(tournament["id"]),
                    champion_trial_ids=[str(item["trial_id"]) for item in champions],
                    dataset=str(dataset["name"]),
                    dataset_identity_sha256=identity,
                )
                pairwise = list(
                    candidate_result.get("pairwise_correlation_evidence") or []
                )
                self.store.patch_cycle_state(
                    str(cycle["id"]),
                    state_patch={
                        "model_ensemble_preregistration_status": candidate_result[
                            "status"
                        ],
                        "model_ensemble_candidate_ids": sorted(
                            str(item["id"])
                            for item in candidate_result.get("candidates") or []
                        ),
                        "model_ensemble_pairwise_evidence_sha256": canonical_sha256(
                            pairwise
                        ),
                        "model_ensemble_pairwise_summary": [
                            {
                                "left_candidate_id": item.get("left_candidate_id"),
                                "right_candidate_id": item.get(
                                    "right_candidate_id"
                                ),
                                "maximum_mean_absolute_daily_rank_correlation": item.get(
                                    "maximum_mean_absolute_daily_rank_correlation"
                                ),
                                "passed": item.get("passed"),
                            }
                            for item in pairwise
                        ],
                    },
                    stage="ensemble",
                )
            except EnsemblePredictionsPending as exc:
                self.store.patch_cycle_state(
                    str(cycle["id"]),
                    state_patch={
                        "model_ensemble_status": "waiting_for_sealed_predictions",
                        "model_ensemble_waiting_reason": str(exc),
                    },
                    stage="ensemble",
                )
                return 0, 0
            except (KeyError, ValueError) as exc:
                reason = f"model ensemble preregistration failed closed: {exc}"
                self.tournaments.block(str(tournament["id"]), reason=reason)
                self.store.patch_cycle_state(
                    str(cycle["id"]),
                    state_patch={
                        "model_ensemble_status": "blocked",
                        "model_ensemble_blocker": reason,
                    },
                    stage="ensemble_blocked",
                )
                return 0, 1

        ensemble_trial_ids = {
            str(item["candidate_id"]): str(item["id"])
            for item in self.tournaments.get_for_cycle(str(cycle["id"]))["trials"]
            if item.get("trial_kind") == "model_ensemble" and item.get("candidate_id")
        }
        try:
            ensembles = [
                self.tournaments.get_ensemble(candidate_id, verify=True)
                for candidate_id in sorted(ensemble_trial_ids)
            ]
        except (KeyError, ValueError) as exc:
            reason = f"model ensemble immutable evidence verification failed: {exc}"
            self.tournaments.block(str(tournament["id"]), reason=reason)
            self.store.patch_cycle_state(
                str(cycle["id"]),
                state_patch={
                    "model_ensemble_status": "blocked",
                    "model_ensemble_blocker": reason,
                },
                stage="ensemble_blocked",
            )
            return 0, 1
        active = [
            item
            for item in ensembles
            if item["status"] in {"awaiting_evaluation", "evaluating"}
        ]
        if active:
            try:
                calendar = (
                    (Path(str(dataset["path"])) / "calendars" / "day.txt")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                _periods, resolution = resolve_research_periods(calendar)
                job = self.model_ensembles.queue_evaluation(
                    tournament_id=str(tournament["id"]),
                    dataset=dataset,
                    evaluation_profiles=resolution["evaluation_profiles"],
                )
            except (KeyError, ValueError) as exc:
                reason = f"model ensemble evaluation could not be frozen: {exc}"
                self.tournaments.block(str(tournament["id"]), reason=reason)
                self.store.patch_cycle_state(
                    str(cycle["id"]),
                    state_patch={
                        "model_ensemble_status": "blocked",
                        "model_ensemble_blocker": reason,
                    },
                    stage="ensemble_blocked",
                )
                return 0, 1
            self.store.patch_cycle_state(
                str(cycle["id"]),
                state_patch={
                    "model_ensemble_status": "evaluating",
                    "model_ensemble_candidate_ids": sorted(ensemble_trial_ids),
                    "model_ensemble_job_id": str(job["id"]) if job else None,
                    "model_ensemble_waiting_reason": None,
                },
                stage="ensemble",
            )
            return int(job is not None and str(job.get("status")) == "queued"), 0

        admitted = [item for item in ensembles if item["status"] == "research_admitted"]
        pool: list[dict[str, Any]] = [
            {
                "kind": "model",
                "candidate_id": str(item["candidate_id"]),
                "trial_id": str(item["trial_id"]),
                "score": tuple(float(value) for value in item["score"]),
                "primary_model_candidate_id": str(item["candidate_id"]),
                "primary_feature_set_id": str(item["feature_set_id"]),
                "component_model_candidate_ids": [str(item["candidate_id"])],
                "component_feature_set_ids": [str(item["feature_set_id"])],
                "manifest_sha256": str(
                    item.get("candidate_manifest_sha256")
                    or item["evidence_sha256"]
                ),
                "admission_evidence_sha256": str(
                    item.get("admission_evidence_sha256")
                    or item["evidence_sha256"]
                ),
                "final_trial_name": str(item["trial_name"]),
            }
            for item in champions
        ]
        champions_by_candidate = {
            str(item["candidate_id"]): item for item in champions
        }
        for ensemble in admitted:
            score = self._model_tournament_score(
                {
                    "cells": [
                        {
                            "profile_id": row["profile_id"],
                            "seed": row["seed"],
                            "gate_status": row["gate_status"],
                            "metrics": row["metrics"],
                        }
                        for row in ensemble["evaluations"]
                    ]
                }
            )
            component_ids = [
                str(item["model_candidate_id"])
                for item in ensemble["components"]
            ]
            primary = max(
                (champions_by_candidate[item] for item in component_ids),
                key=lambda item: (tuple(item["score"]), str(item["candidate_id"])),
            )
            pool.append(
                {
                    "kind": "ensemble",
                    "candidate_id": str(ensemble["id"]),
                    "trial_id": ensemble_trial_ids[str(ensemble["id"])],
                    "score": score,
                    "primary_model_candidate_id": str(primary["candidate_id"]),
                    "primary_feature_set_id": str(primary["feature_set_id"]),
                    "component_model_candidate_ids": component_ids,
                    "component_feature_set_ids": sorted(
                        {
                            str(champions_by_candidate[item]["feature_set_id"])
                            for item in component_ids
                        }
                    ),
                    "manifest_sha256": str(ensemble["manifest_sha256"]),
                    "admission_evidence_sha256": str(
                        ensemble["admission_evidence_sha256"]
                    ),
                    "final_trial_name": (
                        f"model-ensemble:{ensemble_trial_ids[str(ensemble['id'])]}"
                    ),
                }
            )
        if not pool:
            return 0, 0
        model_selection = state.get("model_champion_evidence")
        model_multiple = (
            model_selection.get("multiple_testing")
            if isinstance(model_selection, dict)
            else None
        )
        if (
            not isinstance(model_multiple, dict)
            or model_multiple.get("contract_version")
            != "model-full-multiple-testing-v2"
            or model_multiple.get("evidence_sha256")
            != canonical_sha256(
                {
                    key: value
                    for key, value in model_multiple.items()
                    if key != "evidence_sha256"
                }
            )
        ):
            reason = "prediction selection has no global full-model evidence"
            self.tournaments.block(str(tournament["id"]), reason=reason)
            self.store.patch_cycle_state(
                str(cycle["id"]),
                state_patch={
                    "model_ensemble_status": "blocked",
                    "model_ensemble_blocker": reason,
                },
                stage="ensemble_blocked",
            )
            return 0, 1
        model_definitions = {
            str(item.get("name") or ""): {
                "name": str(item.get("name") or ""),
                "candidate_id": str(item.get("candidate_id") or "not_evaluable"),
                "trial_id": str(item.get("trial_id") or "unknown"),
                "kind": "model",
            }
            for item in model_multiple.get("trial_definitions") or []
        }
        final_series_by_profile: dict[
            str, list[tuple[dict[str, str], pd.Series]]
        ] = {profile_id: [] for profile_id in REQUIRED_RESEARCH_PROFILES}
        for profile_id in REQUIRED_RESEARCH_PROFILES:
            profile_evidence = dict(
                (model_multiple.get("per_profile") or {}).get(profile_id) or {}
            )
            if (
                profile_evidence.get("evidence_sha256")
                != canonical_sha256(
                    {
                        key: value
                        for key, value in profile_evidence.items()
                        if key != "evidence_sha256"
                    }
                )
            ):
                raise ValueError("global full-model profile evidence changed")
            returns_path = Path(
                str(profile_evidence.get("returns_path") or "")
            ).resolve()
            if (
                not returns_path.is_file()
                or file_sha256(returns_path)
                != str(profile_evidence.get("returns_sha256") or "")
            ):
                raise ValueError("global full-model return matrix changed")
            returns = pd.read_parquet(returns_path)
            completed_names = [
                str(item) for item in profile_evidence.get("trial_names") or []
            ]
            if set(returns.columns) != set(completed_names):
                raise ValueError("global full-model return matrix columns changed")
            final_series_by_profile[profile_id].extend(
                (model_definitions[name], returns[name])
                for name in completed_names
            )
        for ensemble in admitted:
            ensemble_id = str(ensemble["id"])
            trial_id = ensemble_trial_ids[ensemble_id]
            definition = {
                "name": f"model-ensemble:{trial_id}",
                "candidate_id": ensemble_id,
                "trial_id": trial_id,
                "kind": "model_ensemble",
            }
            for profile_id in REQUIRED_RESEARCH_PROFILES:
                rows = [
                    item
                    for item in ensemble["evaluations"]
                    if str(item["profile_id"]) == profile_id
                    and int(item["seed"]) in REQUIRED_MODEL_SEEDS
                    and str(item["gate_status"]) == "passed"
                ]
                if {int(item["seed"]) for item in rows} != set(
                    REQUIRED_MODEL_SEEDS
                ) or len(rows) != len(REQUIRED_MODEL_SEEDS):
                    raise ValueError(
                        "ensemble final-selection return grid is incomplete"
                    )
                returns: list[pd.Series] = []
                for row in rows:
                    cell = dict(row.get("evidence") or {})
                    report_path = Path(
                        str(cell.get("portfolio_report_path") or "")
                    ).resolve()
                    if (
                        not report_path.is_file()
                        or file_sha256(report_path)
                        != str(cell.get("portfolio_report_sha256") or "")
                    ):
                        raise ValueError(
                            "ensemble final-selection report changed"
                        )
                    report = pd.read_parquet(report_path)
                    if not {"return", "bench", "cost"}.issubset(report.columns):
                        raise ValueError(
                            "ensemble final-selection report is incomplete"
                        )
                    returns.append(
                        (
                            pd.to_numeric(report["return"], errors="coerce")
                            - pd.to_numeric(report["bench"], errors="coerce")
                            - pd.to_numeric(report["cost"], errors="coerce")
                        ).rename(str(row["seed"]))
                    )
                final_series_by_profile[profile_id].append(
                    (
                        definition,
                        pd.concat(returns, axis=1, join="inner").mean(axis=1),
                    )
                )
        ensemble_definitions = [
            {
                "name": (
                    "model-ensemble:"
                    f"{ensemble_trial_ids[str(item['id'])]}"
                ),
                "candidate_id": str(item["id"]),
                "trial_id": ensemble_trial_ids[str(item["id"])],
                "kind": "model_ensemble",
            }
            for item in ensembles
        ]
        family_definitions = [
            *model_definitions.values(),
            *ensemble_definitions,
        ]
        try:
            final_multiple = _profile_family_multiple_testing(
                research_run_id=(
                    f"tournament:{tournament['id']}:prediction_finalists"
                ),
                trial_series_by_profile=final_series_by_profile,
                family_definitions=family_definitions,
                output=(
                    self.settings.data_root
                    / "artifacts"
                    / "model-tournaments"
                    / str(tournament["id"])
                    / "prediction-finalist-multiple-testing"
                ),
                contract_version="prediction-finalist-multiple-testing-v2",
            )
        except (OSError, ValueError) as exc:
            reason = f"prediction-finalist shared statistics failed closed: {exc}"
            self.tournaments.block(str(tournament["id"]), reason=reason)
            self.store.patch_cycle_state(
                str(cycle["id"]),
                state_patch={
                    "model_ensemble_status": "blocked",
                    "model_ensemble_blocker": reason,
                },
                stage="ensemble_blocked",
            )
            return 0, 1
        eligible = set(final_multiple["eligible_trial_names"])
        pool = [item for item in pool if item["final_trial_name"] in eligible]
        if not pool:
            reason = "no model or ensemble passed the shared finalist Holm/PBO gate"
            self.tournaments.block(str(tournament["id"]), reason=reason)
            self.store.patch_cycle_state(
                str(cycle["id"]),
                state_patch={
                    "model_ensemble_status": "blocked",
                    "model_ensemble_blocker": reason,
                    "prediction_finalist_multiple_testing": final_multiple,
                },
                stage="ensemble_blocked",
            )
            return 0, 1
        winner = max(
            pool,
            key=lambda item: (
                tuple(item["score"]),
                item["kind"] == "model",  # prefer the simpler signal on an exact tie
                str(item["candidate_id"]),
            ),
        )
        if winner["kind"] == "ensemble":
            self._advance_tournament_trial(str(winner["trial_id"]), "selected")
        refreshed = self.tournaments.get_for_cycle(str(cycle["id"]))
        selected_trial_ids = sorted(
            str(item["id"])
            for item in refreshed["trials"]
            if item["status"] == "selected"
        )
        selection_evidence = {
            "contract_version": "prediction-champion-selection-v1",
            "dataset_identity_sha256": identity,
            "selection_data": "pre_final_only",
            "final_oos_opened": False,
            "score_order": [
                "three_window_equal_mean_annualized_excess_return_with_cost",
                "three_window_worst_annualized_excess_return_with_cost",
                "three_window_equal_mean_information_ratio",
                "three_window_equal_mean_rank_ic",
                "negative_three_window_equal_mean_turnover",
                "negative_absolute_three_window_equal_mean_max_drawdown",
            ],
            "exact_tie_policy": "prefer_single_model_then_candidate_id",
            "eligible": [
                {**item, "score": list(item["score"])}
                for item in sorted(
                    pool, key=lambda value: (value["kind"], value["candidate_id"])
                )
            ],
            "selected_kind": winner["kind"],
            "selected_candidate_id": winner["candidate_id"],
            "global_multiple_testing": final_multiple,
            "global_multiple_testing_evidence_sha256": final_multiple[
                "evidence_sha256"
            ],
            "models_and_ensembles_share_one_finalist_family": True,
            "selection_is_not_an_uncorrected_additional_hypothesis_test": True,
            "failed_and_rejected_trials_retained": True,
            "selected_trial_ids": selected_trial_ids,
        }
        selection_evidence["evidence_sha256"] = canonical_sha256(selection_evidence)
        self.tournaments.complete_selection(
            str(tournament["id"]),
            selected_trial_ids=selected_trial_ids,
            multiple_testing=selection_evidence,
        )
        frozen_winner = {**winner, "score": list(winner["score"])}
        self.store.patch_cycle_state(
            str(cycle["id"]),
            state_patch={
                "model_ensemble_status": (
                    "complete" if ensembles else candidate_result["status"]
                ),
                "model_ensemble_candidate_ids": sorted(ensemble_trial_ids),
                "model_ensemble_waiting_reason": None,
                "prediction_champion": frozen_winner,
                "prediction_champion_evidence": selection_evidence,
                "prediction_champion_status": "validated_current_identity",
                "prediction_champion_roll_forward": None,
                "fin_quant_status": "ready",
                "fin_quant_blocker": None,
                "research_tournament_id": str(tournament["id"]),
                "research_tournament_status": "succeeded",
            },
            stage="joint_optimization",
        )
        return 0, 0

    def _roll_forward_prediction_champion(
        self,
        cycle: dict[str, Any],
        dataset: dict[str, Any],
    ) -> dict[str, Any]:
        """Retain predecessor lineage without promoting it on a new vintage.

        Model research is monthly while market data is published daily.  The
        previous recipe is useful as an incumbent reference, but its old
        selection/evaluation is never copied into the current champion fields.
        Until exact current-identity revalidation exists, fin_quant remains
        fail-closed with an explicit pending state.
        """

        state = dict(cycle.get("state") or {})
        current_identity = str(
            (dataset.get("provenance") or {}).get("dataset_identity_sha256") or ""
        )
        current_error = _prediction_champion_identity_error(cycle, dataset)
        if current_error is None:
            return cycle
        current_champion = state.get("prediction_champion")
        current_selection = state.get("prediction_champion_evidence")
        current_model_selection = state.get("model_champion_evidence")
        if isinstance(current_champion, dict) and current_champion.get("candidate_id"):
            source_identity = str(
                (current_selection or {}).get("dataset_identity_sha256")
                if isinstance(current_selection, dict)
                else ""
            )
            rollover = {
                "contract_version": "prediction-champion-roll-forward-v2",
                "source_cycle_id": None,
                "source_dataset_identity_sha256": source_identity,
                "target_dataset_identity_sha256": current_identity,
                "dataset_lineage_id": str(cycle.get("dataset_lineage_id") or ""),
                "source_selection_evidence_sha256": (
                    current_selection.get("evidence_sha256")
                    if isinstance(current_selection, dict)
                    else None
                ),
                "source_model_selection_evidence_sha256": (
                    current_model_selection.get("evidence_sha256")
                    if isinstance(current_model_selection, dict)
                    else None
                ),
                "reference_only": True,
                "requires_current_identity_revalidation": True,
                "eligible_for_fin_quant": False,
                "old_predictions_reused": False,
                "old_scores_reused": False,
                "final_oos_opened": False,
            }
            rollover["evidence_sha256"] = canonical_sha256(rollover)
            return self.store.patch_cycle_state(
                str(cycle["id"]),
                state_patch={
                    "prior_prediction_champion": dict(current_champion),
                    "prior_prediction_champion_evidence": (
                        dict(current_selection)
                        if isinstance(current_selection, dict)
                        else None
                    ),
                    "prior_model_champion_evidence": (
                        dict(current_model_selection)
                        if isinstance(current_model_selection, dict)
                        else None
                    ),
                    "prediction_champion": None,
                    "prediction_champion_evidence": None,
                    "model_champion_evidence": None,
                    "prediction_champion_status": (
                        "pending_current_identity_revalidation"
                    ),
                    "fin_quant_status": "pending_current_identity_revalidation",
                    "fin_quant_blocker": current_error,
                    "prediction_champion_roll_forward": rollover,
                },
            )
        existing_rollover = state.get("prediction_champion_roll_forward")
        if (
            state.get("prediction_champion_status")
            == "pending_current_identity_revalidation"
            and isinstance(existing_rollover, dict)
            and existing_rollover.get("target_dataset_identity_sha256")
            == current_identity
        ):
            return cycle
        lineage_id = str(
            dataset.get("lineage_id")
            or (dataset.get("provenance") or {}).get("dataset_lineage_id")
            or ""
        )
        current_end = str(dataset.get("end_date") or "")
        for predecessor in self.store.list_cycles(limit=500):
            if str(predecessor.get("id")) == str(cycle.get("id")):
                continue
            if predecessor.get("horizon_profile") != cycle.get("horizon_profile"):
                continue
            if str(predecessor.get("dataset_lineage_id") or "") != lineage_id:
                continue
            predecessor_state = dict(predecessor.get("state") or {})
            champion = predecessor_state.get("prediction_champion")
            selection = predecessor_state.get("prediction_champion_evidence")
            model_selection = predecessor_state.get("model_champion_evidence")
            if (
                not isinstance(champion, dict)
                or not isinstance(selection, dict)
                or not isinstance(model_selection, dict)
                or model_selection.get("contract_version")
                != "model-family-champions-v2"
                or not isinstance(selection.get("global_multiple_testing"), dict)
                or selection["global_multiple_testing"].get("contract_version")
                != "prediction-finalist-multiple-testing-v2"
                or not isinstance(model_selection.get("multiple_testing"), dict)
                or model_selection["multiple_testing"].get("contract_version")
                != "model-full-multiple-testing-v2"
            ):
                continue
            if canonical_sha256(
                {key: value for key, value in selection.items() if key != "evidence_sha256"}
            ) != str(selection.get("evidence_sha256") or ""):
                continue
            if canonical_sha256(
                {
                    key: value
                    for key, value in model_selection.items()
                    if key != "evidence_sha256"
                }
            ) != str(model_selection.get("evidence_sha256") or ""):
                continue
            predecessor_end = str(predecessor_state.get("dataset_end_date") or "")
            if predecessor_end and current_end and predecessor_end > current_end:
                continue
            source_identity = str(predecessor.get("dataset_identity_sha256") or "")
            with self.engine.connect() as connection:
                if champion.get("kind") == "model":
                    valid = connection.scalar(
                        select(model_candidates.c.id).where(
                            model_candidates.c.id
                            == str(champion.get("candidate_id") or ""),
                            model_candidates.c.status == "research_admitted",
                            model_candidates.c.dataset_identity_sha256
                            == source_identity,
                            model_candidates.c.dataset_lineage_id == lineage_id,
                        )
                    )
                elif champion.get("kind") == "ensemble":
                    valid = connection.scalar(
                        select(model_ensemble_candidates.c.id).where(
                            model_ensemble_candidates.c.id
                            == str(champion.get("candidate_id") or ""),
                            model_ensemble_candidates.c.status == "research_admitted",
                            model_ensemble_candidates.c.dataset_identity_sha256
                            == source_identity,
                        )
                    )
                else:
                    valid = None
            if valid is None:
                continue
            rollover = {
                "contract_version": "prediction-champion-roll-forward-v2",
                "source_cycle_id": str(predecessor["id"]),
                "source_dataset_identity_sha256": source_identity,
                "target_dataset_identity_sha256": current_identity,
                "dataset_lineage_id": lineage_id,
                "source_selection_evidence_sha256": selection["evidence_sha256"],
                "source_model_selection_evidence_sha256": model_selection[
                    "evidence_sha256"
                ],
                "reference_only": True,
                "requires_current_identity_revalidation": True,
                "eligible_for_fin_quant": False,
                "old_predictions_reused": False,
                "old_scores_reused": False,
                "final_oos_opened": False,
            }
            rollover["evidence_sha256"] = canonical_sha256(rollover)
            return self.store.patch_cycle_state(
                str(cycle["id"]),
                state_patch={
                    "prior_prediction_champion": dict(champion),
                    "prior_prediction_champion_evidence": dict(selection),
                    "prior_model_champion_evidence": dict(model_selection),
                    "prediction_champion": None,
                    "prediction_champion_evidence": None,
                    "model_champion_evidence": None,
                    "prediction_champion_status": (
                        "pending_current_identity_revalidation"
                    ),
                    "fin_quant_status": "pending_current_identity_revalidation",
                    "fin_quant_blocker": (
                        "prediction champion awaits current dataset identity "
                        "revalidation"
                    ),
                    "prediction_champion_roll_forward": rollover,
                },
            )
        return cycle

    def _active_sota_feature_set_id(
        self,
        dataset: dict[str, Any],
        *,
        horizon_profile: str = SHORT_1_5D,
    ) -> str | None:
        resolved = self.factor_autopilot.resolve_active_sota(
            dataset,
            horizon_profile=horizon_profile,
        )
        if resolved is None:
            return None
        definition = dict(resolved["feature_set"])
        return str(definition["id"])

    def _ensure_horizon_factor_bundle(
        self,
        cycle: dict[str, Any],
        dataset: dict[str, Any],
        *,
        feature_set_id: str,
    ) -> dict[str, Any]:
        """Freeze the base/SOTA factor champion on its existing cycle.

        Alpha158 (or the configured immutable base) is a real incumbent even
        before RD-Agent admits an incremental factor.  Recording an empty-
        increment bundle gives every horizon a content-addressed champion
        identity without opening another research or promotion path.
        """

        state = dict(cycle.get("state") or {})
        existing = state.get("horizon_factor_bundle")
        if existing is not None:
            validated = validate_horizon_factor_bundle(existing)
            if (
                state.get("horizon_factor_bundle_sha256")
                != validated["bundle_sha256"]
                or validated["horizon_profile"]
                != cycle.get("horizon_profile")
                or validated["dataset_identity_sha256"]
                != cycle.get("dataset_identity_sha256")
            ):
                raise ValueError("autopilot horizon factor champion changed in place")
            return cycle

        feature_set = get_feature_set(feature_set_id)
        calendar = (
            (Path(str(dataset["path"])) / "calendars" / "day.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        horizon_profile = str(cycle.get("horizon_profile") or "")
        periods, resolution = resolve_research_window_contract(
            dataset,
            calendar,
            horizon_profile=horizon_profile,
            feature_set=feature_set,
        )
        label_binding = resolve_research_label_binding(
            {
                "horizon_profile": horizon_profile,
                "dataset": str(dataset["name"]),
                "dataset_identity_sha256": str(
                    dataset["provenance"]["dataset_identity_sha256"]
                ),
                "periods": periods,
                "feature_set": feature_set,
                "research_window_contract": resolution[
                    "research_window_contract"
                ],
                "research_window_contract_sha256": resolution[
                    "research_window_contract_sha256"
                ],
                "label_horizon_sessions": primary_label_horizon_sessions(
                    horizon_profile
                ),
            }
        )
        if label_binding is None:
            raise ValueError("active horizon factor champion has no label binding")
        bundle = build_horizon_factor_bundle(
            feature_set=feature_set,
            incremental_factors=[],
            research_label_binding=label_binding,
        )
        return self.store.patch_cycle_state(
            str(cycle["id"]),
            state_patch={
                "horizon_factor_bundle": bundle,
                "horizon_factor_bundle_sha256": bundle["bundle_sha256"],
                "horizon_factor_bundle_feature_set_id": feature_set["id"],
                "horizon_factor_bundle_research_label_binding_sha256": (
                    label_binding["binding_sha256"]
                ),
            },
        )

    def _quant_input_sha256(
        self, cycle: dict[str, Any], dataset: dict[str, Any]
    ) -> str | None:
        if _prediction_champion_identity_error(cycle, dataset) is not None:
            return None
        state = dict(cycle.get("state") or {})
        champion = state.get("prediction_champion")
        selection = state.get("prediction_champion_evidence")
        model_selection = state.get("model_champion_evidence")
        if (
            not isinstance(champion, dict)
            or not champion.get("candidate_id")
            or not isinstance(selection, dict)
            or canonical_sha256(
                {key: value for key, value in selection.items() if key != "evidence_sha256"}
            )
            != selection.get("evidence_sha256")
            or not isinstance(model_selection, dict)
            or model_selection.get("contract_version")
            != "model-family-champions-v2"
            or not isinstance(selection.get("global_multiple_testing"), dict)
            or selection["global_multiple_testing"].get("contract_version")
            != "prediction-finalist-multiple-testing-v2"
            or not isinstance(model_selection.get("multiple_testing"), dict)
            or model_selection["multiple_testing"].get("contract_version")
            != "model-full-multiple-testing-v2"
            or canonical_sha256(
                {
                    key: value
                    for key, value in model_selection.items()
                    if key != "evidence_sha256"
                }
            )
            != model_selection.get("evidence_sha256")
        ):
            return None
        horizon_profile = str(cycle.get("horizon_profile") or "")
        primary_label_horizon_sessions(horizon_profile)
        sota_feature_set_id = self._active_sota_feature_set_id(
            dataset, horizon_profile=horizon_profile
        )
        sota_definition_sha256 = (
            str(get_feature_set(sota_feature_set_id)["definition_sha256"])
            if sota_feature_set_id
            else None
        )
        # This is the research-change identity, not the evaluation-vintage
        # identity.  Daily data publication alone must not create a fresh
        # hypothesis; a changed champion or SOTA definition does.
        payload = {
            "contract_version": "fin-quant-research-input-v3",
            "prediction_champion_evidence_sha256": selection["evidence_sha256"],
            "prediction_champion_kind": champion.get("kind"),
            "prediction_champion_id": champion.get("candidate_id"),
            "prediction_champion_manifest_sha256": champion.get("manifest_sha256"),
            "prediction_champion_admission_evidence_sha256": champion.get(
                "admission_evidence_sha256"
            ),
            "model_selection_evidence_sha256": model_selection[
                "evidence_sha256"
            ],
            "primary_model_candidate_id": champion.get(
                "primary_model_candidate_id"
            ),
            "component_model_candidate_ids": sorted(
                str(item)
                for item in champion.get("component_model_candidate_ids") or []
            ),
            "active_sota_feature_set_definition_sha256": sota_definition_sha256,
        }
        return canonical_sha256(payload)

    def _quant_due(
        self,
        cycle: dict[str, Any],
        dataset: dict[str, Any],
        now: datetime,
        config: dict[str, Any],
        *,
        input_sha256: str | None,
    ) -> bool:
        if input_sha256 is None:
            return False
        if _prediction_champion_identity_error(cycle, dataset) is not None:
            return False
        state = dict(cycle.get("state") or {})
        champion = dict(state.get("prediction_champion") or {})
        selection = dict(state.get("prediction_champion_evidence") or {})
        source_identity = str(selection.get("dataset_identity_sha256") or "")
        current_identity = str(
            (dataset.get("provenance") or {}).get("dataset_identity_sha256") or ""
        )
        if len(source_identity) != 64 or source_identity != current_identity:
            return False
        with self.engine.connect() as connection:
            if champion.get("kind") == "model":
                has_prediction_champion = connection.scalar(
                    select(model_candidates.c.id).where(
                        model_candidates.c.id == str(champion.get("candidate_id") or ""),
                        model_candidates.c.status == "research_admitted",
                        model_candidates.c.dataset_identity_sha256 == source_identity,
                        model_candidates.c.dataset_lineage_id
                        == str(cycle["dataset_lineage_id"]),
                    )
                )
            elif champion.get("kind") == "ensemble":
                has_prediction_champion = connection.scalar(
                    select(model_ensemble_candidates.c.id).where(
                        model_ensemble_candidates.c.id
                        == str(champion.get("candidate_id") or ""),
                        model_ensemble_candidates.c.status == "research_admitted",
                        model_ensemble_candidates.c.dataset_identity_sha256
                        == source_identity,
                    )
                )
            else:
                return False
            active_parallel = connection.scalar(
                select(autopilot_branches.c.id)
                .where(
                    autopilot_branches.c.cycle_id == str(cycle["id"]),
                    autopilot_branches.c.scenario.in_(
                        ("fin_factor", "fin_model", "fin_factor_report", "factor_sota")
                    ),
                    autopilot_branches.c.status.in_(("queued", "running", "evaluating")),
                )
                .limit(1)
            )
        if has_prediction_champion is None or active_parallel is not None:
            return False
        horizon_profile = str(cycle.get("horizon_profile") or "")
        primary_label_horizon_sessions(horizon_profile)
        latest = self.store.latest_branch(
            "fin_quant", horizon_profile=horizon_profile
        )
        if latest is None:
            return True
        if (latest.get("details") or {}).get("quant_input_sha256") == input_sha256:
            return False
        if str(latest.get("cycle_id") or "") == str(cycle["id"]):
            return False
        return _branch_created_at(latest) <= now - timedelta(
            days=config["quant_cooldown_days"]
        )

    def _enqueue_reports(
        self,
        cycle: dict[str, Any],
        dataset: dict[str, Any],
        *,
        config: dict[str, Any],
        now: datetime,
    ) -> int:
        local = now.astimezone(ZoneInfo("Asia/Shanghai")).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        remaining = config["report_daily_limit"] - self.store.count_since(
            "fin_factor_report", local.astimezone(UTC)
        )
        if remaining <= 0:
            return 0
        available = [
            item
            for item in self.assets.list_assets(limit=1000)
            if item.get("asset_type") == "research_report"
            and item.get("status") == "registered"
            and item.get("consumption") is None
        ]
        calendar = (
            (Path(str(dataset["path"])) / "calendars" / "day.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        periods, _resolution = resolve_research_periods(calendar)
        pre_final_end = date.fromisoformat(periods["valid_end"])
        available = [
            item
            for item in available
            if _asset_available_on(item) is not None
            and _asset_available_on(item) <= pre_final_end
        ]
        created = 0
        # One active report run at a time; later scheduler ticks drain the rest.
        for asset in available[: min(remaining, 1)]:
            scope = str(asset["id"])
            if self.store.branch_for_scope(cycle["id"], "fin_factor_report", scope):
                continue
            self._enqueue(
                cycle,
                dataset,
                "fin_factor_report",
                scope,
                config=config,
                asset_ids=[scope],
            )
            created += 1
        return created

    def _enqueue(
        self,
        cycle: dict[str, Any],
        dataset: dict[str, Any],
        scenario_id: str,
        scope_key: str,
        *,
        config: dict[str, Any],
        asset_ids: list[str] | None = None,
        feature_set_id: str | None = None,
        tournament_id: str | None = None,
        branch_details: dict[str, Any] | None = None,
    ) -> None:
        scenario = get_rdagent_scenario(scenario_id)
        runtime = probe_rdagent(self.settings, Path(__file__).resolve().parents[2])
        require_ready_scenario(runtime, self.settings, scenario_id)
        calendar = (
            (Path(str(dataset["path"])) / "calendars" / "day.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        feature_set = (
            get_feature_set(
                feature_set_id
                or config[
                    "model_feature_set_id"
                    if scenario_id == "fin_model"
                    else "quant_feature_set_id"
                ]
            )
            if scenario.requires_feature_set
            else None
        )
        active_horizon = (
            str(cycle.get("horizon_profile") or "")
            if scenario_id in {"fin_factor", "fin_model", "fin_quant"}
            else None
        )
        if active_horizon is not None:
            primary_label = primary_label_horizon_sessions(active_horizon)
            policy = primary_label_policy_contract()
            if (
                cycle.get("primary_label_policy_sha256") != policy["policy_sha256"]
                or (cycle.get("state") or {}).get("primary_label_policy") != policy
            ):
                raise ValueError("autopilot cycle primary-label policy changed")
            periods, resolution = resolve_research_window_contract(
                dataset,
                calendar,
                horizon_profile=active_horizon,
                feature_set=feature_set,
            )
        else:
            periods, resolution = resolve_research_periods(calendar)
        resolved_assets = resolve_rdagent_assets(
            self.settings,
            scenario,
            asset_ids or [],
            pre_final_end=date.fromisoformat(periods["valid_end"]),
        )
        config_prefix = {
            "fin_factor": "factor",
            "fin_model": "model",
            "fin_factor_report": "report",
            "fin_quant": "quant",
        }[scenario_id]
        loop_n = int(config[f"{config_prefix}_loop_n"])
        duration = str(config[f"{config_prefix}_duration"])
        objective = {
            "fin_factor": (
                "Discover economically distinct A-share factors under strict PIT "
                "and cost governance."
            ),
            "fin_model": (
                "Research reproducible prediction models on the frozen governed "
                "feature set."
            ),
            "fin_factor_report": (
                "Extract testable factor hypotheses from this verified research "
                "report PDF."
            ),
            "fin_quant": (
                "Jointly improve the frozen factor set and prediction model with "
                "governed ablations."
            ),
        }[scenario_id]
        expected_runtime = expected_rdagent_runtime_identity(runtime, scenario_id)
        runtime_fingerprint = hashlib.sha256(
            json.dumps(
                expected_runtime, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()[:16]
        manifest_sha256 = resolved_assets["manifest_sha256"]
        run = self.research.create_run(
            kind=(
                f"{scenario.research_kind}:{active_horizon}"
                if active_horizon is not None
                else scenario.research_kind
            ),
            objective=objective,
            dataset=str(dataset["name"]),
            requested_by="autopilot",
            budget={"loop_n": loop_n, "duration": duration},
            config={
                "scenario": scenario_id,
                "periods": periods,
                "evaluation_profiles": resolution["evaluation_profiles"],
                "period_resolution": resolution,
                "dataset_path": dataset["path"],
                "dataset_identity_sha256": dataset["provenance"]["dataset_identity_sha256"],
                "autopilot_cycle_id": cycle["id"],
                "research_tournament_id": tournament_id,
                "asset_ids": list(manifest_sha256),
                "asset_manifest_sha256": manifest_sha256,
                "feature_set": feature_set,
                **(
                    {
                        "horizon_profile": active_horizon,
                        "primary_label_policy": policy,
                        "primary_label_policy_sha256": policy["policy_sha256"],
                        "research_window_contract": resolution[
                            "research_window_contract"
                        ],
                        "research_window_contract_sha256": resolution[
                            "research_window_contract_sha256"
                        ],
                        "label_horizon_sessions": primary_label,
                    }
                    if active_horizon is not None
                    else {}
                ),
                "expected_rdagent_runtime": expected_runtime,
                **(
                    {
                        "prediction_champion": dict(
                            (branch_details or {}).get("prediction_champion") or {}
                        ),
                        "prediction_champion_evidence": dict(
                            (branch_details or {}).get(
                                "prediction_champion_evidence"
                            )
                            or {}
                        ),
                    }
                    if scenario_id == "fin_quant"
                    else {}
                ),
            },
            artifact_path=self.settings.data_root / "artifacts" / "rdagent",
        )
        try:
            if manifest_sha256:
                self.assets.reserve_automatic(
                    research_run_id=run["id"],
                    scenario=scenario_id,
                    asset_manifest_sha256=manifest_sha256,
                    actor="autopilot",
                )
            job = self.jobs.create(
                scenario.job_kind,
                {
                    "scenario": scenario_id,
                    "research_run_id": run["id"],
                    "dataset": dataset["name"],
                    "dataset_path": dataset["path"],
                    "dataset_identity_sha256": dataset["provenance"][
                        "dataset_identity_sha256"
                    ],
                    "dataset_lineage_id": dataset["lineage_id"],
                    "objective": objective,
                    "loop_n": loop_n,
                    "duration": duration,
                    "periods": periods,
                    "evaluation_profiles": resolution["evaluation_profiles"],
                    "period_resolution": resolution,
                    "asset_ids": list(manifest_sha256),
                    "asset_manifest_sha256": manifest_sha256,
                    "feature_set": feature_set,
                    **(
                        {
                            "horizon_profile": active_horizon,
                            "primary_label_policy": policy,
                            "primary_label_policy_sha256": policy[
                                "policy_sha256"
                            ],
                            "research_window_contract": resolution[
                                "research_window_contract"
                            ],
                            "research_window_contract_sha256": resolution[
                                "research_window_contract_sha256"
                            ],
                            "label_horizon_sessions": primary_label,
                        }
                        if active_horizon is not None
                        else {}
                    ),
                    "research_tournament_id": tournament_id,
                    "expected_rdagent_runtime": expected_runtime,
                    **(
                        {
                            "prediction_champion": dict(
                                (branch_details or {}).get(
                                    "prediction_champion"
                                )
                                or {}
                            ),
                            "prediction_champion_evidence": dict(
                                (branch_details or {}).get(
                                    "prediction_champion_evidence"
                                )
                                or {}
                            ),
                        }
                        if scenario_id == "fin_quant"
                        else {}
                    ),
                },
                self.settings.data_root
                / "platform"
                / "logs"
                / f"autopilot-{scenario_id}-{run['id']}.log",
                dedupe_active_kind=False,
                idempotency_key=(
                    f"autopilot:{cycle['id']}:{scenario_id}:{scope_key}:"
                    f"{RDAGENT_INTEGRATION_CONTRACT_VERSION}:{runtime_fingerprint}"
                ),
            )
            self.research.attach_job(run["id"], job["id"])
            self.store.create_branch(
                cycle["id"],
                scenario=scenario_id,
                scope_key=scope_key,
                research_run_id=run["id"],
                job_id=job["id"],
                details={
                    "dataset_identity_sha256": dataset["provenance"][
                        "dataset_identity_sha256"
                    ],
                    "feature_set_id": feature_set["id"] if feature_set else None,
                    "horizon_profile": active_horizon,
                    "primary_label_policy_sha256": (
                        policy["policy_sha256"]
                        if active_horizon is not None
                        else None
                    ),
                    "research_tournament_id": tournament_id,
                    "asset_ids": list(manifest_sha256),
                    "final_oos_opened": False,
                    **dict(branch_details or {}),
                },
            )
        except Exception as exc:
            self.research.mark_run(
                run["id"],
                "failed",
                actor="autopilot",
                error=f"autopilot enqueue failed: {exc}",
            )
            raise
