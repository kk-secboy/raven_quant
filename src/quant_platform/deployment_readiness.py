from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import distinct, func, select, text

from quant_data.cninfo_announcements import load_trade_calendar_open_days
from quant_data.config import Settings
from quant_data.database import (
    alerts,
    allocation_schedule_groups,
    audit_events,
    open_database,
    recommendation_portfolios,
    recommendation_snapshots,
    schedules,
    simulation_batches,
    simulation_nav,
    simulation_portfolios,
    strategy_allocation_events,
    strategy_allocation_members,
    strategy_allocation_nav,
    strategy_allocations,
    strategy_forward_only_rehabilitations,
    strategy_health_snapshots,
    strategy_promotion_stages,
    strategy_versions,
    users,
)
from quant_data.execution_contract import (
    require_daily_qlib_contract,
    require_native_daily_execution_controls,
)

from .data_task_store import DataTaskStore
from .feature_drift import validate_factor_psi_observation
from .health_store import OperationalHealthStore
from .information_schedule import (
    STRUCTURED_INFORMATION_SOURCES,
    normalize_information_factor_refresh_payload,
    normalize_information_schedule_payload,
    resolve_information_evaluation_dataset,
)
from .model_calibration_drift import validate_model_calibration_observation
from .replay_governance import (
    TERMINAL_CASH_ONLY_BACKTEST_ID,
    TERMINAL_CASH_ONLY_CONTRACT_VERSION,
    TERMINAL_CASH_ONLY_JOB_ID,
    TERMINAL_CASH_ONLY_VERSION_ID,
    require_terminal_cash_only_receipt,
)
from .research_automation import (
    DEFAULT_REQUIRED_RESEARCH_TRADING_DAYS,
    DEFAULT_RESEARCH_PERIOD_POLICY,
    MINIMUM_PROFILE_TRAINING_DAYS,
    RESEARCH_EVALUATION_PROFILES,
)
from .research_horizon import canonical_sha256
from .runtime_secret_store import RuntimeSecretStore
from .schedule_store import ACTIVE_SCHEDULE_KINDS
from .scheduler import AUTOMATED_DATA_BUNDLES
from .services import list_qlib_datasets_for_display
from .strategy_recipes import TRANSPARENT_RESEARCH_BASELINE_IDS, get_strategy_recipe
from .transparent_baseline_lockbox import (
    ALL_UNAVAILABLE_CASH_ONLY_ACTION,
    ALL_UNAVAILABLE_CASH_ONLY_AUTHORITY,
    ALL_UNAVAILABLE_CASH_ONLY_CONTRACT_VERSION,
    ALL_UNAVAILABLE_CASH_ONLY_RUNNER,
    LOCKBOX_CONFIG_KEY,
    LOCKBOX_CONTRACT_VERSION_V3,
    validate_all_unavailable_cash_only_audit_event,
    validate_joint_lockbox,
)
from .transparent_baseline_runner import (
    FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION,
)

RESEARCH_LONGEST_VALIDATION_TRADING_DAYS = max(
    int(profile["validation_trading_days"])
    for profile in RESEARCH_EVALUATION_PROFILES
)
RESEARCH_MINIMUM_TRADING_DAYS = DEFAULT_REQUIRED_RESEARCH_TRADING_DAYS
_GOVERNED_DATA_PIPELINE_BUNDLES = frozenset(AUTOMATED_DATA_BUNDLES)
_GOVERNED_INFORMATION_CORPUS_DATASETS = frozenset(
    {"cctv_news", "irm_qa_sh", "irm_qa_sz", "major_news"}
)
_GOVERNED_DATA_SCHEDULE_SUITE_KINDS = frozenset(
    {
        "data_pipeline",
        "information_pipeline",
        "information_factor_refresh",
        "ashare_5m_sync",
        "auxiliary_data_pipeline",
    }
)
_PRODUCT_HORIZONS = ("short_1_5d", "swing_1_6m", "long_1_3y")
_OPERABLE_PROMOTION_STAGES = frozenset({"paper", "recommendation_enabled"})
_NON_BLOCKING_PAPER_HEALTH = frozenset(
    {"healthy", "watch", "insufficient_evidence"}
)
_NON_BLOCKING_RECOMMENDATION_HEALTH = frozenset({"healthy", "watch"})
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_DAILY_CLOSE = time(15, 0)
_STRATEGY_HEALTH_COLLECTOR_ACTOR = "system:strategy-health-collector"
_TRANSPARENT_BASELINE_BOOTSTRAP_ACTOR = "system:transparent-baseline-bootstrap"
_FORWARD_ONLY_REHABILITATION_CONTRACT_VERSION = "forward-only-rehabilitation-v1"
_CONSUMED_HISTORICAL_REPLAY = "consumed_historical_replay"
_HISTORICAL_DESCRIPTION_ONLY = "historical_description_only"
_SOURCE_CASH_ONLY_SCOPE = "cash_only_projection_only"
_SOURCE_TRANSPARENT_BASELINE_VERSION_ID = "4414d202dbb641608975e5305bc18da4"


def _latest_closed_trading_day(
    open_days: list[date],
    *,
    now: datetime,
) -> date:
    """Return the last exchange-open day whose daily bar may be complete.

    The persisted SSE calendar is authoritative.  On an open day before the
    15:00 close, today's unfinished bar is deliberately excluded.
    """

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("business-readiness time must be timezone-aware")
    local = now.astimezone(_SHANGHAI)
    cutoff = local.date()
    if local.timetz().replace(tzinfo=None) < _DAILY_CLOSE:
        cutoff -= timedelta(days=1)
    eligible = [value for value in open_days if value <= cutoff]
    if not eligible:
        raise ValueError("trade calendar has no closed trading day")
    return max(eligible)


def _daily_qlib_business_check(
    data_root: Path,
    *,
    now: datetime,
) -> dict[str, Any]:
    """Check the publisher projection used by the daily production loop.

    This endpoint must remain bounded, so it reads the worker-published Qlib
    catalog projection.  The projection itself records the strict sealed-file
    verification result; an unsealed or unverifiable dataset never qualifies.
    """

    try:
        open_days = load_trade_calendar_open_days(data_root)
        expected = _latest_closed_trading_day(open_days, now=now)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "status": "blocked",
            "message": "the persisted exchange calendar is unavailable or incomplete",
            "reason": str(exc)[:500],
            "expected_end_date": None,
            "eligible_datasets": [],
        }

    eligible: list[dict[str, Any]] = []
    observed: list[dict[str, Any]] = []
    for dataset in list_qlib_datasets_for_display(data_root):
        if str(dataset.get("frequency") or "") != "day":
            continue
        name = str(dataset.get("name") or "")
        end_text = str(dataset.get("end_date") or "")[:10]
        reasons: list[str] = []
        try:
            end_date = date.fromisoformat(end_text)
        except ValueError:
            end_date = None
            reasons.append("invalid_end_date")
        provenance = dataset.get("daily_contract")
        if not isinstance(provenance, dict):
            # Direct unit fixtures and old in-memory publishers may still
            # provide the authoritative provenance object. Persisted browser
            # projections use the bounded daily-contract subset.
            provenance = dataset.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
        if dataset.get("ready") is not True:
            reasons.append("not_ready")
        if dataset.get("reproducible") is not True:
            reasons.append("not_reproducible")
        if dataset.get("lineage_verified") is not True:
            reasons.append("lineage_unverified")
        if dataset.get("output_files_verified") is not True:
            reasons.append("sealed_outputs_unverified")
        try:
            require_daily_qlib_contract(provenance)
            require_native_daily_execution_controls(provenance, start=expected)
        except (TypeError, ValueError) as exc:
            reasons.append(f"daily_contract:{exc}")
        if end_date is None or end_date < expected:
            reasons.append("stale")
        evidence = {
            "name": name,
            "end_date": end_text or None,
            "output_verification": dataset.get("output_verification"),
            "reasons": reasons,
        }
        observed.append(evidence)
        if not reasons:
            eligible.append(evidence)
    latest_observed = max(
        (str(item.get("end_date") or "") for item in observed),
        default=None,
    )
    ready = bool(eligible)
    return {
        "status": "ok" if ready else "blocked",
        "message": (
            "a sealed daily Qlib dataset covers the latest closed trading day"
            if ready
            else "no sealed daily Qlib dataset covers the latest closed trading day"
        ),
        "expected_end_date": expected.isoformat(),
        "latest_observed_end_date": latest_observed,
        "eligible_datasets": [item["name"] for item in eligible],
        "datasets": observed,
    }


def _strategy_health_evidence_check(
    snapshot: Any,
    *,
    version_id: str,
    horizon_profile: str,
    horizon_contract_sha256: str,
    current_dataset_identity_sha256: str | None,
    expected_trade_date: date,
    now: datetime,
    max_age_seconds: int,
    current_dataset_lineage_id: str | None = None,
    current_batch_id: str | None = None,
    current_source_snapshot_id: str | None = None,
    expected_feature_date: date | None = None,
) -> dict[str, Any]:
    """Verify the newest activity-health row and its embedded drift receipt."""

    if snapshot is None:
        return {"ready": False, "reasons": ["strategy_health_evidence_missing"]}
    row = snapshot._mapping if hasattr(snapshot, "_mapping") else snapshot
    values = dict(row)
    reasons: list[str] = []
    criteria = dict(values.get("criteria_json") or {})
    evidence = dict(values.get("evidence_json") or {})
    criteria_sha256 = str(values.get("criteria_sha256") or "")
    evidence_sha256 = str(values.get("evidence_sha256") or "")
    if canonical_sha256(criteria) != criteria_sha256:
        reasons.append("strategy_health_criteria_seal_invalid")
    if canonical_sha256(evidence) != evidence_sha256:
        reasons.append("strategy_health_evidence_seal_invalid")
    as_of = values.get("as_of")
    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        reasons.append("strategy_health_as_of_invalid")
    else:
        normalized_as_of = as_of.astimezone(UTC).replace(microsecond=0)
        age_seconds = (now.astimezone(UTC) - normalized_as_of).total_seconds()
        if age_seconds < -300:
            reasons.append("strategy_health_evidence_from_future")
        elif age_seconds > max_age_seconds:
            reasons.append("strategy_health_evidence_stale")
        snapshot_payload = {
            "contract_version": "strategy-health-snapshot-v1",
            "strategy_version_id": version_id,
            "horizon_profile": horizon_profile,
            "horizon_contract_sha256": horizon_contract_sha256,
            "as_of": normalized_as_of.isoformat(),
            "health_status": str(values.get("health_status") or ""),
            "criteria_json": criteria,
            "criteria_sha256": criteria_sha256,
            "evidence_json": evidence,
            "evidence_sha256": evidence_sha256,
            "recorded_by": str(values.get("recorded_by") or ""),
        }
        expected_snapshot_sha256 = canonical_sha256(snapshot_payload)
        if (
            str(values.get("id") or "") != expected_snapshot_sha256
            or str(values.get("snapshot_sha256") or "") != expected_snapshot_sha256
        ):
            reasons.append("strategy_health_snapshot_seal_invalid")
    if values.get("recorded_by") != _STRATEGY_HEALTH_COLLECTOR_ACTOR:
        reasons.append("strategy_health_not_periodically_collected")
    if evidence.get("contract_version") != "strategy-health-live-evidence-v2":
        reasons.append("strategy_health_live_evidence_missing")
    expected_text = expected_trade_date.isoformat()
    if evidence.get("evidence_trade_date") != expected_text:
        reasons.append("strategy_health_nav_evidence_stale")
    feature_date = expected_feature_date or expected_trade_date
    feature_text = feature_date.isoformat()
    if evidence.get("feature_drift_current_end") != feature_text:
        reasons.append("strategy_health_feature_evidence_stale")
    if evidence.get("feature_signal_date") != feature_text:
        reasons.append("strategy_health_feature_signal_binding_invalid")
    if evidence.get("simulation_batch_id") != current_batch_id:
        reasons.append("strategy_health_batch_binding_invalid")
    if evidence.get("daily_dataset_identity_sha256") != current_dataset_identity_sha256:
        reasons.append("strategy_health_dataset_binding_invalid")
    if evidence.get("daily_dataset_lineage_id") != current_dataset_lineage_id:
        reasons.append("strategy_health_dataset_lineage_invalid")
    if evidence.get("source_snapshot_id") != current_source_snapshot_id:
        reasons.append("strategy_health_source_snapshot_invalid")
    if evidence.get("feature_drift_evidence_available") is not True:
        reasons.append("strategy_health_feature_evidence_missing")
    if not current_dataset_identity_sha256:
        reasons.append("strategy_health_dataset_identity_missing")
    else:
        try:
            observation = validate_factor_psi_observation(
                evidence.get("feature_drift_observation"),
                strategy_version_id=version_id,
                current_dataset_identity_sha256=current_dataset_identity_sha256,
                expected_as_of=feature_date,
            )
            if (
                observation.get("observation_sha256")
                != evidence.get("feature_drift_observation_sha256")
                or float(observation.get("feature_drift"))
                != float(evidence.get("feature_drift"))
            ):
                reasons.append("strategy_health_feature_evidence_mismatch")
        except (TypeError, ValueError):
            reasons.append("strategy_health_feature_evidence_invalid")
    model_required = evidence.get("model_calibration_required") is True
    if model_required:
        if evidence.get("model_calibration_evidence_available") is not True:
            reasons.append("strategy_health_model_calibration_missing")
        else:
            try:
                model_observation = validate_model_calibration_observation(
                    evidence.get("model_calibration_observation"),
                    strategy_version_id=version_id,
                    current_dataset_identity_sha256=str(
                        current_dataset_identity_sha256 or ""
                    ),
                    expected_as_of=feature_date,
                )
                if (
                    model_observation.get("observation_sha256")
                    != evidence.get("model_calibration_observation_sha256")
                    or float(model_observation["model_calibration_drift"])
                    != float(evidence.get("model_calibration_drift"))
                ):
                    reasons.append("strategy_health_model_calibration_mismatch")
            except (TypeError, ValueError):
                reasons.append("strategy_health_model_calibration_invalid")
    return {
        "ready": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "as_of": as_of.isoformat() if isinstance(as_of, datetime) else None,
        "evidence_trade_date": evidence.get("evidence_trade_date"),
        "feature_drift_current_end": evidence.get("feature_drift_current_end"),
        "snapshot_sha256": values.get("snapshot_sha256"),
    }


def _assess_horizon_candidates(
    horizon: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Choose the authoritative runnable lane for one product horizon.

    A verified recommendation version is authoritative when present and may
    not silently fall back to an older paper candidate.  Before its evidence
    matures, an active isolated paper account is a valid production lane.
    """

    recommendation = [
        item
        for item in candidates
        if item.get("promotion_stage") == "recommendation_enabled"
    ]
    pool = recommendation or [
        item for item in candidates if item.get("promotion_stage") == "paper"
    ]
    evaluated: list[dict[str, Any]] = []
    for raw in pool:
        item = dict(raw)
        stage = str(item.get("promotion_stage") or "")
        raw_health = item.get("health_status")
        health = str(raw_health) if raw_health else "missing"
        contract_ready = bool(item.get("contract_ready"))
        daily_execution = (
            item.get("signal_frequency") == "day"
            and item.get("execution_frequency") == "day"
        )
        if stage == "recommendation_enabled":
            runner_ready = int(item.get("active_recommendation_portfolios") or 0) > 0
            health_ready = health in _NON_BLOCKING_RECOMMENDATION_HEALTH
            runner = "daily_recommendation_refresh"
        else:
            runner_ready = (
                item.get("paper_stage_status") == "active"
                and item.get("simulation_status") == "active"
            )
            health_ready = health in _NON_BLOCKING_PAPER_HEALTH
            runner = "daily_isolated_paper_order_plan"
        blockers = []
        if not contract_ready:
            blockers.append("strategy_contract_invalid")
        if not daily_execution:
            blockers.append("daily_execution_contract_missing")
        if not runner_ready:
            blockers.append("production_runner_unavailable")
        if not health_ready:
            blockers.append(f"strategy_health_{health}")
        if item.get("health_evidence_ready") is not True:
            blockers.extend(
                str(reason)
                for reason in item.get("health_evidence_reasons") or [
                    "strategy_health_evidence_missing"
                ]
            )
        item.update(
            {
                "health_status": health,
                "runner": runner,
                "runner_ready": runner_ready,
                "blocking_reasons": blockers,
                "operable": not blockers,
            }
        )
        evaluated.append(item)
    # Database callers provide newest-first candidates, matching AdviceService.
    # Do not hide a broken current version by falling back to stale evidence.
    authoritative = evaluated[0] if evaluated else None
    selected = authoritative if authoritative and authoritative["operable"] else None
    return {
        "horizon": horizon,
        "status": "ok" if selected is not None else "blocked",
        "stage": selected.get("promotion_stage") if selected else None,
        "strategy_version_id": selected.get("strategy_version_id") if selected else None,
        "health_status": selected.get("health_status") if selected else None,
        "runner": selected.get("runner") if selected else None,
        "message": (
            "horizon has an operable governed production lane"
            if selected is not None
            else "horizon has no operable governed production lane"
        ),
        "candidates": evaluated,
    }


def _validated_cash_only_horizon_lanes(
    rows: Sequence[Any],
) -> dict[str, dict[str, Any]]:
    """Project unavailable horizons from the current governed lockbox.

    ``rows`` must be ordered newest-first.  A cash-only lane is deliberately
    narrower than a strategy lane: it is accepted only from the current recipe
    version's fully validated v3 joint lockbox.  Runtime failures, missing
    versions and free-form error strings never enter this projection.
    """

    current_recipe_versions = {
        recipe_id: str(get_strategy_recipe(recipe_id)["version"])
        for recipe_id in TRANSPARENT_RESEARCH_BASELINE_IDS
    }
    current_rows: list[tuple[dict[str, Any], Any]] = []
    for raw_row in rows:
        row = raw_row._mapping if hasattr(raw_row, "_mapping") else raw_row
        if not isinstance(row, Mapping):
            continue
        values = dict(row)
        config = values.get("config_json")
        if not isinstance(config, Mapping):
            continue
        recipe_id = str(config.get("recipe_id") or "")
        if (
            recipe_id not in current_recipe_versions
            or str(config.get("recipe_version") or "")
            != current_recipe_versions[recipe_id]
        ):
            continue
        if values.get("created_by") != _TRANSPARENT_BASELINE_BOOTSTRAP_ACTOR:
            continue
        current_rows.append((values, config.get(LOCKBOX_CONFIG_KEY)))
    if not current_rows:
        return {}

    # The newest current-recipe row defines the current batch.  If that batch
    # is malformed, fail closed instead of falling back to an older lockbox.
    newest_lockbox = current_rows[0][1]
    if not isinstance(newest_lockbox, Mapping):
        return {}
    current_batch_sha256 = str(newest_lockbox.get("batch_sha256") or "")
    if len(current_batch_sha256) != 64:
        return {}
    batch_rows = [
        (row, raw_lockbox)
        for row, raw_lockbox in current_rows
        if isinstance(raw_lockbox, Mapping)
        and str(raw_lockbox.get("batch_sha256") or "") == current_batch_sha256
    ]
    try:
        validated = [validate_joint_lockbox(raw) for _, raw in batch_rows]
    except (TypeError, ValueError):
        return {}
    if not validated or any(item != validated[0] for item in validated[1:]):
        return {}
    lockbox = validated[0]
    if lockbox.get("contract_version") != LOCKBOX_CONTRACT_VERSION_V3:
        return {}
    selection = lockbox.get("unopened_history_selection")
    current_versions = set(current_recipe_versions.values())
    if (
        not isinstance(selection, Mapping)
        or len(current_versions) != 1
        or str(selection.get("current_recipe_version") or "")
        != next(iter(current_versions))
    ):
        return {}

    declared_members = {
        (
            str(member["recipe_id"]),
            str(member["recipe_version"]),
            str(member["horizon_profile"]),
        )
        for member in lockbox["members"]
    }
    observed_members: set[tuple[str, str, str]] = set()
    for row, _ in batch_rows:
        config = row["config_json"]
        recipe_id = str(config.get("recipe_id") or "")
        member = (
            recipe_id,
            str(config.get("recipe_version") or ""),
            str(config.get("horizon_profile") or ""),
        )
        if (
            member in declared_members
            and str(row.get("horizon_profile") or member[2]) == member[2]
        ):
            observed_members.add(member)
    if observed_members != declared_members:
        return {}

    lanes: dict[str, dict[str, Any]] = {}
    for item in lockbox.get("unavailable_horizons") or []:
        horizon = str(item["horizon_profile"])
        lanes[horizon] = {
            "horizon": horizon,
            "status": "ok",
            "stage": "cash_only",
            "strategy_version_id": None,
            "health_status": "not_applicable",
            "runner": "cash_only_no_orders",
            "message": (
                "horizon is sealed unavailable and its sleeve remains in cash"
            ),
            "candidates": [],
            "cash_only": True,
            "sleeve_action": "remain_in_cash",
            "new_entries_allowed": False,
            "recommendation_eligible": False,
            "lockbox_evidence": {
                "batch_sha256": lockbox["batch_sha256"],
                "recipe_id": item["recipe_id"],
                "status": item["status"],
                "reason": item["reason"],
                "evidence_sha256": item["evidence_sha256"],
            },
        }
    return lanes


def _validated_all_unavailable_cash_only_horizon_lanes(
    audit_rows: Sequence[Any],
    *,
    lockbox_rows: Sequence[Any],
) -> dict[str, dict[str, Any]]:
    """Project the newest current-recipe no-OOS receipt as inert cash sleeves.

    Any current-recipe StrategyVersion supersedes this standalone declaration:
    partial unavailability must then come from that version's v3 joint lockbox.
    A malformed or stale newest receipt fails closed and is never replaced by
    an older declaration.
    """

    current_versions = {
        recipe_id: str(get_strategy_recipe(recipe_id)["version"])
        for recipe_id in TRANSPARENT_RESEARCH_BASELINE_IDS
    }
    if len(set(current_versions.values())) != 1:
        return {}
    current_recipe_version = next(iter(current_versions.values()))
    for raw_row in lockbox_rows:
        row = raw_row._mapping if hasattr(raw_row, "_mapping") else raw_row
        if not isinstance(row, Mapping):
            continue
        config = row.get("config_json")
        if (
            isinstance(config, Mapping)
            and str(config.get("recipe_id") or "") in current_versions
            and str(config.get("recipe_version") or "")
            == current_recipe_version
            and row.get("created_by") == _TRANSPARENT_BASELINE_BOOTSTRAP_ACTOR
        ):
            return {}
    if not audit_rows:
        return {}
    try:
        receipt = validate_all_unavailable_cash_only_audit_event(
            audit_rows[0],
            expected_recipe_version=current_recipe_version,
        )
    except (TypeError, ValueError):
        return {}
    if (
        receipt.get("contract_version")
        != ALL_UNAVAILABLE_CASH_ONLY_CONTRACT_VERSION
        or receipt.get("authority") != ALL_UNAVAILABLE_CASH_ONLY_AUTHORITY
        or receipt.get("cash_only_scope") != ALL_UNAVAILABLE_CASH_ONLY_AUTHORITY
        or receipt.get("runner") != ALL_UNAVAILABLE_CASH_ONLY_RUNNER
        or receipt.get("strategy_version_created") is not False
        or receipt.get("oos_reserved") is not False
        or receipt.get("paper_eligible") is not False
        or receipt.get("recommendation_eligible") is not False
        or receipt.get("orders_eligible") is not False
    ):
        return {}
    lanes: dict[str, dict[str, Any]] = {}
    for item in receipt["unavailable_horizons"]:
        horizon = str(item["horizon_profile"])
        lanes[horizon] = {
            "horizon": horizon,
            "status": "ok",
            "stage": "cash_only",
            "strategy_version_id": None,
            "health_status": "not_applicable",
            "runner": ALL_UNAVAILABLE_CASH_ONLY_RUNNER,
            "message": (
                "the current public baseline has no honest unopened OOS window; "
                "this sleeve remains in cash"
            ),
            "candidates": [],
            "cash_only": True,
            "sleeve_action": "remain_in_cash",
            "new_entries_allowed": False,
            "recommendation_eligible": False,
            "cash_only_evidence": {
                "receipt_sha256": receipt["receipt_sha256"],
                "authority": receipt["authority"],
                "dataset": receipt["dataset"],
                "dataset_identity_sha256": receipt[
                    "dataset_identity_sha256"
                ],
                "dataset_lineage_id": receipt["dataset_lineage_id"],
                "current_recipe_version": receipt["current_recipe_version"],
                "history_selection_sha256": receipt[
                    "unopened_history_selection"
                ]["selection_sha256"],
                "recipe_id": item["recipe_id"],
                "unavailable_evidence_sha256": item["evidence_sha256"],
                "strategy_version_created": False,
                "oos_reserved": False,
                "orders_eligible": False,
            },
        }
    return lanes


def _validated_rehabilitation_cash_only_horizon_lanes(
    rows: Sequence[Any],
    *,
    short_lane: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Project source-bound cash sleeves for the exact operable replay version.

    A rehabilitation receipt describes an already-opened historical replay.  It
    is therefore never treated as a current-recipe lockbox or sealed final OOS.
    The receipt may only carry forward the source lockbox's conservative
    swing/long unavailability after the same target short version has reached an
    operable paper or recommendation stage.
    """

    selected_version_id = str(short_lane.get("strategy_version_id") or "")
    selected_stage = str(short_lane.get("stage") or "")
    if (
        short_lane.get("status") != "ok"
        or not selected_version_id
        or selected_stage not in _OPERABLE_PROMOTION_STAGES
    ):
        return {}
    matching: list[dict[str, Any]] = []
    for raw_row in rows:
        row = raw_row._mapping if hasattr(raw_row, "_mapping") else raw_row
        if isinstance(row, Mapping) and str(row.get("strategy_version_id") or "") == (
            selected_version_id
        ):
            matching.append(dict(row))
    if len(matching) != 1:
        return {}
    row = matching[0]
    config = row.get("target_config_json")
    qualification = row.get("qualification_json")
    evidence_hashes = row.get("source_unavailable_evidence_sha256s_json")
    if (
        not isinstance(config, Mapping)
        or not isinstance(qualification, Mapping)
        or not isinstance(evidence_hashes, Mapping)
    ):
        return {}
    receipt_sha256 = str(row.get("receipt_sha256") or "")
    qualification_receipt = str(qualification.get("receipt_sha256") or "")
    qualification_core = {
        key: value for key, value in qualification.items() if key != "receipt_sha256"
    }
    rehabilitation_recipe_version = (
        FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
    )
    normalized_evidence_hashes = {
        str(key): str(value) for key, value in evidence_hashes.items()
    }
    expected_evidence_horizons = {"swing_1_6m", "long_1_3y"}
    hash_fields = (
        "source_lockbox_batch_sha256",
        "source_lockbox_member_sha256",
        "source_history_selection_sha256",
        "source_unavailable_horizons_sha256",
    )
    qualification_pairs = {
        "source_lockbox_contract_version": row.get(
            "source_lockbox_contract_version"
        ),
        "source_lockbox_batch_sha256": row.get("source_lockbox_batch_sha256"),
        "source_lockbox_member_sha256": row.get("source_lockbox_member_sha256"),
        "source_history_selection_sha256": row.get(
            "source_history_selection_sha256"
        ),
        "source_unavailable_horizons_sha256": row.get(
            "source_unavailable_horizons_sha256"
        ),
        "source_unavailable_evidence_sha256s": normalized_evidence_hashes,
        "source_cash_only_scope": row.get("source_cash_only_scope"),
    }
    if (
        receipt_sha256 != qualification_receipt
        or len(receipt_sha256) != 64
        or canonical_sha256(qualification_core) != receipt_sha256
        or row.get("contract_version")
        != _FORWARD_ONLY_REHABILITATION_CONTRACT_VERSION
        or row.get("evidence_mode") != _CONSUMED_HISTORICAL_REPLAY
        or row.get("authority") != _HISTORICAL_DESCRIPTION_ONLY
        or row.get("source_strategy_version_id")
        != _SOURCE_TRANSPARENT_BASELINE_VERSION_ID
        or row.get("source_lockbox_contract_version")
        != LOCKBOX_CONTRACT_VERSION_V3
        or row.get("source_cash_only_scope") != _SOURCE_CASH_ONLY_SCOPE
        or row.get("recipe_id") != "short_relative_strength"
        or row.get("horizon_profile") != "short_1_5d"
        or row.get("target_status") != "approved"
        or row.get("target_promotion_stage") != selected_stage
        or row.get("target_horizon_profile") != "short_1_5d"
        or row.get("target_evidence_mode") != _CONSUMED_HISTORICAL_REPLAY
        or config.get("recipe_id") != "short_relative_strength"
        or config.get("recipe_version") != rehabilitation_recipe_version
        or config.get("horizon_profile") != "short_1_5d"
        or config.get("evidence_mode") != _CONSUMED_HISTORICAL_REPLAY
        or qualification.get("historical_replay_opened") is not True
        or qualification.get("consumed_oos_replayed") is not True
        or qualification.get("final_oos_opened") is not True
        or qualification.get("capital_eligible") is not False
        or qualification.get("sealed_final_oos") is not False
        or qualification.get("unseen_oos") is not False
        or qualification.get("authority") != _HISTORICAL_DESCRIPTION_ONLY
        or qualification.get("strategy_version_id") != selected_version_id
        or qualification.get("source_strategy_version_id")
        != _SOURCE_TRANSPARENT_BASELINE_VERSION_ID
        or any(
            qualification.get(key) != value
            for key, value in qualification_pairs.items()
        )
        or set(normalized_evidence_hashes) != expected_evidence_horizons
        or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in normalized_evidence_hashes.values()
        )
        or any(
            len(str(row.get(field) or "")) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(row.get(field) or "")
            )
            for field in hash_fields
        )
    ):
        return {}

    lanes: dict[str, dict[str, Any]] = {}
    for horizon in _PRODUCT_HORIZONS[1:]:
        lanes[horizon] = {
            "horizon": horizon,
            "status": "ok",
            "stage": "cash_only",
            "strategy_version_id": None,
            "health_status": "not_applicable",
            "runner": "cash_only_no_orders",
            "message": (
                "horizon is source-revalidated unavailable and its sleeve remains "
                "in cash; the historical replay is not sealed final OOS"
            ),
            "candidates": [],
            "cash_only": True,
            "sleeve_action": "remain_in_cash",
            "new_entries_allowed": False,
            "recommendation_eligible": False,
            "rehabilitation_evidence": {
                "receipt_sha256": receipt_sha256,
                "target_short_strategy_version_id": selected_version_id,
                "source_strategy_version_id": row["source_strategy_version_id"],
                "source_lockbox_contract_version": row[
                    "source_lockbox_contract_version"
                ],
                "source_lockbox_batch_sha256": row[
                    "source_lockbox_batch_sha256"
                ],
                "source_history_selection_sha256": row[
                    "source_history_selection_sha256"
                ],
                "source_unavailable_horizons_sha256": row[
                    "source_unavailable_horizons_sha256"
                ],
                "source_unavailable_evidence_sha256": normalized_evidence_hashes[
                    horizon
                ],
                "source_cash_only_scope": _SOURCE_CASH_ONLY_SCOPE,
                "authority": _HISTORICAL_DESCRIPTION_ONLY,
                "sealed_final_oos": False,
                "unseen_oos": False,
            },
        }
    return lanes


def _validated_terminal_cash_only_horizon_lanes(
    connection: Any,
    *,
    data_root: Path,
) -> dict[str, dict[str, Any]]:
    """Project the exact terminally rejected short baseline as cash only.

    This receipt is deliberately weaker than a successful formal backtest: it
    proves that the one pinned public baseline exhausted its immutable attempt
    and failed every required robustness scenario.  It can therefore close the
    product sleeve with ``NO_ACTION`` but can never approve a StrategyVersion,
    create a paper account, or emit a recommendation.

    Artifact payloads are fully hashed when the receipt is first registered.
    Readiness uses the bounded verifier so the health endpoint does not re-read
    hundreds of megabytes on every poll; all canonical receipt, database,
    source-lockbox, and immutable identity checks still run here.
    """

    try:
        receipt = require_terminal_cash_only_receipt(
            connection,
            data_root=data_root,
            verify_artifact_hashes=False,
        )
    except Exception:  # noqa: BLE001 - readiness must fail closed
        return {}
    if not isinstance(receipt, Mapping):
        return {}
    gate = receipt.get("robustness_gate")
    receipt_sha256 = str(receipt.get("receipt_sha256") or "")
    if (
        receipt.get("contract_version") != TERMINAL_CASH_ONLY_CONTRACT_VERSION
        or receipt.get("strategy_version_id")
        != TERMINAL_CASH_ONLY_VERSION_ID
        or receipt.get("backtest_id") != TERMINAL_CASH_ONLY_BACKTEST_ID
        or receipt.get("job_id") != TERMINAL_CASH_ONLY_JOB_ID
        or receipt.get("horizon_profile") != "short_1_5d"
        or receipt.get("authority") != _SOURCE_CASH_ONLY_SCOPE
        or receipt.get("cash_only_scope") != _SOURCE_CASH_ONLY_SCOPE
        or receipt.get("formal_result_complete") is not False
        or receipt.get("approval_eligible") is not False
        or receipt.get("rerun_allowed") is not False
        or not isinstance(gate, Mapping)
        or gate.get("passed") != 0
        or gate.get("total") != 4
        or gate.get("min_pass_rate") != 1.0
        or gate.get("passed_gate") is not False
        or len(receipt_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in receipt_sha256
        )
    ):
        return {}
    return {
        "short_1_5d": {
            "horizon": "short_1_5d",
            "status": "ok",
            "stage": "cash_only",
            "strategy_version_id": None,
            "health_status": "not_applicable",
            "runner": "cash_only_no_orders",
            "message": (
                "the pinned short baseline was terminally rejected; its sleeve "
                "remains in cash while governed research seeks a replacement"
            ),
            "candidates": [],
            "cash_only": True,
            "sleeve_action": "remain_in_cash",
            "new_entries_allowed": False,
            "recommendation_eligible": False,
            "terminal_failure_evidence": {
                "receipt_sha256": receipt_sha256,
                "strategy_version_id": TERMINAL_CASH_ONLY_VERSION_ID,
                "backtest_id": TERMINAL_CASH_ONLY_BACKTEST_ID,
                "job_id": TERMINAL_CASH_ONLY_JOB_ID,
                "authority": _SOURCE_CASH_ONLY_SCOPE,
                "cash_only_scope": _SOURCE_CASH_ONLY_SCOPE,
                "formal_result_complete": False,
                "approval_eligible": False,
                "rerun_allowed": False,
                "robustness_gate": dict(gate),
            },
        }
    }


def _project_three_horizon_production(
    candidates: Mapping[str, list[dict[str, Any]]],
    cash_only_lanes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge real production lanes with sealed, intentionally empty sleeves."""

    lanes = {
        horizon: _assess_horizon_candidates(horizon, candidates.get(horizon, []))
        for horizon in _PRODUCT_HORIZONS
    }
    applied_cash_only_horizons: list[str] = []
    for horizon in _PRODUCT_HORIZONS:
        cash_lane = cash_only_lanes.get(horizon)
        if cash_lane is not None and lanes[horizon]["status"] != "ok":
            lanes[horizon] = dict(cash_lane)
            applied_cash_only_horizons.append(horizon)
    missing_or_blocked = [
        horizon for horizon, lane in lanes.items() if lane["status"] != "ok"
    ]
    return {
        "status": "ok" if not missing_or_blocked else "blocked",
        "message": (
            "all product horizons have an operable or governed cash-only lane"
            if not missing_or_blocked
            else "one or more product horizons have no operable or governed cash-only lane"
        ),
        "required_horizons": list(_PRODUCT_HORIZONS),
        "blocked_horizons": missing_or_blocked,
        "cash_only_horizons": applied_cash_only_horizons,
        "horizons": lanes,
    }


def _is_governed_incremental_sync(row: Any) -> bool:
    payload = row.payload_json
    if not isinstance(payload, dict):
        return False
    lookback_days = payload.get("lookback_days")
    return (
        row.timezone == "Asia/Shanghai"
        and bool(row.trading_days_only)
        and payload.get("profile") == "full"
        and payload.get("snapshot_start") == "2008-01-01"
        and payload.get("build_qlib") is True
        and isinstance(lookback_days, int)
        and not isinstance(lookback_days, bool)
        and 1 <= lookback_days <= 30
    )


def _is_governed_full_data_pipeline(row: Any) -> bool:
    payload = row.payload_json
    if not isinstance(payload, dict):
        return False
    bundles = payload.get("bundles")
    lookback_days = payload.get("lookback_days", 7)
    return (
        row.timezone == "Asia/Shanghai"
        and bool(row.trading_days_only)
        and payload.get("profile") == "full"
        and payload.get("snapshot_start") == "2008-01-01"
        and isinstance(lookback_days, int)
        and not isinstance(lookback_days, bool)
        and 1 <= lookback_days <= 90
        and isinstance(bundles, list)
        and len(bundles) == len(_GOVERNED_DATA_PIPELINE_BUNDLES)
        and all(isinstance(bundle, str) for bundle in bundles)
        and set(bundles) == _GOVERNED_DATA_PIPELINE_BUNDLES
    )


def _is_governed_suite_data_pipeline(row: Any) -> bool:
    return _is_governed_full_data_pipeline(row) and row.run_time >= time(15, 10)


def _is_governed_information_pipeline(row: Any) -> bool:
    try:
        payload = normalize_information_schedule_payload(row.payload_json)
    except (TypeError, ValueError):
        return False
    return (
        row.timezone == "Asia/Shanghai"
        and not bool(row.trading_days_only)
        and payload["lookback_days"] == 7
        and payload["regulatory_only"] is True
        and payload["download_limit"] == 0
        and payload["enable_nlp"] is True
        and payload["announcement_categories"] == ["regulatory_letter"]
        and payload["announcement_nlp_limit"] == 500
        and payload["include_corpus_nlp"] is True
        and set(payload["corpus_datasets"])
        == _GOVERNED_INFORMATION_CORPUS_DATASETS
        and payload["corpus_nlp_limit"] == 500
        and payload["batch_size"] == 50
        and payload["major_news_per_day"] == 40
        and payload["irm_per_instrument_day"] == 2
        and payload["include_event_labels"] is True
        and payload["include_factor_evaluation"] is False
        and payload["factor_evaluation"] is None
        and payload["snapshot_name"] == ""
        and payload["horizons"] == [1, 3, 5, 20]
        and payload["benchmark_code"] == "000300.SH"
    )


def _is_governed_information_factor_refresh(
    row: Any,
    *,
    data_root: Path,
    reproducible_dataset_names: set[str],
) -> bool:
    try:
        payload = normalize_information_factor_refresh_payload(row.payload_json)
    except (TypeError, ValueError):
        return False
    evaluation = payload["factor_evaluation"] or {}
    try:
        resolve_information_evaluation_dataset(data_root, evaluation)
    except (OSError, TypeError, ValueError):
        return False
    return (
        row.timezone == "Asia/Shanghai"
        and not bool(row.trading_days_only)
        and set(payload["sources"]) == STRUCTURED_INFORMATION_SOURCES
        and payload["weekday"] == 4
        and evaluation.get("dataset") in reproducible_dataset_names
        and evaluation.get("universe") == "cn_all"
        and evaluation.get("benchmark") == "SH000300"
    )


def _is_governed_ashare_5m_sync(row: Any) -> bool:
    payload = row.payload_json
    if not isinstance(payload, dict):
        return False
    try:
        date.fromisoformat(str(payload.get("history_start") or ""))
    except ValueError:
        return False
    lookback_days = payload.get("lookback_days")
    return (
        row.timezone == "Asia/Shanghai"
        and row.run_time >= time(15, 10)
        and bool(row.trading_days_only)
        and payload.get("daily_dataset") in (None, "")
        and isinstance(lookback_days, int)
        and not isinstance(lookback_days, bool)
        and 1 <= lookback_days <= 30
        and set(payload) <= {"history_start", "daily_dataset", "lookback_days"}
    )


def _is_governed_auxiliary_data_pipeline(row: Any) -> bool:
    payload = row.payload_json
    if not isinstance(payload, dict):
        return False
    try:
        date.fromisoformat(str(payload.get("history_start") or ""))
    except ValueError:
        return False
    return (
        row.timezone == "Asia/Shanghai"
        and not bool(row.trading_days_only)
        and isinstance(payload.get("max_stocks"), int)
        and not isinstance(payload.get("max_stocks"), bool)
        and 1 <= payload["max_stocks"] <= 500
        and isinstance(payload.get("max_options"), int)
        and not isinstance(payload.get("max_options"), bool)
        and 1 <= payload["max_options"] <= 500
        and isinstance(payload.get("strategy_minute_symbols"), list)
        and bool(payload["strategy_minute_symbols"])
    )


def _governed_schedule_suite_state(
    rows: list[Any],
    *,
    data_root: Path,
    reproducible_dataset_names: set[str],
) -> tuple[bool, dict[str, list[str]]]:
    by_kind: dict[str, list[Any]] = {}
    for row in rows:
        by_kind.setdefault(str(row.kind), []).append(row)
    governed: dict[str, list[str]] = {
        "data_pipeline": [],
        "information_pipeline": [],
        "information_factor_refresh": [],
        "ashare_5m_sync": [],
        "auxiliary_data_pipeline": [],
    }
    for row in rows:
        accepted = False
        if row.kind == "data_pipeline":
            accepted = _is_governed_suite_data_pipeline(row)
        elif row.kind == "information_pipeline":
            accepted = _is_governed_information_pipeline(row)
        elif row.kind == "information_factor_refresh":
            accepted = _is_governed_information_factor_refresh(
                row,
                data_root=data_root,
                reproducible_dataset_names=reproducible_dataset_names,
            )
        elif row.kind == "ashare_5m_sync":
            accepted = _is_governed_ashare_5m_sync(row)
        elif row.kind == "auxiliary_data_pipeline":
            accepted = _is_governed_auxiliary_data_pipeline(row)
        if accepted:
            governed[row.kind].append(str(row.id))
    ready = (
        len(rows) == len(_GOVERNED_DATA_SCHEDULE_SUITE_KINDS)
        and set(by_kind) == _GOVERNED_DATA_SCHEDULE_SUITE_KINDS
        and all(len(by_kind[kind]) == 1 for kind in _GOVERNED_DATA_SCHEDULE_SUITE_KINDS)
        and all(len(governed[kind]) == 1 for kind in _GOVERNED_DATA_SCHEDULE_SUITE_KINDS)
    )
    return ready, governed


def _now() -> datetime:
    return datetime.now(UTC)


def _check(
    check_id: str,
    title: str,
    passed: bool,
    evidence: str,
    remediation: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "title": title,
        "status": "pass" if passed else "block",
        "evidence": evidence,
        "remediation": None if passed else remediation,
        "details": details or {},
    }


def _profile(profile_id: str, title: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    blockers = [item for item in checks if item["status"] == "block"]
    return {
        "id": profile_id,
        "title": title,
        "status": "ready" if not blockers else "blocked",
        "passed": len(checks) - len(blockers),
        "total": len(checks),
        "blocker_count": len(blockers),
        "checks": checks,
    }


class DeploymentReadinessStore:
    """Evidence-backed go/no-go assessment for each supported deployment boundary."""

    def __init__(self, settings: Settings, project_root: Path) -> None:
        self.settings = settings
        self.project_root = project_root.resolve()
        self.engine = open_database(settings.database_url)
        self.data_tasks = DataTaskStore(settings.database_url)
        self.health = OperationalHealthStore(settings)
        self.runtime_secrets = RuntimeSecretStore(
            settings.database_url, settings.platform_secret_key
        )

    def assess(self, now: datetime | None = None) -> dict[str, Any]:
        current = now or _now()
        # DataTaskStore.list() reconciles the operational projection and groups
        # the durable work-unit ledger.  Build it once per assessment so the
        # remaining profiles cannot repeat that expensive work.
        tasks = {
            str(item["task_key"]): item
            for item in self.data_tasks.list()
        }
        research_checks = self._research_checks(tasks)
        recommendation_checks = [*research_checks, *self._recommendation_checks()]
        allocation_checks = [*recommendation_checks, *self._allocation_checks(current)]
        profiles = [
            _profile("research", "研究与回测", research_checks),
            _profile("recommendation_tracking", "推荐组合与假设跟踪", recommendation_checks),
            _profile("strategy_allocation", "多策略推荐组合", allocation_checks),
        ]
        highest_ready = next(
            (item["id"] for item in reversed(profiles) if item["status"] == "ready"),
            None,
        )
        return {
            "generated_at": current.isoformat(timespec="seconds"),
            "policy_version": "2026-08-24.1",
            "highest_ready_profile": highest_ready,
            "live_trading_supported": False,
            "profiles": profiles,
        }

    def business_loop_readiness(self, now: datetime | None = None) -> dict[str, Any]:
        """Return the bounded checks that gate the user-facing daily loop.

        This is intentionally narrower than :meth:`assess`: a paper-validating
        horizon is operational even though it has not accumulated enough time
        to publish verified advice.  Conversely, fresh processes alone are not
        enough when daily data or one of the three product lanes is missing.
        """

        current = now or _now()
        try:
            daily_data = _daily_qlib_business_check(
                self.settings.data_root,
                now=current,
            )
        except Exception as exc:  # noqa: BLE001 - readiness must fail closed
            daily_data = {
                "status": "unavailable",
                "message": "daily Qlib readiness could not be evaluated",
                "reason": str(exc)[:500],
            }
        try:
            expected_trade_date = date.fromisoformat(
                str(daily_data["expected_end_date"])
            )
            horizons = self._horizon_production_check(
                expected_trade_date=expected_trade_date,
                now=current,
            )
        except Exception as exc:  # noqa: BLE001 - readiness must fail closed
            horizons = {
                "status": "unavailable",
                "message": "three-horizon production readiness could not be evaluated",
                "reason": str(exc)[:500],
                "required_horizons": list(_PRODUCT_HORIZONS),
                "horizons": {},
            }
        checks = {
            "daily_qlib_data": daily_data,
            "three_horizon_production": horizons,
        }
        blockers = [
            {
                "check": name,
                "status": str(check.get("status") or "unavailable"),
                "message": str(check.get("message") or name),
            }
            for name, check in checks.items()
            if check.get("status") != "ok"
        ]
        return {
            "status": "ok" if not blockers else "blocked",
            "checks": checks,
            "blockers": blockers,
        }

    def _horizon_production_check(
        self,
        *,
        expected_trade_date: date,
        now: datetime,
    ) -> dict[str, Any]:
        def sealed_sha256(value: Any) -> bool:
            text_value = str(value or "")
            return len(text_value) == 64 and all(
                character in "0123456789abcdef" for character in text_value
            )

        candidates: dict[str, list[dict[str, Any]]] = {
            horizon: [] for horizon in _PRODUCT_HORIZONS
        }
        terminal_cash_only_lanes: dict[str, dict[str, Any]] = {}
        with self.engine.connect() as connection:
            all_unavailable_cash_only_rows = connection.execute(
                select(audit_events)
                .where(
                    audit_events.c.action == ALL_UNAVAILABLE_CASH_ONLY_ACTION
                )
                .order_by(audit_events.c.created_at.desc(), audit_events.c.id.desc())
            ).all()
            lockbox_rows = connection.execute(
                select(
                    strategy_versions.c.id,
                    strategy_versions.c.status,
                    strategy_versions.c.horizon_profile,
                    strategy_versions.c.config_json,
                    strategy_versions.c.created_by,
                    strategy_versions.c.created_at,
                )
                .where(
                    strategy_versions.c.is_legacy.is_(False),
                    strategy_versions.c.created_by
                    == _TRANSPARENT_BASELINE_BOOTSTRAP_ACTOR,
                    strategy_versions.c.config_json["recipe_id"]
                    .as_string()
                    .in_(TRANSPARENT_RESEARCH_BASELINE_IDS),
                )
                .order_by(
                    strategy_versions.c.created_at.desc(),
                    strategy_versions.c.id.desc(),
                )
            ).all()
            rehabilitation_rows = connection.execute(
                select(
                    strategy_forward_only_rehabilitations.c.receipt_sha256,
                    strategy_forward_only_rehabilitations.c.source_strategy_version_id,
                    strategy_forward_only_rehabilitations.c.source_lockbox_contract_version,
                    strategy_forward_only_rehabilitations.c.source_lockbox_batch_sha256,
                    strategy_forward_only_rehabilitations.c.source_lockbox_member_sha256,
                    strategy_forward_only_rehabilitations.c.source_history_selection_sha256,
                    strategy_forward_only_rehabilitations.c.source_unavailable_horizons_sha256,
                    strategy_forward_only_rehabilitations.c.source_unavailable_evidence_sha256s_json,
                    strategy_forward_only_rehabilitations.c.source_cash_only_scope,
                    strategy_forward_only_rehabilitations.c.strategy_version_id,
                    strategy_forward_only_rehabilitations.c.contract_version,
                    strategy_forward_only_rehabilitations.c.evidence_mode,
                    strategy_forward_only_rehabilitations.c.authority,
                    strategy_forward_only_rehabilitations.c.recipe_id,
                    strategy_forward_only_rehabilitations.c.horizon_profile,
                    strategy_forward_only_rehabilitations.c.qualification_json,
                    strategy_versions.c.status.label("target_status"),
                    strategy_versions.c.promotion_stage.label(
                        "target_promotion_stage"
                    ),
                    strategy_versions.c.horizon_profile.label(
                        "target_horizon_profile"
                    ),
                    strategy_versions.c.evidence_mode.label("target_evidence_mode"),
                    strategy_versions.c.config_json.label("target_config_json"),
                    strategy_forward_only_rehabilitations.c.created_at,
                )
                .select_from(
                    strategy_forward_only_rehabilitations.join(
                        strategy_versions,
                        strategy_versions.c.id
                        == strategy_forward_only_rehabilitations.c.strategy_version_id,
                    )
                )
                .order_by(strategy_forward_only_rehabilitations.c.created_at.desc())
            ).mappings().all()
            versions = connection.execute(
                select(
                    strategy_versions.c.id,
                    strategy_versions.c.horizon_profile,
                    strategy_versions.c.promotion_stage,
                    strategy_versions.c.signal_frequency,
                    strategy_versions.c.execution_frequency,
                    strategy_versions.c.execution_contract_hash,
                    strategy_versions.c.horizon_contract_sha256,
                    strategy_versions.c.strategy_rules_sha256,
                    strategy_versions.c.approved_at,
                )
                .where(
                    strategy_versions.c.status == "approved",
                    strategy_versions.c.is_legacy.is_(False),
                    strategy_versions.c.horizon_profile.in_(_PRODUCT_HORIZONS),
                    strategy_versions.c.promotion_stage.in_(_OPERABLE_PROMOTION_STAGES),
                )
                .order_by(strategy_versions.c.approved_at.desc())
            ).all()
            for version in versions:
                version_id = str(version.id)
                latest_health = connection.execute(
                    select(strategy_health_snapshots)
                    .where(
                        strategy_health_snapshots.c.strategy_version_id == version_id
                    )
                    .order_by(
                        strategy_health_snapshots.c.as_of.desc(),
                        strategy_health_snapshots.c.recorded_at.desc(),
                    )
                    .limit(1)
                ).first()
                stage = connection.execute(
                    select(
                        strategy_promotion_stages.c.status,
                        strategy_promotion_stages.c.simulation_portfolio_id,
                    )
                    .where(
                        strategy_promotion_stages.c.strategy_version_id == version_id
                    )
                    .order_by(strategy_promotion_stages.c.stage_index.desc())
                    .limit(1)
                ).first()
                simulation_status = None
                current_daily_dataset_identity = None
                current_daily_dataset_lineage = None
                current_batch_id = None
                current_source_snapshot_id = None
                current_feature_date = None
                if stage is not None and stage.simulation_portfolio_id is not None:
                    simulation = connection.execute(
                        select(
                            simulation_portfolios.c.status,
                            simulation_portfolios.c.daily_dataset_identity_sha256,
                            simulation_portfolios.c.daily_dataset_lineage_id,
                        ).where(
                            simulation_portfolios.c.id == stage.simulation_portfolio_id
                        )
                    ).first()
                    if simulation is not None:
                        simulation_status = simulation.status
                        batch_rows = connection.execute(
                            select(
                                simulation_batches.c.id,
                                simulation_batches.c.signal_date,
                                simulation_batches.c.trade_date,
                                simulation_batches.c.source_snapshot_id,
                                simulation_batches.c.daily_dataset_identity_sha256,
                                simulation_batches.c.daily_dataset_lineage_id,
                            )
                            .where(
                                simulation_batches.c.portfolio_id
                                == stage.simulation_portfolio_id,
                                simulation_batches.c.status == "succeeded",
                                simulation_batches.c.trade_date
                                == expected_trade_date,
                            )
                            .order_by(simulation_batches.c.finished_at.desc())
                            .limit(2)
                        ).all()
                        if len(batch_rows) == 1:
                            batch = batch_rows[0]
                            identity = str(batch.daily_dataset_identity_sha256)
                            lineage = str(batch.daily_dataset_lineage_id)
                            source_snapshot_id = str(batch.source_snapshot_id or "")
                            if (
                                len(identity) == 64
                                and len(lineage) == 64
                                and lineage
                                == str(simulation.daily_dataset_lineage_id)
                                and source_snapshot_id == identity
                                and batch.trade_date == expected_trade_date
                                and batch.signal_date <= batch.trade_date
                            ):
                                current_daily_dataset_identity = identity
                                current_daily_dataset_lineage = lineage
                                current_batch_id = str(batch.id)
                                current_source_snapshot_id = source_snapshot_id
                                current_feature_date = batch.signal_date
                active_recommendation_portfolios = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(recommendation_portfolios)
                        .where(
                            recommendation_portfolios.c.strategy_version_id == version_id,
                            recommendation_portfolios.c.status == "active",
                        )
                    )
                    or 0
                )
                horizon = str(version.horizon_profile)
                health_check = _strategy_health_evidence_check(
                    latest_health,
                    version_id=version_id,
                    horizon_profile=horizon,
                    horizon_contract_sha256=str(version.horizon_contract_sha256),
                    current_dataset_identity_sha256=current_daily_dataset_identity,
                    expected_trade_date=expected_trade_date,
                    now=now,
                    max_age_seconds=max(
                        600,
                        int(self.settings.strategy_health_snapshot_seconds) * 2,
                    ),
                    current_dataset_lineage_id=current_daily_dataset_lineage,
                    current_batch_id=current_batch_id,
                    current_source_snapshot_id=current_source_snapshot_id,
                    expected_feature_date=current_feature_date,
                )
                candidates[horizon].append(
                    {
                        "strategy_version_id": version_id,
                        "promotion_stage": str(version.promotion_stage),
                        "signal_frequency": str(version.signal_frequency),
                        "execution_frequency": str(version.execution_frequency),
                        "contract_ready": all(
                            (
                                sealed_sha256(version.execution_contract_hash),
                                sealed_sha256(version.horizon_contract_sha256),
                                sealed_sha256(version.strategy_rules_sha256),
                            )
                        ),
                        "health_status": (
                            str(latest_health.health_status)
                            if latest_health is not None
                            else None
                        ),
                        "health_evidence_ready": health_check["ready"],
                        "health_evidence_reasons": health_check["reasons"],
                        "health_evidence": health_check,
                        "paper_stage_status": (
                            str(stage.status) if stage is not None else None
                        ),
                        "simulation_status": (
                            str(simulation_status) if simulation_status is not None else None
                        ),
                        "active_recommendation_portfolios": (
                            active_recommendation_portfolios
                        ),
                    }
                )
            terminal_cash_only_lanes = (
                _validated_terminal_cash_only_horizon_lanes(
                    connection,
                    data_root=self.settings.data_root,
                )
            )
        short_lane = _assess_horizon_candidates(
            "short_1_5d", candidates["short_1_5d"]
        )
        rehabilitation_cash_only_lanes = (
            _validated_rehabilitation_cash_only_horizon_lanes(
                rehabilitation_rows,
                short_lane=short_lane,
            )
        )
        all_unavailable_cash_only_lanes = (
            _validated_all_unavailable_cash_only_horizon_lanes(
                all_unavailable_cash_only_rows,
                lockbox_rows=lockbox_rows,
            )
        )
        # A genuine current-recipe sealed lockbox remains authoritative when it
        # exists. The rehabilitation projection is an independent source
        # reference and never impersonates that sealed path.
        cash_only_lanes = {
            **all_unavailable_cash_only_lanes,
            **rehabilitation_cash_only_lanes,
            **terminal_cash_only_lanes,
            **_validated_cash_only_horizon_lanes(lockbox_rows),
        }
        return _project_three_horizon_production(candidates, cash_only_lanes)

    def _recommendation_checks(self) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            portfolio_count = int(
                connection.scalar(select(func.count()).select_from(recommendation_portfolios)) or 0
            )
            snapshot_count = int(
                connection.scalar(select(func.count()).select_from(recommendation_snapshots)) or 0
            )
            simulation_count = int(
                connection.scalar(select(func.count()).select_from(simulation_portfolios)) or 0
            )
            certified_nav_count = int(
                connection.scalar(
                    select(func.count()).select_from(simulation_nav).where(
                        simulation_nav.c.performance_certified.is_(True)
                    )
                )
                or 0
            )
            degraded_nav_count = int(
                connection.scalar(
                    select(func.count()).select_from(simulation_nav).where(
                        simulation_nav.c.status == "degraded"
                    )
                )
                or 0
            )
            nav_history = connection.execute(
                select(
                    simulation_nav.c.portfolio_id,
                    simulation_nav.c.status,
                    simulation_nav.c.performance_certified,
                )
            ).all()
            unsupported_active = int(
                connection.scalar(
                    select(func.count())
                    .select_from(schedules)
                    .where(
                        ~schedules.c.kind.in_(ACTIVE_SCHEDULE_KINDS),
                        schedules.c.status == "active",
                    )
                )
                or 0
            )
        nav_by_portfolio: dict[str, list[Any]] = {}
        for row in nav_history:
            nav_by_portfolio.setdefault(str(row.portfolio_id), []).append(row)
        replay_ready = any(
            len(rows) >= 60
            and all(
                bool(row.performance_certified) and row.status == "healthy"
                for row in rows
            )
            for rows in nav_by_portfolio.values()
        )
        maximum_replay_days = max(
            (len(rows) for rows in nav_by_portfolio.values()), default=0
        )
        return [
            _check(
                "simulation_accounts",
                "Recommendation targets are bound to simulation accounts",
                simulation_count > 0,
                f"transactional simulation accounts: {simulation_count}",
                "Create and activate a 5-minute simulation account for a recommendation target",
            ),
            _check(
                "certified_simulation_nav",
                "Simulation NAV is certifiable",
                certified_nav_count > 0 and degraded_nav_count == 0,
                (
                    f"certified NAV rows: {certified_nav_count}; "
                    f"degraded NAV rows: {degraded_nav_count}"
                ),
                "Run simulation booking and resolve stale or missing valuations",
            ),
            _check(
                "simulation_60_day_replay",
                "A simulation account has at least 60 certified trading days",
                replay_ready,
                f"maximum simulation history: {maximum_replay_days} days",
                "Replay at least 60 trading days with no degraded or uncertified NAV rows",
            ),
            _check(
                "recommendation_schema",
                "推荐领域模型已启用",
                True,
                f"推荐组合 {portfolio_count} 个，快照 {snapshot_count} 个",
                "运行数据库迁移后再启用推荐跟踪",
            ),
            _check(
                "unsupported_schedules_retired",
                "非生产调度已退休",
                unsupported_active == 0,
                f"仍活动的非生产调度 {unsupported_active} 个",
                "将不属于当前研究与推荐管线的调度设为 retired",
            ),
        ]

    def _research_checks(
        self,
        tasks: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        required_pipeline = (
            "cn_ashare_daily_full",
            "cn_data_verify",
            "cn_snapshot_build",
            "cn_qlib_build",
            "cn_qlib_baseline",
        )
        task_states = {
            key: str(tasks.get(key, {}).get("status", "missing")) for key in required_pipeline
        }
        pipeline_ready = all(status == "succeeded" for status in task_states.values())
        # Readiness is an operational display, not an admission or capital
        # boundary.  Use the persisted bounded projection here; every formal
        # research/backtest/simulation action still performs strict per-dataset
        # provenance and sealed-output verification before it may proceed.
        reproducible_datasets = [
            item
            for item in list_qlib_datasets_for_display(self.settings.data_root)
            if item["ready"]
            and item.get("reproducible")
            and item.get("lineage_verified")
        ]
        datasets = [
            item
            for item in reproducible_datasets
            if int(item["trading_days"]) >= RESEARCH_MINIMUM_TRADING_DAYS
        ]
        maximum_trading_days = max(
            (int(item["trading_days"]) for item in reproducible_datasets),
            default=0,
        )
        latest_health = self.health.latest()
        health_ready = bool(latest_health and latest_health["status"] == "ok")
        rdagent_status = (
            str(
                latest_health.get("components", {})
                .get("rdagent_runtime", {})
                .get("status", "missing")
            )
            if latest_health
            else "missing"
        )
        secret_health = self.runtime_secrets.health()
        tushare_record = self.runtime_secrets.describe("tushare")
        tushare_evidence = "未保存经验证的 Tushare 凭据"
        tushare_verified = False
        if tushare_record:
            try:
                credentials = self.runtime_secrets.get("tushare") or {}
                metadata = tushare_record.get("metadata_json") or {}
                tushare_verified = bool(
                    credentials.get("api_url")
                    and credentials.get("token")
                    and metadata.get("verified_at")
                )
                tushare_evidence = (
                    "数据库凭据可解密且具有验证时间"
                    if tushare_verified
                    else "数据库凭据缺少地址、令牌或验证时间"
                )
            except ValueError as exc:
                tushare_evidence = f"数据库凭据无法解密：{exc}"
        elif self.settings.api_url and self.settings.token and pipeline_ready:
            tushare_verified = True
            tushare_evidence = "部署凭据已被完整初始化数据管线验证"
        with self.engine.connect() as connection:
            database_head = connection.scalar(
                text("SELECT version_num FROM quantlab.alembic_version")
            )
            active_admins = int(
                connection.scalar(
                    select(func.count())
                    .select_from(users)
                    .where(users.c.role == "admin", users.c.active.is_(True))
                )
                or 0
            )
            active_data_schedules = connection.execute(
                select(
                    schedules.c.id,
                    schedules.c.kind,
                    schedules.c.timezone,
                    schedules.c.run_time,
                    schedules.c.trading_days_only,
                    schedules.c.payload_json,
                ).where(
                    schedules.c.kind.in_(
                        ("incremental_sync", *_GOVERNED_DATA_SCHEDULE_SUITE_KINDS)
                    ),
                    schedules.c.status == "active",
                )
            ).all()
            critical_alerts = int(
                connection.scalar(
                    select(func.count())
                    .select_from(alerts)
                    .where(alerts.c.severity == "critical", alerts.c.status == "open")
                )
                or 0
            )
        governed_incremental_ids = [
            str(row.id)
            for row in active_data_schedules
            if row.kind == "incremental_sync" and _is_governed_incremental_sync(row)
        ]
        rejected_incremental_ids = [
            str(row.id)
            for row in active_data_schedules
            if row.kind == "incremental_sync" and not _is_governed_incremental_sync(row)
        ]
        governed_pipeline_ids = [
            str(row.id)
            for row in active_data_schedules
            if row.kind == "data_pipeline" and _is_governed_full_data_pipeline(row)
        ]
        rejected_pipeline_ids = [
            str(row.id)
            for row in active_data_schedules
            if row.kind == "data_pipeline" and not _is_governed_full_data_pipeline(row)
        ]
        legacy_schedule_ready = (
            len(active_data_schedules) == 1
            and len(governed_incremental_ids) + len(governed_pipeline_ids) == 1
        )
        suite_ready, governed_suite_ids = _governed_schedule_suite_state(
            active_data_schedules,
            data_root=self.settings.data_root,
            reproducible_dataset_names={str(item["name"]) for item in datasets},
        )
        data_schedule_ready = legacy_schedule_ready or suite_ready
        schedule_mode = (
            "governed_suite_v1"
            if suite_ready
            else "legacy_single"
            if legacy_schedule_ready
            else "invalid"
        )
        governed_suite_id_set = {
            schedule_id
            for ids in governed_suite_ids.values()
            for schedule_id in ids
        }
        rejected_suite_ids = [
            str(row.id)
            for row in active_data_schedules
            if row.kind in _GOVERNED_DATA_SCHEDULE_SUITE_KINDS
            and str(row.id) not in governed_suite_id_set
        ]
        code_head = self._code_schema_head()
        return [
            _check(
                "schema_current",
                "数据库结构为当前版本",
                bool(code_head and database_head == code_head),
                f"数据库 {database_head or 'missing'}，代码 {code_head or 'missing'}",
                "执行数据库迁移到当前 Alembic head",
            ),
            _check(
                "authentication_enabled",
                "认证与管理员已启用",
                self.settings.auth_mode == "required" and active_admins > 0,
                f"认证模式 {self.settings.auth_mode}，活动管理员 {active_admins} 个",
                "启用 required 认证并创建活动管理员",
            ),
            _check(
                "runtime_secret_storage",
                "运行时密钥存储可用",
                secret_health["status"] == "ok",
                str(secret_health.get("message") or secret_health["status"]),
                "修复平台密钥并确认已有密文可以解密",
            ),
            _check(
                "tushare_verified",
                "Tushare 凭据已验证",
                tushare_verified,
                tushare_evidence,
                "通过设置接口保存并验证 Tushare 凭据",
            ),
            _check(
                "initialization_pipeline",
                "初始化数据管线完成",
                pipeline_ready,
                f"任务状态 {task_states}",
                "完成下载、校验、快照、Qlib 构建和基线任务",
            ),
            _check(
                "reproducible_qlib_dataset",
                "存在可复现 Qlib 数据集",
                bool(datasets),
                (
                    f"满足至少 {RESEARCH_MINIMUM_TRADING_DAYS} 个交易日及血缘要求的"
                    f"数据集 {len(datasets)} 个；当前最长 {maximum_trading_days} 个交易日"
                ),
                (
                    "构建包含至少 "
                    f"{RESEARCH_MINIMUM_TRADING_DAYS} 个交易日的数据集："
                    f"训练 {MINIMUM_PROFILE_TRAINING_DAYS} + 最长验证 "
                    f"{RESEARCH_LONGEST_VALIDATION_TRADING_DAYS}"
                    f" + 隔离 {DEFAULT_RESEARCH_PERIOD_POLICY['embargo_trading_days']}"
                    f" + 最终测试 {DEFAULT_RESEARCH_PERIOD_POLICY['test_trading_days']}"
                ),
                details={
                    "minimum_trading_days": RESEARCH_MINIMUM_TRADING_DAYS,
                    "maximum_available_trading_days": maximum_trading_days,
                    "eligible_dataset_count": len(datasets),
                },
            ),
            _check(
                "operational_health",
                "研究运行健康",
                health_ready,
                str(latest_health["status"] if latest_health else "missing"),
                "恢复数据库、Worker、队列和市场数据健康",
            ),
            _check(
                "rdagent_runtime",
                "RD-Agent 运行时可用",
                rdagent_status == "ok",
                f"RD-Agent 状态 {rdagent_status}",
                "配置并启动 RD-Agent 研究运行时",
            ),
            _check(
                "incremental_schedule",
                "受治理的数据更新调度已启用",
                data_schedule_ready,
                (
                    f"活动数据更新调度 {len(active_data_schedules)} 个；"
                    f"模式 {schedule_mode}；"
                    f"合格 incremental_sync {len(governed_incremental_ids)} 个；"
                    f"不合格 incremental_sync {len(rejected_incremental_ids)} 个；"
                    f"合格 full data_pipeline {len(governed_pipeline_ids)} 个；"
                    f"不合格 data_pipeline {len(rejected_pipeline_ids)} 个；"
                    f"合格五计划组件 {sum(len(ids) for ids in governed_suite_ids.values())} 个"
                ),
                (
                    "保留一个兼容的受治理 legacy 数据计划，或精确启用五计划套件："
                    "18:00 full data_pipeline、23:30 ashare_5m_sync、02:00 bounded "
                    "information_pipeline、周五 12:30 information_factor_refresh、"
                    "04:00 auxiliary_data_pipeline"
                ),
                details={
                    "mode": schedule_mode,
                    "active_schedule_ids": [str(row.id) for row in active_data_schedules],
                    "incremental_sync_ids": governed_incremental_ids,
                    "rejected_incremental_sync_ids": rejected_incremental_ids,
                    "governed_data_pipeline_ids": governed_pipeline_ids,
                    "rejected_data_pipeline_ids": rejected_pipeline_ids,
                    "governed_suite_ids": governed_suite_ids,
                    "rejected_suite_ids": rejected_suite_ids,
                    "required_suite_kinds": sorted(_GOVERNED_DATA_SCHEDULE_SUITE_KINDS),
                    "required_bundles": sorted(_GOVERNED_DATA_PIPELINE_BUNDLES),
                },
            ),
            _check(
                "critical_alerts_clear",
                "严重告警已闭环",
                critical_alerts == 0,
                f"未处理 critical 告警 {critical_alerts} 条",
                "处理所有 critical 告警后重新验收",
            ),
        ]

    def _allocation_checks(self, current: datetime) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            active_rows = connection.execute(
                select(
                    strategy_allocations.c.id,
                    strategy_allocations.c.analysis_json,
                    strategy_allocations.c.max_pairwise_correlation,
                ).where(
                    strategy_allocations.c.status == "active",
                    strategy_allocations.c.is_legacy.is_(False),
                )
            ).all()
            eligible_ids: list[str] = []
            provisioned_ids: list[str] = []
            automated_ids: list[str] = []
            evidence: list[dict[str, Any]] = []
            for row in active_rows:
                analysis = row.analysis_json or {}
                observed = analysis.get("highest_pairwise_correlation")
                member_count = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(strategy_allocation_members)
                        .where(strategy_allocation_members.c.allocation_id == row.id)
                    )
                    or 0
                )
                provisioned_count = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(strategy_allocation_members)
                        .where(
                            strategy_allocation_members.c.allocation_id == row.id,
                            strategy_allocation_members.c.recommendation_portfolio_id.is_not(None),
                        )
                    )
                    or 0
                )
                correlation_passed = (
                    observed is not None
                    and float(observed) <= float(row.max_pairwise_correlation)
                    and member_count >= 2
                )
                if correlation_passed:
                    eligible_ids.append(str(row.id))
                if correlation_passed and provisioned_count == member_count:
                    provisioned_ids.append(str(row.id))
                group_status = connection.scalar(
                    select(allocation_schedule_groups.c.status).where(
                        allocation_schedule_groups.c.allocation_id == row.id
                    )
                )
                scheduled_count = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(schedules)
                        .where(
                            schedules.c.kind == "recommendation_refresh",
                            schedules.c.payload_json["allocation_id"].as_string() == str(row.id),
                            schedules.c.desired_status == "active",
                            schedules.c.status == "active",
                        )
                    )
                    or 0
                )
                automation_ready = (
                    correlation_passed
                    and provisioned_count == member_count
                    and group_status == "active"
                    and scheduled_count == member_count
                )
                if automation_ready:
                    automated_ids.append(str(row.id))
                evidence.append(
                    {
                        "allocation_id": str(row.id),
                        "correlation": observed,
                        "limit": float(row.max_pairwise_correlation),
                        "members": member_count,
                        "provisioned": provisioned_count,
                        "schedule_group": group_status,
                        "scheduled": scheduled_count,
                    }
                )
            nav_days = 0
            latest_nav_date = None
            if provisioned_ids:
                nav_days = int(
                    connection.scalar(
                        select(func.count(distinct(strategy_allocation_nav.c.trade_date))).where(
                            strategy_allocation_nav.c.allocation_id.in_(provisioned_ids)
                        )
                    )
                    or 0
                )
                latest_nav_date = connection.scalar(
                    select(func.max(strategy_allocation_nav.c.trade_date)).where(
                        strategy_allocation_nav.c.allocation_id.in_(provisioned_ids)
                    )
                )
            open_events = 0
            if provisioned_ids:
                open_events = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(strategy_allocation_events)
                        .where(
                            strategy_allocation_events.c.allocation_id.in_(provisioned_ids),
                            strategy_allocation_events.c.severity == "critical",
                            strategy_allocation_events.c.status.in_(["open", "acknowledged"]),
                        )
                    )
                    or 0
                )
        nav_age = (current.date() - latest_nav_date).days if latest_nav_date else None
        continuity_ready = nav_days >= 5 and nav_age is not None and nav_age <= 7
        return [
            _check(
                "low_correlation_allocation",
                "低相关策略组合",
                bool(eligible_ids),
                f"活动组合 {len(active_rows)} 个，相关性门禁通过 {len(eligible_ids)} 个",
                "使用至少两个已审批策略建立低相关组合",
                details={"allocations": evidence},
            ),
            _check(
                "recommendation_children",
                "成员推荐组合已配置",
                bool(provisioned_ids),
                f"完整配置成员推荐组合的策略组合 {len(provisioned_ids)} 个",
                "审批组合并为每个成员创建推荐组合",
            ),
            _check(
                "allocation_automation",
                "成员推荐刷新调度已启用",
                bool(automated_ids),
                f"完整启用推荐刷新调度的策略组合 {len(automated_ids)} 个",
                "为每个成员配置 recommendation_refresh 调度",
                details={"automated_allocation_ids": automated_ids},
            ),
            _check(
                "allocation_continuity",
                "组合级假设净值连续",
                continuity_ready,
                (
                    f"组合净值 {nav_days} 个交易日，最新 {latest_nav_date}，距今 {nav_age} 天"
                    if latest_nav_date
                    else "尚无对齐的组合级假设净值"
                ),
                "至少积累 5 个交易日的组合级假设净值",
            ),
            _check(
                "allocation_risk_events_clear",
                "组合级风险事件已闭环",
                open_events == 0,
                f"未关闭的组合级 critical 风险事件 {open_events} 条",
                "处理组合级回撤事件后重新验收",
            ),
        ]

    def _code_schema_head(self) -> str | None:
        try:
            config = Config(str(self.project_root / "alembic.ini"))
            config.set_main_option("script_location", str(self.project_root / "migrations"))
            heads = ScriptDirectory.from_config(config).get_heads()
        except Exception:  # pragma: no cover - deployment diagnostic must fail closed
            return None
        return heads[0] if len(heads) == 1 else None
