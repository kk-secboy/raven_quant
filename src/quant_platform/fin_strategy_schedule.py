"""Managed calendar schedules for the three governed fin_strategy horizons."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from typing import Any

from quant_data.execution_contract import require_daily_qlib_contract

from .feature_set_registry import transparent_strategy_feature_set
from .research_automation import normalize_research_period_policy
from .schedule_store import ScheduleStore
from .strategy_recipes import TRANSPARENT_RESEARCH_BASELINE_IDS, get_strategy_recipe

MANAGED_FIN_STRATEGY_SCHEDULE_VERSION = "managed-fin-strategy-schedules-v2"
MANAGED_FIN_STRATEGY_DRIFT_TRIGGER_VERSION = "managed-fin-strategy-drift-trigger-v1"
LATEST_REPRODUCIBLE_DAILY_DATASET = "managed:latest-reproducible-daily"
MANAGED_FIN_STRATEGY_ACTOR = "system:fin-strategy-scheduler"
MANAGED_FIN_STRATEGY_RUN_TIME = time(23, 0)
MANAGED_FIN_STRATEGY_MISFIRE_SECONDS = 3 * 24 * 60 * 60

_CADENCES = {
    "short_relative_strength": "weekly_first_session",
    "swing_trend": "monthly_first_session",
    "long_quality_value": "quarterly_and_post_reporting",
}
_SCHEDULE_NAMES = {
    "short_relative_strength": "QuantLab / fin_strategy / short",
    "swing_trend": "QuantLab / fin_strategy / swing",
    "long_quality_value": "QuantLab / fin_strategy / long",
}

_DRIFT_TRIGGER_POLICY = {
    "contract_version": MANAGED_FIN_STRATEGY_DRIFT_TRIGGER_VERSION,
    "source": "strategy_health_snapshots",
    "metric": "feature_drift",
    "threshold_key": "watch_feature_drift",
    "comparison": "greater_than_or_equal",
    "required_hard_gates": ["data_integrity_ok", "ledger_reconciled"],
    "dedupe": "contiguous_breach_episode",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def build_managed_fin_strategy_schedule_specs() -> list[dict[str, Any]]:
    """Build stable schedule payloads without binding a daily snapshot name."""

    specs: list[dict[str, Any]] = []
    for recipe_id in TRANSPARENT_RESEARCH_BASELINE_IDS:
        recipe = get_strategy_recipe(recipe_id)
        feature_set = transparent_strategy_feature_set(recipe_id)
        managed = {
            "contract_version": MANAGED_FIN_STRATEGY_SCHEDULE_VERSION,
            "recipe_id": recipe_id,
            "recipe_version": str(recipe["version"]),
            "recipe_sha256": canonical_sha256(recipe),
            "horizon_profile": str(recipe["horizon"]),
            "feature_set_id": str(feature_set["id"]),
            "feature_set_definition_sha256": str(
                feature_set["definition_sha256"]
            ),
            "dataset_policy": "latest_reproducible_daily",
            "cadence": _CADENCES[recipe_id],
            "trigger_policy": {
                "calendar": "persisted_exchange_trade_calendar",
                "drift_trigger": dict(_DRIFT_TRIGGER_POLICY),
            },
        }
        managed["contract_sha256"] = canonical_sha256(managed)
        specs.append(
            {
                "name": _SCHEDULE_NAMES[recipe_id],
                "kind": "rdagent_research",
                "timezone": "Asia/Shanghai",
                "run_time": MANAGED_FIN_STRATEGY_RUN_TIME,
                "trading_days_only": True,
                "misfire_grace_seconds": MANAGED_FIN_STRATEGY_MISFIRE_SECONDS,
                "payload": {
                    "scenario": "fin_strategy",
                    "objective": str(recipe["rdagent_objective"]),
                    "dataset": LATEST_REPRODUCIBLE_DAILY_DATASET,
                    "dataset_policy": "latest_reproducible_daily",
                    "loop_n": 1,
                    "duration": "30m",
                    "requested_by": MANAGED_FIN_STRATEGY_ACTOR,
                    "asset_ids": [],
                    "feature_set_id": feature_set["id"],
                    "horizon_profile": recipe["horizon"],
                    "period_mode": "rolling",
                    "period_policy": normalize_research_period_policy(
                        horizon_profile=str(recipe["horizon"])
                    ),
                    "managed_fin_strategy": managed,
                },
            }
        )
    return specs


def validate_managed_fin_strategy_payload(value: Any) -> dict[str, Any] | None:
    """Return a normalized managed binding, or ``None`` for operator schedules."""

    if not isinstance(value, Mapping):
        return None
    managed = value.get("managed_fin_strategy")
    if managed is None:
        return None
    if not isinstance(managed, Mapping):
        raise ValueError("managed fin_strategy schedule binding is invalid")
    allowed = {
        "contract_version",
        "recipe_id",
        "recipe_version",
        "recipe_sha256",
        "horizon_profile",
        "feature_set_id",
        "feature_set_definition_sha256",
        "dataset_policy",
        "cadence",
        "trigger_policy",
        "contract_sha256",
    }
    if (
        set(managed) != allowed
        or managed.get("contract_version") != MANAGED_FIN_STRATEGY_SCHEDULE_VERSION
    ):
        raise ValueError("managed fin_strategy schedule contract is invalid")
    candidate = dict(managed)
    supplied_sha256 = _require_sha256(
        candidate.pop("contract_sha256", None), field="managed schedule contract_sha256"
    )
    if canonical_sha256(candidate) != supplied_sha256:
        raise ValueError("managed fin_strategy schedule contract digest changed")
    recipe_id = str(candidate.get("recipe_id") or "")
    if recipe_id not in TRANSPARENT_RESEARCH_BASELINE_IDS:
        raise ValueError("managed fin_strategy schedule recipe is invalid")
    recipe = get_strategy_recipe(recipe_id)
    feature_set = transparent_strategy_feature_set(recipe_id)
    trigger_policy = candidate.get("trigger_policy")
    if (
        candidate.get("recipe_version") != recipe["version"]
        or candidate.get("recipe_sha256") != canonical_sha256(recipe)
        or candidate.get("horizon_profile") != recipe["horizon"]
        or candidate.get("feature_set_id") != feature_set["id"]
        or candidate.get("feature_set_definition_sha256")
        != feature_set["definition_sha256"]
        or candidate.get("dataset_policy") != "latest_reproducible_daily"
        or candidate.get("cadence") != _CADENCES[recipe_id]
        or trigger_policy
        != {
            "calendar": "persisted_exchange_trade_calendar",
            "drift_trigger": _DRIFT_TRIGGER_POLICY,
        }
        or value.get("scenario") != "fin_strategy"
        or value.get("dataset") != LATEST_REPRODUCIBLE_DAILY_DATASET
        or value.get("dataset_policy") != "latest_reproducible_daily"
        or value.get("feature_set_id") != feature_set["id"]
        or value.get("horizon_profile") != recipe["horizon"]
    ):
        raise ValueError("managed fin_strategy schedule differs from this release")
    return {**candidate, "contract_sha256": supplied_sha256}


def select_latest_reproducible_daily_dataset(
    datasets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the newest valid daily publication and preserve its exact identity."""

    candidates: list[dict[str, Any]] = []
    for raw in datasets:
        if (
            not raw.get("ready")
            or not raw.get("reproducible")
            or not raw.get("output_files_verified")
            or str(raw.get("frequency") or "") != "day"
        ):
            continue
        provenance = raw.get("provenance")
        if not isinstance(provenance, Mapping):
            continue
        try:
            require_daily_qlib_contract(dict(provenance))
            identity = _require_sha256(
                provenance.get("dataset_identity_sha256"),
                field="dataset_identity_sha256",
            )
            lineage = _require_sha256(
                raw.get("lineage_id") or provenance.get("dataset_lineage_id"),
                field="dataset_lineage_id",
            )
        except ValueError:
            continue
        if str(provenance.get("dataset_lineage_id") or "") != lineage:
            continue
        item = dict(raw)
        item["provenance"] = dict(provenance)
        item["dataset_identity_sha256"] = identity
        item["dataset_lineage_id"] = lineage
        candidates.append(item)
    if not candidates:
        raise ValueError("no reproducible governed daily Qlib dataset is ready")
    return max(
        candidates,
        key=lambda item: (
            str(item.get("end_date") or ""),
            int(item.get("trading_days") or 0),
            str(item.get("name") or ""),
        ),
    )


def managed_fin_strategy_due_event(
    managed: Mapping[str, Any],
    *,
    scheduled_date: date,
    trading_days: Sequence[date],
) -> dict[str, Any]:
    """Resolve one calendar-only due event from the persisted exchange calendar."""

    ordered = sorted(set(trading_days))
    if not ordered:
        raise ValueError("persisted exchange trading calendar is empty")
    if scheduled_date > ordered[-1]:
        raise ValueError("persisted exchange trading calendar does not cover the schedule date")
    if scheduled_date not in set(ordered):
        return {"due": False, "reason": "exchange_closed", "event": None}
    index = ordered.index(scheduled_date)
    previous = ordered[index - 1] if index else None
    cadence = str(managed.get("cadence") or "")
    event: str | None = None
    if cadence == "weekly_first_session":
        current_week = scheduled_date.isocalendar()[:2]
        previous_week = previous.isocalendar()[:2] if previous is not None else None
        if current_week != previous_week:
            event = f"week:{scheduled_date:%G-W%V}"
    elif cadence == "monthly_first_session":
        if previous is None or (scheduled_date.year, scheduled_date.month) != (
            previous.year,
            previous.month,
        ):
            event = f"month:{scheduled_date:%Y-%m}"
    elif cadence == "quarterly_and_post_reporting":
        if (
            scheduled_date.month in {1, 4, 7, 10}
            and (
                previous is None
                or (scheduled_date.year, scheduled_date.month)
                != (previous.year, previous.month)
            )
        ):
            event = f"quarter:{scheduled_date.year}-Q{(scheduled_date.month - 1) // 3 + 1}"
        if previous is not None:
            deadlines = (
                (date(scheduled_date.year, 4, 30), "annual_q1"),
                (date(scheduled_date.year, 8, 31), "interim"),
                (date(scheduled_date.year, 10, 31), "q3"),
            )
            report_event = next(
                (
                    f"post_report:{scheduled_date.year}:{label}"
                    for deadline, label in deadlines
                    if previous <= deadline < scheduled_date
                ),
                None,
            )
            event = report_event or event
    else:
        raise ValueError("managed fin_strategy cadence is invalid")
    return {
        "due": event is not None,
        "reason": "calendar_due" if event is not None else "not_cadence_boundary",
        "event": event,
    }


def reconcile_managed_fin_strategy_schedules(
    store: ScheduleStore,
    *,
    enabled: bool,
    actor: str = MANAGED_FIN_STRATEGY_ACTOR,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Idempotently reconcile exactly three schedules in the existing store."""

    schedules = []
    for spec in build_managed_fin_strategy_schedule_specs():
        schedules.append(
            store.upsert_managed(
                **spec,
                actor=actor,
                enabled=enabled,
                now=now,
            )
        )
    return schedules
