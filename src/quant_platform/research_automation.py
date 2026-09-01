from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd
from scipy.optimize import linprog

from .cost_model import CN_COST_SCHEDULE_BOOK
from .factor_evaluator import normalize_series
from .factor_library import ECONOMIC_FAMILIES, compile_qlib_expression
from .feature_set_registry import get_feature_set
from .model_research_governance import MODEL_LABEL_HORIZON_TRADING_DAYS
from .rdagent_runtime import validate_duration, validate_duration_limit
from .rdagent_scenarios import (
    get_rdagent_scenario,
    validate_asset_id,
    validate_feature_set_id,
)
from .research_horizon import (
    LEGACY_AMBIGUOUS,
    LONG_1_3Y,
    SHORT_1_5D,
    SWING_1_6M,
    canonical_sha256,
    primary_label_policy_contract,
    research_horizon_contract,
)
from .research_window import (
    build_research_window_contract,
    resolve_required_field_coverage,
)

RESEARCH_PERIOD_KEYS = (
    "train_start",
    "train_end",
    "valid_start",
    "valid_end",
    "test_start",
    "test_end",
)

DEFAULT_RESEARCH_PERIOD_POLICY = {
    "test_trading_days": 252,
    "embargo_trading_days": 20,
}

RESEARCH_EVALUATION_PROFILES = (
    {
        "id": "recent_3y",
        "label": "近期型",
        "role": "primary",
        "validation_trading_days": 756,
    },
    {
        "id": "robust_10y",
        "label": "稳健型",
        "role": "stress",
        "validation_trading_days": 2520,
    },
    {
        "id": "balanced_5y",
        "label": "均衡型",
        "role": "confirmation",
        "validation_trading_days": 1260,
    },
)
MINIMUM_PROFILE_TRAINING_DAYS = 252
MULTI_PROFILE_CONSENSUS_VERSION = "multi-profile-consensus-v1"
ROLLING_PERIOD_RESOLUTION_VERSION = "rolling_multi_profile_qlib_calendar_v2"
HORIZON_PERIOD_RESOLUTION_VERSION = "rolling_three_horizon_qlib_calendar_v1"
COMMON_FEATURE_SET_CALENDAR_CONTRACT_VERSION = (
    "feature-set-tournament-common-calendar-v1"
)
HORIZON_RESEARCH_SCENARIOS = frozenset(
    {"fin_factor", "fin_model", "fin_quant", "fin_strategy"}
)


class ResearchWindowUnavailableError(ValueError):
    """A statistically required research window cannot be formed honestly.

    ``evidence`` is deliberately carried with the failure so orchestrators can
    expose the unavailable horizon without weakening, shortening, or silently
    relabelling its final OOS contract.
    """

    def __init__(self, message: str, *, evidence: dict[str, Any]) -> None:
        super().__init__(message)
        self.evidence = evidence

_HORIZON_ALIASES = {
    "short": SHORT_1_5D,
    "swing": SWING_1_6M,
    "long": LONG_1_3Y,
    SHORT_1_5D: SHORT_1_5D,
    SWING_1_6M: SWING_1_6M,
    LONG_1_3Y: LONG_1_3Y,
    LEGACY_AMBIGUOUS: LEGACY_AMBIGUOUS,
}


def normalize_research_horizon_profile(value: Any = None) -> str:
    """Normalize user labels without inferring a horizon for old research."""

    raw = str(value or LEGACY_AMBIGUOUS).strip().lower()
    try:
        return _HORIZON_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(f"unsupported research horizon profile: {value}") from exc


def _first_governed_cost_trading_day(calendar_days: list[str]) -> tuple[int, str, str]:
    """Return the first listed session covered by the authoritative CN costs."""

    cost_effective_from = CN_COST_SCHEDULE_BOOK.versions[0].effective_from
    for index, day in enumerate(calendar_days):
        if day >= cost_effective_from:
            return index, day, cost_effective_from
    raise ValueError(
        "Qlib calendar has no trading day covered by the authoritative CN cost schedule"
    )


def required_multi_profile_trading_days(
    *,
    test_trading_days: int = DEFAULT_RESEARCH_PERIOD_POLICY["test_trading_days"],
    embargo_trading_days: int = DEFAULT_RESEARCH_PERIOD_POLICY["embargo_trading_days"],
    purge_trading_days: int = MODEL_LABEL_HORIZON_TRADING_DAYS,
    label_maturity_trading_days: int = 0,
) -> int:
    """Return the minimum calendar coverage required by the governed profiles."""

    longest_validation = max(
        int(item["validation_trading_days"]) for item in RESEARCH_EVALUATION_PROFILES
    )
    return (
        MINIMUM_PROFILE_TRAINING_DAYS
        + int(purge_trading_days)
        + longest_validation
        + int(embargo_trading_days)
        + int(test_trading_days)
        + int(label_maturity_trading_days)
    )


DEFAULT_REQUIRED_RESEARCH_TRADING_DAYS = required_multi_profile_trading_days()


def normalize_research_period_policy(
    value: Any = None,
    *,
    horizon_profile: str | None = None,
) -> dict[str, int]:
    """Normalize the platform-owned rolling-window policy.

    The policy is deliberately expressed in trading days.  Calendar dates are
    resolved later against the selected immutable Qlib dataset, then frozen on
    the concrete research run/campaign.
    """

    if value is None:
        raw: dict[str, Any] = {}
    elif isinstance(value, dict):
        raw = value
    else:
        raise ValueError("rdagent_research period_policy must be an object")
    profile = normalize_research_horizon_profile(horizon_profile)
    horizon = research_horizon_contract(profile)
    defaults = dict(DEFAULT_RESEARCH_PERIOD_POLICY)
    if profile != LEGACY_AMBIGUOUS:
        defaults = {
            "test_trading_days": int(horizon.sealed_oos_sessions or 0),
            # The horizon contract is the modelling minimum.  A capital-facing
            # final OOS also enters the shared alpha-spending ledger, whose
            # immutable contract requires at least the platform default
            # 20-session embargo.  Freeze the stricter value before research so
            # a short-horizon winner cannot become impossible to preregister.
            "embargo_trading_days": max(
                int(horizon.embargo_sessions or 0),
                int(DEFAULT_RESEARCH_PERIOD_POLICY["embargo_trading_days"]),
            ),
        }
    policy: dict[str, int] = {}
    for key, default in defaults.items():
        try:
            policy[key] = int(raw.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rdagent_research {key} must be an integer") from exc
    minimum_test = 252 if profile == LEGACY_AMBIGUOUS else int(horizon.sealed_oos_sessions or 0)
    minimum_embargo = (
        6
        if profile == LEGACY_AMBIGUOUS
        else max(
            int(horizon.embargo_sessions or 0),
            int(DEFAULT_RESEARCH_PERIOD_POLICY["embargo_trading_days"]),
        )
    )
    if policy["test_trading_days"] < minimum_test:
        raise ValueError(
            f"rdagent_research {profile} requires at least {minimum_test} final-test trading days"
        )
    if profile == LEGACY_AMBIGUOUS:
        if not 6 <= policy["embargo_trading_days"] <= 253:
            raise ValueError("rdagent_research embargo must contain 6 to 253 trading days")
    elif policy["embargo_trading_days"] < minimum_embargo:
        raise ValueError(
            f"rdagent_research {profile} requires at least {minimum_embargo} embargo trading days"
        )
    return policy


def normalize_explicit_research_periods(raw_periods: Any) -> dict[str, str]:
    if not isinstance(raw_periods, dict):
        raise ValueError("rdagent_research periods must be an object")
    periods: dict[str, str] = {}
    parsed: dict[str, date] = {}
    for key in RESEARCH_PERIOD_KEYS:
        value = str(raw_periods.get(key) or "")
        try:
            parsed[key] = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"rdagent_research {key} must be an ISO date") from exc
        periods[key] = parsed[key].isoformat()
    if not (
        parsed["train_start"]
        <= parsed["train_end"]
        < parsed["valid_start"]
        <= parsed["valid_end"]
        < parsed["test_start"]
        <= parsed["test_end"]
    ):
        raise ValueError(
            "rdagent_research train, validation, and test periods must be ordered "
            "and non-overlapping"
        )
    return periods


def normalize_research_schedule_payload(
    payload: dict[str, Any],
    *,
    max_loops: int,
    max_duration: str | None = None,
    allow_explicit_periods: bool = False,
) -> dict[str, Any]:
    """Validate and normalize one durable scheduled RD-Agent research request.

    Explicit dates are reserved for platform-internal, already-resolved campaign
    snapshots.  Operator-facing requests must remain rolling so they cannot
    bypass the governed three-profile window contract.
    """

    scenario = get_rdagent_scenario(str(payload.get("scenario") or "fin_factor"))
    horizon_profile = normalize_research_horizon_profile(
        payload.get("horizon_profile")
        or payload.get("strategy_horizon_profile")
        or payload.get("horizon")
    )
    if scenario.id in HORIZON_RESEARCH_SCENARIOS and horizon_profile == LEGACY_AMBIGUOUS:
        raise ValueError(
            f"rdagent_research {scenario.id} requires an explicit horizon profile"
        )
    objective = str(payload.get("objective") or "").strip()
    dataset = str(payload.get("dataset") or "").strip()
    requested_by = str(payload.get("requested_by") or "scheduler").strip()
    if len(objective) < 10 or len(objective) > 2000:
        raise ValueError("rdagent_research objective must contain 10 to 2000 characters")
    if scenario.requires_dataset and not dataset:
        raise ValueError(f"rdagent_research {scenario.id} requires a Qlib dataset")
    if not scenario.requires_dataset and dataset:
        raise ValueError(f"rdagent_research {scenario.id} does not accept a Qlib dataset")
    if len(requested_by) < 2 or len(requested_by) > 100:
        raise ValueError("rdagent_research requested_by must contain 2 to 100 characters")

    try:
        loop_n = int(payload.get("loop_n", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("rdagent_research loop_n must be an integer") from exc
    if loop_n < 1 or loop_n > max_loops:
        raise ValueError(f"rdagent_research loop_n must be between 1 and {max_loops}")
    duration = validate_duration(str(payload.get("duration") or "30m"))
    if max_duration is not None:
        duration = validate_duration_limit(duration, max_duration)

    raw_asset_ids = payload.get("asset_ids") or []
    if not isinstance(raw_asset_ids, list):
        raise ValueError("rdagent_research asset_ids must be an array")
    asset_ids = [validate_asset_id(value) for value in raw_asset_ids]
    if len(set(asset_ids)) != len(asset_ids):
        raise ValueError("rdagent_research asset_ids contain duplicates")
    if scenario.id == "fin_factor_report" and len(asset_ids) > loop_n:
        raise ValueError(
            "rdagent_research fin_factor_report accepts at most one report per loop"
        )
    if (
        asset_ids or not scenario.auto_select_assets
    ) and not scenario.min_assets <= len(asset_ids) <= scenario.max_assets:
        raise ValueError(f"rdagent_research {scenario.id} has an invalid asset count")
    feature_set_id = validate_feature_set_id(payload.get("feature_set_id"))
    feature_set = None
    if scenario.requires_feature_set:
        if feature_set_id is None:
            raise ValueError(f"rdagent_research {scenario.id} requires feature_set_id")
        feature_set = get_feature_set(feature_set_id)
    elif feature_set_id is not None:
        raise ValueError(f"rdagent_research {scenario.id} does not accept feature_set_id")

    period_mode = str(payload.get("period_mode") or "rolling").strip().lower()
    if period_mode not in {"rolling", "explicit"}:
        raise ValueError("rdagent_research period_mode must be rolling or explicit")
    if period_mode == "explicit" and not allow_explicit_periods:
        raise ValueError(
            "rdagent_research explicit periods are platform-internal; "
            "use the rolling period policy"
        )
    normalized = {
        "scenario": scenario.id,
        "objective": objective,
        "dataset": dataset,
        "loop_n": loop_n,
        "duration": duration,
        "requested_by": requested_by,
        "asset_ids": asset_ids,
        # Schedule creation persists the normalized payload and the scheduler
        # validates it again at dispatch time.  Keep the opaque registry ID as
        # well as the frozen definition so normalization is idempotent.
        "feature_set_id": feature_set_id,
        "feature_set": feature_set,
        "horizon_profile": horizon_profile,
    }
    if scenario.id in HORIZON_RESEARCH_SCENARIOS:
        policy = primary_label_policy_contract()
        if (
            payload.get("primary_label_policy") not in (None, policy)
            or payload.get("primary_label_policy_sha256")
            not in (None, policy["policy_sha256"])
        ):
            raise ValueError("rdagent_research primary-label policy changed")
        normalized.update(
            {
                "primary_label_policy": policy,
                "primary_label_policy_sha256": policy["policy_sha256"],
            }
        )
    if not scenario.requires_dataset:
        return normalized
    if period_mode == "explicit":
        normalized["period_mode"] = "explicit"
        normalized["periods"] = normalize_explicit_research_periods(payload.get("periods"))
    else:
        normalized["period_mode"] = "rolling"
        normalized["period_policy"] = normalize_research_period_policy(
            payload.get("period_policy"),
            horizon_profile=horizon_profile,
        )
    return normalized


def derive_rolling_research_periods(
    calendar_days: list[str],
    *,
    train_days: int,
    validation_days: int,
    test_days: int,
    embargo_days: int,
) -> dict[str, str]:
    """Build non-overlapping rolling windows from an actual Qlib trading calendar."""

    if min(train_days, validation_days, test_days, embargo_days) < 1:
        raise ValueError("research window lengths must be positive")
    ordered = sorted(dict.fromkeys(calendar_days))
    total = train_days + validation_days + embargo_days + test_days
    if len(ordered) < total:
        raise ValueError(
            f"Qlib calendar has {len(ordered)} trading days; continuous research requires {total}"
        )
    selected = ordered[-total:]
    train_end = train_days - 1
    valid_start = train_days
    valid_end = train_days + validation_days - 1
    test_start = train_days + validation_days + embargo_days
    return {
        "train_start": selected[0],
        "train_end": selected[train_end],
        "valid_start": selected[valid_start],
        "valid_end": selected[valid_end],
        "test_start": selected[test_start],
        "test_end": selected[-1],
    }


def _derive_multi_profile_research_periods(
    calendar_days: list[str],
    *,
    test_days: int,
    embargo_days: int,
    purge_days: int = MODEL_LABEL_HORIZON_TRADING_DAYS,
    label_maturity_days: int = 0,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    """Build a primary window plus honest, cost-covered stress profiles.

    Every effective profile shares the exact sealed final-OOS boundary.  The
    requested 3/5/10-year validation depths are targets, not permission to
    invent coverage before the authoritative cost schedule.  A target may be
    truncated or declared unavailable, and duplicate effective windows are
    never presented as independent confirmation.
    """

    if min(test_days, embargo_days, purge_days) < 1 or label_maturity_days < 0:
        raise ValueError("research final-test, purge, embargo, and maturity lengths are invalid")
    ordered = sorted(dict.fromkeys(calendar_days))
    minimum_layout = (
        MINIMUM_PROFILE_TRAINING_DAYS
        + purge_days
        + 1
        + embargo_days
        + test_days
        + label_maturity_days
    )
    if len(ordered) < minimum_layout:
        raise ResearchWindowUnavailableError(
            f"Qlib calendar has {len(ordered)} trading days; the frozen final OOS, "
            f"purge, embargo, maturity, training, and one validation session require "
            f"{minimum_layout}",
            evidence={
                "contract_version": "horizon-multi-profile-resolution-v1",
                "calendar_trading_days": len(ordered),
                "minimum_layout_trading_days": minimum_layout,
                "capital_evaluation_eligible": False,
                "capital_evaluation_unavailable_reason": (
                    "insufficient_calendar_for_frozen_timing_contract"
                ),
                "requested_profiles": [],
                "effective_profiles": [],
                "unavailable_profiles": [],
            },
        )
    test_end_index = len(ordered) - label_maturity_days - 1
    test_start_index = test_end_index - test_days + 1
    valid_end_index = test_start_index - embargo_days - 1
    _, first_cost_trading_day, cost_effective_from = _first_governed_cost_trading_day(
        ordered
    )
    cost_start_index = next(
        index for index, day in enumerate(ordered) if day >= first_cost_trading_day
    )
    earliest_valid_start_index = MINIMUM_PROFILE_TRAINING_DAYS + purge_days
    profiles: list[dict[str, Any]] = []
    requested_profiles: list[dict[str, Any]] = []
    unavailable_profiles: list[dict[str, Any]] = []
    effective_windows: dict[tuple[int, int], str] = {}
    for spec in RESEARCH_EVALUATION_PROFILES:
        requested_validation_days = int(spec["validation_trading_days"])
        requested_valid_start_index = valid_end_index - requested_validation_days + 1
        valid_start_index = max(
            requested_valid_start_index,
            cost_start_index,
            earliest_valid_start_index,
        )
        constraints: list[str] = []
        if requested_valid_start_index < 0:
            constraints.append("requested_validation_precedes_available_calendar")
        if valid_start_index == cost_start_index and (
            requested_valid_start_index < cost_start_index
        ):
            constraints.append("authoritative_cn_cost_schedule_start")
        if valid_start_index == earliest_valid_start_index and (
            requested_valid_start_index < earliest_valid_start_index
        ):
            constraints.append("minimum_training_and_purge_floor")
        requested_profile = {
            "id": str(spec["id"]),
            "label": str(spec["label"]),
            "role": str(spec["role"]),
            "requested_validation_trading_days": requested_validation_days,
            "requested_valid_start": (
                ordered[requested_valid_start_index]
                if requested_valid_start_index >= 0
                else None
            ),
        }
        if valid_start_index > valid_end_index:
            unavailable = {
                **requested_profile,
                "status": "unavailable",
                "effective_validation_trading_days": 0,
                "unavailable_reason": "no_cost_covered_validation_session",
                "binding_constraints": constraints,
            }
            requested_profiles.append(unavailable)
            unavailable_profiles.append(unavailable)
            continue
        train_end_index = valid_start_index - purge_days - 1
        effective_training_days = train_end_index + 1
        if effective_training_days < MINIMUM_PROFILE_TRAINING_DAYS:
            unavailable = {
                **requested_profile,
                "status": "unavailable",
                "effective_validation_trading_days": 0,
                "effective_training_trading_days": effective_training_days,
                "unavailable_reason": "minimum_training_history_not_met",
                "binding_constraints": constraints,
            }
            requested_profiles.append(unavailable)
            unavailable_profiles.append(unavailable)
            continue
        effective_validation_days = valid_end_index - valid_start_index + 1
        window_key = (valid_start_index, valid_end_index)
        duplicate_of = effective_windows.get(window_key)
        if duplicate_of is not None and spec["role"] == "confirmation":
            primary_profile = next(
                (item for item in profiles if item["role"] == "primary"), None
            )
            stress_profile = next(
                (item for item in profiles if item["role"] == "stress"), None
            )
            if primary_profile is not None and stress_profile is not None:
                primary_start = ordered.index(
                    str(primary_profile["periods"]["valid_start"])
                )
                stress_start = ordered.index(
                    str(stress_profile["periods"]["valid_start"])
                )
                # A requested confirmation depth that collapses onto the
                # stress boundary may be shortened to the deterministic
                # midpoint.  This records what was actually tested and avoids
                # counting a one-session perturbation as independent evidence.
                if stress_start + 2 <= primary_start:
                    valid_start_index = (stress_start + primary_start) // 2
                    constraints.append(
                        "confirmation_shortened_to_distinct_cost_covered_midpoint"
                    )
                    train_end_index = valid_start_index - purge_days - 1
                    effective_training_days = train_end_index + 1
                    effective_validation_days = (
                        valid_end_index - valid_start_index + 1
                    )
                    window_key = (valid_start_index, valid_end_index)
                    duplicate_of = effective_windows.get(window_key)
        if duplicate_of is not None:
            unavailable = {
                **requested_profile,
                "status": "unavailable",
                "effective_validation_trading_days": effective_validation_days,
                "effective_valid_start": ordered[valid_start_index],
                "effective_valid_end": ordered[valid_end_index],
                "unavailable_reason": "duplicate_effective_validation_window",
                "duplicate_of_profile_id": duplicate_of,
                "binding_constraints": constraints,
            }
            requested_profiles.append(unavailable)
            unavailable_profiles.append(unavailable)
            continue
        effective_windows[window_key] = str(spec["id"])
        truncated = effective_validation_days != requested_validation_days
        periods = {
            "train_start": ordered[0],
            "train_end": ordered[train_end_index],
            "valid_start": ordered[valid_start_index],
            "valid_end": ordered[valid_end_index],
            "test_start": ordered[test_start_index],
            "test_end": ordered[test_end_index],
        }
        effective = {
            **spec,
            "status": "available",
            "requested_validation_trading_days": requested_validation_days,
            "effective_validation_trading_days": effective_validation_days,
            "effective_training_trading_days": effective_training_days,
            "requested_valid_start": requested_profile["requested_valid_start"],
            "effective_valid_start": ordered[valid_start_index],
            "effective_valid_end": ordered[valid_end_index],
            "validation_window_truncated": truncated,
            "validation_window_truncation_reason": (
                "+".join(constraints) if truncated else None
            ),
            "binding_constraints": constraints,
            "authoritative_cost_schedule_effective_from": cost_effective_from,
            "authoritative_cost_schedule_first_trading_day": first_cost_trading_day,
            "periods": periods,
        }
        profiles.append(effective)
        requested_profiles.append(effective)
    primary = [item for item in profiles if item["role"] == "primary"]
    stresses = [item for item in profiles if item["role"] == "stress"]
    distinct_stress = bool(
        primary
        and any(item["periods"] != primary[0]["periods"] for item in stresses)
    )
    resolution = {
        "contract_version": "horizon-multi-profile-resolution-v1",
        "calendar_trading_days": len(ordered),
        "authoritative_cost_schedule_effective_from": cost_effective_from,
        "authoritative_cost_schedule_first_trading_day": first_cost_trading_day,
        "requested_profiles": requested_profiles,
        "effective_profiles": [str(item["id"]) for item in profiles],
        "unavailable_profiles": unavailable_profiles,
        "required_capital_roles": ["primary", "stress"],
        "capital_evaluation_eligible": bool(primary and distinct_stress),
        "capital_evaluation_unavailable_reason": (
            None
            if primary and distinct_stress
            else "primary_and_distinct_stress_profiles_required"
        ),
    }
    if not resolution["capital_evaluation_eligible"]:
        raise ResearchWindowUnavailableError(
            "multi-profile capital evaluation requires an available primary profile "
            "and one genuinely different cost-covered stress profile",
            evidence=resolution,
        )
    discovery = dict(primary[0]["periods"])
    for item in profiles:
        item["profile_resolution_contract_version"] = resolution["contract_version"]
    return dict(discovery), profiles, resolution


def derive_multi_profile_research_periods(
    calendar_days: list[str],
    *,
    test_days: int,
    embargo_days: int,
    purge_days: int = MODEL_LABEL_HORIZON_TRADING_DAYS,
    label_maturity_days: int = 0,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Compatibility facade returning only executable effective profiles."""

    discovery, profiles, _ = _derive_multi_profile_research_periods(
        calendar_days,
        test_days=test_days,
        embargo_days=embargo_days,
        purge_days=purge_days,
        label_maturity_days=label_maturity_days,
    )
    return discovery, profiles


def resolve_research_periods(
    calendar_days: list[str],
    *,
    periods: Any = None,
    period_policy: Any = None,
    horizon_profile: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Resolve explicit dates or derive rolling dates and return audit metadata."""

    ordered = sorted(dict.fromkeys(str(day).strip() for day in calendar_days if str(day).strip()))
    if not ordered:
        raise ValueError("Qlib trading calendar is empty")
    try:
        parsed = [date.fromisoformat(day) for day in ordered]
    except ValueError as exc:
        raise ValueError("Qlib trading calendar contains an invalid ISO date") from exc
    if parsed != sorted(set(parsed)):
        raise ValueError("Qlib trading calendar must contain unique ordered dates")

    profile = normalize_research_horizon_profile(horizon_profile)
    horizon = research_horizon_contract(profile)
    labels = (
        (MODEL_LABEL_HORIZON_TRADING_DAYS,)
        if profile == LEGACY_AMBIGUOUS
        else horizon.label_horizons_sessions
    )
    purge_days = (
        MODEL_LABEL_HORIZON_TRADING_DAYS
        if profile == LEGACY_AMBIGUOUS
        else int(horizon.purge_sessions or 0)
    )
    maturity_days = 0 if profile == LEGACY_AMBIGUOUS else max(labels)

    if periods is not None:
        resolved = normalize_explicit_research_periods(periods)
        _, first_cost_trading_day, _ = _first_governed_cost_trading_day(ordered)
        if resolved["valid_start"] < first_cost_trading_day:
            raise ValueError(
                "explicit research validation starts before the first trading day "
                "covered by the authoritative CN cost schedule"
            )
        mode = "explicit_periods_v1"
        policy = None
        profiles = [
            {
                "id": "explicit",
                "label": "显式窗口",
                "role": "primary",
                "weight": 1.0,
                "periods": resolved,
            }
        ]
        indices = {day: index for index, day in enumerate(ordered)}
        missing = [value for value in resolved.values() if value not in indices]
        if missing:
            raise ValueError("explicit research periods must use Qlib trading sessions")
        purge_gap = indices[resolved["valid_start"]] - indices[resolved["train_end"]] - 1
        embargo_gap = indices[resolved["test_start"]] - indices[resolved["valid_end"]] - 1
        maturity_tail = len(ordered) - indices[resolved["test_end"]] - 1
        if profile != LEGACY_AMBIGUOUS and purge_gap < purge_days:
            raise ValueError(
                f"explicit {profile} research requires at least {purge_days} purge sessions"
            )
        required_embargo = max(
            int(horizon.embargo_sessions or 0),
            int(DEFAULT_RESEARCH_PERIOD_POLICY["embargo_trading_days"]),
        )
        if profile != LEGACY_AMBIGUOUS and embargo_gap < required_embargo:
            raise ValueError(
                f"explicit {profile} research requires at least "
                f"{required_embargo} embargo sessions"
            )
        if profile != LEGACY_AMBIGUOUS and maturity_tail < maturity_days:
            raise ValueError(
                f"explicit {profile} research requires {maturity_days} post-OOS "
                "sessions for label maturity"
            )
        effective_embargo_days = embargo_gap
        effective_maturity_days = maturity_tail if profile != LEGACY_AMBIGUOUS else 0
    else:
        policy = normalize_research_period_policy(
            period_policy,
            horizon_profile=profile,
        )
        resolved, profiles, profile_resolution = _derive_multi_profile_research_periods(
            ordered,
            test_days=policy["test_trading_days"],
            embargo_days=policy["embargo_trading_days"],
            purge_days=purge_days,
            label_maturity_days=maturity_days,
        )
        mode = (
            ROLLING_PERIOD_RESOLUTION_VERSION
            if profile == LEGACY_AMBIGUOUS
            else HORIZON_PERIOD_RESOLUTION_VERSION
        )
        effective_embargo_days = policy["embargo_trading_days"]
        effective_maturity_days = maturity_days
    if periods is not None:
        profile_resolution = {
            "contract_version": "explicit-profile-resolution-v1",
            "requested_profiles": profiles,
            "effective_profiles": [str(item["id"]) for item in profiles],
            "unavailable_profiles": [],
            "required_capital_roles": ["primary"],
            "capital_evaluation_eligible": True,
            "capital_evaluation_unavailable_reason": None,
        }
    latest_mature_label_sessions: dict[str, str] = {}
    for label in labels:
        if len(ordered) <= label:
            raise ValueError(f"Qlib calendar cannot mature the {label}-session label")
        latest_mature_label_sessions[str(label)] = ordered[-label - 1]
    return resolved, {
        "mode": mode,
        "policy": policy,
        "horizon_profile": profile,
        "horizon_contract_sha256": horizon.sha256,
        "label_horizons_sessions": list(labels),
        "purge_trading_days": purge_days,
        "embargo_trading_days": effective_embargo_days,
        "label_maturity_enforced": profile != LEGACY_AMBIGUOUS,
        "label_maturity_tail_trading_days": effective_maturity_days,
        "latest_mature_label_sessions": latest_mature_label_sessions,
        "calendar_start": ordered[0],
        "calendar_end": ordered[-1],
        "calendar_trading_days": len(ordered),
        "final_test_trading_days": sum(
            resolved["test_start"] <= day <= resolved["test_end"] for day in ordered
        ),
        "evaluation_profiles": profiles,
        "evaluation_profile_resolution": profile_resolution,
    }


def resolve_research_window_contract(
    dataset: dict[str, Any],
    calendar_days: list[str],
    *,
    periods: Any = None,
    period_policy: Any = None,
    horizon_profile: str | None = None,
    feature_set: dict[str, Any] | None = None,
    universe: str = "cn_all_governed_ashare_and_etf",
    random_seed: int = 42,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Resolve dates and freeze the complete first-class research contract."""

    profile = normalize_research_horizon_profile(horizon_profile)
    effective_calendar = list(calendar_days)
    requested_data_cutoff_session: str | None = None
    effective_field_cutoff_session: str | None = None
    field_evidence: dict[str, Any] | None = None
    if profile != LEGACY_AMBIGUOUS:
        features = (feature_set or {}).get("features")
        if not isinstance(features, dict) or not features:
            raise ValueError("active horizon research requires a governed feature set")
        required_fields = sorted(
            {
                field
                for expression in features.values()
                for field in compile_qlib_expression(str(expression)).required_fields
            }
        )
        ordered_calendar = sorted(
            dict.fromkeys(
                str(day).strip() for day in calendar_days if str(day).strip()
            )
        )
        if not ordered_calendar:
            raise ValueError("Qlib trading calendar is empty")
        requested_data_cutoff_session = ordered_calendar[-1]
        provenance = dataset.get("provenance") or {}
        if not isinstance(provenance, dict):
            raise ValueError("dataset provenance must be an object")
        field_evidence = resolve_required_field_coverage(
            provenance,
            required_fields,
            data_cutoff_session=requested_data_cutoff_session,
        )
        effective_start = str(field_evidence["effective_field_start_session"])
        effective_end = str(field_evidence["effective_field_available_to"])
        effective_calendar = [
            day
            for day in ordered_calendar
            if effective_start <= day <= effective_end
        ]
        if not effective_calendar:
            raise ValueError("selected dataset fields have no common Qlib trading sessions")
        effective_field_cutoff_session = effective_calendar[-1]
        field_evidence = {
            **field_evidence,
            "effective_field_cutoff_session": effective_field_cutoff_session,
        }
        if periods is not None:
            explicit = normalize_explicit_research_periods(periods)
            if explicit["train_start"] < effective_calendar[0]:
                raise ValueError(
                    "explicit research begins before all selected factor fields are available"
                )
            if explicit["test_end"] > effective_field_cutoff_session:
                raise ValueError(
                    "explicit research ends after all selected factor fields are available"
                )
    resolved, evidence = resolve_research_periods(
        effective_calendar,
        periods=periods,
        period_policy=period_policy,
        horizon_profile=profile,
    )
    contract = build_research_window_contract(
        dataset=dataset,
        calendar_days=effective_calendar,
        periods=resolved,
        period_resolution=evidence,
        horizon_profile=str(evidence["horizon_profile"]),
        feature_set=feature_set,
        universe=universe,
        random_seed=random_seed,
        requested_data_cutoff_session=requested_data_cutoff_session,
        effective_field_cutoff_session=effective_field_cutoff_session,
    )
    evidence = {
        **evidence,
        **({"required_field_coverage": field_evidence} if field_evidence else {}),
        "research_window_contract": contract.to_dict(),
        "research_window_contract_sha256": contract.sha256,
    }
    return resolved, evidence


def resolve_common_feature_set_calendar(
    dataset: dict[str, Any],
    calendar_days: list[str],
    feature_sets: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    """Freeze one honest calendar for a feature-set tournament.

    A fixed-model feature screen is only attributable to its feature set when
    every candidate sees the same train/validation/OOS dates.  Resolve the
    continuously usable range of the union of all preregistered fields, then
    exclude every earlier/later session for every candidate.  Missing history
    is never represented by fabricated zero-valued rows.
    """

    ordered = sorted(
        dict.fromkeys(str(day).strip() for day in calendar_days if str(day).strip())
    )
    if not ordered:
        raise ValueError("Qlib trading calendar is empty")
    if not feature_sets:
        raise ValueError("feature-set tournament has no candidates")
    provenance = dataset.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError("dataset provenance must be an object")

    requirements: list[dict[str, Any]] = []
    all_fields: set[str] = set()
    seen_ids: set[str] = set()
    for raw_feature_set in feature_sets:
        feature_set = dict(raw_feature_set)
        feature_set_id = str(feature_set.get("id") or "")
        definition_sha256 = str(feature_set.get("definition_sha256") or "")
        features = feature_set.get("features")
        if (
            not feature_set_id
            or feature_set_id in seen_ids
            or len(definition_sha256) != 64
            or not isinstance(features, dict)
            or not features
        ):
            raise ValueError("feature-set tournament candidate is invalid")
        required_fields = sorted(
            {
                field
                for expression in features.values()
                for field in compile_qlib_expression(str(expression)).required_fields
            }
        )
        if not required_fields:
            raise ValueError("feature-set tournament candidate uses no dataset fields")
        seen_ids.add(feature_set_id)
        all_fields.update(required_fields)
        requirements.append(
            {
                "feature_set_id": feature_set_id,
                "feature_set_definition_sha256": definition_sha256,
                "required_fields": required_fields,
            }
        )

    field_evidence = resolve_required_field_coverage(
        provenance,
        sorted(all_fields),
        data_cutoff_session=ordered[-1],
    )
    common_start = str(field_evidence["effective_field_start_session"])
    common_end = str(field_evidence["effective_field_available_to"])
    common_calendar = [day for day in ordered if common_start <= day <= common_end]
    if not common_calendar:
        raise ValueError("feature-set tournament fields have no common Qlib sessions")

    evidence = {
        "contract_version": COMMON_FEATURE_SET_CALENDAR_CONTRACT_VERSION,
        "feature_sets": sorted(requirements, key=lambda item: item["feature_set_id"]),
        "required_field_union": sorted(all_fields),
        "required_field_coverage": field_evidence,
        "calendar_start": common_calendar[0],
        "calendar_end": common_calendar[-1],
        "calendar_trading_days": len(common_calendar),
        "missing_history_policy": "exclude_sessions_fail_closed_never_zero_backfill",
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return common_calendar, evidence


def build_multi_profile_consensus(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Build the explicit admission record for the three governed profiles."""

    expected = {str(item["id"]): item for item in RESEARCH_EVALUATION_PROFILES}
    evaluations = candidate.get("profile_evaluations")
    if not isinstance(evaluations, list):
        return None
    by_profile = {
        profile_id: item
        for item in evaluations
        if isinstance(item, dict)
        and isinstance(item.get("metrics"), dict)
        and (
            profile_id := str(
                (item.get("metrics") or {}).get("research_profile", {}).get("id") or ""
            )
        )
        in expected
    }
    if set(by_profile) != set(expected):
        return None
    candidate_id = str(candidate.get("id") or "")
    code_sha256 = str(candidate.get("code_sha256") or "")
    values_sha256 = str(candidate.get("values_sha256") or "")
    if not candidate_id or len(code_sha256) != 64 or len(values_sha256) != 64:
        return None
    ordered = [by_profile[profile_id] for profile_id in sorted(expected)]
    if any(
        str(item.get("factor_candidate_id") or candidate_id) != candidate_id
        for item in ordered
    ):
        return None
    if any(
        str(item.get("candidate_code_sha256") or "") != code_sha256
        or str(item.get("candidate_values_sha256") or "") != values_sha256
        for item in ordered
    ):
        return None
    identities = {str(item.get("dataset_identity_sha256") or "") for item in ordered}
    test_windows = {
        (str(item.get("test_start") or ""), str(item.get("test_end") or "")) for item in ordered
    }
    if len(identities) != 1 or "" in identities or len(test_windows) != 1:
        return None
    profile_periods = {
        key: {
            period_key: str(by_profile[key].get(period_key) or "")
            for period_key in RESEARCH_PERIOD_KEYS
        }
        for key in sorted(expected)
    }
    if any(not value for periods in profile_periods.values() for value in periods.values()):
        return None
    if len({periods["train_start"] for periods in profile_periods.values()}) != 1:
        return None
    if len({periods["valid_end"] for periods in profile_periods.values()}) != 1:
        return None
    if not (
        profile_periods["robust_10y"]["valid_start"]
        < profile_periods["balanced_5y"]["valid_start"]
        < profile_periods["recent_3y"]["valid_start"]
    ):
        return None
    recent = by_profile["recent_3y"]
    balanced = by_profile["balanced_5y"]
    robust = by_profile["robust_10y"]
    if recent.get("gate_status") != "passed" or balanced.get("gate_status") != "passed":
        return None
    metrics = [item["metrics"] for item in (recent, balanced, robust)]
    directions = {str(item.get("direction") or "") for item in metrics}
    if len(directions) != 1 or "" in directions:
        return None
    robust_metrics = robust["metrics"]
    robust_return = robust_metrics.get("cost_adjusted_return")
    if (
        robust_metrics.get("coverage_gate_passed") is not True
        or robust_return is None
        or float(robust_return) < 0.0
    ):
        return None
    evaluation_ids = {key: str(by_profile[key].get("id") or "") for key in sorted(expected)}
    evidence_sha256 = {
        key: str(by_profile[key].get("evidence_sha256") or "") for key in sorted(expected)
    }
    metrics_sha256 = {
        key: str(by_profile[key].get("metrics_sha256") or "") for key in sorted(expected)
    }
    if (
        any(len(value) != 32 for value in evaluation_ids.values())
        or any(len(value) != 64 for value in evidence_sha256.values())
        or any(len(value) != 64 for value in metrics_sha256.values())
    ):
        return None
    test_start, test_end = next(iter(test_windows))
    return {
        "version": MULTI_PROFILE_CONSENSUS_VERSION,
        "status": "passed",
        "candidate_id": candidate_id,
        "candidate_code_sha256": code_sha256,
        "candidate_values_sha256": values_sha256,
        "dataset_identity_sha256": next(iter(identities)),
        "test_start": test_start,
        "test_end": test_end,
        "direction": next(iter(directions)),
        "evaluation_ids": evaluation_ids,
        "evaluation_evidence_sha256": evidence_sha256,
        "evaluation_metrics_sha256": metrics_sha256,
        "profile_periods": profile_periods,
        "profile_gate_status": {
            key: str(by_profile[key].get("gate_status") or "") for key in sorted(expected)
        },
        "rules": {
            "recent_gate_passed": True,
            "balanced_gate_passed": True,
            "all_directions_agree": True,
            "robust_coverage_passed": True,
            "robust_cost_adjusted_return_nonnegative": True,
        },
    }


def rank_multi_profile_candidates(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
    reference_candidates: list[dict[str, Any]] | None = None,
    max_abs_spearman: float = 0.75,
    max_factors_per_family: int = 3,
) -> list[dict[str, Any]]:
    """Rank only candidates with an explicit three-profile admission record."""

    eligible: list[dict[str, Any]] = []
    expected = {str(item["id"]): item for item in RESEARCH_EVALUATION_PROFILES}
    for candidate in candidates:
        consensus = build_multi_profile_consensus(candidate)
        if consensus is None:
            continue
        by_profile = {
            profile_id: item
            for item in candidate["profile_evaluations"]
            if isinstance(item, dict)
            and isinstance(item.get("metrics"), dict)
            and (
                profile_id := str(
                    (item.get("metrics") or {})
                    .get("research_profile", {})
                    .get("id")
                    or ""
                )
            )
            in expected
        }
        profile_scores: dict[str, float] = {}
        for profile_id in expected:
            evaluation = by_profile[profile_id]
            profile_candidate = {**candidate, "latest_evaluation": evaluation}
            profile_score = factor_rank_score(profile_candidate)
            if not math.isfinite(profile_score):
                # The stress profile may miss the strict significance gate but
                # must still be directionally consistent and non-destructive.
                profile_metrics = evaluation["metrics"]
                profile_score = round(
                    abs(float(profile_metrics.get("icir") or 0.0))
                    + abs(float(profile_metrics.get("rank_icir") or 0.0))
                    + 4.0 * max(0.0, float(profile_metrics.get("cost_adjusted_return") or 0.0))
                    - 0.25 * max(0.0, float(profile_metrics.get("turnover") or 0.0)),
                    10,
                )
            profile_scores[profile_id] = profile_score
        eligible.append(
            {
                **candidate,
                # Nested windows are dependent observations, so they must not
                # be averaged as if they were three independent experiments.
                # Recent evidence ranks; balanced and robust evidence are gates.
                "automation_score": round(profile_scores["recent_3y"], 10),
                "profile_scores": profile_scores,
                "profile_gate_status": {
                    key: by_profile[key]["gate_status"] for key in sorted(by_profile)
                },
                "profile_consensus": consensus,
            }
        )
    ranked = sorted(
        eligible,
        key=lambda item: (-float(item["automation_score"]), str(item.get("id") or "")),
    )
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    references = list(reference_candidates or [])
    for candidate in ranked:
        if any(
            _factor_spearman(candidate, other) > max_abs_spearman
            for other in [*references, *selected]
        ):
            continue
        families = candidate_economic_families(candidate)
        if any(family_counts.get(family, 0) >= max_factors_per_family for family in families):
            continue
        selected.append(candidate)
        for family in families:
            family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) == limit:
            break
    return selected


def select_latest_program_dataset(
    datasets: list[dict[str, Any]], *, lineage_id: str
) -> dict[str, Any] | None:
    """Select the newest reproducible dataset without crossing its approved lineage."""

    eligible = [
        item
        for item in datasets
        if item.get("ready")
        and item.get("reproducible")
        and item.get("lineage_verified")
        and item.get("lineage_id") == lineage_id
        and item.get("end_date")
        and (item.get("provenance") or {}).get("dataset_identity_sha256")
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (str(item["end_date"]), str(item["name"])))


def select_latest_program_rebase_dataset(
    datasets: list[dict[str, Any]], *, anchor: dict[str, Any]
) -> dict[str, Any] | None:
    """Select a newer, contract-compatible dataset for an audited lineage rebase.

    A source or builder code revision intentionally starts a new immutable
    lineage.  Long-running research policies must not become pinned forever,
    but neither may they silently treat the new lineage as a descendant.  This
    selector therefore requires the public Qlib data contract to preserve every
    existing field and unit.  Additive PIT-governed fields are allowed; the
    controller records the lineage change as a separate governance event.
    """

    anchor_provenance = anchor.get("provenance") or {}
    scalar_contract_keys = (
        "field_contract_version",
        "frequency",
        "eligibility_contract_version",
        "source_start_date",
    )
    anchor_contract = {
        key: anchor_provenance.get(key) for key in scalar_contract_keys
    }
    anchor_fields = set(anchor_provenance.get("fields") or [])
    anchor_units = anchor_provenance.get("field_units") or {}
    if not (
        anchor_provenance.get("dataset_contract_sha256")
        and anchor_fields
        and isinstance(anchor_units, dict)
    ):
        return None
    eligible = []
    for item in datasets:
        provenance = item.get("provenance") or {}
        if not (
            item.get("ready")
            and item.get("reproducible")
            and item.get("lineage_verified")
            and item.get("lineage_id")
            and item.get("lineage_id") != anchor.get("lineage_id")
            and item.get("end_date")
            and str(item["end_date"]) > str(anchor.get("end_date") or "")
            and provenance.get("dataset_identity_sha256")
        ):
            continue
        candidate_contract = {
            key: provenance.get(key) for key in scalar_contract_keys
        }
        candidate_fields = set(provenance.get("fields") or [])
        candidate_units = provenance.get("field_units") or {}
        preserves_units = isinstance(candidate_units, dict) and all(
            anchor_units.get(field) is None
            or candidate_units.get(field) == anchor_units.get(field)
            for field in anchor_fields
        )
        if (
            candidate_contract == anchor_contract
            and anchor_fields.issubset(candidate_fields)
            and preserves_units
        ):
            eligible.append(item)
    if not eligible:
        return None
    return max(eligible, key=lambda item: (str(item["end_date"]), str(item["name"])))


def factor_rank_score(candidate: dict[str, Any]) -> float:
    """Return a deterministic score using independent Qlib evidence only."""

    evaluation = candidate.get("latest_evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("gate_status") != "passed":
        return -math.inf
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, dict):
        return -math.inf

    def number(name: str, default: float = 0.0) -> float:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return default

    return round(
        abs(number("icir"))
        + abs(number("rank_icir"))
        + 4.0 * max(0.0, number("cost_adjusted_return"))
        - 0.25 * max(0.0, number("turnover")),
        10,
    )


def rank_factor_candidates(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
    reference_candidates: list[dict[str, Any]] | None = None,
    max_abs_spearman: float = 0.75,
    max_factors_per_family: int = 3,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("factor selection limit must be positive")
    eligible = []
    for candidate in candidates:
        score = factor_rank_score(candidate)
        if math.isfinite(score):
            eligible.append({**candidate, "automation_score": score})
    ranked = sorted(
        eligible,
        key=lambda item: (-float(item["automation_score"]), str(item.get("id") or "")),
    )
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    references = list(reference_candidates or [])
    for candidate in ranked:
        if any(
            _factor_spearman(candidate, other) > max_abs_spearman
            for other in [*references, *selected]
        ):
            continue
        families = candidate_economic_families(candidate)
        if any(family_counts.get(family, 0) >= max_factors_per_family for family in families):
            continue
        selected.append(candidate)
        for family in families:
            family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) == limit:
            break
    return selected


def candidate_economic_families(candidate: dict[str, Any]) -> tuple[str, ...]:
    tags = {
        str(value)
        for value in (candidate.get("family_tags") or [])
        if str(value) in ECONOMIC_FAMILIES and str(value) != "mixed"
    }
    primary = str(candidate.get("economic_family") or "mixed")
    if primary in ECONOMIC_FAMILIES and primary != "mixed":
        tags.add(primary)
    return tuple(sorted(tags or {"mixed"}))


def allocate_governed_factor_weights(
    candidates: list[dict[str, Any]],
    *,
    max_factor_weight: float = 0.25,
    max_family_weight: float = 0.35,
) -> list[float]:
    """Find explicit weights without silently relaxing factor/family caps."""

    if not candidates:
        raise ValueError("factor weighting requires at least one candidate")
    families = sorted(
        {family for item in candidates for family in candidate_economic_families(item)}
    )
    family_rows = [
        [
            1.0 if family in candidate_economic_families(item) else 0.0
            for item in candidates
        ]
        for family in families
    ]
    scores = [float(item.get("automation_score") or 0.0) for item in candidates]
    result = linprog(
        c=[-score - index * 1e-12 for index, score in enumerate(scores)],
        A_ub=family_rows,
        b_ub=[max_family_weight] * len(family_rows),
        A_eq=[[1.0] * len(candidates)],
        b_eq=[1.0],
        bounds=[(0.01, max_factor_weight)] * len(candidates),
        method="highs",
    )
    if not result.success:
        raise ValueError(
            "selected factors cannot satisfy the 25% single-factor and 35% family caps"
        )
    weights = [float(value) for value in result.x]
    if abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError("governed factor weights failed the sum-to-one invariant")
    return weights


def _factor_spearman(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_path = left.get("values_path")
    right_path = right.get("values_path")
    if not left_path or not right_path:
        return 0.0

    def load(path: str, name: str) -> pd.Series:
        frame = (
            pd.read_parquet(path) if str(path).lower().endswith(".parquet") else pd.read_hdf(path)
        )
        return normalize_series(frame, name)

    try:
        left_values = load(str(left_path), "left")
        right_values = load(str(right_path), "right")
    except (OSError, ValueError):
        return 1.0
    pair = pd.concat([left_values, right_values], axis=1, join="inner").dropna()
    if pair.empty:
        return 1.0
    daily = pair.groupby(level="datetime").apply(
        lambda group: group.iloc[:, 0].rank().corr(group.iloc[:, 1].rank())
    )
    return float(daily.abs().mean()) if daily.notna().any() else 1.0


def factor_spearman(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Public governed similarity primitive used by selection and the DB ledger."""

    return _factor_spearman(left, right)
