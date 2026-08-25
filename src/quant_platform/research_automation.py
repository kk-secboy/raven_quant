from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd

from .factor_evaluator import normalize_series
from .feature_set_registry import get_feature_set
from .model_research_governance import MODEL_LABEL_HORIZON_TRADING_DAYS
from .rdagent_runtime import validate_duration, validate_duration_limit
from .rdagent_scenarios import (
    get_rdagent_scenario,
    validate_asset_id,
    validate_feature_set_id,
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
    "embargo_trading_days": 5,
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


def required_multi_profile_trading_days(
    *,
    test_trading_days: int = DEFAULT_RESEARCH_PERIOD_POLICY["test_trading_days"],
    embargo_trading_days: int = DEFAULT_RESEARCH_PERIOD_POLICY["embargo_trading_days"],
) -> int:
    """Return the minimum calendar coverage required by the governed profiles."""

    longest_validation = max(
        int(item["validation_trading_days"]) for item in RESEARCH_EVALUATION_PROFILES
    )
    return (
        MINIMUM_PROFILE_TRAINING_DAYS
        + MODEL_LABEL_HORIZON_TRADING_DAYS
        + longest_validation
        + int(embargo_trading_days)
        + int(test_trading_days)
    )


DEFAULT_REQUIRED_RESEARCH_TRADING_DAYS = required_multi_profile_trading_days()


def normalize_research_period_policy(value: Any = None) -> dict[str, int]:
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
    policy: dict[str, int] = {}
    for key, default in DEFAULT_RESEARCH_PERIOD_POLICY.items():
        try:
            policy[key] = int(raw.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rdagent_research {key} must be an integer") from exc
    if policy["test_trading_days"] < 252:
        raise ValueError("rdagent_research requires at least 252 final-test trading days")
    if not 5 <= policy["embargo_trading_days"] <= 63:
        raise ValueError("rdagent_research embargo must contain 5 to 63 trading days")
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
        "feature_set": feature_set,
    }
    if not scenario.requires_dataset:
        return normalized
    if period_mode == "explicit":
        normalized["period_mode"] = "explicit"
        normalized["periods"] = normalize_explicit_research_periods(payload.get("periods"))
    else:
        normalized["period_mode"] = "rolling"
        normalized["period_policy"] = normalize_research_period_policy(payload.get("period_policy"))
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


def derive_multi_profile_research_periods(
    calendar_days: list[str],
    *,
    test_days: int,
    embargo_days: int,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Build one discovery window and three pre-final evaluation profiles.

    Every profile shares the exact same final OOS boundary.  Only validation
    history changes, so comparing profiles cannot select on final-test results.
    """

    if min(test_days, embargo_days) < 1:
        raise ValueError("research final-test and embargo lengths must be positive")
    ordered = sorted(dict.fromkeys(calendar_days))
    required = required_multi_profile_trading_days(
        test_trading_days=test_days,
        embargo_trading_days=embargo_days,
    )
    if len(ordered) < required:
        raise ValueError(
            f"Qlib calendar has {len(ordered)} trading days; multi-profile research "
            f"requires {required}"
        )
    selected = ordered[-required:]
    test_start_index = len(selected) - test_days
    valid_end_index = test_start_index - embargo_days - 1
    profiles: list[dict[str, Any]] = []
    for spec in RESEARCH_EVALUATION_PROFILES:
        valid_start_index = valid_end_index - int(spec["validation_trading_days"]) + 1
        train_end_index = valid_start_index - MODEL_LABEL_HORIZON_TRADING_DAYS - 1
        periods = {
            "train_start": selected[0],
            "train_end": selected[train_end_index],
            "valid_start": selected[valid_start_index],
            "valid_end": selected[valid_end_index],
            "test_start": selected[test_start_index],
            "test_end": selected[-1],
        }
        profiles.append({**spec, "periods": periods})
    discovery = next(item["periods"] for item in profiles if item["role"] == "primary")
    return dict(discovery), profiles


def resolve_research_periods(
    calendar_days: list[str],
    *,
    periods: Any = None,
    period_policy: Any = None,
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

    if periods is not None:
        resolved = normalize_explicit_research_periods(periods)
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
    else:
        policy = normalize_research_period_policy(period_policy)
        resolved, profiles = derive_multi_profile_research_periods(
            ordered,
            test_days=policy["test_trading_days"],
            embargo_days=policy["embargo_trading_days"],
        )
        mode = "rolling_multi_profile_qlib_calendar_v1"
    return resolved, {
        "mode": mode,
        "policy": policy,
        "calendar_start": ordered[0],
        "calendar_end": ordered[-1],
        "calendar_trading_days": len(ordered),
        "final_test_trading_days": sum(
            resolved["test_start"] <= day <= resolved["test_end"] for day in ordered
        ),
        "evaluation_profiles": profiles,
    }


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
    references = list(reference_candidates or [])
    for candidate in ranked:
        if any(
            _factor_spearman(candidate, other) > max_abs_spearman
            for other in [*references, *selected]
        ):
            continue
        selected.append(candidate)
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
    references = list(reference_candidates or [])
    for candidate in ranked:
        if any(
            _factor_spearman(candidate, other) > max_abs_spearman
            for other in [*references, *selected]
        ):
            continue
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


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
