"""Governed activity-health policy for promoted strategy versions.

This module deliberately does not create a second strategy lifecycle.  A
health snapshot is an append-only observation attached to the existing
``StrategyVersion``.  Its only execution authority is a fail-closed new-risk
gate: ``restricted``, ``suspended`` and ``retired`` may reduce or exit an
existing position, but may never increase it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from math import isfinite
from typing import Any

from .research_horizon import (
    LONG_1_3Y,
    SHORT_1_5D,
    SWING_1_6M,
    canonical_sha256,
)

COLLECTOR_ACTOR = "system:strategy-health-collector"
HEALTHY = "healthy"
WATCH = "watch"
RESTRICTED = "restricted"
SUSPENDED = "suspended"
RETIRED = "retired"

STRATEGY_HEALTH_STATUSES = (HEALTHY, WATCH, RESTRICTED, SUSPENDED, RETIRED)
NEW_RISK_ALLOWED_STATUSES = frozenset({HEALTHY, WATCH})
NEW_RISK_BLOCKED_STATUSES = frozenset({RESTRICTED, SUSPENDED, RETIRED})

HEALTH_WINDOWS_BY_HORIZON = {
    SHORT_1_5D: (20, 60),
    SWING_1_6M: (63, 126),
    LONG_1_3Y: (252, 504, 756),
}

DEFAULT_HEALTH_CRITERIA = {
    SHORT_1_5D: {
        "watch_drawdown": 0.06,
        "restrict_drawdown": 0.10,
        "suspend_drawdown": 0.15,
        "watch_feature_drift": 0.20,
        "restrict_feature_drift": 0.35,
        "watch_model_calibration_drift": 0.20,
        "restrict_model_calibration_drift": 0.35,
        "watch_execution_rejection_rate": 0.10,
        "restrict_execution_rejection_rate": 0.25,
        "watch_cost_ratio": 0.35,
        "restrict_cost_ratio": 0.60,
    },
    SWING_1_6M: {
        "watch_drawdown": 0.08,
        "restrict_drawdown": 0.12,
        "suspend_drawdown": 0.18,
        "watch_feature_drift": 0.20,
        "restrict_feature_drift": 0.35,
        "watch_model_calibration_drift": 0.20,
        "restrict_model_calibration_drift": 0.35,
        "watch_execution_rejection_rate": 0.10,
        "restrict_execution_rejection_rate": 0.25,
        "watch_cost_ratio": 0.30,
        "restrict_cost_ratio": 0.55,
    },
    LONG_1_3Y: {
        "watch_drawdown": 0.10,
        "restrict_drawdown": 0.15,
        "suspend_drawdown": 0.22,
        "watch_feature_drift": 0.20,
        "restrict_feature_drift": 0.35,
        "watch_model_calibration_drift": 0.20,
        "restrict_model_calibration_drift": 0.35,
        "watch_execution_rejection_rate": 0.10,
        "restrict_execution_rejection_rate": 0.25,
        "watch_cost_ratio": 0.25,
        "restrict_cost_ratio": 0.50,
    },
}

_SEVERITY = {HEALTHY: 0, WATCH: 1, RESTRICTED: 2, SUSPENDED: 3, RETIRED: 4}

FEATURE_DRIFT_EPISODE_CONTRACT_VERSION = "strategy-feature-drift-episode-v1"


def resolve_feature_drift_episode(
    snapshots: Iterable[Mapping[str, Any]],
    *,
    expected_strategy_version_id: str,
    expected_horizon_profile: str,
    expected_horizon_contract_sha256: str,
    observed_at: datetime,
) -> dict[str, Any]:
    """Resolve one content-addressed feature-drift breach episode.

    The input is the append-only health history for the exact incumbent,
    newest first.  A missing or ungoverned latest observation fails closed;
    it is never interpreted as zero drift.  Once the latest observation is a
    governed breach, older contiguous breaches resolve to one stable episode
    start, so daily scheduler retries cannot create fresh research work.
    """

    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("feature drift observed_at must be timezone-aware")
    if expected_horizon_profile not in HEALTH_WINDOWS_BY_HORIZON:
        raise ValueError("feature drift horizon is invalid")
    _require_digest(
        expected_horizon_contract_sha256,
        field="feature drift horizon contract",
    )
    cutoff = observed_at.astimezone(UTC)
    ordered: list[dict[str, Any]] = []
    for raw in snapshots:
        try:
            normalized = _normalize_drift_snapshot(
                raw,
                expected_strategy_version_id=expected_strategy_version_id,
                expected_horizon_profile=expected_horizon_profile,
                expected_horizon_contract_sha256=expected_horizon_contract_sha256,
            )
        except (KeyError, TypeError, ValueError):
            return _no_drift_event("feature_drift_evidence_invalid")
        if normalized["as_of"] > cutoff or normalized["recorded_at"] > cutoff:
            continue
        ordered.append(normalized)
    ordered.sort(
        key=lambda item: (
            item["as_of"],
            item["recorded_at"],
            item["snapshot_sha256"],
        ),
        reverse=True,
    )
    if not ordered:
        return _no_drift_event("feature_drift_evidence_missing")

    latest = ordered[0]
    if latest["observation"] is None:
        return _no_drift_event("feature_drift_evidence_missing")
    if not latest["observation"]["hard_gates_passed"]:
        return _no_drift_event("feature_drift_hard_gate_failed")
    if not latest["observation"]["breached"]:
        return _no_drift_event("feature_drift_below_watch_threshold")

    episode_start = latest
    for candidate in ordered[1:]:
        observation = candidate["observation"]
        if (
            observation is None
            or not observation["hard_gates_passed"]
            or not observation["breached"]
        ):
            break
        episode_start = candidate

    identity = {
        "contract_version": FEATURE_DRIFT_EPISODE_CONTRACT_VERSION,
        "strategy_version_id": expected_strategy_version_id,
        "horizon_profile": expected_horizon_profile,
        "episode_start_snapshot_sha256": episode_start["snapshot_sha256"],
    }
    trigger_id = canonical_sha256(identity)
    latest_observation = latest["observation"]
    return {
        "due": True,
        "reason": "feature_drift_episode_due",
        "trigger_id": trigger_id,
        "event": {
            **identity,
            "trigger_id": trigger_id,
            "kind": "feature_drift_episode",
            "source": "strategy_health_snapshots",
            "metric": "feature_drift",
            "threshold_key": "watch_feature_drift",
            "latest_snapshot_sha256": latest["snapshot_sha256"],
            "latest_as_of": latest["as_of"].isoformat(),
            "latest_recorded_at": latest["recorded_at"].isoformat(),
            "latest_evidence_sha256": latest["evidence_sha256"],
            "latest_criteria_sha256": latest["criteria_sha256"],
            "feature_drift": latest_observation["feature_drift"],
            "watch_feature_drift": latest_observation["watch_feature_drift"],
        },
    }


def _normalize_drift_snapshot(
    raw: Mapping[str, Any],
    *,
    expected_strategy_version_id: str,
    expected_horizon_profile: str,
    expected_horizon_contract_sha256: str,
) -> dict[str, Any]:
    snapshot = dict(raw)
    snapshot_sha256 = _require_digest(
        snapshot.get("snapshot_sha256"), field="strategy health snapshot"
    )
    if _require_digest(snapshot.get("id"), field="strategy health snapshot id") != snapshot_sha256:
        raise ValueError("strategy health snapshot id differs from its seal")
    if str(snapshot.get("strategy_version_id") or "") != expected_strategy_version_id:
        raise ValueError("strategy health snapshot belongs to another version")
    if str(snapshot.get("horizon_profile") or "") != expected_horizon_profile:
        raise ValueError("strategy health snapshot belongs to another horizon")
    as_of = _aware_datetime(snapshot.get("as_of"), field="strategy health as_of")
    recorded_at = _aware_datetime(
        snapshot.get("recorded_at"), field="strategy health recorded_at"
    )
    criteria = dict(snapshot.get("criteria_json") or {})
    evidence = dict(snapshot.get("evidence_json") or {})
    criteria_sha256 = _require_digest(
        snapshot.get("criteria_sha256"), field="strategy health criteria"
    )
    evidence_sha256 = _require_digest(
        snapshot.get("evidence_sha256"), field="strategy health evidence"
    )
    if canonical_sha256(criteria) != criteria_sha256:
        raise ValueError("strategy health criteria seal is invalid")
    if canonical_sha256(evidence) != evidence_sha256:
        raise ValueError("strategy health evidence seal is invalid")
    sealed_snapshot = {
        "contract_version": "strategy-health-snapshot-v1",
        "strategy_version_id": expected_strategy_version_id,
        "horizon_profile": expected_horizon_profile,
        "horizon_contract_sha256": expected_horizon_contract_sha256,
        "as_of": as_of.astimezone(UTC).replace(microsecond=0).isoformat(),
        "health_status": str(snapshot.get("health_status") or ""),
        "criteria_json": criteria,
        "criteria_sha256": criteria_sha256,
        "evidence_json": evidence,
        "evidence_sha256": evidence_sha256,
        "recorded_by": str(snapshot.get("recorded_by") or ""),
    }
    if canonical_sha256(sealed_snapshot) != snapshot_sha256:
        raise ValueError("strategy health snapshot content seal is invalid")
    return {
        "snapshot_sha256": snapshot_sha256,
        "as_of": as_of.astimezone(UTC),
        "recorded_at": recorded_at.astimezone(UTC),
        "criteria_sha256": criteria_sha256,
        "evidence_sha256": evidence_sha256,
        "observation": _drift_observation(criteria, evidence),
    }


def _drift_observation(
    criteria: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any] | None:
    required = {
        "watch_feature_drift",
        "feature_drift",
        "data_integrity_ok",
        "ledger_reconciled",
    }
    if not required.issubset(set(criteria) | set(evidence)):
        return None
    if "watch_feature_drift" not in criteria or any(
        field not in evidence
        for field in ("feature_drift", "data_integrity_ok", "ledger_reconciled")
    ):
        return None
    try:
        feature_drift = float(evidence["feature_drift"])
        threshold = float(criteria["watch_feature_drift"])
    except (TypeError, ValueError):
        return None
    if (
        not isfinite(feature_drift)
        or feature_drift < 0
        or not isfinite(threshold)
        or not 0 <= threshold <= 1
        or not isinstance(evidence["data_integrity_ok"], bool)
        or not isinstance(evidence["ledger_reconciled"], bool)
    ):
        return None
    hard_gates_passed = (
        evidence["data_integrity_ok"] is True
        and evidence["ledger_reconciled"] is True
    )
    return {
        "feature_drift": feature_drift,
        "watch_feature_drift": threshold,
        "hard_gates_passed": hard_gates_passed,
        "breached": feature_drift >= threshold,
    }


def _require_digest(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _aware_datetime(value: Any, *, field: str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value or ""))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed


def _no_drift_event(reason: str) -> dict[str, Any]:
    return {"due": False, "reason": reason, "trigger_id": None, "event": None}


def health_allows_new_risk(status: str | None) -> bool:
    """Return whether the latest health state may increase exposure.

    Missing/unknown state fails closed.  Legacy versions that predate the
    explicit horizon contract are filtered by the caller and do not use this
    gate.
    """

    return str(status or "") in NEW_RISK_ALLOWED_STATUSES


def transition_strategy_health(previous: str | None, proposed: str) -> str:
    """Apply monotone deterioration and one-step-at-a-time recovery.

    Hard evidence can move directly to ``suspended``.  Recovery is deliberately
    slower: one fresh assessment can improve by at most one state.  ``retired``
    is terminal and cannot be reopened by routine health collection.
    """

    if proposed not in STRATEGY_HEALTH_STATUSES:
        raise ValueError(f"unsupported proposed strategy health: {proposed}")
    if previous is None:
        return proposed
    if previous not in STRATEGY_HEALTH_STATUSES:
        raise ValueError(f"unsupported previous strategy health: {previous}")
    if previous == RETIRED:
        return RETIRED
    previous_level = _SEVERITY[previous]
    proposed_level = _SEVERITY[proposed]
    if proposed_level >= previous_level:
        return proposed
    return next(
        status for status, severity in _SEVERITY.items() if severity == previous_level - 1
    )


def assess_strategy_health(
    horizon: str,
    evidence: Mapping[str, Any],
    *,
    previous_status: str | None = None,
    criteria: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministically assess one strategy-health observation.

    Evidence uses loss-positive fractions: ``drawdown=0.12`` means a 12%
    drawdown.  Data integrity and ledger reconciliation are hard gates.
    Feature drift, execution rejection and realized cost/edge are graded
    warnings.  Callers persist the returned criteria and evidence with
    :meth:`StrategyStore.record_health_snapshot`.
    """

    if horizon not in HEALTH_WINDOWS_BY_HORIZON:
        raise ValueError(f"unsupported strategy health horizon: {horizon}")
    resolved_criteria = dict(DEFAULT_HEALTH_CRITERIA[horizon])
    if criteria is not None:
        resolved_criteria.update(dict(criteria))
    _validate_criteria(resolved_criteria)
    normalized = _normalize_evidence(evidence)

    proposed = HEALTHY
    reasons: list[str] = []
    hard_failures: list[str] = []
    if not normalized["data_integrity_ok"]:
        hard_failures.append("data_integrity_failed")
    if not normalized["ledger_reconciled"]:
        hard_failures.append("ledger_reconciliation_failed")
    if normalized["drawdown"] >= resolved_criteria["suspend_drawdown"]:
        hard_failures.append("drawdown_suspend_limit_breached")
    if hard_failures:
        proposed = SUSPENDED
        reasons.extend(hard_failures)
    else:
        restricted = _threshold_reasons(normalized, resolved_criteria, "restrict")
        watch = _threshold_reasons(normalized, resolved_criteria, "watch")
        if restricted:
            proposed = RESTRICTED
            reasons.extend(restricted)
        elif watch:
            proposed = WATCH
            reasons.extend(watch)
        else:
            reasons.append("all_activity_health_checks_passed")

    status = transition_strategy_health(previous_status, proposed)
    if status != proposed:
        reasons.append(f"recovery_limited_from_{previous_status}_toward_{proposed}")
    return {
        "contract_version": "strategy-health-policy-v1",
        "horizon_profile": horizon,
        "windows_trading_days": list(HEALTH_WINDOWS_BY_HORIZON[horizon]),
        "previous_status": previous_status,
        "proposed_status": proposed,
        "health_status": status,
        "allow_new_risk": health_allows_new_risk(status),
        "criteria": resolved_criteria,
        "evidence": normalized,
        "reasons": reasons,
    }


def cap_targets_for_health(
    targets: Mapping[str, float],
    previous_targets: Mapping[str, float],
    health_status: str | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Prevent a health-restricted sleeve from increasing any instrument.

    Reductions and exits remain intact.  A retired strategy targets zero.  A
    missing state is treated like suspension for explicit three-horizon
    production versions.
    """

    normalized_targets = _finite_weights(targets, label="targets")
    previous = _finite_weights(previous_targets, label="previous targets")
    if health_status == RETIRED:
        capped = {}
    elif health_allows_new_risk(health_status):
        capped = normalized_targets
    else:
        capped = {
            instrument: min(weight, previous.get(instrument, 0.0))
            for instrument, weight in normalized_targets.items()
            if min(weight, previous.get(instrument, 0.0)) > 1e-12
        }
    blocked = {
        instrument: {
            "requested_weight": weight,
            "allowed_weight": capped.get(instrument, 0.0),
            "reason": f"strategy_health_{health_status or 'missing'}_blocks_new_risk",
        }
        for instrument, weight in normalized_targets.items()
        if weight > capped.get(instrument, 0.0) + 1e-12
    }
    return capped, {
        "health_status": health_status or "missing",
        "allow_new_risk": health_allows_new_risk(health_status),
        "blocked_increases": blocked,
    }


def _threshold_reasons(
    evidence: Mapping[str, Any], criteria: Mapping[str, float], level: str
) -> list[str]:
    names = [
        ("drawdown", f"{level}_drawdown"),
        ("feature_drift", f"{level}_feature_drift"),
        (
            "execution_rejection_rate",
            f"{level}_execution_rejection_rate",
        ),
        ("cost_ratio", f"{level}_cost_ratio"),
    ]
    if evidence.get("model_calibration_drift") is not None:
        names.append(
            (
                "model_calibration_drift",
                f"{level}_model_calibration_drift",
            )
        )
    return [
        f"{metric}_{level}_limit_breached"
        for metric, threshold in names
        if float(evidence[metric]) >= float(criteria[threshold])
    ]


def _validate_criteria(criteria: Mapping[str, Any]) -> None:
    for name, value in criteria.items():
        number = float(value)
        if not isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"strategy health criterion {name} must be in [0, 1]")
    for metric in (
        "drawdown",
        "feature_drift",
        "model_calibration_drift",
        "execution_rejection_rate",
        "cost_ratio",
    ):
        watch = float(criteria[f"watch_{metric}"])
        restricted = float(criteria[f"restrict_{metric}"])
        if watch >= restricted:
            raise ValueError(f"watch {metric} must be below restrict {metric}")
    if float(criteria["restrict_drawdown"]) >= float(criteria["suspend_drawdown"]):
        raise ValueError("restrict drawdown must be below suspend drawdown")


def _normalize_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    required = {"data_integrity_ok", "ledger_reconciled", "feature_drift"}
    missing = required.difference(evidence)
    if missing:
        raise ValueError(f"strategy health evidence is missing: {sorted(missing)}")
    result: dict[str, Any] = {
        "data_integrity_ok": evidence["data_integrity_ok"] is True,
        "ledger_reconciled": evidence["ledger_reconciled"] is True,
    }
    for name in (
        "drawdown",
        "turnover",
        "cost_ratio",
        "execution_rejection_rate",
        "feature_drift",
    ):
        number = float(evidence.get(name, 0.0))
        if not isfinite(number) or number < 0:
            raise ValueError(f"strategy health evidence {name} must be finite and non-negative")
        result[name] = number
    model_required = evidence.get("model_calibration_required") is True
    model_available = evidence.get("model_calibration_evidence_available") is True
    raw_model_drift = evidence.get("model_calibration_drift")
    if model_required and (not model_available or raw_model_drift is None):
        raise ValueError("required model calibration evidence is unavailable")
    if raw_model_drift is None:
        model_drift = None
    else:
        model_drift = float(raw_model_drift)
        if not isfinite(model_drift) or model_drift < 0:
            raise ValueError(
                "strategy health evidence model_calibration_drift must be finite "
                "and non-negative"
            )
    result["model_calibration_required"] = model_required
    result["model_calibration_evidence_available"] = model_available
    result["model_calibration_drift"] = model_drift
    result["data_completeness"] = float(evidence.get("data_completeness", 1.0))
    if not 0 <= result["data_completeness"] <= 1:
        raise ValueError("strategy health data completeness must be in [0, 1]")
    return result


def _finite_weights(values: Mapping[str, float], *, label: str) -> dict[str, float]:
    result = {str(instrument): float(weight) for instrument, weight in values.items()}
    if any(not instrument for instrument in result):
        raise ValueError(f"{label} require non-empty instruments")
    if any(not isfinite(weight) or weight < 0 for weight in result.values()):
        raise ValueError(f"{label} must be finite and non-negative")
    return result
