"""Governed activity-health policy for promoted strategy versions.

This module deliberately does not create a second strategy lifecycle.  A
health snapshot is an append-only observation attached to the existing
``StrategyVersion``.  Its only execution authority is a fail-closed new-risk
gate: ``restricted``, ``suspended`` and ``retired`` may reduce or exit an
existing position, but may never increase it.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from .research_horizon import LONG_1_3Y, SHORT_1_5D, SWING_1_6M

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
        "watch_execution_rejection_rate": 0.10,
        "restrict_execution_rejection_rate": 0.25,
        "watch_cost_ratio": 0.25,
        "restrict_cost_ratio": 0.50,
    },
}

_SEVERITY = {HEALTHY: 0, WATCH: 1, RESTRICTED: 2, SUSPENDED: 3, RETIRED: 4}


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
    names = (
        ("drawdown", f"{level}_drawdown"),
        ("feature_drift", f"{level}_feature_drift"),
        (
            "execution_rejection_rate",
            f"{level}_execution_rejection_rate",
        ),
        ("cost_ratio", f"{level}_cost_ratio"),
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
    required = {"data_integrity_ok", "ledger_reconciled"}
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
        "model_calibration_drift",
        "feature_drift",
    ):
        number = float(evidence.get(name, 0.0))
        if not isfinite(number) or number < 0:
            raise ValueError(f"strategy health evidence {name} must be finite and non-negative")
        result[name] = number
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
