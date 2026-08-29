from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from quant_platform.member_risk_gate import compose_strategy_risk_state
from quant_platform.promotion import _initial_promotion_health
from quant_platform.research_horizon import canonical_sha256
from quant_platform.strategy_health import (
    HEALTH_WINDOWS_BY_HORIZON,
    RESTRICTED,
    SUSPENDED,
    WATCH,
    assess_strategy_health,
    cap_targets_for_health,
    resolve_feature_drift_episode,
    transition_strategy_health,
)

pytestmark = pytest.mark.no_database


def _healthy_evidence(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "data_integrity_ok": True,
        "ledger_reconciled": True,
        "drawdown": 0.02,
        "turnover": 0.10,
        "cost_ratio": 0.10,
        "execution_rejection_rate": 0.01,
        "model_calibration_drift": 0.03,
        "feature_drift": 0.05,
        "data_completeness": 1.0,
    }
    result.update(overrides)
    return result


_HORIZON_SHA256 = "c" * 64
_VERSION_ID = "strategy-version-a"


def _drift_snapshot(
    as_of: datetime,
    *,
    feature_drift: object,
    threshold: object = 0.20,
    data_integrity_ok: object = True,
    ledger_reconciled: object = True,
) -> dict[str, object]:
    criteria = {"watch_feature_drift": threshold}
    evidence = {
        "feature_drift": feature_drift,
        "data_integrity_ok": data_integrity_ok,
        "ledger_reconciled": ledger_reconciled,
    }
    criteria_sha256 = canonical_sha256(criteria)
    evidence_sha256 = canonical_sha256(evidence)
    payload = {
        "contract_version": "strategy-health-snapshot-v1",
        "strategy_version_id": _VERSION_ID,
        "horizon_profile": "short_1_5d",
        "horizon_contract_sha256": _HORIZON_SHA256,
        "as_of": as_of.isoformat(),
        "health_status": "watch",
        "criteria_json": criteria,
        "criteria_sha256": criteria_sha256,
        "evidence_json": evidence,
        "evidence_sha256": evidence_sha256,
        "recorded_by": "health-test",
    }
    snapshot_sha256 = canonical_sha256(payload)
    return {
        "id": snapshot_sha256,
        "snapshot_sha256": snapshot_sha256,
        "strategy_version_id": _VERSION_ID,
        "horizon_profile": "short_1_5d",
        "as_of": as_of,
        "health_status": "watch",
        "criteria_json": criteria,
        "criteria_sha256": criteria_sha256,
        "evidence_json": evidence,
        "evidence_sha256": evidence_sha256,
        "recorded_by": "health-test",
        "recorded_at": as_of,
    }


def _resolve_drift(*snapshots: dict[str, object], observed_at: datetime) -> dict:
    return resolve_feature_drift_episode(
        snapshots,
        expected_strategy_version_id=_VERSION_ID,
        expected_horizon_profile="short_1_5d",
        expected_horizon_contract_sha256=_HORIZON_SHA256,
        observed_at=observed_at,
    )


def test_feature_drift_contiguous_breach_is_one_stable_episode() -> None:
    start = datetime(2026, 8, 25, 15, 0, tzinfo=UTC)
    below = _drift_snapshot(start, feature_drift=0.10)
    first_high = _drift_snapshot(start + timedelta(days=1), feature_drift=0.21)
    second_high = _drift_snapshot(start + timedelta(days=2), feature_drift=0.28)

    first = _resolve_drift(
        first_high,
        below,
        observed_at=start + timedelta(days=1, minutes=1),
    )
    retry = _resolve_drift(
        second_high,
        first_high,
        below,
        observed_at=start + timedelta(days=2, minutes=1),
    )

    assert first["due"] is True
    assert retry["due"] is True
    assert retry["trigger_id"] == first["trigger_id"]
    assert retry["event"]["episode_start_snapshot_sha256"] == first_high["id"]
    assert retry["event"]["latest_snapshot_sha256"] == second_high["id"]


def test_feature_drift_recovery_then_new_breach_creates_new_episode() -> None:
    start = datetime(2026, 8, 25, 15, 0, tzinfo=UTC)
    first_high = _drift_snapshot(start, feature_drift=0.25)
    recovered = _drift_snapshot(start + timedelta(days=1), feature_drift=0.05)
    second_high = _drift_snapshot(start + timedelta(days=2), feature_drift=0.23)

    first = _resolve_drift(first_high, observed_at=start + timedelta(minutes=1))
    second = _resolve_drift(
        second_high,
        recovered,
        first_high,
        observed_at=start + timedelta(days=2, minutes=1),
    )

    assert first["due"] is second["due"] is True
    assert first["trigger_id"] != second["trigger_id"]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"feature_drift": float("nan")}, "feature_drift_evidence_missing"),
        ({"feature_drift": -0.1}, "feature_drift_evidence_missing"),
        ({"data_integrity_ok": False}, "feature_drift_hard_gate_failed"),
        ({"ledger_reconciled": False}, "feature_drift_hard_gate_failed"),
    ],
)
def test_feature_drift_invalid_or_failed_evidence_never_triggers(
    overrides: dict[str, object], reason: str
) -> None:
    observed_at = datetime(2026, 8, 28, 15, 1, tzinfo=UTC)
    snapshot = _drift_snapshot(
        observed_at - timedelta(minutes=1),
        feature_drift=overrides.get("feature_drift", 0.30),
        data_integrity_ok=overrides.get("data_integrity_ok", True),
        ledger_reconciled=overrides.get("ledger_reconciled", True),
    )

    result = _resolve_drift(snapshot, observed_at=observed_at)

    assert result == {
        "due": False,
        "reason": reason,
        "trigger_id": None,
        "event": None,
    }


def test_feature_drift_uses_sealed_custom_threshold_and_rejects_tampering() -> None:
    observed_at = datetime(2026, 8, 28, 15, 1, tzinfo=UTC)
    snapshot = _drift_snapshot(
        observed_at - timedelta(minutes=1),
        feature_drift=0.25,
        threshold=0.30,
    )
    below = _resolve_drift(snapshot, observed_at=observed_at)
    snapshot["criteria_json"] = {"watch_feature_drift": 0.10}
    tampered = _resolve_drift(snapshot, observed_at=observed_at)

    assert below["reason"] == "feature_drift_below_watch_threshold"
    assert tampered["reason"] == "feature_drift_evidence_invalid"


def test_feature_drift_excludes_future_or_late_recorded_snapshots() -> None:
    observed_at = datetime(2026, 8, 28, 15, 0, tzinfo=UTC)
    current = _drift_snapshot(
        observed_at - timedelta(minutes=1), feature_drift=0.10
    )
    future = _drift_snapshot(
        observed_at + timedelta(minutes=1), feature_drift=0.30
    )
    late = _drift_snapshot(
        observed_at - timedelta(minutes=2), feature_drift=0.30
    )
    late["recorded_at"] = observed_at + timedelta(minutes=1)

    result = _resolve_drift(future, late, current, observed_at=observed_at)

    assert result["reason"] == "feature_drift_below_watch_threshold"


def test_health_windows_are_horizon_specific() -> None:
    assert HEALTH_WINDOWS_BY_HORIZON == {
        "short_1_5d": (20, 60),
        "swing_1_6m": (63, 126),
        "long_1_3y": (252, 504, 756),
    }


def test_health_assessment_escalates_and_recovers_one_step_at_a_time() -> None:
    watch = assess_strategy_health(
        "short_1_5d",
        _healthy_evidence(feature_drift=0.22),
    )
    assert watch["health_status"] == WATCH
    assert watch["allow_new_risk"] is True

    restricted = assess_strategy_health(
        "short_1_5d",
        _healthy_evidence(feature_drift=0.40),
        previous_status=WATCH,
    )
    assert restricted["health_status"] == RESTRICTED
    assert restricted["allow_new_risk"] is False

    suspended = assess_strategy_health(
        "short_1_5d",
        _healthy_evidence(ledger_reconciled=False),
        previous_status=RESTRICTED,
    )
    assert suspended["health_status"] == SUSPENDED
    assert suspended["allow_new_risk"] is False

    recovery = assess_strategy_health(
        "short_1_5d",
        _healthy_evidence(),
        previous_status=SUSPENDED,
    )
    assert recovery["proposed_status"] == "healthy"
    assert recovery["health_status"] == RESTRICTED
    assert recovery["allow_new_risk"] is False
    assert transition_strategy_health(RESTRICTED, "healthy") == WATCH


def test_restricted_targets_allow_reductions_and_exits_but_never_increases() -> None:
    targets, gate = cap_targets_for_health(
        {"A": 0.20, "B": 0.05, "C": 0.10},
        {"A": 0.10, "B": 0.10},
        RESTRICTED,
    )

    assert targets == {"A": pytest.approx(0.10), "B": pytest.approx(0.05)}
    assert set(gate["blocked_increases"]) == {"A", "C"}
    assert gate["allow_new_risk"] is False


def test_health_gate_composes_with_existing_risk_state_without_blocking_exits() -> None:
    state = compose_strategy_risk_state(
        "strategy-a",
        strategy_health_gate={
            "health_status": RESTRICTED,
            "snapshot_id": "health-1",
        },
    )

    assert state["state"] == "strategy_health_restricted"
    assert state["allow_new_risk"] is False
    assert state["risk_exposure_override"] == 1.0
    assert state["strategy_health_gate"]["snapshot_id"] == "health-1"


def test_passing_forward_gate_bootstraps_healthy_or_watch_activity_health() -> None:
    version = SimpleNamespace(id="version-a", horizon_profile="short_1_5d")
    evaluation = {
        "passed": True,
        "contract_version": "promotion-chain-v1",
        "criteria_sha256": "a" * 64,
        "criteria_json": {
            "thresholds": {
                "min_data_completeness": 0.99,
                "min_reconciliation_rate": 0.99,
                "max_cost_deviation": 0.02,
            }
        },
        "stage_id": "stage-a",
        "evidence": {
            "data_completeness": 1.0,
            "reconciliation_rate": 1.0,
            "cost_deviation": 0.005,
            "ungoverned_batches": 0,
            "invalid_round_trip_fills": 0,
            "forward_trading_days": 90,
            "decision_batches": 60,
            "review_events": 60,
            "closed_round_trips": 30,
            "financial_report_reviews": 0,
        },
    }

    status, criteria, evidence = _initial_promotion_health(version, evaluation)
    assert status == "healthy"
    assert criteria["source_forward_gate_criteria_sha256"] == "a" * 64
    assert evidence["data_integrity_ok"] is True
    assert evidence["ledger_reconciled"] is True

    evaluation["evidence"] = {
        **evaluation["evidence"],
        "data_completeness": 0.995,
    }
    status, _criteria, evidence = _initial_promotion_health(version, evaluation)
    assert status == "watch"
    assert evidence["initial_health_status"] == "watch"
