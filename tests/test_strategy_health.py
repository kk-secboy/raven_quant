from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_platform.member_risk_gate import compose_strategy_risk_state
from quant_platform.promotion import _initial_promotion_health
from quant_platform.strategy_health import (
    HEALTH_WINDOWS_BY_HORIZON,
    RESTRICTED,
    SUSPENDED,
    WATCH,
    assess_strategy_health,
    cap_targets_for_health,
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
