from __future__ import annotations

import pytest

from quant_platform.member_risk_gate import compose_strategy_risk_state

pytestmark = pytest.mark.no_database


def test_allocation_reduction_exposes_numeric_cap_and_requires_reactivation() -> None:
    state = compose_strategy_risk_state(
        "pair-version",
        allocation_gates=[
            {
                "allocation_id": "allocation-a",
                "allocation_status": "risk_reduction_pending",
                "state": "risk_reduction",
                "risk_exposure_override": 0.5,
                "event_ids": [11],
                "requires_reactivation": True,
            }
        ],
    )

    assert state["state"] == "risk_reduction"
    assert state["allocation_risk_state"] == "risk_reduction"
    assert state["member_risk_state"] == "active"
    assert state["risk_exposure_override"] == pytest.approx(0.5)
    assert state["allow_new_risk"] is False
    assert state["recovery"] == {
        "member_events_must_be_resolved": [],
        "allocation_events_must_be_resolved": [11],
        "allocation_ids_requiring_reactivation": ["allocation-a"],
        "member_gate_reopens_on_resolution": False,
        "allocation_gate_requires_explicit_active_state": True,
    }


def test_liquidation_dominates_reduction_across_allocations() -> None:
    state = compose_strategy_risk_state(
        "pair-version",
        allocation_gates=[
            {
                "allocation_id": "allocation-reduce",
                "state": "risk_reduction",
                "risk_exposure_override": 0.5,
                "event_ids": [21],
                "requires_reactivation": True,
            },
            {
                "allocation_id": "allocation-liquidate",
                "state": "liquidation",
                "risk_exposure_override": 0.0,
                "event_ids": [22],
                "requires_reactivation": True,
            },
        ],
    )

    assert state["state"] == "liquidation"
    assert state["risk_exposure_override"] == 0.0
    assert state["event_ids"] == [21, 22]
    assert state["allocation_ids"] == [
        "allocation-liquidate",
        "allocation-reduce",
    ]


def test_member_drawdown_gate_remains_distinct_from_allocation_exposure() -> None:
    state = compose_strategy_risk_state(
        "pair-version",
        member_event_ids=[31],
        member_allocation_ids=["allocation-a"],
    )

    assert state["state"] == "pause_new_risk"
    assert state["member_risk_state"] == "pause_new_risk"
    assert state["allocation_risk_state"] == "active"
    assert state["risk_exposure_override"] == 1.0
    assert state["allow_new_risk"] is False
    assert state["recovery"]["member_gate_reopens_on_resolution"] is True
