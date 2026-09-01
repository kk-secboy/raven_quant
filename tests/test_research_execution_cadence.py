from __future__ import annotations

import pytest

from quant_platform.research_execution_cadence import (
    build_research_execution_cadence_contract,
    is_research_decision_step,
    validate_research_execution_cadence_contract,
)
from quant_platform.research_horizon import (
    LONG_1_3Y,
    SHORT_1_5D,
    SWING_1_6M,
)

pytestmark = pytest.mark.no_database


@pytest.mark.parametrize(
    ("profile", "interval"),
    ((SHORT_1_5D, 1), (SWING_1_6M, 5), (LONG_1_3Y, 21)),
)
def test_research_execution_cadence_is_bound_to_the_horizon(
    profile: str,
    interval: int,
) -> None:
    contract = build_research_execution_cadence_contract(profile)

    assert contract["decision_interval_sessions"] == interval
    assert contract["signal_lag_sessions"] == 1
    assert contract["execution_timing"] == "next_trading_session"
    assert validate_research_execution_cadence_contract(
        contract,
        expected_horizon_profile=profile,
    ) == contract


@pytest.mark.parametrize(
    ("interval", "due_steps"),
    ((1, {0, 1, 2, 3, 4, 5}), (5, {0, 5}), (21, {0})),
)
def test_research_decision_steps_hold_between_governed_rebalances(
    interval: int,
    due_steps: set[int],
) -> None:
    observed = {
        step
        for step in range(6)
        if is_research_decision_step(
            step,
            decision_interval_sessions=interval,
        )
    }

    assert observed == due_steps


def test_research_execution_cadence_rejects_tampering_and_legacy() -> None:
    contract = build_research_execution_cadence_contract(SWING_1_6M)
    with pytest.raises(ValueError, match="contract is invalid"):
        validate_research_execution_cadence_contract(
            {**contract, "decision_interval_sessions": 1},
            expected_horizon_profile=SWING_1_6M,
        )
    with pytest.raises(ValueError, match="active horizon"):
        build_research_execution_cadence_contract("legacy_ambiguous")
