from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .research_horizon import canonical_sha256, research_horizon_contract

RESEARCH_EXECUTION_CADENCE_CONTRACT_VERSION = "research-execution-cadence-v1"


def build_research_execution_cadence_contract(
    horizon_profile: str,
) -> dict[str, Any]:
    """Bind a research portfolio backtest to its product decision cadence."""

    horizon = research_horizon_contract(str(horizon_profile))
    interval = horizon.decision_interval_sessions
    if interval is None or isinstance(interval, bool) or int(interval) <= 0:
        raise ValueError("research execution cadence requires an active horizon")
    contract: dict[str, Any] = {
        "contract_version": RESEARCH_EXECUTION_CADENCE_CONTRACT_VERSION,
        "horizon_profile": horizon.horizon_profile,
        "horizon_contract_sha256": horizon.sha256,
        "decision_interval_sessions": int(interval),
        "decision_anchor": "evaluation_first_execution_session",
        "signal_lag_sessions": 1,
        "execution_timing": "next_trading_session",
        "non_decision_action": "hold_no_orders",
    }
    contract["evidence_sha256"] = canonical_sha256(contract)
    return contract


def validate_research_execution_cadence_contract(
    value: Mapping[str, Any],
    *,
    expected_horizon_profile: str | None = None,
) -> dict[str, Any]:
    """Rebuild and verify a self-contained cadence contract."""

    if not isinstance(value, Mapping):
        raise ValueError("research execution cadence must be an object")
    profile = str(value.get("horizon_profile") or "")
    if expected_horizon_profile is not None and profile != str(
        expected_horizon_profile
    ):
        raise ValueError("research execution cadence uses another horizon")
    expected = build_research_execution_cadence_contract(profile)
    if dict(value) != expected:
        raise ValueError("research execution cadence contract is invalid")
    return expected


def is_research_decision_step(
    trade_step: int,
    *,
    decision_interval_sessions: int,
) -> bool:
    """Return whether a zero-based evaluation step may rebalance."""

    if isinstance(trade_step, bool) or not isinstance(trade_step, int) or trade_step < 0:
        raise ValueError("research trade step must be a non-negative integer")
    if (
        isinstance(decision_interval_sessions, bool)
        or not isinstance(decision_interval_sessions, int)
        or decision_interval_sessions <= 0
    ):
        raise ValueError("research decision interval must be a positive integer")
    return trade_step % decision_interval_sessions == 0


__all__ = [
    "RESEARCH_EXECUTION_CADENCE_CONTRACT_VERSION",
    "build_research_execution_cadence_contract",
    "is_research_decision_step",
    "validate_research_execution_cadence_contract",
]
