"""Immutable policy-state handoff between consecutive paper decisions.

The paper account is recalculated every trading day even when a swing or
long-horizon strategy is between scheduled rebalances.  This projection gives
the worker one sealed previous-snapshot shape shared with recommendation
refreshes, so cadence is derived from durable batches rather than process
memory.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date
from math import isfinite
from typing import Any

PAPER_POLICY_STATE_VERSION = "paper-policy-state-v1"
_MAX_POLICY_STATE_BYTES = 1_000_000


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("paper policy state must be finite canonical JSON") from exc
    if len(encoded) > _MAX_POLICY_STATE_BYTES:
        raise ValueError("paper policy state exceeds the governed size limit")
    return encoded


def seal_paper_policy_state(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize and hash the policy state emitted by ``PortfolioPolicy``."""

    state = dict(value or {})
    encoded = _canonical_bytes(state)
    # JSON round-tripping strips Mapping subclasses and prevents a caller from
    # smuggling mutable/custom values into the immutable batch payload.
    normalized = json.loads(encoded.decode("utf-8"))
    return {
        "contract_version": PAPER_POLICY_STATE_VERSION,
        "position_state": normalized,
        "position_state_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def validate_paper_policy_state(value: Mapping[str, Any]) -> dict[str, Any]:
    envelope = dict(value or {})
    if envelope.get("contract_version") != PAPER_POLICY_STATE_VERSION:
        raise ValueError("paper policy state contract version is unsupported")
    state = envelope.get("position_state")
    if not isinstance(state, Mapping):
        raise ValueError("paper policy state position_state must be an object")
    sealed = seal_paper_policy_state(state)
    if envelope != sealed:
        raise ValueError("paper policy state seal is invalid")
    return sealed


def previous_snapshot_from_paper_batch(
    batch: Mapping[str, Any],
    *,
    expected_portfolio_id: str,
    expected_promotion_stage_id: str,
) -> dict[str, Any]:
    """Project a succeeded governed batch into recommendation-manifest shape."""

    row = dict(batch)
    if str(row.get("status") or "") != "succeeded":
        raise ValueError("previous paper policy state requires a succeeded batch")
    if str(row.get("portfolio_id") or "") != str(expected_portfolio_id):
        raise ValueError("previous paper batch belongs to another account")
    payload = row.get("target_payload")
    if payload is None:
        payload = row.get("target_payload_json")
    if not isinstance(payload, Mapping):
        raise ValueError("previous paper batch target payload is missing")
    plan = payload.get("governed_order_plan")
    if not isinstance(plan, Mapping) or str(plan.get("promotion_stage_id") or "") != str(
        expected_promotion_stage_id
    ):
        raise ValueError("previous paper batch is not bound to the active stage")
    state = payload.get("paper_policy_state")
    if not isinstance(state, Mapping):
        raise ValueError("previous paper batch has no sealed policy state")
    validated = validate_paper_policy_state(state)
    signal_date = _iso_date(row.get("signal_date"), field="signal_date")
    trade_date = _iso_date(row.get("trade_date"), field="trade_date")
    if trade_date <= signal_date:
        raise ValueError("previous daily paper batch violates next-session execution")
    return {
        "as_of_date": signal_date.isoformat(),
        "effective_date": trade_date.isoformat(),
        "position_state": validated["position_state"],
        "position_state_sha256": validated["position_state_sha256"],
        "paper_policy_state_contract_version": PAPER_POLICY_STATE_VERSION,
        "paper_batch_id": str(row.get("id") or ""),
    }


def bind_current_paper_holdings(
    snapshot: Mapping[str, Any] | None,
    holdings: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Bind prior policy memory to the current reconciled simulation ledger.

    Policy memory describes staged exits and execution progress; position
    quantity, cost basis and trading-session age belong to the actual paper
    ledger.  The latter therefore replaces stale age entries, while state for
    instruments no longer held is removed.  Duplicate or unproven holdings
    fail closed instead of producing an ambiguous rebalance input.
    """

    if snapshot is None:
        return None
    result = dict(snapshot)
    raw_state = result.get("position_state")
    if not isinstance(raw_state, Mapping):
        raise ValueError("previous paper snapshot has no position state")
    state = json.loads(_canonical_bytes(dict(raw_state)).decode("utf-8"))
    normalized: list[dict[str, Any]] = []
    instruments: set[str] = set()
    durable_ages: dict[str, int] = {}
    for raw in holdings:
        item = dict(raw)
        instrument = str(item.get("instrument") or "").strip().upper()
        if not instrument or instrument in instruments:
            raise ValueError("current paper holdings require unique instruments")
        weight = float(item.get("weight") or 0.0)
        average_cost = float(item.get("average_cost") or 0.0)
        age = item.get("holding_age_sessions")
        if (
            not isfinite(weight)
            or weight <= 0
            or not isfinite(average_cost)
            or average_cost <= 0
            or isinstance(age, bool)
            or not isinstance(age, int)
            or age < 0
        ):
            raise ValueError(
                "current paper holdings require positive finite weights/costs "
                "and proven non-negative session ages"
            )
        instruments.add(instrument)
        durable_ages[instrument] = age
        normalized.append(
            {
                **item,
                "instrument": instrument,
                "weight": weight,
                "average_cost": average_cost,
                "holding_age_sessions": age,
            }
        )
    stages = state.get("take_profit_stages") or {}
    if not isinstance(stages, Mapping):
        raise ValueError("previous paper take-profit state must be an object")
    normalized_stages: dict[str, int] = {}
    for raw_instrument, raw_stage in stages.items():
        instrument = str(raw_instrument).strip().upper()
        if instrument not in instruments:
            continue
        if isinstance(raw_stage, bool):
            raise ValueError("previous paper take-profit stages must be integers")
        try:
            stage = int(raw_stage)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "previous paper take-profit stages must be integers"
            ) from exc
        if stage < 0:
            raise ValueError("previous paper take-profit stages cannot be negative")
        normalized_stages[instrument] = stage
    state["take_profit_stages"] = normalized_stages
    state["holding_age_sessions"] = durable_ages
    prior_state_sha256 = result.get("position_state_sha256")
    reconciled_state_bytes = _canonical_bytes(state)
    result["position_state"] = state
    result["prior_position_state_sha256"] = prior_state_sha256
    result["position_state_sha256"] = hashlib.sha256(
        reconciled_state_bytes
    ).hexdigest()
    result["holdings"] = normalized
    result["current_holdings_sha256"] = hashlib.sha256(
        _canonical_bytes(normalized)
    ).hexdigest()
    result["holding_age_source"] = "simulation_position_lots_qlib_calendar"
    return result


def _iso_date(value: Any, *, field: str) -> date:
    if isinstance(value, date):
        return value
    try:
        parsed = date.fromisoformat(str(value or ""))
    except ValueError as exc:
        raise ValueError(f"previous paper batch {field} is invalid") from exc
    return parsed
