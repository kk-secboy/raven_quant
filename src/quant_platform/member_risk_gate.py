from __future__ import annotations

from typing import Any

from sqlalchemy import select

from quant_data.database import (
    recommendation_portfolios,
    strategy_allocation_events,
    strategy_allocation_members,
    strategy_allocations,
)

MEMBER_DRAWDOWN_RULE = "max_member_drawdown"
PAUSE_NEW_RISK_STATE = "pause_new_risk"
ALLOCATION_REDUCTION_RULE = "max_drawdown_reduce"
ALLOCATION_LIQUIDATION_RULE = "max_drawdown_liquidate"
RISK_REDUCTION_STATE = "risk_reduction"
LIQUIDATION_STATE = "liquidation"
ACTIVE_STATE = "active"
PAUSED_STATE = "paused"

OPEN_EVENT_STATUSES = ("open", "acknowledged")
ALLOCATION_RULE_STATES = {
    ALLOCATION_REDUCTION_RULE: (RISK_REDUCTION_STATE, 0.5),
    ALLOCATION_LIQUIDATION_RULE: (LIQUIDATION_STATE, 0.0),
}
ALLOCATION_STATUS_STATES = {
    "risk_reduction_pending": (RISK_REDUCTION_STATE, 0.5),
    "liquidation_pending": (LIQUIDATION_STATE, 0.0),
    "paused": (PAUSED_STATE, 0.0),
}


def compose_strategy_risk_state(
    strategy_version_id: str,
    *,
    member_event_ids: list[int] | None = None,
    member_allocation_ids: list[str] | None = None,
    allocation_gates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Merge member and allocation gates using the most restrictive exposure."""

    normalized_id = strategy_version_id.strip()
    if not normalized_id:
        raise ValueError("strategy version id is required")
    member_ids = sorted({int(item) for item in (member_event_ids or [])})
    gates = [dict(item) for item in (allocation_gates or [])]
    override = min(
        [1.0]
        + [
            float(item.get("risk_exposure_override", 1.0))
            for item in gates
        ]
    )
    if not 0.0 <= override <= 1.0:
        raise ValueError("risk exposure override must be between zero and one")
    allocation_states = {str(item.get("state") or ACTIVE_STATE) for item in gates}
    if LIQUIDATION_STATE in allocation_states:
        allocation_state = LIQUIDATION_STATE
    elif PAUSED_STATE in allocation_states:
        allocation_state = PAUSED_STATE
    elif RISK_REDUCTION_STATE in allocation_states:
        allocation_state = RISK_REDUCTION_STATE
    else:
        allocation_state = ACTIVE_STATE
    member_state = PAUSE_NEW_RISK_STATE if member_ids else ACTIVE_STATE
    if allocation_state != ACTIVE_STATE:
        state = allocation_state
    else:
        state = member_state
    allocation_event_ids = sorted(
        {
            int(event_id)
            for item in gates
            for event_id in item.get("event_ids", [])
        }
    )
    allocation_ids = sorted(
        {
            *(str(item) for item in (member_allocation_ids or [])),
            *(
                str(item["allocation_id"])
                for item in gates
                if item.get("allocation_id")
            ),
        }
    )
    reactivation_ids = sorted(
        {
            str(item["allocation_id"])
            for item in gates
            if item.get("requires_reactivation") and item.get("allocation_id")
        }
    )
    return {
        "strategy_version_id": normalized_id,
        "state": state,
        "member_risk_state": member_state,
        "allocation_risk_state": allocation_state,
        "allow_new_risk": not member_ids and override >= 1.0,
        "risk_exposure_override": override,
        "event_ids": sorted({*member_ids, *allocation_event_ids}),
        "member_event_ids": member_ids,
        "allocation_event_ids": allocation_event_ids,
        "allocation_ids": allocation_ids,
        "allocation_gates": gates,
        "recovery": {
            "member_events_must_be_resolved": member_ids,
            "allocation_events_must_be_resolved": allocation_event_ids,
            "allocation_ids_requiring_reactivation": reactivation_ids,
            "member_gate_reopens_on_resolution": bool(member_ids),
            "allocation_gate_requires_explicit_active_state": bool(reactivation_ids),
        },
    }


def event_matches_strategy_version(
    connection: Any,
    event: Any,
    *,
    strategy_version_id: str,
    recommendation_portfolio_id: str | None = None,
) -> bool:
    details = dict(event.details_json or {})
    if str(details.get("strategy_version_id") or "") == strategy_version_id:
        return True
    event_portfolio_id = (
        str(event.recommendation_portfolio_id)
        if event.recommendation_portfolio_id
        else None
    )
    if recommendation_portfolio_id and event_portfolio_id == recommendation_portfolio_id:
        return True
    if not event_portfolio_id:
        return False
    event_version_id = connection.scalar(
        select(recommendation_portfolios.c.strategy_version_id).where(
            recommendation_portfolios.c.id == event_portfolio_id
        )
    )
    return str(event_version_id or "") == strategy_version_id


def has_open_member_gate(
    connection: Any,
    *,
    allocation_id: str,
    strategy_version_id: str,
    recommendation_portfolio_id: str | None,
) -> bool:
    events = connection.execute(
        select(strategy_allocation_events).where(
            strategy_allocation_events.c.allocation_id == allocation_id,
            strategy_allocation_events.c.rule == MEMBER_DRAWDOWN_RULE,
            strategy_allocation_events.c.status.in_(OPEN_EVENT_STATUSES),
        )
    ).all()
    return any(
        event_matches_strategy_version(
            connection,
            event,
            strategy_version_id=strategy_version_id,
            recommendation_portfolio_id=recommendation_portfolio_id,
        )
        for event in events
    )


def has_open_allocation_gate(
    connection: Any,
    *,
    allocation_id: str,
    rule: str,
) -> bool:
    if rule not in ALLOCATION_RULE_STATES:
        raise ValueError(f"unsupported allocation risk rule: {rule}")
    return (
        connection.execute(
            select(strategy_allocation_events.c.id)
            .where(
                strategy_allocation_events.c.allocation_id == allocation_id,
                strategy_allocation_events.c.rule == rule,
                strategy_allocation_events.c.status.in_(OPEN_EVENT_STATUSES),
            )
            .limit(1)
        ).first()
        is not None
    )


def load_allocation_risk_state(
    connection: Any,
    allocation_id: str,
) -> dict[str, Any]:
    normalized_id = allocation_id.strip()
    if not normalized_id:
        raise ValueError("allocation id is required")
    allocation = connection.execute(
        select(
            strategy_allocations.c.id,
            strategy_allocations.c.status,
        ).where(strategy_allocations.c.id == normalized_id)
    ).first()
    if allocation is None:
        raise KeyError(normalized_id)
    events = connection.execute(
        select(
            strategy_allocation_events.c.id,
            strategy_allocation_events.c.rule,
        ).where(
            strategy_allocation_events.c.allocation_id == normalized_id,
            strategy_allocation_events.c.rule.in_(tuple(ALLOCATION_RULE_STATES)),
            strategy_allocation_events.c.status.in_(OPEN_EVENT_STATUSES),
        )
    ).all()
    candidates: list[tuple[str, float]] = []
    status_state = ALLOCATION_STATUS_STATES.get(str(allocation.status))
    if status_state:
        candidates.append(status_state)
    candidates.extend(
        ALLOCATION_RULE_STATES[str(event.rule)]
        for event in events
        if str(event.rule) in ALLOCATION_RULE_STATES
    )
    override = min([1.0] + [item[1] for item in candidates])
    if override <= 0.0:
        state = (
            LIQUIDATION_STATE
            if any(item[0] == LIQUIDATION_STATE for item in candidates)
            else PAUSED_STATE
        )
    elif override <= 0.5:
        state = RISK_REDUCTION_STATE
    else:
        state = ACTIVE_STATE
    return {
        "allocation_id": normalized_id,
        "allocation_status": str(allocation.status),
        "state": state,
        "risk_exposure_override": override,
        "event_ids": sorted(int(item.id) for item in events),
        "requires_reactivation": str(allocation.status) in ALLOCATION_STATUS_STATES,
    }


def load_strategy_risk_state(
    connection: Any,
    strategy_version_id: str,
) -> dict[str, Any]:
    normalized_id = strategy_version_id.strip()
    if not normalized_id:
        raise ValueError("strategy version id is required")
    matching = _matching_member_events(connection, normalized_id)
    allocation_ids = list(
        connection.scalars(
            select(strategy_allocation_members.c.allocation_id).where(
                strategy_allocation_members.c.strategy_version_id == normalized_id
            )
        )
    )
    gates = [
        load_allocation_risk_state(connection, str(allocation_id))
        for allocation_id in allocation_ids
    ]
    return compose_strategy_risk_state(
        normalized_id,
        member_event_ids=[int(item.id) for item in matching],
        member_allocation_ids=[str(item.allocation_id) for item in matching],
        allocation_gates=gates,
    )


def load_member_risk_state(connection: Any, strategy_version_id: str) -> dict[str, Any]:
    """Return the original 8% member-only new-risk gate."""

    normalized_id = strategy_version_id.strip()
    if not normalized_id:
        raise ValueError("strategy version id is required")
    matching = _matching_member_events(connection, normalized_id)
    paused = bool(matching)
    return {
        "strategy_version_id": normalized_id,
        "state": PAUSE_NEW_RISK_STATE if paused else ACTIVE_STATE,
        "allow_new_risk": not paused,
        "event_ids": [int(item.id) for item in matching],
        "allocation_ids": sorted({str(item.allocation_id) for item in matching}),
    }


def _matching_member_events(
    connection: Any,
    strategy_version_id: str,
) -> list[Any]:
    member_events = connection.execute(
        select(strategy_allocation_events).where(
            strategy_allocation_events.c.rule == MEMBER_DRAWDOWN_RULE,
            strategy_allocation_events.c.status.in_(OPEN_EVENT_STATUSES),
        )
    ).all()
    return [
        event
        for event in member_events
        if event_matches_strategy_version(
            connection,
            event,
            strategy_version_id=strategy_version_id,
        )
    ]
