"""Account-level security netting layer (design 6.10/8.1/9.2).

Fixed order (design 6.10): apply the frozen AllocationArtifact budgets once →
each member strategy emits security targets inside its budget → security-level
netting → account hard constraints → ExecutionPolicy. This module is the
netting step: it merges the per-member signed demands per instrument
algebraically (buys positive, sells negative), so opposite demands offset
internally and only the net is ever traded — no member pair may create real
opposite orders to manufacture turnover, and capacity/fees are computed once
on the net.

``strategy_contributions`` keeps both sides of the attribution (design 9.2):
each member's capital budget, its pre-netting signed target change, and its
post-netting contribution allocated by the frozen same-side pro-rata rule
(同向净需求比例分配; a fully offset demand contributes zero).

The plan is a *planning artifact*: the execution/simulation chain is not
rewired here, but the output shape (net target weights, signed net trades,
cash remainder, execution-policy reference) is directly consumable by it.
The idempotency key follows the design 9.2 stable-key semantics —
``account_id + allocation_artifact_id (final target version) + decision_date
+ inputs_as_of + policy_version + tranche_index`` — ``strategy_id`` is never
part of the key, and a retry with identical inputs replays the stored plan
instead of creating a second one.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import insert, select

from quant_data.database import (
    account_netting_plans,
    open_database,
    recommendation_holdings,
    recommendation_snapshots,
    strategy_allocation_artifacts,
    strategy_allocation_members,
    strategy_allocations,
)

NETTING_PLAN_VERSION = "account-netting-plan-v1"
DEFAULT_EXECUTION_POLICY = "next_bar"
EXECUTION_POLICIES = (DEFAULT_EXECUTION_POLICY, "twap", "vwap")

_TOLERANCE = 1e-9


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _now() -> datetime:
    return datetime.now(UTC)


def plan_idempotency_key(
    *,
    account_id: str,
    allocation_artifact_id: str,
    decision_date: date,
    inputs_as_of: date,
    policy_version: str,
    tranche_index: int,
) -> str:
    """Stable six-component key (design 9.2); strategy_id is never a component."""

    return _canonical_hash(
        {
            "account_id": str(account_id),
            "allocation_artifact_id": str(allocation_artifact_id),
            "decision_date": pd_date(decision_date),
            "inputs_as_of": pd_date(inputs_as_of),
            "policy_version": str(policy_version),
            "tranche_index": int(tranche_index),
        }
    )


def pd_date(value: date) -> str:
    day = value if isinstance(value, date) else date.fromisoformat(str(value))
    return day.isoformat()


def net_member_demands(
    member_demands: dict[str, dict[str, float]],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Algebraically merge signed per-member demands per instrument.

    Returns ``(net_deltas, contributions)``. Contributions preserve both the
    gross (pre-netting) demand and the post-netting contribution per member:
    the winning side shares the net pro-rata to its gross demand (the frozen
    same-side rule of design 9.2), the offset side contributes zero.
    """

    gross: dict[str, dict[str, float]] = {}
    for member, demands in member_demands.items():
        for instrument, value in demands.items():
            delta = float(value)
            if not delta:
                continue
            gross.setdefault(str(instrument), {})[str(member)] = (
                gross.setdefault(str(instrument), {}).get(str(member), 0.0) + delta
            )
    net_deltas: dict[str, float] = {}
    contributions: dict[str, Any] = {}
    for instrument in sorted(gross):
        members = gross[instrument]
        positive = sum(value for value in members.values() if value > 0)
        negative = sum(value for value in members.values() if value < 0)
        net = positive + negative
        per_member: dict[str, Any] = {}
        for member in sorted(members):
            gross_delta = members[member]
            if net > 0 and positive > 0 and gross_delta > 0:
                share = net * gross_delta / positive
            elif net < 0 and negative < 0 and gross_delta < 0:
                share = net * gross_delta / negative
            else:
                share = 0.0
            per_member[member] = {
                "gross_delta": gross_delta,
                "net_contribution": share,
            }
        if abs(net) > _TOLERANCE:
            net_deltas[instrument] = net
        contributions[instrument] = {
            "net_delta": net,
            "members": per_member,
        }
    return net_deltas, contributions


def build_account_netting_plan(
    *,
    account_id: str,
    allocation_artifact_id: str,
    decision_date: date,
    inputs_as_of: date,
    policy_version: str,
    member_budgets: dict[str, float],
    member_targets: dict[str, dict[str, float]],
    member_current_weights: dict[str, dict[str, float]] | None = None,
    total_capital: float = 1.0,
    execution_policy: str = DEFAULT_EXECUTION_POLICY,
    tranche_index: int = 0,
    max_instrument_weight: float | None = None,
) -> dict[str, Any]:
    """Build the account-level netted target plan (pure; no I/O).

    ``member_budgets`` are account-level capital budgets (weights of
    investable capital, summing to at most one — the remainder is cash).
    ``member_targets`` are long-only target weights inside each member's
    sleeve. ``member_current_weights`` optionally carries each member's
    current sleeve weights; without it every target is a fresh buy.
    ``max_instrument_weight`` (the account hard constraint applied after
    netting, e.g. ``PortfolioPolicyConfig.max_position_weight``) clamps net
    targets, the overflow moving to cash.
    """

    if execution_policy not in EXECUTION_POLICIES:
        raise ValueError(f"execution policy must be one of {EXECUTION_POLICIES}")
    if tranche_index < 0:
        raise ValueError("tranche index must be non-negative")
    if total_capital <= 0:
        raise ValueError("total capital must be positive")
    budgets = {str(member): float(weight) for member, weight in member_budgets.items()}
    if any(weight < -_TOLERANCE for weight in budgets.values()):
        raise ValueError("member budgets must be non-negative")
    if sum(budgets.values()) > 1.0 + _TOLERANCE:
        raise ValueError("member budgets exceed investable capital")
    if max_instrument_weight is not None and not 0 < max_instrument_weight <= 1:
        raise ValueError("max instrument weight must be in (0, 1]")

    currents = member_current_weights or {}
    demands: dict[str, dict[str, float]] = {}
    account_current: dict[str, float] = {}
    net_targets: dict[str, float] = {}
    for member, budget in budgets.items():
        targets = {
            str(key): float(value)
            for key, value in (member_targets.get(member) or {}).items()
        }
        if any(value < -_TOLERANCE for value in targets.values()):
            raise ValueError("long-only member targets must be non-negative")
        if sum(targets.values()) > 1.0 + 1e-6:
            raise ValueError(f"member {member} targets exceed the member sleeve")
        sleeve_current = {
            str(key): float(value) for key, value in (currents.get(member) or {}).items()
        }
        for instrument in sorted(set(targets) | set(sleeve_current)):
            target = targets.get(instrument, 0.0)
            current = sleeve_current.get(instrument, 0.0)
            net_targets[instrument] = net_targets.get(instrument, 0.0) + budget * target
            account_current[instrument] = (
                account_current.get(instrument, 0.0) + budget * current
            )
            delta = budget * (target - current)
            if abs(delta) > _TOLERANCE:
                demands.setdefault(member, {})[instrument] = delta

    net_deltas, contributions = net_member_demands(demands)

    clamps: dict[str, Any] = {}
    clamped_targets: dict[str, float] = {}
    for instrument in sorted(net_targets):
        weight = net_targets[instrument]
        if max_instrument_weight is not None and weight > max_instrument_weight:
            clamps[instrument] = {
                "raw_weight": weight,
                "clamped_weight": float(max_instrument_weight),
            }
            weight = float(max_instrument_weight)
        if weight > _TOLERANCE:
            clamped_targets[instrument] = weight

    net_trades: dict[str, float] = {}
    for instrument in sorted(set(clamped_targets) | set(account_current)):
        trade = clamped_targets.get(instrument, 0.0) - account_current.get(instrument, 0.0)
        if abs(trade) > _TOLERANCE:
            net_trades[instrument] = trade
    # Clamping re-scales the post-net attribution on the winning side so the
    # recorded contributions stay consistent with the constrained plan.
    for instrument in clamps:
        raw_delta = contributions.get(instrument, {}).get("net_delta", 0.0)
        if abs(raw_delta) > _TOLERANCE:
            factor = net_trades.get(instrument, 0.0) / raw_delta
            for member_entry in contributions[instrument]["members"].values():
                member_entry["net_contribution"] *= factor
            contributions[instrument]["net_delta"] = net_trades.get(instrument, 0.0)

    cash_weight = 1.0 - sum(clamped_targets.values())
    key = plan_idempotency_key(
        account_id=account_id,
        allocation_artifact_id=allocation_artifact_id,
        decision_date=decision_date,
        inputs_as_of=inputs_as_of,
        policy_version=policy_version,
        tranche_index=tranche_index,
    )
    plan: dict[str, Any] = {
        "plan_version": NETTING_PLAN_VERSION,
        "plan_key": key,
        "account_id": str(account_id),
        "allocation_artifact_id": str(allocation_artifact_id),
        "decision_date": pd_date(decision_date),
        "inputs_as_of": pd_date(inputs_as_of),
        "policy_version": str(policy_version),
        "execution_policy": execution_policy,
        "tranche_index": int(tranche_index),
        "total_capital": float(total_capital),
        "member_budgets": budgets,
        "net_targets": {
            instrument: {
                "weight": weight,
                "target_value": weight * float(total_capital),
            }
            for instrument, weight in clamped_targets.items()
        },
        "net_trades": {
            instrument: {
                "delta_weight": delta,
                "side": "buy" if delta > 0 else "sell",
                "trade_value": abs(delta) * float(total_capital),
            }
            for instrument, delta in net_trades.items()
        },
        "cash_weight": cash_weight,
        "strategy_contributions": contributions,
        "constraint_clamps": clamps,
        "max_instrument_weight": max_instrument_weight,
    }
    plan["plan_hash"] = _canonical_hash(
        {
            "plan_version": NETTING_PLAN_VERSION,
            "plan_key": key,
            "member_budgets": budgets,
            "net_targets": clamped_targets,
            "net_trades": net_trades,
            "cash_weight": cash_weight,
            "strategy_contributions": contributions,
            "execution_policy": execution_policy,
        }
    )
    return plan


class AccountNettingStore:
    """Persist and replay account netting plans (append-only, idempotent)."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def create_plan(self, *, actor: str, **plan_kwargs: Any) -> dict[str, Any]:
        if len(actor.strip()) < 2:
            raise ValueError("a responsible actor is required")
        plan = build_account_netting_plan(**plan_kwargs)
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(account_netting_plans).where(
                    account_netting_plans.c.plan_key == plan["plan_key"]
                )
            ).first()
            if existing is not None:
                if str(existing.plan_hash) != plan["plan_hash"]:
                    raise ValueError(
                        "account netting plan idempotency key conflict: identical key, "
                        "different content"
                    )
                replay = dict(existing.plan_json)
                replay["id"] = str(existing.id)
                replay["idempotent_replay"] = True
                return replay
            plan_id = uuid.uuid4().hex
            connection.execute(
                insert(account_netting_plans).values(
                    id=plan_id,
                    plan_key=plan["plan_key"],
                    account_id=plan["account_id"],
                    allocation_artifact_id=plan["allocation_artifact_id"],
                    decision_date=date.fromisoformat(plan["decision_date"]),
                    inputs_as_of=date.fromisoformat(plan["inputs_as_of"]),
                    policy_version=plan["policy_version"],
                    execution_policy=plan["execution_policy"],
                    tranche_index=plan["tranche_index"],
                    plan_hash=plan["plan_hash"],
                    plan_json=plan,
                    created_by=actor.strip(),
                    created_at=_now(),
                )
            )
        plan["id"] = plan_id
        plan["idempotent_replay"] = False
        return plan

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(account_netting_plans).where(account_netting_plans.c.id == plan_id)
            ).first()
        if row is None:
            raise KeyError(plan_id)
        plan = dict(row.plan_json)
        plan["id"] = str(row.id)
        return plan

    def build_plan_for_allocation(
        self,
        allocation_id: str,
        *,
        actor: str,
        execution_policy: str = DEFAULT_EXECUTION_POLICY,
        tranche_index: int = 0,
        member_current_weights: dict[str, dict[str, float]] | None = None,
        max_instrument_weight: float | None = None,
    ) -> dict[str, Any]:
        """Assemble plan inputs from the ledger and persist the netted plan.

        Budgets come from the latest valid AllocationArtifact (applied once);
        member targets come from each member portfolio's latest succeeded
        recommendation snapshot. A budgeted member without any succeeded
        snapshot fails closed — silently dropping its demand would fabricate
        the account target.
        """

        with self.engine.connect() as connection:
            allocation = connection.execute(
                select(strategy_allocations).where(
                    strategy_allocations.c.id == allocation_id
                )
            ).first()
            if allocation is None:
                raise KeyError(allocation_id)
            artifact = connection.execute(
                select(strategy_allocation_artifacts)
                .where(strategy_allocation_artifacts.c.allocation_id == allocation_id)
                .order_by(
                    strategy_allocation_artifacts.c.decision_date.desc(),
                    strategy_allocation_artifacts.c.created_at.desc(),
                )
                .limit(1)
            ).first()
            if artifact is None:
                raise ValueError("allocation has no AllocationArtifact to apply")
            budgets = {
                str(member): float(weight)
                for member, weight in (artifact.member_weights_json or {}).items()
            }
            members = connection.execute(
                select(strategy_allocation_members).where(
                    strategy_allocation_members.c.allocation_id == allocation_id
                )
            ).all()
            portfolio_by_member = {
                str(member.strategy_version_id): (
                    str(member.recommendation_portfolio_id)
                    if member.recommendation_portfolio_id
                    else None
                )
                for member in members
            }
            targets: dict[str, dict[str, float]] = {}
            missing: list[str] = []
            for version_id, budget in budgets.items():
                portfolio_id = portfolio_by_member.get(version_id)
                snapshot = None
                if portfolio_id:
                    snapshot = connection.execute(
                        select(recommendation_snapshots)
                        .where(
                            recommendation_snapshots.c.portfolio_id == portfolio_id,
                            recommendation_snapshots.c.status == "succeeded",
                        )
                        .order_by(recommendation_snapshots.c.created_at.desc())
                        .limit(1)
                    ).first()
                if snapshot is None:
                    if budget > _TOLERANCE:
                        missing.append(version_id)
                    targets[version_id] = {}
                    continue
                holdings = connection.execute(
                    select(
                        recommendation_holdings.c.instrument,
                        recommendation_holdings.c.weight,
                    ).where(recommendation_holdings.c.snapshot_id == snapshot.id)
                ).all()
                targets[version_id] = {
                    str(row.instrument): float(row.weight) for row in holdings
                }
            if missing:
                raise ValueError(
                    "budgeted allocation members have no succeeded recommendation "
                    f"snapshot: {sorted(missing)}"
                )
        return self.create_plan(
            actor=actor,
            account_id=str(allocation.id),
            allocation_artifact_id=str(artifact.id),
            decision_date=artifact.decision_date,
            inputs_as_of=artifact.inputs_as_of,
            policy_version=(
                f"allocation:{allocation.allocation_method}/{allocation.decision_frequency}"
            ),
            member_budgets=budgets,
            member_targets=targets,
            member_current_weights=member_current_weights,
            total_capital=float(allocation.total_capital),
            execution_policy=execution_policy,
            tranche_index=tranche_index,
            max_instrument_weight=max_instrument_weight,
        )
