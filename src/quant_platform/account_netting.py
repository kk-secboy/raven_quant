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
from math import isfinite
from typing import Any

from sqlalchemy import insert, select

from quant_data.database import (
    account_netting_plans,
    open_database,
    recommendation_holdings,
    recommendation_snapshots,
    simulation_batches,
    strategy_allocation_artifacts,
    strategy_allocation_members,
    strategy_allocations,
    strategy_health_snapshots,
    strategy_versions,
)

from .research_horizon import LONG_1_3Y, SHORT_1_5D, SWING_1_6M
from .strategy_health import cap_targets_for_health

NETTING_PLAN_VERSION = "account-netting-plan-v5-primary-ledger-capital"
DEFAULT_EXECUTION_POLICY = "open"
EXECUTION_POLICIES = (DEFAULT_EXECUTION_POLICY, "next_bar", "twap", "vwap")

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
            if not isfinite(delta):
                raise ValueError("member demands must contain only finite values")
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
    industry_memberships: dict[str, str] | None = None,
    max_industry_weight: float | None = None,
    input_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the account-level netted target plan (pure; no I/O).

    ``member_budgets`` are account-level capital budgets (weights of
    investable capital, summing to at most one — the remainder is cash).
    ``member_targets`` are long-only target weights inside each member's
    sleeve. ``member_current_weights`` optionally carries each member's
    current sleeve weights; without it every target is a fresh buy.
    ``max_instrument_weight`` (the account hard constraint applied after
    netting, e.g. ``PortfolioPolicyConfig.max_position_weight``) clamps net
    targets, the overflow moving to cash.  ``max_industry_weight`` applies to
    the unified account after member netting. Missing industry metadata fails
    closed for the affected target rather than silently bypassing the cap.
    """

    if execution_policy not in EXECUTION_POLICIES:
        raise ValueError(f"execution policy must be one of {EXECUTION_POLICIES}")
    if tranche_index < 0:
        raise ValueError("tranche index must be non-negative")
    capital = float(total_capital)
    if not isfinite(capital) or capital <= 0:
        raise ValueError("total capital must be positive")
    budgets = {str(member): float(weight) for member, weight in member_budgets.items()}
    if len(budgets) != len(member_budgets):
        raise ValueError("member budget identifiers must be unique after normalization")
    if any(not isfinite(weight) or weight < -_TOLERANCE for weight in budgets.values()):
        raise ValueError("member budgets must be finite and non-negative")
    if sum(budgets.values()) > 1.0 + _TOLERANCE:
        raise ValueError("member budgets exceed investable capital")
    normalized_cap = (
        None if max_instrument_weight is None else float(max_instrument_weight)
    )
    if normalized_cap is not None and (
        not isfinite(normalized_cap) or not 0 < normalized_cap <= 1
    ):
        raise ValueError("max instrument weight must be finite and in (0, 1]")
    normalized_industry_cap = (
        None if max_industry_weight is None else float(max_industry_weight)
    )
    if normalized_industry_cap is not None and (
        not isfinite(normalized_industry_cap)
        or not 0 < normalized_industry_cap <= 1
    ):
        raise ValueError("max industry weight must be finite and in (0, 1]")
    industries = {
        str(instrument): str(industry).strip()
        for instrument, industry in (industry_memberships or {}).items()
        if str(industry).strip()
    }
    target_books = {str(member): values for member, values in member_targets.items()}
    current_books = {
        str(member): values for member, values in (member_current_weights or {}).items()
    }
    if len(target_books) != len(member_targets) or len(current_books) != len(
        member_current_weights or {}
    ):
        raise ValueError("member identifiers must be unique after normalization")
    unknown_targets = set(target_books).difference(budgets)
    unknown_currents = set(current_books).difference(budgets)
    if unknown_targets or unknown_currents:
        raise ValueError("member targets and current weights require a matching budget")

    demands: dict[str, dict[str, float]] = {}
    account_current: dict[str, float] = {}
    net_targets: dict[str, float] = {}
    normalized_member_targets: dict[str, dict[str, float]] = {}
    normalized_member_current_weights: dict[str, dict[str, float]] = {}
    for member, budget in budgets.items():
        raw_targets = target_books.get(member) or {}
        targets = {
            str(key): float(value)
            for key, value in raw_targets.items()
        }
        if len(targets) != len(raw_targets):
            raise ValueError("target instruments must be unique after normalization")
        if any(
            not isfinite(value) or value < -_TOLERANCE for value in targets.values()
        ):
            raise ValueError("long-only member targets must be finite and non-negative")
        if sum(targets.values()) > 1.0 + 1e-6:
            raise ValueError(f"member {member} targets exceed the member sleeve")
        normalized_member_targets[member] = targets
        raw_current = current_books.get(member) or {}
        sleeve_current = {
            str(key): float(value)
            for key, value in raw_current.items()
        }
        if len(sleeve_current) != len(raw_current):
            raise ValueError("current instruments must be unique after normalization")
        if any(
            not isfinite(value) or value < -_TOLERANCE
            for value in sleeve_current.values()
        ):
            raise ValueError(
                "long-only member current weights must be finite and non-negative"
            )
        if sum(sleeve_current.values()) > 1.0 + 1e-6:
            raise ValueError(f"member {member} current weights exceed the member sleeve")
        normalized_member_current_weights[member] = sleeve_current
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
        if normalized_cap is not None and weight > normalized_cap:
            clamps[instrument] = {
                "raw_weight": weight,
                "clamped_weight": normalized_cap,
            }
            weight = normalized_cap
        if weight > _TOLERANCE:
            clamped_targets[instrument] = weight

    industry_clamps: dict[str, Any] = {}
    industry_exposure: dict[str, float] = {}
    if normalized_industry_cap is not None:
        missing_industry = [
            instrument for instrument in clamped_targets if instrument not in industries
        ]
        for instrument in missing_industry:
            raw_weight = clamped_targets.pop(instrument)
            industry_clamps[instrument] = {
                "industry": None,
                "raw_weight": raw_weight,
                "clamped_weight": 0.0,
                "reason": "missing_industry_membership_blocks_target",
            }
        industry_groups: dict[str, list[str]] = {}
        for instrument in clamped_targets:
            industry_groups.setdefault(industries[instrument], []).append(instrument)
        for industry, instruments in sorted(industry_groups.items()):
            raw_exposure = sum(clamped_targets[item] for item in instruments)
            if raw_exposure > normalized_industry_cap + _TOLERANCE:
                scale = normalized_industry_cap / raw_exposure
                for instrument in instruments:
                    raw_weight = clamped_targets[instrument]
                    clamped_targets[instrument] = raw_weight * scale
                    industry_clamps[instrument] = {
                        "industry": industry,
                        "raw_weight": raw_weight,
                        "clamped_weight": clamped_targets[instrument],
                        "reason": "account_industry_weight_cap",
                    }
            industry_exposure[industry] = sum(
                clamped_targets[item] for item in instruments
            )

    net_trades: dict[str, float] = {}
    for instrument in sorted(set(clamped_targets) | set(account_current)):
        trade = clamped_targets.get(instrument, 0.0) - account_current.get(instrument, 0.0)
        if abs(trade) > _TOLERANCE:
            net_trades[instrument] = trade
    # Account constraints re-scale the post-net attribution on the winning
    # side so every instrument's virtual contributions still reconcile exactly
    # to the unified account trade. A sign-changing constraint adjustment is
    # explicit instead of being silently attributed to an unrelated sleeve.
    for instrument in set(clamps) | set(industry_clamps):
        raw_delta = contributions.get(instrument, {}).get("net_delta", 0.0)
        constrained_delta = net_trades.get(instrument, 0.0)
        member_entries = contributions.setdefault(
            instrument, {"net_delta": 0.0, "members": {}}
        )["members"]
        if (
            abs(raw_delta) > _TOLERANCE
            and raw_delta * constrained_delta >= 0
        ):
            factor = constrained_delta / raw_delta
            for member_entry in member_entries.values():
                member_entry["net_contribution"] *= factor
        else:
            for member_entry in member_entries.values():
                member_entry["net_contribution"] = 0.0
            if abs(constrained_delta) > _TOLERANCE:
                member_entries["__account_constraint__"] = {
                    "gross_delta": 0.0,
                    "net_contribution": constrained_delta,
                }
        contributions[instrument]["net_delta"] = constrained_delta

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
        "total_capital": capital,
        "member_budgets": budgets,
        "member_targets": normalized_member_targets,
        "member_current_weights": normalized_member_current_weights,
        "net_targets": {
            instrument: {
                "weight": weight,
                "target_value": weight * capital,
            }
            for instrument, weight in clamped_targets.items()
        },
        "net_trades": {
            instrument: {
                "delta_weight": delta,
                "side": "buy" if delta > 0 else "sell",
                "trade_value": abs(delta) * capital,
            }
            for instrument, delta in net_trades.items()
        },
        "cash_weight": cash_weight,
        "strategy_contributions": contributions,
        "constraint_clamps": clamps,
        "max_instrument_weight": normalized_cap,
        "industry_memberships": {
            instrument: industries[instrument]
            for instrument in sorted(clamped_targets)
            if instrument in industries
        },
        "industry_exposure": industry_exposure,
        "industry_constraint_clamps": industry_clamps,
        "max_industry_weight": normalized_industry_cap,
        "input_evidence": dict(input_evidence or {}),
    }
    plan["plan_hash"] = _canonical_hash(
        {
            "plan_version": NETTING_PLAN_VERSION,
            "plan_key": key,
            "member_budgets": budgets,
            "member_targets": normalized_member_targets,
            "member_current_weights": normalized_member_current_weights,
            "total_capital": capital,
            "net_targets": clamped_targets,
            "net_trades": net_trades,
            "cash_weight": cash_weight,
            "strategy_contributions": contributions,
            "constraint_clamps": clamps,
            "industry_memberships": industries,
            "industry_exposure": industry_exposure,
            "industry_constraint_clamps": industry_clamps,
            "max_instrument_weight": normalized_cap,
            "max_industry_weight": normalized_industry_cap,
            "execution_policy": execution_policy,
            "input_evidence": dict(input_evidence or {}),
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
        max_gross_exposure: float = 1.0,
        max_industry_weight: float | None = None,
        primary_account: dict[str, Any] | None = None,
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
            horizons = {
                str(row.id): str(row.horizon_profile or "legacy_ambiguous")
                for row in connection.execute(
                    select(
                        strategy_versions.c.id,
                        strategy_versions.c.horizon_profile,
                    ).where(strategy_versions.c.id.in_(tuple(budgets)))
                )
            }
            gross_limit = float(max_gross_exposure)
            if not isfinite(gross_limit) or not 0 < gross_limit <= 1:
                raise ValueError("maximum gross exposure must be finite and in (0, 1]")
            budget_mass = sum(budgets.values())
            if budget_mass > gross_limit + _TOLERANCE:
                scale = gross_limit / budget_mass
                budgets = {member: weight * scale for member, weight in budgets.items()}
            targets: dict[str, dict[str, float]] = {}
            snapshot_evidence: dict[str, dict[str, Any]] = {}
            industry_observations: dict[str, tuple[date, str]] = {}
            missing: list[str] = []
            for version_id, budget in budgets.items():
                portfolio_id = portfolio_by_member.get(version_id)
                member_snapshots: list[Any] = []
                if portfolio_id:
                    member_snapshots = list(
                        connection.execute(
                            select(recommendation_snapshots)
                            .where(
                                recommendation_snapshots.c.portfolio_id == portfolio_id,
                                recommendation_snapshots.c.status == "succeeded",
                            )
                            .order_by(recommendation_snapshots.c.created_at.desc())
                            .limit(2)
                        ).all()
                    )
                snapshot = member_snapshots[0] if member_snapshots else None
                if snapshot is None:
                    if budget > _TOLERANCE:
                        missing.append(version_id)
                    targets[version_id] = {}
                    continue
                snapshot_evidence[version_id] = {
                    "snapshot_id": str(snapshot.id),
                    "as_of_date": snapshot.as_of_date.isoformat(),
                    "effective_date": (
                        snapshot.effective_date.isoformat()
                        if snapshot.effective_date is not None
                        else snapshot.as_of_date.isoformat()
                    ),
                    "dataset_identity_sha256": str(snapshot.dataset_identity_sha256),
                }
                holdings = connection.execute(
                    select(
                        recommendation_holdings.c.instrument,
                        recommendation_holdings.c.weight,
                    ).where(recommendation_holdings.c.snapshot_id == snapshot.id)
                ).all()
                targets[version_id] = {
                    str(row.instrument): float(row.weight) for row in holdings
                }
                payload = dict(snapshot.snapshot_json or {})
                snapshot_industries = {
                    str(instrument): str(industry)
                    for instrument, industry in (
                        payload.get("industry_memberships") or {}
                    ).items()
                    if str(industry).strip()
                }
                for item in payload.get("holdings") or []:
                    if isinstance(item, dict) and item.get("industry"):
                        snapshot_industries.setdefault(
                            str(item.get("instrument") or ""),
                            str(item["industry"]),
                        )
                for instrument, industry in snapshot_industries.items():
                    observed = industry_observations.get(instrument)
                    if observed is None or snapshot.as_of_date >= observed[0]:
                        industry_observations[instrument] = (
                            snapshot.as_of_date,
                            industry,
                        )

                if horizons.get(version_id) in {
                    SHORT_1_5D,
                    SWING_1_6M,
                    LONG_1_3Y,
                }:
                    health = connection.execute(
                        select(strategy_health_snapshots)
                        .where(
                            strategy_health_snapshots.c.strategy_version_id == version_id
                        )
                        .order_by(
                            strategy_health_snapshots.c.as_of.desc(),
                            strategy_health_snapshots.c.recorded_at.desc(),
                        )
                        .limit(1)
                    ).first()
                    health_status = str(health.health_status) if health is not None else None
                    previous_targets: dict[str, float] = {}
                    if len(member_snapshots) > 1:
                        previous_holdings = connection.execute(
                            select(
                                recommendation_holdings.c.instrument,
                                recommendation_holdings.c.weight,
                            ).where(
                                recommendation_holdings.c.snapshot_id
                                == member_snapshots[1].id
                            )
                        ).all()
                        previous_targets = {
                            str(row.instrument): float(row.weight)
                            for row in previous_holdings
                        }
                    targets[version_id], health_gate = cap_targets_for_health(
                        targets[version_id],
                        previous_targets,
                        health_status,
                    )
                    health_gate["snapshot_id"] = (
                        str(health.id) if health is not None else None
                    )
                    snapshot_evidence[version_id]["strategy_health_gate"] = health_gate
            if missing:
                raise ValueError(
                    "budgeted allocation members have no succeeded recommendation "
                    f"snapshot: {sorted(missing)}"
                )
            decision_date = max(
                date.fromisoformat(value["effective_date"])
                for value in snapshot_evidence.values()
            )
            inputs_as_of = max(
                date.fromisoformat(value["as_of_date"])
                for value in snapshot_evidence.values()
            )
            if decision_date > artifact.valid_until:
                raise ValueError("allocation artifact expired before the latest member target")
            continuity_evidence: dict[str, Any] | None = None
            if member_current_weights is None and primary_account is not None:
                primary_id = str(primary_account.get("portfolio_id") or "").strip()
                if not primary_id:
                    raise ValueError("primary account capital evidence requires a portfolio id")
                prior = connection.execute(
                    select(account_netting_plans, simulation_batches.c.id.label("batch_id"))
                    .select_from(
                        simulation_batches.join(
                            account_netting_plans,
                            account_netting_plans.c.id
                            == simulation_batches.c.account_netting_plan_id,
                        )
                    )
                    .where(
                        simulation_batches.c.portfolio_id == primary_id,
                        simulation_batches.c.status == "succeeded",
                        account_netting_plans.c.inputs_as_of <= inputs_as_of,
                    )
                    .order_by(
                        simulation_batches.c.trade_date.desc(),
                        simulation_batches.c.created_at.desc(),
                    )
                    .limit(1)
                ).first()
                if prior is not None:
                    prior_targets = dict(
                        dict(prior.plan_json or {}).get("member_targets") or {}
                    )
                    if any(not isinstance(value, dict) for value in prior_targets.values()):
                        raise ValueError(
                            "prior primary-account plan has invalid durable member targets"
                        )
                    member_current_weights = {
                        str(member): {
                            str(instrument): float(weight)
                            for instrument, weight in dict(prior_targets.get(member) or {}).items()
                        }
                        for member in budgets
                    }
                    continuity_evidence = {
                        "portfolio_id": primary_id,
                        "prior_plan_id": str(prior.id),
                        "prior_batch_id": str(prior.batch_id),
                        "prior_allocation_id": str(prior.account_id),
                        "carried_members": sorted(set(prior_targets).intersection(budgets)),
                    }
            if member_current_weights is None:
                prior = connection.execute(
                    select(account_netting_plans)
                    .where(
                        account_netting_plans.c.account_id == str(allocation.id),
                        account_netting_plans.c.inputs_as_of < inputs_as_of,
                    )
                    .order_by(
                        account_netting_plans.c.inputs_as_of.desc(),
                        account_netting_plans.c.created_at.desc(),
                    )
                    .limit(1)
                ).first()
                if prior is not None:
                    prior_targets = dict(
                        dict(prior.plan_json or {}).get("member_targets") or {}
                    )
                    if set(prior_targets) != set(budgets) or any(
                        not isinstance(value, dict)
                        for value in prior_targets.values()
                    ):
                        raise ValueError(
                            "prior account plan has no complete durable member targets"
                        )
                    member_current_weights = {
                        str(member): {
                            str(instrument): float(weight)
                            for instrument, weight in values.items()
                        }
                        for member, values in prior_targets.items()
                    }
        account_capital = float(allocation.total_capital)
        primary_evidence: dict[str, Any] | None = None
        if primary_account is not None:
            primary_id = str(primary_account.get("portfolio_id") or "").strip()
            primary_source = str(primary_account.get("source_id") or "").strip()
            account_capital = float(primary_account.get("nav") or 0.0)
            if (
                not primary_id
                or primary_source != str(allocation.id)
                or not isfinite(account_capital)
                or account_capital <= 0
            ):
                raise ValueError(
                    "primary account capital evidence does not match the active allocation"
                )
            primary_evidence = {
                "portfolio_id": primary_id,
                "source_id": primary_source,
                "nav": account_capital,
                "updated_at": str(primary_account.get("updated_at") or ""),
            }
        capital_identity = _canonical_hash(
            primary_evidence
            or {
                "allocation_id": str(allocation.id),
                "total_capital": account_capital,
            }
        )[:16]
        return self.create_plan(
            actor=actor,
            account_id=str(allocation.id),
            allocation_artifact_id=str(artifact.id),
            decision_date=decision_date,
            inputs_as_of=inputs_as_of,
            policy_version=(
                f"allocation:{allocation.allocation_method}/{allocation.decision_frequency}:"
                f"gross<={gross_limit:.6f}:"
                f"instrument<={float(max_instrument_weight or 1.0):.6f}:"
                f"industry<={float(max_industry_weight or 1.0):.6f}:"
                f"capital={capital_identity}:"
                f"{NETTING_PLAN_VERSION}"
            ),
            member_budgets=budgets,
            member_targets=targets,
            member_current_weights=member_current_weights,
            total_capital=account_capital,
            execution_policy=execution_policy,
            tranche_index=tranche_index,
            max_instrument_weight=max_instrument_weight,
            industry_memberships={
                instrument: value[1]
                for instrument, value in industry_observations.items()
            },
            max_industry_weight=max_industry_weight,
            input_evidence={
                "member_snapshots": snapshot_evidence,
                "allocation_artifact_inputs_as_of": artifact.inputs_as_of.isoformat(),
                "max_gross_exposure": gross_limit,
                "primary_account": primary_evidence,
                "allocation_continuity": continuity_evidence,
                "industry_membership_as_of": {
                    instrument: observed[0].isoformat()
                    for instrument, observed in industry_observations.items()
                },
            },
        )
