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
    simulation_events,
    simulation_nav,
    simulation_portfolios,
    simulation_positions,
    strategy_allocation_artifacts,
    strategy_allocation_members,
    strategy_allocations,
    strategy_versions,
)

from .research_horizon import LONG_1_3Y, SHORT_1_5D, SWING_1_6M
from .strategy_health import cap_targets_for_health
from .strategy_health_authority import load_production_health_gate

NETTING_PLAN_VERSION = "account-netting-plan-v6-actual-sleeve-inventory"
SLEEVE_INVENTORY_POLICY_VERSION = (
    "actual-account-weight-prior-plan-member-pro-rata-v1"
)
DEFAULT_EXECUTION_POLICY = "open"
EXECUTION_POLICIES = (DEFAULT_EXECUTION_POLICY, "next_bar", "twap", "vwap")

_TOLERANCE = 1e-9


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def sleeve_inventory_policy_contract() -> dict[str, Any]:
    """Return the immutable policy used to project one real account into sleeves."""

    contract: dict[str, Any] = {
        "policy_version": SLEEVE_INVENTORY_POLICY_VERSION,
        "actual_inventory": "position_market_value_divided_by_primary_nav",
        "target_component": "prior_net_target_by_prior_member_account_target_ratio",
        "residual_basis_precedence": [
            "prior_negative_gross_demand",
            "prior_member_current_account_inventory",
            "prior_absolute_gross_demand",
            "prior_member_account_targets",
        ],
        "allocation": "target_component_then_unfilled_residual_sorted_pro_rata",
        "unattributed_positive_inventory": "fail_closed",
    }
    return {**contract, "policy_sha256": _canonical_hash(contract)}


def primary_position_inventory_evidence(
    *,
    portfolio_id: str,
    nav: float,
    positions: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Seal the exact primary-ledger positions used by sleeve attribution."""

    normalized_nav = float(nav)
    if not isfinite(normalized_nav) or normalized_nav <= 0:
        raise ValueError("primary account NAV must be positive")
    facts: dict[str, Any] = {}
    actual_weights: dict[str, float] = {}
    for raw in sorted(positions, key=lambda value: str(value.get("instrument") or "")):
        instrument = str(raw.get("instrument") or "").strip()
        quantity = int(raw.get("quantity") or 0)
        market_value = float(raw.get("market_value") or 0.0)
        if (
            not instrument
            or quantity < 0
            or not isfinite(market_value)
            or market_value < 0
        ):
            raise ValueError("primary account has invalid long-only position inventory")
        if quantity > 0 and market_value <= 0:
            raise ValueError(
                "primary account position has no positive marked market value: "
                f"{instrument}"
            )
        market_price = raw.get("market_price")
        normalized_market_price = (
            float(market_price) if market_price is not None else None
        )
        if normalized_market_price is not None and (
            not isfinite(normalized_market_price) or normalized_market_price < 0
        ):
            raise ValueError("primary account position has an invalid market price")
        market_date = raw.get("market_date")
        updated_at = raw.get("updated_at")
        account_weight = market_value / normalized_nav
        if account_weight > _TOLERANCE:
            actual_weights[instrument] = account_weight
        facts[instrument] = {
            "quantity": quantity,
            "market_value": market_value,
            "market_price": normalized_market_price,
            "market_date": (
                market_date.isoformat()
                if hasattr(market_date, "isoformat")
                else (str(market_date) if market_date else None)
            ),
            "stale": bool(raw.get("stale")),
            "updated_at": (
                updated_at.isoformat()
                if hasattr(updated_at, "isoformat")
                else str(updated_at or "")
            ),
            "account_weight": account_weight,
        }
    if sum(actual_weights.values()) > 1.0 + 1e-6:
        raise ValueError("actual account inventory exceeds primary account NAV")
    evidence: dict[str, Any] = {
        "portfolio_id": str(portfolio_id),
        "nav": normalized_nav,
        "positions": facts,
    }
    evidence["positions_sha256"] = _canonical_hash(evidence)
    return evidence, actual_weights


def allocate_actual_sleeve_inventory(
    *,
    actual_account_weights: dict[str, float],
    prior_plan: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Allocate real account weights to virtual sleeves deterministically.

    The unified account is the accounting authority.  A previous plan is used
    only for each instrument's member *ratio*; its target weights are never
    treated as proof that an order filled.  Exact account weights are assigned
    in sorted member order, with the final member receiving the floating-point
    residual so sleeve inventory reconciles to the account exactly.
    """

    actual = {
        str(instrument): float(weight)
        for instrument, weight in actual_account_weights.items()
    }
    if len(actual) != len(actual_account_weights) or any(
        not isfinite(weight) or weight < -_TOLERANCE for weight in actual.values()
    ):
        raise ValueError("actual account inventory must contain finite non-negative weights")
    actual = {
        instrument: max(weight, 0.0)
        for instrument, weight in actual.items()
        if weight > _TOLERANCE
    }
    if sum(actual.values()) > 1.0 + 1e-6:
        raise ValueError("actual account inventory exceeds primary account NAV")

    raw_budgets = prior_plan.get("member_budgets") or {}
    raw_targets = prior_plan.get("member_targets") or {}
    if not isinstance(raw_budgets, dict) or not isinstance(raw_targets, dict):
        raise ValueError("prior plan has invalid member target attribution")
    budgets = {str(member): float(weight) for member, weight in raw_budgets.items()}
    if any(not isfinite(weight) or weight < -_TOLERANCE for weight in budgets.values()):
        raise ValueError("prior plan has invalid member budgets")

    def _add_basis(
        destination: dict[str, dict[str, float]],
        member: str,
        instrument: str,
        weight: float,
    ) -> None:
        if weight > _TOLERANCE:
            destination.setdefault(instrument, {})[member] = (
                destination.setdefault(instrument, {}).get(member, 0.0) + weight
            )

    target_basis: dict[str, dict[str, float]] = {}
    for raw_member, raw_book in raw_targets.items():
        member = str(raw_member)
        if not isinstance(raw_book, dict):
            raise ValueError("prior plan has invalid member target attribution")
        budget = budgets.get(member, 0.0)
        for raw_instrument, raw_weight in raw_book.items():
            weight = float(raw_weight)
            if not isfinite(weight) or weight < -_TOLERANCE:
                raise ValueError("prior plan has invalid member target weights")
            _add_basis(target_basis, member, str(raw_instrument), budget * weight)

    current_basis: dict[str, dict[str, float]] = {}
    raw_current_account = prior_plan.get("member_current_account_weights")
    if isinstance(raw_current_account, dict):
        for raw_member, raw_book in raw_current_account.items():
            member = str(raw_member)
            if not isinstance(raw_book, dict):
                raise ValueError("prior plan has invalid exact sleeve inventory")
            for raw_instrument, raw_weight in raw_book.items():
                weight = float(raw_weight)
                if not isfinite(weight) or weight < -_TOLERANCE:
                    raise ValueError("prior plan has invalid exact sleeve inventory")
                _add_basis(current_basis, member, str(raw_instrument), weight)
    else:
        raw_current = prior_plan.get("member_current_weights") or {}
        if not isinstance(raw_current, dict):
            raise ValueError("prior plan has invalid legacy sleeve inventory")
        for raw_member, raw_book in raw_current.items():
            member = str(raw_member)
            if not isinstance(raw_book, dict):
                raise ValueError("prior plan has invalid legacy sleeve inventory")
            budget = budgets.get(member, 0.0)
            for raw_instrument, raw_weight in raw_book.items():
                weight = float(raw_weight)
                if not isfinite(weight) or weight < -_TOLERANCE:
                    raise ValueError("prior plan has invalid legacy sleeve inventory")
                _add_basis(current_basis, member, str(raw_instrument), budget * weight)

    negative_demand_basis: dict[str, dict[str, float]] = {}
    gross_demand_basis: dict[str, dict[str, float]] = {}
    raw_contributions = prior_plan.get("strategy_contributions") or {}
    if not isinstance(raw_contributions, dict):
        raise ValueError("prior plan has invalid strategy attribution")
    for raw_instrument, raw_entry in raw_contributions.items():
        if not isinstance(raw_entry, dict):
            raise ValueError("prior plan has invalid strategy attribution")
        raw_members = raw_entry.get("members") or {}
        if not isinstance(raw_members, dict):
            raise ValueError("prior plan has invalid strategy attribution")
        for raw_member, raw_values in raw_members.items():
            if not isinstance(raw_values, dict):
                raise ValueError("prior plan has invalid strategy attribution")
            gross_delta = float(raw_values.get("gross_delta") or 0.0)
            if not isfinite(gross_delta):
                raise ValueError("prior plan has invalid strategy attribution")
            member = str(raw_member)
            instrument = str(raw_instrument)
            if gross_delta < -_TOLERANCE:
                _add_basis(negative_demand_basis, member, instrument, abs(gross_delta))
            if abs(gross_delta) > _TOLERANCE:
                _add_basis(gross_demand_basis, member, instrument, abs(gross_delta))

    raw_net_targets = prior_plan.get("net_targets") or {}
    if not isinstance(raw_net_targets, dict):
        raise ValueError("prior plan has invalid account targets")
    prior_net_targets: dict[str, float] = {}
    for raw_instrument, raw_entry in raw_net_targets.items():
        if not isinstance(raw_entry, dict):
            raise ValueError("prior plan has invalid account targets")
        weight = float(raw_entry.get("weight") or 0.0)
        if not isfinite(weight) or weight < -_TOLERANCE:
            raise ValueError("prior plan has invalid account targets")
        prior_net_targets[str(raw_instrument)] = max(weight, 0.0)

    residual_sources = (
        ("prior_negative_gross_demand", negative_demand_basis),
        ("prior_member_current_account_inventory", current_basis),
        ("prior_absolute_gross_demand", gross_demand_basis),
        ("prior_member_account_targets", target_basis),
    )

    def _allocate_component(
        amount: float, basis: dict[str, float]
    ) -> dict[str, float]:
        members = sorted(basis)
        total_basis = sum(basis.values())
        remaining = amount
        result: dict[str, float] = {}
        for member in members[:-1]:
            member_weight = amount * basis[member] / total_basis
            result[member] = member_weight
            remaining -= member_weight
        result[members[-1]] = max(remaining, 0.0)
        return result

    inventory: dict[str, dict[str, float]] = {}
    allocations: dict[str, Any] = {}
    for instrument in sorted(actual):
        assigned: dict[str, float] = {}
        allocation_steps: list[dict[str, Any]] = []
        target_members = {
            member: weight
            for member, weight in (target_basis.get(instrument) or {}).items()
            if weight > _TOLERANCE
        }
        target_component = min(
            actual[instrument], prior_net_targets.get(instrument, 0.0)
        )
        if target_component > _TOLERANCE and target_members:
            component = _allocate_component(target_component, target_members)
            for member, weight in component.items():
                assigned[member] = assigned.get(member, 0.0) + weight
            allocation_steps.append(
                {
                    "source": "prior_member_account_targets",
                    "account_weight": target_component,
                    "basis_account_weights": {
                        member: target_members[member] for member in sorted(target_members)
                    },
                    "allocated_account_weights": component,
                }
            )

        residual = actual[instrument] - sum(assigned.values())
        for candidate_name, candidate in residual_sources:
            if residual <= _TOLERANCE:
                break
            candidate_basis = {
                member: weight
                for member, weight in (candidate.get(instrument) or {}).items()
                if weight > _TOLERANCE
            }
            if sum(candidate_basis.values()) > _TOLERANCE:
                component = _allocate_component(residual, candidate_basis)
                for member, weight in component.items():
                    assigned[member] = assigned.get(member, 0.0) + weight
                allocation_steps.append(
                    {
                        "source": candidate_name,
                        "account_weight": residual,
                        "basis_account_weights": {
                            member: candidate_basis[member]
                            for member in sorted(candidate_basis)
                        },
                        "allocated_account_weights": component,
                    }
                )
                residual = 0.0
                break
        if residual > _TOLERANCE or not assigned:
            raise ValueError(
                "actual primary position has no prior member attribution: "
                f"{instrument}"
            )
        for member, weight in assigned.items():
            inventory.setdefault(member, {})[instrument] = weight
        allocations[instrument] = {
            "actual_account_weight": actual[instrument],
            "basis_source": "+".join(step["source"] for step in allocation_steps),
            "allocation_steps": allocation_steps,
            "allocated_account_weights": assigned,
        }

    policy = sleeve_inventory_policy_contract()
    evidence: dict[str, Any] = {
        **policy,
        "prior_plan_hash": str(prior_plan.get("plan_hash") or ""),
        "actual_account_weights": actual,
        "instrument_allocations": allocations,
    }
    evidence["allocation_sha256"] = _canonical_hash(evidence)
    return inventory, evidence


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


def _validated_persisted_plan_payload(row: Any) -> dict[str, Any]:
    """Verify a stored plan's identity and economic-input hash.

    ``account_netting_plans`` is append-only by convention, but continuity is
    too important to trust a JSON reference alone.  Recompute both the stable
    plan key and the economic plan hash from the persisted payload before a
    prior plan can own sleeve inventory on another day.
    """

    plan = dict(row.plan_json or {})
    column_identity = {
        "plan_key": str(row.plan_key),
        "account_id": str(row.account_id),
        "allocation_artifact_id": str(row.allocation_artifact_id),
        "decision_date": row.decision_date.isoformat(),
        "inputs_as_of": row.inputs_as_of.isoformat(),
        "policy_version": str(row.policy_version),
        "execution_policy": str(row.execution_policy),
        "tranche_index": int(row.tranche_index),
        "plan_hash": str(row.plan_hash),
    }
    textual_fields = (
        "plan_key",
        "account_id",
        "allocation_artifact_id",
        "decision_date",
        "inputs_as_of",
        "policy_version",
        "execution_policy",
        "plan_hash",
    )
    if any(
        str(plan.get(key)) != str(column_identity[key]) for key in textual_fields
    ) or int(plan.get("tranche_index", -1)) != int(row.tranche_index):
        raise ValueError("account netting plan columns differ from the sealed payload")
    expected_key = plan_idempotency_key(
        account_id=column_identity["account_id"],
        allocation_artifact_id=column_identity["allocation_artifact_id"],
        decision_date=row.decision_date,
        inputs_as_of=row.inputs_as_of,
        policy_version=column_identity["policy_version"],
        tranche_index=int(row.tranche_index),
    )
    if expected_key != column_identity["plan_key"]:
        raise ValueError("account netting plan key seal is invalid")

    raw_targets = plan.get("net_targets")
    raw_trades = plan.get("net_trades")
    if not isinstance(raw_targets, dict) or not isinstance(raw_trades, dict):
        raise ValueError("account netting plan economic inputs are invalid")
    try:
        net_targets = {
            str(instrument): float(values["weight"])
            for instrument, values in raw_targets.items()
            if isinstance(values, dict)
        }
        net_trades = {
            str(instrument): float(values["delta_weight"])
            for instrument, values in raw_trades.items()
            if isinstance(values, dict)
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("account netting plan economic inputs are invalid") from exc
    if len(net_targets) != len(raw_targets) or len(net_trades) != len(raw_trades):
        raise ValueError("account netting plan economic inputs are invalid")
    hash_payload: dict[str, Any] = {
        "plan_version": str(plan.get("plan_version") or ""),
        "plan_key": column_identity["plan_key"],
        "member_budgets": plan.get("member_budgets") or {},
        "member_targets": plan.get("member_targets") or {},
        "member_current_weights": plan.get("member_current_weights") or {},
        "total_capital": float(plan.get("total_capital") or 0.0),
        "net_targets": net_targets,
        "net_trades": net_trades,
        "cash_weight": float(plan.get("cash_weight") or 0.0),
        "strategy_contributions": plan.get("strategy_contributions") or {},
        "constraint_clamps": plan.get("constraint_clamps") or {},
        "industry_memberships": plan.get("industry_memberships") or {},
        "industry_exposure": plan.get("industry_exposure") or {},
        "industry_constraint_clamps": plan.get("industry_constraint_clamps") or {},
        "max_instrument_weight": plan.get("max_instrument_weight"),
        "max_industry_weight": plan.get("max_industry_weight"),
        "execution_policy": column_identity["execution_policy"],
        "input_evidence": plan.get("input_evidence") or {},
    }
    if plan.get("plan_version") == NETTING_PLAN_VERSION:
        hash_payload["member_current_account_weights"] = (
            plan.get("member_current_account_weights") or {}
        )
    if _canonical_hash(hash_payload) != column_identity["plan_hash"]:
        raise ValueError("account netting plan input hash is invalid")
    return plan


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
    member_current_account_weights: dict[str, dict[str, float]] | None = None,
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
    current sleeve weights. ``member_current_account_weights`` is the
    authoritative live-account form: it carries actual account weights already
    allocated to sleeves and may include a departed member whose residual
    position must still be sold. The two current-inventory forms are mutually
    exclusive; without either every target is a fresh buy.
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
    current_account_books = {
        str(member): values
        for member, values in (member_current_account_weights or {}).items()
    }
    if member_current_weights is not None and member_current_account_weights is not None:
        raise ValueError(
            "member current sleeve weights and exact account weights are mutually exclusive"
        )
    if len(target_books) != len(member_targets) or len(current_books) != len(
        member_current_weights or {}
    ) or len(current_account_books) != len(member_current_account_weights or {}):
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
    normalized_member_current_account_weights: dict[str, dict[str, float]] = {}
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
        normalized_member_current_account_weights[member] = {
            instrument: budget * weight for instrument, weight in sleeve_current.items()
        }

    if member_current_account_weights is not None:
        normalized_member_current_account_weights = {}
        for member, raw_book in current_account_books.items():
            account_book = {
                str(instrument): float(weight)
                for instrument, weight in raw_book.items()
            }
            if len(account_book) != len(raw_book):
                raise ValueError(
                    "current account instruments must be unique after normalization"
                )
            if any(
                not isfinite(weight) or weight < -_TOLERANCE
                for weight in account_book.values()
            ):
                raise ValueError(
                    "member current account weights must be finite and non-negative"
                )
            normalized_member_current_account_weights[member] = account_book
        if (
            sum(
                weight
                for book in normalized_member_current_account_weights.values()
                for weight in book.values()
            )
            > 1.0 + 1e-6
        ):
            raise ValueError("member current account weights exceed primary account NAV")

    target_account_books = {
        member: {
            instrument: budgets[member] * weight
            for instrument, weight in targets.items()
        }
        for member, targets in normalized_member_targets.items()
    }
    for member in sorted(
        set(target_account_books) | set(normalized_member_current_account_weights)
    ):
        targets = target_account_books.get(member) or {}
        currents = normalized_member_current_account_weights.get(member) or {}
        for instrument in sorted(set(targets) | set(currents)):
            target = targets.get(instrument, 0.0)
            current = currents.get(instrument, 0.0)
            net_targets[instrument] = net_targets.get(instrument, 0.0) + target
            account_current[instrument] = account_current.get(instrument, 0.0) + current
            delta = target - current
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
        "member_current_account_weights": normalized_member_current_account_weights,
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
        # Keep the complete normalized input, not only surviving targets.  The
        # plan hash already commits to this full map; persisting the same map
        # makes that input hash independently reproducible during continuity
        # selection.
        "industry_memberships": {
            instrument: industries[instrument] for instrument in sorted(industries)
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
            "member_current_account_weights": (
                normalized_member_current_account_weights
            ),
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

    @staticmethod
    def _authoritative_prior_plan(
        connection: Any,
        *,
        portfolio_id: str,
        current_allocation: Any,
        decision_date: date,
        inputs_as_of: date,
    ) -> dict[str, Any] | None:
        """Select one valued prior plan across execution and decision-only days.

        A zero-order sleeve transfer is economically effective only after its
        valuation replay succeeds.  Such batches deliberately have no plan FK,
        so their sealed source event is the join authority.  Executed plans and
        those decision-only plans are validated under the same portfolio,
        allocation-lineage and input-hash rules, then ordered on their actual
        trade/effective time.
        """

        from .simulation_store import (  # local import avoids the module cycle
            ACCOUNT_DECISION_VALUATION_VERSION,
            ACCOUNT_ORDER_DECISION_EVENT_TYPE,
            validate_account_order_decision_event,
        )

        portfolio = connection.execute(
            select(simulation_portfolios).where(
                simulation_portfolios.c.id == portfolio_id
            )
        ).first()
        if portfolio is None:
            raise ValueError("primary account continuity portfolio is unavailable")
        if str(portfolio.source_type) != "allocation" or str(portfolio.source_id) != str(
            current_allocation.id
        ):
            raise ValueError("primary account continuity does not match its allocation")

        common_columns = (
            simulation_batches.c.id.label("continuity_batch_id"),
            simulation_batches.c.trade_date.label("continuity_trade_date"),
            simulation_batches.c.finished_at.label("continuity_finished_at"),
            simulation_batches.c.created_at.label("continuity_batch_created_at"),
            simulation_batches.c.target_payload_json.label("continuity_target_payload"),
            strategy_allocation_artifacts.c.allocation_id.label(
                "continuity_artifact_allocation_id"
            ),
            strategy_allocation_artifacts.c.inputs_as_of.label(
                "continuity_artifact_inputs_as_of"
            ),
            strategy_allocation_artifacts.c.artifact_hash.label(
                "continuity_artifact_hash"
            ),
            strategy_allocations.c.analysis_json.label(
                "continuity_allocation_analysis"
            ),
        )
        executed = connection.execute(
            select(
                account_netting_plans,
                *common_columns,
            )
            .select_from(
                simulation_batches.join(
                    account_netting_plans,
                    account_netting_plans.c.id
                    == simulation_batches.c.account_netting_plan_id,
                )
                .join(
                    simulation_nav,
                    (simulation_nav.c.portfolio_id == simulation_batches.c.portfolio_id)
                    & (simulation_nav.c.trade_date == simulation_batches.c.trade_date),
                )
                .join(
                    strategy_allocation_artifacts,
                    strategy_allocation_artifacts.c.id
                    == account_netting_plans.c.allocation_artifact_id,
                )
                .join(
                    strategy_allocations,
                    strategy_allocations.c.id
                    == strategy_allocation_artifacts.c.allocation_id,
                )
            )
            .where(
                simulation_batches.c.portfolio_id == portfolio_id,
                simulation_batches.c.status == "succeeded",
                account_netting_plans.c.inputs_as_of <= inputs_as_of,
                account_netting_plans.c.decision_date <= decision_date,
                simulation_batches.c.trade_date <= decision_date,
            )
            .order_by(
                simulation_batches.c.trade_date.desc(),
                simulation_batches.c.finished_at.desc(),
                simulation_batches.c.created_at.desc(),
            )
            .limit(64)
        ).all()
        decision_only = connection.execute(
            select(
                account_netting_plans,
                *common_columns,
                simulation_events.c.id.label("continuity_event_id"),
                simulation_events.c.batch_id.label("continuity_event_batch_id"),
                simulation_events.c.trade_date.label("continuity_event_trade_date"),
                simulation_events.c.created_at.label("continuity_event_created_at"),
                simulation_events.c.details_json.label("continuity_event_payload"),
            )
            .select_from(
                simulation_batches.join(
                    simulation_events,
                    simulation_events.c.id == simulation_batches.c.source_snapshot_id,
                )
                .join(
                    account_netting_plans,
                    account_netting_plans.c.id
                    == simulation_events.c.details_json[
                        "account_netting_plan_id"
                    ].as_string(),
                )
                .join(
                    simulation_nav,
                    (simulation_nav.c.portfolio_id == simulation_batches.c.portfolio_id)
                    & (simulation_nav.c.trade_date == simulation_batches.c.trade_date),
                )
                .join(
                    strategy_allocation_artifacts,
                    strategy_allocation_artifacts.c.id
                    == account_netting_plans.c.allocation_artifact_id,
                )
                .join(
                    strategy_allocations,
                    strategy_allocations.c.id
                    == strategy_allocation_artifacts.c.allocation_id,
                )
            )
            .where(
                simulation_batches.c.portfolio_id == portfolio_id,
                simulation_batches.c.status == "succeeded",
                simulation_batches.c.account_netting_plan_id.is_(None),
                simulation_events.c.portfolio_id == portfolio_id,
                simulation_events.c.event_type == ACCOUNT_ORDER_DECISION_EVENT_TYPE,
                simulation_events.c.batch_id.is_(None),
                account_netting_plans.c.inputs_as_of <= inputs_as_of,
                account_netting_plans.c.decision_date <= decision_date,
                simulation_batches.c.trade_date <= decision_date,
            )
            .order_by(
                simulation_batches.c.trade_date.desc(),
                simulation_batches.c.finished_at.desc(),
                simulation_batches.c.created_at.desc(),
            )
            .limit(64)
        ).all()

        current_analysis = dict(current_allocation.analysis_json or {})
        current_lineage = str(
            current_analysis.get("allocation_dataset_lineage_id") or ""
        ).lower()
        if (
            not _is_sha256(current_lineage)
            or current_lineage != str(portfolio.daily_dataset_lineage_id).lower()
        ):
            raise ValueError("primary account allocation lineage is invalid")

        candidates: list[dict[str, Any]] = []
        for kind, rows in (("execution", executed), ("decision_only", decision_only)):
            for row in rows:
                plan = _validated_persisted_plan_payload(row)
                artifact_hash = str(row.continuity_artifact_hash or "").lower()
                if (
                    str(row.account_id)
                    != str(row.continuity_artifact_allocation_id)
                    or str(row.allocation_artifact_id)
                    != str(plan.get("allocation_artifact_id") or "")
                    or not _is_sha256(artifact_hash)
                    or str(row.continuity_artifact_inputs_as_of)
                    != str((plan.get("input_evidence") or {}).get(
                        "allocation_artifact_inputs_as_of"
                    ) or "")
                ):
                    raise ValueError("prior account plan allocation input seal is invalid")
                prior_analysis = dict(row.continuity_allocation_analysis or {})
                prior_lineage = str(
                    prior_analysis.get("allocation_dataset_lineage_id") or ""
                ).lower()
                if not _is_sha256(prior_lineage) or prior_lineage != current_lineage:
                    raise ValueError(
                        "prior account plan belongs to another allocation lineage"
                    )

                input_evidence = plan.get("input_evidence") or {}
                primary = (
                    input_evidence.get("primary_account")
                    if isinstance(input_evidence, dict)
                    else None
                )
                if (
                    not isinstance(primary, dict)
                    or str(primary.get("portfolio_id") or "") != portfolio_id
                    or str(primary.get("source_id") or "") != str(row.account_id)
                ):
                    raise ValueError(
                        "prior account plan is not sealed to the primary portfolio"
                    )

                target_payload = dict(row.continuity_target_payload or {})
                target_version = f"three-horizon-netting:{row.plan_hash}"
                if kind == "execution":
                    order_plan = target_payload.get("order_plan")
                    if (
                        not isinstance(order_plan, dict)
                        or str(order_plan.get("account_netting_plan_id") or "")
                        != str(row.id)
                        or str(order_plan.get("target_version") or "") != target_version
                    ):
                        raise ValueError(
                            "executed account plan batch has invalid input binding"
                        )
                    event_id = None
                else:
                    event_payload = dict(row.continuity_event_payload or {})
                    validated_event = validate_account_order_decision_event(
                        event_payload,
                        portfolio_id=portfolio_id,
                        account_netting_plan_id=str(row.id),
                    )
                    order_plan = target_payload.get("order_plan")
                    governed = target_payload.get("governed_order_plan")
                    if (
                        str(row.continuity_event_id)
                        != str(validated_event["event_sha256"])
                        or row.continuity_event_batch_id is not None
                        or row.continuity_event_trade_date
                        != row.continuity_trade_date
                        or str(validated_event.get("target_version") or "")
                        != target_version
                        or not isinstance(order_plan, dict)
                        or order_plan.get("decision_valuation_replay") is not True
                        or str(order_plan.get("decision_event_sha256") or "")
                        != str(row.continuity_event_id)
                        or order_plan.get("account_netting_plan_id") is not None
                        or order_plan.get("actions") != []
                        or not isinstance(governed, dict)
                        or governed.get("format_version")
                        != ACCOUNT_DECISION_VALUATION_VERSION
                        or str(governed.get("decision_event_sha256") or "")
                        != str(row.continuity_event_id)
                    ):
                        raise ValueError(
                            "decision-only account plan valuation binding is invalid"
                        )
                    event_id = str(row.continuity_event_id)

                effective_at = (
                    row.continuity_finished_at
                    or row.continuity_batch_created_at
                    or row.created_at
                )
                candidates.append(
                    {
                        "kind": kind,
                        "plan": plan,
                        "plan_id": str(row.id),
                        "batch_id": str(row.continuity_batch_id),
                        "event_id": event_id,
                        "trade_date": row.continuity_trade_date,
                        "effective_at": effective_at,
                        "allocation_id": str(row.account_id),
                        "allocation_lineage_id": prior_lineage or current_lineage,
                        "artifact_hash": artifact_hash,
                    }
                )
        if not candidates:
            return None
        candidates.sort(
            key=lambda value: (value["trade_date"], value["effective_at"]),
            reverse=True,
        )
        selected = candidates[0]
        tied = [
            value
            for value in candidates[1:]
            if (value["trade_date"], value["effective_at"])
            == (selected["trade_date"], selected["effective_at"])
            and value["plan_id"] != selected["plan_id"]
        ]
        if tied:
            raise ValueError("account plan continuity has an ambiguous effective time")
        return selected

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

        if primary_account is not None and member_current_weights is not None:
            raise ValueError(
                "primary account sleeve inventory must come from the real ledger"
            )
        member_current_account_weights: dict[str, dict[str, float]] | None = None
        sleeve_inventory_evidence: dict[str, Any] | None = None
        primary_position_evidence: dict[str, Any] | None = None
        with self.engine.connect() as connection:
            allocation = connection.execute(
                select(strategy_allocations).where(
                    strategy_allocations.c.id == allocation_id
                )
            ).first()
            if allocation is None:
                raise KeyError(allocation_id)
            if primary_account is not None:
                primary_id = str(primary_account.get("portfolio_id") or "").strip()
                primary_source = str(primary_account.get("source_id") or "").strip()
                primary_nav = float(primary_account.get("nav") or 0.0)
                if (
                    not primary_id
                    or primary_source != str(allocation.id)
                    or not isfinite(primary_nav)
                    or primary_nav <= 0
                ):
                    raise ValueError(
                        "primary account capital evidence does not match the active allocation"
                    )
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
                            # A historical backfill can finish after a newer
                            # trading-day snapshot.  Creation time is therefore
                            # not the business ordering for account targets.
                            .order_by(
                                recommendation_snapshots.c.as_of_date.desc(),
                                recommendation_snapshots.c.created_at.desc(),
                            )
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
                    authority = load_production_health_gate(connection, version_id)
                    health_status = str(authority.get("health_status") or "invalid")
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
                    targets[version_id], health_cap = cap_targets_for_health(
                        targets[version_id],
                        previous_targets,
                        health_status,
                    )
                    health_gate = {**health_cap, **authority}
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
            if primary_account is not None:
                position_rows = connection.execute(
                    select(simulation_positions)
                    .where(simulation_positions.c.portfolio_id == primary_id)
                    .order_by(simulation_positions.c.instrument)
                ).all()
                primary_position_evidence, actual_account_weights = (
                    primary_position_inventory_evidence(
                        portfolio_id=primary_id,
                        nav=primary_nav,
                        positions=[dict(position._mapping) for position in position_rows],
                    )
                )
                prior = (
                    self._authoritative_prior_plan(
                        connection,
                        portfolio_id=primary_id,
                        current_allocation=allocation,
                        decision_date=decision_date,
                        inputs_as_of=inputs_as_of,
                    )
                    if actual_account_weights
                    else None
                )
                if prior is None and actual_account_weights:
                    raise ValueError(
                        "primary account has positions but no valued plan for sleeve attribution"
                    )
                prior_plan = dict(prior["plan"]) if prior is not None else {}
                member_current_account_weights, sleeve_inventory_evidence = (
                    allocate_actual_sleeve_inventory(
                        actual_account_weights=actual_account_weights,
                        prior_plan=prior_plan,
                    )
                )
                if prior is not None:
                    sleeve_inventory_evidence.update(
                        {
                            "prior_plan_id": prior["plan_id"],
                            "prior_batch_id": prior["batch_id"],
                            "prior_allocation_id": prior["allocation_id"],
                            "prior_source_kind": prior["kind"],
                            "prior_decision_event_id": prior["event_id"],
                            "prior_effective_trade_date": prior[
                                "trade_date"
                            ].isoformat(),
                            "prior_allocation_lineage_id": prior[
                                "allocation_lineage_id"
                            ],
                            "prior_allocation_artifact_hash": prior[
                                "artifact_hash"
                            ],
                        }
                    )
                    sleeve_inventory_evidence["allocation_sha256"] = _canonical_hash(
                        {
                            key: value
                            for key, value in sleeve_inventory_evidence.items()
                            if key != "allocation_sha256"
                        }
                    )
                    continuity_evidence = {
                        "portfolio_id": primary_id,
                        "prior_plan_id": prior["plan_id"],
                        "prior_batch_id": prior["batch_id"],
                        "prior_allocation_id": prior["allocation_id"],
                        "prior_source_kind": prior["kind"],
                        "prior_decision_event_id": prior["event_id"],
                        "prior_effective_trade_date": prior[
                            "trade_date"
                        ].isoformat(),
                        "prior_effective_at": prior["effective_at"].isoformat(),
                        "prior_plan_hash": str(prior_plan["plan_hash"]),
                        "prior_allocation_lineage_id": prior[
                            "allocation_lineage_id"
                        ],
                        "prior_allocation_artifact_hash": prior["artifact_hash"],
                        "carried_members": sorted(member_current_account_weights),
                    }
            if member_current_weights is None and primary_account is None:
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
                "positions_sha256": str(
                    (primary_position_evidence or {}).get("positions_sha256") or ""
                ),
                "sleeve_inventory_policy_sha256": str(
                    (sleeve_inventory_evidence or {}).get("policy_sha256") or ""
                ),
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
            member_current_account_weights=member_current_account_weights,
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
                "primary_position_inventory": primary_position_evidence,
                "sleeve_inventory_allocation": sleeve_inventory_evidence,
                "allocation_continuity": continuity_evidence,
                "industry_membership_as_of": {
                    instrument: observed[0].isoformat()
                    for instrument, observed in industry_observations.items()
                },
            },
        )
