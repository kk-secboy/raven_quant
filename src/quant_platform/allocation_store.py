from __future__ import annotations

import calendar
import hashlib
import json
import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    backtest_runs,
    recommendation_portfolios,
    row_dict,
    simulation_nav,
    simulation_portfolios,
    strategies,
    strategy_allocation_artifacts,
    strategy_allocation_events,
    strategy_allocation_members,
    strategy_allocation_nav,
    strategy_allocations,
)

from .member_risk_gate import (
    ALLOCATION_LIQUIDATION_RULE,
    ALLOCATION_REDUCTION_RULE,
    has_open_allocation_gate,
    has_open_member_gate,
    load_allocation_risk_state,
    load_member_risk_state,
    load_strategy_risk_state,
)
from .risk_math import COVARIANCE_MODEL_VERSION
from .strategy_allocation import (
    analyze_strategy_allocation,
    renormalize_budgets_for_suspended,
)
from .strategy_store import StrategyStore


def _now() -> datetime:
    return datetime.now(UTC)


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


DECISION_FREQUENCIES = frozenset({"weekly", "monthly"})


def next_decision_date(decision_date: date, frequency: str) -> date:
    """Next frozen decision day; the artifact stays valid strictly before it."""

    if frequency == "weekly":
        return decision_date + timedelta(days=7)
    if frequency == "monthly":
        month = decision_date.month + 1
        year = decision_date.year + (1 if month > 12 else 0)
        month = month - 12 if month > 12 else month
        day = min(decision_date.day, calendar.monthrange(year, month)[1])
        return date(year, month, day)
    raise ValueError(f"unknown decision frequency: {frequency}")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


class AllocationStore:
    """Low-correlation strategy allocations measured only from simulation NAV."""

    def __init__(self, database_url: str) -> None:
        self.strategies = StrategyStore(database_url)
        self.engine = self.strategies.engine

    @staticmethod
    def _daily_returns(backtest: dict[str, Any]) -> pd.Series:
        root = Path(str(backtest["artifact_path"]))
        candidates = (
            root / str(backtest["id"]) / "daily_returns.parquet",
            root / "daily_returns.parquet",
        )
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise ValueError(f"backtest {backtest['id']} has no daily return artifact")
        frame = pd.read_parquet(source)
        date_column = next(
            (name for name in ("datetime", "trade_date", "date") if name in frame.columns),
            None,
        )
        if date_column is None:
            raise ValueError(f"backtest {backtest['id']} daily artifact has no date")
        if "net_return" in frame:
            values = pd.to_numeric(frame["net_return"], errors="coerce")
        elif {"return", "cost"}.issubset(frame.columns):
            values = pd.to_numeric(frame["return"], errors="coerce") - pd.to_numeric(
                frame["cost"], errors="coerce"
            )
        else:
            raise ValueError(f"backtest {backtest['id']} daily artifact is incomplete")
        timestamps = pd.to_datetime(frame[date_column], errors="coerce")
        series = pd.Series(values.to_numpy(), index=timestamps).dropna().sort_index()
        if series.index.duplicated().any():
            raise ValueError(f"backtest {backtest['id']} has duplicate return dates")
        return series

    def create(
        self,
        *,
        name: str,
        strategy_version_ids: list[str],
        dataset: str,
        total_capital: float,
        allocation_method: str,
        lookback_days: int,
        target_volatility: float,
        max_pairwise_correlation: float,
        max_strategy_weight: float,
        max_member_drawdown: float,
        max_drawdown_reduce: float,
        max_drawdown_liquidate: float,
        fixed_weights: dict[str, float] | None,
        actor: str,
        member_specs: list[dict[str, Any]] | None = None,
        decision_frequency: str = "monthly",
    ) -> dict[str, Any]:
        version_ids = list(
            dict.fromkeys(item.strip() for item in strategy_version_ids if item.strip())
        )
        if len(version_ids) < 2 or len(version_ids) > 10:
            raise ValueError("a strategy allocation requires 2 to 10 unique strategy versions")
        if decision_frequency not in DECISION_FREQUENCIES:
            raise ValueError("allocation decision frequency must be weekly or monthly")
        if not name.strip() or not dataset.strip() or not actor.strip():
            raise ValueError("name, dataset and actor are required")
        if total_capital < 500_000:
            raise ValueError("strategy allocation capital must be at least 500000")
        if not 0 < max_member_drawdown < max_drawdown_reduce < max_drawdown_liquidate <= 0.50:
            raise ValueError("member, reduction and liquidation drawdowns must be increasing")
        supplied = {
            str(item.get("strategy_version_id") or "").strip(): dict(item)
            for item in (member_specs or [])
        }
        if supplied and set(supplied) != set(version_ids):
            raise ValueError("allocation member specifications do not match strategy versions")
        governance: dict[str, dict[str, Any]] = {}
        for version_id in version_ids:
            item = supplied.get(version_id, {})
            role = str(item.get("role") or "core")
            if role not in {"core", "satellite"}:
                raise ValueError("allocation member role must be core or satellite")
            risk_budget = float(item.get("risk_budget", 1.0))
            default_cap = min(max_strategy_weight, 0.15 if role == "satellite" else 0.70)
            member_cap = float(item.get("member_cap") or default_cap)
            if not 0 < risk_budget <= 1:
                raise ValueError("allocation member risk budget must be between zero and one")
            if not 0 < member_cap <= min(max_strategy_weight, 0.70):
                raise ValueError("allocation member cap exceeds the strategy limit")
            if role == "satellite" and member_cap > 0.15:
                raise ValueError("a satellite member cap may not exceed 15%")
            governance[version_id] = {
                "role": role,
                "risk_budget": risk_budget,
                "member_cap": member_cap,
            }

        returns: dict[str, pd.Series] = {}
        members: list[dict[str, Any]] = []
        with self.engine.connect() as connection:
            for version_id in version_ids:
                version = self.strategies.get_version(version_id)
                if version["status"] != "approved" or version.get("is_legacy"):
                    raise ValueError("allocations require approved non-legacy strategy versions")
                if (
                    version.get("strategy_type") == "pair"
                    and governance[version_id]["role"] != "satellite"
                ):
                    raise ValueError("pair strategies may only enter an allocation as satellites")
                backtest_row = connection.execute(
                    select(backtest_runs)
                    .where(
                        backtest_runs.c.strategy_version_id == version_id,
                        backtest_runs.c.status == "succeeded",
                        backtest_runs.c.dataset == dataset,
                        backtest_runs.c.is_legacy.is_(False),
                    )
                    .order_by(backtest_runs.c.finished_at.desc())
                    .limit(1)
                ).first()
                if backtest_row is None:
                    raise ValueError(
                        f"strategy version {version_id} has no formal {dataset} backtest"
                    )
                backtest = row_dict(backtest_row)
                strategy_name = connection.execute(
                    select(strategies.c.name).where(strategies.c.id == version["strategy_id"])
                ).scalar_one()
                returns[version_id] = self._daily_returns(backtest)
                members.append(
                    {
                        "strategy_version_id": version_id,
                        "strategy_name": str(strategy_name),
                        "strategy_version": version["version"],
                        "backtest_id": backtest["id"],
                    }
                )
        analysis = analyze_strategy_allocation(
            pd.concat(returns, axis=1, join="inner"),
            method=allocation_method,
            lookback_days=lookback_days,
            target_volatility=target_volatility,
            max_pairwise_correlation=max_pairwise_correlation,
            # Qlib/RD-Agent are the numerical baseline; product-level core,
            # satellite and member caps are enforced immediately below.  Solving
            # the requested risk budgets with those caps inside the numerical
            # optimizer can turn a clear governance rejection into a misleading
            # solver-tolerance failure.
            max_strategy_weight=1.0,
            fixed_weights=fixed_weights,
            risk_budgets={
                version_id: governance[version_id]["risk_budget"]
                for version_id in version_ids
            },
        )
        analysis = self._apply_member_governance(analysis, governance, max_strategy_weight)
        allocation_id = uuid.uuid4().hex
        now = _now()
        decision_date = now.date()
        valid_until = next_decision_date(decision_date, decision_frequency)
        capital = _decimal(total_capital)
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(strategy_allocations).values(
                        id=allocation_id,
                        name=name.strip(),
                        dataset=dataset.strip(),
                        status="draft",
                        is_legacy=False,
                        allocation_method=allocation_method,
                        decision_frequency=decision_frequency,
                        lookback_days=lookback_days,
                        target_volatility=target_volatility,
                        max_pairwise_correlation=max_pairwise_correlation,
                        max_strategy_weight=max_strategy_weight,
                        max_member_drawdown=max_member_drawdown,
                        max_drawdown_reduce=max_drawdown_reduce,
                        max_drawdown_liquidate=max_drawdown_liquidate,
                        total_capital=capital,
                        cash_reserve=capital * _decimal(analysis["cash_weight"]),
                        nav=capital,
                        high_water_mark=capital,
                        analysis_json=analysis,
                        created_by=actor.strip(),
                        created_at=now,
                        updated_at=now,
                    )
                )
                for member in members:
                    evidence = analysis["members"][member["strategy_version_id"]]
                    connection.execute(
                        insert(strategy_allocation_members).values(
                            allocation_id=allocation_id,
                            strategy_version_id=member["strategy_version_id"],
                            backtest_id=member["backtest_id"],
                            target_weight=evidence["target_weight"],
                            role=governance[member["strategy_version_id"]]["role"],
                            risk_budget=governance[member["strategy_version_id"]][
                                "risk_budget"
                            ],
                            member_cap=governance[member["strategy_version_id"]][
                                "member_cap"
                            ],
                            annualized_volatility=evidence["annualized_volatility"],
                            risk_contribution=evidence["risk_contribution"],
                            created_at=now,
                        )
                    )
                self._write_artifact(
                    connection,
                    allocation_id,
                    decision_date=decision_date,
                    analysis=analysis,
                    valid_until=valid_until,
                )
                self._event(
                    connection,
                    allocation_id,
                    event_type="allocation.created",
                    severity="info",
                    rule="low_correlation_risk_budget",
                    observed=analysis["highest_pairwise_correlation"],
                    limit=max_pairwise_correlation,
                    details={"actor": actor.strip(), "members": members},
                )
        except IntegrityError as exc:
            raise ValueError(f"strategy allocation name {name!r} already exists") from exc
        return self.get(allocation_id)

    @staticmethod
    def _apply_member_governance(
        analysis: dict[str, Any],
        governance: dict[str, dict[str, Any]],
        max_strategy_weight: float,
    ) -> dict[str, Any]:
        """Enforce product-level core/satellite member caps on a solved analysis."""

        unscaled = {
            version_id: float(analysis["members"][version_id]["unscaled_weight"])
            for version_id in governance
        }
        for version_id, weight in unscaled.items():
            if weight > governance[version_id]["member_cap"] + 1e-9:
                raise ValueError(
                    f"strategy {version_id} weight exceeds its core/satellite member cap"
                )
        core_weight = sum(
            weight
            for version_id, weight in unscaled.items()
            if governance[version_id]["role"] == "core"
        )
        satellite_weight = sum(
            weight
            for version_id, weight in unscaled.items()
            if governance[version_id]["role"] == "satellite"
        )
        if core_weight < 0.70 - 1e-9 or satellite_weight > 0.30 + 1e-9:
            raise ValueError("allocation must keep at least 70% core and at most 30% satellite")
        analysis["core_satellite"] = {
            "core_weight": core_weight,
            "satellite_weight": satellite_weight,
            "members": governance,
        }
        analysis["solver"]["allocation_governance_wrapper"] = (
            "project_core_satellite_caps_v1"
        )
        analysis["solver"]["governed_max_strategy_weight"] = max_strategy_weight
        return analysis

    @staticmethod
    def _assert_single_active_allocation(connection: Any, allocation_id: str) -> None:
        other = connection.execute(
            select(strategy_allocations.c.id, strategy_allocations.c.name).where(
                strategy_allocations.c.status == "active",
                strategy_allocations.c.id != allocation_id,
            )
        ).first()
        if other is not None:
            raise ValueError(
                f"strategy allocation {other.name!r} is already active; pause it before "
                "activating another one (design 6.10: a single active allocation policy)"
            )

    def _write_artifact(
        self,
        connection: Any,
        allocation_id: str,
        *,
        decision_date: date,
        analysis: dict[str, Any],
        valid_until: date,
    ) -> str:
        """Persist one AllocationArtifact (design 6.10) inside the transaction."""

        member_weights = {
            version_id: float(evidence["target_weight"])
            for version_id, evidence in analysis["members"].items()
        }
        inputs_as_of = pd.Timestamp(analysis["period_end"]).date()
        artifact_hash = _canonical_hash(
            {
                "allocation_id": allocation_id,
                "decision_date": decision_date.isoformat(),
                "inputs_as_of": inputs_as_of.isoformat(),
                "valid_until": valid_until.isoformat(),
                "member_weights": member_weights,
                "covariance_model_version": analysis.get("covariance_model_version"),
                "solver": analysis.get("solver"),
            }
        )
        artifact_id = uuid.uuid4().hex
        connection.execute(
            insert(strategy_allocation_artifacts).values(
                id=artifact_id,
                allocation_id=allocation_id,
                decision_date=decision_date,
                inputs_as_of=inputs_as_of,
                valid_until=valid_until,
                member_weights_json=member_weights,
                analysis_json=analysis,
                artifact_hash=artifact_hash,
                created_at=_now(),
            )
        )
        return artifact_id

    @staticmethod
    def _suspended_member_weights(
        connection: Any, members: Any
    ) -> dict[str, float]:
        """Members whose recommendation portfolio is paused or fully de-risked.

        Suspension only takes effect on the next frozen decision day (budgets
        are never re-estimated mid-cycle); detection is deterministic: a paused
        member portfolio or ``risk_exposure_override <= 0`` marks the member.
        """

        suspended: dict[str, float] = {}
        for member in members:
            portfolio_id = member.recommendation_portfolio_id
            if not portfolio_id:
                continue
            portfolio = connection.execute(
                select(
                    recommendation_portfolios.c.status,
                    recommendation_portfolios.c.risk_exposure_override,
                ).where(recommendation_portfolios.c.id == portfolio_id)
            ).first()
            if portfolio is None:
                continue
            override = portfolio.risk_exposure_override
            if str(portfolio.status) == "paused" or (
                override is not None and float(override) <= 0.0
            ):
                suspended[str(member.strategy_version_id)] = float(member.target_weight)
        return suspended

    def _solve_member_budgets(
        self, connection: Any, allocation: Any, members: Any
    ) -> dict[str, Any]:
        """Load member evidence and solve budgets with the frozen policy method."""

        returns: dict[str, pd.Series] = {}
        governance: dict[str, dict[str, Any]] = {}
        for member in members:
            version_id = str(member.strategy_version_id)
            backtest = connection.execute(
                select(backtest_runs).where(backtest_runs.c.id == member.backtest_id)
            ).first()
            if backtest is None:
                raise ValueError(f"member {version_id} backtest evidence is missing")
            returns[version_id] = self._daily_returns(row_dict(backtest))
            governance[version_id] = {
                "role": str(member.role),
                "risk_budget": float(member.risk_budget),
                "member_cap": float(member.member_cap),
            }
        # A fixed-budget policy re-applies the frozen user weights instead of
        # fabricating a new optimization input (design 6.10 fallback semantics).
        fixed_weights = (
            {
                version_id: float(member.target_weight)
                for version_id, member in (
                    (str(item.strategy_version_id), item) for item in members
                )
            }
            if allocation.allocation_method == "fixed"
            else None
        )
        analysis = analyze_strategy_allocation(
            pd.concat(returns, axis=1, join="inner"),
            method=allocation.allocation_method,
            lookback_days=int(allocation.lookback_days),
            target_volatility=float(allocation.target_volatility),
            max_pairwise_correlation=float(allocation.max_pairwise_correlation),
            max_strategy_weight=1.0,
            fixed_weights=fixed_weights,
            risk_budgets={
                version_id: governance[version_id]["risk_budget"] for version_id in governance
            },
        )
        return self._apply_member_governance(
            analysis, governance, float(allocation.max_strategy_weight)
        )

    def _resolve_with_suspended(
        self,
        connection: Any,
        allocation: Any,
        members: Any,
        suspended: dict[str, float],
        decision_date: date,
    ) -> dict[str, Any]:
        """Decision-day re-solve with suspended members (design 6.10 fallback).

        The policy method re-solves on the active set when at least two active
        members remain; a single active member keeps its frozen budget as the
        simple baseline (no re-estimation is possible with fewer than two
        return series), and with no active member everything falls to cash.
        The suspended budget always moves to cash via
        ``renormalize_budgets_for_suspended`` — never redistributed as
        leverage, never re-estimated outside the frozen decision day.
        """

        previous_weights = {
            str(member.strategy_version_id): float(member.target_weight) for member in members
        }
        active = [
            member
            for member in members
            if str(member.strategy_version_id) not in suspended
        ]
        if len(active) >= 2:
            analysis = self._solve_member_budgets(connection, allocation, active)
        else:
            fallback: dict[str, Any] = {
                "method": str(allocation.allocation_method),
                "period_start": decision_date.isoformat(),
                "period_end": decision_date.isoformat(),
                "covariance_model_version": None,
                "solver": {"success": True},
            }
            if len(active) == 1:
                version_id = str(active[0].strategy_version_id)
                weight = previous_weights[version_id]
                fallback.update(
                    {
                        "fallback_reason": "single_active_member_keeps_frozen_budget",
                        "solver": {"success": True, "engine": "frozen_budget_fallback_v1"},
                        "members": {
                            version_id: {
                                "unscaled_weight": weight,
                                "target_weight": weight,
                                "annualized_volatility": None,
                                "risk_contribution": None,
                                "risk_budget": None,
                            }
                        },
                    }
                )
            else:
                fallback.update(
                    {
                        "fallback_reason": "no_active_members_all_cash",
                        "solver": {"success": True, "engine": "cash_fallback_v1"},
                        "members": {},
                    }
                )
            analysis = fallback
        return renormalize_budgets_for_suspended(
            analysis, previous_weights=previous_weights, suspended=suspended
        )

    def _resolve_artifact(
        self, connection: Any, allocation: Any, decision_date: date
    ) -> None:
        """Re-solve member budgets on a frozen decision day and apply them once."""

        members = connection.execute(
            select(strategy_allocation_members).where(
                strategy_allocation_members.c.allocation_id == allocation.id
            )
        ).all()
        if not members:
            raise ValueError("allocation has no members to re-solve")
        suspended = self._suspended_member_weights(connection, members)
        if suspended:
            analysis = self._resolve_with_suspended(
                connection, allocation, members, suspended, decision_date
            )
        else:
            analysis = self._solve_member_budgets(connection, allocation, members)
        frequency = str(getattr(allocation, "decision_frequency", None) or "monthly")
        valid_until = next_decision_date(decision_date, frequency)
        for member in members:
            version_id = str(member.strategy_version_id)
            connection.execute(
                update(strategy_allocation_members)
                .where(
                    strategy_allocation_members.c.allocation_id == allocation.id,
                    strategy_allocation_members.c.strategy_version_id == version_id,
                )
                .values(target_weight=analysis["members"][version_id]["target_weight"])
            )
        connection.execute(
            update(strategy_allocations)
            .where(strategy_allocations.c.id == allocation.id)
            .values(
                cash_reserve=_decimal(allocation.total_capital)
                * _decimal(analysis["cash_weight"]),
                updated_at=_now(),
            )
        )
        self._write_artifact(
            connection,
            str(allocation.id),
            decision_date=decision_date,
            analysis=analysis,
            valid_until=valid_until,
        )
        self._event(
            connection,
            str(allocation.id),
            event_type="allocation.artifact_resolved",
            severity="info",
            rule="frozen_decision_day",
            details={
                "decision_date": decision_date.isoformat(),
                "valid_until": valid_until.isoformat(),
                **(
                    {"suspended_members": suspended, "renormalization": analysis["renormalization"]}
                    if suspended
                    else {}
                ),
            },
        )

    def approve(self, allocation_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        if not actor.strip() or len(reason.strip()) < 10:
            raise ValueError("actor and a meaningful approval reason are required")
        now = _now()
        with self.engine.begin() as connection:
            allocation = connection.execute(
                select(strategy_allocations)
                .where(strategy_allocations.c.id == allocation_id)
                .with_for_update()
            ).first()
            if allocation is None:
                raise KeyError(allocation_id)
            if allocation.is_legacy:
                raise ValueError("legacy paper-backed allocations are read-only")
            if allocation.status != "draft":
                raise ValueError("only draft strategy allocations may be approved")
            if actor.strip() == allocation.created_by:
                raise ValueError("strategy allocation approval requires a second operator")
            self._assert_single_active_allocation(connection, allocation_id)
            analysis = dict(allocation.analysis_json or {})
            if analysis.get("covariance_model_version") != COVARIANCE_MODEL_VERSION:
                raise ValueError("allocation covariance evidence is obsolete")
            if allocation.allocation_method == "risk_parity":
                solver = analysis.get("solver") or {}
                contributions = [
                    float(item.get("risk_contribution", -1.0))
                    for item in (analysis.get("members") or {}).values()
                ]
                if (
                    solver.get("success") is not True
                    or solver.get("maximum_risk_budget_error") is None
                    or float(solver["maximum_risk_budget_error"]) > 0.02
                    or not contributions
                    or min(contributions) < -1e-8
                ):
                    raise ValueError(
                        "risk parity solver evidence is invalid; use inverse_volatility or fixed"
                    )
            members = connection.execute(
                select(strategy_allocation_members).where(
                    strategy_allocation_members.c.allocation_id == allocation_id
                )
            ).all()
            certified_evidence: dict[str, Any] = {}
            member_by_version = {
                str(member.strategy_version_id): member for member in members
            }
            for member in members:
                evidence = self._certified_simulation_evidence(
                    connection, str(member.strategy_version_id)
                )
                if evidence is None:
                    raise ValueError(
                        "allocation approval requires five independently reviewed and "
                        "certified simulation NAV days "
                        f"for strategy {member.strategy_version_id}"
                    )
                evidence["target_weight"] = float(member.target_weight)
                certified_evidence[str(member.strategy_version_id)] = evidence
            analysis["approval_simulation_evidence"] = certified_evidence
            date_sets = {
                tuple(item["trade_date"] for item in evidence["nav_rows"])
                for evidence in certified_evidence.values()
            }
            if len(date_sets) != 1:
                raise ValueError("allocation member simulation NAV dates are not aligned")
            aligned_dates = next(iter(date_sets))
            total_capital = float(allocation.total_capital)
            cash_reserve = float(allocation.cash_reserve)
            combined_nav = [
                cash_reserve
                + sum(
                    total_capital
                    * float(member_by_version[version_id].target_weight)
                    * float(evidence["nav_rows"][index]["nav"])
                    / float(evidence["initial_cash"])
                    for version_id, evidence in certified_evidence.items()
                )
                for index in range(len(aligned_dates))
            ]
            combined_peak = max(combined_nav)
            analysis["approval_simulation_nav"] = {
                "performance_certified": True,
                "review_status": "independently_reviewed",
                "reviewed_days": len(aligned_dates),
                "period_start": aligned_dates[0],
                "period_end": aligned_dates[-1],
                "latest_nav": combined_nav[-1],
                "drawdown": combined_nav[-1] / combined_peak - 1.0,
                "evidence_sha256": _canonical_hash(certified_evidence),
            }
            for member in members:
                version = self.strategies.get_version(str(member.strategy_version_id))
                strategy_name = connection.execute(
                    select(strategies.c.name).where(strategies.c.id == version["strategy_id"])
                ).scalar_one()
                initial_value = _decimal(allocation.total_capital) * _decimal(member.target_weight)
                if initial_value < Decimal("100000"):
                    raise ValueError(f"allocated value for {strategy_name} is below 100000")
                if version.get("strategy_type") == "pair":
                    continue
                portfolio_id = uuid.uuid4().hex
                connection.execute(
                    insert(recommendation_portfolios).values(
                        id=portfolio_id,
                        name=f"{allocation.name} / {strategy_name} v{version['version']}"[:150],
                        strategy_version_id=member.strategy_version_id,
                        dataset=allocation.dataset,
                        status="active",
                        base_currency="CNY",
                        hypothetical_initial_value=initial_value,
                        risk_exposure_override=1.0,
                        recommendation_scope="allocation_member",
                        created_by=allocation.created_by,
                        created_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    update(strategy_allocation_members)
                    .where(
                        strategy_allocation_members.c.allocation_id == allocation_id,
                        strategy_allocation_members.c.strategy_version_id
                        == member.strategy_version_id,
                    )
                    .values(recommendation_portfolio_id=portfolio_id)
                )
            connection.execute(
                update(strategy_allocations)
                .where(strategy_allocations.c.id == allocation_id)
                .values(
                    status="active",
                    approved_by=actor.strip(),
                    approval_reason=reason.strip(),
                    approved_at=now,
                    analysis_json=analysis,
                    updated_at=now,
                )
            )
            self._event(
                connection,
                allocation_id,
                event_type="allocation.approved",
                severity="info",
                rule="four_eyes_approval",
                details={"actor": actor.strip(), "reason": reason.strip()},
            )
        return self.get(allocation_id)

    @staticmethod
    def _certified_simulation_evidence(
        connection: Any, strategy_version_id: str
    ) -> dict[str, Any] | None:
        simulations = connection.execute(
            select(simulation_portfolios)
            .where(
                simulation_portfolios.c.source_type == "strategy_version",
                simulation_portfolios.c.source_id == strategy_version_id,
                simulation_portfolios.c.status.in_(["active", "paused"]),
            )
            .order_by(simulation_portfolios.c.updated_at.desc())
        ).all()
        for simulation in simulations:
            rows = connection.execute(
                select(simulation_nav)
                .where(
                    simulation_nav.c.portfolio_id == simulation.id,
                    simulation_nav.c.performance_certified.is_(True),
                    simulation_nav.c.reviewed_at.is_not(None),
                    simulation_nav.c.nav_scope == "member_ledger",
                )
                .order_by(simulation_nav.c.trade_date.desc())
                .limit(5)
            ).all()
            if len(rows) == 5:
                return {
                    "simulation_portfolio_id": str(simulation.id),
                    "execution_contract_hash": str(simulation.execution_contract_hash),
                    "initial_cash": float(simulation.initial_cash),
                    "first_trade_date": rows[-1].trade_date.isoformat(),
                    "last_trade_date": rows[0].trade_date.isoformat(),
                    "reviewed_days": 5,
                    "latest_nav": float(rows[0].nav),
                    "review_audit_sha256": _canonical_hash(
                        {
                            row.trade_date.isoformat(): {
                                "reviewed_by": str(row.reviewed_by),
                                "reviewed_at": row.reviewed_at.isoformat(),
                                "review_evidence_sha256": str(
                                    row.review_evidence_sha256
                                ),
                                "review_note": str(row.review_note),
                            }
                            for row in reversed(rows)
                        }
                    ),
                    "nav_rows": [
                        {
                            "trade_date": row.trade_date.isoformat(),
                            "nav": float(row.nav),
                            "daily_return": float(row.daily_return),
                            "drawdown": float(row.drawdown),
                            "reviewed_by": str(row.reviewed_by),
                            "reviewed_at": row.reviewed_at.isoformat(),
                            "review_evidence_sha256": str(
                                row.review_evidence_sha256
                            ),
                            "review_note": str(row.review_note),
                        }
                        for row in reversed(rows)
                    ],
                }
        return None

    def set_status(self, allocation_id: str, status: str, *, actor: str) -> dict[str, Any]:
        if status not in {"active", "paused"}:
            raise ValueError("strategy allocation status must be active or paused")
        if len(actor.strip()) < 2:
            raise ValueError("a responsible actor is required")
        now = _now()
        with self.engine.begin() as connection:
            allocation = connection.execute(
                select(strategy_allocations).where(strategy_allocations.c.id == allocation_id)
            ).first()
            if allocation is None:
                raise KeyError(allocation_id)
            if allocation.is_legacy:
                raise ValueError("legacy paper-backed allocations are read-only")
            if allocation.status == "draft":
                raise ValueError("draft strategy allocations cannot be operated")
            if status == "active":
                unresolved = connection.execute(
                    select(strategy_allocation_events.c.id)
                    .where(
                        strategy_allocation_events.c.allocation_id == allocation_id,
                        strategy_allocation_events.c.severity == "critical",
                        strategy_allocation_events.c.status.in_(["open", "acknowledged"]),
                    )
                    .limit(1)
                ).first()
                if unresolved is not None:
                    raise ValueError("critical allocation events must be resolved first")
                self._assert_single_active_allocation(connection, allocation_id)
            portfolio_ids = self._portfolio_ids(connection, allocation_id)
            member_version_ids = list(
                connection.scalars(
                    select(strategy_allocation_members.c.strategy_version_id).where(
                        strategy_allocation_members.c.allocation_id == allocation_id
                    )
                )
            )
            connection.execute(
                update(recommendation_portfolios)
                .where(recommendation_portfolios.c.id.in_(portfolio_ids))
                .values(
                    status=status,
                    risk_exposure_override=1.0 if status == "active" else 0.0,
                    updated_at=now,
                )
            )
            connection.execute(
                update(simulation_portfolios)
                .where(
                    (
                        (simulation_portfolios.c.source_type == "recommendation")
                        & simulation_portfolios.c.source_id.in_(portfolio_ids)
                    )
                    | (
                        (simulation_portfolios.c.source_type == "strategy_version")
                        & simulation_portfolios.c.source_id.in_(member_version_ids)
                    )
                    | (
                        (simulation_portfolios.c.source_type == "allocation")
                        & (simulation_portfolios.c.source_id == allocation_id)
                    )
                )
                .values(status=status, updated_at=now)
            )
            connection.execute(
                update(strategy_allocations)
                .where(strategy_allocations.c.id == allocation_id)
                .values(status=status, updated_at=now)
            )
            self._event(
                connection,
                allocation_id,
                event_type=f"allocation.{status}",
                severity="info",
                rule="operator_state_change",
                details={"actor": actor.strip()},
            )
        return self.get(allocation_id)

    def refresh(
        self,
        allocation_id: str,
        *,
        actor: str = "allocation-engine",
    ) -> dict[str, Any]:
        producer = actor.strip()
        if len(producer) < 2:
            raise ValueError("allocation NAV producer is required")
        now = _now()
        with self.engine.begin() as connection:
            allocation = connection.execute(
                select(strategy_allocations)
                .where(strategy_allocations.c.id == allocation_id)
                .with_for_update()
            ).first()
            if allocation is None:
                raise KeyError(allocation_id)
            if allocation.is_legacy:
                raise ValueError("legacy paper-backed allocations are read-only")
            if allocation.status not in {"active", "risk_reduction_pending", "liquidation_pending"}:
                raise ValueError("strategy allocation is not refreshable")
            # Frozen-decision-day gate (design 6.10/8.1): member budgets are
            # only re-solved when the current artifact has expired; every
            # other refresh reuses the still-valid artifact and never
            # re-estimates risk on the fly.
            if allocation.status == "active":
                current_artifact = connection.execute(
                    select(strategy_allocation_artifacts)
                    .where(
                        strategy_allocation_artifacts.c.allocation_id == allocation_id
                    )
                    .order_by(
                        strategy_allocation_artifacts.c.decision_date.desc(),
                        strategy_allocation_artifacts.c.created_at.desc(),
                    )
                    .limit(1)
                ).first()
                if current_artifact is None or now.date() >= current_artifact.valid_until:
                    try:
                        self._resolve_artifact(connection, allocation, now.date())
                    except ValueError as exc:
                        # A failed re-solve never blocks NAV recording: the
                        # previous budgets stay in force and the next refresh
                        # retries the decision day.
                        self._event(
                            connection,
                            allocation_id,
                            event_type="allocation.artifact_reschedule_failed",
                            severity="info",
                            rule="frozen_decision_day",
                            details={"error": str(exc)},
                        )
                    allocation = connection.execute(
                        select(strategy_allocations)
                        .where(strategy_allocations.c.id == allocation_id)
                        .with_for_update()
                    ).first()
            members = connection.execute(
                select(strategy_allocation_members).where(
                    strategy_allocation_members.c.allocation_id == allocation_id
                )
            ).all()
            latest: dict[str, dict[str, Any]] = {}
            for member in members:
                portfolio_id = member.recommendation_portfolio_id
                simulation = connection.execute(
                    select(simulation_portfolios).where(
                        simulation_portfolios.c.source_type == "strategy_version",
                        simulation_portfolios.c.source_id == member.strategy_version_id,
                    )
                    .order_by(simulation_portfolios.c.updated_at.desc())
                    .limit(1)
                ).first()
                if simulation is None and portfolio_id:
                    simulation = connection.execute(
                        select(simulation_portfolios).where(
                            simulation_portfolios.c.source_type == "recommendation",
                            simulation_portfolios.c.source_id == portfolio_id,
                        )
                        .order_by(simulation_portfolios.c.updated_at.desc())
                        .limit(1)
                    ).first()
                if simulation is None:
                    return {
                        "id": allocation_id,
                        "status": allocation.status,
                        "refresh_status": "waiting_for_simulation_portfolio",
                    }
                nav_row = connection.execute(
                    select(simulation_nav)
                    .where(simulation_nav.c.portfolio_id == simulation.id)
                    .order_by(simulation_nav.c.trade_date.desc())
                    .limit(1)
                ).first()
                if nav_row is None:
                    return {
                        "id": allocation_id,
                        "status": allocation.status,
                        "refresh_status": "waiting_for_member_nav",
                    }
                if not nav_row.performance_certified:
                    return {
                        "id": allocation_id,
                        "status": allocation.status,
                        "refresh_status": "waiting_for_certified_simulation_nav",
                    }
                latest[str(member.strategy_version_id)] = {
                    "nav": nav_row,
                    "simulation": simulation,
                    "target_weight": _decimal(member.target_weight),
                }
            dates = {item["nav"].trade_date for item in latest.values()}
            if len(dates) != 1:
                return {
                    "id": allocation_id,
                    "status": allocation.status,
                    "refresh_status": "waiting_for_aligned_trade_date",
                }
            trade_date = next(iter(dates))
            if connection.execute(
                select(strategy_allocation_nav.c.trade_date).where(
                    strategy_allocation_nav.c.allocation_id == allocation_id,
                    strategy_allocation_nav.c.trade_date == trade_date,
                )
            ).first():
                return {
                    "id": allocation_id,
                    "status": allocation.status,
                    "refresh_status": "already_recorded",
                }
            total_capital = _decimal(allocation.total_capital)
            member_nav: dict[str, float] = {}
            for version_id, item in latest.items():
                initial_cash = _decimal(item["simulation"].initial_cash)
                if initial_cash <= 0:
                    raise ValueError("member simulation initial cash is invalid")
                normalized_value = (
                    total_capital
                    * item["target_weight"]
                    * _decimal(item["nav"].nav)
                    / initial_cash
                )
                member_nav[version_id] = float(normalized_value)
            nav = _decimal(allocation.cash_reserve) + sum(
                (_decimal(value) for value in member_nav.values()), Decimal("0")
            )
            previous = connection.execute(
                select(strategy_allocation_nav.c.nav)
                .where(strategy_allocation_nav.c.allocation_id == allocation_id)
                .order_by(strategy_allocation_nav.c.trade_date.desc())
                .limit(1)
            ).scalar_one_or_none()
            previous_nav = _decimal(previous or allocation.total_capital)
            daily_return = float(nav / previous_nav - 1) if previous_nav else 0.0
            high_water_mark = max(_decimal(allocation.high_water_mark), nav)
            drawdown = float(nav / high_water_mark - 1) if high_water_mark else 0.0
            history = list(
                connection.execute(
                    select(strategy_allocation_nav.c.daily_return)
                    .where(strategy_allocation_nav.c.allocation_id == allocation_id)
                    .order_by(strategy_allocation_nav.c.trade_date.desc())
                    .limit(59)
                ).scalars()
            )
            history.append(daily_return)
            volatility = (
                float(np.std(history, ddof=1) * np.sqrt(252.0))
                if len(history) > 1
                else 0.0
            )
            member_weights = {
                key: float(_decimal(value) / nav) if nav else 0.0
                for key, value in member_nav.items()
            }
            drawdown_loss = abs(min(drawdown, 0.0))
            if drawdown_loss >= float(allocation.max_drawdown_liquidate):
                if not has_open_allocation_gate(
                    connection,
                    allocation_id=allocation_id,
                    rule=ALLOCATION_LIQUIDATION_RULE,
                ):
                    self._event(
                        connection,
                        allocation_id,
                        event_type="allocation_circuit_breaker",
                        severity="critical",
                        rule=ALLOCATION_LIQUIDATION_RULE,
                        observed=drawdown_loss,
                        limit=float(allocation.max_drawdown_liquidate),
                        details={
                            "trade_date": trade_date.isoformat(),
                            "risk_state": "liquidation",
                            "risk_exposure_override": 0.0,
                        },
                    )
            elif drawdown_loss >= float(allocation.max_drawdown_reduce):
                if not has_open_allocation_gate(
                    connection,
                    allocation_id=allocation_id,
                    rule=ALLOCATION_REDUCTION_RULE,
                ):
                    self._event(
                        connection,
                        allocation_id,
                        event_type="allocation_circuit_breaker",
                        severity="critical",
                        rule=ALLOCATION_REDUCTION_RULE,
                        observed=drawdown_loss,
                        limit=float(allocation.max_drawdown_reduce),
                        details={
                            "trade_date": trade_date.isoformat(),
                            "risk_state": "risk_reduction",
                            "risk_exposure_override": 0.5,
                        },
                    )
            allocation_gate = load_allocation_risk_state(connection, allocation_id)
            override = float(allocation_gate["risk_exposure_override"])
            if override <= 0.0:
                next_status = "liquidation_pending"
            elif override < 1.0:
                next_status = "risk_reduction_pending"
            else:
                next_status = str(allocation.status)
            for member in members:
                version_id = str(member.strategy_version_id)
                portfolio_id = (
                    str(member.recommendation_portfolio_id)
                    if member.recommendation_portfolio_id
                    else None
                )
                member_loss = abs(min(float(latest[version_id]["nav"].drawdown), 0.0))
                if member_loss >= float(allocation.max_member_drawdown):
                    if not has_open_member_gate(
                        connection,
                        allocation_id=allocation_id,
                        strategy_version_id=version_id,
                        recommendation_portfolio_id=portfolio_id,
                    ):
                        self._event(
                            connection,
                            allocation_id,
                            recommendation_portfolio_id=portfolio_id,
                            event_type="member_circuit_breaker",
                            severity="critical",
                            rule="max_member_drawdown",
                            observed=member_loss,
                            limit=float(allocation.max_member_drawdown),
                            details={
                                "trade_date": trade_date.isoformat(),
                                "strategy_version_id": version_id,
                                "risk_state": "pause_new_risk",
                            },
                        )
                if portfolio_id:
                    connection.execute(
                        update(recommendation_portfolios)
                        .where(recommendation_portfolios.c.id == portfolio_id)
                        .values(risk_exposure_override=override, updated_at=now)
                    )
            connection.execute(
                insert(strategy_allocation_nav).values(
                    allocation_id=allocation_id,
                    trade_date=trade_date,
                    nav=nav,
                    daily_return=daily_return,
                    annualized_volatility=volatility,
                    drawdown=drawdown,
                    member_nav_json=member_nav,
                    member_weights_json=member_weights,
                    created_at=now,
                )
            )
            connection.execute(
                update(strategy_allocations)
                .where(strategy_allocations.c.id == allocation_id)
                .values(
                    status=next_status,
                    nav=nav,
                    high_water_mark=high_water_mark,
                    updated_at=now,
                )
            )
            self._sync_allocation_simulation_nav(
                connection,
                allocation=allocation,
                trade_date=trade_date,
                allocation_nav=nav,
                daily_return=daily_return,
                drawdown=drawdown,
                produced_by=producer,
                now=now,
            )
        result = self.get(allocation_id)
        result["refresh_status"] = "recorded"
        return result

    @staticmethod
    def _sync_allocation_simulation_nav(
        connection: Any,
        *,
        allocation: Any,
        trade_date: Any,
        allocation_nav: Decimal,
        daily_return: float,
        drawdown: float,
        produced_by: str,
        now: datetime,
    ) -> None:
        accounts = connection.execute(
            select(simulation_portfolios).where(
                simulation_portfolios.c.source_type == "allocation",
                simulation_portfolios.c.source_id == allocation.id,
                simulation_portfolios.c.status.in_(["active", "paused"]),
            )
        ).all()
        total_capital = _decimal(allocation.total_capital)
        if total_capital <= 0:
            raise ValueError("allocation total capital is invalid")
        allocation_ratio = allocation_nav / total_capital
        reserve_ratio = _decimal(allocation.cash_reserve) / total_capital
        for account in accounts:
            account_nav = _decimal(account.initial_cash) * allocation_ratio
            account_cash = _decimal(account.initial_cash) * reserve_ratio
            account_market_value = account_nav - account_cash
            connection.execute(
                pg_insert(simulation_nav)
                .values(
                    portfolio_id=account.id,
                    trade_date=trade_date,
                    cash=account_cash,
                    market_value=account_market_value,
                    nav=account_nav,
                    daily_return=daily_return,
                    drawdown=drawdown,
                    market_date=trade_date,
                    has_stale_prices=False,
                    status="healthy",
                    performance_certified=True,
                    nav_scope="aggregate_view",
                    produced_by=produced_by,
                    created_at=now,
                )
                .on_conflict_do_nothing(
                    index_elements=[
                        simulation_nav.c.portfolio_id,
                        simulation_nav.c.trade_date,
                    ]
                )
            )
            connection.execute(
                update(simulation_portfolios)
                .where(simulation_portfolios.c.id == account.id)
                .values(
                    cash=account_cash,
                    nav=account_nav,
                    high_water_mark=max(_decimal(account.high_water_mark), account_nav),
                    updated_at=now,
                )
            )
    def acknowledge_event(self, allocation_id: str, event_id: int, *, actor: str) -> dict[str, Any]:
        return self._update_event(allocation_id, event_id, actor=actor, resolution=None)

    def resolve_event(
        self, allocation_id: str, event_id: int, *, actor: str, reason: str
    ) -> dict[str, Any]:
        if len(reason.strip()) < 10:
            raise ValueError("risk event resolution reason must be meaningful")
        return self._update_event(allocation_id, event_id, actor=actor, resolution=reason)

    def _update_event(
        self,
        allocation_id: str,
        event_id: int,
        *,
        actor: str,
        resolution: str | None,
    ) -> dict[str, Any]:
        if len(actor.strip()) < 2:
            raise ValueError("a responsible actor is required")
        now = _now()
        with self.engine.begin() as connection:
            event = connection.execute(
                select(strategy_allocation_events)
                .where(
                    strategy_allocation_events.c.id == event_id,
                    strategy_allocation_events.c.allocation_id == allocation_id,
                )
                .with_for_update()
            ).first()
            if event is None:
                raise KeyError(event_id)
            if event.status == "resolved":
                raise ValueError("risk event is already resolved")
            values: dict[str, Any] = {
                "status": "resolved" if resolution is not None else "acknowledged",
                "acknowledged_by": actor.strip(),
                "acknowledged_at": now,
            }
            if resolution is not None:
                values.update(
                    resolved_by=actor.strip(), resolved_at=now, resolution_reason=resolution.strip()
                )
            connection.execute(
                update(strategy_allocation_events)
                .where(strategy_allocation_events.c.id == event_id)
                .values(**values)
            )
        return self.get_event(allocation_id, event_id)

    def get_event(self, allocation_id: str, event_id: int) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(strategy_allocation_events).where(
                    strategy_allocation_events.c.id == event_id,
                    strategy_allocation_events.c.allocation_id == allocation_id,
                )
            ).first()
        if row is None:
            raise KeyError(event_id)
        return self._event_row(row)

    def get(self, allocation_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            allocation = connection.execute(
                select(strategy_allocations).where(strategy_allocations.c.id == allocation_id)
            ).first()
            if allocation is None:
                raise KeyError(allocation_id)
            result = row_dict(allocation)
            result["analysis"] = result.pop("analysis_json")
            result["members"] = [
                row_dict(item)
                for item in connection.execute(
                    select(strategy_allocation_members).where(
                        strategy_allocation_members.c.allocation_id == allocation_id
                    )
                )
            ]
            result["nav_history"] = [
                row_dict(item)
                for item in connection.execute(
                    select(strategy_allocation_nav)
                    .where(strategy_allocation_nav.c.allocation_id == allocation_id)
                    .order_by(strategy_allocation_nav.c.trade_date.desc())
                    .limit(500)
                )
            ]
            result["events"] = [
                self._event_row(item)
                for item in connection.execute(
                    select(strategy_allocation_events)
                    .where(strategy_allocation_events.c.allocation_id == allocation_id)
                    .order_by(strategy_allocation_events.c.created_at.desc())
                    .limit(500)
                )
            ]
            artifacts = []
            for item in connection.execute(
                select(strategy_allocation_artifacts)
                .where(strategy_allocation_artifacts.c.allocation_id == allocation_id)
                .order_by(
                    strategy_allocation_artifacts.c.decision_date.desc(),
                    strategy_allocation_artifacts.c.created_at.desc(),
                )
                .limit(50)
            ):
                artifact = row_dict(item)
                artifact["member_weights"] = artifact.pop("member_weights_json")
                artifact["analysis"] = artifact.pop("analysis_json")
                artifacts.append(artifact)
            result["artifacts"] = artifacts
        return result

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            ids = [
                str(item)
                for item in connection.execute(
                    select(strategy_allocations.c.id)
                    .order_by(strategy_allocations.c.updated_at.desc())
                    .limit(limit)
                ).scalars()
            ]
        return [self.get(item) for item in ids]

    def member_risk_state(self, strategy_version_id: str) -> dict[str, Any]:
        """Return the original 8% member-only new-risk gate."""

        normalized_id = strategy_version_id.strip()
        if not normalized_id:
            raise ValueError("strategy version id is required")
        with self.engine.connect() as connection:
            return load_member_risk_state(connection, normalized_id)

    def strategy_risk_state(self, strategy_version_id: str) -> dict[str, Any]:
        """Return member and allocation circuit breakers for one strategy version."""

        normalized_id = strategy_version_id.strip()
        if not normalized_id:
            raise ValueError("strategy version id is required")
        with self.engine.connect() as connection:
            return load_strategy_risk_state(connection, normalized_id)

    def refresh_for_portfolio(
        self,
        portfolio_id: str,
        *,
        actor: str = "allocation-engine",
    ) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            allocation_ids = list(
                connection.execute(
                    select(strategy_allocation_members.c.allocation_id).where(
                        strategy_allocation_members.c.recommendation_portfolio_id == portfolio_id
                    )
                ).scalars()
            )
        return [self.refresh(str(item), actor=actor) for item in allocation_ids]

    def refresh_for_simulation_source(
        self,
        source_type: str,
        source_id: str,
        *,
        actor: str = "simulation-worker",
    ) -> list[dict[str, Any]]:
        if source_type == "recommendation":
            return self.refresh_for_portfolio(source_id, actor=actor)
        if source_type == "allocation":
            try:
                return [self.refresh(source_id, actor=actor)]
            except ValueError:
                return []
        if source_type != "strategy_version":
            return []
        with self.engine.connect() as connection:
            allocation_ids = list(
                connection.scalars(
                    select(strategy_allocation_members.c.allocation_id).where(
                        strategy_allocation_members.c.strategy_version_id == source_id
                    )
                )
            )
        results: list[dict[str, Any]] = []
        for allocation_id in allocation_ids:
            try:
                results.append(self.refresh(str(allocation_id), actor=actor))
            except ValueError:
                continue
        return results

    @staticmethod
    def _portfolio_ids(connection: Any, allocation_id: str) -> list[str]:
        rows = connection.execute(
            select(strategy_allocation_members.c.recommendation_portfolio_id).where(
                strategy_allocation_members.c.allocation_id == allocation_id
            )
        ).scalars()
        return [str(item) for item in rows if item]

    @staticmethod
    def _event(
        connection: Any,
        allocation_id: str,
        *,
        event_type: str,
        severity: str,
        rule: str,
        observed: float | None = None,
        limit: float | None = None,
        recommendation_portfolio_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        connection.execute(
            insert(strategy_allocation_events).values(
                allocation_id=allocation_id,
                recommendation_portfolio_id=recommendation_portfolio_id,
                severity=severity,
                event_type=event_type,
                rule=rule,
                observed=observed,
                limit_value=limit,
                status="open" if severity == "critical" else "resolved",
                details_json=details or {},
                created_at=_now(),
            )
        )

    @staticmethod
    def _event_row(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["details"] = result.pop("details_json")
        return result
