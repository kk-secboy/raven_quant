"""Single-account orchestration for the three governed strategy horizons."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select, text, update

from quant_data.config import Settings
from quant_data.database import (
    backtest_runs,
    recommendation_portfolios,
    recommendation_snapshots,
    simulation_portfolios,
    strategy_allocation_members,
    strategy_allocations,
    strategy_versions,
)

from .account_netting import AccountNettingStore
from .allocation_store import AllocationStore
from .cost_model import COST_SCHEDULE_VERSION
from .investor_profile import (
    InvestorSimulationProfileStore,
    investor_profile_permission,
)
from .market_rules import lot_floor, order_unit_rules
from .recommendation_actions import plan_account_actions
from .research_horizon import LONG_1_3Y, SHORT_1_5D, SWING_1_6M
from .services import list_qlib_datasets
from .simulation_order_state import OPEN_STATUSES
from .simulation_store import SimulationStore

THREE_HORIZON_ACCOUNT_VERSION = "three-horizon-account-v1"
THREE_HORIZON_PRIMARY_SIMULATION_ACTOR = "three-horizon-account"
THREE_HORIZON_ACCOUNT_LOCK_KEY = 7_215_083_172_040_011
HORIZON_WEIGHTS = {
    SHORT_1_5D: 0.20,
    SWING_1_6M: 0.40,
    LONG_1_3Y: 0.40,
}


def _active_horizon_fixed_weights(
    versions: dict[str, str], *, max_gross_exposure: float
) -> dict[str, float]:
    """Reserve missing horizon sleeves as cash without renormalizing them."""

    gross = float(max_gross_exposure)
    if not 0 < gross <= 1:
        raise ValueError("three-horizon maximum gross exposure must be in (0, 1]")
    unknown = set(versions).difference(HORIZON_WEIGHTS)
    if unknown:
        raise ValueError(f"unknown three-horizon sleeves: {sorted(unknown)}")
    return {
        versions[horizon]: HORIZON_WEIGHTS[horizon] * gross
        for horizon in HORIZON_WEIGHTS
        if horizon in versions
    }


def _select_current_three_horizon_dataset(
    *,
    formal_datasets: dict[str, str],
    formal_lineages: dict[str, str],
    qlib_datasets: list[dict[str, Any]],
) -> str:
    """Choose the daily snapshot without weakening formal evidence identity."""

    if formal_lineages and len(formal_lineages) != len(formal_datasets):
        raise ValueError(
            "three-horizon formal dataset lineage evidence is incomplete: "
            + str(sorted(formal_lineages))
        )
    if formal_lineages:
        unique_lineages = set(formal_lineages.values())
        if len(unique_lineages) != 1:
            raise ValueError(
                "three-horizon formal datasets do not share one governed lineage: "
                + str(formal_lineages)
            )
        lineage = next(iter(unique_lineages))
        compatible = [
            item
            for item in qlib_datasets
            if item.get("ready")
            and item.get("reproducible")
            and str(item.get("lineage_id") or "") == lineage
        ]
        if not compatible:
            raise ValueError(
                "three-horizon governed dataset lineage has no usable current snapshot"
            )
        # ``list_qlib_datasets`` is newest-first.  Formal evidence remains
        # bound to its original immutable snapshot while advice advances on
        # the latest compatible publication from the same lineage.
        return str(compatible[0]["name"])
    unique = set(formal_datasets.values())
    if len(unique) != 1:
        raise ValueError(
            "three-horizon legacy evidence requires one identical daily dataset: "
            + str(formal_datasets)
        )
    return next(iter(unique))


class ThreeHorizonAccountService:
    """Create and advance the sole 20/40/40 recommendation account.

    Research and forward paper accounts remain isolated.  This service starts
    as soon as one horizon has ``recommendation_enabled`` authority. Missing
    sleeves remain cash at their frozen 20/40/40 budget; later horizons replace
    the active allocation atomically through the existing Allocation,
    AccountNetting and Simulation stores instead of creating another ledger.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.allocations = AllocationStore(settings.database_url)
        self.netting = AccountNettingStore(settings.database_url)
        self.simulations = SimulationStore(settings.database_url)
        self.profiles = InvestorSimulationProfileStore(settings.database_url)
        self.engine = self.allocations.engine

    def tick(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Serialize the multi-transaction cutover saga across scheduler replicas."""

        with self.engine.connect() as lock_connection:
            lock_connection.execute(
                text("SELECT pg_advisory_lock(:lock_key)"),
                {"lock_key": THREE_HORIZON_ACCOUNT_LOCK_KEY},
            )
            try:
                return self._tick_locked(now=now)
            finally:
                lock_connection.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": THREE_HORIZON_ACCOUNT_LOCK_KEY},
                )

    def _tick_locked(self, *, now: datetime | None = None) -> dict[str, Any]:
        current = now or datetime.now(UTC)
        profile = self.profiles.get_active("primary")
        if profile is None:
            return {"status": "onboarding_required", "advanced": False}
        versions = self._active_versions()
        missing = [horizon for horizon in HORIZON_WEIGHTS if horizon not in versions]
        if not versions:
            return {
                "status": "waiting_for_verified_horizons",
                "missing_horizons": missing,
                "advanced": False,
            }
        dataset, dataset_lineage_id = self._common_formal_dataset(versions)
        daily_dataset = self._governed_daily_dataset(dataset)
        allocation = self._ensure_allocation(
            versions=versions,
            dataset=dataset,
            dataset_lineage_id=dataset_lineage_id,
            profile=profile,
        )
        if allocation["status"] != "active":
            try:
                prepared_cutover = self._prepare_primary_account_cutover(
                    replacement_allocation_id=str(allocation["id"]),
                    daily_dataset=daily_dataset,
                )
            except ValueError as exc:
                return {
                    "status": "waiting_for_allocation_evidence",
                    "phase": "account_cutover",
                    "allocation_id": str(allocation["id"]),
                    "reason": str(exc),
                    "advanced": False,
                }
            try:
                allocation = self.allocations.approve(
                    str(allocation["id"]),
                    actor="system:auto-promotion",
                    reason=(
                        "Every included horizon passed its sealed forward gate; activate "
                        "the frozen 20/40/40 policy while missing sleeves remain cash."
                    ),
                )
            except ValueError as exc:
                restore_error = None
                if prepared_cutover is not None:
                    try:
                        self.simulations.abort_allocation_source_replacement(
                            str(prepared_cutover["portfolio_id"]),
                            old_allocation_id=str(
                                prepared_cutover["old_allocation_id"]
                            ),
                            new_allocation_id=str(allocation["id"]),
                        )
                    except (KeyError, ValueError) as restore_exc:
                        restore_error = str(restore_exc)
                return {
                    "status": "waiting_for_allocation_evidence",
                    "allocation_id": str(allocation["id"]),
                    "reason": (
                        str(exc)
                        if restore_error is None
                        else f"{exc}; previous account remains safely paused: {restore_error}"
                    ),
                    "advanced": False,
                }
        try:
            simulation = self._ensure_simulation(
                allocation=allocation,
                dataset_name=dataset,
                initial_cash=float(profile["initial_capital"]),
                daily_dataset=daily_dataset,
            )
        except (KeyError, ValueError) as exc:
            return {
                "status": "waiting_for_allocation_evidence",
                "phase": "account_cutover",
                "allocation_id": str(allocation["id"]),
                "reason": str(exc),
                "advanced": False,
            }
        try:
            plan = self.netting.build_plan_for_allocation(
                str(allocation["id"]),
                actor="three-horizon-netting",
                execution_policy="open",
                max_instrument_weight=0.08,
                max_industry_weight=0.25,
                max_gross_exposure=float(profile["max_gross_exposure"]),
                primary_account={
                    "portfolio_id": str(simulation["id"]),
                    "source_id": str(simulation["source_id"]),
                    "nav": float(simulation["nav"]),
                    "updated_at": simulation.get("updated_at"),
                },
            )
        except ValueError as exc:
            return {
                "status": "waiting_for_member_targets",
                "allocation_id": str(allocation["id"]),
                "simulation_portfolio_id": str(simulation["id"]),
                "reason": str(exc),
                "advanced": False,
            }
        _prices, member_snapshot_evidence = self._latest_snapshot_payloads(
            str(allocation["id"])
        )
        if set(member_snapshot_evidence) != set(versions.values()):
            return {
                "status": "waiting_for_member_targets",
                "allocation_id": str(allocation["id"]),
                "simulation_portfolio_id": str(simulation["id"]),
                "reason": "three-horizon cutover requires one succeeded snapshot per member",
                "advanced": False,
            }
        try:
            batch = self._materialize_order_plan(
                allocation=allocation,
                simulation=simulation,
                plan=plan,
                investor_profile=profile,
                now=current,
            )
        except ValueError as exc:
            return {
                "status": "waiting_for_execution_dataset",
                "allocation_id": str(allocation["id"]),
                "simulation_portfolio_id": str(simulation["id"]),
                "reason": str(exc),
                "advanced": False,
            }
        return {
            "status": "ready" if batch is not None else "no_action",
            "contract_version": THREE_HORIZON_ACCOUNT_VERSION,
            "allocation_id": str(allocation["id"]),
            "simulation_portfolio_id": str(simulation["id"]),
            "netting_plan_id": str(plan["id"]),
            "strategy_version_ids": dict(versions),
            "missing_horizons": missing,
            "reserved_horizon_cash_weight": sum(
                HORIZON_WEIGHTS[horizon] * float(profile["max_gross_exposure"])
                for horizon in missing
            ),
            "member_snapshot_evidence": member_snapshot_evidence,
            "order_batch": batch,
            "advanced": batch is not None,
        }

    def _active_versions(self) -> dict[str, str]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    strategy_versions.c.id,
                    strategy_versions.c.horizon_profile,
                )
                .where(
                    strategy_versions.c.status == "approved",
                    strategy_versions.c.promotion_stage == "recommendation_enabled",
                    strategy_versions.c.horizon_profile.in_(tuple(HORIZON_WEIGHTS)),
                )
                .order_by(strategy_versions.c.approved_at.desc())
            ).all()
        result: dict[str, str] = {}
        for row in rows:
            result.setdefault(str(row.horizon_profile), str(row.id))
        return result

    def _common_formal_dataset(
        self, versions: dict[str, str]
    ) -> tuple[str, str | None]:
        datasets: dict[str, str] = {}
        lineages: dict[str, str] = {}
        with self.engine.connect() as connection:
            for horizon, version_id in versions.items():
                row = connection.execute(
                    select(
                        backtest_runs.c.dataset,
                        backtest_runs.c.metrics_json,
                    )
                    .where(
                        backtest_runs.c.strategy_version_id == version_id,
                        backtest_runs.c.status == "succeeded",
                        backtest_runs.c.is_legacy.is_(False),
                    )
                    .order_by(backtest_runs.c.finished_at.desc())
                    .limit(1)
                ).first()
                if row is None:
                    raise ValueError(f"{horizon} has no formal backtest dataset")
                datasets[horizon] = str(row.dataset)
                metrics = dict(row.metrics_json or {})
                receipt = metrics.get("capital_oos_receipt")
                provenance = metrics.get("provenance")
                lineage = str(
                    (receipt or {}).get("dataset_lineage_id")
                    if isinstance(receipt, dict)
                    else ""
                ) or str(
                    (provenance or {}).get("dataset_lineage_id")
                    if isinstance(provenance, dict)
                    else ""
                )
                if lineage:
                    lineage = lineage.lower()
                    if len(lineage) != 64 or any(
                        character not in "0123456789abcdef" for character in lineage
                    ):
                        raise ValueError(
                            f"{horizon} formal backtest has an invalid dataset lineage"
                        )
                    lineages[horizon] = lineage
        selected = _select_current_three_horizon_dataset(
            formal_datasets=datasets,
            formal_lineages=lineages,
            qlib_datasets=list_qlib_datasets(self.settings.data_root),
        )
        return selected, (next(iter(set(lineages.values()))) if lineages else None)

    def _allocation_name(self, versions: dict[str, str]) -> str:
        identity = hashlib.sha256(
            "|".join(f"{key}:{versions[key]}" for key in sorted(versions)).encode()
        ).hexdigest()[:12]
        return f"three-horizon-primary-{identity}"

    def _ensure_allocation(
        self,
        *,
        versions: dict[str, str],
        dataset: str,
        dataset_lineage_id: str | None,
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        name = self._allocation_name(versions)
        with self.engine.connect() as connection:
            allocation_id = connection.scalar(
                select(strategy_allocations.c.id).where(strategy_allocations.c.name == name)
            )
        if allocation_id:
            return self.allocations.get(str(allocation_id))
        ordered_ids = [
            versions[horizon] for horizon in HORIZON_WEIGHTS if horizon in versions
        ]
        gross_exposure = float(profile["max_gross_exposure"])
        fixed_weights = _active_horizon_fixed_weights(
            versions, max_gross_exposure=gross_exposure
        )
        return self.allocations.create(
            name=name,
            strategy_version_ids=ordered_ids,
            dataset=dataset,
            total_capital=float(profile["initial_capital"]),
            allocation_method="fixed",
            lookback_days=252,
            target_volatility=0.50,
            max_pairwise_correlation=0.99,
            max_strategy_weight=0.40,
            max_member_drawdown=0.08,
            max_drawdown_reduce=0.10,
            max_drawdown_liquidate=0.15,
            fixed_weights=fixed_weights,
            actor="system:auto-promotion",
            member_specs=[
                {
                    "strategy_version_id": versions[horizon],
                    "role": "core",
                    "risk_budget": HORIZON_WEIGHTS[horizon],
                    "member_cap": HORIZON_WEIGHTS[horizon] * gross_exposure,
                }
                for horizon in HORIZON_WEIGHTS
                if horizon in versions
            ],
            decision_frequency="monthly",
            dataset_lineage_id=dataset_lineage_id,
        )

    def _governed_daily_dataset(self, dataset_name: str) -> dict[str, Any]:
        datasets = {
            item["name"]: item for item in list_qlib_datasets(self.settings.data_root)
        }
        daily = datasets.get(dataset_name)
        if daily is None or not daily.get("ready") or not daily.get("reproducible"):
            raise ValueError("three-horizon daily execution dataset is unavailable")
        lineage_id = str(daily.get("lineage_id") or "")
        if len(lineage_id) != 64:
            raise ValueError("three-horizon daily dataset has no verified lineage")
        return daily

    def _ensure_simulation(
        self,
        *,
        allocation: dict[str, Any],
        dataset_name: str,
        initial_cash: float,
        daily_dataset: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        daily = daily_dataset or self._governed_daily_dataset(dataset_name)
        if str(daily.get("name") or "") != dataset_name:
            raise ValueError("three-horizon daily dataset identity changed during cutover")
        lineage_id = str(daily.get("lineage_id") or "")
        if len(lineage_id) != 64:
            raise ValueError("three-horizon daily dataset has no verified lineage")
        with self.engine.begin() as connection:
            member_portfolios = connection.scalars(
                select(strategy_allocation_members.c.recommendation_portfolio_id).where(
                    strategy_allocation_members.c.allocation_id == allocation["id"],
                    strategy_allocation_members.c.recommendation_portfolio_id.is_not(None),
                )
            ).all()
            connection.execute(
                update(recommendation_portfolios)
                .where(recommendation_portfolios.c.id.in_(member_portfolios))
                .values(
                    dataset_roll_policy="latest_compatible",
                    dataset_lineage_id=lineage_id,
                )
            )
            owned = connection.execute(
                select(
                    simulation_portfolios.c.id,
                    simulation_portfolios.c.source_id,
                    simulation_portfolios.c.status,
                )
                .where(
                    simulation_portfolios.c.source_type == "allocation",
                    simulation_portfolios.c.created_by
                    == THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
                )
                .order_by(simulation_portfolios.c.created_at)
            ).all()
            foreign = connection.execute(
                select(simulation_portfolios.c.id).where(
                    simulation_portfolios.c.source_type == "allocation",
                    simulation_portfolios.c.source_id == allocation["id"],
                    simulation_portfolios.c.created_by
                    != THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
                )
            ).first()
        if len(owned) > 1:
            raise ValueError(
                "multiple three-horizon primary ledgers already exist; refusing to merge them"
            )
        if foreign is not None:
            raise ValueError(
                "the active allocation is bound to a non-primary simulation ledger"
            )
        if owned:
            primary = owned[0]
            if str(primary.source_id) != str(allocation["id"]):
                self.simulations.replace_allocation_source(
                    str(primary.id),
                    old_allocation_id=str(primary.source_id),
                    new_allocation_id=str(allocation["id"]),
                    actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
                    daily_dataset=daily,
                    execution_dataset=daily,
                )
            elif str(primary.status) != "active":
                self.simulations.set_status(str(primary.id), "active")
            return self.simulations.get(str(primary.id))
        simulation = self.simulations.create(
            name="three-horizon-primary unified paper",
            source_type="allocation",
            source_id=str(allocation["id"]),
            daily_dataset=daily,
            execution_dataset=daily,
            initial_cash=initial_cash,
            execution_policy={
                "execution_algorithm": "open",
                "execution_frequency": "day",
                "max_participation": 0.01,
            },
            cost_schedule_version=COST_SCHEDULE_VERSION,
            actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
            daily_roll_policy="latest_compatible",
            execution_roll_policy="latest_compatible",
        )
        return self.simulations.set_status(str(simulation["id"]), "active")

    def _prepare_primary_account_cutover(
        self,
        *,
        replacement_allocation_id: str,
        daily_dataset: dict[str, Any],
    ) -> dict[str, str] | None:
        """Quiesce the one serving ledger before allocation approval replaces it."""

        with self.engine.connect() as connection:
            owned = connection.execute(
                select(
                    simulation_portfolios.c.id,
                    simulation_portfolios.c.source_id,
                )
                .where(
                    simulation_portfolios.c.source_type == "allocation",
                    simulation_portfolios.c.created_by
                    == THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
                )
                .order_by(simulation_portfolios.c.created_at)
            ).all()
            active_allocation_id = connection.scalar(
                select(strategy_allocations.c.id).where(
                    strategy_allocations.c.status == "active"
                )
            )
        if len(owned) > 1:
            raise ValueError(
                "multiple three-horizon primary ledgers already exist; cutover is blocked"
            )
        if not owned:
            if active_allocation_id is not None:
                raise ValueError(
                    "an active three-horizon allocation has no persistent primary ledger"
                )
            return None
        primary = owned[0]
        if str(primary.source_id) == str(replacement_allocation_id):
            return None
        if active_allocation_id is None or str(primary.source_id) != str(
            active_allocation_id
        ):
            raise ValueError(
                "the primary ledger does not match the currently active allocation"
            )
        self.simulations.prepare_allocation_source_replacement(
            str(primary.id),
            old_allocation_id=str(primary.source_id),
            new_allocation_id=str(replacement_allocation_id),
            actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
            daily_dataset=daily_dataset,
            execution_dataset=daily_dataset,
        )
        return {
            "portfolio_id": str(primary.id),
            "old_allocation_id": str(primary.source_id),
        }

    def _latest_snapshot_payloads(
        self, allocation_id: str
    ) -> tuple[dict[str, float], dict[str, dict[str, str]]]:
        prices: dict[str, tuple[date, float]] = {}
        evidence: dict[str, dict[str, str]] = {}
        with self.engine.connect() as connection:
            members = connection.execute(
                select(strategy_allocation_members).where(
                    strategy_allocation_members.c.allocation_id == allocation_id
                )
            ).all()
            for member in members:
                snapshot = connection.execute(
                    select(recommendation_snapshots)
                    .where(
                        recommendation_snapshots.c.portfolio_id
                        == member.recommendation_portfolio_id,
                        recommendation_snapshots.c.status == "succeeded",
                    )
                    .order_by(recommendation_snapshots.c.as_of_date.desc())
                    .limit(1)
                ).first()
                if snapshot is None:
                    continue
                evidence[str(member.strategy_version_id)] = {
                    "snapshot_id": str(snapshot.id),
                    "as_of_date": snapshot.as_of_date.isoformat(),
                }
                for instrument, raw_price in dict(
                    (snapshot.snapshot_json or {}).get("reference_prices") or {}
                ).items():
                    observed = prices.get(str(instrument))
                    value = float(raw_price)
                    if observed is None or snapshot.as_of_date >= observed[0]:
                        prices[str(instrument)] = (snapshot.as_of_date, value)
        return {key: value[1] for key, value in prices.items()}, evidence

    def _materialize_order_plan(
        self,
        *,
        allocation: dict[str, Any],
        simulation: dict[str, Any],
        plan: dict[str, Any],
        investor_profile: dict[str, Any],
        now: datetime,
    ) -> dict[str, Any] | None:
        trade_date = date.fromisoformat(str(plan["decision_date"]))
        prices, snapshot_evidence = self._latest_snapshot_payloads(str(allocation["id"]))
        positions = {
            str(item["instrument"]): item
            for item in self.simulations.rows(str(simulation["id"]), "positions")
        }
        open_orders: dict[str, list[dict[str, Any]]] = {}
        for order in self.simulations.rows(str(simulation["id"]), "orders"):
            if str(order.get("status")) not in OPEN_STATUSES:
                continue
            open_orders.setdefault(str(order["instrument"]), []).append(order)
        instruments = set(plan.get("net_targets") or {}) | set(positions) | set(open_orders)
        inputs: list[dict[str, Any]] = []
        limit_prices: dict[str, float] = {}
        permission_blocks: dict[str, dict[str, Any]] = {}
        for instrument in sorted(instruments):
            position = positions.get(instrument) or {}
            price = prices.get(instrument) or float(position.get("market_price") or 0.0)
            target = dict((plan.get("net_targets") or {}).get(instrument) or {})
            if price <= 0:
                inputs.append(
                    {
                        "instrument": instrument,
                        "target_quantity": None,
                        "filled_position": int(position.get("quantity") or 0),
                        "sellable_quantity": int(position.get("available_quantity") or 0),
                        "open_orders": open_orders.get(instrument, []),
                        "hard_blocked_reason": "fresh_reference_price_unavailable",
                    }
                )
                continue
            rules = order_unit_rules(instrument, trade_date)
            raw_quantity = int(float(target.get("target_value") or 0.0) / price)
            target_quantity = lot_floor(raw_quantity, rules) if raw_quantity > 0 else 0
            filled_position = int(position.get("quantity") or 0)
            if target_quantity > filled_position:
                permission = investor_profile_permission(
                    investor_profile,
                    instrument,
                    on_date=trade_date,
                )
                if permission["allowed"] is not True:
                    # A missing permission may never create new risk.  Keep the
                    # confirmed position as the executable target so existing
                    # exposure can still be reduced by a later target.
                    target_quantity = filled_position
                    permission_blocks[instrument] = permission
            limit_prices[instrument] = price
            inputs.append(
                {
                    "instrument": instrument,
                    "target_quantity": target_quantity,
                    "filled_position": filled_position,
                    "sellable_quantity": int(position.get("available_quantity") or 0),
                    "open_orders": open_orders.get(instrument, []),
                    "lot_increment": rules.lot_increment,
                    "min_lot": rules.min_lot,
                }
            )
        actions = plan_account_actions(inputs, now=now)
        for action in actions:
            permission = permission_blocks.get(str(action["instrument"]))
            if permission is not None:
                action["new_risk_blocked"] = True
                action["new_risk_blocked_reason"] = permission["reason"]
                action["investor_permission_key"] = permission["permission_key"]
        actionable = any(
            item["order_plan"] or item["action"] in {"BUY", "SELL", "EXIT"}
            for item in actions
        )
        if not actionable:
            return None
        zone = ZoneInfo("Asia/Shanghai")
        batch, _created = self.simulations.create_order_plan_batch(
            str(simulation["id"]),
            trade_date=trade_date,
            signal_date=date.fromisoformat(str(plan["inputs_as_of"])),
            actions=actions,
            target_version=f"three-horizon-netting:{plan['plan_hash']}",
            actor=THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
            account_netting_plan_id=str(plan["id"]),
            limit_prices=limit_prices,
            not_before=datetime.combine(trade_date, time(9, 30), zone),
            not_after=datetime.combine(trade_date, time(15, 0), zone),
            data_root=self.settings.data_root,
        )
        batch["member_snapshot_evidence"] = snapshot_evidence
        return batch
