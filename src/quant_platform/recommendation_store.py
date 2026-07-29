from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from math import isfinite
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    factor_evaluations,
    recommendation_holdings,
    recommendation_nav,
    recommendation_portfolios,
    recommendation_snapshots,
    row_dict,
)

from .account_risk_state import assess_account_risk
from .cost_model import CostModelConfig
from .market_rules import lot_floor, order_unit_rules
from .portfolio_policy import POLICY_VERSION
from .qlib_backtest import QLIB_ENGINE_VERSION
from .recommendation_actions import (
    RECOMMENDATION_ACTION_MODEL_VERSION,
    plan_account_actions,
)
from .safe_mode import SafeModeStore
from .strategy_store import StrategyStore


def _now() -> datetime:
    return datetime.now(UTC)


def _build_account_action_plan(
    *,
    weights: dict[str, float],
    rule_date: date,
    construction_notional: float,
    account_state: dict[str, dict[str, Any]],
    now: datetime | None,
    permission_store: Any | None,
    account_context: dict[str, Any] | None,
    risk_assessment: dict[str, Any] | None,
    account_value: float | None,
) -> dict[str, Any]:
    initial_value = (
        float(account_value)
        if account_value is not None
        else float(construction_notional)
    )
    if not isfinite(initial_value) or initial_value <= 0:
        raise ValueError("account action planning requires a positive finite account value")
    instruments: list[dict[str, Any]] = []
    for instrument in sorted(set(weights) | set(account_state)):
        state = dict(account_state.get(instrument) or {})
        target_quantity = state.pop("target_quantity", None)
        if instrument in weights and target_quantity is None:
            reference_price = state.pop("reference_price", None)
            if reference_price is None or float(reference_price) <= 0:
                raise ValueError(
                    f"account state for {instrument} needs a reference price or "
                    "an explicit target quantity"
                )
            rules = order_unit_rules(instrument, rule_date)
            target_quantity = lot_floor(
                int(weights[instrument] * initial_value / float(reference_price)),
                rules,
            )
        elif instrument not in weights:
            target_quantity = 0
        lot_rules = order_unit_rules(instrument, rule_date)
        filled_position = int(state.pop("filled_position", 0))
        hard_blocked_reason = state.pop("hard_blocked_reason", None)
        not_executable_reason = state.pop("not_executable_reason", None)
        if permission_store is not None and target_quantity is not None:
            gate = permission_store.gate_for_instrument(
                instrument,
                on_date=rule_date,
                is_buy_action=int(target_quantity) > filled_position,
            )
            if gate is not None:
                if gate["kind"] == "hard" and not hard_blocked_reason:
                    hard_blocked_reason = gate["reason"]
                elif gate["kind"] == "soft" and not not_executable_reason:
                    not_executable_reason = gate["reason"]
        instruments.append(
            {
                "instrument": instrument,
                "target_quantity": target_quantity,
                "filled_position": filled_position,
                "sellable_quantity": state.pop("sellable_quantity", None),
                "open_orders": state.pop("open_orders", []),
                "lot_increment": lot_rules.lot_increment,
                "min_lot": lot_rules.min_lot,
                "hard_blocked_reason": hard_blocked_reason,
                "not_executable_reason": not_executable_reason,
            }
        )
    computed_at = now or _now()
    resolved_context = account_context or {
        "account_type": "main_paper",
        "degraded": False,
    }
    resolved_risk = risk_assessment or assess_account_risk(
        account_state_stale=bool(resolved_context.get("degraded")),
        market_data_trusted=not bool(resolved_context.get("degraded")),
    )
    return {
        "model_version": RECOMMENDATION_ACTION_MODEL_VERSION,
        "computed_at": computed_at.isoformat(),
        "rule_date": rule_date.isoformat(),
        "account_context": resolved_context,
        "account_value": initial_value,
        "account_value_source": (
            "selected_account" if account_value is not None else "construction_notional"
        ),
        "risk_assessment": resolved_risk,
        "items": plan_account_actions(
            instruments,
            now=computed_at,
            risk_assessment=resolved_risk,
        ),
    }


class RecommendationStore:
    """Durable recommendation snapshots; this store has no order or fill concepts."""

    def __init__(self, database_url: str) -> None:
        self.strategies = StrategyStore(database_url)
        self.engine = self.strategies.engine
        self.safe_mode = SafeModeStore(database_url)

    def _assert_v2_strategy(self, version: dict[str, Any]) -> None:
        if version.get("is_legacy"):
            raise ValueError("legacy strategy versions cannot generate recommendations")
        if version["status"] != "approved" or version.get("strategy_type") != "multifactor":
            raise ValueError("recommendations require an approved multifactor strategy")
        # Design 6.11: standalone recommendations require the version to have
        # passed the forward evidence gate. NULL promotion_stage marks legacy
        # rows/fixtures that predate the promotion chain; "paper" is the only
        # stage that blocks.
        if version.get("promotion_stage") == "paper":
            raise ValueError(
                "strategy version is in the paper stage; standalone recommendations "
                "require recommendation_enabled (forward evidence gate plus human approval)"
            )
        evaluation_ids = [item.get("factor_evaluation_id") for item in version["factors"]]
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    factor_evaluations.c.id,
                    factor_evaluations.c.evaluator_version,
                    factor_evaluations.c.is_legacy,
                ).where(factor_evaluations.c.id.in_(evaluation_ids))
            ).all()
        versions = {str(row.id): (str(row.evaluator_version), bool(row.is_legacy)) for row in rows}
        if len(versions) != len(evaluation_ids) or any(
            versions.get(str(item), ("", True))[1]
            or versions.get(str(item), ("", True))[0] != "factor-gate-v3-hac-bh"
            for item in evaluation_ids
        ):
            raise ValueError("legacy factor evaluations cannot generate recommendations")

    def create(
        self,
        *,
        name: str,
        strategy_version_id: str,
        dataset: str,
        hypothetical_initial_value: float,
        actor: str,
        recommendation_scope: str = "standalone",
        dataset_roll_policy: str = "pinned",
        dataset_lineage_id: str | None = None,
    ) -> dict[str, Any]:
        version = self.strategies.get_version(strategy_version_id)
        self._assert_v2_strategy(version)
        if hypothetical_initial_value < 100_000:
            raise ValueError("hypothetical initial value must be at least 100000")
        if not name.strip() or not dataset.strip() or not actor.strip():
            raise ValueError("name, dataset and actor are required")
        if recommendation_scope not in {"standalone", "allocation_member"}:
            raise ValueError("recommendation_scope must be standalone or allocation_member")
        if dataset_roll_policy not in {"pinned", "latest_compatible"}:
            raise ValueError("recommendation dataset roll policy is invalid")
        if dataset_roll_policy == "latest_compatible" and len(
            str(dataset_lineage_id or "")
        ) != 64:
            raise ValueError(
                "latest-compatible recommendations require a verified dataset lineage"
            )
        portfolio_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                # Design 8.1/9.1: a single active recommendation sender at any
                # moment. The partial unique index enforces this at the
                # database layer; fail closed here with a readable error.
                conflict = connection.execute(
                    select(recommendation_portfolios.c.id, recommendation_portfolios.c.name)
                    .where(
                        recommendation_portfolios.c.status == "active",
                        recommendation_portfolios.c.recommendation_scope == "standalone",
                    )
                    .limit(1)
                ).first()
                if conflict is not None:
                    raise ValueError(
                        f"recommendation portfolio {conflict.name!r} is already the active "
                        "sender; pause it before creating a new one"
                    )
                connection.execute(
                    insert(recommendation_portfolios).values(
                        id=portfolio_id,
                        name=name.strip(),
                        strategy_version_id=strategy_version_id,
                        dataset=dataset.strip(),
                        dataset_roll_policy=dataset_roll_policy,
                        dataset_lineage_id=dataset_lineage_id,
                        status="active",
                        base_currency="CNY",
                        hypothetical_initial_value=Decimal(str(hypothetical_initial_value)),
                        risk_exposure_override=1.0,
                        recommendation_scope=recommendation_scope,
                        created_by=actor.strip(),
                        created_at=now,
                        updated_at=now,
                    )
                )
        except IntegrityError as exc:
            raise ValueError(f"recommendation portfolio name {name!r} already exists") from exc
        return self.get(portfolio_id)

    def get(self, portfolio_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(recommendation_portfolios).where(
                    recommendation_portfolios.c.id == portfolio_id
                )
            ).first()
            if row is None:
                raise KeyError(portfolio_id)
            result = row_dict(row)
            snapshots = [
                self._snapshot_row(item, connection)
                for item in connection.execute(
                    select(recommendation_snapshots)
                    .where(recommendation_snapshots.c.portfolio_id == portfolio_id)
                    .order_by(recommendation_snapshots.c.as_of_date.desc())
                    .limit(100)
                )
            ]
            nav = [
                row_dict(item)
                for item in connection.execute(
                    select(recommendation_nav)
                    .where(recommendation_nav.c.portfolio_id == portfolio_id)
                    .order_by(recommendation_nav.c.trade_date)
                )
            ]
        result["snapshots"] = snapshots
        result["latest_snapshot"] = snapshots[0] if snapshots else None
        result["construction_notional"] = result.pop("hypothetical_initial_value")
        result["historical_hypothetical_observations"] = nav
        return result

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            ids = connection.scalars(
                select(recommendation_portfolios.c.id)
                .order_by(recommendation_portfolios.c.updated_at.desc())
                .limit(limit)
            ).all()
        return [self.get(str(item)) for item in ids]

    def set_status(
        self, portfolio_id: str, status: str, *, actor: str = "recommendation-operator"
    ) -> dict[str, Any]:
        if status not in {"active", "paused", "retired"}:
            raise ValueError("recommendation status must be active, paused or retired")
        if len(actor.strip()) < 2:
            raise ValueError("a responsible actor is required")
        with self.engine.begin() as connection:
            portfolio = connection.execute(
                select(recommendation_portfolios)
                .where(recommendation_portfolios.c.id == portfolio_id)
                .with_for_update()
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            if portfolio.recommendation_scope != "standalone":
                raise ValueError(
                    "allocation member portfolios are managed by their allocation"
                )
            if status == "active":
                # Design 8.1/9.1: a single active recommendation sender.
                conflict = connection.execute(
                    select(recommendation_portfolios.c.id, recommendation_portfolios.c.name)
                    .where(
                        recommendation_portfolios.c.status == "active",
                        recommendation_portfolios.c.recommendation_scope == "standalone",
                        recommendation_portfolios.c.id != portfolio_id,
                    )
                    .limit(1)
                ).first()
                if conflict is not None:
                    raise ValueError(
                        f"recommendation portfolio {conflict.name!r} is already the active "
                        "sender; pause it before activating another one"
                    )
            connection.execute(
                update(recommendation_portfolios)
                .where(recommendation_portfolios.c.id == portfolio_id)
                .values(status=status, updated_at=_now())
            )
        return self.get(portfolio_id)

    def create_snapshot(
        self,
        *,
        portfolio_id: str,
        as_of_date: date,
        dataset: str,
        dataset_identity_sha256: str,
        dataset_lineage_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        # Design 11.3: new recommendations stop while safe mode is active.
        self.safe_mode.assert_inactive(action="recommendation snapshot creation")
        portfolio = self.get(portfolio_id)
        if portfolio["status"] != "active":
            raise ValueError("only active recommendation portfolios may refresh")
        roll_policy = str(portfolio.get("dataset_roll_policy") or "pinned")
        if roll_policy == "pinned" and dataset != portfolio["dataset"]:
            raise ValueError("pinned recommendation snapshot dataset must match its portfolio")
        if roll_policy == "latest_compatible" and (
            len(str(dataset_lineage_id or "")) != 64
            or dataset_lineage_id != portfolio.get("dataset_lineage_id")
        ):
            raise ValueError(
                "recommendation snapshot must use a verified compatible dataset lineage"
            )
        if len(dataset_identity_sha256) != 64:
            raise ValueError("recommendation snapshot requires immutable dataset identity")
        version = self.strategies.get_version(portfolio["strategy_version_id"])
        cost_model = CostModelConfig.from_mapping(version["config"])
        snapshot_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(recommendation_snapshots).values(
                        id=snapshot_id,
                        portfolio_id=portfolio_id,
                        as_of_date=as_of_date,
                        status="queued",
                        snapshot_json=None,
                        cost_model_json=cost_model.to_dict(),
                        policy_version=POLICY_VERSION,
                        backtest_engine_version=QLIB_ENGINE_VERSION,
                        dataset=dataset,
                        dataset_identity_sha256=dataset_identity_sha256,
                        dataset_lineage_id=dataset_lineage_id,
                        strategy_version_id=portfolio["strategy_version_id"],
                        created_at=now,
                    )
                )
        except IntegrityError:
            with self.engine.connect() as connection:
                row = connection.execute(
                    select(recommendation_snapshots).where(
                        recommendation_snapshots.c.portfolio_id == portfolio_id,
                        recommendation_snapshots.c.as_of_date == as_of_date,
                    )
                ).one()
                return self._snapshot_row(row, connection), False
        return self.get_snapshot(snapshot_id), True

    def attach_job(self, snapshot_id: str, job_id: str) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(recommendation_snapshots)
                .where(
                    recommendation_snapshots.c.id == snapshot_id,
                    recommendation_snapshots.c.status.in_(("queued", "running")),
                )
                .values(job_id=job_id, status="running", started_at=_now())
            )
            if not result.rowcount:
                raise KeyError(snapshot_id)

    def apply_result(
        self,
        snapshot_id: str,
        result: dict[str, Any],
        *,
        account_state: dict[str, dict[str, Any]] | None = None,
        permission_store: Any | None = None,
        account_context: dict[str, Any] | None = None,
        risk_assessment: dict[str, Any] | None = None,
        account_value: float | None = None,
    ) -> dict[str, Any]:
        if (
            result.get("status") != "ok"
            or result.get("policy_version") != POLICY_VERSION
            or result.get("backtest_engine_version") != QLIB_ENGINE_VERSION
        ):
            raise ValueError("recommendation result is not from the current PortfolioPolicy")
        holdings = result.get("holdings")
        if not isinstance(holdings, list):
            raise ValueError("recommendation result holdings must be a list")
        cash_weight = result.get("cash_weight")
        if not isinstance(cash_weight, int | float) or not isfinite(float(cash_weight)):
            raise ValueError("recommendation result requires a finite cash weight")
        if any(not isinstance(item, dict) for item in holdings):
            raise ValueError("recommendation holdings must contain objects")
        instruments = [str(item.get("instrument") or "") for item in holdings]
        weights = [float(item.get("weight", float("nan"))) for item in holdings]
        reference_prices = result.get("reference_prices")
        if (
            any(not instrument for instrument in instruments)
            or len(instruments) != len(set(instruments))
            or any(not isfinite(weight) or weight < 0 for weight in weights)
            or not 0 <= float(cash_weight) <= 1
            or abs(sum(weights) + float(cash_weight) - 1.0) > 1e-6
        ):
            raise ValueError("recommendation holdings and cash weight are inconsistent")
        if (
            not isinstance(reference_prices, dict)
            or any(instrument not in reference_prices for instrument in instruments)
            or any(
                not isfinite(float(reference_prices[instrument]))
                or float(reference_prices[instrument]) <= 0
                for instrument in instruments
            )
        ):
            raise ValueError(
                "recommendation holdings require positive finite reference prices"
            )
        now = _now()
        effective_date = date.fromisoformat(result["effective_date"])
        planned_account_actions = None
        if account_state is not None:
            preview = self.get_snapshot(snapshot_id)
            portfolio = self.get(str(preview["portfolio_id"]))
            planned_account_actions = _build_account_action_plan(
                weights=dict(zip(instruments, weights, strict=True)),
                rule_date=effective_date,
                construction_notional=float(portfolio["construction_notional"]),
                account_state=account_state,
                now=now,
                permission_store=permission_store,
                account_context=account_context,
                risk_assessment=risk_assessment,
                account_value=account_value,
            )
        with self.engine.begin() as connection:
            snapshot = connection.execute(
                select(recommendation_snapshots)
                .where(recommendation_snapshots.c.id == snapshot_id)
                .with_for_update()
            ).first()
            if snapshot is None:
                raise KeyError(snapshot_id)
            expected = {
                "portfolio_id": str(snapshot.portfolio_id),
                "strategy_version_id": str(snapshot.strategy_version_id),
                "dataset": str(snapshot.dataset),
                "dataset_identity_sha256": str(snapshot.dataset_identity_sha256),
                "as_of_date": snapshot.as_of_date.isoformat(),
            }
            mismatches = [
                field for field, value in expected.items() if str(result.get(field) or "") != value
            ]
            if mismatches:
                raise ValueError(
                    "recommendation result identity does not match snapshot: "
                    + ", ".join(mismatches)
                )
            if snapshot.status not in {"queued", "running"}:
                raise ValueError("recommendation snapshot is already terminal")
            if not isinstance(result.get("cost_model"), dict):
                raise ValueError("recommendation result has no cost model")
            result_cost_model = CostModelConfig.from_mapping(result["cost_model"]).to_dict()
            if result_cost_model != dict(snapshot.cost_model_json):
                raise ValueError("recommendation result cost model does not match snapshot")
            if effective_date <= snapshot.as_of_date:
                raise ValueError("recommendation effective date must follow its signal date")
            connection.execute(
                update(recommendation_snapshots)
                .where(recommendation_snapshots.c.id == snapshot_id)
                .values(
                    effective_date=effective_date,
                    status="succeeded",
                    snapshot_json=result,
                    account_actions_json=planned_account_actions,
                    cost_model_json=result_cost_model,
                    error=None,
                    finished_at=now,
                )
            )
            if holdings:
                connection.execute(
                    insert(recommendation_holdings),
                    [
                        {
                            "snapshot_id": snapshot_id,
                            "instrument": item["instrument"],
                            "weight": item["weight"],
                            "previous_weight": item.get("previous_weight", 0.0),
                            "weight_change": item.get("weight_change", item["weight"]),
                            "action": item.get("action", "increase"),
                            "reason": item.get(
                                "reason", "signal ranking and portfolio constraints"
                            ),
                            "average_cost": item.get("average_cost"),
                            "take_profit_stage": int(item.get("take_profit_stage", 0)),
                        }
                        for item in holdings
                    ],
                )
            connection.execute(
                update(recommendation_portfolios)
                .where(recommendation_portfolios.c.id == snapshot.portfolio_id)
                .values(updated_at=now)
            )
        return self.get_snapshot(snapshot_id)

    def mark_failed(self, snapshot_id: str, error: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(recommendation_snapshots)
                .where(
                    recommendation_snapshots.c.id == snapshot_id,
                    recommendation_snapshots.c.status.in_(("queued", "running")),
                )
                .values(status="failed", error=error, finished_at=_now())
            )

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(recommendation_snapshots).where(recommendation_snapshots.c.id == snapshot_id)
            ).first()
            if row is None:
                raise KeyError(snapshot_id)
            return self._snapshot_row(row, connection)

    @staticmethod
    def _snapshot_row(row: Any, connection: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["snapshot"] = result.pop("snapshot_json")
        result["account_actions"] = result.pop("account_actions_json")
        result["cost_model"] = result.pop("cost_model_json")
        result["holdings"] = [
            row_dict(item)
            for item in connection.execute(
                select(recommendation_holdings)
                .where(recommendation_holdings.c.snapshot_id == row.id)
                .order_by(recommendation_holdings.c.weight.desc())
            )
        ]
        return result

    def attach_account_actions(
        self,
        snapshot_id: str,
        *,
        account_state: dict[str, dict[str, Any]],
        now: datetime | None = None,
        permission_store: Any | None = None,
        account_context: dict[str, Any] | None = None,
        risk_assessment: dict[str, Any] | None = None,
        account_value: float | None = None,
    ) -> dict[str, Any]:
        """Attach the two-dimension account action plan (design 8.4) to a snapshot.

        ``account_state`` maps instrument → account-side facts:
        ``filled_position`` (confirmed fills), ``sellable_quantity``,
        ``open_orders`` (simulation-ledger rows with side/requested/filled/
        expires_at), ``reference_price`` and optional ``target_quantity``
        overrides, plus ``hard_blocked_reason`` / ``not_executable_reason``.
        Target quantities default to ``lot_floor(weight × hypothetical initial
        value ÷ reference_price)``; instruments present in the account state
        but absent from the holdings get target zero (EXIT semantics).  The
        plan is stored in ``account_actions_json``; the legacy
        increase/decrease holdings export is left untouched.

        When ``permission_store`` (a :class:`MarketPermissionStore`) is given,
        the personal market permission (design 8.7) gates every line at the
        rule date: ``disabled``/``sell_only`` block buy actions with a recorded
        reason (SELL/EXIT stay allowed), ``unknown``/expired permissions mark
        the line not executable (``simulation_only``) without erasing the
        advice. ``account_context`` records which account the state came from
        (``main_paper`` by default; ``manual_shadow`` with freshness/degraded
        flags when built from :class:`ShadowAccountStore`) so shadow, model
        and simulation accounts stay visibly separate (design 8.6/8.1).
        """

        snapshot = self.get_snapshot(snapshot_id)
        if snapshot["status"] != "succeeded":
            raise ValueError("account actions require a succeeded recommendation snapshot")
        portfolio = self.get(str(snapshot["portfolio_id"]))
        rule_date = snapshot["effective_date"] or snapshot["as_of_date"]
        weights = {
            str(item["instrument"]): float(item["weight"]) for item in snapshot["holdings"]
        }
        plan = _build_account_action_plan(
            weights=weights,
            rule_date=rule_date,
            construction_notional=float(portfolio["construction_notional"]),
            account_state=account_state,
            now=now,
            permission_store=permission_store,
            account_context=account_context,
            risk_assessment=risk_assessment,
            account_value=account_value,
        )
        with self.engine.begin() as connection:
            result = connection.execute(
                update(recommendation_snapshots)
                .where(
                    recommendation_snapshots.c.id == snapshot_id,
                    recommendation_snapshots.c.status == "succeeded",
                )
                .values(account_actions_json=plan)
            )
            if not result.rowcount:
                raise KeyError(snapshot_id)
        return self.get_snapshot(snapshot_id)
