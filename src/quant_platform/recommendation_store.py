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

from .cost_model import CostModelConfig
from .portfolio_policy import POLICY_VERSION
from .qlib_backtest import QLIB_ENGINE_VERSION
from .strategy_store import StrategyStore


def _now() -> datetime:
    return datetime.now(UTC)


class RecommendationStore:
    """Durable recommendation snapshots; this store has no order or fill concepts."""

    def __init__(self, database_url: str) -> None:
        self.strategies = StrategyStore(database_url)
        self.engine = self.strategies.engine

    def _assert_v2_strategy(self, version: dict[str, Any]) -> None:
        if version.get("is_legacy"):
            raise ValueError("legacy strategy versions cannot generate recommendations")
        if version["status"] != "approved" or version.get("strategy_type") != "multifactor":
            raise ValueError("recommendations require an approved multifactor strategy")
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
            or not versions.get(str(item), ("", True))[0].startswith("factor-gate-v2")
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
    ) -> dict[str, Any]:
        version = self.strategies.get_version(strategy_version_id)
        self._assert_v2_strategy(version)
        if hypothetical_initial_value < 100_000:
            raise ValueError("hypothetical initial value must be at least 100000")
        if not name.strip() or not dataset.strip() or not actor.strip():
            raise ValueError("name, dataset and actor are required")
        portfolio_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(recommendation_portfolios).values(
                        id=portfolio_id,
                        name=name.strip(),
                        strategy_version_id=strategy_version_id,
                        dataset=dataset.strip(),
                        status="active",
                        base_currency="CNY",
                        hypothetical_initial_value=Decimal(str(hypothetical_initial_value)),
                        risk_exposure_override=1.0,
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
        result["hypothetical_performance"] = nav
        return result

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            ids = connection.scalars(
                select(recommendation_portfolios.c.id)
                .order_by(recommendation_portfolios.c.updated_at.desc())
                .limit(limit)
            ).all()
        return [self.get(str(item)) for item in ids]

    def set_status(self, portfolio_id: str, status: str) -> dict[str, Any]:
        if status not in {"active", "paused", "retired"}:
            raise ValueError("recommendation status must be active, paused or retired")
        with self.engine.begin() as connection:
            result = connection.execute(
                update(recommendation_portfolios)
                .where(recommendation_portfolios.c.id == portfolio_id)
                .values(status=status, updated_at=_now())
            )
            if not result.rowcount:
                raise KeyError(portfolio_id)
        return self.get(portfolio_id)

    def create_snapshot(
        self,
        *,
        portfolio_id: str,
        as_of_date: date,
        dataset: str,
        dataset_identity_sha256: str,
    ) -> tuple[dict[str, Any], bool]:
        portfolio = self.get(portfolio_id)
        if portfolio["status"] != "active":
            raise ValueError("only active recommendation portfolios may refresh")
        if dataset != portfolio["dataset"]:
            raise ValueError("recommendation snapshot dataset must match its portfolio")
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

    def apply_result(self, snapshot_id: str, result: dict[str, Any]) -> dict[str, Any]:
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
        if (
            any(not instrument for instrument in instruments)
            or len(instruments) != len(set(instruments))
            or any(not isfinite(weight) or weight < 0 for weight in weights)
            or not 0 <= float(cash_weight) <= 1
            or abs(sum(weights) + float(cash_weight) - 1.0) > 1e-6
        ):
            raise ValueError("recommendation holdings and cash weight are inconsistent")
        now = _now()
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
            effective_date = date.fromisoformat(result["effective_date"])
            if effective_date <= snapshot.as_of_date:
                raise ValueError("recommendation effective date must follow its signal date")
            connection.execute(
                update(recommendation_snapshots)
                .where(recommendation_snapshots.c.id == snapshot_id)
                .values(
                    effective_date=effective_date,
                    status="succeeded",
                    snapshot_json=result,
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
            observation = result.get("hypothetical_observation")
            if isinstance(observation, dict):
                prior_peak = connection.scalar(
                    select(recommendation_nav.c.hypothetical_value)
                    .where(recommendation_nav.c.portfolio_id == snapshot.portfolio_id)
                    .order_by(recommendation_nav.c.hypothetical_value.desc())
                    .limit(1)
                )
                value = Decimal(str(observation["hypothetical_value"]))
                peak = max(value, Decimal(str(prior_peak or value)))
                drawdown = float(value / peak - 1) if peak else 0.0
                connection.execute(
                    insert(recommendation_nav).values(
                        portfolio_id=snapshot.portfolio_id,
                        trade_date=date.fromisoformat(observation["trade_date"]),
                        hypothetical_value=value,
                        daily_return=float(observation["daily_return"]),
                        benchmark_return=float(observation["benchmark_return"]),
                        drawdown=drawdown,
                        turnover=float(observation["turnover"]),
                        estimated_cost=Decimal(str(observation["estimated_cost"])),
                        created_at=now,
                    )
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
