"""Strategy promotion chain: paper stage and forward evidence gate.

Design 4.5/6.11/7.4/9.5: the lifecycle is
``research -> candidate -> paper -> recommendation_enabled``.

- ``candidate -> paper`` happens automatically when the formal hard gate
  (``StrategyStore.approve``) passes: an isolated forward paper stage is
  opened with its own simulation account, capital and evidence counters —
  historical backtest evidence never substitutes forward evidence.
- ``paper -> recommendation_enabled`` requires the version's pre-registered
  forward gate (minimum natural time, independent decision batches, completed
  holding/rebalance cycles, data completeness, ledger reconciliation rate and
  cost deviation) *and* a human approval. Insufficient evidence keeps the
  version in paper and is reported as ``insufficient_evidence``; thresholds
  are never lowered to admit a candidate.
- A substantive source-contract drift (design 9.5, detected through the
  simulation source-contract guard) freezes the old stage read-only and opens
  a new one whose evidence starts from zero; stages are never concatenated.

Evidence is derived from the simulation ledger of the stage's own paper
account: succeeded batches are independent decisions, batch conservation
results feed the reconciliation rate, fills feed the realized cost rate, and
the NAV span measures natural time. The completed-cycle proxy is a succeeded
batch containing at least one sell fill (a round trip); it defaults to zero
required cycles because cycle shape is strategy-specific.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, insert, select, update

from quant_data.database import (
    backtest_runs,
    open_database,
    simulation_batches,
    simulation_fills,
    simulation_nav,
    simulation_portfolios,
    strategy_events,
    strategy_forward_gates,
    strategy_promotion_stages,
    strategy_versions,
)
from quant_platform.cost_model import (
    CN_COST_SCHEDULE_VERSIONS,
    COST_SCHEDULE_VERSION,
    CostModelConfig,
)
from quant_platform.simulation_store import SimulationStore

PROMOTION_CONTRACT_VERSION = "promotion-chain-v1"

STAGE_PAPER = "paper"
STAGE_RECOMMENDATION_ENABLED = "recommendation_enabled"

_STAGE_ACTIVE = "active"
_STAGE_AWAITING = "awaiting_simulation"
_STAGE_FROZEN = "frozen"

_REFERENCE_ORDER_VALUE = 100_000.0
_DEFAULT_PAPER_INITIAL_CASH = 100_000.0
_RECONCILIATION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ForwardGateThresholds:
    """Pre-registered forward evidence gate (design 4.5/6.11).

    Defaults target the platform's medium/low-frequency monthly/weekly
    cadence: twenty calendar days span roughly one monthly observation
    window, ten decision batches cover two weekly or ten daily independent
    decisions, the ledger reconciliation rate must be perfect (append-only
    cash discipline), data completeness tolerates one failed batch in twenty,
    and the realized one-side cost rate may deviate from the scheduled rate
    by at most 50bp. These are the documented default template, not a
    universal mathematical gate — each strategy pre-registers its own values
    (design 6.11: a fixed 60 days is one strategy's config, never a global
    rule).
    """

    min_forward_calendar_days: int = 20
    min_decision_batches: int = 10
    min_completed_cycles: int = 0
    min_data_completeness: float = 0.95
    min_reconciliation_rate: float = 1.0
    max_cost_deviation: float = 0.005

    def __post_init__(self) -> None:
        if (
            self.min_forward_calendar_days < 0
            or self.min_decision_batches < 0
            or self.min_completed_cycles < 0
        ):
            raise ValueError("forward gate day/batch/cycle thresholds must be non-negative")
        if not 0 < self.min_data_completeness <= 1 or not 0 < self.min_reconciliation_rate <= 1:
            raise ValueError("forward gate rates must be in (0, 1]")
        if not 0 <= self.max_cost_deviation < 1:
            raise ValueError("forward gate cost deviation must be in [0, 1)")


def _now() -> datetime:
    return datetime.now(UTC)


def _insufficient(reasons: list[str], **extra: Any) -> dict[str, Any]:
    return {"status": "insufficient_evidence", "passed": False, "reasons": reasons, **extra}


class PromotionStore:
    """Paper stage lifecycle and forward evidence gate evaluation."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.engine = open_database(database_url)

    # ------------------------------------------------------------------
    # Gate registration (pre-paper only)
    # ------------------------------------------------------------------

    def register_forward_gate(
        self,
        version_id: str,
        *,
        actor: str,
        thresholds: ForwardGateThresholds | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Pre-register (or replace) the forward gate; blocked once in paper."""

        if len(actor.strip()) < 2:
            raise ValueError("a responsible actor is required")
        gate = thresholds or ForwardGateThresholds(**overrides)
        with self.engine.begin() as connection:
            version = connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == version_id)
            ).first()
            if version is None:
                raise KeyError(version_id)
            if version.promotion_stage in (STAGE_PAPER, STAGE_RECOMMENDATION_ENABLED):
                raise ValueError(
                    "the forward gate must be pre-registered before the version enters paper"
                )
            now = _now()
            values = {
                **asdict(gate),
                "registered_by": actor.strip(),
                "updated_at": now,
            }
            existing = connection.execute(
                select(strategy_forward_gates).where(
                    strategy_forward_gates.c.strategy_version_id == version_id
                )
            ).first()
            if existing is None:
                connection.execute(
                    insert(strategy_forward_gates).values(
                        strategy_version_id=version_id, registered_at=now, **values
                    )
                )
            else:
                connection.execute(
                    update(strategy_forward_gates)
                    .where(strategy_forward_gates.c.strategy_version_id == version_id)
                    .values(**values)
                )
        return {"strategy_version_id": version_id, **asdict(gate)}

    # ------------------------------------------------------------------
    # Paper stage opening (candidate -> paper, automatic after the hard gate)
    # ------------------------------------------------------------------

    def open_paper_stage(self, version_id: str, *, actor: str) -> dict[str, Any]:
        """Open the isolated paper stage after the formal hard gate passes.

        Idempotent: an already-active stage is returned unchanged. When the
        approval backtest carries no consumable dataset descriptors
        (``datasets.json`` in the artifact root), the stage stays in
        ``awaiting_simulation`` until :meth:`attach_paper_simulation` binds an
        isolated account — evidence cannot accumulate without it.
        """

        with self.engine.begin() as connection:
            version = connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == version_id)
            ).first()
            if version is None:
                raise KeyError(version_id)
            if str(version.status) != "approved":
                raise ValueError("paper stage requires an approved strategy version")
            existing = connection.execute(
                select(strategy_promotion_stages).where(
                    strategy_promotion_stages.c.strategy_version_id == version_id,
                    strategy_promotion_stages.c.status.in_([_STAGE_ACTIVE, _STAGE_AWAITING]),
                )
            ).first()
            if existing is not None:
                return self._stage_dict(existing)
            next_index = (
                connection.execute(
                    select(func.max(strategy_promotion_stages.c.stage_index)).where(
                        strategy_promotion_stages.c.strategy_version_id == version_id
                    )
                ).scalar()
                or 0
            ) + 1
            stage_id = uuid.uuid4().hex
            connection.execute(
                insert(strategy_promotion_stages).values(
                    id=stage_id,
                    strategy_version_id=version_id,
                    stage_index=next_index,
                    simulation_portfolio_id=None,
                    status=_STAGE_AWAITING,
                    opened_at=_now(),
                    created_by=actor.strip(),
                )
            )
        try:
            datasets = self._load_backtest_datasets(version_id)
            if datasets is None:
                raise ValueError("approval backtest carries no dataset descriptors")
            self.attach_paper_simulation(
                version_id,
                actor=actor,
                daily_dataset=datasets["daily"],
                execution_dataset=datasets["execution"],
            )
        except ValueError as exc:
            self._event(
                version_id,
                event_type="strategy.paper_stage_awaiting_simulation",
                actor=actor,
                payload={"stage_id": stage_id, "error": str(exc)},
            )
        return self.current_stage(version_id)

    def attach_paper_simulation(
        self,
        version_id: str,
        *,
        actor: str,
        daily_dataset: dict[str, Any],
        execution_dataset: dict[str, Any],
        initial_cash: float | None = None,
    ) -> dict[str, Any]:
        """Bind the isolated paper simulation account to the awaiting stage."""

        with self.engine.begin() as connection:
            stage = connection.execute(
                select(strategy_promotion_stages)
                .where(
                    strategy_promotion_stages.c.strategy_version_id == version_id,
                    strategy_promotion_stages.c.status == _STAGE_AWAITING,
                )
                .order_by(strategy_promotion_stages.c.stage_index.desc())
                .limit(1)
            ).first()
            if stage is None:
                raise ValueError("no paper stage is waiting for a simulation account")
            version = connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == version_id)
            ).one()
        config = dict(version.config_json or {})
        cash = float(
            initial_cash
            if initial_cash is not None
            else config.get("paper_initial_cash", _DEFAULT_PAPER_INITIAL_CASH)
        )
        simulation = SimulationStore(self.database_url)
        portfolio = simulation.create(
            name=(
                f"paper:{version_id[:8]}:v{int(version.version)}:stage{int(stage.stage_index)}"
            )[:150],
            source_type="strategy_version",
            source_id=version_id,
            daily_dataset=daily_dataset,
            execution_dataset=execution_dataset,
            initial_cash=cash,
            execution_policy={
                "execution_algorithm": str(config.get("execution_method") or "twap")
            },
            cost_schedule_version=str(
                config.get("cost_schedule_version") or config.get("version") or ""
            ),
            actor=actor,
        )
        simulation.set_status(portfolio["id"], "active")
        with self.engine.begin() as connection:
            connection.execute(
                update(strategy_promotion_stages)
                .where(strategy_promotion_stages.c.id == stage.id)
                .values(
                    simulation_portfolio_id=portfolio["id"],
                    status=_STAGE_ACTIVE,
                    source_contract_hash=portfolio["execution_contract_hash"],
                    initial_cash=cash,
                )
            )
        self._event(
            version_id,
            event_type="strategy.paper_stage_opened",
            actor=actor,
            payload={
                "stage_id": str(stage.id),
                "simulation_portfolio_id": portfolio["id"],
                "initial_cash": cash,
            },
        )
        return self.current_stage(version_id)

    # ------------------------------------------------------------------
    # Forward evidence gate
    # ------------------------------------------------------------------

    def current_stage(self, version_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(strategy_promotion_stages)
                .where(
                    strategy_promotion_stages.c.strategy_version_id == version_id,
                    strategy_promotion_stages.c.status.in_([_STAGE_ACTIVE, _STAGE_AWAITING]),
                )
                .order_by(strategy_promotion_stages.c.stage_index.desc())
                .limit(1)
            ).first()
        return self._stage_dict(row) if row is not None else None

    def list_stages(self, version_id: str) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(strategy_promotion_stages)
                .where(strategy_promotion_stages.c.strategy_version_id == version_id)
                .order_by(strategy_promotion_stages.c.stage_index)
            ).all()
        return [self._stage_dict(row) for row in rows]

    def evaluate_forward_gate(self, version_id: str) -> dict[str, Any]:
        """Evaluate the pre-registered gate against the active stage's evidence."""

        with self.engine.begin() as connection:
            version = connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == version_id)
            ).first()
            if version is None:
                raise KeyError(version_id)
            gate = connection.execute(
                select(strategy_forward_gates).where(
                    strategy_forward_gates.c.strategy_version_id == version_id
                )
            ).first()
            if gate is None:
                return _insufficient(["forward evidence gate is not pre-registered"])
            stage = connection.execute(
                select(strategy_promotion_stages)
                .where(
                    strategy_promotion_stages.c.strategy_version_id == version_id,
                    strategy_promotion_stages.c.status.in_([_STAGE_ACTIVE, _STAGE_AWAITING]),
                )
                .order_by(strategy_promotion_stages.c.stage_index.desc())
                .limit(1)
            ).first()
            if stage is None:
                return _insufficient(["no active paper stage exists"])
            if stage.simulation_portfolio_id is None:
                return _insufficient(
                    ["paper stage has no isolated simulation account; evidence is zero"]
                )
            portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == stage.simulation_portfolio_id
                )
            ).first()
            try:
                SimulationStore._require_current_source_contract(connection, portfolio)
            except ValueError as exc:
                # Design 9.5: substantive contract drift freezes the stage
                # read-only and starts a fresh stage from zero evidence.
                connection.execute(
                    update(strategy_promotion_stages)
                    .where(strategy_promotion_stages.c.id == stage.id)
                    .values(
                        status=_STAGE_FROZEN,
                        frozen_at=_now(),
                        freeze_reason=str(exc),
                    )
                )
                connection.execute(
                    insert(strategy_promotion_stages).values(
                        id=uuid.uuid4().hex,
                        strategy_version_id=version_id,
                        stage_index=int(stage.stage_index) + 1,
                        simulation_portfolio_id=None,
                        status=_STAGE_AWAITING,
                        opened_at=_now(),
                        created_by="promotion-chain",
                    )
                )
                connection.execute(
                    insert(strategy_events).values(
                        strategy_id=str(version.strategy_id),
                        strategy_version_id=version_id,
                        event_type="strategy.paper_stage_frozen_contract_drift",
                        actor="promotion-chain",
                        payload_json={"stage_id": str(stage.id), "error": str(exc)},
                        created_at=_now(),
                    )
                )
                return _insufficient(
                    [
                        "paper stage frozen on source contract drift; "
                        "a new stage starts from zero evidence",
                        str(exc),
                    ],
                    stage_reset=True,
                )
            evidence = self._collect_evidence(connection, stage, portfolio)

        checks = {
            "forward_calendar_days": (
                evidence["forward_calendar_days"],
                int(gate.min_forward_calendar_days),
                "min",
            ),
            "decision_batches": (
                evidence["decision_batches"],
                int(gate.min_decision_batches),
                "min",
            ),
            "completed_cycles": (
                evidence["completed_cycles"],
                int(gate.min_completed_cycles),
                "min",
            ),
            "data_completeness": (
                evidence["data_completeness"],
                float(gate.min_data_completeness),
                "min",
            ),
            "reconciliation_rate": (
                evidence["reconciliation_rate"],
                float(gate.min_reconciliation_rate),
                "min",
            ),
            "cost_deviation": (
                evidence["cost_deviation"],
                float(gate.max_cost_deviation),
                "max",
            ),
        }
        results = {
            name: {
                "observed": observed,
                "threshold": threshold,
                "passed": (
                    observed >= threshold if mode == "min" else observed <= threshold
                ),
            }
            for name, (observed, threshold, mode) in checks.items()
        }
        failures = [name for name, result in results.items() if not result["passed"]]
        if failures:
            return _insufficient(
                [f"{name} below/above the pre-registered threshold" for name in failures],
                checks=results,
                evidence=evidence,
                stage_id=str(stage.id),
            )
        return {
            "status": "ok",
            "passed": True,
            "reasons": [],
            "checks": results,
            "evidence": evidence,
            "stage_id": str(stage.id),
            "contract_version": PROMOTION_CONTRACT_VERSION,
        }

    def promote(self, version_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        """paper -> recommendation_enabled: forward gate plus human approval."""

        if not actor.strip() or len(reason.strip()) < 10:
            raise ValueError("actor and a meaningful approval reason are required")
        evaluation = self.evaluate_forward_gate(version_id)
        if not evaluation["passed"]:
            raise ValueError(
                "forward evidence gate is not satisfied (insufficient_evidence; "
                "thresholds are never lowered): " + "; ".join(evaluation["reasons"])
            )
        with self.engine.begin() as connection:
            version = connection.execute(
                select(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .with_for_update()
            ).first()
            if version is None:
                raise KeyError(version_id)
            if str(version.status) != "approved" or version.promotion_stage != STAGE_PAPER:
                raise ValueError("only a paper-stage approved version can be promoted")
            now = _now()
            connection.execute(
                update(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .values(promotion_stage=STAGE_RECOMMENDATION_ENABLED)
            )
            connection.execute(
                update(strategy_promotion_stages)
                .where(strategy_promotion_stages.c.id == evaluation["stage_id"])
                .values(promoted_at=now)
            )
            connection.execute(
                insert(strategy_events).values(
                    strategy_id=str(version.strategy_id),
                    strategy_version_id=version_id,
                    event_type="strategy.recommendation_enabled",
                    actor=actor.strip(),
                    payload_json={
                        "reason": reason.strip(),
                        "evidence": evaluation["evidence"],
                        "stage_id": evaluation["stage_id"],
                    },
                    created_at=now,
                )
            )
        return {
            "strategy_version_id": version_id,
            "promotion_stage": STAGE_RECOMMENDATION_ENABLED,
            "evidence": evaluation["evidence"],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_backtest_datasets(self, version_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            backtest = connection.execute(
                select(backtest_runs)
                .where(
                    backtest_runs.c.strategy_version_id == version_id,
                    backtest_runs.c.status == "succeeded",
                )
                .order_by(backtest_runs.c.created_at.desc())
                .limit(1)
            ).first()
        if backtest is None or not backtest.artifact_path:
            return None
        descriptor = Path(str(backtest.artifact_path)) / "datasets.json"
        if not descriptor.is_file():
            return None
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
        daily = payload.get("daily")
        execution = payload.get("execution")
        if not isinstance(daily, dict) or not isinstance(execution, dict):
            return None
        return {"daily": daily, "execution": execution}

    def _collect_evidence(
        self, connection: Any, stage: Any, portfolio: Any
    ) -> dict[str, Any]:
        portfolio_id = str(stage.simulation_portfolio_id)
        batches = connection.execute(
            select(
                simulation_batches.c.id,
                simulation_batches.c.status,
                simulation_batches.c.summary_json,
            ).where(simulation_batches.c.portfolio_id == portfolio_id)
        ).all()
        succeeded = [batch for batch in batches if str(batch.status) == "succeeded"]
        nav_span = connection.execute(
            select(
                func.min(simulation_nav.c.trade_date),
                func.max(simulation_nav.c.trade_date),
            ).where(simulation_nav.c.portfolio_id == portfolio_id)
        ).one()
        calendar_days = 0
        if nav_span[0] is not None and nav_span[1] is not None:
            calendar_days = (nav_span[1] - nav_span[0]).days + 1
        reconciled = 0
        for batch in succeeded:
            conservation = (batch.summary_json or {}).get("conservation") or {}
            difference = conservation.get("cash_difference")
            if difference is not None and abs(float(difference)) <= _RECONCILIATION_TOLERANCE:
                reconciled += 1
        fill_stats = connection.execute(
            select(
                func.coalesce(func.sum(simulation_fills.c.fee), 0.0),
                func.coalesce(func.sum(simulation_fills.c.gross_value), 0.0),
            ).where(
                simulation_fills.c.batch_id.in_(
                    [str(batch.id) for batch in succeeded] or [""]
                )
            )
        ).one()
        sell_batches = connection.execute(
            select(simulation_fills.c.batch_id)
            .where(
                simulation_fills.c.batch_id.in_([str(batch.id) for batch in succeeded] or [""]),
                simulation_fills.c.side == "sell",
            )
            .distinct()
        ).all()
        total_fees = float(fill_stats[0])
        total_value = float(fill_stats[1])
        realized_rate = total_fees / total_value if total_value > 0 else 0.0
        scheduled_rate = self._scheduled_one_side_rate(str(portfolio.cost_schedule_version))
        return {
            "forward_calendar_days": calendar_days,
            "decision_batches": len(succeeded),
            "total_batches": len(batches),
            "completed_cycles": len(sell_batches),
            "data_completeness": (len(succeeded) / len(batches)) if batches else 0.0,
            "reconciliation_rate": (reconciled / len(succeeded)) if succeeded else 0.0,
            "realized_cost_rate": realized_rate,
            "scheduled_cost_rate": scheduled_rate,
            "cost_deviation": abs(realized_rate - scheduled_rate),
        }

    @staticmethod
    def _scheduled_one_side_rate(cost_schedule_version: str) -> float:
        if cost_schedule_version == COST_SCHEDULE_VERSION:
            config = CostModelConfig()
        else:
            config = next(
                (
                    item
                    for item in CN_COST_SCHEDULE_VERSIONS
                    if item.version == cost_schedule_version
                ),
                None,
            )
        if config is None:
            raise ValueError(f"unknown cost schedule version: {cost_schedule_version}")
        participation = config.max_volume_participation
        buy = config.estimate(
            side="buy",
            gross_value=_REFERENCE_ORDER_VALUE,
            participation=participation,
        )
        sell = config.estimate(
            side="sell",
            gross_value=_REFERENCE_ORDER_VALUE,
            participation=participation,
        )
        return (buy + sell) / (2.0 * _REFERENCE_ORDER_VALUE)

    def _event(self, version_id: str, *, event_type: str, actor: str, payload: dict) -> None:
        with self.engine.begin() as connection:
            version = connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == version_id)
            ).first()
            connection.execute(
                insert(strategy_events).values(
                    strategy_id=str(version.strategy_id) if version else "",
                    strategy_version_id=version_id,
                    event_type=event_type,
                    actor=actor.strip(),
                    payload_json=payload,
                    created_at=_now(),
                )
            )

    def record_paper_stage_failure(self, version_id: str, *, actor: str, error: str) -> None:
        """Best-effort failure trace when auto paper opening hits a system error."""

        try:
            self._event(
                version_id,
                event_type="strategy.paper_stage_open_failed",
                actor=actor,
                payload={"error": error},
            )
        except Exception:  # the approval itself is already committed; tracing is best-effort
            pass

    @staticmethod
    def _stage_dict(row: Any) -> dict[str, Any]:
        return {
            "id": str(row.id),
            "strategy_version_id": str(row.strategy_version_id),
            "stage_index": int(row.stage_index),
            "simulation_portfolio_id": (
                str(row.simulation_portfolio_id) if row.simulation_portfolio_id else None
            ),
            "status": str(row.status),
            "source_contract_hash": (
                str(row.source_contract_hash) if row.source_contract_hash else None
            ),
            "initial_cash": float(row.initial_cash) if row.initial_cash is not None else None,
            "opened_at": row.opened_at.isoformat(),
            "frozen_at": row.frozen_at.isoformat() if row.frozen_at else None,
            "freeze_reason": str(row.freeze_reason) if row.freeze_reason else None,
            "promoted_at": row.promoted_at.isoformat() if row.promoted_at else None,
        }
