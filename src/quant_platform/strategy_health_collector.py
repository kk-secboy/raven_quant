"""Periodic governed health observations for active three-horizon strategies."""

from __future__ import annotations

from datetime import UTC, date, datetime
from math import isfinite
from pathlib import Path
from typing import Any

from sqlalchemy import select

from quant_data.database import (
    open_database,
    simulation_batches,
    simulation_fills,
    simulation_nav,
    simulation_orders,
    simulation_portfolios,
    strategy_health_snapshots,
    strategy_promotion_stages,
    strategy_versions,
)

from .promotion import PromotionStore
from .simulation_store import QLIB_ORDER_PLAN_FORMAT_VERSION
from .strategy_feature_drift_source import StrategyFeatureDriftSource
from .strategy_health import HEALTH_WINDOWS_BY_HORIZON, assess_strategy_health
from .strategy_store import StrategyStore

COLLECTOR_ACTOR = "system:strategy-health-collector"
_ACTIVE_PROMOTION_STAGES = ("paper", "recommendation_enabled")
_TERMINAL_EXECUTION_STATUSES = frozenset(
    {"filled", "partial_filled_expired", "rejected", "expired"}
)
_ADVERSE_EXECUTION_STATUSES = frozenset(
    {"partial_filled_expired", "rejected", "expired"}
)
_RECONCILIATION_TOLERANCE = 0.01


def _aware(value: Any, *, field: str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value or ""))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(UTC)


def _loss_positive_drawdown(row: Any) -> float:
    raw = row.twr_drawdown if row.twr_drawdown is not None else row.drawdown
    value = float(raw or 0.0)
    if not isfinite(value):
        raise ValueError("strategy NAV drawdown is not finite")
    return max(0.0, -value)


def _value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    mapping = getattr(row, "_mapping", None)
    if mapping is not None:
        return mapping.get(key)
    return getattr(row, key, None)


def resolve_latest_batch_binding(
    lane: dict[str, Any],
    nav_rows: list[Any],
    batches: list[Any],
) -> dict[str, Any]:
    """Bind health to the exact succeeded batch behind the newest NAV."""

    if not nav_rows:
        raise ValueError("strategy health is waiting for its first NAV")
    trade_date = _value(nav_rows[0], "trade_date")
    matches = [
        row
        for row in batches
        if _value(row, "trade_date") == trade_date
        and str(_value(row, "status") or "") == "succeeded"
    ]
    if len(matches) != 1:
        raise ValueError("latest strategy NAV has no unique succeeded simulation batch")
    batch = matches[0]
    identity = str(_value(batch, "daily_dataset_identity_sha256") or "")
    lineage = str(_value(batch, "daily_dataset_lineage_id") or "")
    source_snapshot_id = str(_value(batch, "source_snapshot_id") or "")
    target_payload = _value(batch, "target_payload_json")
    governed_plan = (
        target_payload.get("governed_order_plan")
        if isinstance(target_payload, dict)
        else None
    )
    formal_backtest_id = (
        str(governed_plan.get("formal_backtest_id") or "")
        if isinstance(governed_plan, dict)
        else ""
    )
    plan_manifest_sha256 = (
        str(governed_plan.get("manifest_sha256") or "")
        if isinstance(governed_plan, dict)
        else ""
    )
    signal_date = _value(batch, "signal_date")
    if (
        len(identity) != 64
        or len(lineage) != 64
        or lineage != str(lane["daily_dataset_lineage_id"])
        or source_snapshot_id != identity
        or not isinstance(signal_date, date)
        or not isinstance(trade_date, date)
        or signal_date > trade_date
        or not isinstance(governed_plan, dict)
        or governed_plan.get("format_version") != QLIB_ORDER_PLAN_FORMAT_VERSION
        or governed_plan.get("promotion_stage_id") != str(lane["promotion_stage_id"])
        or not formal_backtest_id
        or len(plan_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in plan_manifest_sha256.lower()
        )
    ):
        raise ValueError("latest strategy batch dataset/source lineage is invalid")
    return {
        "simulation_batch_id": str(_value(batch, "id") or ""),
        "trade_date": trade_date,
        "signal_date": signal_date,
        "daily_dataset": str(_value(batch, "daily_dataset") or ""),
        "daily_dataset_identity_sha256": identity,
        "daily_dataset_lineage_id": lineage,
        "source_snapshot_id": source_snapshot_id,
        "formal_backtest_id": formal_backtest_id,
        "order_plan_manifest_sha256": plan_manifest_sha256.lower(),
    }


def strategy_health_collection_due(
    latest_snapshot: dict[str, Any] | None,
    latest_batch: dict[str, Any],
    *,
    now: datetime,
    interval_seconds: int,
) -> bool:
    """A new paper batch bypasses the periodic interval immediately."""

    if latest_snapshot is None:
        return True
    latest_as_of = _aware(latest_snapshot["as_of"], field="strategy health as_of")
    if latest_as_of > now:
        raise ValueError("strategy health history is in the future")
    evidence = dict(latest_snapshot.get("evidence_json") or {})
    same_batch = (
        evidence.get("evidence_trade_date")
        == latest_batch["trade_date"].isoformat()
        and evidence.get("simulation_batch_id")
        == latest_batch["simulation_batch_id"]
        and evidence.get("daily_dataset_identity_sha256")
        == latest_batch["daily_dataset_identity_sha256"]
    )
    if not same_batch:
        return True
    return (now - latest_as_of).total_seconds() >= interval_seconds


class StrategyHealthCollector:
    """Assess and append health for active paper/recommendation strategy lanes."""

    def __init__(
        self,
        database_url: str,
        *,
        data_root: Path,
        interval_seconds: int = 3600,
    ) -> None:
        self.database_url = database_url
        self.engine = open_database(database_url)
        self.interval_seconds = max(300, int(interval_seconds))
        self.strategies = StrategyStore(database_url)
        self.feature_drift = StrategyFeatureDriftSource(database_url, data_root)

    def collect_due(self, now: datetime | None = None) -> dict[str, Any]:
        current = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("strategy health collection time must be timezone-aware")
        result: dict[str, Any] = {
            "contract_version": "strategy-health-collector-run-v1",
            "observed_at": current.isoformat(),
            "scanned": 0,
            "recorded": 0,
            "skipped_not_due": 0,
            "failures": [],
            "snapshots": [],
        }
        for lane in self._active_lanes():
            result["scanned"] += 1
            version_id = str(lane["strategy_version_id"])
            try:
                latest_batch = self._latest_lane_batch(lane, current)
                latest = self._latest_snapshot(version_id, actor=COLLECTOR_ACTOR)
                if not strategy_health_collection_due(
                    latest,
                    latest_batch,
                    now=current,
                    interval_seconds=self.interval_seconds,
                ):
                    result["skipped_not_due"] += 1
                    continue
                snapshot = self._collect_and_record(lane, current, latest_batch)
            except Exception as exc:  # noqa: BLE001 - isolate each active lane
                result["failures"].append(
                    {"strategy_version_id": version_id, "error": str(exc)[:1000]}
                )
                continue
            result["recorded"] += 1
            result["snapshots"].append(
                {
                    "strategy_version_id": version_id,
                    "snapshot_id": str(snapshot["id"]),
                    "health_status": str(snapshot["health_status"]),
                }
            )
        return result

    def pending_requests(self, now: datetime | None = None) -> dict[str, Any]:
        """Return bounded, exact lane identities for durable worker enqueueing.

        This method intentionally performs only indexed metadata queries. Qlib,
        parquet, and formal artifacts are never opened in the scheduler process.
        """

        current = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
        requests: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for lane in self._active_lanes():
            version_id = str(lane["strategy_version_id"])
            try:
                latest_batch = self._latest_lane_batch(lane, current)
                latest = self._latest_snapshot(version_id, actor=COLLECTOR_ACTOR)
                if not strategy_health_collection_due(
                    latest,
                    latest_batch,
                    now=current,
                    interval_seconds=self.interval_seconds,
                ):
                    continue
            except Exception as exc:  # noqa: BLE001 - isolate each active lane
                failures.append(
                    {"strategy_version_id": version_id, "error": str(exc)[:1000]}
                )
                continue
            requests.append(
                {
                    "strategy_version_id": version_id,
                    "promotion_stage_id": str(lane["promotion_stage_id"]),
                    "simulation_batch_id": str(latest_batch["simulation_batch_id"]),
                    "formal_backtest_id": str(latest_batch["formal_backtest_id"]),
                    "daily_dataset_identity_sha256": str(
                        latest_batch["daily_dataset_identity_sha256"]
                    ),
                    "requested_at": current.isoformat(),
                }
            )
        return {
            "contract_version": "strategy-health-collection-requests-v1",
            "requested_at": current.isoformat(),
            "scanned": len(requests) + len(failures),
            "requests": requests,
            "failures": failures,
        }

    def collect_one(
        self,
        *,
        strategy_version_id: str,
        promotion_stage_id: str,
        simulation_batch_id: str,
        formal_backtest_id: str,
        daily_dataset_identity_sha256: str,
        observed_at: datetime,
    ) -> dict[str, Any]:
        """Collect one exact lane in a worker, rejecting stale enqueue identities."""

        current = _aware(observed_at, field="strategy health observed_at").replace(
            microsecond=0
        )
        matches = [
            lane
            for lane in self._active_lanes()
            if str(lane["strategy_version_id"]) == strategy_version_id
            and str(lane["promotion_stage_id"]) == promotion_stage_id
        ]
        if len(matches) != 1:
            raise ValueError("strategy health job no longer names one active lane")
        lane = matches[0]
        latest_batch = self._latest_lane_batch(lane, current)
        expected = {
            "simulation_batch_id": simulation_batch_id,
            "formal_backtest_id": formal_backtest_id,
            "daily_dataset_identity_sha256": daily_dataset_identity_sha256,
        }
        if any(str(latest_batch[key]) != value for key, value in expected.items()):
            return {
                "contract_version": "strategy-health-collect-result-v1",
                "status": "superseded",
                "strategy_version_id": strategy_version_id,
                "promotion_stage_id": promotion_stage_id,
                "simulation_batch_id": simulation_batch_id,
                "observed_at": current.isoformat(),
            }
        latest = self._latest_snapshot(strategy_version_id, actor=COLLECTOR_ACTOR)
        if not strategy_health_collection_due(
            latest,
            latest_batch,
            now=current,
            interval_seconds=self.interval_seconds,
        ):
            return {
                "contract_version": "strategy-health-collect-result-v1",
                "status": "not_due",
                "strategy_version_id": strategy_version_id,
                "promotion_stage_id": promotion_stage_id,
                "simulation_batch_id": simulation_batch_id,
                "observed_at": current.isoformat(),
            }
        snapshot = self._collect_and_record(lane, current, latest_batch)
        return {
            "contract_version": "strategy-health-collect-result-v1",
            "status": "recorded",
            "strategy_version_id": strategy_version_id,
            "promotion_stage_id": promotion_stage_id,
            "simulation_batch_id": simulation_batch_id,
            "formal_backtest_id": formal_backtest_id,
            "daily_dataset_identity_sha256": daily_dataset_identity_sha256,
            "observed_at": current.isoformat(),
            "snapshot_id": str(snapshot["id"]),
            "health_status": str(snapshot["health_status"]),
        }

    def _collect_and_record(
        self,
        lane: dict[str, Any],
        current: datetime,
        latest_batch: dict[str, Any],
    ) -> dict[str, Any]:
        version_id = str(lane["strategy_version_id"])
        evidence = self._collect_lane_evidence(
            lane, current, latest_batch=latest_batch
        )
        previous = self._latest_snapshot(version_id)
        previous_status = (
            str(previous["health_status"]) if previous is not None else None
        )
        assessment = assess_strategy_health(
            str(lane["horizon_profile"]),
            evidence,
            previous_status=previous_status,
        )
        persisted_evidence = {
            **assessment["evidence"],
            **dict(evidence["provenance"]),
            "assessment_reasons": list(assessment["reasons"]),
            "windows_trading_days": list(assessment["windows_trading_days"]),
        }
        return self.strategies.record_health_snapshot(
            version_id,
            as_of=current,
            health_status=str(assessment["health_status"]),
            criteria=dict(assessment["criteria"]),
            evidence=persisted_evidence,
            actor=COLLECTOR_ACTOR,
        )

    def _active_lanes(self) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    strategy_versions.c.id.label("strategy_version_id"),
                    strategy_versions.c.horizon_profile,
                    strategy_versions.c.config_json,
                    strategy_versions.c.promotion_stage,
                    strategy_promotion_stages.c.id.label("promotion_stage_id"),
                    strategy_promotion_stages.c.stage_index,
                    simulation_portfolios.c.id.label("simulation_portfolio_id"),
                    simulation_portfolios.c.cost_schedule_version,
                    simulation_portfolios.c.daily_dataset_identity_sha256,
                    simulation_portfolios.c.daily_dataset_lineage_id,
                )
                .join(
                    strategy_promotion_stages,
                    strategy_promotion_stages.c.strategy_version_id
                    == strategy_versions.c.id,
                )
                .join(
                    simulation_portfolios,
                    simulation_portfolios.c.id
                    == strategy_promotion_stages.c.simulation_portfolio_id,
                )
                .where(
                    strategy_versions.c.status == "approved",
                    strategy_versions.c.is_legacy.is_(False),
                    strategy_versions.c.promotion_stage.in_(_ACTIVE_PROMOTION_STAGES),
                    strategy_versions.c.horizon_profile.in_(
                        tuple(HEALTH_WINDOWS_BY_HORIZON)
                    ),
                    strategy_promotion_stages.c.status == "active",
                    simulation_portfolios.c.status == "active",
                    simulation_portfolios.c.source_type == "strategy_version",
                    simulation_portfolios.c.source_id == strategy_versions.c.id,
                    simulation_portfolios.c.promotion_stage_id
                    == strategy_promotion_stages.c.id,
                    simulation_portfolios.c.execution_adapter == "long_only",
                )
                .order_by(
                    strategy_versions.c.id,
                    strategy_promotion_stages.c.stage_index.desc(),
                )
            ).mappings().all()
        # A version is permitted only one active stage, but fail deterministically
        # if old data violates that invariant instead of double-recording it.
        lanes: dict[str, dict[str, Any]] = {}
        for raw in rows:
            lane = dict(raw)
            version_id = str(lane["strategy_version_id"])
            if version_id in lanes:
                raise ValueError(
                    f"strategy version {version_id} has multiple active paper stages"
                )
            lanes[version_id] = lane
        return list(lanes.values())

    def _latest_snapshot(
        self, version_id: str, *, actor: str | None = None
    ) -> dict[str, Any] | None:
        statement = select(strategy_health_snapshots).where(
            strategy_health_snapshots.c.strategy_version_id == version_id
        )
        if actor is not None:
            statement = statement.where(
                strategy_health_snapshots.c.recorded_by == actor
            )
        with self.engine.connect() as connection:
            row = connection.execute(
                statement.order_by(
                    strategy_health_snapshots.c.as_of.desc(),
                    strategy_health_snapshots.c.recorded_at.desc(),
                    strategy_health_snapshots.c.snapshot_sha256.desc(),
                ).limit(1)
            ).mappings().first()
        return dict(row) if row is not None else None

    def _latest_lane_batch(
        self, lane: dict[str, Any], now: datetime
    ) -> dict[str, Any]:
        portfolio_id = str(lane["simulation_portfolio_id"])
        with self.engine.connect() as connection:
            nav_rows = connection.execute(
                select(simulation_nav)
                .where(
                    simulation_nav.c.portfolio_id == portfolio_id,
                    simulation_nav.c.created_at <= now,
                )
                .order_by(simulation_nav.c.trade_date.desc())
                .limit(1)
            ).all()
            batches = connection.execute(
                select(
                    simulation_batches.c.id,
                    simulation_batches.c.status,
                    simulation_batches.c.signal_date,
                    simulation_batches.c.trade_date,
                    simulation_batches.c.source_snapshot_id,
                    simulation_batches.c.daily_dataset,
                    simulation_batches.c.daily_dataset_identity_sha256,
                    simulation_batches.c.daily_dataset_lineage_id,
                    simulation_batches.c.target_payload_json,
                ).where(
                    simulation_batches.c.portfolio_id == portfolio_id,
                    simulation_batches.c.trade_date
                    == (nav_rows[0].trade_date if nav_rows else now.date()),
                    simulation_batches.c.created_at <= now,
                )
            ).all()
        return resolve_latest_batch_binding(lane, nav_rows, batches)

    def _collect_lane_evidence(
        self,
        lane: dict[str, Any],
        now: datetime,
        *,
        latest_batch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        version_id = str(lane["strategy_version_id"])
        portfolio_id = str(lane["simulation_portfolio_id"])
        horizon = str(lane["horizon_profile"])
        windows = HEALTH_WINDOWS_BY_HORIZON[horizon]
        row_limit = max(windows)
        with self.engine.connect() as connection:
            nav_rows = connection.execute(
                select(simulation_nav)
                .where(
                    simulation_nav.c.portfolio_id == portfolio_id,
                    simulation_nav.c.created_at <= now,
                )
                .order_by(simulation_nav.c.trade_date.desc())
                .limit(row_limit)
            ).all()
            cutoff_date = nav_rows[-1].trade_date if nav_rows else now.date()
            batches = connection.execute(
                select(
                    simulation_batches.c.id,
                    simulation_batches.c.status,
                    simulation_batches.c.signal_date,
                    simulation_batches.c.trade_date,
                    simulation_batches.c.source_snapshot_id,
                    simulation_batches.c.daily_dataset,
                    simulation_batches.c.daily_dataset_identity_sha256,
                    simulation_batches.c.daily_dataset_lineage_id,
                    simulation_batches.c.summary_json,
                    simulation_batches.c.target_payload_json,
                )
                .where(
                    simulation_batches.c.portfolio_id == portfolio_id,
                    simulation_batches.c.trade_date >= cutoff_date,
                    simulation_batches.c.created_at <= now,
                    simulation_batches.c.status.in_(("succeeded", "failed")),
                )
                .order_by(simulation_batches.c.trade_date.desc())
                .limit(row_limit * 2)
            ).all()
            orders = connection.execute(
                select(
                    simulation_orders.c.status,
                    simulation_orders.c.filled_value,
                    simulation_batches.c.trade_date,
                )
                .join(
                    simulation_batches,
                    simulation_batches.c.id == simulation_orders.c.batch_id,
                )
                .where(
                    simulation_batches.c.portfolio_id == portfolio_id,
                    simulation_batches.c.trade_date >= cutoff_date,
                    simulation_orders.c.created_at <= now,
                )
            ).all()
            fill_totals = connection.execute(
                select(simulation_fills.c.fee, simulation_fills.c.gross_value)
                .join(
                    simulation_batches,
                    simulation_batches.c.id == simulation_fills.c.batch_id,
                )
                .where(
                    simulation_batches.c.portfolio_id == portfolio_id,
                    simulation_batches.c.trade_date >= cutoff_date,
                    simulation_fills.c.executed_at <= now,
                )
            ).all()
        if not nav_rows or not batches:
            raise ValueError("strategy health is waiting for its first complete paper batch")
        latest_batch = latest_batch or resolve_latest_batch_binding(
            lane, nav_rows, batches
        )
        valid_nav = [
            row
            for row in nav_rows
            if bool(row.performance_certified)
            and not bool(row.has_stale_prices)
            and str(row.status) == "healthy"
            and row.market_date == row.trade_date
        ]
        nav_integrity_ok = bool(nav_rows) and len(valid_nav) == len(nav_rows)
        window_metrics: dict[str, Any] = {}
        for window in windows:
            sample = nav_rows[:window]
            window_metrics[str(window)] = {
                "observations": len(sample),
                "drawdown": max(
                    (_loss_positive_drawdown(row) for row in sample),
                    default=0.0,
                ),
            }
        drawdown = max(
            (float(metric["drawdown"]) for metric in window_metrics.values()),
            default=0.0,
        )

        terminal_orders = [
            row for row in orders if str(row.status) in _TERMINAL_EXECUTION_STATUSES
        ]
        adverse_orders = [
            row for row in terminal_orders if str(row.status) in _ADVERSE_EXECUTION_STATUSES
        ]
        rejection_rate = (
            len(adverse_orders) / len(terminal_orders) if terminal_orders else 0.0
        )
        gross_value = sum(float(row.gross_value or 0.0) for row in fill_totals)
        fees = sum(float(row.fee or 0.0) for row in fill_totals)
        realized_cost_rate = fees / gross_value if gross_value > 0 else 0.0
        scheduled_cost_rate = PromotionStore._scheduled_one_side_rate(
            str(lane["cost_schedule_version"])
        )
        cost_ratio = (
            abs(realized_cost_rate - scheduled_cost_rate) / scheduled_cost_rate
            if scheduled_cost_rate > 0 and gross_value > 0
            else 0.0
        )
        average_nav = (
            sum(float(row.nav) for row in valid_nav) / len(valid_nav)
            if valid_nav
            else 0.0
        )
        turnover = gross_value / average_nav if average_nav > 0 else 0.0

        reconciled_batches = 0
        for batch in batches:
            conservation = dict(batch.summary_json or {}).get("conservation") or {}
            difference = conservation.get("cash_difference")
            if (
                str(batch.status) == "succeeded"
                and difference is not None
                and abs(float(difference)) <= _RECONCILIATION_TOLERANCE
            ):
                reconciled_batches += 1
        ledger_reconciled = bool(batches) and reconciled_batches == len(batches)
        feature = self.feature_drift.observe(
            version_id=version_id,
            formal_backtest_id=latest_batch["formal_backtest_id"],
            current_dataset_identity_sha256=str(
                latest_batch["daily_dataset_identity_sha256"]
            ),
            current_dataset_lineage_id=str(
                latest_batch["daily_dataset_lineage_id"]
            ),
        )
        if feature["current_end"] != latest_batch["signal_date"].isoformat():
            raise ValueError("strategy feature observation does not reach the batch signal date")
        model_required = (
            str(dict(lane.get("config_json") or {}).get("signal_source") or "factor_score")
            == "model_prediction"
        )
        calibration = (
            self.feature_drift.observe_model_calibration(
                version_id=version_id,
                formal_backtest_id=latest_batch["formal_backtest_id"],
                current_dataset_identity_sha256=str(
                    latest_batch["daily_dataset_identity_sha256"]
                ),
                current_dataset_lineage_id=str(
                    latest_batch["daily_dataset_lineage_id"]
                ),
                signal_date=latest_batch["signal_date"],
            )
            if model_required
            else None
        )
        data_integrity_ok = nav_integrity_ok
        latest_trade_date = nav_rows[0].trade_date.isoformat() if nav_rows else None
        return {
            "data_integrity_ok": data_integrity_ok,
            "ledger_reconciled": ledger_reconciled,
            "drawdown": drawdown,
            "turnover": turnover,
            "cost_ratio": cost_ratio,
            "execution_rejection_rate": rejection_rate,
            "model_calibration_drift": (
                float(calibration["model_calibration_drift"])
                if calibration is not None
                else None
            ),
            "model_calibration_required": model_required,
            "model_calibration_evidence_available": calibration is not None,
            "feature_drift": float(feature["feature_drift"]),
            "data_completeness": (
                len(valid_nav) / len(nav_rows) if nav_rows else 0.0
            ),
            "provenance": {
                "contract_version": "strategy-health-live-evidence-v2",
                "source": "isolated_simulation_ledger_and_sealed_factor_materialization",
                "strategy_version_id": version_id,
                "simulation_portfolio_id": portfolio_id,
                "promotion_stage_id": str(lane["promotion_stage_id"]),
                "observed_at": now.isoformat(),
                "evidence_trade_date": latest_trade_date,
                "feature_signal_date": latest_batch["signal_date"].isoformat(),
                "simulation_batch_id": latest_batch["simulation_batch_id"],
                "daily_dataset": latest_batch["daily_dataset"],
                "daily_dataset_identity_sha256": latest_batch[
                    "daily_dataset_identity_sha256"
                ],
                "daily_dataset_lineage_id": latest_batch[
                    "daily_dataset_lineage_id"
                ],
                "source_snapshot_id": latest_batch["source_snapshot_id"],
                "formal_backtest_id": latest_batch["formal_backtest_id"],
                "order_plan_manifest_sha256": latest_batch[
                    "order_plan_manifest_sha256"
                ],
                "portfolio_anchor_dataset_identity_sha256": str(
                    lane["daily_dataset_identity_sha256"]
                ),
                "feature_drift_evidence_available": True,
                "feature_drift_observed_at": feature["observed_at"],
                "feature_drift_current_end": feature["current_end"],
                "feature_drift_observation_sha256": feature[
                    "observation_sha256"
                ],
                "feature_drift_observation": feature,
                "model_calibration_evidence_available": calibration is not None,
                "model_calibration_status": (
                    "available" if calibration is not None else "not_applicable_factor_strategy"
                ),
                "model_calibration_observation": calibration,
                "model_calibration_observation_sha256": (
                    calibration["observation_sha256"]
                    if calibration is not None
                    else None
                ),
                "nav_integrity_ok": nav_integrity_ok,
                "nav_observations": len(nav_rows),
                "valid_nav_observations": len(valid_nav),
                "window_metrics": window_metrics,
                "terminal_order_count": len(terminal_orders),
                "adverse_order_count": len(adverse_orders),
                "realized_cost_rate": realized_cost_rate,
                "scheduled_cost_rate": scheduled_cost_rate,
                "gross_fill_value": gross_value,
                "fee_total": fees,
                "terminal_batch_count": len(batches),
                "reconciled_batch_count": reconciled_batches,
            },
        }
