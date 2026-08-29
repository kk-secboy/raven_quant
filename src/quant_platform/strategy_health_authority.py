"""Single production authority for live strategy-health evidence.

Research, manual reviews and lifecycle events may append health observations,
but only the periodic collector can authorize new capital or promotion.  This
module binds that collector row to the exact active paper ledger batch and
validates every content-addressed seal before exposing ``allow_new_risk``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from typing import Any

from sqlalchemy import select

from quant_data.database import (
    simulation_batches,
    simulation_nav,
    simulation_portfolios,
    strategy_health_snapshots,
    strategy_promotion_stages,
    strategy_versions,
)
from quant_data.execution_contract import QLIB_ORDER_PLAN_FORMAT_VERSION

from .feature_drift import validate_factor_psi_observation
from .model_calibration_drift import validate_model_calibration_observation
from .research_horizon import (
    LONG_1_3Y,
    SHORT_1_5D,
    SWING_1_6M,
    canonical_sha256,
)
from .strategy_health import COLLECTOR_ACTOR, health_allows_new_risk

LIVE_EVIDENCE_CONTRACT_VERSION = "strategy-health-live-evidence-v2"
DEFAULT_PRODUCTION_HEALTH_MAX_AGE_SECONDS = 7200
_PRODUCT_HORIZONS = frozenset({SHORT_1_5D, SWING_1_6M, LONG_1_3Y})


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    row = value._mapping if hasattr(value, "_mapping") else value
    return dict(row)


def _failure(
    reasons: list[str],
    *,
    horizon_profile: str | None = None,
    snapshot: Any = None,
) -> dict[str, Any]:
    values = _mapping(snapshot)
    return {
        "health_status": "invalid" if values else "missing",
        "allow_new_risk": False,
        "ready": False,
        "reasons": list(dict.fromkeys(reasons)),
        "reason": "; ".join(dict.fromkeys(reasons)),
        "horizon_profile": horizon_profile,
        "snapshot_id": str(values.get("id") or "") or None,
        "snapshot_sha256": str(values.get("snapshot_sha256") or "") or None,
        "as_of": (
            values["as_of"].isoformat()
            if isinstance(values.get("as_of"), datetime)
            else None
        ),
    }


def current_paper_health_binding(
    connection: Any,
    strategy_version_id: str,
) -> dict[str, Any]:
    """Resolve the unique active paper ledger and its latest succeeded batch."""

    version = connection.execute(
        select(
            strategy_versions.c.id,
            strategy_versions.c.horizon_profile,
            strategy_versions.c.horizon_contract_sha256,
        ).where(strategy_versions.c.id == strategy_version_id)
    ).first()
    if version is None:
        raise KeyError(strategy_version_id)
    horizon = str(version.horizon_profile or "")
    if horizon not in _PRODUCT_HORIZONS:
        raise ValueError("production health authority requires an explicit horizon")
    stages = connection.execute(
        select(
            strategy_promotion_stages.c.id.label("promotion_stage_id"),
            strategy_promotion_stages.c.simulation_portfolio_id,
            simulation_portfolios.c.daily_dataset_identity_sha256.label(
                "portfolio_dataset_identity_sha256"
            ),
            simulation_portfolios.c.daily_dataset_lineage_id.label(
                "portfolio_dataset_lineage_id"
            ),
        )
        .join(
            simulation_portfolios,
            simulation_portfolios.c.id
            == strategy_promotion_stages.c.simulation_portfolio_id,
        )
        .where(
            strategy_promotion_stages.c.strategy_version_id == strategy_version_id,
            strategy_promotion_stages.c.status == "active",
            simulation_portfolios.c.status == "active",
            simulation_portfolios.c.source_type == "strategy_version",
            simulation_portfolios.c.source_id == strategy_version_id,
            simulation_portfolios.c.promotion_stage_id
            == strategy_promotion_stages.c.id,
            simulation_portfolios.c.execution_adapter == "long_only",
        )
        .order_by(strategy_promotion_stages.c.stage_index.desc())
        .limit(2)
    ).all()
    if len(stages) != 1:
        raise ValueError("strategy has no unique active paper-health ledger")
    stage = stages[0]
    nav = connection.execute(
        select(simulation_nav.c.trade_date)
        .where(simulation_nav.c.portfolio_id == stage.simulation_portfolio_id)
        .order_by(
            simulation_nav.c.trade_date.desc(),
            simulation_nav.c.created_at.desc(),
        )
        .limit(1)
    ).first()
    if nav is None:
        raise ValueError("strategy health is waiting for its first paper NAV")
    batches = connection.execute(
        select(
            simulation_batches.c.id,
            simulation_batches.c.signal_date,
            simulation_batches.c.trade_date,
            simulation_batches.c.source_snapshot_id,
            simulation_batches.c.daily_dataset,
            simulation_batches.c.daily_dataset_identity_sha256,
            simulation_batches.c.daily_dataset_lineage_id,
            simulation_batches.c.target_payload_json,
        ).where(
            simulation_batches.c.portfolio_id == stage.simulation_portfolio_id,
            simulation_batches.c.trade_date == nav.trade_date,
            simulation_batches.c.status == "succeeded",
        )
    ).all()
    if len(batches) != 1:
        raise ValueError("latest paper NAV has no unique succeeded simulation batch")
    batch = batches[0]
    identity = str(batch.daily_dataset_identity_sha256 or "")
    lineage = str(batch.daily_dataset_lineage_id or "")
    source_snapshot_id = str(batch.source_snapshot_id or "")
    target_payload = batch.target_payload_json
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
    order_plan_manifest_sha256 = (
        str(governed_plan.get("manifest_sha256") or "").lower()
        if isinstance(governed_plan, dict)
        else ""
    )
    if (
        len(identity) != 64
        or len(lineage) != 64
        or lineage != str(stage.portfolio_dataset_lineage_id or "")
        or source_snapshot_id != identity
        or not isinstance(batch.signal_date, date)
        or not isinstance(batch.trade_date, date)
        or batch.signal_date > batch.trade_date
        or not isinstance(governed_plan, dict)
        or governed_plan.get("format_version") != QLIB_ORDER_PLAN_FORMAT_VERSION
        or governed_plan.get("promotion_stage_id")
        != str(stage.promotion_stage_id)
        or not formal_backtest_id
        or len(order_plan_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in order_plan_manifest_sha256
        )
    ):
        raise ValueError("latest paper batch dataset/source lineage is invalid")
    return {
        "strategy_version_id": strategy_version_id,
        "horizon_profile": horizon,
        "horizon_contract_sha256": str(version.horizon_contract_sha256 or ""),
        "promotion_stage_id": str(stage.promotion_stage_id),
        "simulation_portfolio_id": str(stage.simulation_portfolio_id),
        "simulation_batch_id": str(batch.id),
        "trade_date": batch.trade_date,
        "signal_date": batch.signal_date,
        "daily_dataset": str(batch.daily_dataset or ""),
        "daily_dataset_identity_sha256": identity,
        "daily_dataset_lineage_id": lineage,
        "source_snapshot_id": source_snapshot_id,
        "formal_backtest_id": formal_backtest_id,
        "order_plan_manifest_sha256": order_plan_manifest_sha256,
    }


def validate_production_health_snapshot(
    snapshot: Any,
    *,
    binding: dict[str, Any],
    now: datetime,
    max_age_seconds: int = DEFAULT_PRODUCTION_HEALTH_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    """Validate one collector row against the exact current paper batch."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("production health validation time must be timezone-aware")
    values = _mapping(snapshot)
    horizon = str(binding.get("horizon_profile") or "")
    if not values:
        return _failure(
            ["strategy_health_evidence_missing"],
            horizon_profile=horizon,
        )
    reasons: list[str] = []
    criteria = dict(values.get("criteria_json") or {})
    evidence = dict(values.get("evidence_json") or {})
    criteria_sha256 = str(values.get("criteria_sha256") or "")
    evidence_sha256 = str(values.get("evidence_sha256") or "")
    if canonical_sha256(criteria) != criteria_sha256:
        reasons.append("strategy_health_criteria_seal_invalid")
    if canonical_sha256(evidence) != evidence_sha256:
        reasons.append("strategy_health_evidence_seal_invalid")
    as_of = values.get("as_of")
    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        reasons.append("strategy_health_as_of_invalid")
        normalized_as_of = None
    else:
        normalized_as_of = as_of.astimezone(UTC).replace(microsecond=0)
        age_seconds = (now.astimezone(UTC) - normalized_as_of).total_seconds()
        if age_seconds < -300:
            reasons.append("strategy_health_evidence_from_future")
        elif age_seconds > max(300, int(max_age_seconds)):
            reasons.append("strategy_health_evidence_stale")
        signal_floor = datetime.combine(binding["signal_date"], time.min, tzinfo=UTC)
        if normalized_as_of < signal_floor:
            reasons.append("strategy_health_precedes_current_signal")
        snapshot_payload = {
            "contract_version": "strategy-health-snapshot-v1",
            "strategy_version_id": binding["strategy_version_id"],
            "horizon_profile": horizon,
            "horizon_contract_sha256": binding["horizon_contract_sha256"],
            "as_of": normalized_as_of.isoformat(),
            "health_status": str(values.get("health_status") or ""),
            "criteria_json": criteria,
            "criteria_sha256": criteria_sha256,
            "evidence_json": evidence,
            "evidence_sha256": evidence_sha256,
            "recorded_by": str(values.get("recorded_by") or ""),
        }
        expected_snapshot_sha256 = canonical_sha256(snapshot_payload)
        if (
            str(values.get("id") or "") != expected_snapshot_sha256
            or str(values.get("snapshot_sha256") or "") != expected_snapshot_sha256
        ):
            reasons.append("strategy_health_snapshot_seal_invalid")
    if str(values.get("strategy_version_id") or "") != binding["strategy_version_id"]:
        reasons.append("strategy_health_version_binding_invalid")
    if str(values.get("horizon_profile") or "") != horizon:
        reasons.append("strategy_health_horizon_binding_invalid")
    if values.get("recorded_by") != COLLECTOR_ACTOR:
        reasons.append("strategy_health_not_periodically_collected")
    if evidence.get("contract_version") != LIVE_EVIDENCE_CONTRACT_VERSION:
        reasons.append("strategy_health_live_evidence_missing")
    expected_trade_date = binding["trade_date"].isoformat()
    expected_signal_date = binding["signal_date"].isoformat()
    expected_pairs = {
        "strategy_version_id": binding["strategy_version_id"],
        "simulation_portfolio_id": binding["simulation_portfolio_id"],
        "promotion_stage_id": binding["promotion_stage_id"],
        "simulation_batch_id": binding["simulation_batch_id"],
        "evidence_trade_date": expected_trade_date,
        "feature_signal_date": expected_signal_date,
        "feature_drift_current_end": expected_signal_date,
        "daily_dataset": binding["daily_dataset"],
        "daily_dataset_identity_sha256": binding[
            "daily_dataset_identity_sha256"
        ],
        "daily_dataset_lineage_id": binding["daily_dataset_lineage_id"],
        "source_snapshot_id": binding["source_snapshot_id"],
        "formal_backtest_id": binding["formal_backtest_id"],
        "order_plan_manifest_sha256": binding["order_plan_manifest_sha256"],
    }
    for field, expected in expected_pairs.items():
        if evidence.get(field) != expected:
            reasons.append(f"strategy_health_{field}_binding_invalid")
    if evidence.get("feature_drift_evidence_available") is not True:
        reasons.append("strategy_health_feature_evidence_missing")
    else:
        try:
            observation = validate_factor_psi_observation(
                evidence.get("feature_drift_observation"),
                strategy_version_id=binding["strategy_version_id"],
                current_dataset_identity_sha256=binding[
                    "daily_dataset_identity_sha256"
                ],
                expected_as_of=binding["signal_date"],
            )
            if (
                observation.get("observation_sha256")
                != evidence.get("feature_drift_observation_sha256")
                or float(observation.get("feature_drift"))
                != float(evidence.get("feature_drift"))
            ):
                reasons.append("strategy_health_feature_evidence_mismatch")
        except (TypeError, ValueError):
            reasons.append("strategy_health_feature_evidence_invalid")
    if evidence.get("model_calibration_required") is True:
        if evidence.get("model_calibration_evidence_available") is not True:
            reasons.append("strategy_health_model_calibration_missing")
        else:
            try:
                observation = validate_model_calibration_observation(
                    evidence.get("model_calibration_observation"),
                    strategy_version_id=binding["strategy_version_id"],
                    current_dataset_identity_sha256=binding[
                        "daily_dataset_identity_sha256"
                    ],
                    expected_as_of=binding["signal_date"],
                )
                if (
                    observation.get("observation_sha256")
                    != evidence.get("model_calibration_observation_sha256")
                    or float(observation["model_calibration_drift"])
                    != float(evidence.get("model_calibration_drift"))
                ):
                    reasons.append("strategy_health_model_calibration_mismatch")
            except (TypeError, ValueError):
                reasons.append("strategy_health_model_calibration_invalid")
    health_status = str(values.get("health_status") or "")
    ready = not reasons
    allows = ready and health_allows_new_risk(health_status)
    return {
        "health_status": health_status if ready else "invalid",
        "observed_health_status": health_status or None,
        "allow_new_risk": allows,
        "ready": ready,
        "reasons": list(dict.fromkeys(reasons)),
        "reason": (
            "sealed collector health permits new risk"
            if allows
            else "; ".join(dict.fromkeys(reasons))
            or f"strategy_health_{health_status}_blocks_new_risk"
        ),
        "horizon_profile": horizon,
        "snapshot_id": str(values.get("id") or "") or None,
        "snapshot_sha256": str(values.get("snapshot_sha256") or "") or None,
        "as_of": as_of.isoformat() if isinstance(as_of, datetime) else None,
        "binding": {
            key: value.isoformat() if isinstance(value, date) else value
            for key, value in binding.items()
        },
    }


def load_production_health_gate(
    connection: Any,
    strategy_version_id: str,
    *,
    now: datetime | None = None,
    max_age_seconds: int = DEFAULT_PRODUCTION_HEALTH_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    """Load the only health evidence allowed to authorize production risk."""

    current = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    try:
        binding = current_paper_health_binding(connection, strategy_version_id)
    except (KeyError, TypeError, ValueError) as exc:
        return _failure([str(exc)], horizon_profile=None)
    latest_manual = connection.execute(
        select(strategy_health_snapshots)
        .where(
            strategy_health_snapshots.c.strategy_version_id == strategy_version_id,
            strategy_health_snapshots.c.recorded_by != COLLECTOR_ACTOR,
            strategy_health_snapshots.c.recorded_by.not_like("system:%"),
        )
        .order_by(
            strategy_health_snapshots.c.as_of.desc(),
            strategy_health_snapshots.c.recorded_at.desc(),
            strategy_health_snapshots.c.id.desc(),
        )
        .limit(1)
    ).first()
    manual_values = _mapping(latest_manual)
    if manual_values:
        manual_status = str(manual_values.get("health_status") or "")
        if manual_status in {"restricted", "suspended", "retired"}:
            blocked = _failure(
                [f"manual_strategy_health_{manual_status}_blocks_new_risk"],
                horizon_profile=str(binding["horizon_profile"]),
                snapshot=latest_manual,
            )
            blocked["health_status"] = manual_status
            blocked["observed_health_status"] = manual_status
            return blocked
    # Manual healthy/watch is only an explicit release of the durable manual
    # latch.  It never authorizes risk itself: a collector observation recorded
    # at or after the release must still pass every production binding below.
    snapshot = connection.execute(
        select(strategy_health_snapshots)
        .where(
            strategy_health_snapshots.c.strategy_version_id == strategy_version_id,
            strategy_health_snapshots.c.recorded_by == COLLECTOR_ACTOR,
        )
        .order_by(
            strategy_health_snapshots.c.as_of.desc(),
            strategy_health_snapshots.c.recorded_at.desc(),
            strategy_health_snapshots.c.id.desc(),
        )
        .limit(1)
    ).first()
    if snapshot is None:
        return _failure(
            ["strategy_health_collector_evidence_missing"],
            horizon_profile=str(binding["horizon_profile"]),
        )
    if manual_values and manual_status in {"healthy", "watch"}:
        collector_values = _mapping(snapshot)
        collector_as_of = collector_values.get("as_of")
        collector_recorded_at = collector_values.get("recorded_at")
        manual_as_of = manual_values.get("as_of")
        manual_recorded_at = manual_values.get("recorded_at")
        if (
            not isinstance(collector_as_of, datetime)
            or not isinstance(manual_as_of, datetime)
            or not isinstance(collector_recorded_at, datetime)
            or not isinstance(manual_recorded_at, datetime)
            or collector_as_of.tzinfo is None
            or manual_as_of.tzinfo is None
            or collector_as_of.utcoffset() is None
            or manual_as_of.utcoffset() is None
            or collector_recorded_at.tzinfo is None
            or manual_recorded_at.tzinfo is None
            or collector_recorded_at.utcoffset() is None
            or manual_recorded_at.utcoffset() is None
        ):
            return _failure(
                ["strategy_health_collector_precedes_manual_release"],
                horizon_profile=str(binding["horizon_profile"]),
                snapshot=snapshot,
            )
        collector_recorded = collector_recorded_at.astimezone(UTC)
        manual_recorded = manual_recorded_at.astimezone(UTC)
        collector_order = (
            collector_as_of.astimezone(UTC),
            collector_recorded,
            str(collector_values.get("id") or ""),
        )
        manual_order = (
            manual_as_of.astimezone(UTC),
            manual_recorded,
            str(manual_values.get("id") or ""),
        )
        # ``as_of`` can be identical to the operator release (both are
        # second-normalized).  The durable insertion timestamp therefore also
        # has to prove the collector was not recorded before the release.
        if collector_recorded < manual_recorded or collector_order < manual_order:
            return _failure(
                ["strategy_health_collector_precedes_manual_release"],
                horizon_profile=str(binding["horizon_profile"]),
                snapshot=snapshot,
            )
    return validate_production_health_snapshot(
        snapshot,
        binding=binding,
        now=current,
        max_age_seconds=max_age_seconds,
    )
