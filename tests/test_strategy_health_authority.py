from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

import quant_platform.strategy_health_authority as authority_module
from quant_platform.member_risk_gate import compose_strategy_risk_state
from quant_platform.research_horizon import canonical_sha256
from quant_platform.strategy_health_authority import (
    COLLECTOR_ACTOR,
    validate_production_health_snapshot,
)

pytestmark = pytest.mark.no_database


def _binding() -> dict:
    return {
        "strategy_version_id": "version-a",
        "horizon_profile": "short_1_5d",
        "horizon_contract_sha256": "9" * 64,
        "promotion_stage_id": "stage-a",
        "simulation_portfolio_id": "portfolio-a",
        "simulation_batch_id": "batch-a",
        "trade_date": date(2026, 8, 28),
        "signal_date": date(2026, 8, 27),
        "daily_dataset": "daily-a",
        "daily_dataset_identity_sha256": "f" * 64,
        "daily_dataset_lineage_id": "1" * 64,
        "source_snapshot_id": "f" * 64,
    }


def _snapshot(*, as_of: datetime, actor: str = COLLECTOR_ACTOR) -> dict:
    criteria = {"watch_feature_drift": 0.20}
    observation = {
        "observation_sha256": "8" * 64,
        "feature_drift": 0.01,
    }
    evidence = {
        "contract_version": "strategy-health-live-evidence-v2",
        "strategy_version_id": "version-a",
        "simulation_portfolio_id": "portfolio-a",
        "promotion_stage_id": "stage-a",
        "evidence_trade_date": "2026-08-28",
        "feature_signal_date": "2026-08-27",
        "simulation_batch_id": "batch-a",
        "daily_dataset": "daily-a",
        "daily_dataset_identity_sha256": "f" * 64,
        "daily_dataset_lineage_id": "1" * 64,
        "source_snapshot_id": "f" * 64,
        "feature_drift_current_end": "2026-08-27",
        "feature_drift_evidence_available": True,
        "feature_drift": 0.01,
        "feature_drift_observation_sha256": "8" * 64,
        "feature_drift_observation": observation,
        "model_calibration_required": False,
    }
    criteria_sha256 = canonical_sha256(criteria)
    evidence_sha256 = canonical_sha256(evidence)
    payload = {
        "contract_version": "strategy-health-snapshot-v1",
        "strategy_version_id": "version-a",
        "horizon_profile": "short_1_5d",
        "horizon_contract_sha256": "9" * 64,
        "as_of": as_of.isoformat(),
        "health_status": "healthy",
        "criteria_json": criteria,
        "criteria_sha256": criteria_sha256,
        "evidence_json": evidence,
        "evidence_sha256": evidence_sha256,
        "recorded_by": actor,
    }
    seal = canonical_sha256(payload)
    return {
        **payload,
        "as_of": as_of,
        "id": seal,
        "snapshot_sha256": seal,
    }


def test_production_health_requires_fresh_collector_batch_binding(monkeypatch) -> None:
    monkeypatch.setattr(
        authority_module,
        "validate_factor_psi_observation",
        lambda value, **_kwargs: dict(value),
    )
    as_of = datetime(2026, 8, 28, 8, 0, tzinfo=UTC)
    binding = _binding()
    collector = _snapshot(as_of=as_of)

    ready = validate_production_health_snapshot(
        collector,
        binding=binding,
        now=as_of + timedelta(minutes=5),
        max_age_seconds=3600,
    )
    arbitrary = validate_production_health_snapshot(
        _snapshot(as_of=as_of, actor="system:auto-promotion"),
        binding=binding,
        now=as_of + timedelta(minutes=5),
        max_age_seconds=3600,
    )
    stale = validate_production_health_snapshot(
        collector,
        binding=binding,
        now=as_of + timedelta(hours=2),
        max_age_seconds=3600,
    )
    replaced_batch = validate_production_health_snapshot(
        collector,
        binding={**binding, "simulation_batch_id": "batch-b"},
        now=as_of + timedelta(minutes=5),
        max_age_seconds=3600,
    )

    assert ready["ready"] is True
    assert ready["allow_new_risk"] is True
    assert arbitrary["allow_new_risk"] is False
    assert "strategy_health_not_periodically_collected" in arbitrary["reasons"]
    assert "strategy_health_evidence_stale" in stale["reasons"]
    assert replaced_batch["allow_new_risk"] is False
    assert "strategy_health_simulation_batch_id_binding_invalid" in replaced_batch[
        "reasons"
    ]


def test_composed_risk_gate_never_reinterprets_unready_healthy_as_authorized() -> None:
    state = compose_strategy_risk_state(
        "version-a",
        strategy_health_gate={
            "health_status": "healthy",
            "allow_new_risk": False,
            "ready": False,
            "reasons": ["strategy_health_evidence_stale"],
        },
    )

    assert state["allow_new_risk"] is False
    assert state["state"] == "strategy_health_invalid"
