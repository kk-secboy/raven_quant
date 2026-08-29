from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.deployment_readiness import _strategy_health_evidence_check
from quant_platform.feature_drift import (
    FEATURE_DRIFT_FORMULA_VERSION,
    build_factor_psi_observation,
    build_strategy_health_feature_set,
    validate_factor_psi_observation,
)
from quant_platform.research_horizon import canonical_sha256
from quant_platform.strategy_feature_drift_source import StrategyFeatureDriftSource
from quant_platform.strategy_recipes import get_strategy_recipe

pytestmark = pytest.mark.no_database


def _values(
    dates: pd.DatetimeIndex,
    *,
    instruments: int = 120,
    shift: float = 0.0,
) -> pd.Series:
    index = pd.MultiIndex.from_product(
        [dates, [f"SH{600000 + item:06d}" for item in range(instruments)]],
        names=["datetime", "instrument"],
    )
    values = np.linspace(-2.0, 2.0, len(index)) + shift
    return pd.Series(values, index=index, name="factor")


def _contract(*, factor_ids: list[str], reference_end: date) -> dict:
    value = {
        "contract_version": "strategy-feature-drift-reference-v1",
        "strategy_version_id": "version-a",
        "strategy_rules_sha256": "a" * 64,
        "factor_set_sha256": "b" * 64,
        "feature_set_definition_sha256": "c" * 64,
        "factor_ids": factor_ids,
        "formal_backtest_id": "backtest-a",
        "formal_artifact_manifest_sha256": "d" * 64,
        "reference_dataset_identity_sha256": "e" * 64,
        "reference_start": "2026-06-01",
        "reference_end": reference_end.isoformat(),
        "current_window_sessions": 20,
        "bins": 10,
        "minimum_reference_sessions": 20,
        "minimum_reference_observations": 500,
        "minimum_current_observations": 100,
    }
    return {**value, "contract_sha256": canonical_sha256(value)}


def _observation(*, shift: float = 0.0, bootstrap: bool = False) -> dict:
    reference_dates = pd.bdate_range("2026-06-01", periods=30)
    current_dates = (
        pd.DatetimeIndex([reference_dates[-1]])
        if bootstrap
        else pd.bdate_range(reference_dates[-1].date() + timedelta(days=1), periods=3)
    )
    as_of = current_dates[-1].date()
    reference = {
        "factor-a": _values(reference_dates),
        "factor-b": _values(reference_dates, shift=0.25),
    }
    current = {
        "factor-a": _values(current_dates, shift=shift),
        "factor-b": _values(current_dates, shift=0.25),
    }
    return build_factor_psi_observation(
        reference_values=reference,
        current_values=current,
        contract=_contract(
            factor_ids=sorted(reference),
            reference_end=reference_dates[-1].date(),
        ),
        as_of=as_of,
        current_dataset_identity_sha256="f" * 64,
        current_dataset_lineage_id="1" * 64,
        materialization_manifest_sha256="2" * 64,
        materialized_file_sha256={name: "3" * 64 for name in reference},
        reference_file_sha256={name: "4" * 64 for name in reference},
    )


def test_strategy_feature_set_freezes_exact_baseline_expressions() -> None:
    recipe = get_strategy_recipe("short_relative_strength")
    baseline = {
        "contract_version": "qlib-six-factor-baseline-v1",
        "frequency": "day",
        "evaluation_api": "qlib.data.D.features",
        "factors": list(recipe["factor_baseline"]),
        "preprocessing": [
            "cross_sectional_winsorize_1_99",
            "cross_sectional_zscore",
            "pit_tradability_filter",
        ],
        "neutralization_stage": "build_governed_signal",
    }
    version = {
        "id": "version-a",
        "strategy_rules_sha256": "a" * 64,
        "factors": [],
        "config": {
            "signal_source": "factor_score",
            "factor_source_mode": "qlib_baseline",
            "baseline_definition": baseline,
            "baseline_definition_sha256": canonical_sha256(baseline),
        },
    }

    feature_set = build_strategy_health_feature_set(version)

    assert set(feature_set["features"]) == {
        item["id"] for item in recipe["factor_baseline"]
    }
    assert len(feature_set["definition_sha256"]) == 64


def test_feature_drift_bootstraps_from_one_complete_cross_section() -> None:
    observation = _observation(bootstrap=True)

    validated = validate_factor_psi_observation(
        observation,
        strategy_version_id="version-a",
        current_dataset_identity_sha256="f" * 64,
        expected_as_of=date.fromisoformat(observation["current_end"]),
    )

    assert validated["formula_version"] == FEATURE_DRIFT_FORMULA_VERSION
    assert all(
        item["current_observations"] == 120
        for item in validated["factors"].values()
    )


def test_feature_drift_uses_worst_shifted_factor() -> None:
    observation = _observation(shift=5.0)

    assert observation["aggregation"]["method"] == "maximum"
    assert observation["aggregation"]["max_factor_id"] == "factor-a"
    assert observation["feature_drift"] > 0.5


def test_feature_drift_rejects_tampering_and_stale_as_of() -> None:
    observation = _observation()
    stale_date = date.fromisoformat(observation["current_end"]) + timedelta(days=1)

    with pytest.raises(ValueError, match="stale"):
        validate_factor_psi_observation(
            observation,
            strategy_version_id="version-a",
            current_dataset_identity_sha256="f" * 64,
            expected_as_of=stale_date,
        )

    observation["factors"]["factor-a"]["raw_psi"] = 99.0
    with pytest.raises(ValueError, match="seal"):
        validate_factor_psi_observation(
            observation,
            strategy_version_id="version-a",
            current_dataset_identity_sha256="f" * 64,
        )


def test_materialized_factor_file_hash_is_verified(tmp_path: Path) -> None:
    identity = "f" * 64
    definition = {
        "contract_version": "strategy-health-feature-set-v1",
        "id": "strategy-health:version-a:1234567890abcdef",
        "name": "test",
        "features": {"factor-a": "$close/Ref($close,5)-1"},
        "source": "strategy-version:version-a:" + "1" * 64,
    }
    feature_set = {
        **definition,
        "definition_sha256": canonical_sha256(definition),
    }
    root = (
        tmp_path
        / "artifacts"
        / "factor-library-materializations"
        / identity
        / feature_set["definition_sha256"][:16]
    )
    values = root / "values" / "factor-a.h5"
    values.parent.mkdir(parents=True)
    _values(pd.bdate_range("2026-08-01", periods=2)).to_frame().to_hdf(
        values, key="data", mode="w"
    )
    digest = hashlib.sha256(values.read_bytes()).hexdigest()
    recent = root / "recent" / "factor-a.parquet"
    recent.parent.mkdir()
    _values(pd.bdate_range("2026-08-01", periods=2)).to_frame().to_parquet(recent)
    recent_digest = hashlib.sha256(recent.read_bytes()).hexdigest()
    manifest = {
        "contract_version": "factor-library-materialization-v1",
        "dataset_identity_sha256": identity,
        "feature_set_id": feature_set["id"],
        "feature_set_definition_sha256": feature_set["definition_sha256"],
        "feature_set": feature_set,
        "universe": "cn_all",
        "start": "2026-08-01",
        "end": "2026-08-04",
        "status": "complete",
        "completed": {
            "factor-a": {
                "relative_path": "values/factor-a.h5",
                "sha256": digest,
                "recent_relative_path": "recent/factor-a.parquet",
                "recent_sha256": recent_digest,
                "recent_start": "2026-08-03",
                "recent_end": "2026-08-04",
            }
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    source = StrategyFeatureDriftSource.__new__(StrategyFeatureDriftSource)
    source.data_root = tmp_path.resolve()

    loaded, hashes, _manifest = source._current_values(
        current_dataset_identity_sha256=identity,
        feature_set=feature_set,
        require_job_receipt=False,
    )
    assert set(loaded) == {"factor-a"}
    assert hashes == {"factor-a": recent_digest}

    recent.write_bytes(recent.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="changed"):
        source._current_values(
            current_dataset_identity_sha256=identity,
            feature_set=feature_set,
            require_job_receipt=False,
        )


def _health_row(observation: dict, *, as_of: datetime) -> dict:
    criteria = {"watch_feature_drift": 0.20}
    evidence = {
        "contract_version": "strategy-health-live-evidence-v2",
        "evidence_trade_date": observation["current_end"],
        "feature_signal_date": observation["current_end"],
        "simulation_batch_id": "batch-a",
        "daily_dataset_identity_sha256": "f" * 64,
        "daily_dataset_lineage_id": "1" * 64,
        "source_snapshot_id": "f" * 64,
        "feature_drift_current_end": observation["current_end"],
        "feature_drift_evidence_available": True,
        "feature_drift": observation["feature_drift"],
        "feature_drift_observation_sha256": observation["observation_sha256"],
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
        "recorded_by": "system:strategy-health-collector",
    }
    seal = canonical_sha256(payload)
    return {
        **payload,
        "as_of": as_of,
        "id": seal,
        "snapshot_sha256": seal,
    }


def test_readiness_accepts_only_fresh_sealed_health_observation() -> None:
    observation = _observation()
    expected = date.fromisoformat(observation["current_end"])
    as_of = datetime.combine(expected, datetime.min.time(), tzinfo=UTC) + timedelta(
        hours=8
    )
    row = _health_row(observation, as_of=as_of)

    ready = _strategy_health_evidence_check(
        row,
        version_id="version-a",
        horizon_profile="short_1_5d",
        horizon_contract_sha256="9" * 64,
        current_dataset_identity_sha256="f" * 64,
        current_dataset_lineage_id="1" * 64,
        current_batch_id="batch-a",
        current_source_snapshot_id="f" * 64,
        expected_trade_date=expected,
        now=as_of + timedelta(minutes=5),
        max_age_seconds=3600,
    )
    stale = _strategy_health_evidence_check(
        row,
        version_id="version-a",
        horizon_profile="short_1_5d",
        horizon_contract_sha256="9" * 64,
        current_dataset_identity_sha256="f" * 64,
        current_dataset_lineage_id="1" * 64,
        current_batch_id="batch-a",
        current_source_snapshot_id="f" * 64,
        expected_trade_date=expected,
        now=as_of + timedelta(hours=2),
        max_age_seconds=3600,
    )
    row["evidence_json"]["feature_drift"] = 0.99
    tampered = _strategy_health_evidence_check(
        row,
        version_id="version-a",
        horizon_profile="short_1_5d",
        horizon_contract_sha256="9" * 64,
        current_dataset_identity_sha256="f" * 64,
        current_dataset_lineage_id="1" * 64,
        current_batch_id="batch-a",
        current_source_snapshot_id="f" * 64,
        expected_trade_date=expected,
        now=as_of + timedelta(minutes=5),
        max_age_seconds=3600,
    )

    assert ready["ready"] is True
    assert "strategy_health_evidence_stale" in stale["reasons"]
    assert "strategy_health_evidence_seal_invalid" in tampered["reasons"]
