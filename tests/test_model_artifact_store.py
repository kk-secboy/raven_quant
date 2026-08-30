from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from governance_fixtures import create_strategy_version
from sqlalchemy import update

from quant_data.database import strategy_versions
from quant_platform.model_artifact_store import ModelArtifactStore
from quant_platform.model_research_governance import canonical_sha256

NOW = datetime(2026, 7, 28, 8, 0, tzinfo=UTC)
EXECUTION_ENVIRONMENT_SHA256 = "e" * 64
MODEL_RECIPE = {
    "model": "linear",
    "alpha": 0.1,
    "model_engine": "ridge_baseline",
}


def _lifecycle_store(database_url: str, tmp_path) -> tuple[ModelArtifactStore, str]:
    """Isolate artifact lifecycle tests from the separately tested admission grid."""

    version_id = create_strategy_version(
        database_url,
        tmp_path,
        recipe_id="short_relative_strength",
    )
    store = ModelArtifactStore(database_url)
    version = deepcopy(store.strategies.get_version(version_id))
    version["config"] = {
        **version["config"],
        "signal_source": "model_prediction",
    }
    version["model_signal"] = {
        "dataset": "qlib-daily",
        "model_recipe_sha256": canonical_sha256(MODEL_RECIPE),
        "recipe": deepcopy(MODEL_RECIPE),
    }
    # Admission-grid integrity has dedicated end-to-end tests.  This module
    # freezes that boundary so it can exercise candidate rotation, expiry and
    # file tampering without rebuilding nine research cells per test.
    store.strategies.get_version = lambda _version_id: deepcopy(version)
    store._governed_execution_environment_sha256 = (  # noqa: SLF001
        lambda _version_id: EXECUTION_ENVIRONMENT_SHA256
    )
    with store.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved")
        )
    return store, version_id


def _artifact(
    store: ModelArtifactStore,
    tmp_path,
    version_id: str,
    *,
    key: str,
    recipe: dict | None = None,
    valid_until: datetime | None = None,
    data_cutoff_at: datetime = NOW,
):
    path = tmp_path / f"{key}.parquet"
    pd.DataFrame(
        {"datetime": [pd.Timestamp("2026-07-28")], "instrument": ["SH600000"], "score": [0.1]}
    ).set_index(["datetime", "instrument"]).to_parquet(path)
    predictions_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint = tmp_path / f"{key}-checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "coefficients": [0.1],
                "feature_count": 1,
                "format": "ridge-numeric-v1",
                "intercept": 0.0,
                "model_engine": "ridge_baseline",
                "model_spec_sha256": "c" * 64,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model_data_contract = {
        "contract_version": "model-data-contract-v1-train-window-normalized",
        "feature_normalization": {
            "class": "RobustZScoreNorm",
            "fit_start_time": "2020-01-01",
            "fit_end_time": "2025-12-31",
        },
    }
    model_data_contract_sha256 = hashlib.sha256(
        json.dumps(
            model_data_contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return store.create(
        strategy_version_id=version_id,
        artifact_key=key,
        model_recipe=recipe or MODEL_RECIPE,
        dataset="qlib-daily",
        dataset_identity_sha256="a" * 64,
        execution_environment_sha256=EXECUTION_ENVIRONMENT_SHA256,
        training_start=date(2020, 1, 1),
        training_end=date(2025, 12, 31),
        data_cutoff_at=data_cutoff_at,
        scheduled_refit_at=data_cutoff_at,
        valid_until=valid_until or NOW + timedelta(days=90),
        artifact_path=path,
        predictions_sha256=predictions_sha256,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_format="ridge_numeric_json",
        model_data_contract_sha256=model_data_contract_sha256,
        training_kind="monthly_retrain",
        training_evidence={
            "model_engine": "ridge_baseline",
            "model_data_contract": model_data_contract,
            "operation": "retrain",
            "periods": {
                "train_start": "2020-01-01",
                "train_end": "2025-06-30",
                "valid_start": "2025-07-03",
                "valid_end": "2025-12-31",
            },
            "retrain_reason": "monthly_first_trading_day",
        },
        actor="model-trainer",
    )


def test_routine_refit_rotates_active_artifact_without_changing_spec(
    database_url: str, tmp_path
) -> None:
    store, version_id = _lifecycle_store(database_url, tmp_path)
    first = _artifact(store, tmp_path, version_id, key="refit-1")
    active_first = store.activate(first["id"], actor="model-reviewer", now=NOW)
    second = _artifact(
        store,
        tmp_path,
        version_id,
        key="refit-2",
        data_cutoff_at=NOW + timedelta(days=1),
    )
    active_second = store.activate(
        second["id"],
        actor="model-reviewer",
        now=NOW + timedelta(days=1),
    )

    assert active_first["status"] == "active"
    assert active_second["status"] == "active"
    assert store.get(first["id"])["status"] == "retired"
    selected = store.select_for_inference(
        version_id, now=NOW + timedelta(days=2)
    )
    assert selected["id"] == second["id"]
    assert selected["selection_status"] == "active"


def test_routine_refit_cannot_change_frozen_model_recipe(
    database_url: str, tmp_path
) -> None:
    store, version_id = _lifecycle_store(database_url, tmp_path)
    first = _artifact(store, tmp_path, version_id, key="refit-1")
    store.activate(first["id"], actor="model-reviewer", now=NOW)

    with pytest.raises(ValueError, match="StrategySpec"):
        _artifact(
            store,
            tmp_path,
            version_id,
            key="changed-model",
            recipe={"model": "xgboost", "depth": 8},
        )


def test_failed_refit_keeps_previous_active_model(
    database_url: str, tmp_path
) -> None:
    store, version_id = _lifecycle_store(database_url, tmp_path)
    first = _artifact(store, tmp_path, version_id, key="active")
    store.activate(first["id"], actor="model-reviewer", now=NOW)
    failed = _artifact(store, tmp_path, version_id, key="failed")
    store.mark_failed(failed["id"], reason="training data quality gate failed")

    selected = store.select_for_inference(version_id, now=NOW + timedelta(days=1))
    assert selected["id"] == first["id"]
    assert store.get(failed["id"])["status"] == "failed"


@pytest.mark.no_database
def test_live_model_artifact_must_exactly_cover_current_signal_date(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction_path = tmp_path / "daily-live.parquet"
    pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2026-07-28")] * 50,
            "instrument": [f"SH{600000 + offset:06d}" for offset in range(50)],
            "score": [0.1 + offset / 10_000 for offset in range(50)],
        }
    ).set_index(["datetime", "instrument"]).to_parquet(prediction_path)
    prediction_sha256 = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    selected = {
        "id": "artifact-live",
        "selection_status": "active",
        "dataset_identity_sha256": "a" * 64,
        "artifact_path": str(prediction_path),
        "artifact_sha256": prediction_sha256,
        "predictions_sha256": prediction_sha256,
        "checkpoint_path": str(tmp_path / "checkpoint.json"),
        "checkpoint_sha256": "c" * 64,
        "checkpoint_format": "ridge_numeric_json",
        "training_evidence": {"model_engine": "ridge_baseline"},
    }
    store = object.__new__(ModelArtifactStore)
    store.strategies = SimpleNamespace(
        get_version=lambda _version_id: {
            "config": {"signal_source": "model_prediction"}
        }
    )
    store.select_for_inference = lambda *_args, **_kwargs: selected
    monkeypatch.setattr(
        "quant_platform.model_artifact_store.verify_governed_checkpoint",
        lambda *_args, **_kwargs: None,
    )

    resolved = store.require_for_inference(
        "version-live",
        dataset_identity_sha256="a" * 64,
        signal_date=date(2026, 7, 28),
        now=NOW,
    )
    assert resolved["id"] == "artifact-live"

    with pytest.raises(ValueError, match="exact OOS window"):
        store.require_for_inference(
            "version-live",
            dataset_identity_sha256="a" * 64,
            signal_date=date(2026, 7, 29),
            now=NOW + timedelta(days=1),
        )


def test_expired_active_model_falls_back_to_simple_baseline(
    database_url: str, tmp_path
) -> None:
    store, version_id = _lifecycle_store(database_url, tmp_path)
    artifact = _artifact(
        store,
        tmp_path,
        version_id,
        key="short-lived",
        valid_until=NOW + timedelta(days=1),
    )
    store.activate(artifact["id"], actor="model-reviewer", now=NOW)

    selected = store.select_for_inference(
        version_id, now=NOW + timedelta(days=2)
    )
    assert selected == {
        "status": "simple_baseline_required",
        "reason": "active_model_artifact_expired",
        "contract_version": "model-artifact-lifecycle-v2-checkpointed",
    }


def test_artifact_tampering_is_rejected_on_activation(
    database_url: str, tmp_path
) -> None:
    store, version_id = _lifecycle_store(database_url, tmp_path)
    artifact = _artifact(store, tmp_path, version_id, key="tampered")
    (tmp_path / "tampered.parquet").write_bytes(b"changed")

    with pytest.raises(ValueError, match="immutable verification"):
        store.activate(artifact["id"], actor="model-reviewer", now=NOW)
