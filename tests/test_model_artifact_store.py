from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from governance_fixtures import create_strategy_version

from quant_platform.model_artifact_store import ModelArtifactStore

NOW = datetime(2026, 7, 28, 8, 0, tzinfo=UTC)


def _artifact(
    store: ModelArtifactStore,
    tmp_path,
    version_id: str,
    *,
    key: str,
    recipe: dict | None = None,
    valid_until: datetime | None = None,
):
    path = tmp_path / f"{key}.bin"
    path.write_bytes(f"artifact:{key}".encode())
    return store.create(
        strategy_version_id=version_id,
        artifact_key=key,
        model_recipe=recipe or {"model": "linear", "alpha": 0.1},
        dataset="qlib-daily",
        dataset_identity_sha256="a" * 64,
        training_start=date(2020, 1, 1),
        training_end=date(2025, 12, 31),
        data_cutoff_at=NOW,
        scheduled_refit_at=NOW,
        valid_until=valid_until or NOW + timedelta(days=90),
        artifact_path=path,
        predictions_sha256="b" * 64,
        actor="model-trainer",
    )


def test_routine_refit_rotates_active_artifact_without_changing_spec(
    database_url: str, tmp_path
) -> None:
    version_id = create_strategy_version(database_url, tmp_path)
    store = ModelArtifactStore(database_url)
    first = _artifact(store, tmp_path, version_id, key="refit-1")
    active_first = store.activate(first["id"], actor="model-reviewer", now=NOW)
    second = _artifact(store, tmp_path, version_id, key="refit-2")
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
    version_id = create_strategy_version(database_url, tmp_path)
    store = ModelArtifactStore(database_url)
    first = _artifact(store, tmp_path, version_id, key="refit-1")
    store.activate(first["id"], actor="model-reviewer", now=NOW)

    with pytest.raises(ValueError, match="new StrategySpec"):
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
    version_id = create_strategy_version(database_url, tmp_path)
    store = ModelArtifactStore(database_url)
    first = _artifact(store, tmp_path, version_id, key="active")
    store.activate(first["id"], actor="model-reviewer", now=NOW)
    failed = _artifact(store, tmp_path, version_id, key="failed")
    store.mark_failed(failed["id"], reason="training data quality gate failed")

    selected = store.select_for_inference(version_id, now=NOW + timedelta(days=1))
    assert selected["id"] == first["id"]
    assert store.get(failed["id"])["status"] == "failed"


def test_expired_active_model_falls_back_to_simple_baseline(
    database_url: str, tmp_path
) -> None:
    version_id = create_strategy_version(database_url, tmp_path)
    store = ModelArtifactStore(database_url)
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
        "contract_version": "model-artifact-lifecycle-v1",
    }


def test_artifact_tampering_is_rejected_on_activation(
    database_url: str, tmp_path
) -> None:
    version_id = create_strategy_version(database_url, tmp_path)
    store = ModelArtifactStore(database_url)
    artifact = _artifact(store, tmp_path, version_id, key="tampered")
    (tmp_path / "tampered.bin").write_bytes(b"changed")

    with pytest.raises(ValueError, match="immutable verification"):
        store.activate(artifact["id"], actor="model-reviewer", now=NOW)
