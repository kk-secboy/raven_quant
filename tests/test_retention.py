import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quant_platform.api import create_app
from quant_platform.job_store import JobStore
from quant_platform.retention import RETENTION_CONFIRMATION, DataRetentionManager


def _dataset(data_root: Path, name: str, created_at: str) -> None:
    snapshot = data_root / "snapshots" / name
    qlib = data_root / "qlib" / name
    snapshot.mkdir(parents=True)
    (snapshot / "data.bin").write_bytes((name * 10).encode())
    (snapshot / "manifest.json").write_text(
        json.dumps({"name": name, "created_at": created_at, "datasets": {}}),
        encoding="utf-8",
    )
    (qlib / "metadata").mkdir(parents=True)
    (qlib / "features").mkdir()
    (qlib / "features" / "fixture.day.bin").write_bytes(b"qlib")
    (qlib / "metadata" / "provenance.json").write_text(
        json.dumps({"created_at": created_at}), encoding="utf-8"
    )


def test_retention_protects_references_and_requires_explicit_confirmation(
    database_url: str, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    _dataset(data_root, "protected", "2025-01-01T00:00:00+00:00")
    _dataset(data_root, "execution", "2025-01-01T12:00:00+00:00")
    _dataset(data_root, "eligible", "2025-01-02T00:00:00+00:00")
    _dataset(data_root, "latest", "2025-01-03T00:00:00+00:00")
    JobStore(database_url).create(
        "qlib_baseline",
        {"dataset": "protected"},
        tmp_path / "baseline.log",
    )
    JobStore(database_url).create(
        "strategy_backtest",
        {"execution_dataset": {"name": "execution"}},
        tmp_path / "strategy-backtest.log",
    )
    manager = DataRetentionManager(data_root, database_url)
    plan = manager.plan(
        keep_latest=1,
        min_age_days=1,
        now=datetime(2025, 2, 1, tzinfo=UTC),
    )
    entries = {item["name"]: item for item in plan["entries"]}
    assert entries["protected"]["state"] == "protected"
    assert entries["execution"]["state"] == "protected"
    assert entries["latest"]["state"] == "keep_latest"
    assert entries["eligible"]["state"] == "eligible"
    assert plan["eligible_bytes"] == entries["eligible"]["bytes"]

    with pytest.raises(ValueError, match="confirmation"):
        manager.apply(
            ["eligible"],
            confirmation="wrong",
            keep_latest=1,
            min_age_days=1,
        )
    result = manager.apply(
        ["eligible"],
        confirmation=RETENTION_CONFIRMATION,
        keep_latest=1,
        min_age_days=1,
    )
    assert result["status"] == "deleted"
    assert not (data_root / "snapshots" / "eligible").exists()
    assert not (data_root / "qlib" / "eligible").exists()
    assert (data_root / "snapshots" / "protected").exists()


def test_retention_api_is_dry_run_by_default(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    _dataset(data_root, "fixture", "2025-01-01T00:00:00+00:00")
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/data-retention?keep_latest=1&min_age_days=1")
    assert response.status_code == 200
    assert response.json()["entries"][0]["name"] == "fixture"
    assert (data_root / "snapshots" / "fixture").exists()


@pytest.mark.no_database
def test_display_retention_uses_persisted_inventory_without_rescanning(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    _dataset(data_root, "fixture", "2025-01-01T00:00:00+00:00")
    manager = DataRetentionManager(
        data_root,
        "postgresql+psycopg://quantlab:quantlab@127.0.0.1:1/unused",
    )
    monkeypatch.setattr(manager, "_protected_datasets", lambda: {})
    manager.refresh_display_inventory(now=datetime(2025, 2, 1, tzinfo=UTC))

    def fail_scan(path: Path) -> int:
        raise AssertionError(f"warm display plan rescanned {path}")

    monkeypatch.setattr(manager, "_directory_size", fail_scan)
    plan = manager.display_plan(
        keep_latest=1,
        min_age_days=1,
        now=datetime(2025, 2, 1, tzinfo=UTC),
    )

    assert plan["cache_state"] == "fresh"
    assert plan["refreshing"] is False
    assert plan["entries"][0]["name"] == "fixture"
    assert plan["entries"][0]["inventory_complete"] is True


@pytest.mark.no_database
def test_display_retention_never_marks_new_unmeasured_dataset_eligible(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    _dataset(data_root, "known", "2025-01-01T00:00:00+00:00")
    manager = DataRetentionManager(
        data_root,
        "postgresql+psycopg://quantlab:quantlab@127.0.0.1:1/unused",
    )
    monkeypatch.setattr(manager, "_protected_datasets", lambda: {})
    manager.refresh_display_inventory(now=datetime(2025, 2, 1, tzinfo=UTC))
    _dataset(data_root, "new-to-cache", "2025-01-02T00:00:00+00:00")
    monkeypatch.setattr(manager, "_start_inventory_refresh", lambda: None)

    plan = manager.display_plan(
        keep_latest=1,
        min_age_days=1,
        now=datetime(2025, 3, 1, tzinfo=UTC),
    )
    entries = {item["name"]: item for item in plan["entries"]}

    assert plan["cache_state"] == "stale"
    assert entries["new-to-cache"]["inventory_complete"] is False
    assert entries["new-to-cache"]["state"] != "eligible"
