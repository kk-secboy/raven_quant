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
    _dataset(data_root, "eligible", "2025-01-02T00:00:00+00:00")
    _dataset(data_root, "latest", "2025-01-03T00:00:00+00:00")
    JobStore(database_url).create(
        "qlib_baseline",
        {"dataset": "protected"},
        tmp_path / "baseline.log",
    )
    manager = DataRetentionManager(data_root, database_url)
    plan = manager.plan(
        keep_latest=1,
        min_age_days=1,
        now=datetime(2025, 2, 1, tzinfo=UTC),
    )
    entries = {item["name"]: item for item in plan["entries"]}
    assert entries["protected"]["state"] == "protected"
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
