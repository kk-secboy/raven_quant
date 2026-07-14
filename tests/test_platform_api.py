import json
from pathlib import Path

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from quant_platform.alert_store import AlertStore
from quant_platform.api import create_app
from quant_platform.job_store import JobStore


def test_api_reports_empty_local_state(tmp_path: Path, monkeypatch, database_url: str) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("TUSHARE_API_URL", raising=False)
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("PLATFORM_SECRET_KEY", Fernet.generate_key().decode("ascii"))
    app = create_app(tmp_path)
    with TestClient(app) as client:
        health = client.get("/api/health")
        overview = client.get("/api/overview")
        readiness = client.get("/api/operations/readiness")
        datasets = client.get("/api/datasets")
        bootstrap = client.post(
            "/api/jobs/bootstrap",
            json={"profile": "core", "start": "2024-01-01", "end": "latest"},
        )
    assert health.json() == {
        "status": "ok",
        "database": "postgresql",
        "worker_mode": "embedded",
        "runtime_secret_storage": "ok",
        "runtime_secret_records": 0,
    }
    assert overview.json()["credentials_configured"] is False
    assert overview.json()["readiness_percent"] < 100
    assert overview.json()["actionable_tasks"] > overview.json()["ready_tasks"]
    assert overview.json()["legacy_download_coverage"] == overview.json()["coverage"]
    assert readiness.status_code == 200
    assert readiness.json()["profiles"][0]["status"] == "blocked"
    assert any(item["name"] == "daily" for item in datasets.json())
    assert bootstrap.status_code == 409
    assert bootstrap.json()["detail"] == (
        "missing deployment secret: TUSHARE_API_URL, TUSHARE_TOKEN"
    )
    assert client.get("/api/qlib/datasets").json() == []
    assert client.get("/api/qlib/experiments").json() == []
    assert client.get("/api/strategy-allocations").json() == []
    allocation = client.post(
        "/api/strategy-allocations",
        json={
            "name": "missing dataset allocation",
            "dataset": "missing",
            "members": [
                {"strategy_version_id": "version-a"},
                {"strategy_version_id": "version-b"},
            ],
        },
    )
    assert allocation.status_code == 409
    assert "strategy allocation" in allocation.json()["detail"]
    allocation_schedule = client.post(
        "/api/strategy-allocations/missing/schedule",
        json={"run_time": "15:30", "actor": "operator"},
    )
    assert allocation_schedule.status_code == 404
    assert allocation_schedule.json()["detail"] == "strategy allocation not found"
    invalid_schedule = client.post(
        "/api/strategy-allocations/missing/schedule",
        json={"run_time": "14:30", "actor": "operator"},
    )
    assert invalid_schedule.status_code == 422
    qlib_job = client.post(
        "/api/jobs/qlib-baseline",
        json={"dataset": "missing", "topk": 50, "n_drop": 5},
    )
    assert qlib_job.status_code == 409


def test_job_store_is_durable_and_exclusive(tmp_path: Path, database_url: str) -> None:
    store = JobStore(database_url)
    first = store.create("bootstrap", {"profile": "core"}, tmp_path / "job.log")
    assert first["status"] == "queued"
    claimed = store.claim_next()
    assert claimed and claimed["id"] == first["id"]
    assert claimed["status"] == "running"
    store.finish(first["id"], exit_code=0, result={"metric": 1.0})
    completed = store.get(first["id"])
    assert completed["status"] == "succeeded"
    assert completed["progress"] == {"metric": 1.0}


def test_api_exposes_fail_closed_broker_boundary(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setenv("BROKER_MODE", "disabled")
    app = create_app(tmp_path)
    with TestClient(app) as client:
        state = client.get("/api/broker")
        destination = client.post(
            "/api/broker/destinations",
            json={
                "name": "QMT sandbox",
                "account_ref": "SIM-API",
                "portfolio_id": "missing-portfolio",
            },
        )
    assert state.status_code == 200
    assert state.json()["readiness"]["status"] == "disabled"
    assert state.json()["readiness"]["live_supported"] is False
    assert destination.status_code == 404
    assert destination.json()["detail"] == "paper portfolio not found"


def test_api_creates_bounded_rdagent_research_run(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    dataset = data_root / "qlib" / "research-snapshot"
    (dataset / "calendars").mkdir(parents=True)
    (dataset / "instruments").mkdir()
    (dataset / "features").mkdir()
    (dataset / "calendars" / "day.txt").write_text("2018-01-01\n2026-07-10\n", encoding="utf-8")
    (dataset / "instruments" / "cn_all.txt").write_text(
        "SH600000\t2018-01-01\t2026-07-10\n", encoding="utf-8"
    )
    (dataset / "metadata").mkdir()
    (dataset / "metadata" / "provenance.json").write_text(
        json.dumps(
            {
                "dataset_identity_sha256": "a" * 64,
                "snapshot_manifest_sha256": "b" * 64,
                "qlib_builder_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setattr(
        "quant_platform.api.probe_rdagent",
        lambda _settings, _root: {"status": "ok", "ready": True, "blockers": []},
    )
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/api/rdagent/runs",
            json={
                "objective": "Find low-turnover quality factors for CSI 300 enhancement.",
                "dataset": "research-snapshot",
                "loop_n": 1,
                "duration": "30m",
                "periods": {
                    "train_start": "2018-01-01",
                    "train_end": "2021-12-31",
                    "valid_start": "2022-01-01",
                    "valid_end": "2023-12-31",
                    "test_start": "2024-01-01",
                    "test_end": "2026-07-10",
                },
            },
        )
        runs = client.get("/api/rdagent/runs").json()
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["budget"] == {"loop_n": 1, "duration": "30m"}
    assert runs[0]["id"] == response.json()["id"]


def test_api_manages_schedules_and_alert_acknowledgement(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    alert = AlertStore(database_url).create(
        source_type="job",
        source_id="failed-job",
        severity="critical",
        category="job_failure",
        title="Scheduled job failed",
        message="fixture failure",
        dedupe_key="api-alert-fixture",
    )
    app = create_app(tmp_path)
    with TestClient(app) as client:
        created = client.post(
            "/api/schedules",
            json={
                "name": "daily full sync",
                "kind": "incremental_sync",
                "timezone": "Asia/Shanghai",
                "run_time": "18:00",
                "trading_days_only": True,
                "payload": {"profile": "full", "lookback_days": 7, "build_qlib": True},
                "misfire_grace_seconds": 3600,
                "actor": "operator",
            },
        )
        schedules = client.get("/api/schedules").json()
        acknowledged = client.post(
            f"/api/alerts/{alert['id']}/acknowledge",
            json={"actor": "risk-owner"},
        )
    assert created.status_code == 201
    assert schedules[0]["kind"] == "incremental_sync"
    assert acknowledged.status_code == 200
    assert acknowledged.json()["status"] == "acknowledged"
