from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import requests
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import update

from quant_data.config import Settings
from quant_data.database import jobs
from quant_platform.api import create_app
from quant_platform.deployment_readiness import DeploymentReadinessStore
from quant_platform.health_store import (
    OperationalHealthStore,
    safe_mode_recovery_health_status,
)
from quant_platform.job_store import JobStore
from quant_platform.runtime_secret_store import RuntimeSecretStore
from quant_platform.safe_mode import SafeModeStore
from quant_platform.scheduler import SchedulerEngine


def _dataset(data_root: Path, end_date: str, *, name: str = "health-snapshot") -> None:
    root = data_root / "qlib" / name
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text(f"2024-01-02\n{end_date}\n", encoding="utf-8")
    (root / "instruments" / "cn_all.txt").write_text(
        f"SH600000\t2024-01-02\t{end_date}\n", encoding="utf-8"
    )


def _settings(database_url: str, data_root: Path, **values: object) -> Settings:
    base: dict[str, object] = {
        "api_url": "https://api.tushare.pro",
        "token": "token",
        "data_root": data_root,
        "database_url": database_url,
        "embedded_worker": True,
        "rdagent_enabled": False,
        "health_snapshot_seconds": 300,
        "platform_secret_key": Fernet.generate_key().decode("ascii"),
    }
    base.update(values)
    return Settings(**base)  # type: ignore[arg-type]


def test_health_history_records_fresh_data_and_api_exposes_it(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    data_root = tmp_path / "data"
    _dataset(data_root, now.date().isoformat())
    settings = _settings(database_url, data_root)
    store = OperationalHealthStore(settings)
    snapshot = store.collect_and_record(now)
    assert snapshot["status"] == "ok"
    assert snapshot["components"]["market_data"]["age_days"] == 0
    assert store.due(now + timedelta(seconds=299)) is False
    assert store.due(now + timedelta(seconds=300)) is True

    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("AUTH_MODE", "disabled")
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/operations/health?limit=10")
    assert response.status_code == 200
    assert response.json()["latest"]["id"] == snapshot["id"]
    assert len(response.json()["history"]) == 1


def _configure_readyz_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    database_url: str,
    data_root: Path,
    platform_secret_key: str,
) -> None:
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("AUTH_MODE", "required")
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "true")
    monkeypatch.setenv("RDAGENT_ENABLED", "false")
    monkeypatch.setenv("PLATFORM_SECRET_KEY", platform_secret_key)
    monkeypatch.setenv("TUSHARE_API_URL", "https://api.tushare.pro")
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setenv("HEALTH_SNAPSHOT_SECONDS", "60")
    monkeypatch.setenv("QUANTLAB_RELEASE_ID", "release-test-1")
    monkeypatch.setenv("QUANTLAB_CONFIG_DIGEST", "config-test-1")
    monkeypatch.delenv("SCHEDULER_URL", raising=False)


def test_readyz_is_public_and_requires_current_business_health(
    database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    data_root = tmp_path / "data"
    _dataset(data_root, now.date().isoformat())
    key = Fernet.generate_key().decode("ascii")
    settings = _settings(
        database_url,
        data_root,
        health_snapshot_seconds=60,
        platform_secret_key=key,
        quantlab_release_id="release-test-1",
        quantlab_config_digest="config-test-1",
    )
    OperationalHealthStore(settings).collect_and_record(now)
    _configure_readyz_environment(
        monkeypatch,
        database_url=database_url,
        data_root=data_root,
        platform_secret_key=key,
    )
    monkeypatch.setattr(
        DeploymentReadinessStore,
        "business_loop_readiness",
        lambda _self: {
            "status": "ok",
            "checks": {
                "daily_qlib_data": {"status": "ok", "message": "fresh"},
                "three_horizon_production": {
                    "status": "ok",
                    "message": "paper lanes active",
                },
            },
            "blockers": [],
        },
    )

    with TestClient(create_app(tmp_path)) as client:
        ready = client.get("/api/readyz")
        SafeModeStore(database_url).activate(
            reason="readiness regression fixture",
            source="manual",
            actor="test",
        )
        blocked = client.get("/api/readyz")

    assert ready.status_code == 200
    assert ready.headers["cache-control"] == "no-store"
    assert ready.json()["status"] == "ready"
    assert ready.json()["checks"]["scheduler"]["source"] == (
        "durable_health_snapshot"
    )
    assert blocked.status_code == 503
    assert blocked.json()["status"] == "not_ready"
    assert blocked.json()["checks"]["safe_mode"] == {
        "status": "blocked",
        "message": "safe mode is active",
        "active": True,
    }


def test_readyz_rejects_a_stale_scheduler_heartbeat(
    database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded_at = datetime.now(UTC).replace(microsecond=0) - timedelta(minutes=3)
    data_root = tmp_path / "data"
    _dataset(data_root, recorded_at.date().isoformat())
    key = Fernet.generate_key().decode("ascii")
    settings = _settings(
        database_url,
        data_root,
        health_snapshot_seconds=60,
        platform_secret_key=key,
    )
    OperationalHealthStore(settings).collect_and_record(recorded_at)
    _configure_readyz_environment(
        monkeypatch,
        database_url=database_url,
        data_root=data_root,
        platform_secret_key=key,
    )

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert "scheduler_heartbeat" in body["checks"]["operational_health"][
        "blocking_components"
    ]
    assert body["checks"]["scheduler"]["status"] == "unavailable"


def test_health_accepts_intraday_qlib_calendar_timestamp(
    database_url: str, tmp_path: Path
) -> None:
    now = datetime(2026, 8, 21, 8, 0, tzinfo=UTC)
    data_root = tmp_path / "data"
    _dataset(data_root, "2026-08-21 15:00:00")

    observation = OperationalHealthStore(_settings(database_url, data_root)).collect(now)

    market_data = observation["components"]["market_data"]
    assert market_data["status"] == "ok"
    assert market_data["end_date"] == "2026-08-21 15:00:00"
    assert market_data["age_days"] == 0


def test_health_degrades_instead_of_crashing_on_invalid_qlib_calendar_boundary(
    database_url: str, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    _dataset(data_root, "invalid-calendar-boundary")

    observation = OperationalHealthStore(_settings(database_url, data_root)).collect()

    market_data = observation["components"]["market_data"]
    assert observation["status"] == "degraded"
    assert market_data["status"] == "degraded"
    assert market_data["invalid_datasets"] == [
        {"dataset": "health-snapshot", "end_date": "invalid-calendar-boundary"}
    ]


def test_health_fails_closed_if_any_ready_qlib_calendar_boundary_is_invalid(
    database_url: str, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    _dataset(data_root, "2026-08-21", name="valid-dataset")
    _dataset(data_root, "!invalid", name="invalid-dataset")

    observation = OperationalHealthStore(_settings(database_url, data_root)).collect()

    market_data = observation["components"]["market_data"]
    assert market_data["status"] == "degraded"
    assert market_data["invalid_datasets"] == [
        {"dataset": "invalid-dataset", "end_date": "!invalid"}
    ]


def test_health_rejects_timezone_aware_qlib_calendar_timestamp(
    database_url: str, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    _dataset(data_root, "2026-08-21T15:00:00+08:00")

    observation = OperationalHealthStore(_settings(database_url, data_root)).collect()

    assert observation["components"]["market_data"]["status"] == "degraded"


@pytest.mark.no_database
def test_safe_mode_recovery_health_ignores_only_safe_mode_component() -> None:
    healthy_except_safe_mode = {
        "status": "degraded",
        "recorded_at": "2026-08-29T12:00:01+00:00",
        "components": {
            "postgresql": {"status": "ok"},
            "safe_mode": {"status": "degraded"},
            "broker_boundary": {"status": "not_applicable"},
        },
    }
    assert safe_mode_recovery_health_status(healthy_except_safe_mode) == "ok"
    assert (
        safe_mode_recovery_health_status(
            healthy_except_safe_mode,
            triggered_at="2026-08-29T12:00:00+00:00",
        )
        == "ok"
    )
    assert (
        safe_mode_recovery_health_status(
            healthy_except_safe_mode,
            triggered_at="2026-08-29T12:00:01+00:00",
        )
        == "degraded"
    )

    unhealthy_worker = {
        **healthy_except_safe_mode,
        "components": {
            **healthy_except_safe_mode["components"],
            "qlib_worker": {"status": "unavailable"},
        },
    }
    assert safe_mode_recovery_health_status(unhealthy_worker) == "degraded"
    assert safe_mode_recovery_health_status(None) == "missing"


def test_safe_mode_release_accepts_health_degraded_only_by_safe_mode(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    data_root = tmp_path / "data"
    _dataset(data_root, now.date().isoformat())
    key = Fernet.generate_key().decode("ascii")
    settings = _settings(database_url, data_root, platform_secret_key=key)
    safe_mode = SafeModeStore(database_url)
    safe_mode.activate(
        reason="data quality recovery fixture",
        source="data_quality_gate",
        actor="system",
    )
    snapshot = OperationalHealthStore(settings).collect_and_record(
        datetime.now(UTC) + timedelta(seconds=1)
    )
    assert snapshot["status"] == "degraded"
    assert snapshot["components"]["safe_mode"]["status"] == "degraded"
    assert safe_mode_recovery_health_status(snapshot) == "ok"

    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("AUTH_MODE", "disabled")
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setenv("RDAGENT_ENABLED", "false")
    monkeypatch.setenv("PLATFORM_SECRET_KEY", key)
    monkeypatch.setenv("TUSHARE_API_URL", "https://api.tushare.pro")
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setattr(
        DeploymentReadinessStore,
        "business_loop_readiness",
        lambda _self: {
            "status": "blocked",
            "checks": {
                "daily_qlib_data": {"status": "ok"},
                "three_horizon_production": {"status": "blocked"},
            },
            "blockers": [{"check": "three_horizon_production"}],
        },
    )
    app = create_app(tmp_path)
    with TestClient(app) as client:
        bypass = client.post(
            "/api/platform/safe-mode/release",
            json={
                "actor": "recovery-operator",
                "reason": "attempted recovery without the mandatory health gate",
                "require_health_ok": False,
            },
        )
        assert bypass.status_code == 422
        response = client.post(
            "/api/platform/safe-mode/release",
            json={
                "actor": "recovery-operator",
                "reason": "data quality gate passed and publication was verified",
            },
        )
    assert response.status_code == 200
    assert response.json()["active"] is False


def test_safe_mode_release_rejects_stale_daily_data(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    data_root = tmp_path / "data"
    _dataset(data_root, now.date().isoformat())
    key = Fernet.generate_key().decode("ascii")
    settings = _settings(database_url, data_root, platform_secret_key=key)
    safe_mode = SafeModeStore(database_url)
    safe_mode.activate(
        reason="data quality recovery fixture",
        source="data_quality_gate",
        actor="system",
    )
    OperationalHealthStore(settings).collect_and_record(datetime.now(UTC) + timedelta(seconds=1))

    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("AUTH_MODE", "disabled")
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setenv("RDAGENT_ENABLED", "false")
    monkeypatch.setenv("PLATFORM_SECRET_KEY", key)
    monkeypatch.setenv("TUSHARE_API_URL", "https://api.tushare.pro")
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setattr(
        DeploymentReadinessStore,
        "business_loop_readiness",
        lambda _self: {
            "status": "blocked",
            "checks": {
                "daily_qlib_data": {"status": "blocked"},
                "three_horizon_production": {"status": "blocked"},
            },
            "blockers": [
                {"check": "daily_qlib_data"},
                {"check": "three_horizon_production"},
            ],
        },
    )
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/api/platform/safe-mode/release",
            json={
                "actor": "recovery-operator",
                "reason": "attempted recovery with stale daily data",
            },
        )

    assert response.status_code == 409
    assert safe_mode.status()["active"] is True


def test_stale_running_job_degrades_health(database_url: str, tmp_path: Path) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    data_root = tmp_path / "data"
    _dataset(data_root, now.date().isoformat())
    settings = _settings(database_url, data_root, stale_job_hours=1)
    job = JobStore(database_url).create("bootstrap", {"profile": "core"}, tmp_path / "job.log")
    with OperationalHealthStore(settings).engine.begin() as connection:
        connection.execute(
            update(jobs)
            .where(jobs.c.id == job["id"])
            .values(status="running", started_at=now - timedelta(hours=2))
        )
    observation = OperationalHealthStore(settings).collect(now)
    assert observation["status"] == "degraded"
    assert observation["components"]["job_queue"]["stale_running"] == 1


def test_recent_failed_job_is_audited_without_degrading_idle_queue(
    database_url: str, tmp_path: Path
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    data_root = tmp_path / "data"
    _dataset(data_root, now.date().isoformat())
    settings = _settings(database_url, data_root)
    job = JobStore(database_url).create(
        "bootstrap", {"profile": "full"}, tmp_path / "failed-job.log"
    )
    with OperationalHealthStore(settings).engine.begin() as connection:
        connection.execute(
            update(jobs)
            .where(jobs.c.id == job["id"])
            .values(status="failed", finished_at=now)
        )

    observation = OperationalHealthStore(settings).collect(now)

    queue = observation["components"]["job_queue"]
    assert queue["status"] == "ok"
    assert queue["queued"] == 0
    assert queue["running"] == 0
    assert queue["failed_24h"] == 1
    assert observation["status"] == "ok"


def test_health_hot_loads_encrypted_tushare_credentials(
    database_url: str, tmp_path: Path
) -> None:
    key = Fernet.generate_key().decode("ascii")
    settings = _settings(
        database_url,
        tmp_path / "data",
        api_url="",
        token="",
        platform_secret_key=key,
    )
    store = OperationalHealthStore(settings)
    before = store.collect()["components"]["credentials"]
    assert before["status"] == "bootstrap_required"

    RuntimeSecretStore(database_url, key).put(
        "tushare",
        {"api_url": "https://api.tushare.pro", "token": "dynamic-token"},
        metadata={"api_url": "https://api.tushare.pro"},
        updated_by=None,
    )
    after = store.collect()["components"]["credentials"]
    assert after == {"status": "ok", "message": "Tushare credentials configured"}


def test_health_fails_closed_instead_of_using_environment_when_key_is_missing(
    database_url: str, tmp_path: Path
) -> None:
    key = Fernet.generate_key().decode("ascii")
    RuntimeSecretStore(database_url, key).put(
        "tushare",
        {"api_url": "https://api.tushare.pro", "token": "database-token"},
        metadata={"api_url": "https://api.tushare.pro"},
        updated_by=None,
    )
    settings = _settings(
        database_url,
        tmp_path / "data",
        api_url="https://environment.example/api",
        token="stale-environment-token",
        platform_secret_key="",
    )
    components = OperationalHealthStore(settings).collect()["components"]
    assert components["runtime_secret_storage"]["status"] == "unavailable"
    assert components["credentials"]["status"] == "unavailable"


def test_scheduler_projects_unavailable_worker_health_alert(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(
        database_url,
        tmp_path / "data",
        embedded_worker=False,
        qlib_worker_url="http://qlib-worker:8770",
    )

    monkeypatch.setattr(
        "quant_platform.health_store.requests.get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            requests.RequestException("worker unavailable")
        ),
    )
    result = SchedulerEngine(settings).tick(datetime.now(UTC).replace(microsecond=0))
    assert result["health_recorded"] == 1
    alerts = SchedulerEngine(settings).alerts.list()
    assert any(item["category"] == "component_health" for item in alerts)
