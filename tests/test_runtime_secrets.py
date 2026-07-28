from __future__ import annotations

from pathlib import Path

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import select

from quant_data.config import Settings
from quant_data.database import open_database, runtime_secrets
from quant_platform.api import create_app
from quant_platform.job_store import JobStore
from quant_platform.runtime_secret_store import RuntimeSecretStore
from quant_platform.scheduler import SchedulerEngine
from quant_platform.worker import LocalJobWorker


class _ValidTushareResponse:
    def raise_for_status(self) -> None:
        return

    def json(self) -> dict:
        return {"code": 0, "msg": None, "data": {"fields": [], "items": []}}


def test_runtime_secret_store_encrypts_and_never_describes_plaintext(database_url: str) -> None:
    key = Fernet.generate_key().decode("ascii")
    store = RuntimeSecretStore(database_url, key)
    store.put(
        "tushare",
        {"api_url": "https://api.tushare.pro", "token": "secret-token-value"},
        metadata={"api_url": "https://api.tushare.pro", "verified_at": "2026-07-12"},
        updated_by=None,
    )

    assert store.get("tushare") == {
        "api_url": "https://api.tushare.pro",
        "token": "secret-token-value",
    }
    description = store.describe("tushare")
    assert description and "secret-token-value" not in str(description)
    with open_database(database_url).connect() as connection:
        ciphertext = connection.scalar(
            select(runtime_secrets.c.ciphertext).where(runtime_secrets.c.name == "tushare")
        )
    assert ciphertext and "secret-token-value" not in ciphertext
    assert store.health() == {
        "status": "ok",
        "message": "encrypted runtime storage ready; 1 records validated",
        "record_count": 1,
    }
    assert RuntimeSecretStore(database_url, "").health()["status"] == "unavailable"
    assert (
        RuntimeSecretStore(database_url, Fernet.generate_key().decode("ascii")).health()["status"]
        == "unavailable"
    )


def test_settings_api_validates_saves_and_enables_bootstrap(
    tmp_path: Path,
    monkeypatch,
    database_url: str,
) -> None:
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("PLATFORM_SECRET_KEY", key)
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr(
        "quant_platform.api.requests.post",
        lambda *args, **kwargs: _ValidTushareResponse(),
    )
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/api/settings/tushare",
        json={"api_url": "https://api.tushare.pro", "token": "server-token-value"},
    )
    assert response.status_code == 200
    assert "server-token-value" not in response.text
    status = client.get("/api/settings").json()
    assert status["tushare"]["configured"] is True
    assert status["tushare"]["source"] == "database"

    alert_url = "https://alerts.example.internal/hooks/secret-route"
    alert_response = client.post(
        "/api/settings/alerts",
        json={"webhook_url": alert_url},
    )
    assert alert_response.status_code == 200
    assert alert_url not in alert_response.text
    alert_status = client.get("/api/settings").json()["alerts"]
    assert alert_status == {
        "configured": True,
        "source": "database",
        "endpoint_host": "alerts.example.internal",
        "updated_at": alert_status["updated_at"],
    }
    with open_database(database_url).connect() as connection:
        alert_ciphertext = connection.scalar(
            select(runtime_secrets.c.ciphertext).where(runtime_secrets.c.name == "alert_webhook")
        )
    assert alert_ciphertext and alert_url not in alert_ciphertext
    assert (
        client.post(
            "/api/settings/alerts",
            json={"webhook_url": "http://remote.example/hook"},
        ).status_code
        == 422
    )

    bootstrap = client.post(
        "/api/jobs/bootstrap",
        json={
            "profile": "full",
            "start": "2024-01-01",
            "end": "latest",
            "build_qlib": True,
        },
    )
    assert bootstrap.status_code == 202
    assert bootstrap.json()["payload"]["build_qlib"] is False
    assert bootstrap.json()["payload"]["finalize_after_download"] is True
    assert bootstrap.json()["payload"]["end"] == bootstrap.json()["payload"]["snapshot_end"]
    assert bootstrap.json()["payload"]["end"] != "latest"


def test_worker_injects_latest_tushare_secret(database_url: str, tmp_path: Path) -> None:
    key = Fernet.generate_key().decode("ascii")
    settings = Settings(
        api_url="",
        token="",
        data_root=tmp_path / "data",
        database_url=database_url,
        platform_secret_key=key,
    )
    RuntimeSecretStore(database_url, key).put(
        "tushare",
        {"api_url": "https://api.tushare.pro", "token": "latest-token"},
        metadata={"api_url": "https://api.tushare.pro"},
        updated_by=None,
    )
    worker = LocalJobWorker(JobStore(database_url), tmp_path, settings)
    command, _result, env = worker._command(
        {
            "kind": "bootstrap",
            "payload": {
                "profile": "full",
                "start": "2024-01-01",
                "end": "latest",
                "snapshot_end": "2024-02-02",
                "build_qlib": True,
            },
        }
    )
    assert env == {
        "TUSHARE_API_URL": "https://api.tushare.pro",
        "TUSHARE_TOKEN": "latest-token",
    }
    assert "--download-only" in command
    assert "--build-qlib" not in command
    assert command[command.index("--end") + 1] == "2024-02-02"


def test_scheduler_hot_loads_latest_encrypted_alert_webhook(
    database_url: str,
    tmp_path: Path,
) -> None:
    key = Fernet.generate_key().decode("ascii")
    settings = Settings(
        api_url="",
        token="",
        data_root=tmp_path / "data",
        database_url=database_url,
        platform_secret_key=key,
        alert_webhook_url="https://environment.example/hook",
    )
    secrets = RuntimeSecretStore(database_url, key)
    secrets.put(
        "alert_webhook",
        {"webhook_url": "https://first.example/hook"},
        metadata={"enabled": True, "endpoint_host": "first.example"},
        updated_by=None,
    )
    scheduler = SchedulerEngine(settings)
    delivered_to: list[str] = []
    scheduler.schedules.materialize_due = lambda _now: 0  # type: ignore[method-assign]
    scheduler.schedules.claim_run = lambda **_kwargs: None  # type: ignore[method-assign]
    scheduler.project_alerts = lambda: 0  # type: ignore[method-assign]
    scheduler.health.due = lambda _now: False  # type: ignore[method-assign]
    scheduler.alerts.deliver_pending = (  # type: ignore[method-assign]
        lambda url: delivered_to.append(url) or 0
    )

    scheduler.tick()
    secrets.put(
        "alert_webhook",
        {"webhook_url": "https://second.example/hook"},
        metadata={"enabled": True, "endpoint_host": "second.example"},
        updated_by=None,
    )
    scheduler.tick()
    secrets.put(
        "alert_webhook",
        {"webhook_url": ""},
        metadata={"enabled": False, "endpoint_host": ""},
        updated_by=None,
    )
    scheduler.tick()

    assert delivered_to == [
        "https://first.example/hook",
        "https://second.example/hook",
        "",
    ]


def test_scheduler_does_not_fall_back_when_database_alert_secret_is_unreadable(
    database_url: str, tmp_path: Path
) -> None:
    key = Fernet.generate_key().decode("ascii")
    RuntimeSecretStore(database_url, key).put(
        "alert_webhook",
        {"webhook_url": "https://database.example/hook"},
        metadata={"enabled": True, "endpoint_host": "database.example"},
        updated_by=None,
    )
    scheduler = SchedulerEngine(
        Settings(
            api_url="",
            token="",
            data_root=tmp_path / "data",
            database_url=database_url,
            platform_secret_key="",
            alert_webhook_url="https://stale-environment.example/hook",
        )
    )
    assert scheduler._alert_webhook_url() == ""
    health = scheduler.health.collect()
    assert health["components"]["runtime_secret_storage"]["status"] == "unavailable"
    result = scheduler.tick()
    assert result["health_recorded"] == 1
    secret_alerts = [
        item
        for item in scheduler.alerts.list()
        if item["details"].get("component") == "runtime_secret_storage"
    ]
    assert secret_alerts and secret_alerts[0]["severity"] == "critical"


def test_api_health_rejects_an_unreadable_runtime_secret_store(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    key = Fernet.generate_key().decode("ascii")
    RuntimeSecretStore(database_url, key).put(
        "alert_webhook",
        {"webhook_url": "https://database.example/hook"},
        metadata={"enabled": True, "endpoint_host": "database.example"},
        updated_by=None,
    )
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("AUTH_MODE", "disabled")
    monkeypatch.setenv("PLATFORM_SECRET_KEY", Fernet.generate_key().decode("ascii"))
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["detail"]["runtime_secret_storage"] == "unavailable"
