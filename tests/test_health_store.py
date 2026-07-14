from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import update

from quant_data.config import Settings
from quant_data.database import jobs
from quant_platform.api import create_app
from quant_platform.health_store import OperationalHealthStore
from quant_platform.job_store import JobStore
from quant_platform.runtime_secret_store import RuntimeSecretStore
from quant_platform.scheduler import SchedulerEngine


def _dataset(data_root: Path, end_date: str) -> None:
    root = data_root / "qlib" / "health-snapshot"
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
