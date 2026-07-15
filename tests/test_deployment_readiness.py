from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from cryptography.fernet import Fernet

from quant_data.config import Settings
from quant_platform.auth_store import AuthStore
from quant_platform.data_task_store import DataTaskStore
from quant_platform.deployment_readiness import DeploymentReadinessStore
from quant_platform.health_store import OperationalHealthStore
from quant_platform.job_store import JobStore
from quant_platform.runtime_secret_store import RuntimeSecretStore
from quant_platform.schedule_store import ScheduleStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _settings(monkeypatch, database_url: str, data_root: Path, *, auth_mode: str) -> Settings:
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("AUTH_MODE", auth_mode)
    monkeypatch.setenv("TUSHARE_API_URL", "https://api.tushare.pro")
    monkeypatch.setenv("TUSHARE_TOKEN", "verified-test-token")
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setenv("BROKER_MODE", "disabled")
    monkeypatch.setenv("PLATFORM_SECRET_KEY", Fernet.generate_key().decode("ascii"))
    return Settings.from_env(PROJECT_ROOT / ".env.missing")


def _qlib_dataset(data_root: Path) -> None:
    target = data_root / "qlib" / "acceptance-snapshot"
    (target / "calendars").mkdir(parents=True)
    (target / "instruments").mkdir()
    (target / "features").mkdir()
    (target / "metadata").mkdir()
    start = date(2024, 1, 1)
    days = [(start + timedelta(days=index)).isoformat() for index in range(504)]
    (target / "calendars" / "day.txt").write_text("\n".join(days) + "\n", encoding="utf-8")
    (target / "instruments" / "cn_all.txt").write_text(
        f"SH600000\t{days[0]}\t{days[-1]}\n", encoding="utf-8"
    )
    (target / "metadata" / "provenance.json").write_text(
        json.dumps(
            {
                "dataset_identity_sha256": "a" * 64,
                "snapshot_manifest_sha256": "b" * 64,
                "dataset_lineage_id": "c" * 64,
                "lineage_verified": True,
            }
        ),
        encoding="utf-8",
    )


def test_empty_deployment_is_fail_closed(tmp_path: Path, monkeypatch, database_url: str) -> None:
    settings = _settings(monkeypatch, database_url, tmp_path / "data", auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()

    result = DeploymentReadinessStore(settings, PROJECT_ROOT).assess()

    assert result["highest_ready_profile"] is None
    assert result["live_trading_supported"] is False
    research = result["profiles"][0]
    assert research["status"] == "blocked"
    blocked = {item["id"] for item in research["checks"] if item["status"] == "block"}
    assert {
        "authentication_enabled",
        "tushare_verified",
        "initialization_pipeline",
        "reproducible_qlib_dataset",
        "operational_health",
        "rdagent_runtime",
        "incremental_schedule",
    }.issubset(blocked)


def test_readiness_requires_runtime_secrets_to_be_decryptable(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    settings = _settings(monkeypatch, database_url, tmp_path / "data", auth_mode="required")
    RuntimeSecretStore(database_url, settings.platform_secret_key).put(
        "tushare",
        {"api_url": "https://api.tushare.pro", "token": "database-token"},
        metadata={
            "api_url": "https://api.tushare.pro",
            "verified_at": datetime.now(UTC).isoformat(),
        },
        updated_by=None,
    )
    wrong_key = replace(settings, platform_secret_key=Fernet.generate_key().decode("ascii"))
    result = DeploymentReadinessStore(wrong_key, PROJECT_ROOT).assess()
    checks = {item["id"]: item for item in result["profiles"][0]["checks"]}
    assert checks["runtime_secret_storage"]["status"] == "block"
    assert checks["tushare_verified"]["status"] == "block"
    assert "无法解密" in checks["tushare_verified"]["evidence"]


def test_research_readiness_requires_complete_evidence_chain(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    settings = _settings(monkeypatch, database_url, data_root, auth_mode="required")
    AuthStore(database_url).bootstrap_admin(
        username="admin",
        display_name="Administrator",
        password="Secure-Admin-123!",
    )
    jobs = JobStore(database_url)
    for kind in ("bootstrap", "data_verify", "data_snapshot", "data_qlib", "qlib_baseline"):
        job = jobs.create(kind, {"fixture": True}, tmp_path / f"{kind}.log")
        jobs.finish(job["id"], exit_code=0, result={"accepted": True})
    DataTaskStore(database_url).sync_catalog()
    _qlib_dataset(data_root)
    ScheduleStore(database_url).create(
        name="daily data sync",
        kind="incremental_sync",
        timezone="Asia/Shanghai",
        run_time=time(18, 0),
        trading_days_only=True,
        payload={"profile": "full", "lookback_days": 7},
        misfire_grace_seconds=3600,
        actor="admin",
    )
    current = datetime.now(UTC)
    OperationalHealthStore(settings).record(
        {
            "status": "ok",
            "components": {
                "postgresql": {"status": "ok", "message": "ready"},
                "rdagent_runtime": {"status": "ok", "message": "ready"},
            },
            "summary": {
                "component_count": 2,
                "ok_count": 2,
                "problem_count": 0,
                "bootstrap_count": 0,
            },
            "recorded_at": current,
        }
    )

    result = DeploymentReadinessStore(settings, PROJECT_ROOT).assess(now=current)

    research, recommendation, allocation, pair = result["profiles"]
    assert result["highest_ready_profile"] == "recommendation_tracking"
    assert research["status"] == "ready"
    assert research["passed"] == research["total"]
    assert pair["status"] == "blocked"
    assert (
        next(item for item in pair["checks"] if item["id"] == "pair_minute_data")["status"]
        == "block"
    )
    assert recommendation["status"] == "ready"
    assert (
        next(
            item
            for item in recommendation["checks"]
            if item["id"] == "unsupported_schedules_retired"
        )["status"]
        == "pass"
    )
    assert (
        next(item for item in research["checks"] if item["id"] == "schema_current")["status"]
        == "pass"
    )
