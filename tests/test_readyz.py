from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from quant_platform.api import _scheduler_endpoint_health_check, create_app
from quant_platform.deployment_readiness import DeploymentReadinessStore
from quant_platform.health_store import OperationalHealthStore
from quant_platform.runtime_secret_store import RuntimeSecretStore
from quant_platform.safe_mode import SafeModeStore


@pytest.mark.no_database
def test_readyz_is_public_and_fails_closed_when_safe_mode_engages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"active": False}
    snapshot = {
        "status": "ok",
        "recorded_at": datetime.now(UTC).isoformat(),
        "age_seconds": 1.0,
        "components": {
            "postgresql": {"status": "ok"},
            "scheduler": {
                "status": "ok",
                "release_id": "release-test-1",
                "config_digest": "config-test-1",
            },
            "safe_mode": {"status": "ok"},
            "broker_boundary": {"status": "not_applicable"},
        },
        "summary": {"problem_count": 0},
    }
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://quantlab:quantlab@127.0.0.1:1/not-used",
    )
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("AUTH_MODE", "required")
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setenv("RDAGENT_ENABLED", "false")
    monkeypatch.setenv("PLATFORM_SECRET_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("QUANTLAB_RELEASE_ID", "release-test-1")
    monkeypatch.setenv("QUANTLAB_CONFIG_DIGEST", "config-test-1")
    monkeypatch.delenv("SCHEDULER_URL", raising=False)
    monkeypatch.setattr(OperationalHealthStore, "latest", lambda _self: snapshot)
    monkeypatch.setattr(SafeModeStore, "status", lambda _self: dict(state))
    monkeypatch.setattr(
        RuntimeSecretStore,
        "health",
        lambda _self: {"status": "ok", "message": "available", "record_count": 0},
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

    client = TestClient(create_app(tmp_path))
    try:
        ready = client.get("/api/readyz")
        state["active"] = True
        blocked = client.get("/api/readyz")
        state["active"] = False
        snapshot["components"]["scheduler"]["release_id"] = "mixed-release"
        mixed_release = client.get("/api/readyz")
    finally:
        client.close()

    assert ready.status_code == 200
    assert ready.headers["cache-control"] == "no-store"
    assert ready.json()["ready"] is True
    assert blocked.status_code == 503
    assert blocked.json()["ready"] is False
    assert blocked.json()["checks"]["safe_mode"]["status"] == "blocked"
    assert {item["check"] for item in blocked.json()["blockers"]} == {"safe_mode"}
    assert mixed_release.status_code == 503
    assert mixed_release.json()["checks"]["release_identity"]["status"] == "blocked"


@pytest.mark.no_database
def test_readyz_rejects_an_overdue_active_scheduler_tick() -> None:
    now = datetime(2026, 8, 30, 2, 0, tzinfo=UTC)
    body = {
        "status": "ok",
        "ready": True,
        "last_tick": (now - timedelta(seconds=10)).isoformat(),
        "tick_in_progress": True,
        "tick_started_at": (now - timedelta(seconds=301)).isoformat(),
        "freshness_source": "active_tick",
        "max_active_tick_seconds": 300,
        "release_id": "release-test-1",
        "config_digest": "config-test-1",
    }

    scheduler = _scheduler_endpoint_health_check(
        body,
        response_status_code=200,
        now=now,
        stale_after_seconds=30,
        max_active_tick_seconds=300,
    )

    assert scheduler["status"] == "degraded"
    assert scheduler["active_tick_age_seconds"] == 301
    assert scheduler["max_active_tick_seconds"] == 300
