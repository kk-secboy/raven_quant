from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from pydantic import ValidationError

from quant_platform.api import AutopilotResearchEventRequest, create_app
from quant_platform.auth_policy import has_permission, permission_for
from quant_platform.auth_store import AuthStore
from quant_platform.autopilot import AutopilotController


def _request() -> dict:
    return {
        "event_key": "full-mainline-20260907",
        "horizon_profile": "short_1_5d",
        "reason": "Run the existing complete research pipeline with a frozen budget.",
        "quant_loop_n": 10,
        "quant_duration": "1h",
    }


@pytest.mark.no_database
@pytest.mark.parametrize("field", ["actor", "prediction_champion", "tournament_id", "dataset"])
def test_manual_research_cannot_supply_privileged_bindings(field: str) -> None:
    with pytest.raises(ValidationError):
        AutopilotResearchEventRequest.model_validate({**_request(), field: "untrusted"})


@pytest.mark.no_database
def test_manual_research_requires_automation_permission_and_two_arms() -> None:
    permission = permission_for("POST", "/api/autopilot/research-events")
    assert permission == "automation:manage"
    assert has_permission("operator", permission)
    assert not has_permission("viewer", permission)
    with pytest.raises(ValidationError):
        AutopilotResearchEventRequest.model_validate({**_request(), "quant_loop_n": 1})


def test_manual_research_api_registers_and_audits_without_a_parallel_scheduler(
    tmp_path: Path, monkeypatch, database_url: str,
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("PLATFORM_SECRET_KEY", Fernet.generate_key().decode("ascii"))
    calls = []
    cycle = {"id": "a" * 32, "status": "active", "stage": "parallel_research"}

    def register(self, **kwargs):
        calls.append(kwargs)
        return cycle

    def forbidden_tick(self, *args, **kwargs):
        raise AssertionError("HTTP registration must leave execution to the existing scheduler")

    monkeypatch.setattr(AutopilotController, "start_research_event", register)
    monkeypatch.setattr(AutopilotController, "tick", forbidden_tick)
    client = TestClient(create_app(tmp_path))
    response = client.post("/api/autopilot/research-events", json=_request())
    assert response.status_code == 202, response.text
    assert response.json() == cycle
    assert calls == [{**_request(), "actor": "local-admin"}]
    events = AuthStore(database_url).list_audit()
    event = next(item for item in events if item["action"] == "autopilot.research_event_requested")
    assert event["username"] == "local-admin"
    assert event["details"]["cycle_id"] == cycle["id"]
    assert event["details"]["quant_loop_n"] == 10


def test_manual_research_api_reports_registration_conflict(
    tmp_path: Path, monkeypatch, database_url: str,
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("PLATFORM_SECRET_KEY", Fernet.generate_key().decode("ascii"))

    def conflict(self, **kwargs):
        raise ValueError("an active research event already owns this horizon")

    monkeypatch.setattr(AutopilotController, "start_research_event", conflict)
    response = TestClient(create_app(tmp_path)).post(
        "/api/autopilot/research-events", json=_request(),
    )
    assert response.status_code == 409
    assert "already owns" in response.json()["detail"]
