from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_platform import release_preflight

pytestmark = pytest.mark.no_database


class FakeComposeContext:
    project_name = "quantlab-test"

    def __init__(self, database_row: str) -> None:
        self.database_row = database_row

    def container_id(self, service: str) -> str:
        return "postgres-id" if service == "postgres" else ""

    def run(self, *args: str, capture: bool = False) -> str:
        assert capture is True
        if args == ("config", "--quiet"):
            return ""
        if args[:3] == ("exec", "-T", "postgres"):
            return self.database_row
        if args == ("ps", "--format", "json"):
            rows = []
            for service in sorted(release_preflight.EXPECTED_SERVICES):
                row = {"Service": service, "State": "running"}
                if service in release_preflight.HEALTHCHECK_SERVICES:
                    row["Health"] = "healthy"
                rows.append(row)
            return json.dumps(rows)
        raise AssertionError(f"unexpected compose call: {args}")


class InvalidComposeContext:
    project_name = "quantlab-invalid"

    def run(self, *args: str, capture: bool = False) -> str:
        assert capture is True
        if args == ("config", "--quiet"):
            raise RuntimeError("PLATFORM_SECRET_KEY is required")
        raise AssertionError(f"invalid Compose must not be queried further: {args}")

    def container_id(self, service: str) -> str:
        raise AssertionError(f"invalid Compose must not inspect {service}")

    def docker(self, *args: str, capture: bool = False) -> str:
        assert capture is True
        if args[:3] == ("ps", "-aq", "--filter"):
            return ""
        raise AssertionError(f"unexpected Docker fallback call: {args}")


class InvalidComposeWithRunningProject(InvalidComposeContext):
    def docker(self, *args: str, capture: bool = False) -> str:
        assert capture is True
        if args[:3] == ("ps", "-aq", "--filter"):
            return "container-postgres\ncontainer-api"
        if args[:2] == ("inspect", "container-postgres"):
            rows = []
            for service in sorted(release_preflight.EXPECTED_SERVICES):
                state = {"Status": "running"}
                if service in release_preflight.HEALTHCHECK_SERVICES:
                    state["Health"] = {"Status": "healthy"}
                rows.append(
                    {
                        "Id": f"container-{service}",
                        "Config": {
                            "Labels": {"com.docker.compose.service": service},
                        },
                        "State": state,
                    }
                )
            return json.dumps(rows)
        if args[:2] == ("exec", "container-postgres"):
            return "0|0|0|0|0017_qmt_execution_state"
        raise AssertionError(f"unexpected Docker fallback call: {args}")


def test_assess_release_ready_when_deployment_is_idle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        release_preflight,
        "_schema_compatibility",
        lambda _root, _revision: ("0020_strategy_allocations", "current", True),
    )
    result = release_preflight.assess_release(
        FakeComposeContext("0|0|0|0|0020_strategy_allocations"),  # type: ignore[arg-type]
        tmp_path,
        minimum_free_gb=0,
    )

    assert result["status"] == "ready"
    assert result["blocker_count"] == 0
    assert all(check["status"] == "pass" for check in result["checks"])


def test_assess_release_blocks_active_durable_work(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        release_preflight,
        "_schema_compatibility",
        lambda _root, _revision: ("0020_strategy_allocations", "current", True),
    )
    result = release_preflight.assess_release(
        FakeComposeContext("1|14472|4|0|0020_strategy_allocations"),  # type: ignore[arg-type]
        tmp_path,
        minimum_free_gb=0,
    )

    assert result["status"] == "blocked"
    assert result["blocker_count"] == 1
    durable = next(check for check in result["checks"] if check["id"] == "durable_work_idle")
    assert durable["status"] == "block"
    assert "pending 14472" in durable["evidence"]


def test_assess_release_allows_dormant_resumable_checkpoints(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        release_preflight,
        "_schema_compatibility",
        lambda _root, _revision: ("0020_strategy_allocations", "current", True),
    )
    result = release_preflight.assess_release(
        FakeComposeContext("0|16668|0|24|0020_strategy_allocations"),  # type: ignore[arg-type]
        tmp_path,
        minimum_free_gb=0,
    )

    assert result["status"] == "ready"
    durable = next(check for check in result["checks"] if check["id"] == "durable_work_idle")
    assert durable["status"] == "pass"
    assert "pending 16668" in durable["evidence"]
    assert "failed 24" in durable["evidence"]


def test_assess_release_reports_invalid_compose_without_crashing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        release_preflight,
        "_schema_compatibility",
        lambda _root, _revision: ("0027_web_config_templates", "unknown", False),
    )

    result = release_preflight.assess_release(
        InvalidComposeContext(),  # type: ignore[arg-type]
        tmp_path,
        minimum_free_gb=0,
    )

    assert result["status"] == "blocked"
    compose = next(check for check in result["checks"] if check["id"] == "compose_config")
    database = next(check for check in result["checks"] if check["id"] == "postgres_running")
    services = next(check for check in result["checks"] if check["id"] == "services_healthy")
    assert compose["status"] == "block"
    assert compose["evidence"] == "PLATFORM_SECRET_KEY is required"
    assert "not found through Docker project labels" in database["evidence"]
    assert "missing" in services["evidence"]


def test_assess_release_inspects_running_project_without_valid_compose(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        release_preflight,
        "_schema_compatibility",
        lambda _root, revision: (
            "0027_web_config_templates",
            "upgrade_required",
            revision == "0017_qmt_execution_state",
        ),
    )

    result = release_preflight.assess_release(
        InvalidComposeWithRunningProject(),  # type: ignore[arg-type]
        tmp_path,
        minimum_free_gb=0,
    )

    assert result["blocker_count"] == 1
    assert result["database_revision"] == "0017_qmt_execution_state"
    assert result["migration_state"] == "upgrade_required"
    assert next(
        check for check in result["checks"] if check["id"] == "postgres_running"
    )["status"] == "pass"
    assert next(
        check for check in result["checks"] if check["id"] == "services_healthy"
    )["status"] == "pass"


def test_compose_services_accepts_json_lines() -> None:
    raw = "\n".join(
        [
            json.dumps({"Service": "api", "State": "running"}),
            json.dumps({"Service": "web", "State": "running"}),
        ]
    )

    services = release_preflight._compose_services(raw)

    assert set(services) == {"api", "web"}


def test_known_older_database_revision_has_upgrade_path() -> None:
    project_root = Path(__file__).resolve().parents[1]

    code_revision, state, compatible = release_preflight._schema_compatibility(
        project_root,
        "0017_qmt_execution_state",
    )

    assert code_revision == "0037_single_mainline_contract"
    assert state == "upgrade_required"
    assert compatible is True


def test_unknown_database_revision_fails_closed() -> None:
    project_root = Path(__file__).resolve().parents[1]

    code_revision, state, compatible = release_preflight._schema_compatibility(
        project_root,
        "9999_unknown_revision",
    )

    assert code_revision == "0037_single_mainline_contract"
    assert state == "unknown_database_revision"
    assert compatible is False
