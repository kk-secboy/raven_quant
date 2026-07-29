from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from quant_data.database import open_database, users
from quant_platform.api import create_app
from quant_platform.auth_policy import has_permission, permission_for
from quant_platform.auth_store import AuthenticationError, AuthStore
from scripts.reset_admin_password import main as reset_admin_password

ADMIN_PASSWORD = "Admin-test-only-42!"
VIEWER_PASSWORD = "Viewer-test-only-42!"


def _bootstrap(store: AuthStore) -> dict:
    return store.bootstrap_admin(
        username="admin",
        display_name="Test Administrator",
        password=ADMIN_PASSWORD,
    )


def test_auth_store_hashes_passwords_and_revokes_sessions(database_url: str) -> None:
    store = AuthStore(database_url)
    admin = _bootstrap(store)

    with open_database(database_url).connect() as connection:
        password_hash = connection.scalar(
            select(users.c.password_hash).where(users.c.id == admin["id"])
        )
    assert password_hash != ADMIN_PASSWORD
    assert str(password_hash).startswith("$argon2id$")

    user, token, expires_at = store.login(
        username="admin",
        password=ADMIN_PASSWORD,
        session_hours=12,
        ip_hash=store.hash_ip("127.0.0.1"),
        user_agent="pytest",
    )
    assert user["id"] == admin["id"]
    assert token not in json.dumps(user)
    session = store.validate_session(token)
    assert session and session["session_id"]
    assert store.validate_session(token, now=expires_at) is None

    store.logout(token)
    assert store.validate_session(token) is None


def test_login_failures_persist_and_lock_account(database_url: str) -> None:
    store = AuthStore(database_url)
    _bootstrap(store)
    started = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)

    for attempt in range(5):
        with pytest.raises(AuthenticationError, match="invalid username or password"):
            store.login(
                username="admin",
                password="wrong-password-42!",
                session_hours=12,
                ip_hash=None,
                user_agent="pytest",
                now=started + timedelta(seconds=attempt),
            )

    locked = store.list_users()[0]
    assert locked["failed_login_attempts"] == 5
    assert locked["locked_until"] is not None
    with pytest.raises(AuthenticationError, match="temporarily locked"):
        store.login(
            username="admin",
            password=ADMIN_PASSWORD,
            session_hours=12,
            ip_hash=None,
            user_agent="pytest",
            now=started + timedelta(minutes=1),
        )

    user, _token, _expires = store.login(
        username="admin",
        password=ADMIN_PASSWORD,
        session_hours=12,
        ip_hash=None,
        user_agent="pytest",
        now=started + timedelta(minutes=16),
    )
    assert user["failed_login_attempts"] == 0
    assert user["locked_until"] is None


def test_last_active_administrator_cannot_be_disabled(database_url: str) -> None:
    store = AuthStore(database_url)
    admin = _bootstrap(store)
    with pytest.raises(ValueError, match="last active administrator"):
        store.set_active(admin["id"], False)


def test_console_password_recovery_revokes_sessions_and_is_audited(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AuthStore(database_url)
    _bootstrap(store)
    _user, old_token, _expires = store.login(
        username="admin",
        password=ADMIN_PASSWORD,
        session_hours=12,
        ip_hash=None,
        user_agent="pytest",
    )
    replacement = "Recovered-admin-test-84!"
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setattr(sys, "stdin", StringIO(replacement + "\n"))
    monkeypatch.setattr(sys, "argv", ["reset_admin_password.py", "--username", "admin"])

    reset_admin_password()

    assert store.validate_session(old_token) is None
    with pytest.raises(AuthenticationError, match="invalid username or password"):
        store.login(
            username="admin",
            password=ADMIN_PASSWORD,
            session_hours=12,
            ip_hash=None,
            user_agent="pytest",
        )
    user, _token, _expires = store.login(
        username="admin",
        password=replacement,
        session_hours=12,
        ip_hash=None,
        user_agent="pytest",
    )
    assert user["failed_login_attempts"] == 0
    assert any(item["action"] == "auth.password_recovered" for item in store.list_audit())


@pytest.mark.no_database
def test_role_permission_matrix_is_closed_by_default() -> None:
    assert has_permission("admin", permission_for("POST", "/api/auth/users"))
    assert has_permission("researcher", permission_for("POST", "/api/rdagent/runs"))
    assert has_permission(
        "researcher", permission_for("POST", "/api/research-programs")
    )
    assert has_permission(
        "researcher", permission_for("POST", "/api/strategies/strategy-id/versions")
    )
    assert has_permission(
        "researcher",
        permission_for(
            "POST", "/api/strategy-versions/version-id/parameter-experiments"
        ),
    )
    assert not has_permission(
        "researcher", permission_for("POST", "/api/strategy-versions/version-id/approve")
    )
    assert not has_permission("researcher", permission_for("POST", "/api/factors/id/promote"))
    assert has_permission("operator", permission_for("POST", "/api/schedules"))
    assert has_permission(
        "operator", permission_for("POST", "/api/recommendation-portfolios")
    )
    assert has_permission(
        "operator", permission_for("PUT", "/api/recommendation-accounts/active")
    )
    nav_review_permission = permission_for(
        "POST",
        "/api/simulation-portfolios/account-id/nav/2026-07-13/review",
    )
    assert has_permission("admin", nav_review_permission)
    assert not has_permission("operator", nav_review_permission)
    assert has_permission("researcher", permission_for("POST", "/api/strategy-allocations"))
    assert not has_permission(
        "researcher",
        permission_for("POST", "/api/strategy-allocations/allocation-id/approve"),
    )
    assert has_permission(
        "operator",
        permission_for("POST", "/api/strategy-allocations/allocation-id/refresh"),
    )
    assert has_permission(
        "operator",
        permission_for("POST", "/api/strategy-allocations/allocation-id/status"),
    )
    assert has_permission(
        "operator",
        permission_for("POST", "/api/strategy-allocations/allocation-id/schedule"),
    )
    assert has_permission(
        "operator",
        permission_for("DELETE", "/api/strategy-allocations/allocation-id/schedule"),
    )
    assert not has_permission(
        "researcher",
        permission_for("POST", "/api/strategy-allocations/allocation-id/schedule/status"),
    )
    assert has_permission(
        "operator",
        permission_for(
            "POST",
            "/api/strategy-allocations/allocation-id/events/1/resolve",
        ),
    )
    assert not has_permission("viewer", permission_for("POST", "/api/schedules"))
    assert has_permission("viewer", permission_for("GET", "/api/overview"))
    assert has_permission(
        "viewer", permission_for("GET", "/api/settings/strategy-defaults")
    )
    assert not has_permission(
        "researcher", permission_for("PUT", "/api/settings/strategy-defaults")
    )
    assert permission_for("POST", "/api/unknown-write") == "admin:write"


def test_required_auth_bootstrap_rbac_origin_actor_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    database_url: str,
) -> None:
    monkeypatch.setenv("AUTH_MODE", "required")
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    app = create_app(tmp_path)

    with TestClient(app) as admin_client:
        assert admin_client.get("/api/auth/state").json()["status"] == "bootstrap_required"
        denied = admin_client.get("/api/overview")
        assert denied.status_code == 401
        assert denied.json()["detail"] == "bootstrap_required"

        bootstrapped = admin_client.post(
            "/api/auth/bootstrap",
            json={
                "username": "admin",
                "display_name": "Test Administrator",
                "password": ADMIN_PASSWORD,
            },
        )
        assert bootstrapped.status_code == 201
        cookie = bootstrapped.headers["set-cookie"].lower()
        assert "httponly" in cookie
        assert "samesite=strict" in cookie
        assert bootstrapped.headers["cache-control"] == "no-store"
        assert admin_client.get("/api/auth/state").json()["status"] == "authenticated"

        viewer = admin_client.post(
            "/api/auth/users",
            json={
                "username": "viewer",
                "display_name": "Read Only",
                "role": "viewer",
                "password": VIEWER_PASSWORD,
            },
        )
        assert viewer.status_code == 201

        schedule = admin_client.post(
            "/api/schedules",
            json={
                "name": "authenticated schedule",
                "kind": "incremental_sync",
                "timezone": "Asia/Shanghai",
                "run_time": "18:00",
                "trading_days_only": True,
                "payload": {"profile": "core", "lookback_days": 7, "build_qlib": True},
                "misfire_grace_seconds": 3600,
                "actor": "spoofed-client-actor",
            },
        )
        assert schedule.status_code == 201
        assert schedule.json()["created_by"] == "admin"

        rejected_origin = admin_client.post(
            f"/api/schedules/{schedule.json()['id']}/status",
            headers={"Origin": "https://malicious.example"},
            json={"status": "paused"},
        )
        assert rejected_origin.status_code == 403

        audit = admin_client.get("/api/audit").json()
        assert any(item["action"] == "auth.bootstrap_succeeded" for item in audit)
        assert any(item["action"] == "request.origin_rejected" for item in audit)
        serialized_audit = json.dumps(audit)
        assert ADMIN_PASSWORD not in serialized_audit
        assert VIEWER_PASSWORD not in serialized_audit
        assert "quantlab_session" not in serialized_audit

    with TestClient(app) as viewer_client:
        login = viewer_client.post(
            "/api/auth/login",
            json={"username": "viewer", "password": VIEWER_PASSWORD},
        )
        assert login.status_code == 200
        assert viewer_client.get("/api/overview").status_code == 200
        denied_write = viewer_client.post(
            "/api/schedules",
            json={
                "name": "viewer must not create",
                "kind": "incremental_sync",
                "timezone": "Asia/Shanghai",
                "run_time": "18:30",
                "trading_days_only": True,
                "payload": {"profile": "core", "lookback_days": 7, "build_qlib": True},
                "misfire_grace_seconds": 3600,
            },
        )
        assert denied_write.status_code == 403
