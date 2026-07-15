from __future__ import annotations

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"*"},
    "researcher": {"read", "research:write", "strategy:write"},
    "operator": {
        "read",
        "data:write",
        "portfolio:write",
        "automation:manage",
        "alerts:manage",
    },
    "viewer": {"read"},
}


def permission_for(method: str, path: str) -> str:
    if path == "/api/settings/strategy-defaults" and method == "GET":
        return "read"
    if path.startswith("/api/settings"):
        return "settings:manage"
    if method == "GET":
        if path.startswith("/api/broker"):
            return "broker:manage"
        if path.startswith("/api/auth/users") or path.startswith("/api/audit"):
            return "users:manage"
        return "read"
    if path in {"/api/auth/logout", "/api/auth/password"}:
        return "read"
    if path.startswith("/api/auth/users"):
        return "users:manage"
    if path in {
        "/api/jobs/bootstrap",
        "/api/jobs/finalize-data",
        "/api/jobs/margin-eligibility",
        "/api/jobs/core-intraday",
        "/api/jobs/minute-qlib",
        "/api/jobs/supplemental-download",
    }:
        return "data:write"
    if path.startswith("/api/jobs/") and (
        path.endswith("/retry") or path.endswith("/cancel")
    ):
        return "data:write"
    if path in {"/api/jobs/qlib-baseline", "/api/jobs/minute-research"} or path.startswith(
        "/api/rdagent/"
    ):
        return "research:write"
    if path.startswith("/api/research-campaigns") or path.startswith(
        "/api/research-programs"
    ):
        return "research:write"
    if path.startswith("/api/factors/") and path.endswith("/promote"):
        return "factor:approve"
    if path.startswith("/api/factors/"):
        return "research:write"
    if (
        path == "/api/strategies"
        or path == "/api/pair-strategies"
        or path.startswith("/api/pair-strategies/")
        or (path.startswith("/api/strategies/") and path.endswith("/versions"))
        or path.endswith("/backtests")
        or path.endswith("/pair-backtests")
        or path.endswith("/parameter-experiments")
    ):
        return "strategy:write"
    if path.startswith("/api/strategy-versions/") and path.endswith("/approve"):
        return "strategy:approve"
    if path.startswith("/api/strategy-allocations/") and path.endswith("/approve"):
        return "strategy:approve"
    if path.startswith("/api/strategy-allocations/") and path.endswith("/refresh"):
        return "portfolio:write"
    if path.startswith("/api/strategy-allocations/") and (
        path.endswith("/schedule") or path.endswith("/schedule/status")
    ):
        return "automation:manage"
    if path.startswith("/api/strategy-allocations/") and (
        path.endswith("/status") or path.endswith("/acknowledge") or path.endswith("/resolve")
    ):
        return "portfolio:write"
    if path.startswith("/api/strategy-allocations"):
        return "strategy:write"
    if path.startswith("/api/portfolios") or path.startswith("/api/pair-portfolios"):
        return "portfolio:write"
    if path.startswith("/api/schedules"):
        return "automation:manage"
    if path.startswith("/api/alerts"):
        return "alerts:manage"
    if path.startswith("/api/broker"):
        return "broker:manage"
    return "admin:write"


def has_permission(role: str, permission: str) -> bool:
    permissions = ROLE_PERMISSIONS.get(role, set())
    return "*" in permissions or permission in permissions
