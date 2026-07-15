from __future__ import annotations

import ast
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .backup_restore import ComposeContext

EXPECTED_SERVICES = {
    "postgres",
    "api",
    "scheduler",
    "worker",
    "rdagent-docker",
    "rdagent-worker",
    "web",
    "gateway",
}
HEALTHCHECK_SERVICES = EXPECTED_SERVICES - {"web", "gateway"}


def _check(
    check_id: str,
    title: str,
    passed: bool,
    evidence: str,
    remediation: str,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "title": title,
        "status": "pass" if passed else "block",
        "evidence": evidence,
        "remediation": None if passed else remediation,
    }


def _schema_compatibility(
    project_root: Path,
    database_revision: str | None,
) -> tuple[str | None, str, bool]:
    try:
        revisions: dict[str, tuple[str, ...]] = {}
        for path in sorted((project_root / "migrations" / "versions").glob("*.py")):
            values: dict[str, Any] = {}
            for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                    value = node.value
                elif isinstance(node, ast.AnnAssign):
                    target = node.target
                    value = node.value
                else:
                    continue
                if (
                    isinstance(target, ast.Name)
                    and target.id in {"revision", "down_revision"}
                    and value is not None
                ):
                    values[target.id] = ast.literal_eval(value)
            revision = values.get("revision")
            down_revision = values.get("down_revision")
            if not isinstance(revision, str) or revision in revisions:
                raise ValueError(f"invalid or duplicate revision in {path.name}")
            if down_revision is None:
                parents: tuple[str, ...] = ()
            elif isinstance(down_revision, str):
                parents = (down_revision,)
            elif isinstance(down_revision, (tuple, list)) and all(
                isinstance(item, str) for item in down_revision
            ):
                parents = tuple(down_revision)
            else:
                raise ValueError(f"invalid down_revision in {path.name}")
            revisions[revision] = parents
        referenced = {parent for parents in revisions.values() for parent in parents}
        if not referenced.issubset(revisions):
            raise ValueError("migration graph references a missing revision")
        heads = set(revisions) - referenced
    except Exception:  # pragma: no cover - diagnostics must fail closed
        return None, "unknown", False
    if len(heads) != 1:
        return None, "multiple_code_heads", False
    code_revision = next(iter(heads))
    if not database_revision:
        return code_revision, "unknown_database_revision", False
    if database_revision not in revisions:
        return code_revision, "unknown_database_revision", False
    upgrade_chain: set[str] = set()
    pending = [code_revision]
    while pending:
        revision = pending.pop()
        if revision in upgrade_chain:
            continue
        upgrade_chain.add(revision)
        pending.extend(revisions[revision])
    if database_revision == code_revision:
        return code_revision, "current", True
    if database_revision in upgrade_chain:
        return code_revision, "upgrade_required", True
    return code_revision, "incompatible", False


def _compose_services(raw: str) -> dict[str, dict[str, Any]]:
    stripped = raw.strip()
    if not stripped:
        return {}
    try:
        value = json.loads(stripped)
        rows = value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in stripped.splitlines() if line.strip()]
    return {str(row.get("Service")): row for row in rows}


def _docker_project_services(
    context: ComposeContext,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    raw_ids = context.docker(
        "ps",
        "-aq",
        "--filter",
        f"label=com.docker.compose.project={context.project_name}",
        capture=True,
    )
    container_ids = [item.strip() for item in raw_ids.splitlines() if item.strip()]
    if not container_ids:
        return {}, {}
    inspections = json.loads(context.docker("inspect", *container_ids, capture=True))
    services: dict[str, dict[str, Any]] = {}
    ids_by_service: dict[str, str] = {}
    for item in inspections:
        labels = item.get("Config", {}).get("Labels", {}) or {}
        service = str(labels.get("com.docker.compose.service", "")).strip()
        container_id = str(item.get("Id", "")).strip()
        if not service or not container_id:
            continue
        state = item.get("State", {}) or {}
        health = state.get("Health", {}) or {}
        services[service] = {
            "Service": service,
            "State": state.get("Status", "unknown"),
            "Health": health.get("Status", ""),
        }
        ids_by_service[service] = container_id
    return services, ids_by_service


def assess_release(
    context: ComposeContext,
    project_root: Path,
    *,
    minimum_free_gb: float = 20.0,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        context.run("config", "--quiet", capture=True)
        compose_valid = True
        compose_evidence = "Compose configuration resolved successfully"
    except Exception as exc:
        compose_valid = False
        compose_evidence = str(exc)[:500]
    checks.append(
        _check(
            "compose_config",
            "Compose configuration",
            compose_valid,
            compose_evidence,
            "Fix environment variables, images, or Compose configuration.",
        )
    )

    fallback_services: dict[str, dict[str, Any]] = {}
    fallback_ids: dict[str, str] = {}
    fallback_error = ""
    if not compose_valid:
        try:
            fallback_services, fallback_ids = _docker_project_services(context)
        except Exception as exc:
            fallback_error = str(exc)[:500]

    postgres_error = ""
    if compose_valid:
        try:
            postgres_running = bool(context.container_id("postgres"))
        except Exception as exc:
            postgres_running = False
            postgres_error = str(exc)[:500]
    else:
        postgres_running = bool(fallback_ids.get("postgres"))
        postgres_error = (
            fallback_error or "PostgreSQL container was not found through Docker project labels"
        )
    checks.append(
        _check(
            "postgres_running",
            "Control database online",
            postgres_running,
            (
                "PostgreSQL container is running"
                if postgres_running
                else postgres_error or "PostgreSQL container is not running"
            ),
            "Restore the current deployment's PostgreSQL service first.",
        )
    )

    queue = {
        "active_jobs": -1,
        "pending_units": -1,
        "running_units": -1,
        "failed_units": -1,
    }
    database_revision = None
    database_query_ok = False
    if postgres_running:
        try:
            query = (
                "SELECT "
                "(SELECT count(*) FROM quantlab.jobs "
                "WHERE status IN ('queued','running')) || '|' || "
                "(SELECT count(*) FROM quantlab.work_units "
                "WHERE status='pending') || '|' || "
                "(SELECT count(*) FROM quantlab.work_units "
                "WHERE status='running') || '|' || "
                "(SELECT count(*) FROM quantlab.work_units "
                "WHERE status='failed') || '|' || "
                "(SELECT version_num FROM quantlab.alembic_version);"
            )
            database_args = (
                ("exec", "-T", "postgres") if compose_valid else ("exec", fallback_ids["postgres"])
            )
            runner = context.run if compose_valid else context.docker
            raw = runner(
                *database_args,
                "psql",
                "-U",
                "quantlab",
                "-d",
                "quantlab",
                "-Atc",
                query,
                capture=True,
            ).splitlines()[0]
            values = raw.split("|")
            if len(values) != 5:
                raise ValueError("unexpected release preflight database response")
            queue = {
                "active_jobs": int(values[0]),
                "pending_units": int(values[1]),
                "running_units": int(values[2]),
                "failed_units": int(values[3]),
            }
            database_revision = values[4]
            database_query_ok = True
        except Exception as exc:
            checks.append(
                _check(
                    "database_query",
                    "Release-state query",
                    False,
                    str(exc)[:500],
                    "Verify migrations and PostgreSQL connectivity before release.",
                )
            )

    # Pending and failed units are durable, dormant checkpoints. They are often
    # exactly what a bug-fix release needs to preserve and resume. Only a live
    # job or a currently running/leased unit can race a container replacement.
    # Treating dormant checkpoints as active work creates an unrecoverable
    # release deadlock after a downloader fails.
    queue_complete = database_query_ok and queue["active_jobs"] == 0 and queue["running_units"] == 0
    checks.append(
        _check(
            "durable_work_idle",
            "No durable work is executing",
            queue_complete,
            (
                f"active jobs {queue['active_jobs']}; pending {queue['pending_units']}; "
                f"running {queue['running_units']}; failed {queue['failed_units']}"
            ),
            "Wait for active jobs and running work units; dormant checkpoints "
            "are preserved across release.",
        )
    )

    code_revision, migration_state, schema_compatible = _schema_compatibility(
        project_root,
        database_revision,
    )
    checks.append(
        _check(
            "schema_compatible",
            "Database has a recognized migration path",
            schema_compatible,
            (
                f"database {database_revision or 'unknown'}; "
                f"code {code_revision or 'unknown'}; state {migration_state}"
            ),
            "Confirm the upgrade path before deploying an unknown schema transition.",
        )
    )

    try:
        services = (
            _compose_services(context.run("ps", "--format", "json", capture=True))
            if compose_valid
            else fallback_services
        )
    except Exception as exc:
        services = {}
        service_error = str(exc)[:500]
    else:
        service_error = fallback_error if not compose_valid else ""
    missing = EXPECTED_SERVICES - set(services)
    stopped = sorted(
        name
        for name in EXPECTED_SERVICES & set(services)
        if services[name].get("State") != "running"
    )
    unhealthy = sorted(
        name
        for name in HEALTHCHECK_SERVICES & set(services)
        if services[name].get("Health") != "healthy"
    )
    services_ready = not missing and not stopped and not unhealthy
    checks.append(
        _check(
            "services_healthy",
            "Current eight-service baseline is healthy",
            services_ready,
            (
                f"missing {sorted(missing)}; stopped {stopped}; unhealthy {unhealthy}"
                if not service_error
                else service_error
            ),
            "Restore every current service before introducing a new release.",
        )
    )

    free_bytes = shutil.disk_usage(project_root.resolve()).free
    required_bytes = int(minimum_free_gb * 1024**3)
    checks.append(
        _check(
            "disk_capacity",
            "Release and rollback disk headroom",
            free_bytes >= required_bytes,
            f"free {free_bytes / 1024**3:.1f} GiB; required {minimum_free_gb:.1f} GiB",
            "Free build cache or expand storage while retaining rollback images and backups.",
        )
    )

    blockers = [item for item in checks if item["status"] == "block"]
    return {
        "status": "ready" if not blockers else "blocked",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "project_name": context.project_name,
        "blocker_count": len(blockers),
        "queue": queue,
        "database_revision": database_revision,
        "code_revision": code_revision,
        "migration_state": migration_state,
        "checks": checks,
    }
