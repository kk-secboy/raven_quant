from __future__ import annotations

import ast
import json
import os
import re
import shutil
from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from dotenv import dotenv_values

from .backup_restore import ComposeContext
from .deployment_services import (
    CORE_RUNTIME_SERVICES,
    NON_HEALTHCHECK_SERVICES,
    OPTIONAL_PROFILE_SERVICES,
)
from .release_identity import (
    RELEASE_IDENTITY_ENV_KEYS,
    STATEFUL_RELEASE_IDENTITY_EXEMPT,
    normalized_release_identity,
    release_identity_labels,
)

# Compatibility names remain public for release tooling and tests, but both now
# describe the complete long-running topology.  The former eight-service list
# caused mixed releases by silently ignoring five active worker aliases.
LEGACY_EXPECTED_SERVICES = set(CORE_RUNTIME_SERVICES)
EXPECTED_SERVICES = set(CORE_RUNTIME_SERVICES)
HEALTHCHECK_SERVICES = EXPECTED_SERVICES - NON_HEALTHCHECK_SERVICES

_RUNTIME_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_IMMUTABLE_IMAGE_REFERENCE = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
_CORE_IMAGE_SETTINGS = {
    "RDAGENT_RUNTIME_IMAGE_DIGEST": _RUNTIME_IMAGE_DIGEST,
    "RDAGENT_QLIB_SANDBOX_IMAGE": _IMMUTABLE_IMAGE_REFERENCE,
    "RDAGENT_DATA_SCIENCE_IMAGE": _IMMUTABLE_IMAGE_REFERENCE,
    "MODEL_SANDBOX_IMAGE": _IMMUTABLE_IMAGE_REFERENCE,
}
_GPU_IMAGE_SETTINGS = {
    "RDAGENT_FINETUNE_IMAGE": _IMMUTABLE_IMAGE_REFERENCE,
    "RDAGENT_FINETUNE_BENCHMARK_IMAGE": _IMMUTABLE_IMAGE_REFERENCE,
    "RDAGENT_FINETUNE_GPU_PROBE_IMAGE": _IMMUTABLE_IMAGE_REFERENCE,
}


def expected_services(context: ComposeContext) -> set[str]:
    services = set(EXPECTED_SERVICES)
    for profile in getattr(context, "profiles", ()):
        services.update(OPTIONAL_PROFILE_SERVICES.get(profile, ()))
    return services


def _deployment_environment(context: ComposeContext) -> dict[str, str]:
    configured = dotenv_values(context.env_file)
    names = {
        *_CORE_IMAGE_SETTINGS,
        *_GPU_IMAGE_SETTINGS,
    }
    return {
        name: str(
            (os.environ[name] if name in os.environ else configured.get(name)) or ""
        ).strip()
        for name in names
    }


def _docker_storage_configuration(context: ComposeContext) -> tuple[bool, str]:
    try:
        configured = dotenv_values(context.env_file)
        values = {}
        for name, default in (
            ("QUANTLAB_DATA_HOST_PATH", "/data/quantlab"),
            ("RDAGENT_DOCKER_HOST_PATH", "/data/quantlab-rdagent-docker"),
            ("RDAGENT_REGISTRY_HOST_PATH", "/data/quantlab-rdagent-registry"),
        ):
            raw = os.environ[name] if name in os.environ else configured.get(name)
            values[name] = PurePosixPath(str(raw or default).strip())
    except Exception as exc:
        return False, f"RD-Agent storage configuration could not be read: {type(exc).__name__}"
    if not all(path.is_absolute() for path in values.values()):
        return False, "RD-Agent Docker and registry storage paths must be absolute"
    market_data = values["QUANTLAB_DATA_HOST_PATH"]
    isolated = {
        values["RDAGENT_DOCKER_HOST_PATH"],
        values["RDAGENT_REGISTRY_HOST_PATH"],
    }
    if len(isolated) != 2:
        return False, "RD-Agent Docker and registry storage paths must be distinct"
    for path in isolated:
        try:
            path.relative_to(market_data)
        except ValueError:
            if path.parent == market_data.parent:
                continue
            return False, "RD-Agent image storage must be a sibling of market data"
        return False, "RD-Agent image storage must be outside QUANTLAB_DATA_HOST_PATH"
    return True, "DinD layers and release registry are isolated from governed market data"


def _required_image_settings(context: ComposeContext) -> dict[str, re.Pattern[str]]:
    required = dict(_CORE_IMAGE_SETTINGS)
    if "gpu" in set(getattr(context, "profiles", ())):
        required.update(_GPU_IMAGE_SETTINGS)
    return required


def _immutable_image_configuration(context: ComposeContext) -> tuple[bool, str]:
    try:
        configured = _deployment_environment(context)
    except Exception as exc:
        return False, f"deployment environment could not be read: {type(exc).__name__}"
    required = _required_image_settings(context)
    invalid: list[str] = []
    for name, pattern in required.items():
        value = configured[name].lower()
        if not pattern.fullmatch(value):
            invalid.append(name)
    if invalid:
        return False, "missing or mutable image identity: " + ", ".join(sorted(invalid))
    if (
        configured["RDAGENT_QLIB_SANDBOX_IMAGE"].lower()
        == configured["RDAGENT_DATA_SCIENCE_IMAGE"].lower()
    ):
        return False, "Qlib and Data Science must use separately sealed sandbox images"
    return True, "all required runtime and sandbox images use immutable digests"


def _preloaded_image_availability(context: ComposeContext) -> tuple[bool, str]:
    try:
        configured = _deployment_environment(context)
        expected_runtime = configured["RDAGENT_RUNTIME_IMAGE_DIGEST"].lower()
        runtime_services = ["rdagent-worker", "rdagent-data-science-worker"]
        if "gpu" in set(getattr(context, "profiles", ())):
            runtime_services.append("rdagent-llm-finetune-worker")
        mismatched_runtime: list[str] = []
        for service in runtime_services:
            container_id = context.container_id(service)
            if not container_id:
                mismatched_runtime.append(service)
                continue
            actual = context.docker(
                "inspect",
                "--format",
                "{{.Image}}",
                container_id,
                capture=True,
            ).splitlines()[0].strip().lower()
            if actual != expected_runtime:
                mismatched_runtime.append(service)

        nested_images = tuple(
            dict.fromkeys(
                configured[name]
                for name in (
                    "RDAGENT_QLIB_SANDBOX_IMAGE",
                    "RDAGENT_DATA_SCIENCE_IMAGE",
                    "MODEL_SANDBOX_IMAGE",
                )
            )
        )
        nested_raw = context.run(
            "exec",
            "-T",
            "rdagent-docker",
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            *nested_images,
            capture=True,
        )
        nested_ids = [item.strip().lower() for item in nested_raw.splitlines() if item.strip()]
        nested_ready = len(nested_ids) == len(nested_images) and all(
            _RUNTIME_IMAGE_DIGEST.fullmatch(item) for item in nested_ids
        )

        gpu_ready = True
        if "gpu" in set(getattr(context, "profiles", ())):
            gpu_images = tuple(configured[name] for name in _GPU_IMAGE_SETTINGS)
            gpu_raw = context.docker(
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                *gpu_images,
                capture=True,
            )
            gpu_ids = [item.strip().lower() for item in gpu_raw.splitlines() if item.strip()]
            gpu_ready = len(gpu_ids) == len(gpu_images) and all(
                _RUNTIME_IMAGE_DIGEST.fullmatch(item) for item in gpu_ids
            )
    except Exception as exc:
        return False, f"immutable image availability probe failed: {type(exc).__name__}"
    problems: list[str] = []
    if mismatched_runtime:
        problems.append("runtime identity mismatch: " + ", ".join(mismatched_runtime))
    if not nested_ready:
        problems.append("RD-Agent DinD sandbox images are not preloaded")
    if not gpu_ready:
        problems.append("GPU sandbox images are not preloaded on the host daemon")
    if problems:
        return False, "; ".join(problems)
    return True, "runtime identities match and every governed sandbox image is preloaded"


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


def _runtime_release_identity(
    context: ComposeContext,
    services: Collection[str],
) -> tuple[bool, str]:
    configured = dotenv_values(context.env_file)
    try:
        expected_environment = normalized_release_identity(
            {key: configured.get(key) for key in RELEASE_IDENTITY_ENV_KEYS}
        )
        expected_labels = release_identity_labels(expected_environment)
    except ValueError as exc:
        return False, f"release identity is invalid in deployment environment: {exc}"
    mismatches: list[str] = []
    inspected = 0
    for service in sorted(set(services) - STATEFUL_RELEASE_IDENTITY_EXEMPT):
        container_id = context.container_id(service)
        if not container_id:
            mismatches.append(f"{service}:missing")
            continue
        inspection = json.loads(context.docker("inspect", container_id, capture=True))[0]
        entries = inspection.get("Config", {}).get("Env") or []
        environment = {
            str(item).split("=", 1)[0]: str(item).split("=", 1)[1]
            for item in entries
            if "=" in str(item)
        }
        labels = {
            str(key): str(value)
            for key, value in (inspection.get("Config", {}).get("Labels") or {}).items()
        }
        inspected += 1
        for variable, expected in expected_environment.items():
            if environment.get(variable) != expected:
                mismatches.append(f"{service}:env:{variable}")
        for label, expected in expected_labels.items():
            if labels.get(label) != expected:
                mismatches.append(f"{service}:label:{label}")
    if mismatches:
        return False, "mixed or unstamped services: " + ", ".join(mismatches)
    return (
        True,
        f"{inspected} stateless services share release "
        f"{expected_environment['QUANTLAB_RELEASE_ID']} as "
        f"{expected_environment['QUANTLAB_RELEASE_KIND']} aliasing "
        f"{expected_environment['QUANTLAB_RELEASE_ALIAS_OF']} and config "
        f"{expected_environment['QUANTLAB_CONFIG_DIGEST']}",
    )


def assess_release(
    context: ComposeContext,
    project_root: Path,
    *,
    minimum_free_gb: float = 20.0,
    required_services: Collection[str] | None = None,
    require_immutable_images: bool = True,
    verify_image_availability: bool = True,
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

    storage_valid, storage_evidence = _docker_storage_configuration(context)
    checks.append(
        _check(
            "rdagent_image_storage",
            "RD-Agent image storage uses dedicated data-disk paths",
            storage_valid,
            storage_evidence,
            "Set RDAGENT_DOCKER_HOST_PATH and RDAGENT_REGISTRY_HOST_PATH to "
            "distinct absolute siblings of QUANTLAB_DATA_HOST_PATH.",
        )
    )

    if require_immutable_images:
        images_valid, images_evidence = _immutable_image_configuration(context)
    else:
        images_valid = True
        images_evidence = "immutable image configuration will be sealed during this upgrade"
    checks.append(
        _check(
            "immutable_images",
            "Immutable RD-Agent execution images",
            images_valid,
            images_evidence,
            "Configure the required sha256-pinned runtime and sandbox images; "
            "enable the gpu profile only after its three images are pinned.",
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

    if verify_image_availability:
        availability_valid, availability_evidence = _preloaded_image_availability(
            context
        )
    else:
        availability_valid = True
        availability_evidence = "image availability verification is deferred for this phase"
    checks.append(
        _check(
            "immutable_images_preloaded",
            "Immutable execution images are present in their target daemons",
            availability_valid,
            availability_evidence,
            "Run the supported release upgrade to publish and preload the sealed "
            "sandbox images before starting research.",
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
    required = (
        set(required_services)
        if required_services is not None
        else expected_services(context)
    )
    healthchecked = required - NON_HEALTHCHECK_SERVICES
    missing = required - set(services)
    stopped = sorted(
        name
        for name in required & set(services)
        if services[name].get("State") != "running"
    )
    unhealthy = sorted(
        name
        for name in healthchecked & set(services)
        if services[name].get("Health") != "healthy"
    )
    services_ready = not missing and not stopped and not unhealthy
    checks.append(
        _check(
            "services_healthy",
            f"Current {len(required)}-service baseline is healthy",
            services_ready,
            (
                f"missing {sorted(missing)}; stopped {stopped}; unhealthy {unhealthy}"
                if not service_error
                else service_error
            ),
            "Restore every current service before introducing a new release.",
        )
    )

    if require_immutable_images and services_ready:
        try:
            release_identity_valid, release_identity_evidence = (
                _runtime_release_identity(context, required)
            )
        except Exception as exc:
            release_identity_valid = False
            release_identity_evidence = str(exc)[:500]
    elif require_immutable_images:
        release_identity_valid = False
        release_identity_evidence = "service health must pass before identity inspection"
    else:
        release_identity_valid = True
        release_identity_evidence = "release identity will be stamped during this upgrade"
    checks.append(
        _check(
            "release_identity",
            "One immutable release and configuration across stateless services",
            release_identity_valid,
            release_identity_evidence,
            "Recreate every default-profile service from one release Compose contract.",
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
