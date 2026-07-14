from __future__ import annotations

import json
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .backup_restore import (
    WRITER_SERVICES,
    ComposeContext,
    create_backup,
    load_and_verify_manifest,
    restore_backup,
)
from .release_preflight import assess_release

BUILT_SERVICES = ("api", "scheduler", "worker", "rdagent-worker", "web")
ROLLBACK_TAG = re.compile(
    r"^quantlab-rollback:(?P<release>[0-9]{8}t[0-9]{6}z)-"
    r"(?P<service>api|scheduler|worker|rdagent-worker|web)$"
)


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _capture_rollback_images(
    context: ComposeContext,
    release_id: str,
) -> dict[str, str]:
    tags: dict[str, str] = {}
    for service in BUILT_SERVICES:
        container_id = context.container_id(service)
        if not container_id:
            raise RuntimeError(f"cannot capture rollback image: {service} is not running")
        image_id = context.docker(
            "inspect",
            "--format",
            "{{.Image}}",
            container_id,
            capture=True,
        ).splitlines()[0]
        if not image_id.startswith("sha256:"):
            raise RuntimeError(f"cannot resolve rollback image for {service}")
        tag = f"quantlab-rollback:{release_id.lower()}-{service}"
        context.docker("tag", image_id, tag)
        tags[service] = tag
    return tags


def _rollback_override(path: Path, tags: dict[str, str]) -> None:
    path.write_text(
        json.dumps(
            {
                "services": {
                    service: {"image": tag, "build": None} for service, tag in sorted(tags.items())
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _prune_rollback_images(context: ComposeContext, retention_count: int) -> list[str]:
    raw = context.docker(
        "image",
        "ls",
        "--format",
        "{{.Repository}}:{{.Tag}}",
        "--filter",
        "reference=quantlab-rollback:*",
        capture=True,
    )
    releases: dict[str, list[str]] = {}
    for tag in raw.splitlines():
        match = ROLLBACK_TAG.fullmatch(tag.strip())
        if match:
            releases.setdefault(match.group("release"), []).append(tag.strip())
    retained = set(sorted(releases, reverse=True)[:retention_count])
    removed = sorted(
        tag for release, tags in releases.items() if release not in retained for tag in tags
    )
    if removed:
        context.docker("image", "rm", "-f", *removed, check=False)
    return removed


def _rollback_context(
    context: ComposeContext,
    override_file: Path,
) -> ComposeContext:
    return ComposeContext(
        project_name=context.project_name,
        env_file=context.env_file,
        compose_files=(*context.compose_files, override_file.resolve()),
    )


def _restore_previous_release(
    context: ComposeContext,
    backup_directory: Path,
    rollback_tags: dict[str, str],
    *,
    wait_timeout: int,
) -> dict[str, Any]:
    manifest = load_and_verify_manifest(backup_directory)
    with tempfile.TemporaryDirectory(prefix="quantlab-release-rollback-") as temporary:
        override_file = Path(temporary) / "rollback.compose.json"
        _rollback_override(override_file, rollback_tags)
        rollback = _rollback_context(context, override_file)
        rollback.run("stop", *WRITER_SERVICES, check=False)
        rollback.run(
            "up",
            "-d",
            "--no-build",
            "--wait",
            "--wait-timeout",
            str(wait_timeout),
            "postgres",
        )
        restored_revision = restore_backup(
            rollback,
            backup_directory,
            confirmed=True,
        )
        if restored_revision != manifest["schema_revision"]:
            raise RuntimeError(
                "rollback schema mismatch: "
                f"expected {manifest['schema_revision']}, got {restored_revision}"
            )
        rollback.run(
            "up",
            "-d",
            "--no-build",
            "--force-recreate",
            "--remove-orphans",
            "--wait",
            "--wait-timeout",
            str(wait_timeout),
        )
        return {
            "schema_revision": restored_revision,
            "images": rollback_tags,
            "services": rollback.running_services(),
        }


def run_release_upgrade(
    context: ComposeContext,
    project_root: Path,
    backup_root: Path,
    *,
    confirmed: bool,
    retention_count: int = 14,
    minimum_free_gb: float = 20.0,
    wait_timeout: int = 300,
    pull_images: bool = False,
    rollback_image_retention: int = 3,
) -> dict[str, Any]:
    if not confirmed:
        raise ValueError("release upgrade requires --confirm-upgrade")
    if retention_count < 1:
        raise ValueError("retention_count must be positive")
    if wait_timeout < 30:
        raise ValueError("wait_timeout must be at least 30 seconds")
    if rollback_image_retention < 1:
        raise ValueError("rollback_image_retention must be positive")

    release_id = _stamp()
    result: dict[str, Any] = {
        "release_id": release_id,
        "project_name": context.project_name,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "failed",
        "live_trading_enabled": False,
        "checks": {},
        "rollback_images": {},
        "backup_directory": None,
    }
    backup_directory: Path | None = None
    rollback_tags: dict[str, str] = {}
    state_mutated = False
    try:
        initial = assess_release(
            context,
            project_root,
            minimum_free_gb=minimum_free_gb,
        )
        result["checks"]["initial_preflight"] = initial
        if initial["status"] != "ready":
            result["status"] = "blocked"
            result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            return result

        rollback_tags = _capture_rollback_images(context, release_id)
        result["rollback_images"] = rollback_tags
        build_arguments = ["build"]
        if pull_images:
            build_arguments.append("--pull")
        context.run(*build_arguments, *BUILT_SERVICES)

        final_gate = assess_release(
            context,
            project_root,
            minimum_free_gb=minimum_free_gb,
        )
        result["checks"]["post_build_preflight"] = final_gate
        if final_gate["status"] != "ready":
            result["status"] = "blocked"
            result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            return result

        backup_directory = create_backup(
            context,
            backup_root,
            retention_count=retention_count,
            restart_services=False,
        )
        result["backup_directory"] = str(backup_directory)
        state_mutated = True

        context.run(
            "up",
            "-d",
            "--remove-orphans",
            "--wait",
            "--wait-timeout",
            str(wait_timeout),
        )
        acceptance = assess_release(
            context,
            project_root,
            minimum_free_gb=minimum_free_gb,
        )
        result["checks"]["post_upgrade_preflight"] = acceptance
        if acceptance["status"] != "ready" or acceptance["migration_state"] != "current":
            raise RuntimeError("post-upgrade release acceptance did not pass")

        result["status"] = "succeeded"
        try:
            result["pruned_rollback_images"] = _prune_rollback_images(
                context,
                rollback_image_retention,
            )
        except Exception as exc:  # cleanup must never roll back an accepted release
            result["pruned_rollback_images"] = []
            result["cleanup_warning"] = f"{type(exc).__name__}: {exc}"
        result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if not state_mutated or backup_directory is None or not rollback_tags:
            result["status"] = "failed"
            result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            return result
        try:
            result["rollback"] = _restore_previous_release(
                context,
                backup_directory,
                rollback_tags,
                wait_timeout=wait_timeout,
            )
        except Exception as rollback_exc:
            result["status"] = "rollback_failed"
            result["rollback_error"] = f"{type(rollback_exc).__name__}: {rollback_exc}"
        else:
            result["status"] = "rolled_back"
        result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        return result
