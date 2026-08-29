"""One-time convergence of a mixed Compose project onto one release contract.

This module exists only for adopting deployments that were historically
updated service-by-service.  It does not migrate or restore PostgreSQL, touch
``/data``, or change platform safe mode.  The current containers are recreated
from their already-running immutable image IDs and one canonical Compose/env
pair, producing a hash-chained append-only receipt that can also drive an
image-only rollback.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from .backup_restore import ComposeContext
from .control_plane_lock import control_plane_locked
from .deployment_services import (
    CORE_RUNTIME_SERVICES,
    OPTIONAL_PROFILE_SERVICES,
    WRITER_SERVICES,
)
from .release_identity import (
    RELEASE_IDENTITY_ENV_KEYS,
    STATEFUL_RELEASE_IDENTITY_EXEMPT,
    release_identity_environment,
    release_identity_labels,
)

_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_BASELINE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_CONFIG_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_LABEL_PREFIXES = ("com.docker.compose.", "quantlab.")


def _now() -> datetime:
    return datetime.now(UTC)


def _stamp() -> str:
    return _now().strftime("%Y%m%dT%H%M%SZ")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _required_runtime_services(context: ComposeContext) -> frozenset[str]:
    services = set(CORE_RUNTIME_SERVICES)
    for profile in getattr(context, "profiles", ()):
        services.update(OPTIONAL_PROFILE_SERVICES.get(str(profile), ()))
    return frozenset(services)


def _validate_receipt_root(context: ComposeContext, receipt_root: Path) -> Path:
    root = receipt_root.resolve()
    if root == root.parent:
        raise ValueError("canonical-baseline receipt root must not be a filesystem root")
    configured = dotenv_values(context.env_file)
    forbidden: list[Path] = []
    if os.name != "nt":
        forbidden.append(Path("/data"))
    for name in (
        "QUANTLAB_DATA_HOST_PATH",
        "RDAGENT_DOCKER_HOST_PATH",
        "RDAGENT_REGISTRY_HOST_PATH",
    ):
        raw = str(configured.get(name) or "").strip()
        candidate = Path(raw) if raw else None
        if candidate is not None and candidate.is_absolute():
            forbidden.append(candidate)
    if any(root == item.resolve() or _inside(root, item) for item in forbidden):
        raise ValueError("canonical-baseline receipts must stay outside governed /data paths")
    return root


def _environment_assignment_key(line: str) -> str | None:
    candidate = line.lstrip()
    if not candidate or candidate.startswith("#"):
        return None
    if candidate.startswith("export "):
        candidate = candidate.removeprefix("export ").lstrip()
    if "=" not in candidate:
        return None
    return candidate.split("=", 1)[0].strip() or None


def _configuration_digest(context: ComposeContext) -> str:
    """Hash canonical inputs while excluding the two self-referential stamps."""

    environment_lines: list[str] = []
    for line in context.env_file.read_text(encoding="utf-8-sig").splitlines():
        key = _environment_assignment_key(line)
        if key not in RELEASE_IDENTITY_ENV_KEYS:
            environment_lines.append(line)
    identity = {
        "contract_version": "quantlab-canonical-baseline-config-v1",
        "project_name": context.project_name,
        "profiles": sorted(str(item) for item in getattr(context, "profiles", ())),
        "environment_sha256": _sha256_bytes(
            ("\n".join(environment_lines) + "\n").encode("utf-8")
        ),
        "compose": [
            {
                "name": Path(item).name,
                "sha256": _sha256_bytes(Path(item).read_bytes()),
            }
            for item in context.compose_files
        ],
    }
    return _sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _artifact_hashes(context: ComposeContext) -> dict[str, Any]:
    return {
        "environment": {
            "path": str(context.env_file.resolve()),
            "sha256": _sha256_bytes(context.env_file.read_bytes()),
        },
        "compose": [
            {
                "path": str(Path(path).resolve()),
                "sha256": _sha256_bytes(Path(path).read_bytes()),
            }
            for path in context.compose_files
        ],
    }


def _atomic_write(path: Path, content: bytes, *, mode: int = 0o600) -> None:
    path = path.resolve()
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        temporary.unlink(missing_ok=True)


def _update_environment_identity(path: Path, identity: dict[str, str]) -> None:
    original = path.read_text(encoding="utf-8-sig")
    output: list[str] = []
    for line in original.splitlines():
        if _environment_assignment_key(line) not in RELEASE_IDENTITY_ENV_KEYS:
            output.append(line)
    output.extend(f"{key}={value}" for key, value in identity.items())
    _atomic_write(path, ("\n".join(output) + "\n").encode("utf-8"), mode=0o600)


class _AppendOnlyReceipt:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        descriptor = os.open(
            self.path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        os.close(descriptor)
        self.sequence = 0
        self.previous_sha256 = "0" * 64

    def append(self, event: str, payload: dict[str, Any]) -> None:
        self.sequence += 1
        record: dict[str, Any] = {
            "sequence": self.sequence,
            "recorded_at": _now().isoformat(timespec="microseconds"),
            "event": event,
            "previous_sha256": self.previous_sha256,
            "payload": payload,
        }
        record_sha256 = _sha256_bytes(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        record["record_sha256"] = record_sha256
        line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(descriptor, line)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.previous_sha256 = record_sha256

    def seal(self) -> None:
        if os.name != "nt":
            os.chmod(self.path, stat.S_IRUSR)


def _inspect_services(
    context: ComposeContext,
    services: frozenset[str],
    *,
    require_healthy: bool,
) -> dict[str, dict[str, Any]]:
    captured: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for service in sorted(services):
        container_id = context.container_id(service, all_states=True)
        if not container_id:
            failures.append(f"{service}:missing")
            continue
        raw = json.loads(context.docker("inspect", container_id, capture=True))
        if not isinstance(raw, list) or len(raw) != 1:
            failures.append(f"{service}:unreadable")
            continue
        inspection = raw[0]
        image_id = str(inspection.get("Image") or "").lower()
        state = inspection.get("State") or {}
        health = state.get("Health") or {}
        labels = inspection.get("Config", {}).get("Labels") or {}
        environment = inspection.get("Config", {}).get("Env") or []
        safe_environment = {
            key: value
            for item in environment
            if "=" in str(item)
            for key, value in (str(item).split("=", 1),)
            if key in RELEASE_IDENTITY_ENV_KEYS
        }
        safe_labels = {
            str(key): str(value)
            for key, value in labels.items()
            if str(key).startswith(_SAFE_LABEL_PREFIXES)
        }
        if not _IMAGE_ID.fullmatch(image_id):
            failures.append(f"{service}:mutable_image_identity")
        if state.get("Running") is not True:
            failures.append(f"{service}:not_running")
        health_status = str(health.get("Status") or "")
        if require_healthy and health_status and health_status != "healthy":
            failures.append(f"{service}:health_{health_status}")
        captured[service] = {
            "container_id": str(inspection.get("Id") or container_id),
            "image_id": image_id,
            "state": str(state.get("Status") or "unknown"),
            "health": health_status or None,
            "config_labels": safe_labels,
            "release_environment": safe_environment,
        }
    if failures:
        raise RuntimeError("runtime service discovery failed: " + ", ".join(failures))
    return captured


def _durable_queue_state(context: ComposeContext) -> dict[str, int]:
    query = (
        "SELECT "
        "(SELECT count(*) FROM quantlab.jobs WHERE status IN ('queued','running')) "
        "|| '|' || "
        "(SELECT count(*) FROM quantlab.work_units WHERE status='running');"
    )
    raw = context.run(
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "quantlab",
        "-d",
        "quantlab",
        "-Atc",
        query,
        capture=True,
    )
    for line in reversed(raw.splitlines()):
        values = line.strip().split("|")
        if len(values) != 2:
            continue
        try:
            active_jobs, running_units = (int(item) for item in values)
        except ValueError:
            continue
        if active_jobs < 0 or running_units < 0:
            break
        return {"active_jobs": active_jobs, "running_units": running_units}
    raise RuntimeError("durable queue state is unreadable")


def _canonical_service_config_hashes(
    context: ComposeContext,
    services: frozenset[str],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for service in sorted(services):
        raw = context.run("config", "--hash", service, capture=True)
        parsed: str | None = None
        for line in reversed(raw.splitlines()):
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] == service and _CONFIG_SHA256.fullmatch(
                parts[1]
            ):
                parsed = parts[1]
                break
        if parsed is None:
            raise RuntimeError(f"canonical Compose config hash is unreadable for {service}")
        hashes[service] = parsed
    return hashes


def _config_equivalence(
    captured: dict[str, dict[str, Any]],
    canonical_hashes: dict[str, str],
) -> tuple[bool, list[str]]:
    mismatched = [
        service
        for service, expected in sorted(canonical_hashes.items())
        if captured[service]["config_labels"].get(
            "com.docker.compose.config-hash"
        )
        != expected
    ]
    return not mismatched, mismatched


def _queue_is_idle(state: dict[str, int]) -> bool:
    return state == {"active_jobs": 0, "running_units": 0}


def _override_payload(
    services: frozenset[str],
    captured: dict[str, dict[str, Any]],
    *,
    identity: dict[str, str] | None,
) -> dict[str, Any]:
    service_payload: dict[str, Any] = {}
    for service in sorted(services):
        entry: dict[str, Any] = {"image": captured[service]["image_id"]}
        if identity is not None:
            entry["labels"] = release_identity_labels(identity)
            if service not in STATEFUL_RELEASE_IDENTITY_EXEMPT:
                entry["environment"] = identity
        service_payload[service] = entry
    return {"services": service_payload}


def _write_override(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
        mode=0o600,
    )


def _context_with_override(context: ComposeContext, override: Path) -> ComposeContext:
    return ComposeContext(
        project_name=context.project_name,
        env_file=context.env_file,
        compose_files=(*context.compose_files, override.resolve()),
        profiles=context.profiles,
        project_directory=context.project_directory,
    )


def _up_arguments(services: set[str] | frozenset[str], wait_timeout: int) -> tuple[str, ...]:
    return (
        "up",
        "-d",
        "--no-build",
        "--no-deps",
        "--force-recreate",
        "--remove-orphans",
        "--wait",
        "--wait-timeout",
        str(wait_timeout),
        *sorted(services),
    )


def _verify_converged_services(
    context: ComposeContext,
    services: frozenset[str],
    captured: dict[str, dict[str, Any]],
    *,
    identity: dict[str, str],
    override: Path,
) -> dict[str, dict[str, Any]]:
    observed = _inspect_services(context, services, require_healthy=True)
    failures: list[str] = []
    override_path = str(override.resolve())
    for service in sorted(services):
        item = observed[service]
        labels = item["config_labels"]
        if item["image_id"] != captured[service]["image_id"]:
            failures.append(f"{service}:image")
        if labels.get("com.docker.compose.project") != context.project_name:
            failures.append(f"{service}:project")
        if labels.get("com.docker.compose.service") != service:
            failures.append(f"{service}:service")
        if override_path not in labels.get("com.docker.compose.project.config_files", ""):
            failures.append(f"{service}:compose_contract")
        for label, expected in release_identity_labels(identity).items():
            if labels.get(label) != expected:
                failures.append(f"{service}:{label}")
        if (
            service not in STATEFUL_RELEASE_IDENTITY_EXEMPT
            and item["release_environment"] != identity
        ):
            failures.append(f"{service}:release_environment")
    if failures:
        raise RuntimeError("canonical baseline verification failed: " + ", ".join(failures))
    return observed


def _verify_rollback_images(
    context: ComposeContext,
    services: frozenset[str],
    captured: dict[str, dict[str, Any]],
    *,
    require_healthy: bool = True,
) -> dict[str, dict[str, Any]]:
    observed = _inspect_services(context, services, require_healthy=require_healthy)
    mismatched = [
        service
        for service in sorted(services)
        if observed[service]["image_id"] != captured[service]["image_id"]
    ]
    if mismatched:
        raise RuntimeError("rollback image verification failed: " + ", ".join(mismatched))
    return observed


@control_plane_locked
def converge_canonical_baseline(
    context: ComposeContext,
    receipt_root: Path,
    *,
    confirmed: bool = False,
    release_id: str | None = None,
    wait_timeout: int = 600,
) -> dict[str, Any]:
    """Converge a mixed deployment, or return a mutation-free dry run.

    ``confirmed=False`` is deliberately the default.  The dry run still
    inspects every required service and checks the durable queues, but creates
    no directory, receipt, override, image tag, container, or environment edit.
    """

    if wait_timeout < 30:
        raise ValueError("canonical baseline wait_timeout must be at least 30 seconds")
    root = _validate_receipt_root(context, receipt_root)
    services = _required_runtime_services(context)
    captured = _inspect_services(context, services, require_healthy=True)
    before_queue = _durable_queue_state(context)
    config_digest = _configuration_digest(context)
    chosen_release_id = release_id or f"canonical-{_stamp()}-{config_digest[:12]}"
    if not _BASELINE_ID.fullmatch(chosen_release_id):
        raise ValueError("canonical baseline release_id is invalid")
    if not _CONFIG_SHA256.fullmatch(config_digest):  # pragma: no cover - hashlib invariant
        raise RuntimeError("canonical baseline configuration digest is invalid")
    identity = release_identity_environment(chosen_release_id, config_digest)
    baseline_directory = root / chosen_release_id
    receipt_path = baseline_directory / "receipt.jsonl"
    canonical_override = baseline_directory / "canonical.override.json"
    rollback_override = baseline_directory / "rollback.override.json"
    result: dict[str, Any] = {
        "status": "dry_run" if confirmed is False else "blocked",
        "mutation_confirmed": confirmed,
        "release_id": chosen_release_id,
        "config_digest": config_digest,
        "release_identity": identity,
        "required_services": sorted(services),
        "writer_services": sorted(services.intersection(WRITER_SERVICES)),
        "queue_before_stop": before_queue,
        "canonical_inputs": _artifact_hashes(context),
        "service_snapshot": captured,
        "receipt_path": str(receipt_path.resolve()),
        "canonical_override": str(canonical_override.resolve()),
        "safe_mode_action": "none",
        "postgres_data_action": "none",
        "governed_data_action": "none",
        "live_trading_enabled": False,
    }
    if not _queue_is_idle(before_queue):
        result["status"] = "blocked"
        result["blocker"] = "durable_queue_not_idle"
        return result
    canonical_hashes = _canonical_service_config_hashes(context, services)
    equivalent, mismatched_services = _config_equivalence(
        captured,
        canonical_hashes,
    )
    result["canonical_service_config_hashes"] = canonical_hashes
    result["rollback_config_equivalence"] = {
        "status": "pass" if equivalent else "block",
        "mismatched_services": mismatched_services,
        "evidence": (
            "every running service has the canonical Compose config hash"
            if equivalent
            else "image-only rollback cannot reproduce non-canonical service configuration"
        ),
    }
    if not equivalent:
        result["status"] = "blocked"
        result["blocker"] = "rollback_contract_not_config_equivalent"
        return result
    if not confirmed:
        return result
    if baseline_directory.exists():
        raise FileExistsError(
            f"canonical baseline directory already exists: {baseline_directory}"
        )
    root.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(root, 0o700)
    baseline_directory.mkdir(mode=0o700)
    receipt = _AppendOnlyReceipt(receipt_path)
    receipt.append(
        "discovery_completed",
        {
            "release_id": chosen_release_id,
            "config_digest": config_digest,
            "required_services": sorted(services),
            "queue": before_queue,
            "canonical_inputs": result["canonical_inputs"],
            "services": captured,
        },
    )

    original_environment = context.env_file.read_bytes()
    stopped = tuple(sorted(services.intersection(WRITER_SERVICES)))
    identity_mutated = False
    canonical_context: ComposeContext | None = None
    try:
        if stopped:
            context.run("stop", *stopped)
        receipt.append("writers_stopped", {"services": list(stopped)})
        after_stop_queue = _durable_queue_state(context)
        result["queue_after_stop"] = after_stop_queue
        receipt.append("post_stop_queue_checked", after_stop_queue)
        if not _queue_is_idle(after_stop_queue):
            if stopped:
                context.run("start", *stopped, check=False)
            # ``docker compose start`` has no ``--wait`` contract.  At this point
            # no image or configuration was changed, so verify the only rollback
            # property that matters without racing application health checks.
            restored = _verify_rollback_images(
                context,
                services,
                captured,
                require_healthy=False,
            )
            receipt.append(
                "convergence_blocked_after_stop",
                {
                    "queue": after_stop_queue,
                    "original_images_verified": sorted(restored),
                },
            )
            receipt.seal()
            result["status"] = "blocked"
            result["blocker"] = "durable_queue_became_active_while_stopping_writers"
            result["rollback"] = {"status": "not_needed", "images_verified": True}
            return result

        _update_environment_identity(context.env_file, identity)
        identity_mutated = True
        result["environment_after"] = {
            "path": str(context.env_file.resolve()),
            "sha256": _sha256_bytes(context.env_file.read_bytes()),
        }
        _write_override(
            canonical_override,
            _override_payload(
                services,
                captured,
                identity=identity,
            ),
        )
        canonical_context = _context_with_override(context, canonical_override)
        receipt.append(
            "canonical_contract_persisted",
            {
                "environment_sha256": result["environment_after"]["sha256"],
                "override_path": str(canonical_override.resolve()),
                "override_sha256": _sha256_bytes(canonical_override.read_bytes()),
            },
        )

        core = set(services) - {"scheduler", "gateway"}
        if core:
            canonical_context.run(
                *_up_arguments(core, wait_timeout),
                timeout=wait_timeout + 30,
            )
        for service in ("scheduler", "gateway"):
            if service in services:
                canonical_context.run(
                    *_up_arguments({service}, wait_timeout),
                    timeout=wait_timeout + 30,
                )
        observed = _verify_converged_services(
            canonical_context,
            services,
            captured,
            identity=identity,
            override=canonical_override,
        )
        receipt.append(
            "convergence_succeeded",
            {
                "services": observed,
                "safe_mode_action": "none",
                "postgres_data_action": "none",
                "governed_data_action": "none",
            },
        )
        receipt.seal()
        result["status"] = "succeeded"
        result["services"] = observed
        return result
    except Exception as exc:  # rollback is the purpose of this bounded adopter
        result["error"] = f"{type(exc).__name__}: {exc}"
        try:
            receipt.append("convergence_failed", {"error": result["error"]})
        except Exception:  # pragma: no cover - preserve the primary failure
            pass
        if identity_mutated:
            _atomic_write(context.env_file, original_environment, mode=0o600)
        try:
            active_context = canonical_context or context
            if stopped:
                active_context.run("stop", *stopped, check=False)
            _write_override(
                rollback_override,
                _override_payload(
                    services,
                    captured,
                    identity=None,
                ),
            )
            rollback_context = _context_with_override(context, rollback_override)
            rollback_context.run(
                *_up_arguments(set(services), wait_timeout),
                timeout=wait_timeout + 30,
            )
            restored = _verify_rollback_images(rollback_context, services, captured)
            receipt.append(
                "rollback_succeeded",
                {
                    "services": sorted(restored),
                    "rollback_override": str(rollback_override.resolve()),
                    "rollback_override_sha256": _sha256_bytes(
                        rollback_override.read_bytes()
                    ),
                },
            )
            result["status"] = "rolled_back"
            result["rollback"] = {"status": "succeeded", "images_verified": True}
        except Exception as rollback_exc:  # pragma: no cover - covered through status tests
            result["status"] = "rollback_failed"
            result["rollback"] = {
                "status": "failed",
                "error": f"{type(rollback_exc).__name__}: {rollback_exc}",
            }
            try:
                receipt.append("rollback_failed", result["rollback"])
            except Exception:
                pass
        receipt.seal()
        return result
