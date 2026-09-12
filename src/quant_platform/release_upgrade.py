from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .backup_restore import (
    CONTROL_PLANE_BACKUP_FORMAT_VERSION,
    FULL_BACKUP_FORMAT_VERSION,
    WRITER_SERVICES,
    ComposeContext,
    _platform_secret_key_fingerprint,
    assess_control_plane_backup_capacity,
    create_backup,
    load_and_verify_manifest,
    restore_backup,
)
from .control_plane_lock import control_plane_locked
from .deployment_services import (
    BUILT_APPLICATION_SERVICES,
    PROFILE_BUILT_SERVICES,
)
from .release_identity import (
    RELEASE_IDENTITY_ENV_KEYS,
    STATEFUL_RELEASE_IDENTITY_EXEMPT,
    release_identity_environment,
)
from .release_preflight import (
    LEGACY_EXPECTED_SERVICES,
    _runtime_release_identity,
    assess_release,
    expected_services,
)

BUILT_SERVICES = BUILT_APPLICATION_SERVICES
INTRODUCED_SERVICES: set[str] = set()
_ROLLBACK_SERVICES = tuple(
    sorted(
        LEGACY_EXPECTED_SERVICES.union(BUILT_SERVICES).union(
            PROFILE_BUILT_SERVICES.get("gpu", ())
        )
    )
)
ROLLBACK_TAG = re.compile(
    r"^quantlab-rollback:(?P<release>[0-9]{8}t[0-9]{6}z)-"
    rf"(?P<service>{'|'.join(re.escape(item) for item in _ROLLBACK_SERVICES)})$"
)
_GIB = 1024**3
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_REGISTRY_PORT = 55000
_QLIB_COMMIT = "d5379c520f66a39953bad76234a7019a72796fd0"
_GOVERNED_SANDBOX_CONTEXTS = {
    "qlib": "qlib-sandbox",
    "model": "model-sandbox",
}
_ADMISSION_SERVICES = ("gateway", "scheduler", "api")
_MODEL_REUSE_EXTRA_MODULES = (
    "qlib_portfolio_calendar.py", "qlib_research_strategy.py", "qlib_workflow.py",
    "research_execution_cadence.py", "research_horizon.py", "upstream_versions.py",
)
_MODEL_REUSE_ENTRYPOINTS = (
    "scripts/evaluate_model_batch.py", "scripts/model_sandbox_runner.py",
    "scripts/prepare_model_data.py",
)


def _model_reuse_source_inventory(root: Path) -> dict[str, str]:
    """Bind numeric execution/preparation and the build inputs of its image.

    The controller and the image are different source views: runner scripts are
    mounted at execution time, while dependencies survive in site-packages after
    the sandbox Dockerfile removes /app. Both views must agree before reuse.
    """
    from .runtime_source_closure import local_python_source_closure_inventory

    closure = local_python_source_closure_inventory(root, entry_paths=_MODEL_REUSE_ENTRYPOINTS)
    required = [
        *(item["path"] for item in closure["inventory"]),
        *_MODEL_REUSE_ENTRYPOINTS,
        "pyproject.toml", ".dockerignore", "deploy/Dockerfile.worker",
        "deploy/Dockerfile.governed-full-source-overlay", "deploy/model-sandbox/Dockerfile",
        *("src/quant_platform/" + name for name in _MODEL_REUSE_EXTRA_MODULES),
        "src/quant_platform/model_recompute.py", "src/quant_platform/model_prepared_data.py",
        "src/quant_platform/model_prepared_execution.py", "src/quant_data/__init__.py",
    ]
    files = {root / name for name in required}
    files.update((root / "src/quant_platform").glob("model_*.py"))
    files.update((root / "src/quant_data").rglob("*.py"))
    files.update(path for path in (root / "deploy/model-sandbox").rglob("*") if path.is_file())
    result = {}
    for path in sorted(files):
        if not path.is_file() or path.is_symlink() or not _inside_path(path, root):
            raise RuntimeError("model sandbox reuse source missing or unsafe: " + str(path))
        result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _model_reuse_runtime_probe(sources: dict[str, str], *, installed_only: bool) -> str:
    expected = {
        name: digest for name, digest in sources.items()
        if name.startswith("src/") or (not installed_only and name in _MODEL_REUSE_ENTRYPOINTS)
    }
    return "\n".join((
        "import hashlib, json, sysconfig",
        "from pathlib import Path",
        "expected = " + repr(expected),
        "installed_only = " + repr(installed_only),
        "purelib = Path(sysconfig.get_paths()['purelib'])",
        "observed = {}",
        "for name, want in expected.items():",
        "    paths = [] if installed_only else [Path('/app') / name]",
        "    if name.startswith('src/'): paths.append(purelib / name[4:])",
        "    for path in paths:",
        "        if not path.is_file() or path.is_symlink(): raise RuntimeError('unsafe: '+name)",
        "        if hashlib.sha256(path.read_bytes()).hexdigest() != want:",
        "            raise RuntimeError('model reuse source differs: '+name)",
        "    observed[name] = want",
        "print(json.dumps(observed, sort_keys=True))",
    ))


def _assert_model_reuse_image(context: ComposeContext, image: str, image_id: str) -> None:
    observed = context.run(
        "exec", "-T", "rdagent-docker", "docker", "image", "inspect", "--format", "{{.Id}}",
        image, capture=True, timeout=30,
    ).strip()
    if observed != image_id:
        raise RuntimeError("preserved model sandbox is missing or its image ID changed")


def _verify_model_reuse_runtime(
    context: ComposeContext, sources: dict[str, str], image: str, *, live_worker: bool,
) -> None:
    installed_only = not live_worker
    probe = _model_reuse_runtime_probe(sources, installed_only=installed_only)
    command = (
        ("exec", "-T", "worker", "python", "-I", "-B", "-c", probe)
        if live_worker else (
            "exec", "-T", "rdagent-docker", "docker", "run", "--rm", "--network", "none",
            "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", "256m", "--cpus", "1", "--pids-limit", "64", "--entrypoint", "python",
            image, "-I", "-B", "-c", probe,
        )
    )
    expected = {
        name: digest for name, digest in sources.items()
        if name.startswith("src/") or (live_worker and name in _MODEL_REUSE_ENTRYPOINTS)
    }
    observed = json.loads(context.run(*command, capture=True, timeout=120))
    if observed != expected:
        raise RuntimeError("model sandbox reuse source verification returned incomplete evidence")


def _verify_model_sandbox_reuse(
    context: ComposeContext, project_root: Path, baseline_root: Path,
    image: str, image_id: str,
) -> dict[str, Any]:
    if not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image) or not _IMAGE_ID.fullmatch(image_id):
        raise ValueError("model sandbox reuse requires a digest reference and exact image ID")
    from dotenv import dotenv_values

    if dotenv_values(context.env_file).get("MODEL_SANDBOX_IMAGE") != image:
        raise RuntimeError("candidate does not preserve the configured model sandbox")
    if dotenv_values(baseline_root / "deploy/.env").get("MODEL_SANDBOX_IMAGE") != image:
        raise RuntimeError("requested model sandbox differs from the existing release")
    sources = _model_reuse_source_inventory(project_root)
    if _model_reuse_source_inventory(baseline_root) != sources:
        raise RuntimeError("model execution, preparation, or image build sources changed")
    _assert_model_reuse_image(context, image, image_id)
    _verify_model_reuse_runtime(context, sources, image, live_worker=True)
    _verify_model_reuse_runtime(context, sources, image, live_worker=False)
    return {
        "contract_version": "model-sandbox-reuse-v1", "status": "verified",
        "image": image, "image_id": image_id, "sources": sources,
        "source_inventory_sha256": hashlib.sha256(
            json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "baseline_root": str(baseline_root.resolve()),
        "live_worker_sources_verified": True, "sandbox_installed_sources_verified": True,
    }


@dataclass(frozen=True, slots=True)
class RollbackComposeContract:
    project_name: str
    working_directory: Path
    env_source: Path
    env_content: bytes
    compose_sources: tuple[Path, ...]
    compose_contents: tuple[bytes, ...]
    profiles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReusableBackup:
    directory: Path
    manifest: dict[str, Any]
    data_source: Path
    expected_data_bytes: int
    observed_data_bytes: int
    size_tolerance_bytes: int


def _rollback_contract_context(
    contract: RollbackComposeContract,
) -> ComposeContext:
    return ComposeContext(
        project_name=contract.project_name,
        env_file=contract.env_source,
        compose_files=contract.compose_sources,
        profiles=contract.profiles,
        project_directory=contract.working_directory,
    )


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _built_services(context: ComposeContext) -> tuple[str, ...]:
    services = list(BUILT_SERVICES)
    for profile in getattr(context, "profiles", ()):
        services.extend(PROFILE_BUILT_SERVICES.get(profile, ()))
    return tuple(dict.fromkeys(services))


def _inside_path(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _capture_rollback_compose_contract(
    context: ComposeContext,
    project_root: Path,
    *,
    services: tuple[str, ...],
    allow_missing: frozenset[str] = frozenset(),
    trusted_external_root: Path | None = None,
) -> RollbackComposeContract:
    """Capture the exact Compose contract owning the running stateless services.

    A protected stateful service can legitimately survive several immutable
    application releases, so its Compose provenance labels can point at a
    release directory which retention has already removed.  It remains part
    of the live project identity check, but it must not select (or veto) the
    application contract used to restore the writers and control plane.
    """

    identities: set[tuple[Path, tuple[Path, ...], Path | None]] = set()
    present_services: set[str] = set()
    for service in services:
        container_id = context.container_id(service)
        if not container_id:
            if service in allow_missing:
                continue
            raise RuntimeError(
                f"cannot capture rollback Compose contract: {service} is not running"
            )
        inspection = json.loads(context.docker("inspect", container_id, capture=True))[0]
        labels = inspection.get("Config", {}).get("Labels") or {}
        if labels.get("com.docker.compose.project") != context.project_name:
            raise RuntimeError(f"rollback container {service} belongs to another Compose project")
        if labels.get("com.docker.compose.service") != service:
            raise RuntimeError(f"rollback container {service} has inconsistent service labels")
        present_services.add(service)
        if service in STATEFUL_RELEASE_IDENTITY_EXEMPT:
            continue
        working_raw = str(
            labels.get("com.docker.compose.project.working_dir") or ""
        ).strip()
        files_raw = str(
            labels.get("com.docker.compose.project.config_files") or ""
        ).strip()
        environment_raw = str(
            labels.get("com.docker.compose.project.environment_file") or ""
        ).strip()
        if not working_raw or not files_raw:
            raise RuntimeError(
                f"rollback container {service} lacks its original Compose file labels"
            )
        working_directory = Path(working_raw).expanduser().resolve()
        if not working_directory.is_dir():
            raise RuntimeError("the previous release working directory no longer exists")
        raw_sources = tuple(
            (
                Path(item.strip())
                if Path(item.strip()).is_absolute()
                else working_directory / item.strip()
            )
            for item in files_raw.split(",")
            if item.strip()
        )
        if any(item.is_symlink() for item in raw_sources):
            raise RuntimeError("previous release Compose files must not be symlinks")
        compose_sources = tuple(item.resolve() for item in raw_sources)
        if not compose_sources or any(not item.is_file() for item in compose_sources):
            raise RuntimeError("the previous release Compose files are unavailable")
        for item in compose_sources:
            if _inside_path(item, working_directory):
                continue
            if trusted_external_root is None or not _inside_path(
                item, trusted_external_root.resolve()
            ):
                raise RuntimeError(
                    "external rollback Compose override is outside the trusted backup root"
                )
            metadata = item.stat()
            if os.name != "nt":
                if metadata.st_uid != os.geteuid():
                    raise RuntimeError(
                        "external rollback Compose override is not owned by the release user"
                    )
                if stat.S_IMODE(metadata.st_mode) & 0o022:
                    raise RuntimeError(
                        "external rollback Compose override is group/world writable"
                    )
        environment_source: Path | None = None
        if environment_raw:
            raw_environment_source = Path(environment_raw).expanduser()
            if not raw_environment_source.is_absolute():
                raw_environment_source = working_directory / raw_environment_source
            if raw_environment_source.is_symlink():
                raise RuntimeError(
                    "the previous release Compose environment file must not be a symlink"
                )
            environment_source = raw_environment_source.resolve()
            if not environment_source.is_file():
                raise RuntimeError(
                    "the previous release Compose environment file is unavailable"
                )
            if not _inside_path(environment_source, working_directory):
                if trusted_external_root is None or not _inside_path(
                    environment_source, trusted_external_root.resolve()
                ):
                    raise RuntimeError(
                        "external rollback Compose environment is outside the trusted backup root"
                    )
                metadata = environment_source.stat()
                if os.name != "nt":
                    if metadata.st_uid != os.geteuid():
                        raise RuntimeError(
                            "external rollback Compose environment is not owned by the release user"
                        )
                    if stat.S_IMODE(metadata.st_mode) & 0o022:
                        raise RuntimeError(
                            "external rollback Compose environment is group/world writable"
                        )
        identities.add((working_directory, compose_sources, environment_source))
    if not identities:
        raise RuntimeError(
            "cannot capture rollback Compose contract without a running "
            "stateless release service"
        )
    if len(identities) != 1:
        raise RuntimeError(
            "running stateless rollback services do not share one Compose release contract"
        )

    working_directory, compose_sources, labeled_env_source = identities.pop()
    if labeled_env_source is not None:
        existing_envs = (labeled_env_source,)
    else:
        relative_env: Path | None = None
        try:
            relative_env = context.env_file.resolve().relative_to(project_root.resolve())
        except ValueError:
            pass
        candidates = [(working_directory / context.env_file.name).resolve()]
        if relative_env is not None:
            candidates.append((working_directory / relative_env).resolve())
        candidates.extend(
            (item.parent / context.env_file.name).resolve() for item in compose_sources
        )
        existing_envs = tuple(
            dict.fromkeys(
                item
                for item in candidates
                if item.is_file() and _inside_path(item, working_directory)
            )
        )
    if not existing_envs:
        raise RuntimeError("the previous release Compose environment file is unavailable")
    env_source = existing_envs[0]
    env_content = env_source.read_bytes()
    if any(item.read_bytes() != env_content for item in existing_envs[1:]):
        raise RuntimeError("the previous release Compose environment is ambiguous")

    profiles = tuple(
        profile
        for profile, profile_services in PROFILE_BUILT_SERVICES.items()
        if present_services.intersection(profile_services)
    )
    contract = RollbackComposeContract(
        project_name=context.project_name,
        working_directory=working_directory,
        env_source=env_source,
        env_content=env_content,
        compose_sources=compose_sources,
        compose_contents=tuple(item.read_bytes() for item in compose_sources),
        profiles=profiles,
    )
    _rollback_contract_context(contract).run("config", "--quiet")
    return contract


def _persist_rollback_compose_contract(
    contract: RollbackComposeContract,
    backup_directory: Path,
    *,
    directory_name: str = "rollback-compose-contract",
) -> ComposeContext:
    """Persist the old Compose bytes without copying plaintext release secrets.

    The original environment file remains the rollback authority.  It was
    already required to be a regular, trusted file by
    :func:`_capture_rollback_compose_contract`; binding its path and digest in
    this manifest gives us an exact rollback contract without silently adding
    credentials to an otherwise sanitized backup directory.
    """

    if not re.fullmatch(r"rollback-compose-contract(?:-[0-9a-z]+)?", directory_name):
        raise ValueError("rollback Compose contract directory name is invalid")
    root = (backup_directory.resolve() / directory_name).resolve()
    if not _inside_path(root, backup_directory.resolve()):
        raise RuntimeError("rollback Compose contract target escapes the backup")
    root.mkdir(mode=0o700)
    if contract.env_source.is_symlink() or not contract.env_source.is_file():
        raise RuntimeError("rollback Compose environment is no longer available")
    if contract.env_source.read_bytes() != contract.env_content:
        raise RuntimeError("rollback Compose environment changed after capture")
    compose_targets: list[Path] = []
    compose_entries: list[dict[str, str]] = []
    for index, (source, content) in enumerate(
        zip(contract.compose_sources, contract.compose_contents, strict=True)
    ):
        suffix = source.suffix if source.suffix in {".yaml", ".yml", ".json"} else ".yaml"
        target = root / f"compose-{index:02d}{suffix}"
        _atomic_replace(target, content)
        target.chmod(0o600)
        compose_targets.append(target)
        compose_entries.append(
            {
                "source": str(source),
                "snapshot": target.name,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    manifest = {
        "format_version": 1,
        "project_name": contract.project_name,
        "working_directory": str(contract.working_directory),
        "env_source": str(contract.env_source),
        "env_snapshot": None,
        "env_sha256": hashlib.sha256(contract.env_content).hexdigest(),
        "profiles": list(contract.profiles),
        "compose_files": compose_entries,
    }
    manifest_target = root / "manifest.json"
    _atomic_replace(
        manifest_target,
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    manifest_target.chmod(0o600)
    for target, content in zip(compose_targets, contract.compose_contents, strict=True):
        if target.read_bytes() != content:
            raise RuntimeError("persisted rollback Compose file failed verification")
    if contract.env_source.read_bytes() != contract.env_content:
        raise RuntimeError("persisted rollback environment failed verification")
    rollback = ComposeContext(
        project_name=contract.project_name,
        env_file=contract.env_source,
        compose_files=tuple(compose_targets),
        profiles=contract.profiles,
        project_directory=contract.working_directory,
    )
    rollback.run("config", "--quiet")
    return rollback


def _effective_env_value(context: ComposeContext, name: str) -> str:
    if name in os.environ:
        return os.environ[name].strip()
    prefix = f"{name}="
    for line in context.env_file.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _reuse_database_state(context: ComposeContext) -> tuple[int, int, str]:
    query = (
        "SELECT "
        "(SELECT count(*) FROM quantlab.jobs WHERE status IN ('queued','running')) "
        "|| '|' || (SELECT count(*) FROM quantlab.work_units WHERE status='running') "
        "|| '|' || (SELECT version_num FROM quantlab.alembic_version);"
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
    ).splitlines()
    try:
        active_jobs, running_units, revision = raw[-1].split("|", 2)
        return int(active_jobs), int(running_units), revision.strip()
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("reusable-backup database state is unreadable") from exc


def _quiesce_release_admission(
    context: ComposeContext,
    *,
    stopped_services: list[str],
    wait_timeout: int,
) -> dict[str, Any]:
    """Close producers before draining work accepted after the idle preflight.

    Compose may reorder a multi-service stop by dependencies. Stop each admission
    service separately, keeping consumers alive until accepted work completes.
    Record ownership before each command so a partially failed stop can be undone.
    """
    running = set(context.running_services())
    _active, _units, schema = _reuse_database_state(context)
    for service in _ADMISSION_SERVICES:
        if service not in running:
            continue
        stopped_services.append(service)
        context.run("stop", "--timeout", str(wait_timeout), service)
        if service in context.running_services():
            raise RuntimeError(f"release admission service remained running: {service}")

    started = time.monotonic()
    while True:
        if set(context.running_services()).intersection(_ADMISSION_SERVICES):
            raise RuntimeError("release admission resumed while accepted work was draining")
        active_jobs, running_units, revision = _reuse_database_state(context)
        if revision != schema:
            raise RuntimeError("database schema changed while release admission was quiesced")
        if active_jobs == 0 and running_units == 0:
            return {
                "status": "pass",
                "stopped_services": list(stopped_services),
                "schema_revision": schema,
                "active_jobs": 0,
                "running_units": 0,
                "drain_seconds": time.monotonic() - started,
                "accepted_work_cancelled": False,
            }
        if time.monotonic() - started >= wait_timeout:
            raise RuntimeError(
                "accepted work did not drain before release timeout: "
                f"{active_jobs} active jobs, {running_units} running work units"
            )
        time.sleep(2)


def _assert_release_backup_idle(context: ComposeContext, expected_schema: str) -> None:
    """Reject a stop/admission race before any PostgreSQL dump is produced."""
    remaining = set(context.running_services()).intersection(WRITER_SERVICES)
    if remaining:
        raise RuntimeError("writer services remained active before release backup: "
                           + ", ".join(sorted(remaining)))
    active_jobs, running_units, schema = _reuse_database_state(context)
    if active_jobs or running_units:
        raise RuntimeError("durable work became active before release backup")
    if schema != expected_schema:
        raise RuntimeError("database schema changed before release backup")


def _restore_release_admission(context: ComposeContext, stopped_services: list[str]) -> None:
    # Reverse the stop order: API first, scheduler next, external ingress last.
    for service in reversed(stopped_services):
        context.run("start", service)
        if service not in context.running_services():
            raise RuntimeError(f"release admission service did not restart: {service}")


def _data_usage_bytes(context: ComposeContext, source: Path) -> int:
    raw = context.docker(
        "run",
        "--rm",
        "--volume",
        f"{source}:/source:ro",
        "postgres:16-alpine",
        "du",
        "-sk",
        "/source",
        capture=True,
    )
    return int(raw.splitlines()[-1].split()[0]) * 1024


def _assert_no_data_file_newer(data_source: Path, created_at: datetime) -> None:
    cutoff_ns = int(created_at.timestamp() * 1_000_000_000)

    def fail(error: OSError) -> None:
        raise error

    for directory, _subdirectories, filenames in os.walk(
        data_source, followlinks=False, onerror=fail
    ):
        for filename in filenames:
            candidate = Path(directory, filename)
            if candidate.stat(follow_symlinks=False).st_mtime_ns > cutoff_ns:
                raise RuntimeError(
                    "governed data contains a file newer than the reusable backup: "
                    f"{candidate}"
                )


def _verify_reuse_contract_source(
    contract: RollbackComposeContract,
    backup_directory: Path,
) -> None:
    contract_root = (backup_directory / "rollback-compose-contract").resolve()
    if not contract_root.is_dir():
        raise RuntimeError("reusable backup has no rollback Compose contract")
    if any(not _inside_path(source, contract_root) for source in contract.compose_sources):
        raise RuntimeError(
            "running services are not owned by the reusable backup rollback contract"
        )
    manifest_path = contract_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("reusable backup rollback contract is invalid") from exc
    env_snapshot = str(manifest.get("env_snapshot") or "").strip()
    if env_snapshot:
        # Accept already-created legacy rollback contracts, while new releases
        # never copy the plaintext environment into the backup.
        env_target = (contract_root / env_snapshot).resolve()
        if env_target != contract.env_source.resolve():
            raise RuntimeError("reusable backup rollback environment does not match")
    else:
        try:
            expected_env = Path(str(manifest["env_source"])).expanduser().resolve()
        except (KeyError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                "reusable backup rollback environment source is invalid"
            ) from exc
        if expected_env != contract.env_source.resolve():
            raise RuntimeError("reusable backup rollback environment does not match")
    if (
        contract.env_source.is_symlink()
        or not contract.env_source.is_file()
        or contract.env_source.read_bytes() != contract.env_content
    ):
        raise RuntimeError("reusable backup rollback environment changed after capture")
    if hashlib.sha256(contract.env_content).hexdigest() != manifest.get("env_sha256"):
        raise RuntimeError("reusable backup rollback environment checksum mismatch")
    entries = manifest.get("compose_files") or []
    targets = tuple(
        (contract_root / str(item.get("snapshot") or "")).resolve()
        for item in entries
        if isinstance(item, dict)
    )
    actual_targets = tuple(item.resolve() for item in contract.compose_sources)
    if len(entries) != len(targets) or actual_targets[: len(targets)] != targets:
        raise RuntimeError("reusable backup rollback Compose files do not match")
    extras = actual_targets[len(targets) :]
    if any(
        item.parent != contract_root or item.name != "rollback.override.json"
        for item in extras
    ):
        raise RuntimeError("reusable backup has an unexpected rollback Compose override")
    for item, content in zip(
        entries,
        contract.compose_contents[: len(entries)],
        strict=True,
    ):
        if hashlib.sha256(content).hexdigest() != item.get("sha256"):
            raise RuntimeError("reusable backup rollback Compose checksum mismatch")


def _validate_reusable_backup(
    context: ComposeContext,
    backup_root: Path,
    backup_directory: Path,
    contract: RollbackComposeContract,
) -> ReusableBackup:
    root = backup_root.resolve()
    if backup_directory.is_symlink():
        raise ValueError("reusable backup must not be a symlink")
    candidate = backup_directory.resolve()
    if candidate == root or not _inside_path(candidate, root):
        raise ValueError("reusable backup must be a real directory inside backup_root")
    manifest = load_and_verify_manifest(
        candidate,
        use_verification_receipt=True,
    )
    if manifest.get("format_version") != FULL_BACKUP_FORMAT_VERSION:
        raise ValueError("only a full v1 backup can be reused for an exact release snapshot")
    if manifest.get("project_name") != context.project_name:
        raise ValueError("reusable backup belongs to another Compose project")
    expected_secret = str(manifest.get("platform_secret_key_sha256") or "")
    if not expected_secret or _platform_secret_key_fingerprint(context) != expected_secret:
        raise ValueError("reusable backup platform secret does not match")
    active_jobs, running_units, revision = _reuse_database_state(context)
    if active_jobs or running_units:
        raise RuntimeError("reusable backup requires an idle durable queue")
    if manifest.get("schema_revision") != revision:
        raise ValueError("reusable backup schema does not match the current database")
    _verify_reuse_contract_source(contract, candidate)
    created_at = datetime.fromisoformat(str(manifest.get("created_at") or ""))
    if created_at.tzinfo is None:
        raise ValueError("reusable backup creation time must include a timezone")
    source_raw = context.data_volume()
    data_source = Path(source_raw)
    configured_source = Path(
        _effective_env_value(context, "QUANTLAB_DATA_HOST_PATH") or "/data/quantlab"
    )
    if not data_source.is_absolute() or data_source.resolve() != configured_source.resolve():
        raise RuntimeError("reusable backup requires the governed absolute data bind mount")
    _assert_no_data_file_newer(data_source.resolve(), created_at)
    expected_bytes = int(manifest["data_volume"]["uncompressed_bytes"])
    observed_bytes = _data_usage_bytes(context, data_source.resolve())
    tolerance = 1024**2
    if abs(observed_bytes - expected_bytes) > tolerance:
        raise RuntimeError("governed data size does not match the reusable backup")
    return ReusableBackup(
        directory=candidate,
        manifest=manifest,
        data_source=data_source.resolve(),
        expected_data_bytes=expected_bytes,
        observed_data_bytes=observed_bytes,
        size_tolerance_bytes=tolerance,
    )


def _stop_writers_for_backup_reuse(
    context: ComposeContext,
    *,
    expected_schema_revision: str,
) -> tuple[str, ...]:
    running = set(context.running_services())
    stopped = tuple(service for service in WRITER_SERVICES if service in running)
    try:
        if stopped:
            context.run("stop", *stopped)
        remaining = set(context.running_services()).intersection(WRITER_SERVICES)
        if remaining:
            raise RuntimeError(
                "writer services remained active during backup reuse: "
                + ", ".join(sorted(remaining))
            )
        active_jobs, running_units, revision = _reuse_database_state(context)
        if active_jobs or running_units:
            raise RuntimeError("durable work became active while writers were stopping")
        if revision != expected_schema_revision:
            raise RuntimeError("database schema changed while writers were stopping")
    except Exception:
        if stopped:
            context.run("start", *stopped, check=False)
        raise
    return stopped


def _registry_port(context: ComposeContext) -> int:
    raw = _effective_env_value(context, "RDAGENT_REGISTRY_PORT")
    port = int(raw or _REGISTRY_PORT)
    if not 1024 <= port <= 65535:
        raise ValueError("RDAGENT_REGISTRY_PORT must be between 1024 and 65535")
    return port


def _atomic_replace(path: Path, content: bytes) -> None:
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
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _environment_assignment_key(line: str) -> str | None:
    candidate = line.lstrip()
    if not candidate or candidate.startswith("#"):
        return None
    if candidate.startswith("export "):
        candidate = candidate.removeprefix("export ").lstrip()
    if "=" not in candidate:
        return None
    return candidate.split("=", 1)[0].strip() or None


def _update_environment(path: Path, values: dict[str, str]) -> None:
    original = path.read_text(encoding="utf-8-sig")
    output: list[str] = []
    for line in original.splitlines():
        if _environment_assignment_key(line) not in values:
            output.append(line)
    output.extend(f"{key}={value}" for key, value in sorted(values.items()))
    _atomic_replace(path, ("\n".join(output) + "\n").encode("utf-8"))


def _release_configuration_digest(context: ComposeContext) -> str:
    """Hash the release configuration without recursively hashing its stamp."""

    environment_lines: list[str] = []
    for line in context.env_file.read_text(encoding="utf-8-sig").splitlines():
        key = _environment_assignment_key(line)
        if key not in RELEASE_IDENTITY_ENV_KEYS:
            environment_lines.append(line)
    compose_files = tuple(
        Path(item).resolve() for item in getattr(context, "compose_files", ())
    )
    identity = {
        "contract_version": "quantlab-release-config-v1",
        "project_name": context.project_name,
        "profiles": sorted(str(item) for item in getattr(context, "profiles", ())),
        "environment_sha256": hashlib.sha256(
            ("\n".join(environment_lines) + "\n").encode("utf-8")
        ).hexdigest(),
        "compose": [
            {
                "name": item.name,
                "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
            }
            for item in compose_files
        ],
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stamp_release_identity(
    context: ComposeContext,
    release_id: str,
) -> dict[str, str]:
    identity = release_identity_environment(
        release_id.lower(),
        _release_configuration_digest(context),
    )
    _update_environment(context.env_file, identity)
    return identity


def _repo_digest(raw: str, repository: str) -> str:
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Docker image repository digests are invalid") from exc
    for value in values or []:
        item = str(value).strip().lower()
        prefix = repository.lower() + "@"
        if item.startswith(prefix) and _IMAGE_ID.fullmatch(item[len(prefix) :]):
            return item[len(prefix) :]
    raise RuntimeError("Docker image has no digest for the local release registry")


def _wait_for_registry(port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v2/"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("local release registry did not become ready")


def _governed_sandbox_evidence(project_root: Path) -> dict[str, dict[str, str]]:
    evidence: dict[str, dict[str, str]] = {}
    for name, directory in _GOVERNED_SANDBOX_CONTEXTS.items():
        dockerfile = project_root.resolve() / "deploy" / directory / "Dockerfile"
        source = dockerfile.read_bytes()
        text = source.decode("utf-8")
        forbidden = (
            "git clone",
            "git fetch",
            "git reset",
            "apt-get",
            " pip ",
            "pip install",
            "curl ",
            "wget ",
            ":latest",
        )
        if any(token in text for token in forbidden):
            raise RuntimeError(f"governed {name} sandbox contains a networked build operation")
        evidence[name] = {
            "dockerfile_sha256": hashlib.sha256(source).hexdigest(),
            "build_daemon": "host",
        }
    if _QLIB_COMMIT not in (
        project_root.resolve() / "deploy" / "qlib-sandbox" / "Dockerfile"
    ).read_text(encoding="utf-8"):
        raise RuntimeError("governed Qlib sandbox does not assert the platform Qlib commit")
    evidence["qlib"]["qlib_commit"] = _QLIB_COMMIT
    evidence["qlib"]["source_path"] = "/opt/qlib"
    return evidence


def _build_and_publish_host_image(
    context: ComposeContext,
    *,
    context_root: Path,
    host_image_tag: str,
    host_repository: str,
    dind_repository: str,
    timeout: int,
    build_args: tuple[str, ...] = (),
) -> str:
    command = ["build", "--network", "none"]
    for argument in build_args:
        command.extend(("--build-arg", argument))
    command.extend(("--tag", host_image_tag, str(context_root.resolve())))
    context.docker(*command, timeout=timeout)
    context.docker("push", host_image_tag, timeout=timeout)
    digests_raw = context.docker(
        "image",
        "inspect",
        "--format",
        "{{json .RepoDigests}}",
        host_image_tag,
        capture=True,
    )
    digest = _repo_digest(digests_raw, host_repository)
    image = f"{dind_repository}@{digest}"
    context.run(
        "exec",
        "-T",
        "rdagent-docker",
        "docker",
        "pull",
        image,
        timeout=timeout,
    )
    return image


def _dind_smoke(
    context: ComposeContext,
    image: str,
    command: tuple[str, ...],
    *,
    timeout: int,
    volumes: tuple[str, ...] = (),
) -> None:
    arguments = [
        "exec",
        "-T",
        "rdagent-docker",
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
    ]
    for volume in volumes:
        arguments.extend(("--volume", volume))
    arguments.extend((image, *command))
    context.run(
        *arguments,
        timeout=timeout,
    )


def _qlib_provider_candidates(calendars: list[str]) -> list[str]:
    """Return newest-first provider roots discovered inside the DinD data mount."""

    return [
        item.strip().removesuffix("/calendars/day.txt")
        for item in calendars
        if item.strip().endswith("/calendars/day.txt")
    ]


def _configured_service_images(
    context: ComposeContext,
    *service_names: str,
) -> dict[str, str]:
    """Resolve service images from the final Compose model used for deployment."""

    try:
        payload = json.loads(
            context.run("config", "--format", "json", capture=True)
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("rendered Compose configuration is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("rendered Compose configuration must be a JSON object")
    services = payload.get("services")
    if not isinstance(services, dict):
        raise RuntimeError("rendered Compose configuration has no services object")

    resolved: dict[str, str] = {}
    for service_name in service_names:
        service = services.get(service_name)
        if not isinstance(service, dict):
            raise RuntimeError(
                f"rendered Compose configuration has no {service_name!r} service"
            )
        image = service.get("image")
        if (
            not isinstance(image, str)
            or not image
            or image != image.strip()
            or any(character.isspace() for character in image)
        ):
            raise RuntimeError(
                f"rendered Compose service {service_name!r} has no valid image"
            )
        resolved[service_name] = image
    return resolved


def _representative_build_services(
    configured_images: dict[str, str],
    services: tuple[str, ...],
) -> tuple[str, ...]:
    """Select one deterministic Compose build target for each mutable image alias."""

    missing = [service for service in services if service not in configured_images]
    if missing:
        raise RuntimeError(
            "configured build images are missing services: " + ", ".join(missing)
        )
    selected: list[str] = []
    seen_images: set[str] = set()
    for service in services:
        image = configured_images[service]
        if image in seen_images:
            continue
        seen_images.add(image)
        selected.append(service)
    return tuple(selected)


def _prepare_sandbox_images(
    context: ComposeContext,
    project_root: Path,
    release_id: str,
    *,
    wait_timeout: int,
    preserved_model_sandbox: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal locally built sandboxes in a loopback-only deployment registry.

    The registry is used only while publishing. Runtime jobs are offline and
    consume the digest-pinned images already present in the dedicated DinD
    daemon. Registry data and DinD layers live on the data disk via Compose.
    """

    if preserved_model_sandbox is not None:
        proof = preserved_model_sandbox
        if (
            proof.get("contract_version") != "model-sandbox-reuse-v1"
            or proof.get("status") != "verified"
            or proof.get("live_worker_sources_verified") is not True
            or proof.get("sandbox_installed_sources_verified") is not True
            or proof.get("sources") != _model_reuse_source_inventory(project_root)
        ):
            raise RuntimeError("model sandbox preservation proof is missing or changed")
        _assert_model_reuse_image(context, proof["image"], proof["image_id"])
        _verify_model_reuse_runtime(context, proof["sources"], proof["image"], live_worker=False)
    port = _registry_port(context)
    host_registry = f"127.0.0.1:{port}"
    dind_registry = "rdagent-registry:5000"
    release_tag = release_id.lower()
    configured_images = _configured_service_images(
        context,
        "worker",
        "rdagent-worker",
    )
    runtime_source = configured_images["rdagent-worker"]
    runtime_image_id = context.docker(
        "image",
        "inspect",
        "--format",
        "{{.Id}}",
        runtime_source,
        capture=True,
    ).splitlines()[0].strip().lower()
    if not _IMAGE_ID.fullmatch(runtime_image_id):
        raise RuntimeError("built RD-Agent runtime has no immutable image ID")

    host_base_repository = f"{host_registry}/quantlab/worker-sandbox-base"
    host_base_tag = f"{host_base_repository}:{release_tag}"
    host_qlib_repository = f"{host_registry}/quantlab/qlib-sandbox"
    host_model_repository = f"{host_registry}/quantlab/model-sandbox"
    host_qlib_tag = f"{host_qlib_repository}:{release_tag}"
    host_model_tag = f"{host_model_repository}:{release_tag}"
    host_published_tags = (
        host_base_tag,
        host_qlib_tag,
        host_model_tag,
    )
    dind_base_repository = f"{dind_registry}/quantlab/worker-sandbox-base"
    dind_qlib_repository = f"{dind_registry}/quantlab/qlib-sandbox"
    dind_model_repository = f"{dind_registry}/quantlab/model-sandbox"
    image_timeout = max(wait_timeout, 3600)

    context.run(
        "--profile",
        "sandbox-registry",
        "up",
        "-d",
        "rdagent-registry",
    )
    try:
        _wait_for_registry(port, min(wait_timeout, 120))
        sandbox_base_source = configured_images["worker"]
        sandbox_base_image_id = context.docker(
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            sandbox_base_source,
            capture=True,
        ).splitlines()[0].strip().lower()
        if not _IMAGE_ID.fullmatch(sandbox_base_image_id):
            raise RuntimeError("built sandbox base has no immutable image ID")
        context.docker("tag", sandbox_base_image_id, host_base_tag)
        context.docker("push", host_base_tag, capture=True, timeout=image_timeout)
        base_digests_raw = context.docker(
            "image",
            "inspect",
            "--format",
            "{{json .RepoDigests}}",
            host_base_tag,
            capture=True,
        )
        base_digest = _repo_digest(base_digests_raw, host_base_repository)
        base_image = f"{dind_base_repository}@{base_digest}"
        context.run(
            "exec",
            "-T",
            "rdagent-docker",
            "docker",
            "pull",
            base_image,
            timeout=image_timeout,
        )
        source_evidence = _governed_sandbox_evidence(project_root)
        host_base_image = f"{host_base_repository}@{base_digest}"
        qlib_image = _build_and_publish_host_image(
            context,
            context_root=project_root.resolve() / "deploy" / "qlib-sandbox",
            host_image_tag=host_qlib_tag,
            host_repository=host_qlib_repository,
            dind_repository=dind_qlib_repository,
            timeout=image_timeout,
            build_args=(f"QLIB_SANDBOX_BASE_IMAGE={host_base_image}",),
        )
        model_image = (
            str(preserved_model_sandbox["image"])
            if preserved_model_sandbox is not None else _build_and_publish_host_image(
                context,
                context_root=project_root.resolve() / "deploy" / "model-sandbox",
                host_image_tag=host_model_tag,
                host_repository=host_model_repository,
                dind_repository=dind_model_repository,
                timeout=image_timeout,
                build_args=(f"MODEL_SANDBOX_BASE_IMAGE={host_base_image}",),
            )
        )

        qlib_calendars = context.run(
            "exec",
            "-T",
            "rdagent-docker",
            "sh",
            "-lc",
            "find /data/qlib -path '*/calendars/day.txt' -type f | sort -r",
            capture=True,
        ).splitlines()
        qlib_homes = _qlib_provider_candidates(qlib_calendars)
        if not qlib_homes:
            raise RuntimeError("no Qlib dataset is available for sandbox smoke testing")
        qlib_smoke_home = ""
        for qlib_home in qlib_homes:
            try:
                _dind_smoke(
                    context,
                    qlib_image,
                    (
                        "python",
                        "-c",
                        "import catboost, numpy as np, qlib, tables, torch, xgboost; "
                        "assert catboost and tables and xgboost; "
                        "assert torch.from_numpy(np.arange(3)).numpy().tolist() "
                        "== [0, 1, 2]; "
                        "qlib.init(provider_uri='/qlib'); "
                        "from qlib.data import D; calendar=D.calendar(freq='day'); "
                        "assert len(calendar) > 1; "
                        "features=D.features(D.instruments('cn_all'), ['$close'], "
                        "start_time=calendar[-2], end_time=calendar[-1], freq='day'); "
                        "assert not features.empty",
                    ),
                    timeout=image_timeout,
                    volumes=(f"{qlib_home}:/qlib:ro",),
                )
            except Exception:
                continue
            qlib_smoke_home = qlib_home
            break
        if not qlib_smoke_home:
            raise RuntimeError("no available Qlib dataset passed the sandbox smoke test")
        _dind_smoke(
            context,
            qlib_image,
            (
                "sh",
                "-lc",
                f'test "$QLIB_COMMIT" = "{_QLIB_COMMIT}"; '
                'test "$(readlink -f /workspace/qlib)" = "/opt/qlib"',
            ),
            timeout=image_timeout,
        )
        _dind_smoke(
            context,
            model_image,
            (
                "python",
                "-c",
                "import lightgbm, numpy as np, qlib, torch; "
                "assert lightgbm and qlib; "
                "assert torch.from_numpy(np.arange(3)).numpy().tolist() "
                "== [0, 1, 2]",
            ),
            timeout=image_timeout,
        )
        if preserved_model_sandbox is not None:
            _assert_model_reuse_image(context, model_image, preserved_model_sandbox["image_id"])
        sealed = {
            "RDAGENT_RUNTIME_IMAGE_DIGEST": runtime_image_id,
            "QUANTLAB_WORKER_RUNTIME_IMAGE_DIGEST": sandbox_base_image_id,
            "RDAGENT_QLIB_SANDBOX_IMAGE": qlib_image,
            "MODEL_SANDBOX_IMAGE": model_image,
        }
        _update_environment(context.env_file, sealed)
        return {
            **sealed,
            "source_evidence": source_evidence,
            "runtime_base_image_id": runtime_image_id,
            "sandbox_base_image_id": sandbox_base_image_id,
            "qlib_smoke_dataset": qlib_smoke_home,
            "smoke_passed": True,
            "model_sandbox_reuse": preserved_model_sandbox,
        }
    finally:
        context.run(
            "--profile",
            "sandbox-registry",
            "stop",
            "rdagent-registry",
            check=False,
        )
        # Runtime jobs use the independently pulled digest inside DinD.  The
        # loopback publishing tags are no longer executable dependencies once
        # sealing finishes; keeping them would pin one large host image set per
        # drill/release after its temporary registry is gone.
        context.docker("image", "rm", "-f", *host_published_tags, check=False)


@control_plane_locked
def prepare_drill_sandbox_bootstrap(
    context: ComposeContext,
    project_root: Path,
    release_id: str,
    *,
    wait_timeout: int,
) -> dict[str, Any]:
    """Give a blank upgrade drill the same sealed sandbox contract as production.

    An ordinary release inherits digest-pinned sandboxes from its predecessor.
    A scratch drill has no predecessor, while the evaluation worker correctly
    refuses to become healthy without a preloaded model sandbox.  Start only the
    isolated DinD (and its drill-data seed dependency), then run the governed
    production sealing path before any worker starts.  No health or worker queue
    contract is relaxed.
    """

    context.run(
        "up",
        "-d",
        "--no-build",
        "--wait",
        "--wait-timeout",
        str(wait_timeout),
        "rdagent-docker",
    )
    return _prepare_sandbox_images(
        context,
        project_root,
        f"drill-bootstrap-{release_id}",
        wait_timeout=wait_timeout,
    )


def _existing_storage_anchor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(path)
        candidate = parent
    return candidate


def _assess_backup_capacity(
    context: ComposeContext,
    backup_root: Path,
    *,
    minimum_free_gb: float,
    format_version: int = CONTROL_PLANE_BACKUP_FORMAT_VERSION,
) -> dict[str, Any]:
    """Fail closed unless the target can hold the selected backup and headroom."""

    if format_version == CONTROL_PLANE_BACKUP_FORMAT_VERSION:
        control_plane = assess_control_plane_backup_capacity(
            context,
            backup_root,
            minimum_free_gb=minimum_free_gb,
        )
        return {
            **control_plane,
            "id": "backup_capacity",
            "title": "Control-plane backup target capacity",
            "remediation": (
                None
                if control_plane["status"] == "pass"
                else "Choose a backup root with space for the database snapshot, "
                "bounded control archive, and release headroom."
            ),
        }
    try:
        if minimum_free_gb < 0:
            raise ValueError("minimum_free_gb must not be negative")
        anchor = _existing_storage_anchor(backup_root)
        free_bytes = shutil.disk_usage(anchor).free
        if format_version == FULL_BACKUP_FORMAT_VERSION:
            source = context.data_volume()
            raw = context.docker(
                "run",
                "--rm",
                "--volume",
                f"{source}:/source:ro",
                "postgres:16-alpine",
                "du",
                "-sk",
                "/source",
                capture=True,
            )
            source_kib = int(raw.splitlines()[-1].split()[0])
            payload_upper_bound = source_kib * 1024
            payload_label = (
                f"full /data upper bound {payload_upper_bound / _GIB:.1f} GiB"
            )
        else:
            raise ValueError("unsupported backup format")
        required_bytes = payload_upper_bound + int(minimum_free_gb * _GIB)
        passed = free_bytes >= required_bytes
        evidence = (
            f"format v{format_version}; target {backup_root.resolve()}; "
            f"free {free_bytes / _GIB:.1f} GiB; {payload_label}; retained headroom "
            f"{minimum_free_gb:.1f} GiB; required {required_bytes / _GIB:.1f} GiB"
        )
    except Exception as exc:
        passed = False
        evidence = f"backup capacity could not be measured: {type(exc).__name__}: {exc}"
    return {
        "id": "backup_capacity",
        "title": "Coordinated backup target capacity",
        "status": "pass" if passed else "block",
        "evidence": evidence,
        "remediation": (
            None
            if passed
            else "Choose a backup root with space for the selected backup payload "
            "plus release headroom."
        ),
    }


def _capture_rollback_images(
    context: ComposeContext,
    release_id: str,
    *,
    services: tuple[str, ...] = BUILT_SERVICES,
    allow_missing: frozenset[str] = frozenset(),
    repository: str = "quantlab-rollback",
) -> dict[str, str]:
    tags: dict[str, str] = {}
    for service in services:
        container_id = context.container_id(service)
        if not container_id:
            if service in allow_missing:
                continue
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
        tag = f"{repository}:{release_id.lower()}-{service}"
        context.docker("tag", image_id, tag)
        tags[service] = tag
    return tags


def _restore_built_image_aliases(
    context: ComposeContext,
    configured_images: dict[str, str],
    rollback_tags: dict[str, str],
) -> list[str]:
    """Restore mutable Compose image aliases after an uncommitted build.

    Compose builds may retag ``image:`` references before any container or
    database mutation happens.  A release that then blocks must put those
    aliases back, otherwise a later operator command could start unaccepted
    code even though this release reported ``blocked``.
    """

    targets: dict[str, tuple[str, str]] = {}
    for service, image in configured_images.items():
        rollback_tag = rollback_tags.get(service)
        if rollback_tag is None or "@sha256:" in image:
            continue
        raw = context.docker(
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            rollback_tag,
            capture=True,
        ).splitlines()
        image_id = raw[0].strip().lower() if raw else ""
        if not _IMAGE_ID.fullmatch(image_id):
            raise RuntimeError(f"rollback image for {service} has no immutable image ID")
        previous = targets.get(image)
        if previous is not None and previous[1] != image_id:
            raise RuntimeError(
                f"shared Compose image {image!r} had inconsistent rollback images"
            )
        targets[image] = (rollback_tag, image_id)

    restored: list[str] = []
    for image, (rollback_tag, expected_id) in sorted(targets.items()):
        context.docker("tag", rollback_tag, image)
        raw = context.docker(
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            image,
            capture=True,
        ).splitlines()
        actual_id = raw[0].strip().lower() if raw else ""
        if actual_id != expected_id:
            raise RuntimeError(f"failed to restore mutable Compose image alias {image!r}")
        restored.append(image)
    return restored


def _capture_service_storage(
    context: ComposeContext,
    service: str,
    destination: str,
) -> dict[str, str] | None:
    container_id = context.container_id(service)
    if not container_id:
        return None
    inspection = json.loads(context.docker("inspect", container_id, capture=True))[0]
    for mount in inspection.get("Mounts", []):
        if str(mount.get("Destination") or "") != destination:
            continue
        mount_type = str(mount.get("Type") or "")
        source = str(
            mount.get("Name") if mount_type == "volume" else mount.get("Source")
        ).strip()
        if mount_type not in {"bind", "volume"} or not source:
            raise RuntimeError(f"unsupported rollback storage for {service}:{destination}")
        return {"type": mount_type, "source": source, "target": destination}
    return None


def _rollback_override(
    path: Path,
    tags: dict[str, str],
    disabled_services: frozenset[str] = frozenset(),
    rdagent_docker_storage: dict[str, str] | None = None,
) -> None:
    services: dict[str, Any] = {
        service: {"image": tag} for service, tag in sorted(tags.items())
    }
    worker_image = tags.get("worker")
    dind_image = tags.get("rdagent-docker")
    if worker_image or dind_image:
        factor_builder = services.setdefault("factor-sandbox-builder", {})
    if worker_image:
        factor_builder.setdefault("environment", {})[
            "FACTOR_SANDBOX_BASE_IMAGE"
        ] = worker_image
    if dind_image:
        factor_builder["image"] = dind_image
    for service in sorted(disabled_services):
        services[service] = {"profiles": ["rollback-disabled"]}
    payload: dict[str, Any] = {"services": services}
    if rdagent_docker_storage is not None:
        storage = dict(rdagent_docker_storage)
        if storage["type"] == "volume":
            source = "rdagent_docker_rollback"
            payload["volumes"] = {
                source: {"external": True, "name": storage["source"]}
            }
        else:
            source = storage["source"]
        services.setdefault("rdagent-docker", {})["volumes"] = [
            {
                "type": storage["type"],
                "source": source,
                "target": storage["target"],
            }
        ]
    path.write_text(
        json.dumps(
            payload,
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
    *,
    disabled_services: frozenset[str] = frozenset(),
) -> ComposeContext:
    # Compose appends sequence values such as ``profiles`` across files. A
    # rollback override therefore cannot remove the base service's ``gpu``
    # profile by assigning a different one. Disable a profile at the CLI level
    # when every service introduced by that profile is absent from the prior
    # release.
    disabled_profiles = {
        profile
        for profile, services in PROFILE_BUILT_SERVICES.items()
        if set(services).issubset(disabled_services)
    }
    return ComposeContext(
        project_name=context.project_name,
        env_file=context.env_file,
        compose_files=(*context.compose_files, override_file.resolve()),
        profiles=tuple(
            profile for profile in context.profiles if profile not in disabled_profiles
        ),
        project_directory=context.project_directory,
    )


def _stop_and_verify_new_writers(context: ComposeContext) -> None:
    context.run("stop", *WRITER_SERVICES)
    active = {
        item.strip()
        for item in context.docker(
            "ps",
            "--filter",
            f"label=com.docker.compose.project={context.project_name}",
            "--format",
            '{{.Label "com.docker.compose.service"}}',
            capture=True,
        ).splitlines()
        if item.strip()
    }
    remaining = sorted(active.intersection(WRITER_SERVICES))
    if remaining:
        raise RuntimeError(
            "new release writers are still running; rollback restore was not started: "
            + ", ".join(remaining)
        )


def _restore_previous_release(
    context: ComposeContext,
    rollback_base_context: ComposeContext,
    backup_directory: Path,
    rollback_tags: dict[str, str],
    *,
    wait_timeout: int,
    minimum_free_gb: float = 1.0,
    disabled_services: frozenset[str] = frozenset(),
    rdagent_docker_storage: dict[str, str] | None = None,
) -> dict[str, Any]:
    # Stop using the new contract first so newly introduced writers cannot be
    # omitted by the previous release's service list.  This must happen before
    # even the cached backup check: a cache miss may scan a very large archive,
    # and no new writer may remain live during that window.
    _stop_and_verify_new_writers(context)
    manifest = load_and_verify_manifest(
        backup_directory,
        use_verification_receipt=True,
    )
    override_file = (
        rollback_base_context.compose_files[0].parent / "rollback.override.json"
    ).resolve()
    _rollback_override(
        override_file,
        rollback_tags,
        disabled_services=disabled_services,
        rdagent_docker_storage=rdagent_docker_storage,
    )
    rollback = _rollback_context(
        rollback_base_context,
        override_file,
        disabled_services=disabled_services,
    )
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
        minimum_free_gb=minimum_free_gb,
        use_verification_receipt=True,
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
        "compose_override": str(override_file),
    }


def _record_cutover(context: ComposeContext) -> str:
    """Seal the last idle instant before any new release writer starts."""

    remaining = sorted(
        set(context.running_services()).intersection(WRITER_SERVICES)
    )
    if remaining:
        raise RuntimeError(
            "cannot record release cutover while writer services are running: "
            + ", ".join(remaining)
        )
    state_query = (
        "SELECT "
        "(SELECT count(*) FROM quantlab.jobs "
        "WHERE status IN ('queued','running')) || '|' || "
        "(SELECT count(*) FROM quantlab.work_units "
        "WHERE status='running');"
    )
    state_raw = context.run(
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "quantlab",
        "-d",
        "quantlab",
        "-Atc",
        state_query,
        capture=True,
    ).splitlines()
    try:
        active_jobs, running_units = state_raw[-1].split("|", 1)
        if int(active_jobs) or int(running_units):
            raise RuntimeError(
                "durable work became active before the release cutover was sealed"
            )
    except RuntimeError:
        raise
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("release cutover database state is unreadable") from exc
    # Use a second statement so the database timestamp is observably later
    # than the idle confirmation. Any out-of-band row inserted between these
    # statements is therefore pre-cutover and will fail final acceptance.
    clock_raw = context.run(
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "quantlab",
        "-d",
        "quantlab",
        "-Atc",
        "SELECT to_char(clock_timestamp() AT TIME ZONE 'UTC', "
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"');",
        capture=True,
    ).splitlines()
    try:
        cutover = datetime.fromisoformat(clock_raw[-1].replace("Z", "+00:00"))
        if cutover.tzinfo is None:
            raise ValueError("cutover timestamp has no timezone")
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("release cutover database clock is unreadable") from exc
    return cutover.astimezone(UTC).isoformat(timespec="microseconds")


def _post_cutover_durable_state(
    context: ComposeContext,
    cutover_at: str,
) -> dict[str, int]:
    """Classify active work by the database-clock cutover boundary."""

    try:
        cutover = datetime.fromisoformat(cutover_at)
        if cutover.tzinfo is None:
            raise ValueError("cutover timestamp has no timezone")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("release cutover timestamp is invalid") from exc
    canonical = cutover.astimezone(UTC).isoformat(timespec="microseconds")
    query = (
        "SELECT "
        "(SELECT count(*) FROM quantlab.jobs "
        "WHERE status IN ('queued','running')) || '|' || "
        "(SELECT count(*) FROM quantlab.jobs "
        "WHERE status IN ('queued','running') "
        f"AND created_at < TIMESTAMPTZ '{canonical}') || '|' || "
        "(SELECT count(*) FROM quantlab.work_units "
        "WHERE status='running') || '|' || "
        "(SELECT count(*) FROM quantlab.work_units "
        "WHERE status='running' "
        "AND GREATEST(created_at, updated_at) "
        f"< TIMESTAMPTZ '{canonical}');"
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
    ).splitlines()
    try:
        values = [int(item) for item in raw[-1].split("|")]
        if len(values) != 4 or any(item < 0 for item in values):
            raise ValueError("invalid durable-work counts")
        active_jobs, pre_cutover_jobs, running_units, pre_cutover_units = values
        if pre_cutover_jobs > active_jobs or pre_cutover_units > running_units:
            raise ValueError("invalid pre-cutover durable-work counts")
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("post-cutover durable-work state is unreadable") from exc
    return {
        "active_jobs": active_jobs,
        "pre_cutover_active_jobs": pre_cutover_jobs,
        "post_cutover_active_jobs": active_jobs - pre_cutover_jobs,
        "running_units": running_units,
        "pre_cutover_running_units": pre_cutover_units,
        "post_cutover_running_units": running_units - pre_cutover_units,
    }


def _post_start_acceptance(
    assessment: dict[str, Any],
    durable_state: dict[str, int],
) -> dict[str, Any]:
    """Evaluate the release after job-producing services have started.

    The two pre-mutation gates require the durable queue to be idle.  Once the
    scheduler and workers are healthy, however, they may immediately enqueue
    or claim scheduled work. Only rows created or claimed at/after the sealed
    database-clock cutover are tolerated. Every earlier row and every other
    preflight blocker still fails closed.
    """

    checks = assessment.get("checks")
    if not isinstance(checks, list):
        return {
            "status": "block",
            "ignored_blockers": [],
            "blocking_checks": ["invalid_assessment"],
            "durable_state": durable_state,
            "evidence": "post-start preflight did not return a check list",
        }
    blockers = sorted(
        {
            str(check.get("id") or "unknown")
            for check in checks
            if isinstance(check, dict) and check.get("status") == "block"
        }
    )
    required_durable_keys = {
        "active_jobs",
        "pre_cutover_active_jobs",
        "post_cutover_active_jobs",
        "running_units",
        "pre_cutover_running_units",
        "post_cutover_running_units",
    }
    durable_state_valid = (
        required_durable_keys.issubset(durable_state)
        and all(
            isinstance(durable_state[key], int) and durable_state[key] >= 0
            for key in required_durable_keys
        )
        and durable_state["active_jobs"]
        == durable_state["pre_cutover_active_jobs"]
        + durable_state["post_cutover_active_jobs"]
        and durable_state["running_units"]
        == durable_state["pre_cutover_running_units"]
        + durable_state["post_cutover_running_units"]
    )
    new_work_only = (
        durable_state_valid
        and durable_state["pre_cutover_active_jobs"] == 0
        and durable_state["pre_cutover_running_units"] == 0
    )
    ignored = (
        ["durable_work_idle"]
        if "durable_work_idle" in blockers and new_work_only
        else []
    )
    blocking = [check_id for check_id in blockers if check_id not in ignored]
    if not durable_state_valid:
        blocking.append("invalid_durable_work_state")
    elif not new_work_only:
        blocking.append("pre_cutover_durable_work")
    blocking = sorted(set(blocking))
    assessment_status = str(assessment.get("status") or "")
    status_consistent = assessment_status == ("blocked" if blockers else "ready")
    accepted = (
        status_consistent
        and not blocking
        and assessment.get("migration_state") == "current"
    )
    if accepted and ignored:
        evidence = "all release checks passed; only post-cutover durable work is active"
    elif accepted:
        evidence = "all release checks passed"
    elif not durable_state_valid:
        evidence = "post-cutover durable-work provenance is invalid"
    elif not new_work_only:
        evidence = "pre-cutover durable work is active"
    else:
        evidence = "post-start release health, image, schema, or service checks failed"
    return {
        "status": "pass" if accepted else "block",
        "ignored_blockers": ignored,
        "blocking_checks": blocking,
        "durable_state": durable_state,
        "evidence": evidence,
    }


def _gateway_smoke(context: ComposeContext) -> dict[str, Any]:
    """Verify the externally exposed gateway only after release acceptance."""

    container_id = context.container_id("gateway")
    if not container_id:
        raise RuntimeError("gateway container was not created")
    inspection = json.loads(context.docker("inspect", container_id, capture=True))[0]
    state = inspection.get("State") or {}
    if state.get("Status") != "running" or state.get("Running") is not True:
        raise RuntimeError("gateway container is not running")
    raw = context.run(
        "exec",
        "-T",
        "gateway",
        "wget",
        "-qO-",
        "http://127.0.0.1:8080/api/health",
        capture=True,
        timeout=15,
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("gateway health response is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise RuntimeError("gateway health response is not healthy")
    return {
        "status": "pass",
        "container_id": container_id,
        "api_status": "ok",
    }


def _release_identity_acceptance(
    context: ComposeContext,
    services: set[str] | frozenset[str],
) -> dict[str, str]:
    valid, evidence = _runtime_release_identity(context, services)
    return {
        "status": "pass" if valid else "block",
        "evidence": evidence,
    }


def _switch_stable_release_link(stable_link: Path, project_root: Path) -> dict[str, str]:
    """Atomically point the operational stable path at an accepted release."""

    target = project_root.expanduser().resolve(strict=True)
    if not target.is_dir():
        raise RuntimeError("accepted release target is not a directory")
    link = stable_link.expanduser().absolute()
    if link == target:
        raise RuntimeError("stable release link must not equal the release directory")
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(link) and not link.is_symlink():
        raise RuntimeError("stable release path exists and is not a symbolic link")
    temporary = link.parent / f".{link.name}.next-{os.getpid()}"
    if os.path.lexists(temporary):
        raise RuntimeError("stale stable-release switch path exists")
    try:
        os.symlink(target, temporary, target_is_directory=True)
        os.replace(temporary, link)
        if link.resolve(strict=True) != target:
            raise RuntimeError("stable release link verification failed")
        if os.name == "posix":
            directory_fd = os.open(link.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()
    return {
        "status": "pass",
        "path": str(link),
        "target": str(target),
    }


@control_plane_locked
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
    rollback_tag_repository: str = "quantlab-rollback",
    prune_rollback_images: bool = True,
    reuse_backup: Path | None = None,
    stable_release_link: Path | None = None,
    preserve_model_sandbox_image: str | None = None,
    preserve_model_sandbox_image_id: str | None = None,
) -> dict[str, Any]:
    if not confirmed:
        raise ValueError("release upgrade requires --confirm-upgrade")
    if retention_count < 1:
        raise ValueError("retention_count must be positive")
    if wait_timeout < 30:
        raise ValueError("wait_timeout must be at least 30 seconds")
    if rollback_image_retention < 1:
        raise ValueError("rollback_image_retention must be positive")
    if not re.fullmatch(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", rollback_tag_repository):
        raise ValueError("rollback_tag_repository must be a lowercase Docker repository")
    if bool(preserve_model_sandbox_image) != bool(preserve_model_sandbox_image_id):
        raise ValueError("model sandbox preservation requires both reference and image ID")
    if preserve_model_sandbox_image and (
        stable_release_link is None or not stable_release_link.is_symlink()
    ):
        raise ValueError("model sandbox preservation requires the existing stable release link")

    release_id = _stamp()
    result: dict[str, Any] = {
        "release_id": release_id,
        "project_name": context.project_name,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "failed",
        "live_trading_enabled": False,
        "checks": {},
        "rollback_images": {},
        "rollback_compose_contract": None,
        "rollback_disabled_services": [],
        "rollback_rdagent_docker_storage": None,
        "backup_directory": None,
        "backup_reused": False,
        "cutover_at": None,
    }
    backup_directory: Path | None = None
    rollback_tags: dict[str, str] = {}
    rollback_contract: RollbackComposeContract | None = None
    rollback_base_context: ComposeContext | None = None
    rollback_disabled_services: frozenset[str] = frozenset()
    rollback_rdagent_docker_storage: dict[str, str] | None = None
    environment_before: bytes | None = None
    environment_changed = False
    state_mutated = False
    release_committed = False
    configured_build_images: dict[str, str] = {}
    build_aliases_dirty = False
    admission_stopped: list[str] = []
    preserved_model_sandbox: dict[str, Any] | None = None

    def restore_uncommitted_build_aliases() -> None:
        nonlocal build_aliases_dirty
        if not build_aliases_dirty:
            return
        result["restored_build_image_aliases"] = _restore_built_image_aliases(
            context,
            configured_build_images,
            rollback_tags,
        )
        build_aliases_dirty = False

    try:
        initial = assess_release(
            context,
            project_root,
            minimum_free_gb=minimum_free_gb,
            required_services=LEGACY_EXPECTED_SERVICES,
            require_immutable_images=False,
            verify_image_availability=False,
        )
        result["checks"]["initial_preflight"] = initial
        if initial["status"] != "ready":
            result["status"] = "blocked"
            result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            return result
        backup_capacity = (
            _assess_backup_capacity(
                context,
                backup_root,
                minimum_free_gb=minimum_free_gb,
                format_version=CONTROL_PLANE_BACKUP_FORMAT_VERSION,
            )
            if reuse_backup is None
            else {
                "status": "pass",
                "evidence": "explicit backup reuse requested; exact snapshot validation pending",
            }
        )
        result["checks"]["backup_capacity"] = backup_capacity
        if backup_capacity["status"] != "pass":
            result["status"] = "blocked"
            result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            return result

        built_services = _built_services(context)
        rollback_contract = _capture_rollback_compose_contract(
            context,
            project_root,
            services=tuple(
                sorted(LEGACY_EXPECTED_SERVICES.union(built_services))
            ),
            allow_missing=frozenset(INTRODUCED_SERVICES),
            trusted_external_root=backup_root,
        )
        rollback_base_context = _rollback_contract_context(rollback_contract)
        if preserve_model_sandbox_image:
            assert stable_release_link is not None and preserve_model_sandbox_image_id is not None
            baseline_root = stable_release_link.resolve(strict=True)
            if baseline_root / "deploy/compose.yaml" not in rollback_contract.compose_sources:
                raise RuntimeError("stable model baseline differs from the live rollback contract")
            preserved_model_sandbox = _verify_model_sandbox_reuse(
                context, project_root, baseline_root,
                preserve_model_sandbox_image, preserve_model_sandbox_image_id,
            )
            result["checks"]["model_sandbox_reuse"] = preserved_model_sandbox
        result["rollback_compose_contract"] = {
            "working_directory": str(rollback_contract.working_directory),
            "env_file": str(rollback_contract.env_source),
            "compose_files": [
                str(item) for item in rollback_contract.compose_sources
            ],
            "profiles": list(rollback_contract.profiles),
        }
        rollback_rdagent_docker_storage = _capture_service_storage(
            context,
            "rdagent-docker",
            "/var/lib/docker",
        )
        result["rollback_rdagent_docker_storage"] = rollback_rdagent_docker_storage
        rollback_tags = _capture_rollback_images(
            context,
            release_id,
            services=tuple(
                sorted(LEGACY_EXPECTED_SERVICES.union(built_services))
            ),
            allow_missing=frozenset(INTRODUCED_SERVICES),
            repository=rollback_tag_repository,
        )
        introduced_services = set(built_services) & INTRODUCED_SERVICES
        rollback_disabled_services = frozenset(
            introduced_services - rollback_tags.keys()
        )
        result["rollback_images"] = rollback_tags
        result["rollback_disabled_services"] = sorted(rollback_disabled_services)
        configured_build_images = _configured_service_images(context, *built_services)
        representative_build_services = _representative_build_services(
            configured_build_images,
            built_services,
        )
        build_arguments = ["build"]
        if pull_images:
            build_arguments.append("--pull")
        # A partially successful Compose build can already overwrite mutable
        # aliases, even when the command subsequently fails.
        build_aliases_dirty = True
        context.run(*build_arguments, *representative_build_services)

        final_gate = assess_release(
            context,
            project_root,
            minimum_free_gb=minimum_free_gb,
            required_services=LEGACY_EXPECTED_SERVICES,
            require_immutable_images=False,
            verify_image_availability=False,
        )
        result["checks"]["post_build_preflight"] = final_gate
        if final_gate["status"] != "ready":
            restore_uncommitted_build_aliases()
            result["status"] = "blocked"
            result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            return result
        post_build_backup_capacity = (
            _assess_backup_capacity(
                context,
                backup_root,
                minimum_free_gb=minimum_free_gb,
                format_version=CONTROL_PLANE_BACKUP_FORMAT_VERSION,
            )
            if reuse_backup is None
            else {
                "status": "pass",
                "evidence": "no new governed data archive will be created",
            }
        )
        result["checks"]["post_build_backup_capacity"] = (
            post_build_backup_capacity
        )
        if post_build_backup_capacity["status"] != "pass":
            restore_uncommitted_build_aliases()
            result["status"] = "blocked"
            result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            return result

        if reuse_backup is None:
            admission = _quiesce_release_admission(
                context, stopped_services=admission_stopped, wait_timeout=wait_timeout,
            )
            result["checks"]["admission_quiescence"] = admission

            def guard_backup() -> None:
                _assert_release_backup_idle(context, admission["schema_revision"])
                result["checks"]["pre_backup_durable_work"] = {
                    "status": "pass", "active_jobs": 0, "running_units": 0,
                    "schema_revision": admission["schema_revision"],
                    "writer_services_running": [],
                }

            backup_directory = create_backup(
                context,
                backup_root,
                retention_count=retention_count,
                restart_services=False,
                format_version=CONTROL_PLANE_BACKUP_FORMAT_VERSION,
                minimum_free_gb=minimum_free_gb,
                pre_dump_guard=guard_backup,
            )
            state_mutated = True
            rollback_base_context = _persist_rollback_compose_contract(
                rollback_contract,
                backup_directory,
            )
        else:
            reusable = _validate_reusable_backup(
                context,
                backup_root,
                reuse_backup,
                rollback_contract,
            )
            result["checks"]["reused_backup"] = {
                "status": "pass",
                "directory": str(reusable.directory),
                "schema_revision": reusable.manifest["schema_revision"],
                "expected_data_bytes": reusable.expected_data_bytes,
                "observed_data_bytes": reusable.observed_data_bytes,
                "size_tolerance_bytes": reusable.size_tolerance_bytes,
            }
            stopped = _stop_writers_for_backup_reuse(
                context,
                expected_schema_revision=str(reusable.manifest["schema_revision"]),
            )
            try:
                created_at = datetime.fromisoformat(
                    str(reusable.manifest["created_at"])
                )
                _assert_no_data_file_newer(reusable.data_source, created_at)
                observed_after_stop = _data_usage_bytes(context, reusable.data_source)
                if (
                    abs(observed_after_stop - reusable.expected_data_bytes)
                    > reusable.size_tolerance_bytes
                ):
                    raise RuntimeError(
                        "governed data changed while writers were being stopped"
                    )
            except Exception:
                if stopped:
                    context.run("start", *stopped, check=False)
                raise
            backup_directory = reusable.directory
            state_mutated = True
            result["backup_reused"] = True
            rollback_base_context = _persist_rollback_compose_contract(
                rollback_contract,
                backup_directory,
                directory_name=f"rollback-compose-contract-{release_id.lower()}",
            )
        result["backup_directory"] = str(backup_directory)
        result["rollback_compose_contract"]["snapshot_directory"] = str(
            rollback_base_context.compose_files[0].parent
        )
        # Every writer is stopped by either the fresh-backup or exact-reuse
        # path.  Recheck the durable queue after that stop and take the boundary
        # from PostgreSQL's clock so every final provenance query uses the same
        # authoritative database timeline.
        cutover_at = _record_cutover(context)
        result["cutover_at"] = cutover_at

        # The sandbox builder is intentionally one-shot.  Remove any completed
        # container from an older release so Compose must seed the freshly built
        # worker image into the isolated RD-Agent daemon again.
        context.run("rm", "-s", "-f", "factor-sandbox-builder", check=False)
        # Keep every job-producing service stopped while the new isolated
        # Docker store is seeded. This prevents the scheduler or a worker from
        # claiming durable work between the coordinated backup and acceptance.
        context.run(
            "up",
            "-d",
            "--force-recreate",
            "--remove-orphans",
            "--wait",
            "--wait-timeout",
            str(wait_timeout),
            "rdagent-docker",
        )
        environment_before = context.env_file.read_bytes()
        result["sandbox_images"] = _prepare_sandbox_images(
            context,
            project_root,
            release_id,
            wait_timeout=wait_timeout,
            **({"preserved_model_sandbox": preserved_model_sandbox}
               if preserved_model_sandbox is not None else {}),
        )
        release_identity = _stamp_release_identity(context, release_id)
        result["release_identity"] = {
            "release_id": release_identity["QUANTLAB_RELEASE_ID"],
            "config_digest": release_identity["QUANTLAB_CONFIG_DIGEST"],
            "release_kind": release_identity["QUANTLAB_RELEASE_KIND"],
            "alias_of": release_identity["QUANTLAB_RELEASE_ALIAS_OF"],
            "canonical_baseline": (
                release_identity["QUANTLAB_CANONICAL_BASELINE"] == "true"
            ),
        }
        environment_changed = True
        release_services = expected_services(context)
        core_services = tuple(
            sorted(release_services.difference({"scheduler", "gateway"}))
        )
        context.run(
            "up",
            "-d",
            "--remove-orphans",
            "--wait",
            "--wait-timeout",
            str(wait_timeout),
            *core_services,
        )
        core_acceptance = assess_release(
            context,
            project_root,
            minimum_free_gb=minimum_free_gb,
            required_services=core_services,
        )
        result["checks"]["post_upgrade_core_preflight"] = core_acceptance
        if (
            core_acceptance["status"] != "ready"
            or core_acceptance["migration_state"] != "current"
        ):
            raise RuntimeError("post-upgrade core release acceptance did not pass")

        # Perform every rollback-capable check while scheduler and gateway are
        # still stopped.  No accepted durable write may race a database rollback.
        durable_state = _post_cutover_durable_state(context, cutover_at)
        pre_activation_acceptance = _post_start_acceptance(
            core_acceptance,
            durable_state,
        )
        result["checks"]["pre_activation_acceptance"] = pre_activation_acceptance
        if pre_activation_acceptance["status"] != "pass":
            raise RuntimeError("pre-activation release acceptance did not pass")
        core_identity = _release_identity_acceptance(context, set(core_services))
        result["checks"]["pre_activation_release_identity"] = core_identity
        if core_identity["status"] != "pass":
            raise RuntimeError("pre-activation release identity did not pass")

        # This is the commit boundary.  Starting scheduler or gateway can create
        # durable work; failures after this point are fail-closed activation
        # failures and must never restore the old database snapshot.
        release_committed = True
        build_aliases_dirty = False

        context.run(
            "up",
            "-d",
            "--no-deps",
            "--wait",
            "--wait-timeout",
            str(wait_timeout),
            "scheduler",
            "gateway",
        )
        acceptance = assess_release(
            context,
            project_root,
            minimum_free_gb=minimum_free_gb,
            required_services=release_services,
        )
        result["checks"]["post_activation_preflight"] = acceptance
        activated_durable_state = _post_cutover_durable_state(context, cutover_at)
        post_activation_acceptance = _post_start_acceptance(
            acceptance,
            activated_durable_state,
        )
        result["checks"]["post_activation_acceptance"] = post_activation_acceptance
        if post_activation_acceptance["status"] != "pass":
            raise RuntimeError("post-activation release acceptance did not pass")
        final_identity = _release_identity_acceptance(
            context,
            release_services,
        )
        result["checks"]["final_release_identity"] = final_identity
        if final_identity["status"] != "pass":
            raise RuntimeError("final all-service release identity did not pass")
        result["checks"]["gateway_smoke"] = _gateway_smoke(context)
        if stable_release_link is not None:
            result["stable_release_link"] = _switch_stable_release_link(
                stable_release_link,
                project_root,
            )

        result["status"] = "succeeded"
        if prune_rollback_images:
            try:
                result["pruned_rollback_images"] = _prune_rollback_images(
                    context,
                    rollback_image_retention,
                )
            except Exception as exc:  # cleanup must never roll back an accepted release
                result["pruned_rollback_images"] = []
                result["cleanup_warning"] = f"{type(exc).__name__}: {exc}"
        else:
            result["pruned_rollback_images"] = []
        result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if release_committed:
            build_aliases_dirty = False
            stop = context.run(
                "stop",
                "scheduler",
                "gateway",
                check=False,
            )
            result["activation_fail_closed"] = {
                "stopped_services": ["gateway", "scheduler"],
                "command_output": stop,
                "rollback_permitted": False,
            }
            result["status"] = "activation_failed"
            result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            return result
        try:
            restore_uncommitted_build_aliases()
        except Exception as alias_exc:
            result["image_alias_restore_error"] = (
                f"{type(alias_exc).__name__}: {alias_exc}"
            )
        if environment_changed and environment_before is not None:
            try:
                _atomic_replace(context.env_file, environment_before)
            except Exception as environment_exc:
                result["environment_restore_error"] = (
                    f"{type(environment_exc).__name__}: {environment_exc}"
                )
        if (
            not state_mutated
            or backup_directory is None
            or not rollback_tags
            or rollback_base_context is None
        ):
            if admission_stopped:
                try:
                    _restore_release_admission(context, admission_stopped)
                    result["restored_admission_services"] = list(reversed(admission_stopped))
                except Exception as admission_exc:
                    result["admission_restore_error"] = (
                        f"{type(admission_exc).__name__}: {admission_exc}"
                    )
            result["status"] = "failed"
            result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            return result
        try:
            result["rollback"] = _restore_previous_release(
                context,
                rollback_base_context,
                backup_directory,
                rollback_tags,
                wait_timeout=wait_timeout,
                minimum_free_gb=minimum_free_gb,
                disabled_services=rollback_disabled_services,
                rdagent_docker_storage=rollback_rdagent_docker_storage,
            )
        except Exception as rollback_exc:
            result["status"] = "rollback_failed"
            result["rollback_error"] = f"{type(rollback_exc).__name__}: {rollback_exc}"
        else:
            result["status"] = "rolled_back"
        result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        return result
