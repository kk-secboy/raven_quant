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
    WRITER_SERVICES,
    ComposeContext,
    _platform_secret_key_fingerprint,
    create_backup,
    load_and_verify_manifest,
    restore_backup,
)
from .release_preflight import (
    LEGACY_EXPECTED_SERVICES,
    assess_release,
    expected_services,
)

BUILT_SERVICES = (
    "api",
    "scheduler",
    "worker",
    "rdagent-worker",
    "rdagent-data-science-worker",
    "web",
)
PROFILE_BUILT_SERVICES = {
    "gpu": ("rdagent-llm-finetune-worker",),
}
INTRODUCED_SERVICES = {
    "rdagent-data-science-worker",
    "rdagent-llm-finetune-worker",
}
_ROLLBACK_SERVICES = tuple(
    sorted(
        LEGACY_EXPECTED_SERVICES.union(BUILT_SERVICES).union(
            PROFILE_BUILT_SERVICES["gpu"]
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
    "data_science": "data-science-sandbox",
    "model": "model-sandbox",
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
    """Capture the exact Compose release contract owning the running containers."""

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
        present_services.add(service)
    if len(identities) != 1:
        raise RuntimeError("running rollback services do not share one Compose release contract")

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
    """Persist the old Compose/env bytes while preserving its project directory."""

    if not re.fullmatch(r"rollback-compose-contract(?:-[0-9a-z]+)?", directory_name):
        raise ValueError("rollback Compose contract directory name is invalid")
    root = (backup_directory.resolve() / directory_name).resolve()
    if not _inside_path(root, backup_directory.resolve()):
        raise RuntimeError("rollback Compose contract target escapes the backup")
    root.mkdir(mode=0o700)
    env_target = root / "environment.env"
    _atomic_replace(env_target, contract.env_content)
    env_target.chmod(0o600)
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
        "env_snapshot": env_target.name,
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
    if env_target.read_bytes() != contract.env_content:
        raise RuntimeError("persisted rollback environment failed verification")
    rollback = ComposeContext(
        project_name=contract.project_name,
        env_file=env_target,
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
    sources = (contract.env_source, *contract.compose_sources)
    if any(not _inside_path(source, contract_root) for source in sources):
        raise RuntimeError(
            "running services are not owned by the reusable backup rollback contract"
        )
    manifest_path = contract_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("reusable backup rollback contract is invalid") from exc
    env_target = (contract_root / str(manifest.get("env_snapshot") or "")).resolve()
    if env_target != contract.env_source.resolve():
        raise RuntimeError("reusable backup rollback environment does not match")
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


def _update_environment(path: Path, values: dict[str, str]) -> None:
    original = path.read_text(encoding="utf-8-sig")
    remaining = dict(values)
    output: list[str] = []
    for line in original.splitlines():
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in remaining and not line.lstrip().startswith("#"):
            output.append(f"{key}={remaining.pop(key)}")
        else:
            output.append(line)
    if remaining:
        if output and output[-1]:
            output.append("")
        output.extend(f"{key}={value}" for key, value in sorted(remaining.items()))
    _atomic_replace(path, ("\n".join(output) + "\n").encode("utf-8"))


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


def _prepare_sandbox_images(
    context: ComposeContext,
    project_root: Path,
    release_id: str,
    *,
    wait_timeout: int,
) -> dict[str, Any]:
    """Seal locally built sandboxes in a loopback-only deployment registry.

    The registry is used only while publishing. Runtime jobs are offline and
    consume the digest-pinned images already present in the dedicated DinD
    daemon. Registry data and DinD layers live on the data disk via Compose.
    """

    port = _registry_port(context)
    host_registry = f"127.0.0.1:{port}"
    dind_registry = "rdagent-registry:5000"
    release_tag = release_id.lower()
    runtime_source = f"{context.project_name}-rdagent-worker:latest"
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
    host_data_science_repository = f"{host_registry}/quantlab/data-science-sandbox"
    host_model_repository = f"{host_registry}/quantlab/model-sandbox"
    dind_base_repository = f"{dind_registry}/quantlab/worker-sandbox-base"
    dind_qlib_repository = f"{dind_registry}/quantlab/qlib-sandbox"
    dind_data_science_repository = f"{dind_registry}/quantlab/data-science-sandbox"
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
        sandbox_base_source = "quantlab-worker-runtime:v2"
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
            host_image_tag=f"{host_qlib_repository}:{release_tag}",
            host_repository=host_qlib_repository,
            dind_repository=dind_qlib_repository,
            timeout=image_timeout,
            build_args=(f"QLIB_SANDBOX_BASE_IMAGE={host_base_image}",),
        )
        data_science_image = _build_and_publish_host_image(
            context,
            context_root=project_root.resolve() / "deploy" / "data-science-sandbox",
            host_image_tag=f"{host_data_science_repository}:{release_tag}",
            host_repository=host_data_science_repository,
            dind_repository=dind_data_science_repository,
            timeout=image_timeout,
            build_args=(f"DATA_SCIENCE_SANDBOX_BASE_IMAGE={host_base_image}",),
        )
        model_image = _build_and_publish_host_image(
            context,
            context_root=project_root.resolve() / "deploy" / "model-sandbox",
            host_image_tag=f"{host_model_repository}:{release_tag}",
            host_repository=host_model_repository,
            dind_repository=dind_model_repository,
            timeout=image_timeout,
            build_args=(f"MODEL_SANDBOX_BASE_IMAGE={host_base_image}",),
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
                        "features=D.features(D.instruments('all'), ['$close'], "
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
            data_science_image,
            (
                "python",
                "-c",
                "import matplotlib, numpy, pandas, seaborn, sklearn, xgboost; "
                "assert matplotlib and numpy and pandas and seaborn and sklearn and xgboost",
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
        sealed = {
            "RDAGENT_RUNTIME_IMAGE_DIGEST": runtime_image_id,
            "RDAGENT_QLIB_SANDBOX_IMAGE": qlib_image,
            "RDAGENT_DATA_SCIENCE_IMAGE": data_science_image,
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
        }
    finally:
        context.run(
            "--profile",
            "sandbox-registry",
            "stop",
            "rdagent-registry",
            check=False,
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
) -> dict[str, Any]:
    """Fail closed unless the target can hold a worst-case full data copy.

    The backup writer creates the new archive before retention removes an old
    generation. Gzip ratios are data dependent, so planning from a nominal
    fixed headroom can fill the filesystem. Use the uncompressed governed data
    mount size as the conservative upper bound and retain the requested
    operational headroom in addition to it.
    """

    try:
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
        source_bytes = source_kib * 1024
        anchor = _existing_storage_anchor(backup_root)
        free_bytes = shutil.disk_usage(anchor).free
        required_bytes = source_bytes + int(minimum_free_gb * _GIB)
        passed = free_bytes >= required_bytes
        evidence = (
            f"source {source}; target {backup_root.resolve()}; "
            f"free {free_bytes / _GIB:.1f} GiB; "
            f"data upper bound {source_bytes / _GIB:.1f} GiB; retained headroom "
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
            else "Choose a backup root with space for one full uncompressed data "
            "generation plus release headroom."
        ),
    }


def _capture_rollback_images(
    context: ComposeContext,
    release_id: str,
    *,
    services: tuple[str, ...] = BUILT_SERVICES,
    allow_missing: frozenset[str] = frozenset(),
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
        tag = f"quantlab-rollback:{release_id.lower()}-{service}"
        context.docker("tag", image_id, tag)
        tags[service] = tag
    return tags


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
    reuse_backup: Path | None = None,
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
        )
        introduced_services = set(built_services) & INTRODUCED_SERVICES
        rollback_disabled_services = frozenset(
            introduced_services - rollback_tags.keys()
        )
        result["rollback_images"] = rollback_tags
        result["rollback_disabled_services"] = sorted(rollback_disabled_services)
        build_arguments = ["build"]
        if pull_images:
            build_arguments.append("--pull")
        context.run(*build_arguments, *built_services)

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
            result["status"] = "blocked"
            result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            return result
        post_build_backup_capacity = (
            _assess_backup_capacity(
                context,
                backup_root,
                minimum_free_gb=minimum_free_gb,
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
            result["status"] = "blocked"
            result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            return result

        if reuse_backup is None:
            backup_directory = create_backup(
                context,
                backup_root,
                retention_count=retention_count,
                restart_services=False,
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
        )
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

        # The scheduler is the first component allowed to create new durable
        # work.  Gateway remains stopped so no external request can race the
        # final service and provenance checks.
        context.run(
            "up",
            "-d",
            "--no-deps",
            "--wait",
            "--wait-timeout",
            str(wait_timeout),
            "scheduler",
        )
        # Keep the externally exposed gateway closed until every database,
        # image, service and durable-work provenance check has passed. Otherwise
        # a request accepted during this window could be lost by rollback.
        acceptance = assess_release(
            context,
            project_root,
            minimum_free_gb=minimum_free_gb,
            required_services=release_services.difference({"gateway"}),
        )
        result["checks"]["post_upgrade_preflight"] = acceptance
        durable_state = _post_cutover_durable_state(context, cutover_at)
        post_start_acceptance = _post_start_acceptance(
            acceptance,
            durable_state,
        )
        result["checks"]["post_upgrade_acceptance"] = post_start_acceptance
        if post_start_acceptance["status"] != "pass":
            raise RuntimeError("post-upgrade release acceptance did not pass")

        context.run(
            "up",
            "-d",
            "--no-deps",
            "--wait",
            "--wait-timeout",
            str(wait_timeout),
            "gateway",
        )
        result["checks"]["gateway_smoke"] = _gateway_smoke(context)

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
