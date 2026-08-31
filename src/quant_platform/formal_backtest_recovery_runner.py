"""Host-side coordinator for the single sealed v17 recovery execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import URL, make_url

from quant_data.database import jobs
from quant_platform.formal_backtest_interruption_recovery import (
    V17_INTERRUPTION_RECOVERY_PROFILE,
    V17_RECOVERY_CONTROLLER_SHA256,
    V17_RECOVERY_DATABASE_APPLICATION_NAME,
    FormalBacktestInterruptionRecoveryStore,
)

REFERENCE_CONTAINER = "quantlab-platform-evaluation-worker-1"
SCHEDULER_CONTAINER = "quantlab-platform-scheduler-1"
ONE_SHOT_CONTAINER = "quantlab-v17-recovery-attempt-2"
CONTROLLER_CONTAINER_PATH = "/opt/quantlab-v17-recovery/controller.py"
_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _application_database_url(database_url: str, application_name: str) -> str:
    url = make_url(database_url).update_query_dict(
        {"application_name": application_name}
    )
    return url.render_as_string(hide_password=False)


def resolve_host_database_url(environment: Mapping[str, str]) -> str:
    """Resolve the host-side database endpoint without using Settings' dev default."""
    explicit = str(environment.get("DATABASE_URL") or "").strip()
    if explicit:
        return explicit

    password = str(environment.get("POSTGRES_PASSWORD") or "")
    if not password or any(character in password for character in ("\x00", "\r", "\n")):
        raise ValueError(
            "DATABASE_URL or a single-line POSTGRES_PASSWORD is required"
        )
    host = str(environment.get("POSTGRES_BIND_ADDRESS") or "127.0.0.1").strip()
    if host in {"", "0.0.0.0", "::", "[::]", "*"}:
        host = "127.0.0.1"
    if any(character in host for character in ("\x00", "\r", "\n")):
        raise ValueError("POSTGRES_BIND_ADDRESS is invalid")
    port_text = str(environment.get("POSTGRES_PORT") or "55432").strip()
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("POSTGRES_PORT must be an integer") from exc
    if port < 1 or port > 65535:
        raise ValueError("POSTGRES_PORT is outside the TCP port range")
    return URL.create(
        "postgresql+psycopg",
        username="quantlab",
        password=password,
        host=host,
        port=port,
        database="quantlab",
    ).render_as_string(hide_password=False)


def _environment_map(values: Sequence[str]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for item in values:
        key, separator, value = str(item).partition("=")
        if not separator or not _ENVIRONMENT_KEY.fullmatch(key):
            raise ValueError("evaluation-worker contains an invalid environment entry")
        if any(character in value for character in ("\x00", "\r", "\n")):
            raise ValueError("evaluation-worker contains a multiline environment value")
        environment[key] = value
    return environment


@dataclass(frozen=True)
class ReferenceRuntime:
    container_name: str
    image_id: str
    data_host_root: Path
    network: str
    environment: Mapping[str, str]
    memory_bytes: int
    memory_swap_bytes: int
    nano_cpus: int
    shm_size_bytes: int
    running: bool


def parse_reference_inspect(
    document: Mapping[str, Any],
    *,
    container_name: str = REFERENCE_CONTAINER,
) -> ReferenceRuntime:
    state = dict(document.get("State") or {})
    config = dict(document.get("Config") or {})
    host_config = dict(document.get("HostConfig") or {})
    network_settings = dict(document.get("NetworkSettings") or {})
    networks = dict(network_settings.get("Networks") or {})
    data_mounts = [
        dict(item)
        for item in document.get("Mounts") or []
        if str((item or {}).get("Destination") or "") == "/data"
    ]
    if len(data_mounts) != 1 or len(list(document.get("Mounts") or [])) != 1:
        raise ValueError("evaluation-worker must expose only its one /data mount")
    data_mount = data_mounts[0]
    if (
        data_mount.get("Type") != "bind"
        or data_mount.get("RW") is not True
        or len(networks) != 1
    ):
        raise ValueError("evaluation-worker data mount or network is not the sealed shape")
    image_id = str(document.get("Image") or "")
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    if image_id != profile.worker_runtime_image_digest:
        raise ValueError("evaluation-worker is not the sealed v17 worker image")
    data_host_root = Path(str(data_mount.get("Source") or ""))
    if not data_host_root.is_absolute():
        raise ValueError("evaluation-worker /data host source must be absolute")
    memory = int(host_config.get("Memory") or 0)
    memory_swap = int(host_config.get("MemorySwap") or 0)
    nano_cpus = int(host_config.get("NanoCpus") or 0)
    shm_size = int(host_config.get("ShmSize") or 0)
    if memory <= 0 or memory_swap < memory or nano_cpus <= 0 or shm_size <= 0:
        raise ValueError("evaluation-worker resource limits are not bounded")
    return ReferenceRuntime(
        container_name=container_name,
        image_id=image_id,
        data_host_root=data_host_root,
        network=next(iter(networks)),
        environment=_environment_map(list(config.get("Env") or [])),
        memory_bytes=memory,
        memory_swap_bytes=memory_swap,
        nano_cpus=nano_cpus,
        shm_size_bytes=shm_size,
        running=state.get("Running") is True,
    )


class DockerCLI:
    def __init__(self, executable: str = "docker") -> None:
        self.executable = executable

    def _run(
        self,
        arguments: Sequence[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.executable, *arguments],
            check=check,
            capture_output=True,
            text=True,
            shell=False,
        )

    def inspect_reference(
        self,
        container_name: str = REFERENCE_CONTAINER,
    ) -> ReferenceRuntime:
        result = self._run(("inspect", container_name))
        try:
            documents = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError("docker inspect returned invalid JSON") from exc
        if not isinstance(documents, list) or len(documents) != 1:
            raise ValueError("docker inspect did not return exactly one worker")
        return parse_reference_inspect(documents[0], container_name=container_name)

    def require_image(self, image_id: str) -> None:
        result = self._run(("image", "inspect", image_id))
        try:
            documents = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError("docker image inspect returned invalid JSON") from exc
        if (
            not isinstance(documents, list)
            or len(documents) != 1
            or str(documents[0].get("Id") or "") != image_id
        ):
            raise ValueError("sealed v17 worker image ID is unavailable")

    def stop(self, container_name: str) -> None:
        self._run(("stop", "--time", "60", container_name))

    def container_running(self, container_name: str) -> bool:
        result = self._run(
            ("inspect", "--format", "{{.State.Running}}", container_name),
            check=False,
        )
        if result.returncode != 0:
            raise ValueError(f"required container is unavailable: {container_name}")
        value = result.stdout.strip().lower()
        if value not in {"true", "false"}:
            raise ValueError("docker container state is invalid")
        return value == "true"

    def start_if_stopped(self, container_name: str) -> None:
        if not self.container_running(container_name):
            self._run(("start", container_name))

    def run_one_shot(self, command: Sequence[str]) -> int:
        result = self._run(tuple(command), check=False)
        return int(result.returncode)

    def require_one_shot_absent(self) -> None:
        result = self._run(("inspect", ONE_SHOT_CONTAINER), check=False)
        if result.returncode == 0:
            raise RuntimeError("a prior v17 one-shot container still exists")

    def force_remove_one_shot(self) -> None:
        self._run(("rm", "--force", ONE_SHOT_CONTAINER), check=False)
        self.require_one_shot_absent()


def _logical_host_path(logical_path: str, *, data_host_root: Path) -> Path:
    logical = PurePosixPath(logical_path)
    if not logical.is_absolute() or logical.parts[:2] != ("/", "data"):
        raise ValueError("recovery path must be rooted at /data")
    root = data_host_root.resolve()
    candidate = root.joinpath(*logical.parts[2:])
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError("recovery path escapes the persistent /data source") from exc
    return candidate


def _assert_no_symlink_ancestors(path: Path, *, root: Path) -> None:
    resolved_root = root.resolve()
    relative = path.relative_to(resolved_root)
    current = resolved_root
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("recovery persistent target contains a symlink")


def prepare_persistent_targets(
    *, data_host_root: Path
) -> tuple[Path, Path]:
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    root = data_host_root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("evaluation-worker persistent /data source is unavailable")
    target = _logical_host_path(profile.target_artifact_path, data_host_root=root)
    log_path = _logical_host_path(profile.target_execution_log_path, data_host_root=root)
    _assert_no_symlink_ancestors(target.parent, root=root)
    target.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_ancestors(target, root=root)
    if target.is_symlink() or not target.is_dir() or any(target.iterdir()):
        raise ValueError("recovery attempt-2 artifact target is not empty")
    _assert_no_symlink_ancestors(log_path.parent, root=root)
    if log_path.exists():
        if log_path.is_symlink() or not log_path.is_file() or log_path.stat().st_size:
            raise ValueError("recovery attempt-2 execution log is not empty")
    else:
        descriptor = os.open(
            log_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        os.close(descriptor)
    os.chmod(log_path, stat.S_IRUSR | stat.S_IWUSR)
    return target, log_path


def build_one_shot_environment(reference: ReferenceRuntime) -> dict[str, str]:
    environment = dict(reference.environment)
    database_url = str(environment.get("DATABASE_URL") or "")
    if not database_url:
        raise ValueError("evaluation-worker DATABASE_URL is missing")
    environment.update(
        {
            "DATABASE_URL": _application_database_url(
                database_url,
                V17_RECOVERY_DATABASE_APPLICATION_NAME,
            ),
            "DATA_ROOT": "/data",
            "WORKER_JOB_KINDS": "strategy_backtest",
            "WORKER_CONCURRENCY": "1",
            "RESEARCH_CPU_BUDGET": "0",
            "RESEARCH_MEMORY_BUDGET_GB": "0",
            "RDAGENT_ENABLED": "false",
            "QUANTLAB_WORKER_RUNTIME_IMAGE_DIGEST": reference.image_id,
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def write_private_environment(path: Path, environment: Mapping[str, str]) -> None:
    lines: list[str] = []
    for key, value in sorted(environment.items()):
        if not _ENVIRONMENT_KEY.fullmatch(key) or any(
            character in value for character in ("\x00", "\r", "\n")
        ):
            raise ValueError("one-shot environment cannot be encoded safely")
        lines.append(f"{key}={value}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise


def build_one_shot_command(
    reference: ReferenceRuntime,
    *,
    environment_file: Path,
    controller_path: Path,
    target_path: Path,
    log_path: Path,
) -> list[str]:
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    expected_target = _logical_host_path(
        profile.target_artifact_path,
        data_host_root=reference.data_host_root,
    )
    expected_log = _logical_host_path(
        profile.target_execution_log_path,
        data_host_root=reference.data_host_root,
    )
    if target_path.resolve() != expected_target.resolve() or log_path.resolve() != (
        expected_log.resolve()
    ):
        raise ValueError("one-shot mounts do not point into the persistent /data source")
    for path in (environment_file, controller_path, target_path, log_path):
        if "," in str(path) or "\n" in str(path) or "\r" in str(path):
            raise ValueError("one-shot bind paths cannot be encoded safely")
    cpus = reference.nano_cpus / 1_000_000_000
    return [
        "run",
        "--rm",
        "--restart",
        "no",
        "--name",
        ONE_SHOT_CONTAINER,
        "--network",
        reference.network,
        "--cpus",
        f"{cpus:g}",
        "--memory",
        str(reference.memory_bytes),
        "--memory-swap",
        str(reference.memory_swap_bytes),
        "--shm-size",
        str(reference.shm_size_bytes),
        "--env-file",
        str(environment_file),
        "--volumes-from",
        f"{reference.container_name}:ro",
        "--mount",
        (
            f"type=bind,src={controller_path},dst={CONTROLLER_CONTAINER_PATH},"
            "readonly"
        ),
        "--mount",
        (
            f"type=bind,src={target_path},dst={profile.source_artifact_path}"
        ),
        "--mount",
        (
            f"type=bind,src={log_path},dst={profile.target_execution_log_path}"
        ),
        reference.image_id,
        "python",
        CONTROLLER_CONTAINER_PATH,
    ]


def _require_worker_idle(
    store: FormalBacktestInterruptionRecoveryStore,
    worker_kinds: tuple[str, ...],
) -> None:
    predicate = [
        jobs.c.status.in_(("queued", "running")),
        jobs.c.id != V17_INTERRUPTION_RECOVERY_PROFILE.job_id,
    ]
    if worker_kinds:
        predicate.append(jobs.c.kind.in_(worker_kinds))
    with store.engine.connect() as connection:
        rows = connection.execute(select(jobs.c.id, jobs.c.kind).where(*predicate)).all()
    if rows:
        raise ValueError(
            "evaluation-worker has runnable jobs and cannot be stopped safely"
        )


def run_v17_recovery(
    *,
    database_url: str,
    actor: str,
    external_interruption: Mapping[str, Any],
    controller_path: Path,
    docker: DockerCLI,
    reference_container: str = REFERENCE_CONTAINER,
    scheduler_container: str = SCHEDULER_CONTAINER,
    store_factory: Callable[..., FormalBacktestInterruptionRecoveryStore] = (
        FormalBacktestInterruptionRecoveryStore
    ),
    idle_check: Callable[... , None] = _require_worker_idle,
) -> dict[str, Any]:
    controller = controller_path.resolve()
    if (
        not controller.is_file()
        or controller.is_symlink()
        or _sha256_file(controller) != V17_RECOVERY_CONTROLLER_SHA256
    ):
        raise ValueError("sealed v17 one-shot controller source hash changed")
    reference = docker.inspect_reference(reference_container)
    docker.require_image(reference.image_id)
    if reference.data_host_root.is_symlink():
        raise ValueError("evaluation-worker /data host source cannot be a symlink")
    data_host_root = reference.data_host_root.resolve()
    store = store_factory(database_url, data_root=data_host_root)
    preflight = store.preflight_source(external_interruption=external_interruption)
    registration_committed = preflight.get("status") == "already_registered"
    if registration_committed:
        try:
            terminal = store.verify_terminal_execution()
        except ValueError:
            if reference.running:
                raise ValueError(
                    "a registered pending recovery requires the historical "
                    "evaluation-worker to remain stopped"
                ) from None
        else:
            return {
                "status": "already_terminal",
                "job_id": V17_INTERRUPTION_RECOVERY_PROFILE.job_id,
                "backtest_id": V17_INTERRUPTION_RECOVERY_PROFILE.backtest_id,
                "receipt_sha256": terminal["receipt_sha256"],
            }
    elif not reference.running:
        raise ValueError(
            "an unregistered recovery requires the evaluation-worker to start running"
        )
    worker_kinds = tuple(
        item.strip()
        for item in str(reference.environment.get("WORKER_JOB_KINDS") or "").split(",")
        if item.strip()
    )
    idle_check(store, worker_kinds)
    scheduler_was_running = docker.container_running(scheduler_container)
    restore_reference_after_terminal = reference.running or registration_committed
    restore_scheduler_after_terminal = scheduler_was_running or registration_committed
    safe_to_restore = False
    try:
        if scheduler_was_running:
            docker.stop(scheduler_container)
        idle_check(store, worker_kinds)
        if reference.running:
            docker.stop(reference.container_name)
        stopped = docker.inspect_reference(reference.container_name)
        if stopped.running:
            raise RuntimeError("evaluation-worker did not stop")
        idle_check(store, worker_kinds)
        environment = build_one_shot_environment(reference)
        with tempfile.TemporaryDirectory(prefix="quantlab-v17-recovery-") as directory:
            environment_file = Path(directory) / "worker.env"
            write_private_environment(environment_file, environment)
            profile = V17_INTERRUPTION_RECOVERY_PROFILE
            target_path = _logical_host_path(
                profile.target_artifact_path,
                data_host_root=data_host_root,
            )
            log_path = _logical_host_path(
                profile.target_execution_log_path,
                data_host_root=data_host_root,
            )
            command = build_one_shot_command(
                reference,
                environment_file=environment_file,
                controller_path=controller,
                target_path=target_path,
                log_path=log_path,
            )
            registration = store.register_and_requeue(
                actor=actor,
                external_interruption=external_interruption,
            )
            registration_committed = True
            store.require_authorized_queue()
            prepared_target, prepared_log = prepare_persistent_targets(
                data_host_root=data_host_root
            )
            if prepared_target != target_path or prepared_log != log_path:
                raise RuntimeError("prepared recovery mounts changed after authorization")
            store.require_authorized_queue()
            runtime_store = store_factory(
                _application_database_url(
                    database_url,
                    V17_RECOVERY_DATABASE_APPLICATION_NAME,
                ),
                data_root=data_host_root,
            )
            docker.require_one_shot_absent()
            try:
                exit_code = docker.run_one_shot(command)
            except BaseException:
                docker.force_remove_one_shot()
                try:
                    runtime_store.settle_controller_failure()
                except ValueError as settlement_exc:
                    raise RuntimeError(
                        "one-shot container launch failed and its exact state could not "
                        "be settled"
                    ) from settlement_exc
                runtime_store.verify_terminal_execution()
                safe_to_restore = True
                raise
            docker.require_one_shot_absent()
        if exit_code != 0:
            docker.force_remove_one_shot()
            runtime_store.settle_controller_failure()
        terminal = runtime_store.verify_terminal_execution()
        safe_to_restore = True
        if terminal["job_status"] != "succeeded":
            raise RuntimeError(
                f"sealed v17 recovery ended {terminal['job_status']} "
                f"(container exit {exit_code})"
            )
        return {
            "status": "succeeded",
            "registration_status": registration["status"],
            "job_id": V17_INTERRUPTION_RECOVERY_PROFILE.job_id,
            "backtest_id": V17_INTERRUPTION_RECOVERY_PROFILE.backtest_id,
            "receipt_sha256": terminal["receipt_sha256"],
            "container_exit_code": exit_code,
        }
    except BaseException as exc:
        if registration_committed and not safe_to_restore:
            for container_name in (reference.container_name, scheduler_container):
                try:
                    if docker.container_running(container_name):
                        docker.stop(container_name)
                except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
                    pass
            raise RuntimeError(
                "v17 recovery is authorized but not terminal; evaluation-worker and "
                "scheduler remain stopped for operator review"
            ) from exc
        raise
    finally:
        if safe_to_restore or not registration_committed:
            if restore_reference_after_terminal:
                docker.start_if_stopped(reference.container_name)
            if restore_scheduler_after_terminal:
                docker.start_if_stopped(scheduler_container)
