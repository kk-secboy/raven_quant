from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
from sqlalchemy.engine import make_url

from quant_data import database as database_module
from quant_platform.formal_backtest_interruption_recovery import (
    V17_INTERRUPTION_RECOVERY_PROFILE,
    V17_RECOVERY_CONTROLLER_SHA256,
    V17_RECOVERY_DATABASE_APPLICATION_NAME,
)
from quant_platform.formal_backtest_recovery_runner import (
    CONTROLLER_CONTAINER_PATH,
    REFERENCE_CONTAINER,
    ReferenceRuntime,
    build_one_shot_command,
    build_one_shot_environment,
    parse_reference_inspect,
    prepare_persistent_targets,
    resolve_host_database_url,
    run_v17_recovery,
)

pytestmark = pytest.mark.no_database


def _reference(data_root: Path, *, running: bool = True) -> ReferenceRuntime:
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    return ReferenceRuntime(
        container_name=REFERENCE_CONTAINER,
        image_id=profile.worker_runtime_image_digest,
        data_host_root=data_root,
        network="quantlab-platform_default",
        environment={
            "DATABASE_URL": (
                "postgresql+psycopg://quantlab:secret@postgres:5432/quantlab"
            ),
            "WORKER_JOB_KINDS": "factor_evaluate,strategy_backtest",
            "PLATFORM_SECRET_KEY": "do-not-print",
        },
        memory_bytes=48 * 1024**3,
        memory_swap_bytes=96 * 1024**3,
        nano_cpus=24_000_000_000,
        shm_size_bytes=64 * 1024**2,
        running=running,
    )


def _inspect_document(data_root: Path) -> dict[str, Any]:
    reference = _reference(data_root)
    return {
        "State": {"Running": True},
        "Image": reference.image_id,
        "Config": {
            "Env": [f"{key}={value}" for key, value in reference.environment.items()]
        },
        "HostConfig": {
            "Memory": reference.memory_bytes,
            "MemorySwap": reference.memory_swap_bytes,
            "NanoCpus": reference.nano_cpus,
            "ShmSize": reference.shm_size_bytes,
        },
        "NetworkSettings": {"Networks": {reference.network: {}}},
        "Mounts": [
            {
                "Type": "bind",
                "Source": str(data_root),
                "Destination": "/data",
                "RW": True,
            }
        ],
    }


def _mount_destinations(command: list[str]) -> dict[str, str]:
    mounts: dict[str, str] = {}
    for index, value in enumerate(command):
        if value != "--mount":
            continue
        fields = dict(
            item.split("=", 1)
            for item in command[index + 1].split(",")
            if "=" in item
        )
        mounts[fields["dst"]] = fields["src"]
    return mounts


def test_reference_inspect_pins_image_data_network_and_resource_limits(
    tmp_path: Path,
) -> None:
    runtime = parse_reference_inspect(_inspect_document(tmp_path))

    assert runtime.image_id == V17_INTERRUPTION_RECOVERY_PROFILE.worker_runtime_image_digest
    assert runtime.data_host_root == tmp_path
    assert runtime.network == "quantlab-platform_default"
    assert runtime.memory_bytes == 48 * 1024**3
    assert runtime.memory_swap_bytes == 96 * 1024**3
    assert runtime.nano_cpus == 24_000_000_000
    assert runtime.shm_size_bytes == 64 * 1024**2


@pytest.mark.parametrize(
    "change", ["image", "mount", "extra_mount", "network", "memory"]
)
def test_reference_inspect_rejects_runtime_drift(tmp_path: Path, change: str) -> None:
    document = _inspect_document(tmp_path)
    if change == "image":
        document["Image"] = "sha256:" + "0" * 64
    elif change == "mount":
        document["Mounts"][0]["RW"] = False
    elif change == "extra_mount":
        document["Mounts"].append(
            {
                "Type": "bind",
                "Source": "/var/run/docker.sock",
                "Destination": "/var/run/docker.sock",
                "RW": True,
            }
        )
    elif change == "network":
        document["NetworkSettings"]["Networks"]["unexpected"] = {}
    else:
        document["HostConfig"]["Memory"] = 0

    with pytest.raises(ValueError):
        parse_reference_inspect(document)


def test_one_shot_command_uses_persistent_alias_and_never_mounts_natural_target(
    tmp_path: Path,
) -> None:
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    reference = _reference(tmp_path)
    target, log_path = prepare_persistent_targets(data_host_root=tmp_path)
    environment_file = tmp_path / "worker.env"
    environment_file.write_text("WORKER_JOB_KINDS=strategy_backtest\n", encoding="utf-8")
    controller = tmp_path / "controller.py"
    controller.write_text("# fixture\n", encoding="utf-8")

    command = build_one_shot_command(
        reference,
        environment_file=environment_file,
        controller_path=controller,
        target_path=target,
        log_path=log_path,
    )
    mounts = _mount_destinations(command)

    assert command[command.index("--volumes-from") + 1] == f"{REFERENCE_CONTAINER}:ro"
    assert mounts[profile.source_artifact_path] == str(target)
    assert profile.target_artifact_path not in mounts
    assert mounts[profile.target_execution_log_path] == str(log_path)
    assert mounts[CONTROLLER_CONTAINER_PATH] == str(controller)
    assert command[command.index("--cpus") + 1] == "24"
    assert command[command.index("--memory") + 1] == str(48 * 1024**3)
    assert command[command.index("--memory-swap") + 1] == str(96 * 1024**3)
    assert command[-3:] == [reference.image_id, "python", CONTROLLER_CONTAINER_PATH]


def test_one_shot_environment_uses_exact_appname_and_one_job_kind(tmp_path: Path) -> None:
    environment = build_one_shot_environment(_reference(tmp_path))
    database_url = make_url(environment["DATABASE_URL"])

    assert database_url.query["application_name"] == (
        V17_RECOVERY_DATABASE_APPLICATION_NAME
    )
    assert database_url.password == "secret"
    assert environment["WORKER_JOB_KINDS"] == "strategy_backtest"
    assert environment["WORKER_CONCURRENCY"] == "1"
    assert environment["DATA_ROOT"] == "/data"
    assert environment["QUANTLAB_WORKER_RUNTIME_IMAGE_DIGEST"] == (
        V17_INTERRUPTION_RECOVERY_PROFILE.worker_runtime_image_digest
    )


def test_host_database_url_prefers_explicit_database_url() -> None:
    explicit = "postgresql+psycopg://operator:sealed@db.example:6543/formal"

    assert resolve_host_database_url(
        {
            "DATABASE_URL": explicit,
            "POSTGRES_PASSWORD": "must-not-replace-explicit",
        }
    ) == explicit


def test_host_database_url_uses_deploy_postgres_values_and_encodes_password() -> None:
    database_url = resolve_host_database_url(
        {
            "POSTGRES_PASSWORD": "p@ss:/?#[]",
            "POSTGRES_PORT": "55432",
        }
    )
    parsed = make_url(database_url)

    assert parsed.drivername == "postgresql+psycopg"
    assert parsed.username == "quantlab"
    assert parsed.password == "p@ss:/?#[]"
    assert parsed.host == "127.0.0.1"
    assert parsed.port == 55432
    assert parsed.database == "quantlab"
    assert "p@ss" not in database_url


def test_persistent_target_and_log_are_created_empty_after_authorization(
    tmp_path: Path,
) -> None:
    target, log_path = prepare_persistent_targets(data_host_root=tmp_path)

    assert target.is_dir() and not any(target.iterdir())
    assert log_path.is_file() and log_path.stat().st_size == 0
    if os.name != "nt":
        assert stat.S_IMODE(log_path.stat().st_mode) & 0o777 == 0o600

    (target / "unexpected").write_text("output", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        prepare_persistent_targets(data_host_root=tmp_path)


class _FakeDocker:
    def __init__(self, reference: ReferenceRuntime, events: list[str], exit_code: int) -> None:
        self.reference = reference
        self.events = events
        self.exit_code = exit_code
        self.running = True
        self.scheduler_running = True
        self.one_shot_present = False

    def inspect_reference(self, _container_name: str) -> ReferenceRuntime:
        self.events.append(f"inspect:{self.running}")
        return ReferenceRuntime(**{**self.reference.__dict__, "running": self.running})

    def require_image(self, _image_id: str) -> None:
        self.events.append("image")

    def container_running(self, container_name: str) -> bool:
        self.events.append(f"state:{container_name}")
        if container_name == REFERENCE_CONTAINER:
            return self.running
        return self.scheduler_running

    def stop(self, container_name: str) -> None:
        self.events.append(f"stop:{container_name}")
        if container_name == REFERENCE_CONTAINER:
            self.running = False
        else:
            self.scheduler_running = False

    def start_if_stopped(self, container_name: str) -> None:
        self.events.append(f"start:{container_name}")
        if container_name == REFERENCE_CONTAINER:
            self.running = True
        else:
            self.scheduler_running = True

    def require_one_shot_absent(self) -> None:
        self.events.append("absent")
        assert self.one_shot_present is False

    def force_remove_one_shot(self) -> None:
        self.events.append("force-remove")
        self.one_shot_present = False

    def run_one_shot(self, command: list[str]) -> int:
        self.events.append("run")
        environment_file = Path(command[command.index("--env-file") + 1])
        assert environment_file.is_file()
        if os.name != "nt":
            assert stat.S_IMODE(environment_file.stat().st_mode) & 0o777 == 0o600
        assert "secret" not in " ".join(command)
        mounts = _mount_destinations(command)
        assert V17_INTERRUPTION_RECOVERY_PROFILE.target_artifact_path not in mounts
        return self.exit_code


class _FakeStore:
    def __init__(
        self,
        events: list[str],
        target: Path,
        *,
        terminal_status: str,
    ) -> None:
        self.events = events
        self.target = target
        self.terminal_status = terminal_status
        self.queue_checks = 0

    def preflight_source(self, **_kwargs: Any) -> dict[str, str]:
        self.events.append("preflight")
        return {"status": "source_verified"}

    def register_and_requeue(self, **_kwargs: Any) -> dict[str, str]:
        assert not self.target.exists()
        self.events.append("register")
        return {"status": "registered_and_queued"}

    def require_authorized_queue(self) -> None:
        self.queue_checks += 1
        if self.queue_checks == 1:
            assert not self.target.exists()
        else:
            assert self.target.is_dir() and not any(self.target.iterdir())
        self.events.append(f"queue:{self.queue_checks}")

    def settle_controller_failure(self) -> None:
        self.events.append("settle")

    def verify_terminal_execution(self) -> dict[str, Any]:
        self.events.append("verify")
        return {
            "job_status": self.terminal_status,
            "receipt_sha256": "receipt",
        }


@pytest.mark.parametrize(("exit_code", "terminal"), [(0, "succeeded"), (7, "failed")])
def test_coordinator_orders_stop_register_mount_run_settle_and_restore(
    tmp_path: Path,
    exit_code: int,
    terminal: str,
) -> None:
    events: list[str] = []
    reference = _reference(tmp_path)
    docker = _FakeDocker(reference, events, exit_code)
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    target = tmp_path.joinpath(*PurePosixPath(profile.target_artifact_path).parts[2:])
    stores: list[_FakeStore] = []

    def store_factory(_database_url: str, *, data_root: Path) -> _FakeStore:
        assert data_root == tmp_path.resolve()
        store = _FakeStore(events, target, terminal_status=terminal)
        stores.append(store)
        return store

    def idle(_store: _FakeStore, _kinds: tuple[str, ...]) -> None:
        events.append("idle")

    controller = Path(__file__).parents[1] / "scripts" / "v17_recovery_oneshot_controller.py"
    if terminal == "succeeded":
        result = run_v17_recovery(
            database_url="postgresql+psycopg://quantlab:secret@host/quantlab",
            actor="operator",
            external_interruption={"exact": True},
            controller_path=controller,
            docker=docker,
            store_factory=store_factory,
            idle_check=idle,
        )
        assert result["status"] == "succeeded"
    else:
        with pytest.raises(RuntimeError, match="ended failed"):
            run_v17_recovery(
                database_url="postgresql+psycopg://quantlab:secret@host/quantlab",
                actor="operator",
                external_interruption={"exact": True},
                controller_path=controller,
                docker=docker,
                store_factory=store_factory,
                idle_check=idle,
            )

    assert events[:11] == [
        "inspect:True",
        "image",
        "preflight",
        "idle",
        "state:quantlab-platform-scheduler-1",
        "stop:quantlab-platform-scheduler-1",
        "idle",
        "stop:quantlab-platform-evaluation-worker-1",
        "inspect:False",
        "idle",
        "register",
    ]
    assert events.index("register") < events.index("queue:1") < events.index("queue:2")
    assert events.index("queue:2") < events.index("run") < events.index("verify")
    if exit_code:
        assert events.index("run") < events.index("settle") < events.index("verify")
    else:
        assert "settle" not in events
    assert events[-2:] == [
        "start:quantlab-platform-evaluation-worker-1",
        "start:quantlab-platform-scheduler-1",
    ]
    assert docker.running is True
    assert docker.scheduler_running is True
    assert len(stores) == 2


def test_authorized_queued_failure_keeps_scheduler_and_worker_stopped(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    docker = _FakeDocker(_reference(tmp_path), events, 0)
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    target = tmp_path.joinpath(*PurePosixPath(profile.target_artifact_path).parts[2:])

    class FailingQueueStore(_FakeStore):
        def require_authorized_queue(self) -> None:
            super().require_authorized_queue()
            if self.queue_checks == 2:
                raise ValueError("prelaunch mount validation failed")

    def store_factory(_database_url: str, *, data_root: Path) -> _FakeStore:
        assert data_root == tmp_path.resolve()
        return FailingQueueStore(events, target, terminal_status="failed")

    controller = Path(__file__).parents[1] / "scripts" / "v17_recovery_oneshot_controller.py"
    with pytest.raises(RuntimeError, match="remain stopped"):
        run_v17_recovery(
            database_url="postgresql+psycopg://quantlab:secret@host/quantlab",
            actor="operator",
            external_interruption={"exact": True},
            controller_path=controller,
            docker=docker,
            store_factory=store_factory,
            idle_check=lambda _store, _kinds: events.append("idle"),
        )

    assert "register" in events
    assert "run" not in events
    assert not any(event.startswith("start:") for event in events)
    assert docker.running is False
    assert docker.scheduler_running is False


def test_registered_pending_recovery_resumes_with_stopped_reference(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    shared: dict[str, bool] = {
        "registered": False,
        "fail_prelaunch_once": True,
        "terminal": False,
    }
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    target = tmp_path.joinpath(*PurePosixPath(profile.target_artifact_path).parts[2:])

    class ResumableDocker(_FakeDocker):
        def run_one_shot(self, command: list[str]) -> int:
            exit_code = super().run_one_shot(command)
            shared["terminal"] = True
            return exit_code

    class ResumableStore(_FakeStore):
        def preflight_source(self, **_kwargs: Any) -> dict[str, str]:
            self.events.append("preflight")
            if shared["registered"]:
                return {"status": "already_registered"}
            return {"status": "source_verified"}

        def register_and_requeue(self, **_kwargs: Any) -> dict[str, str]:
            self.events.append("register")
            if shared["registered"]:
                return {"status": "already_registered"}
            shared["registered"] = True
            return {"status": "registered_and_queued"}

        def require_authorized_queue(self) -> None:
            self.queue_checks += 1
            self.events.append(f"queue:{self.queue_checks}")
            if shared["fail_prelaunch_once"] and self.queue_checks == 2:
                shared["fail_prelaunch_once"] = False
                raise ValueError("prelaunch mount validation failed")

        def verify_terminal_execution(self) -> dict[str, Any]:
            self.events.append("verify")
            if not shared["terminal"]:
                raise ValueError("authorized execution is not terminal")
            return {
                "job_status": "succeeded",
                "receipt_sha256": "receipt",
            }

    docker = ResumableDocker(_reference(tmp_path), events, 0)

    def store_factory(_database_url: str, *, data_root: Path) -> ResumableStore:
        assert data_root == tmp_path.resolve()
        return ResumableStore(events, target, terminal_status="succeeded")

    controller = Path(__file__).parents[1] / "scripts" / "v17_recovery_oneshot_controller.py"
    arguments = {
        "database_url": "postgresql+psycopg://quantlab:secret@host/quantlab",
        "actor": "operator",
        "external_interruption": {"exact": True},
        "controller_path": controller,
        "docker": docker,
        "store_factory": store_factory,
        "idle_check": lambda _store, _kinds: events.append("idle"),
    }

    with pytest.raises(RuntimeError, match="remain stopped"):
        run_v17_recovery(**arguments)
    assert shared["registered"] is True
    assert shared["terminal"] is False
    assert docker.running is False
    assert docker.scheduler_running is False

    result = run_v17_recovery(**arguments)

    assert result["status"] == "succeeded"
    assert result["registration_status"] == "already_registered"
    assert events.count("run") == 1
    assert docker.running is True
    assert docker.scheduler_running is True


def test_unregistered_recovery_rejects_a_stopped_reference(tmp_path: Path) -> None:
    events: list[str] = []
    docker = _FakeDocker(_reference(tmp_path, running=False), events, 0)
    docker.running = False
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    target = tmp_path.joinpath(*PurePosixPath(profile.target_artifact_path).parts[2:])

    def store_factory(_database_url: str, *, data_root: Path) -> _FakeStore:
        assert data_root == tmp_path.resolve()
        return _FakeStore(events, target, terminal_status="succeeded")

    controller = Path(__file__).parents[1] / "scripts" / "v17_recovery_oneshot_controller.py"
    with pytest.raises(ValueError, match="unregistered recovery requires"):
        run_v17_recovery(
            database_url="postgresql+psycopg://quantlab:secret@host/quantlab",
            actor="operator",
            external_interruption={"exact": True},
            controller_path=controller,
            docker=docker,
            store_factory=store_factory,
            idle_check=lambda _store, _kinds: events.append("idle"),
        )

    assert "register" not in events
    assert not any(event.startswith("start:") for event in events)


def test_registered_pending_recovery_rejects_a_running_historical_worker(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    docker = _FakeDocker(_reference(tmp_path, running=True), events, 0)
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    target = tmp_path.joinpath(*PurePosixPath(profile.target_artifact_path).parts[2:])

    class RegisteredPendingStore(_FakeStore):
        def preflight_source(self, **_kwargs: Any) -> dict[str, str]:
            self.events.append("preflight")
            return {"status": "already_registered"}

        def verify_terminal_execution(self) -> dict[str, Any]:
            self.events.append("verify")
            raise ValueError("registered recovery is still queued")

    def store_factory(_database_url: str, *, data_root: Path) -> RegisteredPendingStore:
        assert data_root == tmp_path.resolve()
        return RegisteredPendingStore(events, target, terminal_status="failed")

    controller = Path(__file__).parents[1] / "scripts" / "v17_recovery_oneshot_controller.py"
    with pytest.raises(ValueError, match="requires.*evaluation-worker.*stopped"):
        run_v17_recovery(
            database_url="postgresql+psycopg://quantlab:secret@host/quantlab",
            actor="operator",
            external_interruption={"exact": True},
            controller_path=controller,
            docker=docker,
            store_factory=store_factory,
            idle_check=lambda _store, _kinds: events.append("idle"),
        )

    assert events == ["inspect:True", "image", "preflight", "verify"]
    assert "register" not in events
    assert "run" not in events
    assert not any(event.startswith(("state:", "stop:", "start:")) for event in events)
    assert docker.running is True
    assert docker.scheduler_running is True


def test_already_terminal_replay_is_read_only_for_stopped_services(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    docker = _FakeDocker(_reference(tmp_path, running=False), events, 0)
    docker.running = False
    docker.scheduler_running = False
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    target = tmp_path.joinpath(*PurePosixPath(profile.target_artifact_path).parts[2:])

    class TerminalStore(_FakeStore):
        def preflight_source(self, **_kwargs: Any) -> dict[str, str]:
            self.events.append("preflight")
            return {"status": "already_registered"}

    def store_factory(_database_url: str, *, data_root: Path) -> TerminalStore:
        assert data_root == tmp_path.resolve()
        return TerminalStore(events, target, terminal_status="succeeded")

    controller = Path(__file__).parents[1] / "scripts" / "v17_recovery_oneshot_controller.py"
    result = run_v17_recovery(
        database_url="postgresql+psycopg://quantlab:secret@host/quantlab",
        actor="operator",
        external_interruption={"exact": True},
        controller_path=controller,
        docker=docker,
        store_factory=store_factory,
        idle_check=lambda _store, _kinds: events.append("idle"),
    )

    assert result["status"] == "already_terminal"
    assert "verify" in events
    assert "register" not in events
    assert not any(event.startswith(("start:", "stop:")) for event in events)
    assert docker.running is False
    assert docker.scheduler_running is False


def test_controller_hash_and_full_receipt_are_three_way_identical() -> None:
    controller = Path(__file__).parents[1] / "scripts" / "v17_recovery_oneshot_controller.py"
    controller_sha256 = hashlib.sha256(controller.read_bytes()).hexdigest()
    migration_path = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0086_formal_backtest_interruption_recovery.py"
    )
    spec = importlib.util.spec_from_file_location("migration_0086", migration_path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    receipt = json.loads(migration.EXPECTED_RECEIPT_JSON)
    metadata_receipt = json.loads(
        database_module._FORMAL_BACKTEST_INTERRUPTION_EXPECTED_RECEIPT_JSON
    )
    supplied = receipt.pop("receipt_sha256")
    canonical = json.dumps(
        receipt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert controller_sha256 == V17_RECOVERY_CONTROLLER_SHA256
    assert receipt["execution_controller"]["controller_sha256"] == controller_sha256
    assert hashlib.sha256(canonical).hexdigest() == supplied
    assert supplied == migration.RECEIPT_SHA256
    assert metadata_receipt == {**receipt, "receipt_sha256": supplied}


def test_controller_rejects_a_double_bound_natural_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller_path = (
        Path(__file__).parents[1] / "scripts" / "v17_recovery_oneshot_controller.py"
    )
    spec = importlib.util.spec_from_file_location("v17_controller", controller_path)
    assert spec is not None and spec.loader is not None
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    canonical = tmp_path / "canonical"
    target = tmp_path / "natural-target"
    log_path = tmp_path / "attempt-2.log"
    canonical.mkdir()
    target.mkdir()
    log_path.touch()
    monkeypatch.setattr(controller, "CANONICAL_OUTPUT_PATH", str(canonical))
    monkeypatch.setattr(controller, "TARGET_ARTIFACT_PATH", str(target))
    monkeypatch.setattr(controller, "TARGET_LOG_PATH", str(log_path))
    monkeypatch.setattr(
        controller,
        "_mount_points",
        lambda: {
            "/data": frozenset({"ro"}),
            str(canonical): frozenset({"rw"}),
            str(target): frozenset({"rw"}),
            str(log_path): frozenset({"rw"}),
        },
    )

    with pytest.raises(RuntimeError, match="same-file target"):
        controller._require_mount_contract()

    source = inspect.getsource(controller._require_mount_contract)
    assert '"ro" not in mount_points.get("/data"' in source
    assert '"rw" not in mount_points.get(str(canonical)' in source
    assert "str(target) in mount_points" in source
    assert "os.path.samefile(canonical, target)" in source
    assert "persistent-target-probe" in source
    assert '"rw" not in mount_points.get(str(log_path)' in source
