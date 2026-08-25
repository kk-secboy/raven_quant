from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from quant_platform import release_upgrade

pytestmark = pytest.mark.no_database


class FakeContext:
    project_name = "quantlab-test"
    profiles: tuple[str, ...] = ()

    def __init__(self, env_file: Path | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.env_file = env_file or Path("unused-deploy.env")
        if env_file is not None:
            env_file.write_text("POSTGRES_PASSWORD=test\n", encoding="utf-8")

    def run(self, *args: str, **_kwargs) -> str:
        self.calls.append(args)
        return ""

    def docker(self, *args: str, **_kwargs) -> str:
        self.calls.append(("docker", *args))
        return ""


def _gate(status: str = "ready", migration_state: str = "upgrade_required") -> dict:
    return {"status": status, "migration_state": migration_state, "checks": []}


def _blocked_gate(*check_ids: str, migration_state: str = "current") -> dict:
    return {
        "status": "blocked",
        "migration_state": migration_state,
        "checks": [
            {"id": check_id, "status": "block"} for check_id in check_ids
        ],
    }


def _durable_state(
    *,
    active_jobs: int = 0,
    pre_cutover_active_jobs: int = 0,
    running_units: int = 0,
    pre_cutover_running_units: int = 0,
) -> dict[str, int]:
    return {
        "active_jobs": active_jobs,
        "pre_cutover_active_jobs": pre_cutover_active_jobs,
        "post_cutover_active_jobs": active_jobs - pre_cutover_active_jobs,
        "running_units": running_units,
        "pre_cutover_running_units": pre_cutover_running_units,
        "post_cutover_running_units": running_units - pre_cutover_running_units,
    }


def test_post_start_acceptance_allows_only_new_durable_work() -> None:
    durable_state = _durable_state(active_jobs=1, running_units=2)
    accepted = release_upgrade._post_start_acceptance(
        _blocked_gate("durable_work_idle"),
        durable_state,
    )

    assert accepted == {
        "status": "pass",
        "ignored_blockers": ["durable_work_idle"],
        "blocking_checks": [],
        "durable_state": durable_state,
        "evidence": "all release checks passed; only post-cutover durable work is active",
    }


@pytest.mark.parametrize(
    "assessment",
    [
        _blocked_gate("services_healthy"),
        _blocked_gate("immutable_images_preloaded"),
        _blocked_gate("schema_compatible"),
        _blocked_gate("durable_work_idle", "database_query"),
        _blocked_gate("durable_work_idle", migration_state="upgrade_required"),
    ],
)
def test_post_start_acceptance_keeps_non_queue_checks_fail_closed(
    assessment: dict,
) -> None:
    assert (
        release_upgrade._post_start_acceptance(
            assessment,
            _durable_state(active_jobs=1),
        )["status"]
        == "block"
    )


@pytest.mark.parametrize(
    "durable_state",
    [
        _durable_state(active_jobs=1, pre_cutover_active_jobs=1),
        _durable_state(running_units=1, pre_cutover_running_units=1),
    ],
)
def test_post_start_acceptance_rejects_pre_cutover_work(
    durable_state: dict[str, int],
) -> None:
    result = release_upgrade._post_start_acceptance(
        _blocked_gate("durable_work_idle"),
        durable_state,
    )

    assert result["status"] == "block"
    assert "pre_cutover_durable_work" in result["blocking_checks"]


def test_record_cutover_uses_database_clock_after_idle_recheck() -> None:
    class CutoverContext(FakeContext):
        @staticmethod
        def running_services() -> list[str]:
            return ["postgres"]

        def run(self, *args: str, **_kwargs) -> str:
            self.calls.append(args)
            if "clock_timestamp()" in args[-1]:
                return "2026-08-22T13:14:15.123456Z\n"
            return "0|0\n"

    context = CutoverContext()

    cutover_at = release_upgrade._record_cutover(context)  # type: ignore[arg-type]

    assert cutover_at == "2026-08-22T13:14:15.123456+00:00"
    assert "clock_timestamp()" not in context.calls[0][-1]
    assert "clock_timestamp()" in context.calls[1][-1]


def test_record_cutover_rejects_active_work() -> None:
    class CutoverContext(FakeContext):
        @staticmethod
        def running_services() -> list[str]:
            return ["postgres"]

        def run(self, *args: str, **_kwargs) -> str:
            self.calls.append(args)
            return "1|0\n"

    with pytest.raises(RuntimeError, match="became active"):
        context = CutoverContext()
        release_upgrade._record_cutover(context)  # type: ignore[arg-type]

    assert len(context.calls) == 1


def test_record_cutover_rejects_a_running_writer_before_database_query() -> None:
    class CutoverContext(FakeContext):
        @staticmethod
        def running_services() -> list[str]:
            return ["postgres", "scheduler"]

    context = CutoverContext()

    with pytest.raises(RuntimeError, match="writer services are running"):
        release_upgrade._record_cutover(context)  # type: ignore[arg-type]

    assert context.calls == []


def test_post_cutover_state_uses_job_creation_and_unit_update_times() -> None:
    class CutoverContext(FakeContext):
        def run(self, *args: str, **_kwargs) -> str:
            self.calls.append(args)
            return "2|0|3|0\n"

    context = CutoverContext()

    state = release_upgrade._post_cutover_durable_state(
        context,  # type: ignore[arg-type]
        "2026-08-22T13:14:15.123456+00:00",
    )

    assert state == _durable_state(active_jobs=2, running_units=3)
    assert "created_at < TIMESTAMPTZ" in context.calls[0][-1]
    assert "GREATEST(created_at, updated_at)" in context.calls[0][-1]


def test_gateway_smoke_requires_running_gateway_and_healthy_api() -> None:
    class GatewayContext(FakeContext):
        @staticmethod
        def container_id(service: str) -> str:
            assert service == "gateway"
            return "gateway-container"

        def docker(self, *args: str, **_kwargs) -> str:
            self.calls.append(("docker", *args))
            return json.dumps([{"State": {"Status": "running", "Running": True}}])

        def run(self, *args: str, **_kwargs) -> str:
            self.calls.append(args)
            return json.dumps({"status": "ok", "database": "postgresql"})

    result = release_upgrade._gateway_smoke(GatewayContext())  # type: ignore[arg-type]

    assert result == {
        "status": "pass",
        "container_id": "gateway-container",
        "api_status": "ok",
    }


def _rollback_contract_fixture(
    tmp_path: Path,
) -> tuple[
    release_upgrade.RollbackComposeContract,
    release_upgrade.ComposeContext,
]:
    old_root = tmp_path / "old-release"
    old_deploy = old_root / "deploy"
    old_deploy.mkdir(parents=True)
    old_env = old_deploy / ".env"
    old_compose = old_deploy / "compose.yaml"
    old_env.write_text("POSTGRES_PASSWORD=old\n", encoding="utf-8")
    old_compose.write_text("services: {}\n", encoding="utf-8")
    contract = release_upgrade.RollbackComposeContract(
        project_name="quantlab-test",
        working_directory=old_root,
        env_source=old_env,
        env_content=old_env.read_bytes(),
        compose_sources=(old_compose,),
        compose_contents=(old_compose.read_bytes(),),
        profiles=(),
    )
    snapshot = tmp_path / "backup" / "rollback-compose-contract"
    snapshot.mkdir(parents=True)
    snapshot_env = snapshot / "environment.env"
    snapshot_compose = snapshot / "compose-00.yaml"
    snapshot_env.write_bytes(contract.env_content)
    snapshot_compose.write_bytes(contract.compose_contents[0])
    context = release_upgrade.ComposeContext(
        project_name="quantlab-test",
        env_file=snapshot_env,
        compose_files=(snapshot_compose,),
        project_directory=old_root,
    )
    return contract, context


def test_release_upgrade_requires_explicit_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="confirm-upgrade"):
        release_upgrade.run_release_upgrade(
            FakeContext(),  # type: ignore[arg-type]
            tmp_path,
            tmp_path / "backups",
            confirmed=False,
        )


def test_release_upgrade_stops_before_build_when_preflight_blocks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = FakeContext(tmp_path / "deploy.env")
    monkeypatch.setattr(
        release_upgrade, "assess_release", lambda *_args, **_kwargs: _gate("blocked")
    )

    result = release_upgrade.run_release_upgrade(
        context,  # type: ignore[arg-type]
        tmp_path,
        tmp_path / "backups",
        confirmed=True,
    )

    assert result["status"] == "blocked"
    assert context.calls == []


def test_release_upgrade_builds_backs_up_and_accepts_current_schema(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = FakeContext(tmp_path / "deploy.env")
    rollback_contract, rollback_context = _rollback_contract_fixture(tmp_path)
    gates = iter(
        [
            _gate(),
            _gate(),
            _gate(migration_state="current"),
            _blocked_gate("durable_work_idle"),
        ]
    )
    backup = tmp_path / "backups" / "quantlab-test"
    monkeypatch.setattr(release_upgrade, "_stamp", lambda: "20260713T040000Z")
    def assess(*_args, **_kwargs):
        context.calls.append(("assess_release",))
        return next(gates)

    monkeypatch.setattr(release_upgrade, "assess_release", assess)
    monkeypatch.setattr(
        release_upgrade,
        "_capture_rollback_images",
        lambda *_args, **_kwargs: {"api": "quantlab-rollback:test-api"},
    )
    monkeypatch.setattr(
        release_upgrade,
        "_capture_service_storage",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        release_upgrade,
        "_capture_rollback_compose_contract",
        lambda *_args, **_kwargs: rollback_contract,
    )
    monkeypatch.setattr(
        release_upgrade,
        "_persist_rollback_compose_contract",
        lambda *_args, **_kwargs: rollback_context,
    )
    monkeypatch.setattr(
        release_upgrade,
        "_assess_backup_capacity",
        lambda *_args, **_kwargs: {"status": "pass"},
    )

    def create(*_args, **kwargs) -> Path:
        assert kwargs["restart_services"] is False
        return backup

    monkeypatch.setattr(release_upgrade, "create_backup", create)
    monkeypatch.setattr(
        release_upgrade,
        "_prepare_sandbox_images",
        lambda *_args, **_kwargs: {
            "RDAGENT_RUNTIME_IMAGE_DIGEST": "sha256:" + "a" * 64
        },
    )
    cutover_at = "2026-08-22T13:14:15.123456+00:00"
    durable_state = _durable_state(active_jobs=1)
    monkeypatch.setattr(release_upgrade, "_record_cutover", lambda _context: cutover_at)
    monkeypatch.setattr(
        release_upgrade,
        "_post_cutover_durable_state",
        lambda _context, boundary: durable_state
        if boundary == cutover_at
        else pytest.fail("unexpected cutover boundary"),
    )
    monkeypatch.setattr(
        release_upgrade,
        "_gateway_smoke",
        lambda _context: {"status": "pass", "api_status": "ok"},
    )

    result = release_upgrade.run_release_upgrade(
        context,  # type: ignore[arg-type]
        tmp_path,
        tmp_path / "backups",
        confirmed=True,
        wait_timeout=45,
    )

    assert result["status"] == "succeeded"
    assert result["backup_directory"] == str(backup)
    assert result["cutover_at"] == cutover_at
    assert result["checks"]["post_upgrade_acceptance"] == {
        "status": "pass",
        "ignored_blockers": ["durable_work_idle"],
        "blocking_checks": [],
        "durable_state": durable_state,
        "evidence": "all release checks passed; only post-cutover durable work is active",
    }
    assert result["checks"]["gateway_smoke"]["status"] == "pass"
    assert ("build", *release_upgrade.BUILT_SERVICES) in context.calls
    assert ("rm", "-s", "-f", "factor-sandbox-builder") in context.calls
    core_start = next(
        index
        for index, call in enumerate(context.calls)
        if call[:3] == ("up", "-d", "--remove-orphans")
        and "api" in call
    )
    scheduler_start = next(
        index
        for index, call in enumerate(context.calls)
        if call[:3] == ("up", "-d", "--no-deps") and call[-1] == "scheduler"
    )
    gateway_start = next(
        index
        for index, call in enumerate(context.calls)
        if call[:3] == ("up", "-d", "--no-deps") and call[-1] == "gateway"
    )
    final_assessment = max(
        index for index, call in enumerate(context.calls) if call == ("assess_release",)
    )
    assert core_start < scheduler_start < final_assessment < gateway_start
    assert "scheduler" not in context.calls[core_start]
    assert "gateway" not in context.calls[core_start]


def test_release_upgrade_reuses_exact_live_backup_without_creating_an_archive(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = FakeContext(tmp_path / "deploy.env")
    rollback_contract, rollback_context = _rollback_contract_fixture(tmp_path)
    gates = iter(
        [
            _gate(),
            _gate(),
            _gate(migration_state="current"),
            _gate(migration_state="current"),
        ]
    )
    backup_root = tmp_path / "backups"
    backup = backup_root / "quantlab-existing"
    backup.mkdir(parents=True)
    reusable = release_upgrade.ReusableBackup(
        directory=backup,
        manifest={
            "schema_revision": "0058_simulation_benchmark",
            "created_at": "2026-08-21T14:22:37+00:00",
        },
        data_source=tmp_path / "data",
        expected_data_bytes=123,
        observed_data_bytes=123,
        size_tolerance_bytes=1024**2,
    )
    reusable.data_source.mkdir()
    monkeypatch.setattr(release_upgrade, "_stamp", lambda: "20260822T130000Z")
    monkeypatch.setattr(
        release_upgrade,
        "assess_release",
        lambda *_args, **_kwargs: next(gates),
    )
    monkeypatch.setattr(
        release_upgrade,
        "_capture_rollback_compose_contract",
        lambda *_args, **_kwargs: rollback_contract,
    )
    monkeypatch.setattr(
        release_upgrade,
        "_capture_service_storage",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        release_upgrade,
        "_capture_rollback_images",
        lambda *_args, **_kwargs: {"api": "quantlab-rollback:test-api"},
    )
    monkeypatch.setattr(
        release_upgrade,
        "_validate_reusable_backup",
        lambda *_args, **_kwargs: reusable,
    )
    monkeypatch.setattr(
        release_upgrade,
        "_stop_writers_for_backup_reuse",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        release_upgrade,
        "_assert_no_data_file_newer",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        release_upgrade,
        "_data_usage_bytes",
        lambda *_args, **_kwargs: 123,
    )

    def persist(_contract, directory, *, directory_name):
        assert directory == backup
        assert directory_name == "rollback-compose-contract-20260822t130000z"
        return rollback_context

    monkeypatch.setattr(release_upgrade, "_persist_rollback_compose_contract", persist)
    monkeypatch.setattr(
        release_upgrade,
        "create_backup",
        lambda *_args, **_kwargs: pytest.fail("reuse must not create a new archive"),
    )
    monkeypatch.setattr(
        release_upgrade,
        "_prepare_sandbox_images",
        lambda *_args, **_kwargs: {
            "RDAGENT_RUNTIME_IMAGE_DIGEST": "sha256:" + "a" * 64
        },
    )
    monkeypatch.setattr(
        release_upgrade,
        "_record_cutover",
        lambda _context: "2026-08-22T13:14:15.123456+00:00",
    )
    monkeypatch.setattr(
        release_upgrade,
        "_post_cutover_durable_state",
        lambda *_args: _durable_state(),
    )
    monkeypatch.setattr(
        release_upgrade,
        "_gateway_smoke",
        lambda _context: {"status": "pass", "api_status": "ok"},
    )

    result = release_upgrade.run_release_upgrade(
        context,  # type: ignore[arg-type]
        tmp_path,
        backup_root,
        confirmed=True,
        wait_timeout=45,
        reuse_backup=backup,
    )

    assert result["status"] == "succeeded"
    assert result["backup_reused"] is True
    assert result["backup_directory"] == str(backup)
    assert result["checks"]["reused_backup"]["status"] == "pass"


def test_reusable_backup_must_be_inside_backup_root(monkeypatch, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    backup_root = tmp_path / "backups"
    backup_root.mkdir()
    contract, _context = _rollback_contract_fixture(tmp_path)
    monkeypatch.setattr(
        release_upgrade,
        "load_and_verify_manifest",
        lambda *_args, **_kwargs: pytest.fail("outside paths fail before manifest reads"),
    )

    with pytest.raises(ValueError, match="inside backup_root"):
        release_upgrade._validate_reusable_backup(
            FakeContext(tmp_path / "current.env"),  # type: ignore[arg-type]
            backup_root,
            outside,
            contract,
        )


def test_reusable_backup_validation_enables_the_verification_receipt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backup_root = tmp_path / "backups"
    backup = backup_root / "quantlab-existing"
    backup.mkdir(parents=True)
    contract, _context = _rollback_contract_fixture(tmp_path)

    def load(_path: Path, *, use_verification_receipt: bool = False) -> dict:
        assert use_verification_receipt is True
        raise RuntimeError("receipt wiring observed")

    monkeypatch.setattr(release_upgrade, "load_and_verify_manifest", load)

    with pytest.raises(RuntimeError, match="receipt wiring observed"):
        release_upgrade._validate_reusable_backup(
            FakeContext(tmp_path / "current.env"),  # type: ignore[arg-type]
            backup_root,
            backup,
            contract,
        )


def test_reuse_contract_allows_the_generated_rollback_override(tmp_path: Path) -> None:
    backup = tmp_path / "backup"
    contract_root = backup / "rollback-compose-contract"
    contract_root.mkdir(parents=True)
    environment = contract_root / "environment.env"
    compose = contract_root / "compose-00.yaml"
    override = contract_root / "rollback.override.json"
    environment.write_text("POSTGRES_PASSWORD=old\n", encoding="utf-8")
    compose.write_text("services: {}\n", encoding="utf-8")
    override.write_text('{"services": {}}\n', encoding="utf-8")
    (contract_root / "manifest.json").write_text(
        json.dumps(
            {
                "env_snapshot": environment.name,
                "env_sha256": hashlib.sha256(environment.read_bytes()).hexdigest(),
                "compose_files": [
                    {
                        "snapshot": compose.name,
                        "sha256": hashlib.sha256(compose.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    contract = release_upgrade.RollbackComposeContract(
        project_name="quantlab-test",
        working_directory=tmp_path,
        env_source=environment,
        env_content=environment.read_bytes(),
        compose_sources=(compose, override),
        compose_contents=(compose.read_bytes(), override.read_bytes()),
        profiles=(),
    )

    release_upgrade._verify_reuse_contract_source(contract, backup)


def test_backup_reuse_restarts_old_writers_when_queue_race_is_detected(
    monkeypatch,
) -> None:
    class StopContext(FakeContext):
        def __init__(self) -> None:
            super().__init__()
            self.running_checks = 0

        def running_services(self) -> list[str]:
            self.running_checks += 1
            return ["api", "worker"] if self.running_checks == 1 else []

    context = StopContext()
    monkeypatch.setattr(
        release_upgrade,
        "_reuse_database_state",
        lambda *_args, **_kwargs: (1, 0, "0058_simulation_benchmark"),
    )

    with pytest.raises(RuntimeError, match="durable work became active"):
        release_upgrade._stop_writers_for_backup_reuse(
            context,  # type: ignore[arg-type]
            expected_schema_revision="0058_simulation_benchmark",
        )

    stop_call = next(call for call in context.calls if call[0] == "stop")
    start_call = next(call for call in context.calls if call[0] == "start")
    assert set(stop_call[1:]) == {"api", "worker"}
    assert set(start_call[1:]) == {"api", "worker"}


def test_release_build_targets_cover_data_science_and_optional_gpu() -> None:
    context = FakeContext()
    gpu_context = FakeContext()
    gpu_context.profiles = ("gpu",)

    assert "rdagent-data-science-worker" in release_upgrade._built_services(context)  # type: ignore[arg-type]
    assert "rdagent-llm-finetune-worker" not in release_upgrade._built_services(context)  # type: ignore[arg-type]
    assert "rdagent-llm-finetune-worker" in release_upgrade._built_services(gpu_context)  # type: ignore[arg-type]


def test_rollback_disables_a_service_absent_from_the_previous_release(
    tmp_path: Path,
) -> None:
    override = tmp_path / "rollback.json"

    release_upgrade._rollback_override(
        override,
        {"api": "quantlab-rollback:20260713t040000z-api"},
        frozenset({"rdagent-data-science-worker"}),
    )

    payload = __import__("json").loads(override.read_text(encoding="utf-8"))
    assert payload["services"]["rdagent-data-science-worker"] == {
        "profiles": ["rollback-disabled"]
    }


def test_rollback_context_removes_gpu_profile_when_old_worker_is_absent(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    compose_file = tmp_path / "compose.yaml"
    override_file = tmp_path / "rollback.json"
    context = release_upgrade.ComposeContext(
        project_name="quantlab-test",
        env_file=env_file,
        compose_files=(compose_file,),
        profiles=("gpu",),
    )

    rollback = release_upgrade._rollback_context(
        context,
        override_file,
        disabled_services=frozenset({"rdagent-llm-finetune-worker"}),
    )
    data_science_only = release_upgrade._rollback_context(
        context,
        override_file,
        disabled_services=frozenset({"rdagent-data-science-worker"}),
    )

    assert rollback.profiles == ()
    assert rollback.compose_files[-1] == override_file.resolve()
    assert data_science_only.profiles == ("gpu",)


def test_rollback_override_restores_the_previous_dind_volume(tmp_path: Path) -> None:
    override = tmp_path / "rollback.json"

    release_upgrade._rollback_override(
        override,
        {},
        rdagent_docker_storage={
            "type": "volume",
            "source": "quantlab-platform_rdagent_docker",
            "target": "/var/lib/docker",
        },
    )

    payload = __import__("json").loads(override.read_text(encoding="utf-8"))
    assert payload["volumes"]["rdagent_docker_rollback"] == {
        "external": True,
        "name": "quantlab-platform_rdagent_docker",
    }
    assert payload["services"]["rdagent-docker"]["volumes"][0]["target"] == (
        "/var/lib/docker"
    )


def test_rollback_factor_builder_uses_old_worker_and_dind_images(
    tmp_path: Path,
) -> None:
    override = tmp_path / "rollback.json"
    old_worker = "quantlab-rollback:release-worker"
    old_dind = "quantlab-rollback:release-rdagent-docker"

    release_upgrade._rollback_override(
        override,
        {"worker": old_worker, "rdagent-docker": old_dind},
    )

    builder = json.loads(override.read_text(encoding="utf-8"))["services"][
        "factor-sandbox-builder"
    ]
    assert builder["image"] == old_dind
    assert builder["environment"]["FACTOR_SANDBOX_BASE_IMAGE"] == old_worker
    assert "build" not in builder
    payload = json.loads(override.read_text(encoding="utf-8"))
    assert all("build" not in service for service in payload["services"].values())


def test_release_environment_update_preserves_unrelated_secrets(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "POSTGRES_PASSWORD=keep-me\nRDAGENT_RUNTIME_IMAGE_DIGEST=old\n",
        encoding="utf-8",
    )

    release_upgrade._update_environment(
        env_file,
        {
            "RDAGENT_RUNTIME_IMAGE_DIGEST": "sha256:" + "a" * 64,
            "MODEL_SANDBOX_IMAGE": "registry/model@sha256:" + "b" * 64,
        },
    )

    content = env_file.read_text(encoding="utf-8")
    assert "POSTGRES_PASSWORD=keep-me" in content
    assert "RDAGENT_RUNTIME_IMAGE_DIGEST=sha256:" + "a" * 64 in content
    assert "MODEL_SANDBOX_IMAGE=registry/model@sha256:" + "b" * 64 in content


def test_governed_sandboxes_are_network_free_and_reuse_pinned_worker() -> None:
    root = Path(__file__).resolve().parents[1]

    evidence = release_upgrade._governed_sandbox_evidence(root)

    forbidden = (
        "git clone",
        "git fetch",
        "git reset",
        "apt-get",
        "pip install",
        "curl ",
        "wget ",
        ":latest",
    )
    for directory in (
        "qlib-sandbox",
        "data-science-sandbox",
        "model-sandbox",
    ):
        dockerfile = (root / "deploy" / directory / "Dockerfile").read_text(
            encoding="utf-8"
        )
        assert not any(token in dockerfile for token in forbidden)
        assert "FROM ${" in dockerfile

    worker = (root / "deploy" / "Dockerfile.worker").read_text(encoding="utf-8")
    qlib = (root / "deploy" / "qlib-sandbox" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert evidence["qlib"]["qlib_commit"] == release_upgrade._QLIB_COMMIT
    assert evidence["qlib"]["source_path"] == "/opt/qlib"
    assert "/opt/qlib" in qlib
    assert "torch==2.2.2+cpu" in worker
    assert "catboost==1.2.8" in worker
    assert "xgboost-cpu==2.1.4" in worker
    assert '"xgboost==2.1.4"' not in worker
    assert "forbidden accelerator distributions" in worker
    assert "scikit-learn==1.7.1" in worker
    assert "numpy==1.26.4" in worker
    assert "Cython==3.0.11" in worker
    assert "cvxpy==1.7.5" in worker
    assert "pip uninstall --yes sparsediffpy" in worker
    assert "python -m pip check" in worker
    assert "from qlib.contrib.strategy import TopkDropoutStrategy" in worker
    assert "--no-build-isolation --no-deps --force-reinstall /opt/qlib" in worker
    assert worker.rfind("python -m pip check") > worker.rfind("python -m pip install .")
    final_worker_gate = worker[worker.rfind("python -m pip install .") :]
    assert "np.__version__ == '1.26.4'" in final_worker_gate
    assert "torch.__version__ == '2.2.2+cpu'" in final_worker_gate
    assert "torch.version.cuda is None" in final_worker_gate
    assert "torch.from_numpy" in final_worker_gate
    assert "rolling_mean" in final_worker_gate
    assert "torch.from_numpy" in worker
    assert "rolling_mean" in worker
    assert "torch.from_numpy" in qlib


def test_host_build_publishes_sealed_image_and_dind_only_pulls(
    tmp_path: Path,
) -> None:
    host_repository = "127.0.0.1:55000/quantlab/qlib-sandbox"
    dind_repository = "rdagent-registry:5000/quantlab/qlib-sandbox"
    digest = "sha256:" + "a" * 64

    class BuildContext(FakeContext):
        def docker(self, *args: str, **_kwargs) -> str:
            self.calls.append(("docker", *args))
            if "{{json .RepoDigests}}" in args:
                return f'["{host_repository}@{digest}"]'
            return ""

    context = BuildContext()
    context_root = tmp_path / "qlib"
    context_root.mkdir()

    image = release_upgrade._build_and_publish_host_image(
        context,  # type: ignore[arg-type]
        context_root=context_root,
        host_image_tag=f"{host_repository}:release",
        host_repository=host_repository,
        dind_repository=dind_repository,
        timeout=60,
        build_args=("QLIB_SANDBOX_BASE_IMAGE=base@sha256:" + "b" * 64,),
    )

    host_build = context.calls[0]
    assert host_build[:2] == ("docker", "build")
    assert host_build[2:4] == ("--network", "none")
    assert "QLIB_SANDBOX_BASE_IMAGE=base@sha256:" + "b" * 64 in host_build
    assert host_build[-1] == str(context_root.resolve())
    assert not any(
        call[:3] == ("exec", "-T", "rdagent-docker") and "build" in call
        for call in context.calls
    )
    assert (
        "exec",
        "-T",
        "rdagent-docker",
        "docker",
        "pull",
        f"{dind_repository}@{digest}",
    ) in context.calls
    assert image == f"{dind_repository}@{digest}"


def test_dind_smoke_runs_with_network_disabled() -> None:
    context = FakeContext()

    release_upgrade._dind_smoke(
        context,  # type: ignore[arg-type]
        "registry/sandbox@sha256:" + "a" * 64,
        ("python", "-c", "print('ok')"),
        timeout=60,
    )

    assert ("--network", "none") == context.calls[0][6:8]


def test_rollback_contract_persists_old_compose_env_and_external_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    new_root = tmp_path / "new-release"
    old_root = tmp_path / "old-release"
    backup_root = tmp_path / "backups"
    old_deploy = old_root / "deploy"
    old_deploy.mkdir(parents=True)
    backup_root.mkdir()
    current_env = new_root / "deploy" / ".env"
    current_env.parent.mkdir(parents=True)
    current_env.write_text("POSTGRES_PASSWORD=new\n", encoding="utf-8")
    old_env = old_deploy / ".env"
    old_env.write_text("POSTGRES_PASSWORD=old\n", encoding="utf-8")
    old_compose = old_deploy / "compose.yaml"
    old_compose.write_text(
        "services:\n"
        "  api:\n"
        "    image: old-api\n"
        "    healthcheck:\n"
        "      test: ['CMD', 'old-health']\n"
        "  web:\n"
        "    image: old-web\n",
        encoding="utf-8",
    )
    external_override = backup_root / "old.rollback.compose.json"
    external_override.write_text(
        '{"services":{"api":{"image":"old-api@sha256:abc"}}}\n',
        encoding="utf-8",
    )
    external_override.chmod(0o600)
    external_env = backup_root / "environment.env"
    external_env.write_bytes(old_env.read_bytes())
    external_env.chmod(0o600)
    files_label = f"{old_compose.resolve()},{external_override.resolve()}"

    class LabelContext(FakeContext):
        def container_id(self, service: str) -> str:
            return f"container-{service}"

        def docker(self, *args: str, **_kwargs) -> str:
            service = args[1].removeprefix("container-")
            return json.dumps(
                [
                    {
                        "Config": {
                            "Labels": {
                                "com.docker.compose.project": self.project_name,
                                "com.docker.compose.service": service,
                                "com.docker.compose.project.working_dir": str(
                                    old_deploy.resolve()
                                ),
                                "com.docker.compose.project.config_files": files_label,
                                "com.docker.compose.project.environment_file": str(
                                    external_env.resolve()
                                ),
                            }
                        }
                    }
                ]
            )

    monkeypatch.setattr(
        release_upgrade.ComposeContext,
        "run",
        lambda *_args, **_kwargs: "",
    )
    contract = release_upgrade._capture_rollback_compose_contract(
        LabelContext(current_env),  # type: ignore[arg-type]
        new_root,
        services=("api", "web"),
        trusted_external_root=backup_root,
    )
    backup = backup_root / "quantlab-test"
    backup.mkdir()
    rollback_context = release_upgrade._persist_rollback_compose_contract(
        contract,
        backup,
    )

    assert contract.env_source == external_env.resolve()
    assert rollback_context.project_directory == old_deploy.resolve()
    assert rollback_context.env_file.read_bytes() == old_env.read_bytes()
    assert [item.read_bytes() for item in rollback_context.compose_files] == [
        old_compose.read_bytes(),
        external_override.read_bytes(),
    ]
    manifest = json.loads(
        (rollback_context.env_file.parent / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["working_directory"] == str(old_deploy.resolve())
    assert len(manifest["compose_files"]) == 2
    assert all(len(item["sha256"]) == 64 for item in manifest["compose_files"])


def test_rollback_contract_falls_back_to_environment_in_compose_working_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    new_root = tmp_path / "new-release"
    old_deploy = tmp_path / "old-release" / "deploy"
    old_deploy.mkdir(parents=True)
    current_env = new_root / "deploy" / ".env"
    current_env.parent.mkdir(parents=True)
    current_env.write_text("POSTGRES_PASSWORD=new\n", encoding="utf-8")
    old_env = old_deploy / ".env"
    old_env.write_text("POSTGRES_PASSWORD=old\n", encoding="utf-8")
    old_compose = old_deploy / "compose.yaml"
    old_compose.write_text("services: {}\n", encoding="utf-8")

    class LabelContext(FakeContext):
        def container_id(self, service: str) -> str:
            return f"container-{service}"

        def docker(self, *args: str, **_kwargs) -> str:
            service = args[1].removeprefix("container-")
            return json.dumps(
                [
                    {
                        "Config": {
                            "Labels": {
                                "com.docker.compose.project": self.project_name,
                                "com.docker.compose.service": service,
                                "com.docker.compose.project.working_dir": str(
                                    old_deploy.resolve()
                                ),
                                "com.docker.compose.project.config_files": str(
                                    old_compose.resolve()
                                ),
                            }
                        }
                    }
                ]
            )

    monkeypatch.setattr(
        release_upgrade.ComposeContext,
        "run",
        lambda *_args, **_kwargs: "",
    )

    contract = release_upgrade._capture_rollback_compose_contract(
        LabelContext(current_env),  # type: ignore[arg-type]
        new_root,
        services=("api",),
    )

    assert contract.working_directory == old_deploy.resolve()
    assert contract.env_source == old_env.resolve()
    assert contract.env_content == old_env.read_bytes()


def test_rollback_contract_rejects_labeled_environment_outside_trusted_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    new_root = tmp_path / "new-release"
    old_deploy = tmp_path / "old-release" / "deploy"
    backup_root = tmp_path / "backups"
    outside_root = tmp_path / "outside"
    old_deploy.mkdir(parents=True)
    backup_root.mkdir()
    outside_root.mkdir()
    current_env = new_root / "deploy" / ".env"
    current_env.parent.mkdir(parents=True)
    current_env.write_text("POSTGRES_PASSWORD=new\n", encoding="utf-8")
    old_compose = old_deploy / "compose.yaml"
    old_compose.write_text("services: {}\n", encoding="utf-8")
    external_env = outside_root / "environment.env"
    external_env.write_text("POSTGRES_PASSWORD=old\n", encoding="utf-8")
    external_env.chmod(0o600)

    class LabelContext(FakeContext):
        def container_id(self, service: str) -> str:
            return f"container-{service}"

        def docker(self, *args: str, **_kwargs) -> str:
            service = args[1].removeprefix("container-")
            return json.dumps(
                [
                    {
                        "Config": {
                            "Labels": {
                                "com.docker.compose.project": self.project_name,
                                "com.docker.compose.service": service,
                                "com.docker.compose.project.working_dir": str(
                                    old_deploy.resolve()
                                ),
                                "com.docker.compose.project.config_files": str(
                                    old_compose.resolve()
                                ),
                                "com.docker.compose.project.environment_file": str(
                                    external_env.resolve()
                                ),
                            }
                        }
                    }
                ]
            )

    monkeypatch.setattr(
        release_upgrade.ComposeContext,
        "run",
        lambda *_args, **_kwargs: "",
    )

    with pytest.raises(RuntimeError, match="outside the trusted backup root"):
        release_upgrade._capture_rollback_compose_contract(
            LabelContext(current_env),  # type: ignore[arg-type]
            new_root,
            services=("api",),
            trusted_external_root=backup_root,
        )


def test_rollback_contract_rejects_inconsistent_labeled_environments(
    monkeypatch,
    tmp_path: Path,
) -> None:
    new_root = tmp_path / "new-release"
    old_deploy = tmp_path / "old-release" / "deploy"
    backup_root = tmp_path / "backups"
    old_deploy.mkdir(parents=True)
    backup_root.mkdir()
    current_env = new_root / "deploy" / ".env"
    current_env.parent.mkdir(parents=True)
    current_env.write_text("POSTGRES_PASSWORD=new\n", encoding="utf-8")
    old_compose = old_deploy / "compose.yaml"
    old_compose.write_text("services: {}\n", encoding="utf-8")
    environments = {}
    for service in ("api", "web"):
        environment = backup_root / f"{service}.env"
        environment.write_text("POSTGRES_PASSWORD=old\n", encoding="utf-8")
        environment.chmod(0o600)
        environments[service] = environment

    class LabelContext(FakeContext):
        def container_id(self, service: str) -> str:
            return f"container-{service}"

        def docker(self, *args: str, **_kwargs) -> str:
            service = args[1].removeprefix("container-")
            return json.dumps(
                [
                    {
                        "Config": {
                            "Labels": {
                                "com.docker.compose.project": self.project_name,
                                "com.docker.compose.service": service,
                                "com.docker.compose.project.working_dir": str(
                                    old_deploy.resolve()
                                ),
                                "com.docker.compose.project.config_files": str(
                                    old_compose.resolve()
                                ),
                                "com.docker.compose.project.environment_file": str(
                                    environments[service].resolve()
                                ),
                            }
                        }
                    }
                ]
            )

    monkeypatch.setattr(
        release_upgrade.ComposeContext,
        "run",
        lambda *_args, **_kwargs: "",
    )

    with pytest.raises(RuntimeError, match="do not share one Compose release contract"):
        release_upgrade._capture_rollback_compose_contract(
            LabelContext(current_env),  # type: ignore[arg-type]
            new_root,
            services=("api", "web"),
            trusted_external_root=backup_root,
        )


def test_rollback_restore_refuses_to_continue_while_new_writer_runs() -> None:
    class RunningWriterContext(FakeContext):
        def docker(self, *args: str, **_kwargs) -> str:
            self.calls.append(("docker", *args))
            return "rdagent-data-science-worker"

    context = RunningWriterContext()

    with pytest.raises(RuntimeError, match="still running"):
        release_upgrade._stop_and_verify_new_writers(context)  # type: ignore[arg-type]

    assert context.calls[0][0] == "stop"


def test_rollback_reuses_the_root_owned_backup_verification_receipt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = FakeContext(tmp_path / "new.env")
    _contract, rollback_base_context = _rollback_contract_fixture(tmp_path)
    backup = tmp_path / "backups" / "quantlab-test"
    backup.mkdir(parents=True)
    manifest_calls: list[bool] = []
    restore_calls: list[bool] = []
    events: list[str] = []
    rollback_helper_disabled: list[frozenset[str]] = []

    def load(_path: Path, *, use_verification_receipt: bool = False) -> dict:
        events.append("verify")
        manifest_calls.append(use_verification_receipt)
        return {"schema_revision": "0064_paper_stage_account"}

    def restore(
        _context,
        _path: Path,
        *,
        confirmed: bool,
        minimum_free_gb: float,
        use_verification_receipt: bool = False,
    ) -> str:
        assert confirmed is True
        assert minimum_free_gb == 20
        restore_calls.append(use_verification_receipt)
        return "0064_paper_stage_account"

    class RollbackContext(FakeContext):
        def running_services(self) -> list[str]:
            return ["postgres", "api"]

    rollback = RollbackContext(tmp_path / "rollback.env")
    monkeypatch.setattr(release_upgrade, "load_and_verify_manifest", load)
    monkeypatch.setattr(
        release_upgrade,
        "_stop_and_verify_new_writers",
        lambda *_args, **_kwargs: events.append("stop_writers"),
    )
    def rollback_override(*_args, **kwargs) -> None:
        rollback_helper_disabled.append(kwargs["disabled_services"])

    def rollback_context(*_args, **kwargs):
        rollback_helper_disabled.append(kwargs["disabled_services"])
        return rollback

    monkeypatch.setattr(release_upgrade, "_rollback_override", rollback_override)
    monkeypatch.setattr(release_upgrade, "_rollback_context", rollback_context)
    monkeypatch.setattr(release_upgrade, "restore_backup", restore)

    result = release_upgrade._restore_previous_release(
        context,  # type: ignore[arg-type]
        rollback_base_context,
        backup,
        {"api": "quantlab-rollback:test-api"},
        wait_timeout=45,
        minimum_free_gb=20,
        disabled_services=frozenset({"rdagent-data-science-worker"}),
    )

    assert manifest_calls == [True]
    assert restore_calls == [True]
    assert events[:2] == ["stop_writers", "verify"]
    assert rollback_helper_disabled == [
        frozenset({"rdagent-data-science-worker"}),
        frozenset({"rdagent-data-science-worker"}),
    ]
    assert result["schema_revision"] == "0064_paper_stage_account"


def test_qlib_provider_candidates_keep_newest_first_discovery_order() -> None:
    assert release_upgrade._qlib_provider_candidates(
        [
            "/data/qlib/20260820/calendars/day.txt",
            "/data/qlib/20260819/calendars/day.txt",
            "noise",
        ]
    ) == [
        "/data/qlib/20260820",
        "/data/qlib/20260819",
    ]


def test_release_upgrade_restores_backup_and_old_images_on_failed_acceptance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = FakeContext(tmp_path / "deploy.env")
    rollback_contract, rollback_context = _rollback_contract_fixture(tmp_path)
    gates = iter(
        [
            _gate(),
            _gate(),
            _gate(migration_state="current"),
            _blocked_gate("services_healthy"),
        ]
    )
    backup = tmp_path / "backups" / "quantlab-test"
    tags = {"api": "quantlab-rollback:test-api"}
    monkeypatch.setattr(
        release_upgrade,
        "assess_release",
        lambda *_args, **_kwargs: next(gates),
    )
    monkeypatch.setattr(
        release_upgrade,
        "_capture_rollback_images",
        lambda *_args, **_kwargs: tags,
    )
    monkeypatch.setattr(
        release_upgrade,
        "_capture_service_storage",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        release_upgrade,
        "_capture_rollback_compose_contract",
        lambda *_args, **_kwargs: rollback_contract,
    )
    monkeypatch.setattr(
        release_upgrade,
        "_persist_rollback_compose_contract",
        lambda *_args, **_kwargs: rollback_context,
    )
    monkeypatch.setattr(
        release_upgrade,
        "_assess_backup_capacity",
        lambda *_args, **_kwargs: {"status": "pass"},
    )
    monkeypatch.setattr(release_upgrade, "create_backup", lambda *_args, **_kwargs: backup)
    monkeypatch.setattr(
        release_upgrade,
        "_prepare_sandbox_images",
        lambda *_args, **_kwargs: {
            "RDAGENT_RUNTIME_IMAGE_DIGEST": "sha256:" + "a" * 64
        },
    )
    monkeypatch.setattr(
        release_upgrade,
        "_record_cutover",
        lambda _context: "2026-08-22T13:14:15.123456+00:00",
    )
    monkeypatch.setattr(
        release_upgrade,
        "_post_cutover_durable_state",
        lambda *_args: _durable_state(),
    )
    rollbacks: list[Path] = []

    def rollback(_context, old_context, backup_directory, rollback_tags, **kwargs):
        assert old_context == rollback_context
        assert rollback_tags == tags
        assert kwargs["disabled_services"] == frozenset(
            {"rdagent-data-science-worker"}
        )
        rollbacks.append(backup_directory)
        return {"schema_revision": "0019", "images": tags}

    monkeypatch.setattr(release_upgrade, "_restore_previous_release", rollback)

    result = release_upgrade.run_release_upgrade(
        context,  # type: ignore[arg-type]
        tmp_path,
        tmp_path / "backups",
        confirmed=True,
        wait_timeout=45,
    )

    assert result["status"] == "rolled_back"
    assert rollbacks == [backup]
    assert "post-upgrade release acceptance" in result["error"]


def test_rollback_image_pruning_keeps_newest_release_sets() -> None:
    class Images(FakeContext):
        def docker(self, *args: str, **_kwargs) -> str:
            self.calls.append(("docker", *args))
            if args[:2] == ("image", "ls"):
                return "\n".join(
                    f"quantlab-rollback:{release}-{service}"
                    for release in (
                        "20260713t010000z",
                        "20260713t020000z",
                        "20260713t030000z",
                    )
                    for service in ("api", "web")
                )
            return ""

    context = Images()

    removed = release_upgrade._prune_rollback_images(
        context,  # type: ignore[arg-type]
        2,
    )

    assert removed == [
        "quantlab-rollback:20260713t010000z-api",
        "quantlab-rollback:20260713t010000z-web",
    ]
    assert ("docker", "image", "rm", "-f", *removed) in context.calls


def test_backup_capacity_requires_full_data_copy_plus_headroom(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class CapacityContext(FakeContext):
        @staticmethod
        def data_volume() -> str:
            return "/data/quantlab"

        def docker(self, *args: str, **_kwargs) -> str:
            self.calls.append(("docker", *args))
            return f"{100 * 1024**2}\t/source"

    disk_usage = type("Usage", (), {"free": 110 * 1024**3})()
    monkeypatch.setattr(release_upgrade.shutil, "disk_usage", lambda _path: disk_usage)
    backup_root = tmp_path / "backups"

    result = release_upgrade._assess_backup_capacity(
        CapacityContext(),  # type: ignore[arg-type]
        backup_root,
        minimum_free_gb=20.0,
    )

    assert result["status"] == "block"
    assert "source /data/quantlab" in result["evidence"]
    assert "data upper bound 100.0 GiB" in result["evidence"]
    assert "required 120.0 GiB" in result["evidence"]


def test_release_upgrade_blocks_before_image_capture_when_backup_target_is_small(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = FakeContext()
    monkeypatch.setattr(
        release_upgrade,
        "assess_release",
        lambda *_args, **_kwargs: _gate(),
    )
    monkeypatch.setattr(
        release_upgrade,
        "_assess_backup_capacity",
        lambda *_args, **_kwargs: {"status": "block", "evidence": "too small"},
    )

    result = release_upgrade.run_release_upgrade(
        context,  # type: ignore[arg-type]
        tmp_path,
        tmp_path / "backups",
        confirmed=True,
    )

    assert result["status"] == "blocked"
    assert result["checks"]["backup_capacity"]["evidence"] == "too small"
    assert context.calls == []
