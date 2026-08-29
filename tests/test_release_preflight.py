from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_platform import release_preflight
from quant_platform.release_identity import (
    release_identity_environment,
    release_identity_labels,
)

pytestmark = pytest.mark.no_database
_REAL_IMMUTABLE_IMAGE_CONFIGURATION = release_preflight._immutable_image_configuration
_REAL_DOCKER_STORAGE_CONFIGURATION = release_preflight._docker_storage_configuration
_REAL_PRELOADED_IMAGE_AVAILABILITY = release_preflight._preloaded_image_availability
_REAL_RUNTIME_RELEASE_IDENTITY = release_preflight._runtime_release_identity


@pytest.fixture(autouse=True)
def _configured_immutable_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        release_preflight,
        "_immutable_image_configuration",
        lambda _context: (True, "test image identities are pinned"),
    )
    monkeypatch.setattr(
        release_preflight,
        "_preloaded_image_availability",
        lambda _context: (True, "test images are preloaded"),
    )
    monkeypatch.setattr(
        release_preflight,
        "_docker_storage_configuration",
        lambda _context: (True, "test storage is isolated"),
    )
    monkeypatch.setattr(
        release_preflight,
        "_runtime_release_identity",
        lambda _context, services: (
            True,
            f"{len(set(services) - {'postgres'})} services share the test release",
        ),
    )


def _expected_migration_head(project_root: Path) -> str:
    """Latest alembic revision: the one nobody lists as down_revision."""
    revisions: dict[str, str | None] = {}
    for path in sorted((project_root / "migrations" / "versions").glob("*.py")):
        values: dict[str, object] = {}
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and target.id in {"revision", "down_revision"}:
                values[target.id] = ast.literal_eval(node.value)
        revision = values.get("revision")
        if isinstance(revision, str):
            down = values.get("down_revision")
            revisions[revision] = down if isinstance(down, str) else None
    heads = [rev for rev, down in revisions.items() if rev not in set(revisions.values()) - {None}]
    assert len(heads) == 1, f"expected a single migration head, got {heads}"
    return heads[0]


class FakeComposeContext:
    project_name = "quantlab-test"

    def __init__(self, database_row: str) -> None:
        self.database_row = database_row

    def container_id(self, service: str) -> str:
        return "postgres-id" if service == "postgres" else ""

    def run(self, *args: str, capture: bool = False) -> str:
        assert capture is True
        if args == ("config", "--quiet"):
            return ""
        if args[:3] == ("exec", "-T", "postgres"):
            return self.database_row
        if args == ("ps", "--format", "json"):
            rows = []
            for service in sorted(release_preflight.EXPECTED_SERVICES):
                row = {"Service": service, "State": "running"}
                if service in release_preflight.HEALTHCHECK_SERVICES:
                    row["Health"] = "healthy"
                rows.append(row)
            return json.dumps(rows)
        raise AssertionError(f"unexpected compose call: {args}")


class InvalidComposeContext:
    project_name = "quantlab-invalid"

    def run(self, *args: str, capture: bool = False) -> str:
        assert capture is True
        if args == ("config", "--quiet"):
            raise RuntimeError("PLATFORM_SECRET_KEY is required")
        raise AssertionError(f"invalid Compose must not be queried further: {args}")

    def container_id(self, service: str) -> str:
        raise AssertionError(f"invalid Compose must not inspect {service}")

    def docker(self, *args: str, capture: bool = False) -> str:
        assert capture is True
        if args[:3] == ("ps", "-aq", "--filter"):
            return ""
        raise AssertionError(f"unexpected Docker fallback call: {args}")


class InvalidComposeWithRunningProject(InvalidComposeContext):
    def docker(self, *args: str, capture: bool = False) -> str:
        assert capture is True
        if args[:3] == ("ps", "-aq", "--filter"):
            return "container-postgres\ncontainer-api"
        if args[:2] == ("inspect", "container-postgres"):
            rows = []
            for service in sorted(release_preflight.EXPECTED_SERVICES):
                state = {"Status": "running"}
                if service in release_preflight.HEALTHCHECK_SERVICES:
                    state["Health"] = {"Status": "healthy"}
                rows.append(
                    {
                        "Id": f"container-{service}",
                        "Config": {
                            "Labels": {"com.docker.compose.service": service},
                        },
                        "State": state,
                    }
                )
            return json.dumps(rows)
        if args[:2] == ("exec", "container-postgres"):
            return "0|0|0|0|0017_qmt_execution_state"
        raise AssertionError(f"unexpected Docker fallback call: {args}")


def test_assess_release_ready_when_deployment_is_idle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        release_preflight,
        "_schema_compatibility",
        lambda _root, _revision: ("0020_strategy_allocations", "current", True),
    )
    result = release_preflight.assess_release(
        FakeComposeContext("0|0|0|0|0020_strategy_allocations"),  # type: ignore[arg-type]
        tmp_path,
        minimum_free_gb=0,
    )

    assert result["status"] == "ready"
    assert result["blocker_count"] == 0
    assert all(check["status"] == "pass" for check in result["checks"])


def test_assess_release_blocks_active_durable_work(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        release_preflight,
        "_schema_compatibility",
        lambda _root, _revision: ("0020_strategy_allocations", "current", True),
    )
    result = release_preflight.assess_release(
        FakeComposeContext("1|14472|4|0|0020_strategy_allocations"),  # type: ignore[arg-type]
        tmp_path,
        minimum_free_gb=0,
    )

    assert result["status"] == "blocked"
    assert result["blocker_count"] == 1
    durable = next(check for check in result["checks"] if check["id"] == "durable_work_idle")
    assert durable["status"] == "block"
    assert "pending 14472" in durable["evidence"]


def test_assess_release_allows_dormant_resumable_checkpoints(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        release_preflight,
        "_schema_compatibility",
        lambda _root, _revision: ("0020_strategy_allocations", "current", True),
    )
    result = release_preflight.assess_release(
        FakeComposeContext("0|16668|0|24|0020_strategy_allocations"),  # type: ignore[arg-type]
        tmp_path,
        minimum_free_gb=0,
    )

    assert result["status"] == "ready"
    durable = next(check for check in result["checks"] if check["id"] == "durable_work_idle")
    assert durable["status"] == "pass"
    assert "pending 16668" in durable["evidence"]
    assert "failed 24" in durable["evidence"]


def test_assess_release_reports_invalid_compose_without_crashing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        release_preflight,
        "_schema_compatibility",
        lambda _root, _revision: ("0027_web_config_templates", "unknown", False),
    )

    result = release_preflight.assess_release(
        InvalidComposeContext(),  # type: ignore[arg-type]
        tmp_path,
        minimum_free_gb=0,
    )

    assert result["status"] == "blocked"
    compose = next(check for check in result["checks"] if check["id"] == "compose_config")
    database = next(check for check in result["checks"] if check["id"] == "postgres_running")
    services = next(check for check in result["checks"] if check["id"] == "services_healthy")
    assert compose["status"] == "block"
    assert compose["evidence"] == "PLATFORM_SECRET_KEY is required"
    assert "not found through Docker project labels" in database["evidence"]
    assert "missing" in services["evidence"]


def test_assess_release_inspects_running_project_without_valid_compose(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        release_preflight,
        "_schema_compatibility",
        lambda _root, revision: (
            "0027_web_config_templates",
            "upgrade_required",
            revision == "0017_qmt_execution_state",
        ),
    )

    result = release_preflight.assess_release(
        InvalidComposeWithRunningProject(),  # type: ignore[arg-type]
        tmp_path,
        minimum_free_gb=0,
    )

    assert result["blocker_count"] == 1
    assert result["database_revision"] == "0017_qmt_execution_state"
    assert result["migration_state"] == "upgrade_required"
    assert next(
        check for check in result["checks"] if check["id"] == "postgres_running"
    )["status"] == "pass"
    assert next(
        check for check in result["checks"] if check["id"] == "services_healthy"
    )["status"] == "pass"


def test_compose_services_accepts_json_lines() -> None:
    raw = "\n".join(
        [
            json.dumps({"Service": "api", "State": "running"}),
            json.dumps({"Service": "web", "State": "running"}),
        ]
    )

    services = release_preflight._compose_services(raw)

    assert set(services) == {"api", "web"}


def test_known_older_database_revision_has_upgrade_path() -> None:
    project_root = Path(__file__).resolve().parents[1]

    code_revision, state, compatible = release_preflight._schema_compatibility(
        project_root,
        "0017_qmt_execution_state",
    )

    assert code_revision == _expected_migration_head(project_root)
    assert state == "upgrade_required"
    assert compatible is True


def test_deployed_0058_database_upgrades_to_current_head() -> None:
    project_root = Path(__file__).resolve().parents[1]

    code_revision, state, compatible = release_preflight._schema_compatibility(
        project_root,
        "0058_simulation_benchmark",
    )

    assert code_revision == _expected_migration_head(project_root)
    assert state == "upgrade_required"
    assert compatible is True


def test_all_migration_revision_ids_fit_alembic_version_column() -> None:
    project_root = Path(__file__).resolve().parents[1]
    revisions: dict[str, str | None] = {}
    for path in sorted((project_root / "migrations" / "versions").glob("*.py")):
        values: dict[str, object] = {}
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and target.id in {
                "revision",
                "down_revision",
            }:
                values[target.id] = ast.literal_eval(node.value)
        revision = values.get("revision")
        if isinstance(revision, str):
            down_revision = values.get("down_revision")
            revisions[revision] = (
                down_revision if isinstance(down_revision, str) else None
            )

    assert revisions["0063_model_artifact_env"] == "0062_research_asset_consumptions"
    assert revisions["0064_paper_stage_account"] == "0063_model_artifact_env"
    assert all(len(revision) <= 32 for revision in revisions)
    assert all(
        down_revision is None or down_revision in revisions
        for down_revision in revisions.values()
    )


def test_default_and_gpu_service_contracts() -> None:
    default = type("Context", (), {"profiles": ()})()
    gpu = type("Context", (), {"profiles": ("gpu",)})()

    assert "rdagent-data-science-worker" in release_preflight.expected_services(default)
    assert "rdagent-llm-finetune-worker" not in release_preflight.expected_services(default)
    assert "rdagent-llm-finetune-worker" in release_preflight.expected_services(gpu)


def test_immutable_image_contract_is_profile_aware(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in (
        *release_preflight._CORE_IMAGE_SETTINGS,
        *release_preflight._GPU_IMAGE_SETTINGS,
    ):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    digest = "a" * 64
    env_file.write_text(
        "\n".join(
            (
                f"RDAGENT_RUNTIME_IMAGE_DIGEST=sha256:{digest}",
                f"RDAGENT_QLIB_SANDBOX_IMAGE=example/qlib@sha256:{digest}",
                f"RDAGENT_DATA_SCIENCE_IMAGE=example/ds@sha256:{digest}",
                f"MODEL_SANDBOX_IMAGE=example/model@sha256:{digest}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert _REAL_IMMUTABLE_IMAGE_CONFIGURATION(
        SimpleNamespace(env_file=env_file, profiles=())  # type: ignore[arg-type]
    )[0]
    valid, evidence = _REAL_IMMUTABLE_IMAGE_CONFIGURATION(
        SimpleNamespace(env_file=env_file, profiles=("gpu",))  # type: ignore[arg-type]
    )
    assert valid is False
    assert "RDAGENT_FINETUNE_IMAGE" in evidence

    content = env_file.read_text(encoding="utf-8")
    env_file.write_text(
        content.replace(
            f"RDAGENT_DATA_SCIENCE_IMAGE=example/ds@sha256:{digest}",
            f"RDAGENT_DATA_SCIENCE_IMAGE=example/qlib@sha256:{digest}",
        ),
        encoding="utf-8",
    )
    valid, evidence = _REAL_IMMUTABLE_IMAGE_CONFIGURATION(
        SimpleNamespace(env_file=env_file, profiles=())  # type: ignore[arg-type]
    )
    assert valid is False
    assert "separately sealed" in evidence


@pytest.mark.parametrize(
    ("data_science_runtime", "expected_valid"),
    (("a", True), ("e", False)),
)
def test_preloaded_runtime_requires_data_science_to_share_canonical_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    data_science_runtime: str,
    expected_valid: bool,
) -> None:
    for name in release_preflight._CORE_IMAGE_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    runtime_id = "sha256:" + "a" * 64
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"RDAGENT_RUNTIME_IMAGE_DIGEST={runtime_id}\n"
        f"RDAGENT_QLIB_SANDBOX_IMAGE=registry/qlib@sha256:{'b' * 64}\n"
        f"RDAGENT_DATA_SCIENCE_IMAGE=registry/data-science@sha256:{'c' * 64}\n"
        f"MODEL_SANDBOX_IMAGE=registry/model@sha256:{'d' * 64}\n",
        encoding="utf-8",
    )

    class Context:
        profiles: tuple[str, ...] = ()

        def __init__(self) -> None:
            self.env_file = env_file

        def container_id(self, service: str) -> str:
            return f"container-{service}"

        def docker(self, *args: str, **_kwargs) -> str:
            assert args[:3] == ("inspect", "--format", "{{.Image}}")
            service = args[3].removeprefix("container-")
            return (
                "sha256:" + data_science_runtime * 64
                if service == "rdagent-data-science-worker"
                else runtime_id
            )

        def run(self, *args: str, **_kwargs) -> str:
            assert args[:7] == (
                "exec",
                "-T",
                "rdagent-docker",
                "docker",
                "image",
                "inspect",
                "--format",
            )
            return "\n".join(
                ("sha256:" + character * 64) for character in ("b", "c", "d")
            )

    valid, evidence = _REAL_PRELOADED_IMAGE_AVAILABILITY(
        Context()  # type: ignore[arg-type]
    )

    assert valid is expected_valid
    if expected_valid:
        assert evidence.startswith("runtime identities match")
    else:
        assert evidence == "runtime identity mismatch: rdagent-data-science-worker"


def test_dind_storage_must_be_outside_governed_market_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in (
        "QUANTLAB_DATA_HOST_PATH",
        "RDAGENT_DOCKER_HOST_PATH",
        "RDAGENT_REGISTRY_HOST_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "QUANTLAB_DATA_HOST_PATH=/data/quantlab\n"
        "RDAGENT_DOCKER_HOST_PATH=/data/quantlab/docker\n"
        "RDAGENT_REGISTRY_HOST_PATH=/data/quantlab-registry\n",
        encoding="utf-8",
    )

    valid, evidence = _REAL_DOCKER_STORAGE_CONFIGURATION(
        SimpleNamespace(env_file=env_file)  # type: ignore[arg-type]
    )

    assert valid is False
    assert "outside QUANTLAB_DATA_HOST_PATH" in evidence


def test_unknown_database_revision_fails_closed() -> None:
    project_root = Path(__file__).resolve().parents[1]

    code_revision, state, compatible = release_preflight._schema_compatibility(
        project_root,
        "9999_unknown_revision",
    )

    assert code_revision == _expected_migration_head(project_root)
    assert state == "unknown_database_revision"
    assert compatible is False


class _IdentityContext:
    def __init__(
        self,
        env_file: Path,
        identity: dict[str, str],
        *,
        missing_labels: bool = False,
        mismatched_service: str | None = None,
    ) -> None:
        self.env_file = env_file
        self.identity = identity
        self.missing_labels = missing_labels
        self.mismatched_service = mismatched_service
        self.inspected: list[str] = []

    def container_id(self, service: str) -> str:
        return f"container-{service}"

    def docker(self, *args: str, **_kwargs) -> str:
        assert args[0] == "inspect"
        service = args[1].removeprefix("container-")
        self.inspected.append(service)
        labels = {} if self.missing_labels else release_identity_labels(self.identity)
        environment = dict(self.identity)
        if service == self.mismatched_service:
            labels["quantlab.config-digest"] = "f" * 64
            environment["QUANTLAB_RELEASE_ALIAS_OF"] = "wrong-canonical"
        return json.dumps(
            [
                {
                    "Config": {
                        "Env": [f"{key}={value}" for key, value in environment.items()],
                        "Labels": labels,
                    }
                }
            ]
        )


def _write_identity_environment(path: Path, identity: dict[str, str]) -> None:
    path.write_text(
        "POSTGRES_PASSWORD=test\n"
        + "".join(f"{key}={value}\n" for key, value in identity.items()),
        encoding="utf-8",
    )


def test_runtime_release_identity_verifies_env_and_labels_for_every_stateless_service(
    tmp_path: Path,
) -> None:
    identity = release_identity_environment("release-001", "a" * 64)
    env_file = tmp_path / ".env"
    _write_identity_environment(env_file, identity)
    context = _IdentityContext(env_file, identity)

    valid, evidence = _REAL_RUNTIME_RELEASE_IDENTITY(
        context,  # type: ignore[arg-type]
        {"postgres", "api", "gateway"},
    )

    assert valid is True
    assert "release-001 as canonical aliasing release-001" in evidence
    assert context.inspected == ["api", "gateway"]


def test_runtime_release_identity_rejects_old_containers_without_labels(
    tmp_path: Path,
) -> None:
    identity = release_identity_environment("release-001", "a" * 64)
    env_file = tmp_path / ".env"
    _write_identity_environment(env_file, identity)

    valid, evidence = _REAL_RUNTIME_RELEASE_IDENTITY(
        _IdentityContext(env_file, identity, missing_labels=True),  # type: ignore[arg-type]
        {"postgres", "api"},
    )

    assert valid is False
    assert "api:label:quantlab.release" in evidence
    assert "api:label:quantlab.alias-of" in evidence


def test_runtime_release_identity_rejects_any_service_env_or_label_mismatch(
    tmp_path: Path,
) -> None:
    identity = release_identity_environment("release-001", "a" * 64)
    env_file = tmp_path / ".env"
    _write_identity_environment(env_file, identity)

    valid, evidence = _REAL_RUNTIME_RELEASE_IDENTITY(
        _IdentityContext(  # type: ignore[arg-type]
            env_file,
            identity,
            mismatched_service="gateway",
        ),
        {"postgres", "api", "gateway"},
    )

    assert valid is False
    assert "gateway:env:QUANTLAB_RELEASE_ALIAS_OF" in evidence
    assert "gateway:label:quantlab.config-digest" in evidence


def test_runtime_release_identity_rejects_invalid_canonical_alias_contract(
    tmp_path: Path,
) -> None:
    identity = release_identity_environment("release-001", "a" * 64)
    identity["QUANTLAB_RELEASE_ALIAS_OF"] = "different-release"
    env_file = tmp_path / ".env"
    _write_identity_environment(env_file, identity)
    context = _IdentityContext(env_file, identity)

    valid, evidence = _REAL_RUNTIME_RELEASE_IDENTITY(
        context,  # type: ignore[arg-type]
        {"postgres", "api"},
    )

    assert valid is False
    assert "canonical release must alias itself" in evidence
    assert context.inspected == []
