from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from quant_platform import canonical_baseline
from quant_platform.deployment_services import (
    CORE_RUNTIME_SERVICES,
    OPTIONAL_PROFILE_SERVICES,
    WRITER_SERVICES,
)
from quant_platform.release_identity import release_identity_labels

pytestmark = pytest.mark.no_database


def _image_id(service: str) -> str:
    return "sha256:" + hashlib.sha256(service.encode("utf-8")).hexdigest()


def _service_config_hash(service: str) -> str:
    return hashlib.sha256(f"compose-config-{service}".encode()).hexdigest()


class _OverrideContext:
    def __init__(self, owner: FakeContext, override: Path) -> None:
        self.owner = owner
        self.override = override
        self.project_name = owner.project_name
        self.env_file = owner.env_file
        self.compose_files = (*owner.compose_files, override)
        self.profiles = owner.profiles
        self.project_directory = owner.project_directory

    def run(self, *args: str, **kwargs: Any) -> str:
        return self.owner._run(args, override=self.override, **kwargs)

    def docker(self, *args: str, **kwargs: Any) -> str:
        return self.owner.docker(*args, **kwargs)

    def container_id(self, service: str, *, all_states: bool = False) -> str:
        return self.owner.container_id(service, all_states=all_states)


class FakeContext:
    project_name = "quantlab-test"
    profiles: tuple[str, ...] = ()

    def __init__(
        self,
        tmp_path: Path,
        *,
        queue_states: list[str] | None = None,
        fail_canonical_up: bool = False,
        environment: str = "POSTGRES_PASSWORD=test\n",
        profiles: tuple[str, ...] = (),
    ) -> None:
        self.profiles = profiles
        self.project_directory = tmp_path
        self.env_file = tmp_path / "deploy.env"
        self.env_file.write_text(environment, encoding="utf-8")
        compose = tmp_path / "compose.yaml"
        compose.write_text("services: {}\n", encoding="utf-8")
        self.compose_files = (compose,)
        self.queue_states = list(queue_states or ["0|0"])
        self.fail_canonical_up = fail_canonical_up
        self.canonical_failure_used = False
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.sql_queries: list[str] = []
        self.containers: dict[str, dict[str, Any]] = {}
        runtime_services = set(CORE_RUNTIME_SERVICES)
        for profile in profiles:
            runtime_services.update(OPTIONAL_PROFILE_SERVICES.get(profile, ()))
        for service in runtime_services:
            self.containers[service] = {
                "image_id": _image_id(service),
                "running": True,
                "health": "healthy",
                "labels": {
                    "com.docker.compose.project": self.project_name,
                    "com.docker.compose.service": service,
                    "com.docker.compose.project.config_files": str(compose.resolve()),
                    "com.docker.compose.config-hash": _service_config_hash(service),
                    "quantlab.release": f"mixed-{service}",
                    "quantlab.config-digest": hashlib.sha256(
                        f"config-{service}".encode()
                    ).hexdigest(),
                },
                "environment": [
                    f"QUANTLAB_RELEASE_ID=mixed-{service}",
                    "QUANTLAB_CONFIG_DIGEST="
                    + hashlib.sha256(f"config-{service}".encode()).hexdigest(),
                ],
            }

    def with_override(self, override: Path) -> _OverrideContext:
        return _OverrideContext(self, override)

    def container_id(self, service: str, *, all_states: bool = False) -> str:
        del all_states
        return f"cid-{service}" if service in self.containers else ""

    def docker(self, *args: str, **_kwargs: Any) -> str:
        assert args[0] == "inspect"
        service = args[1].removeprefix("cid-")
        item = self.containers[service]
        state: dict[str, Any] = {
            "Running": item["running"],
            "Status": "running" if item["running"] else "exited",
            "Health": {"Status": item["health"]},
        }
        return json.dumps(
            [
                {
                    "Id": f"cid-{service}",
                    "Image": item["image_id"],
                    "State": state,
                    "Config": {
                        "Labels": item["labels"],
                        "Env": item["environment"],
                    },
                }
            ]
        )

    def run(self, *args: str, **kwargs: Any) -> str:
        return self._run(args, override=None, **kwargs)

    def _run(
        self,
        args: tuple[str, ...],
        *,
        override: Path | None,
        **_kwargs: Any,
    ) -> str:
        self.calls.append((override.name if override else "base", args))
        if args[0] == "exec":
            query = args[-1]
            self.sql_queries.append(query)
            assert query.lstrip().upper().startswith("SELECT ")
            value = self.queue_states[0]
            if len(self.queue_states) > 1:
                self.queue_states.pop(0)
            return value
        if args[:2] == ("config", "--hash"):
            service = args[2]
            return f"{service} {_service_config_hash(service)}\n"
        if args[0] == "stop":
            for service in args[1:]:
                self.containers[service]["running"] = False
            return ""
        if args[0] == "start":
            for service in args[1:]:
                self.containers[service]["running"] = True
                self.containers[service]["health"] = "healthy"
            return ""
        if args[0] != "up" or override is None:
            raise AssertionError(f"unexpected fake Compose call: {args!r}")

        payload = json.loads(override.read_text(encoding="utf-8"))
        configured = payload["services"]
        selected = [service for service in args if service in configured]
        for service in selected:
            entry = configured[service]
            item = self.containers[service]
            item["image_id"] = entry["image"]
            item["running"] = True
            item["health"] = "healthy"
            labels = {
                "com.docker.compose.project": self.project_name,
                "com.docker.compose.service": service,
                "com.docker.compose.project.config_files": ",".join(
                    str(path.resolve()) for path in (*self.compose_files, override)
                ),
            }
            labels.update(entry.get("labels", {}))
            item["labels"] = labels
            identity = entry.get("environment", {})
            item["environment"] = [f"{key}={value}" for key, value in identity.items()]

        if (
            self.fail_canonical_up
            and override.name == "canonical.override.json"
            and not self.canonical_failure_used
        ):
            self.canonical_failure_used = True
            raise RuntimeError("injected canonical recreation failure")
        return ""


@pytest.fixture
def use_fake_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        canonical_baseline,
        "_context_with_override",
        lambda context, override: context.with_override(override),
    )


def _records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    previous = "0" * 64
    for record in records:
        claimed = record.pop("record_sha256")
        assert record["previous_sha256"] == previous
        assert claimed == hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        previous = claimed
        record["record_sha256"] = claimed
    return records


def test_default_dry_run_discovers_full_topology_without_writes(tmp_path: Path) -> None:
    context = FakeContext(tmp_path)
    original_environment = context.env_file.read_bytes()
    receipt_root = tmp_path / "receipts"

    result = canonical_baseline.converge_canonical_baseline(context, receipt_root)

    assert result["status"] == "dry_run"
    assert result["mutation_confirmed"] is False
    assert set(result["required_services"]) == set(CORE_RUNTIME_SERVICES)
    assert set(result["writer_services"]) == set(CORE_RUNTIME_SERVICES).intersection(
        WRITER_SERVICES
    )
    assert context.env_file.read_bytes() == original_environment
    assert not receipt_root.exists()
    assert all(args[0] in {"exec", "config"} for _source, args in context.calls)
    assert len(result["service_snapshot"]) == len(CORE_RUNTIME_SERVICES)
    assert all(item["config_labels"] for item in result["service_snapshot"].values())
    assert result["rollback_config_equivalence"]["status"] == "pass"


def test_config_mismatch_blocks_before_any_canonical_mutation(tmp_path: Path) -> None:
    context = FakeContext(tmp_path)
    context.containers["postgres"]["labels"][
        "com.docker.compose.config-hash"
    ] = "0" * 64
    receipt_root = tmp_path / "receipts"

    result = canonical_baseline.converge_canonical_baseline(
        context,
        receipt_root,
        confirmed=True,
    )

    assert result["status"] == "blocked"
    assert result["blocker"] == "rollback_contract_not_config_equivalent"
    assert result["rollback_config_equivalence"]["mismatched_services"] == [
        "postgres"
    ]
    assert not receipt_root.exists()
    assert all(args[0] in {"exec", "config"} for _source, args in context.calls)


def test_non_idle_queue_blocks_before_any_mutation(tmp_path: Path) -> None:
    context = FakeContext(tmp_path, queue_states=["3|1"])
    receipt_root = tmp_path / "receipts"

    result = canonical_baseline.converge_canonical_baseline(
        context,
        receipt_root,
        confirmed=True,
    )

    assert result["status"] == "blocked"
    assert result["blocker"] == "durable_queue_not_idle"
    assert not receipt_root.exists()
    assert all(args[0] == "exec" for _source, args in context.calls)


def test_enabled_profile_adds_no_optional_runtime(tmp_path: Path) -> None:
    context = FakeContext(tmp_path, profiles=("gpu",))

    result = canonical_baseline.converge_canonical_baseline(
        context,
        tmp_path / "receipts",
    )

    assert result["status"] == "dry_run"
    assert set(result["required_services"]) == set(CORE_RUNTIME_SERVICES)


def test_confirmed_convergence_pins_images_and_writes_chained_receipt(
    tmp_path: Path,
    use_fake_override: None,
) -> None:
    del use_fake_override
    context = FakeContext(
        tmp_path,
        queue_states=["0|0", "0|0"],
        environment=(
            "POSTGRES_PASSWORD=test\n"
            "QUANTLAB_RELEASE_ID=obsolete-first\n"
            "# QUANTLAB_RELEASE_ID=comment-preserved\n"
            "export QUANTLAB_RELEASE_ID=obsolete-last\n"
            f"QUANTLAB_CONFIG_DIGEST={'b' * 64}\n"
        ),
    )
    original_images = {
        service: item["image_id"] for service, item in context.containers.items()
    }

    result = canonical_baseline.converge_canonical_baseline(
        context,
        tmp_path / "receipts",
        confirmed=True,
        release_id="canonical-test-001",
        wait_timeout=30,
    )

    assert result["status"] == "succeeded"
    assert result["safe_mode_action"] == "none"
    assert result["postgres_data_action"] == "none"
    assert result["governed_data_action"] == "none"
    assert {service: item["image_id"] for service, item in context.containers.items()} == (
        original_images
    )
    environment = context.env_file.read_text(encoding="utf-8")
    assert environment.count("QUANTLAB_RELEASE_ID=canonical-test-001") == 1
    assert environment.count(f"QUANTLAB_CONFIG_DIGEST={result['config_digest']}") == 1
    assert "# QUANTLAB_RELEASE_ID=comment-preserved" in environment
    assert "obsolete-first" not in environment
    assert "obsolete-last" not in environment
    for key, value in result["release_identity"].items():
        assert environment.count(f"{key}={value}") == 1

    override = json.loads(Path(result["canonical_override"]).read_text(encoding="utf-8"))
    assert set(override["services"]) == set(CORE_RUNTIME_SERVICES)
    assert {
        service: entry["image"] for service, entry in override["services"].items()
    } == original_images
    assert "environment" not in override["services"]["postgres"]
    assert all(
        entry["image"].startswith("sha256:") for entry in override["services"].values()
    )
    for service, item in result["services"].items():
        labels = item["config_labels"]
        assert {
            key: labels[key]
            for key in release_identity_labels(result["release_identity"])
        } == release_identity_labels(result["release_identity"])
        if service != "postgres":
            assert item["release_environment"] == result["release_identity"]

    up_calls = [args for _source, args in context.calls if args[0] == "up"]
    assert up_calls
    for call in up_calls:
        assert {
            "--no-build",
            "--force-recreate",
            "--remove-orphans",
            "--wait",
        } <= set(call)
    stopped = next(args[1:] for _source, args in context.calls if args[0] == "stop")
    assert set(stopped) == set(CORE_RUNTIME_SERVICES).intersection(WRITER_SERVICES)
    assert all("UPDATE" not in query.upper() for query in context.sql_queries)

    receipt_path = Path(result["receipt_path"])
    records = _records(receipt_path)
    assert [record["event"] for record in records] == [
        "discovery_completed",
        "writers_stopped",
        "post_stop_queue_checked",
        "canonical_contract_persisted",
        "convergence_succeeded",
    ]
    discovery = records[0]["payload"]
    assert discovery["canonical_inputs"]["environment"]["sha256"]
    assert discovery["canonical_inputs"]["compose"][0]["sha256"]
    assert set(discovery["services"]) == set(CORE_RUNTIME_SERVICES)
    if os.name != "nt":
        assert receipt_path.stat().st_mode & 0o777 == 0o400


def test_queue_race_after_writer_stop_restarts_without_changing_identity(
    tmp_path: Path,
) -> None:
    context = FakeContext(tmp_path, queue_states=["0|0", "1|0"])
    original_environment = context.env_file.read_bytes()
    original_images = {
        service: item["image_id"] for service, item in context.containers.items()
    }

    result = canonical_baseline.converge_canonical_baseline(
        context,
        tmp_path / "receipts",
        confirmed=True,
        release_id="canonical-race",
        wait_timeout=30,
    )

    assert result["status"] == "blocked"
    assert result["blocker"] == "durable_queue_became_active_while_stopping_writers"
    assert context.env_file.read_bytes() == original_environment
    assert all(item["running"] for item in context.containers.values())
    assert {service: item["image_id"] for service, item in context.containers.items()} == (
        original_images
    )
    assert not Path(result["canonical_override"]).exists()
    assert _records(Path(result["receipt_path"]))[-1]["event"] == (
        "convergence_blocked_after_stop"
    )


def test_failed_convergence_restores_environment_and_exact_service_images(
    tmp_path: Path,
    use_fake_override: None,
) -> None:
    del use_fake_override
    context = FakeContext(
        tmp_path,
        queue_states=["0|0", "0|0"],
        fail_canonical_up=True,
        environment=(
            "POSTGRES_PASSWORD=test\n"
            "QUANTLAB_RELEASE_ID=old-release\n"
            f"QUANTLAB_CONFIG_DIGEST={'a' * 64}\n"
        ),
    )
    original_environment = context.env_file.read_bytes()
    original_images = {
        service: item["image_id"] for service, item in context.containers.items()
    }

    result = canonical_baseline.converge_canonical_baseline(
        context,
        tmp_path / "receipts",
        confirmed=True,
        release_id="canonical-failure",
        wait_timeout=30,
    )

    assert result["status"] == "rolled_back"
    assert result["rollback"] == {"status": "succeeded", "images_verified": True}
    assert context.env_file.read_bytes() == original_environment
    assert {service: item["image_id"] for service, item in context.containers.items()} == (
        original_images
    )
    rollback = json.loads(
        (Path(result["receipt_path"]).parent / "rollback.override.json").read_text(
            encoding="utf-8"
        )
    )
    assert {
        service: entry["image"] for service, entry in rollback["services"].items()
    } == original_images
    assert all("labels" not in entry for entry in rollback["services"].values())
    assert _records(Path(result["receipt_path"]))[-1]["event"] == "rollback_succeeded"


def test_receipts_are_rejected_inside_configured_governed_data(tmp_path: Path) -> None:
    data_root = tmp_path / "governed-data"
    context = FakeContext(
        tmp_path,
        environment=f"POSTGRES_PASSWORD=test\nQUANTLAB_DATA_HOST_PATH={data_root}\n",
    )
    receipt_root = data_root / "receipts"

    with pytest.raises(ValueError, match="outside governed /data"):
        canonical_baseline.converge_canonical_baseline(context, receipt_root)

    assert not receipt_root.exists()
