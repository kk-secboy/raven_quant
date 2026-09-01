from __future__ import annotations

import argparse
import base64
import json
import secrets
import socket
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from _project import PROJECT_ROOT

from quant_platform.backup_restore import ComposeContext, compose_context
from quant_platform.control_plane_lock import control_plane_locked
from quant_platform.deployment_services import BUILT_APPLICATION_SERVICES
from quant_platform.drill_isolation import isolated_drill_environment
from quant_platform.release_upgrade import (
    prepare_drill_sandbox_bootstrap,
    run_release_upgrade,
)


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _write_env(
    path: Path,
    *,
    data_host_path: Path,
    docker_host_path: Path,
    registry_host_path: Path,
    registry_port: int,
) -> None:
    password = secrets.token_urlsafe(32)
    secret = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")
    path.write_text(
        "\n".join(
            (
                f"POSTGRES_PASSWORD={password}",
                "POSTGRES_BIND_ADDRESS=127.0.0.1",
                "POSTGRES_PORT=0",
                "HTTP_BIND_ADDRESS=127.0.0.1",
                "HTTP_PORT=0",
                "AUTH_MODE=required",
                "AUTH_COOKIE_SECURE=false",
                f"PLATFORM_SECRET_KEY={secret}",
                "BROKER_MODE=disabled",
                # Readiness probes only assert that an RD-Agent credential is
                # configured.  Pin its base URL to a closed loopback port so an
                # accidental drill job cannot contact an external LLM service.
                "OPENAI_API_KEY=drill-not-a-real-credential",
                "OPENAI_API_BASE=http://127.0.0.1:9",
                "REQUESTS_PER_MINUTE=118",
                "DOWNLOAD_WORKERS=2",
                "LOG_MAX_SIZE=5m",
                "LOG_MAX_FILES=2",
                f"QUANTLAB_DATA_HOST_PATH={data_host_path.resolve()}",
                f"RDAGENT_DOCKER_HOST_PATH={docker_host_path.resolve()}",
                f"RDAGENT_REGISTRY_HOST_PATH={registry_host_path.resolve()}",
                f"RDAGENT_REGISTRY_PORT={registry_port}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _write_candidate_runtime_override(
    path: Path,
    suffix: str,
) -> tuple[str, ...]:
    image_families = {
        "api-runtime": ("api",),
        "scheduler-runtime": ("scheduler",),
        "worker-runtime": ("worker", "evaluation-worker", "paper-worker"),
        "rdagent-runtime": (
            "rdagent-worker",
            "rdagent-model-worker",
            "rdagent-report-worker",
            "rdagent-quant-worker",
        ),
        "web": ("web",),
    }
    canonical_builders = {
        "api",
        "scheduler",
        "worker",
        "rdagent-worker",
        "web",
    }
    images = {
        service: f"quantlab-upgrade-drill-{family}:{suffix}"
        for family, services in image_families.items()
        for service in services
    }
    if set(images) != set(BUILT_APPLICATION_SERVICES):
        raise RuntimeError("candidate image families do not cover the built service topology")
    if not canonical_builders <= set(images):
        raise RuntimeError("candidate image families have no canonical builder")
    # A plain JSON ``null`` does not remove an inherited Compose build mapping;
    # it is ignored during merge.  Compose's YAML ``!reset`` tag is required so
    # each shared image family has exactly one builder and every mirror starts
    # from that builder's immutable image ID.
    lines = ["services:"]
    for service, image in sorted(images.items()):
        lines.extend((f"  {service}:", f"    image: {json.dumps(image)}"))
        if service not in canonical_builders:
            lines.append("    build: !reset null")
    # The one-shot factor builder loads this host image into the isolated DinD.
    # It must follow the drill-only worker alias too.
    lines.extend(
        (
            "  factor-sandbox-builder:",
            "    environment:",
            "      FACTOR_SANDBOX_BASE_IMAGE: " + json.dumps(images["worker"]),
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tuple(sorted(set(images.values())))


def _leftovers(context: ComposeContext) -> dict[str, list[str]]:
    label = f"label=com.docker.compose.project={context.project_name}"
    containers = context.docker(
        "ps",
        "-aq",
        "--filter",
        label,
        capture=True,
        check=False,
    ).splitlines()
    volumes = context.docker(
        "volume",
        "ls",
        "-q",
        "--filter",
        label,
        capture=True,
        check=False,
    ).splitlines()
    networks = context.docker(
        "network",
        "ls",
        "-q",
        "--filter",
        label,
        capture=True,
        check=False,
    ).splitlines()
    return {
        "containers": [item for item in containers if item],
        "volumes": [item for item in volumes if item],
        "networks": [item for item in networks if item],
    }


def _leftover_images(context: ComposeContext, images: tuple[str, ...]) -> list[str]:
    return [
        image
        for image in images
        if context.docker(
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            image,
            capture=True,
            check=False,
        )
    ]


@control_plane_locked
@isolated_drill_environment
def run_drill(project_root: Path) -> dict:
    release_id = _stamp()
    suffix = secrets.token_hex(4)
    result = {
        "status": "failed",
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "drill_project": f"quantlab-upgrade-drill-{suffix}",
        "live_trading_enabled": False,
        "cleanup": {},
    }
    context: ComposeContext | None = None
    rollback_tags: list[str] = []
    candidate_runtime_images: tuple[str, ...] = ()
    temporary_manager = tempfile.TemporaryDirectory(prefix="quantlab-release-upgrade-drill-")
    try:
        scratch = Path(temporary_manager.name)
        backup_root = scratch / "backups"
        env_file = backup_root / "drill.env"
        candidate_override = scratch / "candidate-runtime.compose.yaml"
        data_host_path = scratch / "drill-data"
        docker_host_path = scratch / "rdagent-docker"
        registry_host_path = scratch / "rdagent-registry"
        for path in (
            backup_root,
            data_host_path,
            docker_host_path,
            registry_host_path,
        ):
            path.mkdir()
        _write_env(
            env_file,
            data_host_path=data_host_path,
            docker_host_path=docker_host_path,
            registry_host_path=registry_host_path,
            registry_port=_available_loopback_port(),
        )
        candidate_runtime_images = _write_candidate_runtime_override(
            candidate_override,
            suffix,
        )
        baseline_context = compose_context(
            result["drill_project"],
            env_file,
            project_root / "deploy" / "compose.yaml",
            (project_root / "deploy" / "compose.restore-drill.yaml",),
        )
        context = baseline_context
        result["bootstrap_sandbox_images"] = prepare_drill_sandbox_bootstrap(
            baseline_context,
            project_root,
            release_id,
            wait_timeout=240,
        )
        baseline_context.run(
            "up",
            "-d",
            "--no-build",
            "--wait",
            "--wait-timeout",
            "240",
        )
        context = compose_context(
            result["drill_project"],
            env_file,
            project_root / "deploy" / "compose.yaml",
            (
                project_root / "deploy" / "compose.restore-drill.yaml",
                candidate_override,
            ),
        )
        upgrade = run_release_upgrade(
            context,
            project_root,
            backup_root,
            confirmed=True,
            retention_count=1,
            minimum_free_gb=1,
            wait_timeout=240,
            rollback_tag_repository=f"quantlab-upgrade-drill-rollback-{suffix}",
            prune_rollback_images=False,
        )
        result["upgrade"] = upgrade
        rollback_tags = list(upgrade.get("rollback_images", {}).values())
        if upgrade["status"] != "succeeded":
            raise RuntimeError(f"isolated release upgrade returned {upgrade['status']}")
        result["status"] = "succeeded"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if context is not None:
            context.run(
                "--profile",
                "sandbox-registry",
                "down",
                "-v",
                "--remove-orphans",
                check=False,
            )
            if rollback_tags:
                context.docker("image", "rm", "-f", *rollback_tags, check=False)
            if candidate_runtime_images:
                context.docker(
                    "image",
                    "rm",
                    "-f",
                    *candidate_runtime_images,
                    check=False,
                )
            result["cleanup"] = _leftovers(context)
            result["cleanup"]["images"] = _leftover_images(
                context,
                (*candidate_runtime_images, *rollback_tags),
            )
            if any(result["cleanup"].values()):
                result["status"] = "failed"
                result["cleanup_error"] = "isolated Compose resources remain"
            context.docker(
                "run",
                "--rm",
                "--volume",
                f"{scratch}:/scratch",
                "postgres:16-alpine",
                "find",
                "/scratch",
                "-mindepth",
                "1",
                "-maxdepth",
                "1",
                "-exec",
                "rm",
                "-rf",
                "--",
                "{}",
                "+",
                check=False,
            )
        temporary_manager.cleanup()
        result["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        result["drill_id"] = release_id
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a self-cleaning, isolated QuantLab release-upgrade acceptance"
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = run_drill(PROJECT_ROOT)
    report = args.report or (
        PROJECT_ROOT
        / "artifacts"
        / "release-upgrade-drills"
        / f"release-upgrade-drill-{result['drill_id']}.json"
    )
    report = report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "succeeded":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
