from __future__ import annotations

import argparse
import base64
import json
import secrets
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from _project import PROJECT_ROOT

from quant_platform.backup_restore import ComposeContext, compose_context
from quant_platform.release_upgrade import run_release_upgrade


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_env(path: Path) -> None:
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
                "REQUESTS_PER_MINUTE=90",
                "DOWNLOAD_WORKERS=2",
                "LOG_MAX_SIZE=5m",
                "LOG_MAX_FILES=2",
            )
        )
        + "\n",
        encoding="utf-8",
    )


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


def run_drill(project_root: Path) -> dict:
    release_id = _stamp()
    result = {
        "status": "failed",
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "drill_project": f"quantlab-upgrade-drill-{secrets.token_hex(4)}",
        "live_trading_enabled": False,
        "cleanup": {},
    }
    context: ComposeContext | None = None
    rollback_tags: list[str] = []
    temporary_manager = tempfile.TemporaryDirectory(prefix="quantlab-release-upgrade-drill-")
    try:
        scratch = Path(temporary_manager.name)
        env_file = scratch / "drill.env"
        backup_root = scratch / "backups"
        _write_env(env_file)
        context = compose_context(
            result["drill_project"],
            env_file,
            project_root / "deploy" / "compose.yaml",
            (project_root / "deploy" / "compose.restore-drill.yaml",),
        )
        context.run(
            "up",
            "-d",
            "--no-build",
            "--wait",
            "--wait-timeout",
            "240",
        )
        upgrade = run_release_upgrade(
            context,
            project_root,
            backup_root,
            confirmed=True,
            retention_count=1,
            minimum_free_gb=1,
            wait_timeout=240,
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
            context.run("down", "-v", "--remove-orphans", check=False)
            if rollback_tags:
                context.docker("image", "rm", "-f", *rollback_tags, check=False)
            result["cleanup"] = _leftovers(context)
            if any(result["cleanup"].values()):
                result["status"] = "failed"
                result["cleanup_error"] = "isolated Compose resources remain"
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
