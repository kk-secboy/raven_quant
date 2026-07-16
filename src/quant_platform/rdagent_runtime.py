from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import requests

from quant_data.config import Settings
from quant_data.path_utils import to_wsl_path as _to_wsl_path

from .upstream_versions import (
    RDAGENT_COMMIT,
    require_upstream_runtime_identity,
)

_DURATION = re.compile(r"^[1-9][0-9]*(?:m|h)$")


def validate_duration(value: str) -> str:
    value = value.strip().lower()
    if not _DURATION.fullmatch(value):
        raise ValueError("duration must be a positive number followed by m or h")
    return value


def require_rdagent_runtime_identity(value: Any) -> dict[str, Any]:
    return require_upstream_runtime_identity("rdagent", value)


def probe_rdagent(
    settings: Settings,
    project_root: Path,
    runtime_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    if settings.rdagent_worker_url:
        try:
            response = requests.get(
                f"{settings.rdagent_worker_url}/rdagent/status",
                timeout=8,
            )
            response.raise_for_status()
            result = response.json()
            identity = require_rdagent_runtime_identity(
                result.get("rdagent_runtime", result)
            )
            return {**result, **identity}
        except (requests.RequestException, ValueError) as exc:
            return {"status": "unavailable", "ready": False, "error": str(exc)}
    if not settings.rdagent_enabled:
        return {"status": "disabled", "ready": False, "reason": "RDAGENT_ENABLED is false"}
    bridge = project_root / "scripts" / "rdagent_bridge.py"
    is_wsl = os.name == "nt" and settings.rdagent_python.startswith("/")
    command = (
        [
            "wsl",
            "-d",
            settings.rdagent_wsl_distro,
            "--exec",
            settings.rdagent_python,
            _to_wsl_path(bridge),
        ]
        if is_wsl
        else [settings.rdagent_python, str(bridge)]
    )
    command.extend(
        [
            "probe",
            "--llm-key-env",
            settings.rdagent_llm_key_env,
        ]
    )
    repository_path = (
        _to_wsl_path(settings.rdagent_repo) if is_wsl else str(settings.rdagent_repo)
    )
    probe_env = {
        **os.environ,
        **(runtime_env or {}),
        "RDAGENT_COMMIT": RDAGENT_COMMIT,
        "RDAGENT_REPO": repository_path,
    }
    if is_wsl:
        existing_wslenv = probe_env.get("WSLENV", "")
        probe_env["WSLENV"] = ":".join(
            item
            for item in (existing_wslenv, "RDAGENT_COMMIT", "RDAGENT_REPO")
            if item
        )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
            env=probe_env,
        )
        line = next(
            (item for item in reversed(completed.stdout.splitlines()) if item.startswith("{")),
            "{}",
        )
        result = json.loads(line)
        identity = require_rdagent_runtime_identity(result)
        result = {**result, **identity}
    except (
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        return {"status": "unavailable", "ready": False, "error": str(exc)}
    blockers = []
    if not result.get("llm_credentials_configured"):
        blockers.append(f"configure secret {settings.rdagent_llm_key_env}")
    if not result.get("docker_available"):
        blockers.append("Docker is required by the RD-Agent Qlib sandbox")
    result["enabled"] = True
    result["ready"] = result.get("status") == "ok" and not blockers
    result["blockers"] = blockers
    result["limits"] = {
        "max_loops": settings.rdagent_max_loops,
        "max_duration": settings.rdagent_max_duration,
    }
    return result


def rdagent_command(
    settings: Settings,
    *,
    project_root: Path,
    trace_path: Path,
    result_path: Path,
    dataset_path: Path,
    loop_n: int,
    duration: str,
    periods: dict[str, str],
    objective: str,
) -> tuple[list[str], dict[str, str]]:
    duration = validate_duration(duration)
    if loop_n < 1 or loop_n > settings.rdagent_max_loops:
        raise ValueError(f"loop_n must be between 1 and {settings.rdagent_max_loops}")
    if not settings.rdagent_enabled:
        raise ValueError("RD-Agent execution is disabled")
    is_wsl = os.name == "nt" and settings.rdagent_command.startswith("/")
    trace_arg = _to_wsl_path(trace_path) if is_wsl else str(trace_path)
    result_arg = _to_wsl_path(result_path) if is_wsl else str(result_path)
    bridge_arg = (
        _to_wsl_path(project_root / "scripts" / "rdagent_bridge.py")
        if is_wsl
        else str(project_root / "scripts" / "rdagent_bridge.py")
    )
    runner_arg = (
        _to_wsl_path(project_root / "scripts" / "run_rdagent_factor.py")
        if is_wsl
        else str(project_root / "scripts" / "run_rdagent_factor.py")
    )
    if is_wsl:
        command = [
            "wsl",
            "-d",
            settings.rdagent_wsl_distro,
            "--exec",
            settings.rdagent_python,
            runner_arg,
            "--command",
            settings.rdagent_command,
            "--bridge",
            bridge_arg,
            "--trace",
            trace_arg,
            "--result",
            result_arg,
            "--loop-n",
            str(loop_n),
            "--duration",
            duration,
        ]
    else:
        # Local/container runtime uses the module wrapper to run and export in one process group.
        command = [
            settings.rdagent_python,
            str(project_root / "scripts" / "run_rdagent_factor.py"),
            "--command",
            settings.rdagent_command,
            "--bridge",
            bridge_arg,
            "--trace",
            trace_arg,
            "--result",
            result_arg,
            "--loop-n",
            str(loop_n),
            "--duration",
            duration,
        ]
    qlib_home_host = trace_path.parent / "qlib-home"
    qlib_home_host_str = _to_wsl_path(qlib_home_host) if is_wsl else str(qlib_home_host.resolve())
    dataset_host_str = _to_wsl_path(dataset_path) if is_wsl else str(dataset_path.resolve())
    env = {
        "LOG_TRACE_PATH": trace_arg,
        "RDAGENT_COMMAND": settings.rdagent_command,
        "RDAGENT_PYTHON": settings.rdagent_python,
        "RDAGENT_COMMIT": RDAGENT_COMMIT,
        "RDAGENT_REPO": (
            _to_wsl_path(settings.rdagent_repo)
            if is_wsl
            else str(settings.rdagent_repo)
        ),
        "LOOP_N": str(loop_n),
        "DURATION": duration,
        "BRIDGE": bridge_arg,
        "RESULT_PATH": result_arg,
        "QLIB_FACTOR_TRAIN_START": periods["train_start"],
        "QLIB_FACTOR_TRAIN_END": periods["train_end"],
        "QLIB_FACTOR_VALID_START": periods["valid_start"],
        "QLIB_FACTOR_VALID_END": periods["valid_end"],
        "QLIB_FACTOR_TEST_START": periods["test_start"],
        "QLIB_FACTOR_TEST_END": periods["test_end"],
        "QLIB_FACTOR_SCEN": "quant_platform.rdagent_scenario.QuantLabFactorScenario",
        "QUANTLAB_RESEARCH_OBJECTIVE": objective,
        "QLIB_DOCKER_EXTRA_VOLUMES": json.dumps(
            {
                qlib_home_host_str: {
                    "bind": "/root/.qlib/",
                    "mode": "rw",
                },
                dataset_host_str: {
                    "bind": "/root/.qlib/qlib_data/cn_data",
                    "mode": "ro",
                },
            },
            separators=(",", ":"),
        ),
    }
    (qlib_home_host / "qlib_data" / "cn_data").mkdir(parents=True, exist_ok=True)
    if is_wsl:
        source_root = _to_wsl_path(project_root / "src")
        env["PYTHONPATH"] = ":".join(
            item for item in [source_root, os.getenv("PYTHONPATH", "")] if item
        )
        forwarded = [*env, settings.rdagent_llm_key_env]
        existing = os.getenv("WSLENV", "")
        env["WSLENV"] = ":".join([item for item in [existing, *forwarded] if item])
    return command, env
