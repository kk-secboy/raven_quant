from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import requests

from quant_data.config import Settings
from quant_data.path_utils import to_wsl_path as _to_wsl_path
from quant_data.qlib_builder import verify_qlib_output_manifest

from .rdagent_dataset_view import isolate_rdagent_periods, prepare_rdagent_dataset_view
from .rdagent_scenarios import (
    RDAgentScenario,
    get_rdagent_scenario,
    rdagent_scenario_catalog,
    resolve_rdagent_assets,
    validate_feature_set_id,
)
from .research_horizon import LONG_1_3Y, SHORT_1_5D, SWING_1_6M
from .upstream_versions import (
    RDAGENT_COMMIT,
    require_upstream_runtime_identity,
)

_DURATION = re.compile(r"^[1-9][0-9]*(?:m|h)$")
_DIAGNOSTIC_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_DIAGNOSTIC_WINDOWS_PATH = re.compile(r"(?i)[a-z]:[\\/][^\s]+")
_DIAGNOSTIC_UNIX_PATH = re.compile(
    r"/(?:app|data|etc|home|mnt|opt|root|run|srv|tmp|usr|var)(?:/[^\s]*)?"
)


def validate_duration(value: str) -> str:
    value = value.strip().lower()
    if not _DURATION.fullmatch(value):
        raise ValueError("duration must be a positive number followed by m or h")
    return value


def _duration_seconds(value: str) -> int:
    normalized = validate_duration(value)
    amount = int(normalized[:-1])
    return amount * (60 if normalized.endswith("m") else 3600)


def validate_duration_limit(value: str, maximum: str) -> str:
    """Validate a request budget before it can enter any durable queue."""

    normalized = validate_duration(value)
    normalized_maximum = validate_duration(maximum)
    if _duration_seconds(normalized) > _duration_seconds(normalized_maximum):
        raise ValueError(f"duration exceeds configured limit {normalized_maximum}")
    return normalized


def require_rdagent_runtime_identity(value: Any) -> dict[str, Any]:
    return require_upstream_runtime_identity("rdagent", value)


def require_reproducible_rdagent_runtime_identity(value: Any) -> dict[str, Any]:
    """Return the immutable identity allowed to execute a governed RD-Agent job."""

    identity = require_rdagent_runtime_identity(value)
    if not identity.get("source_tree_sha256"):
        raise ValueError("rdagent runtime source tree digest is required")
    if not identity.get("runtime_image_digest"):
        raise ValueError("rdagent runtime image digest is required")
    if identity.get("repository_dirty") is True:
        raise ValueError("rdagent runtime source checkout must be clean")
    if identity.get("production_reproducible") is not True:
        raise ValueError("rdagent runtime is not production reproducible")
    return identity


def require_matching_rdagent_runtime_identity(
    expected: Any, actual: Any
) -> dict[str, Any]:
    """Fail closed if a queued job moved to a different RD-Agent runtime."""

    expected_identity = require_reproducible_rdagent_runtime_identity(expected)
    actual_identity = require_reproducible_rdagent_runtime_identity(actual)
    if actual_identity != expected_identity:
        raise ValueError("rdagent runtime identity changed after enqueue")
    return actual_identity


def expected_rdagent_runtime_identity(
    runtime: dict[str, Any], scenario_id: str
) -> dict[str, Any]:
    """Select and freeze the exact dedicated worker identity for a scenario."""

    worker_key = {
        "data_science": "data_science_worker",
        "llm_finetune": "gpu_worker",
    }.get(get_rdagent_scenario(scenario_id).id)
    source: Any = runtime
    if worker_key is not None:
        worker = runtime.get(worker_key)
        if not isinstance(worker, dict):
            raise ValueError(f"{worker_key} runtime identity is unavailable")
        if worker.get("runtime_identity_matches") is not True:
            raise ValueError(f"{worker_key} runtime identity disagrees with the main worker")
        source = worker.get("runtime_identity")
    else:
        source = runtime.get("runtime_identity", runtime)
    return require_reproducible_rdagent_runtime_identity(source)


def _probe_worker_json(url: str, endpoint: str) -> dict[str, Any]:
    response = requests.get(f"{url}{endpoint}", timeout=8)
    if response.status_code != 503:
        response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("worker readiness response must be an object")
    return value


def _probe_dedicated_rdagent_worker(
    url: str,
    *,
    main_identity: dict[str, Any],
    required_job_kind: str,
    include_gpu: bool = False,
) -> dict[str, Any]:
    health = _probe_worker_json(url, "/health")
    result = health.get("runtime")
    capabilities = health.get("capabilities")
    if not isinstance(result, dict) or not isinstance(capabilities, dict):
        raise ValueError("dedicated worker readiness evidence is unavailable")
    job_kinds = capabilities.get("job_kinds")
    if (
        health.get("status") != "ok"
        or not isinstance(job_kinds, list)
        or required_job_kind not in job_kinds
    ):
        raise ValueError("dedicated worker does not consume its required job kind")
    identity = require_rdagent_runtime_identity(result.get("rdagent_runtime", result))
    worker: dict[str, Any] = {
        "status": result.get("status"),
        "runtime_identity": identity,
        "runtime_identity_matches": identity == main_identity,
        "job_kinds": sorted(set(str(item) for item in job_kinds)),
        "llm_credentials_configured": bool(result.get("llm_credentials_configured")),
        "docker_available": bool(result.get("docker_available")),
        "data_science_sandbox_preloaded": bool(
            result.get("data_science_sandbox_preloaded")
        ),
        "data_science_smoke_passed": bool(result.get("data_science_smoke_passed")),
    }
    if include_gpu:
        for key in (
            "gpu_available",
            "gpu_devices",
            "gpu_memory_free_mb",
            "gpu_driver_version",
            "cuda_version",
            "docker_gpu_runtime_available",
            "docker_gpu_smoke_passed",
            "finetune_images_preloaded",
            "finetune_image_evidence",
            "data_root_free_gb",
            "hf_credentials_configured",
        ):
            worker[key] = result.get(key)
    return worker


def _unavailable_worker() -> dict[str, Any]:
    return {"status": "unavailable", "ready": False}


def _sanitize_diagnostic_output(text: str, runtime_env: dict[str, str]) -> str:
    for name, value in runtime_env.items():
        if any(
            marker in name.upper()
            for marker in ("KEY", "PASSWORD", "SECRET", "TOKEN", "CREDENTIAL")
        ) and len(value) >= 6:
            text = text.replace(value, "[redacted]")
    text = _DIAGNOSTIC_URL.sub("[redacted-url]", text)
    text = _DIAGNOSTIC_WINDOWS_PATH.sub("[redacted-path]", text)
    text = _DIAGNOSTIC_UNIX_PATH.sub("[redacted-path]", text)
    return text[-20_000:]


def run_official_rdagent_health_check(
    settings: Settings,
    project_root: Path,
    runtime_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run upstream diagnostics without Docker pulls or port probing.

    Upstream currently reports individual failures in logs but still exits zero,
    so this result is diagnostic evidence only and never changes platform readiness.
    """

    governed_env = {**os.environ, **(runtime_env or {})}
    local_runtime = probe_rdagent(
        settings,
        project_root,
        runtime_env=runtime_env,
        force_local=True,
    )
    identity = require_reproducible_rdagent_runtime_identity(
        local_runtime.get("runtime_identity", local_runtime)
    )
    is_wsl = os.name == "nt" and settings.rdagent_command.startswith("/")
    command = (
        [
            "wsl",
            "-d",
            settings.rdagent_wsl_distro,
            "--exec",
            settings.rdagent_command,
        ]
        if is_wsl
        else [settings.rdagent_command]
    )
    command.extend(["health_check", "--no-check-docker", "--no-check-ports"])
    if is_wsl:
        exported = []
        for key, value in (runtime_env or {}).items():
            governed_env[key] = value
            exported.append(key)
        existing_wslenv = governed_env.get("WSLENV", "")
        governed_env["WSLENV"] = ":".join(
            item for item in (existing_wslenv, *exported) if item
        )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            env=governed_env,
        )
        output = _sanitize_diagnostic_output(
            "\n".join((completed.stdout, completed.stderr)), governed_env
        )
        return {
            "status": "completed" if completed.returncode == 0 else "failed",
            "exit_code": completed.returncode,
            "diagnostic_only": True,
            "platform_readiness_unchanged": True,
            "checks": {"environment": True, "docker": False, "ports": False},
            "rdagent_runtime": identity,
            "output": output,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "failed",
            "diagnostic_only": True,
            "platform_readiness_unchanged": True,
            "checks": {"environment": True, "docker": False, "ports": False},
            "rdagent_runtime": identity,
            "output": _sanitize_diagnostic_output(str(exc), governed_env),
        }


def _finalize_probe(
    result: dict[str, Any], settings: Settings
) -> dict[str, Any]:
    blockers: list[str] = []
    if not result.get("llm_credentials_configured"):
        blockers.append(f"configure secret {settings.rdagent_llm_key_env}")
    if not result.get("docker_available"):
        blockers.append("Docker is required by the RD-Agent CoSTEER sandbox")
    result["enabled"] = settings.rdagent_enabled
    result["ready"] = result.get("status") == "ok" and not blockers
    result["blockers"] = blockers
    result["limits"] = {
        "max_loops": settings.rdagent_max_loops,
        "max_duration": settings.rdagent_max_duration,
    }
    result["scenarios"] = rdagent_scenario_catalog(result, settings)
    return result


def probe_rdagent(
    settings: Settings,
    project_root: Path,
    runtime_env: dict[str, str] | None = None,
    *,
    force_local: bool = False,
) -> dict[str, Any]:
    if settings.rdagent_worker_url and not force_local:
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
            merged = {**result, **identity, "runtime_identity": identity}
            if settings.rdagent_evaluation_worker_url:
                try:
                    evaluation = _probe_worker_json(
                        settings.rdagent_evaluation_worker_url, "/health"
                    )
                    capabilities = evaluation.get("capabilities")
                    evaluation_runtime = evaluation.get("runtime")
                    if not isinstance(capabilities, dict):
                        raise ValueError("evaluation worker capabilities are unavailable")
                    if not isinstance(evaluation_runtime, dict):
                        raise ValueError("evaluation worker runtime evidence is unavailable")
                    job_kinds = capabilities.get("job_kinds")
                    if not isinstance(job_kinds, list) or not all(
                        isinstance(item, str) for item in job_kinds
                    ):
                        raise ValueError("evaluation worker job kinds are unavailable")
                    merged["evaluation_worker"] = {
                        "status": evaluation.get("status"),
                        "ready": evaluation_runtime.get("status") == "ok",
                        "job_kinds": sorted(set(job_kinds)),
                        "model_sandbox_ready": bool(
                            capabilities.get("model_sandbox_ready")
                        ),
                        "model_sandbox_image_id": capabilities.get(
                            "model_sandbox_image_id"
                        ),
                        "model_sandbox_config_sha256": capabilities.get(
                            "model_sandbox_config_sha256"
                        ),
                    }
                except (requests.RequestException, TypeError, ValueError):
                    merged["evaluation_worker"] = _unavailable_worker()
            else:
                merged["evaluation_worker"] = _unavailable_worker()
            if settings.rdagent_data_science_worker_url:
                try:
                    merged["data_science_worker"] = _probe_dedicated_rdagent_worker(
                        settings.rdagent_data_science_worker_url,
                        main_identity=identity,
                        required_job_kind="rdagent_data_science",
                    )
                except (requests.RequestException, TypeError, ValueError):
                    merged["data_science_worker"] = _unavailable_worker()
            else:
                merged["data_science_worker"] = _unavailable_worker()
            if settings.rdagent_gpu_worker_url:
                try:
                    gpu_worker = _probe_dedicated_rdagent_worker(
                        settings.rdagent_gpu_worker_url,
                        main_identity=identity,
                        required_job_kind="rdagent_llm_finetune",
                        include_gpu=True,
                    )
                    merged["gpu_worker"] = gpu_worker
                    if gpu_worker.get("status") == "ok" and gpu_worker.get(
                        "gpu_available"
                    ):
                        merged["gpu_available"] = True
                        merged["gpu_devices"] = gpu_worker.get("gpu_devices") or []
                        for key in (
                            "gpu_memory_free_mb",
                            "gpu_driver_version",
                            "cuda_version",
                            "docker_gpu_runtime_available",
                            "docker_gpu_evidence",
                            "docker_gpu_smoke_passed",
                            "finetune_images_preloaded",
                            "finetune_image_evidence",
                            "data_root_free_gb",
                            "hf_credentials_configured",
                        ):
                            merged[key] = gpu_worker.get(key)
                except (requests.RequestException, TypeError, ValueError):
                    merged["gpu_worker"] = _unavailable_worker()
                    merged["gpu_worker"]["gpu_available"] = False
            else:
                merged["gpu_worker"] = _unavailable_worker()
                merged["gpu_worker"]["gpu_available"] = False
            return _finalize_probe(merged, settings)
        except (requests.RequestException, ValueError) as exc:
            return _finalize_probe(
                {"status": "unavailable", "ready": False, "error": str(exc)},
                settings,
            )
    if not settings.rdagent_enabled:
        return _finalize_probe(
            {
                "status": "disabled",
                "ready": False,
                "reason": "RDAGENT_ENABLED is false",
            },
            settings,
        )
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
            "--data-root",
            str(settings.data_root),
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
        result = {**result, **identity, "runtime_identity": identity}
    except (
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        return _finalize_probe(
            {"status": "unavailable", "ready": False, "error": str(exc)},
            settings,
        )
    return _finalize_probe(result, settings)


def _runtime_path(path: Path, *, is_wsl: bool) -> str:
    return _to_wsl_path(path) if is_wsl else str(path.resolve())


def _shared_runtime_root(settings: Settings, trace_path: Path) -> Path:
    """Return a run root visible to both the worker and its Docker daemon."""

    if settings.rdagent_docker_shared_root is None:
        return trace_path.parent.resolve()
    shared_root = settings.rdagent_docker_shared_root.resolve()
    shared_root.mkdir(parents=True, exist_ok=True)
    run_name = trace_path.parent.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_name):
        run_name = hashlib.sha256(str(trace_path.parent.resolve()).encode()).hexdigest()
    # Mirror the API-visible /data/artifacts/rdagent/<run> layout at the host
    # daemon's bind-source path. Both aliases address the same physical volume.
    target = (
        shared_root / "artifacts" / "rdagent" / run_name / "docker-runtime"
    ).resolve()
    target.relative_to(shared_root)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _stage_finetune_asset(
    resolved_assets: dict[str, Any],
    *,
    runtime_root: Path,
    audit_root: Path,
) -> tuple[Path, str]:
    """Copy the sealed FT bundle to a Docker-visible, run-specific directory."""

    assets = resolved_assets.get("assets") or []
    if len(assets) != 1:
        raise ValueError("llm_finetune requires exactly one governed bundle")
    asset = assets[0]
    manifest_sha256 = str(asset.get("manifest_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
        raise ValueError("llm_finetune asset manifest identity is invalid")
    entries = []
    for item in asset.get("files") or []:
        relative = Path(str(item.get("relative_path") or ""))
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("llm_finetune asset contains an unsafe relative path")
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": Path(str(item["path"])).stat().st_size,
                "sha256": str(item["sha256"]),
            }
        )
    entries.sort(key=lambda item: item["path"])
    inventory = {
        "contract_version": "quantlab-finetune-staging-v1",
        "asset_id": str(asset["asset_id"]),
        "asset_manifest_sha256": manifest_sha256,
        "files": entries,
    }
    inventory_sha256 = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    destination = runtime_root / f"finetune-files-{manifest_sha256[:16]}"
    evidence_path = destination / "quantlab-staged-inventory.json"

    def verify_staged() -> None:
        if not evidence_path.is_file():
            raise ValueError("existing fine-tune staging directory is incomplete")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence != {**inventory, "inventory_sha256": inventory_sha256}:
            raise ValueError("existing fine-tune staging evidence disagrees")
        actual_files = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
        }
        expected_files = {item["path"] for item in entries} | {
            "quantlab-staged-inventory.json"
        }
        if actual_files != expected_files:
            raise ValueError("fine-tune staging contains unsealed files")
        for entry in entries:
            path = destination / entry["path"]
            if path.stat().st_size != entry["bytes"]:
                raise ValueError("fine-tune staged file size disagrees")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError("fine-tune staged file digest disagrees")

    if destination.exists():
        verify_staged()
    else:
        runtime_root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".finetune-stage-", dir=runtime_root))
        try:
            for source_entry, inventory_entry in zip(
                sorted(asset["files"], key=lambda item: str(item["relative_path"])),
                entries,
                strict=True,
            ):
                target = stage / inventory_entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(str(source_entry["path"])), target)
            (stage / "quantlab-staged-inventory.json").write_text(
                json.dumps(
                    {**inventory, "inventory_sha256": inventory_sha256},
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        verify_staged()

    audit_root.mkdir(parents=True, exist_ok=True)
    (audit_root / "finetune-staging-evidence.json").write_text(
        json.dumps(
            {**inventory, "inventory_sha256": inventory_sha256},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination, inventory_sha256


def _prepare_scenario_inputs(
    settings: Settings,
    scenario: RDAgentScenario,
    *,
    trace_path: Path,
    asset_ids: list[str] | None,
    asset_manifest_sha256: dict[str, str] | None,
    feature_set: dict[str, Any] | None,
    periods: dict[str, str] | None,
    loop_n: int,
) -> tuple[list[str], Path | None, dict[str, str], dict[str, Any] | None]:
    resolved = resolve_rdagent_assets(
        settings,
        scenario,
        asset_ids,
        expected_manifest_sha256=asset_manifest_sha256,
        pre_final_end=(
            date.fromisoformat(str(periods["valid_end"]))
            if periods is not None and periods.get("valid_end")
            else None
        ),
        selection_limit=loop_n if scenario.id == "fin_factor_report" else None,
    )
    input_root = trace_path.parent / "scenario-inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    scenario_options_path: Path | None = None
    paths: list[str] = []
    if scenario.id == "fin_factor_report":
        reports = input_root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        for asset in resolved["assets"]:
            for index, item in enumerate(asset["files"]):
                destination = reports / f"{asset['asset_id']}-{index:03d}.pdf"
                shutil.copy2(item["path"], destination)
                if hashlib.sha256(destination.read_bytes()).hexdigest() != item["sha256"]:
                    raise ValueError("staged report digest disagrees with its governed asset")
        paths = [str(reports)]
    elif scenario.id == "general_model":
        paths = [str(resolved["assets"][0]["files"][0]["path"])]
    elif scenario.id in {"data_science", "llm_finetune"}:
        paths = [str(resolved["assets"][0]["path"])]
    if resolved["scenario_options"]:
        scenario_options_path = input_root / "scenario-options.json"
        scenario_options_path.write_text(
            json.dumps(resolved["scenario_options"], ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    feature_environment: dict[str, str] = {}
    feature_root: Path | None = None
    if scenario.requires_feature_set:
        if not isinstance(feature_set, dict):
            raise ValueError(f"{scenario.id} requires a governed feature set")
        feature_set_id = validate_feature_set_id(str(feature_set.get("id") or ""))
        feature_hash = str(feature_set.get("definition_sha256") or "")
        features = feature_set.get("features")
        if not feature_set_id or not re.fullmatch(r"[0-9a-f]{64}", feature_hash):
            raise ValueError("governed feature set identity is invalid")
        if not isinstance(features, dict) or not features:
            raise ValueError("governed feature set definition is empty")
        feature_root = input_root / "base-features"
        feature_root.mkdir(parents=True, exist_ok=True)
        (feature_root / "base_factors.json").write_text(
            json.dumps(features, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        (feature_root / "definition.json").write_text(
            json.dumps(feature_set, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        feature_environment = {
            "QUANTLAB_FEATURE_SET_ID": feature_set_id,
            "QUANTLAB_FEATURE_SET_DEFINITION_SHA256": feature_hash,
        }
    elif feature_set is not None:
        raise ValueError(f"{scenario.id} does not accept a governed feature set")
    return paths, scenario_options_path, feature_environment, resolved


def rdagent_command(
    settings: Settings,
    *,
    project_root: Path,
    trace_path: Path,
    result_path: Path,
    dataset_path: Path | None,
    loop_n: int,
    duration: str,
    periods: dict[str, str] | None,
    objective: str,
    scenario: str = "fin_factor",
    asset_ids: list[str] | None = None,
    asset_manifest_sha256: dict[str, str] | None = None,
    feature_set: dict[str, Any] | None = None,
    strategy_horizon_profile: str | None = None,
    incumbent_strategy_version_id: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    scenario_spec = get_rdagent_scenario(scenario)
    duration = validate_duration_limit(duration, settings.rdagent_max_duration)
    if loop_n < 1 or loop_n > settings.rdagent_max_loops:
        raise ValueError(f"loop_n must be between 1 and {settings.rdagent_max_loops}")
    if not settings.rdagent_enabled:
        raise ValueError("RD-Agent execution is disabled")
    if scenario_spec.id == "fin_strategy":
        if strategy_horizon_profile not in {SHORT_1_5D, SWING_1_6M, LONG_1_3Y}:
            raise ValueError("fin_strategy requires one governed strategy horizon")
        if incumbent_strategy_version_id is not None and not re.fullmatch(
            r"[0-9a-f]{32}", incumbent_strategy_version_id
        ):
            raise ValueError("fin_strategy incumbent strategy version identity is invalid")
    elif strategy_horizon_profile is not None or incumbent_strategy_version_id is not None:
        raise ValueError("strategy horizon bindings are accepted only by fin_strategy")
    runtime_root = _shared_runtime_root(settings, trace_path)
    is_wsl = os.name == "nt" and settings.rdagent_command.startswith("/")
    trace_arg = _to_wsl_path(trace_path) if is_wsl else str(trace_path)
    result_arg = _to_wsl_path(result_path) if is_wsl else str(result_path)
    bridge_arg = (
        _to_wsl_path(project_root / "scripts" / "rdagent_bridge.py")
        if is_wsl
        else str(project_root / "scripts" / "rdagent_bridge.py")
    )
    runner_arg = (
        _to_wsl_path(project_root / "scripts" / "run_rdagent_scenario.py")
        if is_wsl
        else str(project_root / "scripts" / "run_rdagent_scenario.py")
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
            "--scenario",
            scenario_spec.id,
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
            runner_arg,
            "--command",
            settings.rdagent_command,
            "--scenario",
            scenario_spec.id,
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
    asset_paths, scenario_options_path, feature_environment, resolved_assets = (
        _prepare_scenario_inputs(
            settings,
            scenario_spec,
            trace_path=trace_path,
            asset_ids=asset_ids,
            asset_manifest_sha256=asset_manifest_sha256,
            feature_set=feature_set,
            periods=periods,
            loop_n=loop_n,
        )
    )
    for path in asset_paths:
        command.extend(["--asset", _runtime_path(Path(path), is_wsl=is_wsl)])
    if scenario_options_path is not None:
        command.extend(
            ["--scenario-options", _runtime_path(scenario_options_path, is_wsl=is_wsl)]
        )
    if feature_set is not None:
        command.extend(
            [
                "--feature-set-id",
                str(feature_set["id"]),
                "--feature-set-sha256",
                str(feature_set["definition_sha256"]),
                "--base-features",
                _runtime_path(
                    trace_path.parent / "scenario-inputs" / "base-features",
                    is_wsl=is_wsl,
                ),
            ]
        )
    for asset_id in asset_ids or []:
        command.extend(["--asset-id", asset_id])

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
        "QUANTLAB_RESEARCH_OBJECTIVE": objective,
        "QUANTLAB_STRATEGY_HORIZON": strategy_horizon_profile or "",
        "QUANTLAB_STRATEGY_PARENT_VERSION_ID": incumbent_strategy_version_id or "",
        "RDAGENT_QLIB_SANDBOX_IMAGE": settings.rdagent_qlib_sandbox_image,
        "RDAGENT_DATA_SCIENCE_IMAGE": settings.rdagent_data_science_image,
        "RDAGENT_FINETUNE_IMAGE": settings.rdagent_finetune_image,
        "RDAGENT_FINETUNE_BENCHMARK_IMAGE": settings.rdagent_finetune_benchmark_image,
        "RDAGENT_FINETUNE_GPU_PROBE_IMAGE": settings.rdagent_finetune_gpu_probe_image,
        # RD-Agent workspaces are mounted into Docker by absolute path. Keep
        # them on the platform-owned shared data volume, disable pickle resume,
        # and never depend on the worker image's private /app filesystem.
        "WORKSPACE_PATH": _runtime_path(runtime_root / "workspace", is_wsl=is_wsl),
        "PICKLE_CACHE_FOLDER_PATH_STR": _runtime_path(
            runtime_root / "disabled-pickle-cache", is_wsl=is_wsl
        ),
        "CACHE_WITH_PICKLE": "false",
        "USE_FILE_LOCK": "true",
        # Force every CoSTEER family that supports a selectable environment to
        # use the isolated Docker implementation. Values are platform-owned.
        "MODEL_CoSTEER_env_type": "docker",
        "DS_Coder_CoSTEER_env_type": "docker",
        "DS_Runner_CoSTEER_env_type": "docker",
        "FT_Coder_CoSTEER_env_type": "docker",
        "QLIB_DOCKER_NETWORK": "none",
        "QLIB_DOCKER_BUILD_FROM_DOCKERFILE": "false",
        "QLIB_DOCKER_IMAGE": settings.rdagent_qlib_sandbox_image,
        # The regular research queues are deliberately CPU-only.  RD-Agent's
        # upstream default probes NVIDIA on every Docker run and can leave
        # failed probe containers behind on a non-GPU host.
        "QLIB_DOCKER_ENABLE_GPU": "false",
        "DS_DOCKER_NETWORK": "none",
        "DS_DOCKER_BUILD_FROM_DOCKERFILE": "false",
        "DS_DOCKER_IMAGE": settings.rdagent_data_science_image,
        "DS_DOCKER_MEM_LIMIT": "8g",
        "DS_DOCKER_CPU_COUNT": "4",
        "DS_DOCKER_ENABLE_GPU": "false",
        "FT_DOCKER_NETWORK": "none",
        "BENCHMARK_DOCKER_NETWORK": "none",
        **feature_environment,
    }
    if scenario_spec.requires_dataset:
        if dataset_path is None or periods is None:
            raise ValueError(f"{scenario_spec.id} requires a governed Qlib dataset")
        # Derive the internal train/valid/test windows from pre-final history and
        # physically truncate the provider. The final OOS never enters feedback.
        rdagent_periods = isolate_rdagent_periods(periods)
        provenance_path = dataset_path / "metadata" / "provenance.json"
        try:
            dataset_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "RD-Agent source Qlib dataset provenance is missing or invalid"
            ) from exc
        verify_qlib_output_manifest(dataset_path, dataset_provenance)
        dataset_snapshot_id = str(
            dataset_provenance.get("dataset_identity_sha256")
            or hashlib.sha256(
                json.dumps(
                    dataset_provenance,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
        research_dataset_path = prepare_rdagent_dataset_view(
            dataset_path,
            trace_path.parent / "research-dataset",
            cutoff=rdagent_periods["test_end"],
        )
        qlib_home_host = trace_path.parent / "qlib-home"
        qlib_home_host_str = _runtime_path(qlib_home_host, is_wsl=is_wsl)
        dataset_host_str = _runtime_path(research_dataset_path, is_wsl=is_wsl)
        runtime_root_str = _runtime_path(runtime_root, is_wsl=is_wsl)
        period_names = (
            "train_start",
            "train_end",
            "valid_start",
            "valid_end",
            "test_start",
            "test_end",
        )
        # The official quant loop constructs factor and model runners from their
        # own settings classes.  Supplying only QLIB_QUANT_* therefore lets the
        # component runners silently fall back to upstream dates.  Bind all
        # three official settings groups to the same isolated, pre-final view;
        # the runner validates the effective YAML environment again at execute.
        for prefix in ("QLIB_FACTOR", "QLIB_MODEL", "QLIB_QUANT"):
            for name in period_names:
                env[f"{prefix}_{name.upper()}"] = rdagent_periods[name]
        env.update(
            {
                # Never depend on the upstream conda/default execution mode.
                # Candidate code is executed by the dedicated Docker daemon;
                # image preparation is a deployment preflight, not a runtime
                # reason to grant generated code network access.
                "MODEL_COSTEER_ENV_TYPE": "docker",
                "QLIB_DOCKER_NETWORK": "none",
                "QLIB_DOCKER_ENABLE_CACHE": "false",
                "QLIB_DOCKER_BUILD_FROM_DOCKERFILE": "false",
                "QLIB_DOCKER_IMAGE": settings.rdagent_qlib_sandbox_image,
                "QLIB_DOCKER_MEM_LIMIT": "16g",
                "QLIB_DOCKER_CPU_COUNT": "4",
                "QLIB_FACTOR_SCEN": (
                    "quant_platform.rdagent_scenario.QuantLabFactorFromReportScenario"
                    if scenario_spec.id == "fin_factor_report"
                    else "quant_platform.rdagent_scenario.QuantLabFactorScenario"
                ),
                "QLIB_MODEL_SCEN": "quant_platform.rdagent_scenario.QuantLabModelScenario",
                "QLIB_QUANT_SCEN": "quant_platform.rdagent_scenario.QuantLabQuantScenario",
                "QLIB_FACTOR_RUNNER": (
                    "quant_platform.rdagent_runner.QuantLabFactorRunner"
                ),
                "QLIB_MODEL_RUNNER": (
                    "quant_platform.rdagent_runner.QuantLabModelRunner"
                ),
                "QLIB_QUANT_FACTOR_RUNNER": (
                    "quant_platform.rdagent_runner.QuantLabFactorRunner"
                ),
                "QLIB_QUANT_MODEL_RUNNER": (
                    "quant_platform.rdagent_runner.QuantLabModelRunner"
                ),
                "QUANTLAB_DATASET_SNAPSHOT_ID": dataset_snapshot_id,
                # The pinned factor coder lazily creates daily_pv.h5. Give
                # every research run its own source-data directories so a
                # later/earlier cutoff cannot reuse another run's panel.
                "FACTOR_CoSTEER_data_folder": _runtime_path(
                    runtime_root / "factor-source-data" / "full",
                    is_wsl=is_wsl,
                ),
                "FACTOR_CoSTEER_data_folder_debug": _runtime_path(
                    runtime_root / "factor-source-data" / "debug",
                    is_wsl=is_wsl,
                ),
                "QLIB_DOCKER_EXTRA_VOLUMES": json.dumps(
                    {
                        qlib_home_host_str: {"bind": "/root/.qlib/", "mode": "rw"},
                        dataset_host_str: {
                            "bind": "/root/.qlib/qlib_data/cn_data",
                            "mode": "ro",
                        },
                        # Generated workspaces contain links into this
                        # run-scoped source-data tree.  Expose the tree at the
                        # same absolute path read-only so the Qlib container
                        # can materialize those inputs before launching the
                        # narrower networkless factor sandbox.
                        runtime_root_str: {
                            "bind": runtime_root_str,
                            "mode": "ro",
                        },
                    },
                    separators=(",", ":"),
                ),
            }
        )
        (qlib_home_host / "qlib_data" / "cn_data").mkdir(parents=True, exist_ok=True)
    elif scenario_spec.id == "data_science" and resolved_assets:
        env.update(
            {
                "DS_LOCAL_DATA_PATH": _runtime_path(
                    Path(resolved_assets["assets"][0]["path"]).parent,
                    is_wsl=is_wsl,
                ),
                "DS_SCEN": "rdagent.scenarios.data_science.scen.DataScienceScen",
                "DS_CODER_COSTEER_ENV_TYPE": "docker",
                "DS_DOCKER_NETWORK": "none",
                "DS_DOCKER_ENABLE_GPU": "false",
                "DS_DOCKER_ENABLE_CACHE": "false",
                "DS_DOCKER_BUILD_FROM_DOCKERFILE": "false",
                "DS_DOCKER_IMAGE": settings.rdagent_data_science_image,
                "DS_DOCKER_MEM_LIMIT": "16g",
                "DS_DOCKER_CPU_COUNT": "4",
            }
        )
    elif scenario_spec.id == "llm_finetune":
        if not resolved_assets:
            raise ValueError("llm_finetune requires a governed asset bundle")
        finetune_root, staged_inventory_sha256 = _stage_finetune_asset(
            resolved_assets,
            runtime_root=runtime_root,
            audit_root=trace_path.parent,
        )
        env.update(
            {
                "FT_FILE_PATH": _runtime_path(finetune_root, is_wsl=is_wsl),
                "FT_SCEN": (
                    "quant_platform.rdagent_scenario."
                    "QuantLabOfflineFinetuneScenario"
                ),
                "FT_CODER_COSTEER_ENV_TYPE": "docker",
                "FT_DOCKER_ENABLE_CACHE": "false",
                "FT_DOCKER_BUILD_FROM_DOCKERFILE": "false",
                "FT_DOCKER_IMAGE": settings.rdagent_finetune_image,
                "FT_DOCKER_NETWORK": "none",
                "FT_DOCKER_MEM_LIMIT": "48g",
                "FT_DOCKER_CPU_COUNT": "8",
                "BENCHMARK_DOCKER_BUILD_FROM_DOCKERFILE": "false",
                "BENCHMARK_DOCKER_IMAGE": settings.rdagent_finetune_benchmark_image,
                "BENCHMARK_DOCKER_NETWORK": "none",
                "QUANTLAB_BLOCK_GENERATED_SECRETS": "true",
                "QUANTLAB_FINETUNE_STAGED_INVENTORY_SHA256": staged_inventory_sha256,
            }
        )
    if is_wsl:
        source_root = _to_wsl_path(project_root / "src")
        env["PYTHONPATH"] = ":".join(
            item for item in [source_root, os.getenv("PYTHONPATH", "")] if item
        )
        forwarded = [*env, settings.rdagent_llm_key_env]
        existing = os.getenv("WSLENV", "")
        env["WSLENV"] = ":".join([item for item in [existing, *forwarded] if item])
    return command, env
