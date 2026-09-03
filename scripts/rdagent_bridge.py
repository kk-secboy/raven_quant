#!/usr/bin/env python3
"""Trusted bridge executed inside the pinned RD-Agent environment.

It emits JSON only. Pickle reading is deliberately kept out of the API process and
must only target trace folders produced by the configured RD-Agent runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

RDAGENT_COMMIT = "4f9ecb005881cddc08df0124a2e894c018007679"
FIN_QUANT_REQUIRED_ABLATIONS = (
    "factor_only",
    "model_only",
    "joint",
    "joint_vs_incumbent",
)


def _version() -> str:
    for distribution in ("rdagent", "rd-agent"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def _commit_from_version(version: str) -> str | None:
    match = re.search(r"(?:\+g|\.g)([0-9a-f]{7,40})(?:\b|$)", version.lower())
    if not match:
        return None
    candidate = match.group(1)
    if RDAGENT_COMMIT.startswith(candidate):
        return RDAGENT_COMMIT
    return candidate if len(candidate) == 40 else None


def _repo_commit(path: str | None) -> str | None:
    if not path:
        return None
    root = Path(path)
    if not (root / ".git").exists():
        return None
    try:
        value = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip().lower()
    except (OSError, subprocess.SubprocessError):
        return None
    return value if len(value) == 40 else None


def _source_tree_sha256(path: str | None) -> str | None:
    if not path:
        return None
    root = Path(path).resolve()
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    files = sorted(
        item
        for item in root.rglob("*")
        if item.is_file()
        and not item.is_symlink()
        and not any(
            part in {".git", "__pycache__", "build"}
            or part.endswith(".egg-info")
            for part in item.relative_to(root).parts
        )
        and not item.name.endswith((".pyc", ".pyo"))
    )
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _repo_dirty(path: str | None) -> bool | None:
    if not path or not (Path(path) / ".git").exists():
        return None
    try:
        output = subprocess.run(
            ["git", "-C", path, "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(output.strip())


def _runtime_identity() -> dict[str, Any]:
    version = _version()
    environment_commit = str(os.getenv("RDAGENT_COMMIT") or "").lower()
    if environment_commit and environment_commit != RDAGENT_COMMIT:
        raise RuntimeError(
            "RD-Agent configured commit does not match the validated pin: "
            f"expected {RDAGENT_COMMIT}, got {environment_commit}"
        )
    if version == "unknown":
        raise RuntimeError("RD-Agent runtime distribution version is unavailable")
    verified = {
        source: value
        for source, value in {
            "repository": _repo_commit(os.getenv("RDAGENT_REPO")),
            "distribution": _commit_from_version(version),
        }.items()
        if value
    }
    if not verified:
        raise RuntimeError(
            "RD-Agent runtime commit has no verifiable repository or distribution evidence"
        )
    if len(set(verified.values())) != 1:
        raise RuntimeError(f"RD-Agent runtime commit evidence disagrees: {verified}")
    commit = next(iter(verified.values()))
    if commit != RDAGENT_COMMIT:
        raise RuntimeError(
            "RD-Agent runtime commit is not the validated pin: "
            f"expected {RDAGENT_COMMIT}, got {commit}"
        )
    repository = os.getenv("RDAGENT_REPO")
    image_digest = str(os.getenv("RDAGENT_RUNTIME_IMAGE_DIGEST") or "").strip().lower()
    if image_digest and not re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest):
        raise RuntimeError("RDAGENT_RUNTIME_IMAGE_DIGEST is invalid")
    source_tree_sha256 = _source_tree_sha256(repository)
    repository_dirty = _repo_dirty(repository)
    return {
        "name": "rdagent",
        "version": version,
        "commit": commit,
        "commit_evidence": sorted(verified),
        "source_tree_sha256": source_tree_sha256,
        "repository_dirty": repository_dirty,
        "runtime_image_digest": image_digest or None,
        "production_reproducible": bool(
            source_tree_sha256 and image_digest and repository_dirty is not True
        ),
    }


def _costeer_knowledge_status() -> dict[str, Any]:
    raw = str(os.getenv("QUANTLAB_COSTEER_KNOWLEDGE_STATUS_JSON") or "").strip()
    if not raw:
        raise RuntimeError("CoSTEER knowledge status was not recorded by the governed launcher")
    try:
        status = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("CoSTEER knowledge status is invalid") from exc
    if not isinstance(status, dict) or status.get("contract_version") != (
        "costeer-knowledge-status-v2"
    ):
        raise RuntimeError("CoSTEER knowledge status contract drifted")
    if status.get("status") not in {
        "embedding_retrieval_configured",
        "unconfigured_fail_closed",
    }:
        raise RuntimeError("CoSTEER knowledge status value is unsupported")
    if not isinstance(status.get("embedding_retrieval_configured"), bool):
        raise RuntimeError("CoSTEER knowledge status readiness is invalid")
    if not isinstance(status.get("costeer_used"), bool):
        raise RuntimeError("CoSTEER knowledge use status is invalid")
    return status


def _strategy_feature_ids(args: argparse.Namespace) -> set[str] | None:
    if args.scenario != "fin_strategy":
        return None
    if not args.base_features or not args.feature_set_id or not args.feature_set_sha256:
        raise RuntimeError("fin_strategy export requires its governed feature-set evidence")
    root = Path(args.base_features).expanduser().resolve(strict=True)
    try:
        base_factors = json.loads((root / "base_factors.json").read_text(encoding="utf-8"))
        definition = json.loads((root / "definition.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("fin_strategy staged feature set is unreadable") from exc
    from quant_platform.feature_set_registry import resolve_feature_set

    expected = resolve_feature_set(args.feature_set_id, definition)
    if (
        expected["definition_sha256"] != args.feature_set_sha256
        or definition != expected
        or base_factors != expected["features"]
    ):
        raise RuntimeError("fin_strategy staged feature-set evidence disagrees")
    return set(expected["features"])


def _is_complete_qlib_provider(provider: Path, *, data_root: Path) -> bool:
    """Return whether *provider* is a safe, structurally complete Qlib tree.

    The Docker smoke below remains the authoritative runtime check.  This
    filter only prevents incomplete/stale directories (or paths escaping the
    governed data mount) from being handed to the sandbox in the first place.
    """

    try:
        governed_root = data_root.expanduser().resolve(strict=True)
        resolved = provider.expanduser().resolve(strict=True)
        resolved.relative_to(governed_root)
        calendar = resolved / "calendars" / "day.txt"
        instruments = resolved / "instruments" / "cn_all.txt"
        features = resolved / "features"
        return (
            calendar.is_file()
            and calendar.stat().st_size > 0
            and instruments.is_file()
            and instruments.stat().st_size > 0
            and features.is_dir()
            and next(features.glob("*/*.day.bin"), None) is not None
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _qlib_provider_candidates(data_root: Path, qlib_home: Path) -> list[Path]:
    """Discover governed Qlib providers across current and legacy layouts.

    Production datasets live below ``/data/qlib``.  Older releases may have
    placed exported providers below ``/data/artifacts/qlib`` or at the
    configured Qlib home.  Resolve and deduplicate every candidate, then keep
    only complete providers contained by the governed data root.
    """

    data_root = data_root.expanduser()
    qlib_home = qlib_home.expanduser()
    current_root = data_root / "qlib"
    legacy_root = data_root / "artifacts" / "qlib"
    calendars = [qlib_home / "calendars" / "day.txt"]
    # These patterns are deliberately bounded.  A recursive ``**`` below a
    # Qlib tree would walk every instrument/feature file on each health probe.
    calendars.extend(current_root.glob("*/calendars/day.txt"))
    calendars.extend(legacy_root.glob("*/calendars/day.txt"))
    calendars.extend(legacy_root.glob("*/*/calendars/day.txt"))
    discovered: dict[str, Path] = {}
    for calendar in calendars:
        try:
            provider = calendar.parent.parent.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if _is_complete_qlib_provider(provider, data_root=data_root):
            discovered[str(provider)] = provider

    def freshness(provider: Path) -> tuple[str, str]:
        try:
            days = [
                line.strip()
                for line in (provider / "calendars" / "day.txt").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            end_date = days[-1] if days else ""
        except OSError:
            end_date = ""
        return end_date, str(provider)

    return sorted(discovered.values(), key=freshness, reverse=True)


def probe(args: argparse.Namespace) -> dict[str, Any]:
    import rdagent  # noqa: F401
    from pydantic_ai.mcp import MCPServerStreamableHTTP  # noqa: F401
    from pydantic_ai.providers.litellm import LiteLLMProvider  # noqa: F401

    qlib_home = Path(args.qlib_home).expanduser()
    required = [
        qlib_home / "calendars" / "day.txt",
        qlib_home / "instruments" / "cn_all.txt",
        qlib_home / "features",
    ]
    qlib_data_ready = (
        required[0].is_file()
        and required[0].stat().st_size > 0
        and required[1].is_file()
        and required[1].stat().st_size > 0
        and required[2].is_dir()
    )
    docker_cli = shutil.which("docker")
    docker_available = False
    docker_gpu_runtime_available = False
    docker_gpu_evidence: dict[str, Any] = {}
    finetune_images_preloaded = False
    finetune_image_evidence: dict[str, Any] = {}
    docker_gpu_smoke_passed = False
    qlib_sandbox_preloaded = False
    qlib_smoke_passed = False
    data_science_sandbox_preloaded = False
    data_science_smoke_passed = False
    if docker_cli:
        try:
            docker_available = (
                subprocess.run(
                    [docker_cli, "info"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                ).returncode
                == 0
            )
            if docker_available:
                runtime_probe = subprocess.run(
                    [docker_cli, "info", "--format", "{{json .Runtimes}}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                runtimes: dict[str, Any] = {}
                if runtime_probe.returncode == 0 and runtime_probe.stdout.strip():
                    parsed = json.loads(runtime_probe.stdout)
                    if isinstance(parsed, dict):
                        runtimes = parsed
                cdi_probe = subprocess.run(
                    [docker_cli, "info", "--format", "{{json .CDISpecDirs}}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                cdi_dirs: list[str] = []
                if cdi_probe.returncode == 0 and cdi_probe.stdout.strip():
                    parsed_cdi = json.loads(cdi_probe.stdout)
                    if isinstance(parsed_cdi, list):
                        cdi_dirs = [str(item) for item in parsed_cdi]
                docker_gpu_runtime_available = "nvidia" in runtimes or bool(cdi_dirs)
                docker_gpu_evidence = {
                    "runtimes": sorted(str(item) for item in runtimes),
                    "cdi_spec_dirs": cdi_dirs,
                }
        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            subprocess.SubprocessError,
        ):
            docker_available = False
    identity = _runtime_identity()
    gpu_devices: list[str] = []
    gpu_memory_free_mb = 0
    gpu_driver_version: str | None = None
    cuda_version: str | None = None
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            for line in completed.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 3:
                    continue
                gpu_devices.append(fields[0])
                gpu_memory_free_mb = max(gpu_memory_free_mb, int(float(fields[1])))
                gpu_driver_version = gpu_driver_version or fields[2]
            version_probe = subprocess.run(
                [nvidia_smi], capture_output=True, text=True, timeout=10, check=True
            )
            match = re.search(r"CUDA Version:\s*([0-9.]+)", version_probe.stdout)
            cuda_version = match.group(1) if match else None
        except (OSError, TypeError, ValueError, subprocess.SubprocessError):
            gpu_devices = []
            gpu_memory_free_mb = 0
            gpu_driver_version = None
            cuda_version = None
    image_names = {
        "finetune": str(os.getenv("RDAGENT_FINETUNE_IMAGE") or "").strip(),
        "benchmark": str(
            os.getenv("RDAGENT_FINETUNE_BENCHMARK_IMAGE") or ""
        ).strip(),
        "gpu_probe": str(
            os.getenv("RDAGENT_FINETUNE_GPU_PROBE_IMAGE") or ""
        ).strip(),
    }
    immutable_images = bool(image_names) and all(
        re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image)
        for image in image_names.values()
    )
    if docker_available and immutable_images and docker_cli:
        try:
            for name, image in image_names.items():
                inspected = subprocess.run(
                    [docker_cli, "image", "inspect", "--format", "{{.Id}}", image],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=True,
                ).stdout.strip()
                if not re.fullmatch(r"sha256:[0-9a-f]{64}", inspected):
                    raise ValueError("Docker image has no immutable image ID")
                finetune_image_evidence[name] = {
                    "reference": image,
                    "image_id": inspected,
                }
            finetune_images_preloaded = True
            smoke = subprocess.run(
                [
                    docker_cli,
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--gpus",
                    "all",
                    "--entrypoint",
                    "nvidia-smi",
                    image_names["gpu_probe"],
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=45,
                check=True,
            )
            docker_gpu_smoke_passed = bool(smoke.stdout.strip())
            if docker_gpu_smoke_passed:
                finetune_image_evidence["gpu_smoke_sha256"] = hashlib.sha256(
                    smoke.stdout.encode()
                ).hexdigest()
        except (OSError, TypeError, ValueError, subprocess.SubprocessError):
            finetune_images_preloaded = False
            docker_gpu_smoke_passed = False
    data_root = Path(args.data_root).expanduser()
    try:
        data_root_free_gb = shutil.disk_usage(data_root).free / (1024**3)
    except OSError:
        data_root_free_gb = 0.0
    sandbox_images = {
        "qlib": str(os.getenv("RDAGENT_QLIB_SANDBOX_IMAGE") or "").strip(),
        "data_science": str(
            os.getenv("RDAGENT_DATA_SCIENCE_IMAGE") or ""
        ).strip(),
    }
    if docker_available and docker_cli:
        for name, image in sandbox_images.items():
            if not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image):
                continue
            try:
                image_id = subprocess.run(
                    [docker_cli, "image", "inspect", "--format", "{{.Id}}", image],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=True,
                ).stdout.strip()
                if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
                    continue
                if name == "qlib":
                    qlib_sandbox_preloaded = True
                else:
                    data_science_sandbox_preloaded = True
            except (OSError, subprocess.SubprocessError):
                continue
        if data_science_sandbox_preloaded:
            try:
                data_science_smoke_passed = (
                    subprocess.run(
                        [
                            docker_cli,
                            "run",
                            "--rm",
                            "--network",
                            "none",
                            "--entrypoint",
                            "python",
                            sandbox_images["data_science"],
                            "-c",
                            "import pandas, sklearn; print('quantlab-ds-smoke-ok')",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=45,
                        check=True,
                    ).returncode
                    == 0
                )
            except (OSError, subprocess.SubprocessError):
                data_science_smoke_passed = False
        qlib_candidates = _qlib_provider_candidates(data_root, qlib_home)
        if qlib_sandbox_preloaded and qlib_candidates:
            for provider in qlib_candidates:
                try:
                    qlib_smoke_passed = (
                        subprocess.run(
                            [
                                docker_cli,
                                "run",
                                "--rm",
                                "--network",
                                "none",
                                "--volume",
                                f"{provider}:/root/.qlib/qlib_data/cn_data:ro",
                                "--entrypoint",
                                "python",
                                sandbox_images["qlib"],
                                "-c",
                                (
                                    "import qlib; from qlib.data import D; "
                                    "qlib.init(provider_uri='/root/.qlib/qlib_data/cn_data'); "
                                    "assert len(D.calendar(freq='day')) > 0; "
                                    "assert D.list_instruments("
                                    "D.instruments('cn_all'), freq='day', as_list=True); "
                                    "print('quantlab-qlib-smoke-ok')"
                                ),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=60,
                            check=True,
                        ).returncode
                        == 0
                    )
                except (OSError, subprocess.SubprocessError):
                    qlib_smoke_passed = False
                if qlib_smoke_passed:
                    qlib_data_ready = True
                    qlib_home = provider
                    break
    return {
        "status": "ok",
        "version": identity["version"],
        "commit": identity["commit"],
        "commit_evidence": identity["commit_evidence"],
        "source_tree_sha256": identity["source_tree_sha256"],
        "repository_dirty": identity["repository_dirty"],
        "runtime_image_digest": identity["runtime_image_digest"],
        "production_reproducible": identity["production_reproducible"],
        "python": os.sys.executable,
        "docker_available": docker_available,
        "docker_gpu_runtime_available": docker_gpu_runtime_available,
        "docker_gpu_evidence": docker_gpu_evidence,
        "docker_gpu_smoke_passed": docker_gpu_smoke_passed,
        "finetune_images_preloaded": finetune_images_preloaded,
        "finetune_image_evidence": finetune_image_evidence,
        "qlib_sandbox_preloaded": qlib_sandbox_preloaded,
        "qlib_smoke_passed": qlib_smoke_passed,
        "data_science_sandbox_preloaded": data_science_sandbox_preloaded,
        "data_science_smoke_passed": data_science_smoke_passed,
        "gpu_available": bool(gpu_devices),
        "gpu_devices": gpu_devices,
        "gpu_memory_free_mb": gpu_memory_free_mb,
        "gpu_driver_version": gpu_driver_version,
        "cuda_version": cuda_version,
        "data_root_free_gb": round(data_root_free_gb, 3),
        "hf_credentials_configured": bool(
            os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        ),
        "qlib_data_ready": qlib_data_ready,
        "qlib_home": str(qlib_home),
        "llm_credentials_configured": bool(os.getenv(args.llm_key_env)),
        "llm_key_env": args.llm_key_env,
    }


def _loop_id(tag: str) -> int | None:
    for part in tag.split("."):
        if part.startswith("Loop_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                return None
    return None


def _trace_text(value: Any, *, limit: int = 2_000) -> str:
    """Project human-readable Trace text without exporting code or paths."""

    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _trace_loop_projection(rounds: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the read-only Web view from official RDLoop Trace messages."""

    projected: list[dict[str, Any]] = []
    for loop_id, item in sorted(rounds.items())[:20]:
        hypothesis = item.get("hypothesis")
        hypothesis = hypothesis if isinstance(hypothesis, dict) else {}
        feedback = item.get("feedback")
        feedback = feedback if isinstance(feedback, dict) else {}
        tasks: list[dict[str, str]] = []
        for kind, source in (
            ("factor", item.get("tasks")),
            ("model", item.get("model_tasks")),
        ):
            for task in source if isinstance(source, list) else []:
                if not isinstance(task, dict):
                    continue
                tasks.append(
                    {
                        "kind": kind,
                        "name": _trace_text(task.get("name"), limit=200),
                        "description": _trace_text(
                            task.get("description"), limit=1_000
                        ),
                    }
                )
        implementation_feedback: list[dict[str, Any]] = []
        raw_implementation = item.get("implementation_feedback")
        for decision in raw_implementation if isinstance(raw_implementation, list) else []:
            if not isinstance(decision, dict):
                continue
            implementation_feedback.append(
                {
                    "decision": bool(decision.get("decision")),
                    "feedback": _trace_text(decision.get("feedback")),
                }
            )
        projected.append(
            {
                "loop_id": int(loop_id),
                "hypothesis": {
                    "text": _trace_text(hypothesis.get("hypothesis")),
                    "reason": _trace_text(hypothesis.get("reason")),
                    "action": _trace_text(hypothesis.get("action"), limit=100),
                },
                "tasks": tasks[:20],
                "feedback": {
                    "recorded": bool(feedback),
                    "decision": bool(feedback.get("decision")),
                    "reason": _trace_text(feedback.get("reason")),
                    "hypothesis_evaluation": _trace_text(
                        feedback.get("hypothesis_evaluation")
                    ),
                },
                "implementation_feedback": implementation_feedback[:20],
            }
        )
    return projected


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mapping(value: Any, *, fallback_key: str = "description") -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if value is None:
        return {}
    return {fallback_key: str(value)}


def _task_payload(task: Any) -> tuple[str, dict[str, Any]] | None:
    """Extract one typed task without relying on a fuzzy trace tag."""

    if hasattr(task, "factor_name"):
        return (
            "factor",
            {
                "name": str(getattr(task, "factor_name", "unnamed_factor")),
                "description": str(getattr(task, "factor_description", "")),
                "formulation": getattr(task, "factor_formulation", None),
                "variables": getattr(task, "variables", {}) or {},
            },
        )
    if hasattr(task, "model_type") and hasattr(task, "name"):
        return (
            "model",
            {
                "name": str(getattr(task, "name", "unnamed_model")),
                "description": str(getattr(task, "description", "")),
                "model_type": str(getattr(task, "model_type", "")),
                "architecture": _mapping(getattr(task, "architecture", None)),
                "model_hyperparameters": _mapping(
                    getattr(task, "hyperparameters", None)
                ),
                "training_hyperparameters": _mapping(
                    getattr(task, "training_hyperparameters", None)
                ),
            },
        )
    return None


def _runner_snapshot(content: Any) -> dict[str, Any] | None:
    """Freeze executable evidence from the official runner-result experiment."""

    tasks = list(getattr(content, "sub_tasks", []) or [])
    workspaces = list(getattr(content, "sub_workspace_list", []) or [])
    if not tasks or len(tasks) != len(workspaces):
        return None
    typed = [_task_payload(task) for task in tasks]
    kinds = {item[0] for item in typed if item is not None}
    if len(kinds) != 1 or any(item is None for item in typed):
        return None
    kind = kinds.pop()
    artifacts: list[dict[str, Any]] = []
    for typed_task, workspace in zip(typed, workspaces, strict=True):
        assert typed_task is not None
        files = getattr(workspace, "file_dict", {}) or {}
        filename = "factor.py" if kind == "factor" else "model.py"
        code = files.get(filename)
        if not isinstance(code, str) or not code.strip():
            return None
        source_values = None
        if kind == "factor":
            workspace_path = getattr(workspace, "workspace_path", None)
            candidate_path = (
                Path(workspace_path) / "result.h5" if workspace_path else None
            )
            if candidate_path is not None and candidate_path.is_file():
                source_values = str(candidate_path)
        artifacts.append(
            {
                **typed_task[1],
                "code": code,
                "code_sha256": _sha256(code),
                "source_values_path": source_values,
            }
        )
    base_features = getattr(content, "base_features", {}) or {}
    if not isinstance(base_features, dict):
        return None
    based = []
    for experiment in list(getattr(content, "based_experiments", []) or []):
        based_tasks = list(getattr(experiment, "sub_tasks", []) or [])
        based_workspaces = list(getattr(experiment, "sub_workspace_list", []) or [])
        if len(based_tasks) != len(based_workspaces):
            continue
        for task, workspace in zip(based_tasks, based_workspaces, strict=True):
            task_payload = _task_payload(task)
            files = getattr(workspace, "file_dict", {}) or {} if workspace else {}
            if task_payload is None:
                continue
            filename = "factor.py" if task_payload[0] == "factor" else "model.py"
            code = files.get(filename)
            if isinstance(code, str) and code.strip():
                based.append(
                    {
                        "kind": task_payload[0],
                        **task_payload[1],
                        "code": code,
                        "code_sha256": _sha256(code),
                    }
                )
    return {
        "kind": kind,
        "artifacts": artifacts,
        "based_artifacts": based,
        "base_features": {str(key): str(value) for key, value in base_features.items()},
    }


def _export_workspace_sources(content: Any, root: Path, loop_id: int) -> int:
    """Copy code/config held by runner workspaces into the governed run root."""

    workspaces = list(getattr(content, "sub_workspace_list", []) or [])
    experiment_workspace = getattr(content, "experiment_workspace", None)
    if experiment_workspace is not None:
        workspaces.append(experiment_workspace)
    exported = 0
    for workspace_index, workspace in enumerate(workspaces, start=1):
        files = getattr(workspace, "file_dict", {}) or {}
        if not isinstance(files, dict):
            continue
        for file_index, (raw_name, raw_content) in enumerate(
            sorted(files.items(), key=lambda item: str(item[0])), start=1
        ):
            relative = Path(str(raw_name))
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
                or len(relative.parts) > 8
            ):
                continue
            if isinstance(raw_content, str):
                payload = raw_content.encode("utf-8")
            elif isinstance(raw_content, bytes):
                payload = raw_content
            else:
                continue
            if len(payload) > 64 * 1024 * 1024:
                continue
            safe_parts = [
                "".join(
                    character if character.isalnum() or character in "-_." else "_"
                    for character in part
                )[:120]
                for part in relative.parts
            ]
            destination = (
                root
                / f"loop-{loop_id:03d}"
                / f"workspace-{workspace_index:03d}"
                / f"{file_index:03d}"
                / Path(*safe_parts)
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            exported += 1
    return exported


def _fin_quant_arm_coverage(rounds: dict[int, dict[str, Any]]) -> dict[str, Any]:
    arms = {
        "factor": {
            "proposed_rounds": [],
            "attempted_rounds": [],
            "executable_rounds": [],
            "accepted_executable_rounds": [],
        },
        "model": {
            "proposed_rounds": [],
            "attempted_rounds": [],
            "executable_rounds": [],
            "accepted_executable_rounds": [],
        },
    }
    for loop_id, round_item in sorted(rounds.items()):
        hypothesis = round_item.get("hypothesis") or {}
        action = hypothesis.get("action")
        if action is not None and action not in arms:
            raise RuntimeError(f"fin_quant trace contains unsupported action: {action}")
        if action in arms:
            arms[action]["proposed_rounds"].append(loop_id)
            arms[action]["attempted_rounds"].append(loop_id)
        snapshot = round_item.get("runner_snapshot")
        if not isinstance(snapshot, dict) or snapshot.get("kind") not in arms:
            continue
        kind = str(snapshot["kind"])
        if action in arms and kind != action:
            raise RuntimeError("fin_quant selected arm and runner artifact disagree")
        if loop_id not in arms[kind]["attempted_rounds"]:
            arms[kind]["attempted_rounds"].append(loop_id)
        artifacts = snapshot.get("artifacts")
        executable = isinstance(artifacts, list) and bool(artifacts)
        if executable:
            arms[kind]["executable_rounds"].append(loop_id)
            if (round_item.get("feedback") or {}).get("decision") is True:
                arms[kind]["accepted_executable_rounds"].append(loop_id)
    accepted_arms = [
        name for name, values in arms.items() if values["accepted_executable_rounds"]
    ]
    attempted_arms = [name for name, values in arms.items() if values["attempted_rounds"]]
    result_arms = {
        name: {
            **values,
            "attempted_in_this_run": bool(values["attempted_rounds"]),
            "accepted_in_this_run": bool(values["accepted_executable_rounds"]),
        }
        for name, values in arms.items()
    }
    return {
        "contract_version": "fin-quant-arm-coverage-v2",
        "arms": result_arms,
        "attempted_complete": len(attempted_arms) == 2,
        "complete": len(accepted_arms) == 2,
        "single_arm": len(accepted_arms) == 1,
        "attempted_arms": attempted_arms,
        "accepted_arms": accepted_arms,
        "based_artifacts_count_as_arm_coverage": False,
    }


def _fin_quant_research_outcome(
    coverage: dict[str, Any], bundles: list[dict[str, Any]]
) -> dict[str, Any]:
    """Classify joint research without granting it any capital authority."""

    if bundles:
        if coverage.get("complete") is not True:
            raise RuntimeError("fin_quant bundle escaped complete accepted arm coverage")
        status = "joint_proposal_ready"
        reason_code = None
    elif coverage.get("complete") is True:
        # Both accepted executable arms should always materialize an atomic
        # proposal.  Treat missing code/state as an engineering failure, not a
        # statistical negative.
        raise RuntimeError("fin_quant accepted both arms but materialized no joint proposal")
    else:
        status = "governed_negative"
        reason_code = (
            "arm_acceptance_incomplete"
            if coverage.get("attempted_complete") is True
            else "arm_attempt_coverage_incomplete"
        )
    return {
        "contract_version": "fin-quant-research-outcome-v1",
        "status": status,
        "reason_code": reason_code,
        "attempted_arms": list(coverage.get("attempted_arms") or []),
        "accepted_arms": list(coverage.get("accepted_arms") or []),
        "required_independent_ablations": list(FIN_QUANT_REQUIRED_ABLATIONS),
        "joint_ablation_completed": False,
        "capital_authority": False,
    }


def _materialize_quant_bundles(
    rounds: dict[int, dict[str, Any]],
    *,
    code_root: Path,
    values_root: Path,
    feature_set_id: str | None,
    feature_set_sha256: str | None,
) -> list[dict[str, Any]]:
    """Replay official feedback in order and freeze every accepted active state."""

    accepted_factors: dict[str, dict[str, Any]] = {}
    accepted_model: dict[str, Any] | None = None
    accepted_run_arms: set[str] = set()
    bundles: list[dict[str, Any]] = []
    for loop_id, round_item in sorted(rounds.items()):
        snapshot = round_item.get("runner_snapshot")
        feedback = round_item.get("feedback") or {}
        if not isinstance(snapshot, dict) or feedback.get("decision") is not True:
            continue
        kind = snapshot.get("kind")
        # Runner evidence may carry the active counterpart.  Treat it as trusted
        # only when its typed workspace contains executable code.
        for based in snapshot.get("based_artifacts") or []:
            if based.get("kind") == "factor":
                accepted_factors.setdefault(str(based["name"]), dict(based))
            elif based.get("kind") == "model" and accepted_model is None:
                accepted_model = dict(based)
        if kind == "factor":
            artifacts = list(snapshot.get("artifacts") or [])
            if not artifacts:
                continue
            for factor in artifacts:
                accepted_factors[str(factor["name"])] = dict(factor)
            accepted_run_arms.add("factor")
        elif kind == "model":
            artifacts = list(snapshot.get("artifacts") or [])
            if len(artifacts) != 1:
                continue
            accepted_model = dict(artifacts[0])
            accepted_run_arms.add("model")
        else:
            continue
        if (
            not accepted_factors
            or accepted_model is None
            or accepted_run_arms != {"factor", "model"}
        ):
            continue

        materialized_factors: list[dict[str, Any]] = []
        for index, factor in enumerate(accepted_factors.values(), start=1):
            code = factor.get("code")
            if not isinstance(code, str) or not code.strip():
                materialized_factors = []
                break
            safe_name = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in str(factor["name"])
            )[:80]
            code_path = code_root / f"quant-state-{loop_id:03d}-factor-{index:03d}-{safe_name}.py"
            code_path.write_text(code, encoding="utf-8")
            values_path = None
            source_values = factor.get("source_values_path")
            if source_values and Path(str(source_values)).is_file():
                destination = values_root / (
                    f"quant-state-{loop_id:03d}-factor-{index:03d}-{safe_name}.h5"
                )
                shutil.copy2(Path(str(source_values)), destination)
                values_path = str(destination)
            materialized_factors.append(
                {
                    "candidate_id": _sha256(
                        f"quant-state:{loop_id}:factor:{factor['name']}:{factor['code_sha256']}"
                    )[:32],
                    "name": factor["name"],
                    "description": factor.get("description"),
                    "formulation": factor.get("formulation"),
                    "variables": factor.get("variables") or {},
                    "code_path": str(code_path),
                    "code_sha256": factor["code_sha256"],
                    "submitted_values_path": values_path,
                }
            )
        model_code = accepted_model.get("code")
        if not materialized_factors or not isinstance(model_code, str) or not model_code.strip():
            continue
        model_path = code_root / f"quant-state-{loop_id:03d}-model.py"
        model_path.write_text(model_code, encoding="utf-8")
        recipe = {
            "model_type": accepted_model.get("model_type") or "Tabular",
            "architecture": accepted_model.get("architecture") or {},
            "model_hyperparameters": accepted_model.get("model_hyperparameters") or {},
            "training_hyperparameters": accepted_model.get("training_hyperparameters") or {},
        }
        recipe_sha256 = _sha256(
            json.dumps(recipe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        family = f"rdagent-fin-quant-state-{loop_id:03d}"
        bundles.append(
            {
                "id": _sha256(family)[:32],
                "name": family,
                "description": str(accepted_model.get("description") or family),
                "source_iteration": loop_id,
                "experiment_family_id": family,
                "feature_set_id": feature_set_id,
                "feature_set_definition_sha256": feature_set_sha256,
                "base_features": dict(snapshot.get("base_features") or {}),
                "factors": materialized_factors,
                "model": {
                    "code_path": str(model_path),
                    "code_sha256": accepted_model["code_sha256"],
                    "recipe_sha256": recipe_sha256,
                    **recipe,
                },
                "rdagent_decision": True,
                "rdagent_feedback": feedback.get("hypothesis_evaluation")
                or feedback.get("reason"),
                "arm_coverage": {"factor": True, "model": True},
                "single_arm": False,
                "delivery_status": "research_only",
                "required_independent_ablations": list(
                    FIN_QUANT_REQUIRED_ABLATIONS
                ),
                "joint_ablation_completed": False,
                "capital_authority": False,
            }
        )
    return bundles


def export_trace(args: argparse.Namespace) -> dict[str, Any]:
    from rdagent.log.storage import FileStorage

    trace = Path(args.trace).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    code_root = output.parent / "candidate-code"
    values_root = output.parent / "candidate-values"
    workspace_export_root = output.parent / "workspace-export"
    output.parent.mkdir(parents=True, exist_ok=True)
    code_root.mkdir(parents=True, exist_ok=True)
    values_root.mkdir(parents=True, exist_ok=True)
    workspace_export_root.mkdir(parents=True, exist_ok=True)

    rounds: dict[int, dict[str, Any]] = {}
    tag_counts: dict[str, int] = {}
    strategy_proposals: list[dict[str, Any]] = []
    strategy_proposal_hashes: set[str] = set()
    strategy_feature_ids = _strategy_feature_ids(args)
    expected_strategy_horizon = str(
        os.getenv("QUANTLAB_STRATEGY_HORIZON") or ""
    ).strip()
    expected_strategy_parent = str(
        os.getenv("QUANTLAB_STRATEGY_PARENT_VERSION_ID") or ""
    ).strip() or None
    raw_strategy_signal_binding = str(
        os.getenv("QUANTLAB_STRATEGY_SIGNAL_BINDING_JSON") or ""
    ).strip()
    expected_strategy_signal_binding = None
    if raw_strategy_signal_binding:
        from quant_platform.strategy_research_signal_binding import (
            validate_strategy_research_signal_binding,
        )

        expected_strategy_signal_binding = validate_strategy_research_signal_binding(
            json.loads(raw_strategy_signal_binding)
        )
    if args.scenario == "fin_strategy" and expected_strategy_horizon not in {
        "short_1_5d",
        "swing_1_6m",
        "long_1_3y",
    }:
        raise RuntimeError("fin_strategy export has no governed horizon binding")
    for message in FileStorage(trace).iter_msg():
        tag_counts[message.tag] = tag_counts.get(message.tag, 0) + 1
        content = message.content
        if args.scenario == "fin_strategy" and message.tag.endswith(
            "strategy compiled artifact"
        ):
            from quant_platform.strategy_rule_compiler import (
                validate_compiled_strategy_artifact,
            )

            artifact = validate_compiled_strategy_artifact(
                content,
                allowed_factor_ids=strategy_feature_ids,
            )
            data_contract = artifact["strategy_proposal"]["data_contract"]
            if (
                data_contract["feature_set_id"] != args.feature_set_id
                or data_contract["feature_set_definition_sha256"]
                != args.feature_set_sha256
                or data_contract["dataset_snapshot_id"]
                != str(os.getenv("QUANTLAB_DATASET_SNAPSHOT_ID") or "")
                or artifact["strategy_proposal"]["horizon"]
                != expected_strategy_horizon
                or artifact["strategy_proposal"]["parent_strategy_version_id"]
                != expected_strategy_parent
                or data_contract.get("research_signal_binding")
                != expected_strategy_signal_binding
            ):
                raise RuntimeError("fin_strategy artifact input binding disagrees")
            artifact_sha256 = str(artifact["artifact_sha256"])
            if artifact_sha256 not in strategy_proposal_hashes:
                strategy_proposal_hashes.add(artifact_sha256)
                strategy_proposals.append(artifact)
        loop_id = _loop_id(message.tag)
        if loop_id is None:
            continue
        item = rounds.setdefault(
            loop_id,
            {
                "loop_id": loop_id,
                "tasks": [],
                "model_tasks": [],
                "codes": {},
                "model_codes": {},
                "values": {},
                "feedback": {},
            },
        )
        if "hypothesis generation" in message.tag:
            item["hypothesis"] = {
                "hypothesis": getattr(content, "hypothesis", ""),
                "reason": getattr(content, "reason", ""),
                "action": getattr(content, "action", None),
            }
        elif "experiment generation" in message.tag:
            tasks = getattr(content, "sub_tasks", content)
            if not isinstance(tasks, (list, tuple)):
                tasks = []
            item["tasks"] = [
                {
                    "name": getattr(task, "factor_name", getattr(task, "name", "unnamed_factor")),
                    "description": getattr(
                        task, "factor_description", getattr(task, "description", "")
                    ),
                    "formulation": getattr(task, "factor_formulation", None),
                    "variables": getattr(task, "variables", {}) or {},
                }
                for task in tasks
                if hasattr(task, "factor_name")
            ]
            item["model_tasks"] = [
                {
                    "name": str(getattr(task, "name", "unnamed_model")),
                    "description": str(getattr(task, "description", "")),
                    "model_type": str(getattr(task, "model_type", "")),
                    "architecture": _mapping(getattr(task, "architecture", None)),
                    "model_hyperparameters": _mapping(
                        getattr(task, "hyperparameters", None)
                    ),
                    "training_hyperparameters": _mapping(
                        getattr(task, "training_hyperparameters", None)
                    ),
                }
                for task in tasks
                if hasattr(task, "model_type") and hasattr(task, "name")
            ]
        elif message.tag.endswith("runner result") or ".runner result." in message.tag:
            snapshot = _runner_snapshot(content)
            if snapshot is not None:
                item["runner_snapshot"] = snapshot
            item["workspace_export_count"] = _export_workspace_sources(
                content, workspace_export_root, loop_id
            )
        elif "evolving code" in message.tag and "running" not in message.tag:
            workspaces = (
                content
                if isinstance(content, (list, tuple))
                else getattr(content, "sub_workspace_list", [])
            )
            for workspace in workspaces:
                target = getattr(workspace, "target_task", None)
                name = getattr(target, "factor_name", getattr(target, "name", "unnamed_factor"))
                files = getattr(workspace, "file_dict", {}) or {}
                if "factor.py" in files:
                    item["codes"][name] = files["factor.py"]
                if "model.py" in files:
                    item["model_codes"][name] = files["model.py"]
                workspace_path = getattr(workspace, "workspace_path", None)
                source_values = Path(workspace_path) / "result.h5" if workspace_path else None
                if source_values and source_values.exists():
                    item["values"][name] = str(source_values)
        elif "evolving feedback" in message.tag and "running" not in message.tag:
            decisions = []
            for feedback in content:
                decisions.append(
                    {
                        "decision": bool(getattr(feedback, "final_decision", False)),
                        "feedback": str(getattr(feedback, "final_feedback", "")),
                    }
                )
            item["implementation_feedback"] = decisions
        elif message.tag.endswith("feedback.feedback") or ".feedback.feedback." in message.tag:
            item["feedback"] = {
                "decision": bool(getattr(content, "decision", False)),
                "reason": str(getattr(content, "reason", "")),
                "hypothesis_evaluation": str(getattr(content, "hypothesis_evaluation", "")),
            }

    candidates: list[dict[str, Any]] = []
    model_candidates: list[dict[str, Any]] = []
    for loop_id, item in sorted(rounds.items()):
        implementation_feedback = item.get("implementation_feedback", [])
        for index, task in enumerate(item["tasks"]):
            name = task["name"]
            code = item["codes"].get(name)
            code_path = None
            values_path = None
            code_sha256 = None
            if code:
                safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:80]
                destination = code_root / f"loop-{loop_id:03d}-{safe_name}.py"
                destination.write_text(code, encoding="utf-8")
                code_path = str(destination)
                code_sha256 = _sha256(code)
            source_values = item["values"].get(name)
            if source_values:
                safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:80]
                values_destination = values_root / f"loop-{loop_id:03d}-{safe_name}.h5"
                shutil.copy2(source_values, values_destination)
                values_path = str(values_destination)
            implementation = (
                implementation_feedback[index] if index < len(implementation_feedback) else {}
            )
            hypothesis_feedback = item.get("feedback", {})
            candidates.append(
                {
                    **task,
                    "source_iteration": loop_id,
                    "code_path": code_path,
                    "values_path": values_path,
                    "code_sha256": code_sha256,
                    "rdagent_decision": implementation.get("decision"),
                    "rdagent_feedback": implementation.get("feedback")
                    or hypothesis_feedback.get("hypothesis_evaluation")
                    or hypothesis_feedback.get("reason"),
                    "hypothesis": item.get("hypothesis"),
                }
            )
        for index, task in enumerate(item["model_tasks"]):
            name = task["name"]
            code = item["model_codes"].get(name)
            code_path = None
            code_sha256 = None
            if code:
                safe_name = "".join(
                    character if character.isalnum() or character in "-_" else "_"
                    for character in name
                )[:80]
                destination = code_root / f"loop-{loop_id:03d}-{safe_name}-model.py"
                destination.write_text(code, encoding="utf-8")
                code_path = str(destination)
                code_sha256 = _sha256(code)
            implementation = (
                implementation_feedback[index]
                if index < len(implementation_feedback)
                else {}
            )
            hypothesis_feedback = item.get("feedback", {})
            model_candidates.append(
                {
                    **task,
                    "source_iteration": loop_id,
                    "code_path": code_path,
                    "code_sha256": code_sha256,
                    "rdagent_decision": implementation.get("decision"),
                    "rdagent_feedback": implementation.get("feedback")
                    or hypothesis_feedback.get("hypothesis_evaluation")
                    or hypothesis_feedback.get("reason"),
                    "hypothesis": item.get("hypothesis"),
                }
            )

    fin_quant_coverage = (
        _fin_quant_arm_coverage(rounds) if args.scenario == "fin_quant" else None
    )
    quant_bundles = (
        _materialize_quant_bundles(
            rounds,
            code_root=code_root,
            values_root=values_root,
            feature_set_id=args.feature_set_id,
            feature_set_sha256=args.feature_set_sha256,
        )
        if args.scenario == "fin_quant"
        else []
    )
    fin_quant_outcome = (
        _fin_quant_research_outcome(fin_quant_coverage, quant_bundles)
        if fin_quant_coverage is not None
        else None
    )
    costeer_knowledge = _costeer_knowledge_status()
    if args.scenario == "fin_strategy" and not strategy_proposals:
        raise RuntimeError("fin_strategy produced no governed strategy proposal")
    if args.scenario == "fin_strategy" and len(strategy_proposals) > int(args.loop_n):
        # One evolving loop may compile several challenger drafts before it
        # settles.  The governed competition preregisters one candidate per
        # loop, so keep only the final compiled artifact(s) in trace order.
        strategy_proposals = strategy_proposals[-int(args.loop_n) :]
    if args.scenario == "fin_strategy" and (
        costeer_knowledge["costeer_used"] is not True
        or costeer_knowledge.get("strategy_codegen_used") is not True
        or costeer_knowledge.get("strategy_codegen_target")
        != "allowlisted_rule_ir_and_contract_tests"
        or costeer_knowledge.get("strategy_compiler") != "deterministic_allowlist"
    ):
        raise RuntimeError("fin_strategy execution boundary disagrees")

    result = {
        "status": "ok",
        "scenario": args.scenario,
        "rdagent_runtime": _runtime_identity(),
        "trace_path": str(trace),
        "rounds": len(rounds),
        "trace_contract_version": "rdagent-trace-web-v1",
        "trace_loops": _trace_loop_projection(rounds),
        "trace_summary": {
            "message_count": sum(tag_counts.values()),
            "tag_counts": dict(sorted(tag_counts.items())),
        },
        "asset_ids": list(args.asset_id),
        "feature_set": (
            {
                "id": args.feature_set_id,
                "definition_sha256": args.feature_set_sha256,
            }
            if args.feature_set_id or args.feature_set_sha256
            else None
        ),
        "candidates": candidates,
        "model_candidates": model_candidates,
        "quant_bundles": quant_bundles,
        "fin_quant_coverage": fin_quant_coverage,
        "fin_quant_outcome": fin_quant_outcome,
        "single_arm": bool(fin_quant_coverage and fin_quant_coverage["single_arm"]),
        "costeer_knowledge": costeer_knowledge,
        "strategy_proposals": strategy_proposals,
        # Frozen lab scenarios are export-dead; keep the result key stable.
        "lab_outputs": [],
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--qlib-home", default="~/.qlib/qlib_data/cn_data")
    probe_parser.add_argument("--data-root", default="/data")
    probe_parser.add_argument("--llm-key-env", default="OPENAI_API_KEY")
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--trace", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--scenario", default="fin_factor")
    export_parser.add_argument("--asset-id", action="append", default=[])
    export_parser.add_argument("--feature-set-id")
    export_parser.add_argument("--feature-set-sha256")
    export_parser.add_argument("--base-features")
    args = parser.parse_args()
    result = probe(args) if args.command == "probe" else export_trace(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
