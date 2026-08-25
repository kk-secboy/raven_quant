from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .model_research_governance import canonical_sha256, file_sha256, is_sha256

MODEL_RECOMPUTE_EXECUTOR_VERSION = "model-recompute-docker-v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ALLOWED_IMPORT_ROOTS = {"math", "numpy", "torch", "typing"}
_FORBIDDEN_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
    "__import__",
}
_FORBIDDEN_ATTRIBUTES = {
    "__builtins__",
    "__class__",
    "__dict__",
    "__getattribute__",
    "__globals__",
    "__import__",
    "__subclasses__",
    "popen",
    "run",
    "system",
}


def validate_model_code(source: str) -> None:
    """Restrict generated code to a pure PyTorch model definition.

    The Docker boundary protects the host.  This validator separately protects
    the independent evaluator from candidate code that tries to replace the
    runner, read files, execute commands, or monkeypatch its own score.
    """

    tree = ast.parse(source)
    defined_classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    exported_model: str | None = None
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if not isinstance(
            statement,
            (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.Assign, ast.AnnAssign),
        ):
            raise ValueError("model code may only define a pure model module")
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id == "model_cls":
                    if not isinstance(statement.value, ast.Name):
                        raise ValueError("model_cls must reference a class defined in model.py")
                    exported_model = statement.value.id
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "model_cls"
        ):
            if not isinstance(statement.value, ast.Name):
                raise ValueError("model_cls must reference a class defined in model.py")
            exported_model = statement.value.id
    if exported_model is None or exported_model not in defined_classes:
        raise ValueError("model code must export model_cls as a locally defined class")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            if not roots.issubset(_ALLOWED_IMPORT_ROOTS):
                raise ValueError(
                    f"model code imports forbidden modules: {sorted(roots - _ALLOWED_IMPORT_ROOTS)}"
                )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"model code imports forbidden module: {root or '<relative>'}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                raise ValueError(f"model code calls forbidden builtin: {node.func.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRIBUTES:
            raise ValueError(f"model code accesses forbidden capability: {node.attr}")


def execute_model_candidate(
    *,
    code_path: Path,
    provider_path: Path,
    manifest: dict[str, Any],
    workspace: Path,
    runner_path: Path,
    additional_factors_path: Path | None = None,
    allow_final_oos: bool = False,
    allow_inference: bool = False,
    timeout_seconds: int = 7200,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_id = str(manifest.get("candidate_id") or "")
    if not _SAFE_ID.fullmatch(candidate_id):
        raise ValueError("model candidate id is invalid")
    if not code_path.is_file() or not provider_path.is_dir() or not runner_path.is_file():
        raise ValueError("model recomputation inputs are unavailable")
    validate_model_code(code_path.read_text(encoding="utf-8"))
    expected_code_sha256 = str(manifest.get("code_sha256") or "").lower()
    if not is_sha256(expected_code_sha256) or file_sha256(code_path) != expected_code_sha256:
        raise ValueError("model candidate code hash is invalid")
    if allow_final_oos and allow_inference:
        raise ValueError("model execution cannot open final OOS during live inference")
    if allow_final_oos:
        if manifest.get("final_oos_opened") is not True:
            raise ValueError("formal model recomputation requires an opened final OOS ledger")
        if manifest.get("inference_only") is True:
            raise ValueError("formal model recomputation cannot be marked inference-only")
    else:
        if manifest.get("final_oos_opened") is not False:
            raise ValueError("research or inference model execution cannot open final OOS")
        if (manifest.get("inference_only") is True) is not allow_inference:
            raise ValueError("model inference authorization does not match the manifest")
    image = str(os.environ.get("MODEL_SANDBOX_IMAGE") or "").strip()
    if not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError(
            "model recomputation requires a digest-pinned MODEL_SANDBOX_IMAGE"
        )
    if not shutil.which("docker"):
        raise ValueError("model recomputation requires the Docker CLI")
    image_result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    image_id = image_result.stdout.strip()
    if image_result.returncode != 0 or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", image_id
    ):
        raise ValueError("model sandbox image identity is invalid")
    runner_sha256 = file_sha256(runner_path)
    executor_source_sha256 = file_sha256(Path(__file__).resolve())
    execution_environment = {
        "contract_version": "model-execution-environment-v1",
        "executor_version": MODEL_RECOMPUTE_EXECUTOR_VERSION,
        "executor_source_sha256": executor_source_sha256,
        "runner_sha256": runner_sha256,
        "sandbox_image": image,
        "sandbox_image_id": image_id,
    }
    execution_environment_sha256 = canonical_sha256(execution_environment)
    workspace.mkdir(parents=True, exist_ok=False)
    shutil.copy2(code_path, workspace / "model.py")
    shutil.copy2(runner_path, workspace / "runner.py")
    if additional_factors_path is not None:
        if not additional_factors_path.is_file():
            raise ValueError("additional factor values are unavailable")
        runtime_factors = workspace / "additional_factors.parquet"
        shutil.copy2(additional_factors_path, runtime_factors)
        runtime_manifest_additions = {
            "additional_factors_path": "/work/additional_factors.parquet",
            "additional_factors_sha256": file_sha256(runtime_factors),
        }
    else:
        runtime_manifest_additions = {}
    runtime_manifest = {
        **manifest,
        **runtime_manifest_additions,
        "contract_version": "model-sandbox-input-v1",
        "provider_uri": "/qlib",
        "execution_environment": execution_environment,
        "execution_environment_sha256": execution_environment_sha256,
    }
    (workspace / "manifest.json").write_text(
        json.dumps(runtime_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    workspace.chmod(0o777)
    for name in ("model.py", "runner.py", "manifest.json"):
        (workspace / name).chmod(0o444)
    cidfile = workspace / "container.cid"
    command = [
        "docker",
        "run",
        "--rm",
        "--cidfile",
        str(cidfile),
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "512",
        "--memory",
        "8g",
        "--cpus",
        "4",
        "--user",
        "65534:65534",
        "--env",
        "HOME=/tmp",
        "--env",
        "PYTHONPATH=/work",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=2g",
        "--mount",
        f"type=bind,src={workspace.resolve()},dst=/work",
        "--mount",
        f"type=bind,src={provider_path.resolve()},dst=/qlib,readonly",
        "--workdir",
        "/work",
        image,
        "python",
        "-I",
        "runner.py",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        if cidfile.is_file():
            container_id = cidfile.read_text(encoding="utf-8").strip()
            if container_id:
                subprocess.run(
                    ["docker", "rm", "-f", container_id],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
        raise
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "model execution failed").strip()
        raise ValueError(f"independent model recomputation failed: {message[-4000:]}")
    result_path = workspace / "output" / "result.json"
    if not result_path.is_file():
        raise ValueError("independent model recomputation did not create result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "passed" or result.get("candidate_id") != candidate_id:
        raise ValueError("independent model recomputation result is invalid")
    if result.get("final_oos_opened") is not bool(allow_final_oos):
        raise ValueError("independent model recomputation reported an invalid final OOS state")
    if (result.get("inference_only") is True) is not allow_inference:
        raise ValueError("independent model recomputation reported an invalid inference state")
    predictions_path = workspace / "output" / "predictions.parquet"
    checkpoint_path = workspace / "output" / "checkpoint.pt"
    if (
        not predictions_path.is_file()
        or result.get("predictions_sha256") != file_sha256(predictions_path)
        or not checkpoint_path.is_file()
        or result.get("checkpoint_sha256") != file_sha256(checkpoint_path)
    ):
        raise ValueError("independent model output files failed immutable verification")
    portfolio_report_path = workspace / "output" / "portfolio_report.parquet"
    if not allow_inference and (
        not portfolio_report_path.is_file()
        or result.get("portfolio_report_sha256") != file_sha256(portfolio_report_path)
    ):
        raise ValueError("independent model portfolio report failed immutable verification")
    evidence = {
        "executor_version": MODEL_RECOMPUTE_EXECUTOR_VERSION,
        "candidate_id": candidate_id,
        "code_sha256": expected_code_sha256,
        "input_manifest_sha256": canonical_sha256(runtime_manifest),
        "result_sha256": file_sha256(result_path),
        "predictions_sha256": file_sha256(predictions_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "portfolio_report_sha256": (
            file_sha256(portfolio_report_path) if not allow_inference else None
        ),
        "sandbox_mode": "docker-isolated",
        "sandbox_image": image,
        "sandbox_image_id": image_id,
        "executor_source_sha256": executor_source_sha256,
        "runner_sha256": runner_sha256,
        "execution_environment": execution_environment,
        "execution_environment_sha256": execution_environment_sha256,
        "network_mode": "none",
        "root_filesystem_read_only": True,
        "capabilities_dropped": "ALL",
        "no_new_privileges": True,
        "timeout_seconds": timeout_seconds,
        "final_oos_opened": bool(allow_final_oos),
        "inference_only": bool(allow_inference),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    result["execution_evidence_sha256"] = evidence["evidence_sha256"]
    result["execution_environment_sha256"] = execution_environment_sha256
    return result, evidence
