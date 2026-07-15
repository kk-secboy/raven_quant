from __future__ import annotations

import ast
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ALLOWED_IMPORT_ROOTS = {"math", "numpy", "pandas"}
FORBIDDEN_CALLS = {"breakpoint", "compile", "eval", "exec", "input", "open", "__import__"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_factor_code(source: str) -> None:
    """Reject capabilities that are unnecessary for the RD-Agent factor contract."""

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            if not roots.issubset(ALLOWED_IMPORT_ROOTS):
                raise ValueError(
                    f"factor code imports forbidden modules: {sorted(roots - ALLOWED_IMPORT_ROOTS)}"
                )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"factor code imports forbidden module: {root or '<relative>'}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise ValueError(f"factor code calls forbidden builtin: {node.func.id}")


def normalize_factor_values(frame: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=frame.name or "factor")
    if not isinstance(frame, pd.DataFrame) or frame.shape[1] != 1:
        raise ValueError("recomputed factor output must contain exactly one column")
    if not isinstance(frame.index, pd.MultiIndex) or set(frame.index.names) != {
        "datetime",
        "instrument",
    }:
        raise ValueError("recomputed factor output must use datetime/instrument MultiIndex")
    values = frame.copy()
    if values.index.names != ["datetime", "instrument"]:
        values = values.reorder_levels(["datetime", "instrument"])
    dates = pd.to_datetime(values.index.get_level_values("datetime"), errors="coerce")
    if dates.isna().any():
        raise ValueError("recomputed factor output contains invalid dates")
    values.index = pd.MultiIndex.from_arrays(
        [dates.tz_localize(None), values.index.get_level_values("instrument").astype(str)],
        names=["datetime", "instrument"],
    )
    values.iloc[:, 0] = pd.to_numeric(values.iloc[:, 0], errors="coerce")
    values = values.sort_index()
    if values.index.has_duplicates:
        raise ValueError("recomputed factor output contains duplicate index values")
    return values


def execute_factor_code(
    *,
    code_path: Path,
    input_path: Path,
    workspace: Path,
    timeout_seconds: int = 300,
    python_executable: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = code_path.read_text(encoding="utf-8")
    validate_factor_code(source)
    workspace.mkdir(parents=True, exist_ok=False)
    runtime_code = workspace / "factor.py"
    runtime_input = workspace / "daily_pv.h5"
    shutil.copy2(code_path, runtime_code)
    shutil.copy2(input_path, runtime_input)
    env = {
        "HOME": str(workspace),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }
    completed = subprocess.run(
        [python_executable or sys.executable, "-I", str(runtime_code)],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "factor execution failed").strip()
        raise ValueError(f"independent factor recomputation failed: {message[-2000:]}")
    output = workspace / "result.h5"
    if not output.is_file():
        raise ValueError("independent factor recomputation did not create result.h5")
    values = normalize_factor_values(pd.read_hdf(output))
    evidence = {
        "executor_version": "factor-recompute-v1",
        "code_sha256": sha256_file(code_path),
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output),
        "python_version": sys.version.split()[0],
        "timeout_seconds": timeout_seconds,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
    }
    return values, evidence


def compare_submitted_values(
    submitted_path: Path | None, recomputed: pd.DataFrame
) -> dict[str, Any]:
    if submitted_path is None or not submitted_path.is_file():
        return {"available": False, "exact_match": False}
    submitted = normalize_factor_values(pd.read_hdf(submitted_path))
    left, right = submitted.align(recomputed, join="outer")
    equal = bool(
        left.index.equals(right.index)
        and np.allclose(
            left.iloc[:, 0].to_numpy(dtype=float),
            right.iloc[:, 0].to_numpy(dtype=float),
            equal_nan=True,
            rtol=1e-10,
            atol=1e-12,
        )
    )
    return {
        "available": True,
        "submitted_sha256": sha256_file(submitted_path),
        "exact_match": equal,
        "submitted_rows": len(submitted),
        "recomputed_rows": len(recomputed),
    }
