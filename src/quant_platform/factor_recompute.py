from __future__ import annotations

import ast
import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ALLOWED_IMPORT_ROOTS = {"math", "numpy", "pandas"}
FORBIDDEN_CALLS = {"breakpoint", "compile", "eval", "exec", "input", "open", "__import__"}
FORBIDDEN_FUTURE_METHODS = {"backfill", "bfill"}
PERIOD_METHODS = {"diff", "pct_change", "shift"}
FACTOR_RECOMPUTE_EXECUTOR_VERSION = "factor-recompute-v4-pit-prefix-invariance"
FACTOR_PIT_CONTRACT_VERSION = "factor-pit-prefix-invariance-v1"
FACTOR_PIT_RTOL = 1e-10
FACTOR_PIT_ATOL = 1e-12
FACTOR_MIN_DAILY_FINITE = 50
FACTOR_MIN_COVERAGE_RATIO = 0.80
FACTOR_MIN_GOOD_DAY_RATE = 0.95


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_factor_code(source: str) -> None:
    """Reject unsafe capabilities and obvious forward-looking factor operations."""

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
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if method in FORBIDDEN_FUTURE_METHODS:
            raise ValueError(f"factor code calls forward-looking method: {method}")
        if method in PERIOD_METHODS:
            periods = _call_argument(node, position=0, keyword="periods")
            if (literal := _numeric_literal(periods)) is not None and literal < 0:
                raise ValueError(f"factor code calls {method} with negative periods")
        if method == "roll":
            shift = _call_argument(node, position=1, keyword="shift")
            if (literal := _numeric_literal(shift)) is not None and literal < 0:
                raise ValueError("factor code calls roll with a negative shift")
        if method == "rolling":
            center = _call_argument(node, position=None, keyword="center")
            if _boolean_literal(center) is True:
                raise ValueError("factor code calls forward-looking rolling with center=True")
        if method == "fillna":
            fill_method = _call_argument(node, position=None, keyword="method")
            if _string_literal(fill_method) in FORBIDDEN_FUTURE_METHODS:
                raise ValueError("factor code calls fillna with a backward-fill method")
        if method == "interpolate":
            direction = _call_argument(node, position=None, keyword="limit_direction")
            if _string_literal(direction) in {"backward", "both"}:
                raise ValueError("factor code calls interpolate with future observations")


def _call_argument(
    node: ast.Call, *, position: int | None, keyword: str
) -> ast.expr | None:
    if position is not None and len(node.args) > position:
        return node.args[position]
    return next((item.value for item in node.keywords if item.arg == keyword), None)


def _numeric_literal(node: ast.expr | None) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        value = float(node.operand.value)
        return -value if isinstance(node.op, ast.USub) else value
    return None


def _boolean_literal(node: ast.expr | None) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _string_literal(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower()
    return None


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


def normalize_factor_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the immutable daily input without changing its row set."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("factor recomputation input must be a non-empty DataFrame")
    if not isinstance(frame.index, pd.MultiIndex) or set(frame.index.names) != {
        "datetime",
        "instrument",
    }:
        raise ValueError("factor recomputation input must use datetime/instrument MultiIndex")
    values = frame.copy()
    if values.index.names != ["datetime", "instrument"]:
        values = values.reorder_levels(["datetime", "instrument"])
    dates = pd.to_datetime(values.index.get_level_values("datetime"), errors="coerce")
    if dates.isna().any():
        raise ValueError("factor recomputation input contains invalid dates")
    values.index = pd.MultiIndex.from_arrays(
        [dates.tz_localize(None), values.index.get_level_values("instrument").astype(str)],
        names=["datetime", "instrument"],
    )
    values = values.sort_index()
    if values.index.has_duplicates:
        raise ValueError("factor recomputation input contains duplicate index values")
    return values


def require_exact_factor_index(
    factor_values: pd.DataFrame | pd.Series,
    factor_input: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    """Fail closed instead of letting a later inner join shrink the backtest sample."""

    normalized_values = normalize_factor_values(factor_values)
    normalized_input = normalize_factor_input(factor_input)
    if not normalized_values.index.equals(normalized_input.index):
        missing = normalized_input.index.difference(normalized_values.index)
        unexpected = normalized_values.index.difference(normalized_input.index)
        raise ValueError(
            f"{context} output index does not exactly match its immutable input "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    return normalized_values


def require_exact_oos_coverage(
    factor_values: pd.DataFrame | pd.Series,
    factor_input: pd.DataFrame,
    *,
    test_start: str,
    test_end: str,
    trading_days: Sequence[Any],
    context: str,
    min_daily_finite: int = FACTOR_MIN_DAILY_FINITE,
    min_coverage_ratio: float = FACTOR_MIN_COVERAGE_RATIO,
    min_good_day_rate: float = FACTOR_MIN_GOOD_DAY_RATE,
) -> dict[str, Any]:
    """Prove that a factor has every authoritative input row and OOS trading day."""

    values = require_exact_factor_index(factor_values, factor_input, context=context)
    start = pd.Timestamp(test_start).tz_localize(None)
    end = pd.Timestamp(test_end).tz_localize(None)
    if start > end:
        raise ValueError("final OOS start must not be after its end")
    if min_daily_finite < 5:
        raise ValueError("final OOS minimum daily finite count must be at least 5")
    if not 0 < min_coverage_ratio <= 1 or not 0 < min_good_day_rate <= 1:
        raise ValueError("final OOS coverage thresholds must be in (0, 1]")
    expected_days = pd.DatetimeIndex(pd.to_datetime(list(trading_days), errors="coerce"))
    if expected_days.isna().any():
        raise ValueError("authoritative final OOS calendar contains invalid dates")
    expected_days = expected_days.tz_localize(None).normalize().unique().sort_values()
    if (
        expected_days.empty
        or expected_days[0] != start.normalize()
        or expected_days[-1] != end.normalize()
    ):
        raise ValueError("authoritative final OOS calendar does not match frozen boundaries")
    value_dates = pd.DatetimeIndex(values.index.get_level_values("datetime"))
    oos_mask = (value_dates >= start) & (value_dates <= end)
    oos_values = values.loc[oos_mask]
    actual_days = (
        pd.DatetimeIndex(oos_values.index.get_level_values("datetime"))
        .normalize()
        .unique()
        .sort_values()
    )
    if not actual_days.equals(expected_days):
        missing_days = expected_days.difference(actual_days)
        unexpected_days = actual_days.difference(expected_days)
        raise ValueError(
            f"{context} does not exactly cover final OOS trading days "
            f"(missing={len(missing_days)}, unexpected={len(unexpected_days)})"
        )
    value_dates = pd.DatetimeIndex(oos_values.index.get_level_values("datetime")).normalize()
    finite_mask = np.isfinite(oos_values.iloc[:, 0].to_numpy(dtype=float))
    total_counts = pd.Series(1, index=value_dates).groupby(level=0).sum().reindex(expected_days)
    finite_counts = (
        pd.Series(finite_mask.astype(int), index=value_dates)
        .groupby(level=0)
        .sum()
        .reindex(expected_days, fill_value=0)
    )
    coverage_ratios = finite_counts.div(total_counts)
    good_days = (finite_counts >= min_daily_finite) & (
        coverage_ratios >= min_coverage_ratio
    )
    good_day_rate = float(good_days.mean()) if len(good_days) else 0.0
    if good_day_rate < min_good_day_rate:
        failed_days = expected_days[~good_days.to_numpy()]
        raise ValueError(
            f"{context} final OOS cross-sectional coverage failed on "
            f"{len(failed_days)} of {len(expected_days)} trading days "
            f"(good_day_rate={good_day_rate:.6f}, required={min_good_day_rate:.6f})"
        )
    return {
        "contract_version": "factor-oos-index-exact-v1",
        "test_start": start.date().isoformat(),
        "test_end": end.date().isoformat(),
        "trading_day_count": len(expected_days),
        "row_count": len(oos_values),
        "finite_row_count": int(finite_mask.sum()),
        "min_daily_finite_required": int(min_daily_finite),
        "min_coverage_ratio_required": float(min_coverage_ratio),
        "min_good_day_rate_required": float(min_good_day_rate),
        "minimum_daily_finite_observed": int(finite_counts.min()),
        "minimum_coverage_ratio_observed": float(coverage_ratios.min()),
        "mean_coverage_ratio_observed": float(coverage_ratios.mean()),
        "good_day_rate": good_day_rate,
        "coverage_gate_passed": True,
        "index_exact_match": True,
    }


def _prefix_cutoffs(dates: pd.DatetimeIndex, count: int) -> list[pd.Timestamp]:
    unique_dates = dates.unique().sort_values()
    if count < 2:
        raise ValueError("factor PIT validation requires at least two prefix cutpoints")
    if len(unique_dates) <= count:
        raise ValueError(
            f"factor PIT validation requires more than {count} distinct trading days"
        )
    # Always include the penultimate available date.  Even when a custom
    # implementation leaks only in the newest tail, appending the last day
    # must not be allowed to rewrite that prefix.  The remaining cutoffs are
    # spread through the earlier history to catch global/full-sample transforms.
    positions = [
        ((index + 1) * len(unique_dates)) // count - 1
        for index in range(count - 1)
    ]
    positions.append(len(unique_dates) - 2)
    return [pd.Timestamp(unique_dates[max(0, position)]) for position in positions]


def validate_factor_prefix_invariance(
    *,
    code_path: Path,
    input_path: Path,
    full_values: pd.DataFrame | pd.Series,
    workspace_root: Path,
    timeout_seconds: int = 300,
    python_executable: str | None = None,
    cutpoint_count: int = 3,
) -> dict[str, Any]:
    """Dynamically prove that appending future rows cannot rewrite past values."""

    factor_input = normalize_factor_input(pd.read_hdf(input_path))
    authoritative = require_exact_factor_index(
        full_values, factor_input, context="full factor recomputation"
    )
    dates = pd.DatetimeIndex(factor_input.index.get_level_values("datetime"))
    cutoffs = _prefix_cutoffs(dates, cutpoint_count)
    workspace_root.mkdir(parents=True, exist_ok=False)
    checks: list[dict[str, Any]] = []
    for index, cutoff in enumerate(cutoffs, start=1):
        prefix_mask = dates <= cutoff
        prefix_input = factor_input.loc[prefix_mask]
        prefix_input_path = workspace_root / f"prefix-{index:02d}.h5"
        prefix_input.to_hdf(prefix_input_path, key="data", mode="w")
        prefix_values, prefix_evidence = execute_factor_code(
            code_path=code_path,
            input_path=prefix_input_path,
            workspace=workspace_root / f"run-{index:02d}",
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
        prefix_values = require_exact_factor_index(
            prefix_values,
            prefix_input,
            context=f"factor PIT prefix {cutoff.date().isoformat()}",
        )
        expected = authoritative.reindex(prefix_values.index)
        invariant = np.allclose(
            expected.iloc[:, 0].to_numpy(dtype=float),
            prefix_values.iloc[:, 0].to_numpy(dtype=float),
            equal_nan=True,
            rtol=FACTOR_PIT_RTOL,
            atol=FACTOR_PIT_ATOL,
        )
        if not invariant:
            raise ValueError(
                "factor violates point-in-time prefix invariance at "
                f"{cutoff.date().isoformat()}: future rows changed historical values"
            )
        checks.append(
            {
                "cutoff": cutoff.date().isoformat(),
                "input_rows": len(prefix_input),
                "output_rows": len(prefix_values),
                "input_sha256": prefix_evidence["input_sha256"],
                "output_sha256": prefix_evidence["output_sha256"],
                "invariant": True,
            }
        )
    return {
        "contract_version": FACTOR_PIT_CONTRACT_VERSION,
        "status": "passed",
        "cutpoint_count": len(checks),
        "rtol": FACTOR_PIT_RTOL,
        "atol": FACTOR_PIT_ATOL,
        "checks": checks,
    }


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
    sandbox_image = str(os.environ.get("FACTOR_SANDBOX_IMAGE") or "").strip()
    if sandbox_image:
        completed, sandbox_evidence = _run_container_sandbox(
            workspace=workspace,
            runtime_code=runtime_code,
            image=sandbox_image,
            timeout_seconds=timeout_seconds,
        )
    elif os.environ.get("FACTOR_RECOMPUTE_ALLOW_LOCAL_UNSAFE") == "1":
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
        sandbox_evidence = {"sandbox_mode": "local-test-override"}
    else:
        raise ValueError(
            "factor recomputation requires the isolated container sandbox"
        )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "factor execution failed").strip()
        raise ValueError(f"independent factor recomputation failed: {message[-2000:]}")
    output = workspace / "result.h5"
    if not output.is_file():
        raise ValueError("independent factor recomputation did not create result.h5")
    values = normalize_factor_values(pd.read_hdf(output))
    evidence = {
        "executor_version": FACTOR_RECOMPUTE_EXECUTOR_VERSION,
        "code_sha256": sha256_file(code_path),
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output),
        "python_version": sys.version.split()[0],
        "timeout_seconds": timeout_seconds,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
        **sandbox_evidence,
    }
    return values, evidence


def _run_container_sandbox(
    *,
    workspace: Path,
    runtime_code: Path,
    image: str,
    timeout_seconds: int,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    if not shutil.which("docker"):
        raise ValueError("factor container sandbox requires the Docker CLI")
    if not runtime_code.is_relative_to(workspace):
        raise ValueError("factor runtime code is outside its isolated workspace")
    workspace.chmod(0o777)
    runtime_code.chmod(0o444)
    (workspace / "daily_pv.h5").chmod(0o444)
    image_result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if image_result.returncode != 0:
        raise ValueError("factor sandbox image is unavailable")
    image_id = image_result.stdout.strip()
    if not image_id.startswith("sha256:") or len(image_id) != 71:
        raise ValueError("factor sandbox image identity is invalid")
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
        "128",
        "--memory",
        "2g",
        "--cpus",
        "1",
        "--user",
        "65534:65534",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=256m",
        "--mount",
        f"type=bind,src={workspace.resolve()},dst=/work",
        "--workdir",
        "/work",
        image,
        "python",
        "-I",
        "factor.py",
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
    return completed, {
        "sandbox_mode": "docker-isolated",
        "sandbox_image": image,
        "sandbox_image_id": image_id,
        "network_mode": "none",
        "root_filesystem_read_only": True,
        "capabilities_dropped": "ALL",
        "no_new_privileges": True,
        "pids_limit": 128,
        "memory_limit_bytes": 2 * 1024**3,
        "cpu_limit": 1,
    }


def compare_submitted_values(
    submitted_path: Path | None, recomputed: pd.DataFrame
) -> dict[str, Any]:
    if submitted_path is None or not submitted_path.is_file():
        return {"available": False, "exact_match": False}
    submitted = normalize_factor_values(pd.read_hdf(submitted_path))
    same_index = submitted.index.equals(recomputed.index)
    equal = bool(
        same_index
        and np.allclose(
            submitted.iloc[:, 0].to_numpy(dtype=float),
            recomputed.iloc[:, 0].to_numpy(dtype=float),
            equal_nan=True,
            rtol=1e-10,
            atol=1e-12,
        )
    )
    return {
        "available": True,
        "submitted_sha256": sha256_file(submitted_path),
        "exact_match": equal,
        "index_exact_match": same_index,
        "submitted_rows": len(submitted),
        "recomputed_rows": len(recomputed),
    }
