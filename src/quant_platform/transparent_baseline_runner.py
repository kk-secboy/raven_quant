"""Immutable runner identity for the one allowlisted v7-to-v8 repair."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v8"
)
OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256 = (
    "b230bb66ab1aad446f575c59d2e99564cf82dd734d7fa40f5ce574c41222c686"
)
TRANSPARENT_BASELINE_RUNNER_FIELD = "target_runner_sha256"
TRANSPARENT_BASELINE_JOB_RUNNER_FIELD = "transparent_baseline_runner_sha256"
_TRANSPARENT_RECIPE_IDS = {
    "short_relative_strength",
    "swing_trend",
    "long_quality_value",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_runner_for_recipe(recipe_id: Any, recipe_version: Any) -> str | None:
    if (
        str(recipe_id or "") in _TRANSPARENT_RECIPE_IDS
        and str(recipe_version or "") == OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION
    ):
        return OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
    return None


def require_transparent_baseline_runner(
    *,
    config: Mapping[str, Any],
    job_payload: Mapping[str, Any],
    runner_path: Path,
) -> str | None:
    """Verify the v8 bootstrap, job and executed runner have one byte identity."""

    bootstrap_raw = config.get("transparent_baseline_bootstrap")
    bootstrap = dict(bootstrap_raw) if isinstance(bootstrap_raw, Mapping) else {}
    expected = target_runner_for_recipe(
        config.get("recipe_id"), config.get("recipe_version")
    )
    bootstrap_value = bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
    payload_value = job_payload.get(TRANSPARENT_BASELINE_JOB_RUNNER_FIELD)
    if expected is None:
        if bootstrap_value is not None or payload_value is not None:
            raise ValueError("runner repair identity is forbidden outside transparent v8")
        return None
    if bootstrap_value != expected or payload_value != expected:
        raise ValueError("transparent v8 runner identity differs from its sealed bootstrap")
    try:
        observed = _file_sha256(runner_path)
    except OSError as exc:
        raise ValueError("transparent v8 runner cannot be verified") from exc
    if observed != expected:
        raise ValueError("transparent v8 runner bytes differ from the repair authorization")
    return observed
