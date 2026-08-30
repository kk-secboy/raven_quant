"""Immutable runner identities for allowlisted transparent-baseline repairs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v8"
)
OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256 = (
    "fa1090deaa66ca77a045c1a872f7b6451043e116e717954533217908135c584e"
)
CANONICAL_LF_TARGET_RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-08-30-v9"
CANONICAL_LF_TARGET_RUNNER_SHA256 = (
    "256bbfd579865e7bc1442f1f64003d655241abecb320e5d27b440dd635224d57"
)
TRANSPARENT_BASELINE_RUNNER_FIELD = "target_runner_sha256"
TRANSPARENT_BASELINE_JOB_RUNNER_FIELD = "transparent_baseline_runner_sha256"
_TRANSPARENT_RECIPE_IDS = {
    "short_relative_strength",
    "swing_trend",
    "long_quality_value",
}
_TARGET_RUNNERS = {
    OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION: (
        OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
    ),
    CANONICAL_LF_TARGET_RECIPE_VERSION: CANONICAL_LF_TARGET_RUNNER_SHA256,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_runner_for_recipe(recipe_id: Any, recipe_version: Any) -> str | None:
    if str(recipe_id or "") not in _TRANSPARENT_RECIPE_IDS:
        return None
    return _TARGET_RUNNERS.get(str(recipe_version or ""))


def require_transparent_baseline_runner(
    *,
    config: Mapping[str, Any],
    job_payload: Mapping[str, Any],
    runner_path: Path,
) -> str | None:
    """Verify bootstrap, job and executed runner have one governed byte identity."""

    bootstrap_raw = config.get("transparent_baseline_bootstrap")
    bootstrap = dict(bootstrap_raw) if isinstance(bootstrap_raw, Mapping) else {}
    expected = target_runner_for_recipe(
        config.get("recipe_id"), config.get("recipe_version")
    )
    bootstrap_value = bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
    payload_value = job_payload.get(TRANSPARENT_BASELINE_JOB_RUNNER_FIELD)
    if expected is None:
        if bootstrap_value is not None or payload_value is not None:
            raise ValueError("runner repair identity is forbidden outside transparent v8/v9")
        return None
    version_label = (
        "v8"
        if config.get("recipe_version")
        == OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION
        else "v9"
    )
    if bootstrap_value != expected or payload_value != expected:
        raise ValueError(
            f"transparent {version_label} runner identity differs from its sealed bootstrap"
        )
    try:
        observed = _file_sha256(runner_path)
    except OSError as exc:
        raise ValueError(
            f"transparent {version_label} runner cannot be verified"
        ) from exc
    if observed != expected:
        raise ValueError(
            f"transparent {version_label} runner bytes differ from the repair authorization"
        )
    return observed
