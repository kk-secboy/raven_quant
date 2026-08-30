"""Immutable runner identities for allowlisted transparent-baseline repairs."""

from __future__ import annotations

import hashlib
import json
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
RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v10"
)
RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256 = (
    "0f868f2f3beaf5cff5db461f11c411ab3ef960c620c3f2f902ac385fc79eff4d"
)
RUNTIME_ALIGNMENT_RUNTIME_PATHS = (
    "scripts/run_multifactor_backtest.py",
    "scripts/run_recommendation_refresh.py",
    "src/quant_platform/portfolio_policy.py",
    "src/quant_platform/strategy_backtest.py",
    "src/quant_platform/strategy_rule_runtime.py",
)
RUNTIME_ALIGNMENT_SOURCE_RUNTIME_BUNDLE_SHA256 = (
    "b2b501cf1b59201b8732bacfa787ab38e41867993234aa2f9f605d9b163e7c68"
)
RUNTIME_ALIGNMENT_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "eea7ca8854acbed0370ead8d97fdfdb128059f41d8a5995a5050b7ec9e9f3a34"
)
RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v11"
)
RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256 = (
    "64fa634b4e774279356c5655e70c741890ed9b208ccf75df757a378c0f56a432"
)
RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "687ad83efd9d734a238bd6b52b3f5e670cec1e5d165ec2b25cabcef724feb7cf"
)
TRANSPARENT_BASELINE_RUNNER_FIELD = "target_runner_sha256"
TRANSPARENT_BASELINE_JOB_RUNNER_FIELD = "transparent_baseline_runner_sha256"
TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD = "target_runtime_bundle_sha256"
TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD = (
    "transparent_baseline_runtime_bundle_sha256"
)
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
    RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION: RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256,
    RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION: RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_alignment_bundle_sha256(project_root: Path) -> str:
    """Hash every executable module governed by the v10+ runtime receipts."""

    inventory = []
    for relative in RUNTIME_ALIGNMENT_RUNTIME_PATHS:
        path = project_root / relative
        payload = path.read_bytes()
        inventory.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    encoded = json.dumps(
        inventory,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def target_runner_for_recipe(recipe_id: Any, recipe_version: Any) -> str | None:
    if str(recipe_id or "") not in _TRANSPARENT_RECIPE_IDS:
        return None
    return _TARGET_RUNNERS.get(str(recipe_version or ""))


def target_runtime_bundle_for_recipe(recipe_id: Any, recipe_version: Any) -> str | None:
    if str(recipe_id or "") not in _TRANSPARENT_RECIPE_IDS:
        return None
    if str(recipe_version or "") == RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION:
        return RUNTIME_ALIGNMENT_TARGET_RUNTIME_BUNDLE_SHA256
    if str(recipe_version or "") == RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION:
        return RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256
    return None


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
    expected_bundle = target_runtime_bundle_for_recipe(
        config.get("recipe_id"), config.get("recipe_version")
    )
    bootstrap_value = bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
    payload_value = job_payload.get(TRANSPARENT_BASELINE_JOB_RUNNER_FIELD)
    bootstrap_bundle = bootstrap.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD)
    payload_bundle = job_payload.get(TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD)
    if expected is None:
        if any(
            value is not None
            for value in (
                bootstrap_value,
                payload_value,
                bootstrap_bundle,
                payload_bundle,
            )
        ):
            raise ValueError(
                "runner repair identity is forbidden outside governed transparent recipes"
            )
        return None
    version_label = {
        OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION: "v8",
        CANONICAL_LF_TARGET_RECIPE_VERSION: "v9",
        RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION: "v10",
        RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION: "v11",
    }[str(config.get("recipe_version") or "")]
    if bootstrap_value != expected or payload_value != expected:
        raise ValueError(
            f"transparent {version_label} runner identity differs from its sealed bootstrap"
        )
    if (
        bootstrap_bundle != expected_bundle
        or payload_bundle != expected_bundle
    ):
        raise ValueError(
            f"transparent {version_label} runtime bundle identity differs from its "
            "sealed bootstrap"
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
    if version_label in {"v10", "v11"}:
        try:
            bundle_sha256 = runtime_alignment_bundle_sha256(runner_path.parents[1])
        except OSError as exc:
            raise ValueError(
                f"transparent {version_label} runtime bundle cannot be verified"
            ) from exc
        expected_bundle_sha256 = {
            "v10": RUNTIME_ALIGNMENT_TARGET_RUNTIME_BUNDLE_SHA256,
            "v11": RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256,
        }[version_label]
        if bundle_sha256 != expected_bundle_sha256:
            raise ValueError(
                f"transparent {version_label} runtime bundle differs from the repair "
                "authorization"
            )
    return observed
