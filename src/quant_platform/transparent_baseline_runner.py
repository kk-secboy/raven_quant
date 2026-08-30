"""Immutable runner identities for allowlisted transparent-baseline repairs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_platform.runtime_source_closure import position_risk_source_closure_sha256

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
POSITION_RISK_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v12"
)
POSITION_RISK_TARGET_RUNNER_SHA256 = (
    "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
)
POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "6366bd2b77c1c60ea4069afde43d5335c1e362c18acf2955f13902e7ba9ccbc6"
)
FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v13"
)
FAIL_CLOSED_EXECUTION_TARGET_RUNNER_SHA256 = (
    "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
)
FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "3000d84d183fe589da13d01902f88b1da1d402f47e18e4aafe91831cc59bdd8f"
)
TRANSPARENT_BASELINE_RUNNER_FIELD = "target_runner_sha256"
TRANSPARENT_BASELINE_JOB_RUNNER_FIELD = "transparent_baseline_runner_sha256"
TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD = "target_runtime_bundle_sha256"
TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD = (
    "transparent_baseline_runtime_bundle_sha256"
)
TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD = (
    "target_worker_runtime_image_digest"
)
TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD = (
    "transparent_baseline_worker_runtime_image_digest"
)
TRANSPARENT_BASELINE_RESULT_WORKER_RUNTIME_IMAGE_FIELD = (
    "worker_runtime_image_digest"
)
WORKER_RUNTIME_IMAGE_DIGEST_ENV = "QUANTLAB_WORKER_RUNTIME_IMAGE_DIGEST"
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
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
    POSITION_RISK_TARGET_RECIPE_VERSION: POSITION_RISK_TARGET_RUNNER_SHA256,
    FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION: (
        FAIL_CLOSED_EXECUTION_TARGET_RUNNER_SHA256
    ),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_bundle_sha256(project_root: Path, paths: tuple[str, ...]) -> str:
    inventory = []
    for relative in paths:
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


def runtime_alignment_bundle_sha256(project_root: Path) -> str:
    """Hash the immutable five-module v10/v11 runtime inventory."""

    return _runtime_bundle_sha256(project_root, RUNTIME_ALIGNMENT_RUNTIME_PATHS)


def position_risk_bundle_sha256(project_root: Path) -> str:
    """Hash the fail-closed local Python closure for the current material runtime."""

    return position_risk_source_closure_sha256(project_root)


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
    if str(recipe_version or "") == POSITION_RISK_TARGET_RECIPE_VERSION:
        return POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256
    if str(recipe_version or "") == FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION:
        return FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256
    return None


def target_worker_runtime_image_for_recipe(
    recipe_id: Any,
    recipe_version: Any,
) -> str | None:
    """Return the exact release-built worker image required by material v12+.

    The digest is deliberately release data rather than a source constant: the
    release controller resolves Docker's immutable content ID after building
    the worker and stamps it into every process before any sealed version exists.
    """

    if (
        str(recipe_id or "") not in _TRANSPARENT_RECIPE_IDS
        or str(recipe_version or "")
        not in {
            POSITION_RISK_TARGET_RECIPE_VERSION,
            FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION,
        }
    ):
        return None
    version_label = {
        POSITION_RISK_TARGET_RECIPE_VERSION: "v12",
        FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION: "v13",
    }[str(recipe_version or "")]
    value = str(os.getenv(WORKER_RUNTIME_IMAGE_DIGEST_ENV) or "").strip().lower()
    if not _IMAGE_DIGEST.fullmatch(value):
        raise ValueError(
            f"transparent {version_label} worker runtime image digest is missing or invalid"
        )
    return value


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
    bootstrap_worker_image = bootstrap.get(
        TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD
    )
    payload_worker_image = job_payload.get(
        TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD
    )
    if expected is None:
        if any(
            value is not None
            for value in (
                bootstrap_value,
                payload_value,
                bootstrap_bundle,
                payload_bundle,
                bootstrap_worker_image,
                payload_worker_image,
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
        POSITION_RISK_TARGET_RECIPE_VERSION: "v12",
        FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION: "v13",
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
    expected_worker_image = target_worker_runtime_image_for_recipe(
        config.get("recipe_id"), config.get("recipe_version")
    )
    if expected_worker_image is not None and (
        bootstrap_worker_image != expected_worker_image
        or payload_worker_image != expected_worker_image
    ):
        raise ValueError(
            f"transparent {version_label} worker runtime image differs from its "
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
    if version_label in {"v10", "v11", "v12", "v13"}:
        try:
            bundle_sha256 = (
                position_risk_bundle_sha256(runner_path.parents[1])
                if version_label in {"v12", "v13"}
                else runtime_alignment_bundle_sha256(runner_path.parents[1])
            )
        except OSError as exc:
            raise ValueError(
                f"transparent {version_label} runtime bundle cannot be verified"
            ) from exc
        expected_bundle_sha256 = {
            "v10": RUNTIME_ALIGNMENT_TARGET_RUNTIME_BUNDLE_SHA256,
            "v11": RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256,
            "v12": POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256,
            "v13": FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256,
        }[version_label]
        if bundle_sha256 != expected_bundle_sha256:
            raise ValueError(
                f"transparent {version_label} runtime bundle differs from the repair "
                "authorization"
            )
    return observed
