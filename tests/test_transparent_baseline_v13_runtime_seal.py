from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from quant_data.database import strategy_versions
from quant_platform.strategy_recipes import RECIPE_VERSION
from quant_platform.strategy_store import _transparent_worker_runtime_failures
from quant_platform.transparent_baseline_runner import (
    FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION,
    FAIL_CLOSED_EXECUTION_TARGET_RUNNER_SHA256,
    FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256,
    POSITION_RISK_TARGET_RECIPE_VERSION,
    POSITION_RISK_TARGET_RUNNER_SHA256,
    POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256,
    TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_RESULT_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
    position_risk_bundle_sha256,
    target_runner_for_recipe,
    target_runtime_bundle_for_recipe,
)

pytestmark = pytest.mark.no_database


def _migration_module():
    path = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0080_transparent_baseline_v13_runtime_seal.py"
    )
    spec = importlib.util.spec_from_file_location("baseline_v13_runtime_seal_0080", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v13_recipe_uses_the_append_only_runner_and_runtime_bundle() -> None:
    root = Path(__file__).parents[1]
    runner = root / "scripts" / "run_multifactor_backtest.py"

    assert RECIPE_VERSION == FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION
    for recipe_id in (
        "short_relative_strength",
        "swing_trend",
        "long_quality_value",
    ):
        assert target_runner_for_recipe(recipe_id, RECIPE_VERSION) == (
            FAIL_CLOSED_EXECUTION_TARGET_RUNNER_SHA256
        )
        assert target_runtime_bundle_for_recipe(recipe_id, RECIPE_VERSION) == (
            FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256
        )
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == (
        FAIL_CLOSED_EXECUTION_TARGET_RUNNER_SHA256
    )
    assert position_risk_bundle_sha256(root) == (
        FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256
    )


def test_v12_seals_remain_unchanged_by_v13() -> None:
    assert POSITION_RISK_TARGET_RECIPE_VERSION == (
        "qlib-rdagent-single-mainline-2026-08-30-v12"
    )
    assert POSITION_RISK_TARGET_RUNNER_SHA256 == (
        "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
    )
    assert POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256 == (
        "6366bd2b77c1c60ea4069afde43d5335c1e362c18acf2955f13902e7ba9ccbc6"
    )


def test_v13_migration_and_sqlalchemy_metadata_share_one_seal() -> None:
    migration = _migration_module()
    definition = migration._v13_runtime_identity_constraint()
    constraints = {
        constraint.name: str(constraint.sqltext)
        for constraint in strategy_versions.constraints
        if constraint.name is not None
    }

    assert migration.revision == "0080_baseline_v13_seal"
    assert migration.down_revision == "0079_autopilot_horizon_cycles"
    assert FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION in definition
    assert FAIL_CLOSED_EXECUTION_TARGET_RUNNER_SHA256 in definition
    assert FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256 in definition
    assert constraints["ck_strategy_versions_v13_runtime_identity"] == definition


def test_v13_formal_admission_binds_worker_image_across_all_artifacts() -> None:
    digest = "sha256:" + "d" * 64
    version = {
        "config": {
            "recipe_id": "short_relative_strength",
            "recipe_version": FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION,
            "transparent_baseline_bootstrap": {
                TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: digest,
            },
        }
    }
    manifest = {TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: digest}
    provenance = {TRANSPARENT_BASELINE_RESULT_WORKER_RUNTIME_IMAGE_FIELD: digest}

    assert _transparent_worker_runtime_failures(version, manifest, provenance) == []
    assert _transparent_worker_runtime_failures(version, {}, provenance) == [
        "strategy backtest manifest worker runtime image differs from the sealed version"
    ]
    assert _transparent_worker_runtime_failures(version, manifest, {}) == [
        "formal result worker runtime image differs from the sealed version"
    ]
