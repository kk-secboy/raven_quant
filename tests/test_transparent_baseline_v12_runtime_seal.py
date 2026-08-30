from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from quant_data.database import strategy_versions
from quant_platform.strategy_recipes import RECIPE_VERSION
from quant_platform.strategy_store import _transparent_worker_runtime_failures
from quant_platform.transparent_baseline_runner import (
    POSITION_RISK_TARGET_RECIPE_VERSION,
    POSITION_RISK_TARGET_RUNNER_SHA256,
    POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256,
    RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION,
    RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256,
    RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256,
    TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_RESULT_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
    target_runner_for_recipe,
    target_runtime_bundle_for_recipe,
)

pytestmark = pytest.mark.no_database


def _migration_module():
    path = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0078_transparent_baseline_v12_runtime_seal.py"
    )
    spec = importlib.util.spec_from_file_location("baseline_v12_runtime_seal_0078", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v12_recipe_identity_remains_immutable_history() -> None:
    assert RECIPE_VERSION != POSITION_RISK_TARGET_RECIPE_VERSION
    for recipe_id in (
        "short_relative_strength",
        "swing_trend",
        "long_quality_value",
    ):
        assert target_runner_for_recipe(
            recipe_id, POSITION_RISK_TARGET_RECIPE_VERSION
        ) == (
            POSITION_RISK_TARGET_RUNNER_SHA256
        )
        assert target_runtime_bundle_for_recipe(
            recipe_id, POSITION_RISK_TARGET_RECIPE_VERSION
        ) == (
            POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256
        )
    assert POSITION_RISK_TARGET_RECIPE_VERSION == (
        "qlib-rdagent-single-mainline-2026-08-30-v12"
    )
    assert POSITION_RISK_TARGET_RUNNER_SHA256 == (
        "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
    )
    assert POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256 == (
        "6366bd2b77c1c60ea4069afde43d5335c1e362c18acf2955f13902e7ba9ccbc6"
    )


def test_v11_runtime_identity_remains_immutable_history() -> None:
    assert RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION == (
        "qlib-rdagent-single-mainline-2026-08-30-v11"
    )
    assert RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256 == (
        "64fa634b4e774279356c5655e70c741890ed9b208ccf75df757a378c0f56a432"
    )
    assert RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256 == (
        "687ad83efd9d734a238bd6b52b3f5e670cec1e5d165ec2b25cabcef724feb7cf"
    )
    assert POSITION_RISK_TARGET_RUNNER_SHA256 != (
        RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256
    )
    assert POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256 != (
        RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256
    )


def test_v12_migration_seals_identity_without_expanding_repair_generations() -> None:
    migration = _migration_module()
    definition = migration._v12_runtime_identity_constraint()

    assert migration.revision == "0078_baseline_v12_seal"
    assert migration.down_revision == "0077_baseline_input_scope"
    assert POSITION_RISK_TARGET_RECIPE_VERSION in definition
    assert POSITION_RISK_TARGET_RUNNER_SHA256 in definition
    assert POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256 in definition
    assert "target_worker_runtime_image_digest" in definition
    assert "^sha256:[0-9a-f]{64}$" in definition
    assert "transparent_baseline_pre_result_repairs" not in definition


def test_sqlalchemy_metadata_matches_the_v12_database_seal() -> None:
    migration = _migration_module()
    constraints = {
        constraint.name: str(constraint.sqltext)
        for constraint in strategy_versions.constraints
        if constraint.name is not None
    }

    assert constraints["ck_strategy_versions_v12_runtime_identity"] == (
        migration._v12_runtime_identity_constraint()
    )


def test_v12_formal_admission_binds_worker_image_across_all_artifacts() -> None:
    digest = "sha256:" + "d" * 64
    version = {
        "config": {
            "recipe_id": "short_relative_strength",
            "recipe_version": POSITION_RISK_TARGET_RECIPE_VERSION,
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


def test_compose_passes_worker_image_digest_through_shared_release_environment() -> None:
    compose = yaml.safe_load(
        (Path(__file__).parents[1] / "deploy" / "compose.yaml").read_text(
            encoding="utf-8"
        )
    )

    for service in ("api", "scheduler", "worker", "evaluation-worker", "paper-worker"):
        assert compose["services"][service]["environment"][
            "QUANTLAB_WORKER_RUNTIME_IMAGE_DIGEST"
        ] == "${QUANTLAB_WORKER_RUNTIME_IMAGE_DIGEST:-}"
