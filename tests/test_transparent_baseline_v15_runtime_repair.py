from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from quant_data.database import strategy_versions
from quant_platform.strategy_recipes import RECIPE_VERSION
from quant_platform.transparent_baseline_runner import (
    FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION,
    FILL_AWARE_HOLDING_AGE_TARGET_RUNNER_SHA256,
    FILL_AWARE_HOLDING_AGE_TARGET_RUNTIME_BUNDLE_SHA256,
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION,
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNNER_SHA256,
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256,
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
        / "0082_transparent_baseline_v15_runtime_repair.py"
    )
    spec = importlib.util.spec_from_file_location(
        "baseline_v15_runtime_repair_0082", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v15_recipe_uses_the_append_only_runner_and_runtime_bundle() -> None:
    root = Path(__file__).parents[1]
    runner = root / "scripts" / "run_multifactor_backtest.py"

    assert RECIPE_VERSION == SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION
    assert SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256 == (
        "e361d85d69d77cb6e0de7072db6f8aaff5d83f1e7902fe16ef06c2e28fce1867"
    )
    for recipe_id in (
        "short_relative_strength",
        "swing_trend",
        "long_quality_value",
    ):
        assert target_runner_for_recipe(recipe_id, RECIPE_VERSION) == (
            SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNNER_SHA256
        )
        assert target_runtime_bundle_for_recipe(recipe_id, RECIPE_VERSION) == (
            SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
        )
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == (
        SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNNER_SHA256
    )
    assert position_risk_bundle_sha256(root) == (
        SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
    )


def test_v14_seals_remain_unchanged_by_v15() -> None:
    assert FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION == (
        "qlib-rdagent-single-mainline-2026-08-30-v14"
    )
    assert FILL_AWARE_HOLDING_AGE_TARGET_RUNNER_SHA256 == (
        "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
    )
    assert FILL_AWARE_HOLDING_AGE_TARGET_RUNTIME_BUNDLE_SHA256 == (
        "04f9dce110aaf44d32972c68db3edefafbd08cfdbf1f7dfb98aac9a15409bfe5"
    )


def test_v15_migration_and_sqlalchemy_metadata_share_one_seal() -> None:
    migration = _migration_module()
    definition = migration._v15_runtime_identity_constraint()
    constraints = {
        constraint.name: str(constraint.sqltext)
        for constraint in strategy_versions.constraints
        if constraint.name is not None
    }

    assert migration.revision == "0082_baseline_v15_repair"
    assert migration.down_revision == "0081_baseline_v14_seal"
    assert SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION in definition
    assert SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNNER_SHA256 in definition
    assert SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256 in definition
    assert constraints["ck_strategy_versions_v15_runtime_identity"] == definition


def test_v15_migration_preserves_v2_v5_and_adds_only_exact_v6() -> None:
    migration = _migration_module()
    definition = migration._repair_constraint()

    for contract in (
        "transparent-baseline-pre-result-repair-v2",
        "transparent-baseline-pre-result-repair-v3",
        "transparent-baseline-pre-result-repair-v4",
        "transparent-baseline-pre-result-repair-v5",
        "transparent-baseline-pre-result-repair-v6",
    ):
        assert contract in definition
    assert migration._V6_GENERATION in definition
    assert migration._V6_TARGET_RECIPE in definition
    assert migration._V6_TARGET_RUNNER_SHA256 in definition
    assert migration._V6_TARGET_RUNTIME_BUNDLE_SHA256 in definition
    assert migration._V6_SOURCE_BATCH_SHA256 in definition
    assert migration._V6_SOURCE_BACKTEST_ID in definition
    assert migration._V6_SOURCE_JOB_ID in definition
    assert migration._V6_SOURCE_STRATEGY_VERSION_ID in definition
    assert migration._V6_SOURCE_SELECTION_SHA256 in definition
    assert migration._V6_SOURCE_UNAVAILABLE_HORIZONS_SHA256 in definition
    for evidence_sha256 in migration._V6_UNAVAILABLE_EVIDENCE_SHA256S:
        assert evidence_sha256 in definition
    assert "source_batch_sha256 <> target_batch_sha256" in definition
