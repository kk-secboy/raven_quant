from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_database


def _migration_module():
    path = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0083_transparent_baseline_single_member_evidence.py"
    )
    spec = importlib.util.spec_from_file_location(
        "baseline_v15_single_member_evidence_0083", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v15_evidence_migration_keeps_three_member_rule_and_one_exact_exception() -> None:
    migration = _migration_module()
    definition = migration._evidence_constraint()

    assert migration.revision == "0083_baseline_v15_evidence"
    assert migration.down_revision == "0082_baseline_v15_repair"
    assert "jsonb_array_length(source_backtest_ids_json) = 3" in definition
    assert "jsonb_array_length(target_strategy_version_ids_json) = 3" in definition
    assert "jsonb_array_length(source_backtest_ids_json) = 1" in definition
    assert "jsonb_array_length(target_strategy_version_ids_json) = 1" in definition
    assert migration._V6_RECEIPT_SHA256 in definition
    assert migration._V6_SOURCE_BACKTEST_ID in definition
    assert migration._V6_CONTRACT in definition
    assert migration._V6_GENERATION in definition
    assert migration._V6_TARGET_RECIPE in definition
    assert (
        "verification_json -> 'target_strategy_version_ids' = "
        "target_strategy_version_ids_json"
    ) in definition


def test_legacy_evidence_constraint_does_not_admit_single_member_rows() -> None:
    migration = _migration_module()
    definition = migration._legacy_evidence_constraint()

    assert "jsonb_array_length(source_backtest_ids_json) = 3" in definition
    assert "jsonb_array_length(target_strategy_version_ids_json) = 3" in definition
    assert " = 1" not in definition
