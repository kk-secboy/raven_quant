from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from sqlalchemy import CheckConstraint

from quant_data.database import (
    backtest_runs,
    strategy_forward_only_rehabilitations,
    strategy_incomplete_family_eligibilities,
    strategy_versions,
)

pytestmark = pytest.mark.no_database

_ROOT = Path(__file__).resolve().parents[1]
_MIGRATION_PATH = (
    _ROOT / "migrations" / "versions" / "0088_forward_only_rehabilitation.py"
)


def _migration_module():
    spec = importlib.util.spec_from_file_location(
        "forward_only_rehabilitation_0088",
        _MIGRATION_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_sql(table, name: str) -> str:
    constraint = next(
        item
        for item in table.constraints
        if isinstance(item, CheckConstraint) and item.name == name
    )
    return str(constraint.sqltext)


def test_evidence_mode_backfill_default_and_v18_scope_are_fail_closed() -> None:
    migration = _migration_module()
    assert migration.revision == "0088_forward_only_rehab"
    assert migration.down_revision == "0087_recovery_recipe_path"

    for table in (strategy_versions, backtest_runs):
        default = table.c.evidence_mode.server_default
        assert default is not None
        assert str(default.arg) == "legacy_ambiguous"

    runtime_check = _check_sql(
        strategy_versions,
        "ck_strategy_versions_v18_runtime_identity",
    )
    assert "recipe_id', '') = 'short_relative_strength'" in runtime_check
    assert "evidence_mode = 'consumed_historical_replay'" in runtime_check
    assert "v18' THEN (COALESCE" in runtime_check
    assert "v18' AND" not in runtime_check
    assert "ELSE true END) IS TRUE" in runtime_check


def test_incomplete_family_source_rows_are_restrict_foreign_keys() -> None:
    expected = {
        "source_strategy_version_id": "quantlab.strategy_versions.id",
        "source_backtest_id": "quantlab.backtest_runs.id",
        "source_job_id": "quantlab.jobs.id",
    }
    for column_name, target in expected.items():
        foreign_keys = strategy_incomplete_family_eligibilities.c[
            column_name
        ].foreign_keys
        assert len(foreign_keys) == 1
        foreign_key = next(iter(foreign_keys))
        assert foreign_key.target_fullname == target
        assert foreign_key.ondelete == "RESTRICT"

    migration_source = _MIGRATION_PATH.read_text(encoding="utf-8")
    for target in expected.values():
        assert f'sa.ForeignKey(f"{{SCHEMA}}.{target.removeprefix("quantlab.")}"' in (
            migration_source
        )


def test_rehabilitation_truth_markers_require_json_booleans_not_castable_text() -> None:
    truth_check = _check_sql(
        strategy_forward_only_rehabilitations,
        "ck_forward_only_rehabilitation_evidence",
    )
    expected = {
        "historical_replay_opened": "true",
        "final_oos_opened": "true",
        "capital_eligible": "false",
        "consumed_oos_replayed": "true",
        "sealed_final_oos": "false",
        "unseen_oos": "false",
    }
    for field, value in expected.items():
        assert f"qualification_json -> '{field}' = '{value}'::jsonb" in truth_check
        assert f"qualification_json ->> '{field}')::boolean" not in truth_check
    assert truth_check.endswith("IS TRUE")

    migration_source = _MIGRATION_PATH.read_text(encoding="utf-8")
    for field, value in expected.items():
        assert (
            f'"AND qualification_json -> \'{field}\' = \'{value}\'::jsonb "'
            in migration_source
        )


def test_rehabilitation_jsonb_checks_use_supported_exact_key_operators() -> None:
    truth_check = _check_sql(
        strategy_forward_only_rehabilitations,
        "ck_forward_only_rehabilitation_evidence",
    )
    exact_unavailable_keys = (
        "source_unavailable_evidence_sha256s_json - "
        "ARRAY['swing_1_6m','long_1_3y']) = '{}'::jsonb"
    )

    assert "source_unavailable_evidence_sha256s_json ?&" in truth_check
    assert exact_unavailable_keys in truth_check

    migration_source = _MIGRATION_PATH.read_text(encoding="utf-8")
    assert "jsonb_object_length" not in migration_source
    assert '"AND (source_unavailable_evidence_sha256s_json - "' in migration_source
    assert (
        '"ARRAY[\'swing_1_6m\',\'long_1_3y\']) = \'{}\'::jsonb "'
        in migration_source
    )
    assert (
        "NEW.replay_periods_json -\n"
        "                    ARRAY['start','end','historical_start','historical_end']) ="
        in migration_source
    )
    assert "'{{}}'::jsonb" in migration_source


def test_0088_rowtype_functions_bypass_sqlalchemy_percent_rewriting() -> None:
    source = _MIGRATION_PATH.read_text(encoding="utf-8")

    assert "%%ROWTYPE" not in source
    assert sum(
        line.strip().endswith("%ROWTYPE;") for line in source.splitlines()
    ) == 11
    assert "execution_options(no_parameters=True).exec_driver_sql(statement)" in source
    assert (
        '_execute_raw_ddl(\n        f"""\n'
        "        CREATE OR REPLACE FUNCTION quantlab.validate_forward_only_rehabilitation()"
        in source
    )
    assert (
        '_execute_raw_ddl(\n        f"""\n'
        "        CREATE OR REPLACE FUNCTION quantlab.validate_incomplete_family_eligibility()"
        in source
    )


def test_0088_triggers_bind_family_runtime_periods_criteria_and_artifacts() -> None:
    source = _MIGRATION_PATH.read_text(encoding="utf-8")

    for fragment in (
        "target_strategy.economic_hypothesis_group",
        "NEW.qualification_json -> 'eligible_strategy_version_ids'",
        "NEW.qualification_json -> 'trial_count_audit'",
        "family_eligibility.strategy_trial_count",
        "family_eligibility.trial_count_audit_sha256",
        "'target_runner_sha256' =",
        "'target_runtime_bundle_sha256' = NEW.runtime_bundle_sha256",
        "'target_worker_runtime_image_digest' =",
        "source_backtest.periods_json = NEW.replay_periods_json",
        "NEW.replay_periods_json -",
        "ARRAY['start','end','historical_start','historical_end']",
        "'historical_start' = '2008-01-02'",
        "'historical_end' = '2018-10-10'",
        "NEW.qualification_json -> 'forward_criteria' =",
        "NEW.forward_criteria_json",
        "'execution_manifest_sha256' = NEW.replay_manifest_sha256",
        "'artifact_manifest_sha256' =",
        "NEW.replay_artifact_manifest_sha256",
        "NEW.qualification_json ->> 'replay_result_sha256'",
        "NEW.qualification_json ->> 'replay_daily_returns_sha256'",
    ):
        assert fragment in source

    assert source.count("BEFORE UPDATE OR DELETE ON quantlab.strategy_") >= 2
    assert source.count("BEFORE TRUNCATE ON quantlab.strategy_") >= 2
    assert "trg_forward_only_rehabilitation_append_only" in source
    assert "trg_incomplete_family_eligibility_append_only" in source
    assert "trg_forward_only_rehabilitation_no_truncate" in source
    assert "trg_incomplete_family_eligibility_no_truncate" in source


def test_0088_downgrade_locks_receipts_before_checking_or_dropping() -> None:
    source = _MIGRATION_PATH.read_text(encoding="utf-8")
    downgrade = source[source.index("def downgrade()") :]
    advisory = "forward-only-rehabilitation:0088-downgrade"
    lock_start = downgrade.index(
        '"LOCK TABLE quantlab.strategy_versions, quantlab.backtest_runs, "'
    )
    count_start = downgrade.index('"SELECT (SELECT count(*) FROM "')
    evidence_count_start = downgrade.index(
        '"WHERE evidence_mode <> \'legacy_ambiguous\') + "'
    )
    drop_start = downgrade.index(
        'op.drop_table("strategy_forward_only_rehabilitations"'
    )

    assert advisory in downgrade
    assert '"IN ACCESS EXCLUSIVE MODE"' in downgrade
    assert "quantlab.strategy_incomplete_family_eligibilities" in downgrade
    assert "quantlab.strategy_forward_only_rehabilitations" in downgrade
    assert lock_start < count_start < evidence_count_start < drop_start
