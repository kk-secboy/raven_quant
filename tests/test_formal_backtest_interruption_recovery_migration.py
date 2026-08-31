from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

import pytest

from quant_data.database import formal_backtest_interruption_recoveries
from quant_platform import formal_backtest_interruption_recovery as recovery

pytestmark = pytest.mark.no_database


def _migration_module():
    filename = "0086_formal_backtest_interruption_recovery.py"
    path = Path(__file__).parents[1] / "migrations" / "versions" / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recipe_binding_migration_module():
    filename = "0087_formal_backtest_recovery_recipe_binding.py"
    path = Path(__file__).parents[1] / "migrations" / "versions" / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_pins_the_exact_v17_receipt_and_external_journal() -> None:
    migration = _migration_module()

    assert migration.revision == "0086_formal_bt_interrupt"
    assert migration.down_revision == "0085_baseline_v17_industry"
    assert migration.RECEIPT_SHA256 == recovery.V17_INTERRUPTION_RECOVERY_RECEIPT_SHA256
    assert migration.EXTERNAL_OBSERVED_AT == (
        recovery.V17_EXTERNAL_INTERRUPTION_OBSERVED_AT
    )
    assert migration.EXTERNAL_JOURNAL_SHA256 == recovery.V17_EXTERNAL_JOURNAL_SHA256
    assert migration._production_source_constraint().endswith(") IS TRUE")
    assert migration._receipt_constraint().endswith(") IS TRUE")
    for value in (
        recovery.V17_INTERRUPTION_RECOVERY_RECEIPT_SHA256,
        recovery.V17_EXTERNAL_INTERRUPTION_OBSERVED_AT,
        recovery.V17_EXTERNAL_JOURNAL_SHA256,
        migration.EXTERNAL_JOURNAL_EXCERPT_SQL,
    ):
        assert value in (
            migration._production_source_constraint()
            + migration._receipt_constraint()
        )

    metadata_constraints = {
        constraint.name: str(constraint.sqltext)
        for constraint in formal_backtest_interruption_recoveries.constraints
        if constraint.name is not None
    }
    assert recovery.V17_INTERRUPTION_RECOVERY_RECEIPT_SHA256 in metadata_constraints[
        "ck_formal_backtest_interruption_recovery_v17_source"
    ]
    receipt_constraint = metadata_constraints[
        "ck_formal_backtest_interruption_recovery_receipt"
    ]
    assert recovery.V17_EXTERNAL_JOURNAL_SHA256 in receipt_constraint
    assert migration.EXTERNAL_JOURNAL_EXCERPT_JSON in receipt_constraint
    assert receipt_constraint.endswith("IS TRUE")


def test_job_trigger_closes_payload_kind_and_third_attempt_bypasses() -> None:
    source = inspect.getsource(_migration_module().upgrade)

    assert "OLD.kind = 'strategy_backtest' OR NEW.kind = 'strategy_backtest'" in source
    assert "NEW.payload_json IS DISTINCT FROM OLD.payload_json" in source
    assert "formal backtest terminal jobs are immutable" in source
    assert "OLD.status IN ('succeeded', 'failed', 'cancelled')" in source
    assert "NEW.status IS DISTINCT FROM OLD.status" in source
    assert "NEW.attempts = OLD.attempts + 1" in source
    assert "formal backtest attempts advance only on claim" in source
    assert "NEW.attempts > 2" in source
    assert "NEW.max_attempts <> 2" in source
    assert "formal backtest recovery queued job is frozen" in source
    assert "RECOVERY_AUTHORIZER_APPLICATION_NAME" in source
    assert "trg_formal_backtest_interruption_recovery" in source
    assert "trg_guard_formal_backtest_job_attempts" in source


def test_aggregate_trigger_only_fail_closes_partial_terminal_without_erasing_metrics() -> None:
    source = inspect.getsource(_migration_module().upgrade)

    assert "controller_failure_finalization boolean" in source
    assert "'queued', 'running', 'succeeded', 'failed', 'cancelled'" in source
    assert "NEW.metrics_json IS NOT DISTINCT FROM OLD.metrics_json" in source
    assert "NEW.started_at IS NOT DISTINCT FROM OLD.started_at" in source
    assert "NEW.finished_at IS NOT DISTINCT FROM OLD.finished_at" in source
    assert "current_job.status IN ('failed', 'cancelled')" in source
    assert "IF controller_failure_finalization IS TRUE" in source
    assert "formal backtest recovery terminal aggregate is immutable" in source


class _FunctionDefinitionBind:
    def __init__(self, definition: str, *, recovery_count: int = 0) -> None:
        self.definition = definition
        self.recovery_count = recovery_count
        self.executed: list[tuple[str, dict[str, str] | None]] = []
        self.ddl: list[str] = []

    def scalar(self, statement: Any, params: dict[str, str] | None = None):
        sql = str(statement)
        if "pg_get_functiondef" in sql:
            return self.definition
        if "count(*)" in sql:
            return self.recovery_count
        raise AssertionError(f"unexpected scalar statement: {sql}")

    def execute(self, statement: Any, params: dict[str, str] | None = None) -> None:
        self.executed.append((str(statement), params))

    def exec_driver_sql(self, statement: str) -> None:
        self.ddl.append(statement)
        self.definition = statement


class _FunctionDefinitionOp:
    def __init__(self, bind: _FunctionDefinitionBind) -> None:
        self.bind = bind

    def get_bind(self) -> _FunctionDefinitionBind:
        return self.bind


def test_0087_upgrade_replaces_only_the_deployed_recipe_digest_path() -> None:
    migration = _recipe_binding_migration_module()
    bind = _FunctionDefinitionBind(
        "CREATE OR REPLACE FUNCTION "
        "quantlab.validate_formal_backtest_interruption_recovery()\n"
        "RETURNS trigger AS $$\n"
        "BEGIN\n"
        "            IF NOT ((NEW.verification_json = '{}'::jsonb) IS TRUE) THEN\n"
        "                RAISE EXCEPTION 'invalid';\n"
        "            END IF;\n"
        f"            IF {migration._TOP_LEVEL_RECIPE_BINDING} THEN\n"
        "RETURN NEW;\nEND IF;\n$$ LANGUAGE plpgsql;"
    )
    migration.op = _FunctionDefinitionOp(bind)

    migration.upgrade()

    assert migration.revision == "0087_recovery_recipe_path"
    assert migration.down_revision == "0086_formal_bt_interrupt"
    assert len(bind.ddl) == 4
    validator_ddl, guard_ddl, drop_trigger_ddl, create_trigger_ddl = bind.ddl
    assert migration._BOOTSTRAP_RECIPE_BINDING in validator_ddl
    assert migration._TOP_LEVEL_RECIPE_BINDING not in validator_ddl
    assert migration._LOCKED_VALIDATOR_BEGIN in validator_ddl
    for field in (
        "recipe_id",
        "recipe_version",
        "recipe_sha256",
        "target_runner_sha256",
        "target_runtime_bundle_sha256",
        "target_worker_runtime_image_digest",
        "dataset_identity_sha256",
        "dataset_lineage_id",
        "formal_periods",
    ):
        assert field in validator_ddl
    assert "guard_v17_recovery_strategy_version" in guard_ddl
    assert "recovery_job_status IN ('queued', 'running')" in guard_ddl
    assert "recovery_backtest_status IN ('queued', 'running')" in guard_ddl
    assert "to_jsonb(NEW) - ARRAY[" in guard_ddl
    for field in (
        "status",
        "promotion_stage",
        "approved_by",
        "approval_reason",
        "approved_at",
    ):
        assert f"'{field}'" in guard_ddl
    assert "DROP TRIGGER IF EXISTS trg_guard_v17_recovery_strategy_version" in (
        drop_trigger_ddl
    )
    assert "CREATE TRIGGER trg_guard_v17_recovery_strategy_version" in (
        create_trigger_ddl
    )
    assert any("pg_advisory_xact_lock" in sql for sql, _ in bind.executed)
    assert any(
        "LOCK TABLE quantlab.formal_backtest_interruption_recoveries" in sql
        for sql, _ in bind.executed
    )


@pytest.mark.parametrize(
    "body",
    [
        "RETURN NEW;",
        (
            "source_version.config_json ->> 'recipe_sha256' =\n"
            "                    "
            "'dee3551a73f2ebb3fbbbddfafdf99f4618e3dfdd98981fdb4b4d8849723d5fd8'; "
            "source_version.config_json ->> 'recipe_sha256' =\n"
            "                    "
            "'dee3551a73f2ebb3fbbbddfafdf99f4618e3dfdd98981fdb4b4d8849723d5fd8';"
        ),
    ],
)
def test_0087_upgrade_refuses_an_unexpected_function_source(body: str) -> None:
    migration = _recipe_binding_migration_module()
    bind = _FunctionDefinitionBind(
        "CREATE OR REPLACE FUNCTION "
        "quantlab.validate_formal_backtest_interruption_recovery() "
        "RETURNS trigger AS $$\n"
        "BEGIN\n"
        "            IF NOT ((NEW.verification_json = '{}'::jsonb) IS TRUE) THEN\n"
        "                RAISE EXCEPTION 'invalid';\n"
        "            END IF;\n"
        f"{body}\n"
        "END;\n$$ LANGUAGE plpgsql;"
    )
    migration.op = _FunctionDefinitionOp(bind)

    with pytest.raises(RuntimeError, match="validator source is not exact"):
        migration.upgrade()

    assert bind.ddl == []


def test_0087_replacement_fragments_match_the_0086_upgrade_source() -> None:
    current = _recipe_binding_migration_module()
    previous_source = inspect.getsource(_migration_module().upgrade)

    assert current._VALIDATOR_BEGIN in previous_source
    assert current._TOP_LEVEL_RECIPE_BINDING in previous_source


class _FakeBind:
    def __init__(self, counts: tuple[int, int]) -> None:
        self.counts = iter(counts)

    def scalar(self, _statement: Any, _params: dict[str, str] | None = None) -> int:
        return next(self.counts)

    def execute(self, _statement: Any) -> None:
        return None


class _FakeOp:
    def __init__(self, counts: tuple[int, int]) -> None:
        self.bind = _FakeBind(counts)
        self.mutated = False

    def get_bind(self) -> _FakeBind:
        return self.bind

    def __getattr__(self, _name: str):
        def mutation(*_args: Any, **_kwargs: Any) -> None:
            self.mutated = True

        return mutation


@pytest.mark.parametrize("counts", [(1, 0), (0, 1)])
def test_downgrade_refuses_registered_evidence_or_repeated_execution(
    counts: tuple[int, int],
) -> None:
    migration = _migration_module()
    fake_op = _FakeOp(counts)
    migration.op = fake_op

    with pytest.raises(RuntimeError, match="immutable formal-backtest"):
        migration.downgrade()

    assert fake_op.mutated is False
