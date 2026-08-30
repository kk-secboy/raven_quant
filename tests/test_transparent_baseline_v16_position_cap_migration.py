from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from quant_data.database import strategy_versions
from quant_platform import transparent_baseline_lockbox as lockbox
from quant_platform import transparent_baseline_runner as runner

pytestmark = pytest.mark.no_database


def _migration_module(filename: str = "0084_transparent_baseline_v16_position_cap_repair.py"):
    path = Path(__file__).parents[1] / "migrations" / "versions" / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v16_migration_is_one_atomic_successor_without_weakening_v15_rules() -> None:
    migration = _migration_module()
    previous_repair = _migration_module(
        "0082_transparent_baseline_v15_runtime_repair.py"
    )
    previous_evidence = _migration_module(
        "0083_transparent_baseline_single_member_evidence.py"
    )

    assert migration.revision == "0084_baseline_v16_pos_cap"
    assert migration.down_revision == "0083_baseline_v15_evidence"
    assert migration._previous_repair_constraint() == previous_repair._repair_constraint()
    assert migration._previous_evidence_constraint() == previous_evidence._evidence_constraint()
    assert migration._V7_CONTRACT not in migration._previous_repair_constraint()
    assert migration._V7_RECEIPT_SHA256 not in migration._previous_evidence_constraint()
    assert migration._V6_CONTRACT in migration._repair_constraint()
    assert migration._V6_RECEIPT_SHA256 in migration._v16_evidence_constraint()


def test_v7_same_lineage_exception_is_bound_to_the_exact_production_evidence() -> None:
    migration = _migration_module()
    definition = migration._v7_generation()

    expected_bindings = {
        "_V7_SOURCE_COMMIT": lockbox.DISCRETE_MAX_POSITION_SOURCE_COMMIT,
        "_V7_SOURCE_BATCH_SHA256": lockbox.DISCRETE_MAX_POSITION_SOURCE_BATCH_SHA256,
        "_V7_SOURCE_DATASET_IDENTITY_SHA256": (
            lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET_IDENTITY_SHA256
        ),
        "_V7_SOURCE_DATASET_LINEAGE_ID": (
            lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET_LINEAGE_ID
        ),
        "_V7_SOURCE_RUNNER_SHA256": lockbox.DISCRETE_MAX_POSITION_SOURCE_RUNNER_SHA256,
        "_V7_SOURCE_RUNTIME_BUNDLE_SHA256": (
            lockbox.DISCRETE_MAX_POSITION_SOURCE_BUNDLE_SHA256
        ),
        "_V7_TARGET_RECIPE": lockbox.DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION,
        "_V7_TARGET_RUNNER_SHA256": lockbox.DISCRETE_MAX_POSITION_TARGET_RUNNER_SHA256,
        "_V7_TARGET_RUNTIME_BUNDLE_SHA256": (
            lockbox.DISCRETE_MAX_POSITION_TARGET_BUNDLE_SHA256
        ),
        "_V7_RUNTIME_CONTRACT": lockbox.DISCRETE_MAX_POSITION_RUNTIME_CONTRACT_VERSION,
        "_V7_SOURCE_ARTIFACT_INVENTORIES_SHA256": (
            lockbox.DISCRETE_MAX_POSITION_SOURCE_ARTIFACT_INVENTORIES_SHA256
        ),
        "_V7_SOURCE_SELECTION_SHA256": (
            lockbox.DISCRETE_MAX_POSITION_SOURCE_SELECTION_SHA256
        ),
        "_V7_SOURCE_UNAVAILABLE_HORIZONS_SHA256": (
            lockbox.DISCRETE_MAX_POSITION_SOURCE_UNAVAILABLE_HORIZONS_SHA256
        ),
    }
    for migration_name, expected in expected_bindings.items():
        assert getattr(migration, migration_name) == expected
        assert expected in definition

    source = lockbox.DISCRETE_MAX_POSITION_SOURCE_BINDINGS[
        migration._V7_SOURCE_BACKTEST_ID
    ]
    assert migration._V7_SOURCE_JOB_ID == source["job_id"]
    assert migration._V7_SOURCE_STRATEGY_VERSION_ID == source["strategy_version_id"]
    assert migration._V7_RECEIPT_SHA256 == (
        "dbe47a338ee6fd75c5b6dc471775aaa0155697fa7c31012561f362c7cfb5128a"
    )
    for value in (
        migration._V7_RECEIPT_SHA256,
        migration._V7_SOURCE_BACKTEST_ID,
        migration._V7_SOURCE_JOB_ID,
        migration._V7_SOURCE_STRATEGY_VERSION_ID,
        *migration._V7_UNAVAILABLE_EVIDENCE_SHA256S,
        *migration._V7_TARGET_CHANGE_CODES,
    ):
        assert value in definition


def test_v16_evidence_rule_adds_only_the_exact_v7_single_member() -> None:
    migration = _migration_module()
    definition = migration._v16_evidence_constraint()

    assert "jsonb_array_length(source_backtest_ids_json) = 3" in definition
    assert "jsonb_array_length(target_strategy_version_ids_json) = 3" in definition
    assert definition.count("jsonb_array_length(source_backtest_ids_json) = 1") == 2
    assert definition.count("jsonb_array_length(target_strategy_version_ids_json) = 1") == 2
    for value in (
        migration._V6_RECEIPT_SHA256,
        migration._V7_RECEIPT_SHA256,
        migration._V7_SOURCE_BACKTEST_ID,
        migration._V7_TARGET_RECIPE,
        migration._V7_CONTRACT,
        migration._V7_GENERATION,
    ):
        assert value in definition


def test_v16_runtime_identity_matches_sqlalchemy_metadata() -> None:
    migration = _migration_module()
    definition = migration._v16_runtime_identity_constraint()
    constraints = {
        constraint.name: str(constraint.sqltext)
        for constraint in strategy_versions.constraints
        if constraint.name is not None
    }

    assert migration._V16_RECIPE == runner.DISCRETE_MAX_POSITION_REPAIR_TARGET_RECIPE_VERSION
    assert migration._V16_RUNNER_SHA256 == (
        runner.DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNNER_SHA256
    )
    assert migration._V16_RUNTIME_BUNDLE_SHA256 == (
        runner.DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
    )
    assert constraints[migration._RUNTIME_CONSTRAINT] == definition


class _FakeBind:
    def __init__(self, counts: tuple[int, int, int]) -> None:
        self._counts = iter(counts)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def scalar(self, statement: Any, params: dict[str, str]) -> int:
        self.calls.append((str(statement), params))
        return next(self._counts)


class _FakeOp:
    def __init__(self, counts: tuple[int, int, int] = (0, 0, 0)) -> None:
        self.bind = _FakeBind(counts)
        self.actions: list[tuple[str, str, str, str | None]] = []

    def get_bind(self) -> _FakeBind:
        return self.bind

    def drop_constraint(self, name: str, table: str, **kwargs: Any) -> None:
        self.actions.append(("drop", name, table, None))

    def create_check_constraint(
        self, name: str, table: str, definition: str, **kwargs: Any
    ) -> None:
        self.actions.append(("create", name, table, definition))


def test_upgrade_replaces_both_registry_checks_and_adds_runtime_seal() -> None:
    migration = _migration_module()
    fake_op = _FakeOp()
    migration.op = fake_op

    migration.upgrade()

    assert [(action, name) for action, name, _, _ in fake_op.actions] == [
        ("drop", migration._REPAIR_CONSTRAINT),
        ("drop", migration._EVIDENCE_CONSTRAINT),
        ("create", migration._REPAIR_CONSTRAINT),
        ("create", migration._EVIDENCE_CONSTRAINT),
        ("create", migration._RUNTIME_CONSTRAINT),
    ]
    assert fake_op.actions[2][3] == migration._repair_constraint()
    assert fake_op.actions[3][3] == migration._v16_evidence_constraint()
    assert fake_op.actions[4][3] == migration._v16_runtime_identity_constraint()


@pytest.mark.parametrize("counts", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_downgrade_refuses_each_immutable_v16_or_v7_evidence_class(
    counts: tuple[int, int, int],
) -> None:
    migration = _migration_module()
    fake_op = _FakeOp(counts)
    migration.op = fake_op

    with pytest.raises(RuntimeError, match="immutable v16 strategy"):
        migration.downgrade()

    assert fake_op.actions == []
    assert len(fake_op.bind.calls) == 3
    assert "strategy_versions" in fake_op.bind.calls[0][0]
    assert "transparent_baseline_pre_result_repairs" in fake_op.bind.calls[1][0]
    assert "audit_events" in fake_op.bind.calls[2][0]
    assert fake_op.bind.calls[2][1]["receipt"] == migration._V7_RECEIPT_SHA256


def test_clean_downgrade_restores_exact_0082_and_0083_constraints() -> None:
    migration = _migration_module()
    fake_op = _FakeOp()
    migration.op = fake_op

    migration.downgrade()

    assert [(action, name) for action, name, _, _ in fake_op.actions] == [
        ("drop", migration._RUNTIME_CONSTRAINT),
        ("drop", migration._REPAIR_CONSTRAINT),
        ("drop", migration._EVIDENCE_CONSTRAINT),
        ("create", migration._REPAIR_CONSTRAINT),
        ("create", migration._EVIDENCE_CONSTRAINT),
    ]
    assert fake_op.actions[3][3] == migration._previous_repair_constraint()
    assert fake_op.actions[4][3] == migration._previous_evidence_constraint()
