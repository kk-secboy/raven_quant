from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import pytest

from quant_data.database import strategy_versions
from quant_platform import transparent_baseline_lockbox as lockbox
from quant_platform import transparent_baseline_runner as runner

pytestmark = pytest.mark.no_database


def _migration_module(
    filename: str = "0085_transparent_baseline_v17_industry_capacity_repair.py",
):
    path = Path(__file__).parents[1] / "migrations" / "versions" / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v17_migration_is_one_atomic_successor_preserving_exact_v16_rules() -> None:
    migration = _migration_module()
    previous = _migration_module(
        "0084_transparent_baseline_v16_position_cap_repair.py"
    )

    assert migration.revision == "0085_baseline_v17_industry"
    assert migration.down_revision == "0084_baseline_v16_pos_cap"
    assert migration._previous_repair_constraint() == previous._repair_constraint()
    assert migration._previous_evidence_constraint() == previous._v16_evidence_constraint()
    assert migration._V8_CONTRACT not in migration._previous_repair_constraint()
    assert migration._V8_RECEIPT_SHA256 not in migration._previous_evidence_constraint()
    assert migration._repair_constraint() == (
        "("
        + previous._repair_constraint()[:-1]
        + " OR "
        + migration._v8_generation()
        + ")) IS TRUE"
    )
    assert migration._v17_evidence_constraint() == (
        "("
        + previous._v16_evidence_constraint()[:-1]
        + " OR ("
        + migration._v8_single_member_evidence()
        + "))) IS TRUE"
    )


def test_v8_same_lineage_exception_is_bound_to_exact_production_evidence() -> None:
    migration = _migration_module()
    definition = migration._v8_generation()

    # PostgreSQL CHECK constraints accept NULL.  The exact V8 branch must
    # collapse a missing JSON key to FALSE instead of letting malformed
    # registry evidence pass as UNKNOWN.
    assert definition.endswith(") IS TRUE")

    expected_bindings = {
        "_V8_CONTRACT": lockbox.PRE_RESULT_REPAIR_CONTRACT_VERSION_V8,
        "_V8_GENERATION": lockbox.TOPK_INDUSTRY_CAPACITY_REPAIR_GENERATION,
        "_V8_SOURCE_COMMIT": lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_COMMIT,
        "_V8_SOURCE_BATCH_SHA256": (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_BATCH_SHA256
        ),
        "_V8_SOURCE_DATASET_IDENTITY_SHA256": (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_IDENTITY_SHA256
        ),
        "_V8_SOURCE_DATASET_LINEAGE_ID": (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_LINEAGE_ID
        ),
        "_V8_SOURCE_RUNNER_SHA256": (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_RUNNER_SHA256
        ),
        "_V8_SOURCE_RUNTIME_BUNDLE_SHA256": (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_BUNDLE_SHA256
        ),
        "_V8_TARGET_RECIPE": lockbox.TOPK_INDUSTRY_CAPACITY_TARGET_RECIPE_VERSION,
        "_V8_TARGET_RUNNER_SHA256": (
            lockbox.TOPK_INDUSTRY_CAPACITY_TARGET_RUNNER_SHA256
        ),
        "_V8_TARGET_RUNTIME_BUNDLE_SHA256": (
            lockbox.TOPK_INDUSTRY_CAPACITY_TARGET_BUNDLE_SHA256
        ),
        "_V8_RUNTIME_CONTRACT": (
            lockbox.TOPK_INDUSTRY_CAPACITY_RUNTIME_CONTRACT_VERSION
        ),
        "_V8_SOURCE_ARTIFACT_INVENTORIES_SHA256": (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_ARTIFACT_INVENTORIES_SHA256
        ),
        "_V8_SOURCE_SELECTION_SHA256": (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_SELECTION_SHA256
        ),
        "_V8_SOURCE_UNAVAILABLE_HORIZONS_SHA256": (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_UNAVAILABLE_HORIZONS_SHA256
        ),
    }
    for migration_name, expected in expected_bindings.items():
        assert getattr(migration, migration_name) == expected
        assert expected in definition

    source = lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_BINDINGS[
        migration._V8_SOURCE_BACKTEST_ID
    ]
    assert migration._V8_SOURCE_JOB_ID == source["job_id"]
    assert migration._V8_SOURCE_STRATEGY_VERSION_ID == source["strategy_version_id"]
    assert migration._V8_UNAVAILABLE_EVIDENCE_SHA256S == tuple(
        sorted(lockbox.TOPK_INDUSTRY_CAPACITY_UNAVAILABLE_EVIDENCE_SHA256S)
    )
    assert migration._V8_TARGET_CHANGE_CODES == (
        lockbox.TOPK_INDUSTRY_CAPACITY_TARGET_CHANGE_CODES
    )
    assert re.fullmatch(r"[0-9a-f]{64}", migration._V8_RECEIPT_SHA256)
    assert migration._V8_RECEIPT_SHA256 == (
        "980ea643755d261cc7ee39ffec5e23af3f8e27ebdc1e2f2a647e0802b9dc636d"
    )
    for value in (
        migration._V8_RECEIPT_SHA256,
        migration._V8_SOURCE_BACKTEST_ID,
        migration._V8_SOURCE_JOB_ID,
        migration._V8_SOURCE_STRATEGY_VERSION_ID,
        *migration._V8_UNAVAILABLE_EVIDENCE_SHA256S,
        *migration._V8_TARGET_CHANGE_CODES,
    ):
        assert value in definition


def test_v17_evidence_rule_appends_only_exact_v8_single_member() -> None:
    migration = _migration_module()
    previous = migration._previous_evidence_constraint()
    definition = migration._v17_evidence_constraint()

    assert definition.startswith("(" + previous[:-1])
    assert definition.endswith(") IS TRUE")
    assert "jsonb_array_length(source_backtest_ids_json) = 3" in definition
    assert "jsonb_array_length(target_strategy_version_ids_json) = 3" in definition
    assert definition.count("jsonb_array_length(source_backtest_ids_json) = 1") == 3
    assert (
        definition.count("jsonb_array_length(target_strategy_version_ids_json) = 1")
        == 3
    )
    assert definition.count(migration._V8_RECEIPT_SHA256) == 1
    assert migration._v8_single_member_evidence().endswith(") IS TRUE")
    for value in (
        migration._V8_SOURCE_BACKTEST_ID,
        migration._V8_TARGET_RECIPE,
        migration._V8_CONTRACT,
        migration._V8_GENERATION,
    ):
        assert value in definition


def test_v17_runtime_identity_matches_sqlalchemy_metadata() -> None:
    migration = _migration_module()
    definition = migration._v17_runtime_identity_constraint()
    constraints = {
        constraint.name: str(constraint.sqltext)
        for constraint in strategy_versions.constraints
        if constraint.name is not None
    }

    assert migration._V17_RECIPE == (
        runner.TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION
    )
    assert migration._V17_RUNNER_SHA256 == (
        runner.TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RUNNER_SHA256
    )
    assert migration._V17_RUNTIME_BUNDLE_SHA256 == (
        runner.TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
    )
    assert migration._V17_RUNTIME_BUNDLE_SHA256 == (
        "c3d6aeb00a8f5286ee117f885231b43ad49e461c5183f5cbf32c4601824bb9fc"
    )
    assert constraints[migration._RUNTIME_CONSTRAINT] == definition


class _FakeBind:
    def __init__(self, counts: tuple[int | None, int | None, int | None]) -> None:
        self._counts = iter(counts)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def scalar(self, statement: Any, params: dict[str, str]) -> int | None:
        self.calls.append((str(statement), params))
        return next(self._counts)


class _FakeOp:
    def __init__(
        self,
        counts: tuple[int | None, int | None, int | None] = (0, 0, 0),
    ) -> None:
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


def test_upgrade_replaces_registry_checks_and_adds_v17_runtime_seal() -> None:
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
    assert fake_op.actions[3][3] == migration._v17_evidence_constraint()
    assert fake_op.actions[4][3] == migration._v17_runtime_identity_constraint()


@pytest.mark.parametrize("counts", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_downgrade_refuses_each_immutable_v17_or_v8_evidence_class(
    counts: tuple[int, int, int],
) -> None:
    migration = _migration_module()
    fake_op = _FakeOp(counts)
    migration.op = fake_op

    with pytest.raises(RuntimeError, match="immutable v17 strategy"):
        migration.downgrade()

    assert fake_op.actions == []
    assert len(fake_op.bind.calls) == 3
    assert "strategy_versions" in fake_op.bind.calls[0][0]
    assert fake_op.bind.calls[0][1] == {"recipe": migration._V17_RECIPE}
    assert "transparent_baseline_pre_result_repairs" in fake_op.bind.calls[1][0]
    assert fake_op.bind.calls[1][1] == {
        "contract": migration._V8_CONTRACT,
        "generation": migration._V8_GENERATION,
    }
    assert "audit_events" in fake_op.bind.calls[2][0]
    assert fake_op.bind.calls[2][1] == {
        "receipt": migration._V8_RECEIPT_SHA256,
        "contract": migration._V8_CONTRACT,
        "generation": migration._V8_GENERATION,
    }


@pytest.mark.parametrize("counts", [(0, 0, 0), (None, None, None)])
def test_clean_downgrade_treats_null_as_zero_and_restores_exact_0084_constraints(
    counts: tuple[int | None, int | None, int | None],
) -> None:
    migration = _migration_module()
    fake_op = _FakeOp(counts)
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
