from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import CheckConstraint

from quant_data.database import strategy_versions
from quant_platform.transparent_baseline_runner import (
    FORWARD_ONLY_REHABILITATION_TARGET_RUNTIME_BUNDLE_SHA256,
)

pytestmark = pytest.mark.no_database

_ROOT = Path(__file__).resolve().parents[1]
_MIGRATION_PATH = (
    _ROOT
    / "migrations"
    / "versions"
    / "0089_forward_only_dataset_catalog_repair.py"
)


def _migration_module():
    spec = importlib.util.spec_from_file_location(
        "forward_only_dataset_catalog_repair_0089",
        _MIGRATION_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata_runtime_constraint() -> str:
    return str(
        next(
            item
            for item in strategy_versions.constraints
            if isinstance(item, CheckConstraint)
            and item.name == "ck_strategy_versions_v18_runtime_identity"
        ).sqltext
    )


def test_0089_replaces_only_the_empty_v18_runtime_seal() -> None:
    migration = _migration_module()

    assert migration.revision == "0089_v18_catalog_repair"
    assert migration.down_revision == "0088_forward_only_rehab"
    assert migration._PREVIOUS_RUNTIME_BUNDLE_SHA256 == (
        "c495044915133b41bd7f3c13df3b82fd9624e3e892e51ac4deb892eddbf01dfa"
    )
    assert migration._TARGET_RUNTIME_BUNDLE_SHA256 == (
        FORWARD_ONLY_REHABILITATION_TARGET_RUNTIME_BUNDLE_SHA256
    )
    assert _metadata_runtime_constraint() == migration._runtime_identity_constraint(
        migration._TARGET_RUNTIME_BUNDLE_SHA256
    )

    source = _MIGRATION_PATH.read_text(encoding="utf-8")
    assert "pg_advisory_xact_lock" in source
    assert "IN SHARE ROW EXCLUSIVE MODE" in source
    assert "strategy_versions" in source
    assert "backtest_runs" in source
    assert "strategy_incomplete_family_eligibilities" in source
    assert "strategy_forward_only_rehabilitations" in source
    assert "evidence_mode = 'consumed_historical_replay'" in source


class _FakeBind:
    def __init__(
        self,
        *,
        evidence_counts: dict[str, int] | None = None,
        definition: str,
    ) -> None:
        self.evidence_counts = evidence_counts or {}
        self.definition = definition
        self.executed: list[str] = []
        self.scalar_calls = 0

    def execute(self, statement: Any) -> None:
        self.executed.append(str(statement))

    def scalar(self, statement: Any, params: dict[str, str] | None = None) -> Any:
        self.scalar_calls += 1
        sql = str(statement)
        if "pg_get_constraintdef" in sql:
            assert params == {"constraint": "ck_strategy_versions_v18_runtime_identity"}
            return self.definition
        if "FROM quantlab.strategy_versions " in sql:
            key = "strategy_versions"
        elif "FROM quantlab.backtest_runs AS backtest" in sql:
            key = "backtest_runs"
        elif "strategy_incomplete_family_eligibilities" in sql:
            key = "incomplete_family_receipts"
        else:
            assert "strategy_forward_only_rehabilitations" in sql
            key = "rehabilitation_receipts"
        if key in {"strategy_versions", "backtest_runs"}:
            assert params == {
                "recipe": "qlib-rdagent-single-mainline-2026-08-31-v18"
            }
        else:
            assert params is None
        return self.evidence_counts.get(key, 0)


class _FakeOp:
    def __init__(self, bind: _FakeBind) -> None:
        self.bind = bind
        self.actions: list[tuple[str, str, str]] = []

    def get_bind(self) -> _FakeBind:
        return self.bind

    def drop_constraint(self, name: str, table: str, **_kwargs: Any) -> None:
        self.actions.append(("drop", name, table))

    def create_check_constraint(
        self,
        name: str,
        table: str,
        definition: str,
        **_kwargs: Any,
    ) -> None:
        self.actions.append(("create", name, definition))
        self.bind.definition = definition


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
def test_0089_upgrade_and_downgrade_are_symmetric(direction: str) -> None:
    migration = _migration_module()
    expected = (
        migration._PREVIOUS_RUNTIME_BUNDLE_SHA256
        if direction == "upgrade"
        else migration._TARGET_RUNTIME_BUNDLE_SHA256
    )
    bind = _FakeBind(
        definition=migration._runtime_identity_constraint(expected),
    )
    fake_op = _FakeOp(bind)
    migration.op = fake_op

    getattr(migration, direction)()

    assert len(bind.executed) == 2
    assert "pg_advisory_xact_lock" in bind.executed[0]
    assert "LOCK TABLE" in bind.executed[1]
    assert [action[:2] for action in fake_op.actions] == [
        ("drop", migration._RUNTIME_CONSTRAINT),
        ("create", migration._RUNTIME_CONSTRAINT),
    ]
    target = (
        migration._TARGET_RUNTIME_BUNDLE_SHA256
        if direction == "upgrade"
        else migration._PREVIOUS_RUNTIME_BUNDLE_SHA256
    )
    assert target in fake_op.actions[1][2]


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
@pytest.mark.parametrize(
    "evidence_class",
    [
        "strategy_versions",
        "backtest_runs",
        "incomplete_family_receipts",
        "rehabilitation_receipts",
    ],
)
def test_0089_refuses_to_reseal_after_any_v18_evidence(
    direction: str,
    evidence_class: str,
) -> None:
    migration = _migration_module()
    bind = _FakeBind(evidence_counts={evidence_class: 1}, definition="unused")
    fake_op = _FakeOp(bind)
    migration.op = fake_op

    with pytest.raises(RuntimeError, match="governed v18 evidence exists"):
        getattr(migration, direction)()

    assert fake_op.actions == []
    assert bind.scalar_calls == 4
