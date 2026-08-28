from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from quant_platform.worker import _require_supported_simulation_execution

pytestmark = pytest.mark.no_database

ROOT = Path(__file__).resolve().parents[1]


def test_worker_rejects_every_retired_pair_execution_path() -> None:
    _require_supported_simulation_execution("simulation_replay", execution_adapter="long_only")

    with pytest.raises(ValueError, match="pair simulation execution is retired"):
        _require_supported_simulation_execution("simulation_replay", execution_adapter="pair")
    with pytest.raises(ValueError, match="pair simulation execution is retired"):
        _require_supported_simulation_execution("simulation_replay")
    with pytest.raises(ValueError, match="pair backtest execution is retired"):
        _require_supported_simulation_execution("pair_backtest")


def test_0071_retires_active_pair_accounts_and_every_pending_pair_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = importlib.import_module("migrations.versions.0071_retire_pair_simulation_writes")
    statements: list[str] = []
    monkeypatch.setattr(migration.op, "execute", lambda statement: statements.append(statement))

    migration.upgrade()

    sql = "\n".join(statements).lower()
    assert "job.kind = 'pair_backtest'" in sql
    assert "job.kind = 'simulation_replay'" in sql
    assert "job.status in ('queued', 'running')" in sql
    assert "cancel_requested_at" in sql
    assert "update quantlab.backtest_runs" in sql
    assert "version.strategy_type = 'pair'" in sql
    assert "update quantlab.simulation_batches" in sql
    assert "batch.status in ('queued', 'running')" in sql
    assert "update quantlab.simulation_portfolios" in sql
    assert "execution_adapter = 'pair' and status = 'active'" in sql
    assert "update quantlab.schedules" in sql
    assert "update quantlab.allocation_schedule_groups" in sql
    assert "update quantlab.recommendation_portfolios" in sql
    assert "update quantlab.strategy_allocations" in sql
    assert "version.strategy_type = 'pair'" in sql
    assert "pair allocation execution retired" in sql
    # The jobs table deliberately has no updated_at column.
    job_statement = statements[0].lower()
    assert "updated_at" not in job_statement


def test_pair_allocations_and_schedules_are_fail_closed() -> None:
    allocation_source = (ROOT / "src" / "quant_platform" / "allocation_store.py").read_text(
        encoding="utf-8"
    )
    schedule_source = (ROOT / "src" / "quant_platform" / "schedule_store.py").read_text(
        encoding="utf-8"
    )
    assert "pair allocations are retired" in allocation_source
    assert "_require_long_only_members(connection, allocation_id)" in allocation_source
    assert "pair allocation schedules are retired" in schedule_source
    assert "_require_long_only_allocation(connection, allocation_id)" in schedule_source
