from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from quant_platform.job_store import (
    ORDER_PLAN_AWAITING_EXECUTION_DATA,
    ORDER_PLAN_EXECUTION_TRADE_DATE_KEY,
    ORDER_PLAN_MATERIALIZATION_STATUS_KEY,
    ORDER_PLAN_MATERIALIZED,
)
from quant_platform.scheduler import SchedulerEngine
from quant_platform.simulation_store import ExecutionDataNotReadyError
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


class _WorkerJobStore:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.finished: list[dict[str, Any]] = []

    def create(
        self,
        kind: str,
        payload: dict[str, Any],
        _log_path: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.created.append({"kind": kind, "payload": payload, **kwargs})
        return {"status": "queued"}

    def finish(self, job_id: str, **kwargs: Any) -> None:
        self.finished.append({"job_id": job_id, **kwargs})


def _order_plan_job() -> dict[str, Any]:
    return {
        "id": "plan-job-1",
        "kind": "simulation_order_plan",
        "payload": {
            "simulation_portfolio_id": "paper-1",
            "actor": "autopilot",
        },
    }


def test_worker_persists_sealed_plan_as_awaiting_instead_of_failing(tmp_path: Path) -> None:
    class Simulations:
        @staticmethod
        def create_batch_from_order_plan(*_args: Any, **_kwargs: Any) -> None:
            raise ExecutionDataNotReadyError(date(2026, 8, 31))

    worker = object.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root=tmp_path)
    worker.simulations = Simulations()
    worker.store = _WorkerJobStore()
    worker._retry_transient_database = lambda operation: operation()
    result = {"order_plan_manifest_sha256": "a" * 64}

    settled = worker._settle_simulation_order_plan(_order_plan_job(), result)

    assert settled[ORDER_PLAN_MATERIALIZATION_STATUS_KEY] == (
        ORDER_PLAN_AWAITING_EXECUTION_DATA
    )
    assert settled[ORDER_PLAN_EXECUTION_TRADE_DATE_KEY] == "2026-08-31"
    assert settled["simulation_batch_id"] is None
    assert worker.store.created == []
    assert worker.store.finished == [
        {
            "job_id": "plan-job-1",
            "exit_code": 0,
            "result": settled,
        }
    ]


class _DeferredJobs:
    def __init__(self) -> None:
        self.completed: list[dict[str, Any]] = []
        self.terminated: list[dict[str, Any]] = []
        self.waiting = [
            {
                "id": "plan-job-1",
                "payload": {
                    "simulation_portfolio_id": "paper-1",
                    "actor": "autopilot",
                },
                "progress": {
                    "order_plan_manifest_sha256": "a" * 64,
                    ORDER_PLAN_MATERIALIZATION_STATUS_KEY: (
                        ORDER_PLAN_AWAITING_EXECUTION_DATA
                    ),
                    ORDER_PLAN_EXECUTION_TRADE_DATE_KEY: "2026-08-31",
                },
            }
        ]

    def awaiting_simulation_order_plans(self, *, limit: int) -> list[dict[str, Any]]:
        assert limit == 100
        return list(self.waiting)

    def complete_simulation_order_plan_materialization(
        self, job_id: str, **kwargs: Any
    ) -> bool:
        self.completed.append({"job_id": job_id, **kwargs})
        self.waiting = []
        return True

    def terminate_simulation_order_plan_materialization(
        self, job_id: str, **kwargs: Any
    ) -> bool:
        self.terminated.append({"job_id": job_id, **kwargs})
        self.waiting = [item for item in self.waiting if item["id"] != job_id]
        return True


def _deferred_scheduler(simulations: Any, tmp_path: Path) -> SchedulerEngine:
    scheduler = object.__new__(SchedulerEngine)
    scheduler.settings = SimpleNamespace(data_root=tmp_path)
    scheduler.jobs = _DeferredJobs()
    scheduler.simulations = simulations
    scheduler.alerts = SimpleNamespace(create=lambda **_kwargs: None)
    return scheduler


def test_scheduler_waits_until_execution_descendant_is_ready(tmp_path: Path) -> None:
    class Simulations:
        calls = 0

        @classmethod
        def create_batch_from_order_plan(cls, *_args: Any, **_kwargs: Any) -> None:
            cls.calls += 1
            raise ExecutionDataNotReadyError(date(2026, 8, 31))

    scheduler = _deferred_scheduler(Simulations(), tmp_path)

    assert scheduler._materialize_awaiting_simulation_order_plans(date(2026, 8, 30)) == 0
    assert Simulations.calls == 0
    assert scheduler._materialize_awaiting_simulation_order_plans(date(2026, 8, 31)) == 0
    assert Simulations.calls == 1
    assert scheduler.jobs.completed == []


def test_scheduler_materializes_exact_batch_once_then_stops_polling(tmp_path: Path) -> None:
    class Simulations:
        calls = 0

        @classmethod
        def create_batch_from_order_plan(
            cls, *_args: Any, **_kwargs: Any
        ) -> tuple[dict[str, Any], bool]:
            cls.calls += 1
            return {"id": "batch-1", "trade_date": "2026-08-31"}, True

    scheduler = _deferred_scheduler(Simulations(), tmp_path)

    assert scheduler._materialize_awaiting_simulation_order_plans(date(2026, 8, 31)) == 1
    assert scheduler.jobs.completed == [
        {
            "job_id": "plan-job-1",
            "order_plan_manifest_sha256": "a" * 64,
            "simulation_batch_id": "batch-1",
            "batch_created": True,
        }
    ]
    assert scheduler._materialize_awaiting_simulation_order_plans(date(2026, 9, 1)) == 0
    assert Simulations.calls == 1


def test_bad_awaiting_plan_is_terminally_isolated_and_next_plan_materializes(
    tmp_path: Path,
) -> None:
    class Simulations:
        @staticmethod
        def create_batch_from_order_plan(
            portfolio_id: str, *_args: Any, **_kwargs: Any
        ) -> tuple[dict[str, Any], bool]:
            if portfolio_id == "paper-bad":
                raise ValueError("Qlib order-plan manifest failed immutable verification")
            return {"id": "batch-good", "trade_date": "2026-08-31"}, True

    scheduler = _deferred_scheduler(Simulations(), tmp_path)
    scheduler.jobs.waiting = [
        {
            "id": "plan-bad",
            "payload": {
                "simulation_portfolio_id": "paper-bad",
                "actor": "autopilot",
            },
            "progress": {
                "order_plan_manifest_sha256": "b" * 64,
                ORDER_PLAN_MATERIALIZATION_STATUS_KEY: (
                    ORDER_PLAN_AWAITING_EXECUTION_DATA
                ),
                ORDER_PLAN_EXECUTION_TRADE_DATE_KEY: "2026-08-31",
            },
        },
        {
            "id": "plan-good",
            "payload": {
                "simulation_portfolio_id": "paper-good",
                "actor": "autopilot",
            },
            "progress": {
                "order_plan_manifest_sha256": "c" * 64,
                ORDER_PLAN_MATERIALIZATION_STATUS_KEY: (
                    ORDER_PLAN_AWAITING_EXECUTION_DATA
                ),
                ORDER_PLAN_EXECUTION_TRADE_DATE_KEY: "2026-08-31",
            },
        },
    ]
    alerts: list[dict[str, Any]] = []
    scheduler.alerts = SimpleNamespace(create=lambda **kwargs: alerts.append(kwargs))

    assert scheduler._materialize_awaiting_simulation_order_plans(date(2026, 8, 31)) == 1
    assert scheduler.jobs.terminated[0]["job_id"] == "plan-bad"
    assert scheduler.jobs.terminated[0]["materialization_status"] == "failed"
    assert scheduler.jobs.completed[0]["job_id"] == "plan-good"
    assert alerts[0]["severity"] == "critical"
    assert alerts[0]["source_id"] == "plan-bad"


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (ValueError("simulation portfolio is not active"), "cancelled"),
        (
            ValueError(
                "Qlib paper order-plan is not bound to an active gated promotion stage"
            ),
            "superseded",
        ),
    ],
)
def test_natural_awaiting_plan_terminal_does_not_raise_or_alert(
    tmp_path: Path,
    error: ValueError,
    expected_status: str,
) -> None:
    class Simulations:
        @staticmethod
        def create_batch_from_order_plan(*_args: Any, **_kwargs: Any) -> None:
            raise error

    scheduler = _deferred_scheduler(Simulations(), tmp_path)
    alerts: list[dict[str, Any]] = []
    scheduler.alerts = SimpleNamespace(create=lambda **kwargs: alerts.append(kwargs))

    assert scheduler._materialize_awaiting_simulation_order_plans(date(2026, 8, 31)) == 0
    assert scheduler.jobs.terminated[0]["materialization_status"] == expected_status
    assert alerts == []


def test_worker_marks_immediate_batch_as_materialized_and_enqueues_replay(
    tmp_path: Path,
) -> None:
    class Simulations:
        @staticmethod
        def create_batch_from_order_plan(
            *_args: Any, **_kwargs: Any
        ) -> tuple[dict[str, Any], bool]:
            return {"id": "batch-1", "trade_date": "2026-08-31"}, True

    worker = object.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root=tmp_path)
    worker.simulations = Simulations()
    worker.store = _WorkerJobStore()
    worker._retry_transient_database = lambda operation: operation()

    settled = worker._settle_simulation_order_plan(
        _order_plan_job(),
        {"order_plan_manifest_sha256": "a" * 64},
    )

    assert settled[ORDER_PLAN_MATERIALIZATION_STATUS_KEY] == ORDER_PLAN_MATERIALIZED
    assert settled["simulation_batch_id"] == "batch-1"
    assert worker.store.created[0]["kind"] == "simulation_replay"
    assert worker.store.created[0]["idempotency_key"] == "simulation-replay:batch-1"


def test_predecessor_waiting_blocks_only_its_own_account(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ScalarRows:
        @staticmethod
        def all() -> list[str]:
            return ["paper-blocked", "paper-ready"]

    class Connection:
        def __enter__(self) -> Connection:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def scalars(_statement: object) -> ScalarRows:
            return ScalarRows()

    class Engine:
        @staticmethod
        def connect() -> Connection:
            return Connection()

    class Jobs:
        engine = Engine()

        def __init__(self) -> None:
            self.created: list[dict[str, Any]] = []

        def create(
            self,
            kind: str,
            payload: dict[str, Any],
            _log_path: Path,
            **kwargs: Any,
        ) -> dict[str, str]:
            self.created.append({"kind": kind, "payload": payload, **kwargs})
            return {"status": "queued"}

    class Simulations:
        @staticmethod
        def get(portfolio_id: str) -> dict[str, Any]:
            return {
                "id": portfolio_id,
                "source_id": f"strategy-{portfolio_id}",
                "daily_dataset": "daily-v1",
                "daily_roll_policy": "latest_compatible",
                "daily_dataset_lineage_id": "b" * 64,
            }

        @staticmethod
        def require_order_plan_predecessor_settled(
            portfolio_id: str, *, signal_date: date
        ) -> dict[str, Any]:
            assert signal_date == date(2026, 8, 31)
            if portfolio_id == "paper-blocked":
                raise ValueError("predecessor batch is awaiting execution data")
            return {"ready": True, "status": "first_plan"}

    scheduler = object.__new__(SchedulerEngine)
    scheduler.settings = SimpleNamespace(data_root=tmp_path)
    scheduler.jobs = Jobs()
    scheduler.simulations = Simulations()
    scheduler.strategies = SimpleNamespace(
        get_version=lambda version_id: {
            "id": version_id,
            "config": {"signal_source": "factor_score"},
        }
    )
    scheduler.model_artifacts = SimpleNamespace()
    scheduler.promotions = SimpleNamespace(
        require_paper_signal=lambda *_args, **_kwargs: {
            "id": "stage-1",
            "opened_at": "2026-08-01T00:00:00+00:00",
        }
    )
    dataset = {
        "name": "daily-v1",
        "ready": True,
        "reproducible": True,
        "provenance": {
            "dataset_identity_sha256": "a" * 64,
            "dataset_lineage_id": "b" * 64,
        },
    }
    monkeypatch.setattr(
        "quant_platform.scheduler.list_qlib_datasets", lambda _root: [dataset]
    )
    monkeypatch.setattr(
        "quant_platform.scheduler.select_qlib_dataset",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        "quant_platform.scheduler.qlib_trading_date_on_or_before",
        lambda *_args, **_kwargs: date(2026, 8, 31),
    )

    assert scheduler._enqueue_due_simulation_order_plans(
        datetime(2026, 8, 31, 10, tzinfo=UTC)
    ) == 1
    assert [
        item["payload"]["simulation_portfolio_id"]
        for item in scheduler.jobs.created
    ] == ["paper-ready"]
