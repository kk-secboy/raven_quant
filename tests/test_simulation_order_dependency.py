from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from quant_platform.scheduler import SchedulerEngine

pytestmark = pytest.mark.no_database


DATASET_IDENTITY = "d" * 64


class _ScalarRows:
    def all(self) -> list[str]:
        return ["paper-1"]


class _Connection:
    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def scalars(self, _statement: object) -> _ScalarRows:
        return _ScalarRows()


class _Engine:
    def connect(self) -> _Connection:
        return _Connection()


class _Jobs:
    def __init__(self) -> None:
        self.engine = _Engine()
        self.created: list[dict[str, object]] = []

    def create(
        self,
        kind: str,
        payload: dict[str, object],
        _log_path: object,
        **kwargs: object,
    ) -> dict[str, str]:
        self.created.append({"kind": kind, "payload": payload, **kwargs})
        return {"status": "queued"}


class _ModelArtifacts:
    def __init__(self, *, ready: bool) -> None:
        self.ready = ready
        self.calls: list[dict[str, object]] = []

    def require_for_inference(
        self,
        strategy_version_id: str,
        *,
        dataset_identity_sha256: str,
        signal_date: date,
        now: datetime,
    ) -> dict[str, str]:
        self.calls.append(
            {
                "strategy_version_id": strategy_version_id,
                "dataset_identity_sha256": dataset_identity_sha256,
                "signal_date": signal_date,
                "now": now,
            }
        )
        if not self.ready:
            raise ValueError("daily inference artifact is not ready")
        return {
            "id": "artifact-1",
            "artifact_sha256": "a" * 64,
            "checkpoint_sha256": "c" * 64,
            "dataset_identity_sha256": DATASET_IDENTITY,
        }


def _engine(*, model_ready: bool) -> SchedulerEngine:
    engine = object.__new__(SchedulerEngine)
    from pathlib import Path

    # pathlib-style log construction is the only settings operation performed.
    engine.settings = SimpleNamespace(data_root=Path("test-data"))
    engine.jobs = _Jobs()
    engine.simulations = SimpleNamespace(
        get=lambda _portfolio_id: {
            "id": "paper-1",
            "source_id": "strategy-1",
            "daily_dataset": "daily-v1",
            "daily_roll_policy": "latest_compatible",
            "daily_dataset_lineage_id": "l" * 64,
        },
        require_order_plan_predecessor_settled=lambda *_args, **_kwargs: {
            "ready": True
        },
    )
    engine.strategies = SimpleNamespace(
        get_version=lambda _version_id: {
            "id": "strategy-1",
            "config": {"signal_source": "model_prediction"},
        }
    )
    engine.model_artifacts = _ModelArtifacts(ready=model_ready)
    engine.promotions = SimpleNamespace(
        require_paper_signal=lambda *_args, **_kwargs: {
            "id": "stage-1",
            "opened_at": "2026-08-26T00:00:00+00:00",
        }
    )
    return engine


def _dataset() -> dict[str, object]:
    return {
        "name": "daily-v1",
        "path": "test-data/qlib/daily-v1",
        "ready": True,
        "reproducible": True,
        "provenance": {
            "dataset_identity_sha256": DATASET_IDENTITY,
            "dataset_lineage_id": "l" * 64,
        },
    }


def test_model_paper_order_waits_for_same_dataset_inference_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(model_ready=False)
    monkeypatch.setattr("quant_platform.scheduler.list_qlib_datasets", lambda _root: [_dataset()])
    monkeypatch.setattr(
        "quant_platform.scheduler.select_qlib_dataset", lambda *_args, **_kwargs: _dataset()
    )
    monkeypatch.setattr(
        "quant_platform.scheduler.qlib_trading_date_on_or_before",
        lambda *_args, **_kwargs: date(2026, 8, 26),
    )

    assert engine._enqueue_due_simulation_order_plans(
        datetime(2026, 8, 26, 10, tzinfo=UTC)
    ) == 0
    assert engine.jobs.created == []
    assert engine.model_artifacts.calls[0]["signal_date"] == date(2026, 8, 26)


def test_model_recommendation_waits_before_creating_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(model_ready=False)

    class Recommendations:
        create_calls = 0

        @staticmethod
        def get(_portfolio_id: str) -> dict[str, object]:
            return {
                "id": "recommendation-1",
                "strategy_version_id": "strategy-1",
                "dataset": "daily-v1",
                "dataset_roll_policy": "latest_compatible",
                "dataset_lineage_id": "l" * 64,
                "latest_snapshot": None,
            }

        def create_snapshot(self, **_kwargs: object) -> tuple[dict, bool]:
            self.create_calls += 1
            return {"id": "snapshot-should-not-exist"}, True

    recommendations = Recommendations()
    engine.recommendations = recommendations
    engine.safe_mode = SimpleNamespace(status=lambda: {"active": False})
    engine.simulations = object()
    engine.promotions = SimpleNamespace(
        require_recommendation_signal=lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "quant_platform.scheduler.select_qlib_dataset",
        lambda *_args, **_kwargs: _dataset(),
    )
    monkeypatch.setattr(
        "quant_platform.scheduler.qlib_trading_date_on_or_before",
        lambda *_args, **_kwargs: date(2026, 8, 26),
    )
    monkeypatch.setattr(
        "quant_platform.scheduler.load_calendar_days",
        lambda *_args, **_kwargs: {date(2026, 8, 26)},
    )
    monkeypatch.setattr(
        "quant_platform.scheduler.evaluate_recommendation_gate",
        lambda *_args, **_kwargs: {"passed": True},
    )

    assert engine._enqueue_due_horizon_recommendations(
        datetime(2026, 8, 26, 10, tzinfo=UTC)
    ) == 0
    assert recommendations.create_calls == 0
    assert engine.jobs.created == []
    assert engine.model_artifacts.calls[0]["signal_date"] == date(2026, 8, 26)


def test_model_paper_order_freezes_ready_artifact_and_dataset_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(model_ready=True)
    monkeypatch.setattr("quant_platform.scheduler.list_qlib_datasets", lambda _root: [_dataset()])
    monkeypatch.setattr(
        "quant_platform.scheduler.select_qlib_dataset", lambda *_args, **_kwargs: _dataset()
    )
    monkeypatch.setattr(
        "quant_platform.scheduler.qlib_trading_date_on_or_before",
        lambda *_args, **_kwargs: date(2026, 8, 26),
    )

    assert engine._enqueue_due_simulation_order_plans(
        datetime(2026, 8, 26, 10, tzinfo=UTC)
    ) == 1
    created = engine.jobs.created[0]
    payload = created["payload"]
    assert payload["dataset_identity_sha256"] == DATASET_IDENTITY
    assert payload["model_artifact_binding"] == {
        "id": "artifact-1",
        "artifact_sha256": "a" * 64,
        "checkpoint_sha256": "c" * 64,
        "dataset_identity_sha256": DATASET_IDENTITY,
    }
    assert payload["factor_materialization_binding"] is None
    assert engine.model_artifacts.calls[0]["signal_date"] == date(2026, 8, 26)
    assert created["idempotency_key"] == (
        "simulation-order-plan-v2:paper-1:2026-08-26:" + DATASET_IDENTITY
    )
