from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

import quant_platform.strategy_health_collector as collector_module
from quant_platform.strategy_health_collector import (
    COLLECTOR_ACTOR,
    StrategyHealthCollector,
    resolve_latest_batch_binding,
    strategy_health_collection_due,
)

pytestmark = pytest.mark.no_database


def _lane() -> dict:
    return {
        "daily_dataset_identity_sha256": "a" * 64,
        "daily_dataset_lineage_id": "1" * 64,
        "promotion_stage_id": "stage-paper",
    }


def _batch(*, lineage: str = "1" * 64, trade_date: date = date(2026, 8, 28)):
    return SimpleNamespace(
        id="batch-b",
        status="succeeded",
        signal_date=date(2026, 8, 27),
        trade_date=trade_date,
        source_snapshot_id="b" * 64,
        daily_dataset="daily-b",
        daily_dataset_identity_sha256="b" * 64,
        daily_dataset_lineage_id=lineage,
        target_payload_json={
            "governed_order_plan": {
                "format_version": "qlib-order-plan-v1",
                "promotion_stage_id": "stage-paper",
                "formal_backtest_id": "formal-backtest-b",
                "manifest_sha256": "c" * 64,
            }
        },
    )


def test_health_uses_latest_descendant_batch_not_portfolio_anchor() -> None:
    nav = [SimpleNamespace(trade_date=date(2026, 8, 28))]

    binding = resolve_latest_batch_binding(_lane(), nav, [_batch()])

    assert binding["daily_dataset_identity_sha256"] == "b" * 64
    assert binding["daily_dataset_identity_sha256"] != _lane()[
        "daily_dataset_identity_sha256"
    ]
    assert binding["signal_date"] == date(2026, 8, 27)
    assert binding["formal_backtest_id"] == "formal-backtest-b"


@pytest.mark.parametrize(
    ("batch", "match"),
    [
        (_batch(lineage="2" * 64), "lineage"),
        (_batch(trade_date=date(2026, 8, 27)), "unique succeeded"),
    ],
)
def test_health_rejects_wrong_latest_batch_lineage_or_date(batch, match: str) -> None:
    nav = [SimpleNamespace(trade_date=date(2026, 8, 28))]

    with pytest.raises(ValueError, match=match):
        resolve_latest_batch_binding(_lane(), nav, [batch])


def test_new_batch_forces_health_refresh_before_periodic_interval() -> None:
    now = datetime(2026, 8, 28, 8, 0, 15, tzinfo=UTC)
    latest_snapshot = {
        "as_of": now - timedelta(seconds=15),
        "evidence_json": {
            "evidence_trade_date": "2026-08-27",
            "simulation_batch_id": "batch-a",
            "daily_dataset_identity_sha256": "a" * 64,
        },
    }
    latest_batch = {
        "trade_date": date(2026, 8, 28),
        "simulation_batch_id": "batch-b",
        "daily_dataset_identity_sha256": "b" * 64,
    }

    assert strategy_health_collection_due(
        latest_snapshot,
        latest_batch,
        now=now,
        interval_seconds=3600,
    )


def test_collector_transition_ignores_manual_control_snapshots(monkeypatch) -> None:
    collector = StrategyHealthCollector.__new__(StrategyHealthCollector)
    actor_requests: list[str | None] = []
    captured: dict[str, object] = {}

    def latest(_version_id: str, *, actor: str | None = None):
        actor_requests.append(actor)
        return {"health_status": "healthy"}

    collector._latest_snapshot = latest
    collector._collect_lane_evidence = lambda *_args, **_kwargs: {
        "provenance": {"simulation_batch_id": "batch-a"}
    }
    collector.strategies = SimpleNamespace(
        record_health_snapshot=lambda *args, **kwargs: {
            "id": "snapshot-a",
            "args": args,
            "kwargs": kwargs,
        }
    )

    def assess(_horizon, _evidence, *, previous_status):
        captured["previous_status"] = previous_status
        return {
            "health_status": "healthy",
            "criteria": {"watch_feature_drift": 0.2},
            "evidence": {
                "data_integrity_ok": True,
                "ledger_reconciled": True,
                "feature_drift": 0.01,
            },
            "reasons": ["all_activity_health_checks_passed"],
            "windows_trading_days": [20, 60],
        }

    monkeypatch.setattr(collector_module, "assess_strategy_health", assess)

    collector._collect_and_record(
        {
            "strategy_version_id": "version-a",
            "horizon_profile": "short_1_5d",
        },
        datetime(2026, 8, 28, 8, 0, tzinfo=UTC),
        {"simulation_batch_id": "batch-a"},
    )

    assert actor_requests == [COLLECTOR_ACTOR]
    assert captured["previous_status"] == "healthy"
