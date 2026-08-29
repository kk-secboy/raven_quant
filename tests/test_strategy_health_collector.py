from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

from quant_platform.strategy_health_collector import (
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
