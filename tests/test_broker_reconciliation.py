from datetime import UTC, datetime, timedelta

import pytest

from quant_platform.broker_reconciliation import (
    compare_broker_snapshot,
    validate_broker_snapshot,
)


def _snapshot(now: datetime) -> dict:
    return {
        "status": "ok",
        "environment": "sandbox",
        "account_ref": "SIM-1",
        "as_of": now.isoformat(),
        "cash": 900_000,
        "equity": 1_000_000,
        "positions": [{"instrument": "SH600000", "quantity": 10_000}],
        "orders": [{"client_order_id": "known", "status": "filled"}],
        "trades": [],
    }


def test_reconciliation_detects_stale_unknown_missing_and_balance_differences() -> None:
    now = datetime.now(UTC)
    observed = _snapshot(now - timedelta(minutes=5))
    observed["cash"] = 899_000
    observed["positions"][0]["quantity"] = 9_000
    observed["orders"].append({"client_order_id": "unknown", "status": "open"})
    differences = compare_broker_snapshot(
        expected={
            "cash": 900_000,
            "equity": 1_000_000,
            "positions": [{"instrument": "SH600000", "quantity": 10_000}],
        },
        observed=observed,
        submitted_orders={"known": "broker-known", "missing": "broker-missing"},
        known_client_order_ids={"known", "missing"},
        cash_tolerance=1,
        equity_tolerance=10,
        position_tolerance=0,
        max_snapshot_age_seconds=120,
        now=now,
    )
    assert {item["type"] for item in differences} == {
        "stale_snapshot",
        "cash_mismatch",
        "position_mismatch",
        "unknown_broker_orders",
        "missing_broker_orders",
    }


def test_snapshot_validation_rejects_wrong_account_and_duplicate_positions() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValueError, match="account reference mismatch"):
        validate_broker_snapshot(_snapshot(now), account_ref="OTHER")
    duplicate = _snapshot(now)
    duplicate["positions"].append({"instrument": "SH600000", "quantity": 1})
    with pytest.raises(ValueError, match="unique"):
        validate_broker_snapshot(duplicate, account_ref="SIM-1")
