from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from quant_platform.advice_alerts import action_alert_payloads, visible_account_action

pytestmark = pytest.mark.no_database


@pytest.mark.parametrize(
    ("item", "expected"),
    (
        ({"action": "BUY", "filled_position": 0, "target_quantity": 100}, "BUY"),
        ({"action": "BUY", "filled_position": 100, "target_quantity": 200}, "ADD"),
        ({"action": "SELL", "filled_position": 200, "target_quantity": 100}, "REDUCE"),
        ({"action": "SELL", "filled_position": 200, "target_quantity": 0}, "EXIT"),
        ({"action": "HOLD", "filled_position": 200, "target_quantity": 200}, "HOLD"),
    ),
)
def test_visible_account_action_uses_novice_vocabulary(
    item: dict[str, object], expected: str
) -> None:
    assert visible_account_action(item) == expected


def test_action_alerts_include_every_actionable_change_but_skip_hold_no_action() -> None:
    batch = SimpleNamespace(
        id="batch-1",
        signal_date=date(2026, 8, 28),
        trade_date=date(2026, 8, 31),
        target_payload_json={
            "order_plan": {
                "actions": [
                    {
                        "instrument": "SH600000",
                        "action": "BUY",
                        "filled_position": 0,
                        "target_quantity": 500,
                        "projected_position": 0,
                        "execution_state": "ready",
                        "order_plan": [{"op": "new", "quantity": 500}],
                    },
                    {
                        "instrument": "SZ000001",
                        "action": "SELL",
                        "filled_position": 800,
                        "target_quantity": 300,
                        "projected_position": 800,
                        "execution_state": "ready",
                        "order_plan": [{"op": "new", "quantity": 500}],
                    },
                    {
                        "instrument": "SH600519",
                        "action": "EXIT",
                        "filled_position": 100,
                        "target_quantity": 0,
                        "projected_position": 100,
                        "execution_state": "wait",
                        "order_plan": [],
                    },
                    {
                        "instrument": "SH601398",
                        "action": "HOLD",
                        "filled_position": 100,
                        "target_quantity": 100,
                    },
                    {
                        "instrument": "SZ000002",
                        "action": "NO_ACTION",
                        "filled_position": 0,
                        "target_quantity": 0,
                    },
                ]
            }
        },
    )

    payloads = action_alert_payloads(batch)

    assert [(item["instrument"], item["action"]) for item in payloads] == [
        ("SH600000", "BUY"),
        ("SZ000001", "REDUCE"),
        ("SH600519", "EXIT"),
    ]


def test_new_batch_identity_keeps_repeated_hard_action_escalatable() -> None:
    action = {
        "instrument": "SH600519",
        "action": "EXIT",
        "filled_position": 100,
        "target_quantity": 0,
    }
    first = SimpleNamespace(id="batch-1", target_payload_json={"order_plan": {"actions": [action]}})
    second = SimpleNamespace(
        id="batch-2", target_payload_json={"order_plan": {"actions": [action]}}
    )

    assert action_alert_payloads(first) == action_alert_payloads(second)
    assert first.id != second.id
