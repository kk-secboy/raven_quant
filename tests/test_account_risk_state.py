from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_platform.account_risk_state import (
    RISK_CAUTION,
    RISK_NORMAL,
    RISK_OFF,
    RISK_SCOPE,
    assess_account_risk,
)
from quant_platform.recommendation_actions import plan_account_actions

pytestmark = pytest.mark.no_database
NOW = datetime(2026, 7, 28, 7, 30, tzinfo=UTC)


def _instrument(*, target: int, filled: int = 0) -> dict:
    return {
        "instrument": "SH600000",
        "target_quantity": target,
        "filled_position": filled,
        "sellable_quantity": filled,
        "open_orders": [],
        "lot_increment": 100,
        "min_lot": 100,
    }


def test_normal_risk_keeps_target_and_scope() -> None:
    assessment = assess_account_risk(max_position_weight=0.10)
    assert assessment["risk_state"] == RISK_NORMAL
    assert assessment["risk_scope"] == RISK_SCOPE
    item = plan_account_actions(
        [_instrument(target=500)],
        now=NOW,
        risk_assessment=assessment,
    )[0]
    assert item["action"] == "BUY"
    assert item["target_quantity"] == 500
    assert item["risk_scope"] == RISK_SCOPE


def test_caution_shrinks_new_risk_and_requires_review() -> None:
    assessment = assess_account_risk(max_position_weight=0.30)
    assert assessment["risk_state"] == RISK_CAUTION
    item = plan_account_actions(
        [_instrument(target=1000, filled=200)],
        now=NOW,
        risk_assessment=assessment,
    )[0]
    assert item["original_target_quantity"] == 1000
    assert item["target_quantity"] == 600
    assert item["action"] == "BUY"
    assert item["execution_state"] == "WAIT"
    assert "manual_review" in item["wait_reason"]


def test_trusted_hard_breach_creates_exit_target() -> None:
    assessment = assess_account_risk(investment_wealth_drawdown=0.25)
    assert assessment["risk_state"] == RISK_OFF
    assert assessment["exit_existing_risk"] is True
    item = plan_account_actions(
        [_instrument(target=800, filled=500)],
        now=NOW,
        risk_assessment=assessment,
    )[0]
    assert item["original_target_quantity"] == 800
    assert item["target_quantity"] == 0
    assert item["action"] == "EXIT"


def test_stale_data_preserves_buy_action_but_hard_blocks_new_risk() -> None:
    assessment = assess_account_risk(data_stale=True, market_data_trusted=False)
    assert assessment["risk_state"] == RISK_OFF
    assert assessment["exit_existing_risk"] is False
    item = plan_account_actions(
        [_instrument(target=500)],
        now=NOW,
        risk_assessment=assessment,
    )[0]
    assert item["action"] == "BUY"
    assert item["target_quantity"] == 500
    assert item["execution_state"] == "BLOCKED"
    assert "data_stale" in item["blocked_reason"]


def test_stale_data_keeps_reduction_target_but_waits() -> None:
    assessment = assess_account_risk(account_state_stale=True, market_data_trusted=False)
    item = plan_account_actions(
        [_instrument(target=200, filled=500)],
        now=NOW,
        risk_assessment=assessment,
    )[0]
    assert item["action"] == "SELL"
    assert item["target_quantity"] == 200
    assert item["execution_state"] == "WAIT"
    assert "account_state_stale" in item["wait_reason"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_position_weight", -0.1),
        ("investment_wealth_drawdown", 1.1),
        ("annualized_volatility", float("nan")),
        ("illiquid_weight", float("inf")),
    ],
)
def test_invalid_metrics_fail_closed(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        assess_account_risk(**{field: value})
