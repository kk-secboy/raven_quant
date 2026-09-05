from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.portfolio_policy import PortfolioPolicy, PortfolioPolicyConfig

pytestmark = pytest.mark.no_database


def _rotation(*, capacity: float, previous_shares: float = 1000, turnover: float = 1.0):
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.60, max_daily_turnover=turnover,
    ))
    return policy.decide(
        pd.Series({"held": 1.0, "new": 2.0}),
        {"held": previous_shares * 10.0 / 100_000.0},
        prices=pd.Series({"held": 10.0, "new": 10.0}),
        average_daily_values=pd.Series({"held": capacity / 0.01, "new": 100_000_000.0}),
        portfolio_value=100_000.0,
    )


@pytest.mark.parametrize("capacity,remaining", [(550.0, 1000), (1000.0, 900), (1550.0, 900)])
def test_rotation_sell_keeps_whole_lot_target_inside_capacity(capacity, remaining):
    decision = _rotation(capacity=capacity)
    assert decision.target_weights["held"] * 100_000 / 10 == pytest.approx(remaining)
    assert (1000 - remaining) * 10 <= capacity
    assert decision.position_state["discrete_constraint_validation"]["status"] == "passed"


@pytest.mark.parametrize("capacity,expected_shares", [(550.0, 0), (1000.0, 100), (1550.0, 100)])
def test_buy_below_one_lot_retains_cash_instead_of_reversing_direction(capacity, expected_shares):
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.50, max_daily_turnover=1.0,
    ))
    decision = policy.decide(
        pd.Series({"new": 1.0}), {}, prices=pd.Series({"new": 10.0}),
        average_daily_values=pd.Series({"new": capacity / 0.01}), portfolio_value=100_000.0,
    )
    assert decision.target_weights.get("new", 0) * 100_000 / 10 == pytest.approx(expected_shares)
    assert all(change["action"] == "increase" for change in decision.changes)
    assert decision.position_state["discrete_constraint_validation"]["status"] == "passed"


def test_non_lot_previous_holding_can_sell_to_a_feasible_total_position_lattice():
    # The frozen validator governs total targets, not odd-lot trade increments.
    decision = _rotation(capacity=550.0, previous_shares=950)
    assert decision.target_weights["held"] * 100_000 / 10 == pytest.approx(900)
    assert decision.position_state["discrete_constraint_validation"]["status"] == "passed"


def test_non_lot_previous_holding_can_buy_to_a_feasible_total_position_lattice():
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.10, max_daily_turnover=1.0,
    ))
    decision = policy.decide(
        pd.Series({"held": 1.0}), {"held": 0.095}, prices=pd.Series({"held": 10.0}),
        average_daily_values=pd.Series({"held": 60_000.0}), portfolio_value=100_000.0,
    )
    assert decision.target_weights["held"] == pytest.approx(0.10)
    assert decision.changes[0]["action"] == "increase"


def test_buy_without_a_position_cap_compatible_lot_cannot_turn_into_a_risk_sell():
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.096, max_daily_turnover=1.0,
    ))
    with pytest.raises(ValueError, match="no feasible whole-lot target"):
        policy.decide(
            pd.Series({"held": 1.0}), {"held": 0.095}, prices=pd.Series({"held": 10.0}),
            average_daily_values=pd.Series({"held": 60_000.0}), portfolio_value=100_000.0,
        )


def test_tiny_buy_cannot_round_above_the_continuous_risk_exposure_target():
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.60, max_daily_turnover=1.0,
    ))
    with pytest.raises(ValueError, match="no feasible whole-lot target"):
        policy.decide(
            pd.Series({"held": 1.0}), {"held": 0.095}, prices=pd.Series({"held": 10.0}),
            average_daily_values=pd.Series({"held": 60_000.0}), portfolio_value=100_000.0,
            risk_exposure=0.16,
        )


@pytest.mark.parametrize("previous,desired,daily_value", [
    (0.1000000005, 0.1000000006, 100_000_000.0),
    (0.0999999995, 0.0999999994, 100.0),
])
def test_capacity_tolerance_cannot_authorize_a_reportable_reverse_trade(
    previous, desired, daily_value,
):
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=desired, max_daily_turnover=1.0,
    ))
    with pytest.raises(ValueError, match="no feasible whole-lot target"):
        policy.decide(
            pd.Series({"held": 1.0}), {"held": previous},
            prices=pd.Series({"held": 10.0}),
            average_daily_values=pd.Series({"held": daily_value}),
            portfolio_value=100_000.0,
        )


@pytest.mark.parametrize("shares,price,nav", [
    (900, 13.7, 4_923_617.21),
    (1100, 19.13, 97_321.83),
    (100, 0.37, 9_135_517.73),
])
@pytest.mark.parametrize("toward", [-np.inf, np.inf])
def test_whole_lot_hold_tolerates_share_weight_roundtrip_noise(shares, price, nav, toward):
    previous = np.nextafter(shares * price / nav, toward)
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.60, max_daily_turnover=1.0,
        rebalance_frequency="month",
    ))
    decision = policy.decide(
        pd.Series({"held": 1.0}), {"held": previous},
        prices=pd.Series({"held": price}),
        average_daily_values=pd.Series({"held": 1_000_000.0}),
        portfolio_value=nav, rebalance_due=False,
    )
    assert decision.target_weights["held"] * nav / price == pytest.approx(shares)
    assert decision.changes == []
    assert decision.position_state["discrete_constraint_validation"]["status"] == "passed"


@pytest.mark.parametrize("direction", ["buy", "sell"])
def test_non_lot_holding_without_capacity_direction_intersection_fails_closed(direction):
    with pytest.raises(ValueError, match="no feasible whole-lot target"):
        if direction == "sell":
            _rotation(capacity=400.0, previous_shares=950)
        else:
            policy = PortfolioPolicy(PortfolioPolicyConfig(
                topk=1, n_drop=0, max_position_weight=0.10, max_daily_turnover=1.0,
            ))
            policy.decide(
                pd.Series({"held": 1.0}), {"held": 0.095},
                prices=pd.Series({"held": 10.0}),
                average_daily_values=pd.Series({"held": 40_000.0}), portfolio_value=100_000.0,
            )


@pytest.mark.parametrize("previous_weight", [0.095, 0.10])
def test_off_cadence_hold_cannot_be_rounded_into_an_unsolicited_sale(previous_weight):
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.60, max_daily_turnover=1.0,
        rebalance_frequency="month",
    ))
    arguments = {
        "prices": pd.Series({"held": 10.0}),
        "average_daily_values": pd.Series({"held": 55_000.0}),
        "portfolio_value": 100_000.0, "rebalance_due": False,
    }
    if previous_weight == 0.095:
        with pytest.raises(ValueError, match="no feasible whole-lot target"):
            policy.decide(pd.Series({"held": 1.0}), {"held": previous_weight}, **arguments)
    else:
        decision = policy.decide(
            pd.Series({"held": 1.0}), {"held": previous_weight}, **arguments,
        )
        assert decision.target_weights == {"held": previous_weight}
        assert decision.changes == []


@pytest.mark.parametrize("price,daily_value", [(np.nan, 100_000.0), (10.0, 0.0)])
def test_unavailable_execution_keeps_existing_odd_holding_frozen(price, daily_value):
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.60, max_daily_turnover=1.0,
    ))
    decision = policy.decide(
        pd.Series({"held": 1.0, "new": 2.0}), {"held": 0.095},
        prices=pd.Series({"held": price, "new": 10.0}),
        average_daily_values=pd.Series({"held": daily_value, "new": 100_000_000.0}),
        portfolio_value=100_000.0,
    )
    assert decision.target_weights["held"] == pytest.approx(0.095)
    assert decision.position_state["frozen_instruments"] == ["held"]
    assert not any(change["instrument"] == "held" for change in decision.changes)
    assert decision.position_state["discrete_constraint_validation"]["status"] == "passed"


@pytest.mark.parametrize("offset,trade_lots", [(-0.001, 0), (0.0, 1), (0.001, 1)])
@pytest.mark.parametrize("direction", ["buy", "sell"])
def test_capacity_boundary_uses_the_same_price_notional_as_final_validation(
    offset, trade_lots, direction,
):
    price, nav, previous = 13.7, 5_000_000.0, 1000 if direction == "sell" else 0
    capacity = price * 100 + offset
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.60, max_daily_turnover=1.0,
    ))
    decision = policy.decide(
        pd.Series({"held": 1.0, "new": 2.0} if direction == "sell" else {"held": 1.0}),
        {"held": previous * price / nav} if previous else {},
        prices=pd.Series({"held": price, "new": 10.0}),
        average_daily_values=pd.Series({"held": capacity / 0.01, "new": 1_000_000_000.0}),
        portfolio_value=nav,
    )
    shares = decision.target_weights.get("held", 0) * nav / price
    expected = previous - trade_lots * 100 if direction == "sell" else trade_lots * 100
    assert shares == pytest.approx(expected)
    assert abs(shares - previous) * price <= capacity + 1e-4
    assert decision.position_state["discrete_constraint_validation"]["status"] == "passed"


def test_turnover_rescaling_cannot_round_a_limited_sale_back_outside_capacity():
    decision = _rotation(capacity=1550.0, turnover=0.015)
    report = decision.position_state["discrete_constraint_validation"]
    assert report["status"] == "passed"
    assert decision.expected_turnover <= 0.015 + 1e-8
    assert decision.target_weights["held"] >= 0.09 - 1e-10


def test_risk_exit_still_respects_capacity_while_preserving_turnover_exception():
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.60, max_daily_turnover=0.001,
    ))
    decision = policy.decide(
        pd.Series({"held": 1.0}), {"held": 0.10}, prices=pd.Series({"held": 10.0}),
        average_daily_values=pd.Series({"held": 155_000.0}), portfolio_value=100_000.0,
        instrument_risk_states={"held": "exit"},
    )
    assert decision.target_weights["held"] == pytest.approx(0.09)
    report = decision.position_state["discrete_constraint_validation"]
    assert report["status"] == "passed"
    assert report["risk_turnover_exception"]["status"] == "applied"


def test_capacity_repair_does_not_waive_an_infeasible_position_cap():
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.05, max_daily_turnover=1.0,
    ))
    with pytest.raises(ValueError, match="post-discretization hard constraint violation"):
        policy.decide(
            pd.Series({"held": 1.0}), {"held": 0.10}, prices=pd.Series({"held": 10.0}),
            average_daily_values=pd.Series({"held": 55_000.0}), portfolio_value=100_000.0,
        )


def test_repeated_rounding_of_a_whole_lot_target_does_not_create_extra_sales():
    nav, price = 4_923_617.21, 13.7
    target = pd.Series({"held": 900 * price / nav})
    previous = pd.Series({"held": 1000 * price / nav})
    arguments = {
        "portfolio_value": nav, "lot_size": 100, "frozen_instruments": set(),
        "previous_weights": previous, "max_weight_changes": pd.Series({"held": 0.1}),
    }
    for _ in range(4):
        target = PortfolioPolicy._round_tradable_lots(
            target, pd.Series({"held": price}), **arguments,
        )
        assert target["held"] * nav / price == pytest.approx(900)
