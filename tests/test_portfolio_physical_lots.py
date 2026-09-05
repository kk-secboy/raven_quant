from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from quant_platform.discrete_constraints import ExecutionLotContext, validate_discrete_constraints
from quant_platform.portfolio_policy import PortfolioPolicy, PortfolioPolicyConfig

pytestmark = pytest.mark.no_database


def _context(
    quantities=None, *, mark=10.0, execution=10.0, minimum=100, increment=100, locked=None,
):
    quantities = quantities or {"SZ000001": 0.0}
    return ExecutionLotContext(
        previous_quantities=quantities,
        valuation_prices={key: mark for key in quantities},
        execution_prices={key: execution for key in quantities},
        buy_minimum={key: minimum for key in quantities},
        buy_increment={key: increment for key in quantities},
        sell_increment={key: 1 for key in quantities},
        locked_quantities=locked,
    )


def _decide(context, *, capacity=1_000_000, hold=False, max_position=0.60, **config):
    quantities = pd.Series(context.previous_quantities, dtype=float)
    marks = pd.Series(context.valuation_prices, dtype=float)
    previous = quantities * marks / 100_000.0
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=max_position, max_daily_turnover=1.0,
        **config,
    ))
    return policy.decide(
        pd.Series(1.0, index=quantities.index), previous,
        prices=pd.Series(context.execution_prices), current_prices=marks,
        average_daily_values=pd.Series(capacity / 0.01, index=quantities.index),
        portfolio_value=100_000.0, rebalance_due=not hold,
        execution_lot_context=context,
    )


def test_actual_v37_failure_holds_exact_float32_source_position_without_a_trade():
    # Exact unchanged-v37 isolated replay: SZ002308, signal 2015-10-09,
    # trade 2015-10-12, first IS loop 131. No rounding of the captured inputs.
    nav = 4_993_121.283154555
    previous = 0.04610943491315027
    amount = 122150.10652180569
    factor = 0.1260743886232376
    adjusted_open = 1.878508448600769
    adjusted_close = 1.8848121166229248
    quantity = amount * factor
    raw_mark = adjusted_close / factor
    raw_execution = adjusted_open / factor
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.05, max_daily_turnover=1.0,
        min_rebalance_weight_change=0.01,
    ))
    arguments = dict(
        scores=pd.Series({"SZ002308": 1.0}), previous_weights={"SZ002308": previous},
        average_daily_values=pd.Series({"SZ002308": 692_407_360.0}),
        portfolio_value=nav,
    )
    with pytest.raises(ValueError, match="no feasible whole-lot target"):
        policy.decide(**arguments, prices=pd.Series({"SZ002308": adjusted_open}))
    context = _context({"SZ002308": quantity}, mark=raw_mark, execution=raw_execution)
    decision = policy.decide(
        **arguments, prices=pd.Series({"SZ002308": raw_execution}),
        execution_lot_context=context,
    )
    assert quantity == 15400.000000000002
    assert decision.target_weights == {"SZ002308": previous}
    assert decision.position_state["execution_lot_target_quantities"] == {"SZ002308": quantity}
    assert decision.changes == []
    assert decision.position_state["discrete_constraint_validation"]["status"] == "passed"


@pytest.mark.parametrize("quantity,mark,execution", [
    (1000.0, 10.1, 10.0), (950.0, 10.0, 10.0), (1000.000000000002, 10.0, 10.0),
])
def test_hold_keeps_real_quantity_across_prices_and_legal_odd_lots(quantity, mark, execution):
    decision = _decide(_context({"SZ000001": quantity}, mark=mark, execution=execution), hold=True)
    assert decision.position_state["execution_lot_target_quantities"]["SZ000001"] == quantity
    assert decision.changes == []


@pytest.mark.parametrize("capacity,quantity", [(1999.0, 0), (2000.0, 100), (3999.0, 100)])
def test_buy_capacity_uses_raw_execution_price_while_weights_use_mark(capacity, quantity):
    decision = _decide(_context(mark=10.0, execution=20.0), capacity=capacity)
    assert decision.position_state["execution_lot_target_quantities"]["SZ000001"] == quantity
    assert decision.target_weights.get("SZ000001", 0.0) == quantity * 10 / 100_000
    assert quantity * 20 <= capacity


def test_buy_adds_a_board_lot_to_an_existing_odd_lot():
    decision = _decide(_context({"SZ000001": 950.0}, execution=20.0), capacity=2000.0)
    assert decision.position_state["execution_lot_target_quantities"]["SZ000001"] == 1050.0
    assert decision.changes[0]["action"] == "increase"


@pytest.mark.parametrize("minimum,increment,capacity,expected", [
    (100, 100, 1550.0, 100), (200, 1, 1999.0, 0), (200, 1, 2010.0, 201),
])
def test_actual_board_minimum_and_increment_control_orders(minimum, increment, capacity, expected):
    decision = _decide(_context(minimum=minimum, increment=increment), capacity=capacity)
    assert decision.position_state["execution_lot_target_quantities"]["SZ000001"] == expected


def _sell(context, capacity):
    quantities = pd.Series(context.previous_quantities)
    marks = pd.Series(context.valuation_prices)
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.60, max_daily_turnover=1.0,
    ))
    return policy.decide(
        pd.Series({"SZ000001": 1.0, "SZ000002": 2.0}), quantities * marks / 100_000,
        prices=pd.Series(context.execution_prices),
        average_daily_values=pd.Series({"SZ000001": capacity / 0.01, "SZ000002": 0.0}),
        portfolio_value=100_000, execution_lot_context=context,
    )


@pytest.mark.parametrize("capacity,remaining", [(19.0, 950.0), (20.0, 949.0), (550.0, 923.0)])
def test_sell_respects_capacity_in_real_shares_including_sub_lot_reductions(capacity, remaining):
    decision = _sell(
        _context({"SZ000001": 950.0, "SZ000002": 0.0}, execution=20.0), capacity,
    )
    assert decision.position_state["execution_lot_target_quantities"]["SZ000001"] == remaining
    assert (950 - remaining) * 20 <= capacity


def test_locked_physical_quantity_is_not_sold_or_adjusted_after_validation():
    context = _context(
        {"SZ000001": 950.0, "SZ000002": 0.0},
        locked={"SZ000001": 500.0, "SZ000002": 0.0},
    )
    decision = _sell(context, 100_000.0)
    assert decision.position_state["execution_lot_target_quantities"]["SZ000001"] == 500.0
    assert decision.target_weights["SZ000001"] == 0.05


def test_full_exit_can_remove_only_the_float_remainder_of_a_whole_share_position():
    quantity = 15400.000000000002
    context = _context({"SZ000001": quantity, "SZ000002": 0.0}, mark=1.0, execution=1.0)
    decision = _sell(context, 100_000.0)
    assert decision.position_state["execution_lot_target_quantities"]["SZ000001"] == 0.0
    assert "SZ000001" not in decision.target_weights


@pytest.mark.parametrize("capacity,locked", [(499.0, 0.0), (1000.0, 1000.0)])
def test_position_cap_does_not_override_capacity_or_t1_lock(capacity, locked):
    context = _context({"SZ000001": 1000.0}, locked={"SZ000001": locked})
    with pytest.raises(ValueError, match="no feasible physical-share order"):
        _decide(context, capacity=capacity, max_position=0.095, hold=True)


def test_feasible_position_cap_reduction_keeps_the_existing_hard_gate():
    decision = _decide(_context({"SZ000001": 1000.0}), capacity=500.0, max_position=0.095)
    assert decision.position_state["execution_lot_target_quantities"]["SZ000001"] == 950.0
    assert decision.target_weights["SZ000001"] == 0.095
    assert any(
        item["rule"] == "max_position_weight_risk_reduction" for item in decision.risk_events
    )


def test_final_validator_rejects_capacity_and_invalid_buy_increment_independently():
    context = _context({"SZ000001": 950.0}, execution=20.0)
    report = validate_discrete_constraints(
        {"SZ000001": 0.1}, {"SZ000001": 0.095},
        max_position_weight=0.60, max_daily_turnover=1.0,
        prices=context.execution_prices, portfolio_value=100_000.0, lot_size=100,
        average_daily_values={"SZ000001": 50_000.0}, max_volume_participation=0.01,
        execution_lot_context=context, execution_target_quantities={"SZ000001": 1000.0},
    )
    assert {item["name"] for item in report["violations"]} == {
        "capacity_trade_value", "physical_order_lot",
    }


def test_context_fails_closed_when_quantity_price_nav_binding_is_false():
    context = _context({"SZ000001": 1000.0})
    bad_context = replace(context, previous_quantities={"SZ000001": 1001.0})
    with pytest.raises(ValueError, match="do not match previous weights"):
        PortfolioPolicy(PortfolioPolicyConfig(topk=1, n_drop=0)).decide(
            pd.Series({"SZ000001": 1.0}), {"SZ000001": 0.1},
            prices=pd.Series(context.execution_prices), portfolio_value=100_000,
            execution_lot_context=bad_context,
        )


@pytest.mark.parametrize("constraint", ["turnover", "cash", "industry"])
def test_physical_lot_contract_does_not_bypass_account_constraints(constraint):
    context = _context()
    arguments = {
        "max_daily_turnover": 1.0, "min_cash_weight": 0.0,
        "industries": {"SZ000001": "bank"}, "max_industry_weight": 1.0,
    }
    expected = {
        "turnover": "daily_turnover", "cash": "cash_weight", "industry": "industry_weight",
    }
    if constraint == "turnover":
        arguments["max_daily_turnover"] = 0.01
    elif constraint == "cash":
        arguments["min_cash_weight"] = 0.95
    else:
        arguments["max_industry_weight"] = 0.05
    report = validate_discrete_constraints(
        {"SZ000001": 0.1}, {"SZ000001": 0.0}, max_position_weight=0.60,
        prices=context.execution_prices, portfolio_value=100_000, lot_size=100,
        average_daily_values={"SZ000001": 100_000_000}, max_volume_participation=0.01,
        execution_lot_context=context, execution_target_quantities={"SZ000001": 1000},
        **arguments,
    )
    assert expected[constraint] in {item["name"] for item in report["violations"]}


def test_unheld_unquoted_frozen_symbol_requires_no_fabricated_price():
    context = _context(mark=float("nan"), execution=float("nan"))
    decision = PortfolioPolicy(PortfolioPolicyConfig(topk=1, n_drop=0)).decide(
        pd.Series({"SZ000001": 1.0}), {}, prices=pd.Series(context.execution_prices),
        average_daily_values=pd.Series({"SZ000001": 0.0}), portfolio_value=100_000,
        execution_lot_context=context,
    )
    assert decision.target_weights == {}
    assert decision.position_state["execution_lot_target_quantities"] == {"SZ000001": 0.0}
    assert decision.changes == []


def test_frozen_held_position_retains_exact_quantity_when_execution_quote_is_missing():
    context = _context({"SZ000001": 950.0}, execution=float("nan"))
    decision = _decide(context)
    assert decision.target_weights == {"SZ000001": 0.095}
    assert decision.position_state["execution_lot_target_quantities"] == {"SZ000001": 950.0}


def test_frozen_quantity_cannot_change_behind_an_unchanged_weight():
    context = _context({"SZ000001": 950.0}, execution=float("nan"))
    with pytest.raises(ValueError, match="retain actual physical quantities"):
        validate_discrete_constraints(
            {"SZ000001": 0.095}, {"SZ000001": 0.095}, max_position_weight=0.60,
            max_daily_turnover=1.0, prices=context.execution_prices,
            portfolio_value=100_000, lot_size=100, frozen_instruments={"SZ000001"},
            execution_lot_context=context, execution_target_quantities={"SZ000001": 900},
        )


def test_tiny_weight_does_not_hide_an_invalid_physical_buy_order():
    context = _context()
    report = validate_discrete_constraints(
        {"SZ000001": 1e-11}, {}, max_position_weight=0.60, max_daily_turnover=1.0,
        prices=context.execution_prices, portfolio_value=1e12, lot_size=100,
        average_daily_values={"SZ000001": 1e9}, max_volume_participation=0.01,
        execution_lot_context=context, execution_target_quantities={"SZ000001": 1},
    )
    assert "physical_order_lot" in {item["name"] for item in report["violations"]}


@pytest.mark.parametrize("field,value", [
    ("previous_quantities", {"SZ000001": float("nan")}),
    ("buy_increment", {"SZ000001": 0}),
    ("buy_minimum", {"SZ000001": 0.5}),
    ("sell_increment", {"SZ000001": float("inf")}),
    ("locked_quantities", {"SZ000001": 1001}),
    ("valuation_prices", {"SZ000001": 0}),
    ("execution_prices", {"SZ000001": -1}),
])
def test_invalid_physical_context_is_rejected(field, value):
    context = replace(_context({"SZ000001": 1000.0}), **{field: value})
    with pytest.raises(ValueError):
        context.prepare(
            pd.Series({"SZ000001": 0.1}), portfolio_value=100_000, frozen_instruments=set(),
        )


@pytest.mark.parametrize("capacity,expected", [(1999.9998, 0), (1999.99995, 0), (2000.0, 100)])
def test_physical_order_cannot_round_above_the_continuous_buy_target(capacity, expected):
    decision = _decide(_context(execution=20), capacity=capacity)
    actual = decision.position_state["execution_lot_target_quantities"]["SZ000001"]
    assert actual == expected
    assert actual * 20 <= capacity + 1e-4


@pytest.mark.parametrize("capacity,status", [(1999.9998, "failed"), (1999.99995, "passed")])
def test_final_physical_capacity_retains_existing_monetary_tolerance(capacity, status):
    context = _context(execution=20)
    report = validate_discrete_constraints(
        {"SZ000001": 0.01}, {}, max_position_weight=0.60, max_daily_turnover=1.0,
        prices=context.execution_prices, portfolio_value=100_000, lot_size=100,
        average_daily_values={"SZ000001": capacity / 0.01}, max_volume_participation=0.01,
        execution_lot_context=context, execution_target_quantities={"SZ000001": 100},
    )
    assert report["status"] == status


def test_final_quantity_binding_and_locked_quantity_are_independently_validated():
    context = _context({"SZ000001": 950.0}, locked={"SZ000001": 950.0})
    arguments = dict(
        max_position_weight=0.60, max_daily_turnover=1.0,
        prices=context.execution_prices, portfolio_value=100_000, lot_size=100,
        average_daily_values={"SZ000001": 1e9}, max_volume_participation=0.01,
        execution_lot_context=context,
    )
    with pytest.raises(ValueError, match="do not match final target weights"):
        validate_discrete_constraints(
            {"SZ000001": 0.095}, {"SZ000001": 0.095},
            execution_target_quantities={"SZ000001": 900}, **arguments,
        )
    report = validate_discrete_constraints(
        {"SZ000001": 0.09}, {"SZ000001": 0.095},
        execution_target_quantities={"SZ000001": 900}, **arguments,
    )
    assert "physical_locked_quantity" in {item["name"] for item in report["violations"]}


def test_turnover_rescale_then_risk_ceiling_keeps_the_same_final_quantity_contract():
    context = _context({"SZ000001": 5000.0, "SZ000002": 0.0})
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.60, max_daily_turnover=0.10,
    ))
    decision = policy.decide(
        pd.Series({"SZ000001": 1.0, "SZ000002": 2.0}), {"SZ000001": 0.50},
        prices=pd.Series(context.execution_prices),
        current_prices=pd.Series(context.valuation_prices),
        cost_basis={"SZ000001": 20.0},
        average_daily_values=pd.Series({"SZ000001": 1e9, "SZ000002": 1e9}),
        portfolio_value=100_000, execution_lot_context=context,
    )
    assert decision.target_weights == {"SZ000002": 0.10}
    assert decision.position_state["execution_lot_target_quantities"] == {
        "SZ000001": 0.0, "SZ000002": 1000.0,
    }
    validation = decision.position_state["discrete_constraint_validation"]
    assert validation["status"] == "passed"
    assert validation["risk_turnover_exception"]["status"] == "applied"
    assert validation["risk_turnover_exception"]["no_extra_buys"] is True


def test_position_cap_cannot_be_undone_by_turnover_rescale_or_later_projection():
    context = _context({"SZ000001": 1000.0})
    policy = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.095, max_daily_turnover=0.001,
    ))
    decision = policy.decide(
        pd.Series({"SZ000001": 1.0}), {"SZ000001": 0.10},
        prices=pd.Series(context.execution_prices),
        average_daily_values=pd.Series({"SZ000001": 1e9}),
        portfolio_value=100_000, execution_lot_context=context,
    )
    assert decision.target_weights == {"SZ000001": 0.095}
    assert decision.position_state["execution_lot_target_quantities"] == {"SZ000001": 950.0}
    validation = decision.position_state["discrete_constraint_validation"]
    assert validation["status"] == "passed"
    assert validation["risk_turnover_exception"]["status"] == "applied"


def test_explicit_qlib_full_liquidation_clears_corporate_action_share_equivalents():
    quantity = 10_000 * 0.100037
    context = replace(
        _context({"SZ000001": quantity, "SZ000002": 0.0}), allow_full_liquidation=True,
    )
    decision = _sell(context, 100_000.0)
    assert decision.position_state["execution_lot_target_quantities"]["SZ000001"] == 0.0
    assert "SZ000001" not in decision.target_weights
    validation = decision.position_state["discrete_constraint_validation"]
    assert validation["lot_contract"] == {
        "version": "execution-order-lot-v1", "quantity_basis": "qlib_share_equivalent",
        "rounding_basis": "order_increment", "full_liquidation_allowed": True,
    }
    assert any(item["name"] == "full_position_liquidation" for item in validation["checks"])
    assert validation["status"] == "passed"


@pytest.mark.parametrize("allow_full,capacity,locked,remaining", [
    (False, 100_000.0, 0.0, 0.37),
    (True, 9990.0, 0.0, 1.37),
    (True, 100_000.0, 0.37, 0.37),
])
def test_full_liquidation_permission_never_skips_locks_capacity_or_default_rules(
    allow_full, capacity, locked, remaining,
):
    context = replace(
        _context(
            {"SZ000001": 1000.37, "SZ000002": 0.0},
            locked={"SZ000001": locked, "SZ000002": 0.0},
        ), allow_full_liquidation=allow_full,
    )
    decision = _sell(context, capacity)
    quantity = decision.position_state["execution_lot_target_quantities"]["SZ000001"]
    assert quantity == pytest.approx(remaining, abs=1e-12)
    assert (1000.37 - quantity) * 10 <= capacity + 1e-4
    assert quantity >= locked - 1e-7
    assert not any(
        item["name"] == "full_position_liquidation"
        for item in decision.position_state["discrete_constraint_validation"]["checks"]
    )


@pytest.mark.parametrize("target_quantity,locked,capacity,expected", [
    (100.0, 0.0, 100_000.0, "physical_order_lot"),
    (0.0, 1.0, 100_000.0, "physical_locked_quantity"),
    (0.0, 0.0, 10_000.0, "capacity_trade_value"),
])
def test_final_validator_independently_limits_full_liquidation(
    target_quantity, locked, capacity, expected,
):
    context = replace(
        _context({"SZ000001": 1000.37}, locked={"SZ000001": locked}),
        allow_full_liquidation=True,
    )
    report = validate_discrete_constraints(
        {"SZ000001": target_quantity * 10 / 100_000}, {"SZ000001": 0.100037},
        max_position_weight=0.60, max_daily_turnover=1.0,
        prices=context.execution_prices, portfolio_value=100_000, lot_size=100,
        average_daily_values={"SZ000001": capacity / 0.01}, max_volume_participation=0.01,
        execution_lot_context=context, execution_target_quantities={"SZ000001": target_quantity},
    )
    assert expected in {item["name"] for item in report["violations"]}
