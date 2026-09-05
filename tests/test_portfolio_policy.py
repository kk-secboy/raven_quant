import sys
import types

import numpy as np
import pandas as pd
import pytest

from quant_platform.cost_model import CostModelConfig, infer_cn_asset_type
from quant_platform.portfolio_policy import (
    PortfolioPolicy,
    PortfolioPolicyConfig,
    is_rebalance_due,
)

pytestmark = pytest.mark.no_database


def test_policy_enforces_position_and_turnover_caps() -> None:
    scores = pd.Series({f"SH{600000 + index:06d}": float(100 - index) for index in range(60)})
    previous = {f"SH{600050 + index:06d}": 0.02 for index in range(10)}
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=50,
            n_drop=5,
            max_position_weight=0.02,
            max_daily_turnover=0.15,
        )
    )
    decision = policy.decide(scores, previous)
    assert max(decision.target_weights.values()) <= 0.02 + 1e-12
    assert decision.expected_turnover <= 0.15 + 1e-12
    assert decision.policy_version == policy.version


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"previous_weights": {"one": -0.10}}, "previous weights"),
        ({"previous_weights": {"one": 0.70, "two": 0.40}}, "previous weights"),
        ({"risk_exposure": float("nan")}, "risk exposure"),
        ({"risk_exposure": 1.10}, "risk exposure"),
        ({"portfolio_drawdown": float("nan")}, "portfolio drawdown"),
        ({"portfolio_drawdown": 0.01}, "portfolio drawdown"),
        ({"daily_return": float("nan")}, "daily return"),
    ],
)
def test_policy_rejects_invalid_financial_state(
    overrides: dict[str, object], message: str
) -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(topk=2, n_drop=0, max_position_weight=0.60)
    )

    with pytest.raises(ValueError, match=message):
        policy.decide(
            pd.Series({"one": 1.0, "two": 0.5}),
            **overrides,
        )


def test_policy_uses_normalized_numeric_financial_state() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(topk=2, n_drop=0, max_position_weight=0.60)
    )

    decision = policy.decide(
        pd.Series({"one": 1.0, "two": 0.5}),
        risk_exposure="0.5",  # type: ignore[arg-type]
        portfolio_drawdown="-0.11",  # type: ignore[arg-type]
        daily_return="-0.06",  # type: ignore[arg-type]
    )

    rules = {event["rule"] for event in decision.risk_events}
    assert rules == {"max_drawdown_reduce", "max_daily_loss"}
    assert "risk exposure reduction" in decision.reasons


def test_cost_model_is_shared_and_doubles_every_variable_cost() -> None:
    costs = CostModelConfig()
    assert costs.buy_commission_rate == pytest.approx(0.0005)
    assert costs.sell_commission_rate == pytest.approx(0.0005)
    assert costs.stock_sell_stamp_duty_rate == pytest.approx(0.0005)
    assert costs.etf_sell_stamp_duty_rate == pytest.approx(0.0)
    assert costs.max_volume_participation == pytest.approx(0.01)
    assert costs.market_impact_rate(0.01) == pytest.approx(0.0010)
    assert costs.factor_screening_rate(reference_order_value=100_000) == pytest.approx(
        0.00452
    )
    doubled = costs.doubled()
    assert doubled.buy_commission_rate == pytest.approx(0.0010)
    assert doubled.stock_sell_stamp_duty_rate == pytest.approx(0.0010)
    assert doubled.fixed_slippage_rate == pytest.approx(0.0010)
    assert doubled.impact_at_max_participation == pytest.approx(0.0020)
    assert doubled.min_commission == pytest.approx(10.0)


def test_cost_schedule_is_asset_and_effective_date_specific() -> None:
    costs = CostModelConfig(effective_from="2025-01-01", effective_to="2025-12-31")
    stock = costs.estimate_breakdown(
        side="sell",
        gross_value=100_000,
        participation=0,
        asset_type="stock",
        trade_date=pd.Timestamp("2025-06-03").date(),
    )
    etf = costs.estimate_breakdown(
        side="sell",
        gross_value=100_000,
        participation=0,
        asset_type="etf",
        trade_date=pd.Timestamp("2025-06-03").date(),
    )
    assert stock["stamp_duty"] == pytest.approx(50.0)
    assert etf["stamp_duty"] == pytest.approx(0.0)
    assert stock["transfer_fee"] == pytest.approx(1.0)
    assert etf["transfer_fee"] == pytest.approx(0.0)
    assert stock["total"] - etf["total"] == pytest.approx(51.0)
    with pytest.raises(ValueError, match="no effective cost schedule"):
        costs.estimate(
            side="buy",
            gross_value=100_000,
            participation=0,
            trade_date=pd.Timestamp("2024-12-31").date(),
        )


def test_chinese_asset_type_classifier_distinguishes_stock_and_etf() -> None:
    assert infer_cn_asset_type("SH600000") == "stock"
    assert infer_cn_asset_type("SZ000001") == "stock"
    assert infer_cn_asset_type("SH510300") == "etf"
    assert infer_cn_asset_type("SZ159919") == "etf"
    with pytest.raises(ValueError, match="cannot classify"):
        infer_cn_asset_type("UNKNOWN")


def test_same_policy_inputs_produce_identical_targets() -> None:
    scores = pd.Series({f"SZ{index:06d}": float(index) for index in range(1, 61)})
    policy = PortfolioPolicy(PortfolioPolicyConfig(topk=50, max_position_weight=0.02))
    qlib_decision = policy.decide(scores, {})
    recommendation_decision = policy.decide(scores, {})
    assert qlib_decision.target_weights == recommendation_decision.target_weights


def test_member_drawdown_gate_allows_exits_but_never_adds_risk() -> None:
    scores = pd.Series({"SH600001": 2.0, "SH600000": 1.0})
    previous = {"SH600000": 0.40}
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=1,
            n_drop=0,
            max_position_weight=0.70,
            max_daily_turnover=1.0,
        )
    )

    decision = policy.decide(
        scores,
        previous,
        allow_new_risk=False,
    )

    assert decision.target_weights.get("SH600001", 0.0) == 0.0
    assert decision.target_weights.get("SH600000", 0.0) <= 0.40
    assert all(change["action"] != "increase" for change in decision.changes)
    assert "member_drawdown_pause_new_risk" in {
        event["rule"] for event in decision.risk_events
    }


def test_policy_applies_liquidity_and_round_lot_constraints() -> None:
    scores = pd.Series({"SH600000": 2.0, "SH600001": 1.0})
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.50,
            max_daily_turnover=1.0,
        )
    )
    decision = policy.decide(
        scores,
        {},
        prices=pd.Series({"SH600000": 10.0, "SH600001": 20.0}),
        average_daily_values=pd.Series({"SH600000": 1_000_000, "SH600001": 1_000_000}),
        portfolio_value=5_000_000,
    )
    assert decision.target_weights["SH600000"] <= 0.002
    assert decision.target_weights["SH600001"] <= 0.002
    for instrument, weight in decision.target_weights.items():
        price = 10.0 if instrument == "SH600000" else 20.0
        shares = weight * 5_000_000 / price
        assert shares % 100 == pytest.approx(0.0)


def test_policy_repairs_discrete_overweight_without_relaxing_position_cap() -> None:
    instruments = [f"stock_{index:02d}" for index in range(20)]
    previous = {instrument: 0.05 for instrument in instruments}
    previous["stock_00"] = 0.051
    previous["stock_19"] = 0.049
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=20,
            n_drop=0,
            max_position_weight=0.05,
            max_daily_turnover=0.0005,
            min_rebalance_weight_change=0.002,
        )
    )

    decision = policy.decide(
        pd.Series(
            {
                instrument: float(len(instruments) - index)
                for index, instrument in enumerate(instruments)
            }
        ),
        previous,
        prices=pd.Series(1.0, index=instruments),
        average_daily_values=pd.Series(1_000_000_000.0, index=instruments),
        portfolio_value=1_000_000.0,
    )

    assert decision.policy_version == "portfolio-policy-v3"
    assert max(decision.target_weights.values()) <= 0.05
    assert decision.target_weights["stock_00"] == pytest.approx(0.05)
    assert all(
        weight * 1_000_000.0 % 100 == pytest.approx(0.0)
        for weight in decision.target_weights.values()
    )
    validation = decision.position_state["discrete_constraint_validation"]
    assert validation["status"] == "passed"
    assert validation["risk_turnover_exception"]["status"] == "applied"
    repair = next(
        event
        for event in decision.risk_events
        if event["rule"] == "post_discretization_max_position_repair"
    )
    assert repair == {
        "rule": "post_discretization_max_position_repair",
        "observed": pytest.approx(0.051),
        "limit": pytest.approx(0.05),
        "action": "reduce_position",
        "instrument": "stock_00",
        "price": pytest.approx(1.0),
        "portfolio_value": pytest.approx(1_000_000.0),
        "lot_size": 100,
        "quantity_before": 51_000,
        "quantity_after": 50_000,
        "quantity_reduced": 1_000,
        "target_weight_after": pytest.approx(0.05),
    }


def test_position_cap_reduction_is_not_reintroduced_by_turnover_scaling() -> None:
    instruments = [f"stock_{index:02d}" for index in range(20)]
    previous = {"stock_00": 0.06}
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=20,
            n_drop=0,
            max_position_weight=0.05,
            max_daily_turnover=0.005,
        )
    )

    decision = policy.decide(
        pd.Series(
            {
                instrument: float(len(instruments) - index)
                for index, instrument in enumerate(instruments)
            }
        ),
        previous,
        prices=pd.Series(1.0, index=instruments),
        average_daily_values=pd.Series(1_000_000_000.0, index=instruments),
        portfolio_value=1_000_000.0,
    )

    assert decision.target_weights["stock_00"] == pytest.approx(0.05)
    assert max(decision.target_weights.values()) <= 0.05
    validation = decision.position_state["discrete_constraint_validation"]
    assert validation["status"] == "passed"
    assert validation["risk_turnover_exception"]["status"] == "applied"
    event = next(
        item
        for item in decision.risk_events
        if item["rule"] == "max_position_weight_risk_reduction"
    )
    assert event == {
        "rule": "max_position_weight_risk_reduction",
        "observed": pytest.approx(0.06),
        "limit": pytest.approx(0.05),
        "action": "reduce_position",
        "instrument": "stock_00",
        "target_weight_after": pytest.approx(0.05),
    }


def test_position_cap_repair_sells_one_lot_when_no_lot_fits_under_cap() -> None:
    decision = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=1,
            n_drop=0,
            max_position_weight=0.05,
            max_daily_turnover=1.0,
            min_rebalance_weight_change=0.002,
        )
    ).decide(
        pd.Series({"stock": 1.0}),
        {"stock": 0.0501},
        prices=pd.Series({"stock": 50.1}),
        average_daily_values=pd.Series({"stock": 1_000_000_000.0}),
        portfolio_value=100_000.0,
    )

    assert decision.target_weights == {}
    validation = decision.position_state["discrete_constraint_validation"]
    assert validation["status"] == "passed"
    assert validation["cash_weight"] == pytest.approx(1.0)
    repair = next(
        event
        for event in decision.risk_events
        if event["rule"] == "post_discretization_max_position_repair"
    )
    assert repair["quantity_before"] == 100
    assert repair["quantity_after"] == 0
    assert repair["quantity_reduced"] == 100


def test_missing_execution_price_freezes_only_that_holding_and_continues_batch() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=1,
            n_drop=0,
            max_position_weight=1.0,
            max_daily_turnover=1.0,
        )
    )

    decision = policy.decide(
        pd.Series({"new": 2.0, "held": 1.0}),
        {"held": 0.50},
        prices=pd.Series({"new": 10.0, "held": np.nan}),
        average_daily_values=pd.Series({"new": 1_000_000_000.0, "held": np.nan}),
        portfolio_value=1_000_000,
    )

    assert decision.target_weights == {
        "new": pytest.approx(0.50),
        "held": pytest.approx(0.50),
    }
    assert decision.position_state["frozen_instruments"] == ["held"]
    assert decision.position_state["deferred_target_weights"] == {"held": 0.0}
    assert decision.position_state["discrete_constraint_validation"]["status"] == "passed"
    assert "execution_evidence_unavailable" in {
        event["rule"] for event in decision.risk_events
    }


def test_missing_price_blocks_new_entry_without_aborting_valid_candidate() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.50,
            max_daily_turnover=1.0,
        )
    )

    decision = policy.decide(
        pd.Series({"missing": 2.0, "valid": 1.0}),
        {},
        prices=pd.Series({"missing": np.nan, "valid": 10.0}),
        average_daily_values=pd.Series(
            {"missing": 1_000_000_000.0, "valid": 1_000_000_000.0}
        ),
        portfolio_value=1_000_000,
    )

    assert "missing" not in decision.target_weights
    assert decision.target_weights == {"valid": pytest.approx(0.50)}
    assert decision.position_state["frozen_instruments"] == ["missing"]


def test_governed_exit_can_exceed_turnover_without_funding_extra_buys() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=1,
            n_drop=0,
            max_position_weight=1.0,
            max_daily_turnover=0.15,
        )
    )

    decision = policy.decide(
        pd.Series({"new": 2.0, "held": 1.0}),
        {"held": 1.0},
        instrument_risk_states={"held": "exit"},
    )

    assert decision.target_weights == {"new": pytest.approx(0.15)}
    assert decision.expected_turnover == pytest.approx(1.0)
    validation = decision.position_state["discrete_constraint_validation"]
    exception = validation["risk_turnover_exception"]
    assert validation["status"] == "passed"
    assert validation["turnover_subject_to_limit"] == pytest.approx(0.15)
    assert exception == {
        "status": "applied",
        "instruments": ["held"],
        "actual_turnover": pytest.approx(1.0),
        "turnover_subject_to_limit": pytest.approx(0.15),
        "normal_daily_turnover_limit": pytest.approx(0.15),
        "gross_increase_weight": pytest.approx(0.15),
        "non_exempt_decrease_weight": pytest.approx(0.0),
        "exempt_risk_decrease_weight": pytest.approx(1.0),
        "no_extra_buys": True,
    }
    event = next(
        item
        for item in decision.risk_events
        if item["rule"] == "risk_driven_turnover_exception"
    )
    assert event["instruments"] == ["held"]
    assert event["gross_increase_weight"] == pytest.approx(0.15)


def test_frozen_inherited_position_breach_is_non_worsening_and_does_not_abort_batch() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=3,
            n_drop=0,
            max_position_weight=0.10,
            max_daily_turnover=1.0,
            min_cash_weight=0.10,
        )
    )

    decision = policy.decide(
        pd.Series({"new_best": 3.0, "new_other": 2.0, "held": 1.0}),
        {"held": 0.20},
        prices=pd.Series({"new_best": 10.0, "new_other": 10.0, "held": np.nan}),
        average_daily_values=pd.Series(
            {"new_best": 100_000_000.0, "new_other": 100_000_000.0, "held": np.nan}
        ),
        portfolio_value=1_000_000.0,
    )

    assert decision.target_weights == {
        "new_best": pytest.approx(0.10),
        "held": pytest.approx(0.20),
    }
    validation = decision.position_state["discrete_constraint_validation"]
    assert validation["status"] == "passed"
    assert validation["frozen_inherited_max_position_exceptions"] == [
        {
            "constraint": "max_position_weight",
            "instrument": "held",
            "target_weight": pytest.approx(0.20),
            "previous_weight": pytest.approx(0.20),
            "configured_limit": pytest.approx(0.10),
            "non_worsening": True,
        }
    ]
    assert "frozen_inherited_max_position_exception" in {
        item["rule"] for item in decision.risk_events
    }


def test_zero_adv_freezes_requested_trade_with_explicit_wait_and_continues_batch() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=1,
            n_drop=0,
            max_position_weight=0.50,
            max_daily_turnover=1.0,
        )
    )

    decision = policy.decide(
        pd.Series({"new": 2.0, "held": 1.0}),
        {"held": 0.20},
        prices=pd.Series({"new": 10.0, "held": 10.0}),
        average_daily_values=pd.Series({"new": 100_000_000.0, "held": 0.0}),
        portfolio_value=1_000_000.0,
    )

    assert decision.target_weights == {
        "new": pytest.approx(0.30),
        "held": pytest.approx(0.20),
    }
    assert decision.position_state["frozen_instruments"] == ["held"]
    assert decision.position_state["deferred_target_weights"] == {"held": 0.0}
    event = next(
        item
        for item in decision.risk_events
        if item["rule"] == "execution_evidence_unavailable"
    )
    assert event["instrument"] == "held"
    assert event["action"] == "wait_existing"
    assert event["unavailable_evidence"] == ["average_daily_value"]


def test_zero_adv_is_valid_only_when_no_trade_is_requested() -> None:
    decision = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=1,
            n_drop=0,
            max_position_weight=1.0,
            max_daily_turnover=1.0,
        )
    ).decide(
        pd.Series({"held": 1.0}),
        {"held": 0.20},
        rebalance_due=False,
        prices=pd.Series({"held": 10.0}),
        average_daily_values=pd.Series({"held": 0.0}),
        portfolio_value=1_000_000.0,
    )

    assert decision.target_weights == {"held": pytest.approx(0.20)}
    assert decision.position_state["frozen_instruments"] == []
    assert "execution_evidence_unavailable" not in {
        item["rule"] for item in decision.risk_events
    }


def test_minimum_holding_blocks_normal_churn_but_hard_risk_can_exit() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=1,
            n_drop=0,
            max_position_weight=1.0,
            max_daily_turnover=1.0,
            holding_min_sessions=21,
        )
    )
    scores = pd.Series({"new": 2.0, "held": 1.0})

    held = policy.decide(
        scores,
        {"held": 1.0},
        holding_age_sessions={"held": 5},
    )
    assert held.target_weights == {"held": pytest.approx(1.0)}
    assert held.position_state["suppressed_changes"][0]["rule"] == (
        "minimum_holding_sessions"
    )

    exited = policy.decide(
        scores,
        {"held": 1.0},
        holding_age_sessions={"held": 5},
        instrument_risk_states={"held": "exit"},
    )
    assert exited.target_weights == {}
    assert "instrument_hard_risk_exit" in {
        event["rule"] for event in exited.risk_events
    }


def test_partial_reductions_and_no_trade_band_are_deterministic() -> None:
    no_trade = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.50,
            max_daily_turnover=1.0,
            min_rebalance_weight_change=0.02,
        )
    ).decide(
        pd.Series({"one": 2.0, "two": 1.0}),
        {"one": 0.49, "two": 0.49},
    )
    assert no_trade.target_weights == {
        "one": pytest.approx(0.49),
        "two": pytest.approx(0.49),
    }

    reduced = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.50,
            max_daily_turnover=1.0,
            score_deterioration_reduce_percentile=0.60,
            score_deterioration_reduce_fraction=0.50,
            valuation_reduce_percentile=0.95,
            valuation_reduce_fraction=0.50,
        )
    ).decide(
        pd.Series({"one": 1.0, "two": 2.0}),
        {"one": 0.50, "two": 0.50},
        valuation_percentiles=pd.Series({"one": 1.0, "two": 0.5}),
    )
    assert reduced.target_weights["one"] == pytest.approx(0.25)
    assert reduced.target_weights["two"] == pytest.approx(0.50)
    assert {event["rule"] for event in reduced.risk_events}.issuperset(
        {"score_deterioration_reduce", "valuation_reduce"}
    )


def test_policy_enforces_industry_weight_cap() -> None:
    scores = pd.Series({f"S{index:02d}": float(20 - index) for index in range(20)})
    industries = pd.Series(
        {
            **{f"S{index:02d}": "bank" for index in range(12)},
            **{f"S{index:02d}": "industry" for index in range(12, 20)},
        }
    )
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=10,
            max_position_weight=0.10,
            max_industry_weight=0.60,
            max_daily_turnover=1.0,
        )
    )
    decision = policy.decide(scores, {}, industries=industries)
    bank_weight = sum(
        weight
        for instrument, weight in decision.target_weights.items()
        if industries[instrument] == "bank"
    )
    assert bank_weight <= 0.60 + 1e-12


def test_topk_industry_shortfall_holds_feasible_candidates_and_cash() -> None:
    scores = pd.Series({f"S{index:02d}": float(20 - index) for index in range(20)})
    industries = pd.Series(
        {
            **{f"S{index:02d}": "bank" for index in range(8)},
            **{f"S{index:02d}": "technology" for index in range(8, 14)},
            **{f"S{index:02d}": "__UNKNOWN__" for index in range(14, 20)},
        }
    )
    decision = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=20,
            n_drop=0,
            max_position_weight=0.05,
            max_industry_weight=0.30,
            max_daily_turnover=1.0,
        )
    ).decide(scores, {}, industries=industries)

    assert set(decision.target_weights) == {
        *(f"S{index:02d}" for index in range(6)),
        *(f"S{index:02d}" for index in range(8, 20)),
    }
    assert all(
        weight == pytest.approx(0.05) for weight in decision.target_weights.values()
    )
    assert sum(decision.target_weights.values()) == pytest.approx(0.90)
    industry_weights = pd.Series(decision.target_weights).groupby(industries).sum()
    assert industry_weights.to_dict() == {
        "__UNKNOWN__": pytest.approx(0.30),
        "bank": pytest.approx(0.30),
        "technology": pytest.approx(0.30),
    }
    validation = decision.position_state["discrete_constraint_validation"]
    assert validation["status"] == "passed"
    assert validation["cash_weight"] == pytest.approx(0.10)
    assert "industry_capacity_cash" in decision.reasons


def test_topk_industry_cap_can_leave_cash_with_no_actionable_candidate() -> None:
    decision = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.50,
            max_industry_weight=0.30,
            max_daily_turnover=1.0,
        )
    ).decide(
        pd.Series({"one": 2.0, "two": 1.0}),
        {},
        industries=pd.Series({"one": "__UNKNOWN__", "two": "__UNKNOWN__"}),
    )

    assert decision.target_weights == {}
    assert decision.changes == []
    assert decision.expected_turnover == pytest.approx(0.0)
    validation = decision.position_state["discrete_constraint_validation"]
    assert validation["status"] == "passed"
    assert validation["cash_weight"] == pytest.approx(1.0)
    assert "industry_capacity_cash" in decision.reasons


def test_topk_eligibility_shortfall_keeps_pre_filter_weight_and_cash() -> None:
    instruments = [f"S{index:02d}" for index in range(20)]
    decision = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=20,
            n_drop=0,
            max_position_weight=0.10,
            max_industry_weight=0.30,
            max_daily_turnover=1.0,
            entry_score_min_percentile=0.80,
        )
    ).decide(
        pd.Series(
            {instrument: float(index + 1) for index, instrument in enumerate(instruments)}
        ),
        {},
        industries=pd.Series(
            {
                **{instrument: "bank" for instrument in instruments[-4:]},
                instruments[-5]: "technology",
                **{instrument: "industry" for instrument in instruments[:-5]},
            }
        ),
    )

    assert len(decision.target_weights) == 5
    assert all(
        weight == pytest.approx(0.05) for weight in decision.target_weights.values()
    )
    assert sum(decision.target_weights.values()) == pytest.approx(0.25)
    assert "eligible_universe_cash" in decision.reasons
    assert decision.position_state["discrete_constraint_validation"]["status"] == "passed"


def test_topk_industry_shortfall_preserves_turnover_and_round_lot_constraints() -> None:
    instruments = [f"S{index:02d}" for index in range(20)]
    scores = pd.Series(
        {instrument: float(20 - index) for index, instrument in enumerate(instruments)}
    )
    industries = pd.Series("bank", index=instruments)
    decision = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=20,
            n_drop=0,
            max_position_weight=0.05,
            max_industry_weight=0.30,
            max_daily_turnover=0.10,
        )
    ).decide(
        scores,
        {"S00": 0.05, "S01": 0.05},
        industries=industries,
        prices=pd.Series(10.0, index=instruments),
        average_daily_values=pd.Series(1_000_000_000.0, index=instruments),
        portfolio_value=1_000_000.0,
    )

    assert set(decision.target_weights) == {f"S{index:02d}" for index in range(6)}
    assert decision.target_weights["S00"] == pytest.approx(0.05)
    assert decision.target_weights["S01"] == pytest.approx(0.05)
    assert all(
        0.0 < decision.target_weights[f"S{index:02d}"] <= 0.025
        for index in range(2, 6)
    )
    assert decision.expected_turnover <= 0.10 + 1e-12
    assert sum(decision.target_weights.values()) <= 0.30 + 1e-12
    assert all(
        weight * 1_000_000.0 / 10.0 % 100 == pytest.approx(0.0)
        for weight in decision.target_weights.values()
    )
    validation = decision.position_state["discrete_constraint_validation"]
    assert validation["status"] == "passed"
    assert validation["cash_weight"] >= 0.80


def test_industry_shortfall_still_fails_closed_for_qp_construction() -> None:
    instruments = [f"S{index:02d}" for index in range(20)]
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=20,
            n_drop=0,
            max_position_weight=0.05,
            max_industry_weight=0.30,
            max_daily_turnover=1.0,
            portfolio_construction="benchmark_relative_qp",
        )
    )

    with pytest.raises(ValueError, match="industry constraints leave too few"):
        policy.decide(
            pd.Series(
                {
                    instrument: float(20 - index)
                    for index, instrument in enumerate(instruments)
                }
            ),
            {},
            industries=pd.Series("bank", index=instruments),
        )


def test_topk_policy_ignores_reporting_only_benchmark_industry_deviation() -> None:
    scores = pd.Series(
        {
            "SH600000": 3.0,
            "SH600001": 2.0,
            "SH600002": 1.0,
        }
    )
    industries = pd.Series(
        {
            "SH600000": "rare",
            "SH600001": "bank",
            "SH600002": "bank",
        }
    )
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.50,
            max_industry_weight=1.0,
            max_industry_deviation=0.10,
            max_daily_turnover=1.0,
            portfolio_construction="topk_equal_weight",
        )
    )

    decision = policy.decide(
        scores,
        {},
        industries=industries,
        benchmark_industry_weights=pd.Series({"bank": 1.0}),
    )

    assert set(decision.target_weights) == {"SH600000", "SH600001"}


def test_retention_buffer_uses_score_order_not_previous_input_order() -> None:
    scores = pd.Series(
        {
            "SH600000": 4.0,
            "SH600001": 3.0,
            "SH600002": 2.0,
            "SZ000001": 1.0,
        }
    )
    industries = pd.Series(
        {
            "SH600000": "bank",
            "SH600001": "bank",
            "SH600002": "bank",
            "SZ000001": "technology",
        }
    )
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=2,
            max_position_weight=0.50,
            max_industry_weight=0.50,
            max_daily_turnover=1.0,
        )
    )

    forward = policy.decide(
        scores,
        {"SH600001": 0.50, "SH600002": 0.50},
        industries=industries,
    )
    reversed_input = policy.decide(
        scores,
        {"SH600002": 0.50, "SH600001": 0.50},
        industries=industries,
    )

    assert forward.target_weights == reversed_input.target_weights
    assert set(forward.target_weights) == {"SH600001", "SZ000001"}


def test_industry_neutral_policy_scales_the_stock_sleeve_to_target_volatility() -> None:
    instruments = pd.Index([f"S{index:02d}" for index in range(10)])
    scores = pd.Series(range(10), index=instruments, dtype=float)
    benchmark = pd.Series(0.10, index=instruments)
    industries = pd.Series(["bank"] * 5 + ["technology"] * 5, index=instruments)
    styles = pd.DataFrame(
        {
            "size": 0.0,
            "value": 0.0,
            "growth": 0.0,
            "volatility": 0.0,
        },
        index=instruments,
    )
    covariance = pd.DataFrame(
        np.eye(len(instruments)) * 0.001,
        index=instruments,
        columns=instruments,
    )
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=10,
            n_drop=0,
            max_position_weight=0.15,
            max_daily_turnover=1.0,
            max_industry_weight=0.60,
            max_industry_deviation=0.10,
            max_size_deviation=0.10,
            max_value_deviation=0.10,
            max_growth_deviation=0.10,
            max_volatility_deviation=0.10,
            max_tracking_error=1.0,
            portfolio_construction="industry_neutral_qp",
            target_volatility=0.10,
        )
    )

    decision = policy.decide(
        scores,
        {},
        industries=industries,
        benchmark_weights=benchmark,
        benchmark_industry_weights=pd.Series({"bank": 0.50, "technology": 0.50}),
        style_exposures=styles,
        benchmark_style_exposure={column: 0.0 for column in styles.columns},
        return_covariance=covariance,
    )

    evidence = decision.position_state["target_volatility"]
    assert evidence["unscaled_annualized_volatility"] > 0.10
    assert evidence["exposure_scale"] == pytest.approx(
        0.10 / evidence["unscaled_annualized_volatility"]
    )
    assert sum(decision.target_weights.values()) == pytest.approx(
        evidence["exposure_scale"]
    )
    assert "target volatility exposure scaling" in decision.reasons


def test_constrained_policy_can_ramp_from_cash_under_turnover_cap() -> None:
    instruments = pd.Index([f"S{index:02d}" for index in range(10)])
    scores = pd.Series(range(10), index=instruments, dtype=float)
    benchmark = pd.Series(0.10, index=instruments)
    industries = pd.Series(["bank"] * 5 + ["technology"] * 5, index=instruments)
    styles = pd.DataFrame(
        {
            "size": 0.0,
            "value": 0.0,
            "growth": 0.0,
            "volatility": 0.0,
        },
        index=instruments,
    )
    covariance = pd.DataFrame(
        np.eye(len(instruments)) * 0.001,
        index=instruments,
        columns=instruments,
    )
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=10,
            n_drop=0,
            max_position_weight=0.15,
            max_daily_turnover=0.15,
            max_industry_weight=0.60,
            max_industry_deviation=0.10,
            max_size_deviation=0.10,
            max_value_deviation=0.10,
            max_growth_deviation=0.10,
            max_volatility_deviation=0.10,
            max_tracking_error=1.0,
            portfolio_construction="benchmark_relative_qp",
        )
    )

    decision = policy.decide(
        scores,
        {},
        industries=industries,
        benchmark_weights=benchmark,
        benchmark_industry_weights=pd.Series(
            {"bank": 0.50, "technology": 0.50}
        ),
        style_exposures=styles,
        benchmark_style_exposure={column: 0.0 for column in styles.columns},
        return_covariance=covariance,
    )

    assert decision.expected_turnover == pytest.approx(0.15)
    assert sum(decision.target_weights.values()) == pytest.approx(0.15)
    evidence = decision.position_state["discrete_constraint_validation"]
    assert evidence["status"] == "passed"
    assert decision.position_state["constraint_benchmark_scale"] == pytest.approx(
        0.15
    )


def test_policy_applies_position_and_portfolio_risk_rules() -> None:
    scores = pd.Series({"winner": 2.0, "loser": 1.0})
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.50,
            max_daily_turnover=1.0,
        )
    )
    decision = policy.decide(
        scores,
        {"winner": 0.50, "loser": 0.50},
        current_prices=pd.Series({"winner": 12.5, "loser": 9.0}),
        cost_basis={"winner": 10.0, "loser": 10.0},
        portfolio_drawdown=-0.11,
    )
    assert "winner" not in decision.target_weights
    assert "loser" not in decision.target_weights
    assert {item["rule"] for item in decision.risk_events} == {
        "max_drawdown_reduce",
        "take_profit",
        "stop_loss",
    }


def test_long_thesis_mode_does_not_apply_mechanical_profit_taking() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=1,
            n_drop=0,
            max_position_weight=1.0,
            max_daily_turnover=1.0,
            profit_taking_mode="thesis_only",
        )
    )

    decision = policy.decide(
        pd.Series({"quality_value": 1.0}),
        {"quality_value": 1.0},
        current_prices=pd.Series({"quality_value": 25.0}),
        cost_basis={"quality_value": 10.0},
    )

    assert decision.target_weights == {"quality_value": pytest.approx(1.0)}
    assert not {"take_profit", "take_profit_partial"}.intersection(
        item["rule"] for item in decision.risk_events
    )


def test_rule_only_mode_does_not_apply_mechanical_profit_taking() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=1,
            n_drop=0,
            max_position_weight=1.0,
            max_daily_turnover=1.0,
            profit_taking_mode="rule_only",
        )
    )

    decision = policy.decide(
        pd.Series({"swing": 1.0}),
        {"swing": 1.0},
        current_prices=pd.Series({"swing": 25.0}),
        cost_basis={"swing": 10.0},
    )

    assert decision.target_weights == {"swing": pytest.approx(1.0)}
    assert not {"take_profit", "take_profit_partial"}.intersection(
        item["rule"] for item in decision.risk_events
    )


def test_policy_execution_plan_reaches_target_on_configured_day() -> None:
    scores = pd.Series({"one": 2.0, "two": 1.0})
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.50,
            max_daily_turnover=1.0,
            execution_days=3,
            execution_method="twap",
        )
    )
    first = policy.decide(scores, {})
    second = policy.decide(
        scores,
        first.target_weights,
        execution_state=first.position_state["execution"],
    )
    third = policy.decide(
        scores,
        second.target_weights,
        execution_state=second.position_state["execution"],
    )
    assert first.target_weights == {"one": pytest.approx(1 / 6), "two": pytest.approx(1 / 6)}
    assert second.target_weights == {"one": pytest.approx(1 / 3), "two": pytest.approx(1 / 3)}
    assert third.target_weights == {"one": pytest.approx(0.5), "two": pytest.approx(0.5)}
    assert third.position_state["execution"] == {}


def test_monthly_rebalance_holds_targets_but_allows_risk_exits() -> None:
    scores = pd.Series({"one": 1.0, "two": 2.0})
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.50,
            max_daily_turnover=1.0,
            rebalance_frequency="month",
        )
    )
    held = policy.decide(scores, {"one": 0.5, "two": 0.5}, rebalance_due=False)
    assert held.target_weights == {"one": pytest.approx(0.5), "two": pytest.approx(0.5)}
    assert held.changes == []
    assert "month rebalance cadence hold" in held.reasons

    stopped = policy.decide(
        scores,
        {"one": 0.5, "two": 0.5},
        rebalance_due=False,
        current_prices=pd.Series({"one": 9.0, "two": 10.0}),
        cost_basis={"one": 10.0, "two": 10.0},
    )
    assert "one" not in stopped.target_weights
    assert stopped.target_weights["two"] == pytest.approx(0.5)
    assert {item["rule"] for item in stopped.risk_events} == {"stop_loss"}


def test_off_cadence_review_changes_only_its_scoped_sleeve_and_hard_risk() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.50,
            max_daily_turnover=1.0,
            rebalance_frequency="month",
        )
    )
    scores = pd.Series(
        {
            "reviewed_candidate": 4.0,
            "unrelated_candidate": 3.0,
            "reviewed_holding": 2.0,
            "unrelated_holding": 1.0,
        }
    )
    previous = {"reviewed_holding": 0.25, "unrelated_holding": 0.25}

    reviewed = policy.decide(
        scores,
        previous,
        rebalance_due=True,
        rebalance_instruments={"reviewed_candidate", "reviewed_holding"},
    )

    assert reviewed.target_weights == {
        "reviewed_candidate": pytest.approx(0.50),
        "unrelated_holding": pytest.approx(0.25),
    }
    assert {change["instrument"] for change in reviewed.changes} == {
        "reviewed_candidate",
        "reviewed_holding",
    }
    assert reviewed.position_state["rebalance_instruments"] == [
        "reviewed_candidate",
        "reviewed_holding",
    ]

    risk_exit = policy.decide(
        scores,
        previous,
        rebalance_due=True,
        rebalance_instruments={"reviewed_candidate", "reviewed_holding"},
        instrument_risk_states={"unrelated_holding": "exit"},
    )

    assert "unrelated_holding" not in risk_exit.target_weights
    assert "instrument_hard_risk_exit" in {
        event["rule"] for event in risk_exit.risk_events
    }


def test_non_rebalance_day_continues_governed_multiday_execution() -> None:
    scores = pd.Series({"one": 2.0, "two": 1.0})
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.50,
            max_daily_turnover=1.0,
            execution_days=3,
            execution_method="vwap",
            rebalance_frequency="month",
        )
    )
    first = policy.decide(scores, {}, rebalance_due=True)
    second = policy.decide(
        scores,
        first.target_weights,
        execution_state=first.position_state["execution"],
        rebalance_due=False,
    )
    assert second.target_weights == {"one": pytest.approx(1 / 3), "two": pytest.approx(1 / 3)}
    assert second.position_state["execution"]["remaining_days"] == 1


def test_rebalance_period_gate_handles_day_week_and_month() -> None:
    assert is_rebalance_due("2026-07-16", None, "month")
    assert not is_rebalance_due("2026-07-16", "2026-07-01", "month")
    assert is_rebalance_due("2026-08-03", "2026-07-31", "month")
    assert not is_rebalance_due("2026-07-17", "2026-07-13", "week")
    assert is_rebalance_due("2026-07-20", "2026-07-17", "week")
    assert is_rebalance_due("2026-07-17", "2026-07-16", "day")
    assert is_rebalance_due(
        "2026-07-17 10:05", "2026-07-17 10:00", "bar"
    )


def test_qlib_adapter_and_recommendation_call_return_identical_targets(monkeypatch) -> None:
    class WeightStrategyBase:
        def __init__(self, signal, *, risk_degree):
            self.signal = signal
            assert risk_degree == 1.0

    module = types.ModuleType("qlib.contrib.strategy.signal_strategy")
    module.WeightStrategyBase = WeightStrategyBase
    monkeypatch.setitem(sys.modules, "qlib", types.ModuleType("qlib"))
    monkeypatch.setitem(sys.modules, "qlib.contrib", types.ModuleType("qlib.contrib"))
    monkeypatch.setitem(
        sys.modules, "qlib.contrib.strategy", types.ModuleType("qlib.contrib.strategy")
    )
    monkeypatch.setitem(sys.modules, "qlib.contrib.strategy.signal_strategy", module)

    from quant_platform.qlib_policy_strategy import create_qlib_policy_strategy

    scores = pd.Series({f"SH{600000 + index:06d}": float(50 - index) for index in range(50)})
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(topk=50, max_position_weight=0.02, max_daily_turnover=1.0)
    )
    metadata = {
        "prices": pd.Series(10.0, index=scores.index),
        "average_daily_values": pd.Series(1_000_000_000.0, index=scores.index),
    }

    class Current:
        def get_stock_list(self):
            return []

        def calculate_value(self):
            return 5_000_000

    class Calendar:
        def get_trade_step(self):
            return 7

        def get_step_time(self, trade_step, shift=0):
            assert trade_step == 7
            assert shift == 1
            return pd.Timestamp("2026-07-09"), pd.Timestamp("2026-07-09")

    metadata_dates = []

    def metadata_provider(when, _instruments):
        metadata_dates.append(when)
        return dict(metadata)

    strategy = create_qlib_policy_strategy(
        signal=scores,
        policy=policy,
        metadata_provider=metadata_provider,
    )
    strategy.trade_calendar = Calendar()
    qlib_targets = strategy.generate_target_weight_position(
        scores, Current(), pd.Timestamp("2026-07-10"), pd.Timestamp("2026-07-10")
    )
    recommendation_targets = policy.decide(
        scores, {}, portfolio_value=5_000_000, **metadata
    ).target_weights
    assert qlib_targets == recommendation_targets
    assert metadata_dates == [pd.Timestamp("2026-07-09")]


def test_qlib_adapter_preserves_holding_age_when_an_exit_is_not_filled(
    monkeypatch,
) -> None:
    class WeightStrategyBase:
        def __init__(self, signal, *, risk_degree):
            self.signal = signal
            assert risk_degree == 1.0

    module = types.ModuleType("qlib.contrib.strategy.signal_strategy")
    module.WeightStrategyBase = WeightStrategyBase
    monkeypatch.setitem(sys.modules, "qlib", types.ModuleType("qlib"))
    monkeypatch.setitem(sys.modules, "qlib.contrib", types.ModuleType("qlib.contrib"))
    monkeypatch.setitem(
        sys.modules, "qlib.contrib.strategy", types.ModuleType("qlib.contrib.strategy")
    )
    monkeypatch.setitem(sys.modules, "qlib.contrib.strategy.signal_strategy", module)

    from quant_platform.qlib_policy_strategy import create_qlib_policy_strategy

    class Current:
        def __init__(self, weights: dict[str, float]) -> None:
            self.weights = weights

        def get_stock_list(self):
            return list(self.weights)

        def get_stock_weight(self, instrument):
            return self.weights[instrument]

        def calculate_value(self):
            return 1_000_000

    class Calendar:
        def __init__(self) -> None:
            self.step = 0

        def get_trade_step(self):
            self.step += 1
            return self.step

        def get_step_time(self, trade_step, shift=0):
            assert shift == 1
            return pd.Timestamp("2026-07-01") + pd.offsets.BDay(trade_step), None

    scores = pd.Series({"SH600000": 1.0})
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=1,
            n_drop=0,
            max_position_weight=1.0,
            max_daily_turnover=1.0,
            max_holding_sessions=1,
        )
    )

    def metadata_provider(_when, instruments):
        index = pd.Index(instruments, dtype="object")
        return {
            "prices": pd.Series(10.0, index=index),
            "current_prices": pd.Series(10.0, index=index),
            "average_daily_values": pd.Series(1_000_000_000.0, index=index),
        }

    strategy = create_qlib_policy_strategy(
        signal=scores,
        policy=policy,
        metadata_provider=metadata_provider,
    )
    strategy.trade_calendar = Calendar()

    assert strategy.generate_target_weight_position(
        scores, Current({}), pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-02")
    ) == {"SH600000": 1.0}
    assert strategy.generate_target_weight_position(
        scores,
        Current({"SH600000": 1.0}),
        pd.Timestamp("2026-07-03"),
        pd.Timestamp("2026-07-03"),
    ) == {"SH600000": 1.0}
    assert strategy.generate_target_weight_position(
        scores,
        Current({"SH600000": 1.0}),
        pd.Timestamp("2026-07-06"),
        pd.Timestamp("2026-07-06"),
    ) == {}

    # Simulate the sell being rejected: Qlib still reports the position on the
    # next session.  The adapter must retain complete age state and retry the
    # governed exit instead of aborting the whole backtest.
    assert strategy.generate_target_weight_position(
        scores,
        Current({"SH600000": 1.0}),
        pd.Timestamp("2026-07-07"),
        pd.Timestamp("2026-07-07"),
    ) == {}
    assert strategy._holding_age_sessions == {"SH600000": 3}


def test_qlib_t1_floor_keeps_stock_bought_today_but_not_etf() -> None:
    from quant_platform.qlib_policy_strategy import apply_t1_target_floor

    result = apply_t1_target_floor(
        {},
        locked_quantities={"SH600000": 1000, "SH510300": 1000},
        current_prices=pd.Series({"SH600000": 10.0, "SH510300": 4.0}),
        portfolio_value=1_000_000,
    )

    assert result["SH600000"] == pytest.approx(0.01)
    assert "SH510300" not in result
