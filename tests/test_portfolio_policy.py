import sys
import types

import pandas as pd
import pytest

from quant_platform.cost_model import CostModelConfig, infer_cn_asset_type
from quant_platform.portfolio_policy import PortfolioPolicy, PortfolioPolicyConfig

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


def test_cost_model_is_shared_and_doubles_every_variable_cost() -> None:
    costs = CostModelConfig()
    assert costs.buy_commission_rate == pytest.approx(0.0005)
    assert costs.sell_commission_rate == pytest.approx(0.0005)
    assert costs.stock_sell_stamp_duty_rate == pytest.approx(0.0010)
    assert costs.etf_sell_stamp_duty_rate == pytest.approx(0.0)
    assert costs.max_volume_participation == pytest.approx(0.01)
    assert costs.market_impact_rate(0.01) == pytest.approx(0.0010)
    assert costs.factor_screening_rate(reference_order_value=100_000) == pytest.approx(0.005)
    doubled = costs.doubled()
    assert doubled.buy_commission_rate == pytest.approx(0.0010)
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
    assert stock["stamp_duty"] == pytest.approx(100.0)
    assert etf["stamp_duty"] == pytest.approx(0.0)
    assert stock["total"] - etf["total"] == pytest.approx(100.0)
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


def test_qlib_adapter_and_recommendation_call_return_identical_targets(monkeypatch) -> None:
    class WeightStrategyBase:
        def __init__(self, signal):
            self.signal = signal

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
