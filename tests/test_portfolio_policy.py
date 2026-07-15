import sys
import types

import pandas as pd
import pytest

from quant_platform.cost_model import CostModelConfig
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
    assert costs.sell_commission_rate == pytest.approx(0.0015)
    assert costs.max_volume_participation == pytest.approx(0.01)
    assert costs.market_impact_rate(0.01) == pytest.approx(0.0010)
    assert costs.factor_screening_rate(reference_order_value=100_000) == pytest.approx(0.005)
    doubled = costs.doubled()
    assert doubled.buy_commission_rate == pytest.approx(0.0010)
    assert doubled.fixed_slippage_rate == pytest.approx(0.0010)
    assert doubled.impact_at_max_participation == pytest.approx(0.0020)
    assert doubled.min_commission == pytest.approx(10.0)


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
