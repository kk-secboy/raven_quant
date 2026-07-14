import numpy as np
import pandas as pd
import pytest

from quant_platform.strategy_backtest import (
    build_governed_signal,
    compose_factor_scores,
    run_event_stress_suite,
    run_robustness_suite,
    run_rolling_suite,
    simulate_long_only_topk,
)


def test_governed_signal_is_the_point_in_time_input_for_qlib() -> None:
    dates = pd.bdate_range("2025-01-02", periods=8)
    instruments = [f"SH{600000 + index:06d}" for index in range(6)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    scores = pd.Series(np.tile(np.arange(6, 0, -1), len(dates)), index=index, dtype=float)
    liquidity = pd.Series(
        np.tile([100_000_000.0, 100_000_000.0, 1e9, 1e9, 1e9, 1e9], len(dates)),
        index=index,
    )
    memberships = pd.DataFrame(
        {
            "instrument": instruments,
            "industry": ["bank", "bank", "tech", "tech", "health", "health"],
            "in_date": pd.Timestamp("2020-01-01"),
            "out_date": pd.NaT,
        }
    )
    signal = build_governed_signal(
        scores,
        topk=2,
        liquidity_amount=liquidity,
        industry_memberships=memberships,
        max_industry_weight=0.5,
        min_average_daily_amount=500_000_000.0,
        liquidity_lookback_days=5,
    )

    assert signal.index.names == ["datetime", "instrument"]
    assert signal.groupby(level="datetime").size().eq(2).all()
    assert not set(signal.index.get_level_values("instrument")) & set(instruments[:2])


def test_multifactor_backtest_respects_position_turnover_and_costs() -> None:
    random = np.random.default_rng(21)
    dates = pd.bdate_range("2024-01-02", periods=100)
    instruments = [f"SH{600000 + index:06d}" for index in range(60)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    signal = random.normal(size=len(index))
    factor_a = pd.Series(signal, index=index)
    factor_b = pd.Series(signal * 0.5 + random.normal(scale=0.5, size=len(index)), index=index)
    scores = compose_factor_scores([(factor_a, 0.7, 1), (factor_b, 0.3, 1)])
    returns = pd.Series(signal * 0.01 + random.normal(scale=0.005, size=len(index)), index=index)
    benchmark = pd.Series(0.0001, index=dates)
    metrics, daily, positions = simulate_long_only_topk(
        scores,
        returns,
        benchmark,
        topk=20,
        n_drop=5,
        max_position_weight=0.05,
        max_daily_turnover=0.20,
        open_cost=0.0005,
        close_cost=0.0015,
    )
    assert metrics["trading_days"] == 100
    assert metrics["max_position_weight"] <= 0.05 + 1e-12
    assert daily["turnover"].max() <= 0.20 + 1e-12
    assert metrics["total_cost"] > 0
    assert metrics["annualized_return"] > metrics["benchmark_annualized_return"]
    assert metrics["sharpe_ratio"] is not None
    assert metrics["sortino_ratio"] is not None
    assert metrics["capacity_fill_ratio"] is None
    assert not positions.empty

    amount = pd.Series(1_000_000_000.0, index=index)
    suite_config = {
        "topk": 20,
        "n_drop": 5,
        "max_position_weight": 0.05,
        "max_daily_turnover": 0.20,
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "capacity_notional": 5_000_000,
        "max_volume_participation": 0.01,
        "max_tracking_error": 1.0,
        "max_drawdown": 1.0,
        "max_turnover": 2.0,
        "min_information_ratio": -5.0,
        "min_sharpe_ratio": -5.0,
        "min_sortino_ratio": -5.0,
        "min_robustness_pass_rate": 0.75,
        "min_capacity_fill_ratio": 0.95,
        "rolling_window_days": 60,
        "rolling_step_days": 20,
        "min_rolling_windows": 3,
        "min_rolling_pass_rate": 0.60,
        "event_window_days": 20,
        "event_count": 3,
        "max_event_underperformance": 0.05,
        "min_event_stress_pass_rate": 0.60,
    }
    robustness = run_robustness_suite(
        scores,
        returns,
        benchmark,
        config=suite_config,
        market_amount=amount,
    )
    assert robustness["scenario_count"] == 4
    assert robustness["pass_rate"] >= 0.75
    assert robustness["passed"] is True

    rolling = run_rolling_suite(
        scores, returns, benchmark, config=suite_config, market_amount=amount
    )
    assert rolling["window_count"] == 3
    assert rolling["passed"] is True
    assert all(item["metrics"]["trading_days"] == 60 for item in rolling["windows"])

    event_stress = run_event_stress_suite(
        scores, returns, benchmark, config=suite_config, market_amount=amount
    )
    assert event_stress["event_count"] == 3
    assert event_stress["passed"] is True


def test_capacity_evidence_limits_unfillable_portfolio_notional() -> None:
    dates = pd.bdate_range("2025-01-02", periods=40)
    instruments = [f"SH{600100 + index:06d}" for index in range(10)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    scores = pd.Series(np.tile(np.arange(10), len(dates)), index=index, dtype=float)
    returns = pd.Series(0.001, index=index)
    benchmark = pd.Series(0.0, index=dates)
    amount = pd.Series(1_000.0, index=index)

    metrics, daily, positions = simulate_long_only_topk(
        scores,
        returns,
        benchmark,
        topk=5,
        n_drop=0,
        max_position_weight=0.20,
        max_daily_turnover=1.0,
        open_cost=0.0,
        close_cost=0.0,
        market_amount=amount,
        portfolio_notional=1_000_000,
        max_volume_participation=0.01,
    )

    assert metrics["capacity_fill_ratio"] < 0.01
    assert metrics["liquidity_observations"] == 40
    assert daily["capacity_fill_ratio"].min() < 0.01
    assert positions["weight"].max() < 0.001


def test_backtest_blocks_suspended_and_price_limit_orders() -> None:
    dates = pd.bdate_range("2025-01-02", periods=40)
    instruments = ["SH600100", "SH600101"]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    scores = pd.Series(np.tile([2.0, 1.0], len(dates)), index=index)
    returns = pd.Series(0.001, index=index)
    benchmark = pd.Series(0.0, index=dates)
    controls = pd.DataFrame(
        {
            "open": 10.0,
            "paused": 0.0,
            "up_limit": 11.0,
            "down_limit": 9.0,
        },
        index=index,
    )
    controls.loc[(dates[0], "SH600100"), "open"] = 11.0
    controls.loc[(dates[1], "SH600100"), "paused"] = 1.0

    metrics, daily, positions = simulate_long_only_topk(
        scores,
        returns,
        benchmark,
        topk=1,
        n_drop=0,
        max_position_weight=1.0,
        max_daily_turnover=1.0,
        open_cost=0.0,
        close_cost=0.0,
        market_controls=controls,
    )

    assert metrics["market_controls_enforced"] is True
    assert metrics["blocked_buy_orders"] == 2
    assert daily.iloc[0]["positions"] == 0
    assert positions["datetime"].min() == dates[2]


def test_backtest_enforces_point_in_time_industry_cap() -> None:
    dates = pd.bdate_range("2025-01-02", periods=40)
    instruments = [f"SH{600200 + index:06d}" for index in range(20)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    scores = pd.Series(np.tile(np.arange(20, 0, -1), len(dates)), index=index, dtype=float)
    returns = pd.Series(0.001, index=index)
    benchmark = pd.Series(0.0, index=dates)
    memberships = pd.DataFrame(
        {
            "instrument": instruments,
            "industry": [f"industry-{index // 5}" for index in range(20)],
            "in_date": pd.Timestamp("2020-01-01"),
            "out_date": pd.NaT,
        }
    )
    benchmark_weights = pd.DataFrame(
        {
            "datetime": pd.Timestamp("2024-12-31"),
            "instrument": instruments,
            "weight": 0.05,
        }
    )
    style_exposures = pd.DataFrame(
        [
            {
                "datetime": timestamp,
                "instrument": instrument,
                "log_market_cap": 10.0 + instrument_index * 0.1,
            }
            for timestamp in dates
            for instrument_index, instrument in enumerate(instruments)
        ]
    )

    metrics, _daily, positions = simulate_long_only_topk(
        scores,
        returns,
        benchmark,
        topk=10,
        n_drop=0,
        max_position_weight=0.10,
        max_daily_turnover=1.0,
        open_cost=0.0,
        close_cost=0.0,
        industry_memberships=memberships,
        max_industry_weight=0.30,
        benchmark_weights=benchmark_weights,
        style_exposures=style_exposures,
        max_industry_deviation=0.05,
        max_size_deviation=2.0,
    )

    exposure = positions.groupby(["datetime", "industry"])["weight"].sum()
    assert metrics["industry_controls_enforced"] is True
    assert metrics["max_industry_weight"] <= 0.30 + 1e-12
    assert exposure.max() <= 0.30 + 1e-12
    assert metrics["benchmark_weights_enforced"] is True
    assert metrics["size_neutralization_enforced"] is True
    assert metrics["max_industry_deviation"] <= 0.05 + 1e-12
    assert metrics["max_size_deviation"] <= 2.0


def test_backtest_filters_illiquid_names_and_charges_minimum_commission() -> None:
    dates = pd.bdate_range("2025-01-02", periods=40)
    instruments = [f"SH{600300 + index:06d}" for index in range(10)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    scores = pd.Series(np.tile(np.arange(10, 0, -1), len(dates)), index=index, dtype=float)
    returns = pd.Series(0.001, index=index)
    benchmark = pd.Series(0.0, index=dates)
    liquidity = pd.Series(
        np.tile([100_000_000.0] * 5 + [1_000_000_000.0] * 5, len(dates)),
        index=index,
    )

    metrics, daily, positions = simulate_long_only_topk(
        scores,
        returns,
        benchmark,
        topk=5,
        n_drop=0,
        max_position_weight=0.20,
        max_daily_turnover=1.0,
        open_cost=0.0001,
        close_cost=0.0011,
        liquidity_amount=liquidity,
        min_average_daily_amount=500_000_000.0,
        liquidity_lookback_days=5,
        min_commission=5.0,
        portfolio_notional=5_000_000.0,
    )

    assert metrics["liquidity_filter_enforced"] is True
    assert metrics["liquidity_excluded_observations"] > 0
    assert metrics["min_commission"] == 5.0
    assert metrics["total_cost"] > 0
    assert set(positions["instrument"]).issubset(set(instruments[5:]))
    assert len(daily) == 36


def _risk_replay_fixture(first_return: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    dates = pd.bdate_range("2025-01-02", periods=40)
    instruments = ["SH600900", "SH600901"]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    scores = pd.Series(np.tile([2.0, 1.0], len(dates)), index=index)
    returns = pd.Series(0.0, index=index)
    returns.loc[(dates[0], "SH600900")] = first_return
    benchmark = pd.Series(0.0, index=dates)
    return scores, returns, benchmark


def test_execution_replay_applies_stop_loss_before_the_next_rebalance() -> None:
    scores, returns, benchmark = _risk_replay_fixture(-0.08)

    metrics, daily, _positions = simulate_long_only_topk(
        scores,
        returns,
        benchmark,
        topk=1,
        n_drop=0,
        max_position_weight=1.0,
        max_daily_turnover=1.0,
        open_cost=0.0,
        close_cost=0.0,
        execution_risk_enabled=True,
    )

    assert metrics["execution_risk_overlay_enforced"] is True
    assert metrics["stop_loss_exit_count"] == 1
    assert daily.iloc[1]["stop_loss_exits"] == 1
    assert daily.iloc[1]["positions"] == 0
    assert metrics["execution_risk_thresholds"]["stop_loss"] == pytest.approx(0.07)


def test_execution_replay_applies_staged_take_profit_without_immediate_rebuy() -> None:
    scores, returns, benchmark = _risk_replay_fixture(0.13)

    metrics, daily, positions = simulate_long_only_topk(
        scores,
        returns,
        benchmark,
        topk=1,
        n_drop=0,
        max_position_weight=1.0,
        max_daily_turnover=1.0,
        open_cost=0.0,
        close_cost=0.0,
        execution_risk_enabled=True,
    )

    second_date = daily.index[1]
    second_weight = positions.loc[
        (positions["datetime"] == second_date) & (positions["instrument"] == "SH600900"),
        "weight",
    ].iloc[0]
    assert metrics["partial_take_profit_exit_count"] == 1
    assert second_weight == pytest.approx(0.50)
    assert positions.loc[
        (positions["datetime"] == daily.index[2])
        & (positions["instrument"] == "SH600900"),
        "weight",
    ].iloc[0] == pytest.approx(0.50)


@pytest.mark.parametrize(
    ("first_return", "expected_state", "metric_name", "second_positions"),
    [
        (-0.11, "reduced", "drawdown_reduction_count", 1),
        (-0.16, "liquidated", "drawdown_liquidation_count", 0),
    ],
)
def test_execution_replay_applies_portfolio_drawdown_circuit_breakers(
    first_return: float,
    expected_state: str,
    metric_name: str,
    second_positions: int,
) -> None:
    scores, returns, benchmark = _risk_replay_fixture(first_return)

    metrics, daily, _positions = simulate_long_only_topk(
        scores,
        returns,
        benchmark,
        topk=1,
        n_drop=0,
        max_position_weight=1.0,
        max_daily_turnover=1.0,
        open_cost=0.0,
        close_cost=0.0,
        execution_risk_enabled=True,
        max_daily_loss=0.50,
        stop_loss=0.50,
    )

    assert metrics[metric_name] == 1
    assert metrics["final_portfolio_risk_state"] == expected_state
    assert daily.iloc[1]["positions"] == second_positions


def test_execution_replay_uses_the_stricter_single_position_exit_on_combined_breach() -> None:
    scores, returns, benchmark = _risk_replay_fixture(-0.11)

    metrics, daily, _positions = simulate_long_only_topk(
        scores,
        returns,
        benchmark,
        topk=1,
        n_drop=0,
        max_position_weight=1.0,
        max_daily_turnover=1.0,
        open_cost=0.0,
        close_cost=0.0,
        execution_risk_enabled=True,
    )

    assert metrics["drawdown_reduction_count"] == 1
    assert metrics["stop_loss_exit_count"] == 1
    assert daily.iloc[1]["positions"] == 0
