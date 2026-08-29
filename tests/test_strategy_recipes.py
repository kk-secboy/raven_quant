from __future__ import annotations

import pytest
from pydantic import ValidationError

from quant_platform.api import StrategyConfigRequest
from quant_platform.factor_library import compile_qlib_expression
from quant_platform.portfolio_policy import PortfolioPolicyConfig
from quant_platform.strategy_recipes import (
    RECIPE_VERSION,
    get_strategy_recipe,
    list_strategy_recipes,
)

pytestmark = pytest.mark.no_database


def test_document_strategy_recipes_are_versioned_and_defensive() -> None:
    recipes = list_strategy_recipes()
    assert {item["id"] for item in recipes} == {
        "index_enhancement",
        "short_relative_strength",
        "swing_trend",
        "long_quality_value",
        "full_market_multifactor",
        "minute_mean_reversion",
    }
    assert all(item["version"] == RECIPE_VERSION for item in recipes)

    swing = get_strategy_recipe("swing_trend")
    expected_swing_config = {
        "horizon_profile": "swing_1_6m",
        "factor_source_mode": "qlib_baseline",
        "challenger_weight": 0.0,
        "topk": 20,
        "n_drop": 5,
        "max_position_weight": 0.10,
        "max_daily_turnover": 0.25,
        "max_daily_loss": 0.03,
        "stop_loss": 0.07,
        "profit_taking_mode": "rule_only",
        "take_profit_partial": 2.0,
        "take_profit_partial_fraction": 0.50,
        "take_profit": 5.0,
        "max_drawdown_reduce": 0.10,
        "max_drawdown_liquidate": 0.15,
        "max_drawdown": 0.10,
        "drawdown_reduction_exposure": 0.50,
        "max_industry_weight": 0.30,
        "min_average_daily_amount": 500_000_000,
        "liquidity_lookback_days": 20,
        "require_regulatory_events": True,
        "industry_relative_rank": True,
        "capacity_notional": 5_000_000,
        "max_volume_participation": 0.01,
        "execution_days": 1,
        "execution_method": "open",
        "signal_frequency": "day",
        "signal_period": 63,
        "execution_frequency": "day",
        "rebalance_frequency": "week",
    }
    assert {
        key: swing["config_overrides"][key] for key in expected_swing_config
    } == expected_swing_config
    assert swing["config_overrides"]["strategy_rule_ir"] == swing["strategy_rule_ir"]
    assert swing["config_overrides"]["strategy_rules_sha256"] == swing[
        "strategy_rule_ir"
    ]["rules_sha256"]
    assert "MA20" in swing["rdagent_objective"]
    assert "MA120" in swing["rdagent_objective"]
    assert "行业相对强弱" in swing["rdagent_objective"]
    assert "Wilder ADX(14)" in swing["rdagent_objective"]
    assert [item["id"] for item in swing["factor_baseline"]] == [
        "ma_trend_structure",
        "wilder_adx_14",
        "amount_expansion",
        "bollinger_bandwidth_20",
        "financial_quality",
        "industry_relative_strength_3m",
    ]
    assert sum(item["weight"] for item in swing["factor_baseline"]) == pytest.approx(
        1.0
    )

    multifactor = get_strategy_recipe("full_market_multifactor")
    assert "mf_net_inflow_ratio" in multifactor["rdagent_objective"]
    assert "市场认可度训练标签" in multifactor["rdagent_objective"]

    recipes[0]["config_overrides"]["topk"] = 999
    assert get_strategy_recipe("index_enhancement")["config_overrides"]["topk"] == 100


def test_transparent_short_swing_long_research_baselines_have_fixed_rule_ir() -> None:
    horizons = {
        "short_relative_strength": "short_1_5d",
        "swing_trend": "swing_1_6m",
        "long_quality_value": "long_1_3y",
    }
    for recipe_id, horizon in horizons.items():
        recipe = get_strategy_recipe(recipe_id)
        assert recipe["research_baseline"] is True
        assert recipe["horizon"] == horizon
        assert recipe["strategy_rule_ir"]["horizon"] == horizon
        assert recipe["config_overrides"]["horizon_profile"] == horizon
        assert len(recipe["strategy_rule_ir"]["rules_sha256"]) == 64
        assert recipe["strategy_rule_ir"]["control_order"] == [
            "eligibility_gate",
            "universe_dedup",
            "direction_regime_gate",
            "alpha_rank",
            "entry_timing",
            "exit_state",
            "portfolio_risk",
            "execution_requirement",
        ]

    short = get_strategy_recipe("short_relative_strength")
    assert short["config_overrides"]["signal_period"] == 5
    assert short["config_overrides"]["execution_method"] == "open"
    long = get_strategy_recipe("long_quality_value")
    long_exits = long["strategy_rule_ir"]["slots"]["exit_state"]["components"]
    assert [item["component"] for item in long_exits] == ["thesis_break"]
    assert long["config_overrides"]["rebalance_frequency"] == "month"
    assert long["config_overrides"]["profit_taking_mode"] == "thesis_only"
    assert long["config_overrides"]["industry_relative_rank"] is True
    assert long["config_overrides"]["portfolio_construction"] == "topk_equal_weight"
    for recipe_id in horizons:
        recipe = get_strategy_recipe(recipe_id)
        execution = recipe["config_overrides"]
        assert execution["execution_method"] == "open"
        assert execution["execution_frequency"] == "day"
        for factor in recipe["factor_baseline"]:
            compile_qlib_expression(factor["qlib_expression"])


def test_unknown_strategy_recipe_fails_closed() -> None:
    with pytest.raises(KeyError):
        get_strategy_recipe("missing")


def test_strategy_config_pins_a_supported_recipe_version() -> None:
    swing = get_strategy_recipe("swing_trend")
    config = StrategyConfigRequest.model_validate(
        {
            **swing["config_overrides"],
            "recipe_id": swing["id"],
            "recipe_version": swing["version"],
        }
    )
    assert config.recipe_id == "swing_trend"
    assert config.recipe_version == RECIPE_VERSION
    assert config.factor_source_mode == "qlib_baseline"
    assert config.challenger_weight == pytest.approx(0.0)
    assert config.strategy_rule_ir == swing["strategy_rule_ir"]
    assert config.strategy_rules_sha256 == swing["strategy_rule_ir"]["rules_sha256"]
    assert config.strategy_rule_policy_sha256 == swing["execution_policy"]["policy_sha256"]
    assert config.max_holding_sessions == 126
    assert config.trend_break_lookback_sessions == 20

    for recipe_id in ("short_relative_strength", "long_quality_value"):
        recipe = get_strategy_recipe(recipe_id)
        baseline = StrategyConfigRequest.model_validate(
            {
                **recipe["config_overrides"],
                "recipe_id": recipe["id"],
                "recipe_version": recipe["version"],
            }
        )
        assert baseline.recipe_id == recipe_id
        assert baseline.horizon_profile == recipe["horizon"]

    with pytest.raises(ValidationError, match="recipe version"):
        StrategyConfigRequest.model_validate(
            {"recipe_id": "swing_trend", "recipe_version": "stale"}
        )


def test_explicit_horizon_rule_binding_fails_closed_on_drift() -> None:
    recipe = get_strategy_recipe("short_relative_strength")
    config = {
        **recipe["config_overrides"],
        "recipe_id": recipe["id"],
        "recipe_version": recipe["version"],
    }
    with pytest.raises(ValidationError, match="entry_score_min_percentile"):
        StrategyConfigRequest.model_validate(
            {**config, "entry_score_min_percentile": 0.70}
        )
    with pytest.raises(ValidationError, match="strategy_rules_sha256"):
        StrategyConfigRequest.model_validate(
            {**config, "strategy_rules_sha256": "0" * 64}
        )
    with pytest.raises(ValidationError, match="require strategy_rule_ir"):
        StrategyConfigRequest.model_validate(
            {
                "horizon_profile": "short_1_5d",
                "outer_purge_days": 6,
                "outer_embargo_days": 6,
                "min_backtest_days": 252,
            }
        )


def test_index_enhancement_recipe_uses_benchmark_relative_optimizer() -> None:
    recipe = get_strategy_recipe("index_enhancement")
    config = StrategyConfigRequest.model_validate(
        {
            **recipe["config_overrides"],
            "recipe_id": recipe["id"],
            "recipe_version": recipe["version"],
        }
    )

    assert config.portfolio_construction == "benchmark_relative_qp"
    assert config.optimizer_tracking_penalty > 0
    assert config.execution_days == 3
    assert config.execution_method == "vwap"
    assert config.max_value_deviation == pytest.approx(0.10)
    assert config.signal_frequency == "day"
    assert config.execution_frequency == "5min"
    assert len(config.execution_contract_hash or "") == 64

    with pytest.raises(ValidationError, match=r"topk \* max_position_weight"):
        StrategyConfigRequest.model_validate(
            {
                **recipe["config_overrides"],
                "recipe_id": recipe["id"],
                "recipe_version": recipe["version"],
                "topk": 20,
                "max_position_weight": 0.02,
            }
        )


def test_full_market_recipe_uses_float_cap_industry_neutral_target() -> None:
    recipe = get_strategy_recipe("full_market_multifactor")
    config = StrategyConfigRequest.model_validate(
        {
            **recipe["config_overrides"],
            "recipe_id": recipe["id"],
            "recipe_version": recipe["version"],
        }
    )

    assert recipe["benchmark_role"] == "reporting_only"
    assert recipe["optimization_target"] == "pit_full_market_float_cap"
    assert sum(item["weight"] for item in recipe["factor_baseline"]) == pytest.approx(1.0)
    assert config.portfolio_construction == "industry_neutral_qp"
    assert config.rebalance_frequency == "month"
    assert config.signal_period == 21
    assert config.target_volatility == pytest.approx(0.15)
    assert config.max_position_weight == pytest.approx(0.05)
    assert config.max_industry_weight == pytest.approx(0.15)


def test_minute_mean_reversion_is_long_only_qlib_next_bar_recipe() -> None:
    recipe = get_strategy_recipe("minute_mean_reversion")
    config = StrategyConfigRequest.model_validate(
        {
            **recipe["config_overrides"],
            "recipe_id": recipe["id"],
            "recipe_version": recipe["version"],
        }
    )

    assert recipe["position_side"] == "long_only"
    assert all(item["qlib_expression"] for item in recipe["factor_baseline"])
    assert sum(item["weight"] for item in recipe["factor_baseline"]) == pytest.approx(
        1.0
    )
    assert config.factor_source_mode == "qlib_baseline"
    assert config.challenger_weight == pytest.approx(0.0)
    assert config.signal_frequency == "5min"
    assert config.execution_frequency == "5min"
    assert config.execution_method == "next_bar"
    assert config.execution_lag_bars == 1
    assert config.rebalance_frequency == "bar"
    assert PortfolioPolicyConfig.from_mapping(config.model_dump()).rebalance_frequency == "bar"
