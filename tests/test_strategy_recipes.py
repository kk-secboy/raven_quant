from __future__ import annotations

import pytest
from pydantic import ValidationError

from quant_platform.api import StrategyConfigRequest
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
        "swing_trend",
        "full_market_multifactor",
        "minute_mean_reversion",
    }
    assert all(item["version"] == RECIPE_VERSION for item in recipes)

    swing = get_strategy_recipe("swing_trend")
    assert swing["config_overrides"] == {
        "factor_source_mode": "qlib_baseline",
        "challenger_weight": 0.0,
        "topk": 20,
        "n_drop": 5,
        "max_position_weight": 0.10,
        "max_daily_turnover": 0.25,
        "max_daily_loss": 0.03,
        "stop_loss": 0.07,
        "take_profit_partial": 0.12,
        "take_profit_partial_fraction": 0.50,
        "take_profit": 0.20,
        "max_drawdown_reduce": 0.10,
        "max_drawdown_liquidate": 0.15,
        "max_drawdown": 0.10,
        "drawdown_reduction_exposure": 0.50,
        "max_industry_weight": 0.30,
        "min_average_daily_amount": 500_000_000,
        "liquidity_lookback_days": 20,
        "require_regulatory_events": True,
        "capacity_notional": 5_000_000,
        "max_volume_participation": 0.01,
        "execution_days": 2,
        "execution_method": "twap",
        "signal_frequency": "day",
        "signal_period": 1,
        "execution_frequency": "5min",
    }
    assert "MA5/MA10" in swing["rdagent_objective"]
    assert "Wilder ADX(14)" in swing["rdagent_objective"]
    assert [item["id"] for item in swing["factor_baseline"]] == [
        "ma_trend_structure",
        "wilder_adx_14",
        "amount_expansion",
        "bollinger_bandwidth_20",
        "financial_quality",
    ]
    assert sum(item["weight"] for item in swing["factor_baseline"]) == pytest.approx(
        1.0
    )

    multifactor = get_strategy_recipe("full_market_multifactor")
    assert "mf_net_inflow_ratio" in multifactor["rdagent_objective"]
    assert "市场认可度训练标签" in multifactor["rdagent_objective"]

    recipes[0]["config_overrides"]["topk"] = 999
    assert get_strategy_recipe("index_enhancement")["config_overrides"]["topk"] == 100


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

    with pytest.raises(ValidationError, match="recipe version"):
        StrategyConfigRequest.model_validate(
            {"recipe_id": "swing_trend", "recipe_version": "stale"}
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
