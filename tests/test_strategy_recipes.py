from __future__ import annotations

import pytest
from pydantic import ValidationError

from quant_platform.api import StrategyConfigRequest
from quant_platform.strategy_recipes import (
    RECIPE_VERSION,
    get_strategy_recipe,
    list_strategy_recipes,
)

pytestmark = pytest.mark.no_database


def test_document_strategy_recipes_are_versioned_and_defensive() -> None:
    recipes = list_strategy_recipes()
    assert {item["id"] for item in recipes} == {"index_enhancement", "swing_trend"}
    assert all(item["version"] == RECIPE_VERSION for item in recipes)

    swing = get_strategy_recipe("swing_trend")
    assert swing["config_overrides"] == {
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
        "capacity_notional": 5_000_000,
        "max_volume_participation": 0.01,
    }
    assert "MA5/MA10" in swing["rdagent_objective"]
    assert "Wilder ADX(14)" in swing["rdagent_objective"]

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
