from __future__ import annotations

from copy import deepcopy
from typing import Any

RECIPE_VERSION = "docx-2026-07-13"

_RECIPES: tuple[dict[str, Any], ...] = (
    {
        "id": "index_enhancement",
        "version": RECIPE_VERSION,
        "name": "沪深300 指数增强",
        "category": "multifactor",
        "description": "行业与风格约束下的价值、动量、质量、成长和低波动组合。",
        "benchmark": "SH000300",
        "universe": "cn_all",
        "rdagent_objective": (
            "为沪深300指数增强研究可解释、低换手且无未来数据泄露的日频因子。"
            "优先覆盖12-1个月动量、1个月反转、EP/BP价值、ROE/ROA质量、"
            "盈利增长和20/60日低波动；因子必须支持行业与市值中性化，"
            "并在独立样本外验证IC、换手、容量和成本后收益。"
        ),
        "factor_guidance": [
            "12-1个月动量与1个月反转",
            "EP/BP价值、ROE/ROA质量与盈利增长",
            "20/60日低波动和流动性",
            "行业、市值中性化与z-score标准化",
        ],
        "config_overrides": {
            "topk": 100,
            "n_drop": 10,
            "max_position_weight": 0.02,
            "max_daily_turnover": 0.15,
            "portfolio_construction": "benchmark_relative_qp",
            "optimizer_alpha_weight": 0.05,
            "optimizer_tracking_penalty": 1.0,
            "optimizer_turnover_penalty": 0.10,
            "max_industry_deviation": 0.03,
            "max_tracking_error": 0.03,
            "max_drawdown": 0.10,
            "capacity_notional": 5_000_000,
            "max_volume_participation": 0.01,
            "min_average_daily_amount": 500_000_000,
        },
        "document_evidence": [
            "指数增强使用价值、动量、质量、成长和低波动因子",
            "行业相对基准偏离不超过3%，单票不超过2%，跟踪误差不超过3%",
            "调仓使用分钟线TWAP/VWAP并限制成交量参与率",
        ],
    },
    {
        "id": "swing_trend",
        "version": RECIPE_VERSION,
        "name": "A股波段趋势",
        "category": "multifactor",
        "description": "趋势、波动、成交额和质量过滤组成的中低频波段策略。",
        "benchmark": "SH000300",
        "universe": "cn_all",
        "rdagent_objective": (
            "研究A股中低频波段策略所需的可解释日频因子，严格禁止未来函数。"
            "核心信号包括MA5/MA10金叉、收盘价高于MA20、MA20高于MA60、"
            "Wilder ADX(14)大于20、5日成交额比大于1.3，以及20日布林带宽度。"
            "同时加入ST、停牌、20日平均成交额低于5亿元、重大违规和高财务风险过滤。"
            "输出必须能由Qlib独立复算并接受滚动样本外、成本、容量和事件压力测试。"
        ),
        "factor_guidance": [
            "MA5/MA10/MA20/MA60趋势结构",
            "标准Wilder ADX(14)趋势强度",
            "5日成交额放量比和20日布林带宽度",
            "ST、停牌、流动性、违规和财务质量过滤",
        ],
        "config_overrides": {
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
        },
        "document_evidence": [
            "MA5/10/20/60、ADX、成交额放量与布林带宽度",
            "7%止损、12%减半、20%清仓、10%减仓和15%清仓",
            "20日平均成交额至少5亿元，单行业不超过30%，下单不超过日成交量1%",
        ],
    },
)


def list_strategy_recipes() -> list[dict[str, Any]]:
    """Return immutable document-derived recipes as independent values."""

    return deepcopy(list(_RECIPES))


def get_strategy_recipe(recipe_id: str) -> dict[str, Any]:
    for recipe in _RECIPES:
        if recipe["id"] == recipe_id:
            return deepcopy(recipe)
    raise KeyError(recipe_id)
