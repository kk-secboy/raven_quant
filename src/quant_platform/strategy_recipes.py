from __future__ import annotations

from copy import deepcopy
from typing import Any

RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-07-16-v3"

QLIB_SIX_FACTOR_BASELINE: tuple[dict[str, Any], ...] = (
    {"id": "momentum", "weight": 0.20, "qlib_expression": "Ref($close,21)/Ref($close,252)-1"},
    {"id": "reversal", "weight": 0.10, "qlib_expression": "-(Ref($close,1)/Ref($close,21)-1)"},
    {"id": "value", "weight": 0.20, "qlib_expression": "(1/$pe_ttm+1/$pb)/2"},
    {"id": "quality", "weight": 0.20, "qlib_expression": "($fund_roe+$fund_roa)/2"},
    {
        "id": "growth",
        "weight": 0.10,
        "qlib_expression": "($fund_quarter_revenue_yoy+$fund_quarter_profit_yoy)/2",
    },
    {
        "id": "low_volatility",
        "weight": 0.20,
        "qlib_expression": "-Std($close/Ref($close,1)-1,60)",
    },
)

_SWING_UP_MOVE = "$high-Ref($high,1)"
_SWING_DOWN_MOVE = "Ref($low,1)-$low"
_SWING_TRUE_RANGE = (
    "If(Greater($high-$low,Abs($high-Ref($close,1))),"
    "If(Greater($high-$low,Abs($low-Ref($close,1))),"
    "$high-$low,Abs($low-Ref($close,1))),"
    "If(Greater(Abs($high-Ref($close,1)),Abs($low-Ref($close,1))),"
    "Abs($high-Ref($close,1)),Abs($low-Ref($close,1))))"
)
_SWING_PLUS_DM = (
    f"If(Greater({_SWING_UP_MOVE},{_SWING_DOWN_MOVE}),"
    f"If(Greater({_SWING_UP_MOVE},0),{_SWING_UP_MOVE},0),0)"
)
_SWING_MINUS_DM = (
    f"If(Greater({_SWING_DOWN_MOVE},{_SWING_UP_MOVE}),"
    f"If(Greater({_SWING_DOWN_MOVE},0),{_SWING_DOWN_MOVE},0),0)"
)
_SWING_PLUS_DI = (
    f"100*EMA({_SWING_PLUS_DM},27)/(EMA({_SWING_TRUE_RANGE},27)+1e-12)"
)
_SWING_MINUS_DI = (
    f"100*EMA({_SWING_MINUS_DM},27)/(EMA({_SWING_TRUE_RANGE},27)+1e-12)"
)
SWING_QLIB_BASELINE: tuple[dict[str, Any], ...] = (
    {
        "id": "ma_trend_structure",
        "weight": 0.35,
        "qlib_expression": (
            "Greater(Mean($close,5),Mean($close,10))"
            "+Greater($close,Mean($close,20))"
            "+Greater(Mean($close,20),Mean($close,60))"
        ),
    },
    {
        "id": "wilder_adx_14",
        "weight": 0.25,
        # EMA(27) has alpha=1/14, matching Wilder's recursive smoothing.
        "qlib_expression": (
            f"EMA(100*Abs(({_SWING_PLUS_DI})-({_SWING_MINUS_DI}))"
            f"/(({_SWING_PLUS_DI})+({_SWING_MINUS_DI})+1e-12),27)"
        ),
    },
    {
        "id": "amount_expansion",
        "weight": 0.15,
        "qlib_expression": "Mean($amount,5)/(Mean($amount,20)+1e-12)-1",
    },
    {
        "id": "bollinger_bandwidth_20",
        "weight": 0.10,
        "qlib_expression": "2*Std($close,20)/(Mean($close,20)+1e-12)",
    },
    {
        "id": "financial_quality",
        "weight": 0.15,
        "qlib_expression": "($fund_roe+$fund_roa)/2",
    },
)

_RECIPES: tuple[dict[str, Any], ...] = (
    {
        "id": "index_enhancement",
        "version": RECIPE_VERSION,
        "name": "沪深300指数增强",
        "category": "multifactor",
        "description": "以Qlib六因子为基线，在行业和风格约束下构建沪深300增强组合。",
        "benchmark": "SH000300",
        "universe": "cn_all",
        "factor_baseline": QLIB_SIX_FACTOR_BASELINE,
        "preprocessing": ["PIT行业/市值中性化", "缩尾", "z-score"],
        "rdagent_objective": (
            "为沪深300指数增强研究可解释、低换手且无未来数据泄露的挑战者因子。"
            "挑战者必须由Qlib独立复算，并通过样本外、成本、容量和事件压力测试后，"
            "才能增强或替换六因子基线。"
        ),
        "factor_guidance": [
            "12-1个月动量与1个月反转",
            "EP/BP价值、ROE/ROA质量与盈利增长",
            "20/60日低波动和流动性",
            "行业、市值中性化与z-score标准化",
        ],
        "config_overrides": {
            "factor_source_mode": "qlib_baseline",
            "challenger_weight": 0.0,
            "topk": 100,
            "n_drop": 10,
            "max_position_weight": 0.02,
            "max_daily_turnover": 0.15,
            "portfolio_construction": "benchmark_relative_qp",
            "optimizer_alpha_weight": 0.05,
            "optimizer_tracking_penalty": 1.0,
            "optimizer_turnover_penalty": 0.10,
            "max_industry_deviation": 0.03,
            "max_size_deviation": 0.10,
            "max_value_deviation": 0.10,
            "max_growth_deviation": 0.10,
            "max_volatility_deviation": 0.10,
            "max_tracking_error": 0.03,
            "max_drawdown": 0.10,
            "capacity_notional": 5_000_000,
            "max_volume_participation": 0.01,
            "min_average_daily_amount": 500_000_000,
            "execution_days": 3,
            "execution_method": "vwap",
            "signal_frequency": "day",
            "signal_period": 1,
            "execution_frequency": "5min",
            "rebalance_frequency": "day",
        },
        "document_evidence": [
            "六因子权重为20/10/20/20/10/20",
            "单票2%、行业偏离3%、跟踪误差3%、每日换手15%",
            "默认5分钟执行，并在3个交易日内完成VWAP",
        ],
    },
    {
        "id": "swing_trend",
        "version": RECIPE_VERSION,
        "name": "A股波段趋势",
        "category": "multifactor",
        "description": "由趋势、波动、成交额和质量过滤组成的中低频波段卫星策略。",
        "benchmark": "SH000300",
        "universe": "cn_all",
        "factor_baseline": SWING_QLIB_BASELINE,
        "preprocessing": ["PIT可交易/监管过滤", "缩尾", "截面z-score"],
        "rdagent_objective": (
            "研究A股中低频波段策略所需的可解释日频因子，严格禁止未来函数。"
            "核心信号包括MA5/MA10金叉、收盘价高于MA20、MA20高于MA60、"
            "Wilder ADX(14)大于20、5日成交额放量和20日布林带宽度。"
            "输出必须由Qlib独立复算并接受滚动样本外、成本、容量和事件压力测试。"
        ),
        "factor_guidance": [
            "MA5/MA10/MA20/MA60趋势结构",
            "标准Wilder ADX(14)趋势强度",
            "5日成交额放量比和20日布林带宽度",
            "ST、停牌、流动性、违规和财务质量过滤",
        ],
        "config_overrides": {
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
        },
        "document_evidence": [
            "MA5/10/20/60、ADX、成交额放量与布林带宽度",
            "7%止损、12%减半、20%清仓、10%降仓和15%清仓",
            "20日平均成交额至少5亿元，成交参与率不超过1%",
        ],
    },
    {
        "id": "full_market_multifactor",
        "version": RECIPE_VERSION,
        "name": "全市场行业中性多因子",
        "category": "multifactor",
        "description": "PIT全A股股票池、流通市值行业目标和风格中性的月频核心组合。",
        "benchmark": "SH000300",
        "benchmark_role": "reporting_only",
        "optimization_target": "pit_full_market_float_cap",
        "universe": "cn_all",
        "universe_policy": "pit_all_tradable_ashares",
        "factor_baseline": QLIB_SIX_FACTOR_BASELINE,
        "preprocessing": ["PIT行业/市值中性化", "缩尾", "z-score"],
        "rdagent_objective": (
            "在PIT全A股可交易股票池上研究行业中性多因子挑战者。"
            "候选必须使用同一六因子基线和公告日可得财务数据，不得把沪深300权重"
            "作为优化目标；行业目标按当期全市场流通市值计算，并通过Qlib滚动样本外、"
            "成本、容量和事件压力测试。"
        ),
        "factor_guidance": [
            "动量20%、反转10%、价值20%、质量20%、成长10%、低波动20%",
            "PIT行业和市值中性化、缩尾与z-score",
            "全市场流通市值行业目标和风格暴露中性",
            "月度调仓、容量和换手约束",
        ],
        "config_overrides": {
            "factor_source_mode": "qlib_baseline",
            "challenger_weight": 0.0,
            "topk": 100,
            "n_drop": 10,
            "max_position_weight": 0.05,
            "max_daily_turnover": 0.15,
            "portfolio_construction": "industry_neutral_qp",
            "optimizer_alpha_weight": 0.05,
            "optimizer_tracking_penalty": 1.0,
            "optimizer_turnover_penalty": 0.10,
            "max_industry_weight": 0.15,
            "max_industry_deviation": 0.03,
            "max_size_deviation": 0.03,
            "max_value_deviation": 0.03,
            "max_growth_deviation": 0.03,
            "max_volatility_deviation": 0.03,
            "max_drawdown": 0.08,
            "target_volatility": 0.15,
            "rebalance_frequency": "month",
            "capacity_notional": 5_000_000,
            "max_volume_participation": 0.01,
            "min_average_daily_amount": 500_000_000,
            "execution_days": 3,
            "execution_method": "vwap",
            "signal_frequency": "day",
            "signal_period": 21,
            "execution_frequency": "5min",
        },
        "document_evidence": [
            "PIT全A股股票池按流通市值设定行业目标",
            "100只、单票5%、单行业15%、目标波动率15%",
            "默认5分钟执行并在3个交易日内完成VWAP",
        ],
    },
    {
        "id": "minute_mean_reversion",
        "version": RECIPE_VERSION,
        "name": "分钟超跌均值回归",
        "category": "multifactor",
        "description": "使用Qlib分钟表达式形成的多头超跌回归卫星，严格下一Bar执行。",
        "benchmark": "SH000300",
        "benchmark_role": "reporting_only",
        "universe": "cn_all",
        "position_side": "long_only",
        "factor_baseline": (
            {
                "id": "oversold_60m",
                "weight": 0.50,
                "qlib_expression": "-($close/Mean($close,12)-1)",
            },
            {
                "id": "intraday_vwap_discount",
                "weight": 0.30,
                "qlib_expression": "-($close/$vwap-1)",
            },
            {
                "id": "lower_band_120m",
                "weight": 0.20,
                "qlib_expression": "-(($close-Mean($close,24))/(Std($close,24)+1e-12))",
            },
        ),
        "preprocessing": ["PIT可交易过滤", "缩尾", "截面z-score"],
        "rdagent_objective": (
            "研究A股多头分钟超跌回归信号。默认使用5分钟信号，候选只能使用Qlib"
            "表达式或Qlib模型并由独立复算验证；不得引入Tick、Level-2、其他数据商"
            "或独立分钟回测引擎。信号必须在下一可成交1/5分钟Bar执行，股票卖出遵守T+1。"
        ),
        "factor_guidance": [
            "60分钟价格超跌与120分钟下轨偏离",
            "相对当期VWAP的折价",
            "成交量、停牌、涨跌停和流动性过滤",
            "仅做多并在下一Bar执行",
        ],
        "config_overrides": {
            "factor_source_mode": "qlib_baseline",
            "challenger_weight": 0.0,
            "topk": 20,
            "n_drop": 20,
            "max_position_weight": 0.05,
            "max_daily_turnover": 0.30,
            "portfolio_construction": "topk_equal_weight",
            "max_industry_weight": 0.30,
            "max_drawdown": 0.08,
            "capacity_notional": 5_000_000,
            "max_volume_participation": 0.01,
            "min_average_daily_amount": 500_000_000,
            "execution_days": 1,
            "execution_method": "next_bar",
            "execution_slice_minutes": 5,
            "max_execution_slices": 1,
            "rebalance_frequency": "bar",
            "signal_frequency": "5min",
            "signal_period": 12,
            "execution_frequency": "5min",
        },
        "document_evidence": [
            "第一版为多头超跌回归",
            "默认5分钟信号，使用1/5分钟执行",
            "股票交易遵守T+1，不使用毫秒级高频数据",
        ],
    },
)


def list_strategy_recipes() -> list[dict[str, Any]]:
    """Return immutable product recipes as independent values."""

    return deepcopy(list(_RECIPES))


def get_strategy_recipe(recipe_id: str) -> dict[str, Any]:
    for recipe in _RECIPES:
        if recipe["id"] == recipe_id:
            return deepcopy(recipe)
    raise KeyError(recipe_id)
