from __future__ import annotations

from copy import deepcopy
from typing import Any

from quant_platform.strategy_rule_ir import validate_strategy_rule_ir

# v17, the narrowly scoped v18 consumed-history rehabilitation, and v20-v22
# remain historical identities. v23 is the current ordinary three-horizon
# recipe and seals the PIT acquisition-lineage semantics into the complete
# governed research-to-paper economic closure.
RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-09-01-v23"

QLIB_SIX_FACTOR_BASELINE: tuple[dict[str, Any], ...] = (
    {"id": "momentum", "weight": 0.20, "qlib_expression": "Ref($close,21)/Ref($close,252)-1"},
    # Qlib expressions have no unary-minus operator; rewrite negations as
    # explicit binary subtractions (design: 1-month reversal = 1 - r_21).
    {"id": "reversal", "weight": 0.10, "qlib_expression": "1-Ref($close,1)/Ref($close,21)"},
    {"id": "value", "weight": 0.20, "qlib_expression": "(1/$pe_ttm+1/$pb)/2"},
    {"id": "quality", "weight": 0.20, "qlib_expression": "($fund_roe+$fund_roa)/2"},
    {
        "id": "growth",
        "weight": 0.10,
        # fund_quarter_profit_yoy is declared but never populated (Tushare
        # q_profit_yoy is a non-default column the downloader never receives);
        # fund_op_profit_yoy is the populated quarterly profit-growth field.
        "qlib_expression": "($fund_quarter_revenue_yoy+$fund_op_profit_yoy)/2",
    },
    {
        "id": "low_volatility",
        "weight": 0.20,
        "qlib_expression": "0-Std($close/Ref($close,1)-1,60)",
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
            "Greater($close,Mean($close,20))"
            "+Greater(Mean($close,20),Mean($close,60))"
            "+Greater(Mean($close,60),Mean($close,120))"
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
        "weight": 0.10,
        "qlib_expression": "($fund_roe+$fund_roa)/2",
    },
    {
        "id": "industry_relative_strength_3m",
        "weight": 0.05,
        "qlib_expression": "$close/Ref($close,63)-1",
    },
)

SHORT_QLIB_BASELINE: tuple[dict[str, Any], ...] = (
    {
        "id": "relative_strength_5d",
        "weight": 0.35,
        "qlib_expression": "$close/Ref($close,5)-1",
    },
    {
        "id": "amount_expansion_5d",
        "weight": 0.25,
        "qlib_expression": "Mean($amount,5)/(Mean($amount,20)+1e-12)-1",
    },
    {
        "id": "close_location_5d",
        "weight": 0.20,
        "qlib_expression": "($close-Min($low,5))/(Max($high,5)-Min($low,5)+1e-12)",
    },
    {
        "id": "extension_penalty_5d",
        "weight": 0.20,
        "qlib_expression": "0-Abs($close/Mean($close,5)-1)",
    },
)

LONG_QLIB_BASELINE: tuple[dict[str, Any], ...] = (
    {
        "id": "capital_efficiency",
        "weight": 0.25,
        "qlib_expression": "($fund_roic+$fund_roe)/2",
    },
    {
        "id": "cash_profit_quality",
        "weight": 0.20,
        "qlib_expression": "$fund_sales_cash_to_revenue",
    },
    {
        "id": "earnings_value",
        "weight": 0.20,
        "qlib_expression": "(1/$pe_ttm+1/$pb)/2",
    },
    {
        "id": "durable_growth",
        "weight": 0.15,
        "qlib_expression": "($fund_quarter_revenue_yoy+$fund_op_profit_yoy)/2",
    },
    {
        "id": "balance_sheet_resilience",
        "weight": 0.10,
        "qlib_expression": "0-$fund_debt_to_assets",
    },
    {
        "id": "earnings_stability",
        "weight": 0.10,
        "qlib_expression": "0-Std($fund_roe,252)",
    },
)


def _component(component: str, **parameters: Any) -> dict[str, Any]:
    return {"component": component, "parameters": parameters}


def _transparent_baseline_rules(
    horizon: str,
    factor_baseline: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    weights = {str(item["id"]): float(item["weight"]) for item in factor_baseline}
    holding_days = {"short_1_5d": 5, "swing_1_6m": 126}.get(horizon)
    entry_components = [
        _component("next_open", max_signal_age_days=1),
        _component("score_threshold", minimum_percentile=0.80),
        _component(
            "rebalance_calendar",
            frequency={"short_1_5d": "day", "swing_1_6m": "week", "long_1_3y": "month"}[
                horizon
            ],
        ),
    ]
    if horizon == "short_1_5d":
        entry_components.append(_component("extension_guard", max_return_5d=0.12))
    exit_components: list[dict[str, Any]] = []
    if holding_days is not None:
        exit_components.append(_component("max_holding_days", days=holding_days))
    if horizon == "short_1_5d":
        exit_components.extend(
            [
                _component("score_drop_exit", below_percentile=0.50),
                _component("stop_loss", fraction=0.05),
            ]
        )
    elif horizon == "swing_1_6m":
        exit_components.extend(
            [
                _component(
                    "score_deterioration_reduce",
                    below_percentile=0.35,
                    reduce_fraction=0.50,
                ),
                _component("trend_break", lookback_days=20),
                _component("stop_loss", fraction=0.07),
            ]
        )
    else:
        exit_components.extend(
            [
                _component(
                    "score_deterioration_reduce",
                    below_percentile=0.40,
                    reduce_fraction=0.33,
                ),
                _component(
                    "valuation_reduce",
                    above_percentile=0.95,
                    reduce_fraction=0.33,
                ),
                _component("stop_loss", fraction=0.25),
                _component(
                    "thesis_break",
                    minimum_holding_days=252,
                    review_frequency="month",
                ),
            ]
        )
    topk = {"short_1_5d": 20, "swing_1_6m": 20, "long_1_3y": 30}[horizon]
    position_weight = {"short_1_5d": 0.05, "swing_1_6m": 0.10, "long_1_3y": 0.05}[
        horizon
    ]
    direction_components = [_component("long_only")]
    if horizon == "long_1_3y":
        direction_components.append(
            _component("valuation_regime_filter", max_percentile=0.90)
        )
    else:
        direction_components.append(
            _component(
                "market_trend_filter",
                benchmark="SH000300",
                lookback_days={"short_1_5d": 20, "swing_1_6m": 60}[horizon],
            )
        )
    slots = {
        "eligibility_gate": {
            "required": True,
            "components": [
                _component(
                    "tradable_ashare",
                    min_listing_days={
                        "short_1_5d": 60,
                        "swing_1_6m": 252,
                        "long_1_3y": 756,
                    }[horizon],
                ),
                _component(
                    "liquidity_floor",
                    min_average_daily_amount=500_000_000,
                    lookback_days={
                        "short_1_5d": 20,
                        "swing_1_6m": 20,
                        "long_1_3y": 60,
                    }[horizon],
                ),
                _component("regulatory_exclusion", exclude=["st", "suspended", "delisting_risk"]),
            ],
            "empty_behavior": "no_new_entries",
        },
        "universe_dedup": {
            "required": True,
            "components": [_component("instrument_unique", identity="canonical_instrument_id")],
            "empty_behavior": "no_new_entries",
        },
        "direction_regime_gate": {
            "required": True,
            "components": direction_components,
            "empty_behavior": "remain_in_cash",
        },
        "alpha_rank": {
            "required": True,
            "components": [_component("weighted_factor_rank", weights=weights)],
            "empty_behavior": "remain_in_cash",
        },
        "entry_timing": {
            "required": True,
            "components": entry_components,
            "empty_behavior": "no_new_entries",
        },
        "exit_state": {
            "required": True,
            "components": exit_components,
            "empty_behavior": "hold_existing",
        },
        "portfolio_risk": {
            "required": True,
            "components": [
                _component("topk_equal_weight", topk=topk, max_position_weight=position_weight),
                _component(
                    "max_industry_weight",
                    fraction={"short_1_5d": 0.30, "swing_1_6m": 0.30, "long_1_3y": 0.20}[
                        horizon
                    ],
                ),
                _component(
                    "max_daily_turnover",
                    fraction={"short_1_5d": 0.50, "swing_1_6m": 0.25, "long_1_3y": 0.05}[
                        horizon
                    ],
                ),
                _component(
                    "minimum_trade_band",
                    fraction={
                        "short_1_5d": 0.002,
                        "swing_1_6m": 0.005,
                        "long_1_3y": 0.0025,
                    }[horizon],
                ),
                _component("cash_when_no_edge"),
            ],
            "empty_behavior": "remain_in_cash",
        },
        "execution_requirement": {
            "required": True,
            "components": [
                _component("a_share_t_plus_one"),
                _component("board_lot", shares=100),
                _component("price_limit_guard"),
                _component("liquidity_participation", max_fraction=0.01),
            ],
            "empty_behavior": "no_new_entries",
        },
    }
    return validate_strategy_rule_ir(horizon, slots, allowed_factor_ids=set(weights))


TRANSPARENT_RESEARCH_BASELINE_IDS = (
    "short_relative_strength",
    "swing_trend",
    "long_quality_value",
)

_RECIPES: tuple[dict[str, Any], ...] = (
    {
        "id": "short_relative_strength",
        "version": RECIPE_VERSION,
        "name": "1至5日短线相对强弱",
        "category": "transparent_research_baseline",
        "description": "日线收盘后筛选，下一交易日开盘执行，最多持有5个交易日的透明基线。",
        "benchmark": "SH000300",
        "universe": "cn_all",
        "horizon": "short_1_5d",
        "research_baseline": True,
        "factor_baseline": SHORT_QLIB_BASELINE,
        "strategy_rule_ir": _transparent_baseline_rules("short_1_5d", SHORT_QLIB_BASELINE),
        "preprocessing": ["PIT可交易/监管过滤", "缩尾", "截面z-score"],
        "rdagent_objective": (
            "研究1至5个交易日的日线短线候选，不使用盘中、Tick或Level-2数据。"
            "重点检验相对强弱、成交额扩张、收盘位置与过度追涨惩罚；信号只允许"
            "下一交易日开盘执行，并与本透明基线做同数据、同成本、同容量的滚动样本外比较。"
        ),
        "factor_guidance": [
            "5日相对强弱与成交额扩张",
            "5日区间收盘位置",
            "高位延伸惩罚，避免把已经涨到位当成新买点",
            "最多持有5个交易日且股票卖出遵守T+1",
        ],
        "config_overrides": {
            "horizon_profile": "short_1_5d",
            "factor_source_mode": "qlib_baseline",
            "challenger_weight": 0.0,
            "topk": 20,
            "n_drop": 20,
            "max_position_weight": 0.05,
            "max_daily_turnover": 0.50,
            "max_daily_loss": 0.03,
            "stop_loss": 0.05,
            "take_profit_partial": 0.08,
            "take_profit_partial_fraction": 0.50,
            "take_profit": 0.15,
            "max_industry_weight": 0.30,
            "min_average_daily_amount": 500_000_000,
            "liquidity_lookback_days": 20,
            "require_regulatory_events": True,
            "portfolio_construction": "topk_equal_weight",
            "capacity_notional": 5_000_000,
            "max_volume_participation": 0.01,
            "execution_days": 1,
            "execution_method": "open",
            "signal_frequency": "day",
            "signal_period": 5,
            "execution_frequency": "day",
            "rebalance_frequency": "day",
        },
        "document_evidence": [
            "仅使用每日收盘后可得数据，次日开盘执行",
            "最多持有5个交易日并显式限制追高",
            "同一成本、容量和滚动样本外口径下比较挑战者",
        ],
    },
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
        "horizon": "swing_1_6m",
        "research_baseline": True,
        "factor_baseline": SWING_QLIB_BASELINE,
        "strategy_rule_ir": _transparent_baseline_rules("swing_1_6m", SWING_QLIB_BASELINE),
        "preprocessing": ["PIT可交易/监管过滤", "缩尾", "截面z-score"],
        "rdagent_objective": (
            "研究A股中低频波段策略所需的可解释日频因子，严格禁止未来函数。"
            "核心信号包括收盘价高于MA20、MA20高于MA60、MA60高于MA120、"
            "行业相对强弱、Wilder ADX(14)、5日成交额放量和20日布林带宽度。"
            "输出必须由Qlib独立复算并接受滚动样本外、成本、容量和事件压力测试。"
        ),
        "factor_guidance": [
            "MA20/MA60/MA120趋势结构",
            "63日行业相对强弱",
            "标准Wilder ADX(14)趋势强度",
            "5日成交额放量比和20日布林带宽度",
            "ST、停牌、流动性、违规和财务质量过滤",
        ],
        "config_overrides": {
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
        },
        "document_evidence": [
            "MA20/60/120、行业相对强弱、ADX、成交额放量与布林带宽度",
            "趋势失效或7%止损退出；不使用固定盈利阈值强制止盈",
            "20日平均成交额至少5亿元，成交参与率不超过1%",
        ],
    },
    {
        "id": "long_quality_value",
        "version": RECIPE_VERSION,
        "name": "1至3年质量价值",
        "category": "transparent_research_baseline",
        "description": (
            "以质量、估值、稳健成长和资产负债表为核心，"
            "按月及财报后复核投资逻辑的长线基线。"
        ),
        "benchmark": "SH000300",
        "benchmark_role": "reporting_only",
        "universe": "cn_all",
        "horizon": "long_1_3y",
        "research_baseline": True,
        "factor_baseline": LONG_QLIB_BASELINE,
        "strategy_rule_ir": _transparent_baseline_rules("long_1_3y", LONG_QLIB_BASELINE),
        "preprocessing": ["PIT公告日财务数据", "行业相对排名", "缩尾", "截面z-score"],
        "rdagent_objective": (
            "研究1至3年以上的A股质量价值策略，使用公告日可得的财务数据，关注盈利质量、"
            "估值、稳健成长和资产负债表韧性。不得用短线止盈替代投资逻辑；退出由"
            "月度/财报后基本面复核、逻辑破坏和风险上限共同决定，并与透明基线做滚动样本外比较。"
        ),
        "factor_guidance": [
            "ROE/ROA盈利质量",
            "EP/BP估值与持续成长",
            "资产负债表韧性和盈利稳定性",
            "公告日PIT可得性与月度/财报后投资逻辑复核",
        ],
        "config_overrides": {
            "horizon_profile": "long_1_3y",
            "factor_source_mode": "qlib_baseline",
            "challenger_weight": 0.0,
            "topk": 30,
            "n_drop": 5,
            "max_position_weight": 0.05,
            "max_daily_turnover": 0.05,
            "max_daily_loss": 0.03,
            "stop_loss": 0.25,
            "profit_taking_mode": "thesis_only",
            "take_profit_partial": 2.0,
            "take_profit_partial_fraction": 0.50,
            "take_profit": 5.0,
            "max_industry_weight": 0.20,
            "min_average_daily_amount": 500_000_000,
            "liquidity_lookback_days": 60,
            "require_regulatory_events": True,
            "industry_relative_rank": True,
            "portfolio_construction": "topk_equal_weight",
            "target_volatility": 0.15,
            "capacity_notional": 5_000_000,
            "max_volume_participation": 0.01,
            "execution_days": 1,
            "execution_method": "open",
            "signal_frequency": "day",
            "signal_period": 252,
            "execution_frequency": "day",
            "rebalance_frequency": "month",
        },
        "document_evidence": [
            "1至3年持有期和月度/财报后基本面复核",
            "关闭机械止盈，退出只由投资论点、财务恶化、极端估值和硬风险触发",
            "所有财务字段必须遵守公告日PIT可得性",
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
            "候选必须使用同一六因子基线、公告日可得财务数据，以及Qlib资金面字段"
            "mf_net_inflow_amount、mf_net_inflow_ratio、mf_large_order_imbalance；"
            "公告、研报、新闻、情绪和逻辑因子只能走外部因子评估与晋级通道，不得把"
            "事件后的市场认可度训练标签写回实时特征。不得把沪深300权重作为优化目标；"
            "行业目标按当期全市场流通市值计算，并通过Qlib滚动样本外、成本、容量和"
            "事件压力测试。"
        ),
        "factor_guidance": [
            "动量20%、反转10%、价值20%、质量20%、成长10%、低波动20%",
            "PIT行业和市值中性化、缩尾与z-score",
            "全市场流通市值行业目标和风格暴露中性",
            "资金面候选只使用盘后可得moneyflow字段；信息面只接收已治理晋级artifact",
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
                "qlib_expression": "1-$close/Mean($close,12)",
            },
            {
                "id": "intraday_vwap_discount",
                "weight": 0.30,
                "qlib_expression": "1-$close/$vwap",
            },
            {
                "id": "lower_band_120m",
                "weight": 0.20,
                "qlib_expression": "0-($close-Mean($close,24))/(Std($close,24)+1e-12)",
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


def _with_execution_policy(recipe: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(recipe)
    rule_ir = result.get("strategy_rule_ir")
    if result.get("research_baseline") is not True or not isinstance(rule_ir, dict):
        return result
    from .strategy_rule_compiler import compile_strategy_rule_policy

    allowed_factor_ids = {
        str(item["id"]) for item in result.get("factor_baseline") or []
    }
    policy = compile_strategy_rule_policy(
        str(result["horizon"]),
        rule_ir,
        allowed_factor_ids=allowed_factor_ids,
    )
    result["execution_policy"] = policy
    overrides = dict(result.get("config_overrides") or {})
    projected_fields = {
        "topk",
        "max_position_weight",
        "max_industry_weight",
        "max_daily_turnover",
        "min_average_daily_amount",
        "liquidity_lookback_days",
        "min_listing_days",
        "entry_score_min_percentile",
        "score_drop_exit_percentile",
        "score_deterioration_reduce_percentile",
        "score_deterioration_reduce_fraction",
        "extension_guard_max_return_5d",
        "holding_min_sessions",
        "max_holding_sessions",
        "min_rebalance_weight_change",
        "market_trend_lookback_sessions",
        "market_trend_benchmark",
        "valuation_regime_max_percentile",
        "valuation_reduce_percentile",
        "valuation_reduce_fraction",
        "trend_break_lookback_sessions",
        "thesis_min_holding_sessions",
        "thesis_review_frequency",
        "thesis_break_score_percentile",
        "hard_risk_target_fraction",
        "stop_loss",
        "rebalance_frequency",
        "lot_size",
        "max_volume_participation",
        "execution_method",
        "cash_when_no_edge",
    }
    overrides.update(
        {
            key: value
            for key, value in policy.items()
            if key in projected_fields and value is not None
        }
    )
    overrides.update(
        {
            "strategy_rule_ir": deepcopy(rule_ir),
            "strategy_rules_sha256": str(policy["strategy_rules_sha256"]),
            "strategy_rule_policy_sha256": str(policy["policy_sha256"]),
            "execution_lag_bars": int(policy["execution_lag_sessions"]),
        }
    )
    result["config_overrides"] = overrides
    return result


def list_strategy_recipes() -> list[dict[str, Any]]:
    """Return immutable product recipes as independent values."""

    return [_with_execution_policy(recipe) for recipe in _RECIPES]


def get_strategy_recipe(
    recipe_id: str, *, include_execution_policy: bool = True
) -> dict[str, Any]:
    for recipe in _RECIPES:
        if recipe["id"] == recipe_id:
            return _with_execution_policy(recipe) if include_execution_policy else deepcopy(recipe)
    raise KeyError(recipe_id)
