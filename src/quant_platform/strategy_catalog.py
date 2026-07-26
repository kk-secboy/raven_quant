"""Machine-readable strategy template catalog (design draft 6.4/13).

The design requires every catalog item — implemented or not — to carry a
``catalog_role`` (baseline / alpha_template / recipe_variant / research_only /
model_challenger / research_admission) and an ``implementation_tier``
(foundation / standard / conditional). ``conditional`` items stay in
``research`` or ``blocked_by_data_or_permission`` until their data and
permission gates are met, and must remain visible by name, role and status
instead of silently disappearing from the design.

This registry is the code-level source of truth for that contract. The four
implemented Qlib recipes live in ``strategy_recipes``; this module links them
to their design templates and registers every other template with its current
implementation status so consumers can query the full catalog
(``list_recipe_catalog``) instead of only the runnable subset.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .strategy_recipes import list_strategy_recipes

CATALOG_VERSION = "strategy-catalog-2026-07-21-v1"

CATALOG_ROLES = frozenset(
    {
        "baseline",
        "alpha_template",
        "recipe_variant",
        "research_only",
        "model_challenger",
        "research_admission",
    }
)
IMPLEMENTATION_TIERS = frozenset({"foundation", "standard", "conditional"})
IMPLEMENTATION_STATUSES = frozenset(
    {"implemented", "research", "blocked_by_data_or_permission"}
)

# Design-draft 6.4 template directory. ``recipe_id`` links an implemented Qlib
# recipe from strategy_recipes; ``parent_template_id`` marks recipe variants
# and model challengers that inherit a parent StrategySpec.
_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "template_id": "simple_policy_baseline",
        "name": "现金/宽基ETF简单政策基线",
        "catalog_role": "baseline",
        "implementation_tier": "foundation",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "etf_asset_allocation",
        "name": "ETF资产配置",
        "catalog_role": "alpha_template",
        "implementation_tier": "standard",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "etf_rotation",
        "name": "ETF低换手轮动",
        "catalog_role": "alpha_template",
        "implementation_tier": "standard",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "etf_risk_balanced_recipe",
        "name": "ETF风险平衡配方",
        "catalog_role": "recipe_variant",
        "implementation_tier": "standard",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": "etf_asset_allocation",
        "recipe_id": None,
    },
    {
        "template_id": "personal_stock_core",
        "name": "个人股票核心",
        "catalog_role": "alpha_template",
        "implementation_tier": "standard",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "single_factor_stock",
        "name": "单因子股票多头（动量/反转/价值/质量/成长/低波/流动性）",
        "catalog_role": "alpha_template",
        "implementation_tier": "standard",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "simple_multifactor_stock",
        "name": "简单多因子股票组合",
        "catalog_role": "alpha_template",
        "implementation_tier": "standard",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "index_enhancement",
        "name": "指数增强",
        "catalog_role": "alpha_template",
        "implementation_tier": "standard",
        "implementation_status": "implemented",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": "index_enhancement",
    },
    {
        "template_id": "full_market_multifactor",
        "name": "全市场多因子",
        "catalog_role": "alpha_template",
        "implementation_tier": "standard",
        "implementation_status": "implemented",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": "full_market_multifactor",
    },
    {
        "template_id": "industry_neutral_multifactor",
        "name": "行业中性多因子配方",
        "catalog_role": "recipe_variant",
        "implementation_tier": "standard",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": "full_market_multifactor",
        "recipe_id": None,
    },
    {
        "template_id": "trend_ma",
        "name": "均线趋势",
        "catalog_role": "alpha_template",
        "implementation_tier": "standard",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "trend_breakout",
        "name": "N日突破趋势",
        "catalog_role": "alpha_template",
        "implementation_tier": "standard",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "swing_trend",
        "name": "复合波段趋势",
        "catalog_role": "alpha_template",
        "implementation_tier": "standard",
        "implementation_status": "implemented",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": "swing_trend",
    },
    {
        "template_id": "daily_mean_reversion",
        "name": "日线均值回归",
        "catalog_role": "alpha_template",
        "implementation_tier": "standard",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "weekly_tactical_overlay",
        "name": "周频战术叠加",
        "catalog_role": "alpha_template",
        "implementation_tier": "standard",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "industry_rotation",
        "name": "行业轮动",
        "catalog_role": "alpha_template",
        "implementation_tier": "conditional",
        "implementation_status": "blocked_by_data_or_permission",
        "blocked_reason": "缺少可重建的PIT行业分类历史与资金口径发布时间，禁止正式评估",
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "event_driven_swing",
        "name": "事件驱动波段",
        "catalog_role": "alpha_template",
        "implementation_tier": "conditional",
        "implementation_status": "blocked_by_data_or_permission",
        "blocked_reason": "缺少冻结公开时间的PIT事件库，停在research",
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "etf_long_long_relative_allocation",
        "name": "ETF long-only 相对配置",
        "catalog_role": "alpha_template",
        "implementation_tier": "standard",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "stock_pair_stat_arb",
        "name": "股票对统计套利（永久离线研究）",
        "catalog_role": "research_only",
        "implementation_tier": "conditional",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "minute_mean_reversion_etf",
        "name": "ETF分钟均值回归",
        "catalog_role": "alpha_template",
        "implementation_tier": "conditional",
        "implementation_status": "blocked_by_data_or_permission",
        "blocked_reason": "T+0 ETF 规则清单与独立分钟证据未冻结",
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "stock_intraday_entry_overnight_exit",
        "name": "股票分钟择时买入/隔日退出",
        "catalog_role": "alpha_template",
        "implementation_tier": "conditional",
        "implementation_status": "implemented",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": "minute_mean_reversion",
    },
    {
        "template_id": "independent_minute_alpha",
        "name": "目录外分钟假设受控入口",
        "catalog_role": "research_admission",
        "implementation_tier": "conditional",
        "implementation_status": "blocked_by_data_or_permission",
        "blocked_reason": "任何新分钟逻辑须先走 NewStrategyProposal 取得独立 template_id",
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "ml_cross_sectional_enhancement",
        "name": "ML横截面增强挑战者",
        "catalog_role": "model_challenger",
        "implementation_tier": "standard",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": "personal_stock_core",
        "recipe_id": None,
    },
    {
        "template_id": "ml_trend_enhancement",
        "name": "ML趋势增强挑战者",
        "catalog_role": "model_challenger",
        "implementation_tier": "standard",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": "trend_ma",
        "recipe_id": None,
    },
    {
        "template_id": "ml_ensemble_challenger",
        "name": "ML集成挑战者",
        "catalog_role": "model_challenger",
        "implementation_tier": "standard",
        "implementation_status": "research",
        "blocked_reason": None,
        "parent_template_id": None,
        "recipe_id": None,
    },
    {
        "template_id": "ml_event_probability",
        "name": "ML事件概率挑战者",
        "catalog_role": "model_challenger",
        "implementation_tier": "conditional",
        "implementation_status": "blocked_by_data_or_permission",
        "blocked_reason": "事件父模板尚未通过 PIT 数据门",
        "parent_template_id": "event_driven_swing",
        "recipe_id": None,
    },
)


def list_recipe_catalog() -> list[dict[str, Any]]:
    """Return the full template catalog with live implementation status.

    Every entry carries template_id, name, catalog_role, implementation_tier,
    implementation_status, blocked_reason, parent_template_id and recipe_id.
    Entries whose linked recipe exists in ``strategy_recipes`` are reported
    as ``implemented`` regardless of the static registration, so the catalog
    can never drift from the runnable recipe set.
    """

    implemented_recipes = {item["id"] for item in list_strategy_recipes()}
    catalog = []
    for entry in _CATALOG:
        item = deepcopy(entry)
        if item["recipe_id"] in implemented_recipes:
            item["implementation_status"] = "implemented"
            item["blocked_reason"] = None
        catalog.append(item)
    return catalog


def get_catalog_entry(template_id: str) -> dict[str, Any]:
    for entry in list_recipe_catalog():
        if entry["template_id"] == template_id:
            return entry
    raise KeyError(template_id)


def catalog_entries_by_status(implementation_status: str) -> list[dict[str, Any]]:
    if implementation_status not in IMPLEMENTATION_STATUSES:
        raise ValueError(f"unknown implementation status: {implementation_status}")
    return [
        entry
        for entry in list_recipe_catalog()
        if entry["implementation_status"] == implementation_status
    ]


def validate_recipe_catalog() -> None:
    """Fail closed if the catalog contract is violated."""

    entries = list_recipe_catalog()
    ids = [entry["template_id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("strategy catalog contains duplicate template ids")
    for entry in entries:
        if entry["catalog_role"] not in CATALOG_ROLES:
            raise ValueError(f"unknown catalog role: {entry['catalog_role']}")
        if entry["implementation_tier"] not in IMPLEMENTATION_TIERS:
            raise ValueError(f"unknown implementation tier: {entry['implementation_tier']}")
        if entry["implementation_status"] not in IMPLEMENTATION_STATUSES:
            raise ValueError(
                f"unknown implementation status: {entry['implementation_status']}"
            )
        if entry["implementation_status"] == "blocked_by_data_or_permission" and not entry[
            "blocked_reason"
        ]:
            raise ValueError(
                f"blocked catalog entry {entry['template_id']} has no blocked reason"
            )
    linked = {
        entry["recipe_id"] for entry in entries if entry["recipe_id"] is not None
    }
    missing = {item["id"] for item in list_strategy_recipes()} - linked
    if missing:
        raise ValueError(f"implemented recipes missing from the catalog: {sorted(missing)}")
