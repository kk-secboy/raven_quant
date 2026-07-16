from __future__ import annotations

from dataclasses import asdict

import pytest

from quant_platform.cost_model import COST_SCHEDULE_VERSION
from quant_platform.pair_trading import PairTradingConfig
from quant_platform.recommendation_store import RecommendationStore
from quant_platform.strategy_store import StrategyStore


def _pair(store: StrategyStore) -> dict:
    return store.create_pair(
        name="沪深300 ETF 配对",
        description="使用协整、Kalman 对冲比和分钟 VWAP 执行的受治理 ETF 配对策略。",
        leg_y="SH510300",
        leg_x="SZ159919",
        asset_class="etf",
        shorting_mode="margin_borrow",
        config=asdict(PairTradingConfig()),
        actor="researcher-a",
    )


def _passing_metrics() -> dict:
    digest = "a" * 64
    return {
        "backtest_engine": "quantlab_pair",
        "pair_native_backtest": True,
        "leg_y": "SH510300",
        "leg_x": "SZ159919",
        "initial_pair_evidence": {
            "correlation": 0.92,
            "cointegration_pvalue": 0.01,
            "hedge_ratio": 0.95,
        },
        "max_drawdown": -0.06,
        "sharpe_ratio": 1.2,
        "closed_trade_count": 12,
        "trading_days": 504,
        "rolling_cointegration_pass_rate": 0.90,
        "pair_robustness_pass_rate": 0.75,
        "capacity_fill_ratio": 0.99,
        "minute_execution_enforced": True,
        "shortability_enforced": True,
        "market_controls_enforced": True,
        "atomic_pair_execution_enforced": True,
        "transaction_costs_enforced": True,
        "borrow_cost_enforced": True,
        "cost_schedule_version": COST_SCHEDULE_VERSION,
        "open_position_at_end": False,
        "provenance": {
            "daily_dataset_identity_sha256": digest,
            "daily_snapshot_manifest_sha256": digest,
            "minute_snapshot_manifest_sha256": digest,
            "strategy_config_sha256": digest,
            "execution_manifest_sha256": digest,
            "pair_engine_sha256": digest,
            "shortability_evidence_sha256": digest,
        },
    }


def test_pair_strategy_uses_shared_version_and_backtest_governance(
    database_url: str, tmp_path
) -> None:
    store = StrategyStore(database_url)
    strategy = _pair(store)
    version = strategy["versions"][0]
    assert version["strategy_type"] == "pair"
    assert version["factors"] == []
    assert version["pair"]["leg_y"] == "SH510300"
    backtest = store.create_backtest(
        version_id=version["id"],
        dataset="daily-2024-2026",
        execution_dataset="minute-2024-2026/liquid_stocks_1m",
        periods={"start": "2024-01-01", "end": "2025-12-31"},
        artifact_path=tmp_path,
    )
    store.mark_backtest(backtest["id"], "succeeded", metrics=_passing_metrics())
    approved = store.approve(
        version["id"],
        actor="risk-approver-b",
        reason="协整、成本压力、容量和融券证据均通过独立复核。",
    )
    assert approved["status"] == "approved"
    assert approved["approved_by"] == "risk-approver-b"


def test_pair_strategy_requires_second_person_and_minute_dataset(
    database_url: str, tmp_path
) -> None:
    store = StrategyStore(database_url)
    version = _pair(store)["versions"][0]
    backtest = store.create_backtest(
        version_id=version["id"],
        dataset="daily-2024-2026",
        periods={"start": "2024-01-01", "end": "2025-12-31"},
        artifact_path=tmp_path,
    )
    store.mark_backtest(backtest["id"], "succeeded", metrics=_passing_metrics())
    with pytest.raises(ValueError, match="second operator"):
        store.approve(
            version["id"],
            actor="researcher-a",
            reason="创建人不得自行批准这套配对策略进入下一阶段。",
        )
    with pytest.raises(ValueError, match="minute execution dataset"):
        store.approve(
            version["id"],
            actor="risk-approver-b",
            reason="没有分钟执行数据时必须保持失败关闭。",
        )


def test_recommendation_portfolio_rejects_pair_research_version(
    database_url: str, tmp_path
) -> None:
    strategies = StrategyStore(database_url)
    version = _pair(strategies)["versions"][0]
    backtest = strategies.create_backtest(
        version_id=version["id"],
        dataset="daily-2024-2026",
        execution_dataset="minute-2024-2026/liquid_stocks_1m",
        periods={"start": "2024-01-01", "end": "2025-12-31"},
        artifact_path=tmp_path,
    )
    strategies.mark_backtest(backtest["id"], "succeeded", metrics=_passing_metrics())
    strategies.approve(
        version["id"],
        actor="risk-approver-b",
        reason="批准研究版本，但尚未接入专用价差模拟账本。",
    )
    with pytest.raises(ValueError, match="approved multifactor strategy"):
        RecommendationStore(database_url).create(
            name="invalid-pair-recommendation",
            strategy_version_id=version["id"],
            dataset="daily-2024-2026",
            hypothetical_initial_value=5_000_000,
            actor="operator-c",
        )
