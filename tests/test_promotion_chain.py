"""Promotion chain: paper stage, forward evidence gate, recommendation gating."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from governance_fixtures import (
    PERIODS,
    create_strategy_version,
    formal_backtest_metrics,
)
from qlib_test_doubles import (
    QlibPortfolioOptimizer,
    QlibRiskEstimator,
    qlib_runtime_identity,
)
from sqlalchemy import insert, select, update
from test_strategy_allocation_recommendations import (
    _approve_version,
    _daily_dataset,
    _minute_dataset,
)

import quant_platform.risk_math as risk_math
import quant_platform.strategy_allocation as strategy_allocation
from quant_data.database import (
    open_database,
    simulation_batches,
    simulation_fills,
    simulation_nav,
    simulation_orders,
    simulation_portfolios,
    strategy_versions,
)
from quant_data.execution_contract import (
    MINUTE_EXECUTION_CONTRACT_VERSION,
    MINUTE_SOURCE_UNIT_CONTRACTS,
)
from quant_platform.promotion import (
    ForwardGateThresholds,
    PromotionStore,
)
from quant_platform.recommendation_store import RecommendationStore
from quant_platform.strategy_store import StrategyStore

ACTOR = "promotion-operator"


def _qlib_doubles(monkeypatch) -> None:
    monkeypatch.setattr(risk_math, "_load_qlib_risk_model", lambda: QlibRiskEstimator)
    monkeypatch.setattr(risk_math, "upstream_runtime_identity", qlib_runtime_identity)
    monkeypatch.setattr(
        strategy_allocation,
        "_load_qlib_portfolio_optimizer",
        lambda: QlibPortfolioOptimizer,
    )
    monkeypatch.setattr(
        strategy_allocation, "upstream_runtime_identity", qlib_runtime_identity
    )


def _returns() -> pd.Series:
    dates = pd.bdate_range("2024-01-02", periods=160)
    return pd.Series(np.sin(np.arange(len(dates)) / 5) * 0.01, index=dates)


def _paper_version(database_url: str, tmp_path: Path, monkeypatch) -> str:
    """A real-approved day-execution version (paper, awaiting stage)."""

    _qlib_doubles(monkeypatch)
    return _approve_version(database_url, tmp_path, suffix="promotion", returns=_returns())


def _minute_version(
    database_url: str,
    tmp_path: Path,
    *,
    suffix: str,
    with_datasets: bool = False,
) -> str:
    """A real-approved minute-execution version (5min/twap contract)."""

    version_id = create_strategy_version(
        database_url,
        tmp_path,
        dataset="allocation-data",
        config_overrides={"execution_method": "twap", "execution_slice_minutes": 20},
    )
    strategies = StrategyStore(database_url)
    version = strategies.get_version(version_id)
    artifact = tmp_path / f"minute-backtest-{suffix}"
    artifact.mkdir()
    periods = {
        "start": PERIODS["test_start"].isoformat(),
        "end": PERIODS["test_end"].isoformat(),
    }
    backtest = strategies.create_backtest(
        version_id=version_id,
        dataset="allocation-data",
        execution_dataset="allocation-5m",
        periods=periods,
        artifact_path=artifact,
    )
    factor = version["factors"][0]
    manifest = artifact / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy_version_id": version_id,
                "dataset": "allocation-data",
                "execution_dataset": "allocation-5m",
                "benchmark": version["benchmark"],
                "universe": version["universe"],
                "periods": periods,
                "config": version["config"],
                "factors": [
                    {
                        "candidate_id": factor["factor_candidate_id"],
                        "values_path": factor["values_path"],
                        "code_sha256": factor["code_sha256"],
                        "weight": factor["weight"],
                        "direction": factor["direction"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    metrics = formal_backtest_metrics(version, manifest)
    metrics.update(
        {
            "minute_execution_enforced": True,
            "capacity_fill_ratio": 0.99,
            "execution_model": {
                "method": "twap",
                "frequency": "5min",
                "price_assumption": "minute bar vwap fills",
                "strategy_contract_hash": version["config"]["execution_contract_hash"],
            },
        }
    )
    metrics["provenance"].update(
        {
            "execution_dataset_identity_sha256": "d" * 64,
            "execution_snapshot_manifest_sha256": "e" * 64,
            "execution_qlib_builder_sha256": "f" * 64,
            "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
            "execution_fields": ["vwap", "volume", "paused", "up_limit", "down_limit"],
            "execution_source_datasets": ["ashare_5m"],
            "execution_source_unit_contracts": {
                "ashare_5m": MINUTE_SOURCE_UNIT_CONTRACTS["ashare_5m"]
            },
            "execution_lineage_verified": True,
            "execution_source_lineage_id": "9" * 64,
        }
    )
    strategies.validate_backtest_artifacts(backtest["id"], metrics)
    strategies.mark_backtest(backtest["id"], "succeeded", metrics=metrics)
    if with_datasets:
        (artifact / "datasets.json").write_text(
            json.dumps(
                {"daily": _daily_dataset(), "execution": _minute_dataset()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    strategies.approve(
        version_id,
        actor="allocation-risk-owner",
        reason="Approved independently for promotion chain testing.",
    )
    return version_id


def _attach_simulation(store: PromotionStore, version_id: str) -> dict:
    store.open_paper_stage(version_id, actor=ACTOR)
    stage = store.attach_paper_simulation(
        version_id,
        actor=ACTOR,
        daily_dataset=_daily_dataset(),
        execution_dataset=_minute_dataset(),
    )
    assert stage["status"] == "active"
    return stage


def _register_gate(store: PromotionStore, version_id: str, **overrides) -> None:
    values = {
        "min_forward_calendar_days": 3,
        "min_decision_batches": 2,
        "min_completed_cycles": 1,
        "min_data_completeness": 0.8,
        "min_reconciliation_rate": 1.0,
        "max_cost_deviation": 0.01,
    }
    values.update(overrides)
    thresholds = ForwardGateThresholds(**values)
    engine = store.engine
    with engine.begin() as connection:
        # Gate registration must happen before paper; seed directly for the
        # evidence tests (the registration path itself is tested separately).
        connection.execute(
            insert(_gate_table()).values(
                strategy_version_id=version_id,
                min_forward_calendar_days=thresholds.min_forward_calendar_days,
                min_decision_batches=thresholds.min_decision_batches,
                min_completed_cycles=thresholds.min_completed_cycles,
                min_data_completeness=thresholds.min_data_completeness,
                min_reconciliation_rate=thresholds.min_reconciliation_rate,
                max_cost_deviation=thresholds.max_cost_deviation,
                registered_by=ACTOR,
                registered_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )


def _gate_table():
    from quant_data.database import strategy_forward_gates

    return strategy_forward_gates


def _seed_evidence(
    store: PromotionStore,
    portfolio_id: str,
    *,
    nav_days: int,
    succeeded: int,
    failed: int = 0,
    unreconciled: int = 0,
    sell_batches: int = 0,
    fee: float = 0.0,
    gross: float = 1.0,
) -> None:
    engine = store.engine
    base = date(2026, 7, 1)
    with engine.begin() as connection:
        for index in range(nav_days):
            connection.execute(
                insert(simulation_nav).values(
                    portfolio_id=portfolio_id,
                    trade_date=base + timedelta(days=index),
                    cash=Decimal("1000000"),
                    market_value=Decimal("0"),
                    nav=Decimal("1000000"),
                    daily_return=0.0,
                    drawdown=0.0,
                    has_stale_prices=False,
                    status="certified",
                    performance_certified=True,
                    created_at=datetime.now(UTC),
                )
            )
        batch_ids = []
        for index in range(succeeded + failed):
            batch_id = uuid.uuid4().hex
            ok = index < succeeded
            reconciled = not ok or index >= unreconciled
            summary = (
                {"conservation": {"cash_difference": 0.0 if reconciled else 12.5}}
                if ok
                else None
            )
            connection.execute(
                insert(simulation_batches).values(
                    id=batch_id,
                    portfolio_id=portfolio_id,
                    execution_contract_hash="a" * 64,
                    daily_dataset="promotion-daily",
                    daily_dataset_identity_sha256="b" * 64,
                    daily_dataset_lineage_id="c" * 64,
                    execution_dataset="promotion-minute",
                    execution_dataset_identity_sha256="d" * 64,
                    execution_dataset_lineage_id="e" * 64,
                    simulation_semantics_sha256="f" * 64,
                    signal_date=base,
                    trade_date=base + timedelta(days=index),
                    status="succeeded" if ok else "failed",
                    idempotency_key=f"test-{batch_id}",
                    summary_json=summary,
                    created_at=datetime.now(UTC),
                )
            )
            if ok:
                batch_ids.append(batch_id)
        for index, batch_id in enumerate(batch_ids):
            order_id = uuid.uuid4().hex
            connection.execute(
                insert(simulation_orders).values(
                    id=order_id,
                    batch_id=batch_id,
                    instrument="SH600000",
                    side="sell" if index < sell_batches else "buy",
                    target_weight=0.1,
                    requested_quantity=100,
                    filled_quantity=100,
                    status="filled",
                    requested_value=Decimal("1000"),
                    filled_value=Decimal("1000"),
                    capacity_fill_ratio=1.0,
                    expires_at=datetime.now(UTC),
                    created_at=datetime.now(UTC),
                )
            )
            connection.execute(
                insert(simulation_fills).values(
                    id=uuid.uuid4().hex,
                    order_id=order_id,
                    batch_id=batch_id,
                    instrument="SH600000",
                    side="sell" if index < sell_batches else "buy",
                    executed_at=datetime.now(UTC),
                    quantity=100,
                    price=Decimal("10"),
                    gross_value=Decimal(str(gross)),
                    fee=Decimal(str(fee)),
                    cost_breakdown_json={},
                    minute_volume=1_000_000,
                    capacity_quantity=1_000_000,
                )
            )


# ---------------------------------------------------------------------------
# Lifecycle state machine and automatic paper opening
# ---------------------------------------------------------------------------


def test_approve_moves_version_to_paper_and_opens_stage(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id = _paper_version(database_url, tmp_path, monkeypatch)
    strategies = StrategyStore(database_url)
    version = strategies.get_version(version_id)
    assert version["status"] == "approved"
    assert version["promotion_stage"] == "paper"
    promotion = PromotionStore(database_url)
    stage = promotion.current_stage(version_id)
    # 审批 artifact 无 datasets.json：阶段已开但等待模拟账户（不阻断审批）
    assert stage is not None
    assert stage["status"] == "awaiting_simulation"
    assert stage["stage_index"] == 1
    # 幂等：重复打开返回同一阶段
    again = promotion.open_paper_stage(version_id, actor=ACTOR)
    assert again["id"] == stage["id"]


def test_gate_registration_only_before_paper(database_url: str, tmp_path: Path) -> None:
    version_id = create_strategy_version(database_url, tmp_path)
    promotion = PromotionStore(database_url)
    gate = promotion.register_forward_gate(
        version_id, actor=ACTOR, min_forward_calendar_days=30
    )
    assert gate["min_forward_calendar_days"] == 30
    gate = promotion.register_forward_gate(
        version_id, actor=ACTOR, min_forward_calendar_days=45
    )
    assert gate["min_forward_calendar_days"] == 45
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved", promotion_stage="paper")
        )
    with pytest.raises(ValueError, match="pre-registered before"):
        promotion.register_forward_gate(version_id, actor=ACTOR)


def test_attach_creates_isolated_paper_account(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    _qlib_doubles(monkeypatch)
    version_id = _minute_version(database_url, tmp_path, suffix="attach")
    promotion = PromotionStore(database_url)
    stage = _attach_simulation(promotion, version_id)
    engine = open_database(database_url)
    with engine.connect() as connection:
        portfolio = connection.execute(
            select(simulation_portfolios).where(
                simulation_portfolios.c.id == stage["simulation_portfolio_id"]
            )
        ).one()
    # 独立隔离账户：自己的账本/资本/合同，绑定该版本的冻结来源
    assert str(portfolio.source_type) == "strategy_version"
    assert str(portfolio.source_id) == version_id
    assert float(portfolio.initial_cash) >= 100_000
    assert stage["source_contract_hash"] == str(portfolio.execution_contract_hash)


def test_approve_auto_creates_paper_account_from_dataset_descriptors(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    """candidate -> paper fully automatic: the approval backtest artifact
    carries dataset descriptors, so the hard gate opens the stage and binds
    the isolated paper account in one pass."""

    _qlib_doubles(monkeypatch)
    version_id = _minute_version(database_url, tmp_path, suffix="auto", with_datasets=True)
    promotion = PromotionStore(database_url)
    stage = promotion.current_stage(version_id)
    assert stage["status"] == "active"
    assert stage["simulation_portfolio_id"]
    assert stage["source_contract_hash"]
    # 幂等：重复打开不建第二个账户
    again = promotion.open_paper_stage(version_id, actor=ACTOR)
    assert again["id"] == stage["id"]


# ---------------------------------------------------------------------------
# Forward evidence gate
# ---------------------------------------------------------------------------


def _gated_paper_version(database_url: str, tmp_path: Path, monkeypatch, **gate) -> tuple:
    _qlib_doubles(monkeypatch)
    version_id = _minute_version(database_url, tmp_path, suffix="gated")
    promotion = PromotionStore(database_url)
    _register_gate(promotion, version_id, **gate)
    stage = _attach_simulation(promotion, version_id)
    return version_id, promotion, stage


def test_gate_insufficient_evidence_fail_closed(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    evaluation = promotion.evaluate_forward_gate(version_id)
    assert evaluation["status"] == "insufficient_evidence"
    assert evaluation["passed"] is False
    # 证据不足继续处于 paper，门槛不降级
    with pytest.raises(ValueError, match="insufficient_evidence"):
        promotion.promote(
            version_id, actor="second-operator", reason="Promote after forward evidence."
        )
    version = StrategyStore(database_url).get_version(version_id)
    assert version["promotion_stage"] == "paper"


def test_gate_subitems_and_promotion_with_human_approval(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    portfolio_id = stage["simulation_portfolio_id"]
    _seed_evidence(
        promotion,
        portfolio_id,
        nav_days=4,
        succeeded=3,
        failed=1,
        unreconciled=1,
        sell_batches=1,
        fee=3.0,
        gross=1000.0,
    )
    evaluation = promotion.evaluate_forward_gate(version_id)
    checks = evaluation["checks"]
    assert checks["forward_calendar_days"]["observed"] == 4
    assert checks["decision_batches"]["observed"] == 3
    assert checks["completed_cycles"]["observed"] == 1
    assert checks["data_completeness"]["observed"] == pytest.approx(0.75)
    # 3 个成功批次中 2 个对账通过：对账率门槛 1.0 → 不足
    assert checks["reconciliation_rate"]["observed"] == pytest.approx(2 / 3)
    assert evaluation["passed"] is False
    assert "reconciliation_rate" in str(evaluation["reasons"])
    assert "data_completeness" in str(evaluation["reasons"])

    # 修复证据：全部成功且对账通过、成本偏差在阈值内
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(simulation_batches)
            .where(simulation_batches.c.portfolio_id == portfolio_id)
            .values(
                status="succeeded",
                summary_json={"conservation": {"cash_difference": 0.0}},
            )
        )
    evaluation = promotion.evaluate_forward_gate(version_id)
    assert evaluation["passed"] is True
    assert evaluation["checks"]["cost_deviation"]["observed"] >= 0.0

    # 人工点是四人眼：批准人不能自批晋升
    with pytest.raises(ValueError, match="second operator"):
        promotion.promote(
            version_id,
            actor="allocation-risk-owner",
            reason="Promote after the forward gate passed.",
        )
    result = promotion.promote(
        version_id, actor="second-operator", reason="Promote after the forward gate passed."
    )
    assert result["promotion_stage"] == "recommendation_enabled"
    version = StrategyStore(database_url).get_version(version_id)
    assert version["promotion_stage"] == "recommendation_enabled"


def test_cost_deviation_subitem(database_url: str, tmp_path: Path, monkeypatch) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch, max_cost_deviation=0.0001
    )
    _seed_evidence(
        promotion,
        stage["simulation_portfolio_id"],
        nav_days=4,
        succeeded=3,
        sell_batches=1,
        fee=50.0,  # 5% 费率，远超成本表
        gross=1000.0,
    )
    evaluation = promotion.evaluate_forward_gate(version_id)
    assert evaluation["passed"] is False
    assert evaluation["checks"]["cost_deviation"]["passed"] is False
    assert evaluation["checks"]["cost_deviation"]["observed"] > 0.0001


# ---------------------------------------------------------------------------
# Stage reset on contract drift (design 9.5: no evidence concatenation)
# ---------------------------------------------------------------------------


def test_contract_drift_freezes_stage_and_starts_from_zero(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    portfolio_id = stage["simulation_portfolio_id"]
    _seed_evidence(
        promotion, portfolio_id, nav_days=10, succeeded=10, sell_batches=2
    )
    assert promotion.evaluate_forward_gate(version_id)["passed"] is True
    # 来源合同漂移：成本表版本与冻结政策不一致
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(simulation_portfolios)
            .where(simulation_portfolios.c.id == portfolio_id)
            .values(cost_schedule_version="cn-effective-cost-2005-01-24")
        )
    evaluation = promotion.evaluate_forward_gate(version_id)
    assert evaluation["passed"] is False
    assert evaluation.get("stage_reset") is True
    stages = promotion.list_stages(version_id)
    assert [item["status"] for item in stages] == ["frozen", "awaiting_simulation"]
    assert stages[0]["freeze_reason"]
    # 旧阶段证据不拼接：新阶段无账户、证据从零
    assert stages[1]["simulation_portfolio_id"] is None
    fresh = promotion.evaluate_forward_gate(version_id)
    assert fresh["passed"] is False
    # 冻结阶段只读：opened/frozen 时间戳已落
    assert stages[0]["frozen_at"] is not None


# ---------------------------------------------------------------------------
# Recommendation chain tightening
# ---------------------------------------------------------------------------


def test_paper_version_cannot_create_standalone_recommendation(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    recommendations = RecommendationStore(database_url)
    with pytest.raises(ValueError, match="paper stage"):
        recommendations.create(
            name="paper recommendation",
            strategy_version_id=version_id,
            dataset="allocation-data",
            hypothetical_initial_value=1_000_000,
            actor="operator-a",
        )
    _seed_evidence(
        promotion,
        stage["simulation_portfolio_id"],
        nav_days=4,
        succeeded=3,
        sell_batches=1,
        fee=0.0,
        gross=1000.0,
    )
    promotion.promote(
        version_id, actor="second-operator", reason="Promote after the forward gate passed."
    )
    portfolio = recommendations.create(
        name="enabled recommendation",
        strategy_version_id=version_id,
        dataset="allocation-data",
        hypothetical_initial_value=1_000_000,
        actor="operator-a",
    )
    assert portfolio["status"] == "active"
