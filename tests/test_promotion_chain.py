"""Promotion chain: paper stage, forward evidence gate, recommendation gating."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from governance_fixtures import (
    PERIODS,
    create_promoted_factor,
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
)

import quant_platform.promotion as promotion_module
import quant_platform.risk_math as risk_math
import quant_platform.strategy_allocation as strategy_allocation
from quant_data.database import (
    backtest_runs,
    open_database,
    simulation_batches,
    simulation_fills,
    simulation_nav,
    simulation_orders,
    simulation_portfolios,
    strategy_health_snapshots,
    strategy_promotion_stages,
    strategy_versions,
)
from quant_data.history_bounds import GOVERNED_DAILY_STOCK_SCOPE_VERSION
from quant_platform.promotion import (
    ForwardGateThresholds,
    PromotionStore,
    build_forward_gate_criteria,
    forward_gate_thresholds_for_horizon,
)
from quant_platform.recommendation_store import RecommendationStore
from quant_platform.research_horizon import SHORT_1_5D, canonical_sha256
from quant_platform.simulation_store import SimulationStore
from quant_platform.strategy_store import StrategyStore

ACTOR = "promotion-operator"
SHANGHAI = ZoneInfo("Asia/Shanghai")
FIRST_FINAL_PERIODS = {**PERIODS, "test_end": date(2023, 9, 29)}
SECOND_FINAL_PERIODS = {**PERIODS, "test_start": date(2023, 10, 2)}


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


def _governed_daily_dataset() -> dict:
    dataset = _daily_dataset()
    dataset["end_date"] = promotion_module._now().astimezone(SHANGHAI).date().isoformat()
    dataset["provenance"]["execution_controls"] = {
        "formal_execution_requires_native_controls": True,
        "scope_version": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
        "native_complete_from": "2008-01-01",
    }
    return dataset


def _paper_version(database_url: str, tmp_path: Path, monkeypatch) -> str:
    """A real-approved day-execution version (paper, awaiting stage)."""

    _qlib_doubles(monkeypatch)
    return _approve_version(database_url, tmp_path, suffix="promotion", returns=_returns())


def _short_version(
    database_url: str,
    tmp_path: Path,
    *,
    suffix: str,
    with_datasets: bool = False,
    parent_version_id: str | None = None,
    periods: dict | None = None,
) -> str:
    """A real-approved short-horizon version using daily next-open execution."""

    strategies = StrategyStore(database_url)
    periods = periods or PERIODS
    if parent_version_id is None:
        version_id = create_strategy_version(
            database_url,
            tmp_path,
            dataset="allocation-data",
            periods=periods,
            recipe_id="short_relative_strength",
        )
    else:
        parent = strategies.get_version(parent_version_id)
        factor = create_promoted_factor(
            database_url,
            tmp_path,
            dataset="allocation-data",
            periods=periods,
        )
        created = strategies.create_version(
            str(parent["strategy_id"]),
            benchmark=str(parent["benchmark"]),
            universe=str(parent["universe"]),
            factors=[{"candidate_id": str(factor["id"]), "weight": 1.0}],
            config=dict(parent["config"]),
            actor="promotion-test",
        )
        version_id = str(created["id"])
    version = strategies.get_version(version_id)
    artifact = tmp_path / f"short-backtest-{suffix}"
    artifact.mkdir()
    backtest_periods = {
        "start": periods["test_start"].isoformat(),
        "end": periods["test_end"].isoformat(),
    }
    backtest = strategies.create_backtest(
        version_id=version_id,
        dataset="allocation-data",
        periods=backtest_periods,
        artifact_path=artifact,
    )
    factor = version["factors"][0]
    manifest = artifact / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy_version_id": version_id,
                "dataset": "allocation-data",
                "benchmark": version["benchmark"],
                "universe": version["universe"],
                "periods": backtest_periods,
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
    if with_datasets:
        (artifact / "datasets.json").write_text(
            json.dumps(
                {
                    "daily": _governed_daily_dataset(),
                    "execution": _governed_daily_dataset(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    metrics = formal_backtest_metrics(
        version,
        manifest,
        hypothesis_group_evidence=strategies.hypothesis_group_evidence(version_id),
    )
    strategies.validate_backtest_artifacts(backtest["id"], metrics)
    strategies.mark_backtest(backtest["id"], "succeeded", metrics=metrics)
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
        daily_dataset=_governed_daily_dataset(),
        execution_dataset=_governed_daily_dataset(),
        initial_cash=100_000,
    )
    assert stage["status"] == "active"
    return stage


def _register_gate(store: PromotionStore, version_id: str, **overrides) -> None:
    values = asdict(forward_gate_thresholds_for_horizon(SHORT_1_5D))
    values.update(
        {
            "min_data_completeness": 0.8,
            "min_reconciliation_rate": 1.0,
            "max_cost_deviation": 0.01,
        }
    )
    values.update(overrides)
    thresholds = ForwardGateThresholds(**values)
    engine = store.engine
    with engine.begin() as connection:
        # Approval now registers the immutable default gate before opening the
        # stage. Evidence tests replace the synthetic fixture values directly;
        # the public mutation guard is tested separately below.
        version = connection.execute(
            select(strategy_versions).where(strategy_versions.c.id == version_id)
        ).one()
        criteria = build_forward_gate_criteria(
            horizon_profile=str(version.horizon_profile),
            horizon_contract_sha256=str(version.horizon_contract_sha256),
            thresholds=thresholds,
        )
        connection.execute(
            update(_gate_table())
            .where(_gate_table().c.strategy_version_id == version_id)
            .values(
                **asdict(thresholds),
                criteria_json=criteria,
                criteria_sha256=canonical_sha256(criteria),
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
    valid_round_trips: int = 0,
    backdate_stage: bool = True,
    fee: float = 0.0,
    gross: float = 1.0,
) -> None:
    engine = store.engine
    with engine.begin() as connection:
        promotion_stage = connection.execute(
            select(strategy_promotion_stages).where(
                strategy_promotion_stages.c.simulation_portfolio_id == portfolio_id
            )
        ).one()
        paper_account = connection.execute(
            select(simulation_portfolios).where(
                simulation_portfolios.c.id == portfolio_id
            )
        ).one()
        strategy_version = connection.execute(
            select(strategy_versions).where(
                strategy_versions.c.id == promotion_stage.strategy_version_id
            )
        ).one()
        signal_frequency = str(
            strategy_version.signal_frequency
            or dict(strategy_version.config_json or {}).get("signal_frequency")
            or "day"
        ).lower()
        is_daily_signal = signal_frequency == "day"
        formal_backtest_id = connection.scalar(
            select(backtest_runs.c.id)
            .where(
                backtest_runs.c.strategy_version_id
                == promotion_stage.strategy_version_id,
                backtest_runs.c.status == "succeeded",
                backtest_runs.c.is_legacy.is_(False),
            )
            .order_by(backtest_runs.c.created_at.desc())
            .limit(1)
        )
        assert formal_backtest_id is not None
        observed_at = promotion_module._now()
        opened_at = promotion_stage.opened_at
        evidence_days = max(nav_days, succeeded + failed, 1)
        if backdate_stage:
            opened_at = observed_at - timedelta(days=evidence_days * 2 + 7)
            connection.execute(
                update(strategy_promotion_stages)
                .where(strategy_promotion_stages.c.id == promotion_stage.id)
                .values(opened_at=opened_at)
            )
            connection.execute(
                update(_gate_table())
                .where(
                    _gate_table().c.strategy_version_id
                    == promotion_stage.strategy_version_id
                )
                .values(
                    registered_at=opened_at - timedelta(seconds=1),
                    updated_at=opened_at - timedelta(seconds=1),
                )
            )
            connection.execute(
                update(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .values(created_at=opened_at, updated_at=opened_at)
            )
        sessions = [
            value.date()
            for value in pd.bdate_range(
                start=opened_at.astimezone(SHANGHAI).date() + timedelta(days=1),
                periods=evidence_days + 1,
            )
        ]
        for index in range(nav_days):
            nav_created_at = datetime.combine(
                sessions[index + 1],
                time(7, 5),
                tzinfo=UTC,
            )
            connection.execute(
                insert(simulation_nav).values(
                    portfolio_id=portfolio_id,
                    trade_date=sessions[index + 1],
                    cash=paper_account.initial_cash,
                    market_value=Decimal("0"),
                    nav=paper_account.initial_cash,
                    daily_return=0.0,
                    drawdown=0.0,
                    market_date=sessions[index + 1],
                    has_stale_prices=False,
                    status="certified",
                    performance_certified=True,
                    created_at=nav_created_at,
                )
            )
        batch_ids: list[tuple[str, date]] = []
        for index in range(succeeded + failed):
            batch_id = uuid.uuid4().hex
            ok = index < succeeded
            reconciled = not ok or index >= unreconciled
            summary = (
                {"conservation": {"cash_difference": 0.0 if reconciled else 12.5}}
                if ok
                else None
            )
            batch_created_at = datetime.combine(
                sessions[index],
                time(7, 0),
                tzinfo=UTC,
            )
            batch_started_at = datetime.combine(
                sessions[index + 1],
                time(7, 30),
                tzinfo=UTC,
            )
            batch_finished_at = datetime.combine(
                sessions[index + 1],
                time(7, 31),
                tzinfo=UTC,
            )
            signal_at = None if is_daily_signal else batch_created_at
            execution_not_before = (
                None
                if is_daily_signal
                else datetime.combine(
                    sessions[index + 1],
                    time(1, 30),
                    tzinfo=UTC,
                )
            )
            dataset_bindings = {
                "daily_dataset": str(paper_account.daily_dataset),
                "daily_dataset_identity_sha256": str(
                    paper_account.daily_dataset_identity_sha256
                ),
                "daily_dataset_lineage_id": str(
                    paper_account.daily_dataset_lineage_id
                ),
                "execution_dataset": str(paper_account.execution_dataset),
                "execution_dataset_identity_sha256": str(
                    paper_account.execution_dataset_identity_sha256
                ),
                "execution_dataset_lineage_id": str(
                    paper_account.execution_dataset_lineage_id
                ),
            }
            connection.execute(
                insert(simulation_batches).values(
                    id=batch_id,
                    portfolio_id=portfolio_id,
                    execution_contract_hash=str(
                        paper_account.execution_contract_hash
                    ),
                    daily_dataset=str(paper_account.daily_dataset),
                    daily_dataset_identity_sha256=str(
                        paper_account.daily_dataset_identity_sha256
                    ),
                    daily_dataset_lineage_id=str(
                        paper_account.daily_dataset_lineage_id
                    ),
                    execution_dataset=str(paper_account.execution_dataset),
                    execution_dataset_identity_sha256=str(
                        paper_account.execution_dataset_identity_sha256
                    ),
                    execution_dataset_lineage_id=str(
                        paper_account.execution_dataset_lineage_id
                    ),
                    simulation_semantics_sha256=(
                        SimulationStore._batch_simulation_semantics_sha256(
                            paper_account, dataset_bindings
                        )
                    ),
                    source_snapshot_id=str(
                        paper_account.daily_dataset_identity_sha256
                    ),
                    target_payload_json={
                        "governed_order_plan": {
                            "format_version": "qlib-order-plan-v1",
                            "manifest_sha256": f"{index + 1:064x}",
                            "formal_backtest_id": str(formal_backtest_id),
                            "promotion_stage_id": str(promotion_stage.id),
                            "promotion_stage_opened_at": opened_at.isoformat(),
                            "execution_contract_hash": str(
                                paper_account.execution_contract_hash
                            ),
                            "signal_at": (
                                None if signal_at is None else signal_at.isoformat()
                            ),
                            "execution_not_before": (
                                None
                                if execution_not_before is None
                                else execution_not_before.isoformat()
                            ),
                            "source_snapshot": {
                                "id": str(
                                    paper_account.daily_dataset_identity_sha256
                                ),
                                "dataset_identity_sha256": str(
                                    paper_account.daily_dataset_identity_sha256
                                ),
                                "dataset_lineage_id": str(
                                    paper_account.daily_dataset_lineage_id
                                ),
                            },
                        }
                    },
                    signal_date=sessions[index],
                    trade_date=sessions[index + 1],
                    signal_at=signal_at,
                    execution_not_before=execution_not_before,
                    status="succeeded" if ok else "failed",
                    idempotency_key=f"test-{batch_id}",
                    summary_json=summary,
                    created_at=batch_created_at,
                    started_at=batch_started_at,
                    finished_at=batch_finished_at,
                )
            )
            if ok:
                batch_ids.append((batch_id, sessions[index + 1]))
        for index, (batch_id, trade_date) in enumerate(batch_ids):
            if index < valid_round_trips * 2:
                side = "buy" if index % 2 == 0 else "sell"
            else:
                side = (
                    "sell"
                    if index < valid_round_trips * 2 + sell_batches
                    else "buy"
                )
            order_id = uuid.uuid4().hex
            connection.execute(
                insert(simulation_orders).values(
                    id=order_id,
                    batch_id=batch_id,
                    portfolio_id=portfolio_id,
                    instrument="SH600000",
                    side=side,
                    target_weight=0.1,
                    requested_quantity=100,
                    filled_quantity=100,
                    status="filled",
                    requested_value=Decimal("1000"),
                    filled_value=Decimal("1000"),
                    capacity_fill_ratio=1.0,
                    expires_at=datetime.combine(
                        trade_date + timedelta(days=1),
                        time(7, 0),
                        tzinfo=UTC,
                    ),
                    created_at=datetime.combine(
                        trade_date,
                        time(7, 30),
                        tzinfo=UTC,
                    ),
                )
            )
            connection.execute(
                insert(simulation_fills).values(
                    id=uuid.uuid4().hex,
                    order_id=order_id,
                    batch_id=batch_id,
                    instrument="SH600000",
                    side=side,
                    executed_at=datetime.combine(
                        trade_date,
                        time(7, 0),
                        tzinfo=UTC,
                    ),
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
        version_id,
        actor=ACTOR,
        thresholds=ForwardGateThresholds(
            **{
                **asdict(forward_gate_thresholds_for_horizon(SHORT_1_5D)),
                "min_forward_calendar_days": 30,
            }
        ),
    )
    assert gate["min_forward_calendar_days"] == 30
    gate = promotion.register_forward_gate(
        version_id,
        actor=ACTOR,
        thresholds=ForwardGateThresholds(
            **{
                **asdict(forward_gate_thresholds_for_horizon(SHORT_1_5D)),
                "min_forward_calendar_days": 45,
            }
        ),
    )
    assert gate["min_forward_calendar_days"] == 45
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved", promotion_stage="paper")
        )
    promotion.open_paper_stage(version_id, actor=ACTOR)
    with pytest.raises(ValueError, match="immutable after"):
        promotion.register_forward_gate(version_id, actor=ACTOR)


def test_automatic_paper_open_preserves_preregistered_gate(
    database_url: str, tmp_path: Path
) -> None:
    version_id = create_strategy_version(database_url, tmp_path)
    promotion = PromotionStore(database_url)
    promotion.register_forward_gate(
        version_id,
        actor=ACTOR,
        thresholds=ForwardGateThresholds(
            **{
                **asdict(forward_gate_thresholds_for_horizon(SHORT_1_5D)),
                "min_forward_calendar_days": 45,
            }
        ),
    )
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved", promotion_stage="paper")
        )
    promotion.prepare_paper_stage(version_id, actor=ACTOR)
    with engine.connect() as connection:
        gate = connection.execute(
            select(_gate_table()).where(
                _gate_table().c.strategy_version_id == version_id
            )
        ).one()
    assert int(gate.min_forward_calendar_days) == 45


def test_attach_creates_isolated_paper_account(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    _qlib_doubles(monkeypatch)
    version_id = _short_version(database_url, tmp_path, suffix="attach")
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
    assert str(portfolio.promotion_stage_id) == stage["id"]
    assert float(portfolio.initial_cash) == 100_000
    assert float(portfolio.initial_cash) != 5_000_000
    assert stage["initial_cash"] == 100_000
    assert stage["source_contract_hash"] == str(portfolio.execution_contract_hash)


def test_approve_waits_for_investor_capital_before_creating_paper_account(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    """Approval opens the gate but never invents a novice user's principal."""

    _qlib_doubles(monkeypatch)
    version_id = _short_version(database_url, tmp_path, suffix="auto", with_datasets=True)
    promotion = PromotionStore(database_url)
    stage = promotion.current_stage(version_id)
    assert stage["status"] == "awaiting_simulation"
    assert stage["simulation_portfolio_id"] is None
    stage = _attach_simulation(promotion, version_id)
    assert stage["status"] == "active"
    assert stage["simulation_portfolio_id"]
    assert stage["source_contract_hash"]
    # 幂等：重复打开不建第二个账户
    again = promotion.open_paper_stage(version_id, actor=ACTOR)
    assert again["id"] == stage["id"]


def test_new_autopilot_champion_pauses_prior_primary_without_deleting_history(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    _qlib_doubles(monkeypatch)
    promotion = PromotionStore(database_url)
    engine = open_database(database_url)

    def mark_autopilot(version_id: str) -> None:
        with engine.begin() as connection:
            config = dict(
                connection.scalar(
                    select(strategy_versions.c.config_json).where(
                        strategy_versions.c.id == version_id
                    )
                )
                or {}
            )
            config["autopilot_completion_contract_version"] = (
                "autopilot-completion-v1"
            )
            connection.execute(
                update(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .values(config_json=config)
            )

    first_version = _short_version(
        database_url,
        tmp_path,
        suffix="primary-one",
        periods=FIRST_FINAL_PERIODS,
    )
    mark_autopilot(first_version)
    first_stage = _attach_simulation(promotion, first_version)

    second_version = _short_version(
        database_url,
        tmp_path,
        suffix="primary-two",
        periods=SECOND_FINAL_PERIODS,
    )
    mark_autopilot(second_version)
    second_stage = _attach_simulation(promotion, second_version)

    with engine.connect() as connection:
        stages = {
            str(row.strategy_version_id): row
            for row in connection.execute(
                select(strategy_promotion_stages).where(
                    strategy_promotion_stages.c.strategy_version_id.in_(
                        [first_version, second_version]
                    )
                )
            )
        }
        portfolios = {
            str(row.id): row
            for row in connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id.in_(
                        [
                            first_stage["simulation_portfolio_id"],
                            second_stage["simulation_portfolio_id"],
                        ]
                    )
                )
            )
        }

    assert str(stages[first_version].status) == "frozen"
    assert str(stages[second_version].status) == "active"
    assert str(portfolios[first_stage["simulation_portfolio_id"]].status) == "paused"
    assert str(portfolios[second_stage["simulation_portfolio_id"]].status) == "active"
    assert len(portfolios) == 2  # the superseded ledger is retained and auditable


def test_new_paper_candidate_does_not_pause_recommendation_incumbent(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    _qlib_doubles(monkeypatch)
    promotion = PromotionStore(database_url)
    engine = open_database(database_url)

    def mark_autopilot(version_id: str) -> None:
        with engine.begin() as connection:
            config = dict(
                connection.scalar(
                    select(strategy_versions.c.config_json).where(
                        strategy_versions.c.id == version_id
                    )
                )
                or {}
            )
            config["autopilot_completion_contract_version"] = (
                "autopilot-completion-v1"
            )
            connection.execute(
                update(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .values(config_json=config)
            )

    incumbent = _short_version(
        database_url,
        tmp_path,
        suffix="enabled-incumbent",
        periods=FIRST_FINAL_PERIODS,
    )
    mark_autopilot(incumbent)
    _register_gate(promotion, incumbent)
    incumbent_stage = _attach_simulation(promotion, incumbent)
    _seed_evidence(
        promotion,
        incumbent_stage["simulation_portfolio_id"],
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
    )
    promotion.promote(
        incumbent,
        actor="system:auto-promotion",
        reason="Seed a fully governed recommendation incumbent for the test.",
    )

    challenger = _short_version(
        database_url,
        tmp_path,
        suffix="paper-challenger",
        periods=SECOND_FINAL_PERIODS,
    )
    mark_autopilot(challenger)
    challenger_stage = _attach_simulation(promotion, challenger)

    with engine.connect() as connection:
        incumbent_account = connection.execute(
            select(simulation_portfolios).where(
                simulation_portfolios.c.id
                == incumbent_stage["simulation_portfolio_id"]
            )
        ).one()
        incumbent_promotion_stage = connection.execute(
            select(strategy_promotion_stages).where(
                strategy_promotion_stages.c.id == incumbent_stage["id"]
            )
        ).one()
        challenger_account = connection.execute(
            select(simulation_portfolios).where(
                simulation_portfolios.c.id
                == challenger_stage["simulation_portfolio_id"]
            )
        ).one()

    assert str(incumbent_account.status) == "active"
    assert str(incumbent_promotion_stage.status) == "active"
    assert str(challenger_account.status) == "active"


def test_same_family_paper_candidate_keeps_incumbent_until_final_promotion(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    _qlib_doubles(monkeypatch)
    promotion = PromotionStore(database_url)
    engine = open_database(database_url)

    incumbent = _short_version(
        database_url,
        tmp_path,
        suffix="same-family-incumbent",
        periods=FIRST_FINAL_PERIODS,
    )
    _register_gate(promotion, incumbent)
    incumbent_stage = _attach_simulation(promotion, incumbent)
    _seed_evidence(
        promotion,
        incumbent_stage["simulation_portfolio_id"],
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
    )
    promotion.promote(
        incumbent,
        actor="system:auto-promotion",
        reason="Seed a fully governed same-family incumbent for the test.",
    )

    challenger = _short_version(
        database_url,
        tmp_path,
        suffix="same-family-challenger",
        parent_version_id=incumbent,
        periods=SECOND_FINAL_PERIODS,
    )
    with engine.connect() as connection:
        incumbent_before = connection.execute(
            select(strategy_versions).where(strategy_versions.c.id == incumbent)
        ).one()
        challenger_before = connection.execute(
            select(strategy_versions).where(strategy_versions.c.id == challenger)
        ).one()
    assert str(incumbent_before.strategy_id) == str(challenger_before.strategy_id)
    assert str(incumbent_before.status) == "approved"
    assert str(incumbent_before.promotion_stage) == "recommendation_enabled"
    assert str(challenger_before.status) == "approved"
    assert str(challenger_before.promotion_stage) == "paper"

    _register_gate(promotion, challenger)
    stage = _attach_simulation(promotion, challenger)
    _seed_evidence(
        promotion,
        stage["simulation_portfolio_id"],
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
    )
    promoted = promotion.promote(
        challenger,
        actor="system:auto-promotion",
        reason="All sealed historical and forward gates passed.",
    )
    assert promoted["promotion_stage"] == "recommendation_enabled"

    with engine.connect() as connection:
        incumbent_after = connection.execute(
            select(strategy_versions).where(strategy_versions.c.id == incumbent)
        ).one()
        challenger_after = connection.execute(
            select(strategy_versions).where(strategy_versions.c.id == challenger)
        ).one()
    assert str(incumbent_after.status) == "retired"
    assert str(challenger_after.status) == "approved"
    assert str(challenger_after.promotion_stage) == "recommendation_enabled"


# ---------------------------------------------------------------------------
# Forward evidence gate
# ---------------------------------------------------------------------------


def _gated_paper_version(database_url: str, tmp_path: Path, monkeypatch, **gate) -> tuple:
    _qlib_doubles(monkeypatch)
    version_id = _short_version(database_url, tmp_path, suffix="gated")
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


def test_future_dated_rows_never_count_as_forward_evidence(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    _seed_evidence(
        promotion,
        stage["simulation_portfolio_id"],
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
        backdate_stage=False,
    )

    evaluation = promotion.evaluate_forward_gate(version_id)

    assert evaluation["passed"] is False
    assert evaluation["evidence"]["forward_trading_days"] == 0
    assert evaluation["evidence"]["decision_batches"] == 0
    assert evaluation["evidence"]["closed_round_trips"] == 0


def test_future_fill_does_not_complete_a_forward_round_trip(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    portfolio_id = stage["simulation_portfolio_id"]
    _seed_evidence(
        promotion,
        portfolio_id,
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
    )
    engine = open_database(database_url)
    with engine.begin() as connection:
        batch = connection.execute(
            select(simulation_batches)
            .where(
                simulation_batches.c.portfolio_id == portfolio_id,
                simulation_batches.c.status == "succeeded",
            )
            .order_by(simulation_batches.c.trade_date)
            .limit(1)
        ).one()
        order_id = uuid.uuid4().hex
        connection.execute(
            insert(simulation_orders).values(
                id=order_id,
                batch_id=str(batch.id),
                portfolio_id=portfolio_id,
                instrument="SH600002",
                side="buy",
                target_weight=0.1,
                requested_quantity=100,
                filled_quantity=100,
                status="filled",
                requested_value=Decimal("1000"),
                filled_value=Decimal("1000"),
                capacity_fill_ratio=1.0,
                expires_at=promotion_module._now() + timedelta(days=6),
                created_at=batch.started_at,
            )
        )
        connection.execute(
            insert(simulation_fills).values(
                id=uuid.uuid4().hex,
                order_id=order_id,
                batch_id=str(batch.id),
                instrument="SH600002",
                side="buy",
                executed_at=promotion_module._now() + timedelta(days=5),
                quantity=100,
                price=Decimal("10"),
                gross_value=Decimal("1000"),
                fee=Decimal("0"),
                cost_breakdown_json={},
                minute_volume=1_000_000,
                capacity_quantity=1_000_000,
            )
        )

    evaluation = promotion.evaluate_forward_gate(version_id)

    assert evaluation["passed"] is False
    assert evaluation["evidence"]["forward_trading_days"] == 90
    assert evaluation["evidence"]["closed_round_trips"] == 30
    assert evaluation["evidence"]["invalid_fill_contract_rows"] == 1
    assert evaluation["checks"]["closed_round_trips"]["passed"] is True
    assert evaluation["checks"]["governed_batch_integrity"]["passed"] is False


def test_minute_fill_before_execution_boundary_never_counts(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    portfolio_id = stage["simulation_portfolio_id"]
    engine = open_database(database_url)
    with engine.begin() as connection:
        version = connection.execute(
            select(strategy_versions).where(strategy_versions.c.id == version_id)
        ).one()
        config = dict(version.config_json or {})
        config.update(signal_frequency="5min", execution_frequency="5min")
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(
                signal_frequency="5min",
                execution_frequency="5min",
                config_json=config,
            )
        )
        connection.execute(
            update(simulation_portfolios)
            .where(simulation_portfolios.c.id == portfolio_id)
            .values(execution_frequency="5min")
        )
    _seed_evidence(
        promotion,
        portfolio_id,
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
    )
    with engine.begin() as connection:
        fill = connection.execute(
            select(
                simulation_fills.c.id,
                simulation_batches.c.trade_date,
                simulation_batches.c.execution_not_before,
            )
            .join(
                simulation_batches,
                simulation_batches.c.id == simulation_fills.c.batch_id,
            )
            .where(simulation_batches.c.portfolio_id == portfolio_id)
            .order_by(simulation_batches.c.trade_date)
            .limit(1)
        ).one()
        assert fill.execution_not_before is not None
        too_early = datetime.combine(fill.trade_date, time(1, 0), tzinfo=UTC)
        assert too_early < fill.execution_not_before
        connection.execute(
            update(simulation_fills)
            .where(simulation_fills.c.id == fill.id)
            .values(executed_at=too_early)
        )

    with engine.connect() as connection:
        evidence = promotion._collect_evidence(
            connection,
            connection.execute(
                select(strategy_promotion_stages).where(
                    strategy_promotion_stages.c.id == stage["id"]
                )
            ).one(),
            connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == portfolio_id
                )
            ).one(),
            connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == version_id)
            ).one(),
        )

    assert evidence["invalid_lifecycle_batches"] == 0
    assert evidence["invalid_fill_contract_rows"] == 1


def test_delayed_wall_clock_replay_keeps_valid_forward_evidence(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    portfolio_id = stage["simulation_portfolio_id"]
    _seed_evidence(
        promotion,
        portfolio_id,
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
    )
    engine = open_database(database_url)
    with engine.begin() as connection:
        batch = connection.execute(
            select(simulation_batches)
            .where(
                simulation_batches.c.portfolio_id == portfolio_id,
                simulation_batches.c.status == "succeeded",
            )
            .order_by(simulation_batches.c.trade_date)
            .limit(1)
        ).one()
        delayed_start = datetime.combine(
            batch.trade_date + timedelta(days=3), time(7, 30), tzinfo=UTC
        )
        connection.execute(
            update(simulation_batches)
            .where(simulation_batches.c.id == batch.id)
            .values(
                started_at=delayed_start,
                finished_at=delayed_start + timedelta(minutes=1),
            )
        )

    evaluation = promotion.evaluate_forward_gate(version_id)

    assert evaluation["passed"] is True
    assert evaluation["evidence"]["invalid_lifecycle_batches"] == 0


def test_fill_must_match_its_order_and_earliest_window_end(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    portfolio_id = stage["simulation_portfolio_id"]
    _seed_evidence(
        promotion,
        portfolio_id,
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
    )
    engine = open_database(database_url)
    with engine.begin() as connection:
        fills = connection.execute(
            select(
                simulation_fills.c.id,
                simulation_fills.c.order_id,
                simulation_fills.c.executed_at,
            )
            .join(
                simulation_batches,
                simulation_batches.c.id == simulation_fills.c.batch_id,
            )
            .where(simulation_batches.c.portfolio_id == portfolio_id)
            .order_by(simulation_batches.c.trade_date)
            .limit(2)
        ).all()
        connection.execute(
            update(simulation_fills)
            .where(simulation_fills.c.id == fills[0].id)
            .values(instrument="SH600099")
        )
        connection.execute(
            update(simulation_orders)
            .where(simulation_orders.c.id == fills[1].order_id)
            .values(
                expires_at=fills[1].executed_at - timedelta(minutes=1),
                not_after=fills[1].executed_at + timedelta(days=1),
            )
        )

    evaluation = promotion.evaluate_forward_gate(version_id)

    assert evaluation["passed"] is False
    assert evaluation["evidence"]["invalid_fill_contract_rows"] == 2
    assert evaluation["checks"]["governed_batch_integrity"]["passed"] is False


def test_duplicate_signal_dates_block_forward_promotion(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    portfolio_id = stage["simulation_portfolio_id"]
    _seed_evidence(
        promotion,
        portfolio_id,
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
    )
    engine = open_database(database_url)
    with engine.begin() as connection:
        batches = connection.execute(
            select(simulation_batches)
            .where(
                simulation_batches.c.portfolio_id == portfolio_id,
                simulation_batches.c.status == "succeeded",
            )
            .order_by(simulation_batches.c.signal_date)
            .limit(2)
        ).all()
        connection.execute(
            update(simulation_batches)
            .where(simulation_batches.c.id == batches[1].id)
            .values(signal_date=batches[0].signal_date)
        )

    evaluation = promotion.evaluate_forward_gate(version_id)

    assert evaluation["passed"] is False
    assert evaluation["evidence"]["decision_batches"] == 89
    assert evaluation["evidence"]["duplicate_decision_batches"] == 1
    assert evaluation["checks"]["governed_batch_integrity"]["passed"] is False


def test_incomplete_succeeded_batch_lifecycle_blocks_forward_promotion(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    portfolio_id = stage["simulation_portfolio_id"]
    _seed_evidence(
        promotion,
        portfolio_id,
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
    )
    engine = open_database(database_url)
    with engine.begin() as connection:
        batch_id = connection.scalar(
            select(simulation_batches.c.id)
            .where(
                simulation_batches.c.portfolio_id == portfolio_id,
                simulation_batches.c.status == "succeeded",
            )
            .order_by(simulation_batches.c.trade_date.desc())
            .limit(1)
        )
        connection.execute(
            update(simulation_batches)
            .where(simulation_batches.c.id == batch_id)
            .values(finished_at=None)
        )

    evaluation = promotion.evaluate_forward_gate(version_id)

    assert evaluation["passed"] is False
    assert evaluation["evidence"]["invalid_lifecycle_batches"] == 1
    assert evaluation["checks"]["governed_batch_integrity"]["passed"] is False


def test_same_day_signal_and_trade_batch_never_counts_as_forward_evidence(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    portfolio_id = stage["simulation_portfolio_id"]
    _seed_evidence(
        promotion,
        portfolio_id,
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
    )
    engine = open_database(database_url)
    with engine.begin() as connection:
        batch = connection.execute(
            select(simulation_batches)
            .where(
                simulation_batches.c.portfolio_id == portfolio_id,
                simulation_batches.c.status == "succeeded",
            )
            .order_by(simulation_batches.c.trade_date)
            .limit(1)
        ).one()
        created_at = datetime.combine(batch.signal_date, time(1, 30), tzinfo=UTC)
        started_at = datetime.combine(batch.signal_date, time(2, 0), tzinfo=UTC)
        finished_at = datetime.combine(batch.signal_date, time(4, 0), tzinfo=UTC)
        payload = dict(batch.target_payload_json)
        plan = dict(payload["governed_order_plan"])
        plan.update(
            signal_at=None,
            execution_not_before=None,
        )
        payload["governed_order_plan"] = plan
        connection.execute(
            update(simulation_batches)
            .where(simulation_batches.c.id == batch.id)
            .values(
                trade_date=batch.signal_date,
                signal_at=None,
                execution_not_before=None,
                target_payload_json=payload,
                created_at=created_at,
                started_at=started_at,
                finished_at=finished_at,
            )
        )
        connection.execute(
            update(simulation_fills)
            .where(simulation_fills.c.batch_id == batch.id)
            .values(
                executed_at=datetime.combine(
                    batch.signal_date, time(3, 0), tzinfo=UTC
                )
            )
        )

    evaluation = promotion.evaluate_forward_gate(version_id)

    assert evaluation["passed"] is False
    assert evaluation["evidence"]["invalid_lifecycle_batches"] == 1
    assert evaluation["checks"]["governed_batch_integrity"]["passed"] is False


def test_same_day_t_plus_one_fill_blocks_forward_promotion(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    portfolio_id = stage["simulation_portfolio_id"]
    _seed_evidence(
        promotion,
        portfolio_id,
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
    )
    engine = open_database(database_url)
    with engine.begin() as connection:
        batch = connection.execute(
            select(simulation_batches)
            .where(
                simulation_batches.c.portfolio_id == portfolio_id,
                simulation_batches.c.status == "succeeded",
            )
            .order_by(simulation_batches.c.trade_date)
            .limit(1)
        ).one()
        for side, execution_time in (("buy", time(2, 0)), ("sell", time(3, 0))):
            order_id = uuid.uuid4().hex
            connection.execute(
                insert(simulation_orders).values(
                    id=order_id,
                    batch_id=str(batch.id),
                    portfolio_id=portfolio_id,
                    instrument="SH600001",
                    side=side,
                    target_weight=0.1,
                    requested_quantity=100,
                    filled_quantity=100,
                    status="filled",
                    requested_value=Decimal("1000"),
                    filled_value=Decimal("1000"),
                    capacity_fill_ratio=1.0,
                    expires_at=batch.finished_at + timedelta(days=1),
                    created_at=batch.started_at,
                )
            )
            connection.execute(
                insert(simulation_fills).values(
                    id=uuid.uuid4().hex,
                    order_id=order_id,
                    batch_id=str(batch.id),
                    instrument="SH600001",
                    side=side,
                    executed_at=datetime.combine(
                        batch.trade_date,
                        execution_time,
                        tzinfo=UTC,
                    ),
                    quantity=100,
                    price=Decimal("10"),
                    gross_value=Decimal("1000"),
                    fee=Decimal("0"),
                    cost_breakdown_json={},
                    minute_volume=1_000_000,
                    capacity_quantity=1_000_000,
                )
            )

    evaluation = promotion.evaluate_forward_gate(version_id)

    assert evaluation["passed"] is False
    assert evaluation["evidence"]["closed_round_trips"] == 30
    assert evaluation["evidence"]["invalid_round_trip_fills"] == 1
    assert evaluation["checks"]["governed_batch_integrity"]["passed"] is False


def test_preopen_and_ungoverned_replay_never_become_forward_evidence(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    portfolio_id = stage["simulation_portfolio_id"]
    _seed_evidence(
        promotion,
        portfolio_id,
        nav_days=1,
        succeeded=1,
        sell_batches=1,
    )
    current_stage = promotion.current_stage(version_id)
    opened_at = datetime.fromisoformat(current_stage["opened_at"])
    opened_date = opened_at.astimezone(SHANGHAI).date()
    engine = open_database(database_url)
    with engine.begin() as connection:
        # Even a correctly-shaped batch/NAV is historical evidence when its
        # decision and durable creation predate this promotion stage.
        connection.execute(
            update(simulation_batches)
            .where(simulation_batches.c.portfolio_id == portfolio_id)
            .values(
                signal_date=opened_date,
                trade_date=opened_date,
                created_at=opened_at - timedelta(seconds=1),
            )
        )
        connection.execute(
            update(simulation_nav)
            .where(simulation_nav.c.portfolio_id == portfolio_id)
            .values(
                trade_date=opened_date,
                created_at=opened_at - timedelta(seconds=1),
            )
        )
        # A copied final-OOS/generic payload created after opening is still
        # not a governed forward order plan and must block, not count.
        connection.execute(
            insert(simulation_batches).values(
                id=uuid.uuid4().hex,
                portfolio_id=portfolio_id,
                execution_contract_hash="a" * 64,
                daily_dataset="promotion-daily",
                daily_dataset_identity_sha256="b" * 64,
                daily_dataset_lineage_id="c" * 64,
                execution_dataset="promotion-minute",
                execution_dataset_identity_sha256="d" * 64,
                execution_dataset_lineage_id="e" * 64,
                simulation_semantics_sha256="f" * 64,
                source_snapshot_id="b" * 64,
                target_payload_json={"final_oos_replay": True},
                signal_date=opened_date + timedelta(days=1),
                trade_date=opened_date + timedelta(days=2),
                status="succeeded",
                idempotency_key=f"replay-{uuid.uuid4().hex}",
                summary_json={"conservation": {"cash_difference": 0.0}},
                created_at=datetime.now(UTC),
            )
        )
    evaluation = promotion.evaluate_forward_gate(version_id)
    assert evaluation["passed"] is False
    assert evaluation["evidence"]["decision_batches"] == 0
    assert evaluation["evidence"]["forward_calendar_days"] == 0
    assert evaluation["evidence"]["ungoverned_batches"] == 1
    assert evaluation["checks"]["governed_batch_integrity"]["passed"] is False


def test_paper_signal_must_start_after_stage_open(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    opened_date = datetime.fromisoformat(stage["opened_at"]).astimezone(
        SHANGHAI
    ).date()
    with pytest.raises(ValueError, match="after the promotion stage opened"):
        promotion.require_paper_signal(
            version_id,
            portfolio_id=stage["simulation_portfolio_id"],
            signal_date=opened_date,
        )


def test_gate_subitems_and_atomic_promotion(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch, min_data_completeness=1.0
    )
    portfolio_id = stage["simulation_portfolio_id"]
    _seed_evidence(
        promotion,
        portfolio_id,
        nav_days=90,
        succeeded=89,
        failed=1,
        unreconciled=1,
        valid_round_trips=30,
        fee=3.0,
        gross=1000.0,
    )
    evaluation = promotion.evaluate_forward_gate(version_id)
    checks = evaluation["checks"]
    assert checks["forward_trading_days"]["observed"] == 90
    assert checks["decision_batches"]["observed"] == 89
    assert checks["closed_round_trips"]["observed"] == 30
    assert checks["data_completeness"]["observed"] == pytest.approx(89 / 90)
    # 89 个成功批次中 88 个对账通过：对账率门槛 1.0 → 不足
    assert checks["reconciliation_rate"]["observed"] == pytest.approx(88 / 89)
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
    assert evaluation["evidence"]["invalid_lifecycle_batches"] == 0
    assert evaluation["checks"]["cost_deviation"]["observed"] >= 0.0

    # Scheduler and recovery callers enter the same atomic transaction after
    # the frozen forward gate passes; no second identity can replace evidence.
    result = promotion.promote(
        version_id,
        actor="system:auto-promotion",
        reason="Promote after the forward gate passed.",
    )
    assert result["promotion_stage"] == "recommendation_enabled"
    assert result["initial_health_status"] in {"healthy", "watch"}
    assert len(result["initial_health_snapshot_id"]) == 64
    version = StrategyStore(database_url).get_version(version_id)
    assert version["promotion_stage"] == "recommendation_enabled"
    with engine.connect() as connection:
        health = connection.execute(
            select(strategy_health_snapshots).where(
                strategy_health_snapshots.c.id == result["initial_health_snapshot_id"]
            )
        ).one()
    assert str(health.strategy_version_id) == version_id
    assert str(health.health_status) == result["initial_health_status"]


def test_cost_deviation_subitem(database_url: str, tmp_path: Path, monkeypatch) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch, max_cost_deviation=0.0001
    )
    _seed_evidence(
        promotion,
        stage["simulation_portfolio_id"],
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
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
        promotion,
        portfolio_id,
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
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
    replacement = promotion.attach_paper_simulation(
        version_id,
        actor=ACTOR,
        daily_dataset=_governed_daily_dataset(),
        execution_dataset=_governed_daily_dataset(),
        initial_cash=100_000,
    )
    assert replacement["status"] == "active"
    assert replacement["simulation_portfolio_id"] != portfolio_id
    with engine.connect() as connection:
        replacement_portfolio = connection.execute(
            select(simulation_portfolios).where(
                simulation_portfolios.c.id
                == replacement["simulation_portfolio_id"]
            )
        ).one()
    assert str(replacement_portfolio.promotion_stage_id) == replacement["id"]
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
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
        fee=0.0,
        gross=1000.0,
    )
    promotion.promote(
        version_id,
        actor="allocation-risk-owner",
        reason="Promote after the forward gate passed.",
    )
    portfolio = recommendations.create(
        name="enabled recommendation",
        strategy_version_id=version_id,
        dataset="allocation-data",
        hypothetical_initial_value=1_000_000,
        actor="operator-a",
    )
    assert portfolio["status"] == "active"
