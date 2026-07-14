from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, insert, select, update

from quant_data.database import (
    open_database,
    pair_paper_fills,
    pair_paper_orders,
    pair_paper_portfolios,
    pair_portfolio_nav,
    pair_portfolio_reviews,
    pair_portfolio_risk_events,
)
from quant_platform.pair_portfolio_store import PairPortfolioStore
from quant_platform.pair_trading import PairTradingConfig
from quant_platform.strategy_store import StrategyStore


def _approved_pair(database_url: str, tmp_path) -> dict:
    strategies = StrategyStore(database_url)
    strategy = strategies.create_pair(
        name="pair paper governed strategy",
        description="Approved ETF pair strategy for the dedicated atomic spread paper ledger.",
        leg_y="SH510300",
        leg_x="SZ159919",
        asset_class="etf",
        shorting_mode="margin_borrow",
        config=asdict(PairTradingConfig()),
        actor="researcher-a",
    )
    version = strategy["versions"][0]
    backtest = strategies.create_backtest(
        version_id=version["id"],
        dataset="daily-2024-2026",
        execution_dataset="minute-2024-2026/liquid_stocks_1m+margin_eligibility",
        periods={"start": "2024-01-01", "end": "2026-07-10"},
        artifact_path=tmp_path,
    )
    digest = "a" * 64
    strategies.mark_backtest(
        backtest["id"],
        "succeeded",
        metrics={
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
            "open_position_at_end": False,
            "provenance": {
                "daily_dataset_identity_sha256": digest,
                "daily_snapshot_manifest_sha256": digest,
                "minute_snapshot_manifest_sha256": digest,
                "strategy_config_sha256": digest,
                "execution_manifest_sha256": digest,
                "pair_engine_sha256": digest,
                "shortability_evidence_sha256": digest,
                "daily_dataset_lineage_id": "d" * 64,
                "execution_snapshot_lineage_id": "e" * 64,
            },
        },
    )
    return strategies.approve(
        version["id"],
        actor="risk-approver-b",
        reason="Independent pair risk review accepted every immutable execution gate.",
    )


def _portfolio(store: PairPortfolioStore, version: dict) -> dict:
    return store.create(
        name="dedicated ETF spread paper ledger",
        strategy_version_id=version["id"],
        dataset="daily-2024-2026",
        execution_snapshot="minute-2024-2026",
        minute_dataset="liquid_stocks_1m",
        shortability_dataset="margin_eligibility",
        initial_cash=5_000_000,
        actor="paper-operator-c",
    )


def _entry_result(as_of_date: str = "2026-07-08") -> dict:
    borrow = 4000 * 0.08 / 252
    cash = 5_000_000 - 10 - borrow
    return {
        "status": "ok",
        "as_of_date": as_of_date,
        "trade_date": "2026-07-09",
        "leg_y": "SH510300",
        "leg_x": "SZ159919",
        "action": "entry",
        "reason": "negative_spread",
        "rejection": None,
        "orders": [
            {
                "instrument": "SH510300",
                "leg": "y",
                "side": "buy",
                "requested_quantity": 1000,
                "target_quantity": 1000,
                "status": "filled",
                "reason": None,
            },
            {
                "instrument": "SZ159919",
                "leg": "x",
                "side": "sell",
                "requested_quantity": 800,
                "target_quantity": -800,
                "status": "filled",
                "reason": None,
            },
        ],
        "fills": [
            {
                "instrument": "SH510300",
                "fill_time": "2026-07-09T10:00:00",
                "quantity": 1000,
                "price": 4.0,
                "gross_value": 4000.0,
                "fee": 5.0,
                "slippage": 0.0005,
            },
            {
                "instrument": "SZ159919",
                "fill_time": "2026-07-09T10:00:00",
                "quantity": 800,
                "price": 5.0,
                "gross_value": 4000.0,
                "fee": 5.0,
                "slippage": 0.0005,
            },
        ],
        "state": {
            "status": "active",
            "cash": cash,
            "nav": cash,
            "high_water_mark": 5_000_000,
            "position_direction": 1,
            "quantity_y": 1000,
            "quantity_x": -800,
            "entry_nav": cash,
            "holding_days": 1,
        },
        "closing_prices": {"SH510300": 4.0, "SZ159919": 5.0},
        "metrics": {
            "zscore": -1.7,
            "correlation": 0.91,
            "cointegration_pvalue": 0.02,
            "daily_return": cash / 5_000_000 - 1,
            "drawdown": cash / 5_000_000 - 1,
            "long_value": 4000.0,
            "short_value": 4000.0,
            "gross_exposure": 8000.0 / cash,
            "net_exposure": 0.0,
            "turnover": 8000.0 / 5_000_000,
            "fees": 10.0,
            "borrow_cost": borrow,
            "atomic_pair_execution_enforced": True,
            "shortability_enforced": True,
            "minute_execution_enforced": True,
        },
        "risk_events": [],
        "provenance": {
            "daily_dataset_identity_sha256": "a" * 64,
            "daily_snapshot_manifest_sha256": "a" * 64,
            "minute_snapshot_manifest_sha256": "a" * 64,
            "shortability_evidence_sha256": "a" * 64,
            "strategy_config_sha256": "a" * 64,
            "pair_engine_sha256": "a" * 64,
            "execution_manifest_sha256": "b" * 64,
            "daily_dataset_lineage_id": "d" * 64,
            "execution_snapshot_lineage_id": "e" * 64,
        },
    }


def test_latest_pair_portfolio_pins_resolved_batch_evidence(
    database_url: str, tmp_path
) -> None:
    version = _approved_pair(database_url, tmp_path)
    store = PairPortfolioStore(database_url)
    portfolio = store.create(
        name="rolling pair ledger",
        strategy_version_id=version["id"],
        dataset="daily-2024-2026",
        execution_snapshot="minute-2024-2026",
        minute_dataset="liquid_stocks_1m",
        shortability_dataset="margin_eligibility",
        initial_cash=5_000_000,
        actor="paper-operator-c",
        dataset_roll_policy="latest_compatible",
        dataset_lineage_id="d" * 64,
        execution_roll_policy="latest_compatible",
        execution_lineage_id="e" * 64,
    )
    dataset = {
        "name": "daily-successor",
        "lineage_id": "d" * 64,
        "provenance": {"dataset_identity_sha256": "a" * 64},
    }
    execution = {
        "snapshot": {"name": "minute-successor", "lineage_id": "e" * 64},
        "minute": {"manifest_sha256": "a" * 64},
        "shortability": {"manifest_sha256": "a" * 64},
    }
    batch, created = store.create_batch(
        portfolio_id=portfolio["id"],
        as_of_date=date(2026, 7, 8),
        artifact_path=tmp_path,
        dataset_evidence=dataset,
        execution_evidence=execution,
    )
    assert created is True
    assert batch["dataset"] == "daily-successor"
    assert batch["execution_snapshot"] == "minute-successor"
    assert store.apply_batch(batch["id"], _entry_result())["status"] == "succeeded"


def test_pair_portfolio_requires_matching_approved_execution_evidence(
    database_url: str, tmp_path
) -> None:
    version = _approved_pair(database_url, tmp_path)
    store = PairPortfolioStore(database_url)
    with pytest.raises(ValueError, match="must match"):
        store.create(
            name="wrong execution evidence",
            strategy_version_id=version["id"],
            dataset="daily-2024-2026",
            execution_snapshot="minute-2024-2026",
            minute_dataset="wrong-minute-data",
            shortability_dataset="margin_eligibility",
            initial_cash=5_000_000,
            actor="operator-c",
        )


def test_pair_portfolio_applies_two_fills_nav_and_review_atomically(
    database_url: str, tmp_path
) -> None:
    version = _approved_pair(database_url, tmp_path)
    store = PairPortfolioStore(database_url)
    portfolio = _portfolio(store, version)
    batch, created = store.create_batch(
        portfolio_id=portfolio["id"],
        as_of_date=date(2026, 7, 8),
        artifact_path=tmp_path,
    )
    assert created is True
    applied = store.apply_batch(batch["id"], _entry_result())
    assert applied["status"] == "succeeded"
    current = store.get(portfolio["id"])
    assert current["position_direction"] == 1
    assert current["quantity_y"] == 1000
    assert current["quantity_x"] == -800
    assert len(current["orders"]) == 2
    assert len(current["nav_history"]) == 1
    assert len(current["reviews"]) == 1
    engine = open_database(database_url)
    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(pair_paper_orders)) == 2
        assert connection.scalar(select(func.count()).select_from(pair_paper_fills)) == 2
        assert connection.scalar(select(func.count()).select_from(pair_portfolio_nav)) == 1
        assert connection.scalar(select(func.count()).select_from(pair_portfolio_reviews)) == 1
    store.apply_batch(batch["id"], _entry_result())
    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(pair_paper_fills)) == 2


def test_pair_portfolio_rejects_stale_worker_state(database_url: str, tmp_path) -> None:
    version = _approved_pair(database_url, tmp_path)
    store = PairPortfolioStore(database_url)
    portfolio = _portfolio(store, version)
    batch, _ = store.create_batch(
        portfolio_id=portfolio["id"],
        as_of_date=date(2026, 7, 8),
        artifact_path=tmp_path,
    )
    store.set_status(portfolio["id"], "paused")
    with pytest.raises(ValueError, match="state changed"):
        store.apply_batch(batch["id"], _entry_result())
    assert Decimal(str(store.get(portfolio["id"])["cash"])) == Decimal("5000000")


def test_pair_portfolio_recomputes_cash_instead_of_trusting_worker_result(
    database_url: str, tmp_path
) -> None:
    version = _approved_pair(database_url, tmp_path)
    store = PairPortfolioStore(database_url)
    portfolio = _portfolio(store, version)
    batch, _ = store.create_batch(
        portfolio_id=portfolio["id"],
        as_of_date=date(2026, 7, 8),
        artifact_path=tmp_path,
    )
    result = _entry_result()
    result["state"]["cash"] += 100
    result["state"]["nav"] += 100
    with pytest.raises(ValueError, match="cash does not reconcile"):
        store.apply_batch(batch["id"], result)
    current = store.get(portfolio["id"])
    assert current["position_direction"] == 0
    assert not current["orders"]


def test_pair_risk_lifecycle_blocks_activation_and_active_batch_resolution(
    database_url: str, tmp_path
) -> None:
    version = _approved_pair(database_url, tmp_path)
    store = PairPortfolioStore(database_url)
    portfolio = _portfolio(store, version)
    engine = open_database(database_url)
    with engine.begin() as connection:
        event_id = connection.execute(
            insert(pair_portfolio_risk_events)
            .values(
                portfolio_id=portfolio["id"],
                batch_id=None,
                severity="critical",
                event_type="drawdown",
                rule="max_drawdown",
                observed=-0.20,
                limit_value=-0.15,
                status="open",
                details_json={"source": "test"},
                created_at=datetime.now(UTC),
            )
            .returning(pair_portfolio_risk_events.c.id)
        ).scalar_one()

    store.set_status(portfolio["id"], "paused")
    with pytest.raises(ValueError, match="critical pair risk events"):
        store.set_status(portfolio["id"], "active")
    acknowledged = store.acknowledge_risk_event(portfolio["id"], event_id, actor="risk-operator")
    assert acknowledged["status"] == "acknowledged"

    # An acknowledged event remains unresolved while any Worker result can still mutate state.
    with engine.begin() as connection:
        connection.execute(
            update(pair_paper_portfolios)
            .where(pair_paper_portfolios.c.id == portfolio["id"])
            .values(status="active")
        )
    batch, _ = store.create_batch(
        portfolio_id=portfolio["id"],
        as_of_date=date(2026, 7, 8),
        artifact_path=tmp_path,
    )
    with pytest.raises(ValueError, match="while a batch is active"):
        store.resolve_risk_event(
            portfolio["id"],
            event_id,
            actor="risk-manager",
            reason="Exposure is flat and the incident has been independently reviewed.",
        )
    store.mark_batch(batch["id"], "cancelled", error="risk review")
    resolved = store.resolve_risk_event(
        portfolio["id"],
        event_id,
        actor="risk-manager",
        reason="Exposure is flat and the incident has been independently reviewed.",
    )
    assert resolved["status"] == "resolved"
    assert store.set_status(portfolio["id"], "active")["status"] == "active"


def test_liquidation_pending_pair_portfolio_cannot_be_paused(database_url: str, tmp_path) -> None:
    version = _approved_pair(database_url, tmp_path)
    store = PairPortfolioStore(database_url)
    portfolio = _portfolio(store, version)
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(pair_paper_portfolios)
            .where(pair_paper_portfolios.c.id == portfolio["id"])
            .values(status="liquidation_pending")
        )
    with pytest.raises(ValueError, match="cannot be paused"):
        store.set_status(portfolio["id"], "paused")
