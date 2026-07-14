import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import func, select, update

from quant_data.database import (
    paper_fills,
    paper_orders,
    paper_positions,
    portfolio_nav,
    portfolio_reviews,
    risk_events,
)
from quant_platform.portfolio_store import PortfolioStore
from quant_platform.research_store import ResearchStore
from quant_platform.strategy_store import StrategyStore


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_approval_artifacts(
    *,
    artifact_path: Path,
    version: dict,
    backtest: dict,
    candidate: dict,
) -> dict:
    replay = {
        "execution_risk_overlay_enforced": True,
        "execution_model": "next_open",
        "max_drawdown": -0.10,
        "execution_risk_thresholds": {
            "max_daily_loss": 0.03,
            "stop_loss": 0.07,
            "take_profit_partial": 0.12,
            "take_profit_partial_fraction": 0.50,
            "take_profit": 0.20,
            "max_drawdown_reduce": 0.10,
            "max_drawdown_liquidate": 0.15,
            "drawdown_reduction_exposure": 0.50,
        },
    }
    artifact_path.mkdir(parents=True, exist_ok=True)
    (artifact_path / "execution_replay.json").write_text(
        json.dumps(replay, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "strategy_version_id": version["id"],
        "dataset": backtest["dataset"],
        "benchmark": version["benchmark"],
        "periods": backtest["periods"],
        "config": version["config"],
        "factors": [
            {
                "candidate_id": item["factor_candidate_id"],
                "values_path": item["values_path"],
                "code_sha256": item["code_sha256"],
                "weight": item["weight"],
                "direction": item["direction"],
            }
            for item in version["factors"]
        ],
    }
    manifest_path = artifact_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "execution_risk_overlay_enforced": True,
        "execution_replay": replay,
        "provenance": {
            "dataset_identity_sha256": "a" * 64,
            "snapshot_manifest_sha256": "a" * 64,
            "qlib_builder_sha256": "a" * 64,
            "strategy_config_sha256": _canonical_sha256(version["config"]),
            "execution_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "execution_replay_sha256": _canonical_sha256(replay),
            "factor_values_sha256": {
                candidate["id"]: hashlib.sha256(
                    Path(candidate["values_path"]).read_bytes()
                ).hexdigest()
            },
            "factor_code_sha256": {candidate["id"]: candidate["code_sha256"]},
            "qlib_version": "0.9.8",
        },
    }


def _approved_version(database_url: str, tmp_path: Path, *, max_weight: float = 0.20) -> dict:
    research = ResearchStore(database_url)
    run = research.create_run(
        kind="factor",
        objective="Create a production-governed paper trading factor.",
        dataset="snapshot",
        requested_by="researcher",
        budget={"loop_n": 1, "duration": "30m"},
        config={},
        artifact_path=tmp_path,
    )
    code_path = tmp_path / "factor.py"
    values_path = tmp_path / "factor.h5"
    code_path.write_text("def factor(frame):\n    return frame['close']\n", encoding="utf-8")
    values_path.write_bytes(b"immutable-factor-values")
    candidate = research.add_candidate(
        run["id"],
        name="paper_quality",
        description="Quality factor for paper ledger tests.",
        formulation="roe - leverage",
        variables={},
        source_iteration=0,
        code_path=str(code_path),
        values_path=str(values_path),
        code_sha256=hashlib.sha256(code_path.read_bytes()).hexdigest(),
        rdagent_decision=True,
        rdagent_feedback="ok",
    )
    metrics = {
        "ic": 0.04,
        "icir": 0.8,
        "rank_ic": 0.04,
        "rank_icir": 0.8,
        "turnover": 0.3,
        "max_correlation": 0.3,
        "cost_adjusted_return": 0.06,
        "valid_ic": 0.03,
        "test_days": 500,
        "direction": "normal",
    }
    evaluation_path = tmp_path / "factor-evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "evaluations": [
                    {"candidate_id": candidate["id"], "status": "ok", "metrics": metrics}
                ],
            }
        ),
        encoding="utf-8",
    )
    research.record_evaluation(
        candidate["id"],
        dataset="snapshot",
        train_start=date(2018, 1, 1),
        train_end=date(2021, 12, 31),
        valid_start=date(2022, 1, 1),
        valid_end=date(2023, 12, 31),
        test_start=date(2024, 1, 1),
        test_end=date(2026, 7, 10),
        metrics=metrics,
        artifact_path=str(evaluation_path),
    )
    research.promote(
        candidate["id"],
        actor="factor-owner",
        reason="Independent validation passed for paper trading.",
    )
    strategies = StrategyStore(database_url)
    strategy = strategies.create(
        name=f"paper strategy {max_weight}",
        description="Approved strategy used to validate the paper trading ledger.",
        benchmark="SH000300",
        universe="cn_all",
        factors=[{"candidate_id": candidate["id"], "weight": 1.0}],
        config={
            "topk": 10,
            "n_drop": 2,
            "max_position_weight": max_weight,
            "max_daily_turnover": 0.50,
            "max_tracking_error": 0.20,
            "max_drawdown": 0.25,
            "max_turnover": 0.60,
            "min_information_ratio": 0.0,
            "min_sharpe_ratio": 0.0,
            "min_sortino_ratio": 0.0,
            "min_robustness_pass_rate": 0.75,
            "min_backtest_days": 504,
            "capacity_notional": 5_000_000,
            "max_volume_participation": 0.01,
            "min_capacity_fill_ratio": 0.95,
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "execution_model": "next_open",
        },
        actor="strategy-owner",
    )
    version = strategy["versions"][0]
    backtest = strategies.create_backtest(
        version_id=version["id"],
        dataset="snapshot",
        periods={"start": "2024-01-01", "end": "2026-07-10"},
        artifact_path=tmp_path / "backtest",
    )
    approval_evidence = _write_approval_artifacts(
        artifact_path=tmp_path / "backtest",
        version=version,
        backtest=backtest,
        candidate=candidate,
    )
    strategies.mark_backtest(
        backtest["id"],
        "succeeded",
        metrics={
            "backtest_engine": "qlib",
            "qlib_native_backtest": True,
            **approval_evidence,
            "tracking_error": 0.10,
            "max_drawdown": -0.10,
            "average_turnover": 0.30,
            "information_ratio": 0.50,
            "sharpe_ratio": 0.80,
            "sortino_ratio": 1.10,
            "robustness_pass_rate": 1.0,
            "rolling_pass_rate": 1.0,
            "rolling_window_count": 5,
            "event_stress_pass_rate": 1.0,
            "event_stress_count": 5,
            "capacity_fill_ratio": 1.0,
            "trading_days": 600,
        },
    )
    return strategies.approve(
        version["id"],
        actor="risk-owner",
        reason="All strategy risk gates passed for paper deployment.",
    )


def _result(strategy_version_id: str) -> dict:
    return {
        "status": "ok",
        "strategy_version_id": strategy_version_id,
        "signal_engine": "qlib_governed_signal",
        "as_of_date": "2025-01-02",
        "trade_date": "2025-01-03",
        "benchmark_return": 0.002,
        "orders": [
            {
                "instrument": "SH600000",
                "side": "buy",
                "order_type": "market",
                "target_weight": 0.10,
                "quantity": 1000,
            }
        ],
        "fills": [
            {
                "instrument": "SH600000",
                "quantity": 1000,
                "price": 100,
                "fee": 50,
                "slippage": 0.0005,
                "fill_time": "2025-01-03T01:30:00+00:00",
            }
        ],
        "closing_prices": {"SH600000": 110},
        "industries": {"SH600000": "Bank"},
        "risk_events": [],
    }


def test_portfolio_requires_approval_and_applies_each_batch_once(
    database_url: str, tmp_path: Path
) -> None:
    version = _approved_version(database_url, tmp_path)
    store = PortfolioStore(database_url)
    portfolio = store.create(
        name="governed paper portfolio",
        strategy_version_id=version["id"],
        dataset="snapshot",
        initial_cash=1_000_000,
        actor="portfolio-owner",
    )
    batch, created = store.create_batch(
        portfolio_id=portfolio["id"],
        as_of_date=date(2025, 1, 2),
        artifact_path=tmp_path / "paper",
    )
    duplicate, duplicate_created = store.create_batch(
        portfolio_id=portfolio["id"],
        as_of_date=date(2025, 1, 2),
        artifact_path=tmp_path / "paper",
    )
    assert created is True
    assert duplicate_created is False
    assert duplicate["id"] == batch["id"]
    store.mark_batch(batch["id"], "running")
    store.apply_batch(batch["id"], _result(version["id"]))
    store.apply_batch(batch["id"], _result(version["id"]))
    current = store.get(portfolio["id"])
    assert current["cash"] == 899_950
    assert current["nav"] == 1_009_950
    assert current["positions"][0]["quantity"] == 1000
    assert current["reviews"][0]["status"] == "ok"
    assert current["reviews"][0]["summary"]["net_pnl"] == 9950
    assert current["reviews"][0]["summary"]["attribution_gap"] == 0
    assert current["reviews"][0]["summary"]["best_contributors"] == [
        {"instrument": "SH600000", "pnl": 10000.0}
    ]
    with store.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(paper_fills)) == 1
        assert connection.scalar(select(func.count()).select_from(portfolio_nav)) == 1
        assert connection.scalar(select(func.count()).select_from(portfolio_reviews)) == 1


def test_risk_breach_is_persisted_and_pauses_portfolio(database_url: str, tmp_path: Path) -> None:
    version = _approved_version(database_url, tmp_path, max_weight=0.05)
    store = PortfolioStore(database_url)
    portfolio = store.create(
        name="risk pause portfolio",
        strategy_version_id=version["id"],
        dataset="snapshot",
        initial_cash=1_000_000,
        actor="portfolio-owner",
    )
    batch, _ = store.create_batch(
        portfolio_id=portfolio["id"],
        as_of_date=date(2025, 1, 2),
        artifact_path=tmp_path / "risk-paper",
    )
    store.apply_batch(batch["id"], _result(version["id"]))
    current = store.get(portfolio["id"])
    assert current["status"] == "paused"
    assert current["risk_events"][0]["rule"] == "max_position_weight"
    with store.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(risk_events)) == 1


def test_risk_event_requires_audited_resolution_before_reactivation(
    database_url: str, tmp_path: Path
) -> None:
    version = _approved_version(database_url, tmp_path, max_weight=0.05)
    store = PortfolioStore(database_url)
    portfolio = store.create(
        name="audited risk lifecycle portfolio",
        strategy_version_id=version["id"],
        dataset="snapshot",
        initial_cash=1_000_000,
        actor="portfolio-owner",
    )
    batch, _ = store.create_batch(
        portfolio_id=portfolio["id"],
        as_of_date=date(2025, 1, 2),
        artifact_path=tmp_path / "audited-risk-paper",
    )
    store.apply_batch(batch["id"], _result(version["id"]))
    event = store.get(portfolio["id"])["risk_events"][0]

    acknowledged = store.acknowledge_risk_event(portfolio["id"], event["id"], actor="risk-owner")
    assert acknowledged["status"] == "acknowledged"
    assert acknowledged["acknowledged_by"] == "risk-owner"
    with pytest.raises(ValueError, match="must be resolved"):
        store.set_status(portfolio["id"], "active")
    with pytest.raises(ValueError, match="concentration remains"):
        store.resolve_risk_event(
            portfolio["id"],
            event["id"],
            actor="risk-owner",
            reason="Position concentration was reviewed and requires a corrective trade.",
        )

    with store.engine.begin() as connection:
        connection.execute(
            update(paper_positions)
            .where(paper_positions.c.portfolio_id == portfolio["id"])
            .values(weight=0.04)
        )

    resolved = store.resolve_risk_event(
        portfolio["id"],
        event["id"],
        actor="risk-owner",
        reason="Position concentration was reviewed and the portfolio remains paused.",
    )
    assert resolved["status"] == "resolved"
    assert resolved["resolved_by"] == "risk-owner"
    assert resolved["resolution_reason"].startswith("Position concentration")
    assert store.set_status(portfolio["id"], "active")["status"] == "active"


def test_pretrade_hard_event_and_execution_reason_are_atomic(
    database_url: str, tmp_path: Path
) -> None:
    version = _approved_version(database_url, tmp_path)
    store = PortfolioStore(database_url)
    portfolio = store.create(
        name="daily loss pause portfolio",
        strategy_version_id=version["id"],
        dataset="snapshot",
        initial_cash=1_000_000,
        actor="portfolio-owner",
    )
    batch, _ = store.create_batch(
        portfolio_id=portfolio["id"],
        as_of_date=date(2025, 1, 2),
        artifact_path=tmp_path / "daily-loss-paper",
    )
    result = _result(version["id"])
    result["orders"][0]["reason"] = "stop_loss"
    result["risk_events"] = [
        {
            "severity": "critical",
            "event_type": "circuit_breaker",
            "rule": "max_daily_loss",
            "observed": 0.04,
            "limit_value": 0.03,
            "status": "open",
            "details": {"action": "portfolio_paused_no_new_buys"},
        }
    ]
    store.apply_batch(batch["id"], result)
    current = store.get(portfolio["id"])
    assert current["status"] == "paused"
    assert current["risk_events"][0]["rule"] == "max_daily_loss"
    assert current["positions"][0]["industry"] == "Bank"
    with store.engine.connect() as connection:
        order = connection.execute(select(paper_orders)).one()
        assert order.status == "filled"
        assert order.reason == "stop_loss"


def test_posttrade_drawdown_schedules_non_overridable_liquidation(
    database_url: str, tmp_path: Path
) -> None:
    version = _approved_version(database_url, tmp_path)
    store = PortfolioStore(database_url)
    portfolio = store.create(
        name="drawdown liquidation portfolio",
        strategy_version_id=version["id"],
        dataset="snapshot",
        initial_cash=1_000_000,
        actor="portfolio-owner",
    )
    batch, _ = store.create_batch(
        portfolio_id=portfolio["id"],
        as_of_date=date(2025, 1, 2),
        artifact_path=tmp_path / "drawdown-paper",
    )
    result = _result(version["id"])
    result["orders"][0]["quantity"] = 8000
    result["fills"][0]["quantity"] = 8000
    result["closing_prices"]["SH600000"] = 80
    store.apply_batch(batch["id"], result)
    current = store.get(portfolio["id"])
    assert current["status"] == "liquidation_pending"
    assert any(item["rule"] == "max_drawdown_liquidate" for item in current["risk_events"])
    next_batch, created = store.create_batch(
        portfolio_id=portfolio["id"],
        as_of_date=date(2025, 1, 3),
        artifact_path=tmp_path / "liquidation-paper",
    )
    assert created is True
    assert next_batch["status"] == "queued"
    with pytest.raises(ValueError, match="cannot be overridden"):
        store.set_status(portfolio["id"], "active")


def test_paper_batch_rejects_worker_dataset_substitution(
    database_url: str, tmp_path: Path
) -> None:
    version = _approved_version(database_url, tmp_path)
    store = PortfolioStore(database_url)
    portfolio = store.create(
        name="evidence pinned portfolio",
        strategy_version_id=version["id"],
        dataset="snapshot",
        initial_cash=1_000_000,
        actor="portfolio-owner",
    )
    evidence = {
        "name": "snapshot",
        "lineage_id": "d" * 64,
        "provenance": {"dataset_identity_sha256": "a" * 64},
    }
    batch, _ = store.create_batch(
        portfolio_id=portfolio["id"],
        as_of_date=date(2025, 1, 2),
        artifact_path=tmp_path / "evidence-paper",
        dataset_evidence=evidence,
    )
    result = _result(version["id"])
    result["provenance"] = {
        "daily_dataset_identity_sha256": "b" * 64,
        "daily_dataset_lineage_id": "d" * 64,
    }
    with pytest.raises(ValueError, match="changed the batch-pinned Qlib dataset"):
        store.apply_batch(batch["id"], result)
