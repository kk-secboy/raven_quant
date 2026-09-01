from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from governance_fixtures import (
    DATASET_IDENTITY,
    PERIODS,
    create_strategy_version,
    formal_backtest_metrics,
)
from test_promotion_chain import (
    _governed_daily_dataset,
    _qlib_doubles,
    _record_collector_authority,
    _seed_evidence,
)

import quant_platform.promotion as promotion_module
from quant_platform.cost_model import CostModelConfig
from quant_platform.investor_profile import InvestorSimulationProfileStore
from quant_platform.parameter_experiment_store import ParameterExperimentStore
from quant_platform.portfolio_policy import POLICY_VERSION
from quant_platform.promotion import PromotionStore
from quant_platform.qlib_backtest import QLIB_ENGINE_VERSION
from quant_platform.recommendation_store import RecommendationStore
from quant_platform.simulation_store import SimulationStore
from quant_platform.strategy_store import StrategyStore
from scripts.run_recommendation_refresh import _next_known_trading_date


def test_three_horizon_research_to_recommendation_snapshot(
    tmp_path: Path, database_url: str, monkeypatch
) -> None:
    _qlib_doubles(monkeypatch)
    monkeypatch.setattr(
        promotion_module,
        "_now",
        lambda: datetime(2026, 7, 1, tzinfo=UTC),
    )
    version_id = create_strategy_version(
        database_url,
        tmp_path,
        dataset="synthetic-qlib",
        recipe_id="short_relative_strength",
    )

    experiments = ParameterExperimentStore(database_url)
    experiment = experiments.create(
        strategy_version_id=version_id,
        dataset="synthetic-qlib",
        periods={
            "in_sample": {"start": "2018-05-28", "end": "2019-06-28"},
            "out_of_sample": {"start": "2019-07-01", "end": "2020-12-31"},
        },
        parameter_grid={"topk": [30, 50]},
        baseline_config={"topk": 50},
        trials=[
            {"parameters": {"topk": 30}, "config": {"topk": 30}},
            {"parameters": {"topk": 50}, "config": {"topk": 50}},
        ],
        artifact_root=tmp_path / "experiments",
        created_by="pipeline-test",
    )
    experiments.apply_result(
        experiment["id"],
        {
            "trials": [
                {
                    "trial_index": 0,
                    "status": "succeeded",
                    "score": 0.7,
                    "metrics": {"in_sample": {}, "out_of_sample": {}},
                    "warnings": [],
                },
                {
                    "trial_index": 1,
                    "status": "succeeded",
                    "score": 0.9,
                    "metrics": {"in_sample": {}, "out_of_sample": {}},
                    "warnings": [],
                },
            ],
            "summary": {"selected_trial_index": 1, "selection_source": "validation_only"},
        },
    )
    assert experiments.get(experiment["id"])["summary"]["selected_trial_index"] == 1

    strategies = StrategyStore(database_url)
    version = strategies.get_version(version_id)
    periods = {
        "start": PERIODS["test_start"].isoformat(),
        "end": PERIODS["test_end"].isoformat(),
    }
    artifact = tmp_path / "formal-final-test"
    artifact.mkdir()
    backtest = strategies.create_backtest(
        version_id=version_id,
        dataset="synthetic-qlib",
        periods=periods,
        artifact_path=artifact,
    )
    factor = version["factors"][0]
    manifest = artifact / "manifest.json"
    manifest.write_text(
        json.dumps(
                {
                    "strategy_version_id": version_id,
                    "dataset": "synthetic-qlib",
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
    metrics = formal_backtest_metrics(
        version,
        manifest,
        hypothesis_group_evidence=strategies.hypothesis_group_evidence(version_id),
    )
    strategies.validate_backtest_artifacts(backtest["id"], metrics)
    strategies.mark_backtest(backtest["id"], "succeeded", metrics=metrics)
    promotion = PromotionStore(database_url)
    promotion.register_forward_gate(
        version_id,
        actor="pipeline-risk-owner",
    )
    approved = strategies.approve(
        version_id,
        actor="risk-owner",
        reason="Approved the frozen strategy after its one reserved final test.",
    )
    assert approved["status"] == "approved"
    InvestorSimulationProfileStore(database_url).create_version(
        profile_key="primary",
        initial_capital="5000000",
        risk_profile="balanced",
        min_cash_weight=0.10,
        max_gross_exposure=0.90,
        market_permissions={
            "main_board": True,
            "star_market": False,
            "chi_next": False,
            "beijing_exchange": False,
            "etf": True,
        },
        actor="pipeline-investor",
        activate=True,
    )
    stage = promotion.attach_paper_simulation(
        version_id,
        actor="pipeline-risk-owner",
        daily_dataset={**_governed_daily_dataset(), "name": "synthetic-qlib"},
        execution_dataset={**_governed_daily_dataset(), "name": "synthetic-qlib"},
    )
    paper = SimulationStore(database_url).get(stage["simulation_portfolio_id"])
    binding = paper["execution_policy"]["investor_profile_binding"]
    assert float(paper["initial_cash"]) == 5_000_000.0
    assert binding["profile_id"]
    assert binding["profile_version"] == 1
    assert len(binding["profile_content_sha256"]) == 64
    assert binding["market_permissions"] == {
        "beijing_exchange": False,
        "chi_next": False,
        "etf": True,
        "main_board": True,
        "star_market": False,
    }
    _seed_evidence(
        promotion,
        stage["simulation_portfolio_id"],
        nav_days=90,
        succeeded=90,
        valid_round_trips=30,
        fee=0.0,
        gross=1_000.0,
    )
    monkeypatch.setattr(
        promotion_module,
        "_now",
        lambda: datetime(2026, 7, 9, tzinfo=UTC),
    )
    _record_collector_authority(database_url, version_id, monkeypatch)
    promoted = promotion.promote(
        version_id,
        actor="pipeline-risk-owner",
        reason="Enable recommendations after the governed forward gate passed.",
    )
    assert promoted["promotion_stage"] == "recommendation_enabled"
    monkeypatch.setattr(
        promotion_module,
        "_now",
        lambda: datetime(2026, 7, 10, tzinfo=UTC),
    )

    recommendations = RecommendationStore(database_url)
    portfolio = recommendations.create(
        name="synthetic end-to-end recommendations",
        strategy_version_id=version_id,
        dataset="synthetic-qlib",
        hypothetical_initial_value=5_000_000,
        actor="pipeline-test",
    )
    snapshot, created = recommendations.create_snapshot(
        portfolio_id=portfolio["id"],
        as_of_date=date(2026, 7, 10),
        dataset="synthetic-qlib",
        dataset_identity_sha256=DATASET_IDENTITY,
    )
    assert created is True
    completed = recommendations.apply_result(
        snapshot["id"],
        {
            "status": "ok",
            "portfolio_id": portfolio["id"],
            "strategy_version_id": version_id,
            "dataset": "synthetic-qlib",
            "dataset_identity_sha256": DATASET_IDENTITY,
            "as_of_date": "2026-07-10",
            "policy_version": POLICY_VERSION,
            "backtest_engine_version": QLIB_ENGINE_VERSION,
            "effective_date": "2026-07-13",
            "cost_model": CostModelConfig().to_dict(),
            "cash_weight": 0.98,
            "reference_prices": {"SH600000": 10.0},
            "holdings": [
                {
                    "instrument": "SH600000",
                    "weight": 0.02,
                    "previous_weight": 0.0,
                    "weight_change": 0.02,
                    "action": "increase",
                    "reason": "shared policy signal and constraints",
                }
            ],
            "hypothetical_observation": {
                "trade_date": "2026-07-10",
                "hypothetical_value": 4_999_000,
                "daily_return": 0.0,
                "benchmark_return": 0.0,
                "turnover": 0.02,
                "estimated_cost": 1_000,
            },
        },
    )
    assert completed["status"] == "succeeded"
    assert completed["holdings"][0]["instrument"] == "SH600000"


def test_recommendation_uses_snapshot_known_calendar_for_next_session(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "qlib" / "metadata"
    metadata.mkdir(parents=True)
    pd.DataFrame(
        {"date": pd.to_datetime(["2026-07-24", "2026-07-27", "2026-07-28"])}
    ).to_parquet(metadata / "known_trading_calendar.parquet", index=False)

    assert _next_known_trading_date(
        tmp_path / "qlib", pd.Timestamp("2026-07-24")
    ) == "2026-07-27"
