from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from governance_fixtures import (
    DATASET_IDENTITY,
    PERIODS,
    create_strategy_version,
    formal_backtest_metrics,
)
from sqlalchemy import update

from quant_data.database import strategy_versions
from quant_platform.cost_model import CostModelConfig
from quant_platform.parameter_experiment_store import ParameterExperimentStore
from quant_platform.portfolio_policy import POLICY_VERSION
from quant_platform.qlib_backtest import QLIB_ENGINE_VERSION
from quant_platform.recommendation_store import RecommendationStore
from quant_platform.strategy_store import StrategyStore
from scripts.run_recommendation_refresh import _next_known_trading_date


def test_v2_research_to_final_test_and_recommendation_snapshot(
    tmp_path: Path, database_url: str
) -> None:
    version_id = create_strategy_version(database_url, tmp_path, dataset="synthetic-qlib")

    experiments = ParameterExperimentStore(database_url)
    experiment = experiments.create(
        strategy_version_id=version_id,
        dataset="synthetic-qlib",
        periods={
            "in_sample": {"start": "2022-05-27", "end": "2023-02-28"},
            "out_of_sample": {"start": "2023-03-01", "end": "2023-12-31"},
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
    metrics = formal_backtest_metrics(version, manifest)
    strategies.validate_backtest_artifacts(backtest["id"], metrics)
    strategies.mark_backtest(backtest["id"], "succeeded", metrics=metrics)
    approved = strategies.approve(
        version_id,
        actor="risk-owner",
        reason="Approved the frozen strategy after its one reserved final test.",
    )
    assert approved["status"] == "approved"
    # Fixture shortcut: the promotion chain sets promotion_stage="paper" on
    # approval; this pipeline test predates the forward gate, so it marks the
    # version enabled directly (production must pass the forward evidence
    # gate via PromotionStore.promote).
    with strategies.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(promotion_stage="recommendation_enabled")
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
