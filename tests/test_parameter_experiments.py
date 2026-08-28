from __future__ import annotations

from datetime import date

import pytest

from quant_data.execution_contract import build_strategy_execution_contract
from quant_platform.parameter_experiments import (
    build_portfolio_construction_trials,
    evaluate_trial,
    merge_admitted_trial_ledgers,
    normalize_parameter_grid,
    portfolio_trial_comparability_evidence,
    select_frozen_portfolio_config,
    split_model_portfolio_period,
    split_research_period,
    summarize_trials,
)

pytestmark = pytest.mark.no_database


def test_parameter_grid_is_deterministic_and_bounded() -> None:
    grid, trials = normalize_parameter_grid(
        {"topk": [30, 50], "max_daily_turnover": [0.15, 0.20]}, max_trials=4
    )
    assert list(grid) == ["max_daily_turnover", "topk"]
    assert trials == [
        {"max_daily_turnover": 0.15, "topk": 30},
        {"max_daily_turnover": 0.15, "topk": 50},
        {"max_daily_turnover": 0.20, "topk": 30},
        {"max_daily_turnover": 0.20, "topk": 50},
    ]
    with pytest.raises(ValueError, match="maximum is 3"):
        normalize_parameter_grid(grid, max_trials=3)
    with pytest.raises(ValueError, match="unsupported"):
        normalize_parameter_grid({"future_return": [1, 2]})


def test_model_and_fin_quant_trials_share_one_prior_ledger() -> None:
    def evidence(name: str, sharpe: float) -> dict:
        return {
            "final_oos_opened": False,
            "trial_names": [name],
            "trial_daily_sharpes": [sharpe],
            "trial_count": 1,
            "evidence_sha256": "a" * 64,
        }

    merged = merge_admitted_trial_ledgers(
        {
            "binding_sha256": "b" * 64,
            "model_grid": {"multiple_testing": evidence("model-a", 0.01)},
            "quant_bundle": {"multiple_testing": evidence("joint-a", 0.02)},
        }
    )

    assert merged["trial_count"] == 2
    assert merged["trial_names"] == ["model:model-a", "fin_quant:joint-a"]
    assert merged["trial_daily_sharpes"] == [0.01, 0.02]
    assert merged["final_oos_opened"] is False


def test_portfolio_construction_competition_is_fixed_and_shares_execution() -> None:
    baseline = {
        "signal_frequency": "day",
        "signal_period": 1,
        "rebalance_frequency": "day",
        "execution_frequency": "day",
        "execution_method": "open",
        "execution_days": 1,
        "execution_lag_bars": 1,
        "max_volume_participation": 0.10,
        "max_position_weight": 0.05,
        "max_daily_turnover": 0.20,
        "portfolio_construction": "topk_equal_weight",
    }

    grid, trials = build_portfolio_construction_trials(baseline)

    assert grid == {
        "portfolio_construction": ["topk_equal_weight", "industry_neutral_qp"]
    }
    assert [item["parameters"] for item in trials] == [
        {"portfolio_construction": "topk_equal_weight"},
        {"portfolio_construction": "industry_neutral_qp"},
    ]
    baseline_contract = build_strategy_execution_contract(baseline)
    assert all(
        build_strategy_execution_contract(item["config"]) == baseline_contract
        for item in trials
    )
    normalized, candidates = normalize_parameter_grid(grid, max_trials=2)
    assert normalized == grid
    assert candidates == [item["parameters"] for item in trials]
    with pytest.raises(ValueError, match="portfolio_construction"):
        normalize_parameter_grid({"portfolio_construction": ["unconstrained"]})


def test_period_split_is_non_overlapping_and_requires_history() -> None:
    periods = split_research_period(date(2024, 1, 1), date(2026, 1, 1))
    assert periods["in_sample"] == {"start": "2024-01-01", "end": "2025-03-14"}
    assert periods["out_of_sample"] == {"start": "2025-03-20", "end": "2026-01-01"}
    assert periods["purge_days"] == 1
    assert periods["embargo_days"] == 5
    with pytest.raises(ValueError, match="126"):
        split_research_period(date(2025, 1, 1), date(2025, 4, 1))


def test_model_portfolio_split_reserves_inner_validation_before_trials() -> None:
    periods = split_model_portfolio_period(date(2023, 1, 1), date(2025, 12, 31))

    assert periods["in_sample"]["start"] > "2023-01-01"
    assert periods["in_sample"]["end"] < periods["out_of_sample"]["start"]
    assert periods["out_of_sample"]["end"] == "2025-12-31"
    with pytest.raises(ValueError, match="252"):
        split_model_portfolio_period(date(2025, 1, 1), date(2025, 7, 1))


def test_portfolio_winner_selection_requires_unambiguous_pre_final_evidence() -> None:
    baseline = {
        "signal_frequency": "day",
        "signal_period": 1,
        "rebalance_frequency": "day",
        "execution_frequency": "day",
        "execution_method": "open",
        "execution_days": 1,
        "execution_lag_bars": 1,
        "max_volume_participation": 0.10,
        "max_position_weight": 0.05,
        "max_daily_turnover": 0.20,
    }
    _, trial_specs = build_portfolio_construction_trials(baseline)
    trials = []
    for index, spec in enumerate(trial_specs):
        trials.append(
            {
                **spec,
                "trial_index": index,
                "status": "succeeded",
                "metrics": {
                    segment: {
                        "deflated_sharpe_probability": 0.99,
                        "cost_model": {"version": "test-cost-v1"},
                        "provenance": {
                            "evaluation_mode": "pre_final_portfolio_trial",
                            "evaluation_scope": "pre_final_only",
                            "final_oos_opened": False,
                            "formal_model_predictions_sha256": "a" * 64,
                            "formal_model_checkpoint_sha256": "b" * 64,
                            "model_signal_identity_sha256": "c" * 64,
                            "dataset_identity_sha256": "d" * 64,
                            "formal_model_admission_binding_sha256": "e" * 64,
                            "pre_final_cutoff": "2025-12-31",
                        },
                    }
                    for segment in ("in_sample", "out_of_sample")
                },
            }
        )
    experiment = {
        "id": "experiment-1",
        "status": "succeeded",
        "trials": trials,
        "summary": {
            "trial_count": 2,
            "governed_trial_count": 7,
            "best_trial_index": 1,
            "final_oos_opened": False,
            "warnings": [],
            "comparability": portfolio_trial_comparability_evidence(trials),
        },
    }

    winner = select_frozen_portfolio_config(experiment)

    assert winner["portfolio_construction"] == "industry_neutral_qp"
    assert winner["final_oos_opened"] is False
    assert len(winner["portfolio_config_sha256"]) == 64
    tampered = {
        **experiment,
        "trials": [dict(item) for item in trials],
    }
    tampered["trials"][0] = {
        **tampered["trials"][0],
        "metrics": {
            **tampered["trials"][0]["metrics"],
            "out_of_sample": {
                **tampered["trials"][0]["metrics"]["out_of_sample"],
                "provenance": {
                    **tampered["trials"][0]["metrics"]["out_of_sample"][
                        "provenance"
                    ],
                    "formal_model_predictions_sha256": "f" * 64,
                },
            },
        },
    }
    with pytest.raises(ValueError, match="identical model predictions"):
        select_frozen_portfolio_config(tampered)
    with pytest.raises(ValueError, match="fragile_ranking"):
        select_frozen_portfolio_config(
            {
                **experiment,
                "summary": {
                    **experiment["summary"],
                    "warnings": ["fragile_ranking"],
                },
            }
        )


def test_trial_evaluation_flags_sample_decay_and_summary_risk() -> None:
    score, warnings = evaluate_trial(
        {"information_ratio": 1.2, "annualized_excess_return": 0.12},
        {
            "information_ratio": -0.1,
            "annualized_excess_return": -0.02,
            "max_drawdown": -0.30,
            "average_turnover": 0.4,
            "robustness_pass_rate": 0.4,
            "deflated_sharpe_probability": 0.50,
        },
    )
    assert score < 0
    assert set(warnings) == {
        "oos_sign_reversal",
        "performance_decay",
        "oos_drawdown_high",
        "oos_robustness_low",
        "deflated_sharpe_failed",
    }
    trials = [
        {
            "trial_index": 0,
            "parameters": {"topk": 30},
            "status": "succeeded",
            "score": 1.0,
            "warnings": [],
            "metrics": {
                "in_sample": {"information_ratio": 1.2, "robustness": {"large": True}},
                "out_of_sample": {
                    "information_ratio": 0.8,
                    "robustness": {"large": True},
                    "deflated_sharpe_probability": 0.99,
                },
            },
        },
        {
            "trial_index": 1,
            "parameters": {"topk": 50},
            "status": "succeeded",
            "score": 0.98,
            "warnings": [],
            "metrics": {
                "in_sample": {},
                "out_of_sample": {"deflated_sharpe_probability": 0.98},
            },
        },
    ]
    summary = summarize_trials(trials, {"topk": [30, 50]})
    assert summary["best_trial_index"] == 0
    assert summary["leaderboard"][0]["trial_index"] == 0
    assert summary["leaderboard"][0]["out_of_sample"]["information_ratio"] == 0.8
    assert "robustness" not in summary["leaderboard"][0]["out_of_sample"]
    assert set(summary["warnings"]) == {"fragile_ranking", "boundary_optimum"}
