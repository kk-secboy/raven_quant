from __future__ import annotations

from datetime import date

import pytest

from quant_platform.parameter_experiments import (
    evaluate_trial,
    normalize_parameter_grid,
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


def test_period_split_is_non_overlapping_and_requires_history() -> None:
    periods = split_research_period(date(2024, 1, 1), date(2026, 1, 1))
    assert periods["in_sample"] == {"start": "2024-01-01", "end": "2025-03-14"}
    assert periods["out_of_sample"] == {"start": "2025-03-20", "end": "2026-01-01"}
    assert periods["purge_days"] == 1
    assert periods["embargo_days"] == 5
    with pytest.raises(ValueError, match="126"):
        split_research_period(date(2025, 1, 1), date(2025, 4, 1))


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
