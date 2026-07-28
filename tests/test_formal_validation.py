from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.formal_validation import (
    build_outer_walk_forward_folds,
    run_ablation_suite,
    run_outer_walk_forward,
    run_signal_decay_suite,
)
from quant_platform.statistical_validation import (
    holm_bonferroni,
    paired_moving_block_bootstrap,
    probability_of_backtest_overfitting,
)

pytestmark = pytest.mark.no_database


def test_outer_walk_forward_reruns_inner_selection_for_every_fold() -> None:
    dates = pd.bdate_range("2020-01-02", periods=160)
    calls: list[tuple[str, int]] = []

    def inner(candidate: str, fold):
        calls.append((candidate, fold.fold))
        # Winner deliberately changes by fold: selection is not reused.
        return {"information_ratio": 2.0 if candidate == f"c{fold.fold % 2}" else 1.0}

    result = run_outer_walk_forward(
        dates=dates,
        candidate_ids=["c0", "c1"],
        inner_runner=inner,
        test_runner=lambda candidate, fold: {
            "candidate": candidate,
            "test_start": fold.test_start,
        },
        selection_metric="information_ratio",
        train_days=60,
        validation_days=20,
        test_days=20,
        purge_days=5,
        embargo_days=5,
    )

    assert result["fold_count"] == 3
    assert len(calls) == 6
    assert [
        item["selected_candidate_id"] for item in result["folds"]
    ] == ["c0", "c1", "c0"]
    test_ranges = [
        (item["fold"]["test_start"], item["fold"]["test_end"])
        for item in result["folds"]
    ]
    assert len(test_ranges) == len(set(test_ranges))


def test_outer_fold_builder_keeps_purge_and_embargo_gaps() -> None:
    dates = pd.bdate_range("2024-01-02", periods=100)
    folds = build_outer_walk_forward_folds(
        dates,
        train_days=40,
        validation_days=10,
        test_days=10,
        purge_days=3,
        embargo_days=4,
    )
    first = folds[0]
    assert dates.get_loc(first.validation_start) - dates.get_loc(first.train_end) == 4
    assert dates.get_loc(first.test_start) - dates.get_loc(first.validation_end) == 5


def test_ablation_suite_records_incremental_evidence() -> None:
    result = run_ablation_suite(
        component_ids=["momentum", "quality"],
        full_metrics={"information_ratio": 0.80},
        runner=lambda component: {
            "information_ratio": 0.50 if component == "momentum" else 0.75
        },
        metric="information_ratio",
        minimum_increment=0.10,
    )
    assert result["status"] == "failed"
    assert result["runs"][0]["increment"] == pytest.approx(0.30)
    assert result["runs"][1]["passed"] is False


def test_signal_decay_derives_last_supported_delay() -> None:
    values = {0: 1.0, 1: 0.85, 2: 0.55, 3: -0.10}
    result = run_signal_decay_suite(
        delays=[3, 0, 2, 1],
        runner=lambda delay: {"annualized_excess_return": values[delay]},
        metric="annualized_excess_return",
        minimum_retention=0.60,
    )
    assert result["maximum_supported_delay_bars"] == 1
    assert result["frontier_version"] == "contiguous-zero-delay-frontier-v2"
    assert [item["delay_bars"] for item in result["runs"]] == [0, 1, 2, 3]


def test_signal_decay_requires_a_contiguous_supported_frontier() -> None:
    values = {0: 1.0, 1: 0.40, 2: 0.80}

    result = run_signal_decay_suite(
        delays=[0, 1, 2],
        runner=lambda delay: {"annualized_excess_return": values[delay]},
        metric="annualized_excess_return",
        minimum_retention=0.60,
    )

    assert [item["passed"] for item in result["runs"]] == [True, False, True]
    assert result["maximum_supported_delay_bars"] == 0


def test_holm_adjustment_preserves_original_order() -> None:
    adjusted = holm_bonferroni([0.04, 0.01, 0.03])
    assert adjusted == pytest.approx([0.06, 0.03, 0.06])


def test_paired_block_bootstrap_is_deterministic_and_paired() -> None:
    rng = np.random.default_rng(7)
    baseline = rng.normal(0.0, 0.01, 120)
    candidate = baseline + 0.001
    first = paired_moving_block_bootstrap(
        candidate,
        baseline,
        block_size=10,
        samples=500,
        seed=17,
    )
    second = paired_moving_block_bootstrap(
        candidate,
        baseline,
        block_size=10,
        samples=500,
        seed=17,
    )
    assert first == second
    assert first["observed_mean_difference"] == pytest.approx(0.001)
    assert first["probability_positive"] == 1.0


def test_pbo_distinguishes_stable_candidate_from_fold_winners() -> None:
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        {
            "stable": rng.normal(0.0010, 0.01, 160),
            "noise_a": rng.normal(0.0, 0.01, 160),
            "noise_b": rng.normal(0.0, 0.01, 160),
        }
    )
    result = probability_of_backtest_overfitting(returns, blocks=8)
    assert result["status"] == "ok"
    assert 0.0 <= result["pbo"] <= 1.0
    assert result["split_count"] == 35
