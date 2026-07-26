import numpy as np
import pandas as pd
import pytest

from quant_platform.statistical_validation import (
    benjamini_hochberg,
    deflated_sharpe_probability,
    newey_west_mean_test,
    purged_embargo_split,
)

pytestmark = pytest.mark.no_database


def test_newey_west_detects_persistent_daily_ic() -> None:
    rng = np.random.default_rng(7)
    daily_ic = pd.Series(0.04 + rng.normal(0, 0.02, 250))
    result = newey_west_mean_test(daily_ic, max_lag=5)
    assert result["p_value"] < 0.01
    assert result["max_lag"] == 5


def test_newey_west_zero_variance_is_explicitly_undefined() -> None:
    """A constant series has zero long-run variance: no finite statistic
    exists, so the result is an explicit undefined state (the Sortino
    ``undefined`` pattern), not inf/0.0."""

    result = newey_west_mean_test(pd.Series([0.03] * 60), max_lag=5)

    assert result["status"] == "undefined_zero_hac_variance"
    assert result["test_statistic"] is None
    assert result["p_value"] is None
    assert result["mean"] == pytest.approx(0.03)
    assert result["standard_error"] == pytest.approx(0.0, abs=1e-12)


def test_benjamini_hochberg_controls_one_experiment_family() -> None:
    q_values = benjamini_hochberg([0.01, 0.04, 0.20, 0.80])
    assert q_values == pytest.approx([0.04, 0.08, 0.2666666667, 0.80])
    assert [value <= 0.10 for value in q_values] == [True, True, False, False]


def test_purge_and_embargo_follow_label_horizon() -> None:
    dates = pd.date_range("2025-01-01", periods=40, freq="B")
    result = purged_embargo_split(
        dates,
        validation_end=dates[19],
        test_start=dates[20],
        label_horizon_days=3,
    )
    assert result["purge_days"] == 3
    assert result["embargo_days"] == 5
    assert result["validation"][-1] == dates[16]
    assert result["test"][0] == dates[25]


def test_deflated_sharpe_penalizes_trial_count() -> None:
    rng = np.random.default_rng(13)
    returns = 0.001 + rng.normal(0, 0.01, 500)
    one = deflated_sharpe_probability(returns, trials=1)
    many = deflated_sharpe_probability(returns, trials=100)
    assert one["probability"] > many["probability"]
    assert many["trials"] == 100
