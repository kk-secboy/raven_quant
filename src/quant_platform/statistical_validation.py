from __future__ import annotations

from math import e, isfinite, sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

STATISTICAL_CONTRACT_VERSION = "research-statistics-v1-hac-bh-dsr"


def newey_west_mean_test(values: pd.Series | list[float], *, max_lag: int) -> dict[str, Any]:
    sample = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(sample) < 10:
        raise ValueError("HAC significance requires at least 10 daily observations")
    if not np.isfinite(sample).all() or max_lag < 0:
        raise ValueError("HAC significance inputs are invalid")
    lag = min(int(max_lag), len(sample) - 1)
    centered = sample - sample.mean()
    gamma_zero = float(centered @ centered / len(sample))
    long_run_variance = gamma_zero
    for offset in range(1, lag + 1):
        covariance = float(centered[offset:] @ centered[:-offset] / len(sample))
        long_run_variance += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    long_run_variance = max(0.0, long_run_variance)
    standard_error = sqrt(long_run_variance / len(sample))
    mean = float(sample.mean())
    # Degenerate long-run variance (constant series or exact cancellation):
    # no finite statistic exists. Report an explicit undefined state (same
    # pattern as sortino_status) instead of inf so downstream gates treat it
    # as insufficient evidence. The tolerance is relative to the sample scale
    # because floating-point residue keeps a truly constant series' variance
    # marginally above zero.
    scale = max(float(np.abs(sample).max()), 1e-8)
    if standard_error <= scale * 1e-12:
        statistic = None
        p_value = None
        status = "undefined_zero_hac_variance"
    else:
        statistic = mean / standard_error
        p_value = float(2.0 * norm.sf(abs(statistic)))
        status = "ok"
    return {
        "mean": mean,
        "standard_error": standard_error,
        "test_statistic": statistic,
        "p_value": p_value,
        "max_lag": lag,
        "observations": len(sample),
        "status": status,
        "contract_version": STATISTICAL_CONTRACT_VERSION,
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if len(values) == 0:
        return []
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("BH-FDR p-values must be finite and in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result.tolist()


def purged_embargo_split(
    dates: pd.DatetimeIndex,
    *,
    validation_end: pd.Timestamp,
    test_start: pd.Timestamp,
    label_horizon_days: int,
) -> dict[str, pd.DatetimeIndex | int]:
    if label_horizon_days < 1:
        raise ValueError("label horizon must be positive")
    ordered = pd.DatetimeIndex(pd.to_datetime(dates).unique()).sort_values()
    embargo = max(5, label_horizon_days)
    valid_candidates = ordered[ordered <= pd.Timestamp(validation_end)]
    test_candidates = ordered[ordered >= pd.Timestamp(test_start)]
    purged_validation = valid_candidates[: max(0, len(valid_candidates) - label_horizon_days)]
    embargoed_test = test_candidates[embargo:]
    if len(purged_validation) == 0 or len(embargoed_test) == 0:
        raise ValueError("purge and embargo leave an empty evaluation window")
    return {
        "validation": purged_validation,
        "test": embargoed_test,
        "purge_days": label_horizon_days,
        "embargo_days": embargo,
    }


def deflated_sharpe_probability(
    daily_returns: pd.Series | list[float], *, trials: int
) -> dict[str, Any]:
    values = np.asarray(pd.Series(daily_returns).dropna(), dtype=float)
    if len(values) < 30 or trials < 1 or not np.isfinite(values).all():
        raise ValueError("Deflated Sharpe requires finite returns and at least 30 observations")
    standard_deviation = float(values.std(ddof=1))
    if standard_deviation <= 0:
        return {
            "probability": 0.0,
            "status": "zero_return_variance",
            "trials": trials,
            "contract_version": STATISTICAL_CONTRACT_VERSION,
        }
    daily_sharpe = float(values.mean() / standard_deviation)
    skewness = float(pd.Series(values).skew())
    kurtosis = float(pd.Series(values).kurt() + 3.0)
    sharpe_variance = max(
        1e-12,
        (1.0 - skewness * daily_sharpe + (kurtosis - 1.0) * daily_sharpe**2 / 4.0)
        / (len(values) - 1),
    )
    if trials == 1:
        expected_maximum = 0.0
    else:
        euler_gamma = 0.5772156649015329
        expected_maximum = sqrt(sharpe_variance) * (
            (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / trials)
            + euler_gamma * norm.ppf(1.0 - 1.0 / (trials * e))
        )
    denominator = sqrt(
        max(
            1e-12,
            (1.0 - skewness * daily_sharpe + (kurtosis - 1.0) * daily_sharpe**2 / 4.0)
            / (len(values) - 1),
        )
    )
    probability = float(norm.cdf((daily_sharpe - expected_maximum) / denominator))
    if not isfinite(probability):
        probability = 0.0
        status = "non_finite"
    else:
        status = "ok"
    return {
        "probability": probability,
        "status": status,
        "trials": trials,
        "daily_sharpe": daily_sharpe,
        "annualized_sharpe": daily_sharpe * sqrt(252.0),
        "expected_maximum_daily_sharpe": expected_maximum,
        "observations": len(values),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "contract_version": STATISTICAL_CONTRACT_VERSION,
    }
