from __future__ import annotations

from itertools import combinations
from math import e, isfinite, log, sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

# The existing HAC/BH/DSR factor-evaluation payload remains wire compatible.
# Holm, paired bootstrap and PBO are additive evidence blocks, so they do not
# invalidate already frozen factor artifacts or require a schema migration.
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


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Family-wise adjusted p-values for a frozen small candidate family."""

    values = np.asarray(p_values, dtype=float)
    if len(values) == 0:
        return []
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Holm p-values must be finite and in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.maximum.accumulate(
        ranked * np.arange(len(values), 0, -1, dtype=float)
    )
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result.tolist()


def paired_moving_block_bootstrap(
    candidate_returns: pd.Series | list[float],
    baseline_returns: pd.Series | list[float],
    *,
    block_size: int,
    samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired circular moving-block bootstrap for mean return improvement."""

    candidate = np.asarray(pd.Series(candidate_returns), dtype=float)
    baseline = np.asarray(pd.Series(baseline_returns), dtype=float)
    if (
        len(candidate) != len(baseline)
        or len(candidate) < 30
        or not np.isfinite(candidate).all()
        or not np.isfinite(baseline).all()
    ):
        raise ValueError("paired bootstrap requires equal finite samples of at least 30")
    if not 1 <= block_size <= len(candidate) or samples < 100:
        raise ValueError("paired bootstrap block size or sample count is invalid")
    difference = candidate - baseline
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(len(difference) / block_size))
    estimates = np.empty(samples, dtype=float)
    offsets = np.arange(block_size)
    for sample_no in range(samples):
        starts = rng.integers(0, len(difference), size=block_count)
        indices = ((starts[:, None] + offsets) % len(difference)).ravel()[
            : len(difference)
        ]
        estimates[sample_no] = float(difference[indices].mean())
    observed = float(difference.mean())
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "status": "ok",
        "observed_mean_difference": observed,
        "confidence_interval_95": [float(lower), float(upper)],
        "probability_positive": float(np.mean(estimates > 0.0)),
        "one_sided_p_value": float((np.sum(estimates <= 0.0) + 1) / (samples + 1)),
        "block_size": int(block_size),
        "samples": int(samples),
        "seed": int(seed),
        "observations": len(difference),
        "contract_version": STATISTICAL_CONTRACT_VERSION,
    }


def probability_of_backtest_overfitting(
    trial_returns: pd.DataFrame,
    *,
    blocks: int = 8,
) -> dict[str, Any]:
    """Combinatorially symmetric cross-validation PBO diagnostic.

    Each column is a frozen candidate/trial and each row an aligned return.
    PBO is the fraction of splits where the in-sample winner ranks below the
    test-set median.  It is diagnostic evidence, not a universal approval
    certificate.
    """

    if (
        not isinstance(trial_returns, pd.DataFrame)
        or trial_returns.shape[0] < 40
        or trial_returns.shape[1] < 2
        or blocks < 4
        or blocks % 2
        or trial_returns.shape[0] < blocks * 5
    ):
        raise ValueError("PBO requires at least two trials and even non-trivial blocks")
    values = trial_returns.apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("PBO return matrix must be complete and finite")
    partitions = np.array_split(np.arange(len(values)), blocks)
    train_size = blocks // 2
    logits: list[float] = []
    winners: list[str] = []
    # Complementary splits carry the same information; keep one canonical
    # member of each pair to avoid double weighting.
    for selected in combinations(range(blocks), train_size):
        complement = tuple(index for index in range(blocks) if index not in selected)
        if selected > complement:
            continue
        train_index = np.concatenate([partitions[index] for index in selected])
        test_index = np.concatenate([partitions[index] for index in complement])
        train = values.iloc[train_index]
        test = values.iloc[test_index]
        train_std = train.std(ddof=1).replace(0.0, np.nan)
        train_scores = (train.mean() / train_std).replace(
            [np.inf, -np.inf], np.nan
        )
        if train_scores.notna().sum() < 2:
            continue
        winner = str(train_scores.idxmax())
        test_std = test.std(ddof=1).replace(0.0, np.nan)
        test_scores = (test.mean() / test_std).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if winner not in test_scores or len(test_scores) < 2:
            continue
        # Rank 1 is worst, N is best. Map to (0,1) without endpoints.
        rank = float(test_scores.rank(method="average")[winner])
        relative_rank = (rank - 0.5) / len(test_scores)
        logits.append(log(relative_rank / (1.0 - relative_rank)))
        winners.append(winner)
    if not logits:
        return {
            "status": "undefined_degenerate_trials",
            "pbo": None,
            "split_count": 0,
            "contract_version": STATISTICAL_CONTRACT_VERSION,
        }
    return {
        "status": "ok",
        "pbo": float(np.mean(np.asarray(logits) <= 0.0)),
        "split_count": len(logits),
        "median_logit_rank": float(np.median(logits)),
        "winner_counts": {
            candidate: winners.count(candidate) for candidate in sorted(set(winners))
        },
        "blocks": int(blocks),
        "trials": int(values.shape[1]),
        "observations": int(values.shape[0]),
        "contract_version": STATISTICAL_CONTRACT_VERSION,
    }


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
