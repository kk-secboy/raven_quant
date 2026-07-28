from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.portfolio_optimizer import optimize_benchmark_relative_weights
from quant_platform.risk_math import validate_covariance

pytestmark = pytest.mark.no_database


def _covariance(instruments: pd.Index) -> pd.DataFrame:
    values = np.full((len(instruments), len(instruments)), 0.00002, dtype=float)
    np.fill_diagonal(values, 0.00010)
    return pd.DataFrame(values, index=instruments, columns=instruments)


def test_benchmark_relative_optimizer_enforces_exposure_limits() -> None:
    instruments = pd.Index([f"stock-{index:02d}" for index in range(10)])
    scores = pd.Series(range(10), index=instruments, dtype=float)
    benchmark = pd.Series(0.10, index=instruments, dtype=float)
    previous = pd.Series(0.10, index=instruments, dtype=float)
    industries = pd.Series(["bank"] * 5 + ["technology"] * 5, index=instruments)
    benchmark_industries = pd.Series({"bank": 0.50, "technology": 0.50})
    styles = pd.Series(
        [-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0],
        index=instruments,
    )

    result = optimize_benchmark_relative_weights(
        scores,
        benchmark,
        previous,
        industries=industries,
        benchmark_industry_weights=benchmark_industries,
        style_exposures=styles,
        benchmark_style_exposure=0.0,
        return_covariance=_covariance(instruments),
        max_position_weight=0.15,
        max_industry_weight=0.60,
        max_industry_deviation=0.05,
        max_size_deviation=0.10,
        alpha_weight=0.10,
        tracking_penalty=1.0,
        turnover_penalty=0.20,
        max_tracking_error=0.01,
    )

    assert result.weights.sum() == pytest.approx(1.0)
    assert result.weights.max() <= 0.15 + 1e-8
    industry_weights = result.weights.groupby(industries.reindex(result.weights.index)).sum()
    assert abs(industry_weights["bank"] - 0.50) <= 0.05 + 1e-6
    assert abs(industry_weights["technology"] - 0.50) <= 0.05 + 1e-6
    assert result.max_industry_deviation is not None
    assert result.max_industry_deviation <= 0.05 + 1e-6
    assert result.size_deviation is not None and result.size_deviation <= 0.10 + 1e-6
    assert result.iterations > 0
    active = result.weights.reindex(instruments, fill_value=0.0) - benchmark
    expected_tracking_risk = float(
        np.sqrt(
            active.to_numpy()
            @ (validate_covariance(_covariance(instruments).to_numpy()) * 252.0)
            @ active
        )
    )
    assert result.tracking_risk == pytest.approx(expected_tracking_risk)
    assert result.tracking_risk <= 0.01 + 1e-6
    assert result.portfolio_volatility > 0


def test_benchmark_relative_optimizer_fails_closed_on_infeasible_book() -> None:
    instruments = pd.Index(["one", "two", "three"])
    with pytest.raises(ValueError, match="fully invested"):
        optimize_benchmark_relative_weights(
            pd.Series([3.0, 2.0, 1.0], index=instruments),
            pd.Series([0.4, 0.3, 0.3], index=instruments),
            pd.Series(dtype=float),
            industries=pd.Series(["one", "two", "three"], index=instruments),
            benchmark_industry_weights=pd.Series({"one": 0.4, "two": 0.3, "three": 0.3}),
            style_exposures=pd.Series(0.0, index=instruments),
            benchmark_style_exposure=0.0,
            return_covariance=_covariance(instruments),
            max_position_weight=0.20,
            max_industry_weight=1.0,
            max_industry_deviation=1.0,
            max_size_deviation=10.0,
        )


def test_benchmark_relative_optimizer_enforces_multiple_style_dimensions() -> None:
    instruments = pd.Index([f"stock-{index:02d}" for index in range(10)])
    styles = pd.DataFrame(
        {
            "size": [-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0],
            "value": [1.0, 0.8, 0.6, 0.4, 0.2, -0.2, -0.4, -0.6, -0.8, -1.0],
            "growth": [-0.5, 0.5] * 5,
            "volatility": [0.5, -0.5] * 5,
        },
        index=instruments,
    )
    result = optimize_benchmark_relative_weights(
        pd.Series(range(10), index=instruments, dtype=float),
        pd.Series(0.10, index=instruments),
        pd.Series(0.10, index=instruments),
        industries=pd.Series(["all"] * 10, index=instruments),
        benchmark_industry_weights=pd.Series({"all": 1.0}),
        style_exposures=styles,
        benchmark_style_exposure=pd.Series(0.0, index=styles.columns),
        return_covariance=_covariance(instruments),
        max_position_weight=0.15,
        max_industry_weight=1.0,
        max_industry_deviation=0.0,
        max_size_deviation=0.10,
        max_style_deviations={column: 0.10 for column in styles.columns},
    )
    assert set(result.style_deviations) == set(styles.columns)
    assert max(result.style_deviations.values()) <= 0.10 + 1e-6


def test_optimizer_preserves_exactly_zero_size_deviation() -> None:
    instruments = pd.Index([f"stock-{index:02d}" for index in range(10)])
    result = optimize_benchmark_relative_weights(
        pd.Series(range(10), index=instruments, dtype=float),
        pd.Series(0.10, index=instruments),
        pd.Series(0.10, index=instruments),
        industries=pd.Series(["all"] * 10, index=instruments),
        benchmark_industry_weights=pd.Series({"all": 1.0}),
        style_exposures=pd.Series(0.0, index=instruments, name="size"),
        benchmark_style_exposure=0.0,
        return_covariance=_covariance(instruments),
        max_position_weight=0.15,
        max_industry_weight=1.0,
        max_industry_deviation=0.0,
        max_size_deviation=0.10,
    )

    assert result.style_deviations["size"] == pytest.approx(0.0)
    assert result.size_deviation == pytest.approx(0.0)


def test_benchmark_relative_optimizer_requires_point_in_time_membership() -> None:
    instruments = pd.Index([f"stock-{index}" for index in range(5)])
    with pytest.raises(ValueError, match="point-in-time"):
        optimize_benchmark_relative_weights(
            pd.Series(range(5), index=instruments, dtype=float),
            pd.Series(0.20, index=instruments),
            pd.Series(0.20, index=instruments),
            industries=pd.Series(["bank", "bank", None, "tech", "tech"], index=instruments),
            benchmark_industry_weights=pd.Series({"bank": 0.50, "tech": 0.50}),
            style_exposures=pd.Series(0.0, index=instruments),
            benchmark_style_exposure=0.0,
            return_covariance=_covariance(instruments),
            max_position_weight=0.30,
            max_industry_weight=0.70,
            max_industry_deviation=0.20,
            max_size_deviation=10.0,
        )
