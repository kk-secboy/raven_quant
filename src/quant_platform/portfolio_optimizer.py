from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class PortfolioOptimizationResult:
    weights: pd.Series
    objective: float
    tracking_risk_proxy: float
    active_share: float
    expected_turnover: float
    max_industry_deviation: float | None
    size_deviation: float | None
    iterations: int


def optimize_benchmark_relative_weights(
    scores: pd.Series,
    benchmark_weights: pd.Series,
    previous_weights: pd.Series,
    *,
    industries: pd.Series,
    benchmark_industry_weights: pd.Series,
    style_exposures: pd.Series,
    benchmark_style_exposure: float,
    max_position_weight: float,
    max_industry_weight: float,
    max_industry_deviation: float,
    max_size_deviation: float,
    alpha_weight: float = 0.05,
    tracking_penalty: float = 1.0,
    turnover_penalty: float = 0.10,
) -> PortfolioOptimizationResult:
    """Assign long-only weights to a governed candidate set relative to a benchmark.

    Candidate selection remains a separate, auditable Top-K step. This convex stage
    owns continuous weights and fails closed if point-in-time exposure data is missing
    or the persisted limits cannot be satisfied.
    """

    if not isinstance(scores, pd.Series) or scores.empty or not scores.index.is_unique:
        raise ValueError("optimizer scores must be a unique non-empty Series")
    if not 0 < max_position_weight <= 1:
        raise ValueError("optimizer max position weight must be between zero and one")
    if len(scores) * max_position_weight < 1.0 - 1e-10:
        raise ValueError("optimizer candidate set cannot form a fully invested portfolio")
    if not 0 < max_industry_weight <= 1 or not 0 <= max_industry_deviation <= 1:
        raise ValueError("optimizer industry limits are invalid")
    if max_size_deviation < 0:
        raise ValueError("optimizer size limit must be non-negative")
    if min(alpha_weight, tracking_penalty, turnover_penalty) < 0:
        raise ValueError("optimizer objective weights must be non-negative")
    if tracking_penalty == 0 and alpha_weight == 0 and turnover_penalty == 0:
        raise ValueError("optimizer objective must contain a positive weight")

    instruments = pd.Index(scores.index.astype(str), name="instrument")
    alpha = _finite_series(scores, instruments, "optimizer scores")
    alpha_std = float(alpha.std(ddof=0))
    normalized_alpha = (alpha - alpha.mean()) / alpha_std if alpha_std > 0 else alpha * 0.0
    benchmark = _non_negative_series(benchmark_weights, instruments, "benchmark weights")
    previous = _non_negative_series(previous_weights, instruments, "previous weights")
    industry = _categorical_series(industries, instruments, "industry memberships")
    benchmark_industries = _non_negative_series(
        benchmark_industry_weights,
        pd.Index(benchmark_industry_weights.index.astype(str)),
        "benchmark industry weights",
    )
    styles = _finite_series(style_exposures, instruments, "style exposures")
    if not np.isfinite(float(benchmark_style_exposure)):
        raise ValueError("benchmark style exposure must be finite")

    omitted_benchmark_weight = max(0.0, 1.0 - float(benchmark.sum()))
    previous_cash = max(0.0, 1.0 - float(previous.sum()))
    initial_seed = benchmark + 0.25 * previous + 1.0 / len(instruments)
    initial = _project_capped_simplex(
        initial_seed.to_numpy(dtype=float), max_position_weight, total=1.0
    )
    alpha_values = normalized_alpha.to_numpy(dtype=float)
    benchmark_values = benchmark.to_numpy(dtype=float)
    previous_values = previous.to_numpy(dtype=float)
    style_values = styles.to_numpy(dtype=float)

    def objective(weights: np.ndarray) -> float:
        active = weights - benchmark_values
        traded = weights - previous_values
        return float(
            tracking_penalty * np.dot(active, active)
            - alpha_weight * np.dot(alpha_values, weights)
            + turnover_penalty * np.dot(traded, traded)
        )

    def objective_jacobian(weights: np.ndarray) -> np.ndarray:
        return (
            2.0 * tracking_penalty * (weights - benchmark_values)
            - alpha_weight * alpha_values
            + 2.0 * turnover_penalty * (weights - previous_values)
        )

    constraints: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda weights: float(weights.sum() - 1.0),
            "jac": lambda weights: np.ones_like(weights),
        }
    ]
    industry_names = list(dict.fromkeys([*industry.tolist(), *benchmark_industries.index]))
    for name in industry_names:
        mask = industry.eq(str(name)).to_numpy(dtype=bool)
        benchmark_weight = float(benchmark_industries.get(str(name), 0.0))
        lower = max(0.0, benchmark_weight - max_industry_deviation)
        upper = min(max_industry_weight, benchmark_weight + max_industry_deviation)
        if lower > upper + 1e-10 or (not mask.any() and lower > 1e-10):
            raise ValueError(f"optimizer industry constraint is infeasible: {name}")
        constraints.extend(
            (
                {
                    "type": "ineq",
                    "fun": lambda weights, mask=mask, lower=lower: float(
                        weights[mask].sum() - lower
                    ),
                    "jac": lambda weights, mask=mask: mask.astype(float),
                },
                {
                    "type": "ineq",
                    "fun": lambda weights, mask=mask, upper=upper: float(
                        upper - weights[mask].sum()
                    ),
                    "jac": lambda weights, mask=mask: -mask.astype(float),
                },
            )
        )
    benchmark_style = float(benchmark_style_exposure)
    constraints.extend(
        (
            {
                "type": "ineq",
                "fun": lambda weights: float(
                    max_size_deviation - (np.dot(weights, style_values) - benchmark_style)
                ),
                "jac": lambda weights: -style_values,
            },
            {
                "type": "ineq",
                "fun": lambda weights: float(
                    max_size_deviation + (np.dot(weights, style_values) - benchmark_style)
                ),
                "jac": lambda weights: style_values,
            },
        )
    )

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=objective_jacobian,
        bounds=[(0.0, max_position_weight)] * len(instruments),
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 200, "disp": False},
    )
    if not result.success:
        raise ValueError(f"benchmark-relative optimizer failed: {result.message}")

    weights = pd.Series(result.x, index=instruments, dtype=float)
    weights[weights.abs() < 1e-10] = 0.0
    _validate_solution(weights, max_position_weight, constraints)
    portfolio_industries = weights.groupby(industry).sum()
    industry_labels = portfolio_industries.index.union(benchmark_industries.index)
    measured_industry_deviation = float(
        (
            portfolio_industries.reindex(industry_labels, fill_value=0.0)
            - benchmark_industries.reindex(industry_labels, fill_value=0.0)
        )
        .abs()
        .max()
    )
    measured_size_deviation = abs(
        float(weights.dot(styles)) - float(benchmark_style_exposure)
    )
    active = weights - benchmark
    return PortfolioOptimizationResult(
        weights=weights[weights > 0].sort_values(ascending=False),
        objective=float(result.fun),
        tracking_risk_proxy=float(
            np.sqrt(np.dot(active, active) + omitted_benchmark_weight**2)
        ),
        active_share=0.5 * (float(active.abs().sum()) + omitted_benchmark_weight),
        expected_turnover=0.5
        * (float((weights - previous).abs().sum()) + previous_cash),
        max_industry_deviation=measured_industry_deviation,
        size_deviation=measured_size_deviation,
        iterations=int(result.nit),
    )


def _finite_series(values: pd.Series, index: pd.Index, label: str) -> pd.Series:
    if not isinstance(values, pd.Series) or not values.index.is_unique:
        raise ValueError(f"{label} must be a Series with a unique index")
    normalized = pd.to_numeric(values, errors="coerce")
    normalized.index = normalized.index.astype(str)
    normalized = normalized.reindex(index)
    if normalized.isna().any() or not np.isfinite(normalized.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} must contain complete finite values")
    return normalized.astype(float)


def _non_negative_series(values: pd.Series, index: pd.Index, label: str) -> pd.Series:
    normalized = _finite_series(values.reindex(index, fill_value=0.0), index, label)
    if (normalized < 0).any():
        raise ValueError(f"{label} must be non-negative")
    return normalized


def _categorical_series(values: pd.Series, index: pd.Index, label: str) -> pd.Series:
    if not isinstance(values, pd.Series) or not values.index.is_unique:
        raise ValueError(f"{label} must be a Series with a unique index")
    normalized = values.copy()
    normalized.index = normalized.index.astype(str)
    normalized = normalized.reindex(index)
    if normalized.isna().any() or normalized.astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} must contain complete point-in-time values")
    return normalized.astype(str)


def _project_capped_simplex(values: np.ndarray, cap: float, *, total: float) -> np.ndarray:
    if len(values) * cap < total - 1e-12:
        raise ValueError("capped simplex is infeasible")
    low = float(values.min() - cap)
    high = float(values.max())
    for _ in range(100):
        midpoint = (low + high) / 2.0
        projected = np.clip(values - midpoint, 0.0, cap)
        if projected.sum() > total:
            low = midpoint
        else:
            high = midpoint
    projected = np.clip(values - high, 0.0, cap)
    residual = total - float(projected.sum())
    if abs(residual) > 1e-10:
        available = np.flatnonzero(projected < cap - 1e-12)
        if len(available) == 0:
            raise ValueError("capped simplex projection did not converge")
        projected[available[0]] += residual
    return projected


def _validate_solution(
    weights: pd.Series,
    max_position_weight: float,
    constraints: list[dict[str, Any]],
) -> None:
    tolerance = 1e-6
    if abs(float(weights.sum()) - 1.0) > tolerance:
        raise ValueError("optimizer returned a portfolio that is not fully invested")
    if float(weights.min()) < -tolerance or float(weights.max()) > max_position_weight + tolerance:
        raise ValueError("optimizer returned a position outside configured bounds")
    values = weights.to_numpy(dtype=float)
    for constraint in constraints:
        measured = float(constraint["fun"](values))
        if constraint["type"] == "eq" and abs(measured) > tolerance:
            raise ValueError("optimizer equality constraint was not satisfied")
        if constraint["type"] == "ineq" and measured < -tolerance:
            raise ValueError("optimizer inequality constraint was not satisfied")
