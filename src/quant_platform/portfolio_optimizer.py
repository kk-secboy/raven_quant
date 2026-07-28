from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .risk_math import COVARIANCE_MODEL_VERSION, validate_covariance


@dataclass(frozen=True)
class PortfolioOptimizationResult:
    weights: pd.Series
    objective: float
    tracking_risk: float
    portfolio_volatility: float
    covariance_model_version: str
    active_share: float
    expected_turnover: float
    max_industry_deviation: float | None
    size_deviation: float | None
    style_deviations: dict[str, float]
    iterations: int


def optimize_benchmark_relative_weights(
    scores: pd.Series,
    benchmark_weights: pd.Series,
    previous_weights: pd.Series,
    *,
    industries: pd.Series,
    benchmark_industry_weights: pd.Series,
    style_exposures: pd.Series | pd.DataFrame,
    benchmark_style_exposure: float | pd.Series | dict[str, float],
    return_covariance: pd.DataFrame,
    max_position_weight: float,
    max_industry_weight: float,
    max_industry_deviation: float,
    max_size_deviation: float,
    max_style_deviations: dict[str, float] | None = None,
    alpha_weight: float = 0.05,
    tracking_penalty: float = 1.0,
    turnover_penalty: float = 0.10,
    max_tracking_error: float = 1.0,
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
    if not 0 < max_tracking_error <= 1:
        raise ValueError("optimizer tracking-error limit must be between zero and one")
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
    styles = _finite_frame(style_exposures, instruments, "style exposures")
    if isinstance(benchmark_style_exposure, pd.Series):
        benchmark_styles = benchmark_style_exposure.astype(float)
    elif isinstance(benchmark_style_exposure, dict):
        benchmark_styles = pd.Series(benchmark_style_exposure, dtype=float)
    else:
        benchmark_styles = pd.Series(
            {str(styles.columns[0]): float(benchmark_style_exposure)}, dtype=float
        )
    benchmark_styles = benchmark_styles.reindex(styles.columns)
    if benchmark_styles.isna().any() or not np.isfinite(benchmark_styles).all():
        raise ValueError("benchmark style exposures must be complete and finite")
    style_limits = {
        column: float((max_style_deviations or {}).get(str(column), max_size_deviation))
        for column in styles.columns
    }

    omitted_benchmark_weight = max(0.0, 1.0 - float(benchmark.sum()))
    previous_cash = max(0.0, 1.0 - float(previous.sum()))
    initial_seed = benchmark + 0.25 * previous + 1.0 / len(instruments)
    initial = _project_capped_simplex(
        initial_seed.to_numpy(dtype=float), max_position_weight, total=1.0
    )
    alpha_values = normalized_alpha.to_numpy(dtype=float)
    previous_values = previous.to_numpy(dtype=float)
    risk_universe = instruments.union(
        pd.Index(benchmark_weights.index.astype(str), name="instrument")
    )
    covariance = return_covariance.copy()
    if not isinstance(covariance, pd.DataFrame) or covariance.empty:
        raise ValueError("optimizer requires a point-in-time return covariance matrix")
    covariance.index = covariance.index.astype(str)
    covariance.columns = covariance.columns.astype(str)
    covariance = covariance.reindex(index=risk_universe, columns=risk_universe)
    if covariance.isna().any().any():
        raise ValueError("optimizer return covariance is incomplete")
    annual_covariance = validate_covariance(covariance.to_numpy(dtype=float)) * 252.0
    selected_locations = np.array([risk_universe.get_loc(item) for item in instruments], dtype=int)
    benchmark_risk = pd.to_numeric(benchmark_weights, errors="coerce")
    benchmark_risk.index = benchmark_risk.index.astype(str)
    benchmark_risk = benchmark_risk.reindex(risk_universe, fill_value=0.0).to_numpy(dtype=float)

    def active_risk_vector(weights: np.ndarray) -> np.ndarray:
        portfolio = np.zeros(len(risk_universe), dtype=float)
        portfolio[selected_locations] = weights
        return portfolio - benchmark_risk

    def objective(weights: np.ndarray) -> float:
        active = active_risk_vector(weights)
        traded = weights - previous_values
        return float(
            tracking_penalty * (active @ annual_covariance @ active)
            - alpha_weight * np.dot(alpha_values, weights)
            + turnover_penalty * np.dot(traded, traded)
        )

    def objective_jacobian(weights: np.ndarray) -> np.ndarray:
        active = active_risk_vector(weights)
        return (
            2.0 * tracking_penalty * (annual_covariance @ active)[selected_locations]
            - alpha_weight * alpha_values
            + 2.0 * turnover_penalty * (weights - previous_values)
        )

    constraints: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda weights: float(weights.sum() - 1.0),
            "jac": lambda weights: np.ones_like(weights),
        },
        {
            "type": "ineq",
            "fun": lambda weights: float(
                max_tracking_error**2
                - active_risk_vector(weights) @ annual_covariance @ active_risk_vector(weights)
            ),
            "jac": lambda weights: (
                -2.0
                * (annual_covariance @ active_risk_vector(weights))[selected_locations]
            ),
        },
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
    for column in styles.columns:
        style_values = styles[column].to_numpy(dtype=float)
        benchmark_style = float(benchmark_styles[column])
        limit = style_limits[str(column)]
        constraints.extend(
            (
                {
                    "type": "ineq",
                    "fun": lambda weights,
                    values=style_values,
                    benchmark=benchmark_style,
                    limit=limit: float(limit - (np.dot(weights, values) - benchmark)),
                    "jac": lambda weights, values=style_values: -values,
                },
                {
                    "type": "ineq",
                    "fun": lambda weights,
                    values=style_values,
                    benchmark=benchmark_style,
                    limit=limit: float(limit + (np.dot(weights, values) - benchmark)),
                    "jac": lambda weights, values=style_values: values,
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
    measured_style_deviations = {
        str(column): abs(float(weights.dot(styles[column])) - float(benchmark_styles[column]))
        for column in styles.columns
    }
    size_deviation = measured_style_deviations.get("size")
    if size_deviation is None:
        size_deviation = measured_style_deviations.get("log_market_cap")
    active = weights - benchmark
    full_active = active_risk_vector(weights.to_numpy(dtype=float))
    tracking_risk = float(np.sqrt(max(0.0, full_active @ annual_covariance @ full_active)))
    portfolio_risk = np.zeros(len(risk_universe), dtype=float)
    portfolio_risk[selected_locations] = weights.to_numpy(dtype=float)
    portfolio_volatility = float(
        np.sqrt(max(0.0, portfolio_risk @ annual_covariance @ portfolio_risk))
    )
    return PortfolioOptimizationResult(
        weights=weights[weights > 0].sort_values(ascending=False),
        objective=float(result.fun),
        tracking_risk=tracking_risk,
        portfolio_volatility=portfolio_volatility,
        covariance_model_version=COVARIANCE_MODEL_VERSION,
        active_share=0.5 * (float(active.abs().sum()) + omitted_benchmark_weight),
        expected_turnover=0.5 * (float((weights - previous).abs().sum()) + previous_cash),
        max_industry_deviation=measured_industry_deviation,
        size_deviation=size_deviation,
        style_deviations=measured_style_deviations,
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


def _finite_frame(
    values: pd.Series | pd.DataFrame, index: pd.Index, label: str
) -> pd.DataFrame:
    frame = (
        values.to_frame(name=values.name or "size")
        if isinstance(values, pd.Series)
        else values.copy()
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.index.is_unique:
        raise ValueError(f"{label} must have a unique index and at least one column")
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    frame = frame.apply(pd.to_numeric, errors="coerce").reindex(index)
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} must contain complete finite values")
    return frame.astype(float)


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
