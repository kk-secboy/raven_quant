from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .risk_math import COVARIANCE_MODEL_VERSION, regularize_covariance


def _capped(weights: np.ndarray, maximum: float) -> np.ndarray:
    count = len(weights)
    if count < 2 or maximum < 1.0 / count - 1e-12:
        raise ValueError("max strategy weight is infeasible for the member count")
    values = np.asarray(weights, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("allocation weights must be finite and positive")
    values /= values.sum()
    for _ in range(count + 2):
        over = values > maximum + 1e-12
        if not over.any():
            return values / values.sum()
        values[over] = maximum
        remaining = 1.0 - float(values[over].sum())
        under = ~over
        if remaining <= 0 or not under.any():
            break
        basis = values[under]
        values[under] = remaining * basis / basis.sum()
    if (values > maximum + 1e-9).any():
        raise ValueError("unable to satisfy max strategy weight")
    return values / values.sum()


def _risk_parity(
    covariance: np.ndarray, maximum: float, *, tolerance: float = 0.02
) -> tuple[np.ndarray, dict[str, Any]]:
    count = covariance.shape[0]
    target = np.full(count, 1.0 / count, dtype=float)

    def contributions(weights: np.ndarray) -> np.ndarray:
        marginal = covariance @ weights
        variance = float(weights @ marginal)
        if variance <= 0 or not np.isfinite(variance):
            return np.full(count, np.nan)
        return weights * marginal / variance

    def objective(weights: np.ndarray) -> float:
        measured = contributions(weights)
        if not np.isfinite(measured).all():
            return 1e12
        return float(np.square(measured - target).sum())

    initial = _capped(1.0 / np.sqrt(np.diag(covariance)), maximum)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, maximum)] * count,
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
    )
    weights = np.asarray(result.x, dtype=float)
    measured = contributions(weights)
    error = float(np.max(np.abs(measured - target))) if np.isfinite(measured).all() else np.inf
    if (
        not result.success
        or not np.isfinite(weights).all()
        or abs(float(weights.sum()) - 1.0) > 1e-7
        or float(weights.min()) < -1e-9
        or float(weights.max()) > maximum + 1e-7
        or not np.isfinite(measured).all()
        or float(measured.min()) < -1e-8
        or error > tolerance
    ):
        raise ValueError(
            "risk parity optimization did not produce a feasible equal-risk allocation; "
            "use inverse_volatility or fixed weights"
        )
    return weights, {
        "success": True,
        "message": str(result.message),
        "iterations": int(result.nit),
        "maximum_risk_budget_error": error,
        "risk_budget_tolerance": tolerance,
    }


def analyze_strategy_allocation(
    returns: pd.DataFrame,
    *,
    method: str,
    lookback_days: int,
    target_volatility: float,
    max_pairwise_correlation: float,
    max_strategy_weight: float,
    fixed_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    if method not in {"risk_parity", "inverse_volatility", "fixed"}:
        raise ValueError("allocation method must be risk_parity, inverse_volatility, or fixed")
    if lookback_days < 60:
        raise ValueError("allocation lookback must be at least 60 trading days")
    if not 0 < target_volatility <= 0.50:
        raise ValueError("target volatility must be between 0 and 0.50")
    if not -1 < max_pairwise_correlation < 1:
        raise ValueError("max pairwise correlation must be between -1 and 1")
    if not 0 < max_strategy_weight <= 1:
        raise ValueError("max strategy weight must be between 0 and 1")
    frame = returns.copy().replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if frame.shape[1] < 2:
        raise ValueError("a strategy allocation requires at least two return series")
    if len(frame) < lookback_days:
        raise ValueError(
            f"strategy return overlap has {len(frame)} days; {lookback_days} are required"
        )
    frame = frame.tail(lookback_days)
    columns = [str(item) for item in frame.columns]
    frame.columns = columns
    daily_volatility = frame.std(ddof=1)
    if daily_volatility.isna().any() or (daily_volatility <= 0).any():
        raise ValueError("every strategy needs non-zero return volatility")
    correlation = frame.corr()
    pairwise = [
        float(correlation.iloc[row, column])
        for row in range(len(columns))
        for column in range(row + 1, len(columns))
    ]
    highest_correlation = max(pairwise)
    if highest_correlation > max_pairwise_correlation + 1e-12:
        raise ValueError(
            f"strategy correlation {highest_correlation:.4f} exceeds {max_pairwise_correlation:.4f}"
        )
    covariance = regularize_covariance(frame.cov().to_numpy(dtype=float)) * 252.0
    solver: dict[str, Any] = {
        "success": True,
        "message": "closed-form allocation",
        "iterations": 0,
        "maximum_risk_budget_error": None,
        "risk_budget_tolerance": None,
    }
    if method == "risk_parity":
        base_weights, solver = _risk_parity(covariance, max_strategy_weight)
    elif method == "inverse_volatility":
        inverse = 1.0 / (daily_volatility.to_numpy(dtype=float) * np.sqrt(252.0))
        base_weights = _capped(inverse, max_strategy_weight)
    else:
        if fixed_weights is None or set(fixed_weights) != set(columns):
            raise ValueError("fixed allocation requires one weight for every strategy")
        base_weights = _capped(
            np.array([fixed_weights[column] for column in columns], dtype=float),
            max_strategy_weight,
        )
    portfolio_variance = float(base_weights @ covariance @ base_weights)
    if portfolio_variance <= 0:
        raise ValueError("strategy allocation has no positive portfolio variance")
    portfolio_volatility = float(np.sqrt(portfolio_variance))
    exposure_scale = min(1.0, target_volatility / portfolio_volatility)
    target_weights = base_weights * exposure_scale
    marginal = covariance @ base_weights
    raw_contributions = base_weights * marginal
    risk_contributions = raw_contributions / portfolio_variance
    annualized_volatility = daily_volatility * np.sqrt(252.0)
    return {
        "method": method,
        "lookback_days": lookback_days,
        "period_start": frame.index[0].isoformat(),
        "period_end": frame.index[-1].isoformat(),
        "observations": len(frame),
        "highest_pairwise_correlation": highest_correlation,
        "correlation": {
            row: {column: float(correlation.loc[row, column]) for column in columns}
            for row in columns
        },
        "portfolio_volatility": portfolio_volatility,
        "target_volatility": target_volatility,
        "exposure_scale": exposure_scale,
        "covariance_model_version": COVARIANCE_MODEL_VERSION,
        "solver": solver,
        "cash_weight": 1.0 - float(target_weights.sum()),
        "members": {
            column: {
                "unscaled_weight": float(base_weights[index]),
                "target_weight": float(target_weights[index]),
                "annualized_volatility": float(annualized_volatility[column]),
                "risk_contribution": float(risk_contributions[index]),
            }
            for index, column in enumerate(columns)
        },
    }
