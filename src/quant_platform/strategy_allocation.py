from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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


def _risk_parity(covariance: np.ndarray, maximum: float) -> np.ndarray:
    count = covariance.shape[0]
    weights = np.ones(count, dtype=float) / count
    for _ in range(1000):
        marginal = covariance @ weights
        variance = float(weights @ marginal)
        if variance <= 0:
            raise ValueError("strategy covariance does not define positive portfolio risk")
        contributions = weights * marginal
        target = variance / count
        updated = weights * np.sqrt(target / np.clip(contributions, 1e-12, None))
        updated = _capped(updated, maximum)
        if np.max(np.abs(updated - weights)) < 1e-10:
            return updated
        weights = updated
    return weights


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
    covariance = frame.cov().to_numpy(dtype=float) * 252.0
    if method == "risk_parity":
        base_weights = _risk_parity(covariance, max_strategy_weight)
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
