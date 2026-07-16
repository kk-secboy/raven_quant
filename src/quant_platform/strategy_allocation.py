from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .risk_math import COVARIANCE_MODEL_VERSION, estimate_covariance
from .upstream_versions import QLIB_COMMIT, upstream_runtime_identity


def _load_qlib_portfolio_optimizer() -> type[Any]:
    try:
        from qlib.contrib.strategy.optimizer import PortfolioOptimizer
    except ImportError as exc:  # pragma: no cover - configured runtime assertion
        raise RuntimeError("Qlib PortfolioOptimizer runtime is unavailable") from exc
    return PortfolioOptimizer


def _capped(weights: np.ndarray, maximum: float) -> np.ndarray:
    count = len(weights)
    if count < 2 or maximum < 1.0 / count - 1e-12:
        raise ValueError("max strategy weight is infeasible for the member count")
    values = np.asarray(weights, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or float(values.sum()) <= 0:
        raise ValueError("Qlib allocation weights must be finite and non-negative")
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
        if float(basis.sum()) <= 0:
            basis = np.ones(int(under.sum()), dtype=float)
        values[under] = remaining * basis / basis.sum()
    if (values > maximum + 1e-9).any():
        raise ValueError("unable to satisfy max strategy weight")
    return values / values.sum()


def analyze_strategy_allocation(
    returns: pd.DataFrame,
    *,
    method: str,
    lookback_days: int,
    target_volatility: float,
    max_pairwise_correlation: float,
    max_strategy_weight: float,
    fixed_weights: dict[str, float] | None = None,
    risk_budgets: dict[str, float] | None = None,
    optimizer_factory: Callable[..., Any] | None = None,
    risk_estimator_factory: Callable[..., Any] | None = None,
    runtime_identity: Callable[[str], dict[str, Any]] | None = None,
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
    if risk_budgets is not None and set(risk_budgets) != set(columns):
        raise ValueError("risk budgets require one value for every strategy")
    budget_values = np.array(
        [float((risk_budgets or {}).get(column, 1.0)) for column in columns], dtype=float
    )
    if not np.isfinite(budget_values).all() or (budget_values <= 0).any():
        raise ValueError("strategy risk budgets must be finite and positive")
    target_budget = budget_values / budget_values.sum()
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
    identity_fn = runtime_identity or upstream_runtime_identity
    covariance_frame = estimate_covariance(
        frame,
        estimator_factory=risk_estimator_factory,
        runtime_identity=identity_fn,
    )
    covariance = covariance_frame.to_numpy(dtype=float) * 252.0
    qlib_runtime = identity_fn("qlib")
    if qlib_runtime.get("commit") != QLIB_COMMIT:
        raise RuntimeError("Qlib optimizer is not running from the validated commit")
    solver: dict[str, Any] = {
        "success": True,
        "engine": "qlib.contrib.strategy.optimizer.PortfolioOptimizer",
        "qlib_version": qlib_runtime["version"],
        "qlib_commit": qlib_runtime["commit"],
        "constraint_wrapper": "project_max_member_weight_v1",
        "maximum_risk_budget_error": None,
        "risk_budget_tolerance": None,
    }
    if method in {"risk_parity", "inverse_volatility"}:
        optimizer_cls = optimizer_factory or _load_qlib_portfolio_optimizer()
        qlib_method = "rp" if method == "risk_parity" else "inv"
        optimized = optimizer_cls(method=qlib_method)(S=covariance_frame * 252.0)
        base_weights = _capped(np.asarray(optimized, dtype=float), max_strategy_weight)
        if method == "risk_parity" and not np.allclose(
            target_budget, np.full(len(columns), 1.0 / len(columns)), atol=1e-12
        ):
            def objective(values: np.ndarray) -> float:
                variance = float(values @ covariance @ values)
                if variance <= 0:
                    return 1e6
                contributions = values * (covariance @ values) / variance
                return float(np.square(contributions - target_budget).sum())

            mapped = minimize(
                objective,
                base_weights,
                method="SLSQP",
                bounds=[(1e-8, max_strategy_weight) for _ in columns],
                constraints=[{"type": "eq", "fun": lambda values: float(values.sum() - 1.0)}],
                options={"maxiter": 1000, "ftol": 1e-14},
            )
            if not mapped.success or not np.isfinite(mapped.x).all():
                raise ValueError(
                    "Qlib risk parity result cannot satisfy the governed member risk budgets"
                )
            base_weights = np.asarray(mapped.x, dtype=float)
            solver["constraint_wrapper"] = "project_risk_budget_mapping_v1"
            solver["constraint_iterations"] = int(mapped.nit)
    else:
        if fixed_weights is None or set(fixed_weights) != set(columns):
            raise ValueError("fixed allocation requires one weight for every strategy")
        base_weights = _capped(
            np.array([fixed_weights[column] for column in columns], dtype=float),
            max_strategy_weight,
        )
        solver.update(
            {
                "engine": "project_fixed_weight_constraint_wrapper",
                "constraint_wrapper": "project_max_member_weight_v1",
            }
        )
    portfolio_variance = float(base_weights @ covariance @ base_weights)
    if portfolio_variance <= 0:
        raise ValueError("strategy allocation has no positive portfolio variance")
    marginal = covariance @ base_weights
    raw_contributions = base_weights * marginal
    risk_contributions = raw_contributions / portfolio_variance
    if method == "risk_parity":
        risk_budget_error = float(np.max(np.abs(risk_contributions - target_budget)))
        solver["maximum_risk_budget_error"] = risk_budget_error
        solver["risk_budget_tolerance"] = 0.02
        if float(risk_contributions.min()) < -1e-8 or risk_budget_error > 0.02:
            raise ValueError(
                "Qlib risk parity result violates the governed risk-budget tolerance; "
                "use feasible risk budgets, inverse_volatility, or fixed weights"
            )
    portfolio_volatility = float(np.sqrt(portfolio_variance))
    exposure_scale = min(1.0, target_volatility / portfolio_volatility)
    target_weights = base_weights * exposure_scale
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
                "risk_budget": float(target_budget[index]),
            }
            for index, column in enumerate(columns)
        },
    }
