from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from .portfolio_optimizer import validate_covariance

DISCRETE_CONSTRAINT_MODEL_VERSION = "discrete-constraint-validation-v1"
_TOLERANCE = 1e-8


def _finite_weights(
    values: pd.Series | dict[str, float],
    *,
    label: str,
) -> pd.Series:
    result = (
        values.copy()
        if isinstance(values, pd.Series)
        else pd.Series(values, dtype=float)
    )
    result.index = result.index.astype(str)
    if not result.index.is_unique:
        raise ValueError(f"{label} must have a unique instrument index")
    result = pd.to_numeric(result, errors="coerce").astype(float)
    if result.isna().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} must contain finite values")
    return result


def _categorical_mapping(
    values: pd.Series | dict[str, str] | None,
    *,
    index: pd.Index,
    label: str,
    required: bool,
) -> pd.Series | None:
    if values is None:
        if required:
            raise ValueError(f"{label} is required by the frozen constraint contract")
        return None
    result = (
        values.copy()
        if isinstance(values, pd.Series)
        else pd.Series(values, dtype=str)
    )
    result.index = result.index.astype(str)
    result = result.reindex(index)
    if result.isna().any() or (result.astype(str).str.strip() == "").any():
        raise ValueError(f"{label} must cover every target or existing instrument")
    return result.astype(str)


def _record(
    checks: list[dict[str, Any]],
    *,
    name: str,
    observed: float,
    limit: float,
    relation: str,
    passed: bool,
    scope: str = "account",
) -> None:
    checks.append(
        {
            "name": name,
            "scope": scope,
            "observed": float(observed),
            "limit": float(limit),
            "relation": relation,
            "passed": bool(passed),
        }
    )


def validate_discrete_constraints(
    target_weights: pd.Series | dict[str, float],
    previous_weights: pd.Series | dict[str, float],
    *,
    max_position_weight: float,
    max_daily_turnover: float,
    min_cash_weight: float = 0.0,
    industries: pd.Series | dict[str, str] | None = None,
    max_industry_weight: float | None = None,
    benchmark_industry_weights: pd.Series | dict[str, float] | None = None,
    max_industry_deviation: float | None = None,
    asset_classes: pd.Series | dict[str, str] | None = None,
    max_asset_class_weights: dict[str, float] | None = None,
    benchmark_weights: pd.Series | dict[str, float] | None = None,
    return_covariance: pd.DataFrame | None = None,
    max_tracking_error: float | None = None,
    style_exposures: pd.Series | pd.DataFrame | None = None,
    benchmark_style_exposure: float | pd.Series | dict[str, float] | None = None,
    max_style_deviations: dict[str, float] | None = None,
    average_daily_values: pd.Series | dict[str, float] | None = None,
    portfolio_value: float | None = None,
    max_volume_participation: float | None = None,
    prices: pd.Series | dict[str, float] | None = None,
    lot_size: int | None = None,
    risk_ceiling: pd.Series | dict[str, float] | None = None,
) -> dict[str, Any]:
    """Validate the final, discrete account target against frozen hard limits.

    The function never repairs or relaxes a target.  Missing evidence for an
    explicitly configured constraint is an error; measured violations are
    returned so the caller can fail closed while retaining an audit artifact.
    Covariance input is daily and tracking error is annualized with 252 days,
    matching the optimizer contract.
    """

    if not 0 < max_position_weight <= 1:
        raise ValueError("max_position_weight must be in (0, 1]")
    if not 0 < max_daily_turnover <= 1:
        raise ValueError("max_daily_turnover must be in (0, 1]")
    if not 0 <= min_cash_weight < 1:
        raise ValueError("min_cash_weight must be in [0, 1)")

    target = _finite_weights(target_weights, label="target_weights")
    previous = _finite_weights(previous_weights, label="previous_weights")
    instruments = target.index.union(previous.index)
    target = target.reindex(instruments, fill_value=0.0)
    previous = previous.reindex(instruments, fill_value=0.0)
    checks: list[dict[str, Any]] = []

    minimum_weight = float(target.min()) if len(target) else 0.0
    _record(
        checks,
        name="long_only",
        observed=minimum_weight,
        limit=0.0,
        relation=">=",
        passed=minimum_weight >= -_TOLERANCE,
    )
    total_weight = float(target.sum())
    cash_weight = 1.0 - total_weight
    _record(
        checks,
        name="cash_weight",
        observed=cash_weight,
        limit=min_cash_weight,
        relation=">=",
        passed=cash_weight + _TOLERANCE >= min_cash_weight,
    )
    largest_position = float(target.max()) if len(target) else 0.0
    _record(
        checks,
        name="max_position_weight",
        observed=largest_position,
        limit=max_position_weight,
        relation="<=",
        passed=largest_position <= max_position_weight + _TOLERANCE,
    )
    previous_cash = 1.0 - float(previous.sum())
    turnover = 0.5 * (
        float((target - previous).abs().sum()) + abs(cash_weight - previous_cash)
    )
    _record(
        checks,
        name="daily_turnover",
        observed=turnover,
        limit=max_daily_turnover,
        relation="<=",
        passed=turnover <= max_daily_turnover + _TOLERANCE,
    )

    industry = _categorical_mapping(
        industries,
        index=instruments,
        label="industry memberships",
        required=max_industry_weight is not None
        or benchmark_industry_weights is not None
        or max_industry_deviation is not None,
    )
    if industry is not None:
        industry_weights = target.groupby(industry).sum()
        if max_industry_weight is not None:
            if not 0 < max_industry_weight <= 1:
                raise ValueError("max_industry_weight must be in (0, 1]")
            for name, weight in industry_weights.items():
                _record(
                    checks,
                    name="industry_weight",
                    scope=str(name),
                    observed=float(weight),
                    limit=max_industry_weight,
                    relation="<=",
                    passed=float(weight) <= max_industry_weight + _TOLERANCE,
                )
        if benchmark_industry_weights is not None or max_industry_deviation is not None:
            if benchmark_industry_weights is None or max_industry_deviation is None:
                raise ValueError(
                    "benchmark industry weights and deviation limit must be supplied together"
                )
            benchmark_industry = _finite_weights(
                benchmark_industry_weights,
                label="benchmark_industry_weights",
            )
            names = industry_weights.index.union(benchmark_industry.index)
            for name in names:
                deviation = abs(
                    float(industry_weights.get(name, 0.0))
                    - float(benchmark_industry.get(name, 0.0))
                )
                _record(
                    checks,
                    name="industry_deviation",
                    scope=str(name),
                    observed=deviation,
                    limit=max_industry_deviation,
                    relation="<=",
                    passed=deviation <= max_industry_deviation + _TOLERANCE,
                )

    asset_class = _categorical_mapping(
        asset_classes,
        index=instruments,
        label="asset class memberships",
        required=bool(max_asset_class_weights),
    )
    if max_asset_class_weights:
        assert asset_class is not None
        class_weights = target.groupby(asset_class).sum()
        for name, limit in sorted(max_asset_class_weights.items()):
            limit = float(limit)
            if not 0 <= limit <= 1:
                raise ValueError("asset class limits must be in [0, 1]")
            observed = float(class_weights.get(str(name), 0.0))
            _record(
                checks,
                name="asset_class_weight",
                scope=str(name),
                observed=observed,
                limit=limit,
                relation="<=",
                passed=observed <= limit + _TOLERANCE,
            )

    tracking_inputs = (
        benchmark_weights is not None,
        return_covariance is not None,
        max_tracking_error is not None,
    )
    if any(tracking_inputs):
        if not all(tracking_inputs):
            raise ValueError(
                "benchmark weights, covariance and tracking-error limit must be supplied together"
            )
        benchmark = _finite_weights(
            benchmark_weights,  # type: ignore[arg-type]
            label="benchmark_weights",
        )
        risk_universe = instruments.union(benchmark.index)
        covariance = return_covariance.copy()  # type: ignore[union-attr]
        covariance.index = covariance.index.astype(str)
        covariance.columns = covariance.columns.astype(str)
        covariance = covariance.reindex(index=risk_universe, columns=risk_universe)
        if covariance.isna().any().any():
            raise ValueError("return covariance must cover target and benchmark instruments")
        annual_covariance = validate_covariance(
            covariance.to_numpy(dtype=float)
        ) * 252.0
        active = (
            target.reindex(risk_universe, fill_value=0.0)
            - benchmark.reindex(risk_universe, fill_value=0.0)
        ).to_numpy(dtype=float)
        tracking_error = sqrt(max(0.0, float(active @ annual_covariance @ active)))
        _record(
            checks,
            name="tracking_error",
            observed=tracking_error,
            limit=float(max_tracking_error),
            relation="<=",
            passed=tracking_error <= float(max_tracking_error) + _TOLERANCE,
        )

    if style_exposures is not None or benchmark_style_exposure is not None:
        if (
            style_exposures is None
            or benchmark_style_exposure is None
            or not max_style_deviations
        ):
            raise ValueError(
                "style exposures, benchmark exposure and limits must be supplied together"
            )
        styles = (
            style_exposures.to_frame("style")
            if isinstance(style_exposures, pd.Series)
            else style_exposures.copy()
        )
        styles.index = styles.index.astype(str)
        styles = styles.reindex(instruments)
        if styles.isna().any().any():
            raise ValueError("style exposures must cover every target or existing instrument")
        if isinstance(benchmark_style_exposure, pd.Series):
            benchmark_styles = benchmark_style_exposure.astype(float)
        elif isinstance(benchmark_style_exposure, dict):
            benchmark_styles = pd.Series(benchmark_style_exposure, dtype=float)
        else:
            benchmark_styles = pd.Series(
                {str(styles.columns[0]): float(benchmark_style_exposure)}
            )
        for column in styles.columns:
            name = str(column)
            if name not in max_style_deviations or name not in benchmark_styles:
                raise ValueError(f"style constraint {name} has incomplete frozen evidence")
            observed = abs(
                float(target.dot(pd.to_numeric(styles[column], errors="raise")))
                - float(benchmark_styles[name])
            )
            limit = float(max_style_deviations[name])
            _record(
                checks,
                name="style_deviation",
                scope=name,
                observed=observed,
                limit=limit,
                relation="<=",
                passed=observed <= limit + _TOLERANCE,
            )

    capacity_inputs = (
        average_daily_values is not None,
        portfolio_value is not None,
        max_volume_participation is not None,
    )
    if any(capacity_inputs):
        if not all(capacity_inputs):
            raise ValueError(
                "daily values, portfolio value and participation limit must be supplied together"
            )
        if float(portfolio_value) <= 0 or not 0 < float(max_volume_participation) <= 1:
            raise ValueError("capacity contract contains invalid limits")
        daily_values = _finite_weights(
            average_daily_values,  # type: ignore[arg-type]
            label="average_daily_values",
        ).reindex(instruments)
        if daily_values.isna().any() or (daily_values < 0).any():
            raise ValueError("average daily values must cover every instrument")
        trade_values = (target - previous).abs() * float(portfolio_value)
        capacities = daily_values * float(max_volume_participation)
        for instrument in instruments:
            _record(
                checks,
                name="capacity_trade_value",
                scope=str(instrument),
                observed=float(trade_values[instrument]),
                limit=float(capacities[instrument]),
                relation="<=",
                passed=float(trade_values[instrument])
                <= float(capacities[instrument]) + 1e-4,
            )

    lot_inputs = (prices is not None, portfolio_value is not None, lot_size is not None)
    if any(lot_inputs):
        if not all(lot_inputs):
            raise ValueError("prices, portfolio value and lot size must be supplied together")
        if int(lot_size) < 1:
            raise ValueError("lot_size must be positive")
        price_values = _finite_weights(
            prices,  # type: ignore[arg-type]
            label="prices",
        ).reindex(instruments)
        if price_values.isna().any() or (price_values <= 0).any():
            raise ValueError("prices must cover every instrument")
        quantities = target * float(portfolio_value) / price_values
        for instrument, quantity in quantities.items():
            lots = float(quantity) / int(lot_size)
            distance = abs(lots - round(lots))
            _record(
                checks,
                name="round_lot",
                scope=str(instrument),
                observed=distance,
                limit=1e-7,
                relation="<=",
                passed=distance <= 1e-7,
            )

    if risk_ceiling is not None:
        ceiling = _finite_weights(risk_ceiling, label="risk_ceiling").reindex(
            instruments, fill_value=0.0
        )
        for instrument in instruments:
            excess = float(target[instrument] - ceiling[instrument])
            _record(
                checks,
                name="risk_ceiling_excess",
                scope=str(instrument),
                observed=excess,
                limit=0.0,
                relation="<=",
                passed=excess <= _TOLERANCE,
            )

    violations = [item for item in checks if not item["passed"]]
    return {
        "model_version": DISCRETE_CONSTRAINT_MODEL_VERSION,
        "status": "passed" if not violations else "failed",
        "cash_weight": cash_weight,
        "turnover": turnover,
        "checks": checks,
        "violations": violations,
    }
