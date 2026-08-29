from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .strategy_rule_compiler import validate_strategy_rule_binding


def apply_strategy_rule_alpha_weights(
    normalized_factors: pd.DataFrame,
    baseline_scores: pd.Series,
    config: Mapping[str, Any],
) -> pd.Series:
    """Apply compiled alpha-rank weights to the frozen Qlib factor grid."""

    policy = validate_strategy_rule_binding(config)
    if policy is None:
        return baseline_scores.sort_index().rename("score")
    weights = policy.get("alpha_factor_weights")
    if not isinstance(weights, dict) or set(weights) != set(normalized_factors.columns):
        raise ValueError("strategy alpha weights do not match the governed factor grid")
    ordered = normalized_factors.loc[:, list(weights)].apply(
        pd.to_numeric, errors="coerce"
    )
    result = pd.Series(0.0, index=ordered.index, dtype=float)
    for factor_id, weight in weights.items():
        result = result.add(ordered[factor_id] * float(weight), fill_value=np.nan)
    return result.sort_index().rename("score")


def required_rule_history_sessions(config: Mapping[str, Any]) -> int:
    """Return the daily history required to evaluate compiled strategy rules."""

    values = [
        60,
        int(config.get("liquidity_lookback_days") or 0),
        int(config.get("market_trend_lookback_sessions") or 0),
        int(config.get("trend_break_lookback_sessions") or 0),
        6 if config.get("extension_guard_max_return_5d") is not None else 0,
    ]
    return max(values)


def build_strategy_rule_runtime_metadata(
    config: Mapping[str, Any],
    *,
    instruments: pd.Index,
    close_history: pd.DataFrame,
    benchmark_weights: pd.Series | None = None,
    value_exposures: pd.Series | None = None,
) -> dict[str, Any]:
    """Build PIT-only inputs consumed by :class:`PortfolioPolicy` rules.

    ``close_history`` must already be clipped at the signal timestamp.  The
    helper never forward-fills missing prices because doing so could turn a
    suspension or incomplete snapshot into a false entry or exit signal.
    """

    requested = instruments.astype(str)
    closes = close_history.copy()
    closes.columns = closes.columns.astype(str)
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    closes = closes.sort_index().reindex(columns=requested)
    if closes.empty or closes.index.has_duplicates:
        raise ValueError("strategy-rule close history is missing or duplicated")
    result: dict[str, Any] = {}

    if config.get("extension_guard_max_return_5d") is not None:
        complete = closes.tail(6)
        if len(complete) < 6 or complete.isna().any().any():
            raise ValueError("extension guard requires six complete trading-day closes")
        returns = complete.iloc[-1] / complete.iloc[0] - 1.0
        if not np.isfinite(returns.to_numpy(dtype=float)).all():
            raise ValueError("extension guard produced non-finite returns")
        result["five_day_returns"] = returns.astype(float)

    trend_lookback = int(config.get("trend_break_lookback_sessions") or 0)
    if trend_lookback:
        complete = closes.tail(trend_lookback)
        if len(complete) < trend_lookback or complete.isna().any().any():
            raise ValueError("trend-break rule has incomplete close history")
        moving_average = complete.mean(axis=0)
        result["trend_intact"] = (complete.iloc[-1] >= moving_average).astype(bool)

    market_lookback = int(config.get("market_trend_lookback_sessions") or 0)
    if market_lookback:
        if benchmark_weights is None:
            raise ValueError("market-trend rule requires PIT benchmark weights")
        weights = pd.to_numeric(benchmark_weights, errors="coerce")
        weights.index = weights.index.astype(str)
        weights = weights[weights > 0]
        benchmark_closes = close_history.copy()
        benchmark_closes.columns = benchmark_closes.columns.astype(str)
        benchmark_closes.index = pd.to_datetime(benchmark_closes.index).tz_localize(None)
        benchmark_closes = benchmark_closes.sort_index().reindex(columns=weights.index)
        complete = benchmark_closes.tail(market_lookback)
        if (
            weights.empty
            or len(complete) < market_lookback
            or complete.isna().any().any()
        ):
            raise ValueError("market-trend rule has incomplete benchmark close history")
        normalized_weights = weights / float(weights.sum())
        relative_to_average = complete.iloc[-1] / complete.mean(axis=0)
        breadth_level = float(relative_to_average.mul(normalized_weights).sum())
        if not np.isfinite(breadth_level):
            raise ValueError("market-trend rule produced a non-finite regime value")
        result["market_regime_allows_entries"] = breadth_level >= 1.0

    if config.get("valuation_regime_max_percentile") is not None:
        if value_exposures is None:
            raise ValueError("valuation-regime rule requires PIT value exposures")
        values = pd.to_numeric(value_exposures, errors="coerce")
        values.index = values.index.astype(str)
        values = values.reindex(requested)
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("valuation-regime value exposures are incomplete")
        # Higher 1/PB is cheaper; rank the negative exposure so a larger
        # percentile means more expensive and can be capped directly.
        result["valuation_percentiles"] = (-values).rank(
            method="average", pct=True
        )

    return result
