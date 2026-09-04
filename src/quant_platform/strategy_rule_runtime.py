from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .strategy_rule_compiler import validate_strategy_rule_binding

_CONSTRAINED_PORTFOLIO_CONSTRUCTIONS = frozenset(
    {"benchmark_relative_qp", "industry_neutral_qp"}
)
GOVERNED_STYLE_COLUMNS = ("size", "value", "growth", "volatility")
MAX_STYLE_CROSS_SECTION_MISSING_RATE = 0.05
STYLE_EXPOSURE_CONTRACT_VERSION = "standardized-neutral-imputation-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_governed_style_exposures(
    provider_uri: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the builder-standardized style panel used by every runtime.

    Recommendation refreshes must not reconstruct these values from raw Qlib
    expressions: doing so would bypass the builder's PIT descriptor assembly
    and cross-sectional standardization.  The returned frame deliberately
    preserves sparse missing values; the per-date 5% gate and neutral
    imputation are applied by :func:`latest_governed_style_cross_section`.
    """

    path = Path(provider_uri) / "metadata" / "style_exposures.parquet"
    if not path.is_file():
        raise ValueError(
            "governed Qlib runtime requires standardized point-in-time style metadata"
        )
    required = ["instrument", "datetime", *GOVERNED_STYLE_COLUMNS]
    try:
        frame = pd.read_parquet(path, columns=required)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "standardized style metadata is missing governed exposure columns"
        ) from exc
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.tz_localize(
        None
    )
    if frame[["instrument", "datetime"]].isna().any().any():
        raise ValueError("standardized style metadata contains invalid identity fields")
    frame["instrument"] = frame["instrument"].astype(str)
    if frame.duplicated(["datetime", "instrument"]).any():
        raise ValueError("standardized style metadata contains duplicate instrument dates")
    frame[list(GOVERNED_STYLE_COLUMNS)] = (
        frame[list(GOVERNED_STYLE_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    missing_counts = {
        column: int(frame[column].isna().sum()) for column in GOVERNED_STYLE_COLUMNS
    }
    return frame, {
        "contract_version": STYLE_EXPOSURE_CONTRACT_VERSION,
        "source": "qlib_builder_standardized_style_exposures",
        "path": str(path),
        "sha256": _sha256_file(path),
        "columns": list(GOVERNED_STYLE_COLUMNS),
        "rows": int(len(frame)),
        "missing_counts": missing_counts,
        "max_cross_section_missing_rate": MAX_STYLE_CROSS_SECTION_MISSING_RATE,
        "missing_imputation": "zero_standardized_neutral_exposure",
    }


def latest_governed_style_cross_section(
    frame: pd.DataFrame,
    when: pd.Timestamp,
    *,
    preserve_missing: bool = False,
    required_instruments: Any = None,
) -> pd.DataFrame:
    """Resolve one PIT standardized style cross-section identically everywhere."""

    values = frame.copy()
    values["datetime"] = pd.to_datetime(values["datetime"], errors="coerce").dt.tz_localize(
        None
    )
    timestamp = pd.Timestamp(when).tz_localize(None)
    values = values[values["datetime"] <= timestamp]
    if values.empty:
        raise ValueError(f"point-in-time styles have no values at {timestamp.date()}")
    values = values[values["datetime"] == values["datetime"].max()]
    result = values.set_index(values["instrument"].astype(str)).drop(
        columns=["datetime", "instrument"]
    )
    if result.index.has_duplicates:
        raise ValueError("point-in-time style exposures are duplicated")
    result = result.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    missing_rates = result.isna().mean()
    if required_instruments is not None:
        # The systemic-missing gate exists to guarantee the exposures a
        # strategy actually consumes.  Recent IPOs sit inside their 120-day
        # volatility warmup and are never tradable candidates, so a full-market
        # missing rate would veto windows the strategy can safely trade.
        scoped = result.reindex(
            pd.Index([str(item) for item in required_instruments], dtype=str)
        )
        if len(scoped):
            missing_rates = scoped.isna().mean()
    systemic = missing_rates[missing_rates > MAX_STYLE_CROSS_SECTION_MISSING_RATE]
    if not systemic.empty:
        details = ", ".join(
            f"{column}={rate:.2%}" for column, rate in systemic.items()
        )
        raise ValueError(
            "point-in-time standardized style exposure missing rate exceeds "
            f"{MAX_STYLE_CROSS_SECTION_MISSING_RATE:.0%}: {details}"
        )
    if preserve_missing:
        return result.astype(float)
    # Builder outputs are standardized around zero, making zero the only
    # governed neutral imputation after the systemic-missing gate has passed.
    result = result.fillna(0.0)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("point-in-time style exposures are not finite")
    return result.astype(float)


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


def load_market_trend_close_history(
    data_api: Any,
    *,
    config: Mapping[str, Any],
    start_time: str,
    end_time: str,
) -> pd.DataFrame | None:
    """Load the exact index series bound by the compiled market-trend rule."""

    lookback = int(config.get("market_trend_lookback_sessions") or 0)
    if not lookback:
        return None
    benchmark = str(config.get("market_trend_benchmark") or "").strip()
    if not benchmark:
        raise ValueError("market-trend rule requires a bound benchmark instrument")
    values = data_api.features(
        [benchmark],
        ["$close"],
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    if values.empty or "$close" not in values.columns:
        raise ValueError("Qlib has no bound market-trend benchmark close history")
    try:
        closes = values["$close"].unstack("instrument").sort_index()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Qlib market-trend benchmark history is malformed") from exc
    closes.columns = closes.columns.astype(str)
    if closes.columns.has_duplicates or list(closes.columns) != [benchmark]:
        raise ValueError("Qlib market-trend history differs from the bound benchmark")
    return closes


def build_portfolio_policy_runtime_metadata(
    config: Mapping[str, Any],
    *,
    instruments: pd.Index,
    industries: pd.Series,
    benchmark_weights: pd.Series | None,
    style_exposures: pd.DataFrame | None,
    return_covariance: pd.DataFrame | None,
) -> dict[str, Any]:
    """Scope target metadata to what the selected portfolio policy consumes.

    Every policy needs industries for the actual signal and current holdings.
    Benchmark industries, benchmark styles and covariance are optimizer inputs,
    so a transparent Top-K policy must not fail on gaps in those unused fields.
    """

    requested = pd.Index(instruments.astype(str)).drop_duplicates()
    industry_values = industries.copy()
    industry_values.index = industry_values.index.astype(str)
    if industry_values.index.has_duplicates:
        raise ValueError("point-in-time industries are duplicated")
    requested_industries = industry_values.reindex(requested)
    missing_industries = requested_industries.isna() | (
        requested_industries.astype("string").str.strip() == ""
    )
    if missing_industries.any():
        raise ValueError("signal or current holdings are missing point-in-time industries")

    result: dict[str, Any] = {"industries": requested_industries.astype(str)}
    if str(config.get("portfolio_construction") or "") not in (
        _CONSTRAINED_PORTFOLIO_CONSTRUCTIONS
    ):
        return result

    if benchmark_weights is None:
        raise ValueError("constrained Qlib policy requires point-in-time benchmark weights")
    benchmark = pd.to_numeric(benchmark_weights, errors="coerce")
    benchmark.index = benchmark.index.astype(str)
    if (
        benchmark.index.has_duplicates
        or benchmark.isna().any()
        or not np.isfinite(benchmark.to_numpy(dtype=float)).all()
        or (benchmark < 0).any()
    ):
        raise ValueError("benchmark weights are duplicated or incomplete")

    benchmark_industries = industry_values.reindex(benchmark.index)
    missing_benchmark_industries = benchmark_industries.isna() | (
        benchmark_industries.astype("string").str.strip() == ""
    )
    if missing_benchmark_industries.any():
        raise ValueError("benchmark constituents are missing point-in-time industries")
    if style_exposures is None:
        raise ValueError("constrained Qlib policy requires point-in-time style exposures")
    styles = style_exposures.copy()
    styles.index = styles.index.astype(str)
    if styles.index.has_duplicates or styles.columns.has_duplicates:
        raise ValueError("point-in-time style exposures are duplicated")
    styles = styles.apply(pd.to_numeric, errors="coerce")
    style_instruments = requested.union(benchmark.index)
    required_styles = styles.reindex(style_instruments)
    if required_styles.isna().any().any() or not np.isfinite(
        required_styles.to_numpy(dtype=float)
    ).all():
        raise ValueError("constrained Qlib policy has incomplete point-in-time styles")
    if return_covariance is None:
        raise ValueError("constrained Qlib policy requires point-in-time return covariance")

    result.update(
        {
            "benchmark_weights": benchmark.astype(float),
            "benchmark_industry_weights": benchmark.groupby(
                benchmark_industries.astype(str)
            ).sum(),
            "style_exposures": styles,
            "benchmark_style_exposure": required_styles.reindex(benchmark.index)
            .mul(benchmark, axis=0)
            .sum(),
            "return_covariance": return_covariance,
        }
    )
    return result


def build_strategy_rule_runtime_metadata(
    config: Mapping[str, Any],
    *,
    instruments: pd.Index,
    close_history: pd.DataFrame,
    benchmark_weights: pd.Series | None = None,
    benchmark_close_history: pd.DataFrame | pd.Series | None = None,
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
        if len(complete) < 6:
            result["five_day_returns"] = pd.Series(dtype=float)
        else:
            finite_columns = complete.columns[
                complete.notna().all(axis=0)
                & np.isfinite(complete.to_numpy(dtype=float)).all(axis=0)
                & (complete > 0).all(axis=0)
            ]
            finite_history = complete.loc[:, finite_columns]
            returns = finite_history.iloc[-1] / finite_history.iloc[0] - 1.0
            result["five_day_returns"] = returns.astype(float)

    trend_lookback = int(config.get("trend_break_lookback_sessions") or 0)
    if trend_lookback:
        complete = closes.tail(trend_lookback)
        if len(complete) < trend_lookback:
            result["trend_intact"] = pd.Series(dtype=bool)
        else:
            finite_columns = complete.columns[
                complete.notna().all(axis=0)
                & np.isfinite(complete.to_numpy(dtype=float)).all(axis=0)
                & (complete > 0).all(axis=0)
            ]
            finite_history = complete.loc[:, finite_columns]
            moving_average = finite_history.mean(axis=0)
            result["trend_intact"] = (
                finite_history.iloc[-1] >= moving_average
            ).astype(bool)

    market_lookback = int(config.get("market_trend_lookback_sessions") or 0)
    if market_lookback:
        benchmark = str(config.get("market_trend_benchmark") or "").strip()
        if not benchmark:
            raise ValueError("market-trend rule requires a bound benchmark instrument")
        if benchmark_close_history is None:
            raise ValueError("market-trend rule requires PIT benchmark close history")
        if isinstance(benchmark_close_history, pd.Series):
            if str(benchmark_close_history.name or "") != benchmark:
                raise ValueError("market-trend close history differs from the bound benchmark")
            benchmark_closes = benchmark_close_history.copy()
        else:
            benchmark_frame = benchmark_close_history.copy()
            benchmark_frame.columns = benchmark_frame.columns.astype(str)
            if (
                benchmark_frame.columns.has_duplicates
                or list(benchmark_frame.columns) != [benchmark]
            ):
                raise ValueError("market-trend close history differs from the bound benchmark")
            benchmark_closes = benchmark_frame[benchmark]
        benchmark_closes.index = pd.to_datetime(benchmark_closes.index).tz_localize(None)
        if benchmark_closes.index.has_duplicates:
            raise ValueError("market-trend benchmark close history is duplicated")
        benchmark_closes = pd.to_numeric(
            benchmark_closes.sort_index(), errors="coerce"
        )
        complete = benchmark_closes.tail(market_lookback)
        if (
            len(complete) < market_lookback
            or complete.isna().any()
            or not np.isfinite(complete.to_numpy(dtype=float)).all()
            or (complete <= 0).any()
        ):
            raise ValueError("market-trend rule has incomplete benchmark close history")
        trend_level = float(complete.iloc[-1] / complete.mean())
        if not np.isfinite(trend_level):
            raise ValueError("market-trend rule produced a non-finite regime value")
        result["market_regime_allows_entries"] = trend_level >= 1.0

    if (
        config.get("valuation_regime_max_percentile") is not None
        or config.get("valuation_reduce_percentile") is not None
    ):
        if value_exposures is None:
            values = pd.Series(np.nan, index=requested, dtype=float)
        else:
            values = pd.to_numeric(value_exposures, errors="coerce")
            values.index = values.index.astype(str)
            values = values.reindex(requested).replace([np.inf, -np.inf], np.nan)
        # Higher 1/PB is cheaper; rank the negative exposure so a larger
        # percentile means more expensive and can be capped directly. Missing
        # PIT valuation remains missing so the policy can reject only that new
        # entry without manufacturing a neutral or cheap value.
        result["valuation_percentiles"] = (-values).rank(
            method="average", pct=True
        )

    return result
