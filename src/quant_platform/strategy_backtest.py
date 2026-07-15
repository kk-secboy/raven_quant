from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .factor_evaluator import normalize_series


def compose_factor_scores(
    factors: Sequence[tuple[pd.Series | pd.DataFrame, float, int]],
) -> pd.Series:
    """Compose immutable factor artifacts into one point-in-time Qlib signal."""

    if not factors:
        raise ValueError("at least one factor is required")
    normalized: list[pd.Series] = []
    for index, (values, weight, direction) in enumerate(factors):
        series = normalize_series(values, f"factor_{index}")

        def cross_section(group: pd.Series) -> pd.Series:
            lower, upper = group.quantile([0.01, 0.99])
            clipped = group.clip(lower, upper)
            standard_deviation = clipped.std(ddof=0)
            return (
                (clipped - clipped.mean()) / standard_deviation
                if standard_deviation > 0
                else clipped * 0.0
            )

        zscore = series.groupby(level="datetime", group_keys=False).apply(cross_section)
        normalized.append(zscore * float(weight) * int(direction))
    frame = pd.concat(normalized, axis=1, join="inner").dropna()
    if frame.empty:
        raise ValueError("factor value artifacts have no common observations")
    return frame.sum(axis=1).rename("score")


def build_governed_signal(
    scores: pd.Series | pd.DataFrame,
    *,
    topk: int,
    liquidity_amount: pd.Series | pd.DataFrame | None = None,
    industry_memberships: pd.DataFrame | None = None,
    benchmark_weights: pd.DataFrame | None = None,
    style_exposures: pd.DataFrame | None = None,
    max_industry_weight: float = 1.0,
    max_industry_deviation: float = 1.0,
    min_average_daily_amount: float = 0.0,
    liquidity_lookback_days: int = 20,
) -> pd.Series:
    """Apply eligibility gates before the shared PortfolioPolicy assigns weights."""

    if topk < 1 or liquidity_lookback_days < 2:
        raise ValueError("topk and liquidity lookback are invalid")
    score = normalize_series(scores, "score")
    liquidity = (
        normalize_series(liquidity_amount, "liquidity_amount")
        if liquidity_amount is not None
        else None
    )
    rolling_liquidity = (
        liquidity.unstack("instrument")
        .sort_index()
        .rolling(liquidity_lookback_days, min_periods=min(5, liquidity_lookback_days))
        .mean()
        if liquidity is not None
        else None
    )
    memberships = _normalize_memberships(industry_memberships)
    styles = _normalize_style_snapshots(style_exposures)
    benchmark = _normalize_snapshots(benchmark_weights, "weight")
    rows: list[pd.Series] = []
    for timestamp, daily in score.groupby(level="datetime", sort=True):
        ranking = daily.droplevel("datetime").dropna()
        daily_styles = _style_snapshot(styles, timestamp).reindex(ranking.index)
        design = pd.concat([ranking.rename("score"), daily_styles], axis=1).dropna()
        style_columns = [column for column in design.columns if column != "score"]
        if style_columns and len(design) >= max(5, min(topk, len(ranking))):
            matrix = design[style_columns].to_numpy(dtype=float)
            matrix = np.column_stack([np.ones(len(matrix)), matrix])
            coefficients, *_ = np.linalg.lstsq(
                matrix, design["score"].to_numpy(dtype=float), rcond=None
            )
            ranking.loc[design.index] = design["score"] - matrix @ coefficients
        if min_average_daily_amount > 0:
            daily_amount = (
                rolling_liquidity.loc[timestamp]
                if rolling_liquidity is not None and timestamp in rolling_liquidity.index
                else pd.Series(dtype=float)
            )
            ranking = ranking[
                daily_amount.reindex(ranking.index).fillna(0.0) >= min_average_daily_amount
            ]
        ranking = ranking.sort_values(ascending=False)
        if len(ranking) < topk:
            continue
        industries = _industries_at(memberships, timestamp)
        benchmark_day = _snapshot(benchmark, timestamp, "weight")
        benchmark_industry = (
            benchmark_day.groupby(industries.reindex(benchmark_day.index)).sum()
            if not benchmark_day.empty and not industries.empty
            else pd.Series(dtype=float)
        )
        selected: list[str] = []
        counts: dict[str, int] = {}
        for instrument in ranking.index.astype(str):
            industry = str(industries.get(instrument, "__unknown__"))
            allowed_weight = min(
                max_industry_weight,
                float(benchmark_industry.get(industry, 0.0)) + max_industry_deviation,
            )
            allowed_count = max(1, int(allowed_weight * topk + 1e-9))
            if memberships is not None and counts.get(industry, 0) >= allowed_count:
                continue
            selected.append(instrument)
            counts[industry] = counts.get(industry, 0) + 1
            if len(selected) == topk:
                break
        if len(selected) < topk:
            continue
        governed = ranking.reindex(selected)
        governed.index = pd.MultiIndex.from_product(
            [[timestamp], governed.index], names=["datetime", "instrument"]
        )
        rows.append(governed)
    if not rows:
        raise ValueError("governed signal has no eligible trading dates")
    return pd.concat(rows).sort_index().rename("score")


def _normalize_memberships(values: pd.DataFrame | None) -> pd.DataFrame | None:
    if values is None:
        return None
    required = {"instrument", "industry", "in_date", "out_date"}
    if not required.issubset(values.columns):
        raise ValueError("industry memberships are incomplete")
    result = values.loc[:, sorted(required)].copy()
    result["instrument"] = result["instrument"].astype(str)
    result["industry"] = result["industry"].astype(str)
    result["in_date"] = pd.to_datetime(result["in_date"], errors="coerce")
    result["out_date"] = pd.to_datetime(result["out_date"], errors="coerce")
    return result


def _normalize_snapshots(values: pd.DataFrame | None, column: str) -> pd.DataFrame | None:
    if values is None:
        return None
    required = {"datetime", "instrument", column}
    if not required.issubset(values.columns):
        raise ValueError(f"point-in-time {column} metadata is incomplete")
    result = values.loc[:, ["datetime", "instrument", column]].copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    result["instrument"] = result["instrument"].astype(str)
    result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna()


def _normalize_style_snapshots(values: pd.DataFrame | None) -> pd.DataFrame | None:
    if values is None:
        return None
    required = {"datetime", "instrument"}
    if not required.issubset(values.columns):
        raise ValueError("point-in-time style metadata is incomplete")
    style_columns = [column for column in values.columns if column not in required]
    if not style_columns:
        raise ValueError("point-in-time style metadata has no exposure columns")
    result = values.loc[:, ["datetime", "instrument", *style_columns]].copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    result["instrument"] = result["instrument"].astype(str)
    result[style_columns] = result[style_columns].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return result.dropna()


def _snapshot(values: pd.DataFrame | None, timestamp: pd.Timestamp, column: str) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    eligible = values[values["datetime"] <= timestamp]
    if eligible.empty:
        return pd.Series(dtype=float)
    current = eligible[eligible["datetime"] == eligible["datetime"].max()]
    return current.drop_duplicates("instrument", keep="last").set_index("instrument")[column]


def _style_snapshot(values: pd.DataFrame | None, timestamp: pd.Timestamp) -> pd.DataFrame:
    if values is None:
        return pd.DataFrame()
    eligible = values[values["datetime"] <= timestamp]
    if eligible.empty:
        return pd.DataFrame()
    current = eligible[eligible["datetime"] == eligible["datetime"].max()]
    return current.drop_duplicates("instrument", keep="last").set_index("instrument").drop(
        columns=["datetime"], errors="ignore"
    )


def _industries_at(values: pd.DataFrame | None, timestamp: pd.Timestamp) -> pd.Series:
    if values is None:
        return pd.Series(dtype=str)
    active = values[
        (values["in_date"] <= timestamp)
        & (values["out_date"].isna() | (values["out_date"] >= timestamp))
    ]
    return (
        active.sort_values("in_date")
        .drop_duplicates("instrument", keep="last")
        .set_index("instrument")["industry"]
    )
