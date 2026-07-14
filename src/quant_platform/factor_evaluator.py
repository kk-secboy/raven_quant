from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import pandas as pd


def normalize_series(values: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"{name} must contain exactly one value column")
        values = values.iloc[:, 0]
    if not isinstance(values.index, pd.MultiIndex) or values.index.nlevels != 2:
        raise ValueError(f"{name} must use a datetime/instrument MultiIndex")
    names = list(values.index.names)
    if set(names) == {"datetime", "instrument"}:
        values = values.reorder_levels(["datetime", "instrument"])
    else:
        values.index = values.index.set_names(["datetime", "instrument"])
    frame = values.rename(name).reset_index()
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.tz_localize(None)
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.dropna().set_index(["datetime", "instrument"])[name].sort_index()


def _daily_correlation(frame: pd.DataFrame, method: str) -> pd.Series:
    def correlation(group: pd.DataFrame) -> float:
        if len(group) < 5 or group["factor"].nunique() < 2 or group["label"].nunique() < 2:
            return float("nan")
        if method == "spearman":
            return float(group["factor"].rank().corr(group["label"].rank()))
        return float(group["factor"].corr(group["label"]))

    return frame.groupby(level="datetime", sort=True).apply(correlation).dropna()


def _long_short_returns(frame: pd.DataFrame, cost_rate: float) -> tuple[float, float, float]:
    previous: pd.Series | None = None
    turnovers: list[float] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    for _, group in frame.groupby(level="datetime", sort=True):
        if len(group) < 10:
            continue
        ranks = group["factor"].rank(method="average", pct=True)
        long_index = ranks[ranks >= 0.8].index.get_level_values("instrument")
        short_index = ranks[ranks <= 0.2].index.get_level_values("instrument")
        if not len(long_index) or not len(short_index):
            continue
        weights = pd.Series(0.0, index=group.index.get_level_values("instrument").unique())
        weights.loc[long_index] = 1.0 / len(long_index)
        weights.loc[short_index] = -1.0 / len(short_index)
        returns = group.droplevel("datetime")["label"].groupby(level="instrument").mean()
        gross = float(weights.reindex(returns.index, fill_value=0.0).dot(returns))
        if previous is None:
            turnover = 1.0
        else:
            union = previous.index.union(weights.index)
            turnover = float(
                0.5
                * (weights.reindex(union, fill_value=0.0) - previous.reindex(union, fill_value=0.0))
                .abs()
                .sum()
            )
        previous = weights
        turnovers.append(turnover)
        gross_returns.append(gross)
        net_returns.append(gross - turnover * cost_rate)
    if not net_returns:
        return float("nan"), float("nan"), float("nan")
    return (
        float(pd.Series(turnovers).mean()),
        float(pd.Series(gross_returns).mean() * 252),
        float(pd.Series(net_returns).mean() * 252),
    )


def evaluate_factor_values(
    factor_values: pd.Series | pd.DataFrame,
    forward_returns: pd.Series | pd.DataFrame,
    *,
    valid_start: date,
    valid_end: date,
    test_start: date,
    test_end: date,
    comparison_values: Iterable[pd.Series | pd.DataFrame] = (),
    cost_rate: float = 0.002,
) -> dict[str, float | int | str | None]:
    factor = normalize_series(factor_values, "factor")
    label = normalize_series(forward_returns, "label")
    joined = pd.concat([factor, label], axis=1, join="inner").dropna()
    dates = joined.index.get_level_values("datetime")
    valid = joined[(dates >= pd.Timestamp(valid_start)) & (dates <= pd.Timestamp(valid_end))]
    test = joined[(dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))]
    if valid.empty or test.empty:
        raise ValueError("factor and forward-return data must cover validation and test windows")
    valid_ic_series = _daily_correlation(valid, "pearson")
    if valid_ic_series.empty:
        raise ValueError("validation window has insufficient cross-sectional observations")
    valid_ic = float(valid_ic_series.mean())
    direction = -1.0 if valid_ic < 0 else 1.0
    test = test.copy()
    test["factor"] *= direction
    ic_daily = _daily_correlation(test, "pearson")
    rank_ic_daily = _daily_correlation(test, "spearman")
    if ic_daily.empty or rank_ic_daily.empty:
        raise ValueError("test window has insufficient cross-sectional observations")
    ic = float(ic_daily.mean())
    rank_ic = float(rank_ic_daily.mean())
    icir = float(ic / ic_daily.std(ddof=1)) if ic_daily.std(ddof=1) > 0 else None
    rank_std = rank_ic_daily.std(ddof=1)
    rank_icir = float(rank_ic / rank_std) if rank_std > 0 else None
    turnover, gross_return, cost_adjusted_return = _long_short_returns(test, cost_rate)

    correlations: list[float] = []
    directed_factor = factor * direction
    for comparison in comparison_values:
        other = normalize_series(comparison, "comparison")
        pair = pd.concat([directed_factor, other], axis=1, join="inner").dropna()
        pair_dates = pair.index.get_level_values("datetime")
        pair = pair[
            (pair_dates >= pd.Timestamp(test_start)) & (pair_dates <= pd.Timestamp(test_end))
        ]
        if pair.empty:
            continue
        daily = pair.groupby(level="datetime").apply(
            lambda group: group["factor"].rank().corr(group["comparison"].rank())
        )
        if daily.notna().any():
            correlations.append(float(daily.abs().mean()))

    return {
        "ic": ic,
        "icir": icir,
        "rank_ic": rank_ic,
        "rank_icir": rank_icir,
        "turnover": turnover,
        "max_correlation": max(correlations, default=0.0),
        "cost_adjusted_return": cost_adjusted_return,
        "gross_annualized_return": gross_return,
        "valid_ic": valid_ic,
        "direction": "inverted" if direction < 0 else "original",
        "observations": int(len(test)),
        "test_days": int(test.index.get_level_values("datetime").nunique()),
        "cost_rate": cost_rate,
    }
