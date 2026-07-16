from __future__ import annotations

import math

import pandas as pd

MINUTE_FACTOR_EXPRESSIONS: dict[str, str] = {
    "momentum_5m": "$close/Ref($close,5)-1",
    "momentum_15m": "$close/Ref($close,15)-1",
    "oversold_60m": "-($close/Mean($close,60)-1)",
    "lower_band_120m": "-(($close-Mean($close,120))/(Std($close,120)+1e-12))",
    "vwap_deviation": "$close/$vwap-1",
    "volume_surprise_30m": "$volume/Mean($volume,30)-1",
    "range_pressure": "($close-$low)/($high-$low+1e-12)-0.5",
    "realized_volatility_30m": "Std($close/Ref($close,1)-1,30)",
}


def minute_bar_minutes(frequency: str) -> int:
    normalized = str(frequency).lower()
    if normalized not in {"1min", "5min", "15min", "30min", "60min"}:
        raise ValueError("unsupported minute research frequency")
    return int(normalized.removesuffix("min"))


def minute_factor_expressions(frequency: str) -> dict[str, str]:
    """Return duration-stable Qlib expressions for the selected minute bar."""

    bar_minutes = minute_bar_minutes(frequency)

    def bars(duration: int) -> int:
        return max(1, duration // bar_minutes)

    return {
        "momentum_5m": f"$close/Ref($close,{bars(5)})-1",
        "momentum_15m": f"$close/Ref($close,{bars(15)})-1",
        "oversold_60m": f"-($close/Mean($close,{bars(60)})-1)",
        "lower_band_120m": (
            f"-(($close-Mean($close,{bars(120)}))/"
            f"(Std($close,{bars(120)})+1e-12))"
        ),
        "vwap_deviation": "$close/$vwap-1",
        "volume_surprise_30m": (
            f"$volume/Mean($volume,{bars(30)})-1"
        ),
        "range_pressure": "($close-$low)/($high-$low+1e-12)-0.5",
        "realized_volatility_30m": (
            f"Std($close/Ref($close,1)-1,{bars(30)})"
        ),
    }


def normalize_minute_series(values: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"{name} must contain exactly one value column")
        values = values.iloc[:, 0]
    if not isinstance(values.index, pd.MultiIndex) or values.index.nlevels != 2:
        raise ValueError(f"{name} must use a datetime/instrument MultiIndex")
    values = values.copy()
    if set(values.index.names) == {"datetime", "instrument"}:
        values = values.reorder_levels(["datetime", "instrument"])
    else:
        values.index = values.index.set_names(["datetime", "instrument"])
    frame = values.rename(name).reset_index()
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.tz_localize(None)
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.dropna().set_index(["datetime", "instrument"])[name].sort_index()


def evaluate_minute_factor(
    factor_values: pd.Series | pd.DataFrame,
    forward_returns: pd.Series | pd.DataFrame,
    *,
    horizon_minutes: int,
    cost_rate: float,
    top_fraction: float = 0.20,
    bar_minutes: int = 1,
) -> dict[str, float | int | None]:
    if horizon_minutes < 1:
        raise ValueError("horizon_minutes must be positive")
    if bar_minutes < 1 or horizon_minutes % bar_minutes:
        raise ValueError("horizon_minutes must be an integer multiple of the bar")
    if not 0 < top_fraction <= 0.5:
        raise ValueError("top_fraction must be between 0 and 0.5")
    factor = normalize_minute_series(factor_values, "factor")
    label = normalize_minute_series(forward_returns, "label")
    joined = pd.concat([factor, label], axis=1, join="inner").dropna()
    timestamps = joined.index.get_level_values("datetime").unique().sort_values()
    horizon_bars = horizon_minutes // bar_minutes
    selected = timestamps[::horizon_bars]
    joined = joined[joined.index.get_level_values("datetime").isin(selected)]
    if joined.empty:
        raise ValueError("factor and forward returns have no aligned minute observations")

    pearson: list[float] = []
    spearman: list[float] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    turnovers: list[float] = []
    previous: pd.Series | None = None
    for _, group in joined.groupby(level="datetime", sort=True):
        values = group.droplevel("datetime")
        if len(values) < 10 or values["factor"].nunique() < 2:
            continue
        raw_ic = values["factor"].corr(values["label"])
        rank_ic = values["factor"].rank().corr(values["label"].rank())
        if pd.notna(raw_ic):
            pearson.append(float(raw_ic))
        if pd.notna(rank_ic):
            spearman.append(float(rank_ic))
        ranks = values["factor"].rank(method="average", pct=True)
        long_names = ranks[ranks >= 1 - top_fraction].index
        short_names = ranks[ranks <= top_fraction].index
        if not len(long_names) or not len(short_names):
            continue
        weights = pd.Series(0.0, index=values.index.unique())
        weights.loc[long_names] = 1.0 / len(long_names)
        weights.loc[short_names] = -1.0 / len(short_names)
        gross = float(weights.dot(values["label"].reindex(weights.index)))
        turnover = 1.0
        if previous is not None:
            union = previous.index.union(weights.index)
            turnover = float(
                0.5
                * (weights.reindex(union, fill_value=0.0) - previous.reindex(union, fill_value=0.0))
                .abs()
                .sum()
            )
        previous = weights
        gross_returns.append(gross)
        turnovers.append(turnover)
        net_returns.append(gross - turnover * cost_rate)

    if not spearman or not net_returns:
        raise ValueError("minute factor has insufficient cross-sectional observations")
    rank_series = pd.Series(spearman)
    periods_per_year = 252 * 240 / horizon_minutes
    return {
        "ic": float(pd.Series(pearson).mean()) if pearson else None,
        "rank_ic": float(rank_series.mean()),
        "rank_icir": (
            float(rank_series.mean() / rank_series.std(ddof=1))
            if len(rank_series) > 1 and rank_series.std(ddof=1) > 0
            else None
        ),
        "mean_gross_return": float(pd.Series(gross_returns).mean()),
        "mean_net_return": float(pd.Series(net_returns).mean()),
        "annualized_net_return": float(pd.Series(net_returns).mean() * periods_per_year),
        "annualized_net_sharpe": (
            float(
                pd.Series(net_returns).mean()
                / pd.Series(net_returns).std(ddof=1)
                * math.sqrt(periods_per_year)
            )
            if len(net_returns) > 1 and pd.Series(net_returns).std(ddof=1) > 0
            else None
        ),
        "average_turnover": float(pd.Series(turnovers).mean()),
        "observations": int(len(joined)),
        "rebalance_timestamps": len(net_returns),
        "horizon_minutes": horizon_minutes,
        "cost_rate": cost_rate,
    }
