from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .cost_model import CostModelConfig


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
    if frame.duplicated(["datetime", "instrument"]).any():
        raise ValueError(f"{name} contains duplicate datetime/instrument observations")
    finite = np.isfinite(frame[name].to_numpy(dtype=float, na_value=np.nan))
    if (~finite & frame[name].notna().to_numpy()).any():
        raise ValueError(f"{name} contains infinite values")
    return frame.dropna().set_index(["datetime", "instrument"])[name].sort_index()


def _validation_window(
    values: pd.Series | pd.DataFrame, start: date, end: date
) -> pd.Series | pd.DataFrame:
    if not isinstance(values.index, pd.MultiIndex) or values.index.nlevels != 2:
        return values
    datetime_level = values.index.names.index("datetime") if "datetime" in values.index.names else 0
    dates = pd.to_datetime(values.index.get_level_values(datetime_level)).tz_localize(None)
    return values[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


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
    cost_model: CostModelConfig | None = None,
    reference_order_value: float = 100_000.0,
    min_daily_instruments: int = 50,
    min_coverage_ratio: float = 0.80,
    min_good_day_rate: float = 0.95,
    max_constant_day_rate: float = 0.05,
) -> dict[str, Any]:
    if valid_end >= test_start or test_start > test_end:
        raise ValueError("validation and reserved final-test windows must not overlap")
    if min_daily_instruments < 5:
        raise ValueError("min_daily_instruments must be at least 5")
    if not 0 < min_coverage_ratio <= 1 or not 0 < min_good_day_rate <= 1:
        raise ValueError("coverage thresholds must be in (0, 1]")
    if not 0 <= max_constant_day_rate < 1:
        raise ValueError("max_constant_day_rate must be in [0, 1)")
    factor = normalize_series(_validation_window(factor_values, valid_start, valid_end), "factor")
    label = normalize_series(_validation_window(forward_returns, valid_start, valid_end), "label")
    joined = pd.concat([factor, label], axis=1, join="inner").dropna()
    dates = joined.index.get_level_values("datetime")
    valid = joined[(dates >= pd.Timestamp(valid_start)) & (dates <= pd.Timestamp(valid_end))]
    if valid.empty:
        raise ValueError("factor and forward-return data must cover the validation window")
    valid_days = pd.DatetimeIndex(valid.index.get_level_values("datetime").unique()).sort_values()
    if len(valid_days) < 10:
        raise ValueError("validation window must contain at least 10 trading days")
    direction_count = max(1, int(np.ceil(len(valid_days) * 0.20)))
    if direction_count >= len(valid_days):
        raise ValueError("validation window cannot be split into direction and selection periods")
    direction_end = valid_days[direction_count - 1]
    selection_start = valid_days[direction_count]
    direction_frame = valid[valid.index.get_level_values("datetime") <= direction_end]
    selection = valid[valid.index.get_level_values("datetime") >= selection_start]

    direction_ic_daily = _daily_correlation(direction_frame, "pearson")
    raw_ic_daily = _daily_correlation(selection, "pearson")
    raw_rank_ic_daily = _daily_correlation(selection, "spearman")
    if direction_ic_daily.empty or raw_ic_daily.empty or raw_rank_ic_daily.empty:
        raise ValueError("validation window has insufficient cross-sectional observations")
    raw_valid_ic = float(direction_ic_daily.mean())
    direction = -1.0 if raw_valid_ic < 0 else 1.0
    directed_selection = selection.copy()
    directed_selection["factor"] *= direction
    ic_daily = _daily_correlation(directed_selection, "pearson")
    rank_ic_daily = _daily_correlation(directed_selection, "spearman")
    if ic_daily.empty or rank_ic_daily.empty:
        raise ValueError("selection window has insufficient cross-sectional observations")
    ic = float(ic_daily.mean())
    rank_ic = float(rank_ic_daily.mean())
    icir = float(ic / ic_daily.std(ddof=1)) if ic_daily.std(ddof=1) > 0 else None
    rank_std = rank_ic_daily.std(ddof=1)
    rank_icir = float(rank_ic / rank_std) if rank_std > 0 else None
    costs = cost_model or CostModelConfig()
    screening_cost_rate = costs.factor_screening_rate(reference_order_value=reference_order_value)
    turnover, gross_return, cost_adjusted_return = _long_short_returns(
        directed_selection, screening_cost_rate
    )

    selection_factor = factor[
        (factor.index.get_level_values("datetime") >= selection_start)
        & (factor.index.get_level_values("datetime") <= valid_days[-1])
    ]
    selection_label = label[
        (label.index.get_level_values("datetime") >= selection_start)
        & (label.index.get_level_values("datetime") <= valid_days[-1])
    ]
    factor_counts = selection_factor.groupby(level="datetime").size()
    universe_counts = selection_label.groupby(level="datetime").size()
    coverage = factor_counts.reindex(universe_counts.index, fill_value=0).div(universe_counts)
    daily_minimum = max(min_daily_instruments, 5)
    enough_instruments = factor_counts.reindex(universe_counts.index, fill_value=0) >= daily_minimum
    coverage_pass = enough_instruments & (coverage >= min_coverage_ratio)
    coverage_pass_rate = float(coverage_pass.mean()) if len(coverage_pass) else 0.0
    constant_days = selection_factor.groupby(level="datetime").nunique().le(1)
    constant_day_rate = float(constant_days.mean()) if len(constant_days) else 1.0

    correlations: list[float] = []
    directed_factor = factor * direction
    for comparison in comparison_values:
        other = normalize_series(
            _validation_window(comparison, valid_start, valid_end), "comparison"
        )
        pair = pd.concat([directed_factor, other], axis=1, join="inner").dropna()
        pair_dates = pair.index.get_level_values("datetime")
        pair = pair[(pair_dates >= selection_start) & (pair_dates <= valid_days[-1])]
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
        "valid_ic": float(raw_valid_ic * direction),
        "raw_valid_ic": raw_valid_ic,
        "raw_selection_ic": float(raw_ic_daily.mean()),
        "raw_selection_rank_ic": float(raw_rank_ic_daily.mean()),
        "direction": "inverted" if direction < 0 else "original",
        "observations": int(len(selection)),
        "selection_days": int(selection.index.get_level_values("datetime").nunique()),
        "direction_start": valid_days[0].date().isoformat(),
        "direction_end": direction_end.date().isoformat(),
        "selection_start": selection_start.date().isoformat(),
        "selection_end": valid_days[-1].date().isoformat(),
        "coverage_pass_rate": coverage_pass_rate,
        "mean_coverage_ratio": float(coverage.mean()) if len(coverage) else 0.0,
        "min_daily_instruments_observed": int(factor_counts.min()) if len(factor_counts) else 0,
        "constant_day_rate": constant_day_rate,
        "coverage_gate_passed": bool(
            coverage_pass_rate >= min_good_day_rate and constant_day_rate <= max_constant_day_rate
        ),
        "cost_rate": screening_cost_rate,
        "cost_model": costs.to_dict(),
        "cost_reference_order_value": reference_order_value,
    }
