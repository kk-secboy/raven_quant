"""Keep upstream daily Pearson deduplication without copying the full history."""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

logger = logging.getLogger(__name__)


def deduplicate_daily_factors(
    sota: pd.DataFrame,
    proposed: pd.DataFrame,
    calculate_ic: Callable[[pd.DataFrame, int, int], pd.Series],
) -> pd.DataFrame:
    """Use the upstream IC kernel and selection rule, with one day's working set.

    Group indices contain only row positions. Iterating DataFrameGroupBy itself
    would first make a sorted copy of the entire matrix. Never send these frames
    (or a closure referencing them) to multiprocessing workers.
    """
    old_count, new_count = sota.shape[1], proposed.shape[1]
    old_days = sota.groupby("datetime", sort=True).indices
    new_days = proposed.groupby("datetime", sort=True).indices
    daily = []
    days = sorted(old_days.keys() & new_days.keys())
    for number, day in enumerate(days, 1):
        combined = pd.concat(
            [sota.iloc[old_days[day]], proposed.iloc[new_days[day]]], axis=1
        )
        daily.append(calculate_ic(combined, old_count, new_count))
        if number == 1 or number % 250 == 0 or number == len(days):
            logger.info("Factor deduplication: %s/%s trading days", number, len(days))
    # Dates present on only one side have all-NaN correlations and do not
    # contribute to upstream's skip-NaN mean. Keep equal weighting by date,
    # positive (not absolute) correlation, and the strict upstream threshold.
    correlations = pd.DataFrame(daily, columns=range(old_count * new_count)).mean()
    correlations.index = pd.MultiIndex.from_product([range(old_count), range(new_count)])
    if not old_count or not new_count:
        return proposed.iloc[:, :0]
    maximum = correlations.unstack().max(axis=0)
    return proposed.iloc[:, maximum[maximum < 0.99].index]
