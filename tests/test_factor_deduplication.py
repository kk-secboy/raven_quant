from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.factor_deduplication import deduplicate_daily_factors

pytestmark = pytest.mark.no_database


def _ic(frame, old_count, new_count):
    return pd.Series([
        frame.iloc[:, old].corr(frame.iloc[:, new])
        for old in range(old_count) for new in range(old_count, old_count + new_count)
    ])


@pytest.mark.parametrize("ragged", [False, True])
def test_daily_kernel_matches_full_history_alignment_nan_and_signed_threshold(ragged):
    rng = np.random.default_rng(34)
    index = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=8), range(40)],
        names=["datetime", "instrument"],
    )
    old = pd.DataFrame(rng.normal(size=(len(index), 3)), index=index)
    new = pd.DataFrame({"copy": old[0], "negative": -old[0], "constant": 1.0,
                        "independent": rng.normal(size=len(index)), "missing": np.nan})
    new.iloc[::9, 3] = np.nan
    if ragged:
        old = old.iloc[15:-40].sample(frac=1, random_state=1)
        new = new.iloc[40:-7].sample(frac=1, random_state=2)
    combined = pd.concat([old, new], axis=1)
    mean = combined.groupby("datetime").apply(lambda day: _ic(day, 3, 5)).mean()
    mean.index = pd.MultiIndex.from_product([range(3), range(5)])
    maximum = mean.unstack().max(axis=0)
    expected = new.iloc[:, maximum[maximum < 0.99].index]
    seen_rows = []

    def bounded_ic(day, nold, nnew):
        seen_rows.append(len(day))
        assert day.index.get_level_values("datetime").nunique() == 1
        return _ic(day, nold, nnew)

    actual = deduplicate_daily_factors(old, new, bounded_ic)
    pd.testing.assert_frame_equal(actual, expected)
    assert list(actual) == ["negative", "independent"]
    assert max(seen_rows) <= 40


def test_no_shared_dates_returns_no_valid_factors():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=2), range(3)],
        names=["datetime", "instrument"],
    )
    old = pd.DataFrame({"old": range(3)}, index=index[:3])
    new = pd.DataFrame({"new": range(3)}, index=index[3:])
    pd.testing.assert_frame_equal(deduplicate_daily_factors(old, new, _ic), new.iloc[:, :0])
