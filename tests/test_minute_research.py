from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.minute_research import evaluate_minute_factor

pytestmark = pytest.mark.no_database


def test_evaluates_cross_sectional_minute_factor_with_costs() -> None:
    timestamps = pd.date_range("2026-07-10 09:31", periods=40, freq="min")
    instruments = [f"SH60{index:04d}" for index in range(20)]
    index = pd.MultiIndex.from_product(
        [timestamps, instruments], names=["datetime", "instrument"]
    )
    cross_section = np.tile(np.linspace(-1, 1, len(instruments)), len(timestamps))
    factor = pd.Series(cross_section, index=index)
    label = pd.Series(cross_section * 0.001, index=index)

    metrics = evaluate_minute_factor(
        factor,
        label,
        horizon_minutes=5,
        cost_rate=0.0001,
    )

    assert metrics["rank_ic"] == pytest.approx(1.0)
    assert metrics["rebalance_timestamps"] == 8
    assert metrics["mean_net_return"] > 0
    assert metrics["average_turnover"] >= 0


def test_minute_factor_requires_cross_section() -> None:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-07-10 09:31", periods=5, freq="min"), ["SH600000"]],
        names=["datetime", "instrument"],
    )
    values = pd.Series(1.0, index=index)
    with pytest.raises(ValueError, match="insufficient"):
        evaluate_minute_factor(
            values,
            values,
            horizon_minutes=1,
            cost_rate=0.0001,
        )
