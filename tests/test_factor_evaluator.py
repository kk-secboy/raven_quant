from datetime import date

import numpy as np
import pandas as pd

from quant_platform.factor_evaluator import evaluate_factor_values


def test_factor_evaluator_uses_validation_direction_and_costs() -> None:
    random = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=140)
    instruments = [f"SH{600000 + index:06d}" for index in range(80)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    signal = random.normal(size=len(index))
    factor = pd.Series(-signal, index=index, name="factor")
    returns = pd.Series(signal * 0.02 + random.normal(scale=0.01, size=len(index)), index=index)
    result = evaluate_factor_values(
        factor,
        returns,
        valid_start=date(2022, 1, 3),
        valid_end=date(2022, 3, 31),
        test_start=date(2022, 4, 1),
        test_end=date(2022, 7, 15),
        cost_rate=0.002,
    )
    assert result["direction"] == "inverted"
    assert result["ic"] and result["ic"] > 0.8
    assert result["rank_ic"] and result["rank_ic"] > 0.8
    assert result["cost_adjusted_return"] and result["cost_adjusted_return"] > 0
    assert result["gross_annualized_return"] > result["cost_adjusted_return"]
    assert 0 <= result["turnover"] <= 2
