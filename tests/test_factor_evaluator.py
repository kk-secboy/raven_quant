from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_platform.cost_model import CostModelConfig
from quant_platform.factor_evaluator import evaluate_factor_values

pytestmark = pytest.mark.no_database


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
        cost_model=CostModelConfig(),
        reference_order_value=100_000,
    )
    assert result["direction"] == "inverted"
    assert result["ic"] and result["ic"] > 0.8
    assert result["rank_ic"] and result["rank_ic"] > 0.8
    assert result["cost_adjusted_return"] and result["cost_adjusted_return"] > 0
    assert result["gross_annualized_return"] > result["cost_adjusted_return"]
    assert 0 <= result["turnover"] <= 2
    assert result["cost_rate"] == CostModelConfig().factor_screening_rate(
        reference_order_value=100_000
    )
    assert result["cost_model"] == CostModelConfig().to_dict()


def test_factor_evaluator_never_reads_reserved_test_values() -> None:
    dates = pd.bdate_range("2022-01-03", periods=20)
    instruments = [f"SH{600000 + index:06d}" for index in range(10)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    values = np.tile(np.arange(10, dtype=float), len(dates))
    factor = pd.Series(values, index=index)
    returns = pd.Series(values * 0.01, index=index)
    reserved = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2022-02-01"), "SH600000")] * 2,
        names=["datetime", "instrument"],
    )
    factor = pd.concat([factor, pd.Series([np.inf, np.inf], index=reserved)])
    returns = pd.concat([returns, pd.Series([np.inf, np.inf], index=reserved)])
    result = evaluate_factor_values(
        factor,
        returns,
        valid_start=date(2022, 1, 3),
        valid_end=date(2022, 1, 28),
        test_start=date(2022, 2, 1),
        test_end=date(2022, 3, 31),
        min_daily_instruments=5,
    )
    assert result["selection_end"] == "2022-01-28"
