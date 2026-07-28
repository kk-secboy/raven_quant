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
    daily_signal = random.normal(size=len(instruments))
    signal = np.tile(daily_signal, len(dates))
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
    assert result["selection_end"] == "2022-01-27"
    assert result["final_test_purge_days"] == 1


def test_factor_direction_and_selection_purge_overlapping_forward_labels() -> None:
    dates = pd.bdate_range("2022-01-03", periods=80)
    instruments = [f"SH{600000 + index:06d}" for index in range(20)]
    index = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    signal = np.tile(np.arange(len(instruments), dtype=float), len(dates))
    factor = pd.Series(signal, index=index)
    returns = pd.Series(signal * 0.01, index=index)
    # With a ten-session label, the final ten direction candidates overlap
    # the selection boundary. Their opposite labels must not flip direction.
    direction_boundary = 14
    for day in dates[direction_boundary - 10 : direction_boundary]:
        returns.loc[(day, slice(None))] *= -1.0

    result = evaluate_factor_values(
        factor,
        returns,
        valid_start=dates[0].date(),
        valid_end=dates[-1].date(),
        test_start=(dates[-1] + pd.offsets.BDay(1)).date(),
        test_end=(dates[-1] + pd.offsets.BDay(30)).date(),
        min_daily_instruments=5,
        label_horizon_days=10,
    )

    assert result["direction"] == "original"
    assert result["direction_purge_days"] == 10
    assert result["final_test_purge_days"] == 10
    assert pd.Timestamp(result["direction_end"]) < pd.Timestamp(result["selection_start"])


def test_factor_screening_annualizes_over_the_label_holding_period() -> None:
    random = np.random.default_rng(19)
    dates = pd.bdate_range("2022-01-03", periods=80)
    instruments = [f"SH{600000 + index:06d}" for index in range(20)]
    index = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    daily_signal = random.normal(size=len(instruments))
    signal = np.tile(daily_signal, len(dates))
    factor = pd.Series(signal, index=index)
    returns = pd.Series(signal * 0.01, index=index)
    common = {
        "valid_start": date(2022, 1, 3),
        "valid_end": date(2022, 4, 22),
        "test_start": date(2022, 4, 25),
        "test_end": date(2022, 7, 29),
        "min_daily_instruments": 5,
    }

    one_day = evaluate_factor_values(
        factor, returns, label_horizon_days=1, **common
    )
    five_day = evaluate_factor_values(
        factor, returns, label_horizon_days=5, **common
    )

    assert five_day["gross_annualized_return"] == pytest.approx(
        one_day["gross_annualized_return"] / 5.0
    )
    assert (
        five_day["gross_annualized_return"]
        - five_day["cost_adjusted_return"]
    ) == pytest.approx(
        one_day["gross_annualized_return"]
        - one_day["cost_adjusted_return"],
        rel=0.10,
    )
    assert five_day["return_annualization_horizon_days"] == 5
