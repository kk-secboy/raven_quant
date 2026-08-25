from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.strategy_backtest import compose_factor_scores

pytestmark = pytest.mark.no_database


def test_formal_factor_composition_rejects_mismatched_indexes() -> None:
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    full_index = pd.MultiIndex.from_product(
        [dates, ["SH600000", "SH600001"]], names=["datetime", "instrument"]
    )
    left = pd.Series(range(len(full_index)), index=full_index, dtype=float)
    right = left.iloc[:-1]

    with pytest.raises(ValueError, match="index differs"):
        compose_factor_scores(
            [(left, 0.5, 1), (right, 0.5, 1)],
            require_exact_index=True,
        )


def test_legacy_factor_composition_keeps_the_explicit_non_strict_mode() -> None:
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    full_index = pd.MultiIndex.from_product(
        [dates, ["SH600000", "SH600001"]], names=["datetime", "instrument"]
    )
    left = pd.Series(range(len(full_index)), index=full_index, dtype=float)
    right = left.iloc[:-1]

    result = compose_factor_scores([(left, 0.5, 1), (right, 0.5, 1)])

    assert len(result) == len(right)
