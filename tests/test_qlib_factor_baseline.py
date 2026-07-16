from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.qlib_factor_baseline import (
    FACTOR_SOURCE_QLIB_BASELINE,
    FACTOR_SOURCE_QLIB_BASELINE_PLUS_CHALLENGER,
    QLIB_BASELINE_RECIPE_IDS,
    canonical_sha256,
    combine_factor_sources,
    core_baseline_definition,
    normalize_qlib_baseline_values,
)

pytestmark = pytest.mark.no_database


def _qlib_feature_frame() -> pd.DataFrame:
    instruments = ["SH600000", "SZ000001", "SZ000002", "SH600001"]
    datetimes = pd.to_datetime(["2025-01-02", "2025-01-03"])
    index = pd.MultiIndex.from_product(
        [instruments, datetimes], names=["instrument", "datetime"]
    )
    values = np.arange(1, len(index) * 6 + 1, dtype=float).reshape(len(index), 6)
    return pd.DataFrame(values, index=index)


def test_core_baseline_is_the_immutable_approved_six_factor_contract() -> None:
    index_definition = core_baseline_definition("index_enhancement")
    full_market_definition = core_baseline_definition("full_market_multifactor")

    assert index_definition == full_market_definition
    assert [item["weight"] for item in index_definition["factors"]] == [
        0.20,
        0.10,
        0.20,
        0.20,
        0.10,
        0.20,
    ]
    assert all(item["qlib_expression"] for item in index_definition["factors"])
    assert len(canonical_sha256(index_definition)) == 64


def test_minute_recipe_baseline_is_bound_to_its_qlib_signal_frequency() -> None:
    definition = core_baseline_definition("minute_mean_reversion")

    assert definition["frequency"] == "5min"
    assert [item["id"] for item in definition["factors"]] == [
        "oversold_60m",
        "intraday_vwap_discount",
        "lower_band_120m",
    ]
    assert sum(item["weight"] for item in definition["factors"]) == pytest.approx(1.0)


def test_swing_recipe_is_a_governed_daily_qlib_baseline_family() -> None:
    definition = core_baseline_definition("swing_trend")

    assert "swing_trend" in QLIB_BASELINE_RECIPE_IDS
    assert definition["frequency"] == "day"
    assert [item["id"] for item in definition["factors"]] == [
        "ma_trend_structure",
        "wilder_adx_14",
        "amount_expansion",
        "bollinger_bandwidth_20",
        "financial_quality",
    ]
    assert "EMA(" in definition["factors"][1]["qlib_expression"]
    assert sum(item["weight"] for item in definition["factors"]) == pytest.approx(1.0)


def test_qlib_baseline_values_are_winsorized_zscored_and_composed() -> None:
    definition = core_baseline_definition("index_enhancement")
    raw, normalized, score = normalize_qlib_baseline_values(
        _qlib_feature_frame(), definition
    )

    assert list(raw.columns) == [
        "momentum",
        "reversal",
        "value",
        "quality",
        "growth",
        "low_volatility",
    ]
    assert raw.index.names == ["datetime", "instrument"]
    daily_means = normalized.groupby(level="datetime").mean()
    assert np.allclose(daily_means.to_numpy(), 0.0, atol=1e-12)
    expected = normalized.mul(
        {
            "momentum": 0.20,
            "reversal": 0.10,
            "value": 0.20,
            "quality": 0.20,
            "growth": 0.10,
            "low_volatility": 0.20,
        },
        axis=1,
    ).sum(axis=1)
    pd.testing.assert_series_equal(score, expected.rename("score"))


def test_factor_source_combination_is_explicit() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-01-02"), "SH600000"),
            (pd.Timestamp("2025-01-02"), "SZ000001"),
        ],
        names=["datetime", "instrument"],
    )
    baseline = pd.Series([1.0, -1.0], index=index)
    challenger = pd.Series([-1.0, 1.0], index=index)

    pd.testing.assert_series_equal(
        combine_factor_sources(
            mode=FACTOR_SOURCE_QLIB_BASELINE,
            baseline=baseline,
            challenger=None,
            challenger_weight=0.0,
        ),
        baseline.rename("score"),
    )
    enhanced = combine_factor_sources(
        mode=FACTOR_SOURCE_QLIB_BASELINE_PLUS_CHALLENGER,
        baseline=baseline,
        challenger=challenger,
        challenger_weight=0.25,
    )
    assert enhanced.tolist() == pytest.approx([0.5, -0.5])


def test_formal_runner_calls_qlib_d_features_with_the_frozen_expressions(
    tmp_path: Path,
) -> None:
    from scripts.run_multifactor_backtest import _recompute_qlib_baseline

    definition = core_baseline_definition("index_enhancement")

    class FakeDataApi:
        calls: list[dict] = []

        @staticmethod
        def instruments(universe: str) -> str:
            assert universe == "cn_all"
            return universe

        @classmethod
        def features(
            cls,
            instruments: str,
            expressions: list[str],
            *,
            start_time: str,
            end_time: str,
            freq: str,
        ) -> pd.DataFrame:
            cls.calls.append(
                {
                    "instruments": instruments,
                    "expressions": expressions,
                    "start_time": start_time,
                    "end_time": end_time,
                    "freq": freq,
                }
            )
            return _qlib_feature_frame()

    raw, normalized, score = _recompute_qlib_baseline(
        FakeDataApi,
        universe="cn_all",
        definition=definition,
        start_time="2025-01-01",
        end_time="2025-12-31",
    )

    assert not raw.empty and not normalized.empty and not score.empty
    assert FakeDataApi.calls == [
        {
            "instruments": "cn_all",
            "expressions": [
                item["qlib_expression"] for item in definition["factors"]
            ],
            "start_time": "2025-01-01",
            "end_time": "2025-12-31",
            "freq": "day",
        }
    ]
