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
    index = pd.MultiIndex.from_product([instruments, datetimes], names=["instrument", "datetime"])
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
    raw, normalized, score = normalize_qlib_baseline_values(_qlib_feature_frame(), definition)

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
            "expressions": [item["qlib_expression"] for item in definition["factors"]],
            "start_time": "2025-01-01",
            "end_time": "2025-12-31",
            "freq": "day",
        }
    ]


def test_formal_runner_restricts_exchange_universe_to_eligible_assets() -> None:
    from scripts.run_multifactor_backtest import _eligible_strategy_instruments

    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-01-02"), "SH600000"),
            (pd.Timestamp("2025-01-02"), "SZ000001"),
            (pd.Timestamp("2025-01-02"), "BJ899050"),
        ],
        names=["datetime", "instrument"],
    )
    scores = pd.Series([1.0, 0.5, 0.1], index=index)
    eligibility = pd.DataFrame(
        [
            {"datetime": "2025-01-02", "instrument": "SH600000", "eligible": True},
            {"datetime": "2025-01-02", "instrument": "SZ000001", "eligible": False},
        ]
    )

    assert _eligible_strategy_instruments(scores, eligibility) == ["SH600000"]


def test_standardized_style_snapshot_neutral_imputes_only_sparse_missing() -> None:
    from scripts.run_multifactor_backtest import _latest_style_cross_section

    instruments = [f"S{index:04d}" for index in range(100)]
    frame = pd.DataFrame(
        {
            "datetime": pd.Timestamp("2025-01-02"),
            "instrument": instruments,
            "size": np.linspace(-2.0, 2.0, len(instruments)),
            "value": 0.5,
            "growth": 0.25,
            "volatility": -0.1,
        }
    )
    frame.loc[0, "growth"] = np.nan

    result = _latest_style_cross_section(frame, pd.Timestamp("2025-01-03"))

    assert result.loc["S0000", "growth"] == pytest.approx(0.0)
    assert result.loc["S0001", "growth"] == pytest.approx(0.25)
    assert np.isfinite(result.to_numpy()).all()


def test_standardized_style_snapshot_rejects_systemic_missing() -> None:
    from scripts.run_multifactor_backtest import _latest_style_cross_section

    frame = pd.DataFrame(
        {
            "datetime": pd.Timestamp("2025-01-02"),
            "instrument": [f"S{index:04d}" for index in range(20)],
            "size": 0.0,
            "value": 0.0,
            "growth": [np.nan, np.nan, *([0.0] * 18)],
            "volatility": 0.0,
        }
    )

    with pytest.raises(ValueError, match=r"growth=10.00%"):
        _latest_style_cross_section(frame, pd.Timestamp("2025-01-02"))


def test_formal_runner_loads_versioned_builder_style_contract(tmp_path: Path) -> None:
    from scripts.run_multifactor_backtest import _load_governed_style_exposures

    metadata = tmp_path / "metadata"
    metadata.mkdir()
    source = pd.DataFrame(
        {
            "instrument": ["SH600000", "SZ000001"],
            "datetime": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "size": [-0.5, 0.5],
            "value": [0.2, np.nan],
            "growth": [0.1, -0.1],
            "volatility": [-0.2, 0.2],
            "unused_descriptor": [1.0, 2.0],
        }
    )
    source.to_parquet(metadata / "style_exposures.parquet", index=False)

    frame, evidence = _load_governed_style_exposures(tmp_path)

    assert list(frame.columns) == [
        "instrument",
        "datetime",
        "size",
        "value",
        "growth",
        "volatility",
    ]
    assert evidence["contract_version"] == "standardized-neutral-imputation-v1"
    assert evidence["missing_counts"] == {
        "size": 0,
        "value": 1,
        "growth": 0,
        "volatility": 0,
    }
    assert len(evidence["sha256"]) == 64
