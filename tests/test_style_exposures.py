from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.style_exposures import (
    STYLE_COLUMNS,
    standardize_cross_section,
    standardize_panel,
    weighted_residual,
    weighted_zscore,
    winsorize_mad,
)

pytestmark = pytest.mark.no_database


def test_winsorize_mad_clips_outliers() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    clipped = winsorize_mad(values)
    scaled_mad = 1.5 * 1.4826
    assert clipped.iloc[:5].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert clipped.iloc[5] == pytest.approx(3.5 + 3.0 * scaled_mad)


def test_winsorize_mad_zero_mad_passthrough() -> None:
    values = pd.Series([2.0, 2.0, 2.0, 5.0])
    assert winsorize_mad(values).tolist() == values.tolist()


def test_weighted_zscore_properties() -> None:
    rng = np.random.default_rng(7)
    values = pd.Series(rng.normal(3.0, 2.0, size=50))
    weights = pd.Series(rng.uniform(0.5, 3.0, size=50))
    z = weighted_zscore(values, weights)
    assert np.average(z, weights=weights) == pytest.approx(0.0, abs=1e-12)
    assert np.average(z**2, weights=weights) == pytest.approx(1.0, abs=1e-12)


def test_weighted_zscore_degenerate_cross_section() -> None:
    assert weighted_zscore(pd.Series([4.0]), pd.Series([1.0])).tolist() == [0.0]
    constant = weighted_zscore(pd.Series([2.0, 2.0, 2.0]), pd.Series([1.0, 1.0, 1.0]))
    assert constant.tolist() == [0.0, 0.0, 0.0]
    result = weighted_zscore(pd.Series([1.0, np.nan]), pd.Series([1.0, 1.0]))
    assert result.iloc[0] == 0.0 and pd.isna(result.iloc[1])


def test_weighted_residual_orthogonal_to_factor() -> None:
    rng = np.random.default_rng(11)
    size = pd.Series(rng.normal(0.0, 1.0, size=40))
    values = 0.8 + 2.0 * size + pd.Series(rng.normal(0.0, 0.3, size=40))
    weights = pd.Series(rng.uniform(0.5, 2.0, size=40))
    residual = weighted_residual(values, size, weights)
    assert np.average(residual, weights=weights) == pytest.approx(0.0, abs=1e-10)
    assert np.average(residual * size, weights=weights) == pytest.approx(0.0, abs=1e-10)


def test_weighted_residual_constant_factor_demean() -> None:
    values = pd.Series([1.0, 3.0, 5.0])
    weights = pd.Series([1.0, 1.0, 1.0])
    residual = weighted_residual(values, pd.Series([2.0, 2.0, 2.0]), weights)
    assert residual.tolist() == pytest.approx([-2.0, 0.0, 2.0])


def _cross_section() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    instruments = [f"S{i:03d}" for i in range(60)]
    return pd.DataFrame(
        {
            "log_market_cap": rng.normal(10.0, 1.0, size=60),
            "book_to_price": rng.uniform(0.05, 1.5, size=60),
            "earnings_to_price": rng.uniform(-0.2, 0.3, size=60),
            "momentum": rng.normal(0.1, 0.3, size=60),
            "volatility": rng.uniform(0.01, 0.05, size=60),
            "liquidity": rng.normal(-5.0, 1.0, size=60),
            "revenue_yoy": rng.normal(10.0, 20.0, size=60),
            "netprofit_yoy": rng.normal(8.0, 25.0, size=60),
            "roe": rng.normal(8.0, 5.0, size=60),
            "debt_to_assets": rng.uniform(10.0, 90.0, size=60),
            "float_market_cap": rng.uniform(1e5, 1e7, size=60),
        },
        index=pd.Index(instruments, name="instrument"),
    )


def test_standardize_cross_section_numeric_properties() -> None:
    frame = _cross_section()
    weights = frame["float_market_cap"]
    standardized = standardize_cross_section(frame)
    assert list(standardized.columns) == list(STYLE_COLUMNS)
    for column in standardized.columns:
        z = standardized[column]
        assert np.average(z, weights=weights) == pytest.approx(0.0, abs=1e-8)
        assert np.average(z**2, weights=weights) == pytest.approx(1.0, abs=1e-8)
    size = standardized["size"]
    for column in standardized.columns:
        if column in {"size", "nonlinear_size"}:
            continue
        assert np.average(standardized[column] * size, weights=weights) == pytest.approx(
            0.0, abs=1e-8
        )
    assert np.average(standardized["nonlinear_size"] * size, weights=weights) == pytest.approx(
        0.0, abs=1e-8
    )


def test_standardize_cross_section_value_combines_descriptors() -> None:
    frame = pd.DataFrame(
        {
            "log_market_cap": [10.0, 11.0, 12.0, 13.0],
            "book_to_price": [0.1, 0.4, 0.6, 0.9],
            "earnings_to_price": [0.02, 0.05, 0.08, 0.11],
            "float_market_cap": [1.0, 1.0, 1.0, 1.0],
        },
        index=pd.Index(["A", "B", "C", "D"], name="instrument"),
    )
    standardized = standardize_cross_section(frame)
    bp_z = weighted_zscore(winsorize_mad(frame["book_to_price"]), frame["float_market_cap"])
    ep_z = weighted_zscore(
        winsorize_mad(frame["earnings_to_price"]), frame["float_market_cap"]
    )
    composite = (bp_z + ep_z) / 2.0
    residual = weighted_residual(
        composite, standardized["size"], frame["float_market_cap"]
    )
    expected = weighted_zscore(residual, frame["float_market_cap"])
    assert standardized["value"].tolist() == pytest.approx(expected.tolist())


def test_standardize_panel_groups_by_date_and_passthrough() -> None:
    first = _cross_section().reset_index()
    first["datetime"] = pd.Timestamp("2024-01-31")
    second = _cross_section().reset_index()
    second["datetime"] = pd.Timestamp("2024-02-29")
    panel = pd.concat([first, second], ignore_index=True)
    panel.loc[0, "log_market_cap"] = np.nan  # rows without market cap are dropped

    standardized = standardize_panel(panel)

    assert list(standardized.columns) == [
        "instrument",
        "datetime",
        "log_market_cap",
        *STYLE_COLUMNS,
    ]
    assert len(standardized) == 119
    raw = panel.dropna(subset=["log_market_cap"]).set_index(["instrument", "datetime"])
    lookup = standardized.set_index(["instrument", "datetime"])
    assert lookup["log_market_cap"].tolist() == pytest.approx(
        raw["log_market_cap"].reindex(lookup.index).tolist()
    )
    for timestamp, daily in standardized.groupby("datetime"):
        weights = raw.xs(timestamp, level="datetime")["float_market_cap"].reindex(
            daily["instrument"]
        )
        z = daily.set_index("instrument")["momentum"]
        assert np.average(z, weights=weights) == pytest.approx(0.0, abs=1e-8)


def test_standardize_panel_requires_key_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        standardize_panel(pd.DataFrame({"instrument": ["A"]}))
