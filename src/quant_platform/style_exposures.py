"""Barra CNE5/6-style cross-sectional style exposure standardization.

Raw style descriptors (assembled point-in-time by
``quant_data.style_exposure_panel``) are turned into standardized style
exposures with the classic Barra pipeline, per trade-date cross-section:

1. winsorize each descriptor at +/- ``DEFAULT_WINSORIZE_MAD`` scaled-MAD units
   around the cross-sectional median;
2. z-score against the float-market-cap weighted mean and standard deviation;
3. combine descriptors into a composite style (equal-weight mean of the
   available descriptor z-scores, re-standardized);
4. orthogonalize every style against size (weighted regression on the size
   z-score, keep the residual) and re-standardize — the CNE5/6 treatment that
   keeps non-size styles size-neutral. Industry orthogonalization is left to
   the downstream consumer (the structured risk model already carries explicit
   industry factors, so double-neutralizing here is unnecessary).

The ``size`` style itself is only winsorized and z-scored. ``nonlinear_size``
is the cube of the size z-score, orthogonalized against size and re-scored
(Barra NLSIZE).

All functions are pure pandas/numpy and independently testable; nothing here
knows about snapshots, Qlib, or persistence. Cross-sections with fewer than
two usable observations or zero weighted variance standardize to 0.0 (the
weighted mean) instead of NaN so single-stock metadata rows stay finite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STYLE_SCHEMA_VERSION = 1

MAD_SCALE = 1.4826
DEFAULT_WINSORIZE_MAD = 3.0

WEIGHT_COLUMN = "float_market_cap"

# Descriptor column -> composite style. Descriptors are produced by
# quant_data.style_exposure_panel.build_raw_style_panel.
STYLE_DESCRIPTORS: dict[str, tuple[str, ...]] = {
    "size": ("log_market_cap",),
    "value": ("book_to_price", "earnings_to_price"),
    "momentum": ("momentum",),
    "volatility": ("volatility",),
    "liquidity": ("liquidity",),
    "growth": ("revenue_yoy", "netprofit_yoy"),
    "profitability": ("roe",),
    "leverage": ("debt_to_assets",),
}

# Styles exposed to consumers, in stable order.
STYLE_COLUMNS: tuple[str, ...] = (*STYLE_DESCRIPTORS, "nonlinear_size")

# Columns that may appear in a style panel but are never style factors:
# identifiers, the backward-compatible raw log market cap, and the
# standardization weight.
NON_STYLE_COLUMNS = frozenset({"instrument", "datetime", "log_market_cap", WEIGHT_COLUMN})


def winsorize_mad(values: pd.Series, *, n_mad: float = DEFAULT_WINSORIZE_MAD) -> pd.Series:
    """Clip a cross-section at median +/- n_mad * scaled MAD."""

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    median = numeric.median()
    if not np.isfinite(median):
        return numeric
    mad = float((numeric - median).abs().median()) * MAD_SCALE
    if not np.isfinite(mad) or mad <= 0:
        return numeric
    return numeric.clip(median - n_mad * mad, median + n_mad * mad)


def weighted_zscore(values: pd.Series, weights: pd.Series) -> pd.Series:
    """Weighted z-score; degenerate cross-sections standardize to 0.0."""

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    weight = pd.to_numeric(weights.reindex(numeric.index), errors="coerce").astype(float)
    valid = numeric.notna() & weight.notna() & (weight > 0) & np.isfinite(numeric)
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    if int(valid.sum()) < 2:
        result[valid] = 0.0
        return result
    x = numeric[valid]
    w = weight[valid]
    mean = float(np.dot(x, w) / w.sum())
    variance = float(np.dot((x - mean) ** 2, w) / w.sum())
    if variance <= 0 or not np.isfinite(variance):
        result[valid] = 0.0
        return result
    result[valid] = (x - mean) / np.sqrt(variance)
    return result


def weighted_residual(values: pd.Series, factor: pd.Series, weights: pd.Series) -> pd.Series:
    """Residuals of the weighted regression values ~ 1 + factor."""

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    base = pd.to_numeric(factor.reindex(numeric.index), errors="coerce").astype(float)
    weight = pd.to_numeric(weights.reindex(numeric.index), errors="coerce").astype(float)
    valid = (
        numeric.notna()
        & base.notna()
        & weight.notna()
        & (weight > 0)
        & np.isfinite(numeric)
        & np.isfinite(base)
    )
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    if int(valid.sum()) < 2:
        result[valid] = 0.0
        return result
    x = base[valid]
    y = numeric[valid]
    w = weight[valid]
    total = float(w.sum())
    x_mean = float(np.dot(x, w) / total)
    y_mean = float(np.dot(y, w) / total)
    x_var = float(np.dot((x - x_mean) ** 2, w) / total)
    if x_var <= 0 or not np.isfinite(x_var):
        result[valid] = y - y_mean
        return result
    covariance = float(np.dot((x - x_mean) * (y - y_mean), w) / total)
    slope = covariance / x_var
    result[valid] = y - (y_mean + slope * (x - x_mean))
    return result


def standardize_cross_section(
    frame: pd.DataFrame,
    *,
    weight_column: str = WEIGHT_COLUMN,
    n_mad: float = DEFAULT_WINSORIZE_MAD,
) -> pd.DataFrame:
    """Standardize one trade-date cross-section of raw descriptors.

    ``frame`` is indexed by instrument and carries the descriptor columns of
    ``STYLE_DESCRIPTORS`` plus a float-cap weight column. Returns a frame with
    ``STYLE_COLUMNS`` (all of ``STYLE_DESCRIPTORS`` plus ``nonlinear_size``).
    Styles whose descriptors are entirely missing are all-NaN.
    """

    weights = (
        pd.to_numeric(frame[weight_column], errors="coerce").astype(float)
        if weight_column in frame.columns
        else pd.Series(1.0, index=frame.index)
    )

    def descriptor_zscores(style: str) -> list[pd.Series]:
        scores = []
        for descriptor in STYLE_DESCRIPTORS[style]:
            if descriptor not in frame.columns:
                continue
            raw = pd.to_numeric(frame[descriptor], errors="coerce").astype(float)
            scores.append(weighted_zscore(winsorize_mad(raw, n_mad=n_mad), weights))
        return scores

    composites: dict[str, pd.Series] = {}
    for style in STYLE_DESCRIPTORS:
        scores = descriptor_zscores(style)
        if not scores:
            composites[style] = pd.Series(np.nan, index=frame.index, dtype=float)
            continue
        combined = pd.concat(scores, axis=1).mean(axis=1, skipna=True)
        all_missing = pd.concat(scores, axis=1).isna().all(axis=1)
        composites[style] = combined.mask(all_missing)

    size = composites["size"]
    standardized: dict[str, pd.Series] = {"size": size}
    for style in STYLE_DESCRIPTORS:
        if style == "size":
            continue
        composite = composites[style]
        if composite.notna().sum() < 2:
            standardized[style] = composite
            continue
        orthogonal = weighted_residual(composite, size, weights)
        # No winsorizing after orthogonalization: clipping would break the
        # exact size-neutrality of the residual. Re-standardizing is a linear
        # transform and preserves it.
        standardized[style] = weighted_zscore(orthogonal, weights)

    nonlinear = size**3
    nonlinear = weighted_residual(winsorize_mad(nonlinear, n_mad=n_mad), size, weights)
    standardized["nonlinear_size"] = weighted_zscore(nonlinear, weights)
    return pd.DataFrame({column: standardized[column] for column in STYLE_COLUMNS})


def standardize_panel(
    panel: pd.DataFrame,
    *,
    weight_column: str = WEIGHT_COLUMN,
    n_mad: float = DEFAULT_WINSORIZE_MAD,
) -> pd.DataFrame:
    """Standardize a long raw-descriptor panel per trade date.

    ``panel`` carries ``instrument``, ``datetime``, descriptor columns and the
    float-cap weight column. Returns ``instrument``, ``datetime``, the raw
    ``log_market_cap`` passthrough (backward-compatible with the historical
    style_exposures.parquet schema) and every column of ``STYLE_COLUMNS``.
    Rows without a usable size exposure are dropped, matching the historical
    behavior of dropping rows without market cap.
    """

    required = {"instrument", "datetime", "log_market_cap"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"style panel is missing required columns: {sorted(missing)}")
    frame = panel.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["instrument"] = frame["instrument"].astype(str)
    frame = frame.dropna(subset=["instrument", "datetime", "log_market_cap"])
    frame = frame.drop_duplicates(["instrument", "datetime"], keep="last")
    if frame.empty:
        return pd.DataFrame(
            columns=["instrument", "datetime", "log_market_cap", *STYLE_COLUMNS]
        )

    pieces: list[pd.DataFrame] = []
    for timestamp, daily in frame.groupby("datetime", sort=True):
        cross_section = daily.set_index("instrument")
        standardized = standardize_cross_section(
            cross_section, weight_column=weight_column, n_mad=n_mad
        )
        standardized.insert(0, "datetime", timestamp)
        standardized.insert(
            1,
            "log_market_cap",
            pd.to_numeric(cross_section["log_market_cap"], errors="coerce"),
        )
        pieces.append(standardized.reset_index(names="instrument"))
    result = pd.concat(pieces, ignore_index=True)
    return result.sort_values(["datetime", "instrument"], ignore_index=True)
