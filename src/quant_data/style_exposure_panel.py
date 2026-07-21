"""Raw point-in-time style descriptor panel assembly from snapshot frames.

Builds the raw descriptor columns consumed by
``quant_platform.style_exposures.standardize_panel`` from the immutable
snapshot datasets. Every descriptor is computable from information available
on the row's trade date:

- market-data descriptors (size, value, liquidity, momentum, volatility) come
  from ``daily_basic`` and adjusted closes; both carry the
  ``same_trade_date_after_close`` availability policy, so an exposure dated
  ``t`` supports decisions made after the close of ``t`` (next-session
  trading);
- fundamental descriptors (growth, profitability, leverage) come from
  ``fina_indicator`` through the announcement-date ASOF channel
  (``trade_date > ann_date``, strictly after announcement, matching the
  platform availability registry); the report period ``end_date`` is never
  used for availability.

Window choices follow the simplified Barra CNE5/6 conventions for
medium/low-frequency use: momentum is the trailing 252-trading-day return
skipping the most recent 21 days, volatility is the 120-day standard
deviation of daily adjusted returns, liquidity is the log of the 21-day mean
turnover fraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM_LOOKBACK_DAYS = 252
MOMENTUM_SKIP_DAYS = 21
VOLATILITY_LOOKBACK_DAYS = 120
VOLATILITY_MIN_OBSERVATIONS = 60
LIQUIDITY_LOOKBACK_DAYS = 21
LIQUIDITY_MIN_OBSERVATIONS = 5

# fina_indicator source column -> raw descriptor column.
FUNDAMENTAL_DESCRIPTORS = {
    "roe": "roe",
    "or_yoy": "revenue_yoy",
    "netprofit_yoy": "netprofit_yoy",
    "debt_to_assets": "debt_to_assets",
}

DESCRIPTOR_COLUMNS: tuple[str, ...] = (
    "log_market_cap",
    "book_to_price",
    "earnings_to_price",
    "momentum",
    "volatility",
    "liquidity",
    "revenue_yoy",
    "netprofit_yoy",
    "roe",
    "debt_to_assets",
)


def parse_trade_dates(values: pd.Series) -> pd.Series:
    """Parse snapshot date columns stored as ISO text, YYYYMMDD text or dates."""

    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        # Numeric snapshots store YYYYMMDD integers; pd.to_datetime would
        # misread them as nanosecond epochs.
        return pd.to_datetime(
            series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
        )
    parsed = pd.to_datetime(series.astype(str), format="mixed", errors="coerce")
    return parsed


def build_adjusted_close(
    daily: pd.DataFrame, adj_factor: pd.DataFrame | None
) -> pd.DataFrame | None:
    """Per-instrument adjusted close series from raw daily bars and factors."""

    if daily is None or daily.empty or not {"ts_code", "trade_date", "close"}.issubset(
        daily.columns
    ):
        return None
    frame = daily.loc[:, ["ts_code", "trade_date", "close"]].copy()
    if adj_factor is not None and {"ts_code", "trade_date", "adj_factor"}.issubset(
        adj_factor.columns
    ):
        factors = adj_factor.loc[:, ["ts_code", "trade_date", "adj_factor"]].copy()
        factors["trade_date"] = parse_trade_dates(factors["trade_date"])
        frame["trade_date"] = parse_trade_dates(frame["trade_date"])
        frame = frame.merge(factors, on=["ts_code", "trade_date"], how="left")
        frame["adj_close"] = pd.to_numeric(frame["close"], errors="coerce") * pd.to_numeric(
            frame["adj_factor"], errors="coerce"
        ).fillna(1.0)
    else:
        frame["trade_date"] = parse_trade_dates(frame["trade_date"])
        frame["adj_close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["ts_code", "trade_date", "adj_close"])
    frame = frame.drop_duplicates(["ts_code", "trade_date"], keep="last")
    return frame.sort_values(["ts_code", "trade_date"], ignore_index=True)[
        ["ts_code", "trade_date", "adj_close"]
    ]


def build_raw_style_panel(
    daily_basic: pd.DataFrame,
    adjusted_close: pd.DataFrame | None = None,
    fina_indicator: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the raw descriptor panel from snapshot-level frames.

    ``daily_basic`` needs ``ts_code``, ``trade_date`` and ``total_mv``;
    ``pb``, ``pe_ttm``, ``turnover_rate`` and ``circ_mv`` are used when
    present. ``adjusted_close`` is the output of :func:`build_adjusted_close`.
    ``fina_indicator`` needs ``ts_code``, ``ann_date`` and any of the
    ``FUNDAMENTAL_DESCRIPTORS`` source columns. Returns one row per
    (``ts_code``, ``trade_date``) with ``float_market_cap`` and the
    ``DESCRIPTOR_COLUMNS``; unavailable descriptors stay NaN rather than
    being fabricated.
    """

    required = {"ts_code", "trade_date", "total_mv"}
    if not required.issubset(daily_basic.columns):
        raise ValueError(f"daily_basic frame is missing columns: {sorted(required)}")
    frame = daily_basic.copy()
    frame["trade_date"] = parse_trade_dates(frame["trade_date"])
    frame = frame.dropna(subset=["ts_code", "trade_date"])
    frame = frame.drop_duplicates(["ts_code", "trade_date"], keep="last")
    frame = frame.sort_values(["ts_code", "trade_date"], ignore_index=True)

    total_mv = pd.to_numeric(frame["total_mv"], errors="coerce")
    frame["log_market_cap"] = total_mv.where(total_mv > 0).map(np.log)
    circ_mv = _numeric_column(frame, "circ_mv")
    frame["float_market_cap"] = circ_mv.where(circ_mv > 0).fillna(
        total_mv.where(total_mv > 0)
    )

    pb = _numeric_column(frame, "pb")
    frame["book_to_price"] = (1.0 / pb).where(pb > 0)
    pe = _numeric_column(frame, "pe_ttm")
    frame["earnings_to_price"] = (1.0 / pe).where(pe > 0)

    _add_market_descriptors(frame, adjusted_close)
    _add_fundamental_descriptors(frame, fina_indicator)

    columns = ["ts_code", "trade_date", "float_market_cap", *DESCRIPTOR_COLUMNS]
    return frame.loc[:, columns]


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").astype(float)


def _add_market_descriptors(
    frame: pd.DataFrame, adjusted_close: pd.DataFrame | None
) -> None:
    frame["momentum"] = np.nan
    frame["volatility"] = np.nan
    frame["liquidity"] = np.nan
    if adjusted_close is None or adjusted_close.empty:
        return
    merged = frame.merge(adjusted_close, on=["ts_code", "trade_date"], how="left")
    merged = merged.sort_values(["ts_code", "trade_date"], ignore_index=True)
    grouped = merged.groupby("ts_code", sort=False)["adj_close"]
    skipped = grouped.shift(MOMENTUM_SKIP_DAYS)
    base = grouped.shift(MOMENTUM_LOOKBACK_DAYS)
    valid = (skipped > 0) & (base > 0)
    merged["momentum"] = (skipped / base - 1.0).where(valid)

    returns = grouped.pct_change(fill_method=None)
    merged["volatility"] = returns.groupby(merged["ts_code"], sort=False).transform(
        lambda series: series.rolling(
            VOLATILITY_LOOKBACK_DAYS, min_periods=VOLATILITY_MIN_OBSERVATIONS
        ).std()
    )

    turnover = _numeric_column(frame, "turnover_rate") / 100.0
    merged_turnover = turnover.reindex(merged.index)
    mean_turnover = merged_turnover.groupby(merged["ts_code"], sort=False).transform(
        lambda series: series.rolling(
            LIQUIDITY_LOOKBACK_DAYS, min_periods=LIQUIDITY_MIN_OBSERVATIONS
        ).mean()
    )
    merged["liquidity"] = mean_turnover.where(mean_turnover > 0).map(np.log)

    frame["momentum"] = merged["momentum"].to_numpy()
    frame["volatility"] = merged["volatility"].to_numpy()
    frame["liquidity"] = merged["liquidity"].to_numpy()


def _add_fundamental_descriptors(
    frame: pd.DataFrame, fina_indicator: pd.DataFrame | None
) -> None:
    for target in FUNDAMENTAL_DESCRIPTORS.values():
        frame[target] = np.nan
    if fina_indicator is None or fina_indicator.empty:
        return
    available = {
        source: target
        for source, target in FUNDAMENTAL_DESCRIPTORS.items()
        if source in fina_indicator.columns
    }
    if not available or not {"ts_code", "ann_date"}.issubset(fina_indicator.columns):
        return
    fina = fina_indicator.loc[:, ["ts_code", "ann_date", *available]].copy()
    fina["ann_date"] = parse_trade_dates(fina["ann_date"])
    fina = fina.dropna(subset=["ts_code", "ann_date"])
    # Deterministic revision resolution: the newest announced version of a
    # conflicting (ts_code, ann_date) row wins.
    fina = fina.sort_values(["ts_code", "ann_date"]).drop_duplicates(
        ["ts_code", "ann_date"], keep="last"
    )
    fina = fina.rename(columns=available).sort_values("ann_date", ignore_index=True)

    left = frame.drop(columns=list(available.values()), errors="ignore").reset_index(
        drop=True
    )
    left["_panel_row"] = np.arange(len(left))
    left = left.sort_values("trade_date", ignore_index=True)
    matched = pd.merge_asof(
        left,
        fina,
        left_on="trade_date",
        right_on="ann_date",
        by="ts_code",
        allow_exact_matches=False,  # strictly after the announcement date
    )
    matched = matched.sort_values("_panel_row", ignore_index=True)
    for target in available.values():
        frame[target] = matched[target].to_numpy()
