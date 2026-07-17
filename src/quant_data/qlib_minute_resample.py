from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .execution_contract import QLIB_MINUTE_RESAMPLE_CONTRACT_VERSION

__all__ = [
    "QLIB_MINUTE_RESAMPLE_CONTRACT_VERSION",
    "qlib_resample_index",
    "resample_minute_frame",
    "resample_staging_directory",
    "validate_resample_frequencies",
]

_SUPPORTED_SOURCE_FREQUENCIES = frozenset({"1min", "5min"})
_SUPPORTED_TARGET_FREQUENCIES = frozenset({"15min", "30min", "60min"})


def validate_resample_frequencies(source_frequency: str, target_frequency: str) -> None:
    source = str(source_frequency).lower()
    target = str(target_frequency).lower()
    if source not in _SUPPORTED_SOURCE_FREQUENCIES:
        raise ValueError("Qlib minute resampling requires a native 1/5-minute source")
    if target not in _SUPPORTED_TARGET_FREQUENCIES:
        raise ValueError("Qlib minute resampling supports only 15/30/60-minute targets")
    source_minutes = int(source.removesuffix("min"))
    target_minutes = int(target.removesuffix("min"))
    if target_minutes % source_minutes:
        raise ValueError("target frequency must be an integer multiple of the source")


def qlib_resample_index(
    index: pd.DatetimeIndex,
    source_frequency: str,
    target_frequency: str,
) -> pd.DatetimeIndex:
    """Use Qlib's trading-session calendar to define the target bar timestamps."""

    validate_resample_frequencies(source_frequency, target_frequency)
    try:
        from qlib.utils.resam import resam_calendar
    except ImportError as exc:  # pragma: no cover - asserted in the pinned Qlib runtime
        raise RuntimeError("the configured Qlib runtime is unavailable") from exc
    values = resam_calendar(
        np.asarray([pd.Timestamp(value) for value in index], dtype=object),
        source_frequency,
        target_frequency,
        region="cn",
    )
    return pd.DatetimeIndex(values)


def resample_minute_frame(
    frame: pd.DataFrame,
    *,
    source_frequency: str,
    target_frequency: str,
    calendar_resampler: Callable[
        [pd.DatetimeIndex, str, str], pd.DatetimeIndex
    ] = qlib_resample_index,
) -> pd.DataFrame:
    """Aggregate one normalized instrument with Qlib's A-share target calendar."""

    validate_resample_frequencies(source_frequency, target_frequency)
    required = {
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "factor",
        "amount",
        "paused",
        "up_limit",
        "down_limit",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("minute resample input is missing fields: " + ", ".join(missing))
    values = frame.copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce")
    if values["date"].isna().any() or values["date"].duplicated().any():
        raise ValueError("minute resample input timestamps are invalid or duplicated")
    symbols = values["symbol"].dropna().astype(str).unique()
    if len(symbols) != 1:
        raise ValueError("minute resample input must contain exactly one instrument")
    values.sort_values("date", inplace=True)
    values.set_index("date", inplace=True)
    target_index = calendar_resampler(
        pd.DatetimeIndex(values.index),
        source_frequency,
        target_frequency,
    )
    if target_index.empty:
        raise ValueError("Qlib minute resampling produced an empty target calendar")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "factor",
        "amount",
        "paused",
        "up_limit",
        "down_limit",
        "oi",
    ]
    for column in numeric_columns:
        if column in values:
            values[column] = pd.to_numeric(values[column], errors="coerce")
    values["_vwap_notional"] = values["vwap"] * values["volume"].clip(lower=0).fillna(0)
    values["_source_bars"] = 1
    aggregation: dict[str, Any] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "factor": "last",
        "amount": "sum",
        "up_limit": "last",
        "down_limit": "last",
        "_vwap_notional": "sum",
        "_source_bars": "sum",
    }
    if "oi" in values:
        aggregation["oi"] = "last"
    sampled = (
        values.resample(target_frequency, label="left", closed="left")
        .agg(aggregation)
        .reindex(target_index)
    )
    # Closed-bar guard: emit a target bar only when the source bars fully cover
    # its window.  Intraday snapshots otherwise leak unclosed tail bars whose
    # open/high/low/close/volume are indistinguishable from complete bars.
    expected_source_bars = int(target_frequency.removesuffix("min")) // int(
        source_frequency.removesuffix("min")
    )
    sampled = sampled[sampled["_source_bars"] == expected_source_bars]
    sampled = sampled.dropna(subset=["open", "high", "low", "close"])
    if sampled.empty:
        raise ValueError("Qlib minute resampling produced no complete bars")
    positive_volume = sampled["volume"] > 0
    sampled["vwap"] = sampled["close"]
    sampled.loc[positive_volume, "vwap"] = (
        sampled.loc[positive_volume, "_vwap_notional"]
        / sampled.loc[positive_volume, "volume"]
    )
    sampled["paused"] = (~positive_volume).astype(float)
    sampled["change"] = sampled["close"].pct_change(fill_method=None)
    sampled["symbol"] = symbols[0]
    sampled.drop(columns=["_vwap_notional", "_source_bars"], inplace=True)
    sampled.reset_index(names="date", inplace=True)
    ordered = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "factor",
        "change",
        "amount",
        "paused",
        "up_limit",
        "down_limit",
    ]
    if "oi" in sampled:
        ordered.append("oi")
    return sampled.loc[:, ordered]


def resample_staging_directory(
    source: Path,
    output: Path,
    *,
    source_frequency: str,
    target_frequency: str,
) -> Path:
    validate_resample_frequencies(source_frequency, target_frequency)
    source = source.resolve()
    output = output.resolve()
    files = sorted(source.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("native minute staging contains no instrument Parquet files")
    output.mkdir(parents=True, exist_ok=False)
    for path in files:
        result = resample_minute_frame(
            pd.read_parquet(path),
            source_frequency=source_frequency,
            target_frequency=target_frequency,
        )
        result.to_parquet(output / path.name, index=False, compression="zstd")
    return output
