from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .baostock_provider import BAOSTOCK_SOURCE_VERSION, BaoStockProvider, baostock_code
from .checkpoint import CheckpointStore
from .models import FetchSpec
from .storage import ParquetStore

LEGACY_MARKET_DATASETS = {"trade_cal", "daily", "daily_basic", "adj_factor"}
DEFAULT_OVERLAP_SYMBOLS = (
    "000001.SZ",
    "000002.SZ",
    "000333.SZ",
    "000651.SZ",
    "600000.SH",
    "600030.SH",
    "600036.SH",
    "600276.SH",
    "600519.SH",
    "601318.SH",
)


def baostock_reference_specs(
    start: date,
    end: date,
    *,
    max_attempts: int,
) -> list[FetchSpec]:
    return [
        FetchSpec(
            dataset="trade_cal",
            api_name="baostock_trade_cal",
            scope={
                "source": BAOSTOCK_SOURCE_VERSION,
                "exchange": "SSE",
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            params={"start_date": start.isoformat(), "end_date": end.isoformat()},
            allow_empty=False,
            max_attempts=max_attempts,
        ),
        FetchSpec(
            dataset="baostock_stock_basic",
            api_name="baostock_stock_basic",
            scope={"source": BAOSTOCK_SOURCE_VERSION, "as_of": end.isoformat()},
            params={},
            allow_empty=False,
            max_attempts=max_attempts,
        ),
    ]


def a_share_baostock_codes(
    frame: pd.DataFrame,
    *,
    start: date,
    end: date,
) -> list[str]:
    """Select all A-share stocks whose listing interval overlaps the range."""

    if frame.empty:
        return []
    required = {"code", "ipo_date", "out_date", "security_type"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"BaoStock stock master lacks columns: {sorted(missing)}")
    candidates = frame[frame["security_type"].astype("string") == "1"].copy()
    candidates = candidates[
        candidates["code"].astype("string").str.match(r"^(sh\.6|sz\.[03])\d{5}$", na=False)
    ]
    listed = pd.to_datetime(candidates["ipo_date"], errors="coerce")
    delisted = pd.to_datetime(candidates["out_date"], errors="coerce")
    overlaps = (listed.isna() | (listed <= pd.Timestamp(end))) & (
        delisted.isna() | (delisted >= pd.Timestamp(start))
    )
    return sorted(set(candidates.loc[overlaps, "code"].astype(str)))


def baostock_history_specs(
    codes: Iterable[str],
    start: date,
    end: date,
    *,
    max_attempts: int,
) -> list[FetchSpec]:
    specs: list[FetchSpec] = []
    for code in sorted(set(codes)):
        params = {
            "code": code,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }
        base_scope = {
            "source": BAOSTOCK_SOURCE_VERSION,
            "code": code,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        for dataset, api_name in (
            ("daily", "baostock_daily"),
            ("daily_basic", "baostock_daily_basic"),
            ("adj_factor", "baostock_adj_factor"),
        ):
            specs.append(
                FetchSpec(
                    dataset=dataset,
                    api_name=api_name,
                    scope={**base_scope, "contract": dataset},
                    params=params,
                    allow_empty=True,
                    max_attempts=max_attempts,
                )
            )
    return specs


def planned_baostock_universe(
    checkpoint: CheckpointStore,
    storage: ParquetStore,
    *,
    start: date,
    end: date,
) -> list[str]:
    rows = checkpoint.successful("baostock_stock_basic")
    frame = storage.read_units(rows)
    return a_share_baostock_codes(frame, start=start, end=end)


def _period_units(
    checkpoint: CheckpointStore,
    dataset: str,
    *,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    selected = []
    start_text = start.strftime("%Y%m%d")
    end_text = end.strftime("%Y%m%d")
    for row in checkpoint.successful(dataset):
        scope = dict(row.get("scope_json") or {})
        trade_date = str(scope.get("trade_date") or "")
        if start_text <= trade_date <= end_text:
            selected.append(row)
    return selected


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["trade_date"] = pd.to_datetime(
        normalized["trade_date"], errors="coerce"
    ).dt.strftime("%Y%m%d")
    return normalized.dropna(subset=["ts_code", "trade_date"])


def _relative_error(left: pd.Series, right: pd.Series) -> pd.Series:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    denominator = np.maximum(np.maximum(left_numeric.abs(), right_numeric.abs()), 1e-9)
    return (left_numeric - right_numeric).abs() / denominator


def compare_baostock_overlap(
    tushare_daily: pd.DataFrame,
    baostock_daily: pd.DataFrame,
    tushare_adj: pd.DataFrame,
    baostock_adj: pd.DataFrame,
    *,
    symbols: Iterable[str],
    min_days_per_symbol: int = 180,
) -> dict[str, Any]:
    """Compare source contracts on an overlap year before admitting legacy data."""

    left_daily = _normalize_dates(tushare_daily)
    right_daily = _normalize_dates(baostock_daily)
    left_adj = _normalize_dates(tushare_adj)
    right_adj = _normalize_dates(baostock_adj)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    price_columns = ("open", "high", "low", "close", "pre_close")

    for symbol in sorted(set(symbols)):
        daily = left_daily[left_daily["ts_code"] == symbol].merge(
            right_daily[right_daily["ts_code"] == symbol],
            on=["ts_code", "trade_date"],
            suffixes=("_tushare", "_baostock"),
        )
        item: dict[str, Any] = {"symbol": symbol, "common_days": len(daily)}
        if len(daily) < min_days_per_symbol:
            errors.append(
                f"{symbol} has only {len(daily)} common daily rows; "
                f"requires {min_days_per_symbol}"
            )
            checks.append(item)
            continue

        price_errors = pd.concat(
            [
                (
                    pd.to_numeric(daily[f"{column}_tushare"], errors="coerce")
                    - pd.to_numeric(daily[f"{column}_baostock"], errors="coerce")
                ).abs()
                for column in price_columns
            ],
            ignore_index=True,
        ).dropna()
        pct_error = (
            pd.to_numeric(daily["pct_chg_tushare"], errors="coerce")
            - pd.to_numeric(daily["pct_chg_baostock"], errors="coerce")
        ).abs()
        volume_error = _relative_error(
            daily["vol_tushare"], daily["vol_baostock"]
        ).dropna()
        amount_error = _relative_error(
            daily["amount_tushare"], daily["amount_baostock"]
        ).dropna()
        item.update(
            {
                "price_abs_error_p99": float(price_errors.quantile(0.99)),
                "price_abs_error_max": float(price_errors.max()),
                "pct_chg_abs_error_p99": float(pct_error.quantile(0.99)),
                "volume_relative_error_p99": float(volume_error.quantile(0.99)),
                "amount_relative_error_p99": float(amount_error.quantile(0.99)),
            }
        )

        adj = left_adj[left_adj["ts_code"] == symbol].merge(
            right_adj[right_adj["ts_code"] == symbol],
            on=["ts_code", "trade_date"],
            suffixes=("_tushare", "_baostock"),
        )
        item["common_adj_days"] = len(adj)
        if len(adj) >= min_days_per_symbol:
            tushare_factor = pd.to_numeric(
                adj["adj_factor_tushare"], errors="coerce"
            )
            baostock_factor = pd.to_numeric(
                adj["adj_factor_baostock"], errors="coerce"
            )
            valid = tushare_factor.notna() & baostock_factor.notna()
            tushare_factor = tushare_factor[valid]
            baostock_factor = baostock_factor[valid]
            if not tushare_factor.empty:
                normalized_tushare = tushare_factor / tushare_factor.iloc[0]
                normalized_baostock = baostock_factor / baostock_factor.iloc[0]
                factor_error = _relative_error(
                    normalized_tushare, normalized_baostock
                ).dropna()
                item["adj_factor_relative_error_p99"] = float(
                    factor_error.quantile(0.99)
                )
                item["adj_factor_relative_error_max"] = float(factor_error.max())

        thresholds = {
            "price_abs_error_p99": 0.011,
            "price_abs_error_max": 0.051,
            "pct_chg_abs_error_p99": 0.021,
            "volume_relative_error_p99": 0.005,
            "amount_relative_error_p99": 0.01,
            "adj_factor_relative_error_p99": 0.005,
            "adj_factor_relative_error_max": 0.02,
        }
        for metric, threshold in thresholds.items():
            value = item.get(metric)
            if value is None:
                errors.append(f"{symbol} lacks overlap metric {metric}")
            elif float(value) > threshold:
                errors.append(
                    f"{symbol} {metric}={float(value):.8f} exceeds {threshold:.8f}"
                )
        checks.append(item)

    return {
        "ok": not errors,
        "symbols": sorted(set(symbols)),
        "checks": checks,
        "errors": errors,
        "threshold_policy": {
            "price_abs_error_p99": 0.011,
            "price_abs_error_max": 0.051,
            "pct_chg_abs_error_p99": 0.021,
            "volume_relative_error_p99": 0.005,
            "amount_relative_error_p99": 0.01,
            "adj_factor_relative_error_p99": 0.005,
            "adj_factor_relative_error_max": 0.02,
            "min_days_per_symbol": min_days_per_symbol,
        },
    }


def validate_baostock_overlap(
    checkpoint: CheckpointStore,
    storage: ParquetStore,
    provider: BaoStockProvider,
    *,
    start: date,
    end: date,
    symbols: Iterable[str] = DEFAULT_OVERLAP_SYMBOLS,
    min_days_per_symbol: int = 180,
) -> dict[str, Any]:
    selected_symbols = sorted(set(symbols))
    tushare_daily = storage.read_units(
        _period_units(checkpoint, "daily", start=start, end=end)
    )
    tushare_adj = storage.read_units(
        _period_units(checkpoint, "adj_factor", start=start, end=end)
    )
    bao_daily_frames = []
    bao_adj_frames = []
    for symbol in selected_symbols:
        params = {
            "code": baostock_code(symbol),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }
        bao_daily_frames.append(pd.DataFrame(provider.fetch("baostock_daily", params).rows))
        bao_adj_frames.append(pd.DataFrame(provider.fetch("baostock_adj_factor", params).rows))
    baostock_daily = pd.concat(bao_daily_frames, ignore_index=True)
    baostock_adj = pd.concat(bao_adj_frames, ignore_index=True)
    return {
        "source": BAOSTOCK_SOURCE_VERSION,
        "reference_source": "configured-tushare-gateway",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        **compare_baostock_overlap(
            tushare_daily,
            baostock_daily,
            tushare_adj,
            baostock_adj,
            symbols=selected_symbols,
            min_days_per_symbol=min_days_per_symbol,
        ),
    }
