from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from .catalog import INDEX_CODES

ETF_UNIVERSES: dict[str, tuple[str, ...]] = {
    "broad": ("510050.SH", "510300.SH", "510500.SH", "512100.SH", "159915.SZ"),
    "industry": ("512010.SH", "512170.SH", "512660.SH", "512690.SH", "512880.SH"),
    "gold": ("518880.SH", "159934.SZ"),
    "bond": ("511010.SH", "511260.SH", "511360.SH"),
}

FUTURE_ROOTS = ("IF", "IC", "IM", "IH")


@dataclass(frozen=True, slots=True)
class UniverseSelection:
    symbols_by_dataset: dict[str, list[str]]
    evidence: dict[str, Any]


def select_intraday_universe(
    frames: Mapping[str, pd.DataFrame],
    *,
    max_stocks: int = 100,
    max_options: int = 100,
    etf_categories: tuple[str, ...] = ("broad", "industry", "gold", "bond"),
    start: date | None = None,
    end: date | None = None,
) -> UniverseSelection:
    if not 0 <= max_stocks <= 500:
        raise ValueError("max_stocks must be between 0 and 500")
    if not 0 <= max_options <= 500:
        raise ValueError("max_options must be between 0 and 500")
    unknown_categories = set(etf_categories) - set(ETF_UNIVERSES)
    if unknown_categories:
        raise ValueError(
            "unknown ETF categories: " + ", ".join(sorted(unknown_categories))
        )

    etfs = sorted(
        {
            code
            for category in etf_categories
            for code in ETF_UNIVERSES[category]
        }
    )
    stocks, stock_date = _liquid_stocks(
        frames.get("daily", pd.DataFrame()),
        frames.get("stock_basic", pd.DataFrame()),
        max_stocks,
    )
    futures, futures_date = _active_futures(
        frames.get("fut_mapping", pd.DataFrame()), start=start, end=end
    )
    options, options_date = _active_options(
        frames.get("opt_daily", pd.DataFrame()), max_options
    )

    symbols = {
        "indices_1m": sorted(INDEX_CODES),
        "etf_1m": etfs,
    }
    if stocks:
        symbols["liquid_stocks_1m"] = stocks
    if futures:
        symbols["futures_1m"] = futures
    if options:
        symbols["options_1m"] = options
    return UniverseSelection(
        symbols_by_dataset=symbols,
        evidence={
            "rules": {
                "indices": "fixed major A-share indices",
                "etfs": list(etf_categories),
                "stocks": "highest mean amount over latest 20 available trading days",
                "futures": "latest IF/IC/IM/IH continuous-to-deliverable mappings",
                "options": "highest amount/volume active SSE/SZSE option contracts",
            },
            "source_dates": {
                "stocks": stock_date,
                "futures": futures_date,
                "options": options_date,
            },
            "counts": {key: len(value) for key, value in symbols.items()},
        },
    )


def select_intraday_universe_from_store(
    checkpoint: Any,
    storage: Any,
    *,
    max_stocks: int,
    max_options: int,
    etf_categories: tuple[str, ...],
    start: date | None = None,
    end: date | None = None,
) -> UniverseSelection:
    frames = {
        dataset: storage.read_units(checkpoint.successful(dataset))
        for dataset in ("daily", "stock_basic", "fut_mapping", "opt_daily")
    }
    return select_intraday_universe(
        frames,
        max_stocks=max_stocks,
        max_options=max_options,
        etf_categories=etf_categories,
        start=start,
        end=end,
    )


def _latest_date(frame: pd.DataFrame, column: str) -> tuple[pd.DataFrame, str | None]:
    if frame.empty or column not in frame.columns:
        return frame.iloc[0:0].copy(), None
    dates = pd.to_datetime(frame[column], errors="coerce")
    latest = dates.max()
    if pd.isna(latest):
        return frame.iloc[0:0].copy(), None
    selected = frame.loc[dates == latest].copy()
    return selected, pd.Timestamp(latest).date().isoformat()


def _liquid_stocks(
    daily: pd.DataFrame, stock_basic: pd.DataFrame, limit: int
) -> tuple[list[str], str | None]:
    if limit == 0 or daily.empty or not {"ts_code", "trade_date", "amount"} <= set(daily):
        return [], None
    work = daily[["ts_code", "trade_date", "amount"]].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    dates = sorted(work["trade_date"].dropna().unique())[-20:]
    work = work.loc[work["trade_date"].isin(dates) & work["amount"].gt(0)]
    if not stock_basic.empty and {"ts_code", "name"} <= set(stock_basic):
        names = stock_basic[["ts_code", "name"]].drop_duplicates("ts_code", keep="last")
        work = work.merge(names, on="ts_code", how="left")
        work = work.loc[~work["name"].fillna("").astype(str).str.upper().str.contains("ST")]
    ranked = work.groupby("ts_code", as_index=False)["amount"].mean()
    ranked = ranked.sort_values(["amount", "ts_code"], ascending=[False, True])
    latest = max(dates).date().isoformat() if dates else None
    return ranked["ts_code"].astype(str).head(limit).tolist(), latest


def _active_futures(
    mapping: pd.DataFrame, *, start: date | None, end: date | None
) -> tuple[list[str], str | None]:
    if mapping.empty or not {"ts_code", "mapping_ts_code", "trade_date"} <= set(mapping):
        return [], None
    current = mapping.copy()
    dates = pd.to_datetime(current["trade_date"], errors="coerce")
    if start is not None:
        current = current.loc[dates.dt.date >= start]
        dates = pd.to_datetime(current["trade_date"], errors="coerce")
    if end is not None:
        current = current.loc[dates.dt.date <= end]
    latest_stamp = pd.to_datetime(current["trade_date"], errors="coerce").max()
    latest = None if pd.isna(latest_stamp) else pd.Timestamp(latest_stamp).date().isoformat()
    if current.empty:
        return [], latest
    selected: set[str] = set()
    for root in FUTURE_ROOTS:
        rows = current.loc[
            current["ts_code"].fillna("").astype(str).str.upper().str.startswith(root)
        ]
        if rows.empty:
            continue
        selected.update(rows["ts_code"].dropna().astype(str).str.strip().str.upper())
        selected.update(
            rows["mapping_ts_code"].dropna().astype(str).str.strip().str.upper()
        )
    return sorted(value for value in selected if value), latest


def _active_options(options: pd.DataFrame, limit: int) -> tuple[list[str], str | None]:
    current, latest = _latest_date(options, "trade_date")
    if limit == 0 or current.empty or "ts_code" not in current.columns:
        return [], latest
    current = current.loc[
        current["ts_code"].fillna("").astype(str).str.upper().str.endswith((".SH", ".SZ"))
    ].copy()
    score_columns = [column for column in ("amount", "vol") if column in current.columns]
    if not score_columns:
        return sorted(current["ts_code"].astype(str).unique())[:limit], latest
    for column in score_columns:
        current[column] = pd.to_numeric(current[column], errors="coerce").fillna(0)
    current = current.sort_values(score_columns + ["ts_code"], ascending=False)
    return current["ts_code"].astype(str).drop_duplicates().head(limit).tolist(), latest
