from __future__ import annotations

import json
import threading
from collections import OrderedDict
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

import baostock as bs

from .models import ProviderResult
from .provider import ProviderError

BAOSTOCK_SOURCE_VERSION = "baostock-0.9.3"
BAOSTOCK_HISTORY_CACHE_SIZE = 8
BAOSTOCK_DAILY_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,turn,"
    "tradestatus,pctChg,peTTM,psTTM,pbMRQ,isST"
)


def tushare_code(value: str) -> str:
    """Convert BaoStock's ``sh.600000`` notation into ``600000.SH``."""

    market, symbol = value.strip().lower().split(".", 1)
    if market not in {"sh", "sz", "bj"} or len(symbol) != 6 or not symbol.isdigit():
        raise ValueError(f"unsupported BaoStock security code: {value!r}")
    return f"{symbol}.{market.upper()}"


def baostock_code(value: str) -> str:
    """Convert ``600000.SH`` into BaoStock's ``sh.600000`` notation."""

    symbol, market = value.strip().upper().split(".", 1)
    if market not in {"SH", "SZ", "BJ"} or len(symbol) != 6 or not symbol.isdigit():
        raise ValueError(f"unsupported Tushare security code: {value!r}")
    return f"{market.lower()}.{symbol}"


def _decimal(value: object) -> Decimal | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ProviderError(f"BaoStock returned invalid numeric value {text!r}") from exc


def _float(value: object) -> float | None:
    parsed = _decimal(value)
    return float(parsed) if parsed is not None else None


def _scaled(value: object, divisor: int) -> float | None:
    parsed = _decimal(value)
    return float(parsed / Decimal(divisor)) if parsed is not None else None


def _raw_body(api_name: str, fields: list[str], rows: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        {
            "source": BAOSTOCK_SOURCE_VERSION,
            "api_name": api_name,
            "fields": fields,
            "rows": rows,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


class BaoStockProvider:
    """Small, serialized BaoStock adapter for audited pre-2016 market history.

    BaoStock's Python client owns a process-global socket, so requests must not
    run concurrently.  The adapter keeps one authenticated session and caches
    each symbol/range response so ``daily`` and ``daily_basic`` do not trigger
    duplicate upstream downloads.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history_cache: OrderedDict[
            tuple[str, str, str, str], tuple[list[str], list[dict[str, str]]]
        ] = OrderedDict()
        with self._lock:
            result = bs.login()
        if str(result.error_code) != "0":
            raise ProviderError(
                f"BaoStock login failed code={result.error_code}: {result.error_msg}",
                retryable=True,
            )
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            bs.logout()
        self._closed = True

    def __enter__(self) -> BaoStockProvider:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def fetch(
        self, api_name: str, params: dict[str, object], fields: tuple[str, ...] = ()
    ) -> ProviderResult:
        if self._closed:
            raise RuntimeError("BaoStock provider is closed")
        if api_name == "baostock_trade_cal":
            return self._trade_calendar(params)
        if api_name == "baostock_stock_basic":
            return self._stock_basic()
        if api_name in {
            "baostock_daily",
            "baostock_daily_basic",
            "baostock_adj_factor",
        }:
            return self._history(api_name, params)
        raise ProviderError(f"unsupported BaoStock API: {api_name}", retryable=False)

    def _query_rows(self, result: Any, api_name: str) -> tuple[list[str], list[dict[str, str]]]:
        if str(result.error_code) != "0":
            raise ProviderError(
                f"BaoStock {api_name} failed code={result.error_code}: {result.error_msg}",
                retryable=True,
            )
        columns = [str(field) for field in result.fields]
        rows: list[dict[str, str]] = []
        while result.next():
            rows.append(dict(zip(columns, result.get_row_data(), strict=True)))
        if str(result.error_code) != "0":
            raise ProviderError(
                f"BaoStock {api_name} iteration failed "
                f"code={result.error_code}: {result.error_msg}",
                retryable=True,
            )
        return columns, rows

    def _trade_calendar(self, params: dict[str, object]) -> ProviderResult:
        target_start = date.fromisoformat(str(params["start_date"]))
        target_end = date.fromisoformat(str(params["end_date"]))
        query_start = target_start - timedelta(days=40)
        with self._lock:
            columns, source_rows = self._query_rows(
                bs.query_trade_dates(
                    start_date=query_start.isoformat(),
                    end_date=target_end.isoformat(),
                ),
                "query_trade_dates",
            )
        previous_open = ""
        rows: list[dict[str, Any]] = []
        for source in source_rows:
            calendar_date = date.fromisoformat(source["calendar_date"])
            is_open = str(source["is_trading_day"]) == "1"
            if target_start <= calendar_date <= target_end:
                rows.append(
                    {
                        "exchange": "SSE",
                        "cal_date": calendar_date.strftime("%Y%m%d"),
                        "is_open": "1" if is_open else "0",
                        "pretrade_date": previous_open,
                    }
                )
            if is_open:
                previous_open = calendar_date.strftime("%Y%m%d")
        return ProviderResult(
            "baostock_trade_cal",
            ["exchange", "cal_date", "is_open", "pretrade_date"],
            rows,
            _raw_body("query_trade_dates", columns, source_rows),
            {
                "source": BAOSTOCK_SOURCE_VERSION,
                "target_start": target_start.isoformat(),
                "target_end": target_end.isoformat(),
            },
        )

    def _stock_basic(self) -> ProviderResult:
        with self._lock:
            columns, source_rows = self._query_rows(
                bs.query_stock_basic(),
                "query_stock_basic",
            )
        rows = [
            {
                "code": str(row.get("code") or ""),
                "code_name": str(row.get("code_name") or ""),
                "ipo_date": str(row.get("ipoDate") or ""),
                "out_date": str(row.get("outDate") or ""),
                "security_type": str(row.get("type") or ""),
                "status": str(row.get("status") or ""),
            }
            for row in source_rows
        ]
        return ProviderResult(
            "baostock_stock_basic",
            ["code", "code_name", "ipo_date", "out_date", "security_type", "status"],
            rows,
            _raw_body("query_stock_basic", columns, source_rows),
            {"source": BAOSTOCK_SOURCE_VERSION},
        )

    def _history_rows(
        self,
        code: str,
        start_date: str,
        end_date: str,
        adjustflag: str,
    ) -> tuple[list[str], list[dict[str, str]]]:
        key = (code, start_date, end_date, adjustflag)
        cached = self._history_cache.get(key)
        if cached is not None:
            self._history_cache.move_to_end(key)
            return cached
        with self._lock:
            result = bs.query_history_k_data_plus(
                code,
                BAOSTOCK_DAILY_FIELDS,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=adjustflag,
            )
            fetched = self._query_rows(result, "query_history_k_data_plus")
        self._history_cache[key] = fetched
        self._history_cache.move_to_end(key)
        while len(self._history_cache) > BAOSTOCK_HISTORY_CACHE_SIZE:
            self._history_cache.popitem(last=False)
        return fetched

    def _history(self, api_name: str, params: dict[str, object]) -> ProviderResult:
        code = str(params["code"])
        start_date = str(params["start_date"])
        end_date = str(params["end_date"])
        columns, raw_rows = self._history_rows(code, start_date, end_date, "3")
        traded = [row for row in raw_rows if str(row.get("tradestatus")) == "1"]
        ts_code = tushare_code(code)

        if api_name == "baostock_daily":
            rows = []
            for row in traded:
                close = _float(row.get("close"))
                pre_close = _float(row.get("preclose"))
                rows.append(
                    {
                        "ts_code": ts_code,
                        "trade_date": str(row["date"]).replace("-", ""),
                        "open": _float(row.get("open")),
                        "high": _float(row.get("high")),
                        "low": _float(row.get("low")),
                        "close": close,
                        "pre_close": pre_close,
                        "change": (
                            close - pre_close
                            if close is not None and pre_close is not None
                            else None
                        ),
                        "pct_chg": _float(row.get("pctChg")),
                        # BaoStock publishes shares and yuan.  Tushare's daily
                        # contract uses 100-share lots and thousand yuan.
                        "vol": _scaled(row.get("volume"), 100),
                        "amount": _scaled(row.get("amount"), 1_000),
                    }
                )
            output_columns = [
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "change",
                "pct_chg",
                "vol",
                "amount",
            ]
        elif api_name == "baostock_daily_basic":
            rows = [
                {
                    "ts_code": ts_code,
                    "trade_date": str(row["date"]).replace("-", ""),
                    "close": _float(row.get("close")),
                    "turnover_rate": _float(row.get("turn")),
                    "turnover_rate_f": None,
                    "volume_ratio": None,
                    "pe": None,
                    "pe_ttm": _float(row.get("peTTM")),
                    "pb": _float(row.get("pbMRQ")),
                    "ps": None,
                    "ps_ttm": _float(row.get("psTTM")),
                    "dv_ratio": None,
                    "dv_ttm": None,
                    "total_share": None,
                    "float_share": None,
                    "free_share": None,
                    "total_mv": None,
                    "circ_mv": None,
                }
                for row in traded
            ]
            output_columns = [
                "ts_code",
                "trade_date",
                "close",
                "turnover_rate",
                "turnover_rate_f",
                "volume_ratio",
                "pe",
                "pe_ttm",
                "pb",
                "ps",
                "ps_ttm",
                "dv_ratio",
                "dv_ttm",
                "total_share",
                "float_share",
                "free_share",
                "total_mv",
                "circ_mv",
            ]
        else:
            adjusted_columns, adjusted_rows = self._history_rows(
                code, start_date, end_date, "1"
            )
            adjusted_by_date = {
                str(row["date"]): row
                for row in adjusted_rows
                if str(row.get("tradestatus")) == "1"
            }
            rows = []
            for row in traded:
                adjusted = adjusted_by_date.get(str(row["date"]))
                raw_close = _decimal(row.get("close"))
                adjusted_close = _decimal((adjusted or {}).get("close"))
                if raw_close in {None, Decimal(0)} or adjusted_close is None:
                    continue
                rows.append(
                    {
                        "ts_code": ts_code,
                        "trade_date": str(row["date"]).replace("-", ""),
                        "adj_factor": float(adjusted_close / raw_close),
                    }
                )
            output_columns = ["ts_code", "trade_date", "adj_factor"]
            raw_rows = [
                {"unadjusted": row, "back_adjusted": adjusted_by_date.get(str(row["date"]))}
                for row in raw_rows
            ]
            columns = [*columns, *[f"back_adjusted.{item}" for item in adjusted_columns]]

        return ProviderResult(
            api_name,
            output_columns,
            rows,
            _raw_body("query_history_k_data_plus", columns, raw_rows),
            {
                "source": BAOSTOCK_SOURCE_VERSION,
                "code": code,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
