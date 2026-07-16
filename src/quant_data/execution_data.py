from __future__ import annotations

from calendar import monthrange
from collections.abc import Iterable, Mapping
from datetime import date, datetime, time, timedelta
from typing import Any

from .models import FetchSpec, ProviderResult
from .partitioning import partition_metadata
from .provider import ProviderError

MARGIN_DATASET = "margin_eligibility"
NEWS_DATASET = "news"
NEWS_SOURCES = (
    "sina",
    "wallstreetcn",
    "10jqka",
    "eastmoney",
    "yuncaijing",
    "fenghuang",
    "jinrongjie",
    "cls",
    "yicai",
)
NEWS_FIELDS = ("datetime", "content", "title", "channels")
NEWS_WINDOW_HOURS = 24
MINUTE_DATASETS: dict[str, str] = {
    "ashare_5m": "stk_mins",
    "indices_1m": "idx_mins",
    "etf_1m": "etf_mins",
    "futures_1m": "ft_mins",
    "options_1m": "opt_mins",
    "liquid_stocks_1m": "stk_mins",
}
MINUTE_FREQUENCIES = frozenset({"1min", "5min", "15min", "30min", "60min"})

MARGIN_FIELDS = ("trade_date", "ts_code", "name", "exchange")
MINUTE_FIELDS = ("ts_code", "trade_time", "open", "close", "high", "low", "vol", "amount")
FUTURE_MINUTE_FIELDS = (*MINUTE_FIELDS, "oi")
_TUSHARE_ROW_LIMIT = 8_000
_MARGIN_ROW_LIMIT = 6_000
_NEWS_ROW_LIMIT = 1_500


def margin_specs(trading_dates: Iterable[str], *, max_attempts: int) -> list[FetchSpec]:
    dates = sorted(set(trading_dates))
    return [
        FetchSpec(
            dataset=MARGIN_DATASET,
            api_name="margin_secs",
            scope={"trade_date": trading_date},
            params={"trade_date": trading_date},
            fields=MARGIN_FIELDS,
            allow_empty=False,
            max_attempts=max_attempts,
        )
        for trading_date in dates
    ]


def news_specs(
    start: date,
    end: date,
    *,
    max_attempts: int,
    sources: Iterable[str] = NEWS_SOURCES,
    window_hours: int = NEWS_WINDOW_HOURS,
) -> list[FetchSpec]:
    """Plan one source/day request; capped windows are split after execution."""

    if end < start:
        raise ValueError("end must not be before start")
    if not 1 <= window_hours <= 24 or 24 % window_hours != 0:
        raise ValueError("news window hours must divide one day")
    normalized_sources = sorted({str(value).strip() for value in sources if str(value).strip()})
    unknown = set(normalized_sources) - set(NEWS_SOURCES)
    if unknown:
        raise ValueError(f"unsupported news sources: {', '.join(sorted(unknown))}")
    if not normalized_sources:
        raise ValueError("at least one news source is required")

    specs: list[FetchSpec] = []
    cursor = start
    while cursor <= end:
        day_start = datetime.combine(cursor, time.min)
        day_end = datetime.combine(cursor, time.max.replace(microsecond=0))
        for source in normalized_sources:
            window_start = day_start
            while window_start <= day_end:
                window_end = min(
                    window_start + timedelta(hours=window_hours) - timedelta(seconds=1),
                    day_end,
                )
                specs.append(
                    news_window_spec(
                        source,
                        window_start,
                        window_end,
                        max_attempts=max_attempts,
                    )
                )
                window_start = window_end + timedelta(seconds=1)
        cursor += timedelta(days=1)
    return specs


def news_window_spec(
    source: str,
    window_start: datetime,
    window_end: datetime,
    *,
    max_attempts: int,
) -> FetchSpec:
    """Build one resumable news window, including adaptive partition metadata."""

    start_text = window_start.strftime("%Y-%m-%d %H:%M:%S")
    end_text = window_end.strftime("%Y-%m-%d %H:%M:%S")
    return FetchSpec(
        dataset=NEWS_DATASET,
        api_name="news",
        scope={
            "date": window_start.date().isoformat(),
            "source": source,
            "start": start_text,
            "end": end_text,
            "row_limit": _NEWS_ROW_LIMIT,
            **partition_metadata(
                "datetime",
                window_start,
                window_end,
                value_format="timestamp",
            ),
        },
        params={
            "src": source,
            "start_date": start_text,
            "end_date": end_text,
        },
        fields=NEWS_FIELDS,
        allow_empty=True,
        max_attempts=max_attempts,
    )


def minute_specs(
    symbols_by_dataset: Mapping[str, Iterable[str]],
    *,
    start: date,
    end: date,
    max_attempts: int,
    freq: str = "1min",
    active_ranges_by_dataset: Mapping[
        str, Mapping[str, tuple[date, date]]
    ] | None = None,
    trading_dates: Iterable[str] | None = None,
    windows_by_dataset: Mapping[
        str, Mapping[str, Iterable[tuple[date, date]]]
    ] | None = None,
) -> list[FetchSpec]:
    if end < start:
        raise ValueError("end must not be before start")
    if freq not in MINUTE_FREQUENCIES:
        raise ValueError("unsupported minute frequency")
    unknown = set(symbols_by_dataset) - set(MINUTE_DATASETS)
    if unknown:
        raise ValueError(f"unsupported minute datasets: {', '.join(sorted(unknown))}")

    session_dates = (
        _normalize_trading_dates(trading_dates) if trading_dates is not None else None
    )
    specs: list[FetchSpec] = []
    for dataset, raw_symbols in sorted(symbols_by_dataset.items()):
        symbols = sorted({_normalize_symbol(value) for value in raw_symbols if str(value).strip()})
        raw_active_ranges = (active_ranges_by_dataset or {}).get(dataset, {})
        active_ranges = {
            _normalize_symbol(symbol): bounds for symbol, bounds in raw_active_ranges.items()
        }
        for symbol in symbols:
            symbol_start, symbol_end = active_ranges.get(symbol, (start, end))
            symbol_start = max(start, symbol_start)
            symbol_end = min(end, symbol_end)
            if symbol_end < symbol_start:
                continue
            requested_windows = (windows_by_dataset or {}).get(dataset, {}).get(symbol)
            if requested_windows is not None:
                windows = list(requested_windows)
            elif dataset == "ashare_5m" and freq == "5min" and session_dates is not None:
                windows = _trading_session_ranges(
                    symbol_start,
                    symbol_end,
                    session_dates,
                    max_sessions=150,
                )
            else:
                windows = (
                    _fortnight_ranges(symbol_start, symbol_end)
                    if dataset == "futures_1m"
                    else _month_ranges(symbol_start, symbol_end)
                )
            for window_start, window_end in windows:
                start_time = datetime.combine(window_start, time.min).strftime("%Y-%m-%d %H:%M:%S")
                end_time = datetime.combine(window_end, time.max.replace(microsecond=0)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                specs.append(
                    FetchSpec(
                        dataset=dataset,
                        api_name=MINUTE_DATASETS[dataset],
                        scope={
                            "ts_code": symbol,
                            "start": start_time,
                            "end": end_time,
                            "freq": freq,
                            **partition_metadata(
                                "date",
                                window_start,
                                window_end,
                                value_format="date_timestamp",
                            ),
                        },
                        params={
                            "ts_code": symbol,
                            "start_date": start_time,
                            "end_date": end_time,
                            "freq": freq,
                        },
                        fields=(FUTURE_MINUTE_FIELDS if dataset == "futures_1m" else MINUTE_FIELDS),
                        allow_empty=True,
                        max_attempts=max_attempts,
                    )
                )
    if not specs:
        raise ValueError("at least one minute symbol is required")
    return specs


def validate_and_normalize(spec: FetchSpec, result: ProviderResult) -> ProviderResult:
    if spec.dataset == MARGIN_DATASET:
        return _validate_margin(spec, result)
    if spec.dataset == NEWS_DATASET:
        return _validate_news(spec, result)
    if spec.dataset in MINUTE_DATASETS:
        return _validate_minute(spec, result)
    if (
        spec.scope.get("page_group")
        or spec.scope.get("expected_date_field")
        or spec.scope.get("row_limit") is not None
    ):
        from .supplemental_data import validate_supplemental

        return validate_supplemental(spec, result)
    return result


def _validate_news(spec: FetchSpec, result: ProviderResult) -> ProviderResult:
    _require_columns(spec.dataset, result.columns, set(NEWS_FIELDS))
    if len(result.rows) >= _NEWS_ROW_LIMIT:
        raise ProviderError(
            f"{spec.dataset} returned {len(result.rows)} rows and may be truncated at the "
            f"Tushare {_NEWS_ROW_LIMIT}-row limit; use a smaller source time window",
            retryable=False,
        )
    start = _parse_timestamp(spec.params["start_date"])
    end = _parse_timestamp(spec.params["end_date"])
    source = str(spec.params["src"])
    seen: set[tuple[datetime, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for raw in result.rows:
        row = dict(raw)
        stamp = _parse_timestamp(row.get("datetime"))
        if not start <= stamp <= end:
            raise ProviderError(
                f"{spec.dataset} returned timestamp {stamp.isoformat()} outside requested window",
                retryable=False,
            )
        title = str(row.get("title") or "")
        content = str(row.get("content") or "")
        key = (stamp, title, content)
        if key in seen:
            continue
        seen.add(key)
        row["source"] = source
        rows.append(row)
    columns = [*result.columns]
    if "source" not in columns:
        columns.append("source")
    metadata = {**result.metadata, "source": source}
    return ProviderResult(result.api_name, columns, rows, result.raw_body, metadata)


def _validate_margin(spec: FetchSpec, result: ProviderResult) -> ProviderResult:
    required = set(MARGIN_FIELDS)
    _require_columns(spec.dataset, result.columns, required)
    if len(result.rows) >= _MARGIN_ROW_LIMIT:
        raise ProviderError(
            f"{spec.dataset} returned {len(result.rows)} rows and may be truncated at the "
            f"Tushare {_MARGIN_ROW_LIMIT}-row limit",
            retryable=False,
        )
    expected_date = str(spec.params["trade_date"])
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for raw in result.rows:
        row = dict(raw)
        trade_date = _compact_date(row.get("trade_date"))
        ts_code = str(row.get("ts_code") or "").strip()
        if trade_date != expected_date:
            raise ProviderError(
                f"{spec.dataset} returned trade_date {trade_date!r} outside {expected_date}",
                retryable=False,
            )
        if not ts_code:
            raise ProviderError(f"{spec.dataset} returned a blank ts_code", retryable=False)
        key = (trade_date, ts_code)
        if key in seen:
            raise ProviderError(f"{spec.dataset} returned duplicate key {key}", retryable=False)
        seen.add(key)
        row["trade_date"] = trade_date
        row["ts_code"] = ts_code
        row["shortable"] = True
        rows.append(row)
    columns = [*result.columns]
    if "shortable" not in columns:
        columns.append("shortable")
    return ProviderResult(result.api_name, columns, rows, result.raw_body, dict(result.metadata))


def _validate_minute(spec: FetchSpec, result: ProviderResult) -> ProviderResult:
    _require_columns(spec.dataset, result.columns, set(MINUTE_FIELDS))
    if len(result.rows) >= _TUSHARE_ROW_LIMIT:
        raise ProviderError(
            f"{spec.dataset} returned {len(result.rows)} rows and may be truncated at the "
            f"Tushare {_TUSHARE_ROW_LIMIT}-row limit; use a smaller time window",
            retryable=False,
        )
    start = _parse_timestamp(spec.params["start_date"])
    end = _parse_timestamp(spec.params["end_date"])
    expected_symbol = str(spec.params["ts_code"])
    seen: set[tuple[str, datetime]] = set()
    rows: list[dict[str, Any]] = []
    for raw in result.rows:
        row = dict(raw)
        symbol = str(row.get("ts_code") or "").strip()
        stamp = _parse_timestamp(row.get("trade_time"))
        if symbol != expected_symbol:
            raise ProviderError(
                f"{spec.dataset} returned symbol {symbol!r}, expected {expected_symbol!r}",
                retryable=False,
            )
        if not start <= stamp <= end:
            raise ProviderError(
                f"{spec.dataset} returned timestamp {stamp.isoformat()} outside requested window",
                retryable=False,
            )
        key = (symbol, stamp)
        if key in seen:
            raise ProviderError(f"{spec.dataset} returned duplicate key {key}", retryable=False)
        seen.add(key)
        _validate_bar(spec.dataset, row)
        row["trade_time"] = stamp.isoformat(sep=" ")
        rows.append(row)
    return ProviderResult(
        result.api_name,
        list(result.columns),
        rows,
        result.raw_body,
        dict(result.metadata),
    )


def _validate_bar(dataset: str, row: Mapping[str, Any]) -> None:
    try:
        open_price = float(row["open"])
        close_price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        volume = float(row["vol"])
        amount = float(row["amount"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderError(f"{dataset} returned a non-numeric OHLCV row", retryable=False) from exc
    if min(open_price, close_price, high, low) <= 0:
        raise ProviderError(f"{dataset} returned a non-positive price", retryable=False)
    if high < max(open_price, close_price, low) or low > min(open_price, close_price, high):
        raise ProviderError(f"{dataset} returned an invalid OHLC envelope", retryable=False)
    if volume < 0 or amount < 0:
        raise ProviderError(f"{dataset} returned negative volume or amount", retryable=False)


def _require_columns(dataset: str, columns: Iterable[str], required: set[str]) -> None:
    missing = required - set(columns)
    if missing:
        raise ProviderError(
            f"{dataset} response is missing columns: {', '.join(sorted(missing))}",
            retryable=False,
        )


def _normalize_symbol(value: Any) -> str:
    symbol = str(value).strip().upper()
    if not symbol:
        raise ValueError("minute symbols must not be blank")
    if "." in symbol:
        return symbol
    if len(symbol) > 2 and symbol[:2] in {"SH", "SZ", "BJ"}:
        return f"{symbol[2:]}.{symbol[:2]}"
    raise ValueError(f"symbol must be a Tushare or Qlib code: {symbol}")


def _compact_date(value: Any) -> str:
    raw = str(value or "").strip().replace("-", "")
    if len(raw) >= 8:
        raw = raw[:8]
    return raw


def _parse_timestamp(value: Any) -> datetime:
    try:
        return datetime.fromisoformat(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ProviderError(f"invalid minute timestamp: {value!r}", retryable=False) from exc


def _month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    current = start
    while current <= end:
        last = date(current.year, current.month, monthrange(current.year, current.month)[1])
        window_end = min(last, end)
        ranges.append((current, window_end))
        current = window_end + timedelta(days=1)
    return ranges


def _fortnight_ranges(start: date, end: date) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    current = start
    while current <= end:
        window_end = min(current + timedelta(days=13), end)
        ranges.append((current, window_end))
        current = window_end + timedelta(days=1)
    return ranges


def _trading_session_ranges(
    start: date,
    end: date,
    trading_dates: Iterable[date],
    *,
    max_sessions: int,
) -> list[tuple[date, date]]:
    if max_sessions <= 0:
        raise ValueError("max_sessions must be positive")
    sessions = [value for value in trading_dates if start <= value <= end]
    return [
        (sessions[offset], sessions[min(offset + max_sessions - 1, len(sessions) - 1)])
        for offset in range(0, len(sessions), max_sessions)
    ]


def _normalize_trading_dates(values: Iterable[str]) -> list[date]:
    return sorted(
        {
            datetime.strptime(str(value).replace("-", "")[:8], "%Y%m%d").date()
            for value in values
        }
    )
