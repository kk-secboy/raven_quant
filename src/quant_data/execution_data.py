from __future__ import annotations

from calendar import monthrange
from collections.abc import Iterable, Mapping
from datetime import date, datetime, time, timedelta
from typing import Any

from .models import FetchSpec, ProviderResult
from .provider import ProviderError

MARGIN_DATASET = "margin_eligibility"
MINUTE_DATASETS: dict[str, str] = {
    "indices_1m": "idx_mins",
    "etf_1m": "etf_mins",
    "futures_1m": "ft_mins",
    "options_1m": "opt_mins",
    "liquid_stocks_1m": "stk_mins",
}

MARGIN_FIELDS = ("trade_date", "ts_code", "name", "exchange")
MINUTE_FIELDS = ("ts_code", "trade_time", "open", "close", "high", "low", "vol", "amount")
FUTURE_MINUTE_FIELDS = (*MINUTE_FIELDS, "oi")
_TUSHARE_ROW_LIMIT = 8_000
_MARGIN_ROW_LIMIT = 6_000


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


def minute_specs(
    symbols_by_dataset: Mapping[str, Iterable[str]],
    *,
    start: date,
    end: date,
    max_attempts: int,
    freq: str = "1min",
) -> list[FetchSpec]:
    if end < start:
        raise ValueError("end must not be before start")
    if freq not in {"1min", "5min", "15min", "30min", "60min"}:
        raise ValueError("unsupported minute frequency")
    unknown = set(symbols_by_dataset) - set(MINUTE_DATASETS)
    if unknown:
        raise ValueError(f"unsupported minute datasets: {', '.join(sorted(unknown))}")

    specs: list[FetchSpec] = []
    for dataset, raw_symbols in sorted(symbols_by_dataset.items()):
        symbols = sorted({_normalize_symbol(value) for value in raw_symbols if str(value).strip()})
        windows = (
            _fortnight_ranges(start, end)
            if dataset == "futures_1m"
            else _month_ranges(start, end)
        )
        for symbol in symbols:
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
