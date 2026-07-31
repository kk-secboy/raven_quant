from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from .catalog import (
    CORE_DAILY,
    DISCLOSURE_FIELDS,
    ETF_DAILY,
    INDEX_CATALOG_MARKETS,
    INDEX_CATALOG_PAGE_SIZE,
    INDEX_CODES,
    INDEX_DAILY_BASIC,
    REFERENCE_FIELDS,
    RESEARCH_DAILY,
    DatasetDefinition,
)
from .checkpoint import CheckpointStore
from .execution_data import (
    NEWS_DATASET,
    margin_specs,
    minute_specs,
    news_specs,
    news_window_spec,
)
from .history_bounds import history_start_date
from .models import FetchSpec, canonical_json
from .reference_data import apply_reference_refresh
from .storage import ParquetStore


def compact_date(value: date) -> str:
    return value.strftime("%Y%m%d")


CN_MARKET_TIMEZONE = ZoneInfo("Asia/Shanghai")


def today_cn(now: datetime | None = None) -> date:
    """Return "today" in the A-share market timezone (Asia/Shanghai).

    Naive datetimes are interpreted as UTC.  All pipeline date defaults must
    use the market timezone so an evening UTC run does not plan for the wrong
    trading day.
    """

    current = now if now is not None else datetime.now(CN_MARKET_TIMEZONE)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(CN_MARKET_TIMEZONE).date()


class BootstrapPlanner:
    def __init__(self, checkpoint: CheckpointStore, storage: ParquetStore) -> None:
        self.checkpoint = checkpoint
        self.storage = storage

    def plan_reference(self, start: date, end: date, max_attempts: int) -> int:
        specs: list[FetchSpec] = []
        for status in ("L", "D", "P"):
            specs.append(
                FetchSpec(
                    dataset="stock_basic",
                    api_name="stock_basic",
                    scope={"list_status": status},
                    params={"list_status": status},
                    fields=REFERENCE_FIELDS["stock_basic"],
                    allow_empty=status == "P",
                    max_attempts=max_attempts,
                )
            )
        specs.append(
            FetchSpec(
                dataset="trade_cal",
                api_name="trade_cal",
                scope={"exchange": "SSE", "start": compact_date(start), "end": compact_date(end)},
                params={
                    "exchange": "SSE",
                    "start_date": compact_date(start),
                    "end_date": compact_date(end),
                },
                fields=REFERENCE_FIELDS["trade_cal"],
                max_attempts=max_attempts,
            )
        )
        specs.extend(self.index_catalog_specs(max_attempts, as_of=end))
        specs = apply_reference_refresh(specs, as_of=end)
        return self.checkpoint.add(specs)

    def index_catalog_specs(
        self, max_attempts: int, *, as_of: date | None = None
    ) -> list[FetchSpec]:
        specs = [
            FetchSpec(
                dataset="index_basic",
                api_name="index_basic",
                scope={
                    "market": market,
                    "page_group": f"index_basic:{market}",
                    "page_size": INDEX_CATALOG_PAGE_SIZE,
                    "offset": 0,
                },
                params={"market": market, "limit": INDEX_CATALOG_PAGE_SIZE, "offset": 0},
                fields=REFERENCE_FIELDS["index_basic"],
                allow_empty=True,
                max_attempts=max_attempts,
            )
            for market in INDEX_CATALOG_MARKETS
        ]
        return apply_reference_refresh(specs, as_of=as_of) if as_of else specs

    def trading_dates(self, start: date, end: date) -> list[str]:
        frame = self.storage.read_units(self.checkpoint.successful("trade_cal"))
        if frame.empty:
            raise RuntimeError("trade_cal is unavailable; reference bootstrap must succeed first")
        frame["cal_date"] = pd.to_datetime(frame["cal_date"])
        is_open = frame["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
        mask = is_open & frame["cal_date"].between(pd.Timestamp(start), pd.Timestamp(end))
        return sorted(frame.loc[mask, "cal_date"].dt.strftime("%Y%m%d").unique().tolist())

    def plan_daily(
        self,
        dates: Iterable[str],
        definitions: Iterable[DatasetDefinition],
        max_attempts: int,
    ) -> int:
        specs = []
        lower_bounds = {
            definition.name: history_start_date(definition.name)
            for definition in definitions
        }
        for trade_date in dates:
            for definition in definitions:
                lower_bound = lower_bounds[definition.name]
                if lower_bound is not None and trade_date < compact_date(lower_bound):
                    continue
                scope: dict[str, object] = {"trade_date": trade_date}
                if definition.row_limit is not None:
                    scope.update(
                        {
                            "row_limit": definition.row_limit,
                            "expected_date_field": "trade_date",
                            "expected_date": trade_date,
                        }
                    )
                specs.append(
                    FetchSpec(
                        dataset=definition.name,
                        api_name=definition.api_name,
                        scope=scope,
                        params={"trade_date": trade_date},
                        fields=definition.fields,
                        allow_empty=definition.allow_empty,
                        max_attempts=max_attempts,
                    )
                )
        return self.checkpoint.add(specs)

    def plan_index_context(self, start: date, end: date, max_attempts: int) -> int:
        specs: list[FetchSpec] = []
        index_starts = self.index_history_starts()
        for index_code in INDEX_CODES:
            available_start = max(start, index_starts.get(index_code, start))
            if available_start > end:
                continue
            specs.append(
                FetchSpec(
                    dataset="index_daily",
                    api_name="index_daily",
                    scope={
                        "index_code": index_code,
                        "start": compact_date(available_start),
                        "end": compact_date(end),
                    },
                    params={
                        "ts_code": index_code,
                        "start_date": compact_date(available_start),
                        "end_date": compact_date(end),
                    },
                    allow_empty=False,
                    max_attempts=max_attempts,
                )
            )
            for chunk_start, chunk_end in _month_ranges(available_start, end):
                specs.append(
                    FetchSpec(
                        dataset="index_weight",
                        api_name="index_weight",
                        scope={
                            "index_code": index_code,
                            "start": compact_date(chunk_start),
                            "end": compact_date(chunk_end),
                        },
                        params={
                            "index_code": index_code,
                            "start_date": compact_date(chunk_start),
                            "end_date": compact_date(chunk_end),
                        },
                        allow_empty=True,
                        max_attempts=max_attempts,
                    )
                )
        for trade_date in self.trading_dates(start, end):
            specs.append(
                FetchSpec(
                    dataset=INDEX_DAILY_BASIC.name,
                    api_name=INDEX_DAILY_BASIC.api_name,
                    scope={
                        "trade_date": trade_date,
                    },
                    params={
                        "trade_date": trade_date,
                    },
                    fields=INDEX_DAILY_BASIC.fields,
                    allow_empty=INDEX_DAILY_BASIC.allow_empty,
                    max_attempts=max_attempts,
                )
            )
        return self.checkpoint.add(apply_reference_refresh(specs, as_of=end))

    def index_history_starts(self) -> dict[str, date]:
        """Read index inception dates from the downloaded index master."""

        frame = self.storage.read_units(self.checkpoint.successful("index_basic"))
        if frame.empty or "ts_code" not in frame.columns or "list_date" not in frame.columns:
            return {}
        normalized = frame[["ts_code", "list_date"]].copy()
        normalized["list_date"] = pd.to_datetime(
            normalized["list_date"], errors="coerce"
        )
        normalized = normalized.dropna(subset=["ts_code", "list_date"])
        starts: dict[str, date] = {}
        for symbol, values in normalized.groupby("ts_code")["list_date"]:
            starts[str(symbol)] = values.min().date()
        return starts

    def plan_research_reference(self, start: date, end: date, max_attempts: int) -> int:
        specs: list[FetchSpec] = [
            FetchSpec(
                dataset="fund_basic",
                api_name="fund_basic",
                scope={"market": "E"},
                params={"market": "E"},
                allow_empty=False,
                max_attempts=max_attempts,
            )
        ]
        for level in ("L1", "L2", "L3"):
            specs.append(
                FetchSpec(
                    dataset="index_classify",
                    api_name="index_classify",
                    scope={"level": level, "src": "SW2021"},
                    params={"level": level, "src": "SW2021"},
                    allow_empty=False,
                    max_attempts=max_attempts,
                )
            )
        disclosure_start = history_start_date("disclosure_date")
        for period in _report_periods(start, end):
            if (
                disclosure_start is not None
                and period < compact_date(disclosure_start)
            ):
                continue
            specs.append(
                FetchSpec(
                    dataset="disclosure_date",
                    api_name="disclosure_date",
                    scope={"end_date": period},
                    params={"end_date": period},
                    fields=DISCLOSURE_FIELDS,
                    allow_empty=False,
                    max_attempts=max_attempts,
                )
            )
        return self.checkpoint.add(apply_reference_refresh(specs, as_of=end))

    def industry_codes(self) -> list[str]:
        frame = self.storage.read_units(self.checkpoint.successful("index_classify"))
        if frame.empty:
            return []
        column = "index_code" if "index_code" in frame.columns else "ts_code"
        if "level" in frame.columns:
            frame = frame[frame["level"].astype(str).str.upper() == "L3"]
        return sorted(frame[column].dropna().astype(str).unique().tolist())

    def plan_industry_members(self, max_attempts: int, *, as_of: date | None = None) -> int:
        specs = [
            FetchSpec(
                dataset="index_member_all",
                api_name="index_member_all",
                scope={
                    "l3_code": index_code,
                    "is_new": membership_status,
                    "row_limit": 2_000,
                },
                params={"l3_code": index_code, "is_new": membership_status},
                allow_empty=True,
                max_attempts=max_attempts,
            )
            for index_code in self.industry_codes()
            for membership_status in ("Y", "N")
        ]
        return self.checkpoint.add(apply_reference_refresh(specs, as_of=as_of) if as_of else specs)

    def plan_etf_daily(self, dates: Iterable[str], max_attempts: int) -> int:
        return self.plan_daily(dates, ETF_DAILY, max_attempts)

    def plan_news(self, start: date, end: date, max_attempts: int) -> int:
        return self.checkpoint.add(self.news_specs(start, end, max_attempts))

    def news_specs(self, start: date, end: date, max_attempts: int) -> list[FetchSpec]:
        desired = news_specs(start, end, max_attempts=max_attempts)
        successful_rows = self.checkpoint.successful(NEWS_DATASET)
        successful = [_checkpoint_spec(row) for row in successful_rows]
        successful_by_source_day: dict[
            tuple[str, date], list[tuple[datetime, datetime, FetchSpec]]
        ] = {}
        for candidate in successful:
            try:
                candidate_start = datetime.fromisoformat(str(candidate.params["start_date"]))
                candidate_end = datetime.fromisoformat(str(candidate.params["end_date"]))
            except (KeyError, ValueError):
                continue
            source = str(candidate.params.get("src") or "")
            if candidate_start.date() != candidate_end.date() or not source:
                continue
            successful_by_source_day.setdefault((source, candidate_start.date()), []).append(
                (candidate_start, candidate_end, candidate)
            )
        reusable: dict[str, FetchSpec] = {}
        missing: list[FetchSpec] = []
        for target in desired:
            source = str(target.params["src"])
            day = datetime.fromisoformat(str(target.params["start_date"])).date()
            day_start = datetime.combine(day, time.min)
            day_end = datetime.combine(day, time.max.replace(microsecond=0))
            windows = [
                item
                for item in successful_by_source_day.get((source, day), [])
                if day_start <= item[0] <= item[1] <= day_end
            ]
            if not windows:
                missing.append(target)
                continue

            cursor = day_start
            for window_start, window_end, candidate in sorted(windows, key=lambda item: item[0]):
                reusable[candidate.unit_key] = candidate
                if window_start > cursor:
                    missing.append(
                        news_window_spec(
                            source,
                            cursor,
                            window_start - timedelta(seconds=1),
                            max_attempts=max_attempts,
                        )
                    )
                cursor = max(cursor, window_end + timedelta(seconds=1))
            if cursor <= day_end:
                missing.append(
                    news_window_spec(
                        source,
                        cursor,
                        day_end,
                        max_attempts=max_attempts,
                    )
                )

        planned_keys = {spec.unit_key for spec in [*reusable.values(), *missing]}
        legacy_unfinished = []
        for row in self.checkpoint.unfinished_units(NEWS_DATASET):
            spec = _checkpoint_spec(row)
            if spec.unit_key in planned_keys or spec.scope.get("partition_axis"):
                continue
            try:
                window_start = datetime.fromisoformat(str(spec.params["start_date"]))
                window_end = datetime.fromisoformat(str(spec.params["end_date"]))
            except (KeyError, ValueError):
                continue
            if window_start.date() <= end and window_end.date() >= start:
                legacy_unfinished.append(spec.unit_key)
        self.checkpoint.supersede_units(
            legacy_unfinished,
            "legacy news window superseded by whole-day adaptive planning",
        )
        return [*reusable.values(), *missing]

    def plan_profile(
        self, profile: str, start: date, end: date, max_attempts: int
    ) -> dict[str, int]:
        dates = self.trading_dates(start, end)
        planned = {
            "core_daily": self.plan_daily(dates, CORE_DAILY, max_attempts),
            "index_context": self.plan_index_context(start, end, max_attempts),
        }
        if profile in {"research", "full"}:
            planned["research_daily"] = self.plan_daily(dates, RESEARCH_DAILY, max_attempts)
            planned["research_reference"] = self.plan_research_reference(start, end, max_attempts)
            planned["etf_daily"] = self.plan_etf_daily(dates, max_attempts)
        if profile == "full":
            planned["news"] = self.plan_news(start, end, max_attempts)
        return planned


class ExecutionDataPlanner:
    """Plan bounded execution-data calls without per-symbol/per-day request churn."""

    def __init__(self, checkpoint: CheckpointStore) -> None:
        self.checkpoint = checkpoint

    def plan_margin(self, trading_dates: Iterable[str], max_attempts: int) -> list[FetchSpec]:
        specs = margin_specs(trading_dates, max_attempts=max_attempts)
        self.checkpoint.add(specs)
        self.checkpoint.retry_failed_units(spec.unit_key for spec in specs)
        return specs

    def plan_minutes(
        self,
        symbols_by_dataset: dict[str, Iterable[str]],
        start: date,
        end: date,
        max_attempts: int,
        *,
        freq: str = "1min",
        active_ranges_by_dataset: dict[str, dict[str, tuple[date, date]]] | None = None,
        trading_dates: Iterable[str] | None = None,
    ) -> list[FetchSpec]:
        specs = minute_specs(
            symbols_by_dataset,
            start=start,
            end=end,
            max_attempts=max_attempts,
            freq=freq,
            active_ranges_by_dataset=active_ranges_by_dataset,
            trading_dates=trading_dates,
        )
        if (
            freq == "5min"
            and set(symbols_by_dataset) == {"ashare_5m"}
            and trading_dates is not None
        ):
            specs = self._reuse_a_share_five_minute_history(
                specs,
                symbols_by_dataset["ashare_5m"],
                start=start,
                end=end,
                active_ranges=(active_ranges_by_dataset or {}).get("ashare_5m", {}),
                trading_dates=trading_dates,
                max_attempts=max_attempts,
            )
        else:
            specs = self._reuse_exact_minute_windows(specs)
        self.checkpoint.add(specs)
        return specs

    def _reuse_exact_minute_windows(self, planned: list[FetchSpec]) -> list[FetchSpec]:
        """Keep legacy 1-minute/futures successes when only scope metadata changed."""

        datasets = {spec.dataset for spec in planned}
        successful: dict[tuple[str, str, str, tuple[str, ...]], FetchSpec] = {}
        for dataset in datasets:
            for row in self.checkpoint.successful(dataset):
                candidate = _checkpoint_spec(row)
                key = (
                    candidate.dataset,
                    candidate.api_name,
                    canonical_json(candidate.params),
                    candidate.fields,
                )
                successful[key] = candidate
        result: list[FetchSpec] = []
        for target in planned:
            key = (
                target.dataset,
                target.api_name,
                canonical_json(target.params),
                target.fields,
            )
            candidate = successful.get(key)
            result.append(candidate or target)

        result_keys = {spec.unit_key for spec in result}
        stale = []
        target_params = {
            (spec.dataset, spec.api_name, canonical_json(spec.params), spec.fields)
            for spec in planned
        }
        for dataset in datasets:
            for row in self.checkpoint.unfinished_units(dataset):
                candidate = _checkpoint_spec(row)
                identity = (
                    candidate.dataset,
                    candidate.api_name,
                    canonical_json(candidate.params),
                    candidate.fields,
                )
                if (
                    identity in target_params
                    and candidate.unit_key not in result_keys
                    and not candidate.scope.get("partition_axis")
                ):
                    stale.append(candidate.unit_key)
        self.checkpoint.supersede_units(
            stale,
            "legacy minute window superseded by adaptive scope metadata",
        )
        return result

    def _reuse_a_share_five_minute_history(
        self,
        planned: list[FetchSpec],
        symbols: Iterable[str],
        *,
        start: date,
        end: date,
        active_ranges: dict[str, tuple[date, date]],
        trading_dates: Iterable[str],
        max_attempts: int,
    ) -> list[FetchSpec]:
        sessions = sorted(
            {
                datetime.strptime(str(value).replace("-", "")[:8], "%Y%m%d").date()
                for value in trading_dates
            }
        )
        successful_by_symbol: dict[str, list[tuple[date, date, FetchSpec]]] = {}
        for row in self.checkpoint.successful("ashare_5m"):
            candidate = _checkpoint_spec(row)
            if str(candidate.params.get("freq") or "") != "5min":
                continue
            try:
                window_start = datetime.fromisoformat(str(candidate.params["start_date"])).date()
                window_end = datetime.fromisoformat(str(candidate.params["end_date"])).date()
            except (KeyError, ValueError):
                continue
            symbol = str(candidate.params.get("ts_code") or "").upper()
            successful_by_symbol.setdefault(symbol, []).append(
                (window_start, window_end, candidate)
            )
        reusable: dict[str, FetchSpec] = {}
        windows: dict[str, list[tuple[date, date]]] = {}
        normalized_symbols = sorted({str(value).strip().upper() for value in symbols})
        for symbol in normalized_symbols:
            active_start, active_end = active_ranges.get(symbol, (start, end))
            desired_sessions = [
                value
                for value in sessions
                if max(start, active_start) <= value <= min(end, active_end)
            ]
            covered: set[date] = set()
            for window_start, window_end, candidate in successful_by_symbol.get(symbol, []):
                if not (start <= window_start <= window_end <= end):
                    continue
                reusable[candidate.unit_key] = candidate
                covered.update(
                    value for value in desired_sessions if window_start <= value <= window_end
                )
            missing_sessions = [value for value in desired_sessions if value not in covered]
            symbol_windows: list[tuple[date, date]] = []
            segment: list[date] = []
            for value in desired_sessions:
                if value in covered:
                    if segment:
                        symbol_windows.extend(_session_chunks(segment, 150))
                        segment = []
                else:
                    segment.append(value)
            if segment:
                symbol_windows.extend(_session_chunks(segment, 150))
            if missing_sessions:
                windows[symbol] = symbol_windows

        new_specs = (
            minute_specs(
                {"ashare_5m": windows},
                start=start,
                end=end,
                max_attempts=max_attempts,
                freq="5min",
                active_ranges_by_dataset={"ashare_5m": active_ranges},
                trading_dates=sessions,
                windows_by_dataset={"ashare_5m": windows},
            )
            if windows
            else []
        )
        planned_keys = {spec.unit_key for spec in [*reusable.values(), *new_specs]}
        legacy_unfinished = []
        for row in self.checkpoint.unfinished_units("ashare_5m"):
            spec = _checkpoint_spec(row)
            if spec.unit_key in planned_keys or spec.scope.get("partition_axis"):
                continue
            if str(spec.params.get("ts_code") or "").upper() in normalized_symbols:
                legacy_unfinished.append(spec.unit_key)
        self.checkpoint.supersede_units(
            legacy_unfinished,
            "legacy monthly A-share 5-minute unit superseded by session-budget planning",
        )
        return [*reusable.values(), *new_specs]


def _checkpoint_spec(row: dict[str, object]) -> FetchSpec:
    return FetchSpec(
        dataset=str(row["dataset"]),
        api_name=str(row["api_name"]),
        scope=dict(row.get("scope_json") or {}),
        params=dict(row.get("params_json") or {}),
        fields=tuple(row.get("fields_json") or ()),
        allow_empty=bool(row.get("allow_empty")),
        max_attempts=int(row.get("max_attempts") or 1),
    )


def _session_chunks(values: list[date], size: int) -> list[tuple[date, date]]:
    return [
        (values[offset], values[min(offset + size - 1, len(values) - 1)])
        for offset in range(0, len(values), size)
    ]


def parse_date(value: str, *, latest: date | None = None) -> date:
    if value.lower() == "latest":
        return latest or today_cn()
    return datetime.strptime(value, "%Y-%m-%d").date()


def _quarter_ranges(start: date, end: date) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    cursor = date(start.year, ((start.month - 1) // 3) * 3 + 1, 1)
    while cursor <= end:
        next_month = cursor.month + 3
        next_year = cursor.year
        if next_month > 12:
            next_month -= 12
            next_year += 1
        next_quarter = date(next_year, next_month, 1)
        chunk_start = max(start, cursor)
        chunk_end = min(end, date.fromordinal(next_quarter.toordinal() - 1))
        ranges.append((chunk_start, chunk_end))
        cursor = next_quarter
    return ranges


def _month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        if cursor.month == 12:
            next_month = date(cursor.year + 1, 1, 1)
        else:
            next_month = date(cursor.year, cursor.month + 1, 1)
        ranges.append((max(start, cursor), min(end, next_month - timedelta(days=1))))
        cursor = next_month
    return ranges


def _report_periods(start: date, end: date) -> list[str]:
    periods: list[str] = []
    first_year = start.year - 1
    for year in range(first_year, end.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            period = date(year, month, day)
            if period <= end:
                periods.append(compact_date(period))
    return periods
