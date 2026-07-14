from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime

import pandas as pd

from .catalog import (
    CORE_DAILY,
    DISCLOSURE_FIELDS,
    ETF_DAILY,
    INDEX_CODES,
    REFERENCE_FIELDS,
    RESEARCH_DAILY,
    DatasetDefinition,
)
from .checkpoint import CheckpointStore
from .execution_data import margin_specs, minute_specs
from .models import FetchSpec
from .storage import ParquetStore


def compact_date(value: date) -> str:
    return value.strftime("%Y%m%d")


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
        return self.checkpoint.add(specs)

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
        specs = (
            FetchSpec(
                dataset=definition.name,
                api_name=definition.api_name,
                scope={"trade_date": trade_date},
                params={"trade_date": trade_date},
                fields=definition.fields,
                allow_empty=definition.allow_empty,
                max_attempts=max_attempts,
            )
            for trade_date in dates
            for definition in definitions
        )
        return self.checkpoint.add(specs)

    def plan_index_context(self, start: date, end: date, max_attempts: int) -> int:
        specs: list[FetchSpec] = []
        for index_code in INDEX_CODES:
            specs.append(
                FetchSpec(
                    dataset="index_daily",
                    api_name="index_daily",
                    scope={
                        "index_code": index_code,
                        "start": compact_date(start),
                        "end": compact_date(end),
                    },
                    params={
                        "ts_code": index_code,
                        "start_date": compact_date(start),
                        "end_date": compact_date(end),
                    },
                    allow_empty=False,
                    max_attempts=max_attempts,
                )
            )
            for chunk_start, chunk_end in _quarter_ranges(start, end):
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
        return self.checkpoint.add(specs)

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
        for period in _report_periods(start, end):
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
        return self.checkpoint.add(specs)

    def industry_codes(self) -> list[str]:
        frame = self.storage.read_units(self.checkpoint.successful("index_classify"))
        if frame.empty:
            return []
        column = "index_code" if "index_code" in frame.columns else "ts_code"
        return sorted(frame[column].dropna().astype(str).unique().tolist())

    def plan_industry_members(self, max_attempts: int) -> int:
        specs = [
            FetchSpec(
                dataset="index_member_all",
                api_name="index_member_all",
                scope={"index_code": index_code},
                params={"index_code": index_code},
                allow_empty=False,
                max_attempts=max_attempts,
            )
            for index_code in self.industry_codes()
        ]
        return self.checkpoint.add(specs)

    def plan_etf_daily(self, dates: Iterable[str], max_attempts: int) -> int:
        return self.plan_daily(dates, ETF_DAILY, max_attempts)

    def plan_news(self, start: date, end: date, max_attempts: int) -> int:
        specs: list[FetchSpec] = []
        cursor = start
        while cursor <= end:
            day = cursor.isoformat()
            specs.append(
                FetchSpec(
                    dataset="news",
                    api_name="news",
                    scope={"date": day},
                    params={
                        "start_date": f"{day} 00:00:00",
                        "end_date": f"{day} 23:59:59",
                    },
                    allow_empty=True,
                    max_attempts=max_attempts,
                )
            )
            cursor = date.fromordinal(cursor.toordinal() + 1)
        return self.checkpoint.add(specs)

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
    ) -> list[FetchSpec]:
        specs = minute_specs(
            symbols_by_dataset,
            start=start,
            end=end,
            max_attempts=max_attempts,
        )
        self.checkpoint.add(specs)
        self.checkpoint.retry_failed_units(spec.unit_key for spec in specs)
        return specs


def parse_date(value: str, *, latest: date | None = None) -> date:
    if value.lower() == "latest":
        return latest or date.today()
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


def _report_periods(start: date, end: date) -> list[str]:
    periods: list[str] = []
    first_year = start.year - 1
    for year in range(first_year, end.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            period = date(year, month, day)
            if period <= end:
                periods.append(compact_date(period))
    return periods
