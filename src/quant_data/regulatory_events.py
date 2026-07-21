"""Regulatory major-violation events derived from the anns_d announcement index.

This is the data producer behind the eligibility matrix ``major_violation``
rule (``quant_platform.eligibility``). It scans persisted ``anns_d`` titles for
a small, conservative set of severe regulatory events — CSRC investigations
(立案调查), administrative penalties (行政处罚) and exchange public censures
(公开谴责) — and emits one row per (instrument, event type).

Design choices:

- Title rules, not the LLM/NLP layer: the eligibility defense line must stay
  computable without LLM credentials or downloaded PDF bodies. The rules are
  deliberately conservative (misses are acceptable — a miss simply reverts to
  the pre-existing no-filter behavior — while a false positive would wrongly
  exclude a sound company from backtests). Mild letters (问询函/关注函/
  监管函/警示函) and ordinary 通报批评 discipline are intentionally NOT
  treated as major violations.
- Point-in-time semantics: an instrument is excluded from the first open
  trading day strictly after the announcement date (``known_date``), the same
  ``strictly_after_announcement_date`` policy style as financial reports. The
  exclusion is permanent — these companies are not worth letting back in.
- Fail-closed: an announcement dated beyond the persisted trade_cal raises
  instead of guessing a knowledge date.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date

import pandas as pd

from .cninfo_announcements import next_trading_day

REGULATORY_EVENTS_RULE_VERSION = "regulatory-events-title-rules.v1"
REGULATORY_EVENTS_DATASET = "regulatory_events"

# Ordered (event_type, title pattern) rules. The first matching rule classifies
# a title; every match is a major violation.
EVENT_TYPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("csrc_investigation", re.compile(r"立案调查|立案告知")),
    ("administrative_penalty", re.compile(r"行政处罚")),
    ("public_censure", re.compile(r"公开谴责")),
)

EVENT_COLUMNS = (
    "ts_code",
    "event_date",
    "known_date",
    "major",
    "event_type",
    "title",
    "url",
    "rule_version",
)


class RegulatoryEventsError(RuntimeError):
    """Raised when events cannot be derived without violating PIT discipline."""


def classify_title(title: object) -> str | None:
    """Return the major-violation event type for a title, or None when clean."""

    if title is None:
        return None
    text = str(title)
    for event_type, pattern in EVENT_TYPE_RULES:
        if pattern.search(text):
            return event_type
    return None


def open_days_from_trade_cal(trade_cal: pd.DataFrame) -> list[date]:
    """Extract sorted open trading days from a trade_cal frame; fail closed."""

    if not {"cal_date", "is_open"}.issubset(trade_cal.columns):
        raise RegulatoryEventsError(
            "trade_cal lacks cal_date/is_open; refusing to derive known_date"
        )
    frame = trade_cal.copy()
    text = frame["cal_date"].astype("string").str.replace(r"\.0$", "", regex=True)
    frame["cal_date"] = pd.to_datetime(text, format="%Y%m%d", errors="coerce").fillna(
        pd.to_datetime(frame["cal_date"], errors="coerce")
    )
    is_open = frame["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
    days = sorted({value.date() for value in frame.loc[is_open, "cal_date"].dropna()})
    if not days:
        raise RegulatoryEventsError(
            "trade_cal has no open day; refusing to guess regulatory known_date"
        )
    return days


def _normalize_ann_date(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return parsed.fillna(pd.to_datetime(series, errors="coerce")).dt.normalize()


def _empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": pd.Series(dtype="string"),
            "event_date": pd.Series(dtype="datetime64[ns]"),
            "known_date": pd.Series(dtype="datetime64[ns]"),
            "major": pd.Series(dtype="bool"),
            "event_type": pd.Series(dtype="string"),
            "title": pd.Series(dtype="string"),
            "url": pd.Series(dtype="string"),
            "rule_version": pd.Series(dtype="string"),
        }
    )


def derive_regulatory_events(
    announcements: pd.DataFrame,
    open_days: Sequence[date],
) -> pd.DataFrame:
    """Derive major-violation events from anns_d rows (ts_code/ann_date/title).

    ``open_days`` must come from the persisted trade_cal (see
    ``open_days_from_trade_cal``); known_date is the first open day strictly
    after the announcement date. One output row per (ts_code, event_type),
    keeping the earliest announcement of that type. Raises
    RegulatoryEventsError when a matched announcement lies beyond the calendar
    (fail-closed rather than guessing a knowledge date).
    """

    if announcements.empty:
        return _empty_events_frame()
    required = {"ts_code", "ann_date", "title"}
    missing = sorted(required - set(announcements.columns))
    if missing:
        raise RegulatoryEventsError(f"anns_d frame is missing columns: {missing}")
    frame = announcements.copy()
    frame["ts_code"] = frame["ts_code"].astype("string").str.upper()
    frame["ann_date"] = _normalize_ann_date(frame["ann_date"])
    frame = frame.dropna(subset=["ts_code", "ann_date"])
    frame["event_type"] = frame["title"].map(classify_title)
    matched = frame.dropna(subset=["event_type"])
    if matched.empty:
        return _empty_events_frame()
    matched = matched.sort_values(["ts_code", "event_type", "ann_date"], kind="stable")
    matched = matched.drop_duplicates(["ts_code", "event_type"], keep="first")

    rows: list[dict[str, object]] = []
    for row in matched.itertuples(index=False):
        ann_date = row.ann_date.date()
        try:
            known_date = next_trading_day(ann_date, open_days)
        except LookupError as exc:
            raise RegulatoryEventsError(
                f"no trading day after {ann_date} for {row.ts_code} "
                f"({row.event_type}); extend trade_cal before deriving events"
            ) from exc
        rows.append(
            {
                "ts_code": str(row.ts_code),
                "event_date": ann_date,
                "known_date": known_date,
                "major": True,
                "event_type": str(row.event_type),
                "title": "" if pd.isna(row.title) else str(row.title),
                "url": "" if "url" not in matched.columns or pd.isna(row.url) else str(row.url),
                "rule_version": REGULATORY_EVENTS_RULE_VERSION,
            }
        )
    result = pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
    result["event_date"] = pd.to_datetime(result["event_date"])
    result["known_date"] = pd.to_datetime(result["known_date"])
    return result.sort_values(["ts_code", "event_date", "event_type"], kind="stable").reset_index(
        drop=True
    )
