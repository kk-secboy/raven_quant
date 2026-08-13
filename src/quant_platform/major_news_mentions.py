"""Per-instrument mapping of major_news mentions + mention sentiment factors.

``major_news`` long-form market news is already consumed by
:mod:`quant_platform.corpus_nlp` at market level (MARKET pseudo-instrument).
This module adds the per-instrument layer: it deterministically identifies the
listed companies mentioned in each news item's title+content and propagates the
item's existing LLM sentiment (from the persisted corpus NLP fields index) to
every mentioned instrument. No new LLM calls are made — the per-item sentiment
is reused, so an item that mentions three stocks contributes the same score to
three instruments (documented, conservative first cut; per-(item, stock)
targeted sentiment would need a new LLM pass and is deliberately not done).

Mention rules (``MENTION_RULES_VERSION``, deterministic, no LLM):

1. Alias candidates are the current 简称 (``stock_basic.name``), the full
   company name (``stock_basic.fullname`` when the column was downloaded) and
   every historical name from ``namechange`` with its [start_date, end_date]
   validity interval. An alias is only usable for a news item when the item's
   pub_time date falls inside the alias validity interval — the point-in-time
   name, not today's name, drives the match. Stocks without ``namechange``
   rows use the ``stock_basic`` name valid from ``list_date`` onward.
2. Aliases shorter than ``MIN_ALIAS_CHARS`` characters are dropped entirely:
   two-character names are exactly the high-ambiguity case (平安/万科-style
   fragments and common words), and guessing them is worse than missing them.
3. An alias string that maps to more than one ts_code with *overlapping*
   validity intervals is dropped for all of them (conservative); sequential
   reuse with disjoint intervals stays usable.
4. Matching scans the text left to right; at each position the longest
   matching valid alias wins and the matched span is consumed (non-overlapping
   matches). A stock is mentioned at most once per item.

Outputs land under ``data/major_news_mentions/`` following the report_rc
layout:

- ``aliases.parquet`` — the alias table actually used (audit trail)
- ``events.parquet`` — per-(item, ts_code) mention events with the matched
  alias, row-level ``available_at``/``factor_date`` PIT timestamps
- ``factors/<name>.parquet`` + ``factors/<name>.json`` — factor-values
  artifacts with sha256 manifests: ``major_news_mention_sentiment_daily``
  (mean item sentiment per mentioned instrument, sparse event shape) and
  ``major_news_mention_count_daily`` (distinct mentioning items per
  instrument, a news-attention variable)

PIT semantics: ``available_at`` is the exact major_news pub_time carried by
the corpus NLP fields; the daily factor date follows the corpus_nlp 15:00
cutoff rule against the persisted trade_cal (never weekday guesses). Alias
validity is evaluated at pub_time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from quant_data.cninfo_announcements import (
    _parquet_files,
    _read_parquet_union,
    load_trade_calendar_open_days,
)

from .announcement_factor_registry import (
    ExternalFactorMetadata,
    register_external_factor,
)
from .announcement_nlp import _sha256_file, _write_json_atomic, _write_parquet_atomic
from .corpus_nlp import (
    CORPUS_NLP_DIR,
    DATASET_MAJOR_NEWS,
    factor_date_for,
    load_corpus_items,
)
from .factor_evaluator import normalize_series

if TYPE_CHECKING:
    from .research_store import ResearchStore

PRODUCER_VERSION = "major-news-mentions.v1"
MENTION_RULES_VERSION = "mention-rules.v1"

DATASET = "major_news"
MENTIONS_DIR = "major_news_mentions"

SENTIMENT_FACTOR_NAME = "major_news_mention_sentiment_daily"
COUNT_FACTOR_NAME = "major_news_mention_count_daily"
FACTOR_NAMES = (SENTIMENT_FACTOR_NAME, COUNT_FACTOR_NAME)

# Aliases shorter than this are dropped: two-character names are the
# high-ambiguity case and are never guessed.
MIN_ALIAS_CHARS = 3

IMPORT_RUN_KIND = "major_news_mentions_factor_import"
IMPORT_ACTOR = "major-news-mentions-registrar"
SOURCE_DATASET = "corpus_nlp_fields"

AVAILABILITY_POLICY = {
    name: (
        "available_at = major_news exact pub_time (carried by the corpus NLP "
        "fields index); the mention alias must be valid at the pub_time date "
        "(stock_basic/namechange validity intervals, mention-rules.v1); daily "
        "factor date = pub date when available before 15:00 on a trade_cal "
        "trading day, otherwise the next trade_cal trading day"
    )
    for name in FACTOR_NAMES
}

ALIAS_COLUMNS = ("ts_code", "alias", "kind", "valid_from", "valid_to")
EVENT_COLUMNS = (
    "item_id",
    "ts_code",
    "matched_alias",
    "pub_time",
    "available_at",
    "factor_date",
    "sentiment",
)


def default_factors_dir(data_root: Path) -> Path:
    """Return the directory where major_news mention factor artifacts land."""

    return data_root / MENTIONS_DIR / "factors"


def _load_stock_basic(data_root: Path) -> pd.DataFrame:
    """Read stock_basic names from the units/snapshots layout; fail closed."""

    paths = _parquet_files(data_root, "stock_basic")
    if not paths:
        raise RuntimeError(
            f"stock_basic parquet is unavailable under {data_root}; "
            "run the Tushare reference download task first"
        )
    available = set(
        _read_parquet_union(
            paths, "SELECT * FROM read_parquet(?, union_by_name=true) LIMIT 0"
        ).columns
    )
    if "ts_code" not in available or "name" not in available:
        raise RuntimeError("stock_basic parquet misses required columns: ts_code/name")
    select_parts = [
        "CAST(ts_code AS VARCHAR) AS ts_code",
        "CAST(name AS VARCHAR) AS name",
    ]
    if "fullname" in available:
        select_parts.append("CAST(fullname AS VARCHAR) AS fullname")
    if "list_date" in available:
        select_parts.append(
            "coalesce(try_cast(list_date AS DATE), "
            "try_strptime(CAST(list_date AS VARCHAR), '%Y%m%d')::DATE) AS list_date"
        )
    frame = _read_parquet_union(
        paths,
        f"SELECT {', '.join(select_parts)} FROM read_parquet(?, union_by_name=true)",
    )
    frame = frame.dropna(subset=["ts_code", "name"])
    frame["ts_code"] = frame["ts_code"].astype(str).str.strip().str.upper()
    frame["name"] = frame["name"].astype(str).str.strip()
    frame = frame[frame["ts_code"] != ""]
    return frame.drop_duplicates()


def _load_namechange(data_root: Path) -> pd.DataFrame:
    """Read historical-name intervals; empty frame when not downloaded."""

    paths = _parquet_files(data_root, "namechange")
    if not paths:
        return pd.DataFrame(
            columns=["ts_code", "name", "start_date", "end_date"]
        ).astype({"ts_code": "string", "name": "string"})
    available = set(
        _read_parquet_union(
            paths, "SELECT * FROM read_parquet(?, union_by_name=true) LIMIT 0"
        ).columns
    )
    if not {"ts_code", "name"}.issubset(available):
        return pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"])
    select_parts = [
        "CAST(ts_code AS VARCHAR) AS ts_code",
        "CAST(name AS VARCHAR) AS name",
    ]
    for column in ("start_date", "end_date"):
        if column in available:
            select_parts.append(
                f"coalesce(try_cast({column} AS DATE), "
                f"try_strptime(CAST({column} AS VARCHAR), '%Y%m%d')::DATE) AS {column}"
            )
    frame = _read_parquet_union(
        paths,
        f"SELECT {', '.join(select_parts)} FROM read_parquet(?, union_by_name=true)",
    )
    frame = frame.dropna(subset=["ts_code", "name"])
    frame["ts_code"] = frame["ts_code"].astype(str).str.strip().str.upper()
    frame["name"] = frame["name"].astype(str).str.strip()
    for column in ("start_date", "end_date"):
        if column not in frame.columns:
            frame[column] = pd.NaT
    return frame.drop_duplicates()


def _alias_frame(rows: list[tuple[str, str, str, date | None, date | None]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            {
                "ts_code": pd.Series(dtype="string"),
                "alias": pd.Series(dtype="string"),
                "kind": pd.Series(dtype="string"),
                "valid_from": pd.Series(dtype="datetime64[ns]"),
                "valid_to": pd.Series(dtype="datetime64[ns]"),
            }
        )
    frame = pd.DataFrame(rows, columns=["ts_code", "alias", "kind", "valid_from", "valid_to"])
    frame["valid_from"] = pd.to_datetime(frame["valid_from"])
    frame["valid_to"] = pd.to_datetime(frame["valid_to"])
    return frame


def build_alias_table(
    stock_basic: pd.DataFrame, namechange: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build the point-in-time alias table under MENTION_RULES_VERSION.

    Eligibility: alias length >= MIN_ALIAS_CHARS. Cross-stock conflicts (same
    alias string, overlapping validity intervals) drop the alias for every
    involved ts_code. Identical (ts_code, alias) rows from different sources
    merge onto the union of their validity intervals.
    """

    rows: list[tuple[str, str, str, date | None, date | None]] = []
    for row in stock_basic.itertuples():
        list_date = (
            pd.Timestamp(row.list_date).date()
            if "list_date" in stock_basic.columns and pd.notna(row.list_date)
            else None
        )
        name = str(row.name).strip()
        rows.append((row.ts_code, name, "name", list_date, None))
        fullname = getattr(row, "fullname", None)
        if fullname is not None and str(fullname).strip():
            rows.append((row.ts_code, str(fullname).strip(), "fullname", list_date, None))
    if namechange is not None and not namechange.empty:
        for row in namechange.itertuples():
            start = (
                pd.Timestamp(row.start_date).date() if pd.notna(row.start_date) else None
            )
            end = pd.Timestamp(row.end_date).date() if pd.notna(row.end_date) else None
            rows.append((row.ts_code, str(row.name).strip(), "name", start, end))
    frame = _alias_frame(rows)
    if frame.empty:
        return frame
    frame = frame[frame["alias"].str.len() >= MIN_ALIAS_CHARS]
    # Merge identical (ts_code, alias) rows onto the union of their intervals
    # (None = open-ended on either side).
    merged = (
        frame.groupby(["ts_code", "alias", "kind"], as_index=False)
        .agg(
            valid_from=("valid_from", "min"),
            valid_to=("valid_to", "max"),
            has_open_to=("valid_to", lambda values: bool(values.isna().any())),
            has_open_from=("valid_from", lambda values: bool(values.isna().any())),
        )
    )
    merged.loc[merged["has_open_to"], "valid_to"] = pd.NaT
    merged.loc[merged["has_open_from"], "valid_from"] = pd.NaT
    merged = merged.drop(columns=["has_open_to", "has_open_from"])
    # Cross-stock conflict: same alias string, overlapping intervals -> drop.
    far_future = pd.Timestamp("2262-01-01")
    far_past = pd.Timestamp("1677-01-01")
    dropped: set[str] = set()
    for alias, group in merged.groupby("alias"):
        codes = group["ts_code"].unique()
        if len(codes) < 2:
            continue
        intervals = [
            (
                row.valid_from if pd.notna(row.valid_from) else far_past,
                row.valid_to if pd.notna(row.valid_to) else far_future,
                row.ts_code,
            )
            for row in group.itertuples()
        ]
        intervals.sort()
        for index in range(1, len(intervals)):
            if intervals[index][0] <= intervals[index - 1][1]:
                dropped.add(str(alias))
                break
    merged = merged[~merged["alias"].isin(dropped)]
    return (
        merged[list(ALIAS_COLUMNS)]
        .sort_values(["alias", "ts_code"], kind="stable")
        .reset_index(drop=True)
    )


def _alias_index_at(aliases: pd.DataFrame, day: date) -> dict[str, list[tuple[str, str]]]:
    """first-character -> [(alias, ts_code)] longest-first, valid at ``day``."""

    stamp = pd.Timestamp(day)
    valid = aliases[
        (aliases["valid_from"].isna() | (aliases["valid_from"] <= stamp))
        & (aliases["valid_to"].isna() | (aliases["valid_to"] >= stamp))
    ]
    index: dict[str, list[tuple[str, str]]] = {}
    for row in valid.itertuples():
        bucket = index.setdefault(row.alias[0], [])
        entry = (row.alias, row.ts_code)
        if entry not in bucket:
            bucket.append(entry)
    for bucket in index.values():
        bucket.sort(key=lambda entry: len(entry[0]), reverse=True)
    return index


def find_mentions(
    text: str, alias_index: dict[str, list[tuple[str, str]]]
) -> dict[str, str]:
    """Scan text for alias mentions; return {ts_code: matched_alias}.

    Left-to-right scan; at each position the longest matching alias wins and
    the matched span is consumed. Each ts_code is reported once (first match).
    """

    mentions: dict[str, str] = {}
    position = 0
    length = len(text)
    while position < length:
        bucket = alias_index.get(text[position])
        matched: tuple[str, str] | None = None
        if bucket:
            for alias, ts_code in bucket:
                if text.startswith(alias, position):
                    matched = (alias, ts_code)
                    break
        if matched is None:
            position += 1
            continue
        alias, ts_code = matched
        mentions.setdefault(ts_code, alias)
        position += len(alias)
    return mentions


def _load_fields(
    data_root: Path,
    *,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Read the corpus NLP fields index (major_news rows only); fail closed."""

    fields_path = data_root / CORPUS_NLP_DIR / "fields.parquet"
    if not fields_path.is_file():
        raise RuntimeError(
            f"corpus NLP fields index is missing: {fields_path}; "
            "run the corpus-nlp pipeline first (quant-data corpus-nlp)"
        )
    path = fields_path.as_posix()
    available = set(
        _read_parquet_union(
            [path], "SELECT * FROM read_parquet(?, union_by_name=true) LIMIT 0"
        ).columns
    )
    required = {
        "item_id",
        "source_dataset",
        "sentiment",
        "available_at",
        "processed_at",
        "prompt_version",
        "model",
    }
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"corpus NLP fields index misses required columns: {missing}")

    available_expr = "try_cast(CAST(available_at AS VARCHAR) AS TIMESTAMP)"
    conditions = ["CAST(source_dataset AS VARCHAR) = ?"]
    parameters: list[object] = [DATASET_MAJOR_NEWS]
    if start is not None:
        conditions.append(f"CAST({available_expr} AS DATE) >= ?")
        parameters.append(start)
    if end is not None:
        conditions.append(f"CAST({available_expr} AS DATE) <= ?")
        parameters.append(end)
    where = " AND ".join(conditions)
    return _read_parquet_union(
        [path],
        f"""
        SELECT
            CAST(item_id AS VARCHAR) AS item_id,
            try_cast(CAST(sentiment AS VARCHAR) AS DOUBLE) AS sentiment,
            {available_expr} AS available_at,
            try_cast(CAST(processed_at AS VARCHAR) AS TIMESTAMP) AS processed_at,
            CAST(prompt_version AS VARCHAR) AS prompt_version,
            CAST(model AS VARCHAR) AS model
        FROM read_parquet(?, union_by_name=true)
        WHERE {where}
        QUALIFY row_number() OVER (
            PARTITION BY CAST(item_id AS VARCHAR)
            ORDER BY try_cast(CAST(processed_at AS VARCHAR) AS TIMESTAMP) DESC NULLS LAST,
                     CAST(processed_at AS VARCHAR) DESC
        ) = 1
        """,
        parameters,
    )


def _bounded_month_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Return non-overlapping calendar-month windows covering the range."""

    if end < start:
        raise ValueError("end must be on or after start")
    windows: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        next_month = (
            date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else date(cursor.year, cursor.month + 1, 1)
        )
        window_end = min(end, next_month - timedelta(days=1))
        windows.append((cursor, window_end))
        cursor = next_month
    return windows


def _iter_major_news_batches(
    data_root: Path,
    *,
    start: date | None,
    end: date | None,
) -> Iterator[list[Any]]:
    """Yield bounded major-news batches instead of retaining all full text.

    Production jobs always carry explicit source boundaries.  Splitting those
    boundaries by calendar month preserves the item-id and de-duplication
    contract because publication timestamps assign every item to exactly one
    window.  The optional unbounded API remains compatible for small/manual
    uses, while bounded historical runs release full article bodies after each
    month.
    """

    windows: list[tuple[date | None, date | None]] = (
        [(start, end)]
        if start is None or end is None
        else list(_bounded_month_windows(start, end))
    )
    for window_start, window_end in windows:
        yield load_corpus_items(
            data_root,
            datasets={DATASET_MAJOR_NEWS},
            start=window_start,
            end=window_end,
        )


def build_mention_events(
    data_root: Path,
    *,
    ts_codes: set[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build (events, aliases, provenance) from major_news + corpus fields.

    Fail closed when major_news/stock_basic/trade_cal parquets or the corpus
    fields index are unavailable, or when the calendar cannot cover a
    visibility timestamp. ``provenance`` carries the item count plus the
    prompt_version/model of the most recently processed fields row.
    """

    batches = iter(_iter_major_news_batches(data_root, start=start, end=end))
    # Preserve the original fail-closed ordering: validate/read the source
    # before reporting a downstream corpus-fields gap.
    first_items = next(batches)
    fields = _load_fields(data_root, start=start, end=end)
    sentiment_by_item = fields.set_index("item_id")
    aliases = build_alias_table(_load_stock_basic(data_root), _load_namechange(data_root))
    open_days = load_trade_calendar_open_days(data_root)

    wanted = {code.strip().upper() for code in ts_codes} if ts_codes else None
    events: list[dict[str, Any]] = []
    item_count = 0
    for items in chain((first_items,), batches):
        item_count += len(items)
        # Do not retain a full alias index for every day in the multi-year
        # range.  A monthly cache bounds this structure while reusing it for
        # all items published on the same day.
        index_cache: dict[date, dict[str, list[tuple[str, str]]]] = {}
        for item in items:
            if item.item_id not in sentiment_by_item.index:
                continue  # no successful LLM extraction -> no signal (fail closed)
            field_row = sentiment_by_item.loc[item.item_id]
            day = item.pub_time.date()
            if day not in index_cache:
                index_cache[day] = _alias_index_at(aliases, day)
            text = f"{item.title}\n{item.content}"
            mentions = find_mentions(text, index_cache[day])
            for ts_code, matched_alias in sorted(mentions.items()):
                if wanted is not None and ts_code not in wanted:
                    continue
                available_at = pd.Timestamp(field_row["available_at"]).to_pydatetime()
                try:
                    factor_day = factor_date_for(available_at, open_days)
                except LookupError as exc:
                    raise RuntimeError(
                        f"cannot derive factor_date for item {item.item_id}: {exc}"
                    ) from exc
                events.append(
                    {
                        "item_id": item.item_id,
                        "ts_code": ts_code,
                        "matched_alias": matched_alias,
                        "pub_time": item.pub_time,
                        "available_at": available_at,
                        "factor_date": factor_day,
                        "sentiment": float(field_row["sentiment"]),
                    }
                )
    frame = pd.DataFrame(events, columns=list(EVENT_COLUMNS))
    if not frame.empty:
        for column in ("pub_time", "available_at", "factor_date"):
            frame[column] = pd.to_datetime(frame[column])
        frame = frame.sort_values(
            ["factor_date", "ts_code", "item_id"], kind="stable"
        ).reset_index(drop=True)
    latest = fields.sort_values("processed_at", kind="stable")
    provenance: dict[str, Any] = {
        "items": item_count,
        "prompt_version": str(latest["prompt_version"].iloc[-1]) if len(latest) else "",
        "model": str(latest["model"].iloc[-1]) if len(latest) else "",
    }
    return frame, aliases, provenance


def build_mention_sentiment_series(
    events: pd.DataFrame, name: str = SENTIMENT_FACTOR_NAME
) -> pd.Series:
    """Mean mentioning-item sentiment per (factor_date, ts_code)."""

    grouped = events.groupby(["factor_date", "ts_code"], sort=True)["sentiment"].mean()
    series = grouped.rename(name)
    series.index = series.index.set_names(["datetime", "instrument"])
    return normalize_series(series, name)


def build_mention_count_series(
    events: pd.DataFrame, name: str = COUNT_FACTOR_NAME
) -> pd.Series:
    """Distinct mentioning items per (factor_date, ts_code)."""

    counts = events.groupby(["factor_date", "ts_code"], sort=True)["item_id"].nunique()
    series = counts.astype(float).rename(name)
    series.index = series.index.set_names(["datetime", "instrument"])
    return normalize_series(series, name)


def _write_factor_artifact(
    series: pd.Series,
    factors_dir: Path,
    *,
    name: str,
    provenance: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    """Write the normalized factor-values parquet plus its sha256 manifest."""

    artifact_path = factors_dir / f"{name}.parquet"
    _write_parquet_atomic(series.rename(name).reset_index(), artifact_path)
    manifest: dict[str, Any] = {
        "factor": name,
        "artifact": artifact_path.name,
        "sha256": _sha256_file(artifact_path),
        "rows": int(len(series)),
        "availability_policy": {name: AVAILABILITY_POLICY[name]},
        "source": {
            "dataset": SOURCE_DATASET,
            "source_datasets": [DATASET],
            "producer_version": PRODUCER_VERSION,
            "mention_rules_version": MENTION_RULES_VERSION,
            "prompt_version": provenance["prompt_version"],
            "model": provenance["model"],
            "min_alias_chars": MIN_ALIAS_CHARS,
        },
        "generated_at": now.isoformat(),
    }
    manifest_path = factors_dir / f"{name}.json"
    _write_json_atomic(manifest, manifest_path)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "artifact_path": artifact_path,
    }


@dataclass(slots=True)
class MentionsSummary:
    items: int
    aliases: int
    events: int
    aliases_path: Path
    events_path: Path
    factors: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "succeeded",
            "items": self.items,
            "aliases": self.aliases,
            "events": self.events,
            "aliases_path": str(self.aliases_path),
            "events_path": str(self.events_path),
            "factors": {
                name: {
                    "manifest_path": str(entry["manifest_path"]),
                    "sha256": entry["manifest"]["sha256"],
                    "rows": entry["manifest"]["rows"],
                }
                for name, entry in self.factors.items()
            },
        }


def process_major_news_mentions(
    data_root: Path,
    *,
    ts_codes: set[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    now: Callable[[], datetime] | None = None,
) -> MentionsSummary:
    """Build the alias table, mention events and factor artifacts.

    Deterministic and idempotent: every run recomputes from the persisted
    major_news/stock_basic/namechange parquets, the corpus NLP fields index and
    trade_cal, and atomically rewrites the outputs, so identical inputs yield
    identical artifact sha256 values.
    """

    clock = now or (lambda: datetime.now(UTC))
    events, aliases, provenance = build_mention_events(
        data_root, ts_codes=ts_codes, start=start, end=end
    )

    base = data_root / MENTIONS_DIR
    factors_dir = base / "factors"
    factors_dir.mkdir(parents=True, exist_ok=True)
    aliases_path = base / "aliases.parquet"
    events_path = base / "events.parquet"
    _write_parquet_atomic(aliases, aliases_path)
    _write_parquet_atomic(events, events_path)

    artifacts = {
        SENTIMENT_FACTOR_NAME: _write_factor_artifact(
            build_mention_sentiment_series(events), factors_dir,
            name=SENTIMENT_FACTOR_NAME, provenance=provenance, now=clock(),
        ),
        COUNT_FACTOR_NAME: _write_factor_artifact(
            build_mention_count_series(events), factors_dir,
            name=COUNT_FACTOR_NAME, provenance=provenance, now=clock(),
        ),
    }
    return MentionsSummary(
        items=int(provenance["items"]),
        aliases=int(len(aliases)),
        events=int(len(events)),
        aliases_path=aliases_path,
        events_path=events_path,
        factors=artifacts,
    )


# ---------------------------------------------------------------------------
# Registration into factor_candidates (generic external-factor channel)
# ---------------------------------------------------------------------------

_FACTOR_DESCRIPTIONS = {
    SENTIMENT_FACTOR_NAME: (
        "Per-instrument major_news mention sentiment: mean LLM sentiment of the "
        "long-form news items mentioning the stock (deterministic name matching "
        "with point-in-time alias validity), sparse event shape."
    ),
    COUNT_FACTOR_NAME: (
        "Per-instrument major_news attention: number of distinct long-form news "
        "items mentioning the stock per factor date, sparse event shape."
    ),
}

_FACTOR_FORMULATIONS = {
    SENTIMENT_FACTOR_NAME: (
        "mean(sentiment) over mention events grouped by (factor_date, ts_code); "
        f"mentions from mention-rules.v1 (alias length >= {MIN_ALIAS_CHARS}, "
        "cross-stock conflicts dropped, longest match wins, PIT alias validity "
        "from stock_basic/namechange); sentiment reused from the corpus NLP "
        "fields index (no per-(item, stock) LLM pass)"
    ),
    COUNT_FACTOR_NAME: (
        "count of distinct item_id over mention events grouped by "
        "(factor_date, ts_code); same mention-rules.v1 mapping"
    ),
}


def _code_artifact_source(
    *, factor_name: str, manifest: dict[str, Any], values_sha256: str
) -> str:
    """Deterministic provenance code bound to factor_candidates.code_sha256."""

    source = manifest["source"]
    policy = manifest["availability_policy"][factor_name]
    if factor_name == COUNT_FACTOR_NAME:
        aggregation = (
            "frame.groupby(['factor_date', 'ts_code'], sort=True)['item_id']"
            ".nunique().astype(float)"
        )
    else:
        aggregation = (
            "frame.groupby(['factor_date', 'ts_code'], sort=True)['sentiment'].mean()"
        )
    return f'''"""Provenance code artifact for the externally produced {factor_name} factor.

Generated at factor-registration time by quant_platform.major_news_mentions.
The registered factor values derive from the persisted events.parquet
mention-event intermediate (deterministic mention-rules.v1 mapping of
major_news items onto instruments; sentiment reused from the corpus NLP
fields index, no new LLM calls), normalized with the
factor_evaluator.normalize_series contract.

source dataset: {source["dataset"]}
producer_version: {source["producer_version"]}
mention_rules_version: {source["mention_rules_version"]}
prompt_version: {source["prompt_version"]}
model: {source["model"]}
availability_policy: {policy}
values sha256: {values_sha256}
"""

from __future__ import annotations

import pandas as pd

from quant_platform.factor_evaluator import normalize_series

FACTOR_NAME = {factor_name!r}


def compute_factor(frame: pd.DataFrame) -> pd.Series:
    """Rebuild the factor values from the persisted events.parquet intermediate."""

    frame = frame.copy()
    series = {aggregation}
    series = series.rename(FACTOR_NAME)
    series.index = series.index.set_names(["datetime", "instrument"])
    return normalize_series(series, FACTOR_NAME)
'''


def _mentions_metadata(
    factor_name: str, manifest: dict[str, Any], values_sha256: str
) -> ExternalFactorMetadata:
    source = manifest["source"]
    policy = manifest["availability_policy"]
    return ExternalFactorMetadata(
        description=(
            f"{_FACTOR_DESCRIPTIONS[factor_name]} Availability: "
            f"{policy[factor_name]}. Externally produced by major_news_mentions "
            f"(producer_version={source['producer_version']}, "
            f"mention_rules_version={source['mention_rules_version']}, "
            f"prompt_version={source['prompt_version']}, model={source['model']})."
        ),
        formulation=_FACTOR_FORMULATIONS[factor_name],
        variables={
            "availability_policy": policy,
            "source": source,
            "values_sha256": values_sha256,
            "manifest": None,  # filled by the caller with the manifest path
            "rows": manifest["rows"],
            "min_alias_chars": source["min_alias_chars"],
            "ingested_fields": ["available_at", "ingested_at"],
        },
        code_source=_code_artifact_source(
            factor_name=factor_name, manifest=manifest, values_sha256=values_sha256
        ),
        run_config={
            "producer_version": source["producer_version"],
            "mention_rules_version": source["mention_rules_version"],
            "prompt_version": source["prompt_version"],
            "model": source["model"],
            "availability_policy": policy,
        },
        rdagent_feedback=(
            "externally produced major_news mention factor; "
            "manifest sha256 verified at registration"
        ),
    )


def register_major_news_mentions_factor(
    store: ResearchStore,
    factors_dir: Path,
    *,
    factor_name: str,
    actor: str = IMPORT_ACTOR,
) -> dict[str, Any]:
    """Verify and register one mention factor artifact; idempotent.

    Uses the generic external-factor channel
    (``announcement_factor_registry.register_external_factor``): manifest
    sha256 fail-closed verification, research-run lineage, idempotency key
    (name, values_sha256).
    """

    if factor_name not in FACTOR_NAMES:
        raise ValueError(
            f"unknown major_news mention factor {factor_name!r}; "
            f"expected one of {list(FACTOR_NAMES)}"
        )

    def build_metadata(manifest: dict[str, Any], values_sha256: str) -> ExternalFactorMetadata:
        metadata = _mentions_metadata(factor_name, manifest, values_sha256)
        metadata.variables["manifest"] = str(factors_dir / f"{factor_name}.json")
        return metadata

    return register_external_factor(
        store,
        factors_dir,
        factor_name=factor_name,
        run_kind=IMPORT_RUN_KIND,
        actor=actor,
        build_metadata=build_metadata,
        source_dataset=SOURCE_DATASET,
        required_source_keys=("prompt_version", "model", "mention_rules_version"),
    )
