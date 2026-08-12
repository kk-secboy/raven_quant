"""Corpus NLP processing layer: Tushare text corpus -> LLM extraction -> PIT signal fields.

Reads the already-downloaded Tushare text datasets — ``major_news`` long-form
market-level news, the ``irm_qa_sh``/``irm_qa_sz`` exchange interaction Q&A, and
the policy-signal corpora ``npr`` (国家政策法规库, exact ``pubtime``) and
``cctv_news`` (新闻联播文字稿, date-only) — from the units/snapshots dual
layout, normalizes every row into a unified text item (source_dataset / item_id
/ ts_code / title / content / pub_time), and calls an OpenAI-compatible chat
endpoint for structured sentiment/topic extraction.
Outputs land under ``data/corpus_nlp/`` following the ``announcement_nlp`` layout:

- ``units/fields_<timestamp>_<uuid>.parquet`` — immutable per-run field units
- ``fields.parquet`` — derived structured-fields index with row-level
  ``available_at``/``ingested_at`` PIT timestamps
- ``state.parquet`` — processing ledger keyed by content sha256 +
  prompt_version + model for idempotent reruns and audit
- ``factors/<name>.parquet`` + ``factors/<name>.json`` — factor-values
  artifacts with sha256 manifests: ``news_sentiment_daily`` (market level,
  pseudo-instrument ``MARKET``), ``irm_qa_sentiment_daily`` (per ts_code) and
  ``policy_sentiment_daily`` (market level, over npr + cctv_news)

PIT semantics (design draft 3.3):

- ``major_news``/``npr`` carry an exact publication moment (``pub_time`` /
  ``pubtime``); ``available_at`` = that moment.
- ``irm_qa_*``/``cctv_news`` carry only a date; ``available_at`` conservatively
  moves to the next trading day from the persisted trade_cal (never weekday
  guesses).
- Daily factor datetime: ``available_at`` before 15:00 on a trading day maps to
  that day, otherwise to the next trade_cal trading day.
- ``ingested_at`` is the processing moment of each row.

npr caveat: ``content_html`` is not a default field of the Tushare interface,
so downloads typically carry title + 发文字号/发文机关 only; items then hold
title-level text (the LLM sees the policy title, which for 法规/批复 titles is
the informative part). Rows are still PIT-exact on ``pubtime``.

This module lives in quant_platform (not quant_data) for the same reason as
``announcement_nlp``: it consumes the platform LLM secret store and the platform
factor-series contract.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from bisect import bisect_left
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from quant_data.cninfo_announcements import (
    _parquet_files,
    _read_parquet_union,
    load_trade_calendar_open_days,
    next_trading_day,
)
from quant_data.rate_limit import GlobalRateGate

from .announcement_factor_registry import (
    ExternalFactorMetadata,
    register_external_factor,
)
from .announcement_nlp import (
    MAX_TEXT_CHARS,
    ChatCompleter,
    LlmCredentials,
    LlmExtractionError,
    OpenAIChatClient,
    SecretStoreLike,
    _bounded_float,
    _load_records,
    _sha256_file,
    _write_json_atomic,
    _write_parquet_atomic,
    load_llm_credentials,
)
from .factor_evaluator import normalize_series

PROMPT_VERSION = "corpus-nlp.v2"
DEFAULT_BATCH_SIZE = 40
MAX_BATCH_SIZE = 100
DEFAULT_WORKERS = 4
MAX_WORKERS = 8
DEFAULT_BATCH_ITEM_CHARS = 1_000
DEFAULT_MAJOR_NEWS_PER_DAY = 40
DEFAULT_IRM_PER_INSTRUMENT_DAY = 2

TOPICS = ("macro", "policy", "industry", "company", "market", "other")

DATASET_MAJOR_NEWS = "major_news"
DATASET_IRM_QA_SH = "irm_qa_sh"
DATASET_IRM_QA_SZ = "irm_qa_sz"
DATASET_NPR = "npr"
DATASET_CCTV_NEWS = "cctv_news"
IRM_QA_DATASETS = (DATASET_IRM_QA_SH, DATASET_IRM_QA_SZ)
POLICY_DATASETS = (DATASET_NPR, DATASET_CCTV_NEWS)
SUPPORTED_CORPUS_DATASETS = (DATASET_MAJOR_NEWS, *POLICY_DATASETS, *IRM_QA_DATASETS)
# npr has no persisted production rows as of the audited 2026-08-08 source
# boundary. It remains explicitly supported for future/native backfills, but a
# default production run must not fail merely because that unavailable source
# is absent. Its absence is documented as a source gap rather than fabricated.
DEFAULT_CORPUS_DATASETS = (
    DATASET_MAJOR_NEWS,
    DATASET_CCTV_NEWS,
    *IRM_QA_DATASETS,
)
# Datasets whose rows carry an exact publication moment; every other supported
# dataset is date-only and conservatively visible from the next trading day.
EXACT_TIMESTAMP_DATASETS = (DATASET_MAJOR_NEWS, DATASET_NPR)

CORPUS_NLP_DIR = "corpus_nlp"
# Pseudo-instrument code for market-level series that carry no ts_code; it can
# never collide with a real Tushare code (those look like 000001.SZ).
MARKET_INSTRUMENT = "MARKET"
TRADING_DAY_CUTOFF = time(15, 0)

NEWS_FACTOR_NAME = "news_sentiment_daily"
IRM_QA_FACTOR_NAME = "irm_qa_sentiment_daily"
POLICY_FACTOR_NAME = "policy_sentiment_daily"
CORPUS_FACTOR_NAMES = (NEWS_FACTOR_NAME, IRM_QA_FACTOR_NAME, POLICY_FACTOR_NAME)
AVAILABILITY_POLICY = {
    NEWS_FACTOR_NAME: (
        "available_at=pub_time (exact publication moment); daily factor date = pub date "
        "when it is a trade_cal trading day and available before 15:00, otherwise the "
        "next trade_cal trading day"
    ),
    IRM_QA_FACTOR_NAME: (
        "available_at=next trade_cal trading day after trade_date (date-only source, "
        "conservative rule of design draft 3.3); daily factor date = available_at date"
    ),
    POLICY_FACTOR_NAME: (
        "npr: available_at=pubtime (exact publication moment; downloads usually carry "
        "title-level text only, content_html is not a default interface field); "
        "cctv_news: available_at=next trade_cal trading day after the broadcast date "
        "(date-only source, conservative); daily factor date = available date when "
        "available before 15:00 on a trade_cal trading day, otherwise the next "
        "trade_cal trading day"
    ),
}

FIELDS_COLUMNS = (
    "process_key",
    "source_dataset",
    "item_id",
    "ts_code",
    "pub_time",
    "available_at",
    "ingested_at",
    "sentiment",
    "topic",
    "confidence",
    "model",
    "prompt_version",
    "processed_at",
)
STATE_COLUMNS = (
    "process_key",
    "item_id",
    "source_dataset",
    "prompt_version",
    "model",
    "status",
    "stage",
    "error",
    "processed_at",
    "ts_code",
    "available_at",
)


@dataclass(frozen=True, slots=True)
class CorpusItem:
    """One normalized corpus text item ready for LLM extraction."""

    source_dataset: str
    item_id: str  # content sha256 over the normalized semantic payload
    ts_code: str | None  # None for market-level items (major_news)
    title: str
    content: str
    pub_time: datetime  # tz-naive China time; date-only sources use midnight


def _item_id(
    source_dataset: str, ts_code: str | None, title: str, content: str, pub_time: datetime
) -> str:
    payload = json.dumps(
        {
            "source_dataset": source_dataset,
            "ts_code": ts_code or "",
            "title": title,
            "content": content,
            "pub_time": pub_time.isoformat(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _corpus_parquet_paths(data_root: Path, datasets: Iterable[str]) -> dict[str, list[str]]:
    """Resolve units/snapshots parquet per dataset; fail closed listing all gaps."""

    found: dict[str, list[str]] = {}
    missing: list[str] = []
    for dataset in datasets:
        paths = _parquet_files(data_root, dataset)
        if paths:
            found[dataset] = paths
        else:
            missing.append(dataset)
    if missing:
        details = "; ".join(
            f"{name} (looked under {data_root / 'units' / name} and "
            f"{data_root / 'snapshots'}/*/parquet/{name})"
            for name in missing
        )
        raise RuntimeError(
            f"corpus parquet is unavailable for: {details}; run the corresponding "
            "Tushare download tasks first (cn_institutional for major_news, "
            "research_corpus for irm_qa_sh/irm_qa_sz)"
        )
    return found


def _available_columns(paths: list[str]) -> set[str]:
    frame = _read_parquet_union(paths, "SELECT * FROM read_parquet(?, union_by_name=true) LIMIT 0")
    return set(frame.columns)


def _date_bounds_sql(
    expression: str, *, start: date | None, end: date | None
) -> tuple[str, list[date]]:
    conditions: list[str] = []
    parameters: list[date] = []
    if start is not None:
        conditions.append(f"CAST({expression} AS DATE) >= ?")
        parameters.append(start)
    if end is not None:
        conditions.append(f"CAST({expression} AS DATE) <= ?")
        parameters.append(end)
    return (" WHERE " + " AND ".join(conditions) if conditions else "", parameters)


def _bounded_year_windows(
    start: date | None, end: date | None
) -> list[tuple[date | None, date | None]]:
    """Split a bounded historical scan into year-sized DuckDB queries.

    The production corpus contains many duplicate full-text rows across unit and
    snapshot parquet files.  A single multi-year ``DISTINCT`` + window query can
    exhaust both memory and DuckDB's temporary spill allowance before the
    governed daily cap is applied.  Publication dates partition the selection
    contract, so independent non-overlapping year windows are exactly equivalent
    while keeping each hash/sort bounded.
    """

    if start is None or end is None or start.year == end.year:
        return [(start, end)]
    windows: list[tuple[date | None, date | None]] = []
    year = start.year
    while year <= end.year:
        window_start = start if year == start.year else date(year, 1, 1)
        window_end = end if year == end.year else date(year, 12, 31)
        windows.append((window_start, window_end))
        year += 1
    return windows


def _major_news_items(
    paths: list[str],
    *,
    start: date | None = None,
    end: date | None = None,
    max_items_per_day: int | None = None,
) -> list[CorpusItem]:
    columns = _available_columns(paths)
    missing = sorted({"title", "content", "pub_time"} - columns)
    if missing:
        raise RuntimeError(f"major_news parquet misses required columns: {missing}")
    moment_expr = "try_cast(CAST(pub_time AS VARCHAR) AS TIMESTAMP)"
    if max_items_per_day is not None and max_items_per_day < 1:
        raise ValueError("max_items_per_day must be positive")
    frames: list[pd.DataFrame] = []
    for window_start, window_end in _bounded_year_windows(start, end):
        where, parameters = _date_bounds_sql(moment_expr, start=window_start, end=window_end)
        select_sql = f"""
            SELECT DISTINCT
                CAST(title AS VARCHAR) AS title,
                CAST(content AS VARCHAR) AS content,
                CAST(pub_time AS VARCHAR) AS pub_time,
                {moment_expr} AS moment
            FROM read_parquet(?, union_by_name=true)
            {where}
        """
        if max_items_per_day is not None:
            frames.append(
                _read_parquet_union(
                    paths,
                    f"""
                    WITH source AS ({select_sql}), ranked AS (
                        SELECT *, row_number() OVER (
                            PARTITION BY CAST(moment AS DATE)
                            ORDER BY hash(title, content, pub_time), title, pub_time
                        ) AS selection_rank
                        FROM source
                        WHERE moment IS NOT NULL
                    )
                    SELECT title, content, pub_time
                    FROM ranked
                    WHERE selection_rank <= ?
                    """,
                    [*parameters, max_items_per_day],
                )
            )
        else:
            frames.append(
                _read_parquet_union(
                    paths,
                    f"SELECT title, content, pub_time FROM ({select_sql}) AS source",
                    parameters,
                )
            )
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    frame["moment"] = pd.to_datetime(frame["pub_time"], errors="coerce")
    items: list[CorpusItem] = []
    for row in frame.itertuples():
        # Rows without a usable timestamp or any text cannot produce a signal;
        # drop them with the same discipline as the availability read guard.
        if pd.isna(row.moment):
            continue
        title = "" if row.title is None else str(row.title).strip()
        content = "" if row.content is None else str(row.content).strip()
        if not (title or content):
            continue
        moment = pd.Timestamp(row.moment).to_pydatetime()
        items.append(
            CorpusItem(
                source_dataset=DATASET_MAJOR_NEWS,
                item_id=_item_id(DATASET_MAJOR_NEWS, None, title, content, moment),
                ts_code=None,
                title=title,
                content=content,
                pub_time=moment,
            )
        )
    return items


def _irm_qa_items(
    paths: list[str],
    dataset: str,
    *,
    start: date | None = None,
    end: date | None = None,
    max_items_per_instrument_day: int | None = None,
) -> list[CorpusItem]:
    columns = _available_columns(paths)
    question_col = next((name for name in ("q", "question") if name in columns), None)
    answer_col = next((name for name in ("a", "answer") if name in columns), None)
    missing = sorted({"trade_date", "ts_code"} - columns)
    if question_col is None:
        missing.append("q|question")
    if missing:
        raise RuntimeError(f"{dataset} parquet misses required columns: {missing}")
    trade_date_expr = (
        "coalesce(try_cast(trade_date AS DATE), "
        "try_strptime(CAST(trade_date AS VARCHAR), '%Y%m%d')::DATE)"
    )
    select_parts = [
        "CAST(ts_code AS VARCHAR) AS ts_code",
        f"{trade_date_expr} AS trade_date",
        f'CAST("{question_col}" AS VARCHAR) AS question',
    ]
    if answer_col is not None:
        # Tolerant: the answer column is optional; questions alone still carry signal.
        select_parts.append(f'CAST("{answer_col}" AS VARCHAR) AS answer')
    if max_items_per_instrument_day is not None and max_items_per_instrument_day < 1:
        raise ValueError("max_items_per_instrument_day must be positive")
    frames: list[pd.DataFrame] = []
    for window_start, window_end in _bounded_year_windows(start, end):
        where, parameters = _date_bounds_sql(trade_date_expr, start=window_start, end=window_end)
        select_sql = (
            f"SELECT DISTINCT {', '.join(select_parts)} "
            f"FROM read_parquet(?, union_by_name=true){where}"
        )
        if max_items_per_instrument_day is not None:
            order_columns = "question" + (", answer" if answer_col is not None else "")
            frames.append(
                _read_parquet_union(
                    paths,
                    f"""
                    WITH source AS ({select_sql}), ranked AS (
                        SELECT *, row_number() OVER (
                            PARTITION BY trade_date, upper(trim(ts_code))
                            ORDER BY hash({order_columns}), question
                        ) AS selection_rank
                        FROM source
                        WHERE trade_date IS NOT NULL AND trim(ts_code) <> ''
                    )
                    SELECT * EXCLUDE (selection_rank)
                    FROM ranked
                    WHERE selection_rank <= ?
                    """,
                    [*parameters, max_items_per_instrument_day],
                )
            )
        else:
            frames.append(_read_parquet_union(paths, select_sql, parameters))
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    items: list[CorpusItem] = []
    for row in frame.itertuples():
        if pd.isna(row.trade_date):
            continue
        ts_code = "" if row.ts_code is None else str(row.ts_code).strip().upper()
        question = "" if row.question is None else str(row.question).strip()
        if not (ts_code and question):
            continue
        answer = "" if getattr(row, "answer", None) is None else str(row.answer).strip()
        content = f"问：{question}\n答：{answer}" if answer else f"问：{question}"
        moment = datetime.combine(pd.Timestamp(row.trade_date).date(), time.min)
        items.append(
            CorpusItem(
                source_dataset=dataset,
                item_id=_item_id(dataset, ts_code, question[:80], content, moment),
                ts_code=ts_code,
                title=question[:80],
                content=content,
                pub_time=moment,
            )
        )
    return items


def _npr_items(
    paths: list[str], *, start: date | None = None, end: date | None = None
) -> list[CorpusItem]:
    """Normalize the 国家政策法规库 rows; exact pubtime, tolerant content column.

    The interface defaults to pubtime/title/pcode/puborg/ptype; ``content_html``
    (or ``content``) is optional, so items commonly hold title-level text only.
    """

    columns = _available_columns(paths)
    time_cols = [name for name in ("pubtime", "pub_time") if name in columns]
    content_cols = [name for name in ("content_html", "content") if name in columns]
    missing = sorted({"title"} - columns)
    if not time_cols:
        missing.append("pubtime|pub_time")
    if missing:
        raise RuntimeError(f"npr parquet misses required columns: {missing}")
    # Both naming variants can coexist across download windows (union_by_name);
    # coalesce per row instead of preferring one column for every row.
    time_expr = "coalesce(" + ", ".join(f'CAST("{name}" AS VARCHAR)' for name in time_cols) + ")"
    select_parts = [
        "CAST(title AS VARCHAR) AS title",
        f"{time_expr} AS pub_time",
    ]
    if content_cols:
        content_expr = (
            "coalesce(" + ", ".join(f'CAST("{name}" AS VARCHAR)' for name in content_cols) + ")"
        )
        select_parts.append(f"{content_expr} AS content")
    moment_expr = f"try_cast({time_expr} AS TIMESTAMP)"
    where, parameters = _date_bounds_sql(moment_expr, start=start, end=end)
    frame = _read_parquet_union(
        paths,
        f"SELECT {', '.join(select_parts)} FROM read_parquet(?, union_by_name=true){where}",
        parameters,
    )
    frame["moment"] = pd.to_datetime(frame["pub_time"], errors="coerce")
    items: list[CorpusItem] = []
    for row in frame.itertuples():
        if pd.isna(row.moment):
            continue
        title = "" if row.title is None else str(row.title).strip()
        body = "" if getattr(row, "content", None) is None else str(row.content).strip()
        if not title:
            continue
        # Title-level fallback: policy titles are the informative part when the
        # interface default fields carry no 正文.
        content = body or title
        moment = pd.Timestamp(row.moment).to_pydatetime()
        items.append(
            CorpusItem(
                source_dataset=DATASET_NPR,
                item_id=_item_id(DATASET_NPR, None, title, content, moment),
                ts_code=None,
                title=title,
                content=content,
                pub_time=moment,
            )
        )
    return items


def _cctv_news_items(
    paths: list[str], *, start: date | None = None, end: date | None = None
) -> list[CorpusItem]:
    """Normalize the 新闻联播文字稿 rows (date-only, conservative availability)."""

    columns = _available_columns(paths)
    missing = sorted({"date", "title", "content"} - columns)
    if missing:
        raise RuntimeError(f"cctv_news parquet misses required columns: {missing}")
    broadcast_expr = (
        "coalesce(try_cast(date AS DATE), try_strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE)"
    )
    where, parameters = _date_bounds_sql(broadcast_expr, start=start, end=end)
    frame = _read_parquet_union(
        paths,
        f"""
        SELECT
            {broadcast_expr} AS broadcast_date,
            CAST(title AS VARCHAR) AS title,
            CAST(content AS VARCHAR) AS content
        FROM read_parquet(?, union_by_name=true)
        {where}
        """,
        parameters,
    )
    items: list[CorpusItem] = []
    for row in frame.itertuples():
        if pd.isna(row.broadcast_date):
            continue
        title = "" if row.title is None else str(row.title).strip()
        content = "" if row.content is None else str(row.content).strip()
        if not (title or content):
            continue
        moment = datetime.combine(pd.Timestamp(row.broadcast_date).date(), time.min)
        items.append(
            CorpusItem(
                source_dataset=DATASET_CCTV_NEWS,
                item_id=_item_id(DATASET_CCTV_NEWS, None, title, content, moment),
                ts_code=None,
                title=title,
                content=content or title,
                pub_time=moment,
            )
        )
    return items


def load_corpus_items(
    data_root: Path,
    *,
    datasets: Iterable[str] | None = None,
    ts_codes: set[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    max_major_news_per_day: int | None = None,
    max_irm_per_instrument_day: int | None = None,
) -> list[CorpusItem]:
    """Read corpus parquets from the units/snapshots layout into unified items.

    Fail-closed: raises when any requested dataset has no parquet, naming the
    missing datasets and the paths that were checked. Duplicate rows that live
    in both layouts collapse onto one item via the content sha256 item_id.
    """

    wanted = set(datasets) if datasets else set(SUPPORTED_CORPUS_DATASETS)
    unknown = sorted(wanted - set(SUPPORTED_CORPUS_DATASETS))
    if unknown:
        raise ValueError(f"unsupported corpus dataset: {unknown}")
    ordered = [name for name in SUPPORTED_CORPUS_DATASETS if name in wanted]
    paths_by_dataset = _corpus_parquet_paths(data_root, ordered)

    items: dict[str, CorpusItem] = {}
    loaders = {
        DATASET_MAJOR_NEWS: _major_news_items,
        DATASET_NPR: _npr_items,
        DATASET_CCTV_NEWS: _cctv_news_items,
    }
    for dataset in ordered:
        loader = loaders.get(dataset)
        rows = (
            loader(
                paths_by_dataset[dataset],
                start=start,
                end=end,
                **(
                    {"max_items_per_day": max_major_news_per_day}
                    if dataset == DATASET_MAJOR_NEWS
                    else {}
                ),
            )
            if loader is not None
            else _irm_qa_items(
                paths_by_dataset[dataset],
                dataset,
                start=start,
                end=end,
                max_items_per_instrument_day=max_irm_per_instrument_day,
            )
        )
        for item in rows:
            items.setdefault(item.item_id, item)

    result = list(items.values())
    if ts_codes:
        # A ts_code filter selects instrument-level items only; market-level
        # major_news rows carry no ts_code and are excluded.
        codes = {code.strip().upper() for code in ts_codes if code.strip()}
        result = [item for item in result if item.ts_code is not None and item.ts_code in codes]
    if start is not None:
        result = [item for item in result if item.pub_time.date() >= start]
    if end is not None:
        result = [item for item in result if item.pub_time.date() <= end]
    return sorted(result, key=lambda item: (item.pub_time, item.source_dataset, item.item_id))


def _eligible_corpus_stats(
    paths: list[str],
    dataset: str,
    *,
    start: date | None,
    end: date | None,
    ts_codes: set[str] | None,
) -> dict[str, Any]:
    """Count normalized unique source items without materializing corpus text.

    The query mirrors each loader's validity and de-duplication contract.  It
    lets a production run disclose its denominator and sampling ratio without
    loading the uncapped multi-million-row corpus into Python memory.
    """

    columns = _available_columns(paths)
    codes = {code.strip().upper() for code in (ts_codes or set()) if code.strip()}
    if codes and dataset not in IRM_QA_DATASETS:
        return {
            "eligible_unique_items": 0,
            "date_min": None,
            "date_max": None,
        }

    if dataset == DATASET_MAJOR_NEWS:
        missing = sorted({"title", "content", "pub_time"} - columns)
        if missing:
            raise RuntimeError(f"major_news parquet misses required columns: {missing}")
        moment_expr = "try_cast(CAST(pub_time AS VARCHAR) AS TIMESTAMP)"
        major_frames: list[pd.DataFrame] = []
        for window_start, window_end in _bounded_year_windows(start, end):
            where, parameters = _date_bounds_sql(moment_expr, start=window_start, end=window_end)
            query = f"""
                WITH source AS (
                    SELECT DISTINCT
                        CAST(title AS VARCHAR) AS title,
                        CAST(content AS VARCHAR) AS content,
                        CAST(pub_time AS VARCHAR) AS pub_time,
                        {moment_expr} AS moment
                    FROM read_parquet(?, union_by_name=true)
                    {where}
                ), eligible AS (
                    SELECT * FROM source
                    WHERE moment IS NOT NULL
                      AND (trim(coalesce(title, '')) <> '' OR trim(coalesce(content, '')) <> '')
                )
                SELECT count(*) AS eligible_unique_items,
                       min(CAST(moment AS DATE)) AS date_min,
                       max(CAST(moment AS DATE)) AS date_max
                FROM eligible
            """
            major_frames.append(_read_parquet_union(paths, query, parameters))
        if len(major_frames) > 1:
            combined = pd.concat(major_frames, ignore_index=True)
            valid_min = pd.to_datetime(combined["date_min"], errors="coerce").dropna()
            valid_max = pd.to_datetime(combined["date_max"], errors="coerce").dropna()
            return {
                "eligible_unique_items": int(combined["eligible_unique_items"].sum()),
                "date_min": valid_min.min().date().isoformat() if not valid_min.empty else None,
                "date_max": valid_max.max().date().isoformat() if not valid_max.empty else None,
            }
        frame = major_frames[0]
    elif dataset in IRM_QA_DATASETS:
        question_col = next((name for name in ("q", "question") if name in columns), None)
        answer_col = next((name for name in ("a", "answer") if name in columns), None)
        missing = sorted({"trade_date", "ts_code"} - columns)
        if question_col is None:
            missing.append("q|question")
        if missing:
            raise RuntimeError(f"{dataset} parquet misses required columns: {missing}")
        trade_date_expr = (
            "coalesce(try_cast(trade_date AS DATE), "
            "try_strptime(CAST(trade_date AS VARCHAR), '%Y%m%d')::DATE)"
        )
        answer_select = f', CAST("{answer_col}" AS VARCHAR) AS answer' if answer_col else ""
        grouped_frames: list[pd.DataFrame] = []
        for window_start, window_end in _bounded_year_windows(start, end):
            where, parameters = _date_bounds_sql(
                trade_date_expr, start=window_start, end=window_end
            )
            query = f"""
                WITH source AS (
                    SELECT DISTINCT
                        upper(trim(CAST(ts_code AS VARCHAR))) AS ts_code,
                        {trade_date_expr} AS trade_date,
                        CAST("{question_col}" AS VARCHAR) AS question
                        {answer_select}
                    FROM read_parquet(?, union_by_name=true)
                    {where}
                ), eligible AS (
                    SELECT * FROM source
                    WHERE trade_date IS NOT NULL
                      AND ts_code <> ''
                      AND trim(coalesce(question, '')) <> ''
                )
                SELECT ts_code, count(*) AS eligible_unique_items,
                       min(trade_date) AS date_min, max(trade_date) AS date_max
                FROM eligible
                GROUP BY ts_code
            """
            grouped_frames.append(_read_parquet_union(paths, query, parameters))
        grouped = (
            pd.concat(grouped_frames, ignore_index=True)
            if len(grouped_frames) > 1
            else grouped_frames[0]
        )
        if codes:
            grouped = grouped[grouped["ts_code"].astype(str).isin(codes)]
        if grouped.empty:
            return {
                "eligible_unique_items": 0,
                "date_min": None,
                "date_max": None,
            }
        return {
            "eligible_unique_items": int(grouped["eligible_unique_items"].sum()),
            "date_min": pd.to_datetime(grouped["date_min"]).min().date().isoformat(),
            "date_max": pd.to_datetime(grouped["date_max"]).max().date().isoformat(),
        }
    elif dataset == DATASET_NPR:
        time_cols = [name for name in ("pubtime", "pub_time") if name in columns]
        content_cols = [name for name in ("content_html", "content") if name in columns]
        missing = sorted({"title"} - columns)
        if not time_cols:
            missing.append("pubtime|pub_time")
        if missing:
            raise RuntimeError(f"npr parquet misses required columns: {missing}")
        time_expr = (
            "coalesce(" + ", ".join(f'CAST("{name}" AS VARCHAR)' for name in time_cols) + ")"
        )
        content_expr = (
            "coalesce(" + ", ".join(f'CAST("{name}" AS VARCHAR)' for name in content_cols) + ")"
            if content_cols
            else "NULL::VARCHAR"
        )
        moment_expr = f"try_cast({time_expr} AS TIMESTAMP)"
        where, parameters = _date_bounds_sql(moment_expr, start=start, end=end)
        query = f"""
            WITH source AS (
                SELECT DISTINCT CAST(title AS VARCHAR) AS title,
                       {time_expr} AS pub_time, {content_expr} AS content,
                       {moment_expr} AS moment
                FROM read_parquet(?, union_by_name=true)
                {where}
            ), eligible AS (
                SELECT * FROM source
                WHERE moment IS NOT NULL AND trim(coalesce(title, '')) <> ''
            )
            SELECT count(*) AS eligible_unique_items,
                   min(CAST(moment AS DATE)) AS date_min,
                   max(CAST(moment AS DATE)) AS date_max
            FROM eligible
        """
    elif dataset == DATASET_CCTV_NEWS:
        missing = sorted({"date", "title", "content"} - columns)
        if missing:
            raise RuntimeError(f"cctv_news parquet misses required columns: {missing}")
        date_expr = (
            "coalesce(try_cast(date AS DATE), try_strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE)"
        )
        where, parameters = _date_bounds_sql(date_expr, start=start, end=end)
        query = f"""
            WITH source AS (
                SELECT DISTINCT {date_expr} AS broadcast_date,
                       CAST(title AS VARCHAR) AS title,
                       CAST(content AS VARCHAR) AS content
                FROM read_parquet(?, union_by_name=true)
                {where}
            ), eligible AS (
                SELECT * FROM source
                WHERE broadcast_date IS NOT NULL
                  AND (trim(coalesce(title, '')) <> '' OR trim(coalesce(content, '')) <> '')
            )
            SELECT count(*) AS eligible_unique_items,
                   min(broadcast_date) AS date_min,
                   max(broadcast_date) AS date_max
            FROM eligible
        """
    else:  # guarded by SUPPORTED_CORPUS_DATASETS at the public boundary
        raise ValueError(f"unsupported corpus dataset: {dataset}")

    if dataset != DATASET_MAJOR_NEWS:
        frame = _read_parquet_union(paths, query, parameters)
    row = frame.iloc[0]
    return {
        "eligible_unique_items": int(row["eligible_unique_items"]),
        "date_min": (
            pd.Timestamp(row["date_min"]).date().isoformat()
            if not pd.isna(row["date_min"])
            else None
        ),
        "date_max": (
            pd.Timestamp(row["date_max"]).date().isoformat()
            if not pd.isna(row["date_max"])
            else None
        ),
    }


def build_corpus_selection_audit(
    data_root: Path,
    *,
    selected_datasets: set[str],
    selected_before_limit: Sequence[CorpusItem],
    selected_after_limit: Sequence[CorpusItem],
    ts_codes: set[str] | None,
    start: date | None,
    end: date | None,
    limit: int | None,
    max_major_news_per_day: int | None,
    max_irm_per_instrument_day: int | None,
) -> dict[str, Any]:
    """Describe raw eligible denominators and governed sampling outcomes."""

    ordered = [name for name in SUPPORTED_CORPUS_DATASETS if name in selected_datasets]
    paths_by_dataset = _corpus_parquet_paths(data_root, ordered)
    before_counts = Counter(item.source_dataset for item in selected_before_limit)
    after_counts = Counter(item.source_dataset for item in selected_after_limit)
    sources: dict[str, dict[str, Any]] = {}
    for dataset in ordered:
        stats = _eligible_corpus_stats(
            paths_by_dataset[dataset],
            dataset,
            start=start,
            end=end,
            ts_codes=ts_codes,
        )
        eligible = int(stats["eligible_unique_items"])
        selected_before = int(before_counts[dataset])
        selected_after = int(after_counts[dataset])
        sources[dataset] = {
            **stats,
            "selected_after_policy_before_limit": selected_before,
            "selected_after_limit": selected_after,
            "policy_selection_ratio": (
                selected_before / eligible if eligible else 1.0 if selected_before == 0 else 0.0
            ),
            "final_selection_ratio": (
                selected_after / eligible if eligible else 1.0 if selected_after == 0 else 0.0
            ),
        }
    return {
        "scope": {
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None,
            "ts_codes": sorted(ts_codes or set()),
            "limit": int(limit or 0),
        },
        "policy": {
            "major_news_max_items_per_publication_day": max_major_news_per_day,
            "irm_max_items_per_instrument_publication_day": max_irm_per_instrument_day,
            "selection": "stable DuckDB hash rank; no post-event performance input",
        },
        "eligible_unique_items": sum(
            int(value["eligible_unique_items"]) for value in sources.values()
        ),
        "selected_after_policy_before_limit": len(selected_before_limit),
        "selected_after_limit": len(selected_after_limit),
        "sources": sources,
    }


def available_at_for(item: CorpusItem, open_days: Sequence[date]) -> datetime:
    """Field-level visibility timestamp (design draft 3.3).

    major_news/npr have an exact publication moment; date-only irm_qa/cctv_news
    sources conservatively become visible on the next trade_cal trading day.
    """

    if item.source_dataset in EXACT_TIMESTAMP_DATASETS:
        return item.pub_time
    return datetime.combine(next_trading_day(item.pub_time.date(), open_days), time.min)


def factor_date_for(available_at: datetime, open_days: Sequence[date]) -> date:
    """Daily factor datetime for a visibility timestamp.

    available_at before 15:00 on a trade_cal trading day maps to that day;
    anything later (or on a non-trading day) rolls to the next trade_cal
    trading day. The calendar is the only source of trading days — never
    weekday rules.
    """

    day = available_at.date()
    index = bisect_left(open_days, day)
    is_open = index < len(open_days) and open_days[index] == day
    if is_open and available_at.time() < TRADING_DAY_CUTOFF:
        return day
    return next_trading_day(day, open_days)


def build_extraction_messages(*, item: CorpusItem, text: str) -> list[dict[str, str]]:
    """Versioned prompt for the structured corpus extraction."""

    system = (
        f"You are an A-share text corpus analysis engine (prompt_version={PROMPT_VERSION}). "
        "Respond with exactly one JSON object and nothing else, with keys "
        '"sentiment", "topic", "confidence". '
        '"sentiment" is a float in [-1.0, 1.0] rating the market sentiment of the text '
        "(negative = bearish, positive = bullish). "
        f'"topic" must be one of: {", ".join(TOPICS)}. '
        '"confidence" is a float in [0.0, 1.0].'
    )
    scope = item.ts_code if item.ts_code else f"market-level ({MARKET_INSTRUMENT})"
    user = (
        f"source_dataset: {item.source_dataset}\n"
        f"scope: {scope}\n"
        f"pub_time: {item.pub_time.isoformat(sep=' ')}\n"
        f"title: {item.title}\n\n"
        f"text:\n{text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@dataclass(frozen=True, slots=True)
class CorpusExtraction:
    sentiment: float
    topic: str
    confidence: float


def parse_extraction_payload(raw: str) -> CorpusExtraction:
    """Validate the LLM JSON payload against the strict schema; fail closed.

    Required keys: sentiment ([-1, 1]), topic (fixed enum), confidence
    ([0, 1]). Unknown extra keys are tolerated but ignored; any missing key or
    type/range violation is a failure.
    """

    text = raw.strip()
    if text.startswith("```"):
        # Tolerate a single markdown fence around the JSON object.
        text = text.strip("`").removeprefix("json").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LlmExtractionError(
            f"LLM response is not valid JSON: {exc}", stage="llm_parse"
        ) from exc
    if not isinstance(payload, dict):
        raise LlmExtractionError("LLM response must be a JSON object", stage="llm_parse")
    required = {"sentiment", "topic", "confidence"}
    missing = sorted(required - set(payload))
    if missing:
        raise LlmExtractionError(f"LLM response misses keys: {missing}", stage="llm_parse")
    topic = payload.get("topic")
    if topic not in TOPICS:
        raise LlmExtractionError(
            f"topic must be one of {list(TOPICS)}; got {topic!r}",
            stage="llm_parse",
        )
    sentiment = _bounded_float(payload.get("sentiment"), "sentiment", -1.0, 1.0)
    confidence = _bounded_float(payload.get("confidence"), "confidence", 0.0, 1.0)
    return CorpusExtraction(sentiment=sentiment, topic=str(topic), confidence=confidence)


def build_batch_extraction_messages(
    items: Sequence[CorpusItem], *, max_chars: int = DEFAULT_BATCH_ITEM_CHARS
) -> list[dict[str, str]]:
    """Build one structured request for multiple independent corpus items."""

    if not items:
        raise ValueError("batch must contain at least one corpus item")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    payload = {
        "items": [
            {
                "item_id": item.item_id,
                "source_dataset": item.source_dataset,
                "scope": item.ts_code or f"market-level ({MARKET_INSTRUMENT})",
                "pub_time": item.pub_time.isoformat(sep=" "),
                "title": item.title[:240],
                "text": item.content[:max_chars],
            }
            for item in items
        ]
    }
    system = (
        f"You are an A-share text corpus analysis engine "
        f"(prompt_version={PROMPT_VERSION}). Return exactly one JSON object with "
        'an "items" array. Return exactly one output for every input item_id, '
        "with no duplicates or omissions. Each output must contain item_id, "
        'sentiment, topic, confidence. "sentiment" is a float in [-1.0, 1.0] '
        "(negative=bearish, positive=bullish); "
        f'"topic" must be one of: {", ".join(TOPICS)}; '
        '"confidence" is a float in [0.0, 1.0].'
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def parse_batch_extraction_payload(
    raw: str, *, expected_item_ids: Sequence[str]
) -> dict[str, CorpusExtraction]:
    """Validate a batch response exactly; any omission/duplicate fails the batch."""

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LlmExtractionError(
            f"LLM batch response is not valid JSON: {exc}", stage="llm_parse"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise LlmExtractionError(
            'LLM batch response must be an object with an "items" array',
            stage="llm_parse",
        )
    expected = list(expected_item_ids)
    if len(set(expected)) != len(expected):
        raise ValueError("expected_item_ids must not contain duplicates")
    results: dict[str, CorpusExtraction] = {}
    for row in payload["items"]:
        if not isinstance(row, dict):
            raise LlmExtractionError("LLM batch item must be a JSON object", stage="llm_parse")
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise LlmExtractionError("LLM batch item misses item_id", stage="llm_parse")
        if item_id in results:
            raise LlmExtractionError(
                f"LLM batch response duplicates item_id {item_id}", stage="llm_parse"
            )
        results[item_id] = parse_extraction_payload(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        )
    actual = set(results)
    expected_set = set(expected)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        unexpected = sorted(actual - expected_set)
        raise LlmExtractionError(
            f"LLM batch item_id mismatch; missing={missing}, unexpected={unexpected}",
            stage="llm_parse",
        )
    return results


def _empty_fields_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "process_key": pd.Series(dtype="string"),
            "source_dataset": pd.Series(dtype="string"),
            "item_id": pd.Series(dtype="string"),
            "ts_code": pd.Series(dtype="string"),
            "pub_time": pd.Series(dtype="datetime64[ns]"),
            "available_at": pd.Series(dtype="datetime64[ns]"),
            "ingested_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "sentiment": pd.Series(dtype="float64"),
            "topic": pd.Series(dtype="string"),
            "confidence": pd.Series(dtype="float64"),
            "model": pd.Series(dtype="string"),
            "prompt_version": pd.Series(dtype="string"),
            "processed_at": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )


def _fields_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return _empty_fields_frame()
    frame = pd.DataFrame(list(rows), columns=list(FIELDS_COLUMNS))
    frame["pub_time"] = pd.to_datetime(frame["pub_time"])
    frame["available_at"] = pd.to_datetime(frame["available_at"])
    frame["ingested_at"] = pd.to_datetime(frame["ingested_at"], utc=True)
    frame["processed_at"] = pd.to_datetime(frame["processed_at"], utc=True)
    return frame.sort_values(
        ["available_at", "source_dataset", "item_id"], kind="stable"
    ).reset_index(drop=True)


def _empty_state_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "process_key": pd.Series(dtype="string"),
            "item_id": pd.Series(dtype="string"),
            "source_dataset": pd.Series(dtype="string"),
            "prompt_version": pd.Series(dtype="string"),
            "model": pd.Series(dtype="string"),
            "status": pd.Series(dtype="string"),
            "stage": pd.Series(dtype="string"),
            "error": pd.Series(dtype="string"),
            "processed_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "ts_code": pd.Series(dtype="string"),
            "available_at": pd.Series(dtype="datetime64[ns]"),
        }
    )


def _state_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return _empty_state_frame()
    frame = pd.DataFrame(list(rows), columns=list(STATE_COLUMNS))
    frame["available_at"] = pd.to_datetime(frame["available_at"])
    frame["processed_at"] = pd.to_datetime(frame["processed_at"], utc=True)
    return frame.sort_values(["process_key"], kind="stable").reset_index(drop=True)


def _with_factor_date(frame: pd.DataFrame, open_days: Sequence[date]) -> pd.DataFrame:
    dated = frame.copy()
    dated["factor_date"] = dated["available_at"].map(
        lambda value: factor_date_for(pd.Timestamp(value).to_pydatetime(), open_days)
    )
    dated["sentiment"] = pd.to_numeric(dated["sentiment"], errors="coerce")
    return dated


def build_news_sentiment_series(
    fields: pd.DataFrame, open_days: Sequence[date], name: str = NEWS_FACTOR_NAME
) -> pd.Series:
    """Market-level daily sentiment mean under the MARKET pseudo-instrument."""

    frame = fields[fields["source_dataset"] == DATASET_MAJOR_NEWS]
    dated = _with_factor_date(frame, open_days)
    dated["instrument"] = MARKET_INSTRUMENT
    grouped = dated.groupby(["factor_date", "instrument"], sort=True)["sentiment"].mean()
    series = grouped.rename(name)
    series.index = series.index.set_names(["datetime", "instrument"])
    return normalize_series(series, name)


def build_irm_qa_sentiment_series(
    fields: pd.DataFrame, open_days: Sequence[date], name: str = IRM_QA_FACTOR_NAME
) -> pd.Series:
    """Per-instrument daily sentiment mean over the irm_qa_sh/irm_qa_sz fields."""

    frame = fields[fields["source_dataset"].isin(IRM_QA_DATASETS)]
    dated = _with_factor_date(frame, open_days)
    grouped = dated.groupby(["factor_date", "ts_code"], sort=True)["sentiment"].mean()
    series = grouped.rename(name)
    series.index = series.index.set_names(["datetime", "instrument"])
    return normalize_series(series, name)


def build_policy_sentiment_series(
    fields: pd.DataFrame, open_days: Sequence[date], name: str = POLICY_FACTOR_NAME
) -> pd.Series:
    """Market-level daily sentiment mean over the npr/cctv_news policy fields."""

    frame = fields[fields["source_dataset"].isin(POLICY_DATASETS)]
    dated = _with_factor_date(frame, open_days)
    dated["instrument"] = MARKET_INSTRUMENT
    grouped = dated.groupby(["factor_date", "instrument"], sort=True)["sentiment"].mean()
    series = grouped.rename(name)
    series.index = series.index.set_names(["datetime", "instrument"])
    return normalize_series(series, name)


def _write_factor_artifact(
    series: pd.Series,
    factors_dir: Path,
    *,
    name: str,
    source_datasets: tuple[str, ...],
    selection_policy: Mapping[str, int | None],
    selection_audit: Mapping[str, Any],
    source_scope: Mapping[str, Any],
    model: str,
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
            "dataset": "corpus_nlp_fields",
            "source_datasets": list(source_datasets),
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "selection_policy": dict(selection_policy),
            "selection_audit": dict(selection_audit),
            "scope": dict(source_scope),
        },
        "generated_at": now.isoformat(),
    }
    if name in (NEWS_FACTOR_NAME, POLICY_FACTOR_NAME):
        manifest["instrument_convention"] = (
            f"{MARKET_INSTRUMENT} is a pseudo-instrument code for market-level series "
            "that carry no ts_code"
        )
    manifest_path = factors_dir / f"{name}.json"
    _write_json_atomic(manifest, manifest_path)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "artifact_path": artifact_path,
    }


@dataclass(slots=True)
class CorpusNlpSummary:
    planned: int
    processed: int
    skipped: int
    failed: int
    llm_calls: int
    selection_policy: dict[str, int | None]
    selection_audit: dict[str, Any]
    fields_path: Path | None
    state_path: Path | None
    unit_path: Path | None
    factors: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "failed" if self.failed else "succeeded",
            "planned": self.planned,
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
            "llm_calls": self.llm_calls,
            "selection_policy": self.selection_policy,
            "selection_audit": self.selection_audit,
            "fields_path": str(self.fields_path) if self.fields_path else None,
            "state_path": str(self.state_path) if self.state_path else None,
            "unit_path": str(self.unit_path) if self.unit_path else None,
            "factors": {
                name: {
                    "manifest_path": str(entry["manifest_path"]),
                    "sha256": entry["manifest"]["sha256"],
                    "rows": entry["manifest"]["rows"],
                }
                for name, entry in self.factors.items()
            },
        }


def process_corpus(
    data_root: Path,
    *,
    datasets: Iterable[str] | None = None,
    ts_codes: set[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    limit: int | None = None,
    secret_store: SecretStoreLike | None = None,
    credentials: LlmCredentials | None = None,
    chat_client: ChatCompleter | None = None,
    rate_gate: GlobalRateGate | None = None,
    requests_per_minute: float = 30.0,
    timeout_seconds: float = 120.0,
    max_attempts: int = 3,
    max_chars: int = MAX_TEXT_CHARS,
    batch_size: int = 1,
    workers: int = DEFAULT_WORKERS,
    batch_item_chars: int = DEFAULT_BATCH_ITEM_CHARS,
    max_major_news_per_day: int | None = DEFAULT_MAJOR_NEWS_PER_DAY,
    max_irm_per_instrument_day: int | None = DEFAULT_IRM_PER_INSTRUMENT_DAY,
    checkpoint_every: int = 100,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    now: Callable[[], datetime] | None = None,
    environ: Mapping[str, str] | None = None,
) -> CorpusNlpSummary:
    """Run LLM sentiment/topic structuring over the downloaded text corpus.

    Idempotent on the processing key content-sha256+prompt_version+model: rows
    already succeeded are skipped and failed rows are re-attempted. Checkpoints
    persist state and successful fields without publishing factors, so a long
    interrupted run resumes without repeating completed LLM calls. Every
    failure is recorded in the state ledger and never produces a signal row
    (fail closed).
    """

    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}")
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    if batch_item_chars < 1:
        raise ValueError("batch_item_chars must be positive")
    if max_major_news_per_day is not None and max_major_news_per_day < 1:
        raise ValueError("max_major_news_per_day must be positive")
    if max_irm_per_instrument_day is not None and max_irm_per_instrument_day < 1:
        raise ValueError("max_irm_per_instrument_day must be positive")

    clock = now or (lambda: datetime.now(UTC))
    selected_datasets = set(datasets) if datasets else set(DEFAULT_CORPUS_DATASETS)
    items = load_corpus_items(
        data_root,
        datasets=selected_datasets,
        ts_codes=ts_codes,
        start=start,
        end=end,
        max_major_news_per_day=max_major_news_per_day,
        max_irm_per_instrument_day=max_irm_per_instrument_day,
    )
    selected_before_limit = list(items)
    if limit is not None and limit > 0:
        items = items[:limit]
    selection_audit = build_corpus_selection_audit(
        data_root,
        selected_datasets=selected_datasets,
        selected_before_limit=selected_before_limit,
        selected_after_limit=items,
        ts_codes=ts_codes,
        start=start,
        end=end,
        limit=limit,
        max_major_news_per_day=max_major_news_per_day,
        max_irm_per_instrument_day=max_irm_per_instrument_day,
    )
    # The trading calendar drives both irm_qa availability and every factor
    # date; without it the run must not guess (fail closed).
    open_days = load_trade_calendar_open_days(data_root)

    base = data_root / CORPUS_NLP_DIR
    units_dir = base / "units"
    factors_dir = base / "factors"
    fields_path = base / "fields.parquet"
    state_path = base / "state.parquet"
    units_dir.mkdir(parents=True, exist_ok=True)

    state = _load_records(state_path)
    fields_records = _load_records(fields_path)

    # Prove calendar coverage before the first LLM call or checkpoint write.
    # Date-only sources that cannot derive available_at remain auditable row
    # failures below; an otherwise visible item whose factor date is outside
    # the calendar is a run-level input defect and must fail with no partial
    # persistence (the historical fail-closed contract).
    for existing_field in fields_records.values():
        factor_date_for(pd.Timestamp(existing_field["available_at"]).to_pydatetime(), open_days)
    for item in items:
        try:
            candidate_available_at = available_at_for(item, open_days)
        except LookupError:
            continue
        factor_date_for(candidate_available_at, open_days)

    if credentials is None:
        credentials = load_llm_credentials(secret_store, environ=environ)
    model = credentials.chat_model
    if chat_client is None:
        chat_client = OpenAIChatClient(
            credentials,
            rate_gate=rate_gate or GlobalRateGate(requests_per_minute),
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
        )

    new_field_rows: list[dict] = []
    processed = skipped = failed = completed = llm_calls = 0
    dirty = False
    unit_path: Path | None = None
    last_checkpoint_completed = 0
    pending_batch: list[tuple[CorpusItem, str, datetime, datetime, dict[str, Any]]] = []
    inflight: dict[
        Future[
            tuple[
                dict[str, CorpusExtraction],
                dict[str, LlmExtractionError],
                int,
            ]
        ],
        list[tuple[CorpusItem, str, datetime, datetime, dict[str, Any]]],
    ] = {}

    def publish_progress() -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "planned": len(items),
                "completed": completed,
                "processed": processed,
                "skipped": skipped,
                "failed": failed,
                "llm_calls": llm_calls,
            }
        )

    def persist_checkpoint(*, force: bool = False) -> None:
        nonlocal dirty, new_field_rows, unit_path
        if dirty or force:
            if new_field_rows:
                unit_path = (
                    units_dir / f"fields_{clock():%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}.parquet"
                )
                _write_parquet_atomic(_fields_frame(new_field_rows), unit_path)
                new_field_rows = []
            _write_parquet_atomic(_fields_frame(list(fields_records.values())), fields_path)
            _write_parquet_atomic(_state_frame(list(state.values())), state_path)
            dirty = False
        publish_progress()

    def maybe_checkpoint() -> None:
        nonlocal last_checkpoint_completed
        if completed - last_checkpoint_completed >= checkpoint_every:
            persist_checkpoint()
            last_checkpoint_completed = completed

    def execute_batch(
        batch: list[tuple[CorpusItem, str, datetime, datetime, dict[str, Any]]],
    ) -> tuple[dict[str, CorpusExtraction], dict[str, LlmExtractionError], int]:
        items = [entry[0] for entry in batch]
        call_count = 1
        try:
            if batch_size == 1:
                item = items[0]
                messages = build_extraction_messages(item=item, text=item.content[:max_chars])
                results = {
                    item.item_id: parse_extraction_payload(
                        chat_client.complete(messages, model=model)
                    )
                }
            else:
                messages = build_batch_extraction_messages(
                    items, max_chars=min(max_chars, batch_item_chars)
                )
                results = parse_batch_extraction_payload(
                    chat_client.complete(messages, model=model),
                    expected_item_ids=[item.item_id for item in items],
                )
        except LlmExtractionError as batch_error:
            if len(items) == 1:
                return {}, {items[0].item_id: batch_error}, call_count
            # Batch responses occasionally omit or mutate one item id. Retrying
            # the affected batch item-by-item salvages valid source material and
            # keeps every individual failure explicit in the state ledger.
            results = {}
            errors: dict[str, LlmExtractionError] = {}
            for item in items:
                call_count += 1
                try:
                    messages = build_extraction_messages(item=item, text=item.content[:max_chars])
                    results[item.item_id] = parse_extraction_payload(
                        chat_client.complete(messages, model=model)
                    )
                except LlmExtractionError as exc:
                    errors[item.item_id] = exc
            return results, errors, call_count
        return results, {}, call_count

    def apply_completed(
        done: set[
            Future[
                tuple[
                    dict[str, CorpusExtraction],
                    dict[str, LlmExtractionError],
                    int,
                ]
            ]
        ],
    ) -> None:
        nonlocal processed, failed, completed, llm_calls, dirty
        for future in done:
            batch = inflight.pop(future)
            results, errors, batch_calls = future.result()
            llm_calls += batch_calls
            for item, process_key, available_at, processed_at, state_row in batch:
                result = results.get(item.item_id)
                if result is None:
                    exc = errors[item.item_id]
                    failed += 1
                    state_row.update(status="failed", stage=exc.stage, error=str(exc)[:500])
                    continue
                field_row = {
                    "process_key": process_key,
                    "source_dataset": item.source_dataset,
                    "item_id": item.item_id,
                    "ts_code": item.ts_code,
                    "pub_time": item.pub_time,
                    "available_at": available_at,
                    "ingested_at": processed_at,
                    "sentiment": result.sentiment,
                    "topic": result.topic,
                    "confidence": result.confidence,
                    "model": model,
                    "prompt_version": PROMPT_VERSION,
                    "processed_at": processed_at,
                }
                fields_records[process_key] = field_row
                new_field_rows.append(field_row)
                state_row.update(status="succeeded", stage="completed")
                processed += 1
            dirty = True
            completed += len(batch)
            maybe_checkpoint()

    def submit_pending(executor: ThreadPoolExecutor) -> None:
        nonlocal pending_batch
        if not pending_batch:
            return
        batch = pending_batch
        pending_batch = []
        inflight[executor.submit(execute_batch, batch)] = batch

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for item in items:
            process_key = f"{item.item_id}:{PROMPT_VERSION}:{model}"
            existing = state.get(process_key)
            if existing is not None and str(existing["status"]) == "succeeded":
                skipped += 1
                completed += 1
                maybe_checkpoint()
                continue
            processed_at = clock()
            state_row: dict[str, Any] = {
                "process_key": process_key,
                "item_id": item.item_id,
                "source_dataset": item.source_dataset,
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "status": "",
                "stage": "",
                "error": None,
                "processed_at": processed_at,
                "ts_code": item.ts_code,
                "available_at": None,
            }
            state[process_key] = state_row
            dirty = True
            try:
                available_at = available_at_for(item, open_days)
            except LookupError as exc:
                failed += 1
                state_row.update(status="failed", stage="availability", error=str(exc)[:500])
                completed += 1
                maybe_checkpoint()
            else:
                state_row["available_at"] = available_at
                pending_batch.append((item, process_key, available_at, processed_at, state_row))
                if len(pending_batch) >= batch_size:
                    submit_pending(executor)
                if len(inflight) >= workers * 2:
                    done, _pending = wait(set(inflight), return_when=FIRST_COMPLETED)
                    apply_completed(done)

        submit_pending(executor)
        while inflight:
            done, _pending = wait(set(inflight), return_when=FIRST_COMPLETED)
            apply_completed(done)

    persist_checkpoint(force=True)

    fields = _fields_frame(list(fields_records.values()))
    scope_process_keys = {f"{item.item_id}:{PROMPT_VERSION}:{model}" for item in items}
    scoped_fields = fields[
        (fields["prompt_version"].astype(str) == PROMPT_VERSION)
        & (fields["model"].astype(str) == model)
        & (fields["process_key"].astype(str).isin(scope_process_keys))
    ]
    selection_policy: dict[str, int | None] = {
        "major_news_max_items_per_publication_day": max_major_news_per_day,
        "irm_max_items_per_instrument_publication_day": max_irm_per_instrument_day,
    }
    source_scope = {
        "start_date": start.isoformat() if start else None,
        "end_date": end.isoformat() if end else None,
        "ts_codes": sorted(ts_codes or set()),
        "limit": int(limit or 0),
        "planned": len(items),
        "batch_size": batch_size,
        "workers": workers,
        "process_keys_sha256": hashlib.sha256(
            "\n".join(sorted(scope_process_keys)).encode("utf-8")
        ).hexdigest(),
        "process_key_count": len(scope_process_keys),
    }
    news_series = build_news_sentiment_series(scoped_fields, open_days)
    irm_qa_series = build_irm_qa_sentiment_series(scoped_fields, open_days)
    policy_series = build_policy_sentiment_series(scoped_fields, open_days)

    news_artifact = _write_factor_artifact(
        news_series,
        factors_dir,
        name=NEWS_FACTOR_NAME,
        source_datasets=tuple(
            dataset for dataset in (DATASET_MAJOR_NEWS,) if dataset in selected_datasets
        ),
        selection_policy=selection_policy,
        selection_audit=selection_audit,
        source_scope=source_scope,
        model=model,
        now=clock(),
    )
    irm_qa_artifact = _write_factor_artifact(
        irm_qa_series,
        factors_dir,
        name=IRM_QA_FACTOR_NAME,
        source_datasets=tuple(
            dataset for dataset in IRM_QA_DATASETS if dataset in selected_datasets
        ),
        selection_policy=selection_policy,
        selection_audit=selection_audit,
        source_scope=source_scope,
        model=model,
        now=clock(),
    )
    policy_artifact = _write_factor_artifact(
        policy_series,
        factors_dir,
        name=POLICY_FACTOR_NAME,
        source_datasets=tuple(
            dataset for dataset in POLICY_DATASETS if dataset in selected_datasets
        ),
        selection_policy=selection_policy,
        selection_audit=selection_audit,
        source_scope=source_scope,
        model=model,
        now=clock(),
    )
    return CorpusNlpSummary(
        planned=len(items),
        processed=processed,
        skipped=skipped,
        failed=failed,
        llm_calls=llm_calls,
        selection_policy=selection_policy,
        selection_audit=selection_audit,
        fields_path=fields_path,
        state_path=state_path,
        unit_path=unit_path,
        factors={
            NEWS_FACTOR_NAME: news_artifact,
            IRM_QA_FACTOR_NAME: irm_qa_artifact,
            POLICY_FACTOR_NAME: policy_artifact,
        },
    )


# ---------------------------------------------------------------------------
# Registration into factor_candidates (generic external-factor channel)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from .research_store import ResearchStore

IMPORT_RUN_KIND = "corpus_nlp_factor_import"
IMPORT_ACTOR = "corpus-nlp-registrar"
SOURCE_DATASET = "corpus_nlp_fields"

_FACTOR_DESCRIPTIONS = {
    NEWS_FACTOR_NAME: (
        "Market-level long-form news sentiment: daily mean LLM sentiment score "
        "of major_news items under the MARKET pseudo-instrument."
    ),
    IRM_QA_FACTOR_NAME: (
        "Exchange interaction Q&A sentiment: daily mean LLM sentiment score of "
        "irm_qa_sh/irm_qa_sz items per ts_code (sparse event shape)."
    ),
    POLICY_FACTOR_NAME: (
        "Policy-signal sentiment: daily mean LLM sentiment score over the npr "
        "(国家政策法规库, often title-level) and cctv_news (新闻联播) policy "
        "corpora under the MARKET pseudo-instrument."
    ),
}

_FACTOR_FORMULATIONS = {
    NEWS_FACTOR_NAME: (
        "mean(sentiment) over major_news fields grouped by (factor_date, MARKET); "
        "factor_date = pub date when available before 15:00 on a trade_cal "
        "trading day, otherwise the next trade_cal trading day"
    ),
    IRM_QA_FACTOR_NAME: (
        "mean(sentiment) over irm_qa_sh/irm_qa_sz fields grouped by "
        "(factor_date, ts_code); factor_date = available_at date = next "
        "trade_cal trading day after trade_date"
    ),
    POLICY_FACTOR_NAME: (
        "mean(sentiment) over npr/cctv_news fields grouped by (factor_date, "
        "MARKET); npr available_at = exact pubtime, cctv_news available_at = "
        "next trade_cal trading day after the broadcast date"
    ),
}


def _corpus_code_artifact_source(
    *, factor_name: str, manifest: dict[str, Any], values_sha256: str
) -> str:
    """Deterministic provenance code bound to factor_candidates.code_sha256.

    The transformation mirrors the series builders above so the registered
    values can be rebuilt from the persisted fields index plus trade_cal.
    """

    source = manifest["source"]
    policy = manifest["availability_policy"][factor_name]
    datasets = source["source_datasets"]
    if factor_name == IRM_QA_FACTOR_NAME:
        instrument_block = 'group_keys = ["factor_date", "ts_code"]'
    else:
        instrument_block = (
            'dated["instrument"] = "MARKET"\n    group_keys = ["factor_date", "instrument"]'
        )
    return f'''"""Provenance code artifact for the externally produced {factor_name} factor.

Generated at factor-registration time by quant_platform.corpus_nlp. The
registered factor values derive from the corpus NLP fields index (LLM
sentiment); sentiment is read from the persisted fields.parquet, factor dates
are recomputed with the persisted trade_cal via
corpus_nlp.factor_date_for, and the series is normalized with the
factor_evaluator.normalize_series contract.

source dataset: {source["dataset"]}
source corpora: {datasets}
prompt_version: {source["prompt_version"]}
model: {source["model"]}
availability_policy: {policy}
values sha256: {values_sha256}
"""

from __future__ import annotations

from datetime import date
from collections.abc import Sequence

import pandas as pd

from quant_platform.corpus_nlp import factor_date_for
from quant_platform.factor_evaluator import normalize_series

FACTOR_NAME = {factor_name!r}
SOURCE_DATASETS = {datasets!r}


def compute_factor(fields: pd.DataFrame, open_days: Sequence[date]) -> pd.Series:
    """Rebuild the factor values from the corpus NLP fields index + trade_cal."""

    dated = fields[fields["source_dataset"].isin(SOURCE_DATASETS)].copy()
    dated["factor_date"] = dated["available_at"].map(
        lambda value: factor_date_for(pd.Timestamp(value).to_pydatetime(), open_days)
    )
    dated["sentiment"] = pd.to_numeric(dated["sentiment"], errors="coerce")
    {instrument_block}
    series = dated.groupby(group_keys, sort=True)["sentiment"].mean()
    series = series.rename(FACTOR_NAME)
    series.index = series.index.set_names(["datetime", "instrument"])
    return normalize_series(series, FACTOR_NAME)
'''


def _corpus_metadata(
    factor_name: str, manifest: dict[str, Any], values_sha256: str
) -> ExternalFactorMetadata:
    source = manifest["source"]
    policy = manifest["availability_policy"]
    return ExternalFactorMetadata(
        description=(
            f"{_FACTOR_DESCRIPTIONS[factor_name]} Availability: "
            f"{policy[factor_name]}. Externally produced by corpus_nlp "
            f"(prompt_version={source['prompt_version']}, model={source['model']}, "
            f"source corpora: {', '.join(source['source_datasets'])})."
        ),
        formulation=_FACTOR_FORMULATIONS[factor_name],
        variables={
            "availability_policy": policy,
            "source": source,
            "values_sha256": values_sha256,
            "manifest": None,  # filled by the caller with the manifest path
            "rows": manifest["rows"],
            "ingested_fields": ["available_at", "ingested_at"],
        },
        code_source=_corpus_code_artifact_source(
            factor_name=factor_name, manifest=manifest, values_sha256=values_sha256
        ),
        run_config={
            "prompt_version": source["prompt_version"],
            "model": source["model"],
            "availability_policy": policy,
        },
        rdagent_feedback=(
            "externally produced corpus NLP factor; manifest sha256 verified at registration"
        ),
    )


def default_factors_dir(data_root: Path) -> Path:
    """Return the directory where corpus NLP factor artifacts land."""

    return data_root / CORPUS_NLP_DIR / "factors"


def register_corpus_factor(
    store: ResearchStore,
    factors_dir: Path,
    *,
    factor_name: str,
    actor: str = IMPORT_ACTOR,
) -> dict[str, Any]:
    """Verify and register one corpus NLP factor artifact; idempotent.

    Uses the generic external-factor channel
    (``announcement_factor_registry.register_external_factor``): manifest
    sha256 fail-closed verification, research-run lineage, idempotency key
    (name, values_sha256).
    """

    if factor_name not in CORPUS_FACTOR_NAMES:
        raise ValueError(
            f"unknown corpus factor {factor_name!r}; expected one of {list(CORPUS_FACTOR_NAMES)}"
        )

    def build_metadata(manifest: dict[str, Any], values_sha256: str) -> ExternalFactorMetadata:
        metadata = _corpus_metadata(factor_name, manifest, values_sha256)
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
        required_source_keys=("prompt_version", "model"),
    )
