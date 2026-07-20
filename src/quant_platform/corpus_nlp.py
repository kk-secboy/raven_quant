"""Corpus NLP processing layer: Tushare text corpus -> LLM extraction -> PIT signal fields.

Reads the already-downloaded Tushare text datasets — ``major_news`` long-form
market-level news and the ``irm_qa_sh``/``irm_qa_sz`` exchange interaction Q&A —
from the units/snapshots dual layout, normalizes every row into a unified text
item (source_dataset / item_id / ts_code / title / content / pub_time), and calls
an OpenAI-compatible chat endpoint for structured sentiment/topic extraction.
Outputs land under ``data/corpus_nlp/`` following the ``announcement_nlp`` layout:

- ``units/fields_<timestamp>_<uuid>.parquet`` — immutable per-run field units
- ``fields.parquet`` — derived structured-fields index with row-level
  ``available_at``/``ingested_at`` PIT timestamps
- ``state.parquet`` — processing ledger keyed by content sha256 +
  prompt_version + model for idempotent reruns and audit
- ``factors/<name>.parquet`` + ``factors/<name>.json`` — factor-values
  artifacts with sha256 manifests: ``news_sentiment_daily`` (market level,
  pseudo-instrument ``MARKET``) and ``irm_qa_sentiment_daily`` (per ts_code)

PIT semantics (design draft 3.3):

- ``major_news`` carries an exact ``pub_time``; ``available_at`` = pub_time.
- ``irm_qa_*`` carries only a trade date; ``available_at`` conservatively moves
  to the next trading day from the persisted trade_cal (never weekday guesses).
- Daily factor datetime: ``available_at`` before 15:00 on a trading day maps to
  that day, otherwise to the next trade_cal trading day.
- ``ingested_at`` is the processing moment of each row.

This module lives in quant_platform (not quant_data) for the same reason as
``announcement_nlp``: it consumes the platform LLM secret store and the platform
factor-series contract.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from bisect import bisect_left
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd

from quant_data.cninfo_announcements import (
    _parquet_files,
    _read_parquet_union,
    load_trade_calendar_open_days,
    next_trading_day,
)
from quant_data.rate_limit import GlobalRateGate

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

PROMPT_VERSION = "corpus-nlp.v1"

TOPICS = ("macro", "policy", "industry", "company", "market", "other")

DATASET_MAJOR_NEWS = "major_news"
DATASET_IRM_QA_SH = "irm_qa_sh"
DATASET_IRM_QA_SZ = "irm_qa_sz"
IRM_QA_DATASETS = (DATASET_IRM_QA_SH, DATASET_IRM_QA_SZ)
SUPPORTED_CORPUS_DATASETS = (DATASET_MAJOR_NEWS, *IRM_QA_DATASETS)

CORPUS_NLP_DIR = "corpus_nlp"
# Pseudo-instrument code for market-level series that carry no ts_code; it can
# never collide with a real Tushare code (those look like 000001.SZ).
MARKET_INSTRUMENT = "MARKET"
TRADING_DAY_CUTOFF = time(15, 0)

NEWS_FACTOR_NAME = "news_sentiment_daily"
IRM_QA_FACTOR_NAME = "irm_qa_sentiment_daily"
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
    frame = _read_parquet_union(
        paths, "SELECT * FROM read_parquet(?, union_by_name=true) LIMIT 0"
    )
    return set(frame.columns)


def _major_news_items(paths: list[str]) -> list[CorpusItem]:
    columns = _available_columns(paths)
    missing = sorted({"title", "content", "pub_time"} - columns)
    if missing:
        raise RuntimeError(f"major_news parquet misses required columns: {missing}")
    frame = _read_parquet_union(
        paths,
        """
        SELECT
            CAST(title AS VARCHAR) AS title,
            CAST(content AS VARCHAR) AS content,
            CAST(pub_time AS VARCHAR) AS pub_time
        FROM read_parquet(?, union_by_name=true)
        """,
    )
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


def _irm_qa_items(paths: list[str], dataset: str) -> list[CorpusItem]:
    columns = _available_columns(paths)
    question_col = next((name for name in ("q", "question") if name in columns), None)
    answer_col = next((name for name in ("a", "answer") if name in columns), None)
    missing = sorted({"trade_date", "ts_code"} - columns)
    if question_col is None:
        missing.append("q|question")
    if missing:
        raise RuntimeError(f"{dataset} parquet misses required columns: {missing}")
    select_parts = [
        "CAST(ts_code AS VARCHAR) AS ts_code",
        "coalesce(try_cast(trade_date AS DATE), "
        "try_strptime(CAST(trade_date AS VARCHAR), '%Y%m%d')::DATE) AS trade_date",
        f'CAST("{question_col}" AS VARCHAR) AS question',
    ]
    if answer_col is not None:
        # Tolerant: the answer column is optional; questions alone still carry signal.
        select_parts.append(f'CAST("{answer_col}" AS VARCHAR) AS answer')
    frame = _read_parquet_union(
        paths,
        f"SELECT {', '.join(select_parts)} FROM read_parquet(?, union_by_name=true)",
    )
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


def load_corpus_items(
    data_root: Path,
    *,
    datasets: Iterable[str] | None = None,
    ts_codes: set[str] | None = None,
    start: date | None = None,
    end: date | None = None,
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
    for dataset in ordered:
        rows = (
            _major_news_items(paths_by_dataset[dataset])
            if dataset == DATASET_MAJOR_NEWS
            else _irm_qa_items(paths_by_dataset[dataset], dataset)
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


def available_at_for(item: CorpusItem, open_days: Sequence[date]) -> datetime:
    """Field-level visibility timestamp (design draft 3.3).

    major_news has an exact publication moment; date-only irm_qa sources
    conservatively become visible on the next trade_cal trading day.
    """

    if item.source_dataset == DATASET_MAJOR_NEWS:
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


def _write_factor_artifact(
    series: pd.Series,
    factors_dir: Path,
    *,
    name: str,
    source_datasets: tuple[str, ...],
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
        },
        "generated_at": now.isoformat(),
    }
    if name == NEWS_FACTOR_NAME:
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
    fields_path: Path | None
    state_path: Path | None
    unit_path: Path | None
    factors: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "succeeded",
            "planned": self.planned,
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
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
    now: Callable[[], datetime] | None = None,
    environ: Mapping[str, str] | None = None,
) -> CorpusNlpSummary:
    """Run LLM sentiment/topic structuring over the downloaded text corpus.

    Idempotent on the processing key content-sha256+prompt_version+model: rows
    already succeeded are skipped, failed rows are re-attempted. Every failure
    is recorded in the state ledger and never produces a signal row
    (fail closed).
    """

    clock = now or (lambda: datetime.now(UTC))
    items = load_corpus_items(
        data_root, datasets=datasets, ts_codes=ts_codes, start=start, end=end
    )
    if limit is not None and limit > 0:
        items = items[:limit]
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
    processed = skipped = failed = 0
    for item in items:
        process_key = f"{item.item_id}:{PROMPT_VERSION}:{model}"
        existing = state.get(process_key)
        if existing is not None and str(existing["status"]) == "succeeded":
            skipped += 1
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
        try:
            available_at = available_at_for(item, open_days)
        except LookupError as exc:
            failed += 1
            state_row.update(status="failed", stage="availability", error=str(exc)[:500])
            continue
        state_row["available_at"] = available_at
        try:
            messages = build_extraction_messages(item=item, text=item.content[:max_chars])
            result = parse_extraction_payload(chat_client.complete(messages, model=model))
        except LlmExtractionError as exc:
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

    fields = _fields_frame(list(fields_records.values()))
    # Factor dates are derived before any write: a calendar too short for the
    # visibility timestamps fails the run closed instead of persisting a
    # partial artifact.
    news_series = build_news_sentiment_series(fields, open_days)
    irm_qa_series = build_irm_qa_sentiment_series(fields, open_days)

    unit_path: Path | None = None
    if new_field_rows:
        unit_path = (
            units_dir / f"fields_{clock():%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}.parquet"
        )
        _write_parquet_atomic(_fields_frame(new_field_rows), unit_path)

    _write_parquet_atomic(fields, fields_path)
    _write_parquet_atomic(_state_frame(list(state.values())), state_path)

    news_artifact = _write_factor_artifact(
        news_series,
        factors_dir,
        name=NEWS_FACTOR_NAME,
        source_datasets=(DATASET_MAJOR_NEWS,),
        model=model,
        now=clock(),
    )
    irm_qa_artifact = _write_factor_artifact(
        irm_qa_series,
        factors_dir,
        name=IRM_QA_FACTOR_NAME,
        source_datasets=IRM_QA_DATASETS,
        model=model,
        now=clock(),
    )
    return CorpusNlpSummary(
        planned=len(items),
        processed=processed,
        skipped=skipped,
        failed=failed,
        fields_path=fields_path,
        state_path=state_path,
        unit_path=unit_path,
        factors={
            NEWS_FACTOR_NAME: news_artifact,
            IRM_QA_FACTOR_NAME: irm_qa_artifact,
        },
    )
