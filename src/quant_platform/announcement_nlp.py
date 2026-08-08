"""Announcement NLP processing layer: PDF text -> LLM extraction -> PIT signal fields.

Reads the cninfo announcement index (``data/announcements/index.parquet``) produced by
``quant_data.cninfo_announcements``, extracts the PDF text layer, and calls an
OpenAI-compatible chat endpoint for structured extraction (event type, tone score,
key numbers, causal impact channels, horizon, direction, and confidence). Outputs
land under ``data/announcements/nlp/``:

- ``units/fields_<timestamp>_<uuid>.parquet`` — immutable per-run field units
- ``fields.parquet`` — derived structured-fields index (PIT: available_at is the
  field visibility timestamp inherited from the announcement index)
- ``state.parquet`` — processing ledger keyed by sha256+prompt_version+model for
  idempotent reruns and audit
- ``factors/<name>.parquet`` + ``factors/<name>.json`` — factor-values artifact
  following the ``factor_evaluator.normalize_series`` contract, with sha256

This module lives in quant_platform (not quant_data) because it consumes the
platform LLM secret store and the platform factor-series contract; quant_data
already depends on quant_platform (qlib_builder -> eligibility), never the
reverse, so the dependency direction stays one-way.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import requests
from pypdf import PdfReader

from quant_data.rate_limit import GlobalRateGate

from .factor_evaluator import normalize_series

PROMPT_VERSION = "announcement-nlp.v2"

EVENT_TYPES = (
    "earnings_forecast",
    "periodic_report",
    "regulatory_letter",
    "repurchase",
    "dividend",
    "major_contract",
    "litigation",
    "equity_change",
    "other",
)

IMPACT_DIRECTIONS = ("positive", "negative", "neutral", "mixed", "uncertain")
IMPACT_HORIZONS = (
    "immediate",
    "short_term",
    "medium_term",
    "long_term",
    "uncertain",
)
IMPACT_CHANNELS = (
    "earnings",
    "cash_flow",
    "balance_sheet",
    "governance",
    "regulatory",
    "industry_demand",
    "capacity",
    "pricing",
    "cost",
    "capital_allocation",
    "share_supply",
    "litigation",
    "other",
)
MAX_LOGIC_SUMMARY_CHARS = 240

ANNOUNCEMENTS_DIR = "announcements"
NLP_SUBDIR = "nlp"
LLM_SECRET_NAME = "llm"
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
MAX_TEXT_CHARS = 12_000
USER_AGENT = "quantlab-announcement-nlp/1.0"

FIELDS_COLUMNS = (
    "process_key",
    "ts_code",
    "ann_date",
    "available_at",
    "ingested_at",
    "event_type",
    "tone_score",
    "key_numbers",
    "impact_direction",
    "impact_horizon",
    "impact_channels",
    "logic_summary",
    "confidence",
    "source_sha256",
    "model",
    "prompt_version",
    "processed_at",
)
STATE_COLUMNS = (
    "process_key",
    "source_sha256",
    "prompt_version",
    "model",
    "status",
    "stage",
    "error",
    "processed_at",
    "ts_code",
    "available_at",
)

FACTOR_NAME = "announcement_tone"
LOGIC_FACTOR_NAME = "announcement_logic_score"
# Same availability_policy style as qlib_builder._research_feature_contract:
# the field becomes visible at available_at, the first trading day strictly
# after the announcement date (derived by the cninfo downloader).
AVAILABILITY_POLICY = {
    FACTOR_NAME: "available_at_first_trading_day_after_announcement",
    LOGIC_FACTOR_NAME: "available_at_first_trading_day_after_announcement",
}


class PdfTextExtractionError(RuntimeError):
    """The PDF body cannot be parsed or carries no usable text layer."""


class LlmCredentialsError(RuntimeError):
    """LLM credentials are not configured; never fall back to fake keys."""


class LlmExtractionError(RuntimeError):
    """LLM call or response validation failed; fail closed, no signal emitted."""

    def __init__(self, message: str, *, stage: str) -> None:
        super().__init__(message)
        self.stage = stage


def extract_pdf_text(path: Path, *, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Return the text layer of a PDF; raise PdfTextExtractionError when unusable.

    Scanned documents without a text layer yield an empty extraction and are
    treated as failures: no signal is produced for them.
    """

    try:
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as exc:  # pypdf raises many parser-specific errors; fail closed
        raise PdfTextExtractionError(f"cannot extract text from {path.name}: {exc}") from exc
    if not text:
        raise PdfTextExtractionError(
            f"{path.name} has no text layer (scanned document?); no signal can be produced"
        )
    return text[:max_chars]


@dataclass(frozen=True, slots=True)
class LlmCredentials:
    api_key: str
    api_base: str
    chat_model: str
    source: str  # "runtime_secret_store" | "environment"


class SecretStoreLike(Protocol):
    def get(self, name: str) -> dict[str, str] | None: ...


def load_llm_credentials(
    secret_store: SecretStoreLike | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> LlmCredentials:
    """Resolve the LLM credential triple: runtime secret store first, env fallback.

    Raises LlmCredentialsError listing the configuration paths when nothing
    usable is configured; a missing key is never silently replaced.
    """

    env = os.environ if environ is None else environ
    record = secret_store.get(LLM_SECRET_NAME) if secret_store is not None else None
    if record and record.get("api_key", "").strip():
        return LlmCredentials(
            api_key=record["api_key"].strip(),
            api_base=(record.get("api_base") or "").strip().rstrip("/") or DEFAULT_API_BASE,
            chat_model=(record.get("chat_model") or "").strip() or DEFAULT_CHAT_MODEL,
            source="runtime_secret_store",
        )
    env_key = env.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return LlmCredentials(
            api_key=env_key,
            api_base=env.get("OPENAI_API_BASE", "").strip().rstrip("/") or DEFAULT_API_BASE,
            chat_model=env.get("CHAT_MODEL", "").strip() or DEFAULT_CHAT_MODEL,
            source="environment",
        )
    raise LlmCredentialsError(
        "LLM credentials are not configured; set them via the control plane "
        "(POST /api/settings/llm with api_key/api_base/chat_model, stored encrypted in the "
        "runtime_secrets table) or export OPENAI_API_KEY/OPENAI_API_BASE/CHAT_MODEL"
    )


class ChatCompleter(Protocol):
    """Injectable chat client; tests substitute a scripted fake, never a real LLM."""

    def complete(self, messages: list[dict[str, str]], *, model: str) -> str: ...


class OpenAIChatClient:
    """Bounded-retry client for OpenAI-compatible chat completions.

    The API key is only used for the Authorization header and never appears in
    errors or logs.
    """

    def __init__(
        self,
        credentials: LlmCredentials,
        *,
        rate_gate: GlobalRateGate,
        timeout_seconds: float = 120.0,
        max_attempts: int = 3,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.credentials = credentials
        self.rate_gate = rate_gate
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self.session = session or requests.Session()
        self.sleeper = sleeper

    def complete(self, messages: list[dict[str, str]], *, model: str) -> str:
        url = f"{self.credentials.api_base}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.credentials.api_key}",
            "User-Agent": USER_AGENT,
        }
        last_error: LlmExtractionError | None = None
        for attempt in range(1, self.max_attempts + 1):
            self.rate_gate.wait()
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=(10.0, self.timeout_seconds),
                )
            except requests.RequestException as exc:
                last_error = LlmExtractionError(f"LLM request failed: {exc}", stage="llm_call")
            else:
                status = int(response.status_code)
                if 200 <= status < 300:
                    return self._response_content(response)
                retryable = status == 429 or status >= 500
                last_error = LlmExtractionError(
                    f"LLM endpoint returned HTTP {status}", stage="llm_call"
                )
                if status == 429:
                    # Shared cooldown, same discipline as the download providers.
                    self.rate_gate.cooldown(60.0)
                    break
                if not retryable or attempt >= self.max_attempts:
                    break
            if attempt >= self.max_attempts:
                break
            self.sleeper(min(20.0, 2.0 ** (attempt - 1)) + random.uniform(0.0, 1.0))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _response_content(response: requests.Response) -> str:
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LlmExtractionError(
                "malformed chat completion response body", stage="llm_call"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise LlmExtractionError("empty chat completion content", stage="llm_call")
        return content


def build_extraction_messages(
    *,
    ts_code: str,
    ann_date: date,
    title: str,
    category: str,
    text: str,
) -> list[dict[str, str]]:
    """Versioned prompt for the structured announcement extraction."""

    system = (
        f"You are an A-share announcement analysis engine (prompt_version={PROMPT_VERSION}). "
        "Respond with exactly one JSON object and nothing else, with keys "
        '"event_type", "tone_score", "key_numbers", "impact_direction", '
        '"impact_horizon", "impact_channels", "logic_summary", "confidence". '
        f'"event_type" must be one of: {", ".join(EVENT_TYPES)}. '
        '"tone_score" is a float in [-1.0, 1.0]: for periodic reports it rates the '
        "management tone (negative = pessimistic, positive = optimistic); for regulatory "
        "letters it rates issue severity (negative = severe). "
        '"key_numbers" is a JSON object with the key figures mentioned (e.g. forecast net '
        'profit bounds, year-over-year change ranges); use {} when none can be extracted. '
        f'"impact_direction" must be one of: {", ".join(IMPACT_DIRECTIONS)}. '
        f'"impact_horizon" must be one of: {", ".join(IMPACT_HORIZONS)}. '
        f'"impact_channels" is a JSON array containing only: {", ".join(IMPACT_CHANNELS)}; '
        'use [] when the document does not support a causal channel. '
        f'"logic_summary" is an evidence-grounded causal summary of at most '
        f'{MAX_LOGIC_SUMMARY_CHARS} characters; use an empty string when unsupported. '
        'Do not infer facts not present in the announcement. '
        '"confidence" is a float in [0.0, 1.0].'
    )
    user = (
        f"ts_code: {ts_code}\n"
        f"ann_date: {ann_date.isoformat()}\n"
        f"category: {category}\n"
        f"title: {title}\n\n"
        f"announcement text:\n{text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    event_type: str
    tone_score: float
    key_numbers: dict[str, Any]
    impact_direction: str
    impact_horizon: str
    impact_channels: tuple[str, ...]
    logic_summary: str
    confidence: float


def _bounded_float(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LlmExtractionError(
            f"{name} must be a number in [{low}, {high}]", stage="llm_parse"
        )
    result = float(value)
    if not low <= result <= high:
        raise LlmExtractionError(f"{name}={result} is outside [{low}, {high}]", stage="llm_parse")
    return result


def parse_extraction_payload(raw: str) -> ExtractionResult:
    """Validate the LLM JSON payload against the strict schema; fail closed.

    Required keys: event_type (fixed enum), tone_score ([-1, 1]), key_numbers
    (JSON object), constrained impact direction/horizon/channels, a bounded
    evidence-grounded summary, and confidence ([0, 1]). Unknown extra keys are
    tolerated but ignored; any missing key or type/range violation is a failure.
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
    required = {
        "event_type",
        "tone_score",
        "key_numbers",
        "impact_direction",
        "impact_horizon",
        "impact_channels",
        "logic_summary",
        "confidence",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise LlmExtractionError(f"LLM response misses keys: {missing}", stage="llm_parse")
    event_type = payload.get("event_type")
    if event_type not in EVENT_TYPES:
        raise LlmExtractionError(
            f"event_type must be one of {list(EVENT_TYPES)}; got {event_type!r}",
            stage="llm_parse",
        )
    tone_score = _bounded_float(payload.get("tone_score"), "tone_score", -1.0, 1.0)
    key_numbers = payload.get("key_numbers")
    if not isinstance(key_numbers, dict):
        raise LlmExtractionError("key_numbers must be a JSON object", stage="llm_parse")
    impact_direction = payload.get("impact_direction")
    if impact_direction not in IMPACT_DIRECTIONS:
        raise LlmExtractionError(
            f"impact_direction must be one of {list(IMPACT_DIRECTIONS)}; "
            f"got {impact_direction!r}",
            stage="llm_parse",
        )
    impact_horizon = payload.get("impact_horizon")
    if impact_horizon not in IMPACT_HORIZONS:
        raise LlmExtractionError(
            f"impact_horizon must be one of {list(IMPACT_HORIZONS)}; got {impact_horizon!r}",
            stage="llm_parse",
        )
    impact_channels = payload.get("impact_channels")
    if (
        not isinstance(impact_channels, list)
        or any(not isinstance(channel, str) for channel in impact_channels)
        or any(channel not in IMPACT_CHANNELS for channel in impact_channels)
        or len(set(impact_channels)) != len(impact_channels)
    ):
        raise LlmExtractionError(
            "impact_channels must be a duplicate-free JSON array of governed channels",
            stage="llm_parse",
        )
    logic_summary = payload.get("logic_summary")
    if not isinstance(logic_summary, str) or len(logic_summary) > MAX_LOGIC_SUMMARY_CHARS:
        raise LlmExtractionError(
            f"logic_summary must be a string of at most {MAX_LOGIC_SUMMARY_CHARS} characters",
            stage="llm_parse",
        )
    confidence = _bounded_float(payload.get("confidence"), "confidence", 0.0, 1.0)
    return ExtractionResult(
        event_type=str(event_type),
        tone_score=tone_score,
        key_numbers=key_numbers,
        impact_direction=str(impact_direction),
        impact_horizon=str(impact_horizon),
        impact_channels=tuple(impact_channels),
        logic_summary=logic_summary.strip(),
        confidence=confidence,
    )


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd", engine="pyarrow")
    os.replace(temporary, path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _empty_fields_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "process_key": pd.Series(dtype="string"),
            "ts_code": pd.Series(dtype="string"),
            "ann_date": pd.Series(dtype="datetime64[ns]"),
            "available_at": pd.Series(dtype="datetime64[ns]"),
            "ingested_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "event_type": pd.Series(dtype="string"),
            "tone_score": pd.Series(dtype="float64"),
            "key_numbers": pd.Series(dtype="string"),
            "impact_direction": pd.Series(dtype="string"),
            "impact_horizon": pd.Series(dtype="string"),
            "impact_channels": pd.Series(dtype="string"),
            "logic_summary": pd.Series(dtype="string"),
            "confidence": pd.Series(dtype="float64"),
            "source_sha256": pd.Series(dtype="string"),
            "model": pd.Series(dtype="string"),
            "prompt_version": pd.Series(dtype="string"),
            "processed_at": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )


def _fields_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return _empty_fields_frame()
    frame = pd.DataFrame(list(rows), columns=list(FIELDS_COLUMNS))
    frame["ann_date"] = pd.to_datetime(frame["ann_date"])
    frame["available_at"] = pd.to_datetime(frame["available_at"])
    frame["ingested_at"] = pd.to_datetime(frame["ingested_at"], utc=True)
    frame["processed_at"] = pd.to_datetime(frame["processed_at"], utc=True)
    return frame.sort_values(
        ["available_at", "ts_code", "process_key"], kind="stable"
    ).reset_index(drop=True)


def _empty_state_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "process_key": pd.Series(dtype="string"),
            "source_sha256": pd.Series(dtype="string"),
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


def _load_records(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    records: dict[str, dict] = {}
    for row in frame.itertuples(index=False):
        record = {column: getattr(row, column) for column in frame.columns}
        records[str(record["process_key"])] = record
    return records


def load_announcement_index(
    data_root: Path,
    *,
    ts_codes: set[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    categories: set[str] | None = None,
) -> pd.DataFrame:
    """Load the cninfo announcement metadata index with optional filters.

    Fail-closed: raises when the index parquet has not been produced yet.
    """

    index_path = data_root / ANNOUNCEMENTS_DIR / "index.parquet"
    if not index_path.is_file():
        raise RuntimeError(
            f"announcement index is unavailable at {index_path}; "
            "run `quant-data cninfo-announcements` first"
        )
    frame = pd.read_parquet(index_path)
    if frame.empty:
        return frame
    if ts_codes:
        wanted = {code.strip().upper() for code in ts_codes if code.strip()}
        frame = frame[frame["ts_code"].astype(str).str.upper().isin(sorted(wanted))]
    if start is not None:
        frame = frame[frame["ann_date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["ann_date"] <= pd.Timestamp(end)]
    if categories:
        frame = frame[frame["category"].astype(str).isin(sorted(categories))]
    return frame.sort_values(["ann_date", "ts_code", "url"], kind="stable").reset_index(
        drop=True
    )


def build_tone_factor_series(fields: pd.DataFrame, name: str = FACTOR_NAME) -> pd.Series:
    """Aggregate structured fields into the datetime/instrument tone-score series.

    Multiple announcements for the same instrument on the same availability
    date are averaged; the result follows the normalize_series contract.
    """

    frame = fields[["available_at", "ts_code", "tone_score"]].copy()
    frame["tone_score"] = pd.to_numeric(frame["tone_score"], errors="coerce")
    grouped = frame.groupby(["available_at", "ts_code"], sort=True)["tone_score"].mean()
    series = grouped.rename(name)
    series.index = series.index.set_names(["datetime", "instrument"])
    return normalize_series(series, name)


_DIRECTION_SCORE = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
    "mixed": 0.0,
    "uncertain": 0.0,
}
_HORIZON_WEIGHT = {
    "immediate": 1.0,
    "short_term": 0.85,
    "medium_term": 0.65,
    "long_term": 0.45,
    "uncertain": 0.0,
}


def build_logic_factor_series(
    fields: pd.DataFrame, name: str = LOGIC_FACTOR_NAME
) -> pd.Series:
    """Build an explainable directional logic signal from governed NLP enums.

    This is deliberately not a free-form LLM score.  Direction and horizon are
    mapped through frozen tables and multiplied by the extraction confidence;
    mixed/neutral/uncertain cases emit zero.  Post-event price response is never
    consumed here and is produced separately as a training label.
    """

    frame = fields[
        ["available_at", "ts_code", "impact_direction", "impact_horizon", "confidence"]
    ].copy()
    frame["direction_score"] = frame["impact_direction"].map(_DIRECTION_SCORE)
    frame["horizon_weight"] = frame["impact_horizon"].map(_HORIZON_WEIGHT)
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    frame[name] = frame["direction_score"] * frame["horizon_weight"] * frame["confidence"]
    grouped = frame.groupby(["available_at", "ts_code"], sort=True)[name].mean()
    series = grouped.rename(name)
    series.index = series.index.set_names(["datetime", "instrument"])
    return normalize_series(series, name)


def write_factor_artifact(
    fields: pd.DataFrame,
    factors_dir: Path,
    *,
    name: str = FACTOR_NAME,
    model: str,
    now: datetime,
) -> dict[str, Any]:
    """Write the normalized factor-values parquet plus its sha256 manifest.

    The artifact is shaped for a later ResearchStore.add_candidate import
    (values_path + sha256); database registration is intentionally left to a
    research-run context instead of being faked here.
    """

    current = fields[
        (fields["prompt_version"].astype(str) == PROMPT_VERSION)
        & (fields["model"].astype(str) == model)
    ]
    if name == FACTOR_NAME:
        series = build_tone_factor_series(current, name)
    elif name == LOGIC_FACTOR_NAME:
        series = build_logic_factor_series(current, name)
    else:
        raise ValueError(f"unsupported announcement factor: {name}")
    artifact_path = factors_dir / f"{name}.parquet"
    _write_parquet_atomic(series.rename(name).reset_index(), artifact_path)
    manifest = {
        "factor": name,
        "artifact": artifact_path.name,
        "sha256": _sha256_file(artifact_path),
        "rows": int(len(series)),
        "availability_policy": {name: AVAILABILITY_POLICY[name]},
        "source": {
            "dataset": "announcement_nlp_fields",
            "prompt_version": PROMPT_VERSION,
            "model": model,
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
class NlpSummary:
    planned: int
    processed: int
    skipped: int
    unavailable: int
    failed: int
    fields_path: Path | None
    state_path: Path | None
    unit_path: Path | None
    factor_manifest_path: Path | None
    factor_sha256: str | None
    factor_rows: int
    logic_factor_manifest_path: Path | None
    logic_factor_sha256: str | None
    logic_factor_rows: int

    def as_dict(self) -> dict[str, Any]:
        status = (
            "failed"
            if self.failed
            else "succeeded_with_source_gaps"
            if self.unavailable
            else "succeeded"
        )
        return {
            "status": status,
            "planned": self.planned,
            "processed": self.processed,
            "skipped": self.skipped,
            "unavailable": self.unavailable,
            "failed": self.failed,
            "fields_path": str(self.fields_path) if self.fields_path else None,
            "state_path": str(self.state_path) if self.state_path else None,
            "unit_path": str(self.unit_path) if self.unit_path else None,
            "factor_manifest_path": (
                str(self.factor_manifest_path) if self.factor_manifest_path else None
            ),
            "factor_sha256": self.factor_sha256,
            "factor_rows": self.factor_rows,
            "logic_factor_manifest_path": (
                str(self.logic_factor_manifest_path)
                if self.logic_factor_manifest_path
                else None
            ),
            "logic_factor_sha256": self.logic_factor_sha256,
            "logic_factor_rows": self.logic_factor_rows,
        }


def process_announcements(
    data_root: Path,
    *,
    ts_codes: set[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    categories: set[str] | None = None,
    limit: int | None = None,
    secret_store: SecretStoreLike | None = None,
    credentials: LlmCredentials | None = None,
    chat_client: ChatCompleter | None = None,
    rate_gate: GlobalRateGate | None = None,
    requests_per_minute: float = 30.0,
    timeout_seconds: float = 120.0,
    max_attempts: int = 3,
    max_chars: int = MAX_TEXT_CHARS,
    factor_name: str = FACTOR_NAME,
    checkpoint_every: int = 100,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    now: Callable[[], datetime] | None = None,
    environ: Mapping[str, str] | None = None,
) -> NlpSummary:
    """Run PDF extraction + LLM structuring over the announcement index.

    Idempotent on the processing key sha256+prompt_version+model: rows already
    succeeded are skipped, transient/LLM failures are re-attempted, and PDFs
    without a usable text layer are retained as auditable source gaps until a
    prompt/extractor version changes. Checkpoints persist state and successful
    fields without publishing a factor, so an interrupted large run resumes
    without repeating already completed LLM calls.
    """

    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")

    clock = now or (lambda: datetime.now(UTC))
    frame = load_announcement_index(
        data_root, ts_codes=ts_codes, start=start, end=end, categories=categories
    )
    if limit is not None and limit > 0:
        frame = frame.head(limit)

    base = data_root / ANNOUNCEMENTS_DIR / NLP_SUBDIR
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
    processed = skipped = unavailable = failed = completed = 0
    dirty = False
    unit_path: Path | None = None

    def publish_progress() -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "planned": len(frame),
                "completed": completed,
                "processed": processed,
                "skipped": skipped,
                "unavailable": unavailable,
                "failed": failed,
            }
        )

    def persist_checkpoint(*, force: bool = False) -> None:
        nonlocal dirty, new_field_rows, unit_path
        if dirty or force:
            if new_field_rows:
                unit_path = (
                    units_dir
                    / f"fields_{clock():%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}.parquet"
                )
                _write_parquet_atomic(_fields_frame(new_field_rows), unit_path)
                new_field_rows = []
            _write_parquet_atomic(_fields_frame(list(fields_records.values())), fields_path)
            _write_parquet_atomic(_state_frame(list(state.values())), state_path)
            dirty = False
        publish_progress()

    for row in frame.itertuples():
        process_key = f"{row.sha256}:{PROMPT_VERSION}:{model}"
        existing = state.get(process_key)
        if existing is not None:
            existing_status = str(existing["status"])
            if existing_status == "succeeded":
                skipped += 1
                completed += 1
                if completed % checkpoint_every == 0:
                    persist_checkpoint()
                continue
            if existing_status == "source_unavailable":
                unavailable += 1
                completed += 1
                if completed % checkpoint_every == 0:
                    persist_checkpoint()
                continue
        processed_at = clock()
        state_row: dict[str, Any] = {
            "process_key": process_key,
            "source_sha256": str(row.sha256),
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "status": "",
            "stage": "",
            "error": None,
            "processed_at": processed_at,
            "ts_code": str(row.ts_code),
            "available_at": pd.Timestamp(row.available_at).date(),
        }
        state[process_key] = state_row
        dirty = True
        try:
            text = extract_pdf_text(data_root / str(row.file_path), max_chars=max_chars)
            messages = build_extraction_messages(
                ts_code=str(row.ts_code),
                ann_date=pd.Timestamp(row.ann_date).date(),
                title=str(row.title),
                category=str(row.category),
                text=text,
            )
            result = parse_extraction_payload(chat_client.complete(messages, model=model))
        except PdfTextExtractionError as exc:
            unavailable += 1
            state_row.update(
                status="source_unavailable", stage="pdf_extract", error=str(exc)[:500]
            )
        except LlmExtractionError as exc:
            failed += 1
            state_row.update(status="failed", stage=exc.stage, error=str(exc)[:500])
        else:
            field_row = {
                "process_key": process_key,
                "ts_code": str(row.ts_code),
                "ann_date": pd.Timestamp(row.ann_date).date(),
                "available_at": pd.Timestamp(row.available_at).date(),
                "ingested_at": processed_at,
                "event_type": result.event_type,
                "tone_score": result.tone_score,
                "key_numbers": json.dumps(
                    result.key_numbers,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "impact_direction": result.impact_direction,
                "impact_horizon": result.impact_horizon,
                "impact_channels": json.dumps(
                    result.impact_channels, ensure_ascii=False, separators=(",", ":")
                ),
                "logic_summary": result.logic_summary,
                "confidence": result.confidence,
                "source_sha256": str(row.sha256),
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "processed_at": processed_at,
            }
            fields_records[process_key] = field_row
            new_field_rows.append(field_row)
            state_row.update(status="succeeded", stage="completed")
            processed += 1
        completed += 1
        if completed % checkpoint_every == 0:
            persist_checkpoint()

    persist_checkpoint(force=True)
    fields = _fields_frame(list(fields_records.values()))

    artifact = write_factor_artifact(
        fields, factors_dir, name=factor_name, model=model, now=clock()
    )
    logic_artifact = write_factor_artifact(
        fields, factors_dir, name=LOGIC_FACTOR_NAME, model=model, now=clock()
    )
    return NlpSummary(
        planned=len(frame),
        processed=processed,
        skipped=skipped,
        unavailable=unavailable,
        failed=failed,
        fields_path=fields_path,
        state_path=state_path,
        unit_path=unit_path,
        factor_manifest_path=artifact["manifest_path"],
        factor_sha256=artifact["manifest"]["sha256"],
        factor_rows=artifact["manifest"]["rows"],
        logic_factor_manifest_path=logic_artifact["manifest_path"],
        logic_factor_sha256=logic_artifact["manifest"]["sha256"],
        logic_factor_rows=logic_artifact["manifest"]["rows"],
    )
