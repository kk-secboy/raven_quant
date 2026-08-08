from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .corpus_nlp import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_IRM_PER_INSTRUMENT_DAY,
    DEFAULT_MAJOR_NEWS_PER_DAY,
)

INFORMATION_CORPUS_DATASETS = {
    "major_news",
    "npr",
    "cctv_news",
    "irm_qa_sh",
    "irm_qa_sz",
}
INFORMATION_ANNOUNCEMENT_CATEGORIES = {"announcement", "regulatory_letter"}
INFORMATION_SCHEDULE_KEYS = {
    "lookback_days",
    "regulatory_only",
    "download_limit",
    "enable_nlp",
    "announcement_categories",
    "announcement_nlp_limit",
    "include_corpus_nlp",
    "corpus_datasets",
    "corpus_nlp_limit",
    "batch_size",
    "major_news_per_day",
    "irm_per_instrument_day",
    "include_event_labels",
    "snapshot_name",
    "horizons",
    "benchmark_code",
}


def _integer(
    payload: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = payload.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"information_pipeline {key} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"information_pipeline {key} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"information_pipeline {key} must be between {minimum} and {maximum}")
    return value


def _boolean(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"information_pipeline {key} must be a boolean")
    return value


def _string_list(
    payload: dict[str, Any],
    key: str,
    *,
    allowed: set[str],
    default: list[str],
) -> list[str]:
    value = payload.get(key, default)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"information_pipeline {key} must be a string list")
    normalized = sorted(set(value))
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise ValueError(f"information_pipeline {key} contains unsupported values: {unknown}")
    return normalized


def normalize_information_schedule_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and freeze the bounded daily information-pipeline policy.

    Raw regulatory downloads are safe by default. LLM work must be explicitly
    enabled and always has a finite per-run cap so a recurring schedule cannot
    create an unbounded bill after a long outage or backfill.
    """

    if not isinstance(payload, dict):
        raise ValueError("information_pipeline payload must be an object")
    unknown_keys = sorted(set(payload) - INFORMATION_SCHEDULE_KEYS)
    if unknown_keys:
        raise ValueError(f"information_pipeline contains unsupported payload keys: {unknown_keys}")

    enable_nlp = _boolean(payload, "enable_nlp", False)
    include_corpus_nlp = _boolean(payload, "include_corpus_nlp", enable_nlp)
    include_event_labels = _boolean(payload, "include_event_labels", enable_nlp)
    if (include_corpus_nlp or include_event_labels) and not enable_nlp:
        raise ValueError("information_pipeline NLP consumers require enable_nlp=true")

    categories = _string_list(
        payload,
        "announcement_categories",
        allowed=INFORMATION_ANNOUNCEMENT_CATEGORIES,
        default=["regulatory_letter"],
    )
    if enable_nlp and not categories:
        raise ValueError("information_pipeline announcement_categories must not be empty")
    corpus_datasets = _string_list(
        payload,
        "corpus_datasets",
        allowed=INFORMATION_CORPUS_DATASETS,
        default=[],
    )

    horizons_raw = payload.get("horizons", [1, 3, 5, 20])
    if not isinstance(horizons_raw, list) or not horizons_raw:
        raise ValueError("information_pipeline horizons must be a non-empty list")
    if any(isinstance(value, bool) for value in horizons_raw):
        raise ValueError("information_pipeline horizons must contain integers")
    try:
        horizons = sorted({int(value) for value in horizons_raw})
    except (TypeError, ValueError) as exc:
        raise ValueError("information_pipeline horizons must contain integers") from exc
    if any(value < 1 or value > 252 for value in horizons):
        raise ValueError("information_pipeline horizons must be between 1 and 252")

    snapshot_name = str(payload.get("snapshot_name") or "").strip()
    if snapshot_name and (len(snapshot_name) < 3 or len(snapshot_name) > 120):
        raise ValueError("information_pipeline snapshot_name length is invalid")
    benchmark_code = str(payload.get("benchmark_code") or "000300.SH").strip().upper()
    if len(benchmark_code) < 9 or len(benchmark_code) > 12:
        raise ValueError("information_pipeline benchmark_code length is invalid")

    return {
        "lookback_days": _integer(payload, "lookback_days", 7, minimum=1, maximum=30),
        "regulatory_only": _boolean(payload, "regulatory_only", True),
        "download_limit": _integer(payload, "download_limit", 0, minimum=0, maximum=1_000_000),
        "enable_nlp": enable_nlp,
        "announcement_categories": categories,
        "announcement_nlp_limit": _integer(
            payload, "announcement_nlp_limit", 500, minimum=1, maximum=10_000
        ),
        "include_corpus_nlp": include_corpus_nlp,
        "corpus_datasets": corpus_datasets,
        "corpus_nlp_limit": _integer(payload, "corpus_nlp_limit", 500, minimum=1, maximum=10_000),
        "batch_size": _integer(
            payload,
            "batch_size",
            DEFAULT_BATCH_SIZE,
            minimum=1,
            maximum=100,
        ),
        "major_news_per_day": _integer(
            payload,
            "major_news_per_day",
            DEFAULT_MAJOR_NEWS_PER_DAY,
            minimum=0,
            maximum=10_000,
        ),
        "irm_per_instrument_day": _integer(
            payload,
            "irm_per_instrument_day",
            DEFAULT_IRM_PER_INSTRUMENT_DAY,
            minimum=0,
            maximum=100,
        ),
        "include_event_labels": include_event_labels,
        "snapshot_name": snapshot_name,
        "horizons": horizons,
        "benchmark_code": benchmark_code,
    }


def latest_verified_snapshot(data_root: Path, *, as_of: date) -> str:
    """Return the latest immutable snapshot that passed its blocking gate."""

    root = data_root / "snapshots"
    candidates: list[tuple[date, str]] = []
    if not root.is_dir():
        raise ValueError("no immutable snapshots are available")
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            verification = json.loads((path / "verification.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or not isinstance(verification, dict):
                continue
            end_date = date.fromisoformat(str(manifest["end_date"]))
        except (
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            continue
        if verification.get("ok") is not True or verification.get("errors"):
            continue
        if end_date <= as_of:
            candidates.append((end_date, path.name))
    if not candidates:
        raise ValueError("no verified immutable snapshot covers the information pipeline")
    return max(candidates)[1]
