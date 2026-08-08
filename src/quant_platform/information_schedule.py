from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from quant_data.execution_contract import require_daily_qlib_contract

from .corpus_nlp import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_IRM_PER_INSTRUMENT_DAY,
    DEFAULT_MAJOR_NEWS_PER_DAY,
)
from .services import list_qlib_datasets

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
    "include_factor_evaluation",
    "factor_evaluation",
    "snapshot_name",
    "horizons",
    "benchmark_code",
}

INFORMATION_EVALUATION_KEYS = {"dataset", "periods", "universe", "benchmark"}
STRUCTURED_INFORMATION_SOURCES = {
    "report_rc",
    "major_news_mentions",
    "news_flash",
}
STRUCTURED_INFORMATION_STARTS = {
    "report_rc": date(2010, 1, 1),
    "major_news_mentions": date(2018, 11, 20),
    "news_flash": date(2018, 11, 20),
}
INFORMATION_FACTOR_REFRESH_KEYS = {"sources", "weekday", "factor_evaluation"}
RESEARCH_PERIOD_KEYS = (
    "train_start",
    "train_end",
    "valid_start",
    "valid_end",
    "test_start",
    "test_end",
)


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
    include_factor_evaluation = _boolean(payload, "include_factor_evaluation", False)
    if (include_corpus_nlp or include_event_labels or include_factor_evaluation) and not enable_nlp:
        raise ValueError("information_pipeline NLP consumers require enable_nlp=true")

    evaluation = _normalize_factor_evaluation(
        payload.get("factor_evaluation"), enabled=include_factor_evaluation
    )

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
        "include_factor_evaluation": include_factor_evaluation,
        "factor_evaluation": evaluation,
        "snapshot_name": snapshot_name,
        "horizons": horizons,
        "benchmark_code": benchmark_code,
    }


def normalize_information_factor_refresh_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate the weekly deterministic information-factor refresh policy."""

    if not isinstance(payload, dict):
        raise ValueError("information_factor_refresh payload must be an object")
    unknown_keys = sorted(set(payload) - INFORMATION_FACTOR_REFRESH_KEYS)
    if unknown_keys:
        raise ValueError(
            "information_factor_refresh contains unsupported payload keys: "
            f"{unknown_keys}"
        )
    sources = payload.get("sources", sorted(STRUCTURED_INFORMATION_SOURCES))
    if not isinstance(sources, list) or not sources or not all(
        isinstance(item, str) for item in sources
    ):
        raise ValueError("information_factor_refresh sources must be a non-empty string list")
    normalized_sources = sorted(set(sources))
    unknown_sources = sorted(set(normalized_sources) - STRUCTURED_INFORMATION_SOURCES)
    if unknown_sources:
        raise ValueError(
            "information_factor_refresh sources contain unsupported values: "
            f"{unknown_sources}"
        )
    weekday = payload.get("weekday", 4)
    if isinstance(weekday, bool):
        raise ValueError("information_factor_refresh weekday must be an integer")
    try:
        weekday = int(weekday)
    except (TypeError, ValueError) as exc:
        raise ValueError("information_factor_refresh weekday must be an integer") from exc
    if not 0 <= weekday <= 4:
        raise ValueError("information_factor_refresh weekday must be between 0 and 4")
    evaluation = _normalize_factor_evaluation(
        payload.get("factor_evaluation"), enabled=True
    )
    return {
        "sources": normalized_sources,
        "weekday": weekday,
        "factor_evaluation": evaluation,
    }


def _normalize_factor_evaluation(value: Any, *, enabled: bool) -> dict[str, Any] | None:
    if not enabled:
        if value not in (None, {}):
            raise ValueError(
                "information_pipeline factor_evaluation requires "
                "include_factor_evaluation=true"
            )
        return None
    if not isinstance(value, dict):
        raise ValueError("information_pipeline factor_evaluation must be an object")
    unknown = sorted(set(value) - INFORMATION_EVALUATION_KEYS)
    if unknown:
        raise ValueError(
            f"information_pipeline factor_evaluation contains unsupported keys: {unknown}"
        )
    dataset = str(value.get("dataset") or "").strip()
    if len(dataset) < 3 or len(dataset) > 120:
        raise ValueError("information_pipeline factor_evaluation dataset is required")
    periods = value.get("periods")
    if not isinstance(periods, dict) or set(periods) != set(RESEARCH_PERIOD_KEYS):
        raise ValueError(
            "information_pipeline factor_evaluation periods must contain exactly "
            + ", ".join(RESEARCH_PERIOD_KEYS)
        )
    try:
        parsed = {key: date.fromisoformat(str(periods[key])) for key in RESEARCH_PERIOD_KEYS}
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "information_pipeline factor_evaluation periods must be ISO dates"
        ) from exc
    if not (
        parsed["train_start"]
        <= parsed["train_end"]
        < parsed["valid_start"]
        <= parsed["valid_end"]
        < parsed["test_start"]
        <= parsed["test_end"]
    ):
        raise ValueError(
            "information_pipeline factor_evaluation train, validation, and test "
            "periods must be ordered and non-overlapping"
        )
    if (parsed["test_start"] - parsed["valid_end"]).days <= 5:
        raise ValueError(
            "information_pipeline factor_evaluation requires a purge/embargo gap "
            "greater than 5 days"
        )
    universe = str(value.get("universe") or "cn_all").strip()
    benchmark = str(value.get("benchmark") or "SH000300").strip().upper()
    if len(universe) < 2 or len(universe) > 100:
        raise ValueError("information_pipeline factor_evaluation universe is invalid")
    if len(benchmark) < 4 or len(benchmark) > 32:
        raise ValueError("information_pipeline factor_evaluation benchmark is invalid")
    return {
        "dataset": dataset,
        "periods": {key: parsed[key].isoformat() for key in RESEARCH_PERIOD_KEYS},
        "universe": universe,
        "benchmark": benchmark,
    }


def resolve_information_evaluation_dataset(
    data_root: Path, evaluation: dict[str, Any]
) -> dict[str, Any]:
    """Resolve one pinned, reproducible daily Qlib dataset for scheduled evaluation."""

    available = {item["name"]: item for item in list_qlib_datasets(data_root)}
    dataset = available.get(str(evaluation["dataset"]))
    if not dataset or not dataset.get("ready"):
        raise ValueError("information factor evaluation Qlib dataset is not ready")
    if not dataset.get("reproducible"):
        raise ValueError(
            "information factor evaluation requires immutable Qlib provenance"
        )
    if dataset.get("frequency") != "day":
        raise ValueError("information factor evaluation requires a daily Qlib dataset")
    require_daily_qlib_contract(dataset.get("provenance") or {})
    periods = evaluation["periods"]
    if dataset.get("start_date") and periods["train_start"] < dataset["start_date"]:
        raise ValueError("information factor evaluation starts before the Qlib dataset")
    if dataset.get("end_date") and periods["test_end"] > dataset["end_date"]:
        raise ValueError("information factor evaluation ends after the Qlib dataset")
    calendar_path = Path(str(dataset["path"])) / "calendars" / "day.txt"
    try:
        calendar = [
            date.fromisoformat(line.strip())
            for line in calendar_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError) as exc:
        raise ValueError("information factor evaluation Qlib calendar is invalid") from exc
    valid_start = date.fromisoformat(periods["valid_start"])
    valid_end = date.fromisoformat(periods["valid_end"])
    test_start = date.fromisoformat(periods["test_start"])
    test_end = date.fromisoformat(periods["test_end"])
    valid_days = sum(valid_start <= day <= valid_end for day in calendar)
    test_days = sum(test_start <= day <= test_end for day in calendar)
    if valid_days < 126 or test_days < 252:
        raise ValueError(
            "information factor evaluation requires at least 126 validation and "
            f"252 final-test trading days; got {valid_days} and {test_days}"
        )
    return dataset


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
