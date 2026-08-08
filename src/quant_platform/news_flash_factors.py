"""Market-level news-flash intensity factor from the 9-source ``news`` dataset.

The ``news`` dataset (9 快讯 sources, exact ``datetime`` per row) is a
high-frequency market-level information stream. A per-row LLM sentiment would
duplicate the existing major_news market-level sentiment factor
(``news_sentiment_daily``) at a much higher cost — the two corpora report the
same market events, and major_news already carries the curated long-form
version. What the flash stream adds *on top* is its volume dynamics: abnormal
news-flow intensity is a classic attention/regime variable that major_news
(a thin, editorially filtered feed) cannot measure. This module therefore
deliberately produces only the deterministic aggregation feature, no second
sentiment score.

Factor (``news_flash_intensity_daily``, MARKET pseudo-instrument,
market_timeseries shape):

- ``available_at`` = the exact news ``datetime``; the daily factor date follows
  the corpus_nlp 15:00 cutoff rule against the persisted trade_cal (a flash at
  16:00 counts toward the next trading day — never weekday guesses).
- ``flash_count(D)`` = flashes mapped to factor date D.
- ``intensity(D) = flash_count(D) / mean(flash_count over the trailing 20
  trade_cal open days strictly before D)``; the denominator needs at least 5
  prior open days and uses strictly-prior days only (no look-ahead). Days with
  zero flashes emit no value (treated as a likely data gap, not a signal).

Outputs land under ``data/news_flash/`` following the report_rc layout:
``daily_counts.parquet`` (every open day in range, zero-filled — the exact
input of the provenance recompute) plus ``factors/<name>.parquet`` +
``factors/<name>.json`` with a sha256 manifest, registered into
``factor_candidates`` via the generic external-factor channel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
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
from .corpus_nlp import MARKET_INSTRUMENT, factor_date_for
from .factor_evaluator import normalize_series

if TYPE_CHECKING:
    from .research_store import ResearchStore

PRODUCER_VERSION = "news-flash-factors.v1"

DATASET = "news"
NEWS_FLASH_DIR = "news_flash"

INTENSITY_FACTOR_NAME = "news_flash_intensity_daily"
FACTOR_NAMES = (INTENSITY_FACTOR_NAME,)

# Trailing window (in trade_cal open days, strictly before the factor date)
# for the intensity denominator, and the minimum history required to emit a
# value at all.
TRAILING_WINDOW_DAYS = 20
MIN_HISTORY_DAYS = 5

IMPORT_RUN_KIND = "news_flash_factor_import"
IMPORT_ACTOR = "news-flash-registrar"

AVAILABILITY_POLICY = {
    INTENSITY_FACTOR_NAME: (
        "available_at = exact news datetime; factor date = publication date when "
        "available before 15:00 on a trade_cal trading day, otherwise the next "
        f"trade_cal trading day; intensity = flash_count / mean(flash_count over "
        f"the trailing {TRAILING_WINDOW_DAYS} open days strictly before the "
        f"factor date, min {MIN_HISTORY_DAYS} prior days — denominator never "
        "includes the factor date itself (no look-ahead)"
    ),
}

COUNTS_COLUMNS = ("factor_date", "flash_count")


def default_factors_dir(data_root: Path) -> Path:
    """Return the directory where news flash factor artifacts land."""

    return data_root / NEWS_FLASH_DIR / "factors"


def load_news_flash_datetimes(
    data_root: Path, *, start: date | None = None, end: date | None = None
) -> pd.Series:
    """Read the news flash publication datetimes; fail closed.

    Only the exact ``datetime`` column is needed for a volume factor; rows
    without a parseable timestamp are dropped with the same discipline as the
    corpus availability read guard.
    """

    paths = _parquet_files(data_root, DATASET)
    if not paths:
        raise RuntimeError(
            f"news parquet is unavailable under {data_root}; "
            "run the Tushare news download task first"
        )
    available = set(
        _read_parquet_union(
            paths, "SELECT * FROM read_parquet(?, union_by_name=true) LIMIT 0"
        ).columns
    )
    if "datetime" not in available:
        raise RuntimeError("news parquet misses required columns: ['datetime']")
    clauses = []
    parameters: list[object] = []
    if start is not None:
        clauses.append("TRY_CAST(datetime AS TIMESTAMP) >= ?")
        parameters.append(datetime.combine(start, datetime.min.time()))
    if end is not None:
        clauses.append("TRY_CAST(datetime AS TIMESTAMP) < ?")
        parameters.append(datetime.combine(end + timedelta(days=1), datetime.min.time()))
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    frame = _read_parquet_union(
        paths,
        "SELECT CAST(datetime AS VARCHAR) AS datetime "
        f"FROM read_parquet(?, union_by_name=true){where}",
        parameters,
    )
    moments = pd.to_datetime(frame["datetime"], errors="coerce")
    return moments.dropna().sort_values(kind="stable").reset_index(drop=True)


def build_daily_counts(
    moments: pd.Series, open_days: Any, *, window_days: int = TRAILING_WINDOW_DAYS
) -> pd.DataFrame:
    """Map flashes onto factor dates and zero-fill every open day in range.

    The trailing-denominator recompute needs the full open-day grid (missing
    days are real zeros on this grid, distinct from "no data"), so the
    intermediate covers every open day between the first and last factor date.
    """

    if window_days < 1:
        raise ValueError("window_days must be positive")
    if moments.empty:
        return pd.DataFrame(
            {
                "factor_date": pd.Series(dtype="datetime64[ns]"),
                "flash_count": pd.Series(dtype="int64"),
            }
        )
    factor_days = [
        factor_date_for(pd.Timestamp(moment).to_pydatetime(), open_days)
        for moment in moments
    ]
    counts = pd.Series(factor_days).value_counts().sort_index()
    grid = [day for day in open_days if counts.index[0] <= day <= counts.index[-1]]
    frame = pd.DataFrame({"factor_date": pd.to_datetime(grid)})
    frame["flash_count"] = (
        frame["factor_date"].dt.date.map(counts).fillna(0).astype("int64")
    )
    return frame[list(COUNTS_COLUMNS)]


def build_intensity_series(
    counts: pd.DataFrame,
    *,
    window_days: int = TRAILING_WINDOW_DAYS,
    min_history_days: int = MIN_HISTORY_DAYS,
    name: str = INTENSITY_FACTOR_NAME,
) -> pd.Series:
    """flash_count / trailing strictly-prior open-day mean, per factor date."""

    if window_days < 1 or min_history_days < 1:
        raise ValueError("window_days and min_history_days must be positive")
    if counts.empty:
        empty = pd.Series(dtype="float64", name=name)
        empty.index = pd.MultiIndex.from_arrays(
            [[], []], names=["datetime", "instrument"]
        )
        return empty
    series = counts.set_index("factor_date")["flash_count"].astype(float)
    history = series.shift(1).rolling(window_days, min_periods=min_history_days).mean()
    intensity = (series / history).dropna()
    intensity = intensity[series.loc[intensity.index] > 0]
    frame = pd.DataFrame({"datetime": intensity.index, "value": intensity.to_numpy()})
    frame["instrument"] = MARKET_INSTRUMENT
    result = frame.set_index(["datetime", "instrument"])["value"].rename(name)
    return normalize_series(result, name)


def _write_factor_artifact(
    series: pd.Series,
    factors_dir: Path,
    *,
    name: str,
    now: datetime,
    start: date | None,
    end: date | None,
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
            "dataset": DATASET,
            "producer_version": PRODUCER_VERSION,
            "trailing_window_days": TRAILING_WINDOW_DAYS,
            "min_history_days": MIN_HISTORY_DAYS,
            "requested_start": start.isoformat() if start is not None else None,
            "requested_end": end.isoformat() if end is not None else None,
        },
        "instrument_convention": (
            f"{MARKET_INSTRUMENT} is a pseudo-instrument code for market-level "
            "series that carry no ts_code"
        ),
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
class NewsFlashSummary:
    flashes: int
    count_days: int
    counts_path: Path
    factors: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "succeeded",
            "flashes": self.flashes,
            "count_days": self.count_days,
            "counts_path": str(self.counts_path),
            "factors": {
                name: {
                    "manifest_path": str(entry["manifest_path"]),
                    "sha256": entry["manifest"]["sha256"],
                    "rows": entry["manifest"]["rows"],
                }
                for name, entry in self.factors.items()
            },
        }


def process_news_flash(
    data_root: Path,
    *,
    window_days: int = TRAILING_WINDOW_DAYS,
    min_history_days: int = MIN_HISTORY_DAYS,
    start: date | None = None,
    end: date | None = None,
    now: Callable[[], datetime] | None = None,
) -> NewsFlashSummary:
    """Build the daily flash counts and the intensity factor artifact.

    Deterministic and idempotent: every run recomputes from the persisted news
    parquets plus trade_cal and atomically rewrites the outputs. Fails closed
    when news/trade_cal parquets are missing or the calendar cannot cover a
    publication timestamp.
    """

    clock = now or (lambda: datetime.now(UTC))
    if start is not None and end is not None and end < start:
        raise ValueError("end must not be before start")
    moments = load_news_flash_datetimes(data_root, start=start, end=end)
    # The trading calendar drives factor dates and the trailing grid; without
    # it the run must not guess (fail closed).
    open_days = load_trade_calendar_open_days(data_root)
    counts = build_daily_counts(moments, open_days, window_days=window_days)
    if start is not None and not counts.empty:
        counts = counts.loc[counts["factor_date"].dt.date >= start].reset_index(
            drop=True
        )
    if end is not None and not counts.empty:
        counts = counts.loc[counts["factor_date"].dt.date <= end].reset_index(
            drop=True
        )
    series = build_intensity_series(
        counts, window_days=window_days, min_history_days=min_history_days
    )

    base = data_root / NEWS_FLASH_DIR
    factors_dir = base / "factors"
    factors_dir.mkdir(parents=True, exist_ok=True)
    counts_path = base / "daily_counts.parquet"
    _write_parquet_atomic(counts, counts_path)
    artifact = _write_factor_artifact(
        series,
        factors_dir,
        name=INTENSITY_FACTOR_NAME,
        now=clock(),
        start=start,
        end=end,
    )
    return NewsFlashSummary(
        flashes=int(len(moments)),
        count_days=int(len(counts)),
        counts_path=counts_path,
        factors={INTENSITY_FACTOR_NAME: artifact},
    )


# ---------------------------------------------------------------------------
# Registration into factor_candidates (generic external-factor channel)
# ---------------------------------------------------------------------------

_FACTOR_DESCRIPTION = (
    "News-flash intensity: daily count of 9-source market news flashes divided "
    "by its trailing strictly-prior open-day mean — a market-level news-flow / "
    "attention regime variable under the MARKET pseudo-instrument. "
    "Deliberately not another sentiment score: flash sentiment would duplicate "
    "the major_news market-level news_sentiment_daily factor at a much higher "
    "LLM cost, while the volume dynamic is the increment major_news cannot "
    "provide."
)

_FACTOR_FORMULATION = (
    f"flash_count(D) / mean(flash_count over the trailing {TRAILING_WINDOW_DAYS} "
    f"trade_cal open days strictly before D, min {MIN_HISTORY_DAYS} prior days); "
    "factor date = publication date when available before 15:00 on a trade_cal "
    "trading day, otherwise the next trade_cal trading day; zero-flash days "
    "emit no value"
)


def _code_artifact_source(*, manifest: dict[str, Any], values_sha256: str) -> str:
    """Deterministic provenance code bound to factor_candidates.code_sha256."""

    source = manifest["source"]
    policy = manifest["availability_policy"][INTENSITY_FACTOR_NAME]
    return f'''"""Provenance code artifact for the {INTENSITY_FACTOR_NAME} factor.

Generated at factor-registration time by quant_platform.news_flash_factors.
The registered factor values derive from the persisted daily_counts.parquet
intermediate (every trade_cal open day in range, zero-filled), normalized with
the factor_evaluator.normalize_series contract.

source dataset: {source["dataset"]}
producer_version: {source["producer_version"]}
trailing_window_days: {source["trailing_window_days"]}
min_history_days: {source["min_history_days"]}
availability_policy: {policy}
values sha256: {values_sha256}
"""

from __future__ import annotations

import pandas as pd

from quant_platform.factor_evaluator import normalize_series

FACTOR_NAME = {INTENSITY_FACTOR_NAME!r}
WINDOW_DAYS = {source["trailing_window_days"]}
MIN_HISTORY_DAYS = {source["min_history_days"]}


def compute_factor(frame: pd.DataFrame) -> pd.Series:
    """Rebuild the factor values from the persisted daily_counts intermediate."""

    series = frame.set_index("factor_date")["flash_count"].astype(float)
    history = series.shift(1).rolling(WINDOW_DAYS, min_periods=MIN_HISTORY_DAYS).mean()
    intensity = (series / history).dropna()
    intensity = intensity[series.loc[intensity.index] > 0]
    out = pd.DataFrame({{"datetime": intensity.index, "value": intensity.to_numpy()}})
    out["instrument"] = "MARKET"
    result = out.set_index(["datetime", "instrument"])["value"].rename(FACTOR_NAME)
    return normalize_series(result, FACTOR_NAME)
'''


def _news_flash_metadata(manifest: dict[str, Any], values_sha256: str) -> ExternalFactorMetadata:
    source = manifest["source"]
    policy = manifest["availability_policy"]
    return ExternalFactorMetadata(
        description=(
            f"{_FACTOR_DESCRIPTION} Availability: "
            f"{policy[INTENSITY_FACTOR_NAME]}. Externally produced by "
            f"news_flash_factors (producer_version={source['producer_version']})."
        ),
        formulation=_FACTOR_FORMULATION,
        variables={
            "availability_policy": policy,
            "source": source,
            "values_sha256": values_sha256,
            "manifest": None,  # filled by the caller with the manifest path
            "rows": manifest["rows"],
            "ingested_fields": ["available_at", "ingested_at"],
        },
        code_source=_code_artifact_source(manifest=manifest, values_sha256=values_sha256),
        run_config={
            "producer_version": source["producer_version"],
            "availability_policy": policy,
        },
        rdagent_feedback=(
            "externally produced news flash intensity factor; "
            "manifest sha256 verified at registration"
        ),
    )


def register_news_flash_factor(
    store: ResearchStore,
    factors_dir: Path,
    *,
    factor_name: str = INTENSITY_FACTOR_NAME,
    actor: str = IMPORT_ACTOR,
) -> dict[str, Any]:
    """Verify and register the news flash factor artifact; idempotent.

    Uses the generic external-factor channel
    (``announcement_factor_registry.register_external_factor``): manifest
    sha256 fail-closed verification, research-run lineage, idempotency key
    (name, values_sha256).
    """

    if factor_name not in FACTOR_NAMES:
        raise ValueError(
            f"unknown news flash factor {factor_name!r}; expected one of {list(FACTOR_NAMES)}"
        )

    def build_metadata(manifest: dict[str, Any], values_sha256: str) -> ExternalFactorMetadata:
        metadata = _news_flash_metadata(manifest, values_sha256)
        metadata.variables["manifest"] = str(factors_dir / f"{factor_name}.json")
        return metadata

    return register_external_factor(
        store,
        factors_dir,
        factor_name=factor_name,
        run_kind=IMPORT_RUN_KIND,
        actor=actor,
        build_metadata=build_metadata,
        source_dataset=DATASET,
        required_source_keys=("producer_version",),
    )
