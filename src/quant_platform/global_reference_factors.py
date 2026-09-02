"""Structured global-reference factors from peripheral market data (no LLM).

Peripheral sessions (US/HK dailies, global broad indexes, US treasury yields)
close after the A-share close of the same calendar date, so every source row
dated D becomes knowable for A-share research on the next calendar day
(``foreign_close_next_calendar_day`` in quant_data.availability).  Factor
values therefore land on the first trade_cal open day strictly after the
source session date and use the MARKET pseudo-instrument
(``market_timeseries`` shape in external_factor_evaluation).

Everything here is deterministic: identical persisted inputs yield identical
artifact sha256 values, and every missing input fails closed instead of being
fabricated.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from quant_data.cninfo_announcements import (
    _parquet_files,
    _read_parquet_union,
    load_trade_calendar_open_days,
    next_trading_day,
)

from .announcement_factor_registry import (
    ExternalFactorMetadata,
    register_external_factor,
)
from .announcement_nlp import _sha256_file, _write_json_atomic, _write_parquet_atomic
from .external_factor_evaluation import MARKET_INSTRUMENT
from .factor_evaluator import normalize_series
from .global_reference_sector_map import (
    SECTOR_LEADERS,
    SECTOR_SLUGS,
    sector_factor_name,
    sector_leader_map_identity,
)

if TYPE_CHECKING:
    from .research_store import ResearchStore

PRODUCER_VERSION = "global-reference-factors.v2"

GLOBAL_REFERENCE_DIR = "global_reference"

SPX_OVERNIGHT_FACTOR_NAME = "global_ref_spx_overnight"
IXIC_OVERNIGHT_FACTOR_NAME = "global_ref_ixic_overnight"
HSI_OVERNIGHT_FACTOR_NAME = "global_ref_hsi_overnight"
US10Y_CHANGE_FACTOR_NAME = "global_ref_us10y_change"
SECTOR_FACTOR_NAMES = tuple(sector_factor_name(slug) for slug in SECTOR_SLUGS)
FACTOR_NAMES = (
    SPX_OVERNIGHT_FACTOR_NAME,
    IXIC_OVERNIGHT_FACTOR_NAME,
    HSI_OVERNIGHT_FACTOR_NAME,
    *SECTOR_FACTOR_NAMES,
    US10Y_CHANGE_FACTOR_NAME,
)

IMPORT_RUN_KIND = "global_reference_factor_import"
IMPORT_ACTOR = "global-reference-registrar"

# Global broad indexes confirmed present in index_global on the production
# relay (plan stage A, 2026-09-02).  Codes match the provider's ts_code.
INDEX_FACTOR_CODES = {
    SPX_OVERNIGHT_FACTOR_NAME: "SPX",
    IXIC_OVERNIGHT_FACTOR_NAME: "IXIC",
    HSI_OVERNIGHT_FACTOR_NAME: "HSI",
}
# Sector transmission is proxied by the governed GICS sector-leader baskets in
# global_reference_sector_map (versioned, hash-bound into factor manifests).
# us_tycr column holding the 10-year treasury yield (percent).
US10Y_COLUMN = "y10"

# Shared availability wording: the dated foreign session closes after the
# A-share close, so the row is knowable from the next calendar day; the factor
# lands on the first trade_cal open day strictly after the source date.
AVAILABILITY_POLICY = {
    name: (
        "available_at = first trade_cal trading day strictly after the source "
        "session date (foreign_close_next_calendar_day: a foreign session "
        "dated D closes after the A-share close of D and is knowable from the "
        "next calendar day); factor_date = available_at; market_timeseries "
        "shape on the MARKET pseudo-instrument"
    )
    for name in FACTOR_NAMES
}

_FACTOR_DATASETS = {
    SPX_OVERNIGHT_FACTOR_NAME: "index_global",
    IXIC_OVERNIGHT_FACTOR_NAME: "index_global",
    HSI_OVERNIGHT_FACTOR_NAME: "index_global",
    **{name: "us_daily" for name in SECTOR_FACTOR_NAMES},
    US10Y_CHANGE_FACTOR_NAME: "us_tycr",
}


def global_reference_factors_dir(data_root: Path) -> Path:
    """Return the directory where global-reference factor artifacts land."""

    return data_root / GLOBAL_REFERENCE_DIR / "factors"


def _date_select(column: str) -> str:
    return (
        f"coalesce(try_cast({column} AS DATE), "
        f"try_strptime(CAST({column} AS VARCHAR), '%Y%m%d')::DATE) AS {column}"
    )


def _load_market_frame(
    data_root: Path,
    dataset: str,
    *,
    value_columns: tuple[str, ...],
    symbols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Read a peripheral daily dataset from the units/snapshots layout.

    Fail closed: missing parquets or missing required columns raise instead of
    fabricating coverage.  Rows duplicated across units and snapshots collapse
    onto one row.  ``value_columns`` are optional numeric candidates (the
    return column differs by provider revision); the frame keeps whichever
    exist and the caller decides whether that is enough.  ``symbols`` pushes a
    ts_code whitelist down into the scan so the full peripheral cross-section
    (millions of rows) never leaves the parquet reader; both the units and the
    snapshots layers are filtered at scan time.
    """

    paths = _parquet_files(data_root, dataset)
    if not paths:
        raise RuntimeError(
            f"{dataset} parquet is unavailable under {data_root}; "
            "run the supplemental peripheral download task first"
        )
    key_column = "date" if dataset.startswith("us_t") else "trade_date"
    available = set(
        _read_parquet_union(
            paths, "SELECT * FROM read_parquet(?, union_by_name=true) LIMIT 0"
        ).columns
    )
    required = {key_column} if dataset == "us_tycr" else {"ts_code", key_column}
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"{dataset} parquet misses required columns: {missing}")
    select_parts = [_date_select(key_column)]
    if "ts_code" in required:
        select_parts.insert(0, "CAST(ts_code AS VARCHAR) AS ts_code")
    # ingested_at drives generation-aware deduplication below; legacy fixture
    # parquets without it fall back to a plain deterministic collapse.
    has_ingested_at = "ingested_at" in available
    if has_ingested_at:
        select_parts.append("ingested_at")
    observed_values: list[str] = []
    for column in value_columns:
        if column in available:
            select_parts.append(f'try_cast("{column}" AS DOUBLE) AS "{column}"')
            observed_values.append(column)
    where = ""
    if symbols and "ts_code" in required:
        whitelist = ", ".join(
            "'" + code.replace("'", "''") + "'" for code in sorted(symbols)
        )
        where = f" WHERE CAST(ts_code AS VARCHAR) IN ({whitelist})"
    frame = _read_parquet_union(
        paths,
        f"SELECT {', '.join(select_parts)} "
        f"FROM read_parquet(?, union_by_name=true){where}",
    )
    for column in observed_values:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[key_column])
    frame[key_column] = pd.to_datetime(frame[key_column]).dt.normalize()
    subset = ["ts_code", key_column] if "ts_code" in frame.columns else [key_column]
    if "ts_code" in frame.columns:
        frame["ts_code"] = frame["ts_code"].astype(str).str.strip().str.upper()
        frame = frame[frame["ts_code"] != ""]
    # The same provider row can live in both the units and the snapshots
    # layout, and revisioned peripheral datasets (adjusted prices, backfilled
    # fields) republish a business key across ingestion generations.  Keep the
    # latest generation per key, mirroring storage._snapshot_source_query's
    # LATEST_GENERATION_KEYS rule; identical rows always collapse.
    if has_ingested_at:
        frame = frame.sort_values(
            [*subset, "ingested_at"], na_position="first", kind="stable"
        )
    else:
        frame = frame.sort_values(subset, kind="stable")
    frame = frame.drop_duplicates(subset=subset, keep="last")
    return frame.sort_values(subset, kind="stable").reset_index(drop=True)


def _sample_ts_codes(data_root: Path, dataset: str, limit: int = 10) -> list[str]:
    """Small provider ts_code sample for fail-closed diagnostics only."""

    paths = _parquet_files(data_root, dataset)
    if not paths:
        return []
    frame = _read_parquet_union(
        paths,
        "SELECT DISTINCT CAST(ts_code AS VARCHAR) AS ts_code "
        f"FROM read_parquet(?, union_by_name=true) LIMIT {int(limit)}",
    )
    return sorted(frame["ts_code"].astype(str))


def _daily_return(frame: pd.DataFrame, *, dataset: str) -> pd.Series:
    """Fractional daily return: provider pct_chg (percent) or close/pre_close."""

    if "pct_chg" in frame.columns and frame["pct_chg"].notna().any():
        return frame["pct_chg"] / 100.0
    if {"close", "pre_close"} <= set(frame.columns):
        denominator = frame["pre_close"].where(frame["pre_close"] != 0)
        return frame["close"] / denominator - 1.0
    raise RuntimeError(
        f"{dataset} parquet provides neither pct_chg nor close/pre_close; "
        "the daily return cannot be derived without fabricating values"
    )


def _to_market_series(
    rows: pd.DataFrame, *, name: str, open_days: list[date]
) -> pd.Series:
    """Map source dates onto the A-share PIT grid and normalize the shape."""

    if rows.empty:
        raise RuntimeError(f"{name}: no source rows survive the requested window")
    frame = rows.copy()
    frame["factor_date"] = [
        # next_trading_day works on plain dates; normalize the pandas key type.
        next_trading_day(day.date() if isinstance(day, pd.Timestamp) else day, open_days)
        for day in frame["source_date"]
    ]
    # A-share holidays compress several foreign sessions onto one A-share open
    # day (e.g. a US Monday session when A-shares are closed).  All of them are
    # knowable by that pre-open; keep the latest source session's value.
    frame = frame.sort_values("source_date", kind="stable")
    frame = frame.drop_duplicates(subset=["factor_date"], keep="last")
    factor_dates = list(frame["factor_date"])
    rows = frame
    series = pd.Series(
        rows["value"].to_numpy(dtype=float),
        index=pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(pd.Series(factor_dates)),
                [MARKET_INSTRUMENT] * len(rows),
            ],
            names=["datetime", "instrument"],
        ),
        name=name,
    )
    return normalize_series(series, name)


def build_global_reference_series(
    data_root: Path,
    *,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, pd.Series]:
    """Compute every global-reference factor series; fail closed on gaps.

    The A-share trade calendar is the only date authority: without it the PIT
    grid cannot be derived and the run must not guess.
    """

    open_days = load_trade_calendar_open_days(data_root)

    def _window(frame: pd.DataFrame, key_column: str) -> pd.DataFrame:
        if start is not None:
            frame = frame[frame[key_column] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame[key_column] <= pd.Timestamp(end)]
        return frame.reset_index(drop=True)

    frames: dict[str, pd.DataFrame] = {}

    index_frame = _window(
        _load_market_frame(
            data_root,
            "index_global",
            value_columns=("pct_chg", "close", "pre_close"),
            symbols=tuple(set(INDEX_FACTOR_CODES.values())),
        ),
        "trade_date",
    )
    for name, code in INDEX_FACTOR_CODES.items():
        rows = index_frame[index_frame["ts_code"] == code].copy()
        if rows.empty:
            sample = _sample_ts_codes(data_root, "index_global")
            raise RuntimeError(
                f"index_global has no rows for {code} (observed ts_code sample: "
                f"{sample}); the overnight factor {name} cannot be built without "
                "the real index series"
            )
        rows["value"] = _daily_return(rows, dataset="index_global")
        rows = rows.dropna(subset=["value"])
        frames[name] = rows.rename(columns={"trade_date": "source_date"})[
            ["source_date", "value"]
        ]

    us_frame = _window(
        _load_market_frame(
            data_root,
            "us_daily",
            value_columns=("pct_chg", "close", "pre_close"),
            symbols=tuple(
                sorted({code for _label, codes in SECTOR_LEADERS.values() for code in codes})
            ),
        ),
        "trade_date",
    )
    for slug in SECTOR_SLUGS:
        label, members = SECTOR_LEADERS[slug]
        name = sector_factor_name(slug)
        basket = us_frame[us_frame["ts_code"].isin(members)].copy()
        missing_codes = sorted(set(members) - set(basket["ts_code"].unique()))
        if missing_codes:
            sample = _sample_ts_codes(data_root, "us_daily")
            raise RuntimeError(
                f"us_daily has no rows for {label} sector basket members "
                f"{missing_codes} (observed ts_code sample: {sample}); the "
                "sector proxy cannot be synthesized without all members"
            )
        basket["value"] = _daily_return(basket, dataset="us_daily")
        basket = basket.dropna(subset=["value"])
        # Equal weight, and only on dates where every member trades.
        member_counts = basket.groupby("trade_date")["ts_code"].nunique()
        complete_dates = member_counts[member_counts == len(members)].index
        basket = basket[basket["trade_date"].isin(complete_dates)]
        grouped = basket.groupby("trade_date", sort=True)["value"].mean()
        frames[name] = grouped.rename_axis("source_date").reset_index()

    treasury = _window(
        _load_market_frame(data_root, "us_tycr", value_columns=(US10Y_COLUMN,)),
        "date",
    )
    if US10Y_COLUMN not in treasury.columns or treasury[US10Y_COLUMN].notna().sum() == 0:
        raise RuntimeError(
            f"us_tycr parquet lacks the {US10Y_COLUMN} 10-year yield column; "
            f"observed columns: {sorted(treasury.columns)}"
        )
    treasury = treasury.dropna(subset=[US10Y_COLUMN]).sort_values("date", kind="stable")
    treasury["value"] = treasury[US10Y_COLUMN].diff()
    treasury = treasury.dropna(subset=["value"])
    frames[US10Y_CHANGE_FACTOR_NAME] = treasury.rename(columns={"date": "source_date"})[
        ["source_date", "value"]
    ]

    return {
        name: _to_market_series(frames[name], name=name, open_days=open_days)
        for name in FACTOR_NAMES
    }


def _write_factor_artifact(
    series: pd.Series,
    factors_dir: Path,
    *,
    name: str,
    now: datetime,
    source_window: dict[str, Any],
) -> dict[str, Any]:
    """Write the normalized factor-values parquet plus its sha256 manifest."""

    artifact_path = factors_dir / f"{name}.parquet"
    _write_parquet_atomic(series.rename(name).reset_index(), artifact_path)
    source: dict[str, Any] = {
        "dataset": GLOBAL_REFERENCE_DIR,
        "source_dataset": _FACTOR_DATASETS[name],
        "producer_version": PRODUCER_VERSION,
        "observation_cadence": "daily",
        "instrument": MARKET_INSTRUMENT,
        **source_window,
    }
    if name in SECTOR_FACTOR_NAMES:
        # Sector factors are bound to the exact governed basket mapping.
        identity = sector_leader_map_identity()
        source["sector_leader_map_version"] = identity["version"]
        source["sector_leader_map_sha256"] = identity["sha256"]
    manifest: dict[str, Any] = {
        "factor": name,
        "artifact": artifact_path.name,
        "sha256": _sha256_file(artifact_path),
        "rows": int(len(series)),
        "availability_policy": {name: AVAILABILITY_POLICY[name]},
        "source": source,
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
class GlobalReferenceSummary:
    series_rows: int
    series_path: Path
    factors: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "succeeded",
            "series_rows": self.series_rows,
            "series_path": str(self.series_path),
            "factors": {
                name: {
                    "manifest_path": str(entry["manifest_path"]),
                    "sha256": entry["manifest"]["sha256"],
                    "rows": entry["manifest"]["rows"],
                }
                for name, entry in self.factors.items()
            },
        }


def process_global_reference(
    data_root: Path,
    *,
    start: date | None = None,
    end: date | None = None,
    now: Callable[[], datetime] | None = None,
) -> GlobalReferenceSummary:
    """Build the shared daily-series intermediate and the factor artifacts.

    Deterministic and idempotent: every run recomputes from the persisted
    peripheral parquets plus trade_cal and atomically rewrites the outputs, so
    identical inputs yield identical artifact sha256 values.
    """

    clock = now or (lambda: datetime.now(UTC))
    series = build_global_reference_series(data_root, start=start, end=end)

    # One persisted intermediate feeds every factor's provenance code artifact.
    intermediate = pd.concat(
        [
            pd.DataFrame(
                {
                    "factor": name,
                    "factor_date": value.index.get_level_values("datetime"),
                    "value": value.to_numpy(dtype=float),
                }
            )
            for name, value in series.items()
        ],
        ignore_index=True,
    ).sort_values(["factor", "factor_date"], kind="stable")
    base = data_root / GLOBAL_REFERENCE_DIR
    factors_dir = base / "factors"
    factors_dir.mkdir(parents=True, exist_ok=True)
    series_path = base / "daily_series.parquet"
    _write_parquet_atomic(intermediate, series_path)

    source_window = {
        "start_date": start.isoformat() if start else None,
        "end_date": end.isoformat() if end else None,
        "index_codes": ",".join(INDEX_FACTOR_CODES.values()),
        "us10y_column": US10Y_COLUMN,
    }
    artifacts = {
        name: _write_factor_artifact(
            series[name], factors_dir, name=name, now=clock(),
            source_window=source_window,
        )
        for name in FACTOR_NAMES
    }
    return GlobalReferenceSummary(
        series_rows=int(len(intermediate)),
        series_path=series_path,
        factors=artifacts,
    )


# ---------------------------------------------------------------------------
# Registration into factor_candidates (generic external-factor channel)
# ---------------------------------------------------------------------------

_SECTOR_DESCRIPTIONS = {
    sector_factor_name(slug): (
        f"Equal-weight daily return of the GICS {label} sector leader basket "
        f"({', '.join(members)}) from US daily bars, only on dates where every "
        "member trades."
    )
    for slug, (label, members) in SECTOR_LEADERS.items()
}

_SECTOR_FORMULATIONS = {
    sector_factor_name(slug): (
        f"mean of member daily returns across {', '.join(members)} per source "
        "date, dates missing any member excluded; factor_date = first "
        "trade_cal open day strictly after the source date"
    )
    for slug, (_label, members) in SECTOR_LEADERS.items()
}

_FACTOR_DESCRIPTIONS = {
    SPX_OVERNIGHT_FACTOR_NAME: (
        "Overnight S&P 500 return (fractional) from the global index dataset, "
        "visible to A-share research from the next calendar day."
    ),
    IXIC_OVERNIGHT_FACTOR_NAME: (
        "Overnight Nasdaq Composite return (fractional) from the global index "
        "dataset, visible to A-share research from the next calendar day."
    ),
    HSI_OVERNIGHT_FACTOR_NAME: (
        "Overnight Hang Seng Index return (fractional) from the global index "
        "dataset, visible to A-share research from the next calendar day."
    ),
    **_SECTOR_DESCRIPTIONS,
    US10Y_CHANGE_FACTOR_NAME: (
        "Daily first difference of the US 10-year treasury yield "
        f"(us_tycr.{US10Y_COLUMN}, percentage points)."
    ),
}

_FACTOR_FORMULATIONS = {
    SPX_OVERNIGHT_FACTOR_NAME: (
        "pct_chg/100 (provider) or close/pre_close - 1 for SPX per source date; "
        "factor_date = first trade_cal open day strictly after the source date"
    ),
    IXIC_OVERNIGHT_FACTOR_NAME: (
        "pct_chg/100 (provider) or close/pre_close - 1 for IXIC per source date; "
        "factor_date = first trade_cal open day strictly after the source date"
    ),
    HSI_OVERNIGHT_FACTOR_NAME: (
        "pct_chg/100 (provider) or close/pre_close - 1 for HSI per source date; "
        "factor_date = first trade_cal open day strictly after the source date"
    ),
    **_SECTOR_FORMULATIONS,
    US10Y_CHANGE_FACTOR_NAME: (
        f"diff of us_tycr.{US10Y_COLUMN} ordered by date; factor_date = first "
        "trade_cal open day strictly after the source date"
    ),
}


def _code_artifact_source(
    *, factor_name: str, manifest: dict[str, Any], values_sha256: str
) -> str:
    """Deterministic provenance code bound to factor_candidates.code_sha256."""

    source = manifest["source"]
    policy = manifest["availability_policy"][factor_name]
    return f'''"""Provenance code artifact for the externally produced {factor_name} factor.

Generated at factor-registration time by quant_platform.global_reference_factors.
The registered factor values derive from the persisted global_reference
daily_series.parquet intermediate (structured peripheral market data, no LLM),
filtered to this factor and normalized with the
factor_evaluator.normalize_series contract.  available_at/factor_date is the
first trade_cal trading day strictly after the source session date.

source dataset: {source["dataset"]}
producer_version: {source["producer_version"]}
availability_policy: {policy}
values sha256: {values_sha256}
"""

from __future__ import annotations

import pandas as pd

from quant_platform.external_factor_evaluation import MARKET_INSTRUMENT
from quant_platform.factor_evaluator import normalize_series

FACTOR_NAME = {factor_name!r}


def compute_factor(frame: pd.DataFrame) -> pd.Series:
    """Rebuild the factor values from the persisted daily_series intermediate."""

    frame = frame[frame["factor"] == FACTOR_NAME].copy()
    frame["instrument"] = MARKET_INSTRUMENT
    series = frame.set_index(["factor_date", "instrument"])["value"].astype(float)
    series.index = series.index.set_names(["datetime", "instrument"])
    return normalize_series(series, FACTOR_NAME)
'''


def _global_reference_metadata(
    factor_name: str, manifest: dict[str, Any], values_sha256: str
) -> ExternalFactorMetadata:
    source = manifest["source"]
    policy = manifest["availability_policy"]
    return ExternalFactorMetadata(
        description=(
            f"{_FACTOR_DESCRIPTIONS[factor_name]} Availability: "
            f"{policy[factor_name]}. Externally produced by "
            f"global_reference_factors (producer_version={source['producer_version']})."
        ),
        formulation=_FACTOR_FORMULATIONS[factor_name],
        variables={
            "availability_policy": policy,
            "source": source,
            "values_sha256": values_sha256,
            "manifest": None,  # filled by the caller with the manifest path
            "rows": manifest["rows"],
        },
        code_source=_code_artifact_source(
            factor_name=factor_name, manifest=manifest, values_sha256=values_sha256
        ),
        run_config={
            "producer_version": source["producer_version"],
            "availability_policy": policy,
        },
        rdagent_feedback=(
            "externally produced global-reference structured factor; "
            "manifest sha256 verified at registration"
        ),
    )


def _global_reference_provenance_identity(manifest: dict[str, Any]) -> str:
    """Bind registration idempotency to values plus the producer contract."""

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("global-reference manifest misses its source block")
    identity = {
        key: str(source.get(key) or "")
        for key in (
            "producer_version",
            "dataset",
            "source_dataset",
            "start_date",
            "end_date",
            "index_codes",
            "us10y_column",
            "sector_leader_map_version",
            "sector_leader_map_sha256",
        )
    }
    return hashlib.sha256(
        json.dumps(
            identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def register_global_reference_factor(
    store: ResearchStore,
    factors_dir: Path,
    *,
    factor_name: str,
    actor: str = IMPORT_ACTOR,
) -> dict[str, Any]:
    """Verify and register one global-reference factor artifact; idempotent.

    Uses the generic external-factor channel
    (``announcement_factor_registry.register_external_factor``): manifest
    sha256 fail-closed verification, research-run lineage, idempotency key
    (name, values_sha256, provenance identity).
    """

    if factor_name not in FACTOR_NAMES:
        raise ValueError(
            f"unknown global-reference factor {factor_name!r}; "
            f"expected one of {list(FACTOR_NAMES)}"
        )

    def build_metadata(manifest: dict[str, Any], values_sha256: str) -> ExternalFactorMetadata:
        metadata = _global_reference_metadata(factor_name, manifest, values_sha256)
        metadata.variables["manifest"] = str(factors_dir / f"{factor_name}.json")
        return metadata

    return register_external_factor(
        store,
        factors_dir,
        factor_name=factor_name,
        run_kind=IMPORT_RUN_KIND,
        actor=actor,
        build_metadata=build_metadata,
        source_dataset="global_reference",
        required_source_keys=("producer_version",),
        provenance_identity_from_manifest=_global_reference_provenance_identity,
    )
