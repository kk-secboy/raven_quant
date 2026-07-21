"""Structured factor production from the Tushare report_rc sell-side research dataset.

``report_rc`` (券商研报盈利预测与评级) is downloaded by the cn_institutional
task (``quant_data.supplemental_data._cn_institutional_specs``, monthly
``start_date``/``end_date`` pages of 3,000 rows) but had no consumer. This
module is the structured consumer: no LLM is involved — the dataset already
carries machine-readable ratings, target prices and earnings forecasts.

Source schema (Tushare doc_id=292, downloaded with default fields so every
column lands): ts_code, name, report_date (研报发布日期), report_title,
report_type, classify, org_name, author_name, quarter (预测报告期),
op_rt/op_pr/tp/np (预测营业收入/营业利润/利润总额/净利润，万元), eps (预测每股收益，元),
pe, rd, roe, ev_ebitda, rating (卖方评级), max_price/min_price (预测目标价区间).

PIT semantics (design draft 3.3):

- ``report_date`` is the report publication date; the provider refreshes each
  evening (19-22) for that day's reports. The dataset is registered in
  ``quant_data.availability`` as ``strictly_after_announcement_date`` on
  ``report_date`` (conservative: usable the day after publication) with
  recoverability ``native_history``.
- Row-level ``available_at`` = first trade_cal trading day strictly after
  ``report_date`` (never weekday guesses; a calendar too short fails the run).
- Factor values are sampled on a **weekly observation grid** — the last
  trade_cal open day of each ISO week — and only aggregate rows whose
  ``available_at`` is on or before the grid day. Two reasons: (a) the platform
  is medium-low frequency, weekly aggregation matches the rebalancing
  cadence; (b) the external evaluation channel
  (``quant_platform.external_factor_evaluation``) fails closed on factors
  present on more than half of the label days, and raw report events occur on
  essentially every trading day (thousands of rows/day), so daily event
  factors would be rejected as "neither event-sparse nor dense".

Rating scale normalization (``RATING_LADDER_VERSION``): brokers use
incompatible rating vocabularies (买入/增持/中性 vs 强烈推荐/推荐/谨慎推荐 vs
优于大市/与大市同步 ...). Two rules keep this defensible:

1. Absolute levels are never compared across institutions. Rating-change
   events are computed strictly inside one (ts_code, org_name) chain, so the
   delta is meaningful even if two brokers' "买入" differ.
2. The string→level ladder is an exact-match table over common vocabularies;
   unknown ratings fail closed (excluded from change events) instead of being
   guessed. Rows without org_name cannot form an intra-org chain and are
   excluded from events as well. Both still count for coverage.

Outputs land under ``data/report_rc/`` following the announcement/corpus NLP
layout:

- ``fields.parquet`` — normalized report rows with rating_level, available_at,
  factor_date, ingested_at plus the raw forecast columns (eps/np/max_price/
  min_price/quarter) kept as the extension slot for future expectation-gap
  factors (e.g. target-price implied upside, forecast dispersion).
- ``events_rating.parquet`` / ``events_eps.parquet`` — intra-org rating-change
  and EPS-revision events (the per-event source of the factor values).
- ``coverage.parquet`` — weekly trailing-window report counts per ts_code.
- ``factors/<name>.parquet`` + ``factors/<name>.json`` — factor-values
  artifacts with sha256 manifests, registered into ``factor_candidates`` via
  the generic external-factor channel in
  ``quant_platform.announcement_factor_registry.register_external_factor``
  and evaluated by ``external_factor_evaluation`` (sparse_event shape).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
from .factor_evaluator import normalize_series

if TYPE_CHECKING:
    from .research_store import ResearchStore

PRODUCER_VERSION = "report-rc-factors.v1"
RATING_LADDER_VERSION = "rating-ladder.v1"

DATASET = "report_rc"
REPORT_RC_DIR = "report_rc"

RATING_CHANGE_FACTOR_NAME = "report_rc_rating_change"
COVERAGE_FACTOR_NAME = "report_rc_coverage_20d"
EPS_REVISION_FACTOR_NAME = "report_rc_eps_revision"
FACTOR_NAMES = (
    RATING_CHANGE_FACTOR_NAME,
    COVERAGE_FACTOR_NAME,
    EPS_REVISION_FACTOR_NAME,
)

# Trailing window (in trade_cal open days) for the analyst-coverage factor.
COVERAGE_WINDOW_DAYS = 20
# Lower bound for the |previous EPS| denominator of relative revisions; a
# near-zero base would explode the ratio into a meaningless outlier.
EPS_DENOMINATOR_FLOOR = 0.01

IMPORT_RUN_KIND = "report_rc_factor_import"
IMPORT_ACTOR = "report-rc-registrar"

# Exact-match rating ladder. Levels run 1 (most bearish) to 5 (most bullish).
# Anything not listed maps to None and is excluded from change events — the
# table is extended only by bumping RATING_LADDER_VERSION with a change note.
RATING_LEVELS: dict[str, int] = {
    "卖出": 1,
    "回避": 1,
    "减持": 2,
    "谨慎减持": 2,
    "跑输行业": 2,
    "弱于大市": 2,
    "中性": 3,
    "持有": 3,
    "观望": 3,
    "与大市同步": 3,
    "同步大市": 3,
    "增持": 4,
    "谨慎增持": 4,
    "推荐": 4,
    "谨慎推荐": 4,
    "优于大市": 4,
    "跑赢行业": 4,
    "买入": 5,
    "强烈推荐": 5,
    "强推": 5,
}

AVAILABILITY_POLICY = {
    name: (
        "available_at = first trade_cal trading day strictly after report_date "
        "(strictly_after_announcement_date on report_date, conservative: the "
        "provider refreshes the evening of the publication day); weekly "
        "observation grid — the factor value lands on the last trade_cal open "
        "day of the ISO week containing available_at and aggregates only rows "
        "available on or before that day"
    )
    for name in FACTOR_NAMES
}

_REQUIRED_COLUMNS = ("ts_code", "report_date", "org_name", "rating")
# Optional forecast columns: when absent (e.g. a reduced-fields download) the
# EPS-revision factor degrades to empty while the raw-material extension slot
# in fields.parquet stays in place.
_FORECAST_COLUMNS = ("quarter", "eps", "np", "max_price", "min_price")
_TEXT_COLUMNS = ("report_title", "author_name")

FIELDS_COLUMNS = (
    "ts_code",
    "org_name",
    "author_name",
    "report_date",
    "report_title",
    "rating",
    "rating_level",
    "quarter",
    "eps",
    "np",
    "max_price",
    "min_price",
    "available_at",
    "factor_date",
    "ingested_at",
)

RATING_EVENT_COLUMNS = (
    "ts_code",
    "org_name",
    "report_date",
    "factor_date",
    "prev_report_date",
    "prev_rating",
    "new_rating",
    "prev_level",
    "new_level",
    "rating_delta",
)

EPS_EVENT_COLUMNS = (
    "ts_code",
    "org_name",
    "quarter",
    "report_date",
    "factor_date",
    "prev_eps",
    "new_eps",
    "eps_revision",
)

COVERAGE_COLUMNS = ("factor_date", "ts_code", "report_count")


def default_factors_dir(data_root: Path) -> Path:
    """Return the directory where report_rc factor artifacts land."""

    return data_root / REPORT_RC_DIR / "factors"


def rating_level(rating: object) -> int | None:
    """Map a raw broker rating string onto the 1-5 ladder; None if unknown."""

    if rating is None:
        return None
    return RATING_LEVELS.get(str(rating).strip())


def load_report_rc_reports(
    data_root: Path,
    *,
    ts_codes: set[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Read report_rc parquets from the units/snapshots layout; fail closed.

    Raises when no report_rc parquet exists or required columns are missing.
    Rows duplicated across units and snapshots collapse onto one row.
    """

    paths = _parquet_files(data_root, DATASET)
    if not paths:
        raise RuntimeError(
            f"report_rc parquet is unavailable under {data_root}; "
            "run the Tushare cn_institutional download task first"
        )
    available = set(
        _read_parquet_union(
            paths, "SELECT * FROM read_parquet(?, union_by_name=true) LIMIT 0"
        ).columns
    )
    missing = sorted(set(_REQUIRED_COLUMNS) - available)
    if missing:
        raise RuntimeError(f"report_rc parquet misses required columns: {missing}")
    select_parts = [
        "CAST(ts_code AS VARCHAR) AS ts_code",
        "coalesce(try_cast(report_date AS DATE), "
        "try_strptime(CAST(report_date AS VARCHAR), '%Y%m%d')::DATE) AS report_date",
        "CAST(org_name AS VARCHAR) AS org_name",
        "CAST(rating AS VARCHAR) AS rating",
    ]
    for column in (*_FORECAST_COLUMNS, *_TEXT_COLUMNS):
        if column in available:
            if column in {"eps", "np", "max_price", "min_price"}:
                select_parts.append(f'try_cast("{column}" AS DOUBLE) AS "{column}"')
            else:
                select_parts.append(f'CAST("{column}" AS VARCHAR) AS "{column}"')
    frame = _read_parquet_union(
        paths,
        f"SELECT {', '.join(select_parts)} FROM read_parquet(?, union_by_name=true)",
    )
    for column in (*_FORECAST_COLUMNS, *_TEXT_COLUMNS):
        if column not in frame.columns:
            frame[column] = None
    frame = frame.dropna(subset=["ts_code", "report_date"])
    frame["ts_code"] = frame["ts_code"].astype(str).str.strip().str.upper()
    frame = frame[frame["ts_code"] != ""]
    frame["org_name"] = frame["org_name"].fillna("").astype(str).str.strip()
    frame["rating"] = frame["rating"].fillna("").astype(str).str.strip()
    for column in _TEXT_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    frame["quarter"] = frame["quarter"].fillna("").astype(str).str.strip()
    # Identical rows can live in both the units and the snapshots layout.
    frame = frame.drop_duplicates()
    if ts_codes:
        wanted = {code.strip().upper() for code in ts_codes if code.strip()}
        frame = frame[frame["ts_code"].isin(sorted(wanted))]
    if start is not None:
        frame = frame[frame["report_date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["report_date"] <= pd.Timestamp(end)]
    return frame.sort_values(
        ["ts_code", "org_name", "report_date", "quarter"], kind="stable"
    ).reset_index(drop=True)


def weekly_grid_days(open_days: Sequence[date]) -> dict[date, date]:
    """Map each open day to the last open day of its ISO week (the grid day)."""

    grid: dict[tuple[int, int], date] = {}
    for day in open_days:
        iso = day.isocalendar()
        grid[(iso[0], iso[1])] = day  # open_days are sorted: the last write wins
    return {day: grid[(day.isocalendar()[0], day.isocalendar()[1])] for day in open_days}


def _fields_frame(
    reports: pd.DataFrame, open_days: Sequence[date], *, ingested_at: datetime
) -> pd.DataFrame:
    """Attach rating levels and PIT timestamps to the normalized report rows."""

    frame = reports.copy()
    if frame.empty:
        return pd.DataFrame(
            {
                column: pd.Series(dtype="datetime64[ns]")
                if column in {"report_date", "available_at", "factor_date"}
                else pd.Series(dtype="datetime64[ns, UTC]")
                if column == "ingested_at"
                else pd.Series(dtype="float64")
                if column in {"eps", "np", "max_price", "min_price"}
                else pd.Series(dtype="Int64")
                if column == "rating_level"
                else pd.Series(dtype="string")
                for column in FIELDS_COLUMNS
            }
        )
    availability: dict[date, date] = {}
    for value in sorted(frame["report_date"].unique()):
        report_day = pd.Timestamp(value).date()
        try:
            availability[report_day] = next_trading_day(report_day, open_days)
        except LookupError as exc:
            raise RuntimeError(
                f"cannot derive available_at for report_date {report_day}: {exc}"
            ) from exc
    grid = weekly_grid_days(open_days)
    frame["available_at"] = frame["report_date"].map(
        lambda value: pd.Timestamp(availability[pd.Timestamp(value).date()])
    )
    frame["factor_date"] = frame["available_at"].map(
        lambda value: pd.Timestamp(grid[pd.Timestamp(value).date()])
    )
    frame["rating_level"] = frame["rating"].map(rating_level).astype("Int64")
    frame["ingested_at"] = pd.Timestamp(ingested_at)
    return frame[list(FIELDS_COLUMNS)]


def build_rating_change_events(fields: pd.DataFrame) -> pd.DataFrame:
    """Intra-(ts_code, org_name) rating transitions with a non-zero ladder delta.

    Consecutive *known* ratings of the same institution are compared; unknown
    ratings and anonymous rows are excluded (fail closed) but still count for
    coverage. Multiple quarter rows of one report collapse to a single
    report-level row first (deterministic quarter order).
    """

    if fields.empty:
        return pd.DataFrame(columns=list(RATING_EVENT_COLUMNS))
    rated = fields[
        fields["rating_level"].notna() & (fields["org_name"] != "")
    ].copy()
    if rated.empty:
        return pd.DataFrame(columns=list(RATING_EVENT_COLUMNS))
    rated = rated.sort_values(
        ["ts_code", "org_name", "report_date", "quarter"], kind="stable"
    )
    reports = rated.drop_duplicates(["ts_code", "org_name", "report_date"], keep="first")
    grouped = reports.groupby(["ts_code", "org_name"], sort=False)
    prev_level = grouped["rating_level"].shift()
    delta = reports["rating_level"].astype("float64") - prev_level.astype("float64")
    mask = prev_level.notna() & (delta != 0)
    events = reports.loc[mask].copy()
    events["prev_report_date"] = grouped["report_date"].shift()[mask]
    events["prev_rating"] = grouped["rating"].shift()[mask]
    events["new_rating"] = events["rating"]
    events["prev_level"] = prev_level[mask].astype("int64")
    events["new_level"] = events["rating_level"].astype("int64")
    events["rating_delta"] = delta[mask]
    return events[list(RATING_EVENT_COLUMNS)].sort_values(
        ["factor_date", "ts_code", "org_name"], kind="stable"
    ).reset_index(drop=True)


def build_eps_revision_events(
    fields: pd.DataFrame, *, denominator_floor: float = EPS_DENOMINATOR_FLOOR
) -> pd.DataFrame:
    """Intra-(ts_code, org_name, quarter) relative EPS forecast revisions.

    revision = (eps_t - eps_{t-1}) / max(|eps_{t-1}|, denominator_floor). Only
    the nearest forecast quarter of each report is kept — mixing forecast
    horizons in one aggregate is not meaningful. This is the raw material for
    expectation-revision factors; the forecast columns in fields.parquet are
    the extension slot for richer variants (target-price upside, dispersion).
    """

    if fields.empty:
        return pd.DataFrame(columns=list(EPS_EVENT_COLUMNS))
    frame = fields[
        fields["eps"].notna()
        & (fields["quarter"] != "")
        & (fields["org_name"] != "")
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=list(EPS_EVENT_COLUMNS))
    frame = frame.sort_values(
        ["ts_code", "org_name", "quarter", "report_date"], kind="stable"
    )
    frame = frame.drop_duplicates(
        ["ts_code", "org_name", "quarter", "report_date"], keep="first"
    )
    grouped = frame.groupby(["ts_code", "org_name", "quarter"], sort=False)
    prev_eps = grouped["eps"].shift()
    denominator = prev_eps.abs().clip(lower=denominator_floor)
    revision = (frame["eps"] - prev_eps) / denominator
    mask = prev_eps.notna()
    events = frame.loc[mask].copy()
    events["prev_eps"] = prev_eps[mask]
    events["new_eps"] = events["eps"]
    events["eps_revision"] = revision[mask]
    # Nearest forecast quarter per report only (quarter strings like 2024Q4
    # sort chronologically).
    events = events.sort_values(
        ["ts_code", "org_name", "report_date", "quarter"], kind="stable"
    ).drop_duplicates(["ts_code", "org_name", "report_date"], keep="first")
    return events[list(EPS_EVENT_COLUMNS)].sort_values(
        ["factor_date", "ts_code", "org_name"], kind="stable"
    ).reset_index(drop=True)


def build_coverage_frame(
    fields: pd.DataFrame,
    open_days: Sequence[date],
    *,
    window_days: int = COVERAGE_WINDOW_DAYS,
) -> pd.DataFrame:
    """Weekly trailing-window report counts per ts_code (analyst attention).

    For each weekly grid day, count the distinct (org_name, report_date)
    reports whose available_at falls inside the trailing ``window_days``
    trade_cal open days ending on the grid day. Instruments without any report
    in the window carry no value (sparse event shape).
    """

    if window_days < 1:
        raise ValueError("coverage window_days must be positive")
    if fields.empty or not open_days:
        return pd.DataFrame(columns=list(COVERAGE_COLUMNS))
    reports = fields.drop_duplicates(["ts_code", "org_name", "report_date"])
    day_index = pd.DatetimeIndex([pd.Timestamp(day) for day in open_days])
    grid_days = sorted(set(weekly_grid_days(open_days).values()))
    counts = reports.groupby(["available_at", "ts_code"], sort=True).size()
    matrix = counts.unstack("ts_code").reindex(day_index, fill_value=0)
    rolled = matrix.rolling(window_days, min_periods=1).sum()
    grid_index = pd.DatetimeIndex([pd.Timestamp(day) for day in grid_days])
    sampled = rolled.loc[rolled.index.isin(grid_index)]
    long = sampled.reset_index(names="factor_date").melt(
        id_vars="factor_date", var_name="ts_code", value_name="report_count"
    )
    long = long[long["report_count"] > 0]
    long["report_count"] = long["report_count"].astype("int64")
    return long[list(COVERAGE_COLUMNS)].sort_values(
        ["factor_date", "ts_code"], kind="stable"
    ).reset_index(drop=True)


def build_rating_change_series(
    events: pd.DataFrame, name: str = RATING_CHANGE_FACTOR_NAME
) -> pd.Series:
    """Mean intra-week rating delta per (factor_date, ts_code)."""

    grouped = events.groupby(["factor_date", "ts_code"], sort=True)["rating_delta"].mean()
    series = grouped.rename(name)
    series.index = series.index.set_names(["datetime", "instrument"])
    return normalize_series(series, name)


def build_eps_revision_series(
    events: pd.DataFrame, name: str = EPS_REVISION_FACTOR_NAME
) -> pd.Series:
    """Mean intra-week nearest-quarter EPS revision per (factor_date, ts_code)."""

    grouped = events.groupby(["factor_date", "ts_code"], sort=True)["eps_revision"].mean()
    series = grouped.rename(name)
    series.index = series.index.set_names(["datetime", "instrument"])
    return normalize_series(series, name)


def build_coverage_series(
    coverage: pd.DataFrame, name: str = COVERAGE_FACTOR_NAME
) -> pd.Series:
    """Trailing-window report count per (factor_date, ts_code)."""

    series = coverage.set_index(["factor_date", "ts_code"])["report_count"].astype(float)
    series = series.rename(name)
    series.index = series.index.set_names(["datetime", "instrument"])
    return normalize_series(series, name)


def _write_factor_artifact(
    series: pd.Series,
    factors_dir: Path,
    *,
    name: str,
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
            "dataset": DATASET,
            "producer_version": PRODUCER_VERSION,
            "rating_ladder_version": RATING_LADDER_VERSION,
            "observation_cadence": "weekly",
            "coverage_window_days": COVERAGE_WINDOW_DAYS,
            "eps_denominator_floor": EPS_DENOMINATOR_FLOOR,
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
class ReportRcSummary:
    reports: int
    rating_events: int
    eps_events: int
    coverage_rows: int
    fields_path: Path
    rating_events_path: Path
    eps_events_path: Path
    coverage_path: Path
    factors: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "succeeded",
            "reports": self.reports,
            "rating_events": self.rating_events,
            "eps_events": self.eps_events,
            "coverage_rows": self.coverage_rows,
            "fields_path": str(self.fields_path),
            "rating_events_path": str(self.rating_events_path),
            "eps_events_path": str(self.eps_events_path),
            "coverage_path": str(self.coverage_path),
            "factors": {
                name: {
                    "manifest_path": str(entry["manifest_path"]),
                    "sha256": entry["manifest"]["sha256"],
                    "rows": entry["manifest"]["rows"],
                }
                for name, entry in self.factors.items()
            },
        }


def process_report_rc(
    data_root: Path,
    *,
    ts_codes: set[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    coverage_window_days: int = COVERAGE_WINDOW_DAYS,
    now: Callable[[], datetime] | None = None,
) -> ReportRcSummary:
    """Build fields, event/coverage intermediates and factor artifacts.

    Deterministic and idempotent: every run recomputes from the persisted
    report_rc parquets plus trade_cal and atomically rewrites the outputs, so
    identical inputs yield identical artifact sha256 values. Fails closed when
    report_rc/trade_cal parquets are missing or the calendar cannot cover a
    report_date.
    """

    clock = now or (lambda: datetime.now(UTC))
    reports = load_report_rc_reports(data_root, ts_codes=ts_codes, start=start, end=end)
    # The trading calendar drives availability and the weekly grid; without it
    # the run must not guess (fail closed).
    open_days = load_trade_calendar_open_days(data_root)
    fields = _fields_frame(reports, open_days, ingested_at=clock())

    rating_events = build_rating_change_events(fields)
    eps_events = build_eps_revision_events(fields)
    coverage = build_coverage_frame(
        fields, open_days, window_days=coverage_window_days
    )

    base = data_root / REPORT_RC_DIR
    factors_dir = base / "factors"
    factors_dir.mkdir(parents=True, exist_ok=True)
    fields_path = base / "fields.parquet"
    rating_events_path = base / "events_rating.parquet"
    eps_events_path = base / "events_eps.parquet"
    coverage_path = base / "coverage.parquet"
    _write_parquet_atomic(fields, fields_path)
    _write_parquet_atomic(rating_events, rating_events_path)
    _write_parquet_atomic(eps_events, eps_events_path)
    _write_parquet_atomic(coverage, coverage_path)

    artifacts = {
        RATING_CHANGE_FACTOR_NAME: _write_factor_artifact(
            build_rating_change_series(rating_events), factors_dir,
            name=RATING_CHANGE_FACTOR_NAME, now=clock(),
        ),
        COVERAGE_FACTOR_NAME: _write_factor_artifact(
            build_coverage_series(coverage), factors_dir,
            name=COVERAGE_FACTOR_NAME, now=clock(),
        ),
        EPS_REVISION_FACTOR_NAME: _write_factor_artifact(
            build_eps_revision_series(eps_events), factors_dir,
            name=EPS_REVISION_FACTOR_NAME, now=clock(),
        ),
    }
    return ReportRcSummary(
        reports=int(len(reports)),
        rating_events=int(len(rating_events)),
        eps_events=int(len(eps_events)),
        coverage_rows=int(len(coverage)),
        fields_path=fields_path,
        rating_events_path=rating_events_path,
        eps_events_path=eps_events_path,
        coverage_path=coverage_path,
        factors=artifacts,
    )


# ---------------------------------------------------------------------------
# Registration into factor_candidates (generic external-factor channel)
# ---------------------------------------------------------------------------

_FACTOR_DESCRIPTIONS = {
    RATING_CHANGE_FACTOR_NAME: (
        "Analyst rating-change event factor: mean intra-week change in the "
        "normalized 1-5 rating ladder (upgrades positive, downgrades negative), "
        "computed strictly inside each (ts_code, org_name) chain so incompatible "
        "broker scales are never compared across institutions."
    ),
    COVERAGE_FACTOR_NAME: (
        "Analyst-attention factor: number of distinct sell-side reports "
        "covering the stock within the trailing 20 trade_cal open days, sampled "
        "on the weekly grid; classic coverage/attention variable."
    ),
    EPS_REVISION_FACTOR_NAME: (
        "Earnings-forecast revision factor: mean intra-week relative revision "
        "of the nearest-quarter EPS forecast, computed intra "
        "(ts_code, org_name, quarter) with a floored denominator."
    ),
}

_FACTOR_FORMULATIONS = {
    RATING_CHANGE_FACTOR_NAME: (
        "mean over same-ISO-week events of ladder_level(new) - ladder_level(prev) "
        f"per (factor_date, ts_code); chains are intra (ts_code, org_name); "
        f"ladder={RATING_LADDER_VERSION} (exact-match table, unknown ratings "
        "excluded); factor_date = last trade_cal open day of the ISO week "
        "containing available_at"
    ),
    COVERAGE_FACTOR_NAME: (
        f"count of distinct (org_name, report_date) reports with available_at in "
        f"the trailing {COVERAGE_WINDOW_DAYS} trade_cal open days ending on the "
        "weekly grid day, per (factor_date, ts_code); only counts > 0 emitted"
    ),
    EPS_REVISION_FACTOR_NAME: (
        "mean over same-ISO-week events of (eps_t - eps_{t-1}) / "
        f"max(|eps_{{t-1}}|, {EPS_DENOMINATOR_FLOOR}) per (factor_date, ts_code); "
        "chains intra (ts_code, org_name, quarter); nearest forecast quarter "
        "per report only"
    ),
}


def _code_artifact_source(
    *, factor_name: str, manifest: dict[str, Any], values_sha256: str
) -> str:
    """Deterministic provenance code bound to factor_candidates.code_sha256.

    The transformation mirrors the builders above so the registered values can
    be rebuilt from the persisted intermediates (events / coverage frames).
    """

    source = manifest["source"]
    policy = manifest["availability_policy"][factor_name]
    if factor_name == COVERAGE_FACTOR_NAME:
        input_frame = "coverage.parquet"
        value_column = "report_count"
        aggregation = "frame.set_index(['factor_date', 'ts_code'])['report_count'].astype(float)"
    elif factor_name == RATING_CHANGE_FACTOR_NAME:
        input_frame = "events_rating.parquet"
        value_column = "rating_delta"
        aggregation = (
            "frame.groupby(['factor_date', 'ts_code'], sort=True)['rating_delta'].mean()"
        )
    else:
        input_frame = "events_eps.parquet"
        value_column = "eps_revision"
        aggregation = (
            "frame.groupby(['factor_date', 'ts_code'], sort=True)['eps_revision'].mean()"
        )
    return f'''"""Provenance code artifact for the externally produced {factor_name} factor.

Generated at factor-registration time by quant_platform.report_rc_factors.
The registered factor values derive from the report_rc sell-side research
dataset (structured, no LLM); {value_column} is read from the persisted
{input_frame} intermediate and normalized with the
factor_evaluator.normalize_series contract. available_at is the first
trade_cal trading day strictly after report_date; factor_date is the last
trade_cal open day of the ISO week containing available_at.

source dataset: {source["dataset"]}
producer_version: {source["producer_version"]}
rating_ladder_version: {source["rating_ladder_version"]}
availability_policy: {policy}
values sha256: {values_sha256}
"""

from __future__ import annotations

import pandas as pd

from quant_platform.factor_evaluator import normalize_series

FACTOR_NAME = {factor_name!r}


def compute_factor(frame: pd.DataFrame) -> pd.Series:
    """Rebuild the factor values from the persisted {input_frame} intermediate."""

    frame = frame.copy()
    series = {aggregation}
    series = series.rename(FACTOR_NAME)
    series.index = series.index.set_names(["datetime", "instrument"])
    return normalize_series(series, FACTOR_NAME)
'''


def _report_rc_metadata(
    factor_name: str, manifest: dict[str, Any], values_sha256: str
) -> ExternalFactorMetadata:
    source = manifest["source"]
    policy = manifest["availability_policy"]
    return ExternalFactorMetadata(
        description=(
            f"{_FACTOR_DESCRIPTIONS[factor_name]} Availability: "
            f"{policy[factor_name]} — values become visible at available_at, the "
            "first trade_cal trading day strictly after report_date, sampled on "
            "the weekly grid. Externally produced by report_rc_factors "
            f"(producer_version={source['producer_version']}, "
            f"rating_ladder_version={source['rating_ladder_version']})."
        ),
        formulation=_FACTOR_FORMULATIONS[factor_name],
        variables={
            "availability_policy": policy,
            "source": source,
            "values_sha256": values_sha256,
            "manifest": None,  # filled by the caller with the manifest path
            "rows": manifest["rows"],
            "rating_ladder": dict(RATING_LEVELS),
            "ingested_fields": ["available_at", "ingested_at"],
        },
        code_source=_code_artifact_source(
            factor_name=factor_name, manifest=manifest, values_sha256=values_sha256
        ),
        run_config={
            "producer_version": source["producer_version"],
            "rating_ladder_version": source["rating_ladder_version"],
            "availability_policy": policy,
        },
        rdagent_feedback=(
            "externally produced report_rc structured factor; "
            "manifest sha256 verified at registration"
        ),
    )


def register_report_rc_factor(
    store: ResearchStore,
    factors_dir: Path,
    *,
    factor_name: str,
    actor: str = IMPORT_ACTOR,
) -> dict[str, Any]:
    """Verify and register one report_rc factor artifact; idempotent.

    Uses the generic external-factor channel
    (``announcement_factor_registry.register_external_factor``): manifest
    sha256 fail-closed verification, research-run lineage, idempotency key
    (name, values_sha256).
    """

    if factor_name not in FACTOR_NAMES:
        raise ValueError(
            f"unknown report_rc factor {factor_name!r}; expected one of {list(FACTOR_NAMES)}"
        )

    def build_metadata(manifest: dict[str, Any], values_sha256: str) -> ExternalFactorMetadata:
        metadata = _report_rc_metadata(factor_name, manifest, values_sha256)
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
