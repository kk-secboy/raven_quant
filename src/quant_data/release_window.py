from __future__ import annotations

import hashlib
import json
import re
from calendar import monthrange
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from .catalog import ALL_DEFINITIONS
from .reference_data import select_current_reference_units

RELEASE_WINDOW_SELECTOR_VERSION = (
    "release-window-selector-v5-source-history-stk-surv-fina-audit-pagination-bound"
)
QLIB_DAILY_REQUIRED_DATASETS = frozenset(
    {"stock_basic", "trade_cal", "daily", "adj_factor", "daily_basic", "stk_limit"}
)
QLIB_RESEARCH_REQUIRED_DATASETS = frozenset(
    {
        *QLIB_DAILY_REQUIRED_DATASETS,
        "moneyflow",
        "fina_indicator",
        "index_weight",
        "balancesheet",
        "fina_audit",
        "namechange",
        "index_member_all",
    }
)

_RANGE_KEYS = (
    ("partition_start", "partition_end"),
    ("expected_date_start", "expected_date_end"),
    ("window_start", "window_end"),
    ("source_start_date", "source_end_date"),
    ("query_start", "query_end"),
    ("day_start", "day_end"),
    ("active_start", "active_end"),
    ("start_date", "end_date"),
    ("start_m", "end_m"),
    ("start_week", "end_week"),
    ("start", "end"),
)

_POINT_DATE_KEYS = (
    "trade_date",
    "trading_date",
    "cal_date",
    "expected_date",
    "ann_date",
    "f_ann_date",
    "actual_ann_date",
    "announcement_date",
    "publish_date",
    "pub_date",
    "imp_date",
    "report_date",
    "event_date",
    "ex_date",
    "pay_date",
    "record_date",
    "float_date",
    "change_date",
    "suspend_date",
    "surv_date",
    "nav_date",
    "requested_date",
    "date",
    "datetime",
    "month",
    "m",
    "quarter",
    "period",
    "year_week",
    "week",
    "year",
    # A lone range endpoint is used as a natural point period by several
    # statement/disclosure endpoints (for example end_date=20231231).
    "end_date",
    "start_date",
)

_GENERATION_DATE_KEYS = ("as_of", "as_of_date")

# These interfaces are requested by accounting/report period, while the rows
# become knowable and are clipped by announcement date in the snapshot builder.
# A period before snapshot_start can therefore carry legitimate announcements
# into the release window. Keep such units as carry-in candidates and let the
# builder's ann_date/f_ann_date predicate perform the exact row-level clipping.
PIT_CARRY_IN_DATASETS = frozenset(
    {
        "income",
        "balancesheet",
        "cashflow",
        "fina_indicator",
        "fina_indicator_nondefault",
        "forecast",
        "express",
        "fina_audit",
        "fina_mainbz",
    }
)
_INTERVAL_CARRY_IN_DATASETS = frozenset({"index_member_all", "namechange"})


@dataclass(frozen=True, slots=True)
class ReleaseWindowSelection:
    snapshot_start: date | None
    snapshot_end: date
    profile: str | None
    requested_datasets: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    plan_scope_sha256: str
    selected_unit_set_sha256: str
    unit_identities: tuple[dict[str, Any], ...]

    @property
    def scope_sha256(self) -> str:
        """Backward-compatible alias for the explicitly named plan scope hash."""

        return self.plan_scope_sha256

    def report(self) -> dict[str, Any]:
        return {
            "selector_version": RELEASE_WINDOW_SELECTOR_VERSION,
            "snapshot_start": (
                self.snapshot_start.isoformat() if self.snapshot_start else None
            ),
            "snapshot_end": self.snapshot_end.isoformat(),
            "profile": self.profile,
            "requested_datasets": list(self.requested_datasets),
            "plan_scope_sha256": self.plan_scope_sha256,
            # Keep the old key in reports consumed by pre-selector tooling.
            "scope_sha256": self.plan_scope_sha256,
            "selected_unit_set_sha256": self.selected_unit_set_sha256,
            "selected_unit_count": len(self.rows),
            "selected_unit_identities": [dict(item) for item in self.unit_identities],
        }


def select_release_window_units(
    rows: Iterable[dict[str, Any]],
    *,
    snapshot_start: date | None,
    snapshot_end: date,
    datasets: set[str] | frozenset[str] | None = None,
    profile: str | None = None,
) -> ReleaseWindowSelection:
    """Select the exact active work-unit generation a snapshot would publish.

    Selection is performed on *all* non-superseded plan rows, not only successful
    rows. Therefore a pending current reference generation or adaptive child
    partition blocks publication, while pending units wholly outside the release
    window do not. The same returned rows must drive plan completeness, checksum
    verification and snapshot materialization.
    """

    if snapshot_start is not None and snapshot_end < snapshot_start:
        raise ValueError("release snapshot end must not be before its start")
    materialized = [dict(row) for row in rows]
    requested = tuple(
        sorted(
            str(value)
            for value in (
                datasets
                if datasets is not None
                else {row.get("dataset") for row in materialized if row.get("dataset")}
            )
        )
    )
    requested_set = set(requested)
    active: list[dict[str, Any]] = []
    for row in materialized:
        dataset = str(row.get("dataset") or "")
        if not dataset or (requested_set and dataset not in requested_set):
            continue
        if str(row.get("status") or "") == "superseded":
            continue
        if _intersects_release_window(
            row,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        ):
            active.append(row)

    selected = select_current_reference_units(active, snapshot_end=snapshot_end)
    selected = sorted(
        (dict(row) for row in selected),
        key=lambda row: (str(row["dataset"]), str(row["unit_key"])),
    )
    scope_payload = {
        "selector_version": RELEASE_WINDOW_SELECTOR_VERSION,
        "snapshot_start": snapshot_start.isoformat() if snapshot_start else None,
        "snapshot_end": snapshot_end.isoformat(),
        "profile": profile,
        "requested_datasets": list(requested),
    }
    plan_scope_sha256 = _canonical_sha256(scope_payload)
    identities = tuple(_unit_identity(row) for row in selected)
    return ReleaseWindowSelection(
        snapshot_start=snapshot_start,
        snapshot_end=snapshot_end,
        profile=profile,
        requested_datasets=requested,
        rows=tuple(selected),
        plan_scope_sha256=plan_scope_sha256,
        selected_unit_set_sha256=_canonical_sha256(list(identities)),
        unit_identities=identities,
    )


def summarize_release_plan(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return verification counters over selected release rows only."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        grouped.setdefault(str(row["dataset"]), []).append(row)
    result: list[dict[str, Any]] = []
    for dataset, members in sorted(grouped.items()):
        statuses = [str(row.get("status") or "") for row in members]
        successful = [
            row for row, status in zip(members, statuses, strict=True)
            if status == "succeeded"
        ]
        result.append(
            {
                "dataset": dataset,
                "planned": len(members),
                "succeeded": len(successful),
                "failed": statuses.count("failed"),
                "pending": statuses.count("pending"),
                "running": statuses.count("running"),
                "superseded": 0,
                "empty": sum(int(row.get("row_count") or 0) == 0 for row in successful),
                "allowed_empty": sum(
                    int(row.get("row_count") or 0) == 0 and bool(row.get("allow_empty"))
                    for row in successful
                ),
                "unexpected_empty": sum(
                    int(row.get("row_count") or 0) == 0 and not bool(row.get("allow_empty"))
                    for row in successful
                ),
                "rows": sum(int(row.get("row_count") or 0) for row in successful),
            }
        )
    return result


def _intersects_release_window(
    row: dict[str, Any], *, snapshot_start: date | None, snapshot_end: date
) -> bool:
    scope = dict(row.get("scope_json") or {})
    params = dict(row.get("params_json") or {})
    values = {**params, **scope}
    dataset = str(row.get("dataset") or "")

    # As-of values identify a reference generation, not the provider rows'
    # natural time axis. An older master can still be required by a newer
    # snapshot, but a generation from after the frozen end date cannot enter it.
    for key in _GENERATION_DATE_KEYS:
        if key not in values:
            continue
        bounds = _date_bounds(values.get(key))
        if bounds is not None and bounds[0] > snapshot_end:
            return False

    bounds = _request_bounds(dataset, values)
    if bounds is None:
        # Reference/master requests without a natural request interval may
        # contain rows relevant to any historical window. Include conservatively.
        return True
    lower, upper = bounds
    if dataset in _INTERVAL_CARRY_IN_DATASETS:
        return lower <= snapshot_end
    if dataset in PIT_CARRY_IN_DATASETS:
        return lower <= snapshot_end
    window_lower = snapshot_start or date.min
    return lower <= snapshot_end and upper >= window_lower


def _request_bounds(dataset: str, values: dict[str, Any]) -> tuple[date, date] | None:
    if values.get("partition_axis") in {"date", "datetime"}:
        bounds = _paired_bounds(values, "partition_start", "partition_end")
        if bounds is not None:
            return bounds
    for start_key, end_key in _RANGE_KEYS:
        bounds = _paired_bounds(values, start_key, end_key)
        if bounds is not None:
            return bounds

    definition = ALL_DEFINITIONS.get(dataset)
    definition_key = str(definition.date_field or "") if definition else ""
    keys = (
        (definition_key,) if definition_key and definition_key not in _POINT_DATE_KEYS else ()
    ) + _POINT_DATE_KEYS
    for key in keys:
        if not key or key not in values:
            continue
        bounds = _date_bounds(values.get(key))
        if bounds is not None:
            return bounds
    return None


def _paired_bounds(
    values: dict[str, Any], start_key: str, end_key: str
) -> tuple[date, date] | None:
    if start_key not in values or end_key not in values:
        return None
    start = _date_bounds(values.get(start_key))
    end = _date_bounds(values.get(end_key))
    if start is None or end is None:
        return None
    lower, upper = start[0], end[1]
    return (lower, upper) if lower <= upper else None


def _date_bounds(value: Any) -> tuple[date, date] | None:
    if isinstance(value, datetime):
        point = value.date()
        return point, point
    if isinstance(value, date):
        return value, value
    text = str(value or "").strip()
    if not text:
        return None

    quarter = re.fullmatch(r"(\d{4})[- ]?Q([1-4])", text, flags=re.IGNORECASE)
    if quarter:
        year, number = int(quarter.group(1)), int(quarter.group(2))
        month = (number - 1) * 3 + 1
        start = date(year, month, 1)
        end_month = month + 2
        return start, date(year, end_month, monthrange(year, end_month)[1])
    week = re.fullmatch(r"(\d{4})-?W(\d{1,2})", text, flags=re.IGNORECASE)
    if week:
        try:
            start = date.fromisocalendar(int(week.group(1)), int(week.group(2)), 1)
        except ValueError:
            return None
        return start, start + timedelta(days=6)
    month = re.fullmatch(r"(\d{4})[-/]?(\d{2})", text)
    if month:
        year, number = int(month.group(1)), int(month.group(2))
        if not 1 <= number <= 12:
            return None
        return date(year, number, 1), date(year, number, monthrange(year, number)[1])
    if re.fullmatch(r"\d{4}", text):
        year = int(text)
        return date(year, 1, 1), date(year, 12, 31)

    compact = re.sub(r"[^0-9]", "", text)
    if len(compact) >= 8:
        try:
            point = datetime.strptime(compact[:8], "%Y%m%d").date()
        except ValueError:
            point = None
        if point is not None:
            return point, point
    try:
        point = date.fromisoformat(text[:10])
    except ValueError:
        return None
    return point, point


def _unit_identity(row: dict[str, Any]) -> dict[str, Any]:
    scope = dict(row.get("scope_json") or {})
    return {
        "dataset": str(row["dataset"]),
        "unit_key": str(row["unit_key"]),
        "scope_sha256": _canonical_sha256(scope),
        "status": str(row.get("status") or ""),
        # This is the immutable file digest checked immediately before the
        # quality gate is issued and compared again by snapshot publication.
        "sha256": str(row.get("sha256") or "") or None,
        "row_count": (
            int(row["row_count"]) if row.get("row_count") is not None else None
        ),
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
