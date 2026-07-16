from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from .models import FetchSpec

PARTITION_AXIS = "partition_axis"
PARTITION_START = "partition_start"
PARTITION_END = "partition_end"


def partition_metadata(
    axis: str,
    start: date | datetime,
    end: date | datetime,
    *,
    start_param: str = "start_date",
    end_param: str = "end_date",
    value_format: str | None = None,
    values: list[str] | None = None,
) -> dict[str, Any]:
    """Return stable scope metadata for a bisectable provider request."""

    if axis not in {"date", "datetime"}:
        raise ValueError("partition axis must be date or datetime")
    if end < start:
        raise ValueError("partition end must not be before start")
    if value_format is None:
        value_format = "compact" if axis == "date" else "timestamp"
    metadata: dict[str, Any] = {
        PARTITION_AXIS: axis,
        PARTITION_START: _scope_value(start),
        PARTITION_END: _scope_value(end),
        "partition_start_param": start_param,
        "partition_end_param": end_param,
        "partition_value_format": value_format,
    }
    if values is not None:
        metadata["partition_values"] = list(values)
    return metadata


def split_partition_spec(spec: FetchSpec) -> list[FetchSpec]:
    """Bisect one date/datetime partition without overlap or gaps."""

    axis, start, end = partition_bounds(spec)
    if start == end:
        grain = "single-day" if axis == "date" else "single-second"
        raise RuntimeError(
            f"{grain} {spec.dataset} partition {_scope_value(start)} still reached "
            "the provider limit"
        )
    if axis == "date":
        assert isinstance(start, date) and not isinstance(start, datetime)
        assert isinstance(end, date) and not isinstance(end, datetime)
        left_end = start + timedelta(days=(end - start).days // 2)
        right_start = left_end + timedelta(days=1)
    else:
        assert isinstance(start, datetime) and isinstance(end, datetime)
        seconds = int((end - start).total_seconds())
        left_end = start + timedelta(seconds=seconds // 2)
        right_start = left_end + timedelta(seconds=1)
    return [
        resize_partition_spec(spec, start, left_end, parent_partition=spec.unit_key),
        resize_partition_spec(spec, right_start, end, parent_partition=spec.unit_key),
    ]


def resize_partition_spec(
    spec: FetchSpec,
    start: date | datetime,
    end: date | datetime,
    *,
    parent_partition: str | None = None,
) -> FetchSpec:
    """Clone a partition with new bounds and a reset pagination cursor."""

    axis = str(spec.scope.get(PARTITION_AXIS) or "")
    expected_type = datetime if axis == "datetime" else date
    if not axis or not isinstance(start, expected_type) or not isinstance(end, expected_type):
        raise ValueError("partition bounds do not match the parent axis")
    if axis == "date" and (isinstance(start, datetime) or isinstance(end, datetime)):
        raise ValueError("date partition bounds must not include time")
    if end < start:
        raise ValueError("partition end must not be before start")

    start_param = str(spec.scope.get("partition_start_param") or "start_date")
    end_param = str(spec.scope.get("partition_end_param") or "end_date")
    value_format = str(spec.scope.get("partition_value_format") or "compact")
    params = dict(spec.params)
    params[start_param] = _parameter_value(start, value_format, is_end=False)
    params[end_param] = _parameter_value(end, value_format, is_end=True)

    scope = dict(spec.scope)
    scope[PARTITION_START] = _scope_value(start)
    scope[PARTITION_END] = _scope_value(end)
    if parent_partition:
        scope["parent_partition"] = parent_partition
        scope["supersedes_partition"] = parent_partition
    values = scope.get("partition_values")
    if isinstance(values, list):
        scope["partition_values"] = [
            value for value in values if start <= _parse_partition_value(axis, value) <= end
        ]

    parent_group = scope.get("page_group")
    if parent_group:
        root = str(scope.get("partition_group_root") or parent_group)
        scope["partition_group_root"] = root
        scope["page_group"] = (
            f"{root}:split:{_group_value(start)}:{_group_value(end)}"
        )
        scope["supersedes_page_group"] = str(parent_group)
        scope["offset"] = 0
        scope.pop("page_index", None)
        scope.pop("offset_origin", None)
        params["offset"] = 0
        if "page_size" in scope:
            params["limit"] = int(scope["page_size"])

    expected_field = scope.get("expected_date_field")
    if expected_field:
        scope.pop("expected_date", None)
        scope["expected_date_start"] = _compact_date(start)
        scope["expected_date_end"] = _compact_date(end)

    return FetchSpec(
        dataset=spec.dataset,
        api_name=spec.api_name,
        scope=scope,
        params=params,
        fields=spec.fields,
        allow_empty=spec.allow_empty,
        max_attempts=spec.max_attempts,
    )


def partition_bounds(spec: FetchSpec) -> tuple[str, date | datetime, date | datetime]:
    axis = str(spec.scope.get(PARTITION_AXIS) or "")
    if axis not in {"date", "datetime"}:
        raise ValueError(f"{spec.dataset} is not an adaptive partition")
    return (
        axis,
        _parse_partition_value(axis, spec.scope.get(PARTITION_START)),
        _parse_partition_value(axis, spec.scope.get(PARTITION_END)),
    )


def is_adaptive_partition(spec: FetchSpec) -> bool:
    return spec.scope.get(PARTITION_AXIS) in {"date", "datetime"}


def is_partition_overflow_error(message: str) -> bool:
    normalized = message.lower().replace("-", " ")
    return any(
        marker in normalized
        for marker in (
            "may be truncated",
            "offset cap",
            "code=50101",
            "pagination did not reach",
            "provider limit",
        )
    )


def _parse_partition_value(axis: str, value: Any) -> date | datetime:
    text = str(value or "").strip()
    if axis == "datetime":
        return datetime.fromisoformat(text)
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").date()
    return date.fromisoformat(text[:10])


def _parameter_value(
    value: date | datetime, value_format: str, *, is_end: bool
) -> str:
    if value_format == "compact":
        return _compact_date(value)
    if value_format == "iso_date":
        return value.strftime("%Y-%m-%d")
    if value_format == "date_timestamp":
        if isinstance(value, datetime):
            raise ValueError("date_timestamp partitions require date bounds")
        suffix = "23:59:59" if is_end else "00:00:00"
        return f"{value.isoformat()} {suffix}"
    if value_format == "timestamp":
        if not isinstance(value, datetime):
            raise ValueError("timestamp partitions require datetime bounds")
        return value.strftime("%Y-%m-%d %H:%M:%S")
    raise ValueError(f"unsupported partition value format: {value_format}")


def _scope_value(value: date | datetime) -> str:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    return value.isoformat()


def _group_value(value: date | datetime) -> str:
    return _scope_value(value).replace(" ", "T").replace(":", "").replace("-", "")


def _compact_date(value: date | datetime) -> str:
    return value.strftime("%Y%m%d")
