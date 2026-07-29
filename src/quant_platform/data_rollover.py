from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from quant_data.snapshot_lineage import assert_snapshot_descendant

from .services import list_qlib_datasets, list_snapshots, resolve_snapshot_dataset

ROLL_POLICIES = {"pinned", "latest_compatible"}


def _qlib_calendar(dataset: dict[str, Any]) -> list[date]:
    calendar_path = Path(str(dataset["path"])) / "calendars" / "day.txt"
    try:
        calendar = sorted(
            {
                date.fromisoformat(value)
                for value in calendar_path.read_text(encoding="utf-8").splitlines()
                if value.strip()
            }
        )
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("Qlib dataset calendar is unavailable or invalid") from exc
    if not calendar:
        raise ValueError("Qlib dataset calendar is empty")
    return calendar


def qlib_trading_date_on_or_before(
    dataset: dict[str, Any],
    as_of_date: date,
) -> date:
    """Return the latest persisted trading day no later than ``as_of_date``."""

    eligible = [value for value in _qlib_calendar(dataset) if value <= as_of_date]
    if not eligible:
        raise ValueError("Qlib dataset has no trading day on or before the requested date")
    return eligible[-1]


def next_qlib_trading_date(dataset: dict[str, Any], signal_date: date) -> date:
    calendar = _qlib_calendar(dataset)
    next_dates = [value for value in calendar if value > signal_date]
    if not next_dates:
        raise ValueError("Qlib dataset has no later trading day")
    return next_dates[0]


def select_qlib_dataset(
    data_root: Path,
    *,
    anchor_name: str,
    roll_policy: str,
    lineage_id: str | None,
    required_date: date,
    require_later_date: bool = False,
) -> dict[str, Any]:
    if roll_policy not in ROLL_POLICIES:
        raise ValueError("unsupported dataset roll policy")
    datasets = {item["name"]: item for item in list_qlib_datasets(data_root)}
    anchor = datasets.get(anchor_name)
    if not anchor or not anchor["ready"] or not anchor.get("reproducible"):
        raise ValueError("anchor Qlib dataset is not ready and reproducible")
    if roll_policy == "pinned":
        _require_dataset_date(anchor, required_date, require_later_date=require_later_date)
        return anchor
    if (
        not lineage_id
        or not anchor.get("lineage_verified")
        or anchor.get("lineage_id") != lineage_id
    ):
        raise ValueError("latest-compatible Qlib policy requires a verified anchor lineage")

    valid: list[dict[str, Any]] = []
    anchor_manifest = _source_manifest(data_root, anchor)
    for candidate in datasets.values():
        if (
            not candidate["ready"]
            or not candidate.get("reproducible")
            or not candidate.get("lineage_verified")
            or candidate.get("lineage_id") != lineage_id
        ):
            continue
        try:
            _require_dataset_date(
                candidate,
                required_date,
                require_later_date=require_later_date,
            )
            assert_snapshot_descendant(
                anchor_manifest=anchor_manifest,
                candidate_manifest=_source_manifest(data_root, candidate),
            )
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            continue
        valid.append(candidate)
    if not valid:
        raise ValueError("no verified latest-compatible Qlib descendant covers the requested date")
    return max(valid, key=lambda item: (str(item.get("end_date") or ""), str(item["name"])))


def select_execution_snapshot(
    data_root: Path,
    *,
    anchor_name: str,
    roll_policy: str,
    lineage_id: str | None,
    required_date: date,
    minute_dataset: str,
    shortability_dataset: str,
) -> dict[str, Any]:
    if roll_policy not in ROLL_POLICIES:
        raise ValueError("unsupported execution roll policy")
    snapshots = {item.get("name"): item for item in list_snapshots(data_root)}
    anchor = snapshots.get(anchor_name)
    if not isinstance(anchor, dict):
        raise ValueError("anchor execution snapshot is unavailable")
    if roll_policy == "pinned":
        selected = anchor
    else:
        if not lineage_id or anchor.get("lineage_id") != lineage_id:
            raise ValueError(
                "latest-compatible execution policy requires a verified anchor lineage"
            )
        selected = _latest_execution_descendant(
            anchor,
            snapshots.values(),
            lineage_id=lineage_id,
            required_date=required_date,
        )
    try:
        end_date = date.fromisoformat(str(selected["end_date"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("execution snapshot has no valid end date") from exc
    if end_date < required_date:
        raise ValueError("execution snapshot does not cover the required trade date")
    name = str(selected["name"])
    minute = resolve_snapshot_dataset(
        data_root,
        snapshot_name=name,
        dataset_name=minute_dataset,
    )
    shortability = resolve_snapshot_dataset(
        data_root,
        snapshot_name=name,
        dataset_name=shortability_dataset,
    )
    return {"snapshot": selected, "minute": minute, "shortability": shortability}


def _latest_execution_descendant(
    anchor: dict[str, Any],
    candidates: Any,
    *,
    lineage_id: str,
    required_date: date,
) -> dict[str, Any]:
    valid: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("lineage_id") != lineage_id:
            continue
        try:
            candidate_end = date.fromisoformat(str(candidate["end_date"]))
            if candidate_end < required_date:
                continue
            assert_snapshot_descendant(
                anchor_manifest=anchor,
                candidate_manifest=candidate,
            )
        except (KeyError, ValueError):
            continue
        valid.append(candidate)
    if not valid:
        raise ValueError("no verified execution-snapshot descendant covers the trade date")
    return max(valid, key=lambda item: (str(item.get("end_date") or ""), str(item["name"])))


def _source_manifest(data_root: Path, dataset: dict[str, Any]) -> dict[str, Any]:
    provenance = dataset.get("provenance") or {}
    snapshot_name = str(provenance.get("snapshot_name") or "")
    if not snapshot_name:
        raise ValueError("Qlib dataset has no source snapshot")
    return json.loads(
        (data_root / "snapshots" / snapshot_name / "manifest.json").read_text(encoding="utf-8")
    )


def _require_dataset_date(
    dataset: dict[str, Any],
    required_date: date,
    *,
    require_later_date: bool,
) -> None:
    try:
        start = date.fromisoformat(str(dataset["start_date"]))
        end = date.fromisoformat(str(dataset["end_date"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("Qlib dataset has no valid date coverage") from exc
    if required_date < start or required_date > end:
        raise ValueError("Qlib dataset does not cover the requested signal date")
    if require_later_date and required_date >= end:
        raise ValueError("Qlib dataset has no later trading-day coverage")
