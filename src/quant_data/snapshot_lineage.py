from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_lineage_id(kind: str, configuration: dict[str, Any]) -> str:
    return canonical_sha256({"kind": kind, "configuration": configuration})


def file_contract_sha256(files: dict[str, Path]) -> str:
    return canonical_sha256(
        {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(files.items())
        }
    )


def prepare_lineage_metadata(
    snapshots_root: Path,
    *,
    lineage_id: str,
    end_date: date,
    successful_units: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    parent = latest_compatible_snapshot(
        snapshots_root,
        lineage_id=lineage_id,
        end_date=end_date,
    )
    if parent is None:
        return {
            "lineage_id": lineage_id,
            "parent_snapshot": None,
            "parent_manifest_sha256": None,
            "lineage_generation": 0,
        }
    parent_name, manifest, manifest_sha256 = parent
    _assert_append_only(manifest, successful_units)
    return {
        "lineage_id": lineage_id,
        "parent_snapshot": parent_name,
        "parent_manifest_sha256": manifest_sha256,
        "lineage_generation": int(manifest.get("lineage_generation") or 0) + 1,
    }


def latest_compatible_snapshot(
    snapshots_root: Path,
    *,
    lineage_id: str,
    end_date: date | None = None,
) -> tuple[str, dict[str, Any], str] | None:
    candidates: list[tuple[date, str, dict[str, Any], str]] = []
    if not snapshots_root.exists():
        return None
    for path in snapshots_root.iterdir():
        if not path.is_dir() or path.name.startswith("."):
            continue
        manifest_path = path / "manifest.json"
        try:
            raw = manifest_path.read_bytes()
            manifest = json.loads(raw)
            candidate_end = date.fromisoformat(str(manifest["end_date"]))
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
            continue
        if manifest.get("lineage_id") != lineage_id:
            continue
        if end_date is not None and candidate_end > end_date:
            continue
        candidates.append(
            (
                candidate_end,
                path.name,
                manifest,
                hashlib.sha256(raw).hexdigest(),
            )
        )
    if not candidates:
        return None
    _, name, manifest, digest = max(candidates, key=lambda item: (item[0], item[1]))
    return name, manifest, digest


def assert_snapshot_descendant(
    *,
    anchor_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
) -> None:
    lineage_id = str(anchor_manifest.get("lineage_id") or "")
    if not lineage_id or candidate_manifest.get("lineage_id") != lineage_id:
        raise ValueError("candidate snapshot is not in the anchor lineage")
    _assert_manifest_units_subset(anchor_manifest, candidate_manifest)
    anchor_end = date.fromisoformat(str(anchor_manifest["end_date"]))
    candidate_end = date.fromisoformat(str(candidate_manifest["end_date"]))
    if candidate_end < anchor_end:
        raise ValueError("candidate snapshot ends before the anchor snapshot")


def _assert_append_only(
    parent_manifest: dict[str, Any],
    successful_units: dict[str, list[dict[str, Any]]],
) -> None:
    candidate = {
        "datasets": {
            dataset: {"source_units": _unit_identities(rows)}
            for dataset, rows in successful_units.items()
        }
    }
    _assert_manifest_units_subset(parent_manifest, candidate)


def _assert_manifest_units_subset(
    ancestor: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    ancestor_datasets = ancestor.get("datasets")
    candidate_datasets = candidate.get("datasets")
    if not isinstance(ancestor_datasets, dict) or not isinstance(candidate_datasets, dict):
        raise ValueError("snapshot lineage requires dataset manifests")
    if set(ancestor_datasets) != set(candidate_datasets):
        raise ValueError("snapshot lineage dataset set changed")
    for dataset, entry in ancestor_datasets.items():
        source_units = entry.get("source_units") if isinstance(entry, dict) else None
        candidate_entry = candidate_datasets.get(dataset)
        candidate_units = (
            candidate_entry.get("source_units") if isinstance(candidate_entry, dict) else None
        )
        if not isinstance(source_units, list) or not isinstance(candidate_units, list):
            raise ValueError("snapshot lineage requires source-unit identities")
        old = {_unit_tuple(item) for item in source_units}
        new = {_unit_tuple(item) for item in candidate_units}
        if not old.issubset(new):
            raise ValueError(f"snapshot lineage rewrote or removed source units for {dataset}")


def _unit_identities(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "unit_key": str(row["unit_key"]),
            "sha256": str(row.get("sha256") or ""),
            "row_count": int(row.get("row_count") or 0),
        }
        for row in sorted(rows, key=lambda item: str(item["unit_key"]))
    ]


def _unit_tuple(item: Any) -> tuple[str, str, int]:
    if not isinstance(item, dict):
        raise ValueError("snapshot source-unit identity is invalid")
    return (
        str(item.get("unit_key") or ""),
        str(item.get("sha256") or ""),
        int(item.get("row_count") or 0),
    )
