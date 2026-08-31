from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from .reference_data import reference_manifest_metadata


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


def verify_snapshot_lineage(snapshot_path: Path) -> dict[str, Any]:
    """Verify the declared lineage contract and every immediate parent link."""

    return _verify_snapshot_lineage(snapshot_path.resolve(), visited=set())


def _verify_snapshot_lineage(
    snapshot_path: Path, *, visited: set[Path]
) -> dict[str, Any]:
    if snapshot_path in visited:
        raise ValueError("snapshot lineage contains a parent cycle")
    visited.add(snapshot_path)
    manifest_path = snapshot_path / "manifest.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"snapshot lineage manifest is missing or invalid: {snapshot_path}"
        ) from exc
    contract = manifest.get("lineage_contract")
    if not isinstance(contract, dict) or set(contract) != {"kind", "configuration"}:
        raise ValueError("snapshot lineage contract is missing or invalid")
    kind = contract.get("kind")
    configuration = contract.get("configuration")
    if not isinstance(kind, str) or not kind or not isinstance(configuration, dict):
        raise ValueError("snapshot lineage contract is missing or invalid")
    expected_lineage_id = make_lineage_id(kind, configuration)
    if manifest.get("lineage_id") != expected_lineage_id:
        raise ValueError("snapshot lineage id does not match its immutable contract")
    generation = manifest.get("lineage_generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("snapshot lineage generation is invalid")
    parent_name = manifest.get("parent_snapshot")
    parent_digest = manifest.get("parent_manifest_sha256")
    if generation == 0:
        if parent_name is not None or parent_digest is not None:
            raise ValueError("root snapshot lineage must not declare a parent")
        return manifest
    if not isinstance(parent_name, str) or not parent_name or not isinstance(parent_digest, str):
        raise ValueError("snapshot lineage parent evidence is incomplete")
    if Path(parent_name).name != parent_name:
        raise ValueError("snapshot lineage parent name is unsafe")
    snapshots_root = snapshot_path.parent.resolve()
    parent_path = (snapshots_root / parent_name).resolve()
    if parent_path.parent != snapshots_root or parent_path == snapshot_path:
        raise ValueError("snapshot lineage parent path is unsafe")
    parent_manifest_path = parent_path / "manifest.json"
    try:
        parent_raw = parent_manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError("snapshot lineage parent manifest is missing") from exc
    if hashlib.sha256(parent_raw).hexdigest() != parent_digest:
        raise ValueError("snapshot lineage parent manifest hash does not match")
    parent_manifest = _verify_snapshot_lineage(parent_path, visited=visited)
    if int(parent_manifest.get("lineage_generation", -1)) + 1 != generation:
        raise ValueError("snapshot lineage generation does not follow its parent")
    assert_snapshot_descendant(
        anchor_manifest=parent_manifest,
        candidate_manifest=manifest,
    )
    return manifest


def file_contract_sha256(files: dict[str, Path]) -> str:
    return canonical_sha256(
        {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(files.items())
        }
    )


def resolve_verified_snapshot_anchor(
    snapshots_root: Path,
    name: str,
    *,
    required_profile: str,
    required_start: date,
    maximum_end: date,
    required_dataset: str,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a separately governed cross-lineage repair input.

    An anchor is deliberately not a lineage parent. It may bridge an ingestion
    contract change, but only after its own lineage, quality gate, and every
    consumed dataset file have been verified.
    """

    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name)
        or Path(name).name != name
        or name in {".", ".."}
    ):
        raise ValueError("industry history anchor must be one safe snapshot name")
    root = snapshots_root.resolve()
    unresolved = root / name
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise ValueError("industry history anchor snapshot is missing or unsafe")
    try:
        anchor = unresolved.resolve(strict=True)
    except OSError as exc:
        raise ValueError("industry history anchor snapshot is missing") from exc
    if anchor.parent != root or anchor.name != name or anchor.is_symlink():
        raise ValueError("industry history anchor escapes the snapshot root")

    manifest_path = anchor / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("industry history anchor manifest is missing or unsafe")
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("industry history anchor manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise ValueError("industry history anchor manifest is invalid")
    verified_manifest = verify_snapshot_lineage(anchor)
    if verified_manifest != manifest:
        raise ValueError("industry history anchor changed during verification")
    if manifest.get("name") != name:
        raise ValueError("industry history anchor manifest name does not match")
    if manifest.get("profile") != required_profile:
        raise ValueError("industry history anchor profile does not match")
    if manifest.get("start_date") != required_start.isoformat():
        raise ValueError("industry history anchor start date does not match")
    try:
        anchor_end = date.fromisoformat(str(manifest["end_date"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("industry history anchor end date is invalid") from exc
    if anchor_end > maximum_end:
        raise ValueError("industry history anchor ends after the target snapshot")
    quality_gate = manifest.get("quality_gate")
    if not isinstance(quality_gate, dict) or quality_gate.get("ok") is not True:
        raise ValueError("industry history anchor has no passing quality gate")
    datasets = manifest.get("datasets")
    entry = datasets.get(required_dataset) if isinstance(datasets, dict) else None
    if not isinstance(entry, dict) or int(entry.get("rows") or 0) <= 0:
        raise ValueError(
            f"industry history anchor has no usable {required_dataset} dataset"
        )
    source_sha256 = str(entry.get("source_sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("industry history anchor dataset source hash is invalid")
    files = entry.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("industry history anchor dataset file manifest is missing")
    file_evidence: list[dict[str, Any]] = []
    declared_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("industry history anchor dataset file entry is invalid")
        relative = Path(str(item.get("path") or ""))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or tuple(relative.parts[:2]) != ("parquet", required_dataset)
            or relative.suffix != ".parquet"
        ):
            raise ValueError("industry history anchor contains an unsafe dataset file")
        relative_posix = relative.as_posix()
        if relative_posix in declared_paths:
            raise ValueError("industry history anchor repeats a dataset file path")
        declared_paths.add(relative_posix)
        target = anchor / relative
        if _path_contains_symlink(anchor, relative) or not target.is_file():
            raise ValueError("industry history anchor dataset file is missing or unsafe")
        try:
            resolved = target.resolve(strict=True)
            resolved.relative_to(anchor)
        except (OSError, ValueError) as exc:
            raise ValueError("industry history anchor dataset file escapes snapshot") from exc
        expected_size_value = item.get("bytes")
        if (
            isinstance(expected_size_value, bool)
            or not isinstance(expected_size_value, int)
            or expected_size_value < 0
        ):
            raise ValueError("industry history anchor dataset file size is invalid")
        expected_size = expected_size_value
        expected_sha256 = str(item.get("sha256") or "").lower()
        if (
            resolved.stat().st_size != expected_size
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or _sha256_file(resolved) != expected_sha256
        ):
            raise ValueError("industry history anchor dataset file hash does not match")
        file_evidence.append(
            {
                "path": relative_posix,
                "bytes": expected_size,
                "sha256": expected_sha256,
            }
        )
    dataset_root = anchor / "parquet" / required_dataset
    if _path_contains_symlink(anchor, Path("parquet") / required_dataset):
        raise ValueError("industry history anchor dataset directory is unsafe")
    actual_paths: set[str] = set()
    if dataset_root.is_dir():
        for path in dataset_root.rglob("*.parquet"):
            relative = path.relative_to(anchor)
            if _path_contains_symlink(anchor, relative) or not path.is_file():
                raise ValueError("industry history anchor dataset file is unsafe")
            actual_paths.add(relative.as_posix())
    if actual_paths != declared_paths:
        raise ValueError("industry history anchor contains unmanifested dataset files")
    evidence = {
        "contract_version": "industry-history-anchor-v1",
        "snapshot_name": name,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "lineage_id": str(manifest.get("lineage_id") or ""),
        "profile": required_profile,
        "start_date": required_start.isoformat(),
        "end_date": anchor_end.isoformat(),
        "dataset": required_dataset,
        "dataset_source_sha256": source_sha256,
        "dataset_files_sha256": canonical_sha256(file_evidence),
        "carry_rule_version": "index-member-delisted-parent-carry-v1",
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return anchor, evidence


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_contains_symlink(root: Path, relative: Path) -> bool:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


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
        try:
            verify_snapshot_lineage(path)
        except ValueError:
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
            dataset: {
                "source_units": _unit_identities(rows),
                "reference_refresh": reference_manifest_metadata(rows),
            }
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
    missing_datasets = set(ancestor_datasets) - set(candidate_datasets)
    if missing_datasets:
        raise ValueError(
            "snapshot lineage removed datasets: " + ", ".join(sorted(missing_datasets))
        )
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
        old_refresh = _reference_refresh_buckets(entry)
        new_refresh = _reference_refresh_buckets(candidate_entry)
        if old_refresh or new_refresh:
            if not old_refresh or not new_refresh:
                raise ValueError(
                    f"snapshot lineage lost reference-refresh evidence for {dataset}"
                )
            if min(new_refresh) < min(old_refresh) or max(new_refresh) < max(old_refresh):
                raise ValueError(
                    f"snapshot lineage moved reference generation backwards for {dataset}"
                )
            if max(new_refresh) > max(old_refresh):
                continue
        if not old.issubset(new):
            raise ValueError(f"snapshot lineage rewrote or removed source units for {dataset}")


def _reference_refresh_buckets(entry: Any) -> tuple[str, ...]:
    if not isinstance(entry, dict):
        return ()
    metadata = entry.get("reference_refresh")
    if not isinstance(metadata, dict):
        return ()
    buckets = metadata.get("selected_buckets")
    if not isinstance(buckets, list) or not buckets:
        return ()
    normalized = tuple(sorted(str(value) for value in buckets if value))
    if len(normalized) != len(buckets):
        raise ValueError("snapshot reference-refresh buckets are invalid")
    return normalized


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
