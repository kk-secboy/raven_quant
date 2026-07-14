from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from .catalog import ALL_DEFINITIONS
from .checkpoint import CheckpointStore


def verify_downloads(checkpoint: CheckpointStore, data_root: Path) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for row in checkpoint.verification_rows():
        item = dict(row)
        if item["succeeded"] != item["planned"]:
            errors.append(
                f"{item['dataset']}: {item['succeeded']}/{item['planned']} units succeeded"
            )
        definition = ALL_DEFINITIONS.get(item["dataset"])
        if item["empty"] and definition and not definition.allow_empty:
            errors.append(f"{item['dataset']}: {item['empty']} unexpected empty units")
        elif item["empty"]:
            warnings.append(f"{item['dataset']}: {item['empty']} allowed empty units")
        datasets.append(item)

    missing_files = 0
    bad_checksums = 0
    for row in checkpoint.successful():
        path = data_root / row["output_path"]
        if not path.exists():
            missing_files += 1
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            bad_checksums += 1
    if missing_files:
        errors.append(f"{missing_files} successful unit files are missing")
    if bad_checksums:
        errors.append(f"{bad_checksums} successful unit files failed checksum validation")

    duplicate_checks: dict[str, int] = {}
    connection = duckdb.connect()
    try:
        for dataset, definition in ALL_DEFINITIONS.items():
            if not definition.primary_key:
                continue
            unit_dir = data_root / "units" / dataset
            if not unit_dir.exists() or not any(unit_dir.glob("*.parquet")):
                continue
            glob = str((unit_dir / "*.parquet").resolve()).replace("'", "''")
            key = ",".join(f'"{column}"' for column in definition.primary_key)
            duplicates = connection.execute(
                f"""
                SELECT count(*) - count(DISTINCT ({key}))
                FROM read_parquet('{glob}', union_by_name=true)
                """
            ).fetchone()[0]
            duplicate_checks[dataset] = int(duplicates)
            if duplicates:
                errors.append(f"{dataset}: {duplicates} duplicate primary-key rows")
    finally:
        connection.close()

    return {
        "checked_at": datetime.now(UTC).isoformat(),
        "ok": not errors,
        "datasets": datasets,
        "duplicate_checks": duplicate_checks,
        "errors": errors,
        "warnings": warnings,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
