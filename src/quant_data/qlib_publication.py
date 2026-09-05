"""Bind data-pipeline consumers to the exact sealed Qlib publication."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .execution_contract import require_daily_qlib_contract
from .qlib_builder import verify_qlib_output_manifest
from .snapshot_lineage import canonical_sha256

QLIB_PUBLICATION_CONTRACT_VERSION = "qlib-publication-v1"


def _named_directory(root: Path, name: str) -> Path:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("Qlib publication directory name is invalid")
    resolved = (root / name).resolve(strict=True)
    if resolved.parent != root.resolve() or not resolved.is_dir():
        raise ValueError("Qlib publication directory is outside its governed root")
    return resolved


def _read_object(path: Path) -> tuple[dict[str, Any], str]:
    content = path.read_bytes()
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Qlib publication metadata must be a JSON object")
    return value, hashlib.sha256(content).hexdigest()


def build_qlib_publication_receipt(
    data_root: Path, requested_snapshot_name: str, dataset_path: Path
) -> dict[str, Any]:
    """Verify an explicit output and bind it to the requested immutable snapshot.

    A recovery output must prove the exact source manifest through its audited
    recovery receipt. No directory discovery or suffix-based selection is used.
    """

    requested = _named_directory(data_root / "snapshots", requested_snapshot_name)
    _, requested_sha256 = _read_object(requested / "manifest.json")
    output = dataset_path.resolve(strict=True)
    if output.parent != (data_root / "qlib").resolve() or not output.is_dir():
        raise ValueError("Qlib publication output is outside its governed root")
    provenance, provenance_sha256 = _read_object(output / "metadata" / "provenance.json")
    require_daily_qlib_contract(provenance)
    snapshot_name = str(provenance.get("snapshot_name") or "")
    snapshot = _named_directory(data_root / "snapshots", snapshot_name)
    manifest, snapshot_sha256 = _read_object(snapshot / "manifest.json")
    if (
        output.name != snapshot_name
        or provenance.get("snapshot_manifest_sha256") != snapshot_sha256
        or provenance.get("source_lineage_id") != manifest.get("lineage_id")
    ):
        raise ValueError("Qlib publication snapshot identity does not match its provenance")
    if snapshot != requested:
        recovery = manifest.get("ingested_at_recovery")
        lineage_contract = manifest.get("lineage_contract")
        if (
            not isinstance(recovery, dict)
            or not isinstance(lineage_contract, dict)
            or recovery.get("source_snapshot") != requested_snapshot_name
            or recovery.get("source_snapshot_manifest_sha256") != requested_sha256
            or recovery.get("receipt_sha256") != canonical_sha256(
                {key: value for key, value in recovery.items() if key != "receipt_sha256"}
            )
            or lineage_contract.get("kind")
            != "qlib_daily_source_ingested_at_successor"
        ):
            raise ValueError("Qlib publication is not a proven recovery of the requested snapshot")
    for key in ("dataset_identity_sha256", "dataset_lineage_id", "source_lineage_id"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(key) or "")):
            raise ValueError(f"Qlib publication {key} is missing or invalid")
    verify_qlib_output_manifest(output, provenance)
    return {
        "contract_version": QLIB_PUBLICATION_CONTRACT_VERSION,
        "requested_snapshot_name": requested_snapshot_name,
        "requested_snapshot_manifest_sha256": requested_sha256,
        "dataset": output.name,
        "dataset_path": str(output),
        "snapshot_name": snapshot_name,
        "snapshot_manifest_sha256": snapshot_sha256,
        "dataset_identity_sha256": provenance["dataset_identity_sha256"],
        "dataset_lineage_id": provenance["dataset_lineage_id"],
        "provenance_sha256": provenance_sha256,
    }


def validate_qlib_publication_receipt(
    data_root: Path, requested_snapshot_name: str, receipt: Any
) -> dict[str, Any]:
    """Revalidate the publisher receipt before a successor can consume it."""

    if not isinstance(receipt, dict) or (
        receipt.get("contract_version") != QLIB_PUBLICATION_CONTRACT_VERSION
        or receipt.get("requested_snapshot_name") != requested_snapshot_name
        or not isinstance(receipt.get("dataset_path"), str)
        or not receipt["dataset_path"]
    ):
        raise ValueError("data Qlib job requires a matching publication receipt")
    actual = build_qlib_publication_receipt(
        data_root, requested_snapshot_name, Path(receipt["dataset_path"])
    )
    if any(receipt.get(key) != value for key, value in actual.items()):
        raise ValueError("Qlib publication receipt does not match the sealed output")
    return actual
