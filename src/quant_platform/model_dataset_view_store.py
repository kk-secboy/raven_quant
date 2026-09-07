"""Atomically published, fully checksummed pre-final provider views for one job."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from . import rdagent_dataset_view as view_runtime
from .model_cell_store import _regular, atomic_json, read_json
from .model_prepared_cache import _locked, _mkdir, _safe
from .model_research_governance import canonical_sha256, file_sha256


def _inventory(provider: Path) -> dict[str, str]:
    result = {}
    for directory, directories, files in os.walk(_safe(provider), followlinks=False):
        for name in directories:
            _safe(Path(directory) / name)
        for name in files:
            path = _regular(Path(directory) / name)
            result[path.relative_to(provider).as_posix()] = file_sha256(path)
    if not result or not any(name.startswith("features/") for name in result):
        raise ValueError("stable model provider view has no physical feature files")
    return dict(sorted(result.items()))


def prepare_model_dataset_view(
    provider: Path, root: Path, *, cutoff: str, provenance: dict,
) -> Path:
    request = {
        "contract_version": "model-dataset-view-store-v1",
        "source": str(provider.resolve()), "provider_provenance": provenance,
        "cutoff": cutoff,
        "view_source_sha256": file_sha256(Path(view_runtime.__file__)),
        "store_source_sha256": file_sha256(Path(__file__)),
    }
    root = _mkdir(root)
    key = canonical_sha256(request)
    destination = _safe(root / key)
    with _locked(root / f"{key}.lock", exclusive=True):
        if destination.exists():
            receipt = read_json(destination / "receipt.json")
            digest = receipt.pop("receipt_sha256", None)
            if (
                digest != canonical_sha256(receipt) or receipt.get("request") != request
                or _inventory(destination / "provider") != receipt.get("files")
            ):
                raise ValueError("stable model provider view failed full artifact verification")
            # Retain the original interval-boundary, calendar and universe validation.
            return view_runtime.prepare_rdagent_dataset_view(
                provider, destination / "provider", cutoff=cutoff,
            )
        staging = root / f".{key}.{uuid.uuid4().hex}.building"
        staging.mkdir()
        view = view_runtime.prepare_rdagent_dataset_view(
            provider, staging / "provider", cutoff=cutoff,
        )
        receipt = {"request": request, "files": _inventory(view)}
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_json(staging / "receipt.json", receipt)
        # Staging stays outside the published identity on interruption; never overwrite a view.
        staging.rename(destination)
        if os.name != "nt":
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return destination / "provider"
