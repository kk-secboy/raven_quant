from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import quant_platform.services as services
from quant_data.execution_contract import QLIB_OUTPUT_MANIFEST_VERSION

pytestmark = pytest.mark.no_database


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_dataset(data_root: Path) -> tuple[Path, Path]:
    dataset = data_root / "qlib" / "daily-fixture"
    calendar = dataset / "calendars" / "day.txt"
    instruments = dataset / "instruments" / "cn_all.txt"
    feature = dataset / "features" / "sh600000" / "close.day.bin"
    provenance_path = dataset / "metadata" / "provenance.json"
    for path in (calendar, instruments, feature, provenance_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    calendar.write_text("2024-01-02\n", encoding="utf-8")
    instruments.write_text("SH600000\t2024-01-02\t2024-01-02\n", encoding="utf-8")
    feature.write_bytes(b"good")
    files = []
    for path in (calendar, instruments, feature):
        files.append(
            {
                "path": path.relative_to(dataset).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    provenance_path.write_text(
        json.dumps(
            {
                "frequency": "day",
                "dataset_identity_sha256": "a" * 64,
                "snapshot_manifest_sha256": "b" * 64,
                "output_manifest": {
                    "version": QLIB_OUTPUT_MANIFEST_VERSION,
                    "files": files,
                },
            }
        ),
        encoding="utf-8",
    )
    return dataset, feature


def _forget_memory_cache() -> None:
    with services._QLIB_OUTPUT_VERIFY_CACHE_LOCK:
        services._QLIB_OUTPUT_VERIFY_MEMORY.clear()


def test_qlib_catalog_reuses_only_an_unchanged_exact_file_stat_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    dataset, feature = _seed_dataset(data_root)
    original_verify = services.verify_qlib_output_manifest
    calls = 0

    def counted_verify(path: Path, provenance: dict) -> None:
        nonlocal calls
        calls += 1
        original_verify(path, provenance)

    monkeypatch.setattr(services, "verify_qlib_output_manifest", counted_verify)
    _forget_memory_cache()

    first = services.list_qlib_datasets(data_root)[0]
    assert first["reproducible"] is True
    assert first["output_verification"] == "full_sha256"
    assert calls == 1

    second = services.list_qlib_datasets(data_root)[0]
    assert second["reproducible"] is True
    assert second["output_verification"] == "cached_full_sha256"
    assert calls == 1

    # A new process has no memory cache. The persisted cache may be reused only
    # after restatting the exact sealed file collection.
    _forget_memory_cache()
    persisted = services.list_qlib_datasets(data_root)[0]
    assert persisted["reproducible"] is True
    assert persisted["output_verification"] == "cached_full_sha256"
    assert calls == 1

    previous = feature.stat()
    feature.write_bytes(b"evil")
    os.utime(
        feature,
        ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000),
    )
    _forget_memory_cache()
    corrupted = services.list_qlib_datasets(data_root)[0]
    assert corrupted["reproducible"] is False
    assert corrupted["output_files_verified"] is False
    assert corrupted["output_verification"] == "failed"
    assert calls == 2

    feature.write_bytes(b"good")
    os.utime(
        feature,
        ns=(previous.st_atime_ns, previous.st_mtime_ns + 2_000_000_000),
    )
    _forget_memory_cache()
    repaired = services.list_qlib_datasets(data_root)[0]
    assert repaired["reproducible"] is True
    assert calls == 3

    unexpected = dataset / "features" / "sh600000" / "unexpected.day.bin"
    unexpected.write_bytes(b"extra")
    _forget_memory_cache()
    added = services.list_qlib_datasets(data_root)[0]
    assert added["reproducible"] is False
    assert added["output_verification"] == "failed"
    # Collection mismatch is rejected from metadata alone; no body hash needed.
    assert calls == 3
