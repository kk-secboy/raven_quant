from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from quant_data.catalog import (
    CORE_DAILY,
    CORPORATE_EVENTS,
    ETF_DAILY,
    FUNDAMENTALS,
    RESEARCH_DAILY,
)
from quant_data.checkpoint import CheckpointStore
from quant_data.config import Settings
from quant_data.execution_contract import QLIB_OUTPUT_MANIFEST_VERSION
from quant_data.qlib_builder import verify_qlib_output_manifest

_QLIB_OUTPUT_VERIFY_CACHE_LOCK = threading.Lock()
_QLIB_OUTPUT_VERIFY_MEMORY_TTL_SECONDS = 60.0
_QLIB_OUTPUT_VERIFY_CACHE_MAX_ENTRIES = 128
_QLIB_OUTPUT_VERIFY_CACHE_VERSION = 2
_QLIB_OUTPUT_VERIFY_CACHE_FILE = ".catalog-output-verification-v2.json"
_QLIB_PROVENANCE_PATH = "metadata/provenance.json"
_QLIB_OUTPUT_VERIFY_MEMORY: dict[str, dict[str, Any]] = {}

# The strict catalog above is deliberately expensive: it restats every sealed
# output before it may describe a dataset as reproducible.  Browser pages do
# not make capital decisions, so they read this small, persisted projection
# instead.  A publication changes the lightweight marker signature; HTTP
# reads keep returning the previous good projection, and the data worker
# replaces it after publication.  Formal consumers continue to call
# ``list_qlib_datasets``.
_QLIB_DISPLAY_CATALOG_CACHE_VERSION = 4
_QLIB_DISPLAY_CATALOG_CACHE_FILE = ".catalog-display-v4.json"
_QLIB_DISPLAY_CATALOG_CACHE_MAX_BYTES = 5_000_000
_QLIB_DISPLAY_CATALOG_MAX_DATASETS = 5_000
_QLIB_DISPLAY_CATALOG_CACHE_LOCK = threading.Lock()
_QLIB_DISPLAY_CATALOG_MEMORY: dict[str, dict[str, Any]] = {}
_DAILY_QLIB_DISPLAY_CONTRACT_FIELDS = (
    "frequency",
    "field_contract_version",
    "source_volume_unit",
    "qlib_volume_unit",
    "source_amount_unit",
    "qlib_amount_unit",
    "source_hand_size",
    "index_volume_policy",
    "lineage_verified",
    "governed_etf_whitelist",
    "execution_controls",
)

# Snapshot manifests preserve every source-unit hash and can be hundreds of
# megabytes each.  The browser only needs snapshot identity and dataset-level
# coverage.  Keep a separate, non-authoritative catalog that is parsed with
# bounded memory and published by the data worker when a new immutable
# snapshot directory appears. Browser requests only read the last artifact.
_SNAPSHOT_DISPLAY_CATALOG_CACHE_VERSION = 1
_SNAPSHOT_DISPLAY_CATALOG_CACHE_FILE = ".catalog-display-v1.json"
_SNAPSHOT_DISPLAY_CATALOG_CACHE_MAX_BYTES = 5_000_000
_SNAPSHOT_DISPLAY_CATALOG_CACHE_LOCK = threading.Lock()
_SNAPSHOT_DISPLAY_CATALOG_MEMORY: dict[str, dict[str, Any]] = {}
_SNAPSHOT_DISPLAY_TOP_LEVEL_FIELDS = frozenset(
    {
        "created_at",
        "end_date",
        "frequency",
        "name",
        "snapshot_type",
        "start_date",
    }
)
_SNAPSHOT_DISPLAY_DATASET_FIELDS = frozenset(
    {
        "date_field",
        "date_max",
        "date_min",
        "empty_units",
        "rows",
        "source_rows",
        "unit_files",
    }
)


def _qlib_output_stat_fingerprint(
    dataset_path: Path,
    provenance: dict[str, Any],
    *,
    provenance_sha256: str,
) -> str | None:
    """Fingerprint the exact sealed file set without reading feature bodies."""

    recorded = provenance.get("output_manifest")
    files = recorded.get("files") if isinstance(recorded, dict) else None
    if (
        not isinstance(recorded, dict)
        or recorded.get("version") != QLIB_OUTPUT_MANIFEST_VERSION
        or not isinstance(files, list)
        or not files
    ):
        return None

    expected: dict[str, int] = {}
    for item in files:
        if not isinstance(item, dict):
            return None
        relative_text = str(item.get("path") or "")
        relative = Path(relative_text)
        size = item.get("bytes")
        sha256 = str(item.get("sha256") or "").lower()
        if (
            not relative_text
            or relative.is_absolute()
            or relative.as_posix() != relative_text
            or ".." in relative.parts
            or relative_text == _QLIB_PROVENANCE_PATH
            or relative_text in expected
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            return None
        expected[relative_text] = size

    actual: dict[str, tuple[int, int, int]] = {}
    try:
        root = dataset_path.resolve(strict=True)
        for directory, directory_names, file_names in os.walk(root):
            directory_names.sort()
            file_names.sort()
            directory_path = Path(directory)
            for file_name in file_names:
                target = directory_path / file_name
                relative_text = target.relative_to(root).as_posix()
                if relative_text == _QLIB_PROVENANCE_PATH:
                    continue
                stat = target.stat()
                actual[relative_text] = (
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                    int(stat.st_ctime_ns),
                )
    except OSError:
        return None
    if set(actual) != set(expected) or any(
        actual[relative][0] != size for relative, size in expected.items()
    ):
        return None

    digest = hashlib.sha256()
    digest.update(f"v{_QLIB_OUTPUT_VERIFY_CACHE_VERSION}\0".encode())
    digest.update(provenance_sha256.encode("ascii"))
    for relative in sorted(actual):
        size, mtime_ns, ctime_ns = actual[relative]
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(f"{size}:{mtime_ns}:{ctime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def _read_qlib_output_verify_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("version") != _QLIB_OUTPUT_VERIFY_CACHE_VERSION
        or not isinstance(payload.get("entries"), dict)
    ):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for key, value in payload["entries"].items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        fingerprint = str(value.get("fingerprint") or "").lower()
        provenance_sha256 = str(value.get("provenance_sha256") or "").lower()
        verified_at_ns = value.get("verified_at_ns")
        if (
            len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
            or len(provenance_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in provenance_sha256
            )
            or isinstance(verified_at_ns, bool)
            or not isinstance(verified_at_ns, int)
            or verified_at_ns < 0
        ):
            continue
        result[key] = {
            "fingerprint": fingerprint,
            "provenance_sha256": provenance_sha256,
            "verified_at_ns": verified_at_ns,
        }
    return result


def _write_qlib_output_verify_cache(
    cache_path: Path, entries: dict[str, dict[str, Any]]
) -> None:
    ordered = sorted(
        entries.items(),
        key=lambda item: int(item[1].get("verified_at_ns") or 0),
        reverse=True,
    )[:_QLIB_OUTPUT_VERIFY_CACHE_MAX_ENTRIES]
    payload = {
        "version": _QLIB_OUTPUT_VERIFY_CACHE_VERSION,
        "entries": dict(ordered),
    }
    temporary = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, cache_path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _remember_qlib_output_verification(
    dataset_key: str,
    *,
    provenance_sha256: str,
    verified: bool,
) -> None:
    now = time.monotonic()
    _QLIB_OUTPUT_VERIFY_MEMORY[dataset_key] = {
        "provenance_sha256": provenance_sha256,
        "verified": verified,
        "expires_at": now + _QLIB_OUTPUT_VERIFY_MEMORY_TTL_SECONDS,
    }
    expired = [
        key
        for key, value in _QLIB_OUTPUT_VERIFY_MEMORY.items()
        if float(value.get("expires_at") or 0) <= now
    ]
    for key in expired:
        _QLIB_OUTPUT_VERIFY_MEMORY.pop(key, None)
    while len(_QLIB_OUTPUT_VERIFY_MEMORY) > _QLIB_OUTPUT_VERIFY_CACHE_MAX_ENTRIES:
        _QLIB_OUTPUT_VERIFY_MEMORY.pop(next(iter(_QLIB_OUTPUT_VERIFY_MEMORY)))


def _cached_qlib_output_verification(
    dataset_path: Path,
    provenance: dict[str, Any],
    *,
    provenance_sha256: str,
) -> tuple[bool, str]:
    """Catalog-check sealed outputs with bounded cache and metadata-only refreshes.

    A one-minute memory TTL prevents polling storms. After it expires, every
    sealed path is restatted and the exact path/size/mtime/ctime collection is
    compared with the manifest. An unchanged fingerprint may reuse a persisted
    successful full-SHA result. Formal consumers intentionally do not use this
    cache and still call ``verify_qlib_output_manifest`` before reading data.
    """

    try:
        dataset_key = str(dataset_path.resolve(strict=True))
    except OSError:
        return False, "failed"
    with _QLIB_OUTPUT_VERIFY_CACHE_LOCK:
        memory = _QLIB_OUTPUT_VERIFY_MEMORY.get(dataset_key)
        if (
            memory is not None
            and memory.get("provenance_sha256") == provenance_sha256
            and float(memory.get("expires_at") or 0) > time.monotonic()
        ):
            verified = bool(memory.get("verified"))
            return verified, "cached_full_sha256" if verified else "failed"

        fingerprint = _qlib_output_stat_fingerprint(
            dataset_path,
            provenance,
            provenance_sha256=provenance_sha256,
        )
        cache_path = dataset_path.parent / _QLIB_OUTPUT_VERIFY_CACHE_FILE
        entries = _read_qlib_output_verify_cache(cache_path)
        persisted = entries.get(dataset_key)
        if fingerprint is None:
            if persisted is not None:
                entries.pop(dataset_key, None)
                _write_qlib_output_verify_cache(cache_path, entries)
            _remember_qlib_output_verification(
                dataset_key,
                provenance_sha256=provenance_sha256,
                verified=False,
            )
            return False, "failed"
        if (
            persisted is not None
            and persisted.get("fingerprint") == fingerprint
            and persisted.get("provenance_sha256") == provenance_sha256
        ):
            _remember_qlib_output_verification(
                dataset_key,
                provenance_sha256=provenance_sha256,
                verified=True,
            )
            return True, "cached_full_sha256"

        try:
            verify_qlib_output_manifest(dataset_path, provenance)
            stable_fingerprint = _qlib_output_stat_fingerprint(
                dataset_path,
                provenance,
                provenance_sha256=provenance_sha256,
            )
            verified = stable_fingerprint == fingerprint
        except (OSError, ValueError):
            verified = False
        if verified:
            entries[dataset_key] = {
                "fingerprint": fingerprint,
                "provenance_sha256": provenance_sha256,
                "verified_at_ns": time.time_ns(),
            }
        else:
            entries.pop(dataset_key, None)
        _write_qlib_output_verify_cache(cache_path, entries)
        _remember_qlib_output_verification(
            dataset_key,
            provenance_sha256=provenance_sha256,
            verified=verified,
        )
        return verified, "full_sha256" if verified else "failed"


def resolve_snapshot_dataset(
    data_root: Path,
    *,
    snapshot_name: str,
    dataset_name: str,
) -> dict[str, Any]:
    snapshots_root = (data_root / "snapshots").resolve()
    snapshot = (snapshots_root / snapshot_name).resolve()
    try:
        snapshot.relative_to(snapshots_root)
    except ValueError as exc:
        raise ValueError("snapshot name resolves outside the snapshot root") from exc
    manifest_path = snapshot / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("immutable snapshot manifest is missing or invalid") from exc
    entry = manifest.get("datasets", {}).get(dataset_name)
    if not isinstance(entry, dict):
        raise ValueError(f"snapshot does not contain dataset {dataset_name}")
    source_sha256 = str(entry.get("source_sha256") or "")
    if len(source_sha256) != 64:
        raise ValueError("snapshot dataset has no source identity")
    files = entry.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("snapshot dataset has no immutable Parquet files")
    resolved_files: list[str] = []
    parquet_root = (snapshot / "parquet").resolve()
    dataset_path = (parquet_root / dataset_name).resolve()
    try:
        dataset_path.relative_to(parquet_root)
    except ValueError as exc:
        raise ValueError("dataset name resolves outside the snapshot Parquet root") from exc
    for item in files:
        relative = Path(str(item.get("path") or ""))
        target = (snapshot / relative).resolve()
        try:
            target.relative_to(snapshot)
        except ValueError as exc:
            raise ValueError("snapshot manifest contains an unsafe file path") from exc
        if not target.is_file() or target.stat().st_size != int(item.get("bytes") or -1):
            raise ValueError(f"snapshot file is missing or has the wrong size: {relative}")
        resolved_files.append(str(target))
    return {
        "snapshot_name": snapshot_name,
        "dataset_name": dataset_name,
        "snapshot_path": str(snapshot),
        "dataset_path": str(dataset_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_sha256": source_sha256,
        "snapshot_lineage_id": manifest.get("lineage_id"),
        "snapshot_lineage_generation": manifest.get("lineage_generation"),
        "parent_snapshot": manifest.get("parent_snapshot"),
        "start_date": manifest.get("start_date"),
        "end_date": manifest.get("end_date"),
        "rows": int(entry.get("rows") or 0),
        "files": resolved_files,
    }


def resolve_snapshot_manifest(data_root: Path, snapshot_name: str) -> dict[str, Any]:
    """Resolve and validate an immutable snapshot without trusting its name as a path."""

    snapshots_root = (data_root / "snapshots").resolve()
    snapshot = (snapshots_root / snapshot_name).resolve()
    try:
        snapshot.relative_to(snapshots_root)
    except ValueError as exc:
        raise ValueError("snapshot name resolves outside the snapshot root") from exc
    manifest_path = snapshot / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("immutable snapshot manifest is missing or invalid") from exc
    if not isinstance(manifest, dict):
        raise ValueError("immutable snapshot manifest is missing or invalid")
    return {
        "name": snapshot.name,
        "path": str(snapshot),
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def dataset_catalog(checkpoint: CheckpointStore) -> list[dict[str, Any]]:
    profiles: dict[str, str] = {
        "stock_basic": "core",
        "trade_cal": "core",
        "index_basic": "core",
        "index_daily": "core",
        "index_dailybasic": "core",
        "index_weight": "core",
        "fund_basic": "research",
        "index_classify": "research",
        "index_member_all": "research",
        "disclosure_date": "research",
        "news": "full",
    }
    profiles.update({item.name: "core" for item in CORE_DAILY})
    profiles.update({item.name: "research" for item in (*RESEARCH_DAILY, *ETF_DAILY)})
    profiles.update({item.name: "full" for item in (*FUNDAMENTALS, *CORPORATE_EVENTS)})
    profiles.update(
        {
            "margin_eligibility": "research",
            "indices_1m": "research",
            "etf_1m": "research",
            "futures_1m": "research",
            "options_1m": "research",
            "liquid_stocks_1m": "research",
        }
    )
    aggregates: dict[str, dict[str, int]] = defaultdict(
        lambda: {"planned": 0, "succeeded": 0, "failed": 0, "running": 0, "rows": 0}
    )
    for row in checkpoint.counts():
        item = aggregates[row["dataset"]]
        item["planned"] += int(row["units"])
        item[row["status"]] = int(row["units"])
        item["rows"] += int(row["rows"])
    result = []
    for name, profile in sorted(profiles.items(), key=lambda item: (item[1], item[0])):
        stats = aggregates[name]
        completed = stats["succeeded"]
        planned = stats["planned"]
        state = "ready" if planned and completed == planned else "partial" if completed else "empty"
        result.append(
            {
                "name": name,
                "profile": profile,
                **stats,
                "coverage": round(completed / planned * 100, 1) if planned else 0.0,
                "state": state,
            }
        )
    return result


class CheckpointCatalogProjection:
    """Persist and refresh checkpoint aggregates outside HTTP request work.

    ``work_units`` can contain millions of immutable rows, so calculating the
    same grouped counts independently for the overview and dataset catalog on
    every browser poll is needlessly expensive.  This cache is deliberately
    kept at the API/display boundary: strict research and admission code still
    queries its authoritative inputs directly.
    """

    def __init__(
        self,
        checkpoint: CheckpointStore,
        *,
        ttl_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        cache_path: Path | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._checkpoint = checkpoint
        self._ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._cache_path = cache_path
        self._catalog = self._read_cache()
        self._checked_at = 0.0
        self._refreshing = False

    def _read_cache(self) -> list[dict[str, Any]] | None:
        if self._cache_path is None:
            return None
        try:
            if self._cache_path.stat().st_size > 10_000_000:
                return None
            payload = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        catalog = payload.get("catalog") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or not isinstance(catalog, list)
            or any(not isinstance(item, dict) for item in catalog)
        ):
            return None
        return catalog

    def _write_cache(self, catalog: list[dict[str, Any]]) -> None:
        if self._cache_path is None:
            return
        temporary = self._cache_path.with_name(
            f".{self._cache_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(
                    {"version": 1, "generated_at_ns": time.time_ns(), "catalog": catalog},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            os.replace(temporary, self._cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    def refresh(self) -> list[dict[str, Any]]:
        catalog = dataset_catalog(self._checkpoint)
        try:
            self._write_cache(catalog)
        except OSError:
            # The in-memory display projection remains usable when a read-only
            # or temporarily full cache volume prevents persistence.
            pass
        with self._lock:
            self._catalog = catalog
            self._checked_at = self._clock()
            self._refreshing = False
        return catalog

    def _refresh_in_background(self) -> None:
        try:
            self.refresh()
        except Exception:
            with self._lock:
                self._checked_at = self._clock()
                self._refreshing = False

    def get(self) -> list[dict[str, Any]]:
        now = self._clock()
        launch = False
        with self._lock:
            if self._catalog is not None and now - self._checked_at < self._ttl_seconds:
                return self._catalog
            if not self._refreshing:
                self._refreshing = True
                launch = True
            catalog = self._catalog or []
        if launch:
            try:
                threading.Thread(
                    target=self._refresh_in_background,
                    name="checkpoint-catalog-projection",
                    daemon=True,
                ).start()
            except Exception:
                with self._lock:
                    self._refreshing = False
                    self._checked_at = self._clock()
        return catalog


def list_snapshots(data_root: Path) -> list[dict[str, Any]]:
    root = data_root / "snapshots"
    snapshots = []
    if not root.exists():
        return snapshots
    for path in sorted((item for item in root.iterdir() if item.is_dir()), reverse=True):
        manifest_path = path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("snapshot manifest must be a JSON object")
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            manifest = {"name": path.name, "datasets": {}, "invalid": True}
        snapshots.append(manifest)
    return snapshots


def _snapshot_display_inventory(data_root: Path) -> tuple[str, dict[str, list[int]]]:
    root = data_root / "snapshots"
    digest = hashlib.sha256()
    digest.update(f"v{_SNAPSHOT_DISPLAY_CATALOG_CACHE_VERSION}\0".encode("ascii"))
    markers: dict[str, list[int]] = {}
    if not root.exists():
        digest.update(b"missing")
        return digest.hexdigest(), markers
    try:
        directories = sorted(
            (item for item in root.iterdir() if item.is_dir()),
            key=lambda item: item.name,
        )
    except OSError:
        digest.update(b"unreadable")
        return digest.hexdigest(), markers
    for directory in directories:
        manifest = directory / "manifest.json"
        try:
            stat = manifest.stat()
            marker = [int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns)]
        except OSError:
            marker = [-1, -1, -1]
        markers[directory.name] = marker
        digest.update(directory.name.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
        digest.update(f"{marker[0]}:{marker[1]}:{marker[2]}\n".encode("ascii"))
    return digest.hexdigest(), markers


def _snapshot_display_scalar(line: str) -> tuple[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith('"') or ":" not in stripped:
        return None
    raw_key, raw_value = stripped.split(":", 1)
    raw_value = raw_value.strip().removesuffix(",").strip()
    if not raw_value or raw_value[0] in "[{":
        return None
    try:
        key = json.loads(raw_key)
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        return None
    if not isinstance(key, str) or isinstance(value, (dict, list)):
        return None
    return key, value


def _stream_snapshot_display_summary(manifest_path: Path) -> dict[str, Any]:
    """Read a pretty-printed snapshot manifest without materializing source units.

    SnapshotStorage is the only publisher and writes two-space-indented JSON.
    Dataset source-unit arrays are the large portion; line-by-line parsing
    retains only immediate scalar coverage fields and dataset names.
    """

    summary: dict[str, Any] = {"datasets": {}}
    datasets = summary["datasets"]
    in_datasets = False
    current_dataset: str | None = None
    saw_document_end = False
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                stripped = line.lstrip(" ")
                indent = len(line) - len(stripped)
                if not in_datasets:
                    if indent == 0 and stripped.strip() == "}":
                        saw_document_end = True
                        continue
                    if indent == 2 and stripped.startswith('"datasets"'):
                        in_datasets = True
                        current_dataset = None
                        continue
                    if indent == 2:
                        scalar = _snapshot_display_scalar(line)
                        if scalar and scalar[0] in _SNAPSHOT_DISPLAY_TOP_LEVEL_FIELDS:
                            summary[scalar[0]] = scalar[1]
                    continue
                if indent == 2 and stripped.startswith("}"):
                    in_datasets = False
                    current_dataset = None
                    continue
                if indent == 4 and stripped.startswith('"') and ":" in stripped:
                    raw_key, raw_value = stripped.split(":", 1)
                    if raw_value.strip().startswith("{"):
                        try:
                            current_dataset = str(json.loads(raw_key))
                        except json.JSONDecodeError:
                            current_dataset = None
                        if current_dataset:
                            datasets[current_dataset] = {}
                    continue
                if indent == 6 and current_dataset:
                    scalar = _snapshot_display_scalar(line)
                    if scalar and scalar[0] in _SNAPSHOT_DISPLAY_DATASET_FIELDS:
                        datasets[current_dataset][scalar[0]] = scalar[1]
    except (OSError, UnicodeDecodeError):
        return {
            "name": manifest_path.parent.name,
            "datasets": {},
            "invalid": True,
        }
    declared_name = summary.get("name")
    summary["name"] = manifest_path.parent.name
    summary["invalid"] = bool(
        not saw_document_end or declared_name != manifest_path.parent.name
    )
    return summary


def _read_snapshot_display_catalog(cache_path: Path) -> dict[str, Any] | None:
    try:
        if cache_path.stat().st_size > _SNAPSHOT_DISPLAY_CATALOG_CACHE_MAX_BYTES:
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    entries = payload.get("entries") if isinstance(payload, dict) else None
    signature = payload.get("signature") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("version") != _SNAPSHOT_DISPLAY_CATALOG_CACHE_VERSION
        or not isinstance(signature, str)
        or len(signature) != 64
        or not isinstance(entries, dict)
        or any(
            not isinstance(name, str)
            or not isinstance(entry, dict)
            or not isinstance(entry.get("marker"), list)
            or len(entry["marker"]) != 3
            or not all(isinstance(value, int) for value in entry["marker"])
            or not isinstance(entry.get("summary"), dict)
            for name, entry in entries.items()
        )
    ):
        return None
    return payload


def _snapshot_display_catalog_cache_fingerprint(
    cache_path: Path,
) -> tuple[int, int, int] | None:
    try:
        stat = cache_path.stat()
    except OSError:
        return None
    return int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns)


def _write_snapshot_display_catalog(cache_path: Path, payload: dict[str, Any]) -> None:
    temporary = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, cache_path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def refresh_snapshot_display_catalog(data_root: Path) -> list[dict[str, Any]]:
    root = data_root / "snapshots"
    cache_path = root / _SNAPSHOT_DISPLAY_CATALOG_CACHE_FILE
    cache_key = str(cache_path.resolve())
    previous = _read_snapshot_display_catalog(cache_path) or {}
    previous_entries = previous.get("entries") or {}
    for _attempt in range(2):
        before_signature, markers = _snapshot_display_inventory(data_root)
        entries: dict[str, dict[str, Any]] = {}
        for name, marker in markers.items():
            old = previous_entries.get(name)
            if isinstance(old, dict) and old.get("marker") == marker:
                summary = old.get("summary")
            else:
                summary = _stream_snapshot_display_summary(root / name / "manifest.json")
            entries[name] = {"marker": marker, "summary": summary}
        after_signature, after_markers = _snapshot_display_inventory(data_root)
        if before_signature == after_signature and markers == after_markers:
            payload = {
                "version": _SNAPSHOT_DISPLAY_CATALOG_CACHE_VERSION,
                "signature": after_signature,
                "generated_at_ns": time.time_ns(),
                "entries": entries,
            }
            _write_snapshot_display_catalog(cache_path, payload)
            payload["_cache_fingerprint"] = (
                _snapshot_display_catalog_cache_fingerprint(cache_path)
            )
            with _SNAPSHOT_DISPLAY_CATALOG_CACHE_LOCK:
                _SNAPSHOT_DISPLAY_CATALOG_MEMORY[cache_key] = payload
            return [entries[name]["summary"] for name in sorted(entries, reverse=True)]
        previous_entries = entries
    raise RuntimeError("snapshot publication changed during display catalog refresh")


def list_snapshots_for_display(data_root: Path) -> list[dict[str, Any]]:
    root = data_root / "snapshots"
    cache_path = root / _SNAPSHOT_DISPLAY_CATALOG_CACHE_FILE
    cache_key = str(cache_path.resolve())
    with _SNAPSHOT_DISPLAY_CATALOG_CACHE_LOCK:
        payload = _SNAPSHOT_DISPLAY_CATALOG_MEMORY.get(cache_key)
        fingerprint = _snapshot_display_catalog_cache_fingerprint(cache_path)
        if payload is None or payload.get("_cache_fingerprint") != fingerprint:
            disk_payload = _read_snapshot_display_catalog(cache_path)
            if disk_payload is not None:
                disk_payload["_cache_fingerprint"] = fingerprint
                payload = disk_payload
                _SNAPSHOT_DISPLAY_CATALOG_MEMORY[cache_key] = disk_payload
        if payload is None:
            return []
        entries = payload["entries"]
        return [entries[name]["summary"] for name in sorted(entries, reverse=True)]


def list_qlib_datasets(data_root: Path) -> list[dict[str, Any]]:
    root = data_root / "qlib"
    datasets = []
    if not root.exists():
        return datasets
    for path in sorted((item for item in root.iterdir() if item.is_dir()), reverse=True):
        provenance_path = path / "metadata" / "provenance.json"
        try:
            provenance_bytes = provenance_path.read_bytes()
            provenance = json.loads(provenance_bytes)
            provenance_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
        except (FileNotFoundError, json.JSONDecodeError):
            provenance = None
            provenance_sha256 = ""
        frequency = str((provenance or {}).get("frequency") or "day")
        calendar = path / "calendars" / f"{frequency}.txt"
        instrument_candidates = (
            path / "instruments" / "cn_all.txt",
            path / "instruments" / "liquid_all.txt",
            path / "instruments" / "all.txt",
        )
        instruments = next(
            (candidate for candidate in instrument_candidates if candidate.exists()),
            instrument_candidates[-1],
        )
        features = path / "features"
        output_manifest = (provenance or {}).get("output_manifest")
        sealed_outputs = bool(
            isinstance(output_manifest, dict)
            and output_manifest.get("version") == QLIB_OUTPUT_MANIFEST_VERSION
            and isinstance(output_manifest.get("files"), list)
            and output_manifest["files"]
        )
        output_files_verified = False
        output_verification = "unsealed"
        if sealed_outputs and provenance is not None:
            output_files_verified, output_verification = _cached_qlib_output_verification(
                path,
                provenance,
                provenance_sha256=provenance_sha256,
            )
        days = calendar.read_text(encoding="utf-8").splitlines() if calendar.exists() else []
        stocks = (
            instruments.read_text(encoding="utf-8").splitlines() if instruments.exists() else []
        )
        datasets.append(
            {
                "name": path.name,
                "path": str(path),
                "ready": bool(days and stocks and features.exists()),
                "reproducible": bool(
                    provenance
                    and provenance.get("dataset_identity_sha256")
                    and provenance.get("snapshot_manifest_sha256")
                    and output_files_verified
                ),
                "provenance": provenance,
                "frequency": frequency,
                "lineage_id": (provenance or {}).get("dataset_lineage_id"),
                "lineage_verified": bool((provenance or {}).get("lineage_verified")),
                "output_files_verified": output_files_verified,
                "output_verification": output_verification,
                "start_date": days[0] if days else None,
                "end_date": days[-1] if days else None,
                "trading_days": len(days),
                "instruments": len(stocks),
            }
        )
    return datasets


def _qlib_display_catalog_signature(data_root: Path) -> str:
    """Fingerprint dataset publication markers without walking feature files."""

    root = data_root / "qlib"
    digest = hashlib.sha256()
    digest.update(f"v{_QLIB_DISPLAY_CATALOG_CACHE_VERSION}\0".encode("ascii"))
    if not root.exists():
        digest.update(b"missing")
        return digest.hexdigest()

    try:
        datasets = sorted(
            (item for item in root.iterdir() if item.is_dir()),
            key=lambda item: item.name,
        )
    except OSError:
        digest.update(b"unreadable")
        return digest.hexdigest()

    for dataset in datasets:
        digest.update(dataset.name.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
        # Qlib datasets are immutable once their provenance manifest is
        # published.  These small directories/files are sufficient to detect
        # a new publication or an in-progress dataset becoming readable; the
        # potentially millions of feature files are intentionally not walked.
        marker_paths = [
            dataset / "metadata" / "provenance.json",
            dataset / "calendars",
            dataset / "instruments",
            dataset / "features",
        ]
        for marker in marker_paths:
            try:
                stat = marker.stat()
                digest.update(marker.relative_to(dataset).as_posix().encode("utf-8"))
                digest.update(
                    f":{int(stat.st_size)}:{int(stat.st_mtime_ns)}:{int(stat.st_ctime_ns)}\n".encode(
                        "ascii"
                    )
                )
            except OSError:
                digest.update(marker.relative_to(dataset).as_posix().encode("utf-8"))
                digest.update(b":missing\n")

        for directory_name in ("calendars", "instruments"):
            directory = dataset / directory_name
            try:
                children = sorted(
                    (item for item in directory.iterdir() if item.is_file()),
                    key=lambda item: item.name,
                )
            except OSError:
                children = []
            for child in children:
                try:
                    stat = child.stat()
                except OSError:
                    continue
                digest.update(child.relative_to(dataset).as_posix().encode("utf-8"))
                digest.update(
                    f":{int(stat.st_size)}:{int(stat.st_mtime_ns)}:{int(stat.st_ctime_ns)}\n".encode(
                        "ascii"
                    )
                )
    return digest.hexdigest()


def _read_qlib_display_catalog_cache(cache_path: Path) -> dict[str, Any] | None:
    try:
        if cache_path.stat().st_size > _QLIB_DISPLAY_CATALOG_CACHE_MAX_BYTES:
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    signature = payload.get("signature") if isinstance(payload, dict) else None
    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("version") != _QLIB_DISPLAY_CATALOG_CACHE_VERSION
        or not isinstance(signature, str)
        or len(signature) != 64
        or any(character not in "0123456789abcdef" for character in signature)
        or not isinstance(datasets, list)
        or len(datasets) > _QLIB_DISPLAY_CATALOG_MAX_DATASETS
        or any(not isinstance(item, dict) for item in datasets)
        or any(
            forbidden in item
            for item in datasets
            for forbidden in ("path", "provenance", "output_manifest")
        )
    ):
        return None
    return payload


def _qlib_display_catalog_cache_fingerprint(cache_path: Path) -> tuple[int, int, int] | None:
    """Return a metadata-only cross-process publication fingerprint."""

    try:
        stat = cache_path.stat()
    except OSError:
        return None
    return int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns)


def _write_qlib_display_catalog_cache(
    cache_path: Path,
    *,
    signature: str,
    datasets: list[dict[str, Any]],
) -> None:
    payload = {
        "version": _QLIB_DISPLAY_CATALOG_CACHE_VERSION,
        "signature": signature,
        "generated_at_ns": time.time_ns(),
        "datasets": datasets,
    }
    temporary = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, cache_path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _project_qlib_dataset_for_display(dataset: dict[str, Any]) -> dict[str, Any]:
    """Drop the sealed output manifest and host path from browser responses.

    A production provenance document can contain hundreds of thousands of
    per-file hashes.  Sending it through a catalog endpoint made a nominal
    status request exceed 100 MiB even after the filesystem scan was cached.
    The UI needs only this bounded summary; authoritative identity remains in
    the immutable provenance file and is re-read by strict consumers.
    """

    provenance = dataset.get("provenance")
    provenance_row = provenance if isinstance(provenance, dict) else {}
    daily_contract = {
        key: provenance_row.get(key)
        for key in _DAILY_QLIB_DISPLAY_CONTRACT_FIELDS
        if key in provenance_row
    }
    return {
        key: dataset.get(key)
        for key in (
            "name",
            "ready",
            "reproducible",
            "frequency",
            "lineage_id",
            "lineage_verified",
            "output_files_verified",
            "output_verification",
            "start_date",
            "end_date",
            "trading_days",
            "instruments",
        )
    } | {
        "dataset_identity_sha256": provenance_row.get("dataset_identity_sha256"),
        "snapshot_manifest_sha256": provenance_row.get("snapshot_manifest_sha256"),
        "daily_contract": daily_contract,
    }


def refresh_qlib_display_catalog(data_root: Path) -> list[dict[str, Any]]:
    """Synchronously rebuild the non-authoritative browser projection."""

    datasets: list[dict[str, Any]] | None = None
    signature = ""
    # Never label a scan of the old publication with the signature of a newer
    # one.  Snapshot publication is normally atomic; the retry only covers the
    # narrow interval where a directory changes during the catalog pass.
    for _attempt in range(2):
        before = _qlib_display_catalog_signature(data_root)
        strict_rows = list_qlib_datasets(data_root)
        after = _qlib_display_catalog_signature(data_root)
        if before == after:
            signature = after
            datasets = [
                _project_qlib_dataset_for_display(item)
                for item in strict_rows
            ]
            break
    if datasets is None:
        raise RuntimeError("Qlib dataset publication changed during display catalog refresh")
    cache_path = data_root / "qlib" / _QLIB_DISPLAY_CATALOG_CACHE_FILE
    cache_key = str(cache_path.resolve())
    payload = {
        "version": _QLIB_DISPLAY_CATALOG_CACHE_VERSION,
        "signature": signature,
        "generated_at_ns": time.time_ns(),
        "datasets": datasets,
    }
    _write_qlib_display_catalog_cache(
        cache_path,
        signature=signature,
        datasets=datasets,
    )
    payload["_cache_fingerprint"] = _qlib_display_catalog_cache_fingerprint(cache_path)
    with _QLIB_DISPLAY_CATALOG_CACHE_LOCK:
        _QLIB_DISPLAY_CATALOG_MEMORY[cache_key] = payload
    return datasets


def list_qlib_datasets_for_display(data_root: Path) -> list[dict[str, Any]]:
    """Read the last publisher-generated browser projection without scanning.

    HTTP requests never launch ``list_qlib_datasets``.  A new Qlib publication
    may make the signature stale, but the previous good projection remains
    visible until the data worker explicitly calls
    :func:`refresh_qlib_display_catalog`.  Formal consumers still use the
    strict catalog and therefore keep all immutable-output checks.
    """

    cache_path = data_root / "qlib" / _QLIB_DISPLAY_CATALOG_CACHE_FILE
    cache_key = str(cache_path.resolve())
    signature = _qlib_display_catalog_signature(data_root)
    with _QLIB_DISPLAY_CATALOG_CACHE_LOCK:
        payload = _QLIB_DISPLAY_CATALOG_MEMORY.get(cache_key)
        cache_fingerprint = _qlib_display_catalog_cache_fingerprint(cache_path)
        disk_payload: dict[str, Any] | None = None
        if payload is None or payload.get("_cache_fingerprint") != cache_fingerprint:
            disk_payload = _read_qlib_display_catalog_cache(cache_path)
            if disk_payload is not None:
                disk_payload["_cache_fingerprint"] = cache_fingerprint
                _QLIB_DISPLAY_CATALOG_MEMORY[cache_key] = disk_payload
                payload = disk_payload
        if payload is not None and payload.get("signature") == signature:
            return payload["datasets"]
        # The Qlib publisher normally runs in another container/process.  A
        # mismatching in-memory signature must therefore re-read the tiny
        # persisted projection; otherwise an API process could serve yesterday's
        # catalog forever even though the worker successfully published today's.
        if disk_payload is None:
            disk_payload = _read_qlib_display_catalog_cache(cache_path)
            if disk_payload is not None:
                disk_payload["_cache_fingerprint"] = cache_fingerprint
        if disk_payload is not None and disk_payload.get("signature") == signature:
            _QLIB_DISPLAY_CATALOG_MEMORY[cache_key] = disk_payload
            return disk_payload["datasets"]
        candidates = [item for item in (payload, disk_payload) if item is not None]
        if candidates:
            newest = max(
                candidates,
                key=lambda item: int(item.get("generated_at_ns") or 0),
            )
            _QLIB_DISPLAY_CATALOG_MEMORY[cache_key] = newest
            return newest["datasets"]
    return []


def list_qlib_experiments(
    data_root: Path,
    *,
    job_ids: list[str] | tuple[str, ...],
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Read a bounded DB-selected experiment set without scanning artifacts."""

    if limit <= 0 or limit > 200:
        raise ValueError("Qlib experiment limit must be between 1 and 200")
    root = data_root / "artifacts" / "qlib"
    experiments = []
    if not root.exists():
        return experiments
    seen: set[str] = set()
    for raw_job_id in job_ids:
        job_id = str(raw_job_id).strip().lower()
        if (
            len(experiments) >= limit
            or job_id in seen
            or len(job_id) != 32
            or any(character not in "0123456789abcdef" for character in job_id)
        ):
            continue
        seen.add(job_id)
        path = root / job_id
        result_path = path / "result.json"
        try:
            if result_path.stat().st_size > 5_000_000:
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict):
            continue
        experiments.append({"id": job_id, **result})
    return experiments


def _execution_environment_label(*, wsl: bool = False) -> str:
    if wsl:
        return "Windows / WSL · CPU"
    system = platform.system() or ("Windows" if os.name == "nt" else "Linux")
    containerized = Path("/.dockerenv").exists() or bool(os.getenv("container"))
    parts = [system]
    if containerized:
        parts.append("Docker")
    parts.append("CPU")
    return " · ".join(parts)


def probe_qlib(settings: Settings, project_root: Path) -> dict[str, Any]:
    if settings.qlib_worker_url:
        try:
            response = requests.get(
                f"{settings.qlib_worker_url}/qlib/status",
                timeout=8,
            )
            response.raise_for_status()
            result = response.json()
            if isinstance(result, dict):
                # In production the API and Qlib worker are sibling Linux
                # containers.  Older locked worker images may not yet return
                # this display-only field, so the API supplies its deployment
                # environment without inventing a WSL label.
                result.setdefault("execution_environment", _execution_environment_label())
            return result
        except (requests.RequestException, ValueError) as exc:
            return {"status": "unavailable", "error": str(exc)}
    script = project_root / "scripts" / "run_qlib_baseline.py"
    is_wsl = os.name == "nt" and settings.qlib_python.startswith("/")
    command = (
        [
            "wsl",
            "-d",
            settings.qlib_wsl_distro,
            "--exec",
            settings.qlib_python,
            _windows_to_wsl(script),
            "--probe",
        ]
        if is_wsl
        else [settings.qlib_python, str(script), "--probe"]
    )
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=45, check=True)
        line = next(
            (item for item in reversed(completed.stdout.splitlines()) if item.startswith("{")),
            "{}",
        )
        result = json.loads(line)
        if isinstance(result, dict):
            result.setdefault(
                "execution_environment",
                _execution_environment_label(wsl=is_wsl),
            )
        return result
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "error": str(exc)}


def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}{resolved.as_posix().split(':', 1)[1]}" if drive else resolved.as_posix()


def system_summary(
    settings: Settings,
    checkpoint: CheckpointStore,
    jobs: list[dict],
    data_tasks: list[dict] | None = None,
    *,
    catalog: list[dict[str, Any]] | None = None,
) -> dict:
    # Multiple operational endpoints may reuse one bounded display snapshot.
    # Without one this remains a live query, preserving existing semantics.
    catalog = catalog if catalog is not None else dataset_catalog(checkpoint)
    total_rows = sum(item["rows"] for item in catalog)
    planned = sum(item["planned"] for item in catalog)
    succeeded = sum(item["succeeded"] for item in catalog)
    running_work_units = sum(item["running"] for item in catalog)
    snapshots = list_snapshots_for_display(settings.data_root)
    # Overview is operational display only; strict consumers independently
    # re-verify the selected dataset before research or capital use.
    qlib_datasets = len(list_qlib_datasets_for_display(settings.data_root))
    active_job_rows = [job for job in jobs if job["status"] in {"queued", "running"}]
    active_jobs = len(active_job_rows)
    active_bootstrap_jobs = sum(job.get("kind") == "bootstrap" for job in active_job_rows)
    active_finalize_jobs = sum(
        job.get("kind") in {"data_verify", "data_snapshot", "data_qlib", "qlib_build"}
        for job in active_job_rows
    )
    actionable_tasks = [
        task
        for task in (data_tasks or [])
        if task.get("implementation_status") not in {"permission_probe", "external_source_required"}
    ]
    ready_tasks = sum(task.get("status") == "succeeded" for task in actionable_tasks)
    partial_tasks = sum(task.get("status") == "partial" for task in actionable_tasks)
    failed_tasks = sum(task.get("status") == "failed" for task in actionable_tasks)
    running_tasks = sum(task.get("status") in {"queued", "running"} for task in actionable_tasks)
    retry_waiting_tasks = sum(
        task.get("execution_phase")
        in {"rate_limit_cooldown", "retry_waiting", "recoverable_failure"}
        for task in actionable_tasks
    )
    blocked_tasks = sum(
        task.get("execution_phase") == "blocked_prerequisite" for task in actionable_tasks
    )
    terminal_failed_tasks = sum(
        task.get("execution_phase") == "terminal_failure" for task in actionable_tasks
    )
    startable_tasks = sum(
        task.get("execution_phase") in {"ready_to_start", "partial"}
        and task.get("dependencies_satisfied") is True
        for task in actionable_tasks
    )
    waiting_tasks = max(
        len(actionable_tasks) - ready_tasks - partial_tasks - failed_tasks - running_tasks,
        0,
    )
    readiness_percent = (
        round(
            sum(
                100.0 if task.get("status") == "succeeded" else float(task.get("coverage") or 0.0)
                for task in actionable_tasks
            )
            / len(actionable_tasks),
            1,
        )
        if actionable_tasks
        else 0.0
    )
    legacy_download_coverage = round(succeeded / planned * 100, 1) if planned else 0.0
    return {
        "mode": "local-research",
        "credentials_configured": bool(settings.api_url and settings.token),
        "data_root": str(settings.data_root),
        "rows": total_rows,
        "planned_units": planned,
        "succeeded_units": succeeded,
        # Kept for API compatibility. The Web UI labels this as the legacy
        # foundational download completion and never as overall readiness.
        "coverage": legacy_download_coverage,
        "legacy_download_coverage": legacy_download_coverage,
        "readiness_percent": readiness_percent,
        "ready_tasks": ready_tasks,
        "actionable_tasks": len(actionable_tasks),
        "partial_tasks": partial_tasks,
        "failed_tasks": failed_tasks,
        "running_tasks": running_tasks,
        "waiting_tasks": waiting_tasks,
        "retry_waiting_tasks": retry_waiting_tasks,
        "blocked_tasks": blocked_tasks,
        "terminal_failed_tasks": terminal_failed_tasks,
        "startable_tasks": startable_tasks,
        "snapshots": len(snapshots),
        "qlib_datasets": qlib_datasets,
        "active_jobs": active_jobs,
        # Action-specific counts keep unrelated long-running data jobs from
        # disabling every control in the data center.
        "active_bootstrap_jobs": active_bootstrap_jobs,
        "active_finalize_jobs": active_finalize_jobs,
        # This is the live checkpoint activity across datasets, independent of
        # whether a long-running CLI job has emitted a structured progress
        # artifact yet. It prevents the UI from claiming "0 requests" while
        # the downloader has leased work units.
        "running_work_units": running_work_units,
        "components": [
            {"name": "PostgreSQL", "state": "ready"},
            {"name": "Data Center", "state": "ready"},
            {"name": "Qlib", "state": "configured" if settings.qlib_python else "needs_setup"},
            {"name": "RD-Agent", "state": "configured"},
            {"name": "Paper Portfolio", "state": "ready"},
            {"name": "Scheduler", "state": "configured" if settings.scheduler_url else "local"},
        ],
    }
