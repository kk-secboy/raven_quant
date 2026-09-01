from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
from governance_fixtures import governed_etf_ready_evidence

import quant_platform.services as services
from quant_data.execution_contract import (
    DAILY_QLIB_FIELD_CONTRACT_VERSION,
    QLIB_OUTPUT_MANIFEST_VERSION,
)
from quant_data.history_bounds import GOVERNED_DAILY_STOCK_SCOPE_VERSION

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
                "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
                "source_volume_unit": "hand",
                "qlib_volume_unit": "share",
                "source_amount_unit": "thousand_cny",
                "qlib_amount_unit": "cny",
                "source_hand_size": 100,
                "index_volume_policy": "excluded_non_tradable_benchmark",
                "lineage_verified": True,
                "governed_etf_whitelist": governed_etf_ready_evidence(),
                "execution_controls": {
                    "scope_version": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
                },
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


def _forget_display_cache(data_root: Path) -> None:
    cache_path = data_root / "qlib" / services._QLIB_DISPLAY_CATALOG_CACHE_FILE
    with services._QLIB_DISPLAY_CATALOG_CACHE_LOCK:
        services._QLIB_DISPLAY_CATALOG_MEMORY.clear()
    cache_path.unlink(missing_ok=True)


def _forget_snapshot_display_cache(data_root: Path) -> None:
    cache_path = (
        data_root
        / "snapshots"
        / services._SNAPSHOT_DISPLAY_CATALOG_CACHE_FILE
    )
    with services._SNAPSHOT_DISPLAY_CATALOG_CACHE_LOCK:
        services._SNAPSHOT_DISPLAY_CATALOG_MEMORY.clear()
    cache_path.unlink(missing_ok=True)


def _seed_snapshot(data_root: Path, name: str, *, frequency: str = "day") -> Path:
    root = data_root / "snapshots" / name
    root.mkdir(parents=True)
    manifest = {
        "name": name,
        "created_at": "2026-08-26T00:00:00+00:00",
        "datasets": {
            "daily" if frequency == "day" else "ashare_5m": {
                "rows": 123,
                "unit_files": 5_000,
                "date_field": "trade_date" if frequency == "day" else "trade_time",
                "date_min": "2024-01-02",
                "date_max": "2026-08-25",
                "source_units": [
                    {"unit_key": f"unit-{index}", "sha256": "a" * 64, "row_count": 1}
                    for index in range(5_000)
                ],
            }
        },
        "frequency": frequency,
        "start_date": "2024-01-02",
        "end_date": "2026-08-25",
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


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


def test_qlib_catalog_does_not_restat_immutable_outputs_on_scheduler_timescale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    _seed_dataset(data_root)
    _forget_memory_cache()
    current = [100.0]
    fingerprint_calls = 0
    original_fingerprint = services._qlib_output_stat_fingerprint

    def counted_fingerprint(*args: object, **kwargs: object) -> str | None:
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        return original_fingerprint(*args, **kwargs)

    monkeypatch.setattr(services.time, "monotonic", lambda: current[0])
    monkeypatch.setattr(
        services, "_qlib_output_stat_fingerprint", counted_fingerprint
    )

    assert services.list_qlib_datasets(data_root)[0]["reproducible"] is True
    initial_calls = fingerprint_calls
    assert initial_calls >= 1

    # The scheduler polls every few seconds.  Even an hour of polls must reuse
    # the immutable catalog proof instead of walking every Qlib feature file.
    current[0] += 60 * 60
    assert services.list_qlib_datasets(data_root)[0]["reproducible"] is True
    assert fingerprint_calls == initial_calls


def test_display_catalog_persists_only_a_bounded_browser_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    dataset, _ = _seed_dataset(data_root)
    _forget_display_cache(data_root)
    strict_row = services.list_qlib_datasets(data_root)[0]
    strict_row["path"] = str(dataset)
    strict_row["provenance"]["output_manifest"]["files"] = [
        {"path": f"features/sh600000/{index}.day.bin", "bytes": 4, "sha256": "a" * 64}
        for index in range(5_000)
    ]
    monkeypatch.setattr(services, "list_qlib_datasets", lambda _root: [strict_row])

    rows = services.refresh_qlib_display_catalog(data_root)
    assert len(rows) == 1
    assert "path" not in rows[0]
    assert "provenance" not in rows[0]
    assert rows[0]["dataset_identity_sha256"] == "a" * 64
    assert rows[0]["daily_contract"]["frequency"] == "day"

    cache_path = data_root / "qlib" / services._QLIB_DISPLAY_CATALOG_CACHE_FILE
    payload = cache_path.read_text(encoding="utf-8")
    assert "output_manifest" not in payload
    assert "features/sh600000" not in payload
    assert cache_path.stat().st_size < 10_000

    # A process restart reads the persisted artifact without invoking the
    # expensive strict inventory again.
    with services._QLIB_DISPLAY_CATALOG_CACHE_LOCK:
        services._QLIB_DISPLAY_CATALOG_MEMORY.clear()
    monkeypatch.setattr(
        services,
        "list_qlib_datasets",
        lambda _root: pytest.fail(
            "strict catalog should not run for an unchanged display signature"
        ),
    )
    assert services.list_qlib_datasets_for_display(data_root) == rows


def test_display_catalog_returns_stale_until_publisher_refreshes_new_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    first_dataset, _ = _seed_dataset(data_root)
    _forget_display_cache(data_root)
    first = {
        "name": first_dataset.name,
        "ready": True,
        "reproducible": True,
        "frequency": "day",
        "provenance": {"dataset_identity_sha256": "a" * 64},
    }
    current = [first]
    monkeypatch.setattr(services, "list_qlib_datasets", lambda _root: current)
    initial = services.refresh_qlib_display_catalog(data_root)

    second_dataset = data_root / "qlib" / "new-publication"
    (second_dataset / "metadata").mkdir(parents=True)
    (second_dataset / "metadata" / "provenance.json").write_text("{}", encoding="utf-8")
    current = [first, {**first, "name": second_dataset.name}]
    started = time.monotonic()
    stale = services.list_qlib_datasets_for_display(data_root)
    assert time.monotonic() - started < 0.5
    assert stale == initial
    assert services.list_qlib_datasets_for_display(data_root) == initial

    refreshed = services.refresh_qlib_display_catalog(data_root)
    assert len(refreshed) == 2
    assert services.list_qlib_datasets_for_display(data_root) == refreshed


def test_display_catalog_observes_a_new_projection_published_by_another_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    first_dataset, _ = _seed_dataset(data_root)
    _forget_display_cache(data_root)
    first = {
        "name": first_dataset.name,
        "ready": True,
        "reproducible": True,
        "frequency": "day",
        "provenance": {"dataset_identity_sha256": "a" * 64},
    }
    monkeypatch.setattr(services, "list_qlib_datasets", lambda _root: [first])
    assert len(services.refresh_qlib_display_catalog(data_root)) == 1

    second_dataset = data_root / "qlib" / "new-publication"
    (second_dataset / "metadata").mkdir(parents=True)
    (second_dataset / "metadata" / "provenance.json").write_text(
        "{}", encoding="utf-8"
    )
    second = services._project_qlib_dataset_for_display(
        {**first, "name": second_dataset.name}
    )
    cache_path = data_root / "qlib" / services._QLIB_DISPLAY_CATALOG_CACHE_FILE
    services._write_qlib_display_catalog_cache(
        cache_path,
        signature=services._qlib_display_catalog_signature(data_root),
        datasets=[services._project_qlib_dataset_for_display(first), second],
    )

    assert len(services.list_qlib_datasets_for_display(data_root)) == 2


def test_display_catalog_observes_same_signature_republished_by_another_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    first_dataset, _ = _seed_dataset(data_root)
    _forget_display_cache(data_root)
    first = {
        "name": first_dataset.name,
        "ready": True,
        "reproducible": True,
        "frequency": "day",
        "provenance": {"dataset_identity_sha256": "a" * 64},
    }
    monkeypatch.setattr(services, "list_qlib_datasets", lambda _root: [first])
    assert services.refresh_qlib_display_catalog(data_root)[0]["ready"] is True

    cache_path = data_root / "qlib" / services._QLIB_DISPLAY_CATALOG_CACHE_FILE
    services._write_qlib_display_catalog_cache(
        cache_path,
        signature=services._qlib_display_catalog_signature(data_root),
        datasets=[
            services._project_qlib_dataset_for_display({**first, "ready": False})
        ],
    )

    assert services.list_qlib_datasets_for_display(data_root)[0]["ready"] is False


def test_display_catalog_publisher_reports_atomic_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    first_dataset, _ = _seed_dataset(data_root)
    _forget_display_cache(data_root)
    first = {
        "name": first_dataset.name,
        "ready": True,
        "reproducible": True,
        "frequency": "day",
        "provenance": {"dataset_identity_sha256": "a" * 64},
    }
    monkeypatch.setattr(services, "list_qlib_datasets", lambda _root: [first])
    monkeypatch.setattr(
        services.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("cache volume is read-only")),
    )

    with pytest.raises(OSError, match="read-only"):
        services.refresh_qlib_display_catalog(data_root)


@pytest.mark.parametrize("damaged_cache", [False, True])
def test_display_catalog_cold_cache_never_starts_a_strict_scan_from_http_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damaged_cache: bool,
) -> None:
    data_root = tmp_path / "data"
    cache_path = data_root / "qlib" / services._QLIB_DISPLAY_CATALOG_CACHE_FILE
    _forget_display_cache(data_root)
    if damaged_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("{damaged", encoding="utf-8")

    calls = 0

    def forbidden_refresh(_root: Path) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        raise AssertionError("display read must not start a strict refresh")

    monkeypatch.setattr(services, "refresh_qlib_display_catalog", forbidden_refresh)
    started = time.monotonic()
    assert services.list_qlib_datasets_for_display(data_root) == []
    assert time.monotonic() - started < 0.5
    assert services.list_qlib_datasets_for_display(data_root) == []
    assert calls == 0


def test_qlib_experiment_catalog_reads_only_db_selected_bounded_artifacts(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    root = data_root / "artifacts" / "qlib"
    selected = ["a" * 32, "b" * 32]
    ignored = "c" * 32
    for job_id in [*selected, ignored]:
        directory = root / job_id
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(
            json.dumps({"score": job_id[0]}), encoding="utf-8"
        )

    rows = services.list_qlib_experiments(
        data_root,
        job_ids=(selected[1], selected[0], selected[1], "../escape"),
        limit=2,
    )

    assert [item["id"] for item in rows] == [selected[1], selected[0]]
    assert all(item["id"] != ignored for item in rows)


def test_snapshot_display_catalog_streams_only_bounded_coverage_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    manifest = _seed_snapshot(data_root, "snapshot-a")
    _forget_snapshot_display_cache(data_root)

    rows = services.refresh_snapshot_display_catalog(data_root)
    assert rows == [
        {
            "name": "snapshot-a",
            "created_at": "2026-08-26T00:00:00+00:00",
            "datasets": {
                "daily": {
                    "rows": 123,
                    "unit_files": 5_000,
                    "date_field": "trade_date",
                    "date_min": "2024-01-02",
                    "date_max": "2026-08-25",
                }
            },
            "frequency": "day",
            "start_date": "2024-01-02",
            "end_date": "2026-08-25",
            "invalid": False,
        }
    ]
    cache_path = (
        data_root
        / "snapshots"
        / services._SNAPSHOT_DISPLAY_CATALOG_CACHE_FILE
    )
    cache_text = cache_path.read_text(encoding="utf-8")
    assert "source_units" not in cache_text
    assert "unit-4999" not in cache_text
    assert cache_path.stat().st_size < 5_000
    assert manifest.stat().st_size > 500_000

    with services._SNAPSHOT_DISPLAY_CATALOG_CACHE_LOCK:
        services._SNAPSHOT_DISPLAY_CATALOG_MEMORY.clear()
    monkeypatch.setattr(
        services,
        "_stream_snapshot_display_summary",
        lambda _path: pytest.fail("unchanged manifest should use the persisted summary"),
    )
    assert services.list_snapshots_for_display(data_root) == rows


def test_snapshot_display_catalog_returns_stale_until_publisher_refreshes(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _seed_snapshot(data_root, "snapshot-a")
    _forget_snapshot_display_cache(data_root)
    initial = services.refresh_snapshot_display_catalog(data_root)

    _seed_snapshot(data_root, "snapshot-b", frequency="5min")
    started = time.monotonic()
    assert services.list_snapshots_for_display(data_root) == initial
    assert time.monotonic() - started < 0.5

    assert services.list_snapshots_for_display(data_root) == initial
    refreshed = services.refresh_snapshot_display_catalog(data_root)
    assert [item["name"] for item in refreshed] == ["snapshot-b", "snapshot-a"]
    assert refreshed[0]["frequency"] == "5min"


@pytest.mark.parametrize("damaged_cache", [False, True])
def test_snapshot_display_catalog_cold_cache_never_starts_a_manifest_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damaged_cache: bool,
) -> None:
    data_root = tmp_path / "data"
    cache_path = (
        data_root
        / "snapshots"
        / services._SNAPSHOT_DISPLAY_CATALOG_CACHE_FILE
    )
    _forget_snapshot_display_cache(data_root)
    if damaged_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("not-json", encoding="utf-8")

    calls = 0

    def blocked_refresh(_root: Path) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        raise AssertionError("HTTP display read must not refresh snapshot manifests")

    monkeypatch.setattr(services, "refresh_snapshot_display_catalog", blocked_refresh)
    started = time.monotonic()
    assert services.list_snapshots_for_display(data_root) == []
    assert time.monotonic() - started < 0.5
    assert services.list_snapshots_for_display(data_root) == []
    assert calls == 0
