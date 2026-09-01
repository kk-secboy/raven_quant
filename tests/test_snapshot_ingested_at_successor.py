from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

import quant_data.storage as storage_module
from quant_data.models import ProviderResult
from quant_data.snapshot_lineage import make_lineage_id, verify_snapshot_lineage
from quant_data.storage import ParquetStore

pytestmark = pytest.mark.no_database


class _CheckpointFixture:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.requested: list[set[str]] = []

    def successful_units(self, unit_keys) -> list[dict]:
        requested = set(unit_keys)
        self.requested.append(requested)
        return [dict(row) for row in self.rows if row["unit_key"] in requested]


def _legacy_unit(
    store: ParquetStore,
    *,
    dataset: str,
    unit_key: str,
    rows: list[dict],
    updated_at: datetime,
) -> dict:
    path = store.units_root / dataset / f"{unit_key}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return {
        "unit_key": unit_key,
        "dataset": dataset,
        "status": "succeeded",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "output_path": path.relative_to(store.root).as_posix(),
        "updated_at": updated_at,
    }


def _current_unit(store: ParquetStore) -> dict:
    result = ProviderResult(
        api_name="daily",
        columns=["ts_code", "trade_date", "close"],
        rows=[{"ts_code": "000001.SZ", "trade_date": "20240102", "close": 10.0}],
        raw_body=b"",
    )
    outcome = store.write_unit("daily", "daily-current", result)
    return {
        "unit_key": "daily-current",
        "dataset": "daily",
        "status": "succeeded",
        "sha256": outcome.sha256,
        "row_count": outcome.row_count,
        "output_path": outcome.output_path,
        "updated_at": datetime(2024, 1, 3, tzinfo=UTC),
    }


def _source_snapshot(store: ParquetStore, rows: list[dict]) -> Path:
    selected: dict[str, list[dict]] = {}
    for row in rows:
        selected.setdefault(row["dataset"], []).append(dict(row))
    lineage_contract = {
        "kind": "qlib_daily_source",
        "configuration": {"fixture": "legacy-missing-ingested-at"},
    }
    return store.build_snapshot(
        name="legacy-source",
        successful_units=selected,
        manifest_extra={
            "profile": "full",
            "start_date": "2008-01-01",
            "end_date": "2024-12-31",
            "quality_gate": {"ok": True, "fixture": True},
            "lineage_contract": lineage_contract,
            "lineage_id": make_lineage_id(
                lineage_contract["kind"], lineage_contract["configuration"]
            ),
            "parent_snapshot": None,
            "parent_manifest_sha256": None,
            "lineage_generation": 0,
        },
    )


def _dataset_file(snapshot: Path, dataset: str) -> Path:
    files = sorted((snapshot / "parquet" / dataset).rglob("*.parquet"))
    assert len(files) == 1
    return files[0]


def test_successor_uses_earliest_proven_ledger_time_and_keeps_source_immutable(
    tmp_path: Path,
) -> None:
    store = ParquetStore(tmp_path / "data")
    provider_row = {
        "ts_code": "000001.SZ",
        "ann_date": "20240401",
        "end_date": "20231231",
        "roe": 12.5,
    }
    later = _legacy_unit(
        store,
        dataset="fina_indicator",
        unit_key="fina-later",
        rows=[provider_row],
        updated_at=datetime(2024, 7, 2, 9, tzinfo=UTC),
    )
    earlier = _legacy_unit(
        store,
        dataset="fina_indicator",
        unit_key="fina-earlier",
        rows=[provider_row],
        updated_at=datetime(2024, 7, 1, 8, tzinfo=UTC),
    )
    current = _current_unit(store)
    source = _source_snapshot(store, [later, earlier, current])
    source_manifest_before = (source / "manifest.json").read_bytes()
    source_financial_before = _dataset_file(source, "fina_indicator").read_bytes()
    # Recovery is scoped to fina_indicator.  A raw unit for an unaffected,
    # already-sealed dataset may be unavailable without blocking projection reuse.
    (store.root / current["output_path"]).unlink()
    checkpoint = _CheckpointFixture([later, earlier, current])

    successor = store.build_ingested_at_successor(
        name="recovered-successor",
        source_snapshot=source,
        checkpoint=checkpoint,
    )

    recovered = pd.read_parquet(_dataset_file(successor, "fina_indicator"))
    assert len(recovered) == 1
    assert recovered.loc[0, "ingested_at"] == pd.Timestamp(
        "2024-07-01T08:00:00+00:00"
    )
    assert not _dataset_file(successor, "fina_indicator").samefile(
        _dataset_file(source, "fina_indicator")
    )
    # A dataset with complete row-level acquisition evidence remains the exact
    # immutable projection from the source snapshot.
    assert _dataset_file(successor, "daily").samefile(_dataset_file(source, "daily"))
    assert (source / "manifest.json").read_bytes() == source_manifest_before
    assert _dataset_file(source, "fina_indicator").read_bytes() == source_financial_before

    manifest = verify_snapshot_lineage(successor)
    receipt = manifest["ingested_at_recovery"]
    assert receipt["evidence_source"] == "quantlab.work_units.succeeded.updated_at"
    assert checkpoint.requested == [{"fina-earlier", "fina-later"}]
    assert receipt["recovery_datasets"] == ["fina_indicator"]
    assert receipt["verified_unit_count"] == 2
    assert receipt["recovered_unit_count"] == 2
    assert receipt["recovered_row_count"] == 2
    parity = receipt["provider_parity"]
    assert len(parity["parity_sha256"]) == 64
    assert parity["datasets"][0]["source_minus_target_rows"] == 0
    assert parity["datasets"][0]["target_minus_source_rows"] == 0
    entry = manifest["datasets"]["fina_indicator"]
    assert entry["source_rows"] == 2
    assert entry["rows"] == 1
    assert pd.Timestamp(entry["ingested_at_min"]).tz_convert(UTC) == pd.Timestamp(
        "2024-07-01T08:00:00+00:00"
    )
    assert pd.Timestamp(entry["ingested_at_max"]).tz_convert(UTC) == pd.Timestamp(
        "2024-07-02T09:00:00+00:00"
    )
    assert entry["ingested_at_recovery"]["unit_count"] == 2
    assert entry["provider_parity"]["parity_sha256"] == parity["datasets"][0][
        "parity_sha256"
    ]
    assert entry["source_sha256"] != json.loads(
        source_manifest_before
    )["datasets"]["fina_indicator"]["source_sha256"]


def test_successor_rejects_ledger_identity_drift(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    unit = _legacy_unit(
        store,
        dataset="fina_indicator",
        unit_key="fina",
        rows=[
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240401",
                "end_date": "20231231",
                "roe": 12.5,
            }
        ],
        updated_at=datetime(2024, 7, 1, tzinfo=UTC),
    )
    source = _source_snapshot(store, [unit])
    changed = {**unit, "row_count": 2}

    with pytest.raises(ValueError, match="ledger identity changed"):
        store.build_ingested_at_successor(
            name="rejected-successor",
            source_snapshot=source,
            checkpoint=_CheckpointFixture([changed]),
        )


def test_successor_rejects_changed_durable_file(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    unit = _legacy_unit(
        store,
        dataset="fina_indicator",
        unit_key="fina",
        rows=[
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240401",
                "end_date": "20231231",
                "roe": 12.5,
            }
        ],
        updated_at=datetime(2024, 7, 1, tzinfo=UTC),
    )
    source = _source_snapshot(store, [unit])
    unit_path = store.root / unit["output_path"]
    unit_path.write_bytes(unit_path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="file checksum changed"):
        store.build_ingested_at_successor(
            name="rejected-successor",
            source_snapshot=source,
            checkpoint=_CheckpointFixture([unit]),
        )


def test_successor_rejects_missing_raw_unit_in_affected_dataset(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    unit = _legacy_unit(
        store,
        dataset="fina_indicator",
        unit_key="fina",
        rows=[
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240401",
                "end_date": "20231231",
                "roe": 12.5,
            }
        ],
        updated_at=datetime(2024, 7, 1, tzinfo=UTC),
    )
    source = _source_snapshot(store, [unit])
    (store.root / unit["output_path"]).unlink()

    with pytest.raises(ValueError, match="file is missing or unsafe"):
        store.build_ingested_at_successor(
            name="rejected-successor",
            source_snapshot=source,
            checkpoint=_CheckpointFixture([unit]),
        )


def test_successor_rejects_naive_ledger_time(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    unit = _legacy_unit(
        store,
        dataset="fina_indicator",
        unit_key="fina",
        rows=[
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240401",
                "end_date": "20231231",
                "roe": 12.5,
            }
        ],
        updated_at=datetime(2024, 7, 1),
    )
    source = _source_snapshot(store, [unit])

    with pytest.raises(ValueError, match="not UTC-bound"):
        store.build_ingested_at_successor(
            name="rejected-successor",
            source_snapshot=source,
            checkpoint=_CheckpointFixture([unit]),
        )


def test_successor_recovers_partitioned_dataset_from_raw_unit(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    unit = _legacy_unit(
        store,
        dataset="daily",
        unit_key="daily-legacy",
        rows=[{"ts_code": "000001.SZ", "trade_date": "20240102", "close": 10.0}],
        updated_at=datetime(2024, 1, 3, 4, tzinfo=UTC),
    )
    source = _source_snapshot(store, [unit])

    successor = store.build_ingested_at_successor(
        name="partitioned-successor",
        source_snapshot=source,
        checkpoint=_CheckpointFixture([unit]),
        datasets={"daily"},
    )

    recovered = pd.read_parquet(_dataset_file(successor, "daily"))
    assert recovered.loc[0, "ingested_at"] == pd.Timestamp(
        "2024-01-03T04:00:00+00:00"
    )
    assert not _dataset_file(successor, "daily").samefile(_dataset_file(source, "daily"))


def test_recovery_query_uses_hash_join_shape_for_thousands_of_units() -> None:
    recovery = {
        f"/data/units/fina_indicator/{index}.parquet": {
            "ledger_updated_at": "2024-07-01T00:00:00+00:00"
        }
        for index in range(1_000)
    }

    query = storage_module._ingested_at_recovery_source_query(
        "['/data/units/fina_indicator/0.parquet']",
        {"ts_code", "ingested_at"},
        recovery,
    )

    assert "VALUES" in query
    assert "LEFT JOIN recovery_map" in query
    assert "CASE WHEN" not in query
    assert query.count("CAST(") == 1_000


def test_successor_rejects_reserved_virtual_filename_column(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    unit = _legacy_unit(
        store,
        dataset="fina_indicator",
        unit_key="fina",
        rows=[
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240401",
                "end_date": "20231231",
                "roe": 12.5,
                "filename": "provider-owned-value",
            }
        ],
        updated_at=datetime(2024, 7, 1, tzinfo=UTC),
    )
    source = _source_snapshot(store, [unit])

    with pytest.raises(ValueError, match="reserved recovery column filename"):
        store.build_ingested_at_successor(
            name="rejected-successor",
            source_snapshot=source,
            checkpoint=_CheckpointFixture([unit]),
        )


def test_successor_provider_parity_failure_is_not_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ParquetStore(tmp_path / "data")
    unit = _legacy_unit(
        store,
        dataset="fina_indicator",
        unit_key="fina",
        rows=[
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240401",
                "end_date": "20231231",
                "roe": 12.5,
            }
        ],
        updated_at=datetime(2024, 7, 1, tzinfo=UTC),
    )
    source = _source_snapshot(store, [unit])
    original = storage_module._snapshot_source_query

    def _corrupt_provider_projection(dataset, quoted_paths, columns, *, source_sql=None):
        query = original(
            dataset,
            quoted_paths,
            columns,
            source_sql=source_sql,
        )
        if dataset == "fina_indicator":
            return f"SELECT * REPLACE (roe + 1 AS roe) FROM ({query})"
        return query

    monkeypatch.setattr(
        storage_module,
        "_snapshot_source_query",
        _corrupt_provider_projection,
    )

    with pytest.raises(ValueError, match="provider parity failed"):
        store.build_ingested_at_successor(
            name="parity-failed-successor",
            source_snapshot=source,
            checkpoint=_CheckpointFixture([unit]),
        )
    assert not (store.snapshots_root / "parity-failed-successor").exists()


def test_successor_rejects_unsafe_dataset_name_before_linking(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    unit = _legacy_unit(
        store,
        dataset="fina_indicator",
        unit_key="fina",
        rows=[{"ts_code": "000001.SZ", "ann_date": "20240401", "roe": 12.5}],
        updated_at=datetime(2024, 7, 1, tzinfo=UTC),
    )
    source = _source_snapshot(store, [unit])
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["datasets"]["../escape"] = dict(manifest["datasets"]["fina_indicator"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe dataset name"):
        store.build_ingested_at_successor(
            name="unsafe-successor",
            source_snapshot=source,
            checkpoint=_CheckpointFixture([unit]),
        )


def test_successor_rejects_unmanifested_file_in_unaffected_projection(
    tmp_path: Path,
) -> None:
    store = ParquetStore(tmp_path / "data")
    fina = _legacy_unit(
        store,
        dataset="fina_indicator",
        unit_key="fina",
        rows=[{"ts_code": "000001.SZ", "ann_date": "20240401", "roe": 12.5}],
        updated_at=datetime(2024, 7, 1, tzinfo=UTC),
    )
    daily = _current_unit(store)
    source = _source_snapshot(store, [fina, daily])
    unmanifested = source / "parquet" / "daily" / "unmanifested.txt"
    unmanifested.write_text("not sealed", encoding="utf-8")

    with pytest.raises(ValueError, match="unmanifested"):
        store.build_ingested_at_successor(
            name="unsafe-successor",
            source_snapshot=source,
            checkpoint=_CheckpointFixture([fina, daily]),
        )


def test_successor_rejects_symlink_in_unaffected_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ParquetStore(tmp_path / "data")
    fina = _legacy_unit(
        store,
        dataset="fina_indicator",
        unit_key="fina",
        rows=[{"ts_code": "000001.SZ", "ann_date": "20240401", "roe": 12.5}],
        updated_at=datetime(2024, 7, 1, tzinfo=UTC),
    )
    daily = _current_unit(store)
    source = _source_snapshot(store, [fina, daily])
    link = source / "parquet" / "daily" / "linked.parquet"
    try:
        os.symlink(_dataset_file(source, "daily"), link)
    except OSError:
        # Windows CI accounts may lack SeCreateSymbolicLinkPrivilege.  Preserve
        # deterministic coverage of the fail-closed branch with a path-local
        # filesystem predicate instead of skipping the security assertion.
        link.write_bytes(b"link-placeholder")
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: path == link or original_is_symlink(path),
        )

    with pytest.raises(ValueError, match="tree is unsafe"):
        store.build_ingested_at_successor(
            name="unsafe-successor",
            source_snapshot=source,
            checkpoint=_CheckpointFixture([fina, daily]),
        )


def test_cli_recovery_dataset_is_allowlisted() -> None:
    import typer

    from quant_data.cli import snapshot_ingested_at_successor

    with pytest.raises(typer.BadParameter, match="only the audited fina_indicator"):
        snapshot_ingested_at_successor(
            source="legacy-source",
            name="successor",
            dataset="daily",
        )
