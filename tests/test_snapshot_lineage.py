import json
from datetime import date
from pathlib import Path

import pytest

from quant_data.snapshot_lineage import (
    assert_snapshot_descendant,
    file_contract_sha256,
    make_lineage_id,
    prepare_lineage_metadata,
    verify_snapshot_lineage,
)


def _unit(key: str, digest: str, rows: int) -> dict[str, object]:
    return {"unit_key": key, "sha256": digest, "row_count": rows}


def _write_manifest(root: Path, name: str, manifest: dict[str, object]) -> None:
    path = root / name
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_prepares_append_only_snapshot_successor(tmp_path: Path) -> None:
    configuration = {"start": "2024-01-01"}
    lineage_id = make_lineage_id("daily", configuration)
    ancestor = {
        "name": "daily-v1",
        "lineage_id": lineage_id,
        "lineage_generation": 0,
        "lineage_contract": {"kind": "daily", "configuration": configuration},
        "parent_snapshot": None,
        "parent_manifest_sha256": None,
        "start_date": "2024-01-01",
        "end_date": "2024-01-02",
        "datasets": {
            "daily": {"source_units": [_unit("day-1", "a" * 64, 10)]},
        },
    }
    _write_manifest(tmp_path, "daily-v1", ancestor)

    metadata = prepare_lineage_metadata(
        tmp_path,
        lineage_id=lineage_id,
        end_date=date(2024, 1, 3),
        successful_units={
            "daily": [
                _unit("day-1", "a" * 64, 10),
                _unit("day-2", "b" * 64, 11),
            ]
        },
    )

    assert metadata["parent_snapshot"] == "daily-v1"
    assert metadata["lineage_generation"] == 1
    assert len(str(metadata["parent_manifest_sha256"])) == 64


def test_successor_allows_new_dataset_and_forward_reference_refresh(tmp_path: Path) -> None:
    configuration = {"start": "2024-01-01"}
    lineage_id = make_lineage_id("daily", configuration)
    ancestor = {
        "name": "daily-v1",
        "lineage_id": lineage_id,
        "lineage_generation": 0,
        "lineage_contract": {"kind": "daily", "configuration": configuration},
        "parent_snapshot": None,
        "parent_manifest_sha256": None,
        "start_date": "2024-01-01",
        "end_date": "2024-01-02",
        "datasets": {
            "stock_basic": {
                "source_units": [_unit("master-old", "a" * 64, 10)],
                "reference_refresh": {
                    "selected_buckets": ["2024-01-01"],
                },
            },
        },
    }
    _write_manifest(tmp_path, "daily-v1", ancestor)
    refreshed = {
        **_unit("master-new", "b" * 64, 11),
        "scope_json": {"reference_refresh_bucket": "2024-02-01"},
    }

    metadata = prepare_lineage_metadata(
        tmp_path,
        lineage_id=lineage_id,
        end_date=date(2024, 2, 2),
        successful_units={
            "stock_basic": [refreshed],
            "daily": [_unit("day-1", "c" * 64, 20)],
        },
    )

    assert metadata["parent_snapshot"] == "daily-v1"
    assert metadata["lineage_generation"] == 1


def test_verifies_contract_and_rejects_forged_lineage_id(tmp_path: Path) -> None:
    configuration = {"start": "2024-01-01"}
    manifest = {
        "name": "daily-v1",
        "lineage_id": make_lineage_id("daily", configuration),
        "lineage_contract": {"kind": "daily", "configuration": configuration},
        "lineage_generation": 0,
        "parent_snapshot": None,
        "parent_manifest_sha256": None,
        "start_date": "2024-01-01",
        "end_date": "2024-01-02",
        "datasets": {"daily": {"source_units": [_unit("day-1", "a" * 64, 10)]}},
    }
    _write_manifest(tmp_path, "daily-v1", manifest)
    assert verify_snapshot_lineage(tmp_path / "daily-v1") == manifest

    manifest["lineage_id"] = "f" * 64
    (tmp_path / "daily-v1" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match"):
        verify_snapshot_lineage(tmp_path / "daily-v1")


def test_file_contract_digest_changes_when_ingestion_code_changes(tmp_path: Path) -> None:
    source = tmp_path / "provider.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    first = file_contract_sha256({"provider": source})
    source.write_text("VERSION = 2\n", encoding="utf-8")
    second = file_contract_sha256({"provider": source})

    assert len(first) == 64
    assert first != second


@pytest.mark.parametrize(
    "candidate_units",
    [
        [_unit("day-1", "c" * 64, 10)],
        [_unit("day-2", "b" * 64, 11)],
        [_unit("day-1", "a" * 64, 9)],
    ],
)
def test_rejects_rewritten_removed_or_recounted_source_units(
    candidate_units: list[dict[str, object]],
) -> None:
    lineage_id = "d" * 64
    ancestor = {
        "lineage_id": lineage_id,
        "start_date": "2024-01-01",
        "end_date": "2024-01-02",
        "datasets": {
            "daily": {"source_units": [_unit("day-1", "a" * 64, 10)]},
        },
    }
    candidate = {
        "lineage_id": lineage_id,
        "start_date": "2024-01-01",
        "end_date": "2024-01-03",
        "datasets": {"daily": {"source_units": candidate_units}},
    }

    with pytest.raises(ValueError, match="rewrote or removed"):
        assert_snapshot_descendant(
            anchor_manifest=ancestor,
            candidate_manifest=candidate,
        )


def test_rejects_cross_lineage_or_earlier_candidate() -> None:
    units = [_unit("day-1", "a" * 64, 10)]
    ancestor = {
        "lineage_id": "a" * 64,
        "start_date": "2024-01-01",
        "end_date": "2024-01-03",
        "datasets": {"daily": {"source_units": units}},
    }
    candidate = {
        **ancestor,
        "lineage_id": "b" * 64,
        "end_date": "2024-01-04",
    }
    with pytest.raises(ValueError, match="not in the anchor lineage"):
        assert_snapshot_descendant(
            anchor_manifest=ancestor,
            candidate_manifest=candidate,
        )

    candidate["lineage_id"] = ancestor["lineage_id"]
    candidate["end_date"] = "2024-01-02"
    with pytest.raises(ValueError, match="ends before"):
        assert_snapshot_descendant(
            anchor_manifest=ancestor,
            candidate_manifest=candidate,
        )
