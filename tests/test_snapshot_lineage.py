import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from quant_data.snapshot_lineage import (
    assert_snapshot_descendant,
    canonical_sha256,
    file_contract_sha256,
    make_lineage_id,
    prepare_lineage_metadata,
    resolve_verified_snapshot_anchor,
    verify_snapshot_lineage,
)

pytestmark = pytest.mark.no_database


def _unit(key: str, digest: str, rows: int) -> dict[str, object]:
    return {"unit_key": key, "sha256": digest, "row_count": rows}


def _write_manifest(root: Path, name: str, manifest: dict[str, object]) -> None:
    path = root / name
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_industry_anchor(root: Path, name: str = "industry-anchor") -> Path:
    path = root / name
    dataset_file = path / "parquet" / "index_member_all" / "data.parquet"
    dataset_file.parent.mkdir(parents=True)
    dataset_file.write_bytes(b"sealed-industry-history")
    file_sha256 = hashlib.sha256(dataset_file.read_bytes()).hexdigest()
    configuration = {"source": "verified-old-contract"}
    manifest = {
        "name": name,
        "profile": "full",
        "start_date": "2008-01-01",
        "end_date": "2026-08-28",
        "quality_gate": {"ok": True},
        "lineage_id": make_lineage_id("qlib_daily_source", configuration),
        "lineage_contract": {
            "kind": "qlib_daily_source",
            "configuration": configuration,
        },
        "lineage_generation": 0,
        "parent_snapshot": None,
        "parent_manifest_sha256": None,
        "datasets": {
            "index_member_all": {
                "rows": 1,
                "source_sha256": "a" * 64,
                "files": [
                    {
                        "path": "parquet/index_member_all/data.parquet",
                        "bytes": dataset_file.stat().st_size,
                        "sha256": file_sha256,
                    }
                ],
            }
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


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


def test_resolves_verified_cross_lineage_industry_anchor(tmp_path: Path) -> None:
    anchor = _write_industry_anchor(tmp_path)

    resolved, evidence = resolve_verified_snapshot_anchor(
        tmp_path,
        anchor.name,
        required_profile="full",
        required_start=date(2008, 1, 1),
        maximum_end=date(2026, 8, 31),
        required_dataset="index_member_all",
    )

    assert resolved == anchor.resolve()
    assert evidence["snapshot_name"] == anchor.name
    assert evidence["end_date"] == "2026-08-28"
    assert evidence["carry_rule_version"] == (
        "index-member-delisted-parent-carry-v1"
    )
    assert len(evidence["manifest_sha256"]) == 64
    assert len(evidence["dataset_files_sha256"]) == 64
    unsigned = dict(evidence)
    evidence_sha256 = unsigned.pop("evidence_sha256")
    assert evidence_sha256 == canonical_sha256(unsigned)


@pytest.mark.parametrize(
    ("required_profile", "required_start", "maximum_end", "error"),
    [
        ("core", date(2008, 1, 1), date(2026, 8, 31), "profile does not match"),
        ("full", date(2009, 1, 1), date(2026, 8, 31), "start date does not match"),
        ("full", date(2008, 1, 1), date(2026, 8, 27), "ends after"),
    ],
)
def test_industry_anchor_rejects_incompatible_scope(
    tmp_path: Path,
    required_profile: str,
    required_start: date,
    maximum_end: date,
    error: str,
) -> None:
    anchor = _write_industry_anchor(tmp_path)

    with pytest.raises(ValueError, match=error):
        resolve_verified_snapshot_anchor(
            tmp_path,
            anchor.name,
            required_profile=required_profile,
            required_start=required_start,
            maximum_end=maximum_end,
            required_dataset="index_member_all",
        )


def test_industry_anchor_rejects_failed_quality_and_tampered_bytes(
    tmp_path: Path,
) -> None:
    anchor = _write_industry_anchor(tmp_path)
    manifest_path = anchor / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["quality_gate"] = {"ok": False}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="passing quality gate"):
        resolve_verified_snapshot_anchor(
            tmp_path,
            anchor.name,
            required_profile="full",
            required_start=date(2008, 1, 1),
            maximum_end=date(2026, 8, 31),
            required_dataset="index_member_all",
        )

    manifest["quality_gate"] = {"ok": True}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (anchor / "parquet" / "index_member_all" / "data.parquet").write_bytes(
        b"tampered"
    )
    with pytest.raises(ValueError, match="hash does not match"):
        resolve_verified_snapshot_anchor(
            tmp_path,
            anchor.name,
            required_profile="full",
            required_start=date(2008, 1, 1),
            maximum_end=date(2026, 8, 31),
            required_dataset="index_member_all",
        )


def test_industry_anchor_rejects_unsafe_duplicate_or_unmanifested_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="safe snapshot name"):
        resolve_verified_snapshot_anchor(
            tmp_path,
            "../industry-anchor",
            required_profile="full",
            required_start=date(2008, 1, 1),
            maximum_end=date(2026, 8, 31),
            required_dataset="index_member_all",
        )

    anchor = _write_industry_anchor(tmp_path)
    manifest_path = anchor / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["datasets"]["index_member_all"]["files"] *= 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="repeats a dataset file path"):
        resolve_verified_snapshot_anchor(
            tmp_path,
            anchor.name,
            required_profile="full",
            required_start=date(2008, 1, 1),
            maximum_end=date(2026, 8, 31),
            required_dataset="index_member_all",
        )

    manifest["datasets"]["index_member_all"]["files"] = manifest["datasets"][
        "index_member_all"
    ]["files"][:1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (anchor / "parquet" / "index_member_all" / "extra.parquet").write_bytes(b"extra")
    with pytest.raises(ValueError, match="unmanifested dataset files"):
        resolve_verified_snapshot_anchor(
            tmp_path,
            anchor.name,
            required_profile="full",
            required_start=date(2008, 1, 1),
            maximum_end=date(2026, 8, 31),
            required_dataset="index_member_all",
        )
