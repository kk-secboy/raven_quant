from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

import quant_data.storage as storage_module
from quant_data.models import ProviderResult
from quant_data.qlib_builder import QlibBuilder
from quant_data.snapshot_lineage import (
    make_lineage_id,
    resolve_verified_snapshot_anchor,
)
from quant_data.storage import ParquetStore

pytestmark = pytest.mark.no_database


def _result(api_name: str, rows: list[dict]) -> ProviderResult:
    columns = list(rows[0].keys()) if rows else []
    return ProviderResult(api_name=api_name, columns=columns, rows=rows, raw_body=b"")


def _write_unit(store: ParquetStore, dataset: str, unit_key: str, rows: list[dict]) -> dict:
    outcome = store.write_unit(dataset, unit_key, _result(dataset, rows))
    return {
        "unit_key": unit_key,
        "sha256": outcome.sha256,
        "row_count": outcome.row_count,
        "output_path": outcome.output_path,
    }


def _daily_rows(day: str, codes: tuple[str, ...] = ("000001.SZ", "000002.SZ")) -> list[dict]:
    return [
        {"ts_code": code, "trade_date": day, "close": 10.0 + index}
        for index, code in enumerate(codes)
    ]


def _dataset_frame(snapshot: Path, dataset: str) -> pd.DataFrame:
    files = sorted((snapshot / "parquet" / dataset).rglob("*.parquet"))
    assert files, f"no parquet files for {dataset}"
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    sort_columns = [column for column in ("ts_code", "trade_date") if column in frame.columns]
    remaining = [column for column in frame.columns if column not in sort_columns]
    return frame.sort_values(sort_columns + remaining).reset_index(drop=True)


def _manifest_entry(snapshot: Path, dataset: str) -> dict:
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    return manifest["datasets"][dataset]


def test_incremental_build_matches_full_rebuild(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    units_a = [
        _write_unit(store, "daily", "daily_20240102", _daily_rows("20240102")),
        _write_unit(store, "daily", "daily_20240103", _daily_rows("20240103")),
    ]
    base = store.build_snapshot(name="s1", successful_units={"daily": units_a}, manifest_extra={})
    units_b = units_a + [_write_unit(store, "daily", "daily_20240201", _daily_rows("20240201"))]
    full = store.build_snapshot(
        name="s2-full", successful_units={"daily": units_b}, manifest_extra={}
    )
    incremental = store.build_snapshot(
        name="s2-inc",
        successful_units={"daily": units_b},
        manifest_extra={},
        base_snapshot=base,
    )

    pd.testing.assert_frame_equal(
        _dataset_frame(full, "daily"), _dataset_frame(incremental, "daily")
    )
    full_entry = _manifest_entry(full, "daily")
    incremental_entry = _manifest_entry(incremental, "daily")
    for key in ("rows", "date_min", "date_max", "source_sha256", "unit_files"):
        assert incremental_entry[key] == full_entry[key], key


def test_financial_snapshot_keeps_latest_pre_window_announcement_state(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    older = {
        "ts_code": "000001.SZ",
        "ann_date": "20230630",
        "end_date": "20230331",
        "total_assets": 90.0,
    }
    carry_in = {
        "ts_code": "000001.SZ",
        "ann_date": "20231220",
        "end_date": "20230930",
        "total_assets": 100.0,
    }
    units = [
        _write_unit(store, "balancesheet", "old", [older, carry_in]),
        # Resumed/overlapping provider units must not duplicate the carry-in row.
        _write_unit(store, "balancesheet", "overlap", [carry_in]),
        _write_unit(
            store,
            "balancesheet",
            "window",
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20240331",
                    "end_date": "20231231",
                    "total_assets": 110.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "ann_date": "20220115",
                    "end_date": "20211231",
                    "total_assets": 50.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "ann_date": "20250101",
                    "end_date": "20241231",
                    "total_assets": 60.0,
                },
            ],
        ),
    ]

    snapshot = store.build_snapshot(
        name="financial-window",
        successful_units={"balancesheet": units},
        manifest_extra={"start_date": "2024-01-01", "end_date": "2024-12-31"},
    )

    frame = _dataset_frame(snapshot, "balancesheet")
    assert frame[["ts_code", "ann_date"]].astype(str).values.tolist() == [
        ["000001.SZ", "2023-12-20"],
        ["000001.SZ", "2024-03-31"],
        ["000002.SZ", "2022-01-15"],
    ]
    entry = _manifest_entry(snapshot, "balancesheet")
    assert entry["date_filter_mode"] == "announcement_pit_carry_in"
    assert entry["date_min"] == "2022-01-15"
    assert entry["date_max"] == "2024-03-31"


def test_clean_partitions_are_hard_linked_and_dirty_ones_rebuilt(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    units_a = [
        _write_unit(store, "daily", "daily_20240102", _daily_rows("20240102")),
    ]
    base = store.build_snapshot(name="s1", successful_units={"daily": units_a}, manifest_extra={})
    units_b = units_a + [_write_unit(store, "daily", "daily_20240201", _daily_rows("20240201"))]
    incremental = store.build_snapshot(
        name="s2",
        successful_units={"daily": units_b},
        manifest_extra={},
        base_snapshot=base,
    )

    january = Path("parquet/daily/partition_year=2024/partition_month=1/data.parquet")
    february = Path("parquet/daily/partition_year=2024/partition_month=2/data.parquet")
    linked = incremental / january
    original = base / january
    assert linked.exists() and original.exists()
    assert linked.samefile(original)
    assert (incremental / february).exists()
    assert not (base / february).exists()


def test_rebuilt_same_size_partition_gets_a_new_content_digest(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    old_unit = _write_unit(
        store,
        "daily",
        "daily_20240102",
        _daily_rows("20240102"),
    )
    base = store.build_snapshot(
        name="s1",
        successful_units={"daily": [old_unit]},
        manifest_extra={},
    )
    new_unit = _write_unit(
        store,
        "daily",
        "daily_20240102",
        [
            {"ts_code": "000001.SZ", "trade_date": "20240102", "close": 20.0},
            {"ts_code": "000002.SZ", "trade_date": "20240102", "close": 21.0},
        ],
    )
    successor = store.build_snapshot(
        name="s2",
        successful_units={"daily": [new_unit]},
        manifest_extra={},
        base_snapshot=base,
    )

    relative = Path("parquet/daily/partition_year=2024/partition_month=1/data.parquet")
    base_file = base / relative
    successor_file = successor / relative
    assert base_file.stat().st_size == successor_file.stat().st_size
    assert not successor_file.samefile(base_file)

    base_entry = _manifest_entry(base, "daily")["files"][0]
    successor_entry = _manifest_entry(successor, "daily")["files"][0]
    actual_sha256 = hashlib.sha256(successor_file.read_bytes()).hexdigest()
    assert successor_entry["sha256"] == actual_sha256
    assert successor_entry["sha256"] != base_entry["sha256"]


def test_unchanged_dataset_is_fully_linked(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    units = [
        _write_unit(store, "daily", "daily_20240102", _daily_rows("20240102")),
    ]
    base = store.build_snapshot(name="s1", successful_units={"daily": units}, manifest_extra={})
    # The same-source fast path must trust the already sealed projection before
    # resolving or opening dormant raw units.
    (store.root / units[0]["output_path"]).unlink()
    successor = store.build_snapshot(
        name="s2",
        successful_units={"daily": units},
        manifest_extra={},
        base_snapshot=base,
    )
    base_files = sorted((base / "parquet" / "daily").rglob("*.parquet"))
    assert base_files
    for path in base_files:
        relative = path.relative_to(base)
        assert (successor / relative).samefile(path), relative
    assert _manifest_entry(successor, "daily") == _manifest_entry(base, "daily")


def test_snapshot_clips_rows_to_exact_requested_range_within_month(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    unit = _write_unit(
        store,
        "daily",
        "daily_202408",
        _daily_rows("20240802") + _daily_rows("20240805"),
    )

    snapshot = store.build_snapshot(
        name="bounded",
        successful_units={"daily": [unit]},
        manifest_extra={"start_date": "2024-08-02", "end_date": "2024-08-02"},
    )

    frame = _dataset_frame(snapshot, "daily")
    entry = _manifest_entry(snapshot, "daily")
    assert set(frame["trade_date"].dt.strftime("%Y%m%d")) == {"20240802"}
    assert entry["rows"] == 2
    assert entry["source_rows"] == 2
    assert entry["date_min"] == "2024-08-02"
    assert entry["date_max"] == "2024-08-02"


def test_bounded_snapshot_rebuilds_parent_that_contains_out_of_range_rows(
    tmp_path: Path,
) -> None:
    store = ParquetStore(tmp_path / "data")
    unit = _write_unit(
        store,
        "daily",
        "daily_202408",
        _daily_rows("20240802") + _daily_rows("20240805"),
    )
    base = store.build_snapshot(
        name="unbounded-parent",
        successful_units={"daily": [unit]},
        manifest_extra={},
    )

    successor = store.build_snapshot(
        name="bounded-successor",
        successful_units={"daily": [unit]},
        manifest_extra={"start_date": "2024-08-02", "end_date": "2024-08-02"},
        base_snapshot=base,
    )

    frame = _dataset_frame(successor, "daily")
    assert set(frame["trade_date"].dt.strftime("%Y%m%d")) == {"20240802"}
    assert _manifest_entry(successor, "daily")["date_max"] == "2024-08-02"
    relative = Path("parquet/daily/partition_year=2024/partition_month=8/data.parquet")
    assert not (successor / relative).samefile(base / relative)


def test_bounded_snapshot_keeps_memberships_overlapping_requested_range(
    tmp_path: Path,
) -> None:
    store = ParquetStore(tmp_path / "data")
    unit = _write_unit(
        store,
        "index_member_all",
        "industry-members",
        [
            {
                "ts_code": "000001.SZ",
                "l1_code": "801780.SI",
                "in_date": "19910403",
                "out_date": None,
            },
            {
                "ts_code": "000002.SZ",
                "l1_code": "801180.SI",
                "in_date": "20060101",
                "out_date": "20091231",
            },
            {
                "ts_code": "000003.SZ",
                "l1_code": "801230.SI",
                "in_date": "20000101",
                "out_date": "20071231",
            },
            {
                "ts_code": "000004.SZ",
                "l1_code": "801750.SI",
                "in_date": "20250101",
                "out_date": None,
            },
        ],
    )

    snapshot = store.build_snapshot(
        name="bounded-memberships",
        successful_units={"index_member_all": [unit]},
        manifest_extra={"start_date": "2008-01-01", "end_date": "2024-12-31"},
    )

    frame = _dataset_frame(snapshot, "index_member_all")
    entry = _manifest_entry(snapshot, "index_member_all")
    assert set(frame["ts_code"]) == {"000001.SZ", "000002.SZ"}
    assert entry["rows"] == 2
    assert entry["source_rows"] == 2
    assert entry["date_field"] is None
    assert entry["date_filter_mode"] == "interval_overlap"


def test_bounded_memberships_do_not_reuse_incompatible_parent(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    unit = _write_unit(
        store,
        "index_member_all",
        "industry-members",
        [
            {
                "ts_code": "000001.SZ",
                "l1_code": "801780.SI",
                "in_date": "19910403",
                "out_date": None,
            },
            {
                "ts_code": "000004.SZ",
                "l1_code": "801750.SI",
                "in_date": "20250101",
                "out_date": None,
            },
        ],
    )
    parent = store.build_snapshot(
        name="unbounded-memberships",
        successful_units={"index_member_all": [unit]},
        manifest_extra={},
    )

    successor = store.build_snapshot(
        name="bounded-memberships-successor",
        successful_units={"index_member_all": [unit]},
        manifest_extra={"start_date": "2008-01-01", "end_date": "2024-12-31"},
        base_snapshot=parent,
    )

    frame = _dataset_frame(successor, "index_member_all")
    assert set(frame["ts_code"]) == {"000001.SZ"}
    assert not (
        successor / "parquet" / "index_member_all" / "data.parquet"
    ).samefile(parent / "parquet" / "index_member_all" / "data.parquet")


def test_membership_successor_carries_only_missing_explicitly_delisted_stocks(
    tmp_path: Path,
) -> None:
    store = ParquetStore(tmp_path / "data")
    parent_members = _write_unit(
        store,
        "index_member_all",
        "parent-members",
        [
            {
                "ts_code": "000001.SZ",
                "l1_code": "801010.SI",
                "in_date": "20000101",
                "out_date": None,
            },
            {
                "ts_code": "000002.SZ",
                "l1_code": "801020.SI",
                "in_date": "20000101",
                "out_date": None,
            },
            {
                "ts_code": "000003.SZ",
                "l1_code": "801030.SI",
                "in_date": "20000101",
                "out_date": None,
            },
            {
                "ts_code": "000004.SZ",
                "l1_code": "801040.SI",
                "in_date": "20000101",
                "out_date": None,
            },
            {
                "ts_code": "000005.SZ",
                "l1_code": "801050.SI",
                "in_date": "20000101",
                "out_date": None,
            },
            {
                "ts_code": "000005.SZ",
                "l1_code": "801060.SI",
                "in_date": "20010101",
                "out_date": None,
            },
        ],
    )
    parent_stock = _write_unit(
        store,
        "stock_basic",
        "parent-stock",
        [
            {"ts_code": "000001.SZ", "list_status": "D"},
            {"ts_code": "000002.SZ", "list_status": "D"},
            {"ts_code": "000003.SZ", "list_status": "L"},
            {"ts_code": "000004.SZ", "list_status": "D"},
            {"ts_code": "000005.SZ", "list_status": "D"},
        ],
    )
    parent = store.build_snapshot(
        name="membership-parent",
        successful_units={
            "index_member_all": [parent_members],
            "stock_basic": [parent_stock],
        },
        manifest_extra={},
    )

    current_members = _write_unit(
        store,
        "index_member_all",
        "current-members",
        [
            {
                "ts_code": "000002.SZ",
                "l1_code": "801120.SI",
                "in_date": "20020101",
                "out_date": None,
            },
            {
                "ts_code": "000099.SZ",
                "l1_code": "801990.SI",
                "in_date": "20200101",
                "out_date": None,
            },
        ],
    )
    current_stock = _write_unit(
        store,
        "stock_basic",
        "current-stock",
        [
            {"ts_code": "000001.SZ", "list_status": "D"},
            {"ts_code": "000002.SZ", "list_status": "D"},
            {"ts_code": "000003.SZ", "list_status": "L"},
            # 000004.SZ intentionally has no current lifecycle state.
            {"ts_code": "000005.SZ", "list_status": "D"},
            {"ts_code": "000099.SZ", "list_status": "L"},
        ],
    )
    successor = store.build_snapshot(
        name="membership-successor",
        successful_units={
            "index_member_all": [current_members],
            "stock_basic": [current_stock],
        },
        manifest_extra={},
        base_snapshot=parent,
    )

    frame = _dataset_frame(successor, "index_member_all")
    assert set(frame["ts_code"]) == {
        "000001.SZ",
        "000002.SZ",
        "000005.SZ",
        "000099.SZ",
    }
    assert frame.loc[frame["ts_code"] == "000002.SZ", "l1_code"].tolist() == [
        "801120.SI"
    ]
    assert "000003.SZ" not in set(frame["ts_code"])
    assert "000004.SZ" not in set(frame["ts_code"])

    carry = _manifest_entry(successor, "index_member_all")[
        "industry_history_carry"
    ]
    assert carry["rule_version"] == "index-member-delisted-parent-carry-v1"
    assert carry["source_snapshot"] == parent.name
    assert carry["source_kind"] == "lineage_parent"
    assert carry["symbols"] == ["000001.SZ", "000005.SZ"]
    assert carry["symbol_count"] == 2
    assert carry["rows"] == 3
    assert len(carry["parent_dataset_source_sha256"]) == 64
    assert len(carry["parent_dataset_files_sha256"]) == 64

    conflict = QlibBuilder(successor)._industry_membership_conflict_issue(
        {"ts_code", "l1_code", "in_date", "out_date"}
    )
    assert conflict is not None
    assert "000005.SZ" in conflict


def test_membership_history_anchor_can_cross_snapshot_lineages(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    anchor_member = _write_unit(
        store,
        "index_member_all",
        "anchor-member",
        [
            {
                "ts_code": "600001.SH",
                "l1_code": "801040.SI",
                "in_date": "19980122",
                "out_date": None,
            }
        ],
    )
    anchor_configuration = {"source": "old-contract"}
    anchor = store.build_snapshot(
        name="old-contract-anchor",
        successful_units={"index_member_all": [anchor_member]},
        manifest_extra={
            "profile": "full",
            "start_date": "2008-01-01",
            "end_date": "2026-08-28",
            "quality_gate": {"ok": True},
            "lineage_id": make_lineage_id(
                "qlib_daily_source", anchor_configuration
            ),
            "lineage_contract": {
                "kind": "qlib_daily_source",
                "configuration": anchor_configuration,
            },
            "lineage_generation": 0,
            "parent_snapshot": None,
            "parent_manifest_sha256": None,
        },
    )
    anchor_path, anchor_evidence = resolve_verified_snapshot_anchor(
        store.snapshots_root,
        anchor.name,
        required_profile="full",
        required_start=date(2008, 1, 1),
        maximum_end=date(2026, 8, 31),
        required_dataset="index_member_all",
    )
    current_member = _write_unit(
        store,
        "index_member_all",
        "new-contract-member",
        [
            {
                "ts_code": "000099.SZ",
                "l1_code": "801990.SI",
                "in_date": "20200101",
                "out_date": None,
            }
        ],
    )
    current_stock = _write_unit(
        store,
        "stock_basic",
        "new-contract-stock",
        [
            {"ts_code": "600001.SH", "list_status": "D"},
            {"ts_code": "000099.SZ", "list_status": "L"},
        ],
    )
    successor = store.build_snapshot(
        name="new-contract-root",
        successful_units={
            "index_member_all": [current_member],
            "stock_basic": [current_stock],
        },
        manifest_extra={
            "lineage_id": "new-contract",
            "industry_history_anchor": anchor_evidence,
        },
        industry_history_anchor=anchor_path,
    )

    frame = _dataset_frame(successor, "index_member_all")
    assert set(frame["ts_code"]) == {"600001.SH", "000099.SZ"}
    carry = _manifest_entry(successor, "index_member_all")[
        "industry_history_carry"
    ]
    assert carry["source_kind"] == "explicit_anchor"
    assert carry["source_snapshot"] == anchor.name
    assert carry["anchor_manifest_sha256"] == anchor_evidence["manifest_sha256"]
    assert carry["anchor_evidence_sha256"] == anchor_evidence["evidence_sha256"]

    with pytest.raises(ValueError, match="new lineage root"):
        store.build_snapshot(
            name="ambiguous-parent-and-anchor",
            successful_units={
                "index_member_all": [current_member],
                "stock_basic": [current_stock],
            },
            manifest_extra={"industry_history_anchor": anchor_evidence},
            base_snapshot=successor,
            industry_history_anchor=anchor_path,
        )

    next_snapshot = store.build_snapshot(
        name="new-contract-next",
        successful_units={
            "index_member_all": [current_member],
            "stock_basic": [current_stock],
        },
        manifest_extra={"lineage_id": "new-contract"},
        base_snapshot=successor,
    )
    next_frame = _dataset_frame(next_snapshot, "index_member_all")
    assert set(next_frame["ts_code"]) == {"600001.SH", "000099.SZ"}
    next_carry = _manifest_entry(next_snapshot, "index_member_all")[
        "industry_history_carry"
    ]
    assert next_carry["source_kind"] == "lineage_parent"
    assert next_carry["source_snapshot"] == successor.name


def test_fund_basic_lifecycle_master_keeps_pre_window_listing(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    unit = _write_unit(
        store,
        "fund_basic",
        "fund-master",
        [
            {
                "ts_code": "510050.SH",
                "market": "E",
                "issue_date": "20041230",
                "list_date": "20050223",
                "delist_date": None,
            },
            {
                "ts_code": "510300.SH",
                "market": "E",
                "issue_date": "20120504",
                "list_date": "20120528",
                "delist_date": None,
            },
        ],
    )

    snapshot = store.build_snapshot(
        name="bounded-fund-master",
        successful_units={"fund_basic": [unit]},
        manifest_extra={"start_date": "2008-01-01", "end_date": "2024-12-31"},
    )

    frame = _dataset_frame(snapshot, "fund_basic")
    entry = _manifest_entry(snapshot, "fund_basic")
    assert set(frame["ts_code"]) == {"510050.SH", "510300.SH"}
    assert entry["date_field"] is None
    assert entry["date_filter_mode"] is None


def test_fund_basic_lifecycle_master_rebuilds_obsolete_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ParquetStore(tmp_path / "data")
    unit = _write_unit(
        store,
        "fund_basic",
        "fund-master",
        [
            {
                "ts_code": "510050.SH",
                "market": "E",
                "issue_date": "20041230",
                "list_date": "20050223",
                "delist_date": None,
            },
            {
                "ts_code": "510300.SH",
                "market": "E",
                "issue_date": "20120504",
                "list_date": "20120528",
                "delist_date": None,
            },
        ],
    )
    current_candidates = storage_module._date_field_candidates
    monkeypatch.setattr(
        storage_module,
        "_date_field_candidates",
        lambda dataset: ("issue_date",)
        if dataset == "fund_basic"
        else current_candidates(dataset),
    )
    parent = store.build_snapshot(
        name="obsolete-fund-parent",
        successful_units={"fund_basic": [unit]},
        manifest_extra={"start_date": "2008-01-01", "end_date": "2024-12-31"},
    )
    assert set(_dataset_frame(parent, "fund_basic")["ts_code"]) == {"510300.SH"}
    assert _manifest_entry(parent, "fund_basic")["date_filter_mode"] == "point_date"

    monkeypatch.setattr(storage_module, "_date_field_candidates", current_candidates)
    successor = store.build_snapshot(
        name="corrected-fund-successor",
        successful_units={"fund_basic": [unit]},
        manifest_extra={"start_date": "2008-01-01", "end_date": "2024-12-31"},
        base_snapshot=parent,
    )

    frame = _dataset_frame(successor, "fund_basic")
    entry = _manifest_entry(successor, "fund_basic")
    assert set(frame["ts_code"]) == {"510050.SH", "510300.SH"}
    assert entry["date_field"] is None
    assert entry["date_filter_mode"] is None


def test_all_null_date_like_column_falls_back_to_non_partitioned_snapshot(
    tmp_path: Path,
) -> None:
    store = ParquetStore(tmp_path / "data")
    unit = _write_unit(
        store,
        "sge_basic",
        "sge-basic",
        [
            {
                "ts_code": "AU9999.SGE",
                "name": "Gold",
                "trade_time": None,
            }
        ],
    )

    snapshot = store.build_snapshot(
        name="sge-reference",
        successful_units={"sge_basic": [unit]},
        manifest_extra={},
    )

    frame = _dataset_frame(snapshot, "sge_basic")
    entry = _manifest_entry(snapshot, "sge_basic")
    assert len(frame) == 1
    assert entry["rows"] == 1
    assert entry["date_field"] is None
    assert len(entry["files"]) == 1


def test_dropped_unit_forces_full_dataset_rebuild_without_stale_rows(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    old_unit = _write_unit(store, "daily", "daily_20240102", _daily_rows("20240102"))
    base = store.build_snapshot(
        name="s1", successful_units={"daily": [old_unit]}, manifest_extra={}
    )
    new_unit = _write_unit(store, "daily", "daily_20240103", _daily_rows("20240103"))
    successor = store.build_snapshot(
        name="s2",
        successful_units={"daily": [new_unit]},
        manifest_extra={},
        base_snapshot=base,
    )
    frame = _dataset_frame(successor, "daily")
    assert set(frame["trade_date"].dt.strftime("%Y%m%d")) == {"20240103"}


def test_overlapping_added_unit_does_not_duplicate_rows(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    units_a = [
        _write_unit(store, "daily", "daily_20240102", _daily_rows("20240102")),
    ]
    base = store.build_snapshot(name="s1", successful_units={"daily": units_a}, manifest_extra={})
    overlap = _write_unit(
        store,
        "daily",
        "daily_20240102-re",
        _daily_rows("20240102") + _daily_rows("20240103"),
    )
    units_b = units_a + [overlap]
    full = store.build_snapshot(
        name="s2-full", successful_units={"daily": units_b}, manifest_extra={}
    )
    incremental = store.build_snapshot(
        name="s2-inc",
        successful_units={"daily": units_b},
        manifest_extra={},
        base_snapshot=base,
    )
    pd.testing.assert_frame_equal(
        _dataset_frame(full, "daily"), _dataset_frame(incremental, "daily")
    )
    frame = _dataset_frame(incremental, "daily")
    # Full and incremental builds collapse the same provider row even when
    # separate fetches carry different ingestion timestamps.
    assert len(frame) == 4
    assert frame.groupby(["ts_code", "trade_date"], observed=True).size().max() == 1
    entry = _manifest_entry(incremental, "daily")
    assert entry["rows"] == 4
    assert entry["source_rows"] == 6


def test_snapshot_resolves_metadata_drift_and_quarantines_unsafe_keys(
    tmp_path: Path,
) -> None:
    store = ParquetStore(tmp_path / "data")
    ccass_rows = [
        {
            "trade_date": "20250930",
            "ts_code": "00019.HK",
            "name": "太古股份公司A",
            "shareholding": "312939737",
            "hold_nums": "376",
            "hold_ratio": "40.16",
        },
        {
            "trade_date": "20250930",
            "ts_code": "00019.HK",
            "name": "太古股份公司Ａ",
            "shareholding": "312939737",
            "hold_nums": "376",
            "hold_ratio": "40.16",
        },
        {
            "trade_date": "20250930",
            "ts_code": "300300.SZ",
            "name": "海峡创新",
            "shareholding": "354",
            "hold_nums": "7",
            "hold_ratio": "0.00",
        },
        {
            "trade_date": "20250930",
            "ts_code": "300300.SZ",
            "name": "食品饮料ETF天弘",
            "shareholding": "225631",
            "hold_nums": "4",
            "hold_ratio": "0.00",
        },
        {
            "trade_date": "20250930",
            "ts_code": "00001.HK",
            "name": "长和",
            "shareholding": "100",
            "hold_nums": "2",
            "hold_ratio": "0.01",
        },
    ]
    detail_rows = [
        {
            "trade_date": "20250808",
            "ts_code": "00001.HK",
            "name": "长和",
            "col_participant_id": "B01231",
            "col_participant_name": "腾达证券有限公司",
            "col_shareholding": "2368",
            "col_shareholding_percent": "0.00",
        },
        {
            "trade_date": "20250808",
            "ts_code": "00001.HK",
            "name": "長和",
            "col_participant_id": "B01231",
            "col_participant_name": "赢家国际证券有限公司",
            "col_shareholding": "2368",
            "col_shareholding_percent": "0.00",
        },
        {
            "trade_date": "20250808",
            "ts_code": "00019.HK",
            "name": "Swire Pacific",
            "col_participant_id": "B09999",
            "col_participant_name": "Broker C",
            "col_shareholding": "100",
            "col_shareholding_percent": "0.01",
        },
        {
            "trade_date": "20250808",
            "ts_code": "00019.HK",
            "name": "Swire Pacific",
            "col_participant_id": "B09999",
            "col_participant_name": "Broker C",
            "col_shareholding": "200",
            "col_shareholding_percent": "0.02",
        },
    ]
    share_float_rows = [
        {
            "ts_code": "000425.SZ",
            "ann_date": None,
            "float_date": "20190412",
            "float_share": 88536.0,
            "float_ratio": 0.0011,
            "holder_name": "宋希谦",
            "share_type": "股权分置限售股份",
        },
        {
            "ts_code": "000425.SZ",
            "ann_date": None,
            "float_date": "20190412",
            "float_share": 88536.0,
            "float_ratio": 11.302,
            "holder_name": "宋希谦",
            "share_type": "股权分置限售股份",
        },
    ]
    irm_rows = [
        {
            "trade_date": "20250930",
            "ts_code": "000001.SZ",
            "name": "Ping An Bank",
            "q": "What is the revenue outlook?",
            "a": "Revenue should grow.",
            "pub_time": "20250930 10:00:00",
        },
        {
            "trade_date": "20250930",
            "ts_code": "000001.SZ",
            "name": "Ping An Bank",
            "q": "What is the revenue outlook?",
            "a": "Revenue should decline.",
            "pub_time": "20250930 10:05:00",
        },
        {
            "trade_date": "20250930",
            "ts_code": "000002.SZ",
            "name": "Vanke",
            "q": "Has the annual report been published?",
            "a": "Yes.",
            "pub_time": "20250930 11:00:00",
        },
    ]
    units = {
        "ccass_hold": [_write_unit(store, "ccass_hold", "ccass-hold", ccass_rows)],
        "ccass_hold_detail": [_write_unit(store, "ccass_hold_detail", "ccass-detail", detail_rows)],
        "share_float": [_write_unit(store, "share_float", "share-float", share_float_rows)],
        "irm_qa_sz": [_write_unit(store, "irm_qa_sz", "irm-qa", irm_rows)],
    }

    snapshot = store.build_snapshot(
        name="resolved-provider-conflicts",
        successful_units=units,
        manifest_extra={},
    )

    ccass = _dataset_frame(snapshot, "ccass_hold")
    assert set(ccass["ts_code"]) == {"00001.HK", "00019.HK"}
    assert len(ccass) == 2
    detail = _dataset_frame(snapshot, "ccass_hold_detail")
    assert len(detail) == 1
    assert detail.iloc[0]["col_participant_id"] == "B01231"
    assert detail.iloc[0]["col_shareholding"] == "2368"
    detail_manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))[
        "datasets"
    ]["ccass_hold_detail"]
    assert detail_manifest["date_min"] == "2025-08-08"
    assert detail_manifest["date_max"] == "2025-08-08"
    share_float = _dataset_frame(snapshot, "share_float")
    assert len(share_float) == 1
    assert pd.isna(share_float.iloc[0]["float_ratio"])
    assert share_float.iloc[0]["float_share"] == pytest.approx(88536.0)
    irm = _dataset_frame(snapshot, "irm_qa_sz")
    assert len(irm) == 1
    assert irm.iloc[0]["ts_code"] == "000002.SZ"
    assert irm.iloc[0]["a"] == "Yes."


def test_legacy_news_global_dedup_is_preserved_with_base(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    legacy = {
        "datetime": "2024-01-02 08:00:00",
        "title": "旧闻",
        "content": "同一内容",
        "source": None,
    }
    tagged = {
        "datetime": "2024-01-02 08:00:00",
        "title": "旧闻",
        "content": "同一内容",
        "source": "财联社",
    }
    units_a = [_write_unit(store, "news", "news_1", [legacy])]
    base = store.build_snapshot(name="s1", successful_units={"news": units_a}, manifest_extra={})
    units_b = units_a + [_write_unit(store, "news", "news_2", [tagged])]
    successor = store.build_snapshot(
        name="s2",
        successful_units={"news": units_b},
        manifest_extra={},
        base_snapshot=base,
    )
    frame = _dataset_frame(successor, "news")
    assert len(frame) == 1
    assert frame.iloc[0]["source"] == "财联社"


def _us_adj_rows(day: str, close: float, *, pct_change: float | None = 0.5) -> list[dict]:
    return [
        {
            "ts_code": "NVDA",
            "trade_date": day,
            "close": close,
            "pct_change": pct_change,
        }
    ]


def test_latest_generation_snapshot_keeps_the_newest_revision(tmp_path: Path) -> None:
    # Adjusted series republish a business key after corporate actions; the
    # snapshot must keep the newest ingestion generation, never a random one.
    old_store = ParquetStore(
        tmp_path / "data",
        clock=lambda: datetime(2026, 8, 20, 8, 0, tzinfo=UTC),
    )
    old_unit = _write_unit(
        old_store, "us_daily_adj", "gen-old", _us_adj_rows("20260819", 100.0)
    )
    new_store = ParquetStore(
        tmp_path / "data",
        clock=lambda: datetime(2026, 8, 21, 8, 0, tzinfo=UTC),
    )
    new_unit = _write_unit(
        new_store, "us_daily_adj", "gen-new", _us_adj_rows("20260819", 105.0)
    )

    snapshot = new_store.build_snapshot(
        name="latest-gen",
        successful_units={"us_daily_adj": [old_unit, new_unit]},
        manifest_extra={},
    )

    frame = _dataset_frame(snapshot, "us_daily_adj")
    assert len(frame) == 1
    assert frame["close"].tolist() == [105.0]
    assert frame["ingested_at"].tolist() == [pd.Timestamp("2026-08-21 08:00:00+00:00")]


def test_latest_generation_tiebreak_prefers_the_more_complete_row(
    tmp_path: Path,
) -> None:
    # Same generation, same key: the provider double-writes a row with and
    # without derived fields; the filled row wins deterministically.
    store = ParquetStore(
        tmp_path / "data",
        clock=lambda: datetime(2026, 8, 22, 8, 0, tzinfo=UTC),
    )
    sparse = _write_unit(
        store,
        "us_daily_adj",
        "same-gen-a",
        _us_adj_rows("20260820", 21.34, pct_change=None),
    )
    filled = _write_unit(
        store,
        "us_daily_adj",
        "same-gen-b",
        _us_adj_rows("20260820", 21.34, pct_change=3.39),
    )

    snapshot = store.build_snapshot(
        name="same-gen",
        successful_units={"us_daily_adj": [sparse, filled]},
        manifest_extra={},
    )

    frame = _dataset_frame(snapshot, "us_daily_adj")
    assert len(frame) == 1
    assert frame["pct_change"].tolist() == [3.39]


def _member_rows(
    ts_code: str, in_date: str, out_date: str | None, l1: str = "801010.SI"
) -> list[dict]:
    return [
        {
            "ts_code": ts_code,
            "in_date": in_date,
            "out_date": out_date,
            "l1_code": l1,
            "l2_code": "801016.SI",
            "l3_code": "850111.SI",
            "is_new": "N",
        }
    ]


def test_index_member_all_unions_cohorts_with_newest_revision_winning(
    tmp_path: Path,
) -> None:
    # The provider prunes long-delisted members from newer weekly cohorts, so
    # intervals only older cohorts still serve must survive, while an interval
    # revised by a newer cohort (out_date closed) keeps the newer version.
    old_store = ParquetStore(
        tmp_path / "data",
        clock=lambda: datetime(2026, 8, 25, 8, 0, tzinfo=UTC),
    )
    old_units = [
        _write_unit(
            old_store,
            "index_member_all",
            "w1-revised",
            _member_rows("000918.SZ", "19990720", None),
        ),
        _write_unit(
            old_store,
            "index_member_all",
            "w1-history",
            _member_rows("600313.SH", "20010109", "20110107"),
        ),
    ]
    new_store = ParquetStore(
        tmp_path / "data",
        clock=lambda: datetime(2026, 8, 31, 8, 0, tzinfo=UTC),
    )
    new_units = [
        _write_unit(
            new_store,
            "index_member_all",
            "w2-revised",
            _member_rows("000918.SZ", "19990720", "20100115"),
        ),
    ]

    snapshot = new_store.build_snapshot(
        name="member-union",
        successful_units={"index_member_all": [*old_units, *new_units]},
        manifest_extra={},
    )

    frame = _dataset_frame(snapshot, "index_member_all")
    assert len(frame) == 2
    by_code = frame.set_index("ts_code")
    # The revised interval keeps the newer cohort's closed out_date.
    assert str(by_code.loc["000918.SZ", "out_date"].date()) == "2010-01-15"
    assert by_code.loc["000918.SZ", "ingested_at"] == pd.Timestamp(
        "2026-08-31 08:00:00+00:00"
    )
    # The pruned interval survives from the older cohort.
    assert str(by_code.loc["600313.SH", "out_date"].date()) == "2011-01-07"
