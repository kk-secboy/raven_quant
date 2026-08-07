from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_data.models import ProviderResult
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


def test_unchanged_dataset_is_fully_linked(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    units = [
        _write_unit(store, "daily", "daily_20240102", _daily_rows("20240102")),
    ]
    base = store.build_snapshot(name="s1", successful_units={"daily": units}, manifest_extra={})
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
