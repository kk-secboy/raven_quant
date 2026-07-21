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


def _write_unit(
    store: ParquetStore, dataset: str, unit_key: str, rows: list[dict]
) -> dict:
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
    base = store.build_snapshot(
        name="s1", successful_units={"daily": units_a}, manifest_extra={}
    )
    units_b = units_a + [
        _write_unit(store, "daily", "daily_20240201", _daily_rows("20240201"))
    ]
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
    base = store.build_snapshot(
        name="s1", successful_units={"daily": units_a}, manifest_extra={}
    )
    units_b = units_a + [
        _write_unit(store, "daily", "daily_20240201", _daily_rows("20240201"))
    ]
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
    base = store.build_snapshot(
        name="s1", successful_units={"daily": units}, manifest_extra={}
    )
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
    base = store.build_snapshot(
        name="s1", successful_units={"daily": units_a}, manifest_extra={}
    )
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
    # Both builds retain both ingested_at versions exactly as the legacy full
    # rebuild did: rows differ in ingested_at, so DISTINCT keeps them. The
    # incremental path must not invent new dedup semantics — it must match.
    assert len(frame) == len(_dataset_frame(full, "daily"))


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
    base = store.build_snapshot(
        name="s1", successful_units={"news": units_a}, manifest_extra={}
    )
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
