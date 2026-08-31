from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quant_data.catalog import ALL_DEFINITIONS
from quant_data.models import ProviderResult
from quant_data.release_window import (
    select_release_window_units,
    summarize_release_plan,
)
from quant_data.storage import ParquetStore
from quant_data.verify import quality_gate_payload, verify_downloads

pytestmark = pytest.mark.no_database


def _row(
    unit_key: str,
    dataset: str,
    scope: dict,
    *,
    status: str = "succeeded",
    params: dict | None = None,
    output_path: str | None = None,
    sha256: str | None = None,
    row_count: int | None = 1,
    allow_empty: bool = False,
    api_name: str | None = None,
    fields: tuple[str, ...] = (),
) -> dict:
    return {
        "unit_key": unit_key,
        "dataset": dataset,
        "api_name": api_name or dataset,
        "scope_json": scope,
        "params_json": params or {},
        "fields_json": list(fields),
        "allow_empty": allow_empty,
        "status": status,
        "output_path": output_path,
        "sha256": sha256,
        "row_count": row_count,
    }


def test_selector_keeps_financial_period_carry_in_for_announcement_clipping() -> None:
    rows = [
        _row("old", "daily", {"trade_date": "20231229"}),
        _row("inside", "daily", {"trade_date": "20240102"}),
        _row("future", "daily", {"trade_date": "20250102"}, status="pending"),
        _row("old-report", "income", {"end_date": "20230930"}),
    ]

    selected = select_release_window_units(
        rows,
        snapshot_start=date(2024, 1, 1),
        snapshot_end=date(2024, 12, 31),
        datasets={"daily", "income"},
    )

    assert [row["unit_key"] for row in selected.rows] == ["inside", "old-report"]


def test_selector_handles_natural_ranges_and_adaptive_partitions() -> None:
    rows = [
        _row("overlap", "index_daily", {"start": "20231201", "end": "20240201"}),
        _row("before", "index_daily", {"start": "20220101", "end": "20221231"}),
        _row(
            "adaptive",
            "share_float",
            {
                "partition_axis": "date",
                "partition_start": "2024-06-01",
                "partition_end": "2024-06-30",
            },
            status="pending",
        ),
        _row(
            "adaptive-future",
            "share_float",
            {
                "partition_axis": "date",
                "partition_start": "2025-01-01",
                "partition_end": "2025-01-31",
            },
            status="pending",
        ),
    ]

    selected = select_release_window_units(
        rows,
        snapshot_start=date(2024, 1, 1),
        snapshot_end=date(2024, 12, 31),
        datasets={"index_daily", "share_float"},
    )

    assert [row["unit_key"] for row in selected.rows] == ["overlap", "adaptive"]
    summaries = {item["dataset"]: item for item in summarize_release_plan(selected.rows)}
    assert summaries["index_daily"]["planned"] == 1
    assert summaries["index_daily"]["succeeded"] == 1
    assert summaries["share_float"]["planned"] == 1
    assert summaries["share_float"]["pending"] == 1


def test_selector_uses_pending_latest_reference_generation_and_successor_pages() -> None:
    identity = {"market": "SSE", "page_group": "index:SSE", "offset": 0}
    rows = [
        _row(
            "reference-old",
            "index_basic",
            {**identity, "reference_refresh_bucket": "2024-01-01"},
        ),
        _row(
            "reference-current",
            "index_basic",
            {**identity, "reference_refresh_bucket": "2024-06-01"},
            status="pending",
        ),
        _row(
            "reference-future",
            "index_basic",
            {**identity, "reference_refresh_bucket": "2025-01-01"},
            status="pending",
        ),
        _row(
            "capped-parent",
            "share_float",
            {
                "page_group": "share:2024",
                "partition_start": "2024-01-01",
                "partition_end": "2024-12-31",
            },
        ),
        _row(
            "replacement-child",
            "share_float",
            {
                "page_group": "share:2024:child",
                "supersedes_page_group": "share:2024",
                "partition_start": "2024-01-01",
                "partition_end": "2024-12-31",
            },
            status="pending",
        ),
    ]

    selected = select_release_window_units(
        rows,
        snapshot_start=date(2024, 1, 1),
        snapshot_end=date(2024, 12, 31),
        datasets={"index_basic", "share_float"},
    )

    assert [row["unit_key"] for row in selected.rows] == [
        "reference-current",
        "replacement-child",
    ]
    report = selected.report()
    assert len(report["scope_sha256"]) == 64
    assert len(report["selected_unit_set_sha256"]) == 64
    assert [item["unit_key"] for item in report["selected_unit_identities"]] == [
        "reference-current",
        "replacement-child",
    ]


def test_selector_keeps_unfinished_canonical_fina_audit_instead_of_legacy() -> None:
    period = "20231231"
    common_scope = {
        "period": period,
        "page_group": f"fina_audit:{period}",
        "page_size": 1_000,
    }
    common_params = {"period": period, "limit": 1_000}
    rows = [
        _row(
            "legacy-0",
            "fina_audit",
            {**common_scope, "offset": 0},
            params={**common_params, "offset": 0},
        ),
        _row(
            "legacy-1000",
            "fina_audit",
            {**common_scope, "offset": 1_000},
            params={**common_params, "offset": 1_000},
        ),
        _row(
            "canonical-0",
            "fina_audit",
            {
                **common_scope,
                "offset": 0,
                "expected_date_field": "end_date",
                "expected_date": period,
            },
            status="pending",
            params={**common_params, "offset": 0},
            row_count=None,
        ),
        _row(
            "canonical-1000",
            "fina_audit",
            {
                **common_scope,
                "offset": 1_000,
                "expected_date_field": "end_date",
                "expected_date": period,
            },
            status="failed",
            params={**common_params, "offset": 1_000},
            row_count=None,
        ),
    ]

    selected = select_release_window_units(
        rows,
        snapshot_start=date(2024, 1, 1),
        snapshot_end=date(2024, 12, 31),
        datasets={"fina_audit"},
    )

    assert [row["unit_key"] for row in selected.rows] == [
        "canonical-0",
        "canonical-1000",
    ]
    assert selected.report()["selector_version"] == (
        "release-window-selector-v6-governed-etf-daily-publication"
    )
    assert summarize_release_plan(selected.rows) == [
        {
            "dataset": "fina_audit",
            "planned": 2,
            "succeeded": 0,
            "failed": 1,
            "pending": 1,
            "running": 0,
            "superseded": 0,
            "empty": 0,
            "allowed_empty": 0,
            "unexpected_empty": 0,
            "rows": 0,
        }
    ]


def test_release_plan_hash_binds_profile_datasets_and_unit_content() -> None:
    original = _row(
        "daily-20240102",
        "daily",
        {"trade_date": "20240102"},
        sha256="1" * 64,
    )
    base = select_release_window_units(
        [original],
        snapshot_start=date(2024, 1, 1),
        snapshot_end=date(2024, 12, 31),
        datasets={"daily"},
        profile="core",
    )
    different_profile = select_release_window_units(
        [original],
        snapshot_start=date(2024, 1, 1),
        snapshot_end=date(2024, 12, 31),
        datasets={"daily"},
        profile="research",
    )
    changed_content = select_release_window_units(
        [{**original, "sha256": "2" * 64}],
        snapshot_start=date(2024, 1, 1),
        snapshot_end=date(2024, 12, 31),
        datasets={"daily"},
        profile="core",
    )

    assert base.plan_scope_sha256 != different_profile.plan_scope_sha256
    assert base.selected_unit_set_sha256 != changed_content.selected_unit_set_sha256
    gate = quality_gate_payload(
        {
            "ok": True,
            "checked_at": "2026-08-13T00:00:00+00:00",
            "errors": [],
            "release_window": base.report(),
        }
    )
    assert gate["plan_scope_sha256"] == base.plan_scope_sha256
    assert gate["release_window"]["profile"] == "core"
    assert gate["release_window"]["selected_unit_identities"] == [
        dict(base.unit_identities[0])
    ]


def test_verify_checks_only_units_selected_for_release_window(tmp_path: Path) -> None:
    payload = b"inside-release-window"
    relative = "units/optional_series/inside.parquet"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    rows = [
        _row(
            "inside",
            "optional_series",
            {"trade_date": "20240102"},
            output_path=relative,
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
        _row(
            "future-pending",
            "optional_series",
            {"trade_date": "20250102"},
            status="pending",
            row_count=None,
        ),
        _row(
            "old-missing-file",
            "optional_series",
            {"trade_date": "20230102"},
            output_path="units/optional_series/missing.parquet",
            sha256="0" * 64,
        ),
    ]

    class Checkpoint:
        def active_units(self, datasets=None):
            return [
                row for row in rows
                if datasets is None or row["dataset"] in datasets
            ]

    result = verify_downloads(
        Checkpoint(),  # type: ignore[arg-type]
        tmp_path,
        snapshot_start=date(2024, 1, 1),
        snapshot_end=date(2024, 12, 31),
        dataset_filter={"optional_series"},
        required_datasets=set(),
    )

    assert result["errors"] == []
    assert result["ok"] is True
    assert result["datasets"][0]["planned"] == 1
    assert result["datasets"][0]["succeeded"] == 1
    assert result["release_window"]["selected_unit_count"] == 1
    assert result["release_window"]["selected_unit_identities"][0]["unit_key"] == "inside"


def test_optional_profile_dataset_without_a_plan_does_not_block_required_core(
    tmp_path: Path,
) -> None:
    values = {
        "ts_code": "000001.SZ",
        "symbol": "000001",
        "name": "平安银行",
        "list_status": "L",
        "list_date": "19910403",
    }
    written = ParquetStore(tmp_path).write_unit(
        "stock_basic",
        "stock-basic-inside",
        ProviderResult(
            api_name="stock_basic",
            columns=list(values),
            rows=[values],
            raw_body=b"{}",
        ),
    )
    rows = [
        _row(
            "stock-basic-inside",
            "stock_basic",
            {"list_status": "L"},
            output_path=written.output_path,
            sha256=written.sha256,
            row_count=written.row_count,
        )
    ]

    class Checkpoint:
        def active_units(self, datasets=None):
            return [
                row
                for row in rows
                if datasets is None or row["dataset"] in datasets
            ]

    result = verify_downloads(
        Checkpoint(),  # type: ignore[arg-type]
        tmp_path,
        snapshot_start=date(2024, 1, 1),
        snapshot_end=date(2024, 12, 31),
        dataset_filter={"stock_basic", "optional_never_planned"},
        required_datasets={"stock_basic"},
        profile="research",
    )

    assert result["errors"] == []
    assert result["ok"] is True
    assert result["plan_gate"]["missing_planned_datasets"] == []
    assert result["release_window"]["requested_datasets"] == [
        "optional_never_planned",
        "stock_basic",
    ]


def test_selector_keeps_baostock_legacy_rows_but_retires_pre2016_primary_units() -> None:
    rows = [
        _row(
            "primary-old",
            "daily",
            {"trade_date": "20151231"},
            params={"trade_date": "20151231"},
        ),
        _row(
            "legacy-old",
            "daily",
            {
                "source": "baostock-0.9.3",
                "code": "sh.600000",
                "start": "2008-01-01",
                "end": "2015-12-31",
                "contract": "daily",
            },
            api_name="baostock_daily",
        ),
        _row(
            "primary-current",
            "daily",
            {"trade_date": "20160104"},
            params={"trade_date": "20160104"},
        ),
    ]

    selected = select_release_window_units(
        rows,
        snapshot_start=date(2008, 1, 1),
        snapshot_end=date(2016, 1, 4),
        datasets={"daily"},
    )

    assert [row["unit_key"] for row in selected.rows] == [
        "legacy-old",
        "primary-current",
    ]
    assert selected.report()["selector_version"] == (
        "release-window-selector-v6-governed-etf-daily-publication"
    )


def test_selector_keeps_field_contract_overlap_visible_and_fail_closed() -> None:
    current_fields = ALL_DEFINITIONS["fund_daily"].fields
    rows = [
        _row(
            "legacy-only",
            "fund_daily",
            {"trade_date": "20260821"},
            params={"trade_date": "20260821"},
        ),
        _row(
            "legacy-overlap",
            "fund_daily",
            {"trade_date": "20260824"},
            params={"trade_date": "20260824"},
        ),
        _row(
            "current-overlap",
            "fund_daily",
            {"trade_date": "20260824"},
            params={"trade_date": "20260824"},
            fields=current_fields,
            status="pending",
            row_count=None,
        ),
    ]

    selected = select_release_window_units(
        rows,
        snapshot_start=date(2026, 8, 21),
        snapshot_end=date(2026, 8, 24),
        datasets={"fund_daily"},
    )

    assert [row["unit_key"] for row in selected.rows] == [
        "current-overlap",
        "legacy-only",
        "legacy-overlap",
    ]
    assert summarize_release_plan(selected.rows) == [
        {
            "dataset": "fund_daily",
            "planned": 3,
            "succeeded": 2,
            "failed": 0,
            "pending": 1,
            "running": 0,
            "superseded": 0,
            "empty": 0,
            "allowed_empty": 0,
            "unexpected_empty": 0,
            "rows": 2,
        }
    ]


@pytest.mark.parametrize(
    ("dataset", "values"),
    [
        (
            "fund_daily",
            {
                "ts_code": "510300.SH",
                "trade_date": "20260824",
                "open": 4.0,
                "high": 4.1,
                "low": 3.9,
                "close": 4.05,
                "pre_close": 4.0,
                "change": 0.05,
                "pct_chg": 1.25,
                "vol": 100.0,
                "amount": 405.0,
            },
        ),
        (
            "fund_adj",
            {
                "ts_code": "510300.SH",
                "trade_date": "20260824",
                "adj_factor": 1.0,
            },
        ),
    ],
)
def test_verify_and_snapshot_deduplicate_identical_etf_contract_overlap(
    tmp_path: Path, dataset: str, values: dict[str, object]
) -> None:
    current_fields = ALL_DEFINITIONS[dataset].fields
    storage = ParquetStore(tmp_path)
    rows = []
    for unit_key, fields in (
        ("legacy-overlap", ()),
        ("current-overlap", current_fields),
    ):
        written = storage.write_unit(
            dataset,
            unit_key,
            ProviderResult(
                api_name=dataset,
                columns=list(values),
                rows=[values],
                raw_body=b"{}",
            ),
        )
        rows.append(
            _row(
                unit_key,
                dataset,
                {"trade_date": "20260824"},
                params={"trade_date": "20260824"},
                fields=fields,
                output_path=written.output_path,
                sha256=written.sha256,
                row_count=written.row_count,
            )
        )

    class Checkpoint:
        def active_units(self, datasets=None):
            return [
                row
                for row in rows
                if datasets is None or row["dataset"] in datasets
            ]

    report = verify_downloads(
        Checkpoint(),  # type: ignore[arg-type]
        tmp_path,
        snapshot_start=date(2026, 8, 24),
        snapshot_end=date(2026, 8, 24),
        dataset_filter={dataset},
        required_datasets=set(),
    )

    assert report["ok"] is True
    assert report["duplicate_checks"] == {dataset: 1}
    assert report["conflicting_duplicate_checks"] == {dataset: 0}
    assert report["release_window"]["selected_unit_count"] == 2
    assert any(
        f"{dataset}: 1 exact duplicate primary-key rows" in warning
        for warning in report["warnings"]
    )
    assert (tmp_path / rows[0]["output_path"]).is_file()
    snapshot = storage.build_snapshot(
        name=f"{dataset}-field-contract-overlap",
        successful_units={dataset: rows},
        manifest_extra={"profile": "test"},
    )
    published = pd.concat(
        [
            pd.read_parquet(path)
            for path in (snapshot / "parquet" / dataset).rglob("*.parquet")
        ]
    )
    assert len(published) == 1
    assert published.iloc[0]["ts_code"] == "510300.SH"


def test_verify_conflicting_etf_field_contract_overlap_blocks(tmp_path: Path) -> None:
    fields = ALL_DEFINITIONS["fund_daily"].fields
    base = {
        "ts_code": "510300.SH",
        "trade_date": "20260824",
        "open": 4.0,
        "high": 4.1,
        "low": 3.9,
        "close": 4.05,
        "pre_close": 4.0,
        "change": 0.05,
        "pct_chg": 1.25,
        "vol": 100.0,
        "amount": 405.0,
    }
    storage = ParquetStore(tmp_path)
    rows = []
    for unit_key, contract, close in (
        ("legacy-conflict", (), 4.05),
        ("current-conflict", fields, 4.06),
    ):
        values = {**base, "close": close}
        written = storage.write_unit(
            "fund_daily",
            unit_key,
            ProviderResult(
                api_name="fund_daily",
                columns=list(values),
                rows=[values],
                raw_body=b"{}",
            ),
        )
        rows.append(
            _row(
                unit_key,
                "fund_daily",
                {"trade_date": "20260824"},
                params={"trade_date": "20260824"},
                fields=contract,
                output_path=written.output_path,
                sha256=written.sha256,
                row_count=written.row_count,
            )
        )

    class Checkpoint:
        def active_units(self, datasets=None):
            return [
                row
                for row in rows
                if datasets is None or row["dataset"] in datasets
            ]

    report = verify_downloads(
        Checkpoint(),  # type: ignore[arg-type]
        tmp_path,
        snapshot_start=date(2026, 8, 24),
        snapshot_end=date(2026, 8, 24),
        dataset_filter={"fund_daily"},
        required_datasets=set(),
    )

    assert report["ok"] is False
    assert report["duplicate_checks"] == {"fund_daily": 1}
    assert report["conflicting_duplicate_checks"] == {"fund_daily": 1}
    assert any(
        "fund_daily: 1 conflicting business keys" in error
        for error in report["errors"]
    )


def test_verify_does_not_count_retired_primary_and_baostock_as_duplicate(
    tmp_path: Path,
) -> None:
    storage = ParquetStore(tmp_path)
    provider_rows = [
        (
            "primary-old",
            "daily",
            {"trade_date": "20151231"},
            "daily",
            {
                "ts_code": "600000.SH",
                "trade_date": "20151231",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "pre_close": 10.0,
                "change": 0.5,
                "pct_chg": 5.0,
                "vol": 100.0,
                "amount": 105.0,
            },
        ),
        (
            "legacy-old",
            "daily",
            {
                "source": "baostock-0.9.3",
                "code": "sh.600000",
                "start": "2008-01-01",
                "end": "2015-12-31",
                "contract": "daily",
            },
            "baostock_daily",
            {
                "ts_code": "600000.SH",
                "trade_date": "20151231",
                "open": 10.1,
                "high": 11.1,
                "low": 9.1,
                "close": 10.6,
                "pre_close": 10.1,
                "change": 0.5,
                "pct_chg": 4.95,
                "vol": 101.0,
                "amount": 107.06,
            },
        ),
        (
            "primary-current",
            "daily",
            {"trade_date": "20160104"},
            "daily",
            {
                "ts_code": "600000.SH",
                "trade_date": "20160104",
                "open": 10.6,
                "high": 11.0,
                "low": 10.0,
                "close": 10.2,
                "pre_close": 10.6,
                "change": -0.4,
                "pct_chg": -3.77,
                "vol": 102.0,
                "amount": 104.04,
            },
        ),
    ]
    rows = []
    for unit_key, dataset, scope, api_name, values in provider_rows:
        written = storage.write_unit(
            dataset,
            unit_key,
            ProviderResult(
                api_name=api_name,
                columns=list(values),
                rows=[values],
                raw_body=b"{}",
            ),
        )
        rows.append(
            _row(
                unit_key,
                dataset,
                scope,
                params={"trade_date": scope.get("trade_date")}
                if scope.get("trade_date")
                else {},
                api_name=api_name,
                output_path=written.output_path,
                sha256=written.sha256,
                row_count=written.row_count,
            )
        )

    class Checkpoint:
        def active_units(self, datasets=None):
            return [
                row
                for row in rows
                if datasets is None or row["dataset"] in datasets
            ]

    report = verify_downloads(
        Checkpoint(),  # type: ignore[arg-type]
        tmp_path,
        snapshot_start=date(2008, 1, 1),
        snapshot_end=date(2016, 1, 4),
        dataset_filter={"daily"},
        required_datasets=set(),
    )

    assert report["duplicate_checks"] == {"daily": 0}
    assert not any("duplicate primary-key" in item for item in report["errors"])
    assert report["release_window"]["selected_unit_count"] == 2
