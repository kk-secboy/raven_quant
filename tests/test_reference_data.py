import json
from datetime import date

import pytest

from quant_data.models import FetchSpec, ProviderResult
from quant_data.reference_data import (
    AUDITED_REFERENCE_DATASETS,
    REFERENCE_REFRESH_POLICIES,
    apply_reference_refresh,
    reference_refresh_bucket,
    select_current_reference_units,
)
from quant_data.storage import ParquetStore

pytestmark = pytest.mark.no_database


def test_audited_reference_refresh_inventory_is_exact() -> None:
    assert len(AUDITED_REFERENCE_DATASETS) == 29
    assert AUDITED_REFERENCE_DATASETS == {
        "cb_basic",
        "cb_price_chg",
        "cb_rate",
        "cn_cpi",
        "cn_gdp",
        "cn_m",
        "cn_pmi",
        "cn_ppi",
        "cn_schedule",
        "etf_basic",
        "etf_index",
        "fund_basic",
        "fut_basic",
        "fut_trade_cal",
        "fx_obasic",
        "hk_basic",
        "hk_tradecal",
        "index_basic",
        "index_classify",
        "new_share",
        "opt_basic",
        "sf_month",
        "shibor",
        "shibor_lpr",
        "stk_surv",
        "stock_basic",
        "us_basic",
        "us_tradecal",
        "us_tycr",
    }
    assert AUDITED_REFERENCE_DATASETS < set(REFERENCE_REFRESH_POLICIES)


def test_refresh_buckets_follow_reviewed_cadence() -> None:
    as_of = date(2026, 7, 15)
    assert reference_refresh_bucket("stock_basic", as_of) == "2026-07-15"
    assert reference_refresh_bucket("index_basic", as_of) == "2026-07-13"
    assert reference_refresh_bucket("cn_gdp", as_of) == "2026-07"
    assert reference_refresh_bucket("daily", as_of) is None


def test_refresh_versions_only_unbounded_reference_requests() -> None:
    static = FetchSpec("stock_basic", "stock_basic", {"status": "L"}, {"status": "L"})
    bounded = FetchSpec(
        "shibor",
        "shibor",
        {"start_date": "20260701", "end_date": "20260715"},
        {"start_date": "20260701", "end_date": "20260715"},
    )
    refreshed = apply_reference_refresh([static, bounded], as_of=date(2026, 7, 15))
    assert refreshed[0].scope["reference_refresh_bucket"] == "2026-07-15"
    assert "reference_refresh_bucket" not in refreshed[0].params
    assert refreshed[1] == bounded


def test_snapshot_selection_replaces_legacy_and_old_reference_generations() -> None:
    def row(unit_key: str, bucket: str | None) -> dict:
        scope = {"status": "L"}
        if bucket:
            scope.update(
                {
                    "reference_refresh_bucket": bucket,
                    "reference_refresh_cadence": "daily",
                }
            )
        return {"dataset": "stock_basic", "unit_key": unit_key, "scope_json": scope}

    selected = select_current_reference_units(
        [
            row("legacy", None),
            row("day-1", "2026-07-14"),
            row("day-2", "2026-07-15"),
            row("future", "2026-07-16"),
        ],
        snapshot_end=date(2026, 7, 15),
    )
    assert [item["unit_key"] for item in selected] == ["day-2"]


def test_snapshot_selection_keeps_all_pages_in_latest_generation() -> None:
    rows = []
    for bucket in ("2026-07-07", "2026-07-14"):
        for offset in (0, 1000):
            rows.append(
                {
                    "dataset": "index_basic",
                    "unit_key": f"{bucket}:{offset}",
                    "scope_json": {
                        "market": "CSI",
                        "offset": offset,
                        "reference_refresh_bucket": bucket,
                        "reference_refresh_cadence": "weekly",
                    },
                }
            )
    selected = select_current_reference_units(rows, snapshot_end=date(2026, 7, 15))
    assert {item["unit_key"] for item in selected} == {
        "2026-07-14:0",
        "2026-07-14:1000",
    }


def test_snapshot_selection_replaces_a_provider_capped_page_group() -> None:
    parent_group = "share_float:20240101:20240131"
    rows = [
        {
            "dataset": "share_float",
            "unit_key": "monthly-0",
            "scope_json": {"page_group": parent_group, "offset": 0},
        },
        {
            "dataset": "share_float",
            "unit_key": "monthly-100000",
            "scope_json": {"page_group": parent_group, "offset": 100_000},
        },
        {
            "dataset": "share_float",
            "unit_key": "daily-20240101",
            "scope_json": {
                "page_group": f"{parent_group}:daily:20240101",
                "offset": 0,
                "supersedes_page_group": parent_group,
            },
        },
    ]

    selected = select_current_reference_units(rows, snapshot_end=date(2026, 7, 15))

    assert [item["unit_key"] for item in selected] == ["daily-20240101"]


def test_snapshot_manifest_records_historical_bounds_and_reference_version(tmp_path) -> None:
    store = ParquetStore(tmp_path)
    cpi_result = store.write_unit(
        "cn_cpi",
        "cpi-unit",
        ProviderResult(
            api_name="cn_cpi",
            columns=["month", "nt_val"],
            rows=[{"month": "202312", "nt_val": 100.1}],
            raw_body=b"{}",
        ),
    )
    master_result = store.write_unit(
        "stock_basic",
        "master-unit",
        ProviderResult(
            api_name="stock_basic",
            columns=["ts_code", "name"],
            rows=[{"ts_code": "000001.SZ", "name": "Ping An"}],
            raw_body=b"{}",
        ),
    )
    snapshot = store.build_snapshot(
        name="coverage-audit",
        successful_units={
            "cn_cpi": [
                {
                    "dataset": "cn_cpi",
                    "unit_key": "cpi-unit",
                    "output_path": cpi_result.output_path,
                    "row_count": 1,
                    "sha256": cpi_result.sha256,
                    "scope_json": {"start_m": "202312", "end_m": "202312"},
                }
            ],
            "stock_basic": [
                {
                    "dataset": "stock_basic",
                    "unit_key": "master-unit",
                    "output_path": master_result.output_path,
                    "row_count": 1,
                    "sha256": master_result.sha256,
                    "scope_json": {
                        "list_status": "L",
                        "reference_refresh_bucket": "2026-07-15",
                        "reference_refresh_cadence": "daily",
                    },
                }
            ],
        },
        manifest_extra={"profile": "full"},
    )
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["datasets"]["cn_cpi"]["date_min"] == "2023-12-01"
    assert manifest["datasets"]["cn_cpi"]["date_max"] == "2023-12-01"
    assert manifest["coverage_audit"]["historical_before_2024_count"] == 1
    assert manifest["datasets"]["stock_basic"]["reference_refresh"] == {
        "cadence": "daily",
        "selected_buckets": ["2026-07-15"],
        "retention": "append_only_units_and_immutable_snapshots",
    }
