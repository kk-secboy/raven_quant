from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.market_overview import MarketOverviewService

pytestmark = pytest.mark.no_database


def _write_dataset(snapshot: Path, name: str, rows: list[dict]) -> dict:
    target = snapshot / "parquet" / name / "data.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target, index=False)
    return {
        "rows": len(rows),
        "files": [
            {"path": target.relative_to(snapshot).as_posix(), "bytes": target.stat().st_size}
        ],
    }


def _build_snapshot(root: Path) -> Path:
    snapshot = root / "snapshots" / "research-20260714"
    datasets = {
        "daily": _write_dataset(
            snapshot,
            "daily",
            [
                {
                    "ts_code": "600001.SH",
                    "trade_date": "20260713",
                    "close": 10.0,
                    "pre_close": 10.0,
                    "pct_chg": 0.0,
                    "amount": 100_000.0,
                },
                {
                    "ts_code": "600002.SH",
                    "trade_date": "20260713",
                    "close": 11.0,
                    "pre_close": 10.0,
                    "pct_chg": 10.0,
                    "amount": 120_000.0,
                },
                {
                    "ts_code": "600003.SH",
                    "trade_date": "20260713",
                    "close": 9.0,
                    "pre_close": 10.0,
                    "pct_chg": -10.0,
                    "amount": 90_000.0,
                },
                {
                    "ts_code": "600001.SH",
                    "trade_date": "20260714",
                    "close": 10.5,
                    "pre_close": 10.0,
                    "pct_chg": 5.0,
                    "amount": 150_000.0,
                },
                {
                    "ts_code": "600002.SH",
                    "trade_date": "20260714",
                    "close": 11.0,
                    "pre_close": 11.0,
                    "pct_chg": 0.0,
                    "amount": 110_000.0,
                },
                {
                    "ts_code": "600003.SH",
                    "trade_date": "20260714",
                    "close": 8.1,
                    "pre_close": 9.0,
                    "pct_chg": -10.0,
                    "amount": 95_000.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260714",
                    "close": 12.0,
                    "pre_close": 11.0,
                    "pct_chg": 9.09,
                    "amount": 200_000.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "20260714",
                    "close": 8.0,
                    "pre_close": 8.2,
                    "pct_chg": -2.44,
                    "amount": 80_000.0,
                },
                {
                    "ts_code": "000003.SZ",
                    "trade_date": "20260714",
                    "close": 6.1,
                    "pre_close": 6.0,
                    "pct_chg": 1.67,
                    "amount": 70_000.0,
                },
            ],
        ),
        "stock_basic": _write_dataset(
            snapshot,
            "stock_basic",
            [
                {"ts_code": "600001.SH", "name": "浦发样本", "industry": "银行"},
                {"ts_code": "600002.SH", "name": "沪市样本二", "industry": "银行"},
                {"ts_code": "600003.SH", "name": "沪市样本三", "industry": "银行"},
                {"ts_code": "000001.SZ", "name": "深市样本一", "industry": "软件服务"},
                {"ts_code": "000002.SZ", "name": "深市样本二", "industry": "软件服务"},
                {"ts_code": "000003.SZ", "name": "深市样本三", "industry": "软件服务"},
            ],
        ),
        "index_daily": _write_dataset(
            snapshot,
            "index_daily",
            [
                {
                    "ts_code": "000300.SH",
                    "trade_date": "20260714",
                    "close": 4200.0,
                    "pre_close": 4160.0,
                    "pct_chg": 0.96,
                    "amount": 500_000_000.0,
                },
                {
                    "ts_code": "000905.SH",
                    "trade_date": "20260714",
                    "close": 6200.0,
                    "pre_close": 6250.0,
                    "pct_chg": -0.8,
                    "amount": 300_000_000.0,
                },
                {
                    "ts_code": "000852.SH",
                    "trade_date": "20260714",
                    "close": 6500.0,
                    "pre_close": 6480.0,
                    "pct_chg": 0.31,
                    "amount": 280_000_000.0,
                },
                {
                    "ts_code": "000016.SH",
                    "trade_date": "20260714",
                    "close": 3000.0,
                    "pre_close": 2990.0,
                    "pct_chg": 0.33,
                    "amount": 220_000_000.0,
                },
            ],
        ),
        "fund_daily": _write_dataset(
            snapshot,
            "fund_daily",
            [
                {
                    "ts_code": "510300.SH",
                    "trade_date": "20260714",
                    "close": 4.2,
                    "pre_close": 4.15,
                    "pct_chg": 1.2,
                    "amount": 9_000_000.0,
                },
                {
                    "ts_code": "159919.SZ",
                    "trade_date": "20260714",
                    "close": 4.1,
                    "pre_close": 4.12,
                    "pct_chg": -0.49,
                    "amount": 7_000_000.0,
                },
            ],
        ),
        "fund_basic": _write_dataset(
            snapshot,
            "fund_basic",
            [
                {"ts_code": "510300.SH", "name": "沪深300ETF"},
                {"ts_code": "159919.SZ", "name": "沪深300ETF深市"},
            ],
        ),
        "fut_daily": _write_dataset(
            snapshot,
            "fut_daily",
            [
                {
                    "ts_code": "IF2607.CFX",
                    "trade_date": "20260714",
                    "close": 4210.0,
                    "pre_close": 4190.0,
                    "pct_chg": 0.48,
                    "amount": 200_000.0,
                },
                {
                    "ts_code": "IC2607.CFX",
                    "trade_date": "20260714",
                    "close": 6180.0,
                    "pre_close": 6200.0,
                    "pct_chg": -0.32,
                    "amount": 180_000.0,
                },
                {
                    "ts_code": "IM2607.CFX",
                    "trade_date": "20260714",
                    "close": 6510.0,
                    "pre_close": 6500.0,
                    "pct_chg": 0.15,
                    "amount": 160_000.0,
                },
                {
                    "ts_code": "IH2607.CFX",
                    "trade_date": "20260714",
                    "close": 3010.0,
                    "pre_close": 3000.0,
                    "pct_chg": 0.33,
                    "amount": 150_000.0,
                },
            ],
        ),
    }
    manifest = {
        "name": snapshot.name,
        "created_at": datetime.now(UTC).isoformat(),
        "frequency": "day",
        "start_date": "2026-07-13",
        "end_date": "2026-07-14",
        "datasets": datasets,
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return snapshot


def test_market_overview_reports_not_ready_without_daily_snapshot(tmp_path: Path) -> None:
    result = MarketOverviewService(tmp_path, cache_seconds=0).get()

    assert result["status"] == "not_ready"
    assert result["source"]["is_realtime"] is False
    assert result["indices"] == []


def test_market_overview_aggregates_snapshot_without_mixing_frequencies(tmp_path: Path) -> None:
    snapshot = _build_snapshot(tmp_path)

    result = MarketOverviewService(tmp_path, cache_seconds=0).materialize(
        snapshot_name=snapshot.name,
        symbols=["600001.SH", "510300.SH", "000300.SH"]
    )

    assert result["status"] == "ready"
    assert result["source"]["snapshot_name"] == "research-20260714"
    assert result["source"]["as_of"] == "2026-07-14"
    assert result["source"]["is_realtime"] is False
    assert result["breadth"]["instruments"] == 6
    assert result["breadth"]["advances"] == 3
    assert result["breadth"]["declines"] == 2
    assert result["breadth"]["unchanged"] == 1
    assert len(result["indices"]) == 4
    assert {item["product"] for item in result["futures"]} == {"IF", "IC", "IM", "IH"}
    assert result["etfs"][0]["name"] == "沪深300ETF"
    assert {item["industry"] for item in result["sectors"]} == {"银行", "软件服务"}
    assert [item["ts_code"] for item in result["watchlist"]] == [
        "600001.SH",
        "510300.SH",
        "000300.SH",
    ]


def test_market_overview_rejects_snapshot_path_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        MarketOverviewService(tmp_path, cache_seconds=0).get(snapshot_name="../outside")


def test_market_overview_reads_a_persisted_materialized_result(tmp_path: Path) -> None:
    snapshot = _build_snapshot(tmp_path)
    first = MarketOverviewService(tmp_path, cache_seconds=0).materialize(
        snapshot_name=snapshot.name
    )

    class ArtifactOnlyService(MarketOverviewService):
        def _select_snapshot(self, snapshot_name: str | None):
            raise AssertionError("latest dashboard should avoid manifest and parquet scans")

    second = ArtifactOnlyService(tmp_path, cache_seconds=0).get()

    assert second == first
    artifacts = list((tmp_path / "artifacts" / "market-overview").rglob("*.json"))
    assert len(artifacts) == 3


def test_market_overview_cold_projection_never_scans_snapshot_or_duckdb(
    tmp_path: Path,
) -> None:
    _build_snapshot(tmp_path)

    class ProjectionOnlyService(MarketOverviewService):
        def _select_snapshot(self, snapshot_name: str | None):
            raise AssertionError("HTTP projection read must not inspect snapshot manifests")

        def _build(self, snapshot, manifest, watchlist):
            raise AssertionError("HTTP projection read must not open DuckDB")

    started = time.perf_counter()
    result = ProjectionOnlyService(tmp_path, cache_seconds=0).get()

    assert time.perf_counter() - started < 0.2
    assert result["status"] == "not_ready"
    assert "不会现场扫描" in result["message"]


def test_market_overview_corrupt_projection_is_nonblocking_and_uses_previous(
    tmp_path: Path,
) -> None:
    snapshot = _build_snapshot(tmp_path)
    first = MarketOverviewService(tmp_path, cache_seconds=0).materialize(
        snapshot_name=snapshot.name
    )
    # Re-publication creates a durable previous pointer before replacing latest.
    second = MarketOverviewService(tmp_path, cache_seconds=0).materialize(
        snapshot_name=snapshot.name
    )
    assert second == first
    latest = tmp_path / "artifacts" / "market-overview" / "latest.json"
    latest.write_text("{broken", encoding="utf-8")

    class ProjectionOnlyService(MarketOverviewService):
        def _select_snapshot(self, snapshot_name: str | None):
            raise AssertionError("damaged display cache must not trigger a manifest scan")

        def _build(self, snapshot, manifest, watchlist):
            raise AssertionError("damaged display cache must not trigger DuckDB")

    started = time.perf_counter()
    recovered = ProjectionOnlyService(tmp_path, cache_seconds=0).get()

    assert time.perf_counter() - started < 0.2
    assert recovered == first


def test_market_overview_limits_monthly_partitions_before_touching_files(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshots" / "partitioned"
    files = []
    for year, month in ((2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 5)):
        target = (
            snapshot
            / "parquet"
            / "daily"
            / f"partition_year={year}"
            / f"partition_month={month}"
            / "data.parquet"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
        files.append({"path": target.relative_to(snapshot).as_posix()})

    selected = MarketOverviewService._entry_files(
        snapshot,
        {"files": files},
        recent_months=2,
    )

    assert [path.parent.name for path in selected] == [
        "partition_month=4",
        "partition_month=5",
    ]
