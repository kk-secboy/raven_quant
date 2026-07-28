import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_data.checkpoint import CheckpointStore
from quant_data.models import FetchSpec, ProviderResult
from quant_data.runner import DownloadRunner
from quant_data.storage import ParquetStore
from quant_data.verify import verify_downloads


class FakeProvider:
    def fetch(self, api_name, params, fields=()):
        trade_date = params["trade_date"]
        if api_name == "adj_factor":
            rows = [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": trade_date,
                    "adj_factor": 1.0,
                }
            ]
        else:
            rows = [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": trade_date,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.5,
                    "close": 10.5,
                    "vol": 100.0,
                }
            ]
        return ProviderResult(api_name, list(rows[0]), rows, json.dumps(rows).encode())


def test_runner_writes_atomic_parquet_and_snapshot(tmp_path: Path, database_url: str) -> None:
    checkpoint = CheckpointStore(database_url)
    storage = ParquetStore(tmp_path)
    specs = [
        FetchSpec(
            dataset="daily",
            api_name="daily",
            scope={"trade_date": day},
            params={"trade_date": day},
            fields=("ts_code", "trade_date", "open", "high", "low", "close", "vol"),
        )
        for day in ("20240102", "20240103")
    ]
    specs += [
        FetchSpec(
            dataset="adj_factor",
            api_name="adj_factor",
            scope={"trade_date": day},
            params={"trade_date": day},
            fields=("ts_code", "trade_date", "adj_factor"),
        )
        for day in ("20240102", "20240103")
    ]
    checkpoint.add(specs)
    runner = DownloadRunner(
        checkpoint=checkpoint,
        storage=storage,
        provider=FakeProvider(),
        workers=2,
    )
    summary = runner.run({"daily", "adj_factor"})
    assert summary.succeeded == 4
    assert summary.failed == 0
    assert not list((tmp_path / "units").rglob("*.tmp"))
    raw_files = list((tmp_path / "raw").rglob("*.json.gz"))
    assert len(raw_files) == 4

    report = verify_downloads(checkpoint, tmp_path)
    assert report["ok"] is True
    snapshot = storage.build_snapshot(
        name="test",
        successful_units={"daily": checkpoint.successful("daily")},
        manifest_extra={"profile": "test"},
    )
    files = list((snapshot / "parquet" / "daily").rglob("*.parquet"))
    assert files
    frame = pd.concat([pd.read_parquet(path) for path in files])
    assert len(frame) == 2
    assert set(frame["ts_code"]) == {"000001.SZ"}
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    daily = manifest["datasets"]["daily"]
    assert len(daily["source_sha256"]) == 64
    assert daily["files"]
    for item in daily["files"]:
        path = snapshot / item["path"]
        assert item["bytes"] == path.stat().st_size
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_verifier_uses_successor_generation_and_can_ignore_dormant_plans(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    storage = ParquetStore(tmp_path)
    parent_group = "share_float:20240101:20240131"
    parent = FetchSpec(
        dataset="share_float",
        api_name="share_float",
        params={"start_date": "20240101", "end_date": "20240131", "offset": 0},
        scope={"page_group": parent_group, "offset": 0},
    )
    replacement = FetchSpec(
        dataset="share_float",
        api_name="share_float",
        params={"start_date": "20240101", "end_date": "20240101", "offset": 0},
        scope={
            "page_group": f"{parent_group}:daily:20240101",
            "offset": 0,
            "supersedes_page_group": parent_group,
        },
    )
    dormant = FetchSpec(
        dataset="daily",
        api_name="daily",
        params={"trade_date": "20240103"},
        scope={"trade_date": "20240103"},
    )
    checkpoint.add([parent, replacement, dormant])
    row = {
        "ts_code": "000001.SZ",
        "ann_date": "20240101",
        "float_date": "20240101",
        "holder_name": "holder",
        "share_type": "A",
    }
    for spec in (parent, replacement):
        written = storage.write_unit(
            "share_float",
            spec.unit_key,
            ProviderResult(
                api_name="share_float",
                columns=list(row),
                rows=[row],
                raw_body=b"{}",
            ),
        )
        checkpoint.succeed(spec.unit_key, written)

    strict = verify_downloads(checkpoint, tmp_path)
    relaxed = verify_downloads(checkpoint, tmp_path, require_all_planned=False)

    assert strict["ok"] is False
    assert any("daily: 0/1 units succeeded" in item for item in strict["errors"])
    assert relaxed["ok"] is True
    assert relaxed["duplicate_checks"]["share_float"] == 0
    assert any("daily: 0/1 units succeeded" in item for item in relaxed["warnings"])


def test_verifier_downgrades_only_exact_duplicates_for_unstable_pagination_datasets(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    storage = ParquetStore(tmp_path)
    unstable = FetchSpec(
        dataset="dc_hot",
        api_name="dc_hot",
        params={"trade_date": "20240102"},
        scope={"page_group": "dc_hot:20240102", "offset": 0},
    )
    stable = FetchSpec(
        dataset="adj_factor",
        api_name="adj_factor",
        params={"trade_date": "20240102"},
        scope={"trade_date": "20240102"},
    )
    checkpoint.add([unstable, stable])
    hot_row = {
        "trade_date": "20240102",
        "ts_code": "000001.SZ",
        "market": "concept",
        "hot_type": "popularity",
        "rank": 1,
    }
    adj_row = {"ts_code": "000001.SZ", "trade_date": "20240102", "adj_factor": 1.0}
    for spec, row in ((unstable, hot_row), (stable, adj_row)):
        written = storage.write_unit(
            spec.dataset,
            spec.unit_key,
            ProviderResult(
                api_name=spec.api_name,
                columns=list(row),
                rows=[row, dict(row)],
                raw_body=b"{}",
            ),
        )
        checkpoint.succeed(spec.unit_key, written)

    result = verify_downloads(checkpoint, tmp_path)

    assert result["duplicate_checks"] == {"adj_factor": 1, "dc_hot": 1}
    assert result["conflicting_duplicate_checks"] == {"dc_hot": 0}
    assert any(
        "dc_hot: 1 exact duplicate primary-key rows" in item for item in result["warnings"]
    )
    assert any(
        "adj_factor: 1 duplicate primary-key rows" in item for item in result["errors"]
    )


def test_verifier_rejects_conflicting_duplicates_from_unstable_pagination(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    storage = ParquetStore(tmp_path)
    spec = FetchSpec(
        dataset="dc_hot",
        api_name="dc_hot",
        params={"trade_date": "20240102"},
        scope={"page_group": "dc_hot:20240102", "offset": 0},
    )
    checkpoint.add([spec])
    first = {
        "trade_date": "20240102",
        "ts_code": "000001.SZ",
        "market": "concept",
        "hot_type": "popularity",
        "rank": 1,
    }
    second = {**first, "rank": 2}
    written = storage.write_unit(
        spec.dataset,
        spec.unit_key,
        ProviderResult(
            api_name=spec.api_name,
            columns=list(first),
            rows=[first, second],
            raw_body=b"{}",
        ),
    )
    checkpoint.succeed(spec.unit_key, written)

    result = verify_downloads(checkpoint, tmp_path)

    assert result["duplicate_checks"] == {"dc_hot": 1}
    assert result["conflicting_duplicate_checks"] == {"dc_hot": 1}
    assert any(
        "dc_hot: 1 conflicting primary-key rows" in item for item in result["errors"]
    )


def test_verifier_rejects_missing_trading_day_and_stock_quote(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    storage = ParquetStore(tmp_path)
    fixtures = {
        "trade_cal": [
            {"exchange": "SSE", "cal_date": "20240102", "is_open": "1"},
            {"exchange": "SSE", "cal_date": "20240103", "is_open": "1"},
        ],
        "daily": [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "vol": 100.0,
            }
        ],
        "daily_basic": [
            {"ts_code": "000001.SZ", "trade_date": "20240102", "total_mv": 100.0},
            {"ts_code": "000002.SZ", "trade_date": "20240102", "total_mv": 200.0},
        ],
    }
    specs = [
        FetchSpec(
            dataset=dataset,
            api_name=dataset,
            scope={"fixture": dataset},
            params={"fixture": dataset},
        )
        for dataset in fixtures
    ]
    checkpoint.add(specs)
    for spec in specs:
        rows = fixtures[spec.dataset]
        written = storage.write_unit(
            spec.dataset,
            spec.unit_key,
            ProviderResult(spec.api_name, list(rows[0]), rows, b"{}"),
        )
        checkpoint.succeed(spec.unit_key, written)

    report = verify_downloads(checkpoint, tmp_path)

    assert report["ok"] is False
    assert report["completeness_checks"]["missing_trading_days"] == 1
    assert report["completeness_checks"]["stocks_missing_daily_quotes"] == 1
    assert any("open trading days have no quotes" in item for item in report["errors"])
    assert any("stock/date quotes are missing" in item for item in report["errors"])


def test_verifier_scopes_daily_basic_to_complete_bse_history(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    storage = ParquetStore(tmp_path)
    fixtures = {
        "trade_cal": [{"exchange": "SSE", "cal_date": "20221230", "is_open": "1"}],
        "daily": [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20221230",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "vol": 100.0,
            },
            {
                "ts_code": "920001.BJ",
                "trade_date": "20221230",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "vol": 100.0,
            },
        ],
        "daily_basic": [
            {"ts_code": "000001.SZ", "trade_date": "20221230", "total_mv": 100.0}
        ],
        "adj_factor": [
            {"ts_code": "000001.SZ", "trade_date": "20221230", "adj_factor": 1.0},
            {"ts_code": "920001.BJ", "trade_date": "20221230", "adj_factor": 1.0},
        ],
    }
    specs = [
        FetchSpec(
            dataset=dataset,
            api_name=dataset,
            scope={"fixture": dataset},
            params={"fixture": dataset},
        )
        for dataset in fixtures
    ]
    checkpoint.add(specs)
    for spec in specs:
        rows = fixtures[spec.dataset]
        written = storage.write_unit(
            spec.dataset,
            spec.unit_key,
            ProviderResult(spec.api_name, list(rows[0]), rows, b"{}"),
        )
        checkpoint.succeed(spec.unit_key, written)

    report = verify_downloads(checkpoint, tmp_path)

    assert report["ok"] is True, report["errors"]
    assert report["completeness_checks"]["daily_rows_outside_daily_basic_history"] == 1
    assert report["completeness_checks"]["stocks_missing_daily_basic"] == 0


def test_verifier_warns_for_isolated_daily_basic_provider_hole(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    storage = ParquetStore(tmp_path)
    day = "20240102"
    daily_rows = [
        {
            "ts_code": f"{index:06d}.SZ",
            "trade_date": day,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
            "vol": 100.0,
        }
        for index in range(10_001)
    ]
    basic_rows = [
        {"ts_code": row["ts_code"], "trade_date": day, "total_mv": 100.0}
        for row in daily_rows[1:]
    ]
    fixtures = {
        "trade_cal": [{"exchange": "SSE", "cal_date": day, "is_open": "1"}],
        "daily": daily_rows,
        "daily_basic": basic_rows,
        "adj_factor": [
            {"ts_code": row["ts_code"], "trade_date": day, "adj_factor": 1.0}
            for row in daily_rows
        ],
    }
    specs = [
        FetchSpec(
            dataset=dataset,
            api_name=dataset,
            scope={"fixture": dataset},
            params={"fixture": dataset},
        )
        for dataset in fixtures
    ]
    checkpoint.add(specs)
    for spec in specs:
        rows = fixtures[spec.dataset]
        written = storage.write_unit(
            spec.dataset,
            spec.unit_key,
            ProviderResult(spec.api_name, list(rows[0]), rows, b"{}"),
        )
        checkpoint.succeed(spec.unit_key, written)

    report = verify_downloads(checkpoint, tmp_path)

    assert report["ok"] is True, report["errors"]
    assert report["completeness_checks"]["stocks_missing_daily_basic"] == 1
    assert any(
        "daily_basic: 1 stock/date rows" in item
        and "below the 0.01% blocking threshold" in item
        for item in report["warnings"]
    )


def test_verifier_ignores_b_share_codes_outside_market_scope(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    storage = ParquetStore(tmp_path)
    fixtures = {
        "trade_cal": [
            {"exchange": "SSE", "cal_date": "20240102", "is_open": "1"},
        ],
        "daily": [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "vol": 100.0,
            }
        ],
        "daily_basic": [
            {"ts_code": "000001.SZ", "trade_date": "20240102", "total_mv": 100.0},
            # B-share rows (200xxx.SZ / 900xxx.SH) are outside the product's
            # market scope and must not fail the cross-dataset check.
            {"ts_code": "200011.SZ", "trade_date": "20240102", "total_mv": 50.0},
            {"ts_code": "900901.SH", "trade_date": "20240102", "total_mv": 60.0},
        ],
        "adj_factor": [
            {"ts_code": "000001.SZ", "trade_date": "20240102", "adj_factor": 1.0},
        ],
    }
    specs = [
        FetchSpec(
            dataset=dataset,
            api_name=dataset,
            scope={"fixture": dataset},
            params={"fixture": dataset},
        )
        for dataset in fixtures
    ]
    checkpoint.add(specs)
    for spec in specs:
        rows = fixtures[spec.dataset]
        written = storage.write_unit(
            spec.dataset,
            spec.unit_key,
            ProviderResult(spec.api_name, list(rows[0]), rows, b"{}"),
        )
        checkpoint.succeed(spec.unit_key, written)

    report = verify_downloads(checkpoint, tmp_path)

    assert report["ok"] is True
    assert report["completeness_checks"]["stocks_missing_daily_quotes"] == 0
    assert report["completeness_checks"]["stocks_missing_daily_basic"] == 0


def test_verifier_ignores_ghost_codes_absent_from_security_master(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    storage = ParquetStore(tmp_path)
    fixtures = {
        "trade_cal": [
            {"exchange": "SSE", "cal_date": "20240102", "is_open": "1"},
        ],
        "stock_basic": [
            {"ts_code": "000001.SZ", "name": "PA", "list_status": "L"},
        ],
        "daily": [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "vol": 100.0,
            }
        ],
        "daily_basic": [
            {"ts_code": "000001.SZ", "trade_date": "20240102", "total_mv": 100.0},
            # Ghost code: present in daily_basic upstream but absent from the
            # security master and from every quotes interface.
            {"ts_code": "201872.SZ", "trade_date": "20240102", "total_mv": 50.0},
        ],
        "adj_factor": [
            {"ts_code": "000001.SZ", "trade_date": "20240102", "adj_factor": 1.0},
        ],
    }
    specs = [
        FetchSpec(
            dataset=dataset,
            api_name=dataset,
            scope={"fixture": dataset},
            params={"fixture": dataset},
        )
        for dataset in fixtures
    ]
    checkpoint.add(specs)
    for spec in specs:
        rows = fixtures[spec.dataset]
        written = storage.write_unit(
            spec.dataset,
            spec.unit_key,
            ProviderResult(spec.api_name, list(rows[0]), rows, b"{}"),
        )
        checkpoint.succeed(spec.unit_key, written)

    report = verify_downloads(checkpoint, tmp_path)

    assert report["ok"] is True
    assert report["completeness_checks"]["stocks_missing_daily_quotes"] == 0
    assert report["completeness_checks"]["stocks_missing_daily_basic"] == 0
