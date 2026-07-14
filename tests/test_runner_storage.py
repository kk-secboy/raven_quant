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
    checkpoint.add(specs)
    runner = DownloadRunner(
        checkpoint=checkpoint,
        storage=storage,
        provider=FakeProvider(),
        workers=2,
    )
    summary = runner.run({"daily"})
    assert summary.succeeded == 2
    assert summary.failed == 0
    assert not list((tmp_path / "units").rglob("*.tmp"))

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
