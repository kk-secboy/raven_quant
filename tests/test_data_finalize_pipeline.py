from pathlib import Path

from fastapi.testclient import TestClient

from quant_data.checkpoint import CheckpointStore
from quant_data.cli import _profile_datasets
from quant_data.config import Settings
from quant_data.execution_data import MARGIN_DATASET, MINUTE_DATASETS
from quant_data.models import FetchSpec, UnitResult
from quant_platform.api import create_app
from quant_platform.job_store import JobStore
from quant_platform.worker import LocalJobWorker


def _completed_unit(database_url: str) -> None:
    checkpoint = CheckpointStore(database_url)
    spec = FetchSpec(
        dataset="daily",
        api_name="daily",
        scope={"trade_date": "20240102"},
        params={"trade_date": "20240102"},
        fields=("ts_code", "trade_date", "close"),
    )
    checkpoint.add([spec])
    checkpoint.succeed(
        spec.unit_key,
        UnitResult(output_path="units/daily/fixture.parquet", row_count=1, sha256="a" * 64),
    )


def test_finalize_api_requires_completed_downloads_and_queues_verify(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    checkpoint = CheckpointStore(database_url)
    pending = FetchSpec(
        dataset="daily",
        api_name="daily",
        scope={"trade_date": "20240102"},
        params={"trade_date": "20240102"},
        fields=("ts_code", "trade_date", "close"),
    )
    checkpoint.add([pending])
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    app = create_app(tmp_path)
    with TestClient(app) as client:
        blocked = client.post(
            "/api/jobs/finalize-data",
            json={"profile": "full", "start": "2024-01-01", "end": "2024-01-31"},
        )
    assert blocked.status_code == 409
    assert "pending=1" in blocked.json()["detail"]

    checkpoint.succeed(
        pending.unit_key,
        UnitResult(output_path="units/daily/fixture.parquet", row_count=1, sha256="a" * 64),
    )
    with TestClient(app) as client:
        queued = client.post(
            "/api/jobs/finalize-data",
            json={
                "profile": "full",
                "start": "2024-01-01",
                "end": "2024-01-31",
                "snapshot_name": "cn-finalize-fixture",
            },
        )
    assert queued.status_code == 202
    body = queued.json()
    assert body["kind"] == "data_verify"
    assert body["payload"]["snapshot_name"] == "cn-finalize-fixture"


def test_data_finalize_stages_are_durable_idempotent_and_retryable(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    settings = Settings.from_env(tmp_path / ".env")
    jobs = JobStore(database_url)
    worker = LocalJobWorker(jobs, tmp_path, settings)
    payload = {
        "pipeline_id": "pipeline",
        "profile": "full",
        "start": "2024-01-01",
        "end": "2026-07-13",
        "snapshot_name": "cn-durable-fixture",
    }
    verify = jobs.create("data_verify", payload, tmp_path / "verify.log")
    command, result, env = worker._command(verify)
    assert "verify" in command
    assert command[-3:] == [
        "--snapshot-end",
        payload["end"],
        "--allow-incomplete-plans",
    ]
    assert result is None and env == {}
    jobs.finish(verify["id"], exit_code=0)

    bootstrap = {
        "kind": "bootstrap",
        "payload": {
            "profile": "full",
            "start": "2026-07-06",
            "end": "latest",
            "finalize_after_download": True,
            "pipeline_id": "scheduled-pipeline",
            "snapshot_start": "2024-01-01",
            "snapshot_end": "2026-07-13",
            "snapshot_name": "cn-scheduled-fixture",
        },
    }
    scheduled_verify = worker._queue_data_pipeline_successor(bootstrap)
    assert scheduled_verify["kind"] == "data_verify"
    assert scheduled_verify["payload"]["start"] == "2024-01-01"

    snapshot = worker._queue_data_pipeline_successor(verify)
    duplicate = worker._queue_data_pipeline_successor(verify)
    assert duplicate["id"] == snapshot["id"]
    assert snapshot["kind"] == "data_snapshot"
    assert "--name" in worker._command(snapshot)[0]

    qlib = worker._queue_data_pipeline_successor(snapshot)
    assert qlib["kind"] == "data_qlib"
    assert "build-qlib" in worker._command(qlib)[0]

    baseline = worker._queue_data_pipeline_successor(qlib)
    assert baseline["kind"] == "qlib_baseline"
    assert baseline["payload"]["dataset"] == "cn-durable-fixture"

    jobs.finish(snapshot["id"], exit_code=1, error="fixture failure")
    retried = jobs.retry(snapshot["id"])
    assert retried["status"] == "queued"
    assert retried["error"] is None


def test_chained_data_pipeline_keeps_each_download_and_build_as_separate_job(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    settings = Settings.from_env(tmp_path / ".env")
    jobs = JobStore(database_url)
    worker = LocalJobWorker(jobs, tmp_path, settings)
    steps = [
        {
            "kind": "supplemental_cn_macro",
            "payload": {
                "bundle": "cn_macro",
                "start": "2025-01-01",
                "end": "latest",
                "symbols": [],
            },
        },
        {"kind": "data_verify", "payload": {}},
        {"kind": "data_snapshot", "payload": {}},
        {"kind": "data_qlib", "payload": {}},
        {"kind": "qlib_baseline", "payload": {}},
    ]
    current = {
        "kind": "bootstrap",
        "payload": {
            "pipeline_id": "scheduled-chain",
            "profile": "full",
            "start": "2025-01-01",
            "end": "latest",
            "snapshot_start": "2024-01-01",
            "snapshot_end": "2025-01-02",
            "snapshot_name": "cn-chain-fixture",
            "pipeline_steps": steps,
            "pipeline_next_index": 0,
        },
    }

    expected = [step["kind"] for step in steps]
    created = []
    for _kind in expected:
        successor = worker._queue_data_pipeline_successor(current)
        created.append(successor["kind"])
        current = successor

    assert created == expected
    assert current["payload"]["dataset"] == "cn-chain-fixture"
    assert current["payload"]["pipeline_next_index"] == len(steps)
    assert worker._has_data_pipeline_successor(current) is False


def test_five_minute_download_chains_to_minute_qlib_build(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    settings = Settings.from_env(tmp_path / ".env")
    jobs = JobStore(database_url)
    worker = LocalJobWorker(jobs, tmp_path, settings)
    download = {
        "kind": "ashare_5m_download",
        "payload": {
            "pipeline_id": "ashare-5m-fixture",
            "profile": "ashare_intraday",
            "start": "2024-01-01",
            "end": "2025-01-02",
            "snapshot_name": "ashare-5m-20250102",
            "pipeline_steps": [
                {
                    "kind": "minute_qlib",
                    "payload": {
                        "output_name": "ashare-5m-20250102-5min",
                        "target_frequency": "5min",
                    },
                }
            ],
            "pipeline_next_index": 0,
        },
    }

    successor = worker._queue_data_pipeline_successor(download)

    assert successor["kind"] == "minute_qlib"
    assert successor["payload"]["snapshot_name"] == "ashare-5m-20250102"
    assert successor["payload"]["output_name"] == "ashare-5m-20250102-5min"
    assert successor["payload"]["target_frequency"] == "5min"
    assert worker._has_data_pipeline_successor(successor) is False


def test_full_snapshot_contract_keeps_execution_frequency_separate() -> None:
    assert "daily" in _profile_datasets("full")
    assert {"stk_premarket", "stk_auction_o", "stk_auction_c"} <= _profile_datasets("full")
    assert MARGIN_DATASET not in _profile_datasets("full")
    assert not set(MINUTE_DATASETS).intersection(_profile_datasets("full"))
