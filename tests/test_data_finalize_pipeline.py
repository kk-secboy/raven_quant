from pathlib import Path

from fastapi.testclient import TestClient

from quant_data.checkpoint import CheckpointStore
from quant_data.config import Settings
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
    assert command[-1] == "verify"
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
