import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quant_data.checkpoint import CheckpointStore
from quant_data.cli import _profile_datasets
from quant_data.config import Settings
from quant_data.execution_data import MARGIN_DATASET, MINUTE_DATASETS
from quant_data.models import FetchSpec, UnitResult
from quant_platform.api import create_app
from quant_platform.job_store import JobStore
from quant_platform.scheduler import AUTOMATED_DATA_BUNDLES
from quant_platform.worker import LocalJobWorker


def _stub_publication_validation(monkeypatch, data_root: Path, name: str) -> dict:
    # These database tests exercise job durability. The actual filesystem seals
    # and recovery-to-baseline handoff are covered in test_qlib_publication.py.
    receipt = {
        "dataset": name,
        "dataset_path": str(data_root / "qlib" / name),
        "dataset_identity_sha256": "a" * 64,
    }

    def validate(root, requested, result):
        assert root == data_root and requested == name and result == receipt
        return receipt

    monkeypatch.setattr("quant_platform.worker.validate_qlib_publication_receipt", validate)
    return receipt


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


def test_finalize_api_queues_window_scoped_verify_without_global_precheck(
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
        incomplete_profile = client.post(
            "/api/jobs/finalize-data",
            json={"profile": "core", "start": "2024-01-01", "end": "2024-01-31"},
        )
        pending_verification = client.post(
            "/api/jobs/finalize-data",
            json={
                "profile": "full",
                "start": "2024-01-01",
                "end": "2024-01-31",
                "snapshot_name": "cn-pending-window-fixture",
            },
        )
    assert incomplete_profile.status_code == 422
    assert "full data profile" in str(incomplete_profile.json())
    assert pending_verification.status_code == 202
    assert pending_verification.json()["kind"] == "data_verify"
    cancelled = JobStore(database_url).request_cancel(pending_verification.json()["id"])
    assert cancelled["status"] == "cancelled"

    checkpoint.succeed(
        pending.unit_key,
        UnitResult(output_path="units/daily/fixture.parquet", row_count=1, sha256="a" * 64),
    )
    terminal = FetchSpec(
        dataset="daily_basic",
        api_name="daily_basic",
        scope={"trade_date": "20240102"},
        params={"trade_date": "20240102"},
        fields=("ts_code", "trade_date", "close"),
    )
    superseded = FetchSpec(
        dataset="adj_factor",
        api_name="adj_factor",
        scope={"trade_date": "20240102"},
        params={"trade_date": "20240102"},
        fields=("ts_code", "trade_date", "adj_factor"),
    )
    checkpoint.add([terminal, superseded])
    checkpoint.fail(terminal.unit_key, "provider has no published row", terminal=True)
    checkpoint.supersede_units([superseded.unit_key], "outside frozen snapshot")
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
    assert command[-4:] == [
        "--snapshot-end",
        payload["end"],
        "--profile",
        payload["profile"],
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
    assert scheduled_verify["payload"]["end"] == "2026-07-13"
    assert scheduled_verify["payload"]["snapshot_start"] == "2024-01-01"
    assert scheduled_verify["payload"]["snapshot_end"] == "2026-07-13"

    snapshot = worker._queue_data_pipeline_successor(verify)
    duplicate = worker._queue_data_pipeline_successor(verify)
    assert duplicate["id"] == snapshot["id"]
    assert snapshot["kind"] == "data_snapshot"
    assert "--name" in worker._command(snapshot)[0]

    qlib = worker._queue_data_pipeline_successor(snapshot)
    assert qlib["kind"] == "data_qlib"
    assert "build-qlib" in worker._command(qlib)[0]

    receipt = _stub_publication_validation(monkeypatch, settings.data_root, "cn-durable-fixture")
    baseline = worker._queue_data_pipeline_successor(qlib, result=receipt)
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
        {
            "kind": "supplemental_cn_funds",
            "payload": {
                "bundle": "cn_funds",
                "start": "2025-01-03",
                "end": "2025-01-04",
                "symbols": [],
            },
        },
        {
            "kind": "supplemental_global_markets",
            "payload": {
                "bundle": "global_markets",
                "start": "2025-01-05",
                "end": "2025-01-06",
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
        "max_attempts": 3,
        "payload": {
            "pipeline_id": "scheduled-chain",
            "profile": "full",
            "start": "2025-01-01",
            "end": "latest",
            "snapshot_start": "2008-01-01",
            "snapshot_end": "2025-01-31",
            "snapshot_name": "cn-chain-fixture",
            "pipeline_steps": steps,
            "pipeline_next_index": 0,
        },
    }

    expected = [step["kind"] for step in steps]
    receipt = _stub_publication_validation(monkeypatch, settings.data_root, "cn-chain-fixture")
    created = []
    for _kind in expected:
        successor = worker._queue_data_pipeline_successor(
            current, result=receipt if current["kind"] == "data_qlib" else None
        )
        created.append(successor)
        current = successor

    assert [job["kind"] for job in created] == expected
    assert [(job["payload"]["start"], job["payload"]["end"]) for job in created[:3]] == [
        ("2025-01-01", "latest"),
        ("2025-01-03", "2025-01-04"),
        ("2025-01-05", "2025-01-06"),
    ]
    verify = created[3]
    assert verify["kind"] == "data_verify"
    assert verify["payload"]["start"] == "2008-01-01"
    assert verify["payload"]["end"] == "2025-01-31"
    assert all(
        job["payload"]["snapshot_start"] == "2008-01-01"
        and job["payload"]["snapshot_end"] == "2025-01-31"
        for job in created
    )
    assert all(job["max_attempts"] == 3 for job in created)
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
    manifest_bytes = b'{"frequency":"5min","datasets":{"ashare_5m":{}}}'
    manifest_path = (
        settings.data_root / "snapshots" / "ashare-5m-20250102" / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(manifest_bytes)
    expected_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    download = {
        "kind": "ashare_5m_download",
        "payload": {
            "pipeline_id": "ashare-5m-fixture",
            "profile": "ashare_intraday",
            "start": "2024-01-01",
            "end": "2025-01-02",
            "snapshot_name": "ashare-5m-20250102",
            "source_lineage_id": "a" * 64,
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
    assert successor["payload"]["snapshot_start"] == "2024-01-01"
    assert successor["payload"]["snapshot_end"] == "2025-01-02"
    assert successor["payload"]["output_name"] == "ashare-5m-20250102-5min"
    assert successor["payload"]["target_frequency"] == "5min"
    assert successor["payload"]["snapshot_manifest_sha256"] == expected_manifest_sha256
    command, result_path, env = worker._command(successor)
    assert command[-2:] == ["--expected-manifest-sha256", expected_manifest_sha256]
    assert result_path is None
    assert env == {}
    assert worker._has_data_pipeline_successor(successor) is False


def test_worker_accepts_every_automated_supplemental_bundle_in_one_chain(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    settings = Settings.from_env(tmp_path / ".env")
    worker = LocalJobWorker(JobStore(database_url), tmp_path, settings)
    steps = [
        {
            "kind": f"supplemental_{bundle}",
            "payload": {
                "bundle": bundle,
                "start": "2025-01-01",
                "end": "2025-01-31",
                "symbols": [],
            },
        }
        for bundle in AUTOMATED_DATA_BUNDLES
    ]
    current = {
        "kind": "bootstrap",
        "payload": {
            "pipeline_id": "all-automated-bundles-fixture",
            "profile": "full",
            "start": "2025-01-01",
            "end": "2025-01-31",
            "snapshot_start": "2008-01-01",
            "snapshot_end": "2025-01-31",
            "snapshot_name": "cn-all-automated-bundles-fixture",
            "pipeline_steps": steps,
            "pipeline_next_index": 0,
        },
    }

    created = []
    for bundle in AUTOMATED_DATA_BUNDLES:
        current = worker._queue_data_pipeline_successor(current)
        created.append(current["kind"])
        assert current["kind"] == f"supplemental_{bundle}"
        assert current["payload"]["snapshot_start"] == "2008-01-01"
        assert current["payload"]["snapshot_end"] == "2025-01-31"

    assert created == [f"supplemental_{bundle}" for bundle in AUTOMATED_DATA_BUNDLES]
    assert worker._has_data_pipeline_successor(current) is False


def test_information_pipeline_keeps_download_nlp_and_labels_as_durable_jobs(
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
            "kind": "announcement_nlp",
            "payload": {
                "start": "2025-01-01",
                "end": "2025-01-02",
                "categories": ["regulatory_letter"],
                "limit": 100,
            },
        },
        {
            "kind": "announcement_factor_register",
            "payload": {"factor_name": "all", "actor": "information-scheduler"},
        },
        {
            "kind": "corpus_nlp",
            "payload": {
                "start": "2025-01-01",
                "end": "2025-01-02",
                "datasets": ["major_news"],
                "limit": 100,
            },
        },
        {
            "kind": "corpus_factor_register",
            "payload": {"factor_name": "all", "actor": "information-scheduler"},
        },
        {
            "kind": "event_market_response",
            "payload": {
                "snapshot_name": "cn-verified",
                "horizons": [1, 3, 5, 20],
                "benchmark_code": "000300.SH",
            },
        },
        {
            "kind": "information_factor_evaluate",
            "payload": {
                "dataset": "cn-verified",
                "dataset_path": "/data/qlib/cn-verified",
                "dataset_identity_sha256": "a" * 64,
                "periods": {},
            },
        },
        {
            "kind": "multiface_audit",
            "payload": {
                "dataset": "cn-verified",
                "snapshot_name": "cn-verified",
                "require_ready": True,
            },
        },
    ]
    current = {
        "kind": "cninfo_announcements_download",
        "payload": {
            "pipeline_id": "information-fixture",
            "profile": "information",
            "start": "2024-12-30",
            "end": "2025-01-02",
            "snapshot_name": "information-20250102",
            "pipeline_steps": steps,
            "pipeline_next_index": 0,
        },
    }

    created = []
    for expected in (
        "announcement_nlp",
        "announcement_factor_register",
        "corpus_nlp",
        "corpus_factor_register",
        "event_market_response",
        "information_factor_evaluate",
        "multiface_audit",
    ):
        successor = worker._queue_data_pipeline_successor(current)
        assert successor["kind"] == expected
        created.append(successor)
        current = successor

    assert created[0]["payload"]["limit"] == 100
    assert created[2]["payload"]["datasets"] == ["major_news"]
    assert all(
        job["payload"]["pipeline_snapshot_name"] == "information-20250102"
        for job in created
    )
    assert all(
        job["payload"]["snapshot_name"] == "cn-verified" for job in created[4:]
    )
    assert all(
        job["idempotency_key"]
        == f"data-finalize:information-20250102:{job['kind']}"
        for job in created
    )
    assert all(
        Path(job["log_path"]).name
        == f"{job['kind']}-information-20250102.log"
        for job in created
    )
    assert worker._has_data_pipeline_successor(created[-1]) is False


def test_data_pipeline_step_cannot_override_immutable_snapshot_namespace(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    settings = Settings.from_env(tmp_path / ".env")
    worker = LocalJobWorker(JobStore(database_url), tmp_path, settings)
    job = {
        "kind": "event_market_response",
        "payload": {
            "pipeline_id": "immutable-namespace-fixture",
            "profile": "information",
            "start": "2025-01-01",
            "end": "2025-01-02",
            "snapshot_name": "cn-source-fixture",
            "pipeline_snapshot_name": "information-pipeline-fixture",
            "pipeline_steps": [
                {
                    "kind": "multiface_audit",
                    "payload": {
                        "dataset": "cn-source-fixture",
                        "pipeline_snapshot_name": "attacker-controlled-namespace",
                    },
                }
            ],
            "pipeline_next_index": 0,
        },
    }

    with pytest.raises(
        ValueError,
        match="data pipeline step cannot change its snapshot namespace",
    ):
        worker._queue_data_pipeline_successor(job)


def test_full_snapshot_contract_keeps_execution_frequency_separate() -> None:
    assert "daily" in _profile_datasets("full")
    assert {"stk_premarket", "stk_auction_o", "stk_auction_c"} <= _profile_datasets("full")
    assert MARGIN_DATASET not in _profile_datasets("full")
    assert not set(MINUTE_DATASETS).intersection(_profile_datasets("full"))
