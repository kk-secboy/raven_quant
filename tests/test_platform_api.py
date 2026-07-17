import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from quant_data.config import Settings
from quant_platform.alert_store import AlertStore
from quant_platform.api import create_app
from quant_platform.job_store import JobStore
from quant_platform.worker import LocalJobWorker


def _trading_calendar(start: date, end: date) -> str:
    current = start
    days: list[str] = []
    while current <= end:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    return "\n".join(days) + "\n"


def test_api_reports_empty_local_state(tmp_path: Path, monkeypatch, database_url: str) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("TUSHARE_API_URL", raising=False)
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("PLATFORM_SECRET_KEY", Fernet.generate_key().decode("ascii"))
    app = create_app(tmp_path)
    with TestClient(app) as client:
        health = client.get("/api/health")
        overview = client.get("/api/overview")
        readiness = client.get("/api/operations/readiness")
        datasets = client.get("/api/datasets")
        market = client.get("/api/market/overview")
        bootstrap = client.post(
            "/api/jobs/bootstrap",
            json={"profile": "core", "start": "2024-01-01", "end": "latest"},
        )
    assert health.json() == {
        "status": "ok",
        "database": "postgresql",
        "worker_mode": "embedded",
        "runtime_secret_storage": "ok",
        "runtime_secret_records": 0,
    }
    assert overview.json()["credentials_configured"] is False
    assert overview.json()["readiness_percent"] < 100
    assert overview.json()["actionable_tasks"] > overview.json()["ready_tasks"]
    assert overview.json()["legacy_download_coverage"] == overview.json()["coverage"]
    assert readiness.status_code == 200
    assert readiness.json()["profiles"][0]["status"] == "blocked"
    assert any(item["name"] == "daily" for item in datasets.json())
    assert market.status_code == 200
    assert market.json()["status"] == "not_ready"
    assert market.json()["source"]["is_realtime"] is False
    assert bootstrap.status_code == 409
    assert bootstrap.json()["detail"] == (
        "missing deployment secret: TUSHARE_API_URL, TUSHARE_TOKEN"
    )
    assert client.get("/api/qlib/datasets").json() == []
    assert client.get("/api/qlib/experiments").json() == []
    allocations = client.get("/api/strategy-allocations")
    assert allocations.status_code == 200
    assert allocations.json() == []
    allocation = client.post(
        "/api/strategy-allocations",
        json={
            "name": "missing dataset allocation",
            "dataset": "missing",
            "members": [
                {"strategy_version_id": "version-a"},
                {"strategy_version_id": "version-b"},
            ],
        },
    )
    assert allocation.status_code == 409
    allocation_schedule = client.post(
        "/api/strategy-allocations/missing/schedule",
        json={"run_time": "15:30", "actor": "operator"},
    )
    assert allocation_schedule.status_code == 404
    invalid_schedule = client.post(
        "/api/strategy-allocations/missing/schedule",
        json={"run_time": "14:30", "actor": "operator"},
    )
    assert invalid_schedule.status_code == 422
    qlib_job = client.post(
        "/api/jobs/qlib-baseline",
        json={"dataset": "missing", "topk": 50, "n_drop": 5},
    )
    assert qlib_job.status_code == 409


def test_job_store_is_durable_and_exclusive(tmp_path: Path, database_url: str) -> None:
    store = JobStore(database_url)
    first = store.create("bootstrap", {"profile": "core"}, tmp_path / "job.log")
    assert first["status"] == "queued"
    claimed = store.claim_next()
    assert claimed and claimed["id"] == first["id"]
    assert claimed["status"] == "running"
    store.finish(first["id"], exit_code=0, result={"metric": 1.0})
    completed = store.get(first["id"])
    assert completed["status"] == "succeeded"
    assert completed["progress"] == {"metric": 1.0}


def test_job_store_filters_pages_and_cancels_without_deleting_history(
    tmp_path: Path, database_url: str
) -> None:
    store = JobStore(database_url)
    queued = store.create("data_verify", {}, tmp_path / "verify.log")
    cancelled = store.request_cancel(queued["id"])
    assert cancelled["status"] == "cancelled"
    assert cancelled["finished_at"] is not None

    running = store.create("data_snapshot", {}, tmp_path / "snapshot.log")
    claimed = store.claim_next()
    assert claimed and claimed["id"] == running["id"]
    requested = store.request_cancel(running["id"])
    assert requested["status"] == "running"
    assert requested["cancel_requested_at"] is not None
    assert store.cancellation_requested(running["id"]) is True
    store.mark_cancelled(running["id"])

    assert store.count(statuses=("cancelled",)) == 2
    page = store.list(1, offset=1, statuses=("cancelled",))
    assert len(page) == 1
    assert page[0]["id"] == queued["id"]


def test_jobs_api_exposes_filters_total_and_cancel(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    job = JobStore(database_url).create("data_verify", {}, tmp_path / "verify.log")

    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/api/jobs", params={"status": "queued", "limit": 1})
        cancelled = client.post(f"/api/jobs/{job['id']}/cancel")
        filtered = client.get("/api/jobs", params={"status": "cancelled"})

    assert page.status_code == 200
    assert page.headers["x-total-count"] == "1"
    assert page.json()[0]["id"] == job["id"]
    assert cancelled.status_code == 202
    assert cancelled.json()["status"] == "cancelled"
    assert filtered.json()[0]["id"] == job["id"]


def test_worker_cooperatively_terminates_cancelled_child(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    store = JobStore(database_url)
    created = store.create("data_verify", {}, tmp_path / "verify.log")
    claimed = store.claim_next()
    assert claimed and claimed["id"] == created["id"]
    store.request_cancel(created["id"])

    class FakeProcess:
        returncode: int | None = None
        terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            return int(self.returncode or 0)

        def kill(self) -> None:
            self.returncode = -9

    process = FakeProcess()
    monkeypatch.setattr("quant_platform.worker.subprocess.Popen", lambda *args, **kwargs: process)
    settings = Settings(
        api_url="",
        token="",
        data_root=tmp_path / "data",
        database_url=database_url,
    )
    LocalJobWorker(store, tmp_path, settings)._run(claimed)

    assert process.terminated is True
    assert store.get(created["id"])["status"] == "cancelled"


def test_simulation_job_finishes_only_after_ledger_commit(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    store = JobStore(database_url)
    created = store.create(
        "simulation_replay",
        {"simulation_batch_id": "batch-1"},
        tmp_path / "simulation.log",
    )
    claimed = store.claim_next()
    assert claimed and claimed["id"] == created["id"]
    result_path = tmp_path / "result.json"
    events: list[str] = []

    class FakeProcess:
        returncode = 0

        def poll(self) -> int:
            return 0

    def fake_popen(*args, **kwargs) -> FakeProcess:
        del args, kwargs
        result_path.write_text(
            json.dumps(
                {
                    "minute_bars_file": "minute.parquet",
                    "closing_prices": {},
                    "batch_id": "batch-1",
                }
            ),
            encoding="utf-8",
        )
        return FakeProcess()

    class FakeSimulations:
        def process_batch(self, batch_id: str, **kwargs) -> dict:
            del kwargs
            assert batch_id == "batch-1"
            assert store.get(created["id"])["status"] == "running"
            events.append("ledger_committed")
            return {"id": batch_id, "status": "succeeded"}

        def execution_manifest(self, batch_id: str) -> dict:
            assert batch_id == "batch-1"
            return {"source_type": "strategy_version", "source_id": "version-1"}

        def mark_batch_failed(self, batch_id: str, error: str) -> None:
            raise AssertionError(f"unexpected simulation failure {batch_id}: {error}")

    class FakeAllocations:
        def refresh_for_simulation_source(self, source_type: str, source_id: str) -> None:
            assert (source_type, source_id) == ("strategy_version", "version-1")
            events.append("allocation_refreshed")

    settings = Settings(
        api_url="",
        token="",
        data_root=tmp_path / "data",
        database_url=database_url,
    )
    worker = LocalJobWorker(store, tmp_path, settings)
    worker.simulations = FakeSimulations()  # type: ignore[assignment]
    worker.allocations = FakeAllocations()  # type: ignore[assignment]
    monkeypatch.setattr(worker, "_command", lambda job: (["fake"], result_path, {}))
    monkeypatch.setattr("quant_platform.worker.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "quant_platform.worker.pd.read_parquet", lambda path: pd.DataFrame()
    )

    worker._run(claimed)

    assert events == ["ledger_committed", "allocation_refreshed"]
    assert store.get(created["id"])["status"] == "succeeded"


def test_api_keeps_optional_broker_plugin_outside_research_routes(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setenv("BROKER_MODE", "disabled")
    app = create_app(tmp_path)
    with TestClient(app) as client:
        state = client.get("/api/broker")
        settings_route = client.post(
            "/api/settings/broker", json={"gateway_url": "", "hmac_secret": ""}
        )
        capabilities = client.get("/api/capabilities").json()
    assert state.status_code == 404
    assert settings_route.status_code == 404
    assert capabilities["broker_qmt"] is False

    monkeypatch.setenv("BROKER_FEATURE_ENABLED", "true")
    enabled_app = create_app(tmp_path)
    with TestClient(enabled_app) as client:
        enabled = client.get("/api/broker")
        enabled_settings = client.post(
            "/api/settings/broker", json={"gateway_url": "", "hmac_secret": ""}
        )
        enabled_capabilities = client.get("/api/capabilities").json()
    assert enabled.status_code == 404
    assert enabled_settings.status_code == 404
    assert enabled_capabilities["broker_qmt"] is True


def test_api_creates_bounded_rdagent_research_run(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    dataset = data_root / "qlib" / "research-snapshot"
    (dataset / "calendars").mkdir(parents=True)
    (dataset / "instruments").mkdir()
    (dataset / "features").mkdir()
    (dataset / "calendars" / "day.txt").write_text(
        _trading_calendar(date(2018, 1, 1), date(2026, 7, 10)), encoding="utf-8"
    )
    (dataset / "instruments" / "cn_all.txt").write_text(
        "SH600000\t2018-01-01\t2026-07-10\n", encoding="utf-8"
    )
    (dataset / "metadata").mkdir()
    (dataset / "metadata" / "provenance.json").write_text(
        json.dumps(
                {
                    "frequency": "day",
                    "dataset_identity_sha256": "a" * 64,
                    "snapshot_manifest_sha256": "b" * 64,
                    "qlib_builder_sha256": "c" * 64,
                    "dataset_lineage_id": "lineage-research",
                    "field_contract_version": "daily-qlib-field-v3-cny-amount",
                    "source_volume_unit": "hand",
                    "qlib_volume_unit": "share",
                    "source_amount_unit": "thousand_cny",
                    "qlib_amount_unit": "cny",
                    "source_hand_size": 100,
                    "index_volume_policy": "excluded_non_tradable_benchmark",
                    "lineage_verified": True,
                }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setattr(
        "quant_platform.api.probe_rdagent",
        lambda _settings, _root: {"status": "ok", "ready": True, "blockers": []},
    )
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/api/rdagent/runs",
            json={
                "objective": "Find low-turnover quality factors for CSI 300 enhancement.",
                "dataset": "research-snapshot",
                "loop_n": 1,
                "duration": "30m",
                "periods": {
                    "train_start": "2018-01-01",
                    "train_end": "2021-12-31",
                    "valid_start": "2022-01-01",
                    "valid_end": "2023-12-31",
                    "test_start": "2024-01-01",
                    "test_end": "2026-07-10",
                },
            },
        )
        scheduled = client.post(
            "/api/schedules",
            json={
                "name": "daily governed factor research",
                "kind": "rdagent_research",
                "timezone": "Asia/Shanghai",
                "run_time": "20:30",
                "trading_days_only": True,
                "payload": {
                    "objective": "Find low-turnover quality factors for CSI 300 enhancement.",
                    "dataset": "research-snapshot",
                    "loop_n": 1,
                    "duration": "30m",
                    "requested_by": "untrusted-payload-actor",
                    "periods": {
                        "train_start": "2018-01-01",
                        "train_end": "2021-12-31",
                        "valid_start": "2022-01-01",
                        "valid_end": "2023-12-31",
                        "test_start": "2024-01-01",
                        "test_end": "2026-07-10",
                    },
                },
                "misfire_grace_seconds": 1800,
                "actor": "operator",
            },
        )
        runs = client.get("/api/rdagent/runs").json()
        program = client.post(
            "/api/research-programs",
            json={
                "name": "monthly index research",
                "dataset": "research-snapshot",
                "recipe_id": "index_enhancement",
                "loop_n": 1,
                "duration": "30m",
                "min_new_trading_days": 20,
            },
        )
        programs = client.get("/api/research-programs").json()
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["budget"] == {"loop_n": 1, "duration": "30m"}
    assert runs[0]["id"] == response.json()["id"]
    assert scheduled.status_code == 201
    assert scheduled.json()["kind"] == "rdagent_research"
    assert scheduled.json()["payload"]["requested_by"] == "local-admin"
    assert program.status_code == 201
    assert program.json()["dataset_lineage_id"] == "lineage-research"
    assert programs[0]["id"] == program.json()["id"]


def test_api_manages_schedules_and_alert_acknowledgement(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    alert = AlertStore(database_url).create(
        source_type="job",
        source_id="failed-job",
        severity="critical",
        category="job_failure",
        title="Scheduled job failed",
        message="fixture failure",
        dedupe_key="api-alert-fixture",
    )
    app = create_app(tmp_path)
    with TestClient(app) as client:
        created = client.post(
            "/api/schedules",
            json={
                "name": "daily full sync",
                "kind": "incremental_sync",
                "timezone": "Asia/Shanghai",
                "run_time": "18:00",
                "trading_days_only": True,
                "payload": {"profile": "full", "lookback_days": 7, "build_qlib": True},
                "misfire_grace_seconds": 3600,
                "actor": "operator",
            },
        )
        complete = client.post(
            "/api/schedules",
            json={
                "name": "weekly complete data pipeline",
                "kind": "data_pipeline",
                "timezone": "Asia/Shanghai",
                "run_time": "19:00",
                "trading_days_only": True,
                "payload": {
                    "profile": "full",
                    "lookback_days": 14,
                    "bundles": ["cn_extended_daily", "cn_macro", "global_markets"],
                },
                "misfire_grace_seconds": 3600,
                "actor": "operator",
            },
        )
        rejected = client.post(
            "/api/schedules",
            json={
                "name": "unsafe pipeline",
                "kind": "data_pipeline",
                "payload": {"profile": "full", "bundles": ["unknown_bundle"]},
            },
        )
        schedules = client.get("/api/schedules").json()
        acknowledged = client.post(
            f"/api/alerts/{alert['id']}/acknowledge",
            json={"actor": "risk-owner"},
        )
    assert created.status_code == 201
    assert complete.status_code == 201
    assert {item["kind"] for item in schedules} == {"incremental_sync", "data_pipeline"}
    assert rejected.status_code == 422
    assert acknowledged.status_code == 200
    assert acknowledged.json()["status"] == "acknowledged"
