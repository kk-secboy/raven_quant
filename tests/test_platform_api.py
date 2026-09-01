import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from governance_fixtures import governed_etf_ready_evidence

from quant_data.checkpoint import CheckpointStore
from quant_data.config import Settings
from quant_data.execution_contract import DAILY_QLIB_FIELD_CONTRACT_VERSION
from quant_data.history_bounds import GOVERNED_DAILY_STOCK_SCOPE_VERSION
from quant_data.models import FetchSpec
from quant_data.qlib_builder import build_qlib_output_manifest
from quant_platform.alert_store import AlertStore
from quant_platform.api import create_app
from quant_platform.factor_library import compile_qlib_expression
from quant_platform.feature_set_registry import get_feature_set
from quant_platform.job_store import JobStore
from quant_platform.research_horizon import canonical_sha256
from quant_platform.upstream_versions import RDAGENT_COMMIT
from quant_platform.worker import LocalJobWorker


def _trading_calendar(start: date, end: date) -> str:
    current = start
    days: list[str] = []
    while current <= end:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    return "\n".join(days) + "\n"


def test_api_reports_live_work_unit_activity(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("PLATFORM_SECRET_KEY", Fernet.generate_key().decode("ascii"))
    checkpoint = CheckpointStore(database_url)
    spec = FetchSpec(
        dataset="daily",
        api_name="daily",
        scope={"trade_date": "20240102"},
        params={"trade_date": "20240102"},
    )
    checkpoint.add([spec])
    assert checkpoint.claim({"daily"}) is not None

    app = create_app(tmp_path)
    with TestClient(app) as client:
        overview = client.get("/api/overview")
        for _attempt in range(100):
            if overview.json()["running_work_units"] == 1:
                break
            time.sleep(0.02)
            overview = client.get("/api/overview")

    assert overview.status_code == 200
    assert overview.json()["running_work_units"] == 1


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
        pair_strategies = client.get("/api/pair-strategies")
        bootstrap = client.post(
            "/api/jobs/bootstrap",
            json={"profile": "core", "start": "2016-01-01", "end": "latest"},
        )
    assert health.json() == {
        "status": "ok",
        "database": "postgresql",
        "worker_mode": "embedded",
        "runtime_secret_storage": "ok",
        "runtime_secret_records": 0,
    }
    assert overview.status_code == 200, overview.text
    overview_payload = overview.json()
    assert "credentials_configured" in overview_payload, overview_payload
    assert overview_payload["credentials_configured"] is False
    assert overview_payload["readiness_percent"] < 100
    assert overview_payload["actionable_tasks"] > overview_payload["ready_tasks"]
    assert overview_payload["legacy_download_coverage"] == overview_payload["coverage"]
    assert overview_payload["running_work_units"] == 0
    assert readiness.status_code == 200
    assert readiness.json()["profiles"][0]["status"] == "blocked"
    assert any(item["name"] == "daily" for item in datasets.json())
    assert market.status_code == 200
    assert market.json()["status"] == "not_ready"
    assert market.json()["source"]["is_realtime"] is False
    assert pair_strategies.status_code == 200
    assert pair_strategies.json() == []
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
            "total_capital": 1_000_000,
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


def test_failed_job_exposes_later_attempt_even_when_filtered(
    tmp_path: Path, database_url: str
) -> None:
    store = JobStore(database_url)
    payload = {
        "pipeline_id": "pipeline-retry-audit",
        "snapshot_name": "snapshot-retry-audit",
    }
    failed = store.create("bootstrap", payload, tmp_path / "failed.log")
    claimed = store.claim_next()
    assert claimed and claimed["id"] == failed["id"]
    store.finish(failed["id"], exit_code=1, error="old attempt failed")

    successor = store.create("bootstrap", payload, tmp_path / "successor.log")
    claimed_successor = store.claim_next()
    assert claimed_successor and claimed_successor["id"] == successor["id"]

    filtered = store.list(statuses=("failed",))
    old_attempt = next(item for item in filtered if item["id"] == failed["id"])
    assert old_attempt["retry_successor"] == {
        "id": successor["id"],
        "status": "running",
    }
    assert store.get(failed["id"])["retry_successor"]["id"] == successor["id"]

    store.finish(successor["id"], exit_code=0)
    assert store.get(failed["id"])["retry_successor"] == {
        "id": successor["id"],
        "status": "succeeded",
    }


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
    assert enabled_capabilities["broker_qmt"] is False


def test_api_creates_bounded_rdagent_research_run(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    dataset = data_root / "qlib" / "research-snapshot"
    (dataset / "calendars").mkdir(parents=True)
    (dataset / "instruments").mkdir()
    (dataset / "features").mkdir()
    (dataset / "calendars" / "day.txt").write_text(
        _trading_calendar(date(2010, 1, 1), date(2026, 7, 10)), encoding="utf-8"
    )
    (dataset / "instruments" / "cn_all.txt").write_text(
        "SH600000\t2010-01-01\t2026-07-10\n", encoding="utf-8"
    )
    (dataset / "metadata").mkdir()
    feature_set = get_feature_set("governed-baseline")
    required_fields = sorted(
        {
            field
            for expression in feature_set["features"].values()
            for field in compile_qlib_expression(str(expression)).required_fields
        }
    )
    field_year_coverage = {
        "version": "qlib-field-year-source-coverage-v1",
        "source_attribution_policy": "normalized-staging-and-snapshot-contracts-v1",
        "legacy_overlap_policy_version": "overlap-v1",
        "primary_market_history_start": "2010-01-01",
        "fields": {
            field: {
                "source_family": "test",
                "available_from": "2010-01-01",
                "available_to": "2026-07-10",
                "continuous_from": "2010-01-01",
                "research_available_from": "2010-01-01",
                "years": [
                    {
                        "year": 2010,
                        "observed_rows": 1,
                        "non_null_rows": 1,
                        "coverage_ratio": 1.0,
                        "first_session": "2010-01-01",
                        "last_session": "2026-07-10",
                        "source_contracts": ["test-source"],
                    }
                ],
            }
            for field in required_fields
        },
    }
    field_coverage_sha256 = canonical_sha256(field_year_coverage)
    field_year_coverage = {
        **field_year_coverage,
        "coverage_sha256": field_coverage_sha256,
    }
    (dataset / "metadata" / "provenance.json").write_text(
        json.dumps(
            {
                "frequency": "day",
                "dataset_identity_sha256": "a" * 64,
                "snapshot_manifest_sha256": "b" * 64,
                "qlib_builder_sha256": "c" * 64,
                "dataset_lineage_id": "d" * 64,
                "source_lineage_id": "e" * 64,
                "dataset_contract_sha256": "f" * 64,
                "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
                "fields": required_fields,
                "field_units": {field: "normalized" for field in required_fields},
                "research_features": {"version": "pit-research-features-v1"},
                "field_coverage_sha256": field_coverage_sha256,
                "field_year_coverage": field_year_coverage,
                "source_start_date": "2010-01-01",
                "source_end_date": "2026-07-10",
                "source_volume_unit": "hand",
                "qlib_volume_unit": "share",
                "source_amount_unit": "thousand_cny",
                "qlib_amount_unit": "cny",
                "source_hand_size": 100,
                "index_volume_policy": "excluded_non_tradable_benchmark",
                "governed_etf_whitelist": governed_etf_ready_evidence(),
                "lineage_verified": True,
                "output_manifest": build_qlib_output_manifest(dataset),
                "execution_controls": {
                    "native_complete_from": "2010-01-01",
                    "formal_execution_requires_native_controls": True,
                    "scope_version": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setenv(
        "RDAGENT_QLIB_SANDBOX_IMAGE",
        "registry.example/rdagent-qlib@sha256:" + "1" * 64,
    )
    monkeypatch.setenv(
        "MODEL_SANDBOX_IMAGE",
        "registry.example/model-sandbox@sha256:" + "4" * 64,
    )
    runtime_identity = {
        "name": "rdagent",
        "version": "test-runtime",
        "commit": RDAGENT_COMMIT,
        "commit_evidence": ["repository"],
        "source_tree_sha256": "2" * 64,
        "runtime_image_digest": "sha256:" + "3" * 64,
        "repository_dirty": False,
        "production_reproducible": True,
    }
    runtime = {
        **runtime_identity,
        "status": "ok",
        "llm_credentials_configured": True,
        "docker_available": True,
        "qlib_sandbox_preloaded": True,
        "qlib_smoke_passed": True,
        "evaluation_worker": {
            "ready": True,
            "job_kinds": [
                "factor_evaluate",
                "model_evaluate",
                "quant_bundle_evaluate",
            ],
            "model_sandbox_ready": True,
            "model_sandbox_config_sha256": (
                "71c2f13d9116c3749bf31344038ecfa02d9d905ef89fcf2ee6811aa26243690d"
            ),
        },
        "runtime_identity": runtime_identity,
    }
    monkeypatch.setattr(
        "quant_platform.api.probe_rdagent",
        lambda _settings, _root: runtime,
    )
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/api/rdagent/runs",
            json={
                "objective": "Find low-turnover quality factors for CSI 300 enhancement.",
                "scenario": "fin_quant",
                "dataset": "research-snapshot",
                "feature_set_id": "governed-baseline",
                "horizon": "short",
                "loop_n": 1,
                "duration": "30m",
            },
        )
        frozen_run = client.post(
            "/api/rdagent/runs",
            json={
                "objective": "Frozen scenarios must not accept new runs.",
                "scenario": "fin_factor",
                "dataset": "research-snapshot",
                "feature_set_id": "governed-baseline",
                "horizon": "short",
                "loop_n": 1,
                "duration": "30m",
            },
        )
        frozen_model_run = client.post(
            "/api/rdagent/runs",
            json={
                "objective": "Frozen scenarios must not accept new runs.",
                "scenario": "fin_model",
                "dataset": "research-snapshot",
                "feature_set_id": "governed-baseline",
                "horizon": "short",
                "loop_n": 1,
                "duration": "30m",
            },
        )
        frozen_schedule = client.post(
            "/api/schedules",
            json={
                "name": "frozen scenario schedule",
                "kind": "rdagent_research",
                "timezone": "Asia/Shanghai",
                "run_time": "20:30",
                "trading_days_only": True,
                "payload": {
                    "objective": "Frozen scenarios must not accept new schedules.",
                    "scenario": "fin_factor",
                    "dataset": "research-snapshot",
                    "feature_set_id": "governed-baseline",
                    "horizon": "short",
                    "loop_n": 1,
                    "duration": "30m",
                },
                "misfire_grace_seconds": 1800,
                "actor": "operator",
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
                    "scenario": "fin_quant",
                    "dataset": "research-snapshot",
                    "feature_set_id": "governed-baseline",
                    "horizon": "short",
                    "loop_n": 1,
                    "duration": "30m",
                    "requested_by": "untrusted-payload-actor",
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
            },
        )
        programs = client.get("/api/research-programs").json()
        retired_campaign_without_payload = client.post("/api/research-campaigns")
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["budget"] == {"loop_n": 1, "duration": "30m"}
    assert runs[0]["id"] == response.json()["id"]
    assert scheduled.status_code == 201
    assert scheduled.json()["kind"] == "rdagent_research"
    assert scheduled.json()["payload"]["requested_by"] == "local-admin"
    assert frozen_run.status_code == 410
    assert "frozen" in frozen_run.json()["detail"]
    assert frozen_model_run.status_code == 410
    assert "frozen" in frozen_model_run.json()["detail"]
    assert frozen_schedule.status_code == 410
    assert "frozen" in frozen_schedule.json()["detail"]
    assert program.status_code == 410
    assert "legacy research programs are retired" in program.json()["detail"]
    assert programs == []
    assert retired_campaign_without_payload.status_code == 410
    assert "legacy research campaigns are retired" in (
        retired_campaign_without_payload.json()["detail"]
    )


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
    assert {"incremental_sync", "data_pipeline"} <= {
        item["kind"] for item in schedules
    }
    assert rejected.status_code == 422
    assert acknowledged.status_code == 200
    assert acknowledged.json()["status"] == "acknowledged"
