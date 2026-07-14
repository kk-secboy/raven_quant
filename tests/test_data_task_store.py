from pathlib import Path

from fastapi.testclient import TestClient

from quant_data.checkpoint import CheckpointStore
from quant_data.models import FetchSpec, UnitResult
from quant_data.supplemental_data import SUPPORTED_BUNDLES, bundle_datasets
from quant_platform.api import create_app
from quant_platform.data_task_store import DATA_TASK_CATALOG, DataTaskStore
from quant_platform.job_store import JobStore


def test_catalog_is_ordered_and_dependency_aware(database_url: str) -> None:
    store = DataTaskStore(database_url)
    store.sync_catalog()

    tasks = store.list()
    assert tasks[0]["task_key"] == "cn_ashare_daily_full"
    assert tasks[-1]["task_key"] == "tick_level2"
    assert tasks[1]["task_key"] == "cn_data_verify"
    assert tasks[1]["depends_on"] == ["cn_ashare_daily_full"]
    assert tasks[5]["depends_on"] == ["cn_qlib_baseline"]
    assert tasks[5]["dependencies_satisfied"] is False
    assert tasks[-1]["implementation_status"] == "external_source_required"


def test_supplemental_task_catalog_matches_real_downloader_contracts() -> None:
    definitions = {item.task_key: item for item in DATA_TASK_CATALOG}
    assert SUPPORTED_BUNDLES <= definitions.keys()
    for bundle in SUPPORTED_BUNDLES:
        assert set(definitions[bundle].datasets) == bundle_datasets(bundle)


def test_catalog_binds_the_running_bootstrap(database_url: str, tmp_path: Path) -> None:
    jobs = JobStore(database_url)
    job = jobs.create(
        "bootstrap",
        {"profile": "full", "start": "2024-01-01", "end": "latest"},
        tmp_path / "bootstrap.log",
    )
    jobs.claim_next()
    store = DataTaskStore(database_url)
    store.sync_catalog()

    current = store.list()[0]
    assert current["job_id"] == job["id"]
    assert current["status"] == "running"


def test_core_intraday_job_updates_full_and_pair_minute_cards(
    database_url: str, tmp_path: Path
) -> None:
    jobs = JobStore(database_url)
    job = jobs.create(
        "core_intraday_download",
        {"start": "2026-07-10", "end": "2026-07-10", "auto_select": True},
        tmp_path / "intraday.log",
    )
    store = DataTaskStore(database_url)
    store.sync_catalog()

    tasks = {item["task_key"]: item for item in store.list()}
    assert tasks["pair_execution_1m"]["job_id"] == job["id"]
    assert tasks["liquid_intraday_1m"]["job_id"] == job["id"]
    assert tasks["liquid_intraday_1m"]["status"] == "queued"


def test_catalog_tracks_each_finalize_pipeline_stage(database_url: str, tmp_path: Path) -> None:
    store = DataTaskStore(database_url)
    store.sync_catalog()
    jobs = JobStore(database_url)
    verify = jobs.create(
        "data_verify",
        {"snapshot_name": "fixture"},
        tmp_path / "verify.log",
    )

    tasks = {item["task_key"]: item for item in store.list()}
    assert tasks["cn_data_verify"]["job_id"] == verify["id"]
    assert tasks["cn_data_verify"]["status"] == "queued"
    assert tasks["cn_snapshot_build"]["status"] == "planned"


def test_newer_pipeline_stage_supersedes_legacy_bootstrap_failure(
    database_url: str, tmp_path: Path
) -> None:
    jobs = JobStore(database_url)
    bootstrap = jobs.create(
        "bootstrap",
        {"profile": "full", "start": "2024-01-01", "end": "2026-07-12"},
        tmp_path / "bootstrap.log",
    )
    jobs.finish(bootstrap["id"], exit_code=1, error="legacy Qlib validation failed")
    snapshot = jobs.create(
        "data_snapshot",
        {"snapshot_name": "current"},
        tmp_path / "snapshot.log",
    )
    jobs.finish(snapshot["id"], exit_code=0)

    store = DataTaskStore(database_url)
    store.sync_catalog()
    tasks = {item["task_key"]: item for item in store.list()}

    assert tasks["cn_ashare_daily_full"]["status"] == "succeeded"
    assert tasks["cn_ashare_daily_full"]["job_id"] is None
    assert tasks["cn_data_verify"]["status"] == "succeeded"
    assert tasks["cn_snapshot_build"]["status"] == "succeeded"
    assert tasks["cn_snapshot_build"]["job_id"] == snapshot["id"]


def test_api_exposes_persistent_data_tasks(database_url: str, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/data-tasks")

    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) == 18
    assert next(item for item in tasks if item["task_key"] == "hk_market")
    assert next(item for item in tasks if item["task_key"] == "cn_funds")
    assert next(item for item in tasks if item["task_key"] == "global_markets")
    assert {"stock_st", "sw_daily"} <= set(
        next(item for item in tasks if item["task_key"] == "cn_extended_daily")["config"][
            "datasets"
        ]
    )
    assert {"etf_basic", "etf_index"} <= set(
        next(item for item in tasks if item["task_key"] == "cn_funds")["config"]["datasets"]
    )
    assert (
        "ft_limit"
        in next(item for item in tasks if item["task_key"] == "cn_futures")["config"]["datasets"]
    )
    assert (
        "cn_schedule"
        in next(item for item in tasks if item["task_key"] == "cn_macro")["config"]["datasets"]
    )
    assert (
        "us_tycr"
        in next(item for item in tasks if item["task_key"] == "global_markets")["config"][
            "datasets"
        ]
    )
    margin = next(item for item in tasks if item["task_key"] == "cn_margin_eligibility")
    assert margin["implementation_status"] == "ready"
    minute = next(item for item in tasks if item["task_key"] == "liquid_intraday_1m")
    assert minute["depends_on"] == ["cn_options_bonds", "cn_margin_eligibility"]
    assert (
        next(item for item in tasks if item["task_key"] == "cn_extended_daily")[
            "implementation_status"
        ]
        == "ready"
    )


def test_task_card_exposes_failure_reason_and_retry_identity(
    database_url: str, tmp_path: Path
) -> None:
    jobs = JobStore(database_url)
    job = jobs.create(
        "supplemental_cn_funds",
        {"bundle": "cn_funds", "start": "2024-01-01", "end": "2024-01-31"},
        tmp_path / "funds.log",
    )
    jobs.finish(job["id"], exit_code=1, error="provider permission denied")

    store = DataTaskStore(database_url)
    store.sync_catalog()
    task = next(item for item in store.list() if item["task_key"] == "cn_funds")

    assert task["status"] == "failed"
    assert task["job_id"] == job["id"]
    assert task["error"] == "provider permission denied"
    assert task["coverage"] == 0.0


def test_legacy_success_is_partial_when_new_interfaces_are_missing(
    database_url: str, tmp_path: Path
) -> None:
    checkpoint = CheckpointStore(database_url)
    spec = FetchSpec(
        dataset="fund_company",
        api_name="fund_company",
        scope={"all": True},
        params={},
    )
    checkpoint.add([spec])
    unit = checkpoint.claim({"fund_company"})
    assert unit is not None
    checkpoint.succeed(
        unit.unit_key,
        UnitResult("units/fund_company/fixture.parquet", 10, "a" * 64),
    )
    jobs = JobStore(database_url)
    job = jobs.create(
        "supplemental_cn_funds",
        {"bundle": "cn_funds", "start": "2024-01-01", "end": "2024-01-31"},
        tmp_path / "funds.log",
    )
    jobs.finish(job["id"], exit_code=0)

    store = DataTaskStore(database_url)
    store.sync_catalog()
    task = next(item for item in store.list() if item["task_key"] == "cn_funds")

    assert task["status"] == "partial"
    assert task["coverage"] == 11.1
