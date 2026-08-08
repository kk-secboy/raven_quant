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
    assert tasks[-1]["task_key"] == "strategy_specialty_minutes"
    assert tasks[1]["task_key"] == "cn_data_verify"
    assert tasks[1]["depends_on"] == ["cn_ashare_daily_full"]
    assert tasks[5]["depends_on"] == ["cn_qlib_baseline"]
    assert tasks[5]["dependencies_satisfied"] is False
    assert all(item["source"] in {"Tushare", "QuantLab", "Qlib"} for item in tasks)
    assert all(item["config"]["frequency"] != "tick" for item in tasks)
    assert "tick_level2" not in {item["task_key"] for item in tasks}


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


def test_catalog_binds_cninfo_announcement_download(database_url: str, tmp_path: Path) -> None:
    job = JobStore(database_url).create(
        "cninfo_announcements_download",
        {"start": "2016-01-01", "end": "2026-08-03"},
        tmp_path / "announcements.log",
    )

    store = DataTaskStore(database_url)
    store.sync_catalog()
    task = next(item for item in store.list() if item["task_key"] == "cn_cninfo_announcements")

    assert task["job_id"] == job["id"]
    assert task["status"] == "queued"


def test_structured_information_card_tracks_latest_refresh_stage(
    database_url: str, tmp_path: Path
) -> None:
    jobs = JobStore(database_url)
    producer = jobs.create(
        "report_rc_factors",
        {
            "profile": "information_factor_refresh",
            "start": "2010-01-01",
            "end": "2026-08-08",
        },
        tmp_path / "report-rc-factors.log",
    )
    jobs.finish(producer["id"], exit_code=0)
    register = jobs.create(
        "report_rc_factor_register",
        {
            "profile": "information_factor_refresh",
            "start": "2010-01-01",
            "end": "2026-08-08",
        },
        tmp_path / "report-rc-register.log",
    )

    store = DataTaskStore(database_url)
    store.sync_catalog()
    task = next(
        item
        for item in store.list()
        if item["task_key"] == "cn_structured_information_factors"
    )

    assert task["job_id"] == register["id"]
    assert task["status"] == "queued"
    assert task["coverage"] == 0.0


def test_unrelated_information_evaluation_does_not_bind_refresh_card(
    database_url: str, tmp_path: Path
) -> None:
    job = JobStore(database_url).create(
        "information_factor_evaluate",
        {
            "profile": "manual_research",
            "start": "2010-01-01",
            "end": "2026-08-08",
        },
        tmp_path / "manual-evaluation.log",
    )

    store = DataTaskStore(database_url)
    store.sync_catalog()
    task = next(
        item
        for item in store.list()
        if item["task_key"] == "cn_structured_information_factors"
    )

    assert task["job_id"] is None
    assert task["status"] == "planned"
    assert job["status"] == "queued"


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
    assert len(tasks) == len(DATA_TASK_CATALOG)
    assert next(item for item in tasks if item["task_key"] == "hk_market")
    assert next(item for item in tasks if item["task_key"] == "cn_funds")
    assert next(item for item in tasks if item["task_key"] == "global_markets")
    assert next(item for item in tasks if item["task_key"] == "cn_capital_flow")
    assert next(item for item in tasks if item["task_key"] == "research_corpus")
    institutional = next(
        item for item in tasks if item["task_key"] == "cn_institutional"
    )
    assert {
        "report_rc",
        "etf_sh_cons",
        "etf_sz_cons",
        "ci_daily",
        "shibor_quote",
        "major_news",
    } <= set(institutional["config"]["datasets"])
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
    minute_qlib = next(
        item for item in tasks if item["task_key"] == "liquid_intraday_qlib"
    )
    assert minute_qlib["depends_on"] == ["liquid_intraday_1m"]
    ashare_5m = next(item for item in tasks if item["task_key"] == "cn_ashare_5m")
    assert ashare_5m["implementation_status"] == "ready"
    ashare_5m_qlib = next(
        item for item in tasks if item["task_key"] == "cn_ashare_5m_qlib"
    )
    assert ashare_5m_qlib["depends_on"] == ["cn_ashare_5m"]
    assert next(item for item in tasks if item["task_key"] == "cn_futures")[
        "depends_on"
    ] == ["cn_macro"]
    assert (
        next(item for item in tasks if item["task_key"] == "cn_extended_daily")[
            "implementation_status"
        ]
        == "ready"
    )


def test_minute_qlib_job_updates_its_own_task_card(
    database_url: str, tmp_path: Path
) -> None:
    jobs = JobStore(database_url)
    job = jobs.create(
        "minute_qlib",
        {
            "snapshot_name": "execution-fixture",
            "output_name": "execution-fixture-1min",
        },
        tmp_path / "minute-qlib.log",
    )
    store = DataTaskStore(database_url)
    store.sync_catalog()

    tasks = {item["task_key"]: item for item in store.list()}
    assert tasks["liquid_intraday_qlib"]["job_id"] == job["id"]
    assert tasks["liquid_intraday_qlib"]["status"] == "queued"
    assert tasks["liquid_intraday_1m"]["job_id"] is None


def test_five_minute_qlib_job_updates_five_minute_task_card(
    database_url: str, tmp_path: Path
) -> None:
    jobs = JobStore(database_url)
    job = jobs.create(
        "minute_qlib",
        {
            "snapshot_name": "ashare-fixture",
            "output_name": "ashare-fixture-5min",
            "frequency": "5min",
        },
        tmp_path / "minute-qlib-5min.log",
    )
    store = DataTaskStore(database_url)
    store.sync_catalog()

    tasks = {item["task_key"]: item for item in store.list()}
    assert tasks["cn_ashare_5m_qlib"]["job_id"] == job["id"]
    assert tasks["cn_ashare_5m_qlib"]["status"] == "queued"
    assert tasks["liquid_intraday_qlib"]["job_id"] is None


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


def test_verified_current_plan_ignores_superseded_failed_units(
    database_url: str, tmp_path: Path
) -> None:
    checkpoint = CheckpointStore(database_url)
    obsolete = FetchSpec(
        dataset="fund_basic",
        api_name="fund_basic",
        scope={"market": "O", "status": "L", "row_limit": 15_000},
        params={"market": "O", "status": "L"},
        max_attempts=1,
    )
    checkpoint.add([obsolete])
    unit = checkpoint.claim({"fund_basic"})
    assert unit is not None
    checkpoint.fail(unit.unit_key, "provider row limit reached", terminal=True)

    jobs = JobStore(database_url)
    job = jobs.create(
        "supplemental_cn_funds",
        {"bundle": "cn_funds", "start": "2024-01-01", "end": "2024-01-31"},
        tmp_path / "funds.log",
    )
    datasets = {
        name: {"units": 1, "rows": 1} for name in bundle_datasets("cn_funds")
    }
    jobs.finish(
        job["id"],
        exit_code=0,
        result={
            "status": "succeeded",
            "bundle": "cn_funds",
            "datasets": datasets,
            "pagination_verified": True,
        },
    )

    store = DataTaskStore(database_url)
    store.sync_catalog()
    task = next(item for item in store.list() if item["task_key"] == "cn_funds")

    assert task["status"] == "succeeded"
    assert task["coverage"] == 100.0


def test_running_supplemental_progress_accepts_dataset_name_list(
    database_url: str, tmp_path: Path
) -> None:
    jobs = JobStore(database_url)
    job = jobs.create(
        "supplemental_cn_funds",
        {"bundle": "cn_funds", "start": "2024-01-01", "end": "2024-01-31"},
        tmp_path / "funds.log",
    )
    claimed = jobs.claim_next(("supplemental_cn_funds",))
    assert claimed is not None
    assert claimed["id"] == job["id"]
    jobs.update_progress(
        job["id"],
        {
            "status": "running",
            "datasets": sorted(bundle_datasets("cn_funds")),
            "execution_phase": "downloading",
        },
    )

    store = DataTaskStore(database_url)
    store.sync_catalog()
    task = next(item for item in store.list() if item["task_key"] == "cn_funds")

    assert task["status"] == "running"
    assert task["progress"]["datasets"] == sorted(bundle_datasets("cn_funds"))


def test_task_card_exposes_adaptive_checkpoint_and_cooldown_state(
    database_url: str, tmp_path: Path
) -> None:
    checkpoint = CheckpointStore(database_url)
    retrying = FetchSpec(
        dataset="etf_sh_cons",
        api_name="etf_sh_cons",
        scope={"ts_code": "510300.SH", "partition_start": "2024-01-01"},
        params={"ts_code": "510300.SH", "start_date": "20240101"},
        max_attempts=3,
    )
    obsolete = FetchSpec(
        dataset="etf_sh_cons",
        api_name="etf_sh_cons",
        scope={"ts_code": "510300.SH", "trade_date": "20240102"},
        params={"ts_code": "510300.SH", "trade_date": "20240102"},
        max_attempts=3,
    )
    checkpoint.add([retrying])
    claimed = checkpoint.claim({"etf_sh_cons"})
    assert claimed is not None
    checkpoint.fail(
        claimed.unit_key,
        "provider rate limit: requests per minute exceeded",
        retry_after_seconds=180,
    )
    checkpoint.add([obsolete])
    checkpoint.supersede_units([obsolete.unit_key], "repartitioned by adaptive child windows")

    jobs = JobStore(database_url)
    job = jobs.create(
        "supplemental_cn_institutional",
        {
            "bundle": "cn_institutional",
            "start": "2024-01-01",
            "end": "2024-01-31",
        },
        tmp_path / "institutional.log",
    )
    assert jobs.claim_next()["id"] == job["id"]

    store = DataTaskStore(database_url)
    store.sync_catalog()
    task = next(item for item in store.list() if item["task_key"] == "cn_institutional")

    assert task["execution_phase"] == "rate_limit_cooldown"
    assert task["unit_stats"]["retry_waiting"] == 1
    assert task["unit_stats"]["rate_limited"] == 1
    assert task["unit_stats"]["superseded"] == 1
    assert task["unit_stats"]["next_retry_at"] is not None
    assert "整段历史" in task["config"]["request_strategy"]
