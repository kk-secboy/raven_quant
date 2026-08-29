from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from governance_fixtures import governed_etf_ready_evidence

import quant_platform.deployment_readiness as readiness_module
from quant_data.config import Settings
from quant_data.execution_contract import DAILY_QLIB_FIELD_CONTRACT_VERSION
from quant_data.history_bounds import GOVERNED_DAILY_STOCK_SCOPE_VERSION
from quant_data.qlib_builder import build_qlib_output_manifest
from quant_platform.alpha_spending_ledger import CAPITAL_OOS_MIN_EMBARGO_TRADING_DAYS
from quant_platform.auth_store import AuthStore
from quant_platform.data_automation import DEFAULT_STRATEGY_MINUTE_SYMBOLS
from quant_platform.data_task_store import DataTaskStore
from quant_platform.deployment_readiness import (
    RESEARCH_MINIMUM_TRADING_DAYS,
    DeploymentReadinessStore,
)
from quant_platform.health_store import OperationalHealthStore
from quant_platform.job_store import JobStore
from quant_platform.model_research_governance import (
    MODEL_LABEL_HORIZON_TRADING_DAYS,
)
from quant_platform.research_automation import (
    DEFAULT_RESEARCH_PERIOD_POLICY,
    MINIMUM_PROFILE_TRAINING_DAYS,
    RESEARCH_EVALUATION_PROFILES,
)
from quant_platform.runtime_secret_store import RuntimeSecretStore
from quant_platform.schedule_store import ScheduleStore
from quant_platform.scheduler import AUTOMATED_DATA_BUNDLES
from quant_platform.services import refresh_qlib_display_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.no_database
def test_governed_schedule_suite_cardinality_matches_all_five_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Row:
        def __init__(self, kind: str) -> None:
            self.id = f"schedule-{kind}"
            self.kind = kind

    for validator in (
        "_is_governed_suite_data_pipeline",
        "_is_governed_information_pipeline",
        "_is_governed_information_factor_refresh",
        "_is_governed_ashare_5m_sync",
        "_is_governed_auxiliary_data_pipeline",
    ):
        monkeypatch.setattr(
            readiness_module,
            validator,
            lambda *_args, **_kwargs: True,
        )
    rows = [
        Row(kind)
        for kind in sorted(readiness_module._GOVERNED_DATA_SCHEDULE_SUITE_KINDS)
    ]

    ready, governed = readiness_module._governed_schedule_suite_state(
        rows,
        data_root=tmp_path,
        reproducible_dataset_names=set(),
    )

    assert ready is True
    assert set(governed) == readiness_module._GOVERNED_DATA_SCHEDULE_SUITE_KINDS
    assert all(len(ids) == 1 for ids in governed.values())


@pytest.mark.no_database
def test_latest_closed_trading_day_excludes_an_unfinished_session() -> None:
    open_days = [date(2026, 8, 28), date(2026, 8, 31)]

    before_close = readiness_module._latest_closed_trading_day(
        open_days,
        now=datetime(2026, 8, 31, 6, 59, tzinfo=UTC),
    )
    after_close = readiness_module._latest_closed_trading_day(
        open_days,
        now=datetime(2026, 8, 31, 7, 1, tzinfo=UTC),
    )

    assert before_close == date(2026, 8, 28)
    assert after_close == date(2026, 8, 31)


@pytest.mark.no_database
def test_daily_business_check_requires_fresh_sealed_daily_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = {
        "frequency": "day",
        "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
        "source_volume_unit": "hand",
        "qlib_volume_unit": "share",
        "source_amount_unit": "thousand_cny",
        "qlib_amount_unit": "cny",
        "source_hand_size": 100,
        "index_volume_policy": "excluded_non_tradable_benchmark",
        "governed_etf_whitelist": governed_etf_ready_evidence(),
        "lineage_verified": True,
        "execution_controls": {
            "formal_execution_requires_native_controls": True,
            "scope_version": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
            "native_complete_from": "2016-01-04",
        },
    }
    dataset = {
        "name": "daily-production",
        "frequency": "day",
        "ready": True,
        "reproducible": True,
        "lineage_verified": True,
        "output_files_verified": True,
        "output_verification": "verified",
        "end_date": "2026-08-31",
        "daily_contract": provenance,
    }
    monkeypatch.setattr(
        readiness_module,
        "load_trade_calendar_open_days",
        lambda _root: [date(2026, 8, 28), date(2026, 8, 31)],
    )
    monkeypatch.setattr(
        readiness_module,
        "list_qlib_datasets_for_display",
        lambda _root: [dataset],
    )

    fresh = readiness_module._daily_qlib_business_check(
        tmp_path,
        now=datetime(2026, 8, 31, 8, 0, tzinfo=UTC),
    )
    dataset["end_date"] = "2026-08-28"
    stale = readiness_module._daily_qlib_business_check(
        tmp_path,
        now=datetime(2026, 8, 31, 8, 0, tzinfo=UTC),
    )

    assert fresh["status"] == "ok"
    assert fresh["expected_end_date"] == "2026-08-31"
    assert stale["status"] == "blocked"
    assert stale["datasets"][0]["reasons"] == ["stale"]


@pytest.mark.no_database
def test_horizon_readiness_requires_fresh_health_for_paper_and_recommendation() -> None:
    paper = {
        "strategy_version_id": "paper-short",
        "promotion_stage": "paper",
        "signal_frequency": "day",
        "execution_frequency": "day",
        "contract_ready": True,
        "health_status": "healthy",
        "health_evidence_ready": True,
        "health_evidence_reasons": [],
        "paper_stage_status": "active",
        "simulation_status": "active",
        "active_recommendation_portfolios": 0,
    }

    validating = readiness_module._assess_horizon_candidates(
        "short_1_5d",
        [paper],
    )
    unhealthy_recommendation = readiness_module._assess_horizon_candidates(
        "short_1_5d",
        [
            paper,
            {
                **paper,
                "strategy_version_id": "recommendation-short",
                "promotion_stage": "recommendation_enabled",
                "health_status": "suspended",
                "health_evidence_ready": True,
                "active_recommendation_portfolios": 1,
            },
        ],
    )

    assert validating["status"] == "ok"
    assert validating["stage"] == "paper"
    assert validating["health_status"] == "healthy"
    assert unhealthy_recommendation["status"] == "blocked"
    assert unhealthy_recommendation["candidates"][0]["blocking_reasons"] == [
        "strategy_health_suspended"
    ]

    paper["health_evidence_ready"] = False
    paper["health_evidence_reasons"] = ["strategy_health_evidence_stale"]
    stale = readiness_module._assess_horizon_candidates("short_1_5d", [paper])
    assert stale["status"] == "blocked"
    assert stale["candidates"][0]["blocking_reasons"] == [
        "strategy_health_evidence_stale"
    ]


def _settings(monkeypatch, database_url: str, data_root: Path, *, auth_mode: str) -> Settings:
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("AUTH_MODE", auth_mode)
    monkeypatch.setenv("TUSHARE_API_URL", "https://api.tushare.pro")
    monkeypatch.setenv("TUSHARE_TOKEN", "verified-test-token")
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setenv("BROKER_MODE", "disabled")
    monkeypatch.setenv("PLATFORM_SECRET_KEY", Fernet.generate_key().decode("ascii"))
    return Settings.from_env(PROJECT_ROOT / ".env.missing")


def _qlib_dataset(
    data_root: Path, *, trading_days: int = RESEARCH_MINIMUM_TRADING_DAYS
) -> None:
    target = data_root / "qlib" / "acceptance-snapshot"
    (target / "calendars").mkdir(parents=True)
    (target / "instruments").mkdir()
    (target / "features").mkdir()
    (target / "metadata").mkdir()
    start = date(2026, 7, 31) - timedelta(days=trading_days - 1)
    days = [(start + timedelta(days=index)).isoformat() for index in range(trading_days)]
    (target / "calendars" / "day.txt").write_text("\n".join(days) + "\n", encoding="utf-8")
    (target / "instruments" / "cn_all.txt").write_text(
        f"SH600000\t{days[0]}\t{days[-1]}\n", encoding="utf-8"
    )
    (target / "metadata" / "provenance.json").write_text(
        json.dumps(
            {
                "frequency": "day",
                "dataset_identity_sha256": "a" * 64,
                "snapshot_manifest_sha256": "b" * 64,
                "dataset_lineage_id": "c" * 64,
                "source_lineage_id": "d" * 64,
                "lineage_verified": True,
                "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
                "source_volume_unit": "hand",
                "qlib_volume_unit": "share",
                "source_amount_unit": "thousand_cny",
                "qlib_amount_unit": "cny",
                "source_hand_size": 100,
                "index_volume_policy": "excluded_non_tradable_benchmark",
                "governed_etf_whitelist": governed_etf_ready_evidence(),
                "output_manifest": build_qlib_output_manifest(target),
            }
        ),
        encoding="utf-8",
    )
    refresh_qlib_display_catalog(data_root)


def test_empty_deployment_is_fail_closed(tmp_path: Path, monkeypatch, database_url: str) -> None:
    settings = _settings(monkeypatch, database_url, tmp_path / "data", auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()

    result = DeploymentReadinessStore(settings, PROJECT_ROOT).assess()

    assert result["highest_ready_profile"] is None
    assert result["live_trading_supported"] is False
    research = result["profiles"][0]
    assert research["status"] == "blocked"
    blocked = {item["id"] for item in research["checks"] if item["status"] == "block"}
    assert {
        "authentication_enabled",
        "tushare_verified",
        "initialization_pipeline",
        "reproducible_qlib_dataset",
        "operational_health",
        "rdagent_runtime",
        "incremental_schedule",
    }.issubset(blocked)


def test_research_dataset_threshold_matches_multi_profile_contract() -> None:
    assert (
        DEFAULT_RESEARCH_PERIOD_POLICY["embargo_trading_days"]
        == CAPITAL_OOS_MIN_EMBARGO_TRADING_DAYS
    )
    assert RESEARCH_MINIMUM_TRADING_DAYS == (
        MINIMUM_PROFILE_TRAINING_DAYS
        + MODEL_LABEL_HORIZON_TRADING_DAYS
        + max(
            int(profile["validation_trading_days"])
            for profile in RESEARCH_EVALUATION_PROFILES
        )
        + CAPITAL_OOS_MIN_EMBARGO_TRADING_DAYS
        + DEFAULT_RESEARCH_PERIOD_POLICY["test_trading_days"]
    )


def _data_schedule_check(settings: Settings) -> dict:
    result = DeploymentReadinessStore(settings, PROJECT_ROOT).assess()
    return next(
        item
        for item in result["profiles"][0]["checks"]
        if item["id"] == "incremental_schedule"
    )


def _create_data_pipeline_schedule(
    database_url: str,
    *,
    name: str,
    payload: dict,
) -> None:
    ScheduleStore(database_url).create(
        name=name,
        kind="data_pipeline",
        timezone="Asia/Shanghai",
        run_time=time(18, 0),
        trading_days_only=True,
        payload=payload,
        misfire_grace_seconds=3600,
        actor="admin",
    )


def _create_incremental_schedule(
    database_url: str,
    *,
    name: str,
    payload: dict,
) -> None:
    ScheduleStore(database_url).create(
        name=name,
        kind="incremental_sync",
        timezone="Asia/Shanghai",
        run_time=time(18, 0),
        trading_days_only=True,
        payload=payload,
        misfire_grace_seconds=3600,
        actor="admin",
    )


def _create_governed_schedule_suite(
    database_url: str,
    *,
    minute_run_time: time = time(23, 30),
    short_final_oos: bool = False,
    include_npr: bool = False,
) -> None:
    evaluation = {
        "dataset": "acceptance-snapshot",
        "periods": {
            "train_start": "2019-01-01",
            "train_end": "2019-12-31",
            "valid_start": "2020-01-01",
            "valid_end": "2022-12-31",
            "test_start": "2026-06-01" if short_final_oos else "2023-01-09",
            "test_end": "2026-07-31",
        },
        "universe": "cn_all",
        "benchmark": "SH000300",
    }
    contracts = (
        (
            "governed daily raw data and Qlib publication",
            "data_pipeline",
            time(18, 0),
            True,
            {
                "profile": "full",
                "snapshot_start": "2008-01-01",
                "lookback_days": 7,
                "bundles": list(AUTOMATED_DATA_BUNDLES),
            },
        ),
        (
            "governed daily A-share five-minute publication",
            "ashare_5m_sync",
            minute_run_time,
            True,
            {"history_start": "2024-01-01", "lookback_days": 3},
        ),
        (
            "governed daily bounded information NLP",
            "information_pipeline",
            time(2, 0),
            False,
            {
                "lookback_days": 7,
                "regulatory_only": True,
                "download_limit": 0,
                "enable_nlp": True,
                "announcement_categories": ["regulatory_letter"],
                "announcement_nlp_limit": 500,
                "include_corpus_nlp": True,
                "corpus_datasets": [
                    "cctv_news",
                    "irm_qa_sh",
                    "irm_qa_sz",
                    "major_news",
                    *(["npr"] if include_npr else []),
                ],
                "corpus_nlp_limit": 500,
                "batch_size": 50,
                "major_news_per_day": 40,
                "irm_per_instrument_day": 2,
                "include_event_labels": True,
                "include_factor_evaluation": False,
                "horizons": [1, 3, 5, 20],
                "benchmark_code": "000300.SH",
            },
        ),
        (
            "governed weekly structured information factors",
            "information_factor_refresh",
            time(12, 30),
            False,
            {
                "sources": ["major_news_mentions", "news_flash", "report_rc"],
                "weekday": 4,
                "factor_evaluation": evaluation,
            },
        ),
        (
            "governed daily auxiliary research data publication",
            "auxiliary_data_pipeline",
            time(4, 0),
            False,
            {
                "history_start": "2024-01-01",
                "max_stocks": 100,
                "max_options": 100,
                "strategy_minute_symbols": list(DEFAULT_STRATEGY_MINUTE_SYMBOLS),
            },
        ),
    )
    for name, kind, run_time, trading_days_only, payload in contracts:
        ScheduleStore(database_url).create(
            name=name,
            kind=kind,
            timezone="Asia/Shanghai",
            run_time=run_time,
            trading_days_only=trading_days_only,
            payload=payload,
            misfire_grace_seconds=7200,
            actor="admin",
        )


def test_readiness_rejects_reproducible_dataset_below_research_history_contract(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    settings = _settings(monkeypatch, database_url, data_root, auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _qlib_dataset(data_root, trading_days=RESEARCH_MINIMUM_TRADING_DAYS - 1)

    result = DeploymentReadinessStore(settings, PROJECT_ROOT).assess()

    check = next(
        item
        for item in result["profiles"][0]["checks"]
        if item["id"] == "reproducible_qlib_dataset"
    )
    assert check["status"] == "block"
    assert check["details"]["minimum_trading_days"] == RESEARCH_MINIMUM_TRADING_DAYS
    assert check["details"]["maximum_available_trading_days"] == (
        RESEARCH_MINIMUM_TRADING_DAYS - 1
    )


def test_readiness_requires_runtime_secrets_to_be_decryptable(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    settings = _settings(monkeypatch, database_url, tmp_path / "data", auth_mode="required")
    RuntimeSecretStore(database_url, settings.platform_secret_key).put(
        "tushare",
        {"api_url": "https://api.tushare.pro", "token": "database-token"},
        metadata={
            "api_url": "https://api.tushare.pro",
            "verified_at": datetime.now(UTC).isoformat(),
        },
        updated_by=None,
    )
    wrong_key = replace(settings, platform_secret_key=Fernet.generate_key().decode("ascii"))
    result = DeploymentReadinessStore(wrong_key, PROJECT_ROOT).assess()
    checks = {item["id"]: item for item in result["profiles"][0]["checks"]}
    assert checks["runtime_secret_storage"]["status"] == "block"
    assert checks["tushare_verified"]["status"] == "block"
    assert "无法解密" in checks["tushare_verified"]["evidence"]


def test_readiness_accepts_one_governed_full_data_pipeline(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    settings = _settings(monkeypatch, database_url, tmp_path / "data", auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _create_data_pipeline_schedule(
        database_url,
        name="governed full daily pipeline",
        payload={
            "profile": "full",
            "snapshot_start": "2008-01-01",
            "lookback_days": 30,
            "bundles": list(AUTOMATED_DATA_BUNDLES),
        },
    )

    check = _data_schedule_check(settings)

    assert check["status"] == "pass"
    assert len(check["details"]["governed_data_pipeline_ids"]) == 1
    assert check["details"]["incremental_sync_ids"] == []


def test_readiness_accepts_exact_governed_five_schedule_suite(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    settings = _settings(monkeypatch, database_url, data_root, auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _qlib_dataset(data_root)
    _create_governed_schedule_suite(database_url)

    check = _data_schedule_check(settings)

    assert check["status"] == "pass"
    assert check["details"]["mode"] == "governed_suite_v1"
    assert set(check["details"]["governed_suite_ids"]) == {
        "data_pipeline",
        "information_pipeline",
        "information_factor_refresh",
        "ashare_5m_sync",
        "auxiliary_data_pipeline",
    }
    assert all(
        len(ids) == 1 for ids in check["details"]["governed_suite_ids"].values()
    )
    assert check["details"]["rejected_suite_ids"] == []


def test_readiness_rejects_unreviewed_npr_in_governed_schedule_suite(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    settings = _settings(monkeypatch, database_url, data_root, auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _qlib_dataset(data_root)
    _create_governed_schedule_suite(database_url, include_npr=True)

    check = _data_schedule_check(settings)

    assert check["status"] == "block"
    assert check["details"]["mode"] == "invalid"
    assert check["details"]["governed_suite_ids"]["information_pipeline"] == []
    assert len(check["details"]["rejected_suite_ids"]) == 1


def test_readiness_rejects_schedule_suite_time_drift(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    settings = _settings(monkeypatch, database_url, data_root, auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _qlib_dataset(data_root)
    _create_governed_schedule_suite(
        database_url, minute_run_time=time(15, 0)
    )

    check = _data_schedule_check(settings)

    assert check["status"] == "block"
    assert check["details"]["mode"] == "invalid"
    assert len(check["details"]["governed_suite_ids"]["ashare_5m_sync"]) == 0
    assert len(check["details"]["rejected_suite_ids"]) == 1


def test_readiness_rejects_suite_whose_fixed_evaluation_oos_is_too_short(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    settings = _settings(monkeypatch, database_url, data_root, auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _qlib_dataset(data_root)
    _create_governed_schedule_suite(
        database_url, short_final_oos=True
    )

    check = _data_schedule_check(settings)

    assert check["status"] == "block"
    assert check["details"]["mode"] == "invalid"
    assert check["details"]["governed_suite_ids"][
        "information_factor_refresh"
    ] == []


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {
                "profile": "research-assets",
                "snapshot_start": "2008-01-01",
                "bundles": ["research_corpus"],
            },
            id="research-assets",
        ),
        pytest.param(
            {
                "profile": "full",
                "snapshot_start": "2008-01-01",
                "bundles": list(AUTOMATED_DATA_BUNDLES[:-1]),
            },
            id="missing-bundle",
        ),
        pytest.param(
            {
                "profile": "full",
                "snapshot_start": "2018-01-01",
                "bundles": list(AUTOMATED_DATA_BUNDLES),
            },
            id="non-2008-lineage",
        ),
        pytest.param(
            {
                "profile": "full",
                "snapshot_start": "2008-01-01",
                "lookback_days": "invalid",
                "bundles": list(AUTOMATED_DATA_BUNDLES),
            },
            id="invalid-lookback",
        ),
    ],
)
def test_readiness_rejects_ungoverned_active_data_pipelines(
    tmp_path: Path,
    monkeypatch,
    database_url: str,
    payload: dict,
) -> None:
    settings = _settings(monkeypatch, database_url, tmp_path / "data", auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _create_data_pipeline_schedule(
        database_url,
        name="ungoverned active pipeline",
        payload=payload,
    )

    check = _data_schedule_check(settings)

    assert check["status"] == "block"
    assert check["details"]["governed_data_pipeline_ids"] == []
    assert len(check["details"]["rejected_data_pipeline_ids"]) == 1


def test_readiness_rejects_multiple_active_data_refresh_schedules(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    settings = _settings(monkeypatch, database_url, tmp_path / "data", auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _create_data_pipeline_schedule(
        database_url,
        name="governed full daily pipeline",
        payload={
            "profile": "full",
            "snapshot_start": "2008-01-01",
            "bundles": list(AUTOMATED_DATA_BUNDLES),
        },
    )
    _create_incremental_schedule(
        database_url,
        name="duplicate incremental sync",
        payload={
            "profile": "full",
            "snapshot_start": "2008-01-01",
            "lookback_days": 7,
            "build_qlib": True,
        },
    )

    check = _data_schedule_check(settings)

    assert check["status"] == "block"
    assert len(check["details"]["active_schedule_ids"]) == 2


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {
                "profile": "core",
                "snapshot_start": "2008-01-01",
                "lookback_days": 7,
                "build_qlib": True,
            },
            id="non-full-profile",
        ),
        pytest.param(
            {
                "profile": "full",
                "snapshot_start": "2018-01-01",
                "lookback_days": 7,
                "build_qlib": True,
            },
            id="non-2008-lineage",
        ),
        pytest.param(
            {
                "profile": "full",
                "snapshot_start": "2008-01-01",
                "lookback_days": 7,
                "build_qlib": False,
            },
            id="download-only",
        ),
        pytest.param(
            {
                "profile": "full",
                "snapshot_start": "2008-01-01",
                "lookback_days": "invalid",
                "build_qlib": True,
            },
            id="invalid-lookback",
        ),
        pytest.param({}, id="missing-contract"),
    ],
)
def test_readiness_rejects_ungoverned_active_incremental_sync(
    tmp_path: Path,
    monkeypatch,
    database_url: str,
    payload: dict,
) -> None:
    settings = _settings(monkeypatch, database_url, tmp_path / "data", auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _create_incremental_schedule(
        database_url,
        name="ungoverned incremental sync",
        payload=payload,
    )

    check = _data_schedule_check(settings)

    assert check["status"] == "block"
    assert check["details"]["incremental_sync_ids"] == []
    assert len(check["details"]["rejected_incremental_sync_ids"]) == 1


def test_research_readiness_requires_complete_evidence_chain(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    settings = _settings(monkeypatch, database_url, data_root, auth_mode="required")
    AuthStore(database_url).bootstrap_admin(
        username="admin",
        display_name="Administrator",
        password="Secure-Admin-123!",
    )
    jobs = JobStore(database_url)
    for kind in ("bootstrap", "data_verify", "data_snapshot", "data_qlib", "qlib_baseline"):
        job = jobs.create(kind, {"fixture": True}, tmp_path / f"{kind}.log")
        jobs.finish(job["id"], exit_code=0, result={"accepted": True})
    DataTaskStore(database_url).sync_catalog()
    _qlib_dataset(data_root)
    ScheduleStore(database_url).create(
        name="daily data sync",
        kind="incremental_sync",
        timezone="Asia/Shanghai",
        run_time=time(18, 0),
        trading_days_only=True,
        payload={
            "profile": "full",
            "snapshot_start": "2008-01-01",
            "lookback_days": 7,
            "build_qlib": True,
        },
        misfire_grace_seconds=3600,
        actor="admin",
    )
    current = datetime.now(UTC)
    OperationalHealthStore(settings).record(
        {
            "status": "ok",
            "components": {
                "postgresql": {"status": "ok", "message": "ready"},
                "rdagent_runtime": {"status": "ok", "message": "ready"},
            },
            "summary": {
                "component_count": 2,
                "ok_count": 2,
                "problem_count": 0,
                "bootstrap_count": 0,
            },
            "recorded_at": current,
        }
    )

    result = DeploymentReadinessStore(settings, PROJECT_ROOT).assess(now=current)

    research, recommendation, allocation, pair = result["profiles"]
    assert result["highest_ready_profile"] == "research"
    assert research["status"] == "ready"
    assert research["passed"] == research["total"]
    assert (
        next(
            item
            for item in research["checks"]
            if item["id"] == "incremental_schedule"
        )["status"]
        == "pass"
    )
    assert pair["status"] == "blocked"
    assert (
        next(item for item in pair["checks"] if item["id"] == "pair_minute_data")["status"]
        == "block"
    )
    assert recommendation["status"] == "blocked"
    assert (
        next(item for item in recommendation["checks"] if item["id"] == "simulation_accounts")[
            "status"
        ]
        == "block"
    )
    assert (
        next(
            item
            for item in recommendation["checks"]
            if item["id"] == "simulation_60_day_replay"
        )["status"]
        == "block"
    )
    assert (
        next(
            item
            for item in recommendation["checks"]
            if item["id"] == "unsupported_schedules_retired"
        )["status"]
        == "pass"
    )
    assert (
        next(item for item in research["checks"] if item["id"] == "schema_current")["status"]
        == "pass"
    )
