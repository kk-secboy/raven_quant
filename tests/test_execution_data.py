from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from quant_data.checkpoint import CheckpointStore
from quant_data.cli import (
    _historical_a_share_active_ranges,
    _historical_a_share_symbols,
    _historically_active_symbols,
    _open_market_dates,
)
from quant_data.config import Settings
from quant_data.execution_contract import (
    MINUTE_EXECUTION_CONTRACT_VERSION,
    MINUTE_SOURCE_UNIT_CONTRACTS,
)
from quant_data.execution_data import margin_specs, minute_specs, news_specs
from quant_data.models import ProviderResult
from quant_data.provider import ProviderError
from quant_data.runner import DownloadRunner
from quant_data.storage import ParquetStore
from quant_platform.api import create_app
from quant_platform.job_store import JobStore
from quant_platform.runtime_secret_store import RuntimeSecretStore
from quant_platform.worker import LocalJobWorker


class FakeExecutionProvider:
    def fetch(self, api_name, params, fields=()):
        if api_name == "margin_secs":
            rows = [
                {
                    "trade_date": params["trade_date"],
                    "ts_code": "510300.SH",
                    "name": "沪深300ETF",
                    "exchange": "SSE",
                }
            ]
        else:
            rows = [
                {
                    "ts_code": params["ts_code"],
                    "trade_time": f"{params['start_date'][:10]} 09:31:00",
                    "open": 3.5,
                    "close": 3.51,
                    "high": 3.52,
                    "low": 3.49,
                    "vol": 1000,
                    "amount": 3500,
                }
            ]
        return ProviderResult(api_name, list(rows[0]), rows, json.dumps(rows).encode())


def test_plans_daily_market_margin_and_monthly_symbol_windows() -> None:
    margins = margin_specs(["20240103", "20240102", "20240102"], max_attempts=5)
    assert [spec.params for spec in margins] == [
        {"trade_date": "20240102"},
        {"trade_date": "20240103"},
    ]

    minutes = minute_specs(
        {
            "etf_1m": ["SH510300", "159919.SZ"],
            "futures_1m": ["IF2401.CFX"],
        },
        start=date(2024, 1, 20),
        end=date(2024, 2, 5),
        max_attempts=5,
    )
    etf = [spec for spec in minutes if spec.dataset == "etf_1m"]
    futures = [spec for spec in minutes if spec.dataset == "futures_1m"]
    assert len(etf) == 4
    assert len(futures) == 2
    assert {spec.params["ts_code"] for spec in etf} == {"510300.SH", "159919.SZ"}
    assert all(spec.params["freq"] == "1min" for spec in minutes)


@pytest.mark.no_database
def test_news_history_is_clipped_to_documented_provider_start() -> None:
    assert (
        news_specs(
            date(2008, 1, 1),
            date(2018, 11, 19),
            max_attempts=3,
        )
        == []
    )

    specs = news_specs(
        date(2008, 1, 1),
        date(2018, 11, 21),
        max_attempts=3,
    )

    assert len(specs) == 18
    assert specs[0].params["start_date"] == "2018-11-20 00:00:00"
    assert specs[-1].params["end_date"] == "2018-11-21 23:59:59"


@pytest.mark.no_database
def test_full_a_share_five_minute_plans_monthly_resumable_windows() -> None:
    specs = minute_specs(
        {"ashare_5m": ["600000.SH", "000001.SZ", "899050.BJ"]},
        start=date(2024, 1, 20),
        end=date(2024, 2, 5),
        max_attempts=5,
        freq="5min",
    )
    assert len(specs) == 6
    assert {spec.api_name for spec in specs} == {"stk_mins"}
    assert {spec.params["freq"] for spec in specs} == {"5min"}
    assert {spec.dataset for spec in specs} == {"ashare_5m"}


@pytest.mark.no_database
def test_15_30_60_minute_bars_never_create_a_tushare_download_path() -> None:
    for frequency in ("15min", "30min", "60min"):
        with pytest.raises(ValueError, match="resampled by Qlib"):
            minute_specs(
                {"ashare_5m": ["600000.SH"]},
                start=date(2024, 1, 2),
                end=date(2024, 1, 31),
                max_attempts=3,
                freq=frequency,
            )


@pytest.mark.no_database
def test_full_a_share_history_includes_delisted_names_without_survivorship_bias() -> None:
    master = pd.DataFrame(
        [
            {"ts_code": "600000.SH", "list_date": "19991110", "delist_date": None},
            {"ts_code": "000001.SZ", "list_date": "19910403", "delist_date": "20240510"},
            {"ts_code": "600001.SH", "list_date": "19920101", "delist_date": "20231231"},
            {"ts_code": "920001.BJ", "list_date": "20250102", "delist_date": None},
            {"ts_code": "00700.HK", "list_date": "20040616", "delist_date": None},
        ]
    )

    assert _historical_a_share_symbols(
        master,
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
    ) == ["000001.SZ", "600000.SH"]


@pytest.mark.no_database
def test_full_a_share_windows_are_clipped_to_each_stock_lifecycle() -> None:
    master = pd.DataFrame(
        [
            {"ts_code": "600000.SH", "list_date": "19991110", "delist_date": "20240120"},
            {"ts_code": "920001.BJ", "list_date": "20260215", "delist_date": None},
        ]
    )
    active_ranges = _historical_a_share_active_ranges(
        master,
        start=date(2024, 1, 1),
        end=date(2026, 3, 31),
    )
    specs = minute_specs(
        {"ashare_5m": active_ranges},
        start=date(2024, 1, 1),
        end=date(2026, 3, 31),
        max_attempts=3,
        freq="5min",
        active_ranges_by_dataset={"ashare_5m": active_ranges},
    )

    windows = {
        (spec.params["ts_code"], spec.params["start_date"], spec.params["end_date"])
        for spec in specs
    }
    assert windows == {
        ("600000.SH", "2024-01-01 00:00:00", "2024-01-20 23:59:59"),
        ("920001.BJ", "2026-02-15 00:00:00", "2026-02-28 23:59:59"),
        ("920001.BJ", "2026-03-01 00:00:00", "2026-03-31 23:59:59"),
    }


@pytest.mark.no_database
def test_hk_financial_universe_intersects_listing_lifecycle() -> None:
    master = pd.DataFrame(
        [
            {
                "ts_code": "00700.HK",
                "list_date": "20040616",
                "delist_date": None,
                "list_status": "L",
            },
            {
                "ts_code": "00001.HK",
                "list_date": "19860101",
                "delist_date": "20231231",
                "list_status": "D",
            },
            {
                "ts_code": "09999.HK",
                "list_date": "20250102",
                "delist_date": None,
                "list_status": "P",
            },
        ]
    )

    assert _historically_active_symbols(
        master,
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        suffixes=(".HK",),
    ) == ["00700.HK"]


@pytest.mark.no_database
def test_market_calendar_helper_keeps_only_open_sessions() -> None:
    calendar = pd.DataFrame(
        [
            {"cal_date": "20240101", "is_open": 0},
            {"cal_date": "20240102", "is_open": 1},
            {"cal_date": "20240103", "is_open": "true"},
            {"cal_date": "20240104", "is_open": 0},
        ]
    )

    assert _open_market_dates(calendar, start=date(2024, 1, 1), end=date(2024, 1, 4)) == [
        "20240102",
        "20240103",
    ]


def test_runner_normalizes_shortability_and_minute_timestamp(
    tmp_path: Path, database_url: str
) -> None:
    checkpoint = CheckpointStore(database_url)
    storage = ParquetStore(tmp_path)
    specs = [
        *margin_specs(["20240102"], max_attempts=2),
        *minute_specs(
            {"etf_1m": ["510300.SH"]},
            start=date(2024, 1, 2),
            end=date(2024, 1, 2),
            max_attempts=2,
        ),
    ]
    checkpoint.add(specs)
    summary = DownloadRunner(
        checkpoint=checkpoint,
        storage=storage,
        provider=FakeExecutionProvider(),
        workers=2,
    ).run({"margin_eligibility", "etf_1m"})
    assert summary.failed == 0
    margin = storage.read_units(checkpoint.successful("margin_eligibility"))
    minute = storage.read_units(checkpoint.successful("etf_1m"))
    assert margin.loc[0, "shortable"] == True  # noqa: E712
    assert pd.api.types.is_datetime64_any_dtype(minute["trade_time"])


def test_runner_rejects_possible_tushare_minute_truncation(
    tmp_path: Path, database_url: str
) -> None:
    class CappedProvider:
        def fetch(self, api_name, params, fields=()):
            row = {
                "ts_code": params["ts_code"],
                "trade_time": "2024-01-02 09:31:00",
                "open": 1,
                "close": 1,
                "high": 1,
                "low": 1,
                "vol": 1,
                "amount": 1,
            }
            return ProviderResult(api_name, list(row), [row] * 8000, b"{}")

    checkpoint = CheckpointStore(database_url)
    spec = minute_specs(
        {"etf_1m": ["510300.SH"]},
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        max_attempts=2,
    )[0]
    checkpoint.add([spec])
    summary = DownloadRunner(
        checkpoint=checkpoint,
        storage=ParquetStore(tmp_path),
        provider=CappedProvider(),
        workers=1,
    ).run({"etf_1m"})
    assert summary.failed == 1
    failure = checkpoint.failures()[0]
    assert "8000-row limit" in failure["last_error"]
    assert failure["attempts"] == 2


def test_execution_data_api_and_worker_commands(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setenv("PLATFORM_SECRET_KEY", key)
    monkeypatch.setenv("TUSHARE_API_URL", "https://api.tushare.pro")
    monkeypatch.setenv("TUSHARE_TOKEN", "fixture-token")
    snapshot = tmp_path / "data" / "snapshots" / "execution-fixture"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "name": "execution-fixture",
                "frequency": "1min",
                "datasets": {"etf_1m": {"rows": 1}},
            }
        ),
        encoding="utf-8",
    )
    minute_dataset = tmp_path / "data" / "qlib" / "execution-fixture-1min"
    (minute_dataset / "calendars").mkdir(parents=True)
    (minute_dataset / "instruments").mkdir()
    (minute_dataset / "features").mkdir()
    (minute_dataset / "metadata").mkdir()
    (minute_dataset / "calendars" / "1min.txt").write_text(
        "2024-01-02 09:31:00\n2024-01-31 15:00:00\n", encoding="utf-8"
    )
    (minute_dataset / "instruments" / "all.txt").write_text(
        "SH510300\t2024-01-02 09:31:00\t2024-01-31 15:00:00\n", encoding="utf-8"
    )
    (minute_dataset / "metadata" / "provenance.json").write_text(
        json.dumps(
            {
                "frequency": "1min",
                "dataset_identity_sha256": "a" * 64,
                "snapshot_manifest_sha256": "b" * 64,
                "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
                "fields": ["vwap", "volume", "paused", "up_limit", "down_limit"],
                "lineage_verified": True,
                "source_datasets": ["etf_1m"],
                "source_unit_contracts": {"etf_1m": MINUTE_SOURCE_UNIT_CONTRACTS["etf_1m"]},
            }
        ),
        encoding="utf-8",
    )
    app = create_app(tmp_path)
    with TestClient(app) as client:
        margin = client.post(
            "/api/jobs/margin-eligibility",
            json={"start": "2024-01-01", "end": "2024-01-31"},
        )
        intraday = client.post(
            "/api/jobs/core-intraday",
            json={
                "start": "2024-01-01",
                "end": "2024-01-31",
                "etfs": ["510300.SH", "159919.SZ"],
            },
        )
        supplemental = client.post(
            "/api/jobs/supplemental-download",
            json={
                "bundle": "cn_macro",
                "start": "2024-01-01",
                "end": "2024-01-31",
            },
        )
        capital_flow = client.post(
            "/api/jobs/supplemental-download",
            json={
                "bundle": "cn_capital_flow",
                "start": "2024-01-01",
                "end": "2024-01-31",
            },
        )
        specialty_minutes_without_symbols = client.post(
            "/api/jobs/supplemental-download",
            json={
                "bundle": "strategy_specialty_minutes",
                "start": "2024-01-01",
                "end": "2024-01-31",
            },
        )
        specialty_minutes = client.post(
            "/api/jobs/supplemental-download",
            json={
                "bundle": "strategy_specialty_minutes",
                "start": "2024-01-01",
                "end": "2024-01-31",
                "symbols": ["000001.SZ", "600519.SH"],
            },
        )
        ashare_5m = client.post(
            "/api/jobs/ashare-5m",
            json={"start": "2024-01-01", "end": "2024-01-31"},
        )
        market = client.post(
            "/api/jobs/supplemental-download",
            json={
                "bundle": "hk_market",
                "start": "2024-01-01",
                "end": "2024-01-31",
                "symbols": ["00700.HK", "00941.HK"],
            },
        )
        minute_qlib = client.post(
            "/api/jobs/minute-qlib",
            json={"snapshot_name": "execution-fixture"},
        )
        minute_research = client.post(
            "/api/jobs/minute-research",
            json={
                "dataset": "execution-fixture-1min",
                "start": "2024-01-02",
                "end": "2024-01-31",
                "horizons": [5, 15],
            },
        )
    assert margin.status_code == 202
    assert intraday.status_code == 202
    assert supplemental.status_code == 202
    assert capital_flow.status_code == 202
    assert specialty_minutes_without_symbols.status_code == 422
    assert specialty_minutes.status_code == 202
    assert ashare_5m.status_code == 202
    assert market.status_code == 202
    assert minute_qlib.status_code == 202
    assert minute_research.status_code == 202

    announcements = JobStore(database_url).create(
        "cninfo_announcements_download",
        {
            "start": "2024-01-01",
            "end": "2024-01-31",
            "ts_codes": ["000001.SZ", "600519.SH"],
            "limit": 25,
        },
        tmp_path / "announcements.log",
    )

    settings = Settings(
        api_url="",
        token="",
        data_root=tmp_path / "data",
        database_url=database_url,
        platform_secret_key=key,
    )
    RuntimeSecretStore(database_url, key).put(
        "tushare",
        {"api_url": "https://proxy.example/v1", "token": "database-token"},
        metadata={"endpoint_host": "proxy.example"},
        updated_by=None,
    )
    worker = LocalJobWorker(JobStore(database_url), tmp_path, settings)
    margin_command, margin_result, margin_env = worker._command(margin.json())
    intraday_command, intraday_result, intraday_env = worker._command(intraday.json())
    supplemental_command, supplemental_result, supplemental_env = worker._command(
        supplemental.json()
    )
    capital_flow_command, _, _ = worker._command(capital_flow.json())
    specialty_minutes_command, _, _ = worker._command(specialty_minutes.json())
    ashare_5m_command, ashare_5m_result, ashare_5m_env = worker._command(ashare_5m.json())
    market_command, market_result, market_env = worker._command(market.json())
    minute_qlib_command, minute_qlib_result, minute_qlib_env = worker._command(minute_qlib.json())
    minute_research_command, minute_research_result, minute_research_env = worker._command(
        minute_research.json()
    )
    announcement_command, announcement_result, announcement_env = worker._command(announcements)
    assert "margin-eligibility" in margin_command
    assert "core-intraday" in intraday_command
    assert margin_result.name == "result.json"
    assert intraday_result.name == "result.json"
    assert margin_env == {
        "TUSHARE_API_URL": "https://proxy.example/v1",
        "TUSHARE_TOKEN": "database-token",
    }
    assert intraday_env == margin_env
    assert "--auto-universe" in intraday_command
    assert "--max-stocks" in intraday_command
    assert "supplemental-download" in supplemental_command
    assert "cn_macro" in supplemental_command
    assert supplemental_result.name == "result.json"
    assert supplemental_env == margin_env
    assert "cn_capital_flow" in capital_flow_command
    assert "strategy_specialty_minutes" in specialty_minutes_command
    assert "000001.SZ,600519.SH" in specialty_minutes_command
    assert "ashare-5m" in ashare_5m_command
    assert "--snapshot-name" in ashare_5m_command
    assert ashare_5m_result.name == "result.json"
    assert ashare_5m_env == margin_env
    assert "--symbols" in market_command
    assert "00700.HK,00941.HK" in market_command
    assert market_result.name == "result.json"
    assert market_env == margin_env
    assert "build-minute-qlib" in minute_qlib_command
    assert "execution-fixture" in minute_qlib_command
    assert "execution-fixture-1min" in minute_qlib_command
    assert minute_qlib_result is None
    assert minute_qlib_env == {}
    assert any("run_minute_factor_research.py" in item for item in minute_research_command)
    assert "5,15" in minute_research_command
    assert "--tracking-uri" in minute_research_command
    assert settings.mlflow_tracking_uri in minute_research_command
    assert minute_research_result.name == "result.json"
    assert set(minute_research_env) == {"_MLFLOW_SERVER_ARTIFACT_ROOT"}
    assert "cninfo-announcements" in announcement_command
    assert "000001.SZ,600519.SH" in announcement_command
    assert "25" in announcement_command
    assert announcement_result.name == "result.json"
    assert announcement_env == {}


def test_minute_qlib_api_rejects_daily_snapshot(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    snapshot = tmp_path / "data" / "snapshots" / "daily-fixture"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(
        json.dumps({"name": "daily-fixture", "frequency": "day", "datasets": {}}),
        encoding="utf-8",
    )

    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/jobs/minute-qlib",
            json={"snapshot_name": "daily-fixture"},
        )

    assert response.status_code == 409
    assert "supported minute" in response.json()["detail"]


def test_minute_qlib_api_accepts_five_minute_snapshot(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    snapshot = tmp_path / "data" / "snapshots" / "ashare-five-minute"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "name": "ashare-five-minute",
                "frequency": "5min",
                "datasets": {"ashare_5m": {"rows": 1}},
            }
        ),
        encoding="utf-8",
    )

    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/jobs/minute-qlib",
            json={"snapshot_name": "ashare-five-minute"},
        )

    assert response.status_code == 202
    assert response.json()["payload"]["frequency"] == "5min"
    assert response.json()["payload"]["output_name"] == "ashare-five-minute-5min"


def test_minute_qlib_api_builds_30min_from_existing_5min_snapshot(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    snapshot = tmp_path / "data" / "snapshots" / "ashare-five-minute"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "name": "ashare-five-minute",
                "frequency": "5min",
                "datasets": {"ashare_5m": {"rows": 1}},
            }
        ),
        encoding="utf-8",
    )

    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/jobs/minute-qlib",
            json={
                "snapshot_name": "ashare-five-minute",
                "target_frequency": "30min",
            },
        )

    assert response.status_code == 202
    assert response.json()["payload"]["source_frequency"] == "5min"
    assert response.json()["payload"]["target_frequency"] == "30min"
    assert response.json()["payload"]["frequency"] == "30min"
    assert response.json()["payload"]["output_name"] == "ashare-five-minute-30min"


def test_completed_supplemental_job_can_be_created_again_for_missing_units(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setenv("TUSHARE_API_URL", "https://api.tushare.pro")
    monkeypatch.setenv("TUSHARE_TOKEN", "fixture-token")
    app = create_app(tmp_path)
    payload = {
        "bundle": "cn_funds",
        "start": "2024-01-01",
        "end": "2024-01-31",
    }
    with TestClient(app) as client:
        first = client.post("/api/jobs/supplemental-download", json=payload)
    assert first.status_code == 202

    JobStore(database_url).finish(first.json()["id"], exit_code=0)
    with TestClient(app) as client:
        second = client.post("/api/jobs/supplemental-download", json=payload)

    assert second.status_code == 202
    assert second.json()["id"] != first.json()["id"]


def test_invalid_ohlc_is_terminal() -> None:
    spec = minute_specs(
        {"etf_1m": ["510300.SH"]},
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        max_attempts=2,
    )[0]
    row = {
        "ts_code": "510300.SH",
        "trade_time": "2024-01-02 09:31:00",
        "open": 3.5,
        "close": 3.6,
        "high": 3.4,
        "low": 3.3,
        "vol": 1,
        "amount": 1,
    }
    from quant_data.execution_data import validate_and_normalize

    with pytest.raises(ProviderError, match="OHLC"):
        validate_and_normalize(spec, ProviderResult("etf_mins", list(row), [row], b"{}"))


def test_hundredfold_minute_volume_is_normalized_before_storage() -> None:
    spec = minute_specs(
        {"ashare_5m": ["600827.SH"]},
        start=date(2024, 5, 31),
        end=date(2024, 5, 31),
        max_attempts=2,
        freq="5min",
    )[0]
    row = {
        "ts_code": "600827.SH",
        "trade_time": "2024-05-31 09:35:00",
        "open": 8.44,
        "close": 8.45,
        "high": 8.5,
        "low": 8.43,
        "vol": 25_880_000.0,
        "amount": 2_193_095.0,
    }
    from quant_data.execution_data import validate_and_normalize

    result = validate_and_normalize(spec, ProviderResult("stk_mins", list(row), [row], b"{}"))

    assert result.rows[0]["vol"] == 258_800.0
    assert result.metadata["normalized_volume_rows"] == 1
