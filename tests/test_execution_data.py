from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from quant_data.checkpoint import CheckpointStore
from quant_data.config import Settings
from quant_data.execution_data import margin_specs, minute_specs
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
        market = client.post(
            "/api/jobs/supplemental-download",
            json={
                "bundle": "hk_market",
                "start": "2024-01-01",
                "end": "2024-01-31",
                "symbols": ["00700.HK", "00941.HK"],
            },
        )
    assert margin.status_code == 202
    assert intraday.status_code == 202
    assert supplemental.status_code == 202
    assert market.status_code == 202

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
    market_command, market_result, market_env = worker._command(market.json())
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
    assert "--symbols" in market_command
    assert "00700.HK,00941.HK" in market_command
    assert market_result.name == "result.json"
    assert market_env == margin_env


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
