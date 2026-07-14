from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from quant_data.config import Settings
from quant_platform.api import create_app
from quant_platform.job_store import JobStore
from quant_platform.schedule_store import ScheduleStore
from quant_platform.scheduler import SchedulerEngine
from quant_platform.strategy_store import StrategyStore
from quant_platform.worker import LocalJobWorker


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data_fixture(root: Path) -> None:
    qlib = root / "qlib" / "daily-2024-2026"
    (qlib / "calendars").mkdir(parents=True)
    (qlib / "instruments").mkdir()
    (qlib / "features").mkdir()
    (qlib / "metadata").mkdir()
    (qlib / "calendars" / "day.txt").write_text("2024-01-02\n2026-07-10\n", encoding="utf-8")
    (qlib / "instruments" / "cn_all.txt").write_text(
        "SH510300\t2024-01-02\t2026-07-10\nSZ159919\t2024-01-02\t2026-07-10\n",
        encoding="utf-8",
    )
    (qlib / "metadata" / "provenance.json").write_text(
        json.dumps(
            {
                "dataset_identity_sha256": "a" * 64,
                "snapshot_manifest_sha256": "b" * 64,
                "qlib_builder_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    snapshot = root / "snapshots" / "minute-2024-2026"
    minute_dir = snapshot / "parquet" / "liquid_stocks_1m"
    short_dir = snapshot / "parquet" / "margin_eligibility"
    minute_dir.mkdir(parents=True)
    short_dir.mkdir(parents=True)
    minute_path = minute_dir / "bars.parquet"
    short_path = short_dir / "eligibility.parquet"
    pd.DataFrame(
        {
            "ts_code": ["510300.SH", "159919.SZ"],
            "datetime": ["2024-01-03 10:00:00", "2024-01-03 10:00:00"],
            "close": [3.5, 4.2],
            "vol": [1_000_000, 1_000_000],
        }
    ).to_parquet(minute_path, index=False)
    pd.DataFrame(
        {
            "ts_code": ["510300.SH", "159919.SZ"],
            "trade_date": ["20240103", "20240103"],
            "shortable": [1, 1],
        }
    ).to_parquet(short_path, index=False)
    manifest = {
        "name": "minute-2024-2026",
        "start_date": "2024-01-03",
        "end_date": "2026-07-10",
        "datasets": {
            "liquid_stocks_1m": {
                "rows": 2,
                "source_sha256": "d" * 64,
                "files": [
                    {
                        "path": "parquet/liquid_stocks_1m/bars.parquet",
                        "bytes": minute_path.stat().st_size,
                        "sha256": _sha256(minute_path),
                    }
                ],
            },
            "margin_eligibility": {
                "rows": 2,
                "source_sha256": "e" * 64,
                "files": [
                    {
                        "path": "parquet/margin_eligibility/eligibility.parquet",
                        "bytes": short_path.stat().st_size,
                        "sha256": _sha256(short_path),
                    }
                ],
            },
        },
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _passing_pair_metrics() -> dict:
    digest = "a" * 64
    return {
        "backtest_engine": "quantlab_pair",
        "pair_native_backtest": True,
        "leg_y": "SH510300",
        "leg_x": "SZ159919",
        "initial_pair_evidence": {
            "correlation": 0.92,
            "cointegration_pvalue": 0.01,
            "hedge_ratio": 0.95,
        },
        "max_drawdown": -0.06,
        "sharpe_ratio": 1.2,
        "closed_trade_count": 12,
        "trading_days": 504,
        "rolling_cointegration_pass_rate": 0.90,
        "pair_robustness_pass_rate": 0.75,
        "capacity_fill_ratio": 0.99,
        "minute_execution_enforced": True,
        "shortability_enforced": True,
        "market_controls_enforced": True,
        "atomic_pair_execution_enforced": True,
        "transaction_costs_enforced": True,
        "borrow_cost_enforced": True,
        "open_position_at_end": False,
        "provenance": {
            "daily_dataset_identity_sha256": digest,
            "daily_snapshot_manifest_sha256": digest,
            "minute_snapshot_manifest_sha256": digest,
            "strategy_config_sha256": digest,
            "execution_manifest_sha256": digest,
            "pair_engine_sha256": digest,
            "shortability_evidence_sha256": digest,
        },
    }


def test_pair_api_queues_shared_governed_worker_job(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    _data_fixture(data_root)
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    app = create_app(tmp_path)
    with TestClient(app) as client:
        created = client.post(
            "/api/pair-strategies",
            json={
                "name": "ETF pair API",
                "description": "Governed ETF pair strategy created through the product API.",
                "leg_y": "SH510300",
                "leg_x": "SZ159919",
                "asset_class": "etf",
                "actor": "researcher-a",
            },
        )
        assert created.status_code == 201
        version = created.json()["versions"][0]
        queued = client.post(
            f"/api/strategy-versions/{version['id']}/pair-backtests",
            json={
                "dataset": "daily-2024-2026",
                "execution_snapshot": "minute-2024-2026",
                "minute_dataset": "liquid_stocks_1m",
                "shortability_dataset": "margin_eligibility",
                "start": "2024-01-02",
                "end": "2026-07-10",
            },
        )
    assert queued.status_code == 202
    assert queued.json()["execution_dataset"] == (
        "minute-2024-2026/liquid_stocks_1m+margin_eligibility"
    )
    jobs = JobStore(database_url)
    job = jobs.list(limit=10)[0]
    assert job["kind"] == "pair_backtest"
    settings = Settings.from_env(tmp_path / ".env")
    command, result_path, extra_env = LocalJobWorker(jobs, tmp_path, settings)._command(job)
    assert any(str(item).endswith("run_pair_backtest.py") for item in command)
    assert "--minute-path" in command
    assert "--shortability-path" in command
    assert result_path and result_path.name == "result.json"
    assert extra_env == {}


def test_pair_paper_api_queues_dedicated_spread_worker_job(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    _data_fixture(data_root)
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    app = create_app(tmp_path)
    with TestClient(app) as client:
        created = client.post(
            "/api/pair-strategies",
            json={
                "name": "ETF pair paper API",
                "description": "Governed ETF pair strategy for the spread paper API path.",
                "leg_y": "SH510300",
                "leg_x": "SZ159919",
                "asset_class": "etf",
            },
        ).json()
        version = created["versions"][0]
        queued = client.post(
            f"/api/strategy-versions/{version['id']}/pair-backtests",
            json={
                "dataset": "daily-2024-2026",
                "execution_snapshot": "minute-2024-2026",
                "minute_dataset": "liquid_stocks_1m",
                "shortability_dataset": "margin_eligibility",
                "start": "2024-01-02",
                "end": "2026-07-10",
            },
        ).json()
        strategies = StrategyStore(database_url)
        strategies.mark_backtest(queued["id"], "succeeded", metrics=_passing_pair_metrics())
        strategies.approve(
            version["id"],
            actor="independent-risk-reviewer",
            reason="Independent review accepted pair capacity, costs, and provenance.",
        )
        portfolio_response = client.post(
            "/api/pair-portfolios",
            json={
                "name": "API dedicated pair paper ledger",
                "strategy_version_id": version["id"],
                "dataset": "daily-2024-2026",
                "execution_snapshot": "minute-2024-2026",
                "minute_dataset": "liquid_stocks_1m",
                "shortability_dataset": "margin_eligibility",
                "initial_cash": 5_000_000,
            },
        )
        assert portfolio_response.status_code == 201
        portfolio = portfolio_response.json()
        schedule_response = client.post(
            "/api/schedules",
            json={
                "name": "daily pair paper close",
                "kind": "pair_paper_rebalance",
                "run_time": "15:30:00",
                "payload": {"pair_portfolio_id": portfolio["id"]},
            },
        )
        assert schedule_response.status_code == 201
        assert schedule_response.json()["kind"] == "pair_paper_rebalance"
        batch_response = client.post(
            f"/api/pair-portfolios/{portfolio['id']}/rebalance",
            json={"as_of_date": "2024-01-02", "slippage": 0.0005},
        )
        assert batch_response.status_code == 202
        batch = batch_response.json()

    jobs = JobStore(database_url)
    job = next(item for item in jobs.list(limit=20) if item["kind"] == "pair_paper_rebalance")
    assert job["payload"]["pair_portfolio_batch_id"] == batch["id"]
    settings = Settings.from_env(tmp_path / ".env")
    command, result_path, extra_env = LocalJobWorker(jobs, tmp_path, settings)._command(job)
    assert any(str(item).endswith("run_pair_paper_step.py") for item in command)
    assert "--minute-path" in command
    assert "--shortability-path" in command
    assert result_path and result_path.name == "result.json"
    assert extra_env == {}


def test_pair_paper_schedule_materializes_and_queues_atomic_spread_job(
    database_url: str, tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    _data_fixture(data_root)
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    app = create_app(tmp_path)
    with TestClient(app) as client:
        created = client.post(
            "/api/pair-strategies",
            json={
                "name": "scheduled ETF pair paper",
                "description": "Governed pair strategy for the automatic scheduler path.",
                "leg_y": "SH510300",
                "leg_x": "SZ159919",
                "asset_class": "etf",
            },
        ).json()
        version = created["versions"][0]
        queued = client.post(
            f"/api/strategy-versions/{version['id']}/pair-backtests",
            json={
                "dataset": "daily-2024-2026",
                "execution_snapshot": "minute-2024-2026",
                "minute_dataset": "liquid_stocks_1m",
                "shortability_dataset": "margin_eligibility",
                "start": "2024-01-02",
                "end": "2026-07-10",
            },
        ).json()
        strategies = StrategyStore(database_url)
        strategies.mark_backtest(queued["id"], "succeeded", metrics=_passing_pair_metrics())
        strategies.approve(
            version["id"],
            actor="independent-risk-reviewer",
            reason="Independent review accepted scheduled pair execution and provenance.",
        )
        portfolio = client.post(
            "/api/pair-portfolios",
            json={
                "name": "scheduled pair paper ledger",
                "strategy_version_id": version["id"],
                "dataset": "daily-2024-2026",
                "execution_snapshot": "minute-2024-2026",
                "minute_dataset": "liquid_stocks_1m",
                "shortability_dataset": "margin_eligibility",
                "initial_cash": 5_000_000,
            },
        ).json()

    current = datetime(2024, 1, 2, 7, 29, tzinfo=UTC)
    schedules = ScheduleStore(database_url)
    schedules.create(
        name="scheduled pair close",
        kind="pair_paper_rebalance",
        timezone="Asia/Shanghai",
        run_time=time(15, 30),
        trading_days_only=True,
        payload={"pair_portfolio_id": portfolio["id"]},
        misfire_grace_seconds=1800,
        actor="pair-paper-operator",
        now=current,
    )
    settings = Settings.from_env(tmp_path / ".env")
    result = SchedulerEngine(settings).tick(current + timedelta(minutes=1))

    assert result["materialized"] == 1
    assert result["processed"] == 1
    run = schedules.list_runs()[0]
    assert run["status"] == "enqueued"
    job = JobStore(database_url).get(run["job_id"])
    assert job["kind"] == "pair_paper_rebalance"
    assert job["payload"]["pair_portfolio_id"] == portfolio["id"]
    assert job["payload"]["as_of_date"] == "2024-01-02"
    assert job["payload"]["minute_dataset"]["dataset_name"] == "liquid_stocks_1m"
    assert job["payload"]["shortability_dataset"]["dataset_name"] == "margin_eligibility"
