from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from quant_data.execution_contract import MINUTE_EXECUTION_CONTRACT_VERSION
from quant_platform.api import StrategyBacktestRequest, StrategyConfigRequest
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


def test_strategy_backtest_request_accepts_an_explicit_execution_dataset() -> None:
    request = StrategyBacktestRequest(
        dataset="daily",
        execution_dataset="ashare-5m",
        start=date(2024, 1, 1),
        end=date(2025, 1, 1),
    )
    assert request.execution_dataset == "ashare-5m"

    with pytest.raises(ValidationError, match="multiple of five"):
        StrategyConfigRequest(execution_slice_minutes=17)


def test_worker_persists_and_passes_minute_execution_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ConnectionContext:
        def __enter__(self):
            return object()

        def __exit__(self, *_args):
            return None

    class Engine:
        @staticmethod
        def connect() -> ConnectionContext:
            return ConnectionContext()

    class Strategies:
        engine = Engine()

        @staticmethod
        def get_version(_version_id: str) -> dict:
            return {
                "id": "version-1",
                "benchmark": "SH000300",
                "universe": "cn_all",
                "config": {"execution_method": "twap"},
                "factors": [
                    {
                        "factor_candidate_id": "factor-1",
                        "values_path": str(tmp_path / "factor.parquet"),
                        "code_path": str(tmp_path / "factor.py"),
                        "code_sha256": "a" * 64,
                        "source_iteration": 0,
                        "weight": 1.0,
                        "direction": 1,
                    }
                ],
            }

        @staticmethod
        def hypothesis_group_evidence(_version_id: str) -> dict:
            return {
                "economic_hypothesis_group": "hypothesis-1",
                "hypothesis_group_cap": 0.70,
                "shared_experiment_count": 1,
                "strategy_version_ids": ["version-1"],
                "experiment_family_counts": {"factor-1": 1},
            }

        @staticmethod
        def _model_signal_evidence(_connection: object, _config: dict) -> None:
            return None

    worker = object.__new__(LocalJobWorker)
    worker.project_root = tmp_path
    worker.settings = SimpleNamespace(
        data_root=tmp_path,
        qlib_python="python",
        qlib_wsl_distro="Ubuntu-22.04",
        mlflow_tracking_uri="postgresql://tracking",
    )
    worker.strategies = Strategies()
    observed_runner_binding: dict[str, object] = {}

    def require_runner(*, config: dict, job_payload: dict, runner_path: Path) -> str:
        observed_runner_binding.update(
            config=config,
            job_payload=job_payload,
            runner_path=runner_path,
        )
        return "c" * 64

    monkeypatch.setattr(
        "quant_platform.worker.require_transparent_baseline_runner",
        require_runner,
    )
    execution_path = tmp_path / "qlib" / "ashare-5m"
    job = {
        "id": "job-1",
        "kind": "strategy_backtest",
        "payload": {
            "backtest_id": "backtest-1",
            "strategy_version_id": "version-1",
            "dataset": "daily",
            "dataset_path": str(tmp_path / "qlib" / "daily"),
            "execution_dataset": {
                "name": "ashare-5m",
                "path": str(execution_path),
                "frequency": "5min",
                "provenance": {
                    "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION
                },
            },
            "periods": {
                "start": "2024-01-01",
                "end": "2025-01-01",
                "historical_start": "2010-01-01",
                "historical_end": "2023-12-31",
            },
        },
    }

    command, result_path, environment = worker._command(job)

    assert command[command.index("--execution-provider-uri") + 1] == str(execution_path)
    assert command[command.index("--execution-frequency") + 1] == "5min"
    assert command[command.index("--tracking-uri") + 1] == "postgresql://tracking"
    assert result_path == tmp_path / "artifacts" / "backtests" / "backtest-1" / "result.json"
    assert environment == {
        "_MLFLOW_SERVER_ARTIFACT_ROOT": str(tmp_path / "artifacts" / "mlflow")
    }
    manifest = json.loads(
        (tmp_path / "artifacts" / "backtests" / "backtest-1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["execution_dataset"] == "ashare-5m"
    assert manifest["execution_frequency"] == "5min"
    assert manifest["execution_contract_version"] == MINUTE_EXECUTION_CONTRACT_VERSION
    assert manifest["economic_hypothesis_group"] == "hypothesis-1"
    assert manifest["strategy_trial_count"] == 1
    assert manifest["periods"] == {"start": "2024-01-01", "end": "2025-01-01"}
    assert manifest["historical_validation_periods"] == {
        "start": "2010-01-01",
        "end": "2023-12-31",
    }
    assert manifest["factors"][0]["code_path"] == str(tmp_path / "factor.py")
    assert manifest["factors"][0]["factor_execution_mode"] == "frozen_code_recompute"
    assert manifest["transparent_baseline_runner_sha256"] == "c" * 64
    assert observed_runner_binding == {
        "config": {"execution_method": "twap"},
        "job_payload": job["payload"],
        "runner_path": tmp_path / "scripts" / "run_multifactor_backtest.py",
    }


def test_worker_builds_production_qlib_order_plan_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Simulations:
        @staticmethod
        def require_order_plan_predecessor_settled(
            _portfolio_id: str, *, signal_date: date
        ) -> dict:
            assert signal_date == date(2026, 7, 13)
            return {"ready": True, "status": "predecessor_settled"}

        @staticmethod
        def policy_risk_inputs(
            _portfolio_id: str,
            *,
            required_nav_date: date | None = None,
        ) -> dict:
            assert required_nav_date == date(2026, 7, 13)
            return {
                "contract_version": "ledger-policy-risk-v1",
                "risk_scope": "selected_account_only",
                "portfolio_id": "simulation-1",
                "status": "certified",
                "portfolio_drawdown": -0.12,
                "daily_return": -0.03,
                "allow_new_risk": True,
                "reasons": [],
            }

        @staticmethod
        def get(_portfolio_id: str) -> dict:
            return {
                "id": "simulation-1",
                "status": "active",
                "source_type": "strategy_version",
                "source_id": "version-1",
                "execution_adapter": "long_only",
                    "daily_dataset": "daily",
                    "daily_dataset_identity_sha256": "a" * 64,
                    "daily_dataset_lineage_id": "b" * 64,
                    "execution_dataset": "minute-5m",
                    "execution_policy": {
                        "execution_algorithm": "next_bar",
                        "execution_frequency": "5min",
                    },
                    "nav": 1_000_000,
            }

        @staticmethod
        def rows(_portfolio_id: str, _resource: str) -> list[dict]:
            return []

        @staticmethod
        def positions_with_holding_age(
            _portfolio_id: str,
            *,
            calendar_days: set[date],
            as_of_date: date,
            require_complete_age: bool,
        ) -> list[dict]:
            assert calendar_days == {date(2026, 7, 10), date(2026, 7, 13)}
            assert as_of_date == date(2026, 7, 13)
            assert require_complete_age is False
            return [
                {
                    "instrument": "SH600000",
                    "position_side": "long",
                    "market_value": 250_000,
                    "average_cost": 10.25,
                    "holding_age_sessions": 1,
                }
            ]

    class Strategies:
        @staticmethod
        def get_version(_version_id: str) -> dict:
            return {
                "id": "version-1",
                "status": "approved",
                "is_legacy": False,
                "benchmark": "SH000300",
                "universe": "cn_all",
                "signal_frequency": "5min",
                "config": {"execution_contract_hash": "c" * 64},
                "factors": [
                    {
                        "factor_candidate_id": "factor-1",
                        "values_path": str(tmp_path / "factor.parquet"),
                        "weight": 1.0,
                        "direction": 1,
                    }
                ],
            }

        @staticmethod
        def list_backtests(_version_id: str) -> list[dict]:
            return [
                {
                    "id": "backtest-1",
                    "status": "succeeded",
                    "is_legacy": False,
                }
            ]

    class Allocations:
        @staticmethod
        def strategy_risk_state(_version_id: str) -> dict:
            return {
                "risk_exposure_override": 1.0,
                "allow_new_risk": True,
            }

    class Promotions:
        @staticmethod
        def require_paper_signal(
            _version_id: str, *, portfolio_id: str, signal_date: date
        ) -> dict:
            assert portfolio_id == "simulation-1"
            assert signal_date == date(2026, 7, 13)
            return {"id": "paper-stage-1", "opened_at": "2026-04-01T00:00:00+00:00"}

    calendar_path = tmp_path / "qlib" / "daily" / "calendars"
    calendar_path.mkdir(parents=True)
    (calendar_path / "day.txt").write_text(
        "2026-07-10\n2026-07-13\n",
        encoding="utf-8",
    )
    newer_calendar_path = tmp_path / "qlib" / "daily-newer" / "calendars"
    newer_calendar_path.mkdir(parents=True)
    (newer_calendar_path / "day.txt").write_text(
        "2026-07-10\n2026-07-13\n2026-07-14\n",
        encoding="utf-8",
    )
    governed_datasets = [
        {
            "name": "daily",
            "ready": True,
            "reproducible": True,
            "path": str(tmp_path / "qlib" / "daily"),
            "start_date": "2026-07-10",
            "end_date": "2026-07-13",
            "provenance": {
                "dataset_identity_sha256": "a" * 64,
                "dataset_lineage_id": "b" * 64,
                "source_lineage_id": "c" * 64,
                "lineage_verified": True,
            },
        },
        {
            "name": "daily-newer",
            "ready": True,
            "reproducible": True,
            "path": str(tmp_path / "qlib" / "daily-newer"),
            "start_date": "2026-07-10",
            "end_date": "2026-07-14",
            "provenance": {
                "dataset_identity_sha256": "f" * 64,
                "dataset_lineage_id": "b" * 64,
                "source_lineage_id": "c" * 64,
                "lineage_verified": True,
            },
        },
        {
            "name": "minute-5m",
            "ready": True,
            "reproducible": True,
            "path": str(tmp_path / "qlib" / "minute-5m"),
            "provenance": {
                "frequency": "5min",
                "dataset_identity_sha256": "d" * 64,
                "dataset_lineage_id": "e" * 64,
                "source_lineage_id": "c" * 64,
                "lineage_verified": True,
            },
        },
    ]
    monkeypatch.setattr(
        "quant_platform.worker.list_qlib_datasets",
        lambda _root: governed_datasets,
    )
    monkeypatch.setattr(
        "quant_platform.data_rollover.list_qlib_datasets",
        lambda _root: governed_datasets,
    )
    worker = object.__new__(LocalJobWorker)
    worker.project_root = tmp_path
    worker.settings = SimpleNamespace(
        data_root=tmp_path,
        qlib_python="python",
        qlib_wsl_distro="Ubuntu-22.04",
        mlflow_tracking_uri="postgresql://tracking",
    )
    worker.simulations = Simulations()
    worker.strategies = Strategies()
    worker.allocations = Allocations()
    worker.promotions = Promotions()
    job = {
        "id": "order-plan-job-1",
        "kind": "simulation_order_plan",
        "payload": {
            "simulation_portfolio_id": "simulation-1",
            "signal_date": "2026-07-13",
            "signal_at": "2026-07-13T10:05:00+08:00",
            "dataset_identity_sha256": "a" * 64,
            "promotion_stage_id": "paper-stage-1",
            "promotion_stage_opened_at": "2026-04-01T00:00:00+00:00",
            "actor": "simulation-operator",
        },
    }

    command, result_path, environment = worker._command(job)

    assert "--tracking-uri" in command
    assert "--order-plan-root" in command
    assert result_path == (
        tmp_path / "artifacts" / "order-plan-jobs" / job["id"] / "result.json"
    )
    manifest = json.loads(
        (
            tmp_path
            / "artifacts"
            / "order-plan-jobs"
            / job["id"]
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["artifact_kind"] == "simulation_order_plan"
    assert manifest["formal_backtest_id"] == "backtest-1"
    assert manifest["signal_at"] == "2026-07-13T10:05:00+08:00"
    assert manifest["execution_not_before"] == "2026-07-13T10:10:00+08:00"
    assert manifest["signal_dataset"] == {
        "name": "minute-5m",
        "dataset_identity_sha256": "d" * 64,
        "dataset_lineage_id": "e" * 64,
        "source_lineage_id": "c" * 64,
        "frequency": "5min",
    }
    assert manifest["dataset_identity_sha256"] == "a" * 64
    assert manifest["dataset"] == "daily"
    assert manifest["account_risk_state"]["status"] == "certified"
    assert manifest["portfolio_drawdown"] == pytest.approx(-0.12)
    assert manifest["daily_return"] == pytest.approx(-0.03)
    assert manifest["allow_new_risk"] is True
    assert manifest["previous_holdings"] == [
        {
            "instrument": "SH600000",
            "weight": 0.25,
            "average_cost": 10.25,
            "holding_age_sessions": 1,
        }
    ]
    assert manifest["holding_age_sessions"] == {"SH600000": 1}
    assert "--signal-provider-uri" in command
    assert environment == {
        "_MLFLOW_SERVER_ARTIFACT_ROOT": str(tmp_path / "artifacts" / "mlflow")
    }


def test_recommendation_refresh_reuses_last_succeeded_position_state_on_retry(
    tmp_path: Path,
) -> None:
    local_today = datetime.now(UTC).astimezone(ZoneInfo("Asia/Shanghai")).date()
    dataset_path = tmp_path / "qlib" / "daily"
    (dataset_path / "calendars").mkdir(parents=True)
    (dataset_path / "calendars" / "day.txt").write_text(
        local_today.isoformat() + "\n",
        encoding="utf-8",
    )
    (dataset_path / "metadata").mkdir(parents=True)
    (dataset_path / "metadata" / "provenance.json").write_text(
        json.dumps({"dataset_identity_sha256": "a" * 64}),
        encoding="utf-8",
    )
    previous_state = {
        "holding_age_sessions": {"SH600000": 18},
        "take_profit_stages": {"SH600000": 1},
        "execution": {"remaining_days": 2},
    }
    prior = {
        "id": "snapshot-prior",
        "status": "succeeded",
        "as_of_date": date(2026, 7, 10),
        "effective_date": date(2026, 7, 13),
        "holdings": [
            {
                "instrument": "SH600000",
                "weight": 0.40,
                "average_cost": 9.75,
                "take_profit_stage": 1,
            }
        ],
        "snapshot": {"position_state": previous_state},
    }
    current = {
        "id": "snapshot-current",
        "status": "running",
        "as_of_date": local_today,
        "holdings": [],
        "snapshot": None,
    }

    class Recommendations:
        @staticmethod
        def get(_portfolio_id: str) -> dict:
            return {
                "id": "recommendation-1",
                "strategy_version_id": "version-1",
                "construction_notional": 1_000_000,
                "risk_exposure_override": 1.0,
                # The current queued/running row is intentionally the latest;
                # the worker must select the prior succeeded snapshot instead.
                "latest_snapshot": current,
                "snapshots": [current, prior],
            }

    class Strategies:
        @staticmethod
        def get_version(_version_id: str) -> dict:
            return {
                "id": "version-1",
                "status": "approved",
                "benchmark": "SH000300",
                "universe": "cn_all",
                "config": {"execution_contract_hash": "c" * 64},
                "factors": [
                    {
                        "factor_candidate_id": "factor-1",
                        "values_path": str(tmp_path / "factor.parquet"),
                        "weight": 1.0,
                        "direction": 1,
                    }
                ],
            }

    class Promotions:
        @staticmethod
        def require_recommendation_signal(_version_id: str, *, signal_date: date) -> None:
            assert signal_date == local_today

    class Allocations:
        @staticmethod
        def strategy_risk_state(_version_id: str) -> dict:
            return {"risk_exposure_override": 1.0, "allow_new_risk": True}

    class Accounts:
        @staticmethod
        def policy_risk_inputs(
            _portfolio_id: str, *, required_nav_date: date | None = None
        ) -> dict:
            assert required_nav_date == local_today
            return {
                "portfolio_drawdown": -0.01,
                "daily_return": 0.002,
                "allow_new_risk": True,
            }

    worker = object.__new__(LocalJobWorker)
    worker.project_root = tmp_path
    worker.settings = SimpleNamespace(
        data_root=tmp_path,
        qlib_python="python",
        qlib_wsl_distro="Ubuntu-22.04",
    )
    worker.recommendations = Recommendations()
    worker.strategies = Strategies()
    worker.promotions = Promotions()
    worker.allocations = Allocations()
    worker.recommendation_accounts = Accounts()
    job = {
        "id": "recommendation-job-1",
        "kind": "recommendation_refresh",
        "payload": {
            "recommendation_portfolio_id": "recommendation-1",
            "recommendation_snapshot_id": "snapshot-current",
            "dataset": "daily",
            "dataset_path": str(dataset_path),
            "dataset_identity_sha256": "a" * 64,
            "as_of_date": local_today.isoformat(),
        },
    }

    worker._command(job)
    manifest_path = (
        tmp_path
        / "artifacts"
        / "recommendations"
        / "recommendation-1"
        / "snapshot-current"
        / "manifest.json"
    )
    first = manifest_path.read_bytes()
    manifest = json.loads(first)
    assert manifest["previous_holdings"] == prior["holdings"]
    assert manifest["previous_snapshot"]["position_state"] == previous_state

    # A retry reconstructs byte-identical state and does not age the holding.
    worker._command(job)
    assert manifest_path.read_bytes() == first
