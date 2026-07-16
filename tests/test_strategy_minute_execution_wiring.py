from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

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


def test_worker_persists_and_passes_minute_execution_dataset(tmp_path: Path) -> None:
    class Strategies:
        @staticmethod
        def get_version(_version_id: str) -> dict:
            return {
                "id": "version-1",
                "benchmark": "SH000300",
                "config": {"execution_method": "twap"},
                "factors": [
                    {
                        "factor_candidate_id": "factor-1",
                        "values_path": str(tmp_path / "factor.parquet"),
                        "code_sha256": "a" * 64,
                        "weight": 1.0,
                        "direction": 1,
                    }
                ],
            }

    worker = object.__new__(LocalJobWorker)
    worker.project_root = tmp_path
    worker.settings = SimpleNamespace(
        data_root=tmp_path,
        qlib_python="python",
        qlib_wsl_distro="Ubuntu-22.04",
    )
    worker.strategies = Strategies()
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
            "periods": {"start": "2024-01-01", "end": "2025-01-01"},
        },
    }

    command, result_path, environment = worker._command(job)

    assert command[command.index("--execution-provider-uri") + 1] == str(execution_path)
    assert command[command.index("--execution-frequency") + 1] == "5min"
    assert result_path == tmp_path / "artifacts" / "backtests" / "backtest-1" / "result.json"
    assert environment == {}
    manifest = json.loads(
        (tmp_path / "artifacts" / "backtests" / "backtest-1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["execution_dataset"] == "ashare-5m"
    assert manifest["execution_frequency"] == "5min"
    assert manifest["execution_contract_version"] == MINUTE_EXECUTION_CONTRACT_VERSION
