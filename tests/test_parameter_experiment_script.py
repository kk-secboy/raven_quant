from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from qlib_test_doubles import qlib_workflow_identity

pytestmark = pytest.mark.no_database


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "run_parameter_experiment.py"
    spec = importlib.util.spec_from_file_location("run_parameter_experiment_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_segment_is_reused_only_when_period_and_config_match(
    tmp_path: Path,
) -> None:
    script = _script_module()
    config = {"topk": 50, "max_daily_turnover": 0.2}
    periods = {"start": "2024-01-01", "end": "2024-12-31"}
    result = {
        "periods": periods,
        "qlib_workflow": qlib_workflow_identity(),
        "metrics": {
            "backtest_engine": "qlib",
            "qlib_native_backtest": True,
            "provenance": {"strategy_config_sha256": script._canonical_sha256(config)},
        },
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    assert script._read_completed_result(
        result_path, config=config, periods=periods
    ) == result
    assert script._read_completed_result(
        result_path, config={**config, "topk": 80}, periods=periods
    ) is None
    assert script._read_completed_result(
        result_path,
        config=config,
        periods={"start": "2025-01-01", "end": "2025-12-31"},
    ) is None


def test_cross_trial_dsr_rewrites_trial_metrics_and_progress() -> None:
    script = _script_module()
    returns = {
        0: pd.Series([0.001 + (index % 5 - 2) * 0.0002 for index in range(60)]),
        1: pd.Series([0.0005 + (index % 7 - 3) * 0.0003 for index in range(60)]),
    }
    trials = []
    for index in range(2):
        provisional = script.deflated_sharpe_probability(
            returns[index], trials=2
        )
        trials.append(
            {
                "trial_index": index,
                "parameters": {"topk": 20 + index},
                "status": "succeeded",
                "score": 0.0,
                "warnings": ["deflated_sharpe_failed"],
                "error": None,
                "metrics": {
                    "in_sample": {},
                    "out_of_sample": {
                        "deflated_sharpe": provisional,
                        "deflated_sharpe_probability": provisional["probability"],
                    },
                },
            }
        )

    assert script._finalize_cross_trial_dsr(
        trials, returns, trial_count=2
    )
    progress = script._progress_payload(trials, trial_count=2)

    assert progress["trials"][0]["warnings"] == trials[0]["warnings"]
    for trial in trials:
        evidence = trial["metrics"]["out_of_sample"]["deflated_sharpe"]
        assert evidence["status"] == "ok"
        assert evidence["method_version"] == (
            "bailey-lopez-de-prado-cross-trial-v2"
        )
        assert evidence["trial_sharpe_std"] > 0
