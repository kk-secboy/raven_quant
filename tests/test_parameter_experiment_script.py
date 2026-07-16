from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
