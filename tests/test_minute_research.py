from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qlib_test_doubles import qlib_workflow_identity

from quant_platform.minute_research import (
    evaluate_minute_factor,
    minute_factor_expressions,
)
from scripts.run_minute_factor_research import _record_research_result

pytestmark = pytest.mark.no_database


def test_evaluates_cross_sectional_minute_factor_with_costs() -> None:
    timestamps = pd.date_range("2026-07-10 09:31", periods=40, freq="min")
    instruments = [f"SH60{index:04d}" for index in range(20)]
    index = pd.MultiIndex.from_product(
        [timestamps, instruments], names=["datetime", "instrument"]
    )
    cross_section = np.tile(np.linspace(-1, 1, len(instruments)), len(timestamps))
    factor = pd.Series(cross_section, index=index)
    label = pd.Series(cross_section * 0.001, index=index)

    metrics = evaluate_minute_factor(
        factor,
        label,
        horizon_minutes=5,
        cost_rate=0.0001,
    )

    assert metrics["rank_ic"] == pytest.approx(1.0)
    assert metrics["rebalance_timestamps"] == 8
    assert metrics["mean_net_return"] > 0
    assert metrics["average_turnover"] >= 0


def test_minute_factor_requires_cross_section() -> None:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-07-10 09:31", periods=5, freq="min"), ["SH600000"]],
        names=["datetime", "instrument"],
    )
    values = pd.Series(1.0, index=index)
    with pytest.raises(ValueError, match="insufficient"):
        evaluate_minute_factor(
            values,
            values,
            horizon_minutes=1,
            cost_rate=0.0001,
        )


def test_five_minute_research_preserves_duration_and_horizon_semantics() -> None:
    expressions = minute_factor_expressions("5min")
    assert expressions["oversold_60m"] == "-($close/Mean($close,12)-1)"
    assert "Mean($close,24)" in expressions["lower_band_120m"]

    timestamps = pd.date_range("2026-07-10 09:35", periods=12, freq="5min")
    instruments = [f"SH60{index:04d}" for index in range(20)]
    index = pd.MultiIndex.from_product(
        [timestamps, instruments], names=["datetime", "instrument"]
    )
    cross_section = np.tile(np.linspace(-1, 1, len(instruments)), len(timestamps))
    metrics = evaluate_minute_factor(
        pd.Series(cross_section, index=index),
        pd.Series(cross_section * 0.001, index=index),
        horizon_minutes=15,
        bar_minutes=5,
        cost_rate=0.0001,
    )

    assert metrics["rebalance_timestamps"] == 4
    with pytest.raises(ValueError, match="multiple"):
        evaluate_minute_factor(
            pd.Series(cross_section, index=index),
            pd.Series(cross_section * 0.001, index=index),
            horizon_minutes=7,
            bar_minutes=5,
            cost_rate=0.0001,
        )


def test_minute_research_records_result_in_unified_qlib_workflow(
    tmp_path,
) -> None:
    captured: dict = {}

    class FakeWorkflow:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def log_params(self, values):
            captured["params"] = values

        def log_metrics(self, values):
            captured["metrics"] = values

        def set_tags(self, values):
            captured["tags"] = values

        def identity_dict(self):
            return qlib_workflow_identity()

        def save_artifacts(self, path, *, artifact_path):
            captured["saved"] = (path, artifact_path)

    def workflow_factory(**kwargs):
        captured["workflow"] = kwargs
        return FakeWorkflow()

    result = {
        "status": "ok",
        "dataset": "ashare-5min",
        "frequency": "5min",
        "start": "2024-01-02",
        "end": "2024-01-31",
        "horizons": [5, 15],
        "cost_rate": 0.0002,
        "dataset_identity_sha256": "a" * 64,
        "research_code_sha256": "b" * 64,
        "created_at": "2026-07-16T00:00:00+00:00",
        "results": [
            {"factor": "oversold", "status": "ok", "score": 0.12},
            {"factor": "volume", "status": "failed", "error": "insufficient"},
        ],
        "ranking": [{"factor": "oversold", "status": "ok", "score": 0.12}],
    }
    output = tmp_path / "minute-research" / "result.json"

    recorded = _record_research_result(
        result,
        output=output,
        tracking_uri="postgresql://tracking",
        workflow_factory=workflow_factory,
    )

    assert captured["workflow"]["run_kind"] == "minute-factor-research"
    assert captured["workflow"]["tracking_uri"] == "postgresql://tracking"
    assert captured["workflow"]["dataset_identity_sha256"] == "a" * 64
    assert captured["metrics"] == {
        "successful_factor_horizons": 1,
        "failed_factor_horizons": 1,
        "best_score": 0.12,
    }
    assert captured["saved"] == (output.parent, "minute-research")
    assert recorded["qlib_workflow"] == qlib_workflow_identity()
    assert pd.read_json(output, typ="series")["qlib_workflow"] == qlib_workflow_identity()
