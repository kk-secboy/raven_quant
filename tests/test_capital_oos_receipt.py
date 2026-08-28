from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from quant_platform.capital_oos_receipt import (
    capital_oos_receipt,
    formal_oos_paired_returns,
    require_capital_oos_receipt,
)

pytestmark = pytest.mark.no_database


def _batch(*, passed: bool = True) -> dict:
    digest = "a" * 64
    return {
        "id": "batch-1",
        "status": "settled",
        "passed": passed,
        "dataset_identity_sha256": "b" * 64,
        "dataset_lineage_id": "c" * 64,
        "final_oos_start": "2025-01-02",
        "final_oos_end": "2025-12-31",
        "trading_dates_json": ["2025-01-02", "2025-01-03"],
        "trading_dates_sha256": "d" * 64,
        "frozen_bundle_manifest_sha256": "e" * 64,
        "frozen_baseline_manifest_sha256": "f" * 64,
        "settlement_evidence_sha256": "1" * 64,
        "settlement_evidence_json": {
            "supporting_evidence": {"formal_oos_artifact_sha256": digest}
        },
    }


def test_formal_artifact_returns_are_exact_cost_after_pair(tmp_path) -> None:
    artifact = tmp_path / "daily_returns.parquet"
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "return": [0.01, -0.02],
            "cost": [0.001, 0.002],
            "bench": [0.005, -0.01],
        }
    ).to_parquet(artifact, index=False)
    candidate, baseline, evidence = formal_oos_paired_returns(
        artifact, expected_trading_dates=["2025-01-02", "2025-01-03"]
    )
    assert candidate.tolist() == pytest.approx([0.009, -0.022])
    assert baseline.tolist() == pytest.approx([0.005, -0.01])
    assert evidence["formal_oos_artifact_sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()


def test_formal_artifact_rejects_reordered_or_incomplete_dates(tmp_path) -> None:
    artifact = tmp_path / "daily_returns.parquet"
    pd.DataFrame(
        {
            "date": ["2025-01-03", "2025-01-02"],
            "return": [0.01, 0.01],
            "cost": [0.0, 0.0],
            "bench": [0.0, 0.0],
        }
    ).to_parquet(artifact, index=False)
    with pytest.raises(ValueError, match="preregistered window"):
        formal_oos_paired_returns(
            artifact, expected_trading_dates=["2025-01-02", "2025-01-03"]
        )


def test_receipt_is_bound_to_backtest_strategy_dataset_window_and_artifact() -> None:
    artifact_sha = "a" * 64
    receipt = capital_oos_receipt(
        _batch(),
        backtest_id="backtest-1",
        strategy_version_id="version-1",
        dataset="cn-qlib",
        periods={"start": "2025-01-02", "end": "2025-12-31"},
        formal_oos_artifact_sha256=artifact_sha,
    )
    assert require_capital_oos_receipt(
        receipt,
        backtest_id="backtest-1",
        strategy_version_id="version-1",
        dataset="cn-qlib",
        periods={"start": "2025-01-02", "end": "2025-12-31"},
    )["batch_id"] == "batch-1"
    with pytest.raises(ValueError, match="does not match"):
        require_capital_oos_receipt(
            receipt,
            backtest_id="other",
            strategy_version_id="version-1",
            dataset="cn-qlib",
            periods={"start": "2025-01-02", "end": "2025-12-31"},
        )


def test_failed_batch_never_mints_an_approval_receipt() -> None:
    with pytest.raises(ValueError, match="not a settled passing"):
        capital_oos_receipt(
            _batch(passed=False),
            backtest_id="backtest-1",
            strategy_version_id="version-1",
            dataset="cn-qlib",
            periods={"start": "2025-01-02", "end": "2025-12-31"},
            formal_oos_artifact_sha256="a" * 64,
        )
