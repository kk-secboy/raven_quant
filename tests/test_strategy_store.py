from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from governance_fixtures import PERIODS, create_strategy_version, formal_backtest_metrics

from quant_platform.strategy_store import StrategyStore


def test_only_v2_qlib_policy_backtest_can_be_approved(tmp_path: Path, database_url: str) -> None:
    version_id = create_strategy_version(database_url, tmp_path)
    store = StrategyStore(database_url)
    version = store.get_version(version_id)
    with pytest.raises(ValueError, match="successful backtest"):
        store.approve(version_id, actor="risk-owner", reason="No final test exists yet.")

    artifact = tmp_path / "formal-backtest"
    artifact.mkdir()
    periods = {
        "start": PERIODS["test_start"].isoformat(),
        "end": PERIODS["test_end"].isoformat(),
    }
    backtest = store.create_backtest(
        version_id=version_id,
        dataset="snapshot",
        periods=periods,
        artifact_path=artifact,
    )
    factor = version["factors"][0]
    manifest = artifact / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy_version_id": version_id,
                "dataset": "snapshot",
                "benchmark": version["benchmark"],
                "periods": periods,
                "config": version["config"],
                "factors": [
                    {
                        "candidate_id": factor["factor_candidate_id"],
                        "values_path": factor["values_path"],
                        "code_sha256": factor["code_sha256"],
                        "weight": factor["weight"],
                        "direction": factor["direction"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    metrics = formal_backtest_metrics(version, manifest)
    store.validate_backtest_artifacts(backtest["id"], metrics)
    failed_stress = deepcopy(metrics)
    failed_stress["event_stress_pass_rate"] = 0.0
    failed_stress["event_stress_passed"] = False
    store.mark_backtest(backtest["id"], "succeeded", metrics=failed_stress)
    with pytest.raises(ValueError, match="event stress"):
        store.approve(
            version_id,
            actor="risk-owner",
            reason="Event stress results failed the configured gate.",
        )
    store.mark_backtest(backtest["id"], "succeeded", metrics=metrics)
    approved = store.approve(
        version_id,
        actor="risk-owner",
        reason="Independent review accepted the governed final test.",
    )
    assert approved["status"] == "approved"


def test_final_test_can_only_be_created_once(tmp_path: Path, database_url: str) -> None:
    version_id = create_strategy_version(database_url, tmp_path)
    store = StrategyStore(database_url)
    periods = {
        "start": PERIODS["test_start"].isoformat(),
        "end": PERIODS["test_end"].isoformat(),
    }
    created = store.create_backtest(
        version_id=version_id, dataset="snapshot", periods=periods, artifact_path=tmp_path
    )
    with pytest.raises(ValueError, match="only once"):
        store.create_backtest(
            version_id=version_id,
            dataset="snapshot",
            periods=periods,
            artifact_path=tmp_path,
        )
    store.mark_backtest(created["id"], "failed", error="final test gate failed")
    with pytest.raises(ValueError, match="cannot be rerun"):
        store.requeue_backtest(created["id"])
