from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from governance_fixtures import PERIODS, create_strategy_version, formal_backtest_metrics

from quant_data.execution_contract import (
    MINUTE_EXECUTION_CONTRACT_VERSION,
    MINUTE_SOURCE_UNIT_CONTRACTS,
)
from quant_platform.strategy_store import StrategyStore


def test_strategy_versions_share_hypothesis_trial_count_and_cap(
    tmp_path: Path, database_url: str
) -> None:
    first_id = create_strategy_version(database_url, tmp_path)
    store = StrategyStore(database_url)
    first = store.get_version(first_id)
    factor = first["factors"][0]
    second = store.create_version(
        first["strategy_id"],
        benchmark=first["benchmark"],
        universe=first["universe"],
        factors=[
            {
                "candidate_id": factor["factor_candidate_id"],
                "weight": 1.0,
            }
        ],
        config=first["config"],
        actor="researcher",
    )

    first_evidence = store.hypothesis_group_evidence(first_id)
    second_evidence = store.hypothesis_group_evidence(second["id"])
    assert first_evidence == second_evidence
    assert first_evidence["economic_hypothesis_group"] == first["strategy_id"]
    assert first_evidence["hypothesis_group_cap"] == 0.70
    assert first_evidence["shared_experiment_count"] == 2
    assert first_evidence["strategy_version_ids"] == sorted(
        [first_id, second["id"]]
    )


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
                "universe": version["universe"],
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


def test_twap_approval_requires_minute_native_execution_evidence(
    tmp_path: Path, database_url: str
) -> None:
    version_id = create_strategy_version(
        database_url,
        tmp_path,
        config_overrides={
            "execution_method": "twap",
            "execution_slice_minutes": 20,
            "min_capacity_fill_ratio": 0.95,
        },
    )
    store = StrategyStore(database_url)
    version = store.get_version(version_id)
    artifact = tmp_path / "minute-backtest"
    artifact.mkdir()
    periods = {
        "start": PERIODS["test_start"].isoformat(),
        "end": PERIODS["test_end"].isoformat(),
    }
    backtest = store.create_backtest(
        version_id=version_id,
        dataset="snapshot",
        execution_dataset="ashare-5m",
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
                "execution_dataset": "ashare-5m",
                "benchmark": version["benchmark"],
                "universe": version["universe"],
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
    metrics.update(
        {
            "minute_execution_enforced": True,
            "capacity_fill_ratio": 0.99,
            "execution_model": {
                "method": "twap",
                "frequency": "5min",
                "price_assumption": "minute bar vwap fills",
                "strategy_contract_hash": version["config"]["execution_contract_hash"],
            },
        }
    )
    metrics["provenance"].update(
        {
            "execution_dataset_identity_sha256": "d" * 64,
            "execution_snapshot_manifest_sha256": "e" * 64,
            "execution_qlib_builder_sha256": "f" * 64,
            "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
            "execution_fields": ["vwap", "volume", "paused", "up_limit", "down_limit"],
            "execution_source_datasets": ["ashare_5m"],
            "execution_source_unit_contracts": {
                "ashare_5m": MINUTE_SOURCE_UNIT_CONTRACTS["ashare_5m"]
            },
            "execution_lineage_verified": True,
            "execution_source_lineage_id": "9" * 64,
        }
    )
    store.validate_backtest_artifacts(backtest["id"], metrics)

    proxy_metrics = deepcopy(metrics)
    proxy_metrics["minute_execution_enforced"] = False
    store.mark_backtest(backtest["id"], "succeeded", metrics=proxy_metrics)
    with pytest.raises(ValueError, match="minute-native"):
        store.approve(
            version_id,
            actor="risk-owner",
            reason="Daily proxy evidence must not approve a TWAP strategy.",
        )

    store.mark_backtest(backtest["id"], "succeeded", metrics=metrics)
    approved = store.approve(
        version_id,
        actor="risk-owner",
        reason="Minute-native execution evidence passed independent review.",
    )
    assert approved["status"] == "approved"
