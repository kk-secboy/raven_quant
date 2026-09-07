from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from governance_fixtures import PERIODS, formal_backtest_metrics

from quant_platform.api import StrategyConfigRequest
from quant_platform.strategy_artifact_manifest import write_backtest_artifact_manifest
from quant_platform.strategy_store import _multifactor_manifest_failures

pytestmark = pytest.mark.no_database


@pytest.fixture(params=[False, True], ids=["generated-returns", "existing-returns"])
def formal_fixture(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[dict, dict, dict]:
    code = tmp_path / "factor.py"
    values = tmp_path / "factor-values.h5"
    code.write_text("value = 1\n", encoding="utf-8")
    values.write_bytes(b"frozen fixture values")
    factor = {
        "factor_candidate_id": "fixture-factor", "weight": 1.0, "direction": 1,
        "code_path": str(code), "values_path": str(values), "source_iteration": None,
        "code_sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
    }
    version = {
        "id": "fixture-version", "economic_hypothesis_group": "fixture-family",
        "evidence_mode": "sealed_final_oos", "benchmark": "SH000300", "universe": "cn_all",
        "config": StrategyConfigRequest().model_dump(), "factors": [factor],
    }
    artifact = tmp_path / "formal-backtest"
    artifact.mkdir()
    periods = {"start": PERIODS["test_start"].isoformat(),
               "end": PERIODS["test_end"].isoformat()}
    manifest = artifact / "manifest.json"
    manifest.write_text(json.dumps({
        "strategy_version_id": version["id"], "dataset": "snapshot",
        "benchmark": version["benchmark"], "universe": version["universe"],
        "config": version["config"], "periods": periods,
        "factors": [{"candidate_id": factor["factor_candidate_id"],
                     **{key: factor[key] for key in ("weight", "direction", "code_sha256")}}],
    }), encoding="utf-8")
    existing_bytes = None
    daily_path = artifact / "daily_returns.parquet"
    if request.param:
        pd.DataFrame({
            "return": [0.001 + (i % 7) * 0.00001 for i in range(60)],
            "cost": 0.00005, "bench": 0.00002,
        }, index=pd.bdate_range(periods["start"], periods=60)).to_parquet(daily_path)
        existing_bytes = daily_path.read_bytes()
    metrics = formal_backtest_metrics(version, manifest, hypothesis_group_evidence={
        "shared_experiment_count": 1, "economic_hypothesis_group": "fixture-family",
        "strategy_version_ids": [version["id"]],
    })
    if existing_bytes is not None:
        assert daily_path.read_bytes() == existing_bytes
    backtest = {
        "dataset": "snapshot", "artifact_path": str(artifact),
        "evidence_mode": "sealed_final_oos",
        "periods": {**periods, "historical_start": PERIODS["train_start"].isoformat(),
                    "historical_end": PERIODS["valid_end"].isoformat()},
    }
    return version, backtest, metrics


def test_formal_fixture_passes_actual_manifest_and_paired_return_validation(formal_fixture) -> None:
    version, backtest, metrics = formal_fixture
    assert _multifactor_manifest_failures(version, backtest, metrics) == []
    artifact = Path(backtest["artifact_path"])
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    for value in (manifest, metrics, metrics["provenance"]):
        assert value["evidence_mode"] == "sealed_final_oos"
        assert value["evaluation_mode"] == "formal_final_oos"
        assert value["final_oos_opened"] is True
    daily = pd.read_parquet(artifact / "daily_returns.parquet")
    assert metrics["formal_validation"]["paired_block_bootstrap"]["observations"] == len(daily)


def test_formal_fixture_rejects_changed_returns_even_with_updated_file_receipt(formal_fixture):
    version, backtest, metrics = formal_fixture
    artifact = Path(backtest["artifact_path"])
    daily_path = artifact / "daily_returns.parquet"
    daily = pd.read_parquet(daily_path)
    daily["return"] -= 0.01
    daily.to_parquet(daily_path)
    receipt = write_backtest_artifact_manifest(artifact)
    metrics["provenance"]["artifact_manifest_sha256"] = receipt["sha256"]
    metrics["provenance"]["artifact_manifest_file_count"] = receipt["file_count"]
    failures = _multifactor_manifest_failures(version, backtest, metrics)
    assert failures and all("paired bootstrap artifact recomputation failed" in f for f in failures)


def test_formal_fixture_rejects_inconsistent_oos_authority(formal_fixture) -> None:
    version, backtest, metrics = formal_fixture
    metrics["final_oos_opened"] = False
    assert _multifactor_manifest_failures(version, backtest, metrics) == [
        "sealed final OOS evidence authority is inconsistent",
    ]
