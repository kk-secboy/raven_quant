from __future__ import annotations

import hashlib
import runpy
from copy import deepcopy
from pathlib import Path

import pytest

from quant_data.database import strategy_versions
from quant_platform import transparent_baseline_runner as runtime
from quant_platform.strategy_store import _bind_current_transparent_runtime_identity

pytestmark = pytest.mark.no_database
ROOT = Path(__file__).resolve().parents[1]


def test_v42_historical_migration_and_identity_remain_immutable() -> None:
    path = ROOT / "migrations/versions/0114_strategy_runtime_v42.py"
    assert hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == (
        "2dddd34393c3ef38ba1ae383cfcc0e7b446ccc4fb80ed80073d64dfe594ae5d2"
    )
    migration = runpy.run_path(str(path))
    assert migration["down_revision"] == "0113_strategy_runtime_v41"
    for field in ("RECIPE_VERSION", "RUNNER_SHA256", "RUNTIME_BUNDLE_SHA256"):
        assert migration[field] == getattr(runtime, "STRATEGY_RESEARCH_V42_TARGET_" + field)
    constraints = {
        item.name: str(item.sqltext) for item in strategy_versions.constraints
        if hasattr(item, "sqltext")
    }
    assert constraints[migration["CONSTRAINT"]] == migration["_constraint"]()


@pytest.mark.parametrize("recipe_id", [
    "short_relative_strength", "swing_trend", "long_quality_value",
])
def test_v42_evidence_is_preserved_and_cannot_authorize_v43(
    monkeypatch: pytest.MonkeyPatch, recipe_id: str,
) -> None:
    image = "sha256:" + "1" * 64
    monkeypatch.setenv(runtime.WORKER_RUNTIME_IMAGE_DIGEST_ENV, image)
    config = {
        "recipe_id": recipe_id,
        "recipe_version": runtime.STRATEGY_RESEARCH_V42_TARGET_RECIPE_VERSION,
        "evidence_mode": "sealed_final_oos",
        "transparent_baseline_bootstrap": {
            runtime.TRANSPARENT_BASELINE_RUNNER_FIELD:
                runtime.STRATEGY_RESEARCH_V42_TARGET_RUNNER_SHA256,
            runtime.TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD:
                runtime.STRATEGY_RESEARCH_V42_TARGET_RUNTIME_BUNDLE_SHA256,
            runtime.TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: image,
        },
    }
    original = deepcopy(config)
    assert _bind_current_transparent_runtime_identity(config) == original
    payload = runtime.bind_transparent_baseline_job_identity(config=config, job_payload={})
    with pytest.raises(ValueError, match="transparent v42 runtime bundle differs"):
        runtime.require_transparent_baseline_runner(
            config=config, job_payload=payload,
            runner_path=ROOT / "scripts/run_multifactor_backtest.py",
        )
    assert config == original
