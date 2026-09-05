from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import pytest

from quant_data.database import strategy_versions
from quant_platform import transparent_baseline_runner as runtime
from quant_platform.runtime_source_closure import (
    closure_paths,
    position_risk_source_closure_inventory,
)
from quant_platform.strategy_recipes import RECIPE_VERSION

pytestmark = pytest.mark.no_database
ROOT = Path(__file__).resolve().parents[1]


def test_v34_seals_the_actual_runner_and_economic_source_closure() -> None:
    assert RECIPE_VERSION == "qlib-rdagent-single-mainline-2026-09-05-v34"
    assert runtime.STRATEGY_RESEARCH_TARGET_RECIPE_VERSION == RECIPE_VERSION
    assert runtime.STRATEGY_RESEARCH_TARGET_RUNNER_SHA256 == hashlib.sha256(
        (ROOT / "scripts/run_multifactor_backtest.py").read_bytes()
    ).hexdigest()
    assert runtime.STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256 == (
        runtime.position_risk_bundle_sha256(ROOT)
    )
    assert {
        "scripts/run_multifactor_backtest.py",
        "scripts/run_recommendation_refresh.py",
        "src/quant_platform/strategy_rule_runtime.py",
        "src/quant_platform/simulation_engine.py",
    } <= set(closure_paths(position_risk_source_closure_inventory(ROOT)))


def test_v34_migration_and_metadata_preserve_the_v33_seal() -> None:
    previous = runpy.run_path(str(ROOT / "migrations/versions/0104_strategy_runtime_v33.py"))
    current = runpy.run_path(str(ROOT / "migrations/versions/0105_strategy_runtime_v34.py"))
    constraints = {item.name: str(item.sqltext) for item in strategy_versions.constraints
                   if hasattr(item, "sqltext")}
    assert current["down_revision"] == previous["revision"]
    assert current["RECIPE_VERSION"] == RECIPE_VERSION
    assert current["RUNNER_SHA256"] == runtime.STRATEGY_RESEARCH_TARGET_RUNNER_SHA256
    assert current["RUNTIME_BUNDLE_SHA256"] == (
        runtime.STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256
    )
    for migration in (previous, current):
        assert constraints[migration["CONSTRAINT"]] == migration["_constraint"]()
    assert runtime.STRATEGY_RESEARCH_V33_TARGET_RECIPE_VERSION == previous["RECIPE_VERSION"]
    assert runtime.STRATEGY_RESEARCH_V33_TARGET_RUNNER_SHA256 == previous["RUNNER_SHA256"]
    assert runtime.STRATEGY_RESEARCH_V33_TARGET_RUNTIME_BUNDLE_SHA256 == (
        previous["RUNTIME_BUNDLE_SHA256"]
    )
    assert current["RUNTIME_BUNDLE_SHA256"] != previous["RUNTIME_BUNDLE_SHA256"]


def test_current_runtime_actually_checks_the_full_source_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "sha256:" + "1" * 64
    monkeypatch.setenv(runtime.WORKER_RUNTIME_IMAGE_DIGEST_ENV, image)
    config = {
        "recipe_id": "short_relative_strength",
        "recipe_version": RECIPE_VERSION,
        "transparent_baseline_bootstrap": {
            runtime.TRANSPARENT_BASELINE_RUNNER_FIELD: (
                runtime.STRATEGY_RESEARCH_TARGET_RUNNER_SHA256
            ),
            runtime.TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: (
                runtime.STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256
            ),
            runtime.TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: image,
        },
    }
    payload = runtime.bind_transparent_baseline_job_identity(config=config, job_payload={})
    runner_path = ROOT / "scripts/run_multifactor_backtest.py"
    assert runtime.require_transparent_baseline_runner(
        config=config, job_payload=payload, runner_path=runner_path
    ) == runtime.STRATEGY_RESEARCH_TARGET_RUNNER_SHA256
    monkeypatch.setattr(runtime, "position_risk_bundle_sha256", lambda _root: "0" * 64)
    with pytest.raises(ValueError, match="runtime bundle differs"):
        runtime.require_transparent_baseline_runner(
            config=config, job_payload=payload, runner_path=runner_path
        )
