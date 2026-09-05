from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from quant_data.database import strategy_versions
from quant_platform import transparent_baseline_runner as runtime
from quant_platform.strategy_recipes import RECIPE_VERSION

pytestmark = pytest.mark.no_database
ROOT = Path(__file__).resolve().parents[1]


def test_v34_preserves_its_historical_runtime_seal() -> None:
    assert runtime.STRATEGY_RESEARCH_V34_TARGET_RECIPE_VERSION == (
        "qlib-rdagent-single-mainline-2026-09-05-v34"
    )
    assert runtime.STRATEGY_RESEARCH_V34_TARGET_RUNNER_SHA256 == (
        "6df09f5bac9d8e10292a850da64707a1509d4896c4d65febd70aa3287413adf8"
    )
    assert runtime.STRATEGY_RESEARCH_V34_TARGET_RUNTIME_BUNDLE_SHA256 == (
        "6ea1c1e35b84923493f1f1b803189839296bf3c31f05d71ea915dba2f19450a4"
    )
    assert RECIPE_VERSION != runtime.STRATEGY_RESEARCH_V34_TARGET_RECIPE_VERSION
    assert runtime.STRATEGY_RESEARCH_TARGET_RECIPE_VERSION != (
        runtime.STRATEGY_RESEARCH_V34_TARGET_RECIPE_VERSION
    )


def test_v34_migration_and_metadata_preserve_the_v33_seal() -> None:
    previous = runpy.run_path(str(ROOT / "migrations/versions/0104_strategy_runtime_v33.py"))
    current = runpy.run_path(str(ROOT / "migrations/versions/0105_strategy_runtime_v34.py"))
    constraints = {item.name: str(item.sqltext) for item in strategy_versions.constraints
                   if hasattr(item, "sqltext")}
    assert current["down_revision"] == previous["revision"]
    assert current["RECIPE_VERSION"] == runtime.STRATEGY_RESEARCH_V34_TARGET_RECIPE_VERSION
    assert current["RUNNER_SHA256"] == runtime.STRATEGY_RESEARCH_V34_TARGET_RUNNER_SHA256
    assert current["RUNTIME_BUNDLE_SHA256"] == (
        runtime.STRATEGY_RESEARCH_V34_TARGET_RUNTIME_BUNDLE_SHA256
    )
    for migration in (previous, current):
        assert constraints[migration["CONSTRAINT"]] == migration["_constraint"]()
    assert runtime.STRATEGY_RESEARCH_V33_TARGET_RECIPE_VERSION == previous["RECIPE_VERSION"]
    assert runtime.STRATEGY_RESEARCH_V33_TARGET_RUNNER_SHA256 == previous["RUNNER_SHA256"]
    assert runtime.STRATEGY_RESEARCH_V33_TARGET_RUNTIME_BUNDLE_SHA256 == (
        previous["RUNTIME_BUNDLE_SHA256"]
    )
    assert previous["RECIPE_VERSION"] == "qlib-rdagent-single-mainline-2026-09-04-v33"
    assert previous["RUNNER_SHA256"] == (
        "aa6a106935705f341608922f3ba7ef8cb7d8333412b19203917cf2b28613c9e8"
    )
    assert previous["RUNTIME_BUNDLE_SHA256"] == (
        "f7e6511f0b8391b89713e483a1d01600033f984109af2961a902d23cc4437183"
    )
    assert current["RUNTIME_BUNDLE_SHA256"] != previous["RUNTIME_BUNDLE_SHA256"]


def test_v34_job_binding_preserves_old_seals_and_rejects_current_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_id = "short_relative_strength"
    recipe_version = runtime.STRATEGY_RESEARCH_V34_TARGET_RECIPE_VERSION
    runner_sha256 = runtime.STRATEGY_RESEARCH_V34_TARGET_RUNNER_SHA256
    bundle_sha256 = runtime.STRATEGY_RESEARCH_V34_TARGET_RUNTIME_BUNDLE_SHA256
    image = "sha256:" + "1" * 64
    monkeypatch.setenv(runtime.WORKER_RUNTIME_IMAGE_DIGEST_ENV, image)

    assert runtime.target_runner_for_recipe(recipe_id, recipe_version) == runner_sha256
    assert runtime.target_runtime_bundle_for_recipe(recipe_id, recipe_version) == bundle_sha256
    assert runtime.target_worker_runtime_image_for_recipe(recipe_id, recipe_version) == image
    config = {
        "recipe_id": recipe_id,
        "recipe_version": recipe_version,
        "transparent_baseline_bootstrap": {
            runtime.TRANSPARENT_BASELINE_RUNNER_FIELD: runner_sha256,
            runtime.TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: bundle_sha256,
            runtime.TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: image,
        },
    }
    payload = runtime.bind_transparent_baseline_job_identity(config=config, job_payload={})
    assert payload == {
        runtime.TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: runner_sha256,
        runtime.TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: bundle_sha256,
        runtime.TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: image,
    }
    assert config["recipe_version"] == recipe_version
    assert config["transparent_baseline_bootstrap"] == {
        runtime.TRANSPARENT_BASELINE_RUNNER_FIELD: runner_sha256,
        runtime.TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: bundle_sha256,
        runtime.TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: image,
    }

    # New runtime code cannot execute a job frozen against the historical runner.
    assert runner_sha256 != runtime.STRATEGY_RESEARCH_TARGET_RUNNER_SHA256
    assert bundle_sha256 != runtime.STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256
    with pytest.raises(ValueError, match="transparent v34 runner bytes differ"):
        runtime.require_transparent_baseline_runner(
            config=config,
            job_payload=payload,
            runner_path=ROOT / "scripts/run_multifactor_backtest.py",
        )
    # Independently exercise the historical closure guard with matching runner bytes.
    monkeypatch.setattr(runtime, "_file_sha256", lambda _path: runner_sha256)
    monkeypatch.setattr(
        runtime,
        "position_risk_bundle_sha256",
        lambda _root: runtime.STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256,
    )
    with pytest.raises(ValueError, match="transparent v34 runtime bundle differs"):
        runtime.require_transparent_baseline_runner(
            config=config,
            job_payload=payload,
            runner_path=ROOT / "scripts/run_multifactor_backtest.py",
        )
