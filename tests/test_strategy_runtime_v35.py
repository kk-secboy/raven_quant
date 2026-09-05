from __future__ import annotations

import runpy
from datetime import date
from pathlib import Path

import pytest

from quant_data.database import strategy_versions
from quant_data.reference_data import REFERENCE_REFRESH_POLICIES, reference_refresh_bucket
from quant_platform import transparent_baseline_runner as runtime
from quant_platform.strategy_recipes import RECIPE_VERSION
from quant_platform.strategy_store import _transparent_worker_runtime_failures

pytestmark = pytest.mark.no_database
ROOT = Path(__file__).resolve().parents[1]


def test_v35_preserves_its_historical_runtime_seal() -> None:
    assert runtime.STRATEGY_RESEARCH_V35_TARGET_RECIPE_VERSION == (
        "qlib-rdagent-single-mainline-2026-09-05-v35"
    )
    assert runtime.STRATEGY_RESEARCH_V35_TARGET_RUNNER_SHA256 == (
        "6df09f5bac9d8e10292a850da64707a1509d4896c4d65febd70aa3287413adf8"
    )
    assert runtime.STRATEGY_RESEARCH_V35_TARGET_RUNTIME_BUNDLE_SHA256 == (
        "c55ef3e92a8ef1390983399c114c5bf571711f3f2583185e7ed805854e4a72e4"
    )
    assert RECIPE_VERSION != runtime.STRATEGY_RESEARCH_V35_TARGET_RECIPE_VERSION
    assert runtime.STRATEGY_RESEARCH_TARGET_RECIPE_VERSION != (
        runtime.STRATEGY_RESEARCH_V35_TARGET_RECIPE_VERSION
    )


def test_v35_seals_daily_convertible_bond_rating_refresh() -> None:
    assert REFERENCE_REFRESH_POLICIES["cb_rating"].cadence == "daily"
    assert reference_refresh_bucket("cb_rating", date(2026, 9, 4)) == "2026-09-04"
    assert reference_refresh_bucket("cb_rating", date(2026, 9, 5)) == "2026-09-05"


def test_v35_migration_and_metadata_preserve_the_v33_and_v34_seals() -> None:
    v33 = runpy.run_path(str(ROOT / "migrations/versions/0104_strategy_runtime_v33.py"))
    v34 = runpy.run_path(str(ROOT / "migrations/versions/0105_strategy_runtime_v34.py"))
    current = runpy.run_path(str(ROOT / "migrations/versions/0106_strategy_runtime_v35.py"))
    constraints = {
        item.name: str(item.sqltext)
        for item in strategy_versions.constraints
        if hasattr(item, "sqltext")
    }
    assert current["revision"] == "0106_strategy_runtime_v35"
    assert current["down_revision"] == v34["revision"]
    assert v34["down_revision"] == v33["revision"]
    assert current["RECIPE_VERSION"] == runtime.STRATEGY_RESEARCH_V35_TARGET_RECIPE_VERSION
    assert current["RUNNER_SHA256"] == runtime.STRATEGY_RESEARCH_V35_TARGET_RUNNER_SHA256
    assert current["RUNTIME_BUNDLE_SHA256"] == (
        runtime.STRATEGY_RESEARCH_V35_TARGET_RUNTIME_BUNDLE_SHA256
    )
    for migration in (v33, v34, current):
        assert constraints[migration["CONSTRAINT"]] == migration["_constraint"]()
    for version, migration in ((33, v33), (34, v34)):
        for field in ("RECIPE_VERSION", "RUNNER_SHA256", "RUNTIME_BUNDLE_SHA256"):
            assert getattr(runtime, f"STRATEGY_RESEARCH_V{version}_TARGET_{field}") == (
                migration[field]
            )
    assert current["RUNTIME_BUNDLE_SHA256"] != v34["RUNTIME_BUNDLE_SHA256"]


def test_v35_job_binding_preserves_old_seals_and_rejects_current_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "sha256:" + "1" * 64
    monkeypatch.setenv(runtime.WORKER_RUNTIME_IMAGE_DIGEST_ENV, image)
    recipe_version = runtime.STRATEGY_RESEARCH_V35_TARGET_RECIPE_VERSION
    runner_sha256 = runtime.STRATEGY_RESEARCH_V35_TARGET_RUNNER_SHA256
    bundle_sha256 = runtime.STRATEGY_RESEARCH_V35_TARGET_RUNTIME_BUNDLE_SHA256
    assert runtime.target_runner_for_recipe("short_relative_strength", recipe_version) == (
        runner_sha256
    )
    assert runtime.target_runtime_bundle_for_recipe("short_relative_strength", recipe_version) == (
        bundle_sha256
    )
    assert runtime.target_worker_runtime_image_for_recipe(
        "short_relative_strength", recipe_version
    ) == image
    config = {
        "recipe_id": "short_relative_strength",
        "recipe_version": recipe_version,
        "transparent_baseline_bootstrap": {
            runtime.TRANSPARENT_BASELINE_RUNNER_FIELD: runner_sha256,
            runtime.TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: bundle_sha256,
            runtime.TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: image,
        },
    }
    payload = runtime.bind_transparent_baseline_job_identity(config=config, job_payload={})
    runner_path = ROOT / "scripts/run_multifactor_backtest.py"
    assert config["recipe_version"] == recipe_version
    assert payload == {
        runtime.TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: runner_sha256,
        runtime.TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: bundle_sha256,
        runtime.TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: image,
    }
    with pytest.raises(ValueError, match="transparent v35 runner bytes differ"):
        runtime.require_transparent_baseline_runner(
            config=config, job_payload=payload, runner_path=runner_path
        )
    monkeypatch.setattr(runtime, "_file_sha256", lambda _path: runner_sha256)
    monkeypatch.setattr(
        runtime, "position_risk_bundle_sha256",
        lambda _root: runtime.STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256,
    )
    with pytest.raises(ValueError, match="transparent v35 runtime bundle differs"):
        runtime.require_transparent_baseline_runner(
            config=config, job_payload=payload, runner_path=runner_path
        )


@pytest.mark.parametrize("source", ["manifest", "provenance"])
@pytest.mark.parametrize("observed", [None, "sha256:" + "2" * 64])
@pytest.mark.parametrize("recipe_version", [
    runtime.STRATEGY_RESEARCH_V34_TARGET_RECIPE_VERSION,
    runtime.STRATEGY_RESEARCH_V35_TARGET_RECIPE_VERSION,
    runtime.STRATEGY_RESEARCH_TARGET_RECIPE_VERSION,
])
def test_formal_evidence_preserves_historical_and_current_worker_image_checks(
    source: str, observed: str | None, recipe_version: str,
) -> None:
    image = "sha256:" + "1" * 64
    version = {
        "config": {
            "recipe_id": "short_relative_strength",
            "recipe_version": recipe_version,
            "transparent_baseline_bootstrap": {
                runtime.TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: image,
            },
        },
    }
    manifest = {runtime.TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: image}
    provenance = {runtime.TRANSPARENT_BASELINE_RESULT_WORKER_RUNTIME_IMAGE_FIELD: image}
    assert _transparent_worker_runtime_failures(version, manifest, provenance) == []
    if source == "manifest":
        if observed is None:
            manifest.clear()
        else:
            manifest[runtime.TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD] = observed
        expected = "strategy backtest manifest"
    else:
        if observed is None:
            provenance.clear()
        else:
            provenance[runtime.TRANSPARENT_BASELINE_RESULT_WORKER_RUNTIME_IMAGE_FIELD] = observed
        expected = "formal result"
    assert _transparent_worker_runtime_failures(version, manifest, provenance) == [
        f"{expected} worker runtime image differs from the sealed version",
    ]
