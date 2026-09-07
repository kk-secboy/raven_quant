from __future__ import annotations

import hashlib
import runpy
from copy import deepcopy
from pathlib import Path

import pytest

from quant_data.database import strategy_versions
from quant_platform import transparent_baseline_runner as runtime
from quant_platform.runtime_source_closure import (
    _normalized_seal_payload,
    closure_paths,
    position_risk_source_closure_inventory,
)
from quant_platform.strategy_recipes import RECIPE_VERSION
from quant_platform.strategy_store import _bind_current_transparent_runtime_identity

pytestmark = pytest.mark.no_database
ROOT = Path(__file__).resolve().parents[1]


def test_v40_seals_the_unchanged_runner_and_registered_activity_schema() -> None:
    assert RECIPE_VERSION == "qlib-rdagent-single-mainline-2026-09-07-v40"
    assert runtime.STRATEGY_RESEARCH_TARGET_RECIPE_VERSION == RECIPE_VERSION
    assert runtime.STRATEGY_RESEARCH_TARGET_RUNNER_SHA256 == hashlib.sha256(
        (ROOT / "scripts/run_multifactor_backtest.py").read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    assert runtime.STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256 == (
        runtime.position_risk_bundle_sha256(ROOT)
    )
    paths = set(closure_paths(position_risk_source_closure_inventory(ROOT)))
    assert len(paths) == 112
    assert {
        "scripts/run_multifactor_backtest.py",
        "scripts/run_recommendation_refresh.py",
        "src/quant_platform/portfolio_policy.py",
        "src/quant_platform/discrete_constraints.py",
        "src/quant_platform/qlib_policy_strategy.py",
        "src/quant_platform/qlib_exchange.py",
        "src/quant_platform/eligibility.py",
        "src/quant_platform/strategy_rule_runtime.py",
        "src/quant_platform/simulation_engine.py",
        "src/quant_data/reference_data.py",
        "src/quant_platform/strategy_research_evaluation.py",
    } <= paths
    assert "src/quant_platform/parameter_experiment_store.py" not in paths


def test_v40_migration_and_metadata_preserve_the_previous_runtime_seals() -> None:
    migrations = [
        runpy.run_path(str(ROOT / "migrations/versions" / f"{revision}.py"))
        for revision in (
            "0104_strategy_runtime_v33", "0105_strategy_runtime_v34",
            "0106_strategy_runtime_v35", "0107_strategy_runtime_v36",
            "0108_strategy_runtime_v37", "0109_strategy_runtime_v38",
            "0110_strategy_runtime_v39", "0112_strategy_runtime_v40",
        )
    ]
    current = migrations[-1]
    constraints = {
        item.name: str(item.sqltext)
        for item in strategy_versions.constraints
        if hasattr(item, "sqltext")
    }
    assert current["revision"] == "0112_strategy_runtime_v40"
    assert current["RECIPE_VERSION"] == RECIPE_VERSION
    assert current["RUNNER_SHA256"] == runtime.STRATEGY_RESEARCH_TARGET_RUNNER_SHA256
    assert current["RUNTIME_BUNDLE_SHA256"] == (
        runtime.STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256
    )
    for previous, following in zip(migrations[:-2], migrations[1:-1], strict=True):
        assert following["down_revision"] == previous["revision"]
    for version, migration in zip((33, 34, 35, 36, 37, 38, 39), migrations[:-1], strict=True):
        for field in ("RECIPE_VERSION", "RUNNER_SHA256", "RUNTIME_BUNDLE_SHA256"):
            assert getattr(runtime, f"STRATEGY_RESEARCH_V{version}_TARGET_{field}") == (
                migration[field]
            )
    assert current["down_revision"] == "0111_autopilot_research_events"
    activity = runpy.run_path(str(ROOT / "migrations/versions/0111_autopilot_research_events.py"))
    assert activity["down_revision"] == migrations[-2]["revision"]
    for migration in migrations:
        assert constraints[migration["CONSTRAINT"]] == migration["_constraint"]()
    assert current["RUNNER_SHA256"] == migrations[-2]["RUNNER_SHA256"]
    assert current["RUNTIME_BUNDLE_SHA256"] != migrations[-2]["RUNTIME_BUNDLE_SHA256"]


@pytest.mark.parametrize(
    "recipe_id", ["short_relative_strength", "swing_trend", "long_quality_value"],
)
def test_v40_bound_job_checks_current_runner_bundle_and_worker_image(
    monkeypatch: pytest.MonkeyPatch, recipe_id: str,
) -> None:
    image = "sha256:" + "1" * 64
    monkeypatch.setenv(runtime.WORKER_RUNTIME_IMAGE_DIGEST_ENV, image)
    config = {
        "recipe_id": recipe_id,
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
    with monkeypatch.context() as changed:
        changed.setattr(runtime, "_file_sha256", lambda _path: "0" * 64)
        with pytest.raises(ValueError, match="transparent v40 runner bytes differ"):
            runtime.require_transparent_baseline_runner(
                config=config, job_payload=payload, runner_path=runner_path
            )
    with monkeypatch.context() as changed:
        changed.setattr(runtime, "position_risk_bundle_sha256", lambda _root: "0" * 64)
        with pytest.raises(ValueError, match="transparent v40 runtime bundle differs"):
            runtime.require_transparent_baseline_runner(
                config=config, job_payload=payload, runner_path=runner_path
            )
    payload[runtime.TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD] = "sha256:" + "2" * 64
    with pytest.raises(ValueError, match="transparent v40 worker runtime image differs"):
        runtime.require_transparent_baseline_runner(
            config=config, job_payload=payload, runner_path=runner_path
        )


@pytest.mark.parametrize("version", [33, 34, 35, 36, 37, 38, 39])
@pytest.mark.parametrize(
    "recipe_id", ["short_relative_strength", "swing_trend", "long_quality_value"],
)
def test_historical_identity_cannot_execute_current_runner_or_current_closure(
    monkeypatch: pytest.MonkeyPatch, version: int, recipe_id: str,
) -> None:
    image = "sha256:" + "1" * 64
    monkeypatch.setenv(runtime.WORKER_RUNTIME_IMAGE_DIGEST_ENV, image)
    prefix = f"STRATEGY_RESEARCH_V{version}_TARGET_"
    runner = getattr(runtime, prefix + "RUNNER_SHA256")
    bundle = getattr(runtime, prefix + "RUNTIME_BUNDLE_SHA256")
    config = {
        "recipe_id": recipe_id,
        "recipe_version": getattr(runtime, prefix + "RECIPE_VERSION"),
        "transparent_baseline_bootstrap": {
            runtime.TRANSPARENT_BASELINE_RUNNER_FIELD: runner,
            runtime.TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: bundle,
            runtime.TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: image,
        },
    }
    before = deepcopy(config)
    payload = runtime.bind_transparent_baseline_job_identity(config=config, job_payload={})
    old_payload = deepcopy(payload)
    assert payload[runtime.TRANSPARENT_BASELINE_JOB_RUNNER_FIELD] == runner
    assert payload[runtime.TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD] == bundle
    failure = "runtime bundle differs" if version >= 38 else "runner bytes differ"
    with pytest.raises(ValueError, match=f"transparent v{version} {failure}"):
        runtime.require_transparent_baseline_runner(
            config=config, job_payload=payload,
            runner_path=ROOT / "scripts/run_multifactor_backtest.py",
        )
    # Even supplying old runner bytes cannot authorize the new imported code.
    monkeypatch.setattr(runtime, "_file_sha256", lambda _path: runner)
    monkeypatch.setattr(runtime, "position_risk_bundle_sha256",
                        lambda _root: runtime.STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256)
    with pytest.raises(ValueError, match=f"transparent v{version} runtime bundle differs"):
        runtime.require_transparent_baseline_runner(
            config=config, job_payload=payload,
            runner_path=ROOT / "scripts/run_multifactor_backtest.py",
        )
    assert config == before
    assert payload == old_payload


def test_v40_seal_normalization_keeps_economic_source_changes_visible() -> None:
    relative = "src/quant_platform/transparent_baseline_runner.py"
    source = (ROOT / relative).read_text(encoding="utf-8")
    original = _normalized_seal_payload(relative, source)
    changed_seal = source.replace(
        runtime.STRATEGY_RESEARCH_V39_TARGET_RUNTIME_BUNDLE_SHA256, "0" * 64,
    )
    assert _normalized_seal_payload(relative, changed_seal) == original
    changed_current_seal = source.replace(
        runtime.STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256, "f" * 64,
    )
    assert _normalized_seal_payload(relative, changed_current_seal) == original
    changed_logic = source + '\nECONOMIC_FLAG = "' + "f" * 64 + '"\n'
    assert _normalized_seal_payload(relative, changed_logic) != original
    for relative in (
        "src/quant_platform/portfolio_policy.py", "src/quant_platform/discrete_constraints.py",
        "src/quant_platform/qlib_policy_strategy.py", "src/quant_platform/qlib_exchange.py",
        "scripts/run_multifactor_backtest.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert _normalized_seal_payload(relative, source + "\n# changed economic source\n") != (
            _normalized_seal_payload(relative, source)
        )


def test_v40_database_normalization_does_not_hide_admission_rules() -> None:
    relative = "src/quant_data/database.py"
    source = (ROOT / relative).read_text(encoding="utf-8")
    original = _normalized_seal_payload(relative, source)
    changed_seal = source.replace(
        runtime.STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256, "f" * 64,
    )
    assert _normalized_seal_payload(relative, changed_seal) == original
    assert _normalized_seal_payload(
        relative, source.replace("AND evidence_mode = 'sealed_final_oos'", "AND true"),
    ) != original
    assert _normalized_seal_payload(
        relative, source.replace("'short_relative_strength','swing_trend'", "'unapproved_recipe'"),
    ) != original


@pytest.mark.parametrize(
    "recipe_id", ["short_relative_strength", "swing_trend", "long_quality_value"],
)
def test_v40_current_admission_binds_release_without_weakening_sealed_oos(
    monkeypatch: pytest.MonkeyPatch, recipe_id: str,
) -> None:
    image = "sha256:" + "1" * 64
    monkeypatch.setenv(runtime.WORKER_RUNTIME_IMAGE_DIGEST_ENV, image)
    config = {
        "recipe_id": recipe_id, "recipe_version": RECIPE_VERSION,
        "evidence_mode": "sealed_final_oos",
    }
    original = deepcopy(config)
    bound = _bind_current_transparent_runtime_identity(config)
    assert config == original
    assert bound["transparent_baseline_bootstrap"] == {
        runtime.TRANSPARENT_BASELINE_RUNNER_FIELD: runtime.STRATEGY_RESEARCH_TARGET_RUNNER_SHA256,
        runtime.TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: (
            runtime.STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256
        ),
        runtime.TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: image,
    }
    for mode in ("pre_registered_replay", None):
        with pytest.raises(ValueError, match="v40.*requires sealed final OOS"):
            _bind_current_transparent_runtime_identity({**config, "evidence_mode": mode})
    for field in bound["transparent_baseline_bootstrap"]:
        mismatched = deepcopy(bound)
        mismatched["transparent_baseline_bootstrap"][field] = "invalid"
        before = deepcopy(mismatched)
        with pytest.raises(ValueError, match="differs from this release"):
            _bind_current_transparent_runtime_identity(mismatched)
        assert mismatched == before
    monkeypatch.delenv(runtime.WORKER_RUNTIME_IMAGE_DIGEST_ENV)
    with pytest.raises(ValueError, match="v40 worker runtime image digest is missing"):
        _bind_current_transparent_runtime_identity(config)


def test_v40_migration_only_adds_and_removes_its_own_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = runpy.run_path(str(ROOT / "migrations/versions/0112_strategy_runtime_v40.py"))
    operations = []
    monkeypatch.setattr(
        migration["op"], "create_check_constraint",
        lambda *args, **kwargs: operations.append(("create", args, kwargs)),
    )
    monkeypatch.setattr(
        migration["op"], "drop_constraint",
        lambda *args, **kwargs: operations.append(("drop", args, kwargs)),
    )
    monkeypatch.setattr(
        migration["op"], "execute",
        lambda *_args, **_kwargs: pytest.fail("runtime seal migration cannot change old records"),
    )
    migration["upgrade"]()
    migration["downgrade"]()
    assert operations == [
        ("create", (
            "ck_strategy_versions_v40_runtime_identity", "strategy_versions",
            migration["_constraint"](),
        ), {"schema": "quantlab"}),
        ("drop", ("ck_strategy_versions_v40_runtime_identity", "strategy_versions"), {
            "schema": "quantlab", "type_": "check",
        }),
    ]
