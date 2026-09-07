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
)
from quant_platform.strategy_recipes import RECIPE_VERSION
from quant_platform.strategy_store import _bind_current_transparent_runtime_identity

pytestmark = pytest.mark.no_database
ROOT = Path(__file__).resolve().parents[1]


def test_v39_preserves_its_historical_runtime_seal() -> None:
    migration = (ROOT / "migrations/versions/0110_strategy_runtime_v39.py").read_bytes()
    assert hashlib.sha256(migration.replace(b"\r\n", b"\n")).hexdigest() == (
        "d84d205099d3341bc3401c641d166ff71da5a13ceaebbd5ff0339a73a7772821"
    )
    assert runtime.STRATEGY_RESEARCH_V39_TARGET_RECIPE_VERSION == (
        "qlib-rdagent-single-mainline-2026-09-06-v39"
    )
    assert runtime.STRATEGY_RESEARCH_V39_TARGET_RUNNER_SHA256 == (
        "4a94328bf92b822da69530ffccccf0067cc3d2727d512d8d54f51095ef422717"
    )
    assert runtime.STRATEGY_RESEARCH_V39_TARGET_RUNTIME_BUNDLE_SHA256 == (
        "ae93c0c72ea3951d968d458c5e1663f15984c4365366757898a2aa55be2b4b06"
    )
    assert RECIPE_VERSION != runtime.STRATEGY_RESEARCH_V39_TARGET_RECIPE_VERSION


def test_v39_migration_and_metadata_preserve_the_previous_runtime_seals() -> None:
    migrations = [
        runpy.run_path(str(ROOT / "migrations/versions" / f"{revision}.py"))
        for revision in (
            "0104_strategy_runtime_v33", "0105_strategy_runtime_v34",
            "0106_strategy_runtime_v35", "0107_strategy_runtime_v36",
            "0108_strategy_runtime_v37", "0109_strategy_runtime_v38",
            "0110_strategy_runtime_v39",
        )
    ]
    current = migrations[-1]
    constraints = {
        item.name: str(item.sqltext)
        for item in strategy_versions.constraints
        if hasattr(item, "sqltext")
    }
    assert current["revision"] == "0110_strategy_runtime_v39"
    assert current["RECIPE_VERSION"] == runtime.STRATEGY_RESEARCH_V39_TARGET_RECIPE_VERSION
    assert current["RUNNER_SHA256"] == runtime.STRATEGY_RESEARCH_V39_TARGET_RUNNER_SHA256
    assert current["RUNTIME_BUNDLE_SHA256"] == (
        runtime.STRATEGY_RESEARCH_V39_TARGET_RUNTIME_BUNDLE_SHA256
    )
    for previous, following in zip(migrations, migrations[1:], strict=False):
        assert following["down_revision"] == previous["revision"]
    for version, migration in zip((33, 34, 35, 36, 37, 38), migrations[:-1], strict=True):
        for field in ("RECIPE_VERSION", "RUNNER_SHA256", "RUNTIME_BUNDLE_SHA256"):
            assert getattr(runtime, f"STRATEGY_RESEARCH_V{version}_TARGET_{field}") == (
                migration[field]
            )
    for migration in migrations:
        assert constraints[migration["CONSTRAINT"]] == migration["_constraint"]()
    assert current["RUNNER_SHA256"] == migrations[-2]["RUNNER_SHA256"]
    assert current["RUNTIME_BUNDLE_SHA256"] != migrations[-2]["RUNTIME_BUNDLE_SHA256"]


@pytest.mark.parametrize(
    "recipe_id", ["short_relative_strength", "swing_trend", "long_quality_value"],
)
def test_v39_job_binding_preserves_history_and_rejects_current_code(
    monkeypatch: pytest.MonkeyPatch, recipe_id: str,
) -> None:
    image = "sha256:" + "1" * 64
    monkeypatch.setenv(runtime.WORKER_RUNTIME_IMAGE_DIGEST_ENV, image)
    config = {
        "recipe_id": recipe_id,
        "recipe_version": runtime.STRATEGY_RESEARCH_V39_TARGET_RECIPE_VERSION,
        "transparent_baseline_bootstrap": {
            runtime.TRANSPARENT_BASELINE_RUNNER_FIELD: (
                runtime.STRATEGY_RESEARCH_V39_TARGET_RUNNER_SHA256
            ),
            runtime.TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: (
                runtime.STRATEGY_RESEARCH_V39_TARGET_RUNTIME_BUNDLE_SHA256
            ),
            runtime.TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: image,
        },
    }
    before = deepcopy(config)
    payload = runtime.bind_transparent_baseline_job_identity(config=config, job_payload={})
    expected_payload = {
        runtime.TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: (
            runtime.STRATEGY_RESEARCH_V39_TARGET_RUNNER_SHA256
        ),
        runtime.TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: (
            runtime.STRATEGY_RESEARCH_V39_TARGET_RUNTIME_BUNDLE_SHA256
        ),
        runtime.TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: image,
    }
    # Historical v39 evidence cannot authorize the new v41 runner.
    with pytest.raises(ValueError, match="transparent v39 runner bytes differ"):
        runtime.require_transparent_baseline_runner(
            config=config, job_payload=payload,
            runner_path=ROOT / "scripts/run_multifactor_backtest.py",
        )
    monkeypatch.setattr(runtime, "_file_sha256",
                        lambda _: runtime.STRATEGY_RESEARCH_V39_TARGET_RUNNER_SHA256)
    monkeypatch.setattr(runtime, "position_risk_bundle_sha256",
                        lambda _: runtime.STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256)
    with pytest.raises(ValueError, match="transparent v39 runtime bundle differs"):
        runtime.require_transparent_baseline_runner(
            config=config, job_payload=payload,
            runner_path=ROOT / "scripts/run_multifactor_backtest.py",
        )
    assert config == before
    assert payload == expected_payload


@pytest.mark.parametrize("version", [33, 34, 35, 36, 37, 38])
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
    failure = "runner bytes differ"
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


def test_v39_seal_normalization_keeps_economic_source_changes_visible() -> None:
    relative = "src/quant_platform/transparent_baseline_runner.py"
    source = (ROOT / relative).read_text(encoding="utf-8")
    original = _normalized_seal_payload(relative, source)
    changed_seal = source.replace(
        runtime.STRATEGY_RESEARCH_V38_TARGET_RUNTIME_BUNDLE_SHA256, "0" * 64,
    )
    assert _normalized_seal_payload(relative, changed_seal) == original
    changed_current_seal = source.replace(
        runtime.STRATEGY_RESEARCH_V39_TARGET_RUNTIME_BUNDLE_SHA256, "f" * 64,
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


def test_v39_database_normalization_does_not_hide_admission_rules() -> None:
    relative = "src/quant_data/database.py"
    source = (ROOT / relative).read_text(encoding="utf-8")
    original = _normalized_seal_payload(relative, source)
    changed_seal = source.replace(
        runtime.STRATEGY_RESEARCH_V39_TARGET_RUNTIME_BUNDLE_SHA256, "f" * 64,
    )
    assert _normalized_seal_payload(relative, changed_seal) == original
    assert _normalized_seal_payload(
        relative, source.replace("AND evidence_mode = 'sealed_final_oos'", "AND true"),
    ) != original
    assert _normalized_seal_payload(
        relative, source.replace("'short_relative_strength','swing_trend'", "'unapproved_recipe'"),
    ) != original


def test_v39_historical_admission_does_not_rebind_old_evidence() -> None:
    config = {
        "recipe_id": "short_relative_strength",
        "recipe_version": runtime.STRATEGY_RESEARCH_V39_TARGET_RECIPE_VERSION,
        "evidence_mode": "sealed_final_oos",
        "transparent_baseline_bootstrap": {
            runtime.TRANSPARENT_BASELINE_RUNNER_FIELD: (
                runtime.STRATEGY_RESEARCH_V39_TARGET_RUNNER_SHA256
            ),
            runtime.TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: (
                runtime.STRATEGY_RESEARCH_V39_TARGET_RUNTIME_BUNDLE_SHA256
            ),
            runtime.TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: "sha256:" + "1" * 64,
        },
    }
    original = deepcopy(config)
    assert _bind_current_transparent_runtime_identity(config) == original
    assert config == original


def test_v39_migration_only_adds_and_removes_its_own_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = runpy.run_path(str(ROOT / "migrations/versions/0110_strategy_runtime_v39.py"))
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
            "ck_strategy_versions_v39_runtime_identity", "strategy_versions",
            migration["_constraint"](),
        ), {"schema": "quantlab"}),
        ("drop", ("ck_strategy_versions_v39_runtime_identity", "strategy_versions"), {
            "schema": "quantlab", "type_": "check",
        }),
    ]
