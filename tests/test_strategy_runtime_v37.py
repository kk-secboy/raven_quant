from __future__ import annotations

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


def test_v37_preserves_its_historical_runtime_seal() -> None:
    assert runtime.STRATEGY_RESEARCH_V37_TARGET_RECIPE_VERSION == (
        "qlib-rdagent-single-mainline-2026-09-06-v37"
    )
    assert runtime.STRATEGY_RESEARCH_V37_TARGET_RUNNER_SHA256 == (
        "372d8744947144be822aeda155fa2907b6aefda464422b759d1083b772148054"
    )
    assert runtime.STRATEGY_RESEARCH_V37_TARGET_RUNTIME_BUNDLE_SHA256 == (
        "77eccc2d3b7150027e9dde45ff261049272331cb1952b047c28367f552ffd4b1"
    )
    assert RECIPE_VERSION != runtime.STRATEGY_RESEARCH_V37_TARGET_RECIPE_VERSION


def test_v37_migration_and_metadata_preserve_the_previous_runtime_seals() -> None:
    migrations = [
        runpy.run_path(str(ROOT / "migrations/versions" / f"{revision}.py"))
        for revision in (
            "0104_strategy_runtime_v33", "0105_strategy_runtime_v34",
            "0106_strategy_runtime_v35", "0107_strategy_runtime_v36",
            "0108_strategy_runtime_v37",
        )
    ]
    current = migrations[-1]
    constraints = {
        item.name: str(item.sqltext)
        for item in strategy_versions.constraints
        if hasattr(item, "sqltext")
    }
    assert current["revision"] == "0108_strategy_runtime_v37"
    assert current["RECIPE_VERSION"] == runtime.STRATEGY_RESEARCH_V37_TARGET_RECIPE_VERSION
    assert current["RUNNER_SHA256"] == runtime.STRATEGY_RESEARCH_V37_TARGET_RUNNER_SHA256
    assert current["RUNTIME_BUNDLE_SHA256"] == (
        runtime.STRATEGY_RESEARCH_V37_TARGET_RUNTIME_BUNDLE_SHA256
    )
    for previous, following in zip(migrations, migrations[1:], strict=False):
        assert following["down_revision"] == previous["revision"]
    for version, migration in zip((33, 34, 35, 36), migrations[:-1], strict=True):
        for field in ("RECIPE_VERSION", "RUNNER_SHA256", "RUNTIME_BUNDLE_SHA256"):
            assert getattr(runtime, f"STRATEGY_RESEARCH_V{version}_TARGET_{field}") == (
                migration[field]
            )
    for migration in migrations:
        assert constraints[migration["CONSTRAINT"]] == migration["_constraint"]()
    # The runner script is unchanged; the imported capacity implementation is sealed separately.
    assert current["RUNNER_SHA256"] == migrations[-2]["RUNNER_SHA256"]
    assert current["RUNTIME_BUNDLE_SHA256"] != migrations[-2]["RUNTIME_BUNDLE_SHA256"]


@pytest.mark.parametrize(
    "recipe_id", ["short_relative_strength", "swing_trend", "long_quality_value"],
)
def test_v37_job_binding_preserves_history_and_rejects_current_code(
    monkeypatch: pytest.MonkeyPatch, recipe_id: str,
) -> None:
    image = "sha256:" + "1" * 64
    monkeypatch.setenv(runtime.WORKER_RUNTIME_IMAGE_DIGEST_ENV, image)
    config = {
        "recipe_id": recipe_id,
        "recipe_version": runtime.STRATEGY_RESEARCH_V37_TARGET_RECIPE_VERSION,
        "transparent_baseline_bootstrap": {
            runtime.TRANSPARENT_BASELINE_RUNNER_FIELD: (
                runtime.STRATEGY_RESEARCH_V37_TARGET_RUNNER_SHA256
            ),
            runtime.TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: (
                runtime.STRATEGY_RESEARCH_V37_TARGET_RUNTIME_BUNDLE_SHA256
            ),
            runtime.TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: image,
        },
    }
    before = deepcopy(config)
    payload = runtime.bind_transparent_baseline_job_identity(config=config, job_payload={})
    expected_payload = {
        runtime.TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: (
            runtime.STRATEGY_RESEARCH_V37_TARGET_RUNNER_SHA256
        ),
        runtime.TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: (
            runtime.STRATEGY_RESEARCH_V37_TARGET_RUNTIME_BUNDLE_SHA256
        ),
        runtime.TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: image,
    }
    # Historical binding is retained; current runner bytes cannot execute it.
    with pytest.raises(ValueError, match="transparent v37 runner bytes differ"):
        runtime.require_transparent_baseline_runner(
            config=config, job_payload=payload,
            runner_path=ROOT / "scripts/run_multifactor_backtest.py",
        )
    assert config == before
    assert payload == expected_payload


def test_v37_seal_normalization_keeps_economic_source_changes_visible() -> None:
    relative = "src/quant_platform/transparent_baseline_runner.py"
    source = (ROOT / relative).read_text(encoding="utf-8")
    original = _normalized_seal_payload(relative, source)
    changed_seal = source.replace(
        runtime.STRATEGY_RESEARCH_V36_TARGET_RUNTIME_BUNDLE_SHA256, "0" * 64,
    )
    assert _normalized_seal_payload(relative, changed_seal) == original
    changed_current_seal = source.replace(
        runtime.STRATEGY_RESEARCH_V37_TARGET_RUNTIME_BUNDLE_SHA256, "f" * 64,
    )
    assert _normalized_seal_payload(relative, changed_current_seal) == original
    changed_logic = source + '\nECONOMIC_FLAG = "' + "f" * 64 + '"\n'
    assert _normalized_seal_payload(relative, changed_logic) != original
    for relative in (
        "src/quant_platform/portfolio_policy.py", "scripts/run_multifactor_backtest.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert _normalized_seal_payload(relative, source + "\n# changed economic source\n") != (
            _normalized_seal_payload(relative, source)
        )


def test_v37_database_normalization_does_not_hide_admission_rules() -> None:
    relative = "src/quant_data/database.py"
    source = (ROOT / relative).read_text(encoding="utf-8")
    original = _normalized_seal_payload(relative, source)
    changed_seal = source.replace(
        runtime.STRATEGY_RESEARCH_V37_TARGET_RUNTIME_BUNDLE_SHA256, "f" * 64,
    )
    assert _normalized_seal_payload(relative, changed_seal) == original
    assert _normalized_seal_payload(
        relative, source.replace("AND evidence_mode = 'sealed_final_oos'", "AND true"),
    ) != original
    assert _normalized_seal_payload(
        relative, source.replace("'short_relative_strength','swing_trend'", "'unapproved_recipe'"),
    ) != original


def test_v37_historical_admission_does_not_rebind_old_evidence() -> None:
    config = {
        "recipe_id": "short_relative_strength",
        "recipe_version": runtime.STRATEGY_RESEARCH_V37_TARGET_RECIPE_VERSION,
        "evidence_mode": "sealed_final_oos",
        "transparent_baseline_bootstrap": {
            runtime.TRANSPARENT_BASELINE_RUNNER_FIELD: (
                runtime.STRATEGY_RESEARCH_V37_TARGET_RUNNER_SHA256
            ),
            runtime.TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: (
                runtime.STRATEGY_RESEARCH_V37_TARGET_RUNTIME_BUNDLE_SHA256
            ),
            runtime.TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: "sha256:" + "1" * 64,
        },
    }
    original = deepcopy(config)
    assert _bind_current_transparent_runtime_identity(config) == original
    assert config == original


def test_v37_migration_only_adds_and_removes_its_own_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = runpy.run_path(str(ROOT / "migrations/versions/0108_strategy_runtime_v37.py"))
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
            "ck_strategy_versions_v37_runtime_identity", "strategy_versions",
            migration["_constraint"](),
        ), {"schema": "quantlab"}),
        ("drop", ("ck_strategy_versions_v37_runtime_identity", "strategy_versions"), {
            "schema": "quantlab", "type_": "check",
        }),
    ]
