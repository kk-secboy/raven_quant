from pathlib import Path

import pytest

from quant_platform.transparent_baseline_runner import (
    OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
    OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256,
    TRANSPARENT_BASELINE_JOB_RUNNER_FIELD,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    require_transparent_baseline_runner,
)

pytestmark = pytest.mark.no_database


def _config() -> dict:
    return {
        "recipe_id": "short_relative_strength",
        "recipe_version": OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
        "transparent_baseline_bootstrap": {
            TRANSPARENT_BASELINE_RUNNER_FIELD: (
                OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
            )
        },
    }


def test_current_transparent_v8_runner_matches_sealed_identity() -> None:
    runner = Path(__file__).parents[1] / "scripts" / "run_multifactor_backtest.py"

    assert require_transparent_baseline_runner(
        config=_config(),
        job_payload={
            TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: (
                OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
            )
        },
        runner_path=runner,
    ) == OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256


def test_repair_migration_targets_the_current_runner_identity() -> None:
    migration = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0074_transparent_baseline_repair_chain.py"
    ).read_text(encoding="utf-8")

    assert OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256 in migration


def test_transparent_v8_runner_rejects_changed_bytes(tmp_path: Path) -> None:
    runner = tmp_path / "run_multifactor_backtest.py"
    runner.write_text("# changed runner\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runner bytes differ"):
        require_transparent_baseline_runner(
            config=_config(),
            job_payload={
                TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: (
                    OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
                )
            },
            runner_path=runner,
        )


def test_runner_repair_identity_is_forbidden_on_other_recipes(tmp_path: Path) -> None:
    runner = tmp_path / "run_multifactor_backtest.py"
    runner.write_text("# irrelevant\n", encoding="utf-8")
    config = {
        "recipe_id": "unrelated",
        "recipe_version": "other",
        "transparent_baseline_bootstrap": {
            TRANSPARENT_BASELINE_RUNNER_FIELD: (
                OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
            )
        },
    }

    with pytest.raises(ValueError, match="forbidden outside transparent v8"):
        require_transparent_baseline_runner(
            config=config,
            job_payload={},
            runner_path=runner,
        )
