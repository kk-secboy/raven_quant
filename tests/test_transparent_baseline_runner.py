import hashlib
import io
import subprocess
import tarfile
from pathlib import Path

import pytest

from quant_platform.transparent_baseline_runner import (
    CANONICAL_LF_TARGET_RECIPE_VERSION,
    CANONICAL_LF_TARGET_RUNNER_SHA256,
    OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
    OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256,
    TRANSPARENT_BASELINE_JOB_RUNNER_FIELD,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    require_transparent_baseline_runner,
)

pytestmark = pytest.mark.no_database


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def _config(
    recipe_version: str = OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
    runner_sha256: str = OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256,
) -> dict:
    return {
        "recipe_id": "short_relative_strength",
        "recipe_version": recipe_version,
        "transparent_baseline_bootstrap": {
            TRANSPARENT_BASELINE_RUNNER_FIELD: runner_sha256
        },
    }


def test_historical_v8_identity_is_not_rebound_to_current_lf_bytes() -> None:
    runner = Path(__file__).parents[1] / "scripts" / "run_multifactor_backtest.py"

    with pytest.raises(ValueError, match="transparent v8 runner bytes differ"):
        require_transparent_baseline_runner(
            config=_config(),
            job_payload={
                TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: (
                    OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
                )
            },
            runner_path=runner,
        )


def test_current_transparent_v9_runner_matches_canonical_lf_identity() -> None:
    runner = Path(__file__).parents[1] / "scripts" / "run_multifactor_backtest.py"

    assert require_transparent_baseline_runner(
        config=_config(
            CANONICAL_LF_TARGET_RECIPE_VERSION,
            CANONICAL_LF_TARGET_RUNNER_SHA256,
        ),
        job_payload={
            TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: CANONICAL_LF_TARGET_RUNNER_SHA256
        },
        runner_path=runner,
    ) == CANONICAL_LF_TARGET_RUNNER_SHA256


def test_runner_bytes_survive_git_blob_and_archive_with_autocrlf(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    runner_relative = Path("scripts/run_multifactor_backtest.py")
    runner_bytes = (root / runner_relative).read_bytes()
    attributes = (root / ".gitattributes").read_bytes()
    repo = tmp_path / "release-repo"
    archived_runner: bytes

    (repo / runner_relative.parent).mkdir(parents=True)
    (repo / ".gitattributes").write_bytes(attributes)
    (repo / runner_relative).write_bytes(runner_bytes)
    _git(repo, "init", "--quiet")
    _git(repo, "config", "core.autocrlf", "true")
    _git(repo, "config", "user.name", "QuantLab Tests")
    _git(repo, "config", "user.email", "tests@quantlab.invalid")
    _git(repo, "add", ".gitattributes", runner_relative.as_posix())
    _git(repo, "commit", "--quiet", "-m", "Seal transparent baseline runner")

    attributes = _git(
        repo,
        "check-attr",
        "text",
        "eol",
        "--",
        runner_relative.as_posix(),
    ).decode("utf-8")
    blob = _git(repo, "cat-file", "blob", f"HEAD:{runner_relative.as_posix()}")
    archive = _git(
        repo,
        "archive",
        "--format=tar",
        "HEAD",
        runner_relative.as_posix(),
    )
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        archived = tar.extractfile(runner_relative.as_posix())
        assert archived is not None
        archived_runner = archived.read()

    assert f"{runner_relative.as_posix()}: text: set\n" in attributes
    assert f"{runner_relative.as_posix()}: eol: lf\n" in attributes
    assert b"\r" not in runner_bytes
    assert blob == runner_bytes
    assert archived_runner == runner_bytes
    assert len(
        {
            hashlib.sha256(payload).hexdigest()
            for payload in (runner_bytes, blob, archived_runner)
        }
    ) == 1


def test_repair_migration_targets_the_current_runner_identity() -> None:
    migration = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0075_transparent_baseline_canonical_lf_repair.py"
    ).read_text(encoding="utf-8")

    assert CANONICAL_LF_TARGET_RUNNER_SHA256 in migration
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
