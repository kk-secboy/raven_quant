import hashlib
import io
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from quant_platform.runtime_source_closure import (
    position_risk_source_closure_inventory,
)
from quant_platform.transparent_baseline_runner import (
    CANONICAL_LF_TARGET_RECIPE_VERSION,
    CANONICAL_LF_TARGET_RUNNER_SHA256,
    DISCRETE_MAX_POSITION_REPAIR_TARGET_RECIPE_VERSION,
    DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNNER_SHA256,
    DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256,
    FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION,
    FAIL_CLOSED_EXECUTION_TARGET_RUNNER_SHA256,
    FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256,
    FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION,
    FILL_AWARE_HOLDING_AGE_TARGET_RUNNER_SHA256,
    FILL_AWARE_HOLDING_AGE_TARGET_RUNTIME_BUNDLE_SHA256,
    OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
    OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256,
    POSITION_RISK_TARGET_RECIPE_VERSION,
    POSITION_RISK_TARGET_RUNNER_SHA256,
    POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256,
    RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION,
    RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256,
    RUNTIME_ALIGNMENT_TARGET_RUNTIME_BUNDLE_SHA256,
    RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION,
    RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256,
    RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256,
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION,
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNNER_SHA256,
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256,
    TRANSPARENT_BASELINE_JOB_RUNNER_FIELD,
    TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
    WORKER_RUNTIME_IMAGE_DIGEST_ENV,
    position_risk_bundle_sha256,
    require_transparent_baseline_runner,
)

pytestmark = pytest.mark.no_database
_WORKER_IMAGE_DIGEST = "sha256:" + "d" * 64


@pytest.fixture(autouse=True)
def _sealed_worker_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WORKER_RUNTIME_IMAGE_DIGEST_ENV, _WORKER_IMAGE_DIGEST)


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def _v12_source_paths(project_root: Path) -> tuple[str, ...]:
    manifest = position_risk_source_closure_inventory(project_root)
    paths = {
        str(item["path"])
        for item in manifest["inventory"]
        if item["kind"] == "module"
    }
    # api.py is represented by a narrow fragment in the manifest; the release
    # still needs the source file from which that fragment is deterministically cut.
    paths.add("src/quant_platform/api.py")
    return tuple(sorted(paths))


def _config(
    recipe_version: str = OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
    runner_sha256: str = OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256,
) -> dict:
    bootstrap = {TRANSPARENT_BASELINE_RUNNER_FIELD: runner_sha256}
    if recipe_version == RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION:
        bootstrap[TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD] = (
            RUNTIME_ALIGNMENT_TARGET_RUNTIME_BUNDLE_SHA256
        )
    if recipe_version == RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION:
        bootstrap[TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD] = (
            RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256
        )
    if recipe_version == POSITION_RISK_TARGET_RECIPE_VERSION:
        bootstrap[TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD] = (
            POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256
        )
        bootstrap[TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD] = (
            _WORKER_IMAGE_DIGEST
        )
    if recipe_version == FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION:
        bootstrap[TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD] = (
            FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256
        )
        bootstrap[TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD] = (
            _WORKER_IMAGE_DIGEST
        )
    if recipe_version == FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION:
        bootstrap[TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD] = (
            FILL_AWARE_HOLDING_AGE_TARGET_RUNTIME_BUNDLE_SHA256
        )
        bootstrap[TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD] = (
            _WORKER_IMAGE_DIGEST
        )
    if recipe_version == SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION:
        bootstrap[TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD] = (
            SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
        )
        bootstrap[TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD] = (
            _WORKER_IMAGE_DIGEST
        )
    if recipe_version == DISCRETE_MAX_POSITION_REPAIR_TARGET_RECIPE_VERSION:
        bootstrap[TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD] = (
            DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
        )
        bootstrap[TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD] = (
            _WORKER_IMAGE_DIGEST
        )
    return {
        "recipe_id": "short_relative_strength",
        "recipe_version": recipe_version,
        "transparent_baseline_bootstrap": bootstrap,
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


def test_historical_v9_identity_is_not_rebound_to_current_v10_bytes() -> None:
    runner = Path(__file__).parents[1] / "scripts" / "run_multifactor_backtest.py"

    with pytest.raises(ValueError, match="transparent v9 runner bytes differ"):
        require_transparent_baseline_runner(
            config=_config(
                CANONICAL_LF_TARGET_RECIPE_VERSION,
                CANONICAL_LF_TARGET_RUNNER_SHA256,
            ),
            job_payload={
                TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: CANONICAL_LF_TARGET_RUNNER_SHA256
            },
            runner_path=runner,
        )


def test_historical_v10_identity_is_not_rebound_to_current_v12_bytes() -> None:
    runner = Path(__file__).parents[1] / "scripts" / "run_multifactor_backtest.py"

    with pytest.raises(ValueError, match="transparent v10 runner bytes differ"):
        require_transparent_baseline_runner(
            config=_config(
                RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION,
                RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256,
            ),
            job_payload={
                TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: (
                    RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256
                ),
                TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: (
                    RUNTIME_ALIGNMENT_TARGET_RUNTIME_BUNDLE_SHA256
                ),
            },
            runner_path=runner,
        )


def test_historical_v11_identity_is_not_rebound_to_current_v12_bytes() -> None:
    runner = Path(__file__).parents[1] / "scripts" / "run_multifactor_backtest.py"

    with pytest.raises(ValueError, match="transparent v11 runner bytes differ"):
        require_transparent_baseline_runner(
            config=_config(
                RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION,
                RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256,
            ),
            job_payload={
                TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: (
                    RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256
                ),
                TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: (
                    RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256
                ),
            },
            runner_path=runner,
        )


def test_historical_v12_identity_is_not_rebound_to_current_v13_runtime() -> None:
    runner = Path(__file__).parents[1] / "scripts" / "run_multifactor_backtest.py"

    with pytest.raises(ValueError, match="transparent v12 runtime bundle differs"):
        require_transparent_baseline_runner(
            config=_config(
                POSITION_RISK_TARGET_RECIPE_VERSION,
                POSITION_RISK_TARGET_RUNNER_SHA256,
            ),
            job_payload={
                TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: (
                    POSITION_RISK_TARGET_RUNNER_SHA256
                ),
                TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: (
                    POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256
                ),
                TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: (
                    _WORKER_IMAGE_DIGEST
                ),
            },
            runner_path=runner,
        )


def test_historical_v13_identity_is_not_rebound_to_current_v14_runtime() -> None:
    runner = Path(__file__).parents[1] / "scripts" / "run_multifactor_backtest.py"

    with pytest.raises(ValueError, match="transparent v13 runtime bundle differs"):
        require_transparent_baseline_runner(
            config=_config(
                FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION,
                FAIL_CLOSED_EXECUTION_TARGET_RUNNER_SHA256
            ),
            job_payload={
                TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: (
                    FAIL_CLOSED_EXECUTION_TARGET_RUNNER_SHA256
                ),
                TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: (
                    FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256
                ),
                TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: (
                    _WORKER_IMAGE_DIGEST
                ),
            },
            runner_path=runner,
        )


def test_historical_v15_identity_is_not_rebound_to_current_v16_runtime() -> None:
    runner = Path(__file__).parents[1] / "scripts" / "run_multifactor_backtest.py"

    with pytest.raises(ValueError, match="transparent v15 runtime bundle differs"):
        require_transparent_baseline_runner(
            config=_config(
                SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION,
                SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNNER_SHA256,
            ),
            job_payload={
                TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: (
                    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNNER_SHA256
                ),
                TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: (
                    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
                ),
                TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: (
                    _WORKER_IMAGE_DIGEST
                ),
            },
            runner_path=runner,
        )


def test_current_transparent_v16_runner_matches_position_cap_repair_identity() -> None:
    runner = Path(__file__).parents[1] / "scripts" / "run_multifactor_backtest.py"

    assert require_transparent_baseline_runner(
        config=_config(
            DISCRETE_MAX_POSITION_REPAIR_TARGET_RECIPE_VERSION,
            DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNNER_SHA256,
        ),
        job_payload={
            TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: (
                DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNNER_SHA256
            ),
            TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: (
                DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
            ),
            TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: (
                _WORKER_IMAGE_DIGEST
            ),
        },
        runner_path=runner,
    ) == DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNNER_SHA256


def test_v12_rejects_changed_imported_runtime_module(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    for relative in _v12_source_paths(root):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    policy = tmp_path / "src" / "quant_platform" / "portfolio_policy.py"
    policy.write_bytes(policy.read_bytes() + b"\n# unauthorized runtime change\n")

    with pytest.raises(ValueError, match="runtime bundle differs"):
        require_transparent_baseline_runner(
            config=_config(
                POSITION_RISK_TARGET_RECIPE_VERSION,
                POSITION_RISK_TARGET_RUNNER_SHA256,
            ),
            job_payload={
                TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: (
                    POSITION_RISK_TARGET_RUNNER_SHA256
                ),
                TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: (
                    POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256
                ),
                TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: (
                    _WORKER_IMAGE_DIGEST
                ),
            },
            runner_path=tmp_path / "scripts" / "run_multifactor_backtest.py",
        )


def test_v12_rejects_missing_worker_runtime_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(WORKER_RUNTIME_IMAGE_DIGEST_ENV)
    runner = Path(__file__).parents[1] / "scripts" / "run_multifactor_backtest.py"

    with pytest.raises(ValueError, match="worker runtime image digest is missing"):
        require_transparent_baseline_runner(
            config=_config(
                POSITION_RISK_TARGET_RECIPE_VERSION,
                POSITION_RISK_TARGET_RUNNER_SHA256,
            ),
            job_payload={
                TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: (
                    POSITION_RISK_TARGET_RUNNER_SHA256
                ),
                TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: (
                    POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256
                ),
                TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: (
                    _WORKER_IMAGE_DIGEST
                ),
            },
            runner_path=runner,
        )


def test_v12_rejects_worker_runtime_image_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(WORKER_RUNTIME_IMAGE_DIGEST_ENV, "sha256:" + "e" * 64)
    runner = Path(__file__).parents[1] / "scripts" / "run_multifactor_backtest.py"

    with pytest.raises(ValueError, match="worker runtime image differs"):
        require_transparent_baseline_runner(
            config=_config(
                POSITION_RISK_TARGET_RECIPE_VERSION,
                POSITION_RISK_TARGET_RUNNER_SHA256,
            ),
            job_payload={
                TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: (
                    POSITION_RISK_TARGET_RUNNER_SHA256
                ),
                TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: (
                    POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256
                ),
                TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: (
                    _WORKER_IMAGE_DIGEST
                ),
            },
            runner_path=runner,
        )


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


def test_v16_runtime_bundle_survives_git_archive_with_autocrlf(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    source_paths = _v12_source_paths(root)
    repo = tmp_path / "runtime-bundle-repo"
    archive_root = tmp_path / "runtime-bundle-archive"
    (repo / ".gitattributes").parent.mkdir(parents=True)
    (repo / ".gitattributes").write_bytes((root / ".gitattributes").read_bytes())
    for relative in source_paths:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / relative).read_bytes())
    _git(repo, "init", "--quiet")
    _git(repo, "config", "core.autocrlf", "true")
    _git(repo, "config", "user.name", "QuantLab Tests")
    _git(repo, "config", "user.email", "tests@quantlab.invalid")
    _git(repo, "add", ".gitattributes", *source_paths)
    _git(repo, "commit", "--quiet", "-m", "Seal v12 runtime bundle")
    archive = _git(repo, "archive", "--format=tar", "HEAD")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for relative in source_paths:
            archived = tar.extractfile(relative)
            assert archived is not None
            destination = archive_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archived.read())

    assert position_risk_bundle_sha256(root) == (
        DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
    )
    assert position_risk_bundle_sha256(repo) == (
        DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
    )
    assert position_risk_bundle_sha256(archive_root) == (
        DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
    )


def test_repair_migrations_pin_historical_and_current_runner_identities() -> None:
    v9_migration = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0075_transparent_baseline_canonical_lf_repair.py"
    ).read_text(encoding="utf-8")
    v10_migration = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0076_transparent_baseline_runtime_alignment_repair.py"
    ).read_text(encoding="utf-8")
    v11_migration = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0077_transparent_baseline_runtime_input_scope_repair.py"
    ).read_text(encoding="utf-8")
    v12_migration = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0078_transparent_baseline_v12_runtime_seal.py"
    ).read_text(encoding="utf-8")
    v13_migration = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0080_transparent_baseline_v13_runtime_seal.py"
    ).read_text(encoding="utf-8")
    v14_migration = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0081_transparent_baseline_v14_runtime_seal.py"
    ).read_text(encoding="utf-8")
    v15_migration = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0082_transparent_baseline_v15_runtime_repair.py"
    ).read_text(encoding="utf-8")
    v16_migration = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0084_transparent_baseline_v16_position_cap_repair.py"
    ).read_text(encoding="utf-8")

    assert CANONICAL_LF_TARGET_RUNNER_SHA256 in v9_migration
    assert OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256 in v9_migration
    assert CANONICAL_LF_TARGET_RUNNER_SHA256 in v10_migration
    assert RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256 in v10_migration
    assert RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256 in v11_migration
    assert RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256 in v11_migration
    assert RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256 not in v12_migration
    assert POSITION_RISK_TARGET_RUNNER_SHA256 in v12_migration
    assert POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256 in v12_migration
    assert FAIL_CLOSED_EXECUTION_TARGET_RUNNER_SHA256 in v13_migration
    assert FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256 in v13_migration
    assert FILL_AWARE_HOLDING_AGE_TARGET_RUNNER_SHA256 in v14_migration
    assert FILL_AWARE_HOLDING_AGE_TARGET_RUNTIME_BUNDLE_SHA256 in v14_migration
    assert SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNNER_SHA256 in v15_migration
    assert SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256 in v15_migration
    assert DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNNER_SHA256 in v16_migration
    assert DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256 in v16_migration


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

    with pytest.raises(ValueError, match="outside governed transparent recipes"):
        require_transparent_baseline_runner(
            config=config,
            job_payload={},
            runner_path=runner,
        )
