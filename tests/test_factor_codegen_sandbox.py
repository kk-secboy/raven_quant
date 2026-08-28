from __future__ import annotations

import subprocess

import pytest

from scripts import run_factor_codegen_sandbox as sandbox


@pytest.mark.no_database
def test_generated_factor_runs_without_network_or_platform_environment(tmp_path) -> None:
    workspace = (tmp_path / "run" / "factor").resolve()
    script = workspace / "factor.py"
    observed = sandbox._sandbox_command(workspace, script, "sha256:" + "a" * 64)
    assert observed[observed.index("--network") + 1] == "none"
    assert "--read-only" in observed
    assert "--cap-drop" in observed
    assert "DATABASE_URL" not in observed
    assert "OPENAI_API_KEY" not in observed
    assert observed[-3:] == ["python", "-I", str(script.resolve())]


@pytest.mark.no_database
def test_nested_sandbox_mount_uses_docker_daemon_source(tmp_path) -> None:
    workspace = (tmp_path / "inner" / "factor").resolve()
    script = workspace / "factor.py"
    docker_source = (tmp_path / "daemon" / "factor").resolve()
    observed = sandbox._sandbox_command(
        workspace,
        script,
        "sha256:" + "b" * 64,
        docker_source=docker_source,
    )
    assert observed[observed.index("--volume") + 1] == (
        f"{docker_source}:{workspace}:rw"
    )


@pytest.mark.no_database
def test_shared_worker_workspace_does_not_require_container_inspection(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_root = (tmp_path / "data").resolve()
    workspace = (shared_root / "artifacts" / "run" / "factor").resolve()
    workspace.mkdir(parents=True)
    monkeypatch.setenv("RDAGENT_DOCKER_SHARED_ROOT", str(shared_root))

    def unexpected_inspect(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "docker inspect")

    monkeypatch.setattr(subprocess, "check_output", unexpected_inspect)

    assert sandbox._docker_visible_workspace(workspace) == workspace
