from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from quant_platform.backup_restore import ComposeContext

pytestmark = pytest.mark.no_database


def _context(tmp_path: Path) -> ComposeContext:
    env = tmp_path / ".env"
    compose = tmp_path / "compose.yaml"
    env.write_text("POSTGRES_PASSWORD=test\n", encoding="utf-8")
    compose.write_text("services: {}\n", encoding="utf-8")
    return ComposeContext("quantlab-test", env, (compose,))


def test_compose_command_error_preserves_stderr_without_command_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="PLATFORM_SECRET_KEY is required",
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        _context(tmp_path).run("config", "--quiet", capture=True)

    message = str(exc_info.value)
    assert "exit code 1" in message
    assert "PLATFORM_SECRET_KEY is required" in message
    assert "--env-file" not in message


def test_docker_command_error_preserves_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=125,
            stdout="",
            stderr="daemon unavailable",
        ),
    )

    with pytest.raises(RuntimeError, match="docker failed.*daemon unavailable"):
        _context(tmp_path).docker("info", capture=True)
