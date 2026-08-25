from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Any

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


class _FailedStreamingProcess:
    def __init__(self, stdout: str, stderr: str, returncode: int = 17) -> None:
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode
        self.killed = False

    def wait(self, timeout: int | None = None) -> int:
        del timeout
        return self.returncode

    def kill(self) -> None:
        self.killed = True


def test_compose_command_error_streams_and_records_redacted_tail_without_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = _context(tmp_path)
    context.env_file.write_text(
        "POSTGRES_PASSWORD=server-password-123\n",
        encoding="utf-8",
    )
    process = _FailedStreamingProcess(
        "build output is visible\nconnection server-password-123 failed\n",
        "PLATFORM_SECRET_KEY=secret-from-command\nfatal compose marker\n",
    )
    popen_kwargs: dict[str, Any] = {}

    def fake_popen(*_args: Any, **kwargs: Any) -> _FailedStreamingProcess:
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError) as exc_info:
        context.run("up", "-d")

    captured = capsys.readouterr()
    message = str(exc_info.value)
    assert "build output is visible" in captured.out
    assert "fatal compose marker" in captured.err
    assert "fatal compose marker" in message
    assert "stdout" in message
    assert "stderr" in message
    assert "[REDACTED]" in captured.out
    assert "[REDACTED]" in captured.err
    assert "server-password-123" not in captured.out
    assert "server-password-123" not in message
    assert "secret-from-command" not in captured.err
    assert "secret-from-command" not in message
    assert popen_kwargs["stdout"] is subprocess.PIPE
    assert popen_kwargs["stderr"] is subprocess.PIPE


def test_compose_command_error_only_keeps_bounded_output_tail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = _FailedStreamingProcess(
        "early-marker\n" + ("x" * 20_000) + "\nfinal-marker\n",
        "",
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)

    with pytest.raises(RuntimeError) as exc_info:
        _context(tmp_path).run("up", "-d")

    message = str(exc_info.value)
    assert "earlier output omitted" in message
    assert "early-marker" not in message
    assert "final-marker" in message
    assert len(message) < 4_500


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


def test_compose_profiles_are_global_arguments(tmp_path: Path) -> None:
    base = _context(tmp_path)
    context = ComposeContext(
        base.project_name,
        base.env_file,
        base.compose_files,
        ("gpu",),
    )

    assert context.prefix[-2:] == ["--profile", "gpu"]


def test_compose_project_directory_preserves_old_release_relative_paths(
    tmp_path: Path,
) -> None:
    base = _context(tmp_path)
    old_release = tmp_path / "old-release"
    old_release.mkdir()
    context = ComposeContext(
        base.project_name,
        base.env_file,
        base.compose_files,
        project_directory=old_release,
    )

    position = context.prefix.index("--project-directory")
    assert context.prefix[position + 1] == str(old_release.resolve())
