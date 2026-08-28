#!/usr/bin/env python3
"""Execute RD-Agent generated factor code in a sealed Docker sandbox.

RD-Agent's pinned factor coder accepts a ``python_bin`` command but otherwise
executes generated Python in the research worker.  This wrapper keeps the
upstream integration intact while removing network, credentials and platform
mounts from the generated-code process.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

DEFAULT_IMAGE = "quantlab-factor-sandbox:v2"
MAX_LINKED_INPUT_BYTES = 8 * 1024 * 1024 * 1024


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _materialize_linked_inputs(workspace: Path, trusted_roots: tuple[Path, ...]) -> None:
    """Replace trusted RD-Agent data symlinks with sandbox-visible files."""

    total = 0
    for entry in workspace.iterdir():
        if not entry.is_symlink():
            continue
        target = entry.resolve(strict=True)
        if not any(_within(target, root) for root in trusted_roots) or not target.is_file():
            raise RuntimeError(f"untrusted generated-code input link: {entry}")
        total += target.stat().st_size
        if total > MAX_LINKED_INPUT_BYTES:
            raise RuntimeError("generated-code linked inputs exceed 8 GiB")
        entry.unlink()
        shutil.copyfile(target, entry)
        entry.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _image_identity(image: str) -> str:
    output = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        text=True,
        timeout=30,
    ).strip()
    if not output.startswith("sha256:") or len(output) != 71:
        raise RuntimeError("factor sandbox image did not resolve to an immutable id")
    return output


def _sandbox_command(
    workspace: Path,
    script: Path,
    image_id: str,
    *,
    docker_source: Path | None = None,
) -> list[str]:
    source = docker_source or workspace
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "128",
        "--memory",
        "4g",
        "--cpus",
        "2",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=256m",
        "--user",
        "65534:65534",
        "--env",
        "PYTHONHASHSEED=0",
        "--volume",
        f"{source}:{workspace}:rw",
        "--workdir",
        str(workspace),
        image_id,
        "python",
        "-I",
        str(script),
    ]


def _docker_visible_workspace(workspace: Path) -> Path:
    """Resolve an inner Qlib mount back to the sibling Docker daemon path."""

    shared_root = Path(os.environ.get("RDAGENT_DOCKER_SHARED_ROOT", "/data"))
    try:
        resolved_shared_root = shared_root.resolve(strict=True)
    except OSError:
        resolved_shared_root = None
    # The normal factor coder runs in the RD-Agent worker.  Its /data volume
    # is mounted at the identical path in the sibling Docker daemon, so no
    # container inspection is needed (and the daemon cannot inspect the outer
    # Compose worker container in the first place).
    if resolved_shared_root is not None and _within(workspace, resolved_shared_root):
        return workspace

    container_id = str(os.getenv("HOSTNAME") or "").strip()
    if not container_id or not Path("/.dockerenv").exists():
        return workspace
    try:
        payload = json.loads(
            subprocess.check_output(
                ["docker", "inspect", container_id], text=True, timeout=15
            )
        )
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot resolve the governed Qlib workspace mount") from exc
    mounts = payload[0].get("Mounts", []) if isinstance(payload, list) and payload else []
    candidates: list[tuple[int, Path]] = []
    for mount in mounts:
        if not isinstance(mount, dict):
            continue
        destination = Path(str(mount.get("Destination") or "")).resolve()
        source = Path(str(mount.get("Source") or ""))
        try:
            relative = workspace.relative_to(destination)
        except ValueError:
            continue
        candidates.append((len(destination.parts), source / relative))
    if not candidates:
        raise RuntimeError("generated-code workspace is not a governed Docker mount")
    return max(candidates, key=lambda item: item[0])[1]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: run_factor_codegen_sandbox.py FACTOR_SCRIPT")
    script = Path(argv[1]).resolve(strict=True)
    workspace = script.parent.resolve(strict=True)
    shared_root = Path(os.environ.get("RDAGENT_DOCKER_SHARED_ROOT", "/data")).resolve(
        strict=True
    )
    if script.name not in {"factor.py"} and script.suffix != ".py":
        raise RuntimeError("generated-code entrypoint must be a Python file")
    docker_source = _docker_visible_workspace(workspace).resolve(strict=True)
    if not _within(docker_source, shared_root):
        raise RuntimeError("generated-code path is outside the governed shared workspace")

    runtime_root = next(
        (parent for parent in docker_source.parents if parent.name == "docker-runtime"),
        None,
    )
    trusted_roots = [Path("/opt/rdagent/git_ignore_folder").resolve()]
    if runtime_root is not None:
        trusted_roots.append(runtime_root / "factor-source-data")
    _materialize_linked_inputs(workspace, tuple(trusted_roots))
    workspace.chmod(workspace.stat().st_mode | stat.S_IWOTH | stat.S_IXOTH)
    script.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    image_id = _image_identity(os.environ.get("FACTOR_SANDBOX_IMAGE", DEFAULT_IMAGE))
    (workspace / "factor_sandbox_identity.json").write_text(
        json.dumps({"image_id": image_id, "network": "none"}, sort_keys=True),
        encoding="utf-8",
    )

    command = _sandbox_command(
        workspace, script, image_id, docker_source=docker_source
    )
    return subprocess.run(command, check=False, timeout=3600).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
