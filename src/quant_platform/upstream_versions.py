from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
from pathlib import Path
from typing import Any

QLIB_COMMIT = "d5379c520f66a39953bad76234a7019a72796fd0"
RDAGENT_COMMIT = "4f9ecb005881cddc08df0124a2e894c018007679"
_COMMIT_EVIDENCE_SOURCES = frozenset({"repository", "distribution"})


def _distribution_version(names: tuple[str, ...]) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def _commit_from_version(version: str) -> str | None:
    match = re.search(r"(?:\+g|\.g)([0-9a-f]{7,40})(?:\b|$)", version.lower())
    if not match:
        return None
    candidate = match.group(1)
    for expected in (QLIB_COMMIT, RDAGENT_COMMIT):
        if expected.startswith(candidate):
            return expected
    return candidate if len(candidate) == 40 else None


def _repo_commit(path: str | None) -> str | None:
    if not path:
        return None
    root = Path(path)
    if not (root / ".git").exists():
        return None
    try:
        value = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip().lower()
    except (OSError, subprocess.SubprocessError):
        return None
    return value if len(value) == 40 else None


def require_upstream_runtime_identity(kind: str, value: Any) -> dict[str, Any]:
    if kind == "qlib":
        expected = QLIB_COMMIT
    elif kind == "rdagent":
        expected = RDAGENT_COMMIT
    else:
        raise ValueError(f"unsupported upstream runtime: {kind}")
    if not isinstance(value, dict):
        raise ValueError(f"{kind} runtime identity is required")
    version = str(value.get("version") or "").strip()
    commit = str(value.get("commit") or "").strip().lower()
    raw_evidence = value.get("commit_evidence")
    if not version or version in {"unknown", "source-checkout"}:
        raise ValueError(f"{kind} runtime distribution version is unavailable")
    if commit != expected:
        raise ValueError(
            f"{kind} runtime commit is not the validated pin: expected {expected}, got "
            f"{commit or 'missing'}"
        )
    if (
        not isinstance(raw_evidence, list)
        or not raw_evidence
        or any(
            not isinstance(source, str) or source not in _COMMIT_EVIDENCE_SOURCES
            for source in raw_evidence
        )
    ):
        raise ValueError(
            f"{kind} runtime commit has no verifiable repository or distribution evidence"
        )
    evidence = sorted(set(raw_evidence))
    if "distribution" in evidence and _commit_from_version(version) != expected:
        raise ValueError(
            f"{kind} distribution version does not identify the validated commit"
        )
    return {
        "name": kind,
        "version": version,
        "commit": commit,
        "commit_evidence": evidence,
    }


def upstream_runtime_identity(kind: str) -> dict[str, Any]:
    if kind == "qlib":
        expected = QLIB_COMMIT
        version = _distribution_version(("pyqlib", "qlib"))
        environment_commit = os.getenv("QLIB_COMMIT")
        repository_commit = _repo_commit(os.getenv("QLIB_REPO"))
    elif kind == "rdagent":
        expected = RDAGENT_COMMIT
        version = _distribution_version(("rdagent", "rd-agent"))
        environment_commit = os.getenv("RDAGENT_COMMIT")
        repository_commit = _repo_commit(os.getenv("RDAGENT_REPO"))
    else:
        raise ValueError(f"unsupported upstream runtime: {kind}")
    version_commit = _commit_from_version(version)
    if environment_commit and environment_commit.lower() != expected:
        raise RuntimeError(
            f"{kind} configured commit does not match the validated pin: "
            f"expected {expected}, got {environment_commit}"
        )
    if version == "unknown":
        raise RuntimeError(f"{kind} runtime distribution version is unavailable")
    verified = {
        source: value.lower()
        for source, value in {
            "repository": repository_commit,
            "distribution": version_commit,
        }.items()
        if value
    }
    if not verified:
        raise RuntimeError(
            f"{kind} runtime commit has no verifiable repository or distribution evidence"
        )
    if len(set(verified.values())) != 1:
        raise RuntimeError(f"{kind} runtime commit evidence disagrees: {verified}")
    commit = next(iter(verified.values()))
    if commit != expected:
        raise RuntimeError(
            f"{kind} runtime commit is not the validated pin: expected {expected}, got {commit}"
        )
    return require_upstream_runtime_identity(
        kind,
        {
            "name": kind,
            "version": version,
            "commit": commit,
            "commit_evidence": sorted(verified),
        },
    )
