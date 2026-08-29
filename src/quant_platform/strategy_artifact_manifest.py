from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

STRATEGY_BACKTEST_ARTIFACT_MANIFEST_VERSION = (
    "strategy-backtest-artifact-manifest-v1"
)
STRATEGY_BACKTEST_ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"

# result.json contains the manifest digest, artifact_manifest.json cannot hash
# itself, and the immutable input manifest already has its own separately
# validated execution_manifest_sha256 binding.
_EXCLUDED_PATHS = frozenset(
    {
        STRATEGY_BACKTEST_ARTIFACT_MANIFEST_NAME,
        "manifest.json",
        "result.json",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _artifact_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError("strategy backtest artifact tree must not contain symlinks")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        if relative in _EXCLUDED_PATHS:
            continue
        files[relative] = candidate
    return files


def write_backtest_artifact_manifest(artifact_root: str | Path) -> dict[str, Any]:
    """Freeze every completed multifactor output except the three cyclic files."""

    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise ValueError("strategy backtest artifact root is missing")
    files = _artifact_files(root)
    if not files:
        raise ValueError("strategy backtest produced no immutable result artifacts")
    payload: dict[str, Any] = {
        "version": STRATEGY_BACKTEST_ARTIFACT_MANIFEST_VERSION,
        "algorithm": "sha256",
        "excluded_paths": sorted(_EXCLUDED_PATHS),
        "files": [
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for relative, path in sorted(files.items())
        ],
    }
    manifest_path = root / STRATEGY_BACKTEST_ARTIFACT_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "path": STRATEGY_BACKTEST_ARTIFACT_MANIFEST_NAME,
        "version": STRATEGY_BACKTEST_ARTIFACT_MANIFEST_VERSION,
        "sha256": _sha256_file(manifest_path),
        "file_count": len(payload["files"]),
        "files": payload["files"],
    }


def validate_backtest_artifact_manifest(
    artifact_root: str | Path,
    *,
    expected_sha256: Any,
) -> dict[str, Any]:
    """Reject a missing, added, removed, redirected, or modified output file."""

    root = Path(artifact_root).resolve()
    manifest_path = root / STRATEGY_BACKTEST_ARTIFACT_MANIFEST_NAME
    if not _is_sha256(expected_sha256):
        raise ValueError("strategy backtest artifact manifest SHA-256 is missing")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("strategy backtest artifact manifest is missing")
    if _sha256_file(manifest_path) != str(expected_sha256).lower():
        raise ValueError("strategy backtest artifact manifest SHA-256 changed")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("strategy backtest artifact manifest is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("strategy backtest artifact manifest must be a JSON object")
    if payload.get("version") != STRATEGY_BACKTEST_ARTIFACT_MANIFEST_VERSION:
        raise ValueError("strategy backtest artifact manifest version is unsupported")
    if payload.get("algorithm") != "sha256" or payload.get("excluded_paths") != sorted(
        _EXCLUDED_PATHS
    ):
        raise ValueError("strategy backtest artifact manifest contract is invalid")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("strategy backtest artifact manifest contains no files")

    declared: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("strategy backtest artifact manifest entry is invalid")
        relative = str(entry.get("path") or "")
        pure = PurePosixPath(relative)
        if (
            not relative
            or "\\" in relative
            or pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or relative in _EXCLUDED_PATHS
            or relative in declared
        ):
            raise ValueError("strategy backtest artifact manifest path is unsafe or duplicated")
        if not isinstance(entry.get("bytes"), int) or int(entry["bytes"]) < 0:
            raise ValueError("strategy backtest artifact manifest byte count is invalid")
        if not _is_sha256(entry.get("sha256")):
            raise ValueError("strategy backtest artifact manifest file SHA-256 is invalid")
        declared[relative] = entry
    if list(declared) != sorted(declared):
        raise ValueError("strategy backtest artifact manifest files are not canonical")

    actual = _artifact_files(root)
    missing = sorted(set(declared) - set(actual))
    added = sorted(set(actual) - set(declared))
    if missing or added:
        raise ValueError(
            "strategy backtest artifact set changed "
            f"(missing={missing}, added={added})"
        )
    for relative, entry in declared.items():
        path = actual[relative]
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"strategy backtest artifact size changed: {relative}")
        if _sha256_file(path) != str(entry["sha256"]):
            raise ValueError(f"strategy backtest artifact SHA-256 changed: {relative}")
    return payload
