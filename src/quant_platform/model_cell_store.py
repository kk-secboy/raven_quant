"""Durable receipts for exactly one pre-registered model experiment cell.

This is an execution journal, not a statistical result cache: its namespace includes
the complete batch registration and runtime. Terminal failures remain terminal;
only interrupted attempts without a committed outcome may resume.
"""
from __future__ import annotations

import json
import os
import re
import stat
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .model_prepared_cache import _locked, _mkdir, _safe
from .model_research_governance import canonical_sha256, file_sha256

CELL_STORE_VERSION = "model-cell-receipt-v1"


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = _safe(path)
    _mkdir(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _regular(path: Path) -> Path:
    path = _safe(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("model cell receipt references a nonregular or hardlinked file")
    return path


def read_json(path: Path) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("model cell JSON has duplicate fields")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("model cell JSON has nonfinite numbers")

    value = json.loads(_regular(path).read_text(encoding="utf-8"),
                       object_pairs_hook=unique, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError("model cell JSON must be an object")
    return value


def workspace_file_hashes(workspace: Path) -> dict[str, str]:
    """Bind every immutable output, including full prediction and checkpoint bytes."""
    output = _safe(workspace / "output")
    if not output.is_dir():
        raise ValueError("completed model cell has no output directory")
    hashes = {}
    for current, directories, files in os.walk(output, followlinks=False):
        for name in directories:
            _safe(Path(current) / name)
        for name in files:
            path = _regular(Path(current) / name)
            hashes[path.relative_to(workspace).as_posix()] = file_sha256(path)
    if not hashes:
        raise ValueError("completed model cell has empty output")
    for name in ("manifest.json", "model.py", "runner.py"):
        path = workspace / name
        if path.exists():
            hashes[name] = file_sha256(_regular(path))
    return dict(sorted(hashes.items()))


class ModelCellStore:
    def __init__(self, root: Path, batch_identity: Mapping[str, Any]):
        self.batch_identity = dict(batch_identity)
        self.batch_sha256 = canonical_sha256(self.batch_identity)
        self.root = _mkdir(Path(root) / self.batch_sha256)
        binding_path = self.root / "batch.json"
        with _locked(self.root / "batch.lock", exclusive=True):
            if binding_path.exists():
                if read_json(binding_path) != self.batch_identity:
                    raise ValueError("model cell batch identity changed")
            else:
                atomic_json(binding_path, self.batch_identity)

    def key(self, request: Mapping[str, Any]) -> str:
        return canonical_sha256({"batch_sha256": self.batch_sha256, "request": request})

    @contextmanager
    def claim(self, request: Mapping[str, Any]) -> Iterator[Path]:
        key = self.key(request)
        cell = _mkdir(self.root / key)
        with _locked(cell / "execution.lock", exclusive=True):
            yield cell

    def load(self, cell: Path, request: Mapping[str, Any]) -> dict[str, Any] | None:
        receipt_path = _safe(cell / "receipt.json")
        if not receipt_path.exists():
            return None
        receipt = read_json(receipt_path)
        digest = receipt.pop("receipt_sha256", None)
        if digest != canonical_sha256(receipt):
            raise ValueError("model cell receipt checksum mismatch")
        if (
            receipt.get("contract_version") != CELL_STORE_VERSION
            or receipt.get("batch_sha256") != self.batch_sha256
            or receipt.get("request") != request
            or receipt.get("status") not in {"completed", "failed", "resource_blocked"}
        ):
            raise ValueError("model cell receipt identity is invalid")
        workspace = self.workspace(cell, str(receipt.get("attempt_id") or ""))
        if receipt["status"] == "completed":
            if workspace_file_hashes(workspace) != receipt.get("files"):
                raise ValueError("completed model cell artifact bytes changed")
            result, evidence = receipt.get("result"), receipt.get("execution_evidence")
            if not isinstance(result, dict) or not isinstance(evidence, dict):
                raise ValueError("completed model cell has incomplete execution evidence")
        receipt["receipt_sha256"] = digest
        receipt["workspace"] = str(workspace)
        return receipt

    def workspace(self, cell: Path, attempt_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", attempt_id):
            raise ValueError("model cell attempt identity is invalid")
        return _safe(cell / "attempts" / attempt_id / "work")

    def begin(self, cell: Path, cleanup: Callable[[Path], None]) -> tuple[str, Path]:
        attempts = _mkdir(cell / "attempts")
        for old in attempts.iterdir():
            if not re.fullmatch(r"[0-9a-f]{32}", old.name) or not _safe(old).is_dir():
                raise ValueError("model cell has an unknown execution attempt")
            cleanup(_safe(old / "work"))
        attempt_id = uuid.uuid4().hex
        workspace = self.workspace(cell, attempt_id)
        _mkdir(workspace.parent)  # Executor creates work exactly once.
        return attempt_id, workspace

    def commit(
        self, cell: Path, request: Mapping[str, Any], *, attempt_id: str,
        status: str, result: dict | None = None, execution_evidence: dict | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        if (cell / "receipt.json").exists():
            raise ValueError("model cell terminal receipt cannot be overwritten")
        if status not in {"completed", "failed", "resource_blocked"}:
            raise ValueError("interrupted cells cannot publish a terminal receipt")
        workspace = self.workspace(cell, attempt_id)
        receipt = {
            "contract_version": CELL_STORE_VERSION, "batch_sha256": self.batch_sha256,
            "request": dict(request), "attempt_id": attempt_id, "status": status,
            "result": result, "execution_evidence": execution_evidence, "error": error,
            "files": workspace_file_hashes(workspace) if status == "completed" else {},
        }
        if status == "completed" and (not isinstance(result, dict)
                                       or not isinstance(execution_evidence, dict)):
            raise ValueError("completed model cell requires result and execution evidence")
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_json(cell / "receipt.json", receipt)
        return receipt
