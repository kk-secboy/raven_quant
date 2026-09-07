"""Durable cleanup fence before a model job can release global resource tokens."""
from __future__ import annotations

import os
import re
import uuid
from contextlib import contextmanager
from pathlib import Path

from .model_cell_execution import (
    cleanup_cell_containers,
    model_cell_store_root,
    process_identity,
    stop_owned_process,
)
from .model_cell_store import atomic_json, read_json
from .model_prepared_cache import _locked, _mkdir, _safe

BATCH_MARKER = "active-model-batch.json"
MODEL_OWNER_LEASE_ENV = "QUANTLAB_MODEL_BATCH_LEASE"


class ModelCellCleanupPending(RuntimeError):
    """The job must remain running and no next job may claim its resource tokens."""


def model_owner_lease_path(output: Path) -> Path:
    return model_cell_store_root(output).parent / "model-owner.lease"


@contextmanager
def model_owner_lease(output: Path):
    path = model_owner_lease_path(output)
    _mkdir(path.parent)
    with _locked(path, exclusive=False) as lease:
        yield lease


def register_model_batch(
    output: Path, *, pid: int | None = None, job_claim: dict | None = None,
) -> Path:
    job_root = model_cell_store_root(output).parent
    marker = job_root / BATCH_MARKER
    expected = {
        "contract_version": "model-cell-cleanup-fence-v1",
        "result_path": str(output.absolute()), "job_root": str(job_root),
        "batch_process": process_identity(pid) if pid is not None else None,
        "job_claim": job_claim,
        "cleanup_status": "running",
    }
    if marker.exists():
        old = read_json(marker)
        if old.get("result_path") != expected["result_path"]:
            raise ModelCellCleanupPending("previous model batch cleanup remains pending")
        if job_claim is None:
            expected["job_claim"] = old.get("job_claim")
    atomic_json(marker, expected)
    return marker


def cleanup_model_batch(marker: Path) -> None:
    binding = None
    try:
        marker = _safe(marker)
        if not marker.exists():
            return
        binding = read_json(marker)
        output = Path(str(binding.get("result_path") or ""))
        job_root = model_cell_store_root(output).parent
        if (
            marker != job_root / BATCH_MARKER or binding.get("job_root") != str(job_root)
            or binding.get("contract_version") != "model-cell-cleanup-fence-v1"
        ):
            raise ValueError("model cleanup marker has an invalid job namespace")
        stop_owned_process(binding.get("batch_process"), grace_seconds=90)
        attempts = _safe(job_root / "attempts")
        for attempt in attempts.iterdir() if attempts.exists() else ():
            if not re.fullmatch(r"attempt-[0-9]{4}-[0-9a-f]{32}", attempt.name):
                continue
            controls = _safe(attempt / "cell-processes")
            for control in controls.iterdir() if controls.exists() else ():
                if not re.fullmatch(r"[0-9a-f]{32}", control.name):
                    raise ValueError("model cleanup has an unknown process directory")
                for name in ("process.json", "child-process.json"):
                    path = _safe(control / name)
                    if path.exists():
                        stop_owned_process(read_json(path).get("identity"))
        # No host cell process can launch another Docker container beyond this fence.
        cells = _safe(job_root / "model-cells")
        if cells.exists():
            for workspace in cells.glob("*/*/attempts/*/work"):
                cleanup_cell_containers(_safe(workspace))
        audit = _mkdir(job_root / "model-cleanup-audit") / f"{uuid.uuid4().hex}.json"
        atomic_json(audit, {**binding, "cleanup_status": "completed"})
        marker.unlink()
        if os.name != "nt":
            descriptor = os.open(marker.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except Exception as exc:
        if binding is not None:
            try:
                atomic_json(marker, {**binding, "cleanup_status": "blocked", "error": str(exc)})
            except (OSError, ValueError):
                pass  # A malformed/path-unsafe marker also retains the queue charge.
        raise ModelCellCleanupPending(
            f"model execution cleanup is pending; resource reservation retained: {exc}"
        ) from exc


def recover_model_batches(data_root: Path) -> dict:
    """Only ownerless batches may clean up; live shared leases retain their jobs."""
    root = _safe(data_root / "artifacts" / "model-evaluations")
    protected = []
    recoverable = {}
    if root.exists():
        for marker in root.glob(f"*/*/{BATCH_MARKER}"):
            with _locked(marker.parent / "model-owner.lease", exclusive=True,
                         blocking=False) as owner:
                if owner is None:
                    protected.append(marker.parent.name)
                    continue
                cleanup_model_batch(marker)
        for audit in root.glob("*/*/model-cleanup-audit/*.json"):
            binding = read_json(audit)
            job_root = audit.parent.parent
            if (
                binding.get("cleanup_status") == "completed"
                and binding.get("job_root") == str(job_root)
                and job_root.name not in protected
                and not (job_root / BATCH_MARKER).exists()
            ):
                claim = binding.get("job_claim")
                if isinstance(claim, dict) and claim.get("job_id") == job_root.name:
                    recoverable[(job_root.name, str(claim.get("started_at")))] = claim
    return {"protected_job_ids": tuple(sorted(set(protected))),
            "recoverable_model_claims": tuple(recoverable.values())}


def model_batch_marker(output: Path | None) -> Path | None:
    return model_cell_store_root(output).parent / BATCH_MARKER if output is not None else None


@contextmanager
def arm_model_batch_owner(output: Path):
    import json

    from .model_cell_execution import _parent_death_signal

    expected_lease = str(model_owner_lease_path(output))
    if os.environ.get(MODEL_OWNER_LEASE_ENV, expected_lease) != expected_lease:
        raise ValueError("model batch lease does not match its governed job namespace")
    with model_owner_lease(output):
        encoded = os.environ.get("QUANTLAB_MODEL_BATCH_OWNER")
        if encoded:
            owner = json.loads(encoded)
            if process_identity(owner["pid"]) != owner:
                raise ModelCellCleanupPending("model batch worker owner already exited")
            _parent_death_signal(owner["pid"])
        register_model_batch(output, pid=os.getpid())
        os.environ[MODEL_OWNER_LEASE_ENV] = expected_lease
        yield
