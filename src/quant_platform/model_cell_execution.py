"""Bounded subprocess scheduling and exact-attempt recovery for model cells."""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .model_cell_store import ModelCellStore, atomic_json, read_json
from .model_compute_policy import (
    fixed_model_cell_grid_policy,
    governed_cell_resource_allocation,
)
from .model_prepared_cache import _mkdir, _safe
from .model_research_governance import file_sha256

CELL_EXECUTION_VERSION = "model-cell-execution-v1"
_LOG = logging.getLogger(__name__)


class ModelCellCancelled(BaseException):
    """Administrative interruption is resumable, never a statistical retry."""


def process_identity(pid: int) -> dict[str, Any] | None:
    """Linux PID birth + namespace prevents killing a reused or foreign PID."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        stat_fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat_fields[0] == "Z":
            return None
        return {
            "pid": pid, "start_ticks": stat_fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "pid_namespace": os.readlink(f"/proc/{pid}/ns/pid"),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def stop_owned_process(identity: dict[str, Any] | None, *, grace_seconds: float = 30) -> None:
    if identity is None:
        return
    pid = identity.get("pid")
    if type(pid) is not int or pid <= 1 or pid == os.getpid():
        raise ValueError("model cell process identity is invalid")
    if process_identity(pid) != identity:
        return  # Exited, another PID namespace, rebooted host, or reused PID.
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while process_identity(pid) == identity:
        if time.monotonic() >= deadline:
            os.kill(pid, signal.SIGKILL)
            break
        time.sleep(0.1)
    deadline = time.monotonic() + 5
    while process_identity(pid) == identity:
        if time.monotonic() >= deadline:
            raise RuntimeError("model cell process did not exit after termination")
        time.sleep(0.1)


def cell_runtime_identity(runner_path: Path) -> dict[str, Any]:
    image = str(os.environ.get("MODEL_SANDBOX_IMAGE") or "")
    if not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("durable model cells require a digest-pinned sandbox image")
    inspected = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True, text=True, timeout=30, check=False,
    )
    image_id = inspected.stdout.strip()
    if inspected.returncode or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("durable model cells cannot verify the sandbox image")
    source_root = Path(__file__).resolve().parents[1]
    sources = {
        str(path.relative_to(source_root).as_posix()): file_sha256(path)
        for package in ("quant_platform", "quant_data")
        for path in sorted((source_root / package).rglob("*.py"))
    }
    for name in ("evaluate_model_batch.py", "model_sandbox_runner.py", "prepare_model_data.py"):
        path = runner_path.with_name(name)
        sources[f"scripts/{name}"] = file_sha256(path)
    return {
        "contract_version": CELL_EXECUTION_VERSION,
        "sandbox_image": image, "sandbox_image_id": image_id,
        "source_files": sources, "grid_policy": fixed_model_cell_grid_policy(),
    }


def cell_batch_identity(manifest: Mapping[str, Any], runner_path: Path) -> dict[str, Any]:
    if not str(manifest.get("research_run_id") or ""):
        raise ValueError("durable model cells require a pre-registered research run")
    # Keep the complete ordered candidate set, source paths, labels and registration.
    return {"manifest": dict(manifest), "runtime": cell_runtime_identity(runner_path)}


def model_cell_store_root(output: Path) -> Path:
    output = _safe(output.absolute())
    attempt = output.parent
    if (
        output.name != "result.json" or attempt.parent.name != "attempts"
        or not re.fullmatch(r"attempt-[0-9]{4}-[0-9a-f]{32}", attempt.name)
        or attempt.parent.parent.parent.parent.name != "model-evaluations"
    ):
        raise ValueError("model cell journal must belong to one governed evaluation job")
    return attempt.parent.parent / "model-cells"


def _call_request(call: Mapping[str, Any]) -> dict[str, Any]:
    provider = Path(call["provider_path"])
    binding = {}
    for name in (
        "quantlab-rdagent-dataset-view.json", "calendars/day.txt",
        "calendars/day_future.txt", "instruments/cn_all.txt",
    ):
        path = provider / name
        if not path.is_file():
            raise ValueError("model cell requires a complete pre-final physical provider view")
        binding[name] = file_sha256(path)
    code = Path(call["code_path"])
    receipt_sha256 = call["manifest"].get("model_dataset_view_receipt_sha256")
    if receipt_sha256 is not None:
        receipt = _safe(provider.parent / "receipt.json")
        if not receipt.is_file() or file_sha256(receipt) != receipt_sha256:
            raise ValueError("model cell physical provider receipt changed")
        binding["model_dataset_view_receipt_sha256"] = receipt_sha256
    if file_sha256(code) != call["manifest"]["code_sha256"]:
        raise ValueError("model cell candidate code changed")
    return {
        "manifest": dict(call["manifest"]), "provider_binding": binding,
        "code_sha256": file_sha256(code), "timeout_seconds": call["timeout_seconds"],
    }


def cleanup_cell_containers(workspace: Path) -> None:
    """Stop only CIDs whose inspected work mount proves this exact cell owns them."""
    workspace = _safe(workspace)
    for relative, mount in (
        ("container.cid", workspace),
        ("data-preparation/container.cid", workspace / "data-preparation" / "inputs"),
    ):
        cidfile = _safe(workspace / relative)
        if not cidfile.exists():
            continue
        cid = cidfile.read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", cid):
            raise ValueError("model cell has an invalid owned container identity")
        inspected = subprocess.run(["docker", "inspect", cid], capture_output=True,
                                   text=True, timeout=30, check=False)
        if inspected.returncode:
            if "No such object" in inspected.stderr or "No such container" in inspected.stderr:
                continue
            raise RuntimeError("cannot confirm that a model cell container has stopped")
        values = json.loads(inspected.stdout)
        if len(values) != 1 or values[0].get("Id") != cid or not any(
            value.get("Destination") == "/work"
            and Path(str(value.get("Source") or "")).absolute() == mount.absolute()
            for value in values[0].get("Mounts", [])
        ):
            raise ValueError("model cell container work mount does not match its owner")
        removed = subprocess.run(["docker", "rm", "-f", cid], capture_output=True,
                                 text=True, timeout=30, check=False)
        if removed.returncode and not any(
            text in removed.stderr for text in ("No such object", "No such container")
        ):
            raise RuntimeError("model cell container could not be stopped")
        checked = subprocess.run(["docker", "inspect", cid], capture_output=True,
                                 text=True, timeout=30, check=False)
        if not checked.returncode or not any(
            text in checked.stderr for text in ("No such object", "No such container")
        ):
            raise RuntimeError("model cell container removal could not be verified")


@contextmanager
def cancellation_signals():
    def cancel(_number, _frame):
        raise ModelCellCancelled("model cell batch was interrupted")

    previous = {}
    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, cancel)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _parent_death_signal(expected_parent: int) -> None:
    if sys.platform.startswith("linux"):
        # Parent SIGKILL must not leave an orphan cell consuming the next job's budget.
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot arm model cell parent-death signal")
    if os.getppid() != expected_parent:
        raise ModelCellCancelled("model cell owner already exited")


def execute_stored_cell(
    store: ModelCellStore, call: dict[str, Any], *, active_path: Path,
    execute: Callable[..., tuple[dict, dict]] | None = None,
    cleanup: Callable[[Path], None] = cleanup_cell_containers,
) -> dict[str, Any]:
    from .model_recompute import ModelResourceLimitError, execute_model_candidate

    execute = execute or execute_model_candidate
    request = _call_request(call)
    with store.claim(request) as cell:
        old = store.load(cell, request)
        if old is not None:
            return {**old, "reused": True}
        attempt_id, workspace = store.begin(cell, cleanup)
        atomic_json(active_path, {"workspace": str(workspace)})
        invocation = {**call, "workspace": workspace}
        for name in ("code_path", "provider_path", "runner_path"):
            invocation[name] = Path(invocation[name])
        try:
            result, evidence = execute(**invocation)
            cleanup(workspace)
            receipt = store.commit(cell, request, attempt_id=attempt_id, status="completed",
                                   result=result, execution_evidence=evidence)
        except (ModelCellCancelled, KeyboardInterrupt):
            cleanup(workspace)
            raise
        except Exception as exc:
            cleanup(workspace)
            receipt = store.commit(
                cell, request, attempt_id=attempt_id,
                status="resource_blocked" if isinstance(exc, ModelResourceLimitError) else "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        return {**receipt, "workspace": str(workspace), "reused": False}


class ModelCellBatchExecutor:
    """All allocations cover the complete producer + model + cleanup lifecycle."""

    def __init__(self, *, store_root: Path, batch_identity: dict, control_root: Path):
        self.store = ModelCellStore(store_root, batch_identity)
        self.control_root = _mkdir(control_root)
        self.batch_path = self.store.root / "batch.json"

    def _cleanup_active(self, directory: Path) -> None:
        path = directory / "active.json"
        if path.exists():
            workspace = _safe(Path(str(read_json(path).get("workspace") or "")))
            if not workspace.is_relative_to(self.store.root):
                raise ValueError("model cell cleanup escaped its registered batch")
            cleanup_cell_containers(workspace)

    def _publish_progress(self, calls, active, results) -> None:
        cells = []
        warnings = set()
        for index, (_process, directory, _log) in active.items():
            manifest = calls[index]["manifest"]
            progress = None
            observation_status = "not_yet_available"
            try:
                active_path = directory / "active.json"
                if active_path.exists():
                    workspace = _safe(Path(str(read_json(active_path).get("workspace") or "")))
                    if not workspace.is_relative_to(self.store.root):
                        raise ValueError("model progress escaped its registered batch")
                    path = workspace / "execution-progress.json"
                    if path.exists():
                        progress = read_json(path)
                        cell_warnings = progress.get("warnings", [])
                        if (
                            progress.get("contract_version")
                            != "model-execution-progress-v1-observe-only"
                            or not isinstance(cell_warnings, list)
                            or any(value not in {"elapsed_warning", "progress_not_observed"}
                                   for value in cell_warnings)
                        ):
                            raise ValueError("model execution progress contract is invalid")
                        warnings.update(cell_warnings)
                        observation_status = "available"
            except Exception:
                progress = None
                observation_status = "unavailable"
                warnings.add("progress_observation_unavailable")
            cells.append({
                "candidate_id": manifest.get("candidate_id"),
                "model_engine": manifest.get("model_engine"),
                "profile_id": manifest.get("evaluation_profile_id"),
                "seed": manifest.get("seed"),
                "progress": progress,
                "observation_status": observation_status,
            })
        atomic_json(self.control_root.parent / "model-progress.json", {
            "contract_version": "model-batch-progress-v1-observe-only",
            "status": "running",
            "execution_phase": "model_compute",
            "phase_label": "模型实验运行中；耗时提示不自动终止计算",
            "updated_at": datetime.now(UTC).isoformat(),
            "completed_cells": len(results),
            "planned_cells": len(calls),
            "cell_count_scope": "current_group",
            "active_cells": cells,
            "warnings": sorted(warnings),
            "automatic_termination": False,
        })

    def run_many(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        limits = fixed_model_cell_grid_policy()["batch_limits"]
        allocations = [governed_cell_resource_allocation(call["manifest"]) for call in calls]
        if any(
            not 0 < row["cpu_count"] <= limits["cpu_count"]
            or not 0 < row["memory_gb"] <= limits["memory_gb"]
            for row in allocations
        ):
            raise ValueError("model cell allocation exceeds the frozen total batch budget")
        pending = list(range(len(calls)))
        active: dict[int, tuple[subprocess.Popen, Path, Any]] = {}
        results: dict[int, dict[str, Any]] = {}
        last_progress_at = float("-inf")
        previous_completed = -1
        try:
            with cancellation_signals():
                while pending or active:
                    used_cpu = sum(allocations[index]["cpu_count"] for index in active)
                    used_memory = sum(allocations[index]["memory_gb"] for index in active)
                    for index in list(pending):
                        allocation = allocations[index]
                        if (
                            len(active) >= limits["max_cells"]
                            or used_cpu + allocation["cpu_count"] > limits["cpu_count"]
                            or used_memory + allocation["memory_gb"] > limits["memory_gb"]
                            or (allocation["exclusive"] and active)
                            or any(allocations[value]["exclusive"] for value in active)
                        ):
                            continue
                        directory = _mkdir(self.control_root / uuid.uuid4().hex)
                        call = {key: str(value) if isinstance(value, Path) else value
                                for key, value in calls[index].items() if key != "workspace"}
                        atomic_json(directory / "call.json", call)
                        log = (directory / "process.log").open("xb")
                        process = subprocess.Popen(
                            [sys.executable, "-m", "quant_platform.model_cell_execution",
                             "--store-root", str(self.store.root.parent),
                             "--batch", str(self.batch_path), "--control", str(directory),
                             "--parent", str(os.getpid())],
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        )
                        active[index] = (process, directory, log)
                        atomic_json(directory / "process.json", {
                            "identity": process_identity(process.pid),
                            "control_root": str(directory),
                        })
                        pending.remove(index)
                        used_cpu += allocation["cpu_count"]
                        used_memory += allocation["memory_gb"]
                    for index, (process, directory, log) in list(active.items()):
                        if process.poll() is None:
                            continue
                        self._cleanup_active(directory)
                        log.close()
                        if process.returncode != 0:
                            raise RuntimeError(
                                f"model cell process exited {process.returncode}; "
                                f"committed cells are retained; inspect {directory / 'process.log'}"
                            )
                        result = read_json(directory / "response.json")
                        # Parent independently verifies the journal and all output bytes.
                        request = _call_request(calls[index])
                        with self.store.claim(request) as cell:
                            receipt = self.store.load(cell, request)
                        if receipt is None or receipt["receipt_sha256"] != result.get(
                            "receipt_sha256"
                        ):
                            raise ValueError("model cell subprocess returned an unbound receipt")
                        results[index] = {**receipt, "reused": result["reused"]}
                        del active[index]  # Release only after verified process/container exit.
                    if (
                        len(results) != previous_completed
                        or time.monotonic() - last_progress_at >= 30
                    ):
                        try:
                            self._publish_progress(calls, active, results)
                        except Exception:
                            with suppress(Exception):
                                _LOG.exception("Could not publish model batch progress; continuing")
                        previous_completed = len(results)
                        last_progress_at = time.monotonic()
                    if pending or active:
                        time.sleep(0.1)
        finally:
            # A cancelled batch must finish cleanup before its global reservation is released.
            for process, _, _ in active.values():
                if process.poll() is None:
                    process.terminate()
            cleanup_errors = []
            for process, directory, log in active.values():
                try:
                    try:
                        process.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=30)
                    self._cleanup_active(directory)
                except Exception as exc:
                    cleanup_errors.append(exc)
                finally:
                    log.close()
            if cleanup_errors:
                raise RuntimeError("model cell cancellation could not verify container cleanup")
        return [results[index] for index in range(len(calls))]


def main() -> None:
    from .model_cell_recovery import MODEL_OWNER_LEASE_ENV
    from .model_prepared_cache import _locked

    parser = argparse.ArgumentParser()
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--parent", required=True, type=int)
    args = parser.parse_args()
    control = _safe(Path(args.control))
    with cancellation_signals():
        _parent_death_signal(args.parent)
        atomic_json(control / "child-process.json", {"identity": process_identity(os.getpid())})
        batch = read_json(Path(args.batch))
        call = read_json(control / "call.json")
        if cell_runtime_identity(Path(call["runner_path"])) != batch["runtime"]:
            raise ValueError("model cell runtime changed after batch registration")
        store = ModelCellStore(Path(args.store_root), batch)
        lease_path = store.root.parent.parent / "model-owner.lease"
        if os.environ.get(MODEL_OWNER_LEASE_ENV) != str(lease_path):
            raise ValueError("model cell owner lease differs from its registered job")
        with _locked(lease_path, exclusive=False, create=False):
            _parent_death_signal(args.parent)  # Recheck after waiting for a recovery fence.
            result = execute_stored_cell(store, call, active_path=control / "active.json")
            atomic_json(control / "response.json", result)


if __name__ == "__main__":
    main()
