"""Prepare shared data in a trusted container, then lease it to one model sandbox."""
from __future__ import annotations

import errno
import json
import math
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from .model_compute_policy import governed_cell_resource_allocation, model_thread_environment
from .model_data_request import file_digest, prepared_data_request
from .model_prepared_cache import prepared_data_cache
from .model_prepared_data import canonical_key, manifest_sha256

PREPARED_EXECUTION_VERSION = "model-prepared-execution-v1"
PREPARED_SUMMARY_VERSION = "model-prepared-data-summary-v1"
_PRODUCER_MODULES = (
    "model_data_handler.py", "model_data_request.py", "model_prepared_data.py",
    "upstream_versions.py",
)


def prepared_runtime_sources(runner_path: Path) -> dict[str, Path]:
    package = Path(__file__).resolve().parent
    sources = {name: package / name for name in _PRODUCER_MODULES}
    sources["prepare_model_data.py"] = runner_path.with_name("prepare_model_data.py")
    if not all(path.is_file() for path in sources.values()):
        raise ValueError("trusted prepared data runtime is unavailable")
    return sources


def prepared_runtime_identity(runner_path: Path, *, image: str, image_id: str) -> dict[str, str]:
    return {
        "version": PREPARED_EXECUTION_VERSION, "sandbox_image": image,
        "sandbox_image_id": image_id,
        "compute_policy_sha256": file_digest(
            Path(__file__).resolve().with_name("model_compute_policy.py")
        ),
        **{name: file_digest(path) for name, path in prepared_runtime_sources(runner_path).items()},
    }


def _stop_container(cidfile: Path) -> None:
    if cidfile.is_file():
        cid = cidfile.read_text(encoding="ascii").strip()
        if re.fullmatch(r"[0-9a-f]{64}", cid):
            with suppress(OSError, subprocess.SubprocessError):
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True,
                               timeout=30, check=False)


def _validate_producer_summary(
    stdout: str, *, request: dict, destination: Path, capacity: bool, memory_limit_bytes: int,
) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("producer summary contains duplicate fields")
            result[key] = value
        return result

    def nonfinite(_value):
        raise ValueError("producer summary contains non-finite numbers")

    lines = [line for line in stdout.splitlines() if line.strip()]
    summaries = []
    final = None
    for position, line in enumerate(lines):
        try:
            value = json.loads(line, object_pairs_hook=unique_pairs, parse_constant=nonfinite)
        except ValueError as exc:
            if position == len(lines) - 1:
                raise ValueError("producer did not end with a valid structured summary") from exc
            continue
        if isinstance(value, dict) and value.get("contract_version") == PREPARED_SUMMARY_VERSION:
            summaries.append(value)
        if position == len(lines) - 1:
            final = value
    fields = {
        "contract_version", "status", "request_sha256", "manifest_sha256",
        "prepare_seconds", "write_seconds", "elapsed_seconds", "memory",
    }
    if len(summaries) != 1 or final is not summaries[0] or set(final) != fields:
        raise ValueError("producer must return exactly one complete final summary")
    summary = summaries[0]
    expected_status = "storage_capacity_unavailable" if capacity else "prepared"
    if summary["status"] != expected_status or summary["request_sha256"] != canonical_key(request):
        raise ValueError("producer summary status or request identity is invalid")
    if capacity:
        if summary["manifest_sha256"] is not None:
            raise ValueError("capacity failure cannot claim a completed prepared manifest")
    elif summary["manifest_sha256"] != manifest_sha256(destination):
        raise ValueError("producer summary prepared manifest checksum mismatch")
    for field in ("prepare_seconds", "write_seconds", "elapsed_seconds"):
        value = summary[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("producer summary durations must be finite nonnegative numbers")
    if summary["elapsed_seconds"] + 1e-6 < summary["prepare_seconds"] + summary["write_seconds"]:
        raise ValueError("producer summary durations are inconsistent")
    memory = summary["memory"]
    memory_fields = {
        "process_rss_bytes", "process_peak_rss_bytes", "cgroup_version", "cgroup_current_bytes",
        "cgroup_peak_bytes", "cgroup_limit_bytes", "observed_memory_peak_bytes",
    }
    if not isinstance(memory, dict) or set(memory) != memory_fields:
        raise ValueError("producer summary memory counters are incomplete")
    if any(type(value) is not int or value <= 0 for value in memory.values()):
        raise ValueError("producer summary must contain measured positive Linux memory counters")
    if (
        memory["cgroup_version"] not in (1, 2)
        or memory["cgroup_limit_bytes"] != memory_limit_bytes
        or memory["process_peak_rss_bytes"] < memory["process_rss_bytes"]
        or memory["cgroup_peak_bytes"] < memory["cgroup_current_bytes"]
        or memory["observed_memory_peak_bytes"] != max(
            memory["process_peak_rss_bytes"], memory["cgroup_peak_bytes"],
        )
    ):
        raise ValueError("producer summary memory counters or governed limit are inconsistent")
    return summary


def _producer_progress_audit(path: Path, *, exit_code: int | None) -> dict:
    result = {
        "progress_path": "data-preparation/audit/progress.jsonl",
        "progress_sha256": None, "progress_status": None, "progress_partial": True,
    }
    if not path.exists():
        return result
    if path.is_symlink() or not path.is_file():
        raise ValueError("producer progress must be a regular audit file")
    result["progress_sha256"] = file_digest(path)
    last = ""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = line
    try:
        final = json.loads(last)
    except ValueError:
        return result
    if (
        not isinstance(final, dict)
        or final.get("contract_version") != "model-prepared-data-progress-v1"
    ):
        return result
    status = final.get("status")
    expected_stage = {
        "running": None, "completed": "producer_completed", "failed": "producer_failed",
        "capacity": "producer_capacity",
    }
    if not isinstance(status, str) or status not in expected_stage:
        return result
    result["progress_status"] = status
    expected_status = "completed" if exit_code == 0 else "capacity" if exit_code == 75 else "failed"
    result["progress_partial"] = (
        exit_code is None or status != expected_status
        or final.get("stage") != expected_stage[status]
    )
    return result


@contextmanager
def _cache_with_deadline(*, command: list[str], timeout_seconds: int, **kwargs):
    try:
        with prepared_data_cache(**kwargs) as prepared:
            yield prepared
    except TimeoutError as exc:
        raise subprocess.TimeoutExpired(command, timeout_seconds) from exc


def run_with_prepared_data(
    *, command: list[str], workspace: Path, provider: Path, runner_path: Path,
    manifest: dict[str, Any], timeout_seconds: int,
) -> subprocess.CompletedProcess:
    """One deadline covers cache preparation and fitting; scientific budgets stay fixed.

    Only this controller can write the cache. The producer sees no candidate code;
    the model sees only its entry and lease, both read-only. Entry hashes are cell
    evidence, never part of the shared execution-environment identity.
    """
    started = time.monotonic()
    environment = manifest["execution_environment"]
    identity = environment["prepared_data_producer"]
    image = environment["sandbox_image"]
    allocation = governed_cell_resource_allocation(manifest)
    additional = (
        workspace / "additional_factors.parquet"
        if manifest.get("additional_factors_path") else None
    )
    request = prepared_data_request(
        manifest, provider=provider, label_contract=manifest["model_label_contract"],
        producer_identity=identity, additional_factors=additional,
    )
    request_path = workspace / "prepared-request.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    request_path.chmod(0o444)
    sources = prepared_runtime_sources(runner_path)
    cache_root = Path(os.environ.get("MODEL_PREPARED_DATA_ROOT") or (
        Path(os.environ.get("DATA_ROOT") or "data") / "artifacts" / "model-prepared-data"
    )).absolute()
    producer_dir = workspace / "data-preparation"
    binding: dict[str, Any] = {
        "contract_version": PREPARED_EXECUTION_VERSION,
        "request_sha256": canonical_key(request), "mode": "preparing", "producer": None,
    }

    def persist_binding() -> None:
        manifest["prepared_data"] = binding
        path = workspace / "manifest.json"
        if path.exists():
            path.chmod(0o644)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        path.chmod(0o444)

    persist_binding()

    def remaining() -> float:
        seconds = timeout_seconds - (time.monotonic() - started)
        if seconds <= 0:
            raise subprocess.TimeoutExpired(command, timeout_seconds)
        return seconds

    def build(destination: Path) -> None:
        producer_dir.mkdir()
        audit_dir = producer_dir / "audit"
        audit_dir.mkdir(mode=0o777)
        audit_dir.chmod(0o777)
        inputs = producer_dir / "inputs"
        package = inputs / "quant_platform"
        package.mkdir(parents=True)
        (package / "__init__.py").touch()
        for name, source in sources.items():
            target = inputs / name if name == "prepare_model_data.py" else package / name
            shutil.copy2(source, target)
            target.chmod(0o444)
        shutil.copy2(request_path, inputs / "request.json")
        if additional is not None:
            shutil.copy2(additional, inputs / "additional_factors.parquet")
        destination.parent.chmod(0o777)
        cidfile = producer_dir / "container.cid"
        # Build from the same governed limits, but mount only trusted inputs.
        producer_command = [
            "docker", "run", "--rm", "--cidfile", str(cidfile),
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "512",
            "--memory", f"{allocation['memory_gb']}g",
            "--memory-swap", f"{allocation['memory_gb']}g",
            "--cpus", str(allocation["cpu_count"]),
            "--user", "65534:65534", "--env", "HOME=/tmp",
            "--env", "MLFLOW_ALLOW_FILE_STORE=true",
            *[
                argument
                for name, value in model_thread_environment(allocation, preparation=True).items()
                for argument in ("--env", f"{name}={value}")
            ],
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g",
            "--mount", f"type=bind,src={inputs.resolve()},dst=/work,readonly",
            "--mount", f"type=bind,src={provider.resolve()},dst=/qlib,readonly",
            "--mount", f"type=bind,src={audit_dir.resolve()},dst=/audit",
            "--mount", f"type=bind,src={destination.parent},dst=/output",
            "--mount", (
                f"type=bind,src={destination.parent / 'producer.lease'},"
                "dst=/prepared-producer.lease,readonly"
            ),
            "--workdir", "/work", image, "python", "-I", "prepare_model_data.py",
            "--output", f"/output/{destination.name}",
        ]

        def preserve_logs(stdout, stderr, *, exit_code, error_type=None):
            stdout_path, stderr_path = producer_dir / "stdout.log", producer_dir / "stderr.log"
            for path, value in ((stdout_path, stdout), (stderr_path, stderr)):
                text = (
                    value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
                )
                path.write_text(text or "", encoding="utf-8")
            audit = {
                "summary": None, "summary_validated": False, "exit_code": exit_code,
                "stdout_sha256": file_digest(stdout_path),
                "stderr_sha256": file_digest(stderr_path),
                **_producer_progress_audit(audit_dir / "progress.jsonl", exit_code=exit_code),
            }
            if error_type is not None:
                audit["error_type"] = error_type
                audit["log_capture_complete"] = False
            binding["producer"] = audit
            persist_binding()
            return audit

        try:
            completed = subprocess.run(producer_command, capture_output=True, text=True,
                                       timeout=remaining(), check=False)
        except BaseException as exc:
            _stop_container(cidfile)
            preserve_logs(
                getattr(exc, "stdout", None), getattr(exc, "stderr", None),
                exit_code=None, error_type=type(exc).__name__,
            )
            raise
        if completed.returncode != 0:
            _stop_container(cidfile)
        audit = preserve_logs(completed.stdout, completed.stderr, exit_code=completed.returncode)
        if completed.returncode in (0, 75):
            try:
                audit["summary"] = _validate_producer_summary(
                    completed.stdout, request=request, destination=destination,
                    capacity=completed.returncode == 75,
                    memory_limit_bytes=int(allocation["memory_gb"]) * 1024**3,
                )
                audit["summary_validated"] = True
                if completed.returncode == 0:
                    pin = workspace / "prepared-manifest.json"
                    shutil.copy2(destination / "manifest.json", pin)
                    pin.chmod(0o444)
            except ValueError:
                _stop_container(cidfile)
                raise
            finally:
                persist_binding()
        if completed.returncode != 0:
            if completed.returncode == 75:
                raise OSError(errno.ENOSPC, "trusted prepared data storage capacity unavailable")
            message = (completed.stderr or completed.stdout)[-4000:]
            raise PreparedDataBuildError(completed.returncode, message)

    with _cache_with_deadline(
        command=command, timeout_seconds=timeout_seconds,
        cache_root=cache_root, contract=request, build=build, deadline=started + timeout_seconds,
    ) as prepared:
        run_command = list(command)
        binding["mode"] = "uncached_capacity" if prepared is None else "prepared_readonly"
        if prepared is not None:
            binding.update({
                "manifest_sha256": prepared["manifest_sha256"],
                "cache_hit": prepared["cache_hit"],
                "cache_acquire_seconds": prepared["elapsed_seconds"],
            })
            pin = workspace / "prepared-manifest.json"
            if pin.exists():
                if file_digest(pin) != prepared["manifest_sha256"]:
                    raise ValueError("producer and published prepared manifest identities differ")
            else:
                shutil.copy2(prepared["entry"] / "manifest.json", pin)
                pin.chmod(0o444)
            image_index = run_command.index(image)
            run_command[image_index:image_index] = [
                "--mount", f"type=bind,src={prepared['entry']},dst=/prepared,readonly",
                "--mount", f"type=bind,src={prepared['lease_path']},dst=/prepared.lease,readonly",
            ]
        persist_binding()
        try:
            completed = subprocess.run(run_command, capture_output=True, text=True,
                                       timeout=remaining(), check=False)
        except BaseException:
            _stop_container(workspace / "container.cid")
            raise
        if completed.returncode != 0:
            _stop_container(workspace / "container.cid")
        return completed


class PreparedDataBuildError(ValueError):
    def __init__(self, returncode: int, message: str):
        self.returncode = returncode
        super().__init__(f"trusted model data preparation failed (exit={returncode}): {message}")
