from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import typer

from quant_data.config import Settings

from .job_store import JobStore
from .rdagent_runtime import probe_rdagent, run_official_rdagent_health_check
from .rdagent_scenarios import RDAGENT_JOB_KINDS
from .runtime_secret_store import RuntimeSecretStore
from .services import probe_qlib
from .worker import LocalJobWorker

app = typer.Typer(no_args_is_help=False, help="QuantLab durable background worker")

_MODEL_EVALUATION_JOB_KINDS = frozenset({"model_evaluate", "quant_bundle_evaluate"})
_IMMUTABLE_IMAGE = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
_PROBE_INTERVAL_SECONDS = 300.0
_PROBE_STALE_AFTER_SECONDS = 660.0
_PROBE_STOP_TIMEOUT_SECONDS = 5.0


class _PeriodicProbeCache:
    """Run a potentially slow readiness probe away from HTTP request threads."""

    def __init__(
        self,
        name: str,
        probe: Callable[[], dict[str, object]],
        *,
        initial_result: dict[str, object] | None = None,
        interval_seconds: float = _PROBE_INTERVAL_SECONDS,
        stale_after_seconds: float = _PROBE_STALE_AFTER_SECONDS,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("probe interval must be positive")
        if stale_after_seconds <= interval_seconds:
            raise ValueError("probe stale threshold must exceed its interval")
        self._name = name
        self._probe = probe
        self._interval_seconds = interval_seconds
        self._stale_after_seconds = stale_after_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._result = copy.deepcopy(
            initial_result
            or {
                "status": "unavailable",
                "ready": False,
                "error": f"{name} startup probe has not completed",
            }
        )
        self._completed_at: str | None = None
        self._completed_monotonic: float | None = None

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            thread = threading.Thread(
                target=self._loop,
                name=f"{self._name}-readiness-probe",
                daemon=True,
            )
            self._thread = thread
        thread.start()

    def stop(self, timeout: float = _PROBE_STOP_TIMEOUT_SECONDS) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def snapshot(self) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            result = copy.deepcopy(self._result)
            completed_at = self._completed_at
            completed_monotonic = self._completed_monotonic

        if completed_monotonic is None:
            probe_state = "starting"
            age_seconds = None
        else:
            age_seconds = max(0.0, now - completed_monotonic)
            probe_state = "fresh"
            if age_seconds > self._stale_after_seconds:
                probe_state = "stale"
                result.update(
                    {
                        "status": "unavailable",
                        "ready": False,
                        "error": f"{self._name} readiness probe result is stale",
                    }
                )
        result["probe_cache"] = {
            "state": probe_state,
            "checked_at": completed_at,
            "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
            "stale_after_seconds": self._stale_after_seconds,
        }
        return result

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                result = self._probe()
                if not isinstance(result, dict):
                    raise TypeError(
                        f"{self._name} readiness probe returned {type(result).__name__}"
                    )
                result = copy.deepcopy(result)
            except Exception as exc:  # noqa: BLE001 - readiness boundary is fail closed
                result = {
                    "status": "unavailable",
                    "ready": False,
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            with self._lock:
                self._result = result
                self._completed_at = datetime.now(UTC).isoformat()
                self._completed_monotonic = time.monotonic()
            if self._stop.wait(self._interval_seconds):
                return


def _worker_capabilities(settings: Settings) -> dict[str, object]:
    job_kinds = sorted(set(settings.worker_job_kinds))
    model_sandbox_required = bool(
        _MODEL_EVALUATION_JOB_KINDS.intersection(job_kinds)
    )
    result: dict[str, object] = {
        "job_kinds": job_kinds,
        "model_sandbox_required": model_sandbox_required,
        "model_sandbox_ready": not model_sandbox_required,
    }
    if not model_sandbox_required:
        return result
    image = settings.model_sandbox_image
    if not _IMMUTABLE_IMAGE.fullmatch(image):
        result["model_sandbox_error"] = "immutable image digest is not configured"
        return result
    docker = shutil.which("docker")
    if not docker:
        result["model_sandbox_error"] = "Docker CLI is unavailable"
        return result
    try:
        completed = subprocess.run(
            [docker, "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        result["model_sandbox_error"] = "immutable model sandbox image is not preloaded"
        return result
    image_id = completed.stdout.strip().lower()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        result["model_sandbox_error"] = "model sandbox image identity is invalid"
        return result
    result.update(
        {
            "model_sandbox_ready": True,
            "model_sandbox_image_id": image_id,
            "model_sandbox_config_sha256": hashlib.sha256(image.encode()).hexdigest(),
        }
    )
    return result


def _initial_worker_capabilities(settings: Settings) -> dict[str, object]:
    job_kinds = sorted(set(settings.worker_job_kinds))
    return {
        "status": "unavailable",
        "job_kinds": job_kinds,
        "model_sandbox_required": bool(
            _MODEL_EVALUATION_JOB_KINDS.intersection(job_kinds)
        ),
        "model_sandbox_ready": False,
        "capability_error": "worker capability startup probe has not completed",
    }


def _runtime_status(runtimes: dict[str, object], name: str) -> dict[str, object]:
    try:
        runtime = runtimes[name]
        body = runtime() if callable(runtime) else runtime
        if not isinstance(body, dict):
            raise TypeError(f"{name} runtime probe returned {type(body).__name__}")
        return body
    except Exception as exc:  # noqa: BLE001 - fail closed at the process health boundary
        return {
            "status": "unavailable",
            "ready": False,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }


def _capability_status(capabilities: object | None) -> dict[str, object]:
    if capabilities is None:
        return {}
    try:
        body = capabilities() if callable(capabilities) else capabilities
        if not isinstance(body, dict):
            raise TypeError(
                f"worker capability probe returned {type(body).__name__}"
            )
        return body
    except Exception as exc:  # noqa: BLE001 - fail closed at the health boundary
        return {
            "status": "unavailable",
            "model_sandbox_ready": False,
            "capability_error": f"{type(exc).__name__}: {exc}"[:500],
        }


def status_server(
    runtimes: dict[str, object],
    *,
    required_runtime: str,
    capabilities: object | None = None,
    diagnostic: object | None = None,
    port: int = 8770,
) -> ThreadingHTTPServer:
    if required_runtime not in runtimes:
        raise ValueError(f"unknown required runtime: {required_runtime}")

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, body: dict[str, object]) -> None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                runtime = _runtime_status(runtimes, required_runtime)
                worker_capabilities = _capability_status(capabilities)
                runtime_ready = runtime.get("status") == "ok" and runtime.get(
                    "ready", True
                ) is not False
                capabilities_ready = worker_capabilities.get("status") not in {
                    "failed",
                    "unavailable",
                } and not (
                    worker_capabilities.get("model_sandbox_required")
                    and not worker_capabilities.get("model_sandbox_ready")
                )
                healthy = runtime_ready and capabilities_ready
                body = {
                    "status": "ok" if healthy else "unavailable",
                    "worker": "ready" if healthy else "runtime_unavailable",
                    "required_runtime": required_runtime,
                    "runtime": runtime,
                    "capabilities": worker_capabilities,
                }
                status_code = 200 if healthy else 503
            elif self.path == "/qlib/status":
                body = _runtime_status(runtimes, "qlib")
                status_code = 200
            elif self.path == "/rdagent/status":
                body = _runtime_status(runtimes, "rdagent")
                status_code = 200
            else:
                self.send_error(404)
                return
            self._send_json(status_code, body)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/rdagent/health-check" or not callable(diagnostic):
                self.send_error(404)
                return
            try:
                body = diagnostic()
                if not isinstance(body, dict):
                    raise TypeError("diagnostic returned invalid data")
            except Exception as exc:  # noqa: BLE001 - diagnostic boundary is fail closed
                body = {
                    "status": "failed",
                    "diagnostic_only": True,
                    "platform_readiness_unchanged": True,
                    "output": f"{type(exc).__name__}: diagnostic unavailable",
                }
            self._send_json(200, body)

        def log_message(self, _format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


@app.callback(invoke_without_command=True)
def run() -> None:
    """Run the external PostgreSQL-backed worker until it receives a stop signal."""
    root = Path.cwd().resolve()
    settings = Settings.from_env(root / ".env")
    store = JobStore(settings.database_url)
    worker = LocalJobWorker(store, root, settings)
    runtime_secrets = RuntimeSecretStore(settings.database_url, settings.platform_secret_key)
    stopped = threading.Event()

    def rdagent_status() -> dict:
        llm = runtime_secrets.get("llm")
        runtime_env = None
        if llm:
            runtime_env = {
                settings.rdagent_llm_key_env: llm["api_key"],
                "OPENAI_API_BASE": llm.get("api_base", ""),
                "CHAT_MODEL": llm.get("chat_model", "gpt-4.1-mini"),
            }
        return probe_rdagent(
            settings, root, runtime_env=runtime_env, force_local=True
        )

    def rdagent_diagnostic() -> dict:
        llm = runtime_secrets.get("llm")
        runtime_env = None
        if llm:
            runtime_env = {
                settings.rdagent_llm_key_env: llm["api_key"],
                "OPENAI_API_BASE": llm.get("api_base", ""),
                "CHAT_MODEL": llm.get("chat_model", "gpt-4.1-mini"),
            }
        return run_official_rdagent_health_check(
            settings, root, runtime_env=runtime_env
        )

    runtime_probes = {
        "qlib": _PeriodicProbeCache(
            "qlib",
            lambda: probe_qlib(settings, root),
        ),
        "rdagent": _PeriodicProbeCache("rdagent", rdagent_status),
    }
    capability_probe = _PeriodicProbeCache(
        "worker-capabilities",
        lambda: _worker_capabilities(settings),
        initial_result=_initial_worker_capabilities(settings),
    )
    probes = [*runtime_probes.values(), capability_probe]
    runtimes: dict[str, object] = {
        name: probe.snapshot for name, probe in runtime_probes.items()
    }
    required_runtime = (
        "rdagent"
        if settings.worker_job_kinds
        and set(settings.worker_job_kinds).issubset(RDAGENT_JOB_KINDS)
        else "qlib"
    )
    server = status_server(
        runtimes,
        required_runtime=required_runtime,
        capabilities=capability_probe.snapshot,
        diagnostic=rdagent_diagnostic,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)

    def request_stop(_signum: int, _frame: object) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    worker_started = False
    server_started = False
    try:
        for probe in probes:
            probe.start()
        worker.start()
        worker_started = True
        server_thread.start()
        server_started = True
        stopped.wait()
    finally:
        if server_started:
            server.shutdown()
        server.server_close()
        if server_started:
            server_thread.join(timeout=_PROBE_STOP_TIMEOUT_SECONDS)
        if worker_started:
            worker.stop()
        for probe in reversed(probes):
            probe.stop()


if __name__ == "__main__":
    app()
