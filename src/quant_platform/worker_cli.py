from __future__ import annotations

import json
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import typer

from quant_data.config import Settings

from .job_store import JobStore
from .rdagent_runtime import probe_rdagent
from .runtime_secret_store import RuntimeSecretStore
from .services import probe_qlib
from .worker import LocalJobWorker

app = typer.Typer(no_args_is_help=False, help="QuantLab durable background worker")


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


def status_server(
    runtimes: dict[str, object],
    *,
    required_runtime: str,
    port: int = 8770,
) -> ThreadingHTTPServer:
    if required_runtime not in runtimes:
        raise ValueError(f"unknown required runtime: {required_runtime}")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                runtime = _runtime_status(runtimes, required_runtime)
                healthy = runtime.get("status") == "ok"
                body = {
                    "status": "ok" if healthy else "unavailable",
                    "worker": "ready" if healthy else "runtime_unavailable",
                    "required_runtime": required_runtime,
                    "runtime": runtime,
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
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

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
        return probe_rdagent(settings, root, runtime_env=runtime_env)

    runtimes: dict[str, object] = {
        "qlib": probe_qlib(settings, root),
        "rdagent": rdagent_status,
    }
    required_runtime = "rdagent" if settings.worker_job_kinds == ("rdagent_factor",) else "qlib"
    server = status_server(runtimes, required_runtime=required_runtime)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)

    def request_stop(_signum: int, _frame: object) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    worker.start()
    server_thread.start()
    try:
        stopped.wait()
    finally:
        server.shutdown()
        worker.stop()


if __name__ == "__main__":
    app()
