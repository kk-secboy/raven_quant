from __future__ import annotations

import json
import signal
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import typer

from quant_data.config import Settings

from .scheduler import SchedulerEngine

app = typer.Typer(no_args_is_help=False, help="QuantLab durable schedule and alert service")


def status_server(state: dict, port: int = 8780) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_error(404)
                return
            body = {"status": "ok" if state.get("last_error") is None else "degraded", **state}
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


@app.callback(invoke_without_command=True)
def run() -> None:
    """Run the leased PostgreSQL scheduler until it receives a stop signal."""
    root = Path.cwd().resolve()
    settings = Settings.from_env(root / ".env")
    engine = SchedulerEngine(settings)
    stopped = threading.Event()
    state: dict = {"last_tick": None, "last_error": None, "stats": {}}
    server = status_server(state)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)

    def request_stop(_signum: int, _frame: object) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    server_thread.start()
    try:
        while not stopped.is_set():
            try:
                state["stats"] = engine.tick()
                state["last_error"] = None
            except Exception as exc:
                state["last_error"] = str(exc)
            state["last_tick"] = datetime.now(UTC).isoformat(timespec="seconds")
            stopped.wait(settings.scheduler_poll_seconds)
    finally:
        server.shutdown()


if __name__ == "__main__":
    app()
