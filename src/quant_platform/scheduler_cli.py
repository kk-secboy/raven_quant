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


def scheduler_health(
    state: dict,
    *,
    now: datetime | None = None,
    stale_after_seconds: int = 60,
    max_active_tick_seconds: int = 300,
    state_lock: threading.Lock | None = None,
) -> tuple[int, dict]:
    """Return an HTTP status and body for the scheduler's actual tick health."""

    current = now or datetime.now(UTC)
    if state_lock is None:
        snapshot = dict(state)
    else:
        with state_lock:
            snapshot = dict(state)
    last_error = snapshot.get("last_error")
    last_tick_raw = snapshot.get("last_tick")
    tick_in_progress = snapshot.get("tick_in_progress") is True
    tick_started_at_raw = snapshot.get("tick_started_at")
    active_limit = max(60, int(max_active_tick_seconds))
    status = "ok"
    message = "scheduler tick is current"
    age_seconds: float | None = None
    freshness_source = "last_tick"

    if tick_in_progress != (tick_started_at_raw is not None):
        status = "degraded"
        message = "scheduler tick state is inconsistent"
    elif last_error is not None:
        status = "degraded"
        message = "scheduler tick failed"
    elif tick_in_progress and last_tick_raw:
        try:
            tick_started_at = datetime.fromisoformat(str(tick_started_at_raw))
            if tick_started_at.tzinfo is None:
                raise ValueError("tick_started_at must be timezone-aware")
            age_seconds = max(0.0, (current - tick_started_at).total_seconds())
        except (TypeError, ValueError):
            status = "degraded"
            message = "scheduler active tick timestamp is invalid"
        else:
            freshness_source = "active_tick"
            if age_seconds > active_limit:
                status = "degraded"
                message = "scheduler active tick exceeded its maximum duration"
            else:
                message = "scheduler tick is in progress"
    elif not last_tick_raw:
        status = "starting"
        message = "scheduler has not completed its first tick"
    else:
        try:
            last_tick = datetime.fromisoformat(str(last_tick_raw))
            if last_tick.tzinfo is None:
                raise ValueError("last_tick must be timezone-aware")
            age_seconds = max(0.0, (current - last_tick).total_seconds())
        except (TypeError, ValueError):
            status = "degraded"
            message = "scheduler last_tick is invalid"
        else:
            if age_seconds > max(1, stale_after_seconds):
                status = "degraded"
                message = "scheduler tick is stale"

    body = {
        **snapshot,
        "status": status,
        "ready": status == "ok",
        "message": message,
        "age_seconds": age_seconds,
        "stale_after_seconds": max(1, stale_after_seconds),
        "max_active_tick_seconds": active_limit,
        "freshness_source": freshness_source,
    }
    return (200 if status == "ok" else 503), body


def status_server(
    state: dict,
    port: int = 8780,
    *,
    stale_after_seconds: int = 60,
    max_active_tick_seconds: int = 300,
    state_lock: threading.Lock | None = None,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_error(404)
                return
            status_code, body = scheduler_health(
                state,
                stale_after_seconds=stale_after_seconds,
                max_active_tick_seconds=max_active_tick_seconds,
                state_lock=state_lock,
            )
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
    """Run the leased PostgreSQL scheduler until it receives a stop signal."""
    root = Path.cwd().resolve()
    settings = Settings.from_env(root / ".env")
    engine = SchedulerEngine(settings)
    stopped = threading.Event()
    state_lock = threading.Lock()
    state: dict = {
        "last_tick": None,
        "tick_in_progress": False,
        "tick_started_at": None,
        "last_error": None,
        "stats": {},
        "poll_seconds": settings.scheduler_poll_seconds,
        "release_id": settings.quantlab_release_id,
        "config_digest": settings.quantlab_config_digest,
    }
    server = status_server(
        state,
        stale_after_seconds=max(30, settings.scheduler_poll_seconds * 2),
        max_active_tick_seconds=settings.scheduler_max_tick_seconds,
        state_lock=state_lock,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)

    def request_stop(_signum: int, _frame: object) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    server_thread.start()
    try:
        while not stopped.is_set():
            with state_lock:
                state["tick_in_progress"] = True
                state["tick_started_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            stats: dict = {}
            error: str | None = None
            try:
                stats = engine.tick()
            except Exception as exc:
                error = str(exc)
            finally:
                with state_lock:
                    state["stats"] = stats
                    state["last_error"] = error
                    state["last_tick"] = datetime.now(UTC).isoformat(timespec="seconds")
                    state["tick_in_progress"] = False
                    state["tick_started_at"] = None
            stopped.wait(settings.scheduler_poll_seconds)
    finally:
        server.shutdown()


if __name__ == "__main__":
    app()
