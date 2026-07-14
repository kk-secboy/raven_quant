from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from quant_platform.worker_cli import status_server

pytestmark = pytest.mark.no_database


def _request(server, path: str) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{server.server_port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_health_checks_the_worker_required_runtime() -> None:
    server = status_server(
        {
            "qlib": {"status": "ok", "qlib_version": "test"},
            "rdagent": {"status": "disabled"},
        },
        required_runtime="qlib",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _request(server, "/health")
        assert status == 200
        assert body["status"] == "ok"
        assert body["required_runtime"] == "qlib"
        assert body["runtime"]["qlib_version"] == "test"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_health_fails_closed_when_rdagent_probe_is_unavailable() -> None:
    server = status_server(
        {
            "qlib": {"status": "ok"},
            "rdagent": lambda: {
                "status": "unavailable",
                "ready": False,
                "error": "RD-Agent import failed",
            },
        },
        required_runtime="rdagent",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _request(server, "/health")
        assert status == 503
        assert body["worker"] == "runtime_unavailable"
        assert body["runtime"]["error"] == "RD-Agent import failed"

        status, details = _request(server, "/rdagent/status")
        assert status == 200
        assert details["status"] == "unavailable"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_health_converts_probe_exceptions_to_unavailable() -> None:
    def broken_probe() -> dict:
        raise ImportError("missing runtime module")

    server = status_server(
        {"qlib": {"status": "ok"}, "rdagent": broken_probe},
        required_runtime="rdagent",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _request(server, "/health")
        assert status == 503
        assert body["runtime"]["ready"] is False
        assert "ImportError" in body["runtime"]["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_status_server_rejects_an_unknown_required_runtime() -> None:
    with pytest.raises(ValueError, match="unknown required runtime"):
        status_server({"qlib": {"status": "ok"}}, required_runtime="rdagent", port=0)
