from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from quant_platform.worker_cli import _PeriodicProbeCache, status_server

pytestmark = pytest.mark.no_database


def _request(server, path: str) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{server.server_port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _wait_until(predicate, *, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


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


def test_health_rejects_runtime_that_is_present_but_not_ready() -> None:
    server = status_server(
        {
            "qlib": {"status": "ok"},
            "rdagent": {
                "status": "ok",
                "ready": False,
                "blockers": ["Docker is required"],
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
        assert body["status"] == "unavailable"
        assert body["runtime"]["blockers"] == ["Docker is required"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_health_reads_cached_startup_state_without_waiting_for_slow_probe() -> None:
    probe_started = threading.Event()
    release_probe = threading.Event()

    def slow_probe() -> dict[str, object]:
        probe_started.set()
        release_probe.wait(timeout=5)
        return {"status": "ok", "ready": True}

    cache = _PeriodicProbeCache(
        "rdagent-test",
        slow_probe,
        interval_seconds=60,
        stale_after_seconds=120,
    )
    cache.start()
    assert probe_started.wait(timeout=1)
    server = status_server(
        {"qlib": {"status": "ok"}, "rdagent": cache.snapshot},
        required_runtime="rdagent",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        started_at = time.monotonic()
        status, body = _request(server, "/health")
        assert time.monotonic() - started_at < 1
        assert status == 503
        assert body["runtime"]["probe_cache"]["state"] == "starting"

        release_probe.set()
        assert _wait_until(lambda: cache.snapshot().get("status") == "ok")
        status, body = _request(server, "/health")
        assert status == 200
        assert body["runtime"]["probe_cache"]["state"] == "fresh"
    finally:
        release_probe.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        cache.stop()
        assert not cache.running


def test_periodic_probe_failure_replaces_success_instead_of_staying_healthy() -> None:
    call_count = 0
    second_probe_started = threading.Event()
    release_failure = threading.Event()

    def changing_probe() -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"status": "ok", "ready": True}
        second_probe_started.set()
        release_failure.wait(timeout=5)
        raise RuntimeError("runtime disappeared")

    cache = _PeriodicProbeCache(
        "rdagent-test",
        changing_probe,
        interval_seconds=0.02,
        stale_after_seconds=1,
    )
    cache.start()
    try:
        assert _wait_until(lambda: cache.snapshot().get("status") == "ok")
        assert second_probe_started.wait(timeout=1)
        release_failure.set()
        assert _wait_until(lambda: cache.snapshot().get("status") == "unavailable")
        assert "RuntimeError" in str(cache.snapshot().get("error"))
    finally:
        release_failure.set()
        cache.stop()
        assert not cache.running


def test_hung_periodic_probe_makes_the_cached_success_stale() -> None:
    call_count = 0
    second_probe_started = threading.Event()
    release_probe = threading.Event()

    def hanging_probe() -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            second_probe_started.set()
            release_probe.wait(timeout=5)
        return {"status": "ok", "ready": True}

    cache = _PeriodicProbeCache(
        "rdagent-test",
        hanging_probe,
        interval_seconds=0.02,
        stale_after_seconds=0.08,
    )
    cache.start()
    try:
        assert _wait_until(lambda: cache.snapshot().get("status") == "ok")
        assert second_probe_started.wait(timeout=1)
        assert _wait_until(
            lambda: cache.snapshot().get("probe_cache", {}).get("state") == "stale"
        )
        snapshot = cache.snapshot()
        assert snapshot["status"] == "unavailable"
        assert snapshot["ready"] is False
    finally:
        release_probe.set()
        cache.stop()
        assert not cache.running


def test_status_server_rejects_an_unknown_required_runtime() -> None:
    with pytest.raises(ValueError, match="unknown required runtime"):
        status_server({"qlib": {"status": "ok"}}, required_runtime="rdagent", port=0)
