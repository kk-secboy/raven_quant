from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest
import requests

from quant_platform.scheduler_cli import scheduler_health, status_server


@pytest.mark.no_database
def test_scheduler_health_requires_a_fresh_successful_tick() -> None:
    now = datetime(2026, 8, 29, 2, 0, tzinfo=UTC)

    starting_code, starting = scheduler_health(
        {"last_tick": None, "last_error": None, "stats": {}},
        now=now,
        stale_after_seconds=30,
    )
    assert starting_code == 503
    assert starting["status"] == "starting"
    assert starting["ready"] is False

    stale_code, stale = scheduler_health(
        {
            "last_tick": (now - timedelta(seconds=31)).isoformat(),
            "last_error": None,
            "stats": {},
        },
        now=now,
        stale_after_seconds=30,
    )
    assert stale_code == 503
    assert stale["status"] == "degraded"
    assert stale["message"] == "scheduler tick is stale"

    healthy_code, healthy = scheduler_health(
        {
            "last_tick": (now - timedelta(seconds=5)).isoformat(),
            "last_error": None,
            "stats": {"processed": 1},
        },
        now=now,
        stale_after_seconds=30,
    )
    assert healthy_code == 200
    assert healthy["status"] == "ok"
    assert healthy["ready"] is True


@pytest.mark.no_database
def test_scheduler_status_server_returns_503_after_tick_failure() -> None:
    state = {
        "last_tick": datetime.now(UTC).isoformat(timespec="seconds"),
        "last_error": None,
        "stats": {},
    }
    server = status_server(state, port=0, stale_after_seconds=60)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/health"
    try:
        healthy = requests.get(url, timeout=2)
        assert healthy.status_code == 200
        assert healthy.json()["ready"] is True

        state["last_error"] = "tick failed"
        degraded = requests.get(url, timeout=2)
        assert degraded.status_code == 503
        assert degraded.json()["status"] == "degraded"
        assert degraded.json()["ready"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.no_database
def test_scheduler_health_keeps_a_long_active_tick_ready() -> None:
    now = datetime(2026, 8, 29, 2, 0, tzinfo=UTC)
    status_code, body = scheduler_health(
        {
            "last_tick": (now - timedelta(seconds=90)).isoformat(),
            "tick_in_progress": True,
            "tick_started_at": (now - timedelta(seconds=75)).isoformat(),
            "last_error": None,
            "stats": {},
        },
        now=now,
        stale_after_seconds=30,
        max_active_tick_seconds=300,
    )

    assert status_code == 200
    assert body["ready"] is True
    assert body["message"] == "scheduler tick is in progress"
    assert body["freshness_source"] == "active_tick"
    assert body["age_seconds"] == 75


@pytest.mark.no_database
def test_scheduler_health_fails_closed_when_active_tick_exceeds_bound() -> None:
    now = datetime(2026, 8, 29, 2, 0, tzinfo=UTC)
    status_code, body = scheduler_health(
        {
            "last_tick": (now - timedelta(seconds=400)).isoformat(),
            "tick_in_progress": True,
            "tick_started_at": (now - timedelta(seconds=301)).isoformat(),
            "last_error": None,
            "stats": {},
        },
        now=now,
        stale_after_seconds=30,
        max_active_tick_seconds=300,
    )

    assert status_code == 503
    assert body["ready"] is False
    assert body["message"] == "scheduler active tick exceeded its maximum duration"
    assert body["freshness_source"] == "active_tick"
    assert body["age_seconds"] == 301
    assert body["max_active_tick_seconds"] == 300


@pytest.mark.no_database
def test_scheduler_health_reads_tick_fields_under_one_lock() -> None:
    now = datetime(2026, 8, 29, 2, 0, tzinfo=UTC)
    state = {
        "last_tick": (now - timedelta(seconds=5)).isoformat(),
        "tick_in_progress": False,
        "tick_started_at": None,
        "last_error": None,
        "stats": {},
    }
    lock = threading.Lock()
    writer_started = threading.Event()
    writer_finished = threading.Event()

    def writer() -> None:
        with lock:
            state["tick_in_progress"] = True
            writer_started.set()
            assert writer_finished.wait(timeout=2)
            state["tick_started_at"] = (now - timedelta(seconds=10)).isoformat()

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    assert writer_started.wait(timeout=2)
    result: list[tuple[int, dict]] = []
    reader_thread = threading.Thread(
        target=lambda: result.append(
            scheduler_health(
                state,
                now=now,
                stale_after_seconds=30,
                max_active_tick_seconds=300,
                state_lock=lock,
            )
        )
    )
    reader_thread.start()
    writer_finished.set()
    writer_thread.join(timeout=2)
    reader_thread.join(timeout=2)

    assert result[0][0] == 200
    assert result[0][1]["tick_in_progress"] is True
    assert result[0][1]["tick_started_at"] is not None
    assert result[0][1]["freshness_source"] == "active_tick"


@pytest.mark.no_database
def test_scheduler_health_does_not_claim_readiness_before_first_tick() -> None:
    now = datetime(2026, 8, 29, 2, 0, tzinfo=UTC)
    status_code, body = scheduler_health(
        {
            "last_tick": None,
            "tick_in_progress": True,
            "tick_started_at": (now - timedelta(seconds=45)).isoformat(),
            "last_error": None,
            "stats": {},
        },
        now=now,
        stale_after_seconds=30,
    )

    assert status_code == 503
    assert body["status"] == "starting"
