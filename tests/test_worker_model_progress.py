from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_platform import worker as runtime
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


def _worker():
    worker = object.__new__(LocalJobWorker)
    persisted = []
    worker.store = SimpleNamespace(
        update_progress=lambda job_id, value: persisted.append((job_id, value)),
        cancellation_requested=lambda _job_id: False,
    )
    worker._retry_transient_database = lambda call: call()
    return worker, persisted


def _paths(tmp_path: Path):
    attempt = (
        tmp_path / "model-evaluations" / ("a" * 32) / ("b" * 32) / "attempts"
        / ("attempt-0001-" + "c" * 32)
    )
    attempt.mkdir(parents=True)
    return attempt / "result.json", attempt / "model-progress.json"


def _progress():
    return {
        "contract_version": "model-batch-progress-v1-observe-only", "status": "running",
        "execution_phase": "model_compute", "phase_label": "模型计算中",
        "completed_cells": 0, "planned_cells": 3,
        "active_cells": [{"candidate_id": "candidate-1", "seed": 11, "progress": {
            "elapsed_seconds": 3600, "automatic_termination": False,
            "warnings": ["elapsed_warning", "progress_not_observed"],
        }}],
        "warnings": ["elapsed_warning", "progress_not_observed"],
    }


def test_model_sidecar_persists_observation_then_real_result_wins_same_timestamp(
    tmp_path: Path,
) -> None:
    worker, persisted = _worker()
    result, sidecar = _paths(tmp_path)
    progress = _progress()
    sidecar.write_text(json.dumps(progress), encoding="utf-8")
    token = worker._sync_live_progress("job", result, None)
    assert token == -sidecar.stat().st_mtime_ns
    assert persisted == [("job", progress)]
    assert worker._sync_live_progress("job", result, token) == token
    assert len(persisted) == 1
    final = {"status": "ok", "evaluations": [{"candidate_id": "candidate-1"}]}
    result.write_text(json.dumps(final), encoding="utf-8")
    shared_stamp = sidecar.stat().st_mtime_ns
    os.utime(result, ns=(shared_stamp, shared_stamp))
    final_token = worker._sync_live_progress("job", result, token)
    assert final_token == result.stat().st_mtime_ns
    assert persisted == [("job", progress), ("job", final)]
    sidecar.write_text(json.dumps({**progress, "completed_cells": 3}), encoding="utf-8")
    assert worker._sync_live_progress("job", result, final_token) == final_token
    assert len(persisted) == 2


@pytest.mark.parametrize("content", [
    "{", "[]", '{"status":"ok","evaluations":[{}]}',
    json.dumps({**_progress(), "active_cells": [{}, {}, {}]}), " " * (1024 * 1024 + 1),
], ids=["partial-json", "wrong-container", "not-progress", "too-many-cells", "oversized"])
def test_malformed_or_unbounded_model_sidecar_does_not_mutate_job(
    tmp_path: Path, content: str,
) -> None:
    worker, persisted = _worker()
    result, sidecar = _paths(tmp_path)
    sidecar.write_text(content, encoding="utf-8")
    assert worker._sync_live_progress("job", result, None) is None
    assert persisted == []


def test_non_model_job_cannot_import_model_sidecar(tmp_path: Path) -> None:
    worker, persisted = _worker()
    (tmp_path / "model-progress.json").write_text(json.dumps(_progress()), encoding="utf-8")
    assert worker._sync_live_progress("job", tmp_path / "result.json", None) is None
    assert persisted == []


def test_hardlinked_model_sidecar_is_ignored(tmp_path: Path) -> None:
    worker, persisted = _worker()
    result, sidecar = _paths(tmp_path)
    foreign = tmp_path / "foreign.json"
    foreign.write_text(json.dumps(_progress()), encoding="utf-8")
    os.link(foreign, sidecar)
    assert worker._sync_live_progress("job", result, None) is None
    assert persisted == []


def test_worker_keeps_model_child_alive_after_long_elapsed_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, _ = _worker()
    polls = iter((None, None, None, 0))
    process = SimpleNamespace(
        poll=lambda: next(polls),
        terminate=lambda: pytest.fail("elapsed model time cannot terminate the child"),
        kill=lambda: pytest.fail("elapsed model time cannot kill the child"),
    )
    clock = iter((0.0, 100_000.0, 200_000.0, 300_000.0))
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    assert worker._monitor_process("job", None, process) == (False, None)
