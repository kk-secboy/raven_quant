from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from quant_data.config import Settings
from quant_platform.job_store import (
    JobStore,
    research_asset_acquisition_idempotency_key,
)
from quant_platform.scheduler import SchedulerEngine
from quant_platform.worker import LocalJobWorker


def test_job_idempotency_reuses_only_the_exact_payload(database_url: str, tmp_path: Path) -> None:
    store = JobStore(database_url)
    key = "research-assets:auto:2026-08-14:contract"
    payload = {
        "mode": "automatic",
        "snapshot_name": "research-assets-20260814",
        "as_of": "2026-08-14",
        "include_tushare": True,
        "include_arxiv": True,
    }
    first = store.create(
        "research_asset_acquire",
        payload,
        tmp_path / "first.log",
        dedupe_active_kind=False,
        idempotency_key=key,
    )
    repeated = store.create(
        "research_asset_acquire",
        dict(payload),
        tmp_path / "repeated.log",
        dedupe_active_kind=False,
        idempotency_key=key,
    )
    assert repeated["id"] == first["id"]

    with pytest.raises(ValueError, match="different job payload"):
        store.create(
            "research_asset_acquire",
            {**payload, "include_tushare": False},
            tmp_path / "conflict.log",
            dedupe_active_kind=False,
            idempotency_key=key,
        )


def test_automatic_acquisition_key_binds_snapshot_and_enabled_sources() -> None:
    base = {
        "research_day": "2026-08-14",
        "snapshot_name": "research-assets-20260814",
        "include_tushare": True,
        "include_arxiv": True,
    }
    key = research_asset_acquisition_idempotency_key(**base)
    assert key != research_asset_acquisition_idempotency_key(
        **{**base, "snapshot_name": "research-assets-20260814-r2"}
    )
    assert key != research_asset_acquisition_idempotency_key(**{**base, "include_tushare": False})
    assert key != research_asset_acquisition_idempotency_key(**{**base, "include_arxiv": False})


@pytest.mark.no_database
def test_scheduler_queues_nothing_when_tushare_snapshot_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = object.__new__(SchedulerEngine)
    engine.settings = Settings(
        api_url="",
        token="",
        data_root=tmp_path / "data",
        database_url="postgresql://unused",
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        engine,
        "_enqueue_research_asset_source",
        lambda **kwargs: calls.append(kwargs) or 1,
    )

    def missing_snapshot(*_args: Any, **_kwargs: Any) -> str:
        raise ValueError("no verified research asset snapshot")

    monkeypatch.setattr(
        "quant_platform.scheduler.latest_verified_research_asset_snapshot",
        missing_snapshot,
    )

    # The arXiv leg fed the frozen general_model scenario; without it a missing
    # Tushare snapshot simply means no acquisition work for the day.
    assert engine._enqueue_daily_research_assets(datetime(2026, 8, 20, 13, tzinfo=UTC)) == 0
    assert calls == []


@pytest.mark.no_database
def test_scheduler_queues_only_the_tushare_research_report_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = object.__new__(SchedulerEngine)
    engine.settings = Settings(
        api_url="",
        token="",
        data_root=tmp_path / "data",
        database_url="postgresql://unused",
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        engine,
        "_enqueue_research_asset_source",
        lambda **kwargs: calls.append(kwargs) or 1,
    )
    monkeypatch.setattr(
        "quant_platform.scheduler.latest_verified_research_asset_snapshot",
        lambda *_args, **_kwargs: "research-assets-verified",
    )

    assert engine._enqueue_daily_research_assets(datetime(2026, 8, 20, 13, tzinfo=UTC)) == 1
    assert calls == [
        {
            "research_day": datetime(2026, 8, 20).date(),
            "snapshot_name": "research-assets-verified",
            "include_tushare": True,
            "include_arxiv": False,
        }
    ]


@pytest.mark.no_database
def test_worker_imports_partial_assets_and_does_not_retry_blocked_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_path = tmp_path / "result.json"
    captured: dict[str, Any] = {}

    class FakeStore:
        @staticmethod
        def update_progress(_job_id: str, _progress: dict[str, Any]) -> None:
            return None

        def finish_or_retry(self, job_id: str, **kwargs: Any) -> bool:
            captured.update({"job_id": job_id, **kwargs})
            return False

    class FakeCandidates:
        paths: list[Path] = []

        def import_manifest(self, path: Path, *, actor: str) -> dict[str, str]:
            assert actor == "research-asset-worker"
            self.paths.append(path)
            return {
                "content_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
            }

    class FakeProcess:
        returncode = 3

        def poll(self) -> int:
            return self.returncode

    def fake_popen(*args: Any, **kwargs: Any) -> FakeProcess:
        del args, kwargs
        result_path.write_text(
            json.dumps(
                {
                    "status": "blocked",
                    "published_asset_ids": ["arxiv-partial-1"],
                    "blocked": 2,
                    "failed": 1,
                    "tushare_selected": 0,
                    "arxiv_selected": 3,
                }
            ),
            encoding="utf-8",
        )
        return FakeProcess()

    worker = object.__new__(LocalJobWorker)
    worker.store = FakeStore()  # type: ignore[assignment]
    worker.project_root = tmp_path
    worker.settings = Settings(
        api_url="",
        token="",
        data_root=tmp_path / "data",
        database_url="postgresql://unused",
    )
    candidates = FakeCandidates()
    worker.rdagent_candidates = candidates  # type: ignore[assignment]
    monkeypatch.setattr(
        worker,
        "_command",
        lambda _job: (["fake"], result_path, {}),
    )
    monkeypatch.setattr("quant_platform.worker.subprocess.Popen", fake_popen)

    worker._run(
        {
            "id": "asset-job-1",
            "kind": "research_asset_acquire",
            "payload": {"mode": "automatic"},
            "log_path": str(tmp_path / "asset-job.log"),
        }
    )

    assert candidates.paths == [
        tmp_path / "data" / "artifacts" / "research-assets" / "arxiv-partial-1" / "manifest.json"
    ]
    assert captured["retryable"] is False
    assert captured["exit_code"] == 3
    assert captured["result"] == {
        "status": "blocked",
        "mode": "automatic",
        "published": 1,
        "assets": [
            {
                "asset_id": "arxiv-partial-1",
                "content_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
            }
        ],
        "blocked": 2,
        "failed": 1,
        "tushare_selected": 0,
        "arxiv_selected": 3,
        "daily_limits": {"tushare_research_report": 20, "arxiv": 3},
    }


def test_research_asset_snapshot_stops_before_qlib_and_corpus_step_is_allowed(
    tmp_path: Path,
) -> None:
    worker = object.__new__(LocalJobWorker)
    terminal = {
        "id": "snapshot-job",
        "kind": "data_snapshot",
        "payload": {
            "profile": "research-assets",
            "snapshot_name": "research-assets-20260814",
        },
    }
    assert worker._queue_data_pipeline_successor(terminal) is terminal

    captured: dict[str, Any] = {}

    class FakeStore:
        def create(self, kind: str, payload: dict[str, Any], *args: Any, **kwargs: Any) -> dict:
            captured.update({"kind": kind, "payload": payload, "args": args, "kwargs": kwargs})
            return {"id": "corpus-job", "kind": kind, "payload": payload}

    worker.store = FakeStore()  # type: ignore[assignment]
    worker.settings = Settings(
        api_url="",
        token="",
        data_root=tmp_path / "data",
        database_url="postgresql://unused",
    )
    worker.notify = lambda: None  # type: ignore[method-assign]
    worker._queue_data_pipeline_successor(
        {
            "id": "bootstrap-job",
            "kind": "bootstrap",
            "payload": {
                "pipeline_id": "research-assets-pipeline",
                "profile": "research-assets",
                "start": "2017-01-01",
                "end": "2026-08-14",
                "snapshot_name": "research-assets-20260814",
                "pipeline_steps": [
                    {
                        "kind": "supplemental_research_corpus",
                        "payload": {"bundle": "research_corpus"},
                    }
                ],
                "pipeline_next_index": 0,
            },
        }
    )
    assert captured["kind"] == "supplemental_research_corpus"
    assert captured["payload"]["bundle"] == "research_corpus"


def test_research_asset_pipeline_downloads_only_research_report(tmp_path: Path) -> None:
    class FakeSecrets:
        @staticmethod
        def get(_name: str) -> dict[str, str]:
            return {"api_url": "https://example.invalid", "token": "test-token"}

    worker = object.__new__(LocalJobWorker)
    worker.settings = Settings(
        api_url="",
        token="",
        data_root=tmp_path / "data",
        database_url="postgresql://unused",
    )
    worker.runtime_secrets = FakeSecrets()  # type: ignore[assignment]

    command, _result_path, _environment = worker._command(
        {
            "id": "research-report-job",
            "kind": "supplemental_research_corpus",
            "payload": {
                "bundle": "research_corpus",
                "profile": "research-assets",
                "start": "2017-01-01",
                "end": "2026-08-14",
            },
        }
    )

    assert command[3] == "research-report-download"
    assert "supplemental-download" not in command
