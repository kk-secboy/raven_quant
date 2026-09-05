from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from quant_platform.job_store import JobStore
from quant_platform.worker import LocalJobWorker


def _corpus_successor(store: JobStore, tmp_path: Path, name: str) -> dict:
    worker = object.__new__(LocalJobWorker)
    worker.store = store
    worker.settings = SimpleNamespace(data_root=tmp_path)
    worker.notify = lambda: None
    return worker._queue_data_pipeline_successor(
        {
            "id": f"download-{name}",
            "kind": "supplemental_research_corpus",
            "max_attempts": 3,
            "payload": {
                "pipeline_id": f"corpus-{name}",
                "snapshot_name": name,
                "profile": "research-assets",
                "start": "2026-08-01",
                "end": "2026-09-04",
                "pipeline_steps": [{"kind": "data_verify", "payload": {}}],
                "pipeline_next_index": 0,
            },
        }
    )


def test_corpus_successor_queues_its_own_verify_behind_another_pipeline(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)
    first = _corpus_successor(store, tmp_path, "first")
    claimed = store.claim_next(("data_verify",))
    assert claimed is not None and claimed["id"] == first["id"]
    original = deepcopy(store.get(first["id"]))

    second = _corpus_successor(store, tmp_path, "second")
    replay = _corpus_successor(store, tmp_path, "second")

    assert second["id"] == replay["id"] != first["id"]
    assert second["status"] == "queued"
    assert second["payload"]["pipeline_id"] == "corpus-second"
    assert second["payload"]["snapshot_name"] == "second"
    assert store.get(first["id"]) == original
    assert store.claim_next(("data_verify",)) is None
    assert store.get(second["id"])["attempts"] == 0

    # An occupied verification lane must not block other job kinds.
    unrelated = store.create("announcement_nlp", {}, tmp_path / "nlp.log")
    unrelated_claim = store.claim_next()
    assert unrelated_claim is not None and unrelated_claim["id"] == unrelated["id"]
    store.finish(unrelated["id"], exit_code=0)
    store.finish(first["id"], exit_code=0)
    next_claim = store.claim_next(("data_verify",))
    assert next_claim is not None and next_claim["id"] == second["id"]


def test_verify_successor_keeps_strict_idempotency_payload_binding(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)
    first = _corpus_successor(store, tmp_path, "frozen")

    with pytest.raises(ValueError, match="already bound to a different job payload"):
        store.create(
            "data_verify",
            {**first["payload"], "pipeline_id": "different-pipeline"},
            tmp_path / "changed.log",
            dedupe_active_kind=False,
            idempotency_key=first["idempotency_key"],
        )
    assert store.get(first["id"])["payload"]["pipeline_id"] == "corpus-frozen"


def test_concurrent_workers_can_claim_only_one_queued_verification(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)
    first = _corpus_successor(store, tmp_path, "one")
    second = _corpus_successor(store, tmp_path, "two")
    barrier = Barrier(2)

    def claim() -> dict | None:
        concurrent_store = JobStore(database_url)
        try:
            barrier.wait(timeout=10)
            return concurrent_store.claim_next(("data_verify",))
        finally:
            concurrent_store.engine.dispose()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(claim) for _ in range(2)]
        outcomes = [future.result(timeout=20) for future in futures]

    claimed = [item for item in outcomes if item is not None]
    assert len(claimed) == 1
    assert claimed[0]["id"] == first["id"]
    assert store.get(second["id"])["status"] == "queued"
    store.finish(first["id"], exit_code=0)
    next_claim = store.claim_next(("data_verify",))
    assert next_claim is not None and next_claim["id"] == second["id"]
