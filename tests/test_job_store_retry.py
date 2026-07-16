from __future__ import annotations

from pathlib import Path

from quant_platform.job_store import JobStore


def test_transient_research_job_retries_with_a_bounded_backoff(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)
    created = store.create("rdagent_factor", {}, tmp_path / "rdagent.log")
    claimed = store.claim_next()
    assert claimed is not None and claimed["id"] == created["id"]
    assert claimed["attempts"] == 1
    assert claimed["max_attempts"] == 3
    assert store.finish_or_retry(
        claimed["id"], exit_code=1, error="temporary runtime failure", retryable=True
    )
    requeued = store.get(claimed["id"])
    assert requeued["status"] == "queued"
    assert requeued["next_attempt_at"] is not None


def test_final_strategy_backtest_is_never_automatically_retried(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)
    created = store.create("strategy_backtest", {}, tmp_path / "backtest.log")
    claimed = store.claim_next()
    assert claimed is not None and claimed["id"] == created["id"]
    assert claimed["max_attempts"] == 1
    assert not store.finish_or_retry(
        claimed["id"], exit_code=1, error="final test failed", retryable=True
    )
    assert store.get(claimed["id"])["status"] == "failed"


def test_running_job_persists_live_downloader_progress(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)
    created = store.create("ashare_5m_download", {}, tmp_path / "minute.log")
    claimed = store.claim_next()
    assert claimed is not None and claimed["id"] == created["id"]

    store.update_progress(
        claimed["id"],
        {
            "status": "running",
            "execution_phase": "adaptive_recovery",
            "checkpoint": {"succeeded": 12, "superseded": 2},
        },
    )

    progress = store.get(claimed["id"])["progress"]
    assert progress["execution_phase"] == "adaptive_recovery"
    assert progress["checkpoint"]["superseded"] == 2
