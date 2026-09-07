from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from quant_data.database import jobs
from quant_platform.job_store import JobStore


def test_model_recovery_excludes_live_and_unidentified_owners(database_url, tmp_path):
    store = JobStore(database_url)
    created = store.create("model_evaluate", {}, tmp_path / "model.log", max_attempts=2)
    claimed = store.claim_next(("model_evaluate",))
    assert claimed["id"] == created["id"]
    claim = store.active_model_claim_identity(claimed["id"], expected_attempts=claimed["attempts"])
    assert store.recover_interrupted(
        ("model_evaluate",), recoverable_model_claims=()
    ) == 0
    assert store.recover_interrupted(
        ("model_evaluate",), protected_job_ids=(claimed["id"],),
        recoverable_model_claims=(claim,),
    ) == 0
    assert store.get(created["id"])["status"] == "running"
    assert store.recover_interrupted(
        ("model_evaluate",), recoverable_model_claims=(claim,)
    ) == 1
    recovered = store.get(created["id"])
    assert recovered["status"] == "queued"
    assert recovered["attempts"] == 1 and recovered["max_attempts"] == 2


def test_prior_cleanup_receipt_cannot_recover_new_claim_with_same_attempt_count(
    database_url, tmp_path,
):
    store = JobStore(database_url)
    created = store.create("model_evaluate", {}, tmp_path / "model.log", max_attempts=2)
    claimed = store.claim_next(("model_evaluate",))
    claim = store.active_model_claim_identity(claimed["id"], expected_attempts=claimed["attempts"])
    old_start = datetime.fromisoformat(claim["started_at"]) - timedelta(seconds=1)
    assert store.recover_interrupted(("model_evaluate",), recoverable_model_claims=({
        "job_id": claimed["id"], "attempts": claimed["attempts"],
        "started_at": old_start.isoformat(),
    },)) == 0
    assert store.get(created["id"])["status"] == "running"


def test_model_cleanup_claim_preserves_microseconds_and_rejects_same_second_other_claim(
    database_url, tmp_path,
):
    store = JobStore(database_url)
    created = store.create("model_evaluate", {}, tmp_path / "model.log", max_attempts=2)
    claimed = store.claim_next(("model_evaluate",))
    exact_start = datetime(2026, 9, 7, 1, 2, 3, 456789, tzinfo=UTC)
    with store.engine.begin() as connection:
        connection.execute(
            update(jobs).where(jobs.c.id == created["id"]).values(started_at=exact_start)
        )
    claim = store.active_model_claim_identity(claimed["id"], expected_attempts=1)
    assert claim["started_at"] == exact_start.isoformat(timespec="microseconds")
    assert store.get(created["id"])["started_at"] == exact_start.isoformat(timespec="seconds")
    stale = {**claim, "started_at": (exact_start - timedelta(microseconds=1)).isoformat()}
    assert store.recover_interrupted(
        ("model_evaluate",), recoverable_model_claims=(stale,),
    ) == 0
    assert store.get(created["id"])["status"] == "running"
    assert store.recover_interrupted(
        ("model_evaluate",), recoverable_model_claims=(claim,),
    ) == 1


def test_model_claim_capture_rejects_a_changed_attempt_or_terminal_job(database_url, tmp_path):
    store = JobStore(database_url)
    created = store.create("model_evaluate", {}, tmp_path / "model.log", max_attempts=2)
    claimed = store.claim_next(("model_evaluate",))
    with pytest.raises(ValueError, match="running job attempt"):
        store.active_model_claim_identity(claimed["id"], expected_attempts=2)
    store.mark_cancelled(created["id"])
    with pytest.raises(ValueError, match="running job attempt"):
        store.active_model_claim_identity(claimed["id"], expected_attempts=1)
