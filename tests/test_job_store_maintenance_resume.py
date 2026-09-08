from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy import event, select, update

from quant_data.checkpoint import CheckpointStore
from quant_data.database import audit_events, jobs, row_dict, work_units
from quant_data.models import FetchSpec
from quant_platform.job_store import JobStore


def _digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cancelled(store: JobStore, tmp_path: Path, *, claimed: bool = True, **kwargs) -> dict:
    created = store.create(
        kwargs.pop("kind", "supplemental_us_market"),
        kwargs.pop(
            "payload", {"bundle": "us_market", "pipeline_id": "original", "pipeline_next_index": 11}
        ),
        tmp_path / "download.log",
        max_attempts=kwargs.pop("max_attempts", 3),
        idempotency_key="original-download",
        **kwargs,
    )
    if claimed:
        assert store.claim_next()["id"] == created["id"]
        store.update_progress(created["id"], {"completed": 17, "checkpoint": "unchanged"})
    store.request_cancel(created["id"])
    if claimed:
        store.mark_cancelled(created["id"])
    return store.get(created["id"])


def _resume(store: JobStore, job: dict, **kwargs) -> dict:
    arguments = {
        "expected_payload_sha256": _digest(job["payload"]),
        "expected_attempts": job["attempts"],
        "actor": "test-operator",
        "reason": "Authorized maintenance release",
    }
    arguments.update(kwargs)
    return store.resume_cancelled_data_for_maintenance(job["id"], **arguments)


def test_maintenance_resume_retains_checkpoint_and_consumed_attempts(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)
    job = _cancelled(store, tmp_path)
    checkpoint = CheckpointStore(database_url)
    checkpoint.add(
        [
            FetchSpec(
                dataset="us_income",
                api_name="us_income",
                scope={"symbol": "A"},
                params={"ts_code": "A"},
                fields=("ts_code",),
            )
        ]
    )
    with store.engine.begin() as c:
        c.execute(
            update(work_units).values(
                status="succeeded",
                attempts=2,
                output_path="/fixture/preserved.parquet",
                row_count=17,
                sha256="a" * 64,
            )
        )
        units_before = [row_dict(r) for r in c.execute(select(work_units))]
    resumed = _resume(store, job)
    for field in (
        "id",
        "kind",
        "payload",
        "progress",
        "attempts",
        "max_attempts",
        "log_path",
        "idempotency_key",
        "created_at",
    ):
        assert resumed[field] == job[field], field
    assert resumed["status"] == "queued"
    for field in (
        "exit_code",
        "error",
        "started_at",
        "finished_at",
        "cancel_requested_at",
        "next_attempt_at",
    ):
        assert resumed[field] is None
    with store.engine.connect() as c:
        assert [row_dict(r) for r in c.execute(select(work_units))] == units_before
        audit = c.execute(select(audit_events)).one()
        assert audit.details_json["attempts"] == 1
        assert audit.details_json["max_attempts"] == 3
        assert audit.details_json["payload_sha256"] == _digest(job["payload"])
        assert audit.details_json["progress_and_work_units_retained"] is True
    assert _resume(store, job) == resumed
    with store.engine.connect() as c:
        assert len(c.execute(select(audit_events)).all()) == 1
    claimed = store.claim_next()
    assert claimed["id"] == job["id"] and claimed["attempts"] == 2
    assert claimed["max_attempts"] == 3 and claimed["progress"] == job["progress"]
    with pytest.raises(ValueError):
        _resume(store, claimed)
    assert store.get(job["id"]) == claimed


def test_unstarted_cancelled_download_keeps_zero_attempts_until_claim(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)
    job = _cancelled(store, tmp_path, claimed=False)
    assert _resume(store, job)["attempts"] == 0
    assert store.claim_next()["attempts"] == 1


@pytest.mark.parametrize(
    "kind,bundle",
    [
        ("model_evaluate", "us_market"),
        ("strategy_backtest", "us_market"),
        ("data_qlib", "us_market"),
        ("supplemental_download", "us_market"),
        ("supplemental_research_corpus", "research_corpus"),
        ("supplemental_strategy_specialty_minutes", "strategy_specialty_minutes"),
        ("supplemental_unknown", "unknown"),
        ("supplemental_us_market", "hk_market"),
    ],
)
def test_maintenance_resume_rejects_unsupported_commands_and_bundle_drift(
    database_url: str,
    tmp_path: Path,
    kind: str,
    bundle: str,
) -> None:
    store = JobStore(database_url)
    job = _cancelled(store, tmp_path, claimed=False, kind=kind, payload={"bundle": bundle})
    with pytest.raises(ValueError, match="approved checkpoint"):
        _resume(store, job)
    assert store.get(job["id"]) == job


@pytest.mark.parametrize("status", ["queued", "running", "failed", "succeeded"])
def test_maintenance_resume_rejects_non_cancelled_jobs_without_owned_audit(
    database_url: str,
    tmp_path: Path,
    status: str,
) -> None:
    store = JobStore(database_url)
    job = _cancelled(store, tmp_path)
    with store.engine.begin() as c:
        c.execute(update(jobs).where(jobs.c.id == job["id"]).values(status=status))
    before = store.get(job["id"])
    with pytest.raises(ValueError):
        _resume(store, before)
    assert store.get(job["id"]) == before


@pytest.mark.parametrize(
    "overrides",
    [
        {"expected_payload_sha256": "0" * 64},
        {"expected_payload_sha256": "A" * 64},
        {"expected_payload_sha256": "invalid"},
        {"expected_attempts": 0},
        {"expected_attempts": True},
        {"expected_attempts": -1},
        {"actor": ""},
        {"reason": "  "},
    ],
)
def test_maintenance_resume_rejects_input_drift_and_missing_audit_context(
    database_url: str,
    tmp_path: Path,
    overrides: dict,
) -> None:
    store = JobStore(database_url)
    job = _cancelled(store, tmp_path)
    with pytest.raises(ValueError):
        _resume(store, job, **overrides)
    assert store.get(job["id"]) == job
    with store.engine.connect() as c:
        assert not c.execute(select(audit_events)).all()


def test_maintenance_resume_does_not_replenish_exhausted_budget(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)
    job = _cancelled(store, tmp_path, max_attempts=1)
    with pytest.raises(ValueError, match="exhausted"):
        _resume(store, job)
    assert store.get(job["id"]) == job


@pytest.mark.parametrize("field", ["cancel_requested_at", "finished_at"])
def test_maintenance_resume_requires_completed_normal_cancellation(
    database_url: str,
    tmp_path: Path,
    field: str,
) -> None:
    store = JobStore(database_url)
    job = _cancelled(store, tmp_path)
    with store.engine.begin() as c:
        c.execute(update(jobs).where(jobs.c.id == job["id"]).values({field: None}))
    with pytest.raises(ValueError, match="normal cancellation"):
        _resume(store, job)


def test_maintenance_resume_rolls_back_when_audit_cannot_be_persisted(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)
    job = _cancelled(store, tmp_path)

    def reject_audit(conn, clause, multiparams, params, execution_options):
        if getattr(clause, "is_insert", False) and clause.table is audit_events:
            raise RuntimeError("audit store unavailable")

    event.listen(store.engine, "before_execute", reject_audit)
    try:
        with pytest.raises(RuntimeError, match="audit store unavailable"):
            _resume(store, job)
    finally:
        event.remove(store.engine, "before_execute", reject_audit)
    assert store.get(job["id"]) == job


def test_queued_replay_requires_matching_actor_and_attempt_audit(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)
    job = _cancelled(store, tmp_path)
    _resume(store, job)
    with pytest.raises(ValueError, match="matching maintenance"):
        _resume(store, job, actor="different-operator")
    with store.engine.begin() as c:
        c.execute(
            update(audit_events).values(
                details_json={
                    "payload_sha256": _digest(job["payload"]),
                    "attempts": 0,
                    "max_attempts": 3,
                }
            )
        )
    with pytest.raises(ValueError, match="matching maintenance"):
        _resume(store, job)
