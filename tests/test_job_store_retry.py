from __future__ import annotations

import math
from pathlib import Path

from quant_platform.job_store import FORMAL_DATA_AUTO_RETRY_KINDS, JobStore
from quant_platform.scheduler import INFORMATION_CONFLICTING_JOB_KINDS


def test_every_governed_data_or_information_job_can_be_a_retryable_root() -> None:
    assert set(INFORMATION_CONFLICTING_JOB_KINDS) <= FORMAL_DATA_AUTO_RETRY_KINDS


def test_formal_data_pipeline_roots_default_to_three_attempts_without_mutating_identity(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)
    for kind in sorted(FORMAL_DATA_AUTO_RETRY_KINDS):
        payload = {
            "pipeline_id": f"pipeline-{kind}",
            "snapshot_name": "snapshot-fixture",
        }
        idempotency_key = f"pipeline-root:{kind}"

        created = store.create(
            kind,
            payload,
            tmp_path / f"{kind}.log",
            idempotency_key=idempotency_key,
        )

        assert created["max_attempts"] == 3, kind
        assert created["payload"] == payload, kind
        assert created["idempotency_key"] == idempotency_key, kind


def test_explicit_attempt_limit_overrides_formal_data_default(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)

    created = store.create(
        "announcement_nlp",
        {"pipeline_id": "explicit-single-attempt"},
        tmp_path / "announcement.log",
        max_attempts=1,
    )

    assert created["max_attempts"] == 1


def test_non_retryable_formal_data_failure_is_not_requeued(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)
    created = store.create("announcement_nlp", {}, tmp_path / "announcement.log")
    claimed = store.claim_next()
    assert claimed is not None and claimed["id"] == created["id"]

    assert not store.finish_or_retry(
        claimed["id"],
        exit_code=2,
        error="invalid immutable input contract",
        retryable=False,
    )

    failed = store.get(claimed["id"])
    assert failed["status"] == "failed"
    assert failed["attempts"] == 1
    assert failed["next_attempt_at"] is None


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
            "estimated_rate": float("nan"),
        },
    )

    progress = store.get(claimed["id"])["progress"]
    assert progress["execution_phase"] == "adaptive_recovery"
    assert progress["checkpoint"]["superseded"] == 2
    assert progress["estimated_rate"] is None
    assert progress["_quantlab_json_normalization"]["recorded_occurrences"][0]["path"] == [
        "estimated_rate"
    ]


def test_job_result_jsonb_replaces_nan_with_audited_null(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)
    created = store.create("information_factor_evaluate", {}, tmp_path / "evaluation.log")
    claimed = store.claim_next()
    assert claimed is not None and claimed["id"] == created["id"]
    raw_result = {"status": "ok", "metrics": {"turnover": float("nan")}}

    store.finish(claimed["id"], exit_code=0, result=raw_result)

    persisted = store.get(claimed["id"])
    assert persisted["status"] == "succeeded"
    assert persisted["progress"]["metrics"]["turnover"] is None
    metadata = persisted["progress"]["_quantlab_json_normalization"]
    assert metadata["replacement_count"] == 1
    assert metadata["recorded_occurrences"][0]["path"] == ["metrics", "turnover"]
    assert math.isnan(raw_result["metrics"]["turnover"])


def test_retry_progress_jsonb_uses_the_same_non_finite_boundary(
    database_url: str, tmp_path: Path
) -> None:
    store = JobStore(database_url)
    created = store.create("rdagent_factor", {}, tmp_path / "rdagent.log")
    claimed = store.claim_next()
    assert claimed is not None and claimed["id"] == created["id"]

    assert store.finish_or_retry(
        claimed["id"],
        exit_code=1,
        error="temporary failure",
        result={"metric": float("inf")},
        retryable=True,
    )

    progress = store.get(claimed["id"])["progress"]
    assert progress["metric"] is None
    assert progress["_quantlab_json_normalization"]["replacement_count"] == 1
