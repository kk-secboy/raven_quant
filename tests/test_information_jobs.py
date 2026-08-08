from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_data.config import Settings
from quant_platform import worker as worker_module
from quant_platform.api import (
    AnnouncementNlpRequest,
    CorpusNlpRequest,
    EventMarketResponseRequest,
    ExternalFactorEvaluationRequest,
)
from quant_platform.data_task_store import DATA_TASK_CATALOG, job_covers_catalog_scope
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


def _worker(tmp_path: Path) -> LocalJobWorker:
    worker = object.__new__(LocalJobWorker)
    worker.settings = Settings(api_url="", token="", data_root=tmp_path / "data")
    worker.project_root = tmp_path
    return worker


def test_information_request_models_fail_closed() -> None:
    with pytest.raises(ValidationError, match="end must not be before start"):
        AnnouncementNlpRequest(start="2024-02-01", end="2024-01-01")
    with pytest.raises(ValidationError, match="end must not be before start"):
        CorpusNlpRequest(start="2024-02-01", end="2024-01-01")
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        EventMarketResponseRequest(snapshot_name="fixture", horizons=[1, 1])
    with pytest.raises(ValidationError, match="between 1 and 252"):
        EventMarketResponseRequest(snapshot_name="fixture", horizons=[0])
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        ExternalFactorEvaluationRequest(
            dataset="qlib-fixture",
            candidate_ids=["candidate", "candidate"],
            periods={
                "train_start": "2024-01-01",
                "train_end": "2024-06-28",
                "valid_start": "2024-07-01",
                "valid_end": "2024-12-31",
                "test_start": "2025-01-13",
                "test_end": "2026-08-03",
            },
        )


def test_announcement_catalog_records_real_source_boundary() -> None:
    tasks = {definition.task_key: definition for definition in DATA_TASK_CATALOG}

    assert tasks["cn_cninfo_announcements"].range_start == "2016-01-01"
    assert tasks["cn_cninfo_announcements"].estimated_storage_gb == 320
    assert tasks["cn_announcement_nlp"].range_start == "2016-01-01"
    assert tasks["cn_corpus_nlp"].range_start == "2018-11-20"
    assert "cn_ashare_daily_full" in tasks["cn_corpus_nlp"].depends_on
    assert tasks["cn_event_market_response"].range_start == "2016-01-01"


def test_pilot_jobs_cannot_certify_full_information_catalog_scope() -> None:
    assert not job_covers_catalog_scope(
        "cn_cninfo_announcements", {"start": "2016-01-01", "limit": 25}
    )
    assert not job_covers_catalog_scope(
        "cn_announcement_nlp", {"start": "2024-01-01", "limit": 0}
    )
    assert not job_covers_catalog_scope("cn_announcement_nlp", {"limit": 0})
    assert job_covers_catalog_scope(
        "cn_announcement_nlp", {"start": "2016-01-01", "limit": 0}
    )
    assert not job_covers_catalog_scope(
        "cn_announcement_nlp",
        {
            "start": "2016-01-01",
            "limit": 0,
            "ts_codes": ["000001.SZ"],
        },
    )
    assert not job_covers_catalog_scope(
        "cn_announcement_nlp",
        {
            "start": "2016-01-01",
            "limit": 0,
            "categories": ["announcement"],
        },
    )
    assert job_covers_catalog_scope(
        "cn_announcement_nlp",
        {
            "start": "2016-01-01",
            "limit": 0,
            "categories": ["regulatory_letter"],
        },
    )
    assert job_covers_catalog_scope(
        "cn_announcement_nlp", {"start": "2015-12-31", "limit": 0}
    )
    assert not job_covers_catalog_scope("cn_corpus_nlp", {"limit": 100})
    assert not job_covers_catalog_scope("cn_corpus_nlp", {"limit": "not-an-int"})
    assert not job_covers_catalog_scope("cn_corpus_nlp", {"limit": 0})
    assert not job_covers_catalog_scope(
        "cn_corpus_nlp", {"start": "2024-01-01", "limit": 0}
    )
    assert job_covers_catalog_scope(
        "cn_corpus_nlp", {"start": "2018-11-20", "limit": 0}
    )
    assert not job_covers_catalog_scope(
        "cn_corpus_nlp",
        {
            "start": "2018-11-20",
            "limit": 0,
            "datasets": ["major_news"],
        },
    )
    assert not job_covers_catalog_scope(
        "cn_corpus_nlp",
        {
            "start": "2018-11-20",
            "limit": 0,
            "datasets": ["major_news", "cctv_news", "irm_qa_sh", "irm_qa_sz"],
            "ts_codes": ["000001.SZ"],
        },
    )
    assert job_covers_catalog_scope(
        "cn_corpus_nlp",
        {
            "start": "2018-11-20",
            "limit": 0,
            "datasets": ["major_news", "cctv_news", "irm_qa_sh", "irm_qa_sz"],
        },
    )
    assert job_covers_catalog_scope("cn_snapshot_build", {"limit": 100})


def test_worker_builds_announcement_and_corpus_nlp_commands(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    announcement = {
        "id": "announcement-job",
        "kind": "announcement_nlp",
        "payload": {
            "start": "2024-01-01",
            "end": "2026-08-03",
            "ts_codes": ["000001.SZ"],
            "categories": ["regulatory_letter"],
            "limit": 25,
        },
    }
    corpus = {
        "id": "corpus-job",
        "kind": "corpus_nlp",
        "payload": {
            "start": "2024-01-01",
            "end": "2026-08-03",
            "datasets": ["major_news", "npr"],
            "ts_codes": [],
            "limit": 0,
        },
    }

    announcement_command, announcement_result, announcement_env = worker._command(
        announcement
    )
    corpus_command, corpus_result, corpus_env = worker._command(corpus)

    assert "announcement-nlp" in announcement_command
    assert "regulatory_letter" in announcement_command
    assert "000001.SZ" in announcement_command
    assert announcement_result.name == "result.json"
    assert announcement_env == {}
    assert "corpus-nlp" in corpus_command
    assert "major_news,npr" in corpus_command
    assert corpus_result.name == "result.json"
    assert corpus_env == {}


def test_worker_builds_market_response_label_command(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    command, result_path, environment = worker._command(
        {
            "id": "labels-job",
            "kind": "event_market_response",
            "payload": {
                "snapshot_name": "cn-fixture",
                "horizons": [1, 3, 5, 20],
                "benchmark_code": "000300.SH",
            },
        }
    )

    assert "event-market-response" in command
    assert "cn-fixture" in command
    assert "1,3,5,20" in command
    assert "000300.SH" in command
    assert result_path.name == "result.json"
    assert environment == {}


def test_worker_builds_external_factor_evaluation_command(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    values_path = tmp_path / "logic.parquet"
    values_path.write_bytes(b"fixture")
    command, result_path, environment = worker._command(
        {
            "id": "external-eval-job",
            "kind": "external_factor_evaluate",
            "payload": {
                "dataset": "qlib-fixture",
                "dataset_path": str(tmp_path / "qlib"),
                "dataset_identity_sha256": "a" * 64,
                "periods": {
                    "train_start": "2024-01-01",
                    "train_end": "2024-06-28",
                    "valid_start": "2024-07-01",
                    "valid_end": "2024-12-31",
                    "test_start": "2025-01-13",
                    "test_end": "2026-08-03",
                },
                "universe": "cn_all",
                "benchmark": "SH000300",
                "candidates": [
                    {
                        "id": "candidate",
                        "values_path": str(values_path),
                        "code_sha256": "b" * 64,
                        "values_sha256": "c" * 64,
                        "experiment_family_id": "logic",
                        "experiment_count": 1,
                        "label_horizon_days": 1,
                    }
                ],
            },
        }
    )

    assert "evaluate_external_factor_batch.py" in " ".join(command)
    assert "--provider-uri" in command
    assert result_path.name == "result.json"
    assert "_MLFLOW_SERVER_ARTIFACT_ROOT" in environment
    manifest = json.loads((result_path.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidates"][0]["values_sha256"] == "c" * 64
    assert manifest["comparison_values"] == []


def test_worker_imports_external_evaluation_with_bound_periods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker(tmp_path)
    worker.research = object()
    captured: dict = {}

    def fake_import(store, result, **kwargs):
        captured.update(store=store, result=result, **kwargs)
        return []

    monkeypatch.setattr(worker_module, "import_external_evaluations", fake_import)
    job = {
        "id": "external-eval-job",
        "payload": {
            "dataset": "qlib-fixture",
            "dataset_identity_sha256": "a" * 64,
            "periods": {
                "train_start": "2024-01-01",
                "train_end": "2024-06-28",
                "valid_start": "2024-07-01",
                "valid_end": "2024-12-31",
                "test_start": "2025-01-13",
                "test_end": "2026-08-03",
            },
        },
    }
    result = {"status": "ok", "evaluations": []}

    worker._import_external_factor_evaluations(job, result)

    assert captured["store"] is worker.research
    assert captured["result"] == result
    assert captured["dataset"] == "qlib-fixture"
    assert captured["periods"]["test_end"].isoformat() == "2026-08-03"
    assert captured["artifact_path"].name == "result.json"
