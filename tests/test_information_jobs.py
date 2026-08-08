from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_data.config import Settings
from quant_platform.api import (
    AnnouncementNlpRequest,
    CorpusNlpRequest,
    EventMarketResponseRequest,
)
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


def _worker(tmp_path: Path) -> LocalJobWorker:
    worker = object.__new__(LocalJobWorker)
    worker.settings = Settings(api_url="", token="", data_root=tmp_path / "data")
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
