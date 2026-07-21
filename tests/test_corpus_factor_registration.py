"""Database-backed registration tests for the corpus NLP factor channel.

The no_database logic/fail-closed coverage lives in tests/test_corpus_nlp.py;
this file exercises the ResearchStore writes (success, idempotency, CLI).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import quant_platform.db_cli as db_cli
from quant_platform import corpus_nlp as corpus
from quant_platform.research_store import ResearchStore

NOW = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
STORE_RECORD = {
    "api_key": "sk-test-fake-key",
    "api_base": "https://llm.example.invalid/v1",
    "chat_model": "test-model",
}
CALENDAR_DAYS = [
    (date(2024, 1, 2), 1),
    (date(2024, 1, 3), 0),
    (date(2024, 1, 4), 1),
    (date(2024, 1, 5), 1),
    (date(2024, 1, 6), 0),
    (date(2024, 1, 7), 0),
    (date(2024, 1, 8), 1),
    (date(2024, 1, 9), 1),
]


class FakeChatClient:
    def __init__(self, script: list) -> None:
        self.script = list(script)

    def complete(self, messages: list[dict], *, model: str) -> str:
        if not self.script:
            raise AssertionError("unexpected LLM call")
        return self.script.pop(0)


class FakeSecretStore:
    def get(self, name: str) -> dict | None:
        assert name == "llm"
        return STORE_RECORD


def _payload(sentiment: float = 0.5) -> str:
    return json.dumps(
        {"sentiment": sentiment, "topic": "policy", "confidence": 0.9},
        ensure_ascii=False,
    )


def _write_parquet(directory: Path, rows: list[dict], name: str = "data.parquet") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / name, index=False)


def _seed_full(data_root: Path) -> None:
    _write_parquet(
        data_root / "units" / "trade_cal",
        [
            {"cal_date": day.strftime("%Y%m%d"), "is_open": flag, "pretrade_date": ""}
            for day, flag in CALENDAR_DAYS
        ],
    )
    _write_parquet(
        data_root / "units" / "major_news",
        [{"title": "要闻", "content": "正文", "pub_time": "2024-01-02 10:00:00"}],
    )
    _write_parquet(
        data_root / "units" / "npr",
        [{"title": "政策", "pubtime": "2024-01-04 09:00:00"}],
    )
    _write_parquet(
        data_root / "units" / "cctv_news",
        [{"date": "20240105", "title": "联播", "content": "内容"}],
    )
    _write_parquet(
        data_root / "units" / "irm_qa_sh",
        [{"trade_date": "20240102", "ts_code": "600000.SH", "q": "产能如何？"}],
    )
    _write_parquet(
        data_root / "units" / "irm_qa_sz",
        [{"trade_date": "20240102", "ts_code": "000002.SZ", "q": "订单如何？"}],
    )
    corpus.process_corpus(
        data_root,
        secret_store=FakeSecretStore(),
        chat_client=FakeChatClient([_payload()] * 5),
        now=lambda: NOW,
        environ={},
    )


def test_register_success(database_url: str, tmp_path: Path) -> None:
    _seed_full(tmp_path)
    store = ResearchStore(database_url)
    factors_dir = corpus.default_factors_dir(tmp_path)

    result = corpus.register_corpus_factor(
        store, factors_dir, factor_name=corpus.POLICY_FACTOR_NAME
    )

    assert result["created"] is True
    candidate = store.get_candidate(result["candidate_id"])
    assert candidate["name"] == corpus.POLICY_FACTOR_NAME
    assert candidate["status"] == "awaiting_evaluation"
    assert candidate["values_sha256"] == result["values_sha256"]
    variables = candidate["variables"]
    assert variables["source"]["dataset"] == "corpus_nlp_fields"
    assert variables["source"]["source_datasets"] == ["npr", "cctv_news"]
    assert variables["source"]["prompt_version"] == corpus.PROMPT_VERSION
    assert "pubtime" in candidate["description"]
    run = store.get_run(result["run_id"])
    assert run["kind"] == corpus.IMPORT_RUN_KIND
    assert run["status"] == "succeeded"
    assert run["dataset"] == "corpus_nlp_fields"
    events = {event["event_type"] for event in store.list_events(run["id"])}
    assert {"run.created", "candidate.imported", "run.succeeded"} <= events


def test_register_is_idempotent_for_same_sha256(database_url: str, tmp_path: Path) -> None:
    _seed_full(tmp_path)
    store = ResearchStore(database_url)
    factors_dir = corpus.default_factors_dir(tmp_path)

    first = corpus.register_corpus_factor(
        store, factors_dir, factor_name=corpus.NEWS_FACTOR_NAME
    )
    second = corpus.register_corpus_factor(
        store, factors_dir, factor_name=corpus.NEWS_FACTOR_NAME
    )

    assert first["created"] is True
    assert second["created"] is False
    assert second["candidate_id"] == first["candidate_id"]
    candidates = [
        item
        for item in store.list_candidates(limit=100)
        if item["name"] == corpus.NEWS_FACTOR_NAME
    ]
    assert len(candidates) == 1


def test_register_checksum_mismatch_writes_nothing(database_url: str, tmp_path: Path) -> None:
    _seed_full(tmp_path)
    factors_dir = corpus.default_factors_dir(tmp_path)
    entry = (factors_dir / f"{corpus.IRM_QA_FACTOR_NAME}.parquet")
    tampered = pd.read_parquet(entry)
    tampered[corpus.IRM_QA_FACTOR_NAME] = tampered[corpus.IRM_QA_FACTOR_NAME] + 1
    tampered.to_parquet(entry, index=False)
    store = ResearchStore(database_url)

    with pytest.raises(ValueError, match="does not match the manifest sha256"):
        corpus.register_corpus_factor(
            store, factors_dir, factor_name=corpus.IRM_QA_FACTOR_NAME
        )

    assert store.list_candidates(limit=100) == []
    assert store.list_runs(limit=100) == []


def test_cli_registers_all_factors_idempotently(
    database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_full(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    runner = CliRunner()
    first = runner.invoke(db_cli.app, ["register-corpus-factor"])
    assert first.exit_code == 0, first.output
    payload = json.loads(first.output)
    assert {item["factor_name"] for item in payload["factors"]} == set(
        corpus.CORPUS_FACTOR_NAMES
    )
    assert all(item["created"] is True for item in payload["factors"])

    second = runner.invoke(db_cli.app, ["register-corpus-factor"])
    assert second.exit_code == 0, second.output
    repeated = json.loads(second.output)
    assert all(item["created"] is False for item in repeated["factors"])
    first_ids = {item["factor_name"]: item["candidate_id"] for item in payload["factors"]}
    for item in repeated["factors"]:
        assert item["candidate_id"] == first_ids[item["factor_name"]]
