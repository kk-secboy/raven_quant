from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
import requests
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from typer.testing import CliRunner

import quant_data.cli as cli_module
from quant_data import cninfo_announcements as cninfo
from quant_data.cli import app
from quant_data.rate_limit import GlobalRateGate
from quant_platform import announcement_nlp as nlp

pytestmark = pytest.mark.no_database

NOW = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)
STORE_RECORD = {
    "api_key": "sk-test-fake-key",
    "api_base": "https://llm.example.invalid/v1",
    "chat_model": "test-model",
}


def _pdf_bytes(lines: list[str]) -> bytes:
    """Build a minimal text-bearing PDF with pypdf for fixtures."""

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    operators = ["BT /F1 12 Tf 14 TL 72 720 Td"]
    for index, line in enumerate(lines):
        if index:
            operators.append("T*")
        operators.append(f"({line}) Tj")
    operators.append("ET")
    stream = DecodedStreamObject()
    stream.set_data(" ".join(operators).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _blank_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _payload(**overrides) -> str:
    body = {
        "event_type": "periodic_report",
        "tone_score": 0.6,
        "key_numbers": {"net_profit_yoy": "+30%"},
        "impact_direction": "positive",
        "impact_horizon": "short_term",
        "impact_channels": ["earnings"],
        "logic_summary": "Profit growth may improve near-term earnings expectations.",
        "confidence": 0.9,
    }
    body.update(overrides)
    return json.dumps(body, ensure_ascii=False)


def _batch_payload(item_ids: list[str], **overrides) -> str:
    base = json.loads(_payload(**overrides))
    return json.dumps(
        {"items": [{"item_id": item_id, **base} for item_id in item_ids]},
        ensure_ascii=False,
    )


class FakeChatClient:
    """Scripted ChatCompleter stand-in; fails the test on unscripted calls."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[tuple[list[dict], str]] = []

    def complete(self, messages: list[dict], *, model: str) -> str:
        self.calls.append((messages, model))
        if not self.script:
            raise AssertionError("unexpected LLM call")
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeSecretStore:
    def __init__(self, record: dict | None) -> None:
        self.record = record

    def get(self, name: str) -> dict | None:
        assert name == "llm"
        return self.record


def _seed_announcement(
    data_root: Path,
    *,
    ts_code: str,
    ann_date: date,
    available_at: date,
    title: str,
    category: str,
    body: bytes | None,
) -> dict:
    """Write one PDF body (skipped when body is None) and return its index row."""

    digest = hashlib.sha256(body if body is not None else b"missing").hexdigest()
    relative = f"announcements/files/{digest[:2]}/{digest}.pdf"
    if body is not None:
        target = data_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    return {
        "ts_code": ts_code,
        "ann_date": pd.Timestamp(ann_date),
        "available_at": pd.Timestamp(available_at),
        "ingested_at": pd.Timestamp("2026-07-17T00:00:00Z"),
        "title": title,
        "url": f"https://static.cninfo.com.cn/{digest[:8]}.pdf",
        "sha256": digest,
        "category": category,
        "file_path": relative,
        "bytes": len(body) if body is not None else 0,
    }


def _write_index(data_root: Path, rows: list[dict]) -> Path:
    directory = data_root / "announcements"
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=list(cninfo.INDEX_COLUMNS))
    path = directory / "index.parquet"
    frame.to_parquet(path, index=False, compression="zstd", engine="pyarrow")
    return path


def test_extract_pdf_text_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(_pdf_bytes(["annual report 2024", "net profit up 30 percent"]))

    text = nlp.extract_pdf_text(path)

    assert "annual report 2024" in text
    assert "net profit up 30 percent" in text


def test_extract_pdf_text_truncates_to_max_chars(tmp_path: Path) -> None:
    path = tmp_path / "long.pdf"
    path.write_bytes(_pdf_bytes(["x" * 200]))

    assert len(nlp.extract_pdf_text(path, max_chars=50)) == 50


def test_extract_pdf_text_fail_closed_on_blank_and_corrupt(tmp_path: Path) -> None:
    blank = tmp_path / "blank.pdf"
    blank.write_bytes(_blank_pdf_bytes())
    with pytest.raises(nlp.PdfTextExtractionError, match="no text layer"):
        nlp.extract_pdf_text(blank)

    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"this is not a pdf at all")
    with pytest.raises(nlp.PdfTextExtractionError, match="cannot extract text"):
        nlp.extract_pdf_text(corrupt)

    with pytest.raises(nlp.PdfTextExtractionError, match="cannot extract text"):
        nlp.extract_pdf_text(tmp_path / "missing.pdf")


def test_parse_extraction_payload_accepts_valid_and_fenced() -> None:
    result = nlp.parse_extraction_payload(_payload())
    assert result.event_type == "periodic_report"
    assert result.tone_score == 0.6
    assert result.key_numbers == {"net_profit_yoy": "+30%"}
    assert result.impact_direction == "positive"
    assert result.impact_horizon == "short_term"
    assert result.impact_channels == ("earnings",)
    assert result.confidence == 0.9

    fenced = f"```json\n{_payload(event_type='dividend', tone_score=-0.25)}\n```"
    fenced_result = nlp.parse_extraction_payload(fenced)
    assert fenced_result.event_type == "dividend"
    assert fenced_result.tone_score == -0.25


def test_parse_extraction_payload_allows_integer_scores() -> None:
    result = nlp.parse_extraction_payload(_payload(tone_score=1, confidence=0))
    assert result.tone_score == 1.0
    assert result.confidence == 0.0


def test_batch_messages_and_parser_require_exact_item_ids() -> None:
    items = [
        nlp.AnnouncementBatchItem(
            item_id="item-a",
            ts_code="000001.SZ",
            ann_date=date(2024, 1, 2),
            title="first",
            category="announcement",
            text="a" * 50,
        ),
        nlp.AnnouncementBatchItem(
            item_id="item-b",
            ts_code="000002.SZ",
            ann_date=date(2024, 1, 3),
            title="second",
            category="regulatory_letter",
            text="b" * 50,
        ),
    ]
    messages = nlp.build_batch_extraction_messages(items, max_chars=10)
    request = json.loads(messages[1]["content"])
    assert [item["item_id"] for item in request["items"]] == ["item-a", "item-b"]
    assert all(len(item["text"]) == 10 for item in request["items"])

    parsed = nlp.parse_batch_extraction_payload(
        _batch_payload(["item-a", "item-b"]),
        expected_item_ids=["item-a", "item-b"],
    )
    assert set(parsed) == {"item-a", "item-b"}
    with pytest.raises(nlp.LlmExtractionError, match="item_id mismatch"):
        nlp.parse_batch_extraction_payload(
            _batch_payload(["item-a"]),
            expected_item_ids=["item-a", "item-b"],
        )


@pytest.mark.parametrize(
    "raw",
    [
        "this is not json",
        "[1, 2, 3]",
        _payload(tone_score=None),
        _payload(event_type="unknown_type"),
        _payload(event_type=42),
        _payload(tone_score=1.5),
        _payload(tone_score=-1.01),
        _payload(tone_score="very positive"),
        _payload(tone_score=True),
        _payload(key_numbers=["net", "profit"]),
        _payload(key_numbers="n/a"),
        _payload(impact_direction="up"),
        _payload(impact_horizon="forever"),
        _payload(impact_channels="earnings"),
        _payload(impact_channels=["earnings", "earnings"]),
        _payload(impact_channels=["unsupported"]),
        _payload(logic_summary=42),
        _payload(logic_summary="x" * (nlp.MAX_LOGIC_SUMMARY_CHARS + 1)),
        _payload(confidence=1.5),
        _payload(confidence=-0.1),
    ],
)
def test_parse_extraction_payload_fail_closed(raw: str) -> None:
    with pytest.raises(nlp.LlmExtractionError) as captured:
        nlp.parse_extraction_payload(raw)
    assert captured.value.stage == "llm_parse"


def test_parse_extraction_payload_requires_all_keys() -> None:
    for missing in (
        "event_type",
        "tone_score",
        "key_numbers",
        "impact_direction",
        "impact_horizon",
        "impact_channels",
        "logic_summary",
        "confidence",
    ):
        body = json.loads(_payload())
        del body[missing]
        with pytest.raises(nlp.LlmExtractionError, match="misses keys"):
            nlp.parse_extraction_payload(json.dumps(body))


def test_load_llm_credentials_prefers_secret_store() -> None:
    credentials = nlp.load_llm_credentials(FakeSecretStore(STORE_RECORD), environ={})
    assert credentials.api_key == "sk-test-fake-key"
    assert credentials.api_base == "https://llm.example.invalid/v1"
    assert credentials.chat_model == "test-model"
    assert credentials.source == "runtime_secret_store"


def test_load_llm_credentials_falls_back_to_environment() -> None:
    environ = {
        "OPENAI_API_KEY": "sk-env-fake",
        "OPENAI_API_BASE": "https://env.example.invalid/v1/",
        "CHAT_MODEL": "env-model",
    }
    credentials = nlp.load_llm_credentials(FakeSecretStore(None), environ=environ)
    assert credentials.api_key == "sk-env-fake"
    assert credentials.api_base == "https://env.example.invalid/v1"
    assert credentials.chat_model == "env-model"
    assert credentials.source == "environment"


def test_load_llm_credentials_missing_raises_with_config_paths() -> None:
    with pytest.raises(nlp.LlmCredentialsError) as captured:
        nlp.load_llm_credentials(FakeSecretStore(None), environ={})
    message = str(captured.value)
    assert "POST /api/settings/llm" in message
    assert "OPENAI_API_KEY" in message

    with pytest.raises(nlp.LlmCredentialsError):
        nlp.load_llm_credentials(FakeSecretStore({"api_key": "  "}), environ={})


class _FakeChatResponse:
    def __init__(self, status_code: int, body) -> None:
        self.status_code = status_code
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _FakeChatSession:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if not self.outcomes:
            raise AssertionError(f"unexpected LLM HTTP call: {url}")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _chat_body(content: str, *, usage: dict | None = None) -> dict:
    body = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return body


def _openai_client(session: _FakeChatSession, delays: list | None = None):
    sleeper = (lambda seconds: delays.append(seconds)) if delays is not None else (lambda s: None)
    return nlp.OpenAIChatClient(
        nlp.LlmCredentials(
            api_key="sk-test-fake-key",
            api_base="https://llm.example.invalid/v1",
            chat_model="test-model",
            source="test",
        ),
        rate_gate=GlobalRateGate(60_000),
        session=session,
        max_attempts=3,
        sleeper=sleeper,
    )


def test_openai_chat_client_posts_with_auth_header() -> None:
    session = _FakeChatSession([_FakeChatResponse(200, _chat_body(_payload()))])
    client = _openai_client(session)

    content = client.complete([{"role": "user", "content": "hi"}], model="test-model")

    assert json.loads(content)["event_type"] == "periodic_report"
    call = session.calls[0]
    assert call["url"] == "https://llm.example.invalid/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer sk-test-fake-key"
    assert call["json"]["model"] == "test-model"
    assert call["json"]["response_format"] == {"type": "json_object"}


def test_deepseek_v4_disables_thinking_and_records_usage() -> None:
    session = _FakeChatSession(
        [
            _FakeChatResponse(
                200,
                _chat_body(
                    _payload(),
                    usage={
                        "prompt_tokens": 120,
                        "completion_tokens": 30,
                        "total_tokens": 150,
                        "prompt_cache_hit_tokens": 20,
                        "prompt_cache_miss_tokens": 100,
                    },
                ),
            )
        ]
    )
    client = nlp.OpenAIChatClient(
        nlp.LlmCredentials(
            api_key="sk-test-fake-key",
            api_base="https://api.deepseek.com",
            chat_model="deepseek-v4-flash",
            source="test",
        ),
        rate_gate=GlobalRateGate(60_000),
        session=session,
        sleeper=lambda _seconds: None,
    )

    client.complete([], model="deepseek-v4-flash")

    assert session.calls[0]["json"]["thinking"] == {"type": "disabled"}
    assert client.usage_totals() == {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "prompt_cache_hit_tokens": 20,
        "prompt_cache_miss_tokens": 100,
    }


def test_openai_chat_client_retries_retryable_status() -> None:
    session = _FakeChatSession(
        [
            _FakeChatResponse(500, {}),
            _FakeChatResponse(200, _chat_body(_payload())),
        ]
    )
    delays: list[float] = []

    content = _openai_client(session, delays).complete([], model="test-model")

    assert json.loads(content)["confidence"] == 0.9
    assert len(session.calls) == 2
    assert len(delays) == 1


def test_openai_chat_client_fail_closed_on_client_error_without_leaking_key() -> None:
    session = _FakeChatSession([_FakeChatResponse(400, {})])
    with pytest.raises(nlp.LlmExtractionError) as captured:
        _openai_client(session).complete([], model="test-model")
    assert captured.value.stage == "llm_call"
    assert "sk-test-fake-key" not in str(captured.value)
    assert len(session.calls) == 1


def test_openai_chat_client_rejects_malformed_success_body() -> None:
    session = _FakeChatSession([_FakeChatResponse(200, {"unexpected": True})])
    with pytest.raises(nlp.LlmExtractionError) as captured:
        _openai_client(session).complete([], model="test-model")
    assert captured.value.stage == "llm_call"


def test_openai_chat_client_wraps_connection_errors() -> None:
    session = _FakeChatSession([requests.ConnectionError("refused")] * 3)
    with pytest.raises(nlp.LlmExtractionError) as captured:
        _openai_client(session, []).complete([], model="test-model")
    assert captured.value.stage == "llm_call"
    assert "sk-test-fake-key" not in str(captured.value)
    assert len(session.calls) == 3


def _seed_two_announcements(data_root: Path) -> list[dict]:
    rows = [
        _seed_announcement(
            data_root,
            ts_code="000001.SZ",
            ann_date=date(2024, 1, 2),
            available_at=date(2024, 1, 3),
            title="2023年年度报告",
            category="announcement",
            body=_pdf_bytes(["annual report", "net profit up 30 percent"]),
        ),
        _seed_announcement(
            data_root,
            ts_code="000002.SZ",
            ann_date=date(2024, 1, 3),
            available_at=date(2024, 1, 4),
            title="关于对公司的问询函",
            category="regulatory_letter",
            body=_pdf_bytes(["regulatory inquiry letter", "please explain the loss"]),
        ),
    ]
    _write_index(data_root, rows)
    return rows


def _run(data_root: Path, chat, **kwargs):
    kwargs.setdefault("batch_size", 1)
    kwargs.setdefault("workers", 1)
    return nlp.process_announcements(
        data_root,
        secret_store=FakeSecretStore(STORE_RECORD),
        chat_client=chat,
        now=lambda: NOW,
        environ={},
        **kwargs,
    )


def test_process_end_to_end(tmp_path: Path) -> None:
    rows = _seed_two_announcements(tmp_path)
    chat = FakeChatClient(
        [
            _payload(event_type="periodic_report", tone_score=0.6, confidence=0.9),
            _payload(
                event_type="regulatory_letter", tone_score=-0.7, key_numbers={}, confidence=0.8
            ),
        ]
    )

    summary = _run(tmp_path, chat)

    assert summary.planned == 2
    assert summary.processed == 2
    assert summary.failed == 0
    assert summary.skipped == 0
    assert len(chat.calls) == 2
    assert chat.calls[0][1] == "test-model"
    assert nlp.PROMPT_VERSION in chat.calls[0][0][0]["content"]
    assert "net profit up 30 percent" in chat.calls[0][0][1]["content"]

    fields = pd.read_parquet(summary.fields_path)
    assert fields["ts_code"].tolist() == ["000001.SZ", "000002.SZ"]
    assert fields["event_type"].tolist() == ["periodic_report", "regulatory_letter"]
    assert fields["tone_score"].tolist() == [0.6, -0.7]
    first = fields.iloc[0]
    assert first["source_sha256"] == rows[0]["sha256"]
    assert first["model"] == "test-model"
    assert first["prompt_version"] == nlp.PROMPT_VERSION
    assert first["ingested_at"] == pd.Timestamp(NOW)
    assert first["processed_at"] == pd.Timestamp(NOW)
    assert first["available_at"] == pd.Timestamp(date(2024, 1, 3))
    assert json.loads(first["key_numbers"]) == {"net_profit_yoy": "+30%"}
    assert first["impact_direction"] == "positive"
    assert first["impact_horizon"] == "short_term"
    assert json.loads(first["impact_channels"]) == ["earnings"]
    assert json.loads(fields.iloc[1]["key_numbers"]) == {}

    state = pd.read_parquet(summary.state_path)
    assert state["status"].tolist() == ["succeeded", "succeeded"]
    assert set(state["stage"]) == {"completed"}
    assert state["processed_at"].notna().all()

    unit = pd.read_parquet(summary.unit_path)
    assert len(unit) == 2

    factor_path = tmp_path / "announcements/nlp/factors/announcement_tone.parquet"
    factor = pd.read_parquet(factor_path)
    assert list(factor.columns) == ["datetime", "instrument", "announcement_tone"]
    assert factor["datetime"].tolist() == [
        pd.Timestamp(date(2024, 1, 3)),
        pd.Timestamp(date(2024, 1, 4)),
    ]
    assert factor["instrument"].tolist() == ["000001.SZ", "000002.SZ"]
    assert factor["announcement_tone"].tolist() == [0.6, -0.7]

    manifest = json.loads(
        (tmp_path / "announcements/nlp/factors/announcement_tone.json").read_text(encoding="utf-8")
    )
    assert manifest["factor"] == "announcement_tone"
    assert manifest["rows"] == 2
    assert manifest["sha256"] == summary.factor_sha256
    assert manifest["sha256"] == hashlib.sha256(factor_path.read_bytes()).hexdigest()
    assert manifest["availability_policy"] == {
        "announcement_tone": "available_at_first_trading_day_after_announcement"
    }
    assert manifest["source"]["prompt_version"] == nlp.PROMPT_VERSION
    assert manifest["source"]["model"] == "test-model"

    logic_path = tmp_path / "announcements/nlp/factors/announcement_logic_score.parquet"
    logic = pd.read_parquet(logic_path)
    assert logic["announcement_logic_score"].tolist() == [
        pytest.approx(0.765),
        pytest.approx(0.68),
    ]
    assert summary.logic_factor_rows == 2
    assert summary.logic_factor_sha256 == hashlib.sha256(logic_path.read_bytes()).hexdigest()


def test_process_deduplicates_identical_pdf_references_before_batching(tmp_path: Path) -> None:
    body = _pdf_bytes(["same regulatory letter", "explain the material loss"])
    first = _seed_announcement(
        tmp_path,
        ts_code="000001.SZ",
        ann_date=date(2024, 1, 2),
        available_at=date(2024, 1, 3),
        title="Regulatory letter original URL",
        category="regulatory_letter",
        body=body,
    )
    duplicate = {
        **first,
        "title": "Regulatory letter duplicate URL",
        "url": "https://static.cninfo.com.cn/duplicate-reference.pdf",
    }
    _write_index(tmp_path, [first, duplicate])
    item_id = f"{first['sha256']}:{nlp.PROMPT_VERSION}:test-model"
    chat = FakeChatClient([_batch_payload([item_id])])

    summary = _run(tmp_path, chat, batch_size=8)

    assert summary.planned == 1
    assert summary.processed == 1
    assert summary.llm_calls == 1
    assert len(chat.calls) == 1
    assert len(pd.read_parquet(summary.fields_path)) == 1


def test_process_batches_announcements_and_falls_back_per_item(tmp_path: Path) -> None:
    rows = _seed_two_announcements(tmp_path)
    item_ids = [f"{row['sha256']}:{nlp.PROMPT_VERSION}:test-model" for row in rows]
    batched = FakeChatClient([_batch_payload(item_ids)])

    summary = _run(tmp_path, batched, batch_size=2, workers=2)

    assert summary.processed == 2
    assert summary.failed == 0
    assert summary.llm_calls == 1
    assert len(batched.calls) == 1

    retry_root = tmp_path / "fallback"
    retry_rows = _seed_two_announcements(retry_root)
    retry_ids = [f"{row['sha256']}:{nlp.PROMPT_VERSION}:test-model" for row in retry_rows]
    fallback = FakeChatClient(
        [
            _batch_payload(retry_ids[:1]),
            _payload(tone_score=0.2),
            _payload(tone_score=-0.2),
        ]
    )

    recovered = _run(retry_root, fallback, batch_size=2, workers=1)

    assert recovered.processed == 2
    assert recovered.failed == 0
    assert recovered.llm_calls == 3
    assert len(fallback.calls) == 3


def test_process_is_idempotent_on_processing_key(tmp_path: Path) -> None:
    _seed_two_announcements(tmp_path)
    first = _run(tmp_path, FakeChatClient([_payload(), _payload(tone_score=-0.5)]))
    assert first.processed == 2

    chat = FakeChatClient([])
    second = _run(tmp_path, chat)

    assert second.planned == 2
    assert second.skipped == 2
    assert second.processed == 0
    assert second.failed == 0
    assert second.unit_path is None
    assert chat.calls == []
    assert len(pd.read_parquet(second.fields_path)) == 2
    assert second.factor_rows == 2


def test_process_reprocesses_when_prompt_version_changes(tmp_path: Path, monkeypatch) -> None:
    _seed_two_announcements(tmp_path)
    _run(tmp_path, FakeChatClient([_payload(), _payload()]))

    monkeypatch.setattr(nlp, "PROMPT_VERSION", "announcement-nlp.v4")
    chat = FakeChatClient([_payload(tone_score=0.1), _payload(tone_score=0.2)])
    summary = _run(tmp_path, chat)

    assert summary.processed == 2
    assert summary.skipped == 0
    fields = pd.read_parquet(summary.fields_path)
    assert len(fields) == 4  # Both governed prompt versions remain auditable.
    assert set(fields["prompt_version"]) == {"announcement-nlp.v3", "announcement-nlp.v4"}
    # Only the current prompt/model generation is published into the factor.
    assert summary.factor_rows == 2


def test_process_records_pdf_source_gap_and_continues_fail_closed(tmp_path: Path) -> None:
    rows = [
        _seed_announcement(
            tmp_path,
            ts_code="000001.SZ",
            ann_date=date(2024, 1, 2),
            available_at=date(2024, 1, 3),
            title="扫描件公告",
            category="announcement",
            body=b"this is not a pdf at all",
        ),
        _seed_announcement(
            tmp_path,
            ts_code="000002.SZ",
            ann_date=date(2024, 1, 3),
            available_at=date(2024, 1, 4),
            title="年度报告",
            category="announcement",
            body=_pdf_bytes(["annual report text"]),
        ),
    ]
    _write_index(tmp_path, rows)
    chat = FakeChatClient([_payload(tone_score=0.3)])

    summary = _run(tmp_path, chat)

    assert summary.planned == 2
    assert summary.processed == 1
    assert summary.unavailable == 1
    assert summary.failed == 0
    assert summary.as_dict()["status"] == "succeeded_with_source_gaps"
    assert len(chat.calls) == 1  # the corrupt PDF never reaches the LLM

    fields = pd.read_parquet(summary.fields_path)
    assert fields["ts_code"].tolist() == ["000002.SZ"]

    state = pd.read_parquet(summary.state_path)
    source_gap = state[state["status"] == "source_unavailable"].iloc[0]
    assert source_gap["stage"] == "pdf_extract"
    assert "cannot extract text" in source_gap["error"]
    assert source_gap["ts_code"] == "000001.SZ"

    factor = pd.read_parquet(tmp_path / "announcements/nlp/factors/announcement_tone.parquet")
    assert factor["instrument"].tolist() == ["000002.SZ"]

    rerun = _run(tmp_path, FakeChatClient([]))
    assert rerun.processed == 0
    assert rerun.skipped == 1
    assert rerun.unavailable == 1
    assert rerun.failed == 0


def test_process_records_llm_failures_without_signals(tmp_path: Path) -> None:
    _seed_two_announcements(tmp_path)
    chat = FakeChatClient(
        [
            "definitely not json",
            nlp.LlmExtractionError("LLM endpoint returned HTTP 500", stage="llm_call"),
        ]
    )

    summary = _run(tmp_path, chat)

    assert summary.processed == 0
    assert summary.failed == 2
    assert summary.as_dict()["status"] == "failed"
    state = pd.read_parquet(summary.state_path)
    assert set(state["status"]) == {"failed"}
    assert set(state["stage"]) == {"llm_parse", "llm_call"}

    fields = pd.read_parquet(summary.fields_path)
    assert fields.empty
    assert summary.factor_rows == 0
    manifest = json.loads(
        (tmp_path / "announcements/nlp/factors/announcement_tone.json").read_text(encoding="utf-8")
    )
    assert manifest["rows"] == 0


def test_process_checkpoints_and_resumes_after_interruption(tmp_path: Path) -> None:
    _seed_two_announcements(tmp_path)
    progress: list[dict[str, int]] = []

    with pytest.raises(RuntimeError, match="worker interrupted"):
        _run(
            tmp_path,
            FakeChatClient([_payload(tone_score=0.1), RuntimeError("worker interrupted")]),
            checkpoint_every=1,
            progress_callback=progress.append,
        )

    assert len(list((tmp_path / "announcements/nlp/units").glob("*.parquet"))) == 1
    assert len(pd.read_parquet(tmp_path / "announcements/nlp/fields.parquet")) == 1
    assert progress[-1]["completed"] == 1

    resumed_progress: list[dict[str, int]] = []
    resumed = _run(
        tmp_path,
        FakeChatClient([_payload(tone_score=0.2)]),
        checkpoint_every=1,
        progress_callback=resumed_progress.append,
    )

    assert resumed.skipped == 1
    assert resumed.processed == 1
    assert resumed.failed == 0
    assert len(pd.read_parquet(resumed.fields_path)) == 2
    assert len(list((tmp_path / "announcements/nlp/units").glob("*.parquet"))) == 2
    assert resumed_progress[-1] == {
        "planned": 2,
        "completed": 2,
        "processed": 1,
        "skipped": 1,
        "unavailable": 0,
        "failed": 0,
        "llm_calls": 1,
    }


def test_process_reuses_paid_success_after_model_switch(tmp_path: Path) -> None:
    _seed_two_announcements(tmp_path)
    pro = nlp.LlmCredentials(
        api_key="sk-test-fake-key",
        api_base="https://api.deepseek.com",
        chat_model="deepseek-v4-pro",
        source="test",
    )
    flash = nlp.LlmCredentials(
        api_key="sk-test-fake-key",
        api_base="https://api.deepseek.com",
        chat_model="deepseek-v4-flash",
        source="test",
    )
    _run(
        tmp_path,
        FakeChatClient([_payload(tone_score=0.2), _payload(tone_score=-0.4)]),
        credentials=pro,
    )

    resumed = _run(tmp_path, FakeChatClient([]), credentials=flash)

    assert resumed.processed == 0
    assert resumed.skipped == 2
    fields = pd.read_parquet(resumed.fields_path)
    assert len(fields) == 2
    assert set(fields["model"]) == {"deepseek-v4-pro"}
    manifest = json.loads(resumed.factor_manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["model"] == "deepseek-v4-pro"
    assert manifest["source"]["scope"]["requested_model"] == "deepseek-v4-flash"
    assert manifest["source"]["scope"]["models"] == ["deepseek-v4-pro"]


def test_process_publishes_mixed_model_scope_without_duplicate_calls(tmp_path: Path) -> None:
    _seed_two_announcements(tmp_path)
    pro = nlp.LlmCredentials(
        api_key="sk-test-fake-key",
        api_base="https://api.deepseek.com",
        chat_model="deepseek-v4-pro",
        source="test",
    )
    flash = nlp.LlmCredentials(
        api_key="sk-test-fake-key",
        api_base="https://api.deepseek.com",
        chat_model="deepseek-v4-flash",
        source="test",
    )
    with pytest.raises(RuntimeError, match="worker interrupted"):
        _run(
            tmp_path,
            FakeChatClient([_payload(tone_score=0.2), RuntimeError("worker interrupted")]),
            credentials=pro,
            checkpoint_every=1,
        )

    resumed = _run(
        tmp_path,
        FakeChatClient([_payload(tone_score=-0.4)]),
        credentials=flash,
        checkpoint_every=1,
    )

    assert resumed.skipped == 1
    assert resumed.processed == 1
    fields = pd.read_parquet(resumed.fields_path)
    assert len(fields) == 2
    assert set(fields["model"]) == {"deepseek-v4-pro", "deepseek-v4-flash"}
    factor = pd.read_parquet(resumed.factor_manifest_path.with_suffix(".parquet"))
    assert len(factor) == 2
    manifest = json.loads(resumed.factor_manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["model"] == "mixed[deepseek-v4-flash,deepseek-v4-pro]"
    assert manifest["source"]["scope"]["models"] == [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ]


def test_process_retries_failed_rows_on_rerun(tmp_path: Path) -> None:
    _seed_two_announcements(tmp_path)
    failing = FakeChatClient(
        [
            nlp.LlmExtractionError("LLM request failed: boom", stage="llm_call"),
            _payload(tone_score=0.2),
        ]
    )
    first = _run(tmp_path, failing)
    assert first.failed == 1
    assert first.processed == 1

    chat = FakeChatClient([_payload(tone_score=0.4)])
    second = _run(tmp_path, chat)

    assert second.planned == 2
    assert second.skipped == 1  # the previously succeeded row is untouched
    assert second.processed == 1
    assert second.failed == 0
    state = pd.read_parquet(second.state_path)
    assert set(state["status"]) == {"succeeded"}
    assert len(pd.read_parquet(second.fields_path)) == 2


def test_process_missing_index_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="cninfo-announcements"):
        _run(tmp_path, FakeChatClient([]))


def test_process_requires_real_credentials(tmp_path: Path) -> None:
    _seed_two_announcements(tmp_path)
    chat = FakeChatClient([])

    with pytest.raises(nlp.LlmCredentialsError, match="POST /api/settings/llm"):
        nlp.process_announcements(
            tmp_path,
            secret_store=FakeSecretStore(None),
            chat_client=chat,
            now=lambda: NOW,
            environ={},
        )

    assert chat.calls == []
    assert not (tmp_path / "announcements/nlp/state.parquet").exists()
    assert not (tmp_path / "announcements/nlp/fields.parquet").exists()


def test_process_filters_ts_code_dates_and_category(tmp_path: Path) -> None:
    _seed_two_announcements(tmp_path)
    chat = FakeChatClient([_payload(event_type="regulatory_letter", tone_score=-0.5)])

    summary = _run(tmp_path, chat, categories={"regulatory_letter"})

    assert summary.planned == 1
    assert summary.processed == 1
    fields = pd.read_parquet(summary.fields_path)
    assert fields["ts_code"].tolist() == ["000002.SZ"]

    assert _run(tmp_path, FakeChatClient([]), ts_codes={"999999.SZ"}).planned == 0
    assert (
        _run(
            tmp_path,
            FakeChatClient([]),
            start=date(2024, 2, 1),
            end=date(2024, 2, 29),
        ).planned
        == 0
    )
    limited = _run(tmp_path, FakeChatClient([_payload(tone_score=0.1)]), limit=1)
    assert limited.planned == 1
    assert limited.processed == 1


def test_factor_publication_is_restricted_to_the_requested_scope(tmp_path: Path) -> None:
    _seed_two_announcements(tmp_path)
    _run(tmp_path, FakeChatClient([_payload(), _payload(tone_score=-0.5)]))

    scoped = _run(
        tmp_path,
        FakeChatClient([]),
        categories={"regulatory_letter"},
    )

    assert scoped.planned == 1
    assert scoped.skipped == 1
    factor = pd.read_parquet(tmp_path / "announcements/nlp/factors/announcement_tone.parquet")
    assert factor["instrument"].tolist() == ["000002.SZ"]
    manifest = json.loads(
        (tmp_path / "announcements/nlp/factors/announcement_tone.json").read_text(encoding="utf-8")
    )
    assert manifest["source"]["scope"]["categories"] == ["regulatory_letter"]
    assert manifest["source"]["scope_process_key_count"] == 1


def test_factor_artifact_averages_same_day_instruments(tmp_path: Path) -> None:
    rows = [
        _seed_announcement(
            tmp_path,
            ts_code="000001.SZ",
            ann_date=date(2024, 1, 2),
            available_at=date(2024, 1, 3),
            title="业绩预告",
            category="announcement",
            body=_pdf_bytes(["earnings forecast", "profit rising"]),
        ),
        _seed_announcement(
            tmp_path,
            ts_code="000001.SZ",
            ann_date=date(2024, 1, 2),
            available_at=date(2024, 1, 3),
            title="补充公告",
            category="announcement",
            body=_pdf_bytes(["supplementary announcement"]),
        ),
    ]
    _write_index(tmp_path, rows)
    chat = FakeChatClient([_payload(tone_score=0.8), _payload(tone_score=0.4)])

    summary = _run(tmp_path, chat)

    assert summary.factor_rows == 1
    factor = pd.read_parquet(tmp_path / "announcements/nlp/factors/announcement_tone.parquet")
    assert factor["announcement_tone"].tolist() == [pytest.approx(0.6)]
    assert factor["datetime"].tolist() == [pd.Timestamp(date(2024, 1, 3))]


def test_cli_runs_with_injected_fakes(tmp_path: Path, monkeypatch) -> None:
    _seed_two_announcements(tmp_path)
    chat = FakeChatClient([_payload(event_type="regulatory_letter", tone_score=-0.5)])
    monkeypatch.setattr(
        cli_module, "RuntimeSecretStore", lambda *args, **kwargs: FakeSecretStore(STORE_RECORD)
    )
    monkeypatch.setattr(nlp, "OpenAIChatClient", lambda *args, **kwargs: chat)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = CliRunner().invoke(
        app,
        [
            "announcement-nlp",
            "--ts-code",
            "000002.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--category",
            "regulatory_letter",
            "--limit",
            "5",
            "--batch-size",
            "1",
            "--workers",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dataset"] == "announcement_nlp"
    assert payload["planned"] == 1
    assert payload["processed"] == 1
    assert payload["failed"] == 0
    assert payload["factor_rows"] == 1
    assert len(chat.calls) == 1
    assert Path(payload["fields_path"]).is_file()


def test_cli_exits_nonzero_when_an_llm_row_fails(tmp_path: Path, monkeypatch) -> None:
    _seed_two_announcements(tmp_path)
    monkeypatch.setattr(
        cli_module, "RuntimeSecretStore", lambda *args, **kwargs: FakeSecretStore(STORE_RECORD)
    )
    monkeypatch.setattr(
        nlp, "OpenAIChatClient", lambda *args, **kwargs: FakeChatClient(["not-json"])
    )
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    result = CliRunner().invoke(
        app,
        [
            "announcement-nlp",
            "--category",
            "regulatory_letter",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--batch-size",
            "1",
            "--workers",
            "1",
        ],
    )

    assert result.exit_code == 3
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["failed"] == 1


def test_cli_rejects_unknown_category(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    result = CliRunner().invoke(app, ["announcement-nlp", "--category", "bogus"])

    assert result.exit_code != 0
