from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import quant_data.cli as cli_module
from quant_data.cli import app
from quant_platform import corpus_nlp as corpus
from quant_platform.announcement_nlp import LlmCredentialsError, LlmExtractionError

pytestmark = pytest.mark.no_database

NOW = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)
STORE_RECORD = {
    "api_key": "sk-test-fake-key",
    "api_base": "https://llm.example.invalid/v1",
    "chat_model": "test-model",
}
# Trading calendar with a mid-week holiday (2024-01-03) so tests prove the
# persisted trade_cal — not weekday rules — drives availability and factor dates.
OPEN_DAYS = [
    date(2024, 1, 2),
    date(2024, 1, 4),
    date(2024, 1, 5),
    date(2024, 1, 8),
    date(2024, 1, 9),
]
CALENDAR_DAYS = [
    (date(2024, 1, 2), 1),
    (date(2024, 1, 3), 0),  # Wednesday holiday
    (date(2024, 1, 4), 1),
    (date(2024, 1, 5), 1),
    (date(2024, 1, 6), 0),
    (date(2024, 1, 7), 0),
    (date(2024, 1, 8), 1),
    (date(2024, 1, 9), 1),
]


def _payload(**overrides) -> str:
    body = {"sentiment": 0.6, "topic": "company", "confidence": 0.9}
    body.update(overrides)
    return json.dumps(body, ensure_ascii=False)


def _batch_payload(items, **overrides) -> str:
    return json.dumps(
        {
            "items": [
                {
                    "item_id": item.item_id,
                    "sentiment": overrides.get("sentiment", 0.6),
                    "topic": overrides.get("topic", "company"),
                    "confidence": overrides.get("confidence", 0.9),
                }
                for item in items
            ]
        },
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


def _write_parquet(directory: Path, rows: list[dict], name: str = "data.parquet") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / name, index=False)


def _seed_trade_cal(data_root: Path) -> None:
    _write_parquet(
        data_root / "units" / "trade_cal",
        [
            {"cal_date": day.strftime("%Y%m%d"), "is_open": flag, "pretrade_date": ""}
            for day, flag in CALENDAR_DAYS
        ],
    )


def _news_row(title: str, content: str, pub_time: str, src: str = "财联社") -> dict:
    return {"title": title, "content": content, "pub_time": pub_time, "src": src}


def _seed_corpus(data_root: Path) -> None:
    """Seed the three datasets across both units and snapshots layouts."""
    _write_parquet(
        data_root / "units" / "major_news",
        [
            _news_row("央行降准释放流动性", "正文：降准0.5个百分点", "2024-01-02 10:00:00"),
            _news_row("市场情绪受挫", "正文：两市午后跳水", "2024-01-02 16:30:00"),
        ],
    )
    _write_parquet(
        data_root / "snapshots" / "snap1" / "parquet" / "major_news",
        [
            # Exact duplicate of a units row: must collapse onto one item.
            _news_row("央行降准释放流动性", "正文：降准0.5个百分点", "2024-01-02 10:00:00"),
            _news_row("周末宏观综述", "正文：假期政策汇总", "2024-01-06 09:00:00"),
        ],
    )
    _write_parquet(
        data_root / "units" / "irm_qa_sh",
        [
            {
                "trade_date": "20240102",
                "ts_code": "000001.SH",
                "q": "公司产能利用率如何？",
                "a": "产能利用率维持在90%以上。",
            }
        ],
    )
    _write_parquet(
        data_root / "units" / "irm_qa_sz",
        [
            {"trade_date": "20240105", "ts_code": "000002.SZ", "q": "新业务进展如何？"},
        ],
    )
    _write_parquet(
        data_root / "units" / "npr",
        [
            {
                "title": "国务院关于促进资本市场健康发展的意见",
                "pubtime": "2024-01-04 09:00:00",
                "pcode": "国发〔2024〕1号",
                "puborg": "国务院",
                "ptype": "金融",
            }
        ],
    )
    _write_parquet(
        data_root / "units" / "cctv_news",
        [
            {
                "date": "20240105",
                "title": "国务院常务会议部署稳增长政策",
                "content": "会议指出，要加大宏观政策调控力度。",
            }
        ],
    )


def _run(data_root: Path, chat, **kwargs) -> corpus.CorpusNlpSummary:
    kwargs.setdefault("datasets", set(corpus.SUPPORTED_CORPUS_DATASETS))
    return corpus.process_corpus(
        data_root,
        secret_store=FakeSecretStore(STORE_RECORD),
        chat_client=chat,
        now=lambda: NOW,
        environ={},
        **kwargs,
    )


# --- corpus loading ----------------------------------------------------------


def test_load_corpus_items_reads_units_and_snapshots(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)

    items = corpus.load_corpus_items(tmp_path)

    assert len(items) == 7  # the duplicated news row collapses onto one item
    by_dataset = {}
    for item in items:
        by_dataset.setdefault(item.source_dataset, []).append(item)
    assert set(by_dataset) == {"major_news", "irm_qa_sh", "irm_qa_sz", "npr", "cctv_news"}
    assert len(by_dataset["major_news"]) == 3
    npr = by_dataset["npr"][0]
    assert npr.ts_code is None
    assert npr.pub_time == datetime(2024, 1, 4, 9, 0, 0)
    # Title-level fallback: no content column in the seeded npr parquet.
    assert npr.content == npr.title
    cctv = by_dataset["cctv_news"][0]
    assert cctv.pub_time == datetime(2024, 1, 5, 0, 0, 0)
    news = by_dataset["major_news"][0]
    assert news.ts_code is None
    assert news.pub_time == datetime(2024, 1, 2, 10, 0, 0)
    assert len(news.item_id) == 64  # content sha256 hexdigest
    sh = by_dataset["irm_qa_sh"][0]
    assert sh.ts_code == "000001.SH"
    assert sh.content == "问：公司产能利用率如何？\n答：产能利用率维持在90%以上。"
    assert sh.pub_time == datetime(2024, 1, 2, 0, 0, 0)


def test_load_corpus_items_missing_dataset_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError) as captured:
        corpus.load_corpus_items(tmp_path)
    message = str(captured.value)
    for dataset in ("major_news", "irm_qa_sh", "irm_qa_sz", "npr", "cctv_news"):
        assert dataset in message
    assert str(tmp_path / "units" / "major_news") in message
    assert "snapshots" in message

    _write_parquet(tmp_path / "units" / "irm_qa_sz", [{"trade_date": "20240102"}])
    with pytest.raises(RuntimeError, match="irm_qa_sz"):
        corpus.load_corpus_items(tmp_path, datasets={"irm_qa_sz"})


def test_load_corpus_items_rejects_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported corpus dataset"):
        corpus.load_corpus_items(tmp_path, datasets={"ths_hot"})


def test_load_irm_qa_tolerates_column_variants(tmp_path: Path) -> None:
    # question/answer naming instead of q, and no answer column for irm_qa_sz.
    _write_parquet(
        tmp_path / "units" / "irm_qa_sh",
        [
            {
                "trade_date": "2024-01-02",
                "ts_code": "600000.SH",
                "question": "分红计划？",
                "answer": "拟10派3元。",
            }
        ],
    )
    _write_parquet(
        tmp_path / "units" / "irm_qa_sz",
        [{"trade_date": "20240103", "ts_code": "000002.SZ", "q": "订单情况？"}],
    )

    items = corpus.load_corpus_items(tmp_path, datasets={"irm_qa_sh", "irm_qa_sz"})

    assert len(items) == 2
    sh = next(item for item in items if item.source_dataset == "irm_qa_sh")
    assert sh.content == "问：分红计划？\n答：拟10派3元。"
    sz = next(item for item in items if item.source_dataset == "irm_qa_sz")
    assert sz.content == "问：订单情况？"  # missing answer column: question only


def test_load_irm_qa_missing_question_column_fail_closed(tmp_path: Path) -> None:
    _write_parquet(
        tmp_path / "units" / "irm_qa_sh",
        [{"trade_date": "20240102", "ts_code": "600000.SH", "note": "x"}],
    )

    with pytest.raises(RuntimeError, match="q\\|question"):
        corpus.load_corpus_items(tmp_path, datasets={"irm_qa_sh"})


def test_load_major_news_missing_content_column_fail_closed(tmp_path: Path) -> None:
    _write_parquet(
        tmp_path / "units" / "major_news",
        [{"title": "t", "pub_time": "2024-01-02 10:00:00"}],
    )

    with pytest.raises(RuntimeError, match="content"):
        corpus.load_corpus_items(tmp_path, datasets={"major_news"})


def test_load_major_news_drops_unusable_rows(tmp_path: Path) -> None:
    _write_parquet(
        tmp_path / "units" / "major_news",
        [
            _news_row("有效", "正文", "2024-01-02 10:00:00"),
            _news_row("时间无法解析", "正文", "not-a-timestamp"),
            _news_row("", "", "2024-01-02 11:00:00"),
        ],
    )

    items = corpus.load_corpus_items(tmp_path, datasets={"major_news"})

    assert [item.title for item in items] == ["有效"]


# --- availability and factor-date rules --------------------------------------


def test_available_at_rules_per_dataset() -> None:
    news = corpus.CorpusItem(
        source_dataset="major_news",
        item_id="n",
        ts_code=None,
        title="t",
        content="c",
        pub_time=datetime(2024, 1, 2, 16, 30, 0),
    )
    assert corpus.available_at_for(news, OPEN_DAYS) == datetime(2024, 1, 2, 16, 30, 0)

    irm = corpus.CorpusItem(
        source_dataset="irm_qa_sh",
        item_id="i",
        ts_code="600000.SH",
        title="q",
        content="c",
        pub_time=datetime(2024, 1, 2, 0, 0, 0),
    )
    # Date-only source: next trading day, skipping the Wednesday holiday.
    assert corpus.available_at_for(irm, OPEN_DAYS) == datetime(2024, 1, 4, 0, 0, 0)

    npr = corpus.CorpusItem(
        source_dataset="npr",
        item_id="p",
        ts_code=None,
        title="t",
        content="c",
        pub_time=datetime(2024, 1, 2, 17, 30, 0),
    )
    # npr carries an exact publication moment, like major_news.
    assert corpus.available_at_for(npr, OPEN_DAYS) == datetime(2024, 1, 2, 17, 30, 0)

    cctv = corpus.CorpusItem(
        source_dataset="cctv_news",
        item_id="c",
        ts_code=None,
        title="t",
        content="c",
        pub_time=datetime(2024, 1, 2, 0, 0, 0),
    )
    # Date-only source: next trading day, skipping the Wednesday holiday.
    assert corpus.available_at_for(cctv, OPEN_DAYS) == datetime(2024, 1, 4, 0, 0, 0)

    late = corpus.CorpusItem(
        source_dataset="irm_qa_sz",
        item_id="j",
        ts_code="000002.SZ",
        title="q",
        content="c",
        pub_time=datetime(2024, 1, 9, 0, 0, 0),
    )
    with pytest.raises(LookupError, match="no trading day after"):
        corpus.available_at_for(late, OPEN_DAYS)


def test_factor_date_rule_uses_trade_cal_not_weekdays() -> None:
    assert corpus.factor_date_for(datetime(2024, 1, 2, 14, 59), OPEN_DAYS) == date(2024, 1, 2)
    # Exactly 15:00 counts as "after": next trading day, skipping the holiday.
    assert corpus.factor_date_for(datetime(2024, 1, 2, 15, 0), OPEN_DAYS) == date(2024, 1, 4)
    assert corpus.factor_date_for(datetime(2024, 1, 2, 16, 30), OPEN_DAYS) == date(2024, 1, 4)
    # Wednesday holiday morning: before 15:00 but not a trading day.
    assert corpus.factor_date_for(datetime(2024, 1, 3, 9, 0), OPEN_DAYS) == date(2024, 1, 4)
    # Saturday: rolls to Monday via the calendar.
    assert corpus.factor_date_for(datetime(2024, 1, 6, 12, 0), OPEN_DAYS) == date(2024, 1, 8)
    with pytest.raises(LookupError, match="no trading day after"):
        corpus.factor_date_for(datetime(2024, 1, 9, 16, 0), OPEN_DAYS)


# --- LLM payload validation ----------------------------------------------------


def test_parse_extraction_payload_accepts_valid_fenced_and_integer() -> None:
    result = corpus.parse_extraction_payload(_payload())
    assert result.sentiment == 0.6
    assert result.topic == "company"
    assert result.confidence == 0.9

    fenced = f"```json\n{_payload(topic='macro', sentiment=-0.25)}\n```"
    fenced_result = corpus.parse_extraction_payload(fenced)
    assert fenced_result.topic == "macro"
    assert fenced_result.sentiment == -0.25

    integers = corpus.parse_extraction_payload(_payload(sentiment=1, confidence=0))
    assert integers.sentiment == 1.0
    assert integers.confidence == 0.0


@pytest.mark.parametrize(
    "raw",
    [
        "this is not json",
        "[1, 2, 3]",
        _payload(sentiment=None),
        _payload(sentiment=1.01),
        _payload(sentiment=-1.5),
        _payload(sentiment="very positive"),
        _payload(sentiment=True),
        _payload(topic="sector"),
        _payload(topic=42),
        _payload(confidence=1.5),
        _payload(confidence=-0.1),
    ],
)
def test_parse_extraction_payload_fail_closed(raw: str) -> None:
    with pytest.raises(LlmExtractionError) as captured:
        corpus.parse_extraction_payload(raw)
    assert captured.value.stage == "llm_parse"


def test_parse_extraction_payload_requires_all_keys() -> None:
    for missing in ("sentiment", "topic", "confidence"):
        body = json.loads(_payload())
        del body[missing]
        with pytest.raises(LlmExtractionError, match="misses keys"):
            corpus.parse_extraction_payload(json.dumps(body))


def test_batch_messages_and_parser_are_exact_on_item_ids(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    items = corpus.load_corpus_items(tmp_path, datasets={"major_news"})[:2]

    messages = corpus.build_batch_extraction_messages(items, max_chars=4)
    request = json.loads(messages[1]["content"])
    assert [row["item_id"] for row in request["items"]] == [
        item.item_id for item in items
    ]
    assert all(len(row["text"]) <= 4 for row in request["items"])

    parsed = corpus.parse_batch_extraction_payload(
        _batch_payload(items), expected_item_ids=[item.item_id for item in items]
    )
    assert set(parsed) == {item.item_id for item in items}
    assert all(value.sentiment == 0.6 for value in parsed.values())

    missing = json.loads(_batch_payload(items))
    missing["items"].pop()
    with pytest.raises(LlmExtractionError, match="item_id mismatch"):
        corpus.parse_batch_extraction_payload(
            json.dumps(missing), expected_item_ids=[item.item_id for item in items]
        )


def test_process_batches_multiple_items_per_llm_call(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    items = corpus.load_corpus_items(tmp_path)
    batches = [items[index : index + 3] for index in range(0, len(items), 3)]
    chat = FakeChatClient([_batch_payload(batch) for batch in batches])

    summary = _run(tmp_path, chat, batch_size=3)

    assert summary.processed == 7
    assert summary.failed == 0
    assert summary.llm_calls == 3
    assert len(chat.calls) == 3
    assert [len(json.loads(call[0][1]["content"])["items"]) for call in chat.calls] == [
        3,
        3,
        1,
    ]


def test_batch_parse_failure_fails_closed_for_every_item(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    items = corpus.load_corpus_items(tmp_path)[:2]
    incomplete = json.loads(_batch_payload(items))
    incomplete["items"].pop()

    summary = _run(
        tmp_path,
        FakeChatClient([json.dumps(incomplete)]),
        limit=2,
        batch_size=2,
    )

    assert summary.processed == 0
    assert summary.failed == 2
    assert summary.llm_calls == 1
    assert pd.read_parquet(summary.fields_path).empty
    state = pd.read_parquet(summary.state_path)
    assert set(state["status"]) == {"failed"}
    assert set(state["stage"]) == {"llm_parse"}


# --- end-to-end processing -----------------------------------------------------


def test_process_end_to_end(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    chat = FakeChatClient(
        [
            _payload(sentiment=0.8, topic="policy"),
            _payload(sentiment=-0.4, topic="market"),
            _payload(sentiment=0.6, topic="macro"),
            _payload(sentiment=0.5, topic="policy"),
            _payload(sentiment=0.3, topic="policy"),
            _payload(sentiment=0.7, topic="company"),
            _payload(sentiment=-0.2, topic="industry"),
        ]
    )

    summary = _run(tmp_path, chat)

    assert summary.planned == 7
    assert summary.processed == 7
    assert summary.failed == 0
    assert summary.skipped == 0
    assert len(chat.calls) == 7
    assert chat.calls[0][1] == "test-model"
    assert corpus.PROMPT_VERSION in chat.calls[0][0][0]["content"]
    # Items are processed in pub_time order: the irm_qa_sh question comes first.
    assert "000001.SH" in chat.calls[0][0][1]["content"]
    assert "问：公司产能利用率如何？" in chat.calls[0][0][1]["content"]
    assert "market-level (MARKET)" in chat.calls[1][0][1]["content"]

    fields = pd.read_parquet(summary.fields_path)
    assert list(fields.columns) == list(corpus.FIELDS_COLUMNS)
    assert len(fields) == 7
    assert fields["ingested_at"].eq(pd.Timestamp(NOW)).all()
    assert fields["processed_at"].eq(pd.Timestamp(NOW)).all()
    assert set(fields["prompt_version"]) == {corpus.PROMPT_VERSION}
    assert set(fields["model"]) == {"test-model"}

    news = fields[fields["source_dataset"] == "major_news"].sort_values("pub_time")
    assert news["ts_code"].isna().all()
    # major_news: available_at is the exact publication moment.
    assert news["available_at"].tolist() == list(news["pub_time"])
    irm = fields[fields["source_dataset"].isin(["irm_qa_sh", "irm_qa_sz"])].sort_values(
        "pub_time"
    )
    # irm_qa: next trading day after the trade date (holiday-aware).
    assert irm["available_at"].tolist() == [
        pd.Timestamp(date(2024, 1, 4)),
        pd.Timestamp(date(2024, 1, 8)),
    ]
    policy = fields[fields["source_dataset"].isin(["npr", "cctv_news"])].sort_values(
        "pub_time"
    )
    # npr: exact publication moment; cctv_news: next trading day after the
    # broadcast date (2024-01-05 -> 2024-01-08 over the weekend).
    assert policy["available_at"].tolist() == [
        pd.Timestamp(datetime(2024, 1, 4, 9, 0, 0)),
        pd.Timestamp(date(2024, 1, 8)),
    ]

    state = pd.read_parquet(summary.state_path)
    assert state["status"].tolist() == ["succeeded"] * 7
    assert set(state["stage"]) == {"completed"}

    unit = pd.read_parquet(summary.unit_path)
    assert len(unit) == 7

    factors_dir = tmp_path / "corpus_nlp" / "factors"
    news_factor = pd.read_parquet(factors_dir / "news_sentiment_daily.parquet")
    assert list(news_factor.columns) == ["datetime", "instrument", "news_sentiment_daily"]
    assert set(news_factor["instrument"]) == {"MARKET"}
    # In pub_time order the scripted sentiments are: irm_sh 0.8, news 10:00 -0.4,
    # news 16:30 0.6, irm_sz 0.7, Saturday news -0.2. The 10:00 news stays on
    # 01-02, the 16:30 one rolls past the holiday to 01-04, Saturday to 01-08.
    assert news_factor["datetime"].tolist() == [
        pd.Timestamp(date(2024, 1, 2)),
        pd.Timestamp(date(2024, 1, 4)),
        pd.Timestamp(date(2024, 1, 8)),
    ]
    assert news_factor["news_sentiment_daily"].tolist() == [-0.4, 0.6, -0.2]

    policy_factor = pd.read_parquet(factors_dir / "policy_sentiment_daily.parquet")
    assert list(policy_factor.columns) == [
        "datetime",
        "instrument",
        "policy_sentiment_daily",
    ]
    assert set(policy_factor["instrument"]) == {"MARKET"}
    # The npr item stays on 01-04 (available 09:00, before the 15:00 cutoff);
    # the cctv_news item becomes visible on 01-08.
    assert policy_factor["datetime"].tolist() == [
        pd.Timestamp(date(2024, 1, 4)),
        pd.Timestamp(date(2024, 1, 8)),
    ]
    assert policy_factor["policy_sentiment_daily"].tolist() == [0.5, 0.3]

    irm_factor = pd.read_parquet(factors_dir / "irm_qa_sentiment_daily.parquet")
    assert list(irm_factor.columns) == ["datetime", "instrument", "irm_qa_sentiment_daily"]
    assert irm_factor["datetime"].tolist() == [
        pd.Timestamp(date(2024, 1, 4)),
        pd.Timestamp(date(2024, 1, 8)),
    ]
    assert irm_factor["instrument"].tolist() == ["000001.SH", "000002.SZ"]
    assert irm_factor["irm_qa_sentiment_daily"].tolist() == [0.8, 0.7]

    news_manifest = json.loads(
        (factors_dir / "news_sentiment_daily.json").read_text(encoding="utf-8")
    )
    assert news_manifest["factor"] == "news_sentiment_daily"
    assert news_manifest["rows"] == 3
    assert news_manifest["sha256"] == summary.factors["news_sentiment_daily"]["manifest"][
        "sha256"
    ]
    assert news_manifest["sha256"] == hashlib.sha256(
        (factors_dir / "news_sentiment_daily.parquet").read_bytes()
    ).hexdigest()
    assert "pub_time" in news_manifest["availability_policy"]["news_sentiment_daily"]
    assert "15:00" in news_manifest["availability_policy"]["news_sentiment_daily"]
    assert "MARKET" in news_manifest["instrument_convention"]
    assert news_manifest["source"]["source_datasets"] == ["major_news"]
    assert news_manifest["source"]["prompt_version"] == corpus.PROMPT_VERSION

    irm_manifest = json.loads(
        (factors_dir / "irm_qa_sentiment_daily.json").read_text(encoding="utf-8")
    )
    assert irm_manifest["rows"] == 2
    assert irm_manifest["source"]["source_datasets"] == ["irm_qa_sh", "irm_qa_sz"]
    assert "next trade_cal trading day" in irm_manifest["availability_policy"][
        "irm_qa_sentiment_daily"
    ]

    policy_manifest = json.loads(
        (factors_dir / "policy_sentiment_daily.json").read_text(encoding="utf-8")
    )
    assert policy_manifest["rows"] == 2
    assert policy_manifest["source"]["source_datasets"] == ["npr", "cctv_news"]
    assert "MARKET" in policy_manifest["instrument_convention"]
    assert policy_manifest["sha256"] == hashlib.sha256(
        (factors_dir / "policy_sentiment_daily.parquet").read_bytes()
    ).hexdigest()


def test_process_is_idempotent_on_processing_key(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    first = _run(tmp_path, FakeChatClient([_payload()] * 7))
    assert first.processed == 7

    chat = FakeChatClient([])
    second = _run(tmp_path, chat)

    assert second.planned == 7
    assert second.skipped == 7
    assert second.processed == 0
    assert second.failed == 0
    assert second.unit_path is None
    assert chat.calls == []
    assert len(pd.read_parquet(second.fields_path)) == 7
    assert second.factors["news_sentiment_daily"]["manifest"]["rows"] == 3


def test_process_reprocesses_when_prompt_version_changes(tmp_path: Path, monkeypatch) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    _run(tmp_path, FakeChatClient([_payload()] * 7))

    monkeypatch.setattr(corpus, "PROMPT_VERSION", "corpus-nlp.v3")
    chat = FakeChatClient([_payload(sentiment=0.1)] * 7)
    summary = _run(tmp_path, chat)

    assert summary.processed == 7
    assert summary.skipped == 0
    fields = pd.read_parquet(summary.fields_path)
    assert len(fields) == 14  # v1 and v2 rows coexist under distinct processing keys
    assert set(fields["prompt_version"]) == {"corpus-nlp.v2", "corpus-nlp.v3"}
    # Historical prompt generations remain auditable in fields, but only the
    # current prompt/model generation can enter the published factor.
    assert summary.factors["news_sentiment_daily"]["manifest"]["rows"] == 3
    news = pd.read_parquet(
        tmp_path / "corpus_nlp/factors/news_sentiment_daily.parquet"
    )
    assert news["news_sentiment_daily"].tolist() == [0.1, 0.1, 0.1]


def test_process_records_llm_failures_without_signals(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    chat = FakeChatClient(
        [
            "definitely not json",
            LlmExtractionError("LLM endpoint returned HTTP 500", stage="llm_call"),
            _payload(sentiment=0.5),
            _payload(sentiment=0.5),
            _payload(sentiment=0.5),
            _payload(sentiment=0.5),
            _payload(sentiment=0.5),
        ]
    )

    summary = _run(tmp_path, chat)

    assert summary.processed == 5
    assert summary.failed == 2
    assert summary.as_dict()["status"] == "failed"
    state = pd.read_parquet(summary.state_path)
    failures = state[state["status"] == "failed"]
    assert set(failures["stage"]) == {"llm_parse", "llm_call"}
    assert len(pd.read_parquet(summary.fields_path)) == 5
    # Factor artifacts only aggregate the successful rows.
    assert summary.factors["news_sentiment_daily"]["manifest"]["rows"] == 2


def test_process_checkpoints_and_resumes_after_interruption(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    progress: list[dict[str, int]] = []

    with pytest.raises(RuntimeError, match="worker interrupted"):
        _run(
            tmp_path,
            FakeChatClient([_payload(sentiment=0.1), RuntimeError("worker interrupted")]),
            limit=2,
            checkpoint_every=1,
            progress_callback=progress.append,
        )

    assert len(list((tmp_path / "corpus_nlp/units").glob("*.parquet"))) == 1
    assert len(pd.read_parquet(tmp_path / "corpus_nlp/fields.parquet")) == 1
    assert progress[-1]["completed"] == 1

    resumed_progress: list[dict[str, int]] = []
    resumed = _run(
        tmp_path,
        FakeChatClient([_payload(sentiment=0.2)]),
        limit=2,
        checkpoint_every=1,
        progress_callback=resumed_progress.append,
    )

    assert resumed.skipped == 1
    assert resumed.processed == 1
    assert resumed.failed == 0
    assert len(pd.read_parquet(resumed.fields_path)) == 2
    assert len(list((tmp_path / "corpus_nlp/units").glob("*.parquet"))) == 2
    assert resumed_progress[-1] == {
        "planned": 2,
        "completed": 2,
        "processed": 1,
        "skipped": 1,
        "failed": 0,
        "llm_calls": 1,
    }


def test_process_retries_failed_rows_on_rerun(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    failing = FakeChatClient(
        [LlmExtractionError("LLM request failed: boom", stage="llm_call")]
        + [_payload()] * 6
    )
    first = _run(tmp_path, failing)
    assert first.failed == 1
    assert first.processed == 6

    chat = FakeChatClient([_payload(sentiment=0.3)])
    second = _run(tmp_path, chat)

    assert second.planned == 7
    assert second.skipped == 6
    assert second.processed == 1
    assert second.failed == 0
    state = pd.read_parquet(second.state_path)
    assert set(state["status"]) == {"succeeded"}
    assert len(pd.read_parquet(second.fields_path)) == 7


def test_process_records_availability_failure_and_continues(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    # An irm_qa row dated on the last calendar day cannot derive available_at.
    _write_parquet(
        tmp_path / "units" / "irm_qa_sz",
        [{"trade_date": "20240109", "ts_code": "000003.SZ", "q": "年内规划？"}],
        name="extra.parquet",
    )
    chat = FakeChatClient([_payload()] * 7)

    summary = _run(tmp_path, chat)

    assert summary.planned == 8
    assert summary.processed == 7
    assert summary.failed == 1
    assert len(chat.calls) == 7  # the unavailable row never reaches the LLM
    state = pd.read_parquet(summary.state_path)
    failure = state[state["status"] == "failed"].iloc[0]
    assert failure["stage"] == "availability"
    assert failure["source_dataset"] == "irm_qa_sz"
    assert "no trading day after" in failure["error"]


def test_process_missing_corpus_fail_closed(tmp_path: Path) -> None:
    _seed_trade_cal(tmp_path)
    with pytest.raises(RuntimeError, match="major_news"):
        _run(tmp_path, FakeChatClient([]))


def test_process_default_excludes_audited_unavailable_npr(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    (tmp_path / "units" / "npr" / "data.parquet").unlink()
    _seed_trade_cal(tmp_path)

    summary = corpus.process_corpus(
        tmp_path,
        secret_store=FakeSecretStore(STORE_RECORD),
        chat_client=FakeChatClient([_payload()] * 6),
        now=lambda: NOW,
        environ={},
    )

    assert summary.planned == 6
    policy_manifest = summary.factors[corpus.POLICY_FACTOR_NAME]["manifest"]
    assert policy_manifest["source"]["source_datasets"] == ["cctv_news"]


def test_process_missing_trade_cal_fail_closed(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    with pytest.raises(RuntimeError, match="trade_cal"):
        _run(tmp_path, FakeChatClient([]))
    assert not (tmp_path / "corpus_nlp" / "state.parquet").exists()


def test_process_requires_real_credentials(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    chat = FakeChatClient([])

    with pytest.raises(LlmCredentialsError, match="POST /api/settings/llm"):
        corpus.process_corpus(
            tmp_path,
            secret_store=FakeSecretStore(None),
            chat_client=chat,
            now=lambda: NOW,
            environ={},
        )

    assert chat.calls == []
    assert not (tmp_path / "corpus_nlp" / "state.parquet").exists()
    assert not (tmp_path / "corpus_nlp" / "fields.parquet").exists()


def test_process_fail_closed_when_factor_date_beyond_calendar(tmp_path: Path) -> None:
    # News published after the cutoff on the last calendar day: the factor date
    # cannot be derived, so the run must fail before writing any artifact.
    _write_parquet(
        tmp_path / "units" / "major_news",
        [_news_row("盘后快讯", "正文", "2024-01-09 16:00:00")],
    )
    _write_parquet(
        tmp_path / "units" / "irm_qa_sh",
        [{"trade_date": "20240102", "ts_code": "000001.SH", "q": "产能如何？"}],
    )
    _write_parquet(
        tmp_path / "units" / "irm_qa_sz",
        [{"trade_date": "20240102", "ts_code": "000002.SZ", "q": "订单如何？"}],
    )
    _write_parquet(
        tmp_path / "units" / "npr",
        [{"title": "政策", "pubtime": "2024-01-02 09:00:00"}],
    )
    _write_parquet(
        tmp_path / "units" / "cctv_news",
        [{"date": "20240102", "title": "联播", "content": "正文"}],
    )
    _seed_trade_cal(tmp_path)

    with pytest.raises(LookupError, match="no trading day after"):
        _run(tmp_path, FakeChatClient([_payload()] * 5))

    assert not (tmp_path / "corpus_nlp" / "fields.parquet").exists()


def test_process_filters_datasets_ts_codes_dates_and_limit(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)

    irm_only = _run(
        tmp_path,
        FakeChatClient([_payload()] * 2),
        datasets={"irm_qa_sh", "irm_qa_sz"},
    )
    assert irm_only.planned == 2

    by_code = _run(tmp_path, FakeChatClient([]), ts_codes={"000001.SH"})
    # A ts_code filter excludes the market-level major_news items.
    assert by_code.planned == 1

    windowed = _run(tmp_path, FakeChatClient([]), start=date(2024, 2, 1))
    assert windowed.planned == 0


def test_load_corpus_items_pushes_date_bounds_into_duckdb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_parquet(
        tmp_path / "units" / "major_news",
        [
            _news_row("旧闻", "旧正文", "2024-01-01 09:00:00"),
            _news_row("窗口内", "正文", "2024-02-02 09:00:00"),
        ],
    )
    original = corpus._read_parquet_union
    calls: list[tuple[str, tuple[object, ...]]] = []

    def recording_read(paths, query, parameters=()):
        calls.append((query, tuple(parameters)))
        return original(paths, query, parameters)

    monkeypatch.setattr(corpus, "_read_parquet_union", recording_read)

    items = corpus.load_corpus_items(
        tmp_path,
        datasets={"major_news"},
        start=date(2024, 2, 1),
        end=date(2024, 2, 29),
    )

    assert [item.title for item in items] == ["窗口内"]
    bounded = [call for call in calls if call[1]]
    assert len(bounded) == 1
    assert "CAST(try_cast" in bounded[0][0]
    assert bounded[0][1] == (date(2024, 2, 1), date(2024, 2, 29))


def test_load_corpus_items_applies_deterministic_production_caps(tmp_path: Path) -> None:
    _write_parquet(
        tmp_path / "units" / "major_news",
        [
            _news_row(f"新闻{index}", f"正文{index}", "2024-01-02 09:00:00")
            for index in range(5)
        ],
    )
    first = corpus.load_corpus_items(
        tmp_path, datasets={"major_news"}, max_major_news_per_day=2
    )
    second = corpus.load_corpus_items(
        tmp_path, datasets={"major_news"}, max_major_news_per_day=2
    )
    assert len(first) == 2
    assert [item.item_id for item in first] == [item.item_id for item in second]

    _write_parquet(
        tmp_path / "units" / "irm_qa_sh",
        [
            {
                "trade_date": "20240102",
                "ts_code": "000001.SH",
                "q": f"问题{index}",
            }
            for index in range(4)
        ]
        + [
            {
                "trade_date": "20240102",
                "ts_code": "000002.SH",
                "q": "另一个公司",
            }
        ],
    )
    irm = corpus.load_corpus_items(
        tmp_path,
        datasets={"irm_qa_sh"},
        max_irm_per_instrument_day=2,
    )
    assert len([item for item in irm if item.ts_code == "000001.SH"]) == 2
    assert len([item for item in irm if item.ts_code == "000002.SH"]) == 1


def test_process_honors_limit(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)

    limited = _run(tmp_path, FakeChatClient([_payload()]), limit=1)

    assert limited.planned == 1
    assert limited.processed == 1


def test_factor_publication_is_restricted_to_current_corpus_scope(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    _run(tmp_path, FakeChatClient([_payload()] * 7))

    scoped = _run(tmp_path, FakeChatClient([]), limit=1)

    assert scoped.planned == 1
    assert scoped.skipped == 1
    assert sum(
        entry["manifest"]["rows"] for entry in scoped.factors.values()
    ) == 1
    for entry in scoped.factors.values():
        source = entry["manifest"]["source"]
        assert source["scope"]["planned"] == 1
        assert source["scope"]["process_key_count"] == 1


# --- CLI -----------------------------------------------------------------------


def test_cli_runs_with_injected_fakes(tmp_path: Path, monkeypatch) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    production_items = corpus.load_corpus_items(
        tmp_path, datasets=set(corpus.DEFAULT_CORPUS_DATASETS)
    )
    chat = FakeChatClient([_batch_payload(production_items)])
    monkeypatch.setattr(
        cli_module, "RuntimeSecretStore", lambda *args, **kwargs: FakeSecretStore(STORE_RECORD)
    )
    monkeypatch.setattr(corpus, "OpenAIChatClient", lambda *args, **kwargs: chat)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = CliRunner().invoke(
        app,
        [
            "corpus-nlp",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--limit",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dataset"] == "corpus_nlp"
    assert payload["datasets"] == [
        "major_news",
        "cctv_news",
        "irm_qa_sh",
        "irm_qa_sz",
    ]
    assert payload["planned"] == 6
    assert payload["processed"] == 6
    assert payload["failed"] == 0
    assert payload["llm_calls"] == 1
    assert payload["factors"]["news_sentiment_daily"]["rows"] == 3
    assert payload["factors"]["irm_qa_sentiment_daily"]["rows"] == 2
    assert payload["factors"]["policy_sentiment_daily"]["rows"] == 1
    assert len(chat.calls) == 1
    assert Path(payload["fields_path"]).is_file()


def test_cli_exits_nonzero_when_an_llm_row_fails(tmp_path: Path, monkeypatch) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    monkeypatch.setattr(
        cli_module, "RuntimeSecretStore", lambda *args, **kwargs: FakeSecretStore(STORE_RECORD)
    )
    monkeypatch.setattr(
        corpus, "OpenAIChatClient", lambda *args, **kwargs: FakeChatClient(["not-json"])
    )
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    result = CliRunner().invoke(
        app,
        [
            "corpus-nlp",
            "--dataset",
            "major_news",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 3
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["failed"] == 1


def test_cli_rejects_unknown_dataset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    result = CliRunner().invoke(app, ["corpus-nlp", "--dataset", "ths_hot"])

    assert result.exit_code != 0


# --- factor registration (fail closed, no database) ---------------------------

from quant_platform.research_store import ResearchStore  # noqa: E402

DUMMY_URL = "postgresql+psycopg://quantlab:quantlab@127.0.0.1:55433/quantlab_test"


def _unused_store() -> ResearchStore:
    return ResearchStore(DUMMY_URL)


def test_register_rejects_unknown_factor_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown corpus factor"):
        corpus.register_corpus_factor(_unused_store(), tmp_path, factor_name="nope")


def test_register_missing_manifest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest is missing"):
        corpus.register_corpus_factor(
            _unused_store(), tmp_path, factor_name=corpus.POLICY_FACTOR_NAME
        )


def test_register_tampered_artifact_fails_closed(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    summary = _run(tmp_path, FakeChatClient([_payload()] * 7))
    entry = summary.factors[corpus.POLICY_FACTOR_NAME]
    tampered = pd.read_parquet(entry["artifact_path"])
    tampered[corpus.POLICY_FACTOR_NAME] = tampered[corpus.POLICY_FACTOR_NAME] + 1
    tampered.to_parquet(entry["artifact_path"], index=False)
    with pytest.raises(ValueError, match="does not match the manifest sha256"):
        corpus.register_corpus_factor(
            _unused_store(),
            corpus.default_factors_dir(tmp_path),
            factor_name=corpus.POLICY_FACTOR_NAME,
        )


def test_register_unexpected_source_identity_fails_closed(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    summary = _run(tmp_path, FakeChatClient([_payload()] * 7))
    entry = summary.factors[corpus.NEWS_FACTOR_NAME]
    manifest = {**entry["manifest"], "source": {"dataset": "somewhere_else"}}
    entry["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source identity"):
        corpus.register_corpus_factor(
            _unused_store(),
            corpus.default_factors_dir(tmp_path),
            factor_name=corpus.NEWS_FACTOR_NAME,
        )


def test_code_artifact_recomputes_registered_values(tmp_path: Path) -> None:
    _seed_corpus(tmp_path)
    _seed_trade_cal(tmp_path)
    summary = _run(tmp_path, FakeChatClient([_payload()] * 7))
    fields = pd.read_parquet(summary.fields_path)
    builders = {
        corpus.NEWS_FACTOR_NAME: corpus.build_news_sentiment_series,
        corpus.IRM_QA_FACTOR_NAME: corpus.build_irm_qa_sentiment_series,
        corpus.POLICY_FACTOR_NAME: corpus.build_policy_sentiment_series,
    }
    for name, builder in builders.items():
        manifest = summary.factors[name]["manifest"]
        source = corpus._corpus_code_artifact_source(
            factor_name=name, manifest=manifest, values_sha256=manifest["sha256"]
        )
        assert corpus.PROMPT_VERSION in source
        assert manifest["sha256"] in source
        namespace: dict = {}
        exec(compile(source, "<code-artifact>", "exec"), namespace)
        recomputed = namespace["compute_factor"](fields, OPEN_DAYS)
        assert recomputed.equals(builder(fields, OPEN_DAYS))


# --- policy corpus loading (npr / cctv_news) ------------------------------------


def test_load_npr_tolerates_column_variants_and_drops_bad_rows(tmp_path: Path) -> None:
    _write_parquet(
        tmp_path / "units" / "npr",
        [
            # pub_time variant + content_html carried: body text wins.
            {
                "title": "政策甲",
                "pub_time": "2024-01-02 08:00:00",
                "content_html": "<p>全文</p>",
            },
            # Exact pubtime, title-only (the interface default fields).
            {"title": "政策乙", "pubtime": "2024-01-02 17:00:00"},
            # Unparseable timestamp and empty title: dropped.
            {"title": "政策丙", "pubtime": "not-a-timestamp"},
            {"title": "", "pubtime": "2024-01-02 18:00:00"},
        ],
    )

    items = corpus.load_corpus_items(tmp_path, datasets={"npr"})

    assert len(items) == 2
    by_title = {item.title: item for item in items}
    assert by_title["政策甲"].content == "<p>全文</p>"
    assert by_title["政策甲"].pub_time == datetime(2024, 1, 2, 8, 0, 0)
    # Title-level fallback when no content column value exists.
    assert by_title["政策乙"].content == "政策乙"


def test_load_npr_missing_required_columns_fail_closed(tmp_path: Path) -> None:
    _write_parquet(tmp_path / "units" / "npr", [{"pcode": "国发〔2024〕1号"}])

    with pytest.raises(RuntimeError, match="title"):
        corpus.load_corpus_items(tmp_path, datasets={"npr"})
    _write_parquet(
        tmp_path / "units" / "npr", [{"title": "政策"}], name="other.parquet"
    )
    with pytest.raises(RuntimeError, match=r"pubtime\|pub_time"):
        corpus.load_corpus_items(tmp_path, datasets={"npr"})


def test_load_cctv_news_parses_dates_and_drops_bad_rows(tmp_path: Path) -> None:
    _write_parquet(
        tmp_path / "units" / "cctv_news",
        [
            {"date": "20240102", "title": "联播甲", "content": "内容甲"},
            {"date": "2024-01-04", "title": "联播乙", "content": "内容乙"},
            {"date": "bad-date", "title": "联播丙", "content": "内容丙"},
            {"date": "20240105", "title": "", "content": ""},
        ],
    )

    items = corpus.load_corpus_items(tmp_path, datasets={"cctv_news"})

    assert len(items) == 2
    assert items[0].pub_time == datetime(2024, 1, 2, 0, 0, 0)
    assert items[1].pub_time == datetime(2024, 1, 4, 0, 0, 0)
    assert items[0].content == "内容甲"


def test_load_cctv_news_missing_columns_fail_closed(tmp_path: Path) -> None:
    _write_parquet(tmp_path / "units" / "cctv_news", [{"date": "20240102"}])

    with pytest.raises(RuntimeError, match="content"):
        corpus.load_corpus_items(tmp_path, datasets={"cctv_news"})
