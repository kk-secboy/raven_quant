from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import quant_platform.db_cli as db_cli
from quant_data.availability import availability_policy
from quant_platform import corpus_nlp as corpus
from quant_platform import major_news_mentions as m
from quant_platform.external_factor_evaluation import (
    SHAPE_SPARSE_EVENT,
    detect_external_factor_shape,
)
from quant_platform.research_store import ResearchStore

NOW = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
STORE_RECORD = {
    "api_key": "sk-test-fake-key",
    "api_base": "https://llm.example.invalid/v1",
    "chat_model": "test-model",
}
DUMMY_URL = "postgresql+psycopg://quantlab:quantlab@127.0.0.1:55433/quantlab_test"

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


class FakeChatClient:
    """Scripted ChatCompleter stand-in; fails the test on unscripted calls."""

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


def _payload(sentiment: float) -> str:
    return json.dumps(
        {"sentiment": sentiment, "topic": "company", "confidence": 0.9},
        ensure_ascii=False,
    )


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


def _stock_basic_rows() -> list[dict]:
    return [
        {
            "ts_code": "601318.SH",
            "name": "中国平安",
            "fullname": "中国平安保险(集团)股份有限公司",
            "list_date": "20070301",
        },
        {"ts_code": "000001.SZ", "name": "平安银行", "list_date": "19910403"},
        {"ts_code": "000002.SZ", "name": "万科A", "list_date": "19910129"},
        {"ts_code": "600519.SH", "name": "贵州茅台", "list_date": "20010827"},
        # Two-character alias: dropped by the minimum-length rule.
        {"ts_code": "111111.SZ", "name": "平安", "list_date": "20200101"},
        # Cross-stock conflict on the same alias: both dropped.
        {"ts_code": "222222.SZ", "name": "测试集团", "list_date": "20000101"},
        {"ts_code": "333333.SZ", "name": "测试集团", "list_date": "20010101"},
    ]


def _seed_stock_basic(data_root: Path) -> None:
    _write_parquet(data_root / "units" / "stock_basic", _stock_basic_rows())


def _seed_namechange(data_root: Path) -> None:
    _write_parquet(
        data_root / "units" / "namechange",
        [
            # Historical name that expired long before the 2024 news items.
            {
                "ts_code": "600519.SH",
                "name": "茅台股份",
                "start_date": "20010827",
                "end_date": "20051231",
                "ann_date": "20051231",
            },
        ],
    )


# (title, content, pub_time, scripted sentiment)
NEWS_ROWS = [
    (
        "中国平安发布业绩快报",
        "中国平安保险(集团)股份有限公司今日披露。",
        "2024-01-02 10:00:00",
        0.8,
    ),
    ("中国平安再登价值榜单", "机构继续看好中国平安。", "2024-01-02 11:00:00", 1.0),
    ("银行板块走强", "平安银行领涨，万科A跟涨。", "2024-01-02 16:30:00", -0.5),
    ("市场综述", "茅台股份创历史新高，测试集团异动。", "2024-01-05 09:00:00", 0.3),
    ("贵州茅台发布业绩预告", "贵州茅台净利增长。", "2024-01-05 14:00:00", 0.6),
    ("宏观综述", "今日两市震荡整理。", "2024-01-05 15:30:00", 0.1),
]


def _seed_major_news(data_root: Path) -> None:
    _write_parquet(
        data_root / "units" / "major_news",
        [
            {"title": title, "content": content, "pub_time": pub_time}
            for title, content, pub_time, _sentiment in NEWS_ROWS
        ],
    )


def _run_corpus(data_root: Path) -> corpus.CorpusNlpSummary:
    return corpus.process_corpus(
        data_root,
        datasets={"major_news"},
        secret_store=FakeSecretStore(),
        chat_client=FakeChatClient([_payload(sentiment) for *_r, sentiment in NEWS_ROWS]),
        now=lambda: NOW,
        environ={},
    )


def _seed_full(data_root: Path) -> None:
    _seed_trade_cal(data_root)
    _seed_stock_basic(data_root)
    _seed_namechange(data_root)
    _seed_major_news(data_root)
    _run_corpus(data_root)


def _unused_store() -> ResearchStore:
    return ResearchStore(DUMMY_URL)


# --- alias table rules --------------------------------------------------------


@pytest.mark.no_database
def test_alias_table_covers_name_fullname_and_namechange() -> None:
    stock_basic = pd.DataFrame(_stock_basic_rows())
    stock_basic["list_date"] = pd.to_datetime(stock_basic["list_date"])
    namechange = pd.DataFrame(
        [
            {
                "ts_code": "600519.SH",
                "name": "茅台股份",
                "start_date": pd.Timestamp("2001-08-27"),
                "end_date": pd.Timestamp("2005-12-31"),
            }
        ]
    )

    aliases = m.build_alias_table(stock_basic, namechange)

    by_alias = aliases.set_index("alias")
    assert "中国平安" in by_alias.index
    assert "中国平安保险(集团)股份有限公司" in by_alias.index
    assert by_alias.loc["中国平安", "kind"] == "name"
    assert by_alias.loc["中国平安保险(集团)股份有限公司", "kind"] == "fullname"
    # namechange historical interval kept with its validity bounds.
    historical = by_alias.loc["茅台股份"]
    assert historical["ts_code"] == "600519.SH"
    assert historical["valid_from"] == pd.Timestamp("2001-08-27")
    assert historical["valid_to"] == pd.Timestamp("2005-12-31")


@pytest.mark.no_database
def test_alias_table_drops_short_and_conflicting_aliases() -> None:
    stock_basic = pd.DataFrame(_stock_basic_rows())
    stock_basic["list_date"] = pd.to_datetime(stock_basic["list_date"])

    aliases = m.build_alias_table(stock_basic, None)

    names = set(aliases["alias"])
    # Two-character alias dropped (the 平安 ambiguity case).
    assert "平安" not in names
    # Cross-stock conflict: the alias is dropped for both stocks.
    assert "测试集团" not in names
    dropped_codes = set(aliases["ts_code"])
    assert "111111.SZ" not in dropped_codes
    assert "222222.SZ" not in dropped_codes
    assert "333333.SZ" not in dropped_codes
    # Three-character aliases survive.
    assert "万科A" in names


@pytest.mark.no_database
def test_alias_table_keeps_sequential_name_reuse() -> None:
    stock_basic = pd.DataFrame(
        [
            {"ts_code": "444444.SZ", "name": "东方科技", "list_date": "20150601"},
            {"ts_code": "555555.SZ", "name": "西部材料", "list_date": "20000101"},
        ]
    )
    stock_basic["list_date"] = pd.to_datetime(stock_basic["list_date"])
    namechange = pd.DataFrame(
        [
            # The same alias was used by another stock in a disjoint interval.
            {
                "ts_code": "555555.SZ",
                "name": "东方科技",
                "start_date": pd.Timestamp("2000-01-01"),
                "end_date": pd.Timestamp("2010-12-31"),
            },
        ]
    )

    aliases = m.build_alias_table(stock_basic, namechange)

    reuse = aliases[aliases["alias"] == "东方科技"].sort_values("valid_from")
    assert reuse["ts_code"].tolist() == ["555555.SZ", "444444.SZ"]
    index_2005 = m._alias_index_at(aliases, date(2005, 6, 1))
    assert m.find_mentions("东方科技涨停", index_2005) == {"555555.SZ": "东方科技"}
    index_2020 = m._alias_index_at(aliases, date(2020, 6, 1))
    assert m.find_mentions("东方科技涨停", index_2020) == {"444444.SZ": "东方科技"}


# --- mention matching ---------------------------------------------------------


def _index_from(aliases: pd.DataFrame, day: date = date(2024, 1, 2)):
    return m._alias_index_at(aliases, day)


@pytest.mark.no_database
def test_find_mentions_longest_match_and_single_report_per_stock() -> None:
    stock_basic = pd.DataFrame(
        [
            {"ts_code": "600036.SH", "name": "招商银行"},
            {"ts_code": "001979.SZ", "name": "招商蛇口"},
            {"ts_code": "601318.SH", "name": "中国平安"},
        ]
    )
    aliases = m.build_alias_table(stock_basic, None)
    index = _index_from(aliases)

    mentions = m.find_mentions("招商银行和招商蛇口联袂上涨，中国平安中国平安", index)

    # Both 招商-prefixed stocks resolve to their full names (longest match),
    # and the duplicated 中国平安 is reported once.
    assert mentions == {
        "600036.SH": "招商银行",
        "001979.SZ": "招商蛇口",
        "601318.SH": "中国平安",
    }


@pytest.mark.no_database
def test_find_mentions_respects_validity_intervals() -> None:
    stock_basic = pd.DataFrame([{"ts_code": "600519.SH", "name": "贵州茅台"}])
    namechange = pd.DataFrame(
        [
            {
                "ts_code": "600519.SH",
                "name": "茅台股份",
                "start_date": pd.Timestamp("2001-08-27"),
                "end_date": pd.Timestamp("2005-12-31"),
            }
        ]
    )
    aliases = m.build_alias_table(stock_basic, namechange)

    # The historical name matches inside its interval only.
    assert m.find_mentions("茅台股份大涨", _index_from(aliases, date(2003, 5, 9))) == {
        "600519.SH": "茅台股份"
    }
    assert m.find_mentions("茅台股份大涨", _index_from(aliases, date(2024, 1, 2))) == {}
    # The current name matches after listing; the alias table without a
    # list_date treats the current name as always valid.
    assert m.find_mentions("贵州茅台大涨", _index_from(aliases, date(2024, 1, 2))) == {
        "600519.SH": "贵州茅台"
    }


# --- end-to-end pipeline ------------------------------------------------------


@pytest.mark.no_database
def test_process_builds_events_and_factor_artifacts(tmp_path: Path) -> None:
    _seed_full(tmp_path)

    summary = m.process_major_news_mentions(tmp_path, now=lambda: NOW)

    assert summary.items == len(NEWS_ROWS)
    events = pd.read_parquet(summary.events_path)
    assert list(events.columns) == list(m.EVENT_COLUMNS)
    # item1+item6 -> 601318.SH on 01-02; item2 (16:30) -> 000001.SZ + 000002.SZ
    # rolled past the holiday to 01-04; item4 -> 600519.SH on 01-05. The
    # expired 茅台股份 alias and the conflicting 测试集团 alias match nothing.
    assert set(zip(events["ts_code"], events["matched_alias"], strict=True)) == {
        ("601318.SH", "中国平安"),
        ("000001.SZ", "平安银行"),
        ("000002.SZ", "万科A"),
        ("600519.SH", "贵州茅台"),
    }
    by_code = events.groupby("ts_code")
    assert by_code.size().to_dict() == {
        "601318.SH": 2,
        "000001.SZ": 1,
        "000002.SZ": 1,
        "600519.SH": 1,
    }
    # PIT: no factor value precedes its available_at; the 16:30 item rolls to
    # the next trading day after the Wednesday holiday.
    assert (events["factor_date"] >= events["available_at"].dt.normalize()).all()
    bank = events[events["ts_code"] == "000001.SZ"].iloc[0]
    assert bank["factor_date"] == pd.Timestamp(date(2024, 1, 4))
    pingan = events[events["ts_code"] == "601318.SH"]
    assert set(pingan["factor_date"]) == {pd.Timestamp(date(2024, 1, 2))}

    factors_dir = m.default_factors_dir(tmp_path)
    sentiment = pd.read_parquet(factors_dir / f"{m.SENTIMENT_FACTOR_NAME}.parquet")
    keyed = sentiment.set_index(["datetime", "instrument"])[m.SENTIMENT_FACTOR_NAME]
    assert keyed.to_dict() == {
        (pd.Timestamp(date(2024, 1, 2)), "601318.SH"): pytest.approx(0.9),
        (pd.Timestamp(date(2024, 1, 4)), "000001.SZ"): -0.5,
        (pd.Timestamp(date(2024, 1, 4)), "000002.SZ"): -0.5,
        (pd.Timestamp(date(2024, 1, 5)), "600519.SH"): 0.6,
    }
    counts = pd.read_parquet(factors_dir / f"{m.COUNT_FACTOR_NAME}.parquet")
    count_keyed = counts.set_index(["datetime", "instrument"])[m.COUNT_FACTOR_NAME]
    assert count_keyed[(pd.Timestamp(date(2024, 1, 2)), "601318.SH")] == 2.0
    assert count_keyed.sum() == 5.0

    manifest = json.loads(
        (factors_dir / f"{m.SENTIMENT_FACTOR_NAME}.json").read_text(encoding="utf-8")
    )
    assert manifest["rows"] == 4
    assert manifest["source"]["dataset"] == "corpus_nlp_fields"
    assert manifest["source"]["mention_rules_version"] == m.MENTION_RULES_VERSION
    assert manifest["source"]["prompt_version"] == corpus.PROMPT_VERSION
    assert manifest["source"]["model"] == "test-model"
    assert manifest["sha256"] == hashlib.sha256(
        (factors_dir / f"{m.SENTIMENT_FACTOR_NAME}.parquet").read_bytes()
    ).hexdigest()
    assert "mention-rules.v1" in manifest["availability_policy"][m.SENTIMENT_FACTOR_NAME]

    # Deterministic rerun: identical inputs reproduce identical checksums.
    rerun = m.process_major_news_mentions(tmp_path, now=lambda: NOW)
    for name in m.FACTOR_NAMES:
        assert (
            rerun.factors[name]["manifest"]["sha256"]
            == summary.factors[name]["manifest"]["sha256"]
        )


@pytest.mark.no_database
def test_process_excludes_items_without_successful_extraction(tmp_path: Path) -> None:
    _seed_trade_cal(tmp_path)
    _seed_stock_basic(tmp_path)
    _seed_major_news(tmp_path)
    # Only one item gets a successful fields row.
    corpus.process_corpus(
        tmp_path,
        datasets={"major_news"},
        limit=1,
        secret_store=FakeSecretStore(),
        chat_client=FakeChatClient([_payload(0.8)]),
        now=lambda: NOW,
        environ={},
    )

    summary = m.process_major_news_mentions(tmp_path, now=lambda: NOW)

    events = pd.read_parquet(summary.events_path)
    assert set(events["ts_code"]) == {"601318.SH"}


@pytest.mark.no_database
def test_process_missing_inputs_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="major_news"):
        m.process_major_news_mentions(tmp_path, now=lambda: NOW)

    _seed_trade_cal(tmp_path)
    _seed_major_news(tmp_path)
    _run_corpus(tmp_path)
    with pytest.raises(RuntimeError, match="stock_basic"):
        m.process_major_news_mentions(tmp_path, now=lambda: NOW)


@pytest.mark.no_database
def test_process_missing_corpus_fields_fail_closed(tmp_path: Path) -> None:
    _seed_trade_cal(tmp_path)
    _seed_stock_basic(tmp_path)
    _seed_major_news(tmp_path)

    with pytest.raises(RuntimeError, match="corpus-nlp"):
        m.process_major_news_mentions(tmp_path, now=lambda: NOW)


@pytest.mark.no_database
def test_produced_factor_fits_the_sparse_event_shape(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    m.process_major_news_mentions(tmp_path, now=lambda: NOW)
    factors_dir = m.default_factors_dir(tmp_path)
    factor = pd.read_parquet(factors_dir / f"{m.SENTIMENT_FACTOR_NAME}.parquet")
    factor_series = factor.set_index(["datetime", "instrument"])[m.SENTIMENT_FACTOR_NAME]

    # Label universe: 10 instruments on 12 trading days; the factor covers 3.
    days = pd.date_range("2024-01-02", periods=12, freq="B")
    instruments = [f"{code:06d}.SZ" for code in range(10)]
    index = pd.MultiIndex.from_product(
        [days, instruments], names=["datetime", "instrument"]
    )
    labels = pd.Series(0.01, index=index, name="label")

    shape = detect_external_factor_shape(
        factor_series,
        labels,
        valid_start=date(2024, 1, 2),
        valid_end=date(2024, 1, 31),
    )

    assert shape == SHAPE_SPARSE_EVENT


@pytest.mark.no_database
def test_corpus_availability_policies_are_registered() -> None:
    for dataset in (
        "major_news",
        "news",
        "npr",
        "cctv_news",
        "irm_qa_sh",
        "irm_qa_sz",
        "research_report",
    ):
        assert availability_policy(dataset) is not None, dataset


# --- registration (fail closed, no database) -----------------------------------


@pytest.mark.no_database
def test_register_rejects_unknown_factor_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown major_news mention factor"):
        m.register_major_news_mentions_factor(_unused_store(), tmp_path, factor_name="nope")


@pytest.mark.no_database
def test_register_missing_manifest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest is missing"):
        m.register_major_news_mentions_factor(
            _unused_store(), tmp_path, factor_name=m.SENTIMENT_FACTOR_NAME
        )


@pytest.mark.no_database
def test_register_tampered_artifact_fails_closed(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_major_news_mentions(tmp_path, now=lambda: NOW)
    entry = summary.factors[m.SENTIMENT_FACTOR_NAME]
    tampered = pd.read_parquet(entry["artifact_path"])
    tampered[m.SENTIMENT_FACTOR_NAME] = tampered[m.SENTIMENT_FACTOR_NAME] + 1
    tampered.to_parquet(entry["artifact_path"], index=False)
    with pytest.raises(ValueError, match="does not match the manifest sha256"):
        m.register_major_news_mentions_factor(
            _unused_store(),
            m.default_factors_dir(tmp_path),
            factor_name=m.SENTIMENT_FACTOR_NAME,
        )


@pytest.mark.no_database
def test_register_unexpected_source_identity_fails_closed(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_major_news_mentions(tmp_path, now=lambda: NOW)
    entry = summary.factors[m.COUNT_FACTOR_NAME]
    manifest = {**entry["manifest"], "source": {"dataset": "somewhere_else"}}
    entry["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source identity"):
        m.register_major_news_mentions_factor(
            _unused_store(),
            m.default_factors_dir(tmp_path),
            factor_name=m.COUNT_FACTOR_NAME,
        )


@pytest.mark.no_database
def test_code_artifact_recomputes_registered_values(tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_major_news_mentions(tmp_path, now=lambda: NOW)
    events = pd.read_parquet(summary.events_path)
    builders = {
        m.SENTIMENT_FACTOR_NAME: m.build_mention_sentiment_series,
        m.COUNT_FACTOR_NAME: m.build_mention_count_series,
    }
    for name, builder in builders.items():
        manifest = summary.factors[name]["manifest"]
        source = m._code_artifact_source(
            factor_name=name, manifest=manifest, values_sha256=manifest["sha256"]
        )
        assert m.MENTION_RULES_VERSION in source
        assert manifest["sha256"] in source
        namespace: dict = {}
        exec(compile(source, "<code-artifact>", "exec"), namespace)
        recomputed = namespace["compute_factor"](events)
        assert recomputed.equals(builder(events))


# --- registration into factor_candidates (real database) ----------------------


def test_register_success(database_url: str, tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_major_news_mentions(tmp_path, now=lambda: NOW)
    store = ResearchStore(database_url)
    factors_dir = m.default_factors_dir(tmp_path)

    result = m.register_major_news_mentions_factor(
        store, factors_dir, factor_name=m.SENTIMENT_FACTOR_NAME
    )

    assert result["created"] is True
    candidate = store.get_candidate(result["candidate_id"])
    assert candidate["name"] == m.SENTIMENT_FACTOR_NAME
    assert candidate["status"] == "awaiting_evaluation"
    assert candidate["values_sha256"] == result["values_sha256"]
    code_path = Path(candidate["code_path"])
    assert code_path.is_file()
    assert candidate["code_sha256"] == hashlib.sha256(code_path.read_bytes()).hexdigest()
    variables = candidate["variables"]
    assert variables["source"]["dataset"] == "corpus_nlp_fields"
    assert variables["source"]["mention_rules_version"] == m.MENTION_RULES_VERSION
    assert variables["min_alias_chars"] == m.MIN_ALIAS_CHARS
    assert "mention-rules.v1" in candidate["description"]
    run = store.get_run(result["run_id"])
    assert run["kind"] == m.IMPORT_RUN_KIND
    assert run["status"] == "succeeded"
    assert run["dataset"] == "corpus_nlp_fields"
    events = {event["event_type"] for event in store.list_events(run["id"])}
    assert {"run.created", "candidate.imported", "run.succeeded"} <= events
    assert variables["rows"] == summary.factors[m.SENTIMENT_FACTOR_NAME]["manifest"]["rows"]


def test_register_is_idempotent_for_same_sha256(database_url: str, tmp_path: Path) -> None:
    _seed_full(tmp_path)
    m.process_major_news_mentions(tmp_path, now=lambda: NOW)
    store = ResearchStore(database_url)
    factors_dir = m.default_factors_dir(tmp_path)

    first = m.register_major_news_mentions_factor(
        store, factors_dir, factor_name=m.COUNT_FACTOR_NAME
    )
    second = m.register_major_news_mentions_factor(
        store, factors_dir, factor_name=m.COUNT_FACTOR_NAME
    )

    assert first["created"] is True
    assert second["created"] is False
    assert second["candidate_id"] == first["candidate_id"]
    candidates = [
        item
        for item in store.list_candidates(limit=100)
        if item["name"] == m.COUNT_FACTOR_NAME
    ]
    assert len(candidates) == 1


def test_register_checksum_mismatch_writes_nothing(database_url: str, tmp_path: Path) -> None:
    _seed_full(tmp_path)
    summary = m.process_major_news_mentions(tmp_path, now=lambda: NOW)
    entry = summary.factors[m.SENTIMENT_FACTOR_NAME]
    tampered = pd.read_parquet(entry["artifact_path"])
    tampered[m.SENTIMENT_FACTOR_NAME] = tampered[m.SENTIMENT_FACTOR_NAME] + 1
    tampered.to_parquet(entry["artifact_path"], index=False)
    store = ResearchStore(database_url)

    with pytest.raises(ValueError, match="does not match the manifest sha256"):
        m.register_major_news_mentions_factor(
            store,
            m.default_factors_dir(tmp_path),
            factor_name=m.SENTIMENT_FACTOR_NAME,
        )

    assert store.list_candidates(limit=100) == []
    assert store.list_runs(limit=100) == []


def test_cli_registers_all_factors_idempotently(
    database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_full(tmp_path)
    m.process_major_news_mentions(tmp_path, now=lambda: NOW)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    runner = CliRunner()
    first = runner.invoke(db_cli.app, ["register-major-news-mentions-factor"])
    assert first.exit_code == 0, first.output
    payload = json.loads(first.output)
    assert {item["factor_name"] for item in payload["factors"]} == set(m.FACTOR_NAMES)
    assert all(item["created"] is True for item in payload["factors"])

    second = runner.invoke(db_cli.app, ["register-major-news-mentions-factor"])
    assert second.exit_code == 0, second.output
    repeated = json.loads(second.output)
    assert all(item["created"] is False for item in repeated["factors"])
    first_ids = {item["factor_name"]: item["candidate_id"] for item in payload["factors"]}
    for item in repeated["factors"]:
        assert item["candidate_id"] == first_ids[item["factor_name"]]
