from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import requests
from typer.testing import CliRunner

from quant_data import cninfo_announcements as cninfo
from quant_data.cli import app
from quant_data.config import Settings
from quant_data.rate_limit import GlobalRateGate
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database

NOW = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)
OPEN_DAYS = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]


def test_worker_builds_resumable_cninfo_download_command(tmp_path: Path) -> None:
    worker = LocalJobWorker.__new__(LocalJobWorker)
    worker.settings = Settings(api_url="", token="", data_root=tmp_path)

    command, result_path, environment = worker._command(
        {
            "id": "announcement-job",
            "kind": "cninfo_announcements_download",
            "payload": {
                "start": "2024-01-01",
                "end": "2026-08-03",
                "ts_codes": ["000001.SZ", "600519.SH"],
                "limit": 25,
            },
        }
    )

    assert "cninfo-announcements" in command
    assert "000001.SZ,600519.SH" in command
    assert "25" in command
    assert "--regulatory-only" in command
    assert result_path == tmp_path / "artifacts/execution-data/announcement-job/result.json"
    assert environment == {}


def _pdf(label: str) -> bytes:
    return b"%PDF-1.4\n" + label.encode("utf-8") + b"\n%%EOF"


class FakeResponse:
    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content


class FakeSession:
    """Scripted requests.Session stand-in; fails the test on unscripted calls."""

    def __init__(self, script: dict[str, list]) -> None:
        self.script = {url: list(outcomes) for url, outcomes in script.items()}
        self.calls: list[str] = []

    def get(self, url: str, timeout=None, headers=None):
        self.calls.append(url)
        outcomes = self.script.get(url)
        if not outcomes:
            raise AssertionError(f"unexpected HTTP call: {url}")
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client(session: FakeSession, *, max_attempts: int = 3, delays: list | None = None):
    sleeper = (lambda seconds: delays.append(seconds)) if delays is not None else (lambda s: None)
    return cninfo.CninfoHttpClient(
        rate_gate=GlobalRateGate(60_000),
        session=session,
        max_attempts=max_attempts,
        sleeper=sleeper,
    )


def _write_anns_d(data_root: Path, rows: list[dict], *, layout: str = "units") -> Path:
    if layout == "units":
        directory = data_root / "units" / "anns_d"
        target = directory / "anns_d_202401.parquet"
    else:
        directory = (
            data_root
            / "snapshots"
            / "cn-test"
            / "parquet"
            / "anns_d"
            / "partition_year=2024"
            / "partition_month=1"
        )
        target = directory / "data.parquet"
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame["ann_date"] = pd.to_datetime(frame["ann_date"], format="%Y%m%d")
    frame.to_parquet(target, index=False, compression="zstd", engine="pyarrow")
    return target


def _write_trade_cal(
    data_root: Path, open_days: list[date], closed_days: list[date] | None = None
) -> None:
    directory = data_root / "units" / "trade_cal"
    directory.mkdir(parents=True, exist_ok=True)
    rows = [(day, 1) for day in open_days] + [(day, 0) for day in (closed_days or [])]
    frame = pd.DataFrame(rows, columns=["cal_date", "is_open"])
    frame["cal_date"] = pd.to_datetime(frame["cal_date"])
    frame.to_parquet(
        directory / "trade_cal.parquet", index=False, compression="zstd", engine="pyarrow"
    )


def _ann_row(ts_code: str, ann_date: str, title: str, url: str) -> dict:
    return {"ts_code": ts_code, "ann_date": ann_date, "title": title, "url": url}


def _seed_data(data_root: Path, rows: list[dict]) -> None:
    _write_anns_d(data_root, rows)
    _write_trade_cal(data_root, OPEN_DAYS, closed_days=[date(2024, 1, 6), date(2024, 1, 7)])


def _run(data_root: Path, client, **kwargs):
    return cninfo.download_cninfo_announcements(
        data_root, client=client, now=lambda: NOW, **kwargs
    )


def test_manifest_loads_units_and_snapshots_deduped(tmp_path: Path) -> None:
    rows = [
        _ann_row("000001.SZ", "20240102", "年度报告", "https://static.cninfo.com.cn/a.pdf"),
        _ann_row("000002.SZ", "20240103", "问询函回复", "https://static.cninfo.com.cn/b.pdf"),
        _ann_row("000003.SZ", "20240104", "无链接公告", ""),
    ]
    _write_anns_d(tmp_path, rows)
    _write_anns_d(
        tmp_path,
        [
            _ann_row("000009.SZ", "20240103", "重复公告", "https://static.cninfo.com.cn/a.pdf"),
            _ann_row("000004.SZ", "20240105", "快照公告", "https://static.cninfo.com.cn/c.pdf"),
        ],
        layout="snapshot",
    )

    refs = cninfo.load_announcement_manifest(tmp_path)

    urls = [ref.url for ref in refs]
    assert urls == [
        "https://static.cninfo.com.cn/a.pdf",
        "https://static.cninfo.com.cn/b.pdf",
        "https://static.cninfo.com.cn/c.pdf",
    ]
    assert all(ref.url for ref in refs)
    assert refs[0].ann_date == date(2024, 1, 2)


def test_manifest_filters_ts_code_and_date_range(tmp_path: Path) -> None:
    _write_anns_d(
        tmp_path,
        [
            _ann_row("000001.SZ", "20240102", "A", "https://static.cninfo.com.cn/a.pdf"),
            _ann_row("000002.SZ", "20240103", "B", "https://static.cninfo.com.cn/b.pdf"),
            _ann_row("000001.SZ", "20240201", "C", "https://static.cninfo.com.cn/c.pdf"),
        ],
    )

    refs = cninfo.load_announcement_manifest(
        tmp_path,
        ts_codes={"000001.SZ"},
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
    )

    assert [ref.url for ref in refs] == ["https://static.cninfo.com.cn/a.pdf"]


def test_manifest_dedupes_urls_after_detail_page_resolution(tmp_path: Path) -> None:
    base = (
        "https://www.cninfo.com.cn/new/disclosure/detail?"
        "announcementId=1200000001&announcementTime=2024-01-02"
    )
    _write_anns_d(
        tmp_path,
        [
            _ann_row("000001.SZ", "20240102", "A", f"{base}&orgId=a"),
            _ann_row("000002.SZ", "20240102", "A duplicate", f"{base}&orgId=b"),
        ],
    )

    refs = cninfo.load_announcement_manifest(tmp_path)

    assert len(refs) == 1
    assert refs[0].url == (
        "http://static.cninfo.com.cn/finalpage/2024-01-02/1200000001.PDF"
    )


def test_manifest_fail_closed_without_anns_d(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="anns_d"):
        cninfo.load_announcement_manifest(tmp_path)


def test_categorize_title_marks_regulatory_letters() -> None:
    for title in (
        "关于对某某公司的问询函",
        "关注函",
        "监管函",
        "警示函",
        "纪律处分决定书",
    ):
        assert cninfo.categorize_title(title) == "regulatory_letter"
    assert cninfo.categorize_title("2023年年度报告") == "announcement"
    assert cninfo.categorize_title(None) == "announcement"
    assert cninfo.categorize_title("") == "announcement"


def test_regulatory_only_download_filters_manifest_before_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = [
        cninfo.AnnouncementRef("000001.SZ", date(2024, 1, 2), "年度报告", "https://a"),
        cninfo.AnnouncementRef("000002.SZ", date(2024, 1, 3), "关注函", "https://b"),
    ]
    monkeypatch.setattr(cninfo, "load_announcement_manifest", lambda *args, **kwargs: refs)
    monkeypatch.setattr(cninfo, "load_trade_calendar_open_days", lambda *args: OPEN_DAYS)
    session = FakeSession({"https://b": [FakeResponse(200, _pdf("regulatory"))]})

    summary = cninfo.download_cninfo_announcements(
        tmp_path,
        regulatory_only=True,
        limit=1,
        client=_client(session),
        now=lambda: NOW,
    )

    assert summary.planned == 1
    assert summary.downloaded == 1
    assert session.calls == ["https://b"]


def test_next_trading_day_is_strictly_after_ann_date() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 8)]
    assert cninfo.next_trading_day(date(2024, 1, 2), days) == date(2024, 1, 3)
    assert cninfo.next_trading_day(date(2024, 1, 5), days) == date(2024, 1, 8)
    with pytest.raises(LookupError, match="no trading day"):
        cninfo.next_trading_day(date(2024, 1, 8), days)


def test_trade_calendar_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="trade_cal"):
        cninfo.load_trade_calendar_open_days(tmp_path)
    _write_trade_cal(tmp_path, [], closed_days=[date(2024, 1, 6)])
    with pytest.raises(RuntimeError, match="no open day"):
        cninfo.load_trade_calendar_open_days(tmp_path)


def test_download_writes_content_addressed_files_and_index(tmp_path: Path) -> None:
    rows = [
        _ann_row("000001.SZ", "20240102", "2023年年度报告", "https://static.cninfo.com.cn/a.pdf"),
        _ann_row("000002.SZ", "20240103", "关于对某某的问询函", "https://static.cninfo.com.cn/b.pdf"),
    ]
    _seed_data(tmp_path, rows)
    session = FakeSession(
        {
            rows[0]["url"]: [FakeResponse(200, _pdf("annual"))],
            rows[1]["url"]: [FakeResponse(200, _pdf("inquiry"))],
        }
    )

    summary = _run(tmp_path, _client(session))

    assert summary.planned == 2
    assert summary.downloaded == 2
    assert summary.failed == 0
    assert summary.skipped == 0
    assert summary.bytes_written == len(_pdf("annual")) + len(_pdf("inquiry"))
    for body in (_pdf("annual"), _pdf("inquiry")):
        digest = hashlib.sha256(body).hexdigest()
        target = tmp_path / "announcements" / "files" / digest[:2] / f"{digest}.pdf"
        assert target.read_bytes() == body

    index = pd.read_parquet(summary.index_path)
    assert list(index.columns) == list(cninfo.INDEX_COLUMNS)
    assert len(index) == 2
    by_url = index.set_index("url")
    first = by_url.loc[rows[0]["url"]]
    assert first["ts_code"] == "000001.SZ"
    assert first["ann_date"] == pd.Timestamp("2024-01-02")
    # ann_date 2024-01-02 (open) -> strictly next open day 2024-01-03
    assert first["available_at"] == pd.Timestamp("2024-01-03")
    assert first["ingested_at"] == pd.Timestamp(NOW)
    assert first["category"] == "announcement"
    assert first["bytes"] == len(_pdf("annual"))
    assert (tmp_path / first["file_path"]).is_file()
    second = by_url.loc[rows[1]["url"]]
    assert second["category"] == "regulatory_letter"
    assert second["available_at"] == pd.Timestamp("2024-01-04")

    log = pd.read_parquet(summary.log_path)
    assert set(log["status"]) == {"succeeded"}
    assert set(log["api_name"]) == {"cninfo_announcement_pdf"}
    assert log["attempts"].tolist() == [1, 1]
    assert log["bytes"].sum() == summary.bytes_written


def test_download_checkpoints_index_and_reports_live_progress(tmp_path: Path) -> None:
    rows = [
        _ann_row("000001.SZ", "20240102", "公告A", "https://static.cninfo.com.cn/a.pdf"),
        _ann_row("000002.SZ", "20240103", "公告B", "https://static.cninfo.com.cn/b.pdf"),
    ]
    _seed_data(tmp_path, rows)
    session = FakeSession(
        {row["url"]: [FakeResponse(200, _pdf(row["ts_code"]))] for row in rows}
    )
    progress: list[dict[str, int]] = []

    summary = cninfo.download_cninfo_announcements(
        tmp_path,
        client=_client(session),
        now=lambda: NOW,
        checkpoint_every=1,
        progress_callback=lambda value: progress.append(dict(value)),
    )

    assert [item["completed"] for item in progress] == [0, 1, 2]
    assert progress[-1]["downloaded"] == 2
    assert progress[-1]["failed"] == 0
    assert len(pd.read_parquet(summary.index_path)) == 2
    assert len(pd.read_parquet(summary.log_path)) == 2


def test_download_skips_existing_checksum_on_rerun(tmp_path: Path) -> None:
    rows = [
        _ann_row("000001.SZ", "20240102", "年度报告", "https://static.cninfo.com.cn/a.pdf"),
        _ann_row("000002.SZ", "20240103", "关注函", "https://static.cninfo.com.cn/b.pdf"),
    ]
    _seed_data(tmp_path, rows)
    first_session = FakeSession(
        {row["url"]: [FakeResponse(200, _pdf(row["ts_code"]))] for row in rows}
    )
    first = _run(tmp_path, _client(first_session))
    assert first.downloaded == 2
    index_before = pd.read_parquet(first.index_path)

    second_session = FakeSession({})
    second = _run(tmp_path, _client(second_session))

    assert second_session.calls == []
    assert second.downloaded == 0
    assert second.skipped == 2
    assert second.bytes_written == 0
    index_after = pd.read_parquet(second.index_path)
    pd.testing.assert_frame_equal(
        index_before.sort_values("url").reset_index(drop=True),
        index_after.sort_values("url").reset_index(drop=True),
    )
    log = pd.read_parquet(second.log_path)
    assert set(log["status"]) == {"skipped"}


def test_identical_content_from_two_urls_shares_one_file(tmp_path: Path) -> None:
    rows = [
        _ann_row("000001.SZ", "20240102", "公告A", "https://static.cninfo.com.cn/a.pdf"),
        _ann_row("000002.SZ", "20240102", "公告B", "https://static.cninfo.com.cn/b.pdf"),
    ]
    _seed_data(tmp_path, rows)
    body = _pdf("same")
    session = FakeSession({row["url"]: [FakeResponse(200, body)] for row in rows})

    summary = _run(tmp_path, _client(session))

    assert summary.downloaded == 2
    digest = hashlib.sha256(body).hexdigest()
    files = list((tmp_path / "announcements" / "files").rglob("*.pdf"))
    assert len(files) == 1
    index = pd.read_parquet(summary.index_path)
    assert set(index["file_path"]) == {f"announcements/files/{digest[:2]}/{digest}.pdf"}
    assert set(index["sha256"]) == {digest}


def test_changed_content_creates_new_file_and_never_overwrites(tmp_path: Path) -> None:
    row = _ann_row("000001.SZ", "20240102", "年度报告", "https://static.cninfo.com.cn/a.pdf")
    _seed_data(tmp_path, [row])
    body_a = _pdf("version-a")
    _run(
        tmp_path,
        _client(FakeSession({row["url"]: [FakeResponse(200, body_a)]})),
    )
    digest_a = hashlib.sha256(body_a).hexdigest()
    path_a = tmp_path / "announcements" / "files" / digest_a[:2] / f"{digest_a}.pdf"
    assert path_a.is_file()

    # Tamper with the stored file so the checksum no longer matches; the rerun
    # must write the new body as a new content-addressed file and keep the old
    # bytes untouched instead of overwriting in place.
    path_a.write_bytes(b"tampered")
    body_b = _pdf("version-b")
    second = _run(
        tmp_path,
        _client(FakeSession({row["url"]: [FakeResponse(200, body_b)]})),
    )

    assert second.downloaded == 1
    digest_b = hashlib.sha256(body_b).hexdigest()
    path_b = tmp_path / "announcements" / "files" / digest_b[:2] / f"{digest_b}.pdf"
    assert path_b.read_bytes() == body_b
    assert path_a.read_bytes() == b"tampered"
    index = pd.read_parquet(second.index_path)
    assert index.set_index("url").loc[row["url"], "sha256"] == digest_b


def test_same_size_corruption_is_not_skipped(tmp_path: Path) -> None:
    row = _ann_row(
        "000001.SZ", "20240102", "年度报告", "https://static.cninfo.com.cn/a.pdf"
    )
    _seed_data(tmp_path, [row])
    original = _pdf("version-a")
    first = _run(
        tmp_path,
        _client(FakeSession({row["url"]: [FakeResponse(200, original)]})),
    )
    index = pd.read_parquet(first.index_path).set_index("url")
    stored = tmp_path / index.loc[row["url"], "file_path"]
    stored.write_bytes(b"X" * len(original))
    replacement = _pdf("version-b")
    assert len(replacement) == len(original)
    session = FakeSession({row["url"]: [FakeResponse(200, replacement)]})

    second = _run(tmp_path, _client(session))

    assert session.calls == [row["url"]]
    assert second.skipped == 0
    assert second.downloaded == 1
    digest = hashlib.sha256(replacement).hexdigest()
    assert (
        tmp_path / "announcements" / "files" / digest[:2] / f"{digest}.pdf"
    ).read_bytes() == replacement


def test_retryable_failures_back_off_then_succeed(tmp_path: Path) -> None:
    row = _ann_row("000001.SZ", "20240102", "年度报告", "https://static.cninfo.com.cn/a.pdf")
    _seed_data(tmp_path, [row])
    session = FakeSession(
        {
            row["url"]: [
                requests.ConnectionError("connection reset"),
                FakeResponse(503, b"upstream unavailable"),
                FakeResponse(200, _pdf("ok")),
            ]
        }
    )
    delays: list[float] = []

    summary = _run(tmp_path, _client(session, max_attempts=3, delays=delays))

    assert summary.downloaded == 1
    assert len(delays) == 2
    assert all(delay > 0 for delay in delays)
    log = pd.read_parquet(summary.log_path)
    assert log.iloc[0]["status"] == "succeeded"
    assert log.iloc[0]["attempts"] == 3


def test_client_extracts_pdf_from_historical_multipart_wrapper() -> None:
    url = "http://dataclouds.cninfo.com.cn/sjother/regulatory_announcement/a.pdf"
    pdf = _pdf("historical-regulatory-letter")
    wrapped = (
        "–---------7d930d1a850658\r\n".encode()
        + b'Content-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
        + b"Content-Type: application/pdf\r\n\r\n"
        + pdf
        + b"\r\n----------7d930d1a850658--\r\n"
    )
    session = FakeSession({url: [FakeResponse(200, wrapped)]})

    body, attempts = _client(session).get(url)

    assert attempts == 1
    assert body == pdf


def test_missing_source_is_tombstoned_and_excluded_from_index(tmp_path: Path) -> None:
    row = _ann_row("000001.SZ", "20240102", "已删除公告", "https://static.cninfo.com.cn/gone.pdf")
    _seed_data(tmp_path, [row])
    session = FakeSession({row["url"]: [FakeResponse(404, b"not found")]})

    summary = _run(tmp_path, _client(session))

    assert summary.failed == 0
    assert summary.unavailable == 1
    assert summary.as_dict()["status"] == "succeeded_with_source_gaps"
    assert summary.downloaded == 0
    index = pd.read_parquet(summary.index_path)
    assert len(index) == 0
    log = pd.read_parquet(summary.log_path)
    record = log.iloc[0]
    assert record["status"] == "source_unavailable"
    assert record["attempts"] == 1
    assert "404" in record["error"]
    assert record["fetched_at"] == pd.Timestamp(NOW)
    unavailable = pd.read_parquet(summary.unavailable_path)
    assert unavailable.iloc[0]["url"] == row["url"]
    assert unavailable.iloc[0]["status_code"] == 404
    assert unavailable.iloc[0]["first_seen_at"] == pd.Timestamp(NOW)
    assert unavailable.iloc[0]["last_checked_at"] == pd.Timestamp(NOW)

    cached_session = FakeSession({})
    cached = _run(tmp_path, _client(cached_session))

    assert cached_session.calls == []
    assert cached.failed == 0
    assert cached.unavailable == 1
    assert cached.downloaded == 0
    cached_log = pd.read_parquet(cached.log_path)
    assert cached_log.iloc[0]["status"] == "source_unavailable"
    assert cached_log.iloc[0]["attempts"] == 0
    assert "cached HTTP 404" in cached_log.iloc[0]["error"]

    recovered_session = FakeSession({row["url"]: [FakeResponse(200, _pdf("restored"))]})
    recovered = cninfo.download_cninfo_announcements(
        tmp_path,
        client=_client(recovered_session),
        now=lambda: NOW + timedelta(days=31),
    )

    assert recovered_session.calls == [row["url"]]
    assert recovered.downloaded == 1
    assert recovered.unavailable == 0
    assert len(pd.read_parquet(recovered.index_path)) == 1
    assert len(pd.read_parquet(recovered.unavailable_path)) == 0


def test_quality_audit_certifies_files_pit_and_explicit_source_gaps(
    tmp_path: Path,
) -> None:
    rows = [
        _ann_row(
            "000001.SZ",
            "20240102",
            "年度报告",
            "https://static.cninfo.com.cn/a.pdf",
        ),
        _ann_row(
            "000002.SZ",
            "20240103",
            "关注函",
            "https://static.cninfo.com.cn/gone.pdf",
        ),
    ]
    _seed_data(tmp_path, rows)
    session = FakeSession(
        {
            rows[0]["url"]: [FakeResponse(200, _pdf("audited"))],
            rows[1]["url"]: [FakeResponse(404, b"not found")],
        }
    )
    summary = _run(tmp_path, _client(session))

    report = cninfo.audit_cninfo_announcements(
        tmp_path,
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        now=lambda: NOW,
    )

    assert summary.failed == 0
    assert report["ok"] is True
    assert report["coverage"] == {
        "planned": 2,
        "indexed": 1,
        "source_unavailable": 1,
        "missing": 0,
        "ratio": 1.0,
        "missing_samples": [],
        "source_unavailable_samples": [rows[1]["url"]],
    }
    assert report["governance"]["pit_mismatches"] == 0
    assert report["integrity"]["verified_files"] == 1
    assert report["artifacts"]["index_sha256"]
    assert report["report_sha256"]
    report_path = Path(report["report_path"])
    assert report_path.is_file()
    assert (report_path.parent / "latest.json").read_bytes() == report_path.read_bytes()


def test_quality_audit_fails_closed_on_same_size_file_tamper_and_pit_drift(
    tmp_path: Path,
) -> None:
    row = _ann_row(
        "000001.SZ",
        "20240102",
        "年度报告",
        "https://static.cninfo.com.cn/a.pdf",
    )
    _seed_data(tmp_path, [row])
    summary = _run(
        tmp_path,
        _client(FakeSession({row["url"]: [FakeResponse(200, _pdf("original"))]})),
    )
    index = pd.read_parquet(summary.index_path)
    target = tmp_path / index.iloc[0]["file_path"]
    original = target.read_bytes()
    target.write_bytes(original[:-1] + (b"X" if original[-1:] != b"X" else b"Y"))
    index.loc[0, "available_at"] = pd.Timestamp("2024-01-05")
    index.to_parquet(summary.index_path, index=False, compression="zstd", engine="pyarrow")

    report = cninfo.audit_cninfo_announcements(
        tmp_path,
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        now=lambda: NOW,
    )

    assert report["ok"] is False
    assert report["governance"]["pit_mismatches"] == 1
    assert report["integrity"]["sha256_mismatches"] == 1
    assert any("PIT metadata" in error for error in report["errors"])
    assert any("checksum" in error for error in report["errors"])


def test_non_pdf_body_for_pdf_url_is_terminal(tmp_path: Path) -> None:
    row = _ann_row("000001.SZ", "20240102", "异常响应", "https://static.cninfo.com.cn/a.pdf")
    _seed_data(tmp_path, [row])
    session = FakeSession({row["url"]: [FakeResponse(200, b"<html>error</html>")]})

    summary = _run(tmp_path, _client(session))

    assert summary.failed == 1
    assert len(session.calls) == 1
    log = pd.read_parquet(summary.log_path)
    assert "not a PDF" in log.iloc[0]["error"]
    assert list((tmp_path / "announcements" / "files").rglob("*.pdf")) == []


def test_available_at_fail_closed_per_record_when_calendar_ends(tmp_path: Path) -> None:
    rows = [
        _ann_row("000001.SZ", "20240102", "正常公告", "https://static.cninfo.com.cn/a.pdf"),
        _ann_row("000002.SZ", "20240105", "日历外公告", "https://static.cninfo.com.cn/b.pdf"),
    ]
    _seed_data(tmp_path, rows)
    session = FakeSession({rows[0]["url"]: [FakeResponse(200, _pdf("ok"))]})

    summary = _run(tmp_path, _client(session))

    assert summary.downloaded == 1
    assert summary.failed == 1
    log = pd.read_parquet(summary.log_path).set_index("url")
    assert "no trading day" in log.loc[rows[1]["url"], "error"]
    index = pd.read_parquet(summary.index_path)
    assert index["url"].tolist() == [rows[0]["url"]]


def test_empty_selection_needs_no_calendar(tmp_path: Path) -> None:
    _write_anns_d(
        tmp_path,
        [_ann_row("000001.SZ", "20240102", "年度报告", "https://static.cninfo.com.cn/a.pdf")],
    )
    session = FakeSession({})

    summary = _run(tmp_path, _client(session), ts_codes={"999999.SZ"})

    assert summary.planned == 0
    assert summary.failed == 0
    assert summary.log_path is None


def test_cli_downloads_with_filters_and_limit(tmp_path: Path, monkeypatch) -> None:
    rows = [
        _ann_row("000001.SZ", "20240102", "问询函", "https://static.cninfo.com.cn/a.pdf"),
        _ann_row("000002.SZ", "20240103", "年度报告", "https://static.cninfo.com.cn/b.pdf"),
    ]
    _seed_data(tmp_path, rows)
    client = _client(FakeSession({rows[0]["url"]: [FakeResponse(200, _pdf("cli"))]}))
    monkeypatch.setattr(cninfo, "CninfoHttpClient", lambda **kwargs: client)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("REQUESTS_PER_MINUTE", "60000")

    result = CliRunner().invoke(
        app,
        [
            "cninfo-announcements",
            "--ts-code",
            "000001.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "succeeded"
    assert payload["planned"] == 1
    assert payload["downloaded"] == 1
    assert payload["quality_gate"]["ok"] is True
    assert Path(payload["quality_report_path"]).is_file()
    index = pd.read_parquet(tmp_path / "announcements" / "index.parquet")
    assert index["url"].tolist() == [rows[0]["url"]]
    assert index.iloc[0]["category"] == "regulatory_letter"


def test_cli_exits_nonzero_for_genuine_download_failure(tmp_path: Path, monkeypatch) -> None:
    row = _ann_row(
        "000001.SZ", "20240102", "异常响应", "https://static.cninfo.com.cn/a.pdf"
    )
    _seed_data(tmp_path, [row])
    client = _client(FakeSession({row["url"]: [FakeResponse(200, b"<html>error</html>")]}))
    monkeypatch.setattr(cninfo, "CninfoHttpClient", lambda **kwargs: client)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("REQUESTS_PER_MINUTE", "60000")

    result = CliRunner().invoke(
        app,
        [
            "cninfo-announcements",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ],
    )

    assert result.exit_code == 3, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["failed"] == 1
    assert payload["unavailable"] == 0


def test_resolve_pdf_url_maps_detail_page_to_static_pdf() -> None:
    detail = (
        "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=000858"
        "&announcementId=1225343261&orgId=gssz0000858&announcementTime=2026-06-02"
    )
    assert cninfo.resolve_pdf_url(detail, date(2026, 6, 2)) == (
        "http://static.cninfo.com.cn/finalpage/2026-06-02/1225343261.PDF"
    )


def test_resolve_pdf_url_fallbacks_and_passthrough() -> None:
    no_time = "http://www.cninfo.com.cn/new/disclosure/detail?announcementId=1225343261"
    assert cninfo.resolve_pdf_url(no_time, date(2026, 6, 3)) == (
        "http://static.cninfo.com.cn/finalpage/2026-06-03/1225343261.PDF"
    )
    direct = "http://static.cninfo.com.cn/finalpage/2026-06-02/1225343261.PDF"
    assert cninfo.resolve_pdf_url(direct, date(2026, 6, 2)) == direct
    external = "https://example.com/x.pdf"
    assert cninfo.resolve_pdf_url(external, date(2026, 6, 2)) == external
    no_id = "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=000858"
    assert cninfo.resolve_pdf_url(no_id, date(2026, 6, 2)) == no_id


def test_manifest_resolves_detail_page_urls(tmp_path: Path) -> None:
    detail = (
        "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=000858"
        "&announcementId=1225343261&orgId=gssz0000858&announcementTime=2026-06-02"
    )
    _seed_data(tmp_path, [_ann_row("000858.SZ", "20240102", "年度股东会议案", detail)])
    manifest = cninfo.load_announcement_manifest(tmp_path)
    assert [ref.url for ref in manifest] == [
        "http://static.cninfo.com.cn/finalpage/2026-06-02/1225343261.PDF"
    ]
