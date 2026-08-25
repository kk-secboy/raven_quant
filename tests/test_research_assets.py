from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

import quant_data.research_assets as research_assets_module
from quant_data.catalog import ALL_DEFINITIONS, RESEARCH_REPORT_FIELDS
from quant_data.cli import (
    RESEARCH_ASSET_SNAPSHOT_PROFILE,
    _profile_datasets,
    _required_profile_datasets,
    _snapshot_datasets,
)
from quant_data.history_bounds import history_start_date
from quant_data.research_assets import (
    ArxivDiscoveryClient,
    ResearchAssetCandidate,
    ResearchAssetConflictError,
    ResearchAssetDownloadError,
    ResearchAssetNotYetAvailable,
    SafeHttpClient,
    SecurePdfDownloader,
    UnsafeResearchAssetUrl,
    VerifiedPdf,
    acquire_manual_https_pdf,
    acquire_research_assets,
    collected_source_ids,
    import_research_asset_manifests,
    ingest_research_assets,
    load_research_asset_manifest,
    manual_https_pdf_candidate,
    materialize_research_asset,
    parse_arxiv_atom,
    rank_arxiv_candidates,
    register_local_research_asset,
    research_asset_root,
    select_tushare_research_reports,
    validate_asset_id,
    validate_public_https_url,
)
from quant_data.snapshot_lineage import make_lineage_id
from quant_data.supplemental_data import supplemental_specs

pytestmark = pytest.mark.no_database

_PUBLIC_IP = "93.184.216.34"
_PDF_BODY = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n"


class _FakeSocket:
    def getpeername(self) -> tuple[str, int]:
        return (_PUBLIC_IP, 443)


class _FakeConnection:
    sock = _FakeSocket()


class _FakeRaw:
    _connection = _FakeConnection()


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.body = body
        self.closed = False
        self.raw = _FakeRaw()

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [
            self.body[offset : offset + chunk_size]
            for offset in range(0, len(self.body), chunk_size)
        ]

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = responses
        self.urls: list[str] = []
        self.trust_env = True

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.urls.append(url)
        assert kwargs["allow_redirects"] is False
        assert kwargs["stream"] is True
        return self.responses.pop(0)


def _resolver(host: str, port: int) -> list[str]:
    assert port == 443
    return [_PUBLIC_IP]


def test_research_report_contract_is_clipped_to_official_history() -> None:
    assert history_start_date("research_report") == date(2017, 1, 1)
    definition = ALL_DEFINITIONS["research_report"]
    assert definition.fields == RESEARCH_REPORT_FIELDS
    assert definition.primary_key == ("url",)
    specs = supplemental_specs(
        "research_corpus",
        start=date(2016, 12, 30),
        end=date(2017, 1, 2),
        trading_dates=("20161230", "20170102"),
        max_attempts=2,
    )
    reports = [spec for spec in specs if spec.dataset == "research_report"]
    assert [spec.params["trade_date"] for spec in reports] == ["20170101", "20170102"]
    assert all(spec.fields == RESEARCH_REPORT_FIELDS for spec in reports)


def test_research_report_snapshot_profile_is_isolated_from_qlib_full() -> None:
    assert "research_report" not in _profile_datasets("full")
    expected = {"trade_cal", "research_report"}
    assert _profile_datasets(RESEARCH_ASSET_SNAPSHOT_PROFILE) == expected
    assert _snapshot_datasets(
        RESEARCH_ASSET_SNAPSHOT_PROFILE,
        available=expected,
    ) == expected
    assert set(_required_profile_datasets(RESEARCH_ASSET_SNAPSHOT_PROFILE)) == expected


def test_public_https_url_rejects_non_https_credentials_and_private_dns() -> None:
    assert (
        validate_public_https_url("https://example.com/paper.pdf", resolver=_resolver)
        == "https://example.com/paper.pdf"
    )
    with pytest.raises(UnsafeResearchAssetUrl, match="only HTTPS"):
        validate_public_https_url("http://example.com/paper.pdf", resolver=_resolver)
    with pytest.raises(UnsafeResearchAssetUrl, match="credentials"):
        validate_public_https_url(
            "https://user:secret@example.com/paper.pdf",
            resolver=_resolver,
        )
    with pytest.raises(UnsafeResearchAssetUrl, match="backslash"):
        validate_public_https_url(
            "https://example.com\\@127.0.0.1/paper.pdf",
            resolver=_resolver,
        )
    with pytest.raises(UnsafeResearchAssetUrl, match="non-public"):
        validate_public_https_url(
            "https://example.com/paper.pdf",
            resolver=lambda _host, _port: ["127.0.0.1"],
        )


def test_secure_pdf_download_checks_redirect_mime_magic_and_sha() -> None:
    redirect = _FakeResponse(302, headers={"Location": "/files/paper.pdf"})
    pdf_response = _FakeResponse(
        200,
        headers={
            "Content-Type": "application/pdf; charset=binary",
            "Content-Length": str(len(_PDF_BODY)),
        },
        body=_PDF_BODY,
    )
    session = _FakeSession([redirect, pdf_response])
    downloader = SecurePdfDownloader(
        http_client=SafeHttpClient(session=session, resolver=_resolver)
    )
    result = downloader.download("https://example.com/report")
    assert result.final_url == "https://example.com/files/paper.pdf"
    assert result.redirect_chain == ("https://example.com/files/paper.pdf",)
    assert result.sha256 == hashlib.sha256(_PDF_BODY).hexdigest()
    assert session.trust_env is False
    assert redirect.closed and pdf_response.closed

    bad_mime = SecurePdfDownloader(
        http_client=SafeHttpClient(
            session=_FakeSession(
                [_FakeResponse(200, headers={"Content-Type": "text/html"}, body=_PDF_BODY)]
            ),
            resolver=_resolver,
        )
    )
    with pytest.raises(ResearchAssetDownloadError, match="Content-Type"):
        bad_mime.download("https://example.com/report.pdf")

    bad_magic = SecurePdfDownloader(
        http_client=SafeHttpClient(
            session=_FakeSession(
                [
                    _FakeResponse(
                        200,
                        headers={"Content-Type": "application/pdf"},
                        body=b"not-a-pdf",
                    )
                ]
            ),
            resolver=_resolver,
        )
    )
    with pytest.raises(ResearchAssetDownloadError, match="does not start"):
        bad_magic.download("https://example.com/report.pdf")


def test_redirect_target_is_revalidated_against_ssrf() -> None:
    session = _FakeSession(
        [_FakeResponse(302, headers={"Location": "https://internal.example/paper.pdf"})]
    )

    def resolver(host: str, _port: int) -> list[str]:
        return ["10.0.0.8"] if host == "internal.example" else [_PUBLIC_IP]

    downloader = SecurePdfDownloader(
        http_client=SafeHttpClient(session=session, resolver=resolver)
    )
    with pytest.raises(UnsafeResearchAssetUrl, match="non-public"):
        downloader.download("https://example.com/report")


def test_tushare_selection_caps_each_day_and_uses_next_real_open_day() -> None:
    rows = [
        {
            "trade_date": "20260814",
            "title": f"report {index:02d}",
            "url": f"https://research.example/{index}.pdf",
            "ts_code": f"{index:06d}.SZ",
            "abstr": "complete abstract",
        }
        for index in range(21)
    ]
    selected = select_tushare_research_reports(
        rows,
        open_days=(date(2026, 8, 14), date(2026, 8, 17), date(2026, 8, 18)),
    )
    assert len(selected) == 20
    assert {candidate.selection_rank for candidate in selected} == set(range(1, 21))
    assert {candidate.available_at.date() for candidate in selected} == {date(2026, 8, 17)}
    assert all(candidate.available_at > candidate.published_at for candidate in selected)


def test_tushare_asset_is_not_downloaded_before_next_open_day(tmp_path: Path) -> None:
    candidate = select_tushare_research_reports(
        [
            {
                "trade_date": "20260814",
                "title": "report",
                "url": "https://research.example/report.pdf",
            }
        ],
        open_days=(date(2026, 8, 14), date(2026, 8, 17)),
    )[0]

    class Downloader:
        called = False

        def download(self, url: str, *, allowed_hosts: frozenset[str] | None = None) -> VerifiedPdf:
            self.called = True
            return _verified_pdf(_PDF_BODY, url=url)

    downloader = Downloader()
    with pytest.raises(ResearchAssetNotYetAvailable, match="not available until"):
        acquire_research_assets(
            [candidate],
            data_root=tmp_path,
            downloader=downloader,  # type: ignore[arg-type]
            clock=lambda: datetime(2026, 8, 14, 12, tzinfo=UTC),
        )
    assert downloader.called is False


def test_arxiv_atom_is_filtered_and_ranked_to_three() -> None:
    entries = "".join(
        _atom_entry(index, category)
        for index, category in enumerate(
            ("q-fin.ST", "cs.LG", "stat.ML", "cs.SE"),
            start=1,
        )
    )
    body = (
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        f"{entries}"
        "</feed>"
    ).encode()
    candidates = parse_arxiv_atom(body)
    ranked = rank_arxiv_candidates(candidates, as_of=date(2026, 8, 14))
    assert len(ranked) == 3
    assert ranked[0].categories == ("q-fin.ST",)
    assert [candidate.selection_rank for candidate in ranked] == [1, 2, 3]
    assert all(candidate.pdf_url.startswith("https://arxiv.org/pdf/") for candidate in ranked)
    repeated = rank_arxiv_candidates(
        candidates,
        as_of=date(2026, 8, 14),
        excluded_source_ids={candidate.source_id for candidate in ranked},
    )
    assert repeated == []


def test_arxiv_discovery_enforces_official_host_and_atom_mime() -> None:
    body = (
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        f"{_atom_entry(1, 'q-fin.MF')}"
        "</feed>"
    ).encode()
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                headers={"Content-Type": "application/atom+xml"},
                body=body,
            )
            for _ in range(3)
        ]
    )
    client = ArxivDiscoveryClient(
        http_client=SafeHttpClient(session=session, resolver=_resolver),
        sleeper=lambda _seconds: None,
    )
    selected = client.discover(date(2026, 8, 14))
    assert len(selected) == 1
    assert len(session.urls) == 3
    assert all(url.startswith("https://export.arxiv.org/api/query?") for url in session.urls)


def test_materialization_is_fixed_path_verified_idempotent_and_importable(
    tmp_path: Path,
) -> None:
    acquired_at = datetime(2026, 8, 14, 8, tzinfo=UTC)
    candidate = manual_https_pdf_candidate(
        url="https://example.com/research.pdf",
        title="Manual research",
        acquired_at=acquired_at,
    )
    pdf = _verified_pdf(_PDF_BODY)
    published = materialize_research_asset(
        tmp_path,
        candidate,
        pdf,
        acquired_at=acquired_at,
        asset_id="manual-test",
    )
    expected = research_asset_root(tmp_path) / "manual-test"
    assert published.directory == expected
    assert published.manifest_path == expected / "manifest.json"
    assert published.content_path == expected / "content.pdf"
    assert (expected / "manifest.sha256").is_file()
    loaded = load_research_asset_manifest(published.manifest_path)
    assert loaded["asset_id"] == "manual-test"
    assert (loaded["kind"], loaded["type"], loaded["status"]) == (
        "pdf",
        "manual_pdf",
        "ready",
    )
    assert loaded["files"] == [
        {
            "path": "content.pdf",
            "media_type": "application/pdf",
            "bytes": len(_PDF_BODY),
            "sha256": hashlib.sha256(_PDF_BODY).hexdigest(),
        }
    ]

    repeated = materialize_research_asset(
        tmp_path,
        candidate,
        pdf,
        acquired_at=datetime(2026, 8, 14, 9, tzinfo=UTC),
        asset_id="manual-test",
    )
    assert repeated.manifest["acquired_at"] == acquired_at.isoformat()

    importer = _RecordingImporter()
    assert import_research_asset_manifests(tmp_path, importer) == ["manual-test"]
    assert importer.paths == [published.manifest_path]
    assert collected_source_ids(tmp_path, "manual_https") == {candidate.source_id}

    with pytest.raises(ResearchAssetConflictError, match="conflicts"):
        materialize_research_asset(
            tmp_path,
            candidate,
            _verified_pdf(b"%PDF-1.7\nchanged\n%%EOF\n"),
            acquired_at=acquired_at,
            asset_id="manual-test",
        )
    with pytest.raises(ValueError, match="asset_id"):
        validate_asset_id("../escape")
    with pytest.raises(ValueError, match="portable"):
        validate_asset_id("con")


def test_manual_acquisition_uses_verification_clock_not_backfilled_time(tmp_path: Path) -> None:
    claimed_at = datetime(2020, 1, 1, tzinfo=UTC)
    verified_at = datetime(2026, 8, 14, 9, tzinfo=UTC)
    candidate = manual_https_pdf_candidate(
        url="https://example.com/manual.pdf",
        title="Manual",
        acquired_at=claimed_at,
    )
    with pytest.raises(ValueError, match="local verification"):
        materialize_research_asset(
            tmp_path,
            candidate,
            _verified_pdf(_PDF_BODY),
            acquired_at=verified_at,
        )

    class Downloader:
        def download(self, url: str, *, allowed_hosts: frozenset[str] | None = None) -> VerifiedPdf:
            assert url == candidate.pdf_url
            assert allowed_hosts is None
            return _verified_pdf(_PDF_BODY, url=url)

    published = acquire_research_assets(
        [candidate],
        data_root=tmp_path,
        downloader=Downloader(),  # type: ignore[arg-type]
        acquired_at=claimed_at,
        clock=lambda: verified_at,
    )[0]
    assert published.manifest["available_at"] == verified_at.isoformat()
    assert published.manifest["acquired_at"] == verified_at.isoformat()


def test_direct_materialization_rechecks_governed_size_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquired_at = datetime(2026, 8, 14, tzinfo=UTC)
    candidate = manual_https_pdf_candidate(
        url="https://example.com/oversize.pdf",
        title="Oversize",
        acquired_at=acquired_at,
    )
    monkeypatch.setattr(research_assets_module, "DEFAULT_PDF_MAX_BYTES", len(_PDF_BODY) - 1)
    with pytest.raises(ResearchAssetDownloadError, match="size limit"):
        materialize_research_asset(
            tmp_path,
            candidate,
            _verified_pdf(_PDF_BODY),
            acquired_at=acquired_at,
        )


def test_automatic_manifest_priority_prefers_newer_availability(tmp_path: Path) -> None:
    acquired_at = datetime(2026, 8, 14, 8, tzinfo=UTC)

    def publish(source_id: str, available_at: datetime) -> dict[str, object]:
        candidate = ResearchAssetCandidate(
            source_kind="arxiv",
            source_id=source_id,
            title=source_id,
            pdf_url=f"https://arxiv.org/pdf/{source_id}",
            published_at=available_at,
            available_at=available_at,
            categories=("q-fin.ST",),
            selection_rank=1,
            selection_as_of=available_at.date(),
        )
        return dict(
            materialize_research_asset(
                tmp_path,
                candidate,
                _verified_pdf(_PDF_BODY, url=candidate.pdf_url),
                acquired_at=acquired_at,
            ).manifest
        )

    older = publish("2608.00001v1", datetime(2026, 8, 13, 8, tzinfo=UTC))
    newer = publish("2608.00002v1", acquired_at)

    assert int(newer["priority"]) > int(older["priority"])
    assert newer["selection"]["novelty_at"] == acquired_at.isoformat()  # type: ignore[index]


def test_ingestion_reads_verified_snapshot_and_records_pdf_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_research_snapshot(
        tmp_path,
        research_rows=[
            {
                "trade_date": "20260814",
                "title": "report",
                "url": "https://research.example/report.pdf",
            }
        ],
    )

    class FailingDownloader:
        def download(self, url: str, *, allowed_hosts: frozenset[str] | None = None) -> VerifiedPdf:
            raise ResearchAssetDownloadError(f"unavailable: {url}")

    attempted_at = datetime(2026, 8, 17, 9, tzinfo=UTC)
    summary = ingest_research_assets(
        tmp_path,
        snapshot_name="snapshot-v1",
        as_of=date(2026, 8, 17),
        include_arxiv=False,
        downloader=FailingDownloader(),  # type: ignore[arg-type]
        clock=lambda: attempted_at,
    )
    assert summary.tushare_selected == 1
    assert summary.published == ()
    assert summary.failed == 1
    assert summary.blocked[0]["reason_code"] == "pdf_unavailable_or_unsafe"
    lines = summary.blocked_ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    blocked = json.loads(lines[0])
    assert blocked["snapshot_name"] == "snapshot-v1"
    assert blocked["source_id"] == "https://research.example/report.pdf"
    assert list(research_asset_root(tmp_path).glob("*/manifest.json")) == []


def test_ingestion_records_missing_tushare_permission_without_fake_asset(
    tmp_path: Path,
) -> None:
    _write_research_snapshot(tmp_path, research_rows=None)
    summary = ingest_research_assets(
        tmp_path,
        snapshot_name="snapshot-v1",
        as_of=date(2026, 8, 14),
        include_arxiv=False,
        clock=lambda: datetime(2026, 8, 17, tzinfo=UTC),
    )
    assert summary.failed == 1
    assert summary.blocked[0]["reason_code"] == "source_permission_or_dataset_unavailable"
    assert summary.blocked[0]["severity"] == "error"
    assert list(research_asset_root(tmp_path).glob("*/manifest.json")) == []


def test_explicit_tushare_report_date_must_be_eligible_by_as_of(
    tmp_path: Path,
) -> None:
    _write_research_snapshot(
        tmp_path,
        research_rows=[
            {
                "trade_date": "20260814",
                "title": "Friday report",
                "url": "https://research.example/friday.pdf",
            }
        ],
    )

    with pytest.raises(ValueError, match="not PIT-eligible"):
        ingest_research_assets(
            tmp_path,
            snapshot_name="snapshot-v1",
            as_of=date(2026, 8, 14),
            include_arxiv=False,
            tushare_report_date=date(2026, 8, 14),
        )


def test_ingestion_rejects_snapshot_without_passing_quality_gate(tmp_path: Path) -> None:
    _write_research_snapshot(tmp_path, research_rows=None)
    manifest_path = tmp_path / "snapshots" / "snapshot-v1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["quality_gate"] = {"ok": False}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="passing snapshot quality gate"):
        ingest_research_assets(
            tmp_path,
            snapshot_name="snapshot-v1",
            as_of=date(2026, 8, 14),
            include_arxiv=False,
        )


def test_ingestion_rejects_non_research_asset_snapshot_profile(tmp_path: Path) -> None:
    _write_research_snapshot(tmp_path, research_rows=None)
    manifest_path = tmp_path / "snapshots" / "snapshot-v1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["profile"] = "full"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="isolated research-assets snapshot profile"):
        ingest_research_assets(
            tmp_path,
            snapshot_name="snapshot-v1",
            as_of=date(2026, 8, 14),
            include_arxiv=False,
        )


def test_ingestion_excludes_existing_arxiv_sources_and_calls_importer(
    tmp_path: Path,
) -> None:
    acquired_at = datetime(2026, 8, 14, 8, tzinfo=UTC)
    existing = ResearchAssetCandidate(
        source_kind="arxiv",
        source_id="2608.00001v1",
        title="existing",
        pdf_url="https://arxiv.org/pdf/2608.00001v1",
        published_at=acquired_at,
        available_at=acquired_at,
        categories=("q-fin.ST",),
        asset_type="arxiv_paper",
    )
    materialize_research_asset(
        tmp_path,
        existing,
        _verified_pdf(_PDF_BODY, url=existing.pdf_url),
        acquired_at=acquired_at,
    )

    class Discovery:
        excluded: set[str] = set()

        def discover(
            self,
            as_of: date,
            *,
            daily_limit: int,
            excluded_source_ids: set[str],
        ) -> list[ResearchAssetCandidate]:
            assert daily_limit == 3
            self.excluded = set(excluded_source_ids)
            return [
                ResearchAssetCandidate(
                    source_kind="arxiv",
                    source_id="2608.00002v1",
                    title="new",
                    pdf_url="https://arxiv.org/pdf/2608.00002v1",
                    published_at=acquired_at,
                    available_at=acquired_at,
                    categories=("stat.ML",),
                    asset_type="arxiv_paper",
                )
            ]

    class Downloader:
        def download(self, url: str, *, allowed_hosts: frozenset[str] | None = None) -> VerifiedPdf:
            return _verified_pdf(_PDF_BODY, url=url)

    discovery = Discovery()
    importer = _RecordingImporter()
    summary = ingest_research_assets(
        tmp_path,
        snapshot_name="snapshot-v1",
        as_of=date(2026, 8, 14),
        include_tushare=False,
        arxiv_client=discovery,  # type: ignore[arg-type]
        downloader=Downloader(),  # type: ignore[arg-type]
        importer=importer,
        clock=lambda: acquired_at,
    )
    assert discovery.excluded == {"2608.00001v1"}
    assert summary.arxiv_selected == 1
    assert len(summary.published) == 1
    assert len(summary.imported) == 1
    assert importer.paths == [summary.published[0].manifest_path]


def test_arxiv_daily_limit_is_durable_across_failed_retries(tmp_path: Path) -> None:
    _write_research_snapshot(tmp_path, research_rows=None)
    acquired_at = datetime(2026, 8, 14, 8, tzinfo=UTC)

    class Discovery:
        calls = 0

        def discover(
            self,
            as_of: date,
            *,
            daily_limit: int,
            excluded_source_ids: set[str],
        ) -> list[ResearchAssetCandidate]:
            self.calls += 1
            assert daily_limit == 3
            assert excluded_source_ids == set()
            return [
                ResearchAssetCandidate(
                    source_kind="arxiv",
                    source_id=f"2608.1000{index}v1",
                    title=f"candidate {index}",
                    pdf_url=f"https://arxiv.org/pdf/2608.1000{index}v1",
                    published_at=acquired_at,
                    available_at=acquired_at,
                    categories=("q-fin.ST",),
                    asset_type="arxiv_paper",
                    selection_as_of=as_of,
                    selection_rank=index,
                    selection_daily_limit=daily_limit,
                )
                for index in range(1, daily_limit + 1)
            ]

    class FailingDownloader:
        def download(
            self,
            url: str,
            *,
            allowed_hosts: frozenset[str] | None = None,
        ) -> VerifiedPdf:
            raise ResearchAssetDownloadError(f"blocked: {url}")

    discovery = Discovery()
    first = ingest_research_assets(
        tmp_path,
        snapshot_name="snapshot-v1",
        as_of=date(2026, 8, 14),
        include_tushare=False,
        arxiv_client=discovery,  # type: ignore[arg-type]
        downloader=FailingDownloader(),  # type: ignore[arg-type]
        clock=lambda: acquired_at,
    )
    second = ingest_research_assets(
        tmp_path,
        snapshot_name="snapshot-v1",
        as_of=date(2026, 8, 14),
        include_tushare=False,
        arxiv_client=discovery,  # type: ignore[arg-type]
        downloader=FailingDownloader(),  # type: ignore[arg-type]
        clock=lambda: acquired_at,
    )

    assert first.arxiv_selected == 3
    assert first.failed == 3
    assert second.arxiv_selected == 0
    assert discovery.calls == 1


def test_ingestion_records_tushare_rows_without_pdf_url_as_blocked(
    tmp_path: Path,
) -> None:
    _write_research_snapshot(
        tmp_path,
        research_rows=[
            {
                "trade_date": "20260814",
                "title": "Metadata only report",
                "url": None,
            }
        ],
    )
    attempted_at = datetime(2026, 8, 17, 8, tzinfo=UTC)

    summary = ingest_research_assets(
        tmp_path,
        snapshot_name="snapshot-v1",
        as_of=date(2026, 8, 17),
        include_arxiv=False,
        clock=lambda: attempted_at,
    )

    assert summary.published == ()
    assert summary.tushare_selected == 0
    assert [item["reason_code"] for item in summary.blocked] == [
        "pdf_url_unavailable"
    ]
    assert summary.failed == 1
    assert summary.blocked[0]["severity"] == "error"
    assert not any(path.is_dir() for path in research_asset_root(tmp_path).iterdir())


def test_local_pdf_registration_copies_and_seals_content(tmp_path: Path) -> None:
    source = tmp_path / "upload.pdf"
    source.write_bytes(_PDF_BODY)
    acquired_at = datetime(2026, 8, 14, 8, tzinfo=UTC)

    published = register_local_research_asset(
        tmp_path / "data-root",
        asset_id="manual-upload-1",
        kind="pdf",
        source_path=source,
        asset_type="manual_pdf",
        clock=lambda: acquired_at,
    )
    source.write_bytes(b"%PDF-1.7\nchanged\n%%EOF\n")

    manifest = load_research_asset_manifest(published.manifest_path)
    assert published.content_path.read_bytes() == _PDF_BODY
    assert manifest["kind"] == "pdf"
    assert manifest["source"] == {
        "kind": "admin_local_copy",
        "source_id": "manual-upload-1",
    }
    with pytest.raises(ResearchAssetConflictError, match="conflicts"):
        register_local_research_asset(
            tmp_path / "data-root",
            asset_id="manual-upload-1",
            kind="pdf",
            source_path=source,
            asset_type="manual_pdf",
            clock=lambda: acquired_at,
        )


def test_local_dataset_registration_preserves_relative_files(tmp_path: Path) -> None:
    source = tmp_path / "competition"
    (source / "train").mkdir(parents=True)
    (source / "train" / "data.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (source / "README.md").write_text("rules", encoding="utf-8")

    published = register_local_research_asset(
        tmp_path / "data-root",
        asset_id="competition-1",
        kind="dataset",
        source_path=source,
        asset_type="competition_dataset",
        clock=lambda: datetime(2026, 8, 14, 8, tzinfo=UTC),
    )
    manifest = load_research_asset_manifest(published.manifest_path)

    assert manifest["kind"] == "dataset"
    assert [entry["path"] for entry in manifest["files"]] == [
        "README.md",
        "train/data.csv",
    ]


def _write_finetune_bundle(
    source: Path,
    *,
    benchmark: str = "mmlu",
) -> dict[str, object]:
    (source / "models" / "base-v1").mkdir(parents=True)
    (source / "datasets" / "finance-v1").mkdir(parents=True)
    (source / "benchmarks").mkdir()
    (source / ".llama_factory_info").mkdir()
    (source / "models" / "base-v1" / "weights.safetensors").write_bytes(b"weights")
    (source / "datasets" / "finance-v1" / "train.jsonl").write_text(
        '{"messages": []}\n', encoding="utf-8"
    )
    (source / "datasets" / "dataset_info.json").write_text(
        json.dumps(
            {
                "finance-v1": {
                    "total_samples": 1,
                    "tasks": {"_root": {"files": ["train.jsonl"]}},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if benchmark == "FinanceIQ_gen":
        financeiq = source / "benchmarks" / "opencompass_data" / "data" / "FinanceIQ"
        (financeiq / "dev").mkdir(parents=True)
        (financeiq / "test").mkdir()
        (financeiq / "dev" / "few_shot.csv").write_text("q,a\n1,1\n", encoding="utf-8")
        (financeiq / "test" / "questions.csv").write_text("q,a\n2,2\n", encoding="utf-8")
    else:
        benchmark_root = source / "benchmarks" / "opencompass_data" / "data" / benchmark
        benchmark_root.mkdir(parents=True)
        (benchmark_root / "data.json").write_text("{}\n", encoding="utf-8")
    (source / ".llama_factory_info" / "constants.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (source / ".llama_factory_info" / "parameters.json").write_text(
        "{}\n", encoding="utf-8"
    )
    revisions = {"model": "c" * 40, "dataset": "d" * 40, "benchmark": "e" * 40}
    terms = {
        "model": b"Apache License 2.0 terms\n",
        "dataset": b"Creative Commons Attribution 4.0 terms\n",
        "benchmark": b"Benchmark evaluation terms\n",
    }
    for subject in ("model", "dataset", "benchmark"):
        governance = source / "governance" / subject
        governance.mkdir(parents=True)
        (governance / "revision.txt").write_text(
            revisions[subject] + "\n", encoding="ascii"
        )
        (governance / "license-terms.txt").write_bytes(terms[subject])
    metadata: dict[str, object] = {
        "benchmark": benchmark,
        "benchmark_description": "A governed finance benchmark.",
        "dataset": "finance-v1",
        "base_model": "base-v1",
    }
    licenses = {
        "model": "Apache License 2.0",
        "dataset": "Creative Commons Attribution 4.0",
        "benchmark": "Benchmark research license",
    }
    for subject in ("model", "dataset", "benchmark"):
        metadata.update(
            {
                f"{subject}_revision": revisions[subject],
                f"{subject}_license": licenses[subject],
                f"{subject}_license_terms_sha256": hashlib.sha256(
                    terms[subject]
                ).hexdigest(),
                f"{subject}_license_accepted": True,
                f"{subject}_license_accepted_by": "admin@example.com",
                f"{subject}_license_accepted_at": "2026-08-14T07:00:00+00:00",
            }
        )
    return metadata


def test_local_finetune_registration_requires_runtime_metadata(tmp_path: Path) -> None:
    source = tmp_path / "finetune-bundle"
    valid_metadata = _write_finetune_bundle(source)
    with pytest.raises(ValueError, match="model_license_accepted"):
        register_local_research_asset(
            tmp_path / "data-root",
            asset_id="finetune-invalid",
            kind="finetune",
            source_path=source,
            asset_type="finetune_bundle",
            metadata={
                key: value
                for key, value in valid_metadata.items()
                if key != "model_license_accepted"
            },
        )
    with pytest.raises(ValueError, match="benchmark_revision"):
        register_local_research_asset(
            tmp_path / "benchmark-metadata-root",
            asset_id="finetune-missing-benchmark-revision",
            kind="finetune",
            source_path=source,
            asset_type="finetune_bundle",
            metadata={
                key: value
                for key, value in valid_metadata.items()
                if key != "benchmark_revision"
            },
        )

    published = register_local_research_asset(
        tmp_path / "data-root",
        asset_id="finetune-valid",
        kind="finetune",
        source_path=source,
        asset_type="finetune_bundle",
        metadata=valid_metadata,
        clock=lambda: datetime(2026, 8, 14, 8, tzinfo=UTC),
    )
    assert load_research_asset_manifest(published.manifest_path)["metadata"] == valid_metadata

    (source / ".llama_factory_info" / "parameters.json").unlink()
    with pytest.raises(ValueError, match="parameters.json"):
        register_local_research_asset(
            tmp_path / "data-root",
            asset_id="finetune-missing-runtime-contract",
            kind="finetune",
            source_path=source,
            asset_type="finetune_bundle",
            metadata=valid_metadata,
            clock=lambda: datetime(2026, 8, 14, 8, tzinfo=UTC),
        )


def test_local_finetune_contract_binds_revisions_terms_and_dataset_info(
    tmp_path: Path,
) -> None:
    source = tmp_path / "finetune-bundle"
    metadata = _write_finetune_bundle(source)

    bad_revision = {**metadata, "model_revision": "f" * 40}
    with pytest.raises(ValueError, match="model revision disagrees"):
        register_local_research_asset(
            tmp_path / "revision-root",
            asset_id="finetune-bad-revision",
            kind="finetune",
            source_path=source,
            asset_type="finetune_bundle",
            metadata=bad_revision,
            clock=lambda: datetime(2026, 8, 14, 8, tzinfo=UTC),
        )

    bad_terms = {**metadata, "dataset_license_terms_sha256": "f" * 64}
    with pytest.raises(ValueError, match="dataset license terms disagree"):
        register_local_research_asset(
            tmp_path / "terms-root",
            asset_id="finetune-bad-terms",
            kind="finetune",
            source_path=source,
            asset_type="finetune_bundle",
            metadata=bad_terms,
            clock=lambda: datetime(2026, 8, 14, 8, tzinfo=UTC),
        )

    (source / "datasets" / "dataset_info.json").write_text(
        '{"other": {"total_samples": 1, "tasks": {}}}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not describe the selected dataset"):
        register_local_research_asset(
            tmp_path / "dataset-info-root",
            asset_id="finetune-bad-dataset-info",
            kind="finetune",
            source_path=source,
            asset_type="finetune_bundle",
            metadata=metadata,
            clock=lambda: datetime(2026, 8, 14, 8, tzinfo=UTC),
        )


def test_local_finetune_rejects_unknown_and_unsealed_financeiq_benchmarks(
    tmp_path: Path,
) -> None:
    unsupported_source = tmp_path / "unsupported"
    unsupported = _write_finetune_bundle(unsupported_source)
    unsupported["benchmark"] = "finance-v1"
    with pytest.raises(ValueError, match="not supported by pinned RD-Agent"):
        register_local_research_asset(
            tmp_path / "unsupported-root",
            asset_id="finetune-unsupported-benchmark",
            kind="finetune",
            source_path=unsupported_source,
            asset_type="finetune_bundle",
            metadata=unsupported,
        )

    financeiq_source = tmp_path / "financeiq"
    financeiq = _write_finetune_bundle(financeiq_source, benchmark="FinanceIQ_gen")
    (
        financeiq_source
        / "benchmarks"
        / "opencompass_data"
        / "data"
        / "FinanceIQ"
        / "test"
        / "questions.csv"
    ).unlink()
    with pytest.raises(ValueError, match="pre-sealed dev and test"):
        register_local_research_asset(
            tmp_path / "financeiq-root",
            asset_id="finetune-unsealed-financeiq",
            kind="finetune",
            source_path=financeiq_source,
            asset_type="finetune_bundle",
            metadata=financeiq,
            clock=lambda: datetime(2026, 8, 14, 8, tzinfo=UTC),
        )


def test_manual_https_acquisition_uses_verified_time(tmp_path: Path) -> None:
    verified_at = datetime(2026, 8, 14, 8, tzinfo=UTC)

    class Downloader:
        def download(self, url: str) -> VerifiedPdf:
            return _verified_pdf(_PDF_BODY, url=url)

    published = acquire_manual_https_pdf(
        tmp_path,
        url="https://example.com/manual.pdf",
        title="Manual paper",
        published_at=datetime(2020, 1, 1, tzinfo=UTC),
        downloader=Downloader(),  # type: ignore[arg-type]
        clock=lambda: verified_at,
    )
    manifest = load_research_asset_manifest(published.manifest_path)
    assert manifest["available_at"] == verified_at.isoformat()
    assert manifest["source"]["kind"] == "manual_https"


def _write_research_snapshot(
    data_root: Path,
    *,
    research_rows: list[dict[str, object]] | None,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    snapshot = data_root / "snapshots" / "snapshot-v1"
    snapshot.mkdir(parents=True)
    datasets: dict[str, dict[str, object]] = {}

    def write_dataset(name: str, rows: list[dict[str, object]]) -> None:
        path = snapshot / "parquet" / name / "data.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
        relative = path.relative_to(snapshot).as_posix()
        datasets[name] = {
            "files": [
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ]
        }

    write_dataset(
        "trade_cal",
        [
            {"cal_date": "20260814", "is_open": 1},
            {"cal_date": "20260817", "is_open": 1},
        ],
    )
    if research_rows is not None:
        write_dataset("research_report", research_rows)
    configuration = {"profile": "research-assets", "start": "2017-01-01"}
    manifest = {
        "name": "snapshot-v1",
        "profile": "research-assets",
        "lineage_id": make_lineage_id("research_asset_source", configuration),
        "lineage_contract": {
            "kind": "research_asset_source",
            "configuration": configuration,
        },
        "lineage_generation": 0,
        "parent_snapshot": None,
        "parent_manifest_sha256": None,
        "start_date": "2017-01-01",
        "end_date": "2026-08-17",
        "quality_gate": {"ok": True},
        "datasets": datasets,
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class _RecordingImporter:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def import_manifest(self, manifest_path: Path) -> str:
        self.paths.append(manifest_path)
        return manifest_path.parent.name


def _verified_pdf(
    body: bytes,
    *,
    url: str = "https://example.com/research.pdf",
) -> VerifiedPdf:
    return VerifiedPdf(
        requested_url=url,
        final_url=url,
        redirect_chain=(),
        media_type="application/pdf",
        body=body,
        sha256=hashlib.sha256(body).hexdigest(),
    )


def _atom_entry(index: int, category: str) -> str:
    return f"""
    <entry>
      <id>https://arxiv.org/abs/2608.{index:05d}v1</id>
      <updated>2026-08-{10 + index:02d}T12:00:00Z</updated>
      <published>2026-08-{10 + index:02d}T12:00:00Z</published>
      <title>Alpha factor portfolio study {index}</title>
      <summary>Machine learning for financial market risk and trading.</summary>
      <author><name>Author {index}</name></author>
      <category term="{category}" />
      <link type="application/pdf" href="https://arxiv.org/pdf/2608.{index:05d}v1" />
    </entry>
    """
