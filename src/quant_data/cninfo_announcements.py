from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

import duckdb
import pandas as pd
import requests

from .rate_limit import GlobalRateGate

# Title keywords that mark a regulatory letter (问询函/关注函/监管函/警示函/纪律处分).
REGULATORY_TITLE_PATTERN = re.compile(r"问询函|关注函|监管函|警示函|纪律处分")
REGULATORY_CATEGORY = "regulatory_letter"
DEFAULT_CATEGORY = "announcement"

ANNOUNCEMENTS_DIR = "announcements"
INDEX_COLUMNS = (
    "ts_code",
    "ann_date",
    "available_at",
    "ingested_at",
    "title",
    "url",
    "sha256",
    "category",
    "file_path",
    "bytes",
)
LOG_COLUMNS = (
    "api_name",
    "url",
    "ts_code",
    "ann_date",
    "title",
    "fetched_at",
    "status",
    "attempts",
    "bytes",
    "sha256",
    "file_path",
    "error",
)
LOG_API_NAME = "cninfo_announcement_pdf"
UNAVAILABLE_COLUMNS = (
    "url",
    "ts_code",
    "ann_date",
    "title",
    "status_code",
    "first_seen_at",
    "last_checked_at",
    "attempts",
    "error",
)
SOURCE_UNAVAILABLE_STATUS_CODES = frozenset({404, 410})
DEFAULT_UNAVAILABLE_RECHECK_DAYS = 30
QUALITY_SCHEMA_VERSION = "cninfo-announcement-quality.v1"
QUALITY_SAMPLE_LIMIT = 20
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
SSE_BULLETIN_QUERY_URL = (
    "https://query.sse.com.cn/security/stock/queryCompanyBulletin.do"
)
SSE_PDF_BASE_URL = "https://big5.sse.com.cn/site/cht/www.sse.com.cn/"
OFFICIAL_PDF_HOSTS = frozenset(
    {
        "big5.sse.com.cn",
        "reportdocs.static.szse.cn",
        "static.sse.com.cn",
        "www.sse.com.cn",
    }
)
OFFICIAL_PDF_URL_PATTERN = re.compile(
    rb"https?://[A-Za-z0-9._~-]+(?:/[A-Za-z0-9._~!$&'()*+,;=:@%/?#-]*)?\.pdf",
    re.IGNORECASE,
)


class CninfoDownloadError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        attempts: int = 1,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.attempts = attempts
        self.status_code = status_code


def categorize_title(title: object) -> str:
    if title is None:
        return DEFAULT_CATEGORY
    return (
        REGULATORY_CATEGORY
        if REGULATORY_TITLE_PATTERN.search(str(title))
        else DEFAULT_CATEGORY
    )


def _validate_body(url: str, body: bytes) -> None:
    if not body:
        raise CninfoDownloadError(f"empty response body: {url}", retryable=False)
    path = url.split("?", 1)[0].lower()
    if path.endswith(".pdf") and not body.startswith(b"%PDF"):
        preview = body[:80].decode("utf-8", errors="replace")
        raise CninfoDownloadError(
            f"response is not a PDF document: {url}: {preview}",
            retryable=False,
        )


def _embedded_official_pdf_url(body: bytes) -> str | None:
    """Return an official exchange PDF URL exposed by a historical error page."""

    for match in OFFICIAL_PDF_URL_PATTERN.finditer(body):
        candidate = match.group(0).decode("ascii", errors="ignore")
        parsed = urlsplit(candidate)
        if parsed.hostname and parsed.hostname.lower() in OFFICIAL_PDF_HOSTS:
            return candidate
    return None


def _is_explicit_html_not_found(body: bytes) -> bool:
    """Recognize cninfo/SZSE's HTTP-200 wrapper for an unambiguous 404."""

    sample = body[:16_384].lower()
    return (
        b"<title>404</title>" in sample
        and b"/maintain/images/404_" in sample
    )


def _sse_bulletin_query_url(ref: AnnouncementRef) -> str:
    params = {
        "isPagination": "false",
        "productId": ref.ts_code.split(".", 1)[0],
        # ``0101`` is the main-board stock type. Without it, the SSE endpoint
        # silently returns no rows for ordinary SH listings such as 600654
        # and 603778 even when the disclosure exists on the requested date.
        "securityType": "0101,120100,020100,020200,120200",
        "keyWord": "",
        "reportType2": "",
        "reportType": "ALL",
        "beginDate": ref.ann_date.isoformat(),
        "endDate": ref.ann_date.isoformat(),
        "pageHelp.pageSize": "100",
        "pageHelp.beginPage": "1",
        "pageHelp.pageCount": "50",
        "pageHelp.pageNo": "1",
        "pageHelp.cacheSize": "1",
        "pageHelp.endPage": "5",
        "jsonCallBack": "quantlabCallback",
    }
    return f"{SSE_BULLETIN_QUERY_URL}?{urlencode(params)}"


def _select_sse_fallback_url(
    ref: AnnouncementRef, payload: dict[str, object]
) -> str | None:
    rows = payload.get("result")
    if not isinstance(rows, list):
        return None
    wanted_title = "".join(ref.title.split())
    wanted_code = ref.ts_code.split(".", 1)[0]
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = "".join(str(row.get("TITLE") or "").split())
        if title != wanted_title:
            continue
        if str(row.get("SECURITY_CODE") or "") != wanted_code:
            continue
        if str(row.get("SSEDATE") or "") != ref.ann_date.isoformat():
            continue
        # SSE's normal listedinfo PDF host can return its JavaScript challenge
        # page to non-browser clients. The exchange's own Big5 mirror serves
        # the same canonical path as the original PDF without that challenge.
        candidate = urljoin(SSE_PDF_BASE_URL, str(row.get("URL") or "").lstrip("/"))
        parsed = urlsplit(candidate)
        if (
            parsed.hostname
            and parsed.hostname.lower() in OFFICIAL_PDF_HOSTS
            and parsed.path.lower().endswith(".pdf")
        ):
            return candidate
    return None


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_body(url: str, body: bytes) -> bytes:
    """Extract a PDF embedded in cninfo's occasional multipart response.

    Some historical cninfo/static and datacloud URLs return HTTP 200 with a
    single ``form-data`` part instead of a bare PDF. The boundary is not always
    standards-compliant, so parsing by Content-Type is insufficient. Preserve
    normal responses byte-for-byte; for a PDF URL with leading wrapper bytes,
    accept only an actual ``%PDF-`` payload terminated by ``%%EOF`` and strip
    the transport wrapper. A response with no complete PDF still fails closed
    in ``_validate_body``.
    """

    path = url.split("?", 1)[0].lower()
    if not path.endswith(".pdf") or body.startswith(b"%PDF"):
        return body
    pdf_start = body.find(b"%PDF-")
    pdf_end = body.rfind(b"%%EOF")
    if pdf_start < 0 or pdf_end < pdf_start:
        return body
    return body[pdf_start : pdf_end + len(b"%%EOF")]


class CninfoHttpClient:
    """Bounded-retry HTTP GET client for static.cninfo.com.cn bodies."""

    def __init__(
        self,
        *,
        rate_gate: GlobalRateGate,
        timeout_seconds: float = 60.0,
        max_attempts: int = 5,
        cooldown_seconds: float = 180.0,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rate_gate = rate_gate
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self.cooldown_seconds = cooldown_seconds
        self.session = session or requests.Session()
        self.sleeper = sleeper

    def get(self, url: str) -> tuple[bytes, int]:
        """Return (body, attempts); raise CninfoDownloadError after bounded retries."""

        last_error: CninfoDownloadError | None = None
        for attempt in range(1, self.max_attempts + 1):
            self.rate_gate.wait()
            try:
                response = self.session.get(
                    url,
                    timeout=(10.0, self.timeout_seconds),
                    headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
                )
            except requests.RequestException as exc:
                last_error = CninfoDownloadError(str(exc), retryable=True, attempts=attempt)
            else:
                status = int(response.status_code)
                body = bytes(response.content)
                if 200 <= status < 300:
                    body = _normalize_body(url, body)
                    path = url.split("?", 1)[0].lower()
                    if path.endswith(".pdf") and not body.startswith(b"%PDF"):
                        upstream_url = _embedded_official_pdf_url(body)
                        if upstream_url and upstream_url != url:
                            upstream_body, upstream_attempts = self.get(upstream_url)
                            return upstream_body, attempt + upstream_attempts
                        if _is_explicit_html_not_found(body):
                            raise CninfoDownloadError(
                                f"official source returned an HTML 404 page: {url}",
                                retryable=False,
                                attempts=attempt,
                                status_code=404,
                            )
                    try:
                        _validate_body(url, body)
                    except CninfoDownloadError as exc:
                        exc.attempts = attempt
                        raise
                    return body, attempt
                retryable = status == 429 or status >= 500
                last_error = CninfoDownloadError(
                    f"HTTP {status} for {url}",
                    retryable=retryable,
                    attempts=attempt,
                    status_code=status,
                )
            if not last_error.retryable or attempt >= self.max_attempts:
                break
            if last_error.status_code == 429:
                # Shared cooldown, same discipline as the Tushare provider: stop
                # this run quickly instead of hammering a rate-limited host.
                self.rate_gate.cooldown(self.cooldown_seconds)
                break
            delay = min(30.0, (2 ** (attempt - 1)) + random.uniform(0.0, 1.0))
            self.sleeper(delay)
        assert last_error is not None
        raise last_error

    def resolve_sse_pdf_url(self, ref: AnnouncementRef) -> str | None:
        """Resolve an exact SH disclosure to SSE's canonical PDF endpoint."""

        if not ref.ts_code.upper().endswith(".SH"):
            return None
        query_url = _sse_bulletin_query_url(ref)
        self.rate_gate.wait()
        try:
            response = self.session.get(
                query_url,
                timeout=(10.0, self.timeout_seconds),
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "*/*",
                    "Referer": "https://star.sse.com.cn/disclosure/listannouncement/",
                },
            )
        except requests.RequestException:
            return None
        if int(response.status_code) != 200:
            return None
        text = bytes(response.content).decode("utf-8", errors="replace").strip()
        if text.startswith("quantlabCallback(") and text.endswith(")"):
            text = text[len("quantlabCallback(") : -1]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return _select_sse_fallback_url(ref, payload)

    def get_announcement(self, ref: AnnouncementRef) -> tuple[bytes, int, str | None]:
        """Fetch a cninfo body, falling back only to an exact official SSE match."""

        try:
            body, attempts = self.get(ref.url)
            return body, attempts, None
        except CninfoDownloadError as original:
            if original.status_code in SOURCE_UNAVAILABLE_STATUS_CODES:
                raise
            fallback_url = self.resolve_sse_pdf_url(ref)
            if fallback_url is None:
                raise
            try:
                body, attempts = self.get(fallback_url)
            except CninfoDownloadError as fallback_error:
                raise CninfoDownloadError(
                    f"{original}; official SSE fallback {fallback_url} failed: "
                    f"{fallback_error}",
                    retryable=fallback_error.retryable,
                    attempts=original.attempts + fallback_error.attempts,
                    status_code=fallback_error.status_code,
                ) from fallback_error
            return body, original.attempts + attempts, fallback_url


@dataclass(frozen=True, slots=True)
class AnnouncementRef:
    ts_code: str
    ann_date: date
    title: str
    url: str


@dataclass(slots=True)
class DownloadSummary:
    planned: int
    downloaded: int
    skipped: int
    unavailable: int
    failed: int
    bytes_written: int
    index_path: Path | None
    log_path: Path | None
    unavailable_path: Path | None

    def as_dict(self) -> dict:
        return {
            "status": (
                "failed"
                if self.failed
                else "succeeded_with_source_gaps"
                if self.unavailable
                else "succeeded"
            ),
            "planned": self.planned,
            "downloaded": self.downloaded,
            "skipped": self.skipped,
            "unavailable": self.unavailable,
            "failed": self.failed,
            "bytes_written": self.bytes_written,
            "index_path": str(self.index_path) if self.index_path else None,
            "log_path": str(self.log_path) if self.log_path else None,
            "unavailable_path": (
                str(self.unavailable_path) if self.unavailable_path else None
            ),
        }


def _sample(values: Sequence[object]) -> list[str]:
    return [str(value) for value in list(values)[:QUALITY_SAMPLE_LIMIT]]


def _artifact_sha256(path: Path | None) -> str | None:
    return _sha256_path(path) if path is not None and path.is_file() else None


def _write_quality_report(data_root: Path, report: dict, generated_at: datetime) -> Path:
    quality_root = data_root / ANNOUNCEMENTS_DIR / "quality"
    quality_root.mkdir(parents=True, exist_ok=True)
    report_path = quality_root / f"quality_{generated_at:%Y%m%dT%H%M%S%fZ}.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, report_path)
    latest = quality_root / "latest.json"
    temporary_latest = latest.with_suffix(".json.tmp")
    temporary_latest.write_text(payload, encoding="utf-8")
    os.replace(temporary_latest, latest)
    return report_path


def audit_cninfo_announcements(
    data_root: Path,
    *,
    ts_codes: set[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    limit: int | None = None,
    regulatory_only: bool = False,
    verify_hashes: bool = True,
    now: Callable[[], datetime] | None = None,
) -> dict:
    """Audit announcement coverage, PIT metadata, tombstones, and file integrity.

    The audit reuses the exact governed discovery scope used by the downloader.
    A source 404/410 is an explicit, non-fabricated gap and therefore a warning;
    a URL absent from both the immutable index and the tombstone ledger is a
    blocking error.  Full checksum verification is the production default so a
    successful durable download cannot be certified by metadata alone.
    """

    clock = now or (lambda: datetime.now(UTC))
    generated_at = clock()
    refs = load_announcement_manifest(data_root, ts_codes=ts_codes, start=start, end=end)
    if regulatory_only:
        refs = [ref for ref in refs if categorize_title(ref.title) == REGULATORY_CATEGORY]
    if limit is not None and limit > 0:
        refs = refs[:limit]
    refs_by_url = {ref.url: ref for ref in refs}
    planned_urls = set(refs_by_url)

    base = data_root / ANNOUNCEMENTS_DIR
    index_path = base / "index.parquet"
    unavailable_path = base / "source_unavailable.parquet"
    errors: list[str] = []
    warnings: list[str] = []

    if index_path.is_file():
        index = pd.read_parquet(index_path)
    else:
        index = _empty_index_frame()
        if planned_urls:
            errors.append(f"announcement index is missing: {index_path}")
    missing_index_columns = sorted(set(INDEX_COLUMNS) - set(index.columns))
    if missing_index_columns:
        errors.append(f"announcement index is missing columns: {missing_index_columns}")
        index = _empty_index_frame()

    if unavailable_path.is_file():
        unavailable_frame = pd.read_parquet(unavailable_path)
    else:
        unavailable_frame = _unavailable_frame({})
    missing_unavailable_columns = sorted(
        set(UNAVAILABLE_COLUMNS) - set(unavailable_frame.columns)
    )
    if missing_unavailable_columns:
        errors.append(
            "source-unavailable ledger is missing columns: "
            f"{missing_unavailable_columns}"
        )
        unavailable_frame = _unavailable_frame({})

    duplicate_index_urls = int(index["url"].duplicated(keep=False).sum())
    duplicate_unavailable_urls = int(
        unavailable_frame["url"].duplicated(keep=False).sum()
    )
    if duplicate_index_urls:
        errors.append(f"announcement index has {duplicate_index_urls} duplicate URL rows")
    if duplicate_unavailable_urls:
        errors.append(
            "source-unavailable ledger has "
            f"{duplicate_unavailable_urls} duplicate URL rows"
        )

    index_scope = index[index["url"].astype(str).isin(planned_urls)].copy()
    unavailable_scope = unavailable_frame[
        unavailable_frame["url"].astype(str).isin(planned_urls)
    ].copy()
    indexed_urls = set(index_scope["url"].astype(str))
    unavailable_urls = set(unavailable_scope["url"].astype(str))
    overlap_urls = indexed_urls & unavailable_urls
    missing_urls = planned_urls - indexed_urls - unavailable_urls
    if overlap_urls:
        errors.append(
            f"{len(overlap_urls)} URLs exist in both index and source-unavailable ledger"
        )
    if missing_urls:
        errors.append(
            f"{len(missing_urls)} discovered URLs have neither an indexed file nor a "
            "source-unavailable tombstone"
        )
    if unavailable_urls:
        warnings.append(
            f"{len(unavailable_urls)} discovered URLs are unavailable at source (HTTP 404/410)"
        )

    invalid_tombstone_status = unavailable_scope[
        ~pd.to_numeric(unavailable_scope["status_code"], errors="coerce").isin(
            SOURCE_UNAVAILABLE_STATUS_CODES
        )
    ]
    if not invalid_tombstone_status.empty:
        errors.append(
            f"{len(invalid_tombstone_status)} source tombstones are not HTTP 404/410"
        )

    open_days = load_trade_calendar_open_days(data_root) if planned_urls else []
    pit_mismatches: list[str] = []
    metadata_mismatches: list[str] = []
    invalid_paths: list[str] = []
    missing_files: list[str] = []
    size_mismatches: list[str] = []
    sha_mismatches: list[str] = []
    pdf_magic_mismatches: list[str] = []
    verified_files = 0
    root_resolved = data_root.resolve()

    for row in index_scope.itertuples(index=False):
        url = str(row.url)
        ref = refs_by_url[url]
        try:
            expected_available_at = next_trading_day(ref.ann_date, open_days)
        except LookupError as exc:
            pit_mismatches.append(f"{url}: {exc}")
            expected_available_at = None
        actual_ann_date = pd.Timestamp(row.ann_date).date()
        actual_available_at = pd.Timestamp(row.available_at).date()
        if actual_ann_date != ref.ann_date or (
            expected_available_at is not None
            and actual_available_at != expected_available_at
        ):
            pit_mismatches.append(
                f"{url}: ann_date={actual_ann_date}, available_at={actual_available_at}, "
                f"expected={ref.ann_date}/{expected_available_at}"
            )
        expected_category = categorize_title(ref.title)
        if (
            str(row.ts_code) != ref.ts_code
            or str(row.category) != expected_category
            or str(row.title) != ref.title
        ):
            metadata_mismatches.append(url)

        digest = str(row.sha256)
        expected_relative = (
            f"{ANNOUNCEMENTS_DIR}/files/{digest[:2]}/{digest}.pdf"
            if re.fullmatch(r"[0-9a-f]{64}", digest)
            else ""
        )
        relative = str(row.file_path)
        target = data_root / relative
        try:
            safe_path = target.resolve().is_relative_to(root_resolved)
        except (OSError, ValueError):
            safe_path = False
        if not safe_path or not expected_relative or relative != expected_relative:
            invalid_paths.append(url)
            continue
        if not target.is_file():
            missing_files.append(url)
            continue
        expected_bytes = int(row.bytes)
        if expected_bytes <= 0 or target.stat().st_size != expected_bytes:
            size_mismatches.append(url)
            continue
        with target.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                pdf_magic_mismatches.append(url)
                continue
        if verify_hashes and _sha256_path(target) != digest:
            sha_mismatches.append(url)
            continue
        verified_files += 1

    for label, values in (
        ("PIT metadata mismatches", pit_mismatches),
        ("source metadata mismatches", metadata_mismatches),
        ("invalid content-addressed paths", invalid_paths),
        ("missing files", missing_files),
        ("file-size mismatches", size_mismatches),
        ("file checksum mismatches", sha_mismatches),
        ("non-PDF file bodies", pdf_magic_mismatches),
    ):
        if values:
            errors.append(f"{len(values)} {label}")

    planned = len(planned_urls)
    covered = len(indexed_urls | unavailable_urls)
    discovered_dates = sorted(ref.ann_date for ref in refs)
    report = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "generated_at": generated_at.astimezone(UTC).isoformat(),
        "scope": {
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None,
            "ts_codes": sorted(ts_codes or set()),
            "regulatory_only": bool(regulatory_only),
            "limit": int(limit or 0),
            "verify_hashes": bool(verify_hashes),
        },
        "source_boundary": {
            "discovered_min": discovered_dates[0].isoformat() if discovered_dates else None,
            "discovered_max": discovered_dates[-1].isoformat() if discovered_dates else None,
            "note": (
                "cninfo discovery is limited to the earliest trustworthy rows persisted "
                "by anns_d; absent older rows are not fabricated"
            ),
        },
        "coverage": {
            "planned": planned,
            "indexed": len(indexed_urls),
            "source_unavailable": len(unavailable_urls),
            "missing": len(missing_urls),
            "ratio": (covered / planned) if planned else 1.0,
            "missing_samples": _sample(sorted(missing_urls)),
            "source_unavailable_samples": _sample(sorted(unavailable_urls)),
        },
        "governance": {
            "duplicate_index_url_rows": duplicate_index_urls,
            "duplicate_unavailable_url_rows": duplicate_unavailable_urls,
            "index_tombstone_overlap": len(overlap_urls),
            "pit_mismatches": len(pit_mismatches),
            "metadata_mismatches": len(metadata_mismatches),
            "pit_mismatch_samples": _sample(pit_mismatches),
            "metadata_mismatch_samples": _sample(metadata_mismatches),
        },
        "integrity": {
            "indexed_files_in_scope": len(index_scope),
            "verified_files": verified_files,
            "invalid_paths": len(invalid_paths),
            "missing_files": len(missing_files),
            "size_mismatches": len(size_mismatches),
            "sha256_mismatches": len(sha_mismatches),
            "pdf_magic_mismatches": len(pdf_magic_mismatches),
            "invalid_path_samples": _sample(invalid_paths),
            "missing_file_samples": _sample(missing_files),
            "size_mismatch_samples": _sample(size_mismatches),
            "sha256_mismatch_samples": _sample(sha_mismatches),
            "pdf_magic_mismatch_samples": _sample(pdf_magic_mismatches),
        },
        "artifacts": {
            "index_path": str(index_path),
            "index_sha256": _artifact_sha256(index_path),
            "source_unavailable_path": str(unavailable_path),
            "source_unavailable_sha256": _artifact_sha256(unavailable_path),
        },
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }
    report_path = _write_quality_report(data_root, report, generated_at)
    report["report_path"] = str(report_path)
    report["report_sha256"] = _sha256_path(report_path)
    return report


def _parquet_files(data_root: Path, dataset: str) -> list[str]:
    candidates: list[Path] = []
    units_dir = data_root / "units" / dataset
    if units_dir.is_dir():
        candidates.extend(sorted(units_dir.glob("*.parquet")))
    snapshots_dir = data_root / "snapshots"
    if snapshots_dir.is_dir():
        for snapshot in sorted(snapshots_dir.iterdir()):
            dataset_dir = snapshot / "parquet" / dataset
            if dataset_dir.is_dir():
                candidates.extend(sorted(dataset_dir.rglob("*.parquet")))
    return [path.as_posix() for path in candidates]


def _read_parquet_union(
    paths: list[str], query: str, parameters: Sequence[object] = ()
) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        return connection.execute(query, [paths, *parameters]).fetchdf()
    finally:
        connection.close()


def resolve_pdf_url(url: str, ann_date: date) -> str:
    """Resolve a cninfo disclosure detail-page url to its static PDF url.

    Tushare ``anns_d`` ships two url shapes: direct ``static.cninfo.com.cn``
    PDF links, and ``www.cninfo.com.cn/new/disclosure/detail?...`` viewer
    pages.  The viewer page is a JavaScript shell, so the PDF location is
    derived deterministically from its query parameters instead of being
    scraped: ``static.cninfo.com.cn/finalpage/<date>/<announcementId>.PDF``.
    Anything that does not match the detail-page shape is returned unchanged
    (the %PDF magic check in ``_validate_body`` stays the fail-closed guard).
    """

    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.netloc.endswith("cninfo.com.cn"):
        return url
    if not parts.path.startswith("/new/disclosure/detail"):
        return url
    params = parse_qs(parts.query)
    announcement_id = (params.get("announcementId") or [""])[0].strip()
    if not announcement_id:
        return url
    when = (params.get("announcementTime") or [""])[0].strip() or ann_date.isoformat()
    return f"http://static.cninfo.com.cn/finalpage/{when}/{announcement_id}.PDF"


def load_announcement_manifest(
    data_root: Path,
    *,
    ts_codes: set[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> list[AnnouncementRef]:
    """Load the on-disk anns_d index as the download discovery source.

    Reads every anns_d parquet unit and snapshot partition under data_root and
    keeps only rows carrying a usable URL. Raises fail-closed when no anns_d
    parquet has been persisted yet.
    """

    paths = _parquet_files(data_root, "anns_d")
    if not paths:
        raise RuntimeError(
            f"anns_d announcement index is unavailable under {data_root}; "
            "run the Tushare bootstrap (full profile) first"
        )
    frame = _read_parquet_union(
        paths,
        """
        WITH raw AS (
            SELECT
                CAST(ts_code AS VARCHAR) AS ts_code,
                coalesce(
                    try_cast(ann_date AS DATE),
                    try_strptime(CAST(ann_date AS VARCHAR), '%Y%m%d')::DATE
                ) AS ann_date,
                CAST(title AS VARCHAR) AS title,
                trim(CAST(url AS VARCHAR)) AS url
            FROM read_parquet(?, union_by_name=true)
        )
        SELECT DISTINCT ts_code, ann_date, title, url
        FROM raw
        WHERE url <> '' AND ts_code IS NOT NULL AND ann_date IS NOT NULL
        """,
    )
    if frame.empty:
        return []
    frame["ann_date"] = pd.to_datetime(frame["ann_date"])
    if ts_codes:
        wanted = {code.strip().upper() for code in ts_codes if code.strip()}
        frame = frame[frame["ts_code"].str.upper().isin(sorted(wanted))]
    if start is not None:
        frame = frame[frame["ann_date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["ann_date"] <= pd.Timestamp(end)]
    frame = frame.sort_values(["url", "ann_date", "ts_code"], kind="stable")
    frame = frame.drop_duplicates("url", keep="first")
    frame = frame.sort_values(["ann_date", "ts_code", "url"], kind="stable")
    resolved = [
        AnnouncementRef(
            ts_code=str(row.ts_code),
            ann_date=row.ann_date.date(),
            title="" if pd.isna(row.title) else str(row.title),
            url=resolve_pdf_url(str(row.url), row.ann_date.date()),
        )
        for row in frame.itertuples()
    ]
    # Different viewer/detail URLs can resolve to the same static PDF. De-dupe
    # after resolution as well, otherwise a single document is requested and
    # logged multiple times in the same run.
    unique: dict[str, AnnouncementRef] = {}
    for ref in resolved:
        unique.setdefault(ref.url, ref)
    return sorted(unique.values(), key=lambda ref: (ref.ann_date, ref.ts_code, ref.url))


def load_trade_calendar_open_days(data_root: Path) -> list[date]:
    """Return sorted SSE open days from the persisted trade_cal dataset.

    Fail-closed: raises when no trade_cal parquet exists or it contains no
    open day; available_at must never be guessed without the real calendar.
    """

    paths = _parquet_files(data_root, "trade_cal")
    if not paths:
        raise RuntimeError(
            f"trade_cal trading calendar is unavailable under {data_root}; "
            "available_at cannot be derived without it"
        )
    frame = _read_parquet_union(
        paths,
        """
        SELECT DISTINCT
            coalesce(
                try_cast(cal_date AS DATE),
                try_strptime(CAST(cal_date AS VARCHAR), '%Y%m%d')::DATE
            ) AS cal_date,
            CAST(is_open AS VARCHAR) AS is_open
        FROM read_parquet(?, union_by_name=true)
        """,
    )
    if frame.empty:
        raise RuntimeError("trade_cal trading calendar is empty; refusing to guess available_at")
    frame["cal_date"] = pd.to_datetime(frame["cal_date"], errors="coerce")
    is_open = frame["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
    days = sorted({value.date() for value in frame.loc[is_open, "cal_date"].dropna()})
    if not days:
        raise RuntimeError(
            "trade_cal trading calendar has no open day; refusing to guess available_at"
        )
    return days


def next_trading_day(ann_date: date, open_days: Sequence[date]) -> date:
    """First open trading day strictly after ann_date (design draft 3.3)."""

    index = bisect_right(open_days, ann_date)
    if index >= len(open_days):
        raise LookupError(
            f"no trading day after {ann_date} in the persisted trade_cal; "
            "extend the calendar before deriving available_at"
        )
    return open_days[index]


def _empty_index_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": pd.Series(dtype="string"),
            "ann_date": pd.Series(dtype="datetime64[ns]"),
            "available_at": pd.Series(dtype="datetime64[ns]"),
            "ingested_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "title": pd.Series(dtype="string"),
            "url": pd.Series(dtype="string"),
            "sha256": pd.Series(dtype="string"),
            "category": pd.Series(dtype="string"),
            "file_path": pd.Series(dtype="string"),
            "bytes": pd.Series(dtype="int64"),
        }
    )


def _index_records(frame: pd.DataFrame) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for row in frame.itertuples():
        records[str(row.url)] = {
            "ts_code": str(row.ts_code),
            "ann_date": pd.Timestamp(row.ann_date).date(),
            "available_at": pd.Timestamp(row.available_at).date(),
            "ingested_at": pd.Timestamp(row.ingested_at).to_pydatetime(),
            "title": str(row.title),
            "url": str(row.url),
            "sha256": str(row.sha256),
            "category": str(row.category),
            "file_path": str(row.file_path),
            "bytes": int(row.bytes),
        }
    return records


def _index_frame(records: dict[str, dict]) -> pd.DataFrame:
    if not records:
        return _empty_index_frame()
    frame = pd.DataFrame(list(records.values()), columns=list(INDEX_COLUMNS))
    frame["ann_date"] = pd.to_datetime(frame["ann_date"])
    frame["available_at"] = pd.to_datetime(frame["available_at"])
    frame["ingested_at"] = pd.to_datetime(frame["ingested_at"], utc=True)
    frame["bytes"] = frame["bytes"].astype("int64")
    return frame.sort_values(["ann_date", "ts_code", "url"], kind="stable").reset_index(
        drop=True
    )


def _log_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(LOG_COLUMNS))
    frame["ann_date"] = pd.to_datetime(frame["ann_date"])
    frame["fetched_at"] = pd.to_datetime(frame["fetched_at"], utc=True)
    frame["attempts"] = frame["attempts"].astype("int64")
    frame["bytes"] = frame["bytes"].astype("int64")
    return frame


def _unavailable_records(frame: pd.DataFrame) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for row in frame.itertuples():
        records[str(row.url)] = {
            "url": str(row.url),
            "ts_code": str(row.ts_code),
            "ann_date": pd.Timestamp(row.ann_date).date(),
            "title": str(row.title),
            "status_code": int(row.status_code),
            "first_seen_at": pd.Timestamp(row.first_seen_at).to_pydatetime(),
            "last_checked_at": pd.Timestamp(row.last_checked_at).to_pydatetime(),
            "attempts": int(row.attempts),
            "error": str(row.error),
        }
    return records


def _unavailable_frame(records: dict[str, dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            {
                "url": pd.Series(dtype="string"),
                "ts_code": pd.Series(dtype="string"),
                "ann_date": pd.Series(dtype="datetime64[ns]"),
                "title": pd.Series(dtype="string"),
                "status_code": pd.Series(dtype="int64"),
                "first_seen_at": pd.Series(dtype="datetime64[ns, UTC]"),
                "last_checked_at": pd.Series(dtype="datetime64[ns, UTC]"),
                "attempts": pd.Series(dtype="int64"),
                "error": pd.Series(dtype="string"),
            }
        )
    frame = pd.DataFrame(list(records.values()), columns=list(UNAVAILABLE_COLUMNS))
    frame["ann_date"] = pd.to_datetime(frame["ann_date"])
    frame["first_seen_at"] = pd.to_datetime(frame["first_seen_at"], utc=True)
    frame["last_checked_at"] = pd.to_datetime(frame["last_checked_at"], utc=True)
    frame["status_code"] = frame["status_code"].astype("int64")
    frame["attempts"] = frame["attempts"].astype("int64")
    return frame.sort_values(["ann_date", "ts_code", "url"], kind="stable").reset_index(
        drop=True
    )


def _persist_download_state(
    records: dict[str, dict],
    unavailable_records: dict[str, dict],
    log_rows: list[dict],
    *,
    index_path: Path,
    unavailable_path: Path,
    log_path: Path | None,
) -> None:
    """Atomically persist resumable metadata during a long body download."""

    index_frame = _index_frame(records)
    temporary_index = index_path.with_suffix(".parquet.tmp")
    index_frame.to_parquet(
        temporary_index, index=False, compression="zstd", engine="pyarrow"
    )
    os.replace(temporary_index, index_path)
    unavailable_frame = _unavailable_frame(unavailable_records)
    temporary_unavailable = unavailable_path.with_suffix(".parquet.tmp")
    unavailable_frame.to_parquet(
        temporary_unavailable, index=False, compression="zstd", engine="pyarrow"
    )
    os.replace(temporary_unavailable, unavailable_path)
    if log_path is not None and log_rows:
        temporary_log = log_path.with_suffix(".parquet.tmp")
        _log_frame(log_rows).to_parquet(
            temporary_log, index=False, compression="zstd", engine="pyarrow"
        )
        os.replace(temporary_log, log_path)


def download_cninfo_announcements(
    data_root: Path,
    *,
    ts_codes: set[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    limit: int | None = None,
    regulatory_only: bool = False,
    rate_gate: GlobalRateGate | None = None,
    client: CninfoHttpClient | None = None,
    requests_per_minute: float = 30.0,
    timeout_seconds: float = 60.0,
    max_attempts: int = 5,
    cooldown_seconds: float = 180.0,
    checkpoint_every: int = 100,
    unavailable_recheck_days: int = DEFAULT_UNAVAILABLE_RECHECK_DAYS,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    now: Callable[[], datetime] | None = None,
) -> DownloadSummary:
    """Download announcement bodies discovered through the persisted anns_d index.

    Files are content-addressed (sha256) and immutable: an existing file whose
    checksum matches the metadata index is skipped, and a changed body is
    stored as a new file instead of overwriting. Every attempt is appended to
    a per-run download-log parquet next to the metadata index.
    """

    clock = now or (lambda: datetime.now(UTC))
    refs = load_announcement_manifest(data_root, ts_codes=ts_codes, start=start, end=end)
    if regulatory_only:
        refs = [ref for ref in refs if categorize_title(ref.title) == REGULATORY_CATEGORY]
    if limit is not None and limit > 0:
        refs = refs[:limit]

    base = data_root / ANNOUNCEMENTS_DIR
    files_root = base / "files"
    logs_root = base / "logs"
    index_path = base / "index.parquet"
    unavailable_path = base / "source_unavailable.parquet"
    files_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    if index_path.exists():
        records = _index_records(pd.read_parquet(index_path))
    else:
        records = {}
    if unavailable_path.exists():
        unavailable_records = _unavailable_records(pd.read_parquet(unavailable_path))
    else:
        unavailable_records = {}

    if refs:
        open_days = load_trade_calendar_open_days(data_root)
    else:
        open_days = []
    if client is None:
        gate = rate_gate or GlobalRateGate(requests_per_minute)
        client = CninfoHttpClient(
            rate_gate=gate,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            cooldown_seconds=cooldown_seconds,
        )

    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if unavailable_recheck_days < 1:
        raise ValueError("unavailable_recheck_days must be positive")
    log_rows: list[dict] = []
    log_path = (
        logs_root / f"download_log_{clock():%Y%m%dT%H%M%SZ}.parquet" if refs else None
    )
    downloaded = skipped = unavailable = failed = bytes_written = 0
    if progress_callback is not None:
        progress_callback(
            {
                "planned": len(refs),
                "completed": 0,
                "downloaded": 0,
                "skipped": 0,
                "unavailable": 0,
                "failed": 0,
                "bytes_written": 0,
            }
        )
    for position, ref in enumerate(refs):
        if position and position % checkpoint_every == 0:
            _persist_download_state(
                records,
                unavailable_records,
                log_rows,
                index_path=index_path,
                unavailable_path=unavailable_path,
                log_path=log_path,
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "planned": len(refs),
                        "completed": downloaded + skipped + unavailable + failed,
                        "downloaded": downloaded,
                        "skipped": skipped,
                        "unavailable": unavailable,
                        "failed": failed,
                        "bytes_written": bytes_written,
                    }
                )
        log_row: dict = {
            "api_name": LOG_API_NAME,
            "url": ref.url,
            "ts_code": ref.ts_code,
            "ann_date": ref.ann_date,
            "title": ref.title,
            "fetched_at": None,
            "status": "",
            "attempts": 0,
            "bytes": 0,
            "sha256": None,
            "file_path": None,
            "error": None,
        }
        log_rows.append(log_row)
        try:
            available_at = next_trading_day(ref.ann_date, open_days)
        except LookupError as exc:
            failed += 1
            log_row.update(status="failed", fetched_at=clock(), error=str(exc))
            continue

        existing = records.get(ref.url)
        if existing is not None:
            target = data_root / existing["file_path"]
            if (
                target.is_file()
                and target.stat().st_size == existing["bytes"]
                and _sha256_path(target) == existing["sha256"]
            ):
                unavailable_records.pop(ref.url, None)
                skipped += 1
                log_row.update(
                    status="skipped",
                    fetched_at=clock(),
                    bytes=existing["bytes"],
                    sha256=existing["sha256"],
                    file_path=existing["file_path"],
                )
                continue

        unavailable_record = unavailable_records.get(ref.url)
        if unavailable_record is not None:
            last_checked_at = pd.Timestamp(
                unavailable_record["last_checked_at"]
            ).to_pydatetime()
            if clock() - last_checked_at < timedelta(days=unavailable_recheck_days):
                unavailable += 1
                log_row.update(
                    status="source_unavailable",
                    fetched_at=clock(),
                    attempts=0,
                    error=(
                        f"cached HTTP {unavailable_record['status_code']}; "
                        f"next recheck after {unavailable_recheck_days} days"
                    ),
                )
                continue

        fallback_url: str | None = None
        try:
            body, attempts, fallback_url = client.get_announcement(ref)
        except CninfoDownloadError as exc:
            checked_at = clock()
            if exc.status_code in SOURCE_UNAVAILABLE_STATUS_CODES:
                unavailable += 1
                previous = unavailable_records.get(ref.url)
                unavailable_records[ref.url] = {
                    "url": ref.url,
                    "ts_code": ref.ts_code,
                    "ann_date": ref.ann_date,
                    "title": ref.title,
                    "status_code": int(exc.status_code),
                    "first_seen_at": (
                        previous["first_seen_at"] if previous else checked_at
                    ),
                    "last_checked_at": checked_at,
                    "attempts": int(exc.attempts),
                    "error": str(exc),
                }
                log_row.update(
                    status="source_unavailable",
                    fetched_at=checked_at,
                    attempts=exc.attempts,
                    error=str(exc),
                )
            else:
                failed += 1
                log_row.update(
                    status="failed",
                    fetched_at=checked_at,
                    attempts=exc.attempts,
                    error=str(exc),
                )
            continue

        digest = hashlib.sha256(body).hexdigest()
        relative = f"{ANNOUNCEMENTS_DIR}/files/{digest[:2]}/{digest}.pdf"
        target = data_root / relative
        if target.exists():
            if target.stat().st_size != len(body):
                failed += 1
                log_row.update(
                    status="failed",
                    fetched_at=clock(),
                    attempts=attempts,
                    sha256=digest,
                    error=f"content-address collision at {relative}; refusing to overwrite",
                )
                continue
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"{target.name}.tmp")
            temporary.write_bytes(body)
            os.replace(temporary, target)
            bytes_written += len(body)

        ingested_at = clock()
        records[ref.url] = {
            "ts_code": ref.ts_code,
            "ann_date": ref.ann_date,
            "available_at": available_at,
            "ingested_at": ingested_at,
            "title": ref.title,
            "url": ref.url,
            "sha256": digest,
            "category": categorize_title(ref.title),
            "file_path": relative,
            "bytes": len(body),
        }
        unavailable_records.pop(ref.url, None)
        downloaded += 1
        log_row.update(
            status="succeeded",
            fetched_at=ingested_at,
            attempts=attempts,
            bytes=len(body),
            sha256=digest,
            file_path=relative,
            error=(f"official fallback: {fallback_url}" if fallback_url else None),
        )

    _persist_download_state(
        records,
        unavailable_records,
        log_rows,
        index_path=index_path,
        unavailable_path=unavailable_path,
        log_path=log_path,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "planned": len(refs),
                "completed": downloaded + skipped + unavailable + failed,
                "downloaded": downloaded,
                "skipped": skipped,
                "unavailable": unavailable,
                "failed": failed,
                "bytes_written": bytes_written,
            }
        )

    return DownloadSummary(
        planned=len(refs),
        downloaded=downloaded,
        skipped=skipped,
        unavailable=unavailable,
        failed=failed,
        bytes_written=bytes_written,
        index_path=index_path,
        log_path=log_path,
        unavailable_path=unavailable_path,
    )
