from __future__ import annotations

import hashlib
import os
import random
import re
import time
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

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
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
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
    failed: int
    bytes_written: int
    index_path: Path | None
    log_path: Path | None

    def as_dict(self) -> dict:
        return {
            "status": "succeeded",
            "planned": self.planned,
            "downloaded": self.downloaded,
            "skipped": self.skipped,
            "failed": self.failed,
            "bytes_written": self.bytes_written,
            "index_path": str(self.index_path) if self.index_path else None,
            "log_path": str(self.log_path) if self.log_path else None,
        }


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


def _read_parquet_union(paths: list[str], query: str) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        return connection.execute(query, [paths]).fetchdf()
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
    return [
        AnnouncementRef(
            ts_code=str(row.ts_code),
            ann_date=row.ann_date.date(),
            title="" if pd.isna(row.title) else str(row.title),
            url=resolve_pdf_url(str(row.url), row.ann_date.date()),
        )
        for row in frame.itertuples()
    ]


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


def download_cninfo_announcements(
    data_root: Path,
    *,
    ts_codes: set[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    limit: int | None = None,
    rate_gate: GlobalRateGate | None = None,
    client: CninfoHttpClient | None = None,
    requests_per_minute: float = 30.0,
    timeout_seconds: float = 60.0,
    max_attempts: int = 5,
    cooldown_seconds: float = 180.0,
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
    if limit is not None and limit > 0:
        refs = refs[:limit]

    base = data_root / ANNOUNCEMENTS_DIR
    files_root = base / "files"
    logs_root = base / "logs"
    index_path = base / "index.parquet"
    files_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    if index_path.exists():
        records = _index_records(pd.read_parquet(index_path))
    else:
        records = {}

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

    log_rows: list[dict] = []
    downloaded = skipped = failed = bytes_written = 0
    for ref in refs:
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
            if target.is_file() and target.stat().st_size == existing["bytes"]:
                skipped += 1
                log_row.update(
                    status="skipped",
                    fetched_at=clock(),
                    bytes=existing["bytes"],
                    sha256=existing["sha256"],
                    file_path=existing["file_path"],
                )
                continue

        try:
            body, attempts = client.get(ref.url)
        except CninfoDownloadError as exc:
            failed += 1
            log_row.update(
                status="failed",
                fetched_at=clock(),
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
        downloaded += 1
        log_row.update(
            status="succeeded",
            fetched_at=ingested_at,
            attempts=attempts,
            bytes=len(body),
            sha256=digest,
            file_path=relative,
        )

    index_frame = _index_frame(records)
    temporary_index = index_path.with_suffix(".parquet.tmp")
    index_frame.to_parquet(temporary_index, index=False, compression="zstd", engine="pyarrow")
    os.replace(temporary_index, index_path)

    log_path: Path | None = None
    if log_rows:
        log_path = logs_root / f"download_log_{clock():%Y%m%dT%H%M%SZ}.parquet"
        temporary_log = log_path.with_suffix(".parquet.tmp")
        _log_frame(log_rows).to_parquet(
            temporary_log, index=False, compression="zstd", engine="pyarrow"
        )
        os.replace(temporary_log, log_path)

    return DownloadSummary(
        planned=len(refs),
        downloaded=downloaded,
        skipped=skipped,
        failed=failed,
        bytes_written=bytes_written,
        index_path=index_path,
        log_path=log_path,
    )
