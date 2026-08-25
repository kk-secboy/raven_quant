from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import mimetypes
import os
import re
import shutil
import socket
import stat
import tempfile
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from time import sleep
from typing import Any, Protocol
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import requests

RESEARCH_ASSET_SCHEMA_VERSION = "research-asset-manifest-v1"
RESEARCH_ASSET_RELATIVE_ROOT = Path("artifacts") / "research-assets"
RESEARCH_ASSET_QUOTA_RELATIVE_ROOT = Path("artifacts") / "research-asset-quota"
RESEARCH_ASSET_CONTENT_NAME = "content.pdf"
RESEARCH_ASSET_MANIFEST_NAME = "manifest.json"
RESEARCH_ASSET_MANIFEST_SHA256_NAME = "manifest.sha256"
DEFAULT_PDF_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_ATOM_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
DEFAULT_LOCAL_ASSET_MAX_BYTES = 20 * 1024 * 1024 * 1024
DEFAULT_LOCAL_ASSET_MAX_FILES = 10_000
TUSHARE_RESEARCH_REPORT_DAILY_LIMIT = 20
ARXIV_DAILY_LIMIT = 3
TUSHARE_RESEARCH_REPORT_SELECTION_POLICY = "tushare-research-report-daily-v1"
ARXIV_SELECTION_POLICY = "arxiv-quant-daily-v1"

# This is the exact benchmark registry exposed by the pinned RD-Agent commit.
# Do not broaden it from user metadata: an unknown value otherwise reaches an
# upstream dictionary lookup only after the expensive GPU job has started.
FINETUNE_BENCHMARK_ALLOWLIST = frozenset(
    {
        "FinanceIQ_gen",
        "aime24",
        "aime25",
        "bioprobench_err",
        "bioprobench_gen",
        "bioprobench_ord",
        "bioprobench_pqa",
        "chemcotbench",
        "chemcotbench_mol_edit",
        "chemcotbench_mol_opt",
        "chemcotbench_mol_und",
        "chemcotbench_reaction",
        "humaneval",
        "math",
        "mbpp",
        "mmlu",
        "panorama",
        "panorama_noc4pc",
        "panorama_noc4pc_cot",
        "panorama_par4pc",
        "panorama_par4pc_cot",
        "panorama_pi4pc",
        "panorama_pi4pc_cot",
        "tablebench_data_analysis",
        "tablebench_fact_checking",
        "tablebench_gen",
        "tablebench_numerical_reasoning",
        "tablebench_visualization",
    }
)
FINETUNE_FINANCEIQ_DATA_ROOT = "benchmarks/opencompass_data/data/FinanceIQ"
_FINETUNE_GOVERNANCE_SUBJECTS = ("model", "dataset", "benchmark")
_FINETUNE_GOVERNANCE_EVIDENCE = {
    subject: {
        "revision": f"governance/{subject}/revision.txt",
        "license_terms": f"governance/{subject}/license-terms.txt",
    }
    for subject in _FINETUNE_GOVERNANCE_SUBJECTS
}
_FINETUNE_CONTRACT_FILE_MAX_BYTES = 4 * 1024 * 1024

ARXIV_QFIN_CATEGORIES = (
    "q-fin.CP",
    "q-fin.EC",
    "q-fin.GN",
    "q-fin.MF",
    "q-fin.PM",
    "q-fin.PR",
    "q-fin.RM",
    "q-fin.ST",
    "q-fin.TR",
)
ARXIV_TARGET_CATEGORIES = frozenset((*ARXIV_QFIN_CATEGORIES, "cs.LG", "stat.ML"))
ARXIV_DISCOVERY_CATEGORY_GROUPS = (
    ARXIV_QFIN_CATEGORIES,
    ("cs.LG",),
    ("stat.ML",),
)
ARXIV_DISCOVERY_HOST = "export.arxiv.org"
ARXIV_PDF_HOST = "arxiv.org"

_ASSET_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,95}\Z")
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_RESERVED_ASSET_FILENAMES = frozenset(
    {
        RESEARCH_ASSET_MANIFEST_NAME.casefold(),
        RESEARCH_ASSET_MANIFEST_SHA256_NAME.casefold(),
    }
)
_ARXIV_ID_PATTERN = re.compile(
    r"(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})(?:v\d+)?\Z",
    re.IGNORECASE,
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_IMMUTABLE_REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SAFE_REFERENCE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:+-]{0,255}\Z")
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_XML_MEDIA_TYPES = frozenset(
    {"application/atom+xml", "application/xml", "text/xml"}
)
_PDF_EOF_SCAN_BYTES = 16 * 1024
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_Resolver = Callable[[str, int], Sequence[str]]
_Clock = Callable[[], datetime]
_Sleeper = Callable[[float], None]


class ResearchAssetError(RuntimeError):
    """Base error for governed external research assets."""


class UnsafeResearchAssetUrl(ResearchAssetError):
    """The requested URL can reach a non-public or otherwise unsafe target."""


class ResearchAssetDownloadError(ResearchAssetError):
    """A remote body failed transport or content validation."""


class ResearchAssetDiscoveryError(ResearchAssetError):
    """An automatic discovery feed violated its declared contract."""


class ResearchAssetConflictError(ResearchAssetError):
    """An existing immutable asset disagrees with newly acquired content."""


class ResearchAssetNotYetAvailable(ResearchAssetError):
    """A PIT-governed candidate has not reached its allowed publication time."""


class ResearchAssetPermissionError(ResearchAssetError):
    """The immutable snapshot records that a source could not be downloaded."""


@dataclass(frozen=True, slots=True)
class ResearchAssetIngestionSummary:
    snapshot_name: str
    tushare_selected: int
    arxiv_selected: int
    published: tuple[PublishedResearchAsset, ...]
    blocked: tuple[Mapping[str, object], ...]
    blocked_ledger_path: Path
    imported: tuple[object, ...] = ()

    @property
    def failed(self) -> int:
        return sum(1 for item in self.blocked if item.get("severity") == "error")

    def as_dict(self) -> dict[str, object]:
        return {
            "status": "blocked" if self.failed else "succeeded",
            "snapshot_name": self.snapshot_name,
            "tushare_selected": self.tushare_selected,
            "arxiv_selected": self.arxiv_selected,
            "published": len(self.published),
            "published_asset_ids": [item.asset_id for item in self.published],
            "blocked": len(self.blocked),
            "failed": self.failed,
            "blocked_ledger_path": str(self.blocked_ledger_path),
            "imported": len(self.imported),
        }


class ResearchAssetManifestImporter(Protocol):
    """Database-free boundary implemented by a future ResearchAssetStore."""

    def import_manifest(self, manifest_path: Path) -> object:
        """Import one verified on-disk manifest and its sibling content."""


class ResearchAssetDailyQuotaLedger(Protocol):
    """Atomic, database-free reservation boundary for automatic daily sources."""

    def remaining(
        self,
        *,
        source_kind: str,
        selection_day: date,
        daily_limit: int,
    ) -> int:
        """Return the fail-closed number of source slots still available."""

    def reserve(
        self,
        *,
        source_kind: str,
        selection_day: date,
        source_ids: Sequence[str],
        daily_limit: int,
        reserved_at: datetime,
    ) -> tuple[str, ...]:
        """Atomically reserve at most the remaining source slots."""


@dataclass(frozen=True, slots=True)
class DownloadedResource:
    requested_url: str
    final_url: str
    redirect_chain: tuple[str, ...]
    media_type: str
    body: bytes
    sha256: str

    @property
    def size_bytes(self) -> int:
        return len(self.body)


@dataclass(frozen=True, slots=True)
class VerifiedPdf:
    requested_url: str
    final_url: str
    redirect_chain: tuple[str, ...]
    media_type: str
    body: bytes
    sha256: str

    @property
    def size_bytes(self) -> int:
        return len(self.body)


@dataclass(frozen=True, slots=True)
class ResearchAssetCandidate:
    source_kind: str
    source_id: str
    title: str
    pdf_url: str
    published_at: datetime
    available_at: datetime
    authors: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    abstract: str = ""
    asset_type: str = "research_document"
    availability_rule: str = "source-publication-timestamp"
    selection_policy: str = "manual"
    selection_rank: int = 1
    selection_score: float = 0.0
    selection_daily_limit: int = 1
    selection_as_of: date | None = None
    source_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("source_kind", "source_id", "title", "pdf_url", "asset_type"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be blank")
        if len(self.source_id) > 4_096 or len(self.pdf_url) > 4_096:
            raise ValueError("source_id and pdf_url must not exceed 4096 characters")
        if len(self.asset_type) > 128 or any(
            ord(character) < 32 for character in self.asset_type
        ):
            raise ValueError("asset_type must be 1-128 printable characters")
        if len(self.title) > 4_000 or len(self.abstract) > 200_000:
            raise ValueError("research asset text exceeds the governed metadata limit")
        _require_aware(self.published_at, "published_at")
        _require_aware(self.available_at, "available_at")
        if self.available_at < self.published_at:
            raise ValueError("available_at must not be before published_at")
        if self.selection_rank < 1:
            raise ValueError("selection_rank must be positive")
        if self.selection_daily_limit < 1:
            raise ValueError("selection_daily_limit must be positive")
        if not math.isfinite(self.selection_score):
            raise ValueError("selection_score must be finite")
        try:
            normalized_metadata = _json_mapping(self.source_metadata)
            json.dumps(normalized_metadata, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("source_metadata is not safely JSON serializable") from exc


@dataclass(frozen=True, slots=True)
class PublishedResearchAsset:
    asset_id: str
    directory: Path
    manifest_path: Path
    content_path: Path
    manifest: Mapping[str, object]


def research_asset_root(data_root: Path) -> Path:
    """Return the fixed runtime-visible research asset directory."""

    return Path(data_root) / RESEARCH_ASSET_RELATIVE_ROOT


def _ensure_research_asset_root(data_root: Path) -> Path:
    """Create the fixed asset root without accepting links in its ancestry."""

    base = Path(data_root)
    base.mkdir(parents=True, exist_ok=True)
    if _path_is_linkish(base):
        raise ResearchAssetConflictError("configured DATA_ROOT must not be a link")
    artifacts = base / RESEARCH_ASSET_RELATIVE_ROOT.parts[0]
    artifacts.mkdir(exist_ok=True)
    if _path_is_linkish(artifacts) or not artifacts.is_dir():
        raise ResearchAssetConflictError("research artifact root must be a real directory")
    root = research_asset_root(base)
    root.mkdir(exist_ok=True)
    if _path_is_linkish(root) or not root.is_dir():
        raise ResearchAssetConflictError("research asset root must be a real directory")
    _require_root_inside_data_root(base, root)
    return root


class FilesystemResearchAssetDailyQuotaLedger:
    """Reserve automatic-source slots with atomic directory creation.

    A reservation is durable even when the subsequent PDF download is blocked.
    That makes a daily limit a limit on selected acquisition attempts, rather
    than a per-process limit that can be reset by retrying the CLI command.
    Database-backed deployments can inject the protocol above instead.
    """

    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root)

    def remaining(
        self,
        *,
        source_kind: str,
        selection_day: date,
        daily_limit: int,
    ) -> int:
        if daily_limit < 1:
            raise ValueError("daily_limit must be positive")
        day_root = self._day_root(source_kind, selection_day)
        slots = day_root / "slots"
        used_slots = sum(
            1
            for path in slots.iterdir()
            if path.is_dir() and not _path_is_linkish(path)
        )
        return max(0, daily_limit - min(daily_limit, used_slots))

    def reserve(
        self,
        *,
        source_kind: str,
        selection_day: date,
        source_ids: Sequence[str],
        daily_limit: int,
        reserved_at: datetime,
    ) -> tuple[str, ...]:
        _require_aware(reserved_at, "reserved_at")
        if daily_limit < 1:
            raise ValueError("daily_limit must be positive")
        day_root = self._day_root(source_kind, selection_day)
        slots = day_root / "slots"
        sources = day_root / "sources"
        accepted: list[str] = []
        for raw_source_id in source_ids:
            source_id = str(raw_source_id).strip()
            if not source_id:
                continue
            source_key = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
            source_claim = sources / source_key
            if source_claim.exists() or _path_is_linkish(source_claim):
                continue
            claimed_slot: Path | None = None
            for slot_number in range(1, daily_limit + 1):
                candidate_slot = slots / f"{slot_number:03d}"
                try:
                    candidate_slot.mkdir()
                except FileExistsError:
                    continue
                claimed_slot = candidate_slot
                break
            if claimed_slot is None:
                break
            try:
                source_claim.mkdir()
            except FileExistsError:
                # Another process reserved this source first. Release only the
                # exact empty slot created by this process and try the next ID.
                claimed_slot.rmdir()
                continue
            reservation = {
                "schema_version": "research-asset-daily-quota-v1",
                "source_kind": source_kind,
                "source_id": source_id,
                "source_sha256": source_key,
                "selection_day": selection_day.isoformat(),
                "daily_limit": daily_limit,
                "reserved_at": reserved_at.astimezone(UTC).isoformat(),
            }
            reservation_bytes = (
                json.dumps(
                    reservation,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            try:
                _write_new_file(claimed_slot / "reservation.json", reservation_bytes)
                _write_new_file(source_claim / "reservation.json", reservation_bytes)
            except Exception:
                # Keep the atomic slot/source directories as fail-closed quota
                # evidence if a process dies during reservation publication.
                raise
            accepted.append(source_id)
        return tuple(accepted)

    def _day_root(self, source_kind: str, selection_day: date) -> Path:
        normalized_kind = str(source_kind).strip().lower()
        if not _ASSET_ID_PATTERN.fullmatch(normalized_kind):
            raise ValueError("quota source_kind must be a safe lowercase identifier")
        base = self.data_root
        base.mkdir(parents=True, exist_ok=True)
        if _path_is_linkish(base):
            raise ResearchAssetConflictError("configured DATA_ROOT must not be a link")
        artifacts = base / RESEARCH_ASSET_QUOTA_RELATIVE_ROOT.parts[0]
        artifacts.mkdir(exist_ok=True)
        if _path_is_linkish(artifacts) or not artifacts.is_dir():
            raise ResearchAssetConflictError("research artifact root must be a real directory")
        root = base / RESEARCH_ASSET_QUOTA_RELATIVE_ROOT
        root.mkdir(exist_ok=True)
        if _path_is_linkish(root) or not root.is_dir():
            raise ResearchAssetConflictError("research asset quota root must be a real directory")
        _require_root_inside_data_root(base, root)
        source_root = root / normalized_kind
        source_root.mkdir(exist_ok=True)
        if _path_is_linkish(source_root) or not source_root.is_dir():
            raise ResearchAssetConflictError(
                "research asset quota source root must be a real directory"
            )
        day_root = source_root / selection_day.isoformat()
        day_root.mkdir(exist_ok=True)
        if _path_is_linkish(day_root) or not day_root.is_dir():
            raise ResearchAssetConflictError(
                "research asset quota day root must be a real directory"
            )
        (day_root / "slots").mkdir(exist_ok=True)
        (day_root / "sources").mkdir(exist_ok=True)
        for path in (
            source_root,
            day_root,
            day_root / "slots",
            day_root / "sources",
        ):
            if _path_is_linkish(path) or not path.is_dir():
                raise ResearchAssetConflictError(
                    "research asset quota path must not contain links"
                )
        return day_root


def validate_asset_id(asset_id: str) -> str:
    """Reject path components and ambiguous names before local materialization."""

    value = str(asset_id).strip()
    if not _ASSET_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "asset_id must be 1-96 lowercase ASCII letters, digits, '_' or '-'"
        )
    if value in {".", ".."} or ".." in value:
        raise ValueError("asset_id must not contain a parent-directory marker")
    if value.endswith(".") or value.split(".", 1)[0] in _WINDOWS_RESERVED_NAMES:
        raise ValueError("asset_id is not portable across supported filesystems")
    return value


def derive_asset_id(source_kind: str, source_id: str) -> str:
    """Create a stable, path-safe identity without exposing a source URL."""

    prefix = re.sub(r"[^a-z0-9]+", "-", source_kind.lower()).strip("-")
    if not prefix:
        prefix = "research"
    prefix = prefix[:48].rstrip("-")
    material = f"{source_kind}\0{source_id}".encode()
    suffix = hashlib.sha256(material).hexdigest()[:24]
    return validate_asset_id(f"{prefix}-{suffix}")


def _default_resolver(host: str, port: int) -> Sequence[str]:
    values = {
        str(address[4][0])
        for address in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    }
    return sorted(values)


def validate_public_https_url(
    url: str,
    *,
    resolver: _Resolver = _default_resolver,
    allowed_hosts: frozenset[str] | None = None,
) -> str:
    """Validate HTTPS and every resolved address, including redirect targets.

    Proxy environment variables are separately disabled by ``SafeHttpClient``.
    Rejecting a host when *any* answer is non-public prevents a hostname from
    mixing a public address with an internal fallback.
    """

    value = str(url).strip()
    if not value or len(value) > 4_096:
        raise UnsafeResearchAssetUrl("URL is blank or exceeds 4096 characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise UnsafeResearchAssetUrl("URL contains a control character")
    if "\\" in value:
        raise UnsafeResearchAssetUrl("URL contains an ambiguous backslash")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise UnsafeResearchAssetUrl(f"invalid URL: {exc}") from exc
    if parts.scheme.lower() != "https":
        raise UnsafeResearchAssetUrl("only HTTPS research asset URLs are allowed")
    if parts.username is not None or parts.password is not None:
        raise UnsafeResearchAssetUrl("credentials are not allowed in research asset URLs")
    if port not in {None, 443}:
        raise UnsafeResearchAssetUrl("only the standard HTTPS port is allowed")
    if parts.fragment:
        raise UnsafeResearchAssetUrl("URL fragments are not allowed")
    if not parts.hostname:
        raise UnsafeResearchAssetUrl("URL hostname is missing")
    try:
        host = parts.hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise UnsafeResearchAssetUrl("URL hostname is not valid IDNA") from exc
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise UnsafeResearchAssetUrl("local hostnames are not allowed")
    if allowed_hosts is not None and host not in allowed_hosts:
        raise UnsafeResearchAssetUrl(f"hostname is outside the source allowlist: {host}")
    try:
        addresses = tuple(resolver(host, 443))
    except (OSError, ValueError) as exc:
        raise UnsafeResearchAssetUrl(f"hostname cannot be safely resolved: {host}") from exc
    if not addresses:
        raise UnsafeResearchAssetUrl(f"hostname has no resolved address: {host}")
    for raw_address in addresses:
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError as exc:
            raise UnsafeResearchAssetUrl(
                f"hostname returned an invalid IP address: {raw_address}"
            ) from exc
        if not address.is_global:
            raise UnsafeResearchAssetUrl(
                f"hostname resolves to a non-public address: {raw_address}"
            )
    display_host = f"[{host}]" if ":" in host else host
    return urlunsplit(("https", display_host, parts.path or "/", parts.query, ""))


class SafeHttpClient:
    """Bounded, redirect-aware HTTP reader with SSRF and decompression guards."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        resolver: _Resolver = _default_resolver,
        timeout: tuple[float, float] = (5.0, 30.0),
        max_redirects: int = 3,
    ) -> None:
        if max_redirects < 0:
            raise ValueError("max_redirects must not be negative")
        self.session = session or requests.Session()
        # Environment proxies can route a validated public URL to an internal
        # endpoint, so they are intentionally unsupported in this trust path.
        self.session.trust_env = False
        self.resolver = resolver
        self.timeout = timeout
        self.max_redirects = max_redirects

    def fetch(
        self,
        url: str,
        *,
        accepted_media_types: frozenset[str],
        max_bytes: int,
        allowed_hosts: frozenset[str] | None = None,
    ) -> DownloadedResource:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        if not accepted_media_types:
            raise ValueError("accepted_media_types must not be empty")
        requested_url = validate_public_https_url(
            url,
            resolver=self.resolver,
            allowed_hosts=allowed_hosts,
        )
        current_url = requested_url
        redirect_chain: list[str] = []
        while True:
            current_url = validate_public_https_url(
                current_url,
                resolver=self.resolver,
                allowed_hosts=allowed_hosts,
            )
            try:
                response = self.session.get(
                    current_url,
                    allow_redirects=False,
                    stream=True,
                    timeout=self.timeout,
                    headers={
                        "Accept": ", ".join(sorted(accepted_media_types)),
                        "Accept-Encoding": "identity",
                        "User-Agent": "QuantLab-Research-Asset/1.0",
                    },
                )
            except requests.RequestException as exc:
                raise ResearchAssetDownloadError(
                    f"research asset request failed: {current_url}: {exc}"
                ) from exc
            try:
                status_code = int(response.status_code)
                # DNS is resolved before every request, then the actual socket
                # peer is checked as well. This closes the rebinding gap
                # between hostname validation and the connection itself.
                _reject_non_public_peer(response)
                if status_code in _REDIRECT_STATUS_CODES:
                    if len(redirect_chain) >= self.max_redirects:
                        raise ResearchAssetDownloadError("too many research asset redirects")
                    location = _header(response, "Location")
                    if not location:
                        raise ResearchAssetDownloadError("redirect response has no Location")
                    target = validate_public_https_url(
                        urljoin(current_url, location),
                        resolver=self.resolver,
                        allowed_hosts=allowed_hosts,
                    )
                    redirect_chain.append(target)
                    current_url = target
                    continue
                if status_code != 200:
                    raise ResearchAssetDownloadError(
                        f"research asset returned HTTP {status_code}: {current_url}"
                    )
                encoding = _header(response, "Content-Encoding").strip().lower()
                if encoding not in {"", "identity"}:
                    raise ResearchAssetDownloadError(
                        f"compressed responses are rejected: {encoding}"
                    )
                media_type = _header(response, "Content-Type").split(";", 1)[0].strip().lower()
                if media_type not in accepted_media_types:
                    raise ResearchAssetDownloadError(
                        f"unexpected research asset Content-Type: {media_type or '<missing>'}"
                    )
                declared_size = _declared_content_length(response)
                if declared_size is not None and declared_size > max_bytes:
                    raise ResearchAssetDownloadError(
                        f"research asset exceeds {max_bytes} byte limit"
                    )
                body = bytearray()
                try:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            raise ResearchAssetDownloadError(
                                f"research asset exceeds {max_bytes} byte limit"
                            )
                except requests.RequestException as exc:
                    raise ResearchAssetDownloadError(
                        f"research asset body read failed: {current_url}: {exc}"
                    ) from exc
                payload = bytes(body)
                if declared_size is not None and len(payload) != declared_size:
                    raise ResearchAssetDownloadError(
                        "research asset Content-Length does not match received bytes"
                    )
                return DownloadedResource(
                    requested_url=requested_url,
                    final_url=current_url,
                    redirect_chain=tuple(redirect_chain),
                    media_type=media_type,
                    body=payload,
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
            finally:
                response.close()


class SecurePdfDownloader:
    """Download a PDF only after transport, MIME, size and magic validation."""

    def __init__(
        self,
        *,
        http_client: SafeHttpClient | None = None,
        max_bytes: int = DEFAULT_PDF_MAX_BYTES,
    ) -> None:
        if not 1 <= max_bytes <= DEFAULT_PDF_MAX_BYTES:
            raise ValueError(
                f"max_bytes must be between 1 and {DEFAULT_PDF_MAX_BYTES}"
            )
        self.http_client = http_client or SafeHttpClient()
        self.max_bytes = max_bytes

    def download(
        self,
        url: str,
        *,
        allowed_hosts: frozenset[str] | None = None,
    ) -> VerifiedPdf:
        resource = self.http_client.fetch(
            url,
            accepted_media_types=frozenset({"application/pdf"}),
            max_bytes=self.max_bytes,
            allowed_hosts=allowed_hosts,
        )
        if not resource.body.startswith(b"%PDF-"):
            raise ResearchAssetDownloadError("PDF body does not start with %PDF-")
        if b"%%EOF" not in resource.body[-_PDF_EOF_SCAN_BYTES:]:
            raise ResearchAssetDownloadError("PDF body has no terminal %%EOF marker")
        return VerifiedPdf(
            requested_url=resource.requested_url,
            final_url=resource.final_url,
            redirect_chain=resource.redirect_chain,
            media_type=resource.media_type,
            body=resource.body,
            sha256=resource.sha256,
        )


def select_tushare_research_reports(
    rows: Iterable[Mapping[str, object]],
    *,
    open_days: Sequence[date],
    daily_limit: int = TUSHARE_RESEARCH_REPORT_DAILY_LIMIT,
) -> list[ResearchAssetCandidate]:
    """Deterministically select at most 20 reports per publication day.

    ``available_at`` is derived from the real exchange calendar and fails
    closed when the calendar does not extend past a report date.
    """

    if not 1 <= daily_limit <= TUSHARE_RESEARCH_REPORT_DAILY_LIMIT:
        raise ValueError("daily_limit must be between 1 and 20")
    calendar = sorted(set(open_days))
    if not calendar:
        raise ValueError("open_days must not be empty")
    grouped: dict[date, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        report_date = _parse_source_date(row.get("trade_date"))
        url = str(row.get("url") or "").strip()
        title = str(row.get("title") or row.get("file_name") or "").strip()
        if not url or not title:
            continue
        grouped[report_date].setdefault(url, row)

    selected: list[ResearchAssetCandidate] = []
    for report_date, by_url in sorted(grouped.items()):
        next_open_day = _next_open_day(report_date, calendar)
        published_at = datetime.combine(report_date, time.min, tzinfo=_SHANGHAI)
        available_at = datetime.combine(next_open_day, time.min, tzinfo=_SHANGHAI)
        ranked = sorted(
            by_url.items(),
            key=lambda item: (
                -_tushare_report_score(item[1]),
                str(item[1].get("title") or item[1].get("file_name") or ""),
                item[0],
            ),
        )[:daily_limit]
        for rank, (url, row) in enumerate(ranked, start=1):
            title = str(row.get("title") or row.get("file_name") or "").strip()
            authors = tuple(
                part.strip()
                for part in re.split(r"[,，;；、]", str(row.get("author") or ""))
                if part.strip()
            )
            categories = tuple(
                value
                for value in (
                    str(row.get("report_type") or "").strip(),
                    str(row.get("ind_name") or "").strip(),
                )
                if value
            )
            selected.append(
                ResearchAssetCandidate(
                    source_kind="tushare_research_report",
                    source_id=url,
                    title=title,
                    pdf_url=url,
                    published_at=published_at,
                    available_at=available_at,
                    authors=authors,
                    categories=categories,
                    abstract=str(row.get("abstr") or "").strip(),
                    asset_type="research_report",
                    availability_rule="first-open-trading-day-after-trade-date-v1",
                    selection_policy=TUSHARE_RESEARCH_REPORT_SELECTION_POLICY,
                    selection_rank=rank,
                    selection_score=float(_tushare_report_score(row)),
                    selection_daily_limit=daily_limit,
                    selection_as_of=report_date,
                    source_metadata=_json_mapping(row),
                )
            )
    return selected


def build_arxiv_discovery_url(
    as_of: date,
    *,
    lookback_days: int = 7,
    max_results: int = 100,
    categories: Iterable[str] = ARXIV_TARGET_CATEGORIES,
) -> str:
    """Build the official Atom query for q-fin, cs.LG and stat.ML."""

    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    if not 1 <= max_results <= 200:
        raise ValueError("max_results must be between 1 and 200")
    selected_categories = tuple(sorted(set(categories)))
    if not selected_categories or not set(selected_categories) <= ARXIV_TARGET_CATEGORIES:
        raise ValueError("categories must be a non-empty subset of the governed arXiv set")
    category_query = " OR ".join(f"cat:{category}" for category in selected_categories)
    start = as_of - timedelta(days=lookback_days - 1)
    start_utc = datetime.combine(start, time.min, tzinfo=_SHANGHAI).astimezone(UTC)
    end_utc = datetime.combine(as_of, time(23, 59), tzinfo=_SHANGHAI).astimezone(UTC)
    # The arXiv submittedDate contract is GMT; ``as_of`` is the platform's
    # A-share (Asia/Shanghai) research day.
    date_query = f"submittedDate:[{start_utc:%Y%m%d%H%M} TO {end_utc:%Y%m%d%H%M}]"
    params = urlencode(
        {
            "search_query": f"({category_query}) AND {date_query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    return f"https://{ARXIV_DISCOVERY_HOST}/api/query?{params}"


class ArxivDiscoveryClient:
    """Discover and transparently rank a small daily arXiv candidate set."""

    def __init__(
        self,
        *,
        http_client: SafeHttpClient | None = None,
        max_feed_bytes: int = DEFAULT_ATOM_MAX_BYTES,
        sleeper: _Sleeper = sleep,
        inter_request_delay_seconds: float = 3.0,
    ) -> None:
        if not 1 <= max_feed_bytes <= DEFAULT_ATOM_MAX_BYTES:
            raise ValueError(
                f"max_feed_bytes must be between 1 and {DEFAULT_ATOM_MAX_BYTES}"
            )
        if inter_request_delay_seconds < 0:
            raise ValueError("inter_request_delay_seconds must not be negative")
        self.http_client = http_client or SafeHttpClient()
        self.max_feed_bytes = max_feed_bytes
        self.sleeper = sleeper
        self.inter_request_delay_seconds = inter_request_delay_seconds

    def discover(
        self,
        as_of: date,
        *,
        lookback_days: int = 7,
        max_results: int = 100,
        daily_limit: int = ARXIV_DAILY_LIMIT,
        excluded_source_ids: Iterable[str] = (),
    ) -> list[ResearchAssetCandidate]:
        candidates: list[ResearchAssetCandidate] = []
        for index, categories in enumerate(ARXIV_DISCOVERY_CATEGORY_GROUPS):
            url = build_arxiv_discovery_url(
                as_of,
                lookback_days=lookback_days,
                max_results=max_results,
                categories=categories,
            )
            resource = self.http_client.fetch(
                url,
                accepted_media_types=_XML_MEDIA_TYPES,
                max_bytes=self.max_feed_bytes,
                allowed_hosts=frozenset({ARXIV_DISCOVERY_HOST}),
            )
            candidates.extend(parse_arxiv_atom(resource.body))
            if index < len(ARXIV_DISCOVERY_CATEGORY_GROUPS) - 1:
                self.sleeper(self.inter_request_delay_seconds)
        return rank_arxiv_candidates(
            candidates,
            as_of=as_of,
            daily_limit=daily_limit,
            excluded_source_ids=excluded_source_ids,
        )


def parse_arxiv_atom(body: bytes) -> list[ResearchAssetCandidate]:
    """Parse a bounded Atom feed without allowing DTD/entity declarations."""

    upper = body.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ResearchAssetDiscoveryError("arXiv feed contains a forbidden XML declaration")
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as exc:
        raise ResearchAssetDiscoveryError(f"invalid arXiv Atom feed: {exc}") from exc
    atom = "{http://www.w3.org/2005/Atom}"
    if root.tag != f"{atom}feed":
        raise ResearchAssetDiscoveryError("arXiv response root is not an Atom feed")
    candidates: list[ResearchAssetCandidate] = []
    for entry in root.findall(f"{atom}entry"):
        entry_id = _xml_text(entry.find(f"{atom}id"))
        title = _xml_text(entry.find(f"{atom}title"))
        if "/api/errors" in entry_id or title.casefold() == "error":
            raise ResearchAssetDiscoveryError("arXiv API returned an error entry")
        published_at = _parse_source_datetime(_xml_text(entry.find(f"{atom}published")))
        updated_text = _xml_text(entry.find(f"{atom}updated"))
        updated_at = _parse_source_datetime(updated_text) if updated_text else published_at
        authors = tuple(
            value
            for author in entry.findall(f"{atom}author")
            if (value := _xml_text(author.find(f"{atom}name")))
        )
        categories = tuple(
            sorted(
                {
                    str(category.attrib.get("term") or "").strip()
                    for category in entry.findall(f"{atom}category")
                    if str(category.attrib.get("term") or "").strip()
                }
            )
        )
        arxiv_id = _arxiv_id_from_entry(entry, entry_id, atom)
        candidates.append(
            ResearchAssetCandidate(
                source_kind="arxiv",
                source_id=arxiv_id,
                title=title,
                pdf_url=f"https://{ARXIV_PDF_HOST}/pdf/{arxiv_id}",
                # ``updated`` is the publication time of the selected version;
                # using first-version ``published`` would leak a later revision.
                published_at=updated_at,
                available_at=updated_at,
                authors=authors,
                categories=categories,
                abstract=_xml_text(entry.find(f"{atom}summary")),
                asset_type="arxiv_paper",
                availability_rule="arxiv-version-updated-timestamp-v1",
                selection_policy=ARXIV_SELECTION_POLICY,
                source_metadata={
                    "first_published_at": published_at.isoformat(),
                    "updated_at": updated_at.isoformat(),
                    "entry_id": entry_id,
                },
            )
        )
    return candidates


def rank_arxiv_candidates(
    candidates: Iterable[ResearchAssetCandidate],
    *,
    as_of: date,
    daily_limit: int = ARXIV_DAILY_LIMIT,
    excluded_source_ids: Iterable[str] = (),
) -> list[ResearchAssetCandidate]:
    """Return at most three eligible candidates with an auditable score."""

    if not 1 <= daily_limit <= ARXIV_DAILY_LIMIT:
        raise ValueError("daily_limit must be between 1 and 3")
    excluded = {str(value).strip() for value in excluded_source_ids if str(value).strip()}
    eligible: dict[str, ResearchAssetCandidate] = {}
    for candidate in candidates:
        if candidate.source_kind != "arxiv":
            continue
        if candidate.source_id in excluded:
            continue
        local_published_date = candidate.published_at.astimezone(_SHANGHAI).date()
        if local_published_date > as_of:
            continue
        if not any(
            category in ARXIV_TARGET_CATEGORIES
            for category in candidate.categories
        ):
            continue
        existing = eligible.get(candidate.source_id)
        if existing is None or candidate.published_at > existing.published_at:
            eligible[candidate.source_id] = candidate
    ranked = sorted(
        eligible.values(),
        key=lambda item: (
            -_arxiv_score(item, as_of),
            -item.published_at.timestamp(),
            item.source_id,
        ),
    )[:daily_limit]
    return [
        replace(
            candidate,
            selection_policy=ARXIV_SELECTION_POLICY,
            selection_rank=rank,
            selection_score=float(_arxiv_score(candidate, as_of)),
            selection_daily_limit=daily_limit,
            selection_as_of=as_of,
        )
        for rank, candidate in enumerate(ranked, start=1)
    ]


def manual_https_pdf_candidate(
    *,
    url: str,
    title: str,
    acquired_at: datetime,
    source_id: str | None = None,
    published_at: datetime | None = None,
    authors: Sequence[str] = (),
    categories: Sequence[str] = (),
    asset_type: str = "manual_pdf",
    metadata: Mapping[str, object] | None = None,
) -> ResearchAssetCandidate:
    """Create a manual candidate whose PIT time cannot predate acquisition."""

    _require_aware(acquired_at, "acquired_at")
    if published_at is not None:
        _require_aware(published_at, "published_at")
    effective_published_at = published_at or acquired_at
    available_at = max(effective_published_at, acquired_at)
    return ResearchAssetCandidate(
        source_kind="manual_https",
        source_id=source_id or hashlib.sha256(url.strip().encode()).hexdigest(),
        title=title.strip(),
        pdf_url=url.strip(),
        published_at=effective_published_at,
        available_at=available_at,
        authors=tuple(str(author).strip() for author in authors if str(author).strip()),
        categories=tuple(
            str(category).strip() for category in categories if str(category).strip()
        ),
        asset_type=asset_type.strip(),
        availability_rule="not-before-local-acquisition-v1",
        selection_policy="manual-https-v1",
        source_metadata=_json_mapping(metadata or {}),
    )


def acquire_research_assets(
    candidates: Iterable[ResearchAssetCandidate],
    *,
    data_root: Path,
    downloader: SecurePdfDownloader | None = None,
    acquired_at: datetime | None = None,
    clock: _Clock = lambda: datetime.now(UTC),
) -> list[PublishedResearchAsset]:
    """Validate, download and immutably publish each selected candidate."""

    if acquired_at is not None:
        _require_aware(acquired_at, "acquired_at")
    pdf_downloader = downloader or SecurePdfDownloader()
    published: list[PublishedResearchAsset] = []
    for candidate in candidates:
        attempted_at = clock()
        _require_aware(attempted_at, "attempted_at")
        if attempted_at < candidate.available_at:
            raise ResearchAssetNotYetAvailable(
                f"research asset {candidate.source_id} is not available until "
                f"{candidate.available_at.isoformat()}"
            )
        allowed_hosts = (
            frozenset({ARXIV_PDF_HOST}) if candidate.source_kind == "arxiv" else None
        )
        pdf = pdf_downloader.download(candidate.pdf_url, allowed_hosts=allowed_hosts)
        verified_at = clock()
        _require_aware(verified_at, "verified_at")
        effective_acquired_at = max(acquired_at, verified_at) if acquired_at else verified_at
        effective_candidate = candidate
        if candidate.source_kind == "manual_https":
            effective_candidate = replace(
                candidate,
                available_at=max(candidate.published_at, verified_at),
            )
        published.append(
            materialize_research_asset(
                data_root,
                effective_candidate,
                pdf,
                acquired_at=effective_acquired_at,
            )
        )
    return published


def materialize_research_asset(
    data_root: Path,
    candidate: ResearchAssetCandidate,
    pdf: VerifiedPdf,
    *,
    acquired_at: datetime,
    asset_id: str | None = None,
) -> PublishedResearchAsset:
    """Atomically publish ``manifest.json`` and an immutable ``content.pdf``."""

    _require_aware(acquired_at, "acquired_at")
    if acquired_at < candidate.published_at:
        raise ValueError("acquired_at must not be before the source publication time")
    if acquired_at < candidate.available_at:
        raise ValueError("acquired_at must not be before the PIT availability time")
    resolved_asset_id = validate_asset_id(
        asset_id or derive_asset_id(candidate.source_kind, candidate.source_id)
    )
    _validate_verified_pdf(pdf)
    # Never let source metadata claim a URL that differs from the transport
    # actually validated and downloaded.
    try:
        expected_url = urlsplit(candidate.pdf_url)
        requested_url = urlsplit(pdf.requested_url)
        expected_port = expected_url.port or 443
        requested_port = requested_url.port or 443
    except ValueError as exc:
        raise ResearchAssetConflictError("verified PDF URL identity is invalid") from exc
    expected_identity = (
        expected_url.scheme.casefold(),
        expected_url.hostname.casefold() if expected_url.hostname else "",
        expected_port,
        expected_url.path or "/",
        expected_url.query,
    )
    requested_identity = (
        requested_url.scheme.casefold(),
        requested_url.hostname.casefold() if requested_url.hostname else "",
        requested_port,
        requested_url.path or "/",
        requested_url.query,
    )
    if requested_identity != expected_identity:
        raise ResearchAssetConflictError(
            "verified PDF requested URL disagrees with the selected candidate"
        )
    root = _ensure_research_asset_root(data_root)
    target = root / resolved_asset_id
    manifest = _build_manifest(
        resolved_asset_id,
        candidate,
        pdf,
        acquired_at=acquired_at,
    )
    if target.exists() or _path_is_linkish(target):
        return _load_matching_asset(target, manifest)
    if candidate.source_kind == "manual_https" and candidate.available_at < acquired_at:
        raise ValueError(
            "manual HTTPS assets must not become available before local verification"
        )

    stage = Path(tempfile.mkdtemp(prefix=f".{resolved_asset_id}-", dir=root))
    try:
        _write_new_file(stage / RESEARCH_ASSET_CONTENT_NAME, pdf.body)
        _write_sealed_manifest(stage, manifest)
        try:
            os.rename(stage, target)
        except OSError:
            # Unique stage directories plus the final atomic rename avoid
            # stale lock files. A concurrent publisher must produce the same
            # immutable source/hash contract or this validation fails.
            if target.exists() or _path_is_linkish(target):
                return _load_matching_asset(target, manifest)
            raise
        for path in (
            target / RESEARCH_ASSET_CONTENT_NAME,
            target / RESEARCH_ASSET_MANIFEST_NAME,
            target / RESEARCH_ASSET_MANIFEST_SHA256_NAME,
        ):
            path.chmod(0o444)
        return PublishedResearchAsset(
            asset_id=resolved_asset_id,
            directory=target,
            manifest_path=target / RESEARCH_ASSET_MANIFEST_NAME,
            content_path=target / RESEARCH_ASSET_CONTENT_NAME,
            manifest=manifest,
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def iter_research_asset_manifests(data_root: Path) -> list[Path]:
    """Return only safe, direct-child manifests in deterministic order."""

    root = research_asset_root(data_root)
    if not root.exists():
        return []
    if not root.is_dir() or _path_is_linkish(root):
        raise ResearchAssetConflictError("research asset root is not a real directory")
    _require_root_inside_data_root(data_root, root)
    manifests: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if child.name.startswith(".") or _path_is_linkish(child) or not child.is_dir():
            continue
        validate_asset_id(child.name)
        manifest_path = child / RESEARCH_ASSET_MANIFEST_NAME
        if manifest_path.is_file() and not _path_is_linkish(manifest_path):
            manifests.append(manifest_path)
    return manifests


def load_research_asset_manifest(manifest_path: Path) -> dict[str, object]:
    """Load and verify a materialized PDF, dataset, or finetune asset."""

    path = Path(manifest_path)
    if (
        path.name != RESEARCH_ASSET_MANIFEST_NAME
        or _path_is_linkish(path)
        or _path_is_linkish(path.parent)
    ):
        raise ResearchAssetConflictError("invalid research asset manifest path")
    asset_id = validate_asset_id(path.parent.name)
    sidecar_path = path.parent / RESEARCH_ASSET_MANIFEST_SHA256_NAME
    if not sidecar_path.is_file() or _path_is_linkish(sidecar_path):
        raise ResearchAssetConflictError("research asset manifest has no SHA-256 sidecar")
    try:
        if path.stat().st_size > DEFAULT_MANIFEST_MAX_BYTES:
            raise ResearchAssetConflictError("research asset manifest exceeds its size limit")
        if sidecar_path.stat().st_size > 256:
            raise ResearchAssetConflictError("research asset manifest sidecar is oversized")
        manifest_bytes = path.read_bytes()
        expected_manifest_sha256 = sidecar_path.read_text(encoding="ascii").strip().lower()
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchAssetConflictError(f"invalid research asset manifest: {path}") from exc
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256)
        or hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256
    ):
        raise ResearchAssetConflictError("research asset manifest SHA-256 is invalid")
    if not isinstance(payload, dict):
        raise ResearchAssetConflictError("research asset manifest must be a JSON object")
    if payload.get("schema_version") != RESEARCH_ASSET_SCHEMA_VERSION:
        raise ResearchAssetConflictError("unsupported research asset manifest schema")
    if payload.get("asset_id") != asset_id:
        raise ResearchAssetConflictError("manifest asset_id does not match its directory")
    for timestamp_name in ("published_at", "available_at", "acquired_at"):
        timestamp = _manifest_datetime(payload.get(timestamp_name), timestamp_name)
        payload[timestamp_name] = timestamp.isoformat()
    published_at = _manifest_datetime(payload["published_at"], "published_at")
    available_at = _manifest_datetime(payload["available_at"], "available_at")
    acquired_at = _manifest_datetime(payload["acquired_at"], "acquired_at")
    if available_at < published_at or acquired_at < available_at:
        raise ResearchAssetConflictError("research asset PIT timestamps are inconsistent")
    kind = str(payload.get("kind") or "")
    files = payload.get("files")
    if (
        kind not in {"pdf", "dataset", "finetune"}
        or not str(payload.get("type") or "").strip()
        or payload.get("status") != "ready"
        or not isinstance(files, list)
        or not files
        or len(files) > DEFAULT_LOCAL_ASSET_MAX_FILES
    ):
        raise ResearchAssetConflictError(
            "manifest is incompatible with the RD-Agent asset contract"
        )
    total_bytes = 0
    seen_paths: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ResearchAssetConflictError("research asset file entry is invalid")
        relative = str(entry.get("path") or "")
        if relative in seen_paths:
            raise ResearchAssetConflictError("research asset file entries contain duplicates")
        seen_paths.add(relative)
        total_bytes += _verify_materialized_file(path.parent, entry, require_pdf=kind == "pdf")
    if total_bytes > DEFAULT_LOCAL_ASSET_MAX_BYTES:
        raise ResearchAssetConflictError("research asset exceeds the governed total size limit")
    if kind == "pdf":
        content = payload.get("content")
        if not isinstance(content, dict) or files != [content]:
            raise ResearchAssetConflictError("manifest PDF content and files disagree")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ResearchAssetConflictError("research asset metadata must be an object")
    if kind == "finetune":
        try:
            accepted_at = _validate_finetune_metadata(
                metadata,
                file_paths=seen_paths,
                file_reader=lambda relative: (
                    path.parent / _safe_asset_relative_path(relative)
                ).read_bytes(),
            )
            if accepted_at > acquired_at:
                raise ValueError(
                    "finetune metadata accepted_at must not follow asset acquisition"
                )
        except ValueError as exc:
            raise ResearchAssetConflictError(str(exc)) from exc
    elif kind == "dataset" and payload.get("type") in {
        "kaggle_dataset",
        "kaggle_competition",
    }:
        try:
            from .kaggle_assets import validate_kaggle_research_asset_manifest

            validate_kaggle_research_asset_manifest(payload, path.parent)
        except (OSError, ValueError, ResearchAssetError) as exc:
            raise ResearchAssetConflictError(str(exc)) from exc
    return payload


def import_research_asset_manifests(
    data_root: Path,
    importer: ResearchAssetManifestImporter,
) -> list[object]:
    """Verify every fixed-path manifest before handing it to a store adapter."""

    imported: list[object] = []
    for manifest_path in iter_research_asset_manifests(data_root):
        load_research_asset_manifest(manifest_path)
        imported.append(importer.import_manifest(manifest_path))
    return imported


def register_local_research_asset(
    data_root: Path,
    *,
    asset_id: str,
    kind: str,
    source_path: Path,
    asset_type: str,
    metadata: Mapping[str, object] | None = None,
    priority: int = 0,
    source_kind: str | None = None,
    source_id: str | None = None,
    source_version: str | None = None,
    source_sha256: str | None = None,
    clock: _Clock = lambda: datetime.now(UTC),
    importer: ResearchAssetManifestImporter | None = None,
) -> PublishedResearchAsset:
    """Administrator-only local file/directory copy into the immutable asset root.

    This helper intentionally has no API binding. A caller must already have
    host filesystem authority; paths are copied, never retained as runtime
    references, and symlinks/devices are rejected.
    """

    resolved_asset_id = validate_asset_id(asset_id)
    normalized_kind = str(kind).strip().lower()
    if normalized_kind not in {"pdf", "dataset", "finetune"}:
        raise ValueError("kind must be pdf, dataset, or finetune")
    normalized_type = str(asset_type).strip()
    if (
        not normalized_type
        or len(normalized_type) > 128
        or any(ord(character) < 32 for character in normalized_type)
    ):
        raise ValueError("asset_type must be 1-128 characters")
    try:
        normalized_metadata = _json_mapping(metadata or {})
        json.dumps(normalized_metadata, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("research asset metadata is not safely JSON serializable") from exc
    if normalized_kind == "finetune":
        _validate_finetune_metadata(normalized_metadata)
    provenance_values = (source_kind, source_id, source_version, source_sha256)
    if any(value is not None for value in provenance_values):
        if not all(value is not None for value in provenance_values):
            raise ValueError(
                "external source provenance requires kind, id, version, and SHA-256"
            )
        normalized_source_kind = str(source_kind).strip()
        normalized_source_id = str(source_id).strip()
        normalized_source_version = str(source_version).strip()
        normalized_source_sha256 = str(source_sha256).strip()
        if (
            normalized_source_kind not in {"kaggle_dataset", "kaggle_competition"}
            or not 1 <= len(normalized_source_id) <= 256
            or not 1 <= len(normalized_source_version) <= 128
            or not _SHA256_PATTERN.fullmatch(normalized_source_sha256)
            or any(
                ord(character) < 32
                for value in (normalized_source_id, normalized_source_version)
                for character in value
            )
        ):
            raise ValueError("external source provenance is invalid")
        manifest_source: dict[str, object] = {
            "kind": normalized_source_kind,
            "source_id": normalized_source_id,
            "version": normalized_source_version,
            "sha256": normalized_source_sha256,
        }
    else:
        manifest_source = {
            "kind": "admin_local_copy",
            "source_id": resolved_asset_id,
        }
    timestamp = clock()
    _require_aware(timestamp, "registration timestamp")
    sources = _local_source_files(Path(source_path), kind=normalized_kind)
    if normalized_kind == "finetune":
        source_by_relative = {
            relative.as_posix(): source for source, relative in sources
        }
        accepted_at = _validate_finetune_metadata(
            normalized_metadata,
            file_paths=source_by_relative,
            file_reader=lambda relative: source_by_relative[relative].read_bytes(),
        )
        if accepted_at > timestamp:
            raise ValueError(
                "finetune metadata accepted_at must not follow asset registration"
            )
    if not 0 <= priority <= 1_000_000:
        raise ValueError("priority must be between 0 and 1000000")

    root = _ensure_research_asset_root(data_root)
    target = root / resolved_asset_id

    stage = Path(tempfile.mkdtemp(prefix=f".{resolved_asset_id}-", dir=root))
    try:
        entries: list[dict[str, object]] = []
        total_bytes = 0
        for source, relative in sources:
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            entry = _copy_local_file(
                source,
                destination,
                relative_path=relative,
                require_pdf=normalized_kind == "pdf",
                max_bytes=min(
                    DEFAULT_PDF_MAX_BYTES
                    if normalized_kind == "pdf"
                    else DEFAULT_LOCAL_ASSET_MAX_BYTES,
                    DEFAULT_LOCAL_ASSET_MAX_BYTES - total_bytes,
                ),
            )
            total_bytes += int(entry["bytes"])
            if total_bytes > DEFAULT_LOCAL_ASSET_MAX_BYTES:
                raise ValueError("local research asset exceeds the governed total size limit")
            entries.append(entry)
        manifest: dict[str, object] = {
            "schema_version": RESEARCH_ASSET_SCHEMA_VERSION,
            "asset_id": resolved_asset_id,
            "kind": normalized_kind,
            "type": normalized_type,
            "status": "ready",
            "selected_at": None,
            "priority": priority,
            "published_at": timestamp.isoformat(),
            "available_at": timestamp.isoformat(),
            "acquired_at": timestamp.astimezone(UTC).isoformat(),
            "availability_rule": "not-before-admin-local-registration-v1",
            "source": manifest_source,
            "files": entries,
            "metadata": normalized_metadata,
        }
        if normalized_kind == "pdf":
            manifest["content"] = dict(entries[0])
        _write_sealed_manifest(stage, manifest)
        if target.exists() or _path_is_linkish(target):
            existing = load_research_asset_manifest(target / RESEARCH_ASSET_MANIFEST_NAME)
            if not _same_local_contract(existing, manifest):
                raise ResearchAssetConflictError("immutable local asset content conflicts")
            manifest = existing
        else:
            try:
                os.rename(stage, target)
            except OSError as exc:
                if target.exists() or _path_is_linkish(target):
                    existing = load_research_asset_manifest(
                        target / RESEARCH_ASSET_MANIFEST_NAME
                    )
                    if not _same_local_contract(existing, manifest):
                        raise ResearchAssetConflictError(
                            "concurrent local asset content conflicts"
                        ) from exc
                    manifest = existing
                else:
                    raise
        for file_path in target.rglob("*"):
            if file_path.is_file():
                file_path.chmod(0o444)
        result = PublishedResearchAsset(
            asset_id=resolved_asset_id,
            directory=target,
            manifest_path=target / RESEARCH_ASSET_MANIFEST_NAME,
            content_path=target / _first_manifest_file_path(manifest),
            manifest=manifest,
        )
        load_research_asset_manifest(result.manifest_path)
        if importer is not None:
            importer.import_manifest(result.manifest_path)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def acquire_manual_https_pdf(
    data_root: Path,
    *,
    url: str,
    title: str,
    asset_id: str | None = None,
    asset_type: str = "manual_pdf",
    published_at: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    downloader: SecurePdfDownloader | None = None,
    clock: _Clock = lambda: datetime.now(UTC),
    importer: ResearchAssetManifestImporter | None = None,
) -> PublishedResearchAsset:
    """Administrator-controlled HTTPS PDF acquisition with no host-path input."""

    verified_pdf = (downloader or SecurePdfDownloader()).download(url)
    verified_at = clock()
    _require_aware(verified_at, "verified_at")
    candidate = manual_https_pdf_candidate(
        url=url,
        title=title,
        acquired_at=verified_at,
        published_at=published_at,
        asset_type=asset_type,
        metadata=metadata,
    )
    result = materialize_research_asset(
        data_root,
        candidate,
        verified_pdf,
        acquired_at=verified_at,
        asset_id=asset_id,
    )
    if importer is not None:
        importer.import_manifest(result.manifest_path)
    return result


def collected_source_ids(
    data_root: Path,
    source_kind: str,
    *,
    selection_day: date | None = None,
) -> set[str]:
    """Read verified manifests to prevent automatic discovery from repeating assets."""

    values: set[str] = set()
    for manifest_path in iter_research_asset_manifests(data_root):
        manifest = load_research_asset_manifest(manifest_path)
        source = manifest.get("source")
        if not isinstance(source, dict) or source.get("kind") != source_kind:
            continue
        if selection_day is not None:
            selection = manifest.get("selection")
            if (
                not isinstance(selection, dict)
                or selection.get("as_of") != selection_day.isoformat()
            ):
                continue
        source_id = str(source.get("source_id") or "").strip()
        if source_id:
            values.add(source_id)
    return values


def collected_source_count_for_day(
    data_root: Path,
    source_kind: str,
    selection_day: date,
) -> int:
    """Count sealed automatic selections for one governed research day.

    This persists the daily quota across process retries.  Without it, a
    partially successful arXiv job could exclude its first selections and then
    fetch another full daily batch on retry.
    """

    count = 0
    for manifest_path in iter_research_asset_manifests(data_root):
        manifest = load_research_asset_manifest(manifest_path)
        source = manifest.get("source")
        selection = manifest.get("selection")
        if not isinstance(source, dict) or not isinstance(selection, dict):
            continue
        if source.get("kind") != source_kind:
            continue
        try:
            as_of = date.fromisoformat(str(selection.get("as_of") or ""))
        except ValueError:
            continue
        if as_of == selection_day:
            count += 1
    return count


def ingest_research_assets(
    data_root: Path,
    *,
    snapshot_name: str,
    as_of: date,
    include_tushare: bool = True,
    include_arxiv: bool = True,
    tushare_report_date: date | None = None,
    downloader: SecurePdfDownloader | None = None,
    arxiv_client: ArxivDiscoveryClient | None = None,
    quota_ledger: ResearchAssetDailyQuotaLedger | None = None,
    importer: ResearchAssetManifestImporter | None = None,
    clock: _Clock = lambda: datetime.now(UTC),
) -> ResearchAssetIngestionSummary:
    """Acquire governed PDFs from one verified immutable snapshot and arXiv.

    Individual permission/source failures are appended to a local JSONL ledger
    and never create an asset directory. Snapshot lineage or content-integrity
    failures are fatal because continuing would destroy source provenance.
    """

    if not include_tushare and not include_arxiv:
        raise ValueError("at least one research asset source must be enabled")
    snapshot_path: Path | None = None
    manifest: dict[str, Any] = {}
    if include_tushare:
        snapshot_path, manifest = _load_verified_snapshot(data_root, snapshot_name)
    effective_downloader = downloader or SecurePdfDownloader()
    published: list[PublishedResearchAsset] = []
    blocked: list[dict[str, object]] = []
    tushare_selected = arxiv_selected = 0

    if include_tushare:
        assert snapshot_path is not None
        try:
            rows = _read_snapshot_rows(snapshot_path, manifest, "research_report")
        except ResearchAssetPermissionError as exc:
            blocked.append(
                _blocked_record(
                    source_kind="tushare_research_report",
                    source_id=f"snapshot:{snapshot_name}",
                    reason_code="source_permission_or_dataset_unavailable",
                    reason=str(exc),
                    attempted_at=clock(),
                    severity="error",
                )
            )
            rows = []
        if not rows and not any(
            item.get("reason_code") == "source_permission_or_dataset_unavailable"
            for item in blocked
        ):
            blocked.append(
                _blocked_record(
                    source_kind="tushare_research_report",
                    source_id=f"snapshot:{snapshot_name}",
                    reason_code="research_report_dataset_empty",
                    reason=(
                        "the verified snapshot contains no research_report metadata; "
                        "source entitlement or download coverage is not proven"
                    ),
                    attempted_at=clock(),
                    severity="error",
                )
            )
        if rows:
            open_days = _read_snapshot_trade_calendar(snapshot_path, manifest)
            if tushare_report_date is not None:
                try:
                    first_eligible_day = _next_open_day(tushare_report_date, open_days)
                except LookupError as exc:
                    raise ValueError(str(exc)) from exc
                if first_eligible_day > as_of:
                    raise ValueError(
                        "tushare_report_date is not PIT-eligible as of "
                        f"{as_of.isoformat()}; first eligible trading day is "
                        f"{first_eligible_day.isoformat()}"
                    )
            effective_report_date = tushare_report_date or _latest_eligible_report_date(
                rows,
                open_days=open_days,
                as_of=as_of,
            )
            if effective_report_date is None:
                tushare_candidates = []
            else:
                report_rows = [
                    row
                    for row in rows
                    if _parse_source_date(row.get("trade_date"))
                    == effective_report_date
                ]
                if tushare_report_date is not None and not report_rows:
                    blocked.append(
                        _blocked_record(
                            source_kind="tushare_research_report",
                            source_id=f"report-date:{effective_report_date.isoformat()}",
                            reason_code="requested_report_date_unavailable",
                            reason=(
                                "the verified snapshot contains no research_report rows "
                                f"for {effective_report_date.isoformat()}"
                            ),
                            attempted_at=clock(),
                            severity="error",
                            selection_as_of=effective_report_date,
                        )
                    )
                missing_pdf_rows = [
                    row for row in report_rows if not str(row.get("url") or "").strip()
                ]
                if missing_pdf_rows:
                    blocked.append(
                        _blocked_record(
                            source_kind="tushare_research_report",
                            source_id=f"report-date:{effective_report_date.isoformat()}",
                            reason_code="pdf_url_unavailable",
                            reason=(
                                f"{len(missing_pdf_rows)} research_report metadata rows "
                                "contain no usable PDF URL"
                            ),
                            attempted_at=clock(),
                            severity="error",
                            selection_as_of=effective_report_date,
                        )
                    )
                tushare_candidates = select_tushare_research_reports(
                    report_rows,
                    open_days=open_days,
                )
                existing_tushare = collected_source_ids(
                    data_root, "tushare_research_report"
                )
                tushare_candidates = [
                    candidate
                    for candidate in tushare_candidates
                    if candidate.source_id not in existing_tushare
                ]
                remaining = max(
                    0,
                    TUSHARE_RESEARCH_REPORT_DAILY_LIMIT
                    - collected_source_count_for_day(
                        data_root,
                        "tushare_research_report",
                        effective_report_date,
                    ),
                )
                tushare_candidates = tushare_candidates[:remaining]
            tushare_selected = len(tushare_candidates)
            acquired, failures = _acquire_candidates_individually(
                tushare_candidates,
                data_root=data_root,
                downloader=effective_downloader,
                clock=clock,
            )
            published.extend(acquired)
            blocked.extend(failures)

    if include_arxiv:
        client = arxiv_client or ArxivDiscoveryClient()
        effective_quota_ledger = quota_ledger or FilesystemResearchAssetDailyQuotaLedger(
            data_root
        )
        try:
            # Seed the durable quota when upgrading from manifests created
            # before the reservation ledger existed.
            existing_today = sorted(
                collected_source_ids(data_root, "arxiv", selection_day=as_of)
            )
            if existing_today:
                effective_quota_ledger.reserve(
                    source_kind="arxiv",
                    selection_day=as_of,
                    source_ids=existing_today,
                    daily_limit=ARXIV_DAILY_LIMIT,
                    reserved_at=clock(),
                )
            remaining = effective_quota_ledger.remaining(
                source_kind="arxiv",
                selection_day=as_of,
                daily_limit=ARXIV_DAILY_LIMIT,
            )
            if remaining:
                discovered = client.discover(
                    as_of,
                    daily_limit=remaining,
                    excluded_source_ids=collected_source_ids(data_root, "arxiv"),
                )
                used_slots = ARXIV_DAILY_LIMIT - remaining
                discovered = [
                    replace(
                        candidate,
                        selection_rank=used_slots + candidate.selection_rank,
                        selection_daily_limit=ARXIV_DAILY_LIMIT,
                        selection_as_of=as_of,
                    )
                    for candidate in discovered
                ]
                reserved_ids = set(
                    effective_quota_ledger.reserve(
                        source_kind="arxiv",
                        selection_day=as_of,
                        source_ids=[candidate.source_id for candidate in discovered],
                        daily_limit=ARXIV_DAILY_LIMIT,
                        reserved_at=clock(),
                    )
                )
                arxiv_candidates = [
                    candidate
                    for candidate in discovered
                    if candidate.source_id in reserved_ids
                ]
            else:
                arxiv_candidates = []
        except (ResearchAssetError, requests.RequestException, OSError) as exc:
            blocked.append(
                _blocked_record(
                    source_kind="arxiv",
                    source_id=f"discovery:{as_of.isoformat()}",
                    reason_code="discovery_unavailable",
                    reason=str(exc),
                    attempted_at=clock(),
                    severity="error",
                    selection_as_of=as_of,
                )
            )
            arxiv_candidates = []
        arxiv_selected = len(arxiv_candidates)
        acquired, failures = _acquire_candidates_individually(
            arxiv_candidates,
            data_root=data_root,
            downloader=effective_downloader,
            clock=clock,
        )
        published.extend(acquired)
        blocked.extend(failures)

    ledger_path = _append_blocked_records(
        data_root,
        snapshot_name=snapshot_name,
        records=blocked,
    )
    imported: tuple[object, ...] = ()
    if importer is not None:
        imported = tuple(importer.import_manifest(item.manifest_path) for item in published)
    return ResearchAssetIngestionSummary(
        snapshot_name=snapshot_name,
        tushare_selected=tushare_selected,
        arxiv_selected=arxiv_selected,
        published=tuple(published),
        blocked=tuple(blocked),
        blocked_ledger_path=ledger_path,
        imported=imported,
    )


def _load_verified_snapshot(
    data_root: Path,
    snapshot_name: str,
) -> tuple[Path, dict[str, Any]]:
    if (
        not snapshot_name
        or Path(snapshot_name).name != snapshot_name
        or snapshot_name in {".", ".."}
        or snapshot_name.startswith(".")
    ):
        raise ValueError("snapshot_name must be one safe path component")
    snapshots_root = (Path(data_root) / "snapshots").resolve(strict=True)
    unresolved_snapshot = snapshots_root / snapshot_name
    if _path_is_linkish(unresolved_snapshot):
        raise ValueError("snapshot_name resolves through a link or junction")
    snapshot_path = unresolved_snapshot.resolve(strict=True)
    if snapshot_path.parent != snapshots_root or _path_is_linkish(snapshot_path):
        raise ValueError("snapshot_name escapes the configured immutable snapshot root")
    from .snapshot_lineage import verify_snapshot_lineage

    manifest = verify_snapshot_lineage(snapshot_path)
    quality_gate = manifest.get("quality_gate")
    if not isinstance(quality_gate, dict) or quality_gate.get("ok") is not True:
        raise ValueError("research asset ingestion requires a passing snapshot quality gate")
    if manifest.get("profile") != "research-assets":
        raise ValueError(
            "research asset ingestion requires the isolated research-assets snapshot profile"
        )
    lineage_contract = manifest.get("lineage_contract")
    lineage_configuration = (
        lineage_contract.get("configuration")
        if isinstance(lineage_contract, dict)
        else None
    )
    if (
        not isinstance(lineage_contract, dict)
        or lineage_contract.get("kind") != "research_asset_source"
        or not isinstance(lineage_configuration, dict)
        or lineage_configuration.get("profile") != "research-assets"
    ):
        raise ValueError(
            "research asset ingestion requires research_asset_source lineage bound "
            "to the research-assets profile"
        )
    return snapshot_path, manifest


def validate_research_asset_source_snapshot(
    data_root: Path,
    snapshot_name: str,
) -> dict[str, Any]:
    """Verify lineage plus every required source-file hash before selection."""

    snapshot_path, manifest = _load_verified_snapshot(data_root, snapshot_name)
    for dataset in ("trade_cal", "research_report"):
        if not _snapshot_dataset_files(snapshot_path, manifest, dataset):
            raise ResearchAssetPermissionError(
                f"verified snapshot {snapshot_name!r} has no {dataset} source files"
            )
    return manifest


def _snapshot_dataset_files(
    snapshot_path: Path,
    manifest: Mapping[str, object],
    dataset: str,
) -> list[Path]:
    datasets = manifest.get("datasets")
    entry = datasets.get(dataset) if isinstance(datasets, dict) else None
    if not isinstance(entry, dict):
        return []
    entries = entry.get("files")
    if not isinstance(entries, list):
        raise ResearchAssetConflictError(f"snapshot {dataset} file manifest is invalid")
    files: list[Path] = []
    snapshot_root = snapshot_path.resolve(strict=True)
    for item in entries:
        if not isinstance(item, dict) or not item.get("path"):
            raise ResearchAssetConflictError(f"snapshot {dataset} file entry is invalid")
        relative = Path(str(item["path"]))
        if relative.is_absolute():
            raise ResearchAssetConflictError(f"snapshot {dataset} file path is absolute")
        target = (snapshot_path / relative).resolve(strict=True)
        try:
            target.relative_to(snapshot_root)
        except ValueError as exc:
            raise ResearchAssetConflictError(
                f"snapshot {dataset} file escapes the immutable snapshot"
            ) from exc
        current = snapshot_path
        for part in relative.parts:
            current /= part
            if _path_is_linkish(current):
                raise ResearchAssetConflictError(
                    f"snapshot {dataset} file path contains a link"
                )
        if not target.is_file() or _path_is_linkish(target):
            raise ResearchAssetConflictError(f"snapshot {dataset} file is not regular")
        expected_size = item.get("bytes")
        expected_sha256 = str(item.get("sha256") or "").lower()
        if target.stat().st_size != expected_size or not re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256
        ):
            raise ResearchAssetConflictError(f"snapshot {dataset} file evidence is invalid")
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ResearchAssetConflictError(f"snapshot {dataset} file digest disagrees")
        files.append(target)
    return files


def _read_snapshot_rows(
    snapshot_path: Path,
    manifest: Mapping[str, object],
    dataset: str,
) -> list[dict[str, object]]:
    files = _snapshot_dataset_files(snapshot_path, manifest, dataset)
    if not files:
        raise ResearchAssetPermissionError(
            f"verified snapshot {snapshot_path.name!r} has no {dataset} parquet; "
            "the Tushare research_report permission may be unavailable"
        )
    import pyarrow.dataset as arrow_dataset

    try:
        table = arrow_dataset.dataset([str(path) for path in files], format="parquet").to_table()
    except (OSError, ValueError) as exc:
        raise ResearchAssetConflictError(
            f"snapshot {dataset} parquet cannot be read"
        ) from exc
    return [dict(row) for row in table.to_pylist()]


def _read_snapshot_trade_calendar(
    snapshot_path: Path,
    manifest: Mapping[str, object],
) -> list[date]:
    rows = _read_snapshot_rows(snapshot_path, manifest, "trade_cal")
    days = sorted(
        {
            _parse_source_date(row.get("cal_date"))
            for row in rows
            if str(row.get("is_open") or "").strip().casefold()
            in {"1", "true", "t", "yes"}
        }
    )
    if not days:
        raise ResearchAssetConflictError("snapshot trade_cal has no open trading day")
    return days


def _latest_eligible_report_date(
    rows: Iterable[Mapping[str, object]],
    *,
    open_days: Sequence[date],
    as_of: date,
) -> date | None:
    eligible: list[date] = []
    calendar = sorted(set(open_days))
    for row in rows:
        report_date = _parse_source_date(row.get("trade_date"))
        index = bisect_right(calendar, report_date)
        if index < len(calendar) and calendar[index] <= as_of:
            eligible.append(report_date)
    return max(eligible, default=None)


def _acquire_candidates_individually(
    candidates: Iterable[ResearchAssetCandidate],
    *,
    data_root: Path,
    downloader: SecurePdfDownloader,
    clock: _Clock,
) -> tuple[list[PublishedResearchAsset], list[dict[str, object]]]:
    published: list[PublishedResearchAsset] = []
    blocked: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            published.extend(
                acquire_research_assets(
                    [candidate],
                    data_root=data_root,
                    downloader=downloader,
                    clock=clock,
                )
            )
        except ResearchAssetNotYetAvailable as exc:
            blocked.append(
                _blocked_record(
                    source_kind=candidate.source_kind,
                    source_id=candidate.source_id,
                    reason_code="pit_not_yet_available",
                    reason=str(exc),
                    attempted_at=clock(),
                    severity="info",
                    selection_as_of=candidate.selection_as_of,
                )
            )
        except (
            ResearchAssetDownloadError,
            UnsafeResearchAssetUrl,
            requests.RequestException,
            OSError,
        ) as exc:
            blocked.append(
                _blocked_record(
                    source_kind=candidate.source_kind,
                    source_id=candidate.source_id,
                    reason_code="pdf_unavailable_or_unsafe",
                    reason=str(exc),
                    attempted_at=clock(),
                    severity="error",
                    url=candidate.pdf_url,
                    selection_as_of=candidate.selection_as_of,
                )
            )
    return published, blocked


def _blocked_record(
    *,
    source_kind: str,
    source_id: str,
    reason_code: str,
    reason: str,
    attempted_at: datetime,
    severity: str,
    url: str | None = None,
    selection_as_of: date | None = None,
) -> dict[str, object]:
    _require_aware(attempted_at, "attempted_at")
    return {
        "schema_version": "research-asset-blocked-v1",
        "source_kind": source_kind,
        "source_id": source_id,
        "reason_code": reason_code,
        "reason": reason[:2_000],
        "severity": severity,
        "attempted_at": attempted_at.astimezone(UTC).isoformat(),
        **({"url": url} if url else {}),
        **(
            {"selection_as_of": selection_as_of.isoformat()}
            if selection_as_of is not None
            else {}
        ),
    }


def _append_blocked_records(
    data_root: Path,
    *,
    snapshot_name: str,
    records: Sequence[Mapping[str, object]],
) -> Path:
    root = _ensure_research_asset_root(data_root)
    ledger_path = root / "blocked.jsonl"
    if _path_is_linkish(ledger_path):
        raise ResearchAssetConflictError("research asset blocked ledger must not be a symlink")
    if not records:
        return ledger_path
    with ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            payload = {"snapshot_name": snapshot_name, **dict(record)}
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    return ledger_path


def _build_manifest(
    asset_id: str,
    candidate: ResearchAssetCandidate,
    pdf: VerifiedPdf,
    *,
    acquired_at: datetime,
) -> dict[str, object]:
    return {
        "schema_version": RESEARCH_ASSET_SCHEMA_VERSION,
        "asset_id": asset_id,
        # These compatibility fields are the existing RD-Agent runtime
        # contract; the richer fields below remain the store import contract.
        "kind": "pdf",
        "type": candidate.asset_type,
        "status": "ready",
        "selected_at": None,
        # The platform's automatic selector sorts priority descending. Encode
        # availability recency before the within-day rank so a newer governed
        # source is never displaced by an older rank-1 asset.
        "priority": _automatic_selection_priority(candidate),
        "title": candidate.title,
        "authors": list(candidate.authors),
        "categories": list(candidate.categories),
        "abstract": candidate.abstract,
        "published_at": candidate.published_at.isoformat(),
        "available_at": candidate.available_at.isoformat(),
        "acquired_at": acquired_at.astimezone(UTC).isoformat(),
        "availability_rule": candidate.availability_rule,
        "source": {
            "kind": candidate.source_kind,
            "source_id": candidate.source_id,
            "original_url": candidate.pdf_url,
            "requested_url": pdf.requested_url,
            "final_url": pdf.final_url,
            "redirects": list(pdf.redirect_chain),
        },
        "selection": {
            "policy": candidate.selection_policy,
            "rank": candidate.selection_rank,
            "score": candidate.selection_score,
            "daily_limit": candidate.selection_daily_limit,
            "as_of": (
                candidate.selection_as_of.isoformat()
                if candidate.selection_as_of is not None
                else None
            ),
            "novelty_at": candidate.available_at.isoformat(),
        },
        "content": {
            "path": RESEARCH_ASSET_CONTENT_NAME,
            "media_type": pdf.media_type,
            "bytes": pdf.size_bytes,
            "sha256": pdf.sha256,
        },
        "files": [
            {
                "path": RESEARCH_ASSET_CONTENT_NAME,
                "media_type": pdf.media_type,
                "bytes": pdf.size_bytes,
                "sha256": pdf.sha256,
            }
        ],
        "metadata": _json_mapping(candidate.source_metadata),
    }


def _automatic_selection_priority(candidate: ResearchAssetCandidate) -> int:
    availability_day = candidate.available_at.astimezone(_SHANGHAI).date()
    days_since_epoch = max(0, (availability_day - date(2000, 1, 1)).days)
    within_day = max(0, 31 - min(candidate.selection_rank, 31))
    # 32 points per day leaves the lower five bits for deterministic rank.
    # The 1,000,000 ceiling is the existing ResearchAssetStore contract.
    return min(1_000_000, days_since_epoch * 32 + within_day)


def _load_matching_asset(
    target: Path,
    expected: Mapping[str, object],
) -> PublishedResearchAsset:
    if _path_is_linkish(target) or not target.is_dir():
        raise ResearchAssetConflictError(f"research asset path is not a real directory: {target}")
    manifest_path = target / RESEARCH_ASSET_MANIFEST_NAME
    existing = load_research_asset_manifest(manifest_path)
    expected_content = expected["content"]
    existing_content = existing.get("content")
    expected_source = expected["source"]
    existing_source = existing.get("source")
    if not isinstance(existing_content, dict) or not isinstance(expected_content, dict):
        raise ResearchAssetConflictError("research asset content manifest is malformed")
    if not isinstance(existing_source, dict) or not isinstance(expected_source, dict):
        raise ResearchAssetConflictError("research asset source manifest is malformed")
    if (
        existing_content.get("sha256") != expected_content.get("sha256")
        or existing_source.get("kind") != expected_source.get("kind")
        or existing_source.get("source_id") != expected_source.get("source_id")
        or any(
            existing.get(field_name) != expected.get(field_name)
            for field_name in (
                "kind",
                "type",
                "status",
                "title",
                "published_at",
                "available_at",
                "availability_rule",
                "selection",
                "metadata",
            )
        )
    ):
        raise ResearchAssetConflictError(
            f"immutable research asset conflicts with existing content: {target.name}"
        )
    return PublishedResearchAsset(
        asset_id=target.name,
        directory=target,
        manifest_path=manifest_path,
        content_path=target / RESEARCH_ASSET_CONTENT_NAME,
        manifest=existing,
    )


def _path_is_linkish(path: Path) -> bool:
    """Reject symlinks and Windows reparse points such as junctions."""

    try:
        metadata = path.lstat()
    except OSError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = int(getattr(metadata, "st_file_attributes", 0))
    return stat.S_ISLNK(metadata.st_mode) or bool(file_attributes & reparse_flag)


def _safe_asset_relative_path(value: Path | str) -> Path:
    raw = value.as_posix() if isinstance(value, Path) else str(value)
    if not raw or "\\" in raw or any(ord(character) < 32 for character in raw):
        raise ValueError("research asset relative path is invalid")
    path = Path(raw)
    if path.is_absolute() or not path.parts or len(path.as_posix().encode("utf-8")) > 4_096:
        raise ValueError("research asset relative path is invalid")
    for part in path.parts:
        if (
            part in {"", ".", ".."}
            or len(part.encode("utf-8")) > 255
            or part.rstrip(" .") != part
            or any(character in '<>:"|?*' for character in part)
            or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES
        ):
            raise ValueError(f"unsafe research asset path component: {part!r}")
    if path.parts[0].casefold() in _RESERVED_ASSET_FILENAMES:
        raise ValueError("research asset content collides with its sealed manifest")
    return path


def validate_research_asset_relative_path(value: Path | str) -> Path:
    """Expose the immutable asset path contract to governed acquisition adapters."""

    return _safe_asset_relative_path(value)


def _local_source_files(source_path: Path, *, kind: str) -> list[tuple[Path, Path]]:
    """Enumerate administrator-owned source files without following links."""

    source = Path(source_path)
    try:
        source_metadata = source.lstat()
    except OSError as exc:
        raise ValueError(f"local research asset source is unavailable: {source}") from exc
    if _path_is_linkish(source):
        raise ValueError("local research asset source must not be a link or junction")
    if stat.S_ISREG(source_metadata.st_mode):
        if kind == "pdf":
            if source.suffix.casefold() != ".pdf":
                raise ValueError("a local PDF asset source must have a .pdf suffix")
            return [(source.resolve(strict=True), Path(RESEARCH_ASSET_CONTENT_NAME))]
        relative = _safe_asset_relative_path(Path(source.name))
        return [(source.resolve(strict=True), relative)]
    if not stat.S_ISDIR(source_metadata.st_mode):
        raise ValueError("local research asset source must be a regular file or directory")
    if kind == "pdf":
        raise ValueError("a local PDF asset source must be one regular PDF file")

    root = source.resolve(strict=True)
    files: list[tuple[Path, Path]] = []
    for current_value, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_value)
        directory_names.sort()
        file_names.sort()
        for directory_name in directory_names:
            if _path_is_linkish(current / directory_name):
                raise ValueError("local research asset directories must not contain links")
        for file_name in file_names:
            file_path = current / file_name
            if _path_is_linkish(file_path):
                raise ValueError("local research asset directories must not contain links")
            try:
                file_metadata = file_path.lstat()
            except OSError as exc:
                raise ValueError(f"local research asset file disappeared: {file_path}") from exc
            if not stat.S_ISREG(file_metadata.st_mode):
                raise ValueError("local research assets may contain only regular files")
            relative = _safe_asset_relative_path(file_path.relative_to(root))
            files.append((file_path.resolve(strict=True), relative))
            if len(files) > DEFAULT_LOCAL_ASSET_MAX_FILES:
                raise ValueError("local research asset has too many files")
    if not files:
        raise ValueError("local research asset directory has no files")
    return files


def _copy_local_file(
    source: Path,
    destination: Path,
    *,
    relative_path: Path,
    require_pdf: bool,
    max_bytes: int,
) -> dict[str, object]:
    """Copy and hash one stable regular file into a private staging directory."""

    if _path_is_linkish(source):
        raise ValueError("local research asset files must not be links")
    before = source.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("local research asset files must be regular files")
    maximum = min(
        DEFAULT_PDF_MAX_BYTES if require_pdf else DEFAULT_LOCAL_ASSET_MAX_BYTES,
        max_bytes,
    )
    if maximum < 0:
        raise ValueError("local research asset exceeds the governed total size limit")
    if before.st_size > maximum:
        raise ValueError("local research asset file exceeds the governed size limit")

    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(source, flags)
    digest = hashlib.sha256()
    total = 0
    prefix = b""
    tail = b""
    try:
        with os.fdopen(descriptor, "rb") as input_handle, destination.open("xb") as output_handle:
            opened = os.fstat(input_handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError("local research asset files must be regular files")
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise ResearchAssetConflictError("local source changed before it was copied")
            for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
                total += len(chunk)
                if total > maximum:
                    raise ValueError("local research asset file exceeds the governed size limit")
                digest.update(chunk)
                prefix = (prefix + chunk)[:5]
                tail = (tail + chunk)[-_PDF_EOF_SCAN_BYTES:]
                output_handle.write(chunk)
            final = os.fstat(input_handle.fileno())
            output_handle.flush()
            os.fsync(output_handle.fileno())
    except Exception:
        if destination.exists():
            destination.unlink()
        raise
    after = source.stat(follow_symlinks=False)
    if (
        (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
        or opened.st_size != total
        or final.st_size != total
        or opened.st_mtime_ns != final.st_mtime_ns
        or after.st_mtime_ns != final.st_mtime_ns
    ):
        destination.unlink(missing_ok=True)
        raise ResearchAssetConflictError("local source changed while it was copied")
    if require_pdf and (prefix != b"%PDF-" or b"%%EOF" not in tail):
        destination.unlink(missing_ok=True)
        raise ResearchAssetDownloadError("local PDF failed magic or EOF validation")

    relative = _safe_asset_relative_path(relative_path).as_posix()
    media_type = "application/pdf" if require_pdf else (
        mimetypes.guess_type(relative)[0] or "application/octet-stream"
    )
    return {
        "path": relative,
        "media_type": media_type.casefold(),
        "bytes": total,
        "sha256": digest.hexdigest(),
    }


def _verify_materialized_file(
    asset_root: Path,
    entry: Mapping[str, object],
    *,
    require_pdf: bool,
) -> int:
    raw_relative = str(entry.get("path") or "")
    relative = _safe_asset_relative_path(raw_relative)
    if raw_relative != relative.as_posix():
        raise ResearchAssetConflictError("research asset file path is not canonical")
    expected_size = entry.get("bytes")
    expected_digest = str(entry.get("sha256") or "")
    media_type = str(entry.get("media_type") or "").casefold()
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
        or not _SHA256_PATTERN.fullmatch(expected_digest)
        or not media_type
    ):
        raise ResearchAssetConflictError("research asset file contract is invalid")
    maximum = DEFAULT_PDF_MAX_BYTES if require_pdf else DEFAULT_LOCAL_ASSET_MAX_BYTES
    if expected_size > maximum:
        raise ResearchAssetConflictError("research asset file exceeds its governed size limit")

    root = asset_root.resolve(strict=True)
    file_path = asset_root / relative
    try:
        resolved = file_path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ResearchAssetConflictError("research asset file escapes its immutable root") from exc
    current = asset_root
    for part in relative.parts:
        current /= part
        if _path_is_linkish(current):
            raise ResearchAssetConflictError("research asset file path contains a link")
    metadata = resolved.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_size:
        raise ResearchAssetConflictError("research asset file size or type is invalid")

    digest = hashlib.sha256()
    prefix = b""
    tail = b""
    total = 0
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            total += len(chunk)
            if total > maximum:
                raise ResearchAssetConflictError(
                    "research asset file exceeds its governed size limit"
                )
            digest.update(chunk)
            prefix = (prefix + chunk)[:5]
            tail = (tail + chunk)[-_PDF_EOF_SCAN_BYTES:]
    if total != expected_size or digest.hexdigest() != expected_digest:
        raise ResearchAssetConflictError("research asset file digest disagrees")
    if require_pdf and (
        relative.suffix.casefold() != ".pdf"
        or media_type != "application/pdf"
        or prefix != b"%PDF-"
        or b"%%EOF" not in tail
    ):
        raise ResearchAssetConflictError("research asset PDF contract is invalid")
    return total


def _finetune_reference(metadata: Mapping[str, object], field_name: str) -> str:
    value = str(metadata.get(field_name) or "").strip()
    parts = value.replace("\\", "/").split("/")
    if (
        not _SAFE_REFERENCE_PATTERN.fullmatch(value)
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"finetune metadata {field_name} is missing or invalid")
    return value


def _finetune_contract_file(
    file_reader: Callable[[str], bytes], relative_path: str
) -> bytes:
    try:
        body = file_reader(relative_path)
    except (KeyError, OSError) as exc:
        raise ValueError(
            f"finetune sealed evidence {relative_path} is unavailable"
        ) from exc
    if not isinstance(body, bytes) or not 0 < len(body) <= _FINETUNE_CONTRACT_FILE_MAX_BYTES:
        raise ValueError(f"finetune sealed evidence {relative_path} is invalid")
    return body


def validate_finetune_dataset_info(
    dataset: str,
    body: bytes,
    *,
    file_paths: Iterable[str],
) -> dict[str, object]:
    """Validate the pinned upstream dataset description and its file references."""

    if not 0 < len(body) <= _FINETUNE_CONTRACT_FILE_MAX_BYTES:
        raise ValueError("finetune datasets/dataset_info.json is empty or oversized")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("finetune datasets/dataset_info.json is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("finetune datasets/dataset_info.json must be an object")
    selected = payload.get(dataset)
    if not isinstance(selected, dict) or not selected:
        raise ValueError(
            "finetune datasets/dataset_info.json does not describe the selected dataset"
        )
    total_samples = selected.get("total_samples")
    if (
        isinstance(total_samples, bool)
        or not isinstance(total_samples, int)
        or total_samples <= 0
    ):
        raise ValueError("finetune selected dataset must declare positive total_samples")
    tasks = selected.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("finetune selected dataset must declare non-empty tasks")

    normalized_paths = {
        _safe_asset_relative_path(value).as_posix() for value in file_paths
    }
    dataset_root = _safe_asset_relative_path(Path("datasets") / Path(dataset))
    referenced_files = 0
    for task_name, task in tasks.items():
        if (
            not isinstance(task_name, str)
            or not task_name.strip()
            or not isinstance(task, dict)
        ):
            raise ValueError("finetune selected dataset contains an invalid task")
        task_files = task.get("files")
        if not isinstance(task_files, list) or not task_files:
            raise ValueError("finetune selected dataset task has no files")
        for raw_path in task_files:
            if not isinstance(raw_path, str):
                raise ValueError("finetune selected dataset task file is invalid")
            task_relative = _safe_asset_relative_path(raw_path)
            sealed_path = (dataset_root / task_relative).as_posix()
            if sealed_path not in normalized_paths:
                raise ValueError(
                    "finetune dataset_info.json references an unsealed dataset file"
                )
            referenced_files += 1
    if referenced_files == 0:
        raise ValueError("finetune selected dataset has no sealed task files")
    return payload


def validate_finetune_asset_contract(
    metadata: Mapping[str, object],
    *,
    file_paths: Iterable[str] | None = None,
    file_reader: Callable[[str], bytes] | None = None,
) -> dict[str, object]:
    """Return one normalized FT contract shared by registration and execution."""

    benchmark = _finetune_reference(metadata, "benchmark")
    if benchmark not in FINETUNE_BENCHMARK_ALLOWLIST:
        raise ValueError("finetune metadata benchmark is not supported by pinned RD-Agent")
    normalized: dict[str, object] = {
        "benchmark": benchmark,
        "dataset": _finetune_reference(metadata, "dataset"),
        "base_model": _finetune_reference(metadata, "base_model"),
    }
    description = str(metadata.get("benchmark_description") or "").strip()
    if not 10 <= len(description) <= 4_000 or any(
        ord(character) < 32 and character not in "\n\t" for character in description
    ):
        raise ValueError("finetune metadata benchmark_description is missing or invalid")
    normalized["benchmark_description"] = description

    accepted_timestamps: list[datetime] = []
    for subject in _FINETUNE_GOVERNANCE_SUBJECTS:
        revision_field = f"{subject}_revision"
        revision = str(metadata.get(revision_field) or "").strip()
        if not _IMMUTABLE_REVISION_PATTERN.fullmatch(revision):
            raise ValueError(
                f"finetune metadata {revision_field} must be an immutable 40- or "
                "64-character lowercase revision digest"
            )
        normalized[revision_field] = revision

        accepted_field = f"{subject}_license_accepted"
        if metadata.get(accepted_field) is not True:
            raise ValueError(f"finetune metadata {accepted_field} must be true")
        normalized[accepted_field] = True

        for suffix in ("license", "license_accepted_by"):
            field_name = f"{subject}_{suffix}"
            value = str(metadata.get(field_name) or "").strip()
            if not 1 <= len(value) <= 256 or any(ord(character) < 32 for character in value):
                raise ValueError(f"finetune metadata {field_name} is missing or invalid")
            normalized[field_name] = value

        terms_field = f"{subject}_license_terms_sha256"
        terms_sha256 = str(metadata.get(terms_field) or "").strip()
        if not _SHA256_PATTERN.fullmatch(terms_sha256):
            raise ValueError(f"finetune metadata {terms_field} is missing or invalid")
        normalized[terms_field] = terms_sha256

        accepted_at_field = f"{subject}_license_accepted_at"
        accepted_at_raw = str(metadata.get(accepted_at_field) or "").strip()
        try:
            accepted_at = datetime.fromisoformat(
                accepted_at_raw.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                f"finetune metadata {accepted_at_field} is missing or invalid"
            ) from exc
        if accepted_at.tzinfo is None or accepted_at.utcoffset() is None:
            raise ValueError(
                f"finetune metadata {accepted_at_field} must be timezone-aware"
            )
        normalized[accepted_at_field] = accepted_at.isoformat()
        accepted_timestamps.append(accepted_at)

    normalized["latest_license_accepted_at"] = max(accepted_timestamps).isoformat()
    if file_paths is None:
        if file_reader is not None:
            raise ValueError("finetune file reader requires a sealed file inventory")
        return normalized
    if file_reader is None:
        raise ValueError("finetune sealed file inventory requires a file reader")

    normalized_paths = {
        _safe_asset_relative_path(value).as_posix() for value in file_paths
    }
    required_prefixes = {
        "base_model": _safe_asset_relative_path(
            Path("models") / Path(str(normalized["base_model"]))
        ).as_posix()
        + "/",
        "dataset": _safe_asset_relative_path(
            Path("datasets") / Path(str(normalized["dataset"]))
        ).as_posix()
        + "/",
        "benchmark": "benchmarks/opencompass_data/",
    }
    for label, prefix in required_prefixes.items():
        if not any(path.startswith(prefix) for path in normalized_paths):
            raise ValueError(
                f"finetune files must include at least one file below {prefix} "
                f"for metadata {label}"
            )
    required_files = {
        ".llama_factory_info/constants.json",
        ".llama_factory_info/parameters.json",
        "datasets/dataset_info.json",
        *(
            evidence_path
            for evidence in _FINETUNE_GOVERNANCE_EVIDENCE.values()
            for evidence_path in evidence.values()
        ),
    }
    missing = sorted(required_files - normalized_paths)
    if missing:
        raise ValueError(
            "finetune files are missing required sealed contract evidence: "
            + ", ".join(missing)
        )

    for subject, evidence in _FINETUNE_GOVERNANCE_EVIDENCE.items():
        revision_path = evidence["revision"]
        revision_body = _finetune_contract_file(file_reader, revision_path)
        try:
            sealed_revision = revision_body.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"finetune sealed evidence {revision_path} is not ASCII"
            ) from exc
        if sealed_revision != normalized[f"{subject}_revision"]:
            raise ValueError(
                f"finetune {subject} revision disagrees with sealed evidence"
            )
        terms_path = evidence["license_terms"]
        terms_body = _finetune_contract_file(file_reader, terms_path)
        if hashlib.sha256(terms_body).hexdigest() != normalized[
            f"{subject}_license_terms_sha256"
        ]:
            raise ValueError(
                f"finetune {subject} license terms disagree with sealed evidence"
            )

    dataset_info_path = "datasets/dataset_info.json"
    validate_finetune_dataset_info(
        str(normalized["dataset"]),
        _finetune_contract_file(file_reader, dataset_info_path),
        file_paths=normalized_paths,
    )
    if benchmark == "FinanceIQ_gen":
        for partition in ("dev", "test"):
            prefix = f"{FINETUNE_FINANCEIQ_DATA_ROOT}/{partition}/"
            if not any(path.startswith(prefix) for path in normalized_paths):
                raise ValueError(
                    "FinanceIQ_gen requires pre-sealed dev and test benchmark data"
                )
    return normalized


def _validate_finetune_metadata(
    metadata: Mapping[str, object],
    *,
    file_paths: Iterable[str] | None = None,
    file_reader: Callable[[str], bytes] | None = None,
) -> datetime:
    contract = validate_finetune_asset_contract(
        metadata,
        file_paths=file_paths,
        file_reader=file_reader,
    )
    return datetime.fromisoformat(str(contract["latest_license_accepted_at"]))


def _write_sealed_manifest(stage: Path, manifest: Mapping[str, object]) -> None:
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if len(manifest_bytes) > DEFAULT_MANIFEST_MAX_BYTES:
        raise ValueError("research asset manifest exceeds its governed size limit")
    _write_new_file(stage / RESEARCH_ASSET_MANIFEST_NAME, manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest().encode("ascii") + b"\n"
    _write_new_file(stage / RESEARCH_ASSET_MANIFEST_SHA256_NAME, digest)


def _same_local_contract(
    existing: Mapping[str, object], expected: Mapping[str, object]
) -> bool:
    return all(
        existing.get(field_name) == expected.get(field_name)
        for field_name in (
            "asset_id",
            "kind",
            "type",
            "status",
            "priority",
            "availability_rule",
            "source",
            "files",
            "metadata",
        )
    )


def _first_manifest_file_path(manifest: Mapping[str, object]) -> Path:
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
        raise ResearchAssetConflictError("research asset manifest has no first file")
    return _safe_asset_relative_path(str(entries[0].get("path") or ""))


def _write_new_file(path: Path, body: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def _require_root_inside_data_root(data_root: Path, root: Path) -> None:
    try:
        root.resolve(strict=True).relative_to(Path(data_root).resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ResearchAssetConflictError(
            "research asset root escapes the configured DATA_ROOT"
        ) from exc


def _validate_verified_pdf(pdf: VerifiedPdf) -> None:
    if pdf.media_type != "application/pdf":
        raise ResearchAssetDownloadError("verified PDF has a non-PDF media type")
    if pdf.size_bytes > DEFAULT_PDF_MAX_BYTES:
        raise ResearchAssetDownloadError("verified PDF exceeds the governed size limit")
    if hashlib.sha256(pdf.body).hexdigest() != pdf.sha256:
        raise ResearchAssetDownloadError("verified PDF SHA-256 is inconsistent")
    if not pdf.body.startswith(b"%PDF-"):
        raise ResearchAssetDownloadError("verified PDF body has invalid magic")
    if b"%%EOF" not in pdf.body[-_PDF_EOF_SCAN_BYTES:]:
        raise ResearchAssetDownloadError("verified PDF body has no terminal EOF marker")


def _declared_content_length(response: Any) -> int | None:
    raw_value = _header(response, "Content-Length").strip()
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ResearchAssetDownloadError("invalid research asset Content-Length") from exc
    if value < 0:
        raise ResearchAssetDownloadError("negative research asset Content-Length")
    return value


def _header(response: Any, name: str) -> str:
    headers = getattr(response, "headers", {})
    for key, value in headers.items():
        if str(key).casefold() == name.casefold():
            return str(value)
    return ""


def _reject_non_public_peer(response: Any) -> None:
    """Fail closed unless the live connection exposes a public peer."""

    raw = getattr(response, "raw", None)
    connection = getattr(raw, "_connection", None) or getattr(raw, "connection", None)
    sock = getattr(connection, "sock", None)
    if sock is None:
        # urllib3 versions differ in where they expose the underlying TLS
        # socket. Keep the fallback narrow and fail closed for unknown
        # transports instead of silently losing DNS-rebinding protection.
        original_response = getattr(raw, "_fp", None) or getattr(
            raw, "_original_response", None
        )
        fp = getattr(original_response, "fp", None)
        buffered = getattr(fp, "raw", None)
        sock = getattr(buffered, "_sock", None)
    if sock is None:
        raise UnsafeResearchAssetUrl("connected peer socket cannot be validated")
    try:
        peer = str(sock.getpeername()[0])
        address = ipaddress.ip_address(peer)
    except (OSError, ValueError, TypeError, IndexError) as exc:
        raise UnsafeResearchAssetUrl("connected peer address cannot be validated") from exc
    if not address.is_global:
        raise UnsafeResearchAssetUrl(f"connected peer is not public: {peer}")


def _parse_source_date(value: object) -> date:
    text = str(value or "").strip()
    for format_string in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, format_string).date()
        except ValueError:
            continue
    raise ValueError(f"invalid source date: {text or '<blank>'}")


def _parse_source_datetime(value: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ResearchAssetDiscoveryError("arXiv entry timestamp is blank")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchAssetDiscoveryError(f"invalid arXiv timestamp: {text}") from exc
    _require_aware(parsed, "arXiv timestamp")
    return parsed


def _manifest_datetime(value: object, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchAssetConflictError(f"manifest {name} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ResearchAssetConflictError(f"manifest {name} must be timezone-aware")
    return parsed


def _next_open_day(value: date, open_days: Sequence[date]) -> date:
    index = bisect_right(open_days, value)
    if index >= len(open_days):
        raise LookupError(
            f"no trading day after {value}; extend trade_cal before selecting research reports"
        )
    return open_days[index]


def _tushare_report_score(row: Mapping[str, object]) -> int:
    score = 0
    if str(row.get("ts_code") or "").strip():
        score += 4
    if str(row.get("abstr") or "").strip():
        score += 3
    if str(row.get("author") or "").strip():
        score += 1
    if str(row.get("inst_csname") or "").strip():
        score += 1
    if str(row.get("ind_name") or "").strip():
        score += 1
    text = " ".join(
        str(row.get(key) or "") for key in ("title", "file_name", "report_type")
    )
    score += sum(
        2
        for term in ("深度", "首次覆盖", "投资策略", "行业", "专题", "年度")
        if term in text
    )
    return score


def _arxiv_score(candidate: ResearchAssetCandidate, as_of: date) -> int:
    categories = set(candidate.categories)
    score = 0
    if any(category.startswith("q-fin.") for category in categories):
        score += 100
    if "cs.LG" in categories:
        score += 25
    if "stat.ML" in categories:
        score += 20
    text = f"{candidate.title} {candidate.abstract}".casefold()
    score += sum(
        5
        for term in (
            "alpha",
            "asset pricing",
            "factor",
            "portfolio",
            "trading",
            "financial market",
            "time series",
            "risk",
        )
        if term in text
    )
    local_published_date = candidate.published_at.astimezone(_SHANGHAI).date()
    age_days = max(0, (as_of - local_published_date).days)
    score += max(0, 7 - age_days)
    return score


def _arxiv_id_from_entry(entry: ElementTree.Element, entry_id: str, atom: str) -> str:
    candidates: list[str] = []
    for link in entry.findall(f"{atom}link"):
        if str(link.attrib.get("type") or "").lower() != "application/pdf":
            continue
        href = str(link.attrib.get("href") or "").strip()
        path = urlsplit(href).path
        if "/pdf/" in path:
            candidates.append(path.split("/pdf/", 1)[1].removesuffix(".pdf"))
    path = urlsplit(entry_id).path
    if "/abs/" in path:
        candidates.append(path.split("/abs/", 1)[1])
    for value in candidates:
        normalized = value.strip("/")
        if _ARXIV_ID_PATTERN.fullmatch(normalized):
            return normalized
    raise ResearchAssetDiscoveryError(f"arXiv entry has no valid identifier: {entry_id}")


def _xml_text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _json_mapping(values: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in sorted(values.items(), key=lambda item: str(item[0])):
        normalized_key = str(key)
        if normalized_key in normalized:
            raise ValueError("research asset metadata has duplicate normalized keys")
        normalized[normalized_key] = _json_value(value)
    return normalized


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return str(value)


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
