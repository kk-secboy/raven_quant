from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .research_assets import (
    DEFAULT_LOCAL_ASSET_MAX_BYTES,
    DEFAULT_LOCAL_ASSET_MAX_FILES,
    PublishedResearchAsset,
    ResearchAssetConflictError,
    ResearchAssetError,
    derive_asset_id,
    register_local_research_asset,
    validate_research_asset_relative_path,
)

KAGGLE_ALLOWLIST_SCHEMA_VERSION = "kaggle-source-allowlist-v1"
KAGGLE_ASSET_CONTRACT_VERSION = "kaggle-research-asset-v1"
KAGGLE_CAPABILITY = "kaggle_research_asset_acquisition"
KAGGLE_GOVERNANCE_ROOT = Path(".quantlab") / "kaggle"
KAGGLE_INVENTORY_PATH = KAGGLE_GOVERNANCE_ROOT / "inventory.json"
KAGGLE_SOURCE_PATH = KAGGLE_GOVERNANCE_ROOT / "source.json"
KAGGLE_ALLOWLIST_EVIDENCE_PATH = KAGGLE_GOVERNANCE_ROOT / "allowlist-approval.json"
KAGGLE_LICENSE_TERMS_PATH = KAGGLE_GOVERNANCE_ROOT / "license-terms.txt"
DEFAULT_KAGGLE_ARCHIVE_MAX_BYTES = 5 * 1024 * 1024 * 1024
DEFAULT_KAGGLE_TIMEOUT_SECONDS = 6 * 60 * 60
_POLICY_MAX_BYTES = 1024 * 1024
_LICENSE_TERMS_MAX_BYTES = 4 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DATASET_SLUG = re.compile(
    r"[a-z0-9][a-z0-9_-]{0,63}/[a-z0-9][a-z0-9_-]{0,99}\Z"
)
_COMPETITION_SLUG = re.compile(r"[a-z0-9][a-z0-9-]{0,99}\Z")
_DATASET_VERSION = re.compile(r"[1-9][0-9]{0,9}\Z")
_SAFE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SAFE_ENV_NAMES = (
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "PATH",
    "PATHEXT",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "WINDIR",
)


class KaggleAssetError(ResearchAssetError):
    """Base error for the governed Kaggle acquisition boundary."""


class KaggleCapabilityBlocked(KaggleAssetError):
    """A required local capability, permission, or approval is unavailable."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class UnsafeKaggleArchive(KaggleAssetError):
    """The downloaded archive violates the immutable dataset safety contract."""


@dataclass(frozen=True, slots=True)
class KaggleAllowlist:
    declared_by: str
    datasets: frozenset[str]
    competitions: frozenset[str]
    sha256: str

    def permits(self, source_kind: str, slug: str) -> bool:
        allowed = self.datasets if source_kind == "dataset" else self.competitions
        return slug in allowed


@dataclass(frozen=True, slots=True)
class KaggleAcquisition:
    published: PublishedResearchAsset
    source_kind: str
    slug: str
    source_version: str
    archive_sha256: str
    archive_bytes: int
    inventory_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "status": "succeeded",
            "capability": KAGGLE_CAPABILITY,
            "asset_id": self.published.asset_id,
            "kind": "dataset",
            "type": self.published.manifest.get("type"),
            "source_kind": self.source_kind,
            "slug": self.slug,
            "source_version": self.source_version,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "inventory_sha256": self.inventory_sha256,
            "manifest_path": str(self.published.manifest_path),
        }


def validate_kaggle_research_asset_manifest(
    manifest: Mapping[str, object], asset_root: Path
) -> dict[str, object]:
    """Validate the sealed Kaggle provenance, inventory, and license evidence."""

    asset_type = str(manifest.get("type") or "")
    if asset_type not in {"kaggle_dataset", "kaggle_competition"}:
        raise ValueError("research asset is not a Kaggle dataset contract")
    source_kind = asset_type.removeprefix("kaggle_")
    metadata = manifest.get("metadata")
    source = manifest.get("source")
    files = manifest.get("files")
    if not isinstance(metadata, dict) or not isinstance(source, dict):
        raise ValueError("Kaggle research asset provenance is missing")
    if not isinstance(files, list) or not files:
        raise ValueError("Kaggle research asset file inventory is missing")
    if metadata.get("schema_version") != KAGGLE_ASSET_CONTRACT_VERSION:
        raise ValueError("Kaggle research asset schema version is unsupported")
    if metadata.get("source_kind") != source_kind:
        raise ValueError("Kaggle source kind disagrees with the asset type")
    slug = _slug(str(metadata.get("slug") or ""), source_kind)
    source_version = _source_version(
        str(metadata.get("source_version") or ""), source_kind
    )
    archive_sha256 = str(metadata.get("archive_sha256") or "")
    inventory_sha256 = str(metadata.get("inventory_sha256") or "")
    terms_sha256 = str(metadata.get("license_terms_sha256") or "")
    policy_sha256 = str(metadata.get("allowlist_policy_sha256") or "")
    for label, value in (
        ("archive", archive_sha256),
        ("inventory", inventory_sha256),
        ("license terms", terms_sha256),
        ("allowlist policy", policy_sha256),
    ):
        if not _SHA256.fullmatch(value):
            raise ValueError(f"Kaggle {label} SHA-256 is invalid")
    if (
        metadata.get("license_accepted") is not True
        or metadata.get("credentials_persisted") is not False
    ):
        raise ValueError("Kaggle license or credential-persistence contract is invalid")
    license_name = _plain_text(metadata.get("license"), "license", 256)
    accepted_by = _plain_text(
        metadata.get("license_accepted_by"), "license_accepted_by", 256
    )
    declared_by = _plain_text(
        metadata.get("allowlist_declared_by"), "allowlist_declared_by", 256
    )
    accepted_at_raw = str(metadata.get("license_accepted_at") or "")
    try:
        accepted_at = datetime.fromisoformat(accepted_at_raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Kaggle license acceptance timestamp is invalid") from exc
    if accepted_at.tzinfo is None or accepted_at.utcoffset() is None:
        raise ValueError("Kaggle license acceptance timestamp must include a timezone")
    archive_bytes = metadata.get("archive_bytes")
    if (
        isinstance(archive_bytes, bool)
        or not isinstance(archive_bytes, int)
        or archive_bytes <= 0
        or archive_bytes > DEFAULT_LOCAL_ASSET_MAX_BYTES
    ):
        raise ValueError("Kaggle archive byte count is invalid")
    expected_paths = {
        "inventory_path": KAGGLE_INVENTORY_PATH,
        "source_evidence_path": KAGGLE_SOURCE_PATH,
        "allowlist_evidence_path": KAGGLE_ALLOWLIST_EVIDENCE_PATH,
        "license_terms_path": KAGGLE_LICENSE_TERMS_PATH,
    }
    for field_name, expected_path in expected_paths.items():
        if metadata.get(field_name) != expected_path.as_posix():
            raise ValueError(f"Kaggle {field_name} is not fixed to sealed evidence")
    file_entries: dict[str, Mapping[str, object]] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Kaggle manifest contains an invalid file entry")
        relative = validate_research_asset_relative_path(str(entry.get("path") or ""))
        key = relative.as_posix()
        if key in file_entries:
            raise ValueError("Kaggle manifest contains duplicate file entries")
        file_entries[key] = entry
    missing = sorted(
        path.as_posix() for path in expected_paths.values() if path.as_posix() not in file_entries
    )
    if missing:
        raise ValueError("Kaggle manifest is missing sealed evidence: " + ", ".join(missing))

    inventory_body = _sealed_evidence_body(
        asset_root, KAGGLE_INVENTORY_PATH, file_entries
    )
    terms_body = _sealed_evidence_body(
        asset_root, KAGGLE_LICENSE_TERMS_PATH, file_entries
    )
    source_body = _sealed_evidence_body(asset_root, KAGGLE_SOURCE_PATH, file_entries)
    approval_body = _sealed_evidence_body(
        asset_root, KAGGLE_ALLOWLIST_EVIDENCE_PATH, file_entries
    )
    if hashlib.sha256(inventory_body).hexdigest() != inventory_sha256:
        raise ValueError("Kaggle inventory evidence SHA-256 disagrees with metadata")
    if hashlib.sha256(terms_body).hexdigest() != terms_sha256:
        raise ValueError("Kaggle license terms SHA-256 disagrees with metadata")
    source_evidence = _evidence_json(source_body, "source")
    expected_source_evidence = {
        "schema_version": KAGGLE_ASSET_CONTRACT_VERSION,
        "source_kind": source_kind,
        "slug": slug,
        "source_version": source_version,
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "license": license_name,
        "license_terms_sha256": terms_sha256,
        "license_accepted": True,
        "license_accepted_by": accepted_by,
        "license_accepted_at": accepted_at.isoformat(),
        "allowlist_policy_sha256": policy_sha256,
        "allowlist_declared_by": declared_by,
        "inventory_sha256": inventory_sha256,
    }
    if source_evidence != expected_source_evidence:
        raise ValueError("Kaggle sealed source evidence disagrees with metadata")
    approval_evidence = _evidence_json(approval_body, "allowlist")
    if approval_evidence != {
        "schema_version": KAGGLE_ALLOWLIST_SCHEMA_VERSION,
        "administrator_role": "data_science",
        "declared_by": declared_by,
        "policy_sha256": policy_sha256,
        "source_kind": source_kind,
        "slug": slug,
    }:
        raise ValueError("Kaggle sealed allowlist approval disagrees with metadata")
    if source != {
        "kind": f"kaggle_{source_kind}",
        "source_id": slug,
        "version": source_version,
        "sha256": archive_sha256,
    }:
        raise ValueError("Kaggle top-level source provenance is invalid")

    inventory = _evidence_json(inventory_body, "inventory")
    inventory_files = inventory.get("files")
    if (
        inventory.get("schema_version") != "kaggle-inventory-v1"
        or not isinstance(inventory_files, list)
        or inventory.get("file_count") != len(inventory_files)
    ):
        raise ValueError("Kaggle sealed inventory has an invalid contract")
    governed_prefix = KAGGLE_GOVERNANCE_ROOT.as_posix() + "/"
    payload_entries = {
        path: entry
        for path, entry in file_entries.items()
        if not path.startswith(governed_prefix)
    }
    expected_inventory: dict[str, Mapping[str, object]] = {}
    total_bytes = 0
    for entry in inventory_files:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("Kaggle sealed inventory contains an invalid entry")
        relative = validate_research_asset_relative_path(str(entry.get("path") or ""))
        key = relative.as_posix()
        if key.startswith(governed_prefix) or key in expected_inventory:
            raise ValueError("Kaggle sealed inventory contains an unsafe path")
        byte_count = entry.get("bytes")
        digest = str(entry.get("sha256") or "")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or not _SHA256.fullmatch(digest)
        ):
            raise ValueError("Kaggle sealed inventory contains invalid file evidence")
        total_bytes += byte_count
        expected_inventory[key] = entry
    if (
        expected_inventory.keys() != payload_entries.keys()
        or inventory.get("total_bytes") != total_bytes
    ):
        raise ValueError("Kaggle sealed inventory does not cover the payload exactly")
    for path, expected in expected_inventory.items():
        actual = payload_entries[path]
        if actual.get("bytes") != expected["bytes"] or actual.get("sha256") != expected[
            "sha256"
        ]:
            raise ValueError("Kaggle sealed inventory disagrees with manifest file hashes")
    return expected_source_evidence


def kaggle_blocked_result(
    *, source_kind: str, slug: str, reason_code: str, message: str
) -> dict[str, object]:
    """Return the stable non-secret result contract consumed by future API jobs."""

    return {
        "status": "blocked",
        "capability": KAGGLE_CAPABILITY,
        "source_kind": source_kind,
        "slug": slug,
        "reason_code": reason_code,
        "message": message,
    }


def load_kaggle_allowlist(path: Path) -> KaggleAllowlist:
    """Load a local data_science administrator declaration without following links."""

    body = _read_regular_file(path, maximum=_POLICY_MAX_BYTES, label="Kaggle allowlist")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KaggleCapabilityBlocked(
            "allowlist_invalid", "Kaggle allowlist is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "administrator_role",
        "declared_by",
        "datasets",
        "competitions",
    }:
        raise KaggleCapabilityBlocked(
            "allowlist_invalid", "Kaggle allowlist has an unsupported contract"
        )
    if payload.get("schema_version") != KAGGLE_ALLOWLIST_SCHEMA_VERSION:
        raise KaggleCapabilityBlocked(
            "allowlist_invalid", "Kaggle allowlist schema version is unsupported"
        )
    if payload.get("administrator_role") != "data_science":
        raise KaggleCapabilityBlocked(
            "allowlist_role_invalid",
            "Kaggle allowlist must be declared by the data_science administrator role",
        )
    declared_by = _plain_text(payload.get("declared_by"), "allowlist declared_by", 256)
    datasets = _slug_set(payload.get("datasets"), "dataset")
    competitions = _slug_set(payload.get("competitions"), "competition")
    if not datasets and not competitions:
        raise KaggleCapabilityBlocked(
            "allowlist_empty", "Kaggle allowlist contains no approved sources"
        )
    return KaggleAllowlist(
        declared_by=declared_by,
        datasets=datasets,
        competitions=competitions,
        sha256=hashlib.sha256(body).hexdigest(),
    )


def acquire_kaggle_research_asset(
    data_root: Path,
    *,
    allowlist_path: Path,
    source_kind: str,
    slug: str,
    source_version: str,
    license_name: str,
    license_terms_path: Path,
    license_accepted: bool,
    license_accepted_by: str,
    license_accepted_at: datetime,
    expected_archive_sha256: str | None = None,
    asset_id: str | None = None,
    priority: int = 0,
    max_archive_bytes: int = DEFAULT_KAGGLE_ARCHIVE_MAX_BYTES,
    max_unpacked_bytes: int = DEFAULT_LOCAL_ASSET_MAX_BYTES,
    timeout_seconds: int = DEFAULT_KAGGLE_TIMEOUT_SECONDS,
    environ: Mapping[str, str] | None = None,
    executable_resolver: Callable[[str], str | None] = shutil.which,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> KaggleAcquisition:
    """Download one allowlisted Kaggle source and seal a secret-free dataset asset."""

    normalized_kind = _source_kind(source_kind)
    normalized_slug = _slug(slug, normalized_kind)
    normalized_version = _source_version(source_version, normalized_kind)
    allowlist = load_kaggle_allowlist(allowlist_path)
    if not allowlist.permits(normalized_kind, normalized_slug):
        raise KaggleCapabilityBlocked(
            "source_not_allowlisted",
            f"Kaggle {normalized_kind} is not approved by the data_science allowlist",
        )
    normalized_license = _plain_text(license_name, "license", 256)
    normalized_accepted_by = _plain_text(
        license_accepted_by, "license_accepted_by", 256
    )
    if license_accepted is not True:
        raise KaggleCapabilityBlocked(
            "license_not_accepted", "Kaggle source license terms were not accepted"
        )
    if license_accepted_at.tzinfo is None or license_accepted_at.utcoffset() is None:
        raise KaggleCapabilityBlocked(
            "license_acceptance_invalid", "license_accepted_at must include a timezone"
        )
    acquired_at = clock()
    if acquired_at.tzinfo is None or acquired_at.utcoffset() is None:
        raise ValueError("acquisition clock must be timezone-aware")
    if license_accepted_at > acquired_at:
        raise KaggleCapabilityBlocked(
            "license_acceptance_invalid",
            "license_accepted_at must not follow the acquisition time",
        )
    terms = _read_license_terms(license_terms_path)
    terms_sha256 = hashlib.sha256(terms).hexdigest()
    expected_sha256 = str(expected_archive_sha256 or "").strip()
    if expected_sha256 and not _SHA256.fullmatch(expected_sha256):
        raise KaggleCapabilityBlocked(
            "source_revision_invalid", "expected archive SHA-256 is invalid"
        )
    if normalized_kind == "competition" and not expected_sha256:
        raise KaggleCapabilityBlocked(
            "source_revision_unpinned",
            "Kaggle competitions require an expected archive SHA-256",
        )
    if not 1 <= max_archive_bytes <= DEFAULT_LOCAL_ASSET_MAX_BYTES:
        raise ValueError("max_archive_bytes is outside the governed size limit")
    if not 1 <= max_unpacked_bytes <= DEFAULT_LOCAL_ASSET_MAX_BYTES:
        raise ValueError("max_unpacked_bytes is outside the governed size limit")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")

    executable = executable_resolver("kaggle")
    if not executable:
        raise KaggleCapabilityBlocked(
            "kaggle_cli_unavailable", "Kaggle CLI is not installed in the acquisition image"
        )
    credential_environment = _kaggle_credential_environment(environ or os.environ)
    resolved_asset_id = asset_id or derive_asset_id(
        f"kaggle-{normalized_kind}", f"{normalized_slug}@{normalized_version}"
    )

    with tempfile.TemporaryDirectory(prefix="quantlab-kaggle-") as temporary_value:
        temporary = Path(temporary_value)
        download_root = temporary / "download"
        config_root = temporary / "config"
        payload_root = temporary / "payload"
        download_root.mkdir()
        config_root.mkdir()
        payload_root.mkdir()
        command = _kaggle_download_command(
            executable=executable,
            source_kind=normalized_kind,
            slug=normalized_slug,
            source_version=normalized_version,
            download_root=download_root,
        )
        process_environment = _minimal_process_environment(
            environ or os.environ,
            credential_environment,
            config_root=config_root,
            home_root=temporary,
        )
        try:
            completed = command_runner(
                command,
                cwd=str(temporary),
                env=process_environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise KaggleCapabilityBlocked(
                "kaggle_cli_timeout", "Kaggle CLI exceeded the acquisition timeout"
            ) from exc
        except OSError as exc:
            raise KaggleCapabilityBlocked(
                "kaggle_cli_unavailable", "Kaggle CLI could not be started"
            ) from exc
        if completed.returncode != 0:
            # CLI output is deliberately not propagated: provider messages may
            # echo a credential-bearing path or other acquisition-only context.
            raise KaggleCapabilityBlocked(
                "kaggle_cli_failed",
                f"Kaggle CLI exited with status {completed.returncode}",
            )
        archive = _single_downloaded_archive(
            download_root, max_archive_bytes=max_archive_bytes
        )
        archive_sha256, archive_bytes = _hash_regular_file(
            archive, maximum=max_archive_bytes, label="Kaggle archive"
        )
        if expected_sha256 and archive_sha256 != expected_sha256:
            raise UnsafeKaggleArchive(
                "downloaded Kaggle archive disagrees with the approved SHA-256"
            )
        inventory = _extract_kaggle_archive(
            archive,
            payload_root,
            max_unpacked_bytes=max_unpacked_bytes,
        )
        inventory_bytes = _canonical_json_bytes(
            {
                "schema_version": "kaggle-inventory-v1",
                "file_count": len(inventory),
                "total_bytes": sum(int(item["bytes"]) for item in inventory),
                "files": inventory,
            }
        )
        inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
        source_evidence = {
            "schema_version": KAGGLE_ASSET_CONTRACT_VERSION,
            "source_kind": normalized_kind,
            "slug": normalized_slug,
            "source_version": normalized_version,
            "archive_sha256": archive_sha256,
            "archive_bytes": archive_bytes,
            "license": normalized_license,
            "license_terms_sha256": terms_sha256,
            "license_accepted": True,
            "license_accepted_by": normalized_accepted_by,
            "license_accepted_at": license_accepted_at.isoformat(),
            "allowlist_policy_sha256": allowlist.sha256,
            "allowlist_declared_by": allowlist.declared_by,
            "inventory_sha256": inventory_sha256,
        }
        approval_evidence = {
            "schema_version": KAGGLE_ALLOWLIST_SCHEMA_VERSION,
            "administrator_role": "data_science",
            "declared_by": allowlist.declared_by,
            "policy_sha256": allowlist.sha256,
            "source_kind": normalized_kind,
            "slug": normalized_slug,
        }
        _write_governance_file(payload_root, KAGGLE_INVENTORY_PATH, inventory_bytes)
        _write_governance_file(
            payload_root, KAGGLE_SOURCE_PATH, _canonical_json_bytes(source_evidence)
        )
        _write_governance_file(
            payload_root,
            KAGGLE_ALLOWLIST_EVIDENCE_PATH,
            _canonical_json_bytes(approval_evidence),
        )
        _write_governance_file(payload_root, KAGGLE_LICENSE_TERMS_PATH, terms)
        metadata: dict[str, object] = dict(source_evidence)
        metadata.update(
            {
                "inventory_path": KAGGLE_INVENTORY_PATH.as_posix(),
                "source_evidence_path": KAGGLE_SOURCE_PATH.as_posix(),
                "allowlist_evidence_path": KAGGLE_ALLOWLIST_EVIDENCE_PATH.as_posix(),
                "license_terms_path": KAGGLE_LICENSE_TERMS_PATH.as_posix(),
                "credentials_persisted": False,
            }
        )
        credential_values = tuple(
            value
            for name, value in credential_environment.items()
            if name in {"KAGGLE_API_TOKEN", "KAGGLE_KEY"}
        )
        if manifest_contains_kaggle_credentials(
            {"source": source_evidence, "metadata": metadata}, credential_values
        ):
            raise ResearchAssetConflictError(
                "Kaggle credential material must not enter research asset evidence"
            )
        published = register_local_research_asset(
            data_root,
            asset_id=resolved_asset_id,
            kind="dataset",
            source_path=payload_root,
            asset_type=f"kaggle_{normalized_kind}",
            metadata=metadata,
            priority=priority,
            source_kind=f"kaggle_{normalized_kind}",
            source_id=normalized_slug,
            source_version=normalized_version,
            source_sha256=archive_sha256,
            clock=lambda: acquired_at,
        )
        validate_kaggle_research_asset_manifest(
            published.manifest, published.directory
        )
    return KaggleAcquisition(
        published=published,
        source_kind=normalized_kind,
        slug=normalized_slug,
        source_version=normalized_version,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        inventory_sha256=inventory_sha256,
    )


def _source_kind(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"dataset", "competition"}:
        raise KaggleCapabilityBlocked(
            "source_kind_invalid", "Kaggle source kind must be dataset or competition"
        )
    return normalized


def _slug(value: str, source_kind: str) -> str:
    normalized = str(value).strip()
    pattern = _DATASET_SLUG if source_kind == "dataset" else _COMPETITION_SLUG
    if not pattern.fullmatch(normalized):
        raise KaggleCapabilityBlocked(
            "source_slug_invalid", f"Kaggle {source_kind} slug is invalid"
        )
    return normalized


def _source_version(value: str, source_kind: str) -> str:
    normalized = str(value).strip()
    if source_kind == "dataset":
        if not _DATASET_VERSION.fullmatch(normalized):
            raise KaggleCapabilityBlocked(
                "source_revision_invalid",
                "Kaggle dataset source_version must be a positive version number",
            )
    elif not _SAFE_VERSION.fullmatch(normalized):
        raise KaggleCapabilityBlocked(
            "source_revision_invalid", "Kaggle competition source_version is invalid"
        )
    return normalized


def _slug_set(value: object, source_kind: str) -> frozenset[str]:
    if not isinstance(value, list) or len(value) > 1_000:
        raise KaggleCapabilityBlocked(
            "allowlist_invalid", f"Kaggle {source_kind} allowlist must be a bounded list"
        )
    normalized = [_slug(str(item), source_kind) for item in value]
    if len(normalized) != len(set(normalized)):
        raise KaggleCapabilityBlocked(
            "allowlist_invalid", f"Kaggle {source_kind} allowlist contains duplicates"
        )
    return frozenset(normalized)


def _plain_text(value: object, label: str, maximum: int) -> str:
    normalized = str(value or "").strip()
    if (
        not 1 <= len(normalized) <= maximum
        or any(ord(character) < 32 for character in normalized)
    ):
        raise KaggleCapabilityBlocked(
            "governance_invalid", f"Kaggle {label} is missing or invalid"
        )
    return normalized


def _read_license_terms(path: Path) -> bytes:
    body = _read_regular_file(
        path, maximum=_LICENSE_TERMS_MAX_BYTES, label="Kaggle license terms"
    )
    if not body.strip():
        raise KaggleCapabilityBlocked(
            "license_terms_missing", "Kaggle license terms evidence is empty"
        )
    return body


def _read_regular_file(path: Path, *, maximum: int, label: str) -> bytes:
    source = Path(path)
    try:
        before = source.lstat()
    except OSError as exc:
        raise KaggleCapabilityBlocked(
            "governance_file_unavailable", f"{label} file is unavailable"
        ) from exc
    if _metadata_is_linkish(before) or not stat.S_ISREG(before.st_mode):
        raise KaggleCapabilityBlocked(
            "governance_file_unsafe", f"{label} must be one regular non-link file"
        )
    if not 0 < before.st_size <= maximum:
        raise KaggleCapabilityBlocked(
            "governance_file_invalid", f"{label} file is empty or oversized"
        )
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(
        getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(source, flags)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise KaggleCapabilityBlocked(
                    "governance_file_unsafe", f"{label} must be a regular file"
                )
            body = handle.read(maximum + 1)
            final = os.fstat(handle.fileno())
    except OSError as exc:
        raise KaggleCapabilityBlocked(
            "governance_file_unavailable", f"{label} file could not be read"
        ) from exc
    after = source.stat(follow_symlinks=False)
    if (
        len(body) > maximum
        or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        or (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
        or opened.st_size != final.st_size
        or final.st_size != after.st_size
        or opened.st_mtime_ns != final.st_mtime_ns
        or final.st_mtime_ns != after.st_mtime_ns
    ):
        raise ResearchAssetConflictError(f"{label} changed while it was read")
    return body


def _metadata_is_linkish(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        int(getattr(metadata, "st_file_attributes", 0)) & reparse_flag
    )


def _kaggle_credential_environment(environ: Mapping[str, str]) -> dict[str, str]:
    token = str(environ.get("KAGGLE_API_TOKEN") or "").strip()
    username = str(environ.get("KAGGLE_USERNAME") or "").strip()
    key = str(environ.get("KAGGLE_KEY") or "").strip()
    for value in (token, username, key):
        if any(ord(character) < 32 for character in value):
            raise KaggleCapabilityBlocked(
                "kaggle_credentials_invalid", "Kaggle environment credentials are invalid"
            )
    if token:
        return {"KAGGLE_API_TOKEN": token}
    if bool(username) != bool(key):
        raise KaggleCapabilityBlocked(
            "kaggle_credentials_incomplete",
            "KAGGLE_USERNAME and KAGGLE_KEY must both be present",
        )
    if username and key:
        return {"KAGGLE_USERNAME": username, "KAGGLE_KEY": key}
    raise KaggleCapabilityBlocked(
        "kaggle_credentials_missing",
        "Kaggle credentials must be supplied only in the acquisition process environment",
    )


def _minimal_process_environment(
    environ: Mapping[str, str],
    credentials: Mapping[str, str],
    *,
    config_root: Path,
    home_root: Path,
) -> dict[str, str]:
    result = {
        name: str(environ[name])
        for name in _SAFE_ENV_NAMES
        if str(environ.get(name) or "").strip()
    }
    result.update(credentials)
    result.update(
        {
            "KAGGLE_CONFIG_DIR": str(config_root),
            "HOME": str(home_root),
            "USERPROFILE": str(home_root),
            "PYTHONIOENCODING": "utf-8",
        }
    )
    return result


def _kaggle_download_command(
    *,
    executable: str,
    source_kind: str,
    slug: str,
    source_version: str,
    download_root: Path,
) -> list[str]:
    if source_kind == "dataset":
        return [
            executable,
            "datasets",
            "download",
            "--dataset",
            slug,
            "--dataset-version-number",
            source_version,
            "--path",
            str(download_root),
            "--force",
            "--quiet",
        ]
    return [
        executable,
        "competitions",
        "download",
        "--competition",
        slug,
        "--path",
        str(download_root),
        "--force",
        "--quiet",
    ]


def _single_downloaded_archive(root: Path, *, max_archive_bytes: int) -> Path:
    candidates: list[Path] = []
    for current_value, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_value)
        for name in directory_names:
            try:
                metadata = (current / name).lstat()
            except OSError as exc:
                raise UnsafeKaggleArchive("Kaggle download directory changed") from exc
            if _metadata_is_linkish(metadata):
                raise UnsafeKaggleArchive("Kaggle CLI output contains a linked directory")
        for name in file_names:
            path = current / name
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise UnsafeKaggleArchive("Kaggle download file changed") from exc
            if _metadata_is_linkish(metadata) or not stat.S_ISREG(metadata.st_mode):
                raise UnsafeKaggleArchive("Kaggle CLI output is not a regular file")
            if metadata.st_size > max_archive_bytes:
                raise UnsafeKaggleArchive("Kaggle archive exceeds the governed size limit")
            candidates.append(path)
    if len(candidates) != 1 or candidates[0].suffix.casefold() != ".zip":
        raise UnsafeKaggleArchive("Kaggle CLI must produce exactly one ZIP archive")
    return candidates[0]


def _hash_regular_file(path: Path, *, maximum: int, label: str) -> tuple[str, int]:
    before = path.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    total = 0
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(
        getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise UnsafeKaggleArchive(f"{label} is not a regular file")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            total += len(chunk)
            if total > maximum:
                raise UnsafeKaggleArchive(f"{label} exceeds the governed size limit")
            digest.update(chunk)
        final = os.fstat(handle.fileno())
    after = path.stat(follow_symlinks=False)
    if (
        (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        or (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
        or before.st_size != total
        or opened.st_size != final.st_size
        or final.st_size != after.st_size
        or opened.st_mtime_ns != final.st_mtime_ns
        or final.st_mtime_ns != after.st_mtime_ns
    ):
        raise ResearchAssetConflictError(f"{label} changed while it was read")
    return digest.hexdigest(), total


def _extract_kaggle_archive(
    archive: Path, destination: Path, *, max_unpacked_bytes: int
) -> list[dict[str, object]]:
    try:
        package = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise UnsafeKaggleArchive("Kaggle download is not a valid ZIP archive") from exc
    inventory: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    total_declared = 0
    total_extracted = 0
    try:
        members = package.infolist()
        if not members or len(members) > DEFAULT_LOCAL_ASSET_MAX_FILES:
            raise UnsafeKaggleArchive("Kaggle archive file count is invalid")
        for member in members:
            if member.flag_bits & 0x1:
                raise UnsafeKaggleArchive("encrypted Kaggle ZIP entries are not allowed")
            relative = validate_research_asset_relative_path(member.filename.rstrip("/"))
            if relative.parts[0].casefold() == KAGGLE_GOVERNANCE_ROOT.parts[0].casefold():
                raise UnsafeKaggleArchive("Kaggle archive collides with governance evidence")
            normalized_key = relative.as_posix().casefold()
            if normalized_key in seen_paths:
                raise UnsafeKaggleArchive("Kaggle archive contains duplicate paths")
            seen_paths.add(normalized_key)
            unix_mode = (member.external_attr >> 16) & 0xFFFF
            unix_file_type = stat.S_IFMT(unix_mode)
            is_directory = member.is_dir()
            if unix_file_type and stat.S_ISLNK(unix_mode):
                raise UnsafeKaggleArchive("Kaggle archive contains a symbolic link")
            if unix_file_type and not (
                stat.S_ISREG(unix_mode) or stat.S_ISDIR(unix_mode)
            ):
                raise UnsafeKaggleArchive("Kaggle archive contains a special file")
            if is_directory:
                (destination / relative).mkdir(parents=True, exist_ok=True)
                continue
            total_declared += member.file_size
            if member.file_size < 0 or total_declared > max_unpacked_bytes:
                raise UnsafeKaggleArchive("Kaggle archive exceeds the unpacked size limit")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise UnsafeKaggleArchive("Kaggle archive contains a path collision")
            digest = hashlib.sha256()
            written = 0
            try:
                with package.open(member, "r") as source, target.open("xb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        written += len(chunk)
                        if (
                            written > member.file_size
                            or total_extracted + written > max_unpacked_bytes
                        ):
                            raise UnsafeKaggleArchive(
                                "Kaggle archive exceeded its declared size while unpacking"
                            )
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            except Exception:
                target.unlink(missing_ok=True)
                raise
            if written != member.file_size:
                target.unlink(missing_ok=True)
                raise UnsafeKaggleArchive("Kaggle ZIP entry size is inconsistent")
            total_extracted += written
            inventory.append(
                {
                    "path": relative.as_posix(),
                    "bytes": written,
                    "sha256": digest.hexdigest(),
                }
            )
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        if isinstance(exc, KaggleAssetError):
            raise
        raise UnsafeKaggleArchive("Kaggle ZIP extraction failed") from exc
    finally:
        package.close()
    if not inventory:
        raise UnsafeKaggleArchive("Kaggle archive contains no regular files")
    return sorted(inventory, key=lambda item: str(item["path"]))


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sealed_evidence_body(
    asset_root: Path,
    relative: Path,
    file_entries: Mapping[str, Mapping[str, object]],
) -> bytes:
    entry = file_entries[relative.as_posix()]
    maximum = (
        _LICENSE_TERMS_MAX_BYTES
        if relative == KAGGLE_LICENSE_TERMS_PATH
        else _POLICY_MAX_BYTES
    )
    body = _read_regular_file(asset_root / relative, maximum=maximum, label=relative.as_posix())
    if len(body) != entry.get("bytes") or hashlib.sha256(body).hexdigest() != entry.get(
        "sha256"
    ):
        raise ValueError(f"Kaggle sealed evidence {relative.as_posix()} is invalid")
    return body


def _evidence_json(body: bytes, label: str) -> dict[str, object]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Kaggle sealed {label} evidence is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Kaggle sealed {label} evidence must be an object")
    return payload


def _write_governance_file(root: Path, relative: Path, body: bytes) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def manifest_contains_kaggle_credentials(payload: object, credentials: Sequence[str]) -> bool:
    """Testable redaction invariant for manifests and result payloads."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return any(value and value in encoded for value in credentials)
