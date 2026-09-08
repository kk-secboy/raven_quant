from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

from cryptography.fernet import Fernet
from dotenv import dotenv_values

from .control_plane_lock import control_plane_locked
from .deployment_services import WRITER_SERVICES

FULL_BACKUP_FORMAT_VERSION = 1
CONTROL_PLANE_BACKUP_FORMAT_VERSION = 2
BACKUP_FILES = ("manifest.json", "quantlab-postgres.dump", "quantlab-data.tar.gz")
CONTROL_PLANE_BACKUP_FILES = (
    "manifest.json",
    "quantlab-postgres.dump",
    "quantlab-control-plane.tar.gz",
)
CONTROL_PLANE_ARCHIVE_NAME = "quantlab-control-plane.tar.gz"
CONTROL_PLANE_INVENTORY_MEMBER = "immutable-data-manifest-inventory.json"
CONTROL_PLANE_DEPLOYMENT_MEMBER = "deployment/snapshot.json"
CONTROL_PLANE_ENV_MEMBER = "deployment/environment.sanitized.env"
CONTROL_PLANE_MAX_COMPOSE_FILES = 8
CONTROL_PLANE_MAX_CONFIG_FILE_BYTES = 2 * 1024 * 1024
CONTROL_PLANE_MAX_CONFIG_TOTAL_BYTES = 8 * 1024 * 1024
CONTROL_PLANE_MAX_INVENTORY_BYTES = 4 * 1024 * 1024
CONTROL_PLANE_MAX_INVENTORY_ENTRIES = 2048
CONTROL_PLANE_MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
CONTROL_PLANE_MAX_ARCHIVE_MEMBERS = 32
CONTROL_PLANE_MAX_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
_GIB = 1024**3
_STREAM_READ_CHARS = 64 * 1024
_STREAM_TAIL_CHARS = 16 * 1024
_ERROR_DETAIL_CHARS = 4 * 1024
_VERIFICATION_RECEIPT_NAME = ".quantlab-backup-verified-v1.json"
_VERIFICATION_RECEIPT_VERSION = 1
_VERIFICATION_RECEIPT_MAX_BYTES = 64 * 1024
_SENSITIVE_ENV_NAME = re.compile(
    r"(?:^|[_-])(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|credential)(?:$|[_-])",
    re.IGNORECASE,
)
_SENSITIVE_LABEL = (
    r"(?:[A-Za-z0-9]+[_-])*(?:password|passwd|secret|token|api[_-]?key|"
    r"access[_-]?key|private[_-]?key|credential)(?:[_-][A-Za-z0-9]+)*"
)
_SENSITIVE_QUOTED_ASSIGNMENT = re.compile(
    rf"(?P<label>{_SENSITIVE_LABEL}\s*(?:=|:)\s*)"
    r"(?P<quote>[\"'])(?P<value>.*?)(?P=quote)",
    re.IGNORECASE,
)
_SENSITIVE_ASSIGNMENT = re.compile(
    rf"(?P<label>{_SENSITIVE_LABEL}\s*(?:=|:)\s*)(?P<value>[^\s,;\"']+)",
    re.IGNORECASE,
)
_SENSITIVE_OPTION = re.compile(
    r"(?P<label>--(?:password|passwd|secret|token|api-key|access-key|"
    r"private-key|credential)(?:=|\s+))(?P<value>\S+)",
    re.IGNORECASE,
)
_SENSITIVE_YAML_BLOCK = re.compile(
    rf"^(?P<indent>\s*)(?P<prefix>-\s+)?"
    rf"(?P<label>{_SENSITIVE_LABEL}\s*:\s*)[|>](?:[-+])?\d*\s*$",
    re.IGNORECASE,
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+\S+")
_URL_CREDENTIALS = re.compile(
    r"(?i)([a-z][a-z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@"
)

_IMMUTABLE_DATA_INVENTORY_SCRIPT = f"""
import hashlib
import json
import os
from datetime import datetime, timezone

root = "/data"
limit = {CONTROL_PLANE_MAX_INVENTORY_ENTRIES}
entries = []
truncated = False
for current, directories, files in os.walk(root, followlinks=False):
    directories[:] = sorted(
        item
        for item in directories
        if not os.path.islink(os.path.join(current, item))
    )
    if "manifest.json" not in files:
        continue
    path = os.path.join(current, "manifest.json")
    if os.path.islink(path) or not os.path.isfile(path):
        continue
    if len(entries) >= limit:
        truncated = True
        break
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = os.stat(path, follow_symlinks=False)
    entries.append({{
        "path": os.path.relpath(path, root).replace(os.sep, "/"),
        "sha256": digest.hexdigest(),
        "bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
    }})
payload = {{
    "contract_version": "quantlab-immutable-data-inventory-v1",
    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "root": root,
    "manifest_name": "manifest.json",
    "immutable_data_copied": False,
    "max_entries": limit,
    "entry_count": len(entries),
    "truncated": truncated,
    "entries": entries,
}}
print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
""".strip()


class _BoundedTextTail:
    def __init__(self, limit: int = _STREAM_TAIL_CHARS) -> None:
        self._limit = limit
        self._value = ""

    def append(self, value: str) -> None:
        if not value:
            return
        self._value = (self._value + value)[-self._limit :]

    @property
    def value(self) -> str:
        return self._value


def _known_sensitive_values(env_file: Path) -> tuple[str, ...]:
    values: set[str] = set()
    try:
        configured = dotenv_values(env_file)
    except OSError:
        configured = {}
    for source in (configured, os.environ):
        for name, raw in source.items():
            value = str(raw or "")
            if _SENSITIVE_ENV_NAME.search(str(name)) and 6 <= len(value) <= 4096:
                values.add(value)
    return tuple(sorted(values, key=len, reverse=True))


def _redact_output(value: str, sensitive_values: tuple[str, ...]) -> str:
    redacted = value
    for secret in sensitive_values:
        redacted = redacted.replace(secret, "[REDACTED]")
    redacted = _BEARER_TOKEN.sub("Bearer [REDACTED]", redacted)
    redacted = _URL_CREDENTIALS.sub(r"\1[REDACTED]@", redacted)
    redacted = _SENSITIVE_OPTION.sub(r"\g<label>[REDACTED]", redacted)
    redacted = _SENSITIVE_QUOTED_ASSIGNMENT.sub(
        r"\g<label>\g<quote>[REDACTED]\g<quote>",
        redacted,
    )
    return _SENSITIVE_ASSIGNMENT.sub(r"\g<label>[REDACTED]", redacted)


def _valid_sha256(value: object) -> bool:
    candidate = str(value or "").lower()
    return len(candidate) == 64 and all(
        character in "0123456789abcdef" for character in candidate
    )


def _sanitize_config_file(
    path: Path,
    sensitive_values: tuple[str, ...],
) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"deployment snapshot input cannot be read: {path.name}") from exc
    if len(raw) > CONTROL_PLANE_MAX_CONFIG_FILE_BYTES:
        raise ValueError(f"deployment snapshot input is too large: {path.name}")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"deployment snapshot input is not UTF-8: {path.name}") from exc
    lines: list[str] = []
    multiline_quote: str | None = None
    yaml_secret_indent: int | None = None
    for line in text.splitlines():
        if multiline_quote is not None:
            if multiline_quote in line:
                multiline_quote = None
            continue
        if yaml_secret_indent is not None:
            if not line.strip():
                continue
            indentation = len(line) - len(line.lstrip())
            if indentation > yaml_secret_indent:
                continue
            yaml_secret_indent = None
        block_match = _SENSITIVE_YAML_BLOCK.match(line)
        if block_match is not None:
            indentation = block_match.group("indent")
            prefix = block_match.group("prefix") or ""
            lines.append(
                f"{indentation}{prefix}{block_match.group('label')}[REDACTED]"
            )
            yaml_secret_indent = len(indentation)
            continue
        assignment, separator, raw_value = line.partition("=")
        key = assignment.strip() if separator else ""
        if key and _SENSITIVE_ENV_NAME.search(key):
            line = f"{assignment}=[REDACTED]"
            value = raw_value.strip()
            if value[:1] in {'"', "'"} and not value[1:].endswith(value[0]):
                multiline_quote = value[0]
        lines.append(line)
    sanitized = _redact_output("\n".join(lines) + "\n", sensitive_values)
    for secret in sensitive_values:
        if secret in sanitized:
            raise ValueError("deployment snapshot sanitization did not remove a known secret")
    return sanitized.encode("utf-8")


def _validated_manifest_inventory(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("immutable-data manifest inventory is invalid")
    if payload.get("contract_version") != "quantlab-immutable-data-inventory-v1":
        raise ValueError("immutable-data manifest inventory contract is unsupported")
    if payload.get("root") != "/data" or payload.get("manifest_name") != "manifest.json":
        raise ValueError("immutable-data manifest inventory root is invalid")
    if payload.get("immutable_data_copied") is not False:
        raise ValueError("control-plane backup must declare immutable data was not copied")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) > CONTROL_PLANE_MAX_INVENTORY_ENTRIES:
        raise ValueError("immutable-data manifest inventory exceeds its entry bound")
    if payload.get("entry_count") != len(entries):
        raise ValueError("immutable-data manifest inventory count is invalid")
    if payload.get("max_entries") != CONTROL_PLANE_MAX_INVENTORY_ENTRIES:
        raise ValueError("immutable-data manifest inventory bound is invalid")
    if not isinstance(payload.get("truncated"), bool):
        raise ValueError("immutable-data manifest inventory truncation state is invalid")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("immutable-data manifest inventory entry is invalid")
        path = PurePosixPath(str(entry.get("path") or ""))
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or path.name != "manifest.json"
        ):
            raise ValueError("immutable-data manifest inventory path is invalid")
        if not _valid_sha256(entry.get("sha256")):
            raise ValueError("immutable-data manifest inventory checksum is invalid")
        for field in ("bytes", "mtime_ns"):
            value = entry.get(field)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"immutable-data manifest inventory {field} is invalid"
                )
    return payload


def _capture_manifest_inventory(context: ComposeContext) -> dict[str, Any]:
    raw = context.run(
        "run",
        "--rm",
        "--no-deps",
        "api",
        "python",
        "-c",
        _IMMUTABLE_DATA_INVENTORY_SCRIPT,
        capture=True,
    )
    encoded = raw.encode("utf-8")
    if len(encoded) > CONTROL_PLANE_MAX_INVENTORY_BYTES:
        raise ValueError("immutable-data manifest inventory exceeds its byte bound")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("immutable-data manifest inventory is not valid JSON") from exc
    return _validated_manifest_inventory(payload)


def _write_control_plane_archive(
    context: ComposeContext,
    staging: Path,
    inventory: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    compose_files = tuple(
        Path(item).resolve() for item in getattr(context, "compose_files", ())
    )
    if not compose_files:
        raise ValueError("control-plane backup requires at least one Compose file")
    if len(compose_files) > CONTROL_PLANE_MAX_COMPOSE_FILES:
        raise ValueError("control-plane backup has too many Compose files")
    snapshot_root = staging / ".control-plane"
    deployment_root = snapshot_root / "deployment"
    deployment_root.mkdir(parents=True, mode=0o700)
    sensitive_values = _known_sensitive_values(context.env_file)
    environment = _sanitize_config_file(context.env_file.resolve(), sensitive_values)
    if len(environment) > CONTROL_PLANE_MAX_CONFIG_TOTAL_BYTES:
        raise ValueError("sanitized deployment snapshot exceeds its byte bound")
    environment_path = snapshot_root / CONTROL_PLANE_ENV_MEMBER
    environment_path.write_bytes(environment)
    environment_path.chmod(0o600)
    total_config_bytes = len(environment)
    compose_entries: list[dict[str, Any]] = []
    for index, source in enumerate(compose_files, start=1):
        content = _sanitize_config_file(source, sensitive_values)
        total_config_bytes += len(content)
        if total_config_bytes > CONTROL_PLANE_MAX_CONFIG_TOTAL_BYTES:
            raise ValueError("sanitized deployment snapshot exceeds its byte bound")
        configured_suffix = source.suffix.lower()
        suffix = (
            configured_suffix
            if configured_suffix in {".json", ".yaml", ".yml"}
            else ".yaml"
        )
        member = f"deployment/compose-{index:02d}{suffix}"
        target = snapshot_root / member
        target.write_bytes(content)
        target.chmod(0o600)
        compose_entries.append(
            {
                "member": member,
                "source_name": source.name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    inventory_bytes = (
        json.dumps(
            inventory,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if len(inventory_bytes) > CONTROL_PLANE_MAX_INVENTORY_BYTES:
        raise ValueError("immutable-data manifest inventory exceeds its byte bound")
    inventory_path = snapshot_root / CONTROL_PLANE_INVENTORY_MEMBER
    inventory_path.write_bytes(inventory_bytes)
    inventory_path.chmod(0o600)
    snapshot = {
        "format_version": 1,
        "project_name": context.project_name,
        "profiles": list(getattr(context, "profiles", ())),
        "environment": {
            "member": CONTROL_PLANE_ENV_MEMBER,
            "sha256": hashlib.sha256(environment).hexdigest(),
            "bytes": len(environment),
            "sanitized": True,
        },
        "compose_files": compose_entries,
        "immutable_data": {
            "copied": False,
            "inventory_member": CONTROL_PLANE_INVENTORY_MEMBER,
            "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            "inventory_entries": len(inventory["entries"]),
            "inventory_truncated": inventory["truncated"],
        },
    }
    snapshot_bytes = (
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    snapshot_path = snapshot_root / CONTROL_PLANE_DEPLOYMENT_MEMBER
    snapshot_path.write_bytes(snapshot_bytes)
    snapshot_path.chmod(0o600)
    archive = staging / CONTROL_PLANE_ARCHIVE_NAME
    try:
        with tarfile.open(archive, mode="w:gz", format=tarfile.PAX_FORMAT) as handle:
            handle.add(deployment_root, arcname="deployment", recursive=True)
            handle.add(inventory_path, arcname=CONTROL_PLANE_INVENTORY_MEMBER)
    finally:
        shutil.rmtree(snapshot_root, ignore_errors=True)
    archive.chmod(0o600)
    if archive.stat().st_size > CONTROL_PLANE_MAX_ARCHIVE_BYTES:
        raise ValueError("control-plane archive exceeds its byte bound")
    return archive, snapshot


def _read_archive_member(
    archive: tarfile.TarFile,
    member_name: str,
    *,
    limit: int,
) -> bytes:
    try:
        member = archive.getmember(member_name)
    except KeyError as exc:
        raise ValueError(f"control-plane archive member is missing: {member_name}") from exc
    if not member.isfile() or member.size > limit:
        raise ValueError(f"control-plane archive member is invalid: {member_name}")
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"control-plane archive member is unreadable: {member_name}")
    payload = source.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"control-plane archive member is too large: {member_name}")
    return payload


def _validate_control_plane_archive(
    archive_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if archive_path.stat().st_size > CONTROL_PLANE_MAX_ARCHIVE_BYTES:
        raise ValueError("control-plane archive exceeds its byte bound")
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > CONTROL_PLANE_MAX_ARCHIVE_MEMBERS:
                raise ValueError("control-plane archive has too many members")
            total_bytes = 0
            names: set[str] = set()
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or member.name in names:
                    raise ValueError("control-plane archive contains an unsafe path")
                if not (member.isdir() or member.isfile()) or member.issym() or member.islnk():
                    raise ValueError("control-plane archive contains an unsafe member")
                names.add(member.name)
                total_bytes += member.size
            if total_bytes > CONTROL_PLANE_MAX_UNCOMPRESSED_BYTES:
                raise ValueError("control-plane archive exceeds its uncompressed byte bound")
            required = {
                CONTROL_PLANE_DEPLOYMENT_MEMBER,
                CONTROL_PLANE_ENV_MEMBER,
                CONTROL_PLANE_INVENTORY_MEMBER,
            }
            if not required.issubset(names):
                raise ValueError("control-plane archive is incomplete")
            inventory_bytes = _read_archive_member(
                archive,
                CONTROL_PLANE_INVENTORY_MEMBER,
                limit=CONTROL_PLANE_MAX_INVENTORY_BYTES,
            )
            try:
                inventory = _validated_manifest_inventory(json.loads(inventory_bytes))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("control-plane inventory member is invalid") from exc
            snapshot_bytes = _read_archive_member(
                archive,
                CONTROL_PLANE_DEPLOYMENT_MEMBER,
                limit=CONTROL_PLANE_MAX_CONFIG_FILE_BYTES,
            )
            try:
                snapshot = json.loads(snapshot_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("control-plane deployment snapshot is invalid") from exc
            if not isinstance(snapshot, dict) or snapshot.get("format_version") != 1:
                raise ValueError("control-plane deployment snapshot is invalid")
            immutable = snapshot.get("immutable_data") or {}
            if immutable.get("copied") is not False:
                raise ValueError("control-plane deployment snapshot copied immutable data")
            if immutable.get("inventory_member") != CONTROL_PLANE_INVENTORY_MEMBER:
                raise ValueError("control-plane inventory member identity is invalid")
            inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
            if immutable.get("inventory_sha256") != inventory_sha256:
                raise ValueError("control-plane inventory checksum mismatch")
            declared = manifest.get("immutable_data") or {}
            if declared.get("inventory_sha256") != inventory_sha256:
                raise ValueError("backup manifest inventory checksum mismatch")
            if (
                declared.get("inventory_entries") != inventory["entry_count"]
                or declared.get("inventory_truncated") != inventory["truncated"]
                or immutable.get("inventory_entries") != inventory["entry_count"]
                or immutable.get("inventory_truncated") != inventory["truncated"]
            ):
                raise ValueError("control-plane inventory summary mismatch")
            config_entries = [snapshot.get("environment") or {}]
            compose_entries = snapshot.get("compose_files")
            if not isinstance(compose_entries, list) or not compose_entries:
                raise ValueError("control-plane Compose snapshot is missing")
            config_entries.extend(compose_entries)
            for entry in config_entries:
                if not isinstance(entry, dict):
                    raise ValueError("control-plane config snapshot is invalid")
                member_name = str(entry.get("member") or "")
                if member_name not in names:
                    raise ValueError("control-plane config snapshot member is missing")
                content = _read_archive_member(
                    archive,
                    member_name,
                    limit=CONTROL_PLANE_MAX_CONFIG_FILE_BYTES,
                )
                if entry.get("sha256") != hashlib.sha256(content).hexdigest():
                    raise ValueError("control-plane config snapshot checksum mismatch")
                if entry.get("bytes") != len(content):
                    raise ValueError("control-plane config snapshot size mismatch")
                try:
                    config_text = content.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("control-plane config snapshot is not UTF-8") from exc
                if _redact_output(config_text, ()) != config_text:
                    raise ValueError("control-plane config snapshot is not sanitized")
    except (OSError, tarfile.TarError) as exc:
        raise ValueError("control-plane archive is invalid") from exc
    return inventory


def _stream_output(
    source: TextIO,
    destination: TextIO,
    tail: _BoundedTextTail,
    sensitive_values: tuple[str, ...],
) -> None:
    try:
        while chunk := source.readline(_STREAM_READ_CHARS):
            safe_chunk = _redact_output(chunk, sensitive_values)
            tail.append(safe_chunk)
            try:
                destination.write(safe_chunk)
                destination.flush()
            except (OSError, UnicodeError, ValueError):
                pass
    finally:
        source.close()


def _failure_detail(stdout: str, stderr: str) -> str:
    stdout = stdout.strip()
    stderr = stderr.strip()
    if not stdout and not stderr:
        return "no command output"
    streams = (("stdout", stdout), ("stderr", stderr))
    populated = [(name, value) for name, value in streams if value]
    budget = _ERROR_DETAIL_CHARS // len(populated)
    detail: list[str] = []
    for name, value in populated:
        truncated = len(value) > budget
        tail = value[-budget:]
        marker = " (tail; earlier output omitted)" if truncated else ""
        detail.append(f"{name}{marker}:\n{tail}")
    return "\n".join(detail)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
    }


def _sha256_with_stable_identity(path: Path) -> tuple[str, dict[str, int]]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    before_identity = _stat_identity(before)
    if before_identity != _stat_identity(after):
        raise ValueError("backup archive changed while its checksum was being verified")
    return digest.hexdigest(), before_identity


def _root_owned_nonwritable_regular_file(
    value: os.stat_result,
    *,
    exact_mode: int | None = None,
) -> bool:
    mode = stat.S_IMODE(value.st_mode)
    if not stat.S_ISREG(value.st_mode) or int(value.st_uid) != 0:
        return False
    if exact_mode is not None:
        return mode == exact_mode
    return mode & 0o022 == 0


def _verification_receipt_environment_is_trusted(
    backup_directory: Path,
    manifest_path: Path,
    archive_paths: Sequence[Path],
) -> bool:
    if os.name != "posix" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        return False
    try:
        root_stat = backup_directory.lstat()
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or int(root_stat.st_uid) != 0
            or stat.S_IMODE(root_stat.st_mode) & 0o022
        ):
            return False
        for path in (manifest_path, *archive_paths):
            if path.is_symlink() or not _root_owned_nonwritable_regular_file(path.lstat()):
                return False
    except OSError:
        return False
    return True


def _load_verification_receipt(backup_directory: Path) -> dict[str, Any] | None:
    receipt_path = backup_directory / _VERIFICATION_RECEIPT_NAME
    try:
        receipt_stat = receipt_path.lstat()
        if receipt_path.is_symlink() or not _root_owned_nonwritable_regular_file(
            receipt_stat,
            exact_mode=0o600,
        ):
            return None
        if receipt_stat.st_size > _VERIFICATION_RECEIPT_MAX_BYTES:
            return None
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(receipt_path, flags)
        try:
            opened_stat = os.fstat(descriptor)
            if (
                _stat_identity(opened_stat) != _stat_identity(receipt_stat)
                or not _root_owned_nonwritable_regular_file(
                    opened_stat,
                    exact_mode=0o600,
                )
            ):
                return None
            payload = bytearray()
            while len(payload) <= _VERIFICATION_RECEIPT_MAX_BYTES:
                chunk = os.read(
                    descriptor,
                    min(
                        8192,
                        _VERIFICATION_RECEIPT_MAX_BYTES + 1 - len(payload),
                    ),
                )
                if not chunk:
                    break
                payload.extend(chunk)
            if len(payload) > _VERIFICATION_RECEIPT_MAX_BYTES:
                return None
            if _stat_identity(os.fstat(descriptor)) != _stat_identity(opened_stat):
                return None
        finally:
            os.close(descriptor)
        receipt = json.loads(bytes(payload).decode("utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return receipt if isinstance(receipt, dict) else None


def _write_verification_receipt(
    backup_directory: Path,
    verification_key: dict[str, Any],
) -> None:
    receipt_path = backup_directory / _VERIFICATION_RECEIPT_NAME
    temporary = backup_directory / f".{_VERIFICATION_RECEIPT_NAME}.{uuid.uuid4().hex}.tmp"
    payload = (
        json.dumps(
            {
                "format_version": _VERIFICATION_RECEIPT_VERSION,
                "verified_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "verification_key": verification_key,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > _VERIFICATION_RECEIPT_MAX_BYTES:
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written < 1:
                raise OSError("verification receipt write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, receipt_path)
        os.chmod(receipt_path, 0o600, follow_symlinks=False)
    except OSError:
        # A cache write must never make a valid backup unusable. The next
        # validation safely falls back to reading both archives again.
        return
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _absolute_host_path(value: str) -> Path | None:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else None


def _configured_data_host_path(context: ComposeContext) -> Path:
    configured = dotenv_values(context.env_file)
    raw = os.environ.get("QUANTLAB_DATA_HOST_PATH")
    value = (
        raw
        if raw is not None
        else configured.get("QUANTLAB_DATA_HOST_PATH") or "/data/quantlab"
    )
    candidate = Path(str(value))
    if not candidate.is_absolute():
        raise RuntimeError("QUANTLAB_DATA_HOST_PATH must be an absolute host path")
    resolved = candidate.resolve()
    if resolved.parent == resolved or len(resolved.parts) < 3:
        raise RuntimeError("QUANTLAB_DATA_HOST_PATH is too broad for destructive restore")
    return resolved


def _require_backup_outside_data(
    data_source: str,
    backup_path: Path,
    *,
    require_sibling: bool = True,
) -> None:
    data_path = _absolute_host_path(data_source)
    if data_path is None:
        return
    resolved = backup_path.resolve()
    if require_sibling and (
        resolved == data_path.parent or resolved.parent != data_path.parent
    ):
        raise ValueError(
            "backup root must be a dedicated sibling directory of the governed data target"
        )
    if (
        resolved == data_path
        or _inside(resolved, data_path)
        or _inside(data_path, resolved)
    ):
        raise ValueError(
            "backup root and governed /data bind target must not contain one another"
        )


def _restore_target(
    context: ComposeContext,
    data_source: str,
    backup_directory: Path,
) -> Path:
    target = _absolute_host_path(data_source)
    if target is None:
        raise RuntimeError("restore requires an absolute /data bind mount, not a named volume")
    configured = _configured_data_host_path(context)
    if target != configured:
        raise RuntimeError("the live /data bind source does not match QUANTLAB_DATA_HOST_PATH")
    if not target.is_dir():
        raise RuntimeError("the governed /data bind target does not exist or is not a directory")
    if _inside(backup_directory, target) or _inside(target, backup_directory):
        raise ValueError("backup directory and restore target must not contain one another")
    return target


def _volume_usage_bytes(context: ComposeContext, source: str) -> int:
    raw = context.docker(
        "run",
        "--rm",
        "--volume",
        f"{source}:/source:ro",
        "postgres:16-alpine",
        "du",
        "-sk",
        "/source",
        capture=True,
    )
    return int(raw.splitlines()[-1].split()[0]) * 1024


def _validate_archive_listing(listing: str) -> None:
    entries = [item.strip() for item in listing.splitlines() if item.strip()]
    if not entries:
        raise ValueError("backup data archive is empty")
    for entry in entries:
        candidate = PurePosixPath(entry)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("backup data archive contains an unsafe path")


def _atomic_marker(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _begin_restore_marker(
    marker: Path,
    *,
    target: Path,
    backup_directory: Path,
    data_sha256: str,
) -> dict[str, Any]:
    identity = {
        "target": str(target),
        "backup_directory": str(backup_directory),
        "data_sha256": data_sha256,
    }
    attempt = 1
    if marker.exists():
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("existing restore marker is invalid") from exc
        if any(previous.get(key) != value for key, value in identity.items()):
            raise RuntimeError("a different incomplete restore marker already exists")
        if previous.get("state") == "running":
            raise RuntimeError("the same restore is already marked as running")
        attempt = int(previous.get("attempt") or 0) + 1
    payload = {
        **identity,
        "state": "running",
        "attempt": attempt,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _atomic_marker(marker, payload)
    return payload


def _data_mount_source(inspection: dict[str, Any]) -> str:
    for mount in inspection.get("Mounts", []):
        if mount.get("Destination") != "/data":
            continue
        name = str(mount.get("Name") or "").strip()
        if name:
            return name
        if mount.get("Type") == "bind":
            source = str(mount.get("Source") or "").strip()
            if source and PurePosixPath(source).is_absolute():
                return source
            raise RuntimeError("the /data bind mount source must be an absolute path")
    raise RuntimeError("unable to resolve the /data Docker volume or bind mount")


def _platform_secret_key_fingerprint(context: ComposeContext) -> str:
    raw = os.getenv("PLATFORM_SECRET_KEY")
    if raw is None:
        raw = dotenv_values(context.env_file).get("PLATFORM_SECRET_KEY")
    key = str(raw or "").strip()
    if not key:
        raise ValueError("PLATFORM_SECRET_KEY is required for backup and restore")
    try:
        Fernet(key.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise ValueError("PLATFORM_SECRET_KEY must be a valid Fernet key") from exc
    return hashlib.sha256(key.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class ComposeContext:
    project_name: str
    env_file: Path
    compose_files: tuple[Path, ...]
    profiles: tuple[str, ...] = ()
    project_directory: Path | None = None

    @property
    def prefix(self) -> list[str]:
        command = [
            "docker",
            "compose",
            "--project-name",
            self.project_name,
            "--env-file",
            str(self.env_file.resolve()),
        ]
        if self.project_directory is not None:
            command.extend(("--project-directory", str(self.project_directory.resolve())))
        for compose_file in self.compose_files:
            command.extend(("-f", str(compose_file.resolve())))
        for profile in self.profiles:
            command.extend(("--profile", profile))
        return command

    def run(
        self,
        *arguments: str,
        capture: bool = False,
        check: bool = True,
        timeout: int | None = None,
    ) -> str:
        if not capture:
            sensitive_values = _known_sensitive_values(self.env_file)
            process = subprocess.Popen(
                [*self.prefix, *arguments],
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if process.stdout is None or process.stderr is None:
                process.kill()
                process.wait()
                raise RuntimeError("docker compose output pipes could not be created")
            stdout_tail = _BoundedTextTail()
            stderr_tail = _BoundedTextTail()
            readers = (
                threading.Thread(
                    target=_stream_output,
                    args=(process.stdout, sys.stdout, stdout_tail, sensitive_values),
                    daemon=True,
                    name="compose-stdout",
                ),
                threading.Thread(
                    target=_stream_output,
                    args=(process.stderr, sys.stderr, stderr_tail, sensitive_values),
                    daemon=True,
                    name="compose-stderr",
                ),
            )
            for reader in readers:
                reader.start()
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise
            finally:
                for reader in readers:
                    reader.join()
            if check and returncode != 0:
                detail = _failure_detail(stdout_tail.value, stderr_tail.value)
                raise RuntimeError(
                    f"docker compose failed with exit code {returncode}: {detail}"
                )
            return ""

        completed = subprocess.run(
            [*self.prefix, *arguments],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=capture,
            timeout=timeout,
        )
        if check and completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "no command output").strip()
            raise RuntimeError(
                f"docker compose failed with exit code {completed.returncode}: {detail[:2000]}"
            )
        return completed.stdout.strip() if capture else ""

    def docker(
        self,
        *arguments: str,
        capture: bool = False,
        check: bool = True,
        timeout: int | None = None,
    ) -> str:
        completed = subprocess.run(
            ["docker", *arguments],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=capture,
            timeout=timeout,
        )
        if check and completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "no command output").strip()
            raise RuntimeError(
                f"docker failed with exit code {completed.returncode}: {detail[:2000]}"
            )
        return completed.stdout.strip() if capture else ""

    def running_services(self) -> list[str]:
        output = self.run("ps", "--services", "--filter", "status=running", capture=True)
        return [item.strip() for item in output.splitlines() if item.strip()]

    def container_id(self, service: str, *, all_states: bool = False) -> str:
        args = ["ps"]
        if all_states:
            args.append("-a")
        args.extend(("-q", service))
        output = self.run(*args, capture=True)
        return output.splitlines()[0].strip() if output else ""

    def data_volume(self) -> str:
        api_id = self.container_id("api", all_states=True)
        if not api_id:
            self.run("create", "api")
            api_id = self.container_id("api", all_states=True)
        if not api_id:
            raise RuntimeError("unable to create the API service for /data volume discovery")
        inspection = json.loads(self.docker("inspect", api_id, capture=True))[0]
        return _data_mount_source(inspection)


def _existing_storage_anchor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(path)
        candidate = parent
    return candidate


def assess_control_plane_backup_capacity(
    context: ComposeContext,
    backup_root: Path,
    *,
    minimum_free_gb: float,
) -> dict[str, Any]:
    """Measure only the v2 database, bounded archive, and retained headroom."""

    try:
        if minimum_free_gb < 0:
            raise ValueError("minimum_free_gb must not be negative")
        raw = context.run(
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "quantlab",
            "-d",
            "postgres",
            "-Atc",
            "SELECT pg_database_size('quantlab');",
            capture=True,
        )
        database_upper_bound = int(raw.splitlines()[-1])
        if database_upper_bound < 1:
            raise ValueError("PostgreSQL database size must be positive")
        anchor = _existing_storage_anchor(backup_root)
        free_bytes = shutil.disk_usage(anchor).free
        required_bytes = (
            database_upper_bound
            + CONTROL_PLANE_MAX_ARCHIVE_BYTES
            + int(minimum_free_gb * _GIB)
        )
        passed = free_bytes >= required_bytes
        evidence = (
            f"target {backup_root.resolve()}; free {free_bytes / _GIB:.1f} GiB; "
            f"database upper bound {database_upper_bound / _GIB:.1f} GiB; "
            f"bounded control archive {CONTROL_PLANE_MAX_ARCHIVE_BYTES / _GIB:.3f} GiB; "
            "immutable /data not copied; retained headroom "
            f"{minimum_free_gb:.1f} GiB; required {required_bytes / _GIB:.1f} GiB"
        )
    except Exception as exc:
        passed = False
        evidence = (
            "control-plane backup capacity could not be measured: "
            f"{type(exc).__name__}: {exc}"
        )
        database_upper_bound = None
        free_bytes = None
        required_bytes = None
    return {
        "id": "control_plane_backup_capacity",
        "status": "pass" if passed else "block",
        "evidence": evidence,
        "database_upper_bound_bytes": database_upper_bound,
        "free_bytes": free_bytes,
        "required_bytes": required_bytes,
        "minimum_free_gb": minimum_free_gb,
    }


def assess_control_plane_backup_readiness(
    context: ComposeContext,
    backup_root: Path,
    *,
    minimum_free_gb: float,
) -> dict[str, Any]:
    """Check backup-local prerequisites without consulting business readiness."""

    checks: list[dict[str, Any]] = []
    configuration_paths = (context.env_file, *context.compose_files)
    missing = [str(item) for item in configuration_paths if not item.is_file()]
    configuration_error: str | None = None
    if not missing:
        try:
            context.run("config", "--quiet")
        except Exception as exc:
            configuration_error = f"{type(exc).__name__}: {exc}"
    configuration_ready = not missing and configuration_error is None
    checks.append(
        {
            "id": "deployment_configuration",
            "status": "pass" if configuration_ready else "block",
            "evidence": (
                "deployment environment and Compose configuration are valid"
                if configuration_ready
                else (
                    f"missing deployment configuration: {', '.join(missing)}"
                    if missing
                    else f"Docker Compose configuration is invalid: {configuration_error}"
                )
            ),
        }
    )
    try:
        postgres_id = context.container_id("postgres")
        if not postgres_id:
            raise RuntimeError("PostgreSQL container is not running")
        context.run(
            "exec",
            "-T",
            "postgres",
            "pg_isready",
            "-U",
            "quantlab",
            "-d",
            "postgres",
        )
        postgres_evidence = f"PostgreSQL is ready in container {postgres_id}"
        postgres_status = "pass"
    except Exception as exc:
        postgres_status = "block"
        postgres_evidence = f"PostgreSQL is unavailable: {type(exc).__name__}: {exc}"
    checks.append(
        {
            "id": "postgres_ready",
            "status": postgres_status,
            "evidence": postgres_evidence,
        }
    )
    try:
        data_source = context.data_volume()
        _require_backup_outside_data(
            data_source,
            backup_root.resolve(),
            require_sibling=False,
        )
        location_status = "pass"
        location_evidence = (
            f"backup target {backup_root.resolve()} is outside governed data {data_source}"
        )
    except Exception as exc:
        location_status = "block"
        location_evidence = (
            f"backup target or governed data mount is invalid: {type(exc).__name__}: {exc}"
        )
    checks.append(
        {
            "id": "backup_location",
            "status": location_status,
            "evidence": location_evidence,
        }
    )
    capacity = assess_control_plane_backup_capacity(
        context,
        backup_root,
        minimum_free_gb=minimum_free_gb,
    )
    checks.append(capacity)
    return {
        "status": (
            "ready" if all(item["status"] == "pass" for item in checks) else "blocked"
        ),
        "backup_format_version": CONTROL_PLANE_BACKUP_FORMAT_VERSION,
        "business_readiness_consulted": False,
        "immutable_data_copied": False,
        "checks": checks,
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def load_and_verify_manifest(
    backup_directory: Path,
    *,
    use_verification_receipt: bool = False,
) -> dict[str, Any]:
    """Load a v1 full backup or v2 control-plane backup and verify its artifacts.

    The optional receipt is only an optimization: it is accepted solely for a
    root-owned, non-group-writable POSIX backup whose manifest hash and archive
    stat identities are unchanged. Every other case performs full SHA256 reads.
    """
    supplied_root = Path(backup_directory)
    root = supplied_root.resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("backup manifest is missing or invalid") from exc
    if not isinstance(manifest, dict):
        raise ValueError("backup manifest is missing or invalid")
    format_version = manifest.get("format_version")
    if format_version not in {
        FULL_BACKUP_FORMAT_VERSION,
        CONTROL_PLANE_BACKUP_FORMAT_VERSION,
    }:
        raise ValueError("unsupported backup format")
    key_fingerprint = manifest.get("platform_secret_key_sha256")
    if key_fingerprint is not None and (
        not isinstance(key_fingerprint, str)
        or len(key_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in key_fingerprint.lower())
    ):
        raise ValueError("backup platform secret key fingerprint is invalid")
    if format_version == FULL_BACKUP_FORMAT_VERSION:
        sections = ("database", "data_volume")
    else:
        if manifest.get("backup_scope") != "control_plane":
            raise ValueError("control-plane backup scope is invalid")
        if manifest.get("immutable_data_copied") is not False:
            raise ValueError("control-plane backup must declare immutable data was not copied")
        if manifest.get("immutable_data_restore") != "preserve_existing_data_volume":
            raise ValueError("control-plane immutable-data restore contract is invalid")
        if "data_volume" in manifest:
            raise ValueError("control-plane backup must not contain a data-volume archive")
        immutable = manifest.get("immutable_data") or {}
        if (
            not isinstance(immutable, dict)
            or immutable.get("copied") is not False
            or immutable.get("inventory_member") != CONTROL_PLANE_INVENTORY_MEMBER
            or not _valid_sha256(immutable.get("inventory_sha256"))
            or immutable.get("restore_action") != "preserve_existing_data_volume"
        ):
            raise ValueError("control-plane immutable-data inventory contract is invalid")
        sections = ("database", "control_plane")
    archives: dict[str, tuple[Path, str, int | None]] = {}
    archive_identities: dict[str, dict[str, int]] = {}
    for section in sections:
        entry = manifest.get(section) or {}
        candidate = (root / str(entry.get("file", ""))).resolve()
        if not _inside(candidate, root) or not candidate.is_file():
            raise ValueError(f"backup {section} file is missing or outside the backup directory")
        expected = str(entry.get("sha256", "")).lower()
        if not _valid_sha256(expected):
            raise ValueError(f"backup {section} checksum mismatch")
        expected_bytes = entry.get("bytes")
        try:
            parsed_expected_bytes = (
                int(expected_bytes) if expected_bytes is not None else None
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"backup {section} size mismatch") from exc
        observed_identity = _stat_identity(candidate.stat())
        archives[section] = (candidate, expected, parsed_expected_bytes)
        archive_identities[section] = observed_identity

    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    def verification_key() -> dict[str, Any]:
        return {
            "manifest_sha256": manifest_sha256,
            "sections": {
                section: {
                    "file": str((manifest.get(section) or {}).get("file", "")),
                    "sha256": expected,
                    "bytes": expected_bytes,
                    "stat": archive_identities[section],
                }
                for section, (_candidate, expected, expected_bytes) in archives.items()
            },
        }

    cache_is_safe = (
        use_verification_receipt
        and not supplied_root.is_symlink()
        and all(
            expected_bytes is not None
            for _candidate, _expected, expected_bytes in archives.values()
        )
        and _verification_receipt_environment_is_trusted(
            root,
            manifest_path,
            tuple(candidate for candidate, _expected, _bytes in archives.values()),
        )
    )
    current_key = verification_key()
    if cache_is_safe:
        receipt = _load_verification_receipt(root)
        if (
            receipt is not None
            and receipt.get("format_version") == _VERIFICATION_RECEIPT_VERSION
            and receipt.get("verification_key") == current_key
        ):
            return manifest

    for section, (candidate, expected, expected_bytes) in archives.items():
        actual, stable_identity = _sha256_with_stable_identity(candidate)
        if actual != expected:
            raise ValueError(f"backup {section} checksum mismatch")
        if expected_bytes is not None and stable_identity["size"] != expected_bytes:
            raise ValueError(f"backup {section} size mismatch")
        archive_identities[section] = stable_identity

    if format_version == CONTROL_PLANE_BACKUP_FORMAT_VERSION:
        _validate_control_plane_archive(archives["control_plane"][0], manifest)

    if cache_is_safe:
        _write_verification_receipt(root, verification_key())
    return manifest


@control_plane_locked
def create_backup(
    context: ComposeContext,
    backup_root: Path,
    *,
    retention_count: int = 14,
    restart_services: bool = True,
    format_version: int = FULL_BACKUP_FORMAT_VERSION,
    minimum_free_gb: float = 0.0,
    online: bool = False,
    pre_dump_guard: Callable[[], None] | None = None,
) -> Path:
    if retention_count < 1:
        raise ValueError("retention_count must be positive")
    if format_version not in {
        FULL_BACKUP_FORMAT_VERSION,
        CONTROL_PLANE_BACKUP_FORMAT_VERSION,
    }:
        raise ValueError("unsupported backup format")
    if minimum_free_gb < 0:
        raise ValueError("minimum_free_gb must not be negative")
    if online and format_version != CONTROL_PLANE_BACKUP_FORMAT_VERSION:
        raise ValueError("online backup is only supported for control-plane v2")
    if online and pre_dump_guard is not None:
        raise ValueError("a pre-dump quiescence guard requires a coordinated backup")
    key_fingerprint = _platform_secret_key_fingerprint(context)
    root = backup_root.resolve()
    name = f"quantlab-{_utc_stamp()}"
    staging = root / f".{name}.tmp"
    final = root / name
    data_volume = context.data_volume()
    _require_backup_outside_data(
        data_volume,
        root,
        require_sibling=format_version == FULL_BACKUP_FORMAT_VERSION,
    )
    root.mkdir(parents=True, exist_ok=True)
    if staging.exists() or final.exists():
        raise FileExistsError(f"backup destination already exists: {name}")

    stopped: list[str] = []
    if not online:
        running = context.running_services()
        stopped = [service for service in WRITER_SERVICES if service in running]
    postgres_id = context.container_id("postgres")
    if not postgres_id:
        raise RuntimeError("the PostgreSQL service must be running before backup")
    if format_version == CONTROL_PLANE_BACKUP_FORMAT_VERSION and minimum_free_gb:
        capacity = assess_control_plane_backup_capacity(
            context,
            root,
            minimum_free_gb=minimum_free_gb,
        )
        if capacity["status"] != "pass":
            raise RuntimeError(str(capacity["evidence"]))
    dump_in_container = f"/tmp/{name}.dump"
    backup_completed = False
    staging.mkdir(mode=0o700)
    staging.chmod(0o700)
    try:
        if stopped:
            context.run("stop", *stopped)
        if pre_dump_guard is not None:
            pre_dump_guard()
        data_uncompressed_bytes = (
            _volume_usage_bytes(context, data_volume)
            if format_version == FULL_BACKUP_FORMAT_VERSION
            else None
        )
        revision = context.run(
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "quantlab",
            "-d",
            "quantlab",
            "-Atc",
            "SELECT version_num FROM quantlab.alembic_version;",
            capture=True,
        ).splitlines()[0]
        context.run(
            "exec",
            "-T",
            "postgres",
            "pg_dump",
            "-U",
            "quantlab",
            "-d",
            "quantlab",
            "--format=custom",
            "--no-owner",
            "--no-privileges",
            f"--file={dump_in_container}",
        )
        database_file = staging / "quantlab-postgres.dump"
        context.docker("cp", f"{postgres_id}:{dump_in_container}", str(database_file))
        database_file.chmod(0o600)
        context.run("exec", "-T", "postgres", "rm", "-f", dump_in_container)
        common_manifest = {
            "format_version": format_version,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "project_name": context.project_name,
            "schema_revision": revision.strip(),
            "platform_secret_key_sha256": key_fingerprint,
            "database": {
                "file": database_file.name,
                "sha256": _sha256(database_file),
                "bytes": database_file.stat().st_size,
            },
        }
        if format_version == FULL_BACKUP_FORMAT_VERSION:
            data_file = staging / "quantlab-data.tar.gz"
            context.docker(
                "run",
                "--rm",
                "--volume",
                f"{data_volume}:/source:ro",
                "--volume",
                f"{staging}:/backup",
                "postgres:16-alpine",
                "tar",
                "-C",
                "/source",
                "-czf",
                "/backup/quantlab-data.tar.gz",
                ".",
            )
            data_file.chmod(0o600)
            manifest = {
                **common_manifest,
                "data_volume": {
                    "file": data_file.name,
                    "sha256": _sha256(data_file),
                    "bytes": data_file.stat().st_size,
                    "uncompressed_bytes": data_uncompressed_bytes,
                },
            }
        else:
            inventory = _capture_manifest_inventory(context)
            control_plane_file, control_snapshot = _write_control_plane_archive(
                context,
                staging,
                inventory,
            )
            inventory_bytes = (
                json.dumps(
                    inventory,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            manifest = {
                **common_manifest,
                "backup_scope": "control_plane",
                "immutable_data_copied": False,
                "immutable_data_restore": "preserve_existing_data_volume",
                "control_plane": {
                    "file": control_plane_file.name,
                    "sha256": _sha256(control_plane_file),
                    "bytes": control_plane_file.stat().st_size,
                },
                "immutable_data": {
                    "copied": False,
                    "inventory_member": CONTROL_PLANE_INVENTORY_MEMBER,
                    "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
                    "inventory_entries": inventory["entry_count"],
                    "inventory_truncated": inventory["truncated"],
                    "restore_action": "preserve_existing_data_volume",
                },
                "deployment_snapshot": {
                    "sanitized": True,
                    "compose_files": len(control_snapshot["compose_files"]),
                },
            }
        manifest_file = staging / "manifest.json"
        manifest_file.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest_file.chmod(0o600)
        context.docker(
            "run",
            "--rm",
            "--volume",
            f"{staging}:/backup:ro",
            "postgres:16-alpine",
            "pg_restore",
            "--list",
            "/backup/quantlab-postgres.dump",
            capture=True,
        )
        if format_version == FULL_BACKUP_FORMAT_VERSION:
            context.docker(
                "run",
                "--rm",
                "--volume",
                f"{staging}:/backup:ro",
                "postgres:16-alpine",
                "tar",
                "-tzf",
                "/backup/quantlab-data.tar.gz",
                capture=True,
            )
        else:
            _validate_control_plane_archive(control_plane_file, manifest)
        load_and_verify_manifest(staging, use_verification_receipt=True)
        staging.replace(final)
        completed_backups: list[Path] = [final]
        for candidate in sorted(root.glob("quantlab-*"), reverse=True):
            if candidate == final:
                continue
            if not candidate.is_dir() or not _inside(candidate, root):
                continue
            try:
                existing_manifest = load_and_verify_manifest(
                    candidate,
                    use_verification_receipt=True,
                )
            except (OSError, ValueError):
                continue
            if existing_manifest.get("format_version") != format_version:
                continue
            completed_backups.append(candidate)
        for old in completed_backups[retention_count:]:
            shutil.rmtree(old)
        backup_completed = True
        return final
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        context.run(
            "exec",
            "-T",
            "postgres",
            "rm",
            "-f",
            dump_in_container,
            check=False,
        )
        if stopped and (restart_services or not backup_completed):
            context.run("start", *stopped)


@control_plane_locked
def restore_backup(
    context: ComposeContext,
    backup_directory: Path,
    *,
    confirmed: bool,
    minimum_free_gb: float = 1.0,
    use_verification_receipt: bool = False,
) -> str:
    if not confirmed:
        raise ValueError(
            "restore replaces PostgreSQL and, for full v1 backups, /data; "
            "explicit confirmation is required"
        )
    if minimum_free_gb < 0:
        raise ValueError("minimum_free_gb must not be negative")
    root = backup_directory.resolve()
    manifest = load_and_verify_manifest(
        root,
        use_verification_receipt=use_verification_receipt,
    )
    expected_key_fingerprint = manifest.get("platform_secret_key_sha256")
    if expected_key_fingerprint is not None:
        actual_key_fingerprint = _platform_secret_key_fingerprint(context)
        if actual_key_fingerprint != expected_key_fingerprint:
            raise ValueError(
                "target PLATFORM_SECRET_KEY does not match the backup; restore was not started"
            )
    database_file = root / manifest["database"]["file"]
    format_version = int(manifest["format_version"])
    data_volume: str | None = None
    data_target: Path | None = None
    marker: Path | None = None
    marker_payload: dict[str, Any] | None = None
    if format_version == FULL_BACKUP_FORMAT_VERSION:
        data_volume = context.data_volume()
        data_target = _restore_target(context, data_volume, root)
    if not context.container_id("postgres"):
        raise RuntimeError("the target PostgreSQL service must be running")
    context.run(
        "exec",
        "-T",
        "postgres",
        "pg_isready",
        "-U",
        "quantlab",
        "-d",
        "postgres",
    )

    context.docker(
        "run",
        "--rm",
        "--volume",
        f"{root}:/backup:ro",
        "postgres:16-alpine",
        "pg_restore",
        "--list",
        f"/backup/{manifest['database']['file']}",
        capture=True,
    )
    if format_version == FULL_BACKUP_FORMAT_VERSION:
        assert data_volume is not None and data_target is not None
        archive_listing = context.docker(
            "run",
            "--rm",
            "--volume",
            f"{root}:/backup:ro",
            "postgres:16-alpine",
            "tar",
            "-tzf",
            f"/backup/{manifest['data_volume']['file']}",
            capture=True,
        )
        _validate_archive_listing(archive_listing)
        try:
            restored_bytes = int(manifest["data_volume"]["uncompressed_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "backup lacks the uncompressed data size required for safe direct restore"
            ) from exc
        if restored_bytes < 1:
            raise ValueError("backup uncompressed data size must be positive")
        current_bytes = _volume_usage_bytes(context, data_volume)
        available_after_clear = shutil.disk_usage(data_target).free + current_bytes
        required_bytes = restored_bytes + int(minimum_free_gb * _GIB)
        if available_after_clear < required_bytes:
            raise RuntimeError(
                "direct restore capacity is insufficient after reclaiming the current target: "
                f"available {available_after_clear / _GIB:.1f} GiB, "
                f"required {required_bytes / _GIB:.1f} GiB"
            )

    running = context.running_services()
    stopped = [service for service in WRITER_SERVICES if service in running]
    if format_version == FULL_BACKUP_FORMAT_VERSION:
        assert data_target is not None
        marker = data_target.parent / f".{data_target.name}.restore-in-progress.json"
        marker_payload = _begin_restore_marker(
            marker,
            target=data_target,
            backup_directory=root,
            data_sha256=str(manifest["data_volume"]["sha256"]),
        )
    dump_in_container = "/tmp/quantlab-restore.dump"
    try:
        if stopped:
            context.run("stop", *stopped)
        postgres_id = context.container_id("postgres")
        context.docker("cp", str(database_file), f"{postgres_id}:{dump_in_container}")
        context.run(
            "exec",
            "-T",
            "postgres",
            "dropdb",
            "--if-exists",
            "--force",
            "--maintenance-db=postgres",
            "-U",
            "quantlab",
            "quantlab",
        )
        context.run(
            "exec",
            "-T",
            "postgres",
            "createdb",
            "--maintenance-db=postgres",
            "--owner=quantlab",
            "-U",
            "quantlab",
            "quantlab",
        )
        context.run(
            "exec",
            "-T",
            "postgres",
            "pg_restore",
            "--exit-on-error",
            "--no-owner",
            "--no-privileges",
            "-U",
            "quantlab",
            "-d",
            "quantlab",
            dump_in_container,
        )
        if format_version == FULL_BACKUP_FORMAT_VERSION:
            assert data_target is not None
            context.docker(
                "run",
                "--rm",
                "--volume",
                f"{root}:/backup:ro",
                "--volume",
                f"{data_target}:/target",
                "postgres:16-alpine",
                "sh",
                "-euc",
                (
                    "find /target -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +; "
                    'tar -C /target -xzf "/backup/$1"'
                ),
                "quantlab-direct-restore",
                str(manifest["data_volume"]["file"]),
            )
        context.run("run", "--rm", "--no-deps", "api", "quant-db", "upgrade")
        revision = context.run(
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "quantlab",
            "-d",
            "quantlab",
            "-Atc",
            "SELECT version_num FROM quantlab.alembic_version;",
            capture=True,
        ).splitlines()[0]
    except Exception as exc:
        if marker is not None and marker_payload is not None:
            _atomic_marker(
                marker,
                {
                    **marker_payload,
                    "state": "failed",
                    "error": type(exc).__name__,
                    "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
            )
        raise
    finally:
        context.run(
            "exec",
            "-T",
            "postgres",
            "rm",
            "-f",
            dump_in_container,
            check=False,
        )
    try:
        if stopped:
            context.run("start", *stopped)
    except Exception as exc:
        if marker is not None and marker_payload is not None:
            _atomic_marker(
                marker,
                {
                    **marker_payload,
                    "state": "failed",
                    "error": type(exc).__name__,
                    "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
            )
        raise
    if marker is not None:
        marker.unlink()
    return revision.strip()


def compose_context(
    project_name: str,
    env_file: Path,
    compose_file: Path,
    extra_compose_files: Sequence[Path] = (),
    profiles: Sequence[str] = (),
) -> ComposeContext:
    if not project_name.strip():
        raise ValueError("project_name is required")
    files = (compose_file.resolve(), *(item.resolve() for item in extra_compose_files))
    for path in (env_file.resolve(), *files):
        if not path.is_file():
            raise FileNotFoundError(path)
    normalized_profiles = tuple(dict.fromkeys(item.strip() for item in profiles if item.strip()))
    return ComposeContext(
        project_name.strip(),
        env_file.resolve(),
        tuple(files),
        normalized_profiles,
    )
