from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

from cryptography.fernet import Fernet
from dotenv import dotenv_values

WRITER_SERVICES = (
    "gateway",
    "scheduler",
    "worker",
    "rdagent-worker",
    "rdagent-data-science-worker",
    "rdagent-llm-finetune-worker",
    "rdagent-docker",
    "api",
)
BACKUP_FILES = ("manifest.json", "quantlab-postgres.dump", "quantlab-data.tar.gz")
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
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+\S+")
_URL_CREDENTIALS = re.compile(r"(?i)(https?://)[^\s/@:]+:[^\s/@]+@")


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


def _require_backup_outside_data(data_source: str, backup_path: Path) -> None:
    data_path = _absolute_host_path(data_source)
    if data_path is None:
        return
    resolved = backup_path.resolve()
    if resolved == data_path.parent or resolved.parent != data_path.parent:
        raise ValueError(
            "backup root must be a dedicated sibling directory of the governed data target"
        )
    if resolved == data_path or _inside(resolved, data_path):
        raise ValueError("backup root must be outside the governed /data bind target")


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


def load_and_verify_manifest(
    backup_directory: Path,
    *,
    use_verification_receipt: bool = False,
) -> dict[str, Any]:
    """Load a backup and verify both archives.

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
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported backup format")
    key_fingerprint = manifest.get("platform_secret_key_sha256")
    if key_fingerprint is not None and (
        not isinstance(key_fingerprint, str)
        or len(key_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in key_fingerprint.lower())
    ):
        raise ValueError("backup platform secret key fingerprint is invalid")
    archives: dict[str, tuple[Path, str, int | None]] = {}
    archive_identities: dict[str, dict[str, int]] = {}
    for section in ("database", "data_volume"):
        entry = manifest.get(section) or {}
        candidate = (root / str(entry.get("file", ""))).resolve()
        if not _inside(candidate, root) or not candidate.is_file():
            raise ValueError(f"backup {section} file is missing or outside the backup directory")
        expected = str(entry.get("sha256", "")).lower()
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
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

    if cache_is_safe:
        _write_verification_receipt(root, verification_key())
    return manifest


def create_backup(
    context: ComposeContext,
    backup_root: Path,
    *,
    retention_count: int = 14,
    restart_services: bool = True,
) -> Path:
    if retention_count < 1:
        raise ValueError("retention_count must be positive")
    key_fingerprint = _platform_secret_key_fingerprint(context)
    root = backup_root.resolve()
    name = f"quantlab-{_utc_stamp()}"
    staging = root / f".{name}.tmp"
    final = root / name
    data_volume = context.data_volume()
    _require_backup_outside_data(data_volume, root)
    root.mkdir(parents=True, exist_ok=True)
    if staging.exists() or final.exists():
        raise FileExistsError(f"backup destination already exists: {name}")

    running = context.running_services()
    stopped = [service for service in WRITER_SERVICES if service in running]
    postgres_id = context.container_id("postgres")
    if not postgres_id:
        raise RuntimeError("the PostgreSQL service must be running before backup")
    dump_in_container = f"/tmp/{name}.dump"
    backup_completed = False
    staging.mkdir(mode=0o700)
    staging.chmod(0o700)
    try:
        if stopped:
            context.run("stop", *stopped)
        data_uncompressed_bytes = _volume_usage_bytes(context, data_volume)
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
        data_file = staging / "quantlab-data.tar.gz"
        context.docker("cp", f"{postgres_id}:{dump_in_container}", str(database_file))
        database_file.chmod(0o600)
        context.run("exec", "-T", "postgres", "rm", "-f", dump_in_container)
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
            "format_version": 1,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "project_name": context.project_name,
            "schema_revision": revision.strip(),
            "platform_secret_key_sha256": key_fingerprint,
            "database": {
                "file": database_file.name,
                "sha256": _sha256(database_file),
                "bytes": database_file.stat().st_size,
            },
            "data_volume": {
                "file": data_file.name,
                "sha256": _sha256(data_file),
                "bytes": data_file.stat().st_size,
                "uncompressed_bytes": data_uncompressed_bytes,
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
        load_and_verify_manifest(staging, use_verification_receipt=True)
        staging.replace(final)
        completed_backups: list[Path] = [final]
        for candidate in sorted(root.glob("quantlab-*"), reverse=True):
            if candidate == final:
                continue
            if not candidate.is_dir() or not _inside(candidate, root):
                continue
            try:
                load_and_verify_manifest(candidate, use_verification_receipt=True)
            except (OSError, ValueError):
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


def restore_backup(
    context: ComposeContext,
    backup_directory: Path,
    *,
    confirmed: bool,
    minimum_free_gb: float = 1.0,
    use_verification_receipt: bool = False,
) -> str:
    if not confirmed:
        raise ValueError("restore replaces PostgreSQL and /data; explicit confirmation is required")
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
