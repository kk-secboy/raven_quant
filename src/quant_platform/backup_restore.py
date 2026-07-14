from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from dotenv import dotenv_values

WRITER_SERVICES = (
    "gateway",
    "scheduler",
    "worker",
    "rdagent-worker",
    "rdagent-docker",
    "api",
)
BACKUP_FILES = ("manifest.json", "quantlab-postgres.dump", "quantlab-data.tar.gz")


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


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
        for compose_file in self.compose_files:
            command.extend(("-f", str(compose_file.resolve())))
        return command

    def run(
        self,
        *arguments: str,
        capture: bool = False,
        check: bool = True,
        timeout: int | None = None,
    ) -> str:
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
        for mount in inspection.get("Mounts", []):
            if mount.get("Destination") == "/data" and mount.get("Name"):
                return str(mount["Name"])
        raise RuntimeError("unable to resolve the /data Docker volume")


def load_and_verify_manifest(backup_directory: Path) -> dict[str, Any]:
    root = backup_directory.resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("backup manifest is missing or invalid") from exc
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported backup format")
    key_fingerprint = manifest.get("platform_secret_key_sha256")
    if key_fingerprint is not None and (
        not isinstance(key_fingerprint, str)
        or len(key_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in key_fingerprint.lower())
    ):
        raise ValueError("backup platform secret key fingerprint is invalid")
    for section in ("database", "data_volume"):
        entry = manifest.get(section) or {}
        candidate = (root / str(entry.get("file", ""))).resolve()
        if not _inside(candidate, root) or not candidate.is_file():
            raise ValueError(f"backup {section} file is missing or outside the backup directory")
        expected = str(entry.get("sha256", "")).lower()
        if len(expected) != 64 or _sha256(candidate) != expected:
            raise ValueError(f"backup {section} checksum mismatch")
        expected_bytes = entry.get("bytes")
        if expected_bytes is not None and candidate.stat().st_size != int(expected_bytes):
            raise ValueError(f"backup {section} size mismatch")
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
    root.mkdir(parents=True, exist_ok=True)
    name = f"quantlab-{_utc_stamp()}"
    staging = root / f".{name}.tmp"
    final = root / name
    if staging.exists() or final.exists():
        raise FileExistsError(f"backup destination already exists: {name}")

    running = context.running_services()
    stopped = [service for service in WRITER_SERVICES if service in running]
    postgres_id = context.container_id("postgres")
    if not postgres_id:
        raise RuntimeError("the PostgreSQL service must be running before backup")
    data_volume = context.data_volume()
    dump_in_container = f"/tmp/{name}.dump"
    backup_completed = False
    staging.mkdir()
    try:
        if stopped:
            context.run("stop", *stopped)
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
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
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
        load_and_verify_manifest(staging)
        staging.replace(final)
        completed_backups = sorted(root.glob("quantlab-*"), reverse=True)
        for old in completed_backups[retention_count:]:
            if old.is_dir() and _inside(old, root):
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
) -> str:
    if not confirmed:
        raise ValueError("restore replaces PostgreSQL and /data; explicit confirmation is required")
    root = backup_directory.resolve()
    manifest = load_and_verify_manifest(root)
    expected_key_fingerprint = manifest.get("platform_secret_key_sha256")
    if expected_key_fingerprint is not None:
        actual_key_fingerprint = _platform_secret_key_fingerprint(context)
        if actual_key_fingerprint != expected_key_fingerprint:
            raise ValueError(
                "target PLATFORM_SECRET_KEY does not match the backup; restore was not started"
            )
    database_file = root / manifest["database"]["file"]
    data_volume = context.data_volume()
    if not context.container_id("postgres"):
        raise RuntimeError("the target PostgreSQL service must be running")

    running = context.running_services()
    stopped = [service for service in WRITER_SERVICES if service in running]
    stage_volume = f"{context.project_name.replace('-', '_')}_restore_stage_{uuid.uuid4().hex[:10]}"
    dump_in_container = "/tmp/quantlab-restore.dump"
    succeeded = False
    context.docker("volume", "create", stage_volume, capture=True)
    try:
        context.docker(
            "run",
            "--rm",
            "--volume",
            f"{root}:/backup:ro",
            "--volume",
            f"{stage_volume}:/staged",
            "postgres:16-alpine",
            "tar",
            "-C",
            "/staged",
            "-xzf",
            f"/backup/{manifest['data_volume']['file']}",
        )
        if stopped:
            context.run("stop", *stopped)
        postgres_id = context.container_id("postgres")
        context.docker("cp", str(database_file), f"{postgres_id}:{dump_in_container}")
        context.run(
            "exec",
            "-T",
            "postgres",
            "pg_restore",
            "--exit-on-error",
            "--clean",
            "--if-exists",
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
            f"{data_volume}:/target",
            "postgres:16-alpine",
            "sh",
            "-euc",
            "find /target -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +",
        )
        context.docker(
            "run",
            "--rm",
            "--volume",
            f"{stage_volume}:/staged:ro",
            "--volume",
            f"{data_volume}:/target",
            "postgres:16-alpine",
            "sh",
            "-euc",
            "cp -a /staged/. /target/",
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
        succeeded = True
        return revision.strip()
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
        context.docker("volume", "rm", "-f", stage_volume, check=False, capture=True)
        if succeeded and stopped:
            context.run("start", *stopped)


def compose_context(
    project_name: str,
    env_file: Path,
    compose_file: Path,
    extra_compose_files: Sequence[Path] = (),
) -> ComposeContext:
    if not project_name.strip():
        raise ValueError("project_name is required")
    files = (compose_file.resolve(), *(item.resolve() for item in extra_compose_files))
    for path in (env_file.resolve(), *files):
        if not path.is_file():
            raise FileNotFoundError(path)
    return ComposeContext(project_name.strip(), env_file.resolve(), tuple(files))
