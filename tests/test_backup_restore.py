from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from quant_platform.backup_restore import create_backup, load_and_verify_manifest, restore_backup


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _backup(tmp_path: Path) -> Path:
    root = tmp_path / "quantlab-20260713T000000Z"
    root.mkdir()
    database = root / "quantlab-postgres.dump"
    data = root / "quantlab-data.tar.gz"
    database.write_bytes(b"database-dump")
    data.write_bytes(b"data-archive")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "schema_revision": "0019_data_task_center",
                "database": {
                    "file": database.name,
                    "sha256": _digest(database),
                    "bytes": database.stat().st_size,
                },
                "data_volume": {
                    "file": data.name,
                    "sha256": _digest(data),
                    "bytes": data.stat().st_size,
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def test_backup_manifest_verifies_both_archives(tmp_path: Path) -> None:
    root = _backup(tmp_path)

    manifest = load_and_verify_manifest(root)

    assert manifest["schema_revision"] == "0019_data_task_center"


def test_backup_manifest_rejects_tampering(tmp_path: Path) -> None:
    root = _backup(tmp_path)
    (root / "quantlab-data.tar.gz").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_and_verify_manifest(root)


def test_backup_manifest_rejects_path_escape(tmp_path: Path) -> None:
    root = _backup(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["database"]["file"] = "../outside.dump"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the backup directory"):
        load_and_verify_manifest(root)


def test_backup_manifest_rejects_invalid_platform_key_fingerprint(tmp_path: Path) -> None:
    root = _backup(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["platform_secret_key_sha256"] = "not-a-fingerprint"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint is invalid"):
        load_and_verify_manifest(root)


class FakeBackupContext:
    project_name = "quantlab-test"

    def __init__(self, tmp_path: Path, *, fail_copy: bool = False) -> None:
        self.fail_copy = fail_copy
        self.calls: list[tuple[str, ...]] = []
        self.env_file = tmp_path / "compose.env"
        self.env_file.write_text(
            f"PLATFORM_SECRET_KEY={Fernet.generate_key().decode('ascii')}\n",
            encoding="utf-8",
        )

    def running_services(self) -> list[str]:
        return ["postgres", "api", "scheduler"]

    def container_id(self, service: str) -> str:
        return "postgres-id" if service == "postgres" else ""

    def data_volume(self) -> str:
        return "quantlab_data"

    def run(self, *args: str, **_kwargs) -> str:
        self.calls.append(args)
        if "SELECT version_num" in " ".join(args):
            return "0019_data_task_center"
        return ""

    def docker(self, *args: str, **_kwargs) -> str:
        self.calls.append(("docker", *args))
        if args[0] == "cp":
            if self.fail_copy:
                raise RuntimeError("copy failed")
            Path(args[2]).write_bytes(b"database-dump")
        elif args[0] == "run" and "-czf" in args:
            mount = next(item for item in args if item.endswith(":/backup"))
            Path(mount.removesuffix(":/backup"), "quantlab-data.tar.gz").write_bytes(
                b"data-archive"
            )
        return "ok"


def test_successful_backup_can_leave_writers_stopped_for_upgrade(tmp_path: Path) -> None:
    context = FakeBackupContext(tmp_path)

    backup = create_backup(
        context,  # type: ignore[arg-type]
        tmp_path,
        retention_count=1,
        restart_services=False,
    )

    assert backup.is_dir()
    manifest = load_and_verify_manifest(backup)
    assert len(manifest["platform_secret_key_sha256"]) == 64
    assert ("stop", "scheduler", "api") in context.calls
    assert not any(call[0] == "start" for call in context.calls)


def test_failed_backup_restarts_writers_even_in_upgrade_mode(tmp_path: Path) -> None:
    context = FakeBackupContext(tmp_path, fail_copy=True)

    with pytest.raises(RuntimeError, match="copy failed"):
        create_backup(
            context,  # type: ignore[arg-type]
            tmp_path,
            retention_count=1,
            restart_services=False,
        )

    assert ("start", "scheduler", "api") in context.calls


class FakeRestoreContext:
    project_name = "quantlab-restore-test"

    def __init__(self, env_file: Path) -> None:
        self.env_file = env_file
        self.destructive_call_made = False

    def data_volume(self) -> str:
        self.destructive_call_made = True
        raise AssertionError("restore progressed past the platform-key guard")


def test_restore_rejects_wrong_platform_key_before_any_destructive_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PLATFORM_SECRET_KEY", raising=False)
    root = _backup(tmp_path)
    source_key = Fernet.generate_key().decode("ascii")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["platform_secret_key_sha256"] = hashlib.sha256(
        source_key.encode("ascii")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target_env = tmp_path / "target.env"
    target_env.write_text(
        f"PLATFORM_SECRET_KEY={Fernet.generate_key().decode('ascii')}\n",
        encoding="utf-8",
    )
    context = FakeRestoreContext(target_env)

    with pytest.raises(ValueError, match="does not match the backup"):
        restore_backup(  # type: ignore[arg-type]
            context,
            root,
            confirmed=True,
        )

    assert context.destructive_call_made is False


def test_restore_rejects_missing_platform_key_before_any_destructive_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PLATFORM_SECRET_KEY", raising=False)
    root = _backup(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["platform_secret_key_sha256"] = hashlib.sha256(
        Fernet.generate_key()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target_env = tmp_path / "target-missing.env"
    target_env.write_text("POSTGRES_PASSWORD=test\n", encoding="utf-8")
    context = FakeRestoreContext(target_env)

    with pytest.raises(ValueError, match="PLATFORM_SECRET_KEY is required"):
        restore_backup(  # type: ignore[arg-type]
            context,
            root,
            confirmed=True,
        )

    assert context.destructive_call_made is False


def test_old_backup_without_platform_key_fingerprint_remains_readable(tmp_path: Path) -> None:
    manifest = load_and_verify_manifest(_backup(tmp_path))
    assert "platform_secret_key_sha256" not in manifest


def test_new_backup_requires_platform_key_before_stopping_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PLATFORM_SECRET_KEY", raising=False)
    context = FakeBackupContext(tmp_path)
    context.env_file.write_text("POSTGRES_PASSWORD=test\n", encoding="utf-8")

    with pytest.raises(ValueError, match="PLATFORM_SECRET_KEY is required"):
        create_backup(context, tmp_path / "backups")  # type: ignore[arg-type]

    assert context.calls == []
