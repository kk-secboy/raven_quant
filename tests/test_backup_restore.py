from __future__ import annotations

import hashlib
import json
import os
import stat
import tarfile
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from quant_platform import backup_restore as backup_restore_module
from quant_platform.backup_restore import (
    _data_mount_source,
    assess_control_plane_backup_readiness,
    create_backup,
    load_and_verify_manifest,
    restore_backup,
)

pytestmark = pytest.mark.no_database


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
                    "uncompressed_bytes": 1024,
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def test_data_mount_source_accepts_named_volume() -> None:
    assert (
        _data_mount_source(
            {
                "Mounts": [
                    {
                        "Type": "volume",
                        "Name": "quantlab_data",
                        "Destination": "/data",
                    }
                ]
            }
        )
        == "quantlab_data"
    )


def test_data_mount_source_accepts_absolute_bind_mount() -> None:
    assert (
        _data_mount_source(
            {
                "Mounts": [
                    {
                        "Type": "bind",
                        "Source": "/data/quantlab",
                        "Destination": "/data",
                    }
                ]
            }
        )
        == "/data/quantlab"
    )


def test_data_mount_source_rejects_relative_bind_mount() -> None:
    with pytest.raises(RuntimeError, match="must be an absolute path"):
        _data_mount_source(
            {
                "Mounts": [
                    {
                        "Type": "bind",
                        "Source": "data/quantlab",
                        "Destination": "/data",
                    }
                ]
            }
        )


def test_backup_manifest_verifies_both_archives(tmp_path: Path) -> None:
    root = _backup(tmp_path)

    manifest = load_and_verify_manifest(root)

    assert manifest["schema_revision"] == "0019_data_task_center"


def _permit_test_verification_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        backup_restore_module,
        "_verification_receipt_environment_is_trusted",
        lambda *_args, **_kwargs: True,
    )

    def trusted_file(value: os.stat_result, *, exact_mode: int | None = None) -> bool:
        mode = stat.S_IMODE(value.st_mode)
        return stat.S_ISREG(value.st_mode) and (
            exact_mode is None or mode == exact_mode
        )

    monkeypatch.setattr(
        backup_restore_module,
        "_root_owned_nonwritable_regular_file",
        trusted_file,
    )


def test_verification_receipt_skips_unchanged_archive_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("the production verification receipt is POSIX root-only")
    root = _backup(tmp_path)
    _permit_test_verification_receipts(monkeypatch)
    original = backup_restore_module._sha256_with_stable_identity
    hashed: list[str] = []

    def tracked(path: Path) -> tuple[str, dict[str, int]]:
        hashed.append(path.name)
        return original(path)

    monkeypatch.setattr(
        backup_restore_module,
        "_sha256_with_stable_identity",
        tracked,
    )

    load_and_verify_manifest(root, use_verification_receipt=True)

    assert hashed == ["quantlab-postgres.dump", "quantlab-data.tar.gz"]
    receipt_path = root / backup_restore_module._VERIFICATION_RECEIPT_NAME
    assert receipt_path.is_file()
    assert not receipt_path.is_symlink()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    key = receipt["verification_key"]
    assert key["manifest_sha256"] == hashlib.sha256(
        (root / "manifest.json").read_bytes()
    ).hexdigest()
    assert key["sections"]["data_volume"]["sha256"] == _digest(
        root / "quantlab-data.tar.gz"
    )
    assert key["sections"]["data_volume"]["bytes"] == (
        root / "quantlab-data.tar.gz"
    ).stat().st_size
    assert set(key["sections"]["data_volume"]["stat"]) == {
        "device",
        "inode",
        "size",
        "mtime_ns",
    }
    if os.name == "posix":
        assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600

    hashed.clear()
    load_and_verify_manifest(root, use_verification_receipt=True)
    assert hashed == []


def test_verification_receipt_falls_back_when_manifest_hash_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("the production verification receipt is POSIX root-only")
    root = _backup(tmp_path)
    _permit_test_verification_receipts(monkeypatch)
    load_and_verify_manifest(root, use_verification_receipt=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["created_at"] = "2026-08-22T12:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    original = backup_restore_module._sha256_with_stable_identity
    hashed: list[str] = []

    def tracked(path: Path) -> tuple[str, dict[str, int]]:
        hashed.append(path.name)
        return original(path)

    monkeypatch.setattr(
        backup_restore_module,
        "_sha256_with_stable_identity",
        tracked,
    )

    load_and_verify_manifest(root, use_verification_receipt=True)

    assert hashed == ["quantlab-postgres.dump", "quantlab-data.tar.gz"]


def test_verification_receipt_falls_back_when_archive_identity_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("the production verification receipt is POSIX root-only")
    root = _backup(tmp_path)
    _permit_test_verification_receipts(monkeypatch)
    load_and_verify_manifest(root, use_verification_receipt=True)
    data_file = root / "quantlab-data.tar.gz"
    previous = data_file.stat().st_mtime_ns
    data_file.write_bytes(b"tampered!!!!")
    os.utime(data_file, ns=(previous + 1_000_000_000, previous + 1_000_000_000))

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_and_verify_manifest(root, use_verification_receipt=True)


def test_verification_receipt_with_loose_permissions_is_not_trusted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX modes are required for this receipt security check")
    root = _backup(tmp_path)
    _permit_test_verification_receipts(monkeypatch)
    load_and_verify_manifest(root, use_verification_receipt=True)
    receipt_path = root / backup_restore_module._VERIFICATION_RECEIPT_NAME
    receipt_path.chmod(0o644)
    original = backup_restore_module._sha256_with_stable_identity
    hashed: list[str] = []

    def tracked(path: Path) -> tuple[str, dict[str, int]]:
        hashed.append(path.name)
        return original(path)

    monkeypatch.setattr(
        backup_restore_module,
        "_sha256_with_stable_identity",
        tracked,
    )

    load_and_verify_manifest(root, use_verification_receipt=True)

    assert hashed == ["quantlab-postgres.dump", "quantlab-data.tar.gz"]
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600


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
        self.compose_file = tmp_path / "compose.yaml"
        self.compose_file.write_text(
            "services:\n"
            "  api:\n"
            "    environment:\n"
            "      DATABASE_URL: postgresql://quantlab:compose-password@postgres/quantlab\n"
            "      PRIVATE_KEY: |\n"
            "        compose-private-key-body\n",
            encoding="utf-8",
        )
        self.compose_files = (self.compose_file,)
        self.profiles: tuple[str, ...] = ()

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
        if "SELECT pg_database_size" in " ".join(args):
            return str(64 * 1024)
        if "quantlab-immutable-data-inventory-v1" in " ".join(args):
            return json.dumps(
                {
                    "contract_version": "quantlab-immutable-data-inventory-v1",
                    "generated_at": "2026-08-29T00:00:00+00:00",
                    "root": "/data",
                    "manifest_name": "manifest.json",
                    "immutable_data_copied": False,
                    "max_entries": (
                        backup_restore_module.CONTROL_PLANE_MAX_INVENTORY_ENTRIES
                    ),
                    "entry_count": 1,
                    "truncated": False,
                    "entries": [
                        {
                            "path": "snapshots/daily/manifest.json",
                            "sha256": "a" * 64,
                            "bytes": 512,
                            "mtime_ns": 123456789,
                        }
                    ],
                }
            )
        return ""

    def docker(self, *args: str, **_kwargs) -> str:
        self.calls.append(("docker", *args))
        if "du" in args and "/source" in args:
            return "128\t/source"
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
        format_version=1,
    )

    assert backup.is_dir()
    manifest = load_and_verify_manifest(backup)
    assert len(manifest["platform_secret_key_sha256"]) == 64
    if os.name == "posix":
        assert stat.S_IMODE(backup.stat().st_mode) == 0o700
        for name in ("manifest.json", "quantlab-postgres.dump", "quantlab-data.tar.gz"):
            assert stat.S_IMODE((backup / name).stat().st_mode) == 0o600
    assert ("stop", "scheduler", "api") in context.calls
    assert not any(call[0] == "start" for call in context.calls)


def test_control_plane_backup_for_upgrade_remains_quiesced(tmp_path: Path) -> None:
    context = FakeBackupContext(tmp_path)

    backup = create_backup(
        context,  # type: ignore[arg-type]
        tmp_path / "backups",
        retention_count=1,
        restart_services=False,
        format_version=2,
    )

    assert backup.is_dir()
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
            format_version=1,
        )

    assert ("start", "scheduler", "api") in context.calls


def test_control_plane_backup_is_sanitized_bounded_and_does_not_copy_data(
    tmp_path: Path,
) -> None:
    context = FakeBackupContext(tmp_path)
    governed_data = tmp_path / "separate-data-disk" / "quantlab"
    governed_data.mkdir(parents=True)
    context.data_volume = lambda: str(governed_data.resolve())  # type: ignore[method-assign]
    secret_key = context.env_file.read_text(encoding="utf-8").split("=", 1)[1].strip()

    backup = create_backup(
        context,  # type: ignore[arg-type]
        tmp_path / "backups",
        retention_count=1,
        format_version=2,
        minimum_free_gb=0,
        online=True,
    )

    manifest = load_and_verify_manifest(backup)
    assert manifest["format_version"] == 2
    assert manifest["backup_scope"] == "control_plane"
    assert manifest["immutable_data_copied"] is False
    assert manifest["immutable_data_restore"] == "preserve_existing_data_volume"
    assert manifest["immutable_data"]["copied"] is False
    assert backup.parent != governed_data.parent
    assert "data_volume" not in manifest
    assert not (backup / "quantlab-data.tar.gz").exists()
    archive_path = backup / manifest["control_plane"]["file"]
    assert archive_path.stat().st_size <= (
        backup_restore_module.CONTROL_PLANE_MAX_ARCHIVE_BYTES
    )
    with tarfile.open(archive_path, mode="r:gz") as archive:
        environment = archive.extractfile(
            backup_restore_module.CONTROL_PLANE_ENV_MEMBER
        )
        compose = archive.extractfile("deployment/compose-01.yaml")
        inventory = archive.extractfile(
            backup_restore_module.CONTROL_PLANE_INVENTORY_MEMBER
        )
        assert environment is not None and compose is not None and inventory is not None
        environment_text = environment.read().decode("utf-8")
        compose_text = compose.read().decode("utf-8")
        inventory_payload = json.loads(inventory.read())
    assert secret_key not in environment_text
    assert "PLATFORM_SECRET_KEY=[REDACTED]" in environment_text
    assert "compose-password" not in compose_text
    assert "compose-private-key-body" not in compose_text
    assert "postgresql://[REDACTED]@postgres/quantlab" in compose_text
    assert inventory_payload["entries"][0]["path"] == (
        "snapshots/daily/manifest.json"
    )
    assert inventory_payload["immutable_data_copied"] is False
    assert not any("-czf" in call for call in context.calls)
    assert not any("du" in call and "/source" in call for call in context.calls)
    assert not any(call[0] in {"stop", "start"} for call in context.calls)


def test_failed_online_control_plane_backup_does_not_touch_writers(tmp_path: Path) -> None:
    context = FakeBackupContext(tmp_path, fail_copy=True)

    with pytest.raises(RuntimeError, match="copy failed"):
        create_backup(
            context,  # type: ignore[arg-type]
            tmp_path / "backups",
            retention_count=1,
            format_version=2,
            online=True,
        )

    assert not any(call[0] in {"stop", "start"} for call in context.calls)


def test_full_backup_rejects_online_mode_before_touching_services(tmp_path: Path) -> None:
    context = FakeBackupContext(tmp_path)

    with pytest.raises(ValueError, match="only supported for control-plane v2"):
        create_backup(
            context,  # type: ignore[arg-type]
            tmp_path / "backups",
            retention_count=1,
            format_version=1,
            online=True,
        )

    assert context.calls == []


def test_control_plane_backup_preflight_is_independent_of_business_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = FakeBackupContext(tmp_path)
    disk_usage = type("Usage", (), {"free": 12 * 1024**3})()
    monkeypatch.setattr(backup_restore_module.shutil, "disk_usage", lambda _path: disk_usage)

    result = assess_control_plane_backup_readiness(
        context,  # type: ignore[arg-type]
        tmp_path / "backups",
        minimum_free_gb=10,
    )

    assert result["status"] == "ready"
    assert result["business_readiness_consulted"] is False
    assert result["immutable_data_copied"] is False
    assert {item["id"] for item in result["checks"]} == {
        "deployment_configuration",
        "postgres_ready",
        "backup_location",
        "control_plane_backup_capacity",
    }


def test_control_plane_manifest_rejects_a_claim_that_data_was_copied(
    tmp_path: Path,
) -> None:
    context = FakeBackupContext(tmp_path)
    backup = create_backup(
        context,  # type: ignore[arg-type]
        tmp_path / "backups",
        retention_count=1,
        format_version=2,
    )
    manifest_path = backup / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["immutable_data_copied"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must declare immutable data was not copied"):
        load_and_verify_manifest(backup)


def test_backup_rejects_root_inside_or_equal_to_data_parent(tmp_path: Path) -> None:
    context = FakeBackupContext(tmp_path)
    data_target = tmp_path / "quantlab"
    data_target.mkdir()
    context.data_volume = lambda: str(data_target)  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="backup root"):
        create_backup(context, data_target / "backups")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dedicated sibling"):
        create_backup(context, data_target.parent)  # type: ignore[arg-type]

    assert not any("-czf" in call for call in context.calls)


class FakeRestoreContext:
    project_name = "quantlab-restore-test"

    def __init__(self, env_file: Path) -> None:
        self.env_file = env_file
        self.destructive_call_made = False

    def data_volume(self) -> str:
        self.destructive_call_made = True
        raise AssertionError("restore progressed past the platform-key guard")


class DirectRestoreContext:
    project_name = "quantlab-restore-test"

    def __init__(self, tmp_path: Path, target: Path, *, fail_extract: bool = False) -> None:
        self.target = target
        self.fail_extract = fail_extract
        self.calls: list[tuple[str, ...]] = []
        self.env_file = tmp_path / "restore.env"
        self.env_file.write_text(
            f"QUANTLAB_DATA_HOST_PATH={target}\n",
            encoding="utf-8",
        )

    def data_volume(self) -> str:
        return str(self.target)

    def container_id(self, service: str) -> str:
        return "postgres-id" if service == "postgres" else ""

    def running_services(self) -> list[str]:
        return ["postgres", "scheduler", "api"]

    def run(self, *args: str, **_kwargs) -> str:
        self.calls.append(args)
        if "SELECT version_num" in " ".join(args):
            return "0064_paper_stage_account"
        return ""

    def docker(self, *args: str, **_kwargs) -> str:
        self.calls.append(("docker", *args))
        if "-tzf" in args:
            return "./\n./features/value.bin"
        if "du" in args and "/source" in args:
            return "1\t/source"
        if self.fail_extract and any(
            "quantlab-direct-restore" == item for item in args
        ):
            raise RuntimeError("extract failed")
        return ""


class ControlPlaneRestoreContext:
    project_name = "quantlab-test"

    def __init__(self, env_file: Path) -> None:
        self.env_file = env_file
        self.calls: list[tuple[str, ...]] = []
        self.data_volume_called = False

    def data_volume(self) -> str:
        self.data_volume_called = True
        raise AssertionError("v2 restore must not resolve or mutate /data")

    def container_id(self, service: str) -> str:
        return "postgres-id" if service == "postgres" else ""

    def running_services(self) -> list[str]:
        return ["postgres", "scheduler", "api"]

    def run(self, *args: str, **_kwargs) -> str:
        self.calls.append(args)
        if "SELECT version_num" in " ".join(args):
            return "0072_three_horizon_strategy_research"
        return ""

    def docker(self, *args: str, **_kwargs) -> str:
        self.calls.append(("docker", *args))
        return ""


def test_restore_uses_direct_exact_bind_and_removes_marker_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QUANTLAB_DATA_HOST_PATH", raising=False)
    target = tmp_path / "quantlab"
    target.mkdir()
    backup = _backup(tmp_path)
    context = DirectRestoreContext(tmp_path, target)

    revision = restore_backup(
        context,  # type: ignore[arg-type]
        backup,
        confirmed=True,
        minimum_free_gb=0,
    )

    assert revision == "0064_paper_stage_account"
    assert not any(call[1:3] == ("volume", "create") for call in context.calls)
    direct = next(call for call in context.calls if "quantlab-direct-restore" in call)
    drop_database = next(index for index, call in enumerate(context.calls) if "dropdb" in call)
    create_database = next(
        index for index, call in enumerate(context.calls) if "createdb" in call
    )
    restore_database = next(
        index
        for index, call in enumerate(context.calls)
        if "pg_restore" in call and "--exit-on-error" in call
    )
    assert f"{backup.resolve()}:/backup:ro" in direct
    assert f"{target.resolve()}:/target" in direct
    assert "find /target -mindepth 1 -maxdepth 1" in " ".join(direct)
    assert drop_database < create_database < restore_database
    assert "--clean" not in context.calls[restore_database]
    assert "--if-exists" not in context.calls[restore_database]
    assert not (tmp_path / ".quantlab.restore-in-progress.json").exists()
    assert ("start", "scheduler", "api") in context.calls


def test_control_plane_restore_replaces_database_and_never_touches_data(
    tmp_path: Path,
) -> None:
    backup_context = FakeBackupContext(tmp_path)
    backup = create_backup(
        backup_context,  # type: ignore[arg-type]
        tmp_path / "backups",
        retention_count=1,
        format_version=2,
    )
    data_sentinel = tmp_path / "governed-data" / "sentinel.txt"
    data_sentinel.parent.mkdir()
    data_sentinel.write_text("preserve-me", encoding="utf-8")
    context = ControlPlaneRestoreContext(backup_context.env_file)

    revision = restore_backup(
        context,  # type: ignore[arg-type]
        backup,
        confirmed=True,
        minimum_free_gb=0,
    )

    assert revision == "0072_three_horizon_strategy_research"
    assert context.data_volume_called is False
    assert data_sentinel.read_text(encoding="utf-8") == "preserve-me"
    assert any("dropdb" in call for call in context.calls)
    assert any("createdb" in call for call in context.calls)
    assert any(
        "pg_restore" in call and "--exit-on-error" in call for call in context.calls
    )
    assert not any("quantlab-direct-restore" in call for call in context.calls)
    assert not any("/target" in item for call in context.calls for item in call)


def test_failed_direct_restore_keeps_backup_and_retriable_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QUANTLAB_DATA_HOST_PATH", raising=False)
    target = tmp_path / "quantlab"
    target.mkdir()
    backup = _backup(tmp_path)
    context = DirectRestoreContext(tmp_path, target, fail_extract=True)

    with pytest.raises(RuntimeError, match="extract failed"):
        restore_backup(
            context,  # type: ignore[arg-type]
            backup,
            confirmed=True,
            minimum_free_gb=0,
        )

    marker = tmp_path / ".quantlab.restore-in-progress.json"
    assert backup.is_dir()
    assert json.loads(marker.read_text(encoding="utf-8"))["state"] == "failed"
    assert not any(call[0] == "start" for call in context.calls)

    context.fail_extract = False
    assert restore_backup(
        context,  # type: ignore[arg-type]
        backup,
        confirmed=True,
        minimum_free_gb=0,
    ) == "0064_paper_stage_account"
    assert not marker.exists()


def test_restore_rejects_backup_inside_target_before_stopping_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QUANTLAB_DATA_HOST_PATH", raising=False)
    target = tmp_path / "quantlab"
    target.mkdir()
    backup = _backup(target)
    context = DirectRestoreContext(tmp_path, target)

    with pytest.raises(ValueError, match="must not contain"):
        restore_backup(  # type: ignore[arg-type]
            context,
            backup,
            confirmed=True,
            minimum_free_gb=0,
        )

    assert not any(call[0] == "stop" for call in context.calls)


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
