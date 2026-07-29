from __future__ import annotations

from pathlib import Path

import pytest

from quant_platform import release_upgrade

pytestmark = pytest.mark.no_database


class FakeContext:
    project_name = "quantlab-test"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def run(self, *args: str, **_kwargs) -> str:
        self.calls.append(args)
        return ""

    def docker(self, *args: str, **_kwargs) -> str:
        self.calls.append(("docker", *args))
        return ""


def _gate(status: str = "ready", migration_state: str = "upgrade_required") -> dict:
    return {"status": status, "migration_state": migration_state, "checks": []}


def test_release_upgrade_requires_explicit_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="confirm-upgrade"):
        release_upgrade.run_release_upgrade(
            FakeContext(),  # type: ignore[arg-type]
            tmp_path,
            tmp_path / "backups",
            confirmed=False,
        )


def test_release_upgrade_stops_before_build_when_preflight_blocks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = FakeContext()
    monkeypatch.setattr(
        release_upgrade, "assess_release", lambda *_args, **_kwargs: _gate("blocked")
    )

    result = release_upgrade.run_release_upgrade(
        context,  # type: ignore[arg-type]
        tmp_path,
        tmp_path / "backups",
        confirmed=True,
    )

    assert result["status"] == "blocked"
    assert context.calls == []


def test_release_upgrade_builds_backs_up_and_accepts_current_schema(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = FakeContext()
    gates = iter([_gate(), _gate(), _gate(migration_state="current")])
    backup = tmp_path / "backups" / "quantlab-test"
    monkeypatch.setattr(release_upgrade, "_stamp", lambda: "20260713T040000Z")
    monkeypatch.setattr(
        release_upgrade,
        "assess_release",
        lambda *_args, **_kwargs: next(gates),
    )
    monkeypatch.setattr(
        release_upgrade,
        "_capture_rollback_images",
        lambda *_args: {"api": "quantlab-rollback:test-api"},
    )
    monkeypatch.setattr(
        release_upgrade,
        "_assess_backup_capacity",
        lambda *_args, **_kwargs: {"status": "pass"},
    )

    def create(*_args, **kwargs) -> Path:
        assert kwargs["restart_services"] is False
        return backup

    monkeypatch.setattr(release_upgrade, "create_backup", create)

    result = release_upgrade.run_release_upgrade(
        context,  # type: ignore[arg-type]
        tmp_path,
        tmp_path / "backups",
        confirmed=True,
        wait_timeout=45,
    )

    assert result["status"] == "succeeded"
    assert result["backup_directory"] == str(backup)
    assert ("build", *release_upgrade.BUILT_SERVICES) in context.calls
    assert ("rm", "-s", "-f", "factor-sandbox-builder") in context.calls
    assert any(call[:3] == ("up", "-d", "--remove-orphans") for call in context.calls)


def test_release_upgrade_restores_backup_and_old_images_on_failed_acceptance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = FakeContext()
    gates = iter([_gate(), _gate(), _gate("blocked", "current")])
    backup = tmp_path / "backups" / "quantlab-test"
    tags = {"api": "quantlab-rollback:test-api"}
    monkeypatch.setattr(
        release_upgrade,
        "assess_release",
        lambda *_args, **_kwargs: next(gates),
    )
    monkeypatch.setattr(release_upgrade, "_capture_rollback_images", lambda *_args: tags)
    monkeypatch.setattr(
        release_upgrade,
        "_assess_backup_capacity",
        lambda *_args, **_kwargs: {"status": "pass"},
    )
    monkeypatch.setattr(release_upgrade, "create_backup", lambda *_args, **_kwargs: backup)
    rollbacks: list[Path] = []

    def rollback(_context, backup_directory, rollback_tags, **_kwargs):
        assert rollback_tags == tags
        rollbacks.append(backup_directory)
        return {"schema_revision": "0019", "images": tags}

    monkeypatch.setattr(release_upgrade, "_restore_previous_release", rollback)

    result = release_upgrade.run_release_upgrade(
        context,  # type: ignore[arg-type]
        tmp_path,
        tmp_path / "backups",
        confirmed=True,
        wait_timeout=45,
    )

    assert result["status"] == "rolled_back"
    assert rollbacks == [backup]
    assert "post-upgrade release acceptance" in result["error"]


def test_rollback_image_pruning_keeps_newest_release_sets() -> None:
    class Images(FakeContext):
        def docker(self, *args: str, **_kwargs) -> str:
            self.calls.append(("docker", *args))
            if args[:2] == ("image", "ls"):
                return "\n".join(
                    f"quantlab-rollback:{release}-{service}"
                    for release in (
                        "20260713t010000z",
                        "20260713t020000z",
                        "20260713t030000z",
                    )
                    for service in ("api", "web")
                )
            return ""

    context = Images()

    removed = release_upgrade._prune_rollback_images(
        context,  # type: ignore[arg-type]
        2,
    )

    assert removed == [
        "quantlab-rollback:20260713t010000z-api",
        "quantlab-rollback:20260713t010000z-web",
    ]
    assert ("docker", "image", "rm", "-f", *removed) in context.calls


def test_backup_capacity_requires_full_data_copy_plus_headroom(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class CapacityContext(FakeContext):
        @staticmethod
        def data_volume() -> str:
            return "/data/quantlab"

        def docker(self, *args: str, **_kwargs) -> str:
            self.calls.append(("docker", *args))
            return f"{100 * 1024**2}\t/source"

    disk_usage = type("Usage", (), {"free": 110 * 1024**3})()
    monkeypatch.setattr(release_upgrade.shutil, "disk_usage", lambda _path: disk_usage)
    backup_root = tmp_path / "backups"

    result = release_upgrade._assess_backup_capacity(
        CapacityContext(),  # type: ignore[arg-type]
        backup_root,
        minimum_free_gb=20.0,
    )

    assert result["status"] == "block"
    assert "data upper bound 100.0 GiB" in result["evidence"]
    assert "required 120.0 GiB" in result["evidence"]


def test_release_upgrade_blocks_before_image_capture_when_backup_target_is_small(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = FakeContext()
    monkeypatch.setattr(
        release_upgrade,
        "assess_release",
        lambda *_args, **_kwargs: _gate(),
    )
    monkeypatch.setattr(
        release_upgrade,
        "_assess_backup_capacity",
        lambda *_args, **_kwargs: {"status": "block", "evidence": "too small"},
    )

    result = release_upgrade.run_release_upgrade(
        context,  # type: ignore[arg-type]
        tmp_path,
        tmp_path / "backups",
        confirmed=True,
    )

    assert result["status"] == "blocked"
    assert result["checks"]["backup_capacity"]["evidence"] == "too small"
    assert context.calls == []
