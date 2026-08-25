from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quant_data.kaggle_assets import (
    KAGGLE_ALLOWLIST_SCHEMA_VERSION,
    KaggleCapabilityBlocked,
    acquire_kaggle_research_asset,
    load_kaggle_allowlist,
    manifest_contains_kaggle_credentials,
    validate_kaggle_research_asset_manifest,
)
from quant_data.research_assets import load_research_asset_manifest

pytestmark = pytest.mark.no_database


def _allowlist(path: Path, *, owner: str = "owner") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": KAGGLE_ALLOWLIST_SCHEMA_VERSION,
                "administrator_role": "data_science",
                "declared_by": "admin@example.invalid",
                "datasets": [f"{owner}/prices"],
                "competitions": ["market-challenge"],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_kaggle_allowlist_requires_data_science_role(tmp_path: Path) -> None:
    path = _allowlist(tmp_path / "allowlist.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["administrator_role"] = "researcher"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KaggleCapabilityBlocked, match="data_science"):
        load_kaggle_allowlist(path)


def test_kaggle_missing_cli_is_explicitly_blocked(tmp_path: Path) -> None:
    allowlist = _allowlist(tmp_path / "allowlist.json")
    terms = tmp_path / "terms.txt"
    terms.write_text("Apache-2.0 full terms evidence", encoding="utf-8")

    with pytest.raises(KaggleCapabilityBlocked) as exc_info:
        acquire_kaggle_research_asset(
            tmp_path / "data",
            allowlist_path=allowlist,
            source_kind="dataset",
            slug="owner/prices",
            source_version="7",
            license_name="Apache-2.0",
            license_terms_path=terms,
            license_accepted=True,
            license_accepted_by="admin@example.invalid",
            license_accepted_at=datetime(2026, 8, 1, tzinfo=UTC),
            environ={"KAGGLE_USERNAME": "owner", "KAGGLE_KEY": "secret-key"},
            executable_resolver=lambda _name: None,
            clock=lambda: datetime(2026, 8, 2, tzinfo=UTC),
        )

    assert exc_info.value.reason_code == "kaggle_cli_unavailable"


def test_kaggle_dataset_is_sealed_without_credentials(tmp_path: Path) -> None:
    allowlist = _allowlist(tmp_path / "allowlist.json", owner="owner")
    terms = tmp_path / "terms.txt"
    terms.write_text("Apache-2.0 full terms evidence", encoding="utf-8")

    def fake_runner(command: list[str], **kwargs: object) -> object:
        destination = Path(command[command.index("--path") + 1])
        with zipfile.ZipFile(destination / "prices.zip", "w") as archive:
            archive.writestr("train/data.csv", "date,value\n2026-01-01,1\n")

        class Result:
            returncode = 0

        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["KAGGLE_USERNAME"] == "owner"
        assert environment["KAGGLE_KEY"] == "secret-key"
        return Result()

    result = acquire_kaggle_research_asset(
        tmp_path / "data",
        allowlist_path=allowlist,
        source_kind="dataset",
        slug="owner/prices",
        source_version="7",
        license_name="Apache-2.0",
        license_terms_path=terms,
        license_accepted=True,
        license_accepted_by="admin@example.invalid",
        license_accepted_at=datetime(2026, 8, 1, tzinfo=UTC),
        environ={"KAGGLE_USERNAME": "owner", "KAGGLE_KEY": "secret-key"},
        executable_resolver=lambda _name: "kaggle",
        command_runner=fake_runner,  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 8, 2, tzinfo=UTC),
    )

    manifest = load_research_asset_manifest(result.published.manifest_path)
    validated = validate_kaggle_research_asset_manifest(
        manifest, result.published.directory
    )
    assert validated["slug"] == "owner/prices"
    assert validated["source_version"] == "7"
    assert not manifest_contains_kaggle_credentials(manifest, ["secret-key"])
    inventory = json.loads(
        (result.published.directory / ".quantlab/kaggle/inventory.json").read_text(
            encoding="utf-8"
        )
    )
    payload = result.published.directory / "train/data.csv"
    assert inventory["files"] == [
        {
            "path": "train/data.csv",
            "bytes": payload.stat().st_size,
            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        }
    ]


def test_kaggle_archive_rejects_parent_traversal(tmp_path: Path) -> None:
    allowlist = _allowlist(tmp_path / "allowlist.json")
    terms = tmp_path / "terms.txt"
    terms.write_text("license evidence", encoding="utf-8")

    def fake_runner(command: list[str], **_kwargs: object) -> object:
        destination = Path(command[command.index("--path") + 1])
        with zipfile.ZipFile(destination / "prices.zip", "w") as archive:
            archive.writestr("../escape.csv", "bad")

        class Result:
            returncode = 0

        return Result()

    with pytest.raises(ValueError, match="unsafe research asset path"):
        acquire_kaggle_research_asset(
            tmp_path / "data",
            allowlist_path=allowlist,
            source_kind="dataset",
            slug="owner/prices",
            source_version="1",
            license_name="custom",
            license_terms_path=terms,
            license_accepted=True,
            license_accepted_by="admin",
            license_accepted_at=datetime(2026, 8, 1, tzinfo=UTC),
            environ={"KAGGLE_API_TOKEN": "token-secret"},
            executable_resolver=lambda _name: "kaggle",
            command_runner=fake_runner,  # type: ignore[arg-type]
            clock=lambda: datetime(2026, 8, 2, tzinfo=UTC),
        )
