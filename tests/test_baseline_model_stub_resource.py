from __future__ import annotations

from pathlib import Path

import pytest

from quant_platform.baseline_model_stub_resource import (
    BASELINE_MODEL_STUB_SHA256,
    resolve_governed_baseline_model_stub,
)

pytestmark = pytest.mark.no_database


def _canonical_stub() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "baseline_model_stub.py"


def test_resolver_uses_verified_source_checkout_stub(tmp_path: Path) -> None:
    project_root = tmp_path / "checkout"
    package_root = tmp_path / "package"
    stub = project_root / "scripts" / "baseline_model_stub.py"
    stub.parent.mkdir(parents=True)
    stub.write_bytes(_canonical_stub().read_bytes())

    resolved = resolve_governed_baseline_model_stub(
        project_root,
        package_root=package_root,
    )

    assert resolved == stub.resolve()


def test_resolver_uses_verified_wheel_resource_without_checkout(tmp_path: Path) -> None:
    package_root = tmp_path / "site-packages" / "quant_platform"
    stub = package_root / "_artifacts" / "baseline_model_stub.py"
    stub.parent.mkdir(parents=True)
    stub.write_bytes(_canonical_stub().read_bytes())

    resolved = resolve_governed_baseline_model_stub(
        tmp_path / "missing-checkout",
        package_root=package_root,
    )

    assert resolved == stub.resolve()


def test_resolver_rejects_modified_governed_stub(tmp_path: Path) -> None:
    project_root = tmp_path / "checkout"
    stub = project_root / "scripts" / "baseline_model_stub.py"
    stub.parent.mkdir(parents=True)
    stub.write_text("model_cls = object\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 integrity"):
        resolve_governed_baseline_model_stub(
            project_root,
            package_root=tmp_path / "missing-package",
        )


def test_repository_stub_matches_governed_digest() -> None:
    import hashlib

    assert hashlib.sha256(_canonical_stub().read_bytes()).hexdigest() == (
        BASELINE_MODEL_STUB_SHA256
    )
