from __future__ import annotations

from pathlib import Path

import pytest

from scripts.verify_governed_runtime_overlay import verify_overlay

pytestmark = pytest.mark.no_database


def test_governed_overlay_replaces_and_verifies_both_source_views() -> None:
    dockerfile = (
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "Dockerfile.governed-full-source-overlay"
    ).read_text(encoding="utf-8")

    assert "rm -rf /app/src /app/scripts /app/migrations" in dockerfile
    assert "pip install --no-deps --force-reinstall ." in dockerfile
    assert "verify_governed_runtime_overlay.py" in dockerfile
    assert "EXPECTED_RUNTIME_CLOSURE_SHA256" in dockerfile


def test_overlay_verifier_rejects_an_invalid_expected_closure(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected closure"):
        verify_overlay(tmp_path, "not-a-sha256")
