from __future__ import annotations

from pathlib import Path

import pytest

from quant_platform.strategy_artifact_manifest import (
    STRATEGY_BACKTEST_ARTIFACT_MANIFEST_NAME,
    STRATEGY_BACKTEST_ARTIFACT_MANIFEST_VERSION,
    validate_backtest_artifact_manifest,
    write_backtest_artifact_manifest,
)


def _artifact_tree(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "backtest"
    (root / "robustness").mkdir(parents=True)
    (root / "daily_returns.parquet").write_bytes(b"returns")
    (root / "robustness" / "metrics.json").write_bytes(b'{"ok":true}')
    (root / "manifest.json").write_bytes(b"immutable input manifest")
    (root / "result.json").write_bytes(b"self-referential result")
    return root, write_backtest_artifact_manifest(root)


def test_backtest_artifact_manifest_is_versioned_and_complete(tmp_path: Path) -> None:
    root, evidence = _artifact_tree(tmp_path)

    payload = validate_backtest_artifact_manifest(
        root,
        expected_sha256=evidence["sha256"],
    )

    assert evidence["version"] == STRATEGY_BACKTEST_ARTIFACT_MANIFEST_VERSION
    assert evidence["path"] == STRATEGY_BACKTEST_ARTIFACT_MANIFEST_NAME
    assert evidence["file_count"] == 2
    assert [item["path"] for item in payload["files"]] == [
        "daily_returns.parquet",
        "robustness/metrics.json",
    ]


def test_backtest_artifact_manifest_rejects_missing_added_and_modified_files(
    tmp_path: Path,
) -> None:
    root, evidence = _artifact_tree(tmp_path)
    tracked = root / "daily_returns.parquet"
    original = tracked.read_bytes()

    tracked.unlink()
    with pytest.raises(ValueError, match="artifact set changed.*missing"):
        validate_backtest_artifact_manifest(root, expected_sha256=evidence["sha256"])
    tracked.write_bytes(original)

    added = root / "unexpected.bin"
    added.write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="artifact set changed.*added"):
        validate_backtest_artifact_manifest(root, expected_sha256=evidence["sha256"])
    added.unlink()

    tracked.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact SHA-256 changed"):
        validate_backtest_artifact_manifest(root, expected_sha256=evidence["sha256"])


def test_backtest_artifact_manifest_rejects_manifest_tampering(tmp_path: Path) -> None:
    root, evidence = _artifact_tree(tmp_path)
    (root / STRATEGY_BACKTEST_ARTIFACT_MANIFEST_NAME).write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest SHA-256 changed"):
        validate_backtest_artifact_manifest(root, expected_sha256=evidence["sha256"])


def test_result_and_input_manifest_keep_separate_integrity_bindings(tmp_path: Path) -> None:
    root, evidence = _artifact_tree(tmp_path)
    (root / "manifest.json").write_bytes(b"validated by execution_manifest_sha256")
    (root / "result.json").write_bytes(b"result carries artifact manifest digest")

    validate_backtest_artifact_manifest(root, expected_sha256=evidence["sha256"])
