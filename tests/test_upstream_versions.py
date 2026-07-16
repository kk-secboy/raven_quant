from __future__ import annotations

import pytest

from quant_platform import upstream_versions

pytestmark = pytest.mark.no_database


def test_distribution_commit_is_runtime_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        upstream_versions,
        "_distribution_version",
        lambda names: f"0.0.dev0+g{upstream_versions.QLIB_COMMIT}",
    )
    monkeypatch.setattr(upstream_versions, "_repo_commit", lambda path: None)
    monkeypatch.setenv("QLIB_COMMIT", upstream_versions.QLIB_COMMIT)

    identity = upstream_versions.upstream_runtime_identity("qlib")

    assert identity["commit"] == upstream_versions.QLIB_COMMIT
    assert identity["commit_evidence"] == ["distribution"]


def test_environment_variable_alone_cannot_claim_a_validated_runtime(monkeypatch) -> None:
    monkeypatch.setattr(upstream_versions, "_distribution_version", lambda names: "1.2.3")
    monkeypatch.setattr(upstream_versions, "_repo_commit", lambda path: None)
    monkeypatch.setenv("QLIB_COMMIT", upstream_versions.QLIB_COMMIT)

    with pytest.raises(RuntimeError, match="no verifiable repository or distribution evidence"):
        upstream_versions.upstream_runtime_identity("qlib")


def test_repository_and_distribution_commit_evidence_must_agree(monkeypatch) -> None:
    monkeypatch.setattr(
        upstream_versions,
        "_distribution_version",
        lambda names: f"0.0.dev0+g{upstream_versions.QLIB_COMMIT}",
    )
    monkeypatch.setattr(upstream_versions, "_repo_commit", lambda path: "f" * 40)
    monkeypatch.setenv("QLIB_REPO", "ignored")

    with pytest.raises(RuntimeError, match="evidence disagrees"):
        upstream_versions.upstream_runtime_identity("qlib")


def test_configured_commit_mismatch_fails_before_runtime_use(monkeypatch) -> None:
    monkeypatch.setattr(
        upstream_versions,
        "_distribution_version",
        lambda names: f"0.0.dev0+g{upstream_versions.RDAGENT_COMMIT}",
    )
    monkeypatch.setattr(upstream_versions, "_repo_commit", lambda path: None)
    monkeypatch.setenv("RDAGENT_COMMIT", "0" * 40)

    with pytest.raises(RuntimeError, match="configured commit does not match"):
        upstream_versions.upstream_runtime_identity("rdagent")
