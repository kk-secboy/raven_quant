from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from quant_data.config import Settings
from quant_platform.rdagent_runtime import (
    rdagent_command,
    require_rdagent_runtime_identity,
)
from quant_platform.upstream_versions import RDAGENT_COMMIT

pytestmark = pytest.mark.no_database


def _bridge_module():
    path = Path(__file__).parents[1] / "scripts" / "rdagent_bridge.py"
    spec = importlib.util.spec_from_file_location("rdagent_bridge_identity_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bridge_rejects_parent_environment_as_the_only_commit_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _bridge_module()
    monkeypatch.setenv("RDAGENT_COMMIT", RDAGENT_COMMIT)
    monkeypatch.delenv("RDAGENT_REPO", raising=False)
    monkeypatch.setattr(bridge, "_version", lambda: "1.2.3")
    monkeypatch.setattr(bridge, "_repo_commit", lambda _path: None)

    with pytest.raises(RuntimeError, match="no verifiable repository or distribution"):
        bridge._runtime_identity()


def test_bridge_accepts_and_reports_real_distribution_commit_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _bridge_module()
    monkeypatch.setenv("RDAGENT_COMMIT", RDAGENT_COMMIT)
    monkeypatch.setattr(
        bridge, "_version", lambda: f"0.0.dev0+g{RDAGENT_COMMIT}"
    )
    monkeypatch.setattr(bridge, "_repo_commit", lambda _path: None)

    identity = bridge._runtime_identity()

    assert identity == {
        "name": "rdagent",
        "version": f"0.0.dev0+g{RDAGENT_COMMIT}",
        "commit": RDAGENT_COMMIT,
        "commit_evidence": ["distribution"],
    }
    assert require_rdagent_runtime_identity(identity) == identity


def test_runtime_command_forwards_repository_for_bridge_verification(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "RD-Agent"
    settings = Settings(
        api_url="",
        token="",
        data_root=tmp_path / "data",
        rdagent_repo=repository,
        rdagent_python="python",
        rdagent_command="rdagent",
    )

    _, environment = rdagent_command(
        settings,
        project_root=Path(__file__).parents[1],
        trace_path=tmp_path / "trace",
        result_path=tmp_path / "result.json",
        dataset_path=tmp_path / "dataset",
        loop_n=1,
        duration="1h",
        periods={
            "train_start": "2020-01-01",
            "train_end": "2021-12-31",
            "valid_start": "2022-01-01",
            "valid_end": "2022-12-31",
            "test_start": "2023-01-01",
            "test_end": "2024-12-31",
        },
        objective="Generate an auditable Qlib challenger factor.",
    )

    assert environment["RDAGENT_COMMIT"] == RDAGENT_COMMIT
    assert environment["RDAGENT_REPO"] == str(repository)
