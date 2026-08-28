from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
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


def _write_minimal_qlib_provider(
    path: Path,
    *,
    include_feature: bool = True,
    end_date: str = "2024-01-03",
) -> None:
    (path / "calendars").mkdir(parents=True)
    (path / "instruments").mkdir()
    (path / "features" / "sh600000").mkdir(parents=True)
    (path / "calendars" / "day.txt").write_text(
        f"2024-01-02\n{end_date}\n", encoding="utf-8"
    )
    (path / "instruments" / "cn_all.txt").write_text(
        f"SH600000\t2024-01-02\t{end_date}\n", encoding="utf-8"
    )
    if include_feature:
        (path / "features" / "sh600000" / "close.day.bin").write_bytes(
            struct.pack("<3f", 0.0, 10.0, 11.0)
        )


def test_bridge_discovers_production_qlib_layout_and_rejects_incomplete_tree(
    tmp_path: Path,
) -> None:
    bridge = _bridge_module()
    complete = tmp_path / "qlib" / "cn-complete"
    incomplete = tmp_path / "qlib" / "cn-incomplete"
    legacy = tmp_path / "artifacts" / "qlib" / "cn-legacy"
    _write_minimal_qlib_provider(complete, end_date="2024-02-01")
    _write_minimal_qlib_provider(
        incomplete, include_feature=False, end_date="2025-01-01"
    )
    _write_minimal_qlib_provider(legacy, end_date="2024-01-31")

    candidates = bridge._qlib_provider_candidates(
        tmp_path, tmp_path / "missing-default-home"
    )

    assert complete.resolve() in candidates
    assert legacy.resolve() in candidates
    assert incomplete.resolve() not in candidates
    assert len(candidates) == 2
    assert candidates[0] == complete.resolve()
    assert "**/calendars/day.txt" not in (
        Path(__file__).parents[1] / "scripts" / "rdagent_bridge.py"
    ).read_text(encoding="utf-8")


def test_bridge_rejects_parent_environment_as_the_only_commit_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _bridge_module()
    monkeypatch.setenv("RDAGENT_COMMIT", RDAGENT_COMMIT)
    monkeypatch.delenv("RDAGENT_REPO", raising=False)
    monkeypatch.delenv("RDAGENT_RUNTIME_IMAGE_DIGEST", raising=False)
    monkeypatch.setattr(bridge, "_version", lambda: "1.2.3")
    monkeypatch.setattr(bridge, "_repo_commit", lambda _path: None)

    with pytest.raises(RuntimeError, match="no verifiable repository or distribution"):
        bridge._runtime_identity()


def test_bridge_accepts_and_reports_real_distribution_commit_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _bridge_module()
    monkeypatch.setenv("RDAGENT_COMMIT", RDAGENT_COMMIT)
    monkeypatch.delenv("RDAGENT_REPO", raising=False)
    monkeypatch.delenv("RDAGENT_RUNTIME_IMAGE_DIGEST", raising=False)
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
        "source_tree_sha256": None,
        "repository_dirty": None,
        "runtime_image_digest": None,
        "production_reproducible": False,
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

    dataset = tmp_path / "dataset"
    (dataset / "calendars").mkdir(parents=True)
    (dataset / "instruments").mkdir()
    feature = dataset / "features" / "sh600000" / "close.day.bin"
    feature.parent.mkdir(parents=True)
    (dataset / "calendars" / "day.txt").write_text(
        "2020-01-02\n2021-12-31\n2022-08-12\n2022-12-30\n"
        "2023-01-03\n2024-12-31\n",
        encoding="utf-8",
    )
    (dataset / "instruments" / "all.txt").write_text(
        "SH000300\t2020-01-02\t2024-12-31\n"
        "SH600000\t2020-01-02\t2024-12-31\n",
        encoding="utf-8",
    )
    (dataset / "instruments" / "cn_all.txt").write_text(
        "SH600000\t2020-01-02\t2024-12-31\n", encoding="utf-8"
    )
    feature.write_bytes(struct.pack("<7f", 0.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0))
    sealed_files = []
    for path in sorted(item for item in dataset.rglob("*") if item.is_file()):
        sealed_files.append(
            {
                "path": path.relative_to(dataset).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    provenance_path = dataset / "metadata" / "provenance.json"
    provenance_path.parent.mkdir()
    provenance_path.write_text(
        json.dumps(
            {
                "output_manifest": {
                    "version": "qlib-output-files-v1",
                    "files": sealed_files,
                }
            }
        ),
        encoding="utf-8",
    )

    _, environment = rdagent_command(
        settings,
        project_root=Path(__file__).parents[1],
        trace_path=tmp_path / "trace",
        result_path=tmp_path / "result.json",
        dataset_path=dataset,
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
    assert environment["QLIB_FACTOR_TEST_END"] == "2022-12-31"
    assert environment["QLIB_FACTOR_TEST_END"] < "2023-01-01"
    assert environment["MODEL_COSTEER_ENV_TYPE"] == "docker"
    assert environment["QLIB_DOCKER_NETWORK"] == "none"
    assert environment["QLIB_DOCKER_ENABLE_GPU"] == "false"
    assert environment["QLIB_DOCKER_ENABLE_CACHE"] == "false"
    assert environment["QLIB_FACTOR_RUNNER"] == (
        "quant_platform.rdagent_runner.QuantLabFactorRunner"
    )
    assert environment["QLIB_MODEL_RUNNER"] == (
        "quant_platform.rdagent_runner.QuantLabModelRunner"
    )
    assert environment["QLIB_QUANT_FACTOR_RUNNER"] == (
        "quant_platform.rdagent_runner.QuantLabFactorRunner"
    )
    assert environment["QLIB_QUANT_MODEL_RUNNER"] == (
        "quant_platform.rdagent_runner.QuantLabModelRunner"
    )
    assert Path(environment["FACTOR_CoSTEER_data_folder"]).parts[-2:] == (
        "factor-source-data",
        "full",
    )
    assert Path(environment["FACTOR_CoSTEER_data_folder_debug"]).parts[-2:] == (
        "factor-source-data",
        "debug",
    )
    mounted = json.loads(environment["QLIB_DOCKER_EXTRA_VOLUMES"])
    runtime_mounts = [
        (Path(path), config)
        for path, config in mounted.items()
        if config == {"bind": path, "mode": "ro"}
    ]
    assert len(runtime_mounts) == 1
    runtime_source, runtime_config = runtime_mounts[0]
    assert runtime_config == {"bind": str(runtime_source), "mode": "ro"}
    research_dataset = next(
        Path(path) for path, config in mounted.items() if config["bind"].endswith("cn_data")
    )
    assert research_dataset != dataset.resolve()
    assert (research_dataset / "calendars" / "day.txt").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "2020-01-02",
        "2021-12-31",
        "2022-08-12",
        "2022-12-30",
    ]
    assert (research_dataset / "instruments" / "all.txt").read_text(
        encoding="utf-8"
    ).strip().endswith("2022-12-30")
    assert (research_dataset / "instruments" / "cn_all.txt").read_text(
        encoding="utf-8"
    ).strip().endswith("2022-12-30")
    view_manifest = json.loads(
        (research_dataset / "quantlab-rdagent-dataset-view.json").read_text(
            encoding="utf-8"
        )
    )
    assert view_manifest["schema_version"] == 2
    assert view_manifest["market"] == "cn_all"
    assert view_manifest["default_market_alias"] == "cn_all"
    assert (research_dataset / "instruments" / "all.txt").read_bytes() == (
        research_dataset / "instruments" / "cn_all.txt"
    ).read_bytes()
    assert (research_dataset / "features" / "sh600000" / "close.day.bin").stat().st_size == 20
