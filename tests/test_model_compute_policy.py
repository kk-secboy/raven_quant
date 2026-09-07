from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from quant_platform.model_compute_policy import (
    fixed_model_cell_grid_policy,
    governed_cell_resource_allocation,
    model_thread_environment,
    require_model_thread_environment,
)
from quant_platform.model_recompute import governed_model_resource_policy

pytestmark = pytest.mark.no_database


@pytest.mark.parametrize("profile,memory", [("recent_3y", 40), ("balanced_5y", 32),
                                          ("robust_10y", 16), ("unknown", 40)])
@pytest.mark.parametrize("stage", ["screening", "full_validation"])
def test_profile_allocation_matches_executor_hard_limits(profile, memory, stage):
    allocation = governed_cell_resource_allocation({
        "model_engine": "ridge_baseline", "evaluation_profile_id": profile,
        "resource_stage": stage,
    })
    resource = governed_model_resource_policy(
        model_type="Tabular", model_engine="ridge_baseline", requested_hyperparameters={},
        stage=stage, requested_timeout_seconds=1800, seed=11, evaluation_profile_id=profile,
    )
    assert allocation["memory_gb"] == resource["limits"]["memory_gb"] == memory
    for name in ("cpu_count", "compute_threads", "blas_threads", "torch_interop_threads",
                 "dataloader_workers"):
        assert allocation[name] == resource["limits"][name]
    assert allocation["dataloader_workers"] == 0
    assert resource["limits"]["date_segments_modified"] is False
    assert resource["limits"]["universe_modified"] is False


@pytest.mark.parametrize("engine", ["lightgbm_baseline", "rdagent_pytorch", "platform_gru",
                                   "platform_transformer"])
def test_unknown_memory_peaks_keep_conservative_allocation(engine):
    allocation = governed_cell_resource_allocation({
        "model_engine": engine, "evaluation_profile_id": "robust_10y",
    })
    assert allocation["memory_gb"] == 40
    assert allocation["exclusive"] is (engine == "platform_transformer")
    assert allocation["cpu_count"] == (8 if engine == "lightgbm_baseline" else 4)


@pytest.mark.parametrize("stage", ["inference", "production_refit"])
def test_live_or_oos_resource_allocation_never_reuses_research_memory_measurement(stage):
    allocation = governed_cell_resource_allocation({
        "model_engine": "ridge_baseline", "evaluation_profile_id": "robust_10y",
        "resource_stage": stage,
    })
    assert allocation["memory_gb"] == 40


def test_grid_identity_is_common_and_does_not_accept_a_rewritten_table():
    grid = fixed_model_cell_grid_policy()
    baseline = copy.deepcopy(grid)
    allocations = []
    for profile in ("recent_3y", "balanced_5y", "robust_10y"):
        for seed in (11, 29, 47):
            allocations.append(governed_cell_resource_allocation({
                "model_engine": "ridge_baseline", "evaluation_profile_id": profile,
                "seed": seed, "model_cell_grid_policy": grid,
            }))
    assert grid == baseline == fixed_model_cell_grid_policy()
    assert allocations[0] == allocations[1] == allocations[2]
    assert allocations[0]["memory_gb"] != allocations[-1]["memory_gb"]
    grid["batch_limits"]["memory_gb"] = 80
    with pytest.raises(ValueError, match="grid policy"):
        governed_cell_resource_allocation({"model_cell_grid_policy": grid})


def test_cell_cannot_claim_a_smaller_scheduler_reservation_than_its_runtime_limit():
    manifest = {"model_engine": "ridge_baseline", "evaluation_profile_id": "robust_10y"}
    manifest["model_cell_allocation"] = governed_cell_resource_allocation(manifest)
    manifest["model_cell_allocation"]["memory_gb"] = 1
    with pytest.raises(ValueError, match="allocation"):
        governed_cell_resource_allocation(manifest)


@pytest.mark.parametrize("profile", ["cpu2", "cpu4", "cpu8"])
def test_explicit_threads_are_separate_from_data_loader_and_preparation(profile):
    allocation = fixed_model_cell_grid_policy()["compute_profiles"][profile]
    env = model_thread_environment(allocation)
    assert env["OPENBLAS_NUM_THREADS"] == str(allocation["cpu_count"])
    assert env["OMP_DYNAMIC"] == "FALSE"
    assert require_model_thread_environment(allocation, env) == env
    assert model_thread_environment(allocation, preparation=True)["OPENBLAS_NUM_THREADS"] == "1"
    assert allocation["dataloader_workers"] == 0
    with pytest.raises(ValueError, match="thread environment"):
        require_model_thread_environment(allocation, {**env, "OMP_NUM_THREADS": "64"})


def _runner():
    path = Path(__file__).resolve().parents[1] / "scripts/model_sandbox_runner.py"
    spec = importlib.util.spec_from_file_location("model_performance_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_counters_report_cpu_throttling_io_and_ignore_unknown_content(tmp_path):
    runner = _runner()
    (tmp_path / "cpu.stat").write_text(
        "usage_usec 100000\nnr_throttled 3\nthrottled_usec 1234\nsecret_token 123\n"
    )
    (tmp_path / "io.stat").write_text(
        "253:0 rbytes=100 wbytes=200 rios=2 wios=4\n253:1 rbytes=70 unknown=123\n"
    )
    snapshot = runner.model_performance_snapshot(cgroup_root=tmp_path)
    assert snapshot["cgroup_cpu"] == {
        "usage_usec": 100000, "nr_throttled": 3, "throttled_usec": 1234,
    }
    assert snapshot["cgroup_io"] == {"rbytes": 170, "wbytes": 200, "rios": 2, "wios": 4}
    assert "secret_token" not in json.dumps(snapshot)


def test_stage_duration_and_cpu_delta_are_flushed_on_each_stage(tmp_path, monkeypatch):
    runner = _runner()
    readings = iter([(10.0, 4.0), (14.0, 5.5)])

    def snapshot(*args, **kwargs):
        wall, cpu = next(readings)
        return {"performance": {"monotonic_seconds": wall, "process_cpu_seconds": cpu}}

    monkeypatch.setattr(runner, "model_memory_snapshot", snapshot)
    audit = tmp_path / "stages.jsonl"
    first = runner.record_model_memory_stage(audit, "started", governed_limit_bytes=None)
    assert json.loads(audit.read_text().splitlines()[0]) == first
    second = runner.record_model_memory_stage(audit, "fit", governed_limit_bytes=None)
    assert second["performance"]["stage_elapsed_seconds"] == 4.0
    assert second["performance"]["stage_process_cpu_seconds"] == 1.5
    assert len(audit.read_text().splitlines()) == 2
