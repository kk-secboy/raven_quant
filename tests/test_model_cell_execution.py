from __future__ import annotations

import copy
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import quant_platform.model_cell_execution as execution
import quant_platform.model_cell_recovery as recovery
from quant_platform.model_cell_store import ModelCellStore, atomic_json, read_json
from quant_platform.model_compute_policy import governed_cell_resource_allocation
from quant_platform.model_recompute import ModelResourceLimitError
from quant_platform.model_research_governance import file_sha256

pytestmark = pytest.mark.no_database


@pytest.fixture
def cell(tmp_path):
    provider = tmp_path / "view"
    for name in (
        "quantlab-rdagent-dataset-view.json", "calendars/day.txt",
        "calendars/day_future.txt", "instruments/cn_all.txt",
    ):
        path = provider / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    code = tmp_path / "model.py"
    code.write_text("model code", encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text("runner", encoding="utf-8")
    call = {
        "code_path": code, "provider_path": provider, "runner_path": runner,
        "manifest": {
            "candidate_id": "candidate-a", "code_sha256": file_sha256(code), "seed": 11,
            "evaluation_profile_id": "robust_10y", "resource_stage": "full_validation",
            "model_engine": "ridge_baseline", "periods": {"train_end": "2020-01-01"},
        },
        "timeout_seconds": 7200,
    }
    batch = {"manifest": {"research_run_id": "run-1", "candidates": ["a", "b"]},
             "runtime": {"image": "sha256:" + "1" * 64}}
    store = ModelCellStore(tmp_path / "cells", batch)
    return SimpleNamespace(call=call, store=store, batch=batch, root=tmp_path)


def fake_execute(**call):
    output = call["workspace"] / "output"
    output.mkdir(parents=True)
    (output / "predictions.parquet").write_bytes(
        f"prediction seed={call['manifest']['seed']}".encode()
    )
    (output / "checkpoint.bin").write_bytes(b"checkpoint")
    return {"metrics": {"ic": 0.1}}, {"execution_environment_sha256": "e" * 64}


def execute(cell, **kwargs):
    return execution.execute_stored_cell(
        cell.store, cell.call, active_path=cell.root / "active.json",
        cleanup=lambda _: None, execute=kwargs.pop("execute", fake_execute), **kwargs,
    )


def test_resume_uses_exact_committed_cell_without_training(cell):
    first = execute(cell)
    second = execute(cell, execute=lambda **_: pytest.fail("must not refit a committed cell"))
    assert not first["reused"] and second["reused"]
    assert first["workspace"] == second["workspace"]
    assert first["receipt_sha256"] == second["receipt_sha256"]
    assert first["files"] == second["files"]


@pytest.mark.parametrize("field,value", [
    ("seed", 29), ("candidate_id", "candidate-b"),
    ("evaluation_profile_id", "recent_3y"), ("resource_stage", "screening"),
    ("periods", {"train_end": "2021-01-01"}),
])
def test_another_experiment_never_reuses_fitted_checkpoint(cell, field, value):
    first = execute(cell)
    cell.call["manifest"][field] = value
    second = execute(cell)
    assert not second["reused"]
    assert first["workspace"] != second["workspace"]


@pytest.mark.parametrize("change", ["runtime", "candidate_set", "research_run"])
def test_namespace_contains_full_registration_and_runtime(cell, change):
    first = execute(cell)
    batch = copy.deepcopy(cell.batch)
    if change == "runtime":
        batch["runtime"]["image"] = "sha256:" + "2" * 64
    elif change == "candidate_set":
        batch["manifest"]["candidates"].append("c")
    else:
        batch["manifest"]["research_run_id"] = "run-2"
    cell.store = ModelCellStore(cell.root / "cells", batch)
    second = execute(cell)
    assert not second["reused"]
    assert first["workspace"] != second["workspace"]


@pytest.mark.parametrize("mutation", ["change", "missing", "extra", "receipt"])
def test_corrupt_completed_output_fails_closed_without_refitting(cell, mutation):
    first = execute(cell)
    workspace = Path(first["workspace"])
    path = workspace / "output" / "predictions.parquet"
    if mutation == "change":
        path.write_bytes(b"changed")
    elif mutation == "missing":
        path.unlink()
    elif mutation == "extra":
        (workspace / "output" / "extra.bin").write_bytes(b"unexpected")
    else:
        receipt_path = workspace.parents[2] / "receipt.json"
        receipt = read_json(receipt_path)
        receipt["result"]["metrics"]["ic"] = 0.9
        atomic_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="changed|checksum"):
        execute(cell, execute=lambda **_: pytest.fail("corruption is not a retry opportunity"))


@pytest.mark.parametrize("error,status", [
    (ValueError("invalid metric"), "failed"),
    (ModelResourceLimitError("memory cap"), "resource_blocked"),
    (subprocess.TimeoutExpired("model", 7200), "failed"),
])
def test_terminal_failure_is_durable_and_never_retried(cell, error, status):
    def fail(**_):
        raise error

    first = execute(cell, execute=fail)
    second = execute(cell, execute=lambda **_: pytest.fail("failed cell cannot retrain"))
    assert first["status"] == status == second["status"]
    assert second["reused"]
    assert first["error"] == second["error"]


def test_interruption_retains_old_attempt_and_can_resume_missing_receipt(cell):
    old_workspaces = []

    def cancel(**call):
        fake_execute(**call)
        old_workspaces.append(call["workspace"])
        raise execution.ModelCellCancelled()

    with pytest.raises(execution.ModelCellCancelled):
        execute(cell, execute=cancel)
    resumed = execute(cell)
    assert Path(old_workspaces[0]).is_dir()
    assert Path(resumed["workspace"]) != old_workspaces[0]
    assert resumed["status"] == "completed" and not resumed["reused"]


def test_recovery_cleanup_failure_prevents_new_attempt(cell):
    with pytest.raises(execution.ModelCellCancelled):
        execute(cell, execute=lambda **_: (_ for _ in ()).throw(execution.ModelCellCancelled()))
    with pytest.raises(RuntimeError, match="daemon"):
        execution.execute_stored_cell(
            cell.store, cell.call, active_path=cell.root / "active.json",
            execute=lambda **_: pytest.fail("cleanup must finish before running"),
            cleanup=lambda _: (_ for _ in ()).throw(RuntimeError("daemon unavailable")),
        )


def test_changed_provider_binding_never_reuses_result(cell):
    first = execute(cell)
    (cell.call["provider_path"] / "calendars/day.txt").write_bytes(b"another calendar")
    second = execute(cell)
    assert not second["reused"] and first["workspace"] != second["workspace"]


def test_output_namespace_is_stable_only_inside_one_job(tmp_path):
    root = tmp_path / "artifacts/model-evaluations/run/job"
    first = root / "attempts" / ("attempt-0001-" + "1" * 32) / "result.json"
    second = root / "attempts" / ("attempt-0002-" + "2" * 32) / "result.json"
    assert execution.model_cell_store_root(first) == execution.model_cell_store_root(second)
    with pytest.raises(ValueError, match="governed"):
        execution.model_cell_store_root(tmp_path / "result.json")


@pytest.mark.parametrize("progress_failure", [None, "broken_publisher", "write_denied"])
def test_scheduler_parallel_budget_and_deterministic_output(cell, monkeypatch, progress_failure):
    batch = execution.ModelCellBatchExecutor(
        store_root=cell.root / "cells", batch_identity=cell.batch,
        control_root=cell.root / "controls",
    )
    events = []
    running = {}
    calls = []
    for index, profile in enumerate(("robust_10y", "robust_10y", "recent_3y", "balanced_5y")):
        call = copy.deepcopy(cell.call)
        call["manifest"].update(seed=index, evaluation_profile_id=profile)
        calls.append(call)

    class Process:
        def __init__(self, command, **_):
            self.directory = Path(command[command.index("--control") + 1])
            call = read_json(self.directory / "call.json")
            self.index = call["manifest"]["seed"]
            self.allocation = governed_cell_resource_allocation(call["manifest"])
            running[self.index] = self.allocation
            assert sum(row["cpu_count"] for row in running.values()) <= 8
            assert sum(row["memory_gb"] for row in running.values()) <= 40
            events.append(("start", self.index, len(running)))
            result = execution.execute_stored_cell(
                batch.store, call, active_path=self.directory / "active.json",
                execute=fake_execute, cleanup=lambda _: None,
            )
            atomic_json(self.directory / "response.json", result)
            self.returncode = None
            self.pid = 9999999
            self.polls = 0

        def poll(self):
            self.polls += 1
            if self.polls >= (3 if self.index == 0 else 2):
                self.returncode = 0
                running.pop(self.index, None)
            return self.returncode

    monkeypatch.setattr(execution.subprocess, "Popen", Process)
    monkeypatch.setattr(execution, "process_identity", lambda _: None)
    monkeypatch.setattr(execution.time, "sleep", lambda _: None)
    if progress_failure == "broken_publisher":
        def broken_progress(*_args):
            raise ValueError("temporarily malformed observational sidecar")
        monkeypatch.setattr(batch, "_publish_progress", broken_progress)
    elif progress_failure == "write_denied":
        publish = execution.atomic_json

        def denied_progress(path, payload):
            if path.name == "model-progress.json":
                raise PermissionError("progress file is unavailable")
            return publish(path, payload)
        monkeypatch.setattr(execution, "atomic_json", denied_progress)
    outcomes = batch.run_many(calls)
    assert [item["request"]["manifest"]["seed"] for item in outcomes] == [0, 1, 2, 3]
    assert max(event[2] for event in events) == 2
    assert ("start", 2, 1) in events and ("start", 3, 1) in events


@pytest.mark.parametrize("sidecar", [
    "absent", "invalid_active", "outside_workspace", "invalid_json", "wrong_contract",
    "invalid_warnings", "valid",
])
def test_batch_progress_is_observation_only_and_isolates_bad_cells(cell, sidecar):
    batch = execution.ModelCellBatchExecutor(
        store_root=cell.root / "cells", batch_identity=cell.batch,
        control_root=cell.root / "controls",
    )
    active = {}
    for index in range(2):
        directory = cell.root / f"control-{index}"
        workspace = batch.store.root / f"work-{index}"
        directory.mkdir()
        workspace.mkdir()
        active[index] = (None, directory, None)
        atomic_json(directory / "active.json", {"workspace": str(workspace)})
        progress = {
            "contract_version": "model-execution-progress-v1-observe-only",
            "warnings": ["progress_not_observed"],
            "seconds_without_observed_progress": 9000,
            "automatic_termination": False,
        }
        if index == 0:
            if sidecar == "absent":
                continue
            if sidecar == "invalid_active":
                (directory / "active.json").write_text("{")
                continue
            if sidecar == "outside_workspace":
                atomic_json(directory / "active.json", {"workspace": str(cell.root)})
                continue
            if sidecar == "invalid_json":
                (workspace / "execution-progress.json").write_text("{")
                continue
            if sidecar == "wrong_contract":
                progress["contract_version"] = "bad-contract"
            if sidecar == "invalid_warnings":
                progress["warnings"] = [{}]
        atomic_json(workspace / "execution-progress.json", progress)
    calls = [cell.call, cell.call]
    batch._publish_progress(calls, active, {})
    first = read_json(cell.root / "model-progress.json")
    batch._publish_progress(calls, active, {})
    second = read_json(cell.root / "model-progress.json")
    assert second["status"] == "running"
    assert second["automatic_termination"] is False
    assert second["completed_cells"] == 0
    assert second["active_cells"][1]["observation_status"] == "available"
    assert second["active_cells"][1]["progress"]["seconds_without_observed_progress"] == 9000
    assert first["active_cells"] == second["active_cells"]  # Heartbeat is not computation.
    expected = "available" if sidecar == "valid" else (
        "not_yet_available" if sidecar == "absent" else "unavailable"
    )
    assert second["active_cells"][0]["observation_status"] == expected


def test_cleanup_requires_container_mount_ownership(tmp_path, monkeypatch):
    workspace = tmp_path / "work"
    workspace.mkdir()
    cid = "a" * 64
    (workspace / "container.cid").write_text(cid)
    commands = []

    def docker(command, **_):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps([{
            "Id": cid, "Mounts": [{"Destination": "/work", "Source": str(tmp_path / "other")}],
        }]))

    monkeypatch.setattr(execution.subprocess, "run", docker)
    with pytest.raises(ValueError, match="owner"):
        execution.cleanup_cell_containers(workspace)
    assert all(command[1] == "inspect" for command in commands)


def test_cleanup_removal_is_verified_before_return(tmp_path, monkeypatch):
    workspace = tmp_path / "work"
    workspace.mkdir()
    cid = "a" * 64
    (workspace / "container.cid").write_text(cid)
    calls = []

    def docker(command, **_):
        calls.append(command)
        if len(calls) == 3:
            return SimpleNamespace(returncode=1, stdout="", stderr="No such object")
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps([{
            "Id": cid, "Mounts": [{"Destination": "/work", "Source": str(workspace)}],
        }]))

    monkeypatch.setattr(execution.subprocess, "run", docker)
    execution.cleanup_cell_containers(workspace)
    assert [command[1] for command in calls] == ["inspect", "rm", "inspect"]


def test_cleanup_fence_retains_marker_when_daemon_unavailable(tmp_path, monkeypatch):
    output = (tmp_path / "artifacts/model-evaluations/run/job/attempts"
              / ("attempt-0001-" + "1" * 32) / "result.json")
    marker = recovery.register_model_batch(output)
    workspace = (marker.parent / "model-cells" / ("a" * 64) / ("b" * 64)
                 / "attempts" / ("c" * 32) / "work")
    workspace.mkdir(parents=True)
    monkeypatch.setattr(recovery, "cleanup_cell_containers", lambda _: (
        _ for _ in ()
    ).throw(RuntimeError("Docker daemon unavailable")))
    with pytest.raises(recovery.ModelCellCleanupPending, match="reservation retained"):
        recovery.cleanup_model_batch(marker)
    assert read_json(marker)["cleanup_status"] == "blocked"
    monkeypatch.setattr(recovery, "cleanup_cell_containers", lambda _: None)
    recovery.cleanup_model_batch(marker)
    assert not marker.exists()
    assert list(marker.parent.glob("model-cleanup-audit/*.json"))


@pytest.mark.skipif(not os.sys.platform.startswith("linux"), reason="Linux PID birth fence")
def test_pid_identity_is_not_reused_for_another_process():
    process = subprocess.Popen([os.sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        identity = execution.process_identity(process.pid)
        wrong = {**identity, "start_ticks": "0"}
        execution.stop_owned_process(wrong, grace_seconds=0)
        assert process.poll() is None
        execution.stop_owned_process(identity, grace_seconds=0.1)
        process.wait(timeout=5)
        assert process.returncode != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def _job_output(tmp_path):
    return (tmp_path / "artifacts/model-evaluations/run/job/attempts"
            / ("attempt-0001-" + "1" * 32) / "result.json")


def test_live_owner_lease_protects_job_from_other_worker_recovery(tmp_path):
    output = _job_output(tmp_path)
    claim = {"job_id": "job", "attempts": 1, "started_at": "2026-09-07T01:00:00+00:00"}
    with recovery.model_owner_lease(output):
        marker = recovery.register_model_batch(output, job_claim=claim)
        observed = recovery.recover_model_batches(tmp_path)
        assert observed == {"protected_job_ids": ("job",), "recoverable_model_claims": ()}
        assert marker.exists()
    observed = recovery.recover_model_batches(tmp_path)
    assert observed == {"protected_job_ids": (), "recoverable_model_claims": (claim,)}
    assert not marker.exists()


def test_normal_owner_cleanup_does_not_upgrade_its_shared_lease(tmp_path):
    output = _job_output(tmp_path)
    with recovery.model_owner_lease(output):
        marker = recovery.register_model_batch(output)
        recovery.cleanup_model_batch(marker)
        assert not marker.exists()


def test_malformed_cleanup_marker_is_fail_closed(tmp_path):
    marker = recovery.register_model_batch(_job_output(tmp_path))
    marker.write_text('{"result_path": "elsewhere"}', encoding="utf-8")
    with pytest.raises(recovery.ModelCellCleanupPending, match="reservation retained"):
        recovery.cleanup_model_batch(marker)
    assert marker.exists()


@pytest.mark.skipif(not os.sys.platform.startswith("linux"), reason="Linux orphan/lease fence")
def test_parent_sigkill_ends_child_and_releases_its_independent_lease(tmp_path):
    lease = tmp_path / "owner.lease"
    ready = tmp_path / "ready.json"
    child_code = """
import json, os, sys, time
from pathlib import Path
from quant_platform.model_cell_execution import _parent_death_signal, cancellation_signals
from quant_platform.model_prepared_cache import _locked
with cancellation_signals():
    _parent_death_signal(int(sys.argv[1]))
    with _locked(Path(sys.argv[2]), exclusive=False):
        Path(sys.argv[3]).write_text(json.dumps({'pid': os.getpid()}))
        while True:
            time.sleep(0.1)
"""
    owner_code = """
import os, subprocess, sys, time
from pathlib import Path
from quant_platform.model_prepared_cache import _locked
with _locked(Path(sys.argv[1]), exclusive=False):
    subprocess.Popen([sys.executable, '-c', sys.argv[3], str(os.getpid()),
                      sys.argv[1], sys.argv[2]])
    while True:
        time.sleep(0.1)
"""
    owner = subprocess.Popen(
        [os.sys.executable, "-c", owner_code, str(lease), str(ready), child_code],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    child_identity = None
    try:
        deadline = time.monotonic() + 15
        while not ready.exists():
            assert owner.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.05)
        child_identity = execution.process_identity(json.loads(ready.read_text())["pid"])
        assert child_identity is not None
        owner.kill()
        owner.wait(timeout=5)
        while execution.process_identity(child_identity["pid"]) == child_identity:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        from quant_platform.model_prepared_cache import _locked

        with _locked(lease, exclusive=True, blocking=False) as recovered:
            assert recovered is not None
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        if child_identity is not None:
            execution.stop_owned_process(child_identity, grace_seconds=0)
