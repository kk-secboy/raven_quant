from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform import model_prepared_cache as cache
from quant_platform import model_prepared_execution as execution
from quant_platform.model_prepared_data import (
    PreparedModelData,
    canonical_key,
    manifest_sha256,
    sha256file,
    write_prepared_data,
)

pytestmark = pytest.mark.no_database

_IMAGE = "registry.invalid/model@sha256:" + "a" * 64
_PRODUCER_CID = "b" * 64
_MODEL_CID = "c" * 64


def _mounts(command):
    output = []
    for position, value in enumerate(command[:-1]):
        if value != "--mount":
            continue
        item = {}
        for field in command[position + 1].split(","):
            name, separator, content = field.partition("=")
            item[name] = content if separator else True
        output.append(item)
    return output


@pytest.fixture
def setup(tmp_path, monkeypatch):
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "instruments").mkdir()
    (provider / "calendars" / "day.txt").write_text("2020-01-02\n2020-01-03\n")
    (provider / "instruments" / "cn_all.txt").write_text("SH001\t2020-01-02\t2020-01-03\n")
    runner = Path(__file__).resolve().parents[1] / "scripts" / "model_sandbox_runner.py"
    identity = execution.prepared_runtime_identity(
        runner, image=_IMAGE, image_id="sha256:" + "d" * 64,
    )
    manifest = {
        "dataset_identity_sha256": "e" * 64,
        "feature_set": {"features": {"F0": "$close", "F1": "$volume"}},
        "periods": {
            "train_start": "2020-01-02", "train_end": "2020-01-02",
            "valid_start": "2020-01-03", "valid_end": "2020-01-03",
            "test_start": "2020-01-06", "test_end": "2020-01-07",
        },
        "prediction_segment": "valid", "universe": "cn_all",
        "model_label_contract": {
            "label_expression": "Ref($close,-2)/Ref($close,-1)-1",
            "purge_sessions": 0, "embargo_sessions": 0,
        },
        "resource_policy": {"limits": {"memory_gb": 40}},
        "execution_environment": {
            "sandbox_image": _IMAGE, "prepared_data_producer": identity,
            "economic_policy": "unchanged", "seed_policy": "unchanged",
        },
    }
    monkeypatch.setenv("MODEL_PREPARED_DATA_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(
        execution, "prepared_data_cache", partial(cache.prepared_data_cache, min_free_bytes=0),
    )
    return {"root": tmp_path, "provider": provider, "runner": runner, "manifest": manifest}


def _call(setup, name="cell", *, manifest=None):
    workspace = setup["root"] / name
    workspace.mkdir()
    (workspace / "candidate_model.py").write_text("raise AssertionError('candidate must not load')")
    command = [
        "docker", "run", "--rm", "--cidfile", str(workspace / "container.cid"),
        "--mount", f"type=bind,src={workspace},dst=/work",
        "--mount", f"type=bind,src={setup['provider']},dst=/qlib,readonly",
        _IMAGE, "python", "-I", "runner.py",
    ]
    return {
        "command": command, "workspace": workspace, "provider": setup["provider"],
        "runner_path": setup["runner"],
        "manifest": copy.deepcopy(manifest or setup["manifest"]), "timeout_seconds": 100,
    }


def _producer_summary(request, *, seal=None, capacity=False):
    return {
        "contract_version": execution.PREPARED_SUMMARY_VERSION,
        "status": "storage_capacity_unavailable" if capacity else "prepared",
        "request_sha256": canonical_key(request), "manifest_sha256": seal,
        "prepare_seconds": 18.0, "write_seconds": 4.0, "elapsed_seconds": 23.0,
        "memory": {
            "process_rss_bytes": 10 * 1024**2, "process_peak_rss_bytes": 20 * 1024**2,
            "cgroup_version": 2, "cgroup_current_bytes": 25 * 1024**2,
            "cgroup_peak_bytes": 30 * 1024**2, "cgroup_limit_bytes": 40 * 1024**3,
            "observed_memory_peak_bytes": 30 * 1024**2,
        },
    }


def _progress_record(status="completed"):
    return {
        "contract_version": "model-prepared-data-progress-v1",
        "stage": f"producer_{status}" if status != "running" else "handler_loaded_features_8",
        "status": status, "elapsed_seconds": 2.0,
        "memory": {"process_peak_rss_bytes": 20 * 1024**2},
    }


def _fake_docker(
    monkeypatch, *, clock=None, producer_result=0, model_callback=None, producer_stdout=None,
    progress_complete=True,
):
    calls = []

    def run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        if command[:3] == ["docker", "rm", "-f"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "prepare_model_data.py" in command:
            mounts = {item["dst"]: item for item in _mounts(command)}
            lease_path = Path(mounts["/prepared-producer.lease"]["src"])
            assert lease_path.is_file()
            with cache._locked(lease_path, exclusive=True, blocking=False) as lock:
                assert lock is None
            inputs = Path(mounts["/work"]["src"])
            progress_path = Path(mounts["/audit"]["src"]) / "progress.jsonl"
            progress_path.write_text(json.dumps(_progress_record("running")) + "\n")
            request = json.loads((inputs / "request.json").read_text(encoding="utf-8"))
            cidfile = Path(command[command.index("--cidfile") + 1])
            cidfile.write_text(_PRODUCER_CID, encoding="ascii")
            if producer_result:
                progress_path.write_text(json.dumps(_progress_record(
                    "capacity" if producer_result == 75 else "failed",
                )) + "\n")
                stdout = (
                    json.dumps(_producer_summary(request, capacity=True))
                    if producer_result == 75 else ""
                )
                return subprocess.CompletedProcess(
                    command, producer_result, stdout, "producer failed",
                )
            destination = Path(mounts["/output"]["src"]) / Path(
                command[command.index("--output") + 1]
            ).name
            index = pd.MultiIndex.from_product(
                [pd.date_range("2020-01-02", periods=2), ["SH001"]],
                names=["datetime", "instrument"],
            )
            values = np.arange(len(index), dtype=np.float32)
            labels = pd.DataFrame({("label", "LABEL0"): values}, index=index)
            write_prepared_data(
                destination, contract=request,
                data=PreparedModelData(
                    index, {("feature", name): values.copy() for name, _ in request["features"]},
                    labels, labels.copy(),
                ),
            )
            if clock is not None:
                clock[0] += 23
            summary = _producer_summary(request, seal=manifest_sha256(destination))
            if progress_complete:
                with progress_path.open("a") as stream:
                    stream.write(json.dumps(_progress_record()) + "\n")
            stdout = (
                producer_stdout(summary) if producer_stdout is not None else json.dumps(summary)
            )
            return subprocess.CompletedProcess(command, 0, stdout, "")
        if model_callback is not None:
            model_callback(command, kwargs)
        return subprocess.CompletedProcess(command, 0, "model ran", "")

    monkeypatch.setattr(execution.subprocess, "run", run)
    return calls


def test_cold_prepare_then_hit_skips_producer_and_preserves_environment(setup, monkeypatch):
    calls = _fake_docker(monkeypatch)
    original = copy.deepcopy(setup["manifest"]["execution_environment"])
    cold = _call(setup, "cold")
    assert execution.run_with_prepared_data(**cold).returncode == 0
    warm = _call(setup, "warm")
    assert execution.run_with_prepared_data(**warm).returncode == 0
    producer = [command for command, _ in calls if "prepare_model_data.py" in command]
    models = [command for command, _ in calls if command[-1] == "runner.py"]
    assert len(producer) == 1 and len(models) == 2
    assert cold["manifest"]["prepared_data"]["cache_hit"] is False
    assert warm["manifest"]["prepared_data"]["cache_hit"] is True
    assert cold["manifest"]["execution_environment"] == original
    assert warm["manifest"]["execution_environment"] == original
    assert not (warm["workspace"] / "data-preparation").exists()
    audit = cold["manifest"]["prepared_data"]["producer"]
    assert audit["summary_validated"] is True
    assert audit["exit_code"] == 0
    assert audit["summary"]["memory"]["observed_memory_peak_bytes"] == 30 * 1024**2
    assert audit["stdout_sha256"] == sha256file(cold["workspace"] / "data-preparation/stdout.log")
    assert audit["stderr_sha256"] == sha256file(cold["workspace"] / "data-preparation/stderr.log")
    assert audit["progress_partial"] is False and audit["progress_status"] == "completed"
    assert audit["progress_sha256"] == sha256file(
        cold["workspace"] / "data-preparation/audit/progress.jsonl",
    )
    assert warm["manifest"]["prepared_data"]["producer"] is None
    assert "prepare_seconds" not in warm["manifest"]["prepared_data"]
    assert warm["manifest"]["prepared_data"]["cache_acquire_seconds"] >= 0


def test_producer_has_only_curated_inputs_and_never_candidate_workspace(setup, monkeypatch):
    calls = _fake_docker(monkeypatch)
    arguments = _call(setup)
    execution.run_with_prepared_data(**arguments)
    producer = next(command for command, _ in calls if "prepare_model_data.py" in command)
    mounts = {item["dst"]: item for item in _mounts(producer)}
    assert set(mounts) == {"/work", "/qlib", "/output", "/audit", "/prepared-producer.lease"}
    assert all(Path(item["src"]) != arguments["workspace"] for item in mounts.values())
    assert mounts["/work"]["readonly"] is True
    assert mounts["/qlib"]["readonly"] is True
    assert mounts["/prepared-producer.lease"]["readonly"] is True
    assert (
        Path(mounts["/prepared-producer.lease"]["src"])
        == Path(mounts["/output"]["src"]) / "producer.lease"
    )
    assert "readonly" not in mounts["/output"]
    assert "readonly" not in mounts["/audit"]
    assert Path(mounts["/audit"]["src"]) == arguments["workspace"] / "data-preparation/audit"
    assert not (arguments["workspace"] / "output/memory_stages.jsonl").exists()
    inputs = Path(mounts["/work"]["src"])
    assert set(path.name for path in inputs.iterdir()) == {
        "quant_platform", "prepare_model_data.py", "request.json",
    }
    assert not (inputs / "candidate_model.py").exists()
    assert producer[producer.index("--network") + 1] == "none"
    assert producer[producer.index("--memory") + 1] == "40g"
    assert producer[producer.index("--memory-swap") + 1] == "40g"
    assert producer[producer.index("--cpus") + 1] == "4"
    assert "OPENBLAS_NUM_THREADS=1" in producer
    assert "OMP_NUM_THREADS=1" in producer
    assert "--read-only" in producer
    assert "MLFLOW_ALLOW_FILE_STORE=true" in producer


def test_model_has_only_one_readonly_entry_and_live_readonly_lease(setup, monkeypatch):
    observed = []

    def check(command, _kwargs):
        mounts = {item["dst"]: item for item in _mounts(command)}
        entry = Path(mounts["/prepared"]["src"])
        lease = Path(mounts["/prepared.lease"]["src"])
        assert entry.parent == setup["root"] / "cache" / "entries"
        assert entry.name in lease.name
        assert mounts["/prepared"]["readonly"] is True
        assert mounts["/prepared.lease"]["readonly"] is True
        assert not any(item["src"] == str(setup["root"] / "cache") for item in mounts.values())
        with cache._locked(lease, exclusive=True, blocking=False) as lock:
            assert lock is None
        observed.append(entry)

    _fake_docker(monkeypatch, model_callback=check)
    execution.run_with_prepared_data(**_call(setup))
    assert len(observed) == 1


def test_preparation_and_fitting_ignore_legacy_wall_clock_deadline(setup, monkeypatch):
    calls = _fake_docker(monkeypatch)
    arguments = _call(setup)
    arguments["timeout_seconds"] = 1
    # The fake producer reports 23 seconds, already beyond the old entire budget.
    assert execution.run_with_prepared_data(**arguments).returncode == 0
    producer_timeout = next(
        kwargs["timeout"] for command, kwargs in calls if "prepare_model_data.py" in command
    )
    model_timeout = next(
        kwargs["timeout"] for command, kwargs in calls if command[-1] == "runner.py"
    )
    assert producer_timeout is None
    assert model_timeout is None
    progress = json.loads((arguments["workspace"] / "execution-progress.json").read_text())
    assert progress["status"] == "completed"
    assert progress["wall_clock_deadline"] is None


def test_cell_pins_prepared_manifest_outside_the_shared_entry(setup, monkeypatch):
    _fake_docker(monkeypatch)
    arguments = _call(setup)
    execution.run_with_prepared_data(**arguments)
    workspace = arguments["workspace"]
    request = json.loads((workspace / "prepared-request.json").read_text(encoding="utf-8"))
    binding = arguments["manifest"]["prepared_data"]
    entry = setup["root"] / "cache" / "entries" / canonical_key(request)
    pin = workspace / "prepared-manifest.json"
    assert pin.read_bytes() == (entry / "manifest.json").read_bytes()
    assert sha256file(pin) == binding["manifest_sha256"]
    assert canonical_key(request) == binding["request_sha256"]
    persisted = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["prepared_data"] == binding
    assert persisted["execution_environment"] == setup["manifest"]["execution_environment"]
    (entry / "manifest.json").unlink()
    assert sha256file(pin) == binding["manifest_sha256"]


def test_different_prepared_entries_do_not_change_execution_environment(setup, monkeypatch):
    _fake_docker(monkeypatch)
    first = _call(setup, "first-window")
    execution.run_with_prepared_data(**first)
    changed = copy.deepcopy(setup["manifest"])
    changed["feature_set"]["features"]["F0"] = "$close*2"
    second = _call(setup, "second-window", manifest=changed)
    execution.run_with_prepared_data(**second)
    assert (
        first["manifest"]["prepared_data"]["manifest_sha256"]
        != second["manifest"]["prepared_data"]["manifest_sha256"]
    )
    assert first["manifest"]["execution_environment"] == second["manifest"]["execution_environment"]


def test_additional_factor_bytes_are_curated_and_bound_into_request(setup, monkeypatch):
    calls = _fake_docker(monkeypatch)
    manifest = copy.deepcopy(setup["manifest"])
    manifest["additional_factors_path"] = "/work/additional_factors.parquet"
    arguments = _call(setup, manifest=manifest)
    factors = arguments["workspace"] / "additional_factors.parquet"
    factors.write_bytes(b"fake-factor-payload-for-controller-only")
    execution.run_with_prepared_data(**arguments)
    producer = next(command for command, _ in calls if "prepare_model_data.py" in command)
    inputs = Path(next(item["src"] for item in _mounts(producer) if item["dst"] == "/work"))
    request = json.loads((inputs / "request.json").read_text(encoding="utf-8"))
    assert sha256file(inputs / "additional_factors.parquet") == sha256file(factors)
    assert request["additional_factors_sha256"] == sha256file(factors)
    assert not (inputs / "candidate_model.py").exists()


def test_capacity_fallback_preserves_original_command_and_execution_environment(setup, monkeypatch):
    @contextmanager
    def no_capacity(*_args, **_kwargs):
        yield None

    monkeypatch.setattr(execution, "prepared_data_cache", no_capacity)
    calls = _fake_docker(monkeypatch)
    arguments = _call(setup)
    execution.run_with_prepared_data(**arguments)
    assert len(calls) == 1 and calls[0][0] == arguments["command"]
    assert arguments["manifest"]["prepared_data"]["mode"] == "uncached_capacity"
    assert (
        arguments["manifest"]["execution_environment"]
        == setup["manifest"]["execution_environment"]
    )
    assert not (arguments["workspace"] / "prepared-manifest.json").exists()
    assert arguments["manifest"]["prepared_data"]["producer"] is None


@pytest.mark.parametrize("fallback", ["producer_capacity", "artifact_over_budget"])
def test_capacity_fallback_preserves_actual_producer_audit(setup, monkeypatch, fallback):
    if fallback == "artifact_over_budget":
        monkeypatch.setattr(
            execution, "prepared_data_cache",
            partial(cache.prepared_data_cache, min_free_bytes=0, max_bytes=1),
        )
    calls = _fake_docker(monkeypatch, producer_result=75 if fallback == "producer_capacity" else 0)
    arguments = _call(setup)
    assert execution.run_with_prepared_data(**arguments).returncode == 0
    binding = arguments["manifest"]["prepared_data"]
    assert binding["mode"] == "uncached_capacity"
    audit = binding["producer"]
    assert audit["summary_validated"] is True
    assert audit["summary"]["memory"]["observed_memory_peak_bytes"] == 30 * 1024**2
    assert audit["stdout_sha256"] == sha256file(
        arguments["workspace"] / "data-preparation/stdout.log",
    )
    persisted = json.loads((arguments["workspace"] / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["prepared_data"]["producer"] == audit
    model = next(command for command, _ in calls if command[-1] == "runner.py")
    assert not any(item["dst"] == "/prepared" for item in _mounts(model))
    if fallback == "artifact_over_budget":
        assert audit["summary"]["status"] == "prepared"
        assert sha256file(arguments["workspace"] / "prepared-manifest.json") == (
            audit["summary"]["manifest_sha256"]
        )
    else:
        assert audit["summary"]["status"] == "storage_capacity_unavailable"
        assert audit["summary"]["manifest_sha256"] is None


@pytest.mark.parametrize("corruption", [
    "no_summary", "duplicate_summary", "manifest_sha", "request_sha", "negative_duration",
    "nan_duration", "missing_memory", "unknown_memory", "bool_memory", "wrong_limit",
    "wrong_peak", "inconsistent_duration",
])
def test_invalid_producer_summary_never_publishes_or_starts_model(setup, monkeypatch, corruption):
    def corrupt(summary):
        if corruption == "no_summary":
            return "prepared"
        if corruption == "duplicate_summary":
            return json.dumps(summary) + "\n" + json.dumps(summary)
        if corruption == "manifest_sha":
            summary["manifest_sha256"] = "f" * 64
        elif corruption == "request_sha":
            summary["request_sha256"] = "f" * 64
        elif corruption == "negative_duration":
            summary["prepare_seconds"] = -1
        elif corruption == "nan_duration":
            summary["write_seconds"] = float("nan")
        elif corruption == "inconsistent_duration":
            summary["elapsed_seconds"] = 1
        elif corruption == "missing_memory":
            del summary["memory"]
        elif corruption == "unknown_memory":
            summary["memory"]["cgroup_peak_bytes"] = None
        elif corruption == "bool_memory":
            summary["memory"]["cgroup_version"] = True
        elif corruption == "wrong_limit":
            summary["memory"]["cgroup_limit_bytes"] = 80 * 1024**3
        elif corruption == "wrong_peak":
            summary["memory"]["observed_memory_peak_bytes"] = 1
        return json.dumps(summary)

    calls = _fake_docker(monkeypatch, producer_stdout=corrupt)
    arguments = _call(setup)
    with pytest.raises(ValueError, match="producer"):
        execution.run_with_prepared_data(**arguments)
    assert not any(command[-1] == "runner.py" for command, _ in calls)
    assert not list((setup["root"] / "cache/entries").iterdir())
    binding = json.loads((arguments["workspace"] / "manifest.json").read_text(encoding="utf-8"))
    audit = binding["prepared_data"]["producer"]
    assert audit["summary"] is None and audit["summary_validated"] is False
    assert audit["stdout_sha256"] == sha256file(
        arguments["workspace"] / "data-preparation/stdout.log",
    )


def test_sealed_formal_test_is_rejected_before_cache_or_producer(setup, monkeypatch):
    calls = _fake_docker(monkeypatch)
    manifest = copy.deepcopy(setup["manifest"])
    manifest["prediction_segment"] = "test"
    manifest["final_oos_opened"] = False
    with pytest.raises(ValueError, match="opened final OOS"):
        execution.run_with_prepared_data(**_call(setup, manifest=manifest))
    assert calls == []
    assert not (setup["root"] / "cache").exists()


def test_missing_terminal_progress_is_explicitly_partial_even_with_a_valid_summary(
    setup, monkeypatch,
):
    _fake_docker(monkeypatch, progress_complete=False)
    arguments = _call(setup)
    execution.run_with_prepared_data(**arguments)
    audit = arguments["manifest"]["prepared_data"]["producer"]
    assert audit["summary_validated"] is True
    assert audit["progress_partial"] is True and audit["progress_status"] == "running"
    assert audit["progress_sha256"] == sha256file(
        arguments["workspace"] / "data-preparation/audit/progress.jsonl",
    )


def test_cache_wait_and_build_have_no_fixed_deadline(setup, monkeypatch):
    original = execution.prepared_data_cache

    @contextmanager
    def observed(**kwargs):
        assert kwargs["deadline"] is None
        with original(**kwargs) as prepared:
            yield prepared

    monkeypatch.setattr(execution, "prepared_data_cache", observed)
    _fake_docker(monkeypatch)
    arguments = _call(setup)
    assert execution.run_with_prepared_data(**arguments).returncode == 0


def test_real_model_subprocess_outlives_legacy_timeout_and_finishes(setup, monkeypatch):
    @contextmanager
    def uncached(**kwargs):
        assert kwargs["deadline"] is None
        yield None

    monkeypatch.setattr(execution, "prepared_data_cache", uncached)
    arguments = _call(setup)
    arguments["timeout_seconds"] = 1
    arguments["command"] = [
        sys.executable, "-c", "import time; time.sleep(1.1); print('model-finished')",
    ]
    started = time.monotonic()
    result = execution.run_with_prepared_data(**arguments)
    assert time.monotonic() - started > arguments["timeout_seconds"]
    assert result.returncode == 0
    assert result.stdout.strip() == "model-finished"
    progress = json.loads((arguments["workspace"] / "execution-progress.json").read_text())
    assert progress["status"] == "completed"
    assert progress["automatic_termination"] is False


@pytest.mark.parametrize(
    "failure", [subprocess.TimeoutExpired("prepare", 100), KeyboardInterrupt()],
)
def test_preparer_exception_stops_only_its_exact_cid_and_preserves_original_error(
    setup, monkeypatch, failure,
):
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if command[:3] == ["docker", "rm", "-f"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        assert "prepare_model_data.py" in command
        Path(command[command.index("--cidfile") + 1]).write_text(_PRODUCER_CID)
        progress = Path(next(item["src"] for item in _mounts(command) if item["dst"] == "/audit"))
        (progress / "progress.jsonl").write_text(json.dumps(_progress_record("running")) + "\n")
        raise failure

    monkeypatch.setattr(execution.subprocess, "run", run)
    arguments = _call(setup)
    (arguments["workspace"] / "container.cid").write_text(_MODEL_CID)
    with pytest.raises(type(failure)) as raised:
        execution.run_with_prepared_data(**arguments)
    assert raised.value is failure
    assert calls[-1] == ["docker", "rm", "-f", _PRODUCER_CID]
    assert not any(command[-1] == _MODEL_CID for command in calls)
    audit = arguments["manifest"]["prepared_data"]["producer"]
    assert audit["summary"] is None and audit["summary_validated"] is False
    assert audit["error_type"] == type(failure).__name__
    assert audit["log_capture_complete"] is False
    assert audit["progress_partial"] is True and audit["progress_status"] == "running"
    assert audit["progress_sha256"] == sha256file(
        arguments["workspace"] / "data-preparation/audit/progress.jsonl",
    )


@pytest.mark.parametrize("bad_cid", ["", "abc", "b" * 64 + "\n" + "c" * 64, "--all"])
def test_cleanup_rejects_invalid_or_multi_container_identity(tmp_path, monkeypatch, bad_cid):
    path = tmp_path / "container.cid"
    path.write_text(bad_cid)
    calls = []
    monkeypatch.setattr(execution.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    execution._stop_container(path)
    assert calls == []


def test_producer_nonzero_exit_stops_exact_container_before_reporting_failure(setup, monkeypatch):
    calls = _fake_docker(monkeypatch, producer_result=125)
    with pytest.raises(execution.PreparedDataBuildError, match="exit=125"):
        execution.run_with_prepared_data(**_call(setup))
    assert calls[-1][0] == ["docker", "rm", "-f", _PRODUCER_CID]
    assert not any(command[-1] == "runner.py" for command, _ in calls)


def test_cleanup_failure_does_not_replace_preparation_timeout(setup, monkeypatch):
    original = subprocess.TimeoutExpired("prepare", 100)

    def run(command, **_kwargs):
        if command[:3] == ["docker", "rm", "-f"]:
            raise subprocess.TimeoutExpired(command, 30)
        Path(command[command.index("--cidfile") + 1]).write_text(_PRODUCER_CID)
        raise original

    monkeypatch.setattr(execution.subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        execution.run_with_prepared_data(**_call(setup))
    assert raised.value is original


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("model", 100), KeyboardInterrupt()])
def test_model_exception_stops_model_cid_without_stopping_preparer(setup, monkeypatch, failure):
    def fail(command, _kwargs):
        Path(command[command.index("--cidfile") + 1]).write_text(_MODEL_CID)
        raise failure

    calls = _fake_docker(monkeypatch, model_callback=fail)
    with pytest.raises(type(failure)) as raised:
        execution.run_with_prepared_data(**_call(setup))
    assert raised.value is failure
    cleanup = [command for command, _ in calls if command[:3] == ["docker", "rm", "-f"]]
    assert cleanup == [["docker", "rm", "-f", _MODEL_CID]]


def test_changed_preparation_source_changes_pinned_runtime_identity(tmp_path, monkeypatch):
    source = tmp_path / "prepare_model_data.py"
    source.write_text("# producer version 1\n")
    monkeypatch.setattr(
        execution, "prepared_runtime_sources", lambda _runner: {source.name: source},
    )
    first = execution.prepared_runtime_identity(source, image=_IMAGE, image_id="sha256:" + "d" * 64)
    source.write_text("# producer version 2\n")
    second = execution.prepared_runtime_identity(
        source, image=_IMAGE, image_id="sha256:" + "d" * 64,
    )
    assert first[source.name] != second[source.name]
    assert first["sandbox_image"] == second["sandbox_image"] == _IMAGE
