from __future__ import annotations

import copy
import json
import os
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.dialects import postgresql

from quant_platform.api import _public_job, _public_rdagent_run
from quant_platform.run_progress import (
    ParameterProgressReader,
    parameter_progress_statement,
    unavailable_progress,
)

pytestmark = pytest.mark.no_database


def case(tmp_path):
    started = datetime.now(UTC) - timedelta(hours=2)
    output = tmp_path / "artifacts" / "parameter-experiments" / ("a" * 32)
    output.mkdir(parents=True)
    governance = {
        "dataset_identity_sha256": "f" * 64, "stage": "policy_only", "plan_sha256": "e" * 64,
        "mode": "strategy_policy_only_pre_final",
    }
    binding = {
        "job_id": "c" * 32, "experiment_id": "a" * 32, "strategy_version_id": "b" * 32,
        "dataset": "test-day", "payload_dataset": "test-day",
        "payload_strategy_version_id": "b" * 32, "artifact_path": str(output),
        "job_status": "running", "experiment_status": "running", "attempts": 1,
        "job_started_at": started, "trial_count": 2,
        **{f"payload_{key}": value for key, value in governance.items()},
        **{f"governance_{key}": value for key, value in governance.items()},
    }
    job = {
        "id": "c" * 32, "kind": "parameter_experiment", "status": "running", "attempts": 1,
        "started_at": started.isoformat(timespec="seconds"),
        "payload": {"strategy_competition_stage": "policy_only"},
    }
    manifest = {
        "experiment_id": "a" * 32, "strategy_version_id": "b" * 32, "dataset": "test-day",
        "periods": {"governance": governance}, "evaluation_mode": governance["mode"],
    }
    document = {"completed_count": 1, "succeeded_count": 0, "failed_count": 1, "trial_count": 2,
                "trials": [{"score": 123.456, "error": "Authorization: Basic secret",
                            "returns": [1, 2], "status": "failed"}]}
    manifest_path, progress_path = output / "manifest.json", output / "progress.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    progress_path.write_text(json.dumps(document), encoding="utf-8")
    os.utime(manifest_path, (started.timestamp() + 1, started.timestamp() + 1))
    os.utime(progress_path, (started.timestamp() + 60, started.timestamp() + 60))
    return job, binding, manifest_path, progress_path


def rewrite(path, transform):
    stamp = path.stat().st_mtime_ns
    document = json.loads(path.read_text(encoding="utf-8"))
    transform(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    os.utime(path, ns=(stamp, stamp))


def test_partial_failure_stays_running_and_checkpoint_is_not_a_five_minute_heartbeat(tmp_path):
    job, binding, _, progress_path = case(tmp_path)
    original = copy.deepcopy(job)
    progress = ParameterProgressReader(tmp_path).read(job, binding)
    assert progress == {
        "state": "available", "completed_count": 1, "succeeded_count": 0,
        "failed_count": 1, "trial_count": 2,
        "updated_at": datetime.fromtimestamp(progress_path.stat().st_mtime, UTC).isoformat(),
    }
    assert job == original
    record = {**job, "_parameter_progress": progress}
    for public in (
        _public_job(record),
        _public_rdagent_run(
            {"id": "d" * 32, "kind": "fin_strategy", "status": "running"}, linked_job=record
        ),
    ):
        assert public["status"] == "running"
        assert public["presentation"]["display_status"] == "running"
        assert public["presentation"]["reason_code"] == "partial_trial_failure"
        assert public["presentation"]["safe_reason"] == (
            "仍在执行；已完成 1/2 个试验，其中 1 个失败。"
        )
        assert public["presentation"]["execution_phase"] == "policy_only"
        assert public["presentation"]["progress"]["failed_count"] == 1
        for private in ("score", "returns", "123.456", "secret", "Authorization"):
            assert private not in json.dumps(public)


def test_missing_progress_is_unknown_and_never_zero_failures(tmp_path):
    job, binding, _, progress_path = case(tmp_path)
    progress_path.unlink()
    progress = ParameterProgressReader(tmp_path).read(job, binding)
    assert progress == unavailable_progress()
    public = _public_job({**job, "_parameter_progress": progress})["presentation"]
    assert public["display_status"] == "running" and public["reason_code"] is None
    assert public["progress"]["failed_count"] is None


@pytest.mark.parametrize("bad", ["old_manifest", "old_progress", "future_progress", "new_attempt"])
def test_stale_attempt_and_future_files_are_not_presented_as_current(tmp_path, bad):
    job, binding, manifest_path, progress_path = case(tmp_path)
    start = binding["job_started_at"].timestamp()
    if bad == "old_manifest":
        os.utime(manifest_path, (start - 1, start - 1))
    elif bad == "old_progress":
        os.utime(progress_path, (start, start))
    elif bad == "future_progress":
        future = datetime.now(UTC).timestamp() + 1000
        os.utime(progress_path, (future, future))
    else:
        binding["job_started_at"] += timedelta(hours=1)
        binding["attempts"] = 2
    assert ParameterProgressReader(tmp_path).read(job, binding) == unavailable_progress()


@pytest.mark.parametrize("key", ["experiment_id", "strategy_version_id", "dataset"])
def test_manifest_exact_owner_is_required(tmp_path, key):
    job, binding, manifest_path, _ = case(tmp_path)
    rewrite(manifest_path, lambda body: body.update({key: "wrong-owner"}))
    assert ParameterProgressReader(tmp_path).read(job, binding) == unavailable_progress()


@pytest.mark.parametrize("key", ["dataset_identity_sha256", "stage", "plan_sha256", "mode"])
def test_manifest_governance_provenance_must_match_db_and_job(tmp_path, key):
    job, binding, manifest_path, _ = case(tmp_path)
    rewrite(manifest_path, lambda body: body["periods"]["governance"].update({key: "wrong"}))
    assert ParameterProgressReader(tmp_path).read(job, binding) == unavailable_progress()


@pytest.mark.parametrize("change", [
    {"completed_count": 0}, {"failed_count": True}, {"succeeded_count": -1},
    {"trial_count": 3}, {"completed_count": 3, "failed_count": 3},
])
def test_counts_require_registered_total_and_internal_consistency(tmp_path, change):
    job, binding, _, progress_path = case(tmp_path)
    rewrite(progress_path, lambda body: body.update(change))
    assert ParameterProgressReader(tmp_path).read(job, binding) == unavailable_progress()


@pytest.mark.parametrize("change", [
    {"job_id": "wrong"}, {"experiment_status": "failed"}, {"payload_dataset": "wrong"},
    {"payload_strategy_version_id": "wrong"}, {"governance_stage": "full_stack"},
    {"payload_dataset_identity_sha256": None},
])
def test_wrong_database_owner_and_incomplete_provenance_are_unavailable(tmp_path, change):
    job, binding, _, _ = case(tmp_path)
    binding.update(change)
    assert ParameterProgressReader(tmp_path).read(job, binding) == unavailable_progress()


def test_outside_owner_path_and_partially_written_json_fail_closed(tmp_path):
    job, binding, _, progress_path = case(tmp_path)
    bad = {**binding, "artifact_path": str(tmp_path)}
    assert ParameterProgressReader(tmp_path).read(job, bad) == unavailable_progress()
    progress_path.write_text('{"completed_count":', encoding="utf-8")
    assert ParameterProgressReader(tmp_path).read(job, binding) == unavailable_progress()


def test_manifest_changed_during_progress_read_is_unavailable(tmp_path, monkeypatch):
    job, binding, manifest_path, _ = case(tmp_path)
    original = json.loads

    def loads(raw):
        value = original(raw)
        if "completed_count" in value:
            stamp = manifest_path.stat().st_mtime_ns + 1000
            os.utime(manifest_path, ns=(stamp, stamp))
        return value

    monkeypatch.setattr("quant_platform.run_progress.json.loads", loads)
    assert ParameterProgressReader(tmp_path).read(job, binding) == unavailable_progress()


def test_registered_count_query_reads_no_trial_outcomes_or_metrics():
    sql = str(parameter_progress_statement(["c" * 32]).compile(dialect=postgresql.dialect()))
    assert "parameter_experiments.job_id = quantlab.jobs.id" in sql
    assert "parameter_experiments.id = CAST" in sql
    assert "count(*)" in sql
    for forbidden in (".score", ".metrics_json", ".warnings_json", ".summary_json", ".error"):
        assert forbidden not in sql
