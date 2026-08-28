from __future__ import annotations

import hashlib
import importlib.util
import json
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

import quant_platform.worker as worker_module
from quant_platform.factor_evaluation_recovery import (
    RecoverySafetyError,
    inspect_orphan_factor_evaluation,
    validate_factor_evaluation_result_contract,
)
from quant_platform.research_store import FactorGatePolicy, ResearchStore
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database

JOB_ID = "a" * 32
RUN_ID = "run-a"
PERIODS = {
    "train_start": "2016-01-01",
    "train_end": "2020-12-31",
    "valid_start": "2021-01-01",
    "valid_end": "2022-12-31",
    "test_start": "2023-01-09",
    "test_end": "2024-12-31",
}
PROFILE_PERIODS = (
    PERIODS,
    {
        "train_start": "2010-01-01",
        "train_end": "2014-12-31",
        "valid_start": "2015-01-01",
        "valid_end": "2022-12-31",
        "test_start": "2023-01-09",
        "test_end": "2024-12-31",
    },
    {
        "train_start": "2012-01-01",
        "train_end": "2017-12-31",
        "valid_start": "2018-01-01",
        "valid_end": "2022-12-31",
        "test_start": "2023-01-09",
        "test_end": "2024-12-31",
    },
)


class FakeStore:
    def __init__(self, job: dict) -> None:
        self.job = job
        self.finished: list[dict] = []
        self.failed: list[dict] = []

    def get(self, job_id: str) -> dict:
        assert job_id == self.job["id"]
        return self.job

    def finish(self, job_id: str, **kwargs) -> None:
        self.finished.append({"job_id": job_id, **kwargs})

    def finish_or_retry(self, job_id: str, **kwargs) -> bool:
        self.failed.append({"job_id": job_id, **kwargs})
        return False


class FakeResearch:
    def __init__(self, evaluations: dict[str, list[dict]] | None = None) -> None:
        self.marks: list[tuple[str, str, str | None]] = []
        self.evaluations = evaluations or {}

    def mark_run(self, run_id: str, status: str, error: str | None = None) -> None:
        self.marks.append((run_id, status, error))

    def get_run(self, run_id: str) -> dict:
        assert run_id == RUN_ID
        return {"id": run_id, "status": "running"}

def _job(*, status: str = "running", profiles: int = 1) -> dict:
    evaluation_profiles = [
        {"id": f"profile-{index}", "periods": PROFILE_PERIODS[index]}
        for index in range(profiles)
    ]
    return {
        "id": JOB_ID,
        "kind": "factor_evaluate",
        "status": status,
        "payload": {
            "research_run_id": RUN_ID,
            "periods": PERIODS,
            "evaluation_profiles": evaluation_profiles,
            "candidates": [{"id": "candidate-a"}, {"id": "candidate-b"}],
        },
    }


def _evaluation(
    candidate_id: str,
    *,
    status: str = "ok",
    profile_index: int = 0,
) -> dict:
    periods = PROFILE_PERIODS[profile_index]
    if status == "failed":
        return {
            "candidate_id": candidate_id,
            "status": "failed",
            "periods": periods,
            "error": "governed evaluator failed",
        }
    return {
        "candidate_id": candidate_id,
        "status": "ok",
        "periods": periods,
        "metrics": {"research_profile": {"id": f"profile-{profile_index}"}},
        "recomputed_values_path": "/data/recomputed.parquet",
        "recomputed_values_sha256": "b" * 64,
        "recompute_evidence": {},
    }


def _result(evaluations: list[dict]) -> dict:
    return {
        "status": "ok",
        "qlib_workflow": {"run_id": RUN_ID},
        "evaluations": evaluations,
    }


def _write_result(data_root: Path, job: dict, evaluations: list[dict]) -> Path:
    path = (
        data_root
        / "artifacts"
        / "factor-evaluations"
        / job["payload"]["research_run_id"]
        / job["id"]
        / "result.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {"status": "ok", "qlib_workflow": {"run_id": RUN_ID}, "evaluations": evaluations}
        ),
        encoding="utf-8",
    )
    return path.resolve()


def _worker_proc(proc_root: Path) -> None:
    process = proc_root / "1"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python\0/usr/local/bin/quant-worker\0")
    (process / "environ").write_bytes(
        b"WORKER_JOB_KINDS=data_snapshot,factor_evaluate,model_evaluate\0"
    )


def test_inspection_accepts_exact_candidate_profile_coverage(tmp_path: Path) -> None:
    job = _job(profiles=2)
    result_path = _write_result(
        tmp_path,
        job,
        [
            _evaluation("candidate-a"),
            _evaluation("candidate-a", profile_index=1),
            _evaluation("candidate-b"),
            _evaluation("candidate-b", status="failed", profile_index=1),
        ],
    )
    proc_root = tmp_path / "proc"
    _worker_proc(proc_root)

    inspection = inspect_orphan_factor_evaluation(
        FakeStore(job), data_root=tmp_path, job_id=JOB_ID, proc_root=proc_root
    )

    assert inspection.result_path == result_path
    assert inspection.evaluation_count == 4
    assert inspection.succeeded_count == 3
    assert inspection.failed_count == 1
    assert inspection.public_report()["would_finish_as"] == "failed"


@pytest.mark.parametrize(
    ("evaluations", "message"),
    [
        ([], "no evaluations"),
        (
            [
                _evaluation("candidate-a"),
                _evaluation("candidate-a", profile_index=1),
                _evaluation("candidate-b"),
            ],
            "does not cover every frozen candidate/profile",
        ),
        (
            [
                _evaluation("candidate-a"),
                _evaluation("candidate-a"),
                _evaluation("candidate-b"),
                _evaluation("candidate-b", profile_index=1),
            ],
            "duplicates a candidate/profile",
        ),
        (
            [
                _evaluation("candidate-a"),
                _evaluation("candidate-a", profile_index=1),
                _evaluation("candidate-b"),
                _evaluation("candidate-b", profile_index=1),
                _evaluation("candidate-extra"),
            ],
            "unknown candidate",
        ),
        (
            [
                _evaluation("candidate-a"),
                _evaluation("candidate-a", profile_index=1),
                _evaluation("candidate-b"),
                _evaluation("candidate-b", profile_index=1),
                _evaluation("candidate-a", profile_index=2),
            ],
            "extra or altered frozen profile",
        ),
    ],
)
def test_result_contract_rejects_non_exact_candidate_profile_sets(
    evaluations: list[dict], message: str
) -> None:
    with pytest.raises(RecoverySafetyError, match=message):
        validate_factor_evaluation_result_contract(
            _job(profiles=2),
            _result(evaluations),
        )


def test_import_rejects_incomplete_contract_before_any_ledger_lookup(
    tmp_path: Path,
) -> None:
    job = _job(profiles=2)

    class UntouchedLedger:
        def factor_evaluation_outcome_import_state(self, *_args, **_kwargs) -> str:
            pytest.fail("invalid batch must be rejected before ledger lookup")

    worker = object.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root=tmp_path)
    worker.research = UntouchedLedger()

    with pytest.raises(RecoverySafetyError, match="candidate/profile"):
        worker._import_factor_evaluations(
            job,
            _result(
                [
                    _evaluation("candidate-a"),
                    _evaluation("candidate-a", profile_index=1),
                    _evaluation("candidate-b"),
                ]
            ),
        )


def test_recovery_finalizer_rejects_incomplete_contract_before_settlement(
    tmp_path: Path,
) -> None:
    job = _job(profiles=2)
    store = FakeStore(job)
    research = FakeResearch()
    worker = object.__new__(LocalJobWorker)
    worker.store = store
    worker.research = research
    worker.settings = SimpleNamespace(data_root=tmp_path)
    worker._import_factor_evaluations = MethodType(
        lambda *_args, **_kwargs: pytest.fail("invalid batch must not be imported"),
        worker,
    )

    with pytest.raises(RecoverySafetyError, match="candidate/profile"):
        worker.finalize_completed_factor_evaluation(
            job,
            _result(
                [
                    _evaluation("candidate-a"),
                    _evaluation("candidate-a", profile_index=1),
                    _evaluation("candidate-b"),
                ]
            ),
        )

    assert research.marks == []
    assert store.finished == []
    assert store.failed == []


def test_inspection_rejects_one_off_container_pid_namespace(tmp_path: Path) -> None:
    job = _job()
    _write_result(tmp_path, job, [_evaluation("candidate-a"), _evaluation("candidate-b")])
    proc_root = tmp_path / "proc"
    process = proc_root / "1"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python\0/usr/local/bin/quant-web\0")
    (process / "environ").write_bytes(b"WORKER_JOB_KINDS=factor_evaluate\0")

    with pytest.raises(RecoverySafetyError, match="quant-worker PID namespace"):
        inspect_orphan_factor_evaluation(
            FakeStore(job), data_root=tmp_path, job_id=JOB_ID, proc_root=proc_root
        )


def test_inspection_rejects_active_evaluator_and_incomplete_json(tmp_path: Path) -> None:
    job = _job()
    result_path = _write_result(
        tmp_path,
        job,
        [_evaluation("candidate-a"), _evaluation("candidate-b")],
    )
    proc_root = tmp_path / "proc"
    _worker_proc(proc_root)
    child = proc_root / "42"
    child.mkdir()
    (child / "cmdline").write_bytes(
        b"python\0/app/scripts/evaluate_factor_batch.py\0--output\0"
        + str(result_path).encode()
        + b"\0"
    )

    with pytest.raises(RecoverySafetyError, match="still running"):
        inspect_orphan_factor_evaluation(
            FakeStore(job), data_root=tmp_path, job_id=JOB_ID, proc_root=proc_root
        )

    (child / "cmdline").unlink()
    result_path.write_text('{"status":"ok","evaluations":[', encoding="utf-8")
    with pytest.raises(RecoverySafetyError, match="JSON is incomplete"):
        inspect_orphan_factor_evaluation(
            FakeStore(job), data_root=tmp_path, job_id=JOB_ID, proc_root=proc_root
        )


def test_inspection_rejects_wrong_status_and_partial_profile_coverage(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _worker_proc(proc_root)
    stopped = _job(status="failed")
    with pytest.raises(RecoverySafetyError, match="still-running orphan"):
        inspect_orphan_factor_evaluation(
            FakeStore(stopped), data_root=tmp_path, job_id=JOB_ID, proc_root=proc_root
        )

    running = _job(profiles=2)
    _write_result(
        tmp_path,
        running,
        [_evaluation("candidate-a"), _evaluation("candidate-b")],
    )
    with pytest.raises(RecoverySafetyError, match="candidate/profile"):
        inspect_orphan_factor_evaluation(
            FakeStore(running), data_root=tmp_path, job_id=JOB_ID, proc_root=proc_root
        )


@pytest.mark.parametrize(
    ("evaluations", "expected_status"),
    [
        ([_evaluation("candidate-a"), _evaluation("candidate-b")], "succeeded"),
        ([_evaluation("candidate-a"), _evaluation("candidate-b", status="failed")], "failed"),
    ],
)
def test_worker_recovery_reuses_normal_import_and_finish_semantics(
    evaluations: list[dict], expected_status: str, tmp_path: Path
) -> None:
    job = _job()
    job["payload"]["candidates"] = [{"id": "candidate-a"}, {"id": "candidate-b"}]
    result = {
        "status": "ok",
        "qlib_workflow": {"run_id": RUN_ID},
        "evaluations": evaluations,
    }
    store = FakeStore(job)
    research = FakeResearch()
    imported: list[tuple[str, int]] = []
    worker = object.__new__(LocalJobWorker)
    worker.store = store
    worker.research = research
    worker.settings = SimpleNamespace(data_root=tmp_path)

    def fake_import(_self, imported_job: dict, imported_result: dict) -> None:
        imported.append((imported_job["id"], len(imported_result["evaluations"])))

    worker._import_factor_evaluations = MethodType(fake_import, worker)
    outcome = worker.finalize_completed_factor_evaluation(job, result)

    assert outcome["status"] == expected_status
    assert imported == [(JOB_ID, len(evaluations))]
    assert research.marks[-1][1] == expected_status
    if expected_status == "succeeded":
        assert store.finished[-1]["exit_code"] == 0
        assert not store.failed
    else:
        assert store.failed[-1]["exit_code"] == 3
        assert store.failed[-1]["retryable"] is False
        assert not store.finished


def test_worker_recovery_allows_evaluations_imported_by_prior_job(tmp_path: Path) -> None:
    job = _job()
    old_job_id = "b" * 32
    old_artifact = (
        tmp_path
        / "artifacts"
        / "factor-evaluations"
        / RUN_ID
        / old_job_id
        / "result.json"
    )
    research = FakeResearch(
        {
            "candidate-a": [
                {"artifact_path": str(old_artifact), "recompute_evidence": {}}
            ],
            "candidate-b": [
                {
                    "artifact_path": None,
                    "recompute_evidence": {"evaluation_attempt_id": old_job_id},
                }
            ],
        }
    )
    store = FakeStore(job)
    worker = object.__new__(LocalJobWorker)
    worker.store = store
    worker.research = research
    worker.settings = SimpleNamespace(data_root=tmp_path)
    imported: list[str] = []

    def fake_import(_self, imported_job: dict, _result: dict) -> None:
        imported.append(str(imported_job["id"]))

    worker._import_factor_evaluations = MethodType(fake_import, worker)
    outcome = worker.finalize_completed_factor_evaluation(
        job,
        {
            "status": "ok",
            "qlib_workflow": {"run_id": RUN_ID},
            "evaluations": [_evaluation("candidate-a"), _evaluation("candidate-b")],
        },
    )

    assert outcome["status"] == "succeeded"
    assert imported == [JOB_ID]


def test_failed_recovery_settles_research_before_terminal_job(tmp_path: Path) -> None:
    job = _job()
    events: list[str] = []

    class OrderedStore(FakeStore):
        def finish_or_retry(self, job_id: str, **kwargs) -> bool:
            events.append("job.failed")
            return super().finish_or_retry(job_id, **kwargs)

    class OrderedResearch(FakeResearch):
        def mark_run(
            self, run_id: str, status: str, error: str | None = None
        ) -> None:
            events.append(f"run.{status}")
            super().mark_run(run_id, status, error)

    worker = object.__new__(LocalJobWorker)
    worker.store = OrderedStore(job)
    worker.research = OrderedResearch()
    worker.settings = SimpleNamespace(data_root=tmp_path)

    def fake_import(_self, _job: dict, _result: dict) -> None:
        events.append("ledger.imported")

    worker._import_factor_evaluations = MethodType(fake_import, worker)
    outcome = worker.finalize_completed_factor_evaluation(
        job,
        {
            "status": "ok",
            "qlib_workflow": {"run_id": RUN_ID},
            "evaluations": [
                _evaluation("candidate-a", status="failed"),
                _evaluation("candidate-b"),
            ],
        },
    )

    assert outcome["status"] == "failed"
    assert events == ["ledger.imported", "run.failed", "job.failed"]


def test_factor_retry_prediction_uses_claimed_attempt_counters() -> None:
    running = {"status": "running", "attempts": 1, "max_attempts": 3}

    assert LocalJobWorker._job_has_retry_remaining(running, retryable=True) is True
    assert (
        LocalJobWorker._job_has_retry_remaining(
            {**running, "attempts": 3}, retryable=True
        )
        is False
    )
    assert LocalJobWorker._job_has_retry_remaining(running, retryable=False) is False


def _terminal_recovery_fixture(
    tmp_path: Path,
    *,
    run_status: str,
    run_error: str | None = None,
    import_state: str = "identical",
) -> tuple[LocalJobWorker, FakeStore, FakeResearch, dict, dict, Path]:
    job = _job(profiles=0)
    job["payload"].update(
        {
            "dataset": "qlib-dataset-a",
            "dataset_identity_sha256": "c" * 64,
        }
    )
    evaluations = [_evaluation("candidate-a"), _evaluation("candidate-b")]
    result = {
        "status": "ok",
        "qlib_workflow": {"run_id": RUN_ID},
        "evaluations": evaluations,
    }
    result_path = _write_result(tmp_path, job, evaluations)
    store = FakeStore(job)

    class TerminalResearch(FakeResearch):
        def get_run(self, run_id: str) -> dict:
            assert run_id == RUN_ID
            return {"id": run_id, "status": run_status, "error": run_error}

        def factor_evaluation_outcome_import_state(
            self, *_args, **_kwargs
        ) -> str:
            return import_state

    research = TerminalResearch()
    worker = object.__new__(LocalJobWorker)
    worker.store = store
    worker.research = research
    worker.settings = SimpleNamespace(data_root=tmp_path)
    return worker, store, research, job, result, result_path


def test_claimed_job_with_terminal_success_run_only_fills_job_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, store, research, job, result, result_path = _terminal_recovery_fixture(
        tmp_path,
        run_status="succeeded",
    )
    original = result_path.read_bytes()
    monkeypatch.setattr(
        worker_module,
        "inspect_orphan_factor_evaluation",
        lambda *_args, **_kwargs: SimpleNamespace(
            job=job,
            result=result,
            result_path=result_path,
        ),
    )
    worker._command = MethodType(
        lambda _self, _job: pytest.fail("terminal run must not rebuild the command"),
        worker,
    )

    worker._run(job)

    assert store.finished == [{"job_id": JOB_ID, "exit_code": 0, "result": result}]
    assert not store.failed
    assert research.marks == []
    assert result_path.read_bytes() == original


def test_claimed_job_with_terminal_failed_run_only_fills_failed_job_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, store, research, job, result, result_path = _terminal_recovery_fixture(
        tmp_path,
        run_status="failed",
    )
    result["evaluations"][1] = _evaluation("candidate-b", status="failed")
    logical_error = LocalJobWorker._factor_evaluation_logical_error(result)
    assert logical_error is not None
    research.get_run = MethodType(
        lambda _self, _run_id: {
            "id": RUN_ID,
            "status": "failed",
            "error": logical_error,
        },
        research,
    )
    monkeypatch.setattr(
        worker_module,
        "inspect_orphan_factor_evaluation",
        lambda *_args, **_kwargs: SimpleNamespace(
            job=job,
            result=result,
            result_path=result_path,
        ),
    )

    worker._run(job)

    assert not store.finished
    assert store.failed[-1]["exit_code"] == 3
    assert store.failed[-1]["retryable"] is False
    assert store.failed[-1]["result"] == result
    assert research.marks == []


def test_claimed_terminal_run_fails_closed_when_ledger_is_not_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, store, research, job, result, result_path = _terminal_recovery_fixture(
        tmp_path,
        run_status="succeeded",
        import_state="missing",
    )
    original = result_path.read_bytes()
    monkeypatch.setattr(
        worker_module,
        "inspect_orphan_factor_evaluation",
        lambda *_args, **_kwargs: SimpleNamespace(
            job=job,
            result=result,
            result_path=result_path,
        ),
    )

    worker._run(job)

    assert not store.finished
    assert store.failed[-1]["exit_code"] == 1
    assert store.failed[-1]["retryable"] is False
    assert "missing an immutable evaluator outcome" in store.failed[-1]["error"]
    assert research.marks == []
    assert result_path.read_bytes() == original


def test_active_factor_run_keeps_normal_retry_path(tmp_path: Path) -> None:
    worker, store, research, job, _result, _result_path = _terminal_recovery_fixture(
        tmp_path,
        run_status="running",
    )

    assert worker._settle_claimed_factor_evaluation_with_terminal_run(job) is False
    assert not store.finished
    assert not store.failed
    assert research.marks == []


def test_research_store_refuses_implicit_terminal_run_reactivation() -> None:
    class Result:
        @staticmethod
        def first():
            return SimpleNamespace(status="succeeded")

    class Connection:
        @staticmethod
        def execute(_statement):
            return Result()

    class Engine:
        @staticmethod
        def begin():
            return nullcontext(Connection())

    research = object.__new__(ResearchStore)
    research.engine = Engine()

    with pytest.raises(ValueError, match="must be requeued explicitly"):
        research.mark_run(RUN_ID, "running")


def test_worker_recovery_delegates_partial_import_to_exact_outcome_resume(
    tmp_path: Path,
) -> None:
    job = _job()
    store = FakeStore(job)
    imported: list[str] = []
    worker = object.__new__(LocalJobWorker)
    worker.store = store
    worker.research = FakeResearch()
    worker.settings = SimpleNamespace(data_root=tmp_path)
    worker._import_factor_evaluations = MethodType(
        lambda _self, imported_job, _result: imported.append(imported_job["id"]),
        worker,
    )

    outcome = worker.finalize_completed_factor_evaluation(
        job,
        {
            "status": "ok",
            "qlib_workflow": {"run_id": RUN_ID},
            "evaluations": [_evaluation("candidate-a"), _evaluation("candidate-b")],
        },
    )

    assert outcome["status"] == "succeeded"
    assert imported == [JOB_ID]
    assert store.finished
    assert not store.failed


def test_failed_import_binds_ledger_row_to_evaluation_job(tmp_path: Path) -> None:
    job = _job()
    job["payload"]["candidates"] = [{"id": "candidate-a"}]
    job["payload"].update(
        {
            "dataset": "qlib-dataset-a",
            "dataset_identity_sha256": "c" * 64,
        }
    )
    calls: list[dict] = []
    reconciled: list[str] = []

    class FailedEvaluationCapture:
        def factor_evaluation_outcome_import_state(self, *_args, **_kwargs) -> str:
            return "missing"

        def record_failed_evaluation(self, candidate_id: str, **kwargs) -> None:
            calls.append({"candidate_id": candidate_id, **kwargs})

        def reconcile_multi_profile_admission_state(self, candidate_id: str) -> None:
            reconciled.append(candidate_id)

    worker = object.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root=tmp_path)
    worker.research = FailedEvaluationCapture()

    worker._import_factor_evaluations(
        job,
        _result([_evaluation("candidate-a", status="failed")]),
    )

    assert calls[0]["candidate_id"] == "candidate-a"
    assert calls[0]["evaluation_attempt_id"] == JOB_ID
    assert reconciled == ["candidate-a"]


def test_factor_batch_imports_operational_failures_last_per_candidate(
    tmp_path: Path,
) -> None:
    job = _job(profiles=2)
    job["payload"].update(
        {
            "dataset": "qlib-dataset-a",
            "dataset_identity_sha256": "c" * 64,
        }
    )

    class BatchStatusCapture:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []
            self.statuses: dict[str, str] = {}
            self.reconciled: list[str] = []

        def factor_evaluation_outcome_import_state(self, *_args, **_kwargs) -> str:
            return "missing"

        def record_evaluation(self, candidate_id: str, **_kwargs) -> None:
            self.calls.append((candidate_id, "ok"))
            self.statuses[candidate_id] = "gate_passed"

        def record_failed_evaluation(self, candidate_id: str, **_kwargs) -> None:
            self.calls.append((candidate_id, "failed"))
            self.statuses[candidate_id] = "evaluation_failed"

        def reconcile_multi_profile_admission_state(self, candidate_id: str) -> None:
            self.reconciled.append(candidate_id)

    research = BatchStatusCapture()
    worker = object.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root=tmp_path)
    worker.research = research
    candidate_a_ok = _evaluation("candidate-a")
    candidate_b_ok = _evaluation("candidate-b")

    worker._import_factor_evaluations(
        job,
        _result(
            [
                _evaluation("candidate-a", status="failed", profile_index=1),
                candidate_b_ok,
                _evaluation("candidate-b", status="failed", profile_index=1),
                candidate_a_ok,
            ]
        ),
    )

    assert research.calls == [
        ("candidate-a", "ok"),
        ("candidate-a", "failed"),
        ("candidate-b", "ok"),
        ("candidate-b", "failed"),
    ]
    assert research.statuses == {
        "candidate-a": "evaluation_failed",
        "candidate-b": "evaluation_failed",
    }
    assert research.reconciled == ["candidate-a", "candidate-b"]


def test_factor_batch_retry_skips_only_the_identical_outcome(tmp_path: Path) -> None:
    job = _job(profiles=0)
    job["payload"].update(
        {"dataset": "qlib-dataset-a", "dataset_identity_sha256": "c" * 64}
    )

    class PartialImportCapture:
        def __init__(self) -> None:
            self.state_calls: list[tuple[str, str]] = []
            self.recorded: list[tuple[str, str]] = []
            self.reconciled: list[str] = []

        def factor_evaluation_outcome_import_state(
            self, candidate_id: str, **kwargs
        ) -> str:
            key = (candidate_id, str(kwargs["outcome_status"]))
            self.state_calls.append(key)
            return "identical" if key == ("candidate-a", "ok") else "missing"

        def record_evaluation(self, candidate_id: str, **_kwargs) -> None:
            self.recorded.append((candidate_id, "ok"))

        def record_failed_evaluation(self, candidate_id: str, **_kwargs) -> None:
            self.recorded.append((candidate_id, "failed"))

        def reconcile_multi_profile_admission_state(self, candidate_id: str) -> None:
            self.reconciled.append(candidate_id)

    research = PartialImportCapture()
    worker = object.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root=tmp_path)
    worker.research = research

    worker._import_factor_evaluations(
        job,
        _result(
            [
                _evaluation("candidate-a"),
                _evaluation("candidate-b", status="failed"),
            ]
        ),
    )

    assert research.state_calls == [
        ("candidate-a", "ok"),
        ("candidate-b", "failed"),
    ]
    assert research.recorded == [("candidate-b", "failed")]
    assert research.reconciled == ["candidate-a", "candidate-b"]


def test_factor_batch_retry_fails_closed_on_outcome_conflict(tmp_path: Path) -> None:
    job = _job(profiles=0)
    job["payload"]["candidates"] = [{"id": "candidate-a"}]
    job["payload"].update(
        {"dataset": "qlib-dataset-a", "dataset_identity_sha256": "c" * 64}
    )

    class ConflictingImport:
        def factor_evaluation_outcome_import_state(self, *_args, **_kwargs) -> str:
            raise ValueError("factor evaluation outcome conflicts with the immutable ledger")

        def record_evaluation(self, *_args, **_kwargs) -> None:
            pytest.fail("conflicting outcome must not be written")

    worker = object.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root=tmp_path)
    worker.research = ConflictingImport()

    with pytest.raises(ValueError, match="conflicts with the immutable ledger"):
        worker._import_factor_evaluations(
            job,
            _result([_evaluation("candidate-a")]),
        )


def test_store_matches_one_exact_success_outcome_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    metrics = {
        "research_profile": {"id": "recent_3y"},
        "ic": 0.03,
    }
    evidence = {"provider_input_sha256": "d" * 64}
    artifact = tmp_path / "result.json"
    artifact.write_text("{}", encoding="utf-8")

    def canonical_sha256(value: object) -> str:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    policy = FactorGatePolicy()
    gate_status, gate_reasons = policy.evaluate(metrics)
    period_dates = {key: date.fromisoformat(value) for key, value in PERIODS.items()}
    row = {
        "factor_candidate_id": "candidate-a",
        "dataset": "qlib-dataset-a",
        "dataset_identity_sha256": "c" * 64,
        **period_dates,
        "metrics_json": metrics,
        "metrics_sha256": canonical_sha256(metrics),
        "gate_status": gate_status,
        "gate_reasons_json": gate_reasons,
        "artifact_path": str(artifact),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "recomputed_values_sha256": "b" * 64,
        "recompute_evidence_json": evidence,
    }

    class Row:
        def __init__(self, value: dict) -> None:
            self._mapping = value

    class Connection:
        def execute(self, _statement):
            return [Row(row)]

    class Engine:
        def connect(self):
            return nullcontext(Connection())

    store = object.__new__(ResearchStore)
    store.engine = Engine()
    store.policy = policy
    arguments = {
        "evaluation_attempt_id": JOB_ID,
        "artifact_path": str(artifact),
        "dataset": "qlib-dataset-a",
        "dataset_identity_sha256": "c" * 64,
        "periods": period_dates,
        "research_profile_id": "recent_3y",
        "outcome_status": "ok",
        "metrics": metrics,
        "recomputed_values_sha256": "b" * 64,
        "recompute_evidence": evidence,
    }

    assert (
        store.factor_evaluation_outcome_import_state("candidate-a", **arguments)
        == "identical"
    )
    with pytest.raises(ValueError, match="conflicts with the immutable ledger"):
        store.factor_evaluation_outcome_import_state(
            "candidate-a",
            **{**arguments, "recomputed_values_sha256": "e" * 64},
        )


def test_store_matches_legacy_failed_outcome_by_attempt_and_frozen_periods(
    tmp_path: Path,
) -> None:
    period_dates = {key: date.fromisoformat(value) for key, value in PERIODS.items()}
    row = {
        "factor_candidate_id": "candidate-a",
        "dataset": "qlib-dataset-a",
        "dataset_identity_sha256": "c" * 64,
        **period_dates,
        "metrics_json": {},
        "gate_status": "evaluation_failed",
        "gate_reasons_json": ["governed evaluator failed"],
        "artifact_path": None,
        "recompute_evidence_json": {"evaluation_attempt_id": JOB_ID},
    }

    class Row:
        _mapping = row

    class Connection:
        def execute(self, _statement):
            return [Row()]

    class Engine:
        def connect(self):
            return nullcontext(Connection())

    store = object.__new__(ResearchStore)
    store.engine = Engine()
    store.policy = FactorGatePolicy()
    arguments = {
        "evaluation_attempt_id": JOB_ID,
        "artifact_path": str(tmp_path / "result.json"),
        "dataset": "qlib-dataset-a",
        "dataset_identity_sha256": "c" * 64,
        "periods": period_dates,
        "research_profile_id": "recent_3y",
        "outcome_status": "failed",
        "error": "governed evaluator failed",
    }

    assert (
        store.factor_evaluation_outcome_import_state("candidate-a", **arguments)
        == "identical"
    )
    with pytest.raises(ValueError, match="conflicts with the immutable ledger"):
        store.factor_evaluation_outcome_import_state(
            "candidate-a",
            **{**arguments, "error": "different failure"},
        )


def test_recovery_cli_is_dry_run_by_default() -> None:
    script = Path(__file__).parents[1] / "scripts" / "recover_orphan_factor_evaluation.py"
    spec = importlib.util.spec_from_file_location("factor_recovery_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.parse_args([JOB_ID]).apply is False
    with pytest.raises(SystemExit):
        module.parse_args([JOB_ID, "--apply"])
    assert module.parse_args(
        [JOB_ID, "--apply", "--expected-result-sha256", "b" * 64]
    ).apply is True
