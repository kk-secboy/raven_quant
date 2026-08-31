from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import OperationalError

import quant_platform.worker as worker_module
from quant_platform.worker import (
    LocalJobWorker,
    _command_with_cpu_affinity,
    _CpuAffinityPool,
    _frozen_evaluation_feature_set,
    _frozen_model_engine,
    _frozen_rdagent_model_hyperparameters,
    _indexed_independent_evaluations,
    _requires_transformer_exclusive_lane,
)
from quant_platform.worker_cli import (
    _PeriodicProbeCache,
    _worker_capabilities,
    status_server,
)

pytestmark = pytest.mark.no_database


def test_linux_numerical_command_is_hard_limited_to_first_allowed_cpus() -> None:
    command, affinity = _command_with_cpu_affinity(
        ["python", "model.py", "--fit"],
        3,
        platform="linux",
        affinity_getter=lambda _pid: {11, 7, 19, 3},
    )

    assert affinity == (3, 7, 11)
    assert command[:3] == [
        worker_module.sys.executable,
        "-c",
        worker_module._LINUX_CPU_AFFINITY_EXEC,
    ]
    assert command[3:] == ["3,7,11", "python", "model.py", "--fit"]


def test_linux_affinity_failure_is_fail_closed() -> None:
    def unavailable(_pid: int):
        raise OSError("affinity unavailable")

    with pytest.raises(ValueError, match="could not read Linux worker CPU affinity"):
        _command_with_cpu_affinity(
            ["python", "model.py"],
            8,
            platform="linux",
            affinity_getter=unavailable,
        )


def test_windows_keeps_command_and_thread_limit_contract() -> None:
    command, affinity = _command_with_cpu_affinity(
        ["python.exe", "model.py"], 8, platform="win32"
    )

    assert command == ["python.exe", "model.py"]
    assert affinity is None


def test_strategy_health_worker_command_is_exact_and_uses_runtime_clock(tmp_path) -> None:
    worker = object.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root=tmp_path)
    worker.project_root = tmp_path / "project"
    payload = {
        "strategy_version_id": "version-a",
        "promotion_stage_id": "stage-a",
        "simulation_batch_id": "batch-a",
        "formal_backtest_id": "backtest-a",
        "daily_dataset_identity_sha256": "a" * 64,
        "requested_at": "2026-08-30T00:00:01+00:00",
    }

    command, result_path, environment = worker._command(
        {"id": "job-a", "kind": "strategy_health_collect", "payload": payload}
    )

    assert result_path == tmp_path / "artifacts" / "strategy-health-jobs" / "job-a" / "result.json"
    assert environment == {}
    assert "--formal-backtest-id" in command
    assert command[command.index("--formal-backtest-id") + 1] == "backtest-a"
    assert "--observed-at" not in command


def test_snapshot_worker_forwards_only_pipeline_frozen_industry_anchor(tmp_path) -> None:
    class FakeStore:
        def create(
            self,
            kind,
            payload,
            log_path,
            *,
            idempotency_key,
            max_attempts,
        ):
            return {
                "id": f"{kind}-fixture",
                "kind": kind,
                "payload": payload,
                "log_path": str(log_path),
                "idempotency_key": idempotency_key,
                "max_attempts": max_attempts,
            }

    worker = object.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root=tmp_path / "data")
    worker.project_root = tmp_path
    worker.store = FakeStore()
    worker.notify = lambda: None
    base_payload = {
        "pipeline_id": "industry-anchor-pipeline",
        "profile": "full",
        "start": "2008-01-01",
        "end": "2026-08-31",
        "snapshot_name": "cn-industry-anchor-fixture",
    }

    command, _, _ = worker._command(
        {"kind": "data_snapshot", "payload": dict(base_payload)}
    )
    assert "--industry-history-anchor" not in command

    anchor_payload = {
        **base_payload,
        "industry_history_anchor": "cn-good-20260828",
    }
    snapshot = worker._queue_data_pipeline_successor(
        {
            "kind": "data_verify",
            "payload": anchor_payload,
            "max_attempts": 3,
        }
    )
    assert snapshot["payload"]["industry_history_anchor"] == "cn-good-20260828"
    command, _, _ = worker._command(snapshot)
    anchor_index = command.index("--industry-history-anchor")
    assert command[anchor_index + 1] == "cn-good-20260828"

    chained_snapshot = worker._queue_data_pipeline_successor(
        {
            "kind": "data_verify",
            "max_attempts": 3,
            "payload": {
                **anchor_payload,
                "pipeline_id": "industry-anchor-step-pipeline",
                "snapshot_name": "cn-industry-anchor-step-fixture",
                "pipeline_steps": [{"kind": "data_snapshot", "payload": {}}],
                "pipeline_next_index": 0,
            },
        }
    )
    assert chained_snapshot["payload"]["industry_history_anchor"] == (
        "cn-good-20260828"
    )

    with pytest.raises(ValueError, match="cannot change its industry anchor"):
        worker._queue_data_pipeline_successor(
            {
                "kind": "data_verify",
                "payload": {
                    **base_payload,
                    "pipeline_steps": [
                        {
                            "kind": "data_snapshot",
                            "payload": {
                                "industry_history_anchor": "injected-anchor"
                            },
                        }
                    ],
                    "pipeline_next_index": 0,
                },
            }
        )


def test_cpu_affinity_pool_partitions_and_reuses_container_capacity() -> None:
    pool = _CpuAffinityPool(
        platform="linux", affinity_getter=lambda _pid: set(range(24))
    )

    first = pool.acquire(8)
    second = pool.acquire(8)
    third = pool.acquire(8)
    assert first == tuple(range(8))
    assert second == tuple(range(8, 16))
    assert third == tuple(range(16, 24))
    with pytest.raises(ValueError, match="insufficient free CPUs"):
        pool.acquire(1)

    pool.release(second)
    assert pool.acquire(8) == second


def test_only_transformer_jobs_require_the_exclusive_cpu_lane() -> None:
    assert _requires_transformer_exclusive_lane(
        {
            "candidate": {
                "recipe": {
                    "model_hyperparameters": {
                        "model_engine": "platform_transformer"
                    }
                }
            }
        }
    )
    assert _requires_transformer_exclusive_lane(
        {"ensemble_members": [{"model_family": "transformer"}]}
    )
    assert not _requires_transformer_exclusive_lane(
        {"model_engine": "platform_lightgbm"}
    )


def _request(server, path: str) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{server.server_port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _wait_until(predicate, *, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_model_engine_is_preserved_from_the_frozen_strategy_recipe() -> None:
    assert _frozen_model_engine({"recipe": {}}) == "rdagent_pytorch"
    assert (
        _frozen_model_engine(
            {"recipe": {"model_hyperparameters": {"model_engine": "platform_gru"}}}
        )
        == "platform_gru"
    )
    with pytest.raises(ValueError, match="ungoverned model engine"):
        _frozen_model_engine({"recipe": {"model_engine": "changed_after_approval"}})


def test_evaluator_manifest_preserves_the_complete_dynamic_feature_set() -> None:
    feature_set = {
        "id": "research-sota-test",
        "definition_sha256": "a" * 64,
        "features": {"factor_001": "$close/$open-1"},
    }
    frozen = _frozen_evaluation_feature_set(
        {
            "feature_set_id": "research-sota-test",
            "feature_set_definition_sha256": "a" * 64,
            "feature_set": feature_set,
        }
    )
    assert frozen == feature_set
    with pytest.raises(ValueError, match="changed after job creation"):
        _frozen_evaluation_feature_set(
            {
                "feature_set_id": "research-sota-test",
                "feature_set_definition_sha256": "b" * 64,
                "feature_set": feature_set,
            }
        )


def test_rdagent_code_lane_is_frozen_before_candidate_registration() -> None:
    assert _frozen_rdagent_model_hyperparameters({}) == {
        "model_engine": "rdagent_pytorch"
    }
    with pytest.raises(ValueError, match="cannot select a platform model engine"):
        _frozen_rdagent_model_hyperparameters(
            {"model_hyperparameters": {"model_engine": "platform_transformer"}}
        )


def test_independent_evaluation_requires_one_result_per_frozen_candidate() -> None:
    job = {"payload": {"candidates": [{"id": "a"}, {"id": "b"}]}}
    indexed = _indexed_independent_evaluations(
        job,
        {
            "status": "ok",
            "evaluations": [
                {"candidate_id": "a", "status": "passed"},
                {"candidate_id": "b", "status": "resource_blocked"},
            ],
        },
    )
    assert set(indexed) == {"a", "b"}
    with pytest.raises(ValueError, match="duplicate candidate"):
        _indexed_independent_evaluations(
            job,
            {
                "status": "ok",
                "evaluations": [
                    {"candidate_id": "a", "status": "passed"},
                    {"candidate_id": "a", "status": "failed"},
                ],
            },
        )


def test_health_checks_the_worker_required_runtime() -> None:
    server = status_server(
        {
            "qlib": {"status": "ok", "qlib_version": "test"},
            "rdagent": {"status": "disabled"},
        },
        required_runtime="qlib",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _request(server, "/health")
        assert status == 200
        assert body["status"] == "ok"
        assert body["required_runtime"] == "qlib"
        assert body["runtime"]["qlib_version"] == "test"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_health_fails_closed_when_rdagent_probe_is_unavailable() -> None:
    server = status_server(
        {
            "qlib": {"status": "ok"},
            "rdagent": lambda: {
                "status": "unavailable",
                "ready": False,
                "error": "RD-Agent import failed",
            },
        },
        required_runtime="rdagent",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _request(server, "/health")
        assert status == 503
        assert body["worker"] == "runtime_unavailable"
        assert body["runtime"]["error"] == "RD-Agent import failed"

        status, details = _request(server, "/rdagent/status")
        assert status == 200
        assert details["status"] == "unavailable"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_health_converts_probe_exceptions_to_unavailable() -> None:
    def broken_probe() -> dict:
        raise ImportError("missing runtime module")

    server = status_server(
        {"qlib": {"status": "ok"}, "rdagent": broken_probe},
        required_runtime="rdagent",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _request(server, "/health")
        assert status == 503
        assert body["runtime"]["ready"] is False
        assert "ImportError" in body["runtime"]["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_health_rejects_runtime_that_is_present_but_not_ready() -> None:
    server = status_server(
        {
            "qlib": {"status": "ok"},
            "rdagent": {
                "status": "ok",
                "ready": False,
                "blockers": ["Docker is required"],
            },
        },
        required_runtime="rdagent",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _request(server, "/health")
        assert status == 503
        assert body["status"] == "unavailable"
        assert body["runtime"]["blockers"] == ["Docker is required"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_health_fails_when_queue_consumer_thread_is_dead() -> None:
    consumer_alive = False
    server = status_server(
        {"qlib": {"status": "ok"}},
        required_runtime="qlib",
        consumer_running=lambda: consumer_alive,
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _request(server, "/health")
        assert status == 503
        assert body["worker"] == "consumer_unavailable"
        assert body["consumer"] == {"running": False}

        consumer_alive = True
        status, body = _request(server, "/health")
        assert status == 200
        assert body["worker"] == "ready"
        assert body["consumer"] == {"running": True}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_model_evaluation_health_requires_a_sealed_sandbox_image() -> None:
    capabilities = _worker_capabilities(
        SimpleNamespace(
            worker_job_kinds=("model_evaluate", "quant_bundle_evaluate"),
            model_sandbox_image="",
        )
    )
    server = status_server(
        {"qlib": {"status": "ok", "ready": True}},
        required_runtime="qlib",
        capabilities=capabilities,
        consumer_running=lambda: True,
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _request(server, "/health")
        assert status == 503
        assert body["runtime"] == {"status": "ok", "ready": True}
        assert body["consumer"] == {"running": True}
        assert body["capabilities"] == {
            "job_kinds": ["model_evaluate", "quant_bundle_evaluate"],
            "model_sandbox_required": True,
            "model_sandbox_ready": False,
            "model_sandbox_error": "immutable image digest is not configured",
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_worker_retries_transient_database_claim_failure_without_dying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = object.__new__(LocalJobWorker)
    worker._stop = threading.Event()
    worker._wake = threading.Event()
    worker._thread = None
    worker.settings = SimpleNamespace(worker_job_kinds=("rdagent_factor",))
    attempts = 0
    processed: list[str] = []

    class TransientStore:
        def claim_next(self, _allowed_kinds: tuple[str, ...]):
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise OperationalError(
                    "SELECT job",
                    {},
                    RuntimeError("database is in recovery mode"),
                )
            return {"id": "recovered-job"}

    worker.store = TransientStore()

    def process(job: dict) -> None:
        processed.append(str(job["id"]))
        worker._stop.set()

    worker._run = process
    monkeypatch.setattr(worker_module, "_DATABASE_RETRY_INITIAL_SECONDS", 0.005)
    monkeypatch.setattr(worker_module, "_DATABASE_RETRY_MAX_SECONDS", 0.01)

    thread = threading.Thread(target=worker._loop, daemon=True)
    worker._thread = thread
    thread.start()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert attempts == 3
    assert processed == ["recovered-job"]


def test_worker_start_projects_a_terminal_left_by_an_earlier_recovery_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projected: list[tuple[str, str]] = []

    class FakeFactorLibrary:
        def sync_builtin_library(self) -> None:
            return None

        def list_sota(self, *, limit: int) -> list:
            assert limit == 200
            return []

    class FakeStore:
        def recover_interrupted(self, allowed_kinds: tuple[str, ...]) -> int:
            assert allowed_kinds == ("strategy_backtest",)
            return 0

        def interrupted_dependency_failures(
            self, allowed_kinds: tuple[str, ...]
        ) -> list[dict]:
            assert allowed_kinds == ("strategy_backtest",)
            return [{"id": "prior-terminal"}]

    class FakeThread:
        def __init__(self, **_kwargs) -> None:
            self.started = False

        def is_alive(self) -> bool:
            return self.started

        def start(self) -> None:
            self.started = True

    worker = object.__new__(LocalJobWorker)
    worker._thread = None
    worker._initialize_queue = True
    worker.factor_library = FakeFactorLibrary()
    worker.store = FakeStore()
    worker.settings = SimpleNamespace(worker_job_kinds=("strategy_backtest",))
    worker._mark_unhandled_job_failure = lambda job, error: projected.append(
        (str(job["id"]), error)
    )
    monkeypatch.setattr(worker_module.threading, "Thread", FakeThread)

    worker.start()

    assert projected == [
        (
            "prior-terminal",
            worker_module.INTERRUPTED_ATTEMPT_EXHAUSTED_ERROR,
        )
    ]
    assert worker._thread is not None and worker._thread.is_alive()


def test_reprojecting_interrupted_failure_preserves_terminal_backtest_and_settles_oos() -> None:
    oos_calls: list[dict] = []

    class FakeStrategies:
        def get_backtest(self, backtest_id: str) -> dict:
            assert backtest_id == "backtest-a"
            return {"status": "failed"}

        def mark_backtest(self, *_args, **_kwargs) -> None:
            pytest.fail("an existing terminal backtest must not be rewritten")

    class FakeCapitalOOS:
        def settle_batch(self, _batch_id: str, **kwargs) -> None:
            oos_calls.append(kwargs)

    worker = object.__new__(LocalJobWorker)
    worker.strategies = FakeStrategies()
    worker.capital_oos = FakeCapitalOOS()
    worker._retry_transient_database = lambda operation: operation()
    job = {
        "id": "job-a",
        "kind": "strategy_backtest",
        "payload": {
            "backtest_id": "backtest-a",
            "strategy_version_id": "version-a",
            "dataset": "snapshot-a",
            "capital_oos_batch_id": "batch-a",
        },
    }

    worker._mark_unhandled_job_failure(
        job, worker_module.INTERRUPTED_ATTEMPT_EXHAUSTED_ERROR
    )

    assert len(oos_calls) == 1
    assert oos_calls[0]["failed"] is True
    assert oos_calls[0]["failure_reason"] == (
        worker_module.INTERRUPTED_ATTEMPT_EXHAUSTED_ERROR
    )


def test_active_child_survives_progress_and_cancellation_database_outages(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = object.__new__(LocalJobWorker)
    worker._stop = threading.Event()
    worker._wake = threading.Event()
    result_path = tmp_path / "progress.json"
    result_path.write_text('{"completed": 1}', encoding="utf-8")

    progress_attempts = 0
    cancellation_attempts = 0
    persisted: list[dict] = []

    class TransientStore:
        def update_progress(self, _job_id: str, progress: dict) -> None:
            nonlocal progress_attempts
            progress_attempts += 1
            if progress_attempts == 1:
                raise OperationalError(
                    "UPDATE jobs",
                    {},
                    RuntimeError("database is in recovery mode"),
                )
            persisted.append(progress)

        def cancellation_requested(self, _job_id: str) -> bool:
            nonlocal cancellation_attempts
            cancellation_attempts += 1
            if cancellation_attempts == 1:
                raise OperationalError(
                    "SELECT cancel_requested_at",
                    {},
                    RuntimeError("database is in recovery mode"),
                )
            return False

    class SameChild:
        def __init__(self) -> None:
            self.returncode = None
            self.polls = 0
            self.terminated = False

        def poll(self):
            self.polls += 1
            if self.polls == 1:
                return None
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.terminated = True

    worker.store = TransientStore()
    child = SameChild()
    monkeypatch.setattr(worker_module, "_DATABASE_RETRY_INITIAL_SECONDS", 0.001)
    monkeypatch.setattr(worker_module, "_DATABASE_RETRY_MAX_SECONDS", 0.002)
    monkeypatch.setattr(worker_module.time, "sleep", lambda _seconds: None)

    cancelled, progress_mtime_ns = worker._monitor_process(
        "active-job", result_path, child
    )

    assert cancelled is False
    assert progress_mtime_ns == result_path.stat().st_mtime_ns
    assert child.polls == 2
    assert child.terminated is False
    assert progress_attempts == 2
    assert cancellation_attempts == 2
    assert persisted == [{"completed": 1}]


def test_outer_failure_finalization_retries_without_killing_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = object.__new__(LocalJobWorker)
    worker._stop = threading.Event()
    worker._wake = threading.Event()
    worker._thread = None
    worker.settings = SimpleNamespace(worker_job_kinds=("rdagent_factor",))
    finalize_attempts = 0
    claim_attempts = 0

    class TransientStore:
        def claim_next(self, _allowed_kinds: tuple[str, ...]):
            nonlocal claim_attempts
            claim_attempts += 1
            return {"id": "active-job", "payload": {}}

        def finish_or_retry(self, *_args, **_kwargs) -> bool:
            nonlocal finalize_attempts
            finalize_attempts += 1
            if finalize_attempts == 1:
                raise OperationalError(
                    "UPDATE jobs",
                    {},
                    RuntimeError("database is in recovery mode"),
                )
            worker._stop.set()
            return True

    worker.store = TransientStore()

    def failed_run(_job: dict) -> None:
        raise OperationalError(
            "SELECT runtime secret",
            {},
            RuntimeError("database is in recovery mode"),
        )

    worker._run = failed_run
    monkeypatch.setattr(worker_module, "_DATABASE_RETRY_INITIAL_SECONDS", 0.001)
    monkeypatch.setattr(worker_module, "_DATABASE_RETRY_MAX_SECONDS", 0.002)
    monkeypatch.setattr(worker_module.time, "sleep", lambda _seconds: None)

    thread = threading.Thread(target=worker._loop, daemon=True)
    worker._thread = thread
    thread.start()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert claim_attempts == 1
    assert finalize_attempts == 2


def test_failed_parameter_trial_ledger_is_applied_before_process_failure() -> None:
    applied: list[tuple[str, dict]] = []

    class Experiments:
        def apply_result(self, experiment_id: str, result: dict) -> None:
            applied.append((experiment_id, result))

    worker = object.__new__(LocalJobWorker)
    worker.parameter_experiments = Experiments()
    worker._settle_fin_strategy_experiment = lambda *_args, **_kwargs: (  # type: ignore[method-assign]
        (_ for _ in ()).throw(AssertionError("failed trials reached strategy settlement"))
    )
    terminal_result = {
        "status": "failed",
        "failure_kind": "trial_execution_error",
        "error": "trial 1: price-limit column is missing",
        "trials": [
            {
                "trial_index": 1,
                "status": "failed",
                "error": "price-limit column is missing",
            }
        ],
        "summary": {
            "execution_succeeded_count": 0,
            "execution_failed_count": 1,
        },
    }

    exit_code, error = worker._settle_parameter_experiment_process_result(
        {
            "kind": "parameter_experiment",
            "payload": {"parameter_experiment_id": "experiment-1"},
        },
        terminal_result,
        exit_code=1,
        process_error="generic process error",
    )

    assert applied == [("experiment-1", terminal_result)]
    assert exit_code == 1
    assert error == "trial 1: price-limit column is missing"


def test_statistical_rejection_keeps_successful_parameter_process_semantics() -> None:
    applied: list[str] = []
    settlements: list[dict] = []

    class Experiments:
        def apply_result(self, experiment_id: str, _result: dict) -> None:
            applied.append(experiment_id)

    worker = object.__new__(LocalJobWorker)
    worker.parameter_experiments = Experiments()
    worker._settle_fin_strategy_experiment = (  # type: ignore[method-assign]
        lambda _job, result: settlements.append(result) or {"next_gate": "research_rejected"}
    )
    result = {
        "status": "ok",
        "trials": [{"trial_index": 0, "status": "succeeded"}],
        "summary": {
            "succeeded_count": 0,
            "statistically_rejected_count": 1,
            "execution_succeeded_count": 1,
            "execution_failed_count": 0,
        },
    }

    exit_code, error = worker._settle_parameter_experiment_process_result(
        {
            "kind": "parameter_experiment",
            "payload": {"parameter_experiment_id": "experiment-2"},
        },
        result,
        exit_code=0,
        process_error=None,
    )

    assert applied == ["experiment-2"]
    assert settlements == [result]
    assert result["strategy_research_settlement"] == {
        "next_gate": "research_rejected"
    }
    assert exit_code == 0
    assert error is None


@pytest.mark.parametrize("failure_mode", ["returned", "raised"])
def test_failed_periodic_probe_retries_quickly_before_normal_interval(
    failure_mode: str,
) -> None:
    calls = 0

    def recovering_probe() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            if failure_mode == "raised":
                raise RuntimeError("temporary probe failure")
            return {"status": "unavailable", "ready": False}
        return {"status": "ok", "ready": True}

    cache = _PeriodicProbeCache(
        "rdagent-test",
        recovering_probe,
        interval_seconds=60,
        failure_retry_seconds=0.02,
        stale_after_seconds=120,
    )
    cache.start()
    try:
        assert _wait_until(lambda: cache.snapshot().get("status") == "ok")
        assert calls == 2
    finally:
        cache.stop()
        assert not cache.running


def test_health_reads_cached_startup_state_without_waiting_for_slow_probe() -> None:
    probe_started = threading.Event()
    release_probe = threading.Event()

    def slow_probe() -> dict[str, object]:
        probe_started.set()
        release_probe.wait(timeout=5)
        return {"status": "ok", "ready": True}

    cache = _PeriodicProbeCache(
        "rdagent-test",
        slow_probe,
        interval_seconds=60,
        stale_after_seconds=120,
    )
    cache.start()
    assert probe_started.wait(timeout=1)
    server = status_server(
        {"qlib": {"status": "ok"}, "rdagent": cache.snapshot},
        required_runtime="rdagent",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        started_at = time.monotonic()
        status, body = _request(server, "/health")
        assert time.monotonic() - started_at < 1
        assert status == 503
        assert body["runtime"]["probe_cache"]["state"] == "starting"

        release_probe.set()
        assert _wait_until(lambda: cache.snapshot().get("status") == "ok")
        status, body = _request(server, "/health")
        assert status == 200
        assert body["runtime"]["probe_cache"]["state"] == "fresh"
    finally:
        release_probe.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        cache.stop()
        assert not cache.running


def test_periodic_probe_failure_replaces_success_instead_of_staying_healthy() -> None:
    call_count = 0
    second_probe_started = threading.Event()
    release_failure = threading.Event()

    def changing_probe() -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"status": "ok", "ready": True}
        second_probe_started.set()
        release_failure.wait(timeout=5)
        raise RuntimeError("runtime disappeared")

    cache = _PeriodicProbeCache(
        "rdagent-test",
        changing_probe,
        interval_seconds=0.02,
        stale_after_seconds=1,
    )
    cache.start()
    try:
        assert _wait_until(lambda: cache.snapshot().get("status") == "ok")
        assert second_probe_started.wait(timeout=1)
        release_failure.set()
        assert _wait_until(lambda: cache.snapshot().get("status") == "unavailable")
        assert "RuntimeError" in str(cache.snapshot().get("error"))
    finally:
        release_failure.set()
        cache.stop()
        assert not cache.running


def test_hung_periodic_probe_makes_the_cached_success_stale() -> None:
    call_count = 0
    second_probe_started = threading.Event()
    release_probe = threading.Event()

    def hanging_probe() -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            second_probe_started.set()
            release_probe.wait(timeout=5)
        return {"status": "ok", "ready": True}

    cache = _PeriodicProbeCache(
        "rdagent-test",
        hanging_probe,
        interval_seconds=0.02,
        stale_after_seconds=0.08,
    )
    cache.start()
    try:
        assert _wait_until(lambda: cache.snapshot().get("status") == "ok")
        assert second_probe_started.wait(timeout=1)
        assert _wait_until(
            lambda: cache.snapshot().get("probe_cache", {}).get("state") == "stale"
        )
        snapshot = cache.snapshot()
        assert snapshot["status"] == "unavailable"
        assert snapshot["ready"] is False
    finally:
        release_probe.set()
        cache.stop()
        assert not cache.running


def test_status_server_rejects_an_unknown_required_runtime() -> None:
    with pytest.raises(ValueError, match="unknown required runtime"):
        status_server({"qlib": {"status": "ok"}}, required_runtime="rdagent", port=0)
