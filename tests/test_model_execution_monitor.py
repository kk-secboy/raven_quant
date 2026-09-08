import json

import pytest

from quant_platform import model_execution_monitor as monitor_module
from quant_platform.model_execution_monitor import ModelExecutionMonitor

pytestmark = pytest.mark.no_database


def test_long_running_model_warns_and_remains_running(tmp_path, monkeypatch, caplog):
    clock = [0.0]
    monkeypatch.setattr(monitor_module.time, "monotonic", lambda: clock[0])
    monitor = ModelExecutionMonitor(tmp_path)
    monitor.sample()
    clock[0] = 7201.0
    result = monitor.sample()
    assert result["status"] == "running"
    assert result["warnings"] == ["elapsed_warning", "progress_not_observed"]
    assert result["automatic_termination"] is False
    assert result["wall_clock_deadline"] is None
    assert "continuing" in caplog.text
    monitor.finish("completed")
    assert monitor.sample()["status"] == "completed"


def test_only_model_progress_resets_observation_warning(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(monitor_module.time, "monotonic", lambda: clock[0])
    monitor = ModelExecutionMonitor(tmp_path)
    monitor.set_phase("model_compute")
    clock[0] = 1900
    assert "progress_not_observed" in monitor.sample()["warnings"]
    # Writing the monitor heartbeat is not evidence of actual computation.
    clock[0] = 1901
    assert monitor.sample()["seconds_without_observed_progress"] == 1901
    output = tmp_path / "output"
    output.mkdir()
    (output / "memory_stages.jsonl").write_text('{"stage":"prediction_complete"}\n')
    result = monitor.sample()
    assert result["seconds_without_observed_progress"] == 0
    assert result["warnings"] == ["elapsed_warning"]
    assert result["status"] == "running"


@pytest.mark.parametrize(
    "failure,status", [(ValueError, "failed"), (KeyboardInterrupt, "interrupted")],
)
def test_monitor_preserves_real_errors_and_cancellation(tmp_path, failure, status):
    with pytest.raises(failure):
        with ModelExecutionMonitor(tmp_path):
            raise failure("original failure")
    result = json.loads((tmp_path / "execution-progress.json").read_text())
    assert result["status"] == status
    assert result["warnings"] == []


def test_monitor_write_failure_does_not_abort_model(tmp_path, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("audit disk unavailable")

    monkeypatch.setattr(monitor_module, "atomic_json", unavailable)
    with ModelExecutionMonitor(tmp_path) as monitor:
        monitor.finish("completed")
    assert monitor.status == "completed"


@pytest.mark.parametrize("component", ["observations", "writer", "thread", "logging"])
def test_observer_dependency_errors_do_not_abort_computation(tmp_path, monkeypatch, component):
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("observer unavailable")

    if component == "observations":
        monkeypatch.setattr(ModelExecutionMonitor, "_observations", unavailable)
    elif component == "writer":
        monkeypatch.setattr(monitor_module, "atomic_json", unavailable)
    elif component == "thread":
        monkeypatch.setattr(monitor_module.threading.Thread, "start", unavailable)
    else:
        monkeypatch.setattr(ModelExecutionMonitor, "_observations", unavailable)
        monkeypatch.setattr(monitor_module._LOG, "exception", unavailable)
    with ModelExecutionMonitor(tmp_path) as monitor:
        monitor.set_phase("model_compute")
        monitor.finish("completed")
    assert monitor.status == "completed"


def test_temporarily_missing_observations_do_not_reset_progress_clock(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(monitor_module.time, "monotonic", lambda: clock[0])
    monitor = ModelExecutionMonitor(tmp_path)
    signatures = [("output/memory_stages.jsonl", 100, 1)]
    monkeypatch.setattr(monitor, "_observations", lambda: tuple(signatures))
    monitor.sample()
    clock[0] = 2000
    signatures.clear()
    assert monitor.sample()["seconds_without_observed_progress"] == 2000
    clock[0] = 2001
    signatures.append(("output/memory_stages.jsonl", 100, 1))
    assert monitor.sample()["seconds_without_observed_progress"] == 2001
    monitor.set_phase("waiting_for_prepared_data")  # A repeated phase is also a heartbeat.
    assert monitor.sample()["seconds_without_observed_progress"] == 2001
    clock[0] = 2002
    signatures[0] = ("output/memory_stages.jsonl", 200, 2)
    assert monitor.sample()["seconds_without_observed_progress"] == 0


def test_stuck_observer_does_not_block_cancellation(tmp_path):
    waits = []

    class StuckThread:
        def join(self, *, timeout):
            waits.append(timeout)

        def is_alive(self):
            return True

    monitor = ModelExecutionMonitor(tmp_path)
    monitor.thread = StuckThread()
    assert monitor.__exit__(KeyboardInterrupt, KeyboardInterrupt(), None) is None
    assert monitor.status == "interrupted"
    assert waits == [monitor_module.SHUTDOWN_WAIT_SECONDS]


def test_observer_does_not_swallow_administrative_interrupts(tmp_path, monkeypatch):
    def interrupted():
        raise KeyboardInterrupt("stop requested")

    monitor = ModelExecutionMonitor(tmp_path)
    monkeypatch.setattr(monitor, "_observations", interrupted)
    with pytest.raises(KeyboardInterrupt, match="stop requested"):
        monitor.sample()
