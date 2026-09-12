from __future__ import annotations

import json
import os
import sys
from copy import copy
from types import ModuleType, SimpleNamespace

import pytest

from scripts import run_rdagent_module, run_rdagent_scenario

pytestmark = pytest.mark.no_database


def test_streaming_without_outer_deadline_still_checks_exit_and_redacts(monkeypatch, capsys):
    monkeypatch.setenv("TEST_RUN_API_KEY", "private-value-123")
    run_rdagent_scenario._run_streaming_redacted(
        [sys.executable, "-c", "import time; time.sleep(.2); print('private-value-123')"],
        timeout=None,
        env=dict(os.environ),
    )
    assert capsys.readouterr().out == "[REDACTED]\n"
    with pytest.raises(run_rdagent_scenario.subprocess.CalledProcessError):
        run_rdagent_scenario._run_streaming_redacted(
            [sys.executable, "-c", "raise SystemExit(7)"], timeout=None, env=dict(os.environ)
        )


def test_bounded_helpers_still_time_out():
    with pytest.raises(run_rdagent_scenario.subprocess.TimeoutExpired):
        run_rdagent_scenario._run_streaming_redacted(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            timeout=0,
            env=dict(os.environ),
        )


@pytest.mark.parametrize("scenario,expected", [("fin_quant", None), ("fin_strategy", 3900)])
def test_only_fin_quant_drains_before_export(monkeypatch, scenario, expected):
    calls = []
    monkeypatch.setattr(run_rdagent_scenario, "_scenario_command", lambda args: ["research"])
    monkeypatch.setattr(
        run_rdagent_scenario, "_run_streaming_redacted",
        lambda command, **kwargs: calls.append((command, kwargs["timeout"])),
    )
    args = SimpleNamespace(
        scenario=scenario, duration="1h", trace="trace", result="result", bridge="bridge",
        asset_id=[], loop_n=10, feature_set_id=None, feature_set_sha256=None, base_features=None,
    )
    run_rdagent_scenario.run(args)
    assert calls[0] == (["research"], expected)
    assert calls[1][1] == 300


def _install_fake_runtime(monkeypatch):
    class Conf(SimpleNamespace):
        def model_copy(self, *, update):
            result = copy(self)
            result.__dict__.update(update)
            return result

    shared_conf = Conf(default_entry="qrun conf.yaml", running_timeout_period=3600, mem_limit="16g")

    class Runtime:
        conf = shared_conf
        exit_code = 0

        def prepare(self):
            pass

        def run(self, entry=None, local_path=".", env=None, **kwargs):
            if env and env.get("fail"):
                raise RuntimeError("training failed")
            return SimpleNamespace(
                exit_code=self.exit_code, timeout=self.conf.running_timeout_period,
                mem_limit=self.conf.mem_limit,
            )

    class Workspace:
        def inject_files(self, **files):
            assert "test_fea.py" in files

        def run(self, *, env, entry):
            assert isinstance(env, Runtime)
            assert entry == "python test_fea.py"
            return env.run(entry=entry)

    rd_loop = ModuleType("rdagent.components.workflow.rd_loop")

    class Loop:
        LoopTerminationError = RuntimeError

        def __init__(self):
            self.plan = {"features": {"test": "$close"}}
            self.step_n = None
            self.expired = False

        def _check_exit_conditions_on_step(self, loop_id=None, step_id=None):
            if self.step_n is not None:
                if self.step_n <= 0:
                    raise self.LoopTerminationError("Step count reached")
                self.step_n -= 1
            if self.expired:
                raise self.LoopTerminationError("Timer timeout")

        def _init_base_features(self, path):
            expected = json.loads((path / "base_factors.json").read_text())
            if rd_loop.validate_qlib_features(list(expected.values())):
                self.plan["features"] = expected

    rd_loop.RDLoop = Loop
    modules = {
        "rdagent.components": {}, "rdagent.components.workflow": {"rd_loop": rd_loop},
        "rdagent.core": {}, "rdagent.core.experiment": {"FBWorkspace": Workspace},
        "rdagent.utils": {}, "rdagent.utils.env": {"QTDockerEnv": Runtime},
        "rdagent.utils.qlib": {"TEST_FEATURE_CODE": "validate({experessions})"},
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    run_rdagent_module._enable_fin_quant_execution_compatibility()
    return Runtime, Loop, shared_conf


def test_qlib_training_drains_without_changing_other_execution_limits(monkeypatch):
    runtime, _, original = _install_fake_runtime(monkeypatch)
    instance = runtime()
    assert instance.run("qrun conf.yaml").timeout is None
    assert instance.run().timeout is None
    assert instance.run("qrun conf.yaml").mem_limit == "16g"
    assert instance.conf is original
    assert instance.run("python test_fea.py").timeout == 3600
    with pytest.raises(RuntimeError, match="training failed"):
        instance.run("qrun conf.yaml", env={"fail": True})
    assert instance.conf is original
    assert original.running_timeout_period == 3600


def test_governed_features_use_docker_and_cannot_fall_back_even_if_defaults_match(
    monkeypatch, tmp_path,
):
    runtime, loop, _ = _install_fake_runtime(monkeypatch)
    (tmp_path / "base_factors.json").write_text(json.dumps({"test": "$close"}))
    instance = loop()
    instance._init_base_features(tmp_path)
    assert instance.plan["features"] == {"test": "$close"}
    runtime.exit_code = 1
    with pytest.raises(RuntimeError, match="failed to load"):
        instance._init_base_features(tmp_path)


def test_expired_research_budget_keeps_feedback_and_record_but_stops_new_hypotheses(monkeypatch):
    _, loop, _ = _install_fake_runtime(monkeypatch)
    instance = loop()
    instance._check_exit_conditions_on_step(0, 0)
    instance.expired = True
    instance.step_n = 4
    for step in (1, 2, 3, 4):
        instance._check_exit_conditions_on_step(0, step)
    assert instance.step_n == 0
    with pytest.raises(RuntimeError, match="Step count reached"):
        instance._check_exit_conditions_on_step(0, 4)
    instance.step_n = None
    with pytest.raises(RuntimeError, match="Timer timeout"):
        instance._check_exit_conditions_on_step(1, 0)
