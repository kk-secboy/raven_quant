from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.no_database


def _runner_module():
    path = Path(__file__).parents[1] / "scripts" / "run_rdagent_scenario.py"
    spec = importlib.util.spec_from_file_location("run_rdagent_scenario", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openai_compatible_model_gets_explicit_litellm_provider() -> None:
    runner = _runner_module()
    env = {"CHAT_MODEL": "deepseek-v4-flash", "OPENAI_API_BASE": "https://llm.invalid/v1"}
    runner._normalize_openai_compatible_model(env)
    assert env["CHAT_MODEL"] == "openai/deepseek-v4-flash"


def test_explicit_litellm_provider_is_preserved() -> None:
    runner = _runner_module()
    env = {"CHAT_MODEL": "openai/custom", "OPENAI_API_BASE": "https://llm.invalid/v1"}
    runner._normalize_openai_compatible_model(env)
    assert env["CHAT_MODEL"] == "openai/custom"


def test_embedding_provider_detection_requires_a_real_embedding_endpoint() -> None:
    from scripts import run_rdagent_module

    assert not run_rdagent_module._embedding_is_configured({})
    assert not run_rdagent_module._embedding_is_configured(
        {"OPENAI_API_KEY": "chat-only", "OPENAI_API_BASE": "https://chat.invalid"}
    )
    assert run_rdagent_module._embedding_is_configured(
        {"EMBEDDING_OPENAI_API_KEY": "embedding-key"}
    )


def test_disposable_qlib_container_allows_local_mlflow_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_rdagent_module

    class FakeDockerEnv:
        def _run(self, entry=None, local_path=".", env=None, **kwargs):
            return env

    rdagent = ModuleType("rdagent")
    utils = ModuleType("rdagent.utils")
    env_module = ModuleType("rdagent.utils.env")
    env_module.DockerEnv = FakeDockerEnv
    monkeypatch.setitem(sys.modules, "rdagent", rdagent)
    monkeypatch.setitem(sys.modules, "rdagent.utils", utils)
    monkeypatch.setitem(sys.modules, "rdagent.utils.env", env_module)

    run_rdagent_module._enable_qlib_file_tracking_compatibility()

    assert FakeDockerEnv()._run(env={"EXISTING": "yes", "HOSTNAME": "parent-worker"}) == {
        "EXISTING": "yes",
        "MLFLOW_ALLOW_FILE_STORE": "true",
    }
    assert FakeDockerEnv()._run("python test.py", "/tmp/work", {"POSITIONAL": "yes"}) == {
        "POSITIONAL": "yes",
        "MLFLOW_ALLOW_FILE_STORE": "true",
    }
