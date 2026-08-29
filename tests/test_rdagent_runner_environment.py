from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from quant_platform.feature_set_registry import get_feature_set

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

    degraded = run_rdagent_module._costeer_knowledge_status(
        {}, module="rdagent.app.qlib_rd_loop.factor"
    )
    assert degraded["status"] == "degraded_empty_retrieval"
    assert degraded["empty_knowledge_forced"] is True
    configured = run_rdagent_module._costeer_knowledge_status(
        {"EMBEDDING_OPENAI_API_KEY": "embedding-key"},
        module="quant_platform.rdagent_strategy",
    )
    assert configured["status"] == "embedding_retrieval_configured"
    assert configured["costeer_used"] is True
    assert configured["empty_knowledge_forced"] is False
    assert configured["strategy_codegen_used"] is True
    assert configured["strategy_codegen_target"] == (
        "allowlisted_rule_ir_and_contract_tests"
    )
    assert configured["strategy_compiler"] == "deterministic_allowlist"

    strategy_degraded = run_rdagent_module._costeer_knowledge_status(
        {}, module="quant_platform.rdagent_strategy"
    )
    assert strategy_degraded["costeer_used"] is True
    assert strategy_degraded["empty_knowledge_forced"] is True
    assert strategy_degraded["strategy_codegen_used"] is True


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


def test_missing_strategy_embeddings_use_typed_empty_costeer_knowledge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_rdagent_module

    class FakeCoSTEER:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeQueriedKnowledge:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeRAGStrategy:
        pass

    module_names = [
        "rdagent",
        "rdagent.components",
        "rdagent.components.coder",
    ]
    for name in module_names:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    costeer_module = ModuleType("rdagent.components.coder.CoSTEER")
    costeer_module.CoSTEER = FakeCoSTEER
    knowledge_module = ModuleType(
        "rdagent.components.coder.CoSTEER.knowledge_management"
    )
    knowledge_module.CoSTEERQueriedKnowledgeV2 = FakeQueriedKnowledge
    knowledge_module.CoSTEERRAGStrategyV2 = FakeRAGStrategy
    monkeypatch.setitem(sys.modules, costeer_module.__name__, costeer_module)
    monkeypatch.setitem(sys.modules, knowledge_module.__name__, knowledge_module)

    run_rdagent_module._disable_optional_costeer_embeddings()

    coder = FakeCoSTEER(with_knowledge=False, knowledge_self_gen=True)
    assert coder.kwargs["with_knowledge"] is True
    assert coder.kwargs["knowledge_self_gen"] is False
    task = SimpleNamespace(get_task_information=lambda: "governed-strategy-task")
    knowledge = FakeRAGStrategy().query(SimpleNamespace(sub_tasks=[task]), [])
    assert knowledge.success_task_to_knowledge_dict == {}
    assert knowledge.failed_task_info_set == set()
    assert knowledge.task_to_former_failed_traces == {
        "governed-strategy-task": ([], None)
    }


@pytest.mark.parametrize(
    ("scenario", "module"),
    [
        ("fin_factor", "rdagent.app.qlib_rd_loop.factor"),
        ("fin_strategy", "quant_platform.rdagent_strategy"),
    ],
)
def test_factor_and_strategy_commands_bind_verified_base_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    module: str,
) -> None:
    runner = _runner_module()
    feature_set = get_feature_set("governed-baseline")
    root = tmp_path / "base-features"
    root.mkdir()
    (root / "base_factors.json").write_text(
        json.dumps(feature_set["features"]), encoding="utf-8"
    )
    (root / "definition.json").write_text(json.dumps(feature_set), encoding="utf-8")
    monkeypatch.setattr(runner, "_require_preloaded_sandbox", lambda **kwargs: None)
    args = SimpleNamespace(
        scenario=scenario,
        asset=[],
        scenario_options=None,
        feature_set_id=feature_set["id"],
        feature_set_sha256=feature_set["definition_sha256"],
        base_features=str(root),
        command="rdagent",
        loop_n=3,
        duration="30m",
    )

    command = runner._scenario_command(args)

    assert module in command
    assert command[command.index("--base_features_path") + 1] == str(root.resolve())


def test_strategy_bridge_reverifies_staged_feature_members(
    tmp_path: Path,
) -> None:
    bridge_path = Path(__file__).parents[1] / "scripts" / "rdagent_bridge.py"
    spec = importlib.util.spec_from_file_location("rdagent_bridge_features", bridge_path)
    assert spec is not None and spec.loader is not None
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    feature_set = get_feature_set("governed-baseline")
    root = tmp_path / "base-features"
    root.mkdir()
    (root / "base_factors.json").write_text(
        json.dumps(feature_set["features"]), encoding="utf-8"
    )
    (root / "definition.json").write_text(json.dumps(feature_set), encoding="utf-8")
    args = SimpleNamespace(
        scenario="fin_strategy",
        base_features=str(root),
        feature_set_id=feature_set["id"],
        feature_set_sha256=feature_set["definition_sha256"],
    )

    assert bridge._strategy_feature_ids(args) == set(feature_set["features"])

    (root / "base_factors.json").write_text('{"TAMPERED":"$close"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence disagrees"):
        bridge._strategy_feature_ids(args)


def test_strategy_loop_is_horizon_directed_and_keeps_the_incumbent_frozen() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "quant_platform" / "rdagent_strategy.py"
    ).read_text(encoding="utf-8")

    assert 'os.environ.get("QUANTLAB_STRATEGY_HORIZON")' in source
    assert 'os.environ.get("QUANTLAB_STRATEGY_PARENT_VERSION_ID")' in source
    assert "horizon = self.horizon" in source
    assert "parent_strategy_version_id=self.parent_strategy_version_id" in source
    assert "loop_id % len(_HORIZON_BASELINES)" not in source
    assert '"long_1_3y": 756' in source


def test_strategy_loop_uses_official_costeer_for_governed_ir_repairs() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "quant_platform" / "rdagent_strategy.py"
    ).read_text(encoding="utf-8")

    assert "class StrategyProposalCoSTEER(CoSTEER):" in source
    assert 'self.coder.develop(prev_out["proposal"])' in source
    assert "class StrategyProposalEvaluator(RAGEvaluator):" in source
    assert "previous_deterministic_validation" in source
    assert "knowledge_self_gen=False" in source
    assert "TemporaryDirectory(" in source
    assert "compile_strategy_proposal(" in source
    assert "for attempt in range(1, 4)" not in source
