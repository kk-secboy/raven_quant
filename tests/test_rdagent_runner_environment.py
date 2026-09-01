from __future__ import annotations

import ast
import importlib.util
import json
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from quant_platform.feature_set_registry import get_feature_set
from quant_platform.strategy_proposal import (
    STRATEGY_PROPOSAL_VERSION,
    strategy_proposal_json_contract,
    validate_strategy_proposal,
)
from quant_platform.strategy_recipes import get_strategy_recipe
from quant_platform.strategy_rule_compiler import compile_strategy_proposal

pytestmark = pytest.mark.no_database


def _runner_module():
    path = Path(__file__).parents[1] / "scripts" / "run_rdagent_scenario.py"
    spec = importlib.util.spec_from_file_location("run_rdagent_scenario", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strategy_draft_adapter() -> dict[str, Any]:
    """Load only the pure draft adapter without importing optional RD-Agent."""

    path = Path(__file__).parents[1] / "src" / "quant_platform" / "rdagent_strategy.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {
        "_PLATFORM_OWNED_PROPOSAL_FIELDS",
        "_MODEL_OWNED_PROPOSAL_FIELDS",
        "_no_duplicate_proposal_object",
        "_bind_generated_strategy_proposal_json",
        "_generated_strategy_proposal_contract",
    }
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (
                [node.target]
                if isinstance(node, ast.AnnAssign)
                else list(node.targets)
            )
            if any(isinstance(target, ast.Name) and target.id in names for target in targets):
                selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            selected.append(node)
    namespace = {
        "Any": Any,
        "Mapping": Mapping,
        "deepcopy": deepcopy,
        "json": json,
        "strategy_proposal_json_contract": strategy_proposal_json_contract,
        "validate_strategy_proposal": validate_strategy_proposal,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _short_strategy_seed() -> dict[str, Any]:
    from quant_platform.cost_model import COST_SCHEDULE_VERSION

    recipe = get_strategy_recipe("short_relative_strength")
    slots = deepcopy(recipe["strategy_rule_ir"]["slots"])
    weights = slots["alpha_rank"]["components"][0]["parameters"]["weights"]
    first, second = list(weights)[:2]
    weights[first] += 0.05
    weights[second] -= 0.05
    return {
        "contract_version": STRATEGY_PROPOSAL_VERSION,
        "delivery_status": "research_only",
        "name": "short governed challenger",
        "description": "A research-only policy challenger.",
        "horizon": "short_1_5d",
        "economic_hypothesis": "A changed ranking may improve robustness.",
        "baseline_recipe_id": "short_relative_strength",
        "baseline_recipe_version": recipe["version"],
        "baseline_rules_sha256": recipe["strategy_rule_ir"]["rules_sha256"],
        "parent_strategy_version_id": None,
        "changed_slots": ["alpha_rank"],
        "data_contract": {
            "dataset_snapshot_id": "b" * 64,
            "feature_set_id": "governed-baseline",
            "feature_set_definition_sha256": "a" * 64,
            "research_periods": {
                "train_start": "2008-01-02",
                "train_end": "2018-12-28",
                "valid_start": "2019-01-02",
                "valid_end": "2021-12-31",
                "test_start": "2022-01-04",
                "test_end": "2024-12-31",
            },
            "decision_frequency": "day",
            "label_horizon_trading_days": 5,
        },
        "evaluation_contract": {
            "benchmark": "SH000300",
            "primary_metric": "after_cost_information_ratio",
            "cost_schedule_version": COST_SCHEDULE_VERSION,
            "rolling_folds": 5,
            "minimum_oos_observations": 252,
            "final_oos_visible_during_selection": False,
        },
        "slots": slots,
    }


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
    # Pinned RD-Agent CoSTEER forwards extra kwargs to Developer.__init__(scen).
    # Keep the local adapter aligned with the upstream Factor/ModelCoSTEER pattern.
    assert "scen=scenario" in source
    assert 'self.coder.develop(prev_out["proposal"])' in source
    assert "class StrategyProposalEvaluator(RAGEvaluator):" in source
    assert "previous_deterministic_validation" in source
    assert "knowledge_self_gen=False" in source
    assert "TemporaryDirectory(" in source
    assert "compile_strategy_proposal(" in source
    assert "for attempt in range(1, 4)" not in source


def test_strategy_draft_adapter_binds_only_platform_fields_and_stays_strict() -> None:
    adapter = _strategy_draft_adapter()
    bind = adapter["_bind_generated_strategy_proposal_json"]
    seed = _short_strategy_seed()
    generated = {
        field: deepcopy(seed[field])
        for field in (
            "name",
            "description",
            "economic_hypothesis",
            "changed_slots",
            "slots",
        )
    }

    bound = bind(json.dumps(generated), seed=seed)
    for field in adapter["_PLATFORM_OWNED_PROPOSAL_FIELDS"]:
        assert bound[field] == seed[field]

    copied = deepcopy(seed)
    copied["horizon"] = "long_1_3y"
    copied["delivery_status"] = "recommendation_enabled"
    copied["evaluation_contract"]["final_oos_visible_during_selection"] = True
    assert bind(json.dumps(copied), seed=seed) == bound

    copied["python_code"] = "place_order()"
    with pytest.raises(ValueError, match="top-level contract drifted"):
        bind(json.dumps(copied), seed=seed)

    duplicate = json.dumps(generated).replace(
        '"name":', '"name":"duplicate", "name":', 1
    )
    with pytest.raises(ValueError, match="duplicate key"):
        bind(duplicate, seed=seed)


def test_strategy_draft_adapter_preserves_rule_and_model_alpha_fail_closed_checks() -> None:
    adapter = _strategy_draft_adapter()
    bind = adapter["_bind_generated_strategy_proposal_json"]
    seed = _short_strategy_seed()
    generated = {
        field: deepcopy(seed[field])
        for field in adapter["_MODEL_OWNED_PROPOSAL_FIELDS"]
    }
    generated["slots"]["entry_timing"]["components"][0]["component"] = (
        "llm_python_signal"
    )
    bound = bind(json.dumps(generated), seed=seed)
    weights = seed["slots"]["alpha_rank"]["components"][0]["parameters"][
        "weights"
    ]
    with pytest.raises(ValueError, match="is not allowed in slot entry_timing"):
        compile_strategy_proposal(bound, allowed_factor_ids=set(weights))

    # The real public validator verifies the complete signal binding before this
    # branch. Replace only that dependency here to isolate the adapter's final
    # model-score immutability check without importing optional RD-Agent.
    model_seed = deepcopy(seed)
    model_seed["data_contract"]["research_signal_binding"] = {
        "signal_source": "model_prediction"
    }
    adapter["validate_strategy_proposal"] = lambda value: deepcopy(dict(value))
    generated = {
        field: deepcopy(model_seed[field])
        for field in adapter["_MODEL_OWNED_PROPOSAL_FIELDS"]
    }
    alpha = generated["slots"]["alpha_rank"]["components"][0]["parameters"][
        "weights"
    ]
    first, second = list(alpha)[:2]
    alpha[first] -= 0.01
    alpha[second] += 0.01
    with pytest.raises(ValueError, match="changed alpha_rank for a frozen model score"):
        bind(json.dumps(generated), seed=model_seed)

    del generated["slots"]["alpha_rank"]
    with pytest.raises(ValueError, match="changed alpha_rank for a frozen model score"):
        bind(json.dumps(generated), seed=model_seed)


def test_strategy_generation_contract_and_workspace_round_trip_are_host_owned() -> None:
    adapter = _strategy_draft_adapter()
    contract = adapter["_generated_strategy_proposal_contract"]()
    assert contract["model_output_fields"] == list(
        adapter["_MODEL_OWNED_PROPOSAL_FIELDS"]
    )
    assert contract["platform_owned_fields"] == list(
        adapter["_PLATFORM_OWNED_PROPOSAL_FIELDS"]
    )
    assert "top_level_fields" not in contract

    source = (
        Path(__file__).parents[1] / "src" / "quant_platform" / "rdagent_strategy.py"
    ).read_text(encoding="utf-8")
    assert source.count("_bind_generated_strategy_proposal_json(") >= 3
    assert 'bound_payload.pop("proposal_sha256", None)' in source


def test_fin_quant_coverage_policy_attempts_both_arms_then_returns_to_bandit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path(__file__).parents[1] / "scripts" / "run_rdagent_module.py"
    spec = importlib.util.spec_from_file_location("run_rdagent_module", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    class Controller:
        def __init__(self) -> None:
            self.recorded: list[str] = []
            self.official_draws = 0

        def record(self, _metric: object, arm: str) -> None:
            self.recorded.append(arm)

        def decide(self, _metric: object) -> str:
            self.official_draws += 1
            return "factor"

    class HypothesisGenerator:
        targets = "factor"

        def convert_response(self, response: str) -> SimpleNamespace:
            return SimpleNamespace(action=json.loads(response)["action"])

    modules = {
        "rdagent": ModuleType("rdagent"),
        "rdagent.app": ModuleType("rdagent.app"),
        "rdagent.app.qlib_rd_loop": ModuleType("rdagent.app.qlib_rd_loop"),
        "rdagent.scenarios": ModuleType("rdagent.scenarios"),
        "rdagent.scenarios.qlib": ModuleType("rdagent.scenarios.qlib"),
        "rdagent.scenarios.qlib.proposal": ModuleType(
            "rdagent.scenarios.qlib.proposal"
        ),
        "rdagent.app.qlib_rd_loop.conf": ModuleType(
            "rdagent.app.qlib_rd_loop.conf"
        ),
        "rdagent.scenarios.qlib.proposal.bandit": ModuleType(
            "rdagent.scenarios.qlib.proposal.bandit"
        ),
        "rdagent.scenarios.qlib.proposal.quant_proposal": ModuleType(
            "rdagent.scenarios.qlib.proposal.quant_proposal"
        ),
    }
    modules["rdagent.app.qlib_rd_loop.conf"].QUANT_PROP_SETTING = SimpleNamespace(
        action_selection="bandit"
    )
    modules["rdagent.scenarios.qlib.proposal.bandit"].EnvController = Controller
    modules[
        "rdagent.scenarios.qlib.proposal.quant_proposal"
    ].QlibQuantHypothesisGen = HypothesisGenerator
    for name, module in modules.items():
        if "." not in name or name.rsplit(".", 1)[1] in {
            "app",
            "qlib_rd_loop",
            "scenarios",
            "qlib",
            "proposal",
        }:
            module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)

    runner._enable_fin_quant_arm_coverage()
    controller = Controller()
    controller.record(object(), "factor")
    assert controller.decide(object()) == "model"
    controller.record(object(), "model")
    assert controller.decide(object()) == "factor"
    assert controller.recorded == ["factor", "model"]
    assert controller.official_draws == 2

    generator = HypothesisGenerator()
    generator.targets = "model"
    assert generator.convert_response('{"action":"factor"}').action == "model"
