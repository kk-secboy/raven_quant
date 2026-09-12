#!/usr/bin/env python3
"""Launch a pinned RD-Agent module with governed optional-RAG behavior."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any


def _embedding_is_configured(env: dict[str, str]) -> bool:
    return bool(
        str(env.get("EMBEDDING_OPENAI_API_KEY") or "").strip()
        or str(env.get("EMBEDDING_AZURE_API_BASE") or "").strip()
        or str(env.get("HOSTED_VLLM_API_KEY") or "").strip()
    )


def _costeer_knowledge_status(env: dict[str, str], *, module: str) -> dict[str, Any]:
    configured = _embedding_is_configured(env)
    strategy_compiler = module == "quant_platform.rdagent_strategy"
    return {
        "contract_version": "costeer-knowledge-status-v2",
        "status": "embedding_retrieval_configured" if configured else "unconfigured_fail_closed",
        "embedding_retrieval_configured": configured,
        "retrieval_mode": "embedding_rag" if configured else "unconfigured_fail_closed",
        "costeer_used": True,
        "strategy_codegen_used": True if strategy_compiler else None,
        "strategy_codegen_target": (
            "allowlisted_rule_ir_and_contract_tests" if strategy_compiler else None
        ),
        "strategy_compiler": "deterministic_allowlist" if strategy_compiler else None,
    }


def _enable_qlib_file_tracking_compatibility() -> None:
    """Allow RD-Agent's disposable Qlib containers to use local MLflow state."""

    from rdagent.utils.env import DockerEnv

    original_run = DockerEnv._run

    def governed_run(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        bound = inspect.signature(original_run).bind(self, *args, **kwargs)
        child_env = dict(bound.arguments.get("env") or {})
        # Docker assigns the disposable Qlib container its own HOSTNAME.  Do
        # not leak the parent research worker's container id into it: the
        # factor sandbox uses its real hostname to inspect the governed mount
        # on the sibling Docker daemon.
        child_env.pop("HOSTNAME", None)
        child_env.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        bound.arguments["env"] = child_env
        return original_run(*bound.args, **bound.kwargs)

    DockerEnv._run = governed_run  # type: ignore[method-assign]


_FIN_QUANT_ARMS = ("factor", "model")


def _enable_fin_quant_execution_compatibility() -> None:
    """Use the pinned Docker runtime and drain admitted Qlib training steps."""
    from rdagent.components.workflow import rd_loop
    from rdagent.core.experiment import FBWorkspace
    from rdagent.utils.env import QTDockerEnv
    from rdagent.utils.qlib import TEST_FEATURE_CODE

    if getattr(QTDockerEnv, "_quantlab_fin_quant_execution_enabled", False):
        return
    original_run = QTDockerEnv.run
    original_init_features = rd_loop.RDLoop._init_base_features
    original_check_step = rd_loop.RDLoop._check_exit_conditions_on_step
    validation_succeeded = False

    def governed_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = inspect.signature(original_run).bind(self, *args, **kwargs)
        entry = bound.arguments.get("entry") or self.conf.default_entry
        if not str(entry).startswith("qrun "):
            return original_run(*bound.args, **bound.kwargs)
        original_conf = self.conf
        # QTDockerEnv's default config is shared. Copy it so validation and
        # code execution retain their bounded timeout and resource limits.
        self.conf = original_conf.model_copy(update={"running_timeout_period": None})
        try:
            return original_run(*bound.args, **bound.kwargs)
        finally:
            self.conf = original_conf

    def validate_features(expressions: list[str]) -> bool:
        nonlocal validation_succeeded
        workspace = FBWorkspace()
        workspace.inject_files(
            **{"test_fea.py": TEST_FEATURE_CODE.format(experessions=str(expressions))}
        )
        runtime = QTDockerEnv()
        runtime.prepare()
        result = workspace.run(env=runtime, entry="python test_fea.py")
        validation_succeeded = result.exit_code == 0
        return validation_succeeded

    def init_features(self: Any, base_features_path: str | None) -> None:
        nonlocal validation_succeeded
        validation_succeeded = False
        original_init_features(self, base_features_path)
        if base_features_path is None:
            raise RuntimeError("fin_quant requires governed base features")
        expected = json.loads(
            (Path(base_features_path) / "base_factors.json").read_text(encoding="utf-8")
        )
        if not validation_succeeded or self.plan.get("features") != expected:
            raise RuntimeError("fin_quant failed to load the governed base features")

    def check_step(self: Any, loop_id: int | None = None, step_id: int | None = None) -> None:
        # The pinned loop checks its timer before *every* step. After a long
        # training step this would drop feedback/record and lose the result.
        # Stop admitting hypotheses when time is up, but finish an admitted
        # iteration, retaining the upstream explicit step/loop count limits.
        if step_id is None or step_id == 0:
            return original_check_step(self, loop_id, step_id)
        if self.step_n is not None:
            if self.step_n <= 0:
                raise self.LoopTerminationError("Step count reached")
            self.step_n -= 1

    QTDockerEnv.run = governed_run  # type: ignore[method-assign]
    QTDockerEnv._quantlab_fin_quant_execution_enabled = True
    rd_loop.validate_qlib_features = validate_features
    rd_loop.RDLoop._init_base_features = init_features  # type: ignore[method-assign]
    rd_loop.RDLoop._check_exit_conditions_on_step = check_step  # type: ignore[method-assign]


def _next_missing_fin_quant_arm(attempted: set[str]) -> str | None:
    """Return the next required arm in the deterministic coverage order."""

    return next((arm for arm in _FIN_QUANT_ARMS if arm not in attempted), None)


def _enable_fin_quant_arm_coverage() -> None:
    """Give both official fin_quant arms one attempt before normal bandit choice.

    The pinned upstream loop starts with ``factor`` and subsequently delegates
    to its Thompson-sampling controller.  With the platform's former one-loop
    budget that made a factor/model joint experiment impossible; even with a
    larger budget the bandit was free to keep selecting the same arm.  This
    adapter leaves the upstream loop, Trace, reward updates and bandit draw in
    place.  It only overrides a draw while one arm has never been attempted in
    this run, then returns control to the official choice.
    """

    from rdagent.app.qlib_rd_loop.conf import QUANT_PROP_SETTING
    from rdagent.scenarios.qlib.proposal.bandit import EnvController
    from rdagent.scenarios.qlib.proposal.quant_proposal import (
        QlibQuantHypothesisGen,
    )

    if QUANT_PROP_SETTING.action_selection != "bandit":
        raise RuntimeError(
            "governed fin_quant arm coverage requires the pinned official bandit"
        )
    if getattr(EnvController, "_quantlab_arm_coverage_enabled", False):
        return

    original_record = EnvController.record
    original_decide = EnvController.decide
    original_convert_response = QlibQuantHypothesisGen.convert_response

    def governed_record(self: Any, metric: Any, arm: str) -> None:
        if arm not in _FIN_QUANT_ARMS:
            raise RuntimeError(f"official fin_quant selected an unsupported arm: {arm}")
        attempted = set(getattr(self, "_quantlab_attempted_arms", set()))
        attempted.add(arm)
        self._quantlab_attempted_arms = attempted
        original_record(self, metric, arm)

    def governed_decide(self: Any, metric: Any) -> str:
        # Always execute the official draw so controller/RNG behavior resumes
        # from the state it would have had without this bounded coverage rule.
        official_action = original_decide(self, metric)
        if official_action not in _FIN_QUANT_ARMS:
            raise RuntimeError(
                f"official fin_quant bandit selected an unsupported arm: {official_action}"
            )
        attempted = set(getattr(self, "_quantlab_attempted_arms", set()))
        forced_action = _next_missing_fin_quant_arm(attempted)
        self._quantlab_last_action_source = (
            "coverage_policy" if forced_action is not None else "official_bandit"
        )
        return forced_action or official_action

    def governed_convert_response(self: Any, response: str) -> Any:
        hypothesis = original_convert_response(self, response)
        selected_action = str(getattr(self, "targets", ""))
        if selected_action not in _FIN_QUANT_ARMS:
            raise RuntimeError("fin_quant hypothesis generator lost its selected arm")
        # Upstream asks the LLM to echo the action, but does not verify the echo.
        # The host-owned action choice must remain authoritative.
        hypothesis.action = selected_action
        return hypothesis

    EnvController.record = governed_record  # type: ignore[method-assign]
    EnvController.decide = governed_decide  # type: ignore[method-assign]
    EnvController._quantlab_arm_coverage_enabled = True
    QlibQuantHypothesisGen.convert_response = governed_convert_response  # type: ignore[method-assign]


def main(argv: list[str]) -> int:
    if len(argv) < 2 or not (
        argv[1].startswith("rdagent.app.")
        or argv[1] == "quant_platform.rdagent_strategy"
    ):
        raise SystemExit("usage: run_rdagent_module.py ALLOWLISTED_RDAGENT_MODULE [ARGS...]")
    module = argv[1]
    os.environ["QUANTLAB_COSTEER_KNOWLEDGE_STATUS_JSON"] = json.dumps(
        _costeer_knowledge_status(dict(os.environ), module=module),
        sort_keys=True,
        separators=(",", ":"),
    )
    if module != "quant_platform.rdagent_strategy":
        _enable_qlib_file_tracking_compatibility()
    if not _embedding_is_configured(dict(os.environ)):
        raise SystemExit(
            "embedding retrieval is not configured; set the llm secret record's "
            "embedding_api_key / embedding_api_base / embedding_model fields first"
        )
    target = importlib.import_module(module)
    if module == "rdagent.app.qlib_rd_loop.quant":
        _enable_fin_quant_execution_compatibility()
        _enable_fin_quant_arm_coverage()
    entry = getattr(target, "main", None)
    if not callable(entry):
        raise RuntimeError(f"RD-Agent module has no callable main: {module}")
    import fire

    sys.argv = [module, *argv[2:]]
    fire.Fire(entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
