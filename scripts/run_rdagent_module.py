#!/usr/bin/env python3
"""Launch a pinned RD-Agent module with governed optional-RAG behavior."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from typing import Any


def _embedding_is_configured(env: dict[str, str]) -> bool:
    return bool(
        str(env.get("EMBEDDING_OPENAI_API_KEY") or "").strip()
        or str(env.get("EMBEDDING_AZURE_API_BASE") or "").strip()
    )


def _disable_optional_costeer_embeddings() -> None:
    """Keep CoSTEER coding active while replacing unavailable RAG retrieval."""

    from rdagent.components.coder.CoSTEER import CoSTEER
    from rdagent.components.coder.CoSTEER.knowledge_management import (
        CoSTEERQueriedKnowledgeV2,
        CoSTEERRAGStrategyV2,
    )

    original_init = CoSTEER.__init__

    def governed_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # RD-Agent's default multiprocessing coding strategy requires a
        # CoSTEERQueriedKnowledge object even when no reusable knowledge exists.
        # Supplying an empty, typed result preserves normal LLM code generation
        # without calling a chat-only provider's nonexistent embedding endpoint.
        kwargs["with_knowledge"] = True
        kwargs["knowledge_self_gen"] = False
        original_init(self, *args, **kwargs)

    def empty_query(
        _strategy: Any, evo: Any, _evolving_trace: Any
    ) -> CoSTEERQueriedKnowledgeV2:
        task_information = [task.get_task_information() for task in evo.sub_tasks]
        return CoSTEERQueriedKnowledgeV2(
            success_task_to_knowledge_dict={},
            failed_task_info_set=set(),
            task_to_former_failed_traces={key: ([], None) for key in task_information},
            task_to_similar_task_successful_knowledge={
                key: [] for key in task_information
            },
            task_to_similar_error_successful_knowledge={
                key: [] for key in task_information
            },
        )

    CoSTEER.__init__ = governed_init  # type: ignore[method-assign]
    CoSTEERRAGStrategyV2.query = empty_query  # type: ignore[method-assign]


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


def main(argv: list[str]) -> int:
    if len(argv) < 2 or not argv[1].startswith("rdagent.app."):
        raise SystemExit("usage: run_rdagent_module.py RDAGENT_MODULE [ARGS...]")
    module = argv[1]
    _enable_qlib_file_tracking_compatibility()
    if not _embedding_is_configured(dict(os.environ)):
        _disable_optional_costeer_embeddings()
    target = importlib.import_module(module)
    entry = getattr(target, "main", None)
    if not callable(entry):
        raise RuntimeError(f"RD-Agent module has no callable main: {module}")
    import fire

    sys.argv = [module, *argv[2:]]
    fire.Fire(entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
