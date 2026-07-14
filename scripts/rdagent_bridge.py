#!/usr/bin/env python3
"""Trusted bridge executed inside the pinned RD-Agent environment.

It emits JSON only. Pickle reading is deliberately kept out of the API process and
must only target trace folders produced by the configured RD-Agent runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _version() -> str:
    for distribution in ("rdagent", "rd-agent"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "source-checkout"


def probe(args: argparse.Namespace) -> dict[str, Any]:
    import rdagent  # noqa: F401
    from pydantic_ai.mcp import MCPServerStreamableHTTP  # noqa: F401
    from pydantic_ai.providers.litellm import LiteLLMProvider  # noqa: F401

    qlib_home = Path(args.qlib_home).expanduser()
    required = [
        qlib_home / "calendars" / "day.txt",
        qlib_home / "instruments" / "cn_all.txt",
        qlib_home / "features",
    ]
    docker_cli = shutil.which("docker")
    docker_available = False
    if docker_cli:
        try:
            docker_available = (
                subprocess.run(
                    [docker_cli, "info"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                ).returncode
                == 0
            )
        except (OSError, subprocess.SubprocessError):
            docker_available = False
    return {
        "status": "ok",
        "version": _version(),
        "python": os.sys.executable,
        "docker_available": docker_available,
        "qlib_data_ready": all(path.exists() for path in required),
        "qlib_home": str(qlib_home),
        "llm_credentials_configured": bool(os.getenv(args.llm_key_env)),
        "llm_key_env": args.llm_key_env,
    }


def _loop_id(tag: str) -> int | None:
    for part in tag.split("."):
        if part.startswith("Loop_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                return None
    return None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def export_trace(args: argparse.Namespace) -> dict[str, Any]:
    from rdagent.log.storage import FileStorage

    trace = Path(args.trace).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    code_root = output.parent / "candidate-code"
    values_root = output.parent / "candidate-values"
    output.parent.mkdir(parents=True, exist_ok=True)
    code_root.mkdir(parents=True, exist_ok=True)
    values_root.mkdir(parents=True, exist_ok=True)

    rounds: dict[int, dict[str, Any]] = {}
    for message in FileStorage(trace).iter_msg():
        loop_id = _loop_id(message.tag)
        if loop_id is None:
            continue
        item = rounds.setdefault(
            loop_id,
            {"loop_id": loop_id, "tasks": [], "codes": {}, "values": {}, "feedback": {}},
        )
        content = message.content
        if "hypothesis generation" in message.tag:
            item["hypothesis"] = {
                "hypothesis": getattr(content, "hypothesis", ""),
                "reason": getattr(content, "reason", ""),
            }
        elif "experiment generation" in message.tag:
            tasks = getattr(content, "sub_tasks", content)
            item["tasks"] = [
                {
                    "name": getattr(task, "factor_name", getattr(task, "name", "unnamed_factor")),
                    "description": getattr(
                        task, "factor_description", getattr(task, "description", "")
                    ),
                    "formulation": getattr(task, "factor_formulation", None),
                    "variables": getattr(task, "variables", {}) or {},
                }
                for task in tasks
                if hasattr(task, "factor_name")
            ]
        elif "evolving code" in message.tag and "running" not in message.tag:
            for workspace in content:
                target = getattr(workspace, "target_task", None)
                name = getattr(target, "factor_name", getattr(target, "name", "unnamed_factor"))
                files = getattr(workspace, "file_dict", {}) or {}
                if "factor.py" in files:
                    item["codes"][name] = files["factor.py"]
                workspace_path = getattr(workspace, "workspace_path", None)
                source_values = Path(workspace_path) / "result.h5" if workspace_path else None
                if source_values and source_values.exists():
                    item["values"][name] = str(source_values)
        elif "evolving feedback" in message.tag and "running" not in message.tag:
            decisions = []
            for feedback in content:
                decisions.append(
                    {
                        "decision": bool(getattr(feedback, "final_decision", False)),
                        "feedback": str(getattr(feedback, "final_feedback", "")),
                    }
                )
            item["implementation_feedback"] = decisions
        elif message.tag.endswith("feedback.feedback") or ".feedback.feedback." in message.tag:
            item["feedback"] = {
                "decision": bool(getattr(content, "decision", False)),
                "reason": str(getattr(content, "reason", "")),
                "hypothesis_evaluation": str(getattr(content, "hypothesis_evaluation", "")),
            }

    candidates: list[dict[str, Any]] = []
    for loop_id, item in sorted(rounds.items()):
        implementation_feedback = item.get("implementation_feedback", [])
        for index, task in enumerate(item["tasks"]):
            name = task["name"]
            code = item["codes"].get(name)
            code_path = None
            values_path = None
            code_sha256 = None
            if code:
                safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:80]
                destination = code_root / f"loop-{loop_id:03d}-{safe_name}.py"
                destination.write_text(code, encoding="utf-8")
                code_path = str(destination)
                code_sha256 = _sha256(code)
            source_values = item["values"].get(name)
            if source_values:
                safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:80]
                values_destination = values_root / f"loop-{loop_id:03d}-{safe_name}.h5"
                shutil.copy2(source_values, values_destination)
                values_path = str(values_destination)
            implementation = (
                implementation_feedback[index] if index < len(implementation_feedback) else {}
            )
            hypothesis_feedback = item.get("feedback", {})
            candidates.append(
                {
                    **task,
                    "source_iteration": loop_id,
                    "code_path": code_path,
                    "values_path": values_path,
                    "code_sha256": code_sha256,
                    "rdagent_decision": implementation.get("decision"),
                    "rdagent_feedback": implementation.get("feedback")
                    or hypothesis_feedback.get("hypothesis_evaluation")
                    or hypothesis_feedback.get("reason"),
                    "hypothesis": item.get("hypothesis"),
                }
            )

    result = {
        "status": "ok",
        "trace_path": str(trace),
        "rounds": len(rounds),
        "candidates": candidates,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--qlib-home", default="~/.qlib/qlib_data/cn_data")
    probe_parser.add_argument("--llm-key-env", default="OPENAI_API_KEY")
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--trace", required=True)
    export_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = probe(args) if args.command == "probe" else export_trace(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
