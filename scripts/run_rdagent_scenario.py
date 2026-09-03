#!/usr/bin/env python3
"""Run one allowlisted RD-Agent scenario and export a sanitized result."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from quant_platform.feature_set_registry import resolve_feature_set
from quant_platform.rdagent_scenarios import get_rdagent_scenario


def _duration_seconds(value: str) -> int:
    amount = int(value[:-1])
    return amount * (60 if value.endswith("m") else 3600)


def _normalize_openai_compatible_model(env: dict[str, str]) -> None:
    """Give LiteLLM an explicit provider for OpenAI-compatible endpoints."""

    model = str(env.get("CHAT_MODEL") or "").strip()
    if model and "/" not in model and str(env.get("OPENAI_API_BASE") or "").strip():
        env["CHAT_MODEL"] = f"openai/{model}"


def _read_options(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    source = Path(path).resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError("scenario options must be a string-to-string object")
    return value


def _require_path(value: str, *, directory: bool | None = None) -> str:
    path = Path(value).resolve(strict=True)
    if directory is True and not path.is_dir():
        raise ValueError(f"scenario input is not a directory: {path}")
    if directory is False and not path.is_file():
        raise ValueError(f"scenario input is not a file: {path}")
    return str(path)


def _immutable_image(name: str) -> str:
    value = str(os.getenv(name) or "").strip()
    if not __import__("re").fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be an immutable image digest reference")
    return value


def _inspect_preloaded_image(docker: str, image: str) -> str:
    completed = subprocess.run(
        [docker, "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    image_id = completed.stdout.strip().lower()
    if not image_id.startswith("sha256:") or len(image_id) != 71:
        raise ValueError("preloaded Docker image has no immutable image ID")
    return image_id


def _require_preloaded_sandbox(*, configured_name: str, runtime_name: str) -> None:
    docker = shutil.which("docker")
    if not docker:
        raise ValueError("RD-Agent requires Docker for its governed sandbox")
    configured = _immutable_image(configured_name)
    if os.getenv(runtime_name) != configured:
        raise ValueError(f"{runtime_name} disagrees with the governed image digest")
    try:
        _inspect_preloaded_image(docker, configured)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"required Docker image is not preloaded: {configured}") from exc


def _secret_values() -> list[str]:
    markers = ("KEY", "TOKEN", "PASSWORD", "SECRET", "CREDENTIAL")
    return sorted(
        {
            value
            for name, value in os.environ.items()
            if any(marker in name.upper() for marker in markers) and len(value) >= 6
        },
        key=len,
        reverse=True,
    )


def _redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "[REDACTED]")
    return text


def _run_streaming_redacted(command: list[str], *, timeout: int, env: dict[str, str]) -> None:
    """Stream child output without ever persisting configured secret values."""

    secrets = _secret_values()
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdout is None:
        process.kill()
        raise RuntimeError("RD-Agent subprocess has no output stream")
    messages: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        try:
            for line in process.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    ended = False
    while not ended or process.poll() is None:
        if time.monotonic() >= deadline and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise subprocess.TimeoutExpired(command, timeout)
        try:
            item = messages.get(timeout=0.2)
        except queue.Empty:
            continue
        if item is None:
            ended = True
        else:
            print(_redact(item, secrets), end="", flush=True)
    thread.join(timeout=2)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _verify_feature_set(args: argparse.Namespace) -> str | None:
    if args.scenario not in {"fin_quant", "fin_strategy"}:
        if args.feature_set_id or args.feature_set_sha256 or args.base_features:
            raise ValueError(f"{args.scenario} does not accept a governed feature set")
        return None
    if not args.feature_set_id or not args.feature_set_sha256 or not args.base_features:
        raise ValueError(f"{args.scenario} requires a governed feature set")
    root = Path(_require_path(args.base_features, directory=True))
    base_factors = json.loads((root / "base_factors.json").read_text(encoding="utf-8"))
    definition = json.loads((root / "definition.json").read_text(encoding="utf-8"))
    expected = resolve_feature_set(args.feature_set_id, definition)
    if expected["definition_sha256"] != args.feature_set_sha256:
        raise ValueError("governed feature set digest disagrees with the registry")
    if base_factors != expected["features"] or definition != expected:
        raise ValueError("staged governed feature set definition disagrees")
    return str(root)


def _scenario_command(args: argparse.Namespace) -> list[str]:
    scenario = get_rdagent_scenario(args.scenario)
    assets = list(args.asset)
    # Validate the options file when one is supplied; no retained scenario
    # consumes scenario options.
    _read_options(args.scenario_options)
    base_features = _verify_feature_set(args)
    governed_module_runner = str(
        Path(__file__).resolve().with_name("run_rdagent_module.py")
    )
    if scenario.id in {"fin_quant", "fin_factor_report"}:
        _require_preloaded_sandbox(
            configured_name="RDAGENT_QLIB_SANDBOX_IMAGE",
            runtime_name="QLIB_DOCKER_IMAGE",
        )
    if scenario.id in {"fin_quant", "fin_strategy"} and assets:
        raise ValueError(f"{scenario.id} does not accept document/data assets")
    if scenario.id == "fin_quant":
        return [
            sys.executable,
            governed_module_runner,
            "rdagent.app.qlib_rd_loop.quant",
            "--loop_n",
            str(args.loop_n),
            "--all_duration",
            args.duration,
            "--base_features_path",
            str(base_features),
        ]
    if scenario.id == "fin_strategy":
        return [
            sys.executable,
            governed_module_runner,
            "quant_platform.rdagent_strategy",
            "--loop_n",
            str(args.loop_n),
            "--all_duration",
            args.duration,
            "--base_features_path",
            str(base_features),
        ]
    if scenario.id == "fin_factor_report":
        if len(assets) != 1:
            raise ValueError("fin_factor_report requires one staged report directory")
        report_root = Path(_require_path(assets[0], directory=True))
        reports = sorted(report_root.glob("*.pdf"))
        if not 1 <= len(reports) <= args.loop_n:
            raise ValueError(
                "fin_factor_report requires one governed report per allowed loop"
            )
        return [
            sys.executable,
            governed_module_runner,
            "rdagent.app.qlib_rd_loop.factor_from_report",
            "--report_folder",
            str(report_root),
            "--all_duration",
            args.duration,
        ]
    raise ValueError(f"unsupported RD-Agent scenario: {scenario.id}")


def run(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    _normalize_openai_compatible_model(env)
    env["LOG_TRACE_PATH"] = args.trace
    env["QUANTLAB_RDAGENT_SCENARIO"] = args.scenario
    embeddings_configured = bool(
        str(env.get("EMBEDDING_OPENAI_API_KEY") or "").strip()
        or str(env.get("EMBEDDING_AZURE_API_BASE") or "").strip()
        or str(env.get("HOSTED_VLLM_API_KEY") or "").strip()
    )
    env["QUANTLAB_COSTEER_KNOWLEDGE_STATUS_JSON"] = json.dumps(
        {
            "contract_version": "costeer-knowledge-status-v2",
            "status": (
                "embedding_retrieval_configured"
                if embeddings_configured
                else "unconfigured_fail_closed"
            ),
            "embedding_retrieval_configured": embeddings_configured,
            "retrieval_mode": (
                "embedding_rag" if embeddings_configured else "unconfigured_fail_closed"
            ),
            "costeer_used": True,
            "strategy_codegen_used": True if args.scenario == "fin_strategy" else None,
            "strategy_codegen_target": (
                "allowlisted_rule_ir_and_contract_tests"
                if args.scenario == "fin_strategy"
                else None
            ),
            "strategy_compiler": (
                "deterministic_allowlist" if args.scenario == "fin_strategy" else None
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    command = _scenario_command(args)
    # Give RD-Agent a short cleanup/export margin beyond the governed budget.
    _run_streaming_redacted(
        command,
        env=env,
        timeout=_duration_seconds(args.duration) + 300,
    )
    export = [
        sys.executable,
        args.bridge,
        "export",
        "--trace",
        args.trace,
        "--output",
        args.result,
        "--scenario",
        args.scenario,
    ]
    for asset_id in args.asset_id:
        export.extend(["--asset-id", asset_id])
    if args.feature_set_id:
        export.extend(["--feature-set-id", args.feature_set_id])
    if args.feature_set_sha256:
        export.extend(["--feature-set-sha256", args.feature_set_sha256])
    if args.base_features:
        export.extend(["--base-features", args.base_features])
    _run_streaming_redacted(export, env=env, timeout=300)


def build_parser(default_scenario: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--scenario", default=default_scenario, required=default_scenario is None)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--loop-n", required=True, type=int)
    parser.add_argument("--duration", required=True)
    parser.add_argument("--asset", action="append", default=[])
    parser.add_argument("--asset-id", action="append", default=[])
    parser.add_argument("--scenario-options")
    parser.add_argument("--feature-set-id")
    parser.add_argument("--feature-set-sha256")
    parser.add_argument("--base-features")
    return parser


def main(default_scenario: str | None = None) -> None:
    args = build_parser(default_scenario).parse_args()
    run(args)


if __name__ == "__main__":
    main()
