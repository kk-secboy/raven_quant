#!/usr/bin/env python3
"""Run one allowlisted RD-Agent scenario and export a sanitized result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from quant_platform.feature_set_registry import get_feature_set
from quant_platform.rdagent_scenarios import get_rdagent_scenario


def _duration_seconds(value: str) -> int:
    amount = int(value[:-1])
    return amount * (60 if value.endswith("m") else 3600)


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


def _require_gpu(asset_root: str) -> None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise ValueError("llm_finetune requires an NVIDIA GPU runtime")
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("llm_finetune GPU probe failed") from exc
    rows = [
        [field.strip() for field in line.split(",")]
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError("llm_finetune requires an available NVIDIA GPU")
    minimum_memory = max(
        1024, int(os.getenv("RDAGENT_FINETUNE_MIN_GPU_MEMORY_MB", "16384"))
    )
    if max(int(float(row[1])) for row in rows) < minimum_memory:
        raise ValueError(
            f"llm_finetune requires at least {minimum_memory} MiB free GPU memory"
        )
    version_probe = subprocess.run(
        [executable], capture_output=True, text=True, timeout=10, check=True
    )
    if "CUDA Version:" not in version_probe.stdout or not rows[0][2]:
        raise ValueError("llm_finetune requires NVIDIA driver and CUDA evidence")
    docker = shutil.which("docker")
    if not docker:
        raise ValueError("llm_finetune requires a Docker GPU runtime")
    runtime_probe = subprocess.run(
        [docker, "info", "--format", "{{json .Runtimes}}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    cdi_probe = subprocess.run(
        [docker, "info", "--format", "{{json .CDISpecDirs}}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    runtimes = json.loads(runtime_probe.stdout or "{}")
    cdi_dirs = json.loads(cdi_probe.stdout or "[]") if cdi_probe.returncode == 0 else []
    if not isinstance(runtimes, dict) or (
        "nvidia" not in runtimes and not (isinstance(cdi_dirs, list) and cdi_dirs)
    ):
        raise ValueError("llm_finetune Docker NVIDIA runtime/CDI evidence is unavailable")
    finetune_image = _immutable_image("RDAGENT_FINETUNE_IMAGE")
    benchmark_image = _immutable_image("RDAGENT_FINETUNE_BENCHMARK_IMAGE")
    probe_image = _immutable_image("RDAGENT_FINETUNE_GPU_PROBE_IMAGE")
    if os.getenv("FT_DOCKER_IMAGE") != finetune_image:
        raise ValueError("fine-tune runtime image disagrees with the governed digest")
    if os.getenv("BENCHMARK_DOCKER_IMAGE") != benchmark_image:
        raise ValueError("benchmark runtime image disagrees with the governed digest")
    for image in (finetune_image, benchmark_image, probe_image):
        try:
            _inspect_preloaded_image(docker, image)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"required Docker image is not preloaded: {image}") from exc
    try:
        smoke = subprocess.run(
            [
                docker,
                "run",
                "--rm",
                "--network",
                "none",
                "--gpus",
                "all",
                "--entrypoint",
                "nvidia-smi",
                probe_image,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=45,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("pinned Docker GPU smoke test failed") from exc
    if not smoke.stdout.strip():
        raise ValueError("pinned Docker GPU smoke test returned no GPU evidence")
    minimum_disk = max(
        1.0, float(os.getenv("RDAGENT_FINETUNE_MIN_DISK_GB", "50"))
    )
    free_gb = shutil.disk_usage(Path(asset_root)).free / (1024**3)
    if free_gb < minimum_disk:
        raise ValueError(
            f"llm_finetune requires at least {minimum_disk:g} GiB free DATA_ROOT space"
        )


def _verify_finetune_staging() -> None:
    """Prove that the offline model/data/benchmark inputs stayed immutable."""

    root_value = str(os.getenv("FT_FILE_PATH") or "").strip()
    expected_digest = str(
        os.getenv("QUANTLAB_FINETUNE_STAGED_INVENTORY_SHA256") or ""
    ).strip()
    if not root_value or len(expected_digest) != 64:
        raise ValueError("fine-tune staging identity is unavailable")
    root = Path(root_value).resolve(strict=True)
    evidence_path = root / "quantlab-staged-inventory.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("fine-tune staging evidence is invalid")
    actual_digest = str(evidence.pop("inventory_sha256", ""))
    canonical_digest = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual_digest != expected_digest or canonical_digest != expected_digest:
        raise ValueError("fine-tune staging inventory identity disagrees")
    entries = evidence.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("fine-tune staging inventory has no files")
    expected_files = {"quantlab-staged-inventory.json"}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("fine-tune staging file evidence is invalid")
        relative = Path(str(entry.get("path") or ""))
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError("fine-tune staging file path is unsafe")
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("fine-tune staging file escapes its root") from exc
        if not path.is_file() or path.is_symlink():
            raise ValueError("fine-tune staging contains a non-regular input")
        if path.stat().st_size != int(entry.get("bytes") or -1):
            raise ValueError("fine-tune staging file size disagrees")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != str(entry.get("sha256") or ""):
            raise ValueError("fine-tune staging file digest disagrees")
        expected_files.add(relative.as_posix())
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("fine-tune staging contains a symlink")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != expected_files:
        raise ValueError("fine-tune staging file set changed during execution")


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
    if args.scenario not in {"fin_model", "fin_quant"}:
        if args.feature_set_id or args.feature_set_sha256 or args.base_features:
            raise ValueError(f"{args.scenario} does not accept a governed feature set")
        return None
    if not args.feature_set_id or not args.feature_set_sha256 or not args.base_features:
        raise ValueError(f"{args.scenario} requires a governed feature set")
    expected = get_feature_set(args.feature_set_id)
    if expected["definition_sha256"] != args.feature_set_sha256:
        raise ValueError("governed feature set digest disagrees with the registry")
    root = Path(_require_path(args.base_features, directory=True))
    base_factors = json.loads((root / "base_factors.json").read_text(encoding="utf-8"))
    definition = json.loads((root / "definition.json").read_text(encoding="utf-8"))
    if base_factors != expected["features"] or definition != expected:
        raise ValueError("staged governed feature set definition disagrees")
    return str(root)


def _scenario_command(args: argparse.Namespace) -> list[str]:
    scenario = get_rdagent_scenario(args.scenario)
    assets = list(args.asset)
    options = _read_options(args.scenario_options)
    base_features = _verify_feature_set(args)
    if scenario.id in {
        "fin_factor",
        "fin_model",
        "fin_quant",
        "fin_factor_report",
        "general_model",
    }:
        _require_preloaded_sandbox(
            configured_name="RDAGENT_QLIB_SANDBOX_IMAGE",
            runtime_name="QLIB_DOCKER_IMAGE",
        )
    if scenario.id == "data_science":
        _require_preloaded_sandbox(
            configured_name="RDAGENT_DATA_SCIENCE_IMAGE",
            runtime_name="DS_DOCKER_IMAGE",
        )
    if scenario.id in {"fin_factor", "fin_model", "fin_quant"} and assets:
        raise ValueError(f"{scenario.id} does not accept document/data assets")
    if scenario.id == "fin_factor":
        return [
            args.command,
            scenario.command,
            "--loop-n",
            str(args.loop_n),
            "--all-duration",
            args.duration,
        ]
    if scenario.id in {"fin_model", "fin_quant"}:
        module = {
            "fin_model": "rdagent.app.qlib_rd_loop.model",
            "fin_quant": "rdagent.app.qlib_rd_loop.quant",
        }[scenario.id]
        return [
            sys.executable,
            "-m",
            module,
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
            args.command,
            scenario.command,
            "--report-folder",
            str(report_root),
            "--all-duration",
            args.duration,
        ]
    if scenario.id == "general_model":
        if len(assets) != 1:
            raise ValueError("general_model requires exactly one PDF")
        report = _require_path(assets[0], directory=False)
        if Path(report).suffix.lower() != ".pdf":
            raise ValueError("general_model input must be a PDF")
        return [args.command, scenario.command, report]
    if scenario.id == "data_science":
        if len(assets) != 1 or set(options) != {"competition"}:
            raise ValueError("data_science requires one dataset and its competition identity")
        _require_path(assets[0], directory=True)
        return [
            args.command,
            scenario.command,
            "--competition",
            options["competition"],
            "--loop-n",
            str(args.loop_n),
            "--timeout",
            args.duration,
        ]
    if scenario.id == "llm_finetune":
        required = {
            "benchmark",
            "benchmark_description",
            "dataset",
            "base_model",
            "model_license_accepted",
            "dataset_license_accepted",
            "model_revision",
            "dataset_revision",
            "model_license",
            "dataset_license",
            "model_license_terms_sha256",
            "dataset_license_terms_sha256",
            "model_license_accepted_by",
            "dataset_license_accepted_by",
            "model_license_accepted_at",
            "dataset_license_accepted_at",
        }
        if len(assets) != 1 or set(options) != required:
            raise ValueError("llm_finetune requires one sealed configuration asset")
        asset_root = _require_path(assets[0], directory=True)
        if (
            options["model_license_accepted"] != "true"
            or options["dataset_license_accepted"] != "true"
        ):
            raise ValueError("llm_finetune asset licenses are not explicitly accepted")
        _require_gpu(asset_root)
        return [
            args.command,
            scenario.command,
            "--benchmark",
            options["benchmark"],
            "--benchmark-description",
            options["benchmark_description"],
            "--dataset",
            options["dataset"],
            "--base-model",
            options["base_model"],
            "--loop-n",
            str(args.loop_n),
            "--timeout",
            args.duration,
        ]
    raise ValueError(f"unsupported RD-Agent scenario: {scenario.id}")


def run(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env["LOG_TRACE_PATH"] = args.trace
    env["QUANTLAB_RDAGENT_SCENARIO"] = args.scenario
    command = _scenario_command(args)
    if args.scenario == "llm_finetune":
        _verify_finetune_staging()
    # Give RD-Agent a short cleanup/export margin beyond the governed budget.
    try:
        _run_streaming_redacted(
            command,
            env=env,
            timeout=_duration_seconds(args.duration) + 300,
        )
    finally:
        if args.scenario == "llm_finetune":
            _verify_finetune_staging()
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
