from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import date
from pathlib import Path

import pandas as pd

from quant_data.config import Settings
from quant_data.path_utils import to_wsl_path as _to_wsl_path

from .allocation_store import AllocationStore
from .cost_model import CostModelConfig
from .job_store import JobStore
from .parameter_experiment_store import ParameterExperimentStore
from .rdagent_runtime import rdagent_command
from .recommendation_store import RecommendationStore
from .research_store import ResearchStore
from .runtime_secret_store import RuntimeSecretStore
from .services import list_qlib_datasets
from .simulation_store import SimulationStore
from .strategy_store import StrategyStore


class LocalJobWorker:
    """Runs one durable local job at a time in a child Python process."""

    def __init__(self, store: JobStore, project_root: Path, settings: Settings) -> None:
        self.store = store
        self.project_root = project_root
        self.settings = settings
        self.research = ResearchStore(settings.database_url)
        self.strategies = StrategyStore(settings.database_url)
        self.recommendations = RecommendationStore(settings.database_url)
        self.simulations = SimulationStore(settings.database_url)
        self.allocations = AllocationStore(settings.database_url)
        self.parameter_experiments = ParameterExperimentStore(settings.database_url)
        self.runtime_secrets = RuntimeSecretStore(
            settings.database_url, settings.platform_secret_key
        )
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.store.recover_interrupted(self.settings.worker_job_kinds)
        self._thread = threading.Thread(target=self._loop, name="quant-job-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)

    def notify(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.store.claim_next(self.settings.worker_job_kinds)
            if job is None:
                self._wake.wait(timeout=2)
                self._wake.clear()
                continue
            self._run(job)

    def _run(self, job: dict) -> None:
        research_run_id = job["payload"].get("research_run_id")
        backtest_id = job["payload"].get("backtest_id")
        parameter_experiment_id = job["payload"].get("parameter_experiment_id")
        recommendation_snapshot_id = job["payload"].get("recommendation_snapshot_id")
        simulation_batch_id = job["payload"].get("simulation_batch_id")
        if research_run_id:
            self.research.mark_run(research_run_id, "running")
        if backtest_id:
            self.strategies.mark_backtest(backtest_id, "running")
        if parameter_experiment_id:
            self.parameter_experiments.mark(parameter_experiment_id, "running")
        try:
            command, result_path, extra_env = self._command(job)
        except ValueError as exc:
            self.store.finish(job["id"], exit_code=2, error=str(exc))
            if research_run_id:
                self.research.mark_run(research_run_id, "failed", error=str(exc))
            if backtest_id:
                self.strategies.mark_backtest(backtest_id, "failed", error=str(exc))
            if parameter_experiment_id:
                self.parameter_experiments.mark(parameter_experiment_id, "failed", error=str(exc))
            if recommendation_snapshot_id:
                self.recommendations.mark_failed(recommendation_snapshot_id, str(exc))
            if simulation_batch_id:
                self.simulations.mark_batch_failed(simulation_batch_id, str(exc))
            return
        log_path = Path(job["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if result_path is not None:
            result_path.unlink(missing_ok=True)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=creationflags,
                    env={**os.environ, **extra_env},
                )
                cancelled = False
                progress_mtime_ns: int | None = None
                while process.poll() is None:
                    progress_mtime_ns = self._sync_live_progress(
                        job["id"], result_path, progress_mtime_ns
                    )
                    if self.store.cancellation_requested(job["id"]):
                        cancelled = True
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                        break
                    time.sleep(1)
                exit_code = int(process.returncode or 0)
            if cancelled:
                self.store.mark_cancelled(job["id"])
                cancellation_error = "Cancelled by operator"
                if research_run_id:
                    self.research.mark_run(research_run_id, "failed", error=cancellation_error)
                if backtest_id:
                    self.strategies.mark_backtest(backtest_id, "failed", error=cancellation_error)
                if parameter_experiment_id:
                    self.parameter_experiments.mark(
                        parameter_experiment_id, "failed", error=cancellation_error
                    )
                return
            self._sync_live_progress(job["id"], result_path, progress_mtime_ns)
            process_error = (
                None
                if exit_code == 0
                else _failure_message(log_path, f"process exited with code {exit_code}")
            )
            result = None
            if exit_code == 0 and result_path and result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            logical_error = None
            if job["kind"] == "factor_evaluate" and result:
                failures = [
                    item for item in result.get("evaluations", []) if item.get("status") != "ok"
                ]
                if failures:
                    logical_error = "; ".join(
                        f"{item.get('candidate_id')}: {item.get('error', 'evaluation failed')}"
                        for item in failures
                    )
                    exit_code = 3
            if exit_code == 0 and job["kind"] == "strategy_backtest":
                try:
                    if not isinstance(result, dict) or not isinstance(result.get("metrics"), dict):
                        raise ValueError("strategy backtest result is missing metrics")
                    self.strategies.validate_backtest_artifacts(str(backtest_id), result["metrics"])
                except (KeyError, TypeError, ValueError) as exc:
                    logical_error = str(exc)
                    exit_code = 3
            if exit_code == 0 and job["kind"] == "parameter_experiment":
                try:
                    if not isinstance(result, dict):
                        raise ValueError("parameter experiment result is missing")
                    self.parameter_experiments.apply_result(str(parameter_experiment_id), result)
                except (KeyError, TypeError, ValueError) as exc:
                    logical_error = str(exc)
                    exit_code = 3
            pipeline_stage = job["kind"] in {"data_verify", "data_snapshot", "data_qlib"}
            bootstrap_finalize = job["kind"] == "bootstrap" and bool(
                job["payload"].get("finalize_after_download")
            )
            chained_pipeline = self._has_data_pipeline_successor(job)
            if exit_code == 0 and (pipeline_stage or bootstrap_finalize or chained_pipeline):
                try:
                    self._queue_data_pipeline_successor(job)
                except Exception as exc:
                    logical_error = f"could not enqueue next data pipeline stage: {exc}"
                    exit_code = 4
            if exit_code == 0:
                self.store.finish(job["id"], exit_code=0, result=result)
            else:
                failure_error = logical_error or process_error or "job failed"
                requeued = self.store.finish_or_retry(
                    job["id"],
                    exit_code=exit_code,
                    error=failure_error,
                    result=result,
                    retryable=logical_error is None,
                )
                if requeued:
                    return
            if research_run_id:
                if exit_code == 0:
                    if job["kind"] == "rdagent_factor":
                        candidates = self._import_rdagent_candidates(research_run_id, result or {})
                        self._queue_factor_evaluation(job, candidates)
                        self.research.mark_run(
                            research_run_id,
                            "evaluating",
                            runtime={
                                "trace_path": (result or {}).get("trace_path"),
                                "rounds": (result or {}).get("rounds", 0),
                                "candidates": len(candidates),
                            },
                        )
                    elif job["kind"] == "factor_evaluate":
                        self._import_factor_evaluations(job, result or {})
                        self.research.mark_run(research_run_id, "succeeded")
                else:
                    self.research.mark_run(
                        research_run_id,
                        "failed",
                        error=logical_error or process_error,
                    )
            if backtest_id:
                if exit_code == 0 and result:
                    self.strategies.mark_backtest(
                        backtest_id,
                        "succeeded",
                        metrics=result["metrics"],
                    )
                else:
                    self.strategies.mark_backtest(
                        backtest_id,
                        "failed",
                        error=logical_error or process_error,
                    )
            if parameter_experiment_id and exit_code != 0:
                self.parameter_experiments.mark(
                    parameter_experiment_id,
                    "failed",
                    error=logical_error or process_error,
                )
            if recommendation_snapshot_id:
                if exit_code == 0 and result:
                    snapshot = self.recommendations.apply_result(recommendation_snapshot_id, result)
                    batch, created = self.simulations.create_batch_for_snapshot(
                        recommendation_snapshot_id
                    )
                    if created and batch:
                        self.store.create(
                            "simulation_replay",
                            {"simulation_batch_id": batch["id"]},
                            self.settings.data_root
                            / "platform"
                            / "logs"
                            / f"simulation-replay-{batch['id']}.log",
                            dedupe_active_kind=False,
                            idempotency_key=f"simulation-replay:{batch['id']}",
                        )
                    self.allocations.refresh_for_portfolio(str(snapshot["portfolio_id"]))
                else:
                    self.recommendations.mark_failed(
                        recommendation_snapshot_id, logical_error or process_error
                    )
            if simulation_batch_id and exit_code == 0 and result:
                if result_path is None:
                    raise ValueError("simulation replay result path is missing")
                bars = pd.read_parquet(result_path.parent / result["minute_bars_file"])
                batch = self.simulations.process_batch(
                    simulation_batch_id,
                    minute_bars=bars,
                    closing_prices=result["closing_prices"],
                    execution_evidence=result,
                )
                manifest = self.simulations.execution_manifest(simulation_batch_id)
                self.allocations.refresh_for_portfolio(
                    str(manifest["recommendation_portfolio_id"])
                )
            elif simulation_batch_id:
                self.simulations.mark_batch_failed(
                    simulation_batch_id, logical_error or process_error
                )
        except Exception as exc:
            requeued = self.store.finish_or_retry(
                job["id"],
                exit_code=1,
                error=str(exc),
                retryable=True,
            )
            if requeued:
                return
            if research_run_id:
                self.research.mark_run(research_run_id, "failed", error=str(exc))
            if backtest_id:
                self.strategies.mark_backtest(backtest_id, "failed", error=str(exc))
            if parameter_experiment_id:
                self.parameter_experiments.mark(parameter_experiment_id, "failed", error=str(exc))
            if recommendation_snapshot_id:
                self.recommendations.mark_failed(recommendation_snapshot_id, str(exc))
            if simulation_batch_id:
                self.simulations.mark_batch_failed(simulation_batch_id, str(exc))

    def _sync_live_progress(
        self,
        job_id: str,
        result_path: Path | None,
        previous_mtime_ns: int | None,
    ) -> int | None:
        if result_path is None or not result_path.exists():
            return previous_mtime_ns
        try:
            mtime_ns = result_path.stat().st_mtime_ns
            if mtime_ns == previous_mtime_ns:
                return previous_mtime_ns
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return previous_mtime_ns
        if isinstance(payload, dict):
            self.store.update_progress(job_id, payload)
        return mtime_ns

    def _command(self, job: dict) -> tuple[list[str], Path | None, dict[str, str]]:
        payload = job["payload"]
        if job["kind"] in {
            "margin_eligibility_download",
            "core_intraday_download",
            "ashare_5m_download",
        } or job["kind"].startswith("supplemental_"):
            stored = self.runtime_secrets.get("tushare")
            api_url = (stored or {}).get("api_url") or self.settings.api_url
            token = (stored or {}).get("token") or self.settings.token
            if not api_url or not token:
                raise ValueError("Tushare credentials are not configured")
            output = self.settings.data_root / "artifacts" / "execution-data" / job["id"]
            result_path = output / "result.json"
            command = [sys.executable, "-m", "quant_data.cli"]
            if job["kind"] == "margin_eligibility_download":
                command.extend(
                    [
                        "margin-eligibility",
                        "--start",
                        payload["start"],
                        "--end",
                        payload["end"],
                        "--result",
                        str(result_path),
                    ]
                )
            elif job["kind"] == "core_intraday_download":
                command.extend(
                    [
                        "core-intraday",
                        "--start",
                        payload["start"],
                        "--end",
                        payload["end"],
                        "--snapshot-name",
                        payload["snapshot_name"],
                        "--result",
                        str(result_path),
                    ]
                )
                for option, key in (
                    ("--etfs", "etfs"),
                    ("--stocks", "stocks"),
                    ("--indices", "indices"),
                    ("--futures", "futures"),
                    ("--options", "options"),
                ):
                    values = payload.get(key) or []
                    if values:
                        command.extend([option, ",".join(values)])
                if payload.get("auto_select", False):
                    command.extend(
                        [
                            "--auto-universe",
                            "--max-stocks",
                            str(payload.get("max_stocks", 100)),
                            "--max-options",
                            str(payload.get("max_options", 100)),
                            "--etf-categories",
                            ",".join(
                                payload.get("etf_categories")
                                or ["broad", "industry", "gold", "bond"]
                            ),
                        ]
                    )
            elif job["kind"] == "ashare_5m_download":
                command.extend(
                    [
                        "ashare-5m",
                        "--start",
                        payload["start"],
                        "--end",
                        payload["end"],
                        "--snapshot-name",
                        payload["snapshot_name"],
                        "--result",
                        str(result_path),
                    ]
                )
            else:
                command.extend(
                    [
                        "supplemental-download",
                        "--bundle",
                        payload["bundle"],
                        "--start",
                        payload["start"],
                        "--end",
                        payload["end"],
                        "--result",
                        str(result_path),
                    ]
                )
                if payload.get("symbols"):
                    command.extend(["--symbols", ",".join(payload["symbols"])])
            return command, result_path, {"TUSHARE_API_URL": api_url, "TUSHARE_TOKEN": token}
        if job["kind"] == "data_verify":
            return (
                [
                    sys.executable,
                    "-m",
                    "quant_data.cli",
                    "verify",
                    "--snapshot-end",
                    str(payload.get("end") or "latest"),
                    "--allow-incomplete-plans",
                ],
                None,
                {},
            )
        if job["kind"] == "data_snapshot":
            return (
                [
                    sys.executable,
                    "-m",
                    "quant_data.cli",
                    "snapshot",
                    "--name",
                    payload["snapshot_name"],
                    "--start",
                    payload["start"],
                    "--end",
                    payload["end"],
                    "--profile",
                    payload["profile"],
                ],
                None,
                {},
            )
        if job["kind"] == "data_qlib":
            return (
                [
                    sys.executable,
                    "-m",
                    "quant_data.cli",
                    "build-qlib",
                    "--snapshot",
                    payload["snapshot_name"],
                ],
                None,
                {},
            )
        if job["kind"] == "minute_qlib":
            return (
                [
                    sys.executable,
                    "-m",
                    "quant_data.cli",
                    "build-minute-qlib",
                    "--snapshot",
                    payload["snapshot_name"],
                    "--output-name",
                    payload["output_name"],
                ],
                None,
                {},
            )
        if job["kind"] == "minute_research":
            output = self.settings.data_root / "artifacts" / "minute-research" / job["id"]
            result_path = output / "result.json"
            script = self.project_root / "scripts" / "run_minute_factor_research.py"
            is_wsl = os.name == "nt" and self.settings.qlib_python.startswith("/")
            command = (
                [
                    "wsl",
                    "-d",
                    self.settings.qlib_wsl_distro,
                    "--exec",
                    self.settings.qlib_python,
                    _to_wsl_path(script),
                ]
                if is_wsl
                else [self.settings.qlib_python, str(script)]
            )
            command.extend(
                [
                    "--provider-uri",
                    _to_wsl_path(Path(payload["dataset_path"]))
                    if is_wsl
                    else str(Path(payload["dataset_path"])),
                    "--output",
                    _to_wsl_path(result_path) if is_wsl else str(result_path),
                    "--start",
                    payload["start"],
                    "--end",
                    payload["end"],
                    "--horizons",
                    ",".join(str(item) for item in payload["horizons"]),
                    "--cost-rate",
                    str(payload["cost_rate"]),
                ]
            )
            return command, result_path, {}
        if job["kind"] == "bootstrap":
            stored = self.runtime_secrets.get("tushare")
            api_url = (stored or {}).get("api_url") or self.settings.api_url
            token = (stored or {}).get("token") or self.settings.token
            if not api_url or not token:
                raise ValueError("Tushare credentials are not configured")
            command = [
                sys.executable,
                "-m",
                "quant_data.cli",
                "bootstrap",
                "--profile",
                payload["profile"],
                "--start",
                payload["start"],
                "--end",
                payload["end"],
            ]
            command.append("--download-only")
            return (
                command,
                None,
                {
                    "TUSHARE_API_URL": api_url,
                    "TUSHARE_TOKEN": token,
                },
            )
        if job["kind"] == "qlib_baseline":
            output = self.settings.data_root / "artifacts" / "qlib" / job["id"]
            script = self.project_root / "scripts" / "run_qlib_baseline.py"
            is_wsl = os.name == "nt" and self.settings.qlib_python.startswith("/")
            command = (
                [
                    "wsl",
                    "-d",
                    self.settings.qlib_wsl_distro,
                    "--exec",
                    self.settings.qlib_python,
                    _to_wsl_path(script),
                ]
                if is_wsl
                else [self.settings.qlib_python, str(script)]
            )
            command.extend(
                [
                    "--provider-uri",
                    _to_wsl_path(Path(payload["dataset_path"]))
                    if is_wsl
                    else str(Path(payload["dataset_path"])),
                    "--output",
                    _to_wsl_path(output) if is_wsl else str(output),
                    "--tracking-uri",
                    self.settings.mlflow_tracking_uri,
                    "--market",
                    payload["market"],
                    "--benchmark",
                    payload["benchmark"],
                    "--account",
                    str(payload["account"]),
                    "--topk",
                    str(payload["topk"]),
                    "--n-drop",
                    str(payload["n_drop"]),
                    "--open-cost",
                    str(payload["open_cost"]),
                    "--close-cost",
                    str(payload["close_cost"]),
                    "--min-cost",
                    str(payload["min_cost"]),
                ]
            )
            return command, output / "result.json", {}
        if job["kind"] == "rdagent_factor":
            output = self.settings.data_root / "artifacts" / "rdagent" / payload["research_run_id"]
            trace = output / "trace"
            result_path = output / "result.json"
            output.mkdir(parents=True, exist_ok=True)
            command, env = rdagent_command(
                self.settings,
                project_root=self.project_root,
                trace_path=trace,
                result_path=result_path,
                dataset_path=Path(payload["dataset_path"]),
                loop_n=int(payload["loop_n"]),
                duration=str(payload["duration"]),
                periods=payload["periods"],
                objective=str(payload["objective"]),
            )
            llm = self.runtime_secrets.get("llm")
            if llm:
                env[self.settings.rdagent_llm_key_env] = llm["api_key"]
                env["OPENAI_API_BASE"] = llm.get("api_base", "")
                env["CHAT_MODEL"] = llm.get("chat_model", "gpt-4.1-mini")
            return command, result_path, env
        if job["kind"] == "factor_evaluate":
            output = (
                self.settings.data_root
                / "artifacts"
                / "factor-evaluations"
                / payload["research_run_id"]
            )
            output.mkdir(parents=True, exist_ok=True)
            manifest_path = output / "manifest.json"
            result_path = output / "result.json"
            is_wsl = os.name == "nt" and self.settings.qlib_python.startswith("/")

            def runtime_path(value: str) -> str:
                return _to_wsl_path(Path(value)) if is_wsl else str(value)

            promoted = self.research.list_candidates(status="promoted", limit=500)
            manifest = {
                "candidates": [
                    {
                        **item,
                        "code_path": runtime_path(item["code_path"]),
                        "submitted_values_path": runtime_path(item["values_path"]),
                    }
                    for item in payload["candidates"]
                ],
                "dataset_identity_sha256": payload["dataset_identity_sha256"],
                "periods": payload["periods"],
                "universe": payload.get("universe", "cn_all"),
                "min_daily_instruments": int(payload.get("min_daily_instruments", 50)),
                "comparison_values": [
                    runtime_path(item["values_path"])
                    for item in promoted
                    if item.get("values_path") and Path(item["values_path"]).exists()
                ],
                "cost_model": CostModelConfig.from_mapping(payload.get("cost_model")).to_dict(),
                "cost_reference_order_value": float(
                    payload.get("cost_reference_order_value", 100_000.0)
                ),
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            script = self.project_root / "scripts" / "evaluate_factor_batch.py"
            command = (
                [
                    "wsl",
                    "-d",
                    self.settings.qlib_wsl_distro,
                    "--exec",
                    self.settings.qlib_python,
                    _to_wsl_path(script),
                ]
                if is_wsl
                else [self.settings.qlib_python, str(script)]
            )
            command.extend(
                [
                    "--provider-uri",
                    _to_wsl_path(Path(payload["dataset_path"]))
                    if is_wsl
                    else str(Path(payload["dataset_path"])),
                    "--manifest",
                    _to_wsl_path(manifest_path) if is_wsl else str(manifest_path),
                    "--output",
                    _to_wsl_path(result_path) if is_wsl else str(result_path),
                ]
            )
            return command, result_path, {}
        if job["kind"] == "parameter_experiment":
            experiment = self.parameter_experiments.get(payload["parameter_experiment_id"])
            output = Path(experiment["artifact_path"])
            output.mkdir(parents=True, exist_ok=True)
            manifest_path = output / "manifest.json"
            result_path = output / "result.json"
            version = self.strategies.get_version(payload["strategy_version_id"])
            if version.get("strategy_type") != "multifactor":
                raise ValueError("parameter experiments require a multifactor strategy")
            is_wsl = os.name == "nt" and self.settings.qlib_python.startswith("/")

            def runtime_path(value: str) -> str:
                return _to_wsl_path(Path(value)) if is_wsl else str(Path(value))

            manifest = {
                "experiment_id": experiment["id"],
                "strategy_version_id": version["id"],
                "dataset": experiment["dataset"],
                "benchmark": version["benchmark"],
                "periods": experiment["periods"],
                "parameter_grid": experiment["parameter_grid"],
                "factors": [
                    {
                        "candidate_id": item["factor_candidate_id"],
                        "values_path": runtime_path(item["values_path"]),
                        "code_sha256": item["code_sha256"],
                        "weight": item["weight"],
                        "direction": item["direction"],
                    }
                    for item in version["factors"]
                ],
                "trials": [
                    {
                        "trial_index": item["trial_index"],
                        "parameters": item["parameters"],
                        "config": item["config"],
                    }
                    for item in experiment["trials"]
                ],
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            script = self.project_root / "scripts" / "run_parameter_experiment.py"
            command = (
                [
                    "wsl",
                    "-d",
                    self.settings.qlib_wsl_distro,
                    "--exec",
                    self.settings.qlib_python,
                    _to_wsl_path(script),
                ]
                if is_wsl
                else [self.settings.qlib_python, str(script)]
            )
            command.extend(
                [
                    "--provider-uri",
                    _to_wsl_path(Path(payload["dataset_path"]))
                    if is_wsl
                    else str(Path(payload["dataset_path"])),
                    "--manifest",
                    _to_wsl_path(manifest_path) if is_wsl else str(manifest_path),
                    "--output",
                    _to_wsl_path(output) if is_wsl else str(output),
                ]
            )
            return command, result_path, {}
        if job["kind"] == "strategy_backtest":
            output = self.settings.data_root / "artifacts" / "backtests" / payload["backtest_id"]
            output.mkdir(parents=True, exist_ok=True)
            manifest_path = output / "manifest.json"
            result_path = output / "result.json"
            version = self.strategies.get_version(payload["strategy_version_id"])
            is_wsl = os.name == "nt" and self.settings.qlib_python.startswith("/")

            def runtime_path(value: str) -> str:
                return _to_wsl_path(Path(value)) if is_wsl else str(value)

            execution_dataset = payload.get("execution_dataset")
            family_trials: dict[str, int] = {}
            for factor in version["factors"]:
                family = str(factor.get("experiment_family_id") or factor["factor_candidate_id"])
                family_trials[family] = max(
                    family_trials.get(family, 0), int(factor.get("experiment_count") or 1)
                )
            manifest = {
                "strategy_version_id": version["id"],
                "dataset": payload["dataset"],
                "execution_dataset": (
                    execution_dataset.get("name") if isinstance(execution_dataset, dict) else None
                ),
                "execution_frequency": (
                    execution_dataset.get("frequency")
                    if isinstance(execution_dataset, dict)
                    else None
                ),
                "execution_contract_version": (
                    (execution_dataset.get("provenance") or {}).get(
                        "execution_contract_version"
                    )
                    if isinstance(execution_dataset, dict)
                    else None
                ),
                "benchmark": version["benchmark"],
                "strategy_trial_count": max(1, sum(family_trials.values())),
                "periods": payload["periods"],
                "config": version["config"],
                "factors": [
                    {
                        "candidate_id": item["factor_candidate_id"],
                        "values_path": runtime_path(item["values_path"]),
                        "code_sha256": item["code_sha256"],
                        "weight": item["weight"],
                        "direction": item["direction"],
                    }
                    for item in version["factors"]
                ],
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            script = self.project_root / "scripts" / "run_multifactor_backtest.py"
            command = (
                [
                    "wsl",
                    "-d",
                    self.settings.qlib_wsl_distro,
                    "--exec",
                    self.settings.qlib_python,
                    _to_wsl_path(script),
                ]
                if is_wsl
                else [self.settings.qlib_python, str(script)]
            )
            command.extend(
                [
                    "--provider-uri",
                    _to_wsl_path(Path(payload["dataset_path"]))
                    if is_wsl
                    else str(Path(payload["dataset_path"])),
                    "--manifest",
                    _to_wsl_path(manifest_path) if is_wsl else str(manifest_path),
                    "--output",
                    _to_wsl_path(output) if is_wsl else str(output),
                ]
            )
            if isinstance(execution_dataset, dict):
                command.extend(
                    [
                        "--execution-provider-uri",
                        runtime_path(execution_dataset["path"]),
                        "--execution-frequency",
                        str(execution_dataset["frequency"]),
                    ]
                )
            return command, result_path, {}
        if job["kind"] == "pair_backtest":
            output = self.settings.data_root / "artifacts" / "backtests" / payload["backtest_id"]
            output.mkdir(parents=True, exist_ok=True)
            manifest_path = output / "manifest.json"
            result_path = output / "result.json"
            version = self.strategies.get_version(payload["strategy_version_id"])
            if version.get("strategy_type") != "pair" or not version.get("pair"):
                raise ValueError("pair backtest job requires a pair strategy version")
            is_wsl = os.name == "nt" and self.settings.qlib_python.startswith("/")

            def runtime_path(value: str) -> str:
                return _to_wsl_path(Path(value)) if is_wsl else str(Path(value))

            manifest = {
                "strategy_version_id": version["id"],
                "dataset": payload["dataset"],
                "execution_snapshot": payload["execution_snapshot"],
                "periods": payload["periods"],
                "config": version["config"],
                "pair": version["pair"],
                "daily_provenance": payload["daily_provenance"],
                "minute_dataset": payload["minute_dataset"],
                "shortability_dataset": payload["shortability_dataset"],
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            script = self.project_root / "scripts" / "run_pair_backtest.py"
            command = (
                [
                    "wsl",
                    "-d",
                    self.settings.qlib_wsl_distro,
                    "--exec",
                    self.settings.qlib_python,
                    _to_wsl_path(script),
                ]
                if is_wsl
                else [self.settings.qlib_python, str(script)]
            )
            command.extend(
                [
                    "--provider-uri",
                    runtime_path(payload["dataset_path"]),
                    "--minute-path",
                    runtime_path(payload["minute_dataset"]["dataset_path"]),
                    "--shortability-path",
                    runtime_path(payload["shortability_dataset"]["dataset_path"]),
                    "--manifest",
                    runtime_path(str(manifest_path)),
                    "--output",
                    runtime_path(str(output)),
                ]
            )
            return command, result_path, {}
        if job["kind"] == "recommendation_refresh":
            output = (
                self.settings.data_root
                / "artifacts"
                / "recommendations"
                / payload["recommendation_portfolio_id"]
                / payload["recommendation_snapshot_id"]
            )
            output.mkdir(parents=True, exist_ok=True)
            manifest_path = output / "manifest.json"
            result_path = output / "result.json"
            portfolio = self.recommendations.get(payload["recommendation_portfolio_id"])
            version = self.strategies.get_version(portfolio["strategy_version_id"])
            if version["status"] != "approved":
                raise ValueError("recommendation refresh requires an approved strategy version")
            is_wsl = os.name == "nt" and self.settings.qlib_python.startswith("/")

            def runtime_path(value: str) -> str:
                return _to_wsl_path(Path(value)) if is_wsl else str(value)

            latest_snapshot = portfolio.get("latest_snapshot") or {}
            manifest = {
                "portfolio_id": portfolio["id"],
                "strategy_version_id": version["id"],
                "dataset": payload["dataset"],
                "dataset_identity_sha256": payload["dataset_identity_sha256"],
                "as_of_date": payload["as_of_date"],
                "benchmark": version["benchmark"],
                "config": version["config"],
                "construction_notional": float(portfolio["construction_notional"]),
                "risk_exposure": float(portfolio.get("risk_exposure_override", 1.0)),
                "previous_holdings": latest_snapshot.get("holdings") or [],
                "previous_snapshot": (
                    {
                        "as_of_date": str(latest_snapshot["as_of_date"]),
                        "effective_date": str(latest_snapshot.get("effective_date") or ""),
                        "holdings": latest_snapshot.get("holdings") or [],
                    }
                    if latest_snapshot
                    else None
                ),
                "factors": [
                    {
                        "candidate_id": item["factor_candidate_id"],
                        "values_path": runtime_path(item["values_path"]),
                        "weight": item["weight"],
                        "direction": item["direction"],
                    }
                    for item in version["factors"]
                ],
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            script = self.project_root / "scripts" / "run_recommendation_refresh.py"
            command = (
                [
                    "wsl",
                    "-d",
                    self.settings.qlib_wsl_distro,
                    "--exec",
                    self.settings.qlib_python,
                    _to_wsl_path(script),
                ]
                if is_wsl
                else [self.settings.qlib_python, str(script)]
            )
            command.extend(
                [
                    "--provider-uri",
                    _to_wsl_path(Path(payload["dataset_path"]))
                    if is_wsl
                    else str(Path(payload["dataset_path"])),
                    "--manifest",
                    _to_wsl_path(manifest_path) if is_wsl else str(manifest_path),
                    "--output",
                    _to_wsl_path(result_path) if is_wsl else str(result_path),
                ]
            )
            return command, result_path, {}
        if job["kind"] == "simulation_replay":
            manifest = self.simulations.execution_manifest(payload["simulation_batch_id"])
            datasets = {
                item["name"]: item
                for item in list_qlib_datasets(self.settings.data_root)
                if item.get("ready")
            }
            minute_dataset = datasets.get(manifest["execution_dataset"])
            if minute_dataset is None:
                raise ValueError("simulation execution Qlib dataset is unavailable")
            output = (
                self.settings.data_root
                / "artifacts"
                / "simulations"
                / manifest["portfolio_id"]
                / manifest["batch_id"]
            )
            output.mkdir(parents=True, exist_ok=True)
            manifest_path = output / "manifest.json"
            result_path = output / "result.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            is_wsl = os.name == "nt" and self.settings.qlib_python.startswith("/")
            script = self.project_root / "scripts" / "run_simulation_replay.py"
            command = (
                [
                    "wsl",
                    "-d",
                    self.settings.qlib_wsl_distro,
                    "--exec",
                    self.settings.qlib_python,
                    _to_wsl_path(script),
                ]
                if is_wsl
                else [self.settings.qlib_python, str(script)]
            )
            command.extend(
                [
                    "--provider-uri",
                    _to_wsl_path(Path(minute_dataset["path"]))
                    if is_wsl
                    else str(minute_dataset["path"]),
                    "--manifest",
                    _to_wsl_path(manifest_path) if is_wsl else str(manifest_path),
                    "--output",
                    _to_wsl_path(result_path) if is_wsl else str(result_path),
                ]
            )
            return command, result_path, {}
        raise ValueError(f"unsupported job kind: {job['kind']}")

    def _queue_data_pipeline_successor(self, job: dict) -> dict:
        payload = dict(job["payload"])
        snapshot_name = str(payload["snapshot_name"])
        pipeline_steps = payload.get("pipeline_steps")
        next_index = int(payload.get("pipeline_next_index", 0))
        if isinstance(pipeline_steps, list) and next_index < len(pipeline_steps):
            step = pipeline_steps[next_index]
            if not isinstance(step, dict) or not isinstance(step.get("payload", {}), dict):
                raise ValueError("data pipeline contains an invalid step")
            kind = str(step.get("kind") or "")
            allowed = {
                "data_verify",
                "data_snapshot",
                "data_qlib",
                "qlib_baseline",
                *(
                    f"supplemental_{bundle}"
                    for bundle in (
                        "cn_extended_daily",
                        "cn_funds",
                        "cn_macro",
                        "cn_institutional",
                        "cn_futures",
                        "cn_options_bonds",
                        "hk_market",
                        "us_market",
                        "global_markets",
                    )
                ),
            }
            if kind not in allowed:
                raise ValueError(f"unsupported data pipeline step: {kind}")
            successor_payload = {
                "pipeline_id": payload["pipeline_id"],
                "profile": payload["profile"],
                "start": payload.get("snapshot_start", payload["start"]),
                "end": payload.get("snapshot_end", payload["end"]),
                "snapshot_name": payload["snapshot_name"],
                "pipeline_steps": pipeline_steps,
                "pipeline_next_index": next_index + 1,
                **step.get("payload", {}),
            }
            if kind == "qlib_baseline":
                successor_payload.update(
                    {
                        "dataset": snapshot_name,
                        "dataset_path": str(self.settings.data_root / "qlib" / snapshot_name),
                        "market": "cn_all",
                        "benchmark": "SH000300",
                        "account": 5_000_000,
                        "topk": 50,
                        "n_drop": 5,
                        "open_cost": 0.0005,
                        "close_cost": 0.0015,
                        "min_cost": 5.0,
                    }
                )
        elif job["kind"] == "bootstrap":
            if not payload.get("finalize_after_download"):
                raise ValueError("bootstrap job did not request a finalize pipeline")
            kind = "data_verify"
            successor_payload = {
                "pipeline_id": payload["pipeline_id"],
                "profile": payload["profile"],
                "start": payload.get("snapshot_start", payload["start"]),
                "end": payload.get("snapshot_end", payload["end"]),
                "snapshot_name": payload["snapshot_name"],
            }
        elif job["kind"] == "data_verify":
            kind = "data_snapshot"
            successor_payload = payload
        elif job["kind"] == "data_snapshot":
            kind = "data_qlib"
            successor_payload = payload
        elif job["kind"] == "data_qlib":
            kind = "qlib_baseline"
            successor_payload = {
                **payload,
                "dataset": snapshot_name,
                "dataset_path": str(self.settings.data_root / "qlib" / snapshot_name),
                "market": "cn_all",
                "benchmark": "SH000300",
                "account": 5_000_000,
                "topk": 50,
                "n_drop": 5,
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5.0,
            }
        else:
            raise ValueError(f"job {job['kind']} is not a data pipeline stage")
        log_path = self.settings.data_root / "platform" / "logs" / f"{kind}-{snapshot_name}.log"
        successor = self.store.create(
            kind,
            successor_payload,
            log_path,
            idempotency_key=f"data-finalize:{snapshot_name}:{kind}",
        )
        self.notify()
        return successor

    @staticmethod
    def _has_data_pipeline_successor(job: dict) -> bool:
        payload = job.get("payload") or {}
        steps = payload.get("pipeline_steps")
        return isinstance(steps, list) and int(payload.get("pipeline_next_index", 0)) < len(steps)

    def _import_rdagent_candidates(self, run_id: str, result: dict) -> list[dict]:
        imported = []
        source_candidates = result.get("candidates", [])
        for item in source_candidates:
            imported.append(
                self.research.add_candidate(
                    run_id,
                    name=str(item["name"]),
                    description=str(item.get("description") or ""),
                    formulation=item.get("formulation"),
                    variables=item.get("variables") or {},
                    source_iteration=item.get("source_iteration"),
                    code_path=_local_artifact_path(item.get("code_path")),
                    values_path=_local_artifact_path(item.get("values_path")),
                    code_sha256=item.get("code_sha256"),
                    rdagent_decision=item.get("rdagent_decision"),
                    rdagent_feedback=item.get("rdagent_feedback"),
                    experiment_family_id=str(item.get("experiment_family_id") or run_id),
                    label_horizon_days=int(item.get("label_horizon_days") or 1),
                    experiment_count=len(source_candidates),
                )
            )
        return imported

    def _queue_factor_evaluation(self, job: dict, candidates: list[dict]) -> None:
        payload = job["payload"]
        eligible = [
            {
                "id": item["id"],
                "code_path": item["code_path"],
                "values_path": item["values_path"],
                "experiment_family_id": item["experiment_family_id"],
                "label_horizon_days": item["label_horizon_days"],
                "experiment_count": item["experiment_count"],
            }
            for item in candidates
            if item.get("rdagent_decision") is not False
            and item.get("code_path")
            and Path(item["code_path"]).exists()
            and item.get("values_path")
            and Path(item["values_path"]).exists()
        ]
        if not eligible:
            raise ValueError("RD-Agent produced no executable factor value artifacts")
        log_path = (
            self.settings.data_root
            / "platform"
            / "logs"
            / f"factor-evaluate-{payload['research_run_id']}.log"
        )
        evaluation_job = self.store.create(
            "factor_evaluate",
            {
                "research_run_id": payload["research_run_id"],
                "dataset": payload["dataset"],
                "dataset_path": payload["dataset_path"],
                "dataset_identity_sha256": payload["dataset_identity_sha256"],
                "periods": payload["periods"],
                "candidates": eligible,
                "cost_model": CostModelConfig.from_mapping(payload.get("cost_model")).to_dict(),
                "cost_reference_order_value": float(
                    payload.get("cost_reference_order_value", 100_000.0)
                ),
                "universe": payload.get("universe", "cn_all"),
                "min_daily_instruments": int(payload.get("min_daily_instruments", 50)),
            },
            log_path,
            idempotency_key=f"factor-evaluate:{payload['research_run_id']}",
        )
        self.research.attach_job(payload["research_run_id"], evaluation_job["id"])

    def _import_factor_evaluations(self, job: dict, result: dict) -> None:
        payload = job["payload"]
        periods = {key: date.fromisoformat(value) for key, value in payload["periods"].items()}
        artifact_path = (
            self.settings.data_root
            / "artifacts"
            / "factor-evaluations"
            / payload["research_run_id"]
            / "result.json"
        )
        for item in result.get("evaluations", []):
            if item.get("status") != "ok":
                continue
            self.research.record_evaluation(
                item["candidate_id"],
                dataset=payload["dataset"],
                dataset_identity_sha256=payload["dataset_identity_sha256"],
                **periods,
                metrics=item["metrics"],
                artifact_path=str(artifact_path),
                recomputed_values_path=_local_artifact_path(item["recomputed_values_path"]),
                recomputed_values_sha256=item["recomputed_values_sha256"],
                recompute_evidence=item["recompute_evidence"],
            )


def _failure_message(log_path: Path, fallback: str) -> str:
    try:
        with log_path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 65_536))
            text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return fallback
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    lines = [ansi.sub("", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    markers = ("ValueError:", "RuntimeError:", "Error:", "Exception:", "required")
    for line in reversed(lines):
        if any(marker.lower() in line.lower() for marker in markers):
            return line[-1000:]
    return (lines[-1] if lines else fallback)[-1000:]


def _local_artifact_path(value: str | None) -> str | None:
    if not value or os.name != "nt" or not value.startswith("/mnt/"):
        return value
    parts = value.split("/", 3)
    if len(parts) < 4 or len(parts[2]) != 1:
        return value
    return str(Path(f"{parts[2].upper()}:/{parts[3]}"))
