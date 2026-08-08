from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from quant_data.config import Settings
from quant_data.path_utils import to_wsl_path as _to_wsl_path

from .allocation_store import AllocationStore
from .announcement_factor_registry import default_factors_dir as announcement_factors_dir
from .announcement_nlp import FACTOR_NAME as ANNOUNCEMENT_FACTOR_NAME
from .announcement_nlp import LOGIC_FACTOR_NAME as ANNOUNCEMENT_LOGIC_FACTOR_NAME
from .corpus_nlp import (
    CORPUS_FACTOR_NAMES,
)
from .corpus_nlp import (
    DEFAULT_BATCH_SIZE as CORPUS_DEFAULT_BATCH_SIZE,
)
from .corpus_nlp import (
    DEFAULT_IRM_PER_INSTRUMENT_DAY as CORPUS_DEFAULT_IRM_PER_INSTRUMENT_DAY,
)
from .corpus_nlp import (
    DEFAULT_MAJOR_NEWS_PER_DAY as CORPUS_DEFAULT_MAJOR_NEWS_PER_DAY,
)
from .corpus_nlp import default_factors_dir as corpus_factors_dir
from .cost_model import CostModelConfig
from .data_rollover import qlib_trading_date_on_or_before
from .execution_algorithms import execution_time_slots
from .external_factor_evaluation import import_external_evaluations
from .job_store import JobStore
from .major_news_mentions import FACTOR_NAMES as MAJOR_NEWS_MENTION_FACTOR_NAMES
from .major_news_mentions import default_factors_dir as major_news_mentions_factors_dir
from .market_permission import MarketPermissionStore
from .news_flash_factors import FACTOR_NAMES as NEWS_FLASH_FACTOR_NAMES
from .news_flash_factors import default_factors_dir as news_flash_factors_dir
from .parameter_experiment_store import ParameterExperimentStore
from .rdagent_runtime import rdagent_command, require_rdagent_runtime_identity
from .recommendation_account_store import RecommendationAccountStore
from .recommendation_store import RecommendationStore
from .report_rc_factors import FACTOR_NAMES as REPORT_RC_FACTOR_NAMES
from .report_rc_factors import default_factors_dir as report_rc_factors_dir
from .research_store import ResearchStore
from .runtime_secret_store import RuntimeSecretStore
from .services import list_qlib_datasets, resolve_snapshot_dataset
from .simulation_store import SimulationStore
from .strategy_store import StrategyStore


def _qlib_workflow_environment(settings: Settings, *, is_wsl: bool) -> dict[str, str]:
    artifact_root = settings.data_root / "artifacts" / "mlflow"
    return {
        "_MLFLOW_SERVER_ARTIFACT_ROOT": (
            _to_wsl_path(artifact_root) if is_wsl else str(artifact_root)
        )
    }


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
        self.recommendation_accounts = RecommendationAccountStore(
            settings.database_url,
            simulations=self.simulations,
        )
        self.market_permissions = MarketPermissionStore(settings.database_url)
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
        simulation_order_plan_portfolio_id = job["payload"].get(
            "simulation_portfolio_id"
        ) if job["kind"] == "simulation_order_plan" else None
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
            rdagent_identity = None
            if exit_code == 0 and job["kind"] == "rdagent_factor":
                try:
                    if not isinstance(result, dict):
                        raise ValueError("RD-Agent result is missing")
                    rdagent_identity = require_rdagent_runtime_identity(
                        result.get("rdagent_runtime")
                    )
                except (TypeError, ValueError) as exc:
                    logical_error = str(exc)
                    exit_code = 3
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
            if exit_code == 0 and job["kind"] in {
                "external_factor_evaluate",
                "information_factor_evaluate",
            }:
                try:
                    if not isinstance(result, dict):
                        raise ValueError("external factor evaluation result is missing")
                    self._import_external_factor_evaluations(job, result)
                    failures = [
                        item
                        for item in result.get("evaluations", [])
                        if item.get("status") == "failed"
                    ]
                    if failures:
                        raise ValueError(
                            "; ".join(
                                f"{item.get('candidate_id')}: "
                                f"{item.get('error', 'evaluation failed')}"
                                for item in failures
                            )
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    logical_error = str(exc)
                    exit_code = 3
            if exit_code == 0 and job["kind"] in {
                "strategy_backtest",
                "pair_backtest",
            }:
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
            if exit_code == 0 and simulation_batch_id and not isinstance(result, dict):
                logical_error = "simulation replay result is missing"
                exit_code = 3
            if exit_code == 0 and job["kind"] == "simulation_order_plan":
                if (
                    not isinstance(result, dict)
                    or len(str(result.get("order_plan_manifest_sha256") or "")) != 64
                ):
                    logical_error = "Qlib simulation order-plan result is missing"
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
            if exit_code != 0:
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
                                "rdagent_runtime": rdagent_identity,
                                "trace_path": (result or {}).get("trace_path"),
                                "rounds": (result or {}).get("rounds", 0),
                                "candidates": len(candidates),
                            },
                        )
                    elif job["kind"] == "factor_evaluate":
                        self._import_factor_evaluations(job, result or {})
                        self.research.mark_run(research_run_id, "succeeded")
                else:
                    if (
                        job["kind"] == "factor_evaluate"
                        and isinstance(result, dict)
                        and result.get("evaluations")
                    ):
                        # A partially failed batch still owes the trial ledger
                        # every outcome: import ok and failed evaluations before
                        # marking the run failed (design draft 4.2/6.6).
                        self._import_factor_evaluations(job, result)
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
                    account_risk_state = dict(
                        dict(result.get("risk_summary") or {}).get(
                            "account_risk_state"
                        )
                        or {}
                    )
                    action_state = self.recommendation_accounts.account_state_for_actions(
                        str(result["portfolio_id"]),
                        reference_prices=dict(result.get("reference_prices") or {}),
                        account_risk_state=account_risk_state,
                    )
                    snapshot = self.recommendations.apply_result(
                        recommendation_snapshot_id,
                        result,
                        account_state=action_state["account_state"],
                        permission_store=self.market_permissions,
                        account_context=action_state["account_context"],
                        risk_assessment=action_state["risk_assessment"],
                        account_value=action_state["account_value"],
                    )
                    self.allocations.refresh_for_portfolio(str(snapshot["portfolio_id"]))
                else:
                    self.recommendations.mark_failed(
                        recommendation_snapshot_id, logical_error or process_error
                    )
            if simulation_order_plan_portfolio_id and exit_code == 0 and result:
                batch, created = self.simulations.create_batch_from_order_plan(
                    str(simulation_order_plan_portfolio_id),
                    order_plan_manifest_sha256=str(
                        result["order_plan_manifest_sha256"]
                    ),
                    data_root=self.settings.data_root,
                    actor=str(job["payload"].get("actor") or "simulation-order-plan-worker"),
                )
                if created:
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
                result["simulation_batch_id"] = batch["id"]
                result["simulation_batch_created"] = created
                self.store.finish(job["id"], exit_code=0, result=result)
            elif simulation_batch_id and exit_code == 0 and result:
                if result_path is None:
                    raise ValueError("simulation replay result path is missing")
                bars = pd.read_parquet(result_path.parent / result["minute_bars_file"])
                batch = self.simulations.process_batch(
                    simulation_batch_id,
                    minute_bars=bars,
                    closing_prices=result["closing_prices"],
                    execution_evidence=result,
                    corporate_actions=result.get("corporate_actions"),
                    corporate_events=result.get("corporate_events"),
                    industry_snapshot=result.get("industry_snapshot"),
                )
                manifest = self.simulations.execution_manifest(simulation_batch_id)
                self.allocations.refresh_for_simulation_source(
                    str(manifest["source_type"]), str(manifest["source_id"])
                )
                self.store.finish(job["id"], exit_code=0, result=result)
            elif simulation_batch_id:
                self.simulations.mark_batch_failed(
                    simulation_batch_id, logical_error or process_error
                )
            elif exit_code == 0:
                self.store.finish(job["id"], exit_code=0, result=result)
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
        if job["kind"] == "baostock_overlap_validation":
            result_path = Path(payload["result_path"])
            command = [
                sys.executable,
                "-m",
                "quant_data.cli",
                "validate-baostock-overlap",
                "--start",
                payload["start"],
                "--end",
                payload["end"],
                "--result",
                str(result_path),
            ]
            if payload.get("symbols"):
                command.extend(["--symbols", ",".join(payload["symbols"])])
            return command, result_path, {}
        if job["kind"] == "legacy_market_backfill":
            result_path = Path(payload["result_path"])
            command = [
                sys.executable,
                "-m",
                "quant_data.cli",
                "bootstrap-legacy-market",
                "--start",
                payload["start"],
                "--end",
                payload["end"],
                "--validation-report",
                payload["validation_report"],
                "--result",
                str(result_path),
            ]
            return command, result_path, {}
        if job["kind"] == "cninfo_announcements_download":
            output = self.settings.data_root / "artifacts" / "execution-data" / job["id"]
            result_path = output / "result.json"
            command = [
                sys.executable,
                "-m",
                "quant_data.cli",
                "cninfo-announcements",
                "--start",
                str(payload["start"]),
                "--end",
                str(payload["end"]),
                "--result",
                str(result_path),
            ]
            if payload.get("ts_codes"):
                command.extend(["--ts-code", ",".join(payload["ts_codes"])])
            if int(payload.get("limit") or 0) > 0:
                command.extend(["--limit", str(payload["limit"])])
            if payload.get("regulatory_only", True):
                command.append("--regulatory-only")
            return command, result_path, {}
        if job["kind"] in {
            "announcement_nlp",
            "corpus_nlp",
            "event_market_response",
            "report_rc_factors",
            "major_news_mentions",
            "news_flash_factors",
        }:
            output = self.settings.data_root / "artifacts" / "execution-data" / job["id"]
            result_path = output / "result.json"
            command = [sys.executable, "-m", "quant_data.cli"]
            if job["kind"] == "announcement_nlp":
                command.extend(
                    [
                        "announcement-nlp",
                        "--start",
                        str(payload["start"]),
                        "--end",
                        str(payload["end"]),
                        "--result",
                        str(result_path),
                    ]
                )
                if payload.get("ts_codes"):
                    command.extend(["--ts-code", ",".join(payload["ts_codes"])])
                if payload.get("categories"):
                    command.extend(["--category", ",".join(payload["categories"])])
                if int(payload.get("limit") or 0) > 0:
                    command.extend(["--limit", str(payload["limit"])])
            elif job["kind"] == "corpus_nlp":
                command.extend(
                    [
                        "corpus-nlp",
                        "--start",
                        str(payload["start"]),
                        "--end",
                        str(payload["end"]),
                        "--result",
                        str(result_path),
                    ]
                )
                if payload.get("datasets"):
                    command.extend(["--dataset", ",".join(payload["datasets"])])
                if payload.get("ts_codes"):
                    command.extend(["--ts-code", ",".join(payload["ts_codes"])])
                if int(payload.get("limit") or 0) > 0:
                    command.extend(["--limit", str(payload["limit"])])
                command.extend(
                    [
                        "--batch-size",
                        str(int(payload.get("batch_size") or CORPUS_DEFAULT_BATCH_SIZE)),
                        "--major-news-per-day",
                        str(
                            int(
                                payload.get("major_news_per_day")
                                if payload.get("major_news_per_day") is not None
                                else CORPUS_DEFAULT_MAJOR_NEWS_PER_DAY
                            )
                        ),
                        "--irm-per-instrument-day",
                        str(
                            int(
                                payload.get("irm_per_instrument_day")
                                if payload.get("irm_per_instrument_day") is not None
                                else CORPUS_DEFAULT_IRM_PER_INSTRUMENT_DAY
                            )
                        ),
                    ]
                )
            elif job["kind"] == "event_market_response":
                command.extend(
                    [
                        "event-market-response",
                        "--snapshot-name",
                        str(payload["snapshot_name"]),
                        "--horizons",
                        ",".join(str(value) for value in payload.get("horizons", [1, 3, 5, 20])),
                        "--benchmark-code",
                        str(payload.get("benchmark_code") or "000300.SH"),
                        "--result",
                        str(result_path),
                    ]
                )
            elif job["kind"] == "report_rc_factors":
                command.extend(
                    [
                        "report-rc-factors",
                        "--start",
                        str(payload["start"]),
                        "--end",
                        str(payload["end"]),
                        "--result",
                        str(result_path),
                    ]
                )
                if payload.get("ts_codes"):
                    command.extend(["--ts-code", ",".join(payload["ts_codes"])])
            elif job["kind"] == "major_news_mentions":
                command.extend(
                    [
                        "major-news-mentions",
                        "--start",
                        str(payload["start"]),
                        "--end",
                        str(payload["end"]),
                        "--result",
                        str(result_path),
                    ]
                )
                if payload.get("ts_codes"):
                    command.extend(["--ts-code", ",".join(payload["ts_codes"])])
            else:
                command.extend(
                    [
                        "news-flash-factors",
                        "--start",
                        str(payload["start"]),
                        "--end",
                        str(payload["end"]),
                        "--result",
                        str(result_path),
                    ]
                )
            return command, result_path, {}
        if job["kind"] in {
            "announcement_factor_register",
            "corpus_factor_register",
            "report_rc_factor_register",
            "major_news_mentions_factor_register",
            "news_flash_factor_register",
        }:
            command = [sys.executable, "-m", "quant_platform.db_cli"]
            registration_commands = {
                "announcement_factor_register": "register-announcement-factor",
                "corpus_factor_register": "register-corpus-factor",
                "report_rc_factor_register": "register-report-rc-factor",
                "major_news_mentions_factor_register": (
                    "register-major-news-mentions-factor"
                ),
                "news_flash_factor_register": "register-news-flash-factor",
            }
            command.append(registration_commands[job["kind"]])
            if job["kind"] != "news_flash_factor_register":
                command.extend(
                    ["--factor-name", str(payload.get("factor_name") or "all")]
                )
            command.extend(
                [
                    "--actor",
                    str(payload.get("actor") or "information-pipeline-worker"),
                ]
            )
            return command, None, {}
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
                if payload.get("source_lineage_id"):
                    command.extend(
                        ["--source-lineage-id", str(payload["source_lineage_id"])]
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
        if job["kind"] in {
            "weekly_report",
            "monthly_decision_day",
            "preopen_check",
            "intraday_execution_check",
        }:
            output = (
                self.settings.data_root / "artifacts" / "ops-reports" / job["kind"] / job["id"]
            )
            output.mkdir(parents=True, exist_ok=True)
            result_path = output / "result.json"
            command = [
                sys.executable,
                "-m",
                "quant_platform.ops_tasks",
                job["kind"],
                "--date",
                str(payload["local_date"]),
                "--result",
                str(result_path),
            ]
            if payload.get("dataset"):
                command.extend(["--dataset", str(payload["dataset"])])
            if job["kind"] == "intraday_execution_check":
                command.extend(["--as-of", str(payload["as_of"])])
            return command, result_path, {}
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
            command = [
                sys.executable,
                "-m",
                "quant_data.cli",
                "build-minute-qlib",
                "--snapshot",
                payload["snapshot_name"],
                "--output-name",
                payload["output_name"],
            ]
            if payload.get("target_frequency"):
                command.extend(
                    ["--target-frequency", str(payload["target_frequency"])]
                )
            return command, None, {}
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
                    "--tracking-uri",
                    self.settings.mlflow_tracking_uri,
                ]
            )
            return (
                command,
                result_path,
                _qlib_workflow_environment(self.settings, is_wsl=is_wsl),
            )
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
                payload.get("snapshot_end") or payload["end"],
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
            return (
                command,
                output / "result.json",
                _qlib_workflow_environment(self.settings, is_wsl=is_wsl),
            )
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
                "research_run_id": payload["research_run_id"],
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
                    "--tracking-uri",
                    self.settings.mlflow_tracking_uri,
                ]
            )
            return (
                command,
                result_path,
                _qlib_workflow_environment(self.settings, is_wsl=is_wsl),
            )
        if job["kind"] in {"external_factor_evaluate", "information_factor_evaluate"}:
            output = (
                self.settings.data_root
                / "artifacts"
                / "external-factor-evaluations"
                / job["id"]
            )
            output.mkdir(parents=True, exist_ok=True)
            manifest_path = output / "manifest.json"
            result_path = output / "result.json"
            is_wsl = os.name == "nt" and self.settings.qlib_python.startswith("/")

            def runtime_path(value: str | Path) -> str:
                path = Path(value)
                return _to_wsl_path(path) if is_wsl else str(path)

            candidates = (
                self._resolve_information_factor_candidates(payload)
                if job["kind"] == "information_factor_evaluate"
                else payload["candidates"]
            )
            manifest = {
                "research_run_id": job["id"],
                "dataset": payload["dataset"],
                "dataset_identity_sha256": payload["dataset_identity_sha256"],
                "periods": payload["periods"],
                "universe": payload.get("universe", "cn_all"),
                "benchmark": payload.get("benchmark", "SH000300"),
                "candidates": [
                    {
                        **item,
                        "values_path": runtime_path(item["values_path"]),
                    }
                    for item in candidates
                ],
                "comparison_values": [],
                "cost_model": CostModelConfig.from_mapping(
                    payload.get("cost_model")
                ).to_dict(),
                "cost_reference_order_value": float(
                    payload.get("cost_reference_order_value", 100_000.0)
                ),
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if job["kind"] == "information_factor_evaluate" and not candidates:
                result_path.write_text(
                    json.dumps(
                        {
                            "status": "ok",
                            "evaluations": [],
                            "skipped": (
                                "all registered artifacts already have an evaluation outcome"
                            ),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return [sys.executable, "-c", "pass"], result_path, {}
            script = self.project_root / "scripts" / "evaluate_external_factor_batch.py"
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
                    "--manifest",
                    runtime_path(manifest_path),
                    "--output",
                    runtime_path(result_path),
                    "--tracking-uri",
                    self.settings.mlflow_tracking_uri,
                ]
            )
            return (
                command,
                result_path,
                _qlib_workflow_environment(self.settings, is_wsl=is_wsl),
            )
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
                "execution_dataset": (
                    (payload.get("execution_dataset") or {}).get("name")
                ),
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
                    "--tracking-uri",
                    self.settings.mlflow_tracking_uri,
                ]
            )
            execution_dataset = payload.get("execution_dataset")
            if execution_dataset:
                command.extend(
                    [
                        "--execution-provider-uri",
                        runtime_path(str(execution_dataset["path"])),
                        "--execution-frequency",
                        str(execution_dataset["frequency"]),
                    ]
                )
            return (
                command,
                result_path,
                _qlib_workflow_environment(self.settings, is_wsl=is_wsl),
            )
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
            hypothesis_evidence = self.strategies.hypothesis_group_evidence(
                version["id"]
            )
            final_periods = {
                "start": payload["periods"]["start"],
                "end": payload["periods"]["end"],
            }
            historical_validation_periods = {
                "start": payload["periods"]["historical_start"],
                "end": payload["periods"]["historical_end"],
            }
            manifest = {
                "backtest_id": payload["backtest_id"],
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
                "universe": version["universe"],
                "factor_source_mode": version["config"].get("factor_source_mode"),
                "challenger_weight": version["config"].get("challenger_weight"),
                "baseline": (
                    {
                        "definition": version["config"].get("baseline_definition"),
                        "definition_sha256": version["config"].get(
                            "baseline_definition_sha256"
                        ),
                    }
                    if version["config"].get("baseline_definition")
                    else None
                ),
                "strategy_trial_count": hypothesis_evidence[
                    "shared_experiment_count"
                ],
                "economic_hypothesis_group": hypothesis_evidence[
                    "economic_hypothesis_group"
                ],
                "hypothesis_group_evidence": hypothesis_evidence,
                "periods": final_periods,
                "historical_validation_periods": historical_validation_periods,
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
                    "--tracking-uri",
                    self.settings.mlflow_tracking_uri,
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
            return (
                command,
                result_path,
                _qlib_workflow_environment(self.settings, is_wsl=is_wsl),
            )
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
                "backtest_id": payload["backtest_id"],
                "strategy_version_id": version["id"],
                "dataset": payload["dataset"],
                "execution_snapshot": payload["execution_snapshot"],
                "execution_contract_hash": version["execution_contract_hash"],
                "periods": payload["periods"],
                "config": version["config"],
                "pair": {
                    key: version["pair"][key]
                    for key in ("leg_y", "leg_x", "asset_class", "shorting_mode")
                },
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
                    "--tracking-uri",
                    self.settings.mlflow_tracking_uri,
                ]
            )
            return (
                command,
                result_path,
                _qlib_workflow_environment(self.settings, is_wsl=is_wsl),
            )
        if job["kind"] == "simulation_order_plan":
            portfolio = self.simulations.get(payload["simulation_portfolio_id"])
            if (
                portfolio["status"] != "active"
                or portfolio["source_type"] != "strategy_version"
                or portfolio["execution_adapter"] != "long_only"
            ):
                raise ValueError(
                    "simulation order-plan generation requires an active long-only "
                    "strategy-version simulation"
                )
            version = self.strategies.get_version(portfolio["source_id"])
            if version["status"] != "approved" or version.get("is_legacy"):
                raise ValueError(
                    "simulation order-plan generation requires an approved "
                    "non-legacy strategy version"
                )
            formal = next(
                (
                    item
                    for item in self.strategies.list_backtests(version["id"])
                    if item["status"] == "succeeded" and not item.get("is_legacy")
                ),
                None,
            )
            if formal is None:
                raise ValueError(
                    "simulation order-plan generation requires a successful "
                    "formal Qlib backtest"
                )
            datasets = {
                item["name"]: item
                for item in list_qlib_datasets(self.settings.data_root)
                if item.get("ready")
            }
            dataset = datasets.get(portfolio["daily_dataset"])
            if dataset is None:
                raise ValueError("simulation order-plan Qlib daily dataset is unavailable")
            provenance = dict(dataset.get("provenance") or {})
            if (
                provenance.get("dataset_identity_sha256")
                != portfolio["daily_dataset_identity_sha256"]
                or provenance.get("dataset_lineage_id")
                != portfolio["daily_dataset_lineage_id"]
            ):
                raise ValueError(
                    "simulation order-plan Qlib dataset no longer matches the "
                    "bound account snapshot"
                )
            signal_frequency = str(
                version.get("signal_frequency") or "day"
            ).lower()
            signal_at = payload.get("signal_at")
            execution_not_before: str | None = None
            signal_dataset: dict | None = None
            if signal_frequency != "day":
                if not signal_at:
                    raise ValueError("minute simulation order-plan requires signal_at")
                try:
                    signal_timestamp = datetime.fromisoformat(
                        str(signal_at).replace("Z", "+00:00")
                    )
                except ValueError as exc:
                    raise ValueError(
                        "minute simulation order-plan signal_at is invalid"
                    ) from exc
                if (
                    signal_timestamp.tzinfo is None
                    or signal_timestamp.utcoffset() is None
                ):
                    raise ValueError(
                        "minute simulation order-plan signal_at requires a timezone"
                    )
                local_signal = signal_timestamp.astimezone(
                    ZoneInfo("Asia/Shanghai")
                )
                if local_signal.date().isoformat() != str(payload["signal_date"]):
                    raise ValueError(
                        "minute simulation order-plan signal_at does not match signal_date"
                    )
                source_lineage = str(provenance.get("source_lineage_id") or "")
                candidates = [
                    item
                    for item in datasets.values()
                    if item.get("reproducible") is True
                    and dict(item.get("provenance") or {}).get("lineage_verified")
                    is True
                    and str(
                        dict(item.get("provenance") or {}).get("frequency") or ""
                    )
                    == signal_frequency
                    and str(
                        dict(item.get("provenance") or {}).get(
                            "source_lineage_id"
                        )
                        or ""
                    )
                    == source_lineage
                ]
                signal_dataset = next(
                    (
                        item
                        for item in candidates
                        if item["name"] == portfolio["execution_dataset"]
                    ),
                    candidates[0] if candidates else None,
                )
                if signal_dataset is None:
                    raise ValueError(
                        "minute simulation order-plan requires a ready Qlib signal "
                        f"dataset at {signal_frequency} from the bound Tushare lineage"
                    )
                first_slot = execution_time_slots(
                    trade_date=local_signal.date(),
                    policy=dict(portfolio["execution_policy"]),
                    signal_at=signal_timestamp,
                )[0]
                execution_not_before = first_slot.isoformat()
            positions = self.simulations.rows(portfolio["id"], "positions")
            nav = float(portfolio["nav"])
            previous_holdings = [
                {
                    "instrument": str(item["instrument"]),
                    "weight": max(0.0, float(item.get("market_value") or 0.0))
                    / nav,
                }
                for item in positions
                if nav > 0
                and str(item.get("position_side") or "long") == "long"
                and float(item.get("market_value") or 0.0) > 0
            ]
            strategy_risk_state = self.allocations.strategy_risk_state(
                str(version["id"])
            )
            required_nav_date = qlib_trading_date_on_or_before(
                dataset,
                date.fromisoformat(str(payload["signal_date"])),
            )
            account_risk_state = self.simulations.policy_risk_inputs(
                str(portfolio["id"]),
                required_nav_date=required_nav_date,
            )
            output = (
                self.settings.data_root
                / "artifacts"
                / "order-plan-jobs"
                / job["id"]
            )
            output.mkdir(parents=True, exist_ok=True)
            manifest_path = output / "manifest.json"
            result_path = output / "result.json"
            is_wsl = os.name == "nt" and self.settings.qlib_python.startswith("/")

            def runtime_path(value: str | Path) -> str:
                return _to_wsl_path(Path(value)) if is_wsl else str(value)

            manifest = {
                "artifact_kind": "simulation_order_plan",
                "order_plan_job_id": job["id"],
                "simulation_portfolio_id": portfolio["id"],
                "portfolio_id": portfolio["id"],
                "strategy_version_id": version["id"],
                "formal_backtest_id": formal["id"],
                "dataset": portfolio["daily_dataset"],
                "dataset_identity_sha256": portfolio[
                    "daily_dataset_identity_sha256"
                ],
                "dataset_lineage_id": portfolio["daily_dataset_lineage_id"],
                "signal_date": payload["signal_date"],
                "signal_at": signal_at,
                "execution_not_before": execution_not_before,
                "as_of_date": signal_at or payload["signal_date"],
                "benchmark": version["benchmark"],
                "universe": version["universe"],
                "config": version["config"],
                "construction_notional": nav,
                "risk_exposure": float(
                    strategy_risk_state["risk_exposure_override"]
                ),
                "risk_exposure_override": float(
                    strategy_risk_state["risk_exposure_override"]
                ),
                "allow_new_risk": bool(strategy_risk_state["allow_new_risk"])
                and bool(account_risk_state["allow_new_risk"]),
                "member_risk_state": strategy_risk_state,
                "account_risk_state": account_risk_state,
                "portfolio_drawdown": account_risk_state["portfolio_drawdown"],
                "daily_return": account_risk_state["daily_return"],
                "previous_holdings": previous_holdings,
                "previous_snapshot": None,
                "signal_dataset": (
                    {
                        "name": signal_dataset["name"],
                        "dataset_identity_sha256": dict(
                            signal_dataset.get("provenance") or {}
                        ).get("dataset_identity_sha256"),
                        "dataset_lineage_id": dict(
                            signal_dataset.get("provenance") or {}
                        ).get("dataset_lineage_id"),
                        "source_lineage_id": dict(
                            signal_dataset.get("provenance") or {}
                        ).get("source_lineage_id"),
                        "frequency": signal_frequency,
                    }
                    if signal_dataset is not None
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
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
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
                    runtime_path(dataset["path"]),
                    "--manifest",
                    runtime_path(manifest_path),
                    "--output",
                    runtime_path(result_path),
                    "--tracking-uri",
                    self.settings.mlflow_tracking_uri,
                    "--order-plan-root",
                    runtime_path(
                        self.settings.data_root / "artifacts" / "order-plans"
                    ),
                ]
            )
            if signal_dataset is not None:
                command.extend(
                    [
                        "--signal-provider-uri",
                        runtime_path(signal_dataset["path"]),
                    ]
                )
            return (
                command,
                result_path,
                _qlib_workflow_environment(self.settings, is_wsl=is_wsl),
            )
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
            member_risk_state = self.allocations.strategy_risk_state(str(version["id"]))
            required_nav_date = qlib_trading_date_on_or_before(
                {"path": payload["dataset_path"]},
                date.fromisoformat(str(payload["as_of_date"])),
            )
            account_risk_state = self.recommendation_accounts.policy_risk_inputs(
                str(portfolio["id"]),
                required_nav_date=required_nav_date,
            )
            risk_exposure = min(
                float(portfolio.get("risk_exposure_override", 1.0)),
                float(member_risk_state["risk_exposure_override"]),
            )
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
                "universe": version["universe"],
                "config": version["config"],
                "construction_notional": float(portfolio["construction_notional"]),
                "risk_exposure": risk_exposure,
                "risk_exposure_override": risk_exposure,
                "allow_new_risk": bool(member_risk_state["allow_new_risk"])
                and bool(account_risk_state["allow_new_risk"]),
                "member_risk_state": member_risk_state,
                "account_risk_state": account_risk_state,
                "portfolio_drawdown": account_risk_state["portfolio_drawdown"],
                "daily_return": account_risk_state["daily_return"],
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
            pair_plan = manifest.get("governed_pair_plan")
            shortability_dataset = None
            if manifest.get("execution_adapter") == "pair":
                if not isinstance(pair_plan, dict):
                    raise ValueError("pair simulation replay has no governed artifact plan")
                minute_binding = pair_plan.get("minute_dataset")
                shortability_binding = pair_plan.get("shortability_dataset")
                if not isinstance(minute_binding, dict) or not isinstance(
                    shortability_binding, dict
                ):
                    raise ValueError("pair replay artifact has incomplete Tushare bindings")
                snapshot_name = str(pair_plan.get("execution_snapshot") or "")
                resolved_minute = resolve_snapshot_dataset(
                    self.settings.data_root,
                    snapshot_name=snapshot_name,
                    dataset_name=str(minute_binding.get("dataset_name") or ""),
                )
                shortability_dataset = resolve_snapshot_dataset(
                    self.settings.data_root,
                    snapshot_name=snapshot_name,
                    dataset_name=str(shortability_binding.get("dataset_name") or ""),
                )
                for resolved, binding, label in (
                    (resolved_minute, minute_binding, "minute"),
                    (shortability_dataset, shortability_binding, "shortability"),
                ):
                    if (
                        resolved["manifest_sha256"] != binding.get("manifest_sha256")
                        or resolved["source_sha256"] != binding.get("source_sha256")
                    ):
                        raise ValueError(
                            f"pair replay {label} snapshot no longer matches "
                            "the approved backtest artifact"
                        )
                minute_provenance = dict(minute_dataset.get("provenance") or {})
                if (
                    minute_provenance.get("snapshot_name") != snapshot_name
                    or minute_provenance.get("snapshot_manifest_sha256")
                    != resolved_minute["manifest_sha256"]
                    or str(minute_binding.get("dataset_name") or "")
                    not in set(minute_provenance.get("source_datasets") or [])
                ):
                    raise ValueError(
                        "pair simulation Qlib minute dataset is not derived from "
                        "the approved Tushare execution snapshot"
                    )
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
            dividend_dataset = None
            if manifest.get("execution_adapter") != "pair":
                daily_dataset = datasets.get(manifest["daily_dataset"])
                daily_provenance = (
                    dict(daily_dataset.get("provenance") or {}) if daily_dataset else {}
                )
                dividend_snapshot_name = str(daily_provenance.get("snapshot_name") or "")
                if dividend_snapshot_name:
                    try:
                        resolved_dividend = resolve_snapshot_dataset(
                            self.settings.data_root,
                            snapshot_name=dividend_snapshot_name,
                            dataset_name="dividend",
                        )
                    except (FileNotFoundError, ValueError, KeyError):
                        resolved_dividend = None
                    if resolved_dividend is not None:
                        expected_manifest = str(
                            daily_provenance.get("snapshot_manifest_sha256") or ""
                        )
                        if expected_manifest and expected_manifest != str(
                            resolved_dividend["manifest_sha256"]
                        ):
                            raise ValueError(
                                "dividend snapshot no longer matches the bound "
                                "daily dataset"
                            )
                        dividend_dataset = resolved_dividend
            if shortability_dataset is not None:
                shortability_path = Path(shortability_dataset["dataset_path"])
                command.extend(
                    [
                        "--shortability-path",
                        _to_wsl_path(shortability_path)
                        if is_wsl
                        else str(shortability_path),
                        "--shortability-source-sha256",
                        str(shortability_dataset["source_sha256"]),
                        "--shortability-manifest-sha256",
                        str(shortability_dataset["manifest_sha256"]),
                    ]
                )
            if dividend_dataset is not None:
                dividend_path = Path(dividend_dataset["dataset_path"])
                command.extend(
                    [
                        "--dividend-path",
                        _to_wsl_path(dividend_path)
                        if is_wsl
                        else str(dividend_path),
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
                "minute_qlib",
                "qlib_baseline",
                "announcement_nlp",
                "announcement_factor_register",
                "corpus_nlp",
                "corpus_factor_register",
                "event_market_response",
                "information_factor_evaluate",
                "report_rc_factors",
                "report_rc_factor_register",
                "major_news_mentions",
                "major_news_mentions_factor_register",
                "news_flash_factors",
                "news_flash_factor_register",
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
                # Design draft 4.2/6.6: failed/timed-out trials are ledgered as
                # evaluation_failed, never silently dropped.
                self.research.record_failed_evaluation(
                    str(item["candidate_id"]),
                    dataset=payload["dataset"],
                    dataset_identity_sha256=payload["dataset_identity_sha256"],
                    **periods,
                    error=str(item.get("error") or "evaluation failed"),
                )
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

    def _import_external_factor_evaluations(self, job: dict, result: dict) -> None:
        payload = job["payload"]
        periods = {key: date.fromisoformat(value) for key, value in payload["periods"].items()}
        artifact_path = (
            self.settings.data_root
            / "artifacts"
            / "external-factor-evaluations"
            / job["id"]
            / "result.json"
        )
        import_external_evaluations(
            self.research,
            result,
            dataset=str(payload["dataset"]),
            dataset_identity_sha256=str(payload["dataset_identity_sha256"]),
            periods=periods,
            artifact_path=artifact_path,
        )

    def _resolve_information_factor_candidates(
        self, payload: dict[str, object]
    ) -> list[dict[str, object]]:
        """Bind scheduled evaluation to the exact factor artifacts just registered."""

        names = payload.get("factor_names")
        if not isinstance(names, list) or not names or not all(
            isinstance(name, str) for name in names
        ):
            raise ValueError("information factor evaluation requires factor_names")
        if len(set(names)) != len(names):
            raise ValueError("information factor evaluation factor_names contain duplicates")
        known = {
            ANNOUNCEMENT_FACTOR_NAME,
            ANNOUNCEMENT_LOGIC_FACTOR_NAME,
            *CORPUS_FACTOR_NAMES,
            *REPORT_RC_FACTOR_NAMES,
            *MAJOR_NEWS_MENTION_FACTOR_NAMES,
            *NEWS_FLASH_FACTOR_NAMES,
        }
        unknown = sorted(set(names) - known)
        if unknown:
            raise ValueError(f"unsupported information factors: {unknown}")
        period_values = payload.get("periods")
        if not isinstance(period_values, dict):
            raise ValueError("information factor evaluation periods are missing")
        try:
            valid_end = date.fromisoformat(str(period_values["valid_end"]))
            test_start = date.fromisoformat(str(period_values["test_start"]))
        except (KeyError, ValueError) as exc:
            raise ValueError("information factor evaluation periods are invalid") from exc

        candidates: list[dict[str, object]] = []
        for name in names:
            if name in {ANNOUNCEMENT_FACTOR_NAME, ANNOUNCEMENT_LOGIC_FACTOR_NAME}:
                factor_dir = announcement_factors_dir(self.settings.data_root)
            elif name in CORPUS_FACTOR_NAMES:
                factor_dir = corpus_factors_dir(self.settings.data_root)
            elif name in REPORT_RC_FACTOR_NAMES:
                factor_dir = report_rc_factors_dir(self.settings.data_root)
            elif name in MAJOR_NEWS_MENTION_FACTOR_NAMES:
                factor_dir = major_news_mentions_factors_dir(self.settings.data_root)
            else:
                factor_dir = news_flash_factors_dir(self.settings.data_root)
            manifest_path = factor_dir / f"{name}.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"information factor manifest is unavailable: {manifest_path}"
                ) from exc
            values_sha256 = manifest.get("sha256") if isinstance(manifest, dict) else None
            if not isinstance(values_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", values_sha256
            ):
                raise ValueError(f"information factor manifest sha256 is invalid: {name}")
            candidate = self.research.find_candidate(
                name=name, values_sha256=values_sha256
            )
            if candidate is None:
                raise ValueError(
                    f"registered information factor candidate is missing: {name}"
                )
            if candidate.get("status") not in {
                "awaiting_evaluation",
                "evaluation_failed",
            }:
                continue
            variables = candidate.get("variables")
            source = variables.get("source") if isinstance(variables, dict) else None
            if not isinstance(source, dict) or not str(source.get("dataset") or "").strip():
                raise ValueError(f"information factor {name} has no governed source")
            required = ("code_path", "values_path", "code_sha256", "values_sha256")
            if any(not candidate.get(key) for key in required):
                raise ValueError(f"information factor {name} misses immutable artifacts")
            if not Path(str(candidate["code_path"])).is_file() or not Path(
                str(candidate["values_path"])
            ).is_file():
                raise ValueError(f"information factor {name} artifacts are unavailable")
            horizon = int(candidate.get("label_horizon_days") or 1)
            embargo_days = max(5, horizon)
            if (test_start - valid_end).days <= embargo_days:
                raise ValueError(
                    f"information factor {name} requires a purge/embargo gap greater "
                    f"than {embargo_days} days"
                )
            candidates.append(
                {
                    "id": candidate["id"],
                    "values_path": candidate["values_path"],
                    "code_sha256": candidate["code_sha256"],
                    "values_sha256": candidate["values_sha256"],
                    "experiment_family_id": candidate.get("experiment_family_id"),
                    "experiment_count": int(candidate.get("experiment_count") or 1),
                    "label_horizon_days": horizon,
                }
            )
        return candidates


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
