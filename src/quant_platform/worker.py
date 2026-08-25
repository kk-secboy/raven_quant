from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from quant_data.config import Settings
from quant_data.path_utils import to_wsl_path as _to_wsl_path
from quant_data.supplemental_data import SUPPORTED_BUNDLES

from .allocation_store import AllocationStore
from .announcement_factor_registry import default_factors_dir as announcement_factors_dir
from .announcement_nlp import DEFAULT_BATCH_SIZE as ANNOUNCEMENT_DEFAULT_BATCH_SIZE
from .announcement_nlp import DEFAULT_WORKERS as ANNOUNCEMENT_DEFAULT_WORKERS
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
from .corpus_nlp import DEFAULT_WORKERS as CORPUS_DEFAULT_WORKERS
from .corpus_nlp import default_factors_dir as corpus_factors_dir
from .cost_model import CostModelConfig
from .data_rollover import qlib_trading_date_on_or_before, select_qlib_dataset
from .execution_algorithms import execution_time_slots
from .external_factor_evaluation import import_external_evaluations
from .job_store import JobStore
from .major_news_mentions import FACTOR_NAMES as MAJOR_NEWS_MENTION_FACTOR_NAMES
from .major_news_mentions import default_factors_dir as major_news_mentions_factors_dir
from .market_permission import MarketPermissionStore
from .model_artifact_store import ModelArtifactStore
from .news_flash_factors import FACTOR_NAMES as NEWS_FLASH_FACTOR_NAMES
from .news_flash_factors import default_factors_dir as news_flash_factors_dir
from .parameter_experiment_store import ParameterExperimentStore
from .promotion import PromotionStore
from .rdagent_candidate_store import RDAGentCandidateStore
from .rdagent_runtime import (
    probe_rdagent,
    rdagent_command,
    require_matching_rdagent_runtime_identity,
)
from .rdagent_scenarios import get_rdagent_scenario, is_rdagent_job, validate_asset_id
from .recommendation_account_store import RecommendationAccountStore
from .recommendation_store import RecommendationStore
from .report_rc_factors import FACTOR_NAMES as REPORT_RC_FACTOR_NAMES
from .report_rc_factors import default_factors_dir as report_rc_factors_dir
from .research_store import ResearchStore
from .runtime_secret_store import RuntimeSecretStore
from .services import list_qlib_datasets, resolve_snapshot_dataset, resolve_snapshot_manifest
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
        self.rdagent_candidates = RDAGentCandidateStore(settings.database_url)
        self.strategies = StrategyStore(settings.database_url)
        self.model_artifacts = ModelArtifactStore(settings.database_url)
        self.recommendations = RecommendationStore(settings.database_url)
        self.simulations = SimulationStore(settings.database_url)
        self.promotions = PromotionStore(settings.database_url)
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
        simulation_order_plan_portfolio_id = (
            job["payload"].get("simulation_portfolio_id")
            if job["kind"] == "simulation_order_plan"
            else None
        )
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
            result_read_error: str | None = None
            if (
                (exit_code == 0 or job["kind"] == "research_asset_acquire")
                and result_path
                and result_path.exists()
            ):
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    result_read_error = str(exc)
            logical_error = None
            rdagent_identity = None
            if job["kind"] == "research_asset_acquire" and result is not None:
                try:
                    result = self._project_research_asset_acquisition_result(job, result)
                    if result["status"] == "blocked" or int(result["failed"]) > 0:
                        logical_error = "research asset acquisition reported blocked sources"
                        if exit_code == 0:
                            exit_code = 3
                    elif exit_code != 0:
                        logical_error = (
                            "research asset acquisition exited non-zero after publishing "
                            "a complete result"
                        )
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    logical_error = str(exc)
                    exit_code = 3
            elif job["kind"] == "research_asset_acquire" and exit_code == 0:
                logical_error = result_read_error or "research asset acquisition result is missing"
                exit_code = 3
            if exit_code == 0 and is_rdagent_job(job["kind"]):
                try:
                    if not isinstance(result, dict):
                        raise ValueError("RD-Agent result is missing")
                    expected_scenario = str(job["payload"].get("scenario") or "fin_factor")
                    if result.get("scenario") != expected_scenario:
                        raise ValueError("RD-Agent result scenario disagrees with the job")
                    expected_features = job["payload"].get("feature_set")
                    if expected_features is not None and result.get("feature_set") != {
                        "id": expected_features["id"],
                        "definition_sha256": expected_features["definition_sha256"],
                    }:
                        raise ValueError("RD-Agent result feature set identity disagrees")
                    rdagent_identity = require_matching_rdagent_runtime_identity(
                        job["payload"].get("expected_rdagent_runtime"),
                        result.get("rdagent_runtime"),
                    )
                    scenario = get_rdagent_scenario(expected_scenario)
                    trace_summary = result.get("trace_summary") or {}
                    if (
                        scenario.category == "lab"
                        and int(trace_summary.get("message_count") or 0) < 1
                    ):
                        raise ValueError("RD-Agent lab result has no sanitized trace evidence")
                    if scenario.id == "general_model" and not result.get("lab_outputs"):
                        raise ValueError(
                            "general_model produced no implementation-ready model artifact"
                        )
                    if scenario.id == "fin_quant" and not result.get("quant_bundles"):
                        raise ValueError(
                            "fin_quant produced no accepted executable factor-model bundle"
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
            if exit_code == 0 and job["kind"] == "multiface_audit":
                if not isinstance(result, dict) or not isinstance(result.get("ok"), bool):
                    logical_error = "multi-face readiness audit result is missing"
                    exit_code = 3
                elif job["payload"].get("require_ready", True) and not result["ok"]:
                    logical_error = "one or more governed data faces are not ready"
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
            if exit_code == 0 and job["kind"] == "model_refit":
                try:
                    if not isinstance(result, dict) or result_path is None:
                        raise ValueError("model refit result is missing")
                    artifact = self.model_artifacts.create_from_live_refit(
                        strategy_version_id=str(job["payload"]["strategy_version_id"]),
                        source_model_artifact_id=str(
                            job["payload"]["source_model_artifact_id"]
                        ),
                        result_path=result_path,
                        actor="model-refit-worker",
                        valid_for_days=int(job["payload"].get("valid_for_days") or 4),
                    )
                    activated = (
                        artifact
                        if artifact.get("status") == "active"
                        else self.model_artifacts.activate(
                            str(artifact["id"]), actor="model-refit-worker"
                        )
                    )
                    result["model_artifact_id"] = activated["id"]
                    result["model_artifact_status"] = activated["status"]
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
                    if is_rdagent_job(job["kind"]):
                        scenario = get_rdagent_scenario(
                            str(job["payload"].get("scenario") or "fin_factor")
                        )
                        runtime = {
                            "scenario": scenario.id,
                            "rdagent_runtime": rdagent_identity,
                            "trace_path": (result or {}).get("trace_path"),
                            "trace_summary": (result or {}).get("trace_summary"),
                            "rounds": (result or {}).get("rounds", 0),
                            "feature_set": (result or {}).get("feature_set"),
                            "asset_ids": (result or {}).get("asset_ids") or [],
                        }
                        archive = self._archive_rdagent_run_evidence(
                            research_run_id, scenario.id, result or {}
                        )
                        runtime.update(archive)
                        if scenario.id == "fin_model":
                            candidates = self._import_rdagent_model_candidates(
                                research_run_id, job, result or {}
                            )
                            self._queue_model_evaluation(job, candidates)
                            self.research.mark_run(
                                research_run_id,
                                "evaluating",
                                runtime={**runtime, "model_candidates": len(candidates)},
                            )
                        elif scenario.id == "fin_quant":
                            bundles = self._queue_quant_bundle_evaluation(job, result or {})
                            self.research.mark_run(
                                research_run_id,
                                "evaluating",
                                runtime={**runtime, "quant_bundles": bundles},
                            )
                        elif scenario.factor_output:
                            candidates = self._import_rdagent_candidates(
                                research_run_id, result or {}
                            )
                            if scenario.id == "fin_factor_report":
                                for asset_id in job["payload"].get("asset_ids") or []:
                                    for candidate in candidates:
                                        self.rdagent_candidates.link_asset(
                                            asset_id=str(asset_id),
                                            candidate_kind="factor",
                                            candidate_id=str(candidate["id"]),
                                            relationship="extracted_from_report",
                                            actor="worker",
                                        )
                            self._queue_factor_evaluation(job, candidates)
                            self.research.mark_run(
                                research_run_id,
                                "evaluating",
                                runtime={**runtime, "candidates": len(candidates)},
                            )
                        else:
                            lab_archive = self._archive_rdagent_lab_artifacts(
                                research_run_id,
                                scenario.id,
                                result or {},
                                sanitized_result_artifact_id=str(
                                    archive["sanitized_result_artifact_id"]
                                ),
                                sanitized_result_sha256=str(archive["sanitized_result_sha256"]),
                            )
                            self.research.mark_run(
                                research_run_id,
                                "succeeded",
                                runtime={**runtime, **lab_archive},
                            )
                    elif job["kind"] == "model_evaluate":
                        self._import_model_evaluations(job, result or {})
                        self.research.mark_run(research_run_id, "succeeded")
                    elif job["kind"] == "quant_bundle_evaluate":
                        self._import_quant_bundle_evaluation_artifact(job, result or {})
                        self.research.mark_run(research_run_id, "succeeded")
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
                        dict(result.get("risk_summary") or {}).get("account_risk_state") or {}
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
                    order_plan_manifest_sha256=str(result["order_plan_manifest_sha256"]),
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

    def _project_research_asset_acquisition_result(
        self,
        job: dict,
        raw_result: object,
    ) -> dict[str, object]:
        """Import every published asset even when the acquisition is blocked.

        The CLI deliberately exits non-zero when one selected source is blocked,
        but successfully materialized siblings remain immutable valid evidence.
        Project them before the job is finalized and retain the bounded failure
        summary so a consumed daily quota cannot be retried into a false success.
        """

        if not isinstance(raw_result, dict):
            raise ValueError("research asset acquisition result must be an object")
        mode = str(job.get("payload", {}).get("mode") or "")
        if mode == "automatic":
            raw_asset_ids = raw_result.get("published_asset_ids") or []
            if not isinstance(raw_asset_ids, list):
                raise ValueError("automatic research asset result has invalid asset IDs")
            asset_ids = [validate_asset_id(str(value)) for value in raw_asset_ids]
        elif mode == "manual_https":
            raw_asset_id = str(raw_result.get("asset_id") or "")
            asset_ids = [validate_asset_id(raw_asset_id)] if raw_asset_id else []
        else:
            raise ValueError("research asset acquisition mode is invalid")
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError("research asset acquisition result contains duplicate assets")

        imported: list[dict[str, str]] = []
        for asset_id in asset_ids:
            registered = self.rdagent_candidates.import_manifest(
                self.settings.data_root
                / "artifacts"
                / "research-assets"
                / asset_id
                / "manifest.json",
                actor="research-asset-worker",
            )
            imported.append(
                {
                    "asset_id": asset_id,
                    "content_sha256": str(registered["content_sha256"]),
                    "manifest_sha256": str(registered["manifest_sha256"]),
                }
            )

        failed = int(raw_result.get("failed") or 0)
        blocked = int(raw_result.get("blocked") or 0)
        if failed < 0 or blocked < 0 or failed > blocked:
            raise ValueError("research asset blocked-result counts are invalid")
        status = str(raw_result.get("status") or "succeeded")
        if status not in {"succeeded", "blocked"}:
            raise ValueError("research asset acquisition result status is invalid")
        if failed and status != "blocked":
            raise ValueError("failed research asset sources require blocked status")
        return {
            "status": status,
            "mode": mode,
            "published": len(imported),
            "assets": imported,
            "blocked": blocked,
            "failed": failed,
            "tushare_selected": int(raw_result.get("tushare_selected") or 0),
            "arxiv_selected": int(raw_result.get("arxiv_selected") or 0),
            "daily_limits": {"tushare_research_report": 20, "arxiv": 3},
        }

    def _command(self, job: dict) -> tuple[list[str], Path | None, dict[str, str]]:
        payload = job["payload"]
        if job["kind"] == "research_asset_acquire":
            output = (
                self.settings.data_root
                / "artifacts"
                / "research-asset-acquisitions"
                / job["id"]
            )
            output.mkdir(parents=True, exist_ok=True)
            result_path = output / "result.json"
            mode = str(payload.get("mode") or "")
            command = [sys.executable, "-m", "quant_data.cli"]
            if mode == "automatic":
                snapshot_name = str(payload.get("snapshot_name") or "")
                if (
                    not snapshot_name
                    or snapshot_name in {".", ".."}
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", snapshot_name)
                ):
                    raise ValueError("automatic research asset snapshot identity is invalid")
                research_day = date.fromisoformat(str(payload.get("as_of") or ""))
                if not payload.get("include_tushare") and not payload.get("include_arxiv"):
                    raise ValueError("automatic research asset job has no enabled source")
                command.extend(
                    [
                        "research-assets",
                        "--snapshot",
                        snapshot_name,
                        "--as-of",
                        research_day.isoformat(),
                        "--result",
                        str(result_path),
                    ]
                )
                if not payload.get("include_tushare"):
                    command.append("--skip-tushare")
                if not payload.get("include_arxiv"):
                    command.append("--skip-arxiv")
            elif mode == "manual_https":
                document_kind = str(payload.get("document_kind") or "")
                asset_type = {
                    "paper": "manual_paper",
                    "research_report": "research_report",
                }.get(document_kind)
                if asset_type is None:
                    raise ValueError("manual research asset document kind is invalid")
                command.extend(
                    [
                        "research-asset-fetch-pdf",
                        "--url",
                        str(payload["url"]),
                        "--title",
                        str(payload["title"]),
                        "--type",
                        asset_type,
                        "--result",
                        str(result_path),
                    ]
                )
                if payload.get("published_at"):
                    command.extend(["--published-at", str(payload["published_at"])])
            else:
                raise ValueError("research asset acquisition mode is invalid")
            return command, result_path, {}
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
                command.extend(
                    [
                        "--batch-size",
                        str(int(payload.get("batch_size") or ANNOUNCEMENT_DEFAULT_BATCH_SIZE)),
                        "--workers",
                        str(int(payload.get("workers") or ANNOUNCEMENT_DEFAULT_WORKERS)),
                    ]
                )
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
                        "--workers",
                        str(int(payload.get("workers") or CORPUS_DEFAULT_WORKERS)),
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
                "major_news_mentions_factor_register": ("register-major-news-mentions-factor"),
                "news_flash_factor_register": "register-news-flash-factor",
            }
            command.append(registration_commands[job["kind"]])
            if job["kind"] != "news_flash_factor_register":
                command.extend(["--factor-name", str(payload.get("factor_name") or "all")])
            command.extend(
                [
                    "--actor",
                    str(payload.get("actor") or "information-pipeline-worker"),
                ]
            )
            return command, None, {}
        if job["kind"] == "multiface_audit":
            output = self.settings.data_root / "artifacts" / "execution-data" / job["id"]
            result_path = output / "result.json"
            command = [
                sys.executable,
                "-m",
                "quant_platform.db_cli",
                "audit-multiface",
                "--dataset",
                str(payload["dataset"]),
                "--result",
                str(result_path),
            ]
            if payload.get("snapshot_name"):
                command.extend(["--snapshot", str(payload["snapshot_name"])])
            return command, result_path, {}
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
                command.extend(["--source-lineage-id", str(payload["source_lineage_id"])])
                if payload.get("daily_dataset"):
                    command.extend(["--daily-source-dataset", str(payload["daily_dataset"])])
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
                source_lineage_id = str(payload.get("source_lineage_id") or "")
                if not source_lineage_id:
                    raise ValueError(
                        "A-share five-minute download requires a bound daily source lineage"
                    )
                command.extend(["--source-lineage-id", source_lineage_id])
                if payload.get("daily_dataset"):
                    command.extend(["--daily-source-dataset", str(payload["daily_dataset"])])
            elif (
                job["kind"] == "supplemental_research_corpus"
                and payload.get("profile") == "research-assets"
            ):
                command.extend(
                    [
                        "research-report-download",
                        "--start",
                        payload["start"],
                        "--end",
                        payload["end"],
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
        if job["kind"] in {
            "weekly_report",
            "monthly_decision_day",
            "preopen_check",
            "intraday_execution_check",
        }:
            output = self.settings.data_root / "artifacts" / "ops-reports" / job["kind"] / job["id"]
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
                    "--profile",
                    str(payload.get("profile") or "full"),
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
                command.extend(["--target-frequency", str(payload["target_frequency"])])
            expected_manifest_sha256 = str(payload.get("snapshot_manifest_sha256") or "")
            if not expected_manifest_sha256:
                raise ValueError("minute Qlib job has no sealed snapshot manifest digest")
            command.extend(["--expected-manifest-sha256", expected_manifest_sha256])
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
            # Worker bootstrap jobs always stop after durable download units;
            # make that contract explicit so core/research profiles do not
            # inherit the CLI's full-publication Qlib default.
            command.extend(["--download-only", "--no-build-qlib"])
            if payload.get("incremental") is True:
                command.append("--incremental")
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
        if is_rdagent_job(job["kind"]):
            scenario = get_rdagent_scenario(str(payload.get("scenario") or "fin_factor"))
            llm = self.runtime_secrets.get("llm")
            runtime_env = None
            if llm:
                runtime_env = {
                    self.settings.rdagent_llm_key_env: llm["api_key"],
                    "OPENAI_API_BASE": llm.get("api_base", ""),
                    "CHAT_MODEL": llm.get("chat_model", "gpt-4.1-mini"),
                }
            local_runtime = probe_rdagent(
                self.settings,
                self.project_root,
                runtime_env=runtime_env,
                force_local=True,
            )
            require_matching_rdagent_runtime_identity(
                payload.get("expected_rdagent_runtime"),
                local_runtime.get("runtime_identity", local_runtime),
            )
            for asset_id in payload.get("asset_ids") or []:
                self.rdagent_candidates.import_manifest(
                    self.settings.data_root
                    / "artifacts"
                    / "research-assets"
                    / str(asset_id)
                    / "manifest.json",
                    actor="worker",
                )
            output = self.settings.data_root / "artifacts" / "rdagent" / payload["research_run_id"]
            trace = output / "trace"
            result_path = output / "result.json"
            output.mkdir(parents=True, exist_ok=True)
            command, env = rdagent_command(
                self.settings,
                project_root=self.project_root,
                trace_path=trace,
                result_path=result_path,
                dataset_path=(
                    Path(str(payload["dataset_path"]))
                    if scenario.requires_dataset and payload.get("dataset_path")
                    else None
                ),
                loop_n=int(payload["loop_n"]),
                duration=str(payload["duration"]),
                periods=payload.get("periods"),
                objective=str(payload["objective"]),
                scenario=scenario.id,
                asset_ids=list(payload.get("asset_ids") or []),
                asset_manifest_sha256=dict(payload.get("asset_manifest_sha256") or {}),
                feature_set=payload.get("feature_set"),
            )
            if llm:
                env[self.settings.rdagent_llm_key_env] = llm["api_key"]
                env["OPENAI_API_BASE"] = llm.get("api_base", "")
                env["CHAT_MODEL"] = llm.get("chat_model", "gpt-4.1-mini")
            return command, result_path, env
        if job["kind"] in {"model_evaluate", "quant_bundle_evaluate"}:
            if not str(os.getenv("MODEL_SANDBOX_IMAGE") or "").strip():
                raise ValueError("MODEL_SANDBOX_IMAGE is required for model isolation")
            evaluation_name = (
                "model-evaluations"
                if job["kind"] == "model_evaluate"
                else "quant-bundle-evaluations"
            )
            output = (
                self.settings.data_root / "artifacts" / evaluation_name / payload["research_run_id"]
            )
            output.mkdir(parents=True, exist_ok=True)
            manifest_path = output / "manifest.json"
            result_path = output / "result.json"
            is_wsl = os.name == "nt" and self.settings.qlib_python.startswith("/")

            def runtime_path(value: str | Path) -> str:
                path = Path(value)
                return _to_wsl_path(path) if is_wsl else str(path)

            candidates = []
            for candidate in payload["candidates"]:
                item = dict(candidate)
                if job["kind"] == "model_evaluate":
                    item["code_path"] = runtime_path(str(item["code_path"]))
                else:
                    item["factors"] = [
                        {
                            **factor,
                            "code_path": runtime_path(str(factor["code_path"])),
                            **(
                                {
                                    "submitted_values_path": runtime_path(
                                        str(factor["submitted_values_path"])
                                    )
                                }
                                if factor.get("submitted_values_path")
                                else {}
                            ),
                        }
                        for factor in item["factors"]
                    ]
                    item["model"] = {
                        **item["model"],
                        "code_path": runtime_path(str(item["model"]["code_path"])),
                    }
                candidates.append(item)
            manifest = {
                "research_run_id": payload["research_run_id"],
                "candidates": candidates,
                "feature_set_id": payload["feature_set_id"],
                "dataset_identity_sha256": payload["dataset_identity_sha256"],
                "evaluation_profiles": payload.get("evaluation_profiles") or [],
                "universe": payload.get("universe", "cn_all"),
                "benchmark": payload.get("benchmark", "SH000300"),
                "account": int(payload.get("account", 100_000_000)),
                "topk": int(payload.get("topk", 50)),
                "n_drop": int(payload.get("n_drop", 5)),
                "open_cost": float(payload.get("open_cost", 0.0005)),
                "close_cost": float(payload.get("close_cost", 0.0015)),
                "min_cost": float(payload.get("min_cost", 5.0)),
                "model_timeout_seconds": int(payload.get("model_timeout_seconds", 7200)),
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            script_name = (
                "evaluate_model_batch.py"
                if job["kind"] == "model_evaluate"
                else "evaluate_quant_bundle.py"
            )
            script = self.project_root / "scripts" / script_name
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
                    runtime_path(str(payload["dataset_path"])),
                    "--manifest",
                    runtime_path(manifest_path),
                    "--output",
                    runtime_path(result_path),
                ]
            )
            return (
                command,
                result_path,
                _qlib_workflow_environment(self.settings, is_wsl=is_wsl),
            )
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
                "evaluation_profiles": payload.get("evaluation_profiles") or [],
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
                self.settings.data_root / "artifacts" / "external-factor-evaluations" / job["id"]
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
                "evaluation_profiles": payload.get("evaluation_profiles") or [],
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
                "cost_model": CostModelConfig.from_mapping(payload.get("cost_model")).to_dict(),
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
        if job["kind"] == "model_refit":
            version = self.strategies.get_version(str(payload["strategy_version_id"]))
            if version.get("status") != "approved" or version.get("is_legacy"):
                raise ValueError("model refit requires an approved non-legacy StrategySpec")
            model_signal = version.get("model_signal")
            if not isinstance(model_signal, dict):
                raise ValueError("model refit requires a governed model-prediction strategy")
            source_artifact = self.model_artifacts.get(
                str(payload["source_model_artifact_id"])
            )
            if (
                source_artifact.get("strategy_version_id") != version["id"]
                or source_artifact.get("status") != "active"
            ):
                raise ValueError("model refit source is not the active StrategySpec artifact")
            output = (
                self.settings.data_root
                / "artifacts"
                / "model-refits"
                / str(payload["strategy_version_id"])
                / str(payload["signal_date"])
            )
            manifest_path = output.parent / f"{payload['signal_date']}-manifest.json"
            result_path = output / "result.json"
            is_wsl = os.name == "nt" and self.settings.qlib_python.startswith("/")

            def runtime_path(value: str | Path) -> str:
                path = Path(value)
                return _to_wsl_path(path) if is_wsl else str(path)

            candidate = self.rdagent_candidates.get_model_candidate(
                str(model_signal["model_candidate_id"]), verify=True
            )
            base = dict(candidate.get("base_features_manifest_json") or {})
            feature_set = {
                "id": str(base.get("feature_set_id") or ""),
                "contract_version": str(base.get("contract_version") or ""),
                "features": dict(base.get("feature_expressions") or {}),
                "definition_sha256": str(base.get("definition_sha256") or ""),
            }
            manifest = {
                "contract_version": "model-live-refit-v1",
                "strategy_version_id": version["id"],
                "source_model_artifact_id": source_artifact["id"],
                "execution_environment_sha256": str(
                    source_artifact["execution_environment_sha256"]
                ),
                "dataset": str(payload["dataset"]),
                "dataset_identity_sha256": str(payload["dataset_identity_sha256"]),
                "dataset_lineage_id": str(payload["dataset_lineage_id"]),
                "signal_date": str(payload["signal_date"]),
                "universe": version["universe"],
                "feature_set": feature_set,
                "feature_set_definition_sha256": str(
                    model_signal["feature_set_definition_sha256"]
                ),
                "model": {
                    "candidate_id": str(model_signal["model_candidate_id"]),
                    "code_path": runtime_path(str(model_signal["code_path"])),
                    "code_sha256": str(model_signal["model_code_sha256"]),
                    "recipe_sha256": str(model_signal["model_recipe_sha256"]),
                    "model_type": str(model_signal.get("model_type") or "Tabular"),
                    "training_hyperparameters": dict(
                        model_signal.get("training_hyperparameters") or {}
                    ),
                    "seed": int(model_signal["primary_seed"]),
                },
                "refit_policy": dict(model_signal["refit_policy"]),
                "refit_policy_sha256": str(model_signal["refit_policy_sha256"]),
                "bundle_factors": [
                    {
                        "candidate_id": str(item["candidate_id"]),
                        "code_sha256": str(item["code_sha256"]),
                        "code_path": runtime_path(str(item["code_path"])),
                    }
                    for item in model_signal.get("bundle_factors") or []
                ],
            }
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            script = self.project_root / "scripts" / "run_model_refit.py"
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
                    runtime_path(output),
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
                "execution_dataset": ((payload.get("execution_dataset") or {}).get("name")),
                "periods": experiment["periods"],
                "parameter_grid": experiment["parameter_grid"],
                "factors": [
                    {
                        "candidate_id": item["factor_candidate_id"],
                        "values_path": runtime_path(item["values_path"]),
                        "code_path": (
                            runtime_path(item["code_path"]) if item.get("code_path") else None
                        ),
                        "code_sha256": item["code_sha256"],
                        "factor_execution_mode": (
                            "frozen_code_recompute"
                            if item.get("source_iteration") is not None
                            else "frozen_values"
                        ),
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
            hypothesis_evidence = self.strategies.hypothesis_group_evidence(version["id"])
            final_periods = {
                "start": payload["periods"]["start"],
                "end": payload["periods"]["end"],
            }
            historical_validation_periods = {
                "start": payload["periods"]["historical_start"],
                "end": payload["periods"]["historical_end"],
            }
            with self.strategies.engine.connect() as connection:
                model_signal = self.strategies._model_signal_evidence(connection, version["config"])
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
                    (execution_dataset.get("provenance") or {}).get("execution_contract_version")
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
                        "definition_sha256": version["config"].get("baseline_definition_sha256"),
                    }
                    if version["config"].get("baseline_definition")
                    else None
                ),
                "strategy_trial_count": hypothesis_evidence["shared_experiment_count"],
                "economic_hypothesis_group": hypothesis_evidence["economic_hypothesis_group"],
                "hypothesis_group_evidence": hypothesis_evidence,
                "periods": final_periods,
                "historical_validation_periods": historical_validation_periods,
                "config": version["config"],
                "model_signal": (model_signal["identity"] if model_signal is not None else None),
                "model_formal_admission": (
                    model_signal["formal_admission_binding"] if model_signal is not None else None
                ),
                "model_candidate": (
                    {
                        "candidate_manifest": dict(model_signal["candidate"].manifest_json or {}),
                        "feature_set": {
                            "id": str(
                                (model_signal["candidate"].base_features_manifest_json or {}).get(
                                    "feature_set_id"
                                )
                                or ""
                            ),
                            "contract_version": str(
                                (model_signal["candidate"].base_features_manifest_json or {}).get(
                                    "contract_version"
                                )
                                or ""
                            ),
                            "features": dict(
                                (model_signal["candidate"].base_features_manifest_json or {}).get(
                                    "feature_expressions"
                                )
                                or {}
                            ),
                            "definition_sha256": str(
                                model_signal["candidate"].feature_set_definition_sha256
                            ),
                        },
                        "code_path": runtime_path(model_signal["code_path"]),
                        "training_periods": {
                            "train_start": model_signal["evaluation"].train_start.isoformat(),
                            "train_end": model_signal["evaluation"].train_end.isoformat(),
                            "valid_start": model_signal["evaluation"].valid_start.isoformat(),
                            "valid_end": model_signal["evaluation"].valid_end.isoformat(),
                            "seed": int(model_signal["evaluation"].seed),
                        },
                        "primary_profile_id": str(
                            model_signal["evaluation"].profile_id
                        ),
                        "refit_policy": version["config"].get("model_refit_policy"),
                        "refit_policy_sha256": version["config"].get(
                            "model_refit_policy_sha256"
                        ),
                    }
                    if model_signal is not None
                    else None
                ),
                "model_bundle_factors": (
                    [
                        {
                            **{
                                key: item[key]
                                for key in (
                                    "candidate_id",
                                    "feature_name",
                                    "code_sha256",
                                    "direction",
                                    "weight",
                                    "factor_execution_mode",
                                )
                            },
                            "code_path": runtime_path(str(item["code_path"])),
                        }
                        for item in model_signal["bundle_factors"]
                    ]
                    if model_signal is not None
                    else []
                ),
                "min_daily_instruments": int(payload.get("min_daily_instruments", 50)),
                "factors": [
                    {
                        "candidate_id": item["factor_candidate_id"],
                        "values_path": runtime_path(item["values_path"]),
                        "code_path": (
                            runtime_path(item["code_path"]) if item.get("code_path") else None
                        ),
                        "code_sha256": item["code_sha256"],
                        "factor_execution_mode": (
                            "frozen_code_recompute"
                            if item.get("source_iteration") is not None
                            else "frozen_values"
                        ),
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
                    "simulation order-plan generation requires a successful formal Qlib backtest"
                )
            datasets = {
                item["name"]: item
                for item in list_qlib_datasets(self.settings.data_root)
                if item.get("ready")
            }
            anchor = datasets.get(portfolio["daily_dataset"])
            if anchor is None:
                raise ValueError("simulation order-plan Qlib daily dataset is unavailable")
            anchor_provenance = dict(anchor.get("provenance") or {})
            if (
                anchor_provenance.get("dataset_identity_sha256")
                != portfolio["daily_dataset_identity_sha256"]
                or anchor_provenance.get("dataset_lineage_id")
                != portfolio["daily_dataset_lineage_id"]
            ):
                raise ValueError(
                    "simulation order-plan Qlib dataset no longer matches the "
                    "bound account snapshot"
                )
            local_today = datetime.now(UTC).astimezone(ZoneInfo("Asia/Shanghai")).date()
            anchor_date = qlib_trading_date_on_or_before(anchor, local_today)
            dataset = select_qlib_dataset(
                self.settings.data_root,
                anchor_name=str(portfolio["daily_dataset"]),
                roll_policy=str(portfolio.get("daily_roll_policy") or "pinned"),
                lineage_id=portfolio.get("daily_dataset_lineage_id"),
                required_date=anchor_date,
            )
            provenance = dict(dataset.get("provenance") or {})
            signal_date = date.fromisoformat(str(payload["signal_date"]))
            current_available_date = qlib_trading_date_on_or_before(dataset, local_today)
            if signal_date != current_available_date:
                raise ValueError(
                    "paper signal is not the latest currently available governed trading day"
                )
            promotion_stage = self.promotions.require_paper_signal(
                str(version["id"]),
                portfolio_id=str(portfolio["id"]),
                signal_date=signal_date,
            )
            if (
                str(payload.get("promotion_stage_id") or "") != promotion_stage["id"]
                or str(payload.get("promotion_stage_opened_at") or "")
                != promotion_stage["opened_at"]
            ):
                raise ValueError("paper order-plan job changed its promotion-stage binding")
            signal_frequency = str(version.get("signal_frequency") or "day").lower()
            signal_at = payload.get("signal_at")
            execution_not_before: str | None = None
            signal_dataset: dict | None = None
            if signal_frequency != "day":
                if not signal_at:
                    raise ValueError("minute simulation order-plan requires signal_at")
                try:
                    signal_timestamp = datetime.fromisoformat(str(signal_at).replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError("minute simulation order-plan signal_at is invalid") from exc
                if signal_timestamp.tzinfo is None or signal_timestamp.utcoffset() is None:
                    raise ValueError("minute simulation order-plan signal_at requires a timezone")
                local_signal = signal_timestamp.astimezone(ZoneInfo("Asia/Shanghai"))
                if local_signal.date().isoformat() != str(payload["signal_date"]):
                    raise ValueError(
                        "minute simulation order-plan signal_at does not match signal_date"
                    )
                source_lineage = str(provenance.get("source_lineage_id") or "")
                candidates = [
                    item
                    for item in datasets.values()
                    if item.get("reproducible") is True
                    and dict(item.get("provenance") or {}).get("lineage_verified") is True
                    and str(dict(item.get("provenance") or {}).get("frequency") or "")
                    == signal_frequency
                    and str(dict(item.get("provenance") or {}).get("source_lineage_id") or "")
                    == source_lineage
                ]
                signal_dataset = next(
                    (item for item in candidates if item["name"] == portfolio["execution_dataset"]),
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
                    "weight": max(0.0, float(item.get("market_value") or 0.0)) / nav,
                }
                for item in positions
                if nav > 0
                and str(item.get("position_side") or "long") == "long"
                and float(item.get("market_value") or 0.0) > 0
            ]
            strategy_risk_state = self.allocations.strategy_risk_state(str(version["id"]))
            required_nav_date = qlib_trading_date_on_or_before(
                dataset,
                date.fromisoformat(str(payload["signal_date"])),
            )
            account_risk_state = self.simulations.policy_risk_inputs(
                str(portfolio["id"]),
                required_nav_date=required_nav_date,
            )
            output = self.settings.data_root / "artifacts" / "order-plan-jobs" / job["id"]
            output.mkdir(parents=True, exist_ok=True)
            manifest_path = output / "manifest.json"
            result_path = output / "result.json"
            is_wsl = os.name == "nt" and self.settings.qlib_python.startswith("/")

            def runtime_path(value: str | Path) -> str:
                return _to_wsl_path(Path(value)) if is_wsl else str(value)

            model_artifact = None
            if str(version["config"].get("signal_source") or "factor_score") == (
                "model_prediction"
            ):
                model_artifact = self.model_artifacts.require_for_inference(
                    str(version["id"]),
                    dataset_identity_sha256=str(provenance["dataset_identity_sha256"]),
                )

            manifest = {
                "artifact_kind": "simulation_order_plan",
                "order_plan_job_id": job["id"],
                "simulation_portfolio_id": portfolio["id"],
                "portfolio_id": portfolio["id"],
                "strategy_version_id": version["id"],
                "formal_backtest_id": formal["id"],
                "dataset": dataset["name"],
                "dataset_identity_sha256": provenance["dataset_identity_sha256"],
                "dataset_lineage_id": provenance["dataset_lineage_id"],
                "promotion_stage_id": promotion_stage["id"],
                "promotion_stage_opened_at": promotion_stage["opened_at"],
                "signal_date": payload["signal_date"],
                "signal_at": signal_at,
                "execution_not_before": execution_not_before,
                "as_of_date": signal_at or payload["signal_date"],
                "benchmark": version["benchmark"],
                "universe": version["universe"],
                "config": version["config"],
                "model_artifact": (
                    {
                        **model_artifact,
                        "artifact_path": runtime_path(model_artifact["artifact_path"]),
                    }
                    if model_artifact is not None
                    else None
                ),
                "construction_notional": nav,
                "risk_exposure": float(strategy_risk_state["risk_exposure_override"]),
                "risk_exposure_override": float(strategy_risk_state["risk_exposure_override"]),
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
                        "dataset_identity_sha256": dict(signal_dataset.get("provenance") or {}).get(
                            "dataset_identity_sha256"
                        ),
                        "dataset_lineage_id": dict(signal_dataset.get("provenance") or {}).get(
                            "dataset_lineage_id"
                        ),
                        "source_lineage_id": dict(signal_dataset.get("provenance") or {}).get(
                            "source_lineage_id"
                        ),
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
                    runtime_path(self.settings.data_root / "artifacts" / "order-plans"),
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
            recommendation_date = date.fromisoformat(str(payload["as_of_date"]))
            current_available_date = qlib_trading_date_on_or_before(
                {"path": payload["dataset_path"]},
                datetime.now(UTC).astimezone(ZoneInfo("Asia/Shanghai")).date(),
            )
            if recommendation_date != current_available_date:
                raise ValueError(
                    "recommendation refresh is not the latest currently available "
                    "governed trading day"
                )
            self.promotions.require_recommendation_signal(
                str(version["id"]), signal_date=recommendation_date
            )
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

            model_artifact = None
            if str(version["config"].get("signal_source") or "factor_score") == (
                "model_prediction"
            ):
                model_artifact = self.model_artifacts.require_for_inference(
                    str(version["id"]),
                    dataset_identity_sha256=str(payload["dataset_identity_sha256"]),
                )

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
                "model_artifact": (
                    {
                        **model_artifact,
                        "artifact_path": runtime_path(model_artifact["artifact_path"]),
                    }
                    if model_artifact is not None
                    else None
                ),
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
                    if resolved["manifest_sha256"] != binding.get("manifest_sha256") or resolved[
                        "source_sha256"
                    ] != binding.get("source_sha256"):
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
                                "dividend snapshot no longer matches the bound daily dataset"
                            )
                        dividend_dataset = resolved_dividend
            if shortability_dataset is not None:
                shortability_path = Path(shortability_dataset["dataset_path"])
                command.extend(
                    [
                        "--shortability-path",
                        _to_wsl_path(shortability_path) if is_wsl else str(shortability_path),
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
                        _to_wsl_path(dividend_path) if is_wsl else str(dividend_path),
                    ]
                )
            return command, result_path, {}
        raise ValueError(f"unsupported job kind: {job['kind']}")

    def _queue_data_pipeline_successor(self, job: dict) -> dict:
        payload = dict(job["payload"])
        snapshot_name = str(payload["snapshot_name"])
        pipeline_snapshot_name = str(
            payload.get("pipeline_snapshot_name") or snapshot_name
        )
        if not pipeline_snapshot_name:
            raise ValueError("data pipeline snapshot namespace must not be empty")
        if job["kind"] == "data_snapshot" and payload.get("profile") == "research-assets":
            # The isolated metadata snapshot is the terminal publication for
            # PDF acquisition.  It is intentionally not a Qlib market dataset.
            return job
        # ``start``/``end`` may be a single step's incremental download
        # window. Keep the publication boundary separate and immutable across
        # every durable successor. Legacy payloads without the explicit fields
        # retain their former behavior by freezing their current window.
        snapshot_start = str(payload.get("snapshot_start") or payload["start"])
        snapshot_end = str(payload.get("snapshot_end") or payload["end"])
        pipeline_max_attempts = job.get("max_attempts", 1)
        if (
            isinstance(pipeline_max_attempts, bool)
            or not isinstance(pipeline_max_attempts, int)
            or not 1 <= pipeline_max_attempts <= 5
        ):
            raise ValueError("data pipeline max attempts must be an integer from 1 to 5")
        pipeline_steps = payload.get("pipeline_steps")
        next_index = int(payload.get("pipeline_next_index", 0))
        if isinstance(pipeline_steps, list) and next_index < len(pipeline_steps):
            step = pipeline_steps[next_index]
            if not isinstance(step, dict) or not isinstance(step.get("payload", {}), dict):
                raise ValueError("data pipeline contains an invalid step")
            step_payload = dict(step.get("payload", {}))
            step_pipeline_snapshot_name = step_payload.get("pipeline_snapshot_name")
            if (
                step_pipeline_snapshot_name is not None
                and str(step_pipeline_snapshot_name) != pipeline_snapshot_name
            ):
                raise ValueError("data pipeline step cannot change its snapshot namespace")
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
                "multiface_audit",
                *(
                    f"supplemental_{bundle}"
                    for bundle in sorted(SUPPORTED_BUNDLES)
                ),
            }
            if kind not in allowed:
                raise ValueError(f"unsupported data pipeline step: {kind}")
            successor_payload = {
                "pipeline_id": payload["pipeline_id"],
                "profile": payload["profile"],
                "start": snapshot_start,
                "end": snapshot_end,
                "snapshot_name": payload["snapshot_name"],
                "pipeline_steps": pipeline_steps,
                "pipeline_next_index": next_index + 1,
                **step_payload,
                "pipeline_snapshot_name": pipeline_snapshot_name,
                "snapshot_start": snapshot_start,
                "snapshot_end": snapshot_end,
            }
            if kind == "minute_qlib":
                snapshot = resolve_snapshot_manifest(
                    self.settings.data_root,
                    str(successor_payload["snapshot_name"]),
                )
                successor_payload["snapshot_manifest_sha256"] = snapshot["manifest_sha256"]
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
                "start": snapshot_start,
                "end": snapshot_end,
                "snapshot_start": snapshot_start,
                "snapshot_end": snapshot_end,
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
        # Normalize legacy and non-step branches as well, so no successor can
        # accidentally reinterpret a prior step's download range as the
        # publication range.
        successor_payload["snapshot_start"] = snapshot_start
        successor_payload["snapshot_end"] = snapshot_end
        successor_payload["pipeline_snapshot_name"] = pipeline_snapshot_name
        log_path = (
            self.settings.data_root
            / "platform"
            / "logs"
            / f"{kind}-{pipeline_snapshot_name}.log"
        )
        successor = self.store.create(
            kind,
            successor_payload,
            log_path,
            idempotency_key=f"data-finalize:{pipeline_snapshot_name}:{kind}",
            max_attempts=pipeline_max_attempts,
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

    def _archive_rdagent_lab_artifacts(
        self,
        run_id: str,
        scenario_id: str,
        result: dict,
        *,
        sanitized_result_artifact_id: str,
        sanitized_result_sha256: str,
    ) -> dict[str, object]:
        if scenario_id not in {"general_model", "data_science", "llm_finetune"}:
            raise ValueError("only non-capital RD-Agent lab outputs may use this archive")
        root = self.settings.data_root / "artifacts" / "rdagent" / run_id
        result_path = root / "result.json"
        if not result_path.is_file():
            raise ValueError("sanitized RD-Agent lab result is unavailable")
        implementations: list[dict[str, str]] = []
        governed_outputs: list[dict[str, str]] = []
        for index, item in enumerate(result.get("lab_outputs") or [], start=1):
            code_path_value = _local_artifact_path(item.get("code_path"))
            code_path = Path(str(code_path_value or ""))
            expected = str(item.get("code_sha256") or "")
            if not code_path.is_file() or _sha256_path(code_path) != expected:
                raise ValueError("RD-Agent lab implementation artifact changed")
            code_artifact = self.rdagent_candidates.register_run_artifact(
                research_run_id=run_id,
                artifact_type=f"{scenario_id}_implementation_code",
                storage_path=code_path,
                producer="rdagent_bridge",
                actor="worker",
                contract_version="rdagent-lab-implementation-code-v1",
                source_iteration=index,
                metadata={
                    "scenario": scenario_id,
                    "name": item.get("name"),
                    "implementation_ready": bool(item.get("implementation_ready")),
                    "capital_eligible": False,
                },
            )
            implementations.append(
                {
                    "artifact_id": str(code_artifact["id"]),
                    "name": str(item.get("name") or ""),
                    "code_sha256": expected,
                }
            )
        if scenario_id == "data_science":
            submissions = [
                path
                for path in root.rglob("submission.csv")
                if "scenario-inputs" not in path.relative_to(root).parts
            ]
            scores = [
                path
                for path in root.rglob("scores.csv")
                if "scenario-inputs" not in path.relative_to(root).parts
            ]
            if not submissions or not scores:
                raise ValueError(
                    "data_science completed without both submission.csv and scores.csv"
                )
            selected_outputs = [
                ("data_science_submission", submissions[-1]),
                ("data_science_scores", scores[-1]),
            ]
        elif scenario_id == "llm_finetune":
            def generated(paths: list[Path]) -> list[Path]:
                return [
                    path
                    for path in paths
                    if not any(
                        part.startswith("finetune-files-")
                        for part in path.relative_to(root).parts
                    )
                ]

            checkpoints = generated(
                [
                    *root.rglob("*.safetensors"),
                    *root.rglob("adapter_model.bin"),
                    *root.rglob("pytorch_model.bin"),
                ]
            )
            benchmark_results = generated(
                [
                    path
                    for path in root.rglob("*")
                    if path.is_file()
                    and "benchmark_results" in path.relative_to(root).parts
                    and path.suffix.lower() in {".json", ".csv", ".txt"}
                ]
            )
            training_configs = generated(list(root.rglob("train.yaml")))
            staging_evidence = root / "finetune-staging-evidence.json"
            if (
                not checkpoints
                or not benchmark_results
                or not training_configs
                or not staging_evidence.is_file()
            ):
                raise ValueError(
                    "llm_finetune completed without a checkpoint, training config, "
                    "benchmark result, and sealed staging evidence"
                )
            selected_outputs = [
                ("llm_finetune_checkpoint", path) for path in checkpoints
            ] + [
                ("llm_finetune_benchmark_result", benchmark_results[-1]),
                ("llm_finetune_training_config", training_configs[-1]),
                ("llm_finetune_staging_evidence", staging_evidence),
            ]
        else:
            selected_outputs = []
        for artifact_type, path in selected_outputs:
            artifact = self.rdagent_candidates.register_run_artifact(
                research_run_id=run_id,
                artifact_type=artifact_type,
                storage_path=path,
                producer="rdagent_pinned_runtime",
                actor="worker",
                contract_version="rdagent-lab-output-v1",
                metadata={"scenario": scenario_id, "capital_eligible": False},
            )
            governed_outputs.append(
                {
                    "artifact_id": str(artifact["id"]),
                    "artifact_type": artifact_type,
                    "content_sha256": str(artifact["content_sha256"]),
                }
            )
        archive_status = (
            "implementation_ready"
            if scenario_id == "general_model" and implementations
            else "completed"
        )
        audit = {
            "contract_version": "rdagent-lab-audit-v1",
            "scenario": scenario_id,
            "status": archive_status,
            "capital_eligible": False,
            "sanitized_result_artifact_id": sanitized_result_artifact_id,
            "sanitized_result_sha256": sanitized_result_sha256,
            "trace_summary": result.get("trace_summary") or {},
            "rdagent_runtime": result.get("rdagent_runtime") or {},
            "asset_ids": result.get("asset_ids") or [],
            "implementations": implementations,
            "outputs": governed_outputs,
        }
        audit_path = root / "lab-audit.json"
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        audit_artifact = self.rdagent_candidates.register_run_artifact(
            research_run_id=run_id,
            artifact_type=f"{scenario_id}_audit_manifest",
            storage_path=audit_path,
            producer="quantlab_worker",
            actor="worker",
            contract_version="rdagent-lab-audit-v1",
            metadata={"scenario": scenario_id, "status": archive_status},
        )
        return {
            "lab_status": archive_status,
            "sanitized_result_artifact_id": sanitized_result_artifact_id,
            "audit_artifact_id": audit_artifact["id"],
            "implementation_artifacts": implementations,
            "output_artifacts": governed_outputs,
            "capital_eligible": False,
        }

    def _archive_rdagent_run_evidence(
        self, run_id: str, scenario_id: str, result: dict
    ) -> dict[str, object]:
        """Register sanitized output and hashed inventories for every scenario."""

        root = (self.settings.data_root / "artifacts" / "rdagent" / run_id).resolve()
        result_path = (root / "result.json").resolve()
        if not result_path.is_file() or not result_path.is_relative_to(root):
            raise ValueError("sanitized RD-Agent result is unavailable")

        def sanitize(value: object) -> object:
            if isinstance(value, dict):
                return {
                    str(key): sanitize(item)
                    for key, item in value.items()
                    if not str(key).lower().endswith("_path")
                    and not any(
                        token in str(key).lower()
                        for token in ("api_key", "password", "secret", "token")
                    )
                }
            if isinstance(value, list):
                return [sanitize(item) for item in value]
            return value

        sanitized_path = root / "sanitized-result.json"
        sanitized_path.write_text(
            json.dumps(sanitize(result), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        result_artifact = self.rdagent_candidates.register_run_artifact(
            research_run_id=run_id,
            artifact_type=f"{scenario_id}_sanitized_result",
            storage_path=sanitized_path,
            producer="quantlab_worker",
            actor="worker",
            contract_version="rdagent-sanitized-result-v1",
            metadata={"scenario": scenario_id, "capital_eligible": False},
        )

        inventory: list[dict[str, object]] = []
        excluded_roots = {
            "scenario-inputs",
            "isolated-qlib",
        }
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path == sanitized_path:
                continue
            if path.is_symlink():
                raise ValueError("RD-Agent audit evidence cannot contain symbolic links")
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise ValueError("RD-Agent audit evidence escapes its governed run root")
            relative = path.relative_to(root)
            if relative.parts and relative.parts[0] in excluded_roots:
                continue
            # Pickled trace payloads may contain prompts or secrets; record only
            # immutable inventory metadata, never their contents or public paths.
            inventory.append(
                {
                    "relative_path": relative.as_posix(),
                    "size_bytes": resolved.stat().st_size,
                    "sha256": _sha256_path(resolved),
                    "category": (
                        "trace"
                        if relative.parts[:1] == ("trace",)
                        else (
                            "qlib_result"
                            if relative.parts[:1] == ("candidate-values",)
                            else "workspace"
                        )
                    ),
                }
            )
        inventory_payload = {
            "contract_version": "rdagent-trace-inventory-v1",
            "scenario": scenario_id,
            "capital_eligible": False,
            "trace_summary": result.get("trace_summary") or {},
            "files": inventory,
        }
        inventory_path = root / "trace-inventory.json"
        inventory_path.write_text(
            json.dumps(inventory_payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        inventory_artifact = self.rdagent_candidates.register_run_artifact(
            research_run_id=run_id,
            artifact_type=f"{scenario_id}_trace_inventory",
            storage_path=inventory_path,
            producer="quantlab_worker",
            actor="worker",
            contract_version="rdagent-trace-inventory-v1",
            metadata={
                "scenario": scenario_id,
                "file_count": len(inventory),
                "capital_eligible": False,
            },
        )
        return {
            "sanitized_result_artifact_id": str(result_artifact["id"]),
            "sanitized_result_sha256": str(result_artifact["content_sha256"]),
            "trace_inventory_artifact_id": str(inventory_artifact["id"]),
        }

    def _import_rdagent_model_candidates(self, run_id: str, job: dict, result: dict) -> list[dict]:
        payload = job["payload"]
        feature_set = payload.get("feature_set") or {}
        periods = payload.get("periods") or {}
        if not feature_set.get("id") or not periods.get("valid_end"):
            raise ValueError("model candidates have no governed feature set or cutoff")
        imported: list[dict] = []
        for item in result.get("model_candidates") or []:
            code_path_value = _local_artifact_path(item.get("code_path"))
            # Official RD-Agent feedback is research context, never a trial
            # filter. Every executable proposal remains in the shared
            # independent multiple-testing family.
            if not code_path_value:
                continue
            code_path = Path(code_path_value)
            expected_sha256 = str(item.get("code_sha256") or "").lower()
            if (
                not code_path.is_file()
                or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
                or _sha256_path(code_path) != expected_sha256
            ):
                continue
            artifact = self.rdagent_candidates.register_run_artifact(
                research_run_id=run_id,
                artifact_type="rdagent_model_code",
                storage_path=code_path,
                producer="rdagent",
                actor="worker",
                contract_version="rdagent-model-code-v1",
                source_iteration=item.get("source_iteration"),
                metadata={"scenario": "fin_model", "name": item.get("name")},
            )
            candidate = self.rdagent_candidates.create_model_candidate(
                research_run_id=run_id,
                name=str(item.get("name") or f"model-{len(imported) + 1}"),
                description=str(item.get("description") or item.get("name") or "RD-Agent model"),
                model_type=str(item.get("model_type") or "Tabular"),
                code_artifact_id=str(artifact["id"]),
                architecture=dict(item.get("architecture") or {}),
                model_hyperparameters=dict(item.get("model_hyperparameters") or {}),
                training_hyperparameters=dict(item.get("training_hyperparameters") or {}),
                feature_set_id=str(feature_set["id"]),
                dataset=str(payload["dataset"]),
                dataset_identity_sha256=str(payload["dataset_identity_sha256"]),
                dataset_lineage_id=(
                    str(payload["dataset_lineage_id"])
                    if payload.get("dataset_lineage_id")
                    else None
                ),
                pre_final_end=date.fromisoformat(str(periods["valid_end"])),
                final_oos_start=date.fromisoformat(str(periods["test_start"])),
                final_oos_end=date.fromisoformat(str(periods["test_end"])),
                source_iteration=item.get("source_iteration"),
                rdagent_decision=item.get("rdagent_decision"),
                rdagent_feedback=item.get("rdagent_feedback"),
            )
            imported.append(
                {
                    "id": candidate["id"],
                    "code_path": str(code_path),
                    "code_sha256": expected_sha256,
                    "model_type": str(item.get("model_type") or "Tabular"),
                    "training_hyperparameters": dict(item.get("training_hyperparameters") or {}),
                }
            )
        if not imported:
            raise ValueError("RD-Agent produced no executable model candidates")
        return imported

    def _queue_model_evaluation(self, job: dict, candidates: list[dict]) -> None:
        payload = job["payload"]
        feature_set = payload.get("feature_set") or {}
        log_path = (
            self.settings.data_root
            / "platform"
            / "logs"
            / f"model-evaluate-{payload['research_run_id']}.log"
        )
        evaluation_job = self.store.create(
            "model_evaluate",
            {
                "research_run_id": payload["research_run_id"],
                "dataset": payload["dataset"],
                "dataset_path": payload["dataset_path"],
                "dataset_identity_sha256": payload["dataset_identity_sha256"],
                "evaluation_profiles": payload.get("evaluation_profiles") or [],
                "feature_set_id": feature_set["id"],
                "feature_set_definition_sha256": feature_set["definition_sha256"],
                "candidates": candidates,
                "universe": payload.get("universe", "cn_all"),
                "benchmark": payload.get("benchmark", "SH000300"),
            },
            log_path,
            idempotency_key=f"model-evaluate:{payload['research_run_id']}",
        )
        self.research.attach_job(payload["research_run_id"], evaluation_job["id"])

    def _import_model_evaluations(self, job: dict, result: dict) -> None:
        payload = job["payload"]
        artifact_path = (
            self.settings.data_root
            / "artifacts"
            / "model-evaluations"
            / payload["research_run_id"]
            / "result.json"
        )
        artifact = self.rdagent_candidates.register_run_artifact(
            research_run_id=str(payload["research_run_id"]),
            artifact_type="model_independent_evaluation",
            storage_path=artifact_path,
            producer="quantlab_independent_evaluator",
            actor="worker",
            contract_version="model-independent-evaluation-v1",
            metadata={
                "dataset_identity_sha256": payload["dataset_identity_sha256"],
                "feature_set_id": payload["feature_set_id"],
            },
        )
        by_id = {str(item.get("candidate_id")): item for item in result.get("evaluations") or []}
        for candidate in payload["candidates"]:
            candidate_id = str(candidate["id"])
            item = by_id.get(candidate_id)
            if not item or item.get("status") != "passed":
                self.rdagent_candidates.transition_candidate(
                    "model",
                    candidate_id,
                    status="rejected",
                    reason=str((item or {}).get("error") or "independent evaluation failed"),
                    actor="worker",
                )
                continue
            self.rdagent_candidates.ingest_model_evaluation_result(
                model_candidate_id=candidate_id,
                run_artifact_id=str(artifact["id"]),
                actor="worker",
            )

    def _queue_quant_bundle_evaluation(self, job: dict, result: dict) -> int:
        payload = job["payload"]
        feature_set = payload.get("feature_set") or {}
        periods = payload.get("periods") or {}
        eligible: list[dict] = []
        model_artifacts: dict[str, dict] = {}
        for bundle in result.get("quant_bundles") or []:
            factors = []
            for factor in bundle.get("factors") or []:
                code_path = Path(str(_local_artifact_path(factor.get("code_path"))))
                if not code_path.is_file() or _sha256_path(code_path) != factor.get("code_sha256"):
                    raise ValueError("RD-Agent quant factor artifact is invalid")
                values_path = _local_artifact_path(factor.get("submitted_values_path"))
                factors.append(
                    {
                        **factor,
                        "code_path": str(code_path),
                        "submitted_values_path": values_path,
                    }
                )
            model = dict(bundle.get("model") or {})
            model_path = Path(str(_local_artifact_path(model.get("code_path"))))
            if (
                not factors
                or not model_path.is_file()
                or _sha256_path(model_path) != model.get("code_sha256")
                or not re.fullmatch(r"[0-9a-f]{64}", str(model.get("recipe_sha256") or ""))
            ):
                raise ValueError("RD-Agent produced an incomplete quant bundle")
            model_sha256 = str(model["code_sha256"])
            model_artifact = model_artifacts.get(model_sha256)
            if model_artifact is None:
                model_artifact = self.rdagent_candidates.register_run_artifact(
                    research_run_id=str(payload["research_run_id"]),
                    artifact_type="rdagent_quant_model_code",
                    storage_path=model_path,
                    producer="rdagent",
                    actor="worker",
                    contract_version="rdagent-quant-model-code-v1",
                    source_iteration=bundle.get("source_iteration"),
                    metadata={"scenario": "fin_quant"},
                )
                model_artifacts[model_sha256] = model_artifact
            model_candidate = self.rdagent_candidates.create_model_candidate(
                research_run_id=str(payload["research_run_id"]),
                name=f"{bundle.get('name') or bundle['id']}-model",
                description=str(bundle.get("description") or "RD-Agent fin_quant model"),
                model_type=str(model.get("model_type") or "Tabular"),
                code_artifact_id=str(model_artifact["id"]),
                architecture=dict(model.get("architecture") or {}),
                model_hyperparameters=dict(model.get("model_hyperparameters") or {}),
                training_hyperparameters=dict(model.get("training_hyperparameters") or {}),
                feature_set_id=str(feature_set["id"]),
                dataset=str(payload["dataset"]),
                dataset_identity_sha256=str(payload["dataset_identity_sha256"]),
                pre_final_end=date.fromisoformat(str(periods["valid_end"])),
                final_oos_start=date.fromisoformat(str(periods["test_start"])),
                final_oos_end=date.fromisoformat(str(periods["test_end"])),
                source_iteration=bundle.get("source_iteration"),
                rdagent_decision=bundle.get("rdagent_decision"),
                rdagent_feedback=bundle.get("rdagent_feedback"),
            )
            proposal_root = (
                self.settings.data_root
                / "artifacts"
                / "rdagent"
                / str(payload["research_run_id"])
                / "quant-proposals"
            )
            proposal_root.mkdir(parents=True, exist_ok=True)
            proposal_path = proposal_root / f"{bundle['id']}.json"
            proposal_path.write_text(
                json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            proposal_artifact = self.rdagent_candidates.register_run_artifact(
                research_run_id=str(payload["research_run_id"]),
                artifact_type="rdagent_quant_bundle_proposal",
                storage_path=proposal_path,
                producer="rdagent",
                actor="worker",
                contract_version="rdagent-quant-bundle-proposal-v1",
                source_iteration=bundle.get("source_iteration"),
                metadata={
                    "experiment_family_id": bundle["experiment_family_id"],
                    "feature_set_id": feature_set["id"],
                },
            )
            governed = self.rdagent_candidates.create_joint_quant_bundle_candidate(
                research_run_id=str(payload["research_run_id"]),
                name=str(bundle.get("name") or bundle["id"]),
                description=str(bundle.get("description") or "RD-Agent fin_quant bundle"),
                model_candidate_id=str(model_candidate["id"]),
                factors=factors,
                bundle_artifact_id=str(proposal_artifact["id"]),
                experiment_family_id=str(bundle["experiment_family_id"]),
                feature_set_id=str(feature_set["id"]),
                dataset=str(payload["dataset"]),
                dataset_identity_sha256=str(payload["dataset_identity_sha256"]),
                pre_final_end=date.fromisoformat(str(periods["valid_end"])),
                final_oos_start=date.fromisoformat(str(periods["test_start"])),
                final_oos_end=date.fromisoformat(str(periods["test_end"])),
                source_iteration=bundle.get("source_iteration"),
                rdagent_decision=bundle.get("rdagent_decision"),
                rdagent_feedback=bundle.get("rdagent_feedback"),
            )
            frozen_factors = {
                str(item["code_sha256"]): item
                for item in governed["bundle_manifest_json"]["factors"]
            }
            eligible.append(
                {
                    "id": str(governed["id"]),
                    "experiment_family_id": str(bundle["experiment_family_id"]),
                    "feature_set_id": str(feature_set["id"]),
                    "factors": [
                        {
                            **factor,
                            "candidate_id": frozen_factors[str(factor["code_sha256"])][
                                "candidate_id"
                            ],
                        }
                        for factor in factors
                    ],
                    "model": {
                        **model,
                        "code_path": str(model_path),
                        "recipe_sha256": governed["bundle_manifest_json"]["model"]["recipe_sha256"],
                    },
                }
            )
        if not eligible:
            raise ValueError("RD-Agent produced no executable factor/model bundle")
        log_path = (
            self.settings.data_root
            / "platform"
            / "logs"
            / f"quant-bundle-evaluate-{payload['research_run_id']}.log"
        )
        evaluation_job = self.store.create(
            "quant_bundle_evaluate",
            {
                "research_run_id": payload["research_run_id"],
                "dataset": payload["dataset"],
                "dataset_path": payload["dataset_path"],
                "dataset_identity_sha256": payload["dataset_identity_sha256"],
                "evaluation_profiles": payload.get("evaluation_profiles") or [],
                "feature_set_id": feature_set["id"],
                "feature_set_definition_sha256": feature_set["definition_sha256"],
                "candidates": eligible,
                "universe": payload.get("universe", "cn_all"),
                "benchmark": payload.get("benchmark", "SH000300"),
            },
            log_path,
            idempotency_key=f"quant-bundle-evaluate:{payload['research_run_id']}",
        )
        self.research.attach_job(payload["research_run_id"], evaluation_job["id"])
        return len(eligible)

    def _import_quant_bundle_evaluation_artifact(self, job: dict, result: dict) -> None:
        payload = job["payload"]
        if result.get("status") != "ok":
            raise ValueError("quant bundle independent evaluation did not complete")
        artifact_path = (
            self.settings.data_root
            / "artifacts"
            / "quant-bundle-evaluations"
            / payload["research_run_id"]
            / "result.json"
        )
        artifact = self.rdagent_candidates.register_run_artifact(
            research_run_id=str(payload["research_run_id"]),
            artifact_type="quant_bundle_independent_evaluation",
            storage_path=artifact_path,
            producer="quantlab_independent_evaluator",
            actor="worker",
            contract_version="quant-bundle-independent-evaluation-v1",
            metadata={
                "dataset_identity_sha256": payload["dataset_identity_sha256"],
                "feature_set_id": payload["feature_set_id"],
                "passed": sum(
                    1 for item in result.get("evaluations") or [] if item.get("status") == "passed"
                ),
            },
        )
        evaluated_ids = {str(item.get("candidate_id")) for item in result.get("evaluations") or []}
        expected_ids = {str(item["id"]) for item in payload["candidates"]}
        if evaluated_ids != expected_ids:
            raise ValueError("quant evaluation candidate set disagrees with the job")
        for candidate_id in sorted(expected_ids):
            self.rdagent_candidates.ingest_quant_bundle_evaluation_result(
                quant_bundle_candidate_id=candidate_id,
                run_artifact_id=str(artifact["id"]),
                actor="worker",
            )

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
                "evaluation_profiles": payload.get("evaluation_profiles") or [],
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
        artifact_path = (
            self.settings.data_root
            / "artifacts"
            / "factor-evaluations"
            / payload["research_run_id"]
            / "result.json"
        )
        evaluations = sorted(
            result.get("evaluations", []),
            key=lambda item: str(
                ((item.get("metrics") or {}).get("research_profile") or {}).get("id") or ""
            ),
        )
        for item in evaluations:
            period_values = item.get("periods") or payload["periods"]
            periods = {key: date.fromisoformat(value) for key, value in period_values.items()}
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
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) for name in names)
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
            candidate = self.research.find_candidate(name=name, values_sha256=values_sha256)
            if candidate is None:
                raise ValueError(f"registered information factor candidate is missing: {name}")
            candidate_status = str(candidate.get("status") or "")
            if candidate_status not in {"awaiting_evaluation", "evaluation_failed"}:
                latest_evaluation = candidate.get("latest_evaluation")
                # An unchanged factor artifact does not owe another evaluation
                # on the same immutable dataset.  A newly published Qlib
                # identity is different evidence, though, so terminal
                # gate outcomes from the prior identity must be recomputed.
                # Unknown/rejected/promoted lifecycle states remain excluded.
                if (
                    candidate_status not in {
                        "gate_passed",
                        "gate_failed",
                        "insufficient_evidence",
                    }
                    or not isinstance(latest_evaluation, dict)
                    or latest_evaluation.get("dataset_identity_sha256")
                    == payload.get("dataset_identity_sha256")
                ):
                    continue
            variables = candidate.get("variables")
            source = variables.get("source") if isinstance(variables, dict) else None
            if not isinstance(source, dict) or not str(source.get("dataset") or "").strip():
                raise ValueError(f"information factor {name} has no governed source")
            required = ("code_path", "values_path", "code_sha256", "values_sha256")
            if any(not candidate.get(key) for key in required):
                raise ValueError(f"information factor {name} misses immutable artifacts")
            if (
                not Path(str(candidate["code_path"])).is_file()
                or not Path(str(candidate["values_path"])).is_file()
            ):
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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
