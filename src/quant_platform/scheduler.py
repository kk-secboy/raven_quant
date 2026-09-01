from __future__ import annotations

import json
import shutil
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import cast, select
from sqlalchemy.dialects.postgresql import JSONB

from quant_data.cninfo_announcements import load_trade_calendar_open_days
from quant_data.config import Settings
from quant_data.coverage_data import COVERAGE_BUNDLES, DEFAULT_COVERAGE_BUNDLES
from quant_data.database import (
    jobs,
    model_artifacts,
    recommendation_portfolios,
    recommendation_snapshots,
    research_runs,
    row_dict,
    simulation_batches,
    simulation_nav,
    simulation_portfolios,
    strategy_allocation_events,
    strategy_health_snapshots,
    strategy_versions,
)
from quant_data.execution_contract import require_daily_qlib_contract
from quant_data.research_assets import research_report_dates_in_snapshot

from .advice_alerts import UnifiedAccountAdviceAlertProjector
from .alert_store import AlertStore
from .autopilot import AutopilotController
from .data_rollover import qlib_trading_date_on_or_before, select_qlib_dataset
from .feature_set_registry import get_feature_set
from .fin_strategy_schedule import (
    canonical_sha256 as fin_strategy_schedule_sha256,
)
from .fin_strategy_schedule import (
    managed_fin_strategy_due_event,
    reconcile_managed_fin_strategy_schedules,
    select_latest_reproducible_daily_dataset,
    validate_managed_fin_strategy_payload,
)
from .health_store import OperationalHealthStore
from .information_schedule import (
    STRUCTURED_INFORMATION_STARTS,
    latest_verified_research_asset_snapshot,
    latest_verified_snapshot,
    normalize_information_factor_refresh_payload,
    normalize_information_schedule_payload,
    resolve_information_evaluation_dataset,
)
from .job_store import (
    ORDER_PLAN_CANCELLED,
    ORDER_PLAN_EXECUTION_TRADE_DATE_KEY,
    ORDER_PLAN_FAILED,
    ORDER_PLAN_SUPERSEDED,
    JobStore,
    research_asset_acquisition_idempotency_key,
)
from .model_artifact_store import ModelArtifactStore
from .model_drift import build_persistent_drift_evidence
from .model_research_governance import canonical_sha256
from .ops_calendar import (
    evaluate_recommendation_gate,
    is_monthly_decision_day,
    is_weekly_report_day,
    load_calendar_days,
    select_ops_dataset,
)
from .promotion import PromotionStore
from .rdagent_candidate_store import RDAGentCandidateStore
from .rdagent_runtime import expected_rdagent_runtime_identity, probe_rdagent
from .rdagent_scenarios import (
    FROZEN_RDAGENT_SCENARIOS,
    get_rdagent_scenario,
    require_ready_scenario,
    resolve_rdagent_assets,
)
from .recommendation_store import (
    RecommendationStore,
    recommendation_refresh_job_idempotency_key,
    recommendation_refresh_job_payload,
)
from .research_asset_store import ResearchAssetStore
from .research_automation import (
    HORIZON_RESEARCH_SCENARIOS,
    normalize_research_schedule_payload,
    resolve_research_periods,
    resolve_research_window_contract,
)
from .research_horizon import (
    primary_label_horizon_sessions,
    primary_label_policy_contract,
)
from .research_report_backfill import ResearchReportBackfillStore
from .research_store import ResearchStore
from .runtime_secret_store import RuntimeSecretStore
from .safe_mode import SafeModeActiveError, SafeModeStore
from .schedule_store import ScheduleStore
from .services import list_qlib_datasets
from .simulation_store import ExecutionDataNotReadyError, SimulationStore
from .strategy_feature_drift_source import StrategyFeatureDriftSource
from .strategy_health import resolve_feature_drift_episode
from .strategy_health_collector import StrategyHealthCollector
from .strategy_research_signal_binding import (
    build_strategy_research_signal_binding,
    research_feature_set_for_champion_selection,
)
from .strategy_store import StrategyStore
from .three_horizon_account import ThreeHorizonAccountService

AUTOMATED_DATA_BUNDLES = (
    "cn_extended_daily",
    "cn_funds",
    "cn_macro",
    "cn_futures",
    "cn_options_bonds",
    "hk_market",
    "us_market",
    "global_markets",
    "cn_institutional",
    *sorted(DEFAULT_COVERAGE_BUNDLES),
)

INFORMATION_CONFLICTING_JOB_KINDS = (
    "bootstrap",
    "legacy_market_backfill",
    "margin_eligibility_download",
    "core_intraday_download",
    "ashare_5m_download",
    "cninfo_announcements_download",
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
    "data_verify",
    "data_snapshot",
    "data_qlib",
    "minute_qlib",
    "qlib_baseline",
    *(f"supplemental_{bundle}" for bundle in set(AUTOMATED_DATA_BUNDLES) | COVERAGE_BUNDLES),
)


def factor_materialization_manifest_matches(
    manifest: dict[str, Any],
    *,
    dataset_identity_sha256: str,
    feature_set: dict[str, Any],
    start: str,
    end: str,
) -> bool:
    """Match a frozen materialization request without invalidating v1 artifacts.

    The original unified 533-factor v1 manifest predates the optional embedded
    ``feature_set`` and compact ``recent`` files.  Its original identity fields
    remain sufficient.  Strategy-health materializations are new, private
    inputs and require both additions because the collector reads those compact
    files and must prove the exact per-version expression set.
    """

    if not (
        manifest.get("dataset_identity_sha256") == dataset_identity_sha256
        and manifest.get("feature_set_id") == feature_set["id"]
        and manifest.get("feature_set_definition_sha256")
        == feature_set["definition_sha256"]
        and manifest.get("universe") == "cn_all"
        and manifest.get("start") == start
        and manifest.get("end") == end
        and manifest.get("status") in {"complete", "complete_with_blockers"}
    ):
        return False
    if not str(feature_set["id"]).startswith("strategy-health:"):
        return True
    frozen = {
        key: feature_set[key]
        for key in (
            "contract_version",
            "id",
            "name",
            "features",
            "source",
            "materialization_contract",
            "definition_sha256",
        )
    }
    completed = manifest.get("completed")
    if (
        manifest.get("feature_set") != frozen
        or manifest.get("storage_mode") != "recent_only"
        or int(manifest.get("session_limit") or 0) != 64
        or manifest.get("requested_start") != start
        or manifest.get("requested_end") != end
        or manifest.get("materialized_end") != end
        or not isinstance(completed, dict)
    ):
        return False
    required_entry_fields = {
        "recent_relative_path",
        "recent_sha256",
        "recent_start",
        "recent_end",
        "recent_session_limit",
    }
    return all(
        _strategy_health_recent_entry_matches(
            completed.get(factor_id),
            required_fields=required_entry_fields,
            expected_end=end,
        )
        for factor_id in feature_set["features"]
    )


def _strategy_health_recent_entry_matches(
    value: Any,
    *,
    required_fields: set[str],
    expected_end: str,
) -> bool:
    if not isinstance(value, dict) or not required_fields.issubset(value):
        return False
    try:
        recent_start = date.fromisoformat(str(value["recent_start"]))
        recent_end = date.fromisoformat(str(value["recent_end"]))
    except ValueError:
        return False
    return (
        int(value["recent_session_limit"]) == 64
        and recent_start <= recent_end
        and recent_end.isoformat() == expected_end
        and "relative_path" not in value
        and "sha256" not in value
    )


def model_refresh_decision(
    *,
    signal_date: date,
    calendar_days: set[date],
    dataset_name: str,
    dataset_identity_sha256: str,
) -> dict[str, Any]:
    """Choose predict-vs-fit without guessing from weekdays.

    A model is fitted only on the first persisted trading day of a month.
    Other sessions are immutable-checkpoint inference unless a separately
    validated drift/data-contract trigger explicitly requests an early fit.
    """

    monthly_retrain = is_monthly_decision_day(signal_date, calendar_days)
    evidence = (
        {
            "calendar_dataset": dataset_name,
            "calendar_dataset_identity_sha256": dataset_identity_sha256,
            "is_first_trading_day": True,
            "signal_date": signal_date.isoformat(),
            "signal_month": signal_date.strftime("%Y-%m"),
            "trigger": "monthly_first_trading_day",
        }
        if monthly_retrain
        else None
    )
    return {
        "operation": "retrain" if monthly_retrain else "inference",
        "retrain_reason": "monthly_first_trading_day" if monthly_retrain else "",
        "retrain_evidence": evidence,
        "retrain_evidence_sha256": (
            canonical_sha256(evidence) if evidence is not None else ""
        ),
    }

FACTOR_LIBRARY_MIN_FREE_BYTES = 300 * 1024**3
FACTOR_LIBRARY_MATERIALIZATION_RETRY_CONTRACT_VERSION = (
    "factor-library-materialization-retry-v1"
)
FACTOR_LIBRARY_MATERIALIZATION_MAX_EPISODES = 3
FACTOR_LIBRARY_MATERIALIZATION_ATTEMPTS_PER_EPISODE = 2


class ScheduleRunWaiting(RuntimeError):
    """A durable schedule dependency may become ready before its slot deadline."""


def _download_runtime_limits(payload: dict[str, Any]) -> dict[str, int]:
    workers = payload.get("download_workers", 4)
    requests_per_minute = payload.get("requests_per_minute", 99)
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 16:
        raise ValueError("download_workers must be an integer from 1 to 16")
    if (
        isinstance(requests_per_minute, bool)
        or not isinstance(requests_per_minute, int)
        or not 1 <= requests_per_minute <= 99
    ):
        raise ValueError("requests_per_minute must be an integer from 1 to 99")
    return {
        "download_workers": workers,
        "requests_per_minute": requests_per_minute,
    }


def _latest_lineage_coverage_date(
    datasets: dict[str, dict[str, Any]], *, lineage_id: str, as_of: date
) -> date | None:
    values: list[date] = []
    for item in datasets.values():
        if str(item.get("lineage_id") or "") != lineage_id:
            continue
        try:
            end = date.fromisoformat(str(item["end_date"]))
        except (KeyError, ValueError):
            continue
        if end <= as_of:
            values.append(end)
    return max(values) if values else None


def _is_latest_due_weekday_slot(run: dict[str, Any], now: datetime) -> bool:
    """Return whether a daily slot is the latest due weekday at ``now``.

    Full market-data recovery follows the normal scheduler's narrow calendar
    rule: weekends are known closures, while exchange holidays remain for the
    refreshed ``trade_cal`` and downstream quality gates to decide.
    """

    zone = ZoneInfo(str(run["timezone"]))
    scheduled = datetime.fromisoformat(str(run["scheduled_for"])).astimezone(zone)
    local_now = now.astimezone(zone)
    latest_date = local_now.date()
    if local_now.timetz().replace(tzinfo=None) < scheduled.timetz().replace(tzinfo=None):
        latest_date -= timedelta(days=1)
    while latest_date.weekday() >= 5:
        latest_date -= timedelta(days=1)
    return scheduled.date() == latest_date


class SchedulerEngine:
    """Materializes daily slots and safely enqueues durable platform jobs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs = JobStore(settings.database_url)
        self.recommendations = RecommendationStore(settings.database_url)
        self.research = ResearchStore(settings.database_url)
        self.rdagent_candidates = RDAGentCandidateStore(settings.database_url)
        self.research_assets = ResearchAssetStore(settings.database_url)
        self.research_report_backfill = ResearchReportBackfillStore(settings.database_url)
        self.schedules = ScheduleStore(settings.database_url)
        self.alerts = AlertStore(settings.database_url)
        self.advice_alerts = UnifiedAccountAdviceAlertProjector(
            settings.database_url,
            alerts=self.alerts,
        )
        self.health = OperationalHealthStore(settings)
        self.runtime_secrets = RuntimeSecretStore(
            settings.database_url, settings.platform_secret_key
        )
        self.simulations = SimulationStore(settings.database_url)
        self.strategies = StrategyStore(settings.database_url)
        self.strategy_feature_drift = StrategyFeatureDriftSource(
            settings.database_url,
            settings.data_root,
        )
        self.strategy_health_collector = StrategyHealthCollector(
            settings.database_url,
            data_root=settings.data_root,
            interval_seconds=settings.strategy_health_snapshot_seconds,
        )
        self.model_artifacts = ModelArtifactStore(settings.database_url)
        self.promotions = PromotionStore(settings.database_url)
        self.safe_mode = SafeModeStore(settings.database_url)
        self.autopilot = AutopilotController(settings)
        self.three_horizon_account = ThreeHorizonAccountService(settings)
        self._last_fin_strategy_schedule_reconcile_at: datetime | None = None

    def tick(self, now: datetime | None = None) -> dict[str, int]:
        current = now or datetime.now(UTC)
        fin_strategy_schedules_reconciled = (
            self._reconcile_default_fin_strategy_schedules(current)
        )
        research_asset_jobs_enqueued = self._enqueue_daily_research_assets(current)
        research_report_backfill_enqueued = self._enqueue_research_report_backfill(current)
        model_refits_enqueued = self._enqueue_due_model_refits(current)
        factor_library_materializations_enqueued = (
            self._enqueue_due_factor_library_materialization(current)
        )
        materialized = self.schedules.materialize_due(current)
        processed = 0
        while processed < 100:
            run = self.schedules.claim_run(now=current)
            if run is None:
                break
            self._process_run(run, current)
            processed += 1
        try:
            autopilot_result = self.autopilot.tick(now=current)
        except Exception as exc:  # fail one automation lane without stopping data/paper
            autopilot_result = {"cycles": 0, "branches": 0, "failed": 1}
            self.alerts.create(
                source_type="platform",
                source_id="autopilot",
                severity="critical",
                category="autopilot_tick_failed",
                title="自动驾驶研究编排失败",
                message=str(exc),
                dedupe_key=f"platform:autopilot:{current.date().isoformat()}:failed",
            )
        simulation_order_plans_enqueued = self._enqueue_due_simulation_order_plans(current)
        strategy_health_result = self._enqueue_due_strategy_health(current)
        for failure in strategy_health_result["failures"]:
            version_id = str(failure["strategy_version_id"])
            self.alerts.create(
                source_type="strategy_version",
                source_id=version_id,
                severity="warning",
                category="strategy_health_collection_failed",
                title="策略健康证据尚未就绪",
                message=str(failure["error"]),
                dedupe_key=(
                    "strategy-health-collector:"
                    f"{version_id}:{current.astimezone(ZoneInfo('Asia/Shanghai')).date()}"
                ),
            )
        auto_promotions = self._auto_promote_ready_horizons(current)
        strategies_auto_promoted = len(auto_promotions)
        try:
            three_horizon_result = self.three_horizon_account.tick(now=current)
        except Exception as exc:  # fail closed without stopping paper evidence
            three_horizon_result = {"status": "blocked", "reason": str(exc)}
            self.alerts.create(
                source_type="platform",
                source_id="three-horizon-account",
                severity="critical",
                category="three_horizon_account_blocked",
                title="三周期统一模拟账户未能推进",
                message=str(exc),
                dedupe_key=(
                    "platform:three-horizon-account:"
                    f"{current.astimezone(ZoneInfo('Asia/Shanghai')).date()}:blocked"
                ),
            )
        self._settle_activation_cutovers(three_horizon_result, now=current)
        horizon_recommendations_enqueued = self._enqueue_due_horizon_recommendations(current)
        # Transparent public recipes are controls inside the managed
        # ``fin_strategy`` competition.  The retired standalone bootstrap used
        # to create StrategyVersion/OOS/paper state here every 30 minutes,
        # which was a second automatic capital path.  Keep the counters for
        # response compatibility, but never advance that historical workflow.
        transparent_baseline_reconciles = 0
        transparent_baseline_reconcile_failures = 0
        simulation_replays_enqueued = self._enqueue_due_simulation_replays(current)
        projected = self.project_alerts()
        health_recorded = 0
        if self.health.due(current):
            snapshot = self.health.collect_and_record(current)
            health_recorded = 1
            projected += self._project_health_alerts(snapshot)
        delivered = self.alerts.deliver_pending(self._alert_webhook_url())
        return {
            "materialized": materialized,
            "processed": processed,
            "alerts_projected": projected,
            "alerts_delivered": delivered,
            "health_recorded": health_recorded,
            "strategy_health_scanned": int(strategy_health_result["scanned"]),
            "strategy_health_recorded": 0,
            "strategy_health_enqueued": int(strategy_health_result["enqueued"]),
            "strategy_health_failures": len(strategy_health_result["failures"]),
            "simulation_replays_enqueued": simulation_replays_enqueued,
            "simulation_order_plans_enqueued": simulation_order_plans_enqueued,
            "strategies_auto_promoted": strategies_auto_promoted,
            "horizon_recommendations_enqueued": horizon_recommendations_enqueued,
            "fin_strategy_schedules_reconciled": fin_strategy_schedules_reconciled,
            "transparent_baseline_reconciles": transparent_baseline_reconciles,
            "transparent_baseline_reconcile_failures": (
                transparent_baseline_reconcile_failures
            ),
            "three_horizon_account_advanced": int(
                bool(three_horizon_result.get("advanced"))
            ),
            "model_refits_enqueued": model_refits_enqueued,
            "factor_library_materializations_enqueued": (
                factor_library_materializations_enqueued
            ),
            "research_asset_jobs_enqueued": research_asset_jobs_enqueued,
            "research_report_backfill_enqueued": research_report_backfill_enqueued,
            "autopilot_cycles_checked": int(autopilot_result.get("cycles", 0)),
            "autopilot_branches_enqueued": int(autopilot_result.get("branches", 0)),
            "autopilot_failures": int(autopilot_result.get("failed", 0)),
        }

    def _reconcile_default_fin_strategy_schedules(self, now: datetime) -> int:
        """Reconcile managed research hourly without delaying other tick lanes."""

        previous = self._last_fin_strategy_schedule_reconcile_at
        if previous is not None and now - previous < timedelta(hours=1):
            return 0
        self._last_fin_strategy_schedule_reconcile_at = now
        try:
            records = reconcile_managed_fin_strategy_schedules(
                self.schedules,
                enabled=bool(self.settings.rdagent_enabled),
                now=now,
            )
        except Exception as exc:  # noqa: BLE001 - isolate the research lane
            self.alerts.create(
                source_type="platform",
                source_id="managed-fin-strategy-schedules",
                severity="warning",
                category="managed_fin_strategy_schedule_reconcile_failed",
                title="三周期策略研究调度未能完成核对",
                message=str(exc),
                dedupe_key=(
                    "platform:managed-fin-strategy-schedules:"
                    f"{now.astimezone(ZoneInfo('Asia/Shanghai')).date()}:failed"
                ),
            )
            return 0
        return len(records)

    def _auto_promote_ready_horizons(self, now: datetime) -> list[dict[str, Any]]:
        """Advance paper strategies only through their immutable horizon gates."""

        with self.jobs.engine.connect() as connection:
            version_ids = connection.scalars(
                select(strategy_versions.c.id)
                .where(
                    strategy_versions.c.status == "approved",
                    strategy_versions.c.promotion_stage == "paper",
                    strategy_versions.c.horizon_profile.in_(
                        ("short_1_5d", "swing_1_6m", "long_1_3y")
                    ),
                )
                .order_by(strategy_versions.c.approved_at)
                .limit(100)
            ).all()
        promoted: list[dict[str, Any]] = []
        for version_id in version_ids:
            try:
                result = self.promotions.auto_promote_if_ready(str(version_id))
                if result.get("promoted"):
                    promoted.append(result)
            except (KeyError, ValueError) as exc:
                self.alerts.create(
                    source_type="strategy_version",
                    source_id=str(version_id),
                    severity="warning",
                    category="auto_promotion_retry",
                    title="策略自动晋级等待重试",
                    message=str(exc),
                    dedupe_key=(
                        f"strategy:{version_id}:auto-promotion:"
                        f"{now.astimezone(ZoneInfo('Asia/Shanghai')).date()}"
                    ),
                )
        return promoted

    def _settle_activation_cutovers(
        self, account_result: dict[str, Any], *, now: datetime
    ) -> None:
        """Complete or roll back the event-backed promotion/account saga."""

        pending = self.promotions.pending_activation_cutovers()
        if not pending:
            return
        status = str(account_result.get("status") or "blocked")
        if status in {"ready", "no_action"}:
            active_versions = set(
                str(value)
                for value in dict(account_result.get("strategy_version_ids") or {}).values()
            )
            for cutover in pending:
                if cutover["strategy_version_id"] not in active_versions:
                    continue
                try:
                    self.promotions.complete_activation_cutover(
                        cutover["strategy_version_id"],
                        activation_token=cutover["activation_token"],
                        account_evidence=account_result,
                    )
                except (KeyError, ValueError) as exc:
                    self.alerts.create(
                        source_type="strategy_version",
                        source_id=cutover["strategy_version_id"],
                        severity="critical",
                        category="activation_cutover_completion_failed",
                        title="策略与统一账户切换尚未确认",
                        message=str(exc),
                        dedupe_key=(
                            f"strategy:{cutover['strategy_version_id']}:cutover-complete:"
                            f"{now.date().isoformat()}"
                        ),
                    )
            return

        # Onboarding and missing allocation/member evidence are normal
        # asynchronous pre-activation states. No unified order plan can be
        # created yet, and AdviceService continues to project the replaced
        # incumbent. All other account failures are hard cutover failures.
        if status in {
            "onboarding_required",
            "waiting_for_allocation_evidence",
            "waiting_for_member_targets",
            "waiting_for_verified_horizons",
        }:
            return
        reason = str(account_result.get("reason") or status)
        for cutover in pending:
            try:
                self.promotions.rollback_activation_cutover(
                    cutover["strategy_version_id"],
                    activation_token=cutover["activation_token"],
                    reason=f"Three-horizon account cutover failed: {reason}",
                )
            except (KeyError, ValueError) as exc:
                self.alerts.create(
                    source_type="strategy_version",
                    source_id=cutover["strategy_version_id"],
                    severity="critical",
                    category="activation_cutover_rollback_failed",
                    title="策略与统一账户切换回滚失败",
                    message=str(exc),
                    dedupe_key=(
                        f"strategy:{cutover['strategy_version_id']}:cutover-rollback:"
                        f"{now.date().isoformat()}"
                    ),
                )

    def _enqueue_due_horizon_recommendations(self, now: datetime) -> int:
        """Refresh each verified sleeve from the latest complete daily snapshot."""

        if self.safe_mode.status().get("active"):
            return 0
        local_date = now.astimezone(ZoneInfo("Asia/Shanghai")).date()
        with self.jobs.engine.connect() as connection:
            portfolio_ids = connection.scalars(
                select(recommendation_portfolios.c.id)
                .join(
                    strategy_versions,
                    strategy_versions.c.id
                    == recommendation_portfolios.c.strategy_version_id,
                )
                .where(
                    recommendation_portfolios.c.status == "active",
                    recommendation_portfolios.c.recommendation_scope
                    == "allocation_member",
                    strategy_versions.c.status == "approved",
                    strategy_versions.c.promotion_stage == "recommendation_enabled",
                    strategy_versions.c.horizon_profile.in_(
                        ("short_1_5d", "swing_1_6m", "long_1_3y")
                    ),
                )
                .order_by(recommendation_portfolios.c.created_at)
            ).all()
        enqueued = 0
        for portfolio_id in portfolio_ids:
            try:
                portfolio = self.recommendations.get(str(portfolio_id))
                version = self.strategies.get_version(
                    str(portfolio["strategy_version_id"])
                )
                dataset = select_qlib_dataset(
                    self.settings.data_root,
                    anchor_name=str(portfolio["dataset"]),
                    roll_policy=str(portfolio.get("dataset_roll_policy") or "pinned"),
                    lineage_id=portfolio.get("dataset_lineage_id"),
                    required_date=local_date,
                )
                signal_date = qlib_trading_date_on_or_before(dataset, local_date)
                dataset_identity_sha256 = str(
                    dict(dataset.get("provenance") or {}).get(
                        "dataset_identity_sha256"
                    )
                    or ""
                )
                latest = portfolio.get("latest_snapshot") or {}
                if str(latest.get("as_of_date") or "") == signal_date.isoformat():
                    continue
                self.promotions.require_recommendation_signal(
                    str(portfolio["strategy_version_id"]), signal_date=signal_date, now=now
                )
                gate = evaluate_recommendation_gate(
                    self.simulations,
                    portfolio,
                    signal_date,
                    load_calendar_days(dataset["path"]),
                )
                if not gate["passed"]:
                    continue
                model_artifact_binding, factor_materialization_binding = (
                    self._current_live_signal_bindings(
                        version,
                        dataset_identity_sha256=dataset_identity_sha256,
                        signal_date=signal_date,
                        now=now,
                    )
                )
                snapshot, created = self.recommendations.create_snapshot(
                    portfolio_id=str(portfolio_id),
                    as_of_date=signal_date,
                    dataset=str(dataset["name"]),
                    dataset_identity_sha256=dataset_identity_sha256,
                    dataset_lineage_id=dataset.get("lineage_id"),
                )
                if not created:
                    continue
                job = self.jobs.create(
                    "recommendation_refresh",
                    recommendation_refresh_job_payload(
                        snapshot,
                        dataset,
                        model_artifact_binding=model_artifact_binding,
                        factor_materialization_binding=(
                            factor_materialization_binding
                        ),
                    ),
                    self.settings.data_root
                    / "platform"
                    / "logs"
                    / f"recommendation-refresh-{snapshot['id']}.log",
                    dedupe_active_kind=False,
                    idempotency_key=recommendation_refresh_job_idempotency_key(
                        str(snapshot["id"])
                    ),
                )
                self.recommendations.attach_job(str(snapshot["id"]), str(job["id"]))
                enqueued += int(job["status"] in {"queued", "running"})
            except (KeyError, OSError, ValueError):
                # Waiting datasets/stages are retried from the same durable state.
                continue
        return enqueued

    def _current_live_signal_bindings(
        self,
        version: dict[str, Any],
        *,
        dataset_identity_sha256: str,
        signal_date: date,
        now: datetime,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Resolve immutable current-publication inference dependencies.

        Returning from this method is the enqueue gate: callers must not create
        a paper job or recommendation snapshot while either dependency is still
        materializing.  The next scheduler tick retries the same signal date.
        """

        if len(str(dataset_identity_sha256 or "")) != 64:
            raise ValueError("current signal dataset identity is invalid")
        signal_source = str(
            version.get("config", {}).get("signal_source") or "factor_score"
        )
        if signal_source == "model_prediction":
            artifact = self.model_artifacts.require_for_inference(
                str(version["id"]),
                dataset_identity_sha256=dataset_identity_sha256,
                signal_date=signal_date,
                now=now,
            )
            return (
                {
                    "id": str(artifact["id"]),
                    "artifact_sha256": str(artifact["artifact_sha256"]),
                    "checkpoint_sha256": str(artifact["checkpoint_sha256"]),
                    "dataset_identity_sha256": str(
                        artifact["dataset_identity_sha256"]
                    ),
                },
                None,
            )
        if signal_source != "factor_score":
            raise ValueError("strategy signal source is unsupported")
        return (
            None,
            self.strategy_feature_drift.current_challenger_artifact_binding(
                str(version["id"]),
                current_dataset_identity_sha256=dataset_identity_sha256,
                signal_date=signal_date,
            ),
        )

    def _enqueue_due_factor_library_materialization(self, now: datetime) -> int:
        """Materialize the governed factor library once for every sealed daily dataset.

        The job uses the complete registered date range and universe.  It is
        idempotent on dataset identity plus feature-set identity, so every
        scheduler tick is safe and a newly published Qlib version triggers the
        work without a Web button.
        """

        if self.jobs.count(
            statuses=("queued", "running"),
            kinds=("factor_library_materialize", "factor_library_cluster"),
        ):
            return 0
        try:
            free_bytes = shutil.disk_usage(self.settings.data_root).free
        except OSError:
            return 0
        if free_bytes < FACTOR_LIBRARY_MIN_FREE_BYTES:
            self.alerts.create(
                source_type="platform",
                source_id="factor-library",
                severity="warning",
                category="factor_library_disk_guard",
                title="因子库计算已等待磁盘空间",
                message="可用磁盘低于 300GB，未启动新的全因子物化。",
                dedupe_key=(
                    "platform:factor-library:disk-guard:"
                    f"{now.astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat()}"
                ),
                details={"free_bytes": free_bytes},
            )
            return 0

        candidates: list[dict[str, Any]] = []
        for dataset in list_qlib_datasets(self.settings.data_root):
            provenance = dataset.get("provenance")
            if not dataset.get("ready") or not dataset.get("reproducible"):
                continue
            if not isinstance(provenance, dict) or dataset.get("frequency") != "day":
                continue
            try:
                require_daily_qlib_contract(provenance)
                date.fromisoformat(str(dataset["start_date"]))
                date.fromisoformat(str(dataset["end_date"]))
            except (KeyError, TypeError, ValueError):
                continue
            candidates.append(dataset)
        if not candidates:
            return 0
        dataset = max(
            candidates,
            key=lambda item: (
                str(item["end_date"]),
                int(item.get("trading_days") or 0),
                str(item["name"]),
            ),
        )
        identity = str(dataset["provenance"]["dataset_identity_sha256"])
        start = str(dataset["start_date"])
        end = str(dataset["end_date"])
        desired_feature_sets = self._desired_factor_materialization_feature_sets()
        selected: tuple[dict[str, Any], str, dict[str, Any]] | None = None
        for candidate_set in desired_feature_sets:
            output = (
                self.settings.data_root
                / "artifacts"
                / "factor-library-materializations"
                / identity
                / str(candidate_set["definition_sha256"])[:16]
                / "manifest.json"
            )
            manifest: dict[str, Any] = {}
            if output.is_file():
                try:
                    manifest = json.loads(output.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
            if factor_materialization_manifest_matches(
                manifest,
                dataset_identity_sha256=identity,
                feature_set=candidate_set,
                start=start,
                end=end,
            ):
                continue
            base_key = (
                f"factor-library:{identity}:{candidate_set['definition_sha256']}:"
                f"cn_all:{start}:{end}"
            )
            retry_plan = self._factor_materialization_retry_plan(base_key)
            if retry_plan["decision"] == "enqueue":
                selected = (candidate_set, base_key, retry_plan)
                break
            if retry_plan["decision"] in {"cancelled", "exhausted"}:
                self.alerts.create(
                    source_type="platform",
                    source_id=str(candidate_set["definition_sha256"]),
                    severity=(
                        "error" if retry_plan["decision"] == "exhausted" else "warning"
                    ),
                    category="factor_library_materialization_blocked",
                    title="因子物化自动重试已停止",
                    message=(
                        "该冻结因子集已耗尽自动重试预算，需检查失败证据后人工处理。"
                        if retry_plan["decision"] == "exhausted"
                        else "该冻结因子集的最近任务已被取消，不会自动恢复。"
                    ),
                    dedupe_key=(
                        "platform:factor-library:materialization-blocked:"
                        f"{identity}:{candidate_set['definition_sha256']}:"
                        f"{retry_plan['decision']}"
                    ),
                    details={
                        "dataset_identity_sha256": identity,
                        "feature_set_id": candidate_set["id"],
                        "feature_set_definition_sha256": candidate_set[
                            "definition_sha256"
                        ],
                        "retry_contract_version": (
                            FACTOR_LIBRARY_MATERIALIZATION_RETRY_CONTRACT_VERSION
                        ),
                        "max_episodes": FACTOR_LIBRARY_MATERIALIZATION_MAX_EPISODES,
                        "attempts_per_episode": (
                            FACTOR_LIBRARY_MATERIALIZATION_ATTEMPTS_PER_EPISODE
                        ),
                        "latest_job_id": retry_plan.get("parent_job_id"),
                    },
                )
        if selected is None:
            return 0
        feature_set, idempotency_base, retry_plan = selected
        episode = int(retry_plan["episode"])
        materialization_target = {
            "contract_version": "factor-library-materialization-target-v1",
            "dataset_identity_sha256": identity,
            "feature_set_definition_sha256": feature_set["definition_sha256"],
            "universe": "cn_all",
            "start": start,
            "end": end,
        }
        payload = {
            "dataset": str(dataset["name"]),
            "dataset_path": str(dataset["path"]),
            "dataset_identity_sha256": identity,
            "feature_set_id": feature_set["id"],
            "feature_set_definition_sha256": feature_set["definition_sha256"],
            "feature_set_definition": (
                feature_set
                if str(feature_set["id"]).startswith("strategy-health:")
                else None
            ),
            "library_version_id": (
                feature_set["source"]
                if str(feature_set.get("source") or "").startswith(
                    "unified-factor-library"
                )
                else None
            ),
            "universe": "cn_all",
            "start": start,
            "end": end,
            "materialization_target_sha256": canonical_sha256(materialization_target),
            "retry_contract_version": (
                FACTOR_LIBRARY_MATERIALIZATION_RETRY_CONTRACT_VERSION
            ),
            "retry_episode": episode,
            "retry_parent_evidence": retry_plan.get("parent_evidence"),
        }
        try:
            job = self.jobs.create(
                "factor_library_materialize",
                payload,
                self.settings.data_root
                / "platform"
                / "logs"
                / (
                    f"factor-library-{identity[:12]}-"
                    f"{str(feature_set['definition_sha256'])[:12]}-e{episode}.log"
                ),
                idempotency_key=f"{idempotency_base}:episode:{episode}",
                max_attempts=FACTOR_LIBRARY_MATERIALIZATION_ATTEMPTS_PER_EPISODE,
            )
        except ValueError as exc:
            if "active factor_library_materialize job" in str(exc):
                return 0
            raise
        return int(job["status"] in {"queued", "running"})

    def _desired_factor_materialization_feature_sets(self) -> list[dict[str, Any]]:
        desired_feature_sets = [get_feature_set("unified-research-v1")]
        with self.jobs.engine.connect() as connection:
            active_version_ids = connection.scalars(
                select(strategy_versions.c.id)
                .where(
                    strategy_versions.c.status == "approved",
                    strategy_versions.c.is_legacy.is_(False),
                    strategy_versions.c.promotion_stage.in_(
                        ("paper", "recommendation_enabled")
                    ),
                )
                .order_by(strategy_versions.c.id)
            ).all()
        for version_id in active_version_ids:
            try:
                strategy_set = self.strategy_feature_drift.feature_set(str(version_id))
            except (KeyError, TypeError, ValueError):
                # The health collector reports the exact unsupported source;
                # never substitute an unrelated factor library.
                continue
            desired_feature_sets.append(
                {
                    key: strategy_set[key]
                    for key in (
                        "contract_version",
                        "id",
                        "name",
                        "features",
                        "source",
                        "materialization_contract",
                        "definition_sha256",
                    )
                }
            )
        return desired_feature_sets

    def _enqueue_due_strategy_health(self, now: datetime) -> dict[str, Any]:
        """Enqueue exact health lanes; never open Qlib artifacts in the scheduler."""

        pending = self.strategy_health_collector.pending_requests(now)
        enqueued = 0
        interval = max(
            300, int(self.settings.strategy_health_snapshot_seconds)
        )
        collection_slot = int(now.timestamp()) // interval
        for request in pending["requests"]:
            version_id = str(request["strategy_version_id"])
            batch_id = str(request["simulation_batch_id"])
            job = self.jobs.create(
                "strategy_health_collect",
                {**request, "collection_slot": collection_slot},
                self.settings.data_root
                / "platform"
                / "logs"
                / f"strategy-health-{version_id}-{batch_id}-{collection_slot}.log",
                dedupe_active_kind=False,
                idempotency_key=(
                    f"strategy-health:{version_id}:{request['promotion_stage_id']}:"
                    f"{batch_id}:{collection_slot}"
                ),
                max_attempts=2,
            )
            enqueued += int(job["status"] in {"queued", "running"})
        return {
            **pending,
            "enqueued": enqueued,
        }

    def _factor_materialization_retry_plan(self, idempotency_base: str) -> dict[str, Any]:
        episode_keys = {
            idempotency_base: 1,
            **{
                f"{idempotency_base}:episode:{episode}": episode
                for episode in range(1, FACTOR_LIBRARY_MATERIALIZATION_MAX_EPISODES + 1)
            },
        }
        with self.jobs.engine.connect() as connection:
            rows = connection.execute(
                select(
                    jobs.c.id,
                    jobs.c.idempotency_key,
                    jobs.c.status,
                    jobs.c.attempts,
                    jobs.c.max_attempts,
                    jobs.c.exit_code,
                    jobs.c.error,
                    jobs.c.created_at,
                ).where(
                    jobs.c.kind == "factor_library_materialize",
                    jobs.c.idempotency_key.in_(tuple(episode_keys)),
                )
            ).all()
        if not rows:
            return {"decision": "enqueue", "episode": 1}
        history = sorted(
            (
                (episode_keys[str(row.idempotency_key)], row)
                for row in rows
                if str(row.idempotency_key) in episode_keys
            ),
            key=lambda item: (item[0], item[1].created_at, str(item[1].id)),
        )
        episode, parent = history[-1]
        parent_evidence = {
            "job_id": str(parent.id),
            "status": str(parent.status),
            "episode": episode,
            "attempts": int(parent.attempts),
            "max_attempts": int(parent.max_attempts),
            "exit_code": parent.exit_code,
            "error_sha256": canonical_sha256({"error": str(parent.error or "")}),
        }
        parent_evidence["evidence_sha256"] = canonical_sha256(parent_evidence)
        if str(parent.status) in {"queued", "running"}:
            return {
                "decision": "wait",
                "episode": episode,
                "parent_job_id": str(parent.id),
                "parent_evidence": parent_evidence,
            }
        if str(parent.status) == "cancelled":
            return {
                "decision": "cancelled",
                "episode": episode,
                "parent_job_id": str(parent.id),
                "parent_evidence": parent_evidence,
            }
        if episode >= FACTOR_LIBRARY_MATERIALIZATION_MAX_EPISODES:
            return {
                "decision": "exhausted",
                "episode": episode,
                "parent_job_id": str(parent.id),
                "parent_evidence": parent_evidence,
            }
        return {
            "decision": "enqueue",
            "episode": episode + 1,
            "parent_job_id": str(parent.id),
            "parent_evidence": parent_evidence,
        }

    def _enqueue_due_simulation_order_plans(self, now: datetime) -> int:
        """Generate one immutable daily paper order plan per active strategy account."""

        local_date = now.astimezone(ZoneInfo("Asia/Shanghai")).date()
        datasets = {
            item["name"]: item
            for item in list_qlib_datasets(self.settings.data_root)
            if item.get("ready") and item.get("reproducible")
        }
        with self.jobs.engine.connect() as connection:
            portfolio_ids = connection.scalars(
                select(simulation_portfolios.c.id)
                .where(
                    simulation_portfolios.c.status == "active",
                    simulation_portfolios.c.source_type == "strategy_version",
                    simulation_portfolios.c.execution_adapter == "long_only",
                )
                .order_by(simulation_portfolios.c.created_at)
                .limit(100)
            ).all()
        enqueued = 0
        for portfolio_id in portfolio_ids:
            try:
                portfolio = self.simulations.get(str(portfolio_id))
                version = self.strategies.get_version(str(portfolio["source_id"]))
                anchor = datasets.get(str(portfolio["daily_dataset"]))
                if anchor is None:
                    continue
                anchor_date = qlib_trading_date_on_or_before(anchor, local_date)
                current_dataset = select_qlib_dataset(
                    self.settings.data_root,
                    anchor_name=str(portfolio["daily_dataset"]),
                    roll_policy=str(portfolio.get("daily_roll_policy") or "pinned"),
                    lineage_id=portfolio.get("daily_dataset_lineage_id"),
                    required_date=anchor_date,
                )
                signal_date = qlib_trading_date_on_or_before(current_dataset, local_date)
                dataset_identity_sha256 = str(
                    dict(current_dataset.get("provenance") or {}).get(
                        "dataset_identity_sha256"
                    )
                    or ""
                )
                if len(dataset_identity_sha256) != 64:
                    continue

                # Model predictions and active challenger factors are durable
                # dependencies for this exact daily publication.  Merely
                # enqueueing their producers earlier in the tick is not enough:
                # wait here and retry next tick before creating the order job.
                (
                    model_artifact_binding,
                    factor_materialization_binding,
                ) = self._current_live_signal_bindings(
                    version,
                    dataset_identity_sha256=dataset_identity_sha256,
                    signal_date=signal_date,
                    now=now,
                )
                stage = self.promotions.require_paper_signal(
                    str(portfolio["source_id"]),
                    portfolio_id=str(portfolio_id),
                    signal_date=signal_date,
                    now=now,
                )
                self.simulations.require_order_plan_predecessor_settled(
                    str(portfolio_id),
                    signal_date=signal_date,
                )
                job = self.jobs.create(
                    "simulation_order_plan",
                    {
                        "simulation_portfolio_id": str(portfolio_id),
                        "signal_date": signal_date.isoformat(),
                        "signal_at": None,
                        "promotion_stage_id": stage["id"],
                        "promotion_stage_opened_at": stage["opened_at"],
                        "dataset_identity_sha256": dataset_identity_sha256,
                        "model_artifact_binding": model_artifact_binding,
                        "factor_materialization_binding": (
                            factor_materialization_binding
                        ),
                        "actor": "autopilot",
                    },
                    self.settings.data_root
                    / "platform"
                    / "logs"
                    / f"simulation-order-plan-{portfolio_id}.log",
                    dedupe_active_kind=False,
                    idempotency_key=(
                        "simulation-order-plan-v2:"
                        f"{portfolio_id}:{signal_date.isoformat()}:"
                        f"{dataset_identity_sha256}"
                    ),
                )
                if job["status"] in {"queued", "running"}:
                    enqueued += 1
            except (KeyError, OSError, ValueError):
                # A new publication or stage may still be materializing.  The
                # next scheduler tick retries the exact idempotent signal day.
                continue
        return enqueued

    def _materialize_awaiting_simulation_order_plans(self, local_date: date) -> int:
        """Freeze D+1 descendants for sealed plans without regenerating D signals."""

        materialized = 0
        for job in self.jobs.awaiting_simulation_order_plans(limit=100):
            progress = dict(job.get("progress") or {})
            try:
                trade_date = date.fromisoformat(
                    str(progress[ORDER_PLAN_EXECUTION_TRADE_DATE_KEY])
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "awaiting simulation order-plan has no valid execution trade date"
                ) from exc
            if trade_date > local_date:
                continue
            payload = dict(job.get("payload") or {})
            manifest_sha256 = str(progress.get("order_plan_manifest_sha256") or "")
            try:
                batch, created = self.simulations.create_batch_from_order_plan(
                    str(payload["simulation_portfolio_id"]),
                    order_plan_manifest_sha256=manifest_sha256,
                    data_root=self.settings.data_root,
                    actor=str(payload.get("actor") or "simulation-order-plan-scheduler"),
                )
            except ExecutionDataNotReadyError:
                continue
            except SafeModeActiveError:
                # Safe mode is a platform-wide transient.  Keep the immutable
                # plan awaiting so release resumes the exact same artifact.
                continue
            except (KeyError, ValueError) as exc:
                reason = str(exc)
                if isinstance(exc, KeyError) or "portfolio is not active" in reason:
                    terminal_status = ORDER_PLAN_CANCELLED
                elif any(
                    marker in reason
                    for marker in (
                        "not bound to an active gated promotion stage",
                        "outside its genuine forward promotion period",
                    )
                ):
                    terminal_status = ORDER_PLAN_SUPERSEDED
                else:
                    terminal_status = ORDER_PLAN_FAILED
                self.jobs.terminate_simulation_order_plan_materialization(
                    str(job["id"]),
                    order_plan_manifest_sha256=manifest_sha256,
                    materialization_status=terminal_status,
                    reason=reason,
                )
                if terminal_status == ORDER_PLAN_FAILED:
                    self.alerts.create(
                        source_type="simulation_order_plan",
                        source_id=str(job["id"]),
                        severity="critical",
                        category="paper_batch_materialization_failed",
                        title="模拟盘订单计划无法物化",
                        message=(
                            "不可变订单计划无法绑定执行数据；已隔离该计划，"
                            "需恢复证据后按原冻结数据重试。"
                        ),
                        dedupe_key=f"paper-batch-materialization:{job['id']}",
                        details={
                            "job_id": str(job["id"]),
                            "portfolio_id": str(
                                payload.get("simulation_portfolio_id") or ""
                            ),
                            "order_plan_manifest_sha256": manifest_sha256,
                            "reason": reason,
                            "recovery": "retry_frozen_simulation_order_plan",
                        },
                    )
                continue
            self.jobs.complete_simulation_order_plan_materialization(
                str(job["id"]),
                order_plan_manifest_sha256=manifest_sha256,
                simulation_batch_id=str(batch["id"]),
                batch_created=created,
            )
            materialized += 1
        return materialized

    def _enqueue_research_report_backfill(self, now: datetime) -> int:
        """Resume the selected three-year PDF backlog without starving live data."""

        self.research_report_backfill.reconcile()
        if not self.settings.research_asset_auto_enabled:
            return 0
        local = now.astimezone(ZoneInfo("Asia/Shanghai"))
        if (local.hour, local.minute) < (
            self.settings.research_asset_auto_hour,
            self.settings.research_asset_auto_minute,
        ):
            return 0
        try:
            free_bytes = shutil.disk_usage(self.settings.data_root).free
        except OSError:
            # The data volume may not be mounted yet during startup.  This
            # optional backlog lane must wait without aborting health checks,
            # alert delivery, or the other durable scheduler lanes.
            return 0
        if free_bytes < 300 * 1024**3:
            return 0
        with self.jobs.engine.connect() as connection:
            active_live = connection.scalar(
                select(jobs.c.id)
                .where(
                    jobs.c.kind == "research_asset_acquire",
                    jobs.c.status.in_(("queued", "running")),
                    jobs.c.payload_json["include_tushare"].as_boolean() == True,  # noqa: E712
                    jobs.c.payload_json["tushare_report_date"].as_string().is_(None),
                )
                .limit(1)
            )
            main_data_active = connection.scalar(
                select(jobs.c.id)
                .where(
                    jobs.c.status.in_(("queued", "running")),
                    jobs.c.kind.in_(
                        (
                            "bootstrap",
                            "legacy_market_backfill",
                            "data_verify",
                            "data_snapshot",
                            "data_qlib",
                            "minute_qlib",
                            "ashare_5m_download",
                        )
                    ),
                )
                .limit(1)
            )
        if active_live is not None or main_data_active is not None:
            return 0
        research_day = local.date()
        try:
            snapshot_name = latest_verified_research_asset_snapshot(
                self.settings.data_root, as_of=research_day
            )
            dates = research_report_dates_in_snapshot(
                self.settings.data_root,
                snapshot_name=snapshot_name,
                start=date(2023, 8, 25),
                end=research_day,
                as_of=research_day,
            )
        except (OSError, ValueError):
            return 0
        self.research_report_backfill.seed(dates, snapshot_name=snapshot_name)

        day_start_local = datetime.combine(research_day, time.min, tzinfo=local.tzinfo)
        day_start = day_start_local.astimezone(UTC)
        attempts_left = max(
            0, 100 - self.research_report_backfill.attempts_started_since(day_start)
        )
        if attempts_left < 20:
            return 0
        if self.research_report_backfill.bytes_finished_since(day_start) >= 5 * 1024**3:
            return 0
        slots = max(0, 2 - self.research_report_backfill.active_count())
        enqueued = 0
        for row in self.research_report_backfill.pending(limit=min(slots, attempts_left // 20)):
            report_date = row["report_date"]
            idempotency_key = research_asset_acquisition_idempotency_key(
                research_day=research_day.isoformat(),
                snapshot_name=str(row["snapshot_name"]),
                include_tushare=True,
                include_arxiv=False,
                report_date=report_date.isoformat(),
            )
            job = self.jobs.create(
                "research_asset_acquire",
                {
                    "mode": "automatic",
                    "snapshot_name": str(row["snapshot_name"]),
                    "as_of": research_day.isoformat(),
                    "tushare_report_date": report_date.isoformat(),
                    "include_tushare": True,
                    "include_arxiv": False,
                    "requested_by": "research-report-backfill",
                },
                self.settings.data_root
                / "platform"
                / "logs"
                / f"research-report-backfill-{report_date.isoformat()}.log",
                dedupe_active_kind=False,
                idempotency_key=idempotency_key,
                max_attempts=3,
            )
            self.research_report_backfill.mark_queued(report_date, job_id=job["id"])
            enqueued += 1
        return enqueued

    def _enqueue_daily_research_assets(self, now: datetime) -> int:
        """Queue bounded, source-isolated research acquisitions after the local close."""

        if not self.settings.research_asset_auto_enabled:
            return 0
        local = now.astimezone(ZoneInfo("Asia/Shanghai"))
        if (local.hour, local.minute) < (
            self.settings.research_asset_auto_hour,
            self.settings.research_asset_auto_minute,
        ):
            return 0
        research_day = local.date()

        # The arXiv leg fed the frozen general_model scenario and is retired.
        # Only the Tushare research-report leg (fin_factor_report) remains.
        try:
            snapshot_name = latest_verified_research_asset_snapshot(
                self.settings.data_root,
                as_of=research_day,
            )
        except (OSError, ValueError):
            # Data publication is independently scheduled.  Do not bind an
            # acquisition job to a missing or unverified Tushare snapshot.
            return 0
        return self._enqueue_research_asset_source(
            research_day=research_day,
            snapshot_name=snapshot_name,
            include_tushare=True,
            include_arxiv=False,
        )

    def _enqueue_research_asset_source(
        self,
        *,
        research_day: date,
        snapshot_name: str,
        include_tushare: bool,
        include_arxiv: bool,
    ) -> int:
        """Queue exactly one source so provider failures remain independent."""

        if include_tushare == include_arxiv:
            raise ValueError("scheduled research asset jobs must enable exactly one source")
        source = "tushare" if include_tushare else "arxiv"
        idempotency_key = research_asset_acquisition_idempotency_key(
            research_day=research_day.isoformat(),
            snapshot_name=snapshot_name,
            include_tushare=include_tushare,
            include_arxiv=include_arxiv,
        )
        with self.jobs.engine.connect() as connection:
            existing = connection.execute(
                select(jobs.c.id).where(jobs.c.idempotency_key == idempotency_key)
            ).first()
        if existing is not None:
            return 0
        job = self.jobs.create(
            "research_asset_acquire",
            {
                "mode": "automatic",
                "snapshot_name": snapshot_name,
                "as_of": research_day.isoformat(),
                "include_tushare": include_tushare,
                "include_arxiv": include_arxiv,
                "requested_by": "research-asset-scheduler",
            },
            self.settings.data_root
            / "platform"
            / "logs"
            / f"research-assets-{source}-{research_day.isoformat()}.log",
            dedupe_active_kind=False,
            idempotency_key=idempotency_key,
            max_attempts=3,
        )
        return int(job["status"] in {"queued", "running"})

    def _enqueue_due_model_refits(self, now: datetime) -> int:
        """Queue daily checkpoint inference; fit only on the monthly decision day."""

        local_date = now.astimezone(ZoneInfo("Asia/Shanghai")).date()
        with self.jobs.engine.connect() as connection:
            due = connection.execute(
                select(model_artifacts)
                .where(
                    model_artifacts.c.status == "active",
                    model_artifacts.c.scheduled_refit_at.is_not(None),
                    model_artifacts.c.scheduled_refit_at <= now,
                )
                .order_by(model_artifacts.c.scheduled_refit_at)
                .limit(50)
            ).all()
        datasets = {item["name"]: item for item in list_qlib_datasets(self.settings.data_root)}
        enqueued = 0
        for row in due:
            try:
                version = self.model_artifacts.strategies.get_version(str(row.strategy_version_id))
                model_signal = version.get("model_signal")
                lineage_id = (
                    str(model_signal.get("dataset_lineage_id") or "")
                    if isinstance(model_signal, dict)
                    else ""
                )
                anchor = datasets.get(str(row.dataset))
                if version.get("status") != "approved" or anchor is None or len(lineage_id) != 64:
                    continue
                anchor_date = qlib_trading_date_on_or_before(anchor, local_date)
                dataset = select_qlib_dataset(
                    self.settings.data_root,
                    anchor_name=str(row.dataset),
                    roll_policy="latest_compatible",
                    lineage_id=lineage_id,
                    required_date=anchor_date,
                )
                signal_date = qlib_trading_date_on_or_before(dataset, local_date)
                cutoff_date = row.data_cutoff_at.astimezone(ZoneInfo("Asia/Shanghai")).date()
                if signal_date <= cutoff_date:
                    continue
                refresh_decision = model_refresh_decision(
                    signal_date=signal_date,
                    calendar_days=load_calendar_days(str(dataset["path"])),
                    dataset_name=str(dataset["name"]),
                    dataset_identity_sha256=str(
                        dataset["provenance"]["dataset_identity_sha256"]
                    ),
                )
                if refresh_decision["operation"] == "inference":
                    drift_policy = version.get("config", {}).get("model_drift_policy")
                    drift_policy_sha256 = version.get("config", {}).get(
                        "model_drift_policy_sha256"
                    )
                    if (
                        isinstance(drift_policy, dict)
                        and drift_policy.get("contract_version")
                        == "model-drift-policy-v1"
                        and canonical_sha256(drift_policy) == drift_policy_sha256
                    ):
                        drift_evidence = self._persistent_model_drift_evidence(
                            strategy_version_id=str(row.strategy_version_id),
                            signal_date=signal_date,
                            calendar_days=load_calendar_days(str(dataset["path"])),
                            dataset_lineage_id=lineage_id,
                            policy=drift_policy,
                        )
                        if drift_evidence is not None:
                            refresh_decision = {
                                "operation": "retrain",
                                "retrain_reason": "persistent_drift",
                                "retrain_evidence": drift_evidence,
                                "retrain_evidence_sha256": canonical_sha256(
                                    drift_evidence
                                ),
                            }
                job = self.jobs.create(
                    "model_refit",
                    {
                        "strategy_version_id": str(row.strategy_version_id),
                        "source_model_artifact_id": str(row.id),
                        "dataset": dataset["name"],
                        "dataset_path": dataset["path"],
                        "dataset_identity_sha256": dataset["provenance"]["dataset_identity_sha256"],
                        "dataset_lineage_id": lineage_id,
                        "signal_date": signal_date.isoformat(),
                        **refresh_decision,
                        "valid_for_days": 4,
                        "actor": "model-refit-scheduler",
                    },
                    self.settings.data_root
                    / "platform"
                    / "logs"
                    / f"model-refit-{row.strategy_version_id}-{signal_date}.log",
                    dedupe_active_kind=False,
                    idempotency_key=(
                        f"model-refit:{row.strategy_version_id}:{signal_date.isoformat()}"
                    ),
                )
                if job["status"] in {"queued", "running"}:
                    enqueued += 1
            except (KeyError, TypeError, ValueError):
                continue
        return enqueued

    def _persistent_model_drift_evidence(
        self,
        *,
        strategy_version_id: str,
        signal_date: date,
        calendar_days: set[date],
        dataset_lineage_id: str,
        policy: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Build an early-refit trigger from the active isolated paper ledger.

        Missing, stale or uncertified NAV never becomes evidence.  Manual and
        aggregate accounts are excluded so another portfolio cannot retrain the
        governed strategy.
        """

        try:
            window = int(policy["window_trading_days"])
            consecutive = int(policy["consecutive_windows"])
            required = window * consecutive
        except (KeyError, TypeError, ValueError):
            return None
        with self.jobs.engine.connect() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios.c.id)
                .where(
                    simulation_portfolios.c.source_type == "strategy_version",
                    simulation_portfolios.c.source_id == strategy_version_id,
                    simulation_portfolios.c.status == "active",
                    simulation_portfolios.c.execution_adapter == "long_only",
                    simulation_portfolios.c.created_by == "autopilot",
                )
                .order_by(simulation_portfolios.c.created_at.desc())
                .limit(1)
            ).first()
            if portfolio is None:
                return None
            rows = connection.execute(
                select(
                    simulation_nav.c.trade_date,
                    simulation_nav.c.twr_daily_return,
                    simulation_nav.c.benchmark_return,
                    simulation_nav.c.performance_certified,
                    simulation_nav.c.has_stale_prices,
                    simulation_nav.c.status,
                )
                .where(
                    simulation_nav.c.portfolio_id == str(portfolio.id),
                    simulation_nav.c.trade_date <= signal_date,
                )
                .order_by(simulation_nav.c.trade_date.desc())
                .limit(required)
            ).mappings().all()
        try:
            return build_persistent_drift_evidence(
                nav_rows=list(reversed(rows)),
                trading_days=sorted(calendar_days),
                as_of=signal_date,
                dataset_lineage_id=dataset_lineage_id,
                metric=str(policy["metric"]),
                window_trading_days=window,
                consecutive_windows=consecutive,
                threshold=float(policy["threshold"]),
                comparison=str(policy["comparison"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _enqueue_due_simulation_replays(self, now: datetime) -> int:
        """Bind due forward batches only after immutable execution data exists."""

        local_date = now.astimezone(ZoneInfo("Asia/Shanghai")).date()
        self._materialize_awaiting_simulation_order_plans(local_date)
        with self.jobs.engine.connect() as connection:
            snapshot_ids = connection.scalars(
                select(recommendation_snapshots.c.id)
                .join(
                    simulation_portfolios,
                    (simulation_portfolios.c.source_type == "recommendation")
                    & (
                        simulation_portfolios.c.source_id == recommendation_snapshots.c.portfolio_id
                    ),
                )
                .outerjoin(
                    simulation_batches,
                    (simulation_batches.c.portfolio_id == simulation_portfolios.c.id)
                    & (
                        simulation_batches.c.recommendation_snapshot_id
                        == recommendation_snapshots.c.id
                    ),
                )
                .where(
                    recommendation_snapshots.c.status == "succeeded",
                    recommendation_snapshots.c.effective_date <= local_date,
                    simulation_portfolios.c.status == "active",
                    simulation_portfolios.c.execution_adapter == "long_only",
                    simulation_batches.c.id.is_(None),
                )
                .order_by(recommendation_snapshots.c.effective_date)
                .limit(100)
            ).all()
        for snapshot_id in snapshot_ids:
            try:
                self.simulations.create_batches_for_snapshot(
                    str(snapshot_id),
                    actor="forward-simulation-scheduler",
                    data_root=self.settings.data_root,
                )
            except ValueError as exc:
                message = str(exc)
                waiting_markers = (
                    "not available yet",
                    "not ready",
                    "does not cover",
                    "no verified latest-compatible",
                    "unavailable",
                )
                if any(marker in message for marker in waiting_markers):
                    continue
                raise
        with self.jobs.engine.connect() as connection:
            due_batches = connection.scalars(
                select(simulation_batches.c.id)
                .join(
                    simulation_portfolios,
                    simulation_portfolios.c.id == simulation_batches.c.portfolio_id,
                )
                .where(
                    simulation_batches.c.status == "queued",
                    simulation_batches.c.trade_date <= local_date,
                    simulation_portfolios.c.status == "active",
                    simulation_portfolios.c.execution_adapter == "long_only",
                    simulation_batches.c.execution_adapter == "long_only",
                )
                .order_by(simulation_batches.c.trade_date, simulation_batches.c.created_at)
                .limit(100)
            ).all()
        enqueued = 0
        for batch_id in due_batches:
            job = self.jobs.create(
                "simulation_replay",
                {"simulation_batch_id": str(batch_id)},
                self.settings.data_root / "platform" / "logs" / f"simulation-replay-{batch_id}.log",
                dedupe_active_kind=False,
                idempotency_key=f"simulation-replay:{batch_id}",
            )
            if job["status"] in {"queued", "running"}:
                enqueued += 1
        return enqueued

    def _alert_webhook_url(self) -> str:
        try:
            stored = self.runtime_secrets.get("alert_webhook")
        except ValueError:
            return ""
        if stored is not None:
            return str(stored.get("webhook_url") or "")
        return self.settings.alert_webhook_url

    def _process_run(self, run: dict[str, Any], now: datetime) -> None:
        # Claim and enqueue are intentionally separate durable operations.  A
        # shared cutover lock plus a fresh ownership read prevents an already
        # claimed legacy run from creating a job after retirement has returned.
        with self.schedules.dispatch_guard(str(run["id"]), now=now) as dispatchable:
            if not dispatchable:
                return
            self._process_run_guarded(run, now)

    def _process_run_guarded(self, run: dict[str, Any], now: datetime) -> None:
        scheduled_for = datetime.fromisoformat(run["scheduled_for"])
        delay = (now - scheduled_for).total_seconds()
        payload = dict(run.get("payload") or {})
        recoverable_full_data_slot = (
            run["kind"] == "data_pipeline"
            and payload.get("profile") == "full"
            and bool(run["trading_days_only"])
        )
        if delay > int(run["misfire_grace_seconds"]):
            if recoverable_full_data_slot:
                local_date = scheduled_for.astimezone(ZoneInfo(run["timezone"])).date()
                if local_date.weekday() >= 5:
                    self.schedules.finish_run(
                        run["id"],
                        "skipped",
                        message=(
                            "superseded full market-data recovery slot skipped on local "
                            f"weekend {local_date.isoformat()}"
                        ),
                        now=now,
                    )
                    return
                if not _is_latest_due_weekday_slot(run, now):
                    self.schedules.finish_run(
                        run["id"],
                        "skipped",
                        message=(
                            "superseded by a newer due full market-data weekday slot; "
                            f"old slot {local_date.isoformat()} was not enqueued"
                        ),
                        now=now,
                    )
                    return
                # The newest due weekday publication is the bounded catch-up
                # root. Its lookback and downstream trade_cal/quality gates
                # recover observations without one pipeline per stale slot.
            else:
                message = f"schedule was late by {int(delay)} seconds and failed closed"
                self.schedules.finish_run(run["id"], "missed", message=message, now=now)
                self.alerts.create(
                    source_type="schedule_run",
                    source_id=run["id"],
                    severity="critical",
                    category="schedule_misfire",
                    title=f"调度错过执行窗口：{run['schedule_name']}",
                    message=message,
                    dedupe_key=f"schedule-run:{run['id']}:missed",
                    details={"scheduled_for": run["scheduled_for"]},
                )
                return
        try:
            if (
                run["kind"]
                in {
                    "incremental_sync",
                    "data_pipeline",
                    "auxiliary_data_pipeline",
                }
                and run["trading_days_only"]
            ):
                local_date = scheduled_for.astimezone(ZoneInfo(run["timezone"])).date()
                # Do not consult the existing Qlib calendar here: these jobs are
                # responsible for extending that calendar, so a stale calendar
                # would permanently block a weekday catch-up.  Weekends are
                # deterministic; weekday exchange holidays are closed out by the
                # downloaded trade_cal and downstream data-quality verification.
                if local_date.weekday() >= 5:
                    self.schedules.finish_run(
                        run["id"],
                        "skipped",
                        message=(
                            "scheduled data refresh skipped on local weekend "
                            f"{local_date.isoformat()}; no snapshot boundary was created"
                        ),
                        now=now,
                    )
                    return
            if run["kind"] == "incremental_sync":
                job = self._enqueue_incremental(run, scheduled_for)
            elif run["kind"] == "data_pipeline":
                job = self._enqueue_data_pipeline(run, scheduled_for)
            elif run["kind"] == "information_pipeline":
                job = self._enqueue_information_pipeline(run, scheduled_for)
                if job is None:
                    return
            elif run["kind"] == "information_factor_refresh":
                job = self._enqueue_information_factor_refresh(run, scheduled_for)
                if job is None:
                    return
            elif run["kind"] == "ashare_5m_sync":
                job = self._enqueue_ashare_5m(run, scheduled_for)
                if job is None:
                    return
            elif run["kind"] == "auxiliary_data_pipeline":
                job = self._enqueue_auxiliary_data(run, scheduled_for)
                if job is None:
                    return
            elif run["kind"] == "rdagent_research":
                job = self._enqueue_research(run, scheduled_for)
                if job is None:
                    return
            elif run["kind"] == "recommendation_refresh":
                job = self._enqueue_recommendation(run, scheduled_for)
                if job is None:
                    return
            elif run["kind"] in (
                "weekly_report",
                "monthly_decision_day",
                "preopen_check",
                "intraday_execution_check",
            ):
                job = self._enqueue_ops_task(run, scheduled_for)
                if job is None:
                    return
            else:
                raise ValueError(f"unsupported schedule kind: {run['kind']}")
        except ScheduleRunWaiting as exc:
            deadline = scheduled_for + timedelta(
                seconds=int(run["misfire_grace_seconds"])
            )
            retry_delay = max(30, min(300, self.settings.scheduler_poll_seconds * 4))
            retry_at = min(deadline, now + timedelta(seconds=retry_delay))
            self.schedules.wait_run(
                run["id"],
                message=f"waiting for dependency: {exc}",
                retry_at=retry_at,
            )
            return
        except Exception as exc:
            self.schedules.finish_run(run["id"], "failed", message=str(exc), now=now)
            self.alerts.create(
                source_type="schedule_run",
                source_id=run["id"],
                severity="critical",
                category="schedule_failure",
                title=f"自动任务创建失败：{run['schedule_name']}",
                message=str(exc),
                dedupe_key=f"schedule-run:{run['id']}:failed",
                details={"kind": run["kind"], "attempts": run["attempts"]},
            )
            return
        self.schedules.finish_run(run["id"], "enqueued", job_id=job["id"], now=now)

    def _enqueue_incremental(
        self,
        run: dict[str, Any],
        scheduled_for: datetime,
    ) -> dict[str, Any]:
        stored = self.runtime_secrets.get("tushare")
        if not stored and (not self.settings.api_url or not self.settings.token):
            raise ValueError("Tushare credentials are not configured")
        payload = run["payload"]
        local_date = scheduled_for.astimezone(ZoneInfo(run["timezone"])).date()
        lookback_days = max(1, min(30, int(payload.get("lookback_days", 7))))
        snapshot_start = str(payload.get("snapshot_start", "2008-01-01"))
        finalize = bool(payload.get("build_qlib", True))
        snapshot_name = f"cn-{snapshot_start.replace('-', '')}-{local_date:%Y%m%d}"
        log_path = self.settings.data_root / "platform" / "logs" / f"scheduled-sync-{run['id']}.log"
        return self.jobs.create(
            "bootstrap",
            {
                "profile": payload.get("profile", "full"),
                "start": (local_date - timedelta(days=lookback_days)).isoformat(),
                "end": "latest",
                "build_qlib": False,
                "incremental": True,
                "finalize_after_download": finalize,
                "pipeline_id": run["id"],
                "snapshot_start": snapshot_start,
                "snapshot_end": local_date.isoformat(),
                "snapshot_name": snapshot_name,
            },
            log_path,
            idempotency_key=f"schedule-run:{run['id']}",
        )

    def _enqueue_data_pipeline(
        self,
        run: dict[str, Any],
        scheduled_for: datetime,
    ) -> dict[str, Any]:
        stored = self.runtime_secrets.get("tushare")
        if not stored and (not self.settings.api_url or not self.settings.token):
            raise ValueError("Tushare credentials are not configured")
        payload = run["payload"]
        runtime_limits = _download_runtime_limits(payload)
        local_date = scheduled_for.astimezone(ZoneInfo(run["timezone"])).date()
        lookback_days = max(1, min(90, int(payload.get("lookback_days", 7))))
        snapshot_start = str(payload.get("snapshot_start", "2008-01-01"))
        profile = str(payload.get("profile", "full"))
        bundles = payload.get("bundles") or list(AUTOMATED_DATA_BUNDLES)
        unknown = sorted(set(bundles) - set(AUTOMATED_DATA_BUNDLES))
        if unknown:
            raise ValueError(f"unsupported automated data bundles: {unknown}")
        incremental_start = (local_date - timedelta(days=lookback_days)).isoformat()
        snapshot_prefix = "research-assets" if profile == "research-assets" else "cn"
        snapshot_name = f"{snapshot_prefix}-{snapshot_start.replace('-', '')}-{local_date:%Y%m%d}"
        if profile == "research-assets":
            if set(bundles) != {"research_corpus"}:
                raise ValueError("research-assets data pipeline requires exactly research_corpus")
            pipeline_id = f"schedule-run:{run['id']}"
            log_path = (
                self.settings.data_root
                / "platform"
                / "logs"
                / f"scheduled-research-assets-{run['id']}.log"
            )
            return self.jobs.create(
                "supplemental_research_corpus",
                {
                    "bundle": "research_corpus",
                    "start": incremental_start,
                    "end": local_date.isoformat(),
                    "snapshot_start": snapshot_start,
                    "snapshot_end": local_date.isoformat(),
                    "symbols": [],
                    "pipeline_id": pipeline_id,
                    "profile": profile,
                    "snapshot_name": snapshot_name,
                    "pipeline_steps": [
                        {"kind": "data_verify", "payload": {}},
                        {"kind": "data_snapshot", "payload": {}},
                    ],
                    "pipeline_next_index": 0,
                    **runtime_limits,
                },
                log_path,
                idempotency_key=pipeline_id,
            )
        supplemental_steps = [
            {
                "kind": f"supplemental_{bundle}",
                "payload": {
                    "bundle": bundle,
                    "start": incremental_start,
                    "end": local_date.isoformat(),
                    "symbols": [],
                },
            }
            for bundle in bundles
        ]
        publication_steps = [
            {"kind": kind, "payload": {}}
            for kind in ("data_verify", "data_snapshot", "data_qlib", "qlib_baseline")
        ]
        # Daily A-share publication is the latency-critical path. Optional
        # regional and research bundles remain durable successors, but must
        # never delay the verified snapshot or Qlib dataset used by research.
        # Their atomic unit files are picked up by the next daily snapshot.
        pipeline_steps = [*publication_steps, *supplemental_steps]
        pipeline_id = f"schedule-run:{run['id']}"
        log_path = (
            self.settings.data_root / "platform" / "logs" / f"scheduled-pipeline-{run['id']}.log"
        )
        return self.jobs.create(
            "bootstrap",
            {
                "profile": profile,
                "start": incremental_start,
                "end": "latest",
                "build_qlib": False,
                "incremental": True,
                "finalize_after_download": False,
                "pipeline_id": pipeline_id,
                "pipeline_steps": pipeline_steps,
                "pipeline_next_index": 0,
                "snapshot_start": snapshot_start,
                "snapshot_end": local_date.isoformat(),
                "snapshot_name": snapshot_name,
                **runtime_limits,
            },
            log_path,
            idempotency_key=pipeline_id,
        )

    def _enqueue_information_pipeline(
        self,
        run: dict[str, Any],
        scheduled_for: datetime,
    ) -> dict[str, Any] | None:
        payload = normalize_information_schedule_payload(run["payload"])
        active = self.jobs.count(
            statuses=("queued", "running"),
            kinds=INFORMATION_CONFLICTING_JOB_KINDS,
        )
        if active:
            self.schedules.finish_run(
                run["id"],
                "skipped",
                message=f"{active} conflicting data or information job(s) already active",
            )
            return None

        if payload["enable_nlp"]:
            # Decrypt and validate now so a recurring job fails before it creates
            # a raw-download chain that can never reach its paid NLP stages.
            from .announcement_nlp import load_llm_credentials

            load_llm_credentials(self.runtime_secrets)

        local_date = scheduled_for.astimezone(ZoneInfo(run["timezone"])).date()
        start = (local_date - timedelta(days=payload["lookback_days"])).isoformat()
        end = local_date.isoformat()
        pipeline_name = f"information-{local_date:%Y%m%d}"
        pipeline_id = f"schedule-run:{run['id']}"
        steps: list[dict[str, Any]] = []
        evaluation_dataset: dict[str, Any] | None = None
        if payload["include_factor_evaluation"]:
            evaluation_dataset = resolve_information_evaluation_dataset(
                self.settings.data_root, payload["factor_evaluation"]
            )
        if payload["enable_nlp"]:
            from .announcement_nlp import (
                FACTOR_NAME as announcement_tone_factor,
            )
            from .announcement_nlp import (
                LOGIC_FACTOR_NAME as announcement_logic_factor,
            )
            from .announcement_nlp import PROMPT_VERSION as announcement_prompt_version

            factor_names = [announcement_tone_factor, announcement_logic_factor]

            steps.append(
                {
                    "kind": "announcement_nlp",
                    "payload": {
                        "start": start,
                        "end": end,
                        "ts_codes": [],
                        "categories": payload["announcement_categories"],
                        "limit": payload["announcement_nlp_limit"],
                        "prompt_version": announcement_prompt_version,
                    },
                }
            )
            steps.append(
                {
                    "kind": "announcement_factor_register",
                    "payload": {"factor_name": "all", "actor": "information-scheduler"},
                }
            )
            if payload["include_corpus_nlp"]:
                from .corpus_nlp import (
                    DATASET_MAJOR_NEWS,
                    DEFAULT_CORPUS_DATASETS,
                    IRM_QA_DATASETS,
                    IRM_QA_FACTOR_NAME,
                    NEWS_FACTOR_NAME,
                    POLICY_DATASETS,
                    POLICY_FACTOR_NAME,
                )
                from .corpus_nlp import PROMPT_VERSION as corpus_prompt_version

                selected_corpus = set(payload["corpus_datasets"] or DEFAULT_CORPUS_DATASETS)
                if DATASET_MAJOR_NEWS in selected_corpus:
                    factor_names.append(NEWS_FACTOR_NAME)
                if selected_corpus.intersection(IRM_QA_DATASETS):
                    factor_names.append(IRM_QA_FACTOR_NAME)
                if selected_corpus.intersection(POLICY_DATASETS):
                    factor_names.append(POLICY_FACTOR_NAME)

                steps.append(
                    {
                        "kind": "corpus_nlp",
                        "payload": {
                            "start": start,
                            "end": end,
                            "datasets": payload["corpus_datasets"],
                            "ts_codes": [],
                            "limit": payload["corpus_nlp_limit"],
                            "batch_size": payload["batch_size"],
                            "major_news_per_day": payload["major_news_per_day"],
                            "irm_per_instrument_day": payload["irm_per_instrument_day"],
                            "prompt_version": corpus_prompt_version,
                        },
                    }
                )
                steps.append(
                    {
                        "kind": "corpus_factor_register",
                        "payload": {"factor_name": "all", "actor": "information-scheduler"},
                    }
                )
            if payload["include_event_labels"]:
                from .event_market_response import LABEL_SCHEMA_VERSION

                snapshot_name = payload["snapshot_name"] or latest_verified_snapshot(
                    self.settings.data_root, as_of=local_date
                )
                steps.append(
                    {
                        "kind": "event_market_response",
                        "payload": {
                            "snapshot_name": snapshot_name,
                            "horizons": payload["horizons"],
                            "benchmark_code": payload["benchmark_code"],
                            "schema_version": LABEL_SCHEMA_VERSION,
                        },
                    }
                )
            if evaluation_dataset is not None:
                evaluation = payload["factor_evaluation"]
                steps.append(
                    {
                        "kind": "information_factor_evaluate",
                        "payload": {
                            "dataset": evaluation["dataset"],
                            "dataset_path": evaluation_dataset["path"],
                            "dataset_identity_sha256": evaluation_dataset["provenance"][
                                "dataset_identity_sha256"
                            ],
                            "periods": evaluation["periods"],
                            "universe": evaluation["universe"],
                            "benchmark": evaluation["benchmark"],
                            "factor_names": sorted(set(factor_names)),
                        },
                    }
                )
                steps.append(
                    {
                        "kind": "multiface_audit",
                        "payload": {
                            "dataset": evaluation["dataset"],
                            "snapshot_name": (
                                snapshot_name if payload["include_event_labels"] else None
                            ),
                            "require_ready": True,
                        },
                    }
                )

        log_path = (
            self.settings.data_root / "platform" / "logs" / f"scheduled-information-{run['id']}.log"
        )
        return self.jobs.create(
            "cninfo_announcements_download",
            {
                "pipeline_id": pipeline_id,
                "profile": "information",
                "start": start,
                "end": end,
                "snapshot_name": pipeline_name,
                "pipeline_steps": steps,
                "pipeline_next_index": 0,
                "ts_codes": [],
                "limit": payload["download_limit"],
                "regulatory_only": payload["regulatory_only"],
            },
            log_path,
            idempotency_key=pipeline_id,
        )

    def _enqueue_information_factor_refresh(
        self,
        run: dict[str, Any],
        scheduled_for: datetime,
    ) -> dict[str, Any] | None:
        """Enqueue one weekly, full-history structured information refresh."""

        payload = normalize_information_factor_refresh_payload(run["payload"])
        local_date = scheduled_for.astimezone(ZoneInfo(run["timezone"])).date()
        if local_date.weekday() != payload["weekday"]:
            self.schedules.finish_run(
                run["id"],
                "skipped",
                message=(
                    f"information factor refresh is scheduled for weekday {payload['weekday']}"
                ),
            )
            return None
        active = self.jobs.count(
            statuses=("queued", "running"),
            kinds=INFORMATION_CONFLICTING_JOB_KINDS,
        )
        if active:
            self.schedules.finish_run(
                run["id"],
                "skipped",
                message=f"{active} conflicting data or information job(s) already active",
            )
            return None

        evaluation = payload["factor_evaluation"]
        evaluation_dataset = resolve_information_evaluation_dataset(
            self.settings.data_root, evaluation
        )
        source_snapshot_name = str(
            (evaluation_dataset.get("provenance") or {}).get("snapshot_name") or ""
        ).strip()
        if not source_snapshot_name:
            raise ValueError("information factor refresh Qlib dataset has no bound source snapshot")
        end = local_date.isoformat()
        from .announcement_nlp import FACTOR_NAME as announcement_tone_factor
        from .announcement_nlp import LOGIC_FACTOR_NAME as announcement_logic_factor
        from .corpus_nlp import CORPUS_FACTOR_NAMES
        from .event_market_response import LABEL_SCHEMA_VERSION
        from .major_news_mentions import FACTOR_NAMES as mention_factor_names
        from .news_flash_factors import FACTOR_NAMES as news_flash_factor_names
        from .report_rc_factors import FACTOR_NAMES as report_rc_factor_names

        # A new immutable Qlib publication changes the dataset identity used by
        # every information-factor evaluation.  Make this weekly refresh
        # independently complete: bind the already-produced announcement and
        # corpus artifacts, refresh/register the selected structured sources,
        # rebuild training-only event labels for the exact source snapshot, and
        # evaluate all governed information faces before the fail-closed audit.
        stages: list[dict[str, Any]] = [
            {
                "kind": "announcement_factor_register",
                "payload": {"factor_name": "all", "actor": "information-scheduler"},
            },
            {
                "kind": "corpus_factor_register",
                "payload": {"factor_name": "all", "actor": "information-scheduler"},
            },
        ]
        factor_names: list[str] = [
            announcement_tone_factor,
            announcement_logic_factor,
            *CORPUS_FACTOR_NAMES,
            *report_rc_factor_names,
            *mention_factor_names,
            *news_flash_factor_names,
        ]
        selected = set(payload["sources"])
        if "report_rc" in selected:
            stages.append(
                {
                    "kind": "report_rc_factors",
                    "payload": {
                        "start": STRUCTURED_INFORMATION_STARTS["report_rc"].isoformat(),
                        "end": end,
                        "ts_codes": [],
                    },
                }
            )
        stages.append(
            {
                "kind": "report_rc_factor_register",
                "payload": {
                    "factor_name": "all",
                    "actor": "information-scheduler",
                },
            }
        )
        if "major_news_mentions" in selected:
            stages.append(
                {
                    "kind": "major_news_mentions",
                    "payload": {
                        "start": STRUCTURED_INFORMATION_STARTS["major_news_mentions"].isoformat(),
                        "end": end,
                        "ts_codes": [],
                    },
                }
            )
        stages.append(
            {
                "kind": "major_news_mentions_factor_register",
                "payload": {
                    "factor_name": "all",
                    "actor": "information-scheduler",
                },
            }
        )
        if "news_flash" in selected:
            stages.append(
                {
                    "kind": "news_flash_factors",
                    "payload": {
                        "start": STRUCTURED_INFORMATION_STARTS["news_flash"].isoformat(),
                        "end": end,
                    },
                }
            )
        stages.append(
            {
                "kind": "news_flash_factor_register",
                "payload": {"actor": "information-scheduler"},
            }
        )
        stages.append(
            {
                "kind": "event_market_response",
                "payload": {
                    "snapshot_name": source_snapshot_name,
                    "horizons": [1, 3, 5, 20],
                    "benchmark_code": "000300.SH",
                    "schema_version": LABEL_SCHEMA_VERSION,
                },
            }
        )
        stages.append(
            {
                "kind": "information_factor_evaluate",
                "payload": {
                    "dataset": evaluation["dataset"],
                    "dataset_path": evaluation_dataset["path"],
                    "dataset_identity_sha256": evaluation_dataset["provenance"][
                        "dataset_identity_sha256"
                    ],
                    "periods": evaluation["periods"],
                    "universe": evaluation["universe"],
                    "benchmark": evaluation["benchmark"],
                    "factor_names": sorted(set(factor_names)),
                },
            }
        )
        stages.append(
            {
                "kind": "multiface_audit",
                "payload": {
                    "dataset": evaluation["dataset"],
                    "snapshot_name": source_snapshot_name,
                    "require_ready": True,
                },
            }
        )
        first, *remaining = stages
        pipeline_name = f"information-factors-{local_date:%Y%m%d}"
        pipeline_id = f"schedule-run:{run['id']}"
        first_payload = {
            "pipeline_id": pipeline_id,
            "profile": "information_factor_refresh",
            "start": min(STRUCTURED_INFORMATION_STARTS[source] for source in selected).isoformat(),
            "end": end,
            "snapshot_name": pipeline_name,
            "pipeline_steps": remaining,
            "pipeline_next_index": 0,
            **first["payload"],
        }
        log_path = (
            self.settings.data_root
            / "platform"
            / "logs"
            / f"scheduled-information-factors-{run['id']}.log"
        )
        return self.jobs.create(
            first["kind"],
            first_payload,
            log_path,
            idempotency_key=pipeline_id,
        )

    def _enqueue_ashare_5m(
        self,
        run: dict[str, Any],
        scheduled_for: datetime,
    ) -> dict[str, Any] | None:
        stored = self.runtime_secrets.get("tushare")
        if not stored and (not self.settings.api_url or not self.settings.token):
            raise ValueError("Tushare credentials are not configured")
        payload = run["payload"]
        runtime_limits = _download_runtime_limits(payload)
        local_scheduled_for = scheduled_for.astimezone(ZoneInfo(run["timezone"]))
        local_date = local_scheduled_for.date()
        if local_scheduled_for.time().replace(tzinfo=None) < time(15, 10):
            raise ValueError("A-share five-minute sync must run after the market has fully closed")

        # The raw SSE calendar extends beyond the latest published Qlib
        # dataset and therefore distinguishes an exchange holiday from a stale
        # daily publication.  Never infer a closure merely because today's
        # daily Qlib output is missing.
        open_days = set(load_trade_calendar_open_days(self.settings.data_root))
        calendar_horizon = max(open_days)
        if local_date > calendar_horizon:
            raise ValueError(
                "persisted SSE trade calendar does not cover the five-minute sync date"
            )
        if local_date not in open_days:
            self.schedules.finish_run(
                run["id"],
                "skipped",
                message=(
                    "A-share five-minute sync skipped on persisted SSE non-trading day "
                    f"{local_date.isoformat()}"
                ),
            )
            return None

        # Because schedules are constrained to post-close slots, an open local
        # date is also the most recent fully closed trading day.  Bind every
        # downstream artifact to that exact day; a stale daily Qlib publication
        # is an error, not a reason to silently reuse the prior session.
        target_date = local_date
        history_start = date.fromisoformat(str(payload.get("history_start") or "2024-01-01"))
        if history_start > target_date:
            raise ValueError("A-share five-minute history start is after the run date")
        daily_datasets = [
            item
            for item in list_qlib_datasets(self.settings.data_root)
            if item.get("ready")
            and item.get("reproducible")
            and item.get("frequency") == "day"
            and str(item.get("start_date") or "") <= history_start.isoformat()
            and str(item.get("end_date") or "") >= target_date.isoformat()
        ]
        requested_daily = str(payload.get("daily_dataset") or "")
        if requested_daily:
            daily_datasets = [item for item in daily_datasets if item["name"] == requested_daily]
        if not daily_datasets:
            raise ValueError(
                "a reproducible daily Qlib dataset is required before five-minute sync"
            )
        daily_dataset = max(
            daily_datasets,
            key=lambda item: (str(item.get("end_date") or ""), str(item["name"])),
        )
        require_daily_qlib_contract(daily_dataset.get("provenance") or {})
        if qlib_trading_date_on_or_before(daily_dataset, target_date) != target_date:
            raise ValueError("daily Qlib publication does not contain the fully closed trading day")
        source_lineage_id = str(
            (daily_dataset.get("provenance") or {}).get("source_lineage_id") or ""
        )
        if len(source_lineage_id) != 64:
            raise ValueError("daily Qlib dataset has no verified source lineage")
        snapshot_name = f"ashare-5m-incremental-{target_date:%Y%m%d}"
        output_name = f"{snapshot_name}-5min"
        log_path = (
            self.settings.data_root / "platform" / "logs" / f"scheduled-ashare-5m-{run['id']}.log"
        )
        return self.jobs.create(
            "ashare_5m_download",
            {
                "start": history_start.isoformat(),
                "end": target_date.isoformat(),
                "snapshot_name": snapshot_name,
                "source_lineage_id": source_lineage_id,
                "daily_dataset": daily_dataset["name"],
                "pipeline_steps": [
                    {
                        "kind": "minute_qlib",
                        "payload": {
                            "output_name": output_name,
                            "target_frequency": "5min",
                        },
                    }
                ],
                "pipeline_next_index": 0,
                "pipeline_id": f"schedule-run:{run['id']}",
                "profile": "ashare_intraday",
                **runtime_limits,
            },
            log_path,
            idempotency_key=f"schedule-run:{run['id']}",
        )

    def _enqueue_auxiliary_data(
        self,
        run: dict[str, Any],
        scheduled_for: datetime,
    ) -> dict[str, Any] | None:
        """Publish the five catalog capabilities outside the daily raw chain."""

        stored = self.runtime_secrets.get("tushare")
        if not stored and (not self.settings.api_url or not self.settings.token):
            raise ValueError("Tushare credentials are not configured")
        payload = run["payload"]
        runtime_limits = _download_runtime_limits(payload)
        candidates = [
            item
            for item in list_qlib_datasets(self.settings.data_root)
            if item.get("ready")
            and item.get("reproducible")
            and item.get("frequency") == "day"
            and item.get("end_date")
        ]
        if not candidates:
            raise ValueError("auxiliary data refresh requires a reproducible daily Qlib dataset")
        daily_dataset = max(
            candidates,
            key=lambda item: (str(item["end_date"]), str(item["name"])),
        )
        require_daily_qlib_contract(daily_dataset.get("provenance") or {})
        target_date = date.fromisoformat(str(daily_dataset["end_date"]))
        history_start = date.fromisoformat(str(payload.get("history_start") or "2024-01-01"))
        if history_start > target_date:
            raise ValueError("auxiliary data history start is after the latest daily dataset")
        source_lineage_id = str(
            (daily_dataset.get("provenance") or {}).get("source_lineage_id") or ""
        )
        if len(source_lineage_id) != 64:
            raise ValueError("daily Qlib dataset has no verified source lineage")
        symbols = payload.get("strategy_minute_symbols")
        if not isinstance(symbols, list) or not symbols:
            raise ValueError("auxiliary data refresh requires strategy minute symbols")
        max_stocks = int(payload.get("max_stocks", 100))
        max_options = int(payload.get("max_options", 100))
        snapshot_name = f"execution-{history_start:%Y%m%d}-{target_date:%Y%m%d}-auto"
        pipeline_id = f"auxiliary-data:{target_date.isoformat()}"
        root_payload = {
            "start": history_start.isoformat(),
            "end": target_date.isoformat(),
            "snapshot_start": history_start.isoformat(),
            "snapshot_end": target_date.isoformat(),
            "snapshot_name": snapshot_name,
            "pipeline_snapshot_name": snapshot_name,
            "pipeline_id": pipeline_id,
            "profile": "auxiliary_daily",
            **runtime_limits,
            "pipeline_steps": [
                {
                    "kind": "core_intraday_download",
                    "payload": {
                        "start": history_start.isoformat(),
                        "end": target_date.isoformat(),
                        "snapshot_name": snapshot_name,
                        "daily_dataset": daily_dataset["name"],
                        "source_lineage_id": source_lineage_id,
                        "etfs": ["510300.SH", "159919.SZ"],
                        "stocks": [],
                        "indices": [],
                        "futures": [],
                        "options": [],
                        "auto_select": True,
                        "max_stocks": max_stocks,
                        "max_options": max_options,
                        "etf_categories": ["broad", "industry", "gold", "bond"],
                    },
                },
                {
                    "kind": "minute_qlib",
                    "payload": {
                        "output_name": f"{snapshot_name}-1min",
                        "target_frequency": "1min",
                    },
                },
                {
                    "kind": "supplemental_strategy_specialty_minutes",
                    "payload": {
                        "bundle": "strategy_specialty_minutes",
                        "start": history_start.isoformat(),
                        "end": target_date.isoformat(),
                        "symbols": sorted({str(item).upper() for item in symbols}),
                    },
                },
            ],
            "pipeline_next_index": 0,
        }
        log_path = (
            self.settings.data_root
            / "platform"
            / "logs"
            / f"scheduled-auxiliary-data-{target_date:%Y%m%d}.log"
        )
        return self.jobs.create(
            "margin_eligibility_download",
            root_payload,
            log_path,
            idempotency_key=pipeline_id,
        )

    def _managed_fin_strategy_incumbent(
        self, horizon_profile: str
    ) -> dict[str, Any] | None:
        with self.jobs.engine.connect() as connection:
            rows = connection.execute(
                select(
                    strategy_versions.c.id,
                    strategy_versions.c.status,
                    strategy_versions.c.promotion_stage,
                    strategy_versions.c.horizon_profile,
                    strategy_versions.c.horizon_contract_sha256,
                    strategy_versions.c.approved_at,
                    strategy_versions.c.created_at,
                    strategy_versions.c.version,
                ).where(
                    strategy_versions.c.status == "approved",
                    strategy_versions.c.horizon_profile == horizon_profile,
                    strategy_versions.c.promotion_stage.in_(
                        ("paper", "recommendation_enabled")
                    ),
                )
            ).all()
        if not rows:
            return None
        row = max(
            rows,
            key=lambda item: (
                int(str(item.promotion_stage) == "recommendation_enabled"),
                item.approved_at or item.created_at,
                int(item.version),
                str(item.id),
            ),
        )
        return {
            "id": str(row.id),
            "status": str(row.status),
            "promotion_stage": str(row.promotion_stage),
            "horizon_profile": str(row.horizon_profile),
            "horizon_contract_sha256": str(row.horizon_contract_sha256),
            "version": int(row.version),
        }

    def _managed_fin_strategy_drift_event(
        self,
        incumbent_strategy: dict[str, Any],
        *,
        observed_at: datetime,
    ) -> dict[str, Any]:
        """Resolve the exact incumbent's latest governed drift episode."""

        with self.jobs.engine.connect() as connection:
            rows = connection.execute(
                select(strategy_health_snapshots)
                .where(
                    strategy_health_snapshots.c.strategy_version_id
                    == incumbent_strategy["id"],
                    strategy_health_snapshots.c.horizon_profile
                    == incumbent_strategy["horizon_profile"],
                    strategy_health_snapshots.c.as_of <= observed_at,
                    strategy_health_snapshots.c.recorded_at <= observed_at,
                )
                .order_by(
                    strategy_health_snapshots.c.as_of.desc(),
                    strategy_health_snapshots.c.recorded_at.desc(),
                    strategy_health_snapshots.c.snapshot_sha256.desc(),
                )
            )
            return resolve_feature_drift_episode(
                (row_dict(item) for item in rows),
                expected_strategy_version_id=str(incumbent_strategy["id"]),
                expected_horizon_profile=str(incumbent_strategy["horizon_profile"]),
                expected_horizon_contract_sha256=str(
                    incumbent_strategy["horizon_contract_sha256"]
                ),
                observed_at=observed_at,
            )

    def _consumed_managed_fin_strategy_trigger_ids(
        self, trigger_ids: list[str]
    ) -> set[str]:
        """Return trigger ids already attached to any immutable research run."""

        consumed: set[str] = set()
        with self.jobs.engine.connect() as connection:
            trigger_path = research_runs.c.config_json["managed_fin_strategy_run"][
                "trigger_ids"
            ]
            for trigger_id in sorted(set(trigger_ids)):
                existing = connection.scalar(
                    select(research_runs.c.id)
                    .where(
                        research_runs.c.kind == "strategy",
                        trigger_path.op("@>")(cast([trigger_id], JSONB)),
                    )
                    .limit(1)
                )
                if existing is not None:
                    consumed.add(trigger_id)
        return consumed

    def _existing_managed_fin_strategy_run(
        self, run_sha256: str
    ) -> dict[str, Any] | None:
        with self.jobs.engine.connect() as connection:
            rows = connection.execute(
                select(
                    research_runs.c.id,
                    research_runs.c.job_id,
                    research_runs.c.status,
                    research_runs.c.config_json,
                )
                .where(
                    research_runs.c.kind == "strategy",
                    research_runs.c.config_json["managed_fin_strategy_run"][
                        "run_sha256"
                    ].as_string()
                    == run_sha256,
                )
            ).all()
        if len(rows) > 1:
            raise ValueError("managed fin_strategy run identity is not unique")
        if not rows:
            return None
        item = rows[0]
        return {
            "id": str(item.id),
            "job_id": str(item.job_id) if item.job_id else None,
            "status": str(item.status),
        }

    def _enqueue_research(
        self,
        run: dict[str, Any],
        scheduled_for: datetime,
    ) -> dict[str, Any] | None:
        managed = validate_managed_fin_strategy_payload(run["payload"])
        payload = normalize_research_schedule_payload(
            run["payload"],
            max_loops=self.settings.rdagent_max_loops,
            max_duration=self.settings.rdagent_max_duration,
        )
        scenario = get_rdagent_scenario(payload["scenario"])
        if scenario.id in FROZEN_RDAGENT_SCENARIOS:
            # Frozen scenarios keep historical schedules readable, but the
            # dispatcher must never enqueue new work for them.
            self.schedules.finish_run(
                run["id"],
                "skipped",
                message=(
                    f"RD-Agent scenario {scenario.id} is frozen; "
                    "historical runs are read-only"
                ),
            )
            return None
        dataset: dict[str, Any] | None = None
        periods: dict[str, str] | None = None
        period_resolution: dict[str, Any] | None = None
        calendar: list[str] = []
        local_date = scheduled_for.astimezone(ZoneInfo(run["timezone"])).date()
        cadence_event: dict[str, Any] | None = None
        managed_trigger_events: list[dict[str, Any]] = []
        if scenario.requires_dataset:
            available = list_qlib_datasets(self.settings.data_root)
            if managed is not None:
                try:
                    dataset = select_latest_reproducible_daily_dataset(available)
                except ValueError as exc:
                    raise ScheduleRunWaiting(str(exc)) from exc
            else:
                datasets = {item["name"]: item for item in available}
                dataset = datasets.get(payload["dataset"])
            if not dataset or not dataset["ready"] or not dataset.get("reproducible"):
                raise ScheduleRunWaiting(
                    "scheduled RD-Agent research Qlib dataset is not yet reproducible"
                )
            try:
                calendar = (
                    (Path(dataset["path"]) / "calendars" / "day.txt")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                if scenario.id in HORIZON_RESEARCH_SCENARIOS:
                    periods, period_resolution = resolve_research_window_contract(
                        dataset,
                        calendar,
                        periods=payload.get("periods"),
                        period_policy=payload.get("period_policy"),
                        horizon_profile=payload.get("horizon_profile"),
                        feature_set=payload.get("feature_set"),
                    )
                else:
                    # Report-derived factors retain their historical 1-session
                    # research contract.  They do not acquire an investment
                    # horizon merely because their source PDF uses daily data.
                    periods, period_resolution = resolve_research_periods(
                        calendar,
                        periods=payload.get("periods"),
                        period_policy=payload.get("period_policy"),
                    )
            except (FileNotFoundError, OSError) as exc:
                raise ScheduleRunWaiting(
                    "scheduled RD-Agent dataset calendar is still publishing"
                ) from exc
            except ValueError as exc:
                transient_markers = (
                    "field-year coverage",
                    "normalized staging evidence",
                    "coverage ends before the data cutoff",
                    "trading calendar is empty",
                )
                if any(marker in str(exc) for marker in transient_markers):
                    raise ScheduleRunWaiting(str(exc)) from exc
                raise
            provenance = dict(dataset.get("provenance") or {})
            dataset_identity = str(provenance.get("dataset_identity_sha256") or "")
            dataset_lineage = str(
                dataset.get("lineage_id")
                or provenance.get("dataset_lineage_id")
                or ""
            )
            period_resolution["dataset_identity_sha256"] = dataset_identity
            period_resolution["dataset_lineage_id"] = dataset_lineage
            if dataset.get("start_date") and periods["train_start"] < dataset["start_date"]:
                raise ValueError("scheduled RD-Agent training window starts before the dataset")
            if dataset.get("end_date") and periods["test_end"] > dataset["end_date"]:
                raise ValueError("scheduled RD-Agent test window ends after the dataset")
            if managed is not None:
                try:
                    exchange_days = load_trade_calendar_open_days(
                        self.settings.data_root
                    )
                    cadence_event = managed_fin_strategy_due_event(
                        managed,
                        scheduled_date=local_date,
                        trading_days=sorted(exchange_days),
                    )
                except (FileNotFoundError, ValueError) as exc:
                    raise ScheduleRunWaiting(str(exc)) from exc
                if cadence_event["reason"] == "exchange_closed":
                    self.schedules.finish_run(
                        run["id"],
                        "skipped",
                        message=str(cadence_event["reason"]),
                    )
                    return None
            elif run["trading_days_only"] and local_date.isoformat() not in set(calendar):
                self.schedules.finish_run(
                    run["id"], "skipped", message="not a Qlib trading day"
                )
                return None
        elif run["trading_days_only"]:
            raise ValueError("scheduled RD-Agent lab scenarios must disable trading_days_only")

        incumbent_strategy: dict[str, Any] | None = None
        if managed is not None:
            incumbent_strategy = self._managed_fin_strategy_incumbent(
                str(managed["horizon_profile"])
            )
            drift_event: dict[str, Any] | None = None
            if incumbent_strategy is None and not cadence_event["due"]:
                # Cold start is still a governed research comparison: the
                # transparent public recipe frozen by the managed schedule is
                # the control, while the proposal has no production parent.
                # Requiring a paper/recommendation incumbent here creates a
                # circular dependency after every baseline is honestly
                # rejected: no strategy can be researched until one already
                # passed the very gates the research is meant to reach.
                self.schedules.finish_run(
                    run["id"],
                    "skipped",
                    message=(
                        f"{cadence_event['reason']}; no incumbent exists for "
                        "drift evidence"
                    ),
                )
                return None
            if incumbent_strategy is not None:
                drift_event = self._managed_fin_strategy_drift_event(
                    incumbent_strategy,
                    observed_at=scheduled_for,
                )
            if cadence_event["due"]:
                calendar_identity = {
                    "contract_version": "managed-fin-strategy-trigger-id-v1",
                    "kind": "calendar",
                    "schedule_contract_sha256": managed["contract_sha256"],
                    "calendar_event": cadence_event["event"],
                }
                managed_trigger_events.append(
                    {
                        **calendar_identity,
                        "trigger_id": fin_strategy_schedule_sha256(calendar_identity),
                    }
                )
            if drift_event is not None and drift_event["due"]:
                managed_trigger_events.append(dict(drift_event["event"]))
            if not managed_trigger_events:
                self.schedules.finish_run(
                    run["id"],
                    "skipped",
                    message=(
                        f"{cadence_event['reason']}; "
                        + (
                            str(drift_event["reason"])
                            if drift_event is not None
                            else "no incumbent drift evidence"
                        )
                    ),
                )
                return None
            consumed_trigger_ids = self._consumed_managed_fin_strategy_trigger_ids(
                [str(item["trigger_id"]) for item in managed_trigger_events]
            )
            managed_trigger_events = sorted(
                (
                    item
                    for item in managed_trigger_events
                    if str(item["trigger_id"]) not in consumed_trigger_ids
                ),
                key=lambda item: str(item["trigger_id"]),
            )
            if not managed_trigger_events:
                self.schedules.finish_run(
                    run["id"],
                    "skipped",
                    message="managed fin_strategy trigger events are already consumed",
                )
                return None
            if local_date.isoformat() not in set(calendar):
                raise ScheduleRunWaiting(
                    "latest reproducible daily Qlib has not published the schedule session"
                )

        strategy_research_signal_binding: dict[str, Any] | None = None
        if scenario.id == "fin_strategy":
            if dataset is None or not isinstance(payload.get("feature_set"), dict):
                raise ValueError("scheduled fin_strategy governed inputs are incomplete")
            try:
                champion_selection = (
                    self.autopilot.champion_selector.select_champion(
                        dataset=str(dataset["name"]),
                        dataset_identity_sha256=str(
                            dataset["provenance"]["dataset_identity_sha256"]
                        ),
                        horizon_profile=str(payload["horizon_profile"]),
                    )
                )
            except ValueError as exc:
                if str(exc) != (
                    "no independently admitted signal matches this dataset identity"
                    ):
                    raise
                champion_selection = None
            payload["feature_set"] = research_feature_set_for_champion_selection(
                dict(payload["feature_set"]),
                champion_selection,
            )
            periods, period_resolution = resolve_research_window_contract(
                dataset,
                calendar,
                periods=payload.get("periods"),
                period_policy=payload.get("period_policy"),
                horizon_profile=payload.get("horizon_profile"),
                feature_set=payload["feature_set"],
            )
            strategy_research_signal_binding = build_strategy_research_signal_binding(
                horizon_profile=str(payload["horizon_profile"]),
                dataset=str(dataset["name"]),
                dataset_identity_sha256=str(
                    dataset["provenance"]["dataset_identity_sha256"]
                ),
                research_feature_set=dict(payload["feature_set"]),
                champion_selection=champion_selection,
            )

        try:
            runtime = probe_rdagent(
                self.settings, Path(__file__).resolve().parents[2]
            )
            require_ready_scenario(runtime, self.settings, scenario.id)
        except (OSError, RuntimeError, ValueError) as exc:
            if managed is not None:
                raise ScheduleRunWaiting(
                    f"managed fin_strategy runtime is not ready: {exc}"
                ) from exc
            raise
        expected_runtime_identity = expected_rdagent_runtime_identity(runtime, scenario.id)
        auto_selected_assets = not payload["asset_ids"] and scenario.auto_select_assets
        assets = resolve_rdagent_assets(
            self.settings,
            scenario,
            payload["asset_ids"],
            excluded_auto_asset_ids=(
                self.research_assets.unavailable_asset_ids()
                if auto_selected_assets
                else frozenset()
            ),
            pre_final_end=(
                date.fromisoformat(periods["valid_end"]) if periods is not None else None
            ),
            selection_limit=(payload["loop_n"] if scenario.id == "fin_factor_report" else None),
        )
        resolved_asset_ids = list(assets["manifest_sha256"])
        if auto_selected_assets:
            for asset_id in resolved_asset_ids:
                self.rdagent_candidates.import_manifest(
                    self.settings.data_root
                    / "artifacts"
                    / "research-assets"
                    / asset_id
                    / "manifest.json",
                    actor=payload["requested_by"],
                )
        artifact_root = self.settings.data_root / "artifacts" / "rdagent"
        dataset_name = str(dataset["name"]) if dataset is not None else f"lab:{scenario.id}"
        dataset_binding = (
            {
                "contract_version": "scheduled-research-dataset-binding-v1",
                "name": dataset_name,
                "identity_sha256": str(
                    dict(dataset.get("provenance") or {}).get(
                        "dataset_identity_sha256"
                    )
                    or ""
                ),
                "lineage_id": str(
                    dataset.get("lineage_id")
                    or dict(dataset.get("provenance") or {}).get(
                        "dataset_lineage_id"
                    )
                    or ""
                ),
                "start_date": str(dataset.get("start_date") or ""),
                "end_date": str(dataset.get("end_date") or ""),
            }
            if dataset is not None
            else None
        )
        managed_run: dict[str, Any] | None = None
        if managed is not None:
            trigger_ids = [
                str(item["trigger_id"]) for item in managed_trigger_events
            ]
            managed_run = {
                "contract_version": "managed-fin-strategy-run-v3",
                "schedule_contract_sha256": managed["contract_sha256"],
                "calendar_event": cadence_event["event"],
                "trigger_events": managed_trigger_events,
                "trigger_ids": trigger_ids,
                "scheduled_session": local_date.isoformat(),
                "dataset_binding": dataset_binding,
                "research_window_contract_sha256": period_resolution[
                    "research_window_contract_sha256"
                ],
                "feature_set_definition_sha256": payload["feature_set"][
                    "definition_sha256"
                ],
                "strategy_research_signal_binding_sha256": (
                    strategy_research_signal_binding["binding_sha256"]
                ),
                "incumbent_strategy_version_id": (
                    incumbent_strategy["id"]
                    if incumbent_strategy is not None
                    else None
                ),
                "research_control": (
                    {
                        "mode": "approved_strategy_incumbent",
                        "strategy_version_id": incumbent_strategy["id"],
                    }
                    if incumbent_strategy is not None
                    else {
                        "mode": "transparent_public_baseline",
                        "recipe_id": managed["recipe_id"],
                        "recipe_version": managed["recipe_version"],
                        "recipe_sha256": managed["recipe_sha256"],
                    }
                ),
                "runtime_identity_sha256": fin_strategy_schedule_sha256(
                    expected_runtime_identity
                ),
            }
            managed_run["run_sha256"] = fin_strategy_schedule_sha256(managed_run)
        config: dict[str, Any] = {
            "scenario": scenario.id,
            "asset_ids": resolved_asset_ids,
            "asset_manifest_sha256": assets["manifest_sha256"],
            "asset_selection_mode": "automatic" if auto_selected_assets else "explicit",
            "feature_set": payload["feature_set"],
            "horizon_profile": payload["horizon_profile"],
            "expected_rdagent_runtime": expected_runtime_identity,
            "dataset_binding": dataset_binding,
            "incumbent_strategy": incumbent_strategy,
            "managed_fin_strategy_run": managed_run,
            "strategy_research_signal_binding": strategy_research_signal_binding,
            **(
                {
                    "primary_label_policy": primary_label_policy_contract(),
                    "primary_label_policy_sha256": payload[
                        "primary_label_policy_sha256"
                    ],
                }
                if scenario.id in HORIZON_RESEARCH_SCENARIOS
                else {}
            ),
        }
        if dataset is not None and periods is not None and period_resolution is not None:
            config.update(
                {
                    "periods": periods,
                    "evaluation_profiles": period_resolution["evaluation_profiles"],
                    "period_resolution": period_resolution,
                    "research_window_contract": period_resolution[
                        "research_window_contract"
                    ],
                    "research_window_contract_sha256": period_resolution[
                        "research_window_contract_sha256"
                    ],
                    "dataset_path": dataset["path"],
                }
            )
        if managed_run is not None:
            existing = self._existing_managed_fin_strategy_run(
                str(managed_run["run_sha256"])
            )
            if existing is not None:
                if existing["job_id"]:
                    return self.jobs.get(str(existing["job_id"]))
                if existing["status"] in {"queued", "running", "evaluating"}:
                    raise ScheduleRunWaiting(
                        "the exact managed fin_strategy run is still attaching its job"
                    )
                self.schedules.finish_run(
                    run["id"],
                    "skipped",
                    message=(
                        "the exact managed fin_strategy event is already terminal: "
                        f"{existing['status']}"
                    ),
                )
                return None
        try:
            research_run = self.research.create_run(
                kind=scenario.research_kind,
                objective=payload["objective"],
                dataset=dataset_name,
                requested_by=payload["requested_by"],
                budget={"loop_n": payload["loop_n"], "duration": payload["duration"]},
                config=config,
                artifact_path=artifact_root,
            )
        except ValueError as exc:
            if f"active {scenario.research_kind} research run" not in str(exc):
                raise
            if managed is not None:
                raise ScheduleRunWaiting(
                    f"a bounded {scenario.id} research run is already active"
                ) from exc
            self.schedules.finish_run(
                run["id"],
                "skipped",
                message=f"a bounded {scenario.id} research run is already active",
            )
            return None
        if auto_selected_assets:
            try:
                self.research_assets.reserve_automatic(
                    research_run_id=research_run["id"],
                    scenario=scenario.id,
                    asset_manifest_sha256=assets["manifest_sha256"],
                    actor=payload["requested_by"],
                )
            except ValueError as exc:
                self.research.mark_run(
                    research_run["id"],
                    "failed",
                    actor="scheduler",
                    error=str(exc),
                )
                raise
        log_path = (
            self.settings.data_root
            / "platform"
            / "logs"
            / f"rdagent-{scenario.id}-{research_run['id']}.log"
        )
        try:
            job = self.jobs.create(
                scenario.job_kind,
                {
                    "scenario": scenario.id,
                    "research_run_id": research_run["id"],
                    "dataset": dataset_name if dataset is not None else None,
                    "dataset_path": dataset["path"] if dataset else None,
                    "dataset_identity_sha256": (
                        dataset_binding["identity_sha256"] if dataset_binding else None
                    ),
                    "dataset_lineage_id": (
                        dataset_binding["lineage_id"] if dataset_binding else None
                    ),
                    "dataset_binding": dataset_binding,
                    "objective": payload["objective"],
                    "loop_n": payload["loop_n"],
                    "duration": payload["duration"],
                    "periods": periods,
                    "evaluation_profiles": (
                        period_resolution["evaluation_profiles"] if period_resolution else []
                    ),
                    "period_resolution": period_resolution,
                    "research_window_contract": (
                        period_resolution["research_window_contract"]
                        if period_resolution
                        else None
                    ),
                    "research_window_contract_sha256": (
                        period_resolution["research_window_contract_sha256"]
                        if period_resolution
                        else None
                    ),
                    "label_horizon_sessions": (
                        primary_label_horizon_sessions(payload["horizon_profile"])
                        if period_resolution
                        and scenario.id in HORIZON_RESEARCH_SCENARIOS
                        else None
                    ),
                    "asset_ids": resolved_asset_ids,
                    "asset_manifest_sha256": assets["manifest_sha256"],
                    "feature_set": payload["feature_set"],
                    "horizon_profile": payload["horizon_profile"],
                    **(
                        {
                            "primary_label_policy": primary_label_policy_contract(),
                            "primary_label_policy_sha256": payload[
                                "primary_label_policy_sha256"
                            ],
                        }
                        if scenario.id in HORIZON_RESEARCH_SCENARIOS
                        else {}
                    ),
                    "incumbent_strategy": incumbent_strategy,
                    "managed_fin_strategy_run": managed_run,
                    "strategy_research_signal_binding": (
                        strategy_research_signal_binding
                    ),
                    "expected_rdagent_runtime": expected_runtime_identity,
                },
                log_path,
                dedupe_active_kind=False,
                idempotency_key=(
                    f"managed-fin-strategy:{managed_run['run_sha256']}"
                    if managed_run is not None
                    else f"schedule-run:{run['id']}"
                ),
            )
        except Exception as exc:
            self.research.mark_run(
                research_run["id"], "failed", actor="scheduler", error=str(exc)
            )
            raise
        self.research.attach_job(research_run["id"], job["id"])
        return job

    def _enqueue_ops_task(
        self,
        run: dict[str, Any],
        scheduled_for: datetime,
    ) -> dict[str, Any] | None:
        """Enqueue one operational run-calendar task (design draft §10.4).

        Cadence rules (never weekday guesses beyond the weekly report's fixed
        Saturday, and never a bypass of the calendar gate):
        - weekly_report: Saturday local time only; other days skip.
        - monthly_decision_day: first trading day of the month per the
          persisted Qlib calendar; a calendar that cannot decide fails closed.
        - preopen_check: trading days only per the same calendar.
        """

        kind = str(run["kind"])
        local_date = scheduled_for.astimezone(ZoneInfo(run["timezone"])).date()
        if kind == "weekly_report":
            if not is_weekly_report_day(local_date):
                self.schedules.finish_run(
                    run["id"], "skipped", message="not the weekly report day (Saturday)"
                )
                return None
        else:
            try:
                dataset = select_ops_dataset(
                    self.settings.data_root,
                    str(run["payload"].get("dataset") or "") or None,
                )
                calendar_days = load_calendar_days(dataset["path"])
            except (FileNotFoundError, ValueError) as exc:
                raise ScheduleRunWaiting(
                    "governed operations calendar is not ready"
                ) from exc
            if kind == "monthly_decision_day":
                if local_date not in calendar_days:
                    self.schedules.finish_run(
                        run["id"], "skipped", message="not a Qlib trading day"
                    )
                    return None
                if not is_monthly_decision_day(local_date, calendar_days):
                    self.schedules.finish_run(
                        run["id"],
                        "skipped",
                        message="not the first trading day of the month",
                    )
                    return None
            elif local_date not in calendar_days:
                self.schedules.finish_run(run["id"], "skipped", message="not a Qlib trading day")
                return None
        payload = {
            "local_date": local_date.isoformat(),
            "as_of": scheduled_for.isoformat(),
            "schedule_run_id": run["id"],
            **{
                key: value
                for key, value in dict(run["payload"]).items()
                if key != "schedule_run_id"
            },
        }
        log_path = (
            self.settings.data_root
            / "platform"
            / "logs"
            / f"{kind.replace('_', '-')}-{run['id']}.log"
        )
        return self.jobs.create(
            kind,
            payload,
            log_path,
            idempotency_key=f"schedule-run:{run['id']}",
        )

    def _enqueue_recommendation(
        self,
        run: dict[str, Any],
        scheduled_for: datetime,
    ) -> dict[str, Any] | None:
        portfolio_id = str(run["payload"]["recommendation_portfolio_id"])
        portfolio = self.recommendations.get(portfolio_id)
        version = self.strategies.get_version(
            str(portfolio["strategy_version_id"])
        )
        signal_date = scheduled_for.astimezone(ZoneInfo(run["timezone"])).date()
        # Safe mode (design draft 11.3) is the first check of the
        # recommendation gate: while it is active no new recommendation is
        # generated, regardless of reconciliation health.
        safe_state = self.safe_mode.status()
        if safe_state["active"]:
            self.schedules.finish_run(
                run["id"],
                "skipped",
                message=(
                    f"safe_mode active since {safe_state['triggered_at']}: {safe_state['reason']}"
                ),
            )
            self.alerts.create(
                source_type="recommendation_portfolio",
                source_id=portfolio_id,
                severity="critical",
                category="safe_mode_recommendation_blocked",
                title=f"建议生成被 safe_mode 阻断：{portfolio['name']}",
                message=(
                    f"safe_mode 自 {safe_state['triggered_at']} 起生效"
                    f"（来源 {safe_state['source']}）：{safe_state['reason']}。"
                    "不生成新建议；既有建议快照保留。人工解除 safe_mode 后恢复。"
                ),
                dedupe_key=f"safe-mode-blocked:recommendation:{portfolio_id}:{signal_date.isoformat()}",
                details={"safe_mode": safe_state},
            )
            return None
        if portfolio["status"] != "active":
            self.schedules.finish_run(
                run["id"],
                "skipped",
                message=f"recommendation portfolio is {portfolio['status']}",
            )
            return None
        try:
            dataset = select_qlib_dataset(
                self.settings.data_root,
                anchor_name=portfolio["dataset"],
                roll_policy=str(portfolio.get("dataset_roll_policy") or "pinned"),
                lineage_id=portfolio.get("dataset_lineage_id"),
                required_date=signal_date,
            )
            calendar_days = load_calendar_days(dataset["path"])
        except (FileNotFoundError, ValueError) as exc:
            transient_markers = (
                "not ready and reproducible",
                "does not cover the requested signal date",
                "no verified latest-compatible Qlib descendant covers",
                "calendar is unavailable",
                "calendar is empty",
            )
            if any(marker in str(exc) for marker in transient_markers):
                raise ScheduleRunWaiting(str(exc)) from exc
            raise
        if run["trading_days_only"]:
            if signal_date not in calendar_days:
                self.schedules.finish_run(
                    run["id"],
                    "skipped",
                    message="not a Qlib trading day",
                )
                return None
        # Ordering gate (design draft §10.4): recommendations are only
        # generated after the linked simulation account's latest batch has
        # reconciled and its NAV is healthy/certified/fresh. Failing the gate
        # blocks the new snapshot fail-closed; the previous snapshot stays in
        # place and is explicitly reported as retained/stale, never silently
        # reused as if it were a fresh recommendation.
        gate = evaluate_recommendation_gate(self.simulations, portfolio, signal_date, calendar_days)
        if not gate["passed"]:
            latest_snapshot = portfolio.get("latest_snapshot") or {}
            message = "; ".join(gate["reasons"])
            self.alerts.create(
                source_type="recommendation_portfolio",
                source_id=portfolio_id,
                severity="critical",
                category="recommendation_reconciliation_blocked",
                title=f"建议生成被对账顺序门阻断：{portfolio['name']}",
                message=(
                    f"{message}。不生成新建议；既有建议快照 "
                    f"{latest_snapshot.get('id', '无')}（as_of "
                    f"{latest_snapshot.get('as_of_date', '无')}）保留并视为过期，"
                    "需先恢复模拟对账健康。"
                ),
                dedupe_key=f"recommendation-gate:{portfolio_id}:{signal_date.isoformat()}",
                details={
                    "reasons": gate["reasons"],
                    "gate": gate["details"],
                    "retained_snapshot": {
                        "id": latest_snapshot.get("id"),
                        "as_of_date": str(latest_snapshot.get("as_of_date") or ""),
                        "status": "retained_stale",
                    },
                },
            )
            raise ScheduleRunWaiting(f"reconciliation gate blocked: {message}")
        dataset_identity_sha256 = str(
            dict(dataset.get("provenance") or {}).get("dataset_identity_sha256")
            or ""
        )
        try:
            model_artifact_binding, factor_materialization_binding = (
                self._current_live_signal_bindings(
                    version,
                    dataset_identity_sha256=dataset_identity_sha256,
                    signal_date=signal_date,
                    now=scheduled_for,
                )
            )
        except (KeyError, OSError, ValueError) as exc:
            raise ScheduleRunWaiting(
                "current signal artifacts are still materializing"
            ) from exc
        snapshot, created = self.recommendations.create_snapshot(
            portfolio_id=portfolio_id,
            as_of_date=signal_date,
            dataset=dataset["name"],
            dataset_identity_sha256=dataset_identity_sha256,
            dataset_lineage_id=dataset.get("lineage_id"),
        )
        if not created:
            self.schedules.finish_run(
                run["id"],
                "skipped",
                job_id=snapshot.get("job_id"),
                message=f"existing recommendation snapshot {snapshot['id']}",
            )
            return None
        log_path = (
            self.settings.data_root
            / "platform"
            / "logs"
            / f"recommendation-refresh-{snapshot['id']}.log"
        )
        job = self.jobs.create(
            "recommendation_refresh",
            recommendation_refresh_job_payload(
                snapshot,
                dataset,
                model_artifact_binding=model_artifact_binding,
                factor_materialization_binding=factor_materialization_binding,
            ),
            log_path,
            dedupe_active_kind=False,
            idempotency_key=recommendation_refresh_job_idempotency_key(
                str(snapshot["id"])
            ),
        )
        self.recommendations.attach_job(snapshot["id"], job["id"])
        return job

    def project_alerts(self) -> int:
        created = self.advice_alerts.project()
        # Safe-mode auto trigger (design 11.3): persistent degraded or
        # uncertified NAV on any active simulation account is a severe ledger
        # anomaly. Idempotent while safe mode is already active.
        self.safe_mode.check_persistent_nav_anomalies()
        with self.jobs.engine.connect() as connection:
            failed_jobs = connection.execute(
                select(jobs).where(jobs.c.status == "failed").order_by(jobs.c.finished_at.desc())
            ).all()
            open_risks = connection.execute(
                select(strategy_allocation_events).where(
                    strategy_allocation_events.c.status == "open"
                )
            ).all()
        for row in failed_jobs:
            self.alerts.create(
                source_type="job",
                source_id=str(row.id),
                severity="critical",
                category="job_failure",
                title=f"后台任务失败：{row.kind}",
                message=str(row.error or f"exit code {row.exit_code}"),
                dedupe_key=f"job:{row.id}:failed",
                details={"kind": row.kind, "finished_at": str(row.finished_at)},
            )
            created += 1
        for row in open_risks:
            self.alerts.create(
                source_type="risk_event",
                source_id=str(row.id),
                severity=str(row.severity),
                category="recommendation_allocation_risk",
                title=f"推荐策略组合触发风险阈值：{row.rule}",
                message=f"observed={row.observed}, limit={row.limit_value}",
                dedupe_key=f"allocation-risk-event:{row.id}",
                details={
                    "allocation_id": row.allocation_id,
                    "recommendation_portfolio_id": row.recommendation_portfolio_id,
                },
            )
            created += 1
        return created

    def _project_health_alerts(self, snapshot: dict[str, Any]) -> int:
        created = 0
        bucket = datetime.fromisoformat(snapshot["recorded_at"]).strftime("%Y%m%dT%H")
        for name, component in snapshot["components"].items():
            if component.get("status") not in {"degraded", "unavailable"}:
                continue
            severity = (
                "critical"
                if name in {"qlib_worker", "rdagent_worker", "job_queue", "runtime_secret_storage"}
                else "warning"
            )
            self.alerts.create(
                source_type="system_health",
                source_id=str(snapshot["id"]),
                severity=severity,
                category="component_health",
                title=f"系统组件异常：{name}",
                message=str(component.get("message") or component.get("status")),
                dedupe_key=f"system-health:{name}:{bucket}",
                details={"component": name, **component},
            )
            created += 1
        return created
