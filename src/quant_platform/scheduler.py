from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select

from quant_data.cninfo_announcements import load_trade_calendar_open_days
from quant_data.config import Settings
from quant_data.coverage_data import COVERAGE_BUNDLES, DEFAULT_COVERAGE_BUNDLES
from quant_data.database import (
    jobs,
    model_artifacts,
    recommendation_snapshots,
    simulation_batches,
    simulation_portfolios,
    strategy_allocation_events,
)
from quant_data.execution_contract import require_daily_qlib_contract

from .alert_store import AlertStore
from .autonomous_research import AutonomousResearchOrchestrator
from .continuous_research import ContinuousResearchController
from .data_rollover import qlib_trading_date_on_or_before, select_qlib_dataset
from .health_store import OperationalHealthStore
from .information_schedule import (
    STRUCTURED_INFORMATION_STARTS,
    latest_verified_research_asset_snapshot,
    latest_verified_snapshot,
    normalize_information_factor_refresh_payload,
    normalize_information_schedule_payload,
    resolve_information_evaluation_dataset,
)
from .job_store import JobStore, research_asset_acquisition_idempotency_key
from .model_artifact_store import ModelArtifactStore
from .ops_calendar import (
    evaluate_recommendation_gate,
    is_monthly_decision_day,
    is_weekly_report_day,
    load_calendar_days,
    select_ops_dataset,
)
from .rdagent_candidate_store import RDAGentCandidateStore
from .rdagent_runtime import expected_rdagent_runtime_identity, probe_rdagent
from .rdagent_scenarios import (
    get_rdagent_scenario,
    require_ready_scenario,
    resolve_rdagent_assets,
)
from .recommendation_store import RecommendationStore
from .research_asset_store import ResearchAssetStore
from .research_automation import normalize_research_schedule_payload, resolve_research_periods
from .research_store import ResearchStore
from .runtime_secret_store import RuntimeSecretStore
from .safe_mode import SafeModeStore
from .schedule_store import ScheduleStore
from .services import list_qlib_datasets
from .simulation_store import SimulationStore

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


class SchedulerEngine:
    """Materializes daily slots and safely enqueues durable platform jobs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs = JobStore(settings.database_url)
        self.recommendations = RecommendationStore(settings.database_url)
        self.research = ResearchStore(settings.database_url)
        self.rdagent_candidates = RDAGentCandidateStore(settings.database_url)
        self.research_assets = ResearchAssetStore(settings.database_url)
        self.schedules = ScheduleStore(settings.database_url)
        self.alerts = AlertStore(settings.database_url)
        self.health = OperationalHealthStore(settings)
        self.runtime_secrets = RuntimeSecretStore(
            settings.database_url, settings.platform_secret_key
        )
        self.autonomous_research = AutonomousResearchOrchestrator(settings)
        self.continuous_research = ContinuousResearchController(settings)
        self.simulations = SimulationStore(settings.database_url)
        self.model_artifacts = ModelArtifactStore(settings.database_url)
        self.safe_mode = SafeModeStore(settings.database_url)

    def tick(self, now: datetime | None = None) -> dict[str, int]:
        current = now or datetime.now(UTC)
        research_asset_jobs_enqueued = self._enqueue_daily_research_assets(current)
        model_refits_enqueued = self._enqueue_due_model_refits(current)
        materialized = self.schedules.materialize_due(current)
        processed = 0
        while processed < 100:
            run = self.schedules.claim_run(now=current)
            if run is None:
                break
            self._process_run(run, current)
            processed += 1
        program_result = self.continuous_research.tick(limit=5, now=current)
        campaign_result = self.autonomous_research.tick(limit=10)
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
            "research_campaigns_processed": campaign_result["processed"],
            "research_campaigns_deferred": campaign_result["deferred"],
            "research_campaigns_failed": campaign_result["failed"],
            "research_programs_checked": program_result["checked"],
            "research_programs_created": program_result["created"],
            "research_programs_deferred": program_result["deferred"],
            "research_programs_failed": program_result["failed"],
            "alerts_projected": projected,
            "alerts_delivered": delivered,
            "health_recorded": health_recorded,
            "simulation_replays_enqueued": simulation_replays_enqueued,
            "model_refits_enqueued": model_refits_enqueued,
            "research_asset_jobs_enqueued": research_asset_jobs_enqueued,
        }

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

        # arXiv has no dependency on a Tushare research-report snapshot.  Queue
        # it first and under its own idempotency key so a missing entitlement,
        # unavailable report snapshot, or failed Tushare job cannot suppress it.
        enqueued = self._enqueue_research_asset_source(
            research_day=research_day,
            snapshot_name="arxiv-only",
            include_tushare=False,
            include_arxiv=True,
        )
        try:
            snapshot_name = latest_verified_research_asset_snapshot(
                self.settings.data_root,
                as_of=research_day,
            )
        except (OSError, ValueError):
            # Data publication is independently scheduled.  Do not bind an
            # acquisition job to a missing or unverified Tushare snapshot.
            return enqueued
        return enqueued + self._enqueue_research_asset_source(
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
        """Queue one immutable live prediction refresh per approved model strategy."""

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
                version = self.model_artifacts.strategies.get_version(
                    str(row.strategy_version_id)
                )
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
                cutoff_date = row.data_cutoff_at.astimezone(
                    ZoneInfo("Asia/Shanghai")
                ).date()
                if signal_date <= cutoff_date:
                    continue
                job = self.jobs.create(
                    "model_refit",
                    {
                        "strategy_version_id": str(row.strategy_version_id),
                        "source_model_artifact_id": str(row.id),
                        "dataset": dataset["name"],
                        "dataset_path": dataset["path"],
                        "dataset_identity_sha256": dataset["provenance"][
                            "dataset_identity_sha256"
                        ],
                        "dataset_lineage_id": lineage_id,
                        "signal_date": signal_date.isoformat(),
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

    def _enqueue_due_simulation_replays(self, now: datetime) -> int:
        """Bind due forward batches only after immutable execution data exists."""

        local_date = now.astimezone(ZoneInfo("Asia/Shanghai")).date()
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
        scheduled_for = datetime.fromisoformat(run["scheduled_for"])
        delay = (now - scheduled_for).total_seconds()
        if delay > int(run["misfire_grace_seconds"]):
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
            if run["kind"] in {"incremental_sync", "data_pipeline"} and run[
                "trading_days_only"
            ]:
                local_date = scheduled_for.astimezone(
                    ZoneInfo(run["timezone"])
                ).date()
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
                raise ValueError(
                    "research-assets data pipeline requires exactly research_corpus"
                )
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
                },
                log_path,
                idempotency_key=pipeline_id,
            )
        pipeline_steps = [
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
        pipeline_steps.extend(
            {"kind": kind, "payload": {}}
            for kind in ("data_verify", "data_snapshot", "data_qlib", "qlib_baseline")
        )
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
            raise ValueError(
                "information factor refresh Qlib dataset has no bound source snapshot"
            )
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
                        "start": STRUCTURED_INFORMATION_STARTS[
                            "major_news_mentions"
                        ].isoformat(),
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
            "start": min(
                STRUCTURED_INFORMATION_STARTS[source] for source in selected
            ).isoformat(),
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
        local_scheduled_for = scheduled_for.astimezone(ZoneInfo(run["timezone"]))
        local_date = local_scheduled_for.date()
        if local_scheduled_for.time().replace(tzinfo=None) < time(15, 10):
            raise ValueError(
                "A-share five-minute sync must run after the market has fully closed"
            )

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
            raise ValueError(
                "daily Qlib publication does not contain the fully closed trading day"
            )
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
            },
            log_path,
            idempotency_key=f"schedule-run:{run['id']}",
        )

    def _enqueue_research(
        self,
        run: dict[str, Any],
        scheduled_for: datetime,
    ) -> dict[str, Any] | None:
        payload = normalize_research_schedule_payload(
            run["payload"],
            max_loops=self.settings.rdagent_max_loops,
            max_duration=self.settings.rdagent_max_duration,
        )
        scenario = get_rdagent_scenario(payload["scenario"])
        runtime = probe_rdagent(self.settings, Path(__file__).resolve().parents[2])
        require_ready_scenario(runtime, self.settings, scenario.id)
        expected_runtime_identity = expected_rdagent_runtime_identity(
            runtime, scenario.id
        )
        dataset: dict[str, Any] | None = None
        periods: dict[str, str] | None = None
        period_resolution: dict[str, Any] | None = None
        if scenario.requires_dataset:
            datasets = {
                item["name"]: item for item in list_qlib_datasets(self.settings.data_root)
            }
            dataset = datasets.get(payload["dataset"])
            if not dataset or not dataset["ready"] or not dataset.get("reproducible"):
                raise ValueError("scheduled RD-Agent research Qlib dataset is not reproducible")
            calendar = (
                (Path(dataset["path"]) / "calendars" / "day.txt")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            periods, period_resolution = resolve_research_periods(
                calendar,
                periods=payload.get("periods"),
                period_policy=payload.get("period_policy"),
            )
            period_resolution["dataset_identity_sha256"] = dataset["provenance"][
                "dataset_identity_sha256"
            ]
            if dataset.get("start_date") and periods["train_start"] < dataset["start_date"]:
                raise ValueError("scheduled RD-Agent training window starts before the dataset")
            if dataset.get("end_date") and periods["test_end"] > dataset["end_date"]:
                raise ValueError("scheduled RD-Agent test window ends after the dataset")
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
            selection_limit=(
                payload["loop_n"] if scenario.id == "fin_factor_report" else None
            ),
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
        if scenario.requires_dataset and run["trading_days_only"]:
            local_date = scheduled_for.astimezone(ZoneInfo(run["timezone"])).date().isoformat()
            if local_date not in set(calendar):
                self.schedules.finish_run(
                    run["id"], "skipped", message="not a Qlib trading day"
                )
                return None
        elif run["trading_days_only"]:
            raise ValueError("scheduled RD-Agent lab scenarios must disable trading_days_only")
        artifact_root = self.settings.data_root / "artifacts" / "rdagent"
        config: dict[str, Any] = {
            "scenario": scenario.id,
            "asset_ids": resolved_asset_ids,
            "asset_manifest_sha256": assets["manifest_sha256"],
            "asset_selection_mode": "automatic" if auto_selected_assets else "explicit",
            "feature_set": payload["feature_set"],
            "expected_rdagent_runtime": expected_runtime_identity,
        }
        if dataset is not None and periods is not None and period_resolution is not None:
            config.update(
                {
                    "periods": periods,
                    "evaluation_profiles": period_resolution["evaluation_profiles"],
                    "period_resolution": period_resolution,
                    "dataset_path": dataset["path"],
                }
            )
        try:
            research_run = self.research.create_run(
                kind=scenario.research_kind,
                objective=payload["objective"],
                dataset=payload["dataset"] or f"lab:{scenario.id}",
                requested_by=payload["requested_by"],
                budget={"loop_n": payload["loop_n"], "duration": payload["duration"]},
                config=config,
                artifact_path=artifact_root,
            )
        except ValueError as exc:
            if f"active {scenario.research_kind} research run" not in str(exc):
                raise
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
                    "dataset": payload["dataset"] or None,
                    "dataset_path": dataset["path"] if dataset else None,
                    "dataset_identity_sha256": (
                        dataset["provenance"]["dataset_identity_sha256"] if dataset else None
                    ),
                    "dataset_lineage_id": (dataset.get("lineage_id") if dataset else None),
                    "objective": payload["objective"],
                    "loop_n": payload["loop_n"],
                    "duration": payload["duration"],
                    "periods": periods,
                    "evaluation_profiles": (
                        period_resolution["evaluation_profiles"] if period_resolution else []
                    ),
                    "period_resolution": period_resolution,
                    "asset_ids": resolved_asset_ids,
                    "asset_manifest_sha256": assets["manifest_sha256"],
                    "feature_set": payload["feature_set"],
                    "expected_rdagent_runtime": expected_runtime_identity,
                },
                log_path,
                dedupe_active_kind=False,
                idempotency_key=f"schedule-run:{run['id']}",
            )
        except Exception as exc:
            self.research.mark_run(research_run["id"], "failed", actor="scheduler", error=str(exc))
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
            dataset = select_ops_dataset(
                self.settings.data_root,
                str(run["payload"].get("dataset") or "") or None,
            )
            calendar_days = load_calendar_days(dataset["path"])
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
        dataset = select_qlib_dataset(
            self.settings.data_root,
            anchor_name=portfolio["dataset"],
            roll_policy=str(portfolio.get("dataset_roll_policy") or "pinned"),
            lineage_id=portfolio.get("dataset_lineage_id"),
            required_date=signal_date,
        )
        calendar_days = load_calendar_days(dataset["path"])
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
            self.schedules.finish_run(
                run["id"],
                "skipped",
                message=f"reconciliation gate blocked: {message}",
            )
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
            return None
        snapshot, created = self.recommendations.create_snapshot(
            portfolio_id=portfolio_id,
            as_of_date=signal_date,
            dataset=dataset["name"],
            dataset_identity_sha256=dataset["provenance"]["dataset_identity_sha256"],
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
            {
                "recommendation_portfolio_id": portfolio_id,
                "recommendation_snapshot_id": snapshot["id"],
                "dataset": dataset["name"],
                "dataset_path": dataset["path"],
                "dataset_identity_sha256": dataset["provenance"]["dataset_identity_sha256"],
                "as_of_date": signal_date.isoformat(),
            },
            log_path,
            dedupe_active_kind=False,
            idempotency_key=f"schedule-run:{run['id']}",
        )
        self.recommendations.attach_job(snapshot["id"], job["id"])
        return job

    def project_alerts(self) -> int:
        created = 0
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
