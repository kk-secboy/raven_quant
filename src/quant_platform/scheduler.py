from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select

from quant_data.config import Settings
from quant_data.coverage_data import DEFAULT_COVERAGE_BUNDLES
from quant_data.database import jobs, strategy_allocation_events

from .alert_store import AlertStore
from .autonomous_research import AutonomousResearchOrchestrator
from .continuous_research import ContinuousResearchController
from .data_rollover import select_qlib_dataset
from .health_store import OperationalHealthStore
from .job_store import JobStore
from .ops_calendar import (
    evaluate_recommendation_gate,
    is_monthly_decision_day,
    is_weekly_report_day,
    load_calendar_days,
    select_ops_dataset,
)
from .recommendation_store import RecommendationStore
from .research_automation import normalize_research_schedule_payload
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


class SchedulerEngine:
    """Materializes daily slots and safely enqueues durable platform jobs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs = JobStore(settings.database_url)
        self.recommendations = RecommendationStore(settings.database_url)
        self.research = ResearchStore(settings.database_url)
        self.schedules = ScheduleStore(settings.database_url)
        self.alerts = AlertStore(settings.database_url)
        self.health = OperationalHealthStore(settings)
        self.runtime_secrets = RuntimeSecretStore(
            settings.database_url, settings.platform_secret_key
        )
        self.autonomous_research = AutonomousResearchOrchestrator(settings)
        self.continuous_research = ContinuousResearchController(settings)
        self.simulations = SimulationStore(settings.database_url)
        self.safe_mode = SafeModeStore(settings.database_url)

    def tick(self, now: datetime | None = None) -> dict[str, int]:
        current = now or datetime.now(UTC)
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
        }

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
            if run["kind"] == "incremental_sync":
                job = self._enqueue_incremental(run, scheduled_for)
            elif run["kind"] == "data_pipeline":
                job = self._enqueue_data_pipeline(run, scheduled_for)
            elif run["kind"] == "ashare_5m_sync":
                job = self._enqueue_ashare_5m(run, scheduled_for)
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
        snapshot_start = str(payload.get("snapshot_start", "2024-01-01"))
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
        snapshot_start = str(payload.get("snapshot_start", "2024-01-01"))
        bundles = payload.get("bundles") or list(AUTOMATED_DATA_BUNDLES)
        unknown = sorted(set(bundles) - set(AUTOMATED_DATA_BUNDLES))
        if unknown:
            raise ValueError(f"unsupported automated data bundles: {unknown}")
        incremental_start = (local_date - timedelta(days=lookback_days)).isoformat()
        snapshot_name = f"cn-{snapshot_start.replace('-', '')}-{local_date:%Y%m%d}"
        pipeline_steps = [
            {
                "kind": f"supplemental_{bundle}",
                "payload": {
                    "bundle": bundle,
                    "start": incremental_start,
                    "end": "latest",
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
                "profile": payload.get("profile", "full"),
                "start": incremental_start,
                "end": "latest",
                "build_qlib": False,
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

    def _enqueue_ashare_5m(
        self,
        run: dict[str, Any],
        scheduled_for: datetime,
    ) -> dict[str, Any]:
        stored = self.runtime_secrets.get("tushare")
        if not stored and (not self.settings.api_url or not self.settings.token):
            raise ValueError("Tushare credentials are not configured")
        payload = run["payload"]
        local_date = scheduled_for.astimezone(ZoneInfo(run["timezone"])).date()
        lookback_days = max(1, min(30, int(payload.get("lookback_days", 3))))
        start = local_date - timedelta(days=lookback_days - 1)
        snapshot_name = f"ashare-5m-incremental-{local_date:%Y%m%d}"
        log_path = (
            self.settings.data_root / "platform" / "logs" / f"scheduled-ashare-5m-{run['id']}.log"
        )
        return self.jobs.create(
            "ashare_5m_download",
            {
                "start": start.isoformat(),
                "end": local_date.isoformat(),
                "snapshot_name": snapshot_name,
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
            run["payload"], max_loops=self.settings.rdagent_max_loops
        )
        datasets = {item["name"]: item for item in list_qlib_datasets(self.settings.data_root)}
        dataset = datasets.get(payload["dataset"])
        if not dataset or not dataset["ready"] or not dataset.get("reproducible"):
            raise ValueError("scheduled RD-Agent research Qlib dataset is not reproducible")
        periods = payload["periods"]
        if dataset.get("start_date") and periods["train_start"] < dataset["start_date"]:
            raise ValueError("scheduled RD-Agent training window starts before the dataset")
        if dataset.get("end_date") and periods["test_end"] > dataset["end_date"]:
            raise ValueError("scheduled RD-Agent test window ends after the dataset")
        if run["trading_days_only"]:
            local_date = scheduled_for.astimezone(ZoneInfo(run["timezone"])).date().isoformat()
            calendar = set(
                (Path(dataset["path"]) / "calendars" / "day.txt")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            if local_date not in calendar:
                self.schedules.finish_run(run["id"], "skipped", message="not a Qlib trading day")
                return None
        artifact_root = self.settings.data_root / "artifacts" / "rdagent"
        try:
            research_run = self.research.create_run(
                kind="factor",
                objective=payload["objective"],
                dataset=payload["dataset"],
                requested_by=payload["requested_by"],
                budget={"loop_n": payload["loop_n"], "duration": payload["duration"]},
                config={"periods": periods, "dataset_path": dataset["path"]},
                artifact_path=artifact_root,
            )
        except ValueError as exc:
            if "active factor research run" not in str(exc):
                raise
            self.schedules.finish_run(
                run["id"],
                "skipped",
                message="a bounded factor research run is already active",
            )
            return None
        log_path = (
            self.settings.data_root
            / "platform"
            / "logs"
            / f"rdagent-factor-{research_run['id']}.log"
        )
        try:
            job = self.jobs.create(
                "rdagent_factor",
                {
                    "research_run_id": research_run["id"],
                    "dataset": payload["dataset"],
                    "dataset_path": dataset["path"],
                    "dataset_identity_sha256": dataset["provenance"]["dataset_identity_sha256"],
                    "objective": payload["objective"],
                    "loop_n": payload["loop_n"],
                    "duration": payload["duration"],
                    "periods": periods,
                },
                log_path,
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
                self.schedules.finish_run(
                    run["id"], "skipped", message="not a Qlib trading day"
                )
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
                    f"safe_mode active since {safe_state['triggered_at']}: "
                    f"{safe_state['reason']}"
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
            roll_policy="pinned",
            lineage_id=None,
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
        gate = evaluate_recommendation_gate(
            self.simulations, portfolio, signal_date, calendar_days
        )
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
