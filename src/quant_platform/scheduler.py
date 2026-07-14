from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select

from quant_data.config import Settings
from quant_data.database import jobs, risk_events

from .alert_store import AlertStore
from .broker_gateway import BrokerStore
from .data_rollover import (
    next_qlib_trading_date,
    select_execution_snapshot,
    select_qlib_dataset,
)
from .health_store import OperationalHealthStore
from .job_store import JobStore
from .pair_portfolio_store import PairPortfolioStore
from .portfolio_store import PortfolioStore
from .runtime_secret_store import RuntimeSecretStore
from .schedule_store import (
    RUNNABLE_PAIR_PORTFOLIO_STATUSES,
    RUNNABLE_PORTFOLIO_STATUSES,
    ScheduleStore,
)
from .services import list_qlib_datasets


class SchedulerEngine:
    """Materializes daily slots and safely enqueues durable platform jobs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs = JobStore(settings.database_url)
        self.portfolios = PortfolioStore(settings.database_url)
        self.pair_portfolios = PairPortfolioStore(settings.database_url)
        self.schedules = ScheduleStore(settings.database_url)
        self.alerts = AlertStore(settings.database_url)
        self.health = OperationalHealthStore(settings)
        self.brokers = BrokerStore(settings)
        self.runtime_secrets = RuntimeSecretStore(
            settings.database_url, settings.platform_secret_key
        )

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
            elif run["kind"] == "paper_rebalance":
                job = self._enqueue_portfolio(run, scheduled_for)
                if job is None:
                    return
            elif run["kind"] == "pair_paper_rebalance":
                job = self._enqueue_pair_portfolio(run, scheduled_for)
                if job is None:
                    return
            elif run["kind"] == "broker_reconcile":
                result = self._reconcile_broker(run, scheduled_for)
                if result is None:
                    return
                self.schedules.finish_run(
                    run["id"],
                    "succeeded",
                    message=f"broker reconciliation {result['status']}: {result['id']}",
                    now=now,
                )
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

    def _enqueue_portfolio(
        self,
        run: dict[str, Any],
        scheduled_for: datetime,
    ) -> dict[str, Any] | None:
        portfolio_id = str(run["payload"]["portfolio_id"])
        portfolio = self.portfolios.get(portfolio_id)
        if portfolio["status"] not in RUNNABLE_PORTFOLIO_STATUSES:
            self.schedules.finish_run(
                run["id"],
                "skipped",
                message=f"portfolio is {portfolio['status']}",
            )
            return None
        signal_date = scheduled_for.astimezone(ZoneInfo(run["timezone"])).date()
        dataset = select_qlib_dataset(
            self.settings.data_root,
            anchor_name=portfolio["dataset"],
            roll_policy=portfolio["dataset_roll_policy"],
            lineage_id=portfolio.get("dataset_lineage_id"),
            required_date=signal_date,
        )
        if run["trading_days_only"]:
            calendar = set(
                (Path(dataset["path"]) / "calendars" / "day.txt")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            if signal_date.isoformat() not in calendar:
                self.schedules.finish_run(
                    run["id"],
                    "skipped",
                    message="not a Qlib trading day",
                )
                return None
        artifact_root = self.settings.data_root / "artifacts" / "paper-portfolios" / portfolio_id
        batch, created = self.portfolios.create_batch(
            portfolio_id=portfolio_id,
            as_of_date=signal_date,
            artifact_path=artifact_root,
            dataset_evidence=dataset,
        )
        if not created:
            self.schedules.finish_run(
                run["id"],
                "skipped",
                job_id=batch.get("job_id"),
                message=f"existing idempotent batch {batch['id']}",
            )
            return None
        log_path = (
            self.settings.data_root / "platform" / "logs" / f"paper-rebalance-{batch['id']}.log"
        )
        job = self.jobs.create(
            "paper_rebalance",
            {
                "portfolio_id": portfolio_id,
                "portfolio_batch_id": batch["id"],
                "dataset": dataset["name"],
                "dataset_path": dataset["path"],
                "daily_provenance": dataset["provenance"],
                "as_of_date": signal_date.isoformat(),
                "slippage": float(run["payload"].get("slippage", 0.0005)),
            },
            log_path,
            dedupe_active_kind=False,
            idempotency_key=f"schedule-run:{run['id']}",
        )
        self.portfolios.attach_job(batch["id"], job["id"])
        return job

    def _enqueue_pair_portfolio(
        self,
        run: dict[str, Any],
        scheduled_for: datetime,
    ) -> dict[str, Any] | None:
        portfolio_id = str(run["payload"]["pair_portfolio_id"])
        portfolio = self.pair_portfolios.get(portfolio_id)
        if portfolio["status"] not in RUNNABLE_PAIR_PORTFOLIO_STATUSES:
            self.schedules.finish_run(
                run["id"],
                "skipped",
                message=f"pair portfolio is {portfolio['status']}",
            )
            return None
        signal_date = scheduled_for.astimezone(ZoneInfo(run["timezone"])).date()
        dataset = select_qlib_dataset(
            self.settings.data_root,
            anchor_name=portfolio["dataset"],
            roll_policy=portfolio["dataset_roll_policy"],
            lineage_id=portfolio.get("dataset_lineage_id"),
            required_date=signal_date,
            require_later_date=True,
        )
        calendar = (
            (Path(dataset["path"]) / "calendars" / "day.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        if run["trading_days_only"] and signal_date.isoformat() not in set(calendar):
            self.schedules.finish_run(
                run["id"],
                "skipped",
                message="not a Qlib trading day",
            )
            return None
        trade_date = next_qlib_trading_date(dataset, signal_date)
        execution = select_execution_snapshot(
            self.settings.data_root,
            anchor_name=portfolio["execution_snapshot"],
            roll_policy=portfolio["execution_roll_policy"],
            lineage_id=portfolio.get("execution_lineage_id"),
            required_date=trade_date,
            minute_dataset=portfolio["minute_dataset"],
            shortability_dataset=portfolio["shortability_dataset"],
        )
        artifact_root = (
            self.settings.data_root / "artifacts" / "pair-paper-portfolios" / portfolio_id
        )
        batch, created = self.pair_portfolios.create_batch(
            portfolio_id=portfolio_id,
            as_of_date=signal_date,
            artifact_path=artifact_root,
            dataset_evidence=dataset,
            execution_evidence=execution,
        )
        if not created:
            self.schedules.finish_run(
                run["id"],
                "skipped",
                job_id=batch.get("job_id"),
                message=f"existing idempotent pair batch {batch['id']}",
            )
            return None
        log_path = (
            self.settings.data_root
            / "platform"
            / "logs"
            / f"pair-paper-rebalance-{batch['id']}.log"
        )
        job = self.jobs.create(
            "pair_paper_rebalance",
            {
                "pair_portfolio_id": portfolio_id,
                "pair_portfolio_batch_id": batch["id"],
                "dataset": dataset["name"],
                "dataset_path": dataset["path"],
                "dataset_start": dataset["start_date"],
                "daily_provenance": dataset["provenance"],
                "as_of_date": signal_date.isoformat(),
                "minute_dataset": execution["minute"],
                "shortability_dataset": execution["shortability"],
            },
            log_path,
            dedupe_active_kind=False,
            idempotency_key=f"schedule-run:{run['id']}",
        )
        self.pair_portfolios.attach_job(batch["id"], job["id"])
        return job

    def _reconcile_broker(
        self, run: dict[str, Any], scheduled_for: datetime
    ) -> dict[str, Any] | None:
        destination_id = str(run["payload"]["destination_id"])
        if run["trading_days_only"]:
            destination = self.brokers.get_destination(destination_id)
            portfolio = self.portfolios.get(str(destination["portfolio_id"]))
            datasets = {item["name"]: item for item in list_qlib_datasets(self.settings.data_root)}
            dataset = datasets.get(portfolio["dataset"])
            if not dataset or not dataset["ready"] or not dataset.get("reproducible"):
                raise ValueError("broker shadow portfolio Qlib dataset is not ready")
            local_date = scheduled_for.astimezone(ZoneInfo(run["timezone"])).date()
            calendar = set(
                (Path(dataset["path"]) / "calendars" / "day.txt")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            if local_date.isoformat() not in calendar:
                self.schedules.finish_run(
                    run["id"],
                    "skipped",
                    message="not a Qlib trading day",
                )
                return None
        return self.brokers.reconcile(destination_id, actor="scheduler")

    def project_alerts(self) -> int:
        created = 0
        with self.jobs.engine.connect() as connection:
            failed_jobs = connection.execute(
                select(jobs).where(jobs.c.status == "failed").order_by(jobs.c.finished_at.desc())
            ).all()
            open_risks = connection.execute(
                select(risk_events).where(risk_events.c.status == "open")
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
                category="portfolio_risk",
                title=f"模拟组合触发风险阈值：{row.rule}",
                message=f"observed={row.observed}, limit={row.limit_value}",
                dedupe_key=f"risk-event:{row.id}",
                details={"portfolio_id": row.portfolio_id, "batch_id": row.batch_id},
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
