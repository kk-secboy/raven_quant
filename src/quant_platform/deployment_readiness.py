from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import distinct, func, select, text

from quant_data.config import Settings
from quant_data.database import (
    alerts,
    allocation_schedule_groups,
    allocation_schedule_members,
    backtest_runs,
    broker_destinations,
    broker_order_outbox,
    broker_reconciliations,
    open_database,
    pair_paper_portfolios,
    pair_portfolio_batches,
    pair_portfolio_reviews,
    pair_portfolio_risk_events,
    paper_portfolios,
    portfolio_batches,
    portfolio_reviews,
    recommendation_portfolios,
    recommendation_snapshots,
    risk_events,
    runtime_secrets,
    schedules,
    strategy_allocation_events,
    strategy_allocation_members,
    strategy_allocation_nav,
    strategy_allocations,
    strategy_versions,
    users,
)

from .data_task_store import DataTaskStore
from .health_store import OperationalHealthStore
from .runtime_secret_store import RuntimeSecretStore
from .services import list_qlib_datasets


def _now() -> datetime:
    return datetime.now(UTC)


def _valid_digest(value: Any) -> bool:
    text_value = str(value or "")
    return len(text_value) == 64 and all(
        character in "0123456789abcdef" for character in text_value
    )


def _check(
    check_id: str,
    title: str,
    passed: bool,
    evidence: str,
    remediation: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "title": title,
        "status": "pass" if passed else "block",
        "evidence": evidence,
        "remediation": None if passed else remediation,
        "details": details or {},
    }


def _profile(profile_id: str, title: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    blockers = [item for item in checks if item["status"] == "block"]
    return {
        "id": profile_id,
        "title": title,
        "status": "ready" if not blockers else "blocked",
        "passed": len(checks) - len(blockers),
        "total": len(checks),
        "blocker_count": len(blockers),
        "checks": checks,
    }


class DeploymentReadinessStore:
    """Evidence-backed go/no-go assessment for each supported deployment boundary."""

    def __init__(self, settings: Settings, project_root: Path) -> None:
        self.settings = settings
        self.project_root = project_root.resolve()
        self.engine = open_database(settings.database_url)
        self.data_tasks = DataTaskStore(settings.database_url)
        self.health = OperationalHealthStore(settings)
        self.runtime_secrets = RuntimeSecretStore(
            settings.database_url, settings.platform_secret_key
        )

    def assess(self, now: datetime | None = None) -> dict[str, Any]:
        current = now or _now()
        research_checks = self._research_checks()
        pair_checks = [*research_checks, *self._pair_research_checks()]
        pair_paper_checks: list[dict[str, Any]] = []
        paper_checks: list[dict[str, Any]] = []
        diversified_checks: list[dict[str, Any]] = []
        broker_checks: list[dict[str, Any]] = []
        profiles = [
            _profile("research", "研究与回测", research_checks),
            _profile("pair_research", "配对交易研究", pair_checks),
            _profile("pair_paper", "配对交易模拟盘", pair_paper_checks),
            _profile("paper", "模拟盘", paper_checks),
            _profile("broker_sandbox", "券商沙箱", broker_checks),
            _profile("diversified_paper", "多策略模拟盘", diversified_checks),
        ]
        profiles = [
            profiles[0],
            _profile(
                "recommendation_tracking",
                "推荐组合与假设跟踪",
                [*research_checks, *self._recommendation_checks()],
            ),
            profiles[1],
        ]
        highest_ready = next(
            (item["id"] for item in reversed(profiles) if item["status"] == "ready"),
            None,
        )
        return {
            "generated_at": current.isoformat(timespec="seconds"),
            "policy_version": "2026-07-13.7",
            "highest_ready_profile": highest_ready,
            "live_trading_supported": False,
            "profiles": profiles,
        }

    def _recommendation_checks(self) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            portfolio_count = int(
                connection.scalar(select(func.count()).select_from(recommendation_portfolios)) or 0
            )
            snapshot_count = int(
                connection.scalar(select(func.count()).select_from(recommendation_snapshots)) or 0
            )
            legacy_active = int(
                connection.scalar(
                    select(func.count())
                    .select_from(schedules)
                    .where(
                        schedules.c.kind.in_(
                            ["paper_rebalance", "pair_paper_rebalance", "broker_reconcile"]
                        ),
                        schedules.c.status == "active",
                    )
                )
                or 0
            )
        return [
            _check(
                "recommendation_schema",
                "推荐领域模型已启用",
                True,
                f"推荐组合 {portfolio_count} 个，快照 {snapshot_count} 个",
                "运行数据库迁移后再启用推荐跟踪",
            ),
            _check(
                "legacy_execution_retired",
                "旧执行调度已退休",
                legacy_active == 0,
                f"仍活动的旧执行调度 {legacy_active} 个",
                "将旧 Paper、配对执行和券商调度设为 retired",
            ),
        ]

    def _pair_research_checks(self) -> list[dict[str, Any]]:
        tasks = {item["task_key"]: item for item in self.data_tasks.list()}
        minute_status = str(tasks.get("pair_execution_1m", {}).get("status", "missing"))
        shortability_status = str(tasks.get("cn_margin_eligibility", {}).get("status", "missing"))
        with self.engine.connect() as connection:
            pair_rows = connection.execute(
                select(strategy_versions.c.id, backtest_runs.c.metrics_json)
                .select_from(
                    strategy_versions.join(
                        backtest_runs,
                        backtest_runs.c.strategy_version_id == strategy_versions.c.id,
                    )
                )
                .where(
                    strategy_versions.c.strategy_type == "pair",
                    strategy_versions.c.status == "approved",
                    backtest_runs.c.status == "succeeded",
                    backtest_runs.c.execution_dataset.is_not(None),
                )
            ).all()
            approved_pairs = len(
                {
                    str(row.id)
                    for row in pair_rows
                    if _valid_digest(
                        (row.metrics_json or {})
                        .get("provenance", {})
                        .get("daily_dataset_lineage_id")
                    )
                    and _valid_digest(
                        (row.metrics_json or {})
                        .get("provenance", {})
                        .get("execution_snapshot_lineage_id")
                    )
                }
            )
        return [
            _check(
                "pair_minute_data",
                "配对交易分钟执行数据",
                minute_status == "succeeded",
                f"核心资产 1 分钟任务状态 {minute_status}",
                "先完成核心 ETF/股票池 1 分钟数据下载、校验并生成不可变快照。",
            ),
            _check(
                "pair_shortability_data",
                "逐日融券资格证据",
                shortability_status == "succeeded",
                f"融资融券标的资格任务状态 {shortability_status}",
                "下载逐日标的资格并校验，不能用融资融券成交明细代替可融券资格。",
            ),
            _check(
                "approved_pair_strategy",
                "已审批配对策略",
                approved_pairs > 0,
                f"通过分钟执行、协整滚动、压力门禁和双血缘验证的策略 {approved_pairs} 个",
                "使用已验证日线与执行快照创建 ETF 配对版本，完成分钟回测并由第二位操作员审批。",
            ),
        ]

    def _pair_paper_checks(self, current: datetime) -> list[dict[str, Any]]:
        stale_before = current - timedelta(hours=2)
        recent_failure_after = current - timedelta(days=7)
        with self.engine.connect() as connection:
            portfolios = list(
                connection.execute(
                    select(
                        pair_paper_portfolios.c.id,
                        pair_paper_portfolios.c.name,
                        pair_paper_portfolios.c.dataset_roll_policy,
                        pair_paper_portfolios.c.dataset_lineage_id,
                        pair_paper_portfolios.c.execution_roll_policy,
                        pair_paper_portfolios.c.execution_lineage_id,
                    ).where(pair_paper_portfolios.c.status == "active")
                )
            )
            details: list[dict[str, Any]] = []
            for portfolio in portfolios:
                portfolio_id = str(portfolio.id)
                schedule_count = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(schedules)
                        .where(
                            schedules.c.kind == "pair_paper_rebalance",
                            schedules.c.status == "active",
                            schedules.c.payload_json["pair_portfolio_id"].as_string()
                            == portfolio_id,
                        )
                    )
                    or 0
                )
                review_days = int(
                    connection.scalar(
                        select(func.count(distinct(pair_portfolio_reviews.c.trade_date))).where(
                            pair_portfolio_reviews.c.portfolio_id == portfolio_id,
                            pair_portfolio_reviews.c.status == "completed",
                        )
                    )
                    or 0
                )
                latest_review = connection.scalar(
                    select(func.max(pair_portfolio_reviews.c.trade_date)).where(
                        pair_portfolio_reviews.c.portfolio_id == portfolio_id,
                        pair_portfolio_reviews.c.status == "completed",
                    )
                )
                unresolved_critical = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(pair_portfolio_risk_events)
                        .where(
                            pair_portfolio_risk_events.c.portfolio_id == portfolio_id,
                            pair_portfolio_risk_events.c.severity == "critical",
                            pair_portfolio_risk_events.c.status.in_(["open", "acknowledged"]),
                        )
                    )
                    or 0
                )
                stale_batches = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(pair_portfolio_batches)
                        .where(
                            pair_portfolio_batches.c.portfolio_id == portfolio_id,
                            pair_portfolio_batches.c.status.in_(["queued", "running"]),
                            pair_portfolio_batches.c.created_at < stale_before,
                        )
                    )
                    or 0
                )
                recent_failures = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(pair_portfolio_batches)
                        .where(
                            pair_portfolio_batches.c.portfolio_id == portfolio_id,
                            pair_portfolio_batches.c.status == "failed",
                            pair_portfolio_batches.c.created_at >= recent_failure_after,
                        )
                    )
                    or 0
                )
                evidence_gaps = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(pair_portfolio_batches)
                        .where(
                            pair_portfolio_batches.c.portfolio_id == portfolio_id,
                            pair_portfolio_batches.c.status == "succeeded",
                            pair_portfolio_batches.c.created_at >= recent_failure_after,
                            (
                                pair_portfolio_batches.c.dataset_identity_sha256.is_(None)
                                | pair_portfolio_batches.c.execution_manifest_sha256.is_(None)
                            ),
                        )
                    )
                    or 0
                )
                lineage_configured = (
                    portfolio.dataset_roll_policy == "pinned"
                    or _valid_digest(portfolio.dataset_lineage_id)
                ) and (
                    portfolio.execution_roll_policy == "pinned"
                    or _valid_digest(portfolio.execution_lineage_id)
                )
                review_age = (current.date() - latest_review).days if latest_review else None
                continuity = review_days >= 5 and review_age is not None and 0 <= review_age <= 7
                risk_clean = not unresolved_critical and not stale_batches and not recent_failures
                data_governed = lineage_configured and not evidence_gaps
                details.append(
                    {
                        "id": portfolio_id,
                        "name": str(portfolio.name),
                        "active_schedules": schedule_count,
                        "review_days": review_days,
                        "latest_review": latest_review.isoformat() if latest_review else None,
                        "review_age_days": review_age,
                        "unresolved_critical": unresolved_critical,
                        "stale_batches": stale_batches,
                        "recent_failures": recent_failures,
                        "dataset_roll_policy": str(portfolio.dataset_roll_policy),
                        "execution_roll_policy": str(portfolio.execution_roll_policy),
                        "lineage_configured": lineage_configured,
                        "recent_batch_evidence_gaps": evidence_gaps,
                        "continuity": continuity,
                        "risk_clean": risk_clean,
                        "data_governed": data_governed,
                        "governed": bool(
                            schedule_count and continuity and risk_clean and data_governed
                        ),
                    }
                )
        scheduled = sum(1 for item in details if item["active_schedules"])
        continuous = sum(1 for item in details if item["continuity"])
        risk_clean = sum(1 for item in details if item["risk_clean"])
        data_governed = sum(1 for item in details if item["data_governed"])
        governed = sum(1 for item in details if item["governed"])
        best_review_days = max((int(item["review_days"]) for item in details), default=0)
        return [
            _check(
                "pair_paper_portfolio",
                "专用双腿模拟账本",
                bool(details),
                f"活动双腿账本 {len(details)} 个",
                "从已审批配对版本创建独立价差账本，禁止复用多头模拟组合。",
                details={"portfolios": details},
            ),
            _check(
                "pair_paper_schedule",
                "交易日日终自动调度",
                scheduled > 0,
                f"具有活动 pair_paper_rebalance 调度的账本 {scheduled} 个",
                "为双腿账本创建交易日 15:30 之后运行的自动调度。",
            ),
            _check(
                "pair_paper_continuity",
                "连续模拟与盘后复盘",
                continuous > 0,
                f"最多连续复盘 {best_review_days} 个交易日；达标账本 {continuous} 个",
                "至少完成 5 个不同交易日的原子双腿记账，且最近一次复盘不超过 7 天。",
            ),
            _check(
                "pair_paper_data_lineage",
                "配对批次数据血缘",
                data_governed > 0,
                f"策略有效且近 7 日批次证据完整的账本 {data_governed} 个",
                "为账本选择固定或同血缘推进策略，并以新版本重新运行缺少 Qlib/执行快照身份的批次。",
            ),
            _check(
                "pair_paper_risk_clean",
                "配对风险与批次清零",
                risk_clean > 0,
                f"无未解决 critical、陈旧批次和近 7 日失败的账本 {risk_clean} 个",
                "完成强制平仓与风险处置，修复失败批次并恢复连续运行。",
            ),
            _check(
                "pair_paper_governed_run",
                "同一账本完整验收",
                governed > 0,
                f"同时满足调度、连续性和风险门禁的账本 {governed} 个",
                "必须由同一个活动双腿账本同时通过自动调度、连续复盘和风险清零门禁。",
            ),
        ]

    def _research_checks(self) -> list[dict[str, Any]]:
        tasks = {item["task_key"]: item for item in self.data_tasks.list()}
        required_pipeline = (
            "cn_ashare_daily_full",
            "cn_data_verify",
            "cn_snapshot_build",
            "cn_qlib_build",
            "cn_qlib_baseline",
        )
        task_states = {
            key: str(tasks.get(key, {}).get("status", "missing")) for key in required_pipeline
        }
        pipeline_ready = all(status == "succeeded" for status in task_states.values())

        eligible_datasets = [
            item
            for item in list_qlib_datasets(self.settings.data_root)
            if item["ready"]
            and item.get("reproducible")
            and item.get("lineage_verified")
            and int(item["trading_days"]) >= 504
        ]
        best_dataset = max(
            eligible_datasets, key=lambda item: int(item["trading_days"]), default=None
        )
        latest_health = self.health.latest()
        health_age = float(latest_health.get("age_seconds", 0)) if latest_health else None
        health_ready = bool(latest_health and latest_health["status"] == "ok")
        rdagent_status = (
            latest_health.get("components", {}).get("rdagent_runtime", {}).get("status")
            if latest_health
            else None
        )

        with self.engine.connect() as connection:
            schema_revision = connection.scalar(
                text("SELECT version_num FROM quantlab.alembic_version")
            )
            active_admins = int(
                connection.scalar(
                    select(func.count())
                    .select_from(users)
                    .where(users.c.role == "admin", users.c.active.is_(True))
                )
                or 0
            )
            tushare_record = connection.execute(
                select(runtime_secrets.c.metadata_json, runtime_secrets.c.updated_at).where(
                    runtime_secrets.c.name == "tushare"
                )
            ).first()
            successful_bootstraps = int(
                connection.scalar(
                    text(
                        "SELECT count(*) FROM quantlab.jobs "
                        "WHERE kind = 'bootstrap' AND status = 'succeeded'"
                    )
                )
                or 0
            )
            active_syncs = int(
                connection.scalar(
                    select(func.count())
                    .select_from(schedules)
                    .where(schedules.c.kind == "incremental_sync", schedules.c.status == "active")
                )
                or 0
            )
            critical_alerts = int(
                connection.scalar(
                    select(func.count())
                    .select_from(alerts)
                    .where(alerts.c.severity == "critical", alerts.c.status == "open")
                )
                or 0
            )

        code_head = self._code_schema_head()
        auth_ready = self.settings.auth_mode == "required" and active_admins > 0
        secret_storage = self.runtime_secrets.health()
        credential_verified_at = None
        credential_payload: dict[str, str] | None = None
        credential_error = ""
        if tushare_record:
            credential_verified_at = (tushare_record.metadata_json or {}).get("verified_at")
            try:
                credential_payload = self.runtime_secrets.get("tushare")
            except ValueError as exc:
                credential_error = str(exc)
        credentials_ready = bool(
            (
                credential_verified_at
                and credential_payload
                and credential_payload.get("api_url")
                and credential_payload.get("token")
            )
            or (
                not tushare_record
                and self.settings.api_url
                and self.settings.token
                and successful_bootstraps > 0
            )
        )

        return [
            _check(
                "schema_current",
                "数据库迁移版本",
                bool(code_head and schema_revision == code_head),
                f"数据库 {schema_revision or '未知'}；代码 {code_head or '无法读取'}",
                "执行 Alembic 升级，并确认数据库版本与当前代码头一致。",
            ),
            _check(
                "authentication_enabled",
                "身份认证与管理员",
                auth_ready,
                f"AUTH_MODE={self.settings.auth_mode}；活动管理员 {active_admins} 个",
                "启用 required 认证并完成首个管理员初始化。",
            ),
            _check(
                "runtime_secret_storage",
                "运行时密钥存储",
                secret_storage["status"] == "ok",
                str(secret_storage["message"]),
                "配置并保管 PLATFORM_SECRET_KEY；已有密文时必须恢复创建它们的原密钥。",
                details={"record_count": int(secret_storage["record_count"])},
            ),
            _check(
                "tushare_verified",
                "Tushare 凭据已验证",
                credentials_ready,
                (
                    (
                        "数据库凭据记录无法解密"
                        if credential_error
                        else f"数据库验证时间 {credential_verified_at}"
                    )
                    if credential_verified_at
                    else f"成功初始化任务 {successful_bootstraps} 个"
                ),
                "恢复 PLATFORM_SECRET_KEY 后在系统设置中保存并验证 Tushare Token，"
                "或使用环境凭据完成一次成功初始化。",
            ),
            _check(
                "initialization_pipeline",
                "初始化收口流水线",
                pipeline_ready,
                "；".join(f"{key}={value}" for key, value in task_states.items()),
                "等待下载完成后依次通过质量校验、不可变快照、Qlib 构建和基线验收。",
                details={"tasks": task_states},
            ),
            _check(
                "reproducible_qlib_dataset",
                "可复现 Qlib 数据集",
                bool(best_dataset),
                (
                    f"{best_dataset['name']}，{best_dataset['trading_days']} 个交易日，"
                    f"{best_dataset['start_date']} 至 {best_dataset['end_date']}"
                    if best_dataset
                    else "没有同时满足可用、血缘验证且不少于 504 个交易日的数据集"
                ),
                "完成数据收口并生成带不可变溯源和已验证血缘的 Qlib 数据集。",
                details={"eligible_datasets": len(eligible_datasets)},
            ),
            _check(
                "operational_health",
                "持久化运行健康",
                health_ready,
                (
                    f"最近状态 {latest_health['status']}，{int(health_age or 0)} 秒前"
                    if latest_health
                    else "尚无调度器写入的健康快照"
                ),
                "启动调度器和工作进程，处理不可用组件并等待新的健康快照。",
            ),
            _check(
                "rdagent_runtime",
                "RD-Agent 受控运行时",
                rdagent_status == "ok",
                f"最近健康证据状态 {rdagent_status or '缺失'}",
                "配置模型凭据并确认 RD-Agent worker 与隔离沙箱健康。",
            ),
            _check(
                "incremental_schedule",
                "每日增量数据计划",
                active_syncs > 0,
                f"活动 incremental_sync 计划 {active_syncs} 个",
                "创建并启用每日增量同步计划。",
            ),
            _check(
                "critical_alerts_clear",
                "严重告警清零",
                critical_alerts == 0,
                f"未处理 critical 告警 {critical_alerts} 条",
                "处理或确认所有严重告警后重新验收。",
            ),
        ]

    def _paper_checks(self, current: datetime) -> list[dict[str, Any]]:
        recent_after = current - timedelta(days=7)
        with self.engine.connect() as connection:
            approved_versions = int(
                connection.scalar(
                    select(func.count())
                    .select_from(strategy_versions)
                    .where(strategy_versions.c.status == "approved")
                )
                or 0
            )
            governed_backtests = int(
                connection.scalar(
                    select(func.count(distinct(backtest_runs.c.strategy_version_id)))
                    .select_from(
                        backtest_runs.join(
                            strategy_versions,
                            strategy_versions.c.id == backtest_runs.c.strategy_version_id,
                        )
                    )
                    .where(
                        strategy_versions.c.status == "approved",
                        backtest_runs.c.status == "succeeded",
                    )
                )
                or 0
            )
            active_rows = list(
                connection.execute(
                    select(
                        paper_portfolios.c.id,
                        paper_portfolios.c.dataset_roll_policy,
                        paper_portfolios.c.dataset_lineage_id,
                    ).where(paper_portfolios.c.status == "active")
                )
            )
            active_portfolios = len(active_rows)
            lineage_configured = all(
                row.dataset_roll_policy == "pinned" or _valid_digest(row.dataset_lineage_id)
                for row in active_rows
            )
            active_ids = [str(row.id) for row in active_rows]
            missing_batch_evidence = (
                int(
                    connection.scalar(
                        select(func.count())
                        .select_from(portfolio_batches)
                        .where(
                            portfolio_batches.c.portfolio_id.in_(active_ids),
                            portfolio_batches.c.status == "succeeded",
                            portfolio_batches.c.created_at >= recent_after,
                            portfolio_batches.c.dataset_identity_sha256.is_(None),
                        )
                    )
                    or 0
                )
                if active_ids
                else 0
            )
            paper_schedules = int(
                connection.scalar(
                    select(func.count())
                    .select_from(schedules)
                    .where(schedules.c.kind == "paper_rebalance", schedules.c.status == "active")
                )
                or 0
            )
            review_days = int(
                connection.scalar(select(func.count(distinct(portfolio_reviews.c.trade_date)))) or 0
            )
            latest_review_date = connection.scalar(select(func.max(portfolio_reviews.c.trade_date)))
            open_risk_events = int(
                connection.scalar(
                    select(func.count())
                    .select_from(risk_events)
                    .where(risk_events.c.status.in_(["open", "acknowledged"]))
                )
                or 0
            )
        review_age = (current.date() - latest_review_date).days if latest_review_date else None
        continuity_ready = review_days >= 5 and review_age is not None and review_age <= 7
        return [
            _check(
                "approved_strategy",
                "已审批策略版本",
                approved_versions > 0 and governed_backtests > 0,
                f"已审批版本 {approved_versions} 个；有成功治理回测的版本 {governed_backtests} 个",
                "完成 Qlib 治理回测，通过门禁后由管理员审批策略版本。",
            ),
            _check(
                "active_paper_portfolio",
                "活动模拟组合",
                active_portfolios > 0,
                f"活动模拟组合 {active_portfolios} 个",
                "使用已审批策略创建并启用模拟组合。",
            ),
            _check(
                "paper_data_lineage",
                "模拟批次数据血缘",
                bool(active_rows) and lineage_configured and missing_batch_evidence == 0,
                (
                    f"活动账本 {active_portfolios} 个；近 7 日缺少 Qlib 身份的成功批次 "
                    f"{missing_batch_evidence} 个"
                ),
                "为账本选择固定或同血缘推进策略，并以新版本重新运行缺少 Qlib 身份的批次。",
            ),
            _check(
                "paper_schedule",
                "自动模拟调仓计划",
                paper_schedules > 0,
                f"活动 paper_rebalance 计划 {paper_schedules} 个",
                "为模拟组合创建收盘后的自动调仓计划。",
            ),
            _check(
                "paper_continuity",
                "连续模拟运行证据",
                continuity_ready,
                (
                    f"已复盘 {review_days} 个交易日；最近复盘 {latest_review_date}，"
                    f"距今 {review_age} 天"
                    if latest_review_date
                    else "尚无不可变盘后复盘"
                ),
                "至少完成 5 个交易日的模拟调仓与盘后复盘，并保持最近 7 天内有证据。",
            ),
            _check(
                "paper_risk_events_clear",
                "模拟盘风险事件闭环",
                open_risk_events == 0,
                f"未关闭风险事件 {open_risk_events} 条",
                "处理风险事件并确认组合状态后重新验收。",
            ),
        ]

    def _allocation_checks(self, current: datetime) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            active_rows = connection.execute(
                select(
                    strategy_allocations.c.id,
                    strategy_allocations.c.analysis_json,
                    strategy_allocations.c.max_pairwise_correlation,
                ).where(strategy_allocations.c.status == "active")
            ).all()
            eligible_ids: list[str] = []
            provisioned_ids: list[str] = []
            automated_ids: list[str] = []
            evidence: list[dict[str, Any]] = []
            for row in active_rows:
                analysis = row.analysis_json or {}
                observed = analysis.get("highest_pairwise_correlation")
                member_count = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(strategy_allocation_members)
                        .where(strategy_allocation_members.c.allocation_id == row.id)
                    )
                    or 0
                )
                provisioned_count = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(strategy_allocation_members)
                        .where(
                            strategy_allocation_members.c.allocation_id == row.id,
                            strategy_allocation_members.c.portfolio_id.is_not(None),
                        )
                    )
                    or 0
                )
                correlation_passed = (
                    observed is not None
                    and float(observed) <= float(row.max_pairwise_correlation)
                    and member_count >= 2
                )
                if correlation_passed:
                    eligible_ids.append(str(row.id))
                if correlation_passed and provisioned_count == member_count:
                    provisioned_ids.append(str(row.id))
                group_status = connection.scalar(
                    select(allocation_schedule_groups.c.status).where(
                        allocation_schedule_groups.c.allocation_id == row.id
                    )
                )
                scheduled_count = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(allocation_schedule_members)
                        .join(
                            schedules,
                            schedules.c.id == allocation_schedule_members.c.schedule_id,
                        )
                        .where(
                            allocation_schedule_members.c.allocation_id == row.id,
                            schedules.c.desired_status == "active",
                            schedules.c.status == "active",
                        )
                    )
                    or 0
                )
                automation_ready = (
                    correlation_passed
                    and provisioned_count == member_count
                    and group_status == "active"
                    and scheduled_count == member_count
                )
                if automation_ready:
                    automated_ids.append(str(row.id))
                evidence.append(
                    {
                        "allocation_id": str(row.id),
                        "correlation": observed,
                        "limit": float(row.max_pairwise_correlation),
                        "members": member_count,
                        "provisioned": provisioned_count,
                        "schedule_group": group_status,
                        "scheduled": scheduled_count,
                    }
                )
            nav_days = 0
            latest_nav_date = None
            if provisioned_ids:
                nav_days = int(
                    connection.scalar(
                        select(func.count(distinct(strategy_allocation_nav.c.trade_date))).where(
                            strategy_allocation_nav.c.allocation_id.in_(provisioned_ids)
                        )
                    )
                    or 0
                )
                latest_nav_date = connection.scalar(
                    select(func.max(strategy_allocation_nav.c.trade_date)).where(
                        strategy_allocation_nav.c.allocation_id.in_(provisioned_ids)
                    )
                )
            open_events = 0
            if provisioned_ids:
                open_events = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(strategy_allocation_events)
                        .where(
                            strategy_allocation_events.c.allocation_id.in_(provisioned_ids),
                            strategy_allocation_events.c.severity == "critical",
                            strategy_allocation_events.c.status.in_(["open", "acknowledged"]),
                        )
                    )
                    or 0
                )
        nav_age = (current.date() - latest_nav_date).days if latest_nav_date else None
        continuity_ready = nav_days >= 5 and nav_age is not None and nav_age <= 7
        return [
            _check(
                "low_correlation_allocation",
                "低相关策略组合",
                bool(eligible_ids),
                f"活动组合 {len(active_rows)} 个；相关性门禁通过 {len(eligible_ids)} 个",
                "使用至少两个已审批策略及其重叠 Qlib 日收益创建低相关组合。",
                details={"allocations": evidence},
            ),
            _check(
                "risk_budget_children",
                "风险预算子组合已配置",
                bool(provisioned_ids),
                f"已按风险预算创建全部子模拟组合 {len(provisioned_ids)} 个",
                "由第二位管理员审批组合，使目标风险权重落实为独立子模拟账本。",
            ),
            _check(
                "allocation_automation",
                "组合子策略自动调度",
                bool(automated_ids),
                f"完整启用组合调度 {len(automated_ids)} 个",
                "为已启用组合配置盘后调度，并确认每个子组合的期望状态和实际状态均为 active。",
                details={"automated_allocation_ids": automated_ids},
            ),
            _check(
                "allocation_continuity",
                "组合级净值连续性",
                continuity_ready,
                (
                    f"组合净值 {nav_days} 个交易日；最新 {latest_nav_date}；距今 {nav_age} 天"
                    if latest_nav_date
                    else "尚无对齐的组合级净值记录"
                ),
                "至少完成 5 个交易日的全部子组合调仓，并保持最近 7 天内有组合净值。",
            ),
            _check(
                "allocation_risk_events_clear",
                "组合级风险事件闭环",
                open_events == 0,
                f"未关闭组合级 critical 风险事件 {open_events} 条",
                "处理成员 8% 回撤或组合 10%/15% 熔断事件后重新验收。",
            ),
        ]

    def _broker_checks(self, current: datetime) -> list[dict[str, Any]]:
        readiness = self.brokers.readiness(probe=False)
        latest_health = self.health.latest()
        broker_health = (
            latest_health.get("components", {}).get("broker_boundary", {}).get("status")
            if latest_health
            else None
        )
        with self.engine.connect() as connection:
            active_destinations = int(
                connection.scalar(
                    select(func.count())
                    .select_from(broker_destinations)
                    .where(broker_destinations.c.status == "active")
                )
                or 0
            )
            reconcile_schedules = int(
                connection.scalar(
                    select(func.count())
                    .select_from(schedules)
                    .where(schedules.c.kind == "broker_reconcile", schedules.c.status == "active")
                )
                or 0
            )
            latest_match = connection.execute(
                select(broker_reconciliations.c.created_at, broker_reconciliations.c.broker_as_of)
                .where(broker_reconciliations.c.status == "matched")
                .order_by(broker_reconciliations.c.created_at.desc())
                .limit(1)
            ).first()
            failed_outbox = int(
                connection.scalar(
                    select(func.count())
                    .select_from(broker_order_outbox)
                    .where(broker_order_outbox.c.status == "failed")
                )
                or 0
            )
        match_age = (
            (current - latest_match.created_at).total_seconds() / 86400 if latest_match else None
        )
        reconciliation_ready = match_age is not None and match_age <= 7
        return [
            _check(
                "broker_sandbox_attested",
                "券商沙箱运行时",
                self.settings.broker_mode == "sandbox" and broker_health == "ok",
                f"模式 {self.settings.broker_mode}；最近健康证据 {broker_health or '缺失'}",
                "配置 QMT 模拟网关并通过调度器的签名健康探测；实盘模式仍保持禁用。",
                details={"boundary": readiness},
            ),
            _check(
                "active_broker_destination",
                "双人激活的沙箱目的地",
                active_destinations > 0,
                f"活动沙箱目的地 {active_destinations} 个",
                "将沙箱账户绑定模拟组合，并由两名不同管理员完成激活。",
            ),
            _check(
                "broker_reconcile_schedule",
                "券商自动对账计划",
                reconcile_schedules > 0,
                f"活动 broker_reconcile 计划 {reconcile_schedules} 个",
                "创建并启用每日收盘后券商对账计划。",
            ),
            _check(
                "broker_reconciliation_matched",
                "最近券商对账一致",
                reconciliation_ready,
                (
                    f"最近一致对账 {latest_match.created_at.isoformat(timespec='seconds')}，"
                    f"距今 {match_age:.1f} 天"
                    if latest_match and match_age is not None
                    else "尚无一致对账证据"
                ),
                "完成真实 QMT 模拟账户快照对账，并保持最近 7 天内有一致证据。",
            ),
            _check(
                "broker_delivery_clear",
                "券商投递失败清零",
                failed_outbox == 0 and readiness.get("status") != "degraded",
                f"失败 outbox {failed_outbox} 条；边界状态 {readiness.get('status')}",
                "处理失败投递或对账锁，重新验证后再启用沙箱执行。",
            ),
        ]

    def _code_schema_head(self) -> str | None:
        try:
            config = Config(str(self.project_root / "alembic.ini"))
            config.set_main_option("script_location", str(self.project_root / "migrations"))
            heads = ScriptDirectory.from_config(config).get_heads()
        except Exception:  # pragma: no cover - deployment diagnostic must fail closed
            return None
        return heads[0] if len(heads) == 1 else None
