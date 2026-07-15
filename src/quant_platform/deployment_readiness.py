from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import distinct, func, select, text

from quant_data.config import Settings
from quant_data.database import (
    alerts,
    allocation_schedule_groups,
    backtest_runs,
    open_database,
    recommendation_portfolios,
    recommendation_snapshots,
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
from .schedule_store import ACTIVE_SCHEDULE_KINDS
from .services import list_qlib_datasets


def _now() -> datetime:
    return datetime.now(UTC)


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
        recommendation_checks = [*research_checks, *self._recommendation_checks()]
        allocation_checks = [*recommendation_checks, *self._allocation_checks(current)]
        pair_checks = [*research_checks, *self._pair_research_checks()]
        profiles = [
            _profile("research", "研究与回测", research_checks),
            _profile("recommendation_tracking", "推荐组合与假设跟踪", recommendation_checks),
            _profile("strategy_allocation", "多策略推荐组合", allocation_checks),
            _profile("pair_research", "配对交易研究", pair_checks),
        ]
        highest_ready = next(
            (item["id"] for item in reversed(profiles) if item["status"] == "ready"),
            None,
        )
        return {
            "generated_at": current.isoformat(timespec="seconds"),
            "policy_version": "2026-07-16.1",
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
            unsupported_active = int(
                connection.scalar(
                    select(func.count())
                    .select_from(schedules)
                    .where(
                        ~schedules.c.kind.in_(ACTIVE_SCHEDULE_KINDS),
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
                "unsupported_schedules_retired",
                "非生产调度已退休",
                unsupported_active == 0,
                f"仍活动的非生产调度 {unsupported_active} 个",
                "将不属于当前研究与推荐管线的调度设为 retired",
            ),
        ]

    def _pair_research_checks(self) -> list[dict[str, Any]]:
        tasks = {item["task_key"]: item for item in self.data_tasks.list()}
        minute_status = str(tasks.get("pair_execution_1m", {}).get("status", "missing"))
        shortability_status = str(tasks.get("margin_eligibility", {}).get("status", "missing"))
        with self.engine.connect() as connection:
            approved_pairs = int(
                connection.scalar(
                    select(func.count())
                    .select_from(strategy_versions)
                    .where(
                        strategy_versions.c.strategy_type == "pair",
                        strategy_versions.c.status == "approved",
                        strategy_versions.c.is_legacy.is_(False),
                    )
                )
                or 0
            )
            validated_backtests = int(
                connection.scalar(
                    select(func.count())
                    .select_from(backtest_runs)
                    .join(
                        strategy_versions,
                        strategy_versions.c.id == backtest_runs.c.strategy_version_id,
                    )
                    .where(
                        strategy_versions.c.strategy_type == "pair",
                        backtest_runs.c.status == "succeeded",
                        backtest_runs.c.is_legacy.is_(False),
                    )
                )
                or 0
            )
        return [
            _check(
                "pair_minute_data",
                "配对研究分钟数据",
                minute_status == "succeeded",
                f"分钟数据任务状态 {minute_status}",
                "完成配对研究所需的分钟数据快照",
            ),
            _check(
                "pair_shortability_data",
                "逐日可融券证据",
                shortability_status == "succeeded",
                f"可融券资格任务状态 {shortability_status}",
                "下载并校验逐日可融券资格",
            ),
            _check(
                "approved_pair_strategy",
                "已审批配对研究策略",
                approved_pairs > 0 and validated_backtests > 0,
                f"已审批版本 {approved_pairs} 个，成功研究回测 {validated_backtests} 个",
                "完成配对研究回测和独立审批；配对执行不属于本系统",
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
        datasets = [
            item
            for item in list_qlib_datasets(self.settings.data_root)
            if item["ready"]
            and item.get("reproducible")
            and item.get("lineage_verified")
            and int(item["trading_days"]) >= 504
        ]
        latest_health = self.health.latest()
        health_ready = bool(latest_health and latest_health["status"] == "ok")
        rdagent_status = (
            str(
                latest_health.get("components", {})
                .get("rdagent_runtime", {})
                .get("status", "missing")
            )
            if latest_health
            else "missing"
        )
        secret_health = self.runtime_secrets.health()
        tushare_record = self.runtime_secrets.describe("tushare")
        tushare_evidence = "未保存经验证的 Tushare 凭据"
        tushare_verified = False
        if tushare_record:
            try:
                credentials = self.runtime_secrets.get("tushare") or {}
                metadata = tushare_record.get("metadata_json") or {}
                tushare_verified = bool(
                    credentials.get("api_url")
                    and credentials.get("token")
                    and metadata.get("verified_at")
                )
                tushare_evidence = (
                    "数据库凭据可解密且具有验证时间"
                    if tushare_verified
                    else "数据库凭据缺少地址、令牌或验证时间"
                )
            except ValueError as exc:
                tushare_evidence = f"数据库凭据无法解密：{exc}"
        elif self.settings.api_url and self.settings.token and pipeline_ready:
            tushare_verified = True
            tushare_evidence = "部署凭据已被完整初始化数据管线验证"
        with self.engine.connect() as connection:
            database_head = connection.scalar(
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
            incremental_schedules = int(
                connection.scalar(
                    select(func.count())
                    .select_from(schedules)
                    .where(
                        schedules.c.kind == "incremental_sync",
                        schedules.c.status == "active",
                    )
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
        return [
            _check(
                "schema_current",
                "数据库结构为当前版本",
                bool(code_head and database_head == code_head),
                f"数据库 {database_head or 'missing'}，代码 {code_head or 'missing'}",
                "执行数据库迁移到当前 Alembic head",
            ),
            _check(
                "authentication_enabled",
                "认证与管理员已启用",
                self.settings.auth_mode == "required" and active_admins > 0,
                f"认证模式 {self.settings.auth_mode}，活动管理员 {active_admins} 个",
                "启用 required 认证并创建活动管理员",
            ),
            _check(
                "runtime_secret_storage",
                "运行时密钥存储可用",
                secret_health["status"] == "ok",
                str(secret_health.get("message") or secret_health["status"]),
                "修复平台密钥并确认已有密文可以解密",
            ),
            _check(
                "tushare_verified",
                "Tushare 凭据已验证",
                tushare_verified,
                tushare_evidence,
                "通过设置接口保存并验证 Tushare 凭据",
            ),
            _check(
                "initialization_pipeline",
                "初始化数据管线完成",
                pipeline_ready,
                f"任务状态 {task_states}",
                "完成下载、校验、快照、Qlib 构建和基线任务",
            ),
            _check(
                "reproducible_qlib_dataset",
                "存在可复现 Qlib 数据集",
                bool(datasets),
                f"满足 504 交易日和血缘要求的数据集 {len(datasets)} 个",
                "构建带数据身份、快照身份和血缘证明的 Qlib 数据集",
            ),
            _check(
                "operational_health",
                "研究运行健康",
                health_ready,
                str(latest_health["status"] if latest_health else "missing"),
                "恢复数据库、Worker、队列和市场数据健康",
            ),
            _check(
                "rdagent_runtime",
                "RD-Agent 运行时可用",
                rdagent_status == "ok",
                f"RD-Agent 状态 {rdagent_status}",
                "配置并启动 RD-Agent 研究运行时",
            ),
            _check(
                "incremental_schedule",
                "增量数据调度已启用",
                incremental_schedules > 0,
                f"活动增量调度 {incremental_schedules} 个",
                "创建活动的 incremental_sync 调度",
            ),
            _check(
                "critical_alerts_clear",
                "严重告警已闭环",
                critical_alerts == 0,
                f"未处理 critical 告警 {critical_alerts} 条",
                "处理所有 critical 告警后重新验收",
            ),
        ]

    def _allocation_checks(self, current: datetime) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            active_rows = connection.execute(
                select(
                    strategy_allocations.c.id,
                    strategy_allocations.c.analysis_json,
                    strategy_allocations.c.max_pairwise_correlation,
                ).where(
                    strategy_allocations.c.status == "active",
                    strategy_allocations.c.is_legacy.is_(False),
                )
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
                            strategy_allocation_members.c.recommendation_portfolio_id.is_not(None),
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
                        .select_from(schedules)
                        .where(
                            schedules.c.kind == "recommendation_refresh",
                            schedules.c.payload_json["allocation_id"].as_string() == str(row.id),
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
                f"活动组合 {len(active_rows)} 个，相关性门禁通过 {len(eligible_ids)} 个",
                "使用至少两个已审批策略建立低相关组合",
                details={"allocations": evidence},
            ),
            _check(
                "recommendation_children",
                "成员推荐组合已配置",
                bool(provisioned_ids),
                f"完整配置成员推荐组合的策略组合 {len(provisioned_ids)} 个",
                "审批组合并为每个成员创建推荐组合",
            ),
            _check(
                "allocation_automation",
                "成员推荐刷新调度已启用",
                bool(automated_ids),
                f"完整启用推荐刷新调度的策略组合 {len(automated_ids)} 个",
                "为每个成员配置 recommendation_refresh 调度",
                details={"automated_allocation_ids": automated_ids},
            ),
            _check(
                "allocation_continuity",
                "组合级假设净值连续",
                continuity_ready,
                (
                    f"组合净值 {nav_days} 个交易日，最新 {latest_nav_date}，距今 {nav_age} 天"
                    if latest_nav_date
                    else "尚无对齐的组合级假设净值"
                ),
                "至少积累 5 个交易日的组合级假设净值",
            ),
            _check(
                "allocation_risk_events_clear",
                "组合级风险事件已闭环",
                open_events == 0,
                f"未关闭的组合级 critical 风险事件 {open_events} 条",
                "处理组合级回撤事件后重新验收",
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
