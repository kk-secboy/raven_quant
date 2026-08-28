from __future__ import annotations

from datetime import UTC, date, datetime, time
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
    simulation_nav,
    simulation_portfolios,
    strategy_allocation_events,
    strategy_allocation_members,
    strategy_allocation_nav,
    strategy_allocations,
    strategy_versions,
    users,
)

from .data_task_store import DataTaskStore
from .health_store import OperationalHealthStore
from .information_schedule import (
    STRUCTURED_INFORMATION_SOURCES,
    normalize_information_factor_refresh_payload,
    normalize_information_schedule_payload,
    resolve_information_evaluation_dataset,
)
from .research_automation import (
    DEFAULT_REQUIRED_RESEARCH_TRADING_DAYS,
    DEFAULT_RESEARCH_PERIOD_POLICY,
    MINIMUM_PROFILE_TRAINING_DAYS,
    RESEARCH_EVALUATION_PROFILES,
)
from .runtime_secret_store import RuntimeSecretStore
from .schedule_store import ACTIVE_SCHEDULE_KINDS
from .scheduler import AUTOMATED_DATA_BUNDLES
from .services import list_qlib_datasets_for_display

RESEARCH_LONGEST_VALIDATION_TRADING_DAYS = max(
    int(profile["validation_trading_days"])
    for profile in RESEARCH_EVALUATION_PROFILES
)
RESEARCH_MINIMUM_TRADING_DAYS = DEFAULT_REQUIRED_RESEARCH_TRADING_DAYS
_GOVERNED_DATA_PIPELINE_BUNDLES = frozenset(AUTOMATED_DATA_BUNDLES)
_GOVERNED_INFORMATION_CORPUS_DATASETS = frozenset(
    {"cctv_news", "irm_qa_sh", "irm_qa_sz", "major_news"}
)
_GOVERNED_DATA_SCHEDULE_SUITE_KINDS = frozenset(
    {
        "data_pipeline",
        "information_pipeline",
        "information_factor_refresh",
        "ashare_5m_sync",
        "auxiliary_data_pipeline",
    }
)


def _is_governed_incremental_sync(row: Any) -> bool:
    payload = row.payload_json
    if not isinstance(payload, dict):
        return False
    lookback_days = payload.get("lookback_days")
    return (
        row.timezone == "Asia/Shanghai"
        and bool(row.trading_days_only)
        and payload.get("profile") == "full"
        and payload.get("snapshot_start") == "2008-01-01"
        and payload.get("build_qlib") is True
        and isinstance(lookback_days, int)
        and not isinstance(lookback_days, bool)
        and 1 <= lookback_days <= 30
    )


def _is_governed_full_data_pipeline(row: Any) -> bool:
    payload = row.payload_json
    if not isinstance(payload, dict):
        return False
    bundles = payload.get("bundles")
    lookback_days = payload.get("lookback_days", 7)
    return (
        row.timezone == "Asia/Shanghai"
        and bool(row.trading_days_only)
        and payload.get("profile") == "full"
        and payload.get("snapshot_start") == "2008-01-01"
        and isinstance(lookback_days, int)
        and not isinstance(lookback_days, bool)
        and 1 <= lookback_days <= 90
        and isinstance(bundles, list)
        and len(bundles) == len(_GOVERNED_DATA_PIPELINE_BUNDLES)
        and all(isinstance(bundle, str) for bundle in bundles)
        and set(bundles) == _GOVERNED_DATA_PIPELINE_BUNDLES
    )


def _is_governed_suite_data_pipeline(row: Any) -> bool:
    return _is_governed_full_data_pipeline(row) and row.run_time >= time(15, 10)


def _is_governed_information_pipeline(row: Any) -> bool:
    try:
        payload = normalize_information_schedule_payload(row.payload_json)
    except (TypeError, ValueError):
        return False
    return (
        row.timezone == "Asia/Shanghai"
        and not bool(row.trading_days_only)
        and payload["lookback_days"] == 7
        and payload["regulatory_only"] is True
        and payload["download_limit"] == 0
        and payload["enable_nlp"] is True
        and payload["announcement_categories"] == ["regulatory_letter"]
        and payload["announcement_nlp_limit"] == 500
        and payload["include_corpus_nlp"] is True
        and set(payload["corpus_datasets"])
        == _GOVERNED_INFORMATION_CORPUS_DATASETS
        and payload["corpus_nlp_limit"] == 500
        and payload["batch_size"] == 50
        and payload["major_news_per_day"] == 40
        and payload["irm_per_instrument_day"] == 2
        and payload["include_event_labels"] is True
        and payload["include_factor_evaluation"] is False
        and payload["factor_evaluation"] is None
        and payload["snapshot_name"] == ""
        and payload["horizons"] == [1, 3, 5, 20]
        and payload["benchmark_code"] == "000300.SH"
    )


def _is_governed_information_factor_refresh(
    row: Any,
    *,
    data_root: Path,
    reproducible_dataset_names: set[str],
) -> bool:
    try:
        payload = normalize_information_factor_refresh_payload(row.payload_json)
    except (TypeError, ValueError):
        return False
    evaluation = payload["factor_evaluation"] or {}
    try:
        resolve_information_evaluation_dataset(data_root, evaluation)
    except (OSError, TypeError, ValueError):
        return False
    return (
        row.timezone == "Asia/Shanghai"
        and not bool(row.trading_days_only)
        and set(payload["sources"]) == STRUCTURED_INFORMATION_SOURCES
        and payload["weekday"] == 4
        and evaluation.get("dataset") in reproducible_dataset_names
        and evaluation.get("universe") == "cn_all"
        and evaluation.get("benchmark") == "SH000300"
    )


def _is_governed_ashare_5m_sync(row: Any) -> bool:
    payload = row.payload_json
    if not isinstance(payload, dict):
        return False
    try:
        date.fromisoformat(str(payload.get("history_start") or ""))
    except ValueError:
        return False
    lookback_days = payload.get("lookback_days")
    return (
        row.timezone == "Asia/Shanghai"
        and row.run_time >= time(15, 10)
        and bool(row.trading_days_only)
        and payload.get("daily_dataset") in (None, "")
        and isinstance(lookback_days, int)
        and not isinstance(lookback_days, bool)
        and 1 <= lookback_days <= 30
        and set(payload) <= {"history_start", "daily_dataset", "lookback_days"}
    )


def _is_governed_auxiliary_data_pipeline(row: Any) -> bool:
    payload = row.payload_json
    if not isinstance(payload, dict):
        return False
    try:
        date.fromisoformat(str(payload.get("history_start") or ""))
    except ValueError:
        return False
    return (
        row.timezone == "Asia/Shanghai"
        and not bool(row.trading_days_only)
        and isinstance(payload.get("max_stocks"), int)
        and not isinstance(payload.get("max_stocks"), bool)
        and 1 <= payload["max_stocks"] <= 500
        and isinstance(payload.get("max_options"), int)
        and not isinstance(payload.get("max_options"), bool)
        and 1 <= payload["max_options"] <= 500
        and isinstance(payload.get("strategy_minute_symbols"), list)
        and bool(payload["strategy_minute_symbols"])
    )


def _governed_schedule_suite_state(
    rows: list[Any],
    *,
    data_root: Path,
    reproducible_dataset_names: set[str],
) -> tuple[bool, dict[str, list[str]]]:
    by_kind: dict[str, list[Any]] = {}
    for row in rows:
        by_kind.setdefault(str(row.kind), []).append(row)
    governed: dict[str, list[str]] = {
        "data_pipeline": [],
        "information_pipeline": [],
        "information_factor_refresh": [],
        "ashare_5m_sync": [],
        "auxiliary_data_pipeline": [],
    }
    for row in rows:
        accepted = False
        if row.kind == "data_pipeline":
            accepted = _is_governed_suite_data_pipeline(row)
        elif row.kind == "information_pipeline":
            accepted = _is_governed_information_pipeline(row)
        elif row.kind == "information_factor_refresh":
            accepted = _is_governed_information_factor_refresh(
                row,
                data_root=data_root,
                reproducible_dataset_names=reproducible_dataset_names,
            )
        elif row.kind == "ashare_5m_sync":
            accepted = _is_governed_ashare_5m_sync(row)
        elif row.kind == "auxiliary_data_pipeline":
            accepted = _is_governed_auxiliary_data_pipeline(row)
        if accepted:
            governed[row.kind].append(str(row.id))
    ready = (
        len(rows) == 4
        and set(by_kind) == _GOVERNED_DATA_SCHEDULE_SUITE_KINDS
        and all(len(by_kind[kind]) == 1 for kind in _GOVERNED_DATA_SCHEDULE_SUITE_KINDS)
        and all(len(governed[kind]) == 1 for kind in _GOVERNED_DATA_SCHEDULE_SUITE_KINDS)
    )
    return ready, governed


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
        # DataTaskStore.list() reconciles the operational projection and groups
        # the durable work-unit ledger.  Build it once per assessment so the
        # research and pair profiles cannot repeat that expensive work.
        tasks = {
            str(item["task_key"]): item
            for item in self.data_tasks.list()
        }
        research_checks = self._research_checks(tasks)
        recommendation_checks = [*research_checks, *self._recommendation_checks()]
        allocation_checks = [*recommendation_checks, *self._allocation_checks(current)]
        pair_checks = [*research_checks, *self._pair_research_checks(tasks)]
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
            "policy_version": "2026-08-24.1",
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
            simulation_count = int(
                connection.scalar(select(func.count()).select_from(simulation_portfolios)) or 0
            )
            certified_nav_count = int(
                connection.scalar(
                    select(func.count()).select_from(simulation_nav).where(
                        simulation_nav.c.performance_certified.is_(True)
                    )
                )
                or 0
            )
            degraded_nav_count = int(
                connection.scalar(
                    select(func.count()).select_from(simulation_nav).where(
                        simulation_nav.c.status == "degraded"
                    )
                )
                or 0
            )
            nav_history = connection.execute(
                select(
                    simulation_nav.c.portfolio_id,
                    simulation_nav.c.status,
                    simulation_nav.c.performance_certified,
                )
            ).all()
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
        nav_by_portfolio: dict[str, list[Any]] = {}
        for row in nav_history:
            nav_by_portfolio.setdefault(str(row.portfolio_id), []).append(row)
        replay_ready = any(
            len(rows) >= 60
            and all(
                bool(row.performance_certified) and row.status == "healthy"
                for row in rows
            )
            for rows in nav_by_portfolio.values()
        )
        maximum_replay_days = max(
            (len(rows) for rows in nav_by_portfolio.values()), default=0
        )
        return [
            _check(
                "simulation_accounts",
                "Recommendation targets are bound to simulation accounts",
                simulation_count > 0,
                f"transactional simulation accounts: {simulation_count}",
                "Create and activate a 5-minute simulation account for a recommendation target",
            ),
            _check(
                "certified_simulation_nav",
                "Simulation NAV is certifiable",
                certified_nav_count > 0 and degraded_nav_count == 0,
                (
                    f"certified NAV rows: {certified_nav_count}; "
                    f"degraded NAV rows: {degraded_nav_count}"
                ),
                "Run simulation booking and resolve stale or missing valuations",
            ),
            _check(
                "simulation_60_day_replay",
                "A simulation account has at least 60 certified trading days",
                replay_ready,
                f"maximum simulation history: {maximum_replay_days} days",
                "Replay at least 60 trading days with no degraded or uncertified NAV rows",
            ),
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

    def _pair_research_checks(
        self,
        tasks: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
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

    def _research_checks(
        self,
        tasks: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
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
        # Readiness is an operational display, not an admission or capital
        # boundary.  Use the persisted bounded projection here; every formal
        # research/backtest/simulation action still performs strict per-dataset
        # provenance and sealed-output verification before it may proceed.
        reproducible_datasets = [
            item
            for item in list_qlib_datasets_for_display(self.settings.data_root)
            if item["ready"]
            and item.get("reproducible")
            and item.get("lineage_verified")
        ]
        datasets = [
            item
            for item in reproducible_datasets
            if int(item["trading_days"]) >= RESEARCH_MINIMUM_TRADING_DAYS
        ]
        maximum_trading_days = max(
            (int(item["trading_days"]) for item in reproducible_datasets),
            default=0,
        )
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
            active_data_schedules = connection.execute(
                select(
                    schedules.c.id,
                    schedules.c.kind,
                    schedules.c.timezone,
                    schedules.c.run_time,
                    schedules.c.trading_days_only,
                    schedules.c.payload_json,
                ).where(
                    schedules.c.kind.in_(
                        ("incremental_sync", *_GOVERNED_DATA_SCHEDULE_SUITE_KINDS)
                    ),
                    schedules.c.status == "active",
                )
            ).all()
            critical_alerts = int(
                connection.scalar(
                    select(func.count())
                    .select_from(alerts)
                    .where(alerts.c.severity == "critical", alerts.c.status == "open")
                )
                or 0
            )
        governed_incremental_ids = [
            str(row.id)
            for row in active_data_schedules
            if row.kind == "incremental_sync" and _is_governed_incremental_sync(row)
        ]
        rejected_incremental_ids = [
            str(row.id)
            for row in active_data_schedules
            if row.kind == "incremental_sync" and not _is_governed_incremental_sync(row)
        ]
        governed_pipeline_ids = [
            str(row.id)
            for row in active_data_schedules
            if row.kind == "data_pipeline" and _is_governed_full_data_pipeline(row)
        ]
        rejected_pipeline_ids = [
            str(row.id)
            for row in active_data_schedules
            if row.kind == "data_pipeline" and not _is_governed_full_data_pipeline(row)
        ]
        legacy_schedule_ready = (
            len(active_data_schedules) == 1
            and len(governed_incremental_ids) + len(governed_pipeline_ids) == 1
        )
        suite_ready, governed_suite_ids = _governed_schedule_suite_state(
            active_data_schedules,
            data_root=self.settings.data_root,
            reproducible_dataset_names={str(item["name"]) for item in datasets},
        )
        data_schedule_ready = legacy_schedule_ready or suite_ready
        schedule_mode = (
            "governed_suite_v1"
            if suite_ready
            else "legacy_single"
            if legacy_schedule_ready
            else "invalid"
        )
        governed_suite_id_set = {
            schedule_id
            for ids in governed_suite_ids.values()
            for schedule_id in ids
        }
        rejected_suite_ids = [
            str(row.id)
            for row in active_data_schedules
            if row.kind in _GOVERNED_DATA_SCHEDULE_SUITE_KINDS
            and str(row.id) not in governed_suite_id_set
        ]
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
                (
                    f"满足至少 {RESEARCH_MINIMUM_TRADING_DAYS} 个交易日及血缘要求的"
                    f"数据集 {len(datasets)} 个；当前最长 {maximum_trading_days} 个交易日"
                ),
                (
                    "构建包含至少 "
                    f"{RESEARCH_MINIMUM_TRADING_DAYS} 个交易日的数据集："
                    f"训练 {MINIMUM_PROFILE_TRAINING_DAYS} + 最长验证 "
                    f"{RESEARCH_LONGEST_VALIDATION_TRADING_DAYS}"
                    f" + 隔离 {DEFAULT_RESEARCH_PERIOD_POLICY['embargo_trading_days']}"
                    f" + 最终测试 {DEFAULT_RESEARCH_PERIOD_POLICY['test_trading_days']}"
                ),
                details={
                    "minimum_trading_days": RESEARCH_MINIMUM_TRADING_DAYS,
                    "maximum_available_trading_days": maximum_trading_days,
                    "eligible_dataset_count": len(datasets),
                },
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
                "受治理的数据更新调度已启用",
                data_schedule_ready,
                (
                    f"活动数据更新调度 {len(active_data_schedules)} 个；"
                    f"模式 {schedule_mode}；"
                    f"合格 incremental_sync {len(governed_incremental_ids)} 个；"
                    f"不合格 incremental_sync {len(rejected_incremental_ids)} 个；"
                    f"合格 full data_pipeline {len(governed_pipeline_ids)} 个；"
                    f"不合格 data_pipeline {len(rejected_pipeline_ids)} 个；"
                    f"合格五计划组件 {sum(len(ids) for ids in governed_suite_ids.values())} 个"
                ),
                (
                    "保留一个兼容的受治理 legacy 数据计划，或精确启用五计划套件："
                    "18:00 full data_pipeline、23:30 ashare_5m_sync、02:00 bounded "
                    "information_pipeline、周五 12:30 information_factor_refresh、"
                    "04:00 auxiliary_data_pipeline"
                ),
                details={
                    "mode": schedule_mode,
                    "active_schedule_ids": [str(row.id) for row in active_data_schedules],
                    "incremental_sync_ids": governed_incremental_ids,
                    "rejected_incremental_sync_ids": rejected_incremental_ids,
                    "governed_data_pipeline_ids": governed_pipeline_ids,
                    "rejected_data_pipeline_ids": rejected_pipeline_ids,
                    "governed_suite_ids": governed_suite_ids,
                    "rejected_suite_ids": rejected_suite_ids,
                    "required_suite_kinds": sorted(_GOVERNED_DATA_SCHEDULE_SUITE_KINDS),
                    "required_bundles": sorted(_GOVERNED_DATA_PIPELINE_BUNDLES),
                },
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
