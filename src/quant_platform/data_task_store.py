from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import case, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from quant_data.database import data_tasks, jobs, open_database, row_dict, work_units


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class DataTaskDefinition:
    task_key: str
    phase: int
    sort_order: int
    title: str
    description: str
    category: str
    source: str
    implementation_status: str
    depends_on: tuple[str, ...]
    datasets: tuple[str, ...]
    frequency: str
    range_start: str = "2024-01-01"
    range_end: str = "latest"
    estimated_storage_gb: int | None = None


DATA_TASK_CATALOG: tuple[DataTaskDefinition, ...] = (
    DataTaskDefinition(
        "cn_ashare_daily_full",
        1,
        10,
        "A股日频全量初始化",
        "下载行情、复权、估值、财务、公司事件、资金行为、公告与新闻；任务状态以右上角结果为准。",
        "A股",
        "Tushare",
        "ready",
        (),
        ("daily", "adj_factor", "daily_basic", "fundamentals", "events", "news"),
        "daily",
        estimated_storage_gb=25,
    ),
    DataTaskDefinition(
        "cn_data_verify",
        1,
        11,
        "初始化数据质量校验",
        "校验所有下载单元、文件哈希、空结果策略、核心主键重复和终态失败。",
        "数据治理",
        "QuantLab",
        "ready",
        ("cn_ashare_daily_full",),
        ("verification",),
        "on-demand",
    ),
    DataTaskDefinition(
        "cn_snapshot_build",
        1,
        12,
        "不可变 Parquet 快照",
        "合并成功工作单元并生成带源分片和文件 SHA-256 的内容清单。",
        "数据治理",
        "QuantLab",
        "ready",
        ("cn_data_verify",),
        ("snapshot",),
        "on-demand",
        estimated_storage_gb=30,
    ),
    DataTaskDefinition(
        "cn_qlib_build",
        1,
        13,
        "正式 Qlib 数据集构建",
        "从已校验快照生成日历、股票池、二进制特征和点时行业/风格元数据。",
        "研究基础设施",
        "Qlib",
        "ready",
        ("cn_snapshot_build",),
        ("qlib_dataset",),
        "on-demand",
        estimated_storage_gb=20,
    ),
    DataTaskDefinition(
        "cn_qlib_baseline",
        1,
        14,
        "Qlib Alpha158 基线验收",
        "运行带成本、涨跌停、停牌和容量约束的原生 Qlib 基线并保存可复现证据。",
        "研究基础设施",
        "Qlib",
        "ready",
        ("cn_qlib_build",),
        ("alpha158_baseline",),
        "on-demand",
    ),
    DataTaskDefinition(
        "cn_extended_daily",
        2,
        20,
        "A股扩展与特色数据",
        "补充历史ST状态、申万行业行情与成分、股东、机构调研、大宗交易、北向持股和专业因子。",
        "A股",
        "Tushare",
        "ready",
        ("cn_qlib_baseline",),
        (
            "stk_holdernumber",
            "top10_holders",
            "top10_floatholders",
            "stk_surv",
            "block_trade",
            "hk_hold",
            "stk_factor_pro",
            "fina_audit",
            "fina_mainbz",
            "new_share",
            "stock_st",
            "sw_daily",
        ),
        "daily",
        estimated_storage_gb=30,
    ),
    DataTaskDefinition(
        "cn_funds",
        2,
        22,
        "公募基金研究数据",
        "ETF专用基础与跟踪指数、基金公司、基金经理、净值、份额、分红和季度持仓。",
        "基金",
        "Tushare",
        "ready",
        ("cn_extended_daily",),
        (
            "fund_basic",
            "fund_company",
            "fund_manager",
            "fund_nav",
            "fund_share",
            "fund_div",
            "fund_portfolio",
            "etf_basic",
            "etf_index",
        ),
        "daily/quarterly",
        estimated_storage_gb=20,
    ),
    DataTaskDefinition(
        "cn_margin_eligibility",
        2,
        25,
        "融资融券标的资格历史",
        "保存逐交易日、逐证券的可融券资格证据；配对交易禁止从成交明细反推资格。",
        "A股",
        "Tushare/QMT",
        "ready",
        ("cn_funds",),
        ("margin_eligibility",),
        "daily",
        estimated_storage_gb=1,
    ),
    DataTaskDefinition(
        "pair_execution_1m",
        2,
        26,
        "配对策略执行分钟线",
        "按显式 ETF/股票配对池下载1分钟线，并与同区间融券资格合成不可变执行快照。",
        "A股",
        "Tushare",
        "ready",
        ("cn_margin_eligibility",),
        ("etf_1m", "liquid_stocks_1m", "margin_eligibility"),
        "1min",
        estimated_storage_gb=20,
    ),
    DataTaskDefinition(
        "cn_macro",
        3,
        30,
        "中国宏观经济全量",
        "GDP、CPI、PPI、PMI、货币供应、社融、利率及可用于点时回测的数据发布日期。",
        "宏观",
        "Tushare",
        "ready",
        ("cn_funds",),
        (
            "cn_gdp",
            "cn_cpi",
            "cn_ppi",
            "cn_pmi",
            "cn_m",
            "sf_month",
            "shibor",
            "shibor_lpr",
            "cn_schedule",
        ),
        "daily/monthly",
        estimated_storage_gb=2,
    ),
    DataTaskDefinition(
        "cn_futures",
        4,
        40,
        "期货市场",
        "合约、日线、主力连续、持仓、仓单、结算及每日涨跌停与最低保证金率。",
        "跨资产",
        "Tushare",
        "ready",
        ("cn_macro",),
        (
            "fut_basic",
            "fut_trade_cal",
            "fut_mapping",
            "fut_daily",
            "fut_holding",
            "fut_wsr",
            "fut_settle",
            "ft_limit",
        ),
        "daily",
        estimated_storage_gb=20,
    ),
    DataTaskDefinition(
        "cn_options_bonds",
        4,
        50,
        "期权与债券",
        "期权合约和日线；可转债发行、回购、赎回、票息、转股和收益率曲线。",
        "跨资产",
        "Tushare",
        "ready",
        ("cn_futures",),
        (
            "opt_basic",
            "opt_daily",
            "cb_basic",
            "cb_issue",
            "cb_redeem",
            "cb_rate",
            "cb_price_chg",
            "cb_share",
            "cb_rating",
            "top10_cb_holders",
            "cb_daily",
            "repo_daily",
            "yc_cb",
        ),
        "daily",
        estimated_storage_gb=20,
    ),
    DataTaskDefinition(
        "hk_market",
        5,
        60,
        "港股全市场",
        "港股基础信息、交易日历、日线、复权、三大报表和财务指标。",
        "外围市场",
        "Tushare",
        "ready",
        ("cn_options_bonds",),
        (
            "hk_basic",
            "hk_tradecal",
            "hk_daily",
            "hk_daily_adj",
            "hk_income",
            "hk_balancesheet",
            "hk_cashflow",
            "hk_fina_indicator",
        ),
        "daily",
        estimated_storage_gb=35,
    ),
    DataTaskDefinition(
        "us_market",
        5,
        70,
        "美股核心市场",
        "美股基础信息、交易日历、日线、复权、三大报表和财务指标。",
        "外围市场",
        "Tushare",
        "ready",
        ("hk_market",),
        (
            "us_basic",
            "us_tradecal",
            "us_daily",
            "us_daily_adj",
            "us_income",
            "us_balancesheet",
            "us_cashflow",
            "us_fina_indicator",
        ),
        "daily",
        estimated_storage_gb=60,
    ),
    DataTaskDefinition(
        "global_markets",
        5,
        80,
        "全球指数与外汇信号",
        "国际主要指数、外汇日线及美国国债收益率曲线，支持跨市场风险与估值。",
        "外围市场",
        "Tushare",
        "ready",
        ("us_market",),
        ("index_global", "fx_obasic", "fx_daily", "us_tycr"),
        "daily",
        estimated_storage_gb=10,
    ),
    DataTaskDefinition(
        "liquid_intraday_1m",
        6,
        90,
        "核心资产1分钟线",
        "主要指数、ETF、股指期货、商品主力、活跃期权及重点股票池。",
        "分钟行情",
        "Tushare",
        "ready",
        ("cn_options_bonds", "cn_margin_eligibility"),
        ("indices_1m", "etf_1m", "futures_1m", "options_1m", "liquid_stocks_1m"),
        "1min",
        estimated_storage_gb=180,
    ),
    DataTaskDefinition(
        "liquid_intraday_qlib",
        6,
        95,
        "分钟 Qlib 研究数据集",
        "将已完成的核心资产1分钟快照独立转换为 Qlib 二进制数据，下载与构建可分别重试。",
        "研究基础设施",
        "Qlib",
        "ready",
        ("liquid_intraday_1m",),
        ("qlib_minute_dataset",),
        "on-demand",
        estimated_storage_gb=120,
    ),
    DataTaskDefinition(
        "cn_ashare_5m",
        6,
        100,
        "全A股5分钟线",
        "2024年至今全市场5分钟行情；15/30/60分钟由5分钟数据本地聚合。",
        "分钟行情",
        "Tushare",
        "permission_probe",
        ("liquid_intraday_1m",),
        ("ashare_5m",),
        "5min",
        estimated_storage_gb=250,
    ),
    DataTaskDefinition(
        "tick_level2",
        7,
        110,
        "Tick、逐笔与Level-2",
        "仅在获得正式授权数据源和扩容后启用，不占用当前1TB数据盘。",
        "高频",
        "交易所授权数据源",
        "external_source_required",
        ("cn_ashare_5m",),
        ("tick", "transactions", "level2_orderbook"),
        "tick",
        estimated_storage_gb=None,
    ),
)

SUPPLEMENTAL_TASK_KEYS = frozenset(
    {
        "cn_extended_daily",
        "cn_funds",
        "cn_macro",
        "cn_futures",
        "cn_options_bonds",
        "hk_market",
        "us_market",
        "global_markets",
    }
)


class DataTaskStore:
    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def sync_catalog(self) -> None:
        now = _now()
        with self.engine.begin() as connection:
            for definition in DATA_TASK_CATALOG:
                raw = asdict(definition)
                values = {
                    "task_key": definition.task_key,
                    "phase": definition.phase,
                    "sort_order": definition.sort_order,
                    "title": definition.title,
                    "description": definition.description,
                    "category": definition.category,
                    "source": definition.source,
                    "status": "planned",
                    "implementation_status": definition.implementation_status,
                    "depends_on_json": list(definition.depends_on),
                    "config_json": {
                        "datasets": list(definition.datasets),
                        "frequency": definition.frequency,
                        "range_start": definition.range_start,
                        "range_end": definition.range_end,
                    },
                    "estimated_storage_gb": raw["estimated_storage_gb"],
                    "created_at": now,
                    "updated_at": now,
                }
                statement = pg_insert(data_tasks).values(**values)
                connection.execute(
                    statement.on_conflict_do_update(
                        index_elements=[data_tasks.c.task_key],
                        set_={
                            key: value
                            for key, value in values.items()
                            if key not in {"task_key", "status", "created_at"}
                        },
                    )
                )
            self._bind_pipeline_jobs(connection, now)

    @staticmethod
    def _bind_pipeline_jobs(connection, now: datetime) -> None:
        task_by_kind = {
            "bootstrap": "cn_ashare_daily_full",
            "data_verify": "cn_data_verify",
            "data_snapshot": "cn_snapshot_build",
            "data_qlib": "cn_qlib_build",
            "qlib_baseline": "cn_qlib_baseline",
            "margin_eligibility_download": "cn_margin_eligibility",
            "core_intraday_download": "pair_execution_1m",
            "minute_qlib": "liquid_intraday_qlib",
        }
        rows = connection.execute(
            select(jobs.c.id, jobs.c.kind, jobs.c.status, jobs.c.created_at)
            .where(jobs.c.kind.in_(tuple(task_by_kind)))
            .order_by(jobs.c.created_at.desc())
        ).all()
        latest: dict[str, Any] = {}
        for row in rows:
            latest.setdefault(str(row.kind), row)
        for kind, task_key in task_by_kind.items():
            current = latest.get(kind)
            if current is None:
                continue
            connection.execute(
                update(data_tasks)
                .where(data_tasks.c.task_key == task_key)
                .values(job_id=current.id, status=current.status, updated_at=now)
            )
        current_intraday = latest.get("core_intraday_download")
        if current_intraday is not None:
            connection.execute(
                update(data_tasks)
                .where(data_tasks.c.task_key == "liquid_intraday_1m")
                .values(
                    job_id=current_intraday.id,
                    status=current_intraday.status,
                    updated_at=now,
                )
            )

        supplemental_rows = connection.execute(
            select(
                jobs.c.id,
                jobs.c.status,
                jobs.c.payload_json,
                jobs.c.created_at,
            )
            .where(jobs.c.kind.like("supplemental_%"))
            .order_by(jobs.c.created_at.desc())
        ).all()
        latest_supplemental: dict[str, Any] = {}
        for row in supplemental_rows:
            bundle = str((row.payload_json or {}).get("bundle") or "")
            if bundle in SUPPLEMENTAL_TASK_KEYS:
                latest_supplemental.setdefault(bundle, row)
        for task_key, current in latest_supplemental.items():
            connection.execute(
                update(data_tasks)
                .where(data_tasks.c.task_key == task_key)
                .values(job_id=current.id, status=current.status, updated_at=now)
            )

        # A legacy bootstrap used to own download, snapshot and Qlib conversion in
        # one job.  Its final status can therefore be failed even though every
        # download completed and a newer durable stage has already succeeded.
        # Project current pipeline truth onto prerequisite cards while preserving
        # the failed job itself in the immutable job history.
        stages = (
            "bootstrap",
            "data_verify",
            "data_snapshot",
            "data_qlib",
            "qlib_baseline",
        )
        for index, kind in enumerate(stages[:-1]):
            current = latest.get(kind)
            if current is not None and str(current.status) in {"queued", "running", "succeeded"}:
                continue
            downstream = next(
                (
                    latest[candidate]
                    for candidate in stages[index + 1 :]
                    if candidate in latest
                    and str(latest[candidate].status) in {"queued", "running", "succeeded"}
                    and (current is None or latest[candidate].created_at >= current.created_at)
                ),
                None,
            )
            if downstream is None:
                continue
            connection.execute(
                update(data_tasks)
                .where(data_tasks.c.task_key == task_by_kind[kind])
                .values(job_id=None, status="succeeded", updated_at=now)
            )

    def list(self) -> list[dict[str, Any]]:
        with self.engine.begin() as connection:
            self._bind_pipeline_jobs(connection, _now())
            rows = [
                row_dict(row)
                for row in connection.execute(
                    select(data_tasks).order_by(data_tasks.c.phase, data_tasks.c.sort_order)
                )
            ]
            job_ids = [row["job_id"] for row in rows if row.get("job_id")]
            job_states = (
                {
                    str(row.id): {
                        "status": str(row.status),
                        "error": row.error,
                        "progress": row.progress_json,
                    }
                    for row in connection.execute(
                        select(
                            jobs.c.id,
                            jobs.c.status,
                            jobs.c.error,
                            jobs.c.progress_json,
                        ).where(jobs.c.id.in_(job_ids))
                    )
                }
                if job_ids
                else {}
            )
            unit_counts = {
                str(row.dataset): {
                    "planned": int(row.planned or 0),
                    "succeeded": int(row.succeeded or 0),
                    "rows": int(row.rows or 0),
                }
                for row in connection.execute(
                    select(
                        work_units.c.dataset,
                        func.count().label("planned"),
                        func.sum(case((work_units.c.status == "succeeded", 1), else_=0)).label(
                            "succeeded"
                        ),
                        func.sum(work_units.c.row_count).label("rows"),
                    ).group_by(work_units.c.dataset)
                )
            }
            for row in rows:
                if row.get("job_id") and str(row["job_id"]) in job_states:
                    state = job_states[str(row["job_id"])]
                    row["status"] = state["status"]
                    row["error"] = state["error"]
                    row["progress"] = state["progress"]
        for row in rows:
            dependencies = row.pop("depends_on_json")
            row["depends_on"] = dependencies
            row["config"] = row.pop("config_json")
            counts = [
                unit_counts[name] for name in row["config"]["datasets"] if name in unit_counts
            ]
            if str(row["task_key"]) in SUPPLEMENTAL_TASK_KEYS:
                progress = row.get("progress") or {}
                completed_datasets = set((progress.get("datasets") or {}).keys())
                expected_datasets = set(row["config"]["datasets"])
                authoritative_success = (
                    row["status"] == "succeeded"
                    and progress.get("status") == "succeeded"
                    and progress.get("pagination_verified") is True
                    and expected_datasets <= completed_datasets
                )
                if authoritative_success:
                    # The successful job result describes the exact current
                    # plan and proves pagination termination. Historical work
                    # units from superseded request shapes remain available
                    # for audit, but must not lower current readiness.
                    row["coverage"] = 100.0
                else:
                    per_dataset = [
                        (
                            unit_counts[name]["succeeded"] / unit_counts[name]["planned"]
                            if name in unit_counts and unit_counts[name]["planned"]
                            else 0.0
                        )
                        for name in row["config"]["datasets"]
                    ]
                    row["coverage"] = (
                        round(sum(per_dataset) / len(per_dataset) * 100, 1)
                        if per_dataset
                        else 0.0
                    )
                    if row["status"] == "succeeded" and row["coverage"] < 100:
                        row["status"] = "partial"
            else:
                planned = sum(item["planned"] for item in counts)
                succeeded = sum(item["succeeded"] for item in counts)
                row["coverage"] = (
                    100.0
                    if row["status"] == "succeeded"
                    else round(succeeded / planned * 100, 1)
                    if planned
                    else 0.0
                )
            row["rows"] = sum(item["rows"] for item in counts)
        status_by_key = {str(row["task_key"]): str(row["status"]) for row in rows}
        for row in rows:
            dependencies = row["depends_on"]
            row["dependencies_satisfied"] = all(
                status_by_key.get(key) == "succeeded" for key in dependencies
            )
        return rows
