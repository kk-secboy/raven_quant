from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from quant_data.coverage_data import COVERAGE_BUNDLES, coverage_bundle_datasets
from quant_data.database import data_tasks, jobs, open_database, row_dict, work_units


def _now() -> datetime:
    return datetime.now(UTC)


DEFAULT_REQUEST_STRATEGY = (
    "按接口密度批量规划；成功 checkpoint 会复用，旧的未完成请求保留审计并由新计划替代。"
)


REQUEST_STRATEGIES = {
    "cn_ashare_daily_full": "尽量按完整日期窗口批量下载，分页后验证终止页。",
    "cn_extended_daily": "普通日期区间批量分页；offset 或满页触顶时自动二分日期。",
    "cn_funds": "基金份额按月，净值按日分页；分红和持仓按日历日避免漏掉周末公告。",
    "cn_institutional": (
        "ETF 申赎清单按每只 ETF 整段历史获取；分页或行数触顶后自动二分日期。"
    ),
    "cn_capital_flow": "稀疏资金流按月或按年，密集全市场个股资金流继续逐日分页。",
    "research_corpus": "财经新闻按来源和整日请求，达到 1500 条时按时间中点递归拆分。",
    "hk_market": "先下载港股 master 和交易日历，再只规划开市日；自动股票池按上市区间过滤。",
    "us_market": "先下载美股 master 和交易日历，再只规划开市日；财务指标按有效股票池获取。",
    "liquid_intraday_1m": "1 分钟数据保持安全窗口，分页触顶后进入可恢复的自适应分区。",
    "pair_execution_1m": "1 分钟数据保持安全窗口，分页触顶后进入可恢复的自适应分区。",
    "cn_ashare_5m": (
        "每只股票最多 150 个实际交易日一个初始窗口；达到 8000 行时无重叠二分。"
    ),
    "cn_cninfo_announcements": (
        "以已落盘 anns_d 为清单，仅下载标题命中问询函、关注函、监管函、警示函或纪律处分的"
        "高信号巨潮 PDF；全量公告正文因规模超出生产存储边界而不伪装为已覆盖。"
    ),
    "cn_announcement_nlp": (
        "对公告 PDF 做文本抽取后调用 OpenAI 兼容端点做严格 JSON 抽取；"
        "以 sha256+prompt_version+model 为处理键幂等，失败行记录后可重跑。"
    ),
    "cn_corpus_nlp": (
        "对 major_news 长篇新闻与沪深互动易问答调用 OpenAI 兼容端点做严格 JSON 抽取；"
        "以内容 sha256+prompt_version+model 为处理键幂等，失败行记录后可重跑。"
    ),
    "cn_event_market_response": (
        "以已校验不可变日线快照计算公告后1/3/5/20交易日超额收益和方向一致性；"
        "结果只作为训练标签，禁止注册为因子或进入实时推理特征。"
    ),
}


_RATE_LIMIT_PATTERNS = (
    "%rate limit%",
    "%too many request%",
    "%frequency limit%",
    "%限频%",
    "%请求过于频繁%",
    "%请求次数超限%",
    "%每分钟%",
    "%冷却%",
)


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
        "Tushare",
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
        "cn_institutional",
        3,
        35,
        "机构研究与增强数据",
        "券商盈利预测、沪深ETF申赎清单、中信行业指数、Shibor报价明细和长篇财经新闻。ETF清单按标的整段获取，所有限额接口均分页并在触顶时自动二分日期。",
        "研究增强",
        "Tushare",
        "ready",
        ("cn_macro",),
        (
            "report_rc",
            "etf_basic",
            "etf_sh_cons",
            "etf_sz_cons",
            "ci_daily",
            "shibor_quote",
            "major_news",
        ),
        "daily/monthly",
        estimated_storage_gb=25,
    ),
    DataTaskDefinition(
        "cn_governance_risk",
        3,
        36,
        "公司治理、风险与参考资料",
        "ST变更、沪深港通标的、公司与管理层、异常波动、筹码、CCASS、券商推荐、两融与转融通。",
        "A股增强",
        "Tushare",
        "ready",
        ("cn_qlib_baseline",),
        tuple(sorted(coverage_bundle_datasets("cn_governance_risk"))),
        "daily/monthly",
        estimated_storage_gb=35,
    ),
    DataTaskDefinition(
        "cn_capital_flow",
        3,
        37,
        "全市场资金流增强",
        "沪深港通、同花顺与东方财富的个股、行业、概念和全市场资金流。",
        "资金流",
        "Tushare",
        "ready",
        ("cn_qlib_baseline",),
        tuple(sorted(coverage_bundle_datasets("cn_capital_flow"))),
        "daily",
        estimated_storage_gb=20,
    ),
    DataTaskDefinition(
        "cn_fund_index_enhanced",
        3,
        38,
        "ETF、指数与基金增强",
        "ETF份额、指数公告与成分、指数技术因子、交易所统计和基金技术因子。",
        "基金与指数",
        "Tushare",
        "ready",
        ("cn_funds",),
        tuple(sorted(coverage_bundle_datasets("cn_fund_index_enhanced"))),
        "daily",
        estimated_storage_gb=30,
    ),
    DataTaskDefinition(
        "cn_derivatives_enhanced",
        4,
        55,
        "衍生品、黄金与债券增强",
        "期货指数、周度统计、黄金、转债因子、银行间报价、债券大宗和财经日历。",
        "跨资产",
        "Tushare",
        "ready",
        ("cn_options_bonds",),
        tuple(sorted(coverage_bundle_datasets("cn_derivatives_enhanced"))),
        "daily/weekly",
        estimated_storage_gb=25,
    ),
    DataTaskDefinition(
        "global_rates_enhanced",
        5,
        85,
        "海外复权与利率增强",
        "港美股复权因子、LIBOR、HIBOR及美国短中长期利率曲线。",
        "外围市场",
        "Tushare",
        "ready",
        ("global_markets",),
        tuple(sorted(coverage_bundle_datasets("global_rates_enhanced"))),
        "daily",
        estimated_storage_gb=12,
    ),
    DataTaskDefinition(
        "research_corpus",
        5,
        86,
        "研究报告与财经语料",
        "新闻联播、研报、货币政策和交易所互动问答；不可用的微信接口仅保留审计记录。",
        "研究语料",
        "Tushare",
        "ready",
        ("cn_qlib_baseline",),
        tuple(sorted(coverage_bundle_datasets("research_corpus"))),
        "daily/monthly",
        estimated_storage_gb=80,
    ),
    DataTaskDefinition(
        "cn_cninfo_announcements",
        5,
        87,
        "巨潮公告正文与监管函件",
        "以 anns_d 公告索引为清单下载巨潮 PDF 正文，按标题识别问询函、关注函、监管函、"
        "警示函和纪律处分；生产任务只落上述高信号监管类正文，并记录全量公告的存储边界。",
        "研究语料",
        "QuantLab",
        "ready",
        ("cn_ashare_daily_full",),
        ("cninfo_announcements",),
        "daily",
        range_start="2016-01-01",
        estimated_storage_gb=320,
    ),
    DataTaskDefinition(
        "cn_announcement_nlp",
        5,
        88,
        "公告 NLP 信号加工",
        "对巨潮公告正文做 PDF 文本抽取与 LLM 结构化抽取（事件类型、语气分数、关键数值），"
        "按 sha256+prompt_version+model 幂等落不可变 parquet 单元与派生索引，"
        "并生成带 sha256 的 PIT 因子值 artifact。",
        "研究语料",
        "QuantLab",
        "ready",
        ("cn_cninfo_announcements",),
        ("announcement_nlp_fields",),
        "daily",
        range_start="2016-01-01",
    ),
    DataTaskDefinition(
        "cn_corpus_nlp",
        5,
        89,
        "文本语料 NLP 信号加工",
        "对已下载的长篇财经新闻（major_news）、政策法规库（npr）、新闻联播（cctv_news）"
        "与沪深互动易问答（irm_qa_sh/irm_qa_sz）"
        "做 LLM 结构化抽取（情感分、主题、置信度），按内容 sha256+prompt_version+model 幂等"
        "落不可变 parquet 单元与派生字段索引，并生成市场级 news_sentiment_daily、"
        "policy_sentiment_daily 与个股级 irm_qa_sentiment_daily 因子 artifact。",
        "研究语料",
        "QuantLab",
        "ready",
        ("cn_ashare_daily_full", "cn_institutional", "research_corpus"),
        ("corpus_nlp_fields",),
        "daily",
        range_start="2018-11-20",
    ),
    DataTaskDefinition(
        "cn_event_market_response",
        5,
        90,
        "公告后市场认可度训练标签",
        "用沪深300基准计算公告可用日后的1/3/5/20交易日超额收益、成交额异常和方向一致性；"
        "每个标签记录结果观察截止日与最早可用日，仅供训练/评估，禁止进入实时特征。",
        "研究语料",
        "QuantLab",
        "ready",
        ("cn_announcement_nlp", "cn_snapshot_build"),
        ("event_market_response_labels",),
        "daily",
        range_start="2016-01-01",
    ),
    DataTaskDefinition(
        "strategy_specialty",
        7,
        108,
        "策略专项与题材数据",
        "九转、AH比较、涨停板、热点榜、同花顺/东财/通达信/开盘啦概念和指数。",
        "策略可选",
        "Tushare",
        "ready",
        ("cn_qlib_baseline",),
        tuple(sorted(coverage_bundle_datasets("strategy_specialty"))),
        "daily",
        estimated_storage_gb=45,
    ),
    DataTaskDefinition(
        "strategy_specialty_minutes",
        7,
        109,
        "申万指数与港股专项分钟线",
        "按显式策略标的下载申万指数和港股5分钟线，避免无边界全市场请求。",
        "策略可选",
        "Tushare",
        "ready",
        ("liquid_intraday_1m",),
        tuple(sorted(coverage_bundle_datasets("strategy_specialty_minutes"))),
        "5min",
        estimated_storage_gb=80,
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
        "2024年至今全市场5分钟行情；每个标的按最多150个实际交易日规划，触及8000行时自动二分，15/30/60分钟由5分钟数据本地聚合。",
        "分钟行情",
        "Tushare",
        "ready",
        ("liquid_intraday_1m",),
        ("ashare_5m",),
        "5min",
        estimated_storage_gb=250,
    ),
    DataTaskDefinition(
        "cn_ashare_5m_qlib",
        6,
        105,
        "全A股5分钟Qlib数据集",
        "将全A股5分钟不可变快照独立转换为Qlib二进制数据；构建失败不会重新下载行情。",
        "研究基础设施",
        "Qlib",
        "ready",
        ("cn_ashare_5m",),
        ("qlib_ashare_5m_dataset",),
        "5min",
        estimated_storage_gb=180,
    ),
)


_FULL_SCOPE_JOB_TASKS = frozenset(
    {
        "cn_cninfo_announcements",
        "cn_announcement_nlp",
        "cn_corpus_nlp",
    }
)
_DATA_TASK_BY_KEY = {definition.task_key: definition for definition in DATA_TASK_CATALOG}
_CORPUS_CATALOG_DATASETS = frozenset(
    {"major_news", "cctv_news", "irm_qa_sh", "irm_qa_sz"}
)


def _payload_values(payload: dict[str, Any], key: str) -> set[str]:
    """Normalize list-like job filters without treating malformed values as full scope."""

    raw = payload.get(key)
    if raw is None or raw == "":
        return set()
    if isinstance(raw, str):
        return {part.strip() for part in raw.split(",") if part.strip()}
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return {"__invalid__"}
    return {str(part).strip() for part in raw if str(part).strip()}


def job_covers_catalog_scope(task_key: str, payload: dict[str, Any] | None) -> bool:
    """Return whether a durable job is evidence for full catalog completion.

    Limited/pilot information jobs are useful operationally, but binding one
    to the catalog task would make the UI claim the whole data capability is
    complete. For historical tasks with a declared source boundary, also
    require the job to start no later than that boundary. The job still remains
    fully auditable in ``quantlab.jobs``; it simply cannot certify full scope.
    """

    if task_key not in _FULL_SCOPE_JOB_TASKS:
        return True
    job_payload = payload or {}
    try:
        limit = int(job_payload.get("limit") or 0)
    except (TypeError, ValueError):
        return False
    if limit > 0:
        return False
    if _payload_values(job_payload, "ts_codes"):
        return False
    if task_key == "cn_announcement_nlp":
        categories = _payload_values(job_payload, "categories")
        # The production announcement catalog is the governed regulatory
        # subset. An empty filter means all downloaded categories (a superset),
        # otherwise the regulatory category must be present explicitly.
        if categories and "regulatory_letter" not in categories:
            return False
    if task_key == "cn_corpus_nlp":
        datasets = _payload_values(job_payload, "datasets")
        # npr has no persisted production rows and is an audited source gap,
        # not a required fake dataset. The four real corpora are mandatory.
        if datasets and not _CORPUS_CATALOG_DATASETS <= datasets:
            return False
    definition = _DATA_TASK_BY_KEY[task_key]
    if definition.range_start is None:
        return True
    raw_start = str(job_payload.get("start") or "").strip()
    if not raw_start:
        return False
    try:
        return date.fromisoformat(raw_start) <= date.fromisoformat(definition.range_start)
    except ValueError:
        return False

SUPPLEMENTAL_TASK_KEYS = (
    frozenset(
        {
            "cn_extended_daily",
            "cn_funds",
            "cn_macro",
            "cn_institutional",
            "cn_futures",
            "cn_options_bonds",
            "hk_market",
            "us_market",
            "global_markets",
        }
    )
    | COVERAGE_BUNDLES
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
                        "request_strategy": REQUEST_STRATEGIES.get(
                            definition.task_key, DEFAULT_REQUEST_STRATEGY
                        ),
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
            "ashare_5m_download": "cn_ashare_5m",
            "cninfo_announcements_download": "cn_cninfo_announcements",
            "announcement_nlp": "cn_announcement_nlp",
            "corpus_nlp": "cn_corpus_nlp",
            "event_market_response": "cn_event_market_response",
        }
        rows = connection.execute(
            select(
                jobs.c.id,
                jobs.c.kind,
                jobs.c.status,
                jobs.c.payload_json,
                jobs.c.created_at,
            )
            .where(jobs.c.kind.in_((*tuple(task_by_kind), "minute_qlib")))
            .order_by(jobs.c.created_at.desc())
        ).all()
        latest: dict[str, Any] = {}
        for row in rows:
            kind = str(row.kind)
            task_key = task_by_kind.get(kind)
            if task_key is None or not job_covers_catalog_scope(
                task_key, row.payload_json or {}
            ):
                continue
            latest.setdefault(kind, row)
        for kind, task_key in task_by_kind.items():
            current = latest.get(kind)
            if current is None:
                continue
            connection.execute(
                update(data_tasks)
                .where(data_tasks.c.task_key == task_key)
                .values(job_id=current.id, status=current.status, updated_at=now)
            )
        minute_task_by_frequency = {
            "1min": "liquid_intraday_qlib",
            "5min": "cn_ashare_5m_qlib",
        }
        bound_frequencies: set[str] = set()
        for row in rows:
            if str(row.kind) != "minute_qlib":
                continue
            payload = row.payload_json or {}
            frequency = str(payload.get("frequency") or "")
            if not frequency:
                output_name = str(payload.get("output_name") or "")
                frequency = next(
                    (item for item in minute_task_by_frequency if output_name.endswith(f"-{item}")),
                    "1min",
                )
            if frequency in bound_frequencies or frequency not in minute_task_by_frequency:
                continue
            bound_frequencies.add(frequency)
            connection.execute(
                update(data_tasks)
                .where(data_tasks.c.task_key == minute_task_by_frequency[frequency])
                .values(job_id=row.id, status=row.status, updated_at=now)
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
                        "id": str(row.id),
                        "kind": str(row.kind),
                        "status": str(row.status),
                        "error": row.error,
                        "progress": row.progress_json,
                        "payload": row.payload_json,
                        "created_at": row.created_at,
                        "started_at": row.started_at,
                        "finished_at": row.finished_at,
                        "next_attempt_at": row.next_attempt_at,
                        "attempts": int(row.attempts or 0),
                        "max_attempts": int(row.max_attempts or 0),
                    }
                    for row in connection.execute(
                        select(
                            jobs.c.id,
                            jobs.c.kind,
                            jobs.c.status,
                            jobs.c.error,
                            jobs.c.progress_json,
                            jobs.c.payload_json,
                            jobs.c.created_at,
                            jobs.c.started_at,
                            jobs.c.finished_at,
                            jobs.c.next_attempt_at,
                            jobs.c.attempts,
                            jobs.c.max_attempts,
                        ).where(jobs.c.id.in_(job_ids))
                    )
                }
                if job_ids
                else {}
            )
            retryable_failure = and_(
                work_units.c.status == "failed",
                work_units.c.attempts < work_units.c.max_attempts,
            )
            terminal_failure = and_(
                work_units.c.status == "failed",
                work_units.c.attempts >= work_units.c.max_attempts,
            )
            rate_limited_failure = and_(
                retryable_failure,
                or_(
                    *[
                        func.lower(func.coalesce(work_units.c.last_error, "")).like(pattern)
                        for pattern in _RATE_LIMIT_PATTERNS
                    ]
                ),
            )
            unit_counts = {
                str(row.dataset): {
                    "planned": int(row.planned or 0) - int(row.superseded or 0),
                    "succeeded": int(row.succeeded or 0),
                    "pending": int(row.pending or 0),
                    "running": int(row.running or 0),
                    "retry_waiting": int(row.retry_waiting or 0),
                    "terminal_failed": int(row.terminal_failed or 0),
                    "rate_limited": int(row.rate_limited or 0),
                    "superseded": int(row.superseded or 0),
                    "rows": int(row.rows or 0),
                    "next_retry_at": row.next_retry_at,
                }
                for row in connection.execute(
                    select(
                        work_units.c.dataset,
                        func.count().label("planned"),
                        func.sum(case((work_units.c.status == "succeeded", 1), else_=0)).label(
                            "succeeded"
                        ),
                        func.sum(
                            case((work_units.c.status == "pending", 1), else_=0)
                        ).label("pending"),
                        func.sum(
                            case((work_units.c.status == "running", 1), else_=0)
                        ).label("running"),
                        func.sum(case((retryable_failure, 1), else_=0)).label(
                            "retry_waiting"
                        ),
                        func.sum(case((terminal_failure, 1), else_=0)).label(
                            "terminal_failed"
                        ),
                        func.sum(case((rate_limited_failure, 1), else_=0)).label(
                            "rate_limited"
                        ),
                        func.sum(
                            case((work_units.c.status == "superseded", 1), else_=0)
                        ).label("superseded"),
                        func.sum(work_units.c.row_count).label("rows"),
                        func.min(
                            case((retryable_failure, work_units.c.next_retry_at), else_=None)
                        ).label("next_retry_at"),
                    ).group_by(work_units.c.dataset)
                )
            }
            for row in rows:
                if row.get("job_id") and str(row["job_id"]) in job_states:
                    state = job_states[str(row["job_id"])]
                    row["status"] = state["status"]
                    row["error"] = state["error"]
                    row["progress"] = state["progress"]
                    row["job"] = state
        for row in rows:
            dependencies = row.pop("depends_on_json")
            row["depends_on"] = dependencies
            row["config"] = row.pop("config_json")
            counts = [
                unit_counts[name] for name in row["config"]["datasets"] if name in unit_counts
            ]
            if str(row["task_key"]) in SUPPLEMENTAL_TASK_KEYS:
                progress = row.get("progress") or {}
                progress_datasets = progress.get("datasets") or {}
                completed_datasets = (
                    set(progress_datasets.keys())
                    if isinstance(progress_datasets, dict)
                    else set()
                )
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
                        round(sum(per_dataset) / len(per_dataset) * 100, 1) if per_dataset else 0.0
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
            row["unit_stats"] = {
                "planned": sum(item["planned"] for item in counts),
                "succeeded": sum(item["succeeded"] for item in counts),
                "pending": sum(item["pending"] for item in counts),
                "running": sum(item["running"] for item in counts),
                "retry_waiting": sum(item["retry_waiting"] for item in counts),
                "terminal_failed": sum(item["terminal_failed"] for item in counts),
                "rate_limited": sum(item["rate_limited"] for item in counts),
                "superseded": sum(item["superseded"] for item in counts),
                "rows": sum(item["rows"] for item in counts),
                "next_retry_at": min(
                    (
                        item["next_retry_at"]
                        for item in counts
                        if item["next_retry_at"] is not None
                    ),
                    default=None,
                ),
            }
        status_by_key = {str(row["task_key"]): str(row["status"]) for row in rows}
        for row in rows:
            dependencies = row["depends_on"]
            row["dependencies_satisfied"] = all(
                status_by_key.get(key) == "succeeded" for key in dependencies
            )
            stats = row["unit_stats"]
            job = row.get("job") or {}
            progress = row.get("progress") or {}
            status = str(row["status"])
            next_attempt_at = job.get("next_attempt_at")
            if not row["dependencies_satisfied"] and status not in {
                "queued",
                "running",
                "succeeded",
            }:
                phase = "blocked_prerequisite"
            elif status == "queued" and next_attempt_at is not None:
                phase = "retry_waiting"
            elif status == "queued":
                phase = "queued"
            elif status == "running" and stats["rate_limited"] and not stats["running"]:
                phase = "rate_limit_cooldown"
            elif status == "running":
                phase = str(progress.get("execution_phase") or "planning")
            elif status == "failed" and stats["retry_waiting"]:
                phase = "recoverable_failure"
            elif status == "failed":
                phase = "terminal_failure"
            elif status == "succeeded":
                phase = "verified"
            elif status == "partial":
                phase = "partial"
            else:
                phase = "ready_to_start"
            row["execution_phase"] = phase
        return rows
