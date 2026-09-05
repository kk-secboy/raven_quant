from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import pytest

pytestmark = pytest.mark.no_database

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_NAME = "个人量化投资与模拟盘系统设计稿.md"
SOURCE_MARKDOWN_NAME = "如何搭建自己的量化交易系统.md"
DOCX_NAME = "如何搭建自己的量化交易系统.docx"
DOCX_BACKUP_NAME = "如何搭建自己的量化交易系统-GitHub原版副本.docx"
DOCX_SHA256 = "d35136f6546c05cfd8ee998d3716cac430590116ad4e1f32227c1c4759d7143a"
DOCX_BACKUP_SHA256 = "efb59da8b2296b6eec6fb9c81e621bb8c8f45d426fa7bf8a84a6d304531ec50a"
FROZEN_SOURCE_NAMES = {DOCX_BACKUP_NAME, SOURCE_MARKDOWN_NAME}
CONTROLLED_DOCUMENTS = {
    Path("README.md"),
    Path("docs/DEPLOYMENT.md"),
    Path("docs/design-gap-analysis.md"),
    Path("docs/information-factor-refresh.md"),
    Path("docs/information-pipeline-operations.md"),
    Path("docs/legacy-market-backfill.md"),
    Path("docs/peripheral-market-data-plan.md"),
    Path("docs/pit-nlp-gap-report.md"),
    Path("docs/financial-correctness-audit-2026-08-25.md"),
    Path("docs/qlib-rdagent-capability-audit-2026-08-25.md"),
    Path(MARKDOWN_NAME),
    Path(DOCX_NAME),
}
NON_PRODUCT_DOCUMENT_NAMES = {
    "AGENTS.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
}
RETIRED_DOCUMENTS = {
    "docs/ARCHITECTURE.md",
    "docs/DATA_CENTER.md",
    "docs/DOCX_REQUIREMENTS.md",
    "docs/PRODUCT_ACCEPTANCE.md",
    "docs/SHORT_HORIZON_RESEARCH_TODO.md",
    "docs/TUSHARE_COVERAGE.md",
    "web/README.md",
}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
IGNORED_ARTIFACT_PARTS = {
    ".git",
    ".codex_tmp",
    ".codex-tmp",
    ".venv",
    ".pytest-tmp",
    ".worktrees",
    "node_modules",
    ".next",
    ".vinext",
    "dist",
    "build",
}


def _project_artifacts(pattern: str) -> list[Path]:
    return sorted(
        (
            path
            for path in PROJECT_ROOT.rglob(pattern)
            if not any(
                part in IGNORED_ARTIFACT_PARTS or part.startswith(".tmp-pytest")
                for part in path.relative_to(PROJECT_ROOT).parts
            )
        ),
        key=lambda path: path.as_posix(),
    )


def _is_legal_or_instruction_document(path: Path) -> bool:
    name = path.name
    return (
        name in NON_PRODUCT_DOCUMENT_NAMES
        or name.startswith("LICENSE")
        or name.startswith("NOTICE")
    )


def _controlled_document_candidates() -> set[Path]:
    candidates: set[Path] = set()
    for path in PROJECT_ROOT.glob("*.md"):
        if path.name in FROZEN_SOURCE_NAMES:
            continue
        if not _is_legal_or_instruction_document(path):
            candidates.add(path.relative_to(PROJECT_ROOT))
    for directory in (PROJECT_ROOT / "docs", PROJECT_ROOT / "web"):
        if not directory.exists():
            continue
        for pattern in ("*.md", "*.docx"):
            for path in directory.rglob(pattern):
                relative = path.relative_to(PROJECT_ROOT)
                if any(
                    part in {"node_modules", ".next", ".vinext", "dist", "build"}
                    for part in relative.parts
                ):
                    continue
                if not _is_legal_or_instruction_document(path):
                    candidates.add(relative)
    candidates.update(
        path.relative_to(PROJECT_ROOT)
        for path in PROJECT_ROOT.glob("*.docx")
        if path.name not in FROZEN_SOURCE_NAMES
    )
    return candidates


def _local_markdown_targets(document: Path) -> list[Path]:
    targets: list[Path] = []
    for raw_target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
        target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        decoded_path = unquote(parsed.path)
        targets.append((document.parent / decoded_path).resolve())
    return targets


def test_two_frozen_docx_snapshots_are_valid_and_unchanged() -> None:
    docx_files = _project_artifacts("*.docx")
    expected = PROJECT_ROOT / DOCX_NAME
    backup = PROJECT_ROOT / DOCX_BACKUP_NAME
    assert len(docx_files) == 2
    assert set(docx_files) == {expected, backup}
    assert _project_artifacts("*.pdf") == []

    for document in (expected, backup):
        try:
            with ZipFile(document) as archive:
                members = set(archive.namelist())
                required_members = {
                    "[Content_Types].xml",
                    "_rels/.rels",
                    "word/document.xml",
                }
                assert required_members <= members
                ElementTree.fromstring(archive.read("[Content_Types].xml"))
                ElementTree.fromstring(archive.read("word/document.xml"))
        except BadZipFile as exc:  # pragma: no cover - useful assertion message
            raise AssertionError(
                f"{document.name} is not a valid OOXML package"
            ) from exc
    assert hashlib.sha256(expected.read_bytes()).hexdigest() == DOCX_SHA256
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == DOCX_BACKUP_SHA256


def test_controlled_product_documents_match_the_whitelist() -> None:
    assert _controlled_document_candidates() == CONTROLLED_DOCUMENTS


def test_readme_declares_the_document_and_technical_authorities() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "是产品、策略和风险基准，Qlib/RD-Agent 是技术基准" in readme
    assert "不可变快照" in readme
    assert "三周期账户净额" in readme
    assert "`fin_strategy`" in readme
    assert "满足各周期真实前向门后由系统原子自动晋级" in readme
    assert "`policy_only` 只比较买卖和持仓规则，保持同一份分数" in readme
    assert "区间不得早于成本制度首个有效交易日" in readme
    assert "唯一自动资本入口是受管 `fin_strategy` 的 settlement" in readme
    assert "`AutopilotCapitalPipeline`" in readme
    assert "旧 `/api/research-programs` 与 `/api/research-campaigns` 端点已物理删除" in readme
    assert "Loop、Hypothesis、Feedback 和 Trace 摘要" in readme
    assert "本 README 只提供项目入口和" in readme
    assert "不定义另一套产品方案" in readme
    assert "根目录同名 DOCX 仅保留为 3.0 定稿发布快照" in readme
    assert "日常修订不依赖 LibreOffice" in readme
    assert f"`{DOCX_BACKUP_NAME}` 仅保留为冻结的历史原稿" in readme


def test_public_product_copy_does_not_restore_manual_strategy_promotion() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "docs" / "design-gap-analysis.md",
            PROJECT_ROOT / "web" / "app" / "factor-library-panel.tsx",
            PROJECT_ROOT / "web" / "app" / "rdagent-panel.tsx",
        )
    )
    for obsolete in (
        "人工批准 / 模拟盘",
        "正式回测后仍需人工批准",
        "策略发布和模拟盘仍需人工批准",
        "证据门 + 人工批准",
    ):
        assert obsolete not in sources


def test_authoritative_markdown_contains_the_current_contract() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")
    required_contracts = (
        "1.2 三周期策略研究与持续荐股定稿",
        "这是唯一生产主线，不另建第二套 Agent、策略仓库、数据库或生命周期",
        "`short_1_5d`",
        "`swing_1_6m`",
        "`long_1_3y`",
        "前向试跑至少约 3 个月（60 个真实交易日）",
        "前向试跑至少约 6 个月（120 个真实交易日）",
        "正式荐股前向门至少约 6 个月（120 个真实交易日）",
        "满 3 年、36 次月度复核、12 次财报复核只授予“三年前向成熟”标签",
        "通过不可变决定自动晋级，不需要人工挑选",
        "短线约 3 个月、中线约 6 个月、长线约 6 个月后才可能分别出现“已验证荐股”",
        "研究到晋级链按“宽进严出”标定",
        "整条链唯一的生死门是模拟盘前向表现",
        "正式样本外结算改为衰减检查",
        "不参与晋级否决",
        "模型锦标赛（特征集筛选、全量模型决赛、预测冠军终选）的 Holm/PBO 家族校正照常计算并密封，"
        "但不再阻断锦标赛",
        "无人达标时如实记录 `gate_passed=false` 与空 eligible 名单，不阻断流程",
        "外部 NLP 因子（公告、舆情等稀疏事件与市场时序信号）同样只保留硬结构门",
        "因子 SOTA 增量门同理：BH 校正 q 值、配对 HAC p 值与配对 bootstrap 区间照常密封入档，"
        "不再否决",
        "已经开始模拟的版本保留其原门槛",
        "`fin_strategy`",
        "不得因为缺少已批准旧策略而形成“先有正式策略才能研究新策略”的循环等待",
        "`fin-strategy-signal-binding-v1`",
        "唯一自动资本入口是受管 `fin_strategy` settlement",
        "旧 `AutopilotCapitalPipeline` 代码已物理删除，既有 `capital_pipeline` "
        "状态永久标记为 `legacy_readonly`",
        "LLM 不能替换分数源",
        "不得早于对应成本制度首个有效交易日",
        "`StrategyProposal`",
        "确定性编译",
        "`strategy-rule-ir-v1`",
        "`SOTA` 只表示官方研究 Trace 中的当前最好实验",
        "旧 `research_program/research_campaign` 的创建、状态修改、重试、调度与历史查询 API "
        "端点已物理删除",
        "记录只在数据库表中保留 `legacy_readonly` 历史",
        "日常推荐只执行编译后的规则 IR，绝不运行 LLM 代码",
        "每个周期只冻结一份活动冠军因子包",
        "每个周期同一时点最多保留一个活动策略和两个隔离的 `paper_validating` 影子挑战者",
        "同一特征集锦标赛必须先按全部预注册候选所需字段的并集解析共同连续可用日历",
        "共同窗口政策只随新锦标赛制品生效",
        "今日唯一账户操作清单",
        "20/50/30 短中线为主的三周期预算",
        "停牌或当日执行价格证据缺失时只能冻结为 `WAIT` 并延后目标",
        "`research_sota → research_candidate → governed_evaluation_winner → "
        "paper_validating → recommendation_enabled`",
        "`system:auto-promotion`",
        "D 日完整收盘数据 → D+1 开盘或保守日线成交模型",
        "`reporting_only` 基准只用于绩效对照",
        "不能因为运行时提供了基准权重，就给 `topk_equal_weight` 偷加基准行业权重偏离带",
        "风格数据质量门只检查冻结策略实际消费的输入",
        "正式回测和推荐刷新必须采用同一选择规则",
        "无日期参数的可转债评级 `cb_rating` 按日刷新请求桶",
        "同一桶复用已成功检查点，跨桶追加请求",
        "原查询参数、历史检查点和快照保持不变",
        "v34 及更早版本的身份和证据永久保留，不重新绑定到新源码闭包",
        "`qlib-rdagent-single-mainline-2026-09-06-v36`",
        "一次准备 PIT 证券资格和 Qlib 截面查找",
        "输入选择、交易规则、成本和 OOS 边界保持不变",
        "`0107_strategy_runtime_v36`",
        "v35 及更早版本的身份和证据永久保留，不重新绑定到新源码闭包",
        "旧冻结任务不得改用新运行时，已经消费的正式 OOS 不得重用",
        "不能把近期下载窗口当作公告日期窗口",
        "报告期不能替代公告可用时间",
        "没有合格股票时输出现金/`NO_ACTION`",
        "初始资金是首次使用时必填的账户配置",
        "也没有 10 万或 50 万元硬下限",
        "系统能力验收不要求策略盈利",
        "正式回测已经用不可变制品证明核心门槛终局失败时，只能把该周期记为拒绝并投影为现金/`NO_ACTION`",
        "这仍不构成收益保证",
        "当前没有盘中数据接口",
        "生产发布没有 QMT 或任何券商网关例外",
        "不得推导或伪造涨跌停价",
        "逐行 fail-closed 标记覆盖全部缺失行",
        "当前集合存在即完全优先，上市或状态未知股票不得继承",
        "锚点快照、文件哈希、继承股票和行数必须写入不可变证据",
    )
    for contract in required_contracts:
        assert contract in specification

    for obsolete in (
        "证据门 + 人工批准",
        "人工模板设计确认",
        "人工推荐批准",
        "默认关闭的 QMT 沙箱插件是唯一例外",
        "本项目个人 paper 账户首轮默认使用 10 万元",
        "## 15. v1.1",
    ):
        assert obsolete not in specification


def test_capacity_rounding_contract_keeps_constraints_and_versions_separate() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")
    capacity = specification.partition("### 5.5 容量与离散交易")[2].partition(
        "### 5.6 公司行动"
    )[0]
    for contract in (
        "申报单位约束作用于本次买卖增量",
        "合法持仓保持不动时必须保留原数量",
        "真实交易约束不可同时满足时失败关闭",
        "内部数量乘信号日 factor 得到等价原始股数",
        "与 NAV 一致的原始估值价格",
        "当日行情缺失时只复用不晚于信号时点的最后有效 factor",
        "不得填充报价、成交量或可交易状态",
        "没有任何历史因子证据时仍失败关闭",
        "数量口径由每次回测实际使用的执行器决定",
        "原价分钟执行采用 factor=1，即使信号是日线",
        "历史代理切回日线撮合时仍采用日线复权因子",
        "禁止下单器再按另一个价格、默认风险比例或统一 100 股单位二次缩放",
        "T+1 下限必须进入同一次数量校验",
        "仅显式启用 Qlib 原有完整清仓特例",
        "部分卖出不获得任意小数数量权限",
        "推荐目标历史不能冒充真实持仓",
        "离散后的全部硬约束继续复检",
        "`qlib-rdagent-single-mainline-2026-09-06-v38`",
        "`0109_strategy_runtime_v38`",
        "`0108_strategy_runtime_v37`",
        "受影响回测必须重新计算",
        "v37 及更早版本的身份、制品和 OOS 消费记录保持只读",
    ):
        assert contract in capacity
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "当前受管策略运行时为 `qlib-rdagent-single-mainline-2026-09-06-v38`" in readme
    assert "不能沿用旧结果宣称等价" in readme


def test_run_status_presentation_keeps_audit_evidence_and_financial_permissions() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")
    reliability = specification.partition("### 11.2 任务可靠性")[2].partition(
        "### 11.3 最小安全"
    )[0]
    for contract in (
        "当前运行与历史记录必须分区展示",
        "最近结束的研究结果须保持可见并按结束时间排序",
        "执行中已有试验失败时，页面须同时说明仍在执行和已有失败",
        "进度文件更新时间不能冒充执行心跳",
        "刷新失败时保留记录并明确标注“上次状态”",
        "上次成功读取时间",
        "公开失败原因只使用固定原因码和固定安全文案",
        "不能直接展示原始异常、路径、命令、URI 或凭据",
        "有明确中止请求或所属任务取消证据时",
        "不得改写数据库的历史 `failed/cancelled` 状态",
        "研究与执行任务只按持久化的任务 ID 和所属研究 ID 关联",
        "不能按名称、描述或全局最新任务猜测关联",
        "子任务成功不等于整条研究完成",
        "展示投影不得修改金融权限、准入结果、运行时身份或 OOS 消费记录",
    ):
        assert contract in reliability

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for contract in (
        "当前运行与历史记录分开显示",
        "刷新失败时显示“上次状态”与上次成功读取时间",
        "维护中止保留原始审计状态",
        "固定安全原因",
        "不改变金融权限、运行时身份或 OOS 边界",
    ):
        assert contract in readme


def test_design_status_and_four_correctness_lines_do_not_overclaim_implementation() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")
    header = specification.partition("## 1. 产品目标与边界")[0]
    assert "已冻结的目标设计" in header
    assert "不代表当前代码已经实现" in header
    assert "也不代表任何策略已经具有可投资价值" in header

    correctness = specification.partition("### 1.2 四条正确性主线")[2].partition(
        "### 1.3 明确不做"
    )[0]
    for perspective in (
        "数学与统计",
        "金融市场与会计",
        "量化研究",
        "模拟盘投资",
    ):
        assert perspective in correctness

    acceptance = specification.partition("### 12.5 端到端验收")[2].partition(
        "### 12.6 固定实施顺序"
    )[0]
    assert "系统能力验收不要求策略盈利" in acceptance
    assert "正确地输出“候选不合格”“证据不足”或 `NO_ACTION` 也属于通过" in acceptance
    assert "这仍不构成收益保证" in acceptance

    closing = specification.partition("## 15. v1.2 三周期策略研究与持续荐股定稿")[2]
    for disclaimer in (
        "设计完整不等于策略已经通过证据门、适合个人或具有正收益",
        "自动晋级不代表保证盈利",
        "本文定稿不等于当前代码已经完成",
        "当前项目实现是否符合本文，应由代码、配置和验收测试逐项证明",
    ):
        assert disclaimer in closing


def test_current_body_contains_core_contracts_not_only_acceptance_lists() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")
    design_body = specification.partition("## 12. 测试、验收与实施顺序")[0]
    current_contracts = (
        "三个唯一权威",
        "`ExecutionCore`",
        "`state_delta`",
        "`StrategySpec`",
        "`ModelArtifact`",
        "`AccountAllocationPolicy`",
        "`AllocationArtifact`",
        "`ExecutionPolicy`",
        "`StrategyProposal`",
        "`ResearchBrief`",
        "`ExperimentLedger`",
        "`economic_hypothesis_group`",
        "`capital_eligible_strategy`",
        "`controlled_custom`",
        "成本只计算一次",
        "系统唯一采用**交易日会计**",
        "`NAV = Σ互斥现金余额 + Σ(经济持仓数量 × 估值价格) + 公司行动应收 - "
        "应付款及已确认税费负债`",
        "最终样本外是一次性资源，而不是每个候选都能重复领取的区间",
        "`sealed_candidate_set_id/hash`",
        "`native_history`（上游原生历史版本）、`reconstructed`（依据当时公告重建）、"
        "`current_only`（只有当前修订）或 "
        "`unavailable`",
        "research_sota → research_candidate → governed_evaluation_winner → "
        "paper_validating → recommendation_enabled",
        "`forward_paper`",
        "`main_paper`",
        "`manual_shadow`",
        "`active_recommendation_account_id`",
        "`risk_scope=selected_account_only`",
        "no-trade band",
        "`projected_position = filled_position + 有效未完成买单 - 有效未完成卖单`",
        "`keep/cancel/replace/new`",
        "`planned → open → partially_filled → filled | cancelled | expired | rejected`",
        "`effective_at`",
        "`available_at`",
        "`ingested_at`",
        "purge",
        "embargo",
        "HAC 标准误或时间区块 Bootstrap",
        "Implementation Shortfall",
        "`catalog_role`",
        "`implementation_tier`",
    )
    for contract in current_contracts:
        assert contract in design_body


def test_document_version_and_frozen_status_are_consistent() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")
    assert re.search(
        r"^\| 版本 \| 1\.2 三周期策略研究与持续荐股定稿 \|$",
        specification,
        re.MULTILINE,
    )
    assert re.search(
        r"^\| 日期 \| 2026-08-29 \|$",
        specification,
        re.MULTILINE,
    )
    assert re.search(
        r"^\| 文档状态 \| 已冻结的目标设计；不代表当前代码已经实现，"
        r"也不代表任何策略已经具有可投资价值 \|$",
        specification,
        re.MULTILINE,
    )
    closing = specification.partition("## 15. v1.2 三周期策略研究与持续荐股定稿")[2]
    assert "本文冻结为个人投资者或 1—5 人小团队量化平台 v1.2 目标设计" in closing
    assert (
        "只有产品范围变化，或发现会改变数据、回测、模拟账本、投资证据、"
        "账户动作或非实盘边界的原则性错误时，才升级为后续设计版本"
        in closing
    )


def test_markdown_tables_have_consistent_unescaped_pipe_counts() -> None:
    lines = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8").splitlines()
    table: list[tuple[int, str]] = []
    tables: list[list[tuple[int, str]]] = []
    for line_number, line in enumerate(lines, start=1):
        if line.startswith("|"):
            table.append((line_number, line))
        elif table:
            tables.append(table)
            table = []
    if table:
        tables.append(table)

    assert tables
    for rows in tables:
        expected = len(UNESCAPED_PIPE.findall(rows[0][1]))
        for line_number, row in rows[1:]:
            actual = len(UNESCAPED_PIPE.findall(row))
            assert actual == expected, (
                f"Markdown table at line {line_number} has {actual} "
                f"unescaped pipes; expected {expected}"
            )


def test_quantitative_formula_and_state_invariants_are_explicit() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")
    invariants = (
        "adjusted_price(t, b) = raw_price(t) × adj_factor(t) / adj_factor(b)",
        "total_return(t) = raw_price(t) × adj_factor(t) / [raw_price(t-1) × adj_factor(t-1)] - 1",
        "r_t = (V_t - F_t_close) / (V_{t-1} + F_t_open) - 1",
        "investment_wealth_t = investment_wealth_{t-1} × (1 + r_t)",
        "TWR = investment_wealth_T / investment_wealth_0 - 1",
        "max_drawdown = max_t [1 - investment_wealth_t / max_{s≤t}(investment_wealth_s)]",
        "one_way_turnover = 0.5 × [Σ_证券 |target_weight_i - current_weight_i| + "
        "|target_cash_weight - current_cash_weight|]",
        "NAV = Σ互斥现金余额 + Σ(经济持仓数量 × 估值价格) + 公司行动应收 - "
        "应付款及已确认税费负债",
        "projected_position = filled_position + 有效未完成买单 - 有效未完成卖单",
        "ExecutionCore(account_snapshot, market_event, order_intent, rule_config) → fills + "
        "state_delta + reason",
        "planned → open → partially_filled → filled | cancelled | expired | rejected",
        "research_sota → research_candidate → governed_evaluation_winner → "
        "paper_validating → recommendation_enabled",
        "资格 → 去重 → 方向/环境 → 排名/预测 → 入场时机 → 退出状态 → 策略内组合风险 → "
        "执行要求",
        "按冻结日历生成或读取 AllocationArtifact → 将其中预算应用一次 → 各 StrategySpec "
        "在预算内输出证券目标 → 证券级净额合并 → "
        "账户硬约束与 no-trade band → ExecutionPolicy",
        "卖出回款不得同时出现在现金、在途现金和应收款中",
        "待交收证券也不得在经济持仓之外再计一份资产",
        "每项经济成本只能在“成交价格影响”或“现金费用”中出现一次",
        "重复处理同一事件不得再次改变现金、证券或费用",
        "公告只产生信息和提醒，不改变现金、持仓或 NAV",
        "除税费差额外不得再次增加 NAV",
    )
    for invariant in invariants:
        assert invariant in specification


def test_single_account_target_and_no_double_counted_costs() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")

    for required in (
        "将所有有效策略净额合并后，只发布一份账户目标",
        "单个策略信号不能直接改模拟持仓",
        "最后生成唯一最终目标和一份综合建议，再交给模拟盘",
        "多策略净额合并后只能创建账户级综合模拟订单",
        "相反需求先净额抵消；不得让两个策略分别产生相反真实订单来制造换手，"
        "也不得把净额后的同一成交和费用在多个策略重复计入",
        "每项经济成本只能在“成交价格影响”或“现金费用”中出现一次",
        "未成交和延迟单独用于实现差额分析，不能既改成交价又再次从现金扣除",
        "若基金净值、市场价格或收益序列已经内含管理费用，不得再次从账户扣除同一费用",
        "任何冻结、消费和释放都使用同一订单事件键，不能在可用余额和冻结余额各扣一次",
        "模拟交易费用在成交时按冻结规则确认一次；"
        "日终校准或用户导入最终费用时只记录“最终费用减已确认费用”的差额，不得再次全额计提",
    ):
        assert required in specification

    for obsolete in (
        "`baseline_account_target_weight`",
        "`candidate_account_target_weight`",
        "`no_trade_band_utility`",
        "delta_utility = robust_utility",
    ):
        assert obsolete not in specification


def test_final_out_of_sample_is_a_one_time_sealed_resource() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")

    for required in (
        "最终样本外是一次性资源，而不是每个候选都能重复领取的区间",
        "`research_campaign_id`、`hypothesis_family_id`、`oos_vintage_id`、`sealed_at`、"
        "`first_opened_at` 和 `consumed_at`",
        "`sealed_candidate_set_id/hash`、有序候选 ID 与 `StrategySpec` 哈希、主要指标、"
        "通过规则和多重检验规则",
        "任何人、脚本或 Agent 第一次看过结果后，该 `oos_vintage_id` 即标记为 `consumed`",
        "所有 `candidate_formation_ts > first_opened_at` 的候选都不再具有该区间的最终未见样本资格",
        "不论后来如何新建或改名研究活动、假设谱系和策略版本",
        "即使候选在打开前已经存在，只要未列入密封候选集合，"
        "也不能在看过结果后补入该集合或使用该 OOS 晋升",
        "只有首次打开前共同冻结且列入密封集合的候选，才可按预注册规则一起确认，不能看完后再挑赢家",
        "旧 `oos_vintage_id` 仍保持已消费且绝不删除、回退或改写",
        "才允许一次“非统计性前结果修复”",
        "不能因改名、换数据目录或生成新 lineage 自动获得重跑资格",
        "任何看过收益、净值、Sharpe、相对基准或其他绩效后提出的修复一律不得复用该窗口",
        "RD-Agent 研究环境在物理挂载和数据权限上都不能读取最终样本外区间；"
        "只有候选冻结后的独立正式任务可以读取",
        "打开结果后发生任何特征、标签、模型、参数、组合或选择规则修改，都必须形成新候选",
        "最终保留一个未参与选择的样本外区间，只在候选冻结后使用",
        "至少三个滚动窗口的稳定性门必须只使用最终锁箱之前的外层 Walk-forward "
        "或独立模型 profile/seed 网格",
        "最终密封 OOS 只形成一次资本确认和描述性路径诊断，"
        "不能通过切成重叠或更短窗口伪造多个独立证据",
        "`legacy_ambiguous` 历史版本继续按其冻结的原滚动门解释，不能追溯改写",
    ):
        assert required in specification


def test_trial_ledger_and_independence_controls_prevent_self_deception() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")

    for required in (
        "人工、脚本和 RD-Agent 的所有候选使用同一份简化试验台账",
        "成功、失败、中止和被放弃的全部试验",
        "不能只登记赢家",
        "候选之间的继承关系，以及研究人员何时看过哪些结果",
        "查看最终样本外结果后再修改候选，必须创建新候选，并使用之后未参与修改的新数据继续验证；"
        "同谱系候选不能继续复用已经消耗的最终样本外",
        "LLM 可能已经从训练语料或联网检索中知道历史测试期之后的论文、事件和策略结果，"
        "因此旧历史区间不一定是真正未暴露证据",
        "旧历史回测只能支持技术和经济合理性，候选必须继续使用形成后的新数据做隔离前向模拟，"
        "不能仅凭旧历史结果进入个人推荐",
        "多个近似版本、频率或模型共享同一 `economic_hypothesis_group`，"
        "统一进入试验计数和资本上限；改名不能获得新最终样本外或多份独立资金",
        "股票数量不是时间独立样本数",
        "分钟 Bar 数不是独立投资决策数",
        "重复交叉验证折、随机种子和 Monte Carlo 路径不产生新的市场证据",
        "月频策略主要按独立月度决策和完整持有周期判断证据；周频、"
        "日频和分钟策略分别按自己的有效决策、成交和持有周期判断",
        "截面因子先计算每日 IC/Rank IC，再对日期序列做推断，"
        "不能把同一天数千只股票当作数千个独立日期",
        "时间序列相关性可使用 HAC 标准误或时间区块 Bootstrap；"
        "候选相对基线的差异应在相同时间区间做成对比较",
        "区块长度、方法和主要指标必须预先确定，并进行合理敏感性检查",
        "候选很多时，根据假设数量和依赖结构选择一种适当的标准校正方法，例如探索阶段的 FDR，"
        "或最终少量候选的 Holm 校正",
        "DSR、PBO 等只在前提成立时作为诊断，不把所有方法同时堆成通用认证平台",
        "不得伪造 `trial_sharpes`",
        "`not_computable_incomplete_historical_family`",
        "按试验台账中的真实试验数作 Bonferroni 家族校正",
        "资本 OOS alpha 台账仍是独立且不变的门",
        "唯一的历史结果已打开恢复例外固定为 `forward-only-rehabilitation-v1`",
        "`final_oos_opened=true`、`sealed_final_oos=false`、"
        "`unseen_oos=false` 和 `capital_eligible=false`",
        "至少 365 个自然日、252 个真实前向交易日、60 次有效决策和 30 个完整交易闭环",
        "普通未泄露的 `short_1_5d` 新版本仍执行本节和 4.1 节约 3 个月（60 个真实交易日）的前向门",
        "具有完整候选矩阵的 model/`fin_quant` 路径继续执行原有 Holm、PBO 与 DSR 合同",
    ):
        assert required in specification

    for obsolete in (
        "`bootstrap_monte_carlo_precision_contract`",
        "`circular_moving_block_bootstrap`",
        "`joint_resampling_group_hash`",
        "`statistical_inference_contract_version`",
        "N_eff = min(N_unique, N_acf, N_nonoverlap)",
        "禁止把 0.95 或 20% 当作跨研究程序通用门",
    ):
        assert obsolete not in specification


def test_account_candidate_paths_and_no_trade_band_are_explicit() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")

    for required in (
        "分别构建“继续当前有效目标”和“切换到候选目标”的可执行路径，计算完整换仓成本、"
        "整手误差、容量、执行时间和风险",
        "对普通 Alpha 调整应用 no-trade band，先决定保留旧目标还是接受候选目标；"
        "增量不足以覆盖成本和不确定性时保留旧目标及其订单计划",
        "现金缺口、证券失效、权限收紧和风险减仓/退出等硬约束绕过 Alpha no-trade band，"
        "但仍受可卖数量、停牌和市场可交易性限制",
        "账户设置简单的不交易区间，用于避免微小目标变化、数值误差或模型噪声导致频繁交易",
        "佣金、税费、滑点和冲击已经进入换仓成本，不能又重复塞入 no-trade band",
        "具有经过样本外校准的预期收益时，可以比较保守增量收益与完整成本",
        "只有排名分数或规则目标时，使用预先冻结的目标漂移带、最小权重变化和最小合法订单，"
        "不得把 rank score 与人民币成本直接比较",
        "满足相应门槛后才发布新交易建议",
        "`projected_position = filled_position + 有效未完成买单 - 有效未完成卖单`",
        "并逐笔生成 `keep/cancel/replace/new` 计划",
        "新订单只补最终目标与预计持仓之间的差额",
        "目标与订单计划在同一事务提交，每次部分成交后按同一顺序重新计算，任务重试必须得到相同结果",
        "账户动作由最终目标与 `filled_position` 比较，订单数量由最终目标与 "
        "`projected_position` 比较",
    ):
        assert required in specification

    for obsolete in (
        "`no_trade_band_utility`",
        "`required_intent_only_plan_id/hash`",
        "`intent_obligation_class`",
        "`baseline_order_set`",
        "`candidate_transition_plan`",
        "optional_delta_path_utility",
        "total_delta_path_utility",
    ):
        assert obsolete not in specification


def test_cost_cash_and_nav_conservation_prevent_double_counting() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")

    for required in (
        "系统唯一采用**交易日会计**",
        "模拟成交一经确认，立即确认经济持仓或卖出回款以及本笔交易费用；交收只改变可卖、可交易、"
        "可取和科目分类，不再次改变经济资产与 NAV",
        "`tradable_at`、`withdrawable_at` 是现金批次的可用时间属性，`sellable_at` "
        "是证券批次的状态属性，不另记一份现金或证券资产",
        "卖出回款不得同时出现在现金、在途现金和应收款中，"
        "待交收证券也不得在经济持仓之外再计一份资产",
        "模拟订单创建时只把自由现金/证券重分类为冻结状态，不改变 NAV；"
        "买入成交从对应冻结现金批次消费，取消或部分未成交只释放剩余冻结额",
        "交收事件只做状态重分类，除新增确定的费用或税费差额外不得改变 NAV",
        "公司行动应收与普通卖出回款分开保存，每个成交、费用、交收和公司行动事件都必须有唯一键",
        "`可交易现金` 和 `可取现金` 是根据现金批次及其时间属性计算的权限视图，不能再加进 NAV",
        "买入、卖出及随后交收在市场价格不变且不考虑费用时只能改变资产分类，不能令 NAV 重复跳变",
        "实际入金和出金是外部现金流，不得误报为投资收益或亏损；"
        "尚未实际划出的计划用款只是账户现金约束",
        "公司行动前后在没有真实经济损失的情况下应满足价值守恒",
        "闲置现金只按账户实际可获得的配置收益计息；没有对应现金工具时使用零收益，"
        "不能把研究无风险利率当作账户实际收益",
        "尚未转出的计划用款只改变最低现金和投资约束，不进入收益公式",
        "账户内部股息、卖出回款、费用和税费不能再次当作外部现金流",
        "分红、利息、卖出回款、费用和税费都是账户内部现金流，不进入 `F_t_open/F_t_close`",
    ):
        assert required in specification

    for obsolete in (
        "`baseline_tax_cashflow_path`",
        "`candidate_tax_cashflow_path`",
        "`loss_baseline_q`",
        "`personal_real_discount_contract`",
        "`λ_dd_cny_per_unit`",
    ):
        assert obsolete not in specification


def test_single_execution_core_and_forward_stages_have_no_shadow_path() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")

    for required in (
        "项目只实现一个轻量、确定性的 `ExecutionCore`，不是事件溯源平台",
        "它必须是无数据库、无系统时钟、无外部副作用的纯规则函数",
        "两种适配器都只能把同一 `state_delta` 应用一次，不得再次计算或重复应用成交、费用、"
        "交收和现金变化",
        "策略只读取上一事件已经应用完成的账户快照，不能在 Qlib 状态和另一套账本之间择优取值",
        "公司行动和交收由独立账本处理器驱动时，也必须调用同一组纯规则原语并只应用一次",
        "两条适配链在相同历史事件和订单输入上必须通过黄金案例和差分测试；Qlib "
        "未改造的默认执行器只作探索对照",
        "正式 Qlib 工作流只能使用调用 `ExecutionCore` 的自定义 Executor/Exchange 适配器",
        "Qlib 默认 Executor/Exchange 可以作快速探索和兼容对照，但不能形成正式准入或模拟晋升证据",
        "每笔成交由 `ExecutionCore` 返回唯一 `state_delta`，历史 Qlib 适配器或前向 PostgreSQL "
        "事务只应用一次",
        "Qlib 内部账户状态不能代替持久模拟账户",
        "历史分钟数据盘后补齐不能回写并美化当天的前向模拟成交或提醒",
        "多个候选可以共用进程和只读数据，但不得共用账本或资本",
        "同一账户的全部订单切片按成交时点排序，同一时点先卖后买",
        "未来卖出回款不能为较早买入融资",
        "同一证券、同一分钟的所有订单共享参与率容量",
    ):
        assert required in specification

    for obsolete in (
        "`joint_resampling_group_hash`",
        "`joint_resampling_design_hash`",
        "`simulation_batch_source_type=integrated_recommendation`",
        "account_initialization_source_type",
        "`sampled_block_indices_hash`",
    ):
        assert obsolete not in specification


def test_core_contracts_are_synchronized_between_body_and_acceptance() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")
    design_body = specification.partition("## 12. 测试、验收与实施顺序")[0]
    acceptance = specification.partition("## 12. 测试、验收与实施顺序")[2].partition(
        "## 13. 条件启用与范围外能力"
    )[0]

    synchronized_contracts = (
        "ExecutionCore",
        "state_delta",
        "StrategyProposal",
        "StrategySpec",
        "ModelArtifact",
        "AllocationArtifact",
        "AccountAllocationPolicy",
        "ExecutionPolicy",
        "economic_hypothesis_group",
        "forward_paper",
        "recommendation_enabled",
        "no-trade band",
        "Implementation Shortfall",
        "risk_scope=selected_account_only",
        "capital_eligible_strategy",
        "catalog_role",
        "main_paper",
        "manual_shadow",
        "purge",
        "sealed_candidate_set_id",
    )
    for contract in synchronized_contracts:
        assert contract in design_body, f"{contract} missing from design body"
        assert contract in acceptance, f"{contract} missing from acceptance"


def test_paper_isolation_and_promotion_have_no_shadow_path() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")

    for required in (
        "`paper_validating`：自动创建独立隔离模拟账户，允许展示实验账户模拟数量，"
        "但不参与用户综合推荐，也不输出个人主账户数量建议",
        "`recommendation_enabled`：允许参与唯一账户综合建议和主模拟账户；仍不具备实盘交易权限",
        "通过历史门、密封 OOS、独立复算和本周期全部前向门后自动晋级",
        "自动晋级不是放松门槛",
        "证据不足时继续处于模拟验证",
        "历史验证门和前向证据门不能相互替代",
        "`forward_paper` 永远不能成为主动推荐账户",
        "`forward_paper`：只使用运行当时真实取得的数据向前运行；"
        "提醒和成交必须标记实验模拟账户来源，永远不发送最终个人建议",
        "每个实验前向阶段由“`StrategySpec` 版本 + 数据/PIT 语义合同版本 + 证券池规则版本 + "
        "成本规则 + "
        "`AccountAllocationPolicy` 版本/人工预算配置 + `ExecutionPolicy` 与路由规则版本集合 + "
        "主要风险配置”唯一确定",
        "旧阶段保持只读，旧、新自然时间、成交事件和表现不得拼接用于晋升，也不得修改旧阶段 NAV",
        "例行 `ModelArtifact` 切换若超出预注册日历或改变模型配方，必须先形成新 `StrategySpec`，"
        "不能借“refit”绕过重置",
        "主模拟账户的归因不能代替各策略独立 `forward_paper` 的晋升证据",
        "不得把上一计划的 `member_targets` 当作已经成交的袖套持仓",
        "部分成交、拒单或目标不变时，未完成差额仍须保留精确成员贡献并继续补单",
        "证据不足时链路正确停在 `paper_validating`，人工不能覆盖失败门",
        "任一因子、模型、规则、数据合同或执行合同实质变化都形成新版本并从零累计前向证据",
    ):
        assert required in specification

    for obsolete in (
        "`sealed_pre_certification_forward`",
        "`certification_forward`",
        "`pre_certification_forward_eligibility_manifest`",
        "`personal_operational`",
        "`simulation_promotion_firewall_version`",
    ):
        assert obsolete not in specification


def test_alpha_cannot_override_hard_gates() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")

    for required in (
        "硬约束优先于 Alpha。普通模型分数不能覆盖现金需求、不可交易、可卖数量、"
        "风险退出和人工执行能力",
        "风险覆盖只能减少或延后普通 Alpha 风险，不能凭空增加仓位",
        "风险减仓或 `EXIT` 是账户硬门，不是分钟 Alpha 反转，也不保证立即成交",
        "不能用高 Alpha 分数覆盖权限、不可交易、现金、容量或风险门",
        "硬资格、现金、容量和风险门不能被 Alpha 分数覆盖",
        "过滤器属于硬门，不能被高分覆盖",
        "人工不能覆盖失败门",
        "普通现金账户只能向现金缩放风险；未经策略和个人政策明确批准，"
        "不得为达到目标波动率自动加杠杆",
        "执行策略只能在上层最终账户目标和硬风险边界内工作。它不得自行增加目标仓位、"
        "把卖出改成买入、用盘中模型绕过 `recommendation_enabled`",
    ):
        assert required in specification

    for obsolete in (
        "`intent_obligation_class`",
        "`required_intent_only_plan_id/hash`",
        "optional_delta_path_utility",
        "total_delta_path_utility",
    ):
        assert obsolete not in specification


def test_governed_stock_history_excludes_unsupported_bse_and_b_shares() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")

    for required in (
        "产品股票范围只包含受治理的境内 A 股和 ETF 白名单，不包含 B 股",
        "Tushare 在 2023 年以前的北交所历史横截面和代码映射不完整",
        "只有从 2023-01-01 起",
        "更早的零散北交所行情只保留为原始审计数据",
        "禁止补零、代码回填或冒充可投资历史",
    ):
        assert required in specification


def test_insufficient_evidence_states_are_first_class() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")

    for required in (
        "样本不足时标记“证据不足”，不能通过降低门槛晋升",
        "主要结果的置信区间或明确的“证据不足”",
        "样本不足、时间未对齐或分母为零时输出“未定义/证据不足”，不得输出无穷大或伪精确数值",
        "证据不足时继续处于模拟验证",
        "证据不足时只显示“观察/已过期”，不输出伪精确分时点",
        "未经样本外校准的排名分数不得展示成“上涨概率 80%”或“预期收益 10%”",
        "若分母小于等于零、外部现金流时点不明确或账本未对平，"
        "从该日开始的连续表现曲线标记为不可用，必须修复账本并从最后一个已验证状态重建，"
        "不能只跳过该日后继续连乘",
        "系统可以得出“没有可靠机会”“继续持有”“等待执行”或“停止推荐”。"
        "为了保持有信号而降低验证标准，属于错误实现",
        "没有正 Alpha 时系统可以正确拒绝候选并保持简单基线",
    ):
        assert required in specification

    for obsolete in (
        "`inference_result_status`",
        "`not_reliably_estimable`",
        "`baseline_loss_status`",
        "`loss_scenario_set_id/hash`",
        "`statistical_assessment_id`",
    ):
        assert obsolete not in specification


def test_capacity_execution_and_maintenance_use_their_own_evidence() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")

    for required in (
        "固定“日均成交额大于某值”只能用于初筛，不能代替容量验证",
        "策略可以在某个账户规模有效、在另一个规模无效，报告必须明确适用资金范围",
        "执行算法检查完成率与 Implementation Shortfall",
        "下一 Bar、TWAP、VWAP 和切片执行使用 Implementation Shortfall、完成率、未成交率、"
        "参与率、延迟和容量验收，不使用 Alpha、IC "
        "或策略 Sharpe 代替执行证据",
        "`ExecutionPolicy` 不是 Alpha 策略，不获得策略资本或独立账户，不能用 Sharpe、IC 或 "
        "Alpha 证明自己有效",
        "监控输入缺失率、特征/预测分布、模型误差、成本偏差、回撤和相对基线表现。"
        "数据或实现错误立即暂停；统计或经济表现恶化时停止新增风险并进入复核",
        "refit 失败时继续使用仍有效的旧制品；旧制品也失效时回到简单基线或现金",
        "回滚只切换到同一有效 `StrategySpec` 下仍满足数据、规则和风险条件的历史 "
        "`ModelArtifact`，不能恢复已经失效或退役的配方",
        "小、中、大不同账户规模下的整手、最低佣金和容量",
    ):
        assert required in specification

    for obsolete in (
        "`capacity_assessment_id`",
        "`strategy_maintenance_economics_assessment_id`",
        "`maintenance_value_lcb`",
        "`execution_calibration_event_id`",
        "`conditional_goal_success_probability_point`",
    ):
        assert obsolete not in specification


def test_risk_and_drawdown_semantics_are_enforced() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")

    for required in (
        "最大回撤和恢复期基于单位化表现曲线，不基于人民币余额",
        "回撤线是风险处置触发器，不是损失保证。跳空、连续跌停、"
        "停牌和流动性枯竭都可能使实际损失超过处置线",
        "止损、止盈和波动率缩放不能承诺最大损失",
        "实际外部入金/出金不会制造收益或回撤，未实际划出的计划用款不进入外部流",
        "不同频率没有新信号时沿用其上一有效目标，而不是每天重新调仓",
        "连续相同的 `HOLD`、`NO_ACTION` 或 `WAIT` 不重复推送",
        "佣金、税费、滑点和冲击已经进入换仓成本，不能又重复塞入 no-trade band",
        "所有风险指标必须标记 `risk_scope=selected_account_only`",
        "未导入的其他券商账户、基金、现金、房产、负债和未来收入不在计算范围内，"
        "所选模拟/影子账户的回撤和集中度不能描述成个人或家庭总资产风险",
        "信号在不同延迟下重新执行，得到衰减曲线；"
        "提醒中的有效期和最大人工响应延迟取自预注册的可接受衰减边界",
        "连续相同的 `HOLD`、`NO_ACTION` 或 `WAIT` 不重复推送。硬风险提醒不能被普通免打扰设置屏蔽",
    ):
        assert required in specification

    for obsolete in (
        "`drawdown_nav_contract_version`",
        "`investment_drawdown_nav`",
        "`spendable_wealth_path`",
        "`rolling_decision_consistency_contract_version`",
        "`research_process_benchmark_contract_version`",
    ):
        assert obsolete not in specification


def test_account_actions_and_execution_states_are_synchronized_across_surfaces() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")
    definitions = specification.partition("### 8.4 推荐动作与执行状态")[2].partition(
        "### 8.5 推荐内容"
    )[0]
    reminders = specification.partition("## 10. 长线、中线、短线的盘后提醒")[2].partition(
        "## 11. 最小工程与运行保障"
    )[0]
    acceptance = specification.partition("### 12.4 模拟盘与推荐测试")[2].partition(
        "### 12.5 端到端验收"
    )[0]

    assert "账户动作与执行状态是两个维度" in definitions
    for token in ("`BUY`", "`SELL`", "`EXIT`", "`HOLD`", "`NO_ACTION`"):
        assert token in definitions
    for token in ("`READY`", "`WAIT`", "`PARTIAL`", "`CANCELLED`", "`EXPIRED`", "`BLOCKED`"):
        assert token in definitions

    assert (
        "`BUY/SELL/EXIT/HOLD/NO_ACTION` 账户动作与 "
        "`READY/WAIT/PARTIAL/CANCELLED/EXPIRED/BLOCKED` 执行状态分别保存，语义互不混淆"
        in acceptance
    )

    assert "`WAIT` 变为可执行，或可执行变为阻断" in reminders
    assert "连续相同的 `HOLD`、`NO_ACTION` 或 `WAIT` 不重复推送" in reminders


def test_intraday_governance_and_fixed_stage_order_are_explicit() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")

    for required in (
        "当前没有盘中数据接口，不提供盘中异动、分钟 Alpha 或分时执行点",
        "历史分钟数据只能离线回放，不能改变此产品边界",
        "当前没有盘中调度",
        "D 日完整收盘数据 → D+1 开盘或保守日线成交模型",
        "不能显示“盘中异动”“实时埋伏”或分钟买卖点",
        "盘后分钟数据将来可以用于可选成交质量复算",
        "没有实时分钟数据时，盘中提醒和调度必须完全关闭",
    ):
        assert required in specification

    for obsolete in (
        "`intraday_execution_reminder_required`",
        "`independent_intraday_alpha_required`",
        "`stock_intraday_candidate_cap`",
        "默认关闭的 QMT 沙箱插件是唯一例外",
    ):
        assert obsolete not in specification

    for stage in (
        "| 1. 数据基础 |",
        "| 2. 策略与自主研究框架 |",
        "| 3. 正式验证 |",
        "| 4. 模拟基础 |",
        "| 5. 基线、ETF 与账户政策 |",
        "| 6. 股票横截面模板 |",
        "| 7. 趋势与战术模板 |",
        "| 8. 条件模板与挑战者 |",
        "| 9. 推荐与执行提醒 |",
    ):
        assert stage in specification
    assert "阶段顺序表示依赖关系，不再把策略素材缩减为一个示例" in specification


def test_retired_documents_are_absent_and_unreferenced() -> None:
    retained_text = "\n".join(
        (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            Path("README.md"),
            Path("docs/DEPLOYMENT.md"),
            Path(MARKDOWN_NAME),
        )
    )
    for retired in RETIRED_DOCUMENTS:
        assert not (PROJECT_ROOT / retired).exists()
        assert retired not in retained_text


def test_retained_documents_do_not_reactivate_retired_execution_paths() -> None:
    documents = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs/DEPLOYMENT.md",
        PROJECT_ROOT / MARKDOWN_NAME,
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in documents)
    lowered = text.lower()
    assert "/api/portfolios" not in lowered
    for paragraph in re.split(r"\n\s*\n", text):
        lowered_paragraph = paragraph.lower()
        if any(
            marker in lowered_paragraph
            for marker in ("paper/pair_paper", "pair_paper", "paper portfolio")
        ):
            assert any(
                negation in paragraph
                for negation in ("不恢复", "不得", "不作为", "禁止")
            )

    assert "生产发布不打包 QMT 或任何券商网关" in text
    assert "生产发布没有 QMT 或任何券商网关例外" in text
    for paragraph in re.split(r"\n\s*\n", text):
        if "发单" in paragraph or ("券商" in paragraph and "订单" in paragraph):
            assert any(
                negation in paragraph
                for negation in (
                    "不得", "不提供", "不执行", "不连接", "禁止", "不能", "不会", "永不"
                )
            )


def test_authoritative_design_requires_bounded_daily_data_recovery() -> None:
    text = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")

    assert "只合并补跑截至当前的最新已到期工作日数据槽" in text
    assert "非主行情任务超过 `misfire grace` 仍 `fail-closed`" in text


@pytest.mark.parametrize(
    "relative", ["README.md", "docs/DEPLOYMENT.md", MARKDOWN_NAME]
)
def test_local_markdown_links_exist(relative: str) -> None:
    document = PROJECT_ROOT / relative
    targets = _local_markdown_targets(document)
    assert targets, f"{relative} should link to an authoritative local document"
    missing = [str(path) for path in targets if not path.exists()]
    assert not missing
