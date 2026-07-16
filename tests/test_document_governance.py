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
MARKDOWN_NAME = "如何搭建自己的量化交易系统.md"
DOCX_NAME = "如何搭建自己的量化交易系统.docx"
DOCX_BACKUP_NAME = "如何搭建自己的量化交易系统-GitHub原版副本.docx"
DOCX_SHA256 = "d35136f6546c05cfd8ee998d3716cac430590116ad4e1f32227c1c4759d7143a"
DOCX_BACKUP_SHA256 = "efb59da8b2296b6eec6fb9c81e621bb8c8f45d426fa7bf8a84a6d304531ec50a"
CONTROLLED_DOCUMENTS = {
    Path("README.md"),
    Path("docs/DEPLOYMENT.md"),
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
IGNORED_ARTIFACT_PARTS = {
    ".git",
    ".venv",
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
                part in IGNORED_ARTIFACT_PARTS
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
        if path.name != DOCX_BACKUP_NAME
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
    assert "Tushare 不可变快照" in readme
    assert "统一模拟交易" in readme
    assert "本 README 只提供项目入口和" in readme
    assert "不定义另一套产品方案" in readme
    assert "根目录同名 DOCX 仅保留为 3.0 定稿发布快照" in readme
    assert "日常修订不依赖 LibreOffice" in readme
    assert f"`{DOCX_BACKUP_NAME}` 仅保留为冻结的历史原稿" in readme


def test_authoritative_markdown_contains_the_current_contract() -> None:
    specification = (PROJECT_ROOT / MARKDOWN_NAME).read_text(encoding="utf-8")
    required_contracts = (
        "人民币 500 万元",
        "样本外长度",
        "Sharpe Ratio",
        "成交额 0.03%，单笔最低 5 元",
        "日线 `vol`",
        "`adj_factor`",
        "停牌/退市",
        "`cost_model_version`",
        "Benjamini-Hochberg",
        "Deflated Sharpe Probability",
        "年化换手率",
        "暂停、恢复与淘汰",
        "`retired`",
        "买卖推荐",
        "`NO_ACTION`",
        "统一模拟交易",
        "运行责任分工",
        "RD-Agent 是策略研究员",
        "RD-Agent 调用 Qlib 的自动研究是唯一受治理的自动挑战者入口",
        "Qlib 在完整流程中承担两次不同角色",
        "研究与晋升链",
        "正式生产链",
        "固定运行日历",
        "每月最后一个交易日 18:00",
        "每周最后一个交易日 18:15",
        "每周五 20:00 或周末",
        "盘中每个完整 1min/5min Bar 后",
        "1分钟/5分钟策略与5分钟执行的区别",
        "Tushare 实时分钟接口",
        "下一可执行 Bar",
        "RD-Agent 不参与上述盘中循环",
        "项目研究组件库与基于 Qlib 的正式策略模板",
        "双均线、均线结构、突破、ADX、MACD",
        "布林带位置/宽度、RSI、乖离率",
        "`component_id`",
        "`slot_bindings`",
        "`decision_frequency`",
        "`rebalance_frequency`",
        "`monitor_frequency`",
        "`eligibility_gate`",
        "`universe_dedup`",
        "`direction_regime_gate`",
        "`etf_asset_allocation`（ETF 资产配置与选择）",
        "`index_enhancement`（指数增强）",
        "`weekly_tactical_overlay`（周频股票/ETF 战术覆盖）",
        "`pair` 是项目专用双腿策略模板",
        "`runtime_enabled=false`",
        "`approval_status=approved` 且 `runtime_enabled=true`",
        "全部进入可测试、可版本化的研究组件库",
        "不得直接修改正在运行的正式模板",
        "在冻结模板的允许扩展点内做研究",
        "Walk-forward 必须采用嵌套结构",
        "PBO > 20%",
        "策略级信号明细继续保留",
        "唯一的 `final_target_weight`",
        "`integrated_recommendation`",
        "`pending_target_weight`",
        "`evaluation_event`",
        "`decision_event`",
        "`notification_event`",
        "`notification_dedup_key`",
        "`score_semantics`",
        "`benchmark_evidence_type`",
        "`official_total_return_index`",
        "`synthetic_total_return_proxy`",
        "`investable_benchmark_portfolio`",
        "`human_execution_policy`",
        "`account_state_asof`",
        "`execution_evidence_type`",
        "Implementation Shortfall",
        "`P(MDD > 10%)`",
        "50 万元、100 万元和 500 万元",
        "确定性虚拟市场时钟",
        "连续相同的等待、`HOLD` 或 `NO_ACTION` 不得重复创建通知",
        "不要求四种频率的 Alpha 策略同时启用",
        "连续 60 个交易日的前向模拟",
        "券商连接、真实账户、真实资金与真实委托",
    )
    for contract in required_contracts:
        assert contract in specification

    assert "本 Markdown 是 4.4 最终架构与产品基准" in specification
    assert "根目录同名 DOCX 是 3.0 发布快照" in specification


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

    assert "QMT 仅作为默认关闭的可选插件" in text
    assert "页面、调度和模拟任务不得向 QMT 或任何券商网关发单" in text
    for paragraph in re.split(r"\n\s*\n", text):
        if "发单" in paragraph or ("券商" in paragraph and "订单" in paragraph):
            assert any(
                negation in paragraph
                for negation in ("不得", "不提供", "不执行", "不连接", "禁止")
            )


@pytest.mark.parametrize("relative", ["README.md", "docs/DEPLOYMENT.md"])
def test_local_markdown_links_exist(relative: str) -> None:
    document = PROJECT_ROOT / relative
    targets = _local_markdown_targets(document)
    assert targets, f"{relative} should link to an authoritative local document"
    missing = [str(path) for path in targets if not path.exists()]
    assert not missing
