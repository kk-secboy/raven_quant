from pathlib import Path

import pytest

pytestmark = pytest.mark.no_database

ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_old_autopilot_capital_pipeline_is_not_a_runtime_entry() -> None:
    autopilot = _source("src/quant_platform/autopilot.py")
    api = _source("src/quant_platform/api.py")
    scheduler = _source("src/quant_platform/scheduler.py")
    legacy = _source("src/quant_platform/autopilot_capital_pipeline.py")

    for runtime in (autopilot, api, scheduler):
        assert "AutopilotCapitalPipeline(" not in runtime
        assert ".capital_pipeline" not in runtime
        assert "_advance_capital_cycle" not in runtime
    assert 'CAPITAL_PIPELINE_LIFECYCLE = "legacy_readonly"' in legacy
    assert "Current" in legacy
    assert "must not instantiate or call this class" in legacy


def test_read_only_champion_selection_feeds_only_fin_strategy() -> None:
    selector = _source("src/quant_platform/autopilot_champion_selection.py")
    autopilot = _source("src/quant_platform/autopilot.py")
    api = _source("src/quant_platform/api.py")
    scheduler = _source("src/quant_platform/scheduler.py")
    worker = _source("src/quant_platform/worker.py")

    assert "class AutopilotResearchChampionSelector" in selector
    assert "gains no StrategyVersion, formal-OOS, approval, or paper authority" in selector
    assert "self.champion_selector = AutopilotResearchChampionSelector(" in autopilot
    assert "autopilot.champion_selector.select_champion(" in api
    assert "self.autopilot.champion_selector.select_champion(" in scheduler
    assert "def _materialize_fin_strategy_candidate(" in worker
    assert "def _queue_fin_strategy_formal_oos(" in worker
    assert "def _settle_fin_strategy_formal_research(" in worker


def test_status_and_authoritative_docs_name_the_single_capital_entry() -> None:
    api = _source("src/quant_platform/api.py")
    web = _source("web/app/autopilot-panel.tsx")
    readme = _source("README.md")
    specification = _source("个人量化投资与模拟盘系统设计稿.md")

    assert '"automatic_entry": "fin_strategy_settlement"' in api
    assert '"legacy_autopilot_capital_pipeline": "legacy_readonly"' in api
    assert "资本入口：fin_strategy 结算" in web
    assert "旧 Autopilot 资本链不会启动" in web
    for document in (readme, specification):
        assert "AutopilotCapitalPipeline" in document
        assert "legacy_readonly" in document
        assert "唯一自动资本入口" in document
