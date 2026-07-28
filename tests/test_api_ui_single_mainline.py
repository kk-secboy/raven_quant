from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_database

ROOT = Path(__file__).resolve().parents[1]
API_SOURCE = (ROOT / "src" / "quant_platform" / "api.py").read_text(encoding="utf-8")
SCHEDULE_SOURCE = (ROOT / "src" / "quant_platform" / "schedule_store.py").read_text(
    encoding="utf-8"
)
SCHEDULER_SOURCE = (ROOT / "src" / "quant_platform" / "scheduler.py").read_text(encoding="utf-8")
SIMULATION_SOURCE = (ROOT / "src" / "quant_platform" / "simulation_store.py").read_text(
    encoding="utf-8"
)
WORKER_SOURCE = (ROOT / "src" / "quant_platform" / "worker.py").read_text(
    encoding="utf-8"
)


def _class_block(source: str, class_name: str) -> str:
    match = re.search(
        rf"^class {re.escape(class_name)}\b.*?(?=^class \w+\b|^def \w+\b)",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    return match.group(0)


def test_legacy_http_execution_surfaces_stay_retired() -> None:
    retired_route = '@app.api_route("/api/portfolios", methods=["GET", "POST"], status_code=410)'
    assert retired_route in API_SOURCE
    assert '"replacement": "/api/recommendation-portfolios"' in API_SOURCE
    assert '"/api/broker' not in API_SOURCE
    assert '"/api/pair-portfolios' not in API_SOURCE


def test_scheduler_accepts_research_and_data_work_only() -> None:
    block = _class_block(API_SOURCE, "ScheduleCreateRequest")
    expected_kinds = {
        "incremental_sync",
        "data_pipeline",
        "ashare_5m_sync",
        "rdagent_research",
        "recommendation_refresh",
    }
    declared = set(re.findall(r'^\s+"([a-z0-9_]+)",?$', block, flags=re.MULTILINE))
    assert expected_kinds <= declared
    assert not {"broker", "qmt", "order", "paper_rebalance"} & declared

    active_runtime = SCHEDULE_SOURCE + "\n" + SCHEDULER_SOURCE
    for marker in ("quant_broker_gateway", "broker_order", "pair_paper", "QMT", "qmt"):
        assert marker not in active_runtime


def test_unified_simulation_has_only_governed_sources_and_two_adapters() -> None:
    assert (
        'SIMULATION_SOURCE_TYPES = frozenset({"recommendation", "strategy_version", "allocation"})'
        in SIMULATION_SOURCE
    )
    assert 'SIMULATION_EXECUTION_ADAPTERS = frozenset({"long_only", "pair"})' in SIMULATION_SOURCE
    assert 'SIMULATION_EXECUTION_FREQUENCIES = frozenset({"1min", "5min"})' in SIMULATION_SOURCE
    for marker in ("quant_broker_gateway", "broker_order_outbox", "pair_paper_orders", "requests."):
        assert marker not in SIMULATION_SOURCE


def test_pair_replay_api_cannot_accept_client_authored_legs_or_borrow_rates() -> None:
    block = _class_block(API_SOURCE, "PairSimulationReplayRequest")
    assert 'ConfigDict(extra="forbid")' in block
    assert "backtest_id:" in block
    assert "trade_date:" in block
    assert "actor:" in block
    for forbidden in (
        "target_payload",
        "source_snapshot_id",
        "execution_contract_hash",
        "annual_borrow_rate",
        "target_quantity",
    ):
        assert forbidden not in block
    assert (
        '"/api/simulation-portfolios/{portfolio_id}/pair-replays"' in API_SOURCE
    )
    assert (
        "pair simulation batches must be derived from an approved immutable "
        in SIMULATION_SOURCE
    )
    for marker in (
        "resolve_snapshot_dataset(",
        "--shortability-path",
        "--shortability-source-sha256",
        "--shortability-manifest-sha256",
    ):
        assert marker in WORKER_SOURCE


def test_long_only_replay_api_accepts_only_an_immutable_order_plan_identity() -> None:
    block = _class_block(API_SOURCE, "SimulationOrderPlanBatchRequest")
    assert 'ConfigDict(extra="forbid")' in block
    assert "order_plan_manifest_sha256:" in block
    assert "actor:" in block
    for forbidden in (
        "target_payload",
        "target_weights",
        "source_snapshot_id",
        "execution_contract_hash",
        "signal_date",
        "trade_date",
    ):
        assert forbidden not in block
    assert "create_batch_from_order_plan(" in API_SOURCE
    assert "create_batch_for_targets(" not in API_SOURCE
    assert '"/api/simulation-portfolios/{portfolio_id}/order-plans"' in API_SOURCE
    assert '"simulation_order_plan"' in API_SOURCE
    for marker in (
        "qlib_workflow_run(",
        "order_plan_manifest_sha256",
        "target_weights.json",
    ):
        assert marker in (
            ROOT / "scripts" / "run_recommendation_refresh.py"
        ).read_text(encoding="utf-8")
    assert 'if job["kind"] == "simulation_order_plan":' in WORKER_SOURCE


def test_web_uses_only_the_single_mainline_routes() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "web" / "app").glob("*.ts*")
    )
    for marker in ("/api/portfolios", "/api/pair-portfolios", "/api/broker", "settings/broker"):
        assert marker not in sources
    for marker in (
        "/api/rdagent/",
        "/api/factors",
        "/api/strategy-versions/",
        "/api/strategy-allocations",
        "/api/simulation-portfolios",
    ):
        assert marker in sources
    pair_source = (ROOT / "web" / "app" / "pair-satellite-panel.tsx").read_text(
        encoding="utf-8"
    )
    assert "/pair-backtests" in pair_source
    assert "RESEARCH ONLY / NO CAPITAL" in pair_source
    for marker in (
        "/pair-replays",
        "/approve",
        "/api/simulation-portfolios",
        "backtest_id",
        "target_payload",
        "annual_borrow_rate",
    ):
        assert marker not in pair_source
