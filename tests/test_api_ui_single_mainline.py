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
RDAGENT_PANEL_SOURCE = (ROOT / "web" / "app" / "rdagent-panel.tsx").read_text(
    encoding="utf-8"
)
QLIB_PANEL_SOURCE = (ROOT / "web" / "app" / "qlib-panel.tsx").read_text(
    encoding="utf-8"
)
RDAGENT_BRIDGE_SOURCE = (ROOT / "scripts" / "rdagent_bridge.py").read_text(
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


def test_runtime_status_recovers_without_serving_two_layers_of_stale_failure() -> None:
    assert API_SOURCE.count(
        'current_status in {"checking", "unavailable"}'
    ) == 2
    assert '/api/rdagent/status`, { cache: "no-store", forceRefresh: true }' in (
        RDAGENT_PANEL_SOURCE
    )
    assert '/api/qlib/status`, { cache: "no-store", forceRefresh: true }' in (
        QLIB_PANEL_SOURCE
    )


def test_advanced_web_reads_a_sanitized_official_trace_projection() -> None:
    assert '"trace_contract_version": "rdagent-trace-web-v1"' in RDAGENT_BRIDGE_SOURCE
    assert '"trace_loops": _trace_loop_projection(rounds)' in RDAGENT_BRIDGE_SOURCE
    for forbidden in ('"code":', '"code_path":', '"workspace_path":'):
        projection = RDAGENT_BRIDGE_SOURCE.split(
            "def _trace_loop_projection", 1
        )[1].split("def _sha256", 1)[0]
        assert forbidden not in projection
    assert "def _rdagent_trace_view(" in API_SOURCE
    assert "官方 RDLoop / Trace（只读）" in RDAGENT_PANEL_SOURCE
    assert "Hypothesis" in RDAGENT_PANEL_SOURCE
    assert "Feedback" in RDAGENT_PANEL_SOURCE


def test_autopilot_trial_history_is_unified_and_does_not_require_a_tournament() -> None:
    assert 'raise HTTPException(404, "autopilot cycle not found")' in API_SOURCE
    assert "AutopilotTrialAuditService" in API_SOURCE
    assert "list_cycle_trials(cycle)" in API_SOURCE
    assert "does not need a model tournament" in API_SOURCE


def test_legacy_http_execution_surfaces_stay_retired() -> None:
    retired_route = '@app.api_route("/api/portfolios", methods=["GET", "POST"], status_code=410)'
    assert retired_route in API_SOURCE
    assert '"replacement": "/api/recommendation-portfolios"' in API_SOURCE
    assert '"/api/broker' not in API_SOURCE
    assert '"/api/pair-portfolios' not in API_SOURCE


def test_legacy_research_programs_and_campaigns_are_read_only() -> None:
    for retired_schema in (
        "ResearchProgramCreateRequest",
        "ResearchProgramStatusRequest",
        "ResearchCampaignCreateRequest",
        "ResearchCampaignStatusRequest",
    ):
        assert retired_schema not in API_SOURCE
    for route, replacement in (
        ("/api/research-programs", "/api/autopilot"),
        ("/api/research-campaigns", "/api/autopilot"),
    ):
        assert f'@app.get("{route}")' in API_SOURCE
        write_block = API_SOURCE.split(f'@app.post("{route}"', 1)[1].split(
            "@app.", 1
        )[0]
        assert "HTTPException(" in write_block
        assert "410" in write_block
        assert replacement in write_block

    for function_name in (
        "set_research_program_status",
        "check_research_program_now",
        "set_research_campaign_status",
        "retry_research_campaign",
    ):
        block = API_SOURCE.split(f"def {function_name}", 1)[1].split("@app.", 1)[0]
        assert "HTTPException(410" in block
        assert "legacy_research_programs." not in block
        assert "legacy_research_campaigns." not in block


def test_real_broker_gateway_is_not_a_production_build_capability() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    compose = (ROOT / "deploy" / "compose.yaml").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "quant-broker-gateway" not in project
    assert "src/quant_broker_gateway" not in project
    assert "BROKER_" not in compose
    assert "src/quant_broker_gateway" in dockerignore


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
    assert (
        'SIMULATION_EXECUTION_FREQUENCIES = frozenset({"day", "1min", "5min"})'
        in SIMULATION_SOURCE
    )
    for marker in ("quant_broker_gateway", "broker_order_outbox", "pair_paper_orders", "requests."):
        assert marker not in SIMULATION_SOURCE


def test_pair_replay_write_api_is_retired_and_cannot_restart_shorting() -> None:
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
    endpoint = API_SOURCE.split("def create_pair_simulation_replay", 1)[1].split(
        '@app.get("/api/simulation-portfolios/{portfolio_id}")', 1
    )[0]
    assert "HTTPException(" in endpoint
    assert "410" in endpoint
    assert "Autopilot is long-only" in endpoint
    assert "create_pair_batch_from_backtest" not in endpoint
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


def test_pair_shadow_is_not_part_of_the_automatic_capital_line() -> None:
    tick = SCHEDULER_SOURCE.split("def tick", 1)[1].split(
        "def _enqueue_due_factor_library_materialization", 1
    )[0]
    for forbidden in (
        "self._ensure_approved_pair_shadow_accounts(",
        "self._enqueue_due_pair_shadow_backtests(",
        "self._materialize_due_pair_shadow_batches(",
    ):
        assert forbidden not in tick
    for marker in (
        "pair_shadow_accounts_created = 0",
        "pair_shadow_backtests_enqueued = 0",
        "pair_shadow_batches_materialized = 0",
    ):
        assert marker in tick
    for retired_implementation in (
        "def _ensure_approved_pair_shadow_accounts",
        "def _enqueue_due_pair_shadow_backtests",
        "def _materialize_due_pair_shadow_batches",
        'self.jobs.create(\n                    "pair_backtest"',
    ):
        assert retired_implementation not in SCHEDULER_SOURCE
    assert 'if job["kind"] == "pair_backtest":\n            output =' not in WORKER_SOURCE
    assert "scripts/run_pair_backtest.py" not in WORKER_SOURCE


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
    page_source = (ROOT / "web" / "app" / "page.tsx").read_text(encoding="utf-8")
    assert "PairSatellitePanel" not in page_source
    assert "配对卫星" not in page_source
    assert not (ROOT / "web" / "app" / "pair-satellite-panel.tsx").exists()


def test_web_defaults_to_a_safe_autopilot_mainline() -> None:
    page_source = (ROOT / "web" / "app" / "page.tsx").read_text(encoding="utf-8")
    autopilot_source = (ROOT / "web" / "app" / "autopilot-panel.tsx").read_text(
        encoding="utf-8"
    )

    assert 'useState(false)' in page_source
    assert '打开高级管理' in page_source
    assert '<AutopilotPanel' in page_source
    for marker in (
        'SINGLE AUTOPILOT CYCLE',
        '/api/autopilot',
        '/api/data-automation',
        '/api/simulation-portfolios',
        '无需每天点按钮',
        '不连接真实券商',
        '不代表实盘建议',
    ):
        assert marker in autopilot_source
    assert '/api/broker' not in autopilot_source
    assert 'real_trade' not in autopilot_source
