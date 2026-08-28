from __future__ import annotations

from pathlib import Path

import pytest

from quant_platform.simulation_store import paper_target_adjustments

pytestmark = pytest.mark.no_database

ROOT = Path(__file__).resolve().parents[1]


def test_paper_target_adjustments_report_only_factual_frozen_weight_deltas() -> None:
    rows = paper_target_adjustments(
        {"sh600000": 0.25, "SZ000001": 0.10, "SH600519": 0.0},
        {"SH600000": 0.10, "SZ000002": 0.20, "SH600519": 0.0},
    )
    by_instrument = {row["instrument"]: row for row in rows}

    assert by_instrument["SH600000"]["action"] == "increase"
    assert by_instrument["SH600000"]["weight_change"] == pytest.approx(0.15)
    assert by_instrument["SZ000001"]["action"] == "add"
    assert by_instrument["SZ000002"]["action"] == "remove"
    assert by_instrument["SH600519"]["action"] == "hold"
    assert {
        row["reason_basis"] for row in rows
    } == {"frozen_qlib_order_plan_weight_delta"}
    assert all("prediction" not in row["reason"] for row in rows)


def test_daily_paper_target_endpoint_is_read_only_and_not_a_recommendation_path() -> None:
    api = (ROOT / "src" / "quant_platform" / "api.py").read_text(encoding="utf-8")
    store = (ROOT / "src" / "quant_platform" / "simulation_store.py").read_text(
        encoding="utf-8"
    )
    client = (ROOT / "web" / "app" / "api-client.ts").read_text(encoding="utf-8")
    autopilot = (ROOT / "web" / "app" / "autopilot-panel.tsx").read_text(
        encoding="utf-8"
    )
    portfolio = (ROOT / "web" / "app" / "portfolio-panel.tsx").read_text(
        encoding="utf-8"
    )

    assert '@app.get("/api/autopilot/paper-target")' in api
    assert "simulations.current_autopilot_paper_target()" in api
    assert "PAPER_TARGET_PROJECTION_VERSION" in store
    assert "waiting_for_paper_account" in store
    assert "waiting_for_order_plan" in store
    assert "blocked_invalid_order_plan" in store
    assert "_require_governed_long_only_target" in store
    assert "recommendation_enabled\": False" in store
    assert "real_trading_eligible\": False" in store
    assert "export type PaperTargetProjection" in client
    for source in (autopilot, portfolio):
        assert "/api/autopilot/paper-target" in source
        assert "今日模拟候选与目标权重" in source
    assert "不会创建正式推荐" in autopilot
    assert "RecommendationStore" in portfolio
