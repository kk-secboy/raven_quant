from __future__ import annotations

from pathlib import Path

import pytest

from quant_platform.simulation_store import SimulationStore

pytestmark = pytest.mark.no_database

ROOT = Path(__file__).resolve().parents[1]


def test_long_simulation_gate_caps_targets_at_current_exposure() -> None:
    gated = SimulationStore._apply_member_new_risk_gate(
        adapter="long_only",
        target_payload={"target_weights": {"SH600000": 0.50, "SH600001": 0.20}},
        positions={
            "SH600000": {
                "position_side": "long",
                "quantity": 1000,
                "market_value": 300_000,
            }
        },
        portfolio_nav=1_000_000,
    )

    assert gated["target_weights"] == {
        "SH600000": pytest.approx(0.30),
        "SH600001": 0.0,
    }


def test_pair_simulation_gate_preserves_plan_but_only_allows_reduction() -> None:
    plan = {"format_version": "governed-pair-plan-v1", "pair_plan_sha256": "a" * 64}
    gated = SimulationStore._apply_member_new_risk_gate(
        adapter="pair",
        target_payload={
            "atomic_group_id": "pair-1",
            "governed_pair_plan": plan,
            "legs": [
                {
                    "instrument": "SH510300",
                    "leg_no": 1,
                    "position_side": "long",
                    "target_quantity": 1200,
                    "annual_borrow_rate": 0.0,
                },
                {
                    "instrument": "SZ159919",
                    "leg_no": 2,
                    "position_side": "short",
                    "target_quantity": 1000,
                    "annual_borrow_rate": 0.08,
                },
            ],
        },
        positions={
            "SH510300": {
                "position_side": "long",
                "quantity": 1000,
                "market_value": 4_000,
            },
            "SZ159919": {
                "position_side": "short",
                "quantity": 800,
                "market_value": -4_000,
            },
        },
        portfolio_nav=5_000_000,
    )

    assert gated["governed_pair_plan"] == plan
    assert [leg["target_quantity"] for leg in gated["legs"]] == [1000, 800]


def test_allocation_risk_override_scales_long_targets_without_losing_plan() -> None:
    plan = {"manifest_sha256": "a" * 64}
    scaled = SimulationStore._apply_risk_exposure_override(
        adapter="long_only",
        target_payload={
            "target_weights": {"SH600000": 0.40, "SH600001": 0.20},
            "governed_order_plan": plan,
        },
        risk_exposure_override=0.5,
    )

    assert scaled["governed_order_plan"] == plan
    assert scaled["target_weights"] == {
        "SH600000": pytest.approx(0.20),
        "SH600001": pytest.approx(0.10),
    }


def test_allocation_liquidation_zeros_both_pair_legs_atomically() -> None:
    plan = {"pair_plan_sha256": "b" * 64}
    scaled = SimulationStore._apply_risk_exposure_override(
        adapter="pair",
        target_payload={
            "atomic_group_id": "pair-1",
            "governed_pair_plan": plan,
            "legs": [
                {"instrument": "SH510300", "target_quantity": 1100},
                {"instrument": "SZ159919", "target_quantity": 900},
            ],
        },
        risk_exposure_override=0.0,
    )

    assert scaled["governed_pair_plan"] == plan
    assert [leg["target_quantity"] for leg in scaled["legs"]] == [0, 0]


def test_restore_drill_uses_only_unified_simulation_ledger_sentinel() -> None:
    source = (ROOT / "scripts" / "restore_drill.py").read_text(encoding="utf-8")
    assert "simulation_ledger_sentinel" in source
    assert "quantlab.simulation_portfolios" in source
    assert "quantlab.simulation_positions" in source
    assert "quantlab.simulation_nav" in source
    assert "pair_paper" not in source
    assert "pair_portfolio_" not in source


def test_0037_downgrade_blocks_new_simulation_records_without_deleting_them() -> None:
    source = (
        ROOT / "migrations" / "versions" / "0037_single_mainline_contract.py"
    ).read_text(encoding="utf-8")
    assert "0037 downgrade blocked" in source
    assert "signal_at IS NOT NULL" in source
    assert "execution_not_before IS NOT NULL" in source
    assert "HAVING COUNT(*) > 1" in source
    assert "status = 'paused'" in source
    assert "execution_contract_hash = 'legacy-unversioned'" in source
    assert "DELETE FROM quantlab.simulation_batches" not in source
    assert "DELETE FROM quantlab.simulation_portfolios" not in source
