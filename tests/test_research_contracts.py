from __future__ import annotations

import pytest

from quant_platform.research_contracts import (
    RESEARCH_MODES,
    STRATEGY_SLOT_ORDER,
    freeze_research_brief,
    research_mode_catalog,
    validate_strategy_spec_slots,
)

pytestmark = pytest.mark.no_database


def _brief() -> dict:
    return {
        "research_question": "Does a PIT momentum component improve the baseline?",
        "economic_gap": "The baseline has no medium-term trend exposure.",
        "allowed_strategy_families": ["cross_sectional_alpha"],
        "allowed_assets": ["A-share"],
        "allowed_frequencies": ["day"],
        "allowed_data": ["qlib-daily-v1"],
        "prohibited_actions": [
            "production_database_write",
            "broker_order_submission",
            "final_oos_visibility_during_selection",
            "secret_access",
        ],
        "personal_constraints": {"capital": 1_000_000},
        "delivery_status": "research_only",
        "dataset_snapshot": {"identity": "a" * 64},
        "pit_rules": {"decision_time_rule": "available_at_before_decision"},
        "periods": {"train": ["2020", "2022"], "test": ["2023", "2024"]},
        "baseline": {"kind": "equal_weight"},
        "primary_metric": "information_ratio",
        "failure_conditions": {"non_finite": "fail"},
        "cost_capacity_risk": {"cost_schedule": "cn-cash-v1"},
        "experiment_budget": {"candidates": 20, "seeds": [0]},
        "runtime_budget": {"duration": "1h", "concurrency": 1},
        "llm_disclosure": {
            "provider": "configured",
            "model": "configured",
            "knowledge_cutoff": "unknown",
            "network_access": False,
        },
        "final_oos_rule": {
            "visible_during_selection": False,
            "one_time_consumption": True,
        },
    }


def test_all_six_research_modes_are_machine_visible_and_non_capital() -> None:
    catalog = research_mode_catalog()
    assert len(catalog) == 6
    assert {item["research_mode"] for item in catalog} == set(RESEARCH_MODES)
    assert all(item["capital_eligible"] is False for item in catalog)
    assert {
        item["status"] for item in catalog
    } == {"runnable", "registered_requires_specialized_runner"}


@pytest.mark.parametrize("mode", sorted(RESEARCH_MODES))
def test_every_research_mode_can_freeze_a_complete_brief(mode: str) -> None:
    frozen = freeze_research_brief(mode, _brief())
    assert frozen["research_mode"] == mode
    assert len(frozen["brief_sha256"]) == 64


def test_brief_fails_closed_without_prohibitions() -> None:
    brief = _brief()
    brief["prohibited_actions"] = ["broker_order_submission"]
    with pytest.raises(ValueError, match="explicitly prohibit"):
        freeze_research_brief("component_discovery", brief)


def test_strategy_spec_slots_require_fixed_order_and_explicit_empty_behavior() -> None:
    slots = {
        name: {
            "required": name
            in {"eligibility_gate", "alpha_rank", "portfolio_risk", "execution_requirement"},
            "components": (
                [{"component_id": f"{name}-v1"}]
                if name
                in {
                    "eligibility_gate",
                    "alpha_rank",
                    "portfolio_risk",
                    "execution_requirement",
                }
                else []
            ),
            "empty_behavior": "identity",
        }
        for name in STRATEGY_SLOT_ORDER
    }
    result = validate_strategy_spec_slots(slots)
    assert result["control_order"] == list(STRATEGY_SLOT_ORDER)
    assert len(result["slots_sha256"]) == 64

    reordered = dict(reversed(list(slots.items())))
    with pytest.raises(ValueError, match="frozen eight-slot order"):
        validate_strategy_spec_slots(reordered)
