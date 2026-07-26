from __future__ import annotations

import pytest

from quant_platform.strategy_catalog import (
    CATALOG_ROLES,
    IMPLEMENTATION_STATUSES,
    IMPLEMENTATION_TIERS,
    catalog_entries_by_status,
    get_catalog_entry,
    list_recipe_catalog,
    validate_recipe_catalog,
)
from quant_platform.strategy_recipes import list_strategy_recipes

pytestmark = pytest.mark.no_database


def test_catalog_registers_implemented_recipes() -> None:
    catalog = list_recipe_catalog()
    by_template = {entry["template_id"]: entry for entry in catalog}
    implemented = {
        entry["recipe_id"]
        for entry in catalog
        if entry["implementation_status"] == "implemented"
    }
    assert implemented == {item["id"] for item in list_strategy_recipes()}
    assert by_template["index_enhancement"]["catalog_role"] == "alpha_template"
    assert by_template["index_enhancement"]["implementation_tier"] == "standard"
    assert by_template["swing_trend"]["implementation_status"] == "implemented"


def test_conditional_templates_stay_visible_and_blocked() -> None:
    catalog = list_recipe_catalog()
    by_template = {entry["template_id"]: entry for entry in catalog}
    for template_id in (
        "industry_rotation",
        "event_driven_swing",
        "minute_mean_reversion_etf",
        "independent_minute_alpha",
        "ml_event_probability",
    ):
        entry = by_template[template_id]
        assert entry["implementation_tier"] == "conditional"
        assert entry["implementation_status"] == "blocked_by_data_or_permission"
        assert entry["blocked_reason"]
        assert entry["name"]


def test_research_only_pair_arb_is_registered_not_blocked() -> None:
    entry = get_catalog_entry("stock_pair_stat_arb")
    assert entry["catalog_role"] == "research_only"
    assert entry["implementation_tier"] == "conditional"
    assert entry["implementation_status"] == "research"


def test_catalog_contract_validates() -> None:
    validate_recipe_catalog()
    for entry in list_recipe_catalog():
        assert entry["catalog_role"] in CATALOG_ROLES
        assert entry["implementation_tier"] in IMPLEMENTATION_TIERS
        assert entry["implementation_status"] in IMPLEMENTATION_STATUSES


def test_catalog_query_interfaces() -> None:
    blocked = catalog_entries_by_status("blocked_by_data_or_permission")
    assert blocked
    assert all(
        entry["implementation_status"] == "blocked_by_data_or_permission"
        for entry in blocked
    )
    research = catalog_entries_by_status("research")
    assert any(entry["template_id"] == "personal_stock_core" for entry in research)
    with pytest.raises(KeyError):
        get_catalog_entry("missing_template")
    with pytest.raises(ValueError, match="unknown implementation status"):
        catalog_entries_by_status("deployed")
