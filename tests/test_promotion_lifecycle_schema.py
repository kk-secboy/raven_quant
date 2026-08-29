from __future__ import annotations

import pytest

from quant_data.database import strategy_versions

pytestmark = pytest.mark.no_database


def _postgresql_predicate(index_name: str) -> str:
    index = next(item for item in strategy_versions.indexes if item.name == index_name)
    assert index.unique is True
    return str(index.dialect_options["postgresql"]["where"])


def test_historical_approval_does_not_claim_unique_recommendation_authority() -> None:
    family_predicate = _postgresql_predicate("uq_strategy_versions_approved")
    assert "status = 'approved'" in family_predicate
    assert "promotion_stage = 'recommendation_enabled'" in family_predicate
    assert "promotion_stage = 'paper'" not in family_predicate

    horizon_predicate = _postgresql_predicate("uq_strategy_versions_active_horizon")
    assert "promotion_stage = 'recommendation_enabled'" in horizon_predicate
    assert "horizon_profile <> 'legacy_ambiguous'" in horizon_predicate
