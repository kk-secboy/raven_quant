from __future__ import annotations

from datetime import UTC, datetime, timedelta

from quant_platform.recommendation_account_store import (
    ACCOUNT_MANUAL_SHADOW,
    RecommendationAccountStore,
)
from quant_platform.shadow_account import ShadowAccountStore


def test_manual_shadow_selection_is_versioned_and_never_falls_back_when_stale(
    database_url: str,
) -> None:
    imported_at = datetime(2026, 7, 29, 8, 0, tzinfo=UTC)
    ShadowAccountStore(database_url).import_snapshot(
        account_id="personal-shadow-1",
        cash=25_000,
        holdings=[
            {
                "instrument": "SH600000",
                "quantity": 100,
                "sellable_quantity": 100,
            }
        ],
        imported_by="operator",
        imported_at=imported_at,
    )

    selected = RecommendationAccountStore(database_url).select(
        recommendation_portfolio_id="recommendation-1",
        account_type=ACCOUNT_MANUAL_SHADOW,
        account_id="personal-shadow-1",
        actor="operator",
        reason="Use the freshly imported personal shadow account.",
        now=imported_at,
    )

    assert selected["status"] == "selected"
    assert selected["selected_via"] == "explicit"
    assert selected["config_revision"] == 1
    assert selected["account_type"] == ACCOUNT_MANUAL_SHADOW
    assert selected["account_id"] == "personal-shadow-1"
    assert selected["degraded"] is False

    stale = RecommendationAccountStore(database_url).resolve(
        "recommendation-1",
        now=imported_at + timedelta(days=3),
    )
    assert stale["selected_via"] == "explicit"
    assert stale["account_type"] == ACCOUNT_MANUAL_SHADOW
    assert stale["account_id"] == "personal-shadow-1"
    assert stale["degraded"] is True
    assert stale["reasons"] == ["selected_manual_shadow_stale"]
