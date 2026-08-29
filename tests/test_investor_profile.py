from __future__ import annotations

from datetime import date

import pytest

from quant_platform.investor_profile import (
    InvestorSimulationProfileStore,
    investor_profile_permission,
)


def _permissions() -> dict[str, bool]:
    return {
        "main_board": True,
        "star_market": False,
        "chi_next": False,
        "beijing_exchange": False,
        "etf": True,
    }


@pytest.mark.no_database
def test_first_run_permissions_fail_closed_for_new_risk_by_board() -> None:
    profile = {"market_permissions": _permissions()}

    assert investor_profile_permission(
        profile, "600000.SH", on_date=date(2026, 8, 28)
    )["allowed"] is True
    assert investor_profile_permission(
        profile, "510300.SH", on_date=date(2026, 8, 28)
    )["allowed"] is True
    star = investor_profile_permission(
        profile, "688001.SH", on_date=date(2026, 8, 28)
    )
    chinext = investor_profile_permission(
        profile, "300001.SZ", on_date=date(2026, 8, 28)
    )
    bse = investor_profile_permission(
        profile, "830001.BJ", on_date=date(2026, 8, 28)
    )
    unknown = investor_profile_permission(
        profile, "XYZ", on_date=date(2026, 8, 28)
    )

    assert (star["allowed"], star["permission_key"]) == (False, "star_market")
    assert (chinext["allowed"], chinext["permission_key"]) == (False, "chi_next")
    assert (bse["allowed"], bse["permission_key"]) == (False, "beijing_exchange")
    assert unknown == {
        "allowed": False,
        "permission_key": None,
        "reason": "instrument_permission_scope_unknown",
    }


def test_investor_simulation_profile_requires_explicit_capital_and_versions_changes(
    database_url: str,
) -> None:
    store = InvestorSimulationProfileStore(database_url)
    first = store.create_version(
        profile_key="personal-main",
        initial_capital="250000",
        risk_profile="balanced",
        min_cash_weight=0.10,
        max_gross_exposure=0.90,
        market_permissions=_permissions(),
        actor="investor-owner",
        activate=True,
    )
    second = store.create_version(
        profile_key="personal-main",
        initial_capital="300000",
        risk_profile="conservative",
        min_cash_weight=0.20,
        max_gross_exposure=0.80,
        market_permissions=_permissions(),
        actor="investor-owner",
    )

    assert first["version"] == 1
    assert first["initial_capital"] == 250000.0
    assert second["version"] == 2
    assert second["supersedes_id"] == first["id"]
    assert store.get_active("personal-main")["id"] == first["id"]  # type: ignore[index]

    activated = store.activate(second["id"], actor="investor-owner")
    assert activated["status"] == "active"
    assert store.get(first["id"])["status"] == "retired"
    assert [item["version"] for item in store.list_versions("personal-main")] == [2, 1]


@pytest.mark.parametrize("capital", [0, -1, "NaN", "Infinity"])
def test_investor_profile_has_no_implicit_or_invalid_capital(
    database_url: str, capital: object
) -> None:
    with pytest.raises(ValueError, match="initial capital"):
        InvestorSimulationProfileStore(database_url).create_version(
            profile_key="personal-main",
            initial_capital=capital,  # type: ignore[arg-type]
            risk_profile="balanced",
            min_cash_weight=0.10,
            max_gross_exposure=0.90,
            market_permissions=_permissions(),
            actor="investor-owner",
        )
