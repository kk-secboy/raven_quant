from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from quant_platform.recommendation_account_store import (
    ACCOUNT_MAIN_PAPER,
    ACCOUNT_MANUAL_SHADOW,
    RecommendationAccountStore,
)

pytestmark = pytest.mark.no_database
NOW = datetime(2026, 7, 29, 8, 0, tzinfo=UTC)


class FakeConfigs:
    def __init__(self) -> None:
        self.records = {}

    def get(self, key: str):
        return self.records.get(key)

    def put(self, key: str, value: dict, *, actor: str, reason: str):
        assert actor
        assert len(reason) >= 10
        current = self.records.get(key)
        record = {"revision": int((current or {}).get("revision") or 0) + 1, "value": value}
        self.records[key] = record
        return record


class FakeSimulations:
    def __init__(self, accounts: list[dict] | None = None) -> None:
        self.accounts = accounts or []
        self.required_nav_date = None

    def list(self, _limit: int) -> list[dict]:
        return list(self.accounts)

    def get(self, account_id: str) -> dict:
        account = next(
            (item for item in self.accounts if item["id"] == account_id),
            None,
        )
        if account is None:
            raise KeyError(account_id)
        return dict(account)

    def policy_risk_inputs(self, account_id: str, *, required_nav_date: date):
        self.required_nav_date = required_nav_date
        return {
            "status": "certified",
            "portfolio_id": account_id,
            "portfolio_drawdown": -0.05,
            "daily_return": -0.01,
            "allow_new_risk": True,
            "nav_trade_date": required_nav_date.isoformat(),
        }

    def rows(self, account_id: str, resource: str) -> list[dict]:
        if self.get(account_id).get("empty"):
            return []
        if resource == "positions":
            return [
                {
                    "instrument": "SH600000",
                    "quantity": 200,
                    "free_sellable_quantity": 100,
                    "market_value": 2_000.0,
                }
            ]
        if resource == "orders":
            return [
                {
                    "id": "order-1",
                    "instrument": "SH600000",
                    "side": "buy",
                    "status": "open",
                    "requested_quantity": 100,
                    "filled_quantity": 0,
                    "expires_at": None,
                    "created_at": NOW,
                }
            ]
        raise AssertionError(resource)


class FakeShadows:
    def __init__(self, status: str = "fresh") -> None:
        self.status = status

    def freshness(self, account_id: str, *, now=None) -> dict:
        return {
            "account_id": account_id,
            "status": self.status,
            "imported_at": NOW.isoformat(),
            "age_days": 0 if self.status == "fresh" else 5,
            "stale_after_days": 2,
        }

    def account_state_for_actions(self, account_id: str, *, now=None) -> dict:
        freshness = self.freshness(account_id, now=now)
        return {
            "account_state": {
                "SH600000": {
                    "filled_position": 100,
                    "sellable_quantity": 100,
                    "open_orders": [],
                }
            },
            "account_context": {
                "account_type": ACCOUNT_MANUAL_SHADOW,
                "account_id": account_id,
                "degraded": freshness["status"] != "fresh",
                "freshness": freshness,
            },
            "cash": 5_000.0,
        }


def _main(account_id: str = "paper-1", *, empty: bool = False) -> dict:
    return {
        "id": account_id,
        "source_type": "recommendation",
        "source_id": "recommendation-1",
        "status": "active",
        "nav": 10_000.0,
        "empty": empty,
    }


def _store(
    *,
    accounts: list[dict] | None = None,
    shadow_status: str = "fresh",
) -> RecommendationAccountStore:
    return RecommendationAccountStore(
        "unused",
        configs=FakeConfigs(),
        simulations=FakeSimulations(accounts),
        shadows=FakeShadows(shadow_status),
    )


def test_sole_linked_main_paper_is_the_default_selected_account() -> None:
    selected = _store(accounts=[_main()]).resolve("recommendation-1", now=NOW)

    assert selected["status"] == "selected"
    assert selected["selected_via"] == "default"
    assert selected["account_type"] == ACCOUNT_MAIN_PAPER
    assert selected["account_id"] == "paper-1"


def test_multiple_main_paper_ledgers_fail_closed_until_explicit_selection() -> None:
    selected = _store(accounts=[_main("paper-1"), _main("paper-2")]).resolve(
        "recommendation-1",
        now=NOW,
    )

    assert selected["status"] == "ambiguous"
    assert selected["degraded"] is True
    assert selected["candidate_account_ids"] == ["paper-1", "paper-2"]


def test_stale_explicit_shadow_never_silently_falls_back_to_main_paper() -> None:
    configs = FakeConfigs()
    store = RecommendationAccountStore(
        "unused",
        configs=configs,
        simulations=FakeSimulations([_main()]),
        shadows=FakeShadows("fresh"),
    )
    store.select(
        recommendation_portfolio_id="recommendation-1",
        account_type=ACCOUNT_MANUAL_SHADOW,
        account_id="shadow-1",
        actor="operator",
        reason="Select the freshly imported personal shadow account.",
        now=NOW,
    )
    store.shadows.status = "stale"

    selected = store.resolve("recommendation-1", now=NOW)

    assert selected["account_type"] == ACCOUNT_MANUAL_SHADOW
    assert selected["account_id"] == "shadow-1"
    assert selected["degraded"] is True
    assert selected["reasons"] == ["selected_manual_shadow_stale"]


def test_account_selections_are_independent_per_recommendation_portfolio() -> None:
    configs = FakeConfigs()
    store = RecommendationAccountStore(
        "unused",
        configs=configs,
        simulations=FakeSimulations(),
        shadows=FakeShadows("fresh"),
    )
    for portfolio_id, account_id in (
        ("recommendation-1", "shadow-1"),
        ("recommendation-2", "shadow-2"),
    ):
        store.select(
            recommendation_portfolio_id=portfolio_id,
            account_type=ACCOUNT_MANUAL_SHADOW,
            account_id=account_id,
            actor="operator",
            reason="Select an independent account for this recommendation portfolio.",
            now=NOW,
        )

    assert store.resolve("recommendation-1", now=NOW)["account_id"] == "shadow-1"
    assert store.resolve("recommendation-2", now=NOW)["account_id"] == "shadow-2"


def test_main_paper_risk_and_action_state_use_selected_ledger_nav() -> None:
    store = _store(accounts=[_main()])
    risk = store.policy_risk_inputs(
        "recommendation-1",
        required_nav_date=date(2026, 7, 29),
        now=NOW,
    )
    state = store.account_state_for_actions(
        "recommendation-1",
        reference_prices={"SH600000": 10.0},
        account_risk_state=risk,
        now=NOW,
    )

    assert store.simulations.required_nav_date == date(2026, 7, 29)
    assert state["account_value"] == pytest.approx(10_000.0)
    assert state["account_context"]["account_type"] == ACCOUNT_MAIN_PAPER
    assert state["account_state"]["SH600000"]["filled_position"] == 200
    assert state["account_state"]["SH600000"]["sellable_quantity"] == 100
    assert state["account_state"]["SH600000"]["open_orders"][0]["order_id"] == "order-1"
    assert state["risk_assessment"]["risk_scope"] == "selected_account_only"


def test_initial_empty_main_paper_uses_known_initial_nav_for_first_target() -> None:
    store = _store(accounts=[_main(empty=True)])

    state = store.account_state_for_actions(
        "recommendation-1",
        reference_prices={"SH600000": 10.0},
        account_risk_state={
            "status": "initial_empty_account",
            "allow_new_risk": True,
            "portfolio_drawdown": 0.0,
        },
        now=NOW,
    )

    assert state["account_value"] == pytest.approx(10_000.0)
    assert state["account_context"]["degraded"] is False
    assert state["risk_assessment"]["risk_state"] == "normal"
