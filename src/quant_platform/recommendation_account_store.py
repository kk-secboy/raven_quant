"""One selected account for account-level recommendations.

The default is the sole active simulation ledger linked to a recommendation
portfolio (``main_paper``).  A user may explicitly select a fresh imported
``manual_shadow`` account.  Invalid, stale or ambiguous selections never
silently fall back to another account.
"""

from __future__ import annotations

from datetime import date, datetime
from math import isfinite
from typing import Any

from .account_risk_state import (
    LEDGER_POLICY_RISK_VERSION,
    RISK_SCOPE,
    assess_account_risk,
)
from .platform_config_store import PlatformConfigStore
from .shadow_account import ShadowAccountStore
from .simulation_order_state import OPEN_STATUSES
from .simulation_store import SimulationStore

ACTIVE_RECOMMENDATION_ACCOUNT_KEY_PREFIX = "active_recommendation_account:"
ACTIVE_RECOMMENDATION_ACCOUNT_VERSION = "active-recommendation-account-v1"
ACCOUNT_MAIN_PAPER = "main_paper"
ACCOUNT_MANUAL_SHADOW = "manual_shadow"
SELECTABLE_ACCOUNT_TYPES = (ACCOUNT_MAIN_PAPER, ACCOUNT_MANUAL_SHADOW)


def _selection_key(recommendation_portfolio_id: str) -> str:
    return f"{ACTIVE_RECOMMENDATION_ACCOUNT_KEY_PREFIX}{recommendation_portfolio_id}"


class RecommendationAccountStore:
    """Versioned active-account selection and production account-state adapter."""

    def __init__(
        self,
        database_url: str,
        *,
        configs: PlatformConfigStore | None = None,
        simulations: SimulationStore | None = None,
        shadows: ShadowAccountStore | None = None,
    ) -> None:
        self.configs = configs or PlatformConfigStore(database_url)
        self.simulations = simulations or SimulationStore(database_url)
        self.shadows = shadows or ShadowAccountStore(database_url)

    def select(
        self,
        *,
        recommendation_portfolio_id: str,
        account_type: str,
        account_id: str,
        actor: str,
        reason: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Explicitly select one account after validating its current identity."""

        portfolio_id = recommendation_portfolio_id.strip()
        selected_id = account_id.strip()
        if not portfolio_id or not selected_id:
            raise ValueError("recommendation portfolio and account id are required")
        if account_type not in SELECTABLE_ACCOUNT_TYPES:
            raise ValueError(
                "active recommendation account must be main_paper or manual_shadow"
            )
        if account_type == ACCOUNT_MAIN_PAPER:
            account = self.simulations.get(selected_id)
            if (
                str(account.get("source_type")) != "recommendation"
                or str(account.get("source_id")) != portfolio_id
                or str(account.get("status")) != "active"
            ):
                raise ValueError(
                    "main_paper must be an active simulation ledger linked to "
                    "the recommendation portfolio"
                )
        else:
            freshness = self.shadows.freshness(selected_id, now=now)
            if freshness["status"] != "fresh":
                raise ValueError("manual_shadow must have a fresh imported snapshot")
        self.configs.put(
            _selection_key(portfolio_id),
            {
                "contract_version": ACTIVE_RECOMMENDATION_ACCOUNT_VERSION,
                "recommendation_portfolio_id": portfolio_id,
                "account_type": account_type,
                "account_id": selected_id,
            },
            actor=actor,
            reason=reason,
        )
        return self.resolve(portfolio_id, now=now)

    def resolve(
        self,
        recommendation_portfolio_id: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Resolve the explicit selection, or the sole main-paper default."""

        portfolio_id = recommendation_portfolio_id.strip()
        record = self.configs.get(_selection_key(portfolio_id))
        value = dict((record or {}).get("value") or {})
        if (
            value.get("contract_version") == ACTIVE_RECOMMENDATION_ACCOUNT_VERSION
            and str(value.get("recommendation_portfolio_id")) == portfolio_id
        ):
            selection = {
                "status": "selected",
                "selected_via": "explicit",
                "config_revision": record["revision"],
                "recommendation_portfolio_id": portfolio_id,
                "account_type": str(value.get("account_type") or ""),
                "account_id": str(value.get("account_id") or ""),
            }
            return self._validate_resolved(selection, now=now)

        linked = [
            item
            for item in self.simulations.list(1000)
            if str(item.get("source_type")) == "recommendation"
            and str(item.get("source_id")) == portfolio_id
            and str(item.get("status")) == "active"
        ]
        if len(linked) == 1:
            return {
                "status": "selected",
                "selected_via": "default",
                "config_revision": None,
                "recommendation_portfolio_id": portfolio_id,
                "account_type": ACCOUNT_MAIN_PAPER,
                "account_id": str(linked[0]["id"]),
                "degraded": False,
                "reasons": [],
            }
        return {
            "status": "missing" if not linked else "ambiguous",
            "selected_via": "none",
            "config_revision": None,
            "recommendation_portfolio_id": portfolio_id,
            "account_type": None,
            "account_id": None,
            "degraded": True,
            "reasons": (
                ["no_main_paper_account"]
                if not linked
                else ["multiple_main_paper_accounts_require_explicit_selection"]
            ),
            "candidate_account_ids": [str(item["id"]) for item in linked],
        }

    def policy_risk_inputs(
        self,
        recommendation_portfolio_id: str,
        *,
        required_nav_date: date,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Return selected-account risk inputs for PortfolioPolicy."""

        selection = self.resolve(recommendation_portfolio_id, now=now)
        if (
            selection["status"] == "selected"
            and selection["account_type"] == ACCOUNT_MAIN_PAPER
            and not selection["degraded"]
        ):
            return self.simulations.policy_risk_inputs(
                str(selection["account_id"]),
                required_nav_date=required_nav_date,
            )
        if (
            selection["status"] == "selected"
            and selection["account_type"] == ACCOUNT_MANUAL_SHADOW
        ):
            return {
                "contract_version": LEDGER_POLICY_RISK_VERSION,
                "risk_scope": RISK_SCOPE,
                "portfolio_id": str(selection["account_id"]),
                "status": "blocked_manual_shadow_without_certified_nav",
                "portfolio_drawdown": 0.0,
                "daily_return": 0.0,
                "allow_new_risk": False,
                "reasons": ["manual_shadow_has_no_certified_nav_return_chain"],
                "required_nav_date": required_nav_date.isoformat(),
            }
        return {
            "contract_version": LEDGER_POLICY_RISK_VERSION,
            "risk_scope": RISK_SCOPE,
            "portfolio_id": None,
            "status": f"blocked_{selection['status']}_selected_account",
            "portfolio_drawdown": 0.0,
            "daily_return": 0.0,
            "allow_new_risk": False,
            "reasons": list(selection["reasons"]),
            "required_nav_date": required_nav_date.isoformat(),
        }

    def account_state_for_actions(
        self,
        recommendation_portfolio_id: str,
        *,
        reference_prices: dict[str, float],
        account_risk_state: dict[str, Any],
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Build the selected account state consumed by account-action planning."""

        prices = {
            str(instrument): float(value)
            for instrument, value in reference_prices.items()
            if isfinite(float(value)) and float(value) > 0
        }
        selection = self.resolve(recommendation_portfolio_id, now=now)
        if (
            selection["status"] != "selected"
            or selection["degraded"]
            or selection["account_type"] is None
        ):
            reason = "active_recommendation_account_unavailable:" + ",".join(
                selection["reasons"]
            )
            state = {
                instrument: {
                    "reference_price": price,
                    "not_executable_reason": reason,
                }
                for instrument, price in prices.items()
            }
            return {
                "account_state": state,
                "account_context": {
                    **selection,
                    "account_type": selection["account_type"] or ACCOUNT_MAIN_PAPER,
                    "degraded": True,
                },
                "account_value": None,
                "risk_assessment": assess_account_risk(
                    account_state_stale=True,
                    market_data_trusted=False,
                ),
            }
        if selection["account_type"] == ACCOUNT_MANUAL_SHADOW:
            built = self.shadows.account_state_for_actions(
                str(selection["account_id"]),
                now=now,
            )
            state = dict(built["account_state"])
            missing_prices: list[str] = []
            account_value = float(built["cash"])
            for instrument, item in state.items():
                price = prices.get(instrument)
                if price is not None:
                    item["reference_price"] = price
                    account_value += int(item.get("filled_position") or 0) * price
                elif int(item.get("filled_position") or 0) > 0:
                    missing_prices.append(instrument)
            for instrument, price in prices.items():
                state.setdefault(instrument, {})["reference_price"] = price
            context = {**selection, **built["account_context"]}
            if missing_prices:
                context["degraded"] = True
                context["missing_price_instruments"] = sorted(missing_prices)
                for item in state.values():
                    item.setdefault(
                        "not_executable_reason",
                        "manual_shadow_valuation_incomplete: simulation_only",
                    )
            degraded = bool(context.get("degraded"))
            return {
                "account_state": state,
                "account_context": context,
                "account_value": None if degraded else account_value,
                "risk_assessment": assess_account_risk(
                    account_state_stale=degraded,
                    market_data_trusted=not degraded,
                ),
            }

        portfolio = self.simulations.get(str(selection["account_id"]))
        positions = self.simulations.rows(str(selection["account_id"]), "positions")
        orders = [
            item
            for item in self.simulations.rows(str(selection["account_id"]), "orders")
            if str(item.get("status")) in OPEN_STATUSES
        ]
        state: dict[str, dict[str, Any]] = {}
        for position in positions:
            instrument = str(position["instrument"])
            quantity = int(position["quantity"])
            if quantity < 0:
                raise ValueError("main_paper account actions do not support short positions")
            state[instrument] = {
                "filled_position": quantity,
                "sellable_quantity": int(position["free_sellable_quantity"]),
                "open_orders": [],
            }
        for order in orders:
            instrument = str(order["instrument"])
            state.setdefault(instrument, {"open_orders": []}).setdefault(
                "open_orders", []
            ).append(
                {
                    "order_id": str(order["id"]),
                    "side": str(order["side"]),
                    "requested_quantity": int(order["requested_quantity"]),
                    "filled_quantity": int(order["filled_quantity"]),
                    "expires_at": order.get("expires_at"),
                    "created_at": order.get("created_at"),
                }
            )
        for instrument, price in prices.items():
            state.setdefault(instrument, {"open_orders": []})["reference_price"] = price
        trusted = str(account_risk_state.get("status")) in {
            "certified",
            "initial_empty_account",
        } and bool(account_risk_state.get("allow_new_risk"))
        nav = float(portfolio["nav"])
        max_weight = max(
            (
                max(0.0, float(item.get("market_value") or 0.0)) / nav
                for item in positions
                if nav > 0
            ),
            default=0.0,
        )
        drawdown = abs(float(account_risk_state.get("portfolio_drawdown") or 0.0))
        return {
            "account_state": state,
            "account_context": {
                **selection,
                "degraded": not trusted,
                "nav_trade_date": account_risk_state.get("nav_trade_date"),
            },
            "account_value": nav if trusted and nav > 0 else None,
            "risk_assessment": assess_account_risk(
                max_position_weight=min(1.0, max_weight),
                investment_wealth_drawdown=min(1.0, drawdown),
                unresolved_order_count=len(orders),
                data_stale=not trusted,
                account_state_stale=not trusted,
                market_data_trusted=trusted,
            ),
        }

    def _validate_resolved(
        self,
        selection: dict[str, Any],
        *,
        now: datetime | None,
    ) -> dict[str, Any]:
        account_type = selection["account_type"]
        account_id = selection["account_id"]
        if account_type == ACCOUNT_MAIN_PAPER:
            try:
                account = self.simulations.get(account_id)
            except KeyError:
                account = {}
            reasons = []
            if (
                str(account.get("source_type")) != "recommendation"
                or str(account.get("source_id"))
                != selection["recommendation_portfolio_id"]
            ):
                reasons.append("selected_main_paper_identity_mismatch")
            if str(account.get("status")) != "active":
                reasons.append("selected_main_paper_not_active")
            return {
                **selection,
                "degraded": bool(reasons),
                "reasons": reasons,
            }
        if account_type == ACCOUNT_MANUAL_SHADOW:
            freshness = self.shadows.freshness(account_id, now=now)
            return {
                **selection,
                "degraded": freshness["status"] != "fresh",
                "reasons": (
                    []
                    if freshness["status"] == "fresh"
                    else [f"selected_manual_shadow_{freshness['status']}"]
                ),
                "freshness": freshness,
            }
        return {
            **selection,
            "degraded": True,
            "reasons": ["selected_account_type_invalid"],
        }
