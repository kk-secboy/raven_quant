"""Versioned personal market permissions (design draft 8.7).

Every account binds a versioned personal investment policy: per exchange,
board, risk-warning category and ETF subtype one of ``buy_sell``,
``sell_only``, ``disabled`` or ``unknown``, with confirmation source,
``as_of`` and an optional validity end.

Semantics:

- The *effective* permission for a scope on a date is the latest version with
  ``as_of <= date``; no version, or a version whose ``valid_until`` has
  passed, is ``unknown`` — which downgrades advice to ``simulation_only``
  (research/general simulation/target weights only, no pseudo-executable
  buy quantities).
- Write discipline: a new version may only *tighten*
  (``buy_sell → sell_only → disabled → unknown``). A relaxation requires an
  explicit ``relaxation_confirmed=True`` plus a confirmation source, so a
  policy can never be silently loosened.
- Action layer (see :meth:`MarketPermissionStore.gate_for_instrument`):
  ``disabled``/``sell_only`` block new-risk (buy) actions with a recorded
  reason; ``sell_only`` still allows SELL/EXIT; ``unknown``/expired marks the
  action not executable (``simulation_only``) without erasing the advice.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import insert, select

from quant_data.database import (
    market_permission_versions,
    open_database,
    row_dict,
)

from .market_rules import _instrument_digits

PERMISSION_BUY_SELL = "buy_sell"
PERMISSION_SELL_ONLY = "sell_only"
PERMISSION_DISABLED = "disabled"
PERMISSION_UNKNOWN = "unknown"
PERMISSIONS = (
    PERMISSION_BUY_SELL,
    PERMISSION_SELL_ONLY,
    PERMISSION_DISABLED,
    PERMISSION_UNKNOWN,
)

# Higher is more restrictive; a new version must not move to a lower rank
# unless the relaxation is explicitly confirmed.
_TIGHTEN_RANK = {
    PERMISSION_BUY_SELL: 0,
    PERMISSION_SELL_ONLY: 1,
    PERMISSION_DISABLED: 2,
    PERMISSION_UNKNOWN: 3,
}

SCOPE_EXCHANGE = "exchange"
SCOPE_BOARD = "board"
SCOPE_RISK_WARNING = "risk_warning"
SCOPE_ETF_SUBTYPE = "etf_subtype"
SCOPE_TYPES = (SCOPE_EXCHANGE, SCOPE_BOARD, SCOPE_RISK_WARNING, SCOPE_ETF_SUBTYPE)

_ETF_PREFIX_EXCHANGE = (
    ("51", "SSE"),
    ("56", "SSE"),
    ("58", "SSE"),
    ("15", "SZSE"),
    ("16", "SZSE"),
    ("18", "SZSE"),
)


def _now() -> datetime:
    return datetime.now(UTC)


def instrument_scopes(instrument: str) -> list[tuple[str, str]]:
    """Classify an instrument into permission scopes (fail-closed on bad codes).

    Risk-warning (ST/*ST) categories cannot be derived from the code and are
    not guessed here; bind them explicitly via scope_type ``risk_warning``.
    """

    digits = _instrument_digits(instrument)
    for prefix, exchange in _ETF_PREFIX_EXCHANGE:
        if digits.startswith(prefix):
            return [
                (SCOPE_ETF_SUBTYPE, f"etf_{prefix}"),
                (SCOPE_EXCHANGE, exchange),
            ]
    if digits.startswith(("68", "69")):
        return [(SCOPE_BOARD, "star"), (SCOPE_EXCHANGE, "SSE")]
    if digits.startswith(("30",)):
        return [(SCOPE_BOARD, "chinext"), (SCOPE_EXCHANGE, "SZSE")]
    if digits.startswith(("43", "83", "87", "88", "92")):
        return [(SCOPE_BOARD, "bse"), (SCOPE_EXCHANGE, "BSE")]
    if digits.startswith("6"):
        return [(SCOPE_BOARD, "main"), (SCOPE_EXCHANGE, "SSE")]
    if digits.startswith("0"):
        return [(SCOPE_BOARD, "main"), (SCOPE_EXCHANGE, "SZSE")]
    raise ValueError(f"cannot classify instrument into permission scopes: {instrument}")


class MarketPermissionStore:
    """Durable versioned personal market permission policy."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def record_version(
        self,
        *,
        scope_type: str,
        scope_key: str,
        permission: str,
        confirmation_source: str,
        as_of: date,
        valid_until: date | None = None,
        actor: str,
        relaxation_confirmed: bool = False,
    ) -> dict[str, Any]:
        """Append one immutable policy version; tightening-only discipline."""

        if scope_type not in SCOPE_TYPES:
            raise ValueError(f"unknown market permission scope type: {scope_type!r}")
        scope_key = str(scope_key).strip()
        if not scope_key:
            raise ValueError("market permission scope key is required")
        if permission not in PERMISSIONS:
            raise ValueError(f"unknown market permission: {permission!r}")
        if not confirmation_source.strip():
            raise ValueError("market permission versions require a confirmation source")
        if not actor.strip():
            raise ValueError("market permission versions require an actor")
        if not isinstance(as_of, date):
            raise ValueError("as_of must be a date")
        if valid_until is not None and valid_until < as_of:
            raise ValueError("valid_until must not precede as_of")
        now = _now()
        with self.engine.begin() as connection:
            predecessor = connection.execute(
                select(market_permission_versions)
                .where(
                    market_permission_versions.c.scope_type == scope_type,
                    market_permission_versions.c.scope_key == scope_key,
                )
                .order_by(
                    market_permission_versions.c.as_of.desc(),
                    market_permission_versions.c.created_at.desc(),
                )
                .limit(1)
                .with_for_update()
            ).first()
            if (
                predecessor is not None
                and _TIGHTEN_RANK[permission] < _TIGHTEN_RANK[str(predecessor.permission)]
                and not relaxation_confirmed
            ):
                raise ValueError(
                    "personal market permissions can only be tightened "
                    f"({predecessor.permission} -> {permission} is a relaxation); "
                    "a relaxation requires explicit relaxation_confirmed plus a "
                    "confirmation source"
                )
            version_id = uuid.uuid4().hex
            connection.execute(
                insert(market_permission_versions).values(
                    id=version_id,
                    scope_type=scope_type,
                    scope_key=scope_key,
                    permission=permission,
                    confirmation_source=confirmation_source.strip(),
                    as_of=as_of,
                    valid_until=valid_until,
                    supersedes_id=str(predecessor.id) if predecessor is not None else None,
                    relaxation_confirmed=bool(relaxation_confirmed),
                    created_by=actor.strip(),
                    created_at=now,
                )
            )
        return self.get_version(version_id)

    def get_version(self, version_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(market_permission_versions).where(
                    market_permission_versions.c.id == version_id
                )
            ).first()
        if row is None:
            raise KeyError(version_id)
        return row_dict(row)

    def list_versions(
        self, *, scope_type: str | None = None, scope_key: str | None = None
    ) -> list[dict[str, Any]]:
        statement = select(market_permission_versions)
        if scope_type is not None:
            statement = statement.where(market_permission_versions.c.scope_type == scope_type)
        if scope_key is not None:
            statement = statement.where(market_permission_versions.c.scope_key == scope_key)
        statement = statement.order_by(
            market_permission_versions.c.scope_type,
            market_permission_versions.c.scope_key,
            market_permission_versions.c.as_of.desc(),
        )
        with self.engine.connect() as connection:
            return [row_dict(row) for row in connection.execute(statement).all()]

    def effective_permission(
        self, scope_type: str, scope_key: str, *, on_date: date
    ) -> dict[str, Any]:
        """Latest confirmed version for a scope; missing/expired is ``unknown``."""

        with self.engine.connect() as connection:
            row = connection.execute(
                select(market_permission_versions)
                .where(
                    market_permission_versions.c.scope_type == scope_type,
                    market_permission_versions.c.scope_key == scope_key,
                    market_permission_versions.c.as_of <= on_date,
                )
                .order_by(
                    market_permission_versions.c.as_of.desc(),
                    market_permission_versions.c.created_at.desc(),
                )
                .limit(1)
            ).first()
        if row is None:
            return {
                "scope_type": scope_type,
                "scope_key": scope_key,
                "permission": PERMISSION_UNKNOWN,
                "reason": "no_confirmed_permission",
                "version_id": None,
            }
        version = row_dict(row)
        valid_until = version.get("valid_until")
        if valid_until is not None and valid_until < on_date:
            return {
                "scope_type": scope_type,
                "scope_key": scope_key,
                "permission": PERMISSION_UNKNOWN,
                "reason": f"permission_expired:{valid_until.isoformat()}",
                "version_id": str(version["id"]),
            }
        return {
            "scope_type": scope_type,
            "scope_key": scope_key,
            "permission": str(version["permission"]),
            "reason": "confirmed",
            "version_id": str(version["id"]),
        }

    def permission_for_instrument(self, instrument: str, *, on_date: date) -> dict[str, Any]:
        """Most restrictive effective permission across the instrument's scopes.

        A scope with no recorded version at all is *not applicable* and does
        not drag the instrument down (users confirm the scopes they use);
        ``unknown`` results from no confirmed version anywhere, or from an
        expired recorded version — both downgrade to ``simulation_only``.
        """

        scopes = [
            self.effective_permission(scope_type, scope_key, on_date=on_date)
            for scope_type, scope_key in instrument_scopes(instrument)
        ]
        recorded = [item for item in scopes if item["version_id"] is not None]
        if not recorded:
            return {
                "instrument": str(instrument).strip().upper(),
                "permission": PERMISSION_UNKNOWN,
                "reason": "no_confirmed_permission",
                "scopes": scopes,
            }
        worst = max(recorded, key=lambda item: _TIGHTEN_RANK[item["permission"]])
        return {
            "instrument": str(instrument).strip().upper(),
            "permission": worst["permission"],
            "reason": worst["reason"],
            "scopes": scopes,
        }

    def gate_for_instrument(
        self, instrument: str, *, on_date: date, is_buy_action: bool
    ) -> dict[str, str] | None:
        """Translate the instrument permission into an action-layer gate.

        Returns ``None`` when the action is unrestricted, otherwise a gate
        with ``kind`` ``"hard"`` (action blocked with reason) or ``"soft"``
        (advice kept but marked not executable / ``simulation_only``).
        """

        effective = self.permission_for_instrument(instrument, on_date=on_date)
        permission = effective["permission"]
        if permission == PERMISSION_BUY_SELL:
            return None
        if permission in {PERMISSION_SELL_ONLY, PERMISSION_DISABLED}:
            if not is_buy_action:
                # SELL/EXIT stays allowed: sell_only only reduces risk and a
                # disabled scope still permits the pre-registered exit (8.7).
                return None
            return {
                "kind": "hard",
                "reason": f"market_permission_{permission}:{effective['reason']}",
                "permission": permission,
            }
        return {
            "kind": "soft",
            "reason": f"market_permission_simulation_only:{effective['reason']}",
            "permission": permission,
        }
