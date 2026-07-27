"""Manual/CSV personal shadow accounts (design draft 8.6).

Users may import their real holdings, cash, sellable quantities and open
orders as a ``manual_shadow`` account; the system never assumes the user
followed the advice. Shadow, model and simulation accounts stay visibly
separate (``account_type`` on every produced artifact), and when the shadow
state is not fresh the recommendation chain degrades to simulation-only
advice — it never silently falls back to the simulation ledger and passes it
off as the real account.

Freshness rule: a snapshot is ``fresh`` for ``stale_after_days`` natural days
from ``imported_at`` (default 2); older or missing state is degraded and
stops precise-quantity advice.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import insert, select

from quant_data.database import (
    open_database,
    row_dict,
    shadow_account_snapshots,
)

SHADOW_ACCOUNT_TYPE = "manual_shadow"
IMPORT_SOURCES = ("manual", "csv")
DEFAULT_STALE_AFTER_DAYS = 2

CSV_HOLDING_COLUMNS = ("instrument", "quantity", "sellable_quantity")


def _now() -> datetime:
    return datetime.now(UTC)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer: {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer: {value!r}") from exc
    if not number.is_integer():
        raise ValueError(f"{label} must be an integer: {value!r}")
    return int(number)


def _validate_holding(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"shadow holding must be an object: {item!r}")
    unknown = set(item) - {"instrument", "quantity", "sellable_quantity"}
    if unknown:
        raise ValueError(f"shadow holding has unsupported fields {sorted(unknown)}: {item!r}")
    instrument = str(item.get("instrument") or "").strip().upper()
    if not instrument:
        raise ValueError("shadow holding requires an instrument")
    quantity = _as_int(item.get("quantity"), "shadow holding quantity")
    sellable = _as_int(
        item.get("sellable_quantity", quantity), "shadow holding sellable quantity"
    )
    if quantity < 0:
        raise ValueError(f"shadow holding quantity must not be negative: {item!r}")
    if not 0 <= sellable <= quantity:
        raise ValueError(
            f"shadow holding sellable quantity must stay within the position: {item!r}"
        )
    return {
        "instrument": instrument,
        "quantity": quantity,
        "sellable_quantity": sellable,
    }


def _validate_open_order(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"shadow open order must be an object: {item!r}")
    unknown = set(item) - {"instrument", "side", "quantity", "filled_quantity"}
    if unknown:
        raise ValueError(f"shadow open order has unsupported fields {sorted(unknown)}: {item!r}")
    instrument = str(item.get("instrument") or "").strip().upper()
    side = str(item.get("side") or "").strip().lower()
    if not instrument or side not in {"buy", "sell"}:
        raise ValueError(f"shadow open order requires instrument and side buy/sell: {item!r}")
    quantity = _as_int(item.get("quantity"), "shadow open order quantity")
    filled = _as_int(item.get("filled_quantity", 0), "shadow open order filled quantity")
    if quantity <= 0 or not 0 <= filled < quantity:
        raise ValueError(f"shadow open order quantities are invalid: {item!r}")
    return {
        "instrument": instrument,
        "side": side,
        "quantity": quantity,
        "filled_quantity": filled,
    }


class ShadowAccountStore:
    """Durable manual/CSV shadow account snapshots with freshness judgement."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def import_snapshot(
        self,
        *,
        account_id: str,
        cash: float | int | Decimal,
        holdings: list[dict[str, Any]],
        open_orders: list[dict[str, Any]] | None = None,
        import_source: str = "manual",
        imported_by: str,
        notes: str | None = None,
        imported_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Import one immutable full-state snapshot (schema fail-closed)."""

        if not account_id.strip():
            raise ValueError("shadow account id is required")
        if import_source not in IMPORT_SOURCES:
            raise ValueError(f"shadow import source must be one of {IMPORT_SOURCES}")
        if not imported_by.strip():
            raise ValueError("shadow imports require an importer actor")
        cash_value = Decimal(str(cash))
        if not cash_value.is_finite() or cash_value < 0:
            raise ValueError("shadow account cash must be a finite non-negative number")
        validated_holdings = [_validate_holding(item) for item in holdings]
        validated_orders = [_validate_open_order(item) for item in (open_orders or [])]
        content = {
            "cash": str(cash_value),
            "holdings": validated_holdings,
            "open_orders": validated_orders,
        }
        snapshot_id = uuid.uuid4().hex
        with self.engine.begin() as connection:
            connection.execute(
                insert(shadow_account_snapshots).values(
                    id=snapshot_id,
                    account_id=account_id.strip(),
                    import_source=import_source,
                    cash=cash_value,
                    holdings_json=validated_holdings,
                    open_orders_json=validated_orders,
                    content_sha256=_canonical_sha256(content),
                    notes=notes,
                    imported_by=imported_by.strip(),
                    imported_at=imported_at or _now(),
                )
            )
        return self.get_snapshot(snapshot_id)

    def import_csv(
        self,
        *,
        account_id: str,
        content: str,
        imported_by: str,
        cash: float | int | Decimal = 0,
        open_orders: list[dict[str, Any]] | None = None,
        notes: str | None = None,
        imported_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Import holdings from CSV (header ``instrument,quantity,sellable_quantity``).

        Fail-closed: unexpected/missing columns, blank instruments, non-integer
        or inconsistent quantities all reject the whole import.
        """

        reader = csv.reader(io.StringIO(content))
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        if not rows:
            raise ValueError("shadow CSV import is empty")
        header = [cell.strip().lower() for cell in rows[0]]
        if header != list(CSV_HOLDING_COLUMNS):
            raise ValueError(
                "shadow CSV header must be exactly "
                f"{','.join(CSV_HOLDING_COLUMNS)} (got {','.join(header) or 'nothing'})"
            )
        if len(rows) < 2:
            raise ValueError("shadow CSV import contains no holdings rows")
        holdings = []
        for line_number, row in enumerate(rows[1:], start=2):
            if len(row) != len(CSV_HOLDING_COLUMNS):
                raise ValueError(f"shadow CSV row {line_number} has the wrong column count")
            instrument, quantity, sellable = (cell.strip() for cell in row)
            holdings.append(
                _validate_holding(
                    {
                        "instrument": instrument,
                        "quantity": quantity,
                        "sellable_quantity": sellable,
                    }
                )
            )
        return self.import_snapshot(
            account_id=account_id,
            cash=cash,
            holdings=holdings,
            open_orders=open_orders,
            import_source="csv",
            imported_by=imported_by,
            notes=notes,
            imported_at=imported_at,
        )

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(shadow_account_snapshots).where(
                    shadow_account_snapshots.c.id == snapshot_id
                )
            ).first()
        if row is None:
            raise KeyError(snapshot_id)
        return self._decode(row_dict(row))

    def latest_snapshot(self, account_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(shadow_account_snapshots)
                .where(shadow_account_snapshots.c.account_id == account_id)
                .order_by(
                    shadow_account_snapshots.c.imported_at.desc(),
                    shadow_account_snapshots.c.id.desc(),
                )
                .limit(1)
            ).first()
        return self._decode(row_dict(row)) if row is not None else None

    def list_accounts(self) -> list[dict[str, Any]]:
        """Shadow accounts with their latest snapshot, marked manual_shadow."""

        statement = select(shadow_account_snapshots).order_by(
            shadow_account_snapshots.c.account_id,
            shadow_account_snapshots.c.imported_at.desc(),
            shadow_account_snapshots.c.id.desc(),
        )
        with self.engine.connect() as connection:
            rows = [self._decode(row_dict(row)) for row in connection.execute(statement).all()]
        latest: dict[str, dict[str, Any]] = {}
        for snapshot in rows:
            latest.setdefault(str(snapshot["account_id"]), snapshot)
        accounts = []
        for snapshot in latest.values():
            accounts.append(
                {
                    "account_id": snapshot["account_id"],
                    "account_type": SHADOW_ACCOUNT_TYPE,
                    "latest_snapshot_id": snapshot["id"],
                    "import_source": snapshot["import_source"],
                    "imported_by": snapshot["imported_by"],
                    "imported_at": snapshot["imported_at"],
                    "cash": snapshot["cash"],
                    "holding_count": len(snapshot["holdings"]),
                    "open_order_count": len(snapshot["open_orders"]),
                }
            )
        return accounts

    def freshness(
        self,
        account_id: str,
        *,
        now: datetime | None = None,
        stale_after_days: int = DEFAULT_STALE_AFTER_DAYS,
    ) -> dict[str, Any]:
        """Natural-day freshness of the latest snapshot for one account."""

        if stale_after_days < 0:
            raise ValueError("stale_after_days must not be negative")
        current = now or _now()
        snapshot = self.latest_snapshot(account_id)
        if snapshot is None:
            return {
                "account_id": account_id,
                "status": "missing",
                "imported_at": None,
                "age_days": None,
                "stale_after_days": stale_after_days,
            }
        imported_at = snapshot["imported_at"]
        if isinstance(imported_at, str):
            imported_at = datetime.fromisoformat(imported_at)
        age_days = (current.date() - imported_at.date()).days
        return {
            "account_id": account_id,
            "status": "fresh" if age_days <= stale_after_days else "stale",
            "imported_at": imported_at.isoformat(),
            "age_days": age_days,
            "stale_after_days": stale_after_days,
        }

    def account_state_for_actions(
        self,
        account_id: str,
        *,
        now: datetime | None = None,
        stale_after_days: int = DEFAULT_STALE_AFTER_DAYS,
    ) -> dict[str, Any]:
        """Build the ``account_state``/context for account action planning.

        Fresh snapshots contribute filled/sellable quantities and open orders.
        Stale or missing state is *degraded*: quantities are still the last
        imported truth (labelled with ``imported_at``), open orders are dropped
        as untrustworthy, and every instrument carries a ``simulation_only``
        not-executable reason — there is no silent fallback to the simulation
        ledger (design 8.6/8.1: 状态不新鲜时只显示目标权重，不给伪精确数量).
        """

        current = now or _now()
        fresh = self.freshness(account_id, now=current, stale_after_days=stale_after_days)
        degraded = fresh["status"] != "fresh"
        if fresh["status"] == "missing":
            degraded_reason = (
                "manual_shadow_missing: no imported snapshot; advice is simulation_only"
            )
        elif degraded:
            degraded_reason = (
                f"manual_shadow_stale: latest import {fresh['imported_at']} is "
                f"{fresh['age_days']} natural days old (limit {stale_after_days}); "
                "advice is simulation_only"
            )
        else:
            degraded_reason = None
        context = {
            "account_type": SHADOW_ACCOUNT_TYPE,
            "account_id": account_id,
            "degraded": degraded,
            "freshness": fresh,
        }
        snapshot = self.latest_snapshot(account_id)
        if snapshot is None:
            return {"account_state": {}, "account_context": context, "cash": 0.0}
        imported_at = snapshot["imported_at"]
        if isinstance(imported_at, str):
            imported_at = datetime.fromisoformat(imported_at)
        orders_by_instrument: dict[str, list[dict[str, Any]]] = {}
        if not degraded:
            for index, order in enumerate(snapshot["open_orders"]):
                orders_by_instrument.setdefault(order["instrument"], []).append(
                    {
                        "order_id": f"shadow-{snapshot['id']}-{index}",
                        "side": order["side"],
                        "requested_quantity": order["quantity"],
                        "filled_quantity": order["filled_quantity"],
                        "expires_at": None,
                        "created_at": imported_at,
                    }
                )
        account_state: dict[str, dict[str, Any]] = {}
        for holding in snapshot["holdings"]:
            instrument = holding["instrument"]
            entry: dict[str, Any] = {
                "filled_position": holding["quantity"],
                "sellable_quantity": holding["sellable_quantity"],
                "open_orders": orders_by_instrument.get(instrument, []),
            }
            if degraded_reason:
                entry["not_executable_reason"] = degraded_reason
            account_state[instrument] = entry
        for instrument, orders in orders_by_instrument.items():
            account_state.setdefault(instrument, {"open_orders": orders})
        return {
            "account_state": account_state,
            "account_context": context,
            "cash": snapshot["cash"],
        }

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        row["holdings"] = row.pop("holdings_json")
        row["open_orders"] = row.pop("open_orders_json")
        return row
