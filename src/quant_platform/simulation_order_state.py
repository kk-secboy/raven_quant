"""Persistent simulation order state machine and order-plan consumption.

Design draft 8.1 steps 7-8 / 9.2 / 12.4: simulation orders persist across
batches instead of being created and settled inside one day. The lifecycle is

    planned -> open -> filled | partial_filled_expired | rejected | expired
       |         |
       +---------+--> cancelled

- ``planned``: committed by an order plan (same transaction as its batch) but
  not yet executable — its execution batch has not started.
- ``open``: executable; a working order whose window (``not_after``) spans
  multiple days stays ``open`` across batches and accumulates fills
  (``open -> open`` is the carry-over transition).
- Terminal: ``filled``, ``partial_filled_expired`` (some fills, window or day
  closed), ``rejected`` (never executable), ``expired`` (window lapsed with no
  fill), ``cancelled``.

Cancellation discipline: cash moves at fill time in this ledger, so a cancel
never refunds cash — it negates the still-unfilled remainder, exactly once.
Cancelling an already terminal order is an idempotent no-op (skipped with a
note), never a second release.

This module is a pure layer — no database, no clock, no side effects. The
store (:mod:`quant_platform.simulation_store`) applies the returned mutations
inside its plan-commit transaction.
"""

from __future__ import annotations

from typing import Any

ORDER_PLAN_MODEL_VERSION = "simulation-order-plan-v1"

STATUS_PLANNED = "planned"
STATUS_OPEN = "open"
STATUS_FILLED = "filled"
STATUS_PARTIAL_EXPIRED = "partial_filled_expired"
STATUS_REJECTED = "rejected"
STATUS_EXPIRED = "expired"
STATUS_CANCELLED = "cancelled"

OPEN_STATUSES = (STATUS_PLANNED, STATUS_OPEN)
TERMINAL_STATUSES = (
    STATUS_FILLED,
    STATUS_PARTIAL_EXPIRED,
    STATUS_REJECTED,
    STATUS_EXPIRED,
    STATUS_CANCELLED,
)

LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    STATUS_PLANNED: frozenset({STATUS_OPEN, STATUS_CANCELLED, STATUS_EXPIRED}),
    STATUS_OPEN: frozenset(
        {
            STATUS_OPEN,  # working order carried into the next execution day
            STATUS_FILLED,
            STATUS_PARTIAL_EXPIRED,
            STATUS_REJECTED,
            STATUS_EXPIRED,
            STATUS_CANCELLED,
        }
    ),
    STATUS_FILLED: frozenset(),
    STATUS_PARTIAL_EXPIRED: frozenset(),
    STATUS_REJECTED: frozenset(),
    STATUS_EXPIRED: frozenset(),
    STATUS_CANCELLED: frozenset(),
}

OP_KEEP = "keep"
OP_CANCEL = "cancel"
OP_REPLACE = "replace"
OP_NEW = "new"


def is_legal_transition(from_status: str, to_status: str) -> bool:
    return to_status in LEGAL_TRANSITIONS.get(from_status, frozenset())


def assert_transition(from_status: str, to_status: str) -> None:
    if not is_legal_transition(from_status, to_status):
        raise ValueError(
            f"illegal simulation order transition: {from_status!r} -> {to_status!r}"
        )


def normalize_persistent_order(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one persistent order row for plan consumption.

    Accepts the simulation_orders shape (``id``, ``instrument``, ``side``,
    ``requested_quantity``, ``filled_quantity``, ``status``,
    ``limit_price``/``not_before``/``not_after`` optional).
    """

    status = str(row.get("status") or "")
    if status not in LEGAL_TRANSITIONS:
        raise ValueError(f"unknown simulation order status: {row!r}")
    side = str(row.get("side") or "").lower()
    if side not in {"buy", "sell"}:
        raise ValueError(f"persistent order side must be buy or sell: {row!r}")
    requested = int(row.get("requested_quantity", 0))
    filled = int(row.get("filled_quantity", 0))
    if requested <= 0 or filled < 0 or filled > requested:
        raise ValueError(f"persistent order quantities are invalid: {row!r}")
    if status in TERMINAL_STATUSES and filled > requested:
        raise ValueError(f"terminal order over-filled: {row!r}")
    return {
        "order_id": str(row.get("order_id") or row.get("id") or ""),
        "instrument": str(row.get("instrument") or "").upper(),
        "side": side,
        "requested_quantity": requested,
        "filled_quantity": filled,
        "remaining_quantity": requested - filled,
        "status": status,
        "limit_price": row.get("limit_price"),
    }


def apply_order_plan(
    *,
    open_orders: list[dict[str, Any]],
    plan_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Consume keep/cancel/replace/new entries onto persistent open orders.

    ``open_orders`` are the account's currently live orders (status in
    OPEN_STATUSES), ``plan_entries`` the keep/cancel/replace/new output of
    :mod:`quant_platform.recommendation_actions`. Returns ``cancels``,
    ``replaces``, ``news``, ``keeps`` and ``skipped`` mutation lists.

    Fail-closed: a cancel/replace referencing an unknown order raises; a
    replace that would grow the order raises (the planning layer only trims).
    Idempotent: a cancel/replace against an already terminal order is skipped
    (a retried plan application must not double-release).
    """

    orders = {
        order["order_id"]: order
        for order in (normalize_persistent_order(row) for row in open_orders)
    }
    if len(orders) != len(open_orders):
        raise ValueError("duplicate open order ids in the persistent order book")
    cancels: list[dict[str, Any]] = []
    replaces: list[dict[str, Any]] = []
    news: list[dict[str, Any]] = []
    keeps: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    mutated: set[str] = set()
    for entry in plan_entries:
        op = str(entry.get("op") or "")
        order_id = str(entry.get("order_id") or "")
        if op == OP_NEW:
            side = str(entry.get("side") or "").lower()
            quantity = int(entry.get("quantity", 0))
            if side not in {"buy", "sell"} or quantity <= 0:
                raise ValueError(f"invalid new-order plan entry: {entry!r}")
            news.append(
                {
                    "op": OP_NEW,
                    "instrument": str(entry.get("instrument") or "").upper(),
                    "side": side,
                    "quantity": quantity,
                    "limit_price": entry.get("limit_price"),
                    "reason": str(entry.get("reason") or ""),
                }
            )
            continue
        if op not in {OP_KEEP, OP_CANCEL, OP_REPLACE}:
            raise ValueError(f"unsupported order plan op: {entry!r}")
        target = orders.get(order_id)
        if target is None:
            raise ValueError(f"order plan references an unknown order: {order_id!r}")
        if order_id in mutated:
            raise ValueError(f"order plan mutates one order twice: {order_id!r}")
        mutated.add(order_id)
        if target["status"] in TERMINAL_STATUSES:
            # Idempotent replay: the order is already done; skipping avoids a
            # double release of the unfilled remainder.
            skipped.append({"op": op, "order_id": order_id, "status": target["status"]})
            continue
        if op == OP_KEEP:
            keeps.append({"op": OP_KEEP, "order_id": order_id})
            continue
        if op == OP_CANCEL:
            quantity = int(entry.get("quantity", 0))
            if quantity <= 0 or quantity > target["remaining_quantity"]:
                raise ValueError(f"cancel quantity exceeds the open remainder: {entry!r}")
            cancels.append(
                {
                    "op": OP_CANCEL,
                    "order_id": order_id,
                    "released_quantity": target["remaining_quantity"],
                    "reason": str(entry.get("reason") or ""),
                }
            )
            continue
        # replace: the plan quantity is the adjusted *remaining*; the
        # persistent requested quantity becomes filled + adjusted.
        adjusted = int(entry.get("quantity", 0))
        if adjusted <= 0 or adjusted > target["remaining_quantity"]:
            raise ValueError(f"replace quantity is outside the open remainder: {entry!r}")
        new_requested = target["filled_quantity"] + adjusted
        if new_requested >= target["requested_quantity"]:
            raise ValueError(
                f"replace may only trim a persistent order: {entry!r} "
                f"(requested {target['requested_quantity']})"
            )
        replaces.append(
            {
                "op": OP_REPLACE,
                "order_id": order_id,
                "new_requested_quantity": new_requested,
                "released_quantity": target["requested_quantity"] - new_requested,
                "reason": str(entry.get("reason") or ""),
            }
        )
    return {
        "cancels": cancels,
        "replaces": replaces,
        "news": news,
        "keeps": keeps,
        "skipped": skipped,
        "model_version": ORDER_PLAN_MODEL_VERSION,
    }
