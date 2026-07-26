"""Two-dimension recommendation action model (design draft 8.4/8.1 steps 7+9/9.2).

Account *action* and *execution state* are separate dimensions: a ``SELL``
recommendation may sit in ``WAIT`` without losing its sell target.  This
module is a pure planning layer — no database, no system clock, no side
effects.  Per instrument it takes the final target, the confirmed
``filled_position``, the still-valid open orders and the tradability flags,
and returns ``{action, execution_state, projected_position, order_plan,
blocked_reason?}``.

Semantics (design 8.1 steps 7/9 and 8.4):

- ``projected_position = filled_position + valid open buys − valid open sells``
  (only unexpired remaining quantities count; a partially filled order
  contributes its remaining and marks the line ``PARTIAL``).
- The *action* compares the final target with ``filled_position``; the *order
  quantity* compares it with ``projected_position``: new orders only cover
  ``final_target_quantity − projected_position``.
- The order plan is built per order as ``keep/cancel/replace/new`` after the
  final target is fixed: opposite-side valid orders are cancelled, same-side
  orders are kept in creation order until the gross need is covered, a
  boundary order that overshoots is replaced with its adjusted remainder,
  and any leftover difference becomes one ``new`` order (lot-rounded for
  buys, clamped by sellable quantity for sells).
- Hard gates (data/account/market) never erase the action: ``BUY/SELL/EXIT``
  is preserved and the line is marked ``BLOCKED`` with the reason; soft
  unexecutability (price/liquidity/trading state, T+1 sellable shortfall) is
  ``WAIT``.  ``NO_ACTION`` never deletes a previous valid target and is only
  emitted when there is no target and no position.

Sim-ledger alignment note: simulation orders expire at 15:00 on their own
trade date (``simulation_engine`` ``expires_at``), so an unfilled remainder
never survives into the next decision.  When the plan is computed after
expiry with the target still unmet, the stale coverage is reported as
``EXPIRED`` and a fresh ``new`` order re-covers the remainder on the next
decision instead of silently assuming the old order still works.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

ACTION_BUY = "BUY"
ACTION_SELL = "SELL"
ACTION_EXIT = "EXIT"
ACTION_HOLD = "HOLD"
ACTION_NO_ACTION = "NO_ACTION"
ACCOUNT_ACTIONS = (ACTION_BUY, ACTION_SELL, ACTION_EXIT, ACTION_HOLD, ACTION_NO_ACTION)

STATE_READY = "READY"
STATE_WAIT = "WAIT"
STATE_PARTIAL = "PARTIAL"
STATE_CANCELLED = "CANCELLED"
STATE_EXPIRED = "EXPIRED"
STATE_BLOCKED = "BLOCKED"
EXECUTION_STATES = (
    STATE_READY,
    STATE_WAIT,
    STATE_PARTIAL,
    STATE_CANCELLED,
    STATE_EXPIRED,
    STATE_BLOCKED,
)

OP_KEEP = "keep"
OP_CANCEL = "cancel"
OP_REPLACE = "replace"
OP_NEW = "new"

RECOMMENDATION_ACTION_MODEL_VERSION = "recommendation-actions-v1"

_BUY = "buy"
_SELL = "sell"


def normalize_open_order(order: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    """Normalize one open-order row to remaining/expiry semantics.

    Accepts the simulation-ledger shape (``side``, ``requested_quantity``,
    ``filled_quantity``, ``expires_at``, ``created_at``).  An order is *valid*
    while it has remaining quantity and has not expired; an expired remainder
    is reported separately so stale coverage cannot masquerade as a live order.
    """

    side = str(order.get("side") or "").lower()
    if side not in {_BUY, _SELL}:
        raise ValueError(f"open order side must be buy or sell: {order!r}")
    requested = int(order.get("requested_quantity", 0))
    filled = int(order.get("filled_quantity", 0))
    if requested < 0 or filled < 0 or filled > requested:
        raise ValueError(f"open order quantities are invalid: {order!r}")
    remaining = requested - filled
    expires_at = order.get("expires_at")
    if expires_at is not None and now.tzinfo is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=now.tzinfo)
    expired = bool(remaining > 0 and expires_at is not None and now >= expires_at)
    return {
        "order_id": str(order.get("order_id") or order.get("id") or ""),
        "side": side,
        "requested_quantity": requested,
        "filled_quantity": filled,
        "remaining_quantity": remaining,
        "expired": expired,
        "valid": remaining > 0 and not expired,
        "created_at": order.get("created_at"),
    }


def projected_position(
    filled_position: int, open_orders: list[dict[str, Any]]
) -> int:
    """``filled_position + 有效买单 − 有效卖单`` (design 8.1 step 7)."""

    projected = int(filled_position)
    for order in open_orders:
        if not order["valid"]:
            continue
        if order["side"] == _BUY:
            projected += order["remaining_quantity"]
        else:
            projected -= order["remaining_quantity"]
    return projected


def plan_instrument_action(
    *,
    instrument: str,
    target_quantity: int | None,
    filled_position: int,
    open_orders: list[dict[str, Any]] | None = None,
    sellable_quantity: int | None = None,
    now: datetime,
    lot_increment: int = 100,
    min_lot: int = 100,
    hard_blocked_reason: str | None = None,
    not_executable_reason: str | None = None,
) -> dict[str, Any]:
    """Compute the two-dimension action/execution-state plan for one instrument.

    ``target_quantity=None`` means there is no new valid target: the line is
    ``NO_ACTION`` and the previous plan is left untouched (design 8.4:
    ``NO_ACTION`` never deletes a still-valid target).
    """

    instrument = str(instrument).strip().upper()
    if not instrument:
        raise ValueError("instrument is required")
    filled = int(filled_position)
    if filled < 0:
        raise ValueError("filled position must not be negative")
    if lot_increment < 1 or min_lot < 1:
        raise ValueError("lot rules must be positive")
    sellable = filled if sellable_quantity is None else int(sellable_quantity)
    if not 0 <= sellable <= filled:
        raise ValueError("sellable quantity must stay within the filled position")
    orders = [
        normalize_open_order(order, now=now) for order in (open_orders or [])
    ]
    valid = [order for order in orders if order["valid"]]
    projected = projected_position(filled, orders)
    notes: list[str] = []

    if target_quantity is None:
        return {
            "instrument": instrument,
            "action": ACTION_NO_ACTION,
            "execution_state": STATE_BLOCKED if hard_blocked_reason else STATE_READY,
            "target_quantity": None,
            "filled_position": filled,
            "projected_position": projected,
            "sellable_quantity": sellable,
            "order_plan": [],
            "blocked_reason": hard_blocked_reason,
            "wait_reason": None,
            "notes": ["no_valid_target_previous_retained"],
            "model_version": RECOMMENDATION_ACTION_MODEL_VERSION,
        }
    target = int(target_quantity)
    if target < 0:
        raise ValueError("target quantity must not be negative")

    if target > filled:
        action = ACTION_BUY
    elif target == 0 and filled > 0:
        action = ACTION_EXIT
    elif 0 < target < filled:
        action = ACTION_SELL
    elif target == filled and (filled > 0 or valid):
        # 含 target==filled==0 但有偏离目标的有效订单：按 HOLD 取消它们。
        action = ACTION_HOLD
    else:  # target == filled == 0 且无有效订单
        action = ACTION_NO_ACTION

    direction = None
    if action == ACTION_BUY:
        direction = _BUY
    elif action in {ACTION_SELL, ACTION_EXIT}:
        direction = _SELL
    plan: list[dict[str, Any]] = []

    if action == ACTION_HOLD:
        # 已达目标：任何会使持仓偏离目标的有效订单必须同计划取消（8.4）。
        for order in valid:
            plan.append(
                {
                    "op": OP_CANCEL,
                    "order_id": order["order_id"],
                    "side": order["side"],
                    "quantity": order["remaining_quantity"],
                    "reason": "target_already_filled",
                }
            )
    elif direction is not None:
        gross_needed = abs(target - filled)
        opposite = [order for order in valid if order["side"] != direction]
        same_side = sorted(
            (order for order in valid if order["side"] == direction),
            key=lambda order: (
                order["created_at"] is None,
                str(order["created_at"]),
                order["order_id"],
            ),
        )
        for order in opposite:
            plan.append(
                {
                    "op": OP_CANCEL,
                    "order_id": order["order_id"],
                    "side": order["side"],
                    "quantity": order["remaining_quantity"],
                    "reason": "opposite_of_final_target",
                }
            )
        covered = 0
        for order in same_side:
            remaining = order["remaining_quantity"]
            if covered >= gross_needed:
                plan.append(
                    {
                        "op": OP_CANCEL,
                        "order_id": order["order_id"],
                        "side": order["side"],
                        "quantity": remaining,
                        "reason": "excess_beyond_final_target",
                    }
                )
                continue
            if covered + remaining > gross_needed:
                adjusted = gross_needed - covered
                plan.append(
                    {
                        "op": OP_REPLACE,
                        "order_id": order["order_id"],
                        "side": order["side"],
                        "quantity": adjusted,
                        "previous_quantity": remaining,
                        "reason": "trim_to_final_target",
                    }
                )
                covered += adjusted
                continue
            plan.append(
                {
                    "op": OP_KEEP,
                    "order_id": order["order_id"],
                    "side": order["side"],
                    "quantity": remaining,
                    "reason": "covers_final_target",
                }
            )
            covered += remaining
        new_quantity = gross_needed - covered
        if new_quantity > 0:
            if direction == _BUY:
                rounded = new_quantity // lot_increment * lot_increment
                if rounded < new_quantity:
                    notes.append("new_buy_rounded_down_to_lot")
                new_quantity = rounded
                if 0 < new_quantity < min_lot:
                    notes.append("new_buy_below_min_lot")
                    new_quantity = 0
            else:
                # 卖单已被取消/保留的有效卖单占用了部分可卖额度，新卖单只补差额。
                headroom = sellable - covered
                if new_quantity > headroom:
                    notes.append("sell_clamped_by_sellable_quantity")
                    new_quantity = max(0, headroom)
            if new_quantity > 0:
                plan.append(
                    {
                        "op": OP_NEW,
                        "order_id": "",
                        "side": direction,
                        "quantity": new_quantity,
                        "reason": "target_minus_projected_position",
                    }
                )
        if action == ACTION_EXIT:
            notes.append(f"remaining_sellable_quantity={sellable}")

    # 执行状态：硬门 > 暂不可执行 > 部分成交 > 仅取消 > 过期重报 > 就绪。
    carried_ids = {
        entry["order_id"] for entry in plan if entry["op"] in {OP_KEEP, OP_REPLACE}
    }
    partial = any(
        order["order_id"] in carried_ids
        and order["filled_quantity"] > 0
        and order["remaining_quantity"] > 0
        for order in valid
    )
    expired_uncovered = (
        any(order["expired"] for order in orders)
        and not any(
            entry["op"] in {OP_KEEP, OP_REPLACE} for entry in plan
        )
        and any(entry["op"] == OP_NEW for entry in plan)
    )
    only_cancels = bool(plan) and all(entry["op"] == OP_CANCEL for entry in plan)
    if hard_blocked_reason:
        execution_state = STATE_BLOCKED
    elif not_executable_reason or "sell_clamped_by_sellable_quantity" in notes:
        execution_state = STATE_WAIT
    elif partial:
        execution_state = STATE_PARTIAL
    elif only_cancels:
        execution_state = STATE_CANCELLED
    elif expired_uncovered:
        execution_state = STATE_EXPIRED
    else:
        execution_state = STATE_READY

    sellable_shortfall = "sell_clamped_by_sellable_quantity" in notes
    return {
        "instrument": instrument,
        "action": action,
        "execution_state": execution_state,
        "target_quantity": target,
        "filled_position": filled,
        "projected_position": projected,
        "sellable_quantity": sellable,
        "order_plan": plan,
        "blocked_reason": hard_blocked_reason,
        "wait_reason": (
            not_executable_reason
            or ("sellable_quantity_unavailable" if sellable_shortfall else None)
        ),
        "notes": notes,
        "model_version": RECOMMENDATION_ACTION_MODEL_VERSION,
    }


def plan_account_actions(
    instruments: list[dict[str, Any]], *, now: datetime
) -> list[dict[str, Any]]:
    """Vector form of :func:`plan_instrument_action` for one account decision."""

    return [
        plan_instrument_action(now=now, **item)
        for item in sorted(instruments, key=lambda entry: str(entry["instrument"]))
    ]
