"""Deterministic execution-policy catalog (design draft 6.5).

Six versioned policies plan how one final account target is released:

- ``next_bar_baseline`` / ``twap_execution`` / ``vwap_execution``: intraday
  slicing via :func:`build_execution_slices`.
- ``participation_capped_slicing``: explicit volume-participation policy via
  :func:`plan_participation_capped_slices`; every slice stays within the
  participation cap and any quantity the capped liquidity cannot absorb is
  reported as ``unallocated_quantity`` (expires or is re-quoted by the upper
  layer) — never planned or booked as filled.
- ``wait_cancel_replace``: limit-wait/cancel/re-quote plan via
  :func:`plan_wait_cancel_replace`; cancel releases the still-unfilled
  remainder exactly once and replace is cancel+new at a new limit (aligned
  with :mod:`quant_platform.simulation_order_state` semantics, which never
  grows a working order).
- ``multi_day_transition``: cross-day slicing via
  :func:`plan_multi_day_transition`; per-day quantities are allocated
  deterministically by participation capacity and remaining days, each day
  aligned with the order ``not_after`` window (15:00 Asia/Shanghai).

All policies are pure functions of their inputs (same input, same output) and
carry a versioned ``execution_policy_id``. None of them treats an unfilled
target as a filled position: fills are accounted by the engine from actual
fills only.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from math import ceil, floor, isfinite
from typing import Any
from zoneinfo import ZoneInfo

from .market_rules import order_unit_rules, validate_order_quantity

ASHARE_TIMEZONE = ZoneInfo("Asia/Shanghai")
ASHARE_SESSIONS = ((time(10, 0), time(11, 20)), (time(13, 30), time(14, 50)))
ORDER_NOT_AFTER_TIME = time(15, 0)
PRICE_TICK = 0.01

# Canonical design-draft 6.5 policy ids per internal algorithm name.
EXECUTION_POLICY_IDS = {
    "next_bar": "next_bar_baseline",
    "twap": "twap_execution",
    "vwap": "vwap_execution",
    "participation_capped_slicing": "participation_capped_slicing",
    "wait_cancel_replace": "wait_cancel_replace",
    "multi_day_transition": "multi_day_transition",
}
_POLICY_ID_ALIASES = {
    "next_bar_baseline": "next_bar",
    "twap_execution": "twap",
    "vwap_execution": "vwap",
}
_UNFILLED_DISPOSITION = "expire_or_requote_next_cycle"


def normalize_execution_policy(config: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = config or {}
    algorithm = str(raw.get("execution_algorithm", "twap")).strip().lower()
    algorithm = _POLICY_ID_ALIASES.get(algorithm, algorithm)
    if algorithm not in EXECUTION_POLICY_IDS:
        raise ValueError(
            "execution_algorithm must be one of "
            + ", ".join(sorted(EXECUTION_POLICY_IDS))
        )
    slice_minutes = int(raw.get("slice_minutes", 20))
    if slice_minutes < 5 or slice_minutes > 30 or slice_minutes % 5:
        raise ValueError("slice_minutes must be a multiple of 5 between 5 and 30")
    max_slices = int(raw.get("max_slices", 24))
    if max_slices < 1 or max_slices > 64:
        raise ValueError("max_slices must be between 1 and 64")
    max_participation = float(raw.get("max_participation", 0.01))
    if not 0 < max_participation <= 0.20:
        raise ValueError("max_participation must be in (0, 0.20]")
    profile = raw.get("volume_profile")
    if profile is not None and not isinstance(profile, list):
        raise ValueError("volume_profile must be a list")
    slot_volumes = raw.get("slot_volumes")
    if slot_volumes is not None:
        _validate_slot_volumes(slot_volumes)
    execution_frequency = str(
        raw.get("execution_frequency") or raw.get("frequency") or "5min"
    ).strip()
    if execution_frequency not in {"1min", "5min"}:
        raise ValueError("execution_frequency must be 1min or 5min")
    # wait_cancel_replace knobs: check cadence reuses execution_frequency.
    wait_checks = int(raw.get("wait_checks", 6))
    if not 1 <= wait_checks <= 64:
        raise ValueError("wait_checks must be between 1 and 64")
    max_replaces = int(raw.get("max_replaces", 3))
    if not 0 <= max_replaces <= 16:
        raise ValueError("max_replaces must be between 0 and 16")
    replace_step_bps = float(raw.get("replace_step_bps", 10.0))
    if not 0 <= replace_step_bps <= 500:
        raise ValueError("replace_step_bps must be in [0, 500]")
    initial_offset_bps = float(raw.get("initial_offset_bps", 0.0))
    if not -500 <= initial_offset_bps <= 500:
        raise ValueError("initial_offset_bps must be in [-500, 500]")
    # multi_day_transition knobs.
    transition_days = int(raw.get("transition_days", 2))
    if not 2 <= transition_days <= 5:
        raise ValueError("transition_days must be between 2 and 5")
    daily_volumes = raw.get("daily_volumes")
    if daily_volumes is not None:
        if not isinstance(daily_volumes, list) or len(daily_volumes) != transition_days:
            raise ValueError("daily_volumes must be a list with one entry per transition day")
        if any(float(volume) <= 0 for volume in daily_volumes):
            raise ValueError("daily_volumes entries must be positive")
        daily_volumes = [float(volume) for volume in daily_volumes]
    intraday_algorithm = raw.get("intraday_algorithm")
    if intraday_algorithm is not None:
        intraday_algorithm = _POLICY_ID_ALIASES.get(
            str(intraday_algorithm).strip().lower(), str(intraday_algorithm).strip().lower()
        )
        if intraday_algorithm not in {"twap", "vwap", "next_bar"}:
            raise ValueError("intraday_algorithm must be twap, vwap, or next_bar")
    cash_tolerance = float(raw.get("cash_tolerance", 1.0))
    equity_tolerance = float(raw.get("equity_tolerance", 10.0))
    position_tolerance = float(raw.get("position_tolerance", 0.0))
    max_snapshot_age_seconds = int(raw.get("max_snapshot_age_seconds", 120))
    if min(cash_tolerance, equity_tolerance, position_tolerance) < 0:
        raise ValueError("reconciliation tolerances must not be negative")
    if not 10 <= max_snapshot_age_seconds <= 3600:
        raise ValueError("max_snapshot_age_seconds must be between 10 and 3600")
    return {
        "execution_algorithm": algorithm,
        "execution_policy_id": EXECUTION_POLICY_IDS[algorithm],
        "slice_minutes": slice_minutes,
        "max_slices": max_slices,
        "max_participation": max_participation,
        "execution_frequency": execution_frequency,
        # Fallback slice granularity used only when the instrument (and so its
        # board order-unit rules from market_rules) is not supplied.
        "lot_size": int(raw.get("lot_size", 100)),
        "sessions": [["10:00", "11:20"], ["13:30", "14:50"]],
        "volume_profile": profile,
        "slot_volumes": slot_volumes,
        "wait_checks": wait_checks,
        "max_replaces": max_replaces,
        "replace_step_bps": replace_step_bps,
        "initial_offset_bps": initial_offset_bps,
        "transition_days": transition_days,
        "daily_volumes": daily_volumes,
        "intraday_algorithm": intraday_algorithm,
        "cash_tolerance": cash_tolerance,
        "equity_tolerance": equity_tolerance,
        "position_tolerance": position_tolerance,
        "max_snapshot_age_seconds": max_snapshot_age_seconds,
    }


def _validate_slot_volumes(slot_volumes: Any) -> None:
    if not isinstance(slot_volumes, list) or not slot_volumes:
        raise ValueError("slot_volumes must be a non-empty list")
    seen: set[str] = set()
    for item in slot_volumes:
        if not isinstance(item, dict):
            raise ValueError("each slot_volumes item must be an object")
        raw_time = str(item.get("time") or "")
        try:
            hour, minute = (int(part) for part in raw_time.split(":"))
            slot_time = time(hour, minute)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid slot_volumes time {raw_time!r}") from exc
        if raw_time in seen or not _inside_execution_sessions(slot_time):
            raise ValueError("slot_volumes times must be unique and inside execution sessions")
        seen.add(raw_time)
        if float(item.get("volume") or 0) <= 0:
            raise ValueError("slot_volumes volumes must be positive")


def build_execution_slices(
    *,
    quantity: float,
    side: str,
    trade_date: date,
    policy: dict[str, Any],
    signal_at: datetime | None = None,
    instrument: str | None = None,
) -> list[dict[str, Any]]:
    normalized = normalize_execution_policy(policy)
    algorithm = str(normalized["execution_algorithm"])
    if algorithm not in {"next_bar", "twap", "vwap"}:
        raise ValueError(
            f"{algorithm} is planned via its dedicated planner "
            "(plan_participation_capped_slices / plan_wait_cancel_replace / "
            "plan_multi_day_transition), not build_execution_slices"
        )
    integer_quantity, _normalized_side, lot_size = _order_lot_context(
        quantity=quantity,
        side=side,
        trade_date=trade_date,
        normalized=normalized,
        instrument=instrument,
    )

    if algorithm == "next_bar":
        slots = [
            _next_bar_slot(
                trade_date,
                signal_at=signal_at,
                frequency=str(normalized["execution_frequency"]),
            )
        ]
        weights = [1.0]
    elif normalized["execution_algorithm"] == "twap":
        slots = _twap_slots(
            trade_date,
            int(normalized["slice_minutes"]),
            int(normalized["max_slices"]),
        )
        weights = [1.0 / len(slots)] * len(slots)
    else:
        slots, weights = _vwap_slots(
            trade_date,
            normalized.get("volume_profile"),
            int(normalized["max_slices"]),
        )

    full_lots, odd_lot = divmod(integer_quantity, lot_size)
    usable = min(len(slots), full_lots + (1 if odd_lot else 0))
    if usable < 1:
        raise ValueError("execution quantity is below one tradable slice")
    slots = slots[:usable]
    weights = weights[:usable]
    total_weight = sum(weights)
    weights = [weight / total_weight for weight in weights]
    lot_allocations = _largest_remainder(full_lots, weights)
    quantities = [lots * lot_size for lots in lot_allocations]
    if odd_lot:
        quantities[-1] += odd_lot
    result = []
    for sequence, (slot, slice_quantity, _weight) in enumerate(
        zip(slots, quantities, weights, strict=True), start=1
    ):
        if slice_quantity <= 0:
            continue
        result.append(
            {
                "sequence": sequence,
                "scheduled_for": slot.isoformat(),
                "quantity": slice_quantity,
                "target_weight": slice_quantity / integer_quantity,
                "algorithm": normalized["execution_algorithm"],
                "max_participation": normalized["max_participation"],
            }
        )
    if sum(item["quantity"] for item in result) != integer_quantity:
        raise RuntimeError("execution slices do not reconcile to the parent quantity")
    return result


def execution_time_slots(
    *,
    trade_date: date,
    policy: dict[str, Any],
    signal_at: datetime | None = None,
) -> list[datetime]:
    """Return the configured intraday slice timestamps without requiring an order quantity."""

    normalized = normalize_execution_policy(policy)
    algorithm = str(normalized["execution_algorithm"])
    if algorithm == "participation_capped_slicing":
        return [
            slot for slot, _volume in _parse_slot_volumes(
                normalized.get("slot_volumes"), trade_date
            )
        ]
    if algorithm in {"wait_cancel_replace", "multi_day_transition"}:
        raise ValueError(f"{algorithm} has no single-day intraday slot schedule")
    if algorithm == "next_bar":
        return [
            _next_bar_slot(
                trade_date,
                signal_at=signal_at,
                frequency=str(normalized["execution_frequency"]),
            )
        ]
    return _twap_slots(
        trade_date,
        int(normalized["slice_minutes"]),
        int(normalized["max_slices"]),
    )


def _bar_slots(trade_date: date, interval_minutes: int) -> list[datetime]:
    slots: list[datetime] = []
    for start, end in ASHARE_SESSIONS:
        current = datetime.combine(trade_date, start, ASHARE_TIMEZONE)
        boundary = datetime.combine(trade_date, end, ASHARE_TIMEZONE)
        while current <= boundary:
            slots.append(current)
            current += timedelta(minutes=interval_minutes)
    return slots


def _next_bar_slot(
    trade_date: date,
    *,
    signal_at: datetime | None,
    frequency: str,
) -> datetime:
    interval_minutes = 1 if frequency == "1min" else 5
    slots = _bar_slots(trade_date, interval_minutes)
    if signal_at is None:
        return slots[0]
    normalized_signal = (
        signal_at.replace(tzinfo=ASHARE_TIMEZONE)
        if signal_at.tzinfo is None
        else signal_at.astimezone(ASHARE_TIMEZONE)
    )
    if normalized_signal.date() > trade_date:
        raise ValueError("next-bar signal timestamp is after the execution date")
    if normalized_signal.date() < trade_date:
        return slots[0]
    next_slot = next((slot for slot in slots if slot > normalized_signal), None)
    if next_slot is None:
        raise ValueError("next-bar signal has no later execution bar in the governed window")
    return next_slot


def _twap_slots(trade_date: date, interval_minutes: int, max_slices: int) -> list[datetime]:
    slots = []
    for start, end in ASHARE_SESSIONS:
        current = datetime.combine(trade_date, start, ASHARE_TIMEZONE)
        boundary = datetime.combine(trade_date, end, ASHARE_TIMEZONE)
        while current <= boundary:
            slots.append(current)
            current += timedelta(minutes=interval_minutes)
    return _evenly_limit(slots, max_slices)


def _vwap_slots(
    trade_date: date, profile: Any, max_slices: int
) -> tuple[list[datetime], list[float]]:
    if not isinstance(profile, list) or not profile:
        raise ValueError("VWAP execution requires non-empty volume_profile evidence")
    parsed: list[tuple[datetime, float]] = []
    seen = set()
    for item in profile:
        if not isinstance(item, dict):
            raise ValueError("each volume_profile item must be an object")
        raw_time = str(item.get("time") or "")
        try:
            hour, minute = (int(part) for part in raw_time.split(":"))
            slot_time = time(hour, minute)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid volume_profile time {raw_time!r}") from exc
        if raw_time in seen or not _inside_execution_sessions(slot_time):
            raise ValueError("volume_profile times must be unique and inside execution sessions")
        seen.add(raw_time)
        weight = float(item.get("weight") or 0)
        if weight <= 0:
            raise ValueError("volume_profile weights must be positive")
        parsed.append((datetime.combine(trade_date, slot_time, ASHARE_TIMEZONE), weight))
    parsed.sort(key=lambda item: item[0])
    if len(parsed) > max_slices:
        selected = _evenly_limit(parsed, max_slices)
    else:
        selected = parsed
    slots = [item[0] for item in selected]
    weights = [item[1] for item in selected]
    return slots, weights


def _inside_execution_sessions(value: time) -> bool:
    return any(start <= value <= end for start, end in ASHARE_SESSIONS)


def _evenly_limit(values: list[Any], limit: int) -> list[Any]:
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    indexes = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[index] for index in indexes]


def _largest_remainder(total: int, weights: list[float]) -> list[int]:
    if total < 0 or not weights:
        raise ValueError("lot allocation inputs are invalid")
    raw = [total * weight for weight in weights]
    allocated = [floor(value) for value in raw]
    remaining = total - sum(allocated)
    order = sorted(range(len(raw)), key=lambda index: raw[index] - allocated[index], reverse=True)
    for index in order[:remaining]:
        allocated[index] += 1
    return allocated


def _order_lot_context(
    *,
    quantity: float,
    side: str,
    trade_date: date,
    normalized: dict[str, Any],
    instrument: str | None,
) -> tuple[int, str, int]:
    """Validate quantity/side and resolve the lot size against board rules."""

    integer_quantity = int(quantity)
    if quantity != integer_quantity or integer_quantity <= 0:
        raise ValueError("A-share execution quantity must be a positive integer")
    normalized_side = side.strip().lower()
    if normalized_side not in {"buy", "sell"}:
        raise ValueError("execution side must be buy or sell")
    rules = order_unit_rules(instrument, trade_date) if instrument is not None else None
    lot_size = int(normalized["lot_size"])
    if rules is not None:
        # Slice chunks must themselves be valid per-board buy quantities.
        if lot_size < rules.min_lot or (lot_size - rules.min_lot) % rules.lot_increment:
            lot_size = rules.min_lot
    if normalized_side == "buy":
        if rules is not None:
            violations = validate_order_quantity(
                instrument, integer_quantity, side="buy", trade_date=trade_date
            )
            if violations:
                raise ValueError("; ".join(violations))
        elif integer_quantity % lot_size:
            raise ValueError("A-share buy quantity must be a multiple of 100 shares")
    return integer_quantity, normalized_side, lot_size


def _parse_slot_volumes(
    slot_volumes: Any, trade_date: date
) -> list[tuple[datetime, float]]:
    """Parse validated slot-volume evidence into sorted (slot, volume) pairs."""

    if slot_volumes is None:
        raise ValueError(
            "participation_capped_slicing requires non-empty slot_volumes evidence"
        )
    _validate_slot_volumes(slot_volumes)
    parsed = [
        (
            datetime.combine(
                trade_date,
                time(*(int(part) for part in str(item["time"]).split(":"))),
                ASHARE_TIMEZONE,
            ),
            float(item["volume"]),
        )
        for item in slot_volumes
    ]
    parsed.sort(key=lambda item: item[0])
    return parsed


def plan_participation_capped_slices(
    *,
    quantity: float,
    side: str,
    trade_date: date,
    policy: dict[str, Any],
    instrument: str | None = None,
) -> dict[str, Any]:
    """Explicit participation-capped slicing policy (design 6.5).

    Each slot releases at most ``max_participation`` of its expected volume
    (rounded down to whole lots). Quantity the capped liquidity cannot absorb
    is reported as ``unallocated_quantity`` with disposition
    ``expire_or_requote_next_cycle`` — it is never planned into a slice and
    never counted as filled.
    """

    normalized = normalize_execution_policy(policy)
    if normalized["execution_algorithm"] != "participation_capped_slicing":
        raise ValueError("policy execution_algorithm must be participation_capped_slicing")
    integer_quantity, normalized_side, lot_size = _order_lot_context(
        quantity=quantity,
        side=side,
        trade_date=trade_date,
        normalized=normalized,
        instrument=instrument,
    )
    slots = _parse_slot_volumes(normalized.get("slot_volumes"), trade_date)
    cap = float(normalized["max_participation"])

    full_lots, odd_lot = divmod(integer_quantity, lot_size)
    remaining_lots = full_lots
    allocations: list[tuple[datetime, float, int]] = []  # (slot, volume, lots)
    for slot, volume in slots:
        if remaining_lots <= 0:
            break
        capacity_lots = int(floor(volume * cap / lot_size))
        lots = min(capacity_lots, remaining_lots)
        if lots <= 0:
            continue
        allocations.append((slot, volume, lots))
        remaining_lots -= lots
    odd_allocated = 0
    if odd_lot and allocations:
        # Sells may carry one odd-lot remainder; it rides the last slice only
        # while that slice stays within the participation cap.
        slot, volume, lots = allocations[-1]
        if (lots * lot_size + odd_lot) <= floor(volume * cap) + 1e-9:
            odd_allocated = odd_lot
    slices: list[dict[str, Any]] = []
    for sequence, (slot, volume, lots) in enumerate(allocations, start=1):
        slice_quantity = lots * lot_size + (odd_allocated if sequence == len(allocations) else 0)
        if slice_quantity <= 0:
            continue
        participation = slice_quantity / volume
        if participation - cap > 1e-9:
            raise RuntimeError("participation-capped slice exceeds the participation cap")
        slices.append(
            {
                "sequence": sequence,
                "scheduled_for": slot.isoformat(),
                "quantity": slice_quantity,
                "target_weight": slice_quantity / integer_quantity,
                "algorithm": "participation_capped_slicing",
                "policy_id": "participation_capped_slicing",
                "max_participation": cap,
                "participation": participation,
                "expected_slot_volume": volume,
            }
        )
    placed = sum(item["quantity"] for item in slices)
    unallocated = integer_quantity - placed
    return {
        "policy_id": "participation_capped_slicing",
        "slices": slices,
        "allocated_quantity": placed,
        "unallocated_quantity": unallocated,
        "unallocated_disposition": _UNFILLED_DISPOSITION,
        "max_participation": cap,
    }


def plan_wait_cancel_replace(
    *,
    quantity: float,
    side: str,
    trade_date: date,
    policy: dict[str, Any],
    reference_price: float,
    signal_at: datetime | None = None,
    instrument: str | None = None,
) -> dict[str, Any]:
    """Limit-wait / cancel / re-quote plan (design 6.5).

    The plan is deterministic given ``reference_price``: round 1 rests a limit
    order at the reference price (plus the frozen ``initial_offset_bps``,
    positive = more aggressive); each check cadence bar the unfilled order
    waits; after ``wait_checks`` bars it is cancelled and re-quoted one
    ``replace_step_bps`` step more aggressive, up to ``max_replaces``
    replacements. Whatever is still unfilled after the final round expires —
    it is never booked as filled.

    Semantics align with :mod:`quant_platform.simulation_order_state`: a
    cancel releases the still-unfilled remainder exactly once, and a replace
    is cancel + new order at the new limit for the live remainder — it never
    grows the working order and never double-releases cash.
    """

    normalized = normalize_execution_policy(policy)
    if normalized["execution_algorithm"] != "wait_cancel_replace":
        raise ValueError("policy execution_algorithm must be wait_cancel_replace")
    integer_quantity, normalized_side, _lot_size = _order_lot_context(
        quantity=quantity,
        side=side,
        trade_date=trade_date,
        normalized=normalized,
        instrument=instrument,
    )
    reference = float(reference_price)
    if not isfinite(reference) or reference <= 0:
        raise ValueError("wait_cancel_replace requires a positive reference_price")

    frequency = str(normalized["execution_frequency"])
    interval_minutes = 1 if frequency == "1min" else 5
    slots = _bar_slots(trade_date, interval_minutes)
    if signal_at is None:
        start_index = 0
    else:
        normalized_signal = (
            signal_at.replace(tzinfo=ASHARE_TIMEZONE)
            if signal_at.tzinfo is None
            else signal_at.astimezone(ASHARE_TIMEZONE)
        )
        if normalized_signal.date() > trade_date:
            raise ValueError("wait_cancel_replace signal timestamp is after the execution date")
        if normalized_signal.date() < trade_date:
            start_index = 0
        else:
            start_index = next(
                (index for index, slot in enumerate(slots) if slot > normalized_signal),
                None,
            )
            if start_index is None:
                raise ValueError(
                    "wait_cancel_replace signal has no later check bar in the governed window"
                )

    direction = 1.0 if normalized_side == "buy" else -1.0
    wait_checks = int(normalized["wait_checks"])
    max_replaces = int(normalized["max_replaces"])
    step = float(normalized["replace_step_bps"]) / 1e4
    limit = _tick_round(
        reference * (1.0 + direction * float(normalized["initial_offset_bps"]) / 1e4),
        normalized_side,
    )

    rounds: list[dict[str, Any]] = []
    index = start_index
    for round_no in range(1, max_replaces + 2):
        if index >= len(slots):
            break
        cancel_index = index + wait_checks
        truncated = cancel_index >= len(slots)
        cancel_slot = slots[-1] if truncated else slots[cancel_index]
        final = truncated or round_no == max_replaces + 1
        rounds.append(
            {
                "round": round_no,
                "quantity": integer_quantity,
                "quantity_note": "plan-time target; runtime submits the live unfilled remainder",
                "limit_price": limit,
                "submitted_at": slots[index].isoformat(),
                "cancel_after": cancel_slot.isoformat(),
                "on_unfilled": "expire" if final else "cancel_and_requote",
            }
        )
        if final:
            break
        limit = _tick_round(limit * (1.0 + direction * step), normalized_side)
        index = cancel_index
    return {
        "policy_id": "wait_cancel_replace",
        "side": normalized_side,
        "quantity": integer_quantity,
        "reference_price": reference,
        "check_frequency": frequency,
        "wait_checks": wait_checks,
        "max_replaces": max_replaces,
        "replace_step_bps": float(normalized["replace_step_bps"]),
        "rounds": rounds,
        "final_action": "expire_unfilled_remainder",
        "unfilled_disposition": _UNFILLED_DISPOSITION,
        "semantics": {
            "cancel": "releases the still-unfilled remainder exactly once",
            "replace": (
                "cancel plus new order at the new limit for the live remainder; "
                "never grows the order"
            ),
        },
    }


def _tick_round(price: float, side: str) -> float:
    ticks = price / PRICE_TICK
    rounded = ceil(ticks - 1e-9) if side == "buy" else floor(ticks + 1e-9)
    return round(rounded * PRICE_TICK, 2)


def plan_multi_day_transition(
    *,
    quantity: float,
    side: str,
    trade_dates: list[date],
    policy: dict[str, Any],
    instrument: str | None = None,
) -> dict[str, Any]:
    """Cross-day transition slicing (design 6.5).

    Per-day quantities are deterministic: the base daily share is the live
    remainder spread over the remaining days (rounded up to whole lots),
    capped by ``max_participation`` of the day's expected volume when
    ``daily_volumes`` evidence is supplied. Each day is aligned with the order
    execution window (``not_before`` 10:00, ``not_after`` 15:00 Asia/Shanghai,
    matching the engine's 15:00 expiry). Quantity the per-day capacity cannot
    absorb is reported as ``unallocated_quantity`` — it expires or is
    re-quoted by the upper layer, never counted as filled. Conservation:
    ``allocated_quantity + unallocated_quantity == quantity``.
    """

    normalized = normalize_execution_policy(policy)
    if normalized["execution_algorithm"] != "multi_day_transition":
        raise ValueError("policy execution_algorithm must be multi_day_transition")
    if not isinstance(trade_dates, list) or not trade_dates:
        raise ValueError("multi_day_transition requires explicit trade_dates")
    days = [day if isinstance(day, date) else date.fromisoformat(str(day)) for day in trade_dates]
    transition_days = int(normalized["transition_days"])
    if len(days) != transition_days:
        raise ValueError("trade_dates length must equal transition_days")
    if any(later <= earlier for earlier, later in zip(days, days[1:], strict=False)):
        raise ValueError("trade_dates must be strictly increasing")
    daily_volumes = normalized.get("daily_volumes")
    if daily_volumes is not None and len(daily_volumes) != transition_days:
        raise ValueError("daily_volumes must cover every transition day")

    integer_quantity, normalized_side, lot_size = _order_lot_context(
        quantity=quantity,
        side=side,
        trade_date=days[0],
        normalized=normalized,
        instrument=instrument,
    )
    cap = float(normalized["max_participation"])
    intraday = normalized.get("intraday_algorithm")

    remaining = integer_quantity
    day_plans: list[dict[str, Any]] = []
    allocated = 0
    for index, day in enumerate(days):
        days_left = transition_days - index
        if days_left == 1:
            base = remaining  # last day takes the live remainder (sells keep odd lots)
        else:
            base = min(remaining, ceil(remaining / days_left / lot_size) * lot_size)
        if daily_volumes is not None:
            day_volume = float(daily_volumes[index])
            if days_left == 1 and normalized_side == "sell" and remaining % lot_size:
                # Last-day sells may release the odd-lot remainder in shares.
                capacity = min(remaining, int(floor(day_volume * cap)))
            else:
                capacity = int(floor(day_volume * cap / lot_size)) * lot_size
            day_quantity = min(base, capacity)
        else:
            capacity = None
            day_quantity = base
        if day_quantity < 0:
            raise RuntimeError("multi-day allocation produced a negative slice")
        entry: dict[str, Any] = {
            "day": index + 1,
            "trade_date": day.isoformat(),
            "quantity": day_quantity,
            "not_before": datetime.combine(day, ASHARE_SESSIONS[0][0], ASHARE_TIMEZONE).isoformat(),
            "not_after": datetime.combine(day, ORDER_NOT_AFTER_TIME, ASHARE_TIMEZONE).isoformat(),
        }
        if daily_volumes is not None:
            entry["expected_day_volume"] = float(daily_volumes[index])
            entry["participation_capacity"] = capacity
        if intraday and day_quantity > 0:
            sub_policy = dict(policy, execution_algorithm=intraday)
            entry["slices"] = build_execution_slices(
                quantity=day_quantity,
                side=normalized_side,
                trade_date=day,
                policy=sub_policy,
                instrument=instrument,
            )
        day_plans.append(entry)
        remaining -= day_quantity
        allocated += day_quantity
    return {
        "policy_id": "multi_day_transition",
        "side": normalized_side,
        "transition_days": transition_days,
        "days": day_plans,
        "allocated_quantity": allocated,
        "unallocated_quantity": integer_quantity - allocated,
        "unallocated_disposition": _UNFILLED_DISPOSITION,
        "max_participation": cap,
    }
