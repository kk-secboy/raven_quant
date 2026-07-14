from __future__ import annotations

from datetime import date, datetime, time, timedelta
from math import floor
from typing import Any
from zoneinfo import ZoneInfo

ASHARE_TIMEZONE = ZoneInfo("Asia/Shanghai")
ASHARE_SESSIONS = ((time(10, 0), time(11, 20)), (time(13, 30), time(14, 50)))


def normalize_execution_policy(config: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = config or {}
    algorithm = str(raw.get("execution_algorithm", "twap")).strip().lower()
    if algorithm not in {"twap", "vwap"}:
        raise ValueError("execution_algorithm must be twap or vwap")
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
        "slice_minutes": slice_minutes,
        "max_slices": max_slices,
        "max_participation": max_participation,
        "lot_size": 100,
        "sessions": [["10:00", "11:20"], ["13:30", "14:50"]],
        "volume_profile": profile,
        "cash_tolerance": cash_tolerance,
        "equity_tolerance": equity_tolerance,
        "position_tolerance": position_tolerance,
        "max_snapshot_age_seconds": max_snapshot_age_seconds,
    }


def build_execution_slices(
    *,
    quantity: float,
    side: str,
    trade_date: date,
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    normalized = normalize_execution_policy(policy)
    integer_quantity = int(quantity)
    if quantity != integer_quantity or integer_quantity <= 0:
        raise ValueError("A-share execution quantity must be a positive integer")
    normalized_side = side.strip().lower()
    if normalized_side not in {"buy", "sell"}:
        raise ValueError("execution side must be buy or sell")
    lot_size = int(normalized["lot_size"])
    if normalized_side == "buy" and integer_quantity % lot_size:
        raise ValueError("A-share buy quantity must be a multiple of 100 shares")

    if normalized["execution_algorithm"] == "twap":
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
