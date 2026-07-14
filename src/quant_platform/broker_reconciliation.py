from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def compare_broker_snapshot(
    *,
    expected: dict[str, Any],
    observed: dict[str, Any],
    submitted_orders: dict[str, str | None],
    known_client_order_ids: set[str],
    cash_tolerance: float,
    equity_tolerance: float,
    position_tolerance: float,
    max_snapshot_age_seconds: int,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    current = now or datetime.now(UTC)
    differences: list[dict[str, Any]] = []
    broker_as_of = _aware_datetime(observed.get("as_of"))
    age_seconds = max(0.0, (current - broker_as_of).total_seconds())
    if age_seconds > max_snapshot_age_seconds:
        differences.append(
            {
                "type": "stale_snapshot",
                "age_seconds": age_seconds,
                "limit_seconds": max_snapshot_age_seconds,
            }
        )

    _compare_number(
        differences,
        kind="cash_mismatch",
        expected=float(expected["cash"]),
        observed=float(observed["cash"]),
        tolerance=cash_tolerance,
    )
    _compare_number(
        differences,
        kind="equity_mismatch",
        expected=float(expected["equity"]),
        observed=float(observed["equity"]),
        tolerance=equity_tolerance,
    )

    expected_positions = _position_map(expected.get("positions", []))
    observed_positions = _position_map(observed.get("positions", []))
    for instrument in sorted(set(expected_positions) | set(observed_positions)):
        expected_quantity = expected_positions.get(instrument, 0.0)
        observed_quantity = observed_positions.get(instrument, 0.0)
        if abs(expected_quantity - observed_quantity) > position_tolerance:
            differences.append(
                {
                    "type": "position_mismatch",
                    "instrument": instrument,
                    "expected": expected_quantity,
                    "observed": observed_quantity,
                    "difference": observed_quantity - expected_quantity,
                    "tolerance": position_tolerance,
                }
            )

    observed_records = [*observed.get("orders", []), *observed.get("trades", [])]
    broker_client_ids = {
        str(item.get("client_order_id") or "")
        for item in observed_records
        if str(item.get("client_order_id") or "")
    }
    unknown = sorted(broker_client_ids - known_client_order_ids)
    if unknown:
        differences.append({"type": "unknown_broker_orders", "client_order_ids": unknown})
    missing = sorted(set(submitted_orders) - broker_client_ids)
    if missing:
        differences.append({"type": "missing_broker_orders", "client_order_ids": missing})
    for client_order_id, expected_order_id in submitted_orders.items():
        records = [
            item
            for item in observed_records
            if str(item.get("client_order_id") or "") == client_order_id
        ]
        observed_order_ids = {
            str(item.get("order_id") or "") for item in records if str(item.get("order_id") or "")
        }
        if expected_order_id and observed_order_ids and expected_order_id not in observed_order_ids:
            differences.append(
                {
                    "type": "broker_order_id_mismatch",
                    "client_order_id": client_order_id,
                    "expected": expected_order_id,
                    "observed": sorted(observed_order_ids),
                }
            )
        terminal_failures = sorted(
            {
                str(item.get("status") or "").lower()
                for item in records
                if str(item.get("status") or "").lower()
                in {"rejected", "error", "canceled", "cancelled"}
            }
        )
        if terminal_failures:
            differences.append(
                {
                    "type": "broker_order_terminal_failure",
                    "client_order_id": client_order_id,
                    "statuses": terminal_failures,
                }
            )
    return differences


def validate_broker_snapshot(payload: dict[str, Any], *, account_ref: str) -> dict[str, Any]:
    if payload.get("status") != "ok" or payload.get("environment") != "sandbox":
        raise ValueError("broker snapshot did not attest sandbox readiness")
    if payload.get("account_ref") != account_ref:
        raise ValueError("broker snapshot account reference mismatch")
    _aware_datetime(payload.get("as_of"))
    for field in ("cash", "equity"):
        try:
            float(payload[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"broker snapshot {field} is invalid") from exc
    result = dict(payload)
    for field in ("positions", "orders", "trades"):
        value = result.get(field, [])
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError(f"broker snapshot {field} must be a list of objects")
        if len(value) > 10_000:
            raise ValueError(f"broker snapshot {field} exceeds 10000 records")
        result[field] = value
    _position_map(result["positions"])
    return result


def _position_map(rows: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        instrument = str(row.get("instrument") or "").strip()
        if not instrument or instrument in result:
            raise ValueError("position instruments must be non-empty and unique")
        try:
            quantity = float(row.get("quantity"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"position quantity is invalid for {instrument}") from exc
        if quantity < 0:
            raise ValueError(f"position quantity is negative for {instrument}")
        if quantity:
            result[instrument] = quantity
    return result


def _aware_datetime(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("broker snapshot as_of is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("broker snapshot as_of must include timezone")
    return parsed.astimezone(UTC)


def _compare_number(
    differences: list[dict[str, Any]],
    *,
    kind: str,
    expected: float,
    observed: float,
    tolerance: float,
) -> None:
    if abs(expected - observed) > tolerance:
        differences.append(
            {
                "type": kind,
                "expected": expected,
                "observed": observed,
                "difference": observed - expected,
                "tolerance": tolerance,
            }
        )
