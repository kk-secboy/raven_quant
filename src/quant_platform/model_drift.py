from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

MODEL_DRIFT_TRIGGER_CONTRACT_VERSION = "model-drift-trigger-v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_persistent_drift_evidence(
    *,
    nav_rows: Sequence[Mapping[str, Any]],
    trading_days: Sequence[date],
    as_of: date,
    dataset_lineage_id: str,
    metric: str = "cost_after_excess_return",
    window_trading_days: int = 20,
    consecutive_windows: int = 3,
    threshold: float = 0.0,
    comparison: str = "below",
) -> dict[str, Any] | None:
    """Return a fail-closed drift trigger built from certified paper returns.

    The windows are adjacent and non-overlapping.  This deliberately requires
    ``window_trading_days * consecutive_windows`` fresh sessions instead of
    treating three heavily overlapping 20-day means as independent evidence.
    ``twr_daily_return`` is already net of simulated execution costs; subtracting
    the bound benchmark therefore gives the frozen cost-after excess metric.
    """

    if window_trading_days < 20 or consecutive_windows < 3:
        raise ValueError("drift policy is weaker than the governed minimum")
    if comparison not in {"below", "above"}:
        raise ValueError("drift comparison must be below or above")
    if not metric.strip() or not dataset_lineage_id.strip():
        raise ValueError("drift evidence identity is incomplete")
    if not math.isfinite(float(threshold)):
        raise ValueError("drift threshold must be finite")

    eligible_days = sorted({day for day in trading_days if day <= as_of})
    required = window_trading_days * consecutive_windows
    if len(eligible_days) < required or eligible_days[-1] != as_of:
        return None
    expected_days = eligible_days[-required:]

    rows_by_day: dict[date, Mapping[str, Any]] = {}
    for row in nav_rows:
        raw_day = row.get("trade_date")
        day = raw_day if isinstance(raw_day, date) else date.fromisoformat(str(raw_day))
        if day in rows_by_day:
            raise ValueError("duplicate NAV row in drift evidence")
        rows_by_day[day] = row
    if set(rows_by_day) != set(expected_days):
        return None

    daily_excess: list[float] = []
    compact_rows: list[dict[str, Any]] = []
    for day in expected_days:
        row = rows_by_day[day]
        account_return = row.get("twr_daily_return")
        benchmark_return = row.get("benchmark_return")
        certified = row.get("performance_certified") is True
        stale = row.get("has_stale_prices") is True
        status = str(row.get("status") or "")
        if account_return is None or benchmark_return is None or not certified or stale:
            return None
        if status not in {"ok", "succeeded", "certified"}:
            return None
        account_value = float(account_return)
        benchmark_value = float(benchmark_return)
        excess = account_value - benchmark_value
        if not all(math.isfinite(value) for value in (account_value, benchmark_value, excess)):
            return None
        daily_excess.append(excess)
        compact_rows.append(
            {
                "trade_date": day.isoformat(),
                "twr_daily_return": account_value,
                "benchmark_return": benchmark_value,
            }
        )

    window_means: list[float] = []
    windows: list[dict[str, Any]] = []
    for index in range(consecutive_windows):
        start = index * window_trading_days
        stop = start + window_trading_days
        values = daily_excess[start:stop]
        mean = math.fsum(values) / float(window_trading_days)
        window_means.append(mean)
        windows.append(
            {
                "start": expected_days[start].isoformat(),
                "end": expected_days[stop - 1].isoformat(),
                "mean": mean,
            }
        )

    crossed = all(
        value <= float(threshold) if comparison == "below" else value >= float(threshold)
        for value in window_means
    )
    if not crossed:
        return None

    evidence = {
        "contract_version": MODEL_DRIFT_TRIGGER_CONTRACT_VERSION,
        "as_of": as_of.isoformat(),
        "dataset_lineage_id": dataset_lineage_id,
        "metric": metric,
        "observed": window_means[-1],
        "threshold": float(threshold),
        "comparison": comparison,
        "consecutive_windows": consecutive_windows,
        "window_trading_days": window_trading_days,
        "window_semantics": "adjacent_non_overlapping",
        "windows": windows,
        "nav_evidence_sha256": _canonical_sha256(compact_rows),
    }
    return evidence
