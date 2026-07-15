from __future__ import annotations

import math
from datetime import date
from typing import Any

from .rdagent_runtime import validate_duration

RESEARCH_PERIOD_KEYS = (
    "train_start",
    "train_end",
    "valid_start",
    "valid_end",
    "test_start",
    "test_end",
)


def normalize_research_schedule_payload(
    payload: dict[str, Any], *, max_loops: int
) -> dict[str, Any]:
    """Validate and normalize one durable scheduled RD-Agent research request."""

    objective = str(payload.get("objective") or "").strip()
    dataset = str(payload.get("dataset") or "").strip()
    requested_by = str(payload.get("requested_by") or "scheduler").strip()
    if len(objective) < 10 or len(objective) > 2000:
        raise ValueError("rdagent_research objective must contain 10 to 2000 characters")
    if not dataset:
        raise ValueError("rdagent_research requires a Qlib dataset")
    if len(requested_by) < 2 or len(requested_by) > 100:
        raise ValueError("rdagent_research requested_by must contain 2 to 100 characters")

    try:
        loop_n = int(payload.get("loop_n", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("rdagent_research loop_n must be an integer") from exc
    if loop_n < 1 or loop_n > max_loops:
        raise ValueError(f"rdagent_research loop_n must be between 1 and {max_loops}")
    duration = validate_duration(str(payload.get("duration") or "30m"))

    raw_periods = payload.get("periods")
    if not isinstance(raw_periods, dict):
        raise ValueError("rdagent_research requires train, validation, and test periods")
    periods: dict[str, str] = {}
    parsed: dict[str, date] = {}
    for key in RESEARCH_PERIOD_KEYS:
        value = str(raw_periods.get(key) or "")
        try:
            parsed[key] = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"rdagent_research {key} must be an ISO date") from exc
        periods[key] = parsed[key].isoformat()
    if not (
        parsed["train_start"]
        <= parsed["train_end"]
        < parsed["valid_start"]
        <= parsed["valid_end"]
        < parsed["test_start"]
        <= parsed["test_end"]
    ):
        raise ValueError(
            "rdagent_research train, validation, and test periods must be ordered "
            "and non-overlapping"
        )
    return {
        "objective": objective,
        "dataset": dataset,
        "loop_n": loop_n,
        "duration": duration,
        "requested_by": requested_by,
        "periods": periods,
    }


def derive_rolling_research_periods(
    calendar_days: list[str],
    *,
    train_days: int,
    validation_days: int,
    test_days: int,
) -> dict[str, str]:
    """Build non-overlapping rolling windows from an actual Qlib trading calendar."""

    if min(train_days, validation_days, test_days) < 1:
        raise ValueError("research window lengths must be positive")
    ordered = sorted(dict.fromkeys(calendar_days))
    total = train_days + validation_days + test_days
    if len(ordered) < total:
        raise ValueError(
            f"Qlib calendar has {len(ordered)} trading days; continuous research "
            f"requires {total}"
        )
    selected = ordered[-total:]
    train_end = train_days - 1
    valid_start = train_days
    valid_end = train_days + validation_days - 1
    test_start = train_days + validation_days
    return {
        "train_start": selected[0],
        "train_end": selected[train_end],
        "valid_start": selected[valid_start],
        "valid_end": selected[valid_end],
        "test_start": selected[test_start],
        "test_end": selected[-1],
    }


def select_latest_program_dataset(
    datasets: list[dict[str, Any]], *, lineage_id: str
) -> dict[str, Any] | None:
    """Select the newest reproducible dataset without crossing its approved lineage."""

    eligible = [
        item
        for item in datasets
        if item.get("ready")
        and item.get("reproducible")
        and item.get("lineage_verified")
        and item.get("lineage_id") == lineage_id
        and item.get("end_date")
        and (item.get("provenance") or {}).get("dataset_identity_sha256")
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (str(item["end_date"]), str(item["name"])))


def factor_rank_score(candidate: dict[str, Any]) -> float:
    """Return a deterministic score using independent Qlib evidence only."""

    evaluation = candidate.get("latest_evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("gate_status") != "passed":
        return -math.inf
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, dict):
        return -math.inf

    def number(name: str, default: float = 0.0) -> float:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return default

    return round(
        abs(number("icir"))
        + abs(number("rank_icir"))
        + 4.0 * max(0.0, number("cost_adjusted_return"))
        - 0.25 * max(0.0, number("turnover")),
        10,
    )


def rank_factor_candidates(
    candidates: list[dict[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("factor selection limit must be positive")
    eligible = []
    for candidate in candidates:
        score = factor_rank_score(candidate)
        if math.isfinite(score):
            eligible.append({**candidate, "automation_score": score})
    return sorted(
        eligible,
        key=lambda item: (-float(item["automation_score"]), str(item.get("id") or "")),
    )[:limit]


def strategy_comparison_score(metrics: dict[str, Any] | None) -> float:
    """Compare already validated strategy backtests without bypassing approval gates."""

    if not isinstance(metrics, dict):
        return -math.inf

    def number(name: str, default: float = 0.0) -> float:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return default

    information_ratio = number("information_ratio", number("sharpe_ratio"))
    return round(
        information_ratio
        + 0.50 * number("annualized_excess_return")
        - 0.50 * abs(number("max_drawdown"))
        - 0.10 * max(0.0, number("average_turnover")),
        10,
    )


def choose_champion(
    baseline: dict[str, Any], challenger: dict[str, Any]
) -> dict[str, Any]:
    baseline_score = strategy_comparison_score(baseline.get("metrics"))
    challenger_score = strategy_comparison_score(challenger.get("metrics"))
    challenger_wins = challenger_score > baseline_score
    winner = challenger if challenger_wins else baseline
    return {
        "winner_version_id": winner["strategy_version_id"],
        "baseline_score": baseline_score,
        "challenger_score": challenger_score,
        "decision": "challenger" if challenger_wins else "baseline",
    }
