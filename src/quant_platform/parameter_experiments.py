from __future__ import annotations

import itertools
import math
from datetime import date, timedelta
from typing import Any

TUNABLE_PARAMETERS = frozenset(
    {
        "topk",
        "n_drop",
        "max_position_weight",
        "max_daily_turnover",
        "max_industry_deviation",
        "max_size_deviation",
        "optimizer_alpha_weight",
        "optimizer_tracking_penalty",
        "optimizer_turnover_penalty",
        "stop_loss",
        "take_profit_partial",
        "take_profit",
        "max_drawdown_reduce",
        "max_drawdown_liquidate",
        "max_volume_participation",
    }
)


def normalize_parameter_grid(
    parameter_grid: dict[str, list[int | float]], *, max_trials: int = 27
) -> tuple[dict[str, list[int | float]], list[dict[str, int | float]]]:
    if not parameter_grid:
        raise ValueError("parameter grid must not be empty")
    unknown = sorted(set(parameter_grid) - TUNABLE_PARAMETERS)
    if unknown:
        raise ValueError("unsupported experiment parameters: " + ", ".join(unknown))
    normalized: dict[str, list[int | float]] = {}
    trial_count = 1
    for name in sorted(parameter_grid):
        values = parameter_grid[name]
        if not values or len(values) > 9:
            raise ValueError(f"{name} must contain between 1 and 9 values")
        clean: list[int | float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} values must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} values must be finite")
            if value not in clean:
                clean.append(value)
        normalized[name] = clean
        trial_count *= len(clean)
    if trial_count > max_trials:
        raise ValueError(f"parameter grid expands to {trial_count} trials; maximum is {max_trials}")
    names = list(normalized)
    trials = [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(normalized[name] for name in names))
    ]
    return normalized, trials


def split_research_period(
    start: date, end: date, *, label_horizon_days: int = 1
) -> dict[str, Any]:
    days = (end - start).days
    if days < 126:
        raise ValueError("parameter experiments require at least 126 calendar days")
    if label_horizon_days < 1:
        raise ValueError("label horizon must be positive")
    split = start + timedelta(days=round(days * 0.60))
    embargo = max(5, label_horizon_days)
    return {
        "in_sample": {
            "start": start.isoformat(),
            "end": (split - timedelta(days=label_horizon_days)).isoformat(),
        },
        "out_of_sample": {
            "start": (split + timedelta(days=embargo)).isoformat(),
            "end": end.isoformat(),
        },
        "purge_days": label_horizon_days,
        "embargo_days": embargo,
    }


def _number(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def evaluate_trial(
    in_sample: dict[str, Any], out_of_sample: dict[str, Any]
) -> tuple[float, list[str]]:
    is_ir = _number(in_sample, "information_ratio")
    oos_ir = _number(out_of_sample, "information_ratio")
    is_excess = _number(in_sample, "annualized_excess_return")
    oos_excess = _number(out_of_sample, "annualized_excess_return")
    oos_drawdown = abs(_number(out_of_sample, "max_drawdown"))
    oos_turnover = _number(out_of_sample, "average_turnover")
    robustness = _number(out_of_sample, "robustness_pass_rate")
    deflated_sharpe = _number(out_of_sample, "deflated_sharpe_probability")
    score = oos_ir + 0.50 * oos_excess + 0.25 * robustness - 0.50 * oos_drawdown
    score -= 0.10 * oos_turnover
    warnings: list[str] = []
    if is_excess > 0 and oos_excess <= 0:
        warnings.append("oos_sign_reversal")
    if is_ir > 0.20 and oos_ir < is_ir * 0.50:
        warnings.append("performance_decay")
    if oos_drawdown > 0.25:
        warnings.append("oos_drawdown_high")
    if robustness < 0.60:
        warnings.append("oos_robustness_low")
    if deflated_sharpe < 0.95:
        warnings.append("deflated_sharpe_failed")
    return round(score, 8), warnings


def summarize_trials(
    trials: list[dict[str, Any]], parameter_grid: dict[str, list[int | float]]
) -> dict[str, Any]:
    successful = [
        item
        for item in trials
        if item.get("status") == "succeeded"
        and _number(
            (item.get("metrics") or {}).get("out_of_sample", {}),
            "deflated_sharpe_probability",
        )
        >= 0.95
    ]
    ranked = sorted(successful, key=lambda item: float(item.get("score", -math.inf)), reverse=True)
    warnings: list[str] = []
    if any(item.get("status") == "succeeded" for item in trials) and not successful:
        warnings.append("all_trials_failed_deflated_sharpe")
    if len(ranked) >= 2 and float(ranked[0]["score"]) - float(ranked[1]["score"]) < 0.05:
        warnings.append("fragile_ranking")
    if ranked:
        best_parameters = ranked[0]["parameters"]
        if any(
            len(parameter_grid[name]) > 1
            and value in {min(parameter_grid[name]), max(parameter_grid[name])}
            for name, value in best_parameters.items()
        ):
            warnings.append("boundary_optimum")
    else:
        best_parameters = None
    return {
        "trial_count": len(trials),
        "succeeded_count": len(successful),
        "failed_count": len(trials) - len(successful),
        "statistically_rejected_count": sum(
            item.get("status") == "succeeded" for item in trials
        )
        - len(successful),
        "best_trial_index": ranked[0]["trial_index"] if ranked else None,
        "best_parameters": best_parameters,
        "warnings": warnings,
        "leaderboard": [
            {
                "trial_index": item["trial_index"],
                "parameters": item["parameters"],
                "score": item["score"],
                "warnings": item.get("warnings", []),
                "in_sample": _compact_metrics(item.get("metrics", {}).get("in_sample", {})),
                "out_of_sample": _compact_metrics(item.get("metrics", {}).get("out_of_sample", {})),
            }
            for item in ranked
        ],
    }


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, float | int | bool | None]:
    keys = (
        "annualized_return",
        "annualized_excess_return",
        "information_ratio",
        "sharpe_ratio",
        "max_drawdown",
        "average_turnover",
        "robustness_pass_rate",
        "deflated_sharpe_probability",
        "trading_days",
    )
    return {key: metrics.get(key) for key in keys}
