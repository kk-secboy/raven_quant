from __future__ import annotations

import hashlib
import itertools
import json
import math
from datetime import date, timedelta
from typing import Any, TypeAlias

from quant_data.execution_contract import build_strategy_execution_contract

ParameterValue: TypeAlias = int | float | str

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
        "portfolio_construction",
    }
)

PORTFOLIO_CONSTRUCTION_CANDIDATES = (
    "topk_equal_weight",
    "industry_neutral_qp",
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def portfolio_trial_comparability_evidence(
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove TopK and QP ranked the same frozen signal under the same costs."""

    constructions = {
        str((item.get("parameters") or {}).get("portfolio_construction") or "")
        for item in trials
    }
    if (
        len(trials) != 2
        or constructions != set(PORTFOLIO_CONSTRUCTION_CANDIDATES)
        or any(item.get("status") != "succeeded" for item in trials)
    ):
        raise ValueError("portfolio comparison requires succeeded TopK and QP trials")
    contracts = {
        _canonical_sha256(
            build_strategy_execution_contract(dict(item.get("config") or {}))
        )
        for item in trials
    }
    if len(contracts) != 1:
        raise ValueError("portfolio trials changed the shared cost/execution contract")
    evidence: dict[str, Any] = {
        "execution_contract_sha256": next(iter(contracts)),
        "segments": {},
    }
    for segment in ("in_sample", "out_of_sample"):
        segment_metrics = [
            ((item.get("metrics") or {}).get(segment) or {}) for item in trials
        ]
        provenance = [dict(metrics.get("provenance") or {}) for metrics in segment_metrics]
        prediction_hashes = {
            item.get("formal_model_predictions_sha256") for item in provenance
        }
        checkpoint_hashes = {
            item.get("formal_model_checkpoint_sha256") for item in provenance
        }
        signal_identities = {
            item.get("model_signal_identity_sha256") for item in provenance
        }
        dataset_identities = {
            item.get("dataset_identity_sha256") for item in provenance
        }
        admission_bindings = {
            item.get("formal_model_admission_binding_sha256") for item in provenance
        }
        pre_final_cutoffs = {item.get("pre_final_cutoff") for item in provenance}
        cost_models = {
            _canonical_sha256(metrics.get("cost_model"))
            for metrics in segment_metrics
            if isinstance(metrics.get("cost_model"), dict)
            and bool(metrics.get("cost_model"))
        }
        if (
            any(
                item.get("evaluation_mode") != "pre_final_portfolio_trial"
                or item.get("evaluation_scope") != "pre_final_only"
                or item.get("final_oos_opened") is not False
                for item in provenance
            )
            or len(prediction_hashes) != 1
            or not _is_sha256(next(iter(prediction_hashes), None))
            or len(checkpoint_hashes) != 1
            or not _is_sha256(next(iter(checkpoint_hashes), None))
            or len(signal_identities) != 1
            or not _is_sha256(next(iter(signal_identities), None))
            or len(dataset_identities) != 1
            or not _is_sha256(next(iter(dataset_identities), None))
            or len(admission_bindings) != 1
            or not _is_sha256(next(iter(admission_bindings), None))
            or len(pre_final_cutoffs) != 1
            or not str(next(iter(pre_final_cutoffs), ""))
            or len(cost_models) != 1
            or any(
                not isinstance(metrics.get("cost_model"), dict)
                or not metrics.get("cost_model")
                for metrics in segment_metrics
            )
        ):
            raise ValueError(
                "TopK/QP trials did not use identical model predictions and costs"
            )
        evidence["segments"][segment] = {
            "model_predictions_sha256": next(iter(prediction_hashes)),
            "model_checkpoint_sha256": next(iter(checkpoint_hashes)),
            "model_signal_identity_sha256": next(iter(signal_identities)),
            "dataset_identity_sha256": next(iter(dataset_identities)),
            "formal_model_admission_binding_sha256": next(iter(admission_bindings)),
            "pre_final_cutoff": next(iter(pre_final_cutoffs)),
            "cost_model_sha256": next(iter(cost_models)),
        }
    evidence["evidence_sha256"] = _canonical_sha256(evidence)
    return evidence


def merge_admitted_trial_ledgers(
    formal_admission_binding: dict[str, Any],
) -> dict[str, Any]:
    """Merge model and fin_quant attempts into one immutable prior-trial family."""

    sources = [
        (
            "model",
            (formal_admission_binding.get("model_grid") or {}).get(
                "multiple_testing"
            ),
        )
    ]
    quant_bundle = formal_admission_binding.get("quant_bundle")
    if isinstance(quant_bundle, dict):
        sources.append(("fin_quant", quant_bundle.get("multiple_testing")))
    names: list[str] = []
    sharpes: list[float] = []
    source_bindings: list[dict[str, Any]] = []
    for source_name, evidence in sources:
        if not isinstance(evidence, dict):
            raise ValueError(f"{source_name} multiple-testing evidence is missing")
        source_names = evidence.get("trial_names")
        source_sharpes = evidence.get("trial_daily_sharpes")
        if (
            evidence.get("final_oos_opened") is not False
            or not isinstance(source_names, list)
            or not isinstance(source_sharpes, list)
            or int(evidence.get("trial_count") or 0) != len(source_names)
            or len(source_sharpes) != len(source_names)
        ):
            raise ValueError(f"{source_name} multiple-testing evidence is incomplete")
        names.extend(f"{source_name}:{name}" for name in source_names)
        sharpes.extend(float(value) for value in source_sharpes)
        source_bindings.append(
            {
                "source": source_name,
                "trial_count": len(source_names),
                "evidence_sha256": evidence.get("evidence_sha256"),
            }
        )
    if len(set(names)) != len(names) or any(not math.isfinite(value) for value in sharpes):
        raise ValueError("admitted multiple-testing trial family is invalid")
    merged: dict[str, Any] = {
        "contract_version": "parameter-experiment-prior-trials-v1",
        "source": "independent_qlib_recompute",
        "final_oos_opened": False,
        "trial_names": names,
        "trial_daily_sharpes": sharpes,
        "trial_count": len(names),
        "source_bindings": source_bindings,
        "formal_admission_binding_sha256": formal_admission_binding.get(
            "binding_sha256"
        ),
    }
    payload = json.dumps(
        merged, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    merged["evidence_sha256"] = hashlib.sha256(payload).hexdigest()
    return merged


def select_frozen_portfolio_config(experiment: dict[str, Any]) -> dict[str, Any]:
    """Select one statistically accepted, pre-final-only portfolio trial."""

    if experiment.get("status") != "succeeded":
        raise ValueError("portfolio construction competition has not succeeded")
    summary = experiment.get("summary")
    trials = experiment.get("trials")
    if not isinstance(summary, dict) or not isinstance(trials, list):
        raise ValueError("portfolio construction competition evidence is incomplete")
    if (
        int(summary.get("trial_count") or 0) != 2
        or len(trials) != 2
        or summary.get("final_oos_opened") is not False
        or int(summary.get("governed_trial_count") or 0) < 2
    ):
        raise ValueError("portfolio construction competition trial ledger is invalid")
    warnings = set(summary.get("warnings") or [])
    if warnings:
        raise ValueError(
            "portfolio construction competition is blocked: " + ", ".join(sorted(warnings))
        )
    best_index = summary.get("best_trial_index")
    if best_index is None:
        raise ValueError("no portfolio construction trial passed DSR")
    matches = [item for item in trials if int(item.get("trial_index", -1)) == int(best_index)]
    if len(matches) != 1:
        raise ValueError("portfolio construction winner is ambiguous")
    winner = matches[0]
    construction = (winner.get("parameters") or {}).get("portfolio_construction")
    config = winner.get("config")
    metrics = winner.get("metrics") or {}
    if (
        winner.get("status") != "succeeded"
        or construction not in PORTFOLIO_CONSTRUCTION_CANDIDATES
        or not isinstance(config, dict)
        or config.get("portfolio_construction") != construction
        or float(
            (metrics.get("out_of_sample") or {}).get(
                "deflated_sharpe_probability", 0.0
            )
            or 0.0
        )
        < 0.95
        or any(
            ((metrics.get(segment) or {}).get("provenance") or {}).get(
                "evaluation_scope"
            )
            != "pre_final_only"
            or ((metrics.get(segment) or {}).get("provenance") or {}).get(
                "final_oos_opened"
            )
            is not False
            for segment in ("in_sample", "out_of_sample")
        )
    ):
        raise ValueError("portfolio construction winner is not governed pre-final evidence")
    contracts = {
        json.dumps(
            build_strategy_execution_contract(dict(item.get("config") or {})),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in trials
    }
    if len(contracts) != 1:
        raise ValueError("portfolio construction trials used different execution contracts")
    comparability = portfolio_trial_comparability_evidence(trials)
    if summary.get("comparability") != comparability:
        raise ValueError("portfolio construction comparability evidence is invalid")
    frozen = json.loads(json.dumps(config, ensure_ascii=False, sort_keys=True))
    payload = json.dumps(
        frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "experiment_id": str(experiment.get("id") or ""),
        "trial_index": int(best_index),
        "portfolio_construction": construction,
        "portfolio_config": frozen,
        "portfolio_config_sha256": hashlib.sha256(payload).hexdigest(),
        "governed_trial_count": int(summary["governed_trial_count"]),
        "final_oos_opened": False,
    }


def normalize_parameter_grid(
    parameter_grid: dict[str, list[ParameterValue]], *, max_trials: int = 27
) -> tuple[dict[str, list[ParameterValue]], list[dict[str, ParameterValue]]]:
    if not parameter_grid:
        raise ValueError("parameter grid must not be empty")
    unknown = sorted(set(parameter_grid) - TUNABLE_PARAMETERS)
    if unknown:
        raise ValueError("unsupported experiment parameters: " + ", ".join(unknown))
    normalized: dict[str, list[ParameterValue]] = {}
    trial_count = 1
    for name in sorted(parameter_grid):
        values = parameter_grid[name]
        if not values or len(values) > 9:
            raise ValueError(f"{name} must contain between 1 and 9 values")
        clean: list[ParameterValue] = []
        for value in values:
            if name == "portfolio_construction":
                if value not in PORTFOLIO_CONSTRUCTION_CANDIDATES:
                    raise ValueError(
                        "portfolio_construction must be topk_equal_weight or "
                        "industry_neutral_qp"
                    )
            else:
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


def build_portfolio_construction_trials(
    baseline_config: dict[str, Any],
) -> tuple[dict[str, list[ParameterValue]], list[dict[str, Any]]]:
    """Build the fixed TopK/QP comparison without changing execution or costs.

    This is deliberately a two-member, preregistered competition.  QP tuning
    remains a separate governed experiment; adding optimizer knobs here would
    turn a construction comparison into an uncounted parameter search.
    """

    baseline_contract = build_strategy_execution_contract(baseline_config)
    parameter_grid: dict[str, list[ParameterValue]] = {
        "portfolio_construction": list(PORTFOLIO_CONSTRUCTION_CANDIDATES)
    }
    trials: list[dict[str, Any]] = []
    for construction in PORTFOLIO_CONSTRUCTION_CANDIDATES:
        config = {**baseline_config, "portfolio_construction": construction}
        if build_strategy_execution_contract(config) != baseline_contract:
            raise ValueError(
                "portfolio construction trials must share one cost/execution contract"
            )
        trials.append(
            {
                "parameters": {"portfolio_construction": construction},
                "config": config,
            }
        )
    return parameter_grid, trials


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


def split_model_portfolio_period(
    start: date,
    end: date,
    *,
    label_horizon_days: int = 1,
    inner_validation_fraction: float = 0.35,
) -> dict[str, Any]:
    """Reserve an inner validation prefix before the portfolio competition.

    The admitted model recipe is refit on the frozen training segment, uses
    this prefix only for early stopping, and predicts the later two portfolio
    comparison segments.  This prevents either policy from seeing the sealed
    final OOS and avoids fitting on the same dates used to rank TopK versus QP.
    """

    days = (end - start).days
    if days < 252:
        raise ValueError(
            "model portfolio competition requires at least 252 calendar days"
        )
    if not 0.20 <= inner_validation_fraction <= 0.50:
        raise ValueError("inner validation fraction must be between 0.20 and 0.50")
    experiment_start = start + timedelta(days=round(days * inner_validation_fraction))
    return split_research_period(
        experiment_start,
        end,
        label_horizon_days=label_horizon_days,
    )


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
    trials: list[dict[str, Any]], parameter_grid: dict[str, list[ParameterValue]]
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
            name != "portfolio_construction"
            and len(parameter_grid[name]) > 1
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
