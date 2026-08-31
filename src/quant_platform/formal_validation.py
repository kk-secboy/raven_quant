from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_data.snapshot_lineage import canonical_sha256
from quant_platform.statistical_validation import (
    DEFLATED_SHARPE_METHOD_VERSION,
    STATISTICAL_CONTRACT_VERSION,
    paired_moving_block_bootstrap,
)

FORMAL_VALIDATION_CONTRACT_VERSION = (
    "formal-validation-evidence-v4-incomplete-family-bonferroni"
)
PRE_FINAL_HISTORY_CONTRACT_VERSION = "pre-final-history-calendar-v1"
SIGNAL_DECAY_FRONTIER_VERSION = "contiguous-zero-delay-frontier-v2"
FACTOR_SCORE_FAMILYWISE_ALPHA = 0.05
FACTOR_SCORE_INCOMPLETE_FAMILY_MULTIPLE_TESTING_VERSION = (
    "factor-score-incomplete-family-bonferroni-v1"
)
CONSERVATIVE_BONFERRONI_INCOMPLETE_FAMILY_STATUS = (
    "conservative_bonferroni_incomplete_historical_family"
)
NOT_COMPUTABLE_INCOMPLETE_FAMILY_STATUS = (
    "not_computable_incomplete_historical_family"
)
FROZEN_STRATEGY_OUTER_SCOPE = "pre_final_history_current_frozen_strategy"
FACTOR_SCORE_INCOMPLETE_FAMILY_DSR_VERSION = (
    "factor-score-incomplete-family-dsr-not-computable-v1"
)
PAIRED_BOOTSTRAP_EVIDENCE_CONTRACT_VERSION = (
    "paired-moving-block-bootstrap-evidence-v1"
)
PAIRED_BOOTSTRAP_METHOD = "paired_circular_moving_block_bootstrap"
PAIRED_BOOTSTRAP_ESTIMAND = "mean_candidate_net_return_minus_baseline_return"
PAIRED_BOOTSTRAP_INPUT_ALIGNMENT = "ordered_complete_case_pairs"
PAIRED_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
PAIRED_BOOTSTRAP_PARAMETER_KEYS = frozenset({"block_size", "samples", "seed"})
PAIRED_BOOTSTRAP_EVIDENCE_KEYS = frozenset(
    {
        "status",
        "contract_version",
        "statistical_contract_version",
        "method",
        "estimand",
        "input_alignment",
        "input_sha256",
        "observations",
        "observed_mean_difference",
        "confidence_level",
        "confidence_interval_95",
        "probability_positive",
        "one_sided_p_value",
        *PAIRED_BOOTSTRAP_PARAMETER_KEYS,
        "evidence_sha256",
    }
)


def _require_sha256(value: Any, *, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be SHA256")
    return normalized


def _require_plain_int(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return normalized


def validate_paired_bootstrap_parameters(
    value: Any,
    *,
    observations: int | None = None,
) -> dict[str, int]:
    """Validate the exact frozen parameter set used by the paired bootstrap.

    The parameters are deliberately supplied independently of claimed evidence.
    An approval path must never learn its sample count, block size, or seed from
    the result it is trying to verify.
    """

    if not isinstance(value, Mapping):
        raise ValueError("paired bootstrap parameters must be an object")
    keys = set(value)
    if keys != PAIRED_BOOTSTRAP_PARAMETER_KEYS:
        missing = sorted(PAIRED_BOOTSTRAP_PARAMETER_KEYS - keys)
        extra = sorted(keys - PAIRED_BOOTSTRAP_PARAMETER_KEYS)
        raise ValueError(
            "paired bootstrap parameters have invalid fields "
            f"(missing={missing}, extra={extra})"
        )
    block_size = _require_plain_int(
        value["block_size"],
        label="paired bootstrap block_size",
        minimum=1,
    )
    samples = _require_plain_int(
        value["samples"],
        label="paired bootstrap samples",
        minimum=100,
    )
    seed = _require_plain_int(
        value["seed"],
        label="paired bootstrap seed",
        minimum=0,
    )
    if observations is not None:
        count = _require_plain_int(
            observations,
            label="paired bootstrap observations",
            minimum=30,
        )
        if block_size > count:
            raise ValueError("paired bootstrap block_size exceeds observations")
    return {"block_size": block_size, "samples": samples, "seed": seed}


def paired_bootstrap_parameters_from_config(config: Any) -> dict[str, int]:
    """Resolve the runner's frozen bootstrap parameters from version config."""

    if not isinstance(config, Mapping):
        raise ValueError("strategy config must be an object")
    return validate_paired_bootstrap_parameters(
        {
            "block_size": config.get("bootstrap_block_days", 20),
            "samples": config.get("bootstrap_samples", 2000),
            "seed": config.get("validation_seed", 0),
        }
    )


def _paired_return_arrays(
    candidate_net_returns: pd.Series | Sequence[float],
    baseline_returns: pd.Series | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.asarray(pd.Series(candidate_net_returns), dtype=float)
    baseline = np.asarray(pd.Series(baseline_returns), dtype=float)
    if (
        len(candidate) != len(baseline)
        or len(candidate) < 30
        or not np.isfinite(candidate).all()
        or not np.isfinite(baseline).all()
    ):
        raise ValueError(
            "paired bootstrap inputs require equal finite series of at least 30 rows"
        )
    return candidate, baseline


def _daily_return_pairs(daily_returns: Any) -> tuple[pd.Series, pd.Series]:
    if not isinstance(daily_returns, pd.DataFrame):
        try:
            daily_returns = pd.DataFrame(daily_returns)
        except (TypeError, ValueError) as exc:
            raise ValueError("daily returns must be tabular") from exc
    required = {"return", "cost", "bench"}
    missing = sorted(required - set(daily_returns.columns))
    if missing:
        raise ValueError(f"daily returns are missing required columns: {missing}")
    paired = pd.concat(
        [
            (
                pd.to_numeric(daily_returns["return"], errors="coerce")
                - pd.to_numeric(daily_returns["cost"], errors="coerce")
            ).rename("candidate"),
            pd.to_numeric(daily_returns["bench"], errors="coerce").rename(
                "baseline"
            ),
        ],
        axis=1,
        join="inner",
    ).dropna()
    return paired["candidate"], paired["baseline"]


def build_paired_bootstrap_evidence(
    candidate_net_returns: pd.Series | Sequence[float],
    baseline_returns: pd.Series | Sequence[float],
    *,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Build canonical paired-bootstrap evidence from already aligned returns."""

    candidate, baseline = _paired_return_arrays(
        candidate_net_returns,
        baseline_returns,
    )
    frozen = validate_paired_bootstrap_parameters(
        parameters,
        observations=len(candidate),
    )
    calculated = paired_moving_block_bootstrap(
        candidate,
        baseline,
        block_size=frozen["block_size"],
        samples=frozen["samples"],
        seed=frozen["seed"],
    )
    payload = {
        "status": "ok",
        "contract_version": PAIRED_BOOTSTRAP_EVIDENCE_CONTRACT_VERSION,
        "statistical_contract_version": STATISTICAL_CONTRACT_VERSION,
        "method": PAIRED_BOOTSTRAP_METHOD,
        "estimand": PAIRED_BOOTSTRAP_ESTIMAND,
        "input_alignment": PAIRED_BOOTSTRAP_INPUT_ALIGNMENT,
        "input_sha256": canonical_sha256(
            {
                "candidate_net_returns": [float(item) for item in candidate],
                "baseline_returns": [float(item) for item in baseline],
            }
        ),
        "observations": int(calculated["observations"]),
        "observed_mean_difference": float(
            calculated["observed_mean_difference"]
        ),
        "confidence_level": PAIRED_BOOTSTRAP_CONFIDENCE_LEVEL,
        "confidence_interval_95": [
            float(item) for item in calculated["confidence_interval_95"]
        ],
        "probability_positive": float(calculated["probability_positive"]),
        "one_sided_p_value": float(calculated["one_sided_p_value"]),
        **frozen,
    }
    return {**payload, "evidence_sha256": canonical_sha256(payload)}


def build_paired_bootstrap_evidence_from_daily_returns(
    daily_returns: Any,
    *,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently rebuild paired evidence from the persisted Qlib daily report."""

    candidate, baseline = _daily_return_pairs(daily_returns)
    return build_paired_bootstrap_evidence(
        candidate,
        baseline,
        parameters=parameters,
    )


def _canonical_mismatch_paths(
    claimed: Any,
    expected: Any,
    *,
    path: str = "$",
) -> list[str]:
    if isinstance(claimed, Mapping) and isinstance(expected, Mapping):
        mismatches: list[str] = []
        for key in sorted(set(claimed) | set(expected), key=str):
            child = f"{path}.{key}"
            if key not in claimed or key not in expected:
                mismatches.append(child)
            else:
                mismatches.extend(
                    _canonical_mismatch_paths(claimed[key], expected[key], path=child)
                )
        return mismatches
    if isinstance(claimed, list) and isinstance(expected, list):
        mismatches = []
        for index in range(max(len(claimed), len(expected))):
            child = f"{path}[{index}]"
            if index >= len(claimed) or index >= len(expected):
                mismatches.append(child)
            else:
                mismatches.extend(
                    _canonical_mismatch_paths(
                        claimed[index], expected[index], path=child
                    )
                )
        return mismatches
    try:
        equal = canonical_sha256(claimed) == canonical_sha256(expected)
    except (TypeError, ValueError):
        equal = False
    return [] if equal else [path]


def validate_paired_bootstrap_evidence_schema(
    value: Any,
    *,
    expected_parameters: Mapping[str, Any],
    expected_observations: int | None = None,
) -> dict[str, Any]:
    """Fail closed on missing, surplus, malformed, or re-parameterized evidence."""

    if not isinstance(value, Mapping):
        raise ValueError("paired bootstrap evidence must be an object")
    keys = set(value)
    if keys != PAIRED_BOOTSTRAP_EVIDENCE_KEYS:
        missing = sorted(PAIRED_BOOTSTRAP_EVIDENCE_KEYS - keys)
        extra = sorted(keys - PAIRED_BOOTSTRAP_EVIDENCE_KEYS)
        raise ValueError(
            "paired bootstrap evidence has invalid fields "
            f"(missing={missing}, extra={extra})"
        )
    frozen = validate_paired_bootstrap_parameters(expected_parameters)
    observed_parameters = validate_paired_bootstrap_parameters(
        {key: value[key] for key in PAIRED_BOOTSTRAP_PARAMETER_KEYS},
        observations=expected_observations,
    )
    if observed_parameters != frozen:
        raise ValueError("paired bootstrap evidence parameters differ from frozen config")
    observations = _require_plain_int(
        value["observations"],
        label="paired bootstrap observations",
        minimum=30,
    )
    if expected_observations is not None and observations != int(expected_observations):
        raise ValueError("paired bootstrap observations differ from persisted returns")
    if observed_parameters["block_size"] > observations:
        raise ValueError("paired bootstrap block_size exceeds observations")
    interval = value["confidence_interval_95"]
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("paired bootstrap confidence interval must contain two values")
    numeric_fields = {
        "observed_mean_difference": value["observed_mean_difference"],
        "confidence_interval_95[0]": interval[0],
        "confidence_interval_95[1]": interval[1],
        "probability_positive": value["probability_positive"],
        "one_sided_p_value": value["one_sided_p_value"],
    }
    normalized: dict[str, float] = {}
    for label, raw in numeric_fields.items():
        if isinstance(raw, bool):
            raise ValueError(f"paired bootstrap {label} must be finite")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"paired bootstrap {label} must be finite") from exc
        if not np.isfinite(number):
            raise ValueError(f"paired bootstrap {label} must be finite")
        normalized[label] = number
    if normalized["confidence_interval_95[0]"] > normalized[
        "confidence_interval_95[1]"
    ]:
        raise ValueError("paired bootstrap confidence interval is reversed")
    for label in ("probability_positive", "one_sided_p_value"):
        if not 0.0 <= normalized[label] <= 1.0:
            raise ValueError(f"paired bootstrap {label} must be in [0, 1]")
    if (
        value["status"] != "ok"
        or value["contract_version"]
        != PAIRED_BOOTSTRAP_EVIDENCE_CONTRACT_VERSION
        or value["statistical_contract_version"] != STATISTICAL_CONTRACT_VERSION
        or value["method"] != PAIRED_BOOTSTRAP_METHOD
        or value["estimand"] != PAIRED_BOOTSTRAP_ESTIMAND
        or value["input_alignment"] != PAIRED_BOOTSTRAP_INPUT_ALIGNMENT
        or value["confidence_level"] != PAIRED_BOOTSTRAP_CONFIDENCE_LEVEL
    ):
        raise ValueError("paired bootstrap evidence contract is not canonical")
    _require_sha256(value["input_sha256"], label="paired bootstrap input_sha256")
    evidence_sha256 = _require_sha256(
        value["evidence_sha256"],
        label="paired bootstrap evidence_sha256",
    )
    payload = dict(value)
    payload.pop("evidence_sha256")
    if evidence_sha256 != canonical_sha256(payload):
        raise ValueError("paired bootstrap evidence SHA256 is invalid")
    return dict(value)


def validate_paired_bootstrap_evidence(
    value: Any,
    *,
    parameters: Mapping[str, Any],
    candidate_net_returns: pd.Series | Sequence[float] | None = None,
    baseline_returns: pd.Series | Sequence[float] | None = None,
    daily_returns: Any | None = None,
) -> dict[str, Any]:
    """Independently recompute and canonically compare every evidence field.

    Callers must provide either the persisted daily-return table or both aligned
    return series. The claimed payload never supplies recomputation parameters.
    """

    using_daily = daily_returns is not None
    using_series = candidate_net_returns is not None or baseline_returns is not None
    if using_daily == using_series:
        raise ValueError(
            "provide either daily_returns or both paired return series, but not both"
        )
    if using_daily:
        expected = build_paired_bootstrap_evidence_from_daily_returns(
            daily_returns,
            parameters=parameters,
        )
    else:
        if candidate_net_returns is None or baseline_returns is None:
            raise ValueError("both paired return series are required")
        expected = build_paired_bootstrap_evidence(
            candidate_net_returns,
            baseline_returns,
            parameters=parameters,
        )
    validate_paired_bootstrap_evidence_schema(
        value,
        expected_parameters=parameters,
        expected_observations=expected["observations"],
    )
    mismatches = _canonical_mismatch_paths(dict(value), expected)
    if mismatches:
        displayed = ", ".join(mismatches[:8])
        suffix = " ..." if len(mismatches) > 8 else ""
        raise ValueError(
            "paired bootstrap evidence differs from independent recomputation at "
            f"{displayed}{suffix}"
        )
    return expected


def build_factor_score_incomplete_family_multiple_testing(
    *,
    paired_bootstrap: Mapping[str, Any],
    trial_count: int,
    trial_count_audit_sha256: str,
    eligibility_receipt_sha256: str,
) -> dict[str, Any]:
    """Build the conservative factor-family alternative when old returns are absent.

    Bonferroni controls family-wise error under arbitrary dependence without
    inventing the unavailable historical return matrix. PBO remains explicitly
    not computable and the physical trial count is never reduced.
    """

    if isinstance(trial_count, bool) or int(trial_count) <= 1:
        raise ValueError("incomplete-family Bonferroni requires multiple trials")
    count = int(trial_count)
    audit_sha256 = _require_sha256(
        trial_count_audit_sha256,
        label="trial_count_audit_sha256",
    )
    eligibility_sha256 = _require_sha256(
        eligibility_receipt_sha256,
        label="eligibility_receipt_sha256",
    )
    if paired_bootstrap.get("status") != "ok":
        raise ValueError("paired bootstrap must be complete before Bonferroni")
    try:
        raw_p_value = float(paired_bootstrap["one_sided_p_value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("paired bootstrap one-sided p-value is required") from exc
    if not np.isfinite(raw_p_value) or not 0.0 <= raw_p_value <= 1.0:
        raise ValueError("paired bootstrap one-sided p-value must be in [0, 1]")
    adjusted_p_value = min(1.0, raw_p_value * count)
    payload = {
        "status": CONSERVATIVE_BONFERRONI_INCOMPLETE_FAMILY_STATUS,
        "contract_version": FACTOR_SCORE_INCOMPLETE_FAMILY_MULTIPLE_TESTING_VERSION,
        "method": "bonferroni",
        "dependence_assumption": "valid_under_arbitrary_trial_dependence",
        "evidence_scope": "current_frozen_factor_strategy_against_declared_trial_family",
        "trial_count": count,
        "available_candidate_return_series": 1,
        "trial_count_audit_sha256": audit_sha256,
        "eligibility_receipt_sha256": eligibility_sha256,
        "p_value_source": "paired_moving_block_bootstrap",
        "raw_one_sided_p_value": raw_p_value,
        "bonferroni_adjusted_p_value": adjusted_p_value,
        "familywise_alpha": FACTOR_SCORE_FAMILYWISE_ALPHA,
        "gate_passed": adjusted_p_value <= FACTOR_SCORE_FAMILYWISE_ALPHA,
        "pbo": {
            "status": NOT_COMPUTABLE_INCOMPLETE_FAMILY_STATUS,
            "pbo": None,
            "required_trial_return_series": count,
            "available_trial_return_series": 1,
            "reason": "complete aligned historical candidate return matrix is unavailable",
        },
    }
    return {**payload, "evidence_sha256": canonical_sha256(payload)}


def validate_factor_score_incomplete_family_multiple_testing(
    value: Any,
    *,
    paired_bootstrap: Mapping[str, Any],
    trial_count: int,
    trial_count_audit_sha256: str,
    eligibility_receipt_sha256: str,
) -> dict[str, Any]:
    """Rebuild and exactly validate conservative incomplete-family evidence."""

    if not isinstance(value, Mapping):
        raise ValueError("incomplete-family multiple-testing evidence must be an object")
    expected = build_factor_score_incomplete_family_multiple_testing(
        paired_bootstrap=paired_bootstrap,
        trial_count=trial_count,
        trial_count_audit_sha256=trial_count_audit_sha256,
        eligibility_receipt_sha256=eligibility_receipt_sha256,
    )
    if dict(value) != expected:
        raise ValueError("incomplete-family Bonferroni evidence is not canonical")
    return expected


def build_factor_score_incomplete_family_dsr(
    *,
    blocked_dsr: Mapping[str, Any],
    trial_count: int,
    trial_count_audit_sha256: str,
    eligibility_receipt_sha256: str,
) -> dict[str, Any]:
    """Record unavailable DSR inputs without converting them into a pass."""

    if isinstance(trial_count, bool) or int(trial_count) <= 1:
        raise ValueError("incomplete-family DSR requires multiple trials")
    count = int(trial_count)
    audit_sha256 = _require_sha256(
        trial_count_audit_sha256,
        label="trial_count_audit_sha256",
    )
    eligibility_sha256 = _require_sha256(
        eligibility_receipt_sha256,
        label="eligibility_receipt_sha256",
    )
    if (
        blocked_dsr.get("status") != "blocked_missing_trial_sharpe_distribution"
        or blocked_dsr.get("probability") is not None
        or int(blocked_dsr.get("trials") or 0) != count
        or blocked_dsr.get("method_version") != DEFLATED_SHARPE_METHOD_VERSION
        or blocked_dsr.get("contract_version") != STATISTICAL_CONTRACT_VERSION
        or blocked_dsr.get("expected_maximum_daily_sharpe") is not None
        or blocked_dsr.get("trial_sharpe_std") is not None
    ):
        raise ValueError("DSR must be blocked by the missing complete trial distribution")
    payload = {
        **dict(blocked_dsr),
        "status": NOT_COMPUTABLE_INCOMPLETE_FAMILY_STATUS,
        "source_status": "blocked_missing_trial_sharpe_distribution",
        "incomplete_family_contract_version": (
            FACTOR_SCORE_INCOMPLETE_FAMILY_DSR_VERSION
        ),
        "trial_count_audit_sha256": audit_sha256,
        "eligibility_receipt_sha256": eligibility_sha256,
        "trial_sharpes_available": False,
        "reason": "complete historical trial Sharpe distribution is unavailable",
    }
    return {**payload, "evidence_sha256": canonical_sha256(payload)}


def validate_factor_score_incomplete_family_dsr(
    value: Any,
    *,
    trial_count: int,
    trial_count_audit_sha256: str,
    eligibility_receipt_sha256: str,
) -> dict[str, Any]:
    """Validate an explicitly not-computable DSR record and its audit binding."""

    if not isinstance(value, Mapping):
        raise ValueError("incomplete-family DSR evidence must be an object")
    count = int(trial_count)
    audit_sha256 = _require_sha256(
        trial_count_audit_sha256,
        label="trial_count_audit_sha256",
    )
    eligibility_sha256 = _require_sha256(
        eligibility_receipt_sha256,
        label="eligibility_receipt_sha256",
    )
    payload = dict(value)
    evidence_sha256 = payload.pop("evidence_sha256", None)
    if evidence_sha256 != canonical_sha256(payload):
        raise ValueError("incomplete-family DSR evidence SHA256 is invalid")
    if (
        payload.get("status") != NOT_COMPUTABLE_INCOMPLETE_FAMILY_STATUS
        or payload.get("source_status")
        != "blocked_missing_trial_sharpe_distribution"
        or payload.get("incomplete_family_contract_version")
        != FACTOR_SCORE_INCOMPLETE_FAMILY_DSR_VERSION
        or payload.get("probability") is not None
        or int(payload.get("trials") or 0) != count
        or payload.get("method_version") != DEFLATED_SHARPE_METHOD_VERSION
        or payload.get("contract_version") != STATISTICAL_CONTRACT_VERSION
        or payload.get("expected_maximum_daily_sharpe") is not None
        or payload.get("trial_sharpe_std") is not None
        or payload.get("trial_count_audit_sha256") != audit_sha256
        or payload.get("eligibility_receipt_sha256") != eligibility_sha256
        or payload.get("trial_sharpes_available") is not False
        or payload.get("reason")
        != "complete historical trial Sharpe distribution is unavailable"
    ):
        raise ValueError("incomplete-family DSR evidence is not canonical")
    return dict(value)


@dataclass(frozen=True)
class OuterFold:
    fold: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


def build_pre_final_history_evidence(
    dates: pd.DatetimeIndex | Sequence[Any],
    *,
    requested_start: str,
    requested_end: str,
    final_test_start: str,
    final_test_end: str,
    minimum_trading_days: int,
    minimum_embargo_trading_days: int,
) -> dict[str, Any]:
    """Prove that long-history evidence is isolated from the final test.

    This is deliberately a calendar contract, not a performance claim.  It
    prevents a short final-test report from being reused as supposed
    long-history walk-forward evidence and records the exact trading-day
    coverage consumed by the pre-final stability suite.
    """

    history_start = pd.Timestamp(requested_start).normalize()
    history_end = pd.Timestamp(requested_end).normalize()
    test_start = pd.Timestamp(final_test_start).normalize()
    test_end = pd.Timestamp(final_test_end).normalize()
    if history_end < history_start:
        raise ValueError("pre-final history end must not be before its start")
    if test_end < test_start:
        raise ValueError("final test end must not be before its start")
    if history_end >= test_start:
        raise ValueError("pre-final history must end before the final test starts")
    if minimum_trading_days < 252:
        raise ValueError("pre-final history minimum must be at least 252 trading days")
    if minimum_embargo_trading_days < 1:
        raise ValueError("final-test embargo must be at least one trading day")

    ordered = pd.DatetimeIndex(pd.to_datetime(dates).unique()).sort_values().tz_localize(None)
    covered = ordered[(ordered >= history_start) & (ordered <= history_end)]
    embargo = ordered[(ordered > history_end) & (ordered < test_start)]
    if covered.empty:
        raise ValueError("pre-final history has no trading calendar coverage")
    if len(covered) < int(minimum_trading_days):
        raise ValueError(
            "pre-final history has "
            f"{len(covered)} trading days; {minimum_trading_days} are required"
        )
    if len(embargo) < int(minimum_embargo_trading_days):
        raise ValueError(
            "pre-final history leaves "
            f"{len(embargo)} embargo trading days; "
            f"{minimum_embargo_trading_days} are required"
        )
    return {
        "status": "completed",
        "contract_version": PRE_FINAL_HISTORY_CONTRACT_VERSION,
        "requested_periods": {
            "start": history_start.date().isoformat(),
            "end": history_end.date().isoformat(),
        },
        "observed_periods": {
            "start": covered[0].date().isoformat(),
            "end": covered[-1].date().isoformat(),
        },
        "final_test_periods": {
            "start": test_start.date().isoformat(),
            "end": test_end.date().isoformat(),
        },
        "trading_days": int(len(covered)),
        "minimum_trading_days": int(minimum_trading_days),
        "embargo_trading_days": int(len(embargo)),
        "minimum_embargo_trading_days": int(minimum_embargo_trading_days),
        "overlaps_final_test": False,
        "uses_final_test_data": False,
    }


def build_outer_walk_forward_folds(
    dates: pd.DatetimeIndex | Sequence[Any],
    *,
    train_days: int,
    validation_days: int,
    test_days: int,
    purge_days: int,
    embargo_days: int,
) -> list[OuterFold]:
    """Build expanding outer folds with isolated validation and test windows."""

    ordered = pd.DatetimeIndex(pd.to_datetime(dates).unique()).sort_values()
    if (
        train_days < 20
        or validation_days < 5
        or test_days < 5
        or purge_days < 0
        or embargo_days < 0
    ):
        raise ValueError("outer walk-forward window lengths are invalid")
    first_test = train_days + purge_days + validation_days + embargo_days
    folds: list[OuterFold] = []
    test_start_index = first_test
    while test_start_index + test_days <= len(ordered):
        validation_end_index = test_start_index - embargo_days
        validation_start_index = validation_end_index - validation_days
        train_end_index = validation_start_index - purge_days
        if train_end_index < train_days:
            break
        test = ordered[test_start_index : test_start_index + test_days]
        validation = ordered[validation_start_index:validation_end_index]
        train = ordered[:train_end_index]
        folds.append(
            OuterFold(
                fold=len(folds),
                train_start=train[0].date().isoformat(),
                train_end=train[-1].date().isoformat(),
                validation_start=validation[0].date().isoformat(),
                validation_end=validation[-1].date().isoformat(),
                test_start=test[0].date().isoformat(),
                test_end=test[-1].date().isoformat(),
            )
        )
        test_start_index += test_days
    if not folds:
        raise ValueError("outer walk-forward windows leave no complete fold")
    return folds


def run_outer_walk_forward(
    *,
    dates: pd.DatetimeIndex | Sequence[Any],
    candidate_ids: Sequence[str],
    inner_runner: Callable[[str, OuterFold], dict[str, Any]],
    test_runner: Callable[[str, OuterFold], dict[str, Any]],
    selection_metric: str,
    train_days: int,
    validation_days: int,
    test_days: int,
    purge_days: int,
    embargo_days: int,
    minimum_test_metric: float = 0.0,
    minimum_test_pass_rate: float = 0.60,
) -> dict[str, Any]:
    """Rerun candidate selection inside every outer fold, then open its test."""

    candidates = [str(item) for item in candidate_ids]
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("outer walk-forward candidates must be non-empty and unique")
    if not np.isfinite(float(minimum_test_metric)):
        raise ValueError("outer walk-forward test floor must be finite")
    if not 0.0 <= float(minimum_test_pass_rate) <= 1.0:
        raise ValueError("outer walk-forward test pass rate must be in [0, 1]")
    folds = build_outer_walk_forward_folds(
        dates,
        train_days=train_days,
        validation_days=validation_days,
        test_days=test_days,
        purge_days=purge_days,
        embargo_days=embargo_days,
    )
    evidence: list[dict[str, Any]] = []
    for fold in folds:
        inner_results: dict[str, dict[str, Any]] = {}
        scored: list[tuple[float, str]] = []
        for candidate in candidates:
            result = dict(inner_runner(candidate, fold))
            score = result.get(selection_metric)
            if score is None or not np.isfinite(float(score)):
                raise ValueError(
                    f"inner selection metric {selection_metric} is missing or non-finite"
                )
            inner_results[candidate] = result
            scored.append((float(score), candidate))
        selected = max(scored, key=lambda item: (item[0], item[1]))[1]
        test_result = dict(test_runner(selected, fold))
        test_score = test_result.get(selection_metric)
        if test_score is None or not np.isfinite(float(test_score)):
            raise ValueError(
                f"outer test metric {selection_metric} is missing or non-finite"
            )
        test_passed = float(test_score) > float(minimum_test_metric)
        evidence.append(
            {
                "fold": fold.__dict__,
                "inner_selection": inner_results,
                "selected_candidate_id": selected,
                "test_result": test_result,
                "test_metric": float(test_score),
                "test_passed": test_passed,
            }
        )
    test_values = np.asarray([item["test_metric"] for item in evidence], dtype=float)
    test_pass_rate = float(np.mean([item["test_passed"] for item in evidence]))
    mean_test_metric = float(test_values.mean())
    passed = (
        test_pass_rate >= float(minimum_test_pass_rate)
        and mean_test_metric > float(minimum_test_metric)
    )
    return {
        "status": "completed",
        "passed": passed,
        "contract_version": FORMAL_VALIDATION_CONTRACT_VERSION,
        "selection_metric": selection_metric,
        "candidate_ids": candidates,
        "fold_count": len(evidence),
        "purge_days": int(purge_days),
        "embargo_days": int(embargo_days),
        "minimum_test_metric": float(minimum_test_metric),
        "minimum_test_pass_rate": float(minimum_test_pass_rate),
        "test_pass_rate": test_pass_rate,
        "mean_test_metric": mean_test_metric,
        "folds": evidence,
    }


def run_ablation_suite(
    *,
    component_ids: Sequence[str],
    full_metrics: dict[str, Any],
    runner: Callable[[str], dict[str, Any]],
    metric: str,
    minimum_increment: float = 0.0,
) -> dict[str, Any]:
    """Measure each component's frozen incremental contribution."""

    components = [str(item) for item in component_ids]
    if not components or len(components) != len(set(components)):
        raise ValueError("ablation components must be non-empty and unique")
    full_value = full_metrics.get(metric)
    if full_value is None or not np.isfinite(float(full_value)):
        raise ValueError(f"full strategy metric {metric} is missing or non-finite")
    runs: list[dict[str, Any]] = []
    for component in components:
        metrics = dict(runner(component))
        ablated = metrics.get(metric)
        if ablated is None or not np.isfinite(float(ablated)):
            raise ValueError(f"ablation metric {metric} is missing or non-finite")
        increment = float(full_value) - float(ablated)
        runs.append(
            {
                "removed_component_id": component,
                "metrics": metrics,
                "increment": increment,
                "passed": increment >= minimum_increment,
            }
        )
    return {
        "status": "passed" if all(item["passed"] for item in runs) else "failed",
        "contract_version": FORMAL_VALIDATION_CONTRACT_VERSION,
        "metric": metric,
        "full_value": float(full_value),
        "minimum_increment": float(minimum_increment),
        "runs": runs,
    }


def run_signal_decay_suite(
    *,
    delays: Sequence[int],
    runner: Callable[[int], dict[str, Any]],
    metric: str,
    minimum_retention: float,
) -> dict[str, Any]:
    """Rerun delayed execution and derive the last supported signal delay."""

    normalized = sorted({int(item) for item in delays})
    if not normalized or normalized[0] != 0 or any(item < 0 for item in normalized):
        raise ValueError("signal decay delays must be unique non-negative values including zero")
    if not 0 <= minimum_retention <= 1:
        raise ValueError("minimum signal retention must be in [0, 1]")
    runs: list[dict[str, Any]] = []
    base: float | None = None
    for delay in normalized:
        metrics = dict(runner(delay))
        value = metrics.get(metric)
        if value is None or not np.isfinite(float(value)):
            raise ValueError(f"signal decay metric {metric} is missing or non-finite")
        value = float(value)
        if base is None:
            base = value
            if base <= 0:
                raise ValueError("zero-delay signal metric must be positive")
        retention = value / base
        runs.append(
            {
                "delay_bars": delay,
                "metrics": metrics,
                "retention": retention,
                "passed": retention >= minimum_retention and value > 0,
            }
        )
    contiguous: list[int] = []
    for item in runs:
        if not item["passed"]:
            break
        contiguous.append(int(item["delay_bars"]))
    return {
        "status": "completed",
        "contract_version": FORMAL_VALIDATION_CONTRACT_VERSION,
        "metric": metric,
        "minimum_retention": float(minimum_retention),
        "frontier_version": SIGNAL_DECAY_FRONTIER_VERSION,
        # Execution tolerance is a contiguous frontier from zero delay.  A
        # noisy later pass after an earlier failure does not prove that the
        # strategy can tolerate the skipped delay.
        "maximum_supported_delay_bars": max(contiguous) if contiguous else None,
        "runs": runs,
    }
