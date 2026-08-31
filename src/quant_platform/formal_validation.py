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


def _require_sha256(value: Any, *, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be SHA256")
    return normalized


def build_factor_score_incomplete_family_multiple_testing(
    *,
    paired_bootstrap: Mapping[str, Any],
    trial_count: int,
    trial_count_audit_sha256: str,
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
) -> dict[str, Any]:
    """Rebuild and exactly validate conservative incomplete-family evidence."""

    if not isinstance(value, Mapping):
        raise ValueError("incomplete-family multiple-testing evidence must be an object")
    expected = build_factor_score_incomplete_family_multiple_testing(
        paired_bootstrap=paired_bootstrap,
        trial_count=trial_count,
        trial_count_audit_sha256=trial_count_audit_sha256,
    )
    if dict(value) != expected:
        raise ValueError("incomplete-family Bonferroni evidence is not canonical")
    return expected


def build_factor_score_incomplete_family_dsr(
    *,
    blocked_dsr: Mapping[str, Any],
    trial_count: int,
    trial_count_audit_sha256: str,
) -> dict[str, Any]:
    """Record unavailable DSR inputs without converting them into a pass."""

    if isinstance(trial_count, bool) or int(trial_count) <= 1:
        raise ValueError("incomplete-family DSR requires multiple trials")
    count = int(trial_count)
    audit_sha256 = _require_sha256(
        trial_count_audit_sha256,
        label="trial_count_audit_sha256",
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
        "trial_sharpes_available": False,
        "reason": "complete historical trial Sharpe distribution is unavailable",
    }
    return {**payload, "evidence_sha256": canonical_sha256(payload)}


def validate_factor_score_incomplete_family_dsr(
    value: Any,
    *,
    trial_count: int,
    trial_count_audit_sha256: str,
) -> dict[str, Any]:
    """Validate an explicitly not-computable DSR record and its audit binding."""

    if not isinstance(value, Mapping):
        raise ValueError("incomplete-family DSR evidence must be an object")
    count = int(trial_count)
    audit_sha256 = _require_sha256(
        trial_count_audit_sha256,
        label="trial_count_audit_sha256",
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
