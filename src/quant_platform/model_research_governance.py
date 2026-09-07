from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .horizon_factor_bundle import validate_horizon_factor_bundle
from .research_execution_cadence import (
    validate_research_execution_cadence_contract,
)
from .research_horizon import (
    LEGACY_AMBIGUOUS,
    primary_label_horizon_sessions,
    require_label_horizon,
    research_horizon_contract,
)
from .statistical_validation import (
    holm_bonferroni,
    paired_moving_block_bootstrap,
    probability_of_backtest_overfitting,
)

MODEL_RESEARCH_CONTRACT_VERSION = "model-research-independent-v1"
MODEL_PREDICTION_CONTRACT_VERSION = "model-predictions-exact-oos-v1"
QUANT_BUNDLE_CONTRACT_VERSION = "quant-bundle-ablation-v2"
LEGACY_QUANT_BUNDLE_CONTRACT_VERSION = "quant-bundle-ablation-v1"
QUANT_MULTIPLE_TESTING_CONTRACT_VERSION = "quant-pre-final-multiple-testing-v1"
RUN_GROUP_MULTIPLE_TESTING_CONTRACT_VERSION = "rdagent-run-multiple-testing-v1"
REQUIRED_MODEL_SEEDS = (11, 29, 47)
REQUIRED_RESEARCH_PROFILES = ("recent_3y", "balanced_5y", "robust_10y")
REQUIRED_QUANT_ABLATIONS = ("factor_only", "model_only", "joint")
PRIMARY_MODEL_PROFILE = "recent_3y"
PRIMARY_MODEL_SEED = 11
MODEL_LABEL_HORIZON_TRADING_DAYS = 2
LEGACY_MODEL_PREDICTION_HORIZON_SESSIONS = 1
MODEL_LABEL_CONTRACT_VERSION = "model-label-contract-v1"
MODEL_REFIT_POLICY_VERSION = "rolling-primary-model-profile-v2"
# The governed 3031-day research contract leaves 2016 training days for the
# primary 3-year validation profile (plus 5 embargo and 252 sealed final-OOS
# days and a 2-session train/validation label purge). Live inference rolls the
# same geometry and predicts one new signal day; it does not invent a shorter
# training regime.
MODEL_REFIT_POLICY = {
    "contract_version": MODEL_REFIT_POLICY_VERSION,
    "profile_id": PRIMARY_MODEL_PROFILE,
    "seed": PRIMARY_MODEL_SEED,
    "train_trading_days": 2016,
    "train_validation_purge_trading_days": MODEL_LABEL_HORIZON_TRADING_DAYS,
    "validation_trading_days": 756,
    "embargo_trading_days": 5,
    "prediction_trading_days": 1,
}
REQUIRED_MODEL_METRICS = (
    "ic",
    "icir",
    "rank_ic",
    "rank_icir",
    "information_ratio",
    "annualized_excess_return_with_cost",
    "max_drawdown",
    "total_cost",
    "average_turnover",
)
MODEL_METRIC_GATE_VERSION = "model-metric-gate-v1"
MODEL_METRIC_GATE = {
    "minimum_ic": 0.02,
    "minimum_icir": 0.50,
    "minimum_rank_ic": 0.025,
    "minimum_rank_icir": 0.50,
    "minimum_information_ratio": 0.0,
    "minimum_annualized_excess_return_with_cost": 0.0,
    "maximum_absolute_drawdown": 0.35,
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


MODEL_REFIT_POLICY_SHA256 = canonical_sha256(MODEL_REFIT_POLICY)


def resolve_model_label_contract(
    *,
    research_window_contract: Mapping[str, Any] | None = None,
    research_window_contract_sha256: str | None = None,
    label_horizon_sessions: int | None = None,
) -> dict[str, Any]:
    """Resolve one executable forward-return label from a research window.

    Product horizons describe a set of labels.  One model run must select one
    member explicitly; absent a choice, active horizons use the governed
    primary comparison label (5/63/252 sessions).  Old runs retain the
    historical next-session-to-following-session return and are marked legacy
    rather than reinterpreted.
    """

    if research_window_contract is None:
        if label_horizon_sessions not in {None, LEGACY_MODEL_PREDICTION_HORIZON_SESSIONS}:
            raise ValueError("legacy model research supports only its one-session label")
        return {
            "contract_version": MODEL_LABEL_CONTRACT_VERSION,
            "horizon_profile": LEGACY_AMBIGUOUS,
            "legacy": True,
            "allowed_label_horizons_sessions": [
                LEGACY_MODEL_PREDICTION_HORIZON_SESSIONS
            ],
            "label_horizon_sessions": LEGACY_MODEL_PREDICTION_HORIZON_SESSIONS,
            "label_reference_offset_sessions": MODEL_LABEL_HORIZON_TRADING_DAYS,
            "label_expression": "Ref($close,-2)/Ref($close,-1)-1",
            "purge_sessions": MODEL_LABEL_HORIZON_TRADING_DAYS,
            "embargo_sessions": 5,
            "research_window_contract_sha256": None,
        }
    if not isinstance(research_window_contract, Mapping):
        raise ValueError("research window contract must be an object")
    expected_window_sha256 = str(research_window_contract_sha256 or "").lower()
    if (
        len(expected_window_sha256) != 64
        or canonical_sha256(dict(research_window_contract)) != expected_window_sha256
    ):
        raise ValueError("research window contract digest is missing or invalid")
    profile = str(research_window_contract.get("horizon_profile") or "")
    if research_window_contract.get("contract_version") != "research-window-v1":
        raise ValueError("research window contract version is invalid")
    if profile == LEGACY_AMBIGUOUS:
        legacy = resolve_model_label_contract(label_horizon_sessions=label_horizon_sessions)
        return {
            **legacy,
            "research_window_contract_sha256": expected_window_sha256,
        }
    horizon = research_horizon_contract(profile)
    if research_window_contract.get("horizon_contract_sha256") != horizon.sha256:
        raise ValueError("research window horizon digest is invalid")
    raw_labels = research_window_contract.get("label_horizons_sessions") or []
    try:
        allowed = tuple(int(item) for item in raw_labels)
    except (TypeError, ValueError) as exc:
        raise ValueError("research window label horizons are invalid") from exc
    if allowed != horizon.label_horizons_sessions:
        raise ValueError("research window labels differ from the horizon contract")
    selected = (
        int(label_horizon_sessions)
        if label_horizon_sessions is not None
        else primary_label_horizon_sessions(profile)
    )
    require_label_horizon(profile, selected)
    purge = int(research_window_contract.get("purge_sessions") or 0)
    embargo = int(research_window_contract.get("embargo_sessions") or 0)
    if purge < int(horizon.purge_sessions or 0):
        raise ValueError("research window model purge is weaker than the horizon contract")
    if embargo < int(horizon.embargo_sessions or 0):
        raise ValueError("research window model embargo is weaker than the horizon contract")
    if research_window_contract.get("label_maturity_enforced") is not True:
        raise ValueError("active model research requires enforced label maturity")
    return {
        "contract_version": MODEL_LABEL_CONTRACT_VERSION,
        "horizon_profile": profile,
        "legacy": False,
        "allowed_label_horizons_sessions": list(allowed),
        "label_horizon_sessions": selected,
        "label_reference_offset_sessions": selected + 1,
        "label_expression": f"Ref($close,-{selected + 1})/Ref($close,-1)-1",
        "purge_sessions": purge,
        "embargo_sessions": embargo,
        "research_window_contract_sha256": expected_window_sha256,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_multiple_testing_evidence(
    *,
    research_run_id: str,
    trial_series: Sequence[tuple[Mapping[str, str], pd.Series]],
    output: Path,
    seeds: Sequence[int] = REQUIRED_MODEL_SEEDS,
    seed_aggregation: str = "equal_mean_fixed_seeds",
    forced_raw_p_values: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build one immutable Holm/PBO family for every pre-registered run trial."""

    if not research_run_id or not trial_series:
        raise ValueError("run-level multiple testing requires a research run and trials")
    definitions: list[dict[str, str]] = []
    columns: list[pd.Series] = []
    names: set[str] = set()
    for raw_definition, raw_returns in trial_series:
        definition = {str(key): str(value) for key, value in raw_definition.items()}
        name = definition.get("name", "")
        candidate_id = definition.get("candidate_id", "")
        if not name or not candidate_id or name in names:
            raise ValueError("run-level multiple-testing trial identities are invalid")
        series = pd.to_numeric(raw_returns, errors="coerce").rename(name)
        series.index = pd.to_datetime(series.index, errors="coerce").tz_localize(None)
        if series.index.isna().any() or series.index.has_duplicates:
            raise ValueError("run-level multiple-testing returns have an invalid index")
        definitions.append(definition)
        columns.append(series.sort_index())
        names.add(name)
    trials = pd.concat(columns, axis=1, join="inner").dropna()
    if list(trials.columns) != [item["name"] for item in definitions] or len(trials) < 40:
        raise ValueError("run-level multiple-testing return matrix is incomplete")
    values = trials.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("run-level multiple-testing returns are not finite")
    output.mkdir(parents=True, exist_ok=True)
    returns_path = output / "run_multiple_testing_returns.parquet"
    trials.to_parquet(returns_path, compression="zstd")
    forced = {
        str(name): float(value)
        for name, value in dict(forced_raw_p_values or {}).items()
    }
    if any(
        name not in trials.columns or not math.isfinite(value) or not 0.0 <= value <= 1.0
        for name, value in forced.items()
    ):
        raise ValueError("forced run-level p-values are invalid")
    raw_p_values = [
        forced[str(name)]
        if str(name) in forced
        else paired_moving_block_bootstrap(
                trials[name],
                pd.Series(0.0, index=trials.index),
                block_size=min(20, len(trials)),
                samples=2000,
                seed=0,
            )["one_sided_p_value"]
        for name in trials.columns
    ]
    adjusted = holm_bonferroni(raw_p_values)
    # Failed preregistered candidates remain members of the frozen family.  A
    # zero placeholder is never eligible (its forced Holm p-value is 1), but
    # keeping the column in PBO prevents a runtime failure from silently
    # shrinking the comparison family.
    pbo_trial_names = [str(name) for name in trials.columns]
    pbo_trials = trials[pbo_trial_names]
    if len(pbo_trial_names) <= 1:
        pbo = {
            "status": "not_applicable_single_trial",
            "pbo": None,
            "trials": len(pbo_trial_names),
            "observations": len(trials),
        }
        pbo_passed = True
    else:
        pbo = probability_of_backtest_overfitting(pbo_trials, blocks=8)
        pbo_passed = (
            pbo.get("status") == "ok"
            and pbo.get("pbo") is not None
            and float(pbo["pbo"]) <= 0.50
        )
    eligible = [
        definition["name"]
        for definition, adjusted_value in zip(definitions, adjusted, strict=True)
        if float(adjusted_value) <= 0.05 and pbo_passed
    ]
    sharpes = []
    for name in trials.columns:
        standard_deviation = float(trials[name].std(ddof=1))
        sharpes.append(
            0.0
            if name in forced or not math.isfinite(standard_deviation) or standard_deviation == 0.0
            else float(trials[name].mean() / standard_deviation)
        )
    if not all(math.isfinite(value) for value in sharpes):
        raise ValueError("run-level multiple-testing trial Sharpe values are invalid")
    evidence = {
        "contract_version": RUN_GROUP_MULTIPLE_TESTING_CONTRACT_VERSION,
        "source": "independent_qlib_recompute",
        "research_run_id": research_run_id,
        "profile_id": PRIMARY_MODEL_PROFILE,
        "seed_aggregation": str(seed_aggregation),
        "seeds": [int(seed) for seed in seeds],
        "trial_definitions": definitions,
        "trial_names": [item["name"] for item in definitions],
        "trial_count": len(definitions),
        "final_oos_opened": False,
        "raw_p_values": raw_p_values,
        "forced_raw_p_values": forced,
        "holm_adjusted_p_values": adjusted,
        "maximum_adjusted_p_value": 0.05,
        "eligible_trial_names": eligible,
        "pbo": pbo,
        "pbo_trial_names": pbo_trial_names,
        "maximum_pbo": 0.50,
        "trial_daily_sharpes": sharpes,
        "trial_daily_means": [float(trials[name].mean()) for name in trials.columns],
        "returns_path": str(returns_path),
        "returns_sha256": file_sha256(returns_path),
        "observations": len(trials),
        "gate_passed": bool(eligible),
        "statistical_evidence_role": "report_only",
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def validate_run_multiple_testing_evidence(
    evidence: Any,
    *,
    selected_trial_name: str,
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise ValueError("run-level multiple-testing evidence is missing")
    try:
        definitions = [dict(item) for item in evidence["trial_definitions"]]
        trial_names = [str(value) for value in evidence["trial_names"]]
        raw = [float(value) for value in evidence["raw_p_values"]]
        adjusted = [float(value) for value in evidence["holm_adjusted_p_values"]]
        sharpes = [float(value) for value in evidence["trial_daily_sharpes"]]
        means = [float(value) for value in evidence.get("trial_daily_means") or []]
        eligible = [str(value) for value in evidence["eligible_trial_names"]]
        pbo_trial_names = [
            str(value) for value in evidence.get("pbo_trial_names", trial_names)
        ]
        forced = {
            str(name): float(value)
            for name, value in dict(evidence.get("forced_raw_p_values") or {}).items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("run-level multiple-testing values are malformed") from exc
    count = len(trial_names)
    pbo = evidence.get("pbo")
    pbo_valid = (
        isinstance(pbo, Mapping)
        and (
            (
                len(pbo_trial_names) == 1
                and pbo.get("status") == "not_applicable_single_trial"
                and pbo.get("pbo") is None
            )
            or (
                len(pbo_trial_names) > 1
                and pbo.get("status") == "ok"
                and pbo.get("pbo") is not None
                and 0.0 <= float(pbo["pbo"]) <= 1.0
            )
        )
    )
    if (
        evidence.get("contract_version") != RUN_GROUP_MULTIPLE_TESTING_CONTRACT_VERSION
        or evidence.get("source") != "independent_qlib_recompute"
        or not str(evidence.get("research_run_id") or "")
        or evidence.get("profile_id") != PRIMARY_MODEL_PROFILE
        or evidence.get("seed_aggregation") != "equal_mean_fixed_seeds"
        or evidence.get("seeds") != list(REQUIRED_MODEL_SEEDS)
        or evidence.get("trial_count") != count
        or count < 1
        or len(set(trial_names)) != count
        or pbo_trial_names != trial_names
        or any(
            name not in trial_names
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            or raw[trial_names.index(name)] != value
            for name, value in forced.items()
        )
        or [str(item.get("name") or "") for item in definitions] != trial_names
        or any(not str(item.get("candidate_id") or "") for item in definitions)
        or evidence.get("final_oos_opened") is not False
        or len(raw) != count
        or len(adjusted) != count
        or len(sharpes) != count
        or (means and len(means) != count)
        or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in raw + adjusted)
        or not all(math.isfinite(value) for value in sharpes)
        or not all(math.isfinite(value) for value in means)
        or len(set(eligible)) != len(eligible)
        or any(value not in trial_names for value in eligible)
        or selected_trial_name not in trial_names
        or selected_trial_name in forced
        or evidence.get("statistical_evidence_role") not in {None, "report_only"}
        or int(evidence.get("observations") or 0) < 40
        or not pbo_valid
        or not is_sha256(evidence.get("returns_sha256"))
        or not is_sha256(evidence.get("evidence_sha256"))
    ):
        raise ValueError("run-level Holm/PBO gate is invalid or selected trial did not pass")
    if evidence.get("statistical_evidence_role") == "report_only":
        pbo_passed = len(pbo_trial_names) == 1 or float(pbo["pbo"]) <= 0.50
        expected_eligible = [
            name for name, value in zip(trial_names, adjusted, strict=True)
            if value <= 0.05 and pbo_passed
        ]
        if (
            evidence.get("maximum_adjusted_p_value") != 0.05
            or evidence.get("maximum_pbo") != 0.50
            or adjusted != holm_bonferroni(raw)
            or eligible != expected_eligible
            or evidence.get("gate_passed") is not bool(expected_eligible)
            or len(means) != count
        ):
            raise ValueError("run-level statistical report disagrees with its recorded values")
    elif selected_trial_name not in eligible or evidence.get("gate_passed") is not True or (
        len(pbo_trial_names) > 1
        and float(pbo["pbo"]) > float(evidence.get("maximum_pbo", -1.0))
    ):
        raise ValueError("legacy run-level Holm/PBO gate did not pass")
    expected = canonical_sha256(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )
    if evidence.get("evidence_sha256") != expected:
        raise ValueError("run-level multiple-testing evidence SHA-256 is invalid")
    return dict(evidence)


def is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def normalize_model_predictions(value: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        value = value.to_frame(name=value.name or "score")
    if not isinstance(value, pd.DataFrame) or value.shape[1] != 1:
        raise ValueError("model predictions must contain exactly one score column")
    if not isinstance(value.index, pd.MultiIndex) or set(value.index.names) != {
        "datetime",
        "instrument",
    }:
        raise ValueError("model predictions must use a datetime/instrument MultiIndex")
    result = value.copy()
    if result.index.names != ["datetime", "instrument"]:
        result = result.reorder_levels(["datetime", "instrument"])
    dates = pd.to_datetime(result.index.get_level_values("datetime"), errors="coerce")
    if dates.isna().any():
        raise ValueError("model predictions contain invalid dates")
    result.index = pd.MultiIndex.from_arrays(
        [dates.tz_localize(None), result.index.get_level_values("instrument").astype(str)],
        names=["datetime", "instrument"],
    )
    result.columns = ["score"]
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    result = result.sort_index()
    if result.index.has_duplicates:
        raise ValueError("model predictions contain duplicate rows")
    return result


def verify_model_prediction_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    test_start: str,
    test_end: str,
    trading_days: Sequence[Any],
    min_daily_finite: int = 50,
    min_coverage_ratio: float = 0.80,
    min_good_day_rate: float = 0.95,
) -> dict[str, Any]:
    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise ValueError("model prediction artifact does not exist")
    if not is_sha256(expected_sha256) or file_sha256(artifact) != expected_sha256.lower():
        raise ValueError("model prediction artifact SHA-256 is invalid")
    if artifact.suffix.lower() == ".parquet":
        predictions = normalize_model_predictions(pd.read_parquet(artifact))
    elif artifact.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        predictions = normalize_model_predictions(pd.read_hdf(artifact))
    else:
        raise ValueError("model predictions must be stored as parquet or HDF5")

    start = pd.Timestamp(test_start).tz_localize(None).normalize()
    end = pd.Timestamp(test_end).tz_localize(None).normalize()
    expected_days = pd.DatetimeIndex(pd.to_datetime(list(trading_days), errors="coerce"))
    if expected_days.isna().any():
        raise ValueError("authoritative model evaluation calendar contains invalid dates")
    expected_days = expected_days.tz_localize(None).normalize().unique().sort_values()
    if expected_days.empty or expected_days[0] != start or expected_days[-1] != end:
        raise ValueError("model evaluation calendar does not match the frozen OOS window")
    dates = pd.DatetimeIndex(predictions.index.get_level_values("datetime")).normalize()
    all_days = dates.unique().sort_values()
    if not all_days.equals(expected_days):
        raise ValueError("model prediction artifact contains dates outside the exact OOS window")
    oos = predictions.loc[(dates >= start) & (dates <= end)]
    oos_dates = pd.DatetimeIndex(oos.index.get_level_values("datetime")).normalize()
    actual_days = oos_dates.unique().sort_values()
    if not actual_days.equals(expected_days):
        raise ValueError("model predictions do not exactly cover every OOS trading day")

    finite = np.isfinite(oos["score"].to_numpy(dtype=float))
    total_counts = pd.Series(1, index=oos_dates).groupby(level=0).sum().reindex(expected_days)
    finite_counts = (
        pd.Series(finite.astype(int), index=oos_dates)
        .groupby(level=0)
        .sum()
        .reindex(expected_days, fill_value=0)
    )
    coverage = finite_counts.div(total_counts)
    good = (finite_counts >= min_daily_finite) & (coverage >= min_coverage_ratio)
    good_day_rate = float(good.mean()) if len(good) else 0.0
    if good_day_rate < min_good_day_rate:
        raise ValueError("model prediction cross-sectional OOS coverage is insufficient")
    return {
        "contract_version": MODEL_PREDICTION_CONTRACT_VERSION,
        "artifact_path": str(artifact),
        "artifact_sha256": expected_sha256.lower(),
        "test_start": start.date().isoformat(),
        "test_end": end.date().isoformat(),
        "trading_day_count": len(expected_days),
        "row_count": len(oos),
        "finite_row_count": int(finite.sum()),
        "minimum_daily_finite_observed": int(finite_counts.min()),
        "minimum_coverage_ratio_observed": float(coverage.min()),
        "good_day_rate": good_day_rate,
        "coverage_gate_passed": True,
    }


def _require_finite_metrics(metrics: Mapping[str, Any]) -> None:
    missing = [name for name in REQUIRED_MODEL_METRICS if name not in metrics]
    if missing:
        raise ValueError(f"independent model metrics are incomplete: {missing}")
    for name in REQUIRED_MODEL_METRICS:
        try:
            value = float(metrics[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"independent model metric {name} is not numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"independent model metric {name} is not finite")


def model_metric_gate_failures(metrics: Mapping[str, Any]) -> list[str]:
    """Report threshold misses; malformed or missing metrics still raise."""
    _require_finite_metrics(metrics)
    failures: list[str] = []
    minimums = {
        "ic": MODEL_METRIC_GATE["minimum_ic"],
        "icir": MODEL_METRIC_GATE["minimum_icir"],
        "rank_ic": MODEL_METRIC_GATE["minimum_rank_ic"],
        "rank_icir": MODEL_METRIC_GATE["minimum_rank_icir"],
        "information_ratio": MODEL_METRIC_GATE["minimum_information_ratio"],
        "annualized_excess_return_with_cost": MODEL_METRIC_GATE[
            "minimum_annualized_excess_return_with_cost"
        ],
    }
    for name, threshold in minimums.items():
        value = float(metrics[name])
        if value < float(threshold):
            failures.append(f"{name}={value:.8g} < {float(threshold):.8g}")
    drawdown = abs(float(metrics["max_drawdown"]))
    maximum_drawdown = float(MODEL_METRIC_GATE["maximum_absolute_drawdown"])
    if drawdown > maximum_drawdown:
        failures.append(f"abs(max_drawdown)={drawdown:.8g} > {maximum_drawdown:.8g}")
    return failures


def model_metric_report(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Keep economic observations separate from structural research admission."""
    failures = model_metric_gate_failures(metrics)
    return {
        "contract_version": MODEL_METRIC_GATE_VERSION,
        "statistical_evidence_role": "report_only",
        "thresholds": dict(MODEL_METRIC_GATE),
        "gate_passed": not failures,
        "failure_reasons": failures,
    }


def require_model_metric_report(
    cell: Mapping[str, Any], *, context: str, report_only: bool = False
) -> None:
    expected = model_metric_report(cell.get("metrics") or {})
    # Historical evidence remains byte-for-byte intact. Newly recorded reports
    # must agree with the metrics; a report-only label cannot conceal bad data.
    if "metric_report" in cell and cell["metric_report"] != expected:
        raise ValueError(f"{context} metric report is inconsistent")
    if "metric_report" not in cell and not report_only:
        require_model_metric_gate(cell.get("metrics") or {}, context=context)


def require_model_metric_gate(metrics: Mapping[str, Any], *, context: str) -> None:
    """Historical threshold contract for evidence without a report-only marker."""
    failures = model_metric_gate_failures(metrics)
    if failures:
        raise ValueError(f"{context} failed {MODEL_METRIC_GATE_VERSION}: " + "; ".join(failures))


def validate_independent_model_evidence(
    evidence: Mapping[str, Any],
    *,
    candidate_id: str,
    dataset_identity_sha256: str,
    pre_final_end: str,
) -> dict[str, Any]:
    if evidence.get("contract_version") != MODEL_RESEARCH_CONTRACT_VERSION:
        raise ValueError("model evidence contract is missing or obsolete")
    if evidence.get("source") != "independent_qlib_recompute":
        raise ValueError("RD-Agent internal model scores cannot pass the independent gate")
    if str(evidence.get("candidate_id") or "") != candidate_id:
        raise ValueError("model evidence belongs to another candidate")
    if (
        not is_sha256(dataset_identity_sha256)
        or str(evidence.get("dataset_identity_sha256") or "").lower()
        != dataset_identity_sha256.lower()
    ):
        raise ValueError("model evidence dataset identity is invalid")
    if evidence.get("final_oos_opened") is not False:
        raise ValueError("research-stage model evidence must not open the final OOS")
    cadence_value = evidence.get("research_execution_cadence")
    if cadence_value is not None:
        cadence = validate_research_execution_cadence_contract(
            cadence_value,
            expected_horizon_profile=str(evidence.get("horizon_profile") or ""),
        )
        if (
            evidence.get("research_execution_cadence_sha256")
            != cadence["evidence_sha256"]
        ):
            raise ValueError("model evidence decision cadence digest is invalid")
    execution_environment_sha256 = evidence.get("execution_environment_sha256")
    if not is_sha256(execution_environment_sha256):
        raise ValueError("model evidence has no immutable execution environment")
    selected_trial_name = str(
        evidence.get("multiple_testing_trial_name") or candidate_id
    )
    validate_run_multiple_testing_evidence(
        evidence.get("multiple_testing"), selected_trial_name=selected_trial_name
    )

    profiles = evidence.get("profiles")
    if not isinstance(profiles, Mapping) or set(profiles) != set(REQUIRED_RESEARCH_PROFILES):
        raise ValueError(
            "model evidence must contain recent_3y, balanced_5y and robust_10y profiles"
        )
    for profile_name in REQUIRED_RESEARCH_PROFILES:
        profile = profiles[profile_name]
        if not isinstance(profile, Mapping):
            raise ValueError(f"model profile {profile_name} is invalid")
        periods = profile.get("periods")
        required_periods = {
            "train_start",
            "train_end",
            "valid_start",
            "valid_end",
            "test_start",
            "test_end",
        }
        if not isinstance(periods, Mapping) or not required_periods.issubset(periods):
            raise ValueError(f"model profile {profile_name} periods are incomplete")
        if pd.Timestamp(periods["valid_end"]) > pd.Timestamp(pre_final_end) or pd.Timestamp(
            periods["valid_end"]
        ) >= pd.Timestamp(periods["test_start"]):
            raise ValueError(f"model profile {profile_name} reaches the sealed final OOS")
        seeds = profile.get("seeds")
        if not isinstance(seeds, Mapping) or {int(seed) for seed in seeds} != set(
            REQUIRED_MODEL_SEEDS
        ):
            raise ValueError(f"model profile {profile_name} does not contain the fixed seeds")
        for seed in REQUIRED_MODEL_SEEDS:
            result = seeds.get(str(seed), seeds.get(seed))
            if not isinstance(result, Mapping) or result.get("status") != "passed":
                raise ValueError(f"model profile {profile_name} seed {seed} did not pass")
            if pd.Timestamp(result.get("latest_prediction_date")) > pd.Timestamp(
                periods["valid_end"]
            ):
                raise ValueError(
                    f"model profile {profile_name} seed {seed} exposed final OOS predictions"
                )
            require_model_metric_report(
                result,
                context=f"model profile {profile_name} seed {seed}",
                report_only=evidence["multiple_testing"].get("statistical_evidence_role")
                == "report_only",
            )
            if not is_sha256(result.get("predictions_sha256")):
                raise ValueError(f"model profile {profile_name} seed {seed} has no prediction hash")
            if not is_sha256(result.get("portfolio_report_sha256")) or not str(
                result.get("portfolio_report_path") or ""
            ):
                raise ValueError(
                    f"model profile {profile_name} seed {seed} has no return-series proof"
                )
            if not is_sha256(result.get("execution_evidence_sha256")):
                raise ValueError(f"model profile {profile_name} seed {seed} has no execution proof")
            if result.get("execution_environment_sha256") != execution_environment_sha256:
                raise ValueError(
                    f"model profile {profile_name} seed {seed} used another environment"
                )

    normalized = dict(evidence)
    normalized["evidence_sha256"] = canonical_sha256(
        {key: value for key, value in normalized.items() if key != "evidence_sha256"}
    )
    recorded = evidence.get("evidence_sha256")
    if recorded is not None and recorded != normalized["evidence_sha256"]:
        raise ValueError("model evidence SHA-256 is invalid")
    return normalized


def validate_quant_bundle_evidence(
    bundle: Mapping[str, Any],
    *,
    dataset_identity_sha256: str,
) -> dict[str, Any]:
    contract_version = str(bundle.get("contract_version") or "")
    if contract_version not in {
        QUANT_BUNDLE_CONTRACT_VERSION,
        LEGACY_QUANT_BUNDLE_CONTRACT_VERSION,
    }:
        raise ValueError("quant bundle contract is missing or obsolete")
    if not is_sha256(dataset_identity_sha256) or (
        str(bundle.get("dataset_identity_sha256") or "").lower() != dataset_identity_sha256.lower()
    ):
        raise ValueError("quant bundle dataset identity is invalid")
    factors = bundle.get("factors")
    model = bundle.get("model")
    if not isinstance(factors, list) or not factors or not isinstance(model, Mapping):
        raise ValueError("quant bundle must freeze at least one factor and one model")
    baseline = bundle.get("baseline_prediction_champion")
    if contract_version == QUANT_BUNDLE_CONTRACT_VERSION:
        if not isinstance(baseline, Mapping):
            raise ValueError("quant bundle has no frozen incumbent prediction")
        baseline_sha256 = canonical_sha256(
            {
                key: value
                for key, value in baseline.items()
                if key != "evidence_sha256"
            }
        )
        baseline_kind = str(baseline.get("kind") or "")
        if (
            baseline.get("contract_version")
            != "fin-quant-baseline-prediction-v1"
            or baseline_kind not in {"model", "ensemble"}
            or baseline.get("dataset_identity_sha256")
            != dataset_identity_sha256.lower()
            or baseline.get("quant_retraining_supported") is not True
            or baseline.get("evidence_sha256") != baseline_sha256
            or bundle.get("baseline_prediction_champion_sha256")
            != baseline_sha256
            or not is_sha256(baseline.get("candidate_manifest_sha256"))
            or not is_sha256(baseline.get("admission_evidence_sha256"))
            or not is_sha256(baseline.get("selection_evidence_sha256"))
        ):
            raise ValueError("quant bundle incumbent prediction identity is invalid")
        if baseline_kind == "ensemble":
            components = baseline.get("components")
            if (
                baseline.get("combiner") != "equal_rank"
                or baseline.get("stacking") is not False
                or baseline.get("member_retraining_contract_version")
                != "fin-quant-ensemble-member-retraining-v1"
                or not isinstance(components, list)
                or not 2 <= len(components) <= 3
                or not isinstance(baseline.get("profiles"), Mapping)
            ):
                raise ValueError("quant bundle incumbent ensemble recipe is invalid")
            component_ids = [
                str(item.get("model_candidate_id") or "")
                for item in components
                if isinstance(item, Mapping)
            ]
            if (
                len(component_ids) != len(components)
                or "" in component_ids
                or len(set(component_ids)) != len(component_ids)
            ):
                raise ValueError("quant bundle incumbent ensemble members are invalid")
    factor_ids: set[str] = set()
    for factor in factors:
        if not isinstance(factor, Mapping):
            raise ValueError("quant bundle factor record is invalid")
        factor_id = str(factor.get("candidate_id") or "")
        if not factor_id or factor_id in factor_ids or not is_sha256(factor.get("code_sha256")):
            raise ValueError("quant bundle factor identities are invalid")
        factor_ids.add(factor_id)
    horizon_profile = str(bundle.get("horizon_profile") or "").strip()
    horizon_factor_bundle = bundle.get("horizon_factor_bundle")
    if horizon_profile:
        execution_cadence = validate_research_execution_cadence_contract(
            bundle.get("research_execution_cadence") or {},
            expected_horizon_profile=horizon_profile,
        )
        if (
            bundle.get("research_execution_cadence_sha256")
            != execution_cadence["evidence_sha256"]
        ):
            raise ValueError("quant bundle decision cadence digest is invalid")
        factor_bundle = validate_horizon_factor_bundle(
            horizon_factor_bundle if isinstance(horizon_factor_bundle, Mapping) else {}
        )
        observed_incremental = [
            {
                "candidate_id": str(factor.get("candidate_id") or ""),
                "code_sha256": str(factor.get("code_sha256") or ""),
            }
            for factor in factors
        ]
        challenge = factor_bundle["incremental_challenge"]
        if (
            bundle.get("horizon_factor_bundle_sha256")
            != factor_bundle["bundle_sha256"]
            or factor_bundle["horizon_profile"] != horizon_profile
            or int(factor_bundle["label_horizon_sessions"])
            != int(bundle.get("label_horizon_sessions") or 0)
            or factor_bundle["research_label_binding_sha256"]
            != bundle.get("research_label_binding_sha256")
            or factor_bundle["dataset_identity_sha256"]
            != dataset_identity_sha256.lower()
            or factor_bundle["base_feature_set"]["definition_sha256"]
            != bundle.get("feature_set_definition_sha256")
            or factor_bundle["incremental_factors"] != observed_incremental
            or not isinstance(baseline, Mapping)
            or challenge["incumbent_kind"] != baseline.get("kind")
            or challenge["incumbent_candidate_id"]
            != baseline.get("candidate_id")
            or challenge["incumbent_evidence_sha256"]
            != baseline.get("evidence_sha256")
        ):
            raise ValueError("quant horizon factor bundle identity is invalid")
    elif (
        horizon_factor_bundle is not None
        or bundle.get("horizon_factor_bundle_sha256") is not None
    ):
        raise ValueError("legacy quant evidence cannot claim a horizon factor bundle")
    if not is_sha256(model.get("code_sha256")) or not is_sha256(model.get("recipe_sha256")):
        raise ValueError("quant bundle model code and recipe must be immutable")
    ablations = bundle.get("ablations")
    if not isinstance(ablations, Mapping) or set(ablations) != set(REQUIRED_QUANT_ABLATIONS):
        raise ValueError("quant bundle requires factor-only, model-only and joint ablations")
    family = str(bundle.get("experiment_family_id") or "")
    execution_environment_sha256 = bundle.get("execution_environment_sha256")
    if not is_sha256(execution_environment_sha256):
        raise ValueError("quant bundle has no immutable execution environment")
    for name in REQUIRED_QUANT_ABLATIONS:
        result = ablations[name]
        if not isinstance(result, Mapping) or result.get("status") != "passed":
            raise ValueError(f"quant bundle ablation {name} did not pass")
        if result.get("experiment_family_id") != family:
            raise ValueError("quant bundle ablations do not share one experiment family")
        if result.get("source") != "independent_qlib_recompute":
            raise ValueError(f"quant bundle ablation {name} is not independently recomputed")
        if result.get("final_oos_opened") is not False:
            raise ValueError(f"quant bundle ablation {name} opened the sealed final OOS")
        if result.get("dataset_identity_sha256") != dataset_identity_sha256.lower():
            raise ValueError(f"quant bundle ablation {name} used another dataset")
        if result.get("execution_environment_sha256") != execution_environment_sha256:
            raise ValueError(f"quant bundle ablation {name} used another environment")
        if horizon_profile and (
            result.get("research_execution_cadence") != execution_cadence
            or result.get("research_execution_cadence_sha256")
            != execution_cadence["evidence_sha256"]
        ):
            raise ValueError(f"quant bundle ablation {name} changed decision cadence")
        profiles = result.get("profiles")
        if not isinstance(profiles, Mapping) or set(profiles) != set(REQUIRED_RESEARCH_PROFILES):
            raise ValueError(f"quant bundle ablation {name} misses governed profiles")
        for profile_name in REQUIRED_RESEARCH_PROFILES:
            profile = profiles[profile_name]
            if not isinstance(profile, Mapping):
                raise ValueError(f"quant ablation {name}/{profile_name} is invalid")
            seeds = profile.get("seeds")
            if not isinstance(seeds, Mapping) or {int(seed) for seed in seeds} != set(
                REQUIRED_MODEL_SEEDS
            ):
                raise ValueError(f"quant ablation {name}/{profile_name} misses fixed seeds")
            periods = profile.get("periods")
            if not isinstance(periods, Mapping) or not periods.get("valid_end"):
                raise ValueError(f"quant ablation {name}/{profile_name} periods are invalid")
            for seed in REQUIRED_MODEL_SEEDS:
                seed_result = seeds.get(str(seed), seeds.get(seed))
                if not isinstance(seed_result, Mapping) or seed_result.get("status") != "passed":
                    raise ValueError(f"quant ablation {name}/{profile_name}/{seed} failed")
                if pd.Timestamp(seed_result.get("latest_prediction_date")) > pd.Timestamp(
                    periods["valid_end"]
                ):
                    raise ValueError(
                        f"quant ablation {name}/{profile_name}/{seed} exposed final OOS"
                    )
                require_model_metric_report(
                    seed_result,
                    context=f"quant {name} {profile_name}/{seed}",
                    report_only=name != "joint" or (
                        isinstance(bundle.get("multiple_testing"), Mapping)
                        and bundle["multiple_testing"].get("statistical_evidence_role")
                        == "report_only"
                    ),
                )
                if not is_sha256(seed_result.get("predictions_sha256")) or not is_sha256(
                    seed_result.get("execution_evidence_sha256")
                ):
                    raise ValueError(
                        f"quant ablation {name}/{profile_name}/{seed} evidence is incomplete"
                    )
                if (
                    isinstance(baseline, Mapping)
                    and baseline.get("kind") == "ensemble"
                    and name == "factor_only"
                ):
                    member_artifacts = seed_result.get("member_artifacts")
                    baseline_components = baseline.get("components") or []
                    expected_member_ids = {
                        str(item.get("model_candidate_id") or "")
                        for item in baseline_components
                    }
                    observed_member_ids = {
                        str(item.get("model_candidate_id") or "")
                        for item in member_artifacts or []
                        if isinstance(item, Mapping)
                    }
                    if (
                        seed_result.get("prediction_component_kind") != "ensemble"
                        or seed_result.get("combiner") != "equal_rank"
                        or seed_result.get("stacking") is not False
                        or not isinstance(member_artifacts, list)
                        or observed_member_ids != expected_member_ids
                        or len(member_artifacts) != len(expected_member_ids)
                        or any(
                            not is_sha256(item.get("predictions_sha256"))
                            or not is_sha256(item.get("checkpoint_sha256"))
                            or not is_sha256(item.get("portfolio_report_sha256"))
                            or not is_sha256(item.get("execution_evidence_sha256"))
                            for item in member_artifacts
                        )
                    ):
                        raise ValueError(
                            f"quant ensemble factor-only {profile_name}/{seed} "
                            "member evidence is incomplete"
                        )
                if not is_sha256(seed_result.get("portfolio_report_sha256")) or not str(
                    seed_result.get("portfolio_report_path") or ""
                ):
                    raise ValueError(
                        f"quant ablation {name}/{profile_name}/{seed} has no return-series proof"
                    )
                if (
                    seed_result.get("execution_environment_sha256")
                    != execution_environment_sha256
                ):
                    raise ValueError(
                        f"quant ablation {name}/{profile_name}/{seed} used another environment"
                    )
        if not is_sha256(result.get("evidence_sha256")):
            raise ValueError(f"quant bundle ablation {name} has no immutable evidence")
        expected_ablation_sha = canonical_sha256(
            {key: value for key, value in result.items() if key != "evidence_sha256"}
        )
        if result.get("evidence_sha256") != expected_ablation_sha:
            raise ValueError(f"quant bundle ablation {name} evidence SHA-256 is invalid")
    multiple_testing = validate_run_multiple_testing_evidence(
        bundle.get("multiple_testing"),
        selected_trial_name=f"{bundle.get('id')}:joint",
    )
    bundle_id = str(bundle.get("id") or "")
    current_trials = {
        str(item.get("name") or ""): dict(item)
        for item in multiple_testing["trial_definitions"]
        if str(item.get("candidate_id") or "") == bundle_id
        and str(item.get("kind") or "") == "quant_bundle"
    }
    expected_trials = {
        f"{bundle_id}:{ablation}" for ablation in REQUIRED_QUANT_ABLATIONS
    }
    if not bundle_id or set(current_trials) != expected_trials or any(
        str(current_trials[f"{bundle_id}:{ablation}"].get("ablation") or "")
        != ablation
        for ablation in REQUIRED_QUANT_ABLATIONS
    ):
        raise ValueError("quant run-level family does not contain all frozen ablations")
    if contract_version == QUANT_BUNDLE_CONTRACT_VERSION:
        if not isinstance(baseline, Mapping):
            raise ValueError("quant bundle incumbent prediction is missing")
        incumbent_id = str(baseline.get("candidate_id") or "")
        incumbent_trials = [
            item
            for item in multiple_testing["trial_definitions"]
            if str(item.get("name") or "") == f"incumbent:{incumbent_id}"
            and str(item.get("candidate_id") or "") == incumbent_id
            and str(item.get("ablation") or "") == "incumbent"
        ]
        delta_trial_name = f"{bundle_id}:joint_vs_incumbent"
        delta_trials = [
            item
            for item in multiple_testing["trial_definitions"]
            if str(item.get("name") or "") == delta_trial_name
            and str(item.get("candidate_id") or "") == bundle_id
            and str(item.get("kind") or "") == "fin_quant_joint_delta"
            and str(item.get("ablation") or "") == "joint_vs_incumbent"
        ]
        trial_names = [str(value) for value in multiple_testing["trial_names"]]
        delta_index = (
            trial_names.index(delta_trial_name)
            if delta_trial_name in trial_names
            else -1
        )
        raw_delta_p = (
            float(multiple_testing["raw_p_values"][delta_index])
            if delta_index >= 0
            else math.nan
        )
        adjusted_delta_p = (
            float(multiple_testing["holm_adjusted_p_values"][delta_index])
            if delta_index >= 0
            else math.nan
        )
        delta_mean = (
            float(multiple_testing["trial_daily_means"][delta_index])
            if delta_index >= 0
            and len(multiple_testing.get("trial_daily_means") or [])
            == len(trial_names)
            else math.nan
        )
        comparison = bundle.get("incumbent_comparison")
        if not isinstance(comparison, Mapping):
            raise ValueError("quant bundle has no frozen incumbent comparison")
        comparison_sha256 = canonical_sha256(
            {
                key: value
                for key, value in comparison.items()
                if key != "evidence_sha256"
            }
        )
        pbo_eligible = delta_trial_name in multiple_testing["eligible_trial_names"]
        expected_passed = delta_mean > 0.0 and adjusted_delta_p <= 0.05 and pbo_eligible
        report_only = comparison.get("statistical_evidence_role") == "report_only"
        if (
            len(incumbent_trials) != 1
            or len(delta_trials) != 1
            or comparison.get("contract_version")
            != "fin-quant-incumbent-comparison-v2"
            or comparison.get("candidate_id") != bundle_id
            or comparison.get("incumbent_candidate_id") != incumbent_id
            or comparison.get("profile_id") != PRIMARY_MODEL_PROFILE
            or comparison.get("seed_aggregation") != "equal_mean_fixed_seeds"
            or comparison.get("multiple_testing_trial_name") != delta_trial_name
            or comparison.get("pbo_eligible") is not pbo_eligible
            or comparison.get("final_oos_opened") is not False
            or comparison.get("statistical_evidence_role") not in {None, "report_only"}
            or (report_only and multiple_testing.get("statistical_evidence_role") != "report_only")
            or comparison.get("passed") is not expected_passed
            or (not report_only and not expected_passed)
            or not math.isfinite(delta_mean)
            or float(comparison.get("family_observed_mean_difference", math.nan))
            != delta_mean
            or float(comparison.get("raw_one_sided_p_value", math.nan))
            != raw_delta_p
            or float(comparison.get("holm_adjusted_one_sided_p_value", math.nan))
            != adjusted_delta_p
            or comparison.get("maximum_one_sided_p_value") != 0.05
            or not math.isfinite(float(comparison.get("observed_mean_difference", math.nan)))
            or not 0.0 <= float(comparison.get("one_sided_p_value", math.nan)) <= 1.0
            or comparison.get("evidence_sha256") != comparison_sha256
        ):
            raise ValueError("quant joint frozen incumbent comparison is invalid")
    payload = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    expected = canonical_sha256(payload)
    if bundle.get("bundle_sha256") != expected:
        raise ValueError("quant bundle was changed after it was frozen")
    return dict(bundle)
