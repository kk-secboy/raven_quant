#!/usr/bin/env python3
"""Paired incremental ablation against one immutable frozen-model baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_data.qlib_builder import verify_qlib_output_manifest
from quant_platform.cost_model import CostScheduleBook
from quant_platform.factor_evaluator import (
    evaluate_factor_values,
    normalize_series,
    purged_factor_evaluation_days,
)
from quant_platform.factor_library_store import (
    INCREMENTAL_EVIDENCE_VERSION,
    ResearchSotaPolicy,
    validate_incremental_evidence,
    validate_sota_members,
)
from quant_platform.model_recompute import execute_model_candidate
from quant_platform.rdagent_dataset_view import prepare_rdagent_dataset_view
from quant_platform.statistical_validation import (
    benjamini_hochberg,
    newey_west_mean_test,
    paired_moving_block_bootstrap,
)

PROFILE_IDS = ("recent_3y", "balanced_5y", "robust_10y")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        return pd.read_hdf(source)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"unsupported prediction format: {source.suffix}")


def _sealed_prediction(path_value: Any, expected_sha256: Any, label: str) -> pd.Series:
    path = Path(str(path_value or ""))
    expected = str(expected_sha256 or "")
    if not path.is_file() or len(expected) != 64 or _sha256_file(path) != expected:
        raise ValueError(f"{label} prediction artifact is missing or changed")
    return normalize_series(_load(str(path)), label)


def _index_sha256(index: pd.MultiIndex) -> str:
    records = [
        [pd.Timestamp(day).isoformat(), str(instrument)]
        for day, instrument in index.to_list()
    ]
    return _canonical_sha256(records)


def _paired_selection(
    baseline: pd.Series,
    proposed: pd.Series,
    labels: pd.Series | pd.DataFrame,
    *,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    label_horizon_days: int,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    baseline = normalize_series(baseline, "baseline")
    proposed = normalize_series(proposed, "proposed")
    if not baseline.index.equals(proposed.index):
        raise ValueError("baseline and proposed predictions must use the exact same index")
    label = normalize_series(labels, "label")
    joined = pd.concat([baseline, proposed, label], axis=1, join="inner").dropna()
    joined.columns = ["baseline", "proposed", "label"]
    dates = pd.DatetimeIndex(joined.index.get_level_values("datetime"))
    joined = joined[(dates >= valid_start) & (dates <= valid_end)]
    if joined.empty:
        raise ValueError("paired predictions do not cover the validation window")
    windows = purged_factor_evaluation_days(
        pd.DatetimeIndex(joined.index.get_level_values("datetime").unique()),
        label_horizon_days=label_horizon_days,
    )
    selection_days = windows["selection"]
    assert isinstance(selection_days, pd.DatetimeIndex)
    selected = joined[
        pd.DatetimeIndex(joined.index.get_level_values("datetime")).isin(selection_days)
    ]
    return selected, selection_days


def _daily_rank_ic(frame: pd.DataFrame, column: str) -> pd.Series:
    def correlation(group: pd.DataFrame) -> float:
        if len(group) < 5 or group[column].nunique() < 2 or group["label"].nunique() < 2:
            return float("nan")
        return float(group[column].rank().corr(group["label"].rank()))

    return frame.groupby(level="datetime", sort=True).apply(correlation).dropna()


def _daily_long_short_net(
    frame: pd.DataFrame,
    column: str,
    *,
    cost_rate: float,
    holding_period_days: int,
) -> pd.Series:
    previous: pd.Series | None = None
    values: dict[pd.Timestamp, float] = {}
    for period_index, (timestamp, group) in enumerate(
        frame.groupby(level="datetime", sort=True)
    ):
        if period_index % holding_period_days or len(group) < 10:
            continue
        ranks = group[column].rank(method="average", pct=True)
        long_index = ranks[ranks >= 0.8].index.get_level_values("instrument")
        short_index = ranks[ranks <= 0.2].index.get_level_values("instrument")
        if not len(long_index) or not len(short_index):
            continue
        weights = pd.Series(0.0, index=group.index.get_level_values("instrument").unique())
        weights.loc[long_index] = 1.0 / len(long_index)
        weights.loc[short_index] = -1.0 / len(short_index)
        returns = group.droplevel("datetime")["label"].groupby(level="instrument").mean()
        gross = float(weights.reindex(returns.index, fill_value=0.0).dot(returns))
        turnover = 1.0
        if previous is not None:
            union = previous.index.union(weights.index)
            turnover = float(
                0.5
                * (
                    weights.reindex(union, fill_value=0.0)
                    - previous.reindex(union, fill_value=0.0)
                )
                .abs()
                .sum()
            )
        previous = weights
        values[pd.Timestamp(timestamp)] = gross - turnover * cost_rate
    result = pd.Series(values, dtype=float).sort_index()
    if len(result) < 30 or not np.isfinite(result.to_numpy()).all():
        raise ValueError("paired cost-return test requires at least 30 finite cohorts")
    return result


def _ablation_hard_gate(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    checks = {
        "selection_days": lambda value: value >= 100,
        "coverage_pass_rate": lambda value: value >= 0.95,
        "mean_coverage_ratio": lambda value: value >= 0.80,
        "constant_day_rate": lambda value: value <= 0.05,
    }
    reasons: list[str] = []
    for name, predicate in checks.items():
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            reasons.append(f"{name} is missing or non-finite")
        elif not predicate(float(value)):
            reasons.append(f"{name} failed the frozen-model prediction gate")
    return ("passed" if not reasons else "failed", reasons)


def _validate_frozen_model(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("SOTA incremental ablation requires frozen_model")
    model = dict(value)
    if str(model.get("kind") or "") in {
        "",
        "fixed_cross_sectional_zscore_linear",
        "point_estimate_linear_composite",
    }:
        raise ValueError("SOTA incremental ablation cannot use a fixed linear score proxy")
    for key in (
        "model_artifact_id",
        "model_artifact_sha256",
        "training_recipe_sha256",
        "feature_contract_sha256",
    ):
        text = str(model.get(key) or "")
        if not text or (key.endswith("sha256") and len(text) != 64):
            raise ValueError(f"frozen_model {key} is invalid")
    return model


def _generated_ablation_predictions(
    manifest: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    provider: Path,
    artifact_root: Path,
) -> None:
    """Fit the exact same frozen recipe with and without each factor.

    The model recipe, seed, train/validation windows and universe are fixed.
    Only the declared candidate column (or its correlated-cluster replacement)
    changes.  The truncated provider and ``final_oos_opened=False`` prevent the
    model runner from reading the final test window.
    """

    frozen = _validate_frozen_model(manifest.get("frozen_model"))
    feature_set = frozen.get("feature_set")
    if (
        not isinstance(feature_set, dict)
        or not isinstance(feature_set.get("features"), dict)
        or not feature_set["features"]
        or _canonical_sha256(
            {key: value for key, value in feature_set.items() if key != "definition_sha256"}
        )
        != feature_set.get("definition_sha256")
    ):
        raise ValueError("frozen model feature-set identity is invalid")
    code_path = Path(
        str(
            manifest.get("frozen_model_runtime_code_path")
            or frozen.get("code_path")
            or ""
        )
    )
    if not code_path.is_file() or _sha256_file(code_path) != str(
        frozen.get("code_sha256") or ""
    ):
        raise ValueError("frozen model code artifact is missing or changed")
    if frozen.get("final_oos_opened") is not False:
        raise ValueError("factor ablation cannot use a model that opened final OOS")
    if int(frozen.get("seed") or -1) != 11:
        raise ValueError("factor ablation frozen model seed is invalid")
    profiles = {
        str(item["id"]): dict(item["periods"])
        for item in manifest.get("evaluation_profiles") or []
    }
    if set(profiles) != set(PROFILE_IDS):
        raise ValueError("generated ablation requires all governed profiles")
    valid_ends = {str(item["valid_end"]) for item in profiles.values()}
    if len(valid_ends) != 1:
        raise ValueError("generated ablation profiles must share one pre-final cutoff")
    pre_final_end = next(iter(valid_ends))
    view_root = artifact_root / "model-dataset-view"
    ablation_root = artifact_root / "paired-model-runs"
    for target in (view_root, ablation_root):
        if target.exists():
            shutil.rmtree(target)
    view = prepare_rdagent_dataset_view(provider, view_root, cutoff=pre_final_end)
    runner = Path(__file__).resolve().with_name("model_sandbox_runner.py")

    def execution_manifest(
        *,
        candidate_id: str,
        periods: dict[str, Any],
        proposed_feature_set: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "candidate_id": candidate_id,
            "code_sha256": frozen["code_sha256"],
            "model_type": frozen["model_type"],
            "model_engine": frozen["model_engine"],
            "training_hyperparameters": dict(
                frozen.get("training_hyperparameters") or {}
            ),
            "resource_stage": "full_validation",
            "feature_set": proposed_feature_set,
            "periods": periods,
            "seed": 11,
            "dataset_identity_sha256": manifest["dataset_identity_sha256"],
            "universe": manifest.get("universe", "cn_all"),
            "benchmark": manifest.get("benchmark", "SH000300"),
            "account": 100_000_000,
            "topk": 50,
            "n_drop": 5,
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5.0,
            "prediction_segment": "valid",
            "final_oos_opened": False,
            "inference_only": False,
        }

    baseline_outputs: dict[str, dict[str, Any]] = {}
    for profile_id in PROFILE_IDS:
        periods = profiles[profile_id]
        workspace = ablation_root / "baseline" / profile_id
        result, evidence = execute_model_candidate(
            code_path=code_path,
            provider_path=view,
            manifest=execution_manifest(
                candidate_id=f"factor-ablation-baseline-{profile_id}",
                periods=periods,
                proposed_feature_set=feature_set,
            ),
            workspace=workspace,
            runner_path=runner,
            timeout_seconds=7200,
        )
        baseline_outputs[profile_id] = {
            "path": str(workspace / "output" / "predictions.parquet"),
            "sha256": result["predictions_sha256"],
            "execution_evidence_sha256": evidence["evidence_sha256"],
        }

    for candidate in candidates:
        if candidate.get("preflight_rejection_reason"):
            continue
        candidate_id = str(candidate["factor_candidate_id"])
        factor_path = Path(str(candidate.get("values_path") or ""))
        if not factor_path.is_file() or _sha256_file(factor_path) != str(
            candidate.get("candidate_values_sha256") or ""
        ):
            raise ValueError(f"candidate {candidate_id} values are missing or changed")
        values = normalize_series(_load(str(factor_path)), f"factor-{candidate_id}")
        dates = pd.DatetimeIndex(values.index.get_level_values("datetime"))
        values = values.loc[dates <= pd.Timestamp(pre_final_end)]
        if values.empty:
            raise ValueError(f"candidate {candidate_id} has no pre-final values")
        factor_frame = values.to_frame("CANDIDATE_FACTOR")
        factor_frame.columns = pd.MultiIndex.from_tuples(
            [("feature", "CANDIDATE_FACTOR")]
        )
        factor_file = ablation_root / "factors" / f"{candidate_id}.parquet"
        factor_file.parent.mkdir(parents=True, exist_ok=True)
        factor_frame.to_parquet(factor_file)

        proposed_feature_set = json.loads(json.dumps(feature_set))
        replaced = str(candidate.get("replaced_feature_name") or "")
        if replaced:
            if replaced not in proposed_feature_set["features"]:
                raise ValueError("cluster replacement is absent from the frozen feature set")
            proposed_feature_set["features"].pop(replaced)
        proposed_identity = {
            key: value
            for key, value in proposed_feature_set.items()
            if key != "definition_sha256"
        }
        proposed_feature_set["definition_sha256"] = _canonical_sha256(
            proposed_identity
        )
        predictions: dict[str, Any] = {}
        for profile_id in PROFILE_IDS:
            workspace = ablation_root / candidate_id / profile_id
            try:
                result, evidence = execute_model_candidate(
                    code_path=code_path,
                    provider_path=view,
                    manifest=execution_manifest(
                        candidate_id=(
                            f"factor-ablation-{candidate_id[:24]}-{profile_id}"
                        ),
                        periods=profiles[profile_id],
                        proposed_feature_set=proposed_feature_set,
                    ),
                    workspace=workspace,
                    runner_path=runner,
                    additional_factors_path=factor_file,
                    timeout_seconds=7200,
                )
            except Exception as exc:
                candidate["ablation_generation_error"] = (
                    f"paired model generation failed for {profile_id}: {exc}"
                )
                break
            predictions[profile_id] = {
                "baseline_path": baseline_outputs[profile_id]["path"],
                "baseline_sha256": baseline_outputs[profile_id]["sha256"],
                "proposed_path": str(workspace / "output" / "predictions.parquet"),
                "proposed_sha256": result["predictions_sha256"],
                "latest_prediction_date": result["latest_prediction_date"],
                "baseline_execution_evidence_sha256": baseline_outputs[profile_id][
                    "execution_evidence_sha256"
                ],
                "proposed_execution_evidence_sha256": evidence["evidence_sha256"],
                "proposed_feature_set_definition_sha256": proposed_feature_set[
                    "definition_sha256"
                ],
                "final_oos_observations_exposed": False,
            }
        if len(predictions) == len(PROFILE_IDS):
            candidate["ablation_predictions"] = predictions


def _evaluate_candidate(
    candidate: dict[str, Any],
    *,
    profiles: dict[str, dict[str, Any]],
    labels: pd.Series | pd.DataFrame,
    cost_schedule: CostScheduleBook,
    reference_order_value: float,
    min_daily_instruments: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    horizon = int(candidate.get("label_horizon_days") or 1)
    predictions = candidate.get("ablation_predictions")
    evaluation_ids = candidate.get("profile_evaluation_ids")
    evaluation_hashes = candidate.get("profile_evaluation_evidence_sha256")
    if not isinstance(predictions, dict) or set(predictions) != set(PROFILE_IDS):
        raise ValueError("candidate requires paired predictions for all governed profiles")
    if not isinstance(evaluation_ids, dict) or set(evaluation_ids) != set(PROFILE_IDS):
        raise ValueError("candidate requires all governed factor evaluation ids")
    if not isinstance(evaluation_hashes, dict) or set(evaluation_hashes) != set(PROFILE_IDS):
        raise ValueError("candidate requires all governed factor evaluation hashes")
    profile_evidence: dict[str, dict[str, Any]] = {}
    for profile_id in PROFILE_IDS:
        profile = profiles[profile_id]
        periods = profile["periods"]
        prediction = predictions[profile_id]
        if not isinstance(prediction, dict):
            raise ValueError(f"{profile_id} prediction contract is invalid")
        if prediction.get("final_oos_observations_exposed") is not False:
            raise ValueError("incremental ablation prediction exposed final OOS observations")
        if str(prediction.get("latest_prediction_date") or "") > str(periods["valid_end"]):
            raise ValueError("incremental prediction extends beyond the validation boundary")
        baseline_sha = str(prediction.get("baseline_sha256") or "")
        proposed_sha = str(prediction.get("proposed_sha256") or "")
        baseline = _sealed_prediction(
            prediction.get("baseline_path"), baseline_sha, f"{profile_id} baseline"
        )
        proposed = _sealed_prediction(
            prediction.get("proposed_path"), proposed_sha, f"{profile_id} proposed"
        )
        if not baseline.index.equals(proposed.index):
            raise ValueError("paired ablation predictions do not use the same sample")
        kwargs = {
            "valid_start": pd.Timestamp(periods["valid_start"]).date(),
            "valid_end": pd.Timestamp(periods["valid_end"]).date(),
            "test_start": pd.Timestamp(periods["test_start"]).date(),
            "test_end": pd.Timestamp(periods["test_end"]).date(),
            "cost_schedule": cost_schedule,
            "reference_order_value": reference_order_value,
            "min_daily_instruments": min_daily_instruments,
            "label_horizon_days": horizon,
            "fixed_direction": 1,
        }
        baseline_metrics = evaluate_factor_values(baseline, labels, **kwargs)
        proposed_metrics = evaluate_factor_values(proposed, labels, **kwargs)
        hard_status, hard_reasons = _ablation_hard_gate(proposed_metrics)
        paired, _ = _paired_selection(
            baseline,
            proposed,
            labels,
            valid_start=pd.Timestamp(periods["valid_start"]),
            valid_end=pd.Timestamp(periods["valid_end"]),
            label_horizon_days=horizon,
        )
        baseline_rank = _daily_rank_ic(paired, "baseline")
        proposed_rank = _daily_rank_ic(paired, "proposed")
        common_rank = baseline_rank.index.intersection(proposed_rank.index)
        if len(common_rank) < 30:
            raise ValueError("paired RankIC test requires at least 30 common days")
        rank_test = newey_west_mean_test(
            proposed_rank.loc[common_rank] - baseline_rank.loc[common_rank],
            max_lag=horizon,
        )
        cost_rate = float(proposed_metrics["cost_rate"])
        baseline_net = _daily_long_short_net(
            paired, "baseline", cost_rate=cost_rate, holding_period_days=horizon
        )
        proposed_net = _daily_long_short_net(
            paired, "proposed", cost_rate=cost_rate, holding_period_days=horizon
        )
        common_net = baseline_net.index.intersection(proposed_net.index)
        return_test = paired_moving_block_bootstrap(
            proposed_net.loc[common_net],
            baseline_net.loc[common_net],
            block_size=max(2, min(20, int(math.sqrt(len(common_net))))),
            samples=bootstrap_samples,
            seed=11,
        )
        rank_p_value = rank_test.get("p_value")
        if rank_test.get("status") != "ok" or not isinstance(rank_p_value, (int, float)):
            raise ValueError("paired RankIC HAC test is undefined")
        profile_evidence[profile_id] = {
            "evaluation_evidence_sha256": str(evaluation_hashes[profile_id]),
            "delta_rank_ic": float(proposed_metrics["rank_ic"])
            - float(baseline_metrics["rank_ic"]),
            "delta_cost_adjusted_return": float(proposed_metrics["cost_adjusted_return"])
            - float(baseline_metrics["cost_adjusted_return"]),
            "baseline_rank_ic": float(baseline_metrics["rank_ic"]),
            "proposed_rank_ic": float(proposed_metrics["rank_ic"]),
            "baseline_cost_adjusted_return": float(
                baseline_metrics["cost_adjusted_return"]
            ),
            "proposed_cost_adjusted_return": float(
                proposed_metrics["cost_adjusted_return"]
            ),
            "hard_gate_status": hard_status,
            "hard_gate_reasons": hard_reasons,
            "stability_gate_status": (
                "passed"
                if hard_status == "passed"
                and float(proposed_metrics["rank_ic"]) > 0
                and float(proposed_metrics["cost_adjusted_return"]) >= 0
                else "failed"
            ),
            "paired_rank_ic_hac": rank_test,
            "paired_cost_return_bootstrap": return_test,
            "prediction_evidence": {
                "baseline_prediction_sha256": baseline_sha,
                "proposed_prediction_sha256": proposed_sha,
                "paired_index_sha256": _index_sha256(paired.index),
                "final_oos_observations_exposed": False,
            },
        }
    return {
        "candidate": candidate,
        "profiles": profile_evidence,
        "rank_p_value": float(profile_evidence["recent_3y"]["paired_rank_ic_hac"]["p_value"]),
        "return_p_value": float(
            profile_evidence["recent_3y"]["paired_cost_return_bootstrap"][
                "one_sided_p_value"
            ]
        ),
        "evaluation_ids": {key: str(evaluation_ids[key]) for key in PROFILE_IDS},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import qlib
    from qlib.data import D

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    provider = Path(args.provider_uri).resolve(strict=True)
    provenance = json.loads(
        (provider / "metadata" / "provenance.json").read_text(encoding="utf-8")
    )
    verify_qlib_output_manifest(provider, provenance)
    dataset_identity = str(manifest.get("dataset_identity_sha256") or "")
    if provenance.get("dataset_identity_sha256") != dataset_identity:
        raise ValueError("SOTA evaluation provider identity disagrees with the manifest")
    raw_profiles = manifest.get("evaluation_profiles") or []
    profiles = {str(item.get("id")): item for item in raw_profiles}
    if set(profiles) != set(PROFILE_IDS):
        raise ValueError("SOTA evaluation requires the three governed research profiles")
    frozen_model = _validate_frozen_model(manifest.get("frozen_model"))
    experiment_family_id = str(manifest.get("experiment_family_id") or "")
    if not experiment_family_id:
        raise ValueError("SOTA incremental hypotheses require an experiment family")
    bootstrap_samples = int(manifest.get("bootstrap_samples") or 2000)
    if bootstrap_samples < 100:
        raise ValueError("SOTA bootstrap requires at least 100 samples")

    qlib.init(provider_uri=str(provider), region="cn")
    instruments = D.instruments(str(manifest.get("universe") or "cn_all"))
    candidates = list(manifest.get("candidates") or [])
    output = Path(args.output).resolve()
    if manifest.get("generate_ablation_predictions") is True:
        _generated_ablation_predictions(
            manifest,
            candidates,
            provider=provider,
            artifact_root=output.parent,
        )
    hypothesis_count = int(
        manifest.get("experiment_hypothesis_count") or len(candidates)
    )
    if hypothesis_count < len(candidates):
        raise ValueError("experiment hypothesis count cannot omit preregistered candidates")
    horizons = {int(item.get("label_horizon_days") or 1) for item in candidates}
    if len(horizons) != 1:
        raise ValueError("one SOTA ablation family must use one label horizon")
    horizon = next(iter(horizons))
    labels = D.features(
        instruments,
        [f"Ref($close, -{horizon + 1})/Ref($close, -1)-1"],
        start_time=min(item["periods"]["valid_start"] for item in profiles.values()),
        end_time=max(item["periods"]["valid_end"] for item in profiles.values()),
        freq="day",
    )
    cost_schedule = CostScheduleBook.from_mapping(manifest.get("cost_model"))
    attempts: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    trial_records: list[dict[str, Any]] = []
    for candidate in candidates:
        preflight_reason = str(candidate.get("preflight_rejection_reason") or "")
        if preflight_reason:
            trial_records.append(
                {
                    "factor_candidate_id": str(
                        candidate.get("factor_candidate_id") or ""
                    ),
                    "status": "rejected",
                    "reason": preflight_reason,
                }
            )
            rejected.append(
                {
                    "factor_candidate_id": str(
                        candidate.get("factor_candidate_id") or ""
                    ),
                    "reason": preflight_reason,
                }
            )
            continue
        generation_error = str(candidate.get("ablation_generation_error") or "")
        if generation_error:
            trial_records.append(
                {
                    "factor_candidate_id": str(
                        candidate.get("factor_candidate_id") or ""
                    ),
                    "status": "failed",
                    "reason": generation_error,
                }
            )
            rejected.append(
                {
                    "factor_candidate_id": str(
                        candidate.get("factor_candidate_id") or ""
                    ),
                    "reason": generation_error,
                }
            )
            continue
        try:
            attempts.append(
                _evaluate_candidate(
                    candidate,
                    profiles=profiles,
                    labels=labels,
                    cost_schedule=cost_schedule,
                    reference_order_value=float(manifest["cost_reference_order_value"]),
                    min_daily_instruments=int(manifest.get("min_daily_instruments", 50)),
                    bootstrap_samples=bootstrap_samples,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            trial_records.append(
                {
                    "factor_candidate_id": str(
                        candidate.get("factor_candidate_id") or ""
                    ),
                    "status": "failed",
                    "reason": str(exc),
                }
            )
            rejected.append(
                {
                    "factor_candidate_id": str(
                        candidate.get("factor_candidate_id") or ""
                    ),
                    "reason": str(exc),
                }
            )
    multiplicity_reference = manifest.get("multiplicity_reference") or {}
    prior_rank_p = list(multiplicity_reference.get("rank_ic_p_values") or [])
    prior_return_p = list(multiplicity_reference.get("cost_return_p_values") or [])
    prior_count = hypothesis_count - len(candidates)
    if len(prior_rank_p) > prior_count or len(prior_return_p) > prior_count:
        raise ValueError("multiplicity reference exceeds the shared hypothesis ledger")
    if any(
        not isinstance(value, (int, float)) or not 0 <= float(value) <= 1
        for value in [*prior_rank_p, *prior_return_p]
    ):
        raise ValueError("multiplicity reference p-values are invalid")
    failed_current = len(candidates) - len(attempts)
    rank_q_values = benjamini_hochberg(
        [item["rank_p_value"] for item in attempts]
        + [1.0] * failed_current
        + [float(value) for value in prior_rank_p]
        + [1.0] * (prior_count - len(prior_rank_p))
    )[: len(attempts)]
    return_q_values = benjamini_hochberg(
        [item["return_p_value"] for item in attempts]
        + [1.0] * failed_current
        + [float(value) for value in prior_return_p]
        + [1.0] * (prior_count - len(prior_return_p))
    )[: len(attempts)]
    eligible: list[dict[str, Any]] = []
    profile_periods = {key: dict(profiles[key]["periods"]) for key in PROFILE_IDS}
    for attempt, rank_q, return_q in zip(
        attempts, rank_q_values, return_q_values, strict=True
    ):
        candidate = attempt["candidate"]
        incremental = {
            "version": INCREMENTAL_EVIDENCE_VERSION,
            "factor_candidate_id": str(candidate["factor_candidate_id"]),
            "candidate_code_sha256": str(candidate["candidate_code_sha256"]),
            "candidate_values_sha256": str(candidate["candidate_values_sha256"]),
            "dataset_identity_sha256": dataset_identity,
            "frozen_model": frozen_model,
            "action": "replaced"
            if any(
                str(item.get("similarity_cluster_id"))
                == str(candidate.get("similarity_cluster_id"))
                for item in manifest.get("baseline_members") or []
            )
            else "added",
            "evaluation_ids": attempt["evaluation_ids"],
            "profile_periods": profile_periods,
            "window_role_policy": {
                "version": "nested-profile-roles-v1",
                "recent_role": "ranking_and_significance",
                "balanced_role": "non_degradation",
                "robust_role": "direction_and_crash_stress",
                "nested_windows_count_as_independent": False,
                "combined_profile_p_value": None,
            },
            "multiplicity": {
                "method": "benjamini_hochberg",
                "experiment_family_id": experiment_family_id,
                "hypothesis_count": hypothesis_count,
                "rank_ic_q_value": float(rank_q),
                "cost_return_q_value": float(return_q),
                # Wide-in, strict-out: adjusted significance is sealed for the
                # archive; it never vetoes the increment.
                "statistical_evidence_role": "report_only",
            },
            "profiles": attempt["profiles"],
        }
        try:
            validate_incremental_evidence(incremental)
        except ValueError as exc:
            trial_records.append(
                {
                    "factor_candidate_id": str(candidate["factor_candidate_id"]),
                    "status": "rejected",
                    "reason": str(exc),
                    "rank_ic_p_value": attempt["rank_p_value"],
                    "cost_return_p_value": attempt["return_p_value"],
                    "rank_ic_q_value": float(rank_q),
                    "cost_return_q_value": float(return_q),
                    "profiles": attempt["profiles"],
                }
            )
            rejected.append(
                {"factor_candidate_id": str(candidate["factor_candidate_id"]), "reason": str(exc)}
            )
            continue
        trial_records.append(
            {
                "factor_candidate_id": str(candidate["factor_candidate_id"]),
                "status": "eligible",
                "rank_ic_p_value": attempt["rank_p_value"],
                "cost_return_p_value": attempt["return_p_value"],
                "rank_ic_q_value": float(rank_q),
                "cost_return_q_value": float(return_q),
                "profiles": attempt["profiles"],
                "incremental_evidence_sha256": _canonical_sha256(incremental),
            }
        )
        eligible.append({"candidate": candidate, "incremental": incremental})

    # One atomic SOTA replacement per frozen family keeps every accepted
    # increment paired to the exact baseline that it was tested against.
    eligible.sort(
        key=lambda item: (
            max(
                item["incremental"]["multiplicity"]["rank_ic_q_value"],
                item["incremental"]["multiplicity"]["cost_return_q_value"],
            ),
            -float(item["incremental"]["profiles"]["recent_3y"]["delta_rank_ic"]),
            str(item["candidate"]["factor_candidate_id"]),
        )
    )
    selected = eligible[:1]
    selected_ids = {
        str(item["candidate"]["factor_candidate_id"]) for item in selected
    }
    for trial in trial_records:
        if trial["status"] == "eligible":
            if str(trial["factor_candidate_id"]) in selected_ids:
                trial["status"] = "accepted"
            else:
                trial["status"] = "rejected"
                trial["reason"] = (
                    "not selected: one atomic increment is allowed per frozen baseline"
                )
    for item in eligible[1:]:
        rejected.append(
            {
                "factor_candidate_id": str(item["candidate"]["factor_candidate_id"]),
                "reason": "not selected: one atomic increment is allowed per frozen baseline",
            }
        )
    baseline = list(manifest.get("baseline_members") or [])
    accepted: list[dict[str, Any]] = []
    if selected:
        candidate = selected[0]["candidate"]
        incremental = selected[0]["incremental"]
        cluster_id = str(candidate["similarity_cluster_id"])
        replaced = next(
            (item for item in baseline if str(item.get("similarity_cluster_id")) == cluster_id),
            None,
        )
        retained = [item for item in baseline if item is not replaced]
        member = {
            **candidate,
            "action": "replaced" if replaced else "added",
            "replaced_factor_candidate_id": (
                replaced.get("factor_candidate_id") if replaced else None
            ),
            "incremental_evidence": incremental,
            "weight": None,
        }
        baseline = [*retained, member]
        accepted.append(member)
    retained_ids = {str(item["factor_candidate_id"]) for item in baseline} - {
        str(item["factor_candidate_id"]) for item in accepted
    }
    members = [
        {**item, "action": "retained"}
        if str(item["factor_candidate_id"]) in retained_ids
        else item
        for item in baseline
    ]
    if accepted:
        validate_sota_members(members, ResearchSotaPolicy())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "status": "passed" if accepted else "failed",
                "contract_version": "factor-sota-paired-frozen-model-v2",
                "dataset_identity_sha256": dataset_identity,
                "predecessor_id": manifest.get("predecessor_id"),
                "frozen_model": frozen_model,
                "members": members,
                "accepted": accepted,
                "rejected": rejected,
                "trials": trial_records,
                "attempted_hypotheses": len(candidates),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
