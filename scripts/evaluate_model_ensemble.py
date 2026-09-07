from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_data.qlib_builder import verify_qlib_output_manifest
from quant_platform.model_ensemble import (
    MODEL_ENSEMBLE_EVALUATION_CONTRACT_VERSION,
    daily_rank_correlation,
    equal_rank_predictions,
    load_sealed_predictions,
    validate_model_ensemble_label_contract,
)
from quant_platform.model_research_governance import (
    REQUIRED_MODEL_SEEDS,
    REQUIRED_RESEARCH_PROFILES,
    build_run_multiple_testing_evidence,
    canonical_sha256,
    file_sha256,
    model_metric_report,
    validate_run_multiple_testing_evidence,
    verify_model_prediction_artifact,
)
from quant_platform.qlib_portfolio_calendar import (
    resolve_qlib_portfolio_calendar_boundary,
)
from quant_platform.qlib_workflow import (
    qlib_workflow_run,
    qlib_workflow_tracking_uri,
)
from quant_platform.research_execution_cadence import (
    validate_research_execution_cadence_contract,
)


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("independent ensemble metric is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError("independent ensemble metric is not finite")
    return result


def _finalize_ensemble_evaluation(item: dict[str, Any], multiple: dict[str, Any]) -> None:
    ensemble_id = str(item["ensemble_id"])
    validate_run_multiple_testing_evidence(multiple, selected_trial_name=ensemble_id)
    evidence = item["evidence"]
    evidence["multiple_testing"] = multiple
    evidence["multiple_testing_trial_name"] = ensemble_id
    evidence["all_metric_cells_passed"] = bool(item["all_metric_cells_passed"])
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    item["evidence_sha256"] = evidence["evidence_sha256"]
    item["status"] = "passed"
    item.pop("all_metric_cells_passed", None)


def _calendar_between(provider: Path, start: str, end: str) -> list[str]:
    calendar = [
        value.strip()
        for value in (provider / "calendars" / "day.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if value.strip()
    ]
    selected = [value for value in calendar if start <= value <= end]
    if not selected or selected[0] != start or selected[-1] != end:
        raise ValueError("ensemble window does not match the Qlib trading calendar")
    return selected


def _normalized_labels(value: pd.DataFrame) -> pd.Series:
    if not isinstance(value.index, pd.MultiIndex) or set(value.index.names) != {
        "datetime",
        "instrument",
    }:
        raise ValueError("Qlib labels do not use the governed model index")
    frame = value.copy()
    if frame.index.names != ["datetime", "instrument"]:
        frame = frame.reorder_levels(["datetime", "instrument"])
    dates = pd.to_datetime(frame.index.get_level_values("datetime"), errors="coerce")
    if dates.isna().any():
        raise ValueError("Qlib labels contain invalid dates")
    frame.index = pd.MultiIndex.from_arrays(
        [dates.tz_localize(None), frame.index.get_level_values("instrument").astype(str)],
        names=["datetime", "instrument"],
    )
    return pd.to_numeric(frame.iloc[:, 0], errors="coerce").rename("label").sort_index()


def _component_cell(
    component: dict[str, Any], profile_id: str, seed: int
) -> dict[str, str]:
    grid = component.get("prediction_grid") or {}
    profile = (grid.get("profiles") or {}).get(profile_id) or {}
    cell = (profile.get("seeds") or {}).get(str(seed)) or {}
    return {
        "model_candidate_id": str(component["model_candidate_id"]),
        "predictions_path": str(cell.get("predictions_path") or ""),
        "predictions_sha256": str(cell.get("predictions_sha256") or "").lower(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    provider = Path(args.provider_uri).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract_version") != "model-ensemble-evaluation-input-v1":
        raise ValueError("model ensemble evaluation input contract is invalid")
    provenance = json.loads(
        (provider / "metadata" / "provenance.json").read_text(encoding="utf-8")
    )
    verify_qlib_output_manifest(provider, provenance)
    identity = str(manifest.get("dataset_identity_sha256") or "")
    if identity != provenance.get("dataset_identity_sha256"):
        raise ValueError("ensemble evaluator received another sealed dataset")
    profiles = manifest.get("evaluation_profiles") or []
    if {str(item.get("id")) for item in profiles} != set(REQUIRED_RESEARCH_PROFILES):
        raise ValueError("ensemble evaluation requires the three governed profiles")
    by_profile = {str(item["id"]): dict(item) for item in profiles}
    valid_ends = {str(item["periods"]["valid_end"]) for item in profiles}
    final_windows = {
        (str(item["periods"]["test_start"]), str(item["periods"]["test_end"]))
        for item in profiles
    }
    if len(valid_ends) != 1 or len(final_windows) != 1:
        raise ValueError("ensemble profiles do not share the pre-final/OOS boundary")
    member_bindings: dict[str, dict[str, Any]] = {}
    for raw_candidate in manifest.get("candidates") or []:
        if not isinstance(raw_candidate, dict):
            raise ValueError("ensemble evaluation candidate is invalid")
        for raw_component in raw_candidate.get("components") or []:
            if not isinstance(raw_component, dict):
                raise ValueError("ensemble evaluation component is invalid")
            member_id = str(raw_component.get("model_candidate_id") or "")
            binding = raw_component.get("research_label_binding")
            if not member_id or not isinstance(binding, dict):
                raise ValueError("ensemble component has no frozen label binding")
            existing = member_bindings.get(member_id)
            if existing is not None and existing != binding:
                raise ValueError("ensemble component label binding changed between candidates")
            member_bindings[member_id] = dict(binding)
    label_contract = validate_model_ensemble_label_contract(
        manifest.get("ensemble_label_contract") or {},
        member_bindings=member_bindings,
    )
    label_identity = dict(label_contract["label_identity"])
    execution_cadence = validate_research_execution_cadence_contract(
        manifest.get("research_execution_cadence") or {},
        expected_horizon_profile=str(label_identity["horizon_profile"]),
    )
    if (
        label_identity.get("dataset_name") != manifest.get("dataset")
        or label_identity.get("dataset_identity_sha256") != identity
        or dict(label_identity.get("periods") or {})
        != dict(by_profile["recent_3y"].get("periods") or {})
    ):
        raise ValueError("ensemble label target differs from its evaluation manifest")
    label_expression = str(label_identity["label_expression"])

    import qlib
    from qlib.contrib.evaluate import risk_analysis
    from qlib.data import D
    from qlib.workflow.record_temp import PortAnaRecord

    qlib.init(provider_uri=str(provider), region="cn")
    output = Path(args.output).resolve()
    artifact_root = output.parent / "independent-ensemble-evaluations"
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True)
    environment = {
        "contract_version": "model-ensemble-execution-environment-v1",
        "evaluator_sha256": file_sha256(Path(__file__).resolve()),
        "qlib_version": str(getattr(qlib, "__version__", "unknown")),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "cpu_only": True,
        "training_performed": False,
        "combiner": "equal_rank",
        "stacking": False,
    }
    environment_path = artifact_root / "execution_environment.json"
    environment_path.write_text(
        json.dumps(environment, sort_keys=True, indent=2), encoding="utf-8"
    )
    environment_sha = canonical_sha256(environment)
    evaluations: list[dict[str, Any]] = []
    trial_series: list[tuple[dict[str, str], pd.Series]] = []
    fatal_error: str | None = None

    for raw_candidate in manifest.get("candidates") or []:
        candidate = dict(raw_candidate)
        ensemble_id = str(candidate.get("id") or "")
        try:
            frozen = dict(candidate.get("manifest") or {})
            if (
                frozen.get("contract_version") != "quantlab-model-ensemble-v2"
                or canonical_sha256(frozen) != candidate.get("manifest_sha256")
                or frozen.get("dataset_identity_sha256") != identity
                or frozen.get("combiner") != "equal_rank"
                or frozen.get("stacking") is not False
            ):
                raise ValueError("ensemble immutable manifest is invalid")
            components = [dict(item) for item in candidate.get("components") or []]
            frozen_components = [dict(item) for item in frozen.get("components") or []]
            if [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"prediction_grid", "research_label_binding"}
                }
                for item in components
            ] != frozen_components:
                raise ValueError("ensemble components changed after preregistration")
            if not 2 <= len(components) <= 3:
                raise ValueError("ensemble must have two or three frozen members")
            expected_correlations: dict[tuple[str, str], dict[str, Any]] = {}
            for raw_correlation in frozen.get("correlation_evidence") or []:
                correlation = dict(raw_correlation)
                pair = (
                    str(correlation.get("left_candidate_id") or ""),
                    str(correlation.get("right_candidate_id") or ""),
                )
                if "" in pair or pair in expected_correlations:
                    raise ValueError("ensemble correlation evidence identities are invalid")
                expected_correlations[pair] = correlation
            evidence: dict[str, Any] = {
                "contract_version": MODEL_ENSEMBLE_EVALUATION_CONTRACT_VERSION,
                "source": "independent_qlib_recompute",
                "ensemble_id": ensemble_id,
                "ensemble_manifest_sha256": str(candidate["manifest_sha256"]),
                "dataset_identity_sha256": identity,
                "ensemble_label_contract": label_contract,
                "ensemble_label_contract_sha256": label_contract["evidence_sha256"],
                "research_execution_cadence": execution_cadence,
                "research_execution_cadence_sha256": execution_cadence[
                    "evidence_sha256"
                ],
                "combiner": "equal_rank",
                "stacking": False,
                "final_oos_opened": False,
                "execution_environment": environment,
                "execution_environment_sha256": environment_sha,
                "execution_environment_path": str(environment_path),
                "execution_environment_file_sha256": file_sha256(environment_path),
                "profiles": {},
            }
            recent_returns: list[pd.Series] = []
            all_cells_passed = True
            for profile_id in REQUIRED_RESEARCH_PROFILES:
                profile = by_profile[profile_id]
                periods = dict(profile["periods"])
                portfolio_calendar_boundary = (
                    resolve_qlib_portfolio_calendar_boundary(
                        provider,
                        backtest_end=periods["valid_end"],
                    )
                )
                trading_days = _calendar_between(
                    provider, periods["valid_start"], periods["valid_end"]
                )
                labels = _normalized_labels(
                    D.features(
                        instruments=str(manifest.get("universe") or "cn_all"),
                        fields=[label_expression],
                        start_time=periods["valid_start"],
                        end_time=periods["valid_end"],
                        freq="day",
                    )
                )
                seed_results: dict[str, Any] = {}
                for seed in REQUIRED_MODEL_SEEDS:
                    member_frames: list[tuple[str, pd.DataFrame]] = []
                    member_artifacts: list[dict[str, str]] = []
                    for component in components:
                        cell = _component_cell(component, profile_id, seed)
                        predictions = load_sealed_predictions(
                            cell["predictions_path"], cell["predictions_sha256"]
                        )
                        verify_model_prediction_artifact(
                            cell["predictions_path"],
                            expected_sha256=cell["predictions_sha256"],
                            test_start=periods["valid_start"],
                            test_end=periods["valid_end"],
                            trading_days=trading_days,
                        )
                        member_frames.append((cell["model_candidate_id"], predictions))
                        member_artifacts.append(cell)
                    combined, combination = equal_rank_predictions(member_frames)
                    cell_root = artifact_root / ensemble_id / profile_id / f"seed-{seed}"
                    cell_root.mkdir(parents=True)
                    predictions_path = cell_root / "predictions.parquet"
                    combined.to_parquet(predictions_path)
                    coverage = verify_model_prediction_artifact(
                        predictions_path,
                        expected_sha256=file_sha256(predictions_path),
                        test_start=periods["valid_start"],
                        test_end=periods["valid_end"],
                        trading_days=trading_days,
                    )
                    aligned = pd.concat(
                        [combined["score"].rename("score"), labels], axis=1
                    ).dropna()
                    if aligned.empty:
                        raise ValueError("ensemble predictions have no aligned labels")
                    daily_ic = aligned.groupby(level="datetime").apply(
                        lambda frame: frame["score"].corr(frame["label"]),
                        include_groups=False,
                    )
                    daily_rank_ic = aligned.groupby(level="datetime").apply(
                        lambda frame: frame["score"].corr(
                            frame["label"], method="spearman"
                        ),
                        include_groups=False,
                    )
                    with qlib_workflow_run(
                        run_kind="independent-model-ensemble",
                        run_id=f"{ensemble_id}-{profile_id}-seed-{seed}",
                        tracking_uri=qlib_workflow_tracking_uri(),
                        dataset_identity_sha256=identity,
                    ) as workflow:
                        recorder = workflow.get_recorder()
                        workflow.log_params(
                            {"ensemble_id": ensemble_id, "profile": profile_id, "seed": seed}
                        )
                        recorder.save_objects(
                            **{
                                "pred.pkl": combined[["score"]],
                                "label.pkl": labels.to_frame("label"),
                            }
                        )
                        record = PortAnaRecord(
                            recorder,
                            config={
                                "strategy": {
                                    "class": "GovernedDPlusOneTopkDropoutStrategy",
                                    "module_path": "quant_platform.qlib_research_strategy",
                                    "kwargs": {
                                        "signal": "<PRED>",
                                        "topk": int(manifest.get("topk", 50)),
                                        "n_drop": int(manifest.get("n_drop", 5)),
                                        "research_execution_cadence": execution_cadence,
                                    },
                                },
                                "backtest": {
                                    "start_time": periods["valid_start"],
                                    "end_time": periods["valid_end"],
                                    "account": float(manifest.get("account", 100_000_000)),
                                    "benchmark": str(
                                        manifest.get("benchmark") or "SH000300"
                                    ),
                                    "exchange_kwargs": {
                                        "freq": "day",
                                        "limit_threshold": 0.095,
                                        "deal_price": "close",
                                        "open_cost": float(
                                            manifest.get("open_cost", 0.0005)
                                        ),
                                        "close_cost": float(
                                            manifest.get("close_cost", 0.0015)
                                        ),
                                        "min_cost": float(manifest.get("min_cost", 5.0)),
                                    },
                                },
                            },
                            risk_analysis_freq="day",
                        )
                        generated = record.generate()
                        if not isinstance(generated, dict):
                            raise RuntimeError(
                                "Qlib ensemble portfolio record generation was skipped"
                            )
                        record.check(include_self=True, parents=False)
                        report = recorder.load_object(
                            "portfolio_analysis/report_normal_1day.pkl"
                        )
                        workflow_identity = workflow.identity_dict()
                    excess = report["return"] - report["bench"] - report["cost"]
                    risk = risk_analysis(excess, freq="day")["risk"]
                    metrics = {
                        "ic": _finite(daily_ic.mean()),
                        "icir": _finite(daily_ic.mean() / daily_ic.std()),
                        "rank_ic": _finite(daily_rank_ic.mean()),
                        "rank_icir": _finite(
                            daily_rank_ic.mean() / daily_rank_ic.std()
                        ),
                        "information_ratio": _finite(risk.get("information_ratio")),
                        "annualized_excess_return_with_cost": _finite(
                            risk.get("annualized_return")
                        ),
                        "max_drawdown": _finite(risk.get("max_drawdown")),
                        "total_cost": _finite(report["cost"].sum()),
                        "average_turnover": _finite(
                            report.get("turnover", pd.Series(dtype=float)).mean()
                        ),
                    }
                    metric_report = model_metric_report(metrics)
                    all_cells_passed = all_cells_passed and metric_report["gate_passed"]
                    report_path = cell_root / "portfolio_report.parquet"
                    report.to_parquet(report_path)
                    aligned.to_parquet(cell_root / "signals_and_labels.parquet")
                    current_pairwise: list[dict[str, Any]] = []
                    for left_index in range(len(member_frames)):
                        for right_index in range(left_index + 1, len(member_frames)):
                            correlation = daily_rank_correlation(
                                member_frames[left_index][1], member_frames[right_index][1]
                            )
                            left_id = member_frames[left_index][0]
                            right_id = member_frames[right_index][0]
                            expected_grid = expected_correlations.get(
                                (left_id, right_id)
                            ) or expected_correlations.get((right_id, left_id))
                            expected_cell = (
                                (expected_grid or {}).get("cells") or {}
                            ).get(f"{profile_id}:{seed}")
                            if (
                                not isinstance(expected_cell, dict)
                                or expected_cell.get("evidence_sha256")
                                != correlation.get("evidence_sha256")
                            ):
                                raise ValueError(
                                    "ensemble member prediction correlation changed "
                                    "after preregistration"
                                )
                            current_pairwise.append(
                                {
                                    "left_candidate_id": left_id,
                                    "right_candidate_id": right_id,
                                    "correlation": correlation,
                                }
                            )
                    seed_results[str(seed)] = {
                        "status": "passed",
                        "metric_report": metric_report,
                        "metrics": metrics,
                        "latest_prediction_date": periods["valid_end"],
                        "predictions_path": str(predictions_path),
                        "predictions_sha256": file_sha256(predictions_path),
                        "portfolio_report_path": str(report_path),
                        "portfolio_report_sha256": file_sha256(report_path),
                        "coverage": coverage,
                        "combination_evidence": combination,
                        "member_prediction_artifacts": member_artifacts,
                        "pairwise_prediction_correlations": current_pairwise,
                        "execution_environment_sha256": environment_sha,
                        "qlib_workflow": workflow_identity,
                        "portfolio_calendar_boundary": portfolio_calendar_boundary,
                        "final_oos_opened": False,
                    }
                    if profile_id == "recent_3y":
                        returns = (
                            pd.to_numeric(report["return"], errors="coerce")
                            - pd.to_numeric(report["bench"], errors="coerce")
                            - pd.to_numeric(report["cost"], errors="coerce")
                        ).rename(str(seed))
                        returns.index = pd.to_datetime(returns.index).tz_localize(None)
                        recent_returns.append(returns)
                evidence["profiles"][profile_id] = {
                    "periods": periods,
                    "seeds": seed_results,
                }
            return_frame = pd.concat(recent_returns, axis=1, join="inner").dropna()
            if return_frame.shape[1] != len(REQUIRED_MODEL_SEEDS) or len(return_frame) < 40:
                raise ValueError("ensemble recent-window return evidence is incomplete")
            trial_series.append(
                (
                    {
                        "name": ensemble_id,
                        "candidate_id": ensemble_id,
                        "kind": "model_ensemble",
                    },
                    return_frame.mean(axis=1),
                )
            )
            evaluations.append(
                {
                    "ensemble_id": ensemble_id,
                    "status": "pending_multiple_testing",
                    "all_metric_cells_passed": all_cells_passed,
                    "evidence": evidence,
                }
            )
        except Exception as exc:
            fatal_error = f"{ensemble_id}: {exc}"
            evaluations.append(
                {"ensemble_id": ensemble_id, "status": "failed", "error": str(exc)}
            )

    if fatal_error is not None:
        # A preregistered candidate with missing/corrupt evidence makes the
        # comparison family incomplete. Reject the whole batch rather than
        # quietly shrinking the multiple-testing denominator.
        reason = "pre-registered ensemble family is incomplete: " + fatal_error
        evaluations = [
            {
                "ensemble_id": str(item.get("ensemble_id") or ""),
                "status": "failed",
                "error": reason,
            }
            for item in evaluations
        ]
    elif evaluations:
        try:
            multiple = build_run_multiple_testing_evidence(
                research_run_id=f"ensemble-tournament:{manifest['tournament_id']}",
                trial_series=trial_series,
                output=artifact_root / "multiple-testing",
            )
            for item in evaluations:
                _finalize_ensemble_evaluation(item, multiple)
        except Exception as exc:
            reason = f"ensemble shared Holm/PBO gate failed: {exc}"
            evaluations = [
                {
                    "ensemble_id": str(item.get("ensemble_id") or ""),
                    "status": "failed",
                    "error": reason,
                }
                for item in evaluations
            ]
    result = {
        "status": "ok",
        "research_execution_cadence": execution_cadence,
        "research_execution_cadence_sha256": execution_cadence[
            "evidence_sha256"
        ],
        "evaluations": evaluations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
