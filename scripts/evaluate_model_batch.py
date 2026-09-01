from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from quant_data.qlib_builder import verify_qlib_output_manifest
from quant_platform.feature_set_registry import resolve_feature_set
from quant_platform.model_recompute import (
    ModelResourceLimitError,
    execute_model_candidate,
    governed_checkpoint_filename,
)
from quant_platform.model_research_governance import (
    MODEL_RESEARCH_CONTRACT_VERSION,
    REQUIRED_MODEL_SEEDS,
    build_run_multiple_testing_evidence,
    canonical_sha256,
    file_sha256,
    require_model_metric_gate,
    validate_independent_model_evidence,
    verify_model_prediction_artifact,
)
from quant_platform.rdagent_dataset_view import prepare_rdagent_dataset_view


def calendar_between(provider: Path, start: str, end: str) -> list[str]:
    calendar = [
        value.strip()
        for value in (provider / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    selected = [value for value in calendar if start <= value <= end]
    if not selected or selected[0] != start or selected[-1] != end:
        raise ValueError("model validation window does not match the Qlib trading calendar")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    provider = Path(args.provider_uri).resolve()
    manifest: dict[str, Any] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    provenance = json.loads(
        (provider / "metadata" / "provenance.json").read_text(encoding="utf-8")
    )
    verify_qlib_output_manifest(provider, provenance)
    dataset_identity = str(manifest.get("dataset_identity_sha256") or "")
    if dataset_identity != provenance.get("dataset_identity_sha256"):
        raise ValueError("model evaluation provider does not match the sealed dataset")
    profiles = manifest.get("evaluation_profiles") or []
    evaluation_stage = str(manifest.get("evaluation_stage") or "model_full")
    expected_profiles = (
        {"recent_3y"}
        if evaluation_stage == "feature_screen"
        else {"recent_3y", "balanced_5y", "robust_10y"}
    )
    if evaluation_stage not in {"feature_screen", "model_full"} or {
        str(item.get("id")) for item in profiles
    } != expected_profiles:
        raise ValueError("model evaluation profiles do not match its tournament stage")
    if evaluation_stage == "feature_screen":
        tournament_id = str(manifest.get("research_tournament_id") or "")
        candidate_ids = {
            str(item.get("id") or "") for item in manifest.get("candidates") or []
        }
        bindings = manifest.get("candidate_bindings") or []
        bound_candidate_ids = {
            str(item.get("candidate_id") or "")
            for item in bindings
            if isinstance(item, dict)
        }
        trial_ids = {
            str(item.get("trial_id") or "")
            for item in bindings
            if isinstance(item, dict)
        }
        if (
            not tournament_id
            or not candidate_ids
            or "" in candidate_ids
            or len(bindings) != len(candidate_ids)
            or bound_candidate_ids != candidate_ids
            or len(trial_ids) != len(candidate_ids)
            or "" in trial_ids
        ):
            raise ValueError("feature-screen tournament identity is missing or incomplete")
    valid_ends = {str(item["periods"]["valid_end"]) for item in profiles}
    test_windows = {
        (str(item["periods"]["test_start"]), str(item["periods"]["test_end"]))
        for item in profiles
    }
    if len(valid_ends) != 1 or len(test_windows) != 1:
        raise ValueError("model profiles must share one pre-final cutoff and final OOS")
    pre_final_end = next(iter(valid_ends))
    output = Path(args.output).resolve()
    artifact_root = output.parent / "independent-model-evaluations"
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True)
    view = prepare_rdagent_dataset_view(
        provider,
        output.parent / "model-dataset-view",
        cutoff=pre_final_end,
    )
    feature_set_id = str(manifest.get("feature_set_id") or "governed-baseline")
    feature_set = resolve_feature_set(feature_set_id, manifest.get("feature_set"))
    runner_path = Path(__file__).resolve().with_name("model_sandbox_runner.py")
    evaluations: list[dict[str, Any]] = []
    for candidate in manifest.get("candidates") or []:
        candidate_id = str(candidate["id"])

        def execution_manifest(
            periods: dict[str, Any],
            seed: int,
            *,
            resource_stage: str,
            _candidate: dict[str, Any] = candidate,
            _candidate_id: str = candidate_id,
        ) -> dict[str, Any]:
            return {
                "candidate_id": _candidate_id,
                "code_sha256": _candidate["code_sha256"],
                "model_type": _candidate["model_type"],
                "model_engine": str(
                    _candidate.get("model_engine") or "rdagent_pytorch"
                ),
                "training_hyperparameters": _candidate.get("training_hyperparameters") or {},
                "resource_stage": resource_stage,
                "feature_set": feature_set,
                "periods": periods,
                "seed": seed,
                "dataset_identity_sha256": dataset_identity,
                "research_window_contract": manifest.get("research_window_contract"),
                "research_window_contract_sha256": manifest.get(
                    "research_window_contract_sha256"
                ),
                "label_horizon_sessions": _candidate.get("label_horizon_sessions")
                or manifest.get("label_horizon_sessions"),
                "universe": manifest.get("universe", "cn_all"),
                "benchmark": manifest.get("benchmark", "SH000300"),
                "account": manifest.get("account", 100_000_000),
                "topk": manifest.get("topk", 50),
                "n_drop": manifest.get("n_drop", 5),
                "open_cost": manifest.get("open_cost", 0.0005),
                "close_cost": manifest.get("close_cost", 0.0015),
                "min_cost": manifest.get("min_cost", 5.0),
                "final_oos_opened": False,
            }

        evidence: dict[str, Any] = {
            "contract_version": MODEL_RESEARCH_CONTRACT_VERSION,
            "source": "independent_qlib_recompute",
            "candidate_id": candidate_id,
            "dataset_identity_sha256": dataset_identity,
            "feature_set_definition_sha256": feature_set["definition_sha256"],
            "final_oos_opened": False,
            "profiles": {},
        }
        if evaluation_stage == "feature_screen":
            profile = dict(profiles[0])
            screen_periods = dict(profile["periods"])
            screen_seed = int(REQUIRED_MODEL_SEEDS[0])
            workspace = artifact_root / candidate_id / "feature-screen"
            try:
                screen_result, screen_execution = execute_model_candidate(
                    code_path=Path(candidate["code_path"]),
                    provider_path=view,
                    manifest=execution_manifest(
                        screen_periods,
                        screen_seed,
                        resource_stage="screening",
                    ),
                    workspace=workspace,
                    runner_path=runner_path,
                    timeout_seconds=int(manifest.get("model_timeout_seconds", 7200)),
                )
                require_model_metric_gate(
                    screen_result["metrics"],
                    context=f"feature screen {candidate_id}",
                )
                predictions_path = workspace / "output" / "predictions.parquet"
                coverage = verify_model_prediction_artifact(
                    predictions_path,
                    expected_sha256=screen_result["predictions_sha256"],
                    test_start=screen_periods["valid_start"],
                    test_end=screen_periods["valid_end"],
                    trading_days=calendar_between(
                        view,
                        screen_periods["valid_start"],
                        screen_periods["valid_end"],
                    ),
                )
                cell = {
                    "profile_id": "recent_3y",
                    "seed": screen_seed,
                    "gate_status": "passed",
                    "metrics": screen_result["metrics"],
                    "periods": screen_periods,
                    "predictions_path": str(predictions_path),
                    "predictions_sha256": screen_result["predictions_sha256"],
                    "checkpoint_path": str(
                        workspace
                        / "output"
                        / governed_checkpoint_filename(
                            str(candidate.get("model_engine") or "rdagent_pytorch")
                        )
                    ),
                    "checkpoint_sha256": screen_result["checkpoint_sha256"],
                    "checkpoint_format": screen_result["checkpoint_format"],
                    "model_engine": str(
                        candidate.get("model_engine") or "rdagent_pytorch"
                    ),
                    "portfolio_report_path": str(
                        workspace / "output" / "portfolio_report.parquet"
                    ),
                    "portfolio_report_sha256": screen_result[
                        "portfolio_report_sha256"
                    ],
                    "coverage": coverage,
                    "resource_policy": screen_result["resource_policy"],
                    "model_label_contract": screen_result["model_label_contract"],
                    "model_label_contract_sha256": screen_result[
                        "model_label_contract_sha256"
                    ],
                    "execution_evidence_sha256": screen_execution["evidence_sha256"],
                    "execution_environment_sha256": screen_execution[
                        "execution_environment_sha256"
                    ],
                }
                screen_evidence = {
                    "contract_version": "model-feature-screen-v1",
                    "source": "independent_qlib_recompute",
                    "candidate_id": candidate_id,
                    "dataset_identity_sha256": dataset_identity,
                    "feature_set_definition_sha256": feature_set[
                        "definition_sha256"
                    ],
                    "evaluation_stage": "feature_screen",
                    "selection_profile": "recent_3y",
                    "selection_seed": screen_seed,
                    "final_oos_opened": False,
                    "cells": [cell],
                }
                screen_evidence["evidence_sha256"] = canonical_sha256(
                    screen_evidence
                )
                evaluations.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "passed",
                        "evidence": screen_evidence,
                        "evidence_sha256": screen_evidence["evidence_sha256"],
                    }
                )
            except ModelResourceLimitError as exc:
                evaluations.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "resource_blocked",
                        "reason_code": "feature_screen_resource_limit",
                        "error": str(exc),
                    }
                )
            except Exception as exc:
                evaluations.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "failed",
                        "error": f"feature screen failed: {exc}",
                    }
                )
            continue
        screening_profile = next(
            item for item in profiles if str(item.get("id")) == "balanced_5y"
        )
        screening_periods = dict(screening_profile["periods"])
        screening_seed = int(REQUIRED_MODEL_SEEDS[0])
        screening_workspace = artifact_root / candidate_id / "resource-screen"
        try:
            screening_result, screening_execution = execute_model_candidate(
                code_path=Path(candidate["code_path"]),
                provider_path=view,
                manifest=execution_manifest(
                    screening_periods,
                    screening_seed,
                    resource_stage="screening",
                ),
                workspace=screening_workspace,
                runner_path=runner_path,
                timeout_seconds=int(manifest.get("model_timeout_seconds", 7200)),
            )
        except ModelResourceLimitError as exc:
            evaluations.append(
                {
                    "candidate_id": candidate_id,
                    "status": "resource_blocked",
                    "reason_code": "screening_resource_limit",
                    "error": str(exc),
                    "evidence": {
                        **evidence,
                        "resource_screen": {
                            "status": "resource_blocked",
                            "profile_id": "balanced_5y",
                            "seed": screening_seed,
                            "periods": screening_periods,
                            "date_segments_modified": False,
                            "universe_modified": False,
                        },
                    },
                }
            )
            continue
        except Exception as exc:
            evaluations.append(
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "error": f"resource feasibility screen failed: {exc}",
                }
            )
            continue
        evidence["resource_screen"] = {
            "status": "passed",
            "profile_id": "balanced_5y",
            "seed": screening_seed,
            "periods": screening_periods,
            "metrics": screening_result["metrics"],
            "resource_policy": screening_result["resource_policy"],
            "model_label_contract": screening_result["model_label_contract"],
            "model_label_contract_sha256": screening_result[
                "model_label_contract_sha256"
            ],
            "execution_evidence_sha256": screening_execution["evidence_sha256"],
            "execution_environment_sha256": screening_execution[
                "execution_environment_sha256"
            ],
            "date_segments_modified": False,
            "universe_modified": False,
        }
        execution_environments: set[str] = set()
        failure: str | None = None
        resource_block: str | None = None
        for profile in profiles:
            if resource_block is not None:
                break
            profile_id = str(profile["id"])
            periods = dict(profile["periods"])
            seed_results: dict[str, Any] = {}
            days = calendar_between(view, periods["valid_start"], periods["valid_end"])
            for seed in REQUIRED_MODEL_SEEDS:
                workspace = artifact_root / candidate_id / profile_id / f"seed-{seed}"
                try:
                    result, execution_evidence = execute_model_candidate(
                        code_path=Path(candidate["code_path"]),
                        provider_path=view,
                        manifest=execution_manifest(
                            periods, seed, resource_stage="full_validation"
                        ),
                        workspace=workspace,
                        runner_path=runner_path,
                        timeout_seconds=int(manifest.get("model_timeout_seconds", 7200)),
                    )
                    predictions_path = workspace / "output" / "predictions.parquet"
                    coverage = verify_model_prediction_artifact(
                        predictions_path,
                        expected_sha256=result["predictions_sha256"],
                        test_start=periods["valid_start"],
                        test_end=periods["valid_end"],
                        trading_days=days,
                    )
                    seed_results[str(seed)] = {
                        "status": "passed",
                        "metrics": result["metrics"],
                        "latest_prediction_date": result["latest_prediction_date"],
                        "predictions_path": str(predictions_path),
                        "predictions_sha256": result["predictions_sha256"],
                        "checkpoint_path": str(
                            workspace
                            / "output"
                            / governed_checkpoint_filename(
                                str(candidate.get("model_engine") or "rdagent_pytorch")
                            )
                        ),
                        "checkpoint_sha256": result["checkpoint_sha256"],
                        "checkpoint_format": result["checkpoint_format"],
                        "model_engine": str(
                            candidate.get("model_engine") or "rdagent_pytorch"
                        ),
                        "portfolio_report_path": str(
                            workspace / "output" / "portfolio_report.parquet"
                        ),
                        "portfolio_report_sha256": result[
                            "portfolio_report_sha256"
                        ],
                        "execution_evidence": execution_evidence,
                        "execution_evidence_sha256": execution_evidence["evidence_sha256"],
                        "execution_environment_sha256": execution_evidence[
                            "execution_environment_sha256"
                        ],
                        "coverage": coverage,
                        "resource_policy": result["resource_policy"],
                        "model_label_contract": result["model_label_contract"],
                        "model_label_contract_sha256": result[
                            "model_label_contract_sha256"
                        ],
                    }
                    execution_environments.add(
                        str(execution_evidence["execution_environment_sha256"])
                    )
                except ModelResourceLimitError as exc:
                    seed_results[str(seed)] = {
                        "status": "resource_blocked",
                        "error": str(exc),
                    }
                    resource_block = f"{profile_id}/seed-{seed}: {exc}"
                    break
                except Exception as exc:
                    seed_results[str(seed)] = {"status": "failed", "error": str(exc)}
                    failure = failure or f"{profile_id}/seed-{seed}: {exc}"
            evidence["profiles"][profile_id] = {
                "periods": periods,
                "seeds": seed_results,
            }
        if resource_block is not None:
            evaluations.append(
                {
                    "candidate_id": candidate_id,
                    "status": "resource_blocked",
                    "reason_code": "full_validation_resource_limit",
                    "error": resource_block,
                    "evidence": evidence,
                }
            )
            continue
        if failure is None:
            if len(execution_environments) != 1:
                raise ValueError("model evaluation cells used inconsistent environments")
            evidence["execution_environment_sha256"] = next(
                iter(execution_environments)
            )
            evaluations.append(
                {
                    "candidate_id": candidate_id,
                    "status": "passed",
                    "evidence": evidence,
                }
            )
        else:
            evaluations.append(
                {"candidate_id": candidate_id, "status": "failed", "error": failure}
            )
    if evaluation_stage == "feature_screen":
        candidate_bindings = manifest.get("candidate_bindings") or []
        result = {
            "status": "ok",
            "evaluation_stage": evaluation_stage,
            "research_tournament_id": str(
                manifest.get("research_tournament_id") or ""
            ),
            "candidate_bindings_sha256": canonical_sha256(candidate_bindings),
            "evaluations": evaluations,
            "resource_blocked_count": sum(
                item.get("status") == "resource_blocked" for item in evaluations
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False))
        return
    try:
        expected_candidates = manifest.get("candidates") or []
        if len(evaluations) != len(expected_candidates):
            raise ValueError(
                "the pre-registered model run has an incomplete candidate count: "
                f"expected {len(expected_candidates)}, received {len(evaluations)}"
            )
        failed_evaluations = [
            item for item in evaluations if item.get("status") == "failed"
        ]
        if failed_evaluations:
            failure_summary = "; ".join(
                f"{item.get('candidate_id')}: "
                f"{item.get('error', 'model evaluation failed')}"
                for item in failed_evaluations
            )[:3000]
            raise ValueError(
                "pre-registered model candidates failed before Holm/PBO: "
                + failure_summary
            )
        trial_series: list[tuple[dict[str, str], pd.Series]] = []
        passed_evaluations = [
            item for item in evaluations if item.get("status") == "passed"
        ]
        for item in passed_evaluations:
            evidence = item["evidence"]
            seed_returns: list[pd.Series] = []
            seeds = evidence["profiles"]["recent_3y"]["seeds"]
            for seed in REQUIRED_MODEL_SEEDS:
                seed_result = seeds[str(seed)]
                report_path = Path(str(seed_result["portfolio_report_path"]))
                if (
                    not report_path.is_file()
                    or file_sha256(report_path)
                    != seed_result["portfolio_report_sha256"]
                ):
                    raise ValueError("model run-level portfolio report changed")
                report = pd.read_parquet(report_path)
                if not {"return", "bench", "cost"}.issubset(report.columns):
                    raise ValueError("model run-level portfolio report is incomplete")
                series = (
                    pd.to_numeric(report["return"], errors="coerce")
                    - pd.to_numeric(report["bench"], errors="coerce")
                    - pd.to_numeric(report["cost"], errors="coerce")
                ).rename(str(seed))
                series.index = pd.to_datetime(series.index).tz_localize(None)
                seed_returns.append(series)
            frame = pd.concat(seed_returns, axis=1, join="inner").dropna()
            if frame.shape[1] != len(REQUIRED_MODEL_SEEDS) or len(frame) < 40:
                raise ValueError("model run-level seed return matrix is incomplete")
            trial_series.append(
                (
                    {
                        "name": str(item["candidate_id"]),
                        "candidate_id": str(item["candidate_id"]),
                        "kind": "model",
                    },
                    frame.mean(axis=1),
                )
            )
        if passed_evaluations:
            multiple = build_run_multiple_testing_evidence(
                research_run_id=str(manifest["research_run_id"]),
                trial_series=trial_series,
                output=artifact_root / "run-level-multiple-testing",
            )
            for item in passed_evaluations:
                evidence = item["evidence"]
                evidence["multiple_testing"] = multiple
                evidence["evidence_sha256"] = canonical_sha256(evidence)
                validate_independent_model_evidence(
                    evidence,
                    candidate_id=str(item["candidate_id"]),
                    dataset_identity_sha256=dataset_identity,
                    pre_final_end=pre_final_end,
                )
                item["evidence_sha256"] = evidence["evidence_sha256"]
    except Exception as exc:
        error = f"run-level model Holm/PBO gate failed: {exc}"
        evaluations = [
            {
                "candidate_id": str(candidate["id"]),
                "status": "failed",
                "error": error,
            }
            for candidate in manifest.get("candidates") or []
        ]
    result = {
        "status": "ok",
        "evaluations": evaluations,
        "resource_blocked_count": sum(
            item.get("status") == "resource_blocked" for item in evaluations
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
