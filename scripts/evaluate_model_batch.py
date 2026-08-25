from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from quant_data.qlib_builder import verify_qlib_output_manifest
from quant_platform.feature_set_registry import get_feature_set
from quant_platform.model_recompute import execute_model_candidate
from quant_platform.model_research_governance import (
    MODEL_RESEARCH_CONTRACT_VERSION,
    REQUIRED_MODEL_SEEDS,
    build_run_multiple_testing_evidence,
    canonical_sha256,
    file_sha256,
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
    if {str(item.get("id")) for item in profiles} != {
        "recent_3y",
        "balanced_5y",
        "robust_10y",
    }:
        raise ValueError("model evaluation requires the three governed profiles")
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
    feature_set = get_feature_set(str(manifest.get("feature_set_id") or "governed-baseline"))
    runner_path = Path(__file__).resolve().with_name("model_sandbox_runner.py")
    evaluations: list[dict[str, Any]] = []
    for candidate in manifest.get("candidates") or []:
        candidate_id = str(candidate["id"])
        evidence: dict[str, Any] = {
            "contract_version": MODEL_RESEARCH_CONTRACT_VERSION,
            "source": "independent_qlib_recompute",
            "candidate_id": candidate_id,
            "dataset_identity_sha256": dataset_identity,
            "feature_set_definition_sha256": feature_set["definition_sha256"],
            "final_oos_opened": False,
            "profiles": {},
        }
        execution_environments: set[str] = set()
        failure: str | None = None
        for profile in profiles:
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
                        manifest={
                            "candidate_id": candidate_id,
                            "code_sha256": candidate["code_sha256"],
                            "model_type": candidate["model_type"],
                            "training_hyperparameters": candidate.get(
                                "training_hyperparameters"
                            )
                            or {},
                            "feature_set": feature_set,
                            "periods": periods,
                            "seed": seed,
                            "dataset_identity_sha256": dataset_identity,
                            "universe": manifest.get("universe", "cn_all"),
                            "benchmark": manifest.get("benchmark", "SH000300"),
                            "account": manifest.get("account", 100_000_000),
                            "topk": manifest.get("topk", 50),
                            "n_drop": manifest.get("n_drop", 5),
                            "open_cost": manifest.get("open_cost", 0.0005),
                            "close_cost": manifest.get("close_cost", 0.0015),
                            "min_cost": manifest.get("min_cost", 5.0),
                            "final_oos_opened": False,
                        },
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
                        "checkpoint_path": str(workspace / "output" / "checkpoint.pt"),
                        "checkpoint_sha256": result["checkpoint_sha256"],
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
                    }
                    execution_environments.add(
                        str(execution_evidence["execution_environment_sha256"])
                    )
                except Exception as exc:
                    seed_results[str(seed)] = {"status": "failed", "error": str(exc)}
                    failure = failure or f"{profile_id}/seed-{seed}: {exc}"
            evidence["profiles"][profile_id] = {
                "periods": periods,
                "seeds": seed_results,
            }
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
    try:
        if len(evaluations) != len(manifest.get("candidates") or []) or any(
            item.get("status") != "passed" for item in evaluations
        ):
            raise ValueError(
                "the pre-registered model run has incomplete candidate returns"
            )
        trial_series: list[tuple[dict[str, str], pd.Series]] = []
        for item in evaluations:
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
        multiple = build_run_multiple_testing_evidence(
            research_run_id=str(manifest["research_run_id"]),
            trial_series=trial_series,
            output=artifact_root / "run-level-multiple-testing",
        )
        for item in evaluations:
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
    result = {"status": "ok", "evaluations": evaluations}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
