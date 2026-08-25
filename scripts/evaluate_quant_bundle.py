from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from quant_data.qlib_builder import verify_qlib_output_manifest
from quant_platform.factor_recompute import (
    compare_submitted_values,
    execute_factor_code,
    normalize_factor_input,
    require_exact_factor_index,
    sha256_file,
    validate_factor_prefix_invariance,
)
from quant_platform.feature_set_registry import get_feature_set
from quant_platform.model_recompute import execute_model_candidate
from quant_platform.model_research_governance import (
    QUANT_BUNDLE_CONTRACT_VERSION,
    REQUIRED_MODEL_SEEDS,
    REQUIRED_QUANT_ABLATIONS,
    build_run_multiple_testing_evidence,
    canonical_sha256,
    file_sha256,
    validate_quant_bundle_evidence,
    verify_model_prediction_artifact,
)
from quant_platform.rdagent_dataset_view import prepare_rdagent_dataset_view


def _calendar_between(provider: Path, start: str, end: str) -> list[str]:
    calendar = [
        value.strip()
        for value in (provider / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    selected = [value for value in calendar if start <= value <= end]
    if not selected or selected[0] != start or selected[-1] != end:
        raise ValueError("quant ablation window does not match the Qlib trading calendar")
    return selected


def _freeze_factor_values(
    *,
    bundle: dict[str, Any],
    view: Path,
    periods: dict[str, str],
    output: Path,
    universe: str,
) -> tuple[Path, list[dict[str, Any]]]:
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(view), region="cn")
    output.mkdir(parents=True, exist_ok=True)
    factor_input = normalize_factor_input(
        D.features(
            D.instruments(universe),
            ["$open", "$close", "$high", "$low", "$volume", "$factor"],
            start_time=periods["train_start"],
            end_time=periods["valid_end"],
            freq="day",
        )
        .swaplevel()
        .sort_index()
    )
    input_path = output / "daily_pv.h5"
    factor_input.to_hdf(input_path, key="data", mode="w")
    values: list[pd.Series] = []
    evidence: list[dict[str, Any]] = []
    for index, factor in enumerate(bundle["factors"], start=1):
        factor_id = str(factor["candidate_id"])
        root = output / "factors" / f"{index:03d}-{factor_id}"
        recomputed, execution = execute_factor_code(
            code_path=Path(factor["code_path"]),
            input_path=input_path,
            workspace=root / "full",
            timeout_seconds=300,
        )
        recomputed = require_exact_factor_index(
            recomputed,
            factor_input,
            context=f"quant bundle factor {factor_id}",
        )
        pit = validate_factor_prefix_invariance(
            code_path=Path(factor["code_path"]),
            input_path=input_path,
            full_values=recomputed,
            workspace_root=root / "prefix-checks",
            cutpoint_count=3,
        )
        submitted_path = factor.get("submitted_values_path")
        submitted = compare_submitted_values(
            Path(submitted_path) if submitted_path else None,
            recomputed,
        )
        if submitted_path and not submitted.get("exact_match"):
            raise ValueError(f"quant factor {factor_id} submitted values do not recompute")
        values.append(recomputed.iloc[:, 0].rename(f"factor_{index:03d}"))
        item = {
            "candidate_id": factor_id,
            "code_sha256": factor["code_sha256"],
            "execution": execution,
            "pit_invariance": pit,
            "submitted_comparison": submitted,
        }
        item["evidence_sha256"] = canonical_sha256(item)
        evidence.append(item)
    frame = pd.concat(values, axis=1, join="inner").sort_index()
    if not frame.index.equals(factor_input.index):
        raise ValueError("quant bundle factor combination changed the immutable input index")
    path = output / "combined_factors.parquet"
    frame.to_parquet(path)
    return path, evidence


def _run_ablation(
    *,
    name: str,
    bundle: dict[str, Any],
    feature_set: dict[str, Any],
    profiles: list[dict[str, Any]],
    view: Path,
    factor_values_path: Path,
    output: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    use_factors = name in {"factor_only", "joint"}
    use_candidate_model = name in {"model_only", "joint"}
    model = bundle["model"]
    code_path = (
        Path(model["code_path"])
        if use_candidate_model
        else Path(__file__).resolve().with_name("baseline_model_stub.py")
    )
    code_sha256 = file_sha256(code_path)
    if use_candidate_model and code_sha256 != model["code_sha256"]:
        raise ValueError("quant bundle model code changed after proposal")
    result: dict[str, Any] = {
        "status": "passed",
        "source": "independent_qlib_recompute",
        "experiment_family_id": bundle["experiment_family_id"],
        "dataset_identity_sha256": manifest["dataset_identity_sha256"],
        "final_oos_opened": False,
        "profiles": {},
    }
    execution_environments: set[str] = set()
    for profile in profiles:
        periods = dict(profile["periods"])
        seed_results: dict[str, Any] = {}
        days = _calendar_between(view, periods["valid_start"], periods["valid_end"])
        for seed in REQUIRED_MODEL_SEEDS:
            workspace = output / "ablations" / name / str(profile["id"]) / f"seed-{seed}"
            model_result, execution = execute_model_candidate(
                code_path=code_path,
                provider_path=view,
                additional_factors_path=factor_values_path if use_factors else None,
                manifest={
                    "candidate_id": f"{bundle['id']}-{name}",
                    "code_sha256": code_sha256,
                    "model_type": (
                        model.get("model_type", "Tabular") if use_candidate_model else "Tabular"
                    ),
                    "model_engine": (
                        "rdagent_pytorch" if use_candidate_model else "lightgbm_baseline"
                    ),
                    "training_hyperparameters": model.get("training_hyperparameters") or {},
                    "feature_set": feature_set,
                    "additional_factor_count": len(bundle["factors"]) if use_factors else 0,
                    "periods": periods,
                    "seed": seed,
                    "dataset_identity_sha256": manifest["dataset_identity_sha256"],
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
                runner_path=Path(__file__).resolve().with_name("model_sandbox_runner.py"),
                timeout_seconds=int(manifest.get("model_timeout_seconds", 7200)),
            )
            predictions_path = workspace / "output" / "predictions.parquet"
            coverage = verify_model_prediction_artifact(
                predictions_path,
                expected_sha256=model_result["predictions_sha256"],
                test_start=periods["valid_start"],
                test_end=periods["valid_end"],
                trading_days=days,
            )
            seed_results[str(seed)] = {
                "status": "passed",
                "metrics": model_result["metrics"],
                "latest_prediction_date": model_result["latest_prediction_date"],
                "predictions_path": str(predictions_path),
                "predictions_sha256": model_result["predictions_sha256"],
                "checkpoint_path": str(workspace / "output" / "checkpoint.pt"),
                "checkpoint_sha256": model_result["checkpoint_sha256"],
                "execution_evidence_sha256": execution["evidence_sha256"],
                "execution_environment_sha256": execution[
                    "execution_environment_sha256"
                ],
                "coverage": coverage,
                "portfolio_report_path": str(workspace / "output" / "portfolio_report.parquet"),
                "portfolio_report_sha256": file_sha256(
                    workspace / "output" / "portfolio_report.parquet"
                ),
            }
            execution_environments.add(str(execution["execution_environment_sha256"]))
        result["profiles"][str(profile["id"])] = {
            "periods": periods,
            "seeds": seed_results,
        }
    if len(execution_environments) != 1:
        raise ValueError(f"quant ablation {name} used inconsistent execution environments")
    result["execution_environment_sha256"] = next(iter(execution_environments))
    result["evidence_sha256"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    provider = Path(args.provider_uri).resolve()
    manifest: dict[str, Any] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    provenance = json.loads((provider / "metadata" / "provenance.json").read_text(encoding="utf-8"))
    verify_qlib_output_manifest(provider, provenance)
    if provenance.get("dataset_identity_sha256") != manifest.get("dataset_identity_sha256"):
        raise ValueError("quant evaluation provider does not match the sealed dataset")
    profiles = manifest.get("evaluation_profiles") or []
    if {str(item.get("id")) for item in profiles} != {
        "recent_3y",
        "balanced_5y",
        "robust_10y",
    }:
        raise ValueError("quant evaluation requires the three governed profiles")
    valid_ends = {str(item["periods"]["valid_end"]) for item in profiles}
    if len(valid_ends) != 1:
        raise ValueError("quant profiles must share one pre-final cutoff")
    output = Path(args.output).resolve()
    root = output.parent / "independent-quant-evaluations"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    view = prepare_rdagent_dataset_view(
        provider,
        output.parent / "quant-dataset-view",
        cutoff=next(iter(valid_ends)),
    )
    evaluations: list[dict[str, Any]] = []
    for candidate in manifest.get("candidates") or []:
        candidate_root = root / str(candidate["id"])
        try:
            feature_set = get_feature_set(
                str(
                    candidate.get("feature_set_id")
                    or manifest.get("feature_set_id")
                    or "governed-baseline"
                )
            )
            factor_periods = {
                "train_start": min(
                    str(profile["periods"]["train_start"]) for profile in profiles
                ),
                "valid_end": max(
                    str(profile["periods"]["valid_end"]) for profile in profiles
                ),
            }
            factor_values, factor_evidence = _freeze_factor_values(
                bundle=candidate,
                view=view,
                periods=factor_periods,
                output=candidate_root,
                universe=str(manifest.get("universe") or "cn_all"),
            )
            ablations = {
                name: _run_ablation(
                    name=name,
                    bundle=candidate,
                    feature_set=feature_set,
                    profiles=profiles,
                    view=view,
                    factor_values_path=factor_values,
                    output=candidate_root,
                    manifest=manifest,
                )
                for name in REQUIRED_QUANT_ABLATIONS
            }
            execution_environments = {
                str(item["execution_environment_sha256"])
                for item in ablations.values()
            }
            if len(execution_environments) != 1:
                raise ValueError("quant ablations used inconsistent execution environments")
            evidence = {
                "contract_version": QUANT_BUNDLE_CONTRACT_VERSION,
                "id": candidate["id"],
                "dataset_identity_sha256": manifest["dataset_identity_sha256"],
                "feature_set_definition_sha256": feature_set["definition_sha256"],
                "experiment_family_id": candidate["experiment_family_id"],
                "factors": [
                    {
                        "candidate_id": factor["candidate_id"],
                        "code_sha256": factor["code_sha256"],
                    }
                    for factor in candidate["factors"]
                ],
                "factor_recompute_evidence": factor_evidence,
                "combined_factor_values_sha256": sha256_file(factor_values),
                "model": {
                    "code_sha256": candidate["model"]["code_sha256"],
                    "recipe_sha256": candidate["model"]["recipe_sha256"],
                },
                "ablations": ablations,
                "execution_environment_sha256": next(
                    iter(execution_environments)
                ),
                "final_oos_opened": False,
            }
            evaluations.append(
                {
                    "candidate_id": candidate["id"],
                    "status": "passed",
                    "evidence": evidence,
                }
            )
        except Exception as exc:
            evaluations.append(
                {"candidate_id": candidate.get("id"), "status": "failed", "error": str(exc)}
            )
    try:
        if len(evaluations) != len(manifest.get("candidates") or []) or any(
            item.get("status") != "passed" for item in evaluations
        ):
            raise ValueError(
                "the pre-registered quant run has incomplete bundle/ablation returns"
            )
        trial_series: list[tuple[dict[str, str], pd.Series]] = []
        for item in evaluations:
            candidate_id = str(item["candidate_id"])
            for ablation in REQUIRED_QUANT_ABLATIONS:
                seed_returns: list[pd.Series] = []
                seeds = item["evidence"]["ablations"][ablation]["profiles"][
                    "recent_3y"
                ]["seeds"]
                for seed in REQUIRED_MODEL_SEEDS:
                    seed_result = seeds[str(seed)]
                    report_path = Path(str(seed_result["portfolio_report_path"]))
                    if (
                        not report_path.is_file()
                        or file_sha256(report_path)
                        != seed_result["portfolio_report_sha256"]
                    ):
                        raise ValueError("quant run-level portfolio report changed")
                    report = pd.read_parquet(report_path)
                    if not {"return", "bench", "cost"}.issubset(report.columns):
                        raise ValueError("quant run-level portfolio report is incomplete")
                    series = (
                        pd.to_numeric(report["return"], errors="coerce")
                        - pd.to_numeric(report["bench"], errors="coerce")
                        - pd.to_numeric(report["cost"], errors="coerce")
                    ).rename(str(seed))
                    series.index = pd.to_datetime(series.index).tz_localize(None)
                    seed_returns.append(series)
                frame = pd.concat(seed_returns, axis=1, join="inner").dropna()
                if frame.shape[1] != len(REQUIRED_MODEL_SEEDS) or len(frame) < 40:
                    raise ValueError("quant run-level seed return matrix is incomplete")
                trial_series.append(
                    (
                        {
                            "name": f"{candidate_id}:{ablation}",
                            "candidate_id": candidate_id,
                            "kind": "quant_bundle",
                            "ablation": ablation,
                        },
                        frame.mean(axis=1),
                    )
                )
        multiple = build_run_multiple_testing_evidence(
            research_run_id=str(manifest["research_run_id"]),
            trial_series=trial_series,
            output=root / "run-level-multiple-testing",
        )
        for item in evaluations:
            evidence = item["evidence"]
            evidence["multiple_testing"] = multiple
            evidence["bundle_sha256"] = canonical_sha256(evidence)
            validate_quant_bundle_evidence(
                evidence,
                dataset_identity_sha256=manifest["dataset_identity_sha256"],
            )
            item["evidence_sha256"] = evidence["bundle_sha256"]
    except Exception as exc:
        error = f"run-level quant Holm/PBO gate failed: {exc}"
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
