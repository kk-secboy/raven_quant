#!/usr/bin/env python3
"""Run a resumable in-sample/out-of-sample grid on the production Qlib backtester."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_data.qlib_builder import verify_qlib_output_manifest
from quant_platform.parameter_experiments import (
    evaluate_trial,
    portfolio_trial_comparability_evidence,
    summarize_trials,
)
from quant_platform.qlib_workflow import (
    qlib_workflow_run,
    require_qlib_workflow_identity,
)
from quant_platform.statistical_validation import deflated_sharpe_probability
from quant_platform.strategy_health_reference import STRATEGY_HEALTH_REFERENCE_NAME
from quant_platform.strategy_research_evaluation import (
    STRATEGY_RESEARCH_EVALUATION_MODES,
)
from quant_platform.transparent_baseline_runner import (
    TRANSPARENT_BASELINE_JOB_RUNNER_FIELD,
    TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
)

PRE_FINAL_PORTFOLIO_TRIAL_MODE = "pre_final_portfolio_trial"
PRE_FINAL_EVALUATION_MODES = frozenset(
    {PRE_FINAL_PORTFOLIO_TRIAL_MODE, *STRATEGY_RESEARCH_EVALUATION_MODES}
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_completed_result(
    result_path: Path,
    *,
    config: dict[str, Any],
    periods: dict[str, str],
    evaluation_mode: str | None = None,
) -> dict[str, Any] | None:
    if not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = result["metrics"]
        provenance = metrics["provenance"]
        if result.get("periods") != periods:
            return None
        if provenance.get("strategy_config_sha256") != _canonical_sha256(config):
            return None
        if evaluation_mode in PRE_FINAL_EVALUATION_MODES and (
            provenance.get("evaluation_mode") != evaluation_mode
            or provenance.get("evaluation_scope") != "pre_final_only"
            or provenance.get("final_oos_opened") is not False
            or "strategy_health_reference" in provenance
            or "strategy_health_reference" in (result.get("artifacts") or {})
            or (result_path.parent / STRATEGY_HEALTH_REFERENCE_NAME).exists()
            or (result_path.parent / STRATEGY_HEALTH_REFERENCE_NAME).is_symlink()
        ):
            return None
        if (
            metrics.get("backtest_engine") != "qlib"
            or metrics.get("qlib_native_backtest") is not True
        ):
            return None
        require_qlib_workflow_identity(result.get("qlib_workflow"))
        return result
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _run_segment(
    *,
    backtest_script: Path,
    provider_uri: str,
    execution_provider_uri: str | None,
    execution_frequency: str | None,
    base_manifest: dict[str, Any],
    config: dict[str, Any],
    periods: dict[str, str],
    output: Path,
    tracking_uri: str,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    evaluation_mode = base_manifest.get("evaluation_mode")
    completed = _read_completed_result(
        result_path,
        config=config,
        periods=periods,
        evaluation_mode=(str(evaluation_mode) if evaluation_mode else None),
    )
    if completed is not None:
        return completed
    manifest = _build_segment_manifest(
        base_manifest=base_manifest,
        config=config,
        periods=periods,
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # The child keeps writing after its output manifest is sealed. Keep this
    # mutable stream outside that sealed tree; existing in-tree logs stay intact.
    log_path = output.parent / f"{output.name}-backtest.log"
    command = [
        sys.executable,
        str(backtest_script),
        "--provider-uri",
        provider_uri,
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
        "--tracking-uri",
        tracking_uri,
    ]
    if execution_provider_uri and execution_frequency:
        command.extend(
            [
                "--execution-provider-uri",
                execution_provider_uri,
                "--execution-frequency",
                execution_frequency,
            ]
        )
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        raise RuntimeError("; ".join(lines[-8:]) or f"backtest exited {process.returncode}")
    completed = _read_completed_result(
        result_path,
        config=config,
        periods=periods,
        evaluation_mode=(str(evaluation_mode) if evaluation_mode else None),
    )
    if completed is None:
        raise RuntimeError("backtest result failed provenance validation")
    return completed


def _build_segment_manifest(
    *,
    base_manifest: dict[str, Any],
    config: dict[str, Any],
    periods: dict[str, str],
) -> dict[str, Any]:
    """Project one outer experiment into the exact child backtest contract."""

    manifest = {
        "strategy_version_id": base_manifest["strategy_version_id"],
        "dataset": base_manifest["dataset"],
        "benchmark": base_manifest["benchmark"],
        "universe": base_manifest.get("universe", "cn_all"),
        "execution_dataset": base_manifest.get("execution_dataset"),
        "evaluation_mode": base_manifest.get("evaluation_mode"),
        "pre_final_cutoff": base_manifest.get("pre_final_cutoff"),
        "historical_validation_periods": base_manifest.get(
            "historical_validation_periods"
        ),
        "strategy_trial_count": base_manifest.get("strategy_trial_count"),
        "shared_multiple_testing": base_manifest.get("shared_multiple_testing"),
        "model_signal": base_manifest.get("model_signal"),
        "model_formal_admission": base_manifest.get("model_formal_admission"),
        "model_candidate": base_manifest.get("model_candidate"),
        "model_bundle_factors": base_manifest.get("model_bundle_factors") or [],
        "periods": periods,
        "config": config,
        "factors": base_manifest["factors"],
    }
    for identity_field in (
        TRANSPARENT_BASELINE_JOB_RUNNER_FIELD,
        TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD,
        TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
    ):
        if base_manifest.get(identity_field) is not None:
            manifest[identity_field] = base_manifest[identity_field]
    return manifest


def _progress_payload(
    trial_results: list[dict[str, Any]], *, trial_count: int
) -> dict[str, Any]:
    completed = [
        {
            "trial_index": trial["trial_index"],
            "status": trial["status"],
            "score": trial["score"],
            "warnings": trial["warnings"],
            "error": trial["error"],
        }
        for trial in trial_results
    ]
    return {
        "completed_count": len(completed),
        "trial_count": trial_count,
        "succeeded_count": sum(item["status"] == "succeeded" for item in completed),
        "failed_count": sum(item["status"] == "failed" for item in completed),
        "trials": completed,
    }


def _finalize_cross_trial_dsr(
    trial_results: list[dict[str, Any]],
    out_of_sample_returns: dict[int, pd.Series],
    *,
    trial_count: int,
    prior_trial_sharpes: list[float] | None = None,
) -> bool:
    successful_trials = [
        item for item in trial_results if item.get("status") == "succeeded"
    ]
    if len(successful_trials) != trial_count:
        return False
    current_trial_sharpes = [
        float(item["metrics"]["out_of_sample"]["deflated_sharpe"]["daily_sharpe"])
        for item in successful_trials
    ]
    prior_trial_sharpes = list(prior_trial_sharpes or [])
    trial_sharpes = [*prior_trial_sharpes, *current_trial_sharpes]
    governed_trial_count = len(prior_trial_sharpes) + trial_count
    for item in successful_trials:
        trial_index = int(item["trial_index"])
        out_of_sample = item["metrics"]["out_of_sample"]
        dsr = deflated_sharpe_probability(
            out_of_sample_returns[trial_index],
            trials=governed_trial_count,
            trial_sharpes=trial_sharpes,
        )
        out_of_sample["deflated_sharpe"] = dsr
        out_of_sample["deflated_sharpe_probability"] = dsr["probability"]
        item["score"], item["warnings"] = evaluate_trial(
            item["metrics"]["in_sample"], out_of_sample
        )
    return True


def _admitted_trial_sharpes(manifest: dict[str, Any]) -> list[float]:
    evidence = manifest.get("shared_multiple_testing")
    if evidence is None:
        return []
    if not isinstance(evidence, dict):
        raise ValueError("shared multiple-testing evidence is invalid")
    names = evidence.get("trial_names")
    sharpes = evidence.get("trial_daily_sharpes")
    trial_count = evidence.get("trial_count")
    if (
        evidence.get("final_oos_opened") is not False
        or not isinstance(names, list)
        or not isinstance(sharpes, list)
        or int(trial_count or 0) != len(names)
        or len(sharpes) != len(names)
        or len({str(item) for item in names}) != len(names)
    ):
        raise ValueError("shared multiple-testing evidence is incomplete")
    normalized = [float(value) for value in sharpes]
    if any(
        not pd.notna(value) or value == float("inf") or value == float("-inf")
        for value in normalized
    ):
        raise ValueError("shared multiple-testing Sharpe distribution is invalid")
    return normalized


def _validate_portfolio_trial_comparability(
    manifest: dict[str, Any], trial_results: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if manifest.get("evaluation_mode") != PRE_FINAL_PORTFOLIO_TRIAL_MODE:
        return None
    specs = {int(item["trial_index"]): item for item in manifest.get("trials") or []}
    combined = []
    for result in trial_results:
        trial_index = int(result["trial_index"])
        spec = specs.get(trial_index)
        if spec is None:
            raise ValueError("portfolio result contains an unregistered trial")
        combined.append({**result, "config": spec["config"]})
    return portfolio_trial_comparability_evidence(combined)


def _trial_execution_error(trial_results: list[dict[str, Any]]) -> str | None:
    failed = [item for item in trial_results if item.get("status") == "failed"]
    if not failed:
        return None
    details = []
    for item in failed:
        trial_index = int(item.get("trial_index", -1))
        error = str(item.get("error") or "trial failed without an error").strip()
        details.append(f"trial {trial_index}: {error}")
    prefix = (
        f"{len(failed)} of {len(trial_results)} parameter trials failed during execution: "
    )
    return (prefix + "; ".join(details))[:8000]


def _build_terminal_result(
    *,
    manifest: dict[str, Any],
    evaluation_mode: str | None,
    trial_results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    execution_error = _trial_execution_error(trial_results)
    execution_failed_count = sum(
        item.get("status") == "failed" for item in trial_results
    )
    summary["execution_succeeded_count"] = len(trial_results) - execution_failed_count
    summary["execution_failed_count"] = execution_failed_count
    result = {
        "status": "failed" if execution_error else "ok",
        "experiment_id": manifest["experiment_id"],
        "strategy_version_id": manifest["strategy_version_id"],
        "dataset": manifest["dataset"],
        "evaluation_mode": evaluation_mode,
        "final_oos_opened": False,
        "periods": manifest["periods"],
        "trials": trial_results,
        "summary": summary,
    }
    if execution_error:
        result["failure_kind"] = "trial_execution_error"
        result["error"] = execution_error
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--execution-provider-uri")
    parser.add_argument("--execution-frequency")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tracking-uri", required=True)
    args = parser.parse_args()
    manifest: dict[str, Any] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    evaluation_mode = manifest.get("evaluation_mode")
    if (
        manifest.get("model_signal") is not None
        and evaluation_mode != PRE_FINAL_PORTFOLIO_TRIAL_MODE
    ):
        raise ValueError("model parameter experiments must be explicitly pre-final only")
    if evaluation_mode in PRE_FINAL_EVALUATION_MODES:
        cutoff = str(manifest.get("pre_final_cutoff") or "")
        governance = (manifest.get("periods") or {}).get("governance") or {}
        expected_governance_mode = (
            "model_portfolio_pre_final"
            if evaluation_mode == PRE_FINAL_PORTFOLIO_TRIAL_MODE
            else evaluation_mode
        )
        if (
            not cutoff
            or governance.get("mode") != expected_governance_mode
            or governance.get("final_oos_opened") is not False
            or any(
                str((manifest.get("periods") or {}).get(segment, {}).get("end") or "")
                > cutoff
                for segment in ("in_sample", "out_of_sample")
            )
        ):
            raise ValueError("model parameter experiment crosses the pre-final cutoff")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = Path(args.provider_uri) / "metadata" / "provenance.json"
    if not provenance_path.is_file():
        raise ValueError("parameter experiment requires Qlib dataset provenance")
    provider_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    verify_qlib_output_manifest(Path(args.provider_uri), provider_provenance)
    import qlib

    qlib.init(provider_uri=args.provider_uri, region="cn")
    backtest_script = Path(__file__).with_name("run_multifactor_backtest.py")
    trial_results: list[dict[str, Any]] = []
    out_of_sample_returns: dict[int, pd.Series] = {}
    prior_trial_sharpes = _admitted_trial_sharpes(manifest)
    governed_trial_count = len(prior_trial_sharpes) + len(manifest["trials"])
    if int(manifest.get("strategy_trial_count") or governed_trial_count) != governed_trial_count:
        raise ValueError("parameter experiment trial count omits prior admitted trials")
    for trial in manifest["trials"]:
        trial_index = int(trial["trial_index"])
        trial_output = output / f"trial-{trial_index:03d}"
        item: dict[str, Any] = {
            "trial_index": trial_index,
            "parameters": trial["parameters"],
            "status": "failed",
            "score": None,
            "metrics": None,
            "warnings": [],
            "error": None,
        }
        try:
            segment_metrics: dict[str, dict[str, Any]] = {}
            for segment in ("in_sample", "out_of_sample"):
                result = _run_segment(
                    backtest_script=backtest_script,
                    provider_uri=args.provider_uri,
                    execution_provider_uri=args.execution_provider_uri,
                    execution_frequency=args.execution_frequency,
                    base_manifest=manifest,
                    config=trial["config"],
                    periods=manifest["periods"][segment],
                    output=trial_output / segment,
                    tracking_uri=args.tracking_uri,
                )
                segment_metrics[segment] = result["metrics"]
                if segment == "out_of_sample":
                    daily = pd.read_parquet(trial_output / segment / "daily_returns.parquet")
                    returns = pd.to_numeric(daily["return"], errors="coerce") - pd.to_numeric(
                        daily.get("cost", 0.0), errors="coerce"
                    )
                    out_of_sample_returns[trial_index] = returns
                    dsr = deflated_sharpe_probability(
                        returns, trials=governed_trial_count
                    )
                    segment_metrics[segment]["deflated_sharpe"] = dsr
                    segment_metrics[segment]["deflated_sharpe_probability"] = dsr[
                        "probability"
                    ]
            score, warnings = evaluate_trial(
                segment_metrics["in_sample"], segment_metrics["out_of_sample"]
            )
            item.update(
                status="succeeded",
                score=score,
                metrics=segment_metrics,
                warnings=warnings,
            )
        except Exception as exc:
            item["error"] = str(exc)[-4000:]
        trial_results.append(item)
        (output / "progress.json").write_text(
            json.dumps(
                _progress_payload(
                    trial_results, trial_count=len(manifest["trials"])
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    _finalize_cross_trial_dsr(
        trial_results,
        out_of_sample_returns,
        trial_count=len(manifest["trials"]),
        prior_trial_sharpes=prior_trial_sharpes,
    )
    execution_error = _trial_execution_error(trial_results)
    comparability = (
        None
        if execution_error
        else _validate_portfolio_trial_comparability(manifest, trial_results)
    )
    # The last in-loop progress snapshot predates the cross-trial DSR. Rewrite
    # it so the live UI and the final result expose the same scores/warnings.
    (output / "progress.json").write_text(
        json.dumps(
            _progress_payload(trial_results, trial_count=len(manifest["trials"])),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = summarize_trials(trial_results, manifest["parameter_grid"])
    summary["prior_admitted_trial_count"] = len(prior_trial_sharpes)
    summary["governed_trial_count"] = governed_trial_count
    summary["final_oos_opened"] = False
    if comparability is not None:
        summary["comparability"] = comparability
    result = _build_terminal_result(
        manifest=manifest,
        evaluation_mode=(str(evaluation_mode) if evaluation_mode else None),
        trial_results=trial_results,
        summary=summary,
    )
    result_path = output / "result.json"
    # Publish the complete terminal trial ledger before optional tracking.  A
    # tracking failure must not erase the actual per-trial execution errors.
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with qlib_workflow_run(
        run_kind="portfolio-experiment",
        run_id=str(manifest["experiment_id"]),
        tracking_uri=args.tracking_uri,
        dataset_identity_sha256=provider_provenance.get("dataset_identity_sha256"),
    ) as workflow:
        workflow.log_params(
            {
                "experiment_id": manifest["experiment_id"],
                "strategy_version_id": manifest["strategy_version_id"],
                "dataset": manifest["dataset"],
                "benchmark": manifest["benchmark"],
                "trial_count": len(manifest["trials"]),
                "governed_trial_count": governed_trial_count,
                "parameter_grid_sha256": _canonical_sha256(manifest["parameter_grid"]),
                "evaluation_mode": evaluation_mode,
                "final_oos_opened": False,
            }
        )
        workflow.log_metrics(
            {
                "trial_count": len(trial_results),
                "succeeded_count": summary["succeeded_count"],
                "failed_count": summary["failed_count"],
                "best_trial_index": summary.get("best_trial_index"),
                "best_score": (
                    summary["leaderboard"][0]["score"] if summary.get("leaderboard") else None
                ),
            }
        )
        result["qlib_workflow"] = workflow.identity_dict()
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        workflow.save_artifacts(output)
        print(
            json.dumps(
                {"status": result["status"], "summary": summary},
                ensure_ascii=False,
            )
        )
        if execution_error:
            raise RuntimeError(execution_error)


if __name__ == "__main__":
    main()
