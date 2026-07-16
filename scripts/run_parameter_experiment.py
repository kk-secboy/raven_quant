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

from quant_platform.parameter_experiments import evaluate_trial, summarize_trials
from quant_platform.statistical_validation import deflated_sharpe_probability


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_completed_result(
    result_path: Path, *, config: dict[str, Any], periods: dict[str, str]
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
        if (
            metrics.get("backtest_engine") != "qlib"
            or metrics.get("qlib_native_backtest") is not True
        ):
            return None
        return result
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _run_segment(
    *,
    backtest_script: Path,
    provider_uri: str,
    base_manifest: dict[str, Any],
    config: dict[str, Any],
    periods: dict[str, str],
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    completed = _read_completed_result(result_path, config=config, periods=periods)
    if completed is not None:
        return completed
    manifest = {
        "strategy_version_id": base_manifest["strategy_version_id"],
        "dataset": base_manifest["dataset"],
        "benchmark": base_manifest["benchmark"],
        "periods": periods,
        "config": config,
        "factors": base_manifest["factors"],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = output / "backtest.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            [
                sys.executable,
                str(backtest_script),
                "--provider-uri",
                provider_uri,
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        raise RuntimeError("; ".join(lines[-8:]) or f"backtest exited {process.returncode}")
    completed = _read_completed_result(result_path, config=config, periods=periods)
    if completed is None:
        raise RuntimeError("backtest result failed provenance validation")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest: dict[str, Any] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    backtest_script = Path(__file__).with_name("run_multifactor_backtest.py")
    trial_results: list[dict[str, Any]] = []
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
                    base_manifest=manifest,
                    config=trial["config"],
                    periods=manifest["periods"][segment],
                    output=trial_output / segment,
                )
                segment_metrics[segment] = result["metrics"]
                if segment == "out_of_sample":
                    daily = pd.read_parquet(trial_output / segment / "daily_returns.parquet")
                    returns = pd.to_numeric(daily["return"], errors="coerce") - pd.to_numeric(
                        daily.get("cost", 0.0), errors="coerce"
                    )
                    dsr = deflated_sharpe_probability(
                        returns, trials=len(manifest["trials"])
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
        (output / "progress.json").write_text(
            json.dumps(
                {
                    "completed_count": len(completed),
                    "trial_count": len(manifest["trials"]),
                    "succeeded_count": sum(item["status"] == "succeeded" for item in completed),
                    "failed_count": sum(item["status"] == "failed" for item in completed),
                    "trials": completed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    summary = summarize_trials(trial_results, manifest["parameter_grid"])
    if not summary["succeeded_count"]:
        raise RuntimeError("every parameter trial failed; inspect trial backtest logs")
    result = {
        "status": "ok",
        "experiment_id": manifest["experiment_id"],
        "strategy_version_id": manifest["strategy_version_id"],
        "dataset": manifest["dataset"],
        "periods": manifest["periods"],
        "trials": trial_results,
        "summary": summary,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
