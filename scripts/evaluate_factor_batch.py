#!/usr/bin/env python3
"""Independently evaluate RD-Agent factor values against a selected Qlib snapshot."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_platform.cost_model import CostScheduleBook
from quant_platform.factor_evaluator import evaluate_factor_values
from quant_platform.factor_recompute import (
    compare_submitted_values,
    execute_factor_code,
    sha256_file,
)
from quant_platform.qlib_workflow import qlib_workflow_run
from quant_platform.statistical_validation import benjamini_hochberg


def _load_values(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        return pd.read_hdf(source)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"unsupported factor values format: {source.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tracking-uri", required=True)
    args = parser.parse_args()

    import qlib
    from qlib.data import D

    manifest: dict[str, Any] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    candidates = manifest["candidates"]
    periods = manifest["periods"]
    qlib.init(provider_uri=args.provider_uri, region="cn")
    output = Path(args.output)
    recompute_root = output.parent / "recomputed"
    if recompute_root.exists():
        shutil.rmtree(recompute_root)
    recompute_root.mkdir(parents=True)
    factor_input = (
        D.features(
            D.instruments(str(manifest.get("universe") or "cn_all")),
            ["$open", "$close", "$high", "$low", "$volume", "$factor"],
            start_time=periods["train_start"],
            end_time=periods["valid_end"],
            freq="day",
        )
        .swaplevel()
        .sort_index()
    )
    input_path = recompute_root / "daily_pv.h5"
    factor_input.to_hdf(input_path, key="data", mode="w")
    input_sha256 = sha256_file(input_path)
    labels_by_horizon: dict[int, pd.DataFrame] = {}
    comparisons = [_load_values(path) for path in manifest.get("comparison_values", [])]
    evaluations = []
    for item in candidates:
        try:
            candidate_root = recompute_root / str(item["id"])
            recomputed, recompute_evidence = execute_factor_code(
                code_path=Path(item["code_path"]),
                input_path=input_path,
                workspace=candidate_root,
                timeout_seconds=int(manifest.get("factor_recompute_timeout_seconds", 300)),
            )
            recomputed_path = candidate_root / "recomputed.h5"
            recomputed.to_hdf(recomputed_path, key="data", mode="w")
            submitted_comparison = compare_submitted_values(
                Path(item["submitted_values_path"]) if item.get("submitted_values_path") else None,
                recomputed,
            )
            if not submitted_comparison.get("exact_match"):
                raise ValueError("submitted result.h5 does not match independent recomputation")
            recompute_evidence.update(
                {
                    "dataset_identity_sha256": manifest["dataset_identity_sha256"],
                    "provider_input_sha256": input_sha256,
                    "periods": periods,
                    "submitted_comparison": submitted_comparison,
                    "authoritative_values_sha256": sha256_file(recomputed_path),
                }
            )
            label_horizon_days = int(item["label_horizon_days"])
            recompute_evidence["label_horizon_days"] = label_horizon_days
            if label_horizon_days not in labels_by_horizon:
                labels_by_horizon[label_horizon_days] = D.features(
                    D.instruments(str(manifest.get("universe") or "cn_all")),
                    [
                        f"Ref($close, -{label_horizon_days + 1})/"
                        "Ref($close, -1)-1"
                    ],
                    start_time=periods["valid_start"],
                    end_time=periods["valid_end"],
                    freq="day",
                )
            metrics = evaluate_factor_values(
                recomputed,
                labels_by_horizon[label_horizon_days],
                valid_start=pd.Timestamp(periods["valid_start"]).date(),
                valid_end=pd.Timestamp(periods["valid_end"]).date(),
                test_start=pd.Timestamp(periods["test_start"]).date(),
                test_end=pd.Timestamp(periods["test_end"]).date(),
                comparison_values=comparisons,
                cost_schedule=CostScheduleBook.from_mapping(manifest.get("cost_model")),
                reference_order_value=float(manifest["cost_reference_order_value"]),
                min_daily_instruments=int(manifest.get("min_daily_instruments", 50)),
                label_horizon_days=label_horizon_days,
            )
            evaluations.append(
                {
                    "candidate_id": item["id"],
                    "status": "ok",
                    "metrics": metrics,
                    "recomputed_values_path": str(recomputed_path),
                    "recomputed_values_sha256": sha256_file(recomputed_path),
                    "recompute_evidence": recompute_evidence,
                    "experiment_family_id": item["experiment_family_id"],
                    "experiment_count": int(item["experiment_count"]),
                }
            )
        except Exception as exc:
            evaluations.append({"candidate_id": item["id"], "status": "failed", "error": str(exc)})
    families: dict[str, list[dict[str, Any]]] = {}
    for evaluation in evaluations:
        family = str(evaluation.get("experiment_family_id") or "")
        if family:
            families.setdefault(family, []).append(evaluation)
    for family in families.values():
        declared = max(int(item.get("experiment_count") or len(family)) for item in family)
        p_values = [
            float(
                1.0 if (p_value := (item.get("metrics") or {}).get("hac_p_value")) is None
                else p_value
            )
            for item in family
        ]
        p_values.extend([1.0] * max(0, declared - len(p_values)))
        q_values = benjamini_hochberg(p_values)
        for item, q_value in zip(family, q_values, strict=False):
            if item.get("status") == "ok":
                item["metrics"]["bh_q_value"] = q_value
                item["metrics"]["experiment_count"] = declared
    result = {"status": "ok", "evaluations": evaluations}
    output.parent.mkdir(parents=True, exist_ok=True)
    with qlib_workflow_run(
        run_kind="factor-evaluation",
        run_id=str(manifest.get("research_run_id") or output.parent.name),
        tracking_uri=args.tracking_uri,
        dataset_identity_sha256=str(manifest["dataset_identity_sha256"]),
    ) as workflow:
        workflow.log_params(
            {
                "research_run_id": manifest.get("research_run_id") or output.parent.name,
                "universe": manifest.get("universe") or "cn_all",
                "candidate_count": len(candidates),
                "train_start": periods["train_start"],
                "valid_end": periods["valid_end"],
                "test_end": periods["test_end"],
            }
        )
        workflow.log_metrics(
            {
                "candidate_count": len(evaluations),
                "succeeded_count": sum(item.get("status") == "ok" for item in evaluations),
                "failed_count": sum(item.get("status") != "ok" for item in evaluations),
            }
        )
        result["qlib_workflow"] = workflow.identity_dict()
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        workflow.save_artifacts(output.parent)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
