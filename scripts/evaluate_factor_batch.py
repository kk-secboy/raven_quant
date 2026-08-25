#!/usr/bin/env python3
"""Independently evaluate RD-Agent factor values against a selected Qlib snapshot."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_data.qlib_builder import verify_qlib_output_manifest
from quant_platform.cost_model import CostScheduleBook
from quant_platform.factor_evaluator import evaluate_factor_values
from quant_platform.factor_recompute import (
    compare_submitted_values,
    execute_factor_code,
    normalize_factor_input,
    require_exact_factor_index,
    sha256_file,
    validate_factor_prefix_invariance,
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
    profiles = manifest.get("evaluation_profiles") or [
        {
            "id": "explicit",
            "label": "Explicit",
            "role": "primary",
            "weight": 1.0,
            "periods": periods,
        }
    ]
    if len(profiles) > 3:
        raise ValueError("factor evaluation supports at most three concurrent profiles")
    profile_periods = [item["periods"] for item in profiles]
    if any(
        item["test_start"] != periods["test_start"] or item["test_end"] != periods["test_end"]
        for item in profile_periods
    ):
        raise ValueError("all research profiles must share one frozen final OOS window")
    provider = Path(args.provider_uri)
    provenance = json.loads(
        (provider / "metadata" / "provenance.json").read_text(encoding="utf-8")
    )
    verify_qlib_output_manifest(provider, provenance)
    if provenance.get("dataset_identity_sha256") != manifest.get(
        "dataset_identity_sha256"
    ):
        raise ValueError("factor evaluation provider does not match the sealed dataset")
    qlib.init(provider_uri=args.provider_uri, region="cn")
    output = Path(args.output)
    recompute_root = output.parent / "recomputed"
    if recompute_root.exists():
        shutil.rmtree(recompute_root)
    recompute_root.mkdir(parents=True)
    factor_input = normalize_factor_input(
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
    valid_end = pd.Timestamp(periods["valid_end"]).tz_localize(None)
    test_start = pd.Timestamp(periods["test_start"]).tz_localize(None)
    input_dates = pd.DatetimeIndex(factor_input.index.get_level_values("datetime"))
    if input_dates.max() > valid_end or (input_dates >= test_start).any():
        raise ValueError("research factor input must not contain final OOS observations")
    input_path = recompute_root / "daily_pv.h5"
    factor_input.to_hdf(input_path, key="data", mode="w")
    input_sha256 = sha256_file(input_path)
    labels_by_horizon: dict[int, pd.DataFrame] = {}
    comparisons = []
    for path in manifest.get("comparison_values", []):
        comparison = _load_values(path)
        if isinstance(comparison.index, pd.MultiIndex) and "datetime" in comparison.index.names:
            comparison_dates = pd.to_datetime(
                comparison.index.get_level_values("datetime"), errors="coerce"
            )
            comparison = comparison.loc[comparison_dates <= valid_end]
        comparisons.append(comparison)
    evaluations = []
    for item in candidates:
        try:
            candidate_root = recompute_root / str(item["id"])
            recomputed, recompute_evidence = execute_factor_code(
                code_path=Path(item["code_path"]),
                input_path=input_path,
                workspace=candidate_root / "full",
                timeout_seconds=int(manifest.get("factor_recompute_timeout_seconds", 300)),
            )
            recomputed = require_exact_factor_index(
                recomputed,
                factor_input,
                context=f"research candidate {item['id']}",
            )
            pit_evidence = validate_factor_prefix_invariance(
                code_path=Path(item["code_path"]),
                input_path=input_path,
                full_values=recomputed,
                workspace_root=candidate_root / "prefix-checks",
                timeout_seconds=int(manifest.get("factor_recompute_timeout_seconds", 300)),
                cutpoint_count=int(manifest.get("factor_pit_cutpoint_count", 3)),
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
                    "pit_invariance": pit_evidence,
                    "research_data_boundary": {
                        "latest_input_date": input_dates.max().date().isoformat(),
                        "valid_end": valid_end.date().isoformat(),
                        "test_start": test_start.date().isoformat(),
                        "final_oos_observations_exposed": False,
                    },
                }
            )
            label_horizon_days = int(item["label_horizon_days"])
            recompute_evidence["label_horizon_days"] = label_horizon_days
            if label_horizon_days not in labels_by_horizon:
                labels_by_horizon[label_horizon_days] = D.features(
                    D.instruments(str(manifest.get("universe") or "cn_all")),
                    [f"Ref($close, -{label_horizon_days + 1})/Ref($close, -1)-1"],
                    start_time=min(value["valid_start"] for value in profile_periods),
                    end_time=max(value["valid_end"] for value in profile_periods),
                    freq="day",
                )

            def evaluate_profile(
                profile: dict[str, Any],
                recomputed_values: pd.DataFrame = recomputed,
                horizon: int = label_horizon_days,
                base_evidence: dict[str, Any] = recompute_evidence,
                candidate: dict[str, Any] = item,
                candidate_path: Path = recomputed_path,
            ) -> dict[str, Any]:
                current = profile["periods"]
                metrics = evaluate_factor_values(
                    recomputed_values,
                    labels_by_horizon[horizon],
                    valid_start=pd.Timestamp(current["valid_start"]).date(),
                    valid_end=pd.Timestamp(current["valid_end"]).date(),
                    test_start=pd.Timestamp(current["test_start"]).date(),
                    test_end=pd.Timestamp(current["test_end"]).date(),
                    comparison_values=comparisons,
                    cost_schedule=CostScheduleBook.from_mapping(manifest.get("cost_model")),
                    reference_order_value=float(manifest["cost_reference_order_value"]),
                    min_daily_instruments=int(manifest.get("min_daily_instruments", 50)),
                    label_horizon_days=horizon,
                )
                profile_identity = {
                    key: profile[key] for key in ("id", "label", "role", "weight") if key in profile
                }
                metrics["research_profile"] = profile_identity
                evidence = copy.deepcopy(base_evidence)
                evidence["periods"] = current
                return {
                    "candidate_id": candidate["id"],
                    "status": "ok",
                    "periods": current,
                    "metrics": metrics,
                    "recomputed_values_path": str(candidate_path),
                    "recomputed_values_sha256": sha256_file(candidate_path),
                    "recompute_evidence": evidence,
                    "experiment_family_id": candidate["experiment_family_id"],
                    "experiment_count": int(candidate["experiment_count"]),
                }

            def evaluate_profile_safely(
                profile: dict[str, Any], candidate_id: str = str(item["id"])
            ) -> dict[str, Any]:
                try:
                    return evaluate_profile(profile)
                except Exception as exc:
                    return {
                        "candidate_id": candidate_id,
                        "status": "failed",
                        "periods": profile["periods"],
                        "error": str(exc),
                    }

            with ThreadPoolExecutor(max_workers=len(profiles)) as executor:
                evaluations.extend(executor.map(evaluate_profile_safely, profiles))
        except Exception as exc:
            evaluations.extend(
                {
                    "candidate_id": item["id"],
                    "status": "failed",
                    "periods": profile["periods"],
                    "error": str(exc),
                }
                for profile in profiles
            )
    families: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for evaluation in evaluations:
        family = str(evaluation.get("experiment_family_id") or "")
        if family:
            profile_id = str(
                (evaluation.get("metrics") or {}).get("research_profile", {}).get("id") or ""
            )
            families.setdefault((profile_id, family), []).append(evaluation)
    for family in families.values():
        declared = max(int(item.get("experiment_count") or len(family)) for item in family)
        p_values = [
            float(
                1.0
                if (p_value := (item.get("metrics") or {}).get("hac_p_value")) is None
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
                "profile_count": len(profiles),
                "train_start": periods["train_start"],
                "valid_end": periods["valid_end"],
                "test_end": periods["test_end"],
            }
        )
        workflow.log_metrics(
            {
                "candidate_count": len(evaluations),
                "profile_count": len(profiles),
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
