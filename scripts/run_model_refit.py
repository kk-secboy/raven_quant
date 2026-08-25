#!/usr/bin/env python3
"""Produce one governed live prediction table from a frozen model StrategySpec.

This is deliberately separate from formal OOS validation.  It rolls only the
training/validation dates forward, keeps the approved code/recipe/features
unchanged, and never reads labels after the signal date.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_data.execution_contract import require_daily_qlib_contract
from quant_data.qlib_builder import verify_qlib_output_manifest
from quant_platform.factor_recompute import (
    execute_factor_code,
    normalize_factor_input,
    require_exact_factor_index,
    require_exact_oos_coverage,
    sha256_file,
    validate_factor_prefix_invariance,
)
from quant_platform.model_recompute import execute_model_candidate
from quant_platform.model_research_governance import (
    MODEL_REFIT_POLICY,
    MODEL_REFIT_POLICY_SHA256,
    canonical_sha256,
    normalize_model_predictions,
    verify_model_prediction_artifact,
)

REFIT_CONTRACT_VERSION = "model-live-refit-v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("model refit manifest must be an object")
    return value


def _calendar(
    provider: Path,
    signal_date: str,
    policy: dict[str, Any],
) -> tuple[list[str], dict[str, str]]:
    days = [
        line.strip()
        for line in (provider / "calendars" / "day.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and line.strip() <= signal_date
    ]
    train_days = int(policy["train_trading_days"])
    train_validation_purge_days = int(
        policy["train_validation_purge_trading_days"]
    )
    validation_days = int(policy["validation_trading_days"])
    embargo_days = int(policy["embargo_trading_days"])
    prediction_days = int(policy["prediction_trading_days"])
    required = (
        train_days
        + train_validation_purge_days
        + validation_days
        + embargo_days
        + prediction_days
    )
    if len(days) < required or days[-1] != signal_date:
        raise ValueError(
            "model refit history does not satisfy the frozen model training policy"
        )
    selected = days[-required:]
    train_end = train_days - 1
    valid_start = train_days + train_validation_purge_days
    valid_end = valid_start + validation_days - 1
    test_index = valid_end + embargo_days + 1
    periods = {
        "train_start": selected[0],
        "train_end": selected[train_end],
        "valid_start": selected[valid_start],
        "valid_end": selected[valid_end],
        "test_start": selected[test_index],
        "test_end": selected[test_index],
    }
    if periods["test_start"] != signal_date:
        raise ValueError("model refit calendar did not resolve the requested signal date")
    return days, periods


def _recompute_bundle_factors(
    *,
    provider: Path,
    output: Path,
    factors: list[dict[str, Any]],
    periods: dict[str, str],
    dataset_identity_sha256: str,
    universe: str,
) -> tuple[Path | None, list[dict[str, Any]]]:
    if not factors:
        return None, []
    from qlib.data import D

    factor_input = normalize_factor_input(
        D.features(
            D.instruments(universe),
            ["$open", "$close", "$high", "$low", "$volume", "$factor"],
            start_time=periods["train_start"],
            end_time=periods["test_end"],
            freq="day",
        )
        .swaplevel()
        .sort_index()
    )
    factor_root = output / "bundle-factors"
    factor_root.mkdir(parents=True, exist_ok=False)
    input_path = factor_root / "daily_pv.h5"
    factor_input.to_hdf(input_path, key="data", mode="w")
    input_sha256 = sha256_file(input_path)
    values: list[pd.Series] = []
    evidence: list[dict[str, Any]] = []
    for index, item in enumerate(factors):
        candidate_id = str(item.get("candidate_id") or "")
        code_path = Path(str(item.get("code_path") or "")).resolve()
        code_sha256 = str(item.get("code_sha256") or "").lower()
        if not candidate_id or not code_path.is_file() or sha256_file(code_path) != code_sha256:
            raise ValueError(f"model refit bundle factor {candidate_id!r} is missing or changed")
        workspace = factor_root / f"factor-{index:03d}"
        frame, execution = execute_factor_code(
            code_path=code_path,
            input_path=input_path,
            workspace=workspace / "full",
            timeout_seconds=300,
        )
        frame = require_exact_factor_index(
            frame, factor_input, context=f"live bundle factor {candidate_id}"
        )
        pit = validate_factor_prefix_invariance(
            code_path=code_path,
            input_path=input_path,
            full_values=frame,
            workspace_root=workspace / "prefix-checks",
            timeout_seconds=300,
            cutpoint_count=3,
        )
        coverage = require_exact_oos_coverage(
            frame,
            factor_input,
            test_start=periods["test_start"],
            test_end=periods["test_end"],
            trading_days=[periods["test_start"]],
            context=f"live bundle factor {candidate_id}",
            min_daily_finite=50,
        )
        values.append(frame.iloc[:, 0].rename(f"factor_{index:03d}"))
        evidence.append(
            {
                "candidate_id": candidate_id,
                "code_sha256": code_sha256,
                "provider_input_sha256": input_sha256,
                "dataset_identity_sha256": dataset_identity_sha256,
                "execution": execution,
                "pit_invariance": pit,
                "coverage": coverage,
            }
        )
    combined = pd.concat(values, axis=1, join="inner").sort_index()
    if not combined.index.equals(factor_input.index):
        raise ValueError("live bundle factors changed the exact governed Qlib index")
    path = factor_root / "additional_factors.parquet"
    combined.to_parquet(path, compression="zstd")
    return path, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    provider = Path(args.provider_uri).resolve()
    manifest = _load_json(Path(args.manifest).resolve())
    output = Path(args.output).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)
    if manifest.get("contract_version") != REFIT_CONTRACT_VERSION:
        raise ValueError("model refit contract is invalid")
    refit_policy = manifest.get("refit_policy")
    if (
        refit_policy != MODEL_REFIT_POLICY
        or manifest.get("refit_policy_sha256") != MODEL_REFIT_POLICY_SHA256
        or canonical_sha256(refit_policy) != manifest.get("refit_policy_sha256")
    ):
        raise ValueError("model refit training policy is missing or changed")

    provenance = _load_json(provider / "metadata" / "provenance.json")
    require_daily_qlib_contract(provenance)
    verify_qlib_output_manifest(provider, provenance)
    dataset_identity = str(manifest.get("dataset_identity_sha256") or "")
    if dataset_identity != provenance.get("dataset_identity_sha256"):
        raise ValueError("model refit provider does not match the frozen strategy dataset")
    if str(manifest.get("dataset_lineage_id") or "") != str(
        provenance.get("dataset_lineage_id") or ""
    ):
        raise ValueError("model refit provider uses another governed dataset lineage")

    signal_date = str(manifest.get("signal_date") or "")
    _, periods = _calendar(provider, signal_date, dict(refit_policy))
    feature_set = manifest.get("feature_set")
    model = manifest.get("model")
    if not isinstance(feature_set, dict) or not isinstance(model, dict):
        raise ValueError("model refit is missing its frozen model or feature set")
    if feature_set.get("definition_sha256") != manifest.get(
        "feature_set_definition_sha256"
    ):
        raise ValueError("model refit feature-set definition is inconsistent")

    import qlib

    qlib.init(provider_uri=str(provider), region="cn")
    bundle_factors = manifest.get("bundle_factors") or []
    if not isinstance(bundle_factors, list) or any(
        not isinstance(item, dict) for item in bundle_factors
    ):
        raise ValueError("model refit bundle factors are invalid")
    additional_path, factor_evidence = _recompute_bundle_factors(
        provider=provider,
        output=output,
        factors=bundle_factors,
        periods=periods,
        dataset_identity_sha256=dataset_identity,
        universe=str(manifest.get("universe") or "cn_all"),
    )

    workspace = output / "model"
    result, execution = execute_model_candidate(
        code_path=Path(str(model.get("code_path") or "")),
        provider_path=provider,
        additional_factors_path=additional_path,
        manifest={
            "candidate_id": str(model.get("candidate_id") or ""),
            "code_sha256": str(model.get("code_sha256") or ""),
            "model_type": str(model.get("model_type") or "Tabular"),
            "model_engine": "rdagent_pytorch",
            "training_hyperparameters": model.get("training_hyperparameters") or {},
            "feature_set": feature_set,
            "additional_factor_count": len(bundle_factors),
            "periods": periods,
            "prediction_segment": "test",
            "seed": int(model.get("seed") or 11),
            "dataset_identity_sha256": dataset_identity,
            "universe": str(manifest.get("universe") or "cn_all"),
            "final_oos_opened": False,
            "inference_only": True,
        },
        workspace=workspace,
        runner_path=Path(__file__).resolve().with_name("model_sandbox_runner.py"),
        allow_inference=True,
        timeout_seconds=int(manifest.get("timeout_seconds") or 7200),
    )
    source_environment_sha256 = str(
        manifest.get("execution_environment_sha256") or ""
    )
    if (
        result.get("execution_environment_sha256")
        != source_environment_sha256
        or execution.get("execution_environment_sha256")
        != source_environment_sha256
    ):
        raise ValueError(
            "model refit execution environment differs from the approved artifact"
        )
    predictions = workspace / "output" / "predictions.parquet"
    coverage = verify_model_prediction_artifact(
        predictions,
        expected_sha256=str(result["predictions_sha256"]),
        test_start=signal_date,
        test_end=signal_date,
        trading_days=[signal_date],
    )
    normalized = normalize_model_predictions(pd.read_parquet(predictions))
    actual_days = pd.DatetimeIndex(
        normalized.index.get_level_values("datetime")
    ).normalize().unique()
    if len(actual_days) != 1 or actual_days[0] != pd.Timestamp(signal_date):
        raise ValueError("model refit produced predictions outside the requested signal date")

    final_predictions = output / "predictions.parquet"
    final_checkpoint = output / "checkpoint.pt"
    shutil.copy2(predictions, final_predictions)
    shutil.copy2(workspace / "output" / "checkpoint.pt", final_checkpoint)
    payload = {
        "contract_version": REFIT_CONTRACT_VERSION,
        "status": "passed",
        "strategy_version_id": str(manifest.get("strategy_version_id") or ""),
        "source_model_artifact_id": str(manifest.get("source_model_artifact_id") or ""),
        "model_candidate_id": str(model.get("candidate_id") or ""),
        "model_code_sha256": str(model.get("code_sha256") or ""),
        "model_recipe_sha256": str(model.get("recipe_sha256") or ""),
        "feature_set_definition_sha256": str(feature_set.get("definition_sha256") or ""),
        "dataset": str(manifest.get("dataset") or ""),
        "dataset_identity_sha256": dataset_identity,
        "dataset_lineage_id": str(manifest.get("dataset_lineage_id") or ""),
        "signal_date": signal_date,
        "periods": periods,
        "predictions_path": "predictions.parquet",
        "predictions_sha256": sha256_file(final_predictions),
        "checkpoint_path": "checkpoint.pt",
        "checkpoint_sha256": sha256_file(final_checkpoint),
        "coverage": coverage,
        "execution_evidence": execution,
        "execution_environment_sha256": source_environment_sha256,
        "bundle_factor_evidence": factor_evidence,
        "refit_policy": dict(refit_policy),
        "refit_policy_sha256": MODEL_REFIT_POLICY_SHA256,
        "final_oos_opened": False,
        "inference_only": True,
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
