#!/usr/bin/env python3
"""Produce one governed daily prediction from a frozen model StrategySpec.

Most sessions load the active immutable checkpoint and call only ``predict``.
The first trading day of a month (or an explicitly evidenced early trigger)
rolls the training window and creates a new safe-format checkpoint.  Neither
path opens the final OOS or reads labels after the signal date.
"""

from __future__ import annotations

import argparse
import json
import math
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
from quant_platform.model_recompute import (
    GOVERNED_MODEL_ENGINES,
    execute_model_candidate,
    governed_checkpoint_filename,
)
from quant_platform.model_research_governance import (
    MODEL_REFIT_POLICY,
    MODEL_REFIT_POLICY_SHA256,
    canonical_sha256,
    normalize_model_predictions,
    verify_model_prediction_artifact,
)

REFIT_CONTRACT_VERSION = "model-live-refresh-v2"


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
    if not days or days[-1] != signal_date:
        raise ValueError("model refit signal date is not an available Qlib trading day")
    if len(days) < required:
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


def _inference_periods(
    provider: Path,
    signal_date: str,
    frozen: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, str]:
    calendar = [
        line.strip()
        for line in (provider / "calendars" / "day.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and line.strip() <= signal_date
    ]
    positions = {value: index for index, value in enumerate(calendar)}
    required = ("train_start", "train_end", "valid_start", "valid_end")
    if not calendar or calendar[-1] != signal_date or any(
        not isinstance(frozen.get(key), str) for key in required
    ):
        raise ValueError("live inference is missing its frozen training periods")
    try:
        if (
            positions[str(frozen["train_end"])]
            - positions[str(frozen["train_start"])]
            + 1
            != int(policy["train_trading_days"])
            or positions[str(frozen["valid_end"])]
            - positions[str(frozen["valid_start"])]
            + 1
            != int(policy["validation_trading_days"])
        ):
            raise ValueError("live inference frozen training window changed")
        if positions[str(frozen["valid_start"])] - positions[str(frozen["train_end"])] - 1 < int(
            policy["train_validation_purge_trading_days"]
        ):
            raise ValueError("live inference training/validation purge changed")
        if positions[signal_date] - positions[str(frozen["valid_end"])] - 1 < int(
            policy["embargo_trading_days"]
        ):
            raise ValueError("live inference precedes the frozen validation embargo")
    except KeyError as exc:
        raise ValueError("live inference periods are outside the governed calendar") from exc
    return {
        "train_start": str(frozen["train_start"]),
        "train_end": str(frozen["train_end"]),
        "valid_start": str(frozen["valid_start"]),
        "valid_end": str(frozen["valid_end"]),
        "test_start": signal_date,
        "test_end": signal_date,
    }


def _validate_retrain_evidence(
    *,
    provider: Path,
    signal_date: str,
    reason: str,
    evidence: dict[str, Any],
    dataset_identity_sha256: str,
    dataset_lineage_id: str,
    source_model_data_contract_sha256: str,
) -> None:
    if reason == "monthly_first_trading_day":
        calendar = [
            value.strip()
            for value in (provider / "calendars" / "day.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if value.strip()
        ]
        month = signal_date[:7]
        month_days = [value for value in calendar if value.startswith(month)]
        if (
            not month_days
            or signal_date != min(month_days)
            or evidence.get("trigger") != reason
            or evidence.get("is_first_trading_day") is not True
            or evidence.get("signal_date") != signal_date
            or evidence.get("signal_month") != month
            or evidence.get("calendar_dataset_identity_sha256")
            != dataset_identity_sha256
        ):
            raise ValueError("monthly retraining is not the first governed trading day")
        return
    if reason == "persistent_drift":
        try:
            observed = float(evidence["observed"])
            threshold = float(evidence["threshold"])
            consecutive_windows = int(evidence["consecutive_windows"])
            window_trading_days = int(evidence["window_trading_days"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("persistent-drift evidence is incomplete") from exc
        direction = str(evidence.get("comparison") or "")
        crossed = (direction == "above" and observed >= threshold) or (
            direction == "below" and observed <= threshold
        )
        if (
            evidence.get("contract_version") != "model-drift-trigger-v1"
            or evidence.get("as_of") != signal_date
            or evidence.get("dataset_lineage_id") != dataset_lineage_id
            or not str(evidence.get("metric") or "").strip()
            or not math.isfinite(observed)
            or not math.isfinite(threshold)
            or consecutive_windows < 3
            or window_trading_days < 20
            or not crossed
        ):
            raise ValueError("persistent-drift evidence does not cross the frozen trigger")
        return
    if reason == "data_contract_change":
        previous = str(evidence.get("previous_contract_sha256") or "")
        current = str(evidence.get("current_contract_sha256") or "")
        if (
            evidence.get("contract_version") != "model-data-contract-trigger-v1"
            or evidence.get("as_of") != signal_date
            or evidence.get("dataset_lineage_id") != dataset_lineage_id
            or previous != source_model_data_contract_sha256
            or len(current) != 64
            or any(character not in "0123456789abcdef" for character in current)
            or current == previous
            or evidence.get("compatibility_review_passed") is not True
        ):
            raise ValueError("data-contract retraining evidence is invalid")
        return
    raise ValueError("model retraining trigger is not governed")


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
    existing_result_path = output / "result.json"
    if existing_result_path.is_file():
        existing = _load_json(existing_result_path)
        if (
            existing.get("contract_version") != REFIT_CONTRACT_VERSION
            or existing.get("strategy_version_id")
            != str(manifest.get("strategy_version_id") or "")
            or existing.get("source_model_artifact_id")
            != str(manifest.get("source_model_artifact_id") or "")
            or existing.get("signal_date") != str(manifest.get("signal_date") or "")
            or existing.get("operation") != str(manifest.get("operation") or "")
        ):
            raise ValueError("model refresh output already belongs to another contract")
        # The worker-side ModelArtifactStore re-verifies the signed result and
        # every file hash before accepting this idempotent replay.
        print(json.dumps(existing, ensure_ascii=False))
        return
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)
    if manifest.get("contract_version") != REFIT_CONTRACT_VERSION:
        raise ValueError("model refit contract is invalid")
    operation = str(manifest.get("operation") or "")
    if operation not in {"inference", "retrain"}:
        raise ValueError("model live refresh operation is invalid")
    retrain_reason = str(manifest.get("retrain_reason") or "")
    retrain_evidence = manifest.get("retrain_evidence")
    retrain_evidence_sha256 = str(manifest.get("retrain_evidence_sha256") or "")
    if operation == "inference":
        if retrain_reason or retrain_evidence is not None or retrain_evidence_sha256:
            raise ValueError("daily inference cannot carry retraining evidence")
    elif (
        retrain_reason
        not in {"monthly_first_trading_day", "persistent_drift", "data_contract_change"}
        or not isinstance(retrain_evidence, dict)
        or canonical_sha256(retrain_evidence) != retrain_evidence_sha256
    ):
        raise ValueError("model retraining requires explicit immutable trigger evidence")
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
    if operation == "retrain":
        _validate_retrain_evidence(
            provider=provider,
            signal_date=signal_date,
            reason=retrain_reason,
            evidence=dict(retrain_evidence),
            dataset_identity_sha256=dataset_identity,
            dataset_lineage_id=str(manifest.get("dataset_lineage_id") or ""),
            source_model_data_contract_sha256=str(
                manifest.get("source_model_data_contract_sha256") or ""
            ),
        )
    if operation == "retrain":
        _, periods = _calendar(provider, signal_date, dict(refit_policy))
    else:
        periods = _inference_periods(
            provider,
            signal_date,
            dict(manifest.get("frozen_training_periods") or {}),
            dict(refit_policy),
        )
    feature_set = manifest.get("feature_set")
    model = manifest.get("model")
    if not isinstance(feature_set, dict) or not isinstance(model, dict):
        raise ValueError("model refit is missing its frozen model or feature set")
    if feature_set.get("definition_sha256") != manifest.get(
        "feature_set_definition_sha256"
    ):
        raise ValueError("model refit feature-set definition is inconsistent")
    model_engine = str(model.get("model_engine") or "")
    if model_engine not in GOVERNED_MODEL_ENGINES:
        raise ValueError("model refit is missing its frozen governed model engine")

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
    inference_only = operation == "inference"
    result, execution = execute_model_candidate(
        code_path=Path(str(model.get("code_path") or "")),
        provider_path=provider,
        additional_factors_path=additional_path,
        manifest={
            "candidate_id": str(model.get("candidate_id") or ""),
            "code_sha256": str(model.get("code_sha256") or ""),
            "model_type": str(model.get("model_type") or "Tabular"),
            "model_engine": model_engine,
            "training_hyperparameters": model.get("training_hyperparameters") or {},
            "feature_set": feature_set,
            "additional_factor_count": len(bundle_factors),
            "periods": periods,
            "prediction_segment": "test",
            "seed": int(model.get("seed") or 11),
            "resource_stage": "inference" if inference_only else "production_refit",
            "dataset_identity_sha256": dataset_identity,
            "universe": str(manifest.get("universe") or "cn_all"),
            "final_oos_opened": False,
            "inference_only": inference_only,
            "live_retrain": not inference_only,
        },
        workspace=workspace,
        runner_path=Path(__file__).resolve().with_name("model_sandbox_runner.py"),
        allow_inference=inference_only,
        allow_live_retrain=not inference_only,
        source_checkpoint_path=(
            Path(str(manifest.get("source_checkpoint_path") or "")).resolve()
            if inference_only
            else None
        ),
        source_checkpoint_sha256=(
            str(manifest.get("source_checkpoint_sha256") or "")
            if inference_only
            else None
        ),
        source_checkpoint_format=(
            str(manifest.get("source_checkpoint_format") or "")
            if inference_only
            else None
        ),
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
    if (
        operation == "retrain"
        and retrain_reason == "data_contract_change"
        and result.get("model_data_contract_sha256")
        != retrain_evidence.get("current_contract_sha256")
    ):
        raise ValueError(
            "retrained model data contract does not match the evidenced replacement"
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
    shutil.copy2(predictions, final_predictions)
    if inference_only:
        checkpoint_path: str | None = None
        checkpoint_sha256 = str(manifest.get("source_checkpoint_sha256") or "")
        checkpoint_format = str(manifest.get("source_checkpoint_format") or "")
    else:
        final_checkpoint = output / governed_checkpoint_filename(model_engine)
        shutil.copy2(
            workspace / "output" / governed_checkpoint_filename(model_engine),
            final_checkpoint,
        )
        checkpoint_path = final_checkpoint.name
        checkpoint_sha256 = sha256_file(final_checkpoint)
        checkpoint_format = str(result.get("checkpoint_format") or "")
    training_evidence = {
        "operation": operation,
        "periods": periods,
        "retrain_evidence": retrain_evidence if operation == "retrain" else None,
        "retrain_evidence_sha256": (
            retrain_evidence_sha256 if operation == "retrain" else None
        ),
        "retrain_reason": retrain_reason if operation == "retrain" else None,
        "source_model_artifact_id": str(
            manifest.get("source_model_artifact_id") or ""
        ),
    }
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
        "operation": operation,
        "periods": periods,
        "predictions_path": "predictions.parquet",
        "predictions_sha256": sha256_file(final_predictions),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_format": checkpoint_format,
        "checkpoint_reused": inference_only,
        "model_data_contract_sha256": str(
            result.get("model_data_contract_sha256") or ""
        ),
        "model_data_contract": result.get("model_data_contract"),
        "training_evidence": training_evidence,
        "training_evidence_sha256": canonical_sha256(training_evidence),
        "coverage": coverage,
        "execution_evidence": execution,
        "execution_environment_sha256": source_environment_sha256,
        "bundle_factor_evidence": factor_evidence,
        "refit_policy": dict(refit_policy),
        "refit_policy_sha256": MODEL_REFIT_POLICY_SHA256,
        "final_oos_opened": False,
        "inference_only": inference_only,
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
