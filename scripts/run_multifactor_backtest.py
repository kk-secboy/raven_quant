#!/usr/bin/env python3
"""Run a governed multi-factor Top-K backtest on a selected Qlib snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_data.availability import filter_available
from quant_data.execution_contract import (
    require_daily_qlib_contract,
    require_minute_execution_contract,
    require_native_daily_execution_controls,
    require_strategy_execution_contract,
)
from quant_data.qlib_builder import verify_qlib_output_manifest
from quant_platform.cost_model import CostScheduleBook
from quant_platform.eligibility import (
    PreparedPointInTimeRiskStates,
    eligibility_statistics,
)
from quant_platform.execution_algorithms import execution_time_slots
from quant_platform.factor_recompute import (
    execute_factor_code,
    normalize_factor_input,
    require_exact_factor_index,
    require_exact_oos_coverage,
    sha256_file,
    validate_factor_prefix_invariance,
)
from quant_platform.formal_validation import (
    CONSERVATIVE_BONFERRONI_INCOMPLETE_FAMILY_STATUS,
    FORMAL_VALIDATION_CONTRACT_VERSION,
    FROZEN_STRATEGY_OUTER_SCOPE,
    build_factor_score_incomplete_family_dsr,
    build_factor_score_incomplete_family_multiple_testing,
    build_paired_bootstrap_evidence,
    build_pre_final_history_evidence,
    paired_bootstrap_parameters_from_config,
    run_ablation_suite,
    run_outer_walk_forward,
    run_signal_decay_suite,
)
from quant_platform.model_recompute import (
    execute_model_candidate,
    governed_checkpoint_filename,
)
from quant_platform.model_research_governance import (
    MODEL_REFIT_POLICY,
    MODEL_REFIT_POLICY_SHA256,
    REQUIRED_QUANT_ABLATIONS,
    normalize_model_predictions,
    verify_model_prediction_artifact,
)
from quant_platform.model_research_governance import (
    canonical_sha256 as canonical_model_sha256,
)
from quant_platform.model_strategy_contract import (
    MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_REASON,
    MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_STATUS,
    model_signal_identity,
    validate_model_formal_admission_binding,
)
from quant_platform.portfolio_policy import PortfolioPolicy, PortfolioPolicyConfig
from quant_platform.qlib_backtest import (
    QLIB_ENGINE_VERSION,
    run_formal_qlib_backtest,
    run_qlib_validation_suites,
)
from quant_platform.qlib_factor_baseline import (
    FACTOR_SOURCE_PROMOTED_ONLY,
    combine_factor_sources,
    normalize_qlib_baseline_values,
)
from quant_platform.qlib_policy_strategy import create_qlib_policy_strategy
from quant_platform.qlib_workflow import qlib_workflow_run
from quant_platform.replay_governance import (
    EVIDENCE_MODE_REPLAY,
    EVIDENCE_MODE_SEALED,
    REPLAY_MARKERS,
    require_incomplete_family_eligibility,
    require_replay_config,
    require_replay_markers,
)
from quant_platform.risk_math import estimate_covariance
from quant_platform.statistical_validation import (
    deflated_sharpe_probability,
    holm_bonferroni,
)
from quant_platform.strategy_artifact_manifest import write_backtest_artifact_manifest
from quant_platform.strategy_backtest import (
    build_governed_signal,
    compose_factor_scores,
    governed_score_neutralization,
)
from quant_platform.strategy_health_reference import (
    STRATEGY_HEALTH_REFERENCE_NAME,
    STRATEGY_HEALTH_REFERENCE_VERSION,
    build_strategy_health_reference,
    factor_reference_summary,
    model_calibration_reference_summary,
)
from quant_platform.strategy_research_evaluation import (
    STRATEGY_RESEARCH_EVALUATION_MODES,
)
from quant_platform.strategy_rule_runtime import (
    apply_strategy_rule_alpha_weights,
    build_portfolio_policy_runtime_metadata,
    build_strategy_rule_runtime_metadata,
    load_market_trend_close_history,
    policy_style_cross_sections,
)
from quant_platform.strategy_rule_runtime import (
    load_governed_style_exposures as _load_governed_style_exposures,
)
from quant_platform.transparent_baseline_runner import (
    TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_RESULT_WORKER_RUNTIME_IMAGE_FIELD,
    require_transparent_baseline_runner,
)
from quant_platform.upstream_versions import upstream_runtime_identity

FORMAL_FINAL_OOS_MODE = "formal_final_oos"
CONSUMED_HISTORICAL_REPLAY_MODE = "consumed_historical_replay"
PRE_FINAL_PORTFOLIO_TRIAL_MODE = "pre_final_portfolio_trial"
_COVARIANCE_REQUIRED_PORTFOLIO_CONSTRUCTIONS = frozenset(
    {"benchmark_relative_qp", "industry_neutral_qp"}
)
PRE_FINAL_EVALUATION_MODES = frozenset(
    {PRE_FINAL_PORTFOLIO_TRIAL_MODE, *STRATEGY_RESEARCH_EVALUATION_MODES}
)
GOVERNED_MODEL_ENGINES = frozenset(
    {
        "rdagent_pytorch",
        "ridge_baseline",
        "lightgbm_baseline",
        "platform_gru",
        "platform_transformer",
    }
)


def _write_strategy_health_reference(
    output: Path,
    *,
    manifest: dict[str, Any],
    provider_provenance: dict[str, Any],
    periods: dict[str, str],
    signal_source: str,
    factor_source_mode: str,
    baseline_definition: dict[str, Any] | None,
    baseline_raw: pd.DataFrame | None,
    baseline_artifacts: dict[str, Any] | None,
    challenger_entries: list[tuple[str, pd.DataFrame, float, int]],
    formal_factor_items: list[dict[str, Any]],
    model_feature_set: dict[str, Any] | None,
    model_predictions: pd.DataFrame | None,
    model_label_contract: dict[str, Any] | None,
    qlib_data_api: Any,
) -> dict[str, Any]:
    """Freeze bounded formal sufficient statistics before sealing the artifact tree."""

    reference_start = pd.Timestamp(periods["start"]).date()
    reference_end = pd.Timestamp(periods["end"]).date()
    factor_summaries: dict[str, dict[str, Any]] = {}
    if signal_source == "model_prediction":
        if (
            not isinstance(model_feature_set, dict)
            or not isinstance(model_predictions, pd.DataFrame)
            or not isinstance(model_label_contract, dict)
        ):
            raise ValueError("formal model run has no strategy-health reference inputs")
        feature_definition_sha256 = str(model_feature_set.get("definition_sha256") or "")
        for factor_id, expression in sorted(
            dict(model_feature_set.get("features") or {}).items()
        ):
            values = qlib_data_api.features(
                qlib_data_api.instruments(str(manifest.get("universe") or "cn_all")),
                [str(expression)],
                start_time=periods["start"],
                end_time=periods["end"],
                freq="day",
            )
            source_sha256 = _canonical_sha256(
                {
                    "formal_dataset_identity_sha256": provider_provenance[
                        "dataset_identity_sha256"
                    ],
                    "feature_set_definition_sha256": feature_definition_sha256,
                    "factor_id": str(factor_id),
                    "expression": str(expression),
                }
            )
            factor_summaries[str(factor_id)] = factor_reference_summary(
                values.iloc[:, 0],
                factor_id=str(factor_id),
                reference_start=reference_start,
                reference_end=reference_end,
                source_sha256=source_sha256,
            )
            del values
        label_expression = str(model_label_contract.get("label_expression") or "")
        label_contract_sha256 = _canonical_sha256(model_label_contract)
        if not label_expression:
            raise ValueError("formal model label contract has no expression")
        label_values = qlib_data_api.features(
            qlib_data_api.instruments(str(manifest.get("universe") or "cn_all")),
            [label_expression],
            start_time=periods["start"],
            end_time=periods["end"],
            freq="day",
        )
        labels_source_sha256 = _canonical_sha256(
            {
                "formal_dataset_identity_sha256": provider_provenance[
                    "dataset_identity_sha256"
                ],
                "label_contract_sha256": label_contract_sha256,
                "label_expression": label_expression,
            }
        )
        formal_predictions_sha256 = str(
            ((manifest.get("formal_model_artifact") or {}).get("predictions_sha256"))
            or ""
        )
        calibration = model_calibration_reference_summary(
            model_predictions,
            label_values.iloc[:, 0],
            label_horizon_sessions=int(
                model_label_contract["label_horizon_sessions"]
            ),
            label_contract_sha256=label_contract_sha256,
            predictions_sha256=formal_predictions_sha256,
            labels_source_sha256=labels_source_sha256,
        )
        del label_values
    else:
        includes_baseline = factor_source_mode in {
            "qlib_baseline",
            "qlib_baseline_plus_challenger",
        }
        includes_challenger = factor_source_mode in {
            "promoted_only",
            "qlib_baseline_plus_challenger",
            "qlib_challenger_replacement",
        }
        if includes_baseline:
            if (
                not isinstance(baseline_definition, dict)
                or baseline_raw is None
                or not isinstance(baseline_artifacts, dict)
            ):
                raise ValueError("formal baseline reference inputs are missing")
            raw_artifacts = dict(baseline_artifacts.get("raw") or {})
            for item in baseline_definition.get("factors") or []:
                factor_id = str(item.get("id") or "")
                artifact = dict(raw_artifacts.get(factor_id) or {})
                factor_summaries[factor_id] = factor_reference_summary(
                    baseline_raw[factor_id],
                    factor_id=factor_id,
                    reference_start=reference_start,
                    reference_end=reference_end,
                    source_sha256=str(artifact.get("sha256") or ""),
                )
        if includes_challenger:
            formal_by_id = {
                str(item["candidate_id"]): dict(item.get("formal_factor_artifact") or {})
                for item in formal_factor_items
            }
            for candidate_id, values, _weight, _direction in challenger_entries:
                factor_id = f"candidate__{candidate_id}"
                factor_summaries[factor_id] = factor_reference_summary(
                    values,
                    factor_id=factor_id,
                    reference_start=reference_start,
                    reference_end=reference_end,
                    source_sha256=str(formal_by_id[candidate_id].get("sha256") or ""),
                )
        calibration = None
    reference = build_strategy_health_reference(
        strategy_version_id=str(manifest["strategy_version_id"]),
        formal_backtest_id=str(manifest["backtest_id"]),
        formal_dataset_identity_sha256=str(
            provider_provenance["dataset_identity_sha256"]
        ),
        formal_dataset_lineage_id=str(provider_provenance["dataset_lineage_id"]),
        strategy_rules_sha256=str(manifest["strategy_rules_sha256"]),
        signal_source=signal_source,
        reference_start=reference_start,
        reference_end=reference_end,
        factor_summaries=factor_summaries,
        model_calibration=calibration,
    )
    target = output / STRATEGY_HEALTH_REFERENCE_NAME
    target.write_text(
        json.dumps(reference, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "contract_version": STRATEGY_HEALTH_REFERENCE_VERSION,
        "path": STRATEGY_HEALTH_REFERENCE_NAME,
        "sha256": _sha256_file(target),
        "reference_sha256": reference["reference_sha256"],
        "factor_count": len(factor_summaries),
    }


def _require_health_reference_scope(output: Path, evaluation_mode: str) -> None:
    """Do not silently seal a formal reference left in a selection-only directory."""
    reference_path = output / STRATEGY_HEALTH_REFERENCE_NAME
    if evaluation_mode in PRE_FINAL_EVALUATION_MODES and (
        reference_path.exists() or reference_path.is_symlink()
    ):
        raise ValueError("pre-final output must not contain a formal strategy health reference")


def _finalize_backtest_output(
    output: Path,
    *,
    manifest: dict[str, Any],
    manifest_path: str | Path,
    provider_provenance: dict[str, Any],
    periods: dict[str, str],
    metrics: dict[str, Any],
    evaluation_mode: str,
    signal_source: str,
    execution_method: str,
    execution_frequency: str,
    historical_window_opened: bool,
    tracking_uri: str,
    health_reference_inputs: dict[str, Any],
) -> dict[str, Any]:
    """Publish completed output and tracking after calculations and gates settle."""
    _require_health_reference_scope(output, evaluation_mode)
    config = manifest["config"]
    factor_source_mode = str(health_reference_inputs["factor_source_mode"])
    consumed_historical_replay = evaluation_mode == CONSUMED_HISTORICAL_REPLAY_MODE
    strategy_health_reference = None
    if evaluation_mode not in PRE_FINAL_EVALUATION_MODES:
        # Health references bind a formal backtest identity. Selection-only
        # parameter segments have no formal identity and must not publish one.
        strategy_health_reference = _write_strategy_health_reference(
            output, manifest=manifest, provider_provenance=provider_provenance,
            periods=periods, signal_source=signal_source, **health_reference_inputs,
        )
        metrics["provenance"]["strategy_health_reference"] = strategy_health_reference
    result = {
        "status": "ok",
        "backtest_engine": "qlib",
        "evaluation_mode": evaluation_mode,
        "evidence_mode": (
            EVIDENCE_MODE_REPLAY
            if consumed_historical_replay
            else EVIDENCE_MODE_SEALED
        ),
        "final_oos_opened": historical_window_opened,
        **(REPLAY_MARKERS if consumed_historical_replay else {}),
        "metrics": metrics,
        "periods": periods,
        "benchmark": manifest["benchmark"],
        "artifacts": {
            "daily_returns": str(output / "daily_returns.parquet"),
            "score_grid": str(output / "score_grid.parquet"),
            "governed_signal": str(output / "governed_signal.parquet"),
            "qlib_portfolio_report": str(output / "qlib_portfolio_report.parquet"),
            "qlib_positions": str(output / "qlib_positions.pkl"),
            "execution_fills": str(output / "execution_fills.parquet"),
            "execution_model": str(output / "execution_model.json"),
            "robustness": str(output / "robustness.json"),
            "rolling": str(output / "rolling.json"),
            "event_stress": str(output / "event_stress.json"),
            "capacity_curve": str(output / "capacity_curve.json"),
            "formal_validation": str(output / "formal_validation.json"),
            **(
                {"strategy_health_reference": str(output / STRATEGY_HEALTH_REFERENCE_NAME)}
                if strategy_health_reference is not None else {}
            ),
        },
    }
    workflow_run_id = str(
        manifest.get("backtest_id")
        or (
            f"{manifest['strategy_version_id']}-"
            f"{_canonical_sha256({'periods': periods, 'config': config})[:16]}"
        )
    )
    with qlib_workflow_run(
        run_kind=(
            "portfolio-experiment-trial"
            if evaluation_mode in PRE_FINAL_EVALUATION_MODES
            else "formal-backtest"
        ),
        run_id=workflow_run_id,
        tracking_uri=tracking_uri,
        dataset_identity_sha256=provider_provenance.get("dataset_identity_sha256"),
    ) as workflow:
        workflow.log_params(
            {
                "backtest_id": manifest.get("backtest_id") or workflow_run_id,
                "strategy_version_id": manifest["strategy_version_id"],
                "dataset": manifest["dataset"],
                "execution_dataset": manifest.get("execution_dataset"),
                "benchmark": manifest["benchmark"],
                "start": periods["start"],
                "end": periods["end"],
                "execution_method": execution_method,
                "execution_frequency": execution_frequency,
                "strategy_config_sha256": metrics["provenance"]["strategy_config_sha256"],
                "evaluation_mode": evaluation_mode,
                "final_oos_opened": historical_window_opened,
                **(REPLAY_MARKERS if consumed_historical_replay else {}),
            }
        )
        workflow.log_metrics(metrics)
        recorder_identity = workflow.identity_dict()
        manifest["qlib_workflow"] = recorder_identity
        manifest["factor_source_mode"] = factor_source_mode
        manifest["challenger_weight"] = float(config.get("challenger_weight") or 0.0)
        manifest["final_oos_opened"] = historical_window_opened
        if consumed_historical_replay:
            manifest.update(REPLAY_MARKERS)
        Path(manifest_path).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metrics["provenance"]["execution_manifest_sha256"] = _sha256_file(manifest_path)
        metrics["provenance"]["qlib_workflow"] = recorder_identity
        result["qlib_workflow"] = recorder_identity
        artifact_manifest = write_backtest_artifact_manifest(output)
        metrics["provenance"]["artifact_manifest_version"] = artifact_manifest["version"]
        metrics["provenance"]["artifact_manifest_sha256"] = artifact_manifest["sha256"]
        metrics["provenance"]["artifact_manifest_file_count"] = artifact_manifest[
            "file_count"
        ]
        result["artifacts"]["artifact_manifest"] = str(
            output / artifact_manifest["path"]
        )
        (output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        workflow.save_artifacts(output)
    return result


def _require_promotion_dataset_identity(
    provenance: dict[str, Any], *, label: str
) -> None:
    """Reject a formally usable dataset that paper simulation cannot bind."""

    for field in (
        "dataset_identity_sha256",
        "dataset_lineage_id",
        "source_lineage_id",
    ):
        value = str(provenance.get(field) or "").strip().lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{label} dataset provenance {field} must be a SHA-256 digest")


def _promotion_dataset_descriptors(
    *,
    daily_dataset_name: str,
    daily_provenance: dict[str, Any],
    execution_method: str,
    execution_frequency: str,
    formal_execution_start: str,
    execution_dataset_name: str | None = None,
    execution_provenance: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Freeze the exact Qlib datasets consumed by promotion and paper replay.

    A daily next-open strategy executes from the same native daily Qlib
    dataset used for scoring.  It therefore needs a real daily execution
    descriptor, not ``null`` and not a fabricated minute descriptor.  Minute
    strategies keep their independently sealed minute execution dataset.
    """

    normalized_daily_name = str(daily_dataset_name or "").strip()
    if not normalized_daily_name:
        raise ValueError("formal backtest daily dataset name is required")
    require_daily_qlib_contract(daily_provenance)
    require_native_daily_execution_controls(
        daily_provenance,
        start=formal_execution_start,
    )
    _require_promotion_dataset_identity(daily_provenance, label="daily")
    daily = {
        "name": normalized_daily_name,
        "provenance": deepcopy(daily_provenance),
    }

    method = str(execution_method or "").strip().lower()
    frequency = str(execution_frequency or "").strip().lower()
    if method == "open":
        if frequency != "day":
            raise ValueError("daily open execution requires a day execution frequency")
        if execution_dataset_name is not None or execution_provenance is not None:
            raise ValueError(
                "daily open execution must reuse the daily Qlib dataset, not a minute dataset"
            )
        return {"daily": daily, "execution": deepcopy(daily)}

    if method not in {"twap", "vwap", "next_bar"} or frequency not in {
        "1min",
        "5min",
    }:
        raise ValueError("formal execution dataset descriptor contract is unsupported")
    normalized_execution_name = str(execution_dataset_name or "").strip()
    if not normalized_execution_name or not isinstance(execution_provenance, dict):
        raise ValueError("minute execution requires an independently sealed Qlib dataset")
    if normalized_execution_name == normalized_daily_name:
        raise ValueError("minute execution dataset must be distinct from the daily dataset")
    require_minute_execution_contract(execution_provenance, frequency=frequency)
    _require_promotion_dataset_identity(execution_provenance, label="execution")
    if str(execution_provenance.get("source_lineage_id") or "") != str(
        daily_provenance.get("source_lineage_id") or ""
    ):
        raise ValueError("daily and execution datasets must share one verified source lineage")
    return {
        "daily": daily,
        "execution": {
            "name": normalized_execution_name,
            "provenance": deepcopy(execution_provenance),
        },
    }


def _evaluation_mode(manifest: dict[str, Any]) -> str:
    mode = str(manifest.get("evaluation_mode") or FORMAL_FINAL_OOS_MODE)
    if mode not in {
        FORMAL_FINAL_OOS_MODE,
        CONSUMED_HISTORICAL_REPLAY_MODE,
        *PRE_FINAL_EVALUATION_MODES,
    }:
        raise ValueError("unsupported governed backtest evaluation mode")
    config = manifest.get("config")
    config = config if isinstance(config, dict) else {}
    evidence_mode = str(manifest.get("evidence_mode") or config.get("evidence_mode") or "")
    if mode == CONSUMED_HISTORICAL_REPLAY_MODE:
        require_replay_config(config)
        require_replay_markers(manifest, label="backtest manifest")
        if evidence_mode != EVIDENCE_MODE_REPLAY:
            raise ValueError("historical replay manifest evidence mode is inconsistent")
    elif mode == FORMAL_FINAL_OOS_MODE and evidence_mode != EVIDENCE_MODE_SEALED:
        raise ValueError("formal final OOS requires sealed_final_oos evidence mode")
    return mode


def _frozen_model_engine(
    recipe: dict[str, Any], *, recipe_sha256: str
) -> str:
    if canonical_model_sha256(recipe) != recipe_sha256:
        raise ValueError("formal model recipe changed after admission")
    hyperparameters = recipe.get("model_hyperparameters") or {}
    if not isinstance(hyperparameters, dict):
        raise ValueError("formal model recipe hyperparameters are invalid")
    direct = str(recipe.get("model_engine") or "").strip()
    nested = str(hyperparameters.get("model_engine") or "").strip()
    if direct and nested and direct != nested:
        raise ValueError("formal model recipe contains conflicting model engines")
    engine = str(
        direct or nested or "rdagent_pytorch"
    )
    if engine not in GOVERNED_MODEL_ENGINES:
        raise ValueError("formal model recipe requests an ungoverned model engine")
    return engine


def _model_execution_authorization(
    *,
    evaluation_mode: str,
    candidate_manifest: dict[str, Any],
    model_periods: dict[str, Any],
    periods: dict[str, str],
    historical_periods: dict[str, str],
    pre_final_cutoff: str | None,
) -> dict[str, Any]:
    """Return the only allowed model prediction scope for this run.

    The default remains the one-shot final-OOS path.  The alternate path is
    selection-only and can only reuse the admitted primary validation window;
    it cannot move, alias, or cross the candidate's frozen pre-final cutoff.
    """

    candidate_cutoff = str(candidate_manifest.get("pre_final_end") or "")
    final_start = str(candidate_manifest.get("final_oos_start") or "")
    final_end = str(candidate_manifest.get("final_oos_end") or "")
    if not candidate_cutoff or not candidate_cutoff < final_start <= final_end:
        raise ValueError("formal model candidate has an invalid frozen OOS boundary")
    if evaluation_mode == FORMAL_FINAL_OOS_MODE:
        if (
            candidate_cutoff != historical_periods["end"]
            or final_start != periods["start"]
            or final_end != periods["end"]
            or not (
                str(model_periods["train_start"])
                <= str(model_periods["train_end"])
                < str(model_periods["valid_start"])
                <= str(model_periods["valid_end"])
                <= historical_periods["end"]
                < periods["start"]
            )
        ):
            raise ValueError("formal model training periods are not isolated before final OOS")
        return {
            "prediction_segment": "test",
            "final_oos_opened": True,
            "allow_final_oos": True,
            "allow_inference": False,
            "evaluation_scope": "final_oos_once",
        }
    if (
        pre_final_cutoff != candidate_cutoff
        or periods["end"] > candidate_cutoff
        or periods["start"] < str(model_periods["valid_start"])
        or periods["end"] > str(model_periods["valid_end"])
        or not (
            str(model_periods["train_start"])
            <= str(model_periods["train_end"])
            < str(model_periods["valid_start"])
            <= str(model_periods["valid_end"])
            <= candidate_cutoff
            < final_start
        )
    ):
        raise ValueError("pre-final portfolio trial crosses its admitted validation cutoff")
    return {
        "prediction_segment": "test",
        "final_oos_opened": False,
        "allow_final_oos": False,
        "allow_inference": True,
        "evaluation_scope": "pre_final_only",
    }


def _pre_final_execution_periods(
    model_periods: dict[str, Any],
    periods: dict[str, str],
    calendar: Any,
    *,
    embargo_sessions: int = 5,
) -> dict[str, Any]:
    """Create an inner validation/test split strictly before the sealed OOS."""

    eligible = pd.DatetimeIndex(calendar).normalize()
    eligible = eligible[
        (eligible >= pd.Timestamp(model_periods["valid_start"]))
        & (eligible < pd.Timestamp(periods["start"]))
    ]
    if len(eligible) <= embargo_sessions:
        raise ValueError(
            "pre-final portfolio trial has no purged inner validation history"
        )
    return {
        "train_start": str(model_periods["train_start"]),
        "train_end": str(model_periods["train_end"]),
        "valid_start": str(model_periods["valid_start"]),
        "valid_end": eligible[-(embargo_sessions + 1)].date().isoformat(),
        "test_start": periods["start"],
        "test_end": periods["end"],
    }


def _load(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        return pd.read_hdf(source)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"unsupported factor artifact: {source}")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _recompute_qlib_baseline(
    data_api: Any,
    *,
    universe: str,
    definition: dict[str, Any],
    start_time: str,
    end_time: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    expressions = [str(item["qlib_expression"]) for item in definition.get("factors") or []]
    values = data_api.features(
        data_api.instruments(universe),
        expressions,
        start_time=start_time,
        end_time=end_time,
        freq=str(definition.get("frequency") or "day"),
    )
    return normalize_qlib_baseline_values(values, definition)


def _write_baseline_artifacts(
    output: Path,
    *,
    raw: pd.DataFrame,
    normalized: pd.DataFrame,
    composite: pd.Series,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {"raw": {}, "normalized": {}}
    for artifact_kind, frame in (("raw", raw), ("normalized", normalized)):
        root = output / "baseline" / artifact_kind
        root.mkdir(parents=True, exist_ok=True)
        for factor_id in frame.columns:
            path = root / f"{factor_id}.parquet"
            frame[factor_id].rename("value").to_frame().to_parquet(path, compression="zstd")
            artifacts[artifact_kind][str(factor_id)] = {
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "sha256": _sha256_file(path),
            }
    composite_path = output / "baseline" / "composite.parquet"
    composite_path.parent.mkdir(parents=True, exist_ok=True)
    composite.rename("score").to_frame().to_parquet(composite_path, compression="zstd")
    artifacts["composite"] = {
        "path": str(composite_path.relative_to(output)).replace("\\", "/"),
        "sha256": _sha256_file(composite_path),
    }
    return artifacts


def _qlib_instruments(provider_uri: str | Path) -> set[str]:
    root = Path(provider_uri) / "instruments"
    candidates = (root / "cn_all.txt", root / "liquid_all.txt", root / "all.txt")
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise ValueError("minute execution Qlib dataset has no instrument universe")
    return {
        line.split("\t", 1)[0].strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _eligible_strategy_instruments(
    scores: pd.Series, eligibility_matrix: pd.DataFrame
) -> list[str]:
    required = {"instrument", "eligible"}
    if not required.issubset(eligibility_matrix.columns):
        raise ValueError("point-in-time eligibility metadata is incomplete")
    score_instruments = set(scores.index.get_level_values("instrument").astype(str))
    eligible_instruments = set(
        eligibility_matrix.loc[
            eligibility_matrix["eligible"].fillna(False).astype(bool), "instrument"
        ].astype(str)
    )
    instruments = sorted(score_instruments & eligible_instruments)
    if not instruments:
        raise ValueError("factor scores have no point-in-time eligible instruments")
    return instruments


def _minute_warmup_window(
    provider_uri: str | Path, frequency: str, start_time: str, lookback_days: int
) -> tuple[str, str]:
    calendar_path = Path(provider_uri) / "calendars" / f"{frequency}.txt"
    try:
        timestamps = pd.to_datetime(
            [
                line.strip()
                for line in calendar_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ],
            errors="raise",
        )
    except (OSError, ValueError) as exc:
        raise ValueError("minute execution calendar is missing or invalid") from exc
    start_date = pd.Timestamp(start_time).date()
    prior_days = sorted({value.date() for value in timestamps if value.date() < start_date})
    if len(prior_days) < lookback_days:
        raise ValueError(
            f"VWAP execution requires {lookback_days} complete pre-backtest minute trading days"
        )
    selected = prior_days[-lookback_days:]
    return selected[0].isoformat(), selected[-1].isoformat()


def _historical_vwap_profile(
    *,
    instruments: list[str],
    start_time: str,
    end_time: str,
    frequency: str,
    slice_minutes: int,
    max_slices: int,
) -> list[dict[str, Any]]:
    from qlib.data import D

    volume = D.features(
        instruments,
        ["$volume"],
        start_time=start_time,
        end_time=end_time,
        freq=frequency,
    ).reset_index()
    if volume.empty or not {"datetime", "$volume"}.issubset(volume.columns):
        raise ValueError("VWAP warm-up window contains no minute volume evidence")
    volume["datetime"] = pd.to_datetime(volume["datetime"], errors="coerce")
    volume["volume"] = pd.to_numeric(volume["$volume"], errors="coerce")
    volume = volume.dropna(subset=["datetime", "volume"])
    volume = volume[volume["volume"] > 0]
    slots = execution_time_slots(
        trade_date=pd.Timestamp(start_time).date(),
        policy={"slice_minutes": slice_minutes, "max_slices": max_slices},
    )
    slot_names = [item.strftime("%H:%M") for item in slots]
    volume["time"] = volume["datetime"].dt.strftime("%H:%M")
    by_time = volume[volume["time"].isin(slot_names)].groupby("time")["volume"].mean()
    missing = [slot for slot in slot_names if slot not in by_time or not np.isfinite(by_time[slot])]
    if missing:
        raise ValueError(
            "VWAP warm-up evidence is missing configured execution slots: " + ", ".join(missing)
        )
    return [{"time": slot, "weight": float(by_time[slot])} for slot in slot_names]


def _latest_cross_section(frame: pd.DataFrame, when: pd.Timestamp, column: str) -> pd.Series:
    values = frame.copy()
    values["datetime"] = pd.to_datetime(values["datetime"], errors="coerce").dt.tz_localize(None)
    values = values[values["datetime"] <= when]
    if values.empty:
        raise ValueError(f"point-in-time metadata has no {column} values at {when.date()}")
    latest = values["datetime"].max()
    snapshot = values[values["datetime"] == latest]
    result = pd.to_numeric(snapshot[column], errors="coerce")
    result.index = snapshot["instrument"].astype(str)
    if result.index.has_duplicates or result.isna().any():
        raise ValueError(f"point-in-time metadata {column} is duplicated or incomplete")
    return result.astype(float)


def _qlib_cross_section(frame: pd.DataFrame, when: pd.Timestamp, column: str) -> pd.Series:
    values = frame.copy()
    dates = pd.to_datetime(values.index.get_level_values("datetime")).tz_localize(None)
    values.index = pd.MultiIndex.from_arrays(
        [dates, values.index.get_level_values("instrument").astype(str)],
        names=["datetime", "instrument"],
    )
    values = values.sort_index()
    available = values.loc[(slice(None, when), slice(None)), column]
    if available.empty:
        raise ValueError(f"Qlib has no {column} data at {when.date()}")
    return pd.to_numeric(
        available.xs(available.index.get_level_values("datetime").max()), errors="coerce"
    ).astype(float)


class _PreparedQlibCrossSections:
    """Own one normalized panel; select the latest whole-date slice per query.

    The source and returned Series may be changed by their callers without
    changing this lookup. No panel is cached outside its metadata provider.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        values = frame.copy()
        dates = pd.to_datetime(values.index.get_level_values("datetime")).tz_localize(None)
        values.index = pd.MultiIndex.from_arrays(
            [dates, values.index.get_level_values("instrument").astype(str)],
            names=["datetime", "instrument"],
        )
        self._values = values.sort_index()
        self._dates = self._values.index.get_level_values("datetime").unique()
        self._monotonic = self._values.index.is_monotonic_increasing

    def lookup(self, when: pd.Timestamp, column: str) -> pd.Series:
        if not self._monotonic:
            # For example, NaT index entries retain the original MultiIndex
            # slicing failure rather than silently dropping invalid dates.
            available = self._values.loc[(slice(None, when), slice(None)), column]
            if available.empty:
                raise ValueError(f"Qlib has no {column} data at {when.date()}")
            snapshot = available.xs(available.index.get_level_values("datetime").max())
        else:
            # Resolve the column before the empty-date guard, as the original
            # helper does. DatetimeIndex slicing preserves inclusive/partial
            # date and timezone comparison semantics without copying history.
            values = self._values[column]
            date_slice = self._dates.slice_indexer(end=when)
            available_dates = self._dates[date_slice]
            if available_dates.empty:
                raise ValueError(f"Qlib has no {column} data at {when.date()}")
            snapshot = values.xs(available_dates[-1], level="datetime")
        return pd.to_numeric(snapshot, errors="coerce").astype(float)


class _PreparedQlibFactorHistory:
    """Keep past-only account conversion factors separate from quote availability.

    A held adjusted position survives a missing quote. Its latest known positive
    factor can still convert its stored amount and valuation into matching share
    equivalents. This history never supplies an open/close or makes it tradable.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        factors = pd.to_numeric(frame["$factor"], errors="coerce").copy()
        dates = pd.to_datetime(frame.index.get_level_values("datetime")).tz_localize(None)
        if dates.isna().any():
            raise ValueError("Qlib factor history has an unknown observation time")
        factors.index = pd.MultiIndex.from_arrays(
            [dates, frame.index.get_level_values("instrument").astype(str)],
            names=["datetime", "instrument"],
        )
        # Unstack rejects duplicate observations. Preserve the provider's
        # float32 storage; only returned cross sections are converted to float.
        factors = factors.where(np.isfinite(factors) & factors.gt(0))
        self._values = factors.unstack("instrument").sort_index().ffill()

    def lookup(self, when: pd.Timestamp, instruments: pd.Index) -> pd.Series:
        if pd.isna(when):
            raise ValueError("Qlib factor lookup has an unknown signal time")
        dates = self._values.index
        available = dates[dates.slice_indexer(end=when)]
        if available.empty:
            return pd.Series(np.nan, index=instruments.astype(str), name="$factor", dtype=float)
        snapshot = self._values.loc[available[-1]].reindex(instruments.astype(str))
        return snapshot.astype(float).rename("$factor")


def _portfolio_return_covariance(
    strategy_config: dict[str, Any],
    close_matrix: pd.DataFrame,
    risk_instruments: pd.Index,
) -> pd.DataFrame | None:
    """Build optimizer risk only for policies that actually consume it."""

    if (
        str(strategy_config.get("portfolio_construction") or "")
        not in _COVARIANCE_REQUIRED_PORTFOLIO_CONSTRUCTIONS
    ):
        return None
    history = close_matrix.reindex(columns=risk_instruments).tail(61)
    returns = history.pct_change(fill_method=None).dropna(how="any")
    if len(returns) < 60:
        raise ValueError("optimizer requires 60 complete point-in-time return observations")
    return estimate_covariance(returns)


def _market_trend_close_history(
    data_api: Any,
    *,
    strategy_config: dict[str, Any],
    start_time: str,
    end_time: str,
) -> pd.DataFrame | None:
    return load_market_trend_close_history(
        data_api,
        config=strategy_config,
        start_time=start_time,
        end_time=end_time,
    )


def _execution_quantity_mode(execution_method: str, *, historical_proxy: bool = False) -> str:
    """Resolve account units from the effective executor, not the signal frequency."""
    if execution_method not in {"open", "twap", "vwap", "next_bar"}:
        raise ValueError("unsupported Qlib execution quantity mode")
    return "daily_adjusted" if historical_proxy or execution_method == "open" else "minute_raw"


def _metadata_provider(
    memberships: pd.DataFrame,
    benchmark_weights: pd.DataFrame | None,
    styles: pd.DataFrame,
    eligibility_matrix: pd.DataFrame,
    execution_metadata: pd.DataFrame,
    close_history: pd.DataFrame,
    benchmark_close_history: pd.DataFrame | None,
    *,
    strategy_config: dict[str, Any],
    open_field: str = "$open/$factor",
    close_field: str = "$close/$factor",
    intraday_prices: pd.DataFrame | None = None,
):
    membership = memberships.copy()
    membership["in_date"] = pd.to_datetime(membership["in_date"], errors="coerce")
    membership["out_date"] = pd.to_datetime(membership["out_date"], errors="coerce")
    close_matrix = close_history["$close"].unstack("instrument").sort_index()
    execution_lookup = _PreparedQlibCrossSections(execution_metadata)
    factor_lookup = _PreparedQlibFactorHistory(execution_metadata)
    risk_lookup = PreparedPointInTimeRiskStates(eligibility_matrix)
    price_lookup = (
        _PreparedQlibCrossSections(intraday_prices)
        if intraday_prices is not None
        else execution_lookup
    )
    # YoY growth is undefined until a company has had the chance to publish
    # one annual report; the systemic style gate is therefore evaluated only
    # over candidates with a full year of dataset history.  Younger
    # candidates keep their neutral (zero) style imputation.
    _eligibility_first_datetime = eligibility_matrix.groupby("instrument")[
        "datetime"
    ].min()
    _eligibility_calendar_index = {
        day: index
        for index, day in enumerate(sorted(eligibility_matrix["datetime"].unique()))
    }

    def _mature_style_scope(instruments: pd.Index, timestamp: pd.Timestamp) -> list[str]:
        position = _eligibility_calendar_index.get(timestamp)
        if position is None:
            return [str(item) for item in instruments]
        mature: list[str] = []
        for instrument in instruments:
            first = _eligibility_first_datetime.get(str(instrument))
            if first is None:
                continue
            first_position = _eligibility_calendar_index.get(first)
            if (
                first_position is not None
                and position - first_position + 1 >= 252
            ):
                mature.append(str(instrument))
        return mature or [str(item) for item in instruments]

    def provide(
        when: Any, instruments: pd.Index, *, execution_quantity_mode: str = "daily_adjusted",
    ) -> dict[str, Any]:
        if execution_quantity_mode not in {"daily_adjusted", "minute_raw"}:
            raise ValueError("unsupported Qlib execution quantity mode")
        timestamp = pd.Timestamp(when).tz_localize(None)
        market_timestamp = (
            timestamp.normalize() - pd.Timedelta(nanoseconds=1)
            if timestamp != timestamp.normalize()
            else timestamp
        )
        # Read-side availability guard (design draft 3.3): membership intervals
        # and weight snapshots become usable only after the versioned
        # conservative publication lag from the shared registry policy.
        active = (
            filter_available("index_member_all", membership, market_timestamp)
            .sort_values("in_date")
            .drop_duplicates("instrument", keep="last")
        )
        industries = active.set_index(active["instrument"].astype(str))["industry"].astype(str)
        raw_style, style = policy_style_cross_sections(
            styles,
            market_timestamp,
            config=strategy_config,
            required_instruments=_mature_style_scope(instruments, market_timestamp),
        )
        constrained = (
            str(strategy_config.get("portfolio_construction") or "")
            in _COVARIANCE_REQUIRED_PORTFOLIO_CONSTRUCTIONS
        )
        benchmark = (
            _latest_cross_section(
                filter_available("index_weight", benchmark_weights, market_timestamp),
                market_timestamp,
                "weight",
            )
            if constrained and benchmark_weights is not None
            else None
        )
        risk_instruments = instruments.astype(str)
        if constrained and benchmark is not None:
            risk_instruments = risk_instruments.union(benchmark.index.astype(str))
        return_covariance = _portfolio_return_covariance(
            strategy_config,
            close_matrix.loc[:market_timestamp],
            risk_instruments,
        )
        portfolio_metadata = build_portfolio_policy_runtime_metadata(
            strategy_config,
            instruments=instruments,
            industries=industries,
            benchmark_weights=benchmark,
            style_exposures=style,
            return_covariance=return_covariance,
        )
        risk_projection = risk_lookup.project(
            as_of=market_timestamp,
            instruments=instruments,
        ).set_index("instrument")
        execution_prices = price_lookup.lookup(
            timestamp if intraday_prices is not None else market_timestamp,
            "$vwap" if intraday_prices is not None else open_field,
        ).reindex(instruments.astype(str))
        current_prices = price_lookup.lookup(
            timestamp if intraday_prices is not None else market_timestamp,
            "$close" if intraday_prices is not None else close_field,
        ).reindex(instruments.astype(str))
        average_daily_values = execution_lookup.lookup(
            market_timestamp,
            "Ref(Mean($amount, 20), 1)",
        ).reindex(instruments.astype(str))
        non_tradable = ~risk_projection["tradable"].astype(bool)
        if non_tradable.any():
            blocked = risk_projection.index[non_tradable]
            execution_prices.loc[blocked] = np.nan
            average_daily_values.loc[blocked] = np.nan
        result = {
            **portfolio_metadata,
            "prices": execution_prices,
            "current_prices": current_prices,
            # Account units belong to the actual Exchange. A daily signal can
            # execute in raw minute shares, whereas its historical daily proxy
            # uses adjusted amounts. Signal-price sampling is unchanged.
            "qlib_factors": (
                pd.Series(1.0, index=instruments.astype(str))
                if execution_quantity_mode == "minute_raw"
                else factor_lookup.lookup(market_timestamp, instruments)
            ),
            # $amount is CNY yuan under the v3 daily field contract.
            "average_daily_values": average_daily_values,
            "instrument_risk_states": risk_projection["risk_state"],
        }
        result.update(
            build_strategy_rule_runtime_metadata(
                strategy_config,
                instruments=instruments,
                close_history=close_matrix.loc[:market_timestamp],
                benchmark_close_history=(
                    benchmark_close_history.loc[:market_timestamp]
                    if benchmark_close_history is not None
                    else None
                ),
                value_exposures=(
                    raw_style["value"] if "value" in raw_style.columns else None
                ),
            )
        )
        return result

    return provide


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--execution-provider-uri")
    parser.add_argument("--execution-frequency")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tracking-uri", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    evaluation_mode = _evaluation_mode(manifest)
    _require_health_reference_scope(output, evaluation_mode)
    consumed_historical_replay = (
        evaluation_mode == CONSUMED_HISTORICAL_REPLAY_MODE
    )
    historical_window_opened = evaluation_mode in {
        FORMAL_FINAL_OOS_MODE,
        CONSUMED_HISTORICAL_REPLAY_MODE,
    }
    provider_provenance_path = Path(args.provider_uri) / "metadata" / "provenance.json"
    if not provider_provenance_path.exists():
        raise ValueError("formal Qlib backtest requires dataset provenance metadata")
    provider_provenance = json.loads(provider_provenance_path.read_text(encoding="utf-8"))
    require_daily_qlib_contract(provider_provenance)
    verify_qlib_output_manifest(Path(args.provider_uri), provider_provenance)
    periods = manifest["periods"]
    require_native_daily_execution_controls(provider_provenance, start=periods["start"])
    factor_value_hashes = {
        str(item["candidate_id"]): _sha256_file(item["values_path"]) for item in manifest["factors"]
    }
    factor_code_hashes = {
        str(item["candidate_id"]): item.get("code_sha256") for item in manifest["factors"]
    }

    config = manifest["config"]
    # Recheck inside the actual evaluation process.  The parent worker already
    # verifies the same binding before launch; this second boundary proves that
    # the executed Python environment inherited the release-stamped image ID.
    require_transparent_baseline_runner(
        config=config,
        job_payload=manifest,
        runner_path=Path(__file__).resolve(),
    )
    worker_runtime_image_digest = manifest.get(
        TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD
    )
    signal_source = str(config.get("signal_source") or "factor_score")
    if evaluation_mode == PRE_FINAL_PORTFOLIO_TRIAL_MODE and signal_source != (
        "model_prediction"
    ):
        raise ValueError("pre-final portfolio trial mode requires an admitted model signal")
    model_bundle_factors = manifest.get("model_bundle_factors") or []
    if not isinstance(model_bundle_factors, list) or any(
        not isinstance(item, dict) for item in model_bundle_factors
    ):
        raise ValueError("formal model bundle factor manifest is invalid")
    if model_bundle_factors and (
        signal_source != "model_prediction" or config.get("quant_bundle_candidate_id") is None
    ):
        raise ValueError("only a joint model strategy may consume bundle factors")
    formal_factor_items = [*manifest["factors"], *model_bundle_factors]
    formal_factor_ids = [str(item.get("candidate_id") or "") for item in formal_factor_items]
    if any(not item for item in formal_factor_ids) or len(set(formal_factor_ids)) != len(
        formal_factor_ids
    ):
        raise ValueError("formal factor membership contains an invalid or duplicate id")
    strategy_contract = require_strategy_execution_contract(config)
    factor_source_mode = str(config.get("factor_source_mode") or FACTOR_SOURCE_PROMOTED_ONLY)
    signal_frequency = str(config.get("signal_frequency") or "day")
    execution_method = str(config.get("execution_method", "open"))
    minute_execution = execution_method in {"twap", "vwap", "next_bar"}
    minute_signal = signal_frequency != "day"
    if minute_signal and execution_method != "next_bar":
        raise ValueError("minute signals currently require the Qlib next_bar execution adapter")
    configured_execution_frequency = str(config.get("execution_frequency") or "day")
    if minute_execution and args.execution_frequency != configured_execution_frequency:
        raise ValueError(
            "execution dataset frequency does not match the immutable strategy contract"
        )
    if not minute_execution and configured_execution_frequency != "day":
        raise ValueError("daily open execution requires a day strategy execution frequency")
    execution_provenance: dict[str, Any] = {}
    if minute_execution:
        if not args.execution_provider_uri or not args.execution_frequency:
            raise ValueError("minute formal backtests require a minute Qlib dataset")
        execution_provenance_path = (
            Path(args.execution_provider_uri) / "metadata" / "provenance.json"
        )
        if not execution_provenance_path.exists():
            raise ValueError("minute execution Qlib dataset requires provenance metadata")
        execution_provenance = json.loads(execution_provenance_path.read_text(encoding="utf-8"))
        require_minute_execution_contract(execution_provenance, frequency=args.execution_frequency)
        verify_qlib_output_manifest(Path(args.execution_provider_uri), execution_provenance)
    import qlib
    from qlib.data import D

    qlib_runtime = upstream_runtime_identity("qlib")

    provider_uri: str | dict[str, str] = args.provider_uri
    if minute_execution:
        provider_uri = {
            "day": args.provider_uri,
            str(args.execution_frequency): str(args.execution_provider_uri),
        }
    qlib_kernels = int(os.environ.get("QUANTLAB_QLIB_KERNELS", "1") or "1")
    if not 1 <= qlib_kernels <= 64:
        raise ValueError("QUANTLAB_QLIB_KERNELS must be between 1 and 64")
    # Expression evaluation is embarrassingly parallel across instruments;
    # the sequential account loop below stays single-core by design.
    qlib.init(provider_uri=provider_uri, region="cn", kernels=qlib_kernels)
    historical_periods = manifest.get("historical_validation_periods")
    if not isinstance(historical_periods, dict) or set(historical_periods) != {
        "start",
        "end",
    }:
        raise ValueError("formal backtest manifest requires isolated pre-final history periods")
    data_periods = {
        "start": historical_periods["start"],
        "end": periods["end"],
    }
    final_calendar = D.calendar(
        start_time=periods["start"],
        end_time=periods["end"],
        freq="day",
    )
    challenger_entries: list[tuple[str, pd.DataFrame, float, int]] = []
    formal_factor_hashes: dict[str, str] = {}
    formal_factor_evidence: dict[str, dict[str, Any]] = {}
    formal_bundle_factor_hashes: dict[str, str] = {}
    formal_bundle_factor_evidence: dict[str, dict[str, Any]] = {}
    bundle_factor_ids = {str(item["candidate_id"]) for item in model_bundle_factors}
    if formal_factor_items:
        factor_input = normalize_factor_input(
            D.features(
                D.instruments(str(manifest.get("universe") or "cn_all")),
                ["$open", "$close", "$high", "$low", "$volume", "$factor"],
                start_time=data_periods["start"],
                end_time=data_periods["end"],
                freq="day",
            )
            .swaplevel()
            .sort_index()
        )
        factor_root = output / "formal-factor-values"
        # Failed jobs are retryable under the same immutable backtest id.  A
        # partial prior execution is not admissible evidence, so rebuild this
        # narrowly scoped derived directory from the frozen manifest/code.
        if factor_root.exists():
            shutil.rmtree(factor_root)
        factor_root.mkdir(parents=True, exist_ok=False)
        factor_input_path = factor_root / "daily_pv.h5"
        factor_input.to_hdf(factor_input_path, key="data", mode="w")
        factor_input_sha256 = sha256_file(factor_input_path)
        for item in formal_factor_items:
            candidate_id = str(item["candidate_id"])
            safe_id = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in candidate_id
            )[:100]
            if not safe_id:
                raise ValueError("formal factor candidate id is invalid")
            candidate_root = factor_root / safe_id
            execution_mode = str(item.get("factor_execution_mode") or "")
            if execution_mode == "frozen_code_recompute":
                code_path = Path(str(item.get("code_path") or ""))
                expected_code_sha256 = str(item.get("code_sha256") or "")
                if not code_path.is_file() or sha256_file(code_path) != expected_code_sha256:
                    raise ValueError(
                        f"formal factor {candidate_id} frozen code is missing or changed"
                    )
                values, execution_evidence = execute_factor_code(
                    code_path=code_path,
                    input_path=factor_input_path,
                    workspace=candidate_root / "full",
                    timeout_seconds=int(manifest.get("factor_recompute_timeout_seconds", 300)),
                )
                values = require_exact_factor_index(
                    values,
                    factor_input,
                    context=f"formal factor {candidate_id}",
                )
                pit_evidence = validate_factor_prefix_invariance(
                    code_path=code_path,
                    input_path=factor_input_path,
                    full_values=values,
                    workspace_root=candidate_root / "prefix-checks",
                    timeout_seconds=int(manifest.get("factor_recompute_timeout_seconds", 300)),
                    cutpoint_count=int(manifest.get("factor_pit_cutpoint_count", 3)),
                )
                coverage_evidence = require_exact_oos_coverage(
                    values,
                    factor_input,
                    test_start=periods["start"],
                    test_end=periods["end"],
                    trading_days=final_calendar,
                    context=f"formal factor {candidate_id}",
                    min_daily_finite=int(manifest.get("min_daily_instruments", 50)),
                )
                execution_evidence.update(
                    {
                        "dataset_identity_sha256": provider_provenance.get(
                            "dataset_identity_sha256"
                        ),
                        "provider_input_sha256": factor_input_sha256,
                        "periods": {
                            "warmup_start": data_periods["start"],
                            "test_start": periods["start"],
                            "test_end": periods["end"],
                        },
                        "pit_invariance": pit_evidence,
                        "oos_coverage": coverage_evidence,
                    }
                )
            elif execution_mode == "frozen_values":
                values = require_exact_factor_index(
                    _load(item["values_path"]),
                    factor_input,
                    context=f"formal frozen-value factor {candidate_id}",
                )
                coverage_evidence = require_exact_oos_coverage(
                    values,
                    factor_input,
                    test_start=periods["start"],
                    test_end=periods["end"],
                    trading_days=final_calendar,
                    context=f"formal frozen-value factor {candidate_id}",
                    min_daily_finite=int(manifest.get("min_daily_instruments", 50)),
                )
                execution_evidence = {
                    "executor_version": "frozen-values-index-exact-v1",
                    "dataset_identity_sha256": provider_provenance.get("dataset_identity_sha256"),
                    "provider_input_sha256": factor_input_sha256,
                    "periods": {
                        "warmup_start": data_periods["start"],
                        "test_start": periods["start"],
                        "test_end": periods["end"],
                    },
                    "oos_coverage": coverage_evidence,
                }
            else:
                raise ValueError(f"formal factor {candidate_id} has no governed execution mode")
            values_path = candidate_root / "authoritative.h5"
            values_path.parent.mkdir(parents=True, exist_ok=True)
            values.to_hdf(values_path, key="data", mode="w")
            values_sha256 = sha256_file(values_path)
            execution_evidence["authoritative_values_sha256"] = values_sha256
            item["formal_factor_artifact"] = {
                "path": str(values_path.relative_to(output)).replace("\\", "/"),
                "sha256": values_sha256,
                "execution_mode": execution_mode,
                "evidence": execution_evidence,
            }
            target_hashes = (
                formal_bundle_factor_hashes
                if candidate_id in bundle_factor_ids
                else formal_factor_hashes
            )
            target_evidence = (
                formal_bundle_factor_evidence
                if candidate_id in bundle_factor_ids
                else formal_factor_evidence
            )
            target_hashes[candidate_id] = values_sha256
            target_evidence[candidate_id] = execution_evidence
            challenger_entries.append(
                (
                    candidate_id,
                    values,
                    float(item["weight"]),
                    int(item["direction"]),
                )
            )
    challenger_factors = [
        (values, weight, direction)
        for _candidate_id, values, weight, direction in challenger_entries
    ]
    challenger_scores = (
        compose_factor_scores(challenger_factors, require_exact_index=True)
        if challenger_factors
        else None
    )
    formal_model_artifact: dict[str, Any] | None = None
    formal_model_admission: dict[str, Any] | None = None
    model_scores: pd.Series | None = None
    model_predictions: pd.DataFrame | None = None
    model_feature_set: dict[str, Any] | None = None
    model_label_contract: dict[str, Any] | None = None
    additional_factors_path: Path | None = None
    if signal_source == "model_prediction":
        frozen_model = manifest.get("model_candidate")
        if not isinstance(frozen_model, dict):
            raise ValueError("formal model strategy has no frozen candidate manifest")
        expected_identity = model_signal_identity(config)
        if manifest.get("model_signal") != expected_identity:
            raise ValueError("formal model identity does not match the frozen StrategySpec")
        candidate_manifest = frozen_model.get("candidate_manifest")
        feature_set = frozen_model.get("feature_set")
        if not isinstance(candidate_manifest, dict) or not isinstance(feature_set, dict):
            raise ValueError("formal model candidate manifest is incomplete")
        model_feature_set = dict(feature_set)
        raw_label_contract = frozen_model.get("label_contract")
        if not isinstance(raw_label_contract, dict):
            raise ValueError("formal model candidate has no frozen label contract")
        model_label_contract = dict(raw_label_contract)
        if _canonical_sha256(model_label_contract) != str(
            frozen_model.get("label_contract_sha256") or ""
        ):
            raise ValueError("formal model label contract seal is invalid")
        admission_pre_final_end = (
            str(candidate_manifest.get("pre_final_end") or "")
            if evaluation_mode in PRE_FINAL_EVALUATION_MODES
            else historical_periods["end"]
        )
        formal_model_admission = validate_model_formal_admission_binding(
            manifest.get("model_formal_admission"),
            config=config,
            dataset_identity_sha256=str(
                provider_provenance.get("dataset_identity_sha256") or ""
            ),
            pre_final_end=admission_pre_final_end,
        )
        if (
            candidate_manifest.get("id") != config.get("model_candidate_id")
            or candidate_manifest.get("code_sha256") != config.get("model_code_sha256")
            or candidate_manifest.get("recipe_sha256") != config.get("model_recipe_sha256")
            or candidate_manifest.get("feature_set_definition_sha256")
            != config.get("feature_set_definition_sha256")
            or feature_set.get("id") != config.get("feature_set_id")
            or feature_set.get("definition_sha256") != config.get("feature_set_definition_sha256")
            or candidate_manifest.get("dataset_identity_sha256")
            != provider_provenance.get("dataset_identity_sha256")
        ):
            raise ValueError("formal model candidate does not match the governed run")
        recipe = candidate_manifest.get("recipe")
        if not isinstance(recipe, dict):
            raise ValueError("formal model candidate recipe is missing")
        model_periods = frozen_model.get("training_periods")
        if not isinstance(model_periods, dict) or any(
            not str(model_periods.get(key) or "")
            for key in ("train_start", "train_end", "valid_start", "valid_end", "seed")
        ):
            raise ValueError("formal model candidate has no frozen pre-final training periods")
        if (
            frozen_model.get("primary_profile_id") != "recent_3y"
            or int(model_periods["seed"]) != 11
            or frozen_model.get("refit_policy") != MODEL_REFIT_POLICY
            or frozen_model.get("refit_policy_sha256") != MODEL_REFIT_POLICY_SHA256
            or canonical_model_sha256(frozen_model["refit_policy"])
            != frozen_model["refit_policy_sha256"]
        ):
            raise ValueError("formal model primary cell/refit policy is not governed")
        execution_authorization = _model_execution_authorization(
            evaluation_mode=evaluation_mode,
            candidate_manifest=candidate_manifest,
            model_periods=model_periods,
            periods=periods,
            historical_periods=historical_periods,
            pre_final_cutoff=(
                str(manifest.get("pre_final_cutoff") or "")
                if evaluation_mode in PRE_FINAL_EVALUATION_MODES
                else None
            ),
        )
        execution_model_periods = {
            "train_start": str(model_periods["train_start"]),
            "train_end": str(model_periods["train_end"]),
            "valid_start": str(model_periods["valid_start"]),
            "valid_end": str(model_periods["valid_end"]),
            "test_start": periods["start"],
            "test_end": periods["end"],
        }
        if evaluation_mode in PRE_FINAL_EVALUATION_MODES:
            execution_model_periods = _pre_final_execution_periods(
                model_periods,
                periods,
                D.calendar(
                    start_time=str(model_periods["valid_start"]),
                    end_time=periods["start"],
                    freq="day",
                ),
                embargo_sessions=int(config.get("outer_embargo_days", 5)),
            )
        code_path = Path(str(frozen_model.get("code_path") or ""))
        if config.get("quant_bundle_candidate_id") is not None:
            bundle_contract = config.get("quant_bundle_factor_contract")
            if not isinstance(bundle_contract, dict):
                raise ValueError("joint formal model has no bundle factor contract")
            observed_bundle_factors = [
                {
                    key: item.get(key)
                    for key in (
                        "candidate_id",
                        "feature_name",
                        "code_sha256",
                        "direction",
                        "weight",
                    )
                }
                for item in model_bundle_factors
            ]
            if observed_bundle_factors != bundle_contract.get("factors") or any(
                item.get("factor_execution_mode") != "frozen_code_recompute"
                for item in model_bundle_factors
            ):
                raise ValueError(
                    "joint formal model factors do not match the atomic bundle contract"
                )
            if not challenger_entries:
                raise ValueError("joint formal model strategy has no recomputed factor values")
            if {item[0] for item in challenger_entries} != bundle_factor_ids:
                raise ValueError(
                    "joint formal model cannot mix standalone score factors into its bundle"
                )
            combined = pd.concat(
                [
                    values.iloc[:, 0].rename(f"factor_{index:03d}")
                    for index, (_candidate, values, _weight, _direction) in enumerate(
                        challenger_entries
                    )
                ],
                axis=1,
                join="inner",
            ).sort_index()
            if not combined.index.equals(factor_input.index):
                raise ValueError("joint formal model factors changed the exact Qlib input index")
            additional_factors_path = output / "formal-model-factors.parquet"
            combined.to_parquet(additional_factors_path, compression="zstd")
        model_root = output / "formal-model"
        if model_root.exists():
            shutil.rmtree(model_root)
        formal_model_engine = _frozen_model_engine(
            recipe,
            recipe_sha256=str(candidate_manifest["recipe_sha256"]),
        )
        model_result, model_execution = execute_model_candidate(
            code_path=code_path,
            provider_path=Path(args.provider_uri),
            additional_factors_path=additional_factors_path,
            manifest={
                "candidate_id": str(candidate_manifest["id"]),
                "code_sha256": str(candidate_manifest["code_sha256"]),
                "model_type": str(recipe.get("model_type") or "Tabular"),
                "model_engine": formal_model_engine,
                "training_hyperparameters": recipe.get("training_hyperparameters") or {},
                "feature_set": feature_set,
                "additional_factor_count": len(challenger_entries),
                "periods": execution_model_periods,
                "prediction_segment": execution_authorization["prediction_segment"],
                "seed": int(model_periods["seed"]),
                "dataset_identity_sha256": provider_provenance.get("dataset_identity_sha256"),
                "universe": manifest.get("universe", "cn_all"),
                "benchmark": manifest.get("benchmark", "SH000300"),
                "account": config.get("capacity_notional", 100_000_000),
                "topk": config.get("topk", 50),
                "n_drop": config.get("n_drop", 5),
                "open_cost": config.get("open_cost", 0.0005),
                "close_cost": config.get("close_cost", 0.0015),
                "min_cost": config.get("min_cost", 5.0),
                "final_oos_opened": execution_authorization["final_oos_opened"],
                "inference_only": execution_authorization["allow_inference"],
            },
            workspace=model_root,
            runner_path=Path(__file__).resolve().with_name("model_sandbox_runner.py"),
            allow_final_oos=execution_authorization["allow_final_oos"],
            allow_inference=execution_authorization["allow_inference"],
            timeout_seconds=int(manifest.get("model_timeout_seconds", 7200)),
        )
        admitted_environment_sha256 = str(
            formal_model_admission["model_grid"][
                "execution_environment_sha256"
            ]
        )
        if (
            model_result.get("execution_environment_sha256")
            != admitted_environment_sha256
            or model_execution.get("execution_environment_sha256")
            != admitted_environment_sha256
        ):
            raise ValueError(
                "formal model execution environment differs from independent admission"
            )
        predictions_path = model_root / "output" / "predictions.parquet"
        checkpoint_path = (
            model_root / "output" / governed_checkpoint_filename(formal_model_engine)
        )
        model_coverage = verify_model_prediction_artifact(
            predictions_path,
            expected_sha256=model_result["predictions_sha256"],
            test_start=periods["start"],
            test_end=periods["end"],
            trading_days=final_calendar,
            min_daily_finite=int(manifest.get("min_daily_instruments", 50)),
        )
        model_execution.update(
            {
                "dataset_identity_sha256": provider_provenance.get("dataset_identity_sha256"),
                "test_start": periods["start"],
                "test_end": periods["end"],
                "oos_coverage": model_coverage,
            }
        )
        model_predictions = normalize_model_predictions(pd.read_parquet(predictions_path))
        prediction_dates = pd.DatetimeIndex(
            model_predictions.index.get_level_values("datetime")
        ).normalize()
        model_predictions = model_predictions.loc[
            (prediction_dates >= pd.Timestamp(periods["start"]))
            & (prediction_dates <= pd.Timestamp(periods["end"]))
        ]
        model_scores = model_predictions["score"].rename("score")
        formal_model_artifact = {
            "predictions_path": str(predictions_path.relative_to(output)).replace("\\", "/"),
            "predictions_sha256": str(model_result["predictions_sha256"]),
            "checkpoint_path": str(checkpoint_path.relative_to(output)).replace("\\", "/"),
            "checkpoint_sha256": str(model_result["checkpoint_sha256"]),
            "checkpoint_format": str(model_result["checkpoint_format"]),
            "model_data_contract": model_result["model_data_contract"],
            "model_data_contract_sha256": str(
                model_result["model_data_contract_sha256"]
            ),
            "additional_factors_path": (
                str(additional_factors_path.relative_to(output)).replace("\\", "/")
                if additional_factors_path is not None
                else None
            ),
            "additional_factors_sha256": (
                _sha256_file(additional_factors_path)
                if additional_factors_path is not None
                else None
            ),
            "evidence": model_execution,
        }
        manifest["formal_model_artifact"] = formal_model_artifact
    elif manifest.get("model_formal_admission") is not None:
        raise ValueError("factor-score backtest cannot carry model admission evidence")
    baseline_artifacts: dict[str, Any] | None = None
    baseline_definition = config.get("baseline_definition")
    baseline_raw: pd.DataFrame | None = None
    baseline_normalized: pd.DataFrame | None = None
    baseline_scores: pd.Series | None = None
    if isinstance(baseline_definition, dict):
        baseline_raw, baseline_normalized, baseline_scores = _recompute_qlib_baseline(
            D,
            universe=str(manifest.get("universe") or "cn_all"),
            definition=baseline_definition,
            start_time=data_periods["start"],
            end_time=data_periods["end"],
        )
        baseline_artifacts = _write_baseline_artifacts(
            output,
            raw=baseline_raw,
            normalized=baseline_normalized,
            composite=baseline_scores,
        )
        manifest["baseline"] = {
            "definition": baseline_definition,
            "definition_sha256": config.get("baseline_definition_sha256"),
            "computed_by": "qlib.data.D.features",
            "artifacts": baseline_artifacts,
        }
        baseline_scores = apply_strategy_rule_alpha_weights(
            baseline_normalized,
            baseline_scores,
            config,
        )
    if signal_source == "model_prediction":
        if model_scores is None:
            raise ValueError("formal model strategy produced no final-OOS scores")
        scores = model_scores
    elif factor_source_mode == FACTOR_SOURCE_PROMOTED_ONLY:
        if challenger_scores is None:
            raise ValueError("a promoted-only strategy has no challenger factor values")
        scores = challenger_scores
    else:
        if baseline_scores is None:
            raise ValueError("a core strategy has no governed Qlib baseline definition")
        scores = combine_factor_sources(
            mode=factor_source_mode,
            baseline=baseline_scores,
            challenger=challenger_scores,
            challenger_weight=float(config.get("challenger_weight") or 0.0),
        )
    eligibility_path = Path(args.provider_uri) / "metadata" / "eligibility_matrix.parquet"
    if not eligibility_path.is_file():
        raise ValueError("Qlib daily dataset has no point-in-time eligibility matrix")
    eligibility_matrix = pd.read_parquet(eligibility_path)
    instruments = _eligible_strategy_instruments(scores, eligibility_matrix)
    if minute_execution:
        available_instruments = _qlib_instruments(args.execution_provider_uri)
        missing_instruments = sorted(set(instruments) - available_instruments)
        if missing_instruments:
            preview = ", ".join(missing_instruments[:10])
            raise ValueError(
                "minute execution dataset is missing "
                f"{len(missing_instruments)} strategy instruments: " + preview
            )
    # $amount is CNY yuan under the v3 daily field contract.
    liquidity_amount = D.features(
        instruments,
        ["$amount"],
        start_time=data_periods["start"],
        end_time=data_periods["end"],
        freq="day",
    )
    open_field = "$open/$factor"
    close_field = "$close/$factor"
    execution_metadata = D.features(
        instruments,
        [open_field, close_field, "$factor", "Ref(Mean($amount, 20), 1)"],
        start_time=data_periods["start"],
        end_time=data_periods["end"],
        freq="day",
    )
    covariance_start = (
        (pd.Timestamp(data_periods["start"]) - pd.Timedelta(days=120)).date().isoformat()
    )
    close_history = D.features(
        instruments,
        ["$close"],
        start_time=covariance_start,
        end_time=data_periods["end"],
        freq="day",
    )
    benchmark_close_history = _market_trend_close_history(
        D,
        strategy_config=config,
        start_time=data_periods["start"],
        end_time=data_periods["end"],
    )
    intraday_prices = (
        D.features(
            instruments,
            ["$vwap", "$close"],
            start_time=periods["start"],
            end_time=periods["end"],
            freq=str(args.execution_frequency),
        )
        if minute_signal
        else None
    )
    industry_path = Path(args.provider_uri) / "metadata" / "industry_memberships.parquet"
    industry_memberships = pd.read_parquet(industry_path) if industry_path.exists() else None
    industry_cap_enabled = float(manifest["config"].get("max_industry_weight", 1.0)) < 1.0
    if industry_cap_enabled and industry_memberships is None:
        raise ValueError("industry-constrained backtest requires point-in-time industry metadata")
    constrained = (
        str(config.get("portfolio_construction") or "")
        in _COVARIANCE_REQUIRED_PORTFOLIO_CONSTRUCTIONS
    )
    if config.get("portfolio_construction") == "industry_neutral_qp":
        target_weight_path = Path(args.provider_uri) / "metadata" / "full_market_weights.parquet"
        benchmark_weights = (
            pd.read_parquet(target_weight_path) if target_weight_path.exists() else None
        )
        target_weight_label = "full-market float-cap"
    elif constrained:
        target_weight_path = Path(args.provider_uri) / "metadata" / "benchmark_weights.parquet"
        benchmark_weights = (
            pd.read_parquet(target_weight_path) if target_weight_path.exists() else None
        )
        if benchmark_weights is not None:
            benchmark_weights = benchmark_weights[
                benchmark_weights["benchmark"] == manifest["benchmark"]
            ].drop(columns=["benchmark"])
        target_weight_label = "index benchmark"
    else:
        benchmark_weights = None
        target_weight_label = "unused reporting benchmark"
    style_exposures, style_exposure_evidence = _load_governed_style_exposures(args.provider_uri)
    if constrained and (benchmark_weights is None or benchmark_weights.empty):
        raise ValueError(f"constrained backtest requires historical {target_weight_label} weights")
    if style_exposures.empty:
        raise ValueError("index-enhancement backtest requires point-in-time style exposures")
    eligibility_evidence = eligibility_statistics(eligibility_matrix)
    if (
        config.get("require_regulatory_events")
        and not eligibility_evidence["regulatory_data_available"]
    ):
        raise ValueError("strategy requires regulatory events but no reliable source is available")

    def governed_for(
        scenario_config: dict[str, Any],
        signal_scores: pd.Series | None = None,
    ) -> pd.Series:
        scenario_policy_config = PortfolioPolicyConfig.from_mapping(scenario_config)
        neutralize_industry, neutralize_styles = governed_score_neutralization(
            scenario_config
        )
        return build_governed_signal(
            scores if signal_scores is None else signal_scores,
            topk=scenario_policy_config.topk,
            n_drop=scenario_policy_config.n_drop,
            liquidity_amount=liquidity_amount,
            industry_memberships=industry_memberships,
            benchmark_weights=benchmark_weights,
            style_exposures=style_exposures,
            eligibility_matrix=eligibility_matrix,
            max_position_weight=scenario_policy_config.max_position_weight,
            max_industry_weight=scenario_policy_config.max_industry_weight,
            max_industry_deviation=scenario_policy_config.max_industry_deviation,
            min_average_daily_amount=float(scenario_config.get("min_average_daily_amount", 0.0)),
            liquidity_lookback_days=int(scenario_config.get("liquidity_lookback_days", 20)),
            neutralize_industry=neutralize_industry,
            neutralize_style_columns=neutralize_styles,
            benchmark_relative_industry_constraints=(
                scenario_config.get("portfolio_construction")
                in _COVARIANCE_REQUIRED_PORTFOLIO_CONSTRUCTIONS
            ),
        )

    governed_signal = governed_for(config)
    cost_schedule = CostScheduleBook.from_mapping(config)
    # PortfolioPolicy only consumes date-independent broker assumptions (lot size
    # and participation), resolved here at the backtest start date.
    policy_costs = cost_schedule.as_of(pd.Timestamp(periods["start"]).date())
    policy = PortfolioPolicy(PortfolioPolicyConfig.from_mapping(config), policy_costs)
    metadata = _metadata_provider(
        industry_memberships,
        benchmark_weights,
        style_exposures,
        eligibility_matrix,
        execution_metadata,
        close_history,
        benchmark_close_history,
        strategy_config=config,
        open_field=open_field,
        close_field=close_field,
        intraday_prices=intraday_prices,
    )
    execution_policy: dict[str, Any] | None = None
    vwap_profile_evidence: dict[str, Any] | None = None
    if minute_execution:
        slice_minutes = int(config.get("execution_slice_minutes", 20))
        max_slices = int(config.get("max_execution_slices", 24))
        volume_profile = None
        if execution_method == "vwap":
            lookback_days = int(config.get("vwap_lookback_days", 20))
            warmup_start, warmup_end = _minute_warmup_window(
                args.execution_provider_uri,
                args.execution_frequency,
                periods["start"],
                lookback_days,
            )
            volume_profile = _historical_vwap_profile(
                instruments=instruments,
                start_time=warmup_start,
                end_time=warmup_end,
                frequency=args.execution_frequency,
                slice_minutes=slice_minutes,
                max_slices=max_slices,
            )
            vwap_profile_evidence = {
                "start": warmup_start,
                "end": warmup_end,
                "trading_days": lookback_days,
                "profile_sha256": _canonical_sha256(volume_profile),
                "future_data_used": False,
            }
        execution_policy = {
            "execution_algorithm": execution_method,
            "slice_minutes": slice_minutes,
            "max_slices": max_slices,
            "max_participation": policy_costs.max_volume_participation,
            "volume_profile": volume_profile,
        }

    def run(
        start: str,
        end: str,
        costs: CostScheduleBook,
        account: float | None = None,
        scenario_config: dict[str, Any] | None = None,
        signal_scores: pd.Series | None = None,
        historical_proxy: bool = False,
    ):
        if historical_proxy and signal_frequency != "day":
            raise ValueError(
                "pre-final long-history validation supports daily signals only; "
                "minute signals require separately bounded recent evidence"
            )
        effective_config = {**config, **(scenario_config or {})}
        scenario_policy = PortfolioPolicy(
            PortfolioPolicyConfig.from_mapping(effective_config),
            costs.as_of(pd.Timestamp(start).date()),
        )
        strategy = create_qlib_policy_strategy(
            signal=(
                governed_for(effective_config, signal_scores)
                if scenario_config or signal_scores is not None
                else governed_signal
            ),
            policy=scenario_policy,
            metadata_provider=partial(
                metadata,
                execution_quantity_mode=_execution_quantity_mode(
                    execution_method, historical_proxy=historical_proxy,
                ),
            ),
        )
        return run_formal_qlib_backtest(
            strategy=strategy,
            start_time=start,
            end_time=end,
            account=float(account if account is not None else config["capacity_notional"]),
            benchmark=manifest["benchmark"],
            cost_schedule=costs,
            execution_method=("open" if historical_proxy else execution_method),
            signal_frequency=signal_frequency,
            execution_frequency=(
                None
                if historical_proxy
                else (args.execution_frequency if minute_execution else None)
            ),
            execution_policy=None if historical_proxy else execution_policy,
            instruments=instruments,
            annual_minimum_acceptable_return=float(
                effective_config.get("annual_minimum_acceptable_return", 0.0)
            ),
        )

    formal = run(periods["start"], periods["end"], cost_schedule)
    pre_final_calendar = pd.DatetimeIndex(
        D.calendar(
            start_time=historical_periods["start"],
            end_time=periods["start"],
            freq="day",
        )
    )
    history_calendar = pre_final_calendar[
        pre_final_calendar <= pd.Timestamp(historical_periods["end"])
    ]
    if evaluation_mode in PRE_FINAL_EVALUATION_MODES:
        pre_final_history = {
            "status": "not_applicable_pre_final_portfolio_trial",
            "scope": "selection_only",
            "requested_start": historical_periods["start"],
            "requested_end": historical_periods["end"],
            "trial_start": periods["start"],
            "trial_end": periods["end"],
            "pre_final_cutoff": str(manifest["pre_final_cutoff"]),
            "final_oos_opened": False,
        }
    else:
        pre_final_history = build_pre_final_history_evidence(
            pre_final_calendar,
            requested_start=historical_periods["start"],
            requested_end=historical_periods["end"],
            final_test_start=periods["start"],
            final_test_end=periods["end"],
            minimum_trading_days=int(config.get("min_pre_final_history_days", 2520)),
            minimum_embargo_trading_days=int(config.get("outer_embargo_days", 5)),
        )
    pre_final_history["execution_model"] = {
        "method": "open",
        "frequency": "day",
        "scope": "pre_final_signal_and_portfolio_stability_proxy",
        "minute_execution_claimed": False,
    }

    def write_robustness_artifacts(name: str, result: Any) -> dict[str, Any]:
        target = output / "robustness" / name
        target.mkdir(parents=True, exist_ok=True)
        report_path = target / "daily_report.parquet"
        fills_path = target / "fills.parquet"
        metrics_path = target / "metrics.json"
        result.report.to_parquet(report_path, compression="zstd")
        pd.DataFrame(
            result.fills,
            columns=[
                "instrument",
                "date",
                "side",
                "requested_amount",
                "amount",
                "capacity_fill_ratio",
                "trade_price",
                "trade_value",
                "cost",
            ],
        ).to_parquet(fills_path, index=False, compression="zstd")
        metrics_path.write_text(
            json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            key: {
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "sha256": _sha256_file(path),
            }
            for key, path in {
                "daily_report": report_path,
                "fills": fills_path,
                "metrics": metrics_path,
            }.items()
        }

    historical_descriptive_rolling = (
        historical_window_opened
        and str(config.get("horizon_profile") or "legacy_ambiguous")
        != "legacy_ambiguous"
    )
    validation = run_qlib_validation_suites(
        runner=run,
        full_result=formal,
        start_time=periods["start"],
        end_time=periods["end"],
        cost_schedule=cost_schedule,
        config=config,
        capacity_runner=lambda notional: (
            formal
            if abs(notional - float(config["capacity_notional"])) < 1e-6
            else run(periods["start"], periods["end"], cost_schedule, notional)
        ),
        robustness_runner=lambda overrides, costs: run(
            periods["start"], periods["end"], costs, scenario_config=overrides
        ),
        robustness_artifact_writer=write_robustness_artifacts,
        rolling_scope=(
            (
                "consumed_historical_replay_descriptive_only"
                if consumed_historical_replay
                else "sealed_final_oos_descriptive_only"
            )
            if historical_descriptive_rolling
            else "pre_final_stability"
        ),
        rolling_gate_applied=not historical_descriptive_rolling,
    )
    qlib_report = formal.report
    qlib_positions = formal.positions
    net_daily_returns = pd.to_numeric(qlib_report["return"], errors="coerce") - pd.to_numeric(
        qlib_report.get("cost", 0.0), errors="coerce"
    )
    strategy_trial_count = int(manifest.get("strategy_trial_count") or 1)
    trial_count_audit = (
        (manifest.get("hypothesis_group_evidence") or {}).get("trial_count_audit")
        if isinstance(manifest.get("hypothesis_group_evidence"), dict)
        else None
    )
    trial_count_audit_sha256 = (
        _canonical_sha256(trial_count_audit)
        if isinstance(trial_count_audit, dict)
        else None
    )
    incomplete_family_eligibility = None
    if (
        signal_source == "factor_score"
        and strategy_trial_count > 1
        and consumed_historical_replay
    ):
        raw_eligibility = manifest.get("incomplete_factor_family_eligibility")
        if not isinstance(raw_eligibility, dict):
            raise ValueError(
                "multi-trial replay requires frozen incomplete-family eligibility"
            )
        incomplete_family_eligibility = require_incomplete_family_eligibility(
            raw_eligibility,
            hypothesis_group_evidence=dict(
                manifest.get("hypothesis_group_evidence") or {}
            ),
            strategy_version_id=str(manifest["strategy_version_id"]),
        )
    incomplete_factor_family = incomplete_family_eligibility is not None
    if incomplete_factor_family and trial_count_audit_sha256 is None:
        raise ValueError(
            "multi-trial factor formal OOS requires the frozen trial-count audit"
        )

    def write_formal_run_artifacts(category: str, name: str, result: Any) -> dict[str, Any]:
        target = output / "formal-validation" / category / name
        target.mkdir(parents=True, exist_ok=True)
        report_path = target / "daily_report.parquet"
        fills_path = target / "fills.parquet"
        metrics_path = target / "metrics.json"
        result.report.to_parquet(report_path, compression="zstd")
        pd.DataFrame(result.fills).to_parquet(fills_path, index=False, compression="zstd")
        metrics_path.write_text(
            json.dumps(result.metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            key: {
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "sha256": _sha256_file(path),
            }
            for key, path in {
                "daily_report": report_path,
                "fills": fills_path,
                "metrics": metrics_path,
            }.items()
        }

    baseline_weights = {
        str(item["id"]): float(item["weight"])
        for item in (
            baseline_definition.get("factors") or []
            if isinstance(baseline_definition, dict)
            else []
        )
    }
    ablation_components = [
        *(f"baseline:{factor_id}" for factor_id in sorted(baseline_weights)),
        *(
            f"challenger:{candidate_id}"
            for candidate_id, _values, _weight, _direction in challenger_entries
        ),
    ]

    def ablated_scores(component: str) -> pd.Series | None:
        baseline_variant = baseline_scores
        challenger_variant = challenger_scores
        if component.startswith("baseline:"):
            removed = component.split(":", 1)[1]
            remaining = {
                factor_id: weight
                for factor_id, weight in baseline_weights.items()
                if factor_id != removed
            }
            baseline_variant = (
                baseline_normalized.mul(pd.Series(remaining), axis=1).sum(axis=1).rename("score")
                if remaining and baseline_normalized is not None
                else None
            )
        elif component.startswith("challenger:"):
            removed = component.split(":", 1)[1]
            remaining = [
                (values, weight, direction)
                for candidate_id, values, weight, direction in challenger_entries
                if candidate_id != removed
            ]
            challenger_variant = (
                compose_factor_scores(remaining, require_exact_index=True) if remaining else None
            )
        else:
            raise ValueError(f"unknown ablation component: {component}")
        if factor_source_mode == FACTOR_SOURCE_PROMOTED_ONLY:
            return challenger_variant
        if baseline_variant is None:
            return None
        if challenger_variant is None:
            return baseline_variant
        return combine_factor_sources(
            mode=factor_source_mode,
            baseline=baseline_variant,
            challenger=challenger_variant,
            challenger_weight=float(config.get("challenger_weight") or 0.0),
        )

    def run_ablation(component: str) -> dict[str, Any]:
        signal = ablated_scores(component)
        if signal is None:
            # Removing the final alpha component leaves the declared simple
            # benchmark/no-alpha baseline, not an arbitrary Top-K tie break.
            return {
                "annualized_excess_return": 0.0,
                "baseline_state": "no_alpha_component",
                "artifacts": {},
            }
        result = run(
            periods["start"],
            periods["end"],
            cost_schedule,
            signal_scores=signal,
        )
        return {
            **result.metrics,
            "artifacts": write_formal_run_artifacts(
                "ablation", component.replace(":", "-"), result
            ),
        }

    if signal_source == "model_prediction":
        if formal_model_admission is None:
            raise ValueError("formal model admission binding was not validated")
        ablation = {
            "status": MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_STATUS,
            "applicable": False,
            "reason": MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_REASON,
            "validation_kind": "factor_score_component_ablation",
            "final_oos_reopened": False,
            "independent_admission_binding_sha256": formal_model_admission["binding_sha256"],
            "governed_grid_cell_count": (
                (formal_model_admission.get("quant_bundle") or {}).get("cell_count")
                or formal_model_admission["model_grid"]["cell_count"]
            ),
            "runs": [],
        }
    else:
        ablation = run_ablation_suite(
            component_ids=ablation_components,
            full_metrics=formal.metrics,
            runner=run_ablation,
            metric="annualized_excess_return",
            minimum_increment=float(config.get("min_component_increment", 0.0)),
        )

    def delayed_scores(delay: int) -> pd.Series:
        if delay == 0:
            return scores
        shifted = (
            scores.groupby(level="instrument", group_keys=False)
            .shift(delay)
            .dropna()
            .rename("score")
        )
        if shifted.empty:
            raise ValueError("signal decay delay leaves no executable observations")
        return shifted

    def run_delay(delay: int) -> dict[str, Any]:
        if delay == 0:
            return {**formal.metrics, "artifacts": {}}
        result = run(
            periods["start"],
            periods["end"],
            cost_schedule,
            signal_scores=delayed_scores(delay),
        )
        return {
            **result.metrics,
            "artifacts": write_formal_run_artifacts("signal-decay", f"delay-{delay}", result),
        }

    signal_decay = run_signal_decay_suite(
        delays=config.get("signal_decay_delays", [0, 1, 2, 3]),
        runner=run_delay,
        metric="annualized_excess_return",
        minimum_retention=float(config.get("minimum_signal_retention", 0.60)),
    )

    if signal_source == "model_prediction":
        if formal_model_admission is None:
            raise ValueError("formal model admission binding was not validated")
        outer_walk_forward = {
            "status": MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_STATUS,
            "applicable": False,
            "reason": MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_REASON,
            "validation_kind": "factor_score_outer_walk_forward",
            "final_oos_reopened": False,
            "independent_admission_binding_sha256": formal_model_admission["binding_sha256"],
            "governed_grid_cell_count": (
                (formal_model_admission.get("quant_bundle") or {}).get("cell_count")
                or formal_model_admission["model_grid"]["cell_count"]
            ),
        }
    elif strategy_trial_count == 1 or incomplete_factor_family:
        outer_walk_forward = run_outer_walk_forward(
            dates=history_calendar,
            candidate_ids=["frozen-strategy"],
            inner_runner=lambda _candidate, fold: (
                run(
                    fold.validation_start,
                    fold.validation_end,
                    cost_schedule,
                    historical_proxy=True,
                ).metrics
            ),
            test_runner=lambda _candidate, fold: (
                run(
                    fold.test_start,
                    fold.test_end,
                    cost_schedule,
                    historical_proxy=True,
                ).metrics
            ),
            selection_metric="annualized_excess_return",
            train_days=int(config.get("outer_train_days", 252)),
            validation_days=int(config.get("outer_validation_days", 42)),
            test_days=int(config.get("outer_test_days", 42)),
            purge_days=int(config.get("outer_purge_days", 5)),
            embargo_days=int(config.get("outer_embargo_days", 5)),
            minimum_test_metric=float(config.get("minimum_outer_test_excess_return", 0.0)),
            minimum_test_pass_rate=float(config.get("minimum_outer_test_pass_rate", 0.60)),
        )
        outer_walk_forward["candidate_coverage"] = {
            "required_group_trials": strategy_trial_count,
            "provided_candidates": 1,
            "scope": FROZEN_STRATEGY_OUTER_SCOPE,
            "selection_performed": False,
            "historical_candidate_matrix": (
                "incomplete"
                if incomplete_factor_family
                else "not_applicable_single_trial"
            ),
            **(
                {
                    "trial_count_audit_sha256": trial_count_audit_sha256,
                    "eligibility_receipt_sha256": incomplete_family_eligibility[
                        "receipt_sha256"
                    ],
                }
                if incomplete_factor_family
                else {}
            ),
        }
    else:
        outer_walk_forward = {
            "status": "blocked_missing_group_candidate_artifacts",
            "contract_version": FORMAL_VALIDATION_CONTRACT_VERSION,
            "candidate_coverage": {
                "required_group_trials": strategy_trial_count,
                "provided_candidates": 1,
                "scope": "economic_hypothesis_group",
            },
        }

    paired = pd.concat(
        [
            net_daily_returns.rename("candidate"),
            pd.to_numeric(qlib_report["bench"], errors="coerce").rename("baseline"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    paired_bootstrap = build_paired_bootstrap_evidence(
        paired["candidate"],
        paired["baseline"],
        parameters=paired_bootstrap_parameters_from_config(config),
    )
    governed_multiple_testing = None
    expected_governed_trial_names: list[str] = []
    if formal_model_admission is not None:
        governed_multiple_testing = (
            (formal_model_admission.get("quant_bundle") or {}).get("multiple_testing")
            or (formal_model_admission.get("model_grid") or {}).get("multiple_testing")
        )
        trial_audit = (
            (manifest.get("hypothesis_group_evidence") or {}).get("trial_count_audit")
            or {}
        )
        if formal_model_admission.get("quant_bundle") is not None:
            expected_governed_trial_names = sorted(
                f"{candidate_id}:{ablation}"
                for run in trial_audit.get("quant_runs") or []
                for candidate_id in run.get("all_bundle_candidate_ids") or []
                for ablation in REQUIRED_QUANT_ABLATIONS
            )
        else:
            expected_governed_trial_names = sorted(
                str(candidate_id)
                for run in trial_audit.get("model_runs") or []
                for candidate_id in run.get("all_candidate_ids") or []
            )
    if (
        signal_source == "model_prediction"
        and isinstance(governed_multiple_testing, dict)
        and governed_multiple_testing.get("gate_passed") is True
        and int(governed_multiple_testing.get("trial_count") or 0)
        == strategy_trial_count
        and sorted(governed_multiple_testing.get("trial_names") or [])
        == expected_governed_trial_names
        and len(expected_governed_trial_names) == strategy_trial_count
    ):
        multiple_testing = {
            **governed_multiple_testing,
            "status": "ok",
            "evidence_scope": "independent_pre_final_run_trial_family",
            "independent_admission_binding_sha256": formal_model_admission["binding_sha256"],
        }
    elif strategy_trial_count == 1:
        multiple_testing = {
            "status": "not_applicable_single_trial",
            "trial_count": 1,
            "holm_adjusted_p_values": holm_bonferroni([paired_bootstrap["one_sided_p_value"]]),
            "pbo": {
                "status": "not_applicable_single_trial",
                "pbo": None,
            },
        }
    elif incomplete_factor_family:
        multiple_testing = build_factor_score_incomplete_family_multiple_testing(
            paired_bootstrap=paired_bootstrap,
            trial_count=strategy_trial_count,
            trial_count_audit_sha256=str(trial_count_audit_sha256),
            eligibility_receipt_sha256=str(
                incomplete_family_eligibility["receipt_sha256"]
            ),
        )
    else:
        multiple_testing = {
            "status": "blocked_missing_group_candidate_artifacts",
            "trial_count": strategy_trial_count,
            "holm_adjusted_p_values": None,
            "pbo": {
                "status": "blocked_missing_group_candidate_artifacts",
                "pbo": None,
            },
        }
    factor_validation_passed = (
        (
            outer_walk_forward.get("status") == MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_STATUS
            and ablation.get("status") == MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_STATUS
            and formal_model_admission is not None
            and outer_walk_forward.get("independent_admission_binding_sha256")
            == formal_model_admission["binding_sha256"]
            and ablation.get("independent_admission_binding_sha256")
            == formal_model_admission["binding_sha256"]
        )
        if signal_source == "model_prediction"
        else (
            outer_walk_forward.get("status") == "completed"
            and outer_walk_forward.get("passed") is True
            and ablation["status"] == "passed"
        )
    )
    multiple_testing_passed = (
        multiple_testing["status"] == "not_applicable_single_trial"
        or (
            incomplete_factor_family
            and multiple_testing.get("status")
            == CONSERVATIVE_BONFERRONI_INCOMPLETE_FAMILY_STATUS
            and multiple_testing.get("gate_passed") is True
        )
        or (
            signal_source == "model_prediction"
            and multiple_testing.get("status") == "ok"
            and multiple_testing.get("gate_passed") is True
            and multiple_testing.get("final_oos_opened") is False
            and multiple_testing.get("independent_admission_binding_sha256")
            == (formal_model_admission or {}).get("binding_sha256")
        )
    )
    formal_validation = {
        "contract_version": FORMAL_VALIDATION_CONTRACT_VERSION,
        "pre_final_history": pre_final_history,
        "status": (
            "passed"
            if factor_validation_passed
            and signal_decay["maximum_supported_delay_bars"] is not None
            and paired_bootstrap["confidence_interval_95"][0] > 0
            and multiple_testing_passed
            else "failed"
        ),
        "outer_walk_forward": outer_walk_forward,
        "ablation": ablation,
        "signal_decay": signal_decay,
        "paired_block_bootstrap": paired_bootstrap,
        "multiple_testing": multiple_testing,
    }
    if evaluation_mode in PRE_FINAL_EVALUATION_MODES:
        # Selection evidence is useful for ranking the two frozen construction
        # policies, but is never promotion evidence and cannot satisfy approve().
        formal_validation["status"] = "not_applicable_pre_final_only"
        formal_validation["capital_eligible"] = False
        formal_validation["final_oos_opened"] = False
    elif consumed_historical_replay:
        formal_validation.update(REPLAY_MARKERS)
    if formal_model_admission is not None:
        formal_validation["model_admission"] = formal_model_admission
    deflated_sharpe = deflated_sharpe_probability(
        net_daily_returns,
        trials=strategy_trial_count,
        trial_sharpes=(
            multiple_testing.get("trial_daily_sharpes")
            if multiple_testing.get("status") == "ok"
            else None
        ),
    )
    if incomplete_factor_family:
        deflated_sharpe = build_factor_score_incomplete_family_dsr(
            blocked_dsr=deflated_sharpe,
            trial_count=strategy_trial_count,
            trial_count_audit_sha256=str(trial_count_audit_sha256),
            eligibility_receipt_sha256=str(
                incomplete_family_eligibility["receipt_sha256"]
            ),
        )
    metrics = {
        **formal.metrics,
        "policy_version": policy.version,
        "cost_model": cost_schedule.to_dict(),
        "cash_yield": {
            "annual_rate": float(config.get("annual_cash_yield_rate", 0.0)),
            "source": str(config.get("cash_yield_source") or "none_zero_yield"),
            "accounting": "idle cash earns zero without a governed cash instrument",
        },
        "eligibility": eligibility_evidence,
        "deflated_sharpe": deflated_sharpe,
        "deflated_sharpe_probability": deflated_sharpe["probability"],
        "formal_validation": formal_validation,
        "formal_validation_passed": formal_validation["status"] == "passed",
        "evaluation_mode": evaluation_mode,
        "evidence_mode": (
            EVIDENCE_MODE_REPLAY
            if consumed_historical_replay
            else EVIDENCE_MODE_SEALED
        ),
        "strategy_trial_count": strategy_trial_count,
        "evaluation_scope": (
            "pre_final_only"
            if evaluation_mode in PRE_FINAL_EVALUATION_MODES
            else (
                "historical_description_only"
                if consumed_historical_replay
                else "final_oos_once"
            )
        ),
        "final_oos_opened": historical_window_opened,
        **(REPLAY_MARKERS if consumed_historical_replay else {}),
        **(
            {"capital_eligible": False}
            if evaluation_mode in PRE_FINAL_EVALUATION_MODES
            or consumed_historical_replay
            else {}
        ),
        "execution_model": {
            "method": execution_method,
            "days": int(config.get("execution_days", 1)),
            "price_assumption": (
                "next eligible minute bar vwap"
                if execution_method == "next_bar"
                else ("minute bar vwap fills" if minute_execution else "next-day open")
            ),
            "signal_frequency": signal_frequency,
            "frequency": args.execution_frequency if minute_execution else "day",
            "dataset": (
                manifest.get("execution_dataset") if minute_execution else manifest["dataset"]
            ),
            "contract_version": (
                execution_provenance.get("execution_contract_version") if minute_execution else None
            ),
            "slice_minutes": execution_policy.get("slice_minutes") if execution_policy else None,
            "max_slices": execution_policy.get("max_slices") if execution_policy else None,
            "vwap_profile": vwap_profile_evidence,
            "strategy_contract": strategy_contract,
            "strategy_contract_hash": config["execution_contract_hash"],
        },
        "robustness": validation["robustness"],
        "robustness_passed": validation["robustness"]["passed"],
        "robustness_pass_rate": validation["robustness"]["pass_rate"],
        "component_cost_stress": validation["component_cost_stress"],
        "component_cost_stress_passed": validation["component_cost_stress"]["passed"],
        "component_cost_stress_pass_rate": validation["component_cost_stress"]["pass_rate"],
        "rolling": validation["rolling"],
        "rolling_pass_rate": validation["rolling"]["pass_rate"],
        "rolling_passed": validation["rolling"]["passed"],
        "rolling_window_count": validation["rolling"]["window_count"],
        "event_stress": validation["event_stress"],
        "event_stress_count": validation["event_stress"]["event_count"],
        "event_stress_pass_rate": validation["event_stress"]["pass_rate"],
        "event_stress_passed": validation["event_stress"]["passed"],
        "capacity": validation["capacity"],
        "capacity_curve_points": len(validation["capacity"]["points"]),
        "capacity_curve_passed": validation["capacity"]["passed"],
        "provenance": {
            "evaluation_mode": evaluation_mode,
            "evidence_mode": (
                EVIDENCE_MODE_REPLAY
                if consumed_historical_replay
                else EVIDENCE_MODE_SEALED
            ),
            "evaluation_scope": (
                "pre_final_only"
                if evaluation_mode in PRE_FINAL_EVALUATION_MODES
                else (
                    "historical_description_only"
                    if consumed_historical_replay
                    else "final_oos_once"
                )
            ),
            "final_oos_opened": historical_window_opened,
            **(REPLAY_MARKERS if consumed_historical_replay else {}),
            TRANSPARENT_BASELINE_RESULT_WORKER_RUNTIME_IMAGE_FIELD: (
                worker_runtime_image_digest
            ),
            "dataset_identity_sha256": provider_provenance.get("dataset_identity_sha256"),
            "pre_final_cutoff": (
                str(manifest.get("pre_final_cutoff") or "")
                if evaluation_mode in PRE_FINAL_EVALUATION_MODES
                else None
            ),
            "formal_model_admission_binding_sha256": (
                formal_model_admission.get("binding_sha256")
                if formal_model_admission is not None
                else None
            ),
            "model_admission_evidence_sha256": (
                formal_model_admission.get("model_admission_evidence_sha256")
                if formal_model_admission is not None
                else None
            ),
            "quant_bundle_admission_evidence_sha256": (
                (formal_model_admission.get("quant_bundle") or {}).get(
                    "admission_evidence_sha256"
                )
                if formal_model_admission is not None
                else None
            ),
            "snapshot_manifest_sha256": provider_provenance.get("snapshot_manifest_sha256"),
            "qlib_builder_sha256": provider_provenance.get("qlib_builder_sha256"),
            "field_contract_version": provider_provenance.get("field_contract_version"),
            "source_volume_unit": provider_provenance.get("source_volume_unit"),
            "qlib_volume_unit": provider_provenance.get("qlib_volume_unit"),
            "source_hand_size": provider_provenance.get("source_hand_size"),
            "lineage_verified": provider_provenance.get("lineage_verified"),
            "source_lineage_id": provider_provenance.get("source_lineage_id"),
            "style_exposure_contract": style_exposure_evidence,
            "execution_dataset_identity_sha256": execution_provenance.get(
                "dataset_identity_sha256"
            ),
            "execution_snapshot_manifest_sha256": execution_provenance.get(
                "snapshot_manifest_sha256"
            ),
            "execution_qlib_builder_sha256": execution_provenance.get("qlib_builder_sha256"),
            "execution_contract_version": execution_provenance.get("execution_contract_version"),
            "execution_fields": execution_provenance.get("fields"),
            "execution_source_datasets": execution_provenance.get("source_datasets"),
            "execution_source_unit_contracts": execution_provenance.get("source_unit_contracts"),
            "execution_source_lineage_id": execution_provenance.get("source_lineage_id"),
            "execution_lineage_verified": execution_provenance.get("lineage_verified"),
            "strategy_config_sha256": _canonical_sha256(config),
            "execution_manifest_sha256": None,
            "factor_values_sha256": factor_value_hashes,
            "factor_code_sha256": factor_code_hashes,
            "formal_factor_values_sha256": formal_factor_hashes,
            "formal_factor_recompute_evidence": formal_factor_evidence,
            "formal_model_bundle_factor_values_sha256": (
                formal_bundle_factor_hashes if model_bundle_factors else None
            ),
            "formal_model_bundle_factor_recompute_evidence": (
                formal_bundle_factor_evidence if model_bundle_factors else None
            ),
            "quant_bundle_factor_contract_sha256": config.get(
                "quant_bundle_factor_contract_sha256"
            ),
            "signal_source": signal_source,
            "model_signal_identity_sha256": (
                (manifest.get("model_signal") or {}).get("identity_sha256")
                if signal_source == "model_prediction"
                else None
            ),
            "formal_model_predictions_sha256": (
                formal_model_artifact["predictions_sha256"] if formal_model_artifact else None
            ),
            "formal_model_checkpoint_sha256": (
                formal_model_artifact["checkpoint_sha256"] if formal_model_artifact else None
            ),
            "formal_model_checkpoint_format": (
                formal_model_artifact["checkpoint_format"] if formal_model_artifact else None
            ),
            "formal_model_data_contract_sha256": (
                formal_model_artifact["model_data_contract_sha256"]
                if formal_model_artifact
                else None
            ),
            "formal_model_evidence_sha256": (
                canonical_model_sha256(formal_model_artifact["evidence"])
                if formal_model_artifact
                else None
            ),
            "formal_model_execution_environment_sha256": (
                formal_model_artifact["evidence"][
                    "execution_environment_sha256"
                ]
                if formal_model_artifact
                else None
            ),
            "formal_model_additional_factors_sha256": (
                _sha256_file(additional_factors_path)
                if additional_factors_path is not None
                else None
            ),
            "factor_source_mode": factor_source_mode,
            "challenger_weight": float(config.get("challenger_weight") or 0.0),
            "baseline_definition_sha256": config.get("baseline_definition_sha256"),
            "baseline_qlib_expressions": (
                {
                    str(item["id"]): str(item["qlib_expression"])
                    for item in baseline_definition.get("factors") or []
                }
                if isinstance(baseline_definition, dict)
                else None
            ),
            "baseline_preprocessing": (
                baseline_definition.get("preprocessing")
                if isinstance(baseline_definition, dict)
                else None
            ),
            "baseline_raw_values_sha256": (
                {
                    factor_id: entry["sha256"]
                    for factor_id, entry in baseline_artifacts["raw"].items()
                }
                if baseline_artifacts
                else None
            ),
            "baseline_normalized_values_sha256": (
                {
                    factor_id: entry["sha256"]
                    for factor_id, entry in baseline_artifacts["normalized"].items()
                }
                if baseline_artifacts
                else None
            ),
            "baseline_composite_values_sha256": (
                baseline_artifacts["composite"]["sha256"] if baseline_artifacts else None
            ),
            "qlib_version": qlib_runtime["version"],
            "qlib_commit": qlib_runtime["commit"],
            "backtest_engine_version": QLIB_ENGINE_VERSION,
            "policy_version": policy.version,
        },
    }
    qlib_report.reset_index().to_parquet(output / "daily_returns.parquet", index=False)
    scores.to_frame(name="score").to_parquet(output / "score_grid.parquet")
    governed_signal.to_frame().to_parquet(output / "governed_signal.parquet")
    qlib_report.to_parquet(output / "qlib_portfolio_report.parquet")
    pd.to_pickle(qlib_positions, output / "qlib_positions.pkl")
    pd.DataFrame(
        formal.fills,
        columns=[
            "instrument",
            "date",
            "side",
            "requested_amount",
            "amount",
            "capacity_fill_ratio",
            "trade_price",
            "trade_value",
            "cost",
        ],
    ).to_parquet(output / "execution_fills.parquet", index=False)
    (output / "execution_model.json").write_text(
        json.dumps(metrics["execution_model"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "robustness.json").write_text(
        json.dumps(metrics["robustness"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "rolling.json").write_text(
        json.dumps(metrics["rolling"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "event_stress.json").write_text(
        json.dumps(metrics["event_stress"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "capacity_curve.json").write_text(
        json.dumps(metrics["capacity"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "formal_validation.json").write_text(
        json.dumps(metrics["formal_validation"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Dataset descriptors consumed by the promotion chain: after the formal
    # hard gate approves the version, the isolated paper simulation account is
    # created from these verbatim dataset provenance records (design 6.11).
    dataset_descriptors = _promotion_dataset_descriptors(
        daily_dataset_name=str(manifest["dataset"]),
        daily_provenance=provider_provenance,
        execution_method=execution_method,
        execution_frequency=configured_execution_frequency,
        formal_execution_start=str(periods["start"]),
        execution_dataset_name=(
            str(manifest["execution_dataset"]) if minute_execution else None
        ),
        execution_provenance=(execution_provenance if minute_execution else None),
    )
    (output / "datasets.json").write_text(
        json.dumps(
            dataset_descriptors,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    result = _finalize_backtest_output(
        output, manifest=manifest, manifest_path=args.manifest,
        provider_provenance=provider_provenance, periods=periods, metrics=metrics,
        evaluation_mode=evaluation_mode, signal_source=signal_source,
        execution_method=execution_method,
        execution_frequency=args.execution_frequency if minute_execution else "day",
        historical_window_opened=historical_window_opened, tracking_uri=args.tracking_uri,
        health_reference_inputs={
            "factor_source_mode": factor_source_mode,
            "baseline_definition": (
                dict(baseline_definition) if isinstance(baseline_definition, dict) else None
            ),
            "baseline_raw": baseline_raw, "baseline_artifacts": baseline_artifacts,
            "challenger_entries": challenger_entries, "formal_factor_items": formal_factor_items,
            "model_feature_set": model_feature_set, "model_predictions": model_predictions,
            "model_label_contract": model_label_contract, "qlib_data_api": D,
        },
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
