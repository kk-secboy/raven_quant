from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, date, datetime
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, insert, or_, select, text, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    audit_events,
    backtest_runs,
    capital_oos_alpha_batches,
    capital_oos_alpha_families,
    factor_candidates,
    factor_evaluations,
    model_artifacts,
    model_candidates,
    model_ensemble_candidates,
    model_ensemble_evaluations,
    model_evaluations,
    oos_vintages,
    open_database,
    quant_bundle_candidates,
    quant_bundle_evaluations,
    research_campaigns,
    research_programs,
    research_run_artifacts,
    row_dict,
    strategies,
    strategy_events,
    strategy_factors,
    strategy_forward_gates,
    strategy_health_snapshots,
    strategy_pairs,
    strategy_versions,
    transparent_baseline_pre_result_repairs,
)
from quant_data.execution_contract import (
    require_daily_qlib_contract,
    require_minute_execution_contract,
    require_strategy_execution_contract,
    strategy_execution_contract_hash,
)
from quant_platform.alpha_spending_ledger import (
    CapitalOOSAlphaLedgerStore,
    capital_oos_vintage_link_payload,
)
from quant_platform.capital_oos_receipt import (
    require_capital_oos_receipt,
)
from quant_platform.capital_oos_receipt import (
    sha256_file as capital_oos_sha256_file,
)
from quant_platform.cost_model import KNOWN_COST_SCHEDULE_VERSIONS, CostModelConfig
from quant_platform.eligibility import ELIGIBILITY_CONTRACT_VERSION
from quant_platform.factor_library_store import validate_incremental_evidence
from quant_platform.factor_recompute import (
    FACTOR_MIN_COVERAGE_RATIO,
    FACTOR_MIN_DAILY_FINITE,
    FACTOR_MIN_GOOD_DAY_RATE,
    FACTOR_PIT_CONTRACT_VERSION,
    FACTOR_RECOMPUTE_EXECUTOR_VERSION,
    submitted_comparison_is_admissible,
)
from quant_platform.formal_validation import (
    CONSERVATIVE_BONFERRONI_INCOMPLETE_FAMILY_STATUS,
    FACTOR_SCORE_INCOMPLETE_FAMILY_MULTIPLE_TESTING_VERSION,
    FORMAL_VALIDATION_CONTRACT_VERSION,
    FROZEN_STRATEGY_OUTER_SCOPE,
    NOT_COMPUTABLE_INCOMPLETE_FAMILY_STATUS,
    PRE_FINAL_HISTORY_CONTRACT_VERSION,
    SIGNAL_DECAY_FRONTIER_VERSION,
    paired_bootstrap_parameters_from_config,
    validate_factor_score_incomplete_family_dsr,
    validate_factor_score_incomplete_family_multiple_testing,
    validate_paired_bootstrap_evidence,
)
from quant_platform.forward_only_rehabilitation import (
    EVIDENCE_MODE_LEGACY,
    EVIDENCE_MODE_REPLAY,
    EVIDENCE_MODE_SEALED,
    audit_incomplete_family_artifacts,
    build_incomplete_family_eligibility,
    build_qualification,
    incomplete_family_eligibility_for_version,
    insert_incomplete_family_eligibility,
    insert_qualification,
    register_terminal_cash_only_receipt,
    rehabilitation_forward_thresholds,
    require_consumed_vintage,
    require_incomplete_family_eligibility,
    require_replay_config,
    require_replay_markers,
)
from quant_platform.forward_only_rehabilitation import (
    canonical_sha256 as rehabilitation_canonical_sha256,
)
from quant_platform.horizon_factor_bundle import validate_horizon_factor_bundle
from quant_platform.model_ensemble import prediction_grid_from_admission
from quant_platform.model_research_governance import (
    MODEL_REFIT_POLICY,
    MODEL_REFIT_POLICY_SHA256,
    PRIMARY_MODEL_PROFILE,
    PRIMARY_MODEL_SEED,
    REQUIRED_MODEL_SEEDS,
    REQUIRED_QUANT_ABLATIONS,
    REQUIRED_RESEARCH_PROFILES,
    validate_independent_model_evidence,
    validate_quant_bundle_evidence,
)
from quant_platform.model_strategy_contract import (
    MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_REASON,
    MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_STATUS,
    QUANT_BUNDLE_FACTOR_CONTRACT_VERSION,
    QUANT_BUNDLE_FACTOR_WEIGHT_POLICY,
    build_model_ensemble_formal_admission_binding,
    build_model_formal_admission_binding,
    formal_model_artifact_failures,
    model_signal_identity,
    normalize_model_signal_config,
    validate_model_formal_admission_binding,
)
from quant_platform.pair_trading import PairTradingConfig
from quant_platform.qlib_backtest import (
    COMPONENT_COST_STRESS_MULTIPLIERS,
    QLIB_ENGINE_VERSION,
)
from quant_platform.qlib_factor_baseline import (
    FACTOR_SOURCE_QLIB_BASELINE,
    baseline_manifest_failures,
    bind_factor_source_config,
)
from quant_platform.qlib_workflow import require_qlib_workflow_identity
from quant_platform.research_horizon import (
    LEGACY_AMBIGUOUS,
    horizon_columns_from_config,
    normalize_horizon_config,
    require_horizon_row,
    require_label_horizon,
)
from quant_platform.research_horizon import (
    canonical_sha256 as horizon_canonical_sha256,
)
from quant_platform.research_store import FactorGatePolicy
from quant_platform.statistical_validation import DEFLATED_SHARPE_METHOD_VERSION
from quant_platform.strategy_artifact_manifest import (
    STRATEGY_BACKTEST_ARTIFACT_MANIFEST_VERSION,
    validate_backtest_artifact_manifest,
)
from quant_platform.strategy_catalog import strategy_type_capabilities
from quant_platform.strategy_health import (
    COLLECTOR_ACTOR,
    HEALTHY,
    RETIRED,
    WATCH,
    transition_strategy_health,
)
from quant_platform.strategy_research_admission import (
    FIN_STRATEGY_FULL_STACK_ARTIFACT_TYPE,
    FIN_STRATEGY_POLICY_ARTIFACT_TYPE,
    FIN_STRATEGY_WINNER_ARTIFACT_TYPE,
    build_fin_strategy_formal_admission,
    validate_fin_strategy_formal_admission,
)
from quant_platform.strategy_rule_compiler import (
    validate_compiled_strategy_artifact,
    validate_strategy_rule_binding,
)
from quant_platform.strategy_trial_lineage import build_strategy_trial_lineage
from quant_platform.transparent_baseline_lockbox import (
    baseline_oos_sealed_member_set,
    validate_lockbox_link,
    validate_repair_registry_binding,
)
from quant_platform.transparent_baseline_runner import (
    DISCRETE_MAX_POSITION_REPAIR_TARGET_RECIPE_VERSION,
    FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION,
    FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION,
    FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION,
    POSITION_RISK_TARGET_RECIPE_VERSION,
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION,
    STRATEGY_RESEARCH_TARGET_RECIPE_VERSION,
    TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION,
    TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_RESULT_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
    target_runner_for_recipe,
    target_runtime_bundle_for_recipe,
    target_worker_runtime_image_for_recipe,
)
from quant_platform.upstream_versions import QLIB_COMMIT, RDAGENT_COMMIT


def _now() -> datetime:
    return datetime.now(UTC)


def _lock_strategy_trial_family(connection: Any, economic_hypothesis_group: str) -> None:
    group = str(economic_hypothesis_group or "").strip()
    if not group:
        raise ValueError("strategy trial-family lock requires an economic hypothesis group")
    connection.execute(
        select(
            func.pg_advisory_xact_lock(
                func.hashtext(f"strategy-trial-family:{group}")
            )
        )
    )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_image_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and _is_sha256(value.removeprefix("sha256:"))
    )


def _transparent_worker_runtime_failures(
    version: Mapping[str, Any],
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> list[str]:
    """Bind sealed config, formal job manifest and result to one worker image."""

    config_raw = version.get("config")
    config = dict(config_raw) if isinstance(config_raw, Mapping) else {}
    if (
        str(config.get("recipe_version") or "")
        not in {
            POSITION_RISK_TARGET_RECIPE_VERSION,
            FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION,
            FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION,
            SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION,
            DISCRETE_MAX_POSITION_REPAIR_TARGET_RECIPE_VERSION,
            TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION,
            FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION,
            STRATEGY_RESEARCH_TARGET_RECIPE_VERSION,
        }
        or target_runner_for_recipe(
            config.get("recipe_id"), config.get("recipe_version")
        ) is None
    ):
        return []
    bootstrap_raw = config.get("transparent_baseline_bootstrap")
    bootstrap = dict(bootstrap_raw) if isinstance(bootstrap_raw, Mapping) else {}
    expected = bootstrap.get(TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD)
    if not _is_image_digest(expected):
        return ["transparent worker runtime image digest is missing or invalid"]
    failures: list[str] = []
    if manifest.get(TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD) != expected:
        failures.append(
            "strategy backtest manifest worker runtime image differs from the sealed version"
        )
    if provenance.get(TRANSPARENT_BASELINE_RESULT_WORKER_RUNTIME_IMAGE_FIELD) != expected:
        failures.append(
            "formal result worker runtime image differs from the sealed version"
        )
    return failures


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bind_current_transparent_runtime_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Seal every governed current-recipe draft to the release runtime.

    The automated bootstrap already supplies a larger dataset/window contract,
    while the advanced strategy API can create a research-only draft directly
    from a public recipe.  Both write paths must carry the same immutable
    runner, local-source closure and exact worker image identity before the
    database accepts the StrategyVersion.  Existing mismatched values are
    rejected rather than silently replaced.
    """

    recipe_id = config.get("recipe_id")
    recipe_version = config.get("recipe_version")
    is_current_public_recipe = (
        str(recipe_version or "")
        == STRATEGY_RESEARCH_TARGET_RECIPE_VERSION
        and target_runner_for_recipe(recipe_id, recipe_version) is not None
    )
    is_forward_only_rehabilitation = (
        str(recipe_version or "") == FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
    )
    if not is_current_public_recipe and not is_forward_only_rehabilitation:
        return config
    if is_forward_only_rehabilitation and str(recipe_id or "") != "short_relative_strength":
        raise ValueError(
            "the v18 transparent runtime is restricted to the exact short "
            "forward-only rehabilitation entry point"
        )
    if is_forward_only_rehabilitation and config.get("evidence_mode") != EVIDENCE_MODE_REPLAY:
        raise ValueError(
            "the current short transparent baseline is restricted to the exact "
            "forward-only rehabilitation entry point"
        )
    if is_current_public_recipe and config.get("evidence_mode") != EVIDENCE_MODE_SEALED:
        raise ValueError("the v19 strategy-research runtime requires sealed final OOS")
    if target_runner_for_recipe(recipe_id, recipe_version) is None:
        raise ValueError("the current transparent runner identity is unavailable")
    bootstrap_raw = config.get("transparent_baseline_bootstrap")
    if bootstrap_raw is not None and not isinstance(bootstrap_raw, Mapping):
        raise ValueError("transparent current bootstrap must be an object")
    bootstrap = dict(bootstrap_raw or {})
    bindings = {
        TRANSPARENT_BASELINE_RUNNER_FIELD: target_runner_for_recipe(
            recipe_id, recipe_version
        ),
        TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: target_runtime_bundle_for_recipe(
            recipe_id, recipe_version
        ),
        TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: (
            target_worker_runtime_image_for_recipe(recipe_id, recipe_version)
        ),
    }
    for field, expected in bindings.items():
        existing = bootstrap.get(field)
        if expected is None or (existing is not None and existing != expected):
            raise ValueError(
                f"transparent current bootstrap {field} differs from this release"
            )
        bootstrap[field] = expected
    return {**config, "transparent_baseline_bootstrap": bootstrap}


def _integrity_constraint_name(exc: IntegrityError) -> str | None:
    """Return a safe PostgreSQL constraint name without exposing SQL values."""

    diag = getattr(getattr(exc, "orig", None), "diag", None)
    value = getattr(diag, "constraint_name", None)
    normalized = str(value or "").strip()
    return normalized or None


def _factor_evaluation_artifact_metrics(
    path_value: str,
    *,
    candidate_id: str,
    profile_id: str | None,
) -> dict[str, Any]:
    path = Path(path_value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Qlib factor evaluation artifact is unreadable") from exc
    require_qlib_workflow_identity(
        payload.get("qlib_workflow") if isinstance(payload, dict) else None
    )
    evaluations = payload.get("evaluations") if isinstance(payload, dict) else None
    matches = [
        item
        for item in evaluations or []
        if isinstance(item, dict)
        and str(item.get("candidate_id") or "") == candidate_id
        and item.get("status") == "ok"
        and (
            profile_id is None
            or str(((item.get("metrics") or {}).get("research_profile") or {}).get("id") or "")
            == profile_id
        )
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("metrics"), dict):
        raise ValueError("Qlib factor evaluation artifact has no unique governed result")
    return dict(matches[0]["metrics"])


def _validate_governed_factor_evaluation(
    evaluation: dict[str, Any],
    candidate: dict[str, Any],
    *,
    expected_profile_id: str | None,
    require_gate_passed: bool,
) -> dict[str, Any]:
    """Revalidate immutable PIT, recompute and policy evidence at strategy use."""

    candidate_id = str(candidate["id"])
    policy = FactorGatePolicy()
    if evaluation.get("evaluator_version") != policy.version:
        raise ValueError(
            "external frozen-value factors remain research-only until a formal "
            "point-in-time availability and final-OOS publication contract is implemented"
        )
    if (
        str(evaluation.get("factor_candidate_id") or "") != candidate_id
        or evaluation.get("is_legacy") is True
        or evaluation.get("candidate_code_sha256") != candidate.get("code_sha256")
        or evaluation.get("candidate_values_sha256") != candidate.get("values_sha256")
        or evaluation.get("recomputed_values_sha256") != candidate.get("values_sha256")
    ):
        raise ValueError(f"promoted factor {candidate_id} evaluation binding is invalid")
    metrics = evaluation.get("metrics_json")
    expected_policy = asdict(policy)
    if (
        not isinstance(metrics, dict)
        or _canonical_sha256(metrics) != evaluation.get("metrics_sha256")
        or evaluation.get("policy_json") != expected_policy
        or _canonical_sha256(expected_policy) != evaluation.get("policy_sha256")
    ):
        raise ValueError(f"promoted factor {candidate_id} evaluation provenance is invalid")
    gate_status, gate_reasons = policy.evaluate(metrics)
    layers = policy.evaluate_layers(metrics)
    if (
        layers["hard_status"] != "passed"
        or evaluation.get("gate_status") != gate_status
        or list(evaluation.get("gate_reasons_json") or []) != gate_reasons
        or (require_gate_passed and gate_status != "passed")
    ):
        raise ValueError(f"promoted factor {candidate_id} no longer passes its governed gate")
    profile_id = str(((metrics.get("research_profile") or {}).get("id")) or "") or None
    if expected_profile_id is not None and profile_id != expected_profile_id:
        raise ValueError(f"promoted factor {candidate_id} profile identity changed")
    periods = {
        key: evaluation[key].isoformat()
        for key in (
            "train_start",
            "train_end",
            "valid_start",
            "valid_end",
            "test_start",
            "test_end",
        )
    }
    recompute = evaluation.get("recompute_evidence_json")
    pit = recompute.get("pit_invariance") if isinstance(recompute, dict) else None
    boundary = (
        recompute.get("research_data_boundary") if isinstance(recompute, dict) else None
    )
    comparison = (
        recompute.get("submitted_comparison") if isinstance(recompute, dict) else None
    )
    if not (
        isinstance(recompute, dict)
        and recompute.get("executor_version") == FACTOR_RECOMPUTE_EXECUTOR_VERSION
        and recompute.get("sandbox_mode") == "docker-isolated"
        and str(recompute.get("sandbox_image_id") or "").startswith("sha256:")
        and len(str(recompute.get("sandbox_image_id") or "")) == 71
        and recompute.get("network_mode") == "none"
        and recompute.get("root_filesystem_read_only") is True
        and recompute.get("capabilities_dropped") == "ALL"
        and recompute.get("no_new_privileges") is True
        and recompute.get("code_sha256") == candidate.get("code_sha256")
        and recompute.get("dataset_identity_sha256")
        == evaluation.get("dataset_identity_sha256")
        and recompute.get("authoritative_values_sha256")
        == evaluation.get("recomputed_values_sha256")
        and int(recompute.get("label_horizon_days") or 0)
        == int(candidate.get("label_horizon_days") or 1)
        and bool(recompute.get("provider_input_sha256"))
        and recompute.get("periods") == periods
        and isinstance(pit, dict)
        and pit.get("contract_version") == FACTOR_PIT_CONTRACT_VERSION
        and pit.get("status") == "passed"
        and int(pit.get("cutpoint_count") or 0) >= 3
        and isinstance(pit.get("checks"), list)
        and len(pit["checks"]) == int(pit["cutpoint_count"])
        and all(
            isinstance(item, dict) and item.get("invariant") is True
            for item in pit["checks"]
        )
        and isinstance(boundary, dict)
        and boundary.get("valid_end") == periods["valid_end"]
        and boundary.get("test_start") == periods["test_start"]
        and boundary.get("final_oos_observations_exposed") is False
        and str(boundary.get("latest_input_date") or "") <= periods["valid_end"]
        and submitted_comparison_is_admissible(
            comparison,
            authoritative_end=str(boundary.get("latest_input_date") or ""),
        )
        and str((comparison or {}).get("submitted_sha256") or "")
        == str(evaluation.get("submitted_values_sha256") or "")
    ):
        raise ValueError(f"promoted factor {candidate_id} lacks strict PIT/recompute evidence")
    artifact_path = Path(str(evaluation.get("artifact_path") or ""))
    if (
        not artifact_path.is_file()
        or _sha256_file(artifact_path) != evaluation.get("artifact_sha256")
        or _canonical_sha256(
            _factor_evaluation_artifact_metrics(
                str(artifact_path),
                candidate_id=candidate_id,
                profile_id=profile_id,
            )
        )
        != evaluation.get("metrics_sha256")
    ):
        raise ValueError(f"promoted factor {candidate_id} evaluation artifact changed")
    evaluation_evidence = {
        "candidate_id": candidate_id,
        "dataset": evaluation.get("dataset"),
        "dataset_identity_sha256": evaluation.get("dataset_identity_sha256"),
        "periods": periods,
        "gate_status": evaluation.get("gate_status"),
        "gate_reasons": list(evaluation.get("gate_reasons_json") or []),
        "evaluator_version": evaluation.get("evaluator_version"),
        "candidate_code_sha256": evaluation.get("candidate_code_sha256"),
        "candidate_values_sha256": evaluation.get("candidate_values_sha256"),
        "submitted_values_sha256": evaluation.get("submitted_values_sha256"),
        "recompute_evidence_sha256": _canonical_sha256(recompute),
        "artifact_sha256": evaluation.get("artifact_sha256"),
        "metrics_sha256": evaluation.get("metrics_sha256"),
        "policy_sha256": evaluation.get("policy_sha256"),
    }
    if (
        evaluation.get("final_test_key") is not None
        or evaluation.get("final_test_consumed_at") is not None
    ):
        raise ValueError(
            f"promoted factor {candidate_id} final OOS has already been consumed"
        )
    if (
        _canonical_sha256(evaluation_evidence) != evaluation.get("evidence_sha256")
        or evaluation.get("execution_contract_hash") != evaluation.get("evidence_sha256")
        or evaluation.get("qlib_commit") != QLIB_COMMIT
        or evaluation.get("rdagent_commit") != RDAGENT_COMMIT
        or evaluation.get("signal_frequency") != "day"
        or evaluation.get("execution_frequency") != "day"
        or evaluation.get("signal_horizon")
        != f"{int(candidate.get('label_horizon_days') or 1)}d"
    ):
        raise ValueError(f"promoted factor {candidate_id} evaluation seal is invalid")
    return layers


def _validate_governed_profile_consensus(
    candidate: dict[str, Any],
    evaluations_by_profile: Mapping[str, dict[str, Any]],
    consensus: dict[str, Any],
) -> None:
    """Rebuild one sealed three-profile admission from its governed rows.

    The official consensus contract deliberately treats ``robust_10y`` as a
    stress profile: its full factor gate may fail, provided its hard evidence,
    coverage, non-negative cost-adjusted return and direction agreement still
    pass. Strategy publication must preserve that exact contract instead of
    silently strengthening it to three fully-passed gates or trusting the
    persisted consensus JSON without recomputation.
    """

    from quant_platform.research_automation import build_multi_profile_consensus

    expected_profile_ids = {"recent_3y", "balanced_5y", "robust_10y"}
    if set(evaluations_by_profile) != expected_profile_ids:
        raise ValueError(
            f"promoted factor {candidate['id']} consensus evaluations are incomplete"
        )
    normalized_evaluations: list[dict[str, Any]] = []
    for profile_id in sorted(expected_profile_ids):
        evaluation = evaluations_by_profile[profile_id]
        _validate_governed_factor_evaluation(
            evaluation,
            candidate,
            expected_profile_id=profile_id,
            require_gate_passed=profile_id != "robust_10y",
        )
        normalized_evaluations.append(
            {
                **evaluation,
                "metrics": dict(evaluation.get("metrics_json") or {}),
            }
        )
    rebuilt = build_multi_profile_consensus(
        {
            **candidate,
            "profile_evaluations": normalized_evaluations,
        }
    )
    if rebuilt != consensus:
        raise ValueError(
            f"promoted factor {candidate['id']} profile consensus no longer "
            "satisfies governed admission"
        )


def _version_contract_columns(config: dict[str, Any], *, strategy_type: str) -> dict[str, Any]:
    if strategy_type == "pair":
        signal_frequency = "day"
        signal_horizon = "1d"
        execution_frequency = "1min"
        contract_hash = _canonical_sha256(
            {
                "strategy_type": "pair",
                "signal_frequency": signal_frequency,
                "signal_horizon": signal_horizon,
                "execution_frequency": execution_frequency,
                "config": config,
            }
        )
    else:
        signal_frequency = str(config.get("signal_frequency") or "day")
        signal_horizon = f"{int(config.get('signal_period') or 1)}bar"
        execution_frequency = str(config.get("execution_frequency") or "day")
        contract_hash = str(config.get("execution_contract_hash") or "")
        if not _is_sha256(contract_hash):
            raise ValueError("strategy execution contract hash is required")
    return {
        "evidence_mode": str(
            config.get("evidence_mode")
            or (
                EVIDENCE_MODE_LEGACY
                if strategy_type == "pair"
                else EVIDENCE_MODE_SEALED
            )
        ),
        "signal_frequency": signal_frequency,
        "signal_horizon": signal_horizon,
        "execution_frequency": execution_frequency,
        "execution_contract_hash": contract_hash,
        "qlib_version": f"0.0.dev0+g{QLIB_COMMIT}",
        "qlib_commit": QLIB_COMMIT,
        "rdagent_version": f"0.0.dev0+g{RDAGENT_COMMIT}",
        "rdagent_commit": RDAGENT_COMMIT,
        "source_research_artifact_id": config.get("source_research_artifact_id"),
        "strategy_rules_sha256": config.get("strategy_rules_sha256"),
        **horizon_columns_from_config(config),
    }


def _normalize_multifactor_contract(
    config: dict[str, Any], *, factor_count: int, creating_family: bool
) -> dict[str, Any]:
    normalized = normalize_model_signal_config(dict(config))
    normalized.setdefault("evidence_mode", EVIDENCE_MODE_SEALED)
    if normalized["evidence_mode"] not in {
        EVIDENCE_MODE_SEALED,
        EVIDENCE_MODE_REPLAY,
    }:
        raise ValueError("new multifactor strategies require an explicit evidence mode")
    if normalized["signal_source"] == "model_prediction":
        submitted_factor_source = str(
            config.get("factor_source_mode") or "promoted_only"
        )
        if submitted_factor_source not in {
            "promoted_only",
            "not_applicable_model_prediction",
        }:
            raise ValueError(
                "model-prediction strategies cannot bind a factor-score baseline"
            )
        if factor_count:
            raise ValueError("model-prediction strategies cannot bind standalone score factors")
        normalized.update(
            {
                "factor_source_mode": "not_applicable_model_prediction",
                "challenger_weight": 0.0,
                "baseline_definition": None,
                "baseline_definition_sha256": None,
            }
        )
    else:
        normalized = bind_factor_source_config(
            normalized,
            factor_count=factor_count,
            creating_family=creating_family,
        )
    normalized.setdefault("signal_frequency", "day")
    normalized.setdefault("signal_period", 1)
    normalized.setdefault("execution_frequency", "day")
    normalized.setdefault("execution_lag_bars", 1)
    normalized.setdefault("execution_method", "open")
    normalized.setdefault("execution_days", 1)
    normalized.setdefault("execution_slice_minutes", 20)
    normalized.setdefault("max_execution_slices", 24)
    normalized = normalize_horizon_config(normalized)
    normalized = _bind_current_transparent_runtime_identity(normalized)
    if normalized["horizon_profile"] != LEGACY_AMBIGUOUS:
        horizon = normalized["horizon_contract"]
        normalized.setdefault("outer_purge_days", horizon["purge_sessions"])
        normalized.setdefault("outer_embargo_days", horizon["embargo_sessions"])
        normalized.setdefault("min_backtest_days", horizon["sealed_oos_sessions"])
        if normalized["signal_frequency"] != "day":
            raise ValueError("short/swing/long horizon profiles require daily signals")
        if int(normalized["execution_lag_bars"]) != int(
            horizon["execution_lag_sessions"]
        ):
            raise ValueError("execution lag differs from the selected horizon profile")
        if int(normalized.get("outer_purge_days") or 0) < int(
            horizon["purge_sessions"]
        ):
            raise ValueError("outer purge is shorter than the selected horizon contract")
        if int(normalized.get("outer_embargo_days") or 0) < int(
            horizon["embargo_sessions"]
        ):
            raise ValueError("outer embargo is shorter than the selected horizon contract")
        if int(normalized.get("min_backtest_days") or 0) < int(
            horizon["sealed_oos_sessions"]
        ):
            raise ValueError("formal backtest is shorter than the selected sealed OOS")
        validate_strategy_rule_binding(normalized)
    normalized["execution_contract_hash"] = strategy_execution_contract_hash(normalized)
    require_strategy_execution_contract(normalized)
    if normalized["evidence_mode"] == EVIDENCE_MODE_REPLAY:
        require_replay_config(normalized)
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_strategy_source_artifact(
    connection: Any,
    config: Mapping[str, Any],
    *,
    expected_strategy_id: str | None,
) -> dict[str, Any] | None:
    """Verify a compiled fin_strategy artifact at the database write boundary."""

    source_id = str(config.get("source_research_artifact_id") or "").strip()
    if not source_id:
        return None
    row = connection.execute(
        select(research_run_artifacts)
        .where(research_run_artifacts.c.id == source_id)
        .with_for_update()
    ).first()
    if row is None:
        raise ValueError("strategy research source artifact does not exist")
    manifest = dict(row.manifest_json or {})
    path = Path(str(row.storage_path))
    if (
        str(row.status) != "recorded"
        or str(row.artifact_type) != "fin_strategy_compiled_artifact"
        or str(row.contract_version) != "compiled-strategy-proposal-v1"
        or bool(row.capital_eligible)
        or _canonical_sha256(manifest) != str(row.manifest_sha256)
        or manifest.get("id") != source_id
        or manifest.get("artifact_type") != "fin_strategy_compiled_artifact"
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(row.size_bytes)
        or _sha256_file(path) != str(row.content_sha256)
    ):
        raise ValueError("strategy research source artifact is not intact")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("strategy research source artifact is unreadable") from exc
    candidate = raw.get("strategy_spec_candidate") if isinstance(raw, dict) else None
    rule_ir = candidate.get("rule_ir") if isinstance(candidate, dict) else None
    alpha = (
        (((rule_ir or {}).get("slots") or {}).get("alpha_rank") or {}).get(
            "components"
        )
        if isinstance(rule_ir, dict)
        else None
    )
    weights = None
    if isinstance(alpha, list):
        weights = next(
            (
                ((item.get("parameters") or {}).get("weights"))
                for item in alpha
                if isinstance(item, dict)
                and item.get("component") == "weighted_factor_rank"
            ),
            None,
        )
    allowed_factor_ids = set(weights) if isinstance(weights, dict) else None
    artifact = validate_compiled_strategy_artifact(
        raw,
        allowed_factor_ids=allowed_factor_ids,
    )
    proposal = artifact["strategy_proposal"]
    candidate = artifact["strategy_spec_candidate"]
    policy = candidate["execution_policy"]
    expected_values = {
        "horizon_profile": candidate["horizon"],
        "recipe_id": proposal["baseline_recipe_id"],
        "recipe_version": proposal["baseline_recipe_version"],
        "strategy_rule_ir": candidate["rule_ir"],
        "strategy_rules_sha256": artifact["rules_sha256"],
        "strategy_rule_policy_sha256": policy["policy_sha256"],
        "strategy_research_proposal_sha256": artifact["proposal_sha256"],
        "strategy_research_artifact_sha256": artifact["artifact_sha256"],
        "parent_strategy_version_id": proposal["parent_strategy_version_id"],
        "strategy_research_data_contract": proposal["data_contract"],
        "strategy_evaluation_contract": proposal["evaluation_contract"],
    }
    if any(config.get(key) != value for key, value in expected_values.items()):
        raise ValueError("StrategySpec differs from its compiled research artifact")
    parent_id = proposal["parent_strategy_version_id"]
    if parent_id is None:
        if expected_strategy_id is not None:
            raise ValueError("a new family version requires a parent strategy version")
    else:
        parent = connection.execute(
            select(strategy_versions).where(strategy_versions.c.id == str(parent_id))
        ).first()
        if (
            parent is None
            or str(parent.horizon_profile) != str(candidate["horizon"])
            or (
                expected_strategy_id is not None
                and str(parent.strategy_id) != expected_strategy_id
            )
        ):
            raise ValueError("strategy research parent binding is invalid")
        if expected_strategy_id is None:
            raise ValueError("a proposal with a parent must create a family version")
    return artifact


def _verified_research_run_artifact_payload(
    row: Any,
    *,
    expected_type: str,
) -> dict[str, Any]:
    """Read one immutable JSON run artifact through its database seal."""

    manifest = dict(row.manifest_json or {})
    path = Path(str(row.storage_path))
    if (
        str(row.status) != "recorded"
        or str(row.artifact_type) != expected_type
        or bool(row.capital_eligible)
        or _canonical_sha256(manifest) != str(row.manifest_sha256)
        or manifest.get("id") != str(row.id)
        or manifest.get("research_run_id") != str(row.research_run_id)
        or manifest.get("artifact_type") != expected_type
        or manifest.get("content_sha256") != str(row.content_sha256)
        or int(manifest.get("size_bytes") or -1) != int(row.size_bytes)
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(row.size_bytes)
        or _sha256_file(path) != str(row.content_sha256)
    ):
        raise ValueError(f"{expected_type} run artifact is not intact")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{expected_type} run artifact is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{expected_type} run artifact must be a JSON object")
    return payload


def _scenario_artifact_failures(scenarios: dict[str, Any], artifact_root: Path) -> list[str]:
    """Validate each scenario's immutable artifacts against the artifact root."""

    failures: list[str] = []
    for scenario_name, scenario in scenarios.items():
        artifacts = scenario.get("artifacts") if isinstance(scenario, dict) else None
        valid_artifacts = isinstance(artifacts, dict) and set(artifacts) == {
            "daily_report",
            "fills",
            "metrics",
        }
        if valid_artifacts:
            for entry in artifacts.values():
                if not isinstance(entry, dict) or not _is_sha256(entry.get("sha256")):
                    valid_artifacts = False
                    break
                try:
                    artifact_path = (artifact_root / str(entry["path"])).resolve()
                    artifact_path.relative_to(artifact_root)
                except (KeyError, ValueError):
                    valid_artifacts = False
                    break
                if not artifact_path.is_file() or _sha256_file(artifact_path) != entry["sha256"]:
                    valid_artifacts = False
                    break
        if not valid_artifacts:
            failures.append(
                f"robustness scenario {scenario_name} has no complete immutable artifacts"
            )
    return failures


def _pre_final_stability_failures(
    version: Mapping[str, Any], metrics: Mapping[str, Any]
) -> list[str]:
    """Validate stability before the once-only final capital OOS.

    Active horizon strategies must not manufacture several independent tests
    by subdividing their sealed final OOS. Factor strategies use the isolated
    outer walk-forward folds. Model strategies use the independently
    recomputed three-profile/three-seed admission grid; approval later rebuilds
    that binding from the governed database before accepting it. Historical
    ``legacy_ambiguous`` strategies retain their original final-window checks.
    """

    config = version.get("config")
    config = config if isinstance(config, Mapping) else {}
    if str(config.get("horizon_profile") or LEGACY_AMBIGUOUS) == LEGACY_AMBIGUOUS:
        return []
    evidence = metrics.get("formal_validation")
    if not isinstance(evidence, Mapping):
        return ["pre-final stability evidence is required"]
    minimum_windows = int(config.get("min_rolling_windows") or 3)
    minimum_pass_rate = float(config.get("min_rolling_pass_rate", 0.60))

    if str(config.get("signal_source") or "factor_score") == "model_prediction":
        admission = evidence.get("model_admission")
        grid = admission.get("model_grid") if isinstance(admission, Mapping) else None
        multiple = grid.get("multiple_testing") if isinstance(grid, Mapping) else None
        profiles = grid.get("profiles") if isinstance(grid, Mapping) else None
        seeds = grid.get("seeds") if isinstance(grid, Mapping) else None
        expected_cells = len(REQUIRED_RESEARCH_PROFILES) * len(REQUIRED_MODEL_SEEDS)
        if (
            not isinstance(profiles, list)
            or profiles != list(REQUIRED_RESEARCH_PROFILES)
            or len(profiles) < minimum_windows
            or not isinstance(seeds, list)
            or seeds != list(REQUIRED_MODEL_SEEDS)
            or int(grid.get("cell_count") or 0) != expected_cells
            or not isinstance(multiple, Mapping)
            or multiple.get("gate_passed") is not True
            or admission.get("final_oos_opened") is not False
        ):
            return [
                "pre-final model stability requires the complete independent "
                "profile/seed grid and its multiple-testing gate"
            ]
        return []

    outer = evidence.get("outer_walk_forward")
    if not isinstance(outer, Mapping):
        return ["pre-final factor stability requires outer walk-forward evidence"]
    try:
        fold_count = int(outer.get("fold_count") or 0)
        pass_rate = float(outer.get("test_pass_rate"))
    except (TypeError, ValueError):
        return ["pre-final factor stability evidence is malformed"]
    if (
        outer.get("status") != "completed"
        or outer.get("passed") is not True
        or fold_count < minimum_windows
        or pass_rate < minimum_pass_rate
    ):
        return [
            "pre-final factor stability violates the configured outer "
            "walk-forward window or pass-rate gate"
        ]
    return []


def _valid_factor_score_incomplete_family_alternative(
    version: Mapping[str, Any], metrics: Mapping[str, Any]
) -> bool:
    """Validate the narrow conservative substitute for an unavailable old matrix."""

    config = version.get("config")
    config = config if isinstance(config, Mapping) else {}
    if str(config.get("signal_source") or "factor_score") != "factor_score":
        return False
    formal = metrics.get("formal_validation")
    if (
        not isinstance(formal, Mapping)
        or formal.get("contract_version") != FORMAL_VALIDATION_CONTRACT_VERSION
    ):
        return False
    multiple = formal.get("multiple_testing")
    bootstrap = formal.get("paired_block_bootstrap")
    deflated = metrics.get("deflated_sharpe")
    try:
        trials = int((deflated or {}).get("trials") or 0)
        audit_sha256 = str((multiple or {}).get("trial_count_audit_sha256") or "")
        eligibility_sha256 = str(
            (multiple or {}).get("eligibility_receipt_sha256") or ""
        )
        validated_multiple = validate_factor_score_incomplete_family_multiple_testing(
            multiple,
            paired_bootstrap=bootstrap if isinstance(bootstrap, Mapping) else {},
            trial_count=trials,
            trial_count_audit_sha256=audit_sha256,
            eligibility_receipt_sha256=eligibility_sha256,
        )
        validated_dsr = validate_factor_score_incomplete_family_dsr(
            deflated,
            trial_count=trials,
            trial_count_audit_sha256=audit_sha256,
            eligibility_receipt_sha256=eligibility_sha256,
        )
    except (AttributeError, TypeError, ValueError):
        return False
    return (
        validated_multiple.get("gate_passed") is True
        and validated_multiple.get("pbo", {}).get("status")
        == NOT_COMPUTABLE_INCOMPLETE_FAMILY_STATUS
        and validated_multiple.get("pbo", {}).get("pbo") is None
        and validated_dsr.get("probability") is None
        and metrics.get("deflated_sharpe_probability") is None
    )


def _formal_validation_failures(version: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    evidence = metrics.get("formal_validation")
    if not isinstance(evidence, dict):
        return ["formal validation evidence is required"]
    failures: list[str] = []
    if evidence.get("contract_version") != FORMAL_VALIDATION_CONTRACT_VERSION:
        failures.append("formal validation contract version is missing or obsolete")
    if evidence.get("status") != "passed" or metrics.get("formal_validation_passed") is not True:
        failures.append("formal validation suite did not pass")

    config = version.get("config", {})
    signal_source = str(config.get("signal_source") or "factor_score")
    model_prediction = signal_source == "model_prediction"
    history = evidence.get("pre_final_history")
    minimum_history_days = int(config.get("min_pre_final_history_days") or 2520)
    valid_history = False
    if isinstance(history, dict):
        requested_history = history.get("requested_periods")
        observed_history = history.get("observed_periods")
        final_test = history.get("final_test_periods")
        try:
            requested_start = date.fromisoformat(str(requested_history["start"]))
            requested_end = date.fromisoformat(str(requested_history["end"]))
            observed_start = date.fromisoformat(str(observed_history["start"]))
            observed_end = date.fromisoformat(str(observed_history["end"]))
            final_start = date.fromisoformat(str(final_test["start"]))
            final_end = date.fromisoformat(str(final_test["end"]))
            trading_days = int(history["trading_days"])
            recorded_minimum = int(history["minimum_trading_days"])
            embargo_days = int(history["embargo_trading_days"])
            recorded_embargo_minimum = int(history["minimum_embargo_trading_days"])
            history_execution = history["execution_model"]
            valid_history = (
                history.get("status") == "completed"
                and history.get("contract_version") == PRE_FINAL_HISTORY_CONTRACT_VERSION
                and requested_start <= observed_start <= observed_end <= requested_end
                and requested_end < final_start <= final_end
                and trading_days >= minimum_history_days
                and trading_days <= (observed_end - observed_start).days + 1
                and (observed_end - observed_start).days + 1 >= int(minimum_history_days * 7 / 5)
                and recorded_minimum == minimum_history_days
                and embargo_days >= int(config.get("outer_embargo_days") or 5)
                and recorded_embargo_minimum == int(config.get("outer_embargo_days") or 5)
                and history.get("overlaps_final_test") is False
                and history.get("uses_final_test_data") is False
                and isinstance(history_execution, dict)
                and history_execution.get("method") == "open"
                and history_execution.get("frequency") == "day"
                and history_execution.get("minute_execution_claimed") is False
            )
        except (KeyError, TypeError, ValueError):
            valid_history = False
    if not valid_history:
        failures.append(
            f"pre-final history must provide at least {minimum_history_days} isolated trading days"
        )

    outer = evidence.get("outer_walk_forward")
    coverage = outer.get("candidate_coverage") if isinstance(outer, dict) else {}
    trials = int((metrics.get("deflated_sharpe") or {}).get("trials") or 1)
    multiple = evidence.get("multiple_testing")
    conservative_incomplete_family = (
        signal_source == "factor_score"
        and trials > 1
        and isinstance(multiple, dict)
        and multiple.get("status")
        == CONSERVATIVE_BONFERRONI_INCOMPLETE_FAMILY_STATUS
        and multiple.get("contract_version")
        == FACTOR_SCORE_INCOMPLETE_FAMILY_MULTIPLE_TESTING_VERSION
    )
    minimum_outer_test_metric = float(config.get("minimum_outer_test_excess_return", 0.0))
    minimum_outer_test_pass_rate = float(config.get("minimum_outer_test_pass_rate", 0.60))
    outer_folds = outer.get("folds") if isinstance(outer, dict) else None

    def valid_outer_fold(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        try:
            test_metric = float(item.get("test_metric"))
        except (TypeError, ValueError):
            return False
        recorded_passed = item.get("test_passed")
        return (
            isinstance(recorded_passed, bool)
            and isfinite(test_metric)
            and recorded_passed == (test_metric > minimum_outer_test_metric)
        )

    valid_outer_folds = (
        isinstance(outer_folds, list)
        and bool(outer_folds)
        and all(valid_outer_fold(item) for item in outer_folds)
    )
    if valid_outer_folds:
        outer_test_metrics = [float(item["test_metric"]) for item in outer_folds]
        calculated_test_pass_rate = sum(bool(item["test_passed"]) for item in outer_folds) / len(
            outer_folds
        )
        calculated_mean_test_metric = sum(outer_test_metrics) / len(outer_test_metrics)
    else:
        calculated_test_pass_rate = float("-inf")
        calculated_mean_test_metric = float("-inf")
    try:
        recorded_test_pass_rate = float(outer.get("test_pass_rate"))
        recorded_mean_test_metric = float(outer.get("mean_test_metric"))
    except (AttributeError, TypeError, ValueError):
        recorded_test_pass_rate = float("-inf")
        recorded_mean_test_metric = float("-inf")

    if conservative_incomplete_family:
        valid_candidate_coverage = (
            int((coverage or {}).get("required_group_trials") or 0) == trials
            and int((coverage or {}).get("provided_candidates") or 0) == 1
            and (coverage or {}).get("scope") == FROZEN_STRATEGY_OUTER_SCOPE
            and (coverage or {}).get("selection_performed") is False
            and (coverage or {}).get("historical_candidate_matrix") == "incomplete"
            and (coverage or {}).get("trial_count_audit_sha256")
            == multiple.get("trial_count_audit_sha256")
            and outer.get("candidate_ids") == ["frozen-strategy"]
        )
    else:
        valid_candidate_coverage = (
            int((coverage or {}).get("required_group_trials") or 0) == trials
            and int((coverage or {}).get("provided_candidates") or 0) == trials
        )

    if model_prediction:
        admission = evidence.get("model_admission")
        provenance = metrics.get("provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        try:
            validated_admission = validate_model_formal_admission_binding(
                admission,
                config=config,
                dataset_identity_sha256=str(provenance.get("dataset_identity_sha256") or ""),
                pre_final_end=str(
                    ((history or {}).get("requested_periods") or {}).get("end") or ""
                ),
            )
        except (TypeError, ValueError) as exc:
            failures.append(str(exc))
            validated_admission = {}
        expected_binding_sha256 = validated_admission.get("binding_sha256")
        if (
            not isinstance(outer, dict)
            or outer.get("status") != MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_STATUS
            or outer.get("applicable") is not False
            or outer.get("reason") != MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_REASON
            or outer.get("final_oos_reopened") is not False
            or outer.get("independent_admission_binding_sha256") != expected_binding_sha256
        ):
            failures.append(
                "model formal validation must bind outer walk-forward to the "
                "independent pre-final grid without fabricating factor scores"
            )
    elif (
        not isinstance(outer, dict)
        or outer.get("status") != "completed"
        or outer.get("passed") is not True
        or int(outer.get("fold_count") or 0) < 3
        or not valid_candidate_coverage
        or recorded_test_pass_rate < minimum_outer_test_pass_rate
        or recorded_mean_test_metric <= minimum_outer_test_metric
        or not isinstance(outer_folds, list)
        or len(outer_folds) != int(outer.get("fold_count") or 0)
        or not valid_outer_folds
        or abs(recorded_test_pass_rate - calculated_test_pass_rate) > 1e-12
        or abs(recorded_mean_test_metric - calculated_mean_test_metric) > 1e-12
    ):
        failures.append(
            "outer walk-forward candidate coverage is invalid or its OOS gates did not pass"
        )

    baseline = version.get("config", {}).get("baseline_definition")
    expected_components = len(version.get("factors") or []) + len(
        (baseline or {}).get("factors") or []
    )
    ablation = evidence.get("ablation")
    if model_prediction:
        expected_binding_sha256 = (
            (evidence.get("model_admission") or {}).get("binding_sha256")
            if isinstance(evidence.get("model_admission"), dict)
            else None
        )
        if (
            not isinstance(ablation, dict)
            or ablation.get("status") != MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_STATUS
            or ablation.get("applicable") is not False
            or ablation.get("reason") != MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_REASON
            or ablation.get("final_oos_reopened") is not False
            or ablation.get("independent_admission_binding_sha256") != expected_binding_sha256
            or ablation.get("runs") != []
        ):
            failures.append(
                "model component ablation must use sealed independent admission "
                "evidence and must not reopen final OOS"
            )
    elif (
        not isinstance(ablation, dict)
        or ablation.get("status") != "passed"
        or len(ablation.get("runs") or []) != expected_components
        or any(
            not isinstance(item, dict)
            or item.get("passed") is not True
            or not isinstance(item.get("metrics"), dict)
            for item in ablation.get("runs") or []
        )
    ):
        failures.append("complete passing component ablation evidence is required")

    decay = evidence.get("signal_decay")
    if (
        not isinstance(decay, dict)
        or decay.get("status") != "completed"
        or decay.get("frontier_version") != SIGNAL_DECAY_FRONTIER_VERSION
        or decay.get("maximum_supported_delay_bars") is None
        or not decay.get("runs")
    ):
        failures.append("signal-decay evidence did not establish a supported response delay")

    bootstrap = evidence.get("paired_block_bootstrap")
    interval = bootstrap.get("confidence_interval_95") if isinstance(bootstrap, dict) else None
    if (
        not isinstance(bootstrap, dict)
        or bootstrap.get("status") != "ok"
        or not isinstance(interval, list)
        or len(interval) != 2
        or float(interval[0]) <= 0
    ):
        failures.append("paired moving-block bootstrap did not show positive baseline increment")

    if model_prediction:
        pbo = multiple.get("pbo") if isinstance(multiple, dict) else None
        admission_multiple = (
            (
                ((evidence.get("model_admission") or {}).get("quant_bundle") or {}).get(
                    "multiple_testing"
                )
                or ((evidence.get("model_admission") or {}).get("model_grid") or {}).get(
                    "multiple_testing"
                )
            )
            if isinstance(evidence.get("model_admission"), dict)
            else None
        )
        observed_multiple = (
            {
                key: multiple.get(key)
                for key in (
                    "contract_version",
                    "source",
                    "final_oos_opened",
                    "trial_definitions",
                    "trial_names",
                    "trial_count",
                    "holm_adjusted_p_values",
                    "eligible_trial_names",
                    "pbo",
                    "trial_daily_sharpes",
                    "gate_passed",
                )
            }
            if isinstance(multiple, dict)
            else None
        )
        pbo_valid = (
            isinstance(pbo, dict)
            and (
                (
                    trials == 1
                    and pbo.get("status") == "not_applicable_single_trial"
                    and pbo.get("pbo") is None
                )
                or (
                    trials > 1
                    and pbo.get("status") == "ok"
                    and pbo.get("pbo") is not None
                )
            )
        )
        valid_multiple = (
            isinstance(multiple, dict)
            and multiple.get("status") == "ok"
            and multiple.get("trial_count") == trials
            and len(multiple.get("holm_adjusted_p_values") or []) == trials
            and multiple.get("gate_passed") is True
            and pbo_valid
            and isinstance(admission_multiple, dict)
            and observed_multiple == admission_multiple
            and multiple.get("evidence_scope")
            == "independent_pre_final_run_trial_family"
            and multiple.get("independent_admission_binding_sha256")
            == (evidence.get("model_admission") or {}).get("binding_sha256")
        )
    elif trials == 1:
        valid_multiple = (
            isinstance(multiple, dict)
            and multiple.get("status") == "not_applicable_single_trial"
            and len(multiple.get("holm_adjusted_p_values") or []) == 1
        )
    elif conservative_incomplete_family:
        try:
            validated_multiple = validate_factor_score_incomplete_family_multiple_testing(
                multiple,
                paired_bootstrap=bootstrap if isinstance(bootstrap, dict) else {},
                trial_count=trials,
                trial_count_audit_sha256=str(
                    multiple.get("trial_count_audit_sha256") or ""
                ),
                eligibility_receipt_sha256=str(
                    multiple.get("eligibility_receipt_sha256") or ""
                ),
            )
        except (TypeError, ValueError):
            validated_multiple = {}
        valid_multiple = (
            bool(validated_multiple)
            and validated_multiple == multiple
            and multiple.get("gate_passed") is True
        )
    else:
        pbo = multiple.get("pbo") if isinstance(multiple, dict) else None
        valid_multiple = (
            isinstance(multiple, dict)
            and multiple.get("status") == "ok"
            and len(multiple.get("holm_adjusted_p_values") or []) == trials
            and isinstance(pbo, dict)
            and pbo.get("status") == "ok"
            and pbo.get("pbo") is not None
        )
    if not valid_multiple:
        failures.append(
            "multiple-testing evidence must cover the shared hypothesis-group trial count"
        )
    if conservative_incomplete_family and not _valid_factor_score_incomplete_family_alternative(
        version, metrics
    ):
        failures.append(
            "incomplete historical factor family must record DSR and PBO as not computable"
        )
    return failures


def _incomplete_family_manifest_binding_failures(
    manifest: Mapping[str, Any], metrics: Mapping[str, Any]
) -> list[str]:
    """Bind the conservative family size to the frozen manifest audit."""

    formal_evidence = metrics.get("formal_validation")
    formal_evidence = formal_evidence if isinstance(formal_evidence, Mapping) else {}
    multiple = formal_evidence.get("multiple_testing")
    if (
        not isinstance(multiple, Mapping)
        or multiple.get("status")
        != CONSERVATIVE_BONFERRONI_INCOMPLETE_FAMILY_STATUS
    ):
        return []
    hypothesis_group = manifest.get("hypothesis_group_evidence")
    trial_count_audit = (
        hypothesis_group.get("trial_count_audit")
        if isinstance(hypothesis_group, Mapping)
        else None
    )
    audit_sha256 = (
        _canonical_sha256(trial_count_audit)
        if isinstance(trial_count_audit, Mapping)
        else None
    )
    try:
        eligibility = require_incomplete_family_eligibility(
            manifest.get("incomplete_factor_family_eligibility") or {},
            hypothesis_group_evidence=(
                hypothesis_group if isinstance(hypothesis_group, Mapping) else {}
            ),
            strategy_version_id=str(manifest.get("strategy_version_id") or ""),
        )
        eligibility_sha256 = str(eligibility["receipt_sha256"])
    except (KeyError, TypeError, ValueError):
        eligibility_sha256 = ""
    outer = formal_evidence.get("outer_walk_forward")
    outer_coverage = outer.get("candidate_coverage") if isinstance(outer, Mapping) else None
    deflated = metrics.get("deflated_sharpe")
    try:
        manifest_trial_count = int(manifest.get("strategy_trial_count") or 0)
        multiple_trial_count = int(multiple.get("trial_count") or 0)
        deflated_trial_count = int((deflated or {}).get("trials") or 0)
    except (AttributeError, TypeError, ValueError):
        manifest_trial_count = 0
        multiple_trial_count = -1
        deflated_trial_count = -2
    if (
        audit_sha256 is None
        or manifest_trial_count <= 1
        or multiple_trial_count != manifest_trial_count
        or deflated_trial_count != manifest_trial_count
        or multiple.get("trial_count_audit_sha256") != audit_sha256
        or not _is_sha256(eligibility_sha256)
        or multiple.get("eligibility_receipt_sha256") != eligibility_sha256
        or not isinstance(outer_coverage, Mapping)
        or outer_coverage.get("trial_count_audit_sha256") != audit_sha256
        or outer_coverage.get("eligibility_receipt_sha256")
        != eligibility_sha256
        or not isinstance(deflated, Mapping)
        or deflated.get("trial_count_audit_sha256") != audit_sha256
        or deflated.get("eligibility_receipt_sha256") != eligibility_sha256
    ):
        return ["incomplete-family statistics do not bind the manifest trial-count audit"]
    return []


def _multifactor_manifest_failures(
    version: dict[str, Any], backtest: dict[str, Any], metrics: dict[str, Any]
) -> list[str]:
    provenance = metrics.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    artifact_root = Path(str(backtest["artifact_path"]))
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.is_file():
        return ["strategy backtest manifest artifact is missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ["strategy backtest manifest artifact is unreadable"]
    if not isinstance(manifest, dict):
        return ["strategy backtest manifest artifact must be a JSON object"]

    failures: list[str] = []
    evidence_mode = str(version.get("evidence_mode") or EVIDENCE_MODE_LEGACY)
    if str(backtest.get("evidence_mode") or EVIDENCE_MODE_LEGACY) != evidence_mode:
        failures.append("strategy version and backtest evidence modes differ")
    if evidence_mode == EVIDENCE_MODE_REPLAY:
        for label, value in (
            ("manifest", manifest),
            ("metrics", metrics),
            ("provenance", provenance),
        ):
            try:
                require_replay_markers(value, label=f"historical replay {label}")
            except ValueError as exc:
                failures.append(str(exc))
            if value.get("final_oos_opened") is not True:
                failures.append(
                    f"historical replay {label} must admit that the window was opened"
                )
        if any(
            value.get("evaluation_mode") != EVIDENCE_MODE_REPLAY
            for value in (manifest, metrics, provenance)
        ):
            failures.append("historical replay evaluation mode is inconsistent")
    elif evidence_mode == EVIDENCE_MODE_SEALED:
        if (
            manifest.get("evidence_mode") != EVIDENCE_MODE_SEALED
            or metrics.get("evidence_mode") != EVIDENCE_MODE_SEALED
            or provenance.get("evidence_mode") != EVIDENCE_MODE_SEALED
            or manifest.get("evaluation_mode") != "formal_final_oos"
            or metrics.get("evaluation_mode") != "formal_final_oos"
            or provenance.get("evaluation_mode") != "formal_final_oos"
            or manifest.get("final_oos_opened") is not True
            or metrics.get("final_oos_opened") is not True
            or provenance.get("final_oos_opened") is not True
        ):
            failures.append("sealed final OOS evidence authority is inconsistent")
    else:
        failures.append("strategy backtest evidence authority is ambiguous")
    failures.extend(
        _transparent_worker_runtime_failures(version, manifest, provenance)
    )
    if (
        provenance.get("artifact_manifest_version")
        != STRATEGY_BACKTEST_ARTIFACT_MANIFEST_VERSION
    ):
        failures.append("strategy backtest artifact manifest version is missing or obsolete")
    artifact_manifest_valid = False
    try:
        artifact_manifest = validate_backtest_artifact_manifest(
            artifact_root,
            expected_sha256=provenance.get("artifact_manifest_sha256"),
        )
        if int(provenance.get("artifact_manifest_file_count") or -1) != len(
            artifact_manifest["files"]
        ):
            failures.append("strategy backtest artifact manifest file count is inconsistent")
        else:
            artifact_manifest_valid = True
    except (OSError, TypeError, ValueError) as exc:
        failures.append(str(exc))
    if artifact_manifest_valid:
        daily_returns_path = artifact_root / "daily_returns.parquet"
        try:
            daily_returns = pd.read_parquet(daily_returns_path)
            formal_validation = metrics.get("formal_validation")
            claimed_bootstrap = (
                formal_validation.get("paired_block_bootstrap")
                if isinstance(formal_validation, Mapping)
                else None
            )
            validate_paired_bootstrap_evidence(
                claimed_bootstrap,
                daily_returns=daily_returns,
                parameters=paired_bootstrap_parameters_from_config(
                    version.get("config") or {}
                ),
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            failures.append(f"paired bootstrap artifact recomputation failed: {exc}")
    try:
        require_qlib_workflow_identity(provenance.get("qlib_workflow"))
    except ValueError as exc:
        failures.append(str(exc))
    manifest_sha256 = provenance.get("execution_manifest_sha256")
    if not _is_sha256(manifest_sha256) or _sha256_file(manifest_path) != manifest_sha256:
        failures.append("strategy backtest manifest does not match its SHA-256 provenance")
    config_sha256 = provenance.get("strategy_config_sha256")
    expected_config_sha256 = _canonical_sha256(version.get("config"))
    if config_sha256 != expected_config_sha256:
        failures.append("strategy config does not match its SHA-256 provenance")
    if _canonical_sha256(manifest.get("config")) != expected_config_sha256:
        failures.append("strategy backtest manifest config does not match the immutable version")
    for field, expected in (
        ("strategy_version_id", version.get("id")),
        ("dataset", backtest.get("dataset")),
        ("execution_dataset", backtest.get("execution_dataset")),
        ("benchmark", version.get("benchmark")),
        ("universe", version.get("universe")),
    ):
        if manifest.get(field) != expected:
            failures.append(f"strategy backtest manifest {field} does not match the run")
    recorded_periods = backtest.get("periods") or {}
    expected_final_periods = {
        "start": recorded_periods.get("start"),
        "end": recorded_periods.get("end"),
    }
    expected_history_periods = {
        "start": recorded_periods.get("historical_start"),
        "end": recorded_periods.get("historical_end"),
    }
    if manifest.get("periods") != expected_final_periods:
        failures.append("strategy backtest manifest final-test periods do not match the run")
    if manifest.get("historical_validation_periods") != expected_history_periods:
        failures.append("strategy backtest manifest pre-final history periods do not match the run")
    history_evidence = (
        (metrics.get("formal_validation") or {}).get("pre_final_history")
        if isinstance(metrics.get("formal_validation"), dict)
        else None
    )
    if (
        not isinstance(history_evidence, dict)
        or history_evidence.get("requested_periods") != expected_history_periods
        or history_evidence.get("final_test_periods") != expected_final_periods
    ):
        failures.append("pre-final history evidence periods do not match the immutable run")

    failures.extend(_incomplete_family_manifest_binding_failures(manifest, metrics))

    expected_factors = {
        str(item["factor_candidate_id"]): {
            "weight": float(item["weight"]),
            "direction": int(item["direction"]),
            "code_path": item.get("code_path"),
            "code_sha256": item.get("code_sha256"),
            "values_path": item.get("values_path"),
            "source_iteration": item.get("source_iteration"),
        }
        for item in version.get("factors", [])
    }
    manifest_items = manifest.get("factors")
    if not isinstance(manifest_items, list):
        failures.append("strategy backtest manifest factors are missing")
        return failures
    manifest_factors: dict[str, dict[str, Any]] = {}
    for item in manifest_items:
        if not isinstance(item, dict) or not str(item.get("candidate_id") or ""):
            failures.append("strategy backtest manifest contains an invalid factor")
            continue
        candidate_id = str(item["candidate_id"])
        if candidate_id in manifest_factors:
            failures.append("strategy backtest manifest contains duplicate factors")
        manifest_factors[candidate_id] = item
    if set(manifest_factors) != set(expected_factors):
        failures.append("strategy backtest manifest factors do not match the immutable version")
    for candidate_id, expected in expected_factors.items():
        item = manifest_factors.get(candidate_id)
        if item is None:
            continue
        try:
            numeric_matches = (
                abs(float(item.get("weight")) - expected["weight"]) <= 1e-12
                and int(item.get("direction")) == expected["direction"]
            )
        except (TypeError, ValueError):
            numeric_matches = False
        if not numeric_matches or item.get("code_sha256") != expected["code_sha256"]:
            failures.append(
                f"strategy backtest manifest factor {candidate_id} does not match the version"
            )
        expected_execution_mode = (
            "frozen_code_recompute" if expected["source_iteration"] is not None else "frozen_values"
        )
        if item.get("factor_execution_mode") != expected_execution_mode:
            failures.append(
                f"strategy backtest manifest factor {candidate_id} execution mode is invalid"
            )
    code_hashes = provenance.get("factor_code_sha256")
    if not isinstance(code_hashes, dict) or code_hashes != {
        candidate_id: item["code_sha256"] for candidate_id, item in expected_factors.items()
    }:
        failures.append("factor code provenance does not match the immutable version")
    value_hashes = provenance.get("factor_values_sha256")
    for candidate_id, item in expected_factors.items():
        for artifact_kind, path_value, hashes in (
            ("code", item["code_path"], code_hashes),
            ("values", item["values_path"], value_hashes),
        ):
            artifact = Path(str(path_value)) if path_value else None
            recorded = hashes.get(candidate_id) if isinstance(hashes, dict) else None
            if artifact is None or not artifact.is_file():
                failures.append(f"factor {candidate_id} {artifact_kind} artifact is missing")
            elif not _is_sha256(recorded) or _sha256_file(artifact) != recorded:
                failures.append(
                    f"factor {candidate_id} {artifact_kind} artifact does not match provenance"
                )
    formal_hashes = provenance.get("formal_factor_values_sha256")
    formal_evidence = provenance.get("formal_factor_recompute_evidence")
    if not isinstance(formal_hashes, dict) or set(formal_hashes) != set(expected_factors):
        failures.append("formal factor-value provenance is incomplete")
        formal_hashes = {}
    if not isinstance(formal_evidence, dict) or set(formal_evidence) != set(expected_factors):
        failures.append("formal factor recomputation evidence is incomplete")
        formal_evidence = {}
    artifact_root_resolved = artifact_root.resolve()
    for candidate_id, expected in expected_factors.items():
        manifest_factor = manifest_factors.get(candidate_id) or {}
        formal_artifact = manifest_factor.get("formal_factor_artifact")
        if not isinstance(formal_artifact, dict):
            failures.append(f"formal factor {candidate_id} artifact record is missing")
            continue
        relative_path = Path(str(formal_artifact.get("path") or ""))
        artifact = (artifact_root / relative_path).resolve()
        if (
            relative_path.is_absolute()
            or not artifact.is_relative_to(artifact_root_resolved)
            or not artifact.is_file()
        ):
            failures.append(f"formal factor {candidate_id} artifact is missing or outside the run")
            continue
        recorded_hash = formal_hashes.get(candidate_id)
        if (
            not _is_sha256(recorded_hash)
            or formal_artifact.get("sha256") != recorded_hash
            or _sha256_file(artifact) != recorded_hash
        ):
            failures.append(f"formal factor {candidate_id} artifact SHA-256 is invalid")
        evidence = formal_artifact.get("evidence")
        if not isinstance(evidence, dict) or evidence != formal_evidence.get(candidate_id):
            failures.append(f"formal factor {candidate_id} evidence does not match provenance")
            continue
        expected_execution_mode = (
            "frozen_code_recompute" if expected["source_iteration"] is not None else "frozen_values"
        )
        if (
            formal_artifact.get("execution_mode") != expected_execution_mode
            or evidence.get("authoritative_values_sha256") != recorded_hash
            or evidence.get("dataset_identity_sha256") != provenance.get("dataset_identity_sha256")
            or evidence.get("periods")
            != {
                "warmup_start": expected_history_periods["start"],
                "test_start": expected_final_periods["start"],
                "test_end": expected_final_periods["end"],
            }
        ):
            failures.append(f"formal factor {candidate_id} evidence binding is invalid")
        coverage = evidence.get("oos_coverage")
        if not (
            isinstance(coverage, dict)
            and coverage.get("contract_version") == "factor-oos-index-exact-v1"
            and coverage.get("test_start") == expected_final_periods["start"]
            and coverage.get("test_end") == expected_final_periods["end"]
            and coverage.get("index_exact_match") is True
            and int(coverage.get("trading_day_count") or 0) > 0
            and int(coverage.get("row_count") or 0) > 0
            and int(coverage.get("finite_row_count") or 0) > 0
            and coverage.get("coverage_gate_passed") is True
            and int(coverage.get("min_daily_finite_required") or 0) == FACTOR_MIN_DAILY_FINITE
            and float(coverage.get("min_coverage_ratio_required") or 0.0)
            == FACTOR_MIN_COVERAGE_RATIO
            and float(coverage.get("min_good_day_rate_required") or 0.0) == FACTOR_MIN_GOOD_DAY_RATE
            and float(coverage.get("good_day_rate") or 0.0) >= FACTOR_MIN_GOOD_DAY_RATE
        ):
            failures.append(f"formal factor {candidate_id} final OOS coverage is invalid")
        if expected_execution_mode == "frozen_code_recompute":
            pit = evidence.get("pit_invariance")
            if not (
                evidence.get("executor_version") == FACTOR_RECOMPUTE_EXECUTOR_VERSION
                and evidence.get("code_sha256") == expected["code_sha256"]
                and evidence.get("sandbox_mode") == "docker-isolated"
                and str(evidence.get("sandbox_image_id") or "").startswith("sha256:")
                and len(str(evidence.get("sandbox_image_id") or "")) == 71
                and evidence.get("network_mode") == "none"
                and evidence.get("root_filesystem_read_only") is True
                and evidence.get("capabilities_dropped") == "ALL"
                and evidence.get("no_new_privileges") is True
                and isinstance(pit, dict)
                and pit.get("contract_version") == FACTOR_PIT_CONTRACT_VERSION
                and pit.get("status") == "passed"
                and int(pit.get("cutpoint_count") or 0) >= 3
            ):
                failures.append(f"formal factor {candidate_id} PIT recomputation is invalid")
    failures.extend(
        baseline_manifest_failures(
            config=version.get("config") or {},
            factor_count=len(version.get("factors") or []),
            artifact_root=artifact_root,
            manifest=manifest,
            provenance=provenance,
        )
    )
    bundle_contract = (version.get("config") or {}).get("quant_bundle_factor_contract")
    bundle_contract_sha256 = (version.get("config") or {}).get(
        "quant_bundle_factor_contract_sha256"
    )
    manifest_bundle_factors = manifest.get("model_bundle_factors") or []
    if bundle_contract is None:
        if manifest_bundle_factors:
            failures.append("non-joint backtest contains model bundle factors")
    else:
        expected_bundle_factors = bundle_contract.get("factors")
        observed_bundle_factors = (
            [
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
                for item in manifest_bundle_factors
                if isinstance(item, dict)
            ]
            if isinstance(manifest_bundle_factors, list)
            else []
        )
        if (
            not isinstance(bundle_contract, dict)
            or bundle_contract.get("contract_version") != QUANT_BUNDLE_FACTOR_CONTRACT_VERSION
            or bundle_contract.get("weight_policy") != QUANT_BUNDLE_FACTOR_WEIGHT_POLICY
            or bundle_contract.get("score_factor_eligible") is not False
            or bundle_contract.get("standalone_promotion_required") is not False
            or _canonical_sha256(bundle_contract) != bundle_contract_sha256
            or not isinstance(expected_bundle_factors, list)
            or not expected_bundle_factors
            or observed_bundle_factors != expected_bundle_factors
            or len(observed_bundle_factors) != len(manifest_bundle_factors)
            or any(
                item.get("factor_execution_mode") != "frozen_code_recompute"
                for item in manifest_bundle_factors
                if isinstance(item, dict)
            )
        ):
            failures.append(
                "formal model bundle factor membership or policy does not match "
                "the immutable StrategySpec"
            )
        bundle_hashes = provenance.get("formal_model_bundle_factor_values_sha256")
        bundle_evidence = provenance.get("formal_model_bundle_factor_recompute_evidence")
        expected_bundle_ids = {
            str(item.get("candidate_id") or "")
            for item in (expected_bundle_factors or [])
            if isinstance(item, dict)
        }
        if (
            not isinstance(bundle_hashes, dict)
            or set(bundle_hashes) != expected_bundle_ids
            or not isinstance(bundle_evidence, dict)
            or set(bundle_evidence) != expected_bundle_ids
            or provenance.get("quant_bundle_factor_contract_sha256") != bundle_contract_sha256
        ):
            failures.append("formal model bundle factor provenance is incomplete")
            bundle_hashes = {}
            bundle_evidence = {}
        for item in manifest_bundle_factors if isinstance(manifest_bundle_factors, list) else []:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id") or "")
            formal_artifact = item.get("formal_factor_artifact")
            code_path = Path(str(item.get("code_path") or ""))
            code_sha256 = str(item.get("code_sha256") or "")
            if (
                not code_path.is_file()
                or not _is_sha256(code_sha256)
                or _sha256_file(code_path) != code_sha256
                or not isinstance(formal_artifact, dict)
            ):
                failures.append(
                    f"formal model bundle factor {candidate_id} code/artifact is invalid"
                )
                continue
            relative_path = Path(str(formal_artifact.get("path") or ""))
            artifact = (artifact_root / relative_path).resolve()
            recorded_hash = bundle_hashes.get(candidate_id)
            evidence = formal_artifact.get("evidence")
            if (
                relative_path.is_absolute()
                or not artifact.is_relative_to(artifact_root_resolved)
                or not artifact.is_file()
                or not _is_sha256(recorded_hash)
                or formal_artifact.get("sha256") != recorded_hash
                or _sha256_file(artifact) != recorded_hash
                or evidence != bundle_evidence.get(candidate_id)
            ):
                failures.append(
                    f"formal model bundle factor {candidate_id} immutable evidence is invalid"
                )
                continue
            pit = evidence.get("pit_invariance") if isinstance(evidence, dict) else None
            coverage = evidence.get("oos_coverage") if isinstance(evidence, dict) else None
            if (
                formal_artifact.get("execution_mode") != "frozen_code_recompute"
                or evidence.get("authoritative_values_sha256") != recorded_hash
                or evidence.get("code_sha256") != code_sha256
                or evidence.get("dataset_identity_sha256")
                != provenance.get("dataset_identity_sha256")
                or evidence.get("periods")
                != {
                    "warmup_start": expected_history_periods["start"],
                    "test_start": expected_final_periods["start"],
                    "test_end": expected_final_periods["end"],
                }
                or evidence.get("executor_version") != FACTOR_RECOMPUTE_EXECUTOR_VERSION
                or evidence.get("sandbox_mode") != "docker-isolated"
                or evidence.get("network_mode") != "none"
                or evidence.get("root_filesystem_read_only") is not True
                or evidence.get("capabilities_dropped") != "ALL"
                or evidence.get("no_new_privileges") is not True
                or not isinstance(pit, dict)
                or pit.get("contract_version") != FACTOR_PIT_CONTRACT_VERSION
                or pit.get("status") != "passed"
                or int(pit.get("cutpoint_count") or 0) < 3
                or not isinstance(coverage, dict)
                or coverage.get("contract_version") != "factor-oos-index-exact-v1"
                or coverage.get("test_start") != expected_final_periods["start"]
                or coverage.get("test_end") != expected_final_periods["end"]
                or coverage.get("index_exact_match") is not True
                or coverage.get("coverage_gate_passed") is not True
            ):
                failures.append(
                    f"formal model bundle factor {candidate_id} PIT/OOS proof is invalid"
                )
    failures.extend(
        formal_model_artifact_failures(
            config=version.get("config") or {},
            manifest=manifest,
            metrics=metrics,
            artifact_root=artifact_root,
            dataset_identity_sha256=str(provenance.get("dataset_identity_sha256") or ""),
            test_start=str(expected_final_periods["start"] or ""),
            test_end=str(expected_final_periods["end"] or ""),
        )
    )
    if str((version.get("config") or {}).get("signal_source") or "factor_score") == (
        "model_prediction"
    ):
        manifest_admission = manifest.get("model_formal_admission")
        metrics_admission = (
            (metrics.get("formal_validation") or {}).get("model_admission")
            if isinstance(metrics.get("formal_validation"), dict)
            else None
        )
        try:
            validate_model_formal_admission_binding(
                manifest_admission,
                config=version.get("config") or {},
                dataset_identity_sha256=str(provenance.get("dataset_identity_sha256") or ""),
                pre_final_end=str(expected_history_periods["end"] or ""),
            )
        except ValueError as exc:
            failures.append(str(exc))
        if manifest_admission != metrics_admission:
            failures.append(
                "formal model admission differs between manifest and validation evidence"
            )
    return failures


def _pair_artifact_failures(
    version: dict[str, Any], backtest: dict[str, Any], metrics: dict[str, Any]
) -> list[str]:
    artifact_root = Path(str(backtest["artifact_path"]))
    manifest_path = artifact_root / "manifest.json"
    pair_manifest_path = artifact_root / "pair_artifact_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ["pair backtest artifact manifests are missing or unreadable"]
    if not isinstance(manifest, dict) or not isinstance(pair_manifest, dict):
        return ["pair backtest artifact manifests must be JSON objects"]
    provenance = metrics.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    failures: list[str] = []
    try:
        require_qlib_workflow_identity(provenance.get("qlib_workflow"))
    except ValueError as exc:
        failures.append(str(exc))
    if provenance.get("execution_manifest_sha256") != _sha256_file(manifest_path):
        failures.append("pair execution manifest does not match its SHA-256 provenance")
    if provenance.get("pair_artifact_manifest_sha256") != _sha256_file(pair_manifest_path):
        failures.append("pair artifact manifest does not match its SHA-256 provenance")
    expected_config_sha256 = _canonical_sha256(version.get("config") or {})
    expected_pair = {
        key: (version.get("pair") or {}).get(key)
        for key in ("leg_y", "leg_x", "asset_class", "shorting_mode")
    }
    for candidate in (manifest, pair_manifest):
        observed_pair = {key: dict(candidate.get("pair") or {}).get(key) for key in expected_pair}
        if (
            candidate.get("backtest_id") != backtest.get("id")
            or candidate.get("strategy_version_id") != version.get("id")
            or candidate.get("dataset") != backtest.get("dataset")
            or candidate.get("periods") != backtest.get("periods")
            or candidate.get("execution_contract_hash") != version.get("execution_contract_hash")
            or observed_pair != expected_pair
        ):
            failures.append("pair artifact manifest does not match the immutable strategy/backtest")
            break
    if pair_manifest.get("format_version") != "pair-replay-artifact-v1":
        failures.append("pair artifact manifest format is unsupported")
    if pair_manifest.get("strategy_config_sha256") != expected_config_sha256:
        failures.append("pair artifact strategy config identity does not match the version")
    if _canonical_sha256(manifest.get("config") or {}) != expected_config_sha256:
        failures.append("pair execution manifest config does not match the version")
    files = pair_manifest.get("files")
    if not isinstance(files, dict):
        failures.append("pair artifact file manifest is missing")
        return failures
    for name in (
        "daily_returns.parquet",
        "daily_ledger.parquet",
        "kalman_spread.parquet",
        "trades.json",
        "rejections.json",
    ):
        evidence = files.get(name)
        path = artifact_root / name
        if (
            not isinstance(evidence, dict)
            or not path.is_file()
            or path.stat().st_size != int(evidence.get("bytes") or -1)
            or _sha256_file(path) != str(evidence.get("sha256") or "")
        ):
            failures.append(f"pair artifact {name} failed immutable verification")
    return failures


class StrategyStore:
    """Immutable strategy versions backed by promoted factors and audited approvals."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.engine = open_database(database_url)

    @staticmethod
    def _factor_evidence(
        connection: Any, factors: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        candidate_ids = [item["candidate_id"] for item in factors]
        candidate_rows = connection.execute(
            select(factor_candidates).where(factor_candidates.c.id.in_(candidate_ids))
        ).all()
        candidates = {str(row.id): row for row in candidate_rows}
        missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in candidates]
        if missing:
            raise ValueError(f"factor candidates not found: {', '.join(missing)}")
        not_promoted = [row.name for row in candidate_rows if row.status != "promoted"]
        if not_promoted:
            raise ValueError(
                "strategy versions may only use promoted factors: " + ", ".join(not_promoted)
            )
        evidence: dict[str, dict[str, Any]] = {}
        for candidate_id, candidate in candidates.items():
            if not candidate.promoted_evaluation_id or not _is_sha256(
                candidate.promotion_evidence_sha256
            ):
                raise ValueError(
                    f"promoted factor {candidate_id} has no immutable promotion evidence"
                )
            evaluation = connection.execute(
                select(factor_evaluations).where(
                    factor_evaluations.c.id == candidate.promoted_evaluation_id,
                    factor_evaluations.c.factor_candidate_id == candidate_id,
                )
            ).first()
            if not evaluation:
                raise ValueError(f"promoted factor {candidate_id} is not bound to its evaluation")
            candidate_data = row_dict(candidate)
            evaluation_data = row_dict(evaluation)
            admission_path = str(candidate.admission_path or "standalone")
            consensus = candidate.profile_consensus_json
            incremental = candidate.incremental_evidence_json
            if admission_path == "incremental":
                if consensus is not None or not isinstance(incremental, dict):
                    raise ValueError(
                        f"promoted factor {candidate_id} mixes standalone and incremental evidence"
                    )
                validate_incremental_evidence(dict(incremental))
                evaluation_ids = incremental.get("evaluation_ids")
                profiles = incremental.get("profiles")
                profile_periods = incremental.get("profile_periods")
                expected_profile_ids = {"recent_3y", "balanced_5y", "robust_10y"}
                if (
                    _canonical_sha256(incremental)
                    != candidate.incremental_evidence_sha256
                    or incremental.get("factor_candidate_id") != candidate_id
                    or incremental.get("candidate_code_sha256") != candidate.code_sha256
                    or incremental.get("candidate_values_sha256") != candidate.values_sha256
                    or not isinstance(evaluation_ids, dict)
                    or set(evaluation_ids) != expected_profile_ids
                    or len(set(evaluation_ids.values())) != len(expected_profile_ids)
                    or evaluation_ids.get("recent_3y") != str(evaluation.id)
                    or not isinstance(profiles, dict)
                    or set(profiles) != expected_profile_ids
                    or not isinstance(profile_periods, dict)
                    or set(profile_periods) != expected_profile_ids
                ):
                    raise ValueError(
                        f"promoted factor {candidate_id} has invalid incremental evidence"
                    )
                evaluation_rows = connection.execute(
                    select(factor_evaluations).where(
                        factor_evaluations.c.id.in_(list(evaluation_ids.values()))
                    )
                ).all()
                bound = {str(item.id): row_dict(item) for item in evaluation_rows}
                if len(bound) != len(expected_profile_ids):
                    raise ValueError(
                        f"promoted factor {candidate_id} incremental evaluations are missing"
                    )
                dataset_identities: set[str] = set()
                test_windows: set[tuple[date, date]] = set()
                for profile_id in sorted(expected_profile_ids):
                    profile_evaluation = bound.get(str(evaluation_ids[profile_id]))
                    if profile_evaluation is None:
                        raise ValueError(
                            f"promoted factor {candidate_id} incremental profile is missing"
                        )
                    layers = _validate_governed_factor_evaluation(
                        profile_evaluation,
                        candidate_data,
                        expected_profile_id=profile_id,
                        require_gate_passed=False,
                    )
                    if profile_id == "recent_3y" and layers["effect_status"] == "passed":
                        raise ValueError(
                            f"factor {candidate_id} must use standalone admission when it passes"
                        )
                    expected_periods = {
                        key: profile_evaluation[key].isoformat()
                        for key in (
                            "train_start",
                            "train_end",
                            "valid_start",
                            "valid_end",
                            "test_start",
                            "test_end",
                        )
                    }
                    if (
                        profile_periods.get(profile_id) != expected_periods
                        or (profiles.get(profile_id) or {}).get(
                            "evaluation_evidence_sha256"
                        )
                        != profile_evaluation.get("evidence_sha256")
                    ):
                        raise ValueError(
                            f"factor {candidate_id} incremental profile evidence changed"
                        )
                    dataset_identities.add(
                        str(profile_evaluation.get("dataset_identity_sha256") or "")
                    )
                    test_windows.add(
                        (
                            profile_evaluation["test_start"],
                            profile_evaluation["test_end"],
                        )
                    )
                if (
                    dataset_identities != {str(incremental.get("dataset_identity_sha256") or "")}
                    or "" in dataset_identities
                    or len(test_windows) != 1
                ):
                    raise ValueError(
                        f"factor {candidate_id} incremental profiles do not share one dataset/OOS"
                    )
                expected_promotion_evidence = _canonical_sha256(
                    {
                        "version": "factor-promotion-evidence-v3-incremental",
                        "primary_evaluation_evidence_sha256": evaluation.evidence_sha256,
                        "incremental_evidence_sha256": candidate.incremental_evidence_sha256,
                    }
                )
            elif incremental is not None:
                raise ValueError(
                    f"promoted factor {candidate_id} has incremental evidence on a standalone path"
                )
            elif consensus is None:
                profile_id = str(
                    ((evaluation_data.get("metrics_json") or {}).get("research_profile") or {}).get(
                        "id"
                    )
                    or ""
                )
                if profile_id in {"recent_3y", "balanced_5y", "robust_10y"}:
                    raise ValueError(
                        f"promoted factor {candidate_id} lacks multi-profile consensus evidence"
                    )
                _validate_governed_factor_evaluation(
                    evaluation_data,
                    candidate_data,
                    expected_profile_id=None,
                    require_gate_passed=True,
                )
                expected_promotion_evidence = evaluation.evidence_sha256
            else:
                consensus_ids = (
                    consensus.get("evaluation_ids") if isinstance(consensus, dict) else None
                )
                consensus_evidence = (
                    consensus.get("evaluation_evidence_sha256")
                    if isinstance(consensus, dict)
                    else None
                )
                expected_profile_ids = {"recent_3y", "balanced_5y", "robust_10y"}
                if (
                    not isinstance(consensus, dict)
                    or _canonical_sha256(consensus) != candidate.profile_consensus_sha256
                    or consensus.get("status") != "passed"
                    or consensus.get("candidate_id") != candidate_id
                    or consensus.get("candidate_code_sha256") != candidate.code_sha256
                    or consensus.get("candidate_values_sha256") != candidate.values_sha256
                    or not isinstance(consensus_ids, dict)
                    or set(consensus_ids) != expected_profile_ids
                    or not isinstance(consensus_evidence, dict)
                    or set(consensus_evidence) != expected_profile_ids
                    or consensus_ids.get("recent_3y") != str(evaluation.id)
                    or consensus_evidence.get("recent_3y") != evaluation.evidence_sha256
                ):
                    raise ValueError(
                        f"promoted factor {candidate_id} has invalid profile consensus evidence"
                    )
                consensus_rows = connection.execute(
                    select(factor_evaluations).where(
                        factor_evaluations.c.id.in_(list(consensus_ids.values()))
                    )
                ).all()
                bound_evidence = {str(row.id): row_dict(row) for row in consensus_rows}
                evaluations_by_profile: dict[str, dict[str, Any]] = {}
                for profile_id in sorted(expected_profile_ids):
                    profile_evaluation = bound_evidence.get(str(consensus_ids[profile_id]))
                    if (
                        profile_evaluation is None
                        or profile_evaluation.get("factor_candidate_id") != candidate_id
                        or profile_evaluation.get("evidence_sha256")
                        != str(consensus_evidence[profile_id])
                    ):
                        raise ValueError(
                            f"promoted factor {candidate_id} consensus evaluations "
                            "changed or are missing"
                        )
                    evaluations_by_profile[profile_id] = profile_evaluation
                _validate_governed_profile_consensus(
                    candidate_data,
                    evaluations_by_profile,
                    dict(consensus),
                )
                expected_promotion_evidence = _canonical_sha256(
                    {
                        "version": "factor-promotion-evidence-v2-profile-consensus",
                        "primary_evaluation_evidence_sha256": evaluation.evidence_sha256,
                        "profile_consensus_sha256": candidate.profile_consensus_sha256,
                    }
                )
            if (
                expected_promotion_evidence != candidate.promotion_evidence_sha256
                or not _is_sha256(evaluation.evidence_sha256)
            ):
                raise ValueError(f"promoted factor {candidate_id} has invalid promotion evidence")
            if (
                _canonical_sha256(evaluation.metrics_json) != evaluation.metrics_sha256
                or _canonical_sha256(evaluation.policy_json) != evaluation.policy_sha256
            ):
                raise ValueError(f"promoted factor {candidate_id} evaluation provenance is invalid")
            for artifact_kind, path_value, expected in (
                ("code", candidate.code_path, candidate.code_sha256),
                ("values", candidate.values_path, candidate.values_sha256),
                ("evaluation", evaluation.artifact_path, evaluation.artifact_sha256),
            ):
                artifact = Path(str(path_value)) if path_value else None
                if (
                    artifact is None
                    or not artifact.is_file()
                    or not _is_sha256(expected)
                    or _sha256_file(artifact) != expected
                ):
                    raise ValueError(
                        f"promoted factor {candidate_id} {artifact_kind} "
                        "evidence is missing or changed"
                    )
            if (
                evaluation.candidate_code_sha256 != candidate.code_sha256
                or evaluation.candidate_values_sha256 != candidate.values_sha256
            ):
                raise ValueError(
                    f"promoted factor {candidate_id} artifacts do not match its evaluation"
                )
            evidence[candidate_id] = {
                "id": str(evaluation.id),
                "direction": -1 if evaluation.metrics_json.get("direction") == "inverted" else 1,
                "label_horizon_sessions": int(candidate.label_horizon_days or 0),
            }
        return evidence

    @staticmethod
    def _require_factor_horizon_compatibility(
        config: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]]
    ) -> None:
        profile = str(config.get("horizon_profile") or LEGACY_AMBIGUOUS)
        if profile == LEGACY_AMBIGUOUS:
            return
        for candidate_id, item in evidence.items():
            try:
                require_label_horizon(profile, int(item["label_horizon_sessions"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"factor {candidate_id} label horizon is incompatible with {profile}: {exc}"
                ) from exc

    @staticmethod
    def _bundle_factor_evidence(
        connection: Any,
        *,
        bundle: Any,
        validated_bundle: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Bind joint factors as inseparable model feature columns.

        Joint RD-Agent candidates may remain non-promoted.  Their authority is
        the admitted bundle's independent factor/PIT evidence, never a fake
        standalone factor evaluation.  Already-promoted members remain valid
        only when their promotion hash was frozen into the same bundle.
        """

        manifest = dict(bundle.bundle_manifest_json or {})
        manifest_factors = manifest.get("factors")
        if not isinstance(manifest_factors, list) or not manifest_factors:
            raise ValueError("joint bundle has no immutable factor membership")
        member_ids = [str(item.get("candidate_id") or "") for item in manifest_factors]
        if (
            any(not candidate_id for candidate_id in member_ids)
            or len(set(member_ids)) != len(member_ids)
            or member_ids != [str(item) for item in (bundle.factor_candidate_ids_json or [])]
        ):
            raise ValueError("joint bundle factor order or membership is inconsistent")
        candidate_rows = connection.execute(
            select(factor_candidates).where(factor_candidates.c.id.in_(member_ids))
        ).all()
        candidates = {str(item.id): item for item in candidate_rows}
        independent_factors = {
            str(item.get("candidate_id") or ""): item
            for item in validated_bundle.get("factors") or []
            if isinstance(item, dict)
        }
        proofs = {
            str(item.get("candidate_id") or ""): item
            for item in validated_bundle.get("factor_recompute_evidence") or []
            if isinstance(item, dict)
        }
        if (
            set(candidates) != set(member_ids)
            or set(independent_factors) != set(member_ids)
            or set(proofs) != set(member_ids)
        ):
            raise ValueError("joint bundle factor evidence is incomplete")
        runtime_factors: list[dict[str, Any]] = []
        frozen_factors: list[dict[str, Any]] = []
        for index, (candidate_id, manifest_factor) in enumerate(
            zip(member_ids, manifest_factors, strict=True), start=1
        ):
            candidate = candidates[candidate_id]
            independent = independent_factors[candidate_id]
            proof = proofs[candidate_id]
            code_path = Path(str(candidate.code_path or ""))
            expected_code_sha256 = str(manifest_factor.get("code_sha256") or "")
            proof_payload = {key: value for key, value in proof.items() if key != "evidence_sha256"}
            execution = proof.get("execution")
            pit = proof.get("pit_invariance")
            pit_checks = pit.get("checks") if isinstance(pit, dict) else None
            submitted = proof.get("submitted_comparison")
            joint_proposal_member = (
                str(candidate.status) in {"awaiting_evaluation", "evaluating", "evaluated"}
                and (candidate.variables_json or {}).get("source")
                == "rdagent_fin_quant_joint_proposal"
                and (candidate.variables_json or {}).get("bundle_id") == str(bundle.id)
            )
            promoted_member = (
                str(candidate.status) == "promoted"
                and _is_sha256(candidate.promotion_evidence_sha256)
                and manifest_factor.get("promotion_evidence_sha256")
                == str(candidate.promotion_evidence_sha256)
            )
            if (
                str(candidate.research_run_id) != str(bundle.research_run_id)
                or not (joint_proposal_member or promoted_member)
                or str(candidate.code_sha256) != expected_code_sha256
                or independent.get("code_sha256") != expected_code_sha256
                or proof.get("code_sha256") != expected_code_sha256
                or proof.get("evidence_sha256") != _canonical_sha256(proof_payload)
                or not isinstance(execution, dict)
                or execution.get("code_sha256") != expected_code_sha256
                or not _is_sha256(execution.get("input_sha256"))
                or not _is_sha256(execution.get("output_sha256"))
                or execution.get("sandbox_mode") != "docker-isolated"
                or execution.get("network_mode") != "none"
                or execution.get("root_filesystem_read_only") is not True
                or execution.get("capabilities_dropped") != "ALL"
                or execution.get("no_new_privileges") is not True
                or not isinstance(pit, dict)
                or pit.get("contract_version") != FACTOR_PIT_CONTRACT_VERSION
                or pit.get("status") != "passed"
                or not isinstance(pit_checks, list)
                or len(pit_checks) < 3
                or any(
                    not isinstance(check, dict)
                    or check.get("invariant") is not True
                    or not _is_sha256(check.get("input_sha256"))
                    or not _is_sha256(check.get("output_sha256"))
                    for check in pit_checks
                )
                or (
                    isinstance(submitted, dict)
                    and submitted.get("available") is True
                    and (
                        submitted.get("exact_match") is not True
                        or submitted.get("index_exact_match") is not True
                    )
                )
                or not code_path.is_file()
                or _sha256_file(code_path) != expected_code_sha256
            ):
                raise ValueError(
                    f"joint bundle factor {candidate_id} changed or escaped its bundle"
                )
            item = {
                "candidate_id": candidate_id,
                "feature_name": f"factor_{index:03d}",
                "code_sha256": expected_code_sha256,
                "direction": 1,
                "weight": 1.0,
            }
            frozen_factors.append(item)
            runtime_factors.append(
                {
                    **item,
                    "name": str(candidate.name),
                    "code_path": str(code_path),
                    "source_iteration": candidate.source_iteration,
                    "factor_execution_mode": "frozen_code_recompute",
                }
            )
        contract = {
            "contract_version": QUANT_BUNDLE_FACTOR_CONTRACT_VERSION,
            "bundle_candidate_id": str(bundle.id),
            "bundle_manifest_sha256": str(bundle.bundle_manifest_sha256),
            "weight_policy": QUANT_BUNDLE_FACTOR_WEIGHT_POLICY,
            "score_factor_eligible": False,
            "standalone_promotion_required": False,
            "factors": frozen_factors,
        }
        horizon_factor_bundle = manifest.get("horizon_factor_bundle")
        if horizon_factor_bundle is not None:
            horizon_factor_bundle = validate_horizon_factor_bundle(
                horizon_factor_bundle
            )
            if (
                manifest.get("horizon_factor_bundle_sha256")
                != horizon_factor_bundle["bundle_sha256"]
                or horizon_factor_bundle["incremental_factors"]
                != [
                    {
                        "candidate_id": item["candidate_id"],
                        "code_sha256": item["code_sha256"],
                    }
                    for item in frozen_factors
                ]
            ):
                raise ValueError(
                    "joint bundle horizon factor package changed after admission"
                )
            contract.update(
                {
                    "horizon_factor_bundle": horizon_factor_bundle,
                    "horizon_factor_bundle_sha256": horizon_factor_bundle[
                        "bundle_sha256"
                    ],
                }
            )
        return runtime_factors, contract

    @staticmethod
    def _model_ensemble_signal_evidence(
        connection: Any,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Bind an equal-rank ensemble to its grid and every component model."""

        identity = model_signal_identity(config)
        if identity is None or not identity.get("model_ensemble_candidate_id"):
            raise ValueError("ensemble StrategySpec identity is missing")
        ensemble_id = str(identity["model_ensemble_candidate_id"])
        ensemble = connection.execute(
            select(model_ensemble_candidates).where(
                model_ensemble_candidates.c.id == ensemble_id
            )
        ).first()
        if ensemble is None or str(ensemble.status) != "research_admitted":
            raise ValueError("ensemble strategy requires a research-admitted candidate")
        manifest = dict(ensemble.manifest_json or {})
        admission = dict(ensemble.admission_evidence_json or {})
        independent = admission.get("independent_evidence")
        result_path = Path(str(admission.get("result_artifact_path") or ""))
        if (
            _canonical_sha256(manifest) != str(ensemble.manifest_sha256)
            or str(ensemble.manifest_sha256)
            != str(identity["model_ensemble_manifest_sha256"])
            or _canonical_sha256(
                {key: value for key, value in admission.items() if key != "evidence_sha256"}
            )
            != str(admission.get("evidence_sha256") or "")
            or str(admission.get("evidence_sha256") or "")
            != str(ensemble.admission_evidence_sha256)
            or str(ensemble.admission_evidence_sha256)
            != str(identity["model_ensemble_evidence_sha256"])
            or admission.get("status") != "passed"
            or not isinstance(independent, dict)
            or admission.get("independent_evidence_sha256")
            != independent.get("evidence_sha256")
            or _canonical_sha256(
                {key: value for key, value in independent.items() if key != "evidence_sha256"}
            )
            != independent.get("evidence_sha256")
            or not result_path.is_file()
            or _sha256_file(result_path)
            != str(admission.get("result_artifact_sha256") or "")
            or manifest.get("combiner") != "equal_rank"
            or manifest.get("stacking") is not False
            or independent.get("final_oos_opened") is not False
        ):
            raise ValueError("ensemble StrategySpec immutable admission is invalid")
        components = [dict(item) for item in ensemble.components_json or []]
        component_ids = [str(item.get("model_candidate_id") or "") for item in components]
        component_families = [str(item.get("model_family") or "") for item in components]
        if (
            component_ids != list(identity.get("model_component_candidate_ids") or [])
            or component_families != list(identity.get("model_component_families") or [])
            or len(set(component_ids)) != len(component_ids)
            or len(set(component_families)) != len(component_families)
            or not 2 <= len(component_ids) <= 3
        ):
            raise ValueError("ensemble StrategySpec component identities changed")
        grid_rows = connection.execute(
            select(model_ensemble_evaluations).where(
                model_ensemble_evaluations.c.model_ensemble_candidate_id == ensemble_id
            )
        ).all()
        required_grid = {
            (profile, seed)
            for profile in REQUIRED_RESEARCH_PROFILES
            for seed in REQUIRED_MODEL_SEEDS
        }
        observed_grid = {(str(row.profile_id), int(row.seed)) for row in grid_rows}
        primary_evaluation = next(
            (
                row
                for row in grid_rows
                if str(row.profile_id) == PRIMARY_MODEL_PROFILE
                and int(row.seed) == PRIMARY_MODEL_SEED
            ),
            None,
        )

        def cell_artifacts_valid(row: Any) -> bool:
            evidence = dict(row.evidence_json or {})
            for path_key, sha_key in (
                ("predictions_path", "predictions_sha256"),
                ("portfolio_report_path", "portfolio_report_sha256"),
            ):
                path = Path(str(evidence.get(path_key) or ""))
                if (
                    not path.is_file()
                    or not _is_sha256(evidence.get(sha_key))
                    or _sha256_file(path) != str(evidence.get(sha_key))
                ):
                    return False
            return True

        environment_path = Path(
            str(independent.get("execution_environment_path") or "")
        )
        if (
            observed_grid != required_grid
            or len(grid_rows) != len(required_grid)
            or primary_evaluation is None
            or str(primary_evaluation.id)
            != str(identity["model_ensemble_evaluation_id"])
            or any(
                str(row.gate_status) != "passed"
                or _canonical_sha256(dict(row.evidence_json or {}))
                != str(row.evidence_sha256)
                or (row.evidence_json or {}).get("final_oos_opened") is not False
                or (row.evidence_json or {}).get("ensemble_manifest_sha256")
                != str(ensemble.manifest_sha256)
                or not cell_artifacts_valid(row)
                for row in grid_rows
            )
            or not environment_path.is_file()
            or _sha256_file(environment_path)
            != str(independent.get("execution_environment_file_sha256") or "")
            or _canonical_sha256(
                dict(independent.get("execution_environment") or {})
            )
            != str(independent.get("execution_environment_sha256") or "")
        ):
            raise ValueError("ensemble independent evaluation grid is incomplete")

        component_evidence: list[dict[str, Any]] = []
        for component in components:
            candidate_id = str(component["model_candidate_id"])
            candidate = connection.execute(
                select(model_candidates).where(model_candidates.c.id == candidate_id)
            ).first()
            if candidate is None:
                raise ValueError("ensemble component model is missing")
            base = dict(candidate.base_features_manifest_json or {})
            candidate_manifest = dict(candidate.manifest_json or {})
            primary = connection.execute(
                select(model_evaluations).where(
                    model_evaluations.c.model_candidate_id == candidate_id,
                    model_evaluations.c.evidence_role == "independent_gate",
                    model_evaluations.c.gate_status == "passed",
                    model_evaluations.c.oos_vintage_id.is_(None),
                    model_evaluations.c.profile_id == PRIMARY_MODEL_PROFILE,
                    model_evaluations.c.seed == PRIMARY_MODEL_SEED,
                )
            ).first()
            if primary is None:
                raise ValueError("ensemble component has no primary independent cell")
            grid = prediction_grid_from_admission(row_dict(candidate))
            if (
                str(candidate.manifest_sha256)
                != str(component.get("model_manifest_sha256") or "")
                or str(candidate.admission_evidence_sha256)
                != str(component.get("model_admission_evidence_sha256") or "")
                or str(grid["prediction_grid_sha256"])
                != str(component.get("prediction_grid_sha256") or "")
            ):
                raise ValueError("ensemble component immutable evidence changed")
            component_config = {
                "signal_source": "model_prediction",
                "model_candidate_id": candidate_id,
                "model_evaluation_id": str(primary.id),
                "model_code_sha256": str(candidate.code_sha256),
                "model_recipe_sha256": str(candidate_manifest.get("recipe_sha256") or ""),
                "model_evidence_sha256": str(candidate.admission_evidence_sha256),
                "feature_set_id": str(base.get("feature_set_id") or ""),
                "feature_set_definition_sha256": str(
                    candidate.feature_set_definition_sha256
                ),
                "model_primary_profile_id": PRIMARY_MODEL_PROFILE,
                "model_primary_seed": PRIMARY_MODEL_SEED,
                "model_refit_policy": dict(MODEL_REFIT_POLICY),
                "model_refit_policy_sha256": MODEL_REFIT_POLICY_SHA256,
            }
            bound = StrategyStore._model_signal_evidence(connection, component_config)
            if bound is None:
                raise ValueError("ensemble component could not be independently bound")
            component_evidence.append(bound)
        datasets = {
            (
                str(item["candidate"].dataset),
                str(item["candidate"].dataset_identity_sha256),
                item["candidate"].pre_final_end,
                item["candidate"].final_oos_start,
                item["candidate"].final_oos_end,
            )
            for item in component_evidence
        }
        if len(datasets) != 1:
            raise ValueError("ensemble components do not share one sealed data/OOS contract")
        dataset_name, dataset_identity, pre_final_end, _oos_start, _oos_end = next(
            iter(datasets)
        )
        if (
            dataset_name != str(ensemble.dataset)
            or dataset_identity != str(ensemble.dataset_identity_sha256)
        ):
            raise ValueError("ensemble and component dataset identities differ")
        formal_admission_binding = build_model_ensemble_formal_admission_binding(
            config=config,
            dataset_identity_sha256=dataset_identity,
            pre_final_end=pre_final_end.isoformat(),
            ensemble_admission_evidence=admission,
            component_model_bindings=[
                item["formal_admission_binding"] for item in component_evidence
            ],
        )
        return {
            "signal_kind": "ensemble",
            "identity": identity,
            "ensemble_candidate": ensemble,
            "ensemble_evaluation": primary_evaluation,
            "components": component_evidence,
            "formal_admission_binding": formal_admission_binding,
            "candidate": None,
            "evaluation": None,
            "code_path": None,
            "bundle": None,
            "bundle_evaluation": None,
            "bundle_artifact": None,
            "bundle_factors": [],
            "bundle_factor_contract": None,
        }

    @staticmethod
    def _model_signal_evidence(
        connection: Any,
        config: dict[str, Any],
        *,
        require_bundle_factor_contract: bool = True,
    ) -> dict[str, Any] | None:
        """Re-bind a model signal to the independent pre-final evidence grid.

        RD-Agent's internal score is deliberately absent from this contract.
        The candidate remains research-only; this method merely makes it
        eligible to be sealed into a StrategySpec whose final OOS is consumed
        later by ``create_backtest``.
        """

        identity = model_signal_identity(config)
        if identity is None:
            return None
        if identity.get("model_ensemble_candidate_id") is not None:
            if not require_bundle_factor_contract:
                raise ValueError("ensemble strategy cannot request quant bundle factors")
            return StrategyStore._model_ensemble_signal_evidence(connection, config)
        candidate_id = str(identity["model_candidate_id"])
        candidate = connection.execute(
            select(model_candidates).where(model_candidates.c.id == candidate_id)
        ).first()
        if candidate is None:
            raise ValueError(f"model candidate {candidate_id!r} does not exist")
        if str(candidate.status) != "research_admitted" or bool(candidate.capital_eligible):
            raise ValueError(
                "model strategy requires a research-admitted candidate; RD-Agent "
                "internal scores and capital-marked research rows are not accepted"
            )
        admission = dict(candidate.admission_evidence_json or {})
        if (
            admission.get("final_oos_opened") is not False
            or admission.get("evidence_sha256") != str(candidate.admission_evidence_sha256)
            or identity["model_evidence_sha256"] != str(candidate.admission_evidence_sha256)
        ):
            raise ValueError("model candidate independent admission evidence is invalid")
        validate_independent_model_evidence(
            admission,
            candidate_id=candidate_id,
            dataset_identity_sha256=str(candidate.dataset_identity_sha256),
            pre_final_end=candidate.pre_final_end.isoformat(),
        )
        manifest = dict(candidate.manifest_json or {})
        recipe_sha256 = str((manifest.get("recipe") and manifest.get("recipe_sha256")) or "")
        if (
            _canonical_sha256(manifest) != str(candidate.manifest_sha256)
            or identity["model_code_sha256"] != str(candidate.code_sha256)
            or identity["model_recipe_sha256"] != recipe_sha256
            or identity["feature_set_definition_sha256"]
            != str(candidate.feature_set_definition_sha256)
            or identity["feature_set_id"]
            != str((candidate.base_features_manifest_json or {}).get("feature_set_id") or "")
        ):
            raise ValueError("model StrategySpec does not match the immutable candidate")
        artifact = connection.execute(
            select(research_run_artifacts).where(
                research_run_artifacts.c.id == candidate.code_artifact_id
            )
        ).first()
        code_path = Path(str(artifact.storage_path)) if artifact is not None else None
        if (
            artifact is None
            or str(artifact.status) != "recorded"
            or str(artifact.content_sha256) != str(candidate.code_sha256)
            or code_path is None
            or not code_path.is_file()
            or _sha256_file(code_path) != str(candidate.code_sha256)
        ):
            raise ValueError("model candidate code artifact is missing or changed")
        evaluation = connection.execute(
            select(model_evaluations).where(
                model_evaluations.c.id == str(identity["model_evaluation_id"]),
                model_evaluations.c.model_candidate_id == candidate_id,
            )
        ).first()
        if (
            evaluation is None
            or str(evaluation.evidence_role) != "independent_gate"
            or str(evaluation.gate_status) != "passed"
            or str(evaluation.profile_id) != PRIMARY_MODEL_PROFILE
            or int(evaluation.seed) != PRIMARY_MODEL_SEED
            or evaluation.oos_vintage_id is not None
            or str(evaluation.dataset_identity_sha256) != str(candidate.dataset_identity_sha256)
            or str(evaluation.candidate_manifest_sha256) != str(candidate.manifest_sha256)
            or evaluation.valid_end > candidate.pre_final_end
        ):
            raise ValueError("model StrategySpec is not bound to an independent evaluation")
        grid = connection.execute(
            select(
                model_evaluations.c.profile_id,
                model_evaluations.c.seed,
                model_evaluations.c.gate_status,
                model_evaluations.c.evidence_role,
                model_evaluations.c.oos_vintage_id,
                model_evaluations.c.candidate_manifest_sha256,
                model_evaluations.c.metrics_json,
                model_evaluations.c.metrics_sha256,
                model_evaluations.c.evidence_json,
                model_evaluations.c.evidence_sha256,
                model_evaluations.c.run_artifact_id,
            ).where(model_evaluations.c.model_candidate_id == candidate_id)
        ).all()
        required_grid = {
            (profile, seed)
            for profile in REQUIRED_RESEARCH_PROFILES
            for seed in REQUIRED_MODEL_SEEDS
        }
        independent_grid = [item for item in grid if str(item.evidence_role) == "independent_gate"]
        observed_grid = {(str(item.profile_id), int(item.seed)) for item in independent_grid}
        if (
            len(independent_grid) != len(required_grid)
            or observed_grid != required_grid
            or any(
                str(item.gate_status) != "passed"
                or item.oos_vintage_id is not None
                or str(item.candidate_manifest_sha256) != str(candidate.manifest_sha256)
                or _canonical_sha256(dict(item.metrics_json or {})) != str(item.metrics_sha256)
                or _canonical_sha256(dict(item.evidence_json or {})) != str(item.evidence_sha256)
                or (item.evidence_json or {}).get("source") != "independent_qlib_recompute"
                or (item.evidence_json or {}).get("final_oos_opened") is not False
                for item in independent_grid
            )
        ):
            raise ValueError("model candidate independent evaluation grid is incomplete")
        model_evaluation_artifact_ids = {str(item.run_artifact_id) for item in independent_grid}
        model_evaluation_artifacts = {
            str(item.id): item
            for item in connection.execute(
                select(research_run_artifacts).where(
                    research_run_artifacts.c.id.in_(model_evaluation_artifact_ids)
                )
            ).all()
        }
        if (
            set(model_evaluation_artifacts) != model_evaluation_artifact_ids
            or any(
                str(item.status) != "recorded"
                or not Path(str(item.storage_path)).is_file()
                or _sha256_file(Path(str(item.storage_path))) != str(item.content_sha256)
                for item in model_evaluation_artifacts.values()
            )
            or any(
                (item.evidence_json or {}).get("run_artifact_id") != str(item.run_artifact_id)
                or (item.evidence_json or {}).get("run_artifact_sha256")
                != str(model_evaluation_artifacts[str(item.run_artifact_id)].content_sha256)
                for item in independent_grid
            )
        ):
            raise ValueError("model independent evaluation artifacts are missing or changed")

        bundle: Any | None = None
        bundle_evaluation: Any | None = None
        bundle_artifact: Any | None = None
        bundle_factors: list[dict[str, Any]] = []
        bundle_factor_contract: dict[str, Any] | None = None
        if identity.get("quant_bundle_candidate_id") is not None:
            bundle_id = str(identity["quant_bundle_candidate_id"])
            bundle = connection.execute(
                select(quant_bundle_candidates).where(quant_bundle_candidates.c.id == bundle_id)
            ).first()
            bundle_evaluation = connection.execute(
                select(quant_bundle_evaluations).where(
                    quant_bundle_evaluations.c.id == str(identity["quant_bundle_evaluation_id"]),
                    quant_bundle_evaluations.c.quant_bundle_candidate_id == bundle_id,
                )
            ).first()
            if bundle is not None:
                bundle_artifact = connection.execute(
                    select(research_run_artifacts).where(
                        research_run_artifacts.c.id == bundle.bundle_artifact_id
                    )
                ).first()
            bundle_path = (
                Path(str(bundle_artifact.storage_path)) if bundle_artifact is not None else None
            )
            if (
                bundle is None
                or str(bundle.status) != "research_admitted"
                or bool(bundle.capital_eligible)
                or str(bundle.model_candidate_id) != candidate_id
                or str(bundle.bundle_manifest_sha256) != str(identity["quant_bundle_sha256"])
                or str(bundle.dataset_identity_sha256) != str(candidate.dataset_identity_sha256)
                or bundle.pre_final_end != candidate.pre_final_end
                or bundle.final_oos_start != candidate.final_oos_start
                or bundle.final_oos_end != candidate.final_oos_end
                or bundle_evaluation is None
                or str(bundle_evaluation.evidence_role) != "independent_gate"
                or str(bundle_evaluation.ablation) != "joint"
                or str(bundle_evaluation.gate_status) != "passed"
                or bundle_evaluation.oos_vintage_id is not None
                or str(bundle_evaluation.bundle_manifest_sha256)
                != str(bundle.bundle_manifest_sha256)
                or bundle_artifact is None
                or str(bundle_artifact.status) != "recorded"
                or str(bundle_artifact.content_sha256) != str(bundle.bundle_artifact_sha256)
                or bundle_path is None
                or not bundle_path.is_file()
                or _sha256_file(bundle_path) != str(bundle.bundle_artifact_sha256)
                or not isinstance(bundle.admission_evidence_json, dict)
                or bundle.admission_evidence_json.get("final_oos_opened") is not False
                or _canonical_sha256(dict(bundle.admission_evidence_json))
                != str(bundle.admission_evidence_sha256)
            ):
                raise ValueError("joint StrategySpec is not bound to a complete independent bundle")
            quant_grid = connection.execute(
                select(
                    quant_bundle_evaluations.c.ablation,
                    quant_bundle_evaluations.c.profile_id,
                    quant_bundle_evaluations.c.seed,
                    quant_bundle_evaluations.c.gate_status,
                    quant_bundle_evaluations.c.evidence_role,
                    quant_bundle_evaluations.c.oos_vintage_id,
                    quant_bundle_evaluations.c.bundle_manifest_sha256,
                    quant_bundle_evaluations.c.metrics_json,
                    quant_bundle_evaluations.c.metrics_sha256,
                    quant_bundle_evaluations.c.evidence_json,
                    quant_bundle_evaluations.c.evidence_sha256,
                    quant_bundle_evaluations.c.run_artifact_id,
                ).where(quant_bundle_evaluations.c.quant_bundle_candidate_id == bundle_id)
            ).all()
            required_quant_grid = {
                (ablation, profile, seed)
                for ablation in REQUIRED_QUANT_ABLATIONS
                for profile in REQUIRED_RESEARCH_PROFILES
                for seed in REQUIRED_MODEL_SEEDS
            }
            independent_quant_grid = [
                item for item in quant_grid if str(item.evidence_role) == "independent_gate"
            ]
            observed_quant_grid = {
                (str(item.ablation), str(item.profile_id), int(item.seed))
                for item in independent_quant_grid
            }
            if (
                len(independent_quant_grid) != len(required_quant_grid)
                or observed_quant_grid != required_quant_grid
                or any(
                    str(item.gate_status) != "passed"
                    or item.oos_vintage_id is not None
                    or str(item.bundle_manifest_sha256) != str(bundle.bundle_manifest_sha256)
                    or _canonical_sha256(dict(item.metrics_json or {})) != str(item.metrics_sha256)
                    or _canonical_sha256(dict(item.evidence_json or {}))
                    != str(item.evidence_sha256)
                    or (item.evidence_json or {}).get("source") != "independent_qlib_recompute"
                    or (item.evidence_json or {}).get("final_oos_opened") is not False
                    for item in independent_quant_grid
                )
            ):
                raise ValueError("quant bundle independent 27-cell grid is incomplete")
            quant_evaluation_artifact_ids = {
                str(item.run_artifact_id) for item in independent_quant_grid
            }
            quant_evaluation_artifacts = {
                str(item.id): item
                for item in connection.execute(
                    select(research_run_artifacts).where(
                        research_run_artifacts.c.id.in_(quant_evaluation_artifact_ids)
                    )
                ).all()
            }
            if (
                set(quant_evaluation_artifacts) != quant_evaluation_artifact_ids
                or any(
                    str(item.status) != "recorded"
                    or not Path(str(item.storage_path)).is_file()
                    or _sha256_file(Path(str(item.storage_path))) != str(item.content_sha256)
                    for item in quant_evaluation_artifacts.values()
                )
                or any(
                    (item.evidence_json or {}).get("run_artifact_id") != str(item.run_artifact_id)
                    or (item.evidence_json or {}).get("run_artifact_sha256")
                    != str(quant_evaluation_artifacts[str(item.run_artifact_id)].content_sha256)
                    for item in independent_quant_grid
                )
            ):
                raise ValueError("quant independent evaluation artifacts are missing or changed")
            bundle_admission = dict(bundle.admission_evidence_json or {})
            independent_bundle = bundle_admission.get("independent_bundle")
            if not isinstance(independent_bundle, dict):
                raise ValueError("quant bundle independent admission payload is missing")
            validated_bundle = validate_quant_bundle_evidence(
                independent_bundle,
                dataset_identity_sha256=str(candidate.dataset_identity_sha256),
            )
            if (
                bundle_admission.get("independent_bundle_sha256")
                != validated_bundle.get("bundle_sha256")
                or validated_bundle.get("id") != bundle_id
            ):
                raise ValueError("quant bundle independent admission hash is invalid")
            multiple_testing = validated_bundle.get("multiple_testing")
            multiple_returns_path = (
                Path(str(multiple_testing.get("returns_path") or ""))
                if isinstance(multiple_testing, dict)
                else None
            )
            if (
                multiple_returns_path is None
                or not multiple_returns_path.is_file()
                or _sha256_file(multiple_returns_path)
                != str(multiple_testing.get("returns_sha256") or "")
            ):
                raise ValueError(
                    "quant shared multiple-testing return matrix is missing or changed"
                )
            bundle_factors, bundle_factor_contract = StrategyStore._bundle_factor_evidence(
                connection,
                bundle=bundle,
                validated_bundle=validated_bundle,
            )
            recorded_contract = config.get("quant_bundle_factor_contract")
            recorded_contract_sha256 = config.get("quant_bundle_factor_contract_sha256")
            expected_contract_sha256 = _canonical_sha256(bundle_factor_contract)
            if require_bundle_factor_contract and (
                recorded_contract != bundle_factor_contract
                or recorded_contract_sha256 != expected_contract_sha256
                or identity.get("quant_bundle_factor_contract_sha256") != expected_contract_sha256
            ):
                raise ValueError(
                    "joint StrategySpec does not bind the complete immutable bundle factor contract"
                )
        formal_admission_binding = build_model_formal_admission_binding(
            config=config,
            candidate_manifest_sha256=str(candidate.manifest_sha256),
            dataset_identity_sha256=str(candidate.dataset_identity_sha256),
            pre_final_end=candidate.pre_final_end.isoformat(),
            model_admission_evidence=admission,
            model_admission_evidence_sha256=str(candidate.admission_evidence_sha256),
            quant_bundle_manifest_sha256=(
                str(bundle.bundle_manifest_sha256) if bundle is not None else None
            ),
            quant_bundle_admission_evidence=(
                dict(bundle.admission_evidence_json or {}) if bundle is not None else None
            ),
            quant_bundle_admission_evidence_sha256=(
                str(bundle.admission_evidence_sha256) if bundle is not None else None
            ),
        )
        return {
            "signal_kind": "model",
            "identity": identity,
            "candidate": candidate,
            "evaluation": evaluation,
            "code_path": str(code_path),
            "bundle": bundle,
            "bundle_evaluation": bundle_evaluation,
            "bundle_artifact": bundle_artifact,
            "bundle_factors": bundle_factors,
            "bundle_factor_contract": bundle_factor_contract,
            "formal_admission_binding": formal_admission_binding,
        }

    def create(
        self,
        *,
        name: str,
        description: str,
        benchmark: str,
        universe: str,
        factors: list[dict[str, Any]],
        config: dict[str, Any],
        actor: str,
        economic_hypothesis_group: str | None = None,
        hypothesis_group_cap: float = 0.70,
    ) -> dict[str, Any]:
        joint_bundle_requested = config.get("quant_bundle_candidate_id") is not None
        if joint_bundle_requested and factors:
            raise ValueError(
                "joint bundle factors are derived atomically; standalone factor "
                "arguments would allow component substitution"
            )
        config = _normalize_multifactor_contract(
            config,
            factor_count=0 if joint_bundle_requested else len(factors),
            creating_family=True,
        )
        if len({item["candidate_id"] for item in factors}) != len(factors):
            raise ValueError("factor candidates must be unique within a strategy version")
        total_weight = sum(abs(float(item["weight"])) for item in factors)
        if factors and total_weight <= 0:
            raise ValueError("factor weights must not all be zero")
        strategy_id = uuid.uuid4().hex
        group = str(economic_hypothesis_group or strategy_id).strip()
        if not group or len(group) > 200:
            raise ValueError("economic hypothesis group must contain 1 to 200 characters")
        if not 0 < float(hypothesis_group_cap) <= 0.70:
            raise ValueError("hypothesis group capital cap must be in (0, 0.70]")
        version_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                _lock_strategy_trial_family(connection, group)
                _require_strategy_source_artifact(
                    connection,
                    config,
                    expected_strategy_id=None,
                )
                model_evidence = self._model_signal_evidence(
                    connection,
                    config,
                    require_bundle_factor_contract=not joint_bundle_requested,
                )
                if joint_bundle_requested:
                    if (
                        model_evidence is None
                        or model_evidence.get("bundle_factor_contract") is None
                    ):
                        raise ValueError("joint strategy has no admitted quant bundle")
                    authoritative_contract = model_evidence["bundle_factor_contract"]
                    supplied_contract = config.get("quant_bundle_factor_contract")
                    supplied_contract_sha256 = config.get("quant_bundle_factor_contract_sha256")
                    authoritative_sha256 = _canonical_sha256(authoritative_contract)
                    if supplied_contract is not None and supplied_contract != (
                        authoritative_contract
                    ):
                        raise ValueError("supplied joint bundle factor contract changed")
                    if supplied_contract_sha256 is not None and (
                        supplied_contract_sha256 != authoritative_sha256
                    ):
                        raise ValueError("supplied joint bundle factor hash changed")
                    config = {
                        **config,
                        "quant_bundle_factor_contract": authoritative_contract,
                        "quant_bundle_factor_contract_sha256": authoritative_sha256,
                    }
                    config = _normalize_multifactor_contract(
                        config, factor_count=0, creating_family=True
                    )
                    self._model_signal_evidence(connection, config)
                    evaluation_evidence: dict[str, dict[str, Any]] = {}
                else:
                    evaluation_evidence = self._factor_evidence(connection, factors)
                    self._require_factor_horizon_compatibility(config, evaluation_evidence)
                connection.execute(
                    insert(strategies).values(
                        id=strategy_id,
                        name=name,
                        description=description,
                        status="draft",
                        economic_hypothesis_group=group,
                        hypothesis_group_cap=float(hypothesis_group_cap),
                        created_by=actor,
                        created_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    insert(strategy_versions).values(
                        id=version_id,
                        strategy_id=strategy_id,
                        version=1,
                        status="draft",
                        strategy_type="multifactor",
                        **_version_contract_columns(config, strategy_type="multifactor"),
                        benchmark=benchmark,
                        universe=universe,
                        config_json=config,
                        created_by=actor,
                        created_at=now,
                    )
                )
                factor_rows = [
                    {
                        "strategy_version_id": version_id,
                        "factor_candidate_id": item["candidate_id"],
                        "factor_evaluation_id": evaluation_evidence[item["candidate_id"]]["id"],
                        "weight": float(item["weight"]) / total_weight,
                        "direction": evaluation_evidence[item["candidate_id"]]["direction"],
                        "created_at": now,
                    }
                    for item in factors
                ]
                if factor_rows:
                    connection.execute(insert(strategy_factors), factor_rows)
                self._event(
                    connection,
                    strategy_id=strategy_id,
                    version_id=version_id,
                    event_type="strategy.created",
                    actor=actor,
                    payload={"benchmark": benchmark, "universe": universe},
                )
        except IntegrityError as exc:
            constraint = _integrity_constraint_name(exc)
            if constraint == "strategies_name_key":
                raise ValueError(f"strategy name {name!r} already exists") from exc
            detail = f" ({constraint})" if constraint else ""
            raise ValueError(
                f"strategy creation violated a database integrity constraint{detail}"
            ) from exc
        return self.get(strategy_id)

    def create_version(
        self,
        strategy_id: str,
        *,
        benchmark: str,
        universe: str,
        factors: list[dict[str, Any]],
        config: dict[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        """Create a new immutable version even when its inputs match an older one."""

        return self._create_version(
            strategy_id,
            benchmark=benchmark,
            universe=universe,
            factors=factors,
            config=config,
            actor=actor,
            reuse_exact=False,
        )

    def create_version_if_absent(
        self,
        strategy_id: str,
        *,
        benchmark: str,
        universe: str,
        factors: list[dict[str, Any]],
        config: dict[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        """Create once per exact normalized version contract under the family lock.

        This is deliberately separate from :meth:`create_version`: operator and
        research calls retain the established append-a-version behavior. Managed
        release reconciliation can opt into deterministic idempotence without a
        check-then-create race between two scheduler processes.
        """

        return self._create_version(
            strategy_id,
            benchmark=benchmark,
            universe=universe,
            factors=factors,
            config=config,
            actor=actor,
            reuse_exact=True,
        )

    def _create_version(
        self,
        strategy_id: str,
        *,
        benchmark: str,
        universe: str,
        factors: list[dict[str, Any]],
        config: dict[str, Any],
        actor: str,
        reuse_exact: bool,
    ) -> dict[str, Any]:
        joint_bundle_requested = config.get("quant_bundle_candidate_id") is not None
        if joint_bundle_requested and factors:
            raise ValueError(
                "joint bundle factors are derived atomically; standalone factor "
                "arguments would allow component substitution"
            )
        config = _normalize_multifactor_contract(
            config,
            factor_count=0 if joint_bundle_requested else len(factors),
            creating_family=False,
        )
        if len({item["candidate_id"] for item in factors}) != len(factors):
            raise ValueError("factor candidates must be unique within a strategy version")
        weight_inputs = (
            sorted(factors, key=lambda item: str(item["candidate_id"]))
            if reuse_exact
            else factors
        )
        total_weight = sum(abs(float(item["weight"])) for item in weight_inputs)
        if factors and total_weight <= 0:
            raise ValueError("factor weights must not all be zero")
        version_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                strategy = connection.execute(
                    select(strategies).where(strategies.c.id == strategy_id).with_for_update()
                ).first()
                if strategy is None:
                    raise KeyError(strategy_id)
                _lock_strategy_trial_family(
                    connection,
                    str(strategy.economic_hypothesis_group),
                )
                _require_strategy_source_artifact(
                    connection,
                    config,
                    expected_strategy_id=strategy_id,
                )
                family_type = connection.scalar(
                    select(strategy_versions.c.strategy_type)
                    .where(strategy_versions.c.strategy_id == strategy_id)
                    .limit(1)
                )
                if family_type != "multifactor":
                    raise ValueError("pair strategy families require a pair strategy version")
                model_evidence = self._model_signal_evidence(
                    connection,
                    config,
                    require_bundle_factor_contract=not joint_bundle_requested,
                )
                if joint_bundle_requested:
                    if (
                        model_evidence is None
                        or model_evidence.get("bundle_factor_contract") is None
                    ):
                        raise ValueError("joint strategy has no admitted quant bundle")
                    authoritative_contract = model_evidence["bundle_factor_contract"]
                    supplied_contract = config.get("quant_bundle_factor_contract")
                    supplied_contract_sha256 = config.get("quant_bundle_factor_contract_sha256")
                    authoritative_sha256 = _canonical_sha256(authoritative_contract)
                    if supplied_contract is not None and supplied_contract != (
                        authoritative_contract
                    ):
                        raise ValueError("supplied joint bundle factor contract changed")
                    if supplied_contract_sha256 is not None and (
                        supplied_contract_sha256 != authoritative_sha256
                    ):
                        raise ValueError("supplied joint bundle factor hash changed")
                    config = {
                        **config,
                        "quant_bundle_factor_contract": authoritative_contract,
                        "quant_bundle_factor_contract_sha256": authoritative_sha256,
                    }
                    config = _normalize_multifactor_contract(
                        config, factor_count=0, creating_family=False
                    )
                    self._model_signal_evidence(connection, config)
                    evaluation_evidence: dict[str, dict[str, Any]] = {}
                else:
                    evaluation_evidence = self._factor_evidence(connection, factors)
                    self._require_factor_horizon_compatibility(config, evaluation_evidence)
                expected_factors = sorted(
                    (
                        str(item["candidate_id"]),
                        str(evaluation_evidence[item["candidate_id"]]["id"]),
                        float(item["weight"]) / total_weight,
                        int(evaluation_evidence[item["candidate_id"]]["direction"]),
                    )
                    for item in factors
                )
                existing_version_id: str | None = None
                if reuse_exact:
                    expected_config_sha256 = _canonical_sha256(config)
                    if config.get("evidence_mode") == EVIDENCE_MODE_REPLAY:
                        current_recipe_rows = connection.execute(
                            select(
                                strategy_versions.c.id,
                                strategy_versions.c.config_json,
                                strategy_versions.c.benchmark,
                                strategy_versions.c.universe,
                            ).where(
                                strategy_versions.c.strategy_id == strategy_id,
                                strategy_versions.c.strategy_type == "multifactor",
                            )
                        ).all()
                        conflicting = [
                            str(candidate.id)
                            for candidate in current_recipe_rows
                            if dict(candidate.config_json or {}).get("recipe_version")
                            == config.get("recipe_version")
                            and (
                                _canonical_sha256(dict(candidate.config_json or {}))
                                != expected_config_sha256
                                or str(candidate.benchmark) != benchmark
                                or str(candidate.universe) != universe
                            )
                        ]
                        if conflicting:
                            raise ValueError(
                                "strategy family already contains a conflicting current "
                                "forward-only replay version"
                            )
                    candidates = connection.execute(
                        select(
                            strategy_versions.c.id,
                            strategy_versions.c.config_json,
                        ).where(
                            strategy_versions.c.strategy_id == strategy_id,
                            strategy_versions.c.strategy_type == "multifactor",
                            strategy_versions.c.benchmark == benchmark,
                            strategy_versions.c.universe == universe,
                        )
                    ).all()
                    exact: list[str] = []
                    for candidate in candidates:
                        if _canonical_sha256(
                            dict(candidate.config_json or {})
                        ) != expected_config_sha256:
                            continue
                        stored_factors = sorted(
                            (
                                str(row.factor_candidate_id),
                                str(row.factor_evaluation_id),
                                float(row.weight),
                                int(row.direction),
                            )
                            for row in connection.execute(
                                select(
                                    strategy_factors.c.factor_candidate_id,
                                    strategy_factors.c.factor_evaluation_id,
                                    strategy_factors.c.weight,
                                    strategy_factors.c.direction,
                                ).where(
                                    strategy_factors.c.strategy_version_id
                                    == str(candidate.id)
                                )
                            )
                        )
                        if stored_factors == expected_factors:
                            exact.append(str(candidate.id))
                    if len(exact) > 1:
                        raise ValueError(
                            "strategy family contains duplicate exact immutable versions"
                        )
                    if exact:
                        existing_version_id = exact[0]
                        version_id = existing_version_id
                if existing_version_id is None:
                    latest = connection.scalar(
                        select(func.max(strategy_versions.c.version)).where(
                            strategy_versions.c.strategy_id == strategy_id
                        )
                    )
                    version_number = int(latest or 0) + 1
                    connection.execute(
                        insert(strategy_versions).values(
                            id=version_id,
                            strategy_id=strategy_id,
                            version=version_number,
                            status="draft",
                            strategy_type="multifactor",
                            **_version_contract_columns(config, strategy_type="multifactor"),
                            benchmark=benchmark,
                            universe=universe,
                            config_json=config,
                            created_by=actor,
                            created_at=now,
                        )
                    )
                    factor_rows = [
                        {
                            "strategy_version_id": version_id,
                            "factor_candidate_id": item["candidate_id"],
                            "factor_evaluation_id": evaluation_evidence[item["candidate_id"]][
                                "id"
                            ],
                            "weight": float(item["weight"]) / total_weight,
                            "direction": evaluation_evidence[item["candidate_id"]][
                                "direction"
                            ],
                            "created_at": now,
                        }
                        for item in factors
                    ]
                    if factor_rows:
                        connection.execute(insert(strategy_factors), factor_rows)
                    connection.execute(
                        update(strategies)
                        .where(strategies.c.id == strategy_id)
                        .values(updated_at=now)
                    )
                    self._event(
                        connection,
                        strategy_id=strategy_id,
                        version_id=version_id,
                        event_type="strategy.version_created",
                        actor=actor,
                        payload={
                            "version": version_number,
                            "benchmark": benchmark,
                            "universe": universe,
                        },
                    )
                # The family row remains locked until this transaction exits,
                # so a concurrent idempotent caller observes the exact row.
        except IntegrityError as exc:
            raise ValueError("strategy version creation conflicted with another request") from exc
        return self.get_version(version_id)

    @staticmethod
    def _validate_pair_definition(
        *,
        leg_y: str,
        leg_x: str,
        asset_class: str,
        shorting_mode: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        first = leg_y.strip().upper()
        second = leg_x.strip().upper()
        if not first or not second or first == second:
            raise ValueError("pair strategy requires two distinct instruments")
        if asset_class not in {"etf", "stock", "mixed"}:
            raise ValueError("pair asset_class must be etf, stock, or mixed")
        if shorting_mode not in {"shadow_borrow", "margin_borrow"}:
            raise ValueError("pair shorting mode must be shadow_borrow or legacy margin_borrow")
        validated = PairTradingConfig(**config)
        return {
            "leg_y": first,
            "leg_x": second,
            "asset_class": asset_class,
            "shorting_mode": shorting_mode,
            "config": asdict(validated),
        }

    def create_pair(
        self,
        *,
        name: str,
        description: str,
        leg_y: str,
        leg_x: str,
        asset_class: str,
        shorting_mode: str,
        config: dict[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        if not name.strip() or not description.strip() or not actor.strip():
            raise ValueError("pair strategy name, description, and actor are required")
        definition = self._validate_pair_definition(
            leg_y=leg_y,
            leg_x=leg_x,
            asset_class=asset_class,
            shorting_mode=shorting_mode,
            config=config,
        )
        strategy_id = uuid.uuid4().hex
        version_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(strategies).values(
                        id=strategy_id,
                        name=name.strip(),
                        description=description.strip(),
                        status="draft",
                        created_by=actor.strip(),
                        created_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    insert(strategy_versions).values(
                        id=version_id,
                        strategy_id=strategy_id,
                        version=1,
                        status="draft",
                        strategy_type="pair",
                        **_version_contract_columns(definition["config"], strategy_type="pair"),
                        benchmark="CASH",
                        universe=f"pair:{definition['leg_y']}:{definition['leg_x']}",
                        config_json=definition["config"],
                        created_by=actor.strip(),
                        created_at=now,
                    )
                )
                connection.execute(
                    insert(strategy_pairs).values(
                        strategy_version_id=version_id,
                        leg_y=definition["leg_y"],
                        leg_x=definition["leg_x"],
                        asset_class=definition["asset_class"],
                        shorting_mode=definition["shorting_mode"],
                        created_at=now,
                    )
                )
                self._event(
                    connection,
                    strategy_id=strategy_id,
                    version_id=version_id,
                    event_type="strategy.pair_created",
                    actor=actor.strip(),
                    payload={key: definition[key] for key in ("leg_y", "leg_x", "asset_class")},
                )
        except IntegrityError as exc:
            raise ValueError(f"strategy name {name!r} already exists") from exc
        return self.get(strategy_id)

    def create_pair_version(
        self,
        strategy_id: str,
        *,
        leg_y: str,
        leg_x: str,
        asset_class: str,
        shorting_mode: str,
        config: dict[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        if not actor.strip():
            raise ValueError("pair strategy version actor is required")
        definition = self._validate_pair_definition(
            leg_y=leg_y,
            leg_x=leg_x,
            asset_class=asset_class,
            shorting_mode=shorting_mode,
            config=config,
        )
        version_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                strategy = connection.execute(
                    select(strategies).where(strategies.c.id == strategy_id).with_for_update()
                ).first()
                if strategy is None:
                    raise KeyError(strategy_id)
                family_type = connection.scalar(
                    select(strategy_versions.c.strategy_type)
                    .where(strategy_versions.c.strategy_id == strategy_id)
                    .limit(1)
                )
                if family_type != "pair":
                    raise ValueError("multifactor strategy families require promoted factors")
                latest = connection.scalar(
                    select(func.max(strategy_versions.c.version)).where(
                        strategy_versions.c.strategy_id == strategy_id
                    )
                )
                version_number = int(latest or 0) + 1
                connection.execute(
                    insert(strategy_versions).values(
                        id=version_id,
                        strategy_id=strategy_id,
                        version=version_number,
                        status="draft",
                        strategy_type="pair",
                        **_version_contract_columns(definition["config"], strategy_type="pair"),
                        benchmark="CASH",
                        universe=f"pair:{definition['leg_y']}:{definition['leg_x']}",
                        config_json=definition["config"],
                        created_by=actor.strip(),
                        created_at=now,
                    )
                )
                connection.execute(
                    insert(strategy_pairs).values(
                        strategy_version_id=version_id,
                        leg_y=definition["leg_y"],
                        leg_x=definition["leg_x"],
                        asset_class=definition["asset_class"],
                        shorting_mode=definition["shorting_mode"],
                        created_at=now,
                    )
                )
                connection.execute(
                    update(strategies).where(strategies.c.id == strategy_id).values(updated_at=now)
                )
                self._event(
                    connection,
                    strategy_id=strategy_id,
                    version_id=version_id,
                    event_type="strategy.pair_version_created",
                    actor=actor.strip(),
                    payload={
                        "version": version_number,
                        **{key: definition[key] for key in ("leg_y", "leg_x", "asset_class")},
                    },
                )
        except IntegrityError as exc:
            raise ValueError(
                "pair strategy version creation conflicted with another request"
            ) from exc
        return self.get_version(version_id)

    def get(self, strategy_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(strategies).where(strategies.c.id == strategy_id)
            ).first()
            if row is None:
                raise KeyError(strategy_id)
            result = row_dict(row)
            versions = connection.execute(
                select(strategy_versions)
                .where(strategy_versions.c.strategy_id == strategy_id)
                .order_by(strategy_versions.c.version.desc())
            ).all()
        result["versions"] = [self.get_version(str(item.id)) for item in versions]
        return result

    def get_by_name(self, name: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            strategy_id = connection.scalar(
                select(strategies.c.id).where(strategies.c.name == name)
            )
        return self.get(str(strategy_id)) if strategy_id else None

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        statement = select(strategies).order_by(strategies.c.updated_at.desc()).limit(limit)
        with self.engine.connect() as connection:
            ids = [str(row.id) for row in connection.execute(statement)]
        return [self.get(strategy_id) for strategy_id in ids]

    def list_pairs(self, limit: int = 100) -> list[dict[str, Any]]:
        statement = (
            select(strategies.c.id)
            .join(
                strategy_versions,
                strategy_versions.c.strategy_id == strategies.c.id,
            )
            .join(
                strategy_pairs,
                strategy_pairs.c.strategy_version_id == strategy_versions.c.id,
            )
            .group_by(strategies.c.id)
            .order_by(strategies.c.updated_at.desc())
            .limit(limit)
        )
        with self.engine.connect() as connection:
            ids = [str(row.id) for row in connection.execute(statement)]
        return [self.get(strategy_id) for strategy_id in ids]

    def get_version(self, version_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    strategy_versions,
                    strategies.c.economic_hypothesis_group,
                    strategies.c.hypothesis_group_cap,
                )
                .join(strategies, strategies.c.id == strategy_versions.c.strategy_id)
                .where(strategy_versions.c.id == version_id)
            ).first()
            if row is None:
                raise KeyError(version_id)
            factor_rows = connection.execute(
                select(
                    strategy_factors,
                    factor_candidates.c.name,
                    factor_candidates.c.code_path,
                    factor_candidates.c.values_path,
                    factor_candidates.c.code_sha256,
                    factor_candidates.c.source_iteration,
                    factor_candidates.c.experiment_family_id,
                    factor_candidates.c.label_horizon_days,
                    factor_candidates.c.experiment_count,
                )
                .join(
                    factor_candidates,
                    factor_candidates.c.id == strategy_factors.c.factor_candidate_id,
                )
                .where(strategy_factors.c.strategy_version_id == version_id)
            ).all()
            pair_row = connection.execute(
                select(strategy_pairs).where(strategy_pairs.c.strategy_version_id == version_id)
            ).first()
            model_evidence = self._model_signal_evidence(connection, dict(row.config_json or {}))
        result = row_dict(row)
        require_horizon_row(result)
        result["config"] = result.pop("config_json")
        result["horizon_contract"] = dict(result["horizon_contract_json"])
        result["factors"] = [row_dict(item) for item in factor_rows]
        result["pair"] = row_dict(pair_row) if pair_row else None
        result["capabilities"] = strategy_type_capabilities(result["strategy_type"])
        result["simulation_mode"] = (
            "shadow_pair" if result["strategy_type"] == "pair" else "paper"
        )
        result["factor_source_mode"] = result["config"].get("factor_source_mode", "promoted_only")
        result["baseline_definition_sha256"] = result["config"].get("baseline_definition_sha256")
        if model_evidence is None:
            result["model_signal"] = None
        elif model_evidence.get("signal_kind") == "ensemble":
            ensemble = model_evidence["ensemble_candidate"]
            components: list[dict[str, Any]] = []
            for component in model_evidence["components"]:
                candidate = component["candidate"]
                candidate_manifest = dict(candidate.manifest_json or {})
                components.append(
                    {
                        **component["identity"],
                        "code_path": component["code_path"],
                        "model_type": str(candidate.model_type),
                        "architecture": dict(candidate.architecture_json or {}),
                        "model_hyperparameters": dict(
                            candidate.model_hyperparameters_json or {}
                        ),
                        "training_hyperparameters": dict(
                            candidate.training_hyperparameters_json or {}
                        ),
                        "model_manifest_sha256": str(candidate.manifest_sha256),
                        "dataset": str(candidate.dataset),
                        "dataset_identity_sha256": str(
                            candidate.dataset_identity_sha256
                        ),
                        "dataset_lineage_id": candidate_manifest.get(
                            "dataset_lineage_id"
                        ),
                        "pre_final_end": candidate.pre_final_end.isoformat(),
                        "final_oos_start": candidate.final_oos_start.isoformat(),
                        "final_oos_end": candidate.final_oos_end.isoformat(),
                        "recipe": dict(candidate_manifest.get("recipe") or {}),
                        "primary_profile_id": PRIMARY_MODEL_PROFILE,
                        "primary_seed": PRIMARY_MODEL_SEED,
                        "refit_policy": dict(MODEL_REFIT_POLICY),
                        "refit_policy_sha256": MODEL_REFIT_POLICY_SHA256,
                        "primary_training_periods": {
                            "train_start": component[
                                "evaluation"
                            ].train_start.isoformat(),
                            "train_end": component[
                                "evaluation"
                            ].train_end.isoformat(),
                            "valid_start": component[
                                "evaluation"
                            ].valid_start.isoformat(),
                            "valid_end": component[
                                "evaluation"
                            ].valid_end.isoformat(),
                            "seed": int(component["evaluation"].seed),
                        },
                    }
                )
            first = components[0]
            result["model_signal"] = {
                **model_evidence["identity"],
                "signal_kind": "ensemble",
                "combiner": "equal_rank",
                "stacking": False,
                "ensemble_manifest": dict(ensemble.manifest_json or {}),
                "ensemble_manifest_sha256": str(ensemble.manifest_sha256),
                "ensemble_admission_binding": model_evidence[
                    "formal_admission_binding"
                ],
                "components": components,
                "dataset": first["dataset"],
                "dataset_identity_sha256": first["dataset_identity_sha256"],
                "dataset_lineage_id": first["dataset_lineage_id"],
                "pre_final_end": first["pre_final_end"],
                "final_oos_start": first["final_oos_start"],
                "final_oos_end": first["final_oos_end"],
                "primary_profile_id": PRIMARY_MODEL_PROFILE,
                "primary_seed": PRIMARY_MODEL_SEED,
                "primary_training_periods": dict(
                    first["primary_training_periods"]
                ),
                "refit_policy": dict(MODEL_REFIT_POLICY),
                "refit_policy_sha256": MODEL_REFIT_POLICY_SHA256,
                "bundle_factors": [],
            }
        else:
            candidate = model_evidence["candidate"]
            candidate_manifest = dict(candidate.manifest_json or {})
            bundle = model_evidence.get("bundle")
            result["model_signal"] = {
                **model_evidence["identity"],
                "code_path": model_evidence["code_path"],
                "model_type": str(candidate.model_type),
                "architecture": dict(candidate.architecture_json or {}),
                "model_hyperparameters": dict(candidate.model_hyperparameters_json or {}),
                "training_hyperparameters": dict(candidate.training_hyperparameters_json or {}),
                "model_manifest_sha256": str(candidate.manifest_sha256),
                "dataset": str(candidate.dataset),
                "dataset_identity_sha256": str(candidate.dataset_identity_sha256),
                "dataset_lineage_id": candidate_manifest.get("dataset_lineage_id"),
                "pre_final_end": candidate.pre_final_end.isoformat(),
                "final_oos_start": candidate.final_oos_start.isoformat(),
                "final_oos_end": candidate.final_oos_end.isoformat(),
                "recipe": dict(candidate_manifest.get("recipe") or {}),
                "primary_profile_id": PRIMARY_MODEL_PROFILE,
                "primary_seed": PRIMARY_MODEL_SEED,
                "refit_policy": dict(MODEL_REFIT_POLICY),
                "refit_policy_sha256": MODEL_REFIT_POLICY_SHA256,
                "primary_training_periods": {
                    "train_start": model_evidence["evaluation"].train_start.isoformat(),
                    "train_end": model_evidence["evaluation"].train_end.isoformat(),
                    "valid_start": model_evidence["evaluation"].valid_start.isoformat(),
                    "valid_end": model_evidence["evaluation"].valid_end.isoformat(),
                    "seed": int(model_evidence["evaluation"].seed),
                },
                "bundle_manifest": (
                    dict(bundle.bundle_manifest_json or {}) if bundle is not None else None
                ),
                "bundle_artifact_path": (
                    str(model_evidence["bundle_artifact"].storage_path)
                    if bundle is not None
                    else None
                ),
                "bundle_factor_contract": model_evidence.get("bundle_factor_contract"),
                "bundle_factors": model_evidence.get("bundle_factors") or [],
            }
        return result

    def hypothesis_group_evidence(
        self,
        version_id: str,
        *,
        connection: Any | None = None,
    ) -> dict[str, Any]:
        """Return the immutable family-wide trial count used by formal gates.

        Factor experiment families carry their declared count (including
        non-winning variants). Ordinary versions/model wrappers count as new
        trials. A version may share its predecessor's statistical trial only
        through an append-only, validated pre-result implementation-repair
        receipt which proves that no performance information was used.
        Renaming, failing, or versioning alone can therefore never reset
        DSR/PBO inputs.
        """

        if connection is None:
            version = self.get_version(version_id)
        else:
            version_row = connection.execute(
                select(
                    strategy_versions,
                    strategies.c.economic_hypothesis_group,
                    strategies.c.hypothesis_group_cap,
                )
                .join(strategies, strategies.c.id == strategy_versions.c.strategy_id)
                .where(strategy_versions.c.id == version_id)
            ).first()
            if version_row is None:
                raise KeyError(version_id)
            version = row_dict(version_row)
        group = str(version["economic_hypothesis_group"])
        connection_scope = (
            self.engine.connect() if connection is None else nullcontext(connection)
        )
        with connection_scope as connection:
            version_rows = connection.execute(
                select(strategy_versions.c.id, strategy_versions.c.config_json)
                .join(strategies, strategies.c.id == strategy_versions.c.strategy_id)
                .where(
                    strategies.c.economic_hypothesis_group == group,
                    strategy_versions.c.is_legacy.is_(False),
                )
            ).all()
            version_ids = {str(row.id) for row in version_rows}
            version_configs = {
                str(row.id): dict(row.config_json or {}) for row in version_rows
            }
            repair_rows = connection.execute(
                select(transparent_baseline_pre_result_repairs)
            ).all()
            repair_source_backtest_ids = {
                str(item)
                for row in repair_rows
                for item in list(row.source_backtest_ids_json or [])
            }
            backtest_rows = connection.execute(
                select(
                    backtest_runs.c.id,
                    backtest_runs.c.strategy_version_id,
                ).where(
                    or_(
                        backtest_runs.c.strategy_version_id.in_(version_ids),
                        backtest_runs.c.id.in_(repair_source_backtest_ids),
                    )
                )
            ).all()
            backtests_by_id = {str(row.id): row for row in backtest_rows}
            repair_audit_ids = {
                int(row.source_audit_event_id) for row in repair_rows
            }
            repair_audits = {
                int(row.id): row
                for row in (
                    connection.execute(
                        select(audit_events).where(
                            audit_events.c.id.in_(repair_audit_ids)
                        )
                    ).all()
                    if repair_audit_ids
                    else []
                )
            }
            factor_rows = connection.execute(
                select(
                    factor_candidates.c.experiment_family_id,
                    factor_candidates.c.id,
                    factor_candidates.c.experiment_count,
                )
                .join(
                    strategy_factors,
                    strategy_factors.c.factor_candidate_id == factor_candidates.c.id,
                )
                .join(
                    strategy_versions,
                    strategy_versions.c.id == strategy_factors.c.strategy_version_id,
                )
                .join(strategies, strategies.c.id == strategy_versions.c.strategy_id)
                .where(
                    strategies.c.economic_hypothesis_group == group,
                    strategy_versions.c.is_legacy.is_(False),
                )
            ).all()
            bound_models: dict[str, list[str]] = {}
            bound_bundles: dict[str, list[str]] = {}
            bound_ensembles: dict[str, list[str]] = {}
            for row in version_rows:
                config = dict(row.config_json or {})
                model_candidate_id = str(config.get("model_candidate_id") or "")
                bundle_candidate_id = str(config.get("quant_bundle_candidate_id") or "")
                ensemble_candidate_id = str(
                    config.get("model_ensemble_candidate_id") or ""
                )
                if model_candidate_id:
                    bound_models.setdefault(model_candidate_id, []).append(str(row.id))
                if bundle_candidate_id:
                    bound_bundles.setdefault(bundle_candidate_id, []).append(str(row.id))
                if ensemble_candidate_id:
                    bound_ensembles.setdefault(ensemble_candidate_id, []).append(
                        str(row.id)
                    )

            model_rows = (
                connection.execute(
                    select(model_candidates.c.id, model_candidates.c.research_run_id).where(
                        model_candidates.c.id.in_(sorted(bound_models))
                    )
                ).all()
                if bound_models
                else []
            )
            bundle_rows = (
                connection.execute(
                    select(
                        quant_bundle_candidates.c.id,
                        quant_bundle_candidates.c.research_run_id,
                    ).where(quant_bundle_candidates.c.id.in_(sorted(bound_bundles)))
                ).all()
                if bound_bundles
                else []
            )
            ensemble_rows = (
                connection.execute(
                    select(
                        model_ensemble_candidates.c.id,
                        model_ensemble_candidates.c.tournament_id,
                    ).where(
                        model_ensemble_candidates.c.id.in_(sorted(bound_ensembles))
                    )
                ).all()
                if bound_ensembles
                else []
            )
            missing_models = set(bound_models) - {str(row.id) for row in model_rows}
            missing_bundles = set(bound_bundles) - {str(row.id) for row in bundle_rows}
            missing_ensembles = set(bound_ensembles) - {
                str(row.id) for row in ensemble_rows
            }
            if missing_models or missing_bundles or missing_ensembles:
                raise ValueError(
                    "hypothesis-group model/bundle/ensemble bindings reference missing candidates"
                )
            ensemble_tournament_ids = {
                str(row.tournament_id) for row in ensemble_rows
            }
            ensemble_tournament_rows = (
                connection.execute(
                    select(
                        model_ensemble_candidates.c.id,
                        model_ensemble_candidates.c.tournament_id,
                    ).where(
                        model_ensemble_candidates.c.tournament_id.in_(
                            ensemble_tournament_ids
                        )
                    )
                ).all()
                if ensemble_tournament_ids
                else []
            )
            quant_run_ids = {str(row.research_run_id) for row in bundle_rows}
            model_run_ids = {
                str(row.research_run_id)
                for row in model_rows
                if str(row.research_run_id) not in quant_run_ids
            }
            model_run_all_rows = (
                connection.execute(
                    select(model_candidates.c.id, model_candidates.c.research_run_id).where(
                        model_candidates.c.research_run_id.in_(model_run_ids)
                    )
                ).all()
                if model_run_ids
                else []
            )
            quant_run_all_rows = (
                connection.execute(
                    select(
                        quant_bundle_candidates.c.id,
                        quant_bundle_candidates.c.research_run_id,
                    ).where(quant_bundle_candidates.c.research_run_id.in_(quant_run_ids))
                ).all()
                if quant_run_ids
                else []
            )
            model_run_counts = {
                run_id: sum(
                    1 for row in model_run_all_rows if str(row.research_run_id) == run_id
                )
                for run_id in model_run_ids
            }
            quant_run_counts = {
                run_id: sum(
                    1 for row in quant_run_all_rows if str(row.research_run_id) == run_id
                )
                for run_id in quant_run_ids
            }
        family_counts: dict[str, int] = {}
        for row in factor_rows:
            family = str(row.experiment_family_id or row.id)
            family_counts[family] = max(
                family_counts.get(family, 0),
                int(row.experiment_count or 1),
            )
        sorted_version_ids = sorted(version_ids)
        trial_lineage = build_strategy_trial_lineage(
            version_configs=version_configs,
            backtests_by_id=backtests_by_id,
            repair_rows=repair_rows,
            repair_audits=repair_audits,
        )
        strategy_trial_count = int(trial_lineage["strategy_trial_count"])
        factor_trial_count = sum(family_counts.values())
        model_trial_count = sum(model_run_counts.values())
        quant_trial_count = sum(quant_run_counts.values()) * len(REQUIRED_QUANT_ABLATIONS)
        ensemble_trial_count = len(ensemble_tournament_rows)
        research_trial_count = (
            factor_trial_count
            + model_trial_count
            + quant_trial_count
            + ensemble_trial_count
        )
        shared_count = max(1, strategy_trial_count, research_trial_count)
        return {
            "economic_hypothesis_group": group,
            "hypothesis_group_cap": float(version["hypothesis_group_cap"]),
            "shared_experiment_count": shared_count,
            "strategy_version_ids": sorted_version_ids,
            "experiment_family_counts": dict(sorted(family_counts.items())),
            "trial_count_audit": {
                **trial_lineage,
                "factor_trial_count": factor_trial_count,
                "model_trial_count": model_trial_count,
                "quant_trial_count": quant_trial_count,
                "ensemble_trial_count": ensemble_trial_count,
                "research_trial_count": research_trial_count,
                "model_runs": [
                    {
                        "research_run_id": run_id,
                        "candidate_count": count,
                        "trial_count": count,
                        "all_candidate_ids": sorted(
                            str(row.id)
                            for row in model_run_all_rows
                            if str(row.research_run_id) == run_id
                        ),
                        "bound_candidate_ids": sorted(
                            str(row.id) for row in model_rows if str(row.research_run_id) == run_id
                        ),
                    }
                    for run_id, count in sorted(model_run_counts.items())
                ],
                "quant_runs": [
                    {
                        "research_run_id": run_id,
                        "bundle_candidate_count": count,
                        "ablations_per_bundle": len(REQUIRED_QUANT_ABLATIONS),
                        "trial_count": count * len(REQUIRED_QUANT_ABLATIONS),
                        "all_bundle_candidate_ids": sorted(
                            str(row.id)
                            for row in quant_run_all_rows
                            if str(row.research_run_id) == run_id
                        ),
                        "bound_bundle_candidate_ids": sorted(
                            str(row.id) for row in bundle_rows if str(row.research_run_id) == run_id
                        ),
                        "excluded_model_candidates": sorted(
                            str(row.id) for row in model_rows if str(row.research_run_id) == run_id
                        ),
                    }
                    for run_id, count in sorted(quant_run_counts.items())
                ],
                "ensemble_tournaments": [
                    {
                        "tournament_id": tournament_id,
                        "candidate_count": sum(
                            str(row.tournament_id) == tournament_id
                            for row in ensemble_tournament_rows
                        ),
                        "trial_count": sum(
                            str(row.tournament_id) == tournament_id
                            for row in ensemble_tournament_rows
                        ),
                        "all_candidate_ids": sorted(
                            str(row.id)
                            for row in ensemble_tournament_rows
                            if str(row.tournament_id) == tournament_id
                        ),
                        "bound_candidate_ids": sorted(
                            str(row.id)
                            for row in ensemble_rows
                            if str(row.tournament_id) == tournament_id
                        ),
                    }
                    for tournament_id in sorted(ensemble_tournament_ids)
                ],
                "bound_versions": [
                    {
                        "strategy_version_id": str(row.id),
                        "model_candidate_id": str(
                            (row.config_json or {}).get("model_candidate_id") or ""
                        )
                        or None,
                        "quant_bundle_candidate_id": str(
                            (row.config_json or {}).get("quant_bundle_candidate_id") or ""
                        )
                        or None,
                        "model_ensemble_candidate_id": str(
                            (row.config_json or {}).get(
                                "model_ensemble_candidate_id"
                            )
                            or ""
                        )
                        or None,
                    }
                    for row in sorted(version_rows, key=lambda item: str(item.id))
                    if (row.config_json or {}).get("model_candidate_id")
                    or (row.config_json or {}).get("quant_bundle_candidate_id")
                    or (row.config_json or {}).get("model_ensemble_candidate_id")
                ],
            },
        }

    def _require_fin_strategy_formal_admission(
        self,
        connection: Any,
        version: Mapping[str, Any],
        *,
        allow_approved_paper: bool = False,
    ) -> dict[str, Any]:
        """Resolve the one passed policy/full-stack path for a draft version.

        Research-run artifacts remain explicitly non-capital. This binding is
        only permission to preregister and consume one CapitalOOSAlphaLedger
        window; it is not historical approval or recommendation authority.
        """

        config = dict(version.get("config") or version.get("config_json") or {})
        source_id = str(
            version.get("source_research_artifact_id")
            or config.get("source_research_artifact_id")
            or ""
        )
        if not source_id:
            raise ValueError("fin_strategy formal admission has no compiled source")
        parent_id = config.get("parent_strategy_version_id")
        compiled = _require_strategy_source_artifact(
            connection,
            config,
            expected_strategy_id=(
                str(version.get("strategy_id") or "") if parent_id is not None else None
            ),
        )
        if compiled is None:
            raise ValueError("fin_strategy formal admission has no compiled source")
        source_row = connection.execute(
            select(research_run_artifacts)
            .where(research_run_artifacts.c.id == source_id)
            .with_for_update()
        ).one()
        research_run_id = str(source_row.research_run_id)
        artifact_types = (
            "fin_strategy_competition_plan",
            FIN_STRATEGY_POLICY_ARTIFACT_TYPE,
            FIN_STRATEGY_FULL_STACK_ARTIFACT_TYPE,
            FIN_STRATEGY_WINNER_ARTIFACT_TYPE,
        )
        rows = connection.execute(
            select(research_run_artifacts)
            .where(
                research_run_artifacts.c.research_run_id == research_run_id,
                research_run_artifacts.c.artifact_type.in_(artifact_types),
            )
            .order_by(research_run_artifacts.c.created_at, research_run_artifacts.c.id)
            .with_for_update()
        ).all()
        parsed: dict[str, list[tuple[Any, dict[str, Any]]]] = {
            artifact_type: [] for artifact_type in artifact_types
        }
        for row in rows:
            artifact_type = str(row.artifact_type)
            parsed[artifact_type].append(
                (
                    row,
                    _verified_research_run_artifact_payload(
                        row,
                        expected_type=artifact_type,
                    ),
                )
            )

        version_value = dict(version)
        version_value["config"] = config
        version_value["status"] = str(version.get("status") or "")
        version_value["promotion_stage"] = version.get("promotion_stage")
        winners = parsed[FIN_STRATEGY_WINNER_ARTIFACT_TYPE]
        if len(winners) != 1:
            raise ValueError(
                "fin_strategy formal OOS requires one governed run winner decision"
            )
        winner_row, winner_payload = winners[0]
        matches: list[dict[str, Any]] = []
        for plan_row, plan in parsed["fin_strategy_competition_plan"]:
            if (
                plan.get("compiled_artifact_id") != source_id
                or plan.get("compiled_artifact_sha256")
                != config.get("strategy_research_artifact_sha256")
            ):
                continue
            plan_sha256 = str(plan.get("plan_sha256") or "")
            policies = [
                (row, payload)
                for row, payload in parsed[FIN_STRATEGY_POLICY_ARTIFACT_TYPE]
                if ((payload.get("evidence") or {}).get("plan_sha256") == plan_sha256)
            ]
            full_stacks = [
                (row, payload)
                for row, payload in parsed[FIN_STRATEGY_FULL_STACK_ARTIFACT_TYPE]
                if ((payload.get("evidence") or {}).get("plan_sha256") == plan_sha256)
            ]
            for policy_row, policy in policies:
                for full_row, full_stack in full_stacks:
                    try:
                        admission = build_fin_strategy_formal_admission(
                            strategy_version=version_value,
                            compiled_artifact=compiled,
                            competition_plan=plan,
                            policy_evaluation_artifact=policy,
                            full_stack_evaluation_artifact=full_stack,
                            governed_winner_artifact=winner_payload,
                            allow_approved_paper=allow_approved_paper,
                        )
                    except ValueError:
                        continue
                    binding = {
                        "contract_version": "fin-strategy-formal-admission-binding-v1",
                        "admission": admission,
                        "compiled_artifact": {
                            "id": source_id,
                            "content_sha256": str(source_row.content_sha256),
                            "manifest_sha256": str(source_row.manifest_sha256),
                        },
                        "competition_plan_artifact": {
                            "id": str(plan_row.id),
                            "content_sha256": str(plan_row.content_sha256),
                            "manifest_sha256": str(plan_row.manifest_sha256),
                        },
                        "policy_evaluation_artifact": {
                            "id": str(policy_row.id),
                            "content_sha256": str(policy_row.content_sha256),
                            "manifest_sha256": str(policy_row.manifest_sha256),
                        },
                        "full_stack_evaluation_artifact": {
                            "id": str(full_row.id),
                            "content_sha256": str(full_row.content_sha256),
                            "manifest_sha256": str(full_row.manifest_sha256),
                        },
                        "governed_winner_artifact": {
                            "id": str(winner_row.id),
                            "content_sha256": str(winner_row.content_sha256),
                            "manifest_sha256": str(winner_row.manifest_sha256),
                        },
                    }
                    binding["binding_sha256"] = _canonical_sha256(binding)
                    matches.append(binding)
        unique = {str(item["binding_sha256"]): item for item in matches}
        if len(unique) != 1:
            raise ValueError(
                "fin_strategy formal OOS requires exactly one sealed passed "
                "policy-only/full-stack admission path"
            )
        return next(iter(unique.values()))

    def require_fin_strategy_formal_admission(
        self, version_id: str
    ) -> dict[str, Any]:
        """Public read/lock boundary used before reserving the capital OOS."""

        version = self.get_version(version_id)
        with self.engine.begin() as connection:
            return self._require_fin_strategy_formal_admission(connection, version)

    def create_backtest(
        self,
        *,
        version_id: str,
        dataset: str,
        periods: dict[str, str],
        artifact_path: Path,
        execution_dataset: str | None = None,
        trading_dates: Sequence[date | str] | None = None,
        dataset_lineage_id: str | None = None,
        dataset_identity_sha256: str | None = None,
        capital_oos_alpha_batch_id: str | None = None,
        capital_oos_sealed_candidate_set_patch: Mapping[str, Any] | None = None,
        capital_oos_dataset_identity_sha256: str | None = None,
    ) -> dict[str, Any]:
        version = self.get_version(version_id)
        evidence_mode = str(version.get("evidence_mode") or EVIDENCE_MODE_LEGACY)
        if version.get("strategy_type") == "multifactor" and evidence_mode not in {
            EVIDENCE_MODE_SEALED,
            EVIDENCE_MODE_REPLAY,
        }:
            raise ValueError(
                "new multifactor backtests require sealed or consumed-replay evidence mode"
            )
        is_fin_strategy_candidate = bool(
            str(
                version.get("source_research_artifact_id")
                or version.get("config", {}).get("source_research_artifact_id")
                or ""
            ).strip()
        )
        capital_values_present = (
            capital_oos_alpha_batch_id is not None,
            capital_oos_sealed_candidate_set_patch is not None,
            capital_oos_dataset_identity_sha256 is not None,
        )
        if any(capital_values_present) and not all(capital_values_present):
            raise ValueError(
                "capital OOS backtest requires batch, sealed patch, and dataset identity"
            )
        if is_fin_strategy_candidate and not all(capital_values_present):
            raise ValueError(
                "fin_strategy formal OOS requires a preregistered "
                "CapitalOOSAlphaLedger batch"
            )
        if evidence_mode == EVIDENCE_MODE_REPLAY and any(capital_values_present):
            raise ValueError(
                "consumed historical replay cannot reserve or settle a capital OOS batch"
            )
        if evidence_mode == EVIDENCE_MODE_REPLAY and is_fin_strategy_candidate:
            raise ValueError(
                "forward-only rehabilitation is restricted to the transparent public baseline"
            )
        capital_batch_id: str | None = None
        capital_dataset_identity: str | None = None
        capital_sealed_patch: dict[str, Any] | None = None
        if all(capital_values_present):
            if version.get("strategy_type") != "multifactor":
                raise ValueError("capital OOS alpha binding is only valid for multifactor")
            capital_batch_id = str(capital_oos_alpha_batch_id).strip().lower()
            capital_dataset_identity = str(
                capital_oos_dataset_identity_sha256
            ).strip().lower()
            if not _is_sha256(capital_batch_id) or not _is_sha256(
                capital_dataset_identity
            ):
                raise ValueError("capital OOS batch and dataset identity must be SHA-256")
            if not isinstance(capital_oos_sealed_candidate_set_patch, Mapping):
                raise ValueError("capital OOS sealed candidate patch must be an object")
            try:
                capital_sealed_patch = json.loads(
                    json.dumps(
                        dict(capital_oos_sealed_candidate_set_patch),
                        ensure_ascii=False,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "capital OOS sealed candidate patch must be JSON serializable"
                ) from exc
        try:
            requested_start = date.fromisoformat(str(periods["start"]))
            requested_end = date.fromisoformat(str(periods["end"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("backtest final-test periods are invalid") from exc
        if requested_end < requested_start:
            raise ValueError("backtest final-test end must not be before its start")
        recorded_periods = {
            "start": requested_start.isoformat(),
            "end": requested_end.isoformat(),
        }
        backtest_id = uuid.uuid4().hex
        artifact_directory = (
            artifact_path / backtest_id if artifact_path.name == "backtests" else artifact_path
        )
        with self.engine.begin() as connection:
            # Serialise the one-shot formal-test boundary per frozen version.
            # Without this lock two completion workers could both observe no
            # prior row before either inserts, opening the sealed OOS twice.
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:identity))"),
                {"identity": f"strategy-final-backtest:{version_id}"},
            )
            if version.get("strategy_type") == "multifactor":
                _lock_strategy_trial_family(
                    connection,
                    str(version["economic_hypothesis_group"]),
                )
            fin_strategy_admission: dict[str, Any] | None = None
            if is_fin_strategy_candidate:
                fin_strategy_admission = self._require_fin_strategy_formal_admission(
                    connection,
                    version,
                )
                admission = validate_fin_strategy_formal_admission(
                    fin_strategy_admission["admission"]
                )
                expected_periods = dict(admission["formal_periods"])
                supplied_identity = str(dataset_identity_sha256 or "").strip().lower()
                if (
                    admission["dataset"] != dataset
                    or supplied_identity != admission["dataset_identity_sha256"]
                    or any(
                        str(periods.get(key) or "") != value
                        for key, value in expected_periods.items()
                    )
                ):
                    raise ValueError(
                        "fin_strategy formal OOS differs from its sealed "
                        "admission dataset or window"
                    )
                recorded_periods["fin_strategy_formal_admission"] = fin_strategy_admission
            if version.get("strategy_type") == "multifactor":
                model_evidence = self._model_signal_evidence(connection, version["config"])
                prior = connection.execute(
                    select(backtest_runs.c.id).where(
                        backtest_runs.c.strategy_version_id == version_id
                    )
                ).first()
                if prior is not None:
                    raise ValueError("a frozen strategy version may run the final test only once")
                baseline_only = (
                    version["config"].get("factor_source_mode") == FACTOR_SOURCE_QLIB_BASELINE
                    and not version["factors"]
                )
                factor_windows = connection.execute(
                    select(
                        factor_evaluations.c.id,
                        factor_evaluations.c.factor_candidate_id,
                        factor_evaluations.c.dataset,
                        factor_evaluations.c.dataset_identity_sha256,
                        factor_evaluations.c.train_start,
                        factor_evaluations.c.valid_end,
                        factor_evaluations.c.test_start,
                        factor_evaluations.c.test_end,
                        factor_evaluations.c.evaluator_version,
                        factor_evaluations.c.final_test_consumed_at,
                    )
                    .join(
                        strategy_factors,
                        strategy_factors.c.factor_evaluation_id == factor_evaluations.c.id,
                    )
                    .where(strategy_factors.c.strategy_version_id == version_id)
                ).all()
                if factor_windows and any(
                    item.dataset != dataset
                    or str(item.evaluator_version) != "factor-gate-v3-hac-bh"
                    or requested_start != item.test_start
                    or requested_end != item.test_end
                    for item in factor_windows
                ):
                    raise ValueError(
                        "formal backtest must exactly match the reserved final-test window"
                    )
                if not baseline_only and not factor_windows and model_evidence is None:
                    raise ValueError("formal backtest has no governed factor or model signal")
                model_candidate = (
                    model_evidence["candidate"] if model_evidence is not None else None
                )
                ensemble_components = (
                    [item["candidate"] for item in model_evidence["components"]]
                    if model_evidence is not None
                    and model_evidence.get("signal_kind") == "ensemble"
                    else []
                )
                governed_model_candidates = (
                    [model_candidate] if model_candidate is not None else ensemble_components
                )
                if any(
                    str(candidate.dataset) != dataset
                    or requested_start != candidate.final_oos_start
                    or requested_end != candidate.final_oos_end
                    for candidate in governed_model_candidates
                ):
                    raise ValueError(
                        "formal model backtest must exactly match its sealed dataset "
                        "and final-OOS window"
                    )
                if any(item.final_test_consumed_at is not None for item in factor_windows):
                    raise ValueError("reserved final test has already been consumed")
                history_starts = [item.train_start for item in factor_windows]
                history_ends = [item.valid_end for item in factor_windows]
                for governed_candidate in governed_model_candidates:
                    model_rows = connection.execute(
                        select(
                            model_evaluations.c.train_start,
                            model_evaluations.c.valid_end,
                        ).where(
                            model_evaluations.c.model_candidate_id
                            == governed_candidate.id,
                            model_evaluations.c.evidence_role == "independent_gate",
                            model_evaluations.c.gate_status == "passed",
                        )
                    ).all()
                    if not model_rows:
                        raise ValueError("model candidate has no independent evaluation grid")
                    history_starts.append(min(item.train_start for item in model_rows))
                    history_ends.append(governed_candidate.pre_final_end)
                if history_starts:
                    # Every selected factor must have observed the same
                    # pre-final history.  The intersection is authoritative:
                    # a factor with shorter lineage may not borrow another
                    # factor's earlier dates to manufacture ten-year evidence.
                    history_start = max(history_starts)
                    history_end = min(history_ends)
                    supplied_start = periods.get("historical_start")
                    supplied_end = periods.get("historical_end")
                    if (
                        supplied_start is not None
                        and date.fromisoformat(str(supplied_start)) != history_start
                    ) or (
                        supplied_end is not None
                        and date.fromisoformat(str(supplied_end)) != history_end
                    ):
                        raise ValueError(
                            "pre-final history must match the selected factors' "
                            "immutable evaluation window"
                        )
                else:
                    # A pure immutable Qlib baseline has no factor-evaluation
                    # rows from which to derive its development history.
                    # Callers therefore have to pin the dataset-backed window.
                    try:
                        history_start = date.fromisoformat(str(periods["historical_start"]))
                        history_end = date.fromisoformat(str(periods["historical_end"]))
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            "baseline final tests require explicit pre-final history periods"
                        ) from exc
                if history_end < history_start:
                    raise ValueError("pre-final history end must not be before its start")
                if history_end >= requested_start:
                    raise ValueError("pre-final history must end before the final test starts")
                minimum_embargo_days = int(version["config"].get("outer_embargo_days") or 5)
                minimum_history_days = int(
                    version["config"].get("min_pre_final_history_days") or 2520
                )
                if trading_dates is not None:
                    try:
                        supplied_calendar = [
                            value
                            if isinstance(value, date)
                            else date.fromisoformat(str(value))
                            for value in trading_dates
                        ]
                        calendar = sorted(set(supplied_calendar))
                    except (TypeError, ValueError) as exc:
                        raise ValueError("Qlib trading calendar contains invalid dates") from exc
                    if capital_batch_id is not None and supplied_calendar != calendar:
                        raise ValueError(
                            "capital OOS requires a strictly increasing unique Qlib calendar"
                        )
                    history_trading_days = sum(
                        history_start <= value <= history_end for value in calendar
                    )
                    embargo_trading_days = sum(
                        history_end < value < requested_start for value in calendar
                    )
                    final_trading_days = sum(
                        requested_start <= value <= requested_end for value in calendar
                    )
                    minimum_final_days = max(
                        252,
                        int(version["config"].get("min_backtest_days") or 252),
                    )
                    if requested_start not in calendar or requested_end not in calendar:
                        raise ValueError(
                            "final OOS boundaries must be exact Qlib trading dates"
                        )
                    if history_trading_days < minimum_history_days:
                        raise ValueError(
                            "pre-final history has "
                            f"{history_trading_days} trading days; "
                            f"{minimum_history_days} are required"
                        )
                    if embargo_trading_days < minimum_embargo_days:
                        raise ValueError(
                            "pre-final history leaves "
                            f"{embargo_trading_days} trading days for the configured "
                            f"{minimum_embargo_days}-trading-day embargo"
                        )
                    if final_trading_days < minimum_final_days:
                        raise ValueError(
                            "final OOS has "
                            f"{final_trading_days} trading days; "
                            f"{minimum_final_days} are required"
                        )
                else:
                    # Compatibility callers without a bound dataset calendar
                    # still receive a conservative coarse guard. Production
                    # API/orchestrator callers always supply the exact Qlib
                    # calendar before the once-only final sample is consumed.
                    available_calendar_gap = (requested_start - history_end).days - 1
                    if available_calendar_gap < minimum_embargo_days:
                        raise ValueError(
                            "pre-final history leaves too little calendar space for "
                            f"the configured {minimum_embargo_days}-trading-day embargo"
                        )
                    minimum_calendar_days = int(minimum_history_days * 7 / 5)
                    if (history_end - history_start).days + 1 < minimum_calendar_days:
                        raise ValueError(
                            "pre-final history is too short for the configured "
                            f"{minimum_history_days}-trading-day gate"
                        )
                recorded_periods.update(
                    {
                        "historical_start": history_start.isoformat(),
                        "historical_end": history_end.isoformat(),
                    }
                )
                consumed_at = _now()
                candidate_ids = sorted({str(item.factor_candidate_id) for item in factor_windows})
                model_identity = model_evidence["identity"] if model_evidence is not None else None
                if factor_windows:
                    sealed_member_set: dict[str, Any] = {
                        "candidate_ids": candidate_ids,
                        "model_signal": model_identity,
                    }
                    dataset_identities = {
                        str(item.dataset_identity_sha256 or "") for item in factor_windows
                    }
                elif model_evidence is not None:
                    strategy_spec = {
                        "strategy_type": "multifactor",
                        "signal_source": "model_prediction",
                        "benchmark": version["benchmark"],
                        "universe": version["universe"],
                        "config_sha256": _canonical_sha256(version["config"]),
                        "model_signal_identity_sha256": model_identity["identity_sha256"],
                    }
                    sealed_member_set = {
                        "candidate_ids": [],
                        "strategy_spec_sha256": _canonical_sha256(strategy_spec),
                        "model_signal": model_identity,
                    }
                    dataset_identities = set()
                else:
                    sealed_member_set = (
                        {}
                        if evidence_mode == EVIDENCE_MODE_REPLAY
                        else baseline_oos_sealed_member_set(version)
                    )
                    # Baseline-only versions have no factor evaluation carrying
                    # a snapshot identity. This value is audit-only; stable scope
                    # below, never the snapshot name, controls OOS reuse.
                    dataset_identities = set()
                for governed_candidate in governed_model_candidates:
                    dataset_identities.add(
                        str(governed_candidate.dataset_identity_sha256 or "")
                    )
                if capital_batch_id is not None:
                    assert capital_dataset_identity is not None
                    assert capital_sealed_patch is not None
                    if trading_dates is None:
                        raise ValueError(
                            "capital OOS requires the exact Qlib trading calendar"
                        )
                    normalized_lineage = str(dataset_lineage_id or "").strip().lower()
                    if not _is_sha256(normalized_lineage):
                        raise ValueError(
                            "capital OOS requires a valid dataset lineage SHA-256"
                        )
                    observed_identities = {
                        str(value).strip().lower()
                        for value in dataset_identities
                        if str(value).strip()
                    }
                    if observed_identities and observed_identities != {
                        capital_dataset_identity
                    }:
                        raise ValueError(
                            "capital OOS dataset identity differs from governed signal evidence"
                        )
                    final_calendar = [
                        item.isoformat()
                        for item in calendar
                        if requested_start <= item <= requested_end
                    ]
                    embargo_calendar = [
                        item.isoformat()
                        for item in calendar
                        if history_end < item < requested_start
                    ]
                    self._validate_capital_oos_batch_binding(
                        connection,
                        batch_id=capital_batch_id,
                        sealed_candidate_set_patch=capital_sealed_patch,
                        dataset_lineage_id=normalized_lineage,
                        dataset_identity_sha256=capital_dataset_identity,
                        research_data_end=history_end,
                        test_start=requested_start,
                        test_end=requested_end,
                        final_oos_trading_dates=final_calendar,
                        embargo_trading_dates=embargo_calendar,
                    )
                    if set(capital_sealed_patch) & set(sealed_member_set):
                        raise ValueError(
                            "capital OOS sealed candidate patch collides with strategy evidence"
                        )
                    sealed_member_set.update(capital_sealed_patch)
                if evidence_mode == EVIDENCE_MODE_REPLAY:
                    if factor_windows or model_evidence is not None or candidate_ids:
                        raise ValueError(
                            "forward-only rehabilitation accepts only the public baseline"
                        )
                    binding = require_replay_config(version["config"])
                    if (
                        dataset != binding["dataset"]
                        or dict(recorded_periods) != dict(binding["replay_periods"])
                    ):
                        raise ValueError(
                            "historical replay differs from its exact consumed source window"
                        )
                    require_consumed_vintage(
                        connection,
                        version_config=version["config"],
                        dataset_identity_sha256=str(dataset_identity_sha256 or ""),
                        dataset_lineage_id=str(dataset_lineage_id or ""),
                    )
                    hypothesis_evidence = self.hypothesis_group_evidence(
                        version_id,
                        connection=connection,
                    )
                    artifacts_root = next(
                        (
                            parent
                            for parent in (artifact_directory, *artifact_directory.parents)
                            if parent.name == "artifacts"
                        ),
                        None,
                    )
                    if artifacts_root is None:
                        raise ValueError(
                            "historical replay output is outside the governed artifacts root"
                        )
                    eligibility = build_incomplete_family_eligibility(
                        connection,
                        strategy_version_id=version_id,
                        hypothesis_group_evidence=hypothesis_evidence,
                        missing_artifacts=audit_incomplete_family_artifacts(
                            data_root=artifacts_root.parent,
                            observed_at=consumed_at
                        ),
                        cutoff_at=consumed_at,
                        created_by="system:forward-only-rehabilitation",
                    )
                    insert_incomplete_family_eligibility(
                        connection,
                        eligibility,
                        created_at=consumed_at,
                    )
                else:
                    # Every real multifactor final test, including a pure Qlib
                    # baseline, consumes the same governed OOS ledger. A replay
                    # above can only point at an already-consumed exact row.
                    self._seal_and_consume_oos_vintage(
                        connection,
                        strategy_version_id=version_id,
                        candidate_ids=candidate_ids,
                        sealed_member_set=sealed_member_set,
                        dataset_identities=dataset_identities,
                        dataset_lineage_id=dataset_lineage_id,
                        dataset=dataset,
                        test_start=requested_start,
                        test_end=requested_end,
                        consumed_at=consumed_at,
                        capital_oos_alpha_batch_id=capital_batch_id,
                        capital_oos_dataset_identity_sha256=capital_dataset_identity,
                    )
                    for item in factor_windows:
                        key = hashlib.sha256(
                            (
                                f"{version_id}:{item.id}:{dataset}:"
                                f"{recorded_periods['start']}:{recorded_periods['end']}"
                            ).encode()
                        ).hexdigest()
                        connection.execute(
                            update(factor_evaluations)
                            .where(
                                factor_evaluations.c.id == item.id,
                                factor_evaluations.c.final_test_consumed_at.is_(None),
                            )
                            .values(final_test_key=key, final_test_consumed_at=consumed_at)
                        )
            connection.execute(
                insert(backtest_runs).values(
                    id=backtest_id,
                    strategy_version_id=version_id,
                    dataset=dataset,
                    execution_dataset=execution_dataset,
                    signal_frequency=version["signal_frequency"],
                    execution_frequency=version["execution_frequency"],
                    execution_contract_hash=version["execution_contract_hash"],
                    qlib_version=version["qlib_version"],
                    qlib_commit=version["qlib_commit"],
                    rdagent_version=version["rdagent_version"],
                    rdagent_commit=version["rdagent_commit"],
                    status="queued",
                    evidence_mode=evidence_mode,
                    periods_json=recorded_periods,
                    artifact_path=str(artifact_directory),
                    created_at=_now(),
                )
            )
        return self.get_backtest(backtest_id)

    @staticmethod
    def _validate_capital_oos_batch_binding(
        connection: Any,
        *,
        batch_id: str,
        sealed_candidate_set_patch: Mapping[str, Any],
        dataset_lineage_id: str,
        dataset_identity_sha256: str,
        research_data_end: date,
        test_start: date,
        test_end: date,
        final_oos_trading_dates: Sequence[str],
        embargo_trading_dates: Sequence[str],
    ) -> None:
        """Lock and validate preregistration before any final-OOS row is written."""

        batch = connection.execute(
            select(capital_oos_alpha_batches)
            .where(capital_oos_alpha_batches.c.id == batch_id)
            .with_for_update()
        ).first()
        if batch is None:
            raise ValueError("capital OOS alpha batch does not exist")
        if str(batch.status) != "reserved":
            raise ValueError("capital OOS alpha batch is not reserved")
        expected_patch = {
            "capital_oos_alpha_ledger": capital_oos_vintage_link_payload(
                batch_id=str(batch.id),
                preregistration_sha256=str(batch.preregistration_sha256),
                frozen_bundle_manifest_sha256=str(
                    batch.frozen_bundle_manifest_sha256
                ),
                frozen_baseline_manifest_sha256=str(
                    batch.frozen_baseline_manifest_sha256
                ),
            )
        }
        if dict(sealed_candidate_set_patch) != expected_patch:
            raise ValueError(
                "capital OOS sealed candidate patch differs from preregistration"
            )
        if (
            str(batch.dataset_lineage_id) != dataset_lineage_id
            or str(batch.dataset_identity_sha256) != dataset_identity_sha256
            or batch.research_data_end != research_data_end
            or batch.final_oos_start != test_start
            or batch.final_oos_end != test_end
            or list(batch.trading_dates_json or [])
            != list(final_oos_trading_dates)
            or list(batch.embargo_trading_dates_json or [])
            != list(embargo_trading_dates)
        ):
            raise ValueError(
                "formal backtest data or window differs from capital OOS preregistration"
            )
        linked = connection.execute(
            select(oos_vintages.c.id).where(
                oos_vintages.c.capital_oos_alpha_batch_id == batch_id
            )
        ).first()
        if linked is not None:
            raise ValueError("capital OOS alpha batch already has a linked vintage")

    @staticmethod
    def _seal_and_consume_oos_vintage(
        connection: Any,
        *,
        strategy_version_id: str,
        candidate_ids: list[str],
        sealed_member_set: dict[str, Any],
        dataset_identities: set[str],
        dataset_lineage_id: str | None,
        dataset: str,
        test_start: date,
        test_end: date,
        consumed_at: datetime,
        capital_oos_alpha_batch_id: str | None = None,
        capital_oos_dataset_identity_sha256: str | None = None,
    ) -> str:
        """Seal and consume one final-test window in a stable research scope.

        Dataset identities change whenever an immutable snapshot advances, so
        they are audit evidence rather than scope. Program research uses its
        program id; standalone research uses a verified dataset lineage, and
        missing lineage fails closed into one global standalone scope.
        """

        dataset_identity = str(capital_oos_dataset_identity_sha256 or "").strip()
        if dataset_identity and not _is_sha256(dataset_identity):
            raise ValueError("capital OOS dataset identity must be a SHA-256")
        if not dataset_identity:
            dataset_identity = (
                next(iter(dataset_identities))
                if len(dataset_identities) == 1 and dataset_identities != {""}
                else f"name:{dataset}"
            )
        candidate_program_ids = {
            str(row.research_program_id)
            for row in connection.execute(
                select(research_campaigns.c.research_program_id).where(
                    research_campaigns.c.research_run_id.in_(
                        select(factor_candidates.c.research_run_id).where(
                            factor_candidates.c.id.in_(candidate_ids)
                        )
                    ),
                    research_campaigns.c.research_program_id.is_not(None),
                )
            )
        }
        version_program_ids = {
            str(row.research_program_id)
            for row in connection.execute(
                select(research_campaigns.c.research_program_id).where(
                    research_campaigns.c.strategy_version_id == strategy_version_id,
                    research_campaigns.c.research_program_id.is_not(None),
                )
            )
        }
        program_ids = candidate_program_ids | version_program_ids
        if len(program_ids) > 1:
            raise ValueError("final-test members span multiple research programs")

        normalized_lineage = str(dataset_lineage_id or "").strip().lower()
        if normalized_lineage and not _is_sha256(normalized_lineage):
            raise ValueError("final test requires a valid dataset lineage SHA-256")
        capital_family: dict[str, str] | None = None
        if capital_oos_alpha_batch_id is not None:
            family = connection.execute(
                select(
                    capital_oos_alpha_batches.c.family_id,
                    capital_oos_alpha_families.c.capital_oos_family_sha256,
                    capital_oos_alpha_families.c.mandate_json,
                )
                .join(
                    capital_oos_alpha_families,
                    capital_oos_alpha_families.c.id
                    == capital_oos_alpha_batches.c.family_id,
                )
                .where(
                    capital_oos_alpha_batches.c.id == capital_oos_alpha_batch_id
                )
            ).first()
            mandate = dict(family.mandate_json or {}) if family is not None else {}
            family_sha256 = (
                str(family.capital_oos_family_sha256 or "").strip().lower()
                if family is not None
                else ""
            )
            label_horizon_days = mandate.get("label_horizon_days")
            if (
                family is None
                or not _is_sha256(family_sha256)
                or isinstance(label_horizon_days, bool)
                or not isinstance(label_horizon_days, int)
                or label_horizon_days < 1
            ):
                raise ValueError(
                    "capital OOS alpha batch has no stable family/horizon scope"
                )
            capital_family = {
                "family_id": str(family.family_id),
                "family_sha256": family_sha256,
                "label_horizon_days": str(label_horizon_days),
            }
            scope = (
                f"alpha-family:{family_sha256}:"
                f"label:{label_horizon_days}"
            )
            stored_lineage = normalized_lineage or None
            include_legacy_dataset_scopes = False
        elif program_ids:
            program_id = next(iter(program_ids))
            scope = f"program:{program_id}"
            program_lineage = connection.scalar(
                select(research_programs.c.dataset_lineage_id).where(
                    research_programs.c.id == program_id
                )
            )
            stored_lineage = normalized_lineage or str(program_lineage or "").strip() or None
            include_legacy_dataset_scopes = False
        elif normalized_lineage:
            scope = f"lineage:{normalized_lineage}"
            stored_lineage = normalized_lineage
            include_legacy_dataset_scopes = True
        else:
            # Unknown lineage is not permission to start a new scope. All such
            # standalone research shares one conservative, fail-closed ledger.
            scope = "standalone:global"
            stored_lineage = None
            include_legacy_dataset_scopes = True

        lockbox_raw = sealed_member_set.get("transparent_baseline_lockbox")
        lockbox = validate_lockbox_link(lockbox_raw) if lockbox_raw is not None else None

        # Serialize against the reservation writer before resolving a repair
        # batch.  Optimizer-applicability repairs deliberately keep the same
        # dataset lineage while receiving append-only rows in a batch-specific
        # scope.  The sealed batch itself, rather than the caller, is the only
        # authority allowed to select that scope.
        connection.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:oos_scope))"),
            {"oos_scope": scope},
        )
        if lockbox is not None:
            if program_ids or not normalized_lineage:
                raise ValueError(
                    "transparent baseline joint lockbox requires standalone lineage scope"
                )
            batch_rows = []
            for candidate in connection.execute(
                select(oos_vintages)
                .where(oos_vintages.c.dataset_lineage_id == normalized_lineage)
                .with_for_update()
            ).all():
                candidate_members = dict(candidate.sealed_candidate_set_json or {})
                candidate_link = candidate_members.get("transparent_baseline_lockbox")
                if not isinstance(candidate_link, Mapping):
                    continue
                try:
                    normalized_link = validate_lockbox_link(candidate_link)
                except ValueError:
                    continue
                if normalized_link["batch_sha256"] == lockbox["batch_sha256"]:
                    batch_rows.append((candidate, normalized_link))
            batch_scopes = {str(item[0].scope) for item in batch_rows}
            observed_members = {item[1]["member_sha256"] for item in batch_rows}
            expected_lockbox_members = set(lockbox["member_sha256s"])
            base_scope = f"lineage:{normalized_lineage}"
            permitted_scopes = {
                base_scope,
                f"{base_scope}:repair:{lockbox['batch_sha256']}",
            }
            if (
                len(batch_rows) != len(expected_lockbox_members)
                or observed_members != expected_lockbox_members
                or len(batch_scopes) != 1
                or not batch_scopes <= permitted_scopes
            ):
                raise ValueError(
                    "transparent baseline final OOS was not atomically preregistered "
                    "for every statistically available horizon"
                )
            reserved_scope = next(iter(batch_scopes))
            if reserved_scope != base_scope:
                repair_rows = connection.execute(
                    select(transparent_baseline_pre_result_repairs)
                    .where(
                        transparent_baseline_pre_result_repairs.c.target_batch_sha256
                        == lockbox["batch_sha256"]
                    )
                    .with_for_update()
                ).all()
                repair = repair_rows[0] if len(repair_rows) == 1 else None
                batch_version_ids = {
                    str(
                        dict(item[0].sealed_candidate_set_json or {}).get(
                            "strategy_version_id"
                        )
                        or ""
                    )
                    for item in batch_rows
                }
                batch_dataset_identities = {
                    str(item[0].dataset_identity or "") for item in batch_rows
                }
                validate_repair_registry_binding(
                    repair,
                    lockbox_batch_sha256=str(lockbox["batch_sha256"]),
                    strategy_version_id=strategy_version_id,
                    batch_strategy_version_ids=batch_version_ids,
                    batch_dataset_identity_sha256s=batch_dataset_identities,
                    dataset=dataset,
                    dataset_lineage_id=normalized_lineage,
                )
            if reserved_scope != scope:
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:oos_scope))"),
                    {"oos_scope": reserved_scope},
                )
            scope = reserved_scope
            # The lockbox reservation already performed the global overlap
            # check. Consumption is allowed to touch only the exact immutable
            # rows declared by this lockbox, never legacy or source-repair rows.
            include_legacy_dataset_scopes = False

        scope_filter = oos_vintages.c.scope == scope
        if capital_family is not None:
            # Before this scope contract, capital attempts were written under
            # dataset lineage.  They remain binding history for the same
            # governed alpha family and may not be reopened after migration.
            prior_family_vintages = (
                select(oos_vintages.c.id)
                .join(
                    capital_oos_alpha_batches,
                    capital_oos_alpha_batches.c.id
                    == oos_vintages.c.capital_oos_alpha_batch_id,
                )
                .where(
                    capital_oos_alpha_batches.c.family_id
                    == capital_family["family_id"]
                )
            )
            scope_filter = or_(
                scope_filter,
                oos_vintages.c.id.in_(prior_family_vintages),
            )
        if include_legacy_dataset_scopes:
            # Rows written before scope-v2 used dataset identity as scope. Their
            # lineage cannot be reconstructed safely, so they conservatively
            # block overlapping standalone tests after migration.
            scope_filter = or_(scope_filter, oos_vintages.c.scope.like("dataset:%"))
        overlapping_rows = connection.execute(
            select(oos_vintages)
            .where(
                scope_filter,
                oos_vintages.c.test_start <= test_end,
                oos_vintages.c.test_end >= test_start,
            )
            .with_for_update()
        ).all()
        row = next(
            (
                item
                for item in overlapping_rows
                if item.test_start == test_start and item.test_end == test_end
            ),
            None,
        )
        if lockbox is not None:
            # Available public control windows are allowed to overlap only
            # after every exact declared member has been atomically preregistered.
            # No call through create_backtest may create a missing member row.
            batch_rows = []
            for candidate in connection.execute(
                select(oos_vintages).where(oos_vintages.c.scope == scope).with_for_update()
            ).all():
                candidate_members = dict(candidate.sealed_candidate_set_json or {})
                candidate_link = candidate_members.get(
                    "transparent_baseline_lockbox"
                )
                if not isinstance(candidate_link, Mapping):
                    continue
                try:
                    normalized_link = validate_lockbox_link(candidate_link)
                except ValueError:
                    continue
                if normalized_link["batch_sha256"] == lockbox["batch_sha256"]:
                    batch_rows.append((candidate, normalized_link))
            observed_members = {
                item[1]["member_sha256"] for item in batch_rows
            }
            expected_lockbox_members = set(lockbox["member_sha256s"])
            if (
                row is None
                or len(batch_rows) != len(expected_lockbox_members)
                or observed_members != expected_lockbox_members
                or any(
                    validate_lockbox_link(
                        dict(item.sealed_candidate_set_json or {}).get(
                            "transparent_baseline_lockbox"
                        )
                    )["batch_sha256"]
                    != lockbox["batch_sha256"]
                    for item in overlapping_rows
                )
            ):
                raise ValueError(
                    "transparent baseline final OOS was not atomically preregistered "
                    "for every statistically available horizon"
                )
        elif any(item is not row for item in overlapping_rows):
            raise ValueError(
                "final test window overlaps a reserved or consumed OOS vintage "
                "in the same research scope"
            )
        if row is not None:
            if str(row.capital_oos_alpha_batch_id or "") != str(
                capital_oos_alpha_batch_id or ""
            ):
                raise ValueError(
                    "final test window is linked to a different capital OOS batch"
                )
            recorded_members = dict(row.sealed_candidate_set_json or {})
            recorded_candidates = set(recorded_members.get("candidate_ids") or [])
            if candidate_ids and not set(candidate_ids) <= recorded_candidates:
                raise ValueError(
                    "final test window is sealed and this candidate is not in the "
                    "sealed candidate set"
                )
            if recorded_members.get("model_signal") != sealed_member_set.get("model_signal"):
                raise ValueError("final test window is sealed to a different model or joint bundle")
            if not candidate_ids and recorded_members != sealed_member_set:
                raise ValueError("final test window is sealed to a different baseline strategy")
            if row.consumed_at is not None:
                raise ValueError("reserved final test has already been consumed")
            connection.execute(
                update(oos_vintages)
                .where(oos_vintages.c.id == row.id)
                .values(consumed_at=consumed_at)
            )
            return str(row.id)
        vintage_id = uuid.uuid4().hex
        connection.execute(
            insert(oos_vintages).values(
                id=vintage_id,
                scope=scope,
                dataset_identity=dataset_identity,
                dataset_lineage_id=stored_lineage,
                test_start=test_start,
                test_end=test_end,
                sealed_at=consumed_at,
                first_opened_at=consumed_at,
                consumed_at=consumed_at,
                capital_oos_alpha_batch_id=capital_oos_alpha_batch_id,
                sealed_candidate_set_json=sealed_member_set,
                sealed_candidate_set_sha256=_canonical_sha256(sealed_member_set),
                created_at=consumed_at,
            )
        )
        return vintage_id

    def attach_job(self, backtest_id: str, job_id: str) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(backtest_runs).where(backtest_runs.c.id == backtest_id).values(job_id=job_id)
            )
            if not result.rowcount:
                raise KeyError(backtest_id)

    def attach_job_once(self, backtest_id: str, job_id: str) -> None:
        """Attach one immutable job without overwriting a concurrent binding.

        Ordinary pair/research flows retain :meth:`attach_job` because their
        explicit requeue path can replace a prior job.  Consumed-history replay
        is one-shot and uses this compare-and-set boundary instead.
        """

        with self.engine.begin() as connection:
            result = connection.execute(
                update(backtest_runs)
                .where(
                    backtest_runs.c.id == backtest_id,
                    or_(
                        backtest_runs.c.job_id.is_(None),
                        backtest_runs.c.job_id == job_id,
                    ),
                )
                .values(job_id=job_id)
            )
            if result.rowcount:
                return
            existing = connection.execute(
                select(backtest_runs.c.job_id).where(backtest_runs.c.id == backtest_id)
            ).first()
            if existing is None:
                raise KeyError(backtest_id)
            if str(existing.job_id or "") == job_id:
                return
            raise ValueError("formal backtest is already attached to a different job")

    def mark_backtest(
        self,
        backtest_id: str,
        status: str,
        *,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        now = _now()
        values: dict[str, Any] = {"status": status, "error": error}
        if metrics is not None:
            values["metrics_json"] = metrics
        if status == "running":
            values["started_at"] = now
        if status in {"succeeded", "failed", "cancelled"}:
            values["finished_at"] = now
        with self.engine.begin() as connection:
            result = connection.execute(
                update(backtest_runs).where(backtest_runs.c.id == backtest_id).values(**values)
            )
            if not result.rowcount:
                raise KeyError(backtest_id)

    def register_terminal_cash_only(
        self,
        *,
        data_root: Path,
        actor: str,
    ) -> dict[str, Any]:
        """Seal the exact failed public control as cash/NO_ACTION only."""

        with self.engine.begin() as connection:
            return register_terminal_cash_only_receipt(
                connection,
                data_root=data_root,
                actor=actor,
            )

    def requeue_backtest(self, backtest_id: str) -> None:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(backtest_runs.c.status, strategy_versions.c.strategy_type)
                .join(
                    strategy_versions,
                    strategy_versions.c.id == backtest_runs.c.strategy_version_id,
                )
                .where(backtest_runs.c.id == backtest_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(backtest_id)
            if row.status not in {"failed", "cancelled"}:
                raise ValueError("only failed or cancelled backtests may be requeued")
            if row.strategy_type == "multifactor":
                raise ValueError("a formal final test cannot be rerun")
            connection.execute(
                update(backtest_runs)
                .where(backtest_runs.c.id == backtest_id)
                .values(
                    status="queued",
                    metrics_json=None,
                    error=None,
                    started_at=None,
                    finished_at=None,
                )
            )

    def validate_backtest_artifacts(self, backtest_id: str, metrics: dict[str, Any]) -> None:
        """Validate immutable strategy artifacts before a worker reports success."""

        backtest = self.get_backtest(backtest_id)
        version = self.get_version(backtest["strategy_version_id"])
        failures = (
            _pair_artifact_failures(version, backtest, metrics)
            if version.get("strategy_type") == "pair"
            else _multifactor_manifest_failures(version, backtest, metrics)
        )
        if version.get("strategy_type") == "multifactor":
            failures.extend(
                self._hypothesis_group_manifest_failures(version["id"], backtest)
            )
        if failures:
            raise ValueError("strategy backtest artifact validation failed: " + "; ".join(failures))

    def _hypothesis_group_manifest_failures(
        self,
        version_id: str,
        backtest: dict[str, Any],
        *,
        connection: Any | None = None,
    ) -> list[str]:
        manifest_path = Path(str(backtest["artifact_path"])) / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return ["hypothesis-group evidence manifest is unreadable"]
        current = self.hypothesis_group_evidence(
            version_id,
            connection=connection,
        )
        observed = manifest.get("hypothesis_group_evidence")
        if (
            not isinstance(observed, dict)
            or _canonical_sha256(observed) != _canonical_sha256(current)
            or int(manifest.get("strategy_trial_count") or 0)
            != int(current["shared_experiment_count"])
        ):
            return [
                "formal statistics do not bind the current complete hypothesis-group family"
            ]
        return []

    def get_backtest(self, backtest_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(backtest_runs).where(backtest_runs.c.id == backtest_id)
            ).first()
        if row is None:
            raise KeyError(backtest_id)
        result = row_dict(row)
        result["periods"] = result.pop("periods_json")
        result["metrics"] = result.pop("metrics_json")
        return result

    def list_backtests(
        self, version_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        statement = select(backtest_runs)
        if version_id:
            statement = statement.where(backtest_runs.c.strategy_version_id == version_id)
        statement = statement.order_by(backtest_runs.c.created_at.desc()).limit(limit)
        with self.engine.connect() as connection:
            rows = [row_dict(row) for row in connection.execute(statement)]
        for row in rows:
            row["periods"] = row.pop("periods_json")
            row["metrics"] = row.pop("metrics_json")
        return rows

    def _approve_pair(
        self,
        version: dict[str, Any],
        backtest: dict[str, Any],
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        # This is a research approval, not capital approval.  A passing pair may
        # own a persistent shadow ledger, while its catalog capabilities keep
        # recommendations, financing and real trading permanently disabled.
        config = PairTradingConfig(**version["config"])
        metrics = dict(backtest["metrics"] or {})
        failures: list[str] = []
        if actor == version["created_by"]:
            failures.append("pair strategy approval requires a second operator")
        if not backtest.get("execution_dataset"):
            failures.append("pair backtest requires an immutable minute execution dataset")
        if (
            metrics.get("backtest_engine") != "quantlab_pair"
            or metrics.get("pair_native_backtest") is not True
        ):
            failures.append("a native QuantLab pair backtest is required")
        pair = version.get("pair") or {}
        if metrics.get("leg_y") != pair.get("leg_y") or metrics.get("leg_x") != pair.get("leg_x"):
            failures.append("pair backtest instruments do not match the strategy version")
        evidence = metrics.get("initial_pair_evidence")
        if not isinstance(evidence, dict):
            failures.append("initial pair correlation and cointegration evidence is required")
            evidence = {}
        checks: dict[str, tuple[Any, Any, str]] = {
            "correlation": (evidence.get("correlation"), config.min_correlation, "min"),
            "cointegration_pvalue": (
                evidence.get("cointegration_pvalue"),
                config.max_cointegration_pvalue,
                "max",
            ),
            "hedge_ratio_min": (
                evidence.get("hedge_ratio"),
                config.min_hedge_ratio,
                "min",
            ),
            "hedge_ratio_max": (
                evidence.get("hedge_ratio"),
                config.max_hedge_ratio,
                "max",
            ),
            "max_drawdown": (
                abs(float(metrics["max_drawdown"]))
                if metrics.get("max_drawdown") is not None
                else None,
                config.max_drawdown,
                "max",
            ),
            "sharpe_ratio": (metrics.get("sharpe_ratio"), config.min_sharpe_ratio, "min"),
            "closed_trade_count": (
                metrics.get("closed_trade_count"),
                config.min_closed_trades,
                "min",
            ),
            "trading_days": (metrics.get("trading_days"), config.min_backtest_days, "min"),
            "rolling_cointegration_pass_rate": (
                metrics.get("rolling_cointegration_pass_rate"),
                config.min_rolling_cointegration_pass_rate,
                "min",
            ),
            "pair_robustness_pass_rate": (
                metrics.get("pair_robustness_pass_rate"),
                config.min_robustness_pass_rate,
                "min",
            ),
            "capacity_fill_ratio": (
                metrics.get("capacity_fill_ratio"),
                config.min_capacity_fill_ratio,
                "min",
            ),
        }
        for name, (value, threshold, mode) in checks.items():
            if (
                value is None
                or (mode == "max" and value > threshold)
                or (mode == "min" and value < threshold)
            ):
                failures.append(f"{name}={value} violates {mode} {threshold}")
        for name in (
            "minute_execution_enforced",
            "shortability_enforced",
            "market_controls_enforced",
            "atomic_pair_execution_enforced",
            "transaction_costs_enforced",
            "borrow_cost_enforced",
        ):
            if metrics.get(name) is not True:
                failures.append(f"{name} is required for pair strategy approval")
        if metrics.get("open_position_at_end") is not False:
            failures.append("pair backtest must finish without an open spread position")
        if metrics.get("cost_schedule_version") not in KNOWN_COST_SCHEDULE_VERSIONS:
            failures.append("pair backtest cost schedule is missing or obsolete")
        provenance = metrics.get("provenance")
        if not isinstance(provenance, dict):
            failures.append("reproducible pair backtest provenance is required")
        else:
            try:
                require_qlib_workflow_identity(provenance.get("qlib_workflow"))
            except ValueError as exc:
                failures.append(str(exc))
            for field in (
                "daily_dataset_identity_sha256",
                "daily_snapshot_manifest_sha256",
                "minute_snapshot_manifest_sha256",
                "strategy_config_sha256",
                "execution_manifest_sha256",
                "pair_engine_sha256",
                "shortability_evidence_sha256",
            ):
                if not _is_sha256(provenance.get(field)):
                    failures.append(f"provenance {field} must be a SHA-256 digest")
        if failures:
            raise ValueError("pair strategy risk gate failed: " + "; ".join(failures))
        now = _now()
        with self.engine.begin() as connection:
            # Pair execution is retained only as an offline/read-only legacy
            # lane and keeps its original single-approved-version semantics.
            connection.execute(
                update(strategy_versions)
                .where(
                    strategy_versions.c.strategy_id == version["strategy_id"],
                    strategy_versions.c.status == "approved",
                )
                .values(status="retired")
            )
            connection.execute(
                update(strategy_versions)
                .where(strategy_versions.c.id == version["id"])
                .values(
                    status="approved",
                    approved_by=actor,
                    approval_reason=reason,
                    approved_at=now,
                )
            )
            connection.execute(
                update(strategies)
                .where(strategies.c.id == version["strategy_id"])
                .values(status="approved", updated_at=now)
            )
            self._event(
                connection,
                strategy_id=version["strategy_id"],
                version_id=version["id"],
                event_type="strategy.pair_shadow_approved",
                actor=actor,
                payload={
                    "reason": reason,
                    "backtest_id": backtest["id"],
                    "simulation_mode": "shadow_pair",
                    "financing_enabled": False,
                    "real_trading_eligible": False,
                    "gate_evidence": {name: value[0] for name, value in checks.items()},
                },
            )
        return self.get_version(version["id"])

    def approve(self, version_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        version = self.get_version(version_id)
        if str(version.get("evidence_mode") or EVIDENCE_MODE_LEGACY) == EVIDENCE_MODE_REPLAY:
            raise ValueError(
                "consumed historical replay requires the dedicated forward-only admission"
            )
        return self._approve_version(
            version_id,
            actor=actor,
            reason=reason,
            allow_forward_only_rehabilitation=False,
        )

    def admit_forward_only_rehabilitation(
        self,
        version_id: str,
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        """Admit one exact descriptive replay into the existing forward paper stage."""

        version = self.get_version(version_id)
        if str(version.get("evidence_mode") or EVIDENCE_MODE_LEGACY) != EVIDENCE_MODE_REPLAY:
            raise ValueError(
                "forward-only rehabilitation admission requires consumed historical replay"
            )
        return self._approve_version(
            version_id,
            actor=actor,
            reason=reason,
            allow_forward_only_rehabilitation=True,
        )

    def _approve_version(
        self,
        version_id: str,
        *,
        actor: str,
        reason: str,
        allow_forward_only_rehabilitation: bool,
    ) -> dict[str, Any]:
        if not actor.strip() or len(reason.strip()) < 10:
            raise ValueError("actor and a meaningful approval reason are required")
        version = self.get_version(version_id)
        if version.get("is_legacy"):
            raise ValueError("legacy strategy versions must be recreated through evaluation v2")
        backtests = self.list_backtests(version_id=version_id, limit=1)
        if not backtests or backtests[0]["status"] != "succeeded" or not backtests[0]["metrics"]:
            raise ValueError("strategy version requires a successful backtest before approval")
        metrics = backtests[0]["metrics"]
        if backtests[0].get("is_legacy"):
            raise ValueError("legacy backtests cannot approve a new strategy")
        config = version["config"]
        evidence_mode = str(version.get("evidence_mode") or EVIDENCE_MODE_LEGACY)
        replay_admission = evidence_mode == EVIDENCE_MODE_REPLAY
        if replay_admission is not allow_forward_only_rehabilitation:
            raise ValueError("strategy approval evidence mode is not authorized by this action")
        if version.get("strategy_type") == "multifactor":
            if evidence_mode not in {EVIDENCE_MODE_SEALED, EVIDENCE_MODE_REPLAY}:
                raise ValueError("multifactor strategy evidence authority is ambiguous")
            if backtests[0].get("evidence_mode") != evidence_mode:
                raise ValueError("strategy and backtest evidence authority differ")
        if replay_admission:
            require_replay_config(config)
            for label, value in (
                ("metrics", metrics),
                ("provenance", metrics.get("provenance") or {}),
            ):
                require_replay_markers(value, label=f"historical replay {label}")
                if value.get("final_oos_opened") is not True:
                    raise ValueError(
                        "historical replay must admit that its consumed window was opened"
                    )
            if metrics.get("capital_eligible") is not False:
                raise ValueError("historical replay must be explicitly capital-ineligible")
        conservative_incomplete_family = (
            _valid_factor_score_incomplete_family_alternative(version, metrics)
        )
        fin_strategy_admission_binding: dict[str, Any] | None = None
        source_research_artifact_id = str(
            version.get("source_research_artifact_id")
            or config.get("source_research_artifact_id")
            or ""
        ).strip()
        if source_research_artifact_id:
            with self.engine.begin() as connection:
                fin_strategy_admission_binding = (
                    self._require_fin_strategy_formal_admission(connection, version)
                )
            recorded_admission = (backtests[0].get("periods") or {}).get(
                "fin_strategy_formal_admission"
            )
            if recorded_admission != fin_strategy_admission_binding:
                raise ValueError(
                    "fin_strategy formal backtest is not bound to its passed "
                    "policy-only/full-stack evidence"
                )
        # Autopilot strategies are capital-facing only after the single fresh
        # final OOS is both settled in the persistent alpha ledger and bound
        # back to this exact immutable artifact.  ``approve`` is deliberately
        # a second authority boundary: callers cannot bypass the Autopilot
        # completion service by invoking StrategyStore directly.
        requires_capital_oos_receipt = (
            config.get("autopilot_completion_contract_version")
            == "autopilot-completion-v1"
            or bool(source_research_artifact_id)
        )
        if replay_admission and requires_capital_oos_receipt:
            raise ValueError(
                "forward-only rehabilitation cannot reuse a capital OOS or research admission"
            )
        if requires_capital_oos_receipt:
            try:
                receipt = require_capital_oos_receipt(
                    metrics.get("capital_oos_receipt"),
                    backtest_id=str(backtests[0]["id"]),
                    strategy_version_id=version_id,
                    dataset=str(backtests[0]["dataset"]),
                    periods=dict(backtests[0]["periods"]),
                )
                daily_returns = Path(str(backtests[0]["artifact_path"])) / "daily_returns.parquet"
                if not daily_returns.is_file() or capital_oos_sha256_file(daily_returns) != str(
                    receipt["formal_oos_artifact_sha256"]
                ):
                    raise ValueError(
                        "capital OOS receipt artifact does not match the formal backtest"
                    )
                batch = CapitalOOSAlphaLedgerStore(self.database_url).get_batch(
                    str(receipt["batch_id"])
                )
                vintage = CapitalOOSAlphaLedgerStore(self.database_url).get_vintage_binding(
                    str(receipt["batch_id"])
                )
                settlement = batch.get("settlement_evidence_json")
                support = (
                    settlement.get("supporting_evidence")
                    if isinstance(settlement, Mapping)
                    else None
                )
                if (
                    str(batch.get("status") or "") != "settled"
                    or batch.get("passed") is not True
                    or str(batch.get("settlement_evidence_sha256") or "")
                    != str(receipt["batch_settlement_evidence_sha256"])
                    or str(batch.get("dataset_identity_sha256") or "")
                    != str(receipt["dataset_identity_sha256"])
                    or str(batch.get("dataset_lineage_id") or "")
                    != str(receipt["dataset_lineage_id"])
                    or str(batch.get("trading_dates_sha256") or "")
                    != str(receipt["trading_dates_sha256"])
                    or str(batch.get("frozen_bundle_manifest_sha256") or "")
                    != str(receipt["frozen_bundle_manifest_sha256"])
                    or str(batch.get("frozen_baseline_manifest_sha256") or "")
                    != str(receipt["frozen_baseline_manifest_sha256"])
                    or str(vintage.get("final_oos_start") or "")
                    != str(backtests[0]["periods"].get("start") or "")
                    or str(vintage.get("final_oos_end") or "")
                    != str(backtests[0]["periods"].get("end") or "")
                    or not isinstance(support, Mapping)
                    or str(support.get("formal_oos_artifact_sha256") or "")
                    != str(receipt["formal_oos_artifact_sha256"])
                    or str(support.get("backtest_id") or "") != str(backtests[0]["id"])
                    or str(support.get("strategy_version_id") or "") != version_id
                    or str(support.get("dataset") or "")
                    != str(backtests[0]["dataset"])
                    or str(support.get("oos_vintage_id") or "")
                    != str(vintage.get("oos_vintage_id") or "")
                ):
                    raise ValueError(
                        "capital OOS receipt does not match the settled ledger evidence"
                    )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "governed strategy requires a settled passing capital OOS receipt: "
                    + str(exc)
                ) from exc
        if version.get("strategy_type") == "pair":
            return self._approve_pair(
                version,
                backtests[0],
                actor=actor,
                reason=reason,
            )
        drawdown = metrics.get("max_drawdown")
        checks = {
            "tracking_error": (
                metrics.get("tracking_error"),
                config["max_tracking_error"],
                "max",
            ),
            "max_drawdown": (
                abs(float(drawdown)) if drawdown is not None else None,
                config["max_drawdown"],
                "max",
            ),
            "average_turnover": (
                metrics.get("average_turnover"),
                config["max_turnover"],
                "max",
            ),
            "information_ratio": (
                metrics.get("information_ratio"),
                config["min_information_ratio"],
                "min",
            ),
            "sharpe_ratio": (
                metrics.get("sharpe_ratio"),
                config.get("min_sharpe_ratio", 0.0),
                "min",
            ),
            "sortino_ratio": (
                metrics.get("sortino_ratio"),
                config.get("min_sortino_ratio", 0.0),
                "min",
            ),
            "deflated_sharpe_probability": (
                metrics.get("deflated_sharpe_probability"),
                0.95,
                "min",
            ),
            "robustness_pass_rate": (metrics.get("robustness_pass_rate"), 1.0, "min"),
            "event_stress_count": (
                metrics.get("event_stress_count"),
                config.get("event_count", 5),
                "min",
            ),
            "event_stress_pass_rate": (
                metrics.get("event_stress_pass_rate"),
                config.get("min_event_stress_pass_rate", 0.60),
                "min",
            ),
            "trading_days": (
                metrics.get("trading_days"),
                config.get("min_backtest_days", 504),
                "min",
            ),
            "closed_trade_count": (
                metrics.get("closed_trade_count"),
                config.get("min_closed_trades", 20),
                "min",
            ),
            "win_rate": (
                metrics.get("win_rate"),
                config.get("min_win_rate", 0.0),
                "min",
            ),
            "profit_loss_ratio": (
                metrics.get("profit_loss_ratio"),
                config.get("min_profit_loss_ratio", 0.0),
                "min",
            ),
            "capacity_curve_points": (
                metrics.get("capacity_curve_points"),
                3,
                "min",
            ),
        }
        if conservative_incomplete_family:
            # DSR is truthfully not computable because the historical trial
            # matrix does not exist. Its narrow Bonferroni substitute is
            # validated separately; do not convert None into a numeric pass.
            checks.pop("deflated_sharpe_probability")
        if str(config.get("horizon_profile") or LEGACY_AMBIGUOUS) == LEGACY_AMBIGUOUS:
            # Preserve the historical contract for versions whose research
            # horizon is unknown. New horizon strategies prove rolling
            # stability before opening their once-only final OOS.
            checks.update(
                {
                    "rolling_pass_rate": (
                        metrics.get("rolling_pass_rate"),
                        config.get("min_rolling_pass_rate", 0.60),
                        "min",
                    ),
                    "rolling_window_count": (
                        metrics.get("rolling_window_count"),
                        config.get("min_rolling_windows", 3),
                        "min",
                    ),
                }
            )
        execution_method = str(config.get("execution_method", "open"))
        if execution_method in {"twap", "vwap", "next_bar"}:
            checks["capacity_fill_ratio"] = (
                metrics.get("capacity_fill_ratio"),
                config.get("min_capacity_fill_ratio", 0.95),
                "min",
            )
        failures = []
        try:
            require_strategy_execution_contract(config)
        except ValueError as exc:
            failures.append(str(exc))
        if (
            metrics.get("backtest_engine") != "qlib"
            or metrics.get("qlib_native_backtest") is not True
        ):
            failures.append("a Qlib-native backtest is required for approval")
        provenance = metrics.get("provenance")
        expected_factors = {str(item["factor_candidate_id"]) for item in version["factors"]}
        if not isinstance(provenance, dict):
            failures.append("reproducible backtest provenance is required for approval")
        else:
            try:
                require_qlib_workflow_identity(provenance.get("qlib_workflow"))
            except ValueError as exc:
                failures.append(str(exc))
            for field in (
                "dataset_identity_sha256",
                "snapshot_manifest_sha256",
                "qlib_builder_sha256",
                "strategy_config_sha256",
                "execution_manifest_sha256",
            ):
                if not _is_sha256(provenance.get(field)):
                    failures.append(f"provenance {field} must be a SHA-256 digest")
            for field in ("factor_values_sha256", "factor_code_sha256"):
                hashes = provenance.get(field)
                if not isinstance(hashes, dict) or set(hashes) != expected_factors:
                    failures.append(f"provenance {field} does not match strategy factors")
                elif not all(_is_sha256(value) for value in hashes.values()):
                    failures.append(f"provenance {field} contains an invalid SHA-256 digest")
            if execution_method in {"twap", "vwap", "next_bar"}:
                for field in (
                    "execution_dataset_identity_sha256",
                    "execution_snapshot_manifest_sha256",
                    "execution_qlib_builder_sha256",
                ):
                    if not _is_sha256(provenance.get(field)):
                        failures.append(f"provenance {field} must be a SHA-256 digest")
            if (
                not str(provenance.get("qlib_version") or "").strip()
                or provenance.get("qlib_version") == "unknown"
            ):
                failures.append("provenance qlib_version is required")
            qlib_commit = str(provenance.get("qlib_commit") or "")
            if len(qlib_commit) != 40 or any(
                character not in "0123456789abcdef" for character in qlib_commit.lower()
            ):
                failures.append("provenance qlib_commit must identify the pinned upstream")
            try:
                require_daily_qlib_contract(provenance)
            except ValueError as exc:
                failures.append(str(exc))
            if provenance.get("backtest_engine_version") != QLIB_ENGINE_VERSION:
                failures.append("backtest engine version is obsolete or inconsistent")
            if not str(provenance.get("policy_version") or "").strip():
                failures.append("PortfolioPolicy provenance is required")
        if not isinstance(provenance, dict) or metrics.get("policy_version") != provenance.get(
            "policy_version"
        ):
            failures.append("PortfolioPolicy version is missing or inconsistent")
        execution_model_evidence = metrics.get("execution_model")
        if not isinstance(execution_model_evidence, dict) or execution_model_evidence.get(
            "strategy_contract_hash"
        ) != config.get("execution_contract_hash"):
            failures.append("strategy execution contract evidence is missing or inconsistent")
        if metrics.get("event_stress_passed") is not True:
            failures.append("event stress scenarios did not satisfy the configured result gate")
        if (metrics.get("event_stress") or {}).get("state_source") != (
            "full_backtest_carried_positions"
        ):
            failures.append("event stress did not inherit the formal backtest state")
        event_stress = metrics.get("event_stress") or {}
        event_items = event_stress.get("events")
        if (
            event_stress.get("position_state_method") != "formal_fill_ledger_v1"
            or not isinstance(event_items, list)
            or len(event_items) < int(config.get("event_count", 5))
            or any(
                not isinstance(item, dict)
                or item.get("state_source") != "full_backtest_carried_positions"
                or item.get("return_state_source") != "full_backtest_report_slice"
                or not isinstance(item.get("start_holdings"), dict)
                or not isinstance(item.get("state_fill_count"), int)
                for item in event_items
            )
        ):
            failures.append("event stress carried-position evidence is incomplete")
        robustness = metrics.get("robustness")
        artifact_root = Path(backtests[0]["artifact_path"]).resolve()
        if (
            not isinstance(robustness, dict)
            or robustness.get("passed") is not True
            or robustness.get("pass_rate") != 1.0
            or set(robustness.get("scenarios") or {})
            != {"double_cost", "turnover_75pct", "topk_80pct", "zero_retention_buffer"}
        ):
            failures.append("all four independent robustness scenarios are required")
        else:
            failures.extend(_scenario_artifact_failures(robustness["scenarios"], artifact_root))
        component_stress = metrics.get("component_cost_stress")
        if (
            not isinstance(component_stress, dict)
            or component_stress.get("passed") is not True
            or component_stress.get("pass_rate") != 1.0
            or set(component_stress.get("scenarios") or {})
            != set(COMPONENT_COST_STRESS_MULTIPLIERS)
        ):
            failures.append(
                "all component cost stress scenarios are required "
                "(commission/slippage/impact/fill-rate)"
            )
        else:
            failures.extend(
                _scenario_artifact_failures(component_stress["scenarios"], artifact_root)
            )
        if metrics.get("sortino_status") != "ok":
            failures.append("Sortino is undefined or non-finite")
        deflated = metrics.get("deflated_sharpe")
        if not conservative_incomplete_family and (
            not isinstance(deflated, dict)
            or deflated.get("status") != "ok"
            or deflated.get("method_version") != DEFLATED_SHARPE_METHOD_VERSION
        ):
            failures.append("Deflated Sharpe evidence is missing or invalid")
        failures.extend(_formal_validation_failures(version, metrics))
        failures.extend(_pre_final_stability_failures(version, metrics))
        if metrics.get("capacity_curve_passed") is not True:
            failures.append("capacity curve did not satisfy the configured result gate")
        eligibility = metrics.get("eligibility")
        if (
            not isinstance(eligibility, dict)
            or eligibility.get("contract_version") != ELIGIBILITY_CONTRACT_VERSION
            or int(eligibility.get("rows") or 0) <= 0
            or int(eligibility.get("eligible_rows") or 0) <= 0
        ):
            failures.append("point-in-time eligibility evidence is missing or empty")
        elif config.get("require_regulatory_events") and not eligibility.get(
            "regulatory_data_available"
        ):
            failures.append("required regulatory violation data is unavailable")
        if execution_method in {"twap", "vwap", "next_bar"}:
            execution_model = metrics.get("execution_model")
            if not backtests[0].get("execution_dataset"):
                failures.append("minute execution dataset is required for approval")
            if (
                not isinstance(execution_model, dict)
                or execution_model.get("method") != execution_method
                or execution_model.get("frequency") in {None, "day"}
                or execution_model.get("price_assumption")
                not in {"minute bar vwap fills", "next eligible minute bar vwap"}
                or execution_model.get("strategy_contract_hash")
                != config.get("execution_contract_hash")
                or metrics.get("minute_execution_enforced") is not True
            ):
                failures.append("minute-native execution evidence is required")
            try:
                require_minute_execution_contract(
                    {
                        "frequency": (execution_model or {}).get("frequency"),
                        "execution_contract_version": provenance.get("execution_contract_version")
                        if isinstance(provenance, dict)
                        else None,
                        "lineage_verified": provenance.get("execution_lineage_verified")
                        if isinstance(provenance, dict)
                        else None,
                        "fields": provenance.get("execution_fields")
                        if isinstance(provenance, dict)
                        else None,
                        "source_datasets": provenance.get("execution_source_datasets")
                        if isinstance(provenance, dict)
                        else None,
                        "source_unit_contracts": provenance.get("execution_source_unit_contracts")
                        if isinstance(provenance, dict)
                        else None,
                    },
                    frequency=(execution_model or {}).get("frequency"),
                )
            except ValueError as exc:
                failures.append(str(exc))
            if isinstance(provenance, dict) and provenance.get(
                "source_lineage_id"
            ) != provenance.get("execution_source_lineage_id"):
                failures.append("daily and minute backtest datasets do not share source lineage")
        cost_model = metrics.get("cost_model")
        if not isinstance(cost_model, dict):
            failures.append("the unified cost model is required for approval")
        else:
            try:
                effective_costs = CostModelConfig.from_mapping(cost_model)
                backtest_start_date = date.fromisoformat(backtests[0]["periods"]["start"])
                backtest_end_date = date.fromisoformat(backtests[0]["periods"]["end"])
                if date.fromisoformat(effective_costs.effective_from) > backtest_start_date or (
                    effective_costs.effective_to is not None
                    and date.fromisoformat(effective_costs.effective_to) < backtest_end_date
                ):
                    failures.append("cost schedule does not cover the full backtest period")
            except (TypeError, ValueError) as exc:
                failures.append(f"cost schedule is invalid: {exc}")
        failures.extend(_multifactor_manifest_failures(version, backtests[0], metrics))
        failures.extend(
            self._hypothesis_group_manifest_failures(version["id"], backtests[0])
        )
        if isinstance(cost_model, dict) and float(cost_model.get("min_commission", -1.0)) < float(
            config.get("min_commission", 5.0)
        ):
            failures.append("minimum commission evidence is below the configured value")
        for name, (value, threshold, mode) in checks.items():
            if (
                value is None
                or (mode == "max" and value > threshold)
                or (mode == "min" and value < threshold)
            ):
                failures.append(f"{name}={value} violates {mode} {threshold}")
        backtest_start = date.fromisoformat(backtests[0]["periods"]["start"])
        backtest_end = date.fromisoformat(backtests[0]["periods"]["end"])
        expected_model_environment_sha256: str | None = None
        with self.engine.connect() as connection:
            try:
                model_signal_evidence = self._model_signal_evidence(connection, config)
            except ValueError as exc:
                failures.append(str(exc))
                model_signal_evidence = None
            if model_signal_evidence is not None:
                admission_binding = model_signal_evidence.get(
                    "formal_admission_binding"
                )
                admission_grid = (
                    admission_binding.get("model_grid")
                    if isinstance(admission_binding, dict)
                    else None
                )
                expected_model_environment_sha256 = (
                    str(
                        admission_grid.get("execution_environment_sha256") or ""
                    ).lower()
                    if isinstance(admission_grid, dict)
                    else ""
                )
                if not _is_sha256(expected_model_environment_sha256):
                    failures.append(
                        "formal model admission has no immutable execution environment"
                    )
                recorded_admission = (
                    (metrics.get("formal_validation") or {}).get("model_admission")
                    if isinstance(metrics.get("formal_validation"), dict)
                    else None
                )
                if recorded_admission != model_signal_evidence.get("formal_admission_binding"):
                    failures.append(
                        "formal model admission no longer matches the independently "
                        "validated candidate database"
                    )
            for factor in version["factors"]:
                evaluation = connection.execute(
                    select(
                        factor_evaluations.c.test_start,
                        factor_evaluations.c.test_end,
                        factor_evaluations.c.evaluator_version,
                        factor_evaluations.c.is_legacy,
                        factor_evaluations.c.dataset_identity_sha256,
                    ).where(
                        factor_evaluations.c.id == factor["factor_evaluation_id"],
                        factor_evaluations.c.factor_candidate_id == factor["factor_candidate_id"],
                    )
                ).first()
                if evaluation is None:
                    failures.append(
                        f"factor {factor['factor_candidate_id']} has no out-of-sample evidence"
                    )
                elif evaluation.is_legacy or str(evaluation.evaluator_version) != (
                    "factor-gate-v3-hac-bh"
                ):
                    failures.append(
                        f"factor {factor['factor_candidate_id']} uses a legacy evaluation"
                    )
                elif evaluation.dataset_identity_sha256 != provenance.get(
                    "dataset_identity_sha256"
                ):
                    failures.append(
                        f"factor {factor['factor_candidate_id']} dataset identity does not match"
                    )
                elif backtest_start < evaluation.test_start or backtest_end > evaluation.test_end:
                    failures.append(
                        f"backtest {backtest_start}..{backtest_end} falls outside factor "
                        f"test window {evaluation.test_start}..{evaluation.test_end}"
                    )
        if failures:
            raise ValueError("strategy risk gate failed: " + "; ".join(failures))
        prepared_model_artifact: dict[str, Any] | None = None
        if model_signal_evidence is not None:
            # Build and fully revalidate the initial fitted-model record before
            # entering the approval transaction.  It remains an inert
            # candidate until that same transaction approves the StrategySpec.
            from .model_artifact_store import ModelArtifactStore

            backtest_artifact_root = Path(backtests[0]["artifact_path"]).resolve()
            prepared_model_artifact = ModelArtifactStore(
                self.database_url
            ).create_from_formal_backtest(
                strategy_version_id=version_id,
                source_backtest_id=str(backtests[0]["id"]),
                valid_until=None,
                actor=actor,
                backtests_root=backtest_artifact_root.parent,
            )
        now = _now()
        replay_gate = None
        replay_criteria: dict[str, Any] | None = None
        if replay_admission:
            from .promotion import (
                ForwardGateThresholds,
                build_forward_gate_criteria,
                forward_gate_thresholds_for_horizon,
            )

            replay_gate = ForwardGateThresholds(
                **rehabilitation_forward_thresholds(
                    asdict(
                        forward_gate_thresholds_for_horizon(
                            str(version["horizon_profile"])
                        )
                    )
                )
            )
            replay_criteria = build_forward_gate_criteria(
                horizon_profile=str(version["horizon_profile"]),
                horizon_contract_sha256=str(version["horizon_contract_sha256"]),
                thresholds=replay_gate,
            )
        with self.engine.begin() as connection:
            locked_version = connection.execute(
                select(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .with_for_update()
            ).first()
            if locked_version is None:
                raise KeyError(version_id)
            locked_group = connection.scalar(
                select(strategies.c.economic_hypothesis_group).where(
                    strategies.c.id == locked_version.strategy_id
                )
            )
            _lock_strategy_trial_family(connection, str(locked_group or ""))
            locked_backtest = connection.execute(
                select(backtest_runs)
                .where(backtest_runs.c.id == backtests[0]["id"])
                .with_for_update()
            ).first()
            if (
                locked_backtest is None
                or str(locked_backtest.strategy_version_id) != version_id
                or str(locked_backtest.status) != "succeeded"
            ):
                raise ValueError("approval backtest changed during approval")
            fresh_hypothesis = self.hypothesis_group_evidence(
                version_id,
                connection=connection,
            )
            fresh_family_failures = self._hypothesis_group_manifest_failures(
                version_id,
                backtests[0],
                connection=connection,
            )
            if fresh_family_failures:
                raise ValueError(
                    "strategy trial family changed during approval: "
                    + "; ".join(fresh_family_failures)
                )
            if replay_admission:
                frozen_eligibility = incomplete_family_eligibility_for_version(
                    connection,
                    strategy_version_id=version_id,
                    hypothesis_group_evidence=fresh_hypothesis,
                )
                formal = dict(locked_backtest.metrics_json or {}).get(
                    "formal_validation"
                )
                multiple = (
                    formal.get("multiple_testing")
                    if isinstance(formal, Mapping)
                    else None
                )
                if (
                    not isinstance(multiple, Mapping)
                    or multiple.get("eligibility_receipt_sha256")
                    != frozen_eligibility["receipt_sha256"]
                ):
                    raise ValueError(
                        "conservative statistics do not bind the frozen family eligibility"
                    )
            from .promotion import require_horizon_challenger_capacity

            require_horizon_challenger_capacity(
                connection,
                horizon_profile=str(locked_version.horizon_profile),
                version_id=version_id,
            )
            if fin_strategy_admission_binding is not None:
                locked_value = row_dict(locked_version)
                locked_value["config"] = dict(locked_version.config_json or {})
                fresh_admission = self._require_fin_strategy_formal_admission(
                    connection,
                    locked_value,
                )
                if fresh_admission != fin_strategy_admission_binding:
                    raise ValueError(
                        "fin_strategy admission evidence changed during approval"
                    )
            activated_model_artifact_id: str | None = None
            if prepared_model_artifact is not None:
                locked_backtest = connection.execute(
                    select(backtest_runs)
                    .where(backtest_runs.c.id == backtests[0]["id"])
                    .with_for_update()
                ).first()
                artifact = connection.execute(
                    select(model_artifacts)
                    .where(model_artifacts.c.id == prepared_model_artifact["id"])
                    .with_for_update()
                ).first()
                artifact_path = (
                    Path(str(artifact.artifact_path)).resolve()
                    if artifact is not None
                    else None
                )
                expected_artifact_key = f"formal-backtest-{backtests[0]['id']}"
                if (
                    locked_backtest is None
                    or str(locked_backtest.strategy_version_id) != version_id
                    or str(locked_backtest.status) != "succeeded"
                    or artifact is None
                    or str(artifact.strategy_version_id) != version_id
                    or str(artifact.artifact_key) != expected_artifact_key
                    or str(artifact.status) != "candidate"
                    or str(artifact.strategy_spec_sha256)
                    != str(prepared_model_artifact["strategy_spec_sha256"])
                    or str(artifact.model_recipe_sha256)
                    != str(config.get("model_recipe_sha256") or "")
                    or _canonical_sha256(dict(artifact.model_recipe_json or {}))
                    != str(artifact.model_recipe_sha256)
                    or str(artifact.dataset)
                    != str(model_signal_evidence["candidate"].dataset)
                    or str(artifact.dataset_identity_sha256)
                    != str(model_signal_evidence["candidate"].dataset_identity_sha256)
                    or str(artifact.execution_environment_sha256)
                    != str(expected_model_environment_sha256 or "")
                    or artifact.valid_until <= now
                    or artifact_path is None
                    or not artifact_path.is_file()
                    or _sha256_file(artifact_path) != str(artifact.artifact_sha256)
                    or str(artifact.artifact_sha256)
                    != str(artifact.predictions_sha256)
                ):
                    raise ValueError(
                        "model StrategySpec approval requires the exact intact formal "
                        "ModelArtifact candidate"
                    )
                existing_active = connection.execute(
                    select(model_artifacts.c.id)
                    .where(
                        model_artifacts.c.strategy_version_id == version_id,
                        model_artifacts.c.status == "active",
                    )
                    .with_for_update()
                ).first()
                if existing_active is not None:
                    raise ValueError(
                        "model StrategySpec has an active artifact before atomic approval"
                    )
                connection.execute(
                    update(model_artifacts)
                    .where(
                        model_artifacts.c.id == artifact.id,
                        model_artifacts.c.status == "candidate",
                    )
                    .values(
                        status="active",
                        activated_by=actor,
                        activated_at=now,
                        retired_at=None,
                    )
                )
                activated_model_artifact_id = str(artifact.id)
            rehabilitation_receipt: dict[str, Any] | None = None
            if replay_admission:
                if replay_gate is None or replay_criteria is None:
                    raise ValueError("forward-only rehabilitation gate was not frozen")
                locked_version_value = row_dict(locked_version)
                locked_version_value["config"] = dict(locked_version.config_json or {})
                locked_backtest_value = row_dict(locked_backtest)
                locked_backtest_value["periods"] = dict(
                    locked_backtest.periods_json or {}
                )
                locked_backtest_value["metrics"] = dict(
                    locked_backtest.metrics_json or {}
                )
                rehabilitation_receipt = insert_qualification(
                    connection,
                    build_qualification(
                        connection,
                        version=locked_version_value,
                        backtest=locked_backtest_value,
                        forward_criteria=replay_criteria,
                        created_by=actor,
                    ),
                    created_at=now,
                )
                connection.execute(
                    insert(strategy_forward_gates).values(
                        strategy_version_id=version_id,
                        **asdict(replay_gate),
                        criteria_json=replay_criteria,
                        criteria_sha256=rehabilitation_canonical_sha256(
                            replay_criteria
                        ),
                        registered_by=actor.strip(),
                        registered_at=now,
                        updated_at=now,
                    )
                )
            connection.execute(
                update(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .values(
                    status="approved",
                    # Design 6.11: passing the formal hard gate automatically
                    # moves the version to the isolated paper stage; forward
                    # evidence starts accumulating from zero.
                    promotion_stage="paper",
                    approved_by=actor,
                    approval_reason=reason,
                    approved_at=now,
                )
            )
            connection.execute(
                update(strategies)
                .where(strategies.c.id == version["strategy_id"])
                .values(status="approved", updated_at=now)
            )
            self._event(
                connection,
                strategy_id=version["strategy_id"],
                version_id=version_id,
                event_type="strategy.approved",
                actor=actor,
                payload={
                    "reason": reason,
                    "backtest_id": backtests[0]["id"],
                    "gate_evidence": {name: value[0] for name, value in checks.items()},
                    **(
                        {"model_artifact_id": activated_model_artifact_id}
                        if activated_model_artifact_id is not None
                        else {}
                    ),
                    **(
                        {
                            "evidence_mode": EVIDENCE_MODE_REPLAY,
                            "authority": "historical_description_only",
                            "forward_only_receipt_sha256": rehabilitation_receipt[
                                "receipt_sha256"
                            ],
                        }
                        if rehabilitation_receipt is not None
                        else {"evidence_mode": evidence_mode}
                    ),
                },
            )
        # Design 6.11/7.4: candidate -> paper is automatic once the formal
        # hard gate passes.  After the approval commit, a separate transaction
        # first freezes the forward gate and only then may another transaction
        # create the isolated paper stage.  A crash between the two leaves the
        # safe, retryable state "gate registered, no paper evidence".
        from .promotion import PromotionStore

        promotion = PromotionStore(self.database_url)
        try:
            promotion.prepare_paper_stage(version_id, actor=actor)
        except Exception as exc:  # noqa: BLE001 - approval is already committed
            promotion.record_paper_stage_failure(version_id, actor=actor, error=str(exc))
        return self.get_version(version_id)

    @staticmethod
    def append_health_snapshot_in_transaction(
        connection: Any,
        version_id: str,
        *,
        as_of: datetime,
        health_status: str,
        criteria: Mapping[str, Any],
        evidence: Mapping[str, Any],
        actor: str,
        recorded_at: datetime | None = None,
    ) -> str:
        """Append one sealed health observation using the caller's transaction.

        The strategy-version row is locked here, so promotion can create the
        first health observation and switch recommendation authority atomically.
        Exact content-addressed retries return the existing id.
        """

        allowed = {
            "healthy",
            "watch",
            "restricted",
            "suspended",
            "retired",
        }
        if health_status not in allowed:
            raise ValueError(f"unsupported strategy health status: {health_status}")
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("strategy health as_of must be timezone-aware")
        if len(actor.strip()) < 2:
            raise ValueError("a responsible health-snapshot actor is required")
        criteria_json = dict(criteria)
        evidence_json = dict(evidence)
        if not criteria_json:
            raise ValueError("strategy health criteria must not be empty")
        normalized_as_of = as_of.astimezone(UTC).replace(microsecond=0)
        version = connection.execute(
            select(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .with_for_update()
        ).first()
        if version is None:
            raise KeyError(version_id)
        version_row = row_dict(version)
        horizon = require_horizon_row(version_row)
        criteria_sha256 = horizon_canonical_sha256(criteria_json)
        evidence_sha256 = horizon_canonical_sha256(evidence_json)
        snapshot = {
            "contract_version": "strategy-health-snapshot-v1",
            "strategy_version_id": version_id,
            "horizon_profile": horizon.horizon_profile,
            "horizon_contract_sha256": horizon.sha256,
            "as_of": normalized_as_of.isoformat(),
            "health_status": health_status,
            "criteria_json": criteria_json,
            "criteria_sha256": criteria_sha256,
            "evidence_json": evidence_json,
            "evidence_sha256": evidence_sha256,
            "recorded_by": actor.strip(),
        }
        snapshot_sha256 = horizon_canonical_sha256(snapshot)
        values = {
            "id": snapshot_sha256,
            "strategy_version_id": version_id,
            "horizon_profile": horizon.horizon_profile,
            "as_of": normalized_as_of,
            "health_status": health_status,
            "criteria_json": criteria_json,
            "criteria_sha256": criteria_sha256,
            "evidence_json": evidence_json,
            "evidence_sha256": evidence_sha256,
            "snapshot_sha256": snapshot_sha256,
            "recorded_by": actor.strip(),
            "recorded_at": recorded_at or _now(),
        }
        existing = connection.execute(
            select(strategy_health_snapshots.c.id).where(
                strategy_health_snapshots.c.id == snapshot_sha256
            )
        ).first()
        if existing is not None:
            return snapshot_sha256
        # Automatic evidence and human emergency controls share one immutable
        # ledger, but they are independent observation chains.  A collector
        # recovery must not be rejected because an operator latched production
        # risk, and a later operator release must not inherit the collector's
        # transition state.  Other system writers remain isolated by actor.
        normalized_actor = actor.strip()
        chain_predicates = [
            strategy_health_snapshots.c.strategy_version_id == version_id
        ]
        manual_control = not normalized_actor.startswith("system:")
        if normalized_actor == COLLECTOR_ACTOR:
            chain_predicates.append(
                strategy_health_snapshots.c.recorded_by == COLLECTOR_ACTOR
            )
        elif manual_control:
            chain_predicates.extend(
                [
                    strategy_health_snapshots.c.recorded_by != COLLECTOR_ACTOR,
                    strategy_health_snapshots.c.recorded_by.not_like("system:%"),
                ]
            )
        else:
            chain_predicates.append(
                strategy_health_snapshots.c.recorded_by == normalized_actor
            )
        latest_health = connection.execute(
            select(
                strategy_health_snapshots.c.health_status,
                strategy_health_snapshots.c.as_of,
            )
            .where(*chain_predicates)
            .order_by(
                strategy_health_snapshots.c.as_of.desc(),
                strategy_health_snapshots.c.recorded_at.desc(),
            )
            .limit(1)
        ).first()
        previous_status = (
            str(latest_health.health_status) if latest_health is not None else None
        )
        if latest_health is not None and normalized_as_of < latest_health.as_of:
            raise ValueError(
                "strategy health snapshots must not move as_of backward within actor chain"
            )
        # Human healthy/watch is an explicit release command for the durable
        # manual latch after a temporary restriction or suspension.  It still
        # cannot authorize production without a subsequent sealed collector
        # row.  Retired remains terminal for every actor chain.
        allowed_transition = (
            health_status
            if manual_control
            and previous_status != RETIRED
            and health_status in {HEALTHY, WATCH}
            else transition_strategy_health(previous_status, health_status)
        )
        if allowed_transition != health_status:
            raise ValueError(
                "strategy health recovery must proceed one state at a time: "
                f"{previous_status} -> {allowed_transition}"
            )
        connection.execute(insert(strategy_health_snapshots).values(**values))
        return snapshot_sha256

    def record_health_snapshot(
        self,
        version_id: str,
        *,
        as_of: datetime,
        health_status: str,
        criteria: Mapping[str, Any],
        evidence: Mapping[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        """Append one sealed point-in-time strategy health assessment."""

        with self.engine.begin() as connection:
            snapshot_sha256 = self.append_health_snapshot_in_transaction(
                connection,
                version_id,
                as_of=as_of,
                health_status=health_status,
                criteria=criteria,
                evidence=evidence,
                actor=actor,
            )
        return self.get_health_snapshot(snapshot_sha256)

    def get_health_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(strategy_health_snapshots).where(
                    strategy_health_snapshots.c.id == snapshot_id
                )
            ).first()
        if row is None:
            raise KeyError(snapshot_id)
        result = row_dict(row)
        criteria = dict(result.get("criteria_json") or {})
        evidence = dict(result.get("evidence_json") or {})
        if (
            horizon_canonical_sha256(criteria) != result.get("criteria_sha256")
            or horizon_canonical_sha256(evidence) != result.get("evidence_sha256")
        ):
            raise ValueError("strategy health snapshot evidence seal is invalid")
        with self.engine.connect() as connection:
            version = connection.execute(
                select(strategy_versions).where(
                    strategy_versions.c.id == result["strategy_version_id"]
                )
            ).first()
        if version is None:
            raise ValueError("strategy health snapshot references a missing strategy version")
        horizon = require_horizon_row(row_dict(version))
        snapshot = {
            "contract_version": "strategy-health-snapshot-v1",
            "strategy_version_id": result["strategy_version_id"],
            "horizon_profile": result["horizon_profile"],
            "horizon_contract_sha256": horizon.sha256,
            "as_of": result["as_of"],
            "health_status": result["health_status"],
            "criteria_json": criteria,
            "criteria_sha256": result["criteria_sha256"],
            "evidence_json": evidence,
            "evidence_sha256": result["evidence_sha256"],
            "recorded_by": result["recorded_by"],
        }
        if horizon_canonical_sha256(snapshot) != result.get("snapshot_sha256"):
            raise ValueError("strategy health snapshot content seal is invalid")
        result["criteria"] = criteria
        result["evidence"] = evidence
        return result

    def find_version_by_source_artifact(
        self, source_research_artifact_id: str
    ) -> dict[str, Any] | None:
        """Return the one governed draft materialized from a compiled research artifact."""

        source_id = str(source_research_artifact_id or "").strip()
        if not source_id:
            raise ValueError("source research artifact id is required")
        with self.engine.connect() as connection:
            version_id = connection.scalar(
                select(strategy_versions.c.id).where(
                    strategy_versions.c.source_research_artifact_id == source_id
                )
            )
        return self.get_version(str(version_id)) if version_id else None

    def list_health_snapshots(
        self, version_id: str, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        if limit < 1 or limit > 1000:
            raise ValueError("health snapshot limit must be between 1 and 1000")
        with self.engine.connect() as connection:
            ids = connection.execute(
                select(strategy_health_snapshots.c.id)
                .where(strategy_health_snapshots.c.strategy_version_id == version_id)
                .order_by(
                    strategy_health_snapshots.c.as_of.desc(),
                    strategy_health_snapshots.c.recorded_at.desc(),
                )
                .limit(limit)
            ).scalars().all()
        return [self.get_health_snapshot(str(snapshot_id)) for snapshot_id in ids]

    @staticmethod
    def _event(
        connection: Any,
        *,
        strategy_id: str,
        version_id: str | None,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            insert(strategy_events).values(
                strategy_id=strategy_id,
                strategy_version_id=version_id,
                event_type=event_type,
                actor=actor,
                payload_json=payload,
                created_at=_now(),
            )
        )
