from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pytest
from qlib_test_doubles import qlib_workflow_identity

import quant_platform.strategy_store as strategy_store_module
from quant_platform.factor_autopilot import (
    canonical_sha256,
    factor_sota_admission_path,
    resolve_sota_roll_forward,
    validate_factor_sota_result_contract,
    validate_promoted_factor_sota_admission,
)
from quant_platform.factor_library import (
    FACTOR_LIBRARY_CONTRACT_VERSION,
    compile_qlib_expression,
)
from quant_platform.factor_library_store import (
    INCREMENTAL_EVIDENCE_VERSION,
    validate_factor_definition_immutability,
    validate_sota_roll_forward,
)
from quant_platform.research_automation import build_multi_profile_consensus
from quant_platform.research_store import FactorGatePolicy
from quant_platform.strategy_store import (
    _validate_governed_factor_evaluation,
    _validate_governed_profile_consensus,
)
from quant_platform.upstream_versions import QLIB_COMMIT, RDAGENT_COMMIT

pytestmark = pytest.mark.no_database


def test_factor_sota_dual_path_never_repairs_a_hard_gate_failure() -> None:
    passed = {
        profile_id: {"hard_status": "passed", "effect_status": "passed"}
        for profile_id in ("recent_3y", "balanced_5y", "robust_10y")
    }
    assert factor_sota_admission_path("profile_pending", passed) == "standalone"

    weak = {profile_id: dict(value) for profile_id, value in passed.items()}
    weak["recent_3y"]["effect_status"] = "failed"
    assert factor_sota_admission_path("incremental_pending", weak) == "incremental"

    weak["robust_10y"]["hard_status"] = "failed"
    assert factor_sota_admission_path("incremental_pending", weak) is None


def test_strategy_consensus_preserves_robust_stress_profile_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = {
        "id": "candidate-a",
        "code_sha256": "a" * 64,
        "values_sha256": "b" * 64,
    }

    def evaluation(profile_id: str, *, gate_status: str) -> dict:
        suffix = {"recent_3y": "1", "robust_10y": "2", "balanced_5y": "3"}[
            profile_id
        ]
        valid_start = {
            "recent_3y": "2022-01-03",
            "balanced_5y": "2020-01-02",
            "robust_10y": "2015-01-05",
        }[profile_id]
        train_end = {
            "recent_3y": "2021-12-31",
            "balanced_5y": "2019-12-31",
            "robust_10y": "2015-01-02",
        }[profile_id]
        return {
            "id": suffix * 32,
            "factor_candidate_id": candidate["id"],
            "dataset_identity_sha256": "d" * 64,
            "train_start": "2010-01-04",
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": "2024-12-20",
            "test_start": "2025-01-02",
            "test_end": "2025-12-31",
            "candidate_code_sha256": candidate["code_sha256"],
            "candidate_values_sha256": candidate["values_sha256"],
            "evidence_sha256": suffix * 64,
            "metrics_sha256": suffix * 64,
            "gate_status": gate_status,
            "metrics_json": {
                "research_profile": {"id": profile_id},
                "direction": "original",
                "coverage_gate_passed": True,
                "cost_adjusted_return": 0.03,
            },
        }

    evaluations = {
        "recent_3y": evaluation("recent_3y", gate_status="passed"),
        "balanced_5y": evaluation("balanced_5y", gate_status="passed"),
        "robust_10y": evaluation("robust_10y", gate_status="failed"),
    }
    sealed = build_multi_profile_consensus(
        {
            **candidate,
            "profile_evaluations": [
                {**item, "metrics": dict(item["metrics_json"])}
                for item in evaluations.values()
            ],
        }
    )
    assert sealed is not None
    gate_requirements: dict[str, bool] = {}

    def validate_evaluation(
        evaluation_row: dict,
        candidate_row: dict,
        *,
        expected_profile_id: str | None,
        require_gate_passed: bool,
    ) -> dict:
        assert candidate_row == candidate
        assert evaluation_row["id"] == evaluations[expected_profile_id]["id"]
        gate_requirements[str(expected_profile_id)] = require_gate_passed
        return {"hard_status": "passed", "effect_status": "passed"}

    monkeypatch.setattr(
        strategy_store_module,
        "_validate_governed_factor_evaluation",
        validate_evaluation,
    )

    _validate_governed_profile_consensus(candidate, evaluations, sealed)

    assert gate_requirements == {
        "balanced_5y": True,
        "recent_3y": True,
        "robust_10y": False,
    }

    invalid_robust = {
        **evaluations,
        "robust_10y": {
            **evaluations["robust_10y"],
            "metrics_json": {
                **evaluations["robust_10y"]["metrics_json"],
                "cost_adjusted_return": -0.01,
            },
        },
    }
    with pytest.raises(ValueError, match="no longer satisfies governed admission"):
        _validate_governed_profile_consensus(candidate, invalid_robust, sealed)


def _roll_forward_inputs() -> tuple[dict, list[dict], list[dict]]:
    current = {
        "name": "cn-20260826",
        "lineage_id": "cn-daily",
        "lineage_verified": True,
        "end_date": "2026-08-26",
        "provenance": {"dataset_identity_sha256": "b" * 64},
    }
    versions = [
        {
            "id": "sota-old",
            "status": "active",
            "dataset": "cn-20260825",
            "dataset_identity_sha256": "a" * 64,
            "evidence": {},
        }
    ]
    catalog = [
        {
            "name": "cn-20260825",
            "lineage_id": "cn-daily",
            "lineage_verified": True,
            "end_date": "2026-08-25",
            "provenance": {"dataset_identity_sha256": "a" * 64},
        }
    ]
    return current, versions, catalog


def test_factor_sota_rolls_forward_same_lineage_instead_of_alpha158() -> None:
    current, versions, catalog = _roll_forward_inputs()

    resolved = resolve_sota_roll_forward(current, versions, catalog)

    assert resolved is not None
    assert resolved["mode"] == "roll_forward"
    assert resolved["predecessor_id"] == "sota-old"
    assert resolved["source_end_date"] == "2026-08-25"
    assert resolved["target_end_date"] == "2026-08-26"


def test_factor_sota_refuses_fallback_when_prior_provenance_is_missing() -> None:
    current, versions, _ = _roll_forward_inputs()

    with pytest.raises(ValueError, match="refusing Alpha158 fallback"):
        resolve_sota_roll_forward(current, versions, [])


def test_factor_sota_refuses_non_monotonic_same_lineage_roll_forward() -> None:
    current, versions, catalog = _roll_forward_inputs()
    catalog[0]["end_date"] = current["end_date"]

    with pytest.raises(ValueError, match="did not increase monotonically"):
        resolve_sota_roll_forward(current, versions, catalog)


def test_factor_sota_roll_forward_contract_seals_member_and_feature_hashes() -> None:
    current, versions, catalog = _roll_forward_inputs()
    resolved = resolve_sota_roll_forward(current, versions, catalog)
    assert resolved is not None
    resolved.update(
        {
            "source_sota_evidence_sha256": "c" * 64,
            "source_feature_set_definition_sha256": "d" * 64,
            "source_member_set_sha256": "e" * 64,
        }
    )
    validate_sota_roll_forward(
        resolved,
        predecessor_id="sota-old",
        target_dataset=current["name"],
        target_dataset_identity_sha256=current["provenance"][
            "dataset_identity_sha256"
        ],
        target_lineage_id=current["lineage_id"],
        target_end_date=current["end_date"],
    )

    resolved["source_member_set_sha256"] = "short"
    with pytest.raises(ValueError, match="source_member_set_sha256"):
        validate_sota_roll_forward(
            resolved,
            predecessor_id="sota-old",
            target_dataset=current["name"],
            target_dataset_identity_sha256=current["provenance"][
                "dataset_identity_sha256"
            ],
            target_lineage_id=current["lineage_id"],
            target_end_date=current["end_date"],
        )


def test_factor_sota_roll_forward_rebuilds_definition_hash() -> None:
    expression = "Mean($close/Ref($close,1)-1,20)"
    compiled = compile_qlib_expression(expression)
    definition = {
        "expression": expression,
        "expression_sha256": compiled.expression_sha256,
        "required_fields": list(compiled.required_fields),
        "max_lookback_days": compiled.max_lookback_days,
        "economic_family": "price_action",
        "qlib_commit": QLIB_COMMIT,
        "status": "registered",
    }
    identity = {
        "contract_version": FACTOR_LIBRARY_CONTRACT_VERSION,
        "expression_sha256": compiled.expression_sha256,
        "economic_family": definition["economic_family"],
        "required_fields": definition["required_fields"],
        "max_lookback_days": definition["max_lookback_days"],
        "qlib_commit": QLIB_COMMIT,
    }
    definition["definition_sha256"] = canonical_sha256(identity)

    validate_factor_definition_immutability(definition)

    definition["expression"] = "Mean($close/Ref($close,1)-1,21)"
    with pytest.raises(ValueError, match="changed in place"):
        validate_factor_definition_immutability(definition)


def _sota_contract() -> tuple[dict, dict]:
    frozen = {
        "kind": "governed_lightgbm_factor_ablation",
        "model_artifact_id": "model-a",
        "model_artifact_sha256": "1" * 64,
        "training_recipe_sha256": "2" * 64,
        "feature_contract_sha256": "3" * 64,
    }
    payload = {
        "dataset": "snapshot-a",
        "dataset_identity_sha256": "4" * 64,
        "dataset_lineage_id": "cn-daily",
        "dataset_end_date": "2026-08-25",
        "predecessor_id": "sota-a",
        "predecessor_roll_forward": {
            "contract_version": "sota-roll-forward-v1",
            "mode": "exact",
            "predecessor_id": "sota-a",
            "source_dataset": "snapshot-a",
            "source_dataset_identity_sha256": "4" * 64,
            "source_end_date": "2026-08-25",
            "target_dataset": "snapshot-a",
            "target_dataset_identity_sha256": "4" * 64,
            "target_end_date": "2026-08-25",
            "dataset_lineage_id": "cn-daily",
            "lineage_verified": True,
            "source_sota_evidence_sha256": "5" * 64,
            "source_feature_set_definition_sha256": "6" * 64,
            "source_member_set_sha256": "7" * 64,
        },
        "frozen_model": frozen,
        "frozen_model_sha256": canonical_sha256(frozen),
        "final_oos_opened": False,
        "candidates": [
            {"factor_candidate_id": "factor-a", "admission_path": "standalone"},
            {"factor_candidate_id": "factor-b", "admission_path": "incremental"},
        ],
    }
    result = {
        "contract_version": "factor-sota-paired-frozen-model-v2",
        "status": "failed",
        "dataset_identity_sha256": payload["dataset_identity_sha256"],
        "predecessor_id": payload["predecessor_id"],
        "frozen_model": frozen,
        "trials": [
            {"factor_candidate_id": "factor-a", "status": "rejected"},
            {"factor_candidate_id": "factor-b", "status": "failed"},
        ],
        "accepted": [],
        "attempted_hypotheses": 2,
    }
    return payload, result


def test_factor_sota_contract_preserves_complete_negative_trial_family() -> None:
    payload, result = _sota_contract()
    validate_factor_sota_result_contract(payload, result)

    result["trials"].pop()
    with pytest.raises(ValueError, match="omitted preregistered trials"):
        validate_factor_sota_result_contract(payload, result)


def test_factor_sota_contract_allows_only_one_atomic_acceptance() -> None:
    payload, result = _sota_contract()
    result.update(
        {
            "status": "passed",
            "trials": [
                {"factor_candidate_id": "factor-a", "status": "accepted"},
                {"factor_candidate_id": "factor-b", "status": "accepted"},
            ],
            "accepted": [
                {"factor_candidate_id": "factor-a"},
                {"factor_candidate_id": "factor-b"},
            ],
        }
    )
    with pytest.raises(ValueError, match="at most one"):
        validate_factor_sota_result_contract(payload, result)


def _accepted_sota_contract() -> tuple[dict, dict]:
    frozen = {
        "kind": "governed_lightgbm_factor_ablation",
        "model_artifact_id": "model-a",
        "model_artifact_sha256": "1" * 64,
        "training_recipe_sha256": "2" * 64,
        "feature_contract_sha256": "3" * 64,
    }
    evaluation_ids = {
        "recent_3y": "a" * 32,
        "balanced_5y": "b" * 32,
        "robust_10y": "c" * 32,
    }
    common_profile = {
        "evaluation_evidence_sha256": "d" * 64,
        "baseline_rank_ic": 0.02,
        "proposed_rank_ic": 0.03,
        "baseline_cost_adjusted_return": 0.01,
        "proposed_cost_adjusted_return": 0.02,
        "hard_gate_status": "passed",
        "stability_gate_status": "passed",
        "prediction_evidence": {
            "baseline_prediction_sha256": "4" * 64,
            "proposed_prediction_sha256": "5" * 64,
            "paired_index_sha256": "6" * 64,
            "final_oos_observations_exposed": False,
        },
        "paired_rank_ic_hac": {"status": "ok", "mean": 0.01, "p_value": 0.01},
        "paired_cost_return_bootstrap": {
            "status": "ok",
            "confidence_interval_95": [0.001, 0.02],
            "one_sided_p_value": 0.01,
        },
    }
    incremental = {
        "version": INCREMENTAL_EVIDENCE_VERSION,
        "factor_candidate_id": "factor-a",
        "candidate_code_sha256": "7" * 64,
        "candidate_values_sha256": "8" * 64,
        "dataset_identity_sha256": "9" * 64,
        "frozen_model": frozen,
        "evaluation_ids": evaluation_ids,
        "profile_periods": {
            profile_id: {
                "train_start": "2010-01-01",
                "train_end": "2020-12-31",
                "valid_start": "2021-01-01",
                "valid_end": "2023-12-31",
                "test_start": "2024-02-01",
                "test_end": "2025-12-31",
            }
            for profile_id in evaluation_ids
        },
        "window_role_policy": {
            "version": "nested-profile-roles-v1",
            "recent_role": "ranking_and_significance",
            "balanced_role": "non_degradation",
            "robust_role": "direction_and_crash_stress",
            "nested_windows_count_as_independent": False,
            "combined_profile_p_value": None,
        },
        "multiplicity": {
            "method": "benjamini_hochberg",
            "experiment_family_id": "family-a",
            "hypothesis_count": 1,
            "rank_ic_q_value": 0.01,
            "cost_return_q_value": 0.01,
        },
        "profiles": {
            profile_id: {
                **common_profile,
                "delta_rank_ic": 0.01 if profile_id == "recent_3y" else 0.0,
                "delta_cost_adjusted_return": (
                    0.01 if profile_id == "recent_3y" else 0.0
                ),
            }
            for profile_id in evaluation_ids
        },
    }
    proposal = {
        "factor_candidate_id": "factor-a",
        "admission_path": "incremental",
        "candidate_code_sha256": incremental["candidate_code_sha256"],
        "candidate_values_sha256": incremental["candidate_values_sha256"],
        "profile_evaluation_ids": evaluation_ids,
        "similarity_cluster_id": "cluster-a",
    }
    member = {
        **proposal,
        "action": "added",
        "replaced_factor_candidate_id": None,
        "incremental_evidence": incremental,
    }
    payload = {
        "dataset": "snapshot-b",
        "dataset_identity_sha256": incremental["dataset_identity_sha256"],
        "dataset_lineage_id": "cn-daily",
        "dataset_end_date": "2026-08-25",
        "predecessor_id": None,
        "predecessor_roll_forward": None,
        "frozen_model": frozen,
        "frozen_model_sha256": canonical_sha256(frozen),
        "final_oos_opened": False,
        "baseline_members": [],
        "candidates": [proposal],
    }
    result = {
        "contract_version": "factor-sota-paired-frozen-model-v2",
        "status": "passed",
        "dataset_identity_sha256": incremental["dataset_identity_sha256"],
        "predecessor_id": None,
        "frozen_model": frozen,
        "trials": [{"factor_candidate_id": "factor-a", "status": "accepted"}],
        "accepted": [member],
        "members": [member],
        "attempted_hypotheses": 1,
    }
    return payload, result


def test_factor_sota_acceptance_is_bound_to_preregistered_frozen_model() -> None:
    payload, result = _accepted_sota_contract()
    validate_factor_sota_result_contract(payload, result)

    result["accepted"][0]["incremental_evidence"]["candidate_code_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="changed after preregistration"):
        validate_factor_sota_result_contract(payload, result)


def test_factor_sota_contract_preserves_preregistered_admission_path() -> None:
    payload, result = _accepted_sota_contract()
    result["accepted"][0]["admission_path"] = "standalone"

    with pytest.raises(ValueError, match="changed after preregistration"):
        validate_factor_sota_result_contract(payload, result)


def test_factor_sota_contract_accepts_standalone_after_paired_ablation() -> None:
    payload, result = _accepted_sota_contract()
    payload["candidates"][0]["admission_path"] = "standalone"
    result["accepted"][0]["admission_path"] = "standalone"

    validate_factor_sota_result_contract(payload, result)


def test_promoted_standalone_keeps_consensus_and_paired_member_evidence() -> None:
    _, result = _accepted_sota_contract()
    paired = result["accepted"][0]["incremental_evidence"]
    consensus = {
        "status": "passed",
        "candidate_id": "factor-a",
        "candidate_code_sha256": paired["candidate_code_sha256"],
        "candidate_values_sha256": paired["candidate_values_sha256"],
        "dataset_identity_sha256": paired["dataset_identity_sha256"],
        "evaluation_ids": paired["evaluation_ids"],
        "profile_periods": paired["profile_periods"],
    }
    candidate = {
        "id": "factor-a",
        "status": "promoted",
        "admission_path": "standalone",
        "code_sha256": paired["candidate_code_sha256"],
        "values_sha256": paired["candidate_values_sha256"],
        "profile_consensus": consensus,
        "profile_consensus_sha256": canonical_sha256(consensus),
        "incremental_evidence": None,
        "incremental_evidence_sha256": None,
    }

    validate_promoted_factor_sota_admission(
        candidate,
        admission_path="standalone",
        paired_evidence=paired,
    )

    candidate["profile_consensus"]["evaluation_ids"]["recent_3y"] = "f" * 32
    with pytest.raises(ValueError, match="profile consensus"):
        validate_promoted_factor_sota_admission(
            candidate,
            admission_path="standalone",
            paired_evidence=paired,
        )


def test_promoted_incremental_requires_same_paired_evidence_hash() -> None:
    _, result = _accepted_sota_contract()
    paired = result["accepted"][0]["incremental_evidence"]
    candidate = {
        "id": "factor-a",
        "status": "promoted",
        "admission_path": "incremental",
        "code_sha256": paired["candidate_code_sha256"],
        "values_sha256": paired["candidate_values_sha256"],
        "profile_consensus": None,
        "incremental_evidence": paired,
        "incremental_evidence_sha256": canonical_sha256(paired),
    }

    validate_promoted_factor_sota_admission(
        candidate,
        admission_path="incremental",
        paired_evidence=paired,
    )

    candidate["incremental_evidence_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SOTA increment"):
        validate_promoted_factor_sota_admission(
            candidate,
            admission_path="incremental",
            paired_evidence=paired,
        )


def _governed_factor_evaluation(
    tmp_path: Path, *, standalone: bool = False
) -> tuple[dict, dict]:
    candidate = {
        "id": "candidate-a",
        "code_sha256": "5" * 64,
        "values_sha256": "6" * 64,
        "label_horizon_days": 1,
    }
    metrics = {
        "ic": 0.035 if standalone else 0.005,
        "icir": 0.80 if standalone else 0.10,
        "rank_ic": 0.041 if standalone else 0.006,
        "rank_icir": 0.76 if standalone else 0.12,
        "turnover": 0.20,
        "max_correlation": 0.20,
        "cost_adjusted_return": 0.01,
        "selection_days": 300,
        "coverage_pass_rate": 0.99,
        "mean_coverage_ratio": 0.95,
        "constant_day_rate": 0.0,
        "hac_p_value": 0.03 if standalone else 0.40,
        "bh_q_value": 0.04 if standalone else 0.50,
        "raw_valid_ic": 0.01,
        "raw_selection_ic": 0.01,
        "research_profile": {"id": "recent_3y"},
    }
    artifact = tmp_path / "factor-evaluation.json"
    artifact.write_text(
        json.dumps(
            {
                "status": "ok",
                "qlib_workflow": qlib_workflow_identity(),
                "evaluations": [
                    {
                        "candidate_id": candidate["id"],
                        "status": "ok",
                        "metrics": metrics,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    periods = {
        "train_start": date(2015, 1, 1),
        "train_end": date(2020, 12, 31),
        "valid_start": date(2021, 1, 1),
        "valid_end": date(2023, 12, 31),
        "test_start": date(2024, 2, 1),
        "test_end": date(2025, 12, 31),
    }
    submitted_sha256 = "7" * 64
    recompute = {
        "executor_version": "factor-recompute-v4-pit-prefix-invariance",
        "sandbox_mode": "docker-isolated",
        "sandbox_image_id": "sha256:" + "8" * 64,
        "network_mode": "none",
        "root_filesystem_read_only": True,
        "capabilities_dropped": "ALL",
        "no_new_privileges": True,
        "code_sha256": candidate["code_sha256"],
        "dataset_identity_sha256": "9" * 64,
        "authoritative_values_sha256": candidate["values_sha256"],
        "label_horizon_days": 1,
        "provider_input_sha256": "a" * 64,
        "periods": {key: value.isoformat() for key, value in periods.items()},
        "pit_invariance": {
            "contract_version": "factor-pit-prefix-invariance-v1",
            "status": "passed",
            "cutpoint_count": 3,
            "checks": [{"invariant": True}] * 3,
        },
        "research_data_boundary": {
            "latest_input_date": periods["valid_end"].isoformat(),
            "valid_end": periods["valid_end"].isoformat(),
            "test_start": periods["test_start"].isoformat(),
            "final_oos_observations_exposed": False,
        },
        "submitted_comparison": {
            "available": True,
            "exact_match": True,
            "index_exact_match": True,
            "submitted_sha256": submitted_sha256,
        },
    }
    policy = asdict(FactorGatePolicy())
    gate_status, gate_reasons = FactorGatePolicy().evaluate(metrics)
    evaluation = {
        "id": "evaluation-a",
        "factor_candidate_id": candidate["id"],
        "is_legacy": False,
        "dataset": "snapshot-a",
        "dataset_identity_sha256": recompute["dataset_identity_sha256"],
        **periods,
        "metrics_json": metrics,
        "metrics_sha256": canonical_sha256(metrics),
        "policy_json": policy,
        "policy_sha256": canonical_sha256(policy),
        "gate_status": gate_status,
        "gate_reasons_json": gate_reasons,
        "evaluator_version": FactorGatePolicy().version,
        "artifact_path": str(artifact),
        "artifact_sha256": artifact_sha256,
        "candidate_code_sha256": candidate["code_sha256"],
        "candidate_values_sha256": candidate["values_sha256"],
        "submitted_values_sha256": submitted_sha256,
        "recomputed_values_sha256": candidate["values_sha256"],
        "recompute_evidence_json": recompute,
        "signal_frequency": "day",
        "signal_horizon": "1d",
        "execution_frequency": "day",
        "qlib_commit": QLIB_COMMIT,
        "rdagent_commit": RDAGENT_COMMIT,
        "final_test_key": None,
        "final_test_consumed_at": None,
    }
    evidence = {
        "candidate_id": candidate["id"],
        "dataset": evaluation["dataset"],
        "dataset_identity_sha256": evaluation["dataset_identity_sha256"],
        "periods": {key: value.isoformat() for key, value in periods.items()},
        "gate_status": gate_status,
        "gate_reasons": gate_reasons,
        "evaluator_version": evaluation["evaluator_version"],
        "candidate_code_sha256": candidate["code_sha256"],
        "candidate_values_sha256": candidate["values_sha256"],
        "submitted_values_sha256": submitted_sha256,
        "recompute_evidence_sha256": canonical_sha256(recompute),
        "artifact_sha256": artifact_sha256,
        "metrics_sha256": evaluation["metrics_sha256"],
        "policy_sha256": evaluation["policy_sha256"],
    }
    evaluation["evidence_sha256"] = canonical_sha256(evidence)
    evaluation["execution_contract_hash"] = evaluation["evidence_sha256"]
    return evaluation, candidate


def test_strategy_use_revalidates_incremental_hard_pit_gate(tmp_path: Path) -> None:
    evaluation, candidate = _governed_factor_evaluation(tmp_path)
    layers = _validate_governed_factor_evaluation(
        evaluation,
        candidate,
        expected_profile_id="recent_3y",
        require_gate_passed=False,
    )
    assert layers["hard_status"] == "passed"
    assert layers["effect_status"] == "failed"

    evaluation["recompute_evidence_json"]["pit_invariance"]["status"] = "failed"
    with pytest.raises(ValueError, match="strict PIT/recompute"):
        _validate_governed_factor_evaluation(
            evaluation,
            candidate,
            expected_profile_id="recent_3y",
            require_gate_passed=False,
        )


def test_strategy_use_revalidates_standalone_pit_and_effect_gates(
    tmp_path: Path,
) -> None:
    evaluation, candidate = _governed_factor_evaluation(tmp_path, standalone=True)
    layers = _validate_governed_factor_evaluation(
        evaluation,
        candidate,
        expected_profile_id="recent_3y",
        require_gate_passed=True,
    )
    assert layers["hard_status"] == "passed"
    assert layers["effect_status"] == "passed"
