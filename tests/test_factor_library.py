import pytest

from quant_platform.factor_library import (
    FACTOR_DEFINITIONS,
    compile_qlib_expression,
    feature_expression_map,
    library_release_definition,
)
from quant_platform.factor_library_store import (
    INCREMENTAL_EVIDENCE_VERSION,
    ResearchSotaPolicy,
    relationship_for_correlation,
    validate_incremental_evidence,
    validate_sota_members,
)
from quant_platform.research_automation import allocate_governed_factor_weights
from quant_platform.research_store import FactorGatePolicy

pytestmark = pytest.mark.no_database


def _incremental_evidence() -> dict:
    prediction = {
        "baseline_prediction_sha256": "1" * 64,
        "proposed_prediction_sha256": "2" * 64,
        "paired_index_sha256": "3" * 64,
        "final_oos_observations_exposed": False,
    }
    common = {
        "evaluation_evidence_sha256": "7" * 64,
        "baseline_rank_ic": 0.02,
        "proposed_rank_ic": 0.03,
        "baseline_cost_adjusted_return": 0.01,
        "proposed_cost_adjusted_return": 0.02,
        "hard_gate_status": "passed",
        "stability_gate_status": "passed",
        "prediction_evidence": prediction,
        "paired_rank_ic_hac": {"status": "ok", "mean": 0.01, "p_value": 0.01},
        "paired_cost_return_bootstrap": {
            "status": "ok",
            "confidence_interval_95": [0.001, 0.02],
            "one_sided_p_value": 0.01,
        },
    }
    return {
        "version": INCREMENTAL_EVIDENCE_VERSION,
        "frozen_model": {
            "kind": "lightgbm_baseline",
            "model_artifact_id": "model-a",
            "model_artifact_sha256": "4" * 64,
            "training_recipe_sha256": "5" * 64,
            "feature_contract_sha256": "6" * 64,
        },
        "evaluation_ids": {
            "recent_3y": "1" * 32,
            "balanced_5y": "2" * 32,
            "robust_10y": "3" * 32,
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
            "experiment_family_id": "factor-family-a",
            "hypothesis_count": 4,
            "rank_ic_q_value": 0.04,
            "cost_return_q_value": 0.05,
        },
        "profiles": {
            "recent_3y": {
                **common,
                "delta_rank_ic": 0.01,
                "delta_cost_adjusted_return": 0.01,
            },
            "balanced_5y": {
                **common,
                "delta_rank_ic": 0.0,
                "delta_cost_adjusted_return": 0.0,
            },
            "robust_10y": {
                **common,
                "delta_rank_ic": 0.0,
                "delta_cost_adjusted_return": 0.0,
            },
        },
    }


def test_pinned_sources_and_seed_counts_are_exact() -> None:
    release = library_release_definition()
    assert release["source_alias_counts"] == {
        "alpha158": 158,
        "alpha360": 360,
        "alpha20": 20,
        "platform_seed": 24,
    }
    assert release["definition_sha256"] == (
        "11faa8074efebc1d0b57b50c0e5715831f4e40f9812e6d6af23535051f8a14a1"
    )
    assert len(feature_expression_map("alpha158")) == 158
    assert len(feature_expression_map("alpha360")) == 360
    assert len(feature_expression_map("alpha20")) == 20
    assert len(feature_expression_map("platform_seed")) == 24
    assert len(FACTOR_DEFINITIONS) < 158 + 360 + 24
    seed = {item.name: item for item in FACTOR_DEFINITIONS if "platform_seed" in item.family_tags}
    assert seed["kdj_j_9"].max_lookback_days == 12
    assert seed["max_drawdown_60"].max_lookback_days == 118


def test_expression_compiler_rejects_future_and_unknown_capabilities() -> None:
    compiled = compile_qlib_expression("Mean($close/Ref($close,1)-1,20)")
    assert compiled.required_fields == ("close",)
    assert compiled.max_lookback_days == 20
    assert compile_qlib_expression("Ref(Mean($close,5),2)").max_lookback_days == 6
    with pytest.raises(ValueError, match="future"):
        compile_qlib_expression("Ref($close,-1)/$close")
    with pytest.raises(ValueError, match="unsupported functions"):
        compile_qlib_expression("Python($close,20)")
    with pytest.raises(ValueError, match="unavailable fields"):
        compile_qlib_expression("Mean($secret_field,20)")


def test_expression_compiler_allows_pinned_qlib_conditional_operator() -> None:
    compiled = compile_qlib_expression(
        "If(Greater($high-$low,Abs($high-Ref($close,1))),$high-$low,0)"
    )

    assert compiled.functions == ("Abs", "Greater", "If", "Ref")
    assert compiled.max_lookback_days == 1
    assert compiled.to_dict()["contract_version"] == "qlib-expression-allowlist-v2"


def test_duplicate_and_cluster_thresholds_are_distinct() -> None:
    assert relationship_for_correlation(0.75) == "independent"
    assert relationship_for_correlation(0.75001) == "clustered"
    assert relationship_for_correlation(0.95) == "near_duplicate"


def test_sota_family_and_weight_limits_fail_closed() -> None:
    policy = ResearchSotaPolicy()
    evidence = _incremental_evidence()
    members = [
        {
            "factor_candidate_id": f"factor-{index}",
            "similarity_cluster_id": f"cluster-{index}",
            "economic_family": "trend",
            "incremental_evidence": evidence,
        }
        for index in range(4)
    ]
    with pytest.raises(ValueError, match="economic-family"):
        validate_sota_members(members, policy)


def test_factor_gate_separates_hard_failures_from_weak_effects() -> None:
    policy = FactorGatePolicy()
    metrics = {
        "ic": 0.005,
        "icir": 0.10,
        "rank_ic": 0.006,
        "rank_icir": 0.12,
        "turnover": 0.20,
        "max_correlation": 0.20,
        "cost_adjusted_return": 0.01,
        "selection_days": 300,
        "coverage_pass_rate": 0.99,
        "mean_coverage_ratio": 0.95,
        "constant_day_rate": 0.0,
        "hac_p_value": 0.40,
        "bh_q_value": 0.50,
        "raw_valid_ic": 0.01,
        "raw_selection_ic": 0.01,
    }
    layers = policy.evaluate_layers(metrics)
    assert layers["hard_status"] == "passed"
    assert layers["effect_status"] == "failed"

    metrics["coverage_pass_rate"] = 0.50
    assert policy.evaluate_layers(metrics)["hard_status"] == "failed"


def test_incremental_evidence_rejects_linear_proxy_and_point_estimates() -> None:
    evidence = _incremental_evidence()
    evidence["frozen_model"]["kind"] = "fixed_cross_sectional_zscore_linear"
    with pytest.raises(ValueError, match="linear score proxy"):
        validate_incremental_evidence(evidence)

    # Malformed/missing statistical evidence still fails closed; only the
    # significance *thresholds* moved to report-only.
    evidence = _incremental_evidence()
    evidence["profiles"]["recent_3y"]["paired_cost_return_bootstrap"]["status"] = "failed"
    with pytest.raises(ValueError, match="block bootstrap"):
        validate_incremental_evidence(evidence)

    evidence = _incremental_evidence()
    evidence["window_role_policy"]["nested_windows_count_as_independent"] = True
    with pytest.raises(ValueError, match="must not be counted as independent"):
        validate_incremental_evidence(evidence)


def test_incremental_evidence_archives_insignificant_statistics_without_veto() -> None:
    # Wide-in, strict-out: q-values above 0.10, an insignificant paired HAC
    # test and a bootstrap interval crossing zero are sealed report-only and
    # no longer veto the increment.
    evidence = _incremental_evidence()
    evidence["multiplicity"]["rank_ic_q_value"] = 0.50
    evidence["multiplicity"]["cost_return_q_value"] = 0.60
    evidence["multiplicity"]["statistical_evidence_role"] = "report_only"
    evidence["profiles"]["recent_3y"]["paired_rank_ic_hac"] = {
        "status": "ok",
        "mean": 0.01,
        "p_value": 0.40,
    }
    evidence["profiles"]["recent_3y"]["paired_cost_return_bootstrap"] = {
        "status": "ok",
        "confidence_interval_95": [-0.001, 0.02],
        "one_sided_p_value": 0.30,
    }
    validate_incremental_evidence(evidence)


def test_explicit_weighting_blocks_an_infeasible_family_mix() -> None:
    infeasible = [
        {
            "id": f"factor-{index}",
            "economic_family": "trend",
            "automation_score": float(10 - index),
        }
        for index in range(4)
    ]
    with pytest.raises(ValueError, match="cannot satisfy"):
        allocate_governed_factor_weights(infeasible)
    feasible = [
        {
            "id": f"factor-{index}",
            "economic_family": family,
            "automation_score": float(10 - index),
        }
        for index, family in enumerate(
            ("trend", "value", "quality", "growth", "liquidity")
        )
    ]
    weights = allocate_governed_factor_weights(feasible)
    assert sum(weights) == pytest.approx(1.0)
    assert max(weights) <= 0.25
