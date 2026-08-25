import pytest

from quant_platform.model_research_governance import (
    MODEL_RESEARCH_CONTRACT_VERSION,
    REQUIRED_MODEL_SEEDS,
    REQUIRED_RESEARCH_PROFILES,
    RUN_GROUP_MULTIPLE_TESTING_CONTRACT_VERSION,
    canonical_sha256,
)
from quant_platform.model_strategy_contract import (
    build_model_formal_admission_binding,
    model_signal_identity,
    normalize_model_signal_config,
    validate_model_formal_admission_binding,
)


def test_factor_score_strategy_rejects_hidden_model_fields() -> None:
    with pytest.raises(ValueError, match="cannot bind"):
        normalize_model_signal_config(
            {"signal_source": "factor_score", "model_candidate_id": "candidate"}
        )


def test_model_strategy_requires_all_immutable_bindings() -> None:
    config = {
        "signal_source": "model_prediction",
        "model_candidate_id": "model-1",
        "model_evaluation_id": "evaluation-1",
        "model_code_sha256": "a" * 64,
        "model_recipe_sha256": "b" * 64,
        "model_evidence_sha256": "c" * 64,
        "feature_set_id": "governed-baseline",
        "feature_set_definition_sha256": "d" * 64,
    }
    identity = model_signal_identity(config)
    assert identity is not None
    assert len(identity["identity_sha256"]) == 64


def test_formal_admission_binds_one_execution_environment() -> None:
    environment_sha256 = "e" * 64
    metrics = {
        "ic": 0.03,
        "icir": 0.70,
        "rank_ic": 0.04,
        "rank_icir": 0.80,
        "information_ratio": 0.80,
        "annualized_excess_return_with_cost": 0.08,
        "max_drawdown": -0.12,
        "total_cost": 0.01,
        "average_turnover": 0.15,
    }
    periods = {
        "train_start": "2016-01-04",
        "train_end": "2022-12-30",
        "valid_start": "2023-01-03",
        "valid_end": "2025-12-31",
        "test_start": "2026-01-08",
        "test_end": "2026-12-31",
    }
    evidence = {
        "contract_version": MODEL_RESEARCH_CONTRACT_VERSION,
        "source": "independent_qlib_recompute",
        "candidate_id": "model-1",
        "dataset_identity_sha256": "a" * 64,
        "execution_environment_sha256": environment_sha256,
        "multiple_testing": {
            "contract_version": RUN_GROUP_MULTIPLE_TESTING_CONTRACT_VERSION,
            "source": "independent_qlib_recompute",
            "research_run_id": "run-1",
            "profile_id": "recent_3y",
            "seed_aggregation": "equal_mean_fixed_seeds",
            "seeds": list(REQUIRED_MODEL_SEEDS),
            "trial_definitions": [
                {"name": "model-1", "candidate_id": "model-1", "kind": "model"}
            ],
            "trial_names": ["model-1"],
            "trial_count": 1,
            "final_oos_opened": False,
            "raw_p_values": [0.01],
            "holm_adjusted_p_values": [0.01],
            "maximum_adjusted_p_value": 0.05,
            "eligible_trial_names": ["model-1"],
            "pbo": {
                "status": "not_applicable_single_trial",
                "pbo": None,
                "trials": 1,
                "observations": 120,
            },
            "maximum_pbo": 0.50,
            "trial_daily_sharpes": [0.10],
            "returns_path": "run-multiple-testing.parquet",
            "returns_sha256": "9" * 64,
            "observations": 120,
            "gate_passed": True,
        },
        "final_oos_opened": False,
        "profiles": {
            profile: {
                "periods": periods,
                "seeds": {
                    str(seed): {
                        "status": "passed",
                        "metrics": metrics,
                        "latest_prediction_date": periods["valid_end"],
                        "predictions_sha256": "b" * 64,
                        "portfolio_report_path": "portfolio.parquet",
                        "portfolio_report_sha256": "9" * 64,
                        "execution_evidence_sha256": "c" * 64,
                        "execution_environment_sha256": environment_sha256,
                    }
                    for seed in REQUIRED_MODEL_SEEDS
                },
            }
            for profile in REQUIRED_RESEARCH_PROFILES
        },
    }
    evidence["multiple_testing"]["evidence_sha256"] = canonical_sha256(
        evidence["multiple_testing"]
    )
    evidence_sha256 = canonical_sha256(evidence)
    config = {
        "signal_source": "model_prediction",
        "model_candidate_id": "model-1",
        "model_evaluation_id": "evaluation-1",
        "model_code_sha256": "d" * 64,
        "model_recipe_sha256": "f" * 64,
        "model_evidence_sha256": evidence_sha256,
        "feature_set_id": "governed-baseline",
        "feature_set_definition_sha256": "1" * 64,
    }
    binding = build_model_formal_admission_binding(
        config=config,
        candidate_manifest_sha256="2" * 64,
        dataset_identity_sha256="a" * 64,
        pre_final_end=periods["valid_end"],
        model_admission_evidence=evidence,
        model_admission_evidence_sha256=evidence_sha256,
    )

    assert binding["model_grid"]["execution_environment_sha256"] == (
        environment_sha256
    )
    assert validate_model_formal_admission_binding(
        binding,
        config=config,
        dataset_identity_sha256="a" * 64,
        pre_final_end=periods["valid_end"],
    ) == binding
