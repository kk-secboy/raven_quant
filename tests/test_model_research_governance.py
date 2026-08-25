from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_platform.model_research_governance import (
    MODEL_RESEARCH_CONTRACT_VERSION,
    QUANT_BUNDLE_CONTRACT_VERSION,
    RUN_GROUP_MULTIPLE_TESTING_CONTRACT_VERSION,
    canonical_sha256,
    file_sha256,
    validate_independent_model_evidence,
    validate_quant_bundle_evidence,
    verify_model_prediction_artifact,
)


def _multiple_testing(*trial_definitions: dict[str, str]) -> dict:
    names = [item["name"] for item in trial_definitions]
    value = {
        "contract_version": RUN_GROUP_MULTIPLE_TESTING_CONTRACT_VERSION,
        "source": "independent_qlib_recompute",
        "research_run_id": "run-1",
        "profile_id": "recent_3y",
        "seed_aggregation": "equal_mean_fixed_seeds",
        "seeds": [11, 29, 47],
        "trial_definitions": list(trial_definitions),
        "trial_names": names,
        "trial_count": len(names),
        "final_oos_opened": False,
        "raw_p_values": [0.01] * len(names),
        "holm_adjusted_p_values": [0.02] * len(names),
        "maximum_adjusted_p_value": 0.05,
        "eligible_trial_names": names,
        "pbo": (
            {
                "status": "not_applicable_single_trial",
                "pbo": None,
                "trials": 1,
                "observations": 120,
            }
            if len(names) == 1
            else {"status": "ok", "pbo": 0.25}
        ),
        "maximum_pbo": 0.50,
        "trial_daily_sharpes": [0.10] * len(names),
        "returns_path": "run-multiple-testing.parquet",
        "returns_sha256": "f" * 64,
        "observations": 120,
        "gate_passed": True,
    }
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def _metrics() -> dict[str, float]:
    return {
        "ic": 0.03,
        "icir": 0.70,
        "rank_ic": 0.04,
        "rank_icir": 0.80,
        "information_ratio": 0.8,
        "annualized_excess_return_with_cost": 0.08,
        "max_drawdown": -0.12,
        "total_cost": 0.01,
        "average_turnover": 0.15,
    }


def _model_evidence() -> dict:
    profiles = {}
    for name in ("recent_3y", "balanced_5y", "robust_10y"):
        profiles[name] = {
            "periods": {
                "train_start": "2010-01-04",
                "train_end": "2019-12-31",
                "valid_start": "2020-01-02",
                "valid_end": "2025-06-30",
                "test_start": "2025-07-08",
                "test_end": "2026-06-30",
            },
            "seeds": {
                str(seed): {
                    "status": "passed",
                    "metrics": _metrics(),
                    "predictions_sha256": "a" * 64,
                    "portfolio_report_path": "portfolio.parquet",
                    "portfolio_report_sha256": "d" * 64,
                    "execution_evidence_sha256": "b" * 64,
                    "execution_environment_sha256": "e" * 64,
                    "latest_prediction_date": "2025-06-30",
                }
                for seed in (11, 29, 47)
            },
        }
    return {
        "contract_version": MODEL_RESEARCH_CONTRACT_VERSION,
        "source": "independent_qlib_recompute",
        "candidate_id": "candidate-1",
        "dataset_identity_sha256": "c" * 64,
        "execution_environment_sha256": "e" * 64,
        "multiple_testing": _multiple_testing(
            {"name": "candidate-1", "candidate_id": "candidate-1", "kind": "model"}
        ),
        "final_oos_opened": False,
        "profiles": profiles,
    }


def test_model_evidence_requires_independent_three_seed_three_profile_proof() -> None:
    evidence = _model_evidence()
    result = validate_independent_model_evidence(
        evidence,
        candidate_id="candidate-1",
        dataset_identity_sha256="c" * 64,
        pre_final_end="2025-12-31",
    )
    assert len(result["evidence_sha256"]) == 64
    evidence["source"] = "rdagent_internal"
    with pytest.raises(ValueError, match="internal"):
        validate_independent_model_evidence(
            evidence,
            candidate_id="candidate-1",
            dataset_identity_sha256="c" * 64,
            pre_final_end="2025-12-31",
        )


def test_model_evidence_rejects_executable_but_weak_candidate() -> None:
    evidence = _model_evidence()
    evidence["profiles"]["robust_10y"]["seeds"]["29"]["metrics"]["rank_ic"] = 0.0
    with pytest.raises(ValueError, match="model-metric-gate-v1"):
        validate_independent_model_evidence(
            evidence,
            candidate_id="candidate-1",
            dataset_identity_sha256="c" * 64,
            pre_final_end="2025-12-31",
        )


def test_prediction_artifact_requires_daily_cross_sectional_coverage(tmp_path: Path) -> None:
    days = pd.bdate_range("2025-01-02", periods=3)
    index = pd.MultiIndex.from_product(
        [days, [f"{index:06d}.SZ" for index in range(60)]],
        names=["datetime", "instrument"],
    )
    path = tmp_path / "predictions.parquet"
    pd.DataFrame({"score": range(len(index))}, index=index).to_parquet(path)
    result = verify_model_prediction_artifact(
        path,
        expected_sha256=file_sha256(path),
        test_start=str(days[0].date()),
        test_end=str(days[-1].date()),
        trading_days=days,
    )
    assert result["coverage_gate_passed"] is True


def test_prediction_artifact_rejects_rows_outside_frozen_oos(tmp_path: Path) -> None:
    days = pd.bdate_range("2025-01-02", periods=3)
    artifact_days = days.append(pd.DatetimeIndex([days[-1] + pd.offsets.BDay(1)]))
    index = pd.MultiIndex.from_product(
        [artifact_days, [f"{value:06d}.SZ" for value in range(60)]],
        names=["datetime", "instrument"],
    )
    path = tmp_path / "predictions.parquet"
    pd.DataFrame({"score": range(len(index))}, index=index).to_parquet(path)

    with pytest.raises(ValueError, match="outside the exact OOS window"):
        verify_model_prediction_artifact(
            path,
            expected_sha256=file_sha256(path),
            test_start=str(days[0].date()),
            test_end=str(days[-1].date()),
            trading_days=days,
        )


def test_quant_bundle_rejects_component_mutation() -> None:
    def ablation() -> dict:
        value = {
            "status": "passed",
            "source": "independent_qlib_recompute",
            "experiment_family_id": "family-1",
            "dataset_identity_sha256": "c" * 64,
            "execution_environment_sha256": "e" * 64,
            "final_oos_opened": False,
            "profiles": {
                profile: {
                    "periods": {"valid_end": "2025-06-30"},
                    "seeds": {
                        str(seed): {
                            "status": "passed",
                            "latest_prediction_date": "2025-06-30",
                            "metrics": _metrics(),
                            "predictions_sha256": "a" * 64,
                            "portfolio_report_path": "portfolio.parquet",
                            "portfolio_report_sha256": "f" * 64,
                            "execution_evidence_sha256": "b" * 64,
                            "execution_environment_sha256": "e" * 64,
                        }
                        for seed in (11, 29, 47)
                    },
                }
                for profile in ("recent_3y", "balanced_5y", "robust_10y")
            },
        }
        value["evidence_sha256"] = canonical_sha256(value)
        return value

    bundle = {
        "contract_version": QUANT_BUNDLE_CONTRACT_VERSION,
        "id": "bundle-1",
        "dataset_identity_sha256": "c" * 64,
        "experiment_family_id": "family-1",
        "factors": [{"candidate_id": "factor-1", "code_sha256": "a" * 64}],
        "model": {"code_sha256": "b" * 64, "recipe_sha256": "d" * 64},
        "execution_environment_sha256": "e" * 64,
        "ablations": {name: ablation() for name in ("factor_only", "model_only", "joint")},
        "multiple_testing": _multiple_testing(
            *(
                {
                    "name": f"bundle-1:{name}",
                    "candidate_id": "bundle-1",
                    "kind": "quant_bundle",
                    "ablation": name,
                }
                for name in ("factor_only", "model_only", "joint")
            )
        ),
    }
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    validate_quant_bundle_evidence(bundle, dataset_identity_sha256="c" * 64)
    bundle["model"]["recipe_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="changed"):
        validate_quant_bundle_evidence(bundle, dataset_identity_sha256="c" * 64)


def test_quant_bundle_gates_joint_but_keeps_ablations_diagnostic() -> None:
    def ablation(*, rank_ic: float) -> dict:
        value = {
            "status": "passed",
            "source": "independent_qlib_recompute",
            "experiment_family_id": "family-1",
            "dataset_identity_sha256": "c" * 64,
            "execution_environment_sha256": "e" * 64,
            "final_oos_opened": False,
            "profiles": {
                profile: {
                    "periods": {"valid_end": "2025-06-30"},
                    "seeds": {
                        str(seed): {
                            "status": "passed",
                            "latest_prediction_date": "2025-06-30",
                            "metrics": {**_metrics(), "rank_ic": rank_ic},
                            "predictions_sha256": "a" * 64,
                            "portfolio_report_path": "portfolio.parquet",
                            "portfolio_report_sha256": "f" * 64,
                            "execution_evidence_sha256": "b" * 64,
                            "execution_environment_sha256": "e" * 64,
                        }
                        for seed in (11, 29, 47)
                    },
                }
                for profile in ("recent_3y", "balanced_5y", "robust_10y")
            },
        }
        value["evidence_sha256"] = canonical_sha256(value)
        return value

    bundle = {
        "contract_version": QUANT_BUNDLE_CONTRACT_VERSION,
        "id": "bundle-1",
        "dataset_identity_sha256": "c" * 64,
        "experiment_family_id": "family-1",
        "factors": [{"candidate_id": "factor-1", "code_sha256": "a" * 64}],
        "model": {"code_sha256": "b" * 64, "recipe_sha256": "d" * 64},
        "execution_environment_sha256": "e" * 64,
        "ablations": {
            "factor_only": ablation(rank_ic=-0.02),
            "model_only": ablation(rank_ic=0.01),
            "joint": ablation(rank_ic=0.04),
        },
        "multiple_testing": _multiple_testing(
            *(
                {
                    "name": f"bundle-1:{name}",
                    "candidate_id": "bundle-1",
                    "kind": "quant_bundle",
                    "ablation": name,
                }
                for name in ("factor_only", "model_only", "joint")
            )
        ),
    }
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    validate_quant_bundle_evidence(bundle, dataset_identity_sha256="c" * 64)
    bundle["ablations"]["joint"] = ablation(rank_ic=0.0)
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    with pytest.raises(ValueError, match="model-metric-gate-v1"):
        validate_quant_bundle_evidence(bundle, dataset_identity_sha256="c" * 64)
