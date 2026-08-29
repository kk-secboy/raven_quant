from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import CheckConstraint

from quant_data.database import (
    candidate_asset_links,
    metadata,
    model_candidates,
    model_evaluations,
    quant_bundle_candidates,
    quant_bundle_evaluations,
    research_assets,
    research_run_artifacts,
)
from quant_data.research_assets import register_local_research_asset
from quant_platform.model_research_governance import (
    RUN_GROUP_MULTIPLE_TESTING_CONTRACT_VERSION,
    canonical_sha256,
)
from quant_platform.rdagent_candidate_store import (
    REQUIRED_MODEL_SEEDS,
    REQUIRED_QUANT_ABLATIONS,
    REQUIRED_RESEARCH_PROFILES,
    RDAGentCandidateStore,
    _bundle_asset_evidence,
    validate_model_evaluation_evidence,
    validate_quant_bundle_evidence,
)


def _metrics() -> dict[str, float]:
    return {
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


def _multiple_testing(*trial_definitions: dict[str, str]) -> dict:
    names = [item["name"] for item in trial_definitions]
    value = {
        "contract_version": RUN_GROUP_MULTIPLE_TESTING_CONTRACT_VERSION,
        "source": "independent_qlib_recompute",
        "research_run_id": "run-1",
        "profile_id": "recent_3y",
        "seed_aggregation": "equal_mean_fixed_seeds",
        "seeds": list(REQUIRED_MODEL_SEEDS),
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


def _model_evidence(*, source: str = "independent_qlib_recompute") -> dict:
    periods = {
        "train_start": "2015-01-05",
        "train_end": "2022-12-30",
        "valid_start": "2023-01-03",
        "valid_end": "2025-12-31",
        "test_start": "2026-01-05",
        "test_end": "2026-12-31",
    }
    return {
        "contract_version": "model-research-independent-v1",
        "source": source,
        "candidate_id": "model-1",
        "dataset_identity_sha256": "a" * 64,
        "feature_set_definition_sha256": "b" * 64,
        "execution_environment_sha256": "e" * 64,
        "final_oos_opened": False,
        "profiles": {
            profile: {
                "periods": periods,
                "seeds": {
                    str(seed): {
                        "status": "passed",
                        "metrics": _metrics(),
                        "latest_prediction_date": "2025-12-31",
                        "predictions_sha256": "c" * 64,
                        "portfolio_report_path": "portfolio.parquet",
                        "portfolio_report_sha256": "f" * 64,
                        "execution_evidence_sha256": "d" * 64,
                        "execution_environment_sha256": "e" * 64,
                    }
                    for seed in REQUIRED_MODEL_SEEDS
                },
            }
            for profile in REQUIRED_RESEARCH_PROFILES
        },
        "multiple_testing": _multiple_testing(
            {"name": "model-1", "candidate_id": "model-1", "kind": "model"}
        ),
    }


def test_candidate_schema_is_additive_immutable_and_non_capital() -> None:
    assert {
        research_assets.name,
        research_run_artifacts.name,
        model_candidates.name,
        model_evaluations.name,
        quant_bundle_candidates.name,
        quant_bundle_evaluations.name,
        candidate_asset_links.name,
    } <= {table.name for table in metadata.tables.values()}
    assert "manifest_sha256" in model_candidates.c
    assert "feature_set_definition_sha256" in model_candidates.c
    assert "bundle_manifest_sha256" in quant_bundle_candidates.c
    assert "feature_set_definition_sha256" in quant_bundle_candidates.c
    assert "model_ensemble_candidate_id" in quant_bundle_candidates.c
    model_checks = {
        str(constraint.sqltext)
        for constraint in model_candidates.constraints
        if isinstance(constraint, CheckConstraint)
    }
    quant_checks = {
        str(constraint.sqltext)
        for constraint in quant_bundle_candidates.constraints
        if isinstance(constraint, CheckConstraint)
    }
    evaluation_checks = {
        str(constraint.sqltext)
        for constraint in model_evaluations.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert "capital_eligible = false" in model_checks
    assert "capital_eligible = false" in quant_checks
    assert any(
        "model_candidate_id IS NOT NULL"
        in value
        and "model_ensemble_candidate_id IS NOT NULL" in value
        for value in quant_checks
    )
    assert "oos_vintage_id IS NULL" in evaluation_checks


def test_model_admission_validator_requires_fixed_independent_grid() -> None:
    evidence = _model_evidence()
    validated = validate_model_evaluation_evidence(
        evidence,
        candidate_id="model-1",
        dataset_identity_sha256="a" * 64,
        pre_final_end=date(2025, 12, 31),
    )
    assert len(validated["evidence_sha256"]) == 64
    official = _model_evidence(source="rdagent_internal")
    with pytest.raises(ValueError, match="internal"):
        validate_model_evaluation_evidence(
            official,
            candidate_id="model-1",
            dataset_identity_sha256="a" * 64,
            pre_final_end=date(2025, 12, 31),
        )


@pytest.mark.no_database
def test_quant_bundle_validator_requires_all_three_ablations() -> None:
    periods = {
        "train_start": "2015-01-05",
        "train_end": "2022-12-30",
        "valid_start": "2023-01-03",
        "valid_end": "2025-12-31",
        "test_start": "2026-01-05",
        "test_end": "2026-12-31",
    }

    def ablation_evidence(ablation: str) -> dict:
        value = {
            "status": "passed",
            "source": "independent_qlib_recompute",
            "experiment_family_id": "family-1",
            "dataset_identity_sha256": "a" * 64,
            "execution_environment_sha256": "e" * 64,
            "final_oos_opened": False,
            "profiles": {
                profile: {
                    "periods": periods,
                    "seeds": {
                        str(seed): {
                            "status": "passed",
                            "latest_prediction_date": periods["valid_end"],
                            "metrics": _metrics(),
                            "predictions_sha256": "c" * 64,
                            "portfolio_report_path": "portfolio.parquet",
                            "portfolio_report_sha256": "f" * 64,
                            "execution_evidence_sha256": "d" * 64,
                            "execution_environment_sha256": "e" * 64,
                        }
                        for seed in REQUIRED_MODEL_SEEDS
                    },
                }
                for profile in REQUIRED_RESEARCH_PROFILES
            },
        }
        value["evidence_sha256"] = canonical_sha256(value)
        return value

    bundle = {
        "contract_version": "quant-bundle-ablation-v1",
        "id": "bundle-1",
        "dataset_identity_sha256": "a" * 64,
        "experiment_family_id": "family-1",
        "factors": [{"candidate_id": "factor-1", "code_sha256": "b" * 64}],
        "model": {"code_sha256": "c" * 64, "recipe_sha256": "d" * 64},
        "execution_environment_sha256": "e" * 64,
        "ablations": {
            ablation: ablation_evidence(ablation)
            for ablation in REQUIRED_QUANT_ABLATIONS
        },
        "multiple_testing": _multiple_testing(
            *(
                {
                    "name": f"bundle-1:{ablation}",
                    "candidate_id": "bundle-1",
                    "kind": "quant_bundle",
                    "ablation": ablation,
                }
                for ablation in REQUIRED_QUANT_ABLATIONS
            )
        ),
        "final_oos_opened": False,
    }
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    validate_quant_bundle_evidence(bundle, dataset_identity_sha256="a" * 64)
    del bundle["ablations"]["model_only"]
    with pytest.raises(ValueError, match="requires"):
        validate_quant_bundle_evidence(bundle, dataset_identity_sha256="a" * 64)


def test_research_candidates_fail_closed_for_capital() -> None:
    with pytest.raises(ValueError, match="research-only"):
        RDAGentCandidateStore.require_capital_admission("model", "model-1")


def test_bundle_inventory_rechecks_every_file_and_rejects_extras(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    published = register_local_research_asset(
        tmp_path / "data",
        asset_id="dataset-one",
        kind="dataset",
        source_path=source,
        asset_type="competition_dataset",
        clock=lambda: datetime(2026, 8, 14, tzinfo=UTC),
    )
    inventory, digest, size = _bundle_asset_evidence(published.manifest_path)
    assert inventory["kind"] == "dataset"
    assert len(digest) == 64 and size > 0

    (published.directory / "extra.txt").write_text("not sealed", encoding="utf-8")
    with pytest.raises(ValueError, match="unsealed"):
        _bundle_asset_evidence(published.manifest_path)
