from __future__ import annotations

from typing import Any

import pytest

from quant_platform.research_tournament import (
    FULL_PROFILES,
    FULL_SEEDS,
    build_champion_revalidation_manifest,
    build_quant_preregistered_manifest,
    build_quant_screening_evidence,
    canonical_sha256,
)
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


def _parent_tournament() -> dict[str, Any]:
    return {
        "id": "model-tournament",
        "cycle_id": "cycle-1",
        "stage": "feature_screen",
        "status": "succeeded",
        "dataset_identity_sha256": "d" * 64,
        "manifest_sha256": "a" * 64,
        "multiple_testing_sha256": "b" * 64,
    }


def _baseline() -> dict[str, Any]:
    value = {
        "contract_version": "fin-quant-baseline-prediction-v1",
        "kind": "model",
        "candidate_id": "model-incumbent",
    }
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def _candidate(candidate_id: str, suffix: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "bundle_manifest_sha256": suffix * 64,
        "model_candidate_id": f"model-{candidate_id}",
        "factor_candidate_ids": [f"factor-{candidate_id}"],
        "experiment_family_id": "fin-quant-family",
        "feature_set_id": "qlib-alpha158",
        "feature_set_definition_sha256": "e" * 64,
        "baseline_prediction_champion_sha256": _baseline()["evidence_sha256"],
    }


def _revalidation_component(candidate_id: str, family: str, weight: float) -> dict[str, Any]:
    return {
        "source_model_candidate_id": candidate_id,
        "model_family": family,
        "feature_set_id": "qlib-alpha158",
        "feature_set_definition_sha256": "a" * 64,
        "source_model_manifest_sha256": "b" * 64,
        "source_recipe_sha256": "c" * 64,
        "source_code_sha256": "d" * 64,
        "weight": weight,
    }


def test_current_identity_revalidation_preregisters_only_the_frozen_recipe() -> None:
    manifest = build_champion_revalidation_manifest(
        dataset_identity_sha256="e" * 64,
        source_champion={
            "kind": "model",
            "candidate_id": "old-model",
            "manifest_sha256": "b" * 64,
            "admission_evidence_sha256": "f" * 64,
        },
        source_selection_evidence_sha256="1" * 64,
        source_model_evidence_sha256="2" * 64,
        components=[_revalidation_component("old-model", "lightgbm", 1.0)],
    )

    assert manifest["stage"] == "model_screen"
    assert manifest["trial_count"] == 1
    trial = manifest["trials"][0]
    assert trial["spec"]["profiles"] == list(FULL_PROFILES)
    assert trial["spec"]["seeds"] == list(FULL_SEEDS)
    assert trial["spec"]["fixed_recipe"] is True
    assert trial["spec"]["final_oos_opened"] is False
    assert manifest["research_screening_only"] is True


def test_current_identity_revalidation_rejects_changed_ensemble_weights() -> None:
    with pytest.raises(ValueError, match="equal-rank"):
        build_champion_revalidation_manifest(
            dataset_identity_sha256="e" * 64,
            source_champion={
                "kind": "ensemble",
                "candidate_id": "old-ensemble",
                "manifest_sha256": "b" * 64,
                "admission_evidence_sha256": "f" * 64,
            },
            source_selection_evidence_sha256="1" * 64,
            source_model_evidence_sha256="2" * 64,
            components=[
                _revalidation_component("old-ridge", "ridge", 0.75),
                _revalidation_component("old-lgb", "lightgbm", 0.25),
            ],
        )


def test_fin_quant_preregisters_every_candidate_before_execution() -> None:
    manifest = build_quant_preregistered_manifest(
        parent_tournament=_parent_tournament(),
        dataset_identity_sha256="d" * 64,
        baseline_prediction_champion=_baseline(),
        candidates=[_candidate("bundle-b", "2"), _candidate("bundle-a", "1")],
    )

    assert manifest["stage"] == "quant"
    assert manifest["trial_count"] == 2
    assert [item["candidate_id"] for item in manifest["trials"]] == [
        "bundle-a",
        "bundle-b",
    ]
    assert all(
        item["spec"]["required_ablations"]
        == ["factor_only", "model_only", "joint"]
        for item in manifest["trials"]
    )
    assert manifest["batch_statistics"] == {
        "holm": "independent_evaluator_shared_family",
        "pbo": "independent_evaluator_shared_family",
        "failed_candidate_raw_p_value": 1.0,
        "dsr": "deferred_to_pre_final_portfolio_with_full_prior_trial_count",
    }
    assert manifest["research_screening_only"] is True
    assert manifest["not_capital_confirmation"] is True
    assert manifest["cross_cycle_fwer_claimed"] is False
    assert manifest["final_oos_opened"] is False


def test_fin_quant_accepts_only_a_completed_current_identity_revalidation_parent() -> None:
    parent = _parent_tournament()
    parent["stage"] = "model_screen"
    parent["manifest"] = {
        "contract_version": "champion-current-identity-revalidation-v1"
    }

    manifest = build_quant_preregistered_manifest(
        parent_tournament=parent,
        dataset_identity_sha256="d" * 64,
        baseline_prediction_champion=_baseline(),
        candidates=[_candidate("bundle-a", "1")],
    )

    assert manifest["parent_tournament_id"] == "model-tournament"


def test_fin_quant_screening_ledger_retains_pass_reject_and_runtime_failure() -> None:
    evidence = build_quant_screening_evidence(
        tournament_id="quant-tournament",
        parent_tournament_id="model-tournament",
        dataset_identity_sha256="d" * 64,
        outcomes=[
            {"candidate_id": "passed", "status": "passed", "evidence_sha256": "1" * 64},
            {
                "candidate_id": "rejected",
                "status": "rejected",
                "evidence_sha256": "2" * 64,
            },
            {"candidate_id": "failed", "status": "failed", "evidence_sha256": "3" * 64},
        ],
        run_multiple_testing={"evidence_sha256": "4" * 64},
    )

    assert [item["status"] for item in evidence["outcomes"]] == [
        "failed",
        "passed",
        "rejected",
    ]
    assert evidence["passed_candidate_ids"] == ["passed"]
    assert evidence["failed_and_rejected_trials_retained"] is True
    assert evidence["research_screening_only"] is True
    assert evidence["not_capital_confirmation"] is True
    assert evidence["cross_cycle_fwer_claimed"] is False
    assert evidence["final_oos_opened"] is False
    assert evidence["evidence_sha256"] == canonical_sha256(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )


def test_terminal_quant_job_failure_settles_every_preregistered_trial() -> None:
    class TournamentStore:
        def __init__(self) -> None:
            self.settlement: dict[str, Any] | None = None

        @staticmethod
        def get_tournament(_tournament_id: str) -> dict[str, Any]:
            return {"status": "running"}

        def complete_quant_screening(self, tournament_id: str, **values: Any) -> None:
            self.settlement = {"tournament_id": tournament_id, **values}

    worker = object.__new__(LocalJobWorker)
    worker.research_tournaments = TournamentStore()
    job = {
        "kind": "quant_bundle_evaluate",
        "payload": {
            "research_tournament_id": "quant-tournament",
            "parent_research_tournament_id": "model-tournament",
            "dataset_identity_sha256": "d" * 64,
            "research_trial_ids": {
                "bundle-a": "trial-a",
                "bundle-b": "trial-b",
            },
            "candidates": [{"id": "bundle-a"}, {"id": "bundle-b"}],
        },
    }

    worker._settle_quant_tournament_failure(job, reason="evaluator crashed")

    settlement = worker.research_tournaments.settlement
    assert settlement is not None
    assert settlement["tournament_id"] == "quant-tournament"
    assert {
        (item["candidate_id"], item["status"])
        for item in settlement["outcomes"]
    } == {("bundle-a", "failed"), ("bundle-b", "failed")}
    screening = settlement["screening_evidence"]
    assert screening["statistical_settlement"] == (
        "unavailable_batch_execution_failure_no_candidate_admitted"
    )
    assert screening["not_capital_confirmation"] is True
