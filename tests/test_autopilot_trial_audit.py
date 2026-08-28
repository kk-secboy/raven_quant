from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_platform.autopilot_trial_audit import AutopilotTrialAuditService

pytestmark = pytest.mark.no_database


NOW = datetime(2026, 8, 26, tzinfo=UTC)


class _Research:
    def list_candidates(self, *, run_id: str, limit: int) -> list[dict]:
        assert limit == 500
        if run_id != "factor-run":
            return []
        return [
            {
                "id": "factor-1",
                "research_run_id": run_id,
                "name": "volume-price-1",
                "status": "promoted",
                "factor_definition_id": "definition-1",
                "formulation": "Ref($close, 1) / $close - 1",
                "code_sha256": "a" * 64,
                "values_sha256": "b" * 64,
                "experiment_family_id": "family-1",
                "economic_family": "momentum",
                "similarity_cluster_id": "cluster-1",
                "profile_consensus": {"status": "passed"},
                "profile_consensus_sha256": "c" * 64,
                "admission_path": "incremental",
                "incremental_evidence": {"status": "passed"},
                "incremental_evidence_sha256": "d" * 64,
                "promotion_evidence_sha256": "e" * 64,
                "rdagent_decision": True,
                "rdagent_feedback": "candidate",
                "created_at": NOW,
                "updated_at": NOW,
            }
        ]

    def list_evaluations(self, candidate_id: str, *, limit: int) -> list[dict]:
        assert candidate_id == "factor-1"
        assert limit == 500
        return [
            {
                "id": "factor-evaluation-1",
                "factor_candidate_id": candidate_id,
                "gate_status": "passed",
                "artifact_sha256": "f" * 64,
            }
        ]


class _Candidates:
    def run_audit_summary(self, run_id: str) -> dict:
        if run_id == "missing-run":
            raise KeyError(run_id)
        if run_id != "quant-run":
            return {
                "model_candidates": [],
                "model_evaluations": [],
                "quant_bundle_candidates": [],
                "quant_bundle_evaluations": [],
            }
        return {
            "model_candidates": [
                {
                    "id": "model-1",
                    "name": "ridge-1",
                    "status": "research_admitted",
                    "model_type": "ridge",
                    "code_sha256": "1" * 64,
                    "feature_set_definition_sha256": "2" * 64,
                    "dataset": "daily-v1",
                    "dataset_identity_sha256": "3" * 64,
                    "manifest_sha256": "4" * 64,
                    "admission_evidence_sha256": "5" * 64,
                    "rdagent_decision": True,
                    "rdagent_feedback": "accepted",
                    "created_at": NOW,
                    "updated_at": NOW,
                }
            ],
            "model_evaluations": [
                {
                    "id": "model-evaluation-1",
                    "model_candidate_id": "model-1",
                    "profile_id": "recent_3y",
                    "seed": 11,
                    "gate_status": "passed",
                    "evidence_sha256": "6" * 64,
                }
            ],
            "quant_bundle_candidates": [
                {
                    "id": "bundle-1",
                    "name": "joint-1",
                    "status": "rejected",
                    "prediction_component_kind": "model",
                    "model_candidate_id": "model-1",
                    "model_ensemble_candidate_id": None,
                    "factor_candidate_ids": ["factor-1"],
                    "dataset": "daily-v1",
                    "dataset_identity_sha256": "3" * 64,
                    "bundle_manifest_sha256": "7" * 64,
                    "admission_evidence_sha256": None,
                    "rdagent_decision": True,
                    "rdagent_feedback": "try",
                    "created_at": NOW,
                    "updated_at": NOW,
                }
            ],
            "quant_bundle_evaluations": [
                {
                    "id": "bundle-evaluation-1",
                    "quant_bundle_candidate_id": "bundle-1",
                    "ablation": "joint",
                    "profile_id": "recent_3y",
                    "seed": 11,
                    "gate_status": "failed",
                    "evidence_sha256": "8" * 64,
                }
            ],
        }

    def get_model_candidate(self, candidate_id: str) -> dict:
        assert candidate_id == "model-1"
        return {"admission_evidence_json": {"status": "passed"}}

    def get_quant_bundle_candidate(self, candidate_id: str) -> dict:
        assert candidate_id == "bundle-1"
        return {
            "ablation_evidence_json": {"joint": {"status": "failed"}},
            "ablation_evidence_sha256": "9" * 64,
            "admission_evidence_json": None,
            "rejection_reason": "joint challenger did not beat incumbent",
        }


class _Tournaments:
    def __init__(self, *, missing: bool = False) -> None:
        self.missing = missing

    def get_for_cycle(self, cycle_id: str) -> dict:
        if self.missing:
            raise KeyError(cycle_id)
        return {
            "id": "tournament-1",
            "stage": "ensemble",
            "status": "running",
            "manifest_sha256": "0" * 64,
            "selected_trial_ids": [],
            "multiple_testing": {"status": "pending"},
            "multiple_testing_sha256": "a" * 64,
            "trials": [
                {
                    "id": "trial-model-1",
                    "trial_kind": "model",
                    "name": "ridge:alpha158",
                    "status": "passed",
                    "candidate_id": "model-1",
                    "feature_set_id": "qlib-alpha158",
                    "feature_set_definition_sha256": "2" * 64,
                    "model_family": "ridge",
                    "spec": {"seed": 11},
                    "spec_sha256": "b" * 64,
                    "metrics": {"rank_ic": 0.03},
                    "evidence": {"evidence_sha256": "c" * 64},
                    "evidence_sha256": "c" * 64,
                    "resource": {"cpu_only": True},
                    "created_at": NOW,
                    "updated_at": NOW,
                },
                {
                    "id": "trial-ensemble-1",
                    "trial_kind": "model_ensemble",
                    "name": "ensemble:1",
                    "status": "passed",
                    "candidate_id": "ensemble-1",
                    "feature_set_id": None,
                    "feature_set_definition_sha256": None,
                    "model_family": None,
                    "spec": {"combiner": "equal_rank"},
                    "spec_sha256": "d" * 64,
                    "metrics": {"passed": True},
                    "evidence": {"evidence_sha256": "e" * 64},
                    "evidence_sha256": "e" * 64,
                    "resource": {"cpu_only": True},
                    "created_at": NOW,
                    "updated_at": NOW,
                },
            ],
        }

    def get_ensemble(self, candidate_id: str) -> dict:
        assert candidate_id == "ensemble-1"
        return {
            "id": candidate_id,
            "status": "research_admitted",
            "manifest_sha256": "f" * 64,
            "admission_evidence": {"status": "passed"},
            "admission_evidence_sha256": "1" * 64,
            "evaluations": [
                {"profile_id": "recent_3y", "seed": 11, "gate_status": "passed"}
            ],
        }


class _Experiments:
    def get(self, experiment_id: str) -> dict:
        assert experiment_id == "portfolio-experiment-1"
        return {
            "id": experiment_id,
            "job_id": "portfolio-job-1",
            "strategy_version_id": "strategy-version-1",
            "dataset": "daily-v1",
            "status": "succeeded",
            "periods": {"valid_start": "2025-01-01", "valid_end": "2025-06-30"},
            "parameter_grid": [{"portfolio_construction": "topk_equal_weight"}],
            "baseline_config": {"topk": 50},
            "summary": {"winner_trial_index": 0},
            "error": None,
            "created_at": NOW,
            "started_at": NOW,
            "finished_at": NOW,
            "trials": [
                {
                    "id": "portfolio-trial-1",
                    "trial_index": 0,
                    "status": "passed",
                    "parameters": {"portfolio_construction": "topk_equal_weight"},
                    "config": {"topk": 50},
                    "score": 1.2,
                    "metrics": {"annualized_excess_return_with_cost": 0.08},
                    "warnings": [],
                    "error": None,
                    "created_at": NOW,
                    "started_at": NOW,
                    "finished_at": NOW,
                }
            ],
        }


def _service(*, tournament_missing: bool = False) -> AutopilotTrialAuditService:
    return AutopilotTrialAuditService(
        research=_Research(),
        candidates=_Candidates(),
        tournaments=_Tournaments(missing=tournament_missing),
        parameter_experiments=_Experiments(),
    )


def test_cycle_trial_audit_joins_every_existing_governed_ledger() -> None:
    cycle = {
        "id": "cycle-1",
        "state": {
            "capital_pipeline": {
                "portfolio_experiment_id": "portfolio-experiment-1"
            }
        },
        "branches": [
            {
                "id": "factor-branch-1",
                "scenario": "fin_factor",
                "scope_key": "daily:2026-08-26",
                "status": "failed",
                "research_run_id": "factor-run",
                "job_id": "factor-job",
                "details": {"retry_count": 1},
                "error": "independent factor evaluator failed",
                "created_at": NOW,
                "updated_at": NOW,
                "finished_at": NOW,
            },
            {
                "id": "quant-branch-1",
                "scenario": "fin_quant",
                "scope_key": "quant:1",
                "status": "succeeded",
                "research_run_id": "quant-run",
                "job_id": "quant-job",
                "details": {"input_sha256": "2" * 64},
                "error": None,
                "created_at": NOW,
                "updated_at": NOW,
                "finished_at": NOW,
            },
        ],
    }

    records = _service().list_cycle_trials(cycle)
    by_type = {item["record_type"] for item in records}
    assert {
        "preregistered_trial",
        "research_branch",
        "factor_candidate",
        "model_candidate",
        "quant_bundle_candidate",
        "portfolio_experiment",
        "portfolio_trial",
    } <= by_type

    old_trial = next(item for item in records if item["id"] == "trial-model-1")
    assert old_trial["spec"] == {"seed": 11}
    assert old_trial["evidence"] == {"evidence_sha256": "c" * 64}
    assert old_trial["category"] == "model"
    assert old_trial["identity"]["tournament_id"] == "tournament-1"

    ensemble = next(item for item in records if item["id"] == "trial-ensemble-1")
    assert ensemble["category"] == "ensemble"
    assert ensemble["candidate_evidence"]["evaluations"][0]["seed"] == 11

    factor = next(item for item in records if item["id"] == "factor-1")
    assert factor["category"] == "factor"
    assert factor["evidence"]["evaluations"][0]["gate_status"] == "passed"

    bundle = next(item for item in records if item["id"] == "bundle-1")
    assert bundle["category"] == "fin_quant"
    assert bundle["evidence"]["failure_reason"] == (
        "joint challenger did not beat incumbent"
    )
    assert bundle["evidence"]["evaluations"][0]["ablation"] == "joint"

    failed_branch = next(item for item in records if item["id"] == "factor-branch-1")
    assert failed_branch["evidence"]["failure_reason"] == (
        "independent factor evaluator failed"
    )
    assert next(item for item in records if item["id"] == "portfolio-trial-1")[
        "evidence"
    ]["metrics"]["annualized_excess_return_with_cost"] == 0.08


def test_daily_cycle_without_model_tournament_still_returns_branch_and_factor_trials() -> None:
    cycle = {
        "id": "daily-cycle",
        "state": {},
        "branches": [
            {
                "id": "factor-branch-1",
                "scenario": "fin_factor",
                "scope_key": "daily:2026-08-26",
                "status": "running",
                "research_run_id": "factor-run",
                "job_id": "factor-job",
                "details": {},
                "error": None,
                "created_at": NOW,
                "updated_at": NOW,
                "finished_at": None,
            }
        ],
    }

    records = _service(tournament_missing=True).list_cycle_trials(cycle)

    assert [item["id"] for item in records] == ["factor-branch-1", "factor-1"]
    assert {item["category"] for item in records} == {"factor"}
