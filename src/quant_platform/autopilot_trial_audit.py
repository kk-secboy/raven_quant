from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

_SCENARIO_CATEGORY = {
    "fin_factor": "factor",
    "fin_factor_report": "factor",
    "fin_model": "model",
    "fin_quant": "fin_quant",
}

_TRIAL_CATEGORY = {
    "feature_set": "factor",
    "model": "model",
    "model_ensemble": "ensemble",
    "quant_bundle": "fin_quant",
    "portfolio": "portfolio",
}


def _failure_reason(value: Any) -> str | None:
    reason = str(value or "").strip()
    return reason[:3000] or None


class AutopilotTrialAuditService:
    """Build one read-only audit view from the existing immutable ledgers.

    The service deliberately does not derive scores or reinterpret gate
    outcomes.  It only joins records that already belong to an Autopilot
    cycle: preregistered tournament trials, research branches and their
    candidates, ensemble evidence, and the bounded portfolio experiment.
    """

    def __init__(
        self,
        *,
        research: Any,
        candidates: Any,
        tournaments: Any,
        parameter_experiments: Any,
    ) -> None:
        self.research = research
        self.candidates = candidates
        self.tournaments = tournaments
        self.parameter_experiments = parameter_experiments

    def list_cycle_trials(self, cycle: Mapping[str, Any]) -> list[dict[str, Any]]:
        cycle_id = str(cycle["id"])
        records: list[dict[str, Any]] = []
        records.extend(self._tournament_records(cycle_id))

        visited_runs: set[str] = set()
        for branch in cycle.get("branches") or []:
            branch_value = dict(branch)
            records.append(self._branch_record(branch_value))
            run_id = str(branch_value.get("research_run_id") or "")
            if not run_id or run_id in visited_runs:
                continue
            visited_runs.add(run_id)
            records.extend(self._factor_records(run_id))
            records.extend(self._model_and_quant_records(run_id))

        records.extend(self._portfolio_records(cycle))
        return records

    def _tournament_records(self, cycle_id: str) -> list[dict[str, Any]]:
        try:
            tournament = self.tournaments.get_for_cycle(cycle_id)
        except KeyError:
            return []

        result: list[dict[str, Any]] = []
        for raw in tournament.get("trials") or []:
            trial = deepcopy(dict(raw))
            trial_kind = str(trial.get("trial_kind") or "")
            trial["record_type"] = "preregistered_trial"
            trial["category"] = _TRIAL_CATEGORY.get(trial_kind, trial_kind or "research")
            trial["identity"] = {
                "trial_id": str(trial.get("id") or ""),
                "tournament_id": str(tournament.get("id") or ""),
                "tournament_manifest_sha256": tournament.get("manifest_sha256"),
                "spec_sha256": trial.get("spec_sha256"),
                "candidate_id": trial.get("candidate_id"),
                "feature_set_id": trial.get("feature_set_id"),
                "feature_set_definition_sha256": trial.get(
                    "feature_set_definition_sha256"
                ),
                "model_family": trial.get("model_family"),
            }
            trial["tournament"] = {
                "stage": tournament.get("stage"),
                "status": tournament.get("status"),
                "selected_trial_ids": list(
                    tournament.get("selected_trial_ids") or []
                ),
                "multiple_testing": tournament.get("multiple_testing"),
                "multiple_testing_sha256": tournament.get(
                    "multiple_testing_sha256"
                ),
            }
            candidate_id = str(trial.get("candidate_id") or "")
            if trial_kind == "model_ensemble" and candidate_id:
                try:
                    ensemble = self.tournaments.get_ensemble(candidate_id)
                except KeyError:
                    ensemble = None
                if ensemble is not None:
                    trial["candidate_evidence"] = {
                        "status": ensemble.get("status"),
                        "manifest_sha256": ensemble.get("manifest_sha256"),
                        "admission_evidence": ensemble.get("admission_evidence"),
                        "admission_evidence_sha256": ensemble.get(
                            "admission_evidence_sha256"
                        ),
                        "evaluations": list(ensemble.get("evaluations") or []),
                    }
            result.append(trial)
        return result

    @staticmethod
    def _branch_record(branch: Mapping[str, Any]) -> dict[str, Any]:
        scenario = str(branch.get("scenario") or "")
        return {
            "id": str(branch.get("id") or ""),
            "record_type": "research_branch",
            "category": _SCENARIO_CATEGORY.get(scenario, scenario or "research"),
            "name": scenario,
            "status": str(branch.get("status") or ""),
            "identity": {
                "branch_id": str(branch.get("id") or ""),
                "scenario": scenario,
                "scope_key": branch.get("scope_key"),
                "research_run_id": branch.get("research_run_id"),
                "job_id": branch.get("job_id"),
            },
            "evidence": {
                "details": deepcopy(dict(branch.get("details") or {})),
                "failure_reason": _failure_reason(branch.get("error")),
            },
            "created_at": branch.get("created_at"),
            "updated_at": branch.get("updated_at"),
            "finished_at": branch.get("finished_at"),
        }

    def _factor_records(self, run_id: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for raw in self.research.list_candidates(run_id=run_id, limit=500):
            candidate = dict(raw)
            candidate_id = str(candidate["id"])
            evaluations = self.research.list_evaluations(candidate_id, limit=500)
            result.append(
                {
                    "id": candidate_id,
                    "record_type": "factor_candidate",
                    "category": "factor",
                    "name": candidate.get("name"),
                    "status": str(candidate.get("status") or ""),
                    "identity": {
                        "candidate_id": candidate_id,
                        "research_run_id": run_id,
                        "factor_definition_id": candidate.get("factor_definition_id"),
                        "formulation": candidate.get("formulation"),
                        "code_sha256": candidate.get("code_sha256"),
                        "values_sha256": candidate.get("values_sha256"),
                        "experiment_family_id": candidate.get(
                            "experiment_family_id"
                        ),
                        "economic_family": candidate.get("economic_family"),
                        "similarity_cluster_id": candidate.get(
                            "similarity_cluster_id"
                        ),
                    },
                    "evidence": {
                        "evaluations": deepcopy(evaluations),
                        "profile_consensus": candidate.get("profile_consensus"),
                        "profile_consensus_sha256": candidate.get(
                            "profile_consensus_sha256"
                        ),
                        "admission_path": candidate.get("admission_path"),
                        "incremental_evidence": candidate.get(
                            "incremental_evidence"
                        ),
                        "incremental_evidence_sha256": candidate.get(
                            "incremental_evidence_sha256"
                        ),
                        "promotion_evidence_sha256": candidate.get(
                            "promotion_evidence_sha256"
                        ),
                        "rdagent_decision": candidate.get("rdagent_decision"),
                        "rdagent_feedback": candidate.get("rdagent_feedback"),
                    },
                    "created_at": candidate.get("created_at"),
                    "updated_at": candidate.get("updated_at"),
                }
            )
        return result

    def _model_and_quant_records(self, run_id: str) -> list[dict[str, Any]]:
        try:
            archive = self.candidates.run_audit_summary(run_id)
        except KeyError:
            return []

        model_evaluations: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for evaluation in archive.get("model_evaluations") or []:
            model_evaluations[str(evaluation.get("model_candidate_id") or "")].append(
                dict(evaluation)
            )
        bundle_evaluations: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for evaluation in archive.get("quant_bundle_evaluations") or []:
            bundle_evaluations[
                str(evaluation.get("quant_bundle_candidate_id") or "")
            ].append(dict(evaluation))

        result: list[dict[str, Any]] = []
        for raw in archive.get("model_candidates") or []:
            model = dict(raw)
            candidate_id = str(model["id"])
            try:
                detail = self.candidates.get_model_candidate(candidate_id)
            except KeyError:
                detail = {}
            result.append(
                {
                    "id": candidate_id,
                    "record_type": "model_candidate",
                    "category": "model",
                    "name": model.get("name"),
                    "status": str(model.get("status") or ""),
                    "identity": {
                        "candidate_id": candidate_id,
                        "research_run_id": run_id,
                        "model_type": model.get("model_type"),
                        "code_sha256": model.get("code_sha256"),
                        "feature_set_definition_sha256": model.get(
                            "feature_set_definition_sha256"
                        ),
                        "dataset": model.get("dataset"),
                        "dataset_identity_sha256": model.get(
                            "dataset_identity_sha256"
                        ),
                        "manifest_sha256": model.get("manifest_sha256"),
                    },
                    "evidence": {
                        "evaluations": model_evaluations.get(candidate_id, []),
                        "admission_evidence": detail.get("admission_evidence_json"),
                        "admission_evidence_sha256": model.get(
                            "admission_evidence_sha256"
                        ),
                        "failure_reason": _failure_reason(
                            detail.get("rejection_reason")
                        ),
                        "rdagent_decision": model.get("rdagent_decision"),
                        "rdagent_feedback": model.get("rdagent_feedback"),
                    },
                    "created_at": model.get("created_at"),
                    "updated_at": model.get("updated_at"),
                }
            )

        for raw in archive.get("quant_bundle_candidates") or []:
            bundle = dict(raw)
            candidate_id = str(bundle["id"])
            try:
                detail = self.candidates.get_quant_bundle_candidate(candidate_id)
            except KeyError:
                detail = {}
            result.append(
                {
                    "id": candidate_id,
                    "record_type": "quant_bundle_candidate",
                    "category": "fin_quant",
                    "name": bundle.get("name"),
                    "status": str(bundle.get("status") or ""),
                    "identity": {
                        "candidate_id": candidate_id,
                        "research_run_id": run_id,
                        "prediction_component_kind": bundle.get(
                            "prediction_component_kind"
                        ),
                        "model_candidate_id": bundle.get("model_candidate_id"),
                        "model_ensemble_candidate_id": bundle.get(
                            "model_ensemble_candidate_id"
                        ),
                        "factor_candidate_ids": list(
                            bundle.get("factor_candidate_ids") or []
                        ),
                        "dataset": bundle.get("dataset"),
                        "dataset_identity_sha256": bundle.get(
                            "dataset_identity_sha256"
                        ),
                        "bundle_manifest_sha256": bundle.get(
                            "bundle_manifest_sha256"
                        ),
                    },
                    "evidence": {
                        "evaluations": bundle_evaluations.get(candidate_id, []),
                        "ablation_evidence": detail.get("ablation_evidence_json"),
                        "ablation_evidence_sha256": detail.get(
                            "ablation_evidence_sha256"
                        ),
                        "admission_evidence": detail.get("admission_evidence_json"),
                        "admission_evidence_sha256": bundle.get(
                            "admission_evidence_sha256"
                        ),
                        "failure_reason": _failure_reason(
                            detail.get("rejection_reason")
                        ),
                        "rdagent_decision": bundle.get("rdagent_decision"),
                        "rdagent_feedback": bundle.get("rdagent_feedback"),
                    },
                    "created_at": bundle.get("created_at"),
                    "updated_at": bundle.get("updated_at"),
                }
            )
        return result

    def _portfolio_records(self, cycle: Mapping[str, Any]) -> list[dict[str, Any]]:
        capital = dict((cycle.get("state") or {}).get("capital_pipeline") or {})
        experiment_id = str(capital.get("portfolio_experiment_id") or "")
        if not experiment_id:
            return []
        try:
            experiment = self.parameter_experiments.get(experiment_id)
        except KeyError:
            return [
                {
                    "id": experiment_id,
                    "record_type": "portfolio_experiment",
                    "category": "portfolio",
                    "name": "TopK/QP portfolio competition",
                    "status": "missing",
                    "identity": {"parameter_experiment_id": experiment_id},
                    "evidence": {
                        "failure_reason": "portfolio experiment record is unavailable"
                    },
                }
            ]

        parent = {
            "id": experiment_id,
            "record_type": "portfolio_experiment",
            "category": "portfolio",
            "name": "TopK/QP portfolio competition",
            "status": str(experiment.get("status") or ""),
            "identity": {
                "parameter_experiment_id": experiment_id,
                "strategy_version_id": experiment.get("strategy_version_id"),
                "job_id": experiment.get("job_id"),
                "dataset": experiment.get("dataset"),
            },
            "evidence": {
                "periods": experiment.get("periods"),
                "parameter_grid": experiment.get("parameter_grid"),
                "baseline_config": experiment.get("baseline_config"),
                "summary": experiment.get("summary"),
                "failure_reason": _failure_reason(experiment.get("error")),
            },
            "created_at": experiment.get("created_at"),
            "started_at": experiment.get("started_at"),
            "finished_at": experiment.get("finished_at"),
        }
        result = [parent]
        for raw in experiment.get("trials") or []:
            trial = dict(raw)
            result.append(
                {
                    "id": str(trial.get("id") or ""),
                    "record_type": "portfolio_trial",
                    "category": "portfolio",
                    "name": f"portfolio:{trial.get('trial_index')}",
                    "status": str(trial.get("status") or ""),
                    "identity": {
                        "parameter_experiment_id": experiment_id,
                        "trial_id": trial.get("id"),
                        "trial_index": trial.get("trial_index"),
                    },
                    "evidence": {
                        "parameters": trial.get("parameters"),
                        "config": trial.get("config"),
                        "score": trial.get("score"),
                        "metrics": trial.get("metrics"),
                        "warnings": trial.get("warnings"),
                        "failure_reason": _failure_reason(trial.get("error")),
                    },
                    "created_at": trial.get("created_at"),
                    "started_at": trial.get("started_at"),
                    "finished_at": trial.get("finished_at"),
                }
            )
        return result


__all__ = ["AutopilotTrialAuditService"]
