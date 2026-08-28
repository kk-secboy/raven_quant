from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import select

from quant_data.config import Settings
from quant_data.database import (
    model_ensemble_candidates,
    research_tournament_trials,
    research_tournaments,
)

from .job_store import JobStore
from .model_ensemble import (
    EnsemblePredictionsPending,
    bounded_ensemble_combinations,
    pairwise_grid_correlation,
    prediction_grid_from_admission,
)
from .rdagent_candidate_store import RDAGentCandidateStore
from .research_tournament import ResearchTournamentStore, canonical_sha256


class ModelEnsemblePipelineService:
    """Preregister and queue bounded cross-family equal-rank ensembles."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs = JobStore(settings.database_url)
        self.tournaments = ResearchTournamentStore(settings.database_url)
        self.candidates = RDAGentCandidateStore(settings.database_url)
        self.engine = self.tournaments.engine

    def ensure_candidates(
        self,
        *,
        tournament_id: str,
        champion_trial_ids: Sequence[str],
        dataset: str,
        dataset_identity_sha256: str,
    ) -> dict[str, Any]:
        trial_ids = [str(item) for item in champion_trial_ids]
        if len(set(trial_ids)) != len(trial_ids) or not 2 <= len(trial_ids) <= 4:
            raise ValueError("ensemble generation requires two to four unique champion trials")
        with self.engine.connect() as connection:
            tournament = connection.execute(
                select(research_tournaments).where(
                    research_tournaments.c.id == tournament_id
                )
            ).first()
            rows = connection.execute(
                select(research_tournament_trials).where(
                    research_tournament_trials.c.tournament_id == tournament_id,
                    research_tournament_trials.c.id.in_(trial_ids),
                )
            ).all()
        if tournament is None:
            raise KeyError(tournament_id)
        if str(tournament.dataset_identity_sha256) != dataset_identity_sha256:
            raise ValueError("ensemble generation dataset identity changed")
        if len(rows) != len(trial_ids):
            raise ValueError("one or more ensemble champion trials do not exist")
        by_id = {str(row.id): row for row in rows}
        champions: list[dict[str, Any]] = []
        grids: dict[str, dict[str, Any]] = {}
        for trial_id in trial_ids:
            row = by_id[trial_id]
            if (
                str(row.trial_kind) != "model"
                or str(row.status) not in {"passed", "selected"}
                or not str(row.candidate_id or "")
                or not str(row.model_family or "")
            ):
                raise ValueError("ensemble champion is not a passed model trial")
            candidate = self.candidates.get_model_candidate(
                str(row.candidate_id), verify=True
            )
            if str(candidate.get("status")) != "research_admitted":
                raise EnsemblePredictionsPending(
                    "ensemble champion has not completed independent admission"
                )
            if (
                str(candidate.get("dataset")) != dataset
                or str(candidate.get("dataset_identity_sha256"))
                != dataset_identity_sha256
            ):
                raise ValueError("ensemble champion belongs to another dataset")
            grid = prediction_grid_from_admission(candidate)
            candidate_id = str(candidate["id"])
            grids[candidate_id] = grid
            champions.append(
                {
                    "id": candidate_id,
                    "model_family": str(row.model_family),
                    "manifest_sha256": str(candidate["manifest_sha256"]),
                    "admission_evidence_sha256": str(
                        candidate["admission_evidence_sha256"]
                    ),
                    "prediction_grid_sha256": str(grid["prediction_grid_sha256"]),
                }
            )
        families = [str(item["model_family"]) for item in champions]
        if len(set(families)) != len(families):
            raise ValueError("ensemble champions must come from distinct model families")
        pairwise: dict[tuple[str, str], dict[str, Any]] = {}
        for left, right in itertools.combinations(champions, 2):
            left_id = str(left["id"])
            right_id = str(right["id"])
            pairwise[(left_id, right_id)] = pairwise_grid_correlation(
                grids[left_id], grids[right_id]
            )
        specs = bounded_ensemble_combinations(champions, pairwise)
        created: list[dict[str, Any]] = []
        for spec in specs:
            correlations = [
                float(item["maximum_mean_absolute_daily_rank_correlation"])
                for item in spec["correlation_evidence"]
            ]
            created.append(
                self.tournaments.create_ensemble(
                    tournament_id=tournament_id,
                    name=str(spec["name"]),
                    dataset=dataset,
                    dataset_identity_sha256=dataset_identity_sha256,
                    components=[dict(item) for item in spec["components"]],
                    prediction_correlations=correlations,
                    correlation_evidence=[
                        dict(item) for item in spec["correlation_evidence"]
                    ],
                )
            )
        return {
            "status": "ready" if created else "no_admissible_combinations",
            "candidate_count": len(created),
            "candidates": created,
            "pairwise_correlation_evidence": [
                dict(value) for _, value in sorted(pairwise.items())
            ],
        }

    def queue_evaluation(
        self,
        *,
        tournament_id: str,
        dataset: Mapping[str, Any],
        evaluation_profiles: Sequence[Mapping[str, Any]],
        universe: str = "cn_all",
        benchmark: str = "SH000300",
    ) -> dict[str, Any] | None:
        identity = str((dataset.get("provenance") or {}).get("dataset_identity_sha256") or "")
        if len(identity) != 64:
            raise ValueError("ensemble evaluation requires a sealed dataset identity")
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(model_ensemble_candidates).where(
                    model_ensemble_candidates.c.tournament_id == tournament_id,
                    model_ensemble_candidates.c.status.in_(
                        ("awaiting_evaluation", "evaluating")
                    ),
                )
            ).all()
        if not rows:
            return None
        candidates: list[dict[str, Any]] = []
        for row in rows:
            if str(row.dataset_identity_sha256) != identity:
                raise ValueError("ensemble evaluation candidate dataset identity changed")
            components: list[dict[str, Any]] = []
            for component in row.components_json or []:
                model = self.candidates.get_model_candidate(
                    str(component["model_candidate_id"]), verify=True
                )
                grid = prediction_grid_from_admission(model)
                if (
                    str(model["manifest_sha256"])
                    != str(component["model_manifest_sha256"])
                    or str(model["admission_evidence_sha256"])
                    != str(component["model_admission_evidence_sha256"])
                    or str(grid["prediction_grid_sha256"])
                    != str(component["prediction_grid_sha256"])
                ):
                    raise ValueError("ensemble component evidence changed after preregistration")
                components.append({**dict(component), "prediction_grid": grid})
            candidates.append(
                {
                    "id": str(row.id),
                    "manifest": dict(row.manifest_json or {}),
                    "manifest_sha256": str(row.manifest_sha256),
                    "components": components,
                }
            )
        candidates.sort(key=lambda item: str(item["id"]))
        payload = {
            "tournament_id": tournament_id,
            "dataset": str(dataset["name"]),
            "dataset_path": str(dataset["path"]),
            "dataset_identity_sha256": identity,
            "evaluation_profiles": [dict(item) for item in evaluation_profiles],
            "candidates": candidates,
            "universe": str(universe),
            "benchmark": str(benchmark),
            "account": 100_000_000,
            "topk": 50,
            "n_drop": 5,
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5.0,
        }
        candidate_set_sha = canonical_sha256(
            [
                {
                    "id": item["id"],
                    "manifest_sha256": item["manifest_sha256"],
                }
                for item in candidates
            ]
        )
        job = self.jobs.create(
            "model_ensemble_evaluate",
            payload,
            self.settings.data_root
            / "platform"
            / "logs"
            / f"model-ensemble-evaluate-{tournament_id}.log",
            dedupe_active_kind=False,
            idempotency_key=(
                f"model-ensemble-evaluate:{tournament_id}:{candidate_set_sha}"
            ),
            max_attempts=2,
        )
        for candidate in candidates:
            self.tournaments.mark_ensemble_evaluating(str(candidate["id"]))
        return job
