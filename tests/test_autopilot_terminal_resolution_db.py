from __future__ import annotations

from sqlalchemy import select, update
from test_autopilot_research_events import _dataset, _registered

from quant_data.database import (
    autopilot_branches,
    jobs,
    research_runs,
    research_tournament_trials,
    research_tournaments,
)
from quant_platform.autopilot import AutopilotStore
from quant_platform.fin_strategy_research_completion import completed_event
from quant_platform.job_store import JobStore
from quant_platform.research_store import ResearchStore
from quant_platform.research_tournament import ResearchTournamentStore, canonical_sha256


def _sealed(value):
    return {**value, "evidence_sha256": canonical_sha256(value)}


def test_refresh_cycles_consumes_real_sealed_json_and_preserves_resource_failure(
    database_url, tmp_path,
):
    store = AutopilotStore(database_url)
    tournaments = ResearchTournamentStore(database_url)
    research = ResearchStore(database_url)
    job_store = JobStore(database_url)
    dataset = _dataset(tmp_path)

    # All rows are disposable synthetic state fixtures, not model performance
    # or independent admission evidence. Use real registration/transition/seal
    # services so JSON decoding and the production terminal SQL are exercised.
    for broken in (False, True):
        cycle = _registered(
            store, dataset, event_key=f"terminal-json-{broken}",
            completion_mode="managed_fin_strategy",
        )
        tournament = tournaments.ensure_preregistered(
            cycle_id=cycle["id"], dataset_identity_sha256="a" * 64,
        )
        winner, resource_trial = tournament["trials"][:2]
        for status in ("queued", "running", "passed", "selected"):
            tournaments.transition_trial(winner["id"], status, candidate_id="synthetic-champion")
        tournaments.transition_trial(resource_trial["id"], "queued")
        tournaments.transition_trial(
            resource_trial["id"], "failed", metrics={"reason_code": "resource_blocked"},
            evidence={"investment_hypothesis_rejected": False, "final_oos_opened": False},
        )
        for trial in tournament["trials"][2:]:
            tournaments.transition_trial(trial["id"], "rejected")
        selection = _sealed({
            "contract_version": "prediction-champion-selection-v1",
            "dataset_identity_sha256": "a" * 64,
            "selected_candidate_id": "synthetic-champion", "selected_kind": "model",
            "selected_trial_ids": [winner["id"]], "final_oos_opened": False,
            "failed_and_rejected_trials_retained": True,
        })
        tournaments.complete_selection(
            tournament["id"], selected_trial_ids=[winner["id"]], multiple_testing=selection,
        )
        store.patch_cycle_state(cycle["id"], state_patch={
            "research_tournament_id": tournament["id"],
            "prediction_champion": {
                "kind": "model", "candidate_id": "synthetic-champion", "trial_id": winner["id"],
            },
            "prediction_champion_evidence": selection,
            "model_champion_evidence": _sealed({
                "contract_version": "model-family-champions-v2",
                "dataset_identity_sha256": "a" * 64,
                "failed_and_rejected_trials_retained": True,
            }),
        })
        owned = {}
        for scenario, kind in (("fin_model", "model_evaluate"), ("fin_quant", "rdagent_quant")):
            run = research.create_run(
                kind=f"synthetic-terminal-{scenario}", objective="test state settlement",
                dataset=dataset["name"], requested_by="test", budget={}, config={},
                artifact_path=tmp_path / cycle["id"] / scenario,
            )
            job = job_store.create(
                kind, {"research_run_id": run["id"]}, tmp_path / f"{run['id']}.log",
            )
            research.attach_job(run["id"], job["id"])
            branch = store.create_branch(
                cycle["id"], scenario=scenario,
                scope_key="joint" if scenario == "fin_quant" else "screen",
                research_run_id=run["id"], job_id=job["id"],
                details={
                    "branch_kind": (
                        "platform_model_feature_screen" if scenario == "fin_model" else "joint"
                    ),
                    "tournament_id": tournament["id"],
                },
            )
            owned[scenario] = (run, job, branch)
        model_run, model_job, model_branch = owned["fin_model"]
        research.mark_run(
            model_run["id"], "blocked", error="model resource limit",
            runtime={"reason_code": "model_resource_limit"},
        )
        job_store.finish(model_job["id"], exit_code=0, result={"resource_blocked": True})
        store.reconcile()  # The independent joint owner is still queued.
        assert store.get_cycle(cycle["id"])["status"] == "active"
        assert store.get_branch(model_branch["id"])["status"] == "blocked"

        if broken:
            # Only this second fixture's stored seal is corrupted. The first
            # completed cycle and its immutable tournament are never rewritten.
            with store.engine.begin() as connection:
                connection.execute(update(research_tournaments).where(
                    research_tournaments.c.id == tournament["id"]
                ).values(multiple_testing_sha256="0" * 64))

        def retained_rows(
            model_branch=model_branch, model_run=model_run, model_job=model_job,
            tournament=tournament,
        ):
            with store.engine.connect() as connection:
                statements = [
                    select(autopilot_branches).where(autopilot_branches.c.id == model_branch["id"]),
                    select(research_runs).where(research_runs.c.id == model_run["id"]),
                    select(jobs).where(jobs.c.id == model_job["id"]),
                    select(research_tournaments).where(
                        research_tournaments.c.id == tournament["id"]
                    ),
                    select(research_tournament_trials).where(
                        research_tournament_trials.c.tournament_id == tournament["id"]
                    ).order_by(research_tournament_trials.c.id),
                ]
                return [[dict(row) for row in connection.execute(query).mappings()]
                        for query in statements]

        before = retained_rows()
        quant_run, quant_job, _ = owned["fin_quant"]
        research.mark_run(quant_run["id"], "succeeded")
        job_store.finish(quant_job["id"], exit_code=0)
        store.reconcile()  # Runs the actual _refresh_cycles transaction.
        closed = store.get_cycle(cycle["id"])
        assert (closed["status"], closed["stage"]) == (
            ("blocked", "research_blocked") if broken else ("succeeded", "research_complete")
        )
        assert closed["finished_at"] is not None
        assert (completed_event(closed) is not None) is (not broken)
        assert retained_rows() == before
        store._refresh_cycles()
        assert store.get_cycle(cycle["id"]) == closed
        assert retained_rows() == before
