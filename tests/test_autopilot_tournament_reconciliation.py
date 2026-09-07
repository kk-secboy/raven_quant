from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from quant_platform.autopilot import AutopilotController

pytestmark = pytest.mark.no_database


def _controller(trials):
    controller = object.__new__(AutopilotController)
    writes = []

    def transition(trial_id, status, **values):
        writes.append((trial_id, status))
        trials[trial_id].update(status=status, **values)

    controller.tournaments = SimpleNamespace(
        get_trial=lambda trial_id: deepcopy(trials[trial_id]),
        transition_trial=transition,
    )
    return controller, writes


def _selected_trial(candidate_id="screen-candidate"):
    return {
        "status": "selected",
        "candidate_id": candidate_id,
        "metrics": {"original_screen_score": 0.1},
        "evidence": {"contract_version": "model-feature-screen-v1", "cells": ["screen-cell"]},
        "evidence_sha256": "a" * 64,
        "updated_at": "2026-09-07T05:59:14+00:00",
    }


@pytest.mark.parametrize("screen_status", ["research_admitted", "invalidated", "rejected"])
def test_selected_feature_screens_do_not_stop_later_model_reconciliation(screen_status):
    trials = {
        "alpha158-screen": _selected_trial("alpha158-candidate"),
        "seed-screen": _selected_trial("seed-candidate"),
        "alpha158-full": {"status": "queued", "candidate_id": "full-candidate"},
    }
    original_screens = deepcopy({key: value for key, value in trials.items() if "screen" in key})
    controller, writes = _controller(trials)
    branches = []
    runs = {}
    evidence = {}
    for trial_id, trial in trials.items():
        screening = "screen" in trial_id
        candidate_id = trial["candidate_id"]
        branches.append({
            "research_run_id": trial_id,
            "details": {
                "branch_kind": (
                    "platform_model_feature_screen" if screening else "platform_model_model_full"
                ),
                "tournament_id": "tournament",
                "candidate_bindings": [{"trial_id": trial_id, "candidate_id": candidate_id}],
            },
        })
        runs[trial_id] = {"status": "succeeded" if screening else "running"}
        evidence[candidate_id] = {
            "candidate_status": screen_status if screening else "evaluating",
            "cells": [{"profile_id": "independent_gate"}],
        }
    controller.store = SimpleNamespace(get_cycle=lambda _cycle_id: {"branches": branches})
    controller.research = SimpleNamespace(get_run=lambda run_id: runs[run_id])
    controller._candidate_tournament_evidence = lambda candidate_id: evidence[candidate_id]

    for _ in range(2):
        controller._reconcile_model_tournament({"id": "cycle"}, {"id": "tournament"})

    assert trials["alpha158-full"]["status"] == "running"
    assert writes == [("alpha158-full", "running")]
    assert {key: trials[key] for key in original_screens} == original_screens


@pytest.mark.parametrize("candidate_id", [None, "screen-candidate"])
@pytest.mark.parametrize("target", ["passed", "failed", "rejected"])
def test_reobserved_outcome_keeps_selected_trial_and_original_evidence(candidate_id, target):
    trials = {"screen": _selected_trial()}
    before = deepcopy(trials)
    controller, writes = _controller(trials)

    controller._advance_tournament_trial(
        "screen", target, candidate_id=candidate_id,
        metrics={"later_independent_metrics": 2}, evidence={"replacement": True},
    )

    assert trials == before
    assert writes == []


@pytest.mark.parametrize("target", ["passed", "failed", "rejected"])
def test_reobserved_outcome_cannot_change_selected_candidate_binding(target):
    trials = {"screen": _selected_trial()}
    before = deepcopy(trials)
    controller, writes = _controller(trials)

    with pytest.raises(ValueError, match="candidate binding cannot change"):
        controller._advance_tournament_trial(
            "screen", target, candidate_id="different-candidate",
        )

    assert trials == before
    assert writes == []


@pytest.mark.parametrize("target", ["queued", "running"])
def test_selected_trial_still_rejects_reopening(target):
    trials = {"screen": _selected_trial()}
    before = deepcopy(trials)
    controller, writes = _controller(trials)

    with pytest.raises(ValueError, match="terminal result cannot change"):
        controller._advance_tournament_trial("screen", target, candidate_id="screen-candidate")

    assert trials == before
    assert writes == []


@pytest.mark.parametrize("status", ["failed", "rejected"])
def test_failed_or_rejected_trial_is_not_promoted_by_reobserved_pass(status):
    trials = {"screen": {**_selected_trial(), "status": status}}
    before = deepcopy(trials)
    controller, writes = _controller(trials)

    with pytest.raises(ValueError, match="terminal result cannot change"):
        controller._advance_tournament_trial("screen", "passed", candidate_id="screen-candidate")

    assert trials == before
    assert writes == []
