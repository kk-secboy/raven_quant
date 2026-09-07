from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy.sql.dml import Update
from test_autopilot_research_events import _cycle, _dataset, _tick_fixture

from quant_platform.autopilot import (
    AutopilotStore,
    _cycle_terminal_resolution,
    _derived_branch_status,
    _joint_completion_verified,
    normalize_autopilot_config,
)
from quant_platform.fin_strategy_research_completion import completed_event
from quant_platform.research_horizon import SHORT_1_5D
from quant_platform.research_tournament import canonical_sha256

pytestmark = pytest.mark.no_database


def _seal(value):
    return {**value, "evidence_sha256": canonical_sha256(value)}


def _terminal_fixture(tmp_path, *, mode="managed_fin_strategy", kind="model", resource=True):
    dataset = _dataset(tmp_path)
    cycle = _cycle(dataset, completion_mode=mode)
    model_evidence = _seal({
        "contract_version": "model-family-champions-v2",
        "dataset_identity_sha256": "a" * 64,
        "failed_and_rejected_trials_retained": True,
    })
    selection = _seal({
        "contract_version": "prediction-champion-selection-v1",
        "dataset_identity_sha256": "a" * 64,
        "selected_candidate_id": "admitted-champion", "selected_kind": kind,
        "selected_trial_ids": ["selected-trial"],
        "failed_and_rejected_trials_retained": True,
        "final_oos_opened": False,
    })
    cycle["state"].update({
        "prediction_champion": {
            "kind": kind, "candidate_id": "admitted-champion", "trial_id": "selected-trial",
        },
        "prediction_champion_evidence": selection, "model_champion_evidence": model_evidence,
        "research_tournament_id": "current-model-tournament",
    })
    cycle["branches"] = [
        {
            "id": "screen-branch", "scenario": "fin_model",
            "status": _derived_branch_status("blocked", "succeeded") if resource else "succeeded",
            "error": "model resource limit" if resource else None,
            "details": {
                "branch_kind": "platform_model_feature_screen",
                "tournament_id": "current-model-tournament",
                "candidate_bindings": [{"trial_id": "failed-screen", "candidate_id": "screen"}],
            },
        },
        {"id": "joint-branch", "scenario": "fin_quant", "status": "succeeded"},
    ]
    tournament = {
        "id": "current-model-tournament", "cycle_id": cycle["id"],
        "dataset_identity_sha256": "a" * 64, "status": "succeeded",
        "finished_at": "2026-09-07T00:00:00+00:00", "manifest": {},
        "selected_trial_ids": ["selected-trial"], "multiple_testing": deepcopy(selection),
        "multiple_testing_sha256": canonical_sha256(selection),
        "trials": [{
            "id": "failed-screen", "status": "failed" if resource else "passed",
            "metrics": {"reason_code": "resource_blocked" if resource else "passed"},
            "evidence": {"investment_hypothesis_rejected": False},
        }],
    }
    return dataset, cycle, tournament


class _Rows(list):
    def all(self):
        return self

    def first(self):
        return self[0] if self else None


def _row(value):
    return SimpleNamespace(**value, _mapping=value)


class _RefreshConnection:
    """In-memory persisted rows; exercise real refresh queries and state writes."""

    def __init__(self, cycle, tournament):
        self.cycle = cycle
        self.tournament = tournament

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def execute(self, statement):
        if isinstance(statement, Update):
            assert statement.table.name == "autopilot_cycles"
            values = statement.compile().params
            self.cycle.update({
                name: values[name] for name in ("status", "stage", "updated_at", "finished_at")
                if name in values
            })
            return _Rows()
        name = statement.get_final_froms()[0].name
        if name == "autopilot_cycles":
            value = {**self.cycle, "state_json": deepcopy(self.cycle["state"])}
            return _Rows([_row(value)] if self.cycle["status"] == "active" else [])
        if name == "autopilot_branches":
            return _Rows([_row(item) for item in self.cycle["branches"]])
        assert name == "research_tournaments"
        assert self.cycle["state"]["research_tournament_id"] in statement.compile().params.values()
        if self.tournament is None:
            return _Rows()
        value = deepcopy(self.tournament)
        for key in ("manifest", "multiple_testing", "selected_trial_ids"):
            value[key + "_json"] = value.pop(key)
        return _Rows([_row(value)])


def _advance(path, monkeypatch, dataset, cycle, tournament):
    if path == "refresh":
        store = AutopilotStore.__new__(AutopilotStore)
        store.engine = SimpleNamespace(begin=lambda: _RefreshConnection(cycle, tournament))
        store._refresh_cycles()
        return
    controller, captured, _ = _tick_fixture(monkeypatch, dataset, cycle)

    def get_tournament(*_):
        if tournament is None:
            raise KeyError("no frozen tournament")
        return tournament

    controller.tournaments.get_for_cycle = get_tournament
    controller.tournaments.get_tournament = get_tournament
    controller._retry_failed_branch = lambda *_: False
    controller._quant_due = lambda *_args, **_kwargs: False
    controller._tick_horizon(
        current=datetime(2026, 9, 7, tzinfo=UTC), config=normalize_autopilot_config(),
        revision=7, dataset=dataset, horizon_profile=SHORT_1_5D,
    )
    if cycle.get("finished"):
        cycle["finished_at"] = "2026-09-07T00:00:00+00:00"
    assert not captured["quant"]


@pytest.mark.parametrize("path", ["refresh", "tick"])
@pytest.mark.parametrize("mode", ["research_only", "managed_fin_strategy"])
@pytest.mark.parametrize("kind", ["model", "ensemble"])
@pytest.mark.parametrize("resource", [False, True])
def test_successful_joint_closes_both_paths_without_rewriting_failed_candidates(
    tmp_path, monkeypatch, path, mode, kind, resource,
):
    dataset, cycle, tournament = _terminal_fixture(
        tmp_path, mode=mode, kind=kind, resource=resource,
    )
    before = deepcopy((cycle["branches"], tournament, cycle["state"]))
    _advance(path, monkeypatch, dataset, cycle, tournament)
    assert (cycle["status"], cycle["stage"]) == ("succeeded", "research_complete")
    assert (completed_event(cycle) is not None) is (mode == "managed_fin_strategy")
    assert (cycle["branches"], tournament) == before[:2]
    assert all(cycle["state"][key] == value for key, value in before[2].items())


@pytest.mark.parametrize("path", ["refresh", "tick"])
@pytest.mark.parametrize("broken", [
    "dataset", "tournament_dataset", "cycle", "unsealed", "missing_tournament", "champion",
    "selection_hash", "tournament_hash", "selection_binding", "model_evidence", "selected_ids",
])
def test_missing_or_changed_sealed_selection_cannot_complete(
    tmp_path, monkeypatch, path, broken,
):
    dataset, cycle, tournament = _terminal_fixture(tmp_path)
    if broken == "dataset":
        cycle["state"]["prediction_champion_evidence"]["dataset_identity_sha256"] = "c" * 64
    elif broken == "tournament_dataset":
        tournament["dataset_identity_sha256"] = "c" * 64
    elif broken == "cycle":
        tournament["cycle_id"] = "other-cycle"
    elif broken == "unsealed":
        tournament["status"] = "running"
    elif broken == "missing_tournament":
        tournament = None
    elif broken == "champion":
        cycle["state"].pop("prediction_champion")
    elif broken == "selection_hash":
        cycle["state"]["prediction_champion_evidence"]["evidence_sha256"] = "0" * 64
    elif broken == "tournament_hash":
        tournament["multiple_testing_sha256"] = "0" * 64
    elif broken == "selection_binding":
        tournament["multiple_testing"] = {"different": "selection"}
        tournament["multiple_testing_sha256"] = canonical_sha256(tournament["multiple_testing"])
    elif broken == "model_evidence":
        cycle["state"].pop("model_champion_evidence")
    else:
        tournament["selected_trial_ids"] = ["another-trial"]
    before = deepcopy((cycle["branches"], tournament))
    _advance(path, monkeypatch, dataset, cycle, tournament)
    assert (cycle["status"], cycle["stage"]) == ("blocked", "research_blocked")
    assert completed_event(cycle) is None
    assert (cycle["branches"], tournament) == before


@pytest.mark.parametrize("path", ["refresh", "tick"])
@pytest.mark.parametrize("status", ["running", "failed", "blocked"])
def test_unfinished_or_failed_joint_cannot_complete(tmp_path, monkeypatch, path, status):
    dataset, cycle, tournament = _terminal_fixture(tmp_path)
    cycle["branches"][-1]["status"] = status
    before = deepcopy((cycle["branches"], tournament))
    _advance(path, monkeypatch, dataset, cycle, tournament)
    assert cycle["status"] == ("active" if status == "running" else "blocked")
    if status != "running":
        assert cycle["stage"] == "joint_optimization_blocked"
    assert (cycle["branches"], tournament) == before


@pytest.mark.parametrize("kind", ["model", "ensemble"])
@pytest.mark.parametrize("revalidated", [False, True])
def test_existing_successor_and_revalidation_selection_formats(kind, revalidated, tmp_path):
    _, cycle, tournament = _terminal_fixture(tmp_path, kind=kind)
    cycle["state"]["active_research_tournament_id"] = tournament["id"]
    tournament["manifest"]["operational_remediation"] = {"source_tournament_id": "old-tournament"}
    if revalidated:
        selection = dict(cycle["state"]["prediction_champion_evidence"])
        selection.pop("evidence_sha256")
        selection["fixed_prior_champion_current_identity_revalidation"] = True
        selection["global_multiple_testing"] = _seal({"contract_version": "synthetic-finalists"})
        cycle["state"]["prediction_champion_evidence"] = _seal(selection)
        cycle["state"]["current_identity_revalidation"] = {
            "status": "succeeded", "tournament_id": tournament["id"],
        }
        tournament["multiple_testing"] = selection["global_multiple_testing"]
        tournament["multiple_testing_sha256"] = canonical_sha256(tournament["multiple_testing"])
    assert _joint_completion_verified(cycle, tournament)
    cycle["state"]["active_research_tournament_id"] = "unsealed-successor"
    assert not _joint_completion_verified(cycle, tournament)


def test_no_joint_or_other_active_owner_does_not_gain_a_terminal_resolution():
    for branches in (
        [], [("fin_model", "blocked")],
        [("fin_model", "running"), ("fin_quant", "succeeded")],
    ):
        assert _cycle_terminal_resolution(branches, joint_completion_verified=True) is None
