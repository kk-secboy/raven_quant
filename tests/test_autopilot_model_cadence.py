from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import quant_platform.autopilot as module
from quant_platform.autopilot import AutopilotController, normalize_autopilot_config
from quant_platform.research_horizon import (
    LONG_1_3Y,
    SHORT_1_5D,
    SWING_1_6M,
    model_selection_cadence_bucket,
    primary_label_policy_sha256,
    research_cadence_bucket,
)
from quant_platform.research_tournament import FULL_PROFILES, FULL_SEEDS, canonical_sha256

pytestmark = pytest.mark.no_database


def _seal(value):
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def _dataset(end="2026-09-14", identity="d" * 64):
    return {
        "name": f"dataset-{end}", "end_date": end, "lineage_id": "f" * 64,
        "frequency": "day", "provenance": {
            "dataset_identity_sha256": identity, "frequency": "day",
            "field_contract_version": "fields-v8", "eligibility_contract_version": "v1",
            "research_features": {"version": 8},
        },
    }


def _cycle(dataset, *, cycle_id="current", horizon=SHORT_1_5D):
    return {
        "id": cycle_id, "dataset": dataset["name"], "horizon_profile": horizon,
        "dataset_identity_sha256": dataset["provenance"]["dataset_identity_sha256"],
        "dataset_lineage_id": dataset["lineage_id"], "research_event_key": "scheduled",
        "status": "active", "branches": [], "state": {
            "dataset_end_date": dataset["end_date"],
            "research_data_contract": module._research_data_contract(dataset),
        },
    }


def _champion_cycle(end="2026-09-07", *, horizon=SHORT_1_5D):
    dataset = _dataset(end, "a" * 64)
    cycle = _cycle(dataset, cycle_id="previous", horizon=horizon)
    cycle["status"] = "succeeded"
    cycle["branches"] = [{"scenario": "fin_model", "status": "succeeded"}]
    identity = cycle["dataset_identity_sha256"]
    selection = _seal({
        "contract_version": "prediction-champion-selection-v1",
        "dataset_identity_sha256": identity, "selected_kind": "model",
        "selected_candidate_id": "old-model", "final_oos_opened": False,
        "global_multiple_testing": {"contract_version": "prediction-finalist-multiple-testing-v2"},
    })
    model_selection = _seal({
        "contract_version": "model-family-champions-v2", "dataset_identity_sha256": identity,
        "final_oos_opened": False,
        "multiple_testing": {"contract_version": "model-full-multiple-testing-v2"},
    })
    cycle["state"].update({
        "prediction_champion": {
            "kind": "model", "candidate_id": "old-model", "model_family": "ridge",
            "manifest_sha256": "b" * 64, "admission_evidence_sha256": "c" * 64,
        },
        "prediction_champion_evidence": selection, "model_champion_evidence": model_selection,
    })
    return dataset, cycle


class _Store:
    def __init__(self, *cycles):
        self.cycles = {item["id"]: item for item in cycles}
        self.patches = 0

    def list_cycles(self, *, limit):
        assert limit == 500
        return list(self.cycles.values())

    def get_cycle(self, cycle_id):
        return self.cycles[cycle_id]

    def patch_cycle_state(self, cycle_id, *, state_patch, stage=None):
        self.patches += 1
        cycle = self.cycles[cycle_id]
        cycle["state"].update(state_patch)
        if stage:
            cycle["stage"] = stage
        return cycle


def _controller(*cycles):
    controller = AutopilotController.__new__(AutopilotController)
    controller.store = _Store(*cycles)
    controller.settings = SimpleNamespace(data_root="unused")
    return controller


def _mode(controller, cycle, dataset, *, force=False, tournament=None):
    return controller._ensure_model_selection_schedule(
        cycle, dataset, now=datetime(2026, 9, 14, tzinfo=UTC),
        config=normalize_autopilot_config(), force_full=force,
        existing_model_tournament=tournament,
    )["state"]["model_selection_schedule"]


@pytest.mark.parametrize(("horizon", "start", "same", "next_bucket"), [
    (SHORT_1_5D, "2026-09-07", "2026-09-14", "2026-10-01"),
    (SWING_1_6M, "2026-07-01", "2026-08-03", "2026-10-01"),
    (LONG_1_3Y, "2026-07-01", "2026-10-01", "2027-01-04"),
])
def test_full_selection_clock_is_slower_than_hypothesis_clock(horizon, start, same, next_bucket):
    _, previous = _champion_cycle(start, horizon=horizon)
    controller = _controller(previous)
    assert research_cadence_bucket(horizon, start) != research_cadence_bucket(horizon, same)
    assert model_selection_cadence_bucket(horizon, start) == model_selection_cadence_bucket(
        horizon, same
    )
    for end, expected in [(same, False), (next_bucket, True)]:
        assert controller._model_due(
            _dataset(end), datetime.now(UTC), {}, horizon_profile=horizon
        ) is expected


def test_revalidation_does_not_postpone_next_full_selection_even_without_original_cycle():
    _, previous = _champion_cycle("2026-08-03")
    origin = module._full_model_selection_origin(previous)
    previous["state"]["dataset_end_date"] = "2026-09-07"
    previous["state"]["prediction_champion_evidence"][
        "fixed_prior_champion_current_identity_revalidation"
    ] = True
    previous["state"]["prediction_champion_roll_forward"] = {
        "full_model_selection_origin": origin,
    }
    controller = _controller(previous)
    assert controller._model_due(_dataset(), datetime.now(UTC), {}) is True
    previous["state"]["prediction_champion_roll_forward"] = {}
    assert controller._model_due(_dataset(), datetime.now(UTC), {}) is True


@pytest.mark.parametrize("mutation", ["no_champion", "new_lineage", "other_horizon", "future"])
def test_invalid_or_unavailable_selection_cannot_suppress_full_selection(mutation):
    _, previous = _champion_cycle()
    if mutation == "no_champion":
        previous["state"]["prediction_champion"] = None
    elif mutation == "new_lineage":
        previous["dataset_lineage_id"] = "b" * 64
    elif mutation == "other_horizon":
        previous["horizon_profile"] = LONG_1_3Y
    else:
        previous["state"]["dataset_end_date"] = "2026-09-30"
    assert _controller(previous)._model_due(_dataset(), datetime.now(UTC), {}) is True


@pytest.mark.parametrize("force", [False, True])
def test_missing_incumbent_and_manual_request_always_get_complete_selection(force):
    dataset = _dataset()
    cycle = _cycle(dataset)
    controller = _controller(cycle)
    controller._model_due = lambda *_a, **_kw: False
    assert _mode(controller, cycle, dataset, force=force)["mode"] == "full_selection"


def _pending_cycle():
    dataset = _dataset()
    cycle = _cycle(dataset)
    cycle["state"].update({
        "prior_prediction_champion": {"candidate_id": "old-model"},
        "prediction_champion_status": "pending_current_identity_revalidation",
        "prediction_champion_roll_forward": {"research_data_contract_compatible": True},
    })
    controller = _controller(cycle)
    controller._model_due = lambda *_a, **_kw: False
    controller._champion_revalidation_source = lambda _: ()
    return dataset, cycle, controller


def test_revalidation_mode_is_frozen_once_and_ignores_later_clock_change():
    dataset, cycle, controller = _pending_cycle()
    first = deepcopy(_mode(controller, cycle, dataset))
    controller._model_due = lambda *_a, **_kw: pytest.fail("must not reselect mid-activity")
    controller._champion_revalidation_source = lambda _: pytest.fail("source already frozen")
    assert _mode(controller, cycle, dataset) == first
    assert controller.store.patches == 1
    assert first["mode"] == "champion_revalidation"
    assert first["final_oos_opened"] is False


@pytest.mark.parametrize("cause", ["contract", "candidate", "model_due", "manual", "existing"])
def test_full_selection_overrides_recipe_reuse_for_material_changes(cause):
    dataset, cycle, controller = _pending_cycle()
    if cause == "contract":
        cycle["state"]["prediction_champion_roll_forward"].update(
            research_data_contract_compatible=False
        )
    elif cause == "candidate":
        def changed(_):
            raise ValueError("candidate immutable feature definition changed")
        controller._champion_revalidation_source = changed
    elif cause == "model_due":
        controller._model_due = lambda *_a, **_kw: True
    result = _mode(
        controller, cycle, dataset, force=cause == "manual",
        tournament={"id": "existing-full"} if cause == "existing" else None,
    )
    assert result["mode"] == "full_selection"


@pytest.mark.parametrize("change", ["mode", "identity", "date", "force"])
def test_frozen_selection_choice_rejects_mutation(change):
    dataset, cycle, controller = _pending_cycle()
    schedule = _mode(controller, cycle, dataset)
    if change == "mode":
        schedule["mode"] = "full_selection"
    elif change == "identity":
        dataset["provenance"]["dataset_identity_sha256"] = "8" * 64
    elif change == "date":
        dataset["end_date"] = "2026-10-01"
    with pytest.raises(ValueError, match="frozen model selection schedule"):
        _mode(controller, cycle, dataset, force=change == "force")


class _Connection:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def scalar(self, _):
        return "old-model"


def _wire_candidate(controller, monkeypatch):
    candidate = {
        "status": "research_admitted", "manifest_sha256": "b" * 64,
        "admission_evidence_sha256": "c" * 64, "code_sha256": "e" * 64,
        "feature_set_definition_sha256": "9" * 64,
        "base_features_manifest_json": {"feature_set_id": "qlib-alpha158"},
        "manifest_json": {
            "recipe_sha256": "8" * 64, "dataset_identity_sha256": "a" * 64,
            "dataset_lineage_id": "f" * 64,
            "research_label_binding": {"horizon_profile": SHORT_1_5D},
            "primary_label_policy_sha256": primary_label_policy_sha256(),
        },
    }
    controller.platform_models = SimpleNamespace(candidates=SimpleNamespace(
        get_model_candidate=lambda _id, *, verify: candidate,
    ))
    monkeypatch.setattr(module, "get_feature_set", lambda key: {
        "id": key, "definition_sha256": "9" * 64,
    })
    controller.engine = SimpleNamespace(connect=_Connection)
    return candidate


@pytest.mark.parametrize("semantic_change", [False, True])
def test_fresh_cycle_finds_compatible_controlled_champion_without_existing_context(
    monkeypatch, semantic_change
):
    prior_dataset, previous = _champion_cycle()
    dataset = _dataset()
    cycle = _cycle(dataset)
    controller = _controller(cycle, previous)
    _wire_candidate(controller, monkeypatch)
    # Exercise the migration compatibility path for pre-existing rows too.
    previous["state"].pop("research_data_contract")
    if semantic_change:
        dataset["provenance"]["research_features"]["version"] = 9
    monkeypatch.setattr(module, "list_qlib_datasets", lambda _: [prior_dataset, dataset])
    controller._roll_forward_prediction_champion(cycle, dataset)
    result = _mode(controller, cycle, dataset)
    if semantic_change:
        assert result["mode"] == "full_selection"
        assert "prior_prediction_champion" not in cycle["state"]
    else:
        assert result["mode"] == "champion_revalidation"
        assert "research_tournament_id" not in cycle["state"]
        assert cycle["state"]["prediction_champion"] is None
        rollover = cycle["state"]["prediction_champion_roll_forward"]
        assert rollover["full_model_selection_origin"]["cycle_id"] == previous["id"]
        assert rollover["old_predictions_reused"] is False
        assert rollover["old_scores_reused"] is False


@pytest.mark.parametrize("field", ["lineage", "feature", "label", "policy", "identity"])
def test_recipe_compatibility_is_verified_before_revalidation_is_selected(monkeypatch, field):
    prior_dataset, previous = _champion_cycle()
    dataset = _dataset()
    cycle = _cycle(dataset)
    controller = _controller(cycle, previous)
    candidate = _wire_candidate(controller, monkeypatch)
    monkeypatch.setattr(module, "list_qlib_datasets", lambda _: [prior_dataset, dataset])
    controller._roll_forward_prediction_champion(cycle, dataset)
    manifest = candidate["manifest_json"]
    if field == "lineage":
        manifest["dataset_lineage_id"] = "0" * 64
    elif field == "feature":
        candidate["feature_set_definition_sha256"] = "0" * 64
    elif field == "label":
        manifest["research_label_binding"]["horizon_profile"] = LONG_1_3Y
    elif field == "policy":
        manifest["primary_label_policy_sha256"] = "0" * 64
    else:
        manifest["dataset_identity_sha256"] = "0" * 64
    assert _mode(controller, cycle, dataset)["mode"] == "full_selection"


def test_new_research_week_runs_full_incumbent_grid_once_without_full_tournament(
    tmp_path, monkeypatch
):
    from test_autopilot_research_events import _tick_fixture

    prior_dataset, previous = _champion_cycle()
    dataset = _dataset()
    dataset["path"] = str(tmp_path)
    (tmp_path / "calendars").mkdir()
    (tmp_path / "calendars" / "day.txt").write_text("2026-01-01\n", encoding="utf-8")
    cycle = _cycle(dataset)
    controller, captured, _ = _tick_fixture(monkeypatch, dataset, cycle)
    store = _Store(cycle, previous)
    store.ensure_cycle = lambda *_a, **_kw: cycle
    store.latest_branch = lambda *_a, **_kw: {"cycle_id": previous["id"]}
    store.branch_for_scope = lambda _id, scenario, scope: next(
        (b for b in cycle["branches"] if b["scope_key"] == scope and b["scenario"] == scenario),
        None,
    )
    store.create_branch = lambda _id, **values: cycle["branches"].append({
        **values, "status": "queued",
    })
    controller.store = store
    _wire_candidate(controller, monkeypatch)
    controller._roll_forward_prediction_champion = (
        AutopilotController._roll_forward_prediction_champion.__get__(controller)
    )
    controller._current_identity_revalidation_pending = (
        AutopilotController._current_identity_revalidation_pending.__get__(controller)
    )
    controller._quant_due = lambda *_a, **_kw: False
    controller._quant_input_sha256 = lambda *_a: None
    controller._reconcile_current_identity_revalidation = lambda *_a: None
    controller._enqueue_reports = lambda *_a, **_kw: 0
    tournament = {
        "id": "revalidation-ledger", "manifest_sha256": "7" * 64, "status": "running",
        "trials": [{"id": "new-trial", "spec": {"source_model_candidate_id": "old-model"}}],
    }

    def no_full(_):
        raise KeyError("new cycle has no full competition ledger")

    controller.tournaments = SimpleNamespace(
        get_for_cycle=no_full,
        get_tournament=lambda _: tournament,
        ensure_champion_revalidation_preregistered=lambda **kw: (
            captured.setdefault("revalidation_registration", []).append(kw) or tournament
        ),
        mark_running=lambda _: None,
        ensure_preregistered=lambda **_: pytest.fail("weekly research must not run full contest"),
    )
    controller._advance_tournament_trial = lambda *_a, **_kw: None
    profiles = [{"id": profile} for profile in FULL_PROFILES]
    monkeypatch.setattr(module, "resolve_research_window_contract", lambda *_a, **_kw: (
        {}, {"evaluation_profiles": profiles, "research_window_contract": {},
             "research_window_contract_sha256": "6" * 64},
    ))
    monkeypatch.setattr(module, "list_qlib_datasets", lambda _: [dataset, prior_dataset])

    def lane(**values):
        captured.setdefault("revalidation_lanes", []).append(values)
        return {
            "run": {"id": "new-run"}, "job": {"id": "new-job"},
            "bindings": [{"trial_id": "new-trial", "candidate_id": "new-model"}],
        }

    controller.platform_models.ensure_champion_revalidation_lane = lane
    for _ in range(2):
        controller._tick_horizon(
            current=datetime.now(UTC), config=normalize_autopilot_config(), revision=1,
            dataset=dataset, horizon_profile=SHORT_1_5D,
        )
    assert cycle["status"] == "active"
    assert len(captured["revalidation_lanes"]) == 1
    assert captured["revalidation_lanes"][0]["evaluation_profiles"] == profiles
    progress = cycle["state"]["current_identity_revalidation"]
    assert progress["profiles"] == list(FULL_PROFILES)
    assert progress["seeds"] == list(FULL_SEEDS)
    assert progress["final_oos_opened"] is False
    assert len(cycle["branches"]) == 1
    assert cycle["branches"][0]["details"]["branch_kind"] == (
        "champion_current_identity_revalidation"
    )


@pytest.mark.parametrize("old_row", [False, True])
def test_completed_activity_cannot_hide_same_week_data_contract_migration(old_row):
    prior_dataset, previous = _champion_cycle("2026-09-07")
    dataset = _dataset("2026-09-08")
    controller = _controller(previous)
    controller.store.latest_branch = lambda *_a, **_kw: {"cycle_id": previous["id"]}
    if old_row:
        previous["state"].pop("research_data_contract")
    available = {prior_dataset["name"]: prior_dataset}
    assert controller._horizon_branch_due(
        "fin_model", dataset, horizon_profile=SHORT_1_5D, available_datasets=available,
    ) is False
    dataset["provenance"]["field_contract_version"] = "repaired-fields-v9"
    assert controller._horizon_branch_due(
        "fin_model", dataset, horizon_profile=SHORT_1_5D, available_datasets=available,
    ) is True


def test_successful_manual_selection_can_supply_next_scheduled_research():
    _, previous = _champion_cycle()
    previous["research_event_key"] = "manual:complete-mainline"
    assert _controller(previous)._model_due(_dataset(), datetime.now(UTC), {}) is False


@pytest.mark.parametrize("horizon", ["legacy_ambiguous", "other"])
def test_unsupported_model_selection_cadence_is_not_invented(horizon):
    with pytest.raises(ValueError, match="unsupported"):
        model_selection_cadence_bucket(horizon, "2026-09-07")


def test_invalid_model_selection_date_is_rejected():
    with pytest.raises(ValueError, match="date is invalid"):
        model_selection_cadence_bucket(SHORT_1_5D, "not-a-date")
