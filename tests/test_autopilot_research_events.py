from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, date, datetime
from threading import Barrier
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.dialects import postgresql

import quant_platform.autopilot as module
from quant_data.database import autopilot_cycles, jobs, model_candidates, research_runs
from quant_platform.autopilot import AutopilotController, AutopilotStore, normalize_autopilot_config
from quant_platform.rdagent_candidate_store import RDAGentCandidateStore
from quant_platform.research_horizon import SHORT_1_5D, SWING_1_6M
from quant_platform.research_store import ResearchStore
from quant_platform.research_tournament import ResearchTournamentStore, canonical_sha256


def _dataset(tmp_path, *, name="verified-daily", identity="a" * 64, end="2026-09-04"):
    return {
        "name": name, "path": str(tmp_path / name), "end_date": end,
        "ready": True, "reproducible": True, "lineage_verified": True, "frequency": "day",
        "lineage_id": "b" * 64, "provenance": {"dataset_identity_sha256": identity},
    }


def _request(**changes):
    return {
        "event_key": "operator-event-1", "horizon_profile": SHORT_1_5D,
        "actor": "operator", "reason": "预算内继续探索新因子", "quant_loop_n": 10,
        "quant_duration": "1h", "completion_mode": "research_only", **changes,
    }


def _cycle(dataset, **request_changes):
    request = _request(**request_changes)
    config = normalize_autopilot_config({
        "quant_loop_n": request["quant_loop_n"], "quant_duration": request["quant_duration"],
    })
    event = {
        "contract_version": module.MANUAL_RESEARCH_EVENT_CONTRACT, "request": request,
        "config": config, "config_revision": 7, "dataset": module._event_dataset_binding(dataset),
    }
    event["sha256"] = canonical_sha256(event)
    return {
        "id": "manual-cycle", "dataset": dataset["name"],
        "dataset_identity_sha256": dataset["provenance"]["dataset_identity_sha256"],
        "dataset_lineage_id": dataset["lineage_id"], "horizon_profile": request["horizon_profile"],
        "research_event_key": "manual:" + request["event_key"], "config_revision": 7,
        "primary_label_policy_sha256": module.primary_label_policy_sha256(),
        "status": "active", "stage": "parallel_research", "branches": [],
        "state": {
            "research_event": event, "dataset_end_date": dataset["end_date"],
            "label_horizon_sessions": module.primary_label_horizon_sessions(
                request["horizon_profile"]
            ),
            "primary_label_policy": module.primary_label_policy_contract(),
        },
    }


def _controller(store, dataset):
    controller = AutopilotController.__new__(AutopilotController)
    controller.store = store
    controller.settings = SimpleNamespace(
        data_root="unused", rdagent_max_loops=10, rdagent_max_duration="2h"
    )
    controller.config = lambda: (normalize_autopilot_config(), 7)
    controller._latest_dataset = lambda: dataset
    return controller


class _Connection:
    def __init__(self, values, queries=None):
        self.values = iter(values)
        self.queries = queries

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def scalar(self, _query):
        if self.queries is not None:
            self.queries.append(_query)
        return next(self.values)


@pytest.mark.no_database
def test_start_freezes_effective_config_and_only_registers(tmp_path):
    calls = []
    dataset = _dataset(tmp_path)
    store = SimpleNamespace(
        get_research_event=lambda _key: None,
        create_research_event=lambda **kwargs: calls.append(kwargs) or {"id": "registered"},
    )
    controller = _controller(store, dataset)
    original, _ = controller.config()
    controller.config = lambda: (original, 7)
    assert controller.start_research_event(**_request()) == {"id": "registered"}
    assert calls[0]["config_revision"] == 7
    assert calls[0]["dataset"] is dataset
    assert calls[0]["config"] == {**original, "quant_loop_n": 10, "quant_duration": "1h"}
    assert original["quant_loop_n"] == 2
    assert calls[0]["request"] == _request()
    # No JobStore, runtime, tournament, or tick dependency exists on this controller.


@pytest.mark.no_database
def test_same_event_replay_uses_original_inputs_after_data_and_config_change(tmp_path):
    original = _cycle(_dataset(tmp_path))
    original["status"] = "succeeded"
    controller = _controller(
        SimpleNamespace(get_research_event=lambda _key: original),
        _dataset(tmp_path, identity="c" * 64),
    )
    controller.settings.rdagent_max_loops = 3
    controller.config = lambda: pytest.fail("replay must not resolve today's configuration")
    controller._latest_dataset = lambda: pytest.fail("replay must not rebind the data")
    assert controller.start_research_event(**_request()) is original
    with pytest.raises(ValueError, match="another request"):
        controller.start_research_event(**_request(quant_loop_n=9))


@pytest.mark.no_database
@pytest.mark.parametrize("change", [
    {"event_key": "../unsafe"}, {"event_key": ""}, {"horizon_profile": "legacy_ambiguous"},
    {"actor": ""}, {"reason": " "}, {"quant_loop_n": True}, {"quant_loop_n": 1},
    {"quant_loop_n": 21}, {"quant_duration": "forever"},
])
def test_invalid_request_never_reaches_storage(tmp_path, change):
    controller = _controller(SimpleNamespace(), _dataset(tmp_path))
    with pytest.raises(ValueError):
        controller.start_research_event(**_request(**change))


@pytest.mark.no_database
@pytest.mark.parametrize("block", ["loops", "duration", "disabled", "dataset"])
def test_execution_and_availability_limits_are_not_overridden(tmp_path, block):
    controller = _controller(
        SimpleNamespace(get_research_event=lambda _key: None), _dataset(tmp_path)
    )
    if block == "loops":
        controller.settings.rdagent_max_loops = 3
    elif block == "duration":
        controller.settings.rdagent_max_duration = "30m"
    elif block == "disabled":
        controller.config = lambda: (normalize_autopilot_config({"enabled": False}), 8)
    else:
        controller._latest_dataset = lambda: None
    with pytest.raises(ValueError):
        controller.start_research_event(**_request())


@pytest.mark.no_database
@pytest.mark.parametrize("field", ["request", "config", "dataset", "config_revision"])
def test_frozen_event_tampering_is_rejected(tmp_path, field):
    cycle = _cycle(_dataset(tmp_path))
    event = cycle["state"]["research_event"]
    if field == "request":
        event[field]["reason"] = "changed"
    elif field == "config":
        event[field]["report_daily_limit"] = 1
    elif field == "dataset":
        event[field]["dataset_identity_sha256"] = "c" * 64
    else:
        event[field] += 1
    with pytest.raises(ValueError, match="frozen inputs"):
        module._manual_research_event(cycle)


def _tick_fixture(monkeypatch, dataset, cycle):
    old = {
        **deepcopy(cycle), "id": "old-terminal", "research_event_key": "scheduled",
        "status": "succeeded", "state": {"result": "research_cadence_not_due"},
    }
    captured = {"preregistrations": [], "platform": [], "quant": [], "reports": []}

    def update(_id, **values):
        assert _id == cycle["id"]
        cycle.update(values)
        return cycle

    def no_scheduled(*_args, **_kwargs):
        pytest.fail("manual activity must not create or reopen a scheduled cycle")

    store = SimpleNamespace(
        list_cycles=lambda **_: [old, cycle], get_cycle=lambda _id: cycle,
        ensure_cycle=no_scheduled, supersede_research_cycle=no_scheduled,
        set_cycle_state=update, branch_for_scope=lambda *_: next(
            (b for b in cycle["branches"] if b["scenario"] == "fin_quant"), None
        ),
    )
    controller = _controller(store, dataset)
    tournament = {"id": "current-model-tournament", "status": "running"}
    controller.tournaments = SimpleNamespace(
        get_for_cycle=lambda _id: tournament, get_tournament=lambda _id: tournament,
        ensure_preregistered=lambda **kw: captured["preregistrations"].append(kw),
    )
    controller._model_due = lambda *_, **__: False
    controller._roll_forward_prediction_champion = lambda item, _data: item
    controller._active_sota_feature_set_id = lambda *_, **__: None
    controller._ensure_horizon_factor_bundle = lambda item, *_a, **_kw: item
    controller._current_identity_revalidation_pending = lambda *_: False
    controller._ensure_platform_model_branches = lambda *args, **kwargs: (
        captured["platform"].append((args, kwargs)) or 0, 0
    )
    controller._reconcile_model_tournament = lambda *_: None
    controller._screen_selected_feature_sets = lambda *_: []
    controller._enqueue_reports = lambda *_, **kw: captured["reports"].append(kw) or 0
    controller.factor_autopilot = SimpleNamespace(ensure_incremental_lane=lambda **_: None)
    controller._quant_input_sha256 = lambda *_: "e" * 64
    controller.engine = SimpleNamespace(connect=lambda: _Connection(["admitted-model", None]))

    def enqueue(*args, **kwargs):
        captured["quant"].append((args, kwargs))
        cycle["branches"].append({"scenario": "fin_quant", "status": "queued"})

    controller._enqueue = enqueue
    monkeypatch.setattr(module, "list_qlib_datasets", lambda _: [dataset])
    return controller, captured, old


@pytest.mark.no_database
def test_manual_tick_keeps_original_data_config_and_preregisters_normal_model_tournament(
    tmp_path, monkeypatch
):
    dataset = _dataset(tmp_path)
    cycle = _cycle(dataset)
    before = deepcopy(cycle["state"]["research_event"])
    controller, captured, old = _tick_fixture(monkeypatch, dataset, cycle)
    old_before = deepcopy(old)
    newer = _dataset(tmp_path, name="new-vintage", identity="c" * 64, end="2026-09-11")
    config = normalize_autopilot_config({"quant_loop_n": 2, "report_daily_limit": 1})
    result = controller._tick_horizon(
        current=datetime(2026, 9, 12, tzinfo=UTC), config=config, revision=99,
        dataset=newer, horizon_profile=SHORT_1_5D,
    )
    assert result["failed"] == 0
    assert captured["preregistrations"] == [{
        "cycle_id": cycle["id"], "dataset_identity_sha256": "a" * 64,
        "active_sota_feature_set_id": None,
    }]
    assert captured["platform"][0][0][1] is dataset
    assert captured["platform"][0][1] == {"stage": "feature_screen"}
    assert captured["reports"][0]["config"] == before["config"]
    assert not captured["quant"]  # no champion exists yet
    assert cycle["state"]["research_event"] == before
    assert old == old_before


@pytest.mark.no_database
@pytest.mark.parametrize("broken", ["absent", "identity", "not_ready"])
def test_missing_or_replaced_bound_data_blocks_without_switching(tmp_path, monkeypatch, broken):
    dataset = _dataset(tmp_path)
    cycle = _cycle(dataset)
    controller, captured, _ = _tick_fixture(monkeypatch, dataset, cycle)
    if broken == "absent":
        monkeypatch.setattr(module, "list_qlib_datasets", lambda _: [])
    elif broken == "identity":
        dataset["provenance"]["dataset_identity_sha256"] = "c" * 64
    else:
        dataset["ready"] = False
    outcome = controller._tick_horizon(
        current=datetime.now(UTC), config=normalize_autopilot_config(), revision=8,
        dataset=dataset, horizon_profile=SHORT_1_5D,
    )
    assert outcome["failed"] == 1
    assert cycle["status"] == "blocked"
    assert not captured["preregistrations"] and not captured["quant"]


def _add_champion(cycle):
    cycle["state"].update({
        "prediction_champion": {"kind": "model", "candidate_id": "admitted-model"},
        "prediction_champion_evidence": {"dataset_identity_sha256": "a" * 64},
        "model_champion_evidence": {"dataset_identity_sha256": "a" * 64},
        "research_tournament_id": "current-model-tournament",
    })


@pytest.mark.no_database
def test_manual_quant_uses_normal_champion_parent_binding_and_frozen_budget_once(
    tmp_path, monkeypatch
):
    dataset = _dataset(tmp_path)
    cycle = _cycle(dataset)
    _add_champion(cycle)
    controller, captured, _ = _tick_fixture(monkeypatch, dataset, cycle)
    for _ in range(2):
        controller._tick_horizon(
            current=datetime.now(UTC), config=normalize_autopilot_config(), revision=99,
            dataset=dataset, horizon_profile=SHORT_1_5D,
        )
    assert len(captured["quant"]) == 1
    args, options = captured["quant"][0]
    assert args[:4] == (cycle, dataset, "fin_quant", "joint")
    assert options["config"]["quant_loop_n"] == 10
    assert options["config"]["quant_duration"] == "1h"
    assert options["tournament_id"] == "current-model-tournament"
    assert options["branch_details"]["prediction_champion"] == cycle["state"][
        "prediction_champion"
    ]


@pytest.mark.no_database
@pytest.mark.parametrize("block", ["champion", "identity", "admission", "prerequisite", "parent"])
def test_manual_trigger_does_not_skip_existing_quant_gates(tmp_path, monkeypatch, block):
    dataset = _dataset(tmp_path)
    cycle = _cycle(dataset)
    _add_champion(cycle)
    controller, captured, _ = _tick_fixture(monkeypatch, dataset, cycle)
    if block == "champion":
        del cycle["state"]["prediction_champion"]
    elif block == "identity":
        cycle["state"]["prediction_champion_evidence"]["dataset_identity_sha256"] = "c" * 64
    elif block == "admission":
        controller.engine = SimpleNamespace(connect=lambda: _Connection([None, None]))
    elif block == "prerequisite":
        controller.engine = SimpleNamespace(connect=lambda: _Connection(["admitted-model", "busy"]))
    else:
        del cycle["state"]["research_tournament_id"]
    controller._tick_horizon(
        current=datetime.now(UTC), config=normalize_autopilot_config(), revision=7,
        dataset=dataset, horizon_profile=SHORT_1_5D,
    )
    assert not captured["quant"]


@pytest.mark.no_database
def test_model_lineage_gate_uses_real_manifest_column(tmp_path):
    dataset = _dataset(tmp_path)
    cycle = _cycle(dataset)
    _add_champion(cycle)
    queries = []
    controller = _controller(SimpleNamespace(), dataset)
    controller.engine = SimpleNamespace(connect=lambda: _Connection(["model", None], queries))
    assert controller._quant_due(
        cycle, dataset, datetime.now(UTC), normalize_autopilot_config(), input_sha256="e" * 64
    )
    sql = str(queries[0].compile(
        dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
    ))
    assert "manifest_json ->> 'dataset_lineage_id'" in sql
    assert "b" * 64 in sql
    assert "model_candidates.dataset_lineage_id" not in sql


@pytest.mark.no_database
@pytest.mark.parametrize("enabled", [True, False])
def test_global_pause_and_vanished_publication_do_not_restart_old_cycles(tmp_path, enabled):
    cycle = _cycle(_dataset(tmp_path))
    old = {"id": "old", "status": "succeeded", "research_event_key": "scheduled"}
    calls = []
    store = SimpleNamespace(
        reconcile=lambda: None, list_cycles=lambda **_: [old, cycle],
        set_cycle_state=lambda _id, **kw: calls.append((_id, kw)),
    )
    controller = _controller(store, None)
    controller.research = SimpleNamespace(reconcile_autopilot_execution_state=lambda: None)
    controller.config = lambda: (normalize_autopilot_config({"enabled": enabled}), 8)
    result = controller.tick()
    if enabled:
        assert calls[0][0] == cycle["id"]
        assert calls[0][1]["status"] == "blocked"
        assert result["failed"] == 1
    else:
        assert not calls
        assert result == {"cycles": 0, "branches": 0}


def _registered(store, dataset, **changes):
    request = _request(**changes)
    return store.create_research_event(
        request=request, dataset=dataset, config_revision=7,
        config=normalize_autopilot_config({
            "quant_loop_n": request["quant_loop_n"], "quant_duration": request["quant_duration"],
        }),
    )


def test_database_new_event_preserves_old_terminal_and_creates_no_jobs(database_url, tmp_path):
    store = AutopilotStore(database_url)
    dataset = _dataset(tmp_path)
    old = store.ensure_cycle(dataset, config_revision=1)
    old = store.set_cycle_state(
        old["id"], state={**old["state"], "result": "research_cadence_not_due"},
        status="succeeded", stage="complete", finished=True,
    )
    event = _registered(store, dataset)
    assert event["id"] != old["id"]
    assert store.get_cycle(old["id"]) == old
    assert store.ensure_cycle(dataset, config_revision=8) == old
    assert store.ensure_cycle(dataset, config_revision=8, prefer_manual=True) == event
    assert _registered(store, _dataset(tmp_path, identity="c" * 64)) == event
    with store.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(jobs)) == 0
        assert connection.scalar(select(func.count()).select_from(research_runs)) == 0
    for changes in ({"reason": "changed"}, {"horizon_profile": SWING_1_6M}):
        with pytest.raises(ValueError, match="another request"):
            _registered(store, dataset, **changes)
    parent = ResearchTournamentStore(database_url).ensure_preregistered(
        cycle_id=event["id"], dataset_identity_sha256="a" * 64
    )
    assert parent["cycle_id"] == event["id"] and parent["status"] == "planned"
    assert store.get_cycle(old["id"]) == old


@pytest.mark.parametrize("same_key", [True, False])
def test_database_concurrent_events_have_one_horizon_owner(database_url, tmp_path, same_key):
    dataset = _dataset(tmp_path)
    barrier = Barrier(2)

    def create(index):
        store = AutopilotStore(database_url)
        barrier.wait(timeout=10)
        try:
            return _registered(
                store, dataset, event_key="same" if same_key else f"event-{index}"
            )["id"]
        except ValueError as exc:
            assert "owns this horizon" in str(exc)
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create, [0, 1]))
    assert len({result for result in results if result is not None}) == 1
    assert results.count(None) == (0 if same_key else 1)
    store = AutopilotStore(database_url)
    with store.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(autopilot_cycles)) == 1


def test_database_global_key_and_frozen_terminal_are_not_reinterpreted(database_url, tmp_path):
    store = AutopilotStore(database_url)
    dataset = _dataset(tmp_path)
    event = _registered(store, dataset)
    with pytest.raises(ValueError, match="frozen inputs"):
        store.set_cycle_state(event["id"], state={}, stage="complete")
    ended = store.set_cycle_state(
        event["id"], state=event["state"], status="blocked", stage="evaluation_failed",
        finished=True,
    )
    with pytest.raises(ValueError, match="immutable"):
        store.set_cycle_state(ended["id"], state=ended["state"], stage="parallel_research")
    assert _registered(store, dataset) == ended
    next_event = _registered(store, dataset, event_key="next")
    assert next_event["id"] != ended["id"]
    assert store.get_cycle(ended["id"]) == ended


def test_database_active_scheduled_cycle_blocks_manual_registration(database_url, tmp_path):
    store = AutopilotStore(database_url)
    dataset = _dataset(tmp_path)
    scheduled = store.ensure_cycle(dataset, config_revision=1)
    with pytest.raises(ValueError, match="owns this horizon"):
        _registered(store, dataset)
    assert store.get_cycle(scheduled["id"]) == scheduled


def test_database_superseded_scheduled_history_does_not_own_horizon(database_url, tmp_path):
    store = AutopilotStore(database_url)
    dataset = _dataset(tmp_path)
    scheduled = store.ensure_cycle(dataset, config_revision=1)
    historical = store.set_cycle_state(
        scheduled["id"], state={**scheduled["state"], "historical_results_only": True},
        status="paused", stage="superseded", finished=True,
    )
    event = _registered(store, dataset)
    assert event["id"] != historical["id"]
    assert store.get_cycle(historical["id"]) == historical
    paused = store.set_cycle_state(
        event["id"], state=event["state"], status="paused", stage="operator_paused"
    )
    with pytest.raises(ValueError, match="owns this horizon"):
        _registered(store, dataset, event_key="another")
    assert store.get_cycle(event["id"]) == paused


def test_database_model_producer_lineage_remains_required_by_quant_gate(database_url, tmp_path):
    dataset = _dataset(tmp_path)
    run = ResearchStore(database_url).create_run(
        kind="platform_model_feature_screen_test", objective="verify actual model lineage binding",
        dataset=dataset["name"], requested_by="test", budget={}, config={}, artifact_path=tmp_path,
    )
    source = tmp_path / "synthetic_model.py"
    source.write_text("# Synthetic lineage fixture; never executed.\n", encoding="utf-8")
    candidates = RDAGentCandidateStore(database_url)
    artifact = candidates.register_run_artifact(
        research_run_id=run["id"], artifact_type="model_code", storage_path=source,
        producer="test", actor="test", contract_version="synthetic-model-code-v1",
    )
    controller = _controller(AutopilotStore(database_url), dataset)
    controller.engine = controller.store.engine
    for index, lineage in enumerate((None, "c" * 64, "b" * 64)):
        candidate = candidates.create_model_candidate(
            research_run_id=run["id"], name=f"lineage-{index}", description="synthetic fixture",
            model_type="Tabular", code_artifact_id=artifact["id"], architecture={},
            model_hyperparameters={}, training_hyperparameters={}, feature_set_id="qlib-alpha158",
            dataset=dataset["name"], dataset_identity_sha256="a" * 64,
            dataset_lineage_id=lineage, pre_final_end=date(2025, 1, 1),
            final_oos_start=date(2025, 2, 1), final_oos_end=date(2026, 1, 1),
        )
        # Fixture admission isolates the existing SQL membership gate. No model
        # is executed and this is not evidence of independent admission.
        with controller.engine.begin() as connection:
            connection.execute(update(model_candidates).where(
                model_candidates.c.id == candidate["id"]
            ).values(status="research_admitted"))
        cycle = _cycle(dataset)
        _add_champion(cycle)
        cycle["state"]["prediction_champion"]["candidate_id"] = candidate["id"]
        assert controller._quant_due(
            cycle, dataset, datetime.now(UTC), normalize_autopilot_config(), input_sha256="e" * 64
        ) is (lineage == "b" * 64)
