from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from threading import Event
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql

from quant_data.database import schedule_runs, schedules
from quant_platform.autopilot import (
    LEGACY_MANUAL_RESEARCH_EVENT_CONTRACT,
    MANUAL_RESEARCH_EVENT_CONTRACT,
    AutopilotController,
    _event_dataset_binding,
    _manual_research_event,
    normalize_autopilot_config,
)
from quant_platform.fin_strategy_research_completion import (
    COMPLETION_ACTOR,
    COMPLETION_KEY,
    COMPLETION_NAME_PREFIX,
    ManagedResearchCompletion,
    build_completion_binding,
    completed_event,
    require_completion_payload,
    select_completion_dataset,
)
from quant_platform.fin_strategy_schedule import (
    build_managed_fin_strategy_schedule_specs,
    canonical_sha256,
)
from quant_platform.schedule_store import ScheduleStore


def inputs():
    dataset = {
        "name": "frozen-0904", "path": "/data/qlib/frozen-0904", "end_date": "2026-09-04",
        "ready": True, "reproducible": True, "output_files_verified": True, "frequency": "day",
        "lineage_id": "b" * 64,
        "provenance": {"dataset_identity_sha256": "a" * 64, "dataset_lineage_id": "b" * 64},
    }
    request = {
        "event_key": "full-mainline", "horizon_profile": "short_1_5d", "actor": "test",
        "reason": "Complete one governed mainline", "quant_loop_n": 10, "quant_duration": "1h",
        "completion_mode": "managed_fin_strategy",
    }
    event = {
        "contract_version": MANUAL_RESEARCH_EVENT_CONTRACT, "request": request,
        "config": normalize_autopilot_config({"quant_loop_n": 10, "quant_duration": "1h"}),
        "config_revision": 3, "dataset": _event_dataset_binding(dataset),
    }
    event["sha256"] = canonical_sha256(event)
    cycle = {
        "id": "complete-cycle", "research_event_key": "manual:full-mainline",
        "dataset": dataset["name"], "dataset_identity_sha256": "a" * 64,
        "dataset_lineage_id": "b" * 64, "horizon_profile": "short_1_5d", "config_revision": 3,
        "status": "succeeded", "stage": "research_complete", "finished_at": "2026-09-07",
        "state": {"research_event": event, "dataset_end_date": "2026-09-04"},
        "branches": [{"scenario": "fin_quant", "status": "succeeded", "research_run_id": "q"}],
    }
    candidates = [{"id": "joint", "research_run_id": "q", "bundle_manifest_sha256": "c" * 64,
                   "admission_evidence_sha256": "d" * 64}]
    evidence = {
        "dataset": dataset["name"], "dataset_identity_sha256": "a" * 64,
        "horizon_profile": "short_1_5d", "final_oos_opened": False,
        "research_screening_only": True, "not_capital_confirmation": True,
        "eligible_candidates": [{"kind": "joint", "candidate_id": "joint"}],
        "selected_candidate_id": "joint",
    }
    selection = {"champion_selection_evidence": evidence,
                 "champion_selection_evidence_sha256": canonical_sha256(evidence)}
    spec = build_managed_fin_strategy_schedule_specs()[0]
    binding = build_completion_binding(cycle, candidates, selection,
                                       spec["payload"]["managed_fin_strategy"])
    return cycle, dataset, candidates, selection, spec, binding


@pytest.mark.no_database
@pytest.mark.parametrize("change", [
    {"status": "active"}, {"status": "paused"}, {"status": "blocked"},
    {"finished_at": None}, {"stage": "complete"}, {"branches": []},
    {"branches": [{"scenario": "fin_quant", "status": "running"}]},
    {"branches": [{"scenario": "fin_quant", "status": "failed"}]},
])
def test_only_terminal_successful_joint_activity_is_eligible(change):
    cycle, *_ = inputs()
    assert completed_event({**cycle, **change}) is None


@pytest.mark.no_database
def test_scheduled_activity_and_changed_frozen_request_never_handoff():
    cycle, *_ = inputs()
    scheduled = {**cycle, "research_event_key": "scheduled", "state": {}}
    assert completed_event(scheduled) is None
    cycle["state"]["research_event"]["request"]["quant_loop_n"] = 9
    with pytest.raises(ValueError):
        completed_event(cycle)


@pytest.mark.no_database
def test_research_only_default_and_legacy_v1_have_no_completion_authority():
    cycle, *_ = inputs()
    event = cycle["state"]["research_event"]
    event["request"]["completion_mode"] = "research_only"
    event["sha256"] = canonical_sha256({k: v for k, v in event.items() if k != "sha256"})
    assert completed_event(cycle) is None
    event["contract_version"] = LEGACY_MANUAL_RESEARCH_EVENT_CONTRACT
    event["request"].pop("completion_mode")
    event["sha256"] = canonical_sha256({k: v for k, v in event.items() if k != "sha256"})
    before = deepcopy(cycle)
    assert _manual_research_event(cycle) == event
    assert completed_event(cycle) is None
    controller = AutopilotController.__new__(AutopilotController)
    controller.store = SimpleNamespace(get_research_event=lambda _: cycle)
    assert controller.start_research_event(**event["request"]) is cycle
    with pytest.raises(ValueError, match="another request"):
        controller.start_research_event(**event["request"], completion_mode="managed_fin_strategy")
    assert cycle == before


@pytest.mark.no_database
@pytest.mark.parametrize("field,value", [
    ("dataset_identity_sha256", "f" * 64), ("horizon_profile", "swing_1_6m"),
    ("final_oos_opened", True), ("research_screening_only", False),
    ("not_capital_confirmation", False), ("eligible_candidates", []),
])
def test_completion_rejects_wrong_or_incomplete_independent_selection(field, value):
    cycle, _, candidates, selection, spec, _ = inputs()
    selection["champion_selection_evidence"][field] = value
    selection["champion_selection_evidence_sha256"] = canonical_sha256(
        selection["champion_selection_evidence"]
    )
    with pytest.raises(ValueError):
        build_completion_binding(cycle, candidates, selection,
                                 spec["payload"]["managed_fin_strategy"])


@pytest.mark.no_database
def test_payload_hash_binds_scope_and_does_not_consume_calendar():
    cycle, _, _, _, spec, binding = inputs()
    original = deepcopy(cycle)
    payload = {**spec["payload"], COMPLETION_KEY: binding}
    assert require_completion_payload(payload) == binding
    assert "calendar_event" not in binding
    assert cycle == original
    payload[COMPLETION_KEY]["dataset"]["end_date"] = "2026-09-07"
    with pytest.raises(ValueError):
        require_completion_payload(payload)


@pytest.mark.no_database
@pytest.mark.parametrize("field,value", [("loop_n", 10), ("duration", "2h"),
                                         ("objective", "different hypothesis"),
                                         ("requested_by", "different actor")])
def test_owned_schedule_cannot_expand_or_replace_the_frozen_managed_definition(field, value):
    *_, spec, binding = inputs()
    payload = {**spec["payload"], COMPLETION_KEY: binding, field: value}
    with pytest.raises(ValueError, match="definition changed"):
        require_completion_payload(payload)


@pytest.mark.no_database
def test_later_publication_never_replaces_frozen_0904_data(monkeypatch):
    import quant_platform.fin_strategy_schedule as module

    _, dataset, _, _, _, binding = inputs()
    monkeypatch.setattr(module, "require_daily_qlib_contract", lambda _: None)
    later = {**dataset, "name": "later-0907", "end_date": "2026-09-07"}
    assert select_completion_dataset(binding, [later, dataset])["name"] == dataset["name"]
    with pytest.raises(ValueError):
        select_completion_dataset(binding, [later])
    with pytest.raises(ValueError):
        select_completion_dataset(binding, [{**dataset, "path": "/data/replaced"}])
    with pytest.raises(ValueError):
        select_completion_dataset(binding, [{**dataset, "output_files_verified": False}])


@pytest.mark.no_database
def test_normal_calendar_payload_remains_unmodified():
    *_, spec, _ = inputs()
    assert require_completion_payload(spec["payload"]) is None


@pytest.mark.no_database
def test_scheduler_completion_keeps_old_calendar_and_frozen_signal(monkeypatch, tmp_path):
    import quant_platform.fin_strategy_schedule as managed_module
    import quant_platform.scheduler as scheduler_module

    class ReachedNormalSignalGate(Exception):
        pass

    _, dataset, _, selection, spec, binding = inputs()
    root = tmp_path / dataset["name"]
    (root / "calendars").mkdir(parents=True)
    (root / "calendars/day.txt").write_text("2026-09-03\n2026-09-04\n", encoding="utf-8")
    dataset["path"] = str(root)
    binding["dataset"]["path"] = str(root)
    binding["binding_sha256"] = canonical_sha256(
        {k: v for k, v in binding.items() if k != "binding_sha256"}
    )
    later = {**dataset, "name": "later", "end_date": "2026-09-07"}
    monkeypatch.setattr(managed_module, "require_daily_qlib_contract", lambda _: None)
    monkeypatch.setattr(scheduler_module, "list_qlib_datasets", lambda _: [later, dataset])
    monkeypatch.setattr(ManagedResearchCompletion, "verify", lambda *_: binding)
    monkeypatch.setattr(scheduler_module, "load_trade_calendar_open_days",
                        lambda _: pytest.fail("completion must not fabricate a calendar event"))
    periods = {"train_start": "2008-01-02", "train_end": "2022-01-03",
               "valid_start": "2022-01-04", "valid_end": "2024-09-02",
               "test_start": "2024-09-03", "test_end": "2026-09-04"}
    calls = []

    def window(selected, calendar, **_):
        assert selected["name"] == dataset["name"]
        assert "2026-09-07" not in calendar
        return periods, {"research_window_contract_sha256": "e" * 64}

    def signal(**kwargs):
        assert kwargs["dataset"] == dataset["name"]
        assert kwargs["champion_selection"] == selection
        calls.append(kwargs)
        raise ReachedNormalSignalGate

    monkeypatch.setattr(scheduler_module, "resolve_research_window_contract", window)
    monkeypatch.setattr(scheduler_module, "research_feature_set_for_champion_selection",
                        lambda feature, selected: feature)
    monkeypatch.setattr(scheduler_module, "build_strategy_research_signal_binding", signal)
    engine = scheduler_module.SchedulerEngine.__new__(scheduler_module.SchedulerEngine)
    engine.settings = SimpleNamespace(rdagent_max_loops=10, rdagent_max_duration="2h",
                                      data_root=tmp_path)
    engine.autopilot = SimpleNamespace()
    engine.schedules = SimpleNamespace(engine=None)
    engine._managed_fin_strategy_incumbent = lambda _: None
    observed = []
    engine._consumed_managed_fin_strategy_trigger_ids = lambda ids: observed.extend(ids) or set()
    run = {"payload": {**spec["payload"], COMPLETION_KEY: binding}, "timezone": "Asia/Shanghai",
           "trading_days_only": True, "id": "run"}
    # 9/8 is neither the frozen final data session nor the weekly boundary.
    with pytest.raises(ReachedNormalSignalGate):
        engine._enqueue_research_bound(run, datetime(2026, 9, 8, 8, tzinfo=UTC))
    assert len(calls) == 1 and len(observed) == 1


@pytest.mark.no_database
def test_scheduler_paused_completion_never_reaches_dispatch(monkeypatch):
    import quant_platform.scheduler as module

    *_, spec, binding = inputs()

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    engine = module.SchedulerEngine.__new__(module.SchedulerEngine)
    engine.schedules = SimpleNamespace(engine=SimpleNamespace(begin=Connection))
    locked = []
    engine.autopilot = SimpleNamespace(store=SimpleNamespace(
        _lock_horizon=lambda connection, horizon: locked.append(horizon)
    ))
    monkeypatch.setattr(ManagedResearchCompletion, "dispatch_allowed", lambda *_, **__: False)
    engine._enqueue_research_bound = lambda *_: pytest.fail("paused completion dispatched")
    run = {"payload": {**spec["payload"], COMPLETION_KEY: binding}, "schedule_id": "one"}
    with pytest.raises(module.ScheduleRunWaiting, match="paused"):
        engine._enqueue_research(run, datetime.now(UTC))
    assert locked == ["short_1_5d"]


@pytest.mark.no_database
def test_bounded_queries_compile_for_generic_json_and_real_tables():
    queries = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def scalar(self, statement):
            queries.append(statement)
            return None

        def scalars(self, statement):
            queries.append(statement)
            return []

        def execute(self, statement):
            queries.append(statement)
            return SimpleNamespace(mappings=lambda: [])

    service = ManagedResearchCompletion(SimpleNamespace(
        config=lambda: ({"enabled": True}, 3), settings=SimpleNamespace(rdagent_enabled=True),
    ), SimpleNamespace(engine=SimpleNamespace(connect=Connection)))
    assert service.pending_cycle_ids() == []
    assert service.enabled_and_idle("short_1_5d")
    assert service._candidates(inputs()[0]) == []
    for query in queries:
        str(query.compile(dialect=postgresql.dialect()))
    assert len(queries) == 4


def service_fixture(database_url, monkeypatch):
    cycle, _, _, _, _, binding = inputs()
    store = ScheduleStore(database_url)
    controller = SimpleNamespace(store=SimpleNamespace(get_cycle=lambda _: deepcopy(cycle)))
    service = ManagedResearchCompletion(controller, store)
    monkeypatch.setattr(service, "enabled_and_idle", lambda _: True)
    monkeypatch.setattr(service, "bind", lambda *_: deepcopy(binding))
    return service, store, cycle, binding


def test_registration_is_concurrently_idempotent_and_never_retries_terminal(
    database_url, monkeypatch,
):
    service, store, cycle, _ = service_fixture(database_url, monkeypatch)
    now = datetime(2026, 9, 7, 8, tzinfo=UTC)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: service.register(cycle["id"], now), range(2)))
    assert first["id"] == second["id"]
    schedule = store.get(first["schedule_id"])
    assert schedule["status"] == "paused"
    assert schedule["created_by"] == COMPLETION_ACTOR
    store.finish_run(first["id"], "failed", message="diagnostic terminal", now=now)
    assert service.register(cycle["id"], now + timedelta(hours=1))["id"] == first["id"]
    with store.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(schedules)) == 1
        assert connection.scalar(select(func.count()).select_from(schedule_runs)) == 1


def test_later_operator_pause_is_not_the_owned_oneshot_pause(database_url, monkeypatch):
    service, store, cycle, _ = service_fixture(database_url, monkeypatch)
    now = datetime(2026, 9, 7, 8, tzinfo=UTC)
    run = service.register(cycle["id"], now)
    assert service.dispatch_allowed(run["schedule_id"])
    store.set_status(run["schedule_id"], "paused", now=now + timedelta(microseconds=1))
    assert not service.dispatch_allowed(run["schedule_id"])
    assert service.register(cycle["id"], now)["id"] == run["id"]
    assert store.get_by_name(COMPLETION_NAME_PREFIX + cycle["id"])["status"] == "paused"


def test_global_pause_prevents_registration_and_dispatch(database_url, monkeypatch):
    service, store, cycle, _ = service_fixture(database_url, monkeypatch)
    monkeypatch.setattr(service, "enabled_and_idle", lambda _: False)
    assert service.register(cycle["id"], datetime.now(UTC)) is None
    assert store.list() == []


def test_atomic_paused_creation_never_materializes_a_calendar_slot(database_url, monkeypatch):
    service, store, cycle, _ = service_fixture(database_url, monkeypatch)
    now = datetime(2026, 9, 7, 14, 59, 59, tzinfo=UTC)
    first = service.register(cycle["id"], now)
    assert store.materialize_due(now + timedelta(days=2)) == 0
    assert [run["id"] for run in store.list_runs()] == [first["id"]]


def test_dispatch_row_lock_serializes_concurrent_operator_pause(database_url, monkeypatch):
    service, store, cycle, _ = service_fixture(database_url, monkeypatch)
    now = datetime(2026, 9, 7, 8, tzinfo=UTC)
    run = service.register(cycle["id"], now)
    entered = Event()
    release = Event()

    def dispatch():
        with store.engine.begin() as connection:
            assert service.dispatch_allowed(run["schedule_id"], connection=connection,
                                            expected_payload=run["payload"])
            entered.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        dispatching = pool.submit(dispatch)
        assert entered.wait(5)
        pausing = pool.submit(store.set_status, run["schedule_id"], "paused",
                              now=now + timedelta(seconds=1))
        try:
            with pytest.raises(TimeoutError):
                pausing.result(timeout=0.1)
        finally:
            release.set()
        dispatching.result(timeout=5)
        pausing.result(timeout=5)
    assert not service.dispatch_allowed(run["schedule_id"])


def test_completion_wait_beyond_three_days_preserves_time_and_terminal_failure(
    database_url, monkeypatch,
):
    from quant_platform.scheduler import SchedulerEngine, ScheduleRunWaiting

    service, store, cycle, _ = service_fixture(database_url, monkeypatch)
    now = datetime(2026, 9, 7, 8, tzinfo=UTC)
    original = service.register(cycle["id"], now)
    late = now + timedelta(days=4)
    claimed = store.claim_run(now=late)
    engine = SchedulerEngine.__new__(SchedulerEngine)
    engine.schedules = store
    engine.settings = SimpleNamespace(scheduler_poll_seconds=15)
    engine.alerts = SimpleNamespace(create=lambda **_: None)

    def blocked(*_):
        raise ScheduleRunWaiting("upstream horizon still busy")

    engine._enqueue_research = blocked
    engine._process_run_guarded(claimed, late)
    waiting = store.get_run(original["id"])
    assert waiting["status"] == "waiting"
    assert waiting["scheduled_for"] == original["scheduled_for"]
    assert datetime.fromisoformat(waiting["lease_until"]) > late

    def failed(*_):
        raise ValueError("real invalid independent artifact")

    engine._enqueue_research = failed
    next_time = late + timedelta(minutes=2)
    claimed = store.claim_run(now=next_time)
    engine._process_run_guarded(claimed, next_time)
    assert store.get_run(original["id"])["status"] == "failed"
    assert store.claim_run(now=next_time + timedelta(days=10)) is None
    assert service.register(cycle["id"], next_time)["id"] == original["id"]
