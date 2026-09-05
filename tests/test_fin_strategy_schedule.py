from __future__ import annotations

from copy import deepcopy
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from quant_data.config import Settings
from quant_platform.feature_set_registry import get_feature_set
from quant_platform.fin_strategy_schedule import (
    LATEST_REPRODUCIBLE_DAILY_DATASET,
    build_managed_fin_strategy_schedule_specs,
    managed_fin_strategy_due_event,
    reconcile_managed_fin_strategy_schedules,
    select_latest_reproducible_daily_dataset,
    validate_managed_fin_strategy_payload,
)
from quant_platform.rdagent_scenarios import get_rdagent_scenario
from quant_platform.research_store import ResearchStore
from quant_platform.schedule_store import ScheduleStore
from quant_platform.scheduler import SchedulerEngine
from quant_platform.strategy_recipes import get_strategy_recipe


def _managed_lookup_engine(records: list[dict]) -> SchedulerEngine:
    """Exercise the production query bindings against stored run identities."""

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def scalar(self, statement):
            params = statement.compile(dialect=postgresql.dialect()).params
            return next(
                (
                    row["id"]
                    for row in records
                    if row["kind"] == params["kind_1"]
                    and params["param_2"][0] in row["trigger_ids"]
                ),
                None,
            )

        def execute(self, statement):
            params = statement.compile(dialect=postgresql.dialect()).params
            rows = [
                SimpleNamespace(**row)
                for row in records
                if row["kind"] == params["kind_1"]
                and row["run_sha256"] == params["param_2"]
            ]
            return SimpleNamespace(all=lambda: rows)

    engine = object.__new__(SchedulerEngine)
    engine.jobs = SimpleNamespace(engine=SimpleNamespace(connect=Connection))
    return engine


def _stored_managed_run(**overrides) -> dict:
    return {
        "id": "original-run",
        "kind": get_rdagent_scenario("fin_strategy").research_kind,
        "job_id": "original-job",
        "status": "succeeded",
        "error": None,
        "has_artifacts": True,
        "has_payload_jobs": True,
        "trigger_ids": ["a" * 64],
        "run_sha256": "b" * 64,
        **overrides,
    }


@pytest.mark.no_database
def test_managed_lookups_use_the_persisted_scenario_kind() -> None:
    engine = _managed_lookup_engine(
        [
            _stored_managed_run(),
            _stored_managed_run(id="legacy-decoy", kind="strategy", job_id="legacy-job"),
        ]
    )

    assert engine._consumed_managed_fin_strategy_trigger_ids(["a" * 64]) == {"a" * 64}
    assert engine._existing_managed_fin_strategy_run("b" * 64) == {
        "id": "original-run", "job_id": "original-job", "status": "succeeded"
    }


@pytest.mark.no_database
def test_terminal_unattached_run_still_consumes_its_trigger() -> None:
    engine = _managed_lookup_engine(
        [_stored_managed_run(job_id=None, status="failed", has_payload_jobs=False)]
    )

    assert engine._consumed_managed_fin_strategy_trigger_ids(["a" * 64, "c" * 64]) == {
        "a" * 64
    }


@pytest.mark.no_database
def test_existing_job_owner_survives_empty_failed_duplicate_audit_records() -> None:
    records = [_stored_managed_run()]
    for index, error in enumerate(
        [
            "idempotency key is already bound to a different job payload",
            "strategy validation window is shorter than its preregistered OOS",
        ]
    ):
        records.append(
            _stored_managed_run(
                id=f"empty-failure-{index}", job_id=None, status="failed", error=error,
                has_artifacts=False, has_payload_jobs=False,
            )
        )
    before = deepcopy(records)
    engine = _managed_lookup_engine(records)

    assert engine._existing_managed_fin_strategy_run("b" * 64)["id"] == "original-run"
    assert records == before


@pytest.mark.no_database
@pytest.mark.parametrize(
    "overrides",
    [
        {"has_artifacts": True},
        {"has_payload_jobs": True},
        {"status": "queued"},
        {"status": "cancelled"},
        {"job_id": "second-job"},
    ],
)
def test_existing_managed_run_rejects_duplicates_that_may_own_evidence(overrides) -> None:
    duplicate = _stored_managed_run(
        id="duplicate", job_id=None, status="failed",
        has_artifacts=False, has_payload_jobs=False,
    )
    duplicate.update(overrides)
    engine = _managed_lookup_engine([_stored_managed_run(), duplicate])

    with pytest.raises(ValueError, match="identity is not unique"):
        engine._existing_managed_fin_strategy_run("b" * 64)


@pytest.mark.no_database
def test_empty_failures_cannot_be_selected_as_an_original_job_owner() -> None:
    engine = _managed_lookup_engine(
        [
            _stored_managed_run(
                id=f"empty-{index}", job_id=None, status="failed",
                has_artifacts=False, has_payload_jobs=False,
            )
            for index in range(2)
        ]
    )

    with pytest.raises(ValueError, match="identity is not unique"):
        engine._existing_managed_fin_strategy_run("b" * 64)


@pytest.mark.no_database
def test_managed_specs_bind_current_recipe_and_minimal_horizon_features() -> None:
    specs = build_managed_fin_strategy_schedule_specs()

    assert len(specs) == 3
    assert {item["name"] for item in specs} == {
        "QuantLab / fin_strategy / short",
        "QuantLab / fin_strategy / swing",
        "QuantLab / fin_strategy / long",
    }
    for spec in specs:
        payload = spec["payload"]
        managed = validate_managed_fin_strategy_payload(payload)
        assert managed is not None
        recipe = get_strategy_recipe(str(managed["recipe_id"]))
        feature_set = get_feature_set(str(managed["feature_set_id"]))
        assert payload["dataset"] == LATEST_REPRODUCIBLE_DAILY_DATASET
        assert payload["dataset_policy"] == "latest_reproducible_daily"
        assert managed["recipe_version"] == recipe["version"]
        assert feature_set["features"] == {
            str(item["id"]): str(item["qlib_expression"])
            for item in recipe["factor_baseline"]
        }
        assert managed["contract_version"] == "managed-fin-strategy-schedules-v2"
        assert managed["trigger_policy"]["drift_trigger"] == {
            "contract_version": "managed-fin-strategy-drift-trigger-v1",
            "source": "strategy_health_snapshots",
            "metric": "feature_drift",
            "threshold_key": "watch_feature_drift",
            "comparison": "greater_than_or_equal",
            "required_hard_gates": ["data_integrity_ok", "ledger_reconciled"],
            "dedupe": "contiguous_breach_episode",
        }
    short = next(
        item
        for item in specs
        if item["payload"]["managed_fin_strategy"]["recipe_id"]
        == "short_relative_strength"
    )
    short_features = get_feature_set(short["payload"]["feature_set_id"])[
        "features"
    ]
    assert all("$fund_" not in expression for expression in short_features.values())


@pytest.mark.no_database
def test_managed_schedule_rejects_legacy_or_tampered_drift_policy() -> None:
    payload = deepcopy(build_managed_fin_strategy_schedule_specs()[0]["payload"])
    payload["managed_fin_strategy"]["trigger_policy"][
        "drift_trigger"
    ] = "not_implemented"

    with pytest.raises(ValueError, match="contract digest changed"):
        validate_managed_fin_strategy_payload(payload)


@pytest.mark.no_database
def test_managed_cadences_use_persisted_trading_sessions() -> None:
    specs = build_managed_fin_strategy_schedule_specs()
    by_recipe = {
        item["payload"]["managed_fin_strategy"]["recipe_id"]: item["payload"][
            "managed_fin_strategy"
        ]
        for item in specs
    }
    sessions = [
        date(2025, 12, 31),
        date(2026, 1, 2),
        date(2026, 1, 5),
        date(2026, 2, 2),
        date(2026, 4, 1),
        date(2026, 5, 4),
        date(2026, 9, 1),
        date(2026, 11, 2),
    ]

    weekly = managed_fin_strategy_due_event(
        by_recipe["short_relative_strength"],
        scheduled_date=date(2026, 1, 5),
        trading_days=sessions,
    )
    monthly = managed_fin_strategy_due_event(
        by_recipe["swing_trend"],
        scheduled_date=date(2026, 2, 2),
        trading_days=sessions,
    )
    quarterly = managed_fin_strategy_due_event(
        by_recipe["long_quality_value"],
        scheduled_date=date(2026, 4, 1),
        trading_days=sessions,
    )
    post_report = managed_fin_strategy_due_event(
        by_recipe["long_quality_value"],
        scheduled_date=date(2026, 5, 4),
        trading_days=sessions,
    )
    closed = managed_fin_strategy_due_event(
        by_recipe["short_relative_strength"],
        scheduled_date=date(2026, 1, 3),
        trading_days=sessions,
    )

    assert weekly == {"due": True, "reason": "calendar_due", "event": "week:2026-W02"}
    assert monthly["event"] == "month:2026-02"
    assert quarterly["event"] == "quarter:2026-Q2"
    assert post_report["event"] == "post_report:2026:annual_q1"
    assert closed == {"due": False, "reason": "exchange_closed", "event": None}


@pytest.mark.no_database
def test_latest_daily_policy_rejects_unsealed_publications(monkeypatch) -> None:
    from quant_platform import fin_strategy_schedule as module

    monkeypatch.setattr(module, "require_daily_qlib_contract", lambda _value: None)
    base = {
        "ready": True,
        "reproducible": True,
        "output_files_verified": True,
        "frequency": "day",
        "trading_days": 4000,
        "lineage_id": "b" * 64,
        "provenance": {
            "dataset_identity_sha256": "a" * 64,
            "dataset_lineage_id": "b" * 64,
        },
    }
    selected = select_latest_reproducible_daily_dataset(
        [
            {**base, "name": "older", "end_date": "2026-08-27"},
            {
                **base,
                "name": "latest-broken",
                "end_date": "2026-08-29",
                "output_files_verified": False,
            },
            {**base, "name": "latest-good", "end_date": "2026-08-28"},
        ]
    )

    assert selected["name"] == "latest-good"
    assert selected["dataset_identity_sha256"] == "a" * 64
    assert selected["dataset_lineage_id"] == "b" * 64


@pytest.mark.no_database
def test_scheduler_reconcile_lanes_are_throttled_and_isolated(monkeypatch) -> None:
    from quant_platform import scheduler as module

    current = datetime(2026, 8, 29, 2, 0, tzinfo=UTC)
    engine = object.__new__(SchedulerEngine)
    engine._last_fin_strategy_schedule_reconcile_at = None
    engine.schedules = object()
    engine.settings = SimpleNamespace(rdagent_enabled=True)

    class Alerts:
        records: list[dict] = []

        def create(self, **values) -> None:
            self.records.append(values)

    engine.alerts = Alerts()
    calls = {"schedules": 0}

    def schedules(*_args, **_kwargs) -> list[dict]:
        calls["schedules"] += 1
        return [{}, {}, {}]

    monkeypatch.setattr(module, "reconcile_managed_fin_strategy_schedules", schedules)

    assert engine._reconcile_default_fin_strategy_schedules(current) == 3
    assert engine._reconcile_default_fin_strategy_schedules(current) == 0
    assert not hasattr(SchedulerEngine, "_reconcile_transparent_baselines")
    assert calls == {"schedules": 1}
    assert engine.alerts.records == []


def test_managed_schedule_reconcile_is_database_idempotent(database_url: str) -> None:
    store = ScheduleStore(database_url)
    current = datetime(2026, 8, 29, 2, 0, tzinfo=UTC)

    first = reconcile_managed_fin_strategy_schedules(
        store,
        enabled=True,
        actor="test",
        now=current,
    )
    second = reconcile_managed_fin_strategy_schedules(
        store,
        enabled=True,
        actor="test",
        now=current,
    )

    assert len(first) == len(second) == 3
    assert {item["id"] for item in first} == {item["id"] for item in second}
    assert all(item["status"] == "active" for item in second)
    assert all(
        validate_managed_fin_strategy_payload(item["payload"]) is not None
        for item in second
    )


def test_managed_trigger_consumption_includes_unattached_research_run(
    database_url: str, tmp_path: Path
) -> None:
    trigger_id = "a" * 64
    research = ResearchStore(database_url)
    created = research.create_run(
        kind=get_rdagent_scenario("fin_strategy").research_kind,
        objective="test managed trigger consumption",
        dataset="test-dataset",
        requested_by="test",
        budget={"loop_n": 1, "duration": "30m"},
        config={
            "managed_fin_strategy_run": {
                "contract_version": "managed-fin-strategy-run-v2",
                "trigger_ids": [trigger_id],
            }
        },
        artifact_path=tmp_path,
    )
    assert created["job_id"] is None
    engine = object.__new__(SchedulerEngine)
    engine.jobs = SimpleNamespace(engine=research.engine)

    assert engine._consumed_managed_fin_strategy_trigger_ids(
        [trigger_id, "b" * 64]
    ) == {trigger_id}


@pytest.mark.parametrize(
    "incumbent_exists",
    [True, False],
    ids=["approved-incumbent", "transparent-baseline-cold-start"],
)
def test_managed_run_freezes_dataset_window_feature_and_research_control(
    database_url: str,
    tmp_path: Path,
    monkeypatch,
    incumbent_exists: bool,
) -> None:
    from quant_platform import fin_strategy_schedule as schedule_module
    from quant_platform import scheduler as scheduler_module

    settings = Settings(
        api_url="https://api.tushare.pro",
        token="test-token",
        data_root=tmp_path / "data",
        database_url=database_url,
        embedded_worker=False,
        research_asset_auto_enabled=False,
    )
    dataset_root = settings.data_root / "qlib" / "daily-20260105"
    (dataset_root / "calendars").mkdir(parents=True)
    (dataset_root / "calendars" / "day.txt").write_text(
        "2025-12-31\n2026-01-02\n2026-01-05\n",
        encoding="utf-8",
    )
    dataset = {
        "name": "daily-20260105",
        "path": str(dataset_root),
        "ready": True,
        "reproducible": True,
        "output_files_verified": True,
        "frequency": "day",
        "start_date": "2008-01-02",
        "end_date": "2026-01-05",
        "trading_days": 4400,
        "lineage_id": "b" * 64,
        "provenance": {
            "dataset_identity_sha256": "a" * 64,
            "dataset_lineage_id": "b" * 64,
        },
    }
    periods = {
        "train_start": "2008-01-02",
        "train_end": "2022-12-30",
        "valid_start": "2023-01-03",
        "valid_end": "2024-12-31",
        "test_start": "2025-01-02",
        "test_end": "2026-01-05",
    }
    resolution = {
        "evaluation_profiles": [],
        "research_window_contract": {"contract_version": "test-window-v1"},
        "research_window_contract_sha256": "c" * 64,
        "label_horizons_sessions": [1, 2, 3, 5],
    }
    monkeypatch.setattr(
        schedule_module, "require_daily_qlib_contract", lambda _value: None
    )
    monkeypatch.setattr(
        scheduler_module, "list_qlib_datasets", lambda _root: [dataset]
    )
    monkeypatch.setattr(
        scheduler_module,
        "resolve_research_window_contract",
        lambda *_args, **_kwargs: (dict(periods), dict(resolution)),
    )
    monkeypatch.setattr(
        scheduler_module,
        "load_trade_calendar_open_days",
        lambda _root: {date(2026, 1, 2), date(2026, 1, 5)},
    )
    monkeypatch.setattr(
        scheduler_module, "probe_rdagent", lambda *_args, **_kwargs: {"status": "ok"}
    )
    monkeypatch.setattr(
        scheduler_module, "require_ready_scenario", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        scheduler_module,
        "expected_rdagent_runtime_identity",
        lambda *_args, **_kwargs: {"source_tree_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        scheduler_module,
        "resolve_rdagent_assets",
        lambda *_args, **_kwargs: {"manifest_sha256": {}},
    )

    current = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
    spec = build_managed_fin_strategy_schedule_specs()[0]
    store = ScheduleStore(database_url)
    schedule = store.upsert_managed(
        **spec,
        actor="test",
        enabled=True,
        now=current,
    )
    pending = store.trigger_now(str(schedule["id"]), actor="test", now=current)
    claimed = store.claim_run(now=current)
    assert claimed is not None and claimed["id"] == pending["id"]
    engine = SchedulerEngine(settings)
    incumbent = (
        {
            "id": "1" * 32,
            "status": "approved",
            "promotion_stage": "paper",
            "horizon_profile": "short_1_5d",
            "horizon_contract_sha256": "e" * 64,
            "version": 1,
        }
        if incumbent_exists
        else None
    )
    monkeypatch.setattr(
        engine,
        "_managed_fin_strategy_incumbent",
        lambda _horizon: incumbent,
    )
    drift_trigger_id = "f" * 64

    def drift_event(*_args, **_kwargs):
        assert incumbent_exists
        return {
            "due": True,
            "reason": "feature_drift_episode_due",
            "trigger_id": drift_trigger_id,
            "event": {
                "contract_version": "strategy-feature-drift-episode-v1",
                "kind": "feature_drift_episode",
                "trigger_id": drift_trigger_id,
            },
        }

    monkeypatch.setattr(
        engine,
        "_managed_fin_strategy_drift_event",
        drift_event,
    )

    engine._process_run(claimed, current)

    finished = store.get_run(str(pending["id"]))
    assert finished["status"] == "enqueued"
    research_run = ResearchStore(database_url).list_runs()[0]
    assert research_run["dataset"] == "daily-20260105"
    assert research_run["config"]["dataset_binding"] == {
        "contract_version": "scheduled-research-dataset-binding-v1",
        "name": "daily-20260105",
        "identity_sha256": "a" * 64,
        "lineage_id": "b" * 64,
        "start_date": "2008-01-02",
        "end_date": "2026-01-05",
    }
    managed_run = research_run["config"]["managed_fin_strategy_run"]
    assert managed_run["contract_version"] == "managed-fin-strategy-run-v3"
    assert managed_run["calendar_event"] == "week:2026-W02"
    assert len(managed_run["trigger_ids"]) == (2 if incumbent_exists else 1)
    assert (drift_trigger_id in managed_run["trigger_ids"]) is incumbent_exists
    calendar_trigger = next(
        item for item in managed_run["trigger_events"] if item["kind"] == "calendar"
    )
    assert calendar_trigger == {
        "contract_version": "managed-fin-strategy-trigger-id-v1",
        "kind": "calendar",
        "schedule_contract_sha256": spec["payload"]["managed_fin_strategy"][
            "contract_sha256"
        ],
        "calendar_event": "week:2026-W02",
        "trigger_id": calendar_trigger["trigger_id"],
    }
    assert managed_run["research_window_contract_sha256"] == "c" * 64
    assert managed_run["incumbent_strategy_version_id"] == (
        "1" * 32 if incumbent_exists else None
    )
    assert managed_run["research_control"] == (
        {
            "mode": "approved_strategy_incumbent",
            "strategy_version_id": "1" * 32,
        }
        if incumbent_exists
        else {
            "mode": "transparent_public_baseline",
            "recipe_id": "short_relative_strength",
            "recipe_version": get_strategy_recipe("short_relative_strength")[
                "version"
            ],
            "recipe_sha256": spec["payload"]["managed_fin_strategy"][
                "recipe_sha256"
            ],
        }
    )
    assert research_run["config"]["incumbent_strategy"] == incumbent
    assert research_run["config"]["feature_set"]["features"] == {
        str(item["id"]): str(item["qlib_expression"])
        for item in get_strategy_recipe("short_relative_strength")["factor_baseline"]
    }
