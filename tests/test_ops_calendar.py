from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path

import pytest
from governance_fixtures import create_strategy_version
from sqlalchemy import insert, update

from quant_data.config import Settings
from quant_data.database import (
    open_database,
    simulation_batches,
    simulation_nav,
    simulation_orders,
    simulation_portfolios,
    strategy_versions,
)
from quant_platform.alert_store import AlertStore
from quant_platform.job_store import JobStore
from quant_platform.ops_calendar import (
    OpsStores,
    build_monthly_decision_day,
    evaluate_recommendation_gate,
    is_monthly_decision_day,
    is_weekly_report_day,
)
from quant_platform.ops_tasks import run_task
from quant_platform.recommendation_store import RecommendationStore
from quant_platform.schedule_store import ScheduleStore, intraday_occurrence_after
from quant_platform.scheduler import SchedulerEngine
from quant_platform.worker import LocalJobWorker

CALENDAR_DAYS = [  # Feb 2025: Feb 1-2 is a weekend
    date(2025, 1, 30),
    date(2025, 1, 31),
    date(2025, 2, 3),
    date(2025, 2, 4),
    date(2025, 2, 5),
    date(2025, 2, 6),
    date(2025, 2, 7),
]


def _settings(database_url: str, tmp_path: Path) -> Settings:
    return Settings(
        api_url="https://relay.example/api/v1/query",
        token="test-token",
        data_root=tmp_path / "data",
        database_url=database_url,
        embedded_worker=False,
    )


def _seed_qlib_dataset(data_root: Path, name: str, days: list[date]) -> None:
    path = data_root / "qlib" / name
    (path / "calendars").mkdir(parents=True)
    (path / "instruments").mkdir()
    (path / "features").mkdir()
    (path / "metadata").mkdir()
    (path / "calendars" / "day.txt").write_text(
        "\n".join(day.isoformat() for day in days), encoding="utf-8"
    )
    (path / "instruments" / "cn_all.txt").write_text("SH600000\n", encoding="utf-8")
    provenance = {
        "snapshot_name": name,
        "snapshot_manifest_sha256": "a" * 64,
        "dataset_identity_sha256": hashlib.sha256(name.encode()).hexdigest(),
        "dataset_lineage_id": "b" * 64,
        "lineage_verified": True,
    }
    (path / "metadata" / "provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )


# --- pure cadence rules -------------------------------------------------------


@pytest.mark.no_database
def test_weekly_report_day_is_saturday() -> None:
    assert is_weekly_report_day(date(2025, 2, 1)) is True  # Saturday
    assert is_weekly_report_day(date(2025, 2, 2)) is False  # Sunday
    assert is_weekly_report_day(date(2025, 2, 3)) is False  # Monday


@pytest.mark.no_database
def test_monthly_decision_day_is_the_first_trading_day() -> None:
    days = set(CALENDAR_DAYS)
    assert is_monthly_decision_day(date(2025, 2, 3), days) is True
    assert is_monthly_decision_day(date(2025, 2, 4), days) is False
    # A weekend month start: the first trading day is Monday Feb 3, not Feb 1.
    assert is_monthly_decision_day(date(2025, 2, 1), days) is False
    # Fail closed without calendar coverage.
    assert is_monthly_decision_day(date(2025, 3, 3), days) is False


@pytest.mark.no_database
def test_intraday_occurrence_respects_bar_interval_and_lunch_break() -> None:
    assert intraday_occurrence_after(
        datetime(2025, 2, 3, 3, 25, tzinfo=UTC),
        "Asia/Shanghai",
        time(9, 35),
        5,
    ) == datetime(2025, 2, 3, 3, 30, tzinfo=UTC)
    assert intraday_occurrence_after(
        datetime(2025, 2, 3, 3, 30, tzinfo=UTC),
        "Asia/Shanghai",
        time(9, 35),
        5,
    ) == datetime(2025, 2, 3, 5, 5, tzinfo=UTC)
    assert intraday_occurrence_after(
        datetime(2025, 2, 3, 7, 0, tzinfo=UTC),
        "Asia/Shanghai",
        time(9, 35),
        5,
    ) == datetime(2025, 2, 4, 1, 35, tzinfo=UTC)


# --- reconciliation gate (pure logic, fake store) ------------------------------


class FakeSimulations:
    def __init__(self, portfolios=None, batch=None, nav=None) -> None:
        self._portfolios = portfolios or []
        self._batch = batch
        self._nav = nav

    def list(self, limit: int = 100):
        return self._portfolios

    def latest_batch(self, portfolio_id: str):
        return self._batch

    def latest_nav(self, portfolio_id: str):
        return self._nav


def _linked(portfolio_id: str) -> dict:
    return {
        "id": "sim-1",
        "name": "linked sim",
        "source_type": "recommendation",
        "source_id": portfolio_id,
        "status": "active",
        "updated_at": "2025-02-03T00:00:00+00:00",
    }


def _batch(status: str, trade_date: date = date(2025, 2, 3)) -> dict:
    return {"id": "batch-1", "status": status, "trade_date": trade_date}


def _nav(
    *,
    status: str = "healthy",
    certified: bool = True,
    stale: bool = False,
    trade_date: date = date(2025, 2, 3),
) -> dict:
    return {
        "trade_date": trade_date,
        "status": status,
        "performance_certified": certified,
        "has_stale_prices": stale,
    }


@pytest.mark.no_database
def test_gate_passes_without_linked_simulation() -> None:
    result = evaluate_recommendation_gate(
        FakeSimulations(), {"id": "p1"}, date(2025, 2, 3), set(CALENDAR_DAYS)
    )
    assert result["passed"] is True
    assert "no linked simulation" in result["details"]["note"]


@pytest.mark.no_database
def test_gate_blocks_failed_or_missing_batch() -> None:
    portfolio = {"id": "p1"}
    linked = [_linked("p1")]
    failed = evaluate_recommendation_gate(
        FakeSimulations(linked, batch=_batch("failed"), nav=_nav()),
        portfolio,
        date(2025, 2, 3),
        set(CALENDAR_DAYS),
    )
    assert failed["passed"] is False
    assert any("reconciliation did not pass" in reason for reason in failed["reasons"])

    missing = evaluate_recommendation_gate(
        FakeSimulations(linked, batch=None, nav=_nav()),
        portfolio,
        date(2025, 2, 3),
        set(CALENDAR_DAYS),
    )
    assert missing["passed"] is False
    assert any("no reconciled batch" in reason for reason in missing["reasons"])


@pytest.mark.no_database
def test_gate_blocks_degraded_uncertified_and_stale_nav() -> None:
    portfolio = {"id": "p1"}
    linked = [_linked("p1")]
    for nav_kwargs, marker in (
        ({"status": "degraded"}, "degraded"),
        ({"certified": False}, "not performance-certified"),
        ({"stale": True}, "stale prices"),
    ):
        result = evaluate_recommendation_gate(
            FakeSimulations(linked, batch=_batch("succeeded"), nav=_nav(**nav_kwargs)),
            portfolio,
            date(2025, 2, 3),
            set(CALENDAR_DAYS),
        )
        assert result["passed"] is False, nav_kwargs
        assert any(marker in reason for reason in result["reasons"])


@pytest.mark.no_database
def test_gate_blocks_lagging_nav_and_uncovered_calendar() -> None:
    portfolio = {"id": "p1"}
    linked = [_linked("p1")]
    lagging = evaluate_recommendation_gate(
        FakeSimulations(
            linked,
            batch=_batch("succeeded"),
            nav=_nav(trade_date=date(2025, 1, 30)),
        ),
        portfolio,
        date(2025, 2, 3),
        set(CALENDAR_DAYS),
    )
    assert lagging["passed"] is False
    assert any("lagging" in reason for reason in lagging["reasons"])

    uncovered = evaluate_recommendation_gate(
        FakeSimulations(linked, batch=_batch("succeeded"), nav=_nav()),
        portfolio,
        date(2025, 3, 3),
        set(CALENDAR_DAYS),
    )
    assert uncovered["passed"] is False
    assert any("does not cover" in reason for reason in uncovered["reasons"])


@pytest.mark.no_database
def test_gate_passes_when_reconciled_healthy_and_fresh() -> None:
    result = evaluate_recommendation_gate(
        FakeSimulations([_linked("p1")], batch=_batch("succeeded"), nav=_nav()),
        {"id": "p1"},
        date(2025, 2, 3),
        set(CALENDAR_DAYS),
    )
    assert result["passed"] is True
    assert result["reasons"] == []


# --- monthly decision day (fake stores) -----------------------------------------


class FakeAlerts:
    def __init__(self) -> None:
        self.created = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return kwargs


class FakeHealth:
    def latest(self, *, max_age_seconds=None):
        return None


class FakeAllocations:
    def __init__(self, allocations, fail_on=()) -> None:
        self._allocations = allocations
        self._fail_on = set(fail_on)
        self.refreshed = []

    def list(self, limit: int = 100):
        return self._allocations

    def refresh(self, allocation_id: str, *, actor: str = ""):
        if allocation_id in self._fail_on:
            raise ValueError("member NAV is not certified")
        self.refreshed.append(allocation_id)
        return {"id": allocation_id}


def _allocation(allocation_id: str, *, status="active", valid_until=None, legacy=False) -> dict:
    artifacts = (
        [{"valid_until": valid_until.isoformat(), "decision_date": "2025-01-03"}]
        if valid_until
        else []
    )
    return {
        "id": allocation_id,
        "name": f"alloc-{allocation_id}",
        "status": status,
        "is_legacy": legacy,
        "decision_frequency": "monthly",
        "artifacts": artifacts,
    }


@pytest.mark.no_database
def test_monthly_decision_day_refreshes_only_due_allocations(tmp_path: Path) -> None:
    allocations = FakeAllocations(
        [
            _allocation("due-expired", valid_until=date(2025, 2, 3)),  # expired on the day
            _allocation("due-none"),  # no artifact yet
            _allocation("valid", valid_until=date(2025, 3, 3)),
            _allocation("paused", status="paused", valid_until=date(2025, 1, 3)),
            _allocation("legacy", legacy=True, valid_until=date(2025, 1, 3)),
        ],
        fail_on={"due-none"},
    )
    alerts = FakeAlerts()
    stores = OpsStores(
        alerts=alerts,
        simulations=None,
        recommendations=None,
        allocations=allocations,
        health=FakeHealth(),
    )
    settings = _settings("postgresql+psycopg://x:x@127.0.0.1:1/x", tmp_path)

    report = build_monthly_decision_day(settings, stores, date(2025, 2, 3))

    assert allocations.refreshed == ["due-expired"]
    assert {item["allocation_id"] for item in report["allocations_due"]} == {
        "due-expired",
        "due-none",
    }
    assert report["allocations_refreshed"] == ["due-expired"]
    assert report["allocations_refresh_failed"] == [
        {"allocation_id": "due-none", "error": "member NAV is not certified"}
    ]
    categories = {item["category"] for item in alerts.created}
    assert "monthly_decision_day" in categories
    assert "monthly_decision_refresh_failed" in categories
    assert Path(report["artifact_path"]).is_file()


# --- DB-backed scheduler and builder tests --------------------------------------


def test_weekly_report_saturday_enqueues_and_other_days_skip(
    database_url: str, tmp_path: Path
) -> None:
    store = ScheduleStore(database_url)
    store.create(
        name="weekly ops report",
        kind="weekly_report",
        timezone="Asia/Shanghai",
        run_time=time(9, 0),
        trading_days_only=False,
        payload={},
        misfire_grace_seconds=3600,
        actor="operator",
        now=datetime(2025, 1, 31, 0, 0, tzinfo=UTC),  # Friday
    )
    engine = SchedulerEngine(_settings(database_url, tmp_path))

    friday = engine.tick(datetime(2025, 1, 31, 1, 1, tzinfo=UTC))  # local 09:01 Fri
    assert friday["processed"] == 1
    assert store.list_runs()[0]["status"] == "skipped"
    assert "not the weekly report day" in store.list_runs()[0]["message"]

    saturday = engine.tick(datetime(2025, 2, 1, 1, 1, tzinfo=UTC))  # local Sat 09:01
    assert saturday["processed"] == 1
    runs = store.list_runs()
    assert len(runs) == 2
    latest = runs[0]
    assert latest["status"] == "enqueued"
    job = JobStore(database_url).get(latest["job_id"])
    assert job["kind"] == "weekly_report"
    assert job["payload"]["local_date"] == "2025-02-01"
    # Idempotent: re-ticking the same slot materializes nothing new.
    again = engine.tick(datetime(2025, 2, 1, 1, 2, tzinfo=UTC))
    assert again["materialized"] == 0
    assert again["processed"] == 0


def test_weekly_report_builder_persists_artifact_and_deduped_alerts(
    database_url: str, tmp_path: Path
) -> None:
    settings = _settings(database_url, tmp_path)

    first = run_task(settings, "weekly_report", date(2025, 2, 1))
    second = run_task(settings, "weekly_report", date(2025, 2, 1))

    assert first["kind"] == "weekly_report"
    assert first["iso_week"] == "2025-W05"
    assert Path(first["artifact_path"]).is_file()
    alerts = AlertStore(database_url).list()
    assert len(alerts) == 1  # summary alert deduped across reruns
    assert alerts[0]["category"] == "weekly_report"
    assert second["artifact_path"] == first["artifact_path"]


def test_monthly_decision_day_scheduler_first_trading_day_only(
    database_url: str, tmp_path: Path
) -> None:
    settings = _settings(database_url, tmp_path)
    _seed_qlib_dataset(settings.data_root, "snap-ops", CALENDAR_DAYS)
    store = ScheduleStore(database_url)
    store.create(
        name="monthly decision day",
        kind="monthly_decision_day",
        timezone="Asia/Shanghai",
        run_time=time(9, 0),
        trading_days_only=True,
        payload={},
        misfire_grace_seconds=3600,
        actor="operator",
        now=datetime(2025, 2, 1, 0, 0, tzinfo=UTC),  # Saturday 08:00 local
    )
    engine = SchedulerEngine(settings)

    # Saturday Feb 1: not a trading day at all.
    engine.tick(datetime(2025, 2, 1, 1, 1, tzinfo=UTC))
    assert store.list_runs()[0]["status"] == "skipped"
    assert "not a Qlib trading day" in store.list_runs()[0]["message"]

    # Sunday Feb 2: also not a trading day.
    engine.tick(datetime(2025, 2, 2, 1, 1, tzinfo=UTC))
    assert store.list_runs()[0]["status"] == "skipped"

    # Monday Feb 3: first trading day of the month -> enqueued.
    engine.tick(datetime(2025, 2, 3, 1, 1, tzinfo=UTC))
    runs = store.list_runs()
    assert len(runs) == 3
    assert runs[0]["status"] == "enqueued"
    job = JobStore(database_url).get(runs[0]["job_id"])
    assert job["kind"] == "monthly_decision_day"
    assert job["payload"]["local_date"] == "2025-02-03"

    # Tuesday Feb 4: a trading day but not the first of the month.
    engine.tick(datetime(2025, 2, 4, 1, 1, tzinfo=UTC))
    runs = store.list_runs()
    assert len(runs) == 4
    assert runs[0]["status"] == "skipped"
    assert "not the first trading day" in runs[0]["message"]


def test_monthly_decision_day_scheduler_fails_closed_without_dataset(
    database_url: str, tmp_path: Path
) -> None:
    store = ScheduleStore(database_url)
    store.create(
        name="monthly decision day no dataset",
        kind="monthly_decision_day",
        timezone="Asia/Shanghai",
        run_time=time(9, 0),
        trading_days_only=True,
        payload={},
        misfire_grace_seconds=3600,
        actor="operator",
        now=datetime(2025, 2, 3, 0, 0, tzinfo=UTC),  # Monday 08:00 local
    )
    engine = SchedulerEngine(_settings(database_url, tmp_path))

    engine.tick(datetime(2025, 2, 3, 1, 1, tzinfo=UTC))

    run = store.list_runs()[0]
    assert run["status"] == "failed"
    assert "no ready and reproducible Qlib dataset" in run["message"]
    alerts = AlertStore(database_url).list()
    assert any(item["category"] == "schedule_failure" for item in alerts)


def test_preopen_check_scheduler_trading_day_gate_and_builder(
    database_url: str, tmp_path: Path
) -> None:
    settings = _settings(database_url, tmp_path)
    _seed_qlib_dataset(settings.data_root, "snap-ops", CALENDAR_DAYS)
    store = ScheduleStore(database_url)
    store.create(
        name="preopen check",
        kind="preopen_check",
        timezone="Asia/Shanghai",
        run_time=time(8, 30),
        trading_days_only=True,
        payload={},
        misfire_grace_seconds=3600,
        actor="operator",
        now=datetime(2025, 2, 2, 0, 0, tzinfo=UTC),  # Sunday
    )
    engine = SchedulerEngine(settings)

    engine.tick(datetime(2025, 2, 2, 0, 31, tzinfo=UTC))  # local Sun 08:31
    assert store.list_runs()[0]["status"] == "skipped"

    engine.tick(datetime(2025, 2, 3, 0, 31, tzinfo=UTC))  # local Mon 08:31
    runs = store.list_runs()
    assert len(runs) == 2
    assert runs[0]["status"] == "enqueued"
    job = JobStore(database_url).get(runs[0]["job_id"])
    assert job["kind"] == "preopen_check"

    # Builder: dataset ends Jan 31 but Feb 3's previous trading day is Jan 31,
    # so data is ready; summary + deduped alerts, artifact persisted.
    report = run_task(settings, "preopen_check", date(2025, 2, 3))
    assert report["kind"] == "preopen_check"
    assert report["previous_trading_day"] == "2025-01-31"
    assert report["anomalies"] == []
    assert Path(report["artifact_path"]).is_file()
    run_task(settings, "preopen_check", date(2025, 2, 3))
    alerts = AlertStore(database_url).list()
    assert len(alerts) == 1
    assert alerts[0]["category"] == "preopen_check"

    # A lagging dataset whose calendar cannot confirm the trading day fails
    # closed (数据未就绪 → 阻断，不允许绕过质量门).
    _seed_qlib_dataset(settings.data_root, "snap-lag", CALENDAR_DAYS[:2])
    with pytest.raises(ValueError, match="not a trading day"):
        run_task(settings, "preopen_check", date(2025, 2, 3), dataset_anchor="snap-lag")


def test_worker_command_dispatches_ops_kinds(database_url: str, tmp_path: Path) -> None:
    settings = _settings(database_url, tmp_path)
    worker = LocalJobWorker(JobStore(database_url), tmp_path, settings)
    job = {
        "id": "job-1",
        "kind": "weekly_report",
        "payload": {"local_date": "2025-02-01"},
    }
    command, result_path, env = worker._command(job)
    assert command[:3] == [__import__("sys").executable, "-m", "quant_platform.ops_tasks"]
    assert "weekly_report" in command
    assert "--date" in command and "2025-02-01" in command
    assert result_path is not None and result_path.name == "result.json"
    assert env == {}


# --- recommendation reconciliation gate (scheduler integration) ------------------


def _create_portfolio(database_url: str, tmp_path: Path) -> dict:
    version_id = create_strategy_version(database_url, tmp_path)
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved")
        )
    store = RecommendationStore(database_url)
    return store.create(
        name="gated recommendations",
        strategy_version_id=version_id,
        dataset="snap-ops",
        hypothetical_initial_value=5_000_000,
        actor="test",
    )


def _insert_simulation_account(
    database_url: str, portfolio_id: str, *, batch_status: str, nav_status: str,
    nav_certified: bool, nav_stale: bool, trade_date: date,
) -> None:
    engine = open_database(database_url)
    now = datetime(2025, 2, 3, 12, 0, tzinfo=UTC)
    with engine.begin() as connection:
        connection.execute(
            insert(simulation_portfolios).values(
                id="sim-gate-1",
                name="linked gate simulation",
                source_type="recommendation",
                source_id=portfolio_id,
                status="active",
                base_currency="CNY",
                initial_cash=Decimal("1000000"),
                cash=Decimal("1000000"),
                nav=Decimal("1000000"),
                high_water_mark=Decimal("1000000"),
                execution_algorithm="immediate",
                execution_contract_hash="c" * 64,
                execution_dataset="exec-ds",
                daily_dataset="snap-ops",
                daily_dataset_identity_sha256="d" * 64,
                daily_dataset_lineage_id="e" * 64,
                daily_field_contract_version="v1",
                execution_dataset_identity_sha256="f" * 64,
                execution_dataset_lineage_id="0" * 64,
                execution_field_contract_version="v1",
                execution_engine_version="v1",
                cost_schedule_version="v1",
                execution_policy_json={},
                created_by="test",
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            insert(simulation_batches).values(
                id="batch-gate-1",
                portfolio_id="sim-gate-1",
                execution_contract_hash="c" * 64,
                signal_date=trade_date,
                trade_date=trade_date,
                status=batch_status,
                idempotency_key=f"gate-batch-{batch_status}",
                summary_json={},
                created_at=now,
            )
        )
        connection.execute(
            insert(simulation_nav).values(
                portfolio_id="sim-gate-1",
                trade_date=trade_date,
                cash=Decimal("1000000"),
                market_value=Decimal("0"),
                nav=Decimal("1000000"),
                daily_return=0.0,
                drawdown=0.0,
                has_stale_prices=nav_stale,
                status=nav_status,
                performance_certified=nav_certified,
                created_at=now,
            )
        )


def _create_gate_schedule(database_url: str, portfolio_id: str) -> None:
    ScheduleStore(database_url).create(
        name="gated daily recommendations",
        kind="recommendation_refresh",
        timezone="Asia/Shanghai",
        run_time=time(17, 0),
        trading_days_only=True,
        payload={"recommendation_portfolio_id": portfolio_id},
        misfire_grace_seconds=3600,
        actor="operator",
        now=datetime(2025, 2, 3, 8, 0, tzinfo=UTC),  # local 16:00, before slot
    )


def test_intraday_check_projects_only_material_simulation_order_changes(
    database_url: str, tmp_path: Path
) -> None:
    settings = _settings(database_url, tmp_path)
    _seed_qlib_dataset(settings.data_root, "snap-ops", CALENDAR_DAYS)
    portfolio = _create_portfolio(database_url, tmp_path)
    _insert_simulation_account(
        database_url,
        portfolio["id"],
        batch_status="succeeded",
        nav_status="healthy",
        nav_certified=True,
        nav_stale=False,
        trade_date=date(2025, 2, 3),
    )
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            insert(simulation_orders).values(
                id="intraday-order-1",
                batch_id="batch-gate-1",
                portfolio_id="sim-gate-1",
                instrument="SH600000",
                side="buy",
                target_weight=0.10,
                requested_quantity=1000,
                filled_quantity=0,
                status="open",
                requested_value=Decimal("10000"),
                filled_value=Decimal("0"),
                capacity_fill_ratio=1.0,
                expires_at=datetime(2025, 2, 3, 7, 0, tzinfo=UTC),
                not_before=datetime(2025, 2, 3, 1, 35, tzinfo=UTC),
                not_after=datetime(2025, 2, 3, 7, 0, tzinfo=UTC),
                created_at=datetime(2025, 2, 3, 1, 34, tzinfo=UTC),
            )
        )

    first = run_task(
        settings,
        "intraday_execution_check",
        date(2025, 2, 3),
        dataset_anchor="snap-ops",
        as_of=datetime(2025, 2, 3, 1, 35, tzinfo=UTC),
    )
    assert first["minute_alpha_generated"] is False
    assert first["live_price_status"] == "blocked_by_data_or_permission"
    assert first["events"][0]["execution_state"] == "READY"

    with engine.begin() as connection:
        connection.execute(
            update(simulation_orders)
            .where(simulation_orders.c.id == "intraday-order-1")
            .values(filled_quantity=300, filled_value=Decimal("3000"))
        )
    second = run_task(
        settings,
        "intraday_execution_check",
        date(2025, 2, 3),
        dataset_anchor="snap-ops",
        as_of=datetime(2025, 2, 3, 1, 40, tzinfo=UTC),
    )
    assert second["events"][0]["execution_state"] == "PARTIAL"
    assert second["events"][0]["remaining_quantity"] == 700

    # Repeating the same completed-bar facts is silent: the durable alert
    # dedupe key includes order state and filled quantity.
    run_task(
        settings,
        "intraday_execution_check",
        date(2025, 2, 3),
        dataset_anchor="snap-ops",
        as_of=datetime(2025, 2, 3, 1, 45, tzinfo=UTC),
    )
    projected = [
        item
        for item in AlertStore(database_url).list()
        if item["category"] == "intraday_execution_state"
    ]
    assert len(projected) == 2
    assert all(item["details"]["account_source"] == "simulation" for item in projected)


def test_recommendation_refresh_blocked_until_reconciled_then_recovers(
    database_url: str, tmp_path: Path
) -> None:
    settings = _settings(database_url, tmp_path)
    _seed_qlib_dataset(settings.data_root, "snap-ops", CALENDAR_DAYS)
    portfolio = _create_portfolio(database_url, tmp_path)
    _create_gate_schedule(database_url, portfolio["id"])
    engine = SchedulerEngine(settings)
    jobs = JobStore(database_url)
    alerts = AlertStore(database_url)

    # Day 1 (Mon Feb 3): no linked simulation account -> gate passes.
    day1 = engine.tick(datetime(2025, 2, 3, 9, 1, tzinfo=UTC))  # local 17:01
    assert day1["processed"] == 1
    assert len(jobs.list()) == 1

    # Day 2: linked account with a FAILED batch + degraded uncertified NAV.
    _insert_simulation_account(
        database_url,
        portfolio["id"],
        batch_status="failed",
        nav_status="degraded",
        nav_certified=False,
        nav_stale=True,
        trade_date=date(2025, 2, 4),
    )
    day2 = engine.tick(datetime(2025, 2, 4, 9, 1, tzinfo=UTC))
    assert day2["processed"] == 1
    assert len(jobs.list()) == 1  # no new recommendation job
    run = ScheduleStore(database_url).list_runs()[0]
    assert run["status"] == "skipped"
    assert "reconciliation gate blocked" in run["message"]
    blocking = [
        item
        for item in alerts.list()
        if item["category"] == "recommendation_reconciliation_blocked"
    ]
    assert len(blocking) == 1
    assert "retained" in str(blocking[0]["details"]["retained_snapshot"]["status"])
    reasons = " ".join(blocking[0]["details"]["reasons"])
    assert "reconciliation did not pass" in reasons
    assert "degraded" in reasons
    # The old snapshot was never created for day 2: no new snapshot row.
    snapshots = engine.recommendations.get(portfolio["id"])["snapshots"]
    assert {str(item["as_of_date"]) for item in snapshots} == {"2025-02-03"}

    # Day 3: batch reconciled, NAV healthy/certified/fresh -> gate passes again.
    engine_db = open_database(database_url)
    now = datetime(2025, 2, 5, 12, 0, tzinfo=UTC)
    with engine_db.begin() as connection:
        connection.execute(
            insert(simulation_batches).values(
                id="batch-gate-2",
                portfolio_id="sim-gate-1",
                execution_contract_hash="c" * 64,
                signal_date=date(2025, 2, 5),
                trade_date=date(2025, 2, 5),
                status="succeeded",
                idempotency_key="gate-batch-recovered",
                summary_json={},
                created_at=now,
            )
        )
        connection.execute(
            insert(simulation_nav).values(
                portfolio_id="sim-gate-1",
                trade_date=date(2025, 2, 5),
                cash=Decimal("1000000"),
                market_value=Decimal("0"),
                nav=Decimal("1000000"),
                daily_return=0.0,
                drawdown=0.0,
                has_stale_prices=False,
                status="healthy",
                performance_certified=True,
                created_at=now,
            )
        )
    day3 = engine.tick(datetime(2025, 2, 5, 9, 1, tzinfo=UTC))
    assert day3["processed"] == 1
    assert len(jobs.list()) == 2  # the refresh is enqueued again
    assert ScheduleStore(database_url).list_runs()[0]["status"] == "enqueued"
    # The blocking alert dedupes per (portfolio, date): re-ticking day 2's slot
    # never creates a second alert.
    assert (
        len(
            [
                item
                for item in alerts.list()
                if item["category"] == "recommendation_reconciliation_blocked"
            ]
        )
        == 1
    )
