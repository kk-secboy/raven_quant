from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from governance_fixtures import create_strategy_version
from sqlalchemy import insert, select

from quant_data.cli import _trigger_safe_mode_on_quality_gate_failure
from quant_data.config import Settings
from quant_data.database import (
    alerts,
    open_database,
    recommendation_portfolios,
    simulation_batches,
    simulation_nav,
    simulation_portfolios,
)
from quant_platform.health_store import OperationalHealthStore
from quant_platform.recommendation_store import RecommendationStore
from quant_platform.safe_mode import (
    PERSISTENT_NAV_FAILURE_ROWS,
    SafeModeActiveError,
    SafeModeStore,
)
from quant_platform.schedule_store import ScheduleStore
from quant_platform.scheduler import SchedulerEngine
from quant_platform.simulation_store import SimulationStore


def _settings(database_url: str, tmp_path: Path) -> Settings:
    return Settings(
        api_url="https://relay.example/api/v1/query",
        token="test-token",
        data_root=tmp_path / "data",
        database_url=database_url,
        embedded_worker=True,
        rdagent_enabled=False,
        health_snapshot_seconds=300,
        platform_secret_key=Fernet.generate_key().decode("ascii"),
    )


def _seed_simulation_portfolio(database_url: str, *, name: str = "safe-mode-sim") -> str:
    engine = open_database(database_url)
    portfolio_id = uuid.uuid4().hex
    now = datetime.now(UTC)
    with engine.begin() as connection:
        connection.execute(
            insert(simulation_portfolios).values(
                id=portfolio_id,
                name=name,
                recommendation_portfolio_id=None,
                source_type="strategy_version",
                source_id=uuid.uuid4().hex,
                status="active",
                base_currency="CNY",
                initial_cash=Decimal("1000000"),
                cash=Decimal("1000000"),
                nav=Decimal("1000000"),
                high_water_mark=Decimal("1000000"),
                execution_algorithm="twap",
                execution_adapter="long_only",
                execution_frequency="5min",
                execution_contract_hash="f" * 64,
                execution_dataset="exec-5m",
                daily_dataset="daily-data",
                daily_dataset_identity_sha256="a" * 64,
                daily_dataset_lineage_id="b" * 64,
                daily_field_contract_version="daily-qlib-field-v3-cny-amount",
                execution_dataset_identity_sha256="c" * 64,
                execution_dataset_lineage_id="d" * 64,
                execution_field_contract_version="minute-qlib-execution-v4-source-units",
                execution_engine_version="sim-engine-test",
                cost_schedule_version="cost-schedule-test",
                execution_policy_json={"execution_algorithm": "twap"},
                created_by="safe-mode-test",
                created_at=now,
                updated_at=now,
            )
        )
    return portfolio_id


def _seed_queued_batch(database_url: str, portfolio_id: str) -> str:
    engine = open_database(database_url)
    batch_id = uuid.uuid4().hex
    with engine.begin() as connection:
        connection.execute(
            insert(simulation_batches).values(
                id=batch_id,
                portfolio_id=portfolio_id,
                recommendation_snapshot_id=None,
                execution_contract_hash="f" * 64,
                signal_date=date(2026, 7, 20),
                trade_date=date(2026, 7, 21),
                status="queued",
                idempotency_key=f"safe-mode-test:{batch_id}",
                created_by="safe-mode-test",
                created_at=datetime.now(UTC),
            )
        )
    return batch_id


def _seed_nav_rows(
    database_url: str,
    portfolio_id: str,
    *,
    statuses: list[str],
    certified: bool,
    start: date = date(2026, 7, 20),
) -> None:
    engine = open_database(database_url)
    with engine.begin() as connection:
        for offset, status in enumerate(statuses):
            connection.execute(
                insert(simulation_nav).values(
                    portfolio_id=portfolio_id,
                    trade_date=start + timedelta(days=offset),
                    cash=Decimal("1000000"),
                    market_value=Decimal("0"),
                    nav=Decimal("1000000"),
                    daily_return=0.0,
                    drawdown=0.0,
                    has_stale_prices=False,
                    status=status,
                    performance_certified=certified,
                    produced_by="safe-mode-test",
                    created_at=datetime.now(UTC),
                )
            )


def _safe_mode_alerts(database_url: str, category: str) -> list[dict]:
    engine = open_database(database_url)
    with engine.connect() as connection:
        return [
            dict(row._mapping)
            for row in connection.execute(
                select(alerts).where(alerts.c.category == category)
            )
        ]


def test_non_ledger_batch_failure_does_not_engage_safe_mode(database_url: str) -> None:
    portfolio_id = _seed_simulation_portfolio(database_url)
    batch_id = _seed_queued_batch(database_url, portfolio_id)
    store = SimulationStore(database_url)
    store.mark_batch_failed(batch_id, "minute bars missing for trade date")
    assert not SafeModeStore(database_url).is_active()


def test_ledger_conservation_failure_engages_safe_mode(database_url: str) -> None:
    portfolio_id = _seed_simulation_portfolio(database_url)
    batch_id = _seed_queued_batch(database_url, portfolio_id)
    store = SimulationStore(database_url)
    store.mark_batch_failed(batch_id, "simulation ledger cash conservation failed")

    state = SafeModeStore(database_url).status()
    assert state["active"] is True
    assert state["source"] == "simulation_ledger"
    assert batch_id in state["reason"]
    assert state["triggered_by"] == "system"
    triggered = _safe_mode_alerts(database_url, "safe_mode_activated")
    assert len(triggered) == 1
    assert triggered[0]["severity"] == "critical"
    assert "解除" in triggered[0]["message"] or "release" in triggered[0]["message"]
    # The batch failure itself is still recorded (recovery path keeps working).
    assert store.get_batch(batch_id)["status"] == "failed"


def test_verify_quality_gate_failure_engages_safe_mode(database_url: str) -> None:
    settings = SimpleNamespace(database_url=database_url)
    report = {"errors": ["daily: 5 duplicate primary-key rows", "income: 2/3 units"]}
    _trigger_safe_mode_on_quality_gate_failure(settings, report)

    state = SafeModeStore(database_url).status()
    assert state["active"] is True
    assert state["source"] == "data_quality_gate"
    assert "duplicate primary-key" in str(state["details"]["errors"])


def test_persistent_degraded_nav_engages_safe_mode(database_url: str) -> None:
    store = SafeModeStore(database_url)
    portfolio_id = _seed_simulation_portfolio(database_url)
    # Two bad rows are not yet "persistent".
    _seed_nav_rows(database_url, portfolio_id, statuses=["degraded", "degraded"], certified=False)
    assert store.check_persistent_nav_anomalies() is None
    assert not store.is_active()

    healthy_id = _seed_simulation_portfolio(database_url, name="safe-mode-healthy")
    _seed_nav_rows(
        database_url,
        healthy_id,
        statuses=["healthy"] * PERSISTENT_NAV_FAILURE_ROWS,
        certified=True,
    )
    _seed_nav_rows(
        database_url,
        portfolio_id,
        statuses=["degraded"],
        certified=False,
        start=date(2026, 7, 22),
    )
    state = store.check_persistent_nav_anomalies()
    assert state is not None and state["active"] is True
    assert state["source"] == "nav_health"
    offenders = state["details"]["offenders"]
    assert [item["portfolio_id"] for item in offenders] == [portfolio_id]


def test_safe_mode_blocks_recommendations_and_simulation_orders(
    database_url: str, tmp_path: Path
) -> None:
    safe_mode = SafeModeStore(database_url)
    safe_mode.activate(reason="人工演练：账本异常", source="manual", actor="operator-a")

    recommendations = RecommendationStore(database_url)
    with pytest.raises(SafeModeActiveError, match="safe_mode"):
        recommendations.create_snapshot(
            portfolio_id="missing",
            as_of_date=date(2026, 7, 21),
            dataset="snapshot",
            dataset_identity_sha256="a" * 64,
        )
    simulations = SimulationStore(database_url)
    with pytest.raises(SafeModeActiveError, match="safe_mode"):
        simulations.create(
            name="blocked-account",
            daily_dataset={},
            execution_dataset={},
            initial_cash=1.0,
            execution_policy={},
            cost_schedule_version="unknown",
            actor="operator-a",
        )
    with pytest.raises(SafeModeActiveError, match="safe_mode"):
        simulations.create_batches_for_snapshot("missing")
    with pytest.raises(SafeModeActiveError, match="safe_mode"):
        simulations.create_batch_from_order_plan(
            "missing",
            order_plan_manifest_sha256="0" * 64,
            data_root=tmp_path,
            actor="operator-a",
        )
    with pytest.raises(SafeModeActiveError, match="safe_mode"):
        simulations.create_order_plan_batch(
            "missing",
            trade_date=date(2026, 7, 21),
            actions=[],
            target_version="v1",
            actor="operator-a",
        )
    with pytest.raises(SafeModeActiveError, match="safe_mode"):
        simulations.create_pair_batch_from_backtest(
            "missing",
            backtest_id="missing",
            trade_date=date(2026, 7, 21),
            data_root=tmp_path,
            actor="operator-a",
        )


def test_safe_mode_allows_read_and_recovery_paths(
    database_url: str, tmp_path: Path
) -> None:
    portfolio_id = _seed_simulation_portfolio(database_url)
    batch_id = _seed_queued_batch(database_url, portfolio_id)
    safe_mode = SafeModeStore(database_url)
    safe_mode.activate(reason="人工演练：数据异常", source="manual", actor="operator-a")

    simulations = SimulationStore(database_url)
    # 查看路径不变。
    assert simulations.get(portfolio_id)["id"] == portfolio_id
    assert simulations.list(10)
    assert simulations.get_batch(batch_id)["id"] == batch_id
    assert simulations.latest_batch(portfolio_id)["id"] == batch_id
    # 对账/恢复路径可跑：批次失败登记（含复核用的失败原因）不被阻断。
    simulations.mark_batch_failed(batch_id, "reconciliation review failed")
    assert simulations.get_batch(batch_id)["status"] == "failed"


def test_manual_release_requires_actor_reason_and_restores(database_url: str) -> None:
    safe_mode = SafeModeStore(database_url)
    safe_mode.activate(reason="人工演练", source="manual", actor="operator-a")

    with pytest.raises(ValueError, match="actor"):
        safe_mode.deactivate(actor="x", reason="reasonably long reason")
    with pytest.raises(ValueError, match="meaningful reason"):
        safe_mode.deactivate(actor="operator-b", reason="short")
    with pytest.raises(ValueError, match="health check"):
        safe_mode.deactivate(
            actor="operator-b",
            reason="数据已修复并完成恢复演练",
            require_health_ok=True,
            health_status="degraded",
        )
    assert safe_mode.is_active()

    state = safe_mode.deactivate(
        actor="operator-b",
        reason="数据已修复，对账通过，恢复建议与模拟订单",
        require_health_ok=True,
        health_status="ok",
    )
    assert state["active"] is False
    assert state["cleared_by"] == "operator-b"
    assert _safe_mode_alerts(database_url, "safe_mode_released")

    # 阻断已解除：create_snapshot 不再因 safe_mode 失败（组合不存在→KeyError）。
    with pytest.raises(KeyError):
        RecommendationStore(database_url).create_snapshot(
            portfolio_id="missing",
            as_of_date=date(2026, 7, 21),
            dataset="snapshot",
            dataset_identity_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="not active"):
        safe_mode.deactivate(actor="operator-b", reason="safe mode 已经解除过了")


def test_activation_is_idempotent_without_duplicate_alerts(database_url: str) -> None:
    safe_mode = SafeModeStore(database_url)
    first = safe_mode.activate(reason="账本异常", source="simulation_ledger", actor="system")
    second = safe_mode.activate(reason="数据异常", source="data_quality_gate", actor="system")
    assert second["active"] is True
    # The original trigger wins; no second activation and no duplicate alert.
    assert second["source"] == first["source"] == "simulation_ledger"
    assert second["reason"] == first["reason"]
    assert len(_safe_mode_alerts(database_url, "safe_mode_activated")) == 1


def test_health_component_reports_safe_mode(
    database_url: str, tmp_path: Path
) -> None:
    settings = _settings(database_url, tmp_path)
    health = OperationalHealthStore(settings)
    observation = health.collect()
    assert observation["components"]["safe_mode"]["status"] == "ok"

    SafeModeStore(database_url).activate(reason="人工演练", source="manual", actor="operator-a")
    observation = health.collect()
    component = observation["components"]["safe_mode"]
    assert component["status"] == "degraded"
    assert "人工演练" in component["message"]
    assert observation["status"] == "degraded"


def test_scheduler_safe_mode_check_precedes_reconciliation_gate(
    database_url: str, tmp_path: Path
) -> None:
    version_id = create_strategy_version(database_url, tmp_path)
    engine = open_database(database_url)
    portfolio_id = uuid.uuid4().hex
    now = datetime.now(UTC)
    with engine.begin() as connection:
        connection.execute(
            insert(recommendation_portfolios).values(
                id=portfolio_id,
                name="safe-mode-gated recommendations",
                strategy_version_id=version_id,
                dataset="snapshot",
                status="active",
                base_currency="CNY",
                hypothetical_initial_value=Decimal("5000000"),
                risk_exposure_override=1.0,
                created_by="safe-mode-test",
                created_at=now,
                updated_at=now,
            )
        )
    SafeModeStore(database_url).activate(reason="人工演练", source="manual", actor="operator-a")

    current = datetime(2026, 7, 21, 7, 29, tzinfo=UTC)
    store = ScheduleStore(database_url)
    store.create(
        name="daily recommendation refresh",
        kind="recommendation_refresh",
        timezone="Asia/Shanghai",
        run_time=time(15, 30),
        trading_days_only=False,
        payload={"recommendation_portfolio_id": portfolio_id},
        misfire_grace_seconds=1800,
        actor="operator",
        now=current,
    )
    engine_scheduler = SchedulerEngine(_settings(database_url, tmp_path))
    engine_scheduler.tick(current + timedelta(minutes=1))

    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0]["status"] == "skipped"
    assert "safe_mode" in runs[0]["message"]
    blocked = _safe_mode_alerts(database_url, "safe_mode_recommendation_blocked")
    assert len(blocked) == 1
    assert blocked[0]["severity"] == "critical"
    # 不生成新建议快照，也不入队 recommendation_refresh 任务。
    recommendations = RecommendationStore(database_url)
    assert recommendations.get(portfolio_id)["snapshots"] == []
