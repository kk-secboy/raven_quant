"""Store-level tests for external cash flows, TWR chaining and XIRR (DB)."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from test_simulation_store import (
    TRADE_DATE,
    _bars,
    _create_batch,
    _execution_evidence,
)

from quant_data.database import simulation_cash_flows


def test_external_flow_recording_is_idempotent(database_url: str, tmp_path) -> None:
    store, simulation, _batch = _create_batch(database_url, tmp_path)
    first = store.record_external_flow(
        simulation["id"],
        trade_date=TRADE_DATE,
        timing="open",
        amount=200_000.0,
        actor="treasury",
        note="monthly top-up",
    )
    assert first["created"] is True
    replay = store.record_external_flow(
        simulation["id"],
        trade_date=TRADE_DATE,
        timing="open",
        amount=200_000.0,
        actor="treasury",
        note="monthly top-up",
    )
    assert replay["created"] is False
    assert replay["id"] == first["id"]
    flows = store.rows(simulation["id"], "external_flows")
    assert len(flows) == 1
    events = store.rows(simulation["id"], "events")
    assert (
        sum(item["event_type"] == "external_cash_flow_recorded" for item in events) == 1
    )
    with pytest.raises(ValueError, match="different payload"):
        store.record_external_flow(
            simulation["id"],
            trade_date=TRADE_DATE,
            timing="close",
            amount=5_000.0,
            actor="treasury",
            flow_key=first["flow_key"],
        )


def test_external_flow_validation(database_url: str, tmp_path) -> None:
    store, simulation, _batch = _create_batch(database_url, tmp_path)
    with pytest.raises(ValueError, match="timing"):
        store.record_external_flow(
            simulation["id"],
            trade_date=TRADE_DATE,
            timing="midday",
            amount=1_000.0,
            actor="treasury",
        )
    with pytest.raises(ValueError, match="non-zero"):
        store.record_external_flow(
            simulation["id"],
            trade_date=TRADE_DATE,
            timing="open",
            amount=0.0,
            actor="treasury",
        )
    with pytest.raises(ValueError, match="does not exist"):
        store.record_external_flow(
            "missing-portfolio",
            trade_date=TRADE_DATE,
            timing="open",
            amount=1_000.0,
            actor="treasury",
        )


def test_settled_trade_date_rejects_new_flows(database_url: str, tmp_path) -> None:
    store, simulation, batch = _create_batch(database_url, tmp_path)
    store.process_batch(
        batch["id"],
        minute_bars=_bars(),
        closing_prices={"SH600000": {"price": 10.0, "market_date": TRADE_DATE.isoformat()}},
        execution_evidence=_execution_evidence(batch["id"], simulation["execution_contract_hash"]),
    )
    with pytest.raises(ValueError, match="settled trade date"):
        store.record_external_flow(
            simulation["id"],
            trade_date=TRADE_DATE,
            timing="open",
            amount=10_000.0,
            actor="treasury",
        )


def test_process_batch_applies_open_deposit_without_manufacturing_return(
    database_url: str, tmp_path
) -> None:
    store, simulation, batch = _create_batch(database_url, tmp_path)
    store.record_external_flow(
        simulation["id"],
        trade_date=TRADE_DATE,
        timing="open",
        amount=200_000.0,
        actor="treasury",
    )
    completed = store.process_batch(
        batch["id"],
        minute_bars=_bars(),
        closing_prices={"SH600000": {"price": 10.0, "market_date": TRADE_DATE.isoformat()}},
        execution_evidence=_execution_evidence(batch["id"], simulation["execution_contract_hash"]),
    )
    assert completed["status"] == "succeeded"
    nav = store.rows(simulation["id"], "nav")[0]
    assert nav["external_flow_open"] == pytest.approx(200_000.0)
    # 人民币口径日收益 ≈ +20%（入金跳升），TWR 只反映交易损益（费用拖累）。
    assert nav["daily_return"] == pytest.approx(nav["nav"] / 1_000_000.0 - 1.0)
    assert nav["twr_daily_return"] == pytest.approx(nav["nav"] / 1_200_000.0 - 1.0)
    assert nav["twr_status"] == "ok"
    assert nav["investment_wealth"] == pytest.approx(1.0 + nav["twr_daily_return"])
    portfolio = store.get(simulation["id"])
    assert portfolio["investment_wealth"] == pytest.approx(nav["investment_wealth"])
    assert portfolio["twr_high_water_mark"] >= nav["investment_wealth"]
    assert completed["summary"]["conservation"]["cash_difference"] == pytest.approx(
        0.0, abs=1e-6
    )
    flow_types = set()
    with store.engine.connect() as connection:
        for row in connection.execute(
            select(simulation_cash_flows).where(
                simulation_cash_flows.c.portfolio_id == simulation["id"],
                simulation_cash_flows.c.batch_id == batch["id"],
            )
        ):
            flow_types.add(str(row.flow_type))
    assert "external_deposit_open" in flow_types


def test_performance_summary_reports_twr_recovery_and_xirr(
    database_url: str, tmp_path
) -> None:
    store, simulation, batch = _create_batch(database_url, tmp_path)
    store.process_batch(
        batch["id"],
        minute_bars=_bars(),
        closing_prices={"SH600000": {"price": 10.0, "market_date": TRADE_DATE.isoformat()}},
        execution_evidence=_execution_evidence(batch["id"], simulation["execution_contract_hash"]),
    )
    summary = store.performance_summary(simulation["id"])
    assert summary["nav_days"] == 1
    assert summary["external_flow_count"] == 0
    unitized = summary["unitized"]
    assert unitized["status"] == "ok"
    assert unitized["observations"] == 1
    assert unitized["max_drawdown"] == pytest.approx(0.0)
    assert unitized["recovery_trading_days"] == 0
    xirr_result = summary["xirr"]
    # 夹具的批次交易日早于账户创建时钟日，年化符号无意义；只要求有解。
    assert xirr_result["status"] == "ok"
    assert isinstance(xirr_result["rate"], float)
    assert summary["cny_nav_latest"] > 0.0


def test_performance_summary_without_nav_is_insufficient(
    database_url: str, tmp_path
) -> None:
    store, simulation, _batch = _create_batch(database_url, tmp_path)
    summary = store.performance_summary(simulation["id"])
    assert summary["unitized"]["status"] == "insufficient_evidence"
    assert summary["xirr"]["status"] == "insufficient_evidence"
    assert summary["cny_nav_latest"] is None
