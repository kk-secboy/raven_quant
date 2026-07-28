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

EVIDENCE_A = "a" * 64
EVIDENCE_B = "b" * 64


def _settled_fill(database_url: str, tmp_path):
    store, simulation, batch = _create_batch(database_url, tmp_path)
    store.process_batch(
        batch["id"],
        minute_bars=_bars(),
        closing_prices={
            "SH600000": {
                "price": 10.0,
                "market_date": TRADE_DATE.isoformat(),
            }
        },
        execution_evidence=_execution_evidence(
            batch["id"], simulation["execution_contract_hash"]
        ),
    )
    fill = store.rows(simulation["id"], "fills")[0]
    return store, simulation, batch, fill


def test_final_fee_applies_only_delta_and_replay_is_idempotent(
    database_url: str, tmp_path
) -> None:
    store, simulation, batch, fill = _settled_fill(database_url, tmp_path)
    before = store.get(simulation["id"])
    initial_fee = float(fill["fee"])
    final_fee = initial_fee + 2.5

    first = store.record_final_fee(
        simulation["id"],
        fill_id=fill["id"],
        final_fee=final_fee,
        evidence_sha256=EVIDENCE_A,
        source="end_of_day",
        actor="fee-operator",
    )
    assert first["created"] is True
    assert float(first["previously_confirmed_fee"]) == pytest.approx(initial_fee)
    assert float(first["adjustment_amount"]) == pytest.approx(2.5)
    corrected = store.get(simulation["id"])
    assert float(corrected["cash"]) == pytest.approx(float(before["cash"]) - 2.5)
    assert float(corrected["nav"]) == pytest.approx(float(before["nav"]) - 2.5)
    assert float(store.rows(simulation["id"], "fills")[0]["fee"]) == pytest.approx(
        initial_fee
    )

    replay = store.record_final_fee(
        simulation["id"],
        fill_id=fill["id"],
        final_fee=final_fee,
        evidence_sha256=EVIDENCE_A,
        source="end_of_day",
        actor="fee-operator",
    )
    assert replay["created"] is False
    assert store.get(simulation["id"])["cash"] == corrected["cash"]
    assert len(store.rows(simulation["id"], "fee_adjustments")) == 1
    with store.engine.connect() as connection:
        fee_flows = connection.execute(
            select(simulation_cash_flows).where(
                simulation_cash_flows.c.portfolio_id == simulation["id"],
                simulation_cash_flows.c.batch_id == batch["id"],
                simulation_cash_flows.c.flow_type == "fee_adjustment",
            )
        ).all()
    assert len(fee_flows) == 1
    assert float(fee_flows[0].amount) == pytest.approx(-2.5)


def test_revised_final_fee_refunds_only_prior_overconfirmation(
    database_url: str, tmp_path
) -> None:
    store, simulation, _batch, fill = _settled_fill(database_url, tmp_path)
    initial_fee = float(fill["fee"])
    first = store.record_final_fee(
        simulation["id"],
        fill_id=fill["id"],
        final_fee=initial_fee + 3.0,
        evidence_sha256=EVIDENCE_A,
        actor="fee-operator",
    )
    cash_after_first = float(store.get(simulation["id"])["cash"])
    second = store.record_final_fee(
        simulation["id"],
        fill_id=fill["id"],
        final_fee=initial_fee + 1.0,
        evidence_sha256=EVIDENCE_B,
        actor="fee-operator",
    )

    assert float(second["previously_confirmed_fee"]) == pytest.approx(
        initial_fee + 3.0
    )
    assert float(second["adjustment_amount"]) == pytest.approx(-2.0)
    assert float(store.get(simulation["id"])["cash"]) == pytest.approx(
        cash_after_first + 2.0
    )
    assert float(first["adjustment_amount"]) == pytest.approx(3.0)


def test_final_fee_key_cannot_be_reused_for_different_payload(
    database_url: str, tmp_path
) -> None:
    store, simulation, _batch, fill = _settled_fill(database_url, tmp_path)
    key = "statement-line-1"
    store.record_final_fee(
        simulation["id"],
        fill_id=fill["id"],
        final_fee=float(fill["fee"]) + 1.0,
        evidence_sha256=EVIDENCE_A,
        adjustment_key=key,
        actor="fee-operator",
    )
    with pytest.raises(ValueError, match="different payload"):
        store.record_final_fee(
            simulation["id"],
            fill_id=fill["id"],
            final_fee=float(fill["fee"]) + 2.0,
            evidence_sha256=EVIDENCE_B,
            adjustment_key=key,
            actor="fee-operator",
        )


def test_final_fee_requires_valid_evidence(database_url: str, tmp_path) -> None:
    store, simulation, _batch, fill = _settled_fill(database_url, tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        store.record_final_fee(
            simulation["id"],
            fill_id=fill["id"],
            final_fee=float(fill["fee"]),
            evidence_sha256="not-a-hash",
            actor="fee-operator",
        )
