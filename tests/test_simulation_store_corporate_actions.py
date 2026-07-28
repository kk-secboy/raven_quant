from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from governance_fixtures import DATASET_IDENTITY, create_strategy_version
from sqlalchemy import select, update

from quant_data.database import (
    simulation_cash_flows,
    simulation_dividend_actions,
    simulation_dividend_entitlements,
    simulation_position_lots,
    strategy_versions,
)
from quant_data.execution_contract import (
    MINUTE_EXECUTION_CONTRACT_VERSION,
    MINUTE_SOURCE_UNIT_CONTRACTS,
)
from quant_platform.corporate_actions import corporate_actions_sha256
from quant_platform.cost_model import COST_SCHEDULE_VERSION
from quant_platform.portfolio_policy import POLICY_VERSION
from quant_platform.qlib_backtest import QLIB_ENGINE_VERSION
from quant_platform.recommendation_store import RecommendationStore
from quant_platform.simulation_store import SimulationStore

SOURCE_LINEAGE = "9" * 64
EXECUTION_IDENTITY = "d" * 64
EXECUTION_LINEAGE = "e" * 64

DAY_BUY = date(2026, 7, 13)
DAY_EX = date(2026, 7, 14)
DAY_SELL = date(2026, 7, 15)
DAY_PAY = date(2026, 7, 20)

ACTION = {
    "instrument": "SH600000",
    "ex_date": DAY_EX.isoformat(),
    "record_date": DAY_BUY.isoformat(),
    "pay_date": "2026-07-17",
    "cash_div_pretax": 0.5,
    "cash_div_aftertax": 0.5,
    "bonus_share_ratio": 0.2,
    "conversion_ratio": 0.1,
    "list_date": DAY_SELL.isoformat(),
    "source_ts_code": "600000.SH",
}


def _daily_dataset() -> dict:
    return {
        "name": "snapshot",
        "provenance": {
            "frequency": "day",
            "dataset_identity_sha256": DATASET_IDENTITY,
            "dataset_lineage_id": "b" * 64,
            "source_lineage_id": SOURCE_LINEAGE,
            "field_contract_version": "daily-qlib-field-v3-cny-amount",
            "source_volume_unit": "hand",
            "qlib_volume_unit": "share",
            "source_amount_unit": "thousand_cny",
            "qlib_amount_unit": "cny",
            "source_hand_size": 100,
            "index_volume_policy": "excluded_non_tradable_benchmark",
            "lineage_verified": True,
        },
    }


def _execution_dataset() -> dict:
    return {
        "name": "snapshot-5min",
        "provenance": {
            "frequency": "5min",
            "dataset_identity_sha256": EXECUTION_IDENTITY,
            "dataset_lineage_id": EXECUTION_LINEAGE,
            "source_lineage_id": SOURCE_LINEAGE,
            "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
            "fields": ["vwap", "volume", "paused", "up_limit", "down_limit"],
            "source_datasets": ["ashare_5m"],
            "source_unit_contracts": {
                "ashare_5m": MINUTE_SOURCE_UNIT_CONTRACTS["ashare_5m"]
            },
            "lineage_verified": True,
        },
    }


def _evidence(
    batch_id: str,
    contract_hash: str,
    actions: list[dict] | None,
    trade_date: date,
) -> dict:
    evidence = {
        "batch_id": batch_id,
        "dataset_identity_sha256": EXECUTION_IDENTITY,
        "dataset_lineage_id": EXECUTION_LINEAGE,
        "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
        "execution_contract_hash": contract_hash,
        "next_trade_date": (
            pd.Timestamp(trade_date) + pd.offsets.BusinessDay(1)
        ).date().isoformat(),
    }
    if actions is not None:
        evidence["corporate_actions_sha256"] = corporate_actions_sha256(actions)
    return evidence


def _bars(day: date) -> pd.DataFrame:
    return _instrument_bars("SH600000", day)


def _instrument_bars(instrument: str, day: date) -> pd.DataFrame:
    # 两个切片时点：100 股以上目标会拆成 10:00 / 10:20 两个 TWAP 切片。
    return pd.DataFrame(
        [
            {
                "datetime": f"{day.isoformat()} {slot}",
                "instrument": instrument,
                "close": 10.0,
                "vwap": 10.0,
                "volume": 1_000_000,
                "paused": 0,
                "up_limit": 11.0,
                "down_limit": 9.0,
            }
            for slot in ("10:00:00", "10:20:00")
        ]
    )


def _no_trade_bars(day: date) -> pd.DataFrame:
    """无持仓证券的执行 Bar：目标回退为当前持仓，当天不发生交易。"""

    return _instrument_bars("SH600001", day)


def _setup(database_url: str, tmp_path):
    version_id = create_strategy_version(
        database_url,
        tmp_path,
        config_overrides={"execution_frequency": "5min", "execution_method": "twap"},
    )
    recommendations = RecommendationStore(database_url)
    with recommendations.engine.begin() as connection:
        connection.execute(
            update(strategy_versions)
            .where(strategy_versions.c.id == version_id)
            .values(status="approved")
        )
    recommendation = recommendations.create(
        name="corporate action target",
        strategy_version_id=version_id,
        dataset="snapshot",
        hypothetical_initial_value=1_000_000,
        actor="test",
    )
    store = SimulationStore(database_url)
    simulation = store.create(
        name="corporate action simulation",
        recommendation_portfolio_id=recommendation["id"],
        daily_dataset=_daily_dataset(),
        execution_dataset=_execution_dataset(),
        initial_cash=1_000_000,
        execution_policy={"execution_algorithm": "twap"},
        cost_schedule_version=COST_SCHEDULE_VERSION,
        actor="test",
    )
    store.set_status(simulation["id"], "active")
    return store, recommendations, recommendation, simulation, version_id


def _make_batch(
    recommendations: RecommendationStore,
    store: SimulationStore,
    *,
    recommendation_id: str,
    version_id: str,
    as_of: date,
    effective: date,
    holdings: list[dict],
) -> dict:
    snapshot, _ = recommendations.create_snapshot(
        portfolio_id=recommendation_id,
        as_of_date=as_of,
        dataset="snapshot",
        dataset_identity_sha256=DATASET_IDENTITY,
    )
    recommendations.apply_result(
        snapshot["id"],
        {
            "status": "ok",
            "portfolio_id": recommendation_id,
            "strategy_version_id": version_id,
            "dataset": "snapshot",
            "dataset_identity_sha256": DATASET_IDENTITY,
            "as_of_date": as_of.isoformat(),
            "effective_date": effective.isoformat(),
            "policy_version": POLICY_VERSION,
            "backtest_engine_version": QLIB_ENGINE_VERSION,
            "cost_model": snapshot["cost_model"],
            "cash_weight": 1.0 - sum(item["weight"] for item in holdings),
            "holdings": holdings,
        },
    )
    batch, created = store.create_batch_for_snapshot(snapshot["id"])
    assert created is True
    return batch


def _holding(weight: float) -> list[dict]:
    return [
        {
            "instrument": "SH600000",
            "weight": weight,
            "previous_weight": weight,
            "weight_change": 0.0,
            "action": "hold",
            "reason": "governed target",
        }
    ]


def test_corporate_action_full_lifecycle(database_url: str, tmp_path) -> None:
    store, recommendations, recommendation, simulation, version_id = _setup(
        database_url, tmp_path
    )
    contract_hash = simulation["execution_contract_hash"]

    # Day 1：按目标买入 100 股（除权登记日收盘持有）。
    batch = _make_batch(
        recommendations,
        store,
        recommendation_id=recommendation["id"],
        version_id=version_id,
        as_of=date(2026, 7, 10),
        effective=DAY_BUY,
        holdings=[
            {
                "instrument": "SH600000",
                "weight": 0.001,
                "previous_weight": 0.0,
                "weight_change": 0.001,
                "action": "increase",
                "reason": "governed target",
            }
        ],
    )
    completed = store.process_batch(
        batch["id"],
        minute_bars=_bars(DAY_BUY),
        closing_prices={"SH600000": {"price": 10.0, "market_date": DAY_BUY.isoformat()}},
        execution_evidence=_evidence(batch["id"], contract_hash, [], DAY_BUY),
        corporate_actions=[],
    )
    assert completed["status"] == "succeeded"
    positions = store.rows(simulation["id"], "positions")
    assert positions[0]["quantity"] == 100

    # Day 2（除权日）：应收 100×0.5=50；送转 30 股；批次/权利/行动全部落库。
    batch_ex = _make_batch(
        recommendations,
        store,
        recommendation_id=recommendation["id"],
        version_id=version_id,
        as_of=DAY_BUY,
        effective=DAY_EX,
        holdings=_holding(0.001),
    )
    actions = [dict(ACTION)]
    completed = store.process_batch(
        batch_ex["id"],
        minute_bars=_no_trade_bars(DAY_EX),
        closing_prices={"SH600000": {"price": 10.0, "market_date": DAY_EX.isoformat()}},
        execution_evidence=_evidence(
            batch_ex["id"], contract_hash, actions, DAY_EX
        ),
        corporate_actions=actions,
    )
    assert completed["status"] == "succeeded"
    assert completed["summary"]["corporate_actions"] == 1
    with store.engine.connect() as connection:
        action_rows = connection.execute(
            select(simulation_dividend_actions).where(
                simulation_dividend_actions.c.portfolio_id == simulation["id"]
            )
        ).all()
        lot_rows = connection.execute(
            select(simulation_position_lots)
            .where(simulation_position_lots.c.portfolio_id == simulation["id"])
            .order_by(simulation_position_lots.c.lot_key)
        ).all()
        entitlement_rows = connection.execute(
            select(simulation_dividend_entitlements).where(
                simulation_dividend_entitlements.c.portfolio_id == simulation["id"]
            )
        ).all()
    assert len(action_rows) == 1
    action_row = action_rows[0]
    assert action_row.status == "accrued"
    assert float(action_row.receivable_amount) == pytest.approx(50.0)
    # 除权计提：批次取得日 2026-07-13 → 除权 2026-07-14 ≤1 个月 → 20% 档，
    # 负债 = 100×(0.5+0.2)×20% = 14（设计 §5.6 保守负债，NAV 减项）。
    assert float(action_row.tax_liability_amount) == pytest.approx(14.0)
    assert action_row.eligible_quantity == 100
    assert action_row.new_shares == 30
    assert action_row.tax_rule_version == "cn-dividend-tax-2015-09-08"
    assert len(lot_rows) == 2
    bonus = [row for row in lot_rows if row.origin == "bonus_share"][0]
    assert bonus.quantity == 30
    assert bonus.sellable_from == DAY_SELL
    assert bonus.acquired_at == DAY_BUY  # 取得日继承父批次（持有期连续计算）
    assert {row.kind for row in entitlement_rows} == {"cash", "bonus_par"}
    assert all(row.untaxed_quantity == 100 for row in entitlement_rows)
    entitlement_liability = {row.kind: float(row.liability_per_share) for row in entitlement_rows}
    assert entitlement_liability == {
        "cash": pytest.approx(0.5 * 0.20),
        "bonus_par": pytest.approx(0.2 * 0.20),
    }
    positions = store.rows(simulation["id"], "positions")
    assert positions[0]["quantity"] == 130
    nav_rows = store.rows(simulation["id"], "nav")
    assert nav_rows[-1]["corporate_receivables"] == pytest.approx(50.0)
    assert nav_rows[-1]["corporate_tax_liabilities"] == pytest.approx(14.0)
    # NAV = 现金 + 市值 + 应收 − 应付税负债（旧行为不含减项，虚高 14）
    assert nav_rows[-1]["nav"] == pytest.approx(
        nav_rows[-1]["cash"] + nav_rows[-1]["market_value"] + 50.0 - 14.0
    )

    # Day 3（新增股份上市日）：全部卖出；红利税 100×0.5×20% + 100×0.2×20% = 14。
    batch_sell = _make_batch(
        recommendations,
        store,
        recommendation_id=recommendation["id"],
        version_id=version_id,
        as_of=DAY_EX,
        effective=DAY_SELL,
        holdings=[],
    )
    completed = store.process_batch(
        batch_sell["id"],
        minute_bars=_bars(DAY_SELL),
        closing_prices={"SH600000": {"price": 10.0, "market_date": DAY_SELL.isoformat()}},
        execution_evidence=_evidence(
            batch_sell["id"], contract_hash, [], DAY_SELL
        ),
        corporate_actions=[],
    )
    assert completed["status"] == "succeeded"
    with store.engine.connect() as connection:
        tax_flows = connection.execute(
            select(simulation_cash_flows).where(
                simulation_cash_flows.c.portfolio_id == simulation["id"],
                simulation_cash_flows.c.flow_type == "dividend_tax",
            )
        ).all()
    assert len(tax_flows) >= 1
    # 卖出仍在 20% 档：实际税额 = 已提负债 = 14，差额确认为 0，NAV 不跳变
    assert sum(float(flow.amount) for flow in tax_flows) == pytest.approx(-14.0)
    assert store.rows(simulation["id"], "positions") == []
    nav_after_sale = store.rows(simulation["id"], "nav")[-1]
    assert nav_after_sale["corporate_tax_liabilities"] == pytest.approx(0.0)

    # Day 4（到账日后）：应收 50 重分类为现金，NAV 不变，状态 paid；清仓后应收仍在。
    batch_pay = _make_batch(
        recommendations,
        store,
        recommendation_id=recommendation["id"],
        version_id=version_id,
        as_of=date(2026, 7, 17),
        effective=DAY_PAY,
        holdings=[],
    )
    nav_before = store.rows(simulation["id"], "nav")[-1]["nav"]
    completed = store.process_batch(
        batch_pay["id"],
        minute_bars=_no_trade_bars(DAY_PAY),
        closing_prices={"SH600000": {"price": 10.0, "market_date": DAY_PAY.isoformat()}},
        execution_evidence=_evidence(
            batch_pay["id"], contract_hash, actions, DAY_PAY
        ),
        corporate_actions=actions,
    )
    assert completed["status"] == "succeeded"
    nav_after = store.rows(simulation["id"], "nav")[-1]
    assert nav_after["nav"] == pytest.approx(nav_before)
    assert nav_after["corporate_receivables"] == pytest.approx(0.0)
    with store.engine.connect() as connection:
        action_row = connection.execute(
            select(simulation_dividend_actions).where(
                simulation_dividend_actions.c.portfolio_id == simulation["id"]
            )
        ).one()
        pay_flows = connection.execute(
            select(simulation_cash_flows).where(
                simulation_cash_flows.c.portfolio_id == simulation["id"],
                simulation_cash_flows.c.flow_type == "dividend_payment",
            )
        ).all()
    assert action_row.status == "paid"
    assert action_row.paid_batch_id == batch_pay["id"]
    assert len(pay_flows) == 1
    assert float(pay_flows[0].amount) == pytest.approx(50.0)


def test_corporate_actions_evidence_mismatch_fails_closed(database_url: str, tmp_path) -> None:
    store, recommendations, recommendation, simulation, version_id = _setup(
        database_url, tmp_path
    )
    batch = _make_batch(
        recommendations,
        store,
        recommendation_id=recommendation["id"],
        version_id=version_id,
        as_of=date(2026, 7, 10),
        effective=DAY_BUY,
        holdings=_holding(0.001),
    )
    with pytest.raises(ValueError, match="corporate actions"):
        store.process_batch(
            batch["id"],
            minute_bars=_bars(DAY_BUY),
            closing_prices={"SH600000": {"price": 10.0, "market_date": DAY_BUY.isoformat()}},
            execution_evidence=_evidence(
                batch["id"],
                simulation["execution_contract_hash"],
                [],
                DAY_BUY,
            ),
            corporate_actions=[dict(ACTION)],  # 与证据哈希（空列表）不一致
        )
