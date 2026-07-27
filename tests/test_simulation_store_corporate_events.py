from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from governance_fixtures import DATASET_IDENTITY, create_strategy_version
from sqlalchemy import select, update

from quant_data.database import simulation_corporate_events, strategy_versions
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
DAY_EVENT = date(2026, 7, 14)
DAY_SELL = date(2026, 7, 15)

EVENTS = [
    {  # 公告阶段：只产生信息事件，不改账
        "kind": "announcement",
        "instrument": "SH600000",
        "effective_date": DAY_EVENT.isoformat(),
        "stage": "plan",
        "source": "tushare_dividend",
        "details": {"ann_date": DAY_EVENT.isoformat()},
    },
    {  # 拆股：1 拆 2，数量与单位成本同步调整
        "kind": "split",
        "instrument": "SH600000",
        "effective_date": DAY_EVENT.isoformat(),
        "split_ratio": 2.0,
        "source": "adj_factor",
        "details": {"detection": "adj_factor_jump"},
    },
    {  # 需持有人选择：提醒 + 持仓标记
        "kind": "choice_required",
        "instrument": "SH600000",
        "effective_date": DAY_EVENT.isoformat(),
        "title": "配股发行公告",
        "source": "tushare_anns_d",
    },
    {  # 配股：无数据源支撑，fail-closed 记原因
        "kind": "unsupported",
        "instrument": "SH600000",
        "effective_date": DAY_EVENT.isoformat(),
        "unsupported_type": "rights_issue",
        "title": "配股发行公告",
    },
]


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


def _evidence(batch_id: str, contract_hash: str, events: list[dict] | None) -> dict:
    evidence = {
        "batch_id": batch_id,
        "dataset_identity_sha256": EXECUTION_IDENTITY,
        "dataset_lineage_id": EXECUTION_LINEAGE,
        "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
        "execution_contract_hash": contract_hash,
    }
    if events is not None:
        evidence["corporate_events_sha256"] = corporate_actions_sha256(events)
    return evidence


def _instrument_bars(instrument: str, day: date, price: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "datetime": f"{day.isoformat()} {slot}",
                "instrument": instrument,
                "close": price,
                "vwap": price,
                "volume": 1_000_000,
                "paused": 0,
                "up_limit": price * 1.1,
                "down_limit": price * 0.9,
            }
            for slot in ("10:00:00", "10:20:00")
        ]
    )


def _no_trade_bars(day: date) -> pd.DataFrame:
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
        name="corporate event target",
        strategy_version_id=version_id,
        dataset="snapshot",
        hypothetical_initial_value=1_000_000,
        actor="test",
    )
    store = SimulationStore(database_url)
    simulation = store.create(
        name="corporate event simulation",
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


def _buy_holding() -> list[dict]:
    return [
        {
            "instrument": "SH600000",
            "weight": 0.001,
            "previous_weight": 0.0,
            "weight_change": 0.001,
            "action": "increase",
            "reason": "governed target",
        }
    ]


def _event_rows(store: SimulationStore, portfolio_id: str):
    with store.engine.connect() as connection:
        return connection.execute(
            select(simulation_corporate_events)
            .where(simulation_corporate_events.c.portfolio_id == portfolio_id)
            .order_by(simulation_corporate_events.c.event_key)
        ).all()


def test_corporate_event_types_lifecycle(database_url: str, tmp_path) -> None:
    store, recommendations, recommendation, simulation, version_id = _setup(
        database_url, tmp_path
    )
    contract_hash = simulation["execution_contract_hash"]

    # Day 1：买入 100 股。
    batch = _make_batch(
        recommendations,
        store,
        recommendation_id=recommendation["id"],
        version_id=version_id,
        as_of=date(2026, 7, 10),
        effective=DAY_BUY,
        holdings=_buy_holding(),
    )
    completed = store.process_batch(
        batch["id"],
        minute_bars=_instrument_bars("SH600000", DAY_BUY),
        closing_prices={"SH600000": {"price": 10.0, "market_date": DAY_BUY.isoformat()}},
        execution_evidence=_evidence(batch["id"], contract_hash, None),
    )
    assert completed["status"] == "succeeded"
    bought = store.rows(simulation["id"], "positions")[0]
    assert bought["quantity"] == 100
    cost_before = float(bought["average_cost"])  # 含买入费用的单位成本

    # Day 2：公告 + 拆股 + 持有人选择 + unsupported 同日到账前处理。
    batch_event = _make_batch(
        recommendations,
        store,
        recommendation_id=recommendation["id"],
        version_id=version_id,
        as_of=DAY_BUY,
        effective=DAY_EVENT,
        holdings=_buy_holding(),
    )
    events = [dict(item) for item in EVENTS]
    completed = store.process_batch(
        batch_event["id"],
        minute_bars=_no_trade_bars(DAY_EVENT),
        closing_prices={"SH600000": {"price": 5.0, "market_date": DAY_EVENT.isoformat()}},
        execution_evidence=_evidence(batch_event["id"], contract_hash, events),
        corporate_events=events,
    )
    assert completed["status"] == "succeeded"
    assert completed["summary"]["corporate_events"] == len(EVENTS)
    position = store.rows(simulation["id"], "positions")[0]
    assert position["quantity"] == 200
    # 拆股摊薄单位成本（含费用成本同步减半），总成本不变，无虚假亏损。
    assert float(position["average_cost"]) == pytest.approx(cost_before / 2)
    rows = _event_rows(store, simulation["id"])
    assert len(rows) == len(EVENTS)
    assert {row.event_type for row in rows} == {
        "announcement",
        "split",
        "choice_required",
        "unsupported",
    }
    assert len({row.event_key for row in rows}) == len(EVENTS)  # 唯一事件键
    emitted_types = {
        row["event_type"] for row in store.rows(simulation["id"], "events")
    }
    assert "corporate_action_announcement" in emitted_types
    assert "corporate_action_split" in emitted_types
    assert "corporate_action_choice_required" in emitted_types
    assert "corporate_action_unsupported" in emitted_types
    # 拆股不产生现金：现金只含买入支出，NAV = 现金 + 市值（200×5）。
    nav_row = store.rows(simulation["id"], "nav")[-1]
    assert nav_row["market_value"] == pytest.approx(1000.0)
    assert nav_row["nav"] == pytest.approx(nav_row["cash"] + 1000.0)

    # Day 3：同批事件重复供给（重放）→ 幂等，台账不增行、数量不再翻倍；
    # 清仓卖出时 choice_required 标记重新推导并给出可见提示（不阻断）。
    batch_sell = _make_batch(
        recommendations,
        store,
        recommendation_id=recommendation["id"],
        version_id=version_id,
        as_of=DAY_EVENT,
        effective=DAY_SELL,
        holdings=[],
    )
    replayed = [dict(item) for item in EVENTS]
    completed = store.process_batch(
        batch_sell["id"],
        minute_bars=_instrument_bars("SH600000", DAY_SELL, price=5.0),
        closing_prices={"SH600000": {"price": 5.0, "market_date": DAY_SELL.isoformat()}},
        execution_evidence=_evidence(batch_sell["id"], contract_hash, replayed),
        corporate_events=replayed,
    )
    assert completed["status"] == "succeeded"
    assert completed["summary"]["corporate_events"] == 0
    assert len(_event_rows(store, simulation["id"])) == len(EVENTS)
    assert store.rows(simulation["id"], "positions") == []
    sale_warnings = [
        row
        for row in store.rows(simulation["id"], "events")
        if row["event_type"] == "corporate_action_choice_pending_sale"
    ]
    assert len(sale_warnings) == 1


def test_corporate_events_evidence_mismatch_fails_closed(
    database_url: str, tmp_path
) -> None:
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
        holdings=_buy_holding(),
    )
    with pytest.raises(ValueError, match="corporate events"):
        store.process_batch(
            batch["id"],
            minute_bars=_instrument_bars("SH600000", DAY_BUY),
            closing_prices={
                "SH600000": {"price": 10.0, "market_date": DAY_BUY.isoformat()}
            },
            execution_evidence=_evidence(
                batch["id"], simulation["execution_contract_hash"], []
            ),
            corporate_events=[dict(EVENTS[1])],  # 与证据哈希（空列表）不一致
        )
