"""Design-gap items: versioned market permissions (8.7) + manual/CSV shadow accounts (8.6)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from test_recommendation_actions import _make_succeeded_snapshot

from quant_platform.market_permission import (
    MarketPermissionStore,
    instrument_scopes,
)
from quant_platform.shadow_account import ShadowAccountStore

RULE_DATE = date(2026, 7, 13)  # effective_date of the fixture snapshot
NOW = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)


def _permissions(database_url: str) -> MarketPermissionStore:
    return MarketPermissionStore(database_url)


def _shadow(database_url: str) -> ShadowAccountStore:
    return ShadowAccountStore(database_url)


# ---------------------------------------------------------------------------
# 8.7 版本化个人市场权限
# ---------------------------------------------------------------------------


def test_instrument_scopes_classification() -> None:
    assert instrument_scopes("SH600000") == [("board", "main"), ("exchange", "SSE")]
    assert instrument_scopes("SZ000001") == [("board", "main"), ("exchange", "SZSE")]
    assert instrument_scopes("SH688001") == [("board", "star"), ("exchange", "SSE")]
    assert instrument_scopes("SZ300750") == [("board", "chinext"), ("exchange", "SZSE")]
    assert instrument_scopes("SH510300") == [("etf_subtype", "etf_51"), ("exchange", "SSE")]
    with pytest.raises(ValueError, match="order-unit|permission scopes"):
        instrument_scopes("XX")


def test_permission_versions_tighten_only(database_url: str) -> None:
    store = _permissions(database_url)
    store.record_version(
        scope_type="board",
        scope_key="main",
        permission="buy_sell",
        confirmation_source="券商科创板/主板权限确认截图 2026-01",
        as_of=date(2026, 1, 1),
        actor="user",
    )
    tightened = store.record_version(
        scope_type="board",
        scope_key="main",
        permission="sell_only",
        confirmation_source="用户主动收紧：只做减仓",
        as_of=date(2026, 2, 1),
        actor="user",
    )
    assert tightened["permission"] == "sell_only"
    assert tightened["supersedes_id"]
    with pytest.raises(ValueError, match="only be tightened"):
        store.record_version(
            scope_type="board",
            scope_key="main",
            permission="buy_sell",
            confirmation_source="试图静默放宽",
            as_of=date(2026, 3, 1),
            actor="user",
        )
    relaxed = store.record_version(
        scope_type="board",
        scope_key="main",
        permission="buy_sell",
        confirmation_source="券商权限恢复确认函 2026-03",
        as_of=date(2026, 3, 1),
        actor="user",
        relaxation_confirmed=True,
    )
    assert relaxed["relaxation_confirmed"] is True
    effective = store.effective_permission("board", "main", on_date=RULE_DATE)
    assert effective["permission"] == "buy_sell"


def test_permission_requires_confirmation_source(database_url: str) -> None:
    store = _permissions(database_url)
    with pytest.raises(ValueError, match="confirmation source"):
        store.record_version(
            scope_type="board",
            scope_key="star",
            permission="buy_sell",
            confirmation_source="  ",
            as_of=date(2026, 1, 1),
            actor="user",
        )
    with pytest.raises(ValueError, match="unknown market permission"):
        store.record_version(
            scope_type="board",
            scope_key="star",
            permission="margin_ok",
            confirmation_source="x",
            as_of=date(2026, 1, 1),
            actor="user",
        )


def test_effective_permission_missing_and_expired_is_unknown(database_url: str) -> None:
    store = _permissions(database_url)
    missing = store.effective_permission("board", "star", on_date=RULE_DATE)
    assert missing["permission"] == "unknown"
    assert missing["reason"] == "no_confirmed_permission"
    store.record_version(
        scope_type="board",
        scope_key="star",
        permission="buy_sell",
        confirmation_source="科创板权限确认",
        as_of=date(2026, 1, 1),
        valid_until=date(2026, 6, 30),
        actor="user",
    )
    valid = store.effective_permission("board", "star", on_date=date(2026, 6, 30))
    assert valid["permission"] == "buy_sell"
    expired = store.effective_permission("board", "star", on_date=RULE_DATE)
    assert expired["permission"] == "unknown"
    assert expired["reason"].startswith("permission_expired")


def test_disabled_permission_blocks_buy_with_reason(database_url: str, tmp_path) -> None:
    recommendations, snapshot_id = _make_succeeded_snapshot(database_url, tmp_path)
    store = _permissions(database_url)
    store.record_version(
        scope_type="board",
        scope_key="main",
        permission="disabled",
        confirmation_source="账户被券商限制交易",
        as_of=date(2026, 1, 1),
        actor="user",
    )
    updated = recommendations.attach_account_actions(
        snapshot_id,
        account_state={"SH600000": {"reference_price": 10.0, "filled_position": 0}},
        now=NOW,
        permission_store=store,
    )
    items = {item["instrument"]: item for item in updated["account_actions"]["items"]}
    assert items["SH600000"]["action"] == "BUY"  # 动作保留，不静默删除
    assert items["SH600000"]["execution_state"] == "BLOCKED"
    assert "market_permission_disabled" in items["SH600000"]["blocked_reason"]


def test_sell_only_blocks_buy_but_allows_exit(database_url: str, tmp_path) -> None:
    recommendations, snapshot_id = _make_succeeded_snapshot(database_url, tmp_path)
    store = _permissions(database_url)
    store.record_version(
        scope_type="board",
        scope_key="main",
        permission="sell_only",
        confirmation_source="用户收紧为只减仓",
        as_of=date(2026, 1, 1),
        actor="user",
    )
    updated = recommendations.attach_account_actions(
        snapshot_id,
        account_state={
            "SH600000": {"reference_price": 10.0, "filled_position": 0},
            # 不在目标内的持仓：sell_only 下 EXIT 必须仍然放行（8.7 只减仓）。
            "SZ000001": {"filled_position": 200, "sellable_quantity": 200},
        },
        now=NOW,
        permission_store=store,
    )
    items = {item["instrument"]: item for item in updated["account_actions"]["items"]}
    assert items["SH600000"]["action"] == "BUY"
    assert items["SH600000"]["execution_state"] == "BLOCKED"
    assert "market_permission_sell_only" in items["SH600000"]["blocked_reason"]
    assert items["SZ000001"]["action"] == "EXIT"
    assert items["SZ000001"]["execution_state"] == "READY"
    assert items["SZ000001"]["blocked_reason"] is None


def test_unknown_or_expired_permission_marks_simulation_only(database_url: str, tmp_path) -> None:
    recommendations, snapshot_id = _make_succeeded_snapshot(database_url, tmp_path)
    store = _permissions(database_url)
    updated = recommendations.attach_account_actions(
        snapshot_id,
        account_state={"SH600000": {"reference_price": 10.0, "filled_position": 0}},
        now=NOW,
        permission_store=store,
    )
    items = {item["instrument"]: item for item in updated["account_actions"]["items"]}
    assert items["SH600000"]["action"] == "BUY"
    assert items["SH600000"]["execution_state"] == "WAIT"
    assert "market_permission_simulation_only" in items["SH600000"]["wait_reason"]

    store.record_version(
        scope_type="board",
        scope_key="main",
        permission="buy_sell",
        confirmation_source="主板权限确认（年度复核到期）",
        as_of=date(2026, 1, 1),
        valid_until=date(2026, 6, 30),
        actor="user",
    )
    expired = recommendations.attach_account_actions(
        snapshot_id,
        account_state={"SH600000": {"reference_price": 10.0, "filled_position": 0}},
        now=NOW,
        permission_store=store,
    )
    expired_item = {
        item["instrument"]: item for item in expired["account_actions"]["items"]
    }["SH600000"]
    assert expired_item["execution_state"] == "WAIT"
    assert "permission_expired" in expired_item["wait_reason"]


def test_confirmed_buy_sell_leaves_actions_untouched(database_url: str, tmp_path) -> None:
    recommendations, snapshot_id = _make_succeeded_snapshot(database_url, tmp_path)
    store = _permissions(database_url)
    store.record_version(
        scope_type="board",
        scope_key="main",
        permission="buy_sell",
        confirmation_source="主板权限确认",
        as_of=date(2026, 1, 1),
        actor="user",
    )
    updated = recommendations.attach_account_actions(
        snapshot_id,
        account_state={"SH600000": {"reference_price": 10.0, "filled_position": 0}},
        now=NOW,
        permission_store=store,
    )
    items = {item["instrument"]: item for item in updated["account_actions"]["items"]}
    assert items["SH600000"]["action"] == "BUY"
    assert items["SH600000"]["execution_state"] == "READY"
    assert updated["account_actions"]["account_context"]["account_type"] == "main_paper"


# ---------------------------------------------------------------------------
# 8.6 手工/CSV 影子账户
# ---------------------------------------------------------------------------


def test_csv_import_roundtrip(database_url: str) -> None:
    store = _shadow(database_url)
    snapshot = store.import_csv(
        account_id="shadow-1",
        content="instrument,quantity,sellable_quantity\nSH600000,1000,800\nSZ000001,200,200\n",
        cash=123456.78,
        imported_by="user",
    )
    assert snapshot["import_source"] == "csv"
    assert float(snapshot["cash"]) == pytest.approx(123456.78)
    assert snapshot["holdings"] == [
        {"instrument": "SH600000", "quantity": 1000, "sellable_quantity": 800},
        {"instrument": "SZ000001", "quantity": 200, "sellable_quantity": 200},
    ]
    assert len(snapshot["content_sha256"]) == 64
    latest = store.latest_snapshot("shadow-1")
    assert latest["id"] == snapshot["id"]


def test_csv_import_fail_closed(database_url: str) -> None:
    store = _shadow(database_url)
    with pytest.raises(ValueError, match="header"):
        store.import_csv(
            account_id="s", content="code,qty\nSH600000,100\n", imported_by="user"
        )
    with pytest.raises(ValueError, match="empty"):
        store.import_csv(account_id="s", content="  \n", imported_by="user")
    with pytest.raises(ValueError, match="no holdings rows"):
        store.import_csv(
            account_id="s",
            content="instrument,quantity,sellable_quantity\n",
            imported_by="user",
        )
    with pytest.raises(ValueError, match="column count"):
        store.import_csv(
            account_id="s",
            content="instrument,quantity,sellable_quantity\nSH600000,100\n",
            imported_by="user",
        )
    with pytest.raises(ValueError, match="integer"):
        store.import_csv(
            account_id="s",
            content="instrument,quantity,sellable_quantity\nSH600000,10.5,10\n",
            imported_by="user",
        )
    with pytest.raises(ValueError, match="within the position"):
        store.import_csv(
            account_id="s",
            content="instrument,quantity,sellable_quantity\nSH600000,100,800\n",
            imported_by="user",
        )
    # 失败导入不落任何快照。
    assert store.latest_snapshot("s") is None


def test_manual_import_schema_fail_closed(database_url: str) -> None:
    store = _shadow(database_url)
    with pytest.raises(ValueError, match="unsupported fields"):
        store.import_snapshot(
            account_id="s",
            cash=0,
            holdings=[{"instrument": "SH600000", "quantity": 100, "price": 10.0}],
            imported_by="user",
        )
    with pytest.raises(ValueError, match="negative"):
        store.import_snapshot(
            account_id="s",
            cash=0,
            holdings=[{"instrument": "SH600000", "quantity": -100}],
            imported_by="user",
        )
    with pytest.raises(ValueError, match="side buy/sell"):
        store.import_snapshot(
            account_id="s",
            cash=0,
            holdings=[],
            open_orders=[{"instrument": "SH600000", "side": "hold", "quantity": 100}],
            imported_by="user",
        )
    with pytest.raises(ValueError, match="non-negative"):
        store.import_snapshot(account_id="s", cash=-1, holdings=[], imported_by="user")
    snapshot = store.import_snapshot(
        account_id="s",
        cash=1000,
        holdings=[{"instrument": "SH600000", "quantity": 300, "sellable_quantity": 200}],
        open_orders=[{"instrument": "SZ000001", "side": "sell", "quantity": 100}],
        imported_by="user",
        notes="手工录入",
    )
    assert snapshot["import_source"] == "manual"
    assert snapshot["open_orders"][0]["side"] == "sell"


def test_shadow_freshness_natural_days(database_url: str) -> None:
    store = _shadow(database_url)
    store.import_snapshot(
        account_id="fresh-acc",
        cash=0,
        holdings=[{"instrument": "SH600000", "quantity": 100}],
        imported_by="user",
        imported_at=NOW - timedelta(days=1),
    )
    store.import_snapshot(
        account_id="stale-acc",
        cash=0,
        holdings=[{"instrument": "SH600000", "quantity": 100}],
        imported_by="user",
        imported_at=NOW - timedelta(days=5),
    )
    assert store.freshness("fresh-acc", now=NOW)["status"] == "fresh"
    stale = store.freshness("stale-acc", now=NOW)
    assert stale["status"] == "stale"
    assert stale["age_days"] == 5
    assert store.freshness("no-such-acc", now=NOW)["status"] == "missing"


def test_fresh_shadow_drives_precise_actions(database_url: str, tmp_path) -> None:
    recommendations, snapshot_id = _make_succeeded_snapshot(database_url, tmp_path)
    shadow = _shadow(database_url)
    shadow.import_snapshot(
        account_id="shadow-1",
        cash=500_000,
        holdings=[{"instrument": "SH600000", "quantity": 0, "sellable_quantity": 0}],
        open_orders=[{"instrument": "SH600000", "side": "buy", "quantity": 100}],
        imported_by="user",
        imported_at=NOW - timedelta(hours=3),
    )
    built = shadow.account_state_for_actions("shadow-1", now=NOW)
    assert built["account_context"]["account_type"] == "manual_shadow"
    assert built["account_context"]["degraded"] is False
    state = dict(built["account_state"])
    state["SH600000"]["reference_price"] = 10.0
    updated = recommendations.attach_account_actions(
        snapshot_id,
        account_state=state,
        now=NOW,
        account_context=built["account_context"],
    )
    plan = updated["account_actions"]
    assert plan["account_context"]["account_type"] == "manual_shadow"
    items = {item["instrument"]: item for item in plan["items"]}
    assert items["SH600000"]["action"] == "BUY"
    assert items["SH600000"]["execution_state"] == "READY"
    assert [entry["op"] for entry in items["SH600000"]["order_plan"]] == ["keep"]
    assert items["SH600000"]["projected_position"] == 100


def test_stale_shadow_degrades_to_simulation_only(database_url: str, tmp_path) -> None:
    recommendations, snapshot_id = _make_succeeded_snapshot(database_url, tmp_path)
    shadow = _shadow(database_url)
    shadow.import_snapshot(
        account_id="shadow-1",
        cash=500_000,
        holdings=[{"instrument": "SH600000", "quantity": 100, "sellable_quantity": 100}],
        open_orders=[{"instrument": "SH600000", "side": "buy", "quantity": 100}],
        imported_by="user",
        imported_at=NOW - timedelta(days=6),
    )
    built = shadow.account_state_for_actions("shadow-1", now=NOW)
    assert built["account_context"]["degraded"] is True
    assert built["account_context"]["freshness"]["status"] == "stale"
    state = dict(built["account_state"])
    state["SH600000"]["reference_price"] = 10.0
    # 陈旧状态下未完成的影子订单不再被信任（不进入 projected_position）。
    assert state["SH600000"]["open_orders"] == []
    updated = recommendations.attach_account_actions(
        snapshot_id,
        account_state=state,
        now=NOW,
        account_context=built["account_context"],
    )
    items = {item["instrument"]: item for item in updated["account_actions"]["items"]}
    assert items["SH600000"]["execution_state"] == "WAIT"
    assert "manual_shadow_stale" in items["SH600000"]["wait_reason"]
    assert "simulation_only" in items["SH600000"]["wait_reason"]
    assert updated["account_actions"]["account_context"]["degraded"] is True


def test_missing_shadow_is_degraded_not_simulation_fallback(database_url: str, tmp_path) -> None:
    recommendations, snapshot_id = _make_succeeded_snapshot(database_url, tmp_path)
    shadow = _shadow(database_url)
    built = shadow.account_state_for_actions("no-such-account", now=NOW)
    assert built["account_state"] == {}
    assert built["account_context"]["freshness"]["status"] == "missing"
    assert built["account_context"]["degraded"] is True
    # 不静默换源：缺失影子状态时没有持仓/可卖数据注入，目标权重仍可见。
    updated = recommendations.attach_account_actions(
        snapshot_id,
        account_state={"SH600000": {"reference_price": 10.0}},
        now=NOW,
        account_context=built["account_context"],
    )
    items = {item["instrument"]: item for item in updated["account_actions"]["items"]}
    assert items["SH600000"]["target_quantity"] == 100
    assert items["SH600000"]["filled_position"] == 0
    assert updated["account_actions"]["account_context"]["account_type"] == "manual_shadow"


def test_shadow_and_paper_accounts_stay_separate(database_url: str) -> None:
    shadow = _shadow(database_url)
    shadow.import_snapshot(
        account_id="shadow-a",
        cash=1,
        holdings=[{"instrument": "SH600000", "quantity": 100}],
        imported_by="user",
    )
    shadow.import_csv(
        account_id="shadow-b",
        content="instrument,quantity,sellable_quantity\nSZ000001,200,200\n",
        imported_by="user",
    )
    accounts = {account["account_id"]: account for account in shadow.list_accounts()}
    assert set(accounts) == {"shadow-a", "shadow-b"}
    assert all(account["account_type"] == "manual_shadow" for account in accounts.values())
    assert accounts["shadow-b"]["import_source"] == "csv"
    assert accounts["shadow-b"]["holding_count"] == 1
