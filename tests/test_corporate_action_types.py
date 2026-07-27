from datetime import date

import pandas as pd
import pytest

from quant_platform.corporate_actions import (
    CorporateEvent,
    apply_code_change,
    apply_share_split,
    detect_choice_required_events,
    detect_liquidation_events,
    detect_split_events,
    normalize_announcement_rows,
    normalize_namechange_rows,
)
from quant_platform.cost_model import CostModelConfig
from quant_platform.simulation_engine import execute_simulation_day

pytestmark = pytest.mark.no_database

_POLICY = {
    "execution_algorithm": "twap",
    "slice_minutes": 20,
    "max_slices": 1,
    "max_participation": 0.01,
}

DAY = date(2024, 6, 3)


def _instrument_bars(
    instrument: str, day: str, price: float = 10.0, volume: int = 1_000_000
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "datetime": f"{day} 13:30:00",
                "instrument": instrument,
                "close": price,
                "vwap": price,
                "volume": volume,
                "paused": 0,
                "up_limit": price * 1.2,
                "down_limit": price * 0.8,
            }
        ]
    )


def _no_trade_bars(day: str) -> pd.DataFrame:
    return _instrument_bars("SH600001", day, 10.0)


def _position(**overrides) -> dict:
    state = {
        "quantity": 1000,
        "available_quantity": 1000,
        "average_cost": 10.0,
        "last_trade_date": date(2024, 5, 31),
        "market_price": 10.0,
        "market_date": date(2024, 5, 31),
    }
    state.update(overrides)
    return state


def _run_day(**overrides):
    defaults = {
        "trade_date": DAY,
        "cash": 50_000.0,
        "prior_nav": 60_000.0,
        "high_water_mark": 60_000.0,
        "positions": {"SH600000": _position()},
        "target_weights": {},
        "minute_bars": _no_trade_bars("2024-06-03"),
        "closing_prices": {"SH600000": {"price": 10.0, "market_date": DAY}},
        "cost_model": CostModelConfig(),
        "execution_policy": dict(_POLICY),
    }
    defaults.update(overrides)
    return execute_simulation_day(**defaults)


def _event(**overrides) -> dict:
    payload = {
        "kind": "announcement",
        "instrument": "SH600000",
        "effective_date": "2024-06-03",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# 公告阶段信息事件（不改账，与除权事件唯一键关联）
# ---------------------------------------------------------------------------


def test_normalize_announcement_rows_plan_and_implementation() -> None:
    rows = [
        {  # 预案：无除权日——此前被直接丢弃，现在产生 plan 信息事件
            "ts_code": "600000.SH",
            "ann_date": "20240510",
            "div_proc": "预案",
            "ex_date": None,
            "cash_div_tax": 0.5,
            "stk_bo_rate": 0.2,
        },
        {  # 实施公告：带除权日，事件携带 linked_ex_event_key
            "ts_code": "600000.SH",
            "ann_date": "20240528",
            "div_proc": "实施",
            "ex_date": "20240603",
            "cash_div_tax": 0.5,
        },
        {  # 无公告日：无法定位公告阶段，跳过
            "ts_code": "600000.SH",
            "cash_div_tax": 0.5,
        },
    ]
    events = normalize_announcement_rows(rows)
    assert len(events) == 2
    plan, implementation = events
    assert plan.kind == "announcement"
    assert plan.stage == "plan"
    assert plan.event_key == "corporate_action:announcement:SH600000:2024-05-10"
    assert plan.details["related_event_key_prefix"] == "corporate_action:ex:SH600000:"
    assert implementation.stage == "implementation"
    assert (
        implementation.details["linked_ex_event_key"]
        == "corporate_action:ex:SH600000:2024-06-03"
    )


def test_announcement_event_does_not_touch_the_ledger() -> None:
    announcement = _event(stage="plan", details={"ann_date": "2024-06-03"})
    result = _run_day(corporate_events=[announcement])
    # 公告只产生信息事件：现金、持仓、应收、NAV 全部不变。
    assert result["cash"] == pytest.approx(50_000.0)
    assert result["positions"]["SH600000"]["quantity"] == 1000
    assert result["dividend_receivables"] == []
    assert result["nav"] == pytest.approx(60_000.0)
    emitted = [
        event for event in result["events"]
        if event["event_type"] == "corporate_action_announcement"
    ]
    assert len(emitted) == 1
    assert emitted[0]["severity"] == "info"
    assert emitted[0]["details"]["event_key"].startswith(
        "corporate_action:announcement:SH600000:"
    )
    applied = result["corporate_events_applied"]
    assert [item["event_key"] for item in applied] == [
        emitted[0]["details"]["event_key"]
    ]
    # 幂等重放：同一事件键经 applied_event_keys 再供给时不重复入账。
    replay = _run_day(
        corporate_events=[announcement],
        applied_event_keys=[applied[0]["event_key"]],
    )
    assert replay["corporate_events_applied"] == []
    assert not [
        event for event in replay["events"]
        if event["event_type"] == "corporate_action_announcement"
    ]


# ---------------------------------------------------------------------------
# 拆股 / 并股（经济数量与单位成本调整，不产生现金、防虚假亏损）
# ---------------------------------------------------------------------------


def test_split_scales_quantity_and_unit_cost_without_cash() -> None:
    split = _event(kind="split", split_ratio=2.0)
    result = _run_day(
        corporate_events=[split],
        # 除权价机械减半：市值不变，单位成本同步摊薄，无虚假亏损。
        closing_prices={"SH600000": {"price": 5.0, "market_date": DAY}},
    )
    position = result["positions"]["SH600000"]
    assert position["quantity"] == 2000
    assert position["average_cost"] == pytest.approx(5.0)
    total_cost = sum(
        float(lot["cost_basis_total"]) for lot in position.get("lots") or []
    )
    assert total_cost == pytest.approx(10_000.0)
    # 不产生现金、不动 NAV 口径：守恒黄金案例。
    assert result["cash"] == pytest.approx(50_000.0)
    assert result["cash_flows"] == []
    assert result["nav"] == pytest.approx(60_000.0)
    assert result["conservation"]["cash_difference"] == pytest.approx(0.0)
    assert result["corporate_events_applied"][0]["event_type"] == "split"


def test_reverse_split_scales_quantity_down_and_unit_cost_up() -> None:
    reverse = _event(kind="reverse_split", split_ratio=0.5)
    result = _run_day(
        corporate_events=[reverse],
        closing_prices={"SH600000": {"price": 20.0, "market_date": DAY}},
    )
    position = result["positions"]["SH600000"]
    assert position["quantity"] == 500
    assert position["average_cost"] == pytest.approx(20.0)
    total_cost = sum(
        float(lot["cost_basis_total"]) for lot in position.get("lots") or []
    )
    assert total_cost == pytest.approx(10_000.0)
    assert result["cash"] == pytest.approx(50_000.0)
    assert result["nav"] == pytest.approx(60_000.0)


def test_split_apply_once_under_replay() -> None:
    split = _event(kind="split", split_ratio=2.0)
    first = _run_day(corporate_events=[split])
    key = first["corporate_events_applied"][0]["event_key"]
    replay = _run_day(
        positions={"SH600000": first["positions"]["SH600000"]},
        corporate_events=[split],
        applied_event_keys=[key],
    )
    # 重放同一事件键：数量不再翻倍。
    assert replay["positions"]["SH600000"]["quantity"] == 2000
    assert replay["corporate_events_applied"] == []


def test_split_dust_position_fails_visible() -> None:
    dust = _event(kind="reverse_split", split_ratio=0.001)
    result = _run_day(
        positions={"SH600000": _position(quantity=100, available_quantity=100)},
        corporate_events=[dust],
    )
    # 并股后不足 1 股：不臆造份额、不假设现金结算，持仓原样 + critical 留痕。
    assert result["positions"]["SH600000"]["quantity"] == 100
    emitted = [
        event for event in result["events"]
        if event["event_type"] == "corporate_action_split_dust"
    ]
    assert len(emitted) == 1
    assert emitted[0]["severity"] == "critical"


def test_late_split_marks_review_without_rewrite() -> None:
    late = _event(kind="split", split_ratio=2.0, effective_date="2024-05-30")
    result = _run_day(corporate_events=[late])
    assert result["positions"]["SH600000"]["quantity"] == 1000
    emitted = [
        event for event in result["events"]
        if event["event_type"] == "corporate_action_event_late"
    ]
    assert len(emitted) == 1
    assert emitted[0]["severity"] == "critical"
    assert result["nav_row"]["performance_certified"] is False
    # 迟到事件不计入已应用键（持续可见，直到数据修正）。
    assert result["corporate_events_applied"] == []


# ---------------------------------------------------------------------------
# 代码变更（旧代码持仓映射新代码，无经济损益）
# ---------------------------------------------------------------------------


def test_code_change_maps_position_and_receivable() -> None:
    change = _event(
        kind="code_change",
        new_instrument="SH601666",
        details={"new_name": "新名称", "change_reason": "重组"},
    )
    receivable = {
        "instrument": "SH600000",
        "ex_date": DAY,
        "record_date": date(2024, 5, 31),
        "pay_date": date(2024, 6, 10),
        "quantity": 1000,
        "cash_per_share": 0.5,
        "amount": 500.0,
        "tax_rule_version": "cn-dividend-2015-at-sale",
        "valuation_uncertain": False,
    }
    result = _run_day(
        corporate_events=[change],
        dividend_receivables=[receivable],
        closing_prices={"SH601666": {"price": 10.0, "market_date": DAY}},
    )
    assert "SH600000" not in result["positions"]
    moved = result["positions"]["SH601666"]
    assert moved["quantity"] == 1000
    assert moved["average_cost"] == pytest.approx(10.0)
    # 应收跟随新代码，金额不变（不重分类、不增 NAV）。
    assert result["dividend_receivables"][0]["instrument"] == "SH601666"
    assert result["dividend_receivables"][0]["amount"] == pytest.approx(500.0)
    assert result["nav"] == pytest.approx(60_500.0)
    emitted = [
        event for event in result["events"]
        if event["event_type"] == "corporate_action_code_change"
    ]
    assert len(emitted) == 1
    assert emitted[0]["details"]["old_instrument"] == "SH600000"
    assert emitted[0]["details"]["new_instrument"] == "SH601666"
    assert emitted[0]["details"]["position_moved"] is True


def test_code_change_conflicting_positions_fail_closed() -> None:
    positions = {
        "SH600000": _position(),
        "SH601666": _position(quantity=100, available_quantity=100),
    }
    with pytest.raises(RuntimeError, match="merge two live positions"):
        apply_code_change(
            positions=positions,
            event=CorporateEvent.from_mapping(
                _event(kind="code_change", new_instrument="SH601666")
            ),
            trade_date=DAY,
        )


def test_namechange_rows_drive_code_change_and_name_events() -> None:
    events = normalize_namechange_rows(
        [
            {  # 仅名称变更 → 信息事件
                "ts_code": "600000.SH",
                "name": "浦发银行",
                "start_date": "20240101",
                "ann_date": "20231220",
                "change_reason": "改名",
            },
            {  # 带补充新代码 → 代码映射事件
                "ts_code": "600000.SH",
                "new_ts_code": "601666.SH",
                "name": "新名称",
                "start_date": "20240603",
                "change_reason": "重组",
            },
        ]
    )
    assert [event.kind for event in events] == ["name_change", "code_change"]
    assert events[1].instrument == "SH600000"
    assert events[1].new_instrument == "SH601666"
    assert "code_change:SH600000:SH601666:2024-06-03" in events[1].event_key


# ---------------------------------------------------------------------------
# unsupported 类型 fail-closed（记原因，不静默忽略，永不入账）
# ---------------------------------------------------------------------------


def test_unsupported_event_type_fails_closed() -> None:
    rights = _event(
        kind="unsupported",
        unsupported_type="rights_issue",
        title="配股发行公告",
        details={"skip_reason": "no_rights_issue_dataset"},
    )
    result = _run_day(corporate_events=[rights])
    # 不改账，但 fail-closed 记原因，且事件键幂等留痕。
    assert result["positions"]["SH600000"]["quantity"] == 1000
    assert result["cash"] == pytest.approx(50_000.0)
    emitted = [
        event for event in result["events"]
        if event["event_type"] == "corporate_action_unsupported"
    ]
    assert len(emitted) == 1
    assert emitted[0]["severity"] == "critical"
    assert emitted[0]["reason"] == "unsupported_event_type:rights_issue"
    assert result["corporate_events_applied"][0]["event_type"] == "unsupported"


def test_unknown_event_kind_rejected_at_ingest() -> None:
    with pytest.raises(ValueError, match="unknown corporate event kind"):
        CorporateEvent.from_mapping(_event(kind="spin_off"))
    with pytest.raises(ValueError, match="unregistered unsupported event type"):
        CorporateEvent.from_mapping(
            _event(kind="unsupported", unsupported_type="rights_issue_v2")
        )
    with pytest.raises(ValueError, match="split ratio out of range"):
        CorporateEvent.from_mapping(_event(kind="split", split_ratio=1.0))


# ---------------------------------------------------------------------------
# 需持有人选择的复杂事件（提醒 + 持仓标记，卖出提示不阻断）
# ---------------------------------------------------------------------------


def test_choice_required_marks_position_and_warns_on_sale() -> None:
    choice = _event(
        kind="choice_required",
        title="换股吸收合并要约",
        details={"matched_keywords": ["换股", "要约"]},
    )
    # 同日目标清仓：卖出允许成交，但事件可见。
    result = _run_day(
        corporate_events=[choice],
        minute_bars=_instrument_bars("SH600000", "2024-06-03"),
        target_weights={"SH600000": 0.0},
    )
    emitted = [
        event for event in result["events"]
        if event["event_type"] == "corporate_action_choice_required"
    ]
    assert len(emitted) == 1
    assert emitted[0]["severity"] == "warning"
    sale_warnings = [
        event for event in result["events"]
        if event["event_type"] == "corporate_action_choice_pending_sale"
    ]
    assert len(sale_warnings) == 1
    assert sale_warnings[0]["details"]["event_key"] == emitted[0]["details"]["event_key"]
    # 卖出未被硬阻断。
    assert result["positions"].get("SH600000") is None
    assert any(fill["side"] == "sell" for fill in result["fills"])


def test_choice_required_marker_without_trade() -> None:
    choice = _event(kind="choice_required", title="配股缴款提示")
    result = _run_day(corporate_events=[choice])
    marker = result["positions"]["SH600000"].get("choice_pending")
    assert marker is not None
    assert marker["manual_action_required"] is True
    assert result["cash"] == pytest.approx(50_000.0)


def test_detect_choice_required_and_liquidation_from_anns() -> None:
    rows = [
        {"ts_code": "600000.SH", "ann_date": "20240601", "title": "配股发行公告"},
        {"ts_code": "510300.SH", "ann_date": "20240601", "title": "基金份额清盘公告"},
        {"ts_code": "600000.SH", "ann_date": "20240601", "title": "年度股东大会决议"},
    ]
    choices = detect_choice_required_events(rows)
    assert len(choices) == 1
    assert choices[0].kind == "choice_required"
    assert choices[0].details["matched_keywords"] == ["配股"]
    liquidations = detect_liquidation_events(rows)
    assert len(liquidations) == 1
    assert liquidations[0].kind == "unsupported"
    assert liquidations[0].unsupported_type == "fund_liquidation"
    assert liquidations[0].details["skip_reason"] == "no_liquidation_proceeds_data"


# ---------------------------------------------------------------------------
# adj_factor / fund_adj 跳变 → 拆并股 / ETF 折算检测
# ---------------------------------------------------------------------------


def test_detect_split_events_from_adj_factor_jumps() -> None:
    adj_rows = [
        {"ts_code": "600000.SH", "trade_date": "20240531", "adj_factor": 1.0},
        {"ts_code": "600000.SH", "trade_date": "20240603", "adj_factor": 2.0},
        {"ts_code": "600000.SH", "trade_date": "20240604", "adj_factor": 2.1},
        {"ts_code": "510300.SH", "trade_date": "20240531", "adj_factor": 1.0},
        {"ts_code": "510300.SH", "trade_date": "20240603", "adj_factor": 0.25},
    ]
    events = detect_split_events(adj_rows)
    assert len(events) == 2
    stock_split = next(event for event in events if event.instrument == "SH600000")
    etf_conversion = next(event for event in events if event.instrument == "SH510300")
    assert stock_split.kind == "split"
    assert stock_split.split_ratio == pytest.approx(2.0)
    assert stock_split.effective_date == DAY
    assert stock_split.details["detection"] == "adj_factor_jump"
    assert etf_conversion.kind == "reverse_split"
    assert etf_conversion.split_ratio == pytest.approx(0.25)


def test_detect_split_events_skips_dividend_explained_jumps() -> None:
    adj_rows = [
        {"ts_code": "600000.SH", "trade_date": "20240531", "adj_factor": 1.0},
        {"ts_code": "600000.SH", "trade_date": "20240603", "adj_factor": 1.5},
    ]
    events = detect_split_events(
        adj_rows, known_ex_dates={"SH600000": [date(2024, 6, 3)]}
    )
    assert events == []


def test_apply_share_split_preserves_fractional_cost() -> None:
    position = _position(quantity=1001, average_cost=10.0)
    outcome = apply_share_split(
        position=position,
        event=CorporateEvent.from_mapping(_event(kind="split", split_ratio=2.0)),
        trade_date=DAY,
    )
    assert outcome["quantity_after"] == 2002
    total_cost = sum(
        float(lot["cost_basis_total"]) for lot in position.get("lots") or []
    )
    assert total_cost == pytest.approx(1001 * 10.0)
    assert position["average_cost"] == pytest.approx(5.0)
