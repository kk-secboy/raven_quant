from datetime import date

import pandas as pd
import pytest

from quant_platform.corporate_actions import (
    CorporateAction,
    apply_ex_dividend,
    consume_lots_fifo,
    normalize_dividend_rows,
    pending_dividend_tax_liability,
    position_lots,
    settle_dividend_tax,
)
from quant_platform.cost_model import CostModelConfig
from quant_platform.dividend_tax import DIVIDEND_TAX_RULE_BOOK
from quant_platform.simulation_engine import execute_simulation_day

pytestmark = pytest.mark.no_database

_POLICY = {
    "execution_algorithm": "twap",
    "slice_minutes": 20,
    "max_slices": 1,
    "max_participation": 0.01,
}


def _bars(day: str, price: float = 10.0, volume: int = 1_000_000) -> pd.DataFrame:
    return _instrument_bars("SH600000", day, price, volume)


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
    """Bars for an unrelated instrument: the held one has no executable bar."""

    return _instrument_bars("SH600001", day, 10.0)


def _action(**overrides) -> dict:
    payload = {
        "instrument": "SH600000",
        "ex_date": "2024-06-03",
        "record_date": "2024-05-31",
        "pay_date": "2024-06-10",
        "cash_div_pretax": 0.5,
        "cash_div_aftertax": 0.5,
        "bonus_share_ratio": 0.2,
        "conversion_ratio": 0.1,
        "list_date": "2024-06-04",
        "source_ts_code": "600000.SH",
    }
    payload.update(overrides)
    return payload


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


def test_normalize_dividend_rows_filters_and_parses() -> None:
    rows = [
        {  # 预案没有除权日 → 跳过
            "ts_code": "600000.SH",
            "div_proc": "预案",
            "ex_date": None,
            "cash_div_tax": 0.5,
        },
        {  # 全空行 → 跳过
            "ts_code": "600000.SH",
            "ex_date": "20240603",
        },
        {
            "ts_code": "600000.SH",
            "div_proc": "实施",
            "ex_date": "20240603",
            "record_date": "20240531",
            "pay_date": "20240610",
            "cash_div_tax": 0.5,
            "cash_div": 0.5,
            "stk_bo_rate": 0.2,
            "stk_co_rate": 0.1,
            "div_listdate": "20240604",
        },
    ]
    actions = normalize_dividend_rows(rows)
    assert len(actions) == 1
    action = actions[0]
    assert action.instrument == "SH600000"
    assert action.ex_date == date(2024, 6, 3)
    assert action.share_ratio == pytest.approx(0.3)


def test_normalize_dividend_rows_fails_closed_on_corrupt_ratio() -> None:
    with pytest.raises(ValueError, match="out of range"):
        normalize_dividend_rows(
            [{"ts_code": "600000.SH", "ex_date": "20240603", "stk_bo_rate": 99.0}]
        )


def test_ex_dividend_quantity_cost_and_entitlements() -> None:
    position = _position()
    outcome = apply_ex_dividend(
        position=position,
        action=CorporateAction.from_mapping(_action()),
        tax_rule=DIVIDEND_TAX_RULE_BOOK.as_of(date(2024, 5, 31)),
        trade_date=date(2024, 6, 3),
    )
    # 应收 = 1000 × 0.5（税前，2015 版卖出结算）
    assert outcome["receivable"]["amount"] == pytest.approx(500.0)
    # 送转 30%：数量 1300、总成本 10000 不变、单位成本摊薄
    assert outcome["new_shares"] == 300
    assert position["quantity"] == 1300
    lots = position["lots"]
    total_cost = sum(lot["cost_basis_total"] for lot in lots)
    assert total_cost == pytest.approx(10_000.0)
    assert position["average_cost"] == pytest.approx(10_000.0 / 1300)
    bonus = [lot for lot in lots if lot["origin"] == "bonus_share"]
    assert len(bonus) == 1 and bonus[0]["quantity"] == 300
    assert bonus[0]["sellable_from"] == date(2024, 6, 4)
    # 遗留批次取得日不可知；红利股批次继承父批次取得日（持有期连续计算）
    assert bonus[0]["acquired_at"] is None
    legacy = [lot for lot in lots if lot["origin"] == "legacy"][0]
    kinds = {item["kind"]: item for item in legacy["entitlements"]}
    assert kinds["cash"]["income_per_share"] == pytest.approx(0.5)
    # 送股 0.2 × 面值 1 元计入股息税基；转增 0.1 不计
    assert kinds["bonus_par"]["income_per_share"] == pytest.approx(0.2)


def test_at_payment_version_books_after_tax_amount_without_entitlements() -> None:
    position = _position()
    action = CorporateAction.from_mapping(
        _action(
            ex_date="2012-06-04",
            record_date="2012-06-01",
            cash_div_aftertax=0.45,
            bonus_share_ratio=0.0,
            conversion_ratio=0.0,
        )
    )
    outcome = apply_ex_dividend(
        position=position,
        action=action,
        tax_rule=DIVIDEND_TAX_RULE_BOOK.as_of(date(2012, 6, 1)),
        trade_date=date(2012, 6, 4),
    )
    assert outcome["receivable"]["amount"] == pytest.approx(450.0)
    legacy = [lot for lot in position["lots"] if lot["origin"] == "legacy"][0]
    assert legacy["entitlements"] == []


def test_dividend_tax_brackets_and_single_charge() -> None:
    book = DIVIDEND_TAX_RULE_BOOK
    lots = [
        {
            "lot_key": "b1",
            "acquired_at": date(2024, 5, 20),
            "sellable_from": date.min,
            "quantity": 100,
            "cost_basis_total": 1000.0,
            "origin": "buy",
            "entitlements": [
                {
                    "record_date": date(2024, 5, 31),
                    "kind": "cash",
                    "income_per_share": 0.5,
                    "untaxed_quantity": 100,
                }
            ],
        }
    ]
    # 持有 <1 个月卖出 60 股：税 = 60×0.5×20% = 6
    consumed = consume_lots_fifo(lots, 60, trade_date=date(2024, 6, 5))
    tax, details, released = settle_dividend_tax(
        instrument="SH600000", consumed=consumed, sale_date=date(2024, 6, 5), tax_book=book
    )
    assert tax == pytest.approx(6.0)
    # 旧账本权利无 liability_per_share：无负债可释放，卖出全额确认税额
    assert released == pytest.approx(0.0)
    assert details[0]["tax_rule_version"] == "cn-dividend-tax-2015-09-08"
    assert details[0]["liability_released"] == pytest.approx(0.0)
    # 剩余 40 股再卖：只税剩余 40，不重复（税 = 40×0.5×20% = 4）
    consumed = consume_lots_fifo(lots, 40, trade_date=date(2024, 6, 5))
    tax, _, released = settle_dividend_tax(
        instrument="SH600000", consumed=consumed, sale_date=date(2024, 6, 5), tax_book=book
    )
    assert tax == pytest.approx(4.0)
    assert released == pytest.approx(0.0)
    assert lots == []


def test_dividend_tax_rates_by_holding_period() -> None:
    book = DIVIDEND_TAX_RULE_BOOK

    def _tax(acquired: date, sale: date) -> float:
        lots = [
            {
                "lot_key": "b1",
                "acquired_at": acquired,
                "sellable_from": date.min,
                "quantity": 100,
                "cost_basis_total": 1000.0,
                "origin": "buy",
                "entitlements": [
                    {
                        "record_date": date(2024, 5, 31),
                        "kind": "cash",
                        "income_per_share": 1.0,
                        "untaxed_quantity": 100,
                    }
                ],
            }
        ]
        consumed = consume_lots_fifo(lots, 100, trade_date=sale)
        tax, _, _released = settle_dividend_tax(
            instrument="SH600000", consumed=consumed, sale_date=sale, tax_book=book
        )
        return tax

    assert _tax(date(2024, 5, 10), date(2024, 6, 10)) == pytest.approx(20.0)  # 含 1 个月
    assert _tax(date(2024, 5, 10), date(2024, 6, 11)) == pytest.approx(10.0)
    assert _tax(date(2023, 5, 10), date(2024, 6, 11)) == pytest.approx(0.0)  # 超 1 年免征


def test_dividend_tax_skips_etf() -> None:
    lots = [
        {
            "lot_key": "b1",
            "acquired_at": date(2024, 5, 1),
            "sellable_from": date.min,
            "quantity": 100,
            "cost_basis_total": 1000.0,
            "origin": "buy",
            "entitlements": [
                {
                    "record_date": date(2024, 5, 31),
                    "kind": "cash",
                    "income_per_share": 0.5,
                    "untaxed_quantity": 100,
                }
            ],
        }
    ]
    consumed = consume_lots_fifo(lots, 100, trade_date=date(2024, 6, 3))
    tax, details, released = settle_dividend_tax(
        instrument="SH510300",
        consumed=consumed,
        sale_date=date(2024, 6, 3),
        tax_book=DIVIDEND_TAX_RULE_BOOK,
    )
    assert tax == 0.0 and details == [] and released == 0.0


# ---------------------------------------------------------------------------
# Engine-level golden cases（除权→到账→卖出税 全链路 NAV 守恒）
# ---------------------------------------------------------------------------


def _run_day(**overrides):
    defaults = {
        "trade_date": date(2024, 6, 3),
        "cash": 50_000.0,
        "prior_nav": 60_000.0,
        "high_water_mark": 60_000.0,
        "positions": {"SH600000": _position()},
        "target_weights": {},
        "minute_bars": _no_trade_bars("2024-06-03"),
        "closing_prices": {"SH600000": {"price": 7.31, "market_date": date(2024, 6, 3)}},
        "cost_model": CostModelConfig(),
        "execution_policy": dict(_POLICY),
    }
    defaults.update(overrides)
    return execute_simulation_day(**defaults)


def test_ex_date_nav_continuity_with_receivable() -> None:
    result = _run_day(corporate_actions=[_action()])
    # 除权日：市值降、应收升（税前 500 = 1000×0.5），同时按除权时点持有期
    # 档位（2024-05-20 取得 → ≤1 个月 → 20%）计提保守应付税负债
    # 1000×(0.5+0.2)×20% = 140；NAV = 50000 + 1300×7.31 + 500 − 140 = 59863，
    # 不含税前全额虚高（设计 §5.6 冲突 2.6 修正：旧行为 60003 即虚高 140）。
    assert result["nav_row"]["corporate_receivables"] == pytest.approx(500.0)
    assert result["nav_row"]["corporate_tax_liabilities"] == pytest.approx(140.0)
    assert result["nav"] == pytest.approx(59_863.0, abs=2.0)
    assert result["corporate_actions_applied"][0]["tax_liability"] == pytest.approx(140.0)
    assert result["positions"]["SH600000"]["quantity"] == 1300
    assert result["corporate_actions_applied"][0]["kind"] == "ex"
    assert result["nav_row"]["performance_certified"] is True


def test_pay_date_reclassifies_receivable_without_nav_change() -> None:
    ex_result = _run_day(corporate_actions=[_action()])
    pay_day = date(2024, 6, 10)
    result = _run_day(
        trade_date=pay_day,
        cash=ex_result["cash"],
        prior_nav=ex_result["nav"],
        high_water_mark=ex_result["high_water_mark"],
        positions=ex_result["positions"],
        minute_bars=_no_trade_bars("2024-06-10"),
        closing_prices={"SH600000": {"price": 7.31, "market_date": pay_day}},
        dividend_receivables=ex_result["dividend_receivables"],
    )
    assert result["cash"] == pytest.approx(50_500.0)
    assert result["nav_row"]["corporate_receivables"] == pytest.approx(0.0)
    assert result["nav"] == pytest.approx(ex_result["nav"])
    assert result["cash_flows"][0]["flow_type"] == "dividend_payment"
    assert result["corporate_actions_applied"][0]["kind"] == "pay"


def test_sell_after_dividend_charges_tax_once() -> None:
    day = date(2024, 6, 11)
    result = _run_day(
        trade_date=day,
        target_weights={"SH600000": 0.0},
        minute_bars=_bars("2024-06-11", 7.3),
        closing_prices={"SH600000": {"price": 7.3, "market_date": day}},
        corporate_actions=[_action(ex_date="2024-06-03")],
        positions={
            "SH600000": _position(
                quantity=1300,
                average_cost=10_000.0 / 1300,
                _applied_ca_ex_dates=("2024-06-03",),
                lots=[
                    {
                        "lot_key": "legacy",
                        "acquired_at": date(2024, 5, 20),
                        "sellable_from": date.min,
                        "quantity": 1000,
                        "cost_basis_total": 10_000.0 * 1000 / 1300,
                        "origin": "legacy",
                        "entitlements": [
                            {
                                "record_date": date(2024, 5, 31),
                                "kind": "cash",
                                "income_per_share": 0.5,
                                "untaxed_quantity": 1000,
                            },
                            {
                                "record_date": date(2024, 5, 31),
                                "kind": "bonus_par",
                                "income_per_share": 0.2,
                                "untaxed_quantity": 1000,
                            },
                        ],
                    },
                    {
                        "lot_key": "bonus:2024-06-03:legacy",
                        "acquired_at": date(2024, 5, 20),
                        "sellable_from": date(2024, 6, 4),
                        "quantity": 300,
                        "cost_basis_total": 10_000.0 * 300 / 1300,
                        "origin": "bonus_share",
                        "entitlements": [],
                    },
                ],
            )
        },
    )
    tax_flows = [flow for flow in result["cash_flows"] if flow["flow_type"] == "dividend_tax"]
    # 持有 <1 月：1000×0.5×20% + 1000×0.2×20% = 140；红利股批次无股息权利
    assert sum(-flow["amount"] for flow in tax_flows) == pytest.approx(140.0)
    assert result["conservation"]["cash_difference"] == pytest.approx(0.0, abs=1e-6)
    assert "SH600000" not in result["positions"]


def test_bonus_shares_locked_until_list_date() -> None:
    locked = _run_day(
        corporate_actions=[_action()],
        target_weights={"SH600000": 0.0},
        minute_bars=_bars("2024-06-03", 7.31),
    )
    sell_order = [order for order in locked["orders"] if order["side"] == "sell"][0]
    # 除权当日新增 300 股未上市：只能卖出原有 1000 股
    assert sell_order["filled_quantity"] == 1000
    assert locked["positions"]["SH600000"]["quantity"] == 300
    # 新增股份上市日后可卖
    list_day = date(2024, 6, 4)
    unlocked = _run_day(
        trade_date=list_day,
        positions=locked["positions"],
        cash=locked["cash"],
        prior_nav=locked["nav"],
        high_water_mark=locked["high_water_mark"],
        target_weights={"SH600000": 0.0},
        minute_bars=_bars("2024-06-04", 7.31),
        closing_prices={"SH600000": {"price": 7.31, "market_date": list_day}},
        dividend_receivables=locked["dividend_receivables"],
    )
    sell_order = [order for order in unlocked["orders"] if order["side"] == "sell"][0]
    assert sell_order["filled_quantity"] == 300


def test_late_corporate_action_flags_review_without_rewriting() -> None:
    result = _run_day(corporate_actions=[_action(ex_date="2024-05-20")])
    late_events = [
        event for event in result["events"] if event["event_type"] == "corporate_action_late"
    ]
    assert len(late_events) == 1
    assert late_events[0]["severity"] == "critical"
    # 不回写历史：无应收、无新增股份、NAV 无公司行动影响
    assert result["nav_row"]["corporate_receivables"] == 0.0
    assert result["positions"]["SH600000"]["quantity"] == 1000
    assert result["nav_row"]["performance_certified"] is False


def test_applied_ex_date_is_idempotent() -> None:
    result = _run_day(
        corporate_actions=[_action()],
        positions={
            "SH600000": _position(_applied_ca_ex_dates=("2024-06-03",)),
        },
    )
    # 已入账的除权日重复供给：不产生第二笔应收或送转
    assert result["nav_row"]["corporate_receivables"] == 0.0
    assert result["positions"]["SH600000"]["quantity"] == 1000
    assert result["corporate_actions_applied"] == []


def test_irrelevant_old_action_for_unheld_instrument_is_ignored() -> None:
    result = _run_day(
        corporate_actions=[_action(instrument="SZ000001", ex_date="2024-05-20")],
    )
    assert [e for e in result["events"] if e["event_type"] == "corporate_action_late"] == []
    assert result["nav_row"]["performance_certified"] is True


def test_position_lots_synthesizes_legacy_lot_once() -> None:
    position = _position()
    lots = position_lots(position, trade_date=date(2024, 6, 3))
    assert lots is position_lots(position, trade_date=date(2024, 6, 3))
    assert lots[0]["origin"] == "legacy"
    assert lots[0]["acquired_at"] is None
    assert lots[0]["sellable_from"] == date.min


def test_duplicate_action_rows_apply_once() -> None:
    result = _run_day(corporate_actions=[_action(), _action()])
    assert result["nav_row"]["corporate_receivables"] == pytest.approx(500.0)
    assert result["positions"]["SH600000"]["quantity"] == 1300
    ex_applied = [
        item for item in result["corporate_actions_applied"] if item["kind"] == "ex"
    ]
    assert len(ex_applied) == 1


# ---------------------------------------------------------------------------
# 股息税保守负债（设计 §5.6 冲突 2.6：计提/比例释放/差额确认/重放幂等）
# ---------------------------------------------------------------------------


def test_ex_date_accrues_conservative_liability_per_ex_date_bracket() -> None:
    position = _position(
        lots=[
            {
                "lot_key": "b1",
                "acquired_at": date(2024, 5, 1),  # 除权日持有 >1 个月 → 10% 档
                "sellable_from": date.min,
                "quantity": 1000,
                "cost_basis_total": 10_000.0,
                "origin": "buy",
                "entitlements": [],
            }
        ]
    )
    outcome = apply_ex_dividend(
        position=position,
        action=CorporateAction.from_mapping(_action()),
        tax_rule=DIVIDEND_TAX_RULE_BOOK.as_of(date(2024, 5, 31)),
        trade_date=date(2024, 6, 3),
    )
    # 每股负债 = (现金 0.5 + 送股面值 0.2) × 10% = 0.07；1000 股合计 70
    assert outcome["tax_liability"] == pytest.approx(70.0)
    entitlements = {item["kind"]: item for item in position["lots"][0]["entitlements"]}
    assert entitlements["cash"]["liability_per_share"] == pytest.approx(0.05)
    assert entitlements["bonus_par"]["liability_per_share"] == pytest.approx(0.02)


def test_fifo_partial_sale_releases_liability_proportionally() -> None:
    lots = [
        {
            "lot_key": "b1",
            "acquired_at": date(2024, 5, 20),
            "sellable_from": date.min,
            "quantity": 100,
            "cost_basis_total": 1000.0,
            "origin": "buy",
            "entitlements": [
                {
                    "record_date": date(2024, 5, 31),
                    "kind": "cash",
                    "income_per_share": 0.5,
                    "untaxed_quantity": 100,
                    "liability_per_share": 0.1,
                }
            ],
        },
        {
            "lot_key": "b2",
            "acquired_at": date(2024, 5, 20),
            "sellable_from": date.min,
            "quantity": 100,
            "cost_basis_total": 1000.0,
            "origin": "buy",
            "entitlements": [
                {
                    "record_date": date(2024, 5, 31),
                    "kind": "cash",
                    "income_per_share": 0.5,
                    "untaxed_quantity": 100,
                    "liability_per_share": 0.2,
                }
            ],
        },
    ]
    # FIFO 卖 150：批次一 100 全消耗（释放 100×0.1），批次二 50（释放 50×0.2）
    consumed = consume_lots_fifo(lots, 150, trade_date=date(2024, 6, 5))
    tax, _, released = settle_dividend_tax(
        instrument="SH600000", consumed=consumed, sale_date=date(2024, 6, 5),
        tax_book=DIVIDEND_TAX_RULE_BOOK,
    )
    assert tax == pytest.approx(150 * 0.5 * 0.20)
    assert released == pytest.approx(100 * 0.1 + 50 * 0.2)
    # 剩余负债只来自批次二未消耗的 50 股：50×0.2 = 10
    state = {"SH600000": {"quantity": 50, "lots": lots}}
    assert pending_dividend_tax_liability(state) == pytest.approx(10.0)


def test_sale_recognizes_only_difference_when_bracket_drops() -> None:
    # 除权日按 20% 档计提（每股 0.10 + 0.04），卖出日持有跨过 1 个月 → 10%
    lots = [
        {
            "lot_key": "b1",
            "acquired_at": date(2024, 5, 10),
            "sellable_from": date.min,
            "quantity": 100,
            "cost_basis_total": 1000.0,
            "origin": "buy",
            "entitlements": [
                {
                    "record_date": date(2024, 5, 31),
                    "kind": "cash",
                    "income_per_share": 0.5,
                    "untaxed_quantity": 100,
                    "liability_per_share": 0.10,
                },
                {
                    "record_date": date(2024, 5, 31),
                    "kind": "bonus_par",
                    "income_per_share": 0.2,
                    "untaxed_quantity": 100,
                    "liability_per_share": 0.04,
                },
            ],
        }
    ]
    consumed = consume_lots_fifo(lots, 100, trade_date=date(2024, 6, 11))
    tax, details, released = settle_dividend_tax(
        instrument="SH600000", consumed=consumed, sale_date=date(2024, 6, 11),
        tax_book=DIVIDEND_TAX_RULE_BOOK,
    )
    # 实际税 = 100×(0.5+0.2)×10% = 7；释放 100×0.14 = 14；多提冲回 7
    assert tax == pytest.approx(7.0)
    assert released == pytest.approx(14.0)
    assert released - tax == pytest.approx(7.0)
    assert sorted(item["liability_released"] for item in details) == [
        pytest.approx(4.0),
        pytest.approx(10.0),
    ]
    assert pending_dividend_tax_liability({"SH600000": {"lots": lots}}) == pytest.approx(0.0)


def test_engine_sale_day_nav_moves_only_by_tax_difference() -> None:
    position = _position(
        lots=[
            {
                "lot_key": "b1",
                "acquired_at": date(2024, 5, 20),
                "sellable_from": date.min,
                "quantity": 1000,
                "cost_basis_total": 10_000.0,
                "origin": "buy",
                "entitlements": [],
            }
        ]
    )
    ex_day = _run_day(
        corporate_actions=[_action(bonus_share_ratio=0.0, conversion_ratio=0.0)],
        positions={"SH600000": position},
    )
    # 除权日 ≤1 个月 → 20% 档计提 1000×0.5×20% = 100
    assert ex_day["nav_row"]["corporate_tax_liabilities"] == pytest.approx(100.0)
    assert ex_day["nav"] == pytest.approx(50_000 + 7_310 + 500 - 100, abs=1e-6)
    # 持有跨过 1 个月（2024-05-20 → 2024-06-21）后清仓：实际税 10% = 50
    sale_day = date(2024, 6, 21)
    result = _run_day(
        trade_date=sale_day,
        cash=ex_day["cash"],
        prior_nav=ex_day["nav"],
        high_water_mark=ex_day["high_water_mark"],
        positions=ex_day["positions"],
        target_weights={"SH600000": 0.0},
        minute_bars=_bars("2024-06-21", 7.31),
        closing_prices={"SH600000": {"price": 7.31, "market_date": sale_day}},
        dividend_receivables=ex_day["dividend_receivables"],
    )
    tax_flows = [flow for flow in result["cash_flows"] if flow["flow_type"] == "dividend_tax"]
    assert sum(-flow["amount"] for flow in tax_flows) == pytest.approx(50.0)
    sell_fill = [fill for fill in result["fills"] if fill["side"] == "sell"][0]
    assert sell_fill["cost_breakdown"]["dividend_tax_liability_released"] == pytest.approx(
        100.0
    )
    assert result["nav_row"]["corporate_tax_liabilities"] == pytest.approx(0.0)
    # 差额确认（多提冲回）：价格不变时 NAV 变动 = (释放100 − 实税50) − 卖出成本
    assert result["nav"] - ex_day["nav"] == pytest.approx(50.0, abs=20.0)
    # NAV 手算：50000 + (7310 − 成本≈14.69) − 50 税 + 500 应收 − 0 负债
    assert result["nav"] == pytest.approx(57_745.3, abs=0.5)
    assert result["conservation"]["cash_difference"] == pytest.approx(0.0, abs=1e-6)


def test_ex_date_replay_does_not_double_accrue_liability() -> None:
    first = _run_day(corporate_actions=[_action()])
    assert first["nav_row"]["corporate_tax_liabilities"] == pytest.approx(140.0)
    # 模拟 store 重建后的同日重放：除权日已入账标记 + 携带负债的权利随状态回灌
    replay_positions = first["positions"]
    replay_positions["SH600000"]["_applied_ca_ex_dates"] = ("2024-06-03",)
    second = _run_day(
        corporate_actions=[_action()],
        positions=replay_positions,
        cash=first["cash"],
        prior_nav=first["nav"],
        high_water_mark=first["high_water_mark"],
        dividend_receivables=first["dividend_receivables"],
    )
    assert second["corporate_actions_applied"] == []
    assert second["nav_row"]["corporate_receivables"] == pytest.approx(500.0)
    # 负债由状态重算：既不重复计提（≠280）也不丢失（≠0）
    assert second["nav_row"]["corporate_tax_liabilities"] == pytest.approx(140.0)
    assert second["nav"] == pytest.approx(first["nav"])
