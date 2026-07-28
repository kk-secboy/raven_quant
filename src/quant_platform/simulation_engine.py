from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from datetime import date, datetime, time, timedelta
from math import floor, isfinite
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .corporate_actions import (
    ECONOMIC_EVENT_KINDS,
    EVENT_KIND_CHOICE_REQUIRED,
    EVENT_KIND_CODE_CHANGE,
    EVENT_KIND_UNSUPPORTED,
    LOT_ORIGIN_BUY,
    CorporateAction,
    CorporateEvent,
    apply_code_change,
    apply_ex_dividend,
    apply_share_split,
    consume_lots_fifo,
    pending_dividend_tax_liability,
    position_lots,
    settle_dividend_tax,
)
from .cost_model import CostModelConfig, CostScheduleBook, infer_cn_asset_type
from .dividend_tax import DIVIDEND_TAX_RULE_BOOK, DividendTaxRuleBook
from .etf_subtypes import etf_trading_gate
from .execution_algorithms import build_execution_slices, normalize_execution_policy
from .market_rules import (
    OrderUnitRules,
    is_valid_order_quantity,
    lot_floor,
    order_unit_rules,
)
from .unitized_performance import chain_unitized_day

SIMULATION_ENGINE_VERSION = "ashare-minute-simulation-v2"
_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _resolve_cost_schedule(
    cost_model: CostModelConfig | None,
    cost_schedule: CostScheduleBook | None,
) -> CostScheduleBook:
    if (cost_model is None) == (cost_schedule is None):
        raise ValueError("exactly one of cost_model or cost_schedule is required")
    return cost_schedule or CostScheduleBook.from_versions([cost_model])


def execute_simulation_day(
    *,
    trade_date: date,
    cash: float,
    prior_nav: float,
    high_water_mark: float,
    positions: dict[str, dict[str, Any]],
    target_weights: dict[str, float],
    minute_bars: pd.DataFrame,
    closing_prices: dict[str, dict[str, Any]],
    cost_model: CostModelConfig | None = None,
    cost_schedule: CostScheduleBook | None = None,
    execution_policy: dict[str, Any],
    signal_at: datetime | None = None,
    corporate_actions: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    dividend_receivables: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    corporate_events: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    applied_event_keys: Iterable[str] = (),
    dividend_tax_book: DividendTaxRuleBook | None = None,
    order_specs_override: list[dict[str, Any]] | None = None,
    tradable_cash: float | None = None,
    external_flow_open: float = 0.0,
    external_flow_close: float = 0.0,
    prior_investment_wealth: float | None = 1.0,
    twr_high_water_mark: float | None = 1.0,
) -> dict[str, Any]:
    """Execute one next-eligible-bar A-share/ETF rebalance with an auditable ledger.

    撮合路径挂 ETF 子类型白名单门禁（``etf_subtypes``，设计 §1.3/§5.1）：
    未验收子类型的订单 fail-closed 拒单并记录原因；研究/回测路径不经本
    函数，不受此门禁限制。

    ``order_specs_override`` executes persistent order-book specs (from the
    planned/open order lifecycle) instead of deriving orders from target
    weights. Each spec carries ``instrument``, ``side``, ``requested_quantity``
    (the remaining quantity to work today), optional ``order_ref`` (the
    persistent order id, propagated to fills), optional ``limit_price`` (price
    protection: buys never fill above it, sells never below) and optional
    ``not_before``/``not_after`` (execution window; slices outside it are
    skipped, the remainder stays for a later batch).

    ``external_flow_open``/``external_flow_close`` are the day's confirmed
    external cash flows (design 4.4/12.1: deposits positive, withdrawals
    negative; they never count as profit or loss). The open flow is added to
    cash before trading (usable for the day's decisions); the close flow is
    added after execution and only enters investable cash the next day. Both
    are included in the closing NAV and in the cash-conservation check, and
    the unitized TWR chain strips them out of the daily return.

    ``corporate_events`` carries the non-dividend corporate events (design
    5.6 type extension): announcements/name changes are informational only;
    splits/reverse splits/ETF share conversions scale quantity and unit cost
    on their effective date without touching cash; code changes remap the
    position identity; ``choice_required`` events only flag the position for
    manual handling (sells stay allowed but raise a visible warning);
    ``unsupported`` event types fail closed — recorded once with their reason,
    never booked. ``applied_event_keys`` is the persistent idempotency state:
    keys already applied (loaded from the event ledger) are skipped, so
    replaying the same input is a no-op.
    """

    if not isfinite(cash) or cash < 0 or prior_nav <= 0 or high_water_mark <= 0:
        raise ValueError("simulation account balances are invalid")
    available_cash = cash if tradable_cash is None else float(tradable_cash)
    if (
        not isfinite(available_cash)
        or available_cash < 0
        or available_cash > cash + 1e-6
    ):
        raise ValueError("simulation tradable cash view is invalid")
    if not (isfinite(external_flow_open) and isfinite(external_flow_close)):
        raise ValueError("external cash flows must be finite")
    cost_model = _resolve_cost_schedule(cost_model, cost_schedule).as_of(trade_date)
    policy = normalize_execution_policy(execution_policy)
    targets = {str(key): float(value) for key, value in target_weights.items()}
    if any(not isfinite(value) or value < 0 for value in targets.values()):
        raise ValueError("simulation target weights must be finite and non-negative")
    if sum(targets.values()) > 1.0 + 1e-8:
        raise ValueError("simulation target weights exceed one")
    bars = _normalize_bars(minute_bars, trade_date)
    state = deepcopy(positions)
    for instrument, position in state.items():
        quantity = int(position.get("quantity", 0))
        available = int(position.get("available_quantity", 0))
        last_trade = position.get("last_trade_date")
        if quantity < 0 or available < 0 or available > quantity:
            raise ValueError(f"invalid position state for {instrument}")
        if last_trade is None or _as_date(last_trade) < trade_date:
            position["available_quantity"] = quantity

    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    cash_flows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    starting_cash = cash
    if external_flow_open:
        # F_t_open：开盘前确认的外部现金流，当日可投资；不计损益但参与现金守恒。
        if cash + external_flow_open < -1e-6:
            raise RuntimeError("external withdrawal would create negative cash")
        cash += external_flow_open
        available_cash += external_flow_open
        if available_cash < -1e-6:
            raise RuntimeError("external withdrawal exceeds tradable cash")
        cash_flows.append(
            {
                "trade_date": trade_date,
                "flow_type": (
                    "external_deposit_open"
                    if external_flow_open > 0
                    else "external_withdrawal_open"
                ),
                "amount": external_flow_open,
                "balance_after": cash,
            }
        )
    tax_book = dividend_tax_book or DIVIDEND_TAX_RULE_BOOK
    open_receivables: list[dict[str, Any]] = [dict(item) for item in dividend_receivables]
    corporate_actions_applied: list[dict[str, Any]] = []
    late_corporate_action = False
    ex_applied_this_run: set[tuple[str, date]] = set()
    parsed_actions = [
        raw_action
        if isinstance(raw_action, CorporateAction)
        else CorporateAction.from_mapping(raw_action)
        for raw_action in corporate_actions
    ]
    # 应收创建时 pay_date 可能未知：用同批公司行动行里的到账日补齐（不改金额）。
    pay_date_lookup = {
        (item.instrument, item.ex_date): item.pay_date
        for item in parsed_actions
        if item.pay_date is not None
    }
    # 盘前公司行动：先到账（应收→现金，NAV 不变），后除权（应收/送转，NAV 连续）。
    for receivable in list(open_receivables):
        pay_date = receivable.get("pay_date")
        if pay_date is None:
            pay_date = pay_date_lookup.get(
                (str(receivable["instrument"]), _as_date(receivable["ex_date"]))
            )
            if pay_date is not None:
                receivable["pay_date"] = pay_date
        if pay_date is None or _as_date(pay_date) > trade_date:
            continue
        amount = float(receivable["amount"])
        cash += amount
        available_cash += amount
        cash_flows.append(
            {
                "trade_date": trade_date,
                "flow_type": "dividend_payment",
                "amount": amount,
                "balance_after": cash,
            }
        )
        open_receivables.remove(receivable)
        corporate_actions_applied.append(
            {
                "kind": "pay",
                "instrument": str(receivable["instrument"]),
                "ex_date": _as_date(receivable["ex_date"]).isoformat(),
            }
        )
        events.append(
            {
                "severity": "info",
                "event_type": "corporate_action_pay",
                "instrument": str(receivable["instrument"]),
                "reason": "dividend_payment_received",
                "details": {
                    "event_key": (
                        f"corporate_action:pay:{receivable['instrument']}:"
                        f"{_as_date(receivable['ex_date'])}"
                    ),
                    "amount": amount,
                    "scheduled_pay_date": _as_date(pay_date).isoformat(),
                },
            }
        )
    for action in parsed_actions:
        if action.ex_date < trade_date:
            late_position = state.get(action.instrument) or {}
            if action.ex_date.isoformat() in (
                late_position.get("_applied_ca_ex_dates") or ()
            ):
                continue  # 已入账的除权日被重复供给，忽略。
            relevant = (
                int(late_position.get("quantity", 0)) > 0
                or bool(late_position.get("_applied_ca_ex_dates"))
                or any(
                    str(item.get("instrument")) == action.instrument
                    for item in open_receivables
                )
            )
            if not relevant:
                continue  # 除权日前后均未持有，与账户无关。
            # 迟到的公司行动只标记复核，不回写历史（设计 §3.2 禁止静默改写）。
            # 已知边界：除权后、数据到达前已清仓的证券无法由此处发现，
            # 由数据库唯一约束与人工复核兜底。
            late_corporate_action = True
            events.append(
                {
                    "severity": "critical",
                    "event_type": "corporate_action_late",
                    "instrument": action.instrument,
                    "reason": "ex_date_already_passed",
                    "details": {
                        "event_key": (
                            f"corporate_action:late:{action.instrument}:{action.ex_date}"
                        ),
                        "ex_date": action.ex_date.isoformat(),
                    },
                }
            )
            continue
        if action.ex_date > trade_date:
            continue
        position = state.get(action.instrument)
        if position is None or int(position.get("quantity", 0)) <= 0:
            events.append(
                {
                    "severity": "info",
                    "event_type": "corporate_action_no_position",
                    "instrument": action.instrument,
                    "reason": "ex_dividend_without_position",
                    "details": {
                        "event_key": (
                            f"corporate_action:ex:{action.instrument}:{action.ex_date}"
                        ),
                        "ex_date": action.ex_date.isoformat(),
                    },
                }
            )
            continue
        applied_ex_dates = position.get("_applied_ca_ex_dates") or ()
        if action.ex_date.isoformat() in applied_ex_dates:
            continue  # 状态级幂等：该除权日已入账（数据库唯一约束兜底）。
        if (action.instrument, action.ex_date) in ex_applied_this_run:
            continue  # 上游行重复：同一除权日在本次运行中只能入账一次。
        ex_applied_this_run.add((action.instrument, action.ex_date))
        tax_rule = tax_book.as_of(action.record_date or action.ex_date)
        outcome = apply_ex_dividend(
            position=position,
            action=action,
            tax_rule=tax_rule,
            trade_date=trade_date,
        )
        events.extend(outcome["events"])
        applied = {
            "kind": "ex",
            "instrument": action.instrument,
            "ex_date": action.ex_date.isoformat(),
            "record_date": action.record_date.isoformat() if action.record_date else None,
            "pay_date": action.pay_date.isoformat() if action.pay_date else None,
            "list_date": action.list_date.isoformat() if action.list_date else None,
            "eligible_quantity": int(position.get("quantity", 0)) - outcome["new_shares"],
            "new_shares": outcome["new_shares"],
            "bonus_share_ratio": action.bonus_share_ratio,
            "conversion_ratio": action.conversion_ratio,
            "cash_per_share": 0.0,
            "receivable_amount": 0.0,
            "tax_liability": 0.0,
            "tax_rule_version": tax_rule.version,
            "payload_sha256": action.payload_sha256,
            "valuation_uncertain": False,
        }
        receivable = outcome["receivable"]
        if receivable is not None:
            open_receivables.append(receivable)
            applied["cash_per_share"] = float(receivable["cash_per_share"])
            applied["receivable_amount"] = float(receivable["amount"])
            applied["valuation_uncertain"] = bool(receivable["valuation_uncertain"])
        applied["tax_liability"] = float(outcome["tax_liability"])
        corporate_actions_applied.append(applied)

    # 盘前非分红类公司行动（设计 §5.6 类型扩展）：公告/名称变更只留信息事件，
    # 拆并股/代码变更在经济生效日调账，unsupported 类型 fail-closed 记原因。
    # 幂等纪律与除权一致：事件键只应用一次（applied_event_keys + 本次运行集合），
    # 迟到事件只标记复核不回写历史。
    applied_keys = {str(key) for key in applied_event_keys}
    run_event_keys: set[str] = set()
    corporate_events_applied: list[dict[str, Any]] = []
    choice_sale_warned: set[str] = set()
    parsed_events = [
        raw_event
        if isinstance(raw_event, CorporateEvent)
        else CorporateEvent.from_mapping(raw_event)
        for raw_event in corporate_events
    ]
    for event in parsed_events:
        if event.event_key in applied_keys or event.event_key in run_event_keys:
            continue  # 状态级幂等：该事件键已入账（数据库唯一约束兜底）。
        if event.effective_date > trade_date:
            continue  # 未来生效：留给后续交易日。
        position = state.get(event.instrument)
        has_position = position is not None and int(position.get("quantity", 0)) > 0
        if event.kind in ECONOMIC_EVENT_KINDS and event.effective_date < trade_date:
            if has_position or event.kind == EVENT_KIND_CODE_CHANGE:
                # 迟到的经济事件只标记复核，不回写历史（同迟到除权纪律）。
                late_corporate_action = True
                events.append(
                    {
                        "severity": "critical",
                        "event_type": "corporate_action_event_late",
                        "instrument": event.instrument,
                        "reason": "effective_date_already_passed",
                        "details": {
                            "event_key": event.event_key,
                            "kind": event.kind,
                            "effective_date": event.effective_date.isoformat(),
                        },
                    }
                )
            continue
        if event.kind == EVENT_KIND_CODE_CHANGE:
            outcome = apply_code_change(positions=state, event=event, trade_date=trade_date)
            for receivable in open_receivables:
                if str(receivable.get("instrument")) == event.instrument:
                    # 应收跟随证券身份迁移，金额不变（不重分类、不增 NAV）。
                    receivable["instrument"] = event.new_instrument
            events.extend(outcome["events"])
        elif event.kind in ECONOMIC_EVENT_KINDS:
            if not has_position:
                events.append(
                    {
                        "severity": "info",
                        "event_type": "corporate_action_no_position",
                        "instrument": event.instrument,
                        "reason": f"{event.kind}_without_position",
                        "details": {
                            "event_key": event.event_key,
                            "effective_date": event.effective_date.isoformat(),
                        },
                    }
                )
            else:
                outcome = apply_share_split(
                    position=position, event=event, trade_date=trade_date
                )
                events.extend(outcome["events"])
        elif event.kind == EVENT_KIND_UNSUPPORTED:
            # fail-closed：无数据源支撑的类型永不自动入账，记原因留待人工。
            events.append(
                {
                    "severity": "critical",
                    "event_type": "corporate_action_unsupported",
                    "instrument": event.instrument,
                    "reason": f"unsupported_event_type:{event.unsupported_type}",
                    "details": {
                        "event_key": event.event_key,
                        "unsupported_type": event.unsupported_type,
                        "effective_date": event.effective_date.isoformat(),
                        "title": event.title,
                        **dict(event.details or {}),
                    },
                }
            )
        elif event.kind == EVENT_KIND_CHOICE_REQUIRED:
            # 需持有人选择：只提醒并标记持仓待人工处理，不代客选择、不阻断卖出。
            events.append(
                {
                    "severity": "warning",
                    "event_type": "corporate_action_choice_required",
                    "instrument": event.instrument,
                    "reason": "holder_choice_pending_manual_handling",
                    "details": {
                        "event_key": event.event_key,
                        "effective_date": event.effective_date.isoformat(),
                        "title": event.title,
                        **dict(event.details or {}),
                    },
                }
            )
            if has_position:
                position["choice_pending"] = {
                    "event_key": event.event_key,
                    "since": event.effective_date.isoformat(),
                    "title": event.title,
                    "manual_action_required": True,
                }
        else:
            # 公告/名称变更：纯信息事件，不改现金、持仓或 NAV。
            events.append(
                {
                    "severity": "info",
                    "event_type": f"corporate_action_{event.kind}",
                    "instrument": event.instrument,
                    "reason": "informational_only",
                    "details": {
                        "event_key": event.event_key,
                        "stage": event.stage,
                        "effective_date": event.effective_date.isoformat(),
                        "title": event.title,
                        **dict(event.details or {}),
                    },
                }
            )
        run_event_keys.add(event.event_key)
        corporate_events_applied.append(
            {
                "event_key": event.event_key,
                "event_type": event.kind,
                "instrument": event.instrument,
                "effective_date": event.effective_date.isoformat(),
                "payload_sha256": event.payload_sha256,
                "details": {
                    "stage": event.stage,
                    "split_ratio": event.split_ratio,
                    "new_instrument": event.new_instrument,
                    "unsupported_type": event.unsupported_type,
                    "title": event.title,
                    "source": event.source,
                    **dict(event.details or {}),
                },
            }
        )

    reference_prices = _execution_reference_prices(bars)
    instruments = sorted(
        set(state)
        | set(targets)
        | {
            str(spec["instrument"]).upper()
            for spec in (order_specs_override or ())
        }
    )
    lot_rules = {instrument: order_unit_rules(instrument, trade_date) for instrument in instruments}
    desired: dict[str, int] = {}
    for instrument in instruments:
        price = reference_prices.get(instrument)
        if price is None:
            desired[instrument] = int(state.get(instrument, {}).get("quantity", 0))
            continue
        desired[instrument] = lot_floor(
            int(targets.get(instrument, 0.0) * prior_nav / price), lot_rules[instrument]
        )

    order_specs: list[dict[str, Any]] = []
    if order_specs_override is not None:
        # Persistent order book: work the still-open remainders directly; the
        # plan layer already applied lot rounding and sellable clamps, and the
        # fill loop below re-clamps sells against T+1 availability.
        for spec in order_specs_override:
            instrument = str(spec["instrument"]).upper()
            side = str(spec["side"]).lower()
            if side not in {"buy", "sell"}:
                raise ValueError(f"persistent order side must be buy or sell: {spec!r}")
            requested = int(spec["requested_quantity"])
            if requested <= 0:
                raise ValueError(f"persistent order quantity must be positive: {spec!r}")
            order_specs.append(
                {
                    "instrument": instrument,
                    "side": side,
                    "target_weight": float(spec.get("target_weight") or 0.0),
                    "requested_quantity": requested,
                    "requested_value": requested
                    * reference_prices.get(instrument, 0.0),
                    "order_ref": spec.get("order_ref"),
                    "limit_price": spec.get("limit_price"),
                    "not_before": spec.get("not_before"),
                    "not_after": spec.get("not_after"),
                    "reserved_cash": spec.get("reserved_cash"),
                }
            )
    else:
        for instrument in instruments:
            current = int(state.get(instrument, {}).get("quantity", 0))
            delta = desired[instrument] - current
            if delta == 0:
                continue
            side = "buy" if delta > 0 else "sell"
            requested = abs(delta)
            if side == "sell":
                available = _sellable_quantity(state[instrument], trade_date)
                requested = min(requested, available)
                if desired[instrument] > 0:
                    # Non-liquidating sells keep the board lot increment (100 on the
                    # main boards, 1 above the minimum on STAR/BSE); full exits may
                    # sell the odd lot.
                    increment = lot_rules[instrument].lot_increment
                    requested = requested // increment * increment
            if requested <= 0:
                order_specs.append(
                    _rejected_order(
                        instrument,
                        side,
                        targets.get(instrument, 0.0),
                        abs(delta),
                        "t_plus_one_unavailable",
                        reference_prices.get(instrument, 0.0),
                    )
                )
                continue
            order_specs.append(
                {
                    "instrument": instrument,
                    "side": side,
                    "target_weight": targets.get(instrument, 0.0),
                    "requested_quantity": requested,
                    "requested_value": requested * reference_prices.get(instrument, 0.0),
                }
            )

    # A-share sell proceeds are available for buys on the same trading day.
    order_specs.sort(key=lambda item: 0 if item["side"] == "sell" else 1)
    for spec in order_specs:
        if "status" in spec:
            orders.append(spec)
            events.append(_rejection_event(spec))
            continue
        instrument = spec["instrument"]
        side = spec["side"]
        gate_reason = etf_trading_gate(instrument)
        if gate_reason is not None:
            # ETF 子类型白名单门禁（设计 §1.3/§5.1）：未验收子类型 fail-closed
            # 拒单记原因；研究/回测路径不经此处，不受限。
            order = {**spec, **_empty_execution(gate_reason)}
            orders.append(order)
            events.append(_rejection_event(order))
            continue
        requested = int(spec["requested_quantity"])
        instrument_bars = bars[bars["instrument"] == instrument].set_index("datetime")
        if instrument_bars.empty:
            order = {**spec, **_empty_execution("missing_minute_bars")}
            orders.append(order)
            events.append(_rejection_event(order))
            continue
        slices = build_execution_slices(
            quantity=requested,
            side=side,
            trade_date=trade_date,
            policy=policy,
            signal_at=signal_at,
            instrument=instrument,
        )
        remaining = requested
        order_fills: list[dict[str, Any]] = []
        rejection_reasons: list[str] = []
        limit_price = spec.get("limit_price")
        reserved_cash = (
            float(spec["reserved_cash"])
            if spec.get("reserved_cash") is not None
            else None
        )
        if reserved_cash is not None and reserved_cash < 0:
            raise ValueError("persistent order reserved cash must be non-negative")
        not_before = _window_bound(spec.get("not_before"))
        not_after = _window_bound(spec.get("not_after"))
        for execution_slice in slices:
            if remaining <= 0:
                break
            scheduled = pd.Timestamp(execution_slice["scheduled_for"])
            if scheduled.tzinfo is not None:
                scheduled = scheduled.tz_convert(_SHANGHAI).tz_localize(None)
            else:
                scheduled = scheduled.tz_localize(None)
            if not_before is not None and scheduled < not_before:
                rejection_reasons.append("before_execution_window")
                continue
            if not_after is not None and scheduled > not_after:
                rejection_reasons.append("execution_window_elapsed")
                break
            if scheduled not in instrument_bars.index:
                rejection_reasons.append("missing_minute_bar")
                continue
            bar = instrument_bars.loc[scheduled]
            if isinstance(bar, pd.DataFrame):
                raise ValueError("minute execution bars contain duplicate timestamps")
            reason = _bar_rejection_reason(bar, side)
            if reason:
                rejection_reasons.append(reason)
                continue
            price = float(bar["vwap"])
            if limit_price is not None and (
                (side == "buy" and price > float(limit_price))
                or (side == "sell" and price < float(limit_price))
            ):
                # 价格保护：限价位之外不成交，余量留给后续批次。
                rejection_reasons.append("price_protection")
                continue
            minute_volume = int(floor(float(bar["volume"])))
            capacity = int(floor(minute_volume * float(policy["max_participation"])))
            slice_request = min(remaining, int(execution_slice["quantity"]))
            fill_quantity = min(slice_request, capacity)
            if side == "buy":
                fill_quantity = lot_floor(fill_quantity, lot_rules[instrument])
                spendable = available_cash
                if reserved_cash is not None:
                    spendable = min(spendable, reserved_cash)
                fill_quantity = _affordable_buy_quantity(
                    fill_quantity,
                    cash=spendable,
                    price=price,
                    participation=(fill_quantity / minute_volume if minute_volume else 0.0),
                    asset_type=infer_cn_asset_type(instrument),
                    trade_date=trade_date,
                    costs=cost_model,
                    rules=lot_rules[instrument],
                )
            if side == "sell":
                sell_position = state.get(instrument) or {}
                fill_quantity = min(
                    fill_quantity,
                    _sellable_quantity(sell_position, trade_date),
                )
            if fill_quantity <= 0:
                rejection_reasons.append(
                    "insufficient_cash"
                    if side == "buy" and cash <= price * lot_rules[instrument].min_lot
                    else "capacity"
                )
                continue
            participation = fill_quantity / minute_volume
            breakdown = cost_model.estimate_breakdown(
                side=side,
                gross_value=fill_quantity * price,
                participation=participation,
                asset_type=infer_cn_asset_type(instrument),
                trade_date=trade_date,
            )
            gross = fill_quantity * price
            fee = float(breakdown["total"])
            cash_delta = -(gross + fee) if side == "buy" else gross - fee
            if cash + cash_delta < -1e-6:
                raise RuntimeError("simulation execution would create negative cash")
            cash += cash_delta
            available_cash += cash_delta
            if side == "buy" and reserved_cash is not None:
                reserved_cash += cash_delta
            fill = {
                "instrument": instrument,
                "side": side,
                "executed_at": scheduled.to_pydatetime().replace(tzinfo=_SHANGHAI),
                "quantity": fill_quantity,
                "price": price,
                "gross_value": gross,
                "fee": fee,
                "cost_breakdown": breakdown,
                "minute_volume": minute_volume,
                "capacity_quantity": capacity,
            }
            if spec.get("order_ref") is not None:
                fill["order_ref"] = spec["order_ref"]
            order_fills.append(fill)
            fill["_seq"] = len(fills)
            fills.append(fill)
            cash_flows.append(
                {
                    "trade_date": trade_date,
                    "flow_type": "buy_settlement" if side == "buy" else "sell_settlement",
                    "amount": cash_delta,
                    "balance_after": cash,
                    "fill_seq": int(fill["_seq"]),
                    "order_ref": fill.get("order_ref"),
                }
            )
            consumed_lots = _apply_fill(state, fill, trade_date)
            if side == "sell" and instrument not in choice_sale_warned:
                pending_choice = (state.get(instrument) or {}).get("choice_pending")
                if pending_choice:
                    # 待人工处置的持有人选择事件：卖出不硬阻断，但必须可见。
                    choice_sale_warned.add(instrument)
                    events.append(
                        {
                            "severity": "warning",
                            "event_type": "corporate_action_choice_pending_sale",
                            "instrument": instrument,
                            "reason": "sold_while_holder_choice_pending",
                            "details": {
                                "event_key": str(pending_choice.get("event_key")),
                                "fill_quantity": fill_quantity,
                            },
                        }
                    )
            if side == "sell" and consumed_lots:
                dividend_tax, dividend_tax_details, dividend_tax_released = (
                    settle_dividend_tax(
                        instrument=instrument,
                        consumed=consumed_lots,
                        sale_date=trade_date,
                        tax_book=tax_book,
                    )
                )
                if dividend_tax > 0:
                    if cash - dividend_tax < -1e-6:
                        raise RuntimeError("dividend tax would create negative cash")
                    cash -= dividend_tax
                    available_cash -= dividend_tax
                    cash_flows.append(
                        {
                            "trade_date": trade_date,
                            "flow_type": "dividend_tax",
                            "amount": -dividend_tax,
                            "balance_after": cash,
                            "fill_seq": int(fill["_seq"]),
                            "order_ref": fill.get("order_ref"),
                        }
                    )
                if dividend_tax > 0 or dividend_tax_released > 0:
                    # 现金只流出实际税额；已提负债随批次消耗释放（见 NAV 减项），
                    # 卖出对 NAV 的净影响 = 释放额 − 实际税额（差额确认）。
                    fill["cost_breakdown"] = {
                        **fill["cost_breakdown"],
                        "dividend_tax": dividend_tax,
                        "dividend_tax_details": dividend_tax_details,
                        "dividend_tax_liability_released": dividend_tax_released,
                    }
            remaining -= fill_quantity
        filled = requested - remaining
        status = (
            "filled"
            if remaining == 0
            else ("partial_filled_expired" if filled else "rejected")
        )
        reject_reason = None
        if remaining:
            reject_reason = ",".join(sorted(set(rejection_reasons or ["capacity"])))
        order = {
            **spec,
            "filled_quantity": filled,
            "filled_value": sum(item["gross_value"] for item in order_fills),
            "capacity_fill_ratio": filled / requested,
            "status": status,
            "reject_reason": reject_reason,
            "expires_at": datetime.combine(trade_date, time(15, 0), _SHANGHAI),
        }
        orders.append(order)
        if remaining:
            events.append(_rejection_event(order))

    if external_flow_close:
        # F_t_close：盘后确认的外部现金流，当日不可交易，次日进入可投资现金；
        # 计入当日收盘 NAV 与守恒校验，但在 TWR 公式中从分子剔除。
        if cash + external_flow_close < -1e-6:
            raise RuntimeError("external withdrawal would create negative cash")
        cash += external_flow_close
        if external_flow_close < 0:
            available_cash += external_flow_close
            if available_cash < -1e-6:
                raise RuntimeError("external withdrawal exceeds tradable cash")
        cash_flows.append(
            {
                "trade_date": trade_date,
                "flow_type": (
                    "external_deposit_close"
                    if external_flow_close > 0
                    else "external_withdrawal_close"
                ),
                "amount": external_flow_close,
                "balance_after": cash,
            }
        )
    if cash < -1e-6:
        raise RuntimeError("simulation ledger cash conservation failed")
    if abs(cash - (starting_cash + sum(item["amount"] for item in cash_flows))) > 1e-6:
        raise RuntimeError("simulation ledger cash flows do not reconcile")
    receivables_total = round(
        sum(float(item.get("amount", 0.0)) for item in open_receivables), 2
    )
    # 应付股息税负债是 NAV 减项（设计 §5.6：税额未定须记保守税费负债）：
    # 除权日按保守档位计提，卖出结算只确认实际与计提的差额，到账重分类不动它。
    tax_liabilities_total = pending_dividend_tax_liability(state)
    valuation = _value_positions(state, closing_prices, trade_date)
    events.extend(valuation["events"])
    nav = cash + valuation["market_value"] + receivables_total - tax_liabilities_total
    new_peak = max(high_water_mark, nav)
    has_stale = valuation["has_stale_prices"]
    # 单位化 TWR（设计 4.4）：与人民币 NAV 口径并存；外部现金流不制造收益。
    unitized = chain_unitized_day(
        prior_nav=prior_nav,
        nav=nav,
        flow_open=external_flow_open,
        flow_close=external_flow_close,
        prior_wealth=prior_investment_wealth,
        prior_high_water_mark=twr_high_water_mark,
    )
    nav_row = {
        "trade_date": trade_date,
        "cash": cash,
        "tradable_cash": available_cash,
        "market_value": valuation["market_value"],
        "corporate_receivables": receivables_total,
        "corporate_tax_liabilities": tax_liabilities_total,
        "nav": nav,
        "daily_return": nav / prior_nav - 1.0,
        "drawdown": nav / new_peak - 1.0,
        "external_flow_open": external_flow_open,
        "external_flow_close": external_flow_close,
        "twr_daily_return": unitized["daily_return"],
        "investment_wealth": unitized["investment_wealth"],
        "twr_drawdown": unitized["drawdown"],
        "twr_status": unitized["status"],
        "market_date": valuation["market_date"],
        "has_stale_prices": has_stale,
        "status": "degraded" if has_stale else "healthy",
        "performance_certified": (not has_stale) and not late_corporate_action,
    }
    return {
        "engine_version": SIMULATION_ENGINE_VERSION,
        "trade_date": trade_date,
        "cash": cash,
        "nav": nav,
        "high_water_mark": new_peak,
        "investment_wealth": unitized["investment_wealth"],
        "twr_high_water_mark": unitized["high_water_mark"],
        "positions": state,
        "orders": orders,
        "fills": fills,
        "cash_flows": cash_flows,
        "nav_row": nav_row,
        "events": events,
        "dividend_receivables": open_receivables,
        "corporate_actions_applied": corporate_actions_applied,
        "corporate_events_applied": corporate_events_applied,
        "conservation": {
            "cash_difference": cash
            - (starting_cash + sum(item["amount"] for item in cash_flows)),
            "negative_positions": sum(
                1 for item in state.values() if int(item.get("quantity", 0)) < 0
            ),
            **(
                {"corporate_receivables": receivables_total}
                if receivables_total
                else {}
            ),
            **(
                {"corporate_tax_liabilities": tax_liabilities_total}
                if tax_liabilities_total
                else {}
            ),
        },
    }


def execute_atomic_pair_day(
    *,
    trade_date: date,
    cash: float,
    prior_nav: float,
    high_water_mark: float,
    positions: dict[str, dict[str, Any]],
    target_payload: dict[str, Any],
    minute_bars: pd.DataFrame,
    closing_prices: dict[str, dict[str, Any]],
    shortability: dict[str, bool],
    cost_model: CostModelConfig | None = None,
    cost_schedule: CostScheduleBook | None = None,
    execution_policy: dict[str, Any],
    external_flow_open: float = 0.0,
    external_flow_close: float = 0.0,
    prior_investment_wealth: float | None = 1.0,
    twr_high_water_mark: float | None = 1.0,
) -> dict[str, Any]:
    """Execute a governed pair target as one all-filled or all-rejected atomic group."""

    if not isfinite(cash) or cash < 0 or prior_nav <= 0 or high_water_mark <= 0:
        raise ValueError("simulation account balances are invalid")
    if external_flow_open or external_flow_close:
        # 配对账户是离线研究台账，不接受外部现金流（设计 6.4.3/12.1）。
        raise ValueError("pair research ledgers do not accept external cash flows")
    cost_model = _resolve_cost_schedule(cost_model, cost_schedule).as_of(trade_date)
    policy = normalize_execution_policy(execution_policy)
    group_id = str(target_payload.get("atomic_group_id") or "").strip()
    legs = [dict(item) for item in (target_payload.get("legs") or [])]
    if not group_id or len(legs) != 2:
        raise ValueError("pair execution requires exactly one governed atomic group")
    if {int(item.get("leg_no") or 0) for item in legs} != {1, 2}:
        raise ValueError("pair execution leg numbers must be 1 and 2")
    if {str(item.get("position_side")) for item in legs} != {"long", "short"}:
        raise ValueError("pair execution requires one long and one short leg")
    state = deepcopy(positions)
    bars = _normalize_bars(minute_bars, trade_date)
    specs: list[dict[str, Any]] = []
    rejection: str | None = None
    for leg in sorted(legs, key=lambda item: int(item["leg_no"])):
        instrument = str(leg["instrument"]).upper()
        position_side = str(leg["position_side"])
        target = int(leg["target_quantity"])
        lot_rule = order_unit_rules(instrument, trade_date)
        if target < 0 or (target and not is_valid_order_quantity(target, lot_rule)):
            raise ValueError("pair target quantities must satisfy board order-unit rules")
        current_row = state.get(instrument) or {}
        current = int(current_row.get("quantity", 0))
        current_side = str(current_row.get("position_side") or position_side)
        if current and current_side != position_side:
            raise ValueError("pair target cannot reverse an existing leg in one batch")
        if (
            position_side == "long"
            and current
            and (
                current_row.get("last_trade_date") is None
                or _as_date(current_row["last_trade_date"]) < trade_date
            )
        ):
            current_row["available_quantity"] = current
        delta = target - current
        side = (
            ("buy" if delta > 0 else "sell")
            if position_side == "long"
            else ("sell_short" if delta > 0 else "buy_to_cover")
        )
        specs.append(
            {
                "instrument": instrument,
                "side": side,
                "cost_side": "buy" if side in {"buy", "buy_to_cover"} else "sell",
                "atomic_group_id": group_id,
                "leg_no": int(leg["leg_no"]),
                "position_side": position_side,
                "annual_borrow_rate": float(leg.get("annual_borrow_rate") or 0.0),
                "target_quantity": target,
                "starting_quantity": current,
                "target_weight": 0.0,
                "requested_quantity": abs(delta),
            }
        )
    active = [item for item in specs if item["requested_quantity"] > 0]
    if len(active) == 1:
        rejection = "unbalanced_pair_target"
    short_leg = next(item for item in specs if item["position_side"] == "short")
    starting_cash = cash
    cash_flows: list[dict[str, Any]] = []
    carry_borrow_cost = 0.0
    starting_short = int(short_leg["starting_quantity"])
    if starting_short:
        if not 0 < short_leg["annual_borrow_rate"] <= 1:
            raise ValueError("an existing short leg requires a governed borrow rate")
        reference = _execution_reference_prices(bars).get(short_leg["instrument"])
        closing_quote = closing_prices.get(short_leg["instrument"])
        if reference is None and closing_quote is not None:
            market_date = closing_quote.get("market_date")
            closing_date = _as_date(market_date) if market_date is not None else None
            closing_value = float(closing_quote.get("price") or 0.0)
            if closing_date == trade_date and isfinite(closing_value) and closing_value > 0:
                reference = closing_value
        if reference is None or not isfinite(float(reference)) or float(reference) <= 0:
            raise ValueError("existing pair short borrow carry cannot be priced for the trade date")
        carry_borrow_cost = (
            starting_short
            * float(reference)
            * float(short_leg["annual_borrow_rate"])
            / 252.0
        )
        if cash < carry_borrow_cost:
            raise RuntimeError("pair borrow cost would create negative cash")
        cash -= carry_borrow_cost
        state[short_leg["instrument"]]["borrow_cost"] = float(
            state[short_leg["instrument"]].get("borrow_cost", 0.0)
        ) + carry_borrow_cost
        cash_flows.append(
            {
                "trade_date": trade_date,
                "flow_type": "pair_borrow_carry",
                "amount": -carry_borrow_cost,
                "balance_after": cash,
            }
        )
    if (
        short_leg["target_quantity"] > 0
        and shortability.get(short_leg["instrument"]) is not True
    ):
        rejection = "short_borrow_not_authorized"
    if short_leg["target_quantity"] > 0 and not 0 < short_leg["annual_borrow_rate"] <= 1:
        rejection = "borrow_cost_not_governed"
    for spec in active:
        if spec["side"] == "sell":
            available = int(state.get(spec["instrument"], {}).get("available_quantity", 0))
            if available < spec["requested_quantity"]:
                rejection = "t_plus_one_unavailable"

    execution_rows: dict[str, pd.Series] = {}
    executed_at: pd.Timestamp | None = None
    if active and rejection is None:
        leg_frames: dict[str, pd.DataFrame] = {}
        for spec in active:
            frame = bars[bars["instrument"] == spec["instrument"]].set_index("datetime")
            leg_frames[spec["instrument"]] = frame
        common = set.intersection(*(set(frame.index) for frame in leg_frames.values()))
        allowed = sorted(
            value
            for value in common
            if time(10, 0) <= value.time() <= time(11, 20)
            or time(13, 30) <= value.time() <= time(14, 50)
        )
        if not allowed:
            rejection = "missing_common_execution_bar"
        else:
            executed_at = pd.Timestamp(allowed[0])
            for spec in active:
                row = leg_frames[spec["instrument"]].loc[executed_at]
                if isinstance(row, pd.DataFrame):
                    raise ValueError("pair minute bars contain duplicate timestamps")
                reason = _bar_rejection_reason(row, spec["cost_side"])
                if reason:
                    rejection = reason
                    break
                capacity = int(
                    floor(float(row["volume"]) * float(policy["max_participation"]))
                )
                if capacity < spec["requested_quantity"]:
                    rejection = "atomic_capacity"
                    break
                execution_rows[spec["instrument"]] = row

    prepared: list[tuple[dict[str, Any], dict[str, Any], float, float]] = []
    cash_delta_total = 0.0
    if active and rejection is None and executed_at is not None:
        for spec in active:
            row = execution_rows[spec["instrument"]]
            quantity = int(spec["requested_quantity"])
            price = float(row["vwap"])
            minute_volume = int(floor(float(row["volume"])))
            participation = quantity / minute_volume
            costs = cost_model
            borrow_days = 0
            if spec["side"] == "sell_short":
                costs = CostModelConfig(
                    **{
                        **cost_model.to_dict(),
                        "annual_borrow_rate": spec["annual_borrow_rate"],
                    }
                )
                borrow_days = 1
            breakdown = costs.estimate_breakdown(
                side=spec["cost_side"],
                gross_value=quantity * price,
                participation=participation,
                asset_type=infer_cn_asset_type(spec["instrument"]),
                trade_date=trade_date,
                borrow_days=borrow_days,
            )
            gross = quantity * price
            fee = float(breakdown["total"])
            delta = -(gross + fee) if spec["cost_side"] == "buy" else gross - fee
            cash_delta_total += delta
            prepared.append((spec, breakdown, gross, delta))
        if cash + cash_delta_total < -1e-6:
            rejection = "insufficient_cash"

    if rejection is not None:
        orders = [
            {
                **spec,
                "requested_value": float(spec["requested_quantity"])
                * float(_execution_reference_prices(bars).get(spec["instrument"], 0.0)),
                "filled_quantity": 0,
                "filled_value": 0.0,
                "capacity_fill_ratio": 0.0,
                "status": "rejected",
                "reject_reason": rejection,
                "expires_at": datetime.combine(trade_date, time(15, 0), _SHANGHAI),
                "borrow_cost": 0.0,
            }
            for spec in specs
        ]
        events = [
            {
                "severity": "critical",
                "event_type": "atomic_pair_rejected",
                "instrument": None,
                "reason": rejection,
                "details": {"atomic_group_id": group_id},
            }
        ]
        return _pair_result(
            trade_date=trade_date,
            starting_cash=starting_cash,
            cash=cash,
            prior_nav=prior_nav,
            high_water_mark=high_water_mark,
            state=state,
            closing_prices=closing_prices,
            orders=orders,
            fills=[],
            cash_flows=cash_flows,
            events=events,
            certified=False,
            prior_investment_wealth=prior_investment_wealth,
            twr_high_water_mark=twr_high_water_mark,
        )

    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    if not active:
        return _pair_result(
            trade_date=trade_date,
            starting_cash=starting_cash,
            cash=cash,
            prior_nav=prior_nav,
            high_water_mark=high_water_mark,
            state=state,
            closing_prices=closing_prices,
            orders=[],
            fills=[],
            cash_flows=cash_flows,
            events=[],
            certified=True,
            prior_investment_wealth=prior_investment_wealth,
            twr_high_water_mark=twr_high_water_mark,
        )
    assert executed_at is not None
    for spec, breakdown, gross, cash_delta in prepared:
        quantity = int(spec["requested_quantity"])
        row = execution_rows[spec["instrument"]]
        price = float(row["vwap"])
        borrow_cost = float(breakdown["borrow_cost"])
        fill = {
            **{
                key: spec[key]
                for key in (
                    "instrument",
                    "side",
                    "atomic_group_id",
                    "leg_no",
                    "position_side",
                )
            },
            "executed_at": executed_at.to_pydatetime().replace(tzinfo=_SHANGHAI),
            "quantity": quantity,
            "price": price,
            "gross_value": gross,
            "fee": float(breakdown["total"]),
            "borrow_cost": borrow_cost,
            "cost_breakdown": breakdown,
            "minute_volume": int(floor(float(row["volume"]))),
            "capacity_quantity": int(
                floor(float(row["volume"]) * float(policy["max_participation"]))
            ),
        }
        fills.append(fill)
        orders.append(
            {
                **spec,
                "requested_value": gross,
                "filled_quantity": quantity,
                "filled_value": gross,
                "capacity_fill_ratio": 1.0,
                "status": "filled",
                "reject_reason": None,
                "expires_at": datetime.combine(trade_date, time(15, 0), _SHANGHAI),
                "borrow_cost": borrow_cost,
            }
        )
        cash += cash_delta
        cash_flows.append(
            {
                "trade_date": trade_date,
                "flow_type": f"pair_{spec['side']}",
                "amount": cash_delta,
                "balance_after": cash,
            }
        )
        _apply_pair_fill(state, fill, trade_date)
    return _pair_result(
        trade_date=trade_date,
        starting_cash=starting_cash,
        cash=cash,
        prior_nav=prior_nav,
        high_water_mark=high_water_mark,
        state=state,
        closing_prices=closing_prices,
        orders=orders,
        fills=fills,
        cash_flows=cash_flows,
        events=[],
        certified=True,
        prior_investment_wealth=prior_investment_wealth,
        twr_high_water_mark=twr_high_water_mark,
    )


def _apply_pair_fill(
    state: dict[str, dict[str, Any]], fill: dict[str, Any], trade_date: date
) -> None:
    instrument = str(fill["instrument"])
    side = str(fill["side"])
    filled = int(fill["quantity"])
    position = state.setdefault(
        instrument,
        {
            "quantity": 0,
            "available_quantity": 0,
            "average_cost": 0.0,
            "position_side": fill["position_side"],
            "borrow_cost": 0.0,
        },
    )
    quantity = int(position.get("quantity", 0))
    if side in {"buy", "sell_short"}:
        prior_cost = quantity * float(position.get("average_cost", 0.0))
        position["quantity"] = quantity + filled
        position["average_cost"] = (prior_cost + float(fill["gross_value"])) / position[
            "quantity"
        ]
        if side == "sell_short":
            position["available_quantity"] = position["quantity"]
    else:
        if filled > quantity:
            raise RuntimeError("pair close quantity exceeds its governed leg")
        position["quantity"] = quantity - filled
        if side == "sell":
            available = int(position.get("available_quantity", 0))
            if filled > available:
                raise RuntimeError("pair long leg violates T+1")
            position["available_quantity"] = available - filled
        else:
            position["available_quantity"] = position["quantity"]
        if position["quantity"] == 0:
            position["average_cost"] = 0.0
    position.update(
        atomic_group_id=fill["atomic_group_id"],
        leg_no=int(fill["leg_no"]),
        position_side=fill["position_side"],
        borrow_cost=float(position.get("borrow_cost", 0.0))
        + float(fill.get("borrow_cost", 0.0)),
        last_trade_date=trade_date,
    )


def _pair_result(
    *,
    trade_date: date,
    starting_cash: float,
    cash: float,
    prior_nav: float,
    high_water_mark: float,
    state: dict[str, dict[str, Any]],
    closing_prices: dict[str, dict[str, Any]],
    orders: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    cash_flows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    certified: bool,
    prior_investment_wealth: float | None = 1.0,
    twr_high_water_mark: float | None = 1.0,
) -> dict[str, Any]:
    market_value = 0.0
    market_dates: list[date] = []
    stale = False
    for instrument in list(state):
        position = state[instrument]
        quantity = int(position.get("quantity", 0))
        if quantity == 0:
            del state[instrument]
            continue
        quote = closing_prices.get(instrument)
        if quote:
            price = float(quote["price"])
            market_date = _as_date(quote["market_date"])
            is_stale = market_date != trade_date
        else:
            price = float(position.get("market_price") or 0.0)
            previous_date = position.get("market_date")
            market_date = _as_date(previous_date) if previous_date else None
            is_stale = True
        sign = -1.0 if position.get("position_side") == "short" else 1.0
        value = sign * quantity * price
        position.update(
            market_price=price,
            market_date=market_date,
            stale=is_stale,
            market_value=value,
        )
        market_value += value
        stale = stale or is_stale
        if market_date:
            market_dates.append(market_date)
    nav = cash + market_value
    peak = max(high_water_mark, nav)
    certified = certified and not stale
    # 配对台账无外部现金流：单位化链退化为 NAV 收益连乘，口径与主台账一致。
    unitized = chain_unitized_day(
        prior_nav=prior_nav,
        nav=nav,
        flow_open=0.0,
        flow_close=0.0,
        prior_wealth=prior_investment_wealth,
        prior_high_water_mark=twr_high_water_mark,
    )
    cash_difference = cash - (starting_cash + sum(item["amount"] for item in cash_flows))
    if abs(cash_difference) < 1e-9:
        cash_difference = 0.0
    return {
        "engine_version": SIMULATION_ENGINE_VERSION,
        "trade_date": trade_date,
        "cash": cash,
        "nav": nav,
        "high_water_mark": peak,
        "investment_wealth": unitized["investment_wealth"],
        "twr_high_water_mark": unitized["high_water_mark"],
        "positions": state,
        "orders": orders,
        "fills": fills,
        "cash_flows": cash_flows,
        "nav_row": {
            "trade_date": trade_date,
            "cash": cash,
            "market_value": market_value,
            "nav": nav,
            "daily_return": nav / prior_nav - 1.0,
            "drawdown": nav / peak - 1.0,
            "external_flow_open": 0.0,
            "external_flow_close": 0.0,
            "twr_daily_return": unitized["daily_return"],
            "investment_wealth": unitized["investment_wealth"],
            "twr_drawdown": unitized["drawdown"],
            "twr_status": unitized["status"],
            "market_date": min(market_dates) if market_dates else None,
            "has_stale_prices": stale,
            "status": "healthy" if certified else "degraded",
            "performance_certified": certified,
        },
        "events": events,
        "conservation": {
            "cash_difference": cash_difference,
            "negative_positions": 0,
        },
    }


def _normalize_bars(values: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    required = {
        "datetime",
        "instrument",
        "close",
        "vwap",
        "volume",
        "paused",
        "up_limit",
        "down_limit",
    }
    if not required.issubset(values.columns):
        raise ValueError("minute bars are missing execution fields")
    result = values.loc[:, sorted(required)].copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    if result["datetime"].dt.tz is not None:
        result["datetime"] = result["datetime"].dt.tz_convert(_SHANGHAI).dt.tz_localize(None)
    result["instrument"] = result["instrument"].astype(str).str.upper()
    for column in ("close", "vwap", "volume", "paused", "up_limit", "down_limit"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result[result["datetime"].dt.date == trade_date]
    if result.duplicated(["datetime", "instrument"]).any():
        raise ValueError("minute bars contain duplicate instrument timestamps")
    if result[["datetime", "instrument", "close", "volume"]].isna().any().any():
        raise ValueError("minute bars contain invalid values")
    return result.sort_values(["instrument", "datetime"])


def _execution_reference_prices(bars: pd.DataFrame) -> dict[str, float]:
    if bars.empty:
        return {}
    first = bars.groupby("instrument", sort=False).first()
    return {str(index): float(row["close"]) for index, row in first.iterrows()}


def _bar_rejection_reason(bar: pd.Series, side: str) -> str | None:
    if float(bar["paused"]) > 0 or float(bar["volume"]) <= 0:
        return "suspended"
    price = float(bar["close"])
    if not isfinite(price) or price <= 0:
        return "invalid_price"
    if side == "buy" and isfinite(float(bar["up_limit"])) and price >= float(bar["up_limit"]):
        return "limit_up"
    if side == "sell" and isfinite(float(bar["down_limit"])) and price <= float(bar["down_limit"]):
        return "limit_down"
    return None


def _affordable_buy_quantity(
    quantity: int,
    *,
    cash: float,
    price: float,
    participation: float,
    asset_type: str,
    trade_date: date,
    costs: CostModelConfig,
    rules: OrderUnitRules,
) -> int:
    result = lot_floor(min(quantity, int(cash / price)), rules) if price > 0 else 0
    while result > 0:
        gross = result * price
        fee = costs.estimate(
            side="buy",
            gross_value=gross,
            participation=participation,
            asset_type=asset_type,
            trade_date=trade_date,
        )
        if gross + fee <= cash + 1e-9:
            return result
        result = lot_floor(result - rules.lot_increment, rules)
    return 0


def _sellable_quantity(position: dict[str, Any], trade_date: date) -> int:
    """Sellable shares: stored T+1 availability, tightened by per-lot sellable_from.

    无批次的旧持仓行为不变；有批次时取 min(存储可卖, Σ sellable_from<=当日 的批次)，
    送转新增股份在新增股份上市日前由此锁定。
    """

    available = int(position.get("available_quantity", 0))
    lots = position.get("lots")
    if not lots:
        return available
    sellable = 0
    for lot in lots:
        sellable_from = lot.get("sellable_from")
        lot_sellable_from = _as_date(sellable_from) if sellable_from else date.min
        if lot_sellable_from <= trade_date:
            sellable += int(lot.get("quantity", 0))
    return min(available, sellable)


def _apply_fill(
    state: dict[str, dict[str, Any]], fill: dict[str, Any], trade_date: date
) -> list[tuple[dict, int]]:
    instrument = fill["instrument"]
    position = state.setdefault(
        instrument,
        {"quantity": 0, "available_quantity": 0, "average_cost": 0.0},
    )
    quantity = int(position["quantity"])
    filled = int(fill["quantity"])
    lots = position_lots(position, trade_date=trade_date)
    if fill["side"] == "buy":
        total_cost = quantity * float(position.get("average_cost", 0.0))
        total_cost += float(fill["gross_value"]) + float(fill["fee"])
        position["quantity"] = quantity + filled
        position["average_cost"] = total_cost / position["quantity"]
        lots.append(
            {
                "lot_key": f"buy:{fill['executed_at'].isoformat()}:{fill.get('_seq', 0)}",
                "acquired_at": trade_date,
                # 买入批次按 T+1 锁定到下一自然日；日初解锁逻辑与之叠加后等价。
                "sellable_from": trade_date + timedelta(days=1),
                "quantity": filled,
                "cost_basis_total": float(fill["gross_value"]) + float(fill["fee"]),
                "origin": LOT_ORIGIN_BUY,
                "entitlements": [],
            }
        )
        # New buys stay unavailable until the next trading day.
    else:
        available = int(position.get("available_quantity", 0))
        if filled > available or filled > quantity:
            raise RuntimeError("simulation fill creates a short position")
        position["quantity"] = quantity - filled
        position["available_quantity"] = available - filled
        if position["quantity"] == 0:
            position["average_cost"] = 0.0
        position["last_trade_date"] = trade_date
        return consume_lots_fifo(lots, filled, trade_date=trade_date)
    position["last_trade_date"] = trade_date
    return []


def _value_positions(
    state: dict[str, dict[str, Any]],
    closing_prices: dict[str, dict[str, Any]],
    trade_date: date,
) -> dict[str, Any]:
    market_value = 0.0
    market_dates: list[date] = []
    events: list[dict[str, Any]] = []
    stale = False
    for instrument in list(state):
        position = state[instrument]
        quantity = int(position.get("quantity", 0))
        if quantity == 0:
            if position.get("lots"):
                raise RuntimeError(f"position lot invariant violated for {instrument}")
            del state[instrument]
            continue
        quote = closing_prices.get(instrument)
        if quote and bool(quote.get("delisted")) and not bool(quote.get("cash_liquidated")):
            price = 0.0
            quote_date = trade_date
            is_stale = False
            events.append(
                {
                    "severity": "critical",
                    "event_type": "delisted_zero_valuation",
                    "instrument": instrument,
                    "reason": "delisted_without_cash_liquidation",
                    "details": {"quantity": quantity},
                }
            )
        elif quote:
            price = float(quote["price"])
            quote_date = _as_date(quote["market_date"])
            is_stale = quote_date != trade_date
        else:
            previous_price = position.get("market_price")
            previous_date = position.get("market_date")
            if previous_price is None or previous_date is None:
                price = 0.0
                quote_date = None
            else:
                price = float(previous_price)
                quote_date = _as_date(previous_date)
            is_stale = True
        if not isfinite(price) or price < 0:
            raise ValueError(f"invalid closing valuation for {instrument}")
        value = quantity * price
        position.update(
            {
                "market_price": price,
                "market_date": quote_date,
                "stale": is_stale,
                "market_value": value,
            }
        )
        market_value += value
        if quote_date is not None:
            market_dates.append(quote_date)
        if is_stale:
            stale = True
            events.append(
                {
                    "severity": "warning",
                    "event_type": "stale_valuation",
                    "instrument": instrument,
                    "reason": "closing_price_not_current",
                    "details": {"market_date": quote_date.isoformat() if quote_date else None},
                }
            )
    return {
        "market_value": market_value,
        "market_date": min(market_dates) if market_dates else None,
        "has_stale_prices": stale,
        "events": events,
    }


def _empty_execution(reason: str) -> dict[str, Any]:
    return {
        "filled_quantity": 0,
        "filled_value": 0.0,
        "capacity_fill_ratio": 0.0,
        "status": "rejected",
        "reject_reason": reason,
        "expires_at": None,
    }


def _rejected_order(
    instrument: str,
    side: str,
    target_weight: float,
    requested: int,
    reason: str,
    price: float,
) -> dict[str, Any]:
    return {
        "instrument": instrument,
        "side": side,
        "target_weight": target_weight,
        "requested_quantity": requested,
        "requested_value": requested * price,
        **_empty_execution(reason),
    }


def _rejection_event(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "severity": "warning",
        "event_type": "order_rejected" if order["status"] == "rejected" else "order_expired",
        "instrument": order["instrument"],
        "reason": order.get("reject_reason") or "unfilled",
        "details": {
            "requested_quantity": order["requested_quantity"],
            "filled_quantity": order.get("filled_quantity", 0),
        },
    }


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _window_bound(value: Any) -> pd.Timestamp | None:
    """Normalize an execution-window bound to naive Shanghai time (or None)."""

    if value is None:
        return None
    bound = pd.Timestamp(value)
    if bound.tzinfo is not None:
        return bound.tz_convert(_SHANGHAI).tz_localize(None)
    return bound.tz_localize(None)
