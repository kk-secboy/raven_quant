"""Pure corporate-action ledger primitives (现金分红/送转四阶段入账).

No database, no system clock, no side effects — the same discipline as the
project ``ExecutionCore``: every function takes explicit state and returns
explicit deltas.  All rounding rules live here and are deterministic:

- 金额四舍五入到分（0.01 CNY）。
- 送转零碎股：目标新增总数按 持仓×比例 四舍五入到整股；各批次按比例 floor
  分配，余数按取得时间从早到晚逐股补齐（确定性规则，非交易所排序规则，
  属于保守近似并在事件中标明）。
- 除权日资格量 = 除权日盘前持仓量（A 股除权日恒为登记日次一交易日，
  登记收盘与除权开盘之间无法交易，二者等价）。

设计出处：个人量化投资与模拟盘系统设计稿 §5.6/§5.7。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import floor
from typing import Any

from .cost_model import infer_cn_asset_type
from .dividend_tax import (
    SETTLEMENT_AT_PAYMENT,
    SETTLEMENT_AT_SALE,
    DividendTaxRule,
    DividendTaxRuleBook,
    is_dividend_tax_exempt,
    rate_for_holding,
)

DIVIDEND_PAR_VALUE_CNY = 1.0
"""A 股股票默认面值（元）；送股（红利股）按面值计入股息红利税基，转增不计。"""

LOT_ORIGIN_BUY = "buy"
LOT_ORIGIN_BONUS_SHARE = "bonus_share"
LOT_ORIGIN_LEGACY = "legacy"

ENTITLEMENT_KIND_CASH = "cash"
ENTITLEMENT_KIND_BONUS_PAR = "bonus_par"

_MONEY_DIGITS = 2


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"none", "nat", "nan"}:
        return None
    if len(text) == 8 and text.isdigit():
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    return date.fromisoformat(text[:10])


def _round_money(value: float) -> float:
    return round(value + 1e-12, _MONEY_DIGITS)


def _round_half_up(value: float) -> int:
    return int(floor(value + 0.5))


def ts_code_to_instrument(ts_code: str) -> str:
    """``600000.SH`` → ``SH600000``；已是引擎格式则原样返回。"""

    text = str(ts_code).strip().upper()
    if "." in text:
        digits, exchange = text.split(".", 1)
        return f"{exchange}{digits}"
    return text


def _action_payload(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CorporateAction:
    """One normalized cash-dividend / bonus-share event for one instrument."""

    instrument: str
    ex_date: date
    record_date: date | None
    pay_date: date | None
    cash_div_pretax: float
    cash_div_aftertax: float | None
    bonus_share_ratio: float
    conversion_ratio: float
    list_date: date | None
    source_ts_code: str
    payload_sha256: str

    @property
    def share_ratio(self) -> float:
        return self.bonus_share_ratio + self.conversion_ratio

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CorporateAction:
        source = dict(values)
        ex_date = _parse_date(source.get("ex_date"))
        if ex_date is None:
            raise ValueError("corporate action requires an ex_date")
        payload = {
            key: source.get(key)
            for key in (
                "instrument",
                "ex_date",
                "record_date",
                "pay_date",
                "cash_div_pretax",
                "cash_div_aftertax",
                "bonus_share_ratio",
                "conversion_ratio",
                "list_date",
                "source_ts_code",
            )
        }
        return cls(
            instrument=str(source["instrument"]).upper(),
            ex_date=ex_date,
            record_date=_parse_date(source.get("record_date")),
            pay_date=_parse_date(source.get("pay_date")),
            cash_div_pretax=float(source.get("cash_div_pretax") or 0.0),
            cash_div_aftertax=(
                None
                if source.get("cash_div_aftertax") is None
                else float(source["cash_div_aftertax"])
            ),
            bonus_share_ratio=float(source.get("bonus_share_ratio") or 0.0),
            conversion_ratio=float(source.get("conversion_ratio") or 0.0),
            list_date=_parse_date(source.get("list_date")),
            source_ts_code=str(source.get("source_ts_code") or ""),
            payload_sha256=str(source.get("payload_sha256") or _action_payload(payload)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "ex_date": self.ex_date.isoformat(),
            "record_date": self.record_date.isoformat() if self.record_date else None,
            "pay_date": self.pay_date.isoformat() if self.pay_date else None,
            "cash_div_pretax": self.cash_div_pretax,
            "cash_div_aftertax": self.cash_div_aftertax,
            "bonus_share_ratio": self.bonus_share_ratio,
            "conversion_ratio": self.conversion_ratio,
            "list_date": self.list_date.isoformat() if self.list_date else None,
            "source_ts_code": self.source_ts_code,
            "payload_sha256": self.payload_sha256,
        }


def normalize_dividend_rows(rows: Iterable[Mapping[str, Any]]) -> list[CorporateAction]:
    """Normalize raw Tushare ``dividend`` rows; fail-closed on corrupt magnitudes.

    只接受已经给出除权日的行（预案/不分配等没有 ``ex_date``，天然跳过）。
    数量级哨兵：比例 ∈ [0, 10)、每股现金 ∈ [0, 1000)，越界直接抛错，
    不让坏数据静默入账（设计 §3.5 质量门纪律）。
    """

    actions: list[CorporateAction] = []
    for row in rows:
        source = dict(row)
        ex_date = _parse_date(source.get("ex_date"))
        if ex_date is None:
            continue
        ts_code = str(source.get("ts_code") or "")
        bonus = float(source.get("stk_bo_rate") or 0.0)
        conversion = float(source.get("stk_co_rate") or 0.0)
        pretax = float(source.get("cash_div_tax") or 0.0)
        aftertax_raw = source.get("cash_div")
        aftertax = None if aftertax_raw is None else float(aftertax_raw)
        if not (0.0 <= bonus < 10.0 and 0.0 <= conversion < 10.0):
            raise ValueError(f"dividend share ratio out of range for {ts_code}: {source}")
        if pretax < 0.0 or pretax >= 1000.0 or (aftertax is not None and aftertax < 0.0):
            raise ValueError(f"dividend cash amount out of range for {ts_code}: {source}")
        if pretax <= 0.0 and (aftertax or 0.0) <= 0.0 and bonus <= 0.0 and conversion <= 0.0:
            continue
        actions.append(
            CorporateAction.from_mapping(
                {
                    "instrument": ts_code_to_instrument(ts_code),
                    "ex_date": ex_date,
                    "record_date": _parse_date(source.get("record_date")),
                    "pay_date": _parse_date(source.get("pay_date")),
                    "cash_div_pretax": pretax,
                    "cash_div_aftertax": aftertax,
                    "bonus_share_ratio": bonus,
                    "conversion_ratio": conversion,
                    "list_date": _parse_date(source.get("div_listdate")),
                    "source_ts_code": ts_code,
                }
            )
        )
    return actions


def corporate_actions_sha256(actions: Iterable[Mapping[str, Any]]) -> str:
    """Immutable identity for one batch of normalized actions (evidence binding)."""

    canonical = json.dumps(
        [dict(item) for item in actions], sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Position-lot primitives（持仓批次：取得日期/剩余数量/成本基础/可卖起点）
# ---------------------------------------------------------------------------


def position_lots(position: dict[str, Any], *, trade_date: date) -> list[dict[str, Any]]:
    """Return the position's lot list, synthesizing a legacy lot when absent.

    旧账本持仓没有批次：合成一个 ``legacy`` 批次，``acquired_at=None`` 表示
    取得日期不可知——红利税按保守最高档计（设计 §5.6 保守税费负债原则）。
    """

    lots = position.get("lots")
    if lots is not None:
        return lots
    quantity = int(position.get("quantity", 0))
    lots = []
    if quantity > 0:
        lots.append(
            {
                "lot_key": "legacy",
                "acquired_at": None,
                "sellable_from": date.min,
                "quantity": quantity,
                "cost_basis_total": quantity * float(position.get("average_cost", 0.0)),
                "origin": LOT_ORIGIN_LEGACY,
                "entitlements": [],
            }
        )
    position["lots"] = lots
    return lots


def consume_lots_fifo(
    lots: list[dict[str, Any]], quantity: int, *, trade_date: date
) -> list[tuple[dict, int]]:
    """Consume ``quantity`` sellable shares first-in-first-out; scale cost basis.

    先进先出法是财税[2012]85号规定的批次匹配规则。只有 ``sellable_from`` 已
    届的批次参与消耗——卖出委托量此前已按可卖总量钳制，锁定批次（当日买入、
    新增股份上市日前的送转股）不会被消耗。被消耗的批次按剩余比例摊薄成本
    基础；数量归零的批次从列表移除（其引用仍随返回值交给税务结算）。
    """

    remaining = int(quantity)
    consumed: list[tuple[dict, int]] = []
    ordered = sorted(
        (
            lot
            for lot in lots
            if (_parse_date(lot.get("sellable_from")) or date.min) <= trade_date
        ),
        key=lambda lot: (_parse_date(lot.get("acquired_at")) or date.min, str(lot.get("lot_key"))),
    )
    for lot in ordered:
        if remaining <= 0:
            break
        lot_quantity = int(lot.get("quantity", 0))
        if lot_quantity <= 0:
            continue
        take = min(remaining, lot_quantity)
        leftover = lot_quantity - take
        cost = float(lot.get("cost_basis_total", 0.0))
        lot["cost_basis_total"] = cost * leftover / lot_quantity if lot_quantity else 0.0
        lot["quantity"] = leftover
        consumed.append((lot, take))
        remaining -= take
    if remaining > 0:
        raise RuntimeError("lot consumption exceeds the sellable position quantity")
    lots[:] = [lot for lot in lots if int(lot.get("quantity", 0)) > 0]
    return consumed


def settle_dividend_tax(
    *,
    instrument: str,
    consumed: list[tuple[dict, int]],
    sale_date: date,
    tax_book: DividendTaxRuleBook,
) -> tuple[float, list[dict[str, Any]]]:
    """Settle differentiated dividend tax for one sale (2013+ at-sale regime).

    对本次卖出的每个批次，按该批次的股息权利（entitlements）以
    记录日定版本、取得日→卖出日持有期定档计税；同时消耗对应的
    ``untaxed_quantity``，保证同一笔股息所得只被征税一次。
    ETF/基金分配免征（财税字[1998]55号）。返回（税额, 明细）。
    """

    if is_dividend_tax_exempt(infer_cn_asset_type(instrument)):
        return 0.0, []
    total = 0.0
    details: list[dict[str, Any]] = []
    for lot, quantity in consumed:
        entitlements = sorted(
            lot.get("entitlements") or [],
            key=lambda item: (
                _parse_date(item.get("record_date")) or date.min,
                str(item.get("kind")),
            ),
        )
        # 同一股同时携带历次分红的多类权利：卖出 N 股时，每条权利各自消耗 N，
        # 现金分红与送股面值的税基互不替代。
        for entitlement in entitlements:
            untaxed = int(entitlement.get("untaxed_quantity", 0))
            if untaxed <= 0:
                continue
            take = min(quantity, untaxed)
            record_date = _parse_date(entitlement.get("record_date"))
            if record_date is None:
                raise RuntimeError("dividend entitlement is missing its record_date")
            rule = tax_book.as_of(record_date)
            rate = rate_for_holding(
                rule=rule, acquired_at=lot.get("acquired_at"), sale_date=sale_date
            )
            income = take * float(entitlement.get("income_per_share", 0.0))
            tax = income * rate
            total += tax
            if tax > 0:
                details.append(
                    {
                        "lot_key": str(lot.get("lot_key")),
                        "kind": str(entitlement.get("kind")),
                        "record_date": record_date.isoformat(),
                        "quantity": take,
                        "income": _round_money(income),
                        "rate": rate,
                        "tax": _round_money(tax),
                        "tax_rule_version": rule.version,
                    }
                )
            entitlement["untaxed_quantity"] = untaxed - take
    return _round_money(total), details


# ---------------------------------------------------------------------------
# Ex-date / pay-date application
# ---------------------------------------------------------------------------


def apply_ex_dividend(
    *,
    position: dict[str, Any],
    action: CorporateAction,
    tax_rule: DividendTaxRule,
    trade_date: date,
) -> dict[str, Any]:
    """Apply ex-date effects to one position in place; return deltas and events.

    - 现金分红：确认应收（at_sale 版本按税前额；at_payment 版本按税后额，
      税后缺失时按 税前×(1-税率) 保守折算并标记估值不确定）。
    - at_sale 且非豁免时按批次生成股息权利（现金 + 送股面值两类）。
    - 送转：按批次派生红利股批次（取得日继承父批次＝持有期连续计算的明示
      假设；``sellable_from`` 取新增股份上市日，缺失时回退除权日次一自然日
      并标记估值不确定）。总成本基础不变，单位成本随数量摊薄。
    """

    quantity = int(position.get("quantity", 0))
    if quantity <= 0:
        return {"receivable": None, "events": [], "new_shares": 0}
    instrument = action.instrument
    events: list[dict[str, Any]] = []
    valuation_uncertain = False
    exempt = is_dividend_tax_exempt(infer_cn_asset_type(instrument))

    per_share = 0.0
    if exempt or tax_rule.settlement_mode == SETTLEMENT_AT_SALE:
        per_share = action.cash_div_pretax
        if per_share <= 0 and action.cash_div_aftertax:
            # 税前额缺失：退用税后额，应收金额存在不确定性。
            per_share = float(action.cash_div_aftertax)
            valuation_uncertain = True
    elif tax_rule.settlement_mode == SETTLEMENT_AT_PAYMENT:
        per_share = float(action.cash_div_aftertax or 0.0)
        if per_share <= 0 and action.cash_div_pretax > 0:
            # 税后额缺失：按该版本单一税率保守折算。
            per_share = action.cash_div_pretax * (1.0 - tax_rule.rate_within_1m)
            valuation_uncertain = True
    if per_share < 0:
        raise ValueError(f"negative dividend per share for {instrument}")

    lots = position_lots(position, trade_date=trade_date)
    if (
        not exempt
        and tax_rule.settlement_mode == SETTLEMENT_AT_SALE
        and action.record_date is not None
    ):
        for lot in lots:
            lot_quantity = int(lot.get("quantity", 0))
            if lot_quantity <= 0:
                continue
            if action.cash_div_pretax > 0:
                lot.setdefault("entitlements", []).append(
                    {
                        "record_date": action.record_date,
                        "kind": ENTITLEMENT_KIND_CASH,
                        "income_per_share": action.cash_div_pretax,
                        "untaxed_quantity": lot_quantity,
                    }
                )
            if action.bonus_share_ratio > 0:
                lot.setdefault("entitlements", []).append(
                    {
                        "record_date": action.record_date,
                        "kind": ENTITLEMENT_KIND_BONUS_PAR,
                        "income_per_share": action.bonus_share_ratio * DIVIDEND_PAR_VALUE_CNY,
                        "untaxed_quantity": lot_quantity,
                    }
                )

    receivable: dict[str, Any] | None = None
    amount = _round_money(quantity * per_share)
    if amount > 0:
        receivable = {
            "instrument": instrument,
            "ex_date": action.ex_date,
            "record_date": action.record_date,
            "pay_date": action.pay_date,
            "quantity": quantity,
            "cash_per_share": per_share,
            "amount": amount,
            "tax_rule_version": tax_rule.version,
            "valuation_uncertain": valuation_uncertain,
            "payload_sha256": action.payload_sha256,
        }

    new_shares_total = 0
    ratio = action.share_ratio
    if ratio > 0:
        target_total = _round_half_up(quantity * ratio)
        allocations: list[int] = []
        allocated = 0
        for lot in lots:
            share = int(floor(int(lot.get("quantity", 0)) * ratio))
            allocations.append(share)
            allocated += share
        leftover = max(0, target_total - allocated)
        for index in range(len(allocations)):
            if leftover <= 0:
                break
            allocations[index] += 1
            leftover -= 1
        if action.list_date is None:
            valuation_uncertain = True
            events.append(
                {
                    "severity": "warning",
                    "event_type": "dividend_listdate_missing",
                    "instrument": instrument,
                    "reason": "bonus_shares_sellable_from_fallback",
                    "details": {
                        "event_key": (
                            f"corporate_action:listdate_missing:{instrument}:{action.ex_date}"
                        ),
                        "ex_date": action.ex_date.isoformat(),
                        "fallback_sellable_from": (action.ex_date + timedelta(days=1)).isoformat(),
                    },
                }
            )
        for lot, share in zip(lots, allocations, strict=False):
            if share <= 0:
                continue
            parent_quantity = int(lot["quantity"])
            parent_cost = float(lot.get("cost_basis_total", 0.0))
            combined = parent_quantity + share
            per_share_cost = parent_cost / combined if combined else 0.0
            sellable_from = action.list_date or (action.ex_date + timedelta(days=1))
            lots.append(
                {
                    "lot_key": f"bonus:{action.ex_date.isoformat()}:{lot['lot_key']}",
                    "acquired_at": lot.get("acquired_at"),
                    "sellable_from": sellable_from,
                    "quantity": share,
                    "cost_basis_total": per_share_cost * share,
                    "origin": LOT_ORIGIN_BONUS_SHARE,
                    "entitlements": [],
                }
            )
            lot["cost_basis_total"] = per_share_cost * parent_quantity
            new_shares_total += share
        if new_shares_total:
            position["quantity"] = quantity + new_shares_total
            total_cost = sum(float(item.get("cost_basis_total", 0.0)) for item in lots)
            position["average_cost"] = (
                total_cost / position["quantity"] if position["quantity"] else 0.0
            )

    if valuation_uncertain and receivable is not None:
        receivable["valuation_uncertain"] = True
    events.append(
        {
            "severity": "info",
            "event_type": "corporate_action_ex",
            "instrument": instrument,
            "reason": "ex_dividend_applied",
            "details": {
                "event_key": f"corporate_action:ex:{instrument}:{action.ex_date}",
                "ex_date": action.ex_date.isoformat(),
                "record_date": (
                    action.record_date.isoformat() if action.record_date else None
                ),
                "eligible_quantity": quantity,
                "cash_per_share": per_share,
                "receivable_amount": amount,
                "new_shares": new_shares_total,
                "tax_rule_version": tax_rule.version,
                "valuation_uncertain": valuation_uncertain,
            },
        }
    )
    return {"receivable": receivable, "events": events, "new_shares": new_shares_total}
