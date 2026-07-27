"""Pure corporate-action ledger primitives（现金分红/送转四阶段入账 + §5.6 类型扩展）.

No database, no system clock, no side effects — the same discipline as the
project ``ExecutionCore``: every function takes explicit state and returns
explicit deltas.  All rounding rules live here and are deterministic:

- 金额四舍五入到分（0.01 CNY）。
- 送转零碎股：目标新增总数按 持仓×比例 四舍五入到整股；各批次按比例 floor
  分配，余数按取得时间从早到晚逐股补齐（确定性规则，非交易所排序规则，
  属于保守近似并在事件中标明）。
- 除权日资格量 = 除权日盘前持仓量（A 股除权日恒为登记日次一交易日，
  登记收盘与除权开盘之间无法交易，二者等价）。
- at_sale 税制下除权确认应收（税前）的同时，按批次在股息权利上登记
  ``liability_per_share``＝每股应税所得 × 除权日时点持有期档位税率。持有期
  只会随时间变长、税率只降不升，因此该估计是逐批次保守上界；NAV 须减去
  未结算负债（``pending_dividend_tax_liability``），卖出结算时只确认实际
  税额与已提负债的差额（多提冲回），NAV 在除权日不虚高、卖出日不跳变
  （设计 §5.6 保守税费负债原则）。
- 非分红类公司行动（公告/名称变更/拆并股/代码变更/ETF 折算/持有人选择/
  unsupported）统一走 ``CorporateEvent`` 信封：公告只产生信息事件不改账；
  拆并股只调整经济数量与单位成本（总成本不变、不产生现金）；代码变更只
  迁移证券身份；unsupported 类型 fail-closed 记原因，永不自动入账。每类
  事件携带唯一事件键，账户内只应用一次（详见文件尾部类型扩展区）。

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
) -> tuple[float, list[dict[str, Any]], float]:
    """Settle differentiated dividend tax for one sale (2013+ at-sale regime).

    对本次卖出的每个批次，按该批次的股息权利（entitlements）以
    记录日定版本、取得日→卖出日持有期定档计税；同时消耗对应的
    ``untaxed_quantity``，保证同一笔股息所得只被征税一次。
    ETF/基金分配免征（财税字[1998]55号）。

    返回（税额, 明细, 已提负债释放额）。释放额按同一消耗量
    ``take × liability_per_share`` 计算——除权日按保守档位计提的负债随
    批次消耗同比例释放，卖出对 NAV 的净影响只有 实际税额 − 释放额
    （差额确认）；旧账本没有 ``liability_per_share`` 的权利释放为零，
    差额即全额税额（少提方向，向后兼容）。
    """

    if is_dividend_tax_exempt(infer_cn_asset_type(instrument)):
        return 0.0, [], 0.0
    total = 0.0
    released = 0.0
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
            released += take * float(entitlement.get("liability_per_share", 0.0) or 0.0)
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
                        "liability_released": _round_money(
                            take * float(entitlement.get("liability_per_share", 0.0) or 0.0)
                        ),
                        "tax_rule_version": rule.version,
                    }
                )
            entitlement["untaxed_quantity"] = untaxed - take
    return _round_money(total), details, _round_money(released)


def pending_dividend_tax_liability(positions: Mapping[str, Any]) -> float:
    """未结算股息税负债总额：逐批次 未税数量 × 除权日计提的每股负债。

    负债随 ``settle_dividend_tax`` 消耗 ``untaxed_quantity`` 同比例自动释放，
    不需要独立状态机；NAV = 现金 + 市值 + 应收 − 本函数结果。没有
    ``liability_per_share`` 的旧账本权利贡献零（差额在卖出时全额确认）。
    """

    total = 0.0
    for position in positions.values():
        for lot in position.get("lots") or []:
            for entitlement in lot.get("entitlements") or []:
                total += int(entitlement.get("untaxed_quantity", 0)) * float(
                    entitlement.get("liability_per_share", 0.0) or 0.0
                )
    return _round_money(total)


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
    - at_sale 且非豁免时按批次生成股息权利（现金 + 送股面值两类），并按
      除权日时点持有期档位（逐批次保守上界，持有期只增税率只降）在权利上
      登记 ``liability_per_share`` 计提应付税负债；返回 ``tax_liability``
      为本次除权计提总额，NAV 须将其作为减项。
    - 送转：按批次派生红利股批次（取得日继承父批次＝持有期连续计算的明示
      假设；``sellable_from`` 取新增股份上市日，缺失时回退除权日次一自然日
      并标记估值不确定）。总成本基础不变，单位成本随数量摊薄。
    """

    quantity = int(position.get("quantity", 0))
    if quantity <= 0:
        return {"receivable": None, "events": [], "new_shares": 0, "tax_liability": 0.0}
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
    tax_liability = 0.0
    if (
        not exempt
        and tax_rule.settlement_mode == SETTLEMENT_AT_SALE
        and action.record_date is not None
    ):
        for lot in lots:
            lot_quantity = int(lot.get("quantity", 0))
            if lot_quantity <= 0:
                continue
            # 保守计提：以除权日时点的持有期定档——卖出日只会更晚、税率只降
            # 不升，该档即本批次可能适用的最高税率（取得日不可知时为最高档）。
            liability_rate = rate_for_holding(
                rule=tax_rule,
                acquired_at=lot.get("acquired_at"),
                sale_date=trade_date,
            )
            if action.cash_div_pretax > 0:
                liability_per_share = action.cash_div_pretax * liability_rate
                lot.setdefault("entitlements", []).append(
                    {
                        "record_date": action.record_date,
                        "kind": ENTITLEMENT_KIND_CASH,
                        "income_per_share": action.cash_div_pretax,
                        "untaxed_quantity": lot_quantity,
                        "liability_per_share": liability_per_share,
                    }
                )
                tax_liability += lot_quantity * liability_per_share
            if action.bonus_share_ratio > 0:
                liability_per_share = action.bonus_share_ratio * DIVIDEND_PAR_VALUE_CNY * (
                    liability_rate
                )
                lot.setdefault("entitlements", []).append(
                    {
                        "record_date": action.record_date,
                        "kind": ENTITLEMENT_KIND_BONUS_PAR,
                        "income_per_share": action.bonus_share_ratio * DIVIDEND_PAR_VALUE_CNY,
                        "untaxed_quantity": lot_quantity,
                        "liability_per_share": liability_per_share,
                    }
                )
                tax_liability += lot_quantity * liability_per_share
    tax_liability = _round_money(tax_liability)

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
                "tax_liability": tax_liability,
                "tax_rule_version": tax_rule.version,
                "valuation_uncertain": valuation_uncertain,
            },
        }
    )
    return {
        "receivable": receivable,
        "events": events,
        "new_shares": new_shares_total,
        "tax_liability": tax_liability,
    }


# ---------------------------------------------------------------------------
# Extended corporate-event types（设计 §5.6 类型扩展）
#
# 除现金分红/送转外的公司行动统一走 ``CorporateEvent`` 信封：每类事件携带
# 唯一事件键与 ``effective_date``，公告类只产生信息事件不改账，拆并股/代码
# 变更在经济生效日调整账本，unsupported 类型 fail-closed 记原因。数据可得性
# 结论（Tushare 已下载数据集，见 docs/design-gap-analysis.md 阶段 4/9）：
#
# - 公告阶段：dividend 表预案行（无 ex_date）→ ``normalize_announcement_rows``。
# - 拆并股 / ETF 份额折算：无专表，经济效应体现在 adj_factor / fund_adj 跳变
#   → ``detect_split_events`` 推断；也接受显式录入行。
# - 代码/名称变更：namechange 表 → ``normalize_namechange_rows``（含补充
#   ``new_ts_code`` 字段时产生代码映射，否则仅名称信息事件）。
# - 需持有人选择：anns_d 公告标题关键词 → ``detect_choice_required_events``；
#   也接受显式录入。
# - 配股/换股/基金清盘：已下载数据无可支撑数据源（配股无表、换股无表、
#   清盘无回收金额），有据跳过 → ``UNSUPPORTED_EVENT_TYPES`` fail-closed。
# ---------------------------------------------------------------------------

EVENT_KIND_ANNOUNCEMENT = "announcement"
EVENT_KIND_NAME_CHANGE = "name_change"
EVENT_KIND_SPLIT = "split"
EVENT_KIND_REVERSE_SPLIT = "reverse_split"
EVENT_KIND_CODE_CHANGE = "code_change"
EVENT_KIND_CHOICE_REQUIRED = "choice_required"
EVENT_KIND_UNSUPPORTED = "unsupported"

INFORMATIONAL_EVENT_KINDS = (
    EVENT_KIND_ANNOUNCEMENT,
    EVENT_KIND_NAME_CHANGE,
    EVENT_KIND_CHOICE_REQUIRED,
    EVENT_KIND_UNSUPPORTED,
)
ECONOMIC_EVENT_KINDS = (
    EVENT_KIND_SPLIT,
    EVENT_KIND_REVERSE_SPLIT,
    EVENT_KIND_CODE_CHANGE,
)

UNSUPPORTED_EVENT_TYPES = ("rights_issue", "share_exchange", "fund_liquidation")
"""已下载数据无可支撑数据源的公司行动类型（配股/换股/基金清盘）：

- 配股（rights_issue）：Tushare 已下载数据集无配股发行表，无法确定配股
  价格、比例与缴款期，按 goal 规则有据跳过；公告经 ``choice_required``
  提醒人工处理，账本拒绝自动入账。
- 换股（share_exchange）：无换股要约结构化工单数据，同上。
- 基金清盘（fund_liquidation）：fund_basic 只有摘牌日期、无清算回收金额
  与支付日，无法在不假设现金结算的前提下入账（设计 §5.6 禁止统一假设
  现金结算），fail-closed 留待人工确认回收事件。
"""

CHOICE_REQUIRED_KEYWORDS = ("配股", "换股", "要约", "现金选择权", "吸收合并")
"""anns_d 公告标题中提示需要持有人选择的关键词（仅提醒，永不代客选择）。"""

LIQUIDATION_KEYWORDS = ("清盘", "终止上市", "基金合同终止")
"""anns_d 公告标题中提示基金清盘的关键词（unsupported fail-closed 路径）。"""

_SPLIT_JUMP_THRESHOLD = 1.3
"""adj_factor/fund_adj 单日跳变超过该倍数视为拆并股/折算候选（送转由
dividend 表行解释，不落入本检测）。"""


@dataclass(frozen=True)
class CorporateEvent:
    """One normalized non-dividend corporate event with a unique event key.

    ``effective_date`` 是各阶段各自的生效日（公告日/除权日/变更生效日），
    与 ``CorporateAction.ex_date`` 同样参与幂等；``event_key`` 全局唯一，
    账户侧只应用一次（数据库唯一约束兜底）。
    """

    kind: str
    instrument: str
    effective_date: date
    event_key: str
    payload_sha256: str
    stage: str | None = None
    split_ratio: float | None = None
    new_instrument: str | None = None
    title: str | None = None
    unsupported_type: str | None = None
    source: str = ""
    details: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CorporateEvent:
        source = dict(values)
        kind = str(source.get("kind") or "").strip()
        known = INFORMATIONAL_EVENT_KINDS + ECONOMIC_EVENT_KINDS
        if kind not in known:
            # fail-closed：无法识别的事件类型在入口拒绝，绝不静默忽略。
            raise ValueError(f"unknown corporate event kind: {source!r}")
        instrument = str(source.get("instrument") or "").strip().upper()
        if not instrument:
            raise ValueError(f"corporate event requires an instrument: {source!r}")
        effective_date = _parse_date(
            source.get("effective_date")
            or source.get("ex_date")
            or source.get("ann_date")
        )
        if effective_date is None:
            raise ValueError(f"corporate event requires an effective date: {source!r}")
        split_ratio = None
        if kind in (EVENT_KIND_SPLIT, EVENT_KIND_REVERSE_SPLIT):
            split_ratio = float(source.get("split_ratio") or 0.0)
            if not (0.001 <= split_ratio <= 1000.0) or abs(split_ratio - 1.0) < 1e-9:
                raise ValueError(f"split ratio out of range: {source!r}")
            expected = EVENT_KIND_SPLIT if split_ratio > 1.0 else EVENT_KIND_REVERSE_SPLIT
            if kind != expected:
                raise ValueError(f"split kind does not match its ratio: {source!r}")
        new_instrument = None
        if kind == EVENT_KIND_CODE_CHANGE:
            new_instrument = str(source.get("new_instrument") or "").strip().upper()
            if not new_instrument or new_instrument == instrument:
                raise ValueError(f"code change requires a distinct new code: {source!r}")
        unsupported_type = None
        if kind == EVENT_KIND_UNSUPPORTED:
            unsupported_type = str(source.get("unsupported_type") or "").strip()
            if unsupported_type not in UNSUPPORTED_EVENT_TYPES:
                raise ValueError(f"unregistered unsupported event type: {source!r}")
        payload = {
            key: source.get(key)
            for key in (
                "kind",
                "instrument",
                "effective_date",
                "stage",
                "split_ratio",
                "new_instrument",
                "title",
                "unsupported_type",
                "source",
            )
        }
        payload_sha256 = str(source.get("payload_sha256") or _action_payload(payload))
        event_key = str(source.get("event_key") or "").strip() or _corporate_event_key(
            kind=kind,
            instrument=instrument,
            effective_date=effective_date,
            new_instrument=new_instrument,
            unsupported_type=unsupported_type,
            payload_sha256=payload_sha256,
        )
        return cls(
            kind=kind,
            instrument=instrument,
            effective_date=effective_date,
            event_key=event_key,
            payload_sha256=payload_sha256,
            stage=(str(source["stage"]) if source.get("stage") is not None else None),
            split_ratio=split_ratio,
            new_instrument=new_instrument,
            title=(str(source["title"]) if source.get("title") is not None else None),
            unsupported_type=unsupported_type,
            source=str(source.get("source") or ""),
            details=dict(source.get("details") or {}),
        )


def _corporate_event_key(
    *,
    kind: str,
    instrument: str,
    effective_date: date,
    new_instrument: str | None,
    unsupported_type: str | None,
    payload_sha256: str,
) -> str:
    if kind == EVENT_KIND_CODE_CHANGE:
        return (
            f"corporate_action:code_change:{instrument}:{new_instrument}:{effective_date}"
        )
    if kind == EVENT_KIND_UNSUPPORTED:
        return (
            f"corporate_action:unsupported:{unsupported_type}:{instrument}:{effective_date}"
        )
    if kind == EVENT_KIND_CHOICE_REQUIRED:
        # 同日可能有多次要约：带载荷短哈希区分，重放同一数据键不变。
        return (
            f"corporate_action:choice_required:{instrument}:{effective_date}:"
            f"{payload_sha256[:8]}"
        )
    return f"corporate_action:{kind}:{instrument}:{effective_date}"


def normalize_announcement_rows(rows: Iterable[Mapping[str, Any]]) -> list[CorporateEvent]:
    """Dividend-plan rows → informational announcement events (公告阶段不改账).

    无 ``ex_date`` 的预案行此前被直接丢弃；此处转为 ``plan`` 阶段信息事件。
    已给出除权日的实施公告行同样产生 ``implementation`` 阶段事件，并在
    details 中携带 ``linked_ex_event_key`` 与后续除权入账事件关联。
    """

    events: list[CorporateEvent] = []
    for row in rows:
        source = dict(row)
        ann_date = _parse_date(source.get("ann_date"))
        if ann_date is None:
            continue
        ts_code = str(source.get("ts_code") or "")
        instrument = ts_code_to_instrument(ts_code)
        ex_date = _parse_date(source.get("ex_date"))
        stage = "implementation" if ex_date is not None else "plan"
        details: dict[str, Any] = {
            "ann_date": ann_date.isoformat(),
            "stage": stage,
            "cash_div_pretax": float(source.get("cash_div_tax") or 0.0),
            "bonus_share_ratio": float(source.get("stk_bo_rate") or 0.0),
            "conversion_ratio": float(source.get("stk_co_rate") or 0.0),
            "progress": str(source.get("div_proc") or ""),
            # 与后续除权事件的唯一键关联（除权事件键由引擎在除权日产生）。
            "related_event_key_prefix": f"corporate_action:ex:{instrument}:",
        }
        if ex_date is not None:
            details["linked_ex_event_key"] = f"corporate_action:ex:{instrument}:{ex_date}"
        events.append(
            CorporateEvent.from_mapping(
                {
                    "kind": EVENT_KIND_ANNOUNCEMENT,
                    "instrument": instrument,
                    "effective_date": ann_date,
                    "stage": stage,
                    "source": "tushare_dividend",
                    "details": details,
                }
            )
        )
    return events


def normalize_namechange_rows(rows: Iterable[Mapping[str, Any]]) -> list[CorporateEvent]:
    """Tushare ``namechange`` 行 → 名称/代码变更事件。

    行带补充字段 ``new_ts_code``（人工或外部源录入）时产生代码变更事件
    （旧代码持仓映射到新代码）；否则仅产生名称变更信息事件，不动账本。
    """

    events: list[CorporateEvent] = []
    for row in rows:
        source = dict(row)
        effective = _parse_date(source.get("start_date"))
        if effective is None:
            continue
        ts_code = str(source.get("ts_code") or "")
        instrument = ts_code_to_instrument(ts_code)
        new_ts_code = str(source.get("new_ts_code") or "").strip()
        ann_date = _parse_date(source.get("ann_date"))
        details: dict[str, Any] = {
            "ann_date": ann_date.isoformat() if ann_date else None,
            "new_name": str(source.get("name") or ""),
            "change_reason": str(source.get("change_reason") or ""),
        }
        if new_ts_code:
            events.append(
                CorporateEvent.from_mapping(
                    {
                        "kind": EVENT_KIND_CODE_CHANGE,
                        "instrument": instrument,
                        "new_instrument": ts_code_to_instrument(new_ts_code),
                        "effective_date": effective,
                        "source": "tushare_namechange",
                        "details": details,
                    }
                )
            )
        else:
            events.append(
                CorporateEvent.from_mapping(
                    {
                        "kind": EVENT_KIND_NAME_CHANGE,
                        "instrument": instrument,
                        "effective_date": effective,
                        "title": str(source.get("name") or ""),
                        "source": "tushare_namechange",
                        "details": details,
                    }
                )
            )
    return events


def detect_split_events(
    adj_rows: Iterable[Mapping[str, Any]],
    *,
    known_ex_dates: Mapping[str, Iterable[Any]] | None = None,
) -> list[CorporateEvent]:
    """adj_factor / fund_adj 跳变 → 拆并股 / ETF 份额折算候选事件。

    单日复权因子比值 ≥ 阈值（或 ≤ 其倒数）且当日无 dividend 行解释（送转
    已由除权路径入账）时判定为拆股（比值>1）或并股/折算（比值<1）。推断
    比例在 details 中标注 ``detection=adj_factor_jump``；显式录入的行可
    不经本函数直接构造 ``CorporateEvent``。
    """

    known: dict[str, set[date]] = {
        str(instrument).upper(): {
            parsed for value in values if (parsed := _parse_date(value)) is not None
        }
        for instrument, values in (known_ex_dates or {}).items()
    }
    by_instrument: dict[str, list[tuple[date, float]]] = {}
    for row in adj_rows:
        trade_date = _parse_date(row.get("trade_date"))
        factor_raw = row.get("adj_factor")
        if trade_date is None or factor_raw is None:
            continue
        factor = float(factor_raw)
        if factor <= 0:
            continue
        instrument = ts_code_to_instrument(str(row.get("ts_code") or ""))
        by_instrument.setdefault(instrument, []).append((trade_date, factor))
    events: list[CorporateEvent] = []
    for instrument, series in sorted(by_instrument.items()):
        series.sort(key=lambda item: item[0])
        for (prev_date, prev_factor), (trade_date, factor) in zip(
            series, series[1:], strict=False
        ):
            ratio = factor / prev_factor
            if ratio >= _SPLIT_JUMP_THRESHOLD:
                kind = EVENT_KIND_SPLIT
            elif ratio <= 1.0 / _SPLIT_JUMP_THRESHOLD:
                kind = EVENT_KIND_REVERSE_SPLIT
            else:
                continue
            if trade_date in known.get(instrument, set()):
                continue  # 当日除权（送转/分红）已解释因子跳变。
            snapped = round(ratio, 4)
            if abs(snapped - 1.0) < 1e-9:
                continue
            events.append(
                CorporateEvent.from_mapping(
                    {
                        "kind": kind,
                        "instrument": instrument,
                        "effective_date": trade_date,
                        "split_ratio": snapped,
                        "source": "adj_factor",
                        "details": {
                            "detection": "adj_factor_jump",
                            "prev_trade_date": prev_date.isoformat(),
                            "prev_factor": prev_factor,
                            "factor": factor,
                            "raw_ratio": ratio,
                        },
                    }
                )
            )
    return events


def detect_choice_required_events(
    anns_rows: Iterable[Mapping[str, Any]],
) -> list[CorporateEvent]:
    """anns_d 公告标题关键词 → 需持有人选择的提醒事件（只提醒，不改账）。"""

    events: list[CorporateEvent] = []
    for row in anns_rows:
        title = str(row.get("title") or "")
        if not any(keyword in title for keyword in CHOICE_REQUIRED_KEYWORDS):
            continue
        ann_date = _parse_date(row.get("ann_date"))
        if ann_date is None:
            continue
        matched = [keyword for keyword in CHOICE_REQUIRED_KEYWORDS if keyword in title]
        events.append(
            CorporateEvent.from_mapping(
                {
                    "kind": EVENT_KIND_CHOICE_REQUIRED,
                    "instrument": ts_code_to_instrument(str(row.get("ts_code") or "")),
                    "effective_date": ann_date,
                    "title": title,
                    "source": "tushare_anns_d",
                    "details": {"matched_keywords": matched, "ann_date": ann_date.isoformat()},
                }
            )
        )
    return events


def detect_liquidation_events(anns_rows: Iterable[Mapping[str, Any]]) -> list[CorporateEvent]:
    """anns_d 公告标题关键词 → 基金清盘 unsupported 事件（fail-closed 留痕）。"""

    events: list[CorporateEvent] = []
    for row in anns_rows:
        title = str(row.get("title") or "")
        if not any(keyword in title for keyword in LIQUIDATION_KEYWORDS):
            continue
        ann_date = _parse_date(row.get("ann_date"))
        if ann_date is None:
            continue
        events.append(
            CorporateEvent.from_mapping(
                {
                    "kind": EVENT_KIND_UNSUPPORTED,
                    "instrument": ts_code_to_instrument(str(row.get("ts_code") or "")),
                    "effective_date": ann_date,
                    "unsupported_type": "fund_liquidation",
                    "title": title,
                    "source": "tushare_anns_d",
                    "details": {
                        "skip_reason": "no_liquidation_proceeds_data",
                        "ann_date": ann_date.isoformat(),
                    },
                }
            )
        )
    return events


def apply_share_split(
    *,
    position: dict[str, Any],
    event: CorporateEvent,
    trade_date: date,
) -> dict[str, Any]:
    """Apply a split / reverse-split / ETF share conversion on its effective date.

    只调整经济数量与单位成本：总成本基础不变、不产生现金、不动 NAV 口径
    （市值随除权价机械变化由估值层处理，单位成本同步摊薄避免虚假亏损）。
    批次数量按比例 floor 分配、余数按取得时间从早到晚逐股补齐（与送转
    同一确定性规则）；并股归零批次的成本并入最早存活批次，不确认损益。
    """

    if event.split_ratio is None:
        raise ValueError("share split requires a split ratio")
    quantity = int(position.get("quantity", 0))
    if quantity <= 0:
        return {"events": [], "quantity_before": 0, "quantity_after": 0}
    ratio = float(event.split_ratio)
    target_total = _round_half_up(quantity * ratio)
    if target_total < 1:
        # 并股后不足 1 股：不臆造份额、不假设现金结算（设计 §5.6），
        # 持仓保持原样并标记估值不确定，留待人工处理。
        return {
            "events": [
                {
                    "severity": "critical",
                    "event_type": "corporate_action_split_dust",
                    "instrument": event.instrument,
                    "reason": "reverse_split_below_one_share",
                    "details": {
                        "event_key": event.event_key,
                        "quantity": quantity,
                        "split_ratio": ratio,
                        "valuation_uncertain": True,
                    },
                }
            ],
            "quantity_before": quantity,
            "quantity_after": quantity,
        }
    lots = position_lots(position, trade_date=trade_date)
    ordered = sorted(
        lots,
        key=lambda lot: (_parse_date(lot.get("acquired_at")) or date.min, str(lot.get("lot_key"))),
    )
    allocations: dict[int, int] = {}
    allocated = 0
    for index, lot in enumerate(ordered):
        share = int(floor(int(lot.get("quantity", 0)) * ratio))
        allocations[index] = share
        allocated += share
    leftover = max(0, target_total - allocated)
    for index in range(len(ordered)):
        if leftover <= 0:
            break
        allocations[index] += 1
        leftover -= 1
    merged_cost = 0.0
    surviving: list[dict[str, Any]] = []
    for index, lot in enumerate(ordered):
        new_quantity = allocations[index]
        if new_quantity <= 0:
            merged_cost += float(lot.get("cost_basis_total", 0.0))
            continue
        lot["quantity"] = new_quantity
        surviving.append(lot)
    if merged_cost and surviving:
        surviving[0]["cost_basis_total"] = float(surviving[0].get("cost_basis_total", 0.0)) + (
            merged_cost
        )
    total_cost = sum(float(lot.get("cost_basis_total", 0.0)) for lot in surviving)
    position["lots"] = surviving
    position["quantity"] = target_total
    position["available_quantity"] = min(
        int(position.get("available_quantity", 0)), target_total
    )
    position["average_cost"] = total_cost / target_total if target_total else 0.0
    return {
        "events": [
            {
                "severity": "info",
                "event_type": f"corporate_action_{event.kind}",
                "instrument": event.instrument,
                "reason": "share_split_applied",
                "details": {
                    "event_key": event.event_key,
                    "effective_date": event.effective_date.isoformat(),
                    "split_ratio": ratio,
                    "quantity_before": quantity,
                    "quantity_after": target_total,
                    "cost_basis_total": _round_money(total_cost),
                    "average_cost_after": position["average_cost"],
                    **dict(event.details or {}),
                },
            }
        ],
        "quantity_before": quantity,
        "quantity_after": target_total,
    }


def apply_code_change(
    *,
    positions: dict[str, dict[str, Any]],
    event: CorporateEvent,
    trade_date: date,
) -> dict[str, Any]:
    """Map a position from the old instrument code to the new one (证券身份变更).

    数量、批次、成本基础原样迁移，不产生现金或经济损益；新旧代码同时有
    持仓时 fail-closed 抛错（需要人工合并，不静默并账）。
    """

    if not event.new_instrument:
        raise ValueError("code change requires a new instrument")
    old = event.instrument
    new = event.new_instrument
    position = positions.get(old)
    moved = False
    if position is not None and int(position.get("quantity", 0)) > 0:
        existing = positions.get(new)
        if existing is not None and int(existing.get("quantity", 0)) > 0:
            raise RuntimeError(
                f"code change would merge two live positions: {old} and {new}"
            )
        positions[new] = position
        del positions[old]
        moved = True
    return {
        "moved": moved,
        "events": [
            {
                "severity": "info",
                "event_type": "corporate_action_code_change",
                "instrument": new,
                "reason": "instrument_code_mapped" if moved else "code_change_without_position",
                "details": {
                    "event_key": event.event_key,
                    "effective_date": event.effective_date.isoformat(),
                    "old_instrument": old,
                    "new_instrument": new,
                    "position_moved": moved,
                    **dict(event.details or {}),
                },
            }
        ],
    }
