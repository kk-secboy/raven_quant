"""Effective-dated dividend personal income tax (红利差别化税) rules for A-shares.

Rules are recorded per effective range with the naming announcement, mirroring
``CostScheduleBook`` in ``cost_model.py``.  The rule version applied to one
dividend entitlement is selected by the dividend's ``record_date``
(权益登记日) — a documented simplification: the 2012/2015 differentiated
policies attach to the dividend entitlement, not to the later sale.

Settlement modes follow the legislation history:

- ``at_payment``: tax is withheld when the dividend is paid, so the ledger
  books the after-tax amount and no per-lot tracking is needed.
- ``at_sale`` (财税[2012]85号 onwards): the listed company pays the full
  pre-tax dividend; the tax is settled by 中国结算 when the shares are sold,
  matched first-in-first-out by acquisition lot, at a rate depending on the
  holding period of that lot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

SETTLEMENT_AT_PAYMENT = "at_payment"
SETTLEMENT_AT_SALE = "at_sale"
_SETTLEMENT_MODES = frozenset({SETTLEMENT_AT_PAYMENT, SETTLEMENT_AT_SALE})

DIVIDEND_TAX_RULE_VERSION_2000 = "cn-dividend-tax-2000-01-01"
DIVIDEND_TAX_RULE_VERSION_2005 = "cn-dividend-tax-2005-06-13"
DIVIDEND_TAX_RULE_VERSION_2013 = "cn-dividend-tax-2013-01-01"
DIVIDEND_TAX_RULE_VERSION_2015 = "cn-dividend-tax-2015-09-08"

DIVIDEND_TAX_RULE_VERSION_CURRENT = DIVIDEND_TAX_RULE_VERSION_2015

KNOWN_DIVIDEND_TAX_RULE_VERSIONS = frozenset(
    {
        DIVIDEND_TAX_RULE_VERSION_2000,
        DIVIDEND_TAX_RULE_VERSION_2005,
        DIVIDEND_TAX_RULE_VERSION_2013,
        DIVIDEND_TAX_RULE_VERSION_2015,
    }
)


@dataclass(frozen=True)
class DividendTaxRule:
    """One effective-dated dividend tax schedule version."""

    version: str
    effective_from: str
    effective_to: str | None
    settlement_mode: str
    rate_within_1m: float
    rate_1m_to_1y: float
    rate_over_1y: float
    source: str

    def __post_init__(self) -> None:
        if self.version not in KNOWN_DIVIDEND_TAX_RULE_VERSIONS:
            raise ValueError("dividend tax rule version is obsolete")
        if self.settlement_mode not in _SETTLEMENT_MODES:
            raise ValueError("dividend tax settlement mode is invalid")
        rates = (self.rate_within_1m, self.rate_1m_to_1y, self.rate_over_1y)
        if min(rates) < 0 or max(rates) > 1:
            raise ValueError("dividend tax rates must stay within [0, 1]")
        start = date.fromisoformat(self.effective_from)
        if self.effective_to is not None and date.fromisoformat(self.effective_to) < start:
            raise ValueError("dividend tax rule effective dates are invalid")

    def covers(self, on_date: date) -> bool:
        start = date.fromisoformat(self.effective_from)
        end = date.fromisoformat(self.effective_to) if self.effective_to else None
        return on_date >= start and (end is None or on_date <= end)


DIVIDEND_TAX_RULES: tuple[DividendTaxRule, ...] = (
    DividendTaxRule(
        version=DIVIDEND_TAX_RULE_VERSION_2000,
        effective_from="2000-01-01",
        effective_to="2005-06-12",
        settlement_mode=SETTLEMENT_AT_PAYMENT,
        rate_within_1m=0.20,
        rate_1m_to_1y=0.20,
        rate_over_1y=0.20,
        source="《中华人民共和国个人所得税法》：股息红利所得适用 20% 比例税率，支付时扣缴",
    ),
    DividendTaxRule(
        version=DIVIDEND_TAX_RULE_VERSION_2005,
        effective_from="2005-06-13",
        effective_to="2012-12-31",
        settlement_mode=SETTLEMENT_AT_PAYMENT,
        rate_within_1m=0.10,
        rate_1m_to_1y=0.10,
        rate_over_1y=0.10,
        source=(
            "财税[2005]102号：2005-06-13 起股息红利所得暂减按 50% 计入应纳税所得额，"
            "适用 20% 税率，支付时扣缴（实际负担 10%）"
        ),
    ),
    DividendTaxRule(
        version=DIVIDEND_TAX_RULE_VERSION_2013,
        effective_from="2013-01-01",
        effective_to="2015-09-07",
        settlement_mode=SETTLEMENT_AT_SALE,
        rate_within_1m=0.20,
        rate_1m_to_1y=0.10,
        rate_over_1y=0.05,
        source=(
            "财税[2012]85号：2013-01-01 起持股 1 个月以内（含）20%、1 个月至 1 年（含）10%、"
            "超过 1 年 5%；派发时暂不扣缴，转让股票时按先进先出法结算"
        ),
    ),
    DividendTaxRule(
        version=DIVIDEND_TAX_RULE_VERSION_2015,
        effective_from="2015-09-08",
        effective_to=None,
        settlement_mode=SETTLEMENT_AT_SALE,
        rate_within_1m=0.20,
        rate_1m_to_1y=0.10,
        rate_over_1y=0.0,
        source=(
            "财税[2015]101号：2015-09-08 起持股超过 1 年的股息红利所得暂免征收个人所得税；"
            "1 个月以内（含）20%、1 个月至 1 年（含）10%，转让时按先进先出法结算"
        ),
    ),
)


class DividendTaxRuleBook:
    """Ordered effective-dated dividend tax schedule with fail-closed lookup."""

    def __init__(self, rules: tuple[DividendTaxRule, ...] = DIVIDEND_TAX_RULES) -> None:
        if not rules:
            raise ValueError("dividend tax rule book requires at least one version")
        ordered = tuple(sorted(rules, key=lambda item: date.fromisoformat(item.effective_from)))
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.effective_to is None:
                raise ValueError("only the latest dividend tax rule may be open-ended")
            if date.fromisoformat(previous.effective_to) >= date.fromisoformat(
                current.effective_from
            ):
                raise ValueError("dividend tax rule versions must not overlap")
        self.rules = ordered

    def as_of(self, on_date: date) -> DividendTaxRule:
        day = on_date if isinstance(on_date, date) else date.fromisoformat(str(on_date))
        for rule in self.rules:
            if rule.covers(day):
                return rule
        raise ValueError(f"no effective dividend tax rule exists for the date {day}")


DIVIDEND_TAX_RULE_BOOK = DividendTaxRuleBook()


def _add_months(day: date, months: int) -> date:
    """Calendar-month shift clamped to the target month's last day."""

    month_index = day.month - 1 + months
    year = day.year + month_index // 12
    month = month_index % 12 + 1
    if month == 12:
        last_day = date(year, 12, 31).day
    else:
        last_day = (date(year, month + 1, 1) - date(year, month, 1)).days
    return date(year, month, min(day.day, last_day))


def _add_years(day: date, years: int) -> date:
    """Calendar-year shift; 02-29 lands on 02-28 in non-leap target years."""

    try:
        return day.replace(year=day.year + years)
    except ValueError:
        return day.replace(year=day.year + years, day=28)


def rate_for_holding(
    *,
    rule: DividendTaxRule,
    acquired_at: date | None,
    sale_date: date,
) -> float:
    """Differentiated rate for one lot sold on ``sale_date``.

    持股期限含边：满 1 个月整仍属“1 个月以内”（最高档）；满 1 年整属
    “1 个月至 1 年”。``acquired_at is None`` 表示取得日期不可知（历史遗留
    持仓），按设计取保守最高档计税。
    """

    if acquired_at is None:
        return rule.rate_within_1m
    acquired = acquired_at if isinstance(acquired_at, date) else date.fromisoformat(
        str(acquired_at)
    )
    sold = sale_date if isinstance(sale_date, date) else date.fromisoformat(str(sale_date))
    if sold < acquired:
        raise ValueError("sale date precedes the lot acquisition date")
    if sold <= _add_months(acquired, 1):
        return rule.rate_within_1m
    if sold <= _add_years(acquired, 1):
        return rule.rate_1m_to_1y
    return rule.rate_over_1y


def is_dividend_tax_exempt(asset_type: str) -> bool:
    """基金/ETF 向个人投资者分配股息红利时暂不征收个人所得税（财税字[1998]55号）。"""

    return asset_type.strip().lower() == "etf"
