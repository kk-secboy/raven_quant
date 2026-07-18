from datetime import date

import pytest

from quant_platform.dividend_tax import (
    DIVIDEND_TAX_RULE_BOOK,
    DIVIDEND_TAX_RULE_VERSION_2005,
    DIVIDEND_TAX_RULE_VERSION_2013,
    DIVIDEND_TAX_RULE_VERSION_2015,
    DividendTaxRuleBook,
    is_dividend_tax_exempt,
    rate_for_holding,
)

pytestmark = pytest.mark.no_database


def test_rule_versions_resolve_by_record_date() -> None:
    book = DIVIDEND_TAX_RULE_BOOK
    assert book.as_of(date(2010, 6, 1)).version == DIVIDEND_TAX_RULE_VERSION_2005
    assert book.as_of(date(2012, 12, 31)).settlement_mode == "at_payment"
    assert book.as_of(date(2013, 1, 1)).version == DIVIDEND_TAX_RULE_VERSION_2013
    assert book.as_of(date(2015, 9, 7)).rate_over_1y == pytest.approx(0.05)
    assert book.as_of(date(2015, 9, 8)).version == DIVIDEND_TAX_RULE_VERSION_2015
    assert book.as_of(date(2026, 1, 1)).rate_over_1y == pytest.approx(0.0)


def test_rule_book_fails_closed_before_first_version() -> None:
    with pytest.raises(ValueError, match="no effective dividend tax rule"):
        DIVIDEND_TAX_RULE_BOOK.as_of(date(1999, 12, 31))


def test_rule_book_rejects_overlap_and_open_middle() -> None:
    rules = DIVIDEND_TAX_RULE_BOOK.rules
    overlapped = (
        rules[0],
        type(rules[1])(
            version=rules[1].version,
            effective_from="2005-06-01",
            effective_to=rules[1].effective_to,
            settlement_mode=rules[1].settlement_mode,
            rate_within_1m=rules[1].rate_within_1m,
            rate_1m_to_1y=rules[1].rate_1m_to_1y,
            rate_over_1y=rules[1].rate_over_1y,
            source=rules[1].source,
        ),
    )
    with pytest.raises(ValueError, match="must not overlap"):
        DividendTaxRuleBook(overlapped)


def test_holding_period_boundaries_are_inclusive() -> None:
    rule = DIVIDEND_TAX_RULE_BOOK.as_of(date(2024, 1, 1))
    acquired = date(2024, 5, 10)
    # 满 1 个月整仍属“1 个月以内” → 20%
    assert rate_for_holding(rule=rule, acquired_at=acquired, sale_date=date(2024, 6, 10)) == 0.20
    # 超过 1 个月 → 10%
    assert rate_for_holding(rule=rule, acquired_at=acquired, sale_date=date(2024, 6, 11)) == 0.10
    # 满 1 年整仍属“1 个月至 1 年” → 10%
    assert rate_for_holding(rule=rule, acquired_at=acquired, sale_date=date(2025, 5, 10)) == 0.10
    # 超过 1 年 → 免征（2015 版）
    assert rate_for_holding(rule=rule, acquired_at=acquired, sale_date=date(2025, 5, 11)) == 0.0


def test_holding_period_month_end_clamp() -> None:
    rule = DIVIDEND_TAX_RULE_BOOK.as_of(date(2024, 1, 1))
    # 1 月 31 日取得，2 月 29 日（闰年钳位）卖出仍满 1 个月 → 20%
    early = rate_for_holding(rule=rule, acquired_at=date(2024, 1, 31), sale_date=date(2024, 2, 29))
    later = rate_for_holding(rule=rule, acquired_at=date(2024, 1, 31), sale_date=date(2024, 3, 1))
    assert early == 0.20
    assert later == 0.10


def test_unknown_acquisition_date_uses_top_rate() -> None:
    rule = DIVIDEND_TAX_RULE_BOOK.as_of(date(2024, 1, 1))
    assert rate_for_holding(rule=rule, acquired_at=None, sale_date=date(2030, 1, 1)) == 0.20


def test_2013_version_keeps_five_percent_over_one_year() -> None:
    rule = DIVIDEND_TAX_RULE_BOOK.as_of(date(2014, 1, 1))
    rate = rate_for_holding(rule=rule, acquired_at=date(2013, 1, 1), sale_date=date(2014, 1, 2))
    assert rate == 0.05


def test_etf_dividends_are_tax_exempt() -> None:
    assert is_dividend_tax_exempt("etf")
    assert not is_dividend_tax_exempt("stock")
