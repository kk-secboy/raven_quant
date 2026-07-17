from datetime import date

import pytest

from quant_platform.market_rules import (
    BOARD_BSE,
    BOARD_STAR,
    is_valid_order_quantity,
    lot_floor,
    order_unit_rules,
    validate_order_quantity,
)

pytestmark = pytest.mark.no_database

TRADE_DAY = date(2026, 7, 13)


def test_star_board_requires_two_hundred_share_minimum() -> None:
    rules = order_unit_rules("688981.SH", TRADE_DAY)
    assert rules.board == BOARD_STAR
    assert rules.min_lot == 200
    assert rules.lot_increment == 1

    violations = validate_order_quantity("688981.SH", 150, side="buy", trade_date=TRADE_DAY)
    assert violations and "minimum" in violations[0]

    assert validate_order_quantity("688981.SH", 200, side="buy", trade_date=TRADE_DAY) == []
    # Above the 200-share minimum, single-share increments are valid.
    assert validate_order_quantity("688981.SH", 250, side="buy", trade_date=TRADE_DAY) == []


def test_main_board_buy_must_be_hundred_share_multiples() -> None:
    rules = order_unit_rules("600519.SH", TRADE_DAY)
    assert rules.min_lot == 100
    assert rules.lot_increment == 100

    assert validate_order_quantity("600519.SH", 150, side="buy", trade_date=TRADE_DAY)
    assert validate_order_quantity("600519.SH", 200, side="buy", trade_date=TRADE_DAY) == []

    sz_rules = order_unit_rules("000001.SZ", TRADE_DAY)
    assert (sz_rules.min_lot, sz_rules.lot_increment) == (100, 100)


def test_bse_allows_single_share_increments_above_one_hundred() -> None:
    rules = order_unit_rules("920001.BJ", TRADE_DAY)
    assert rules.board == BOARD_BSE
    assert validate_order_quantity("920001.BJ", 150, side="buy", trade_date=TRADE_DAY) == []
    assert validate_order_quantity("920001.BJ", 99, side="buy", trade_date=TRADE_DAY)


def test_odd_lot_sells_are_allowed_and_bounded_by_position() -> None:
    assert (
        validate_order_quantity(
            "600519.SH", 37, side="sell", trade_date=TRADE_DAY, held_quantity=100
        )
        == []
    )
    violations = validate_order_quantity(
        "600519.SH", 137, side="sell", trade_date=TRADE_DAY, held_quantity=100
    )
    assert violations and "exceeds" in violations[0]


def test_star_board_is_fail_closed_before_inception() -> None:
    with pytest.raises(ValueError, match="no order-unit rule covers"):
        order_unit_rules("688981.SH", date(2019, 7, 19))


def test_unknown_board_prefix_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="no order-unit rule covers"):
        order_unit_rules("700001.SH", TRADE_DAY)


def test_lot_floor_rounds_down_to_valid_quantities() -> None:
    star = order_unit_rules("688981.SH", TRADE_DAY)
    assert lot_floor(199, star) == 0
    assert lot_floor(250, star) == 250
    main = order_unit_rules("600519.SH", TRADE_DAY)
    assert lot_floor(250, main) == 200
    assert is_valid_order_quantity(200, main)
    assert not is_valid_order_quantity(150, main)


def test_etf_defaults_to_conservative_t_plus_one_hundred_lots() -> None:
    rules = order_unit_rules("510300.SH", TRADE_DAY)
    assert rules.t_plus == 1
    assert (rules.min_lot, rules.lot_increment) == (100, 100)
