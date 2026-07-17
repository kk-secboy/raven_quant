from __future__ import annotations

from dataclasses import dataclass
from datetime import date

ORDER_UNIT_RULE_VERSION = "cn-order-unit-v1"

# Board identifiers used by OrderUnitRules.board.
BOARD_STAR = "star"
BOARD_BSE = "bse"
BOARD_CHINEXT = "chinext"
BOARD_SH_MAIN = "sh_main"
BOARD_SZ_MAIN = "sz_main"
BOARD_FUND = "fund"


@dataclass(frozen=True)
class OrderUnitRules:
    """Exchange order-unit (申报单位) rules for one instrument at one date."""

    board: str
    min_lot: int
    lot_increment: int
    t_plus: int
    rule_version: str
    source: str


@dataclass(frozen=True)
class _OrderUnitRule:
    """One effective-dated board rule entry.

    Entries are matched in order and the first prefix/code match wins, so a
    future T+0 whitelist for specific fund subtypes can be prepended with
    ``codes`` entries (t_plus=0) ahead of the generic fund rule.  No T+0
    entry is enabled today.
    """

    board: str
    effective_from: str
    effective_to: str | None
    min_lot: int
    lot_increment: int
    t_plus: int
    source: str
    prefixes: tuple[str, ...] = ()
    codes: frozenset[str] = frozenset()


_ORDER_UNIT_RULES: tuple[_OrderUnitRule, ...] = (
    _OrderUnitRule(
        board=BOARD_STAR,
        effective_from="2019-07-22",
        effective_to=None,
        min_lot=200,
        lot_increment=1,
        t_plus=1,
        prefixes=("688", "689"),
        source=(
            "《上海证券交易所科创板股票交易特别规定》：2019-07-22 起单笔申报"
            "最低 200 股，超过 200 股的部分以 1 股递增"
        ),
    ),
    _OrderUnitRule(
        board=BOARD_BSE,
        effective_from="2021-11-15",
        effective_to=None,
        min_lot=100,
        lot_increment=1,
        t_plus=1,
        prefixes=("4", "8", "920"),
        source=(
            "《北京证券交易所交易规则（试行）》：2021-11-15 起单笔申报"
            "最低 100 股，超过 100 股的部分以 1 股递增"
        ),
    ),
    _OrderUnitRule(
        board=BOARD_CHINEXT,
        effective_from="2000-01-01",
        effective_to=None,
        min_lot=100,
        lot_increment=100,
        t_plus=1,
        prefixes=("300", "301"),
        source="创业板交易规则：买入申报 100 股整数倍、最低 100 股",
    ),
    _OrderUnitRule(
        board=BOARD_SH_MAIN,
        effective_from="2000-01-01",
        effective_to=None,
        min_lot=100,
        lot_increment=100,
        t_plus=1,
        prefixes=("60",),
        source="沪市主板交易规则：买入申报 100 股整数倍、最低 100 股",
    ),
    _OrderUnitRule(
        board=BOARD_SZ_MAIN,
        effective_from="2000-01-01",
        effective_to=None,
        min_lot=100,
        lot_increment=100,
        t_plus=1,
        prefixes=("000", "001", "002", "003"),
        source="深市主板交易规则：买入申报 100 股整数倍、最低 100 股",
    ),
    _OrderUnitRule(
        board=BOARD_FUND,
        effective_from="2000-01-01",
        effective_to=None,
        min_lot=100,
        lot_increment=100,
        t_plus=1,
        prefixes=("5", "15", "16", "18"),
        source=(
            "基金/ETF 交易惯例：申报单位 100 份、T+1 保守处理"
            "（T+0 子类型白名单条目预留，本次不启用）"
        ),
    ),
)


def _instrument_digits(instrument: str) -> str:
    value = str(instrument).upper()
    digits = "".join(character for character in value if character.isdigit())[-6:]
    if len(digits) != 6:
        raise ValueError(f"cannot resolve order-unit rules for instrument: {instrument}")
    return digits


def order_unit_rules(instrument: str, trade_date: date) -> OrderUnitRules:
    """Resolve the order-unit rules for one instrument at one date (fail-closed)."""

    digits = _instrument_digits(instrument)
    day = trade_date if isinstance(trade_date, date) else date.fromisoformat(str(trade_date))
    for rule in _ORDER_UNIT_RULES:
        if rule.codes and digits not in rule.codes:
            continue
        if not rule.codes and not digits.startswith(rule.prefixes):
            continue
        start = date.fromisoformat(rule.effective_from)
        end = date.fromisoformat(rule.effective_to) if rule.effective_to else None
        if day < start or (end is not None and day > end):
            raise ValueError(
                f"no order-unit rule covers {instrument} on {day}: "
                f"{rule.board} rules start {rule.effective_from}"
            )
        return OrderUnitRules(
            board=rule.board,
            min_lot=rule.min_lot,
            lot_increment=rule.lot_increment,
            t_plus=rule.t_plus,
            rule_version=ORDER_UNIT_RULE_VERSION,
            source=rule.source,
        )
    raise ValueError(f"no order-unit rule covers {instrument} on {day}")


def lot_floor(quantity: int, rules: OrderUnitRules) -> int:
    """Largest valid buy quantity not exceeding ``quantity`` (0 below min_lot)."""

    if quantity < rules.min_lot:
        return 0
    return rules.min_lot + (quantity - rules.min_lot) // rules.lot_increment * rules.lot_increment


def is_valid_order_quantity(quantity: int, rules: OrderUnitRules) -> bool:
    return quantity >= rules.min_lot and (quantity - rules.min_lot) % rules.lot_increment == 0


def validate_order_quantity(
    instrument: str,
    quantity: float,
    *,
    side: str,
    trade_date: date,
    held_quantity: int = 0,
) -> list[str]:
    """Return order-unit violations for one order (empty list means valid)."""

    rules = order_unit_rules(instrument, trade_date)
    normalized_side = side.strip().lower()
    if normalized_side not in {"buy", "sell"}:
        raise ValueError("order side must be buy or sell")
    integer_quantity = int(quantity)
    if quantity != integer_quantity or integer_quantity <= 0:
        return [f"{instrument}: order quantity must be a positive integer"]
    if normalized_side == "buy":
        if integer_quantity < rules.min_lot:
            return [
                f"{instrument}: buy quantity {integer_quantity} is below the "
                f"{rules.board} minimum of {rules.min_lot} shares"
            ]
        if (integer_quantity - rules.min_lot) % rules.lot_increment:
            return [
                f"{instrument}: buy quantity {integer_quantity} must step by "
                f"{rules.lot_increment} above {rules.min_lot} on {rules.board}"
            ]
        return []
    # A-share odd-lot sells are allowed; the quantity is bounded by the position.
    if integer_quantity > held_quantity:
        return [
            f"{instrument}: sell quantity {integer_quantity} exceeds the held "
            f"position {held_quantity}"
        ]
    return []
