"""ETF 子类型白名单门禁（设计稿 §1.3/§5.1）。

只有验收通过的 ETF 子类型才允许进入模拟撮合；其余基金子类型只能展示/
研究，订单在成交判定处 fail-closed 拒单并记录原因。跨境/债券/黄金/商品/
货币 ETF 须先完成对应日历、回转、估值、费用与风险验收才进入白名单。

本注册表是代码级事实来源，与 ``strategy_catalog`` 同模式：每个子类型记
录验收状态、验收日期与依据（或待验收项），版本号随注册表变更升级。

与 ``market_permission`` 的分工（刻意不复用）：市场权限是*个人账户*的
DB 版本化授权模型，``unknown`` 语义是把建议降级为 simulation_only 而不
拒单；本门禁是*平台级*产品验收，要求对未验收子类型在撮合路径硬拒单。
两者语义不同，强行复用会把个人授权纪律与产品验收混为一谈，故新建轻量
注册表，scope 分类沿用 market_permission 的 etf_subtype 前缀口径。
"""

from __future__ import annotations

from dataclasses import dataclass

from .market_rules import _instrument_digits

ETF_SUBTYPE_GATE_VERSION = "etf-subtype-whitelist-2026-07-27-v1"

SUBTYPE_EQUITY = "equity"
SUBTYPE_CROSS_BORDER = "cross_border"
SUBTYPE_BOND = "bond"
SUBTYPE_GOLD = "gold"
SUBTYPE_COMMODITY = "commodity"
SUBTYPE_MONEY = "money"
SUBTYPE_UNCLASSIFIED = "unclassified"

ETF_SUBTYPES = (
    SUBTYPE_EQUITY,
    SUBTYPE_CROSS_BORDER,
    SUBTYPE_BOND,
    SUBTYPE_GOLD,
    SUBTYPE_COMMODITY,
    SUBTYPE_MONEY,
)


@dataclass(frozen=True)
class EtfSubtypeAcceptance:
    """One ETF subtype's platform acceptance record."""

    subtype: str
    name: str
    accepted: bool
    accepted_on: str | None
    evidence: str
    pending_acceptance: tuple[str, ...] = ()


# 子类型验收注册表。验收状态变更必须同时更新 accepted_on/evidence（或
# pending_acceptance）并升级 ETF_SUBTYPE_GATE_VERSION。
_ETF_SUBTYPE_REGISTRY: tuple[EtfSubtypeAcceptance, ...] = (
    EtfSubtypeAcceptance(
        subtype=SUBTYPE_EQUITY,
        name="股票型 ETF（宽基/行业/主题）",
        accepted=True,
        accepted_on="2026-07-27",
        evidence=(
            "与既有通用基金规则语义一致（申报单位 100 份、T+1 保守回转，与股票型 "
            "ETF 交易规则相符）；宽基/行业 ETF 已在模拟撮合链测试中回归覆盖"
        ),
    ),
    EtfSubtypeAcceptance(
        subtype=SUBTYPE_CROSS_BORDER,
        name="跨境 ETF",
        accepted=False,
        accepted_on=None,
        evidence="",
        pending_acceptance=("跨境交易日历对齐", "净值/估值滞后口径", "回转规则", "费用与风险验收"),
    ),
    EtfSubtypeAcceptance(
        subtype=SUBTYPE_BOND,
        name="债券 ETF",
        accepted=False,
        accepted_on=None,
        evidence="",
        pending_acceptance=("T+0 回转规则", "债券市场日历", "估值口径", "费用与风险验收"),
    ),
    EtfSubtypeAcceptance(
        subtype=SUBTYPE_GOLD,
        name="黄金 ETF",
        accepted=False,
        accepted_on=None,
        evidence="",
        pending_acceptance=("T+0 回转规则", "估值口径", "费用与风险验收"),
    ),
    EtfSubtypeAcceptance(
        subtype=SUBTYPE_COMMODITY,
        name="商品期货 ETF",
        accepted=False,
        accepted_on=None,
        evidence="",
        pending_acceptance=("T+0 回转规则", "期货估值口径", "费用与风险验收"),
    ),
    EtfSubtypeAcceptance(
        subtype=SUBTYPE_MONEY,
        name="货币 ETF",
        accepted=False,
        accepted_on=None,
        evidence="",
        pending_acceptance=("T+0 回转规则", "净值/收益估值口径", "费用与风险验收"),
    ),
)

# 代码段分类是版本化启发式：显式代码覆盖优先，其次按交易所代码段归类。
# 深市 159 段内非股票型 ETF 需列入显式覆盖，未列入的 159 代码默认股票型；
# 新代码进入数据白名单（quant_data.universe）时必须同步补充此处覆盖。
_MONEY_MARKET_CODES = frozenset(
    {
        "511620",
        "511660",
        "511690",
        "511810",
        "511820",
        "511850",
        "511860",
        "511880",
        "511900",
        "511920",
        "511930",
        "511950",
        "511960",
        "511970",
        "511980",
        "511990",
    }
)
_GOLD_CODES = frozenset({"159934", "159937"})
_COMMODITY_CODES = frozenset({"159980", "159981", "159985"})
_CROSS_BORDER_CODES = frozenset({"159920", "159941", "159509", "159513"})


def fund_subtype(instrument: str) -> str | None:
    """Classify a fund/ETF instrument into a subtype; None when not a fund.

    Unrecognised fund codes (LOF、封闭式基金等) return ``unclassified`` so the
    trading gate can reject them fail-closed.
    """

    digits = _instrument_digits(instrument)
    if not digits.startswith(("5", "15", "16", "18")):
        return None
    if digits in _MONEY_MARKET_CODES:
        return SUBTYPE_MONEY
    if digits in _GOLD_CODES:
        return SUBTYPE_GOLD
    if digits in _COMMODITY_CODES:
        return SUBTYPE_COMMODITY
    if digits in _CROSS_BORDER_CODES:
        return SUBTYPE_CROSS_BORDER
    if digits.startswith("518"):
        return SUBTYPE_GOLD
    if digits.startswith("513"):
        return SUBTYPE_CROSS_BORDER
    if digits.startswith("511"):
        return SUBTYPE_BOND
    if digits.startswith(("510", "512", "515", "516", "517", "56", "58", "159")):
        return SUBTYPE_EQUITY
    return SUBTYPE_UNCLASSIFIED


def etf_subtype_acceptance(subtype: str) -> EtfSubtypeAcceptance:
    for entry in _ETF_SUBTYPE_REGISTRY:
        if entry.subtype == subtype:
            return entry
    raise KeyError(subtype)


def list_etf_subtype_registry() -> list[EtfSubtypeAcceptance]:
    return list(_ETF_SUBTYPE_REGISTRY)


def etf_trading_gate(instrument: str) -> str | None:
    """Return the reject reason for the simulation matching path, or None.

    None means trading is unrestricted (non-fund instrument, or an accepted
    subtype). Any fund instrument whose subtype is not acceptance-whitelisted
    is rejected fail-closed; research/backtest paths do not consult this gate.
    """

    subtype = fund_subtype(instrument)
    if subtype is None:
        return None
    if subtype == SUBTYPE_UNCLASSIFIED:
        return "etf_subtype_unclassified"
    if etf_subtype_acceptance(subtype).accepted:
        return None
    return f"etf_subtype_not_accepted:{subtype}"


def validate_etf_subtype_registry() -> None:
    """Fail closed if the acceptance registry contract is violated."""

    subtypes = [entry.subtype for entry in _ETF_SUBTYPE_REGISTRY]
    if len(subtypes) != len(set(subtypes)):
        raise ValueError("ETF subtype registry contains duplicate subtypes")
    if set(subtypes) != set(ETF_SUBTYPES):
        raise ValueError("ETF subtype registry must register every known subtype")
    for entry in _ETF_SUBTYPE_REGISTRY:
        if entry.accepted and (not entry.accepted_on or not entry.evidence):
            raise ValueError(
                f"accepted ETF subtype {entry.subtype} lacks acceptance date/evidence"
            )
        if not entry.accepted and not entry.pending_acceptance:
            raise ValueError(
                f"unaccepted ETF subtype {entry.subtype} lacks pending acceptance items"
            )
