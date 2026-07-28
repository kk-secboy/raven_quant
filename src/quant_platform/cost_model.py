from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
from math import sqrt
from typing import Any

COST_SCHEDULE_VERSION = "cn-effective-cost-v1"

# Effective-dated schedule versions recorded from the named announcements.
COST_SCHEDULE_VERSION_2005 = "cn-effective-cost-2005-01-24"
COST_SCHEDULE_VERSION_2007 = "cn-effective-cost-2007-05-30"
COST_SCHEDULE_VERSION_2008_04 = "cn-effective-cost-2008-04-24"
COST_SCHEDULE_VERSION_2008_09 = "cn-effective-cost-2008-09-19"
COST_SCHEDULE_VERSION_2015 = "cn-effective-cost-2015-08-01"
COST_SCHEDULE_VERSION_2022 = "cn-effective-cost-2022-04-29"
COST_SCHEDULE_VERSION_2023 = "cn-effective-cost-2023-08-28"

KNOWN_COST_SCHEDULE_VERSIONS = frozenset(
    {
        COST_SCHEDULE_VERSION,
        COST_SCHEDULE_VERSION_2005,
        COST_SCHEDULE_VERSION_2007,
        COST_SCHEDULE_VERSION_2008_04,
        COST_SCHEDULE_VERSION_2008_09,
        COST_SCHEDULE_VERSION_2015,
        COST_SCHEDULE_VERSION_2022,
        COST_SCHEDULE_VERSION_2023,
    }
)

# Current-law rates, shared by the bare CostModelConfig defaults below and the
# latest recorded schedule version.  When a new schedule version is recorded,
# append it to CN_COST_SCHEDULE_VERSIONS and update these two constants to its
# rates so bare defaults always track current law.
CURRENT_STOCK_SELL_STAMP_DUTY_RATE = 0.0005
CURRENT_TRANSFER_FEE_RATE = 0.00001


def infer_cn_asset_type(instrument: str) -> str:
    value = str(instrument).upper()
    digits = "".join(character for character in value if character.isdigit())[-6:]
    if len(digits) != 6:
        raise ValueError(f"cannot classify Chinese asset type: {instrument}")
    if digits.startswith(("15", "16", "18", "50", "51", "52", "56", "58")):
        return "etf"
    return "stock"


@dataclass(frozen=True)
class CostModelConfig:
    """One effective-dated cost version shared by research, Qlib and execution.

    Commission, minimum commission, slippage, participation and impact entries
    are conservative broker assumptions, not regulatory rules.  Stamp duty and
    transfer fee rates are recorded per effective range; ``source`` names the
    announcement a recorded version is taken from.
    """

    version: str = COST_SCHEDULE_VERSION
    effective_from: str = "2000-01-01"
    effective_to: str | None = None
    buy_commission_rate: float = 0.0005
    sell_commission_rate: float = 0.0005
    stock_sell_stamp_duty_rate: float = CURRENT_STOCK_SELL_STAMP_DUTY_RATE
    etf_sell_stamp_duty_rate: float = 0.0
    transfer_fee_rate: float = CURRENT_TRANSFER_FEE_RATE
    annual_borrow_rate: float = 0.0
    min_commission: float = 5.0
    fixed_slippage_rate: float = 0.0005
    max_volume_participation: float = 0.01
    impact_at_max_participation: float = 0.0010
    lot_size: int = 100
    source: str = ""

    def __post_init__(self) -> None:
        rates = (
            self.buy_commission_rate,
            self.sell_commission_rate,
            self.stock_sell_stamp_duty_rate,
            self.etf_sell_stamp_duty_rate,
            self.transfer_fee_rate,
            self.annual_borrow_rate,
            self.fixed_slippage_rate,
            self.max_volume_participation,
            self.impact_at_max_participation,
        )
        if min(rates) < 0 or self.max_volume_participation <= 0:
            raise ValueError("cost rates and participation limit must be non-negative")
        if self.min_commission < 0 or self.lot_size < 1:
            raise ValueError("minimum commission and lot size are invalid")
        if self.version not in KNOWN_COST_SCHEDULE_VERSIONS:
            raise ValueError("cost schedule version is obsolete")
        start = date.fromisoformat(self.effective_from)
        if self.effective_to is not None and date.fromisoformat(self.effective_to) < start:
            raise ValueError("cost schedule effective dates are invalid")

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> CostModelConfig:
        source = dict(values or {})
        aliases = {
            "cost_schedule_version": "version",
            "open_cost": "buy_commission_rate",
            "close_cost": "sell_commission_rate",
            "slippage": "fixed_slippage_rate",
            "min_cost": "min_commission",
        }
        normalized: dict[str, Any] = {}
        for key in cls.__dataclass_fields__:
            # None means "not supplied": fall back to the field default so
            # request models with optional cost fields validate cleanly.
            if key in source and source[key] is not None:
                normalized[key] = source[key]
        for old, new in aliases.items():
            if old in source and new not in normalized:
                normalized[new] = source[old]
        if "close_cost" in source and not {
            "sell_commission_rate",
            "stock_sell_stamp_duty_rate",
        }.intersection(source):
            normalized["stock_sell_stamp_duty_rate"] = 0.0
        return cls(**normalized)

    def doubled(self) -> CostModelConfig:
        return replace(
            self,
            buy_commission_rate=self.buy_commission_rate * 2,
            sell_commission_rate=self.sell_commission_rate * 2,
            stock_sell_stamp_duty_rate=self.stock_sell_stamp_duty_rate * 2,
            etf_sell_stamp_duty_rate=self.etf_sell_stamp_duty_rate * 2,
            transfer_fee_rate=self.transfer_fee_rate * 2,
            annual_borrow_rate=self.annual_borrow_rate * 2,
            fixed_slippage_rate=self.fixed_slippage_rate * 2,
            impact_at_max_participation=self.impact_at_max_participation * 2,
            min_commission=self.min_commission * 2,
        )

    def scaled(self, **multipliers: float) -> CostModelConfig:
        """Per-component stress view: multiply named fields, keep the rest.

        Unlike :meth:`doubled`, each stress scenario degrades exactly the
        components named in ``multipliers`` (design draft 7.3: commission,
        spread/slippage, impact and fill capacity are stressed separately).
        """

        unknown = set(multipliers).difference(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown cost model fields for scaling: {sorted(unknown)}")
        updates = {
            name: float(getattr(self, name)) * float(multiplier)
            for name, multiplier in multipliers.items()
        }
        return replace(self, **updates)

    def covers(self, trade_date: date) -> bool:
        start = date.fromisoformat(self.effective_from)
        end = date.fromisoformat(self.effective_to) if self.effective_to else None
        return trade_date >= start and (end is None or trade_date <= end)

    def market_impact_rate(self, participation: float) -> float:
        if participation < 0:
            raise ValueError("participation must be non-negative")
        if participation == 0:
            return 0.0
        ratio = participation / self.max_volume_participation
        return self.impact_at_max_participation * sqrt(ratio)

    def estimate_breakdown(
        self,
        *,
        side: str,
        gross_value: float,
        participation: float,
        asset_type: str = "stock",
        trade_date: date | None = None,
        borrow_days: int = 0,
    ) -> dict[str, float | str]:
        if gross_value < 0:
            raise ValueError("gross value must be non-negative")
        normalized_side = side.lower()
        normalized_asset = asset_type.lower()
        if normalized_side not in {"buy", "sell"}:
            raise ValueError("cost side must be buy or sell")
        if normalized_asset not in {"stock", "etf"}:
            raise ValueError("cost schedule does not support this asset type")
        if borrow_days < 0:
            raise ValueError("borrow days must be non-negative")
        if trade_date is not None and not self.covers(trade_date):
            raise ValueError("no effective cost schedule exists for the trade date")
        commission_rate = (
            self.buy_commission_rate if normalized_side == "buy" else self.sell_commission_rate
        )
        commission = (
            max(self.min_commission, gross_value * commission_rate)
            if gross_value > 0
            else 0.0
        )
        stamp_rate = 0.0
        if normalized_side == "sell":
            stamp_rate = (
                self.stock_sell_stamp_duty_rate
                if normalized_asset == "stock"
                else self.etf_sell_stamp_duty_rate
            )
        stamp_duty = gross_value * stamp_rate
        # ChinaClear's secondary-market transaction transfer fee is an A-share
        # charge. ETF creation/redemption can have separate basket transfer
        # fees, but those are not incurred by this secondary-market simulator.
        transfer_fee = (
            gross_value * self.transfer_fee_rate
            if normalized_asset == "stock"
            else 0.0
        )
        slippage = gross_value * self.fixed_slippage_rate
        impact = gross_value * self.market_impact_rate(participation)
        borrow = gross_value * self.annual_borrow_rate * borrow_days / 252.0
        total = commission + stamp_duty + transfer_fee + slippage + impact + borrow
        return {
            "version": self.version,
            "asset_type": normalized_asset,
            "commission": commission,
            "stamp_duty": stamp_duty,
            "transfer_fee": transfer_fee,
            "slippage": slippage,
            "market_impact": impact,
            "borrow_cost": borrow,
            "total": total,
        }

    def estimate(
        self,
        *,
        side: str,
        gross_value: float,
        participation: float,
        asset_type: str = "stock",
        trade_date: date | None = None,
        borrow_days: int = 0,
    ) -> float:
        return float(
            self.estimate_breakdown(
                side=side,
                gross_value=gross_value,
                participation=participation,
                asset_type=asset_type,
                trade_date=trade_date,
                borrow_days=borrow_days,
            )["total"]
        )

    def factor_screening_rate(
        self,
        *,
        reference_order_value: float,
        participation: float | None = None,
    ) -> float:
        """Conservative round-trip rate used by the validation-only factor screen."""

        if reference_order_value <= 0:
            raise ValueError("factor screening reference order value must be positive")
        assumed_participation = (
            self.max_volume_participation if participation is None else participation
        )
        buy = self.estimate(
            side="buy",
            gross_value=reference_order_value,
            participation=assumed_participation,
        )
        sell = self.estimate(
            side="sell",
            gross_value=reference_order_value,
            participation=assumed_participation,
        )
        return (buy + sell) / reference_order_value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Conservative broker assumptions shared by every recorded version.  They are
# not regulatory rules; only stamp duty and transfer fee are recorded from
# announcements.
_BROKER_ASSUMPTIONS: dict[str, Any] = {
    "buy_commission_rate": 0.0005,
    "sell_commission_rate": 0.0005,
    "min_commission": 5.0,
    "fixed_slippage_rate": 0.0005,
    "max_volume_participation": 0.01,
    "impact_at_max_participation": 0.0010,
}

CN_COST_SCHEDULE_VERSIONS: tuple[CostModelConfig, ...] = (
    CostModelConfig(
        version=COST_SCHEDULE_VERSION_2005,
        effective_from="2005-01-24",
        effective_to="2007-05-29",
        stock_sell_stamp_duty_rate=0.002,
        transfer_fee_rate=0.00002,
        source=(
            "财政部 2005-01：2005-01-24 起证券交易印花税税率由 2‰ 下调为 1‰，"
            "立据双方分别缴纳；双边各 1‰ 按卖出侧合并等效为 0.002（回合成本等价）。"
            "过户费实为沪市按成交面值 0.3‰、深市按成交金额 0.0255‰ 双边收取，"
            "此处按成交金额 0.00002 双边近似"
        ),
        **_BROKER_ASSUMPTIONS,
    ),
    CostModelConfig(
        version=COST_SCHEDULE_VERSION_2007,
        effective_from="2007-05-30",
        effective_to="2008-04-23",
        stock_sell_stamp_duty_rate=0.006,
        transfer_fee_rate=0.00002,
        source=(
            "财政部 2007-05-29 公告：2007-05-30 起证券交易印花税税率由 1‰ "
            "调整为 3‰，立据双方分别缴纳；双边各 3‰ 按卖出侧合并等效为 0.006"
            "（回合成本等价）。过户费同前近似"
        ),
        **_BROKER_ASSUMPTIONS,
    ),
    CostModelConfig(
        version=COST_SCHEDULE_VERSION_2008_04,
        effective_from="2008-04-24",
        effective_to="2008-09-18",
        stock_sell_stamp_duty_rate=0.002,
        transfer_fee_rate=0.00002,
        source=(
            "财政部 国家税务总局 2008-04：2008-04-24 起证券交易印花税税率由 3‰ "
            "调整为 1‰，仍双边征收；双边各 1‰ 按卖出侧合并等效为 0.002。"
            "过户费同前近似"
        ),
        **_BROKER_ASSUMPTIONS,
    ),
    CostModelConfig(
        version=COST_SCHEDULE_VERSION_2008_09,
        effective_from="2008-09-19",
        effective_to="2015-07-31",
        stock_sell_stamp_duty_rate=0.001,
        transfer_fee_rate=0.00002,
        source=(
            "财政部 国家税务总局 2008-09：2008-09-19 起证券交易印花税改由出让方"
            "单边缴纳，税率 1‰ 不变。过户费实为沪市按成交面值 0.3‰、深市按成交"
            "金额 0.0255‰ 双边收取，此处按成交金额 0.00002 双边近似"
        ),
        **_BROKER_ASSUMPTIONS,
    ),
    CostModelConfig(
        version=COST_SCHEDULE_VERSION_2015,
        effective_from="2015-08-01",
        effective_to="2022-04-28",
        stock_sell_stamp_duty_rate=0.001,
        transfer_fee_rate=0.00002,
        source=(
            "中国结算 2015-07《关于调整A股交易过户费收费标准有关事项的通知》："
            "2015-08-01 起过户费按成交金额 0.00002 双边收取（沪深统一）"
        ),
        **_BROKER_ASSUMPTIONS,
    ),
    CostModelConfig(
        version=COST_SCHEDULE_VERSION_2022,
        effective_from="2022-04-29",
        effective_to="2023-08-27",
        stock_sell_stamp_duty_rate=0.001,
        transfer_fee_rate=0.00001,
        source=(
            "中国结算 2022-04-28《关于降低股票交易过户费收费标准的通知》："
            "2022-04-29 起过户费下调 50%，按成交金额 0.00001 双边收取"
        ),
        **_BROKER_ASSUMPTIONS,
    ),
    CostModelConfig(
        version=COST_SCHEDULE_VERSION_2023,
        effective_from="2023-08-28",
        effective_to=None,
        stock_sell_stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
        source=(
            "财政部 税务总局公告 2023 年第 39 号（2023-08-27 公告）："
            "2023-08-28 起证券交易印花税减半征收，卖出印花税 0.0005"
        ),
        **_BROKER_ASSUMPTIONS,
    ),
)


@dataclass(frozen=True)
class CostScheduleBook:
    """Ordered effective-dated cost schedule with fail-closed date resolution.

    ``as_of`` resolves the version covering a trade date and raises when no
    recorded version covers it.  ``doubled`` doubles every recorded version so
    stress scenarios stay effective-dated.  ``flat_view`` exposes the flat
    open_cost/close_cost/min_cost triple Qlib requires.
    """

    versions: tuple[CostModelConfig, ...]

    def __post_init__(self) -> None:
        if not self.versions:
            raise ValueError("cost schedule requires at least one version")
        ordered = tuple(
            sorted(self.versions, key=lambda item: date.fromisoformat(item.effective_from))
        )
        object.__setattr__(self, "versions", ordered)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.effective_to is None:
                raise ValueError("only the latest cost schedule version may be open-ended")
            if date.fromisoformat(previous.effective_to) >= date.fromisoformat(
                current.effective_from
            ):
                raise ValueError("cost schedule versions must not overlap")

    @classmethod
    def from_versions(cls, versions: list[CostModelConfig] | tuple[CostModelConfig, ...]):
        return cls(tuple(versions))

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> CostScheduleBook:
        source = dict(values or {})
        if not source:
            return cls(CN_COST_SCHEDULE_VERSIONS)
        raw_versions = source.get("versions")
        if raw_versions is None:
            return cls((CostModelConfig.from_mapping(source),))
        if not isinstance(raw_versions, list) or not raw_versions:
            raise ValueError("cost schedule versions must be a non-empty list")
        return cls(tuple(CostModelConfig.from_mapping(item) for item in raw_versions))

    def as_of(self, trade_date: date) -> CostModelConfig:
        day = trade_date if isinstance(trade_date, date) else date.fromisoformat(str(trade_date))
        for version in self.versions:
            if version.covers(day):
                return version
        raise ValueError(f"no effective cost schedule exists for the trade date {day}")

    def doubled(self) -> CostScheduleBook:
        """Stress-test view: every recorded version is doubled consistently."""

        return CostScheduleBook(tuple(version.doubled() for version in self.versions))

    def scaled(self, **multipliers: float) -> CostScheduleBook:
        """Per-component stress view applied to every recorded version."""

        return CostScheduleBook(tuple(version.scaled(**multipliers) for version in self.versions))

    def flat_view(self, *, as_of: date) -> dict[str, Any]:
        """Flat Qlib-style cost view resolved at one explicit date."""

        config = self.as_of(as_of)
        return {
            "open_cost": config.buy_commission_rate,
            "close_cost": config.sell_commission_rate,
            "min_cost": config.min_commission,
            "trade_unit": config.lot_size,
            "slippage": config.fixed_slippage_rate,
            "cost_schedule_version": config.version,
            "as_of": as_of.isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"versions": [version.to_dict() for version in self.versions]}


CN_COST_SCHEDULE_BOOK = CostScheduleBook(CN_COST_SCHEDULE_VERSIONS)
