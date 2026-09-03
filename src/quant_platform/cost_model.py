from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
from math import sqrt
from typing import Any

COST_SCHEDULE_VERSION = "cn-effective-cost-v1"

# Effective-dated schedule versions recorded from the named announcements.
#
# Pre-2015 secondary-market fees differ structurally from the current law:
# stamp duty was charged bilaterally until 2008-09-19, Shanghai levied its
# transfer fee on par value (not traded value) until 2015-08-01, and Shenzhen
# levied its own lower traded-value rate.  The model therefore carries a
# buy-side stamp rate plus per-market transfer-fee fields; Shanghai par-value
# fees are computed from the filled quantity and the sealed par-value table
# below.  Every pre-2015 effective range is recorded with its announcement.
COST_SCHEDULE_VERSION_2008 = "cn-effective-cost-2008-01-02"
COST_SCHEDULE_VERSION_2008_STAMP_CUT = "cn-effective-cost-2008-04-24"
COST_SCHEDULE_VERSION_2008_UNILATERAL = "cn-effective-cost-2008-09-19"
COST_SCHEDULE_VERSION_2012 = "cn-effective-cost-2012-06-01"
COST_SCHEDULE_VERSION_2012_SECOND = "cn-effective-cost-2012-09-01"
COST_SCHEDULE_VERSION_2015 = "cn-effective-cost-2015-08-01"
COST_SCHEDULE_VERSION_2022 = "cn-effective-cost-2022-04-29"
COST_SCHEDULE_VERSION_2023 = "cn-effective-cost-2023-08-28"

KNOWN_COST_SCHEDULE_VERSIONS = frozenset(
    {
        COST_SCHEDULE_VERSION,
        COST_SCHEDULE_VERSION_2008,
        COST_SCHEDULE_VERSION_2008_STAMP_CUT,
        COST_SCHEDULE_VERSION_2008_UNILATERAL,
        COST_SCHEDULE_VERSION_2012,
        COST_SCHEDULE_VERSION_2012_SECOND,
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


def infer_cn_exchange(instrument: str) -> str:
    """Return the listing exchange ("sh" or "sz") for one CN instrument."""

    value = str(instrument).upper()
    if value.startswith(("SH", "60", "68", "50", "51", "52", "56", "58")):
        return "sh"
    if value.startswith(("SZ", "00", "30", "15", "16", "18")):
        return "sz"
    digits = "".join(character for character in value if character.isdigit())[-6:]
    if len(digits) != 6:
        raise ValueError(f"cannot classify Chinese exchange: {instrument}")
    if digits.startswith(("60", "68", "50", "51", "52", "56", "58")):
        return "sh"
    if digits.startswith(("00", "30", "15", "16", "18")):
        return "sz"
    raise ValueError(f"cannot classify Chinese exchange: {instrument}")


# Sealed par-value exceptions.  A-share par value is CNY 1.00 for essentially
# every listing; the handful of historical exceptions inside the governed
# 2008+ coverage are recorded here.  No-par red-chip STAR listings only exist
# from 2019 and are outside the pre-2015 ranges that need par value.
CN_A_SHARE_PAR_VALUE_EXCEPTIONS: dict[str, float] = {
    "601899": 0.1,  # 紫金矿业
    "603993": 0.2,  # 洛阳钼业
}
CN_A_SHARE_DEFAULT_PAR_VALUE = 1.0


def infer_cn_par_value(instrument: str) -> float:
    digits = "".join(character for character in str(instrument) if character.isdigit())[-6:]
    if len(digits) != 6:
        raise ValueError(f"cannot resolve Chinese par value: {instrument}")
    return CN_A_SHARE_PAR_VALUE_EXCEPTIONS.get(digits, CN_A_SHARE_DEFAULT_PAR_VALUE)


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
    stock_buy_stamp_duty_rate: float = 0.0
    etf_sell_stamp_duty_rate: float = 0.0
    transfer_fee_rate: float = CURRENT_TRANSFER_FEE_RATE
    sh_transfer_fee_par_rate: float = 0.0
    sz_transfer_fee_rate: float | None = None
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
            self.stock_buy_stamp_duty_rate,
            self.etf_sell_stamp_duty_rate,
            self.transfer_fee_rate,
            self.sh_transfer_fee_par_rate,
            self.annual_borrow_rate,
            self.fixed_slippage_rate,
            self.max_volume_participation,
            self.impact_at_max_participation,
        )
        if min(rates) < 0 or self.max_volume_participation <= 0:
            raise ValueError("cost rates and participation limit must be non-negative")
        if self.sz_transfer_fee_rate is not None and self.sz_transfer_fee_rate < 0:
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
            stock_buy_stamp_duty_rate=self.stock_buy_stamp_duty_rate * 2,
            etf_sell_stamp_duty_rate=self.etf_sell_stamp_duty_rate * 2,
            transfer_fee_rate=self.transfer_fee_rate * 2,
            sh_transfer_fee_par_rate=self.sh_transfer_fee_par_rate * 2,
            sz_transfer_fee_rate=(
                None
                if self.sz_transfer_fee_rate is None
                else self.sz_transfer_fee_rate * 2
            ),
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

    @property
    def uses_per_market_transfer_fee(self) -> bool:
        return self.sh_transfer_fee_par_rate > 0 or self.sz_transfer_fee_rate is not None

    def conservative_transfer_value_rate(self) -> float:
        """Worst-case transfer fee expressed as one traded-value rate.

        Shanghai's pre-2015 par-value fee is bounded by its par rate itself:
        par value never exceeds CNY 1 within the sealed exceptions and A-share
        prices do not sustain below CNY 1, so fee/value <= par_rate.  Valueless
        screening paths use this bound instead of silently dropping the fee.
        """

        return max(
            self.transfer_fee_rate,
            self.sz_transfer_fee_rate or 0.0,
            self.sh_transfer_fee_par_rate,
        )

    def estimate_breakdown(
        self,
        *,
        side: str,
        gross_value: float,
        participation: float,
        asset_type: str = "stock",
        trade_date: date | None = None,
        borrow_days: int = 0,
        instrument: str | None = None,
        quantity: float | None = None,
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
        if quantity is not None and quantity < 0:
            raise ValueError("quantity must be non-negative")
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
        if normalized_asset == "stock":
            stamp_rate = (
                self.stock_buy_stamp_duty_rate
                if normalized_side == "buy"
                else self.stock_sell_stamp_duty_rate
            )
        else:
            stamp_rate = (
                self.etf_sell_stamp_duty_rate if normalized_side == "sell" else 0.0
            )
        stamp_duty = gross_value * stamp_rate
        # ChinaClear's secondary-market transaction transfer fee is an A-share
        # charge. ETF creation/redemption can have separate basket transfer
        # fees, but those are not incurred by this secondary-market simulator.
        transfer_fee = 0.0
        transfer_fee_basis = "none"
        if normalized_asset == "stock" and gross_value > 0:
            if self.uses_per_market_transfer_fee:
                if instrument is None:
                    raise ValueError(
                        "per-market transfer fee requires the traded instrument"
                    )
                exchange = infer_cn_exchange(instrument)
                if exchange == "sh" and self.sh_transfer_fee_par_rate > 0:
                    if quantity is None:
                        raise ValueError(
                            "Shanghai par-value transfer fee requires the filled quantity"
                        )
                    transfer_fee = (
                        quantity
                        * infer_cn_par_value(instrument)
                        * self.sh_transfer_fee_par_rate
                    )
                    transfer_fee_basis = "par_value"
                elif exchange == "sz" and self.sz_transfer_fee_rate is not None:
                    transfer_fee = gross_value * self.sz_transfer_fee_rate
                    transfer_fee_basis = "traded_value"
                else:
                    transfer_fee = gross_value * self.transfer_fee_rate
                    transfer_fee_basis = "traded_value"
            else:
                transfer_fee = gross_value * self.transfer_fee_rate
                transfer_fee_basis = "traded_value"
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
            "transfer_fee_basis": transfer_fee_basis,
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
        instrument: str | None = None,
        quantity: float | None = None,
    ) -> float:
        return float(
            self.estimate_breakdown(
                side=side,
                gross_value=gross_value,
                participation=participation,
                asset_type=asset_type,
                trade_date=trade_date,
                borrow_days=borrow_days,
                instrument=instrument,
                quantity=quantity,
            )["total"]
        )

    def reference_one_side_rate(
        self,
        *,
        side: str,
        gross_value: float,
        participation: float,
    ) -> float:
        """One-side value rate without per-instrument data.

        Uses the conservative transfer-fee bound so version-level reference
        rates stay honest for per-market (pre-2015) schedule versions.
        """

        if gross_value <= 0:
            raise ValueError("reference gross value must be positive")
        if side not in {"buy", "sell"}:
            raise ValueError("cost side must be buy or sell")
        commission_rate = (
            self.buy_commission_rate if side == "buy" else self.sell_commission_rate
        )
        stamp_rate = (
            self.stock_buy_stamp_duty_rate
            if side == "buy"
            else self.stock_sell_stamp_duty_rate
        )
        return (
            max(self.min_commission, gross_value * commission_rate)
            + gross_value
            * (
                stamp_rate
                + self.conservative_transfer_value_rate()
                + self.fixed_slippage_rate
                + self.market_impact_rate(participation)
            )
        ) / gross_value

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
        buy = self.reference_one_side_rate(
            side="buy",
            gross_value=reference_order_value,
            participation=assumed_participation,
        )
        sell = self.reference_one_side_rate(
            side="sell",
            gross_value=reference_order_value,
            participation=assumed_participation,
        )
        return buy + sell

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

# Pre-2015 commissions were broker-negotiated under the 2002 floating regime
# capped at 0.003 per side.  Seal a deliberately cost-pessimistic 0.002 per
# side so pre-2015 evidence is never flattered by today's cheaper assumption.
_PRE_2015_BROKER_ASSUMPTIONS: dict[str, Any] = {
    **_BROKER_ASSUMPTIONS,
    "buy_commission_rate": 0.002,
    "sell_commission_rate": 0.002,
}

CN_COST_SCHEDULE_VERSIONS: tuple[CostModelConfig, ...] = (
    CostModelConfig(
        version=COST_SCHEDULE_VERSION_2008,
        effective_from="2008-01-02",
        effective_to="2008-04-23",
        stock_sell_stamp_duty_rate=0.003,
        stock_buy_stamp_duty_rate=0.003,
        transfer_fee_rate=0.0,
        sh_transfer_fee_par_rate=0.0005,
        sz_transfer_fee_rate=0.0000255,
        source=(
            "财政部 国家税务总局 2007-05-29 通知：2007-05-30 起证券交易印花税 "
            "按成交金额 3‰ 双边征收；中国结算：沪市过户费按成交面额 0.5‰ 双边，"
            "深市按成交金额 0.0255‰ 双边"
        ),
        **_PRE_2015_BROKER_ASSUMPTIONS,
    ),
    CostModelConfig(
        version=COST_SCHEDULE_VERSION_2008_STAMP_CUT,
        effective_from="2008-04-24",
        effective_to="2008-09-18",
        stock_sell_stamp_duty_rate=0.001,
        stock_buy_stamp_duty_rate=0.001,
        transfer_fee_rate=0.0,
        sh_transfer_fee_par_rate=0.0005,
        sz_transfer_fee_rate=0.0000255,
        source=(
            "财政部 国家税务总局：2008-04-24 起证券交易印花税由 3‰ 下调为 "
            "1‰（双边征收）"
        ),
        **_PRE_2015_BROKER_ASSUMPTIONS,
    ),
    CostModelConfig(
        version=COST_SCHEDULE_VERSION_2008_UNILATERAL,
        effective_from="2008-09-19",
        effective_to="2012-05-31",
        stock_sell_stamp_duty_rate=0.001,
        transfer_fee_rate=0.0,
        sh_transfer_fee_par_rate=0.0005,
        sz_transfer_fee_rate=0.0000255,
        source=(
            "财税明电〔2008〕2号：2008-09-19 起证券交易印花税改为对出让方 "
            "单边按 1‰ 征收"
        ),
        **_PRE_2015_BROKER_ASSUMPTIONS,
    ),
    CostModelConfig(
        version=COST_SCHEDULE_VERSION_2012,
        effective_from="2012-06-01",
        effective_to="2012-08-31",
        stock_sell_stamp_duty_rate=0.001,
        transfer_fee_rate=0.0,
        sh_transfer_fee_par_rate=0.000375,
        sz_transfer_fee_rate=0.0000255,
        source=(
            "中国结算 2012-04-30《关于调整A股交易过户费收费标准的通知》："
            "2012-06-01 起沪市过户费按成交面额 0.375‰ 双边收取"
        ),
        **_PRE_2015_BROKER_ASSUMPTIONS,
    ),
    CostModelConfig(
        version=COST_SCHEDULE_VERSION_2012_SECOND,
        effective_from="2012-09-01",
        effective_to="2015-07-31",
        stock_sell_stamp_duty_rate=0.001,
        transfer_fee_rate=0.0,
        sh_transfer_fee_par_rate=0.0003,
        sz_transfer_fee_rate=0.0000255,
        source=(
            "中国结算 2012-08-02《关于进一步调整A股交易过户费收费标准有关事项"
            "的通知》：2012-09-01 起沪市过户费按成交面额 0.3‰ 双边收取"
        ),
        **_PRE_2015_BROKER_ASSUMPTIONS,
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
        if raw_versions is not None:
            if not isinstance(raw_versions, list) or not raw_versions:
                raise ValueError("cost schedule versions must be a non-empty list")
            return cls(tuple(CostModelConfig.from_mapping(item) for item in raw_versions))

        requested_version = str(
            source.get("version")
            or source.get("cost_schedule_version")
            or COST_SCHEDULE_VERSION
        )
        if requested_version == COST_SCHEDULE_VERSION:
            recorded = CN_COST_SCHEDULE_VERSIONS
        else:
            recorded = tuple(
                item for item in CN_COST_SCHEDULE_VERSIONS if item.version == requested_version
            )
            if not recorded:
                raise ValueError(f"unknown cost schedule version: {requested_version}")

        # A flat strategy request freezes broker assumptions, not historical
        # tax law.  Overlay only those assumptions on the authoritative
        # effective-dated regulatory schedule.  Previously the flat request
        # became one version effective from 2000, silently applying today's
        # 0.5‰ sell stamp duty to 2018-2023 backtests.
        template = CostModelConfig.from_mapping(source)
        broker_fields = (
            "buy_commission_rate",
            "sell_commission_rate",
            "annual_borrow_rate",
            "min_commission",
            "fixed_slippage_rate",
            "max_volume_participation",
            "impact_at_max_participation",
            "lot_size",
        )
        regulatory_fields = (
            "stock_sell_stamp_duty_rate",
            "etf_sell_stamp_duty_rate",
            "transfer_fee_rate",
        )
        current_defaults = CostModelConfig()
        explicit_regulatory_overrides = tuple(
            name
            for name in regulatory_fields
            if source.get(name) is not None
            and float(source[name]) != float(getattr(current_defaults, name))
        )
        overlay_fields = (*broker_fields, *explicit_regulatory_overrides)
        return cls(
            tuple(
                replace(
                    version,
                    **{name: getattr(template, name) for name in overlay_fields},
                )
                for version in recorded
            )
        )

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

    def factor_screening_rate(
        self,
        *,
        reference_order_value: float,
        start: date,
        end: date,
        participation: float | None = None,
    ) -> float:
        """Return the conservative effective round-trip rate for one period.

        Factor screening uses one scalar cost for a diagnostic long/short
        series.  Resolve every law version intersecting the validation period
        and use the maximum rather than silently applying today's lower tax to
        older observations.  Formal backtests still resolve every fill by its
        own trade date.
        """

        if end < start:
            raise ValueError("factor screening cost period is invalid")
        self.as_of(start)
        self.as_of(end)
        candidates = [
            version.factor_screening_rate(
                reference_order_value=reference_order_value,
                participation=participation,
            )
            for version in self.versions
            if date.fromisoformat(version.effective_from) <= end
            and (
                version.effective_to is None
                or date.fromisoformat(version.effective_to) >= start
            )
        ]
        if not candidates:
            raise ValueError("no effective cost schedule covers the factor screening period")
        return max(candidates)

    def flat_view(self, *, as_of: date) -> dict[str, Any]:
        """Flat Qlib-style cost view resolved at one explicit date."""

        config = self.as_of(as_of)
        transfer_bound = config.conservative_transfer_value_rate()
        return {
            "open_cost": (
                config.buy_commission_rate
                + config.stock_buy_stamp_duty_rate
                + transfer_bound
            ),
            "close_cost": (
                config.sell_commission_rate
                + config.stock_sell_stamp_duty_rate
                + transfer_bound
            ),
            "min_cost": config.min_commission,
            "trade_unit": config.lot_size,
            "slippage": config.fixed_slippage_rate,
            "cost_schedule_version": config.version,
            "as_of": as_of.isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"versions": [version.to_dict() for version in self.versions]}


CN_COST_SCHEDULE_BOOK = CostScheduleBook(CN_COST_SCHEDULE_VERSIONS)
