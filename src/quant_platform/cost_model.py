from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
from math import sqrt
from typing import Any

COST_SCHEDULE_VERSION = "cn-effective-cost-v1"


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
    """The single cost contract used by research, Qlib and recommendations."""

    version: str = COST_SCHEDULE_VERSION
    effective_from: str = "2000-01-01"
    effective_to: str | None = None
    buy_commission_rate: float = 0.0005
    sell_commission_rate: float = 0.0005
    stock_sell_stamp_duty_rate: float = 0.0010
    etf_sell_stamp_duty_rate: float = 0.0
    transfer_fee_rate: float = 0.0
    annual_borrow_rate: float = 0.0
    min_commission: float = 5.0
    fixed_slippage_rate: float = 0.0005
    max_volume_participation: float = 0.01
    impact_at_max_participation: float = 0.0010
    lot_size: int = 100

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
        if self.version != COST_SCHEDULE_VERSION:
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
            if key in source:
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
        if trade_date is not None:
            start = date.fromisoformat(self.effective_from)
            end = date.fromisoformat(self.effective_to) if self.effective_to else None
            if trade_date < start or (end is not None and trade_date > end):
                raise ValueError("no effective cost schedule exists for the trade date")
        commission_rate = (
            self.buy_commission_rate if normalized_side == "buy" else self.sell_commission_rate
        )
        commission = max(self.min_commission, gross_value * commission_rate)
        stamp_rate = 0.0
        if normalized_side == "sell":
            stamp_rate = (
                self.stock_sell_stamp_duty_rate
                if normalized_asset == "stock"
                else self.etf_sell_stamp_duty_rate
            )
        stamp_duty = gross_value * stamp_rate
        transfer_fee = gross_value * self.transfer_fee_rate
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
