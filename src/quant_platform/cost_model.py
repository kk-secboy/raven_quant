from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import sqrt
from typing import Any


@dataclass(frozen=True)
class CostModelConfig:
    """The single cost contract used by research, Qlib and recommendations."""

    buy_commission_rate: float = 0.0005
    sell_commission_rate: float = 0.0015
    min_commission: float = 5.0
    fixed_slippage_rate: float = 0.0005
    max_volume_participation: float = 0.01
    impact_at_max_participation: float = 0.0010
    lot_size: int = 100

    def __post_init__(self) -> None:
        rates = (
            self.buy_commission_rate,
            self.sell_commission_rate,
            self.fixed_slippage_rate,
            self.max_volume_participation,
            self.impact_at_max_participation,
        )
        if min(rates) < 0 or self.max_volume_participation <= 0:
            raise ValueError("cost rates and participation limit must be non-negative")
        if self.min_commission < 0 or self.lot_size < 1:
            raise ValueError("minimum commission and lot size are invalid")

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> CostModelConfig:
        source = dict(values or {})
        aliases = {
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
        return cls(**normalized)

    def doubled(self) -> CostModelConfig:
        return replace(
            self,
            buy_commission_rate=self.buy_commission_rate * 2,
            sell_commission_rate=self.sell_commission_rate * 2,
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

    def estimate(self, *, side: str, gross_value: float, participation: float) -> float:
        if gross_value < 0:
            raise ValueError("gross value must be non-negative")
        commission_rate = (
            self.buy_commission_rate if side.lower() == "buy" else self.sell_commission_rate
        )
        commission = max(self.min_commission, gross_value * commission_rate)
        variable = gross_value * (self.fixed_slippage_rate + self.market_impact_rate(participation))
        return commission + variable

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
