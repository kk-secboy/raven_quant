from __future__ import annotations

from typing import Any

import numpy as np
from qlib.backtest.decision import Order
from qlib.backtest.exchange import Exchange

from .cost_model import CostModelConfig, infer_cn_asset_type


class SquareRootImpactExchange(Exchange):
    """Qlib Exchange using the platform's single square-root cost contract.

    Qlib still performs suspension, price-limit, participation and lot-size
    clipping.  The returned transaction cost is replaced with the exact shared
    model after Qlib determines the executable amount.
    """

    def __init__(self, *, cost_model: CostModelConfig, **kwargs: Any) -> None:
        self.cost_model = cost_model
        self.fill_log: list[dict[str, Any]] = []
        conservative_buy = (
            cost_model.buy_commission_rate
            + cost_model.fixed_slippage_rate
            + cost_model.impact_at_max_participation
        )
        conservative_sell = (
            cost_model.sell_commission_rate
            + cost_model.fixed_slippage_rate
            + cost_model.impact_at_max_participation
        )
        super().__init__(
            open_cost=conservative_buy,
            close_cost=conservative_sell,
            min_cost=cost_model.min_commission,
            impact_cost=0.0,
            trade_unit=cost_model.lot_size,
            volume_threshold=("current", f"{cost_model.max_volume_participation} * $volume"),
            **kwargs,
        )

    def _calc_trade_info_by_order(
        self,
        order: Order,
        position: Any,
        dealt_order_amount: dict,
    ) -> tuple[float, float, float]:
        trade_price, trade_value, _ = super()._calc_trade_info_by_order(
            order, position, dealt_order_amount
        )
        if trade_value <= 1e-5:
            return trade_price, trade_value, 0.0
        market_value = float(
            self.get_volume(order.stock_id, order.start_time, order.end_time) * trade_price
        )
        participation = (
            min(self.cost_model.max_volume_participation, trade_value / market_value)
            if market_value > 0 and np.isfinite(market_value)
            else self.cost_model.max_volume_participation
        )
        side = "buy" if order.direction == Order.BUY else "sell"
        actual_cost = self.cost_model.estimate(
            side=side,
            gross_value=trade_value,
            participation=participation,
            asset_type=infer_cn_asset_type(str(order.stock_id)),
            trade_date=order.start_time.date(),
        )
        self.fill_log.append(
            {
                "instrument": str(order.stock_id),
                "date": str(order.start_time),
                "side": side,
                "requested_amount": float(order.amount),
                "amount": float(trade_value / trade_price) if trade_price else 0.0,
                "capacity_fill_ratio": (
                    min(1.0, float(trade_value / trade_price) / float(order.amount))
                    if trade_price and float(order.amount) > 0
                    else 0.0
                ),
                "trade_price": float(trade_price),
                "trade_value": float(trade_value),
                "cost": float(actual_cost),
            }
        )
        return trade_price, trade_value, actual_cost
