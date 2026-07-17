from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
from qlib.backtest.decision import Order
from qlib.backtest.exchange import Exchange

from .cost_model import CostModelConfig, CostScheduleBook, infer_cn_asset_type
from .market_rules import OrderUnitRules, order_unit_rules


class SquareRootImpactExchange(Exchange):
    """Qlib Exchange using the platform's single square-root cost contract.

    Qlib still performs suspension, price-limit, participation and lot-size
    clipping.  The returned transaction cost is replaced with the exact shared
    model after Qlib determines the executable amount, resolved per trade date
    from the effective-dated cost schedule.
    """

    def __init__(
        self,
        *,
        cost_model: CostModelConfig | None = None,
        cost_schedule: CostScheduleBook | None = None,
        **kwargs: Any,
    ) -> None:
        if (cost_model is None) == (cost_schedule is None):
            raise ValueError("exactly one of cost_model or cost_schedule is required")
        self.cost_schedule = cost_schedule or CostScheduleBook.from_versions([cost_model])
        self.fill_log: list[dict[str, Any]] = []
        # Qlib's Exchange accepts only one flat open/close/min cost triple, so it
        # is resolved at the backtest start date via flat_view.  A backtest span
        # crossing 2023-08-28 is therefore approximate at this Qlib adapter layer;
        # the authoritative per-fill cost is resolved per trade date from the
        # schedule in _calc_trade_info_by_order below (ExecutionCore semantics).
        start_time = kwargs.get("start_time")
        start_config = (
            self.cost_schedule.as_of(_as_date(start_time))
            if start_time is not None
            else self.cost_schedule.versions[-1]
        )
        conservative_buy = (
            start_config.buy_commission_rate
            + start_config.fixed_slippage_rate
            + start_config.impact_at_max_participation
        )
        conservative_sell = (
            start_config.sell_commission_rate
            + start_config.fixed_slippage_rate
            + start_config.impact_at_max_participation
        )
        super().__init__(
            open_cost=conservative_buy,
            close_cost=conservative_sell,
            min_cost=start_config.min_commission,
            impact_cost=0.0,
            trade_unit=start_config.lot_size,
            volume_threshold=("current", f"{start_config.max_volume_participation} * $volume"),
            **kwargs,
        )

    def _calc_trade_info_by_order(
        self,
        order: Order,
        position: Any,
        dealt_order_amount: dict,
    ) -> tuple[float, float, float]:
        rules: OrderUnitRules | None = None
        try:
            rules = order_unit_rules(str(order.stock_id), order.start_time.date())
        except ValueError:
            self.logger.warning(
                "no board order-unit rules for %s on %s; using flat trade_unit",
                order.stock_id,
                order.start_time,
            )
        original_trade_unit = self.trade_unit
        if rules is not None:
            # Buys round down by the board lot increment (100 on the main
            # boards, 1 above the 200-share minimum on STAR, 1 above 100 on
            # BSE).  Sells round only to whole shares so odd-lot positions can
            # be reduced or exited.
            self.trade_unit = 1 if order.direction == Order.SELL else rules.lot_increment
        try:
            trade_price, trade_value, _ = super()._calc_trade_info_by_order(
                order, position, dealt_order_amount
            )
        finally:
            self.trade_unit = original_trade_unit
        if (
            rules is not None
            and order.direction == Order.BUY
            and trade_value > 1e-5
            and order.deal_amount * (order.factor or 1.0) < rules.min_lot
        ):
            # Below the board minimum declaration (for example fewer than 200
            # shares on STAR): the exchange would reject the order outright.
            order.deal_amount = 0.0
            return trade_price, 0.0, 0.0
        if trade_value <= 1e-5:
            return trade_price, trade_value, 0.0
        trade_date = order.start_time.date()
        cost_model = self.cost_schedule.as_of(trade_date)
        market_value = float(
            self.get_volume(order.stock_id, order.start_time, order.end_time) * trade_price
        )
        participation = (
            min(cost_model.max_volume_participation, trade_value / market_value)
            if market_value > 0 and np.isfinite(market_value)
            else cost_model.max_volume_participation
        )
        side = "buy" if order.direction == Order.BUY else "sell"
        actual_cost = cost_model.estimate(
            side=side,
            gross_value=trade_value,
            participation=participation,
            asset_type=infer_cn_asset_type(str(order.stock_id)),
            trade_date=trade_date,
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


def _as_date(value: Any) -> date:
    text = str(value)
    return date.fromisoformat(text[:10])
