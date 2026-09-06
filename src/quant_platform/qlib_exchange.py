from __future__ import annotations

import math
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
            + start_config.stock_buy_stamp_duty_rate
            + start_config.conservative_transfer_value_rate()
            + start_config.fixed_slippage_rate
            + start_config.impact_at_max_participation
        )
        conservative_sell = (
            start_config.sell_commission_rate
            + start_config.stock_sell_stamp_duty_rate
            + start_config.conservative_transfer_value_rate()
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

    def deal_order(self, order: Order, *args: Any, **kwargs: Any) -> tuple[float, float, float]:
        evidence_start = len(self.fill_log)
        trade_value, trade_cost, trade_price = super().deal_order(order, *args, **kwargs)
        if len(self.fill_log) == evidence_start:
            # Qlib rejects suspension/limit-lock/no-quote orders before calling
            # _calc_trade_info_by_order.  Record those attempts here so the
            # capacity denominator includes every submitted order.
            evidence_price = trade_price
            try:
                usable_price = np.isfinite(float(evidence_price)) and float(evidence_price) > 0
            except (TypeError, ValueError):
                usable_price = False
            if not usable_price:
                try:
                    evidence_price = self.get_deal_price(
                        order.stock_id,
                        order.start_time,
                        order.end_time,
                        order.direction,
                    )
                except (TypeError, ValueError):
                    evidence_price = 0.0
            self._record_fill(
                order,
                trade_price=evidence_price,
                trade_value=trade_value,
                cost=trade_cost,
            )
        return trade_value, trade_cost, trade_price

    def round_amount_by_trade_unit(
        self,
        deal_amount: float,
        factor: float | None = None,
        stock_id: str | None = None,
        start_time: Any = None,
        end_time: Any = None,
    ) -> float:
        """Round inside the current request, volume and cash-clipped bound.

        Upstream adds 0.1 physical shares before flooring. That can undo a
        preceding clip or exceed the submitted amount when adjustment factors
        change. Only four floating-point ULPs around a whole unit count are
        treated as representation noise here; the returned adjusted quantity
        never exceeds this invocation's input. Exact full-position sells use
        Qlib's separate liquidation branch and do not pass through this method.
        """
        if self.trade_w_adj_price or self.trade_unit is None:
            return deal_amount
        factor = float(self._get_factor_or_raise_error(
            factor=factor, stock_id=stock_id, start_time=start_time, end_time=end_time,
        ))
        amount, unit = float(deal_amount), float(self.trade_unit)
        if (
            not math.isfinite(amount) or amount < 0
            or not math.isfinite(factor) or factor <= 0
            or not math.isfinite(unit) or unit <= 0
        ):
            raise ValueError("formal Qlib lot rounding requires finite nonnegative quantity")
        unit_count = amount * factor / unit
        if not math.isfinite(unit_count):
            raise ValueError("formal Qlib lot rounding has a non-finite unit count")
        nearest = round(unit_count)
        count = nearest if _same_float_boundary(unit_count, nearest) else math.floor(unit_count)
        # Division back into adjusted units can itself round upward by one ULP.
        # Keep the exact input representation in that case, never a larger fill.
        return min(amount, count * unit / factor)

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
            and not _same_float_boundary(
                order.deal_amount * (order.factor or 1.0), rules.min_lot,
            )
        ):
            # Below the board minimum declaration (for example fewer than 200
            # shares on STAR): the exchange would reject the order outright.
            order.deal_amount = 0.0
            self._record_fill(order, trade_price=trade_price, trade_value=0.0, cost=0.0)
            return trade_price, 0.0, 0.0
        if trade_value <= 1e-5:
            # A rejected/suspended/limit-locked order is still capacity
            # evidence.  Omitting zero fills would make the aggregate fill
            # ratio condition on successful orders and systematically
            # overstate executable capacity.
            self._record_fill(order, trade_price=trade_price, trade_value=0.0, cost=0.0)
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
            instrument=str(order.stock_id),
            # Qlib trade_price is adjusted; fees charged per physical share
            # must use amount * factor.  CNY trade_value is already invariant.
            quantity=(trade_value / trade_price * self._physical_factor(order)
                      if trade_price > 0 else 0.0),
        )
        self._record_fill(
            order,
            trade_price=trade_price,
            trade_value=trade_value,
            cost=actual_cost,
        )
        return trade_price, trade_value, actual_cost

    def _physical_factor(self, order: Order) -> float:
        factor = order.factor
        if factor is None:
            factor = self.get_factor(order.stock_id, order.start_time, order.end_time)
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            factor = float("nan")
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("formal Qlib execution has no physical-share factor")
        return factor

    def _record_fill(
        self,
        order: Order,
        *,
        trade_price: float,
        trade_value: float,
        cost: float,
    ) -> None:
        requested_amount = max(0.0, float(order.amount))
        try:
            adjusted_price = float(trade_price)
        except (TypeError, ValueError):
            adjusted_price = 0.0
        price = adjusted_price if np.isfinite(adjusted_price) and adjusted_price > 0 else 0.0
        value = max(0.0, float(trade_value))
        executed_amount = value / price if price > 0 else 0.0
        try:
            factor = self._physical_factor(order)
        except ValueError:
            if value > 0:
                raise
            # A no-quote rejection need not have a factor.  Its physical
            # quantity is unknown, not an invented factor-one fill.
            factor = None
        self.fill_log.append(
            {
                "instrument": str(order.stock_id),
                "date": str(order.start_time),
                "side": "buy" if order.direction == Order.BUY else "sell",
                "requested_amount": requested_amount,
                "amount": executed_amount,
                "capacity_fill_ratio": (
                    min(1.0, executed_amount / requested_amount)
                    if requested_amount > 0
                    else 0.0
                ),
                "trade_price": price,
                "trade_value": value,
                "cost": max(0.0, float(cost)),
                # Existing amount/price fields intentionally retain Qlib's
                # continuous adjusted units for cross-date trade statistics.
                "qlib_factor": factor,
                "raw_requested_amount": requested_amount * factor if factor is not None else None,
                "raw_amount": executed_amount * factor if factor is not None else None,
                "raw_trade_price": price / factor if factor is not None and price > 0 else None,
            }
        )


def _same_float_boundary(value: float, boundary: float) -> bool:
    """Recognize arithmetic representation noise, never a fixed share allowance."""
    return abs(value - boundary) <= 4 * max(math.ulp(value), math.ulp(boundary))


def _as_date(value: Any) -> date:
    text = str(value)
    return date.fromisoformat(text[:10])
