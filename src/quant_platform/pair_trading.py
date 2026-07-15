from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import floor, log, sqrt
from typing import Any, Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True, slots=True)
class PairTradingConfig:
    formation_window: int = 60
    min_correlation: float = 0.80
    max_cointegration_pvalue: float = 0.05
    cointegration_recheck_days: int = 5
    entry_zscore: float = 1.50
    exit_zscore: float = 0.50
    stop_zscore: float = 3.00
    max_holding_days: int = 5
    initial_capital: float = 5_000_000.0
    pair_gross_fraction: float = 0.20
    max_volume_participation: float = 0.01
    min_capacity_fill_ratio: float = 0.95
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    min_commission: float = 5.0
    slippage: float = 0.0005
    annual_borrow_rate: float = 0.08
    lot_size: int = 100
    kalman_process_variance: float = 1e-5
    kalman_observation_variance: float = 1e-3
    min_hedge_ratio: float = 0.10
    max_hedge_ratio: float = 10.0
    max_drawdown: float = 0.10
    min_sharpe_ratio: float = 0.0
    min_closed_trades: int = 5
    min_backtest_days: int = 252
    min_rolling_cointegration_pass_rate: float = 0.80
    min_robustness_pass_rate: float = 0.75

    def __post_init__(self) -> None:
        if self.formation_window < 20:
            raise ValueError("formation_window must be at least 20 trading days")
        if not 0 <= self.min_correlation <= 1:
            raise ValueError("min_correlation must be between zero and one")
        if not 0 < self.max_cointegration_pvalue <= 1:
            raise ValueError("max_cointegration_pvalue must be in (0, 1]")
        if self.cointegration_recheck_days < 1:
            raise ValueError("cointegration_recheck_days must be positive")
        if not 0 <= self.exit_zscore < self.entry_zscore < self.stop_zscore:
            raise ValueError("z-score thresholds must satisfy exit < entry < stop")
        if not 1 <= self.max_holding_days <= 20:
            raise ValueError("max_holding_days must be between 1 and 20")
        if self.initial_capital < 100_000:
            raise ValueError("initial_capital must be at least 100000")
        if not 0 < self.pair_gross_fraction <= 1:
            raise ValueError("pair_gross_fraction must be in (0, 1]")
        if not 0 < self.max_volume_participation <= 0.20:
            raise ValueError("max_volume_participation must be in (0, 0.20]")
        if not 0 < self.min_capacity_fill_ratio <= 1:
            raise ValueError("min_capacity_fill_ratio must be in (0, 1]")
        if min(self.open_cost, self.close_cost, self.min_commission, self.slippage) < 0:
            raise ValueError("costs and slippage must be non-negative")
        if self.annual_borrow_rate < 0:
            raise ValueError("annual_borrow_rate must be non-negative")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if min(self.kalman_process_variance, self.kalman_observation_variance) <= 0:
            raise ValueError("Kalman variances must be positive")
        if not 0 < self.min_hedge_ratio < self.max_hedge_ratio:
            raise ValueError("hedge-ratio bounds are invalid")
        if not 0 < self.max_drawdown <= 0.50:
            raise ValueError("max_drawdown must be in (0, 0.50]")
        if not -5 <= self.min_sharpe_ratio <= 10:
            raise ValueError("min_sharpe_ratio must be between -5 and 10")
        if self.min_closed_trades < 1:
            raise ValueError("min_closed_trades must be positive")
        if self.min_backtest_days < 60:
            raise ValueError("min_backtest_days must be at least 60")
        if not 0 <= self.min_rolling_cointegration_pass_rate <= 1:
            raise ValueError("min_rolling_cointegration_pass_rate must be between zero and one")
        if not 0 <= self.min_robustness_pass_rate <= 1:
            raise ValueError("min_robustness_pass_rate must be between zero and one")


@dataclass(frozen=True, slots=True)
class PairEvidence:
    correlation: float
    cointegration_pvalue: float
    intercept: float
    hedge_ratio: float
    half_life_days: float | None
    observations: int


def evaluate_pair(
    y_prices: pd.Series,
    x_prices: pd.Series,
    *,
    min_observations: int = 20,
) -> PairEvidence:
    frame = (
        pd.concat(
            [pd.to_numeric(y_prices, errors="coerce"), pd.to_numeric(x_prices, errors="coerce")],
            axis=1,
            keys=["y", "x"],
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    frame = frame[(frame["y"] > 0) & (frame["x"] > 0)]
    if len(frame) < min_observations:
        raise ValueError(f"pair evidence requires at least {min_observations} aligned observations")
    log_y = np.log(frame["y"].to_numpy(dtype=float))
    log_x = np.log(frame["x"].to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(log_x)), log_x])
    intercept, hedge_ratio = np.linalg.lstsq(design, log_y, rcond=None)[0]
    residual = log_y - intercept - hedge_ratio * log_x
    pvalue = float(coint(log_y, log_x, trend="c", autolag="aic")[1])
    returns = frame.pct_change(fill_method=None).dropna()
    correlation = float(returns["y"].corr(returns["x"]))
    half_life: float | None = None
    if len(residual) >= 3:
        lagged = residual[:-1]
        delta = np.diff(residual)
        ar_design = np.column_stack([np.ones(len(lagged)), lagged])
        speed = float(np.linalg.lstsq(ar_design, delta, rcond=None)[0][1])
        if speed < 0:
            candidate = float(-log(2.0) / speed)
            if np.isfinite(candidate) and candidate > 0:
                half_life = candidate
    return PairEvidence(
        correlation=correlation,
        cointegration_pvalue=pvalue,
        intercept=float(intercept),
        hedge_ratio=float(hedge_ratio),
        half_life_days=half_life,
        observations=len(frame),
    )


def _normalize_market(frame: pd.DataFrame, *, minute: bool) -> pd.DataFrame:
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise ValueError("market data must use a (datetime, instrument) MultiIndex")
    result = frame.copy()
    result.index = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex(result.index.get_level_values(0)).tz_localize(None),
            result.index.get_level_values(1).astype(str),
        ],
        names=["datetime", "instrument"],
    )
    if result.index.has_duplicates:
        raise ValueError("market data contains duplicate datetime/instrument rows")
    required = {"close", "volume"}
    if not minute:
        required |= {"open", "paused", "up_limit", "down_limit", "shortable"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"market data is missing fields: {missing}")
    return result.sort_index()


def _kalman_spread(
    y_prices: pd.Series,
    x_prices: pd.Series,
    config: PairTradingConfig,
) -> pd.DataFrame:
    log_y = np.log(y_prices.to_numpy(dtype=float))
    log_x = np.log(x_prices.to_numpy(dtype=float))
    state = np.array([0.0, 1.0], dtype=float)
    covariance = np.eye(2, dtype=float)
    process = np.eye(2, dtype=float) * config.kalman_process_variance
    observation = float(config.kalman_observation_variance)
    rows: list[tuple[float, float, float]] = []
    for observed_y, observed_x in zip(log_y, log_x, strict=True):
        covariance = covariance + process
        design = np.array([1.0, observed_x], dtype=float)
        innovation = observed_y - float(design @ state)
        innovation_variance = float(design @ covariance @ design + observation)
        gain = covariance @ design / innovation_variance
        state = state + gain * innovation
        covariance = covariance - np.outer(gain, design) @ covariance
        hedge_ratio = float(np.clip(state[1], config.min_hedge_ratio, config.max_hedge_ratio))
        spread = observed_y - float(state[0]) - hedge_ratio * observed_x
        rows.append((float(state[0]), hedge_ratio, float(spread)))
    return pd.DataFrame(
        rows,
        index=y_prices.index,
        columns=["intercept", "hedge_ratio", "spread"],
    )


def _execution_price(
    minute: pd.DataFrame,
    trade_date: pd.Timestamp,
    instrument: str,
    side: Literal["buy", "sell"],
    slippage: float,
) -> tuple[float, float] | None:
    try:
        frame = minute.xs(instrument, level="instrument")
    except KeyError:
        return None
    day = frame[frame.index.normalize() == trade_date.normalize()].copy()
    if day.empty:
        return None
    local_time = day.index.time
    allowed = (
        (local_time >= pd.Timestamp("10:00").time()) & (local_time <= pd.Timestamp("11:20").time())
    ) | (
        (local_time >= pd.Timestamp("13:30").time()) & (local_time <= pd.Timestamp("14:50").time())
    )
    day = day.loc[allowed]
    day = day[
        (pd.to_numeric(day["close"], errors="coerce") > 0)
        & (pd.to_numeric(day["volume"], errors="coerce") > 0)
    ]
    if day.empty:
        return None
    volume = pd.to_numeric(day["volume"], errors="coerce").fillna(0.0)
    if "amount" in day:
        amount = pd.to_numeric(day["amount"], errors="coerce").fillna(0.0)
        vwap = float(amount.sum() / volume.sum()) if amount.sum() > 0 else 0.0
    else:
        prices = pd.to_numeric(day["close"], errors="coerce")
        vwap = float((prices * volume).sum() / volume.sum())
    if not np.isfinite(vwap) or vwap <= 0:
        return None
    adjusted = vwap * (1.0 + slippage if side == "buy" else 1.0 - slippage)
    return float(adjusted), float(volume.sum())


def _trade_allowed(
    daily_row: pd.Series,
    *,
    side: Literal["buy", "sell"],
    execution_price: float,
    opening_short: bool,
) -> str | None:
    paused = daily_row["paused"]
    up_limit = float(daily_row["up_limit"])
    down_limit = float(daily_row["down_limit"])
    if pd.isna(paused):
        return "suspension_state_missing"
    if bool(paused):
        return "suspended"
    if not np.isfinite(up_limit) or not np.isfinite(down_limit) or up_limit <= down_limit:
        return "price_limit_state_missing"
    if side == "buy" and execution_price >= up_limit - 1e-12:
        return "buy_at_up_limit"
    if side == "sell" and execution_price <= down_limit + 1e-12:
        return "sell_at_down_limit"
    if opening_short:
        shortable = daily_row["shortable"]
        authorized = (isinstance(shortable, (bool, np.bool_)) and bool(shortable)) or (
            isinstance(shortable, (int, float, np.integer, np.floating)) and float(shortable) == 1.0
        )
        if pd.isna(shortable) or not authorized:
            return "short_borrow_not_authorized"
    return None


def _performance(daily: pd.DataFrame) -> dict[str, float | int | None]:
    returns = daily["return"].dropna()
    if returns.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": 0.0,
            "trading_days": len(daily),
        }
    annualized = float((1.0 + returns).prod() ** (252.0 / len(returns)) - 1.0)
    volatility = float(returns.std(ddof=1))
    downside = float(returns[returns < 0].std(ddof=1))
    nav = daily["nav"]
    drawdown = nav / nav.cummax() - 1.0
    return {
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "annualized_return": annualized,
        "sharpe_ratio": float(returns.mean() / volatility * sqrt(252.0))
        if volatility > 0
        else None,
        "sortino_ratio": float(returns.mean() / downside * sqrt(252.0)) if downside > 0 else None,
        "max_drawdown": float(drawdown.min()),
        "trading_days": len(daily),
    }


def run_pair_backtest(
    daily_market: pd.DataFrame,
    minute_market: pd.DataFrame,
    *,
    leg_y: str,
    leg_x: str,
    config: PairTradingConfig | None = None,
) -> dict[str, Any]:
    config = config or PairTradingConfig()
    if not leg_y or not leg_x or leg_y == leg_x:
        raise ValueError("pair legs must be two distinct instruments")
    daily = _normalize_market(daily_market, minute=False)
    minute = _normalize_market(minute_market, minute=True)
    close = daily["close"].unstack("instrument")
    if leg_y not in close or leg_x not in close:
        raise ValueError("both pair legs must exist in daily market data")
    prices = close[[leg_y, leg_x]].apply(pd.to_numeric, errors="coerce").dropna()
    prices = prices[(prices > 0).all(axis=1)]
    if len(prices) < config.formation_window + 2:
        raise ValueError("pair backtest has insufficient aligned daily history")
    kalman = _kalman_spread(prices[leg_y], prices[leg_x], config)
    zscore = pd.Series(index=prices.index, dtype=float)
    for offset in range(config.formation_window - 1, len(prices)):
        window = kalman["spread"].iloc[offset - config.formation_window + 1 : offset + 1]
        deviation = float(window.std(ddof=1))
        if deviation > 0 and np.isfinite(deviation):
            zscore.iloc[offset] = float((window.iloc[-1] - window.mean()) / deviation)

    cash = float(config.initial_capital)
    quantities = {leg_y: 0, leg_x: 0}
    position_direction = 0
    holding_days = 0
    entry_nav: float | None = None
    pending: dict[str, Any] | None = None
    evidence: PairEvidence | None = None
    evidence_offset = -config.cointegration_recheck_days
    trades: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    closed_returns: list[float] = []
    stop_count = 0
    breakdown_count = 0
    evidence_check_count = 0
    eligible_evidence_count = 0
    planned_entry_notional = 0.0
    filled_entry_notional = 0.0

    def mark_nav(trade_date: pd.Timestamp) -> float:
        return cash + sum(
            quantities[instrument] * float(prices.loc[trade_date, instrument])
            for instrument in (leg_y, leg_x)
        )

    for offset, trade_date in enumerate(prices.index):
        if offset < config.formation_window:
            daily_rows.append({"datetime": trade_date, "nav": cash, "position": 0})
            continue
        if position_direction and offset == len(prices) - 1:
            pending = {"action": "exit", "reason": "end_of_test", "signal_date": trade_date}
        if pending:
            action = str(pending["action"])
            target_quantities = {leg_y: 0, leg_x: 0}
            hedge_ratio = float(pending.get("hedge_ratio") or kalman.loc[trade_date, "hedge_ratio"])
            direction = int(pending.get("direction") or position_direction)
            nav_before = mark_nav(trade_date)
            if action == "entry":
                gross = max(0.0, nav_before * config.pair_gross_fraction)
                y_weight = 1.0 / (1.0 + abs(hedge_ratio))
                x_weight = 1.0 - y_weight
                reference = {
                    leg_y: float(prices.loc[trade_date, leg_y]),
                    leg_x: float(prices.loc[trade_date, leg_x]),
                }
                target_quantities[leg_y] = (
                    direction
                    * floor(gross * y_weight / reference[leg_y] / config.lot_size)
                    * config.lot_size
                )
                target_quantities[leg_x] = (
                    -direction
                    * floor(gross * x_weight / reference[leg_x] / config.lot_size)
                    * config.lot_size
                )
                planned_entry_notional += sum(
                    abs(target_quantities[item]) * reference[item] for item in target_quantities
                )
            orders: list[dict[str, Any]] = []
            rejection_reason: str | None = None
            for instrument in (leg_y, leg_x):
                delta = target_quantities[instrument] - quantities[instrument]
                if delta == 0:
                    continue
                side: Literal["buy", "sell"] = "buy" if delta > 0 else "sell"
                execution = _execution_price(
                    minute,
                    trade_date,
                    instrument,
                    side,
                    config.slippage,
                )
                if execution is None:
                    rejection_reason = f"{instrument}:missing_valid_minute_execution_window"
                    break
                execution_price, minute_volume = execution
                day_row = daily.loc[(trade_date, instrument)]
                opening_short = target_quantities[instrument] < 0 and quantities[instrument] >= 0
                rejection_reason = _trade_allowed(
                    day_row,
                    side=side,
                    execution_price=execution_price,
                    opening_short=opening_short,
                )
                if rejection_reason:
                    rejection_reason = f"{instrument}:{rejection_reason}"
                    break
                capacity = (
                    floor(minute_volume * config.max_volume_participation / config.lot_size)
                    * config.lot_size
                )
                fill_ratio = min(1.0, capacity / abs(delta)) if delta else 1.0
                if fill_ratio < config.min_capacity_fill_ratio:
                    rejection_reason = f"{instrument}:insufficient_minute_capacity"
                    break
                orders.append(
                    {
                        "instrument": instrument,
                        "delta": int(delta),
                        "side": side,
                        "price": execution_price,
                    }
                )
            if rejection_reason:
                rejections.append(
                    {
                        "trade_date": trade_date.isoformat(),
                        "signal_date": pd.Timestamp(pending["signal_date"]).isoformat(),
                        "action": action,
                        "reason": rejection_reason,
                    }
                )
            else:
                total_cost = 0.0
                total_notional = 0.0
                for order in orders:
                    notional = float(order["delta"]) * float(order["price"])
                    rate = config.open_cost if action == "entry" else config.close_cost
                    commission = max(config.min_commission, abs(notional) * rate)
                    cash -= notional + commission
                    total_cost += commission
                    total_notional += abs(notional)
                    quantities[str(order["instrument"])] += int(order["delta"])
                if action == "entry":
                    position_direction = direction
                    holding_days = 0
                    entry_nav = nav_before - total_cost
                    filled_entry_notional += total_notional
                else:
                    exit_nav = cash
                    if entry_nav and entry_nav > 0:
                        closed_returns.append(float(exit_nav / entry_nav - 1.0))
                    position_direction = 0
                    holding_days = 0
                    entry_nav = None
                trades.append(
                    {
                        "trade_date": trade_date.isoformat(),
                        "signal_date": pd.Timestamp(pending["signal_date"]).isoformat(),
                        "action": action,
                        "reason": pending["reason"],
                        "direction": direction,
                        "hedge_ratio": hedge_ratio,
                        "orders": orders,
                        "gross_notional": total_notional,
                        "cost": total_cost,
                    }
                )
            pending = None

        nav = mark_nav(trade_date)
        if position_direction:
            short_value = sum(
                abs(quantities[item]) * float(prices.loc[trade_date, item])
                for item in (leg_y, leg_x)
                if quantities[item] < 0
            )
            borrow_cost = short_value * config.annual_borrow_rate / 252.0
            cash -= borrow_cost
            nav -= borrow_cost
            holding_days += 1
        daily_rows.append(
            {
                "datetime": trade_date,
                "nav": nav,
                "position": position_direction,
                "quantity_y": quantities[leg_y],
                "quantity_x": quantities[leg_x],
                "zscore": zscore.loc[trade_date],
            }
        )
        if offset >= len(prices) - 1:
            continue
        if offset - evidence_offset >= config.cointegration_recheck_days:
            window = prices.iloc[offset - config.formation_window + 1 : offset + 1]
            evidence = evaluate_pair(
                window[leg_y],
                window[leg_x],
                min_observations=config.formation_window,
            )
            evidence_offset = offset
            evidence_check_count += 1
        assert evidence is not None
        eligible = (
            evidence.correlation >= config.min_correlation
            and evidence.cointegration_pvalue <= config.max_cointegration_pvalue
            and config.min_hedge_ratio <= evidence.hedge_ratio <= config.max_hedge_ratio
        )
        if offset == evidence_offset and eligible:
            eligible_evidence_count += 1
        signal_z = float(zscore.loc[trade_date])
        if position_direction:
            reason: str | None = None
            if not eligible:
                reason = "cointegration_breakdown"
                breakdown_count += 1
            elif abs(signal_z) >= config.stop_zscore:
                reason = "zscore_stop"
                stop_count += 1
            elif abs(signal_z) <= config.exit_zscore:
                reason = "mean_reversion"
            elif holding_days >= config.max_holding_days:
                reason = "max_holding_days"
            if reason:
                pending = {"action": "exit", "reason": reason, "signal_date": trade_date}
        elif eligible and config.entry_zscore <= abs(signal_z) < config.stop_zscore:
            pending = {
                "action": "entry",
                "reason": "negative_spread" if signal_z < 0 else "positive_spread",
                "direction": 1 if signal_z < 0 else -1,
                "hedge_ratio": float(kalman.loc[trade_date, "hedge_ratio"]),
                "signal_date": trade_date,
            }

    daily_frame = pd.DataFrame(daily_rows).set_index("datetime")
    daily_frame["return"] = daily_frame["nav"].pct_change(fill_method=None).fillna(0.0)
    initial_window = prices.iloc[: config.formation_window]
    initial_evidence = evaluate_pair(
        initial_window[leg_y],
        initial_window[leg_x],
        min_observations=config.formation_window,
    )
    exits = [item for item in trades if item["action"] == "exit"]
    metrics: dict[str, Any] = {
        **_performance(daily_frame),
        "backtest_engine": "quantlab_pair",
        "pair_native_backtest": True,
        "leg_y": leg_y,
        "leg_x": leg_x,
        "initial_pair_evidence": asdict(initial_evidence),
        "trade_count": len(trades),
        "closed_trade_count": len(exits),
        "win_rate": float(sum(value > 0 for value in closed_returns) / len(closed_returns))
        if closed_returns
        else None,
        "average_closed_trade_return": float(np.mean(closed_returns)) if closed_returns else None,
        "stop_count": stop_count,
        "cointegration_breakdown_count": breakdown_count,
        "rolling_cointegration_checks": evidence_check_count,
        "rolling_cointegration_pass_rate": (
            float(eligible_evidence_count / evidence_check_count) if evidence_check_count else 0.0
        ),
        "rejected_signal_count": len(rejections),
        "capacity_fill_ratio": min(1.0, float(filled_entry_notional / planned_entry_notional))
        if planned_entry_notional > 0
        else 0.0,
        "open_position_at_end": position_direction != 0,
        "minute_execution_enforced": True,
        "shortability_enforced": True,
        "market_controls_enforced": True,
        "atomic_pair_execution_enforced": True,
        "transaction_costs_enforced": True,
        "borrow_cost_enforced": True,
        "config": asdict(config),
    }
    return {
        "metrics": metrics,
        "daily": daily_frame,
        "trades": trades,
        "rejections": rejections,
        "kalman": kalman.assign(zscore=zscore),
    }


def run_pair_paper_step(
    daily_market: pd.DataFrame,
    minute_market: pd.DataFrame,
    *,
    leg_y: str,
    leg_x: str,
    as_of_date: str,
    state: dict[str, Any],
    config: PairTradingConfig | None = None,
) -> dict[str, Any]:
    """Advance one governed pair-paper ledger day using next-session execution."""
    config = config or PairTradingConfig()
    if not leg_y or not leg_x or leg_y == leg_x:
        raise ValueError("pair legs must be two distinct instruments")
    daily = _normalize_market(daily_market, minute=False)
    minute = _normalize_market(minute_market, minute=True)
    close = daily["close"].unstack("instrument")
    if leg_y not in close or leg_x not in close:
        raise ValueError("both pair legs must exist in daily market data")
    prices = close[[leg_y, leg_x]].apply(pd.to_numeric, errors="coerce").dropna()
    prices = prices[(prices > 0).all(axis=1)]
    signal_date = pd.Timestamp(as_of_date).normalize()
    if signal_date not in prices.index:
        raise ValueError("as_of_date is not an aligned pair trading day")
    signal_offset = int(prices.index.get_loc(signal_date))
    if signal_offset < config.formation_window - 1:
        raise ValueError("pair paper step has insufficient formation history")
    if signal_offset + 1 >= len(prices):
        raise ValueError("pair paper step requires the next trading day")
    trade_date = pd.Timestamp(prices.index[signal_offset + 1]).normalize()

    history = prices.iloc[: signal_offset + 1]
    kalman = _kalman_spread(history[leg_y], history[leg_x], config)
    spread_window = kalman["spread"].iloc[-config.formation_window :]
    deviation = float(spread_window.std(ddof=1))
    if not np.isfinite(deviation) or deviation <= 0:
        raise ValueError("pair spread has no finite formation-window deviation")
    zscore = float((spread_window.iloc[-1] - spread_window.mean()) / deviation)
    evidence_window = history.iloc[-config.formation_window :]
    evidence = evaluate_pair(
        evidence_window[leg_y],
        evidence_window[leg_x],
        min_observations=config.formation_window,
    )
    hedge_ratio = float(kalman.iloc[-1]["hedge_ratio"])
    eligible = (
        evidence.correlation >= config.min_correlation
        and evidence.cointegration_pvalue <= config.max_cointegration_pvalue
        and config.min_hedge_ratio <= evidence.hedge_ratio <= config.max_hedge_ratio
    )

    direction = int(state.get("position_direction") or 0)
    quantity_y = int(state.get("quantity_y") or 0)
    quantity_x = int(state.get("quantity_x") or 0)
    if direction not in {-1, 0, 1}:
        raise ValueError("pair paper position direction is invalid")
    if quantity_y % config.lot_size or quantity_x % config.lot_size:
        raise ValueError("pair paper quantities must respect the configured lot size")
    if direction == 0 and (quantity_y or quantity_x):
        raise ValueError("flat pair paper state cannot retain leg quantities")
    if direction == 1 and not (quantity_y > 0 and quantity_x < 0):
        raise ValueError("positive pair direction requires long Y and short X")
    if direction == -1 and not (quantity_y < 0 and quantity_x > 0):
        raise ValueError("negative pair direction requires short Y and long X")
    cash = float(state.get("cash") or 0.0)
    starting_nav = float(state.get("nav") or 0.0)
    high_water_mark = float(state.get("high_water_mark") or starting_nav)
    if min(cash, starting_nav, high_water_mark) <= 0:
        raise ValueError("pair paper cash, NAV, and high-water mark must be positive")
    holding_days = int(state.get("holding_days") or 0)
    status = str(state.get("status") or "active")
    if status not in {"active", "liquidation_pending"}:
        raise ValueError("pair paper step requires an active or liquidation-pending portfolio")

    action = "hold"
    reason = "no_signal"
    target_direction = direction
    if direction:
        if status == "liquidation_pending":
            action, reason, target_direction = "exit", "risk_liquidation", 0
        elif not eligible:
            action, reason, target_direction = "exit", "cointegration_breakdown", 0
        elif abs(zscore) >= config.stop_zscore:
            action, reason, target_direction = "exit", "zscore_stop", 0
        elif abs(zscore) <= config.exit_zscore:
            action, reason, target_direction = "exit", "mean_reversion", 0
        elif holding_days >= config.max_holding_days:
            action, reason, target_direction = "exit", "max_holding_days", 0
        else:
            reason = "position_held"
    elif eligible and config.entry_zscore <= abs(zscore) < config.stop_zscore:
        action = "entry"
        reason = "negative_spread" if zscore < 0 else "positive_spread"
        target_direction = 1 if zscore < 0 else -1

    targets = {leg_y: quantity_y, leg_x: quantity_x}
    if action == "exit":
        targets = {leg_y: 0, leg_x: 0}
    elif action == "entry":
        gross = starting_nav * config.pair_gross_fraction
        y_weight = 1.0 / (1.0 + abs(hedge_ratio))
        x_weight = 1.0 - y_weight
        targets = {
            leg_y: target_direction
            * floor(gross * y_weight / float(prices.loc[signal_date, leg_y]) / config.lot_size)
            * config.lot_size,
            leg_x: -target_direction
            * floor(gross * x_weight / float(prices.loc[signal_date, leg_x]) / config.lot_size)
            * config.lot_size,
        }
        if not targets[leg_y] or not targets[leg_x]:
            raise ValueError("pair paper capital is insufficient for two lot-sized legs")

    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    rejection: str | None = None
    deltas = {leg_y: targets[leg_y] - quantity_y, leg_x: targets[leg_x] - quantity_x}
    execution_rows: list[dict[str, Any]] = []
    if action in {"entry", "exit"}:
        for instrument in (leg_y, leg_x):
            delta = int(deltas[instrument])
            side: Literal["buy", "sell"] = "buy" if delta > 0 else "sell"
            execution = _execution_price(minute, trade_date, instrument, side, config.slippage)
            if execution is None:
                rejection = f"{instrument}:missing_valid_minute_execution_window"
                break
            execution_price, minute_volume = execution
            opening_short = (
                targets[instrument] < 0 and (quantity_y if instrument == leg_y else quantity_x) >= 0
            )
            denied = _trade_allowed(
                daily.loc[(trade_date, instrument)],
                side=side,
                execution_price=execution_price,
                opening_short=opening_short,
            )
            if denied:
                rejection = f"{instrument}:{denied}"
                break
            capacity = (
                floor(minute_volume * config.max_volume_participation / config.lot_size)
                * config.lot_size
            )
            if min(1.0, capacity / abs(delta)) < config.min_capacity_fill_ratio:
                rejection = f"{instrument}:insufficient_minute_capacity"
                break
            execution_rows.append(
                {
                    "instrument": instrument,
                    "side": side,
                    "delta": delta,
                    "price": float(execution_price),
                }
            )
        if rejection:
            for instrument in (leg_y, leg_x):
                delta = int(deltas[instrument])
                orders.append(
                    {
                        "instrument": instrument,
                        "leg": "y" if instrument == leg_y else "x",
                        "side": "buy" if delta > 0 else "sell",
                        "requested_quantity": abs(delta),
                        "target_quantity": int(targets[instrument]),
                        "status": "rejected",
                        "reason": rejection,
                    }
                )
            targets = {leg_y: quantity_y, leg_x: quantity_x}
            target_direction = direction
        else:
            rate = config.open_cost if action == "entry" else config.close_cost
            for item in execution_rows:
                gross_value = abs(float(item["delta"]) * float(item["price"]))
                fee = max(config.min_commission, gross_value * rate)
                cash -= float(item["delta"]) * float(item["price"]) + fee
                orders.append(
                    {
                        "instrument": item["instrument"],
                        "leg": "y" if item["instrument"] == leg_y else "x",
                        "side": item["side"],
                        "requested_quantity": abs(int(item["delta"])),
                        "target_quantity": int(targets[str(item["instrument"])]),
                        "status": "filled",
                        "reason": None,
                    }
                )
                fills.append(
                    {
                        "instrument": item["instrument"],
                        "fill_time": f"{trade_date.date().isoformat()}T10:00:00",
                        "quantity": abs(int(item["delta"])),
                        "price": float(item["price"]),
                        "gross_value": gross_value,
                        "fee": fee,
                        "slippage": config.slippage,
                    }
                )

    quantity_y = int(targets[leg_y])
    quantity_x = int(targets[leg_x])
    position_direction = int(target_direction)
    closing_prices = {
        leg_y: float(prices.loc[trade_date, leg_y]),
        leg_x: float(prices.loc[trade_date, leg_x]),
    }
    short_value = sum(
        abs(quantity) * closing_prices[instrument]
        for instrument, quantity in ((leg_y, quantity_y), (leg_x, quantity_x))
        if quantity < 0
    )
    borrow_cost = short_value * config.annual_borrow_rate / 252.0
    cash -= borrow_cost
    signed_market_value = quantity_y * closing_prices[leg_y] + quantity_x * closing_prices[leg_x]
    nav = cash + signed_market_value
    if nav <= 0:
        raise ValueError("pair paper NAV must remain positive")
    long_value = sum(
        quantity * closing_prices[instrument]
        for instrument, quantity in ((leg_y, quantity_y), (leg_x, quantity_x))
        if quantity > 0
    )
    high_water_mark = max(high_water_mark, nav)
    drawdown = nav / high_water_mark - 1.0
    daily_return = nav / starting_nav - 1.0
    total_fees = sum(float(item["fee"]) for item in fills)
    gross_traded = sum(float(item["gross_value"]) for item in fills)
    next_status = "active"
    risk_events: list[dict[str, Any]] = []
    if drawdown <= -config.max_drawdown:
        next_status = "liquidation_pending" if position_direction else "paused"
        risk_events.append(
            {
                "severity": "critical",
                "event_type": "drawdown",
                "rule": "max_drawdown",
                "observed": abs(drawdown),
                "limit_value": config.max_drawdown,
                "details": {"action": next_status},
            }
        )
    elif status == "liquidation_pending":
        next_status = "paused" if position_direction == 0 else "liquidation_pending"
    entry_nav = state.get("entry_nav")
    if action == "entry" and not rejection:
        entry_nav = nav
    elif action == "exit" and not rejection:
        entry_nav = None
    next_holding_days = 0
    if position_direction:
        next_holding_days = 1 if direction == 0 else holding_days + 1

    return {
        "status": "ok",
        "as_of_date": signal_date.date().isoformat(),
        "trade_date": trade_date.date().isoformat(),
        "leg_y": leg_y,
        "leg_x": leg_x,
        "action": action,
        "reason": reason,
        "rejection": rejection,
        "orders": orders,
        "fills": fills,
        "state": {
            "status": next_status,
            "cash": cash,
            "nav": nav,
            "high_water_mark": high_water_mark,
            "position_direction": position_direction,
            "quantity_y": quantity_y,
            "quantity_x": quantity_x,
            "entry_nav": entry_nav,
            "holding_days": next_holding_days,
        },
        "closing_prices": closing_prices,
        "metrics": {
            "zscore": zscore,
            "correlation": evidence.correlation,
            "cointegration_pvalue": evidence.cointegration_pvalue,
            "hedge_ratio": hedge_ratio,
            "eligible": eligible,
            "daily_return": daily_return,
            "drawdown": drawdown,
            "long_value": long_value,
            "short_value": short_value,
            "gross_exposure": (long_value + short_value) / nav,
            "net_exposure": signed_market_value / nav,
            "turnover": gross_traded / starting_nav,
            "fees": total_fees,
            "borrow_cost": borrow_cost,
            "atomic_pair_execution_enforced": True,
            "shortability_enforced": True,
            "minute_execution_enforced": True,
        },
        "risk_events": risk_events,
    }


def run_pair_robustness_suite(
    daily_market: pd.DataFrame,
    minute_market: pd.DataFrame,
    *,
    leg_y: str,
    leg_x: str,
    config: PairTradingConfig | None = None,
) -> dict[str, Any]:
    config = config or PairTradingConfig()
    scenarios = {
        "base": config,
        "double_costs": replace(
            config,
            open_cost=config.open_cost * 2.0,
            close_cost=config.close_cost * 2.0,
            slippage=config.slippage * 2.0,
            annual_borrow_rate=config.annual_borrow_rate * 2.0,
        ),
        "lower_entry": replace(
            config,
            entry_zscore=max(config.exit_zscore + 0.05, config.entry_zscore * 0.90),
        ),
        "higher_entry": replace(
            config,
            entry_zscore=min(config.stop_zscore - 0.05, config.entry_zscore * 1.10),
        ),
    }
    results: list[dict[str, Any]] = []
    for name, scenario in scenarios.items():
        metrics = run_pair_backtest(
            daily_market,
            minute_market,
            leg_y=leg_y,
            leg_x=leg_x,
            config=scenario,
        )["metrics"]
        passed = (
            metrics["open_position_at_end"] is False
            and abs(float(metrics["max_drawdown"])) <= config.max_drawdown
            and metrics["sharpe_ratio"] is not None
            and float(metrics["sharpe_ratio"]) >= config.min_sharpe_ratio
            and int(metrics["closed_trade_count"]) >= config.min_closed_trades
            and float(metrics["rolling_cointegration_pass_rate"])
            >= config.min_rolling_cointegration_pass_rate
            and float(metrics["capacity_fill_ratio"]) >= config.min_capacity_fill_ratio
        )
        results.append(
            {
                "name": name,
                "passed": passed,
                "total_return": metrics["total_return"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "closed_trade_count": metrics["closed_trade_count"],
                "rolling_cointegration_pass_rate": metrics["rolling_cointegration_pass_rate"],
                "capacity_fill_ratio": metrics["capacity_fill_ratio"],
            }
        )
    passed_count = sum(bool(item["passed"]) for item in results)
    return {
        "scenarios": results,
        "scenario_count": len(results),
        "passed_count": passed_count,
        "pass_rate": passed_count / len(results),
        "passed": passed_count / len(results) >= config.min_robustness_pass_rate,
    }
