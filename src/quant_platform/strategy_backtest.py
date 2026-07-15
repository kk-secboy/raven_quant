from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .factor_evaluator import normalize_series
from .portfolio_optimizer import optimize_benchmark_relative_weights


def compose_factor_scores(
    factors: Sequence[tuple[pd.Series | pd.DataFrame, float, int]],
) -> pd.Series:
    if not factors:
        raise ValueError("at least one factor is required")
    normalized = []
    for index, (values, weight, direction) in enumerate(factors):
        series = normalize_series(values, f"factor_{index}")

        def cross_section(group: pd.Series) -> pd.Series:
            lower, upper = group.quantile([0.01, 0.99])
            clipped = group.clip(lower, upper)
            std = clipped.std(ddof=0)
            return (clipped - clipped.mean()) / std if std > 0 else clipped * 0.0

        zscore = series.groupby(level="datetime", group_keys=False).apply(cross_section)
        normalized.append(zscore * float(weight) * int(direction))
    frame = pd.concat(normalized, axis=1, join="inner").dropna()
    if frame.empty:
        raise ValueError("factor value artifacts have no common observations")
    return frame.sum(axis=1).rename("score")


def build_governed_signal(
    scores: pd.Series | pd.DataFrame,
    *,
    topk: int,
    liquidity_amount: pd.Series | pd.DataFrame | None = None,
    industry_memberships: pd.DataFrame | None = None,
    benchmark_weights: pd.DataFrame | None = None,
    style_exposures: pd.DataFrame | None = None,
    max_industry_weight: float = 1.0,
    max_industry_deviation: float = 1.0,
    min_average_daily_amount: float = 0.0,
    liquidity_lookback_days: int = 20,
) -> pd.Series:
    """Build the point-in-time constrained signal consumed by Qlib's strategy engine."""
    if topk < 1:
        raise ValueError("topk must be positive")
    score = normalize_series(scores, "score")
    historical_amount = (
        normalize_series(liquidity_amount, "liquidity_amount")
        if liquidity_amount is not None
        else None
    )
    rolling_amount = _rolling_liquidity(historical_amount, liquidity_lookback_days)
    memberships = _normalize_industry_memberships(industry_memberships)
    benchmark_frame = _normalize_benchmark_weights(benchmark_weights)
    style_frame = _normalize_style_exposures(style_exposures)
    rows: list[pd.Series] = []
    for timestamp, daily_score in score.groupby(level="datetime", sort=True):
        cross_section = daily_score.droplevel("datetime").dropna()
        styles = _styles_at(style_frame, timestamp, cross_section.index)
        ranking = _neutralize_size_score(cross_section, styles).sort_values(ascending=False)
        if min_average_daily_amount > 0:
            daily_liquidity = (
                rolling_amount.loc[timestamp]
                if rolling_amount is not None and timestamp in rolling_amount.index
                else pd.Series(dtype=float)
            )
            eligible = daily_liquidity.reindex(ranking.index).fillna(0.0)
            ranking = ranking[eligible >= min_average_daily_amount]
        if len(ranking) < topk:
            continue
        daily_benchmark = _benchmark_weights_at(benchmark_frame, timestamp)
        exposure_universe = ranking.index.union(daily_benchmark.index)
        all_industries = _industries_at(memberships, timestamp, exposure_universe)
        benchmark_industries = (
            daily_benchmark.groupby(all_industries.reindex(daily_benchmark.index)).sum()
            if not daily_benchmark.empty
            else pd.Series(dtype=float)
        )
        selected = _select_with_industry_cap(
            list(ranking.index),
            all_industries.reindex(ranking.index),
            topk=topk,
            max_industry_weight=max_industry_weight,
            benchmark_industries=benchmark_industries,
            max_industry_deviation=max_industry_deviation,
            require_industry=memberships is not None,
        )
        if len(selected) < topk:
            continue
        governed = ranking.reindex(selected)
        governed.index = pd.MultiIndex.from_product(
            [[timestamp], governed.index], names=["datetime", "instrument"]
        )
        rows.append(governed)
    if not rows:
        raise ValueError("governed signal has no eligible trading dates")
    return pd.concat(rows).sort_index().rename("score")


def simulate_long_only_topk(
    scores: pd.Series | pd.DataFrame,
    forward_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    topk: int,
    n_drop: int,
    max_position_weight: float,
    max_daily_turnover: float,
    open_cost: float,
    close_cost: float,
    market_amount: pd.Series | pd.DataFrame | None = None,
    liquidity_amount: pd.Series | pd.DataFrame | None = None,
    market_controls: pd.DataFrame | None = None,
    industry_memberships: pd.DataFrame | None = None,
    max_industry_weight: float = 1.0,
    benchmark_weights: pd.DataFrame | None = None,
    style_exposures: pd.DataFrame | None = None,
    max_industry_deviation: float = 1.0,
    max_size_deviation: float = 10.0,
    portfolio_construction: str = "topk_equal_weight",
    optimizer_alpha_weight: float = 0.05,
    optimizer_tracking_penalty: float = 1.0,
    optimizer_turnover_penalty: float = 0.10,
    min_average_daily_amount: float = 0.0,
    liquidity_lookback_days: int = 20,
    min_commission: float = 0.0,
    portfolio_notional: float = 1_000_000.0,
    max_volume_participation: float = 0.01,
    execution_risk_enabled: bool = False,
    max_daily_loss: float = 0.03,
    stop_loss: float = 0.07,
    take_profit_partial: float = 0.12,
    take_profit_partial_fraction: float = 0.50,
    take_profit: float = 0.20,
    max_drawdown_reduce: float = 0.10,
    max_drawdown_liquidate: float = 0.15,
    drawdown_reduction_exposure: float = 0.50,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if topk < 1 or n_drop < 0 or n_drop > topk:
        raise ValueError("topk and n_drop are invalid")
    if max_position_weight <= 0 or max_daily_turnover <= 0:
        raise ValueError("position and turnover limits must be positive")
    if not 0 < max_industry_weight <= 1:
        raise ValueError("max industry weight must be between zero and one")
    if not 0 <= max_industry_deviation <= 1 or max_size_deviation < 0:
        raise ValueError("benchmark exposure limits are invalid")
    if portfolio_construction not in {"topk_equal_weight", "benchmark_relative_qp"}:
        raise ValueError("portfolio construction method is invalid")
    if min(optimizer_alpha_weight, optimizer_tracking_penalty, optimizer_turnover_penalty) < 0:
        raise ValueError("optimizer objective weights must be non-negative")
    if portfolio_notional <= 0 or not 0 < max_volume_participation <= 1:
        raise ValueError("capacity notional and volume participation must be positive")
    if min_average_daily_amount < 0 or liquidity_lookback_days < 2 or min_commission < 0:
        raise ValueError("liquidity and commission settings are invalid")
    if execution_risk_enabled:
        if not 0 < stop_loss or not 0 < max_daily_loss:
            raise ValueError("loss thresholds must be positive")
        if not 0 < take_profit_partial < take_profit:
            raise ValueError("partial take-profit must be below final take-profit")
        if not 0 < take_profit_partial_fraction < 1:
            raise ValueError("partial take-profit fraction must be between zero and one")
        if not 0 < max_drawdown_reduce < max_drawdown_liquidate:
            raise ValueError("drawdown reduction must be below liquidation")
        if not 0 < drawdown_reduction_exposure < 1:
            raise ValueError("drawdown reduction exposure must be between zero and one")
    score = normalize_series(scores, "score")
    label = normalize_series(forward_returns, "return")
    amount = normalize_series(market_amount, "amount") if market_amount is not None else None
    historical_amount = (
        normalize_series(liquidity_amount, "liquidity_amount")
        if liquidity_amount is not None
        else None
    )
    rolling_amount = _rolling_liquidity(historical_amount, liquidity_lookback_days)
    controls = _normalize_market_controls(market_controls)
    memberships = _normalize_industry_memberships(industry_memberships)
    benchmark_weight_frame = _normalize_benchmark_weights(benchmark_weights)
    style_frame = _normalize_style_exposures(style_exposures)
    if portfolio_construction == "benchmark_relative_qp" and any(
        item is None for item in (memberships, benchmark_weight_frame, style_frame)
    ):
        raise ValueError(
            "benchmark-relative optimization requires industry, benchmark, and style metadata"
        )
    observations = pd.concat([score, label], axis=1, join="inner").dropna()
    benchmark = benchmark_returns.copy()
    benchmark.index = pd.to_datetime(benchmark.index).tz_localize(None)
    benchmark = pd.to_numeric(benchmark, errors="coerce").dropna().sort_index()

    current = pd.Series(dtype=float)
    daily_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    blocked_buy_orders = 0
    blocked_sell_orders = 0
    liquidity_excluded_observations = 0
    position_returns: dict[str, float] = {}
    take_profit_stages: dict[str, int] = {}
    running_nav = 1.0
    high_water_mark = 1.0
    previous_net_return = 0.0
    portfolio_risk_state = "active"
    stop_loss_exit_count = 0
    partial_take_profit_exit_count = 0
    full_take_profit_exit_count = 0
    drawdown_reduction_count = 0
    drawdown_liquidation_count = 0
    daily_loss_pause_count = 0
    optimizer_objectives: list[float] = []
    optimizer_tracking_proxies: list[float] = []
    optimizer_active_shares: list[float] = []
    optimizer_expected_turnovers: list[float] = []
    optimizer_industry_deviations: list[float] = []
    optimizer_size_deviations: list[float] = []
    optimizer_iterations: list[int] = []
    for timestamp, group in observations.groupby(level="datetime", sort=True):
        if timestamp not in benchmark.index or len(group) < topk:
            continue
        by_instrument = group.droplevel("datetime")
        daily_styles = _styles_at(style_frame, timestamp, by_instrument.index)
        neutral_score = _neutralize_size_score(by_instrument["score"], daily_styles)
        ranking = neutral_score.sort_values(ascending=False)
        if min_average_daily_amount > 0:
            daily_liquidity = (
                rolling_amount.loc[timestamp]
                if rolling_amount is not None and timestamp in rolling_amount.index
                else pd.Series(dtype=float)
            )
            eligible = daily_liquidity.reindex(ranking.index).fillna(0.0)
            liquidity_excluded_observations += int((eligible < min_average_daily_amount).sum())
            ranking = ranking[eligible >= min_average_daily_amount]
            by_instrument = by_instrument.reindex(ranking.index)
            if len(ranking) < topk:
                continue
        daily_benchmark = _benchmark_weights_at(benchmark_weight_frame, timestamp)
        buffer = set(ranking.head(topk + n_drop).index)
        retained = [instrument for instrument in current.index if instrument in buffer]
        candidate_order = retained + [
            instrument for instrument in ranking.index if instrument not in retained
        ]
        exposure_universe = ranking.index.union(daily_benchmark.index)
        all_industries = _industries_at(memberships, timestamp, exposure_universe)
        industries = all_industries.reindex(ranking.index)
        benchmark_industries = (
            daily_benchmark.groupby(all_industries.reindex(daily_benchmark.index)).sum()
            if not daily_benchmark.empty
            else pd.Series(dtype=float)
        )
        benchmark_styles = _styles_at(style_frame, timestamp, daily_benchmark.index)
        benchmark_size = (
            float(daily_benchmark.dot(benchmark_styles.reindex(daily_benchmark.index)))
            if not daily_benchmark.empty and style_frame is not None
            else None
        )
        selected = _select_with_industry_cap(
            candidate_order,
            industries,
            topk=topk,
            max_industry_weight=max_industry_weight,
            benchmark_industries=benchmark_industries,
            max_industry_deviation=max_industry_deviation,
            require_industry=memberships is not None,
        )
        optimizer_result = None
        if portfolio_construction == "benchmark_relative_qp":
            if benchmark_size is None:
                raise ValueError("benchmark-relative optimization has no benchmark style exposure")
            optimizer_result = optimize_benchmark_relative_weights(
                ranking.reindex(selected),
                daily_benchmark,
                current,
                industries=all_industries.reindex(selected),
                benchmark_industry_weights=benchmark_industries,
                style_exposures=daily_styles.reindex(selected),
                benchmark_style_exposure=benchmark_size,
                max_position_weight=max_position_weight,
                max_industry_weight=max_industry_weight,
                max_industry_deviation=max_industry_deviation,
                max_size_deviation=max_size_deviation,
                alpha_weight=optimizer_alpha_weight,
                tracking_penalty=optimizer_tracking_penalty,
                turnover_penalty=optimizer_turnover_penalty,
            )
            target = optimizer_result.weights
            optimizer_objectives.append(optimizer_result.objective)
            optimizer_tracking_proxies.append(optimizer_result.tracking_risk_proxy)
            optimizer_active_shares.append(optimizer_result.active_share)
            optimizer_expected_turnovers.append(optimizer_result.expected_turnover)
            if optimizer_result.max_industry_deviation is not None:
                optimizer_industry_deviations.append(
                    optimizer_result.max_industry_deviation
                )
            if optimizer_result.size_deviation is not None:
                optimizer_size_deviations.append(optimizer_result.size_deviation)
            optimizer_iterations.append(optimizer_result.iterations)
        else:
            target_weight = min(1.0 / topk, max_position_weight)
            target = pd.Series(target_weight, index=selected, dtype=float)
        daily_stop_loss_exits = 0
        daily_partial_take_profit_exits = 0
        daily_full_take_profit_exits = 0
        portfolio_risk_action: str | None = None
        daily_loss_breached = False
        stop_loss_candidates: set[str] = set()
        partial_candidates: set[str] = set()
        full_take_profit_candidates: set[str] = set()
        if execution_risk_enabled:
            opening_drawdown = running_nav / high_water_mark - 1.0
            if (
                portfolio_risk_state != "liquidated"
                and opening_drawdown <= -max_drawdown_liquidate
            ):
                portfolio_risk_state = "liquidated"
                portfolio_risk_action = "liquidate"
                drawdown_liquidation_count += 1
            elif (
                portfolio_risk_state == "active"
                and opening_drawdown <= -max_drawdown_reduce
            ):
                portfolio_risk_state = "reduced"
                portfolio_risk_action = "reduce"
                drawdown_reduction_count += 1
            daily_loss_breached = previous_net_return <= -max_daily_loss
            if daily_loss_breached:
                daily_loss_pause_count += 1

            if portfolio_risk_state == "liquidated":
                target = pd.Series(dtype=float)
            elif portfolio_risk_state == "reduced":
                # A historical replay has no human recovery action. On first breach,
                # reduce the existing book; afterwards keep that reduced book and
                # allow only risk exits instead of silently resuming rebalancing.
                reduction = drawdown_reduction_exposure if portfolio_risk_action else 1.0
                target = current.mul(reduction)

            for instrument in current.index:
                position_return = float(position_returns.get(str(instrument), 0.0))
                if position_return <= -stop_loss:
                    target.loc[instrument] = 0.0
                    stop_loss_candidates.add(str(instrument))
                elif position_return >= take_profit:
                    target.loc[instrument] = 0.0
                    full_take_profit_candidates.add(str(instrument))
                elif (
                    position_return >= take_profit_partial
                    and take_profit_stages.get(str(instrument), 0) < 1
                ):
                    target.loc[instrument] = min(
                        float(target.get(instrument, 0.0)),
                        float(current[instrument]) * (1.0 - take_profit_partial_fraction),
                    )
                    partial_candidates.add(str(instrument))
                elif position_return >= take_profit_partial:
                    target.loc[instrument] = min(
                        float(target.get(instrument, 0.0)), float(current[instrument])
                    )
            if daily_loss_breached:
                for instrument in target.index:
                    target.loc[instrument] = min(
                        float(target[instrument]), float(current.get(instrument, 0.0))
                    )
            target = target[target > 1e-12]
        union = current.index.union(target.index)
        current_aligned = current.reindex(union, fill_value=0.0)
        target_aligned = target.reindex(union, fill_value=0.0)
        delta = target_aligned - current_aligned
        desired_notional = float(delta.abs().sum()) * portfolio_notional
        capacity_notional = desired_notional
        executable_delta = delta
        if amount is not None:
            try:
                daily_amount = amount.xs(timestamp, level="datetime")
            except KeyError:
                daily_amount = pd.Series(dtype=float)
            limits = (
                daily_amount.reindex(union)
                .fillna(0.0)
                .clip(lower=0.0)
                .mul(max_volume_participation)
                .div(portfolio_notional)
            )
            executable_delta = delta.clip(lower=-limits, upper=limits)
            capacity_notional = float(executable_delta.abs().sum()) * portfolio_notional
        if controls is not None:
            try:
                daily_controls = controls.xs(timestamp, level="datetime").reindex(union)
            except KeyError:
                daily_controls = pd.DataFrame(index=union)
            paused = daily_controls.get("paused", pd.Series(0.0, index=union)).fillna(1.0)
            open_price = daily_controls.get("open", pd.Series(np.nan, index=union))
            up_limit = daily_controls.get("up_limit", pd.Series(np.nan, index=union))
            down_limit = daily_controls.get("down_limit", pd.Series(np.nan, index=union))
            buy_blocked = (executable_delta > 0) & (
                (paused >= 0.5)
                | open_price.isna()
                | (up_limit.notna() & (open_price >= up_limit * (1.0 - 1e-6)))
            )
            sell_blocked = (executable_delta < 0) & (
                (paused >= 0.5)
                | open_price.isna()
                | (down_limit.notna() & (open_price <= down_limit * (1.0 + 1e-6)))
            )
            blocked_buy_orders += int(buy_blocked.sum())
            blocked_sell_orders += int(sell_blocked.sum())
            executable_delta = executable_delta.mask(buy_blocked | sell_blocked, 0.0)
            capacity_notional = float(executable_delta.abs().sum()) * portfolio_notional
        capacity_fill_ratio = (
            min(1.0, capacity_notional / desired_notional) if desired_notional > 0 else 1.0
        )
        executable_target = current_aligned + executable_delta
        cash_delta = (1.0 - executable_target.sum()) - (1.0 - current_aligned.sum())
        raw_turnover = 0.5 * (float(executable_delta.abs().sum()) + abs(float(cash_delta)))
        scale = min(1.0, max_daily_turnover / raw_turnover) if raw_turnover > 0 else 1.0
        next_weights = current_aligned + executable_delta * scale
        next_weights = next_weights[next_weights.abs() > 1e-12]
        executed_delta = next_weights.reindex(union, fill_value=0.0) - current_aligned
        next_cash = 1.0 - float(next_weights.sum())
        current_cash = 1.0 - float(current_aligned.sum())
        turnover = 0.5 * (float(executed_delta.abs().sum()) + abs(next_cash - current_cash))
        buy_notionals = executed_delta.clip(lower=0) * portfolio_notional
        sell_notionals = -executed_delta.clip(upper=0) * portfolio_notional
        buy_cost = float(
            buy_notionals[buy_notionals > 0]
            .mul(open_cost)
            .clip(lower=min_commission)
            .sum()
            / portfolio_notional
        )
        sell_cost = float(
            sell_notionals[sell_notionals > 0]
            .mul(close_cost)
            .clip(lower=min_commission)
            .sum()
            / portfolio_notional
        )
        cost = buy_cost + sell_cost
        asset_returns = by_instrument["return"].reindex(next_weights.index).fillna(0.0)
        gross_return = float(next_weights.dot(asset_returns))
        net_return = gross_return - cost
        if execution_risk_enabled:
            daily_stop_loss_exits = sum(
                float(executed_delta.get(instrument, 0.0)) < -1e-12
                for instrument in stop_loss_candidates
            )
            daily_partial_take_profit_exits = sum(
                float(executed_delta.get(instrument, 0.0)) < -1e-12
                for instrument in partial_candidates
            )
            daily_full_take_profit_exits = sum(
                float(executed_delta.get(instrument, 0.0)) < -1e-12
                for instrument in full_take_profit_candidates
            )
            for instrument in partial_candidates:
                if float(executed_delta.get(instrument, 0.0)) < -1e-12:
                    take_profit_stages[instrument] = 1
            next_position_returns: dict[str, float] = {}
            next_take_profit_stages: dict[str, int] = {}
            for instrument, next_weight in next_weights.items():
                key = str(instrument)
                old_weight = float(current_aligned.get(instrument, 0.0))
                added_weight = max(0.0, float(executed_delta.get(instrument, 0.0)))
                old_return = float(position_returns.get(key, 0.0))
                if old_weight > 0 and added_weight > 0 and old_return > -1.0:
                    cost_basis = old_weight / (1.0 + old_return) + added_weight
                    pre_return = float(next_weight) / cost_basis - 1.0 if cost_basis > 0 else 0.0
                elif old_weight > 0:
                    pre_return = old_return
                else:
                    pre_return = 0.0
                asset_return = float(asset_returns.get(instrument, 0.0))
                next_position_returns[key] = (1.0 + pre_return) * (1.0 + asset_return) - 1.0
                next_take_profit_stages[key] = int(take_profit_stages.get(key, 0))
            position_returns = next_position_returns
            take_profit_stages = next_take_profit_stages
            running_nav *= 1.0 + net_return
            high_water_mark = max(high_water_mark, running_nav)
            previous_net_return = net_return
            stop_loss_exit_count += daily_stop_loss_exits
            partial_take_profit_exit_count += daily_partial_take_profit_exits
            full_take_profit_exit_count += daily_full_take_profit_exits
        benchmark_return = float(benchmark.loc[timestamp])
        portfolio_industries = (
            next_weights.groupby(all_industries.reindex(next_weights.index)).sum()
            if memberships is not None and not next_weights.empty
            else pd.Series(dtype=float)
        )
        industry_labels = portfolio_industries.index.union(benchmark_industries.index)
        industry_deviation = (
            float(
                (
                    portfolio_industries.reindex(industry_labels, fill_value=0.0)
                    - benchmark_industries.reindex(industry_labels, fill_value=0.0)
                ).abs().max()
            )
            if len(industry_labels)
            else None
        )
        portfolio_size = float(
            next_weights.dot(daily_styles.reindex(next_weights.index).fillna(0.0))
        )
        size_deviation = (
            abs(portfolio_size - benchmark_size) if benchmark_size is not None else None
        )
        daily_rows.append(
            {
                "datetime": timestamp,
                "gross_return": gross_return,
                "net_return": net_return,
                "benchmark_return": benchmark_return,
                "excess_return": net_return - benchmark_return,
                "turnover": turnover,
                "cost": cost,
                "cash_weight": next_cash,
                "positions": int(len(next_weights)),
                "capacity_requested_notional": desired_notional,
                "capacity_executable_notional": capacity_notional,
                "capacity_fill_ratio": capacity_fill_ratio,
                "blocked_buy_orders": int(buy_blocked.sum()) if controls is not None else 0,
                "blocked_sell_orders": int(sell_blocked.sum()) if controls is not None else 0,
                "max_industry_deviation": industry_deviation,
                "size_deviation": size_deviation,
                "stop_loss_exits": daily_stop_loss_exits,
                "partial_take_profit_exits": daily_partial_take_profit_exits,
                "full_take_profit_exits": daily_full_take_profit_exits,
                "daily_loss_breached": daily_loss_breached,
                "portfolio_risk_action": portfolio_risk_action,
                "portfolio_risk_state": portfolio_risk_state,
                "optimizer_objective": (
                    optimizer_result.objective if optimizer_result is not None else None
                ),
                "optimizer_tracking_risk_proxy": (
                    optimizer_result.tracking_risk_proxy
                    if optimizer_result is not None
                    else None
                ),
                "optimizer_active_share": (
                    optimizer_result.active_share if optimizer_result is not None else None
                ),
                "optimizer_expected_turnover": (
                    optimizer_result.expected_turnover
                    if optimizer_result is not None
                    else None
                ),
                "optimizer_industry_deviation": (
                    optimizer_result.max_industry_deviation
                    if optimizer_result is not None
                    else None
                ),
                "optimizer_size_deviation": (
                    optimizer_result.size_deviation
                    if optimizer_result is not None
                    else None
                ),
                "optimizer_iterations": (
                    optimizer_result.iterations if optimizer_result is not None else None
                ),
            }
        )
        for instrument, weight in next_weights.items():
            position_rows.append(
                {
                    "datetime": timestamp,
                    "instrument": instrument,
                    "industry": industries.get(instrument),
                    "weight": float(weight),
                }
            )
        current = next_weights

    daily = pd.DataFrame(daily_rows).set_index("datetime") if daily_rows else pd.DataFrame()
    positions = pd.DataFrame(
        position_rows, columns=["datetime", "instrument", "industry", "weight"]
    )
    if daily.empty or len(daily) < 20:
        raise ValueError("backtest has insufficient aligned trading days")
    nav = (1.0 + daily["net_return"]).cumprod()
    benchmark_nav = (1.0 + daily["benchmark_return"]).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    excess = daily["excess_return"]
    annualized_return = float(nav.iloc[-1] ** (252 / len(daily)) - 1.0)
    benchmark_annualized = float(benchmark_nav.iloc[-1] ** (252 / len(daily)) - 1.0)
    tracking_error = float(excess.std(ddof=1) * 252**0.5)
    information_ratio = (
        float(excess.mean() / excess.std(ddof=1) * 252**0.5) if excess.std(ddof=1) > 0 else None
    )
    net = daily["net_return"]
    annualized_volatility = float(net.std(ddof=1) * 252**0.5)
    sharpe_ratio = float(net.mean() / net.std(ddof=1) * 252**0.5) if net.std(ddof=1) > 0 else None
    downside = net.clip(upper=0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))) * 252**0.5)
    sortino_unbounded = downside_deviation == 0 and float(net.mean()) > 0
    sortino_ratio = (
        float(net.mean() * 252 / downside_deviation)
        if downside_deviation > 0
        else (100.0 if sortino_unbounded else None)
    )
    value_at_risk_95 = float(net.quantile(0.05))
    tail = net[net <= value_at_risk_95]
    expected_shortfall_95 = float(tail.mean()) if not tail.empty else value_at_risk_95
    gross_profit = float(net.clip(lower=0.0).sum())
    gross_loss = abs(float(net.clip(upper=0.0).sum()))
    requested_capacity = float(daily["capacity_requested_notional"].sum())
    executable_capacity = float(daily["capacity_executable_notional"].sum())
    capacity_fill_ratio = (
        min(1.0, executable_capacity / requested_capacity)
        if amount is not None and requested_capacity > 0
        else (1.0 if amount is not None else None)
    )
    max_drawdown = float(drawdown.min())
    industry_exposure = (
        positions.dropna(subset=["industry"])
        .groupby(["datetime", "industry"])["weight"]
        .sum()
        if not positions.empty and "industry" in positions
        else pd.Series(dtype=float)
    )
    metrics = {
        "annualized_return": annualized_return,
        "benchmark_annualized_return": benchmark_annualized,
        "annualized_excess_return": annualized_return - benchmark_annualized,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "max_drawdown": max_drawdown,
        "average_turnover": float(daily["turnover"].mean()),
        "total_cost": float(daily["cost"].sum()),
        "win_rate": float((excess > 0).mean()),
        "positive_day_rate": float((net > 0).mean()),
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "sortino_unbounded": sortino_unbounded,
        "calmar_ratio": (
            float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else None
        ),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else None,
        "max_daily_loss": float(net.min()),
        "value_at_risk_95": value_at_risk_95,
        "expected_shortfall_95": expected_shortfall_95,
        "capacity_notional": float(portfolio_notional),
        "max_volume_participation": float(max_volume_participation),
        "capacity_fill_ratio": capacity_fill_ratio,
        "liquidity_observations": int((daily["capacity_requested_notional"] > 0).sum())
        if amount is not None
        else 0,
        "max_position_weight": float(positions["weight"].max()) if not positions.empty else 0.0,
        "trading_days": int(len(daily)),
        "execution_model": "next_open",
        "blocked_buy_orders": blocked_buy_orders,
        "blocked_sell_orders": blocked_sell_orders,
        "market_controls_enforced": controls is not None,
        "industry_controls_enforced": memberships is not None,
        "max_industry_weight": float(industry_exposure.max())
        if not industry_exposure.empty
        else None,
        "max_industry_deviation": float(daily["max_industry_deviation"].dropna().max())
        if daily["max_industry_deviation"].notna().any()
        else None,
        "max_size_deviation": float(daily["size_deviation"].dropna().max())
        if daily["size_deviation"].notna().any()
        else None,
        "benchmark_weights_enforced": benchmark_weight_frame is not None,
        "size_neutralization_enforced": style_frame is not None,
        "liquidity_filter_enforced": min_average_daily_amount > 0
        and rolling_amount is not None,
        "min_average_daily_amount": float(min_average_daily_amount),
        "liquidity_excluded_observations": liquidity_excluded_observations,
        "min_commission": float(min_commission),
        "execution_risk_overlay_enforced": execution_risk_enabled,
        "stop_loss_exit_count": stop_loss_exit_count,
        "partial_take_profit_exit_count": partial_take_profit_exit_count,
        "full_take_profit_exit_count": full_take_profit_exit_count,
        "drawdown_reduction_count": drawdown_reduction_count,
        "drawdown_liquidation_count": drawdown_liquidation_count,
        "daily_loss_pause_count": daily_loss_pause_count,
        "final_portfolio_risk_state": portfolio_risk_state,
        "execution_risk_thresholds": (
            {
                "max_daily_loss": float(max_daily_loss),
                "stop_loss": float(stop_loss),
                "take_profit_partial": float(take_profit_partial),
                "take_profit_partial_fraction": float(take_profit_partial_fraction),
                "take_profit": float(take_profit),
                "max_drawdown_reduce": float(max_drawdown_reduce),
                "max_drawdown_liquidate": float(max_drawdown_liquidate),
                "drawdown_reduction_exposure": float(drawdown_reduction_exposure),
            }
            if execution_risk_enabled
            else None
        ),
        "portfolio_construction": portfolio_construction,
        "optimizer_days": len(optimizer_objectives),
        "optimizer_execution_replay_enforced": (
            portfolio_construction == "benchmark_relative_qp"
            and len(optimizer_objectives) == len(daily_rows)
        ),
        "optimizer_mean_objective": (
            float(np.mean(optimizer_objectives)) if optimizer_objectives else None
        ),
        "optimizer_max_tracking_risk_proxy": (
            max(optimizer_tracking_proxies) if optimizer_tracking_proxies else None
        ),
        "optimizer_mean_active_share": (
            float(np.mean(optimizer_active_shares)) if optimizer_active_shares else None
        ),
        "optimizer_mean_expected_turnover": (
            float(np.mean(optimizer_expected_turnovers))
            if optimizer_expected_turnovers
            else None
        ),
        "optimizer_max_industry_deviation": (
            max(optimizer_industry_deviations)
            if optimizer_industry_deviations
            else None
        ),
        "optimizer_max_size_deviation": (
            max(optimizer_size_deviations) if optimizer_size_deviations else None
        ),
        "optimizer_max_iterations": max(optimizer_iterations) if optimizer_iterations else None,
    }
    daily = daily.assign(nav=nav, benchmark_nav=benchmark_nav, drawdown=drawdown)
    return metrics, daily, positions


def run_robustness_suite(
    scores: pd.Series | pd.DataFrame,
    forward_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    config: dict[str, Any],
    market_amount: pd.Series | pd.DataFrame | None,
    liquidity_amount: pd.Series | pd.DataFrame | None = None,
    market_controls: pd.DataFrame | None = None,
    industry_memberships: pd.DataFrame | None = None,
    benchmark_weights: pd.DataFrame | None = None,
    style_exposures: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Rerun independent parameter and cost perturbations and fail closed on errors."""

    topk = int(config["topk"])
    n_drop = int(config["n_drop"])
    scenario_overrides = [
        (
            "double_cost",
            {
                "open_cost": float(config["open_cost"]) * 2,
                "close_cost": float(config["close_cost"]) * 2,
            },
        ),
        ("tight_turnover", {"max_daily_turnover": float(config["max_daily_turnover"]) * 0.75}),
        (
            "narrow_topk",
            {"topk": max(5, int(topk * 0.8)), "n_drop": min(n_drop, max(5, int(topk * 0.8)))},
        ),
        ("no_buffer", {"n_drop": 0}),
    ]
    scenarios: list[dict[str, Any]] = []
    for name, overrides in scenario_overrides:
        scenario_config = {
            "topk": topk,
            "n_drop": n_drop,
            "max_position_weight": float(config["max_position_weight"]),
            "max_daily_turnover": float(config["max_daily_turnover"]),
            "open_cost": float(config["open_cost"]),
            "close_cost": float(config["close_cost"]),
            "portfolio_notional": float(config["capacity_notional"]),
            "max_volume_participation": float(config["max_volume_participation"]),
            "max_industry_weight": float(config.get("max_industry_weight", 1.0)),
            "max_industry_deviation": float(config.get("max_industry_deviation", 1.0)),
            "max_size_deviation": float(config.get("max_size_deviation", 10.0)),
            "portfolio_construction": str(
                config.get("portfolio_construction", "topk_equal_weight")
            ),
            "optimizer_alpha_weight": float(config.get("optimizer_alpha_weight", 0.05)),
            "optimizer_tracking_penalty": float(
                config.get("optimizer_tracking_penalty", 1.0)
            ),
            "optimizer_turnover_penalty": float(
                config.get("optimizer_turnover_penalty", 0.10)
            ),
            "min_average_daily_amount": float(config.get("min_average_daily_amount", 0.0)),
            "liquidity_lookback_days": int(config.get("liquidity_lookback_days", 20)),
            "min_commission": float(config.get("min_commission", 0.0)),
            "execution_risk_enabled": True,
            "max_daily_loss": float(config.get("max_daily_loss", 0.03)),
            "stop_loss": float(config.get("stop_loss", 0.07)),
            "take_profit_partial": float(config.get("take_profit_partial", 0.12)),
            "take_profit_partial_fraction": float(
                config.get("take_profit_partial_fraction", 0.50)
            ),
            "take_profit": float(config.get("take_profit", 0.20)),
            "max_drawdown_reduce": float(config.get("max_drawdown_reduce", 0.10)),
            "max_drawdown_liquidate": float(
                config.get("max_drawdown_liquidate", 0.15)
            ),
            "drawdown_reduction_exposure": float(
                config.get("drawdown_reduction_exposure", 0.50)
            ),
            **overrides,
        }
        try:
            metrics, _daily, _positions = simulate_long_only_topk(
                scores,
                forward_returns,
                benchmark_returns,
                market_amount=market_amount,
                liquidity_amount=liquidity_amount,
                market_controls=market_controls,
                industry_memberships=industry_memberships,
                benchmark_weights=benchmark_weights,
                style_exposures=style_exposures,
                **scenario_config,
            )
            passed = all(
                (
                    metrics.get("annualized_excess_return") is not None
                    and float(metrics["annualized_excess_return"]) > 0,
                    metrics.get("tracking_error") is not None
                    and float(metrics["tracking_error"]) <= float(config["max_tracking_error"]),
                    metrics.get("max_drawdown") is not None
                    and abs(float(metrics["max_drawdown"])) <= float(config["max_drawdown"]),
                    metrics.get("average_turnover") is not None
                    and float(metrics["average_turnover"]) <= float(config["max_turnover"]),
                    metrics.get("information_ratio") is not None
                    and float(metrics["information_ratio"])
                    >= float(config["min_information_ratio"]),
                    metrics.get("sharpe_ratio") is not None
                    and float(metrics["sharpe_ratio"]) >= float(config["min_sharpe_ratio"]),
                    metrics.get("sortino_ratio") is not None
                    and float(metrics["sortino_ratio"]) >= float(config["min_sortino_ratio"]),
                    metrics.get("capacity_fill_ratio") is not None
                    and float(metrics["capacity_fill_ratio"])
                    >= float(config["min_capacity_fill_ratio"]),
                    "max_industry_deviation" not in config
                    or (
                        metrics.get("max_industry_deviation") is not None
                        and float(metrics["max_industry_deviation"])
                        <= float(config["max_industry_deviation"])
                    ),
                    "max_size_deviation" not in config
                    or (
                        metrics.get("max_size_deviation") is not None
                        and float(metrics["max_size_deviation"])
                        <= float(config["max_size_deviation"])
                    ),
                )
            )
            scenarios.append(
                {
                    "name": name,
                    "status": "passed" if passed else "failed",
                    "overrides": overrides,
                    "metrics": metrics,
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            scenarios.append(
                {"name": name, "status": "error", "overrides": overrides, "error": str(exc)}
            )
    successful = [item for item in scenarios if item["status"] != "error"]
    passed_count = sum(item["status"] == "passed" for item in scenarios)
    pass_rate = passed_count / len(scenarios) if scenarios else 0.0
    metric_rows = [item["metrics"] for item in successful]
    return {
        "scenario_count": len(scenarios),
        "passed_count": passed_count,
        "pass_rate": pass_rate,
        "passed": pass_rate >= float(config["min_robustness_pass_rate"]),
        "worst_annualized_excess_return": min(
            (float(item["annualized_excess_return"]) for item in metric_rows), default=None
        ),
        "worst_max_drawdown": min(
            (float(item["max_drawdown"]) for item in metric_rows), default=None
        ),
        "scenarios": scenarios,
    }


def run_rolling_suite(
    scores: pd.Series | pd.DataFrame,
    forward_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    config: dict[str, Any],
    market_amount: pd.Series | pd.DataFrame | None,
    liquidity_amount: pd.Series | pd.DataFrame | None = None,
    market_controls: pd.DataFrame | None = None,
    industry_memberships: pd.DataFrame | None = None,
    benchmark_weights: pd.DataFrame | None = None,
    style_exposures: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Reset and rerun the strategy over overlapping out-of-sample windows."""

    window_days = int(config.get("rolling_window_days", 252))
    step_days = int(config.get("rolling_step_days", 63))
    if window_days < 20 or step_days < 1 or step_days > window_days:
        raise ValueError("rolling window and step are invalid")
    benchmark = benchmark_returns.copy()
    benchmark.index = pd.to_datetime(benchmark.index).tz_localize(None)
    dates = pd.DatetimeIndex(sorted(benchmark.dropna().index.unique()))
    windows: list[dict[str, Any]] = []
    for offset in range(0, max(0, len(dates) - window_days + 1), step_days):
        selected = dates[offset : offset + window_days]
        if len(selected) < window_days:
            continue
        start, end = selected[0], selected[-1]
        try:
            metrics, _daily, _positions = simulate_long_only_topk(
                _slice_observations(scores, start, end),
                _slice_observations(forward_returns, start, end),
                benchmark.loc[start:end],
                market_amount=(
                    _slice_observations(market_amount, start, end)
                    if market_amount is not None
                    else None
                ),
                liquidity_amount=(
                    _slice_observations(liquidity_amount, start, end)
                    if liquidity_amount is not None
                    else None
                ),
                market_controls=(
                    _slice_observations(market_controls, start, end)
                    if market_controls is not None
                    else None
                ),
                industry_memberships=industry_memberships,
                benchmark_weights=benchmark_weights,
                style_exposures=style_exposures,
                **_simulation_config(config),
            )
            passed = _metrics_pass_risk_gate(metrics, config)
            windows.append(
                {
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "status": "passed" if passed else "failed",
                    "metrics": metrics,
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            windows.append(
                {
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "status": "error",
                    "error": str(exc),
                }
            )
    passed_count = sum(item["status"] == "passed" for item in windows)
    pass_rate = passed_count / len(windows) if windows else 0.0
    successful = [item["metrics"] for item in windows if "metrics" in item]
    minimum_windows = max(1, int(config.get("min_rolling_windows", 3)))
    return {
        "window_days": window_days,
        "step_days": step_days,
        "window_count": len(windows),
        "passed_count": passed_count,
        "pass_rate": pass_rate,
        "passed": len(windows) >= minimum_windows
        and pass_rate >= float(config.get("min_rolling_pass_rate", 0.60)),
        "worst_annualized_excess_return": min(
            (float(item["annualized_excess_return"]) for item in successful), default=None
        ),
        "worst_max_drawdown": min(
            (float(item["max_drawdown"]) for item in successful), default=None
        ),
        "windows": windows,
    }


def run_event_stress_suite(
    scores: pd.Series | pd.DataFrame,
    forward_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    config: dict[str, Any],
    market_amount: pd.Series | pd.DataFrame | None,
    liquidity_amount: pd.Series | pd.DataFrame | None = None,
    market_controls: pd.DataFrame | None = None,
    industry_memberships: pd.DataFrame | None = None,
    benchmark_weights: pd.DataFrame | None = None,
    style_exposures: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Rerun the worst non-overlapping benchmark windows as historical event stresses."""

    window_days = int(config.get("event_window_days", 20))
    event_count = int(config.get("event_count", 5))
    if window_days < 20 or event_count < 1:
        raise ValueError("event stress window and count are invalid")
    benchmark = benchmark_returns.copy()
    benchmark.index = pd.to_datetime(benchmark.index).tz_localize(None)
    benchmark = pd.to_numeric(benchmark, errors="coerce").dropna().sort_index()
    rolling = (1.0 + benchmark).rolling(window_days).apply(np.prod, raw=True) - 1.0
    candidates = rolling.dropna().sort_values()
    selected_ends: list[pd.Timestamp] = []
    for end in candidates.index:
        if all(
            abs(benchmark.index.get_loc(end) - benchmark.index.get_loc(item)) >= window_days
            for item in selected_ends
        ):
            selected_ends.append(end)
        if len(selected_ends) >= event_count:
            break
    events: list[dict[str, Any]] = []
    for end in selected_ends:
        end_index = int(benchmark.index.get_loc(end))
        start = benchmark.index[end_index - window_days + 1]
        try:
            metrics, daily, _positions = simulate_long_only_topk(
                _slice_observations(scores, start, end),
                _slice_observations(forward_returns, start, end),
                benchmark.loc[start:end],
                market_amount=(
                    _slice_observations(market_amount, start, end)
                    if market_amount is not None
                    else None
                ),
                liquidity_amount=(
                    _slice_observations(liquidity_amount, start, end)
                    if liquidity_amount is not None
                    else None
                ),
                market_controls=(
                    _slice_observations(market_controls, start, end)
                    if market_controls is not None
                    else None
                ),
                industry_memberships=industry_memberships,
                benchmark_weights=benchmark_weights,
                style_exposures=style_exposures,
                **_simulation_config(config),
            )
            strategy_return = float((1.0 + daily["net_return"]).prod() - 1.0)
            benchmark_return = float((1.0 + daily["benchmark_return"]).prod() - 1.0)
            underperformance = benchmark_return - strategy_return
            passed = abs(float(metrics["max_drawdown"])) <= float(
                config["max_drawdown"]
            ) and underperformance <= float(config.get("max_event_underperformance", 0.05))
            events.append(
                {
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "status": "passed" if passed else "failed",
                    "strategy_return": strategy_return,
                    "benchmark_return": benchmark_return,
                    "underperformance": underperformance,
                    "metrics": metrics,
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            events.append(
                {
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "status": "error",
                    "error": str(exc),
                }
            )
    passed_count = sum(item["status"] == "passed" for item in events)
    pass_rate = passed_count / len(events) if events else 0.0
    return {
        "window_days": window_days,
        "event_count": len(events),
        "passed_count": passed_count,
        "pass_rate": pass_rate,
        "passed": len(events) >= event_count
        and pass_rate >= float(config.get("min_event_stress_pass_rate", 0.60)),
        "events": events,
    }


def _slice_observations(
    values: pd.Series | pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series | pd.DataFrame:
    timestamps = pd.to_datetime(values.index.get_level_values("datetime")).tz_localize(None)
    return values.loc[(timestamps >= start) & (timestamps <= end)]


def _normalize_market_controls(values: pd.DataFrame | None) -> pd.DataFrame | None:
    if values is None:
        return None
    if not isinstance(values, pd.DataFrame):
        raise ValueError("market controls must be a DataFrame")
    required = {"open", "paused", "up_limit", "down_limit"}
    missing = required.difference(values.columns)
    if missing:
        raise ValueError(f"market controls are missing columns: {', '.join(sorted(missing))}")
    frame = values.loc[:, sorted(required)].copy()
    if not isinstance(frame.index, pd.MultiIndex) or set(frame.index.names) != {
        "datetime",
        "instrument",
    }:
        raise ValueError("market controls require a datetime/instrument MultiIndex")
    timestamps = pd.to_datetime(frame.index.get_level_values("datetime")).tz_localize(None)
    instruments = frame.index.get_level_values("instrument").astype(str)
    frame.index = pd.MultiIndex.from_arrays(
        [timestamps, instruments], names=["datetime", "instrument"]
    )
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_index()


def _rolling_liquidity(values: pd.Series | None, lookback_days: int) -> pd.DataFrame | None:
    if values is None:
        return None
    frame = values.unstack("instrument").sort_index()
    return frame.rolling(lookback_days, min_periods=min(5, lookback_days)).mean()


def _normalize_industry_memberships(values: pd.DataFrame | None) -> pd.DataFrame | None:
    if values is None:
        return None
    required = {"instrument", "industry", "in_date", "out_date"}
    missing = required.difference(values.columns)
    if missing:
        raise ValueError(f"industry memberships are missing columns: {', '.join(sorted(missing))}")
    frame = values.loc[:, sorted(required)].copy()
    frame["instrument"] = frame["instrument"].astype(str)
    frame["industry"] = frame["industry"].astype(str)
    frame["in_date"] = pd.to_datetime(frame["in_date"], errors="coerce").dt.tz_localize(None)
    frame["out_date"] = pd.to_datetime(frame["out_date"], errors="coerce").dt.tz_localize(None)
    frame = frame.dropna(subset=["instrument", "industry", "in_date"])
    return frame.sort_values(["instrument", "in_date", "industry"])


def _normalize_benchmark_weights(values: pd.DataFrame | None) -> pd.DataFrame | None:
    if values is None:
        return None
    required = {"datetime", "instrument", "weight"}
    missing = required.difference(values.columns)
    if missing:
        raise ValueError(f"benchmark weights are missing columns: {', '.join(sorted(missing))}")
    frame = values.loc[:, sorted(required)].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.tz_localize(None)
    frame["instrument"] = frame["instrument"].astype(str)
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame = frame.dropna()
    frame = frame[frame["weight"] > 0]
    return frame.sort_values(["datetime", "instrument"])


def _normalize_style_exposures(values: pd.DataFrame | None) -> pd.DataFrame | None:
    if values is None:
        return None
    required = {"datetime", "instrument", "log_market_cap"}
    missing = required.difference(values.columns)
    if missing:
        raise ValueError(f"style exposures are missing columns: {', '.join(sorted(missing))}")
    frame = values.loc[:, sorted(required)].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.tz_localize(None)
    frame["instrument"] = frame["instrument"].astype(str)
    frame["log_market_cap"] = pd.to_numeric(frame["log_market_cap"], errors="coerce")
    return frame.dropna().sort_values(["datetime", "instrument"])


def _benchmark_weights_at(values: pd.DataFrame | None, timestamp: pd.Timestamp) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    eligible = values[values["datetime"] <= timestamp]
    if eligible.empty:
        return pd.Series(dtype=float)
    latest = eligible["datetime"].max()
    weights = eligible.loc[eligible["datetime"] == latest].set_index("instrument")["weight"]
    total = float(weights.sum())
    return weights / total if total > 0 else pd.Series(dtype=float)


def _styles_at(
    values: pd.DataFrame | None, timestamp: pd.Timestamp, instruments: pd.Index
) -> pd.Series:
    if values is None or len(instruments) == 0:
        return pd.Series(index=instruments, dtype=float)
    daily = values[values["datetime"] == timestamp].set_index("instrument")["log_market_cap"]
    std = float(daily.std(ddof=0))
    normalized = (daily - daily.mean()) / std if std > 0 else daily * 0.0
    return normalized.reindex(instruments)


def _neutralize_size_score(scores: pd.Series, styles: pd.Series) -> pd.Series:
    aligned = pd.concat([scores.rename("score"), styles.rename("size")], axis=1).dropna()
    if len(aligned) < 3 or float(aligned["size"].std(ddof=0)) == 0:
        return scores
    x = aligned["size"].to_numpy(dtype=float)
    y = aligned["score"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    residual = pd.Series(y - fitted, index=aligned.index)
    result = scores.copy()
    result.loc[residual.index] = residual
    return result


def _industries_at(
    memberships: pd.DataFrame | None,
    timestamp: pd.Timestamp,
    instruments: pd.Index,
) -> pd.Series:
    if memberships is None:
        return pd.Series(index=instruments, dtype="object")
    active = memberships[
        (memberships["in_date"] <= timestamp)
        & (memberships["out_date"].isna() | (memberships["out_date"] >= timestamp))
        & memberships["instrument"].isin(instruments)
    ]
    active = active.sort_values("in_date").drop_duplicates("instrument", keep="last")
    return active.set_index("instrument")["industry"].reindex(instruments)


def _select_with_industry_cap(
    candidates: Sequence[str],
    industries: pd.Series,
    *,
    topk: int,
    max_industry_weight: float,
    benchmark_industries: pd.Series,
    max_industry_deviation: float,
    require_industry: bool,
) -> list[str]:
    absolute_max_names = max(1, int(np.floor(max_industry_weight * topk + 1e-12)))
    minimums = {
        str(industry): max(0, int(np.floor((float(weight) - max_industry_deviation) * topk)))
        for industry, weight in benchmark_industries.items()
    }
    maximums = {
        str(industry): min(
            absolute_max_names,
            max(0, int(np.floor((float(weight) + max_industry_deviation) * topk + 1e-12))),
        )
        for industry, weight in benchmark_industries.items()
    }
    counts: dict[str, int] = {}
    selected: list[str] = []

    def industry_key(instrument: str) -> str | None:
        industry = industries.get(instrument)
        if pd.isna(industry):
            return None if require_industry else f"__unconstrained__:{instrument}"
        return str(industry)

    for industry, minimum in minimums.items():
        for instrument in candidates:
            if len(selected) >= topk or counts.get(industry, 0) >= minimum:
                break
            if instrument in selected or industry_key(instrument) != industry:
                continue
            selected.append(instrument)
            counts[industry] = counts.get(industry, 0) + 1

    for instrument in candidates:
        if instrument in selected:
            continue
        key = industry_key(instrument)
        if key is None:
            continue
        max_names = maximums.get(key, absolute_max_names)
        if counts.get(key, 0) >= max_names:
            continue
        selected.append(instrument)
        counts[key] = counts.get(key, 0) + 1
        if len(selected) >= topk:
            break
    return selected


def _simulation_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "topk": int(config["topk"]),
        "n_drop": int(config["n_drop"]),
        "max_position_weight": float(config["max_position_weight"]),
        "max_daily_turnover": float(config["max_daily_turnover"]),
        "open_cost": float(config["open_cost"]),
        "close_cost": float(config["close_cost"]),
        "portfolio_notional": float(config["capacity_notional"]),
        "max_volume_participation": float(config["max_volume_participation"]),
        "max_industry_weight": float(config.get("max_industry_weight", 1.0)),
        "max_industry_deviation": float(config.get("max_industry_deviation", 1.0)),
        "max_size_deviation": float(config.get("max_size_deviation", 10.0)),
        "portfolio_construction": str(
            config.get("portfolio_construction", "topk_equal_weight")
        ),
        "optimizer_alpha_weight": float(config.get("optimizer_alpha_weight", 0.05)),
        "optimizer_tracking_penalty": float(
            config.get("optimizer_tracking_penalty", 1.0)
        ),
        "optimizer_turnover_penalty": float(
            config.get("optimizer_turnover_penalty", 0.10)
        ),
        "min_average_daily_amount": float(config.get("min_average_daily_amount", 0.0)),
        "liquidity_lookback_days": int(config.get("liquidity_lookback_days", 20)),
        "min_commission": float(config.get("min_commission", 0.0)),
        "execution_risk_enabled": True,
        "max_daily_loss": float(config.get("max_daily_loss", 0.03)),
        "stop_loss": float(config.get("stop_loss", 0.07)),
        "take_profit_partial": float(config.get("take_profit_partial", 0.12)),
        "take_profit_partial_fraction": float(config.get("take_profit_partial_fraction", 0.50)),
        "take_profit": float(config.get("take_profit", 0.20)),
        "max_drawdown_reduce": float(config.get("max_drawdown_reduce", 0.10)),
        "max_drawdown_liquidate": float(config.get("max_drawdown_liquidate", 0.15)),
        "drawdown_reduction_exposure": float(
            config.get("drawdown_reduction_exposure", 0.50)
        ),
    }


def _metrics_pass_risk_gate(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    return all(
        (
            float(metrics["annualized_excess_return"]) > 0,
            float(metrics["tracking_error"]) <= float(config["max_tracking_error"]),
            abs(float(metrics["max_drawdown"])) <= float(config["max_drawdown"]),
            float(metrics["average_turnover"]) <= float(config["max_turnover"]),
            metrics.get("information_ratio") is not None
            and float(metrics["information_ratio"]) >= float(config["min_information_ratio"]),
            metrics.get("sharpe_ratio") is not None
            and float(metrics["sharpe_ratio"]) >= float(config["min_sharpe_ratio"]),
            metrics.get("sortino_ratio") is not None
            and float(metrics["sortino_ratio"]) >= float(config["min_sortino_ratio"]),
            metrics.get("capacity_fill_ratio") is not None
            and float(metrics["capacity_fill_ratio"]) >= float(config["min_capacity_fill_ratio"]),
            "max_industry_deviation" not in config
            or (
                metrics.get("max_industry_deviation") is not None
                and float(metrics["max_industry_deviation"])
                <= float(config["max_industry_deviation"])
            ),
            "max_size_deviation" not in config
            or (
                metrics.get("max_size_deviation") is not None
                and float(metrics["max_size_deviation"])
                <= float(config["max_size_deviation"])
            ),
        )
    )
