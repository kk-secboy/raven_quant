from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .cost_model import CostModelConfig
from .portfolio_optimizer import optimize_benchmark_relative_weights

POLICY_VERSION = "portfolio-policy-v2"


def rebalance_period_key(value: Any, frequency: str) -> tuple[int, ...]:
    timestamp = pd.Timestamp(value).tz_localize(None)
    if frequency == "bar":
        return (
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
        )
    if frequency == "day":
        return (timestamp.year, timestamp.month, timestamp.day)
    if frequency == "week":
        iso = timestamp.isocalendar()
        return (int(iso.year), int(iso.week))
    if frequency == "month":
        return (timestamp.year, timestamp.month)
    raise ValueError("rebalance frequency must be bar, day, week, or month")


def is_rebalance_due(current: Any, previous_rebalance: Any | None, frequency: str) -> bool:
    current_key = rebalance_period_key(current, frequency)
    return previous_rebalance is None or current_key != rebalance_period_key(
        previous_rebalance, frequency
    )


@dataclass(frozen=True)
class PortfolioPolicyConfig:
    topk: int = 50
    n_drop: int = 5
    max_position_weight: float = 0.02
    max_daily_turnover: float = 0.15
    max_industry_weight: float = 0.30
    max_industry_deviation: float = 0.03
    max_tracking_error: float = 0.12
    max_size_deviation: float = 0.30
    max_value_deviation: float = 0.30
    max_growth_deviation: float = 0.30
    max_volatility_deviation: float = 0.30
    max_daily_loss: float = 0.03
    stop_loss: float = 0.07
    take_profit_partial: float = 0.12
    take_profit_partial_fraction: float = 0.50
    take_profit: float = 0.20
    max_drawdown_reduce: float = 0.10
    max_drawdown_liquidate: float = 0.15
    drawdown_reduction_exposure: float = 0.50
    execution_days: int = 1
    execution_method: str = "open"
    portfolio_construction: str = "topk_equal_weight"
    optimizer_alpha_weight: float = 0.05
    optimizer_tracking_penalty: float = 1.0
    optimizer_turnover_penalty: float = 0.10
    target_volatility: float | None = None
    rebalance_frequency: str = "day"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> PortfolioPolicyConfig:
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})

    def __post_init__(self) -> None:
        if self.topk < 1 or not 0 <= self.n_drop <= self.topk:
            raise ValueError("topk and n_drop are invalid")
        if not 0 < self.max_position_weight <= 1 or not 0 < self.max_daily_turnover <= 1:
            raise ValueError("position and turnover limits are invalid")
        if not 0 < self.stop_loss < 1 or not 0 < self.max_daily_loss < 1:
            raise ValueError("loss limits are invalid")
        if not 0 < self.take_profit_partial < self.take_profit:
            raise ValueError("take-profit thresholds are invalid")
        if not 0 < self.take_profit_partial_fraction < 1:
            raise ValueError("partial take-profit fraction is invalid")
        if not 0 < self.max_drawdown_reduce < self.max_drawdown_liquidate < 1:
            raise ValueError("drawdown thresholds are invalid")
        if not 0 < self.drawdown_reduction_exposure < 1:
            raise ValueError("drawdown reduction exposure is invalid")
        if not 1 <= self.execution_days <= 5:
            raise ValueError("execution days must be between one and five")
        if self.execution_method not in {"open", "twap", "vwap", "next_bar"}:
            raise ValueError("execution method must be open, twap, vwap, or next_bar")
        if self.portfolio_construction not in {
            "topk_equal_weight",
            "benchmark_relative_qp",
            "industry_neutral_qp",
        }:
            raise ValueError("unsupported portfolio construction method")
        if not 0 < self.max_tracking_error <= 1:
            raise ValueError("tracking-error limit must be between zero and one")
        if self.target_volatility is not None and not 0 < self.target_volatility <= 0.50:
            raise ValueError("target volatility must be between zero and 0.50")
        rebalance_period_key("2026-01-01", self.rebalance_frequency)


@dataclass(frozen=True)
class PolicyDecision:
    target_weights: dict[str, float]
    changes: list[dict[str, Any]]
    reasons: list[str]
    expected_turnover: float
    policy_version: str
    cost_model: dict[str, Any]
    risk_events: list[dict[str, Any]]
    position_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PortfolioPolicy:
    """Pure portfolio decision policy shared by Qlib and recommendation refreshes."""

    version = POLICY_VERSION

    def __init__(
        self,
        config: PortfolioPolicyConfig,
        cost_model: CostModelConfig | None = None,
    ) -> None:
        self.config = config
        self.cost_model = cost_model or CostModelConfig()

    def decide(
        self,
        scores: pd.Series,
        previous_weights: pd.Series | dict[str, float] | None = None,
        *,
        industries: pd.Series | None = None,
        benchmark_weights: pd.Series | None = None,
        benchmark_industry_weights: pd.Series | None = None,
        style_exposures: pd.Series | pd.DataFrame | None = None,
        benchmark_style_exposure: float | pd.Series | dict[str, float] | None = None,
        return_covariance: pd.DataFrame | None = None,
        prices: pd.Series | None = None,
        average_daily_values: pd.Series | None = None,
        portfolio_value: float | None = None,
        risk_exposure: float = 1.0,
        allow_new_risk: bool = True,
        current_prices: pd.Series | None = None,
        cost_basis: pd.Series | dict[str, float] | None = None,
        take_profit_stages: dict[str, int] | None = None,
        execution_state: dict[str, Any] | None = None,
        portfolio_drawdown: float = 0.0,
        daily_return: float = 0.0,
        rebalance_due: bool = True,
    ) -> PolicyDecision:
        signal = pd.to_numeric(scores, errors="coerce").dropna().astype(float)
        signal.index = signal.index.astype(str)
        if signal.empty or not signal.index.is_unique or not np.isfinite(signal).all():
            raise ValueError("policy scores must be unique, finite and non-empty")
        previous = (
            previous_weights.copy()
            if isinstance(previous_weights, pd.Series)
            else pd.Series(previous_weights or {}, dtype=float)
        )
        previous.index = previous.index.astype(str)
        previous = previous.clip(lower=0.0)
        basis = (
            cost_basis.copy()
            if isinstance(cost_basis, pd.Series)
            else pd.Series(cost_basis or {}, dtype=float)
        )
        basis.index = basis.index.astype(str)
        stages = {str(key): int(value) for key, value in (take_profit_stages or {}).items()}
        risk_events: list[dict[str, Any]] = []
        keep_count = min(len(signal), self.config.topk + self.config.n_drop)
        ranked = signal.sort_values(ascending=False)
        retained = [item for item in previous.index if item in ranked.index[:keep_count]]
        candidates = list(dict.fromkeys([*retained, *ranked.index]))
        if industries is not None:
            industry_by_instrument = industries.astype(str)
            assumed_weight = min(
                1.0 / min(self.config.topk, len(signal)), self.config.max_position_weight
            )
            counts: dict[str, int] = {}
            selected = []
            for instrument in candidates:
                industry = str(industry_by_instrument.get(instrument, "__unknown__"))
                industry_cap = self.config.max_industry_weight
                if benchmark_industry_weights is not None:
                    industry_cap = min(
                        industry_cap,
                        float(benchmark_industry_weights.get(industry, 0.0))
                        + self.config.max_industry_deviation,
                    )
                max_count = int(np.floor(industry_cap / assumed_weight + 1e-12))
                if counts.get(industry, 0) >= max_count:
                    continue
                selected.append(instrument)
                counts[industry] = counts.get(industry, 0) + 1
                if len(selected) == self.config.topk:
                    break
            if len(selected) < min(self.config.topk, len(signal)):
                raise ValueError("industry constraints leave too few eligible instruments")
        else:
            selected = candidates[: self.config.topk]
        selected_scores = ranked.reindex(selected).dropna()
        if selected_scores.empty:
            raise ValueError("policy has no eligible instruments")

        target_volatility_evidence: dict[str, float] = {}
        if self.config.portfolio_construction in {
            "benchmark_relative_qp",
            "industry_neutral_qp",
        }:
            required = (
                industries,
                benchmark_weights,
                benchmark_industry_weights,
                style_exposures,
                benchmark_style_exposure,
                return_covariance,
            )
            if any(item is None for item in required):
                raise ValueError(
                    "constrained Qlib policy requires complete point-in-time target metadata"
                )
            optimized = optimize_benchmark_relative_weights(
                selected_scores,
                benchmark_weights,  # type: ignore[arg-type]
                previous,
                industries=industries,  # type: ignore[arg-type]
                benchmark_industry_weights=benchmark_industry_weights,  # type: ignore[arg-type]
                style_exposures=style_exposures,  # type: ignore[arg-type]
                benchmark_style_exposure=benchmark_style_exposure,
                return_covariance=return_covariance,  # type: ignore[arg-type]
                max_position_weight=self.config.max_position_weight,
                max_industry_weight=self.config.max_industry_weight,
                max_industry_deviation=self.config.max_industry_deviation,
                max_size_deviation=self.config.max_size_deviation,
                max_style_deviations={
                    "size": self.config.max_size_deviation,
                    "value": self.config.max_value_deviation,
                    "growth": self.config.max_growth_deviation,
                    "volatility": self.config.max_volatility_deviation,
                    "log_market_cap": self.config.max_size_deviation,
                },
                alpha_weight=self.config.optimizer_alpha_weight,
                tracking_penalty=self.config.optimizer_tracking_penalty,
                turnover_penalty=self.config.optimizer_turnover_penalty,
                max_tracking_error=self.config.max_tracking_error,
            )
            target = optimized.weights
            if (
                self.config.portfolio_construction == "industry_neutral_qp"
                and self.config.target_volatility is not None
            ):
                if optimized.portfolio_volatility <= 0:
                    raise ValueError("target-volatility scaling requires positive portfolio risk")
                exposure_scale = min(
                    1.0,
                    self.config.target_volatility / optimized.portfolio_volatility,
                )
                target *= exposure_scale
                target_volatility_evidence = {
                    "unscaled_annualized_volatility": optimized.portfolio_volatility,
                    "target_annualized_volatility": self.config.target_volatility,
                    "exposure_scale": exposure_scale,
                }
        else:
            target_weight = min(1.0 / len(selected_scores), self.config.max_position_weight)
            target = pd.Series(target_weight, index=selected_scores.index, dtype=float)

        cadence_hold = not rebalance_due
        if cadence_hold:
            target = previous.reindex(target.index.union(previous.index), fill_value=0.0)

        if portfolio_drawdown <= -self.config.max_drawdown_liquidate:
            target *= 0.0
            risk_events.append(
                self._risk_event(
                    "max_drawdown_liquidate",
                    portfolio_drawdown,
                    self.config.max_drawdown_liquidate,
                    "liquidate",
                )
            )
        elif portfolio_drawdown <= -self.config.max_drawdown_reduce:
            target *= self.config.drawdown_reduction_exposure
            risk_events.append(
                self._risk_event(
                    "max_drawdown_reduce",
                    portfolio_drawdown,
                    self.config.max_drawdown_reduce,
                    "reduce_exposure",
                )
            )
        target *= max(0.0, min(1.0, float(risk_exposure)))
        all_instruments = target.index.union(previous.index)
        target = target.reindex(all_instruments, fill_value=0.0)
        previous = previous.reindex(all_instruments, fill_value=0.0)
        if not allow_new_risk:
            target = pd.concat([target, previous], axis=1).min(axis=1)
            risk_events.append(
                self._risk_event(
                    "member_drawdown_pause_new_risk",
                    1.0,
                    1.0,
                    "no_new_buys",
                )
            )
        if current_prices is not None:
            marks = pd.to_numeric(current_prices, errors="coerce").reindex(all_instruments)
            for instrument in previous[previous > 0].index:
                entry = float(basis.get(instrument, np.nan))
                mark = float(marks.get(instrument, np.nan))
                if not np.isfinite(entry) or entry <= 0 or not np.isfinite(mark) or mark <= 0:
                    continue
                position_return = mark / entry - 1.0
                if position_return <= -self.config.stop_loss:
                    target[instrument] = 0.0
                    risk_events.append(
                        self._risk_event(
                            "stop_loss", position_return, self.config.stop_loss, "exit", instrument
                        )
                    )
                elif position_return >= self.config.take_profit:
                    target[instrument] = 0.0
                    risk_events.append(
                        self._risk_event(
                            "take_profit",
                            position_return,
                            self.config.take_profit,
                            "exit",
                            instrument,
                        )
                    )
                elif position_return >= self.config.take_profit_partial:
                    if stages.get(instrument, 0) < 1:
                        target[instrument] = min(
                            target[instrument],
                            previous[instrument] * (1.0 - self.config.take_profit_partial_fraction),
                        )
                        stages[instrument] = 1
                        risk_events.append(
                            self._risk_event(
                                "take_profit_partial",
                                position_return,
                                self.config.take_profit_partial,
                                "reduce_position",
                                instrument,
                            )
                        )
                    else:
                        target[instrument] = min(target[instrument], previous[instrument])
        if daily_return <= -self.config.max_daily_loss:
            target = pd.concat([target, previous], axis=1).min(axis=1)
            risk_events.append(
                self._risk_event(
                    "max_daily_loss", daily_return, self.config.max_daily_loss, "no_new_buys"
                )
            )
        pending_target: pd.Series | None = None
        remaining_execution_days = 0
        if self.config.execution_days > 1 and not risk_events:
            saved_target = (execution_state or {}).get("target_weights")
            saved_remaining = int((execution_state or {}).get("remaining_days") or 0)
            if isinstance(saved_target, dict) and saved_remaining > 0:
                pending_target = pd.Series(saved_target, dtype=float)
                pending_target.index = pending_target.index.astype(str)
                all_instruments = all_instruments.union(pending_target.index)
                previous = previous.reindex(all_instruments, fill_value=0.0)
                pending_target = pending_target.reindex(all_instruments, fill_value=0.0)
                remaining_execution_days = saved_remaining
            else:
                pending_target = target.copy()
                remaining_execution_days = self.config.execution_days
            target = previous + (pending_target - previous) / remaining_execution_days
        risk_ceiling = target.copy() if risk_events else None
        if (prices is None) != (portfolio_value is None):
            raise ValueError("prices and portfolio_value must be supplied together")
        if prices is not None and portfolio_value is not None:
            if portfolio_value <= 0:
                raise ValueError("portfolio_value must be positive")
            price_values = pd.to_numeric(prices, errors="coerce").reindex(all_instruments)
            if price_values.isna().any() or (price_values <= 0).any():
                raise ValueError("prices must cover every target and existing holding")
            if average_daily_values is not None:
                daily_values = pd.to_numeric(average_daily_values, errors="coerce").reindex(
                    all_instruments
                )
                if daily_values.isna().any() or (daily_values < 0).any():
                    raise ValueError("average daily values must cover every instrument")
                max_change = (
                    daily_values * self.cost_model.max_volume_participation / portfolio_value
                )
                target = target.clip(
                    lower=(previous - max_change).clip(lower=0.0),
                    upper=previous + max_change,
                )
            lots = (
                np.floor(target * portfolio_value / price_values / self.cost_model.lot_size)
                * self.cost_model.lot_size
            )
            target = lots * price_values / portfolio_value
        raw_changes = target - previous
        turnover = self._turnover(target, previous)
        if turnover > self.config.max_daily_turnover and turnover > 0:
            scale = self.config.max_daily_turnover / turnover
            target = previous + raw_changes * scale
            if prices is not None and portfolio_value is not None:
                price_values = pd.to_numeric(prices, errors="coerce").reindex(all_instruments)
                lots = (
                    np.floor(target * portfolio_value / price_values / self.cost_model.lot_size)
                    * self.cost_model.lot_size
                )
                target = lots * price_values / portfolio_value
            raw_changes = target - previous
            turnover = self._turnover(target, previous)
        if risk_ceiling is not None:
            target = pd.concat([target, risk_ceiling.reindex(target.index)], axis=1).min(axis=1)
            if prices is not None and portfolio_value is not None:
                price_values = pd.to_numeric(prices, errors="coerce").reindex(all_instruments)
                lots = (
                    np.floor(target * portfolio_value / price_values / self.cost_model.lot_size)
                    * self.cost_model.lot_size
                )
                target = lots * price_values / portfolio_value
            raw_changes = target - previous
            turnover = self._turnover(target, previous)
        target[target.abs() < 1e-10] = 0.0
        next_execution_state: dict[str, Any] = {}
        if pending_target is not None:
            pending_target = pending_target.reindex(target.index, fill_value=0.0)
            unfinished = float((target - pending_target).abs().sum()) > 1e-8
            next_remaining = max(0, remaining_execution_days - 1)
            if unfinished:
                next_execution_state = {
                    "target_weights": {
                        key: float(value)
                        for key, value in pending_target[pending_target > 0].items()
                    },
                    "remaining_days": max(1, next_remaining),
                    "method": self.config.execution_method,
                }
        changes = [
            {
                "instrument": instrument,
                "action": "increase" if delta > 0 else "decrease",
                "previous_weight": float(previous[instrument]),
                "target_weight": float(target[instrument]),
                "weight_change": float(delta),
                "reason": "signal ranking and portfolio constraints",
            }
            for instrument, delta in raw_changes.items()
            if abs(float(delta)) > 1e-10
        ]
        reasons = ["ranked signal", "position cap", "turnover cap"]
        if cadence_hold:
            reasons.append(f"{self.config.rebalance_frequency} rebalance cadence hold")
        if self.config.execution_days > 1:
            reasons.append(
                f"{self.config.execution_method} execution over {self.config.execution_days} days"
            )
        if risk_exposure < 1:
            reasons.append("risk exposure reduction")
        if target_volatility_evidence:
            reasons.append("target volatility exposure scaling")
        if not allow_new_risk:
            reasons.append("member drawdown gate pauses new risk")
        reasons.extend(str(item["rule"]) for item in risk_events)
        return PolicyDecision(
            target_weights={key: float(value) for key, value in target[target > 0].items()},
            changes=changes,
            reasons=reasons,
            expected_turnover=turnover,
            policy_version=self.version,
            cost_model=self.cost_model.to_dict(),
            risk_events=risk_events,
            position_state={
                "take_profit_stages": stages,
                "execution": next_execution_state,
                **(
                    {"target_volatility": target_volatility_evidence}
                    if target_volatility_evidence
                    else {}
                ),
            },
        )

    @staticmethod
    def _turnover(target: pd.Series, previous: pd.Series) -> float:
        stock_change = float((target - previous).abs().sum())
        cash_change = abs((1.0 - float(target.sum())) - (1.0 - float(previous.sum())))
        return 0.5 * (stock_change + cash_change)

    @staticmethod
    def _risk_event(
        rule: str,
        observed: float,
        limit: float,
        action: str,
        instrument: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rule": rule,
            "observed": float(observed),
            "limit": float(limit),
            "action": action,
        }
        if instrument is not None:
            result["instrument"] = instrument
        return result
