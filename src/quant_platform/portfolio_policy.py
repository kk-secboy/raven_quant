from __future__ import annotations

from collections.abc import Collection
from dataclasses import asdict, dataclass
from decimal import ROUND_FLOOR, Decimal
from typing import Any

import numpy as np
import pandas as pd

from .cost_model import CostModelConfig
from .discrete_constraints import validate_discrete_constraints
from .portfolio_optimizer import optimize_benchmark_relative_weights

POLICY_VERSION = "portfolio-policy-v3"
_DISCRETE_CONSTRAINT_TOLERANCE = 1e-8


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
    min_cash_weight: float = 0.0
    max_asset_class_weights: dict[str, float] | None = None
    max_size_deviation: float = 0.30
    max_value_deviation: float = 0.30
    max_growth_deviation: float = 0.30
    max_volatility_deviation: float = 0.30
    max_daily_loss: float = 0.03
    stop_loss: float = 0.07
    profit_taking_mode: str = "threshold"
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
    entry_score_min_percentile: float = 0.0
    score_drop_exit_percentile: float | None = None
    score_deterioration_reduce_percentile: float | None = None
    score_deterioration_reduce_fraction: float = 0.50
    extension_guard_max_return_5d: float | None = None
    holding_min_sessions: int | None = None
    max_holding_sessions: int | None = None
    min_rebalance_weight_change: float = 0.0
    market_trend_lookback_sessions: int | None = None
    valuation_regime_max_percentile: float | None = None
    valuation_reduce_percentile: float | None = None
    valuation_reduce_fraction: float = 0.50
    trend_break_lookback_sessions: int | None = None
    thesis_min_holding_sessions: int | None = None
    thesis_break_score_percentile: float | None = None
    thesis_review_frequency: str | None = None
    hard_risk_target_fraction: float = 0.50
    cash_when_no_edge: bool = False

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
        if self.profit_taking_mode not in {"threshold", "rule_only", "thesis_only"}:
            raise ValueError("profit-taking mode is invalid")
        if (
            self.profit_taking_mode == "threshold"
            and not 0 < self.take_profit_partial < self.take_profit
        ):
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
        if not 0 <= self.min_cash_weight < 1:
            raise ValueError("minimum cash weight must be between zero and one")
        if self.max_asset_class_weights is not None and any(
            not 0 <= float(limit) <= 1
            for limit in self.max_asset_class_weights.values()
        ):
            raise ValueError("asset class limits must be between zero and one")
        if self.target_volatility is not None and not 0 < self.target_volatility <= 0.50:
            raise ValueError("target volatility must be between zero and 0.50")
        if not 0.0 <= self.entry_score_min_percentile <= 1.0:
            raise ValueError("entry score percentile must be within [0, 1]")
        for name, value in (
            ("score-drop exit percentile", self.score_drop_exit_percentile),
            (
                "score-deterioration reduce percentile",
                self.score_deterioration_reduce_percentile,
            ),
            ("valuation regime percentile", self.valuation_regime_max_percentile),
            ("valuation reduce percentile", self.valuation_reduce_percentile),
            ("thesis-break score percentile", self.thesis_break_score_percentile),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if (
            self.extension_guard_max_return_5d is not None
            and not 0.0 < self.extension_guard_max_return_5d <= 1.0
        ):
            raise ValueError("extension guard return must be within (0, 1]")
        for name, value in (
            ("minimum holding", self.holding_min_sessions),
            ("maximum holding", self.max_holding_sessions),
            ("market-trend lookback", self.market_trend_lookback_sessions),
            ("trend-break lookback", self.trend_break_lookback_sessions),
            ("thesis minimum holding", self.thesis_min_holding_sessions),
        ):
            if value is not None and (isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} sessions must be positive")
        if (
            self.holding_min_sessions is not None
            and self.max_holding_sessions is not None
            and self.holding_min_sessions > self.max_holding_sessions
        ):
            raise ValueError("minimum holding cannot exceed maximum holding")
        if not 0.0 <= self.min_rebalance_weight_change <= 1.0:
            raise ValueError("minimum rebalance weight change must be within [0, 1]")
        for name, value in (
            ("score-deterioration reduce fraction", self.score_deterioration_reduce_fraction),
            ("valuation reduce fraction", self.valuation_reduce_fraction),
            ("hard-risk target fraction", self.hard_risk_target_fraction),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be within [0, 1)")
        if self.thesis_review_frequency not in {None, "month", "quarter"}:
            raise ValueError("thesis review frequency must be month or quarter")
        if (self.thesis_min_holding_sessions is None) != (
            self.thesis_break_score_percentile is None
        ):
            raise ValueError("thesis-break holding and score rules must be configured together")
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
        asset_classes: pd.Series | None = None,
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
        holding_age_sessions: dict[str, int] | None = None,
        five_day_returns: pd.Series | None = None,
        trend_intact: pd.Series | None = None,
        valuation_percentiles: pd.Series | None = None,
        market_regime_allows_entries: bool | None = None,
        instrument_risk_states: pd.Series | dict[str, str] | None = None,
        portfolio_drawdown: float = 0.0,
        daily_return: float = 0.0,
        rebalance_due: bool = True,
        rebalance_instruments: Collection[str] | None = None,
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
        if not previous.index.is_unique:
            raise ValueError("previous weights must use a unique instrument index")
        previous = pd.to_numeric(previous, errors="coerce").astype(float)
        if (
            previous.isna().any()
            or not np.isfinite(previous.to_numpy(dtype=float)).all()
            or (previous < 0).any()
            or float(previous.sum()) > 1.0 + 1e-8
        ):
            raise ValueError(
                "previous weights must be finite, non-negative and sum to at most one"
            )
        normalized_risk_exposure = float(risk_exposure)
        normalized_drawdown = float(portfolio_drawdown)
        normalized_daily_return = float(daily_return)
        if (
            not np.isfinite(normalized_risk_exposure)
            or not 0.0 <= normalized_risk_exposure <= 1.0
        ):
            raise ValueError("risk exposure must be finite and within [0, 1]")
        if (
            not np.isfinite(normalized_drawdown)
            or not -1.0 <= normalized_drawdown <= 0.0
        ):
            raise ValueError("portfolio drawdown must be finite and within [-1, 0]")
        if not np.isfinite(normalized_daily_return) or normalized_daily_return < -1.0:
            raise ValueError("daily return must be finite and at least -100%")
        basis = (
            cost_basis.copy()
            if isinstance(cost_basis, pd.Series)
            else pd.Series(cost_basis or {}, dtype=float)
        )
        basis.index = basis.index.astype(str)
        stages = {str(key): int(value) for key, value in (take_profit_stages or {}).items()}
        ages = {
            str(key): int(value) for key, value in (holding_age_sessions or {}).items()
        }
        if any(value < 0 for value in ages.values()):
            raise ValueError("holding ages must be non-negative trading sessions")
        previous_instruments = set(previous[previous > 0].index)
        normalized_rebalance_instruments: set[str] | None = None
        if rebalance_instruments is not None:
            normalized_rebalance_instruments = {
                str(instrument).strip() for instrument in rebalance_instruments
            }
            if (
                not rebalance_due
                or not normalized_rebalance_instruments
                or "" in normalized_rebalance_instruments
                or not normalized_rebalance_instruments
                <= set(signal.index).union(previous.index)
            ):
                raise ValueError(
                    "scoped rebalance instruments require a due decision and known identities"
                )
        reviewed_previous_instruments = (
            previous_instruments
            if normalized_rebalance_instruments is None
            else previous_instruments & normalized_rebalance_instruments
        )
        if (
            self.config.max_holding_sessions is not None
            or self.config.holding_min_sessions is not None
        ) and ages and not (
            previous_instruments <= set(ages)
        ):
            raise ValueError("holding-period policy requires complete holding-age state")
        if (
            previous_instruments
            and (
                self.config.max_holding_sessions is not None
                or self.config.holding_min_sessions is not None
            )
            and not ages
        ):
            raise ValueError("holding-period policy requires complete holding-age state")
        risk_events: list[dict[str, Any]] = []
        suppressed_changes: list[dict[str, Any]] = []
        score_percentiles = signal.rank(method="average", pct=True)
        new_entry_eligible = score_percentiles >= self.config.entry_score_min_percentile
        if self.config.extension_guard_max_return_5d is not None:
            if five_day_returns is None:
                raise ValueError("extension guard requires point-in-time five-day returns")
            extension = pd.to_numeric(five_day_returns, errors="coerce")
            extension.index = extension.index.astype(str)
            if extension.index.has_duplicates:
                raise ValueError("extension guard five-day returns are duplicated")
            aligned_extension = extension.reindex(signal.index)
            new_entry_eligible &= (
                aligned_extension.notna()
                & np.isfinite(aligned_extension.to_numpy(dtype=float))
                & (aligned_extension <= self.config.extension_guard_max_return_5d)
            )
        if self.config.market_trend_lookback_sessions is not None:
            if market_regime_allows_entries is None:
                raise ValueError("market-trend rule requires point-in-time regime evidence")
            if not market_regime_allows_entries:
                new_entry_eligible[:] = False
        normalized_valuations: pd.Series | None = None
        if (
            self.config.valuation_regime_max_percentile is not None
            or self.config.valuation_reduce_percentile is not None
        ):
            if valuation_percentiles is None:
                raise ValueError("valuation rules require point-in-time percentiles")
            normalized_valuations = pd.to_numeric(valuation_percentiles, errors="coerce")
            normalized_valuations.index = normalized_valuations.index.astype(str)
            if normalized_valuations.index.has_duplicates:
                raise ValueError("valuation evidence is duplicated")
        if self.config.valuation_regime_max_percentile is not None:
            assert normalized_valuations is not None
            aligned_valuation = normalized_valuations.reindex(signal.index)
            new_entry_eligible &= (
                aligned_valuation.notna()
                & np.isfinite(aligned_valuation.to_numpy(dtype=float))
                & (
                    aligned_valuation
                    <= self.config.valuation_regime_max_percentile
                )
            )
        normalized_trends: pd.Series | None = None
        if self.config.trend_break_lookback_sessions is not None:
            if trend_intact is None:
                raise ValueError("trend-break rule requires point-in-time trend evidence")
            normalized_trends = trend_intact.copy()
            normalized_trends.index = normalized_trends.index.astype(str)
            if normalized_trends.index.has_duplicates:
                raise ValueError("trend-break evidence is duplicated")
            aligned_trends = normalized_trends.reindex(signal.index)
            # A missing or already-broken trend can never become a new
            # holding.  Existing holdings are handled separately below so a
            # legitimate suspension does not abort every other decision.
            new_entry_eligible &= aligned_trends.eq(True)
        keep_count = min(len(signal), self.config.topk + self.config.n_drop)
        ranked = signal.sort_values(ascending=False)
        retained = [item for item in ranked.index[:keep_count] if item in previous.index]
        eligible_ranked = [
            item
            for item in ranked.index
            if bool(new_entry_eligible[item]) or item in previous_instruments
        ]
        candidates = list(dict.fromkeys([*retained, *eligible_ranked]))
        benchmark_relative_industry_constraints = self.config.portfolio_construction in {
            "benchmark_relative_qp",
            "industry_neutral_qp",
        }
        target_slots = min(self.config.topk, len(signal))
        assumed_weight = min(
            1.0 / target_slots,
            self.config.max_position_weight,
        )
        partial_cash_target_weight: float | None = None
        partial_cash_reasons: list[str] = []
        if (
            not benchmark_relative_industry_constraints
            and len(candidates) < target_slots
        ):
            # Entry, regime and PIT eligibility are economic gates.  A smaller
            # qualified set must not be re-normalized into larger positions;
            # keep the unallocated sleeve as cash.
            partial_cash_target_weight = assumed_weight
            partial_cash_reasons.append("eligible_universe_cash")
        if industries is not None:
            industry_by_instrument = industries.astype(str)
            counts: dict[str, int] = {}
            selected = []
            for instrument in candidates:
                industry = str(industry_by_instrument.get(instrument, "__unknown__"))
                industry_cap = self.config.max_industry_weight
                if (
                    benchmark_relative_industry_constraints
                    and benchmark_industry_weights is not None
                ):
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
            if len(selected) < min(self.config.topk, len(candidates)):
                if benchmark_relative_industry_constraints:
                    raise ValueError(
                        "industry constraints leave too few eligible instruments"
                    )
                # The industry count limits were calculated with this fixed
                # per-position weight. Re-normalizing a smaller feasible set
                # would breach those limits, so retain the residual as cash.
                partial_cash_target_weight = assumed_weight
                partial_cash_reasons.append("industry_capacity_cash")
        else:
            selected = candidates[: self.config.topk]
        selected_scores = ranked.reindex(selected).dropna()
        target_volatility_evidence: dict[str, float] = {}
        if selected_scores.empty:
            target = pd.Series(dtype=float)
        elif self.config.portfolio_construction in {
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
            target_weight = (
                partial_cash_target_weight
                if partial_cash_target_weight is not None
                else min(
                    1.0 / len(selected_scores),
                    self.config.max_position_weight,
                )
            )
            target = pd.Series(target_weight, index=selected_scores.index, dtype=float)

        cadence_hold = not rebalance_due
        if cadence_hold:
            target = previous.reindex(target.index.union(previous.index), fill_value=0.0)
        elif self.config.holding_min_sessions is not None:
            target, holding_events = self._preserve_minimum_holding(
                target,
                previous,
                signal=signal,
                ages=ages,
                minimum_sessions=self.config.holding_min_sessions,
            )
            suppressed_changes.extend(holding_events)

        if rebalance_due and self.config.min_rebalance_weight_change > 0:
            target = target.reindex(target.index.union(previous.index), fill_value=0.0)
            previous_for_band = previous.reindex(target.index, fill_value=0.0)
            small_changes = (target - previous_for_band).abs().between(
                0.0,
                self.config.min_rebalance_weight_change,
                inclusive="neither",
            )
            for instrument in target.index[small_changes]:
                suppressed_changes.append(
                    {
                        "instrument": str(instrument),
                        "rule": "minimum_rebalance_weight_change",
                        "requested_weight": float(target[instrument]),
                        "retained_weight": float(previous_for_band[instrument]),
                    }
                )
            target.loc[small_changes] = previous_for_band.loc[small_changes]

        if normalized_rebalance_instruments is not None:
            # An off-cadence PIT filing review is a local decision, not a
            # licence to recompute the entire long-horizon account.  Preserve
            # every unrelated holding and suppress unrelated new entries.  A
            # reviewed sleeve may use existing cash, but can never crowd the
            # fixed sleeve above the account exposure/cash ceiling.
            all_scope_instruments = target.index.union(previous.index)
            target = target.reindex(all_scope_instruments, fill_value=0.0)
            previous_for_scope = previous.reindex(all_scope_instruments, fill_value=0.0)
            scoped_mask = target.index.isin(normalized_rebalance_instruments)
            fixed_weight = float(previous_for_scope.loc[~scoped_mask].sum())
            desired_total = min(
                1.0 - self.config.min_cash_weight,
                max(float(previous_for_scope.sum()), float(target.sum())),
            )
            scoped_capacity = max(0.0, desired_total - fixed_weight)
            requested_scoped_weight = float(target.loc[scoped_mask].sum())
            if requested_scoped_weight > scoped_capacity + 1e-12:
                target.loc[scoped_mask] *= scoped_capacity / requested_scoped_weight
            target.loc[~scoped_mask] = previous_for_scope.loc[~scoped_mask]

        if normalized_drawdown <= -self.config.max_drawdown_liquidate:
            target *= 0.0
            risk_events.append(
                self._risk_event(
                    "max_drawdown_liquidate",
                    normalized_drawdown,
                    self.config.max_drawdown_liquidate,
                    "liquidate",
                )
            )
        elif normalized_drawdown <= -self.config.max_drawdown_reduce:
            target *= self.config.drawdown_reduction_exposure
            risk_events.append(
                self._risk_event(
                    "max_drawdown_reduce",
                    normalized_drawdown,
                    self.config.max_drawdown_reduce,
                    "reduce_exposure",
                )
            )
        target *= normalized_risk_exposure
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
                elif (
                    instrument in reviewed_previous_instruments
                    and self.config.profit_taking_mode == "threshold"
                    and position_return >= self.config.take_profit
                ):
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
                elif (
                    instrument in reviewed_previous_instruments
                    and self.config.profit_taking_mode == "threshold"
                    and position_return >= self.config.take_profit_partial
                ):
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
        if self.config.score_drop_exit_percentile is not None:
            for instrument in reviewed_previous_instruments:
                percentile = float(score_percentiles.get(instrument, 0.0))
                if percentile < self.config.score_drop_exit_percentile:
                    target[instrument] = 0.0
                    risk_events.append(
                        self._risk_event(
                            "score_drop_exit",
                            percentile,
                            self.config.score_drop_exit_percentile,
                            "exit",
                            instrument,
                        )
                    )
        if self.config.score_deterioration_reduce_percentile is not None and rebalance_due:
            for instrument in reviewed_previous_instruments:
                percentile = float(score_percentiles.get(instrument, 0.0))
                if (
                    percentile < self.config.score_deterioration_reduce_percentile
                    and float(target.get(instrument, 0.0)) > 0.0
                ):
                    target[instrument] = min(
                        float(target[instrument]),
                        float(previous[instrument])
                        * (1.0 - self.config.score_deterioration_reduce_fraction),
                    )
                    risk_events.append(
                        self._risk_event(
                            "score_deterioration_reduce",
                            percentile,
                            self.config.score_deterioration_reduce_percentile,
                            "reduce_position",
                            instrument,
                        )
                    )
        if self.config.valuation_reduce_percentile is not None and rebalance_due:
            assert normalized_valuations is not None
            held_valuations = normalized_valuations.reindex(
                pd.Index(reviewed_previous_instruments, dtype=str)
            )
            for instrument, percentile in held_valuations.dropna().items():
                if (
                    float(percentile) > self.config.valuation_reduce_percentile
                    and float(target.get(instrument, 0.0)) > 0.0
                ):
                    target[instrument] = min(
                        float(target[instrument]),
                        float(previous[instrument])
                        * (1.0 - self.config.valuation_reduce_fraction),
                    )
                    risk_events.append(
                        self._risk_event(
                            "valuation_reduce",
                            float(percentile),
                            self.config.valuation_reduce_percentile,
                            "reduce_position",
                            str(instrument),
                        )
                    )
        if self.config.max_holding_sessions is not None:
            for instrument in reviewed_previous_instruments:
                age = ages.get(instrument, 0)
                if age >= self.config.max_holding_sessions:
                    target[instrument] = 0.0
                    risk_events.append(
                        self._risk_event(
                            "max_holding_sessions",
                            float(age),
                            float(self.config.max_holding_sessions),
                            "exit",
                            instrument,
                        )
                    )
        if self.config.trend_break_lookback_sessions is not None:
            assert normalized_trends is not None
            trends = normalized_trends.reindex(
                pd.Index(reviewed_previous_instruments, dtype=str)
            )
            for instrument in trends[trends.isna()].index:
                # Missing is not evidence of a trend break.  Preserve the
                # holding (it may be suspended) and make the degraded evidence
                # explicit instead of failing the whole portfolio batch.
                risk_events.append(
                    self._risk_event(
                        "trend_evidence_unavailable",
                        0.0,
                        1.0,
                        "hold_no_new_entry",
                        str(instrument),
                    )
                )
            for instrument, intact in trends.dropna().items():
                if not bool(intact):
                    target[instrument] = 0.0
                    risk_events.append(
                        self._risk_event(
                            "trend_break",
                            0.0,
                            1.0,
                            "exit",
                            str(instrument),
                        )
                    )
        # A long-horizon thesis is reviewed only on its governed monthly or
        # newly PIT-effective financial-report decision.  Daily refreshes may
        # still enforce hard account/risk controls, but must not silently turn
        # the quality/value thesis proxy into a daily trading rule.
        if self.config.thesis_min_holding_sessions is not None and rebalance_due:
            assert self.config.thesis_break_score_percentile is not None
            for instrument in reviewed_previous_instruments:
                age = ages.get(instrument, 0)
                percentile = float(score_percentiles.get(instrument, 0.0))
                if (
                    age >= self.config.thesis_min_holding_sessions
                    and percentile < self.config.thesis_break_score_percentile
                ):
                    target[instrument] = 0.0
                    risk_events.append(
                        self._risk_event(
                            "quant_quality_value_thesis_break_proxy",
                            percentile,
                            self.config.thesis_break_score_percentile,
                            "exit",
                            instrument,
                        )
                    )
        if normalized_daily_return <= -self.config.max_daily_loss:
            target = pd.concat([target, previous], axis=1).min(axis=1)
            risk_events.append(
                self._risk_event(
                    "max_daily_loss",
                    normalized_daily_return,
                    self.config.max_daily_loss,
                    "no_new_buys",
                )
            )
        normalized_risk_states = self._normalize_instrument_risk_states(
            instrument_risk_states,
            index=all_instruments,
        )
        for instrument, state in normalized_risk_states.items():
            previous_weight = float(previous.get(instrument, 0.0))
            desired_weight = float(target.get(instrument, 0.0))
            if state == "normal":
                continue
            if state in {"watch", "restricted"}:
                target[instrument] = min(desired_weight, previous_weight)
                if previous_weight <= 0 and desired_weight <= 0:
                    continue
                risk_events.append(
                    self._risk_event(
                        f"instrument_{state}",
                        1.0,
                        1.0,
                        "no_new_buys",
                        str(instrument),
                    )
                )
            elif state == "reduce":
                if previous_weight <= 0 and desired_weight <= 0:
                    continue
                target[instrument] = min(
                    desired_weight,
                    previous_weight * self.config.hard_risk_target_fraction,
                )
                risk_events.append(
                    self._risk_event(
                        "instrument_hard_risk_reduce",
                        1.0,
                        1.0,
                        "reduce_position",
                        str(instrument),
                    )
                )
            elif state == "exit":
                if previous_weight <= 0 and desired_weight <= 0:
                    continue
                target[instrument] = 0.0
                risk_events.append(
                    self._risk_event(
                        "instrument_hard_risk_exit",
                        1.0,
                        1.0,
                        "exit",
                        str(instrument),
                    )
                )
        governed_risk_decrease_candidates = {
            str(event["instrument"])
            for event in risk_events
            if event.get("instrument") is not None
            and event.get("action") in {"reduce_position", "exit"}
        }
        if any(
            event.get("instrument") is None
            and event.get("action") in {"reduce_exposure", "liquidate"}
            for event in risk_events
        ):
            governed_risk_decrease_candidates.update(
                str(instrument)
                for instrument in target.index
                if float(target.get(instrument, 0.0))
                < float(previous.get(instrument, 0.0)) - 1e-12
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
                if normalized_rebalance_instruments is not None:
                    outside_scope = ~pending_target.index.isin(
                        normalized_rebalance_instruments
                    )
                    pending_target.loc[outside_scope] = previous.loc[outside_scope]
                remaining_execution_days = saved_remaining
            else:
                pending_target = target.copy()
                remaining_execution_days = self.config.execution_days
            target = previous + (pending_target - previous) / remaining_execution_days
        frozen_instruments: set[str] = set()
        deferred_target_weights: dict[str, float] = {}
        price_values: pd.Series | None = None
        if (prices is None) != (portfolio_value is None):
            raise ValueError("prices and portfolio_value must be supplied together")
        if prices is not None and portfolio_value is not None:
            if portfolio_value <= 0:
                raise ValueError("portfolio_value must be positive")
            price_values = prices.copy()
            price_values.index = price_values.index.astype(str)
            if price_values.index.has_duplicates:
                raise ValueError("prices are duplicated")
            price_values = pd.to_numeric(price_values, errors="coerce").reindex(
                all_instruments
            )
            invalid_prices = price_values.isna() | ~np.isfinite(
                price_values.to_numpy(dtype=float)
            ) | (price_values <= 0)
            invalid_execution = invalid_prices.copy()
            daily_values: pd.Series | None = None
            if average_daily_values is not None:
                daily_values = average_daily_values.copy()
                daily_values.index = daily_values.index.astype(str)
                if daily_values.index.has_duplicates:
                    raise ValueError("average daily values are duplicated")
                daily_values = pd.to_numeric(daily_values, errors="coerce").reindex(
                    all_instruments
                )
                requested_execution = (
                    target.reindex(all_instruments, fill_value=0.0)
                    - previous.reindex(all_instruments, fill_value=0.0)
                ).abs() > 1e-10
                invalid_daily_values = daily_values.isna() | ~np.isfinite(
                    daily_values.to_numpy(dtype=float)
                ) | (daily_values < 0)
                unavailable_liquidity = (daily_values <= 0) & requested_execution
                invalid_execution |= invalid_daily_values | unavailable_liquidity
            frozen_instruments = {
                str(instrument) for instrument in all_instruments[invalid_execution]
            }
            desired_before_freeze = target.reindex(all_instruments, fill_value=0.0).copy()
            for instrument in sorted(frozen_instruments):
                desired_weight = float(desired_before_freeze[instrument])
                retained_weight = float(previous[instrument])
                target[instrument] = retained_weight
                if abs(desired_weight - retained_weight) > 1e-10:
                    deferred_target_weights[instrument] = desired_weight
                event = self._risk_event(
                    "execution_evidence_unavailable",
                    0.0,
                    1.0,
                    "wait_existing" if retained_weight > 0 else "block_new_entry",
                    instrument,
                )
                unavailable_evidence: list[str] = []
                if bool(invalid_prices.loc[instrument]):
                    unavailable_evidence.append("price")
                if daily_values is not None and (
                    not np.isfinite(float(daily_values.loc[instrument]))
                    or float(daily_values.loc[instrument]) <= 0
                ):
                    unavailable_evidence.append("average_daily_value")
                event["unavailable_evidence"] = unavailable_evidence
                risk_events.append(event)
            desired_exposure = max(
                float(desired_before_freeze.sum()),
                float(previous.sum()),
            )
            excess = max(0.0, float(target.sum()) - desired_exposure)
            funding_order = sorted(
                (
                    str(instrument)
                    for instrument in target[target > previous + 1e-12].index
                    if str(instrument) not in frozen_instruments
                ),
                key=lambda instrument: (float(signal.get(instrument, -np.inf)), instrument),
            )
            for instrument in funding_order:
                available = float(target[instrument] - previous[instrument])
                reduction = min(available, excess)
                target[instrument] -= reduction
                excess -= reduction
                if excess <= 1e-12:
                    break
            if excess > 1e-8:
                raise ValueError("frozen holdings cannot be funded without leverage")
            tradable = ~target.index.isin(frozen_instruments)
            if daily_values is not None:
                max_change = (
                    daily_values * self.cost_model.max_volume_participation / portfolio_value
                )
                target.loc[tradable] = target.loc[tradable].clip(
                    lower=(previous - max_change).clip(lower=0.0).loc[tradable],
                    upper=(previous + max_change).loc[tradable],
                )
            target = self._round_tradable_lots(
                target,
                price_values,
                portfolio_value=portfolio_value,
                lot_size=self.cost_model.lot_size,
                frozen_instruments=frozen_instruments,
            )
        target, max_position_repair_events = self._repair_max_position_weight(
            target,
            max_position_weight=self.config.max_position_weight,
            frozen_instruments=frozen_instruments,
            prices=price_values,
            portfolio_value=portfolio_value,
            lot_size=self.cost_model.lot_size,
        )
        repaired_instruments = {
            str(event["instrument"]) for event in max_position_repair_events
        }
        for instrument in sorted(str(item) for item in target.index):
            if instrument in frozen_instruments or instrument in repaired_instruments:
                continue
            previous_weight = float(previous[instrument])
            target_weight = float(target[instrument])
            if (
                previous_weight
                > self.config.max_position_weight + _DISCRETE_CONSTRAINT_TOLERANCE
                and target_weight < previous_weight - 1e-12
            ):
                event = self._risk_event(
                    "max_position_weight_risk_reduction",
                    previous_weight,
                    self.config.max_position_weight,
                    "reduce_position",
                    instrument,
                )
                event["target_weight_after"] = target_weight
                max_position_repair_events.append(event)
        risk_events.extend(max_position_repair_events)
        governed_risk_decrease_candidates.update(
            str(event["instrument"])
            for event in max_position_repair_events
            if float(target[str(event["instrument"])])
            < float(previous[str(event["instrument"])]) - 1e-12
        )
        if pending_target is not None:
            for event in max_position_repair_events:
                instrument = str(event["instrument"])
                pending_target[instrument] = min(
                    float(pending_target.get(instrument, 0.0)),
                    self.config.max_position_weight,
                )
        risk_turnover_exempt_instruments = {
            instrument
            for instrument in governed_risk_decrease_candidates
            if instrument in target.index
            and float(target[instrument]) < float(previous[instrument]) - 1e-12
        }
        risk_ceiling: pd.Series | None = None
        if risk_turnover_exempt_instruments:
            risk_ceiling = pd.Series(1.0, index=target.index, dtype=float)
            risk_ceiling.loc[sorted(risk_turnover_exempt_instruments)] = target.loc[
                sorted(risk_turnover_exempt_instruments)
            ]
        raw_changes = target - previous
        turnover = self._turnover(target, previous)
        if turnover > self.config.max_daily_turnover and turnover > 0:
            scale = self.config.max_daily_turnover / turnover
            target = previous + raw_changes * scale
            if price_values is not None and portfolio_value is not None:
                target = self._round_tradable_lots(
                    target,
                    price_values,
                    portfolio_value=portfolio_value,
                    lot_size=self.cost_model.lot_size,
                    frozen_instruments=frozen_instruments,
                )
            raw_changes = target - previous
            turnover = self._turnover(target, previous)
        if risk_ceiling is not None:
            target = pd.concat([target, risk_ceiling.reindex(target.index)], axis=1).min(axis=1)
            if price_values is not None and portfolio_value is not None:
                target = self._round_tradable_lots(
                    target,
                    price_values,
                    portfolio_value=portfolio_value,
                    lot_size=self.cost_model.lot_size,
                    frozen_instruments=frozen_instruments,
                )
            raw_changes = target - previous
            turnover = self._turnover(target, previous)
        target[target.abs() < 1e-10] = 0.0
        constrained_policy = self.config.portfolio_construction in {
            "benchmark_relative_qp",
            "industry_neutral_qp",
        }
        # Relative constraints apply to the invested stock sleeve.  A fresh
        # account is intentionally ramped from cash by the turnover/execution
        # caps, and volatility or risk gates can also retain cash. Comparing
        # that partial sleeve with a 100%-invested benchmark would make the
        # first rebalance mathematically infeasible even when its composition
        # is perfectly benchmark-relative.
        constraint_scale = float(target.sum())
        constraint_benchmark_weights = (
            benchmark_weights * constraint_scale
            if constrained_policy and benchmark_weights is not None
            else None
        )
        constraint_benchmark_industries = (
            benchmark_industry_weights * constraint_scale
            if constrained_policy and benchmark_industry_weights is not None
            else None
        )
        if constrained_policy and benchmark_style_exposure is not None:
            if isinstance(benchmark_style_exposure, pd.Series):
                constraint_benchmark_styles: float | pd.Series | dict[str, float] = (
                    benchmark_style_exposure * constraint_scale
                )
            elif isinstance(benchmark_style_exposure, dict):
                constraint_benchmark_styles = {
                    key: float(value) * constraint_scale
                    for key, value in benchmark_style_exposure.items()
                }
            else:
                constraint_benchmark_styles = (
                    float(benchmark_style_exposure) * constraint_scale
                )
        else:
            constraint_benchmark_styles = {}
        discrete_validation = validate_discrete_constraints(
            target,
            previous,
            max_position_weight=self.config.max_position_weight,
            max_daily_turnover=self.config.max_daily_turnover,
            min_cash_weight=self.config.min_cash_weight,
            industries=industries,
            max_industry_weight=(
                self.config.max_industry_weight if industries is not None else None
            ),
            benchmark_industry_weights=(
                constraint_benchmark_industries
            ),
            max_industry_deviation=(
                self.config.max_industry_deviation if constrained_policy else None
            ),
            asset_classes=asset_classes,
            max_asset_class_weights=self.config.max_asset_class_weights,
            benchmark_weights=constraint_benchmark_weights,
            return_covariance=return_covariance if constrained_policy else None,
            max_tracking_error=(
                self.config.max_tracking_error if constrained_policy else None
            ),
            style_exposures=style_exposures if constrained_policy else None,
            benchmark_style_exposure=(
                constraint_benchmark_styles if constrained_policy else None
            ),
            max_style_deviations=(
                {
                    "size": self.config.max_size_deviation,
                    "value": self.config.max_value_deviation,
                    "growth": self.config.max_growth_deviation,
                    "volatility": self.config.max_volatility_deviation,
                    "log_market_cap": self.config.max_size_deviation,
                }
                if constrained_policy
                else None
            ),
            average_daily_values=average_daily_values,
            portfolio_value=portfolio_value,
            max_volume_participation=(
                self.cost_model.max_volume_participation
                if average_daily_values is not None
                else None
            ),
            prices=prices,
            lot_size=self.cost_model.lot_size if prices is not None else None,
            risk_ceiling=risk_ceiling,
            risk_turnover_exempt_instruments=risk_turnover_exempt_instruments,
            frozen_instruments=frozen_instruments,
        )
        if discrete_validation["status"] != "passed":
            failures = ", ".join(
                f"{item['name']}[{item['scope']}]"
                for item in discrete_validation["violations"]
            )
            raise ValueError(
                "post-discretization hard constraint violation: " + failures
            )
        risk_turnover_exception = discrete_validation["risk_turnover_exception"]
        if risk_turnover_exception["status"] == "applied":
            event = self._risk_event(
                "risk_driven_turnover_exception",
                float(risk_turnover_exception["actual_turnover"]),
                self.config.max_daily_turnover,
                "allow_governed_decreases_only",
            )
            event["instruments"] = list(risk_turnover_exception["instruments"])
            event["turnover_subject_to_limit"] = float(
                risk_turnover_exception["turnover_subject_to_limit"]
            )
            event["gross_increase_weight"] = float(
                risk_turnover_exception["gross_increase_weight"]
            )
            risk_events.append(event)
        for exception in discrete_validation[
            "frozen_inherited_max_position_exceptions"
        ]:
            event = self._risk_event(
                "frozen_inherited_max_position_exception",
                float(exception["target_weight"]),
                float(exception["configured_limit"]),
                "retain_non_worsening_until_tradable",
                str(exception["instrument"]),
            )
            event["previous_weight"] = float(exception["previous_weight"])
            risk_events.append(event)
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
        next_holding_ages = {
            str(instrument): (
                ages.get(str(instrument), 0) + 1
                if str(instrument) in previous_instruments
                else 0
            )
            for instrument in target[target > 0].index
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
        reasons.extend(dict.fromkeys(partial_cash_reasons))
        if cadence_hold:
            reasons.append(f"{self.config.rebalance_frequency} rebalance cadence hold")
        if self.config.execution_days > 1:
            reasons.append(
                f"{self.config.execution_method} execution over {self.config.execution_days} days"
            )
        if normalized_risk_exposure < 1:
            reasons.append("risk exposure reduction")
        if target_volatility_evidence:
            reasons.append("target volatility exposure scaling")
        if not allow_new_risk:
            reasons.append("member drawdown gate pauses new risk")
        if suppressed_changes:
            reasons.append("holding discipline and no-trade band")
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
                "holding_age_sessions": next_holding_ages,
                "constraint_benchmark_scale": constraint_scale,
                "discrete_constraint_validation": discrete_validation,
                "suppressed_changes": suppressed_changes,
                "frozen_instruments": sorted(frozen_instruments),
                "deferred_target_weights": deferred_target_weights,
                **(
                    {
                        "rebalance_instruments": sorted(
                            normalized_rebalance_instruments
                        )
                    }
                    if normalized_rebalance_instruments is not None
                    else {}
                ),
                **(
                    {"target_volatility": target_volatility_evidence}
                    if target_volatility_evidence
                    else {}
                ),
            },
        )

    @classmethod
    def _preserve_minimum_holding(
        cls,
        target: pd.Series,
        previous: pd.Series,
        *,
        signal: pd.Series,
        ages: dict[str, int],
        minimum_sessions: int,
    ) -> tuple[pd.Series, list[dict[str, Any]]]:
        instruments = target.index.union(previous.index)
        result = target.reindex(instruments, fill_value=0.0).astype(float)
        prior = previous.reindex(instruments, fill_value=0.0).astype(float)
        desired_exposure = max(float(result.sum()), float(prior.sum()))
        locked: set[str] = set()
        events: list[dict[str, Any]] = []
        for instrument in prior[prior > 0].index:
            age = int(ages[str(instrument)])
            if age >= minimum_sessions or result[instrument] >= prior[instrument]:
                continue
            result[instrument] = prior[instrument]
            locked.add(str(instrument))
            events.append(
                {
                    "instrument": str(instrument),
                    "rule": "minimum_holding_sessions",
                    "observed_sessions": age,
                    "minimum_sessions": minimum_sessions,
                    "action": "retain_normal_decrease",
                }
            )
        excess = max(0.0, float(result.sum()) - desired_exposure)
        if excess <= 1e-12:
            return result, events
        increases = result[result > prior + 1e-12].index
        funding_order = sorted(
            (str(item) for item in increases if str(item) not in locked),
            key=lambda item: (float(signal.get(item, -np.inf)), item),
        )
        for instrument in funding_order:
            available = float(result[instrument] - prior[instrument])
            reduction = min(available, excess)
            result[instrument] -= reduction
            excess -= reduction
            if excess <= 1e-12:
                break
        if excess > 1e-8:
            raise ValueError("minimum-holding lock cannot be funded without leverage")
        return result, events

    @staticmethod
    def _normalize_instrument_risk_states(
        values: pd.Series | dict[str, str] | None,
        *,
        index: pd.Index,
    ) -> pd.Series:
        if values is None:
            return pd.Series("normal", index=index, dtype=str)
        result = values.copy() if isinstance(values, pd.Series) else pd.Series(values, dtype=str)
        result.index = result.index.astype(str)
        if result.index.has_duplicates:
            raise ValueError("instrument risk states are duplicated")
        result = result.astype(str).str.strip().str.lower().reindex(index, fill_value="normal")
        allowed = {"normal", "watch", "restricted", "reduce", "exit"}
        invalid = sorted(set(result) - allowed)
        if invalid:
            raise ValueError("instrument risk states are invalid: " + ", ".join(invalid))
        return result

    @staticmethod
    def _round_tradable_lots(
        target: pd.Series,
        prices: pd.Series,
        *,
        portfolio_value: float,
        lot_size: int,
        frozen_instruments: set[str],
    ) -> pd.Series:
        result = target.copy().astype(float)
        tradable = ~result.index.isin(frozen_instruments)
        quantities = (
            np.floor(
                result.loc[tradable]
                * portfolio_value
                / prices.loc[tradable]
                / lot_size
            )
            * lot_size
        )
        result.loc[tradable] = quantities * prices.loc[tradable] / portfolio_value
        return result

    @staticmethod
    def _repair_max_position_weight(
        target: pd.Series,
        *,
        max_position_weight: float,
        frozen_instruments: set[str],
        prices: pd.Series | None,
        portfolio_value: float | None,
        lot_size: int,
    ) -> tuple[pd.Series, list[dict[str, Any]]]:
        """Reduce tradable targets to the largest whole-lot position under the cap."""

        result = target.copy().astype(float)
        events: list[dict[str, Any]] = []
        for instrument in sorted(str(item) for item in result.index):
            if instrument in frozen_instruments:
                continue
            observed_weight = float(result[instrument])
            if observed_weight <= (
                max_position_weight + _DISCRETE_CONSTRAINT_TOLERANCE
            ):
                continue

            repaired_weight = max_position_weight
            event = PortfolioPolicy._risk_event(
                "post_discretization_max_position_repair",
                observed_weight,
                max_position_weight,
                "reduce_position",
                instrument,
            )
            if prices is not None and portfolio_value is not None:
                price = float(prices[instrument])
                price_decimal = Decimal(str(price))
                portfolio_decimal = Decimal(str(portfolio_value))
                allowed_notional = Decimal(str(max_position_weight)) * portfolio_decimal
                lot_notional = price_decimal * Decimal(lot_size)
                allowed_lots = int(
                    (allowed_notional / lot_notional).to_integral_value(
                        rounding=ROUND_FLOOR
                    )
                )
                current_quantity = int(
                    round(observed_weight * portfolio_value / price / lot_size)
                ) * lot_size
                repaired_quantity = min(current_quantity, allowed_lots * lot_size)
                while (
                    repaired_quantity > 0
                    and Decimal(repaired_quantity) * price_decimal > allowed_notional
                ):
                    repaired_quantity -= lot_size
                repaired_weight = repaired_quantity * price / portfolio_value
                event.update(
                    {
                        "price": price,
                        "portfolio_value": float(portfolio_value),
                        "lot_size": int(lot_size),
                        "quantity_before": current_quantity,
                        "quantity_after": repaired_quantity,
                        "quantity_reduced": current_quantity - repaired_quantity,
                    }
                )
            result[instrument] = repaired_weight
            event["target_weight_after"] = float(repaired_weight)
            events.append(event)
        return result, events

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
