from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .cost_model import CostModelConfig
from .portfolio_optimizer import optimize_benchmark_relative_weights

POLICY_VERSION = "portfolio-policy-v1"


@dataclass(frozen=True)
class PortfolioPolicyConfig:
    topk: int = 50
    n_drop: int = 5
    max_position_weight: float = 0.02
    max_daily_turnover: float = 0.15
    max_industry_weight: float = 0.30
    max_industry_deviation: float = 0.03
    max_size_deviation: float = 0.30
    portfolio_construction: str = "topk_equal_weight"
    optimizer_alpha_weight: float = 0.05
    optimizer_tracking_penalty: float = 1.0
    optimizer_turnover_penalty: float = 0.10

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> PortfolioPolicyConfig:
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})

    def __post_init__(self) -> None:
        if self.topk < 1 or not 0 <= self.n_drop <= self.topk:
            raise ValueError("topk and n_drop are invalid")
        if not 0 < self.max_position_weight <= 1 or not 0 < self.max_daily_turnover <= 1:
            raise ValueError("position and turnover limits are invalid")


@dataclass(frozen=True)
class PolicyDecision:
    target_weights: dict[str, float]
    changes: list[dict[str, Any]]
    reasons: list[str]
    expected_turnover: float
    policy_version: str
    cost_model: dict[str, Any]

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
        style_exposures: pd.Series | None = None,
        benchmark_style_exposure: float | None = None,
        prices: pd.Series | None = None,
        average_daily_values: pd.Series | None = None,
        portfolio_value: float | None = None,
        risk_exposure: float = 1.0,
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

        if self.config.portfolio_construction == "benchmark_relative_qp":
            required = (
                industries,
                benchmark_weights,
                benchmark_industry_weights,
                style_exposures,
                benchmark_style_exposure,
            )
            if any(item is None for item in required):
                raise ValueError(
                    "benchmark-relative policy requires complete point-in-time metadata"
                )
            optimized = optimize_benchmark_relative_weights(
                selected_scores,
                benchmark_weights,  # type: ignore[arg-type]
                previous,
                industries=industries,  # type: ignore[arg-type]
                benchmark_industry_weights=benchmark_industry_weights,  # type: ignore[arg-type]
                style_exposures=style_exposures,  # type: ignore[arg-type]
                benchmark_style_exposure=float(benchmark_style_exposure),
                max_position_weight=self.config.max_position_weight,
                max_industry_weight=self.config.max_industry_weight,
                max_industry_deviation=self.config.max_industry_deviation,
                max_size_deviation=self.config.max_size_deviation,
                alpha_weight=self.config.optimizer_alpha_weight,
                tracking_penalty=self.config.optimizer_tracking_penalty,
                turnover_penalty=self.config.optimizer_turnover_penalty,
            )
            target = optimized.weights
        else:
            target_weight = min(1.0 / len(selected_scores), self.config.max_position_weight)
            target = pd.Series(target_weight, index=selected_scores.index, dtype=float)

        target *= max(0.0, min(1.0, float(risk_exposure)))
        all_instruments = target.index.union(previous.index)
        target = target.reindex(all_instruments, fill_value=0.0)
        previous = previous.reindex(all_instruments, fill_value=0.0)
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
        target[target.abs() < 1e-10] = 0.0
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
        if risk_exposure < 1:
            reasons.append("risk exposure reduction")
        return PolicyDecision(
            target_weights={key: float(value) for key, value in target[target > 0].items()},
            changes=changes,
            reasons=reasons,
            expected_turnover=turnover,
            policy_version=self.version,
            cost_model=self.cost_model.to_dict(),
        )

    @staticmethod
    def _turnover(target: pd.Series, previous: pd.Series) -> float:
        stock_change = float((target - previous).abs().sum())
        cash_change = abs((1.0 - float(target.sum())) - (1.0 - float(previous.sum())))
        return 0.5 * (stock_change + cash_change)
