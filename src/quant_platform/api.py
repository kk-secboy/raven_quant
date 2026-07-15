from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, date, datetime, time
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import text

from quant_data.checkpoint import CheckpointStore
from quant_data.config import Settings, normalize_api_url
from quant_data.coverage_data import DEFAULT_COVERAGE_BUNDLES
from quant_data.execution_data import MINUTE_DATASETS, MINUTE_FREQUENCIES

from .alert_store import AlertStore
from .allocation_store import AllocationStore
from .auth_policy import ROLE_PERMISSIONS, has_permission, permission_for
from .auth_store import AuthenticationError, AuthStore
from .autonomous_research import AutonomousResearchOrchestrator
from .broker_gateway import (
    BrokerGatewayError,
    BrokerStore,
    validate_broker_gateway_credentials,
)
from .continuous_research import ContinuousResearchController
from .data_rollover import (
    next_qlib_trading_date,
    select_execution_snapshot,
    select_qlib_dataset,
)
from .data_task_store import DataTaskStore
from .deployment_readiness import DeploymentReadinessStore
from .health_store import OperationalHealthStore
from .job_store import JobStore
from .market_overview import MarketOverviewService
from .pair_portfolio_store import PairPortfolioStore
from .parameter_experiment_store import ParameterExperimentStore
from .parameter_experiments import normalize_parameter_grid, split_research_period
from .platform_config_store import PlatformConfigStore
from .portfolio_store import PortfolioStore
from .rdagent_runtime import probe_rdagent, validate_duration
from .research_automation import normalize_research_schedule_payload
from .research_store import ResearchStore
from .retention import DataRetentionManager
from .runtime_secret_store import RuntimeSecretStore
from .schedule_store import ScheduleStore
from .services import (
    dataset_catalog,
    list_qlib_datasets,
    list_qlib_experiments,
    list_snapshots,
    probe_qlib,
    resolve_snapshot_dataset,
    resolve_snapshot_manifest,
    system_summary,
)
from .strategy_recipes import RECIPE_VERSION, get_strategy_recipe, list_strategy_recipes
from .strategy_store import StrategyStore
from .worker import LocalJobWorker


class BootstrapRequest(BaseModel):
    profile: Literal["core", "research", "full"] = "core"
    start: date = Field(default=date(2024, 1, 1))
    end: date | Literal["latest"] = "latest"
    build_qlib: bool = False

    @model_validator(mode="after")
    def validate_range(self) -> BootstrapRequest:
        if isinstance(self.end, date) and self.end < self.start:
            raise ValueError("end must not be before start")
        return self


class DataFinalizeRequest(BaseModel):
    profile: Literal["core", "research", "full"] = "full"
    start: date = Field(default=date(2024, 1, 1))
    end: date | Literal["latest"] = "latest"
    snapshot_name: str | None = Field(default=None, min_length=3, max_length=120)

    @model_validator(mode="after")
    def validate_range(self) -> DataFinalizeRequest:
        if isinstance(self.end, date) and self.end < self.start:
            raise ValueError("end must not be before start")
        return self


class MarginEligibilityRequest(BaseModel):
    start: date = Field(default=date(2024, 1, 1))
    end: date | Literal["latest"] = "latest"

    @model_validator(mode="after")
    def validate_range(self) -> MarginEligibilityRequest:
        if isinstance(self.end, date) and self.end < self.start:
            raise ValueError("end must not be before start")
        return self


class CoreIntradayRequest(BaseModel):
    start: date = Field(default=date(2024, 1, 1))
    end: date | Literal["latest"] = "latest"
    etfs: list[str] = Field(default_factory=lambda: ["510300.SH", "159919.SZ"], max_length=100)
    stocks: list[str] = Field(default_factory=list, max_length=100)
    indices: list[str] = Field(default_factory=list, max_length=30)
    futures: list[str] = Field(default_factory=list, max_length=100)
    options: list[str] = Field(default_factory=list, max_length=200)
    auto_select: bool = True
    max_stocks: int = Field(default=100, ge=0, le=500)
    max_options: int = Field(default=100, ge=0, le=500)
    etf_categories: list[Literal["broad", "industry", "gold", "bond"]] = Field(
        default_factory=lambda: ["broad", "industry", "gold", "bond"]
    )
    snapshot_name: str | None = Field(default=None, min_length=3, max_length=120)

    @model_validator(mode="after")
    def validate_request(self) -> CoreIntradayRequest:
        if isinstance(self.end, date) and self.end < self.start:
            raise ValueError("end must not be before start")
        if not self.auto_select and not any(
            (self.etfs, self.stocks, self.indices, self.futures, self.options)
        ):
            raise ValueError("at least one minute symbol is required")
        return self


class SupplementalDownloadRequest(BaseModel):
    bundle: Literal[
        "cn_extended_daily",
        "cn_funds",
        "cn_macro",
        "cn_institutional",
        "cn_futures",
        "cn_options_bonds",
        "hk_market",
        "us_market",
        "global_markets",
        "cn_governance_risk",
        "cn_capital_flow",
        "cn_fund_index_enhanced",
        "cn_derivatives_enhanced",
        "global_rates_enhanced",
        "research_corpus",
        "strategy_specialty",
        "strategy_specialty_minutes",
    ]
    start: date = Field(default=date(2024, 1, 1))
    end: date | Literal["latest"] = "latest"
    symbols: list[str] = Field(default_factory=list, max_length=2000)

    @model_validator(mode="after")
    def validate_range(self) -> SupplementalDownloadRequest:
        if isinstance(self.end, date) and self.end < self.start:
            raise ValueError("end must not be before start")
        if self.bundle == "strategy_specialty_minutes" and not self.symbols:
            raise ValueError("strategy_specialty_minutes requires explicit symbols")
        return self


class Ashare5mRequest(BaseModel):
    start: date = Field(default=date(2024, 1, 1))
    end: date | Literal["latest"] = "latest"
    snapshot_name: str | None = Field(default=None, min_length=3, max_length=120)

    @model_validator(mode="after")
    def validate_range(self) -> Ashare5mRequest:
        if isinstance(self.end, date) and self.end < self.start:
            raise ValueError("end must not be before start")
        return self


class RetentionApplyRequest(BaseModel):
    names: list[str] = Field(min_length=1, max_length=100)
    confirmation: str
    keep_latest: int = Field(default=7, ge=1, le=100)
    min_age_days: int = Field(default=14, ge=1, le=3650)


class QlibBaselineRequest(BaseModel):
    dataset: str
    market: str = "cn_all"
    benchmark: str = "SH000300"
    account: float = Field(default=5_000_000, ge=100_000)
    topk: int = Field(default=50, ge=1, le=500)
    n_drop: int = Field(default=5, ge=0, le=100)
    open_cost: float = Field(default=0.0005, ge=0, le=0.02)
    close_cost: float = Field(default=0.0015, ge=0, le=0.02)
    min_cost: float = Field(default=5.0, ge=0, le=100)

    @model_validator(mode="after")
    def validate_strategy(self) -> QlibBaselineRequest:
        if self.n_drop > self.topk:
            raise ValueError("n_drop must not exceed topk")
        return self


class MinuteQlibRequest(BaseModel):
    snapshot_name: str = Field(
        min_length=3,
        max_length=120,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    output_name: str | None = Field(
        default=None,
        min_length=3,
        max_length=120,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )


class MinuteResearchRequest(BaseModel):
    dataset: str = Field(min_length=3, max_length=120)
    start: date
    end: date
    horizons: list[int] = Field(default_factory=lambda: [5, 15, 30], min_length=1, max_length=6)
    cost_rate: float = Field(default=0.0002, ge=0, le=0.02)

    @model_validator(mode="after")
    def validate_minute_research(self) -> MinuteResearchRequest:
        if self.end < self.start:
            raise ValueError("end must not be before start")
        if min(self.horizons) < 1 or max(self.horizons) > 240:
            raise ValueError("horizons must be between 1 and 240 minutes")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        return self


class ResearchPeriods(BaseModel):
    train_start: date = date(2018, 1, 1)
    train_end: date = date(2021, 12, 31)
    valid_start: date = date(2022, 1, 1)
    valid_end: date = date(2023, 12, 31)
    test_start: date = date(2024, 1, 1)
    test_end: date = Field(default_factory=date.today)

    @model_validator(mode="after")
    def validate_windows(self) -> ResearchPeriods:
        if not (
            self.train_start
            <= self.train_end
            < self.valid_start
            <= self.valid_end
            < self.test_start
            <= self.test_end
        ):
            raise ValueError(
                "train, validation, and test windows must be ordered and non-overlapping"
            )
        return self


class RDAgentRunRequest(BaseModel):
    objective: str = Field(min_length=10, max_length=2000)
    dataset: str
    loop_n: int = Field(default=1, ge=1, le=20)
    duration: str = "30m"
    requested_by: str = Field(default="local-operator", min_length=2, max_length=100)
    periods: ResearchPeriods = Field(default_factory=ResearchPeriods)

    @model_validator(mode="after")
    def validate_budget(self) -> RDAgentRunRequest:
        validate_duration(self.duration)
        return self


class FactorEvaluationRequest(BaseModel):
    dataset: str
    periods: ResearchPeriods
    metrics: dict[str, float | None]
    artifact_path: str | None = None


class PromotionRequest(BaseModel):
    actor: str = Field(min_length=2, max_length=100)
    reason: str = Field(min_length=10, max_length=2000)


class StrategyFactorRequest(BaseModel):
    candidate_id: str
    weight: float = Field(gt=-10, lt=10)

    @model_validator(mode="after")
    def nonzero_weight(self) -> StrategyFactorRequest:
        if abs(self.weight) < 1e-12:
            raise ValueError("factor weight must not be zero")
        return self


class StrategyConfigRequest(BaseModel):
    recipe_id: Literal["custom", "index_enhancement", "swing_trend"] = "custom"
    recipe_version: str = Field(default="custom", min_length=1, max_length=100)
    topk: int = Field(default=50, ge=5, le=500)
    n_drop: int = Field(default=5, ge=0, le=100)
    max_position_weight: float = Field(default=0.02, gt=0, le=0.20)
    max_daily_turnover: float = Field(default=0.20, gt=0, le=1.0)
    max_daily_loss: float = Field(default=0.03, gt=0, le=0.20)
    stop_loss: float = Field(default=0.07, gt=0, le=0.50)
    take_profit_partial: float = Field(default=0.12, gt=0, le=2.0)
    take_profit_partial_fraction: float = Field(default=0.50, gt=0, lt=1.0)
    take_profit: float = Field(default=0.20, gt=0, le=5.0)
    max_drawdown_reduce: float = Field(default=0.10, gt=0, le=0.50)
    max_drawdown_liquidate: float = Field(default=0.15, gt=0, le=0.80)
    drawdown_reduction_exposure: float = Field(default=0.50, gt=0, lt=1.0)
    max_industry_weight: float = Field(default=0.30, gt=0, le=1.0)
    max_industry_deviation: float = Field(default=0.05, ge=0, le=0.30)
    max_size_deviation: float = Field(default=0.30, ge=0, le=2.0)
    portfolio_construction: Literal["topk_equal_weight", "benchmark_relative_qp"] = (
        "topk_equal_weight"
    )
    optimizer_alpha_weight: float = Field(default=0.05, ge=0, le=10.0)
    optimizer_tracking_penalty: float = Field(default=1.0, ge=0, le=100.0)
    optimizer_turnover_penalty: float = Field(default=0.10, ge=0, le=100.0)
    min_average_daily_amount: float = Field(default=500_000_000, ge=1_000_000, le=100_000_000_000)
    liquidity_lookback_days: int = Field(default=20, ge=5, le=252)
    max_tracking_error: float = Field(default=0.12, gt=0, le=1.0)
    max_drawdown: float = Field(default=0.25, gt=0, le=1.0)
    max_turnover: float = Field(default=0.60, gt=0, le=2.0)
    min_information_ratio: float = Field(default=0.0, ge=-5, le=10)
    min_sharpe_ratio: float = Field(default=0.0, ge=-5, le=10)
    min_sortino_ratio: float = Field(default=0.0, ge=-5, le=20)
    min_robustness_pass_rate: float = Field(default=0.75, ge=0, le=1)
    rolling_window_days: int = Field(default=252, ge=60, le=1260)
    rolling_step_days: int = Field(default=63, ge=20, le=504)
    min_rolling_windows: int = Field(default=3, ge=2, le=20)
    min_rolling_pass_rate: float = Field(default=0.60, ge=0, le=1)
    event_window_days: int = Field(default=20, ge=20, le=126)
    event_count: int = Field(default=5, ge=1, le=20)
    max_event_underperformance: float = Field(default=0.05, ge=0, le=0.50)
    min_event_stress_pass_rate: float = Field(default=0.60, ge=0, le=1)
    min_backtest_days: int = Field(default=504, ge=252, le=2520)
    capacity_notional: float = Field(default=5_000_000, ge=100_000, le=10_000_000_000)
    max_volume_participation: float = Field(default=0.01, gt=0, le=0.20)
    min_capacity_fill_ratio: float = Field(default=0.95, ge=0, le=1)
    open_cost: float = Field(default=0.0005, ge=0, le=0.02)
    close_cost: float = Field(default=0.0015, ge=0, le=0.02)
    min_commission: float = Field(default=5.0, ge=0, le=1000)
    execution_model: Literal["next_open"] = "next_open"

    @model_validator(mode="after")
    def valid_dropout(self) -> StrategyConfigRequest:
        if self.recipe_id == "custom" and self.recipe_version != "custom":
            raise ValueError("custom strategy config must use the custom recipe version")
        if self.recipe_id != "custom" and self.recipe_version != RECIPE_VERSION:
            raise ValueError("strategy recipe version is not supported by this release")
        if self.n_drop > self.topk:
            raise ValueError("n_drop must not exceed topk")
        if self.max_industry_weight < self.max_position_weight:
            raise ValueError("max_industry_weight must not be below max_position_weight")
        if (
            self.portfolio_construction == "benchmark_relative_qp"
            and self.topk * self.max_position_weight < 1.0
        ):
            raise ValueError(
                "benchmark-relative optimization requires topk * max_position_weight >= 1"
            )
        if (
            self.optimizer_alpha_weight == 0
            and self.optimizer_tracking_penalty == 0
            and self.optimizer_turnover_penalty == 0
        ):
            raise ValueError("optimizer objective must contain a positive weight")
        if self.take_profit_partial >= self.take_profit:
            raise ValueError("take_profit_partial must be below take_profit")
        if self.max_drawdown_reduce >= self.max_drawdown_liquidate:
            raise ValueError("max_drawdown_reduce must be below max_drawdown_liquidate")
        if self.rolling_step_days > self.rolling_window_days:
            raise ValueError("rolling_step_days must not exceed rolling_window_days")
        return self


class StrategyCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=150)
    description: str = Field(min_length=10, max_length=2000)
    benchmark: str = "SH000300"
    universe: str = "cn_all"
    factors: list[StrategyFactorRequest] = Field(min_length=1, max_length=20)
    config: StrategyConfigRequest | None = None
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class StrategyVersionCreateRequest(BaseModel):
    benchmark: str = "SH000300"
    universe: str = "cn_all"
    factors: list[StrategyFactorRequest] = Field(min_length=1, max_length=20)
    config: StrategyConfigRequest | None = None
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class StrategyDefaultsUpdateRequest(BaseModel):
    config: StrategyConfigRequest
    reason: str = Field(min_length=10, max_length=2000)


class PairStrategyConfigRequest(BaseModel):
    formation_window: int = Field(default=60, ge=20, le=252)
    min_correlation: float = Field(default=0.80, ge=0, le=1)
    max_cointegration_pvalue: float = Field(default=0.05, gt=0, le=1)
    cointegration_recheck_days: int = Field(default=5, ge=1, le=63)
    entry_zscore: float = Field(default=1.50, gt=0, le=10)
    exit_zscore: float = Field(default=0.50, ge=0, le=10)
    stop_zscore: float = Field(default=3.00, gt=0, le=20)
    max_holding_days: int = Field(default=5, ge=1, le=20)
    initial_capital: float = Field(default=5_000_000, ge=100_000, le=10_000_000_000)
    pair_gross_fraction: float = Field(default=0.20, gt=0, le=1)
    max_volume_participation: float = Field(default=0.01, gt=0, le=0.20)
    min_capacity_fill_ratio: float = Field(default=0.95, gt=0, le=1)
    open_cost: float = Field(default=0.0005, ge=0, le=0.02)
    close_cost: float = Field(default=0.0015, ge=0, le=0.02)
    min_commission: float = Field(default=5.0, ge=0, le=1000)
    slippage: float = Field(default=0.0005, ge=0, le=0.02)
    annual_borrow_rate: float = Field(default=0.08, ge=0, le=1)
    lot_size: int = Field(default=100, ge=1, le=10000)
    kalman_process_variance: float = Field(default=1e-5, gt=0, le=1)
    kalman_observation_variance: float = Field(default=1e-3, gt=0, le=1)
    min_hedge_ratio: float = Field(default=0.10, gt=0, le=100)
    max_hedge_ratio: float = Field(default=10.0, gt=0, le=100)
    max_drawdown: float = Field(default=0.10, gt=0, le=0.50)
    min_sharpe_ratio: float = Field(default=0.0, ge=-5, le=10)
    min_closed_trades: int = Field(default=5, ge=1, le=10000)
    min_backtest_days: int = Field(default=252, ge=60, le=2520)
    min_rolling_cointegration_pass_rate: float = Field(default=0.80, ge=0, le=1)
    min_robustness_pass_rate: float = Field(default=0.75, ge=0, le=1)

    @model_validator(mode="after")
    def valid_pair_thresholds(self) -> PairStrategyConfigRequest:
        if not self.exit_zscore < self.entry_zscore < self.stop_zscore:
            raise ValueError("z-score thresholds must satisfy exit < entry < stop")
        if self.min_hedge_ratio >= self.max_hedge_ratio:
            raise ValueError("min_hedge_ratio must be below max_hedge_ratio")
        return self


class PairStrategyCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=150)
    description: str = Field(min_length=10, max_length=2000)
    leg_y: str = Field(min_length=4, max_length=32)
    leg_x: str = Field(min_length=4, max_length=32)
    asset_class: Literal["etf", "stock", "mixed"] = "etf"
    shorting_mode: Literal["margin_borrow"] = "margin_borrow"
    config: PairStrategyConfigRequest = Field(default_factory=PairStrategyConfigRequest)
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class PairStrategyVersionCreateRequest(BaseModel):
    leg_y: str = Field(min_length=4, max_length=32)
    leg_x: str = Field(min_length=4, max_length=32)
    asset_class: Literal["etf", "stock", "mixed"] = "etf"
    shorting_mode: Literal["margin_borrow"] = "margin_borrow"
    config: PairStrategyConfigRequest = Field(default_factory=PairStrategyConfigRequest)
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class PairStrategyBacktestRequest(BaseModel):
    dataset: str
    execution_snapshot: str
    minute_dataset: str
    shortability_dataset: str
    start: date
    end: date

    @model_validator(mode="after")
    def valid_period(self) -> PairStrategyBacktestRequest:
        if self.end <= self.start:
            raise ValueError("backtest end must be after start")
        if self.minute_dataset == self.shortability_dataset:
            raise ValueError("minute and shortability evidence must be separate datasets")
        return self


class StrategyBacktestRequest(BaseModel):
    dataset: str
    start: date
    end: date

    @model_validator(mode="after")
    def valid_period(self) -> StrategyBacktestRequest:
        if self.end <= self.start:
            raise ValueError("backtest end must be after start")
        return self


class ParameterExperimentRequest(BaseModel):
    dataset: str
    start: date
    end: date
    parameter_grid: dict[str, list[int | float]]
    max_trials: int = Field(default=27, ge=1, le=81)
    actor: str = Field(default="local-operator", min_length=2, max_length=100)

    @model_validator(mode="after")
    def valid_period(self) -> ParameterExperimentRequest:
        split_research_period(self.start, self.end)
        return self


class ResearchCampaignCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=150)
    objective: str = Field(min_length=10, max_length=2000)
    dataset: str
    recipe_id: Literal["index_enhancement", "swing_trend"] = "index_enhancement"
    benchmark: str | None = None
    universe: str | None = None
    loop_n: int = Field(default=2, ge=1, le=20)
    duration: str = "1h"
    periods: ResearchPeriods = Field(default_factory=ResearchPeriods)
    max_factors: int = Field(default=5, ge=1, le=20)
    parameter_grid: dict[str, list[int | float]] = Field(
        default_factory=lambda: {
            "n_drop": [5, 10],
            "max_daily_turnover": [0.15, 0.20, 0.25],
            "max_volume_participation": [0.005, 0.01],
        }
    )
    max_trials: int = Field(default=27, ge=1, le=81)
    strategy_config: StrategyConfigRequest | None = None
    initial_cash: float = Field(default=5_000_000, ge=100_000, le=10_000_000_000)
    timezone: str = "Asia/Shanghai"
    paper_run_time: time = time(15, 30)
    paper_slippage: float = Field(default=0.0005, ge=0, le=0.02)
    misfire_grace_seconds: int = Field(default=1800, ge=60, le=86400)
    actor: str = Field(default="local-operator", min_length=2, max_length=100)

    @model_validator(mode="after")
    def validate_campaign(self) -> ResearchCampaignCreateRequest:
        validate_duration(self.duration)
        split_research_period(self.periods.test_start, self.periods.test_end)
        if self.paper_run_time < time(15, 10):
            raise ValueError("paper simulation must run after the A-share close")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone is not available") from exc
        return self


class ResearchCampaignStatusRequest(BaseModel):
    status: Literal["paused", "running", "cancelled"]
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class ResearchProgramCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=100)
    dataset: str
    recipe_id: Literal["index_enhancement", "swing_trend"] = "index_enhancement"
    objective: str | None = Field(default=None, min_length=10, max_length=2000)
    benchmark: str | None = None
    universe: str | None = None
    train_trading_days: int = Field(default=756, ge=252, le=2520)
    validation_trading_days: int = Field(default=252, ge=63, le=756)
    test_trading_days: int = Field(default=504, ge=252, le=1260)
    min_new_trading_days: int = Field(default=20, ge=1, le=252)
    max_active_campaigns: int = Field(default=1, ge=1, le=3)
    champion_min_score_improvement: float = Field(default=0.05, ge=0, le=5)
    champion_decay_fraction: float = Field(default=0.25, gt=0, le=1)
    loop_n: int = Field(default=2, ge=1, le=20)
    duration: str = "1h"
    max_factors: int = Field(default=5, ge=1, le=20)
    parameter_grid: dict[str, list[int | float]] = Field(
        default_factory=lambda: {
            "n_drop": [5, 10],
            "max_daily_turnover": [0.15, 0.20, 0.25],
            "max_volume_participation": [0.005, 0.01],
        }
    )
    max_trials: int = Field(default=27, ge=1, le=81)
    strategy_config: StrategyConfigRequest | None = None
    initial_cash: float = Field(default=5_000_000, ge=100_000, le=10_000_000_000)
    timezone: str = "Asia/Shanghai"
    paper_run_time: time = time(15, 30)
    paper_slippage: float = Field(default=0.0005, ge=0, le=0.02)
    misfire_grace_seconds: int = Field(default=1800, ge=60, le=86400)
    actor: str = Field(default="local-operator", min_length=2, max_length=100)

    @model_validator(mode="after")
    def validate_program(self) -> ResearchProgramCreateRequest:
        validate_duration(self.duration)
        if self.paper_run_time < time(15, 10):
            raise ValueError("paper simulation must run after the A-share close")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone is not available") from exc
        return self


class ResearchProgramStatusRequest(BaseModel):
    status: Literal["active", "paused", "cancelled"]
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class StrategyApprovalRequest(BaseModel):
    actor: str = Field(min_length=2, max_length=100)
    reason: str = Field(min_length=10, max_length=2000)


class PortfolioCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=150)
    strategy_version_id: str
    dataset: str
    dataset_roll_policy: Literal["pinned", "latest_compatible"] = "pinned"
    initial_cash: float = Field(default=5_000_000, ge=100_000, le=10_000_000_000)
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class PairPortfolioCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=150)
    strategy_version_id: str
    dataset: str
    execution_snapshot: str
    minute_dataset: str
    shortability_dataset: str
    dataset_roll_policy: Literal["pinned", "latest_compatible"] = "pinned"
    execution_roll_policy: Literal["pinned", "latest_compatible"] = "pinned"
    initial_cash: float = Field(default=5_000_000, ge=100_000, le=10_000_000_000)
    actor: str = Field(default="local-operator", min_length=2, max_length=100)

    @model_validator(mode="after")
    def distinct_execution_evidence(self) -> PairPortfolioCreateRequest:
        if self.minute_dataset == self.shortability_dataset:
            raise ValueError("minute and shortability evidence must be separate datasets")
        return self


class StrategyAllocationMemberRequest(BaseModel):
    strategy_version_id: str
    weight: float | None = Field(default=None, gt=0, le=1)


class StrategyAllocationCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=150)
    dataset: str
    total_capital: float = Field(default=5_000_000, ge=500_000, le=10_000_000_000)
    allocation_method: Literal["risk_parity", "inverse_volatility", "fixed"] = "risk_parity"
    lookback_days: int = Field(default=252, ge=60, le=1260)
    target_volatility: float = Field(default=0.15, gt=0, le=0.50)
    max_pairwise_correlation: float = Field(default=0.70, gt=-1, lt=1)
    max_strategy_weight: float = Field(default=0.70, gt=0, le=1)
    max_member_drawdown: float = Field(default=0.08, gt=0, le=0.50)
    max_drawdown_reduce: float = Field(default=0.10, gt=0, le=0.50)
    max_drawdown_liquidate: float = Field(default=0.15, gt=0, le=0.50)
    members: list[StrategyAllocationMemberRequest] = Field(min_length=2, max_length=10)
    actor: str = Field(default="local-operator", min_length=2, max_length=100)

    @model_validator(mode="after")
    def validate_allocation(self) -> StrategyAllocationCreateRequest:
        ids = [item.strategy_version_id for item in self.members]
        if len(ids) != len(set(ids)):
            raise ValueError("strategy allocation members must be unique")
        if self.allocation_method == "fixed" and any(item.weight is None for item in self.members):
            raise ValueError("fixed allocation requires every member weight")
        if not (self.max_member_drawdown < self.max_drawdown_reduce < self.max_drawdown_liquidate):
            raise ValueError("drawdown thresholds must increase from member to liquidation")
        return self


class StrategyAllocationApprovalRequest(BaseModel):
    actor: str = Field(min_length=2, max_length=100)
    reason: str = Field(min_length=10, max_length=2000)


class StrategyAllocationStatusRequest(BaseModel):
    status: Literal["active", "paused"]
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class RiskEventAcknowledgementRequest(BaseModel):
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class RiskEventResolutionRequest(BaseModel):
    actor: str = Field(default="local-operator", min_length=2, max_length=100)
    reason: str = Field(min_length=10, max_length=2000)


class PortfolioStatusRequest(BaseModel):
    status: Literal["active", "paused", "closed"]


class PortfolioRebalanceRequest(BaseModel):
    as_of_date: date
    slippage: float = Field(default=0.0005, ge=0, le=0.02)


class ScheduleCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=150)
    kind: Literal[
        "incremental_sync",
        "data_pipeline",
        "ashare_5m_sync",
        "rdagent_research",
        "paper_rebalance",
        "pair_paper_rebalance",
        "broker_reconcile",
    ]
    timezone: str = "Asia/Shanghai"
    run_time: time = time(15, 30)
    trading_days_only: bool = True
    payload: dict
    misfire_grace_seconds: int = Field(default=1800, ge=60, le=86400)
    actor: str = Field(default="local-operator", min_length=2, max_length=100)

    @model_validator(mode="after")
    def validate_schedule(self) -> ScheduleCreateRequest:
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone is not available") from exc
        if self.kind == "rdagent_research":
            normalize_research_schedule_payload(self.payload, max_loops=20)
        elif self.kind == "paper_rebalance":
            if not self.payload.get("portfolio_id"):
                raise ValueError("paper_rebalance requires portfolio_id")
            if self.run_time < time(15, 10):
                raise ValueError("paper_rebalance must run after the A-share close")
        elif self.kind == "pair_paper_rebalance":
            if not self.payload.get("pair_portfolio_id"):
                raise ValueError("pair_paper_rebalance requires pair_portfolio_id")
            if self.run_time < time(15, 10):
                raise ValueError("pair_paper_rebalance must run after the A-share close")
        elif self.kind == "broker_reconcile":
            if not self.payload.get("destination_id"):
                raise ValueError("broker_reconcile requires destination_id")
            if self.run_time < time(15, 10):
                raise ValueError("broker_reconcile must run after the A-share close")
        elif self.kind == "incremental_sync":
            profile = self.payload.get("profile", "full")
            if profile not in {"core", "research", "full"}:
                raise ValueError("incremental_sync profile is invalid")
        elif self.kind == "ashare_5m_sync":
            if self.run_time < time(15, 10):
                raise ValueError("ashare_5m_sync must run after the A-share close")
            lookback_days = int(self.payload.get("lookback_days", 3))
            if not 1 <= lookback_days <= 30:
                raise ValueError("ashare_5m_sync lookback_days must be between 1 and 30")
        else:
            profile = self.payload.get("profile", "full")
            if profile not in {"core", "research", "full"}:
                raise ValueError("data_pipeline profile is invalid")
            allowed = {
                "cn_extended_daily",
                "cn_funds",
                "cn_macro",
                "cn_institutional",
                "cn_futures",
                "cn_options_bonds",
                "hk_market",
                "us_market",
                "global_markets",
                *DEFAULT_COVERAGE_BUNDLES,
            }
            bundles = self.payload.get("bundles") or sorted(allowed)
            if not isinstance(bundles, list) or not bundles:
                raise ValueError("data_pipeline requires at least one bundle")
            unknown = sorted({str(item) for item in bundles} - allowed)
            if unknown:
                raise ValueError(f"data_pipeline contains unsupported bundles: {unknown}")
        return self


class ScheduleStatusRequest(BaseModel):
    status: Literal["active", "paused"]


class AllocationScheduleRequest(BaseModel):
    timezone: str = "Asia/Shanghai"
    run_time: time = time(15, 30)
    trading_days_only: bool = True
    slippage: float = Field(default=0.0005, ge=0, le=0.02)
    misfire_grace_seconds: int = Field(default=1800, ge=60, le=86400)
    actor: str = Field(default="local-operator", min_length=2, max_length=100)

    @model_validator(mode="after")
    def validate_allocation_schedule(self) -> AllocationScheduleRequest:
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone is not available") from exc
        if self.run_time < time(15, 10):
            raise ValueError("allocation rebalances must run after the A-share close")
        return self


class AllocationScheduleStatusRequest(BaseModel):
    status: Literal["active", "paused"]
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class AllocationScheduleRetireRequest(BaseModel):
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class AlertActionRequest(BaseModel):
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class BrokerDestinationCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=150)
    account_ref: str = Field(min_length=2, max_length=150)
    portfolio_id: str = Field(min_length=16, max_length=100)
    config: dict[str, Any] = Field(default_factory=dict)
    actor: str = Field(default="local-admin", min_length=2, max_length=100)


class BrokerActorRequest(BaseModel):
    actor: str = Field(default="local-admin", min_length=2, max_length=100)


class BrokerBatchRequest(BrokerActorRequest):
    batch_id: str = Field(min_length=16, max_length=100)


class AuthBootstrapRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    display_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=12, max_length=256)


class AuthLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class AuthUserCreateRequest(AuthBootstrapRequest):
    role: Literal["admin", "researcher", "operator", "viewer"]


class AuthUserStatusRequest(BaseModel):
    active: bool


class AuthPasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=256)


class TushareSettingsRequest(BaseModel):
    api_url: str = Field(default="https://api.tushare.pro", min_length=8, max_length=500)
    token: str = Field(min_length=8, max_length=500)

    @model_validator(mode="after")
    def validate_values(self) -> TushareSettingsRequest:
        parsed = urlsplit(self.api_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("api_url must be an absolute HTTP(S) URL")
        if any(character.isspace() for character in self.token):
            raise ValueError("token must not contain whitespace")
        return self


class LlmSettingsRequest(BaseModel):
    api_key: str = Field(min_length=8, max_length=1000)
    api_base: str = Field(default="", max_length=500)
    chat_model: str = Field(default="gpt-4.1-mini", min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_values(self) -> LlmSettingsRequest:
        if any(character.isspace() for character in self.api_key):
            raise ValueError("api_key must not contain whitespace")
        if self.api_base:
            parsed = urlsplit(self.api_base)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("api_base must be an absolute HTTP(S) URL")
        return self


class AlertWebhookSettingsRequest(BaseModel):
    webhook_url: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def validate_values(self) -> AlertWebhookSettingsRequest:
        value = self.webhook_url.strip()
        if not value:
            return self
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("webhook_url must be an absolute HTTP(S) URL")
        local = parsed.hostname in {
            "127.0.0.1",
            "localhost",
            "host.docker.internal",
            "gateway.docker.internal",
        }
        if parsed.scheme != "https" and not local:
            raise ValueError("remote alert webhooks require HTTPS")
        return self


class BrokerGatewaySettingsRequest(BaseModel):
    gateway_url: str = Field(default="", max_length=1000)
    hmac_secret: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def validate_values(self) -> BrokerGatewaySettingsRequest:
        validate_broker_gateway_credentials(
            self.gateway_url,
            self.hmac_secret,
            allow_empty=True,
        )
        return self


def create_app(project_root: Path | None = None) -> FastAPI:
    project_root = (project_root or Path.cwd()).resolve()
    settings = Settings.from_env(project_root / ".env")
    settings.data_root.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointStore(settings.database_url)
    platform_root = settings.data_root / "platform"
    jobs = JobStore(settings.database_url)
    data_tasks = DataTaskStore(settings.database_url)
    research = ResearchStore(settings.database_url)
    strategies = StrategyStore(settings.database_url)
    portfolios = PortfolioStore(settings.database_url)
    pair_portfolios = PairPortfolioStore(settings.database_url)
    parameter_experiments = ParameterExperimentStore(settings.database_url)
    autonomous_research = AutonomousResearchOrchestrator(settings)
    continuous_research = ContinuousResearchController(settings)
    allocations = AllocationStore(settings.database_url)
    schedules = ScheduleStore(settings.database_url)
    alerts = AlertStore(settings.database_url)
    health_history = OperationalHealthStore(settings)
    brokers = BrokerStore(settings)
    auth = AuthStore(settings.database_url)
    runtime_secrets = RuntimeSecretStore(settings.database_url, settings.platform_secret_key)
    platform_configs = PlatformConfigStore(settings.database_url)
    retention = DataRetentionManager(settings.data_root, settings.database_url)
    market_dashboard = MarketOverviewService(settings.data_root)
    deployment_readiness = DeploymentReadinessStore(settings, project_root)
    worker = LocalJobWorker(jobs, project_root, settings)
    qlib_runtime: dict | None = None
    qlib_runtime_lock = Lock()
    rdagent_runtime: dict | None = None
    rdagent_runtime_lock = Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        data_tasks.sync_catalog()
        if settings.embedded_worker:
            worker.start()
        yield
        if settings.embedded_worker:
            worker.stop()

    app = FastAPI(
        title="Quant Research Platform",
        version="0.1.0",
        description="Local control plane for Tushare, Qlib, and RD-Agent workflows.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.auth_allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    public_api_paths = {
        "/api/health",
        "/api/auth/state",
        "/api/auth/bootstrap",
        "/api/auth/login",
    }

    def local_user() -> dict:
        return {
            "id": None,
            "username": "local-admin",
            "display_name": "Local Administrator",
            "role": "admin",
            "permissions": ["*"],
        }

    def authenticated_actor(request: Request, fallback: str = "local-operator") -> str:
        user = getattr(request.state, "user", None)
        return str(user["username"]) if user else fallback

    def tushare_settings() -> tuple[str, str]:
        stored = runtime_secrets.get("tushare")
        if stored:
            return normalize_api_url(stored.get("api_url", "")), stored.get("token", "")
        return settings.api_url, settings.token

    def effective_settings() -> Settings:
        api_url, token = tushare_settings()
        return replace(settings, api_url=api_url, token=token)

    def strategy_defaults_state() -> dict[str, Any]:
        record = platform_configs.get("multifactor_strategy_defaults")
        try:
            validated = StrategyConfigRequest.model_validate(
                record["value"] if record else {}
            ).model_dump()
        except ValueError as exc:
            raise HTTPException(
                503,
                "stored strategy defaults are invalid for this release; an administrator "
                "must review and save them again",
            ) from exc
        return {
            "config": validated,
            "source": "database" if record else "built_in",
            "revision": int(record["revision"]) if record else 0,
            "updated_by": record.get("updated_by") if record else None,
            "updated_at": record.get("updated_at") if record else None,
        }

    def require_qlib_dataset(
        name: str, *, purpose: str, frequency: str | None = None
    ) -> dict:
        available = {item["name"]: item for item in list_qlib_datasets(settings.data_root)}
        dataset = available.get(name)
        if not dataset or not dataset["ready"]:
            raise HTTPException(409, f"{purpose} Qlib dataset is not ready")
        if not dataset.get("reproducible"):
            raise HTTPException(
                409,
                f"{purpose} requires a Qlib dataset with immutable provenance metadata",
            )
        if frequency and dataset.get("frequency") != frequency:
            raise HTTPException(
                409,
                f"{purpose} requires a {frequency} Qlib dataset; selected dataset is "
                f"{dataset.get('frequency') or 'unknown'}",
            )
        return dataset

    def client_ip_hash(request: Request) -> str | None:
        forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        value = forwarded or (request.client.host if request.client else None)
        return auth.hash_ip(value)

    def set_session_cookie(response: Response, token: str) -> None:
        response.set_cookie(
            "quantlab_session",
            token,
            max_age=settings.auth_session_hours * 3600,
            httponly=True,
            secure=settings.auth_cookie_secure,
            samesite="strict",
            path="/",
        )

    @app.middleware("http")
    async def authentication_boundary(request: Request, call_next):
        path = request.url.path
        if request.method == "OPTIONS":
            return await call_next(request)
        token = request.cookies.get("quantlab_session")
        user = local_user() if settings.auth_mode == "disabled" else auth.validate_session(token)
        if user:
            user["permissions"] = sorted(ROLE_PERMISSIONS.get(user["role"], set()))
        request.state.user = user
        if path.startswith("/api") and path not in public_api_paths:
            if user is None:
                state = "bootstrap_required" if auth.user_count() == 0 else "login_required"
                return JSONResponse({"detail": state}, status_code=401)
            if request.method not in {"GET", "HEAD", "OPTIONS"}:
                origin = request.headers.get("origin")
                if origin:
                    origin_host = urlsplit(origin).netloc
                    request_host = request.headers.get("host", "")
                    if origin_host != request_host and origin not in settings.auth_allowed_origins:
                        auth.audit(
                            user=user,
                            username=user["username"],
                            action="request.origin_rejected",
                            method=request.method,
                            path=path,
                            status_code=403,
                            ip_hash=client_ip_hash(request),
                            user_agent=request.headers.get("user-agent"),
                        )
                        return JSONResponse({"detail": "request origin is not allowed"}, 403)
            required = permission_for(request.method, path)
            if not has_permission(user["role"], required):
                auth.audit(
                    user=user,
                    username=user["username"],
                    action="request.permission_denied",
                    method=request.method,
                    path=path,
                    status_code=403,
                    ip_hash=client_ip_hash(request),
                    user_agent=request.headers.get("user-agent"),
                    details={"required_permission": required},
                )
                return JSONResponse({"detail": "permission denied"}, status_code=403)
        response = await call_next(request)
        if path.startswith("/api/auth"):
            response.headers["Cache-Control"] = "no-store"
        if (
            path.startswith("/api")
            and request.method not in {"GET", "HEAD", "OPTIONS"}
            and path not in {"/api/auth/login", "/api/auth/bootstrap"}
            and user is not None
        ):
            auth.audit(
                user=user,
                username=user["username"],
                action="api.mutation",
                method=request.method,
                path=path,
                status_code=response.status_code,
                ip_hash=client_ip_hash(request),
                user_agent=request.headers.get("user-agent"),
            )
        return response

    @app.get("/api/auth/state")
    def auth_state(request: Request) -> dict:
        if settings.auth_mode == "disabled":
            return {"status": "disabled", "user": local_user()}
        user = auth.validate_session(request.cookies.get("quantlab_session"))
        if user:
            user["permissions"] = sorted(ROLE_PERMISSIONS.get(user["role"], set()))
            return {"status": "authenticated", "user": user}
        return {
            "status": "bootstrap_required" if auth.user_count() == 0 else "login_required",
            "user": None,
        }

    @app.post("/api/auth/bootstrap", status_code=201)
    def bootstrap_auth(payload: AuthBootstrapRequest, request: Request, response: Response) -> dict:
        if settings.auth_mode == "disabled":
            raise HTTPException(409, "authentication is disabled")
        try:
            auth.bootstrap_admin(
                username=payload.username,
                display_name=payload.display_name,
                password=payload.password,
            )
            user, token, _expires = auth.login(
                username=payload.username,
                password=payload.password,
                session_hours=settings.auth_session_hours,
                ip_hash=client_ip_hash(request),
                user_agent=request.headers.get("user-agent"),
            )
        except ValueError as exc:
            auth.audit(
                user=None,
                username=payload.username[:64],
                action="auth.bootstrap_failed",
                method="POST",
                path="/api/auth/bootstrap",
                status_code=409,
                ip_hash=client_ip_hash(request),
                user_agent=request.headers.get("user-agent"),
            )
            raise HTTPException(409, str(exc)) from exc
        set_session_cookie(response, token)
        auth.audit(
            user=user,
            username=user["username"],
            action="auth.bootstrap_succeeded",
            method="POST",
            path="/api/auth/bootstrap",
            status_code=201,
            ip_hash=client_ip_hash(request),
            user_agent=request.headers.get("user-agent"),
        )
        return user

    @app.post("/api/auth/login")
    def login_auth(payload: AuthLoginRequest, request: Request, response: Response) -> dict:
        if settings.auth_mode == "disabled":
            raise HTTPException(409, "authentication is disabled")
        try:
            user, token, _expires = auth.login(
                username=payload.username,
                password=payload.password,
                session_hours=settings.auth_session_hours,
                ip_hash=client_ip_hash(request),
                user_agent=request.headers.get("user-agent"),
            )
        except AuthenticationError as exc:
            auth.audit(
                user=None,
                username=payload.username[:64],
                action="auth.login_failed",
                method="POST",
                path="/api/auth/login",
                status_code=401,
                ip_hash=client_ip_hash(request),
                user_agent=request.headers.get("user-agent"),
            )
            raise HTTPException(401, str(exc)) from exc
        set_session_cookie(response, token)
        auth.audit(
            user=user,
            username=user["username"],
            action="auth.login_succeeded",
            method="POST",
            path="/api/auth/login",
            status_code=200,
            ip_hash=client_ip_hash(request),
            user_agent=request.headers.get("user-agent"),
        )
        return user

    @app.post("/api/auth/logout")
    def logout_auth(request: Request, response: Response) -> dict[str, str]:
        auth.logout(request.cookies.get("quantlab_session"))
        response.delete_cookie("quantlab_session", path="/")
        return {"status": "logged_out"}

    @app.get("/api/auth/me")
    def auth_me(request: Request) -> dict:
        return request.state.user

    @app.get("/api/auth/users")
    def list_auth_users(limit: int = Query(200, ge=1, le=500)) -> list[dict]:
        return auth.list_users(limit)

    @app.post("/api/auth/users", status_code=201)
    def create_auth_user(payload: AuthUserCreateRequest) -> dict:
        try:
            return auth.create_user(
                username=payload.username,
                display_name=payload.display_name,
                role=payload.role,
                password=payload.password,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/auth/users/{user_id}/status")
    def set_auth_user_status(user_id: str, payload: AuthUserStatusRequest) -> dict:
        try:
            return auth.set_active(user_id, payload.active)
        except KeyError as exc:
            raise HTTPException(404, "user not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/auth/password")
    def change_auth_password(payload: AuthPasswordRequest, request: Request) -> dict[str, str]:
        try:
            auth.change_password(
                request.state.user["id"],
                current_password=payload.current_password,
                new_password=payload.new_password,
                keep_session_id=request.state.user.get("session_id"),
            )
        except (AuthenticationError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"status": "password_changed"}

    @app.get("/api/audit")
    def list_audit_events(limit: int = Query(200, ge=1, le=1000)) -> list[dict]:
        return auth.list_audit(limit)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        with jobs.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        secret_storage = runtime_secrets.health()
        if secret_storage["status"] != "ok":
            raise HTTPException(
                503,
                detail={
                    "status": "unavailable",
                    "database": "postgresql",
                    "runtime_secret_storage": secret_storage["status"],
                    "message": secret_storage["message"],
                },
            )
        return {
            "status": "ok",
            "database": "postgresql",
            "worker_mode": "embedded" if settings.embedded_worker else "external",
            "runtime_secret_storage": "ok",
            "runtime_secret_records": int(secret_storage["record_count"]),
        }

    @app.get("/api/settings")
    def runtime_settings_status() -> dict:
        secret_storage = runtime_secrets.health()
        tushare_record = runtime_secrets.describe("tushare")
        llm_record = runtime_secrets.describe("llm")
        alert_record = runtime_secrets.describe("alert_webhook")
        broker_record = runtime_secrets.describe("broker_gateway")
        return {
            "storage_ready": secret_storage["status"] == "ok",
            "storage_status": secret_storage["status"],
            "storage_record_count": secret_storage["record_count"],
            "tushare": {
                "configured": bool(tushare_record or (settings.api_url and settings.token)),
                "source": "database"
                if tushare_record
                else ("environment" if settings.api_url and settings.token else "missing"),
                "api_url": (
                    (tushare_record or {}).get("metadata_json", {}).get("api_url")
                    or settings.api_url
                    or "https://api.tushare.pro"
                ),
                "verified_at": ((tushare_record or {}).get("metadata_json", {}).get("verified_at")),
                "updated_at": (tushare_record or {}).get("updated_at"),
            },
            "llm": {
                "configured": bool(llm_record or os.getenv(settings.rdagent_llm_key_env)),
                "source": "database"
                if llm_record
                else ("environment" if os.getenv(settings.rdagent_llm_key_env) else "missing"),
                "api_base": (
                    (llm_record or {}).get("metadata_json", {}).get("api_base")
                    or os.getenv("OPENAI_API_BASE", "")
                ),
                "chat_model": (
                    (llm_record or {}).get("metadata_json", {}).get("chat_model")
                    or os.getenv("CHAT_MODEL", "gpt-4.1-mini")
                ),
                "updated_at": (llm_record or {}).get("updated_at"),
            },
            "alerts": {
                "configured": bool(
                    (alert_record or {}).get("metadata_json", {}).get("enabled")
                    if alert_record
                    else settings.alert_webhook_url
                ),
                "source": "database"
                if alert_record
                else ("environment" if settings.alert_webhook_url else "missing"),
                "endpoint_host": (
                    (alert_record or {}).get("metadata_json", {}).get("endpoint_host", "")
                    if alert_record
                    else (
                        urlsplit(settings.alert_webhook_url).hostname
                        if settings.alert_webhook_url
                        else ""
                    )
                ),
                "updated_at": (alert_record or {}).get("updated_at"),
            },
            "broker": {
                "mode": settings.broker_mode,
                "configured": bool(
                    (broker_record or {}).get("metadata_json", {}).get("enabled")
                    if broker_record
                    else settings.broker_gateway_url and settings.broker_hmac_secret
                ),
                "source": "database"
                if broker_record
                else (
                    "environment"
                    if settings.broker_gateway_url and settings.broker_hmac_secret
                    else "missing"
                ),
                "endpoint_host": (
                    (broker_record or {}).get("metadata_json", {}).get("endpoint_host", "")
                    if broker_record
                    else (
                        urlsplit(settings.broker_gateway_url).hostname
                        if settings.broker_gateway_url
                        else ""
                    )
                ),
                "updated_at": (broker_record or {}).get("updated_at"),
            },
        }

    @app.get("/api/settings/strategy-defaults")
    def get_strategy_defaults() -> dict:
        return strategy_defaults_state()

    @app.put("/api/settings/strategy-defaults")
    def update_strategy_defaults(
        payload: StrategyDefaultsUpdateRequest,
        request: Request,
    ) -> dict:
        try:
            platform_configs.put(
                "multifactor_strategy_defaults",
                payload.config.model_dump(),
                actor=authenticated_actor(request, "local-admin"),
                reason=payload.reason,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return strategy_defaults_state()

    @app.get("/api/settings/strategy-defaults/revisions")
    def list_strategy_default_revisions(
        limit: int = Query(50, ge=1, le=200),
    ) -> list[dict]:
        return platform_configs.list_revisions("multifactor_strategy_defaults", limit)

    @app.post("/api/settings/tushare")
    def update_tushare_settings(payload: TushareSettingsRequest, request: Request) -> dict:
        api_url = normalize_api_url(payload.api_url)
        today = date.today().strftime("%Y%m%d")
        try:
            result = requests.post(
                api_url,
                json={
                    "api_name": "trade_cal",
                    "token": payload.token,
                    "params": {
                        "exchange": "SSE",
                        "start_date": today,
                        "end_date": today,
                    },
                    "fields": "exchange,cal_date,is_open,pretrade_date",
                },
                timeout=15,
            )
            result.raise_for_status()
            body = result.json()
        except (requests.RequestException, ValueError) as exc:
            raise HTTPException(409, "Tushare credential validation failed") from exc
        if not isinstance(body, dict) or body.get("code") != 0:
            raise HTTPException(409, "Tushare rejected the credential")
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        try:
            runtime_secrets.put(
                "tushare",
                {"api_url": api_url, "token": payload.token},
                metadata={"api_url": api_url, "verified_at": timestamp},
                updated_by=request.state.user.get("id"),
            )
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"status": "saved", "configured": True, "verified_at": timestamp}

    @app.post("/api/settings/llm")
    def update_llm_settings(payload: LlmSettingsRequest, request: Request) -> dict:
        try:
            runtime_secrets.put(
                "llm",
                {
                    "api_key": payload.api_key,
                    "api_base": payload.api_base.strip().rstrip("/"),
                    "chat_model": payload.chat_model.strip(),
                },
                metadata={
                    "api_base": payload.api_base.strip().rstrip("/"),
                    "chat_model": payload.chat_model.strip(),
                },
                updated_by=request.state.user.get("id"),
            )
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"status": "saved", "configured": True}

    @app.post("/api/settings/alerts")
    def update_alert_webhook_settings(
        payload: AlertWebhookSettingsRequest,
        request: Request,
    ) -> dict:
        webhook_url = payload.webhook_url.strip()
        try:
            runtime_secrets.put(
                "alert_webhook",
                {"webhook_url": webhook_url},
                metadata={
                    "enabled": bool(webhook_url),
                    "endpoint_host": urlsplit(webhook_url).hostname or "" if webhook_url else "",
                },
                updated_by=request.state.user.get("id"),
            )
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"status": "saved", "configured": bool(webhook_url)}

    @app.post("/api/settings/broker")
    def update_broker_gateway_settings(
        payload: BrokerGatewaySettingsRequest,
        request: Request,
    ) -> dict:
        gateway_url, hmac_secret = validate_broker_gateway_credentials(
            payload.gateway_url,
            payload.hmac_secret,
            allow_empty=True,
        )
        try:
            runtime_secrets.put(
                "broker_gateway",
                {"gateway_url": gateway_url, "hmac_secret": hmac_secret},
                metadata={
                    "enabled": bool(gateway_url and hmac_secret),
                    "endpoint_host": urlsplit(gateway_url).hostname or "" if gateway_url else "",
                },
                updated_by=request.state.user.get("id"),
            )
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {
            "status": "saved",
            "configured": bool(gateway_url and hmac_secret),
            "mode": settings.broker_mode,
        }

    @app.get("/api/overview")
    def overview() -> dict:
        return system_summary(
            effective_settings(), checkpoint, jobs.list(), data_tasks.list()
        )

    @app.get("/api/datasets")
    def datasets() -> list[dict]:
        return dataset_catalog(checkpoint)

    @app.get("/api/market/overview")
    def market_overview(
        snapshot: str | None = Query(default=None, min_length=3, max_length=120),
        symbols: str | None = Query(default=None, max_length=1000),
    ) -> dict:
        requested_symbols = symbols.split(",") if symbols else None
        try:
            return market_dashboard.get(snapshot_name=snapshot, symbols=requested_symbols)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/snapshots")
    def snapshots() -> list[dict]:
        return list_snapshots(settings.data_root)

    @app.get("/api/data-retention")
    def data_retention_plan(
        keep_latest: int = Query(7, ge=1, le=100),
        min_age_days: int = Query(14, ge=1, le=3650),
    ) -> dict:
        return retention.plan(keep_latest=keep_latest, min_age_days=min_age_days)

    @app.post("/api/data-retention/apply")
    def apply_data_retention(payload: RetentionApplyRequest, request: Request) -> dict:
        try:
            result = retention.apply(
                payload.names,
                confirmation=payload.confirmation,
                keep_latest=payload.keep_latest,
                min_age_days=payload.min_age_days,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        auth.audit(
            user=request.state.user,
            username=authenticated_actor(request),
            action="data.retention_applied",
            method="POST",
            path="/api/data-retention/apply",
            status_code=200,
            ip_hash=client_ip_hash(request),
            user_agent=request.headers.get("user-agent"),
            details={
                "datasets": [item["name"] for item in result["deleted"]],
                "reclaimed_bytes": result["reclaimed_bytes"],
            },
        )
        return result

    @app.get("/api/qlib/status")
    def qlib_status(refresh: bool = False) -> dict:
        nonlocal qlib_runtime
        with qlib_runtime_lock:
            if qlib_runtime is None or refresh:
                result = probe_qlib(settings, project_root)
                if result.get("status") == "ok":
                    qlib_runtime = result
                return result
        return qlib_runtime

    @app.get("/api/qlib/datasets")
    def qlib_datasets() -> list[dict]:
        return list_qlib_datasets(settings.data_root)

    @app.get("/api/qlib/experiments")
    def qlib_experiments() -> list[dict]:
        return list_qlib_experiments(settings.data_root)

    @app.get("/api/rdagent/status")
    def rdagent_status(refresh: bool = False) -> dict:
        nonlocal rdagent_runtime
        with rdagent_runtime_lock:
            if rdagent_runtime is None or refresh:
                result = probe_rdagent(settings, project_root)
                if result.get("status") == "ok":
                    rdagent_runtime = result
                return result
        return rdagent_runtime

    @app.get("/api/rdagent/runs")
    def list_research_runs(limit: int = Query(50, ge=1, le=200)) -> list[dict]:
        return research.list_runs(limit)

    @app.get("/api/rdagent/runs/{run_id}")
    def get_research_run(run_id: str) -> dict:
        try:
            run = research.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(404, "research run not found") from exc
        run["candidates"] = research.list_candidates(run_id=run_id)
        run["events"] = research.list_events(run_id)
        return run

    @app.post("/api/rdagent/runs", status_code=202)
    def create_research_run(payload: RDAgentRunRequest, request: Request) -> dict:
        runtime = probe_rdagent(settings, project_root)
        if not runtime.get("ready"):
            blockers = runtime.get("blockers") or [runtime.get("error") or "runtime unavailable"]
            raise HTTPException(409, {"message": "RD-Agent is not ready", "blockers": blockers})
        if payload.loop_n > settings.rdagent_max_loops:
            raise HTTPException(
                422, f"loop_n exceeds configured limit {settings.rdagent_max_loops}"
            )
        dataset = require_qlib_dataset(
            payload.dataset, purpose="RD-Agent research", frequency="day"
        )
        periods = payload.periods.model_dump(mode="json")
        if (
            dataset.get("start_date")
            and payload.periods.train_start.isoformat() < dataset["start_date"]
        ):
            raise HTTPException(409, "training window starts before the selected dataset")
        if dataset.get("end_date") and payload.periods.test_end.isoformat() > dataset["end_date"]:
            raise HTTPException(409, "test window ends after the selected dataset")
        artifact = settings.data_root / "artifacts" / "rdagent"
        try:
            run = research.create_run(
                kind="factor",
                objective=payload.objective,
                dataset=payload.dataset,
                requested_by=authenticated_actor(request, payload.requested_by),
                budget={"loop_n": payload.loop_n, "duration": payload.duration},
                config={"periods": periods, "dataset_path": dataset["path"]},
                artifact_path=artifact,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        log_path = platform_root / "logs" / f"rdagent-factor-{run['id']}.log"
        try:
            job = jobs.create(
                "rdagent_factor",
                {
                    "research_run_id": run["id"],
                    "dataset": payload.dataset,
                    "dataset_path": dataset["path"],
                    "objective": payload.objective,
                    "loop_n": payload.loop_n,
                    "duration": payload.duration,
                    "periods": periods,
                },
                log_path,
            )
        except ValueError as exc:
            research.mark_run(run["id"], "failed", actor="api", error=str(exc))
            raise HTTPException(409, str(exc)) from exc
        research.attach_job(run["id"], job["id"])
        worker.notify()
        return research.get_run(run["id"])

    @app.get("/api/research-programs")
    def list_research_programs(
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        return continuous_research.programs.list(limit=limit)

    @app.get("/api/research-programs/{program_id}")
    def get_research_program(program_id: str) -> dict[str, Any]:
        try:
            return continuous_research.programs.get(program_id)
        except KeyError as exc:
            raise HTTPException(404, "research program not found") from exc

    @app.post("/api/research-programs", status_code=201)
    def create_research_program(
        payload: ResearchProgramCreateRequest, request: Request
    ) -> dict[str, Any]:
        if payload.loop_n > settings.rdagent_max_loops:
            raise HTTPException(
                422, f"loop_n exceeds configured limit {settings.rdagent_max_loops}"
            )
        dataset = require_qlib_dataset(
            payload.dataset, purpose="continuous research", frequency="day"
        )
        if not dataset.get("lineage_verified") or not dataset.get("lineage_id"):
            raise HTTPException(
                409, "continuous research requires a verified Qlib dataset lineage"
            )
        try:
            recipe = get_strategy_recipe(payload.recipe_id)
            actor = authenticated_actor(request, payload.actor)
            objective = payload.objective or recipe["rdagent_objective"]
            if payload.strategy_config is None:
                strategy_config = StrategyConfigRequest.model_validate(
                    {
                        **strategy_defaults_state()["config"],
                        **recipe["config_overrides"],
                        "recipe_id": recipe["id"],
                        "recipe_version": recipe["version"],
                    }
                ).model_dump()
            else:
                strategy_config = payload.strategy_config.model_dump()
            parameter_grid, trial_parameters = normalize_parameter_grid(
                payload.parameter_grid, max_trials=payload.max_trials
            )
            experiment_trials = [
                {
                    "parameters": parameters,
                    "config": StrategyConfigRequest.model_validate(
                        {**strategy_config, **parameters}
                    ).model_dump(),
                }
                for parameters in trial_parameters
            ]
            program = continuous_research.programs.create(
                name=payload.name,
                recipe_id=payload.recipe_id,
                objective=objective,
                benchmark=payload.benchmark or recipe["benchmark"],
                universe=payload.universe or recipe["universe"],
                dataset_lineage_id=str(dataset["lineage_id"]),
                config={
                    "window_days": {
                        "train": payload.train_trading_days,
                        "validation": payload.validation_trading_days,
                        "test": payload.test_trading_days,
                    },
                    "loop_n": payload.loop_n,
                    "duration": payload.duration,
                    "max_factors": payload.max_factors,
                    "strategy_config": strategy_config,
                    "parameter_grid": parameter_grid,
                    "experiment_trials": experiment_trials,
                    "paper": {
                        "initial_cash": payload.initial_cash,
                        "timezone": payload.timezone,
                        "run_time": payload.paper_run_time.isoformat(timespec="minutes"),
                        "slippage": payload.paper_slippage,
                        "misfire_grace_seconds": payload.misfire_grace_seconds,
                    },
                    "manual_strategy_approval": True,
                    "champion_min_score_improvement": payload.champion_min_score_improvement,
                    "champion_decay_fraction": payload.champion_decay_fraction,
                },
                min_new_trading_days=payload.min_new_trading_days,
                max_active_campaigns=payload.max_active_campaigns,
                actor=actor,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return program

    @app.post("/api/research-programs/{program_id}/status")
    def set_research_program_status(
        program_id: str,
        payload: ResearchProgramStatusRequest,
        request: Request,
    ) -> dict[str, Any]:
        try:
            return continuous_research.programs.set_status(
                program_id,
                payload.status,
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "research program not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/research-programs/{program_id}/check-now")
    def check_research_program_now(program_id: str, request: Request) -> dict[str, Any]:
        try:
            return continuous_research.programs.check_now(
                program_id, actor=authenticated_actor(request)
            )
        except KeyError as exc:
            raise HTTPException(404, "research program not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/research-campaigns")
    def list_research_campaigns(
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        return autonomous_research.campaigns.list(limit=limit)

    @app.get("/api/research-campaigns/{campaign_id}")
    def get_research_campaign(campaign_id: str) -> dict[str, Any]:
        try:
            return autonomous_research.campaigns.get(campaign_id)
        except KeyError as exc:
            raise HTTPException(404, "research campaign not found") from exc

    @app.post("/api/research-campaigns", status_code=202)
    def create_research_campaign(
        payload: ResearchCampaignCreateRequest, request: Request
    ) -> dict[str, Any]:
        runtime = probe_rdagent(settings, project_root)
        if not runtime.get("ready"):
            blockers = runtime.get("blockers") or [runtime.get("error") or "runtime unavailable"]
            raise HTTPException(409, {"message": "RD-Agent is not ready", "blockers": blockers})
        if payload.loop_n > settings.rdagent_max_loops:
            raise HTTPException(
                422, f"loop_n exceeds configured limit {settings.rdagent_max_loops}"
            )
        dataset = require_qlib_dataset(
            payload.dataset, purpose="autonomous research", frequency="day"
        )
        if (
            dataset.get("start_date")
            and payload.periods.train_start.isoformat() < dataset["start_date"]
        ):
            raise HTTPException(409, "training window starts before the selected dataset")
        if dataset.get("end_date") and payload.periods.test_end.isoformat() > dataset["end_date"]:
            raise HTTPException(409, "test window ends after the selected dataset")
        try:
            recipe = get_strategy_recipe(payload.recipe_id)
            actor = authenticated_actor(request, payload.actor)
            periods = payload.periods.model_dump(mode="json")
            research_payload = normalize_research_schedule_payload(
                {
                    "objective": payload.objective,
                    "dataset": payload.dataset,
                    "loop_n": payload.loop_n,
                    "duration": payload.duration,
                    "requested_by": actor,
                    "periods": periods,
                },
                max_loops=settings.rdagent_max_loops,
            )
            if payload.strategy_config is None:
                strategy_config = StrategyConfigRequest.model_validate(
                    {
                        **strategy_defaults_state()["config"],
                        **recipe["config_overrides"],
                        "recipe_id": recipe["id"],
                        "recipe_version": recipe["version"],
                    }
                ).model_dump()
            else:
                strategy_config = payload.strategy_config.model_dump()
            parameter_grid, trial_parameters = normalize_parameter_grid(
                payload.parameter_grid, max_trials=payload.max_trials
            )
            experiment_trials = [
                {
                    "parameters": parameters,
                    "config": StrategyConfigRequest.model_validate(
                        {**strategy_config, **parameters}
                    ).model_dump(),
                }
                for parameters in trial_parameters
            ]
            campaign = autonomous_research.create(
                name=payload.name,
                objective=payload.objective,
                dataset=payload.dataset,
                benchmark=payload.benchmark or recipe["benchmark"],
                universe=payload.universe or recipe["universe"],
                recipe_id=payload.recipe_id,
                config={
                    "research": research_payload,
                    "strategy_config": strategy_config,
                    "backtest_periods": {
                        "start": payload.periods.test_start.isoformat(),
                        "end": payload.periods.test_end.isoformat(),
                    },
                    "experiment_periods": split_research_period(
                        payload.periods.test_start, payload.periods.test_end
                    ),
                    "parameter_grid": parameter_grid,
                    "experiment_trials": experiment_trials,
                    "max_factors": payload.max_factors,
                    "paper": {
                        "initial_cash": payload.initial_cash,
                        "timezone": payload.timezone,
                        "run_time": payload.paper_run_time.isoformat(timespec="minutes"),
                        "slippage": payload.paper_slippage,
                        "misfire_grace_seconds": payload.misfire_grace_seconds,
                    },
                    "manual_strategy_approval": True,
                },
                actor=actor,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        autonomous_research.tick(limit=1)
        worker.notify()
        return autonomous_research.campaigns.get(campaign["id"])

    @app.post("/api/research-campaigns/{campaign_id}/status")
    def set_research_campaign_status(
        campaign_id: str,
        payload: ResearchCampaignStatusRequest,
        request: Request,
    ) -> dict[str, Any]:
        try:
            result = autonomous_research.campaigns.set_status(
                campaign_id,
                payload.status,
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "research campaign not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if payload.status == "running":
            autonomous_research.tick(limit=1)
            worker.notify()
            return autonomous_research.campaigns.get(campaign_id)
        return result

    @app.post("/api/research-campaigns/{campaign_id}/retry")
    def retry_research_campaign(campaign_id: str, request: Request) -> dict[str, Any]:
        try:
            autonomous_research.retry(
                campaign_id, actor=authenticated_actor(request)
            )
        except KeyError as exc:
            raise HTTPException(404, "research campaign not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        autonomous_research.tick(limit=1)
        worker.notify()
        return autonomous_research.campaigns.get(campaign_id)

    @app.get("/api/factors")
    def list_factors(
        status: str | None = None,
        run_id: str | None = None,
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict]:
        return research.list_candidates(run_id=run_id, status=status, limit=limit)

    @app.get("/api/factors/gate-policy")
    def factor_gate_policy() -> dict:
        return research.policy_summary()

    @app.post("/api/factors/{candidate_id}/evaluations", status_code=201)
    def record_factor_evaluation(candidate_id: str, payload: FactorEvaluationRequest) -> dict:
        try:
            return research.record_evaluation(
                candidate_id,
                dataset=payload.dataset,
                **payload.periods.model_dump(),
                metrics=payload.metrics,
                artifact_path=payload.artifact_path,
            )
        except KeyError as exc:
            raise HTTPException(404, "factor candidate not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/factors/{candidate_id}/promote")
    def promote_factor(candidate_id: str, payload: PromotionRequest, request: Request) -> dict:
        try:
            return research.promote(
                candidate_id,
                actor=authenticated_actor(request, payload.actor),
                reason=payload.reason,
            )
        except KeyError as exc:
            raise HTTPException(404, "factor candidate not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/strategies")
    def list_strategies(limit: int = Query(100, ge=1, le=500)) -> list[dict]:
        return strategies.list(limit)

    @app.get("/api/strategy-recipes")
    def strategy_recipes() -> dict[str, Any]:
        return {"version": RECIPE_VERSION, "recipes": list_strategy_recipes()}

    @app.get("/api/strategy-recipes/{recipe_id}")
    def strategy_recipe(recipe_id: str) -> dict[str, Any]:
        try:
            return get_strategy_recipe(recipe_id)
        except KeyError as exc:
            raise HTTPException(404, "strategy recipe not found") from exc

    @app.post("/api/strategies", status_code=201)
    def create_strategy(payload: StrategyCreateRequest, request: Request) -> dict:
        try:
            return strategies.create(
                name=payload.name,
                description=payload.description,
                benchmark=payload.benchmark,
                universe=payload.universe,
                factors=[item.model_dump() for item in payload.factors],
                config=(
                    payload.config.model_dump()
                    if payload.config is not None
                    else strategy_defaults_state()["config"]
                ),
                actor=authenticated_actor(request, payload.actor),
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/strategies/{strategy_id}/versions", status_code=201)
    def create_strategy_version(
        strategy_id: str, payload: StrategyVersionCreateRequest, request: Request
    ) -> dict:
        try:
            return strategies.create_version(
                strategy_id,
                benchmark=payload.benchmark,
                universe=payload.universe,
                factors=[item.model_dump() for item in payload.factors],
                config=(
                    payload.config.model_dump()
                    if payload.config is not None
                    else strategy_defaults_state()["config"]
                ),
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/pair-strategies", status_code=201)
    def create_pair_strategy(payload: PairStrategyCreateRequest, request: Request) -> dict:
        try:
            return strategies.create_pair(
                name=payload.name,
                description=payload.description,
                leg_y=payload.leg_y,
                leg_x=payload.leg_x,
                asset_class=payload.asset_class,
                shorting_mode=payload.shorting_mode,
                config=payload.config.model_dump(),
                actor=authenticated_actor(request, payload.actor),
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/pair-strategies/{strategy_id}/versions", status_code=201)
    def create_pair_strategy_version(
        strategy_id: str, payload: PairStrategyVersionCreateRequest, request: Request
    ) -> dict:
        try:
            return strategies.create_pair_version(
                strategy_id,
                leg_y=payload.leg_y,
                leg_x=payload.leg_x,
                asset_class=payload.asset_class,
                shorting_mode=payload.shorting_mode,
                config=payload.config.model_dump(),
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/backtests")
    def list_backtests(
        version_id: str | None = None,
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict]:
        return strategies.list_backtests(version_id=version_id, limit=limit)

    @app.get("/api/parameter-experiments")
    def list_parameter_experiments(
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict]:
        return parameter_experiments.list(limit=limit)

    @app.get("/api/parameter-experiments/{experiment_id}")
    def get_parameter_experiment(experiment_id: str) -> dict:
        try:
            return parameter_experiments.get(experiment_id)
        except KeyError as exc:
            raise HTTPException(404, "parameter experiment not found") from exc

    @app.post(
        "/api/strategy-versions/{version_id}/parameter-experiments", status_code=202
    )
    def create_parameter_experiment(
        version_id: str, payload: ParameterExperimentRequest, request: Request
    ) -> dict:
        dataset = require_qlib_dataset(
            payload.dataset, purpose="parameter experiment", frequency="day"
        )
        if dataset.get("start_date") and payload.start.isoformat() < dataset["start_date"]:
            raise HTTPException(409, "experiment starts before the selected dataset")
        if dataset.get("end_date") and payload.end.isoformat() > dataset["end_date"]:
            raise HTTPException(409, "experiment ends after the selected dataset")
        try:
            version = strategies.get_version(version_id)
            if version.get("strategy_type") != "multifactor":
                raise ValueError("parameter experiments require a multifactor strategy")
            parameter_grid, trial_parameters = normalize_parameter_grid(
                payload.parameter_grid, max_trials=payload.max_trials
            )
            trial_configs = []
            for parameters in trial_parameters:
                config = StrategyConfigRequest.model_validate(
                    {**version["config"], **parameters}
                ).model_dump()
                trial_configs.append({"parameters": parameters, "config": config})
            experiment = parameter_experiments.create(
                strategy_version_id=version_id,
                dataset=payload.dataset,
                periods=split_research_period(payload.start, payload.end),
                parameter_grid=parameter_grid,
                baseline_config=version["config"],
                trials=trial_configs,
                artifact_root=settings.data_root / "artifacts" / "parameter-experiments",
                created_by=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy version not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        log_path = platform_root / "logs" / f"parameter-experiment-{experiment['id']}.log"
        try:
            job = jobs.create(
                "parameter_experiment",
                {
                    "parameter_experiment_id": experiment["id"],
                    "strategy_version_id": version_id,
                    "dataset": payload.dataset,
                    "dataset_path": dataset["path"],
                },
                log_path,
            )
        except ValueError as exc:
            parameter_experiments.mark(experiment["id"], "failed", error=str(exc))
            raise HTTPException(409, str(exc)) from exc
        parameter_experiments.attach_job(experiment["id"], job["id"])
        worker.notify()
        return parameter_experiments.get(experiment["id"])

    @app.post("/api/strategy-versions/{version_id}/backtests", status_code=202)
    def create_strategy_backtest(version_id: str, payload: StrategyBacktestRequest) -> dict:
        dataset = require_qlib_dataset(
            payload.dataset, purpose="strategy backtest", frequency="day"
        )
        if dataset.get("start_date") and payload.start.isoformat() < dataset["start_date"]:
            raise HTTPException(409, "backtest starts before the selected dataset")
        if dataset.get("end_date") and payload.end.isoformat() > dataset["end_date"]:
            raise HTTPException(409, "backtest ends after the selected dataset")
        artifact_root = settings.data_root / "artifacts" / "backtests"
        try:
            version = strategies.get_version(version_id)
            if version.get("strategy_type") != "multifactor":
                raise ValueError("pair strategy versions require the pair-backtests endpoint")
            backtest = strategies.create_backtest(
                version_id=version_id,
                dataset=payload.dataset,
                periods={"start": payload.start.isoformat(), "end": payload.end.isoformat()},
                artifact_path=artifact_root,
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy version not found") from exc
        log_path = platform_root / "logs" / f"strategy-backtest-{backtest['id']}.log"
        try:
            job = jobs.create(
                "strategy_backtest",
                {
                    "backtest_id": backtest["id"],
                    "strategy_version_id": version_id,
                    "dataset": payload.dataset,
                    "dataset_path": dataset["path"],
                    "periods": backtest["periods"],
                },
                log_path,
            )
        except ValueError as exc:
            strategies.mark_backtest(backtest["id"], "failed", error=str(exc))
            raise HTTPException(409, str(exc)) from exc
        strategies.attach_job(backtest["id"], job["id"])
        worker.notify()
        return strategies.get_backtest(backtest["id"])

    @app.post("/api/strategy-versions/{version_id}/pair-backtests", status_code=202)
    def create_pair_strategy_backtest(
        version_id: str, payload: PairStrategyBacktestRequest
    ) -> dict:
        dataset = require_qlib_dataset(
            payload.dataset, purpose="pair strategy backtest", frequency="day"
        )
        if dataset.get("start_date") and payload.start.isoformat() < dataset["start_date"]:
            raise HTTPException(409, "backtest starts before the selected daily dataset")
        if dataset.get("end_date") and payload.end.isoformat() > dataset["end_date"]:
            raise HTTPException(409, "backtest ends after the selected daily dataset")
        try:
            version = strategies.get_version(version_id)
            if version.get("strategy_type") != "pair":
                raise ValueError("pair backtests require a pair strategy version")
            minute = resolve_snapshot_dataset(
                settings.data_root,
                snapshot_name=payload.execution_snapshot,
                dataset_name=payload.minute_dataset,
            )
            shortability = resolve_snapshot_dataset(
                settings.data_root,
                snapshot_name=payload.execution_snapshot,
                dataset_name=payload.shortability_dataset,
            )
            artifact_root = settings.data_root / "artifacts" / "backtests"
            execution_dataset = (
                f"{payload.execution_snapshot}/{payload.minute_dataset}"
                f"+{payload.shortability_dataset}"
            )
            backtest = strategies.create_backtest(
                version_id=version_id,
                dataset=payload.dataset,
                execution_dataset=execution_dataset,
                periods={"start": payload.start.isoformat(), "end": payload.end.isoformat()},
                artifact_path=artifact_root,
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy version not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        log_path = platform_root / "logs" / f"pair-backtest-{backtest['id']}.log"
        try:
            job = jobs.create(
                "pair_backtest",
                {
                    "backtest_id": backtest["id"],
                    "strategy_version_id": version_id,
                    "dataset": payload.dataset,
                    "dataset_path": dataset["path"],
                    "daily_provenance": dataset["provenance"],
                    "execution_snapshot": payload.execution_snapshot,
                    "minute_dataset": minute,
                    "shortability_dataset": shortability,
                    "periods": backtest["periods"],
                },
                log_path,
            )
        except ValueError as exc:
            strategies.mark_backtest(backtest["id"], "failed", error=str(exc))
            raise HTTPException(409, str(exc)) from exc
        strategies.attach_job(backtest["id"], job["id"])
        worker.notify()
        return strategies.get_backtest(backtest["id"])

    @app.post("/api/strategy-versions/{version_id}/approve")
    def approve_strategy(
        version_id: str, payload: StrategyApprovalRequest, request: Request
    ) -> dict:
        try:
            return strategies.approve(
                version_id,
                actor=authenticated_actor(request, payload.actor),
                reason=payload.reason,
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy version not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/strategy-allocations")
    def list_strategy_allocations(limit: int = Query(100, ge=1, le=500)) -> list[dict]:
        items = allocations.list(limit)
        for item in items:
            item["automation"] = schedules.get_allocation_group_optional(str(item["id"]))
        return items

    @app.post("/api/strategy-allocations", status_code=201)
    def create_strategy_allocation(
        payload: StrategyAllocationCreateRequest,
        request: Request,
    ) -> dict:
        require_qlib_dataset(
            payload.dataset, purpose="strategy allocation", frequency="day"
        )
        fixed_weights = (
            {
                item.strategy_version_id: float(item.weight)
                for item in payload.members
                if item.weight is not None
            }
            if payload.allocation_method == "fixed"
            else None
        )
        try:
            return allocations.create(
                name=payload.name,
                strategy_version_ids=[item.strategy_version_id for item in payload.members],
                dataset=payload.dataset,
                total_capital=payload.total_capital,
                allocation_method=payload.allocation_method,
                lookback_days=payload.lookback_days,
                target_volatility=payload.target_volatility,
                max_pairwise_correlation=payload.max_pairwise_correlation,
                max_strategy_weight=payload.max_strategy_weight,
                max_member_drawdown=payload.max_member_drawdown,
                max_drawdown_reduce=payload.max_drawdown_reduce,
                max_drawdown_liquidate=payload.max_drawdown_liquidate,
                fixed_weights=fixed_weights,
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy version not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/strategy-allocations/{allocation_id}")
    def get_strategy_allocation(allocation_id: str) -> dict:
        try:
            result = allocations.get(allocation_id)
            result["automation"] = schedules.get_allocation_group_optional(allocation_id)
            return result
        except KeyError as exc:
            raise HTTPException(404, "strategy allocation not found") from exc

    @app.post("/api/strategy-allocations/{allocation_id}/schedule")
    def configure_strategy_allocation_schedule(
        allocation_id: str,
        payload: AllocationScheduleRequest,
        request: Request,
    ) -> dict:
        try:
            return schedules.create_allocation_group(
                allocation_id,
                timezone=payload.timezone,
                run_time=payload.run_time,
                trading_days_only=payload.trading_days_only,
                slippage=payload.slippage,
                misfire_grace_seconds=payload.misfire_grace_seconds,
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy allocation not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/strategy-allocations/{allocation_id}/schedule/status")
    def set_strategy_allocation_schedule_status(
        allocation_id: str,
        payload: AllocationScheduleStatusRequest,
        request: Request,
    ) -> dict:
        authenticated_actor(request, payload.actor)
        try:
            return schedules.set_allocation_group_status(allocation_id, payload.status)
        except KeyError as exc:
            raise HTTPException(404, "strategy allocation schedule not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.delete("/api/strategy-allocations/{allocation_id}/schedule")
    def retire_strategy_allocation_schedule(
        allocation_id: str,
        payload: AllocationScheduleRetireRequest,
        request: Request,
    ) -> dict:
        authenticated_actor(request, payload.actor)
        try:
            return schedules.set_allocation_group_status(allocation_id, "retired")
        except KeyError as exc:
            raise HTTPException(404, "strategy allocation schedule not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/strategy-allocations/{allocation_id}/approve")
    def approve_strategy_allocation(
        allocation_id: str,
        payload: StrategyAllocationApprovalRequest,
        request: Request,
    ) -> dict:
        try:
            return allocations.approve(
                allocation_id,
                actor=authenticated_actor(request, payload.actor),
                reason=payload.reason,
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy allocation not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/strategy-allocations/{allocation_id}/refresh")
    def refresh_strategy_allocation(allocation_id: str) -> dict:
        try:
            return allocations.refresh(allocation_id)
        except KeyError as exc:
            raise HTTPException(404, "strategy allocation not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/strategy-allocations/{allocation_id}/status")
    def set_strategy_allocation_status(
        allocation_id: str,
        payload: StrategyAllocationStatusRequest,
        request: Request,
    ) -> dict:
        try:
            return allocations.set_status(
                allocation_id,
                payload.status,
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy allocation not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/strategy-allocations/{allocation_id}/events/{event_id}/acknowledge")
    def acknowledge_strategy_allocation_event(
        allocation_id: str,
        event_id: int,
        payload: RiskEventAcknowledgementRequest,
        request: Request,
    ) -> dict:
        try:
            return allocations.acknowledge_event(
                allocation_id,
                event_id,
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy allocation risk event not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/strategy-allocations/{allocation_id}/events/{event_id}/resolve")
    def resolve_strategy_allocation_event(
        allocation_id: str,
        event_id: int,
        payload: RiskEventResolutionRequest,
        request: Request,
    ) -> dict:
        try:
            return allocations.resolve_event(
                allocation_id,
                event_id,
                actor=authenticated_actor(request, payload.actor),
                reason=payload.reason,
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy allocation risk event not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/pair-portfolios")
    def list_pair_portfolios(limit: int = Query(100, ge=1, le=500)) -> list[dict]:
        return pair_portfolios.list(limit)

    @app.post("/api/pair-portfolios", status_code=201)
    def create_pair_portfolio(payload: PairPortfolioCreateRequest, request: Request) -> dict:
        dataset = require_qlib_dataset(
            payload.dataset, purpose="pair paper portfolio", frequency="day"
        )
        try:
            minute = resolve_snapshot_dataset(
                settings.data_root,
                snapshot_name=payload.execution_snapshot,
                dataset_name=payload.minute_dataset,
            )
            shortability = resolve_snapshot_dataset(
                settings.data_root,
                snapshot_name=payload.execution_snapshot,
                dataset_name=payload.shortability_dataset,
            )
            if minute.get("snapshot_lineage_id") != shortability.get("snapshot_lineage_id"):
                raise ValueError("minute and shortability datasets have different lineages")
            return pair_portfolios.create(
                name=payload.name,
                strategy_version_id=payload.strategy_version_id,
                dataset=payload.dataset,
                execution_snapshot=payload.execution_snapshot,
                minute_dataset=payload.minute_dataset,
                shortability_dataset=payload.shortability_dataset,
                initial_cash=payload.initial_cash,
                actor=authenticated_actor(request, payload.actor),
                dataset_roll_policy=payload.dataset_roll_policy,
                dataset_lineage_id=dataset.get("lineage_id"),
                execution_roll_policy=payload.execution_roll_policy,
                execution_lineage_id=minute.get("snapshot_lineage_id"),
            )
        except KeyError as exc:
            raise HTTPException(404, "pair strategy version not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/pair-portfolios/{portfolio_id}")
    def get_pair_portfolio(portfolio_id: str) -> dict:
        try:
            return pair_portfolios.get(portfolio_id)
        except KeyError as exc:
            raise HTTPException(404, "pair portfolio not found") from exc

    @app.post("/api/pair-portfolios/{portfolio_id}/status")
    def set_pair_portfolio_status(portfolio_id: str, payload: PortfolioStatusRequest) -> dict:
        try:
            return pair_portfolios.set_status(portfolio_id, payload.status)
        except KeyError as exc:
            raise HTTPException(404, "pair portfolio not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/pair-portfolios/{portfolio_id}/risk-events/{event_id}/acknowledge")
    def acknowledge_pair_portfolio_risk_event(
        portfolio_id: str,
        event_id: int,
        payload: RiskEventAcknowledgementRequest,
        request: Request,
    ) -> dict:
        try:
            return pair_portfolios.acknowledge_risk_event(
                portfolio_id,
                event_id,
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "pair portfolio risk event not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/pair-portfolios/{portfolio_id}/risk-events/{event_id}/resolve")
    def resolve_pair_portfolio_risk_event(
        portfolio_id: str,
        event_id: int,
        payload: RiskEventResolutionRequest,
        request: Request,
    ) -> dict:
        try:
            return pair_portfolios.resolve_risk_event(
                portfolio_id,
                event_id,
                actor=authenticated_actor(request, payload.actor),
                reason=payload.reason,
            )
        except KeyError as exc:
            raise HTTPException(404, "pair portfolio risk event not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/pair-portfolios/{portfolio_id}/rebalance", status_code=202)
    def rebalance_pair_portfolio(portfolio_id: str, payload: PortfolioRebalanceRequest) -> dict:
        try:
            portfolio = pair_portfolios.get(portfolio_id)
        except KeyError as exc:
            raise HTTPException(404, "pair portfolio not found") from exc
        try:
            dataset = select_qlib_dataset(
                settings.data_root,
                anchor_name=portfolio["dataset"],
                roll_policy=portfolio["dataset_roll_policy"],
                lineage_id=portfolio.get("dataset_lineage_id"),
                required_date=payload.as_of_date,
                require_later_date=True,
            )
            trade_date = next_qlib_trading_date(dataset, payload.as_of_date)
            execution = select_execution_snapshot(
                settings.data_root,
                anchor_name=portfolio["execution_snapshot"],
                roll_policy=portfolio["execution_roll_policy"],
                lineage_id=portfolio.get("execution_lineage_id"),
                required_date=trade_date,
                minute_dataset=portfolio["minute_dataset"],
                shortability_dataset=portfolio["shortability_dataset"],
            )
            artifact_root = (
                settings.data_root / "artifacts" / "pair-paper-portfolios" / portfolio_id
            )
            batch, created = pair_portfolios.create_batch(
                portfolio_id=portfolio_id,
                as_of_date=payload.as_of_date,
                artifact_path=artifact_root,
                dataset_evidence=dataset,
                execution_evidence=execution,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if not created:
            return batch
        log_path = platform_root / "logs" / f"pair-paper-rebalance-{batch['id']}.log"
        try:
            job = jobs.create(
                "pair_paper_rebalance",
                {
                    "pair_portfolio_id": portfolio_id,
                    "pair_portfolio_batch_id": batch["id"],
                    "dataset": dataset["name"],
                    "dataset_path": dataset["path"],
                    "dataset_start": dataset["start_date"],
                    "daily_provenance": dataset["provenance"],
                    "as_of_date": payload.as_of_date.isoformat(),
                    "minute_dataset": execution["minute"],
                    "shortability_dataset": execution["shortability"],
                },
                log_path,
            )
        except ValueError as exc:
            pair_portfolios.mark_batch(batch["id"], "failed", error=str(exc))
            raise HTTPException(409, str(exc)) from exc
        pair_portfolios.attach_job(batch["id"], job["id"])
        worker.notify()
        return pair_portfolios.get_batch(batch["id"])

    @app.get("/api/portfolios")
    def list_portfolios(limit: int = Query(100, ge=1, le=500)) -> list[dict]:
        return portfolios.list(limit)

    @app.post("/api/portfolios", status_code=201)
    def create_portfolio(payload: PortfolioCreateRequest, request: Request) -> dict:
        dataset = require_qlib_dataset(
            payload.dataset, purpose="paper portfolio", frequency="day"
        )
        try:
            return portfolios.create(
                name=payload.name,
                strategy_version_id=payload.strategy_version_id,
                dataset=payload.dataset,
                initial_cash=payload.initial_cash,
                actor=authenticated_actor(request, payload.actor),
                dataset_roll_policy=payload.dataset_roll_policy,
                dataset_lineage_id=dataset.get("lineage_id"),
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy version not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/portfolios/{portfolio_id}")
    def get_portfolio(portfolio_id: str) -> dict:
        try:
            return portfolios.get(portfolio_id)
        except KeyError as exc:
            raise HTTPException(404, "portfolio not found") from exc

    @app.post("/api/portfolios/{portfolio_id}/status")
    def set_portfolio_status(portfolio_id: str, payload: PortfolioStatusRequest) -> dict:
        try:
            return portfolios.set_status(portfolio_id, payload.status)
        except KeyError as exc:
            raise HTTPException(404, "portfolio not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/portfolios/{portfolio_id}/risk-events/{event_id}/acknowledge")
    def acknowledge_portfolio_risk_event(
        portfolio_id: str,
        event_id: int,
        payload: RiskEventAcknowledgementRequest,
        request: Request,
    ) -> dict:
        try:
            return portfolios.acknowledge_risk_event(
                portfolio_id,
                event_id,
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "portfolio risk event not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/portfolios/{portfolio_id}/risk-events/{event_id}/resolve")
    def resolve_portfolio_risk_event(
        portfolio_id: str,
        event_id: int,
        payload: RiskEventResolutionRequest,
        request: Request,
    ) -> dict:
        try:
            return portfolios.resolve_risk_event(
                portfolio_id,
                event_id,
                actor=authenticated_actor(request, payload.actor),
                reason=payload.reason,
            )
        except KeyError as exc:
            raise HTTPException(404, "portfolio risk event not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/portfolios/{portfolio_id}/rebalance", status_code=202)
    def rebalance_portfolio(portfolio_id: str, payload: PortfolioRebalanceRequest) -> dict:
        try:
            portfolio = portfolios.get(portfolio_id)
        except KeyError as exc:
            raise HTTPException(404, "portfolio not found") from exc
        artifact_root = settings.data_root / "artifacts" / "paper-portfolios" / portfolio_id
        try:
            dataset = select_qlib_dataset(
                settings.data_root,
                anchor_name=portfolio["dataset"],
                roll_policy=portfolio["dataset_roll_policy"],
                lineage_id=portfolio.get("dataset_lineage_id"),
                required_date=payload.as_of_date,
            )
            batch, created = portfolios.create_batch(
                portfolio_id=portfolio_id,
                as_of_date=payload.as_of_date,
                artifact_path=artifact_root,
                dataset_evidence=dataset,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if not created:
            return batch
        log_path = platform_root / "logs" / f"paper-rebalance-{batch['id']}.log"
        try:
            job = jobs.create(
                "paper_rebalance",
                {
                    "portfolio_id": portfolio_id,
                    "portfolio_batch_id": batch["id"],
                    "dataset": dataset["name"],
                    "dataset_path": dataset["path"],
                    "daily_provenance": dataset["provenance"],
                    "as_of_date": payload.as_of_date.isoformat(),
                    "slippage": payload.slippage,
                },
                log_path,
            )
        except ValueError as exc:
            portfolios.mark_batch(batch["id"], "failed", error=str(exc))
            raise HTTPException(409, str(exc)) from exc
        portfolios.attach_job(batch["id"], job["id"])
        worker.notify()
        return portfolios.get_batch(batch["id"])

    @app.get("/api/schedules")
    def list_schedules(limit: int = Query(200, ge=1, le=500)) -> list[dict]:
        return schedules.list(limit)

    @app.post("/api/schedules", status_code=201)
    def create_schedule(payload: ScheduleCreateRequest, request: Request) -> dict:
        schedule_actor = authenticated_actor(request, payload.actor)
        schedule_payload = dict(payload.payload)
        if payload.kind == "rdagent_research":
            runtime = probe_rdagent(settings, project_root)
            if not runtime.get("ready"):
                blockers = runtime.get("blockers") or [
                    runtime.get("error") or "runtime unavailable"
                ]
                raise HTTPException(
                    409,
                    {"message": "RD-Agent is not ready", "blockers": blockers},
                )
            try:
                research_payload = normalize_research_schedule_payload(
                    schedule_payload, max_loops=settings.rdagent_max_loops
                )
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            research_payload["requested_by"] = schedule_actor
            schedule_payload = research_payload
            dataset = require_qlib_dataset(
                research_payload["dataset"],
                purpose="scheduled RD-Agent research",
                frequency="day",
            )
            periods = research_payload["periods"]
            if dataset.get("start_date") and periods["train_start"] < dataset["start_date"]:
                raise HTTPException(409, "training window starts before the selected dataset")
            if dataset.get("end_date") and periods["test_end"] > dataset["end_date"]:
                raise HTTPException(409, "test window ends after the selected dataset")
        elif payload.kind == "paper_rebalance":
            try:
                portfolio = portfolios.get(str(schedule_payload["portfolio_id"]))
            except KeyError as exc:
                raise HTTPException(404, "portfolio not found") from exc
            if portfolio["status"] == "closed":
                raise HTTPException(409, "closed portfolios cannot be scheduled")
        elif payload.kind == "pair_paper_rebalance":
            try:
                portfolio = pair_portfolios.get(str(schedule_payload["pair_portfolio_id"]))
            except KeyError as exc:
                raise HTTPException(404, "pair portfolio not found") from exc
            if portfolio["status"] == "closed":
                raise HTTPException(409, "closed pair portfolios cannot be scheduled")
        elif payload.kind == "broker_reconcile":
            try:
                destination = brokers.get_destination(str(schedule_payload["destination_id"]))
            except KeyError as exc:
                raise HTTPException(404, "broker destination not found") from exc
            if destination["status"] != "armed":
                raise HTTPException(409, "broker destination must be armed before scheduling")
        try:
            return schedules.create(
                name=payload.name,
                kind=payload.kind,
                timezone=payload.timezone,
                run_time=payload.run_time,
                trading_days_only=payload.trading_days_only,
                payload=schedule_payload,
                misfire_grace_seconds=payload.misfire_grace_seconds,
                actor=schedule_actor,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/schedules/{schedule_id}/status")
    def set_schedule_status(schedule_id: str, payload: ScheduleStatusRequest) -> dict:
        try:
            return schedules.set_status(schedule_id, payload.status)
        except KeyError as exc:
            raise HTTPException(404, "schedule not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/schedule-runs")
    def list_schedule_runs(limit: int = Query(200, ge=1, le=500)) -> list[dict]:
        return schedules.list_runs(limit)

    @app.get("/api/alerts")
    def list_alerts(
        status: Literal["open", "acknowledged", "resolved"] | None = None,
        limit: int = Query(200, ge=1, le=500),
    ) -> list[dict]:
        return alerts.list(status=status, limit=limit)

    @app.post("/api/alerts/{alert_id}/acknowledge")
    def acknowledge_alert(alert_id: str, payload: AlertActionRequest, request: Request) -> dict:
        try:
            return alerts.acknowledge(alert_id, actor=authenticated_actor(request, payload.actor))
        except KeyError as exc:
            raise HTTPException(404, "alert not found") from exc

    @app.post("/api/alerts/{alert_id}/resolve")
    def resolve_alert(alert_id: str, payload: AlertActionRequest, request: Request) -> dict:
        try:
            return alerts.resolve(alert_id, actor=authenticated_actor(request, payload.actor))
        except KeyError as exc:
            raise HTTPException(404, "alert not found") from exc

    @app.get("/api/scheduler/status")
    def scheduler_status() -> dict:
        if not settings.scheduler_url:
            return {"status": "embedded_or_unconfigured"}
        try:
            response = requests.get(f"{settings.scheduler_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            return {"status": "unavailable", "error": str(exc)}

    @app.get("/api/operations/health")
    def operational_health(limit: int = Query(48, ge=1, le=500)) -> dict:
        return {
            "latest": health_history.latest(),
            "history": health_history.list(limit),
        }

    @app.get("/api/operations/readiness")
    def operational_readiness() -> dict:
        return deployment_readiness.assess()

    @app.get("/api/broker")
    def broker_status(probe: bool = False) -> dict:
        return {
            "readiness": brokers.readiness(probe=probe),
            "destinations": brokers.list_destinations(),
            "outbox": brokers.list_outbox(limit=200),
            "reconciliations": brokers.list_reconciliations(limit=200),
        }

    @app.get("/api/broker/destinations/{destination_id}/events")
    def list_broker_events(
        destination_id: str, limit: int = Query(200, ge=1, le=500)
    ) -> list[dict]:
        return brokers.list_events(destination_id, limit)

    @app.post("/api/broker/destinations", status_code=201)
    def create_broker_destination(
        payload: BrokerDestinationCreateRequest, request: Request
    ) -> dict:
        try:
            return brokers.create_destination(
                name=payload.name,
                account_ref=payload.account_ref,
                portfolio_id=payload.portfolio_id,
                config=payload.config,
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "paper portfolio not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/broker/destinations/{destination_id}/request-activation")
    def request_broker_activation(
        destination_id: str, payload: BrokerActorRequest, request: Request
    ) -> dict:
        try:
            return brokers.request_activation(
                destination_id, actor=authenticated_actor(request, payload.actor)
            )
        except KeyError as exc:
            raise HTTPException(404, "broker destination not found") from exc
        except (ValueError, BrokerGatewayError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/broker/destinations/{destination_id}/approve-activation")
    def approve_broker_activation(
        destination_id: str, payload: BrokerActorRequest, request: Request
    ) -> dict:
        try:
            return brokers.approve_activation(
                destination_id, actor=authenticated_actor(request, payload.actor)
            )
        except KeyError as exc:
            raise HTTPException(404, "broker destination not found") from exc
        except (ValueError, BrokerGatewayError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/broker/destinations/{destination_id}/disarm")
    def disarm_broker_destination(
        destination_id: str, payload: BrokerActorRequest, request: Request
    ) -> dict:
        try:
            return brokers.disarm(destination_id, actor=authenticated_actor(request, payload.actor))
        except KeyError as exc:
            raise HTTPException(404, "broker destination not found") from exc

    @app.post("/api/broker/destinations/{destination_id}/stage")
    def stage_broker_batch(
        destination_id: str, payload: BrokerBatchRequest, request: Request
    ) -> list[dict]:
        try:
            return brokers.stage_batch(
                destination_id,
                payload.batch_id,
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "broker destination or batch not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/broker/destinations/{destination_id}/approve")
    def approve_broker_batch(
        destination_id: str, payload: BrokerBatchRequest, request: Request
    ) -> list[dict]:
        try:
            return brokers.approve_batch(
                destination_id,
                payload.batch_id,
                actor=authenticated_actor(request, payload.actor),
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/broker/destinations/{destination_id}/dispatch")
    def dispatch_broker_batch(
        destination_id: str, payload: BrokerBatchRequest, request: Request
    ) -> dict:
        try:
            return brokers.dispatch_batch(
                destination_id,
                payload.batch_id,
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "broker destination not found") from exc
        except (ValueError, BrokerGatewayError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/broker/destinations/{destination_id}/reconcile")
    def reconcile_broker_destination(
        destination_id: str, payload: BrokerActorRequest, request: Request
    ) -> dict:
        try:
            return brokers.reconcile(
                destination_id, actor=authenticated_actor(request, payload.actor)
            )
        except KeyError as exc:
            raise HTTPException(404, "broker destination or portfolio not found") from exc
        except (ValueError, BrokerGatewayError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/jobs")
    def list_jobs(
        response: Response,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0, le=100_000),
        status: Annotated[list[str] | None, Query()] = None,
        kind: Annotated[list[str] | None, Query()] = None,
    ) -> list[dict]:
        allowed_statuses = {"queued", "running", "succeeded", "failed", "cancelled"}
        unknown = set(status or []) - allowed_statuses
        if unknown:
            raise HTTPException(422, f"unsupported job status: {sorted(unknown)}")
        statuses = tuple(status or [])
        kinds = tuple(item for item in (kind or []) if item.strip())
        response.headers["X-Total-Count"] = str(
            jobs.count(statuses=statuses, kinds=kinds)
        )
        response.headers["Access-Control-Expose-Headers"] = "X-Total-Count"
        return jobs.list(limit, offset=offset, statuses=statuses, kinds=kinds)

    @app.get("/api/data-tasks")
    def list_data_tasks() -> list[dict]:
        return data_tasks.list()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "job not found") from exc

    @app.get("/api/jobs/{job_id}/log")
    def get_job_log(job_id: str, tail: int = Query(200, ge=1, le=2000)) -> dict:
        try:
            job = jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "job not found") from exc
        path = Path(job["log_path"])
        if not path.exists():
            return {"job_id": job_id, "lines": []}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"job_id": job_id, "lines": lines[-tail:]}

    @app.post("/api/jobs/bootstrap", status_code=202)
    def create_bootstrap(payload: BootstrapRequest) -> dict:
        api_url, token = tushare_settings()
        missing = [
            name
            for name, value in (
                ("TUSHARE_API_URL", api_url),
                ("TUSHARE_TOKEN", token),
            )
            if not value
        ]
        if missing:
            raise HTTPException(409, f"missing deployment secret: {', '.join(missing)}")
        requested_end = payload.end if isinstance(payload.end, date) else date.today()
        snapshot_name = (
            f"cn-{payload.start:%Y%m%d}-{requested_end:%Y%m%d}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        )
        serialized = {
            "profile": payload.profile,
            "start": payload.start.isoformat(),
            "end": payload.end.isoformat() if isinstance(payload.end, date) else payload.end,
            "build_qlib": False,
            "finalize_after_download": payload.build_qlib,
            "pipeline_id": uuid.uuid4().hex,
            "snapshot_start": payload.start.isoformat(),
            "snapshot_end": requested_end.isoformat(),
            "snapshot_name": snapshot_name,
        }
        log_path = platform_root / "logs" / f"bootstrap-{payload.profile}-{date.today():%Y%m%d}.log"
        try:
            job = jobs.create("bootstrap", serialized, log_path)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/finalize-data", status_code=202)
    def finalize_data_pipeline(payload: DataFinalizeRequest) -> dict:
        counts = checkpoint.counts()
        if not counts:
            raise HTTPException(409, "no downloaded work units are available to finalize")
        incomplete = {
            str(row["status"]): sum(
                int(item["units"]) for item in counts if str(item["status"]) == str(row["status"])
            )
            for row in counts
            if str(row["status"]) != "succeeded"
        }
        if incomplete:
            detail = ", ".join(f"{key}={value}" for key, value in sorted(incomplete.items()))
            raise HTTPException(409, f"download work units are not complete: {detail}")
        end_date = payload.end if isinstance(payload.end, date) else date.today()
        snapshot_name = payload.snapshot_name or (
            f"cn-{payload.start:%Y%m%d}-{end_date:%Y%m%d}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        )
        pipeline_id = uuid.uuid4().hex
        serialized = {
            "pipeline_id": pipeline_id,
            "profile": payload.profile,
            "start": payload.start.isoformat(),
            "end": end_date.isoformat(),
            "snapshot_name": snapshot_name,
        }
        log_path = platform_root / "logs" / f"data-verify-{snapshot_name}.log"
        try:
            job = jobs.create(
                "data_verify",
                serialized,
                log_path,
                idempotency_key=f"data-finalize:{snapshot_name}:verify",
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/margin-eligibility", status_code=202)
    def download_margin_eligibility(payload: MarginEligibilityRequest) -> dict:
        api_url, token = tushare_settings()
        if not api_url or not token:
            raise HTTPException(409, "Tushare credentials are not configured")
        end_date = payload.end if isinstance(payload.end, date) else date.today()
        serialized = {
            "start": payload.start.isoformat(),
            "end": end_date.isoformat(),
        }
        log_path = (
            platform_root
            / "logs"
            / f"margin-eligibility-{payload.start:%Y%m%d}-{end_date:%Y%m%d}.log"
        )
        try:
            job = jobs.create(
                "margin_eligibility_download",
                serialized,
                log_path,
                idempotency_key=f"margin-eligibility:{payload.start}:{end_date}",
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/core-intraday", status_code=202)
    def download_core_intraday(payload: CoreIntradayRequest) -> dict:
        api_url, token = tushare_settings()
        if not api_url or not token:
            raise HTTPException(409, "Tushare credentials are not configured")
        end_date = payload.end if isinstance(payload.end, date) else date.today()
        snapshot_name = payload.snapshot_name or (
            f"execution-{payload.start:%Y%m%d}-{end_date:%Y%m%d}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        )
        serialized = {
            "start": payload.start.isoformat(),
            "end": end_date.isoformat(),
            "etfs": payload.etfs,
            "stocks": payload.stocks,
            "indices": payload.indices,
            "futures": payload.futures,
            "options": payload.options,
            "auto_select": payload.auto_select,
            "max_stocks": payload.max_stocks,
            "max_options": payload.max_options,
            "etf_categories": payload.etf_categories,
            "snapshot_name": snapshot_name,
        }
        log_path = platform_root / "logs" / f"core-intraday-{snapshot_name}.log"
        try:
            job = jobs.create(
                "core_intraday_download",
                serialized,
                log_path,
                idempotency_key=f"core-intraday:{snapshot_name}",
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/ashare-5m", status_code=202)
    def download_ashare_5m(payload: Ashare5mRequest) -> dict:
        api_url, token = tushare_settings()
        if not api_url or not token:
            raise HTTPException(409, "Tushare credentials are not configured")
        end_date = payload.end if isinstance(payload.end, date) else date.today()
        snapshot_name = payload.snapshot_name or (
            f"ashare-5m-{payload.start:%Y%m%d}-{end_date:%Y%m%d}"
        )
        serialized = {
            "start": payload.start.isoformat(),
            "end": end_date.isoformat(),
            "snapshot_name": snapshot_name,
        }
        log_path = platform_root / "logs" / f"ashare-5m-{snapshot_name}.log"
        try:
            job = jobs.create(
                "ashare_5m_download",
                serialized,
                log_path,
                idempotency_key=f"ashare-5m:{snapshot_name}",
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/supplemental-download", status_code=202)
    def download_supplemental(payload: SupplementalDownloadRequest) -> dict:
        api_url, token = tushare_settings()
        if not api_url or not token:
            raise HTTPException(409, "Tushare credentials are not configured")
        end_date = payload.end if isinstance(payload.end, date) else date.today()
        serialized = {
            "bundle": payload.bundle,
            "start": payload.start.isoformat(),
            "end": end_date.isoformat(),
            "symbols": payload.symbols,
        }
        log_path = (
            platform_root
            / "logs"
            / f"supplemental-{payload.bundle}-{payload.start:%Y%m%d}-{end_date:%Y%m%d}.log"
        )
        try:
            job = jobs.create(
                f"supplemental_{payload.bundle}",
                serialized,
                log_path,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/{job_id}/retry", status_code=202)
    def retry_job(job_id: str) -> dict:
        try:
            job = jobs.retry(job_id)
        except KeyError as exc:
            raise HTTPException(404, "job not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/{job_id}/cancel", status_code=202)
    def cancel_job(job_id: str) -> dict:
        try:
            job = jobs.request_cancel(job_id)
        except KeyError as exc:
            raise HTTPException(404, "job not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/qlib-baseline", status_code=202)
    def create_qlib_baseline(payload: QlibBaselineRequest) -> dict:
        dataset = require_qlib_dataset(
            payload.dataset, purpose="Qlib baseline", frequency="day"
        )
        serialized = {
            "dataset": payload.dataset,
            "dataset_path": dataset["path"],
            "market": payload.market,
            "benchmark": payload.benchmark,
            "account": payload.account,
            "topk": payload.topk,
            "n_drop": payload.n_drop,
            "open_cost": payload.open_cost,
            "close_cost": payload.close_cost,
            "min_cost": payload.min_cost,
        }
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        log_path = platform_root / "logs" / f"qlib-baseline-{stamp}.log"
        try:
            job = jobs.create("qlib_baseline", serialized, log_path)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/minute-qlib", status_code=202)
    def create_minute_qlib(payload: MinuteQlibRequest) -> dict:
        try:
            snapshot = resolve_snapshot_manifest(settings.data_root, payload.snapshot_name)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        manifest = snapshot["manifest"]
        frequency = str(manifest.get("frequency") or "")
        if frequency not in MINUTE_FREQUENCIES:
            raise HTTPException(409, "minute Qlib requires a supported minute snapshot")
        supported = set(MINUTE_DATASETS)
        if not supported.intersection(manifest.get("datasets", {})):
            raise HTTPException(409, "execution snapshot has no supported minute datasets")
        output_name = payload.output_name or f"{payload.snapshot_name}-{frequency}"
        serialized = {
            "snapshot_name": payload.snapshot_name,
            "snapshot_manifest_sha256": snapshot["manifest_sha256"],
            "output_name": output_name,
            "frequency": frequency,
        }
        log_path = platform_root / "logs" / f"minute-qlib-{output_name}.log"
        try:
            job = jobs.create(
                "minute_qlib",
                serialized,
                log_path,
                idempotency_key=f"minute-qlib:{payload.snapshot_name}:{output_name}",
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/minute-research", status_code=202)
    def create_minute_research(payload: MinuteResearchRequest) -> dict:
        dataset = require_qlib_dataset(
            payload.dataset, purpose="minute factor research", frequency="1min"
        )
        serialized = {
            "dataset": payload.dataset,
            "dataset_path": dataset["path"],
            "dataset_identity_sha256": dataset["provenance"][
                "dataset_identity_sha256"
            ],
            "start": payload.start.isoformat(),
            "end": payload.end.isoformat(),
            "horizons": sorted(payload.horizons),
            "cost_rate": payload.cost_rate,
        }
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        log_path = platform_root / "logs" / f"minute-research-{stamp}.log"
        try:
            job = jobs.create("minute_research", serialized, log_path)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    return app


app = create_app()
