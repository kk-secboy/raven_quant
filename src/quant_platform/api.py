from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from threading import Lock, Thread
from time import monotonic
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import text

from quant_data.checkpoint import CheckpointStore
from quant_data.config import Settings, normalize_api_url
from quant_data.coverage_data import DEFAULT_COVERAGE_BUNDLES
from quant_data.execution_contract import (
    require_daily_qlib_contract,
    require_minute_execution_contract,
    require_minute_signal_contract,
    require_native_daily_execution_controls,
    require_strategy_execution_contract,
    strategy_execution_contract_hash,
)
from quant_data.execution_data import (
    MINUTE_DATASETS,
    MINUTE_FREQUENCIES,
    NATIVE_MINUTE_FREQUENCIES,
)
from quant_data.legacy_market import (
    BAOSTOCK_OVERLAP_POLICY_VERSION,
    DEFAULT_OVERLAP_SYMBOLS,
    PRIMARY_OVERLAP_PROVIDER,
    require_audited_overlap_symbols,
    require_current_primary_overlap_evidence,
)
from quant_platform.announcement_nlp import (
    ANNOUNCEMENTS_DIR,
    LOGIC_FACTOR_NAME,
    NLP_SUBDIR,
)
from quant_platform.announcement_nlp import (
    DEFAULT_BATCH_SIZE as ANNOUNCEMENT_DEFAULT_BATCH_SIZE,
)
from quant_platform.announcement_nlp import DEFAULT_WORKERS as ANNOUNCEMENT_DEFAULT_WORKERS
from quant_platform.announcement_nlp import MAX_BATCH_SIZE as ANNOUNCEMENT_MAX_BATCH_SIZE
from quant_platform.announcement_nlp import MAX_WORKERS as ANNOUNCEMENT_MAX_WORKERS
from quant_platform.announcement_nlp import (
    PROMPT_VERSION as ANNOUNCEMENT_PROMPT_VERSION,
)
from quant_platform.corpus_nlp import DEFAULT_BATCH_SIZE as CORPUS_DEFAULT_BATCH_SIZE
from quant_platform.corpus_nlp import (
    DEFAULT_IRM_PER_INSTRUMENT_DAY as CORPUS_DEFAULT_IRM_PER_INSTRUMENT_DAY,
)
from quant_platform.corpus_nlp import (
    DEFAULT_MAJOR_NEWS_PER_DAY as CORPUS_DEFAULT_MAJOR_NEWS_PER_DAY,
)
from quant_platform.corpus_nlp import DEFAULT_WORKERS as CORPUS_DEFAULT_WORKERS
from quant_platform.corpus_nlp import MAX_WORKERS as CORPUS_MAX_WORKERS
from quant_platform.corpus_nlp import PROMPT_VERSION as CORPUS_PROMPT_VERSION
from quant_platform.event_market_response import LABEL_SCHEMA_VERSION
from quant_platform.qlib_factor_baseline import FACTOR_SOURCE_QLIB_BASELINE

from .advice_service import AdviceService
from .alert_store import AlertStore
from .allocation_store import AllocationStore
from .auth_policy import ROLE_PERMISSIONS, has_permission, permission_for
from .auth_store import AuthenticationError, AuthStore
from .autopilot import (
    AUTOPILOT_CONFIG_KEY,
    DEFAULT_AUTOPILOT_CONFIG,
    AutopilotController,
    normalize_autopilot_config,
)
from .autopilot_trial_audit import AutopilotTrialAuditService
from .cost_model import (
    COST_SCHEDULE_VERSION,
    CURRENT_STOCK_SELL_STAMP_DUTY_RATE,
    CURRENT_TRANSFER_FEE_RATE,
    CostModelConfig,
)
from .data_automation import (
    DATA_AUTOMATION_CONFIG_KEY,
    DEFAULT_DATA_AUTOMATION_CONFIG,
    MANAGED_SCHEDULE_NAMES,
    automation_coverage,
    normalize_data_automation_config,
)
from .data_rollover import qlib_trading_date_on_or_before, select_qlib_dataset
from .data_task_store import DataTaskStore
from .deployment_readiness import DeploymentReadinessStore
from .factor_library_store import FactorLibraryStore
from .feature_set_registry import get_feature_set, list_feature_sets, register_feature_set
from .health_store import OperationalHealthStore, safe_mode_recovery_health_status
from .information_schedule import (
    STRUCTURED_INFORMATION_SOURCES,
    latest_verified_research_asset_snapshot,
    normalize_information_factor_refresh_payload,
    normalize_information_schedule_payload,
)
from .investor_profile import InvestorSimulationProfileStore
from .job_store import (
    EVALUATION_STATUS_COUNTS_KEY,
    JobStore,
    research_asset_acquisition_idempotency_key,
)
from .market_overview import MarketOverviewService
from .model_artifact_store import ModelArtifactStore
from .model_research_governance import canonical_sha256
from .ops_calendar import evaluate_recommendation_gate, load_calendar_days
from .parameter_experiment_store import ParameterExperimentStore
from .parameter_experiments import normalize_parameter_grid, split_research_period
from .platform_config_store import PlatformConfigStore
from .promotion import PromotionStore
from .rdagent_candidate_store import RDAGentCandidateStore
from .rdagent_runtime import (
    expected_rdagent_runtime_identity,
    probe_rdagent,
    run_official_rdagent_health_check,
    validate_duration,
    validate_duration_limit,
)
from .rdagent_scenarios import (
    get_rdagent_scenario,
    require_ready_scenario,
    resolve_rdagent_assets,
    scenario_from_research_run,
    validate_asset_id,
    validate_feature_set_id,
)
from .recommendation_account_store import RecommendationAccountStore
from .recommendation_store import (
    RecommendationStore,
    recommendation_refresh_job_idempotency_key,
    recommendation_refresh_job_payload,
)
from .research_asset_store import ResearchAssetStore
from .research_automation import (
    HORIZON_RESEARCH_SCENARIOS,
    normalize_research_period_policy,
    normalize_research_schedule_payload,
    resolve_research_periods,
    resolve_research_window_contract,
)
from .research_campaign_store import ResearchCampaignStore
from .research_horizon import (
    LEGACY_AMBIGUOUS,
    LONG_1_3Y,
    SHORT_1_5D,
    SWING_1_6M,
    primary_label_horizon_sessions,
    primary_label_policy_contract,
    research_horizon_contract,
)
from .research_program_store import ResearchProgramStore
from .research_report_backfill import ResearchReportBackfillStore
from .research_store import ResearchStore
from .research_tournament import ResearchTournamentStore
from .retention import DataRetentionManager
from .runtime_secret_store import RuntimeSecretStore
from .safe_mode import SafeModeStore
from .schedule_store import ScheduleStore, validate_intraday_run_time
from .scheduler import AUTOMATED_DATA_BUNDLES
from .services import (
    CheckpointCatalogProjection,
    list_qlib_datasets,
    list_qlib_datasets_for_display,
    list_qlib_experiments,
    list_snapshots_for_display,
    probe_qlib,
    resolve_snapshot_manifest,
    system_summary,
)
from .simulation_store import SimulationStore
from .strategy_feature_drift_source import StrategyFeatureDriftSource
from .strategy_recipes import RECIPE_VERSION, get_strategy_recipe, list_strategy_recipes
from .strategy_research_signal_binding import (
    build_strategy_research_signal_binding,
    research_feature_set_for_champion_selection,
)
from .strategy_rule_compiler import validate_strategy_rule_binding
from .strategy_store import StrategyStore
from .transparent_baseline_runner import bind_transparent_baseline_job_identity
from .worker import LocalJobWorker


def _scheduler_endpoint_health_check(
    scheduler_body: dict[str, Any],
    *,
    response_status_code: int,
    now: datetime,
    stale_after_seconds: int,
    max_active_tick_seconds: int,
) -> dict[str, Any]:
    """Independently validate the scheduler heartbeat returned to ``readyz``."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("scheduler readiness time must be timezone-aware")
    last_tick_raw = scheduler_body.get("last_tick")
    last_tick = datetime.fromisoformat(str(last_tick_raw))
    if last_tick.tzinfo is None:
        raise ValueError("scheduler last_tick must be timezone-aware")
    scheduler_age = max(0.0, (now - last_tick).total_seconds())
    tick_in_progress = scheduler_body.get("tick_in_progress") is True
    tick_started_at_raw = scheduler_body.get("tick_started_at")
    active_tick_age: float | None = None
    if tick_in_progress:
        tick_started_at = datetime.fromisoformat(str(tick_started_at_raw))
        if tick_started_at.tzinfo is None:
            raise ValueError("scheduler tick_started_at must be timezone-aware")
        active_tick_age = max(0.0, (now - tick_started_at).total_seconds())
    elif tick_started_at_raw is not None:
        raise ValueError("idle scheduler must not retain tick_started_at")
    reported_max_active_tick_seconds = int(
        scheduler_body.get("max_active_tick_seconds") or 0
    )
    freshness_ready = (
        active_tick_age <= max_active_tick_seconds
        if active_tick_age is not None
        else scheduler_age <= stale_after_seconds
    )
    ready = (
        response_status_code == 200
        and scheduler_body.get("status") == "ok"
        and scheduler_body.get("ready") is True
        and reported_max_active_tick_seconds == max_active_tick_seconds
        and freshness_ready
    )
    return {
        "status": "ok" if ready else "degraded",
        "message": (
            "scheduler tick is current"
            if ready
            else "scheduler health or tick freshness check failed"
        ),
        "last_tick": last_tick_raw,
        "age_seconds": scheduler_age,
        "stale_after_seconds": stale_after_seconds,
        "tick_in_progress": tick_in_progress,
        "tick_started_at": tick_started_at_raw,
        "active_tick_age_seconds": active_tick_age,
        "max_active_tick_seconds": max_active_tick_seconds,
        "reported_max_active_tick_seconds": reported_max_active_tick_seconds,
        "freshness_source": scheduler_body.get("freshness_source"),
    }


class BootstrapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: Literal["core", "research", "full"] = "full"
    # The primary Tushare-compatible gateway starts at 2016.  Pre-2016
    # history is admitted separately through the audited BaoStock workflow.
    start: date = Field(default=date(2016, 1, 1))
    snapshot_start: date = Field(default=date(2008, 1, 1))
    end: date | Literal["latest"] = "latest"
    build_qlib: bool = False

    @model_validator(mode="after")
    def validate_range(self) -> BootstrapRequest:
        if self.start != date(2016, 1, 1):
            raise ValueError(
                "full bootstrap primary download start must equal 2016-01-01; "
                "use the incremental scheduler for later updates"
            )
        if isinstance(self.end, date) and self.end < self.start:
            raise ValueError("end must not be before start")
        if self.snapshot_start > self.start:
            raise ValueError("snapshot_start must not be after the primary download start")
        if self.build_qlib and self.profile != "full":
            raise ValueError(
                "Qlib research finalization requires the full data profile; "
                "core/research profiles are download-only subsets"
            )
        return self


class BaoStockOverlapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: date = Field(default=date(2016, 1, 1))
    end: date = Field(default=date(2016, 12, 31))
    symbols: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_range(self) -> BaoStockOverlapRequest:
        if self.end < self.start:
            raise ValueError("end must not be before start")
        require_audited_overlap_symbols(self.symbols or DEFAULT_OVERLAP_SYMBOLS)
        return self


class LegacyMarketBackfillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: date = Field(default=date(2008, 1, 1))
    end: date = Field(default=date(2015, 12, 31))

    @model_validator(mode="after")
    def validate_range(self) -> LegacyMarketBackfillRequest:
        if self.end < self.start:
            raise ValueError("end must not be before start")
        if self.end >= date(2016, 1, 1):
            raise ValueError("legacy backfill must end before 2016-01-01")
        return self


class DataFinalizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: Literal["core", "research", "full", "research-assets"] = "full"
    start: date = Field(default=date(2008, 1, 1))
    end: date | Literal["latest"] = "latest"
    snapshot_name: str | None = Field(default=None, min_length=3, max_length=120)

    @model_validator(mode="after")
    def validate_range(self) -> DataFinalizeRequest:
        if isinstance(self.end, date) and self.end < self.start:
            raise ValueError("end must not be before start")
        if self.profile not in {"full", "research-assets"}:
            raise ValueError(
                "publication requires the full data profile for Qlib or the isolated "
                "research-assets profile; core/research are download-only subsets"
            )
        return self


class AnnouncementNlpRequest(BaseModel):
    start: date = Field(default=date(2024, 1, 1))
    end: date | Literal["latest"] = "latest"
    ts_codes: list[str] = Field(default_factory=list, max_length=2000)
    categories: list[Literal["announcement", "regulatory_letter"]] = Field(
        default_factory=lambda: ["regulatory_letter"]
    )
    limit: int = Field(default=0, ge=0, le=1_000_000)
    batch_size: int = Field(
        default=ANNOUNCEMENT_DEFAULT_BATCH_SIZE,
        ge=1,
        le=ANNOUNCEMENT_MAX_BATCH_SIZE,
    )
    workers: int = Field(
        default=ANNOUNCEMENT_DEFAULT_WORKERS,
        ge=1,
        le=ANNOUNCEMENT_MAX_WORKERS,
    )

    @model_validator(mode="after")
    def validate_range(self) -> AnnouncementNlpRequest:
        if isinstance(self.end, date) and self.end < self.start:
            raise ValueError("end must not be before start")
        return self


class CorpusNlpRequest(BaseModel):
    start: date = Field(default=date(2024, 1, 1))
    end: date | Literal["latest"] = "latest"
    datasets: list[Literal["major_news", "npr", "cctv_news", "irm_qa_sh", "irm_qa_sz"]] = Field(
        default_factory=list
    )
    ts_codes: list[str] = Field(default_factory=list, max_length=2000)
    limit: int = Field(default=0, ge=0, le=1_000_000)
    batch_size: int = Field(default=CORPUS_DEFAULT_BATCH_SIZE, ge=1, le=100)
    workers: int = Field(default=CORPUS_DEFAULT_WORKERS, ge=1, le=CORPUS_MAX_WORKERS)
    major_news_per_day: int = Field(default=CORPUS_DEFAULT_MAJOR_NEWS_PER_DAY, ge=0)
    irm_per_instrument_day: int = Field(default=CORPUS_DEFAULT_IRM_PER_INSTRUMENT_DAY, ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> CorpusNlpRequest:
        if isinstance(self.end, date) and self.end < self.start:
            raise ValueError("end must not be before start")
        return self


class EventMarketResponseRequest(BaseModel):
    snapshot_name: str = Field(min_length=3, max_length=120)
    horizons: list[int] = Field(default_factory=lambda: [1, 3, 5, 20], min_length=1)
    benchmark_code: str = Field(default="000300.SH", min_length=9, max_length=12)

    @model_validator(mode="after")
    def validate_horizons(self) -> EventMarketResponseRequest:
        if any(value <= 0 or value > 252 for value in self.horizons):
            raise ValueError("horizons must be between 1 and 252 trading sessions")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must not contain duplicates")
        return self


class MultifaceAuditRequest(BaseModel):
    dataset: str = Field(min_length=3, max_length=300)
    snapshot_name: str | None = Field(default=None, min_length=3, max_length=120)
    require_ready: bool = True


class MarginEligibilityRequest(BaseModel):
    start: date = Field(default=date(2024, 1, 1))
    end: date | Literal["latest"] = "latest"

    @model_validator(mode="after")
    def validate_range(self) -> MarginEligibilityRequest:
        if isinstance(self.end, date) and self.end < self.start:
            raise ValueError("end must not be before start")
        return self


class CoreIntradayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    daily_dataset: str | None = Field(default=None, min_length=1, max_length=120)

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
    publish_research_assets: bool = False
    snapshot_name: str | None = Field(default=None, min_length=3, max_length=120)

    @model_validator(mode="after")
    def validate_range(self) -> SupplementalDownloadRequest:
        if isinstance(self.end, date) and self.end < self.start:
            raise ValueError("end must not be before start")
        if self.bundle == "strategy_specialty_minutes" and not self.symbols:
            raise ValueError("strategy_specialty_minutes requires explicit symbols")
        if self.publish_research_assets and self.bundle != "research_corpus":
            raise ValueError(
                "publish_research_assets is available only for the research_corpus bundle"
            )
        return self


class Ashare5mRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: date = Field(default=date(2024, 1, 1))
    end: date | Literal["latest"] = "latest"
    snapshot_name: str | None = Field(default=None, min_length=3, max_length=120)
    daily_dataset: str | None = Field(default=None, min_length=1, max_length=120)

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


class SafeModeEngageRequest(BaseModel):
    actor: str = Field(default="", max_length=100)
    reason: str = Field(min_length=5, max_length=1000)


class SafeModeReleaseRequest(BaseModel):
    actor: str = Field(default="", max_length=100)
    reason: str = Field(min_length=10, max_length=1000)
    # Recovery evidence is mandatory.  Literal[True] keeps older clients that
    # send the flag working while rejecting an explicit bypass attempt.
    require_health_ok: Literal[True] = True


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
    target_frequency: Literal["1min", "5min", "15min", "30min", "60min"] | None = None


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
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    test_start: date
    test_end: date

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
        valid_days = sum(
            day.weekday() < 5
            for day in (
                self.valid_start + timedelta(days=offset)
                for offset in range((self.valid_end - self.valid_start).days + 1)
            )
        )
        test_days = sum(
            day.weekday() < 5
            for day in (
                self.test_start + timedelta(days=offset)
                for offset in range((self.test_end - self.test_start).days + 1)
            )
        )
        if valid_days < 126 or test_days < 252:
            raise ValueError("validation requires 126 and final test requires 252 trading days")
        return self


class ResearchPeriodPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_trading_days: int = Field(default=252, ge=252, le=1260)
    embargo_trading_days: int = Field(default=20, ge=6, le=253)

    @model_validator(mode="after")
    def validate_policy(self) -> ResearchPeriodPolicy:
        normalize_research_period_policy(self.model_dump())
        return self


class ResearchAssetAutomaticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: date | None = None
    include_tushare: bool = True
    include_arxiv: bool = True
    actor: str = Field(default="local-operator", min_length=2, max_length=100)

    @model_validator(mode="after")
    def validate_sources(self) -> ResearchAssetAutomaticRequest:
        if not self.include_tushare and not self.include_arxiv:
            raise ValueError("at least one automatic research source is required")
        return self


class ResearchAssetManualHttpsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=12, max_length=2000)
    title: str = Field(min_length=3, max_length=500)
    document_kind: Literal["paper", "research_report"] = "paper"
    published_at: datetime | None = None
    actor: str = Field(default="local-operator", min_length=2, max_length=100)

    @model_validator(mode="after")
    def validate_remote_pdf(self) -> ResearchAssetManualHttpsRequest:
        parsed = urlsplit(self.url.strip())
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError("manual research documents require a public HTTPS URL")
        if self.published_at is not None and (
            self.published_at.tzinfo is None or self.published_at.utcoffset() is None
        ):
            raise ValueError("published_at must include a timezone")
        self.url = self.url.strip()
        self.title = self.title.strip()
        return self


class RDAgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=10, max_length=2000)
    scenario: str = "fin_factor"
    dataset: str | None = None
    asset_ids: list[str] = Field(default_factory=list, max_length=20)
    feature_set_id: str | None = None
    horizon: Literal["short", "swing", "long"] | None = None
    incumbent_strategy_version_id: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{32}$"
    )
    loop_n: int = Field(default=1, ge=1, le=20)
    duration: str = "30m"
    requested_by: str = Field(default="local-operator", min_length=2, max_length=100)
    period_policy: ResearchPeriodPolicy = Field(default_factory=ResearchPeriodPolicy)

    @model_validator(mode="after")
    def validate_budget(self) -> RDAgentRunRequest:
        validate_duration(self.duration)
        scenario = get_rdagent_scenario(self.scenario)
        self.scenario = scenario.id
        if scenario.id in HORIZON_RESEARCH_SCENARIOS:
            if self.horizon is None:
                raise ValueError(
                    f"{scenario.id} requires an explicit short, swing, or long horizon"
                )
            required_period_policy = {
                # The 1-5 day label contract needs six sessions of local
                # purge, but every capital-facing final OOS also inherits the
                # platform-wide 20-session unopened embargo.
                "short": (252, 20),
                "swing": (504, 127),
                "long": (756, 253),
            }[self.horizon]
            required_oos, required_embargo = required_period_policy
            if "period_policy" not in self.model_fields_set:
                self.period_policy = ResearchPeriodPolicy(
                    test_trading_days=required_oos,
                    embargo_trading_days=required_embargo,
                )
            elif (
                self.period_policy.test_trading_days < required_oos
                or self.period_policy.embargo_trading_days < required_embargo
            ):
                raise ValueError(
                    f"{scenario.id} period policy is weaker than its frozen horizon contract"
                )
            if (
                scenario.id != "fin_strategy"
                and self.incumbent_strategy_version_id is not None
            ):
                raise ValueError(
                    "incumbent_strategy_version_id is accepted only by fin_strategy"
                )
        elif self.horizon is not None or self.incumbent_strategy_version_id is not None:
            raise ValueError(
                "horizon is accepted only by active horizon research scenarios"
            )
        self.asset_ids = [validate_asset_id(value) for value in self.asset_ids]
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("asset_ids contain duplicates")
        self.feature_set_id = validate_feature_set_id(self.feature_set_id)
        if scenario.requires_dataset and not str(self.dataset or "").strip():
            raise ValueError(f"{scenario.id} requires a Qlib dataset")
        if not scenario.requires_dataset and self.dataset is not None:
            raise ValueError(f"{scenario.id} does not accept a Qlib dataset")
        if scenario.requires_feature_set:
            if self.feature_set_id is None:
                raise ValueError(f"{scenario.id} requires a governed feature_set_id")
            get_feature_set(self.feature_set_id)
        elif self.feature_set_id is not None:
            raise ValueError(f"{scenario.id} does not accept feature_set_id")
        return self


class GeneralModelValidationRequest(BaseModel):
    """Promote one implementation artifact into the governed model research gate.

    The client may select only opaque platform identities and a frozen recipe.
    It can never provide executable paths, commands, or environment variables.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=16, max_length=64)
    dataset: str = Field(min_length=2, max_length=200)
    feature_set_id: str = Field(min_length=2, max_length=100)
    name: str | None = Field(default=None, min_length=2, max_length=200)
    description: str | None = Field(default=None, min_length=2, max_length=2000)
    model_type: str = Field(default="Tabular", min_length=2, max_length=100)
    architecture: dict[str, Any] = Field(default_factory=dict)
    model_hyperparameters: dict[str, Any] = Field(default_factory=dict)
    training_hyperparameters: dict[str, Any] = Field(default_factory=dict)
    requested_by: str = Field(default="local-operator", min_length=2, max_length=100)
    period_policy: ResearchPeriodPolicy = Field(default_factory=ResearchPeriodPolicy)

    @model_validator(mode="after")
    def validate_governed_ids(self) -> GeneralModelValidationRequest:
        self.feature_set_id = str(validate_feature_set_id(self.feature_set_id))
        get_feature_set(self.feature_set_id)
        if self.period_policy.embargo_trading_days != 20:
            raise ValueError(
                "general_model validation currently requires the governed "
                "20-trading-day final-OOS embargo"
            )
        return self


class FactorLibraryMaterializationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(min_length=2, max_length=200)
    feature_set_id: str = "unified-research-v1"
    universe: str = Field(default="cn_all", min_length=2, max_length=100)
    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def validate_feature_set(self) -> FactorLibraryMaterializationRequest:
        self.feature_set_id = str(validate_feature_set_id(self.feature_set_id))
        get_feature_set(self.feature_set_id)
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("factor materialization start must not be after end")
        return self


_PUBLIC_WINDOWS_PATH = re.compile(r"(?i)(?:^|\s)[a-z]:[\\/]")
_PUBLIC_UNIX_PATH = re.compile(
    r"(?:^|\s)/(?:app|data|etc|home|mnt|opt|root|run|srv|tmp|usr|var)(?:/|\b)"
)
_PUBLIC_SECRET_TEXT = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{12,}|"
    r"(?:api[_ -]?key|authorization|password|secret|token)\s*[:=]\s*\S+)"
)


def _public_string(value: str) -> str:
    lowered = value.lower()
    if (
        "://" in lowered
        or "bearer " in lowered
        or value.startswith(("\\\\", "file:"))
        or _PUBLIC_WINDOWS_PATH.search(value)
        or _PUBLIC_UNIX_PATH.search(value)
        or _PUBLIC_SECRET_TEXT.search(value)
    ):
        return "[redacted]"
    return value


def _sanitize_public_value(value: Any) -> Any:
    hidden_keys = {
        "artifact_path",
        "code_path",
        "command",
        "cwd",
        "dataset_path",
        "env",
        "environment",
        "error",
        "headers",
        "log",
        "log_path",
        "path",
        "paths",
        "recomputed_values_path",
        "storage_path",
        "trace_path",
        "traceback",
        "url",
        "urls",
        "uri",
        "values_path",
        "workspace_path",
        "stderr",
        "stdout",
    }
    safe_boolean_readiness_keys = {
        "credentials_configured",
        "llm_credentials_configured",
    }
    if isinstance(value, dict):
        return {
            str(key): _sanitize_public_value(item)
            for key, item in value.items()
            if str(key).lower() not in hidden_keys
            and not str(key).lower().endswith(("_path", "_paths", "_url", "_uri"))
            and (
                (
                    str(key).lower() in safe_boolean_readiness_keys
                    and isinstance(item, bool)
                )
                or not any(
                    token in str(key).lower()
                    for token in (
                        "api_key",
                        "authorization",
                        "cookie",
                        "credential",
                        "password",
                        "secret",
                        "token",
                    )
                )
            )
        }
    if isinstance(value, list):
        return [_sanitize_public_value(item) for item in value]
    if isinstance(value, str):
        return _public_string(value)
    return value


def _public_rdagent_run(run: dict[str, Any]) -> dict[str, Any]:
    result = _sanitize_public_value(dict(run))
    result["error"] = "research run failed" if run.get("error") else None
    result["scenario"] = scenario_from_research_run(result)
    return result


def _rdagent_trace_view(
    store: RDAGentCandidateStore,
    *,
    research_run_id: str,
    scenario_id: str,
    run_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Read the immutable, path-free Loop/Hypothesis/Feedback projection."""

    artifact_type = f"{scenario_id}_sanitized_result"
    candidate = next(
        (
            item
            for item in run_artifacts
            if item.get("artifact_type") == artifact_type
            and item.get("status") == "recorded"
        ),
        None,
    )
    if candidate is None:
        return {"status": "unavailable", "contract_version": None, "loops": []}
    try:
        artifact = store.get_run_artifact(str(candidate["id"]), verify=True)
        if str(artifact.get("research_run_id") or "") != research_run_id:
            raise ValueError("trace artifact belongs to another research run")
        payload = json.loads(Path(str(artifact["storage_path"])).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("trace projection envelope must be an object")
        contract_version = str(payload.get("trace_contract_version") or "")
        loops = payload.get("trace_loops")
        if contract_version != "rdagent-trace-web-v1" or not isinstance(loops, list):
            return {"status": "legacy_summary_only", "contract_version": None, "loops": []}
        if any(not isinstance(item, dict) for item in loops[:20]):
            raise ValueError("trace projection contains an invalid loop")
        return {
            "status": "recorded",
            "contract_version": contract_version,
            "loops": loops[:20],
            "summary": payload.get("trace_summary") or {},
        }
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {"status": "unavailable", "contract_version": None, "loops": []}


def _public_rdagent_status(status: dict[str, Any]) -> dict[str, Any]:
    """Expose readiness decisions, never host/runtime addressing details."""

    scenario_rows = status.get("scenarios")
    scenarios = scenario_rows if isinstance(scenario_rows, list) else []
    workers: dict[str, Any] = {}
    for key in ("evaluation_worker", "data_science_worker", "gpu_worker"):
        worker = status.get(key)
        if isinstance(worker, dict):
            workers[key] = {
                field: worker.get(field)
                for field in (
                    "status",
                    "ready",
                    "runtime_identity_matches",
                    "model_sandbox_ready",
                    "gpu_available",
                    "gpu_memory_free_mb",
                    "gpu_driver_version",
                    "cuda_version",
                    "docker_gpu_runtime_available",
                    "docker_gpu_smoke_passed",
                    "data_root_free_gb",
                )
                if field in worker
            }
    public = {
        "status": status.get("status"),
        "version": status.get("version"),
        "commit": status.get("commit"),
        "enabled": bool(status.get("enabled")),
        "ready": bool(status.get("ready")),
        "blockers": status.get("blockers") or [],
        "limits": status.get("limits") or {},
        "scenarios": scenarios,
        "llm_credentials_configured": bool(status.get("llm_credentials_configured")),
        "docker_available": bool(status.get("docker_available")),
        "qlib_data_ready": bool(status.get("qlib_data_ready")),
        "workers": workers,
    }
    sanitized = _sanitize_public_value(public)
    # The generic sanitizer correctly strips credential-shaped fields, but this
    # one is an intentionally public readiness boolean (never a credential).
    # Re-add it after sanitization so the UI cannot claim "not configured"
    # while the governed worker has already proved that credentials are present.
    sanitized["llm_credentials_configured"] = bool(
        status.get("llm_credentials_configured")
    )
    return sanitized


_PUBLIC_JOB_PAYLOAD_KEYS = frozenset(
    {
        "as_of",
        "bundle",
        "dataset",
        "document_kind",
        "end",
        "frequency",
        "mode",
        "output_name",
        "pipeline_id",
        "profile",
        "scenario",
        "snapshot_name",
        "start",
        "target_frequency",
    }
)
_PUBLIC_JOB_PROGRESS_KEYS = frozenset(
    {
        "checkpoint",
        "completed_count",
        "datasets",
        "execution_phase",
        "failed_count",
        "phase_label",
        "status",
        "succeeded_count",
        "target",
        "trial_count",
        "updated_at",
    }
)
_MODEL_OUTCOME_EVALUATION_STATUSES = ("passed", "resource_blocked")


def _evaluation_status_count(progress: dict[str, Any], status: str) -> int:
    raw_counts = progress.get(EVALUATION_STATUS_COUNTS_KEY)
    if not isinstance(raw_counts, dict):
        return 0
    try:
        return max(0, int(raw_counts.get(status) or 0))
    except (TypeError, ValueError):
        return 0


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("payload")
    public_payload = (
        {str(key): value for key, value in payload.items() if str(key) in _PUBLIC_JOB_PAYLOAD_KEYS}
        if isinstance(payload, dict)
        else {}
    )
    progress = job.get("progress")
    public_progress = (
        {
            str(key): value
            for key, value in progress.items()
            if str(key) in _PUBLIC_JOB_PROGRESS_KEYS
        }
        if isinstance(progress, dict)
        else None
    )
    result = _sanitize_public_value(
        {
            "id": job.get("id"),
            "kind": job.get("kind"),
            "status": job.get("status"),
            "payload": public_payload,
            "progress": public_progress,
            "attempts": job.get("attempts"),
            "max_attempts": job.get("max_attempts"),
            "next_attempt_at": job.get("next_attempt_at"),
            "cancel_requested_at": job.get("cancel_requested_at"),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "exit_code": job.get("exit_code"),
            "retry_successor": job.get("retry_successor"),
        }
    )
    raw_progress = progress if isinstance(progress, dict) else {}
    outcome_status: str | None = None
    outcome_message: str | None = None
    if str(job.get("kind") or "") == "model_evaluate" and job.get("status") == "succeeded":
        resource_blocked = max(
            int(raw_progress.get("resource_blocked_count") or 0),
            _evaluation_status_count(raw_progress, "resource_blocked"),
        )
        passed = _evaluation_status_count(raw_progress, "passed")
        if resource_blocked:
            outcome_status = "blocked"
            outcome_message = "评估程序已完成，但候选模型超出受治理的计算资源预算。"
        elif passed:
            outcome_status = "passed"
            outcome_message = f"独立模型门禁通过 {passed} 个候选。"
        else:
            outcome_status = "rejected"
            outcome_message = "评估程序已完成，但没有候选通过独立模型门禁。"
    result["outcome_status"] = outcome_status
    result["outcome_message"] = outcome_message
    if job.get("error"):
        result["error"] = "job execution failed"
    else:
        result["error"] = None
    return result


def _public_research_asset_job(job: dict[str, Any]) -> dict[str, Any]:
    """Expose acquisition state without URLs, commands, paths, or raw payloads."""

    progress = dict(job.get("progress") or {})
    return {
        "id": str(job["id"]),
        "kind": str(job["kind"]),
        "status": str(job["status"]),
        "mode": str((job.get("payload") or {}).get("mode") or ""),
        "attempts": int(job.get("attempts") or 0),
        "max_attempts": int(job.get("max_attempts") or 0),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        # Worker/log failures can contain URLs or host paths.  The acquisition
        # detail endpoint intentionally exposes only a stable public message.
        "error": "research asset acquisition failed" if job.get("error") else None,
        "result": {
            key: progress[key]
            for key in (
                "status",
                "mode",
                "published",
                "assets",
                "blocked",
                "failed",
                "tushare_selected",
                "arxiv_selected",
                "daily_limits",
            )
            if key in progress
        },
    }


class FactorEvaluationRequest(BaseModel):
    dataset: str
    periods: ResearchPeriods
    metrics: dict[str, Any]
    artifact_path: str | None = None
    recomputed_values_path: str
    recomputed_values_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recompute_evidence: dict[str, Any]


class ExternalFactorEvaluationRequest(BaseModel):
    dataset: str
    candidate_ids: list[str] = Field(min_length=1, max_length=50)
    periods: ResearchPeriods
    universe: str = Field(default="cn_all", min_length=2, max_length=100)
    benchmark: str = Field(default="SH000300", min_length=4, max_length=32)

    @model_validator(mode="after")
    def validate_candidates(self) -> ExternalFactorEvaluationRequest:
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must not contain duplicates")
        return self


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
    recipe_id: Literal[
        "custom",
        "index_enhancement",
        "short_relative_strength",
        "swing_trend",
        "long_quality_value",
        "full_market_multifactor",
        "minute_mean_reversion",
    ] = "custom"
    recipe_version: str = Field(default="custom", min_length=1, max_length=100)
    factor_source_mode: Literal[
        "promoted_only",
        "qlib_baseline",
        "qlib_baseline_plus_challenger",
        "qlib_challenger_replacement",
        "not_applicable_model_prediction",
    ] = "promoted_only"
    challenger_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    baseline_definition: dict[str, Any] | None = None
    baseline_definition_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    signal_source: Literal["factor_score", "model_prediction"] = "factor_score"
    model_candidate_id: str | None = Field(default=None, min_length=1, max_length=128)
    model_evaluation_id: str | None = Field(default=None, min_length=1, max_length=128)
    model_code_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_recipe_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    feature_set_id: str | None = Field(default=None, min_length=1, max_length=128)
    feature_set_definition_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    quant_bundle_candidate_id: str | None = Field(default=None, min_length=1, max_length=128)
    quant_bundle_evaluation_id: str | None = Field(default=None, min_length=1, max_length=128)
    quant_bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_ensemble_candidate_id: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    model_ensemble_evaluation_id: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    model_ensemble_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    model_ensemble_evidence_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    model_ensemble_combiner: Literal["equal_rank"] | None = None
    model_ensemble_stacking: bool | None = None
    model_component_candidate_ids: list[str] | None = Field(
        default=None, min_length=2, max_length=3
    )
    model_component_families: list[str] | None = Field(
        default=None, min_length=2, max_length=3
    )
    horizon_profile: Literal[
        "short_1_5d", "swing_1_6m", "long_1_3y", "legacy_ambiguous"
    ] = LEGACY_AMBIGUOUS
    source_research_artifact_id: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    strategy_research_proposal_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    strategy_research_artifact_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    parent_strategy_version_id: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    strategy_research_data_contract: dict[str, Any] | None = None
    strategy_evaluation_contract: dict[str, Any] | None = None
    strategy_rule_ir: dict[str, Any] | None = None
    strategy_rules_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    strategy_rule_policy_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    signal_frequency: Literal["day", "1min", "5min", "15min", "30min", "60min"] = "day"
    signal_period: int = Field(default=1, ge=1, le=1260)
    execution_frequency: Literal["day", "1min", "5min", "15min", "30min", "60min"] = "day"
    execution_lag_bars: Literal[1] = 1
    execution_contract_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    rebalance_frequency: Literal["bar", "day", "week", "month"] = "day"
    lot_size: Literal[100] = 100
    min_listing_days: int = Field(default=0, ge=0, le=2520)
    entry_score_min_percentile: float = Field(default=0.0, ge=0.0, le=1.0)
    score_drop_exit_percentile: float | None = Field(default=None, ge=0.0, le=1.0)
    score_deterioration_reduce_percentile: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    score_deterioration_reduce_fraction: float = Field(default=0.50, ge=0.0, lt=1.0)
    extension_guard_max_return_5d: float | None = Field(
        default=None, ge=0.0, le=5.0
    )
    holding_min_sessions: int | None = Field(default=None, ge=1, le=756)
    max_holding_sessions: int | None = Field(default=None, ge=1, le=756)
    min_rebalance_weight_change: float = Field(default=0.0, ge=0.0, le=1.0)
    market_trend_lookback_sessions: int | None = Field(
        default=None, ge=2, le=756
    )
    market_trend_benchmark: str | None = Field(default=None, min_length=2, max_length=32)
    valuation_regime_max_percentile: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    valuation_reduce_percentile: float | None = Field(default=None, ge=0.0, le=1.0)
    valuation_reduce_fraction: float = Field(default=0.50, ge=0.0, lt=1.0)
    trend_break_lookback_sessions: int | None = Field(default=None, ge=2, le=756)
    thesis_min_holding_sessions: int | None = Field(default=None, ge=1, le=756)
    thesis_break_score_percentile: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    thesis_review_frequency: Literal["month"] | None = None
    hard_risk_target_fraction: float = Field(default=0.50, ge=0.0, lt=1.0)
    cash_when_no_edge: bool = False
    topk: int = Field(default=50, ge=5, le=500)
    n_drop: int = Field(default=5, ge=0, le=100)
    max_position_weight: float = Field(default=0.02, gt=0, le=0.20)
    max_daily_turnover: float = Field(default=0.15, gt=0, le=1.0)
    max_daily_loss: float = Field(default=0.03, gt=0, le=0.20)
    stop_loss: float = Field(default=0.07, gt=0, le=0.50)
    profit_taking_mode: Literal["threshold", "rule_only", "thesis_only"] = "threshold"
    take_profit_partial: float = Field(default=0.12, gt=0, le=2.0)
    take_profit_partial_fraction: float = Field(default=0.50, gt=0, lt=1.0)
    take_profit: float = Field(default=0.20, gt=0, le=5.0)
    max_drawdown_reduce: float = Field(default=0.10, gt=0, le=0.50)
    max_drawdown_liquidate: float = Field(default=0.15, gt=0, le=0.80)
    drawdown_reduction_exposure: float = Field(default=0.50, gt=0, lt=1.0)
    max_industry_weight: float = Field(default=0.30, gt=0, le=1.0)
    max_industry_deviation: float = Field(default=0.03, ge=0, le=0.30)
    max_size_deviation: float = Field(default=0.30, ge=0, le=2.0)
    max_value_deviation: float = Field(default=0.30, ge=0, le=2.0)
    max_growth_deviation: float = Field(default=0.30, ge=0, le=2.0)
    max_volatility_deviation: float = Field(default=0.30, ge=0, le=2.0)
    portfolio_construction: Literal[
        "topk_equal_weight", "benchmark_relative_qp", "industry_neutral_qp"
    ] = "topk_equal_weight"
    industry_relative_rank: bool = False
    optimizer_alpha_weight: float = Field(default=0.05, ge=0, le=10.0)
    optimizer_tracking_penalty: float = Field(default=1.0, ge=0, le=100.0)
    optimizer_turnover_penalty: float = Field(default=0.10, ge=0, le=100.0)
    min_average_daily_amount: float = Field(default=500_000_000, ge=1_000_000, le=100_000_000_000)
    liquidity_lookback_days: int = Field(default=20, ge=5, le=252)
    require_regulatory_events: bool = False
    max_tracking_error: float = Field(default=0.12, gt=0, le=1.0)
    min_cash_weight: float = Field(default=0.0, ge=0, lt=1.0)
    max_asset_class_weights: dict[str, float] | None = None
    target_volatility: float = Field(default=0.15, gt=0, le=0.50)
    max_drawdown: float = Field(default=0.25, gt=0, le=1.0)
    max_turnover: float = Field(default=0.60, gt=0, le=2.0)
    min_information_ratio: float = Field(default=0.0, ge=-5, le=10)
    min_sharpe_ratio: float = Field(default=0.0, ge=-5, le=10)
    min_sortino_ratio: float = Field(default=0.0, ge=-5, le=20)
    min_robustness_pass_rate: float = Field(default=1.0, ge=0, le=1)
    annual_minimum_acceptable_return: float = Field(default=0.0, gt=-1.0, le=1.0)
    # The first release does not hold a separate cash instrument, therefore
    # idle cash earns exactly zero. A non-zero research risk-free rate must
    # never leak into account P&L as obtainable cash yield.
    annual_cash_yield_rate: float = Field(default=0.0, ge=0.0, le=0.0)
    cash_yield_source: Literal["none_zero_yield"] = "none_zero_yield"
    rolling_window_days: int = Field(default=252, ge=60, le=1260)
    rolling_step_days: int = Field(default=63, ge=20, le=504)
    min_rolling_windows: int = Field(default=3, ge=2, le=20)
    min_rolling_pass_rate: float = Field(default=0.60, ge=0, le=1)
    outer_train_days: int = Field(default=252, ge=60, le=2520)
    outer_validation_days: int = Field(default=42, ge=10, le=504)
    outer_test_days: int = Field(default=42, ge=10, le=504)
    outer_purge_days: int | None = Field(default=None, ge=1, le=756)
    outer_embargo_days: int | None = Field(default=None, ge=1, le=756)
    minimum_outer_test_excess_return: float = Field(default=0.0, ge=-1.0, le=5.0)
    minimum_outer_test_pass_rate: float = Field(default=0.60, ge=0.50, le=1.0)
    # A production recommendation must be supported by a full pre-final
    # market-history regime sample.  Ten trading years is intentionally
    # separate from the once-only final OOS window below.
    min_pre_final_history_days: int = Field(default=2520, ge=2520, le=7560)
    event_window_days: int = Field(default=20, ge=20, le=126)
    event_count: int = Field(default=5, ge=1, le=20)
    max_event_underperformance: float = Field(default=0.05, ge=0, le=0.50)
    min_event_stress_pass_rate: float = Field(default=0.60, ge=0, le=1)
    min_backtest_days: int | None = Field(default=None, ge=252, le=2520)
    # Legacy compatibility only. New horizon paper accounts resolve their
    # principal from the active investor profile; capacity remains research-only.
    paper_initial_cash: float = Field(default=100_000, gt=0, le=10_000_000_000)
    capacity_notional: float = Field(default=5_000_000, ge=100_000, le=10_000_000_000)
    capacity_curve_notionals: list[float] = Field(
        default_factory=lambda: [5_000_000, 20_000_000, 100_000_000],
        min_length=3,
        max_length=10,
    )
    min_capacity_excess_return: float = Field(default=0.0, ge=-1.0, le=5.0)
    min_closed_trades: int = Field(default=20, ge=0, le=100_000)
    min_win_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    min_profit_loss_ratio: float = Field(default=0.0, ge=0.0, le=100.0)
    execution_days: int = Field(default=1, ge=1, le=5)
    execution_method: Literal["open", "twap", "vwap", "next_bar"] = "open"
    execution_slice_minutes: int = Field(default=20, ge=5, le=30)
    max_execution_slices: int = Field(default=24, ge=1, le=64)
    vwap_lookback_days: int = Field(default=20, ge=5, le=60)
    max_volume_participation: float = Field(default=0.01, gt=0, le=0.20)
    min_capacity_fill_ratio: float = Field(default=0.95, ge=0, le=1)
    cost_schedule_version: str = COST_SCHEDULE_VERSION
    effective_from: str = "2000-01-01"
    effective_to: str | None = None
    buy_commission_rate: float = Field(default=0.0005, ge=0, le=0.02)
    sell_commission_rate: float = Field(default=0.0005, ge=0, le=0.02)
    stock_sell_stamp_duty_rate: float = Field(
        default=CURRENT_STOCK_SELL_STAMP_DUTY_RATE, ge=0, le=0.02
    )
    etf_sell_stamp_duty_rate: float = Field(default=0.0, ge=0, le=0.02)
    transfer_fee_rate: float = Field(default=CURRENT_TRANSFER_FEE_RATE, ge=0, le=0.02)
    annual_borrow_rate: float = Field(default=0.0, ge=0, le=1.0)
    fixed_slippage_rate: float = Field(default=0.0005, ge=0, le=0.02)
    impact_at_max_participation: float = Field(default=0.0010, ge=0, le=0.10)
    min_commission: float = Field(default=5.0, ge=0, le=1000)
    execution_model: Literal["next_open"] = "next_open"

    @model_validator(mode="after")
    def valid_dropout(self) -> StrategyConfigRequest:
        horizon = research_horizon_contract(self.horizon_profile)
        if self.horizon_profile == LEGACY_AMBIGUOUS:
            self.outer_purge_days = self.outer_purge_days or 5
            self.outer_embargo_days = self.outer_embargo_days or 5
            self.min_backtest_days = self.min_backtest_days or 252
        else:
            assert horizon.purge_sessions is not None
            assert horizon.embargo_sessions is not None
            assert horizon.sealed_oos_sessions is not None
            self.outer_purge_days = self.outer_purge_days or horizon.purge_sessions
            self.outer_embargo_days = self.outer_embargo_days or horizon.embargo_sessions
            self.min_backtest_days = self.min_backtest_days or horizon.sealed_oos_sessions
            if self.outer_purge_days < horizon.purge_sessions:
                raise ValueError("outer purge is shorter than the sealed horizon contract")
            if self.outer_embargo_days < horizon.embargo_sessions:
                raise ValueError("outer embargo is shorter than the sealed horizon contract")
            if self.min_backtest_days < horizon.sealed_oos_sessions:
                raise ValueError("backtest period is shorter than the sealed horizon OOS")
        research_binding = (
            self.strategy_research_proposal_sha256,
            self.strategy_research_artifact_sha256,
            self.strategy_research_data_contract,
            self.strategy_evaluation_contract,
        )
        if self.source_research_artifact_id is not None and any(
            value is None for value in research_binding
        ):
            raise ValueError("strategy research artifact binding is incomplete")
        if self.source_research_artifact_id is None and any(
            value is not None for value in (*research_binding, self.parent_strategy_version_id)
        ):
            raise ValueError("strategy research metadata requires a source artifact")
        if self.signal_source == "model_prediction":
            if self.factor_source_mode not in {
                "promoted_only",
                "not_applicable_model_prediction",
            }:
                raise ValueError("model-prediction strategies cannot bind a factor-score baseline")
            if self.model_ensemble_candidate_id is not None:
                if (
                    self.model_candidate_id is not None
                    or self.model_ensemble_combiner != "equal_rank"
                    or self.model_ensemble_stacking is not False
                    or self.model_component_candidate_ids is None
                    or self.model_component_families is None
                    or len(set(self.model_component_candidate_ids))
                    != len(self.model_component_candidate_ids)
                    or len(set(self.model_component_families))
                    != len(self.model_component_families)
                    or len(self.model_component_candidate_ids)
                    != len(self.model_component_families)
                ):
                    raise ValueError(
                        "model ensemble requires distinct equal-rank non-stacking components"
                    )
        elif self.factor_source_mode == "not_applicable_model_prediction":
            raise ValueError(
                "the model-prediction factor-source sentinel is invalid for factor scores"
            )
        if self.recipe_id == "custom" and self.recipe_version != "custom":
            raise ValueError("custom strategy config must use the custom recipe version")
        if self.recipe_id != "custom" and self.recipe_version != RECIPE_VERSION:
            raise ValueError("strategy recipe version is not supported by this release")
        if self.n_drop > self.topk:
            raise ValueError("n_drop must not exceed topk")
        if self.max_industry_weight < self.max_position_weight:
            raise ValueError("max_industry_weight must not be below max_position_weight")
        if self.max_asset_class_weights is not None and any(
            not 0 <= float(limit) <= 1 for limit in self.max_asset_class_weights.values()
        ):
            raise ValueError("asset class limits must be between zero and one")
        if (
            self.portfolio_construction in {"benchmark_relative_qp", "industry_neutral_qp"}
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
        if self.profit_taking_mode == "thesis_only" and self.horizon_profile != "long_1_3y":
            raise ValueError("thesis-only profit taking is reserved for long_1_3y")
        if (
            self.profit_taking_mode == "threshold"
            and self.take_profit_partial >= self.take_profit
        ):
            raise ValueError("take_profit_partial must be below take_profit")
        if self.max_drawdown_reduce >= self.max_drawdown_liquidate:
            raise ValueError("max_drawdown_reduce must be below max_drawdown_liquidate")
        if len(set(self.capacity_curve_notionals)) < 3 or any(
            value <= 0 for value in self.capacity_curve_notionals
        ):
            raise ValueError("capacity curve requires at least three distinct positive notionals")
        if self.rolling_step_days > self.rolling_window_days:
            raise ValueError("rolling_step_days must not exceed rolling_window_days")
        required_outer_days = (
            self.outer_train_days
            + self.outer_purge_days
            + self.outer_validation_days
            + self.outer_embargo_days
            + 3 * self.outer_test_days
        )
        if self.min_pre_final_history_days < required_outer_days:
            raise ValueError(
                "min_pre_final_history_days must leave at least three complete outer test folds"
            )
        if self.execution_slice_minutes % 5:
            raise ValueError("execution_slice_minutes must be a multiple of five")
        CostModelConfig.from_mapping(self.model_dump())
        validate_strategy_rule_binding(self.model_dump())
        config = self.model_dump(exclude={"execution_contract_hash"})
        expected_contract_hash = strategy_execution_contract_hash(config)
        if (
            self.execution_contract_hash is not None
            and self.execution_contract_hash != expected_contract_hash
        ):
            raise ValueError("execution_contract_hash does not match the strategy contract")
        self.execution_contract_hash = expected_contract_hash
        require_strategy_execution_contract(self.model_dump())
        return self


def _rebind_strategy_execution_contract(values: dict[str, Any]) -> StrategyConfigRequest:
    rebound = dict(values)
    rebound.pop("execution_contract_hash", None)
    return StrategyConfigRequest.model_validate(rebound)


class StrategyCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=150)
    description: str = Field(min_length=10, max_length=2000)
    economic_hypothesis_group: str | None = Field(default=None, min_length=1, max_length=200)
    hypothesis_group_cap: float = Field(default=0.70, gt=0, le=0.70)
    benchmark: str = "SH000300"
    universe: str = "cn_all"
    factors: list[StrategyFactorRequest] = Field(default_factory=list, max_length=20)
    config: StrategyConfigRequest | None = None
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class StrategyVersionCreateRequest(BaseModel):
    benchmark: str = "SH000300"
    universe: str = "cn_all"
    factors: list[StrategyFactorRequest] = Field(default_factory=list, max_length=20)
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
    cost_schedule_version: str = COST_SCHEDULE_VERSION
    effective_from: str = "2000-01-01"
    effective_to: str | None = None
    buy_commission_rate: float = Field(default=0.0005, ge=0, le=0.02)
    sell_commission_rate: float = Field(default=0.0005, ge=0, le=0.02)
    stock_sell_stamp_duty_rate: float = Field(
        default=CURRENT_STOCK_SELL_STAMP_DUTY_RATE, ge=0, le=0.02
    )
    etf_sell_stamp_duty_rate: float = Field(default=0.0, ge=0, le=0.02)
    transfer_fee_rate: float = Field(default=CURRENT_TRANSFER_FEE_RATE, ge=0, le=0.02)
    min_commission: float = Field(default=5.0, ge=0, le=1000)
    fixed_slippage_rate: float = Field(default=0.0005, ge=0, le=0.02)
    impact_at_max_participation: float = Field(default=0.0010, ge=0, le=0.10)
    annual_borrow_rate: float = Field(default=0.08, gt=0, le=1)
    # None resolves per-leg board order-unit rules via market_rules.
    lot_size: int | None = Field(default=None, ge=1, le=10000)
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
        CostModelConfig.from_mapping(self.model_dump())
        return self


class PairStrategyCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=150)
    description: str = Field(min_length=10, max_length=2000)
    leg_y: str = Field(min_length=4, max_length=32)
    leg_x: str = Field(min_length=4, max_length=32)
    asset_class: Literal["etf", "stock", "mixed"] = "etf"
    shorting_mode: Literal["shadow_borrow", "margin_borrow"] = "shadow_borrow"
    config: PairStrategyConfigRequest = Field(default_factory=PairStrategyConfigRequest)
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class PairStrategyVersionCreateRequest(BaseModel):
    leg_y: str = Field(min_length=4, max_length=32)
    leg_x: str = Field(min_length=4, max_length=32)
    asset_class: Literal["etf", "stock", "mixed"] = "etf"
    shorting_mode: Literal["shadow_borrow", "margin_borrow"] = "shadow_borrow"
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
    execution_dataset: str | None = None
    start: date
    end: date

    @model_validator(mode="after")
    def valid_period(self) -> StrategyBacktestRequest:
        if self.end <= self.start:
            raise ValueError("backtest end must be after start")
        return self


class ModelArtifactCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_backtest_id: str = Field(min_length=1, max_length=200)
    valid_until: datetime
    actor: str = Field(default="model-operator", min_length=2, max_length=100)

    @model_validator(mode="after")
    def validate_model_artifact(self) -> ModelArtifactCreateRequest:
        if self.valid_until.tzinfo is None or self.valid_until.utcoffset() is None:
            raise ValueError("valid_until must include a timezone")
        return self


class ModelArtifactActivateRequest(BaseModel):
    actor: str = Field(default="model-reviewer", min_length=2, max_length=100)


class ModelRefitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_date: date
    valid_for_days: int = Field(default=4, ge=1, le=7)
    operation: Literal["inference", "retrain"] = "inference"
    retrain_reason: Literal["persistent_drift", "data_contract_change"] | None = None
    retrain_evidence: dict[str, Any] | None = None
    actor: str = Field(default="model-operator", min_length=2, max_length=100)

    @model_validator(mode="after")
    def validate_retrain_trigger(self) -> ModelRefitRequest:
        if self.operation == "inference":
            if self.retrain_reason is not None or self.retrain_evidence is not None:
                raise ValueError("daily inference cannot carry retraining evidence")
        elif self.retrain_reason is None or not isinstance(self.retrain_evidence, dict):
            raise ValueError("early retraining requires explicit trigger evidence")
        return self


class ParameterExperimentRequest(BaseModel):
    dataset: str
    execution_dataset: str | None = None
    start: date
    end: date
    parameter_grid: dict[str, list[int | float]]
    max_trials: int = Field(default=27, ge=1, le=81)
    actor: str = Field(default="local-operator", min_length=2, max_length=100)

    @model_validator(mode="after")
    def valid_period(self) -> ParameterExperimentRequest:
        split_research_period(self.start, self.end)
        return self


class StrategyApprovalRequest(BaseModel):
    actor: str = Field(min_length=2, max_length=100)
    reason: str = Field(min_length=10, max_length=2000)


class PaperStageOpenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class StrategyPromotionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = Field(min_length=2, max_length=100)
    reason: str = Field(min_length=10, max_length=2000)


class StrategyAllocationMemberRequest(BaseModel):
    strategy_version_id: str
    weight: float | None = Field(default=None, gt=0, le=1)
    role: Literal["core", "satellite"] = "core"
    risk_budget: float = Field(default=1.0, gt=0, le=1)
    member_cap: float | None = Field(default=None, gt=0, le=0.70)


class StrategyAllocationCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=150)
    dataset: str
    total_capital: float = Field(gt=0, le=10_000_000_000)
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


class RecommendationPortfolioCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=150)
    strategy_version_id: str
    dataset: str
    dataset_roll_policy: Literal["pinned", "latest_compatible"] = "latest_compatible"
    construction_notional: float = Field(
        gt=0,
        validation_alias=AliasChoices("construction_notional", "hypothetical_initial_value"),
    )
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class RecommendationRefreshRequest(BaseModel):
    as_of_date: date


class ActiveRecommendationAccountRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation_portfolio_id: str = Field(min_length=1, max_length=200)
    account_type: Literal["main_paper", "manual_shadow"]
    account_id: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=10, max_length=2000)


class InvestorMarketPermissionsRequest(BaseModel):
    """Explicit first-run market permissions; no permission is inferred."""

    model_config = ConfigDict(extra="forbid")

    main_board: bool
    star_market: bool
    chi_next: bool
    beijing_exchange: bool
    etf: bool


class InvestorSimulationProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_capital: float = Field(gt=0, le=10_000_000_000)
    risk_profile: Literal["conservative", "balanced", "aggressive", "custom"] = (
        "balanced"
    )
    min_cash_weight: float = Field(default=0.10, ge=0, lt=1)
    max_gross_exposure: float = Field(default=0.90, gt=0, le=1)
    market_permissions: InvestorMarketPermissionsRequest
    actor: str = Field(default="local-operator", min_length=2, max_length=100)

    @model_validator(mode="after")
    def validate_risk_profile(self) -> InvestorSimulationProfileRequest:
        if self.min_cash_weight + self.max_gross_exposure > 1.0 + 1e-12:
            raise ValueError("minimum cash plus maximum exposure cannot exceed 100%")
        if self.risk_profile == "balanced" and (
            abs(self.min_cash_weight - 0.10) > 1e-12
            or abs(self.max_gross_exposure - 0.90) > 1e-12
        ):
            raise ValueError("balanced profile freezes 10% cash and 90% maximum exposure")
        return self


class SimulationPortfolioCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3, max_length=150)
    recommendation_portfolio_id: str | None = None
    source_type: Literal["recommendation", "strategy_version", "allocation"] | None = None
    source_id: str | None = None
    execution_dataset: str | None = None
    execution_frequency: Literal["day", "1min", "5min"] = "day"
    execution_adapter: Literal["long_only", "pair"] | None = None
    execution_contract_hash: str | None = Field(default=None, min_length=64, max_length=64)
    daily_roll_policy: Literal["pinned", "latest_compatible"] = "latest_compatible"
    execution_roll_policy: Literal["pinned", "latest_compatible"] = "latest_compatible"
    initial_cash: float = Field(gt=0)
    execution_algorithm: Literal["open", "twap", "vwap", "next_bar"] | None = None
    slice_minutes: int | None = Field(default=None, ge=5, le=30, multiple_of=5)
    max_slices: int | None = Field(default=None, ge=1, le=64)
    max_participation: float | None = Field(default=None, gt=0, le=0.20)
    cost_schedule_version: str = COST_SCHEDULE_VERSION
    actor: str = Field(default="local-operator", min_length=2, max_length=100)

    @model_validator(mode="after")
    def validate_source(self) -> SimulationPortfolioCreateRequest:
        if self.recommendation_portfolio_id:
            if self.source_type not in {None, "recommendation"}:
                raise ValueError("recommendation_portfolio_id conflicts with source_type")
            if self.source_id and self.source_id != self.recommendation_portfolio_id:
                raise ValueError("recommendation source identifiers disagree")
        elif not self.source_type or not self.source_id:
            raise ValueError("simulation source_type and source_id are required")
        if self.execution_frequency == "day":
            if self.execution_algorithm not in {None, "open"}:
                raise ValueError("daily simulation supports only next-session-open execution")
        elif not self.execution_dataset:
            raise ValueError("minute simulation requires an execution_dataset")
        return self


class SimulationOrderPlanBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_plan_manifest_sha256: str = Field(min_length=64, max_length=64)
    actor: str = Field(default="simulation-operator", min_length=2, max_length=100)


class SimulationOrderPlanGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_date: date
    signal_at: datetime | None = None
    actor: str = Field(default="simulation-operator", min_length=2, max_length=100)

    @model_validator(mode="after")
    def validate_signal_time(self) -> SimulationOrderPlanGenerationRequest:
        if self.signal_at is None:
            return self
        if self.signal_at.tzinfo is None or self.signal_at.utcoffset() is None:
            raise ValueError("simulation signal timestamp must include a timezone")
        shanghai_date = self.signal_at.astimezone(ZoneInfo("Asia/Shanghai")).date()
        if shanghai_date != self.signal_date:
            raise ValueError("simulation signal timestamp does not match signal_date")
        return self


class PairSimulationReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backtest_id: str = Field(min_length=1, max_length=200)
    trade_date: date
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class SimulationNavReviewRequest(BaseModel):
    actor: str = Field(default="local-operator", min_length=2, max_length=100)
    evidence_sha256: str = Field(min_length=64, max_length=64)
    note: str = Field(min_length=10, max_length=2000)


class SimulationFinalFeeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_fee: float = Field(ge=0)
    evidence_sha256: str = Field(min_length=64, max_length=64)
    source: Literal["end_of_day", "user_import"] = "user_import"
    adjustment_key: str | None = Field(default=None, min_length=1, max_length=200)
    actor: str = Field(default="local-operator", min_length=2, max_length=100)


class ScheduleCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3, max_length=150)
    kind: Literal[
        "incremental_sync",
        "data_pipeline",
        "information_pipeline",
        "information_factor_refresh",
        "ashare_5m_sync",
        "auxiliary_data_pipeline",
        "rdagent_research",
        "recommendation_refresh",
        "weekly_report",
        "monthly_decision_day",
        "preopen_check",
        "intraday_execution_check",
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
        for key, minimum, maximum in (
            ("download_workers", 1, 16),
            ("requests_per_minute", 1, 99),
        ):
            if key not in self.payload:
                continue
            value = self.payload[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{key} must be in [{minimum}, {maximum}]")
        if self.kind == "rdagent_research":
            normalize_research_schedule_payload(self.payload, max_loops=20)
        elif self.kind == "recommendation_refresh":
            if not self.payload.get("recommendation_portfolio_id"):
                raise ValueError("recommendation_refresh requires recommendation_portfolio_id")
            if self.run_time < time(15, 10):
                raise ValueError("recommendation_refresh must run after the A-share close")
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
        elif self.kind == "auxiliary_data_pipeline":
            unknown = set(self.payload) - {
                "history_start",
                "max_stocks",
                "max_options",
                "strategy_minute_symbols",
                "download_workers",
                "requests_per_minute",
            }
            if unknown:
                raise ValueError(
                    f"auxiliary_data_pipeline contains unsupported keys: {sorted(unknown)}"
                )
            date.fromisoformat(str(self.payload.get("history_start") or "2024-01-01"))
            symbols = self.payload.get("strategy_minute_symbols")
            if not isinstance(symbols, list) or not symbols:
                raise ValueError("auxiliary_data_pipeline requires strategy_minute_symbols")
            for key in ("max_stocks", "max_options"):
                value = self.payload.get(key, 100)
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 500:
                    raise ValueError(f"auxiliary_data_pipeline {key} must be in [1, 500]")
        elif self.kind == "information_pipeline":
            normalize_information_schedule_payload(self.payload)
        elif self.kind == "information_factor_refresh":
            normalize_information_factor_refresh_payload(self.payload)
        elif self.kind in {"weekly_report", "monthly_decision_day", "preopen_check"}:
            # Operational run-calendar kinds carry no profile/bundle payload;
            # an optional Qlib dataset anchor name is the only recognized key.
            unknown = set(self.payload) - {"dataset"}
            if unknown:
                raise ValueError(f"unsupported {self.kind} payload keys: {sorted(unknown)}")
        elif self.kind == "intraday_execution_check":
            unknown = set(self.payload) - {"dataset", "interval_minutes"}
            if unknown:
                raise ValueError(f"unsupported {self.kind} payload keys: {sorted(unknown)}")
            interval = int(self.payload.get("interval_minutes", 5))
            validate_intraday_run_time(self.run_time, interval)
        else:
            profile = self.payload.get("profile", "full")
            if profile not in {"core", "research", "full", "research-assets"}:
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
                "strategy_specialty",
                *DEFAULT_COVERAGE_BUNDLES,
            }
            bundles = self.payload.get("bundles") or sorted(allowed)
            if not isinstance(bundles, list) or not bundles:
                raise ValueError("data_pipeline requires at least one bundle")
            unknown = sorted({str(item) for item in bundles} - allowed)
            if unknown:
                raise ValueError(f"data_pipeline contains unsupported bundles: {unknown}")
            if profile == "research-assets" and set(bundles) != {"research_corpus"}:
                raise ValueError(
                    "research-assets data_pipeline requires exactly the research_corpus bundle"
                )
        return self


class ScheduleStatusRequest(BaseModel):
    status: Literal["active", "paused"]


class DataAutomationUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any]
    reason: str = Field(min_length=10, max_length=1000)

    @model_validator(mode="after")
    def validate_config(self) -> DataAutomationUpdateRequest:
        self.config = normalize_data_automation_config(self.config)
        return self


class DataAutomationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups: list[
        Literal[
            "market_daily",
            "research_assets",
            "information_daily",
            "information_weekly",
            "ashare_5m",
            "auxiliary_daily",
        ]
    ] = Field(default_factory=lambda: ["market_daily"])


class AutopilotUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any]
    reason: str = Field(min_length=10, max_length=500)

    @model_validator(mode="after")
    def validate_config(self) -> AutopilotUpdateRequest:
        self.config = normalize_autopilot_config(self.config)
        return self


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


def create_app(project_root: Path | None = None) -> FastAPI:
    project_root = (project_root or Path.cwd()).resolve()
    settings = Settings.from_env(project_root / ".env")
    settings.data_root.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointStore(settings.database_url)
    checkpoint_catalog = CheckpointCatalogProjection(
        checkpoint,
        ttl_seconds=30.0,
        cache_path=settings.data_root / "platform" / "cache" / "checkpoint-catalog-v1.json",
    )
    platform_root = settings.data_root / "platform"
    jobs = JobStore(settings.database_url)
    data_tasks = DataTaskStore(settings.database_url)
    research = ResearchStore(settings.database_url)
    factor_library = FactorLibraryStore(research.engine)
    rdagent_candidates = RDAGentCandidateStore(settings.database_url)
    research_assets = ResearchAssetStore(settings.database_url)
    research_report_backfill = ResearchReportBackfillStore(settings.database_url)
    strategies = StrategyStore(settings.database_url)
    promotions = PromotionStore(settings.database_url)
    recommendations = RecommendationStore(settings.database_url)
    simulations = SimulationStore(settings.database_url)
    model_artifacts = ModelArtifactStore(settings.database_url)
    strategy_feature_drift = StrategyFeatureDriftSource(
        settings.database_url,
        settings.data_root,
    )
    parameter_experiments = ParameterExperimentStore(settings.database_url)
    legacy_research_campaigns = ResearchCampaignStore(settings.database_url)
    legacy_research_programs = ResearchProgramStore(settings.database_url)
    research_tournaments = ResearchTournamentStore(settings.database_url)
    autopilot_trial_audit = AutopilotTrialAuditService(
        research=research,
        candidates=rdagent_candidates,
        tournaments=research_tournaments,
        parameter_experiments=parameter_experiments,
    )
    allocations = AllocationStore(settings.database_url)
    investor_profiles = InvestorSimulationProfileStore(settings.database_url)
    advice = AdviceService(settings.database_url, data_root=settings.data_root)
    schedules = ScheduleStore(settings.database_url)
    alerts = AlertStore(settings.database_url)
    health_history = OperationalHealthStore(settings)
    safe_mode = SafeModeStore(settings.database_url)
    auth = AuthStore(settings.database_url)
    runtime_secrets = RuntimeSecretStore(settings.database_url, settings.platform_secret_key)
    platform_configs = PlatformConfigStore(settings.database_url)
    autopilot = AutopilotController(settings)
    recommendation_accounts = RecommendationAccountStore(
        settings.database_url,
        configs=platform_configs,
        simulations=simulations,
    )
    retention = DataRetentionManager(settings.data_root, settings.database_url)
    market_dashboard = MarketOverviewService(settings.data_root)
    deployment_readiness = DeploymentReadinessStore(settings, project_root)
    worker = LocalJobWorker(jobs, project_root, settings)
    runtime_status_ttl_seconds = 60.0
    qlib_runtime: dict | None = None
    qlib_runtime_checked_at = 0.0
    qlib_runtime_lock = Lock()
    qlib_runtime_refreshing = False
    rdagent_runtime: dict | None = None
    rdagent_runtime_checked_at = 0.0
    rdagent_runtime_lock = Lock()
    rdagent_runtime_refreshing = False
    data_task_projection_cache_path = (
        settings.data_root / "platform" / "cache" / "data-task-projection-v1.json"
    )
    try:
        cached_data_tasks = json.loads(
            data_task_projection_cache_path.read_text(encoding="utf-8")
        )
        data_task_projection: list[dict[str, Any]] | None = (
            list(cached_data_tasks["tasks"])
            if isinstance(cached_data_tasks, dict)
            and cached_data_tasks.get("version") == 1
            and isinstance(cached_data_tasks.get("tasks"), list)
            and all(isinstance(item, dict) for item in cached_data_tasks["tasks"])
            else None
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        data_task_projection = None
    data_task_projection_checked_at = 0.0
    data_task_projection_lock = Lock()
    data_task_projection_refreshing = False

    def refresh_qlib_runtime_async(*, force: bool = False) -> dict:
        nonlocal qlib_runtime_checked_at, qlib_runtime_refreshing
        launch = False
        with qlib_runtime_lock:
            current_status = str((qlib_runtime or {}).get("status") or "checking")
            retry_seconds = (
                2.0 if current_status in {"checking", "unavailable"}
                else runtime_status_ttl_seconds
            )
            stale = (
                qlib_runtime is None
                or monotonic() - qlib_runtime_checked_at >= retry_seconds
            )
            if (force or stale) and not qlib_runtime_refreshing:
                qlib_runtime_refreshing = True
                launch = True
            current = dict(qlib_runtime or {"status": "checking"})

        if launch:
            def probe() -> None:
                nonlocal qlib_runtime, qlib_runtime_checked_at, qlib_runtime_refreshing
                try:
                    result = probe_qlib(settings, project_root)
                except Exception as exc:
                    result = {"status": "unavailable", "error": str(exc)}
                with qlib_runtime_lock:
                    qlib_runtime = result
                    qlib_runtime_checked_at = monotonic()
                    qlib_runtime_refreshing = False

            try:
                Thread(target=probe, name="qlib-runtime-status", daemon=True).start()
            except Exception:
                with qlib_runtime_lock:
                    qlib_runtime_checked_at = monotonic()
                    qlib_runtime_refreshing = False
        return current

    def refresh_rdagent_runtime_async(*, force: bool = False) -> dict:
        nonlocal rdagent_runtime_checked_at, rdagent_runtime_refreshing
        launch = False
        with rdagent_runtime_lock:
            current_status = str((rdagent_runtime or {}).get("status") or "checking")
            retry_seconds = (
                2.0 if current_status in {"checking", "unavailable"}
                else runtime_status_ttl_seconds
            )
            stale = (
                rdagent_runtime is None
                or monotonic() - rdagent_runtime_checked_at >= retry_seconds
            )
            if (force or stale) and not rdagent_runtime_refreshing:
                rdagent_runtime_refreshing = True
                launch = True
            current = dict(rdagent_runtime or {"status": "checking", "scenarios": []})

        if launch:
            def probe() -> None:
                nonlocal rdagent_runtime, rdagent_runtime_checked_at
                nonlocal rdagent_runtime_refreshing
                try:
                    result = probe_rdagent(settings, project_root)
                except Exception as exc:
                    result = {
                        "status": "unavailable",
                        "ready": False,
                        "error": str(exc),
                        "scenarios": [],
                    }
                with rdagent_runtime_lock:
                    rdagent_runtime = result
                    rdagent_runtime_checked_at = monotonic()
                    rdagent_runtime_refreshing = False

            try:
                Thread(target=probe, name="rdagent-runtime-status", daemon=True).start()
            except Exception:
                with rdagent_runtime_lock:
                    rdagent_runtime_checked_at = monotonic()
                    rdagent_runtime_refreshing = False
        return current

    def current_data_tasks() -> list[dict[str, Any]]:
        """Return the last small operational projection and refresh off-thread.

        The projection groups millions of immutable work-unit rows.  Active
        downloads keep a five-second view; a settled catalog uses 30 seconds.
        HTTP requests never wait for that GROUP BY. Research/admission paths
        never consume this display-only cache.
        """

        nonlocal data_task_projection_checked_at, data_task_projection_refreshing
        now = monotonic()
        launch = False
        with data_task_projection_lock:
            active_projection = bool(
                data_task_projection
                and any(
                    str(item.get("status")) in {"queued", "running"}
                    for item in data_task_projection
                )
            )
            ttl_seconds = 5.0 if active_projection else 30.0
            if (
                data_task_projection is not None
                and now - data_task_projection_checked_at < ttl_seconds
            ):
                return data_task_projection
            if not data_task_projection_refreshing:
                data_task_projection_refreshing = True
                launch = True
            current = list(data_task_projection or [])

        if launch:
            def refresh_projection() -> None:
                nonlocal data_task_projection, data_task_projection_checked_at
                nonlocal data_task_projection_refreshing
                try:
                    refreshed = data_tasks.list()
                    temporary = data_task_projection_cache_path.with_name(
                        f".{data_task_projection_cache_path.name}.{os.getpid()}.tmp"
                    )
                    try:
                        data_task_projection_cache_path.parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        temporary.write_text(
                            json.dumps(
                                {
                                    "version": 1,
                                    "generated_at": datetime.now(UTC).isoformat(),
                                    "tasks": refreshed,
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                                default=str,
                            ),
                            encoding="utf-8",
                        )
                        os.replace(temporary, data_task_projection_cache_path)
                    except OSError:
                        temporary.unlink(missing_ok=True)
                    with data_task_projection_lock:
                        data_task_projection = refreshed
                        data_task_projection_checked_at = monotonic()
                        data_task_projection_refreshing = False
                except Exception:
                    with data_task_projection_lock:
                        data_task_projection_checked_at = monotonic()
                        data_task_projection_refreshing = False

            try:
                Thread(
                    target=refresh_projection,
                    name="data-task-display-projection",
                    daemon=True,
                ).start()
            except Exception:
                with data_task_projection_lock:
                    data_task_projection_checked_at = monotonic()
                    data_task_projection_refreshing = False
        return current

    def reconcile_default_autopilot_program() -> dict[str, Any]:
        """Compatibility helper that now advances only the single AutopilotCycle."""

        return autopilot.tick()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal data_task_projection, data_task_projection_checked_at
        data_tasks.sync_catalog()
        with data_task_projection_lock:
            if data_task_projection is None:
                data_task_projection = data_tasks.catalog_shell()
                # Keep the shell immediately visible, but force the first
                # request/startup warm-up to launch the full aggregate refresh.
                data_task_projection_checked_at = 0.0
        factor_library.sync_builtin_library()
        for sota in factor_library.list_sota(limit=200):
            register_feature_set(factor_library.sota_feature_set(str(sota["id"])))
        try:
            record = platform_configs.get(DATA_AUTOMATION_CONFIG_KEY)
            if record is None:
                platform_configs.put(
                    DATA_AUTOMATION_CONFIG_KEY,
                    normalize_data_automation_config(DEFAULT_DATA_AUTOMATION_CONFIG),
                    actor="platform-bootstrap",
                    reason="Initialize the governed Web data automation control plane",
                )
            config, _ = data_automation_config()
            reconcile_data_automation(config, actor="platform-bootstrap")
            if platform_configs.get(AUTOPILOT_CONFIG_KEY) is None:
                platform_configs.put(
                    AUTOPILOT_CONFIG_KEY,
                    normalize_autopilot_config(DEFAULT_AUTOPILOT_CONFIG),
                    actor="platform-bootstrap",
                    reason="Initialize the governed automatic research control plane",
                )
            reconcile_default_autopilot_program()
        except Exception as exc:
            alerts.create(
                source_type="platform",
                source_id="data-automation",
                severity="critical",
                category="data_automation_reconcile_failed",
                title="数据自动更新计划未能完成对账",
                message=str(exc),
                dedupe_key="platform:data-automation:reconcile-failed",
            )
        if settings.embedded_worker:
            worker.start()
        # Warm display-only runtime snapshots without making the first browser
        # request wait on Docker/worker probes.
        refresh_qlib_runtime_async(force=True)
        refresh_rdagent_runtime_async(force=True)
        current_data_tasks()
        checkpoint_catalog.get()
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
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Content-Type"],
    )

    public_api_paths = {
        "/api/health",
        "/api/readyz",
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

    def current_live_signal_bindings(
        version: dict[str, Any],
        *,
        dataset_identity_sha256: str,
        signal_date: date,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Fail closed until current model/factor inference artifacts exist."""

        signal_source = str(
            version.get("config", {}).get("signal_source") or "factor_score"
        )
        if signal_source == "model_prediction":
            artifact = model_artifacts.require_for_inference(
                str(version["id"]),
                dataset_identity_sha256=dataset_identity_sha256,
                signal_date=signal_date,
            )
            return (
                {
                    "id": str(artifact["id"]),
                    "artifact_sha256": str(artifact["artifact_sha256"]),
                    "checkpoint_sha256": str(artifact["checkpoint_sha256"]),
                    "dataset_identity_sha256": str(
                        artifact["dataset_identity_sha256"]
                    ),
                },
                None,
            )
        if signal_source != "factor_score":
            raise ValueError("strategy signal source is unsupported")
        return (
            None,
            strategy_feature_drift.current_challenger_artifact_binding(
                str(version["id"]),
                current_dataset_identity_sha256=dataset_identity_sha256,
                signal_date=signal_date,
            ),
        )

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

    def data_automation_config() -> tuple[dict[str, Any], dict[str, Any] | None]:
        record = platform_configs.get(DATA_AUTOMATION_CONFIG_KEY)
        try:
            return normalize_data_automation_config(
                record["value"] if record else DEFAULT_DATA_AUTOMATION_CONFIG
            ), record
        except ValueError as exc:
            raise HTTPException(
                503,
                "stored data automation configuration is invalid for this release",
            ) from exc

    def information_factor_evaluation() -> dict[str, Any] | None:
        existing = schedules.get_by_name(MANAGED_SCHEDULE_NAMES["information_weekly"])
        if existing:
            candidate = (existing.get("payload") or {}).get("factor_evaluation")
            if isinstance(candidate, dict):
                try:
                    return normalize_information_factor_refresh_payload(
                        {
                            "sources": sorted(STRUCTURED_INFORMATION_SOURCES),
                            "weekday": 4,
                            "factor_evaluation": candidate,
                        }
                    )["factor_evaluation"]
                except ValueError:
                    pass
        candidates = [
            item
            for item in list_qlib_datasets(settings.data_root)
            if item.get("ready") and item.get("reproducible") and item.get("frequency") == "day"
        ]
        if not candidates:
            return None
        dataset = max(
            candidates,
            key=lambda item: (str(item.get("end_date") or ""), str(item["name"])),
        )
        calendar_path = Path(dataset["path"]) / "calendars" / "day.txt"
        if not calendar_path.is_file():
            return None
        periods, _ = resolve_research_periods(
            calendar_path.read_text(encoding="utf-8").splitlines()
        )
        return {
            "dataset": dataset["name"],
            "periods": periods,
            "universe": "cn_all",
            "benchmark": "SH000300",
        }

    def reconcile_data_automation(
        config: dict[str, Any],
        *,
        actor: str,
    ) -> list[dict[str, Any]]:
        timezone = str(config["timezone"])
        enabled = bool(config["enabled"])
        managed: list[dict[str, Any]] = []
        runtime_limits = {
            "download_workers": config["download_workers"],
            "requests_per_minute": config["requests_per_minute"],
        }

        def put(
            group: str,
            kind: str,
            run_at: str,
            *,
            trading_days_only: bool,
            payload: dict[str, Any],
        ) -> None:
            managed.append(
                schedules.upsert_managed(
                    name=MANAGED_SCHEDULE_NAMES[group],
                    kind=kind,
                    timezone=timezone,
                    run_time=time.fromisoformat(run_at),
                    trading_days_only=trading_days_only,
                    payload=payload,
                    misfire_grace_seconds=3600,
                    actor=actor,
                    enabled=enabled,
                )
            )

        put(
            "market_daily",
            "data_pipeline",
            config["market_daily_time"],
            trading_days_only=True,
            payload={
                "profile": "full",
                "lookback_days": config["market_lookback_days"],
                "snapshot_start": "2008-01-01",
                "bundles": [
                    bundle for bundle in AUTOMATED_DATA_BUNDLES if bundle != "research_corpus"
                ],
                **runtime_limits,
            },
        )
        put(
            "research_assets",
            "data_pipeline",
            config["research_assets_time"],
            trading_days_only=True,
            payload={
                "profile": "research-assets",
                "lookback_days": config["market_lookback_days"],
                "snapshot_start": config["research_assets_history_start"],
                "bundles": ["research_corpus"],
                **runtime_limits,
            },
        )
        put(
            "information_daily",
            "information_pipeline",
            config["information_daily_time"],
            trading_days_only=False,
            payload={
                "lookback_days": config["information_lookback_days"],
                "regulatory_only": True,
                "download_limit": 0,
                "enable_nlp": True,
                "announcement_categories": ["regulatory_letter"],
                "announcement_nlp_limit": 500,
                "include_corpus_nlp": True,
                "corpus_datasets": [
                    "cctv_news",
                    "irm_qa_sh",
                    "irm_qa_sz",
                    "major_news",
                ],
                "corpus_nlp_limit": 500,
                "batch_size": 50,
                "major_news_per_day": 40,
                "irm_per_instrument_day": 2,
                "include_event_labels": True,
                "include_factor_evaluation": False,
                "snapshot_name": "",
                "horizons": [1, 3, 5, 20],
                "benchmark_code": "000300.SH",
            },
        )
        evaluation = information_factor_evaluation()
        if evaluation is not None:
            put(
                "information_weekly",
                "information_factor_refresh",
                config["information_weekly_time"],
                trading_days_only=False,
                payload={
                    "sources": sorted(STRUCTURED_INFORMATION_SOURCES),
                    "weekday": config["information_weekday"],
                    "factor_evaluation": evaluation,
                },
            )
        put(
            "ashare_5m",
            "ashare_5m_sync",
            config["ashare_5m_time"],
            trading_days_only=True,
            payload={
                "history_start": config["ashare_5m_history_start"],
                "lookback_days": 3,
                **runtime_limits,
            },
        )
        put(
            "auxiliary_daily",
            "auxiliary_data_pipeline",
            config["auxiliary_daily_time"],
            trading_days_only=False,
            payload={
                "history_start": config["auxiliary_history_start"],
                "max_stocks": config["max_stocks"],
                "max_options": config["max_options"],
                "strategy_minute_symbols": config["strategy_minute_symbols"],
                **runtime_limits,
            },
        )
        return managed

    def data_automation_state() -> dict[str, Any]:
        config, record = data_automation_config()
        all_schedules = schedules.list(1000)
        managed = [
            item for item in all_schedules if item["name"] in set(MANAGED_SCHEDULE_NAMES.values())
        ]
        return {
            "config": config,
            "source": "database" if record else "built_in",
            "revision": int(record["revision"]) if record else 0,
            "updated_by": record.get("updated_by") if record else None,
            "updated_at": record.get("updated_at") if record else None,
            "coverage": automation_coverage(
                managed,
                enabled=bool(config["enabled"]),
            ),
            "schedules": managed,
            "blocked": (
                []
                if schedules.get_by_name(MANAGED_SCHEDULE_NAMES["information_weekly"])
                else ["weekly information refresh requires a reproducible daily Qlib dataset"]
            ),
        }

    def require_qlib_dataset(name: str, *, purpose: str, frequency: str | None = None) -> dict:
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
        try:
            provenance = dataset.get("provenance") or {}
            if dataset.get("frequency") == "day":
                require_daily_qlib_contract(provenance)
            else:
                require_minute_execution_contract(
                    provenance, frequency=str(dataset.get("frequency") or "")
                )
        except ValueError as exc:
            raise HTTPException(409, f"{purpose}: {exc}") from exc
        return dataset

    def require_native_execution_controls(
        dataset: dict[str, Any], *, start: date, purpose: str
    ) -> None:
        try:
            require_native_daily_execution_controls(dataset.get("provenance") or {}, start=start)
        except ValueError as exc:
            raise HTTPException(409, f"{purpose}: {exc}") from exc

    def require_bound_daily_source(
        *,
        requested_name: str | None,
        start: date,
        end: date,
        purpose: str,
    ) -> tuple[dict[str, Any], str]:
        """Select one verified daily source and return its snapshot lineage."""

        if requested_name:
            candidates = [
                require_qlib_dataset(
                    requested_name,
                    purpose=purpose,
                    frequency="day",
                )
            ]
        else:
            candidates = [
                item
                for item in list_qlib_datasets(settings.data_root)
                if item.get("ready") and item.get("reproducible") and item.get("frequency") == "day"
            ]
        eligible = [
            item
            for item in candidates
            if (not item.get("start_date") or str(item["start_date"]) <= start.isoformat())
            and (not item.get("end_date") or str(item["end_date"]) >= end.isoformat())
        ]
        if not eligible:
            raise HTTPException(
                409,
                f"{purpose} requires a verified daily Qlib dataset covering the request",
            )
        dataset = max(
            eligible,
            key=lambda item: (str(item.get("end_date") or ""), str(item["name"])),
        )
        try:
            require_daily_qlib_contract(dataset.get("provenance") or {})
        except ValueError as exc:
            raise HTTPException(409, f"{purpose}: {exc}") from exc
        source_lineage_id = str(
            (dataset.get("provenance") or {}).get("source_lineage_id") or ""
        ).lower()
        if len(source_lineage_id) != 64 or any(
            character not in "0123456789abcdef" for character in source_lineage_id
        ):
            raise HTTPException(409, f"{purpose} daily source lineage is invalid")
        return dataset, source_lineage_id

    def read_research_calendar(dataset: dict[str, Any]) -> list[str]:
        calendar_path = Path(dataset["path"]) / "calendars" / "day.txt"
        try:
            calendar = [
                date.fromisoformat(line.strip()).isoformat()
                for line in calendar_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, ValueError) as exc:
            raise HTTPException(409, "Qlib trading calendar is missing or invalid") from exc
        if not calendar:
            raise HTTPException(409, "Qlib trading calendar is empty")
        return calendar

    def resolve_dataset_research_periods(
        dataset: dict[str, Any],
        *,
        periods: ResearchPeriods | dict[str, Any] | None,
        period_policy: ResearchPeriodPolicy | dict[str, Any] | None,
        horizon_profile: str | None = None,
        feature_set: dict[str, Any] | None = None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        raw_periods = periods.model_dump(mode="json") if isinstance(periods, BaseModel) else periods
        raw_policy = (
            period_policy.model_dump(mode="json")
            if isinstance(period_policy, BaseModel)
            else period_policy
        )
        try:
            if horizon_profile is None:
                resolved, evidence = resolve_research_periods(
                    read_research_calendar(dataset),
                    periods=raw_periods,
                    period_policy=raw_policy,
                )
            else:
                resolved, evidence = resolve_research_window_contract(
                    dataset,
                    read_research_calendar(dataset),
                    periods=raw_periods,
                    period_policy=raw_policy,
                    horizon_profile=horizon_profile,
                    feature_set=feature_set,
                )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        require_research_calendar(dataset, resolved)
        if dataset.get("start_date") and resolved["train_start"] < dataset["start_date"]:
            raise HTTPException(409, "training window starts before the selected dataset")
        if dataset.get("end_date") and resolved["test_end"] > dataset["end_date"]:
            raise HTTPException(409, "test window ends after the selected dataset")
        evidence["dataset_identity_sha256"] = (dataset.get("provenance") or {}).get(
            "dataset_identity_sha256"
        )
        return resolved, evidence

    def require_research_calendar(dataset: dict, periods: dict[str, str]) -> None:
        calendar = [date.fromisoformat(day) for day in read_research_calendar(dataset)]
        valid_start = date.fromisoformat(periods["valid_start"])
        valid_end = date.fromisoformat(periods["valid_end"])
        test_start = date.fromisoformat(periods["test_start"])
        test_end = date.fromisoformat(periods["test_end"])
        valid_days = sum(valid_start <= day <= valid_end for day in calendar)
        test_days = sum(test_start <= day <= test_end for day in calendar)
        if valid_days < 126 or test_days < 252:
            raise HTTPException(
                409,
                f"Qlib calendar provides {valid_days} validation and {test_days} final-test "
                "trading days; at least 126 and 252 are required",
            )

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

    @app.get("/api/readyz")
    def business_readiness(response: Response) -> dict[str, Any]:
        """Fail closed unless the durable business loop is currently operable."""

        try:
            latest_health = health_history.latest()
        except Exception:  # noqa: BLE001 - readiness must return 503, not leak DB errors
            latest_health = None
        if latest_health is None:
            operational_check: dict[str, Any] = {
                "status": "missing",
                "message": "no durable operational health snapshot",
                "blocking_components": ["operational_health"],
            }
        else:
            components = latest_health.get("components")
            component_map = components if isinstance(components, dict) else {}
            blocking_components = sorted(
                str(name)
                for name, component in component_map.items()
                if not isinstance(component, dict)
                or component.get("status") not in {"ok", "not_applicable"}
            )
            operational_check = {
                "status": str(latest_health.get("status") or "missing"),
                "message": "durable operational health snapshot",
                "recorded_at": latest_health.get("recorded_at"),
                "age_seconds": latest_health.get("age_seconds"),
                "blocking_components": blocking_components,
            }

        stale_after_seconds = max(30, settings.scheduler_poll_seconds * 2)
        max_active_tick_seconds = settings.scheduler_max_tick_seconds
        scheduler_check: dict[str, Any]
        scheduler_identity: dict[str, Any] = {}
        if settings.scheduler_url:
            try:
                scheduler_response = requests.get(
                    f"{settings.scheduler_url}/health",
                    timeout=5,
                )
                scheduler_body = scheduler_response.json()
                if not isinstance(scheduler_body, dict):
                    raise ValueError("scheduler health body must be an object")
                scheduler_identity = {
                    "release_id": scheduler_body.get("release_id"),
                    "config_digest": scheduler_body.get("config_digest"),
                }
                scheduler_check = _scheduler_endpoint_health_check(
                    scheduler_body,
                    response_status_code=scheduler_response.status_code,
                    now=datetime.now(UTC),
                    stale_after_seconds=stale_after_seconds,
                    max_active_tick_seconds=max_active_tick_seconds,
                )
            except (requests.RequestException, TypeError, ValueError):
                scheduler_check = {
                    "status": "unavailable",
                    "message": "scheduler health endpoint is unavailable",
                    "stale_after_seconds": stale_after_seconds,
                    "max_active_tick_seconds": max_active_tick_seconds,
                }
        else:
            raw_components = (
                latest_health.get("components", {}) if isinstance(latest_health, dict) else {}
            )
            component_map = raw_components if isinstance(raw_components, dict) else {}
            scheduler_component = (
                component_map.get("scheduler_heartbeat")
                or component_map.get("scheduler")
                or {}
            )
            if isinstance(scheduler_component, dict):
                scheduler_details = scheduler_component.get("details")
                details = scheduler_details if isinstance(scheduler_details, dict) else {}
                scheduler_identity = {
                    "release_id": scheduler_component.get("release_id")
                    or details.get("release_id"),
                    "config_digest": scheduler_component.get("config_digest")
                    or details.get("config_digest"),
                }
            scheduler_ready = (
                operational_check["status"] == "ok"
                and isinstance(scheduler_component, dict)
                and scheduler_component.get("status") == "ok"
            )
            scheduler_check = {
                "status": "ok" if scheduler_ready else "unavailable",
                "message": (
                    "scheduler proven by fresh durable health snapshot"
                    if scheduler_ready
                    else "scheduler URL is unconfigured and no fresh heartbeat is available"
                ),
                "source": "durable_health_snapshot",
            }

        expected_release = settings.quantlab_release_id
        expected_config = settings.quantlab_config_digest
        observed_release = str(scheduler_identity.get("release_id") or "")
        observed_config = str(scheduler_identity.get("config_digest") or "")
        release_ready = bool(
            expected_release
            and expected_config
            and observed_release == expected_release
            and observed_config == expected_config
        )
        release_check = {
            "status": "ok" if release_ready else "blocked",
            "message": (
                "API and scheduler use the same immutable release and configuration"
                if release_ready
                else "release/config identity is missing or differs across services"
            ),
            "api_release_id": expected_release or None,
            "scheduler_release_id": observed_release or None,
            "api_config_digest": expected_config or None,
            "scheduler_config_digest": observed_config or None,
        }

        try:
            safe_mode_state = safe_mode.status()
            safe_mode_active = bool(safe_mode_state.get("active"))
            safe_mode_check = {
                "status": "blocked" if safe_mode_active else "ok",
                "message": "safe mode is active" if safe_mode_active else "safe mode is off",
                "active": safe_mode_active,
            }
        except Exception:  # noqa: BLE001 - readiness must fail closed on unreadable state
            safe_mode_check = {
                "status": "unavailable",
                "message": "safe-mode state is unavailable",
                "active": None,
            }

        try:
            secret_storage = runtime_secrets.health()
            secret_check = {
                "status": str(secret_storage.get("status") or "unavailable"),
                "message": str(secret_storage.get("message") or "runtime secret storage"),
            }
        except Exception:  # noqa: BLE001 - readiness must fail closed on unreadable state
            secret_check = {
                "status": "unavailable",
                "message": "runtime secret storage is unavailable",
            }

        try:
            business_loop = deployment_readiness.business_loop_readiness()
            business_checks = business_loop.get("checks")
            if not isinstance(business_checks, dict) or not all(
                isinstance(item, dict) for item in business_checks.values()
            ):
                raise ValueError("business readiness checks must be an object")
        except Exception as exc:  # noqa: BLE001 - readiness must fail closed
            business_checks = {
                "business_loop": {
                    "status": "unavailable",
                    "message": (
                        "business-loop readiness is unavailable: " + str(exc)[:400]
                    ),
                }
            }

        checks = {
            "operational_health": operational_check,
            "scheduler": scheduler_check,
            "release_identity": release_check,
            "safe_mode": safe_mode_check,
            "runtime_secret_storage": secret_check,
            **business_checks,
        }
        blockers = [
            {
                "check": name,
                "status": str(check.get("status") or "unavailable"),
                "message": str(check.get("message") or name),
            }
            for name, check in checks.items()
            if not isinstance(check, dict) or check.get("status") != "ok"
        ]
        ready = not blockers
        response.status_code = 200 if ready else 503
        response.headers["Cache-Control"] = "no-store"
        return _sanitize_public_value(
            {
                "status": "ready" if ready else "not_ready",
                "ready": ready,
                "blockers": blockers,
                "checks": checks,
            }
        )

    @app.get("/api/settings")
    def runtime_settings_status() -> dict:
        secret_storage = runtime_secrets.health()
        tushare_record = runtime_secrets.describe("tushare")
        llm_record = runtime_secrets.describe("llm")
        alert_record = runtime_secrets.describe("alert_webhook")
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

    @app.get("/api/data-automation")
    def get_data_automation() -> dict[str, Any]:
        return _sanitize_public_value(data_automation_state())

    @app.put("/api/data-automation")
    def update_data_automation(
        payload: DataAutomationUpdateRequest,
        request: Request,
    ) -> dict[str, Any]:
        actor = authenticated_actor(request, "local-admin")
        try:
            config = normalize_data_automation_config(payload.config)
            platform_configs.put(
                DATA_AUTOMATION_CONFIG_KEY,
                config,
                actor=actor,
                reason=payload.reason,
            )
            reconcile_data_automation(config, actor=actor)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return _sanitize_public_value(data_automation_state())

    @app.get("/api/data-automation/revisions")
    def list_data_automation_revisions(
        limit: int = Query(50, ge=1, le=200),
    ) -> list[dict[str, Any]]:
        return platform_configs.list_revisions(DATA_AUTOMATION_CONFIG_KEY, limit)

    @app.post("/api/data-automation/reconcile")
    def reconcile_data_automation_endpoint(request: Request) -> dict[str, Any]:
        config, _ = data_automation_config()
        try:
            reconcile_data_automation(
                config,
                actor=authenticated_actor(request, "local-admin"),
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return _sanitize_public_value(data_automation_state())

    @app.get("/api/autopilot")
    def get_autopilot() -> dict[str, Any]:
        config, revision = autopilot.config()
        cycles = []
        for stored_cycle in autopilot.store.list_cycles(limit=20):
            cycle = dict(stored_cycle)
            legacy_capital_state = isinstance(
                dict(cycle.get("state") or {}).get("capital_pipeline"), dict
            )
            cycle["authority_scope"] = (
                "legacy_readonly" if legacy_capital_state else "research_only"
            )
            cycle["capital_entry"] = "fin_strategy_settlement"
            cycles.append(cycle)
        # A long-running research cycle remains the operator's current object
        # even if a newer terminal history row exists.  Old capital rows are
        # visible but cannot become the current writable cycle.
        current_cycle = next(
            (
                item
                for item in cycles
                if item["status"] in {"active", "paused"}
                and item["authority_scope"] != "legacy_readonly"
            ),
            cycles[0] if cycles else None,
        )
        current_tournament = None
        if current_cycle is not None:
            try:
                current_tournament = research_tournaments.get_for_cycle(
                    str(current_cycle["id"])
                )
            except KeyError:
                pass
        recorded_stage = str(
            (current_cycle or {}).get("stage") or "waiting_for_data"
        )
        current_stage = (
            "legacy_readonly"
            if (current_cycle or {}).get("authority_scope") == "legacy_readonly"
            else recorded_stage
        )
        next_action = {
            "waiting_for_data": "等待新的已验证日线数据快照",
            "parallel_research": "并行完成因子、模型和研报研究",
            "feature_screen": "用固定 LightGBM 从四套因子库筛选前两套",
            "model_full": "四类 CPU 模型在前两套特征上完成三窗口验证",
            "ensemble": "比较跨家族等权 Rank 集成并冻结模型冠军",
            "joint_optimization": "运行 fin_quant 消融并冻结研究冠军",
            "research_complete": "等待受管 fin_strategy 研究与资本门禁结算",
            "complete": "等待下一研究周期；策略资本入口由 fin_strategy 结算",
            "legacy_readonly": "旧 Autopilot 资本记录仅供查询；新资本工作只走 fin_strategy",
        }.get(current_stage, "处理当前阻断后继续自动驾驶")
        return _sanitize_public_value(
            {
                "config": config,
                "revision": revision,
                "state": (
                    "running"
                    if config["enabled"]
                    and any(
                        item["status"] == "active"
                        and item["authority_scope"] != "legacy_readonly"
                        for item in cycles
                    )
                    else "idle"
                    if config["enabled"]
                    else "paused"
                ),
                "current_cycle": current_cycle,
                "current_stage": current_stage,
                "next_action": next_action,
                "capital_authority": {
                    "automatic_entry": "fin_strategy_settlement",
                    "legacy_autopilot_capital_pipeline": "legacy_readonly",
                    "autopilot_scope": "factor_model_fin_quant_research_and_champion_selection",
                },
                "tournament": current_tournament,
                "cycles": cycles,
                "report_backfill": research_report_backfill.summary(),
                "real_trading": {
                    "connected": False,
                    "automatic": False,
                    "minimum_paper_calendar_days": config["paper_min_calendar_days"],
                    "decision": "manual_only",
                },
            }
        )

    @app.put("/api/autopilot")
    def update_autopilot(
        payload: AutopilotUpdateRequest,
        request: Request,
    ) -> dict[str, Any]:
        actor = authenticated_actor(request, "local-admin")
        try:
            platform_configs.put(
                AUTOPILOT_CONFIG_KEY,
                normalize_autopilot_config(payload.config),
                actor=actor,
                reason=payload.reason,
            )
            if payload.config.get("enabled") is True:
                reconcile_default_autopilot_program()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return get_autopilot()

    @app.get("/api/autopilot/paper-target")
    def get_autopilot_paper_target() -> dict:
        """Read the current isolated paper target; never creates a recommendation."""

        return _sanitize_public_value(simulations.current_autopilot_paper_target())

    @app.post("/api/autopilot/reconcile")
    def reconcile_autopilot(request: Request) -> dict[str, Any]:
        authenticated_actor(request, "local-admin")
        try:
            result = autopilot.tick()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"result": result, "autopilot": get_autopilot()}

    @app.get("/api/autopilot/cycles")
    def list_autopilot_cycles(
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        return _sanitize_public_value(autopilot.store.list_cycles(limit=limit))

    @app.get("/api/autopilot/cycles/{cycle_id}")
    def get_autopilot_cycle(cycle_id: str) -> dict[str, Any]:
        try:
            return _sanitize_public_value(autopilot.store.get_cycle(cycle_id))
        except KeyError as exc:
            raise HTTPException(404, "autopilot cycle not found") from exc

    @app.get("/api/autopilot/cycles/{cycle_id}/trials")
    def get_autopilot_cycle_trials(cycle_id: str) -> list[dict[str, Any]]:
        try:
            cycle = autopilot.store.get_cycle(cycle_id)
        except KeyError as exc:
            raise HTTPException(404, "autopilot cycle not found") from exc
        # This is one read-only audit view over every existing governed ledger.
        # A daily cycle does not need a model tournament to expose factor,
        # fin_quant, failure, or portfolio records.
        return _sanitize_public_value(autopilot_trial_audit.list_cycle_trials(cycle))

    @app.get("/api/model-ensembles")
    def list_model_ensembles(
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        return _sanitize_public_value(research_tournaments.list_ensembles(limit=limit))

    @app.get("/api/model-ensembles/{ensemble_id}")
    def get_model_ensemble(ensemble_id: str) -> dict[str, Any]:
        try:
            return _sanitize_public_value(research_tournaments.get_ensemble(ensemble_id))
        except KeyError as exc:
            raise HTTPException(404, "model ensemble not found") from exc

    @app.post("/api/data-automation/run", status_code=202)
    def run_data_automation(
        payload: DataAutomationRunRequest,
        request: Request,
    ) -> dict[str, Any]:
        actor = authenticated_actor(request, "local-operator")
        runs: list[dict[str, Any]] = []
        for group in dict.fromkeys(payload.groups):
            schedule = schedules.get_by_name(MANAGED_SCHEDULE_NAMES[group])
            if schedule is None:
                raise HTTPException(409, f"managed schedule {group} is not ready")
            runs.append(schedules.trigger_now(str(schedule["id"]), actor=actor))
        return {"status": "queued", "runs": runs}

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

    @app.get("/api/overview")
    def overview() -> dict:
        catalog = checkpoint_catalog.get()
        return _sanitize_public_value(
            system_summary(
                effective_settings(),
                checkpoint,
                jobs.status_summaries(statuses=("queued", "running")),
                current_data_tasks(),
                catalog=catalog,
            )
        )

    @app.get("/api/datasets")
    def datasets() -> list[dict]:
        return _sanitize_public_value(checkpoint_catalog.get())

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
        return _sanitize_public_value(list_snapshots_for_display(settings.data_root))

    @app.get("/api/data-retention")
    def data_retention_plan(
        keep_latest: int = Query(7, ge=1, le=100),
        min_age_days: int = Query(14, ge=1, le=3650),
    ) -> dict:
        return retention.display_plan(keep_latest=keep_latest, min_age_days=min_age_days)

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
        result = refresh_qlib_runtime_async(force=refresh)
        return _sanitize_public_value(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "qlib_version",
                    "lightgbm_version",
                    "version",
                    "commit",
                    "execution_environment",
                )
                if key in result
            }
        )

    @app.get("/api/qlib/datasets")
    def qlib_datasets() -> list[dict]:
        # Browser inventory is a persisted stale-while-refresh projection.  It
        # must never walk millions of feature files on the request path.  All
        # formal research and capital gates still call list_qlib_datasets.
        return _sanitize_public_value(list_qlib_datasets_for_display(settings.data_root))

    @app.get("/api/qlib/experiments")
    def qlib_experiments() -> list[dict]:
        recent = jobs.list(
            100,
            statuses=("succeeded",),
            kinds=("qlib_baseline", "minute_research"),
            payload_keys=(),
            progress_keys=(),
        )
        return _sanitize_public_value(
            list_qlib_experiments(
                settings.data_root,
                job_ids=tuple(str(item["id"]) for item in recent),
                limit=100,
            )
        )

    @app.get("/api/rdagent/status")
    def rdagent_status(refresh: bool = False) -> dict:
        return _public_rdagent_status(refresh_rdagent_runtime_async(force=refresh))

    @app.get("/api/rdagent/scenarios")
    def rdagent_scenarios(refresh: bool = False) -> list[dict[str, Any]]:
        # Public status is already sanitized, but the catalog remains complete.
        status = rdagent_status(refresh=refresh)
        scenarios = status.get("scenarios")
        return scenarios if isinstance(scenarios, list) else []

    @app.get("/api/rdagent/feature-sets")
    def rdagent_feature_sets() -> list[dict[str, Any]]:
        for sota in factor_library.list_sota(limit=100):
            register_feature_set(factor_library.sota_feature_set(str(sota["id"])))
        return list_feature_sets()

    @app.get("/api/factor-library")
    def factor_library_definitions(
        family: str | None = None,
        source: str | None = None,
        status: str | None = None,
        dataset: str | None = None,
        limit: int = Query(1000, ge=1, le=2000),
    ) -> list[dict[str, Any]]:
        try:
            dataset_record = (
                require_qlib_dataset(dataset, purpose="factor library inspection", frequency="day")
                if dataset
                else None
            )
            available_fields = (
                set((dataset_record["provenance"].get("field_units") or {}).keys())
                if dataset_record
                else None
            )
            definitions = factor_library.list_definitions(
                family=family,
                source=source,
                status=status,
                available_fields=available_fields,
                limit=limit,
            )
            if dataset_record is not None:
                identity = str(dataset_record["provenance"]["dataset_identity_sha256"])
                root = (
                    settings.data_root / "artifacts" / "factor-library-materializations" / identity
                )
                manifests = sorted(
                    root.glob("*/manifest.json"),
                    key=lambda path: path.stat().st_mtime_ns,
                    reverse=True,
                )
                if manifests:
                    materialized = json.loads(manifests[0].read_text(encoding="utf-8"))
                    cluster_path = manifests[0].parent / "clusters" / "result.json"
                    assignments = (
                        json.loads(cluster_path.read_text(encoding="utf-8")).get("assignments")
                        if cluster_path.is_file()
                        else {}
                    ) or {}
                    completed = materialized.get("completed") or {}
                    blocked = materialized.get("blocked") or {}
                    for definition in definitions:
                        if definition["id"] in completed:
                            definition["calculability"] = "materialized"
                            definition["materialization"] = completed[definition["id"]]
                        elif definition["id"] in blocked:
                            definition["calculability"] = "blocked"
                            definition["materialization_blocker"] = blocked[definition["id"]]
                        definition["similarity_cluster_id"] = assignments.get(definition["id"])
            return definitions
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/factor-library/versions")
    def factor_library_version_list() -> list[dict[str, Any]]:
        return factor_library.list_library_versions()

    @app.get("/api/factor-library/materializations")
    def factor_library_materializations(
        limit: int = Query(50, ge=1, le=200),
    ) -> list[dict[str, Any]]:
        return [
            _public_job(item)
            for item in jobs.list(
                limit=limit,
                kinds=("factor_library_materialize", "factor_library_cluster"),
                payload_keys=tuple(sorted(_PUBLIC_JOB_PAYLOAD_KEYS)),
                progress_keys=tuple(sorted(_PUBLIC_JOB_PROGRESS_KEYS)),
            )
        ]

    @app.post("/api/factor-library/materializations", status_code=202)
    def create_factor_library_materialization(
        payload: FactorLibraryMaterializationRequest,
    ) -> dict[str, Any]:
        dataset = require_qlib_dataset(
            payload.dataset, purpose="factor library materialization", frequency="day"
        )
        feature_set = get_feature_set(payload.feature_set_id)
        start = payload.start or date.fromisoformat(str(dataset["start_date"]))
        end = payload.end or date.fromisoformat(str(dataset["end_date"]))
        if start < date.fromisoformat(str(dataset["start_date"])) or end > date.fromisoformat(
            str(dataset["end_date"])
        ):
            raise HTTPException(409, "factor materialization exceeds the sealed dataset")
        identity = str(dataset["provenance"]["dataset_identity_sha256"])
        job = jobs.create(
            "factor_library_materialize",
            {
                "dataset": payload.dataset,
                "dataset_path": dataset["path"],
                "dataset_identity_sha256": identity,
                "feature_set_id": feature_set["id"],
                "feature_set_definition_sha256": feature_set["definition_sha256"],
                "library_version_id": (
                    feature_set["source"]
                    if str(feature_set.get("source") or "").startswith("unified-factor-library")
                    else None
                ),
                "universe": payload.universe,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            platform_root / "logs" / f"factor-library-{identity[:12]}.log",
            idempotency_key=(
                f"factor-library:{identity}:{feature_set['definition_sha256']}:"
                f"{payload.universe}:{start.isoformat()}:{end.isoformat()}"
            ),
            max_attempts=2,
        )
        worker.notify()
        return job

    @app.get("/api/research-sota")
    def research_sota_list(
        limit: int = Query(50, ge=1, le=200),
    ) -> list[dict[str, Any]]:
        return factor_library.list_sota(limit=limit)

    @app.get("/api/research-sota/{version_id}")
    def research_sota_detail(version_id: str) -> dict[str, Any]:
        try:
            return factor_library.get_sota(version_id)
        except KeyError as exc:
            raise HTTPException(404, "research SOTA version not found") from exc

    @app.get("/api/rdagent/assets")
    def rdagent_research_assets(
        limit: int = Query(200, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        return research_assets.list_assets(limit=limit)

    @app.get("/api/rdagent/assets/acquisitions")
    def rdagent_research_asset_acquisitions(
        limit: int = Query(50, ge=1, le=200),
    ) -> list[dict[str, Any]]:
        acquisition_progress_keys = (
            "arxiv_selected",
            "assets",
            "blocked",
            "daily_limits",
            "failed",
            "mode",
            "published",
            "status",
            "tushare_selected",
        )
        return [
            _public_research_asset_job(job)
            for job in jobs.list(
                limit=limit,
                kinds=("research_asset_acquire",),
                payload_keys=("mode",),
                progress_keys=acquisition_progress_keys,
            )
        ]

    @app.get("/api/research-report-backfill")
    def research_report_backfill_status(
        limit: int = Query(500, ge=1, le=2000),
    ) -> dict[str, Any]:
        return {
            "policy": {
                "start": "2023-08-25",
                "selection_limit_per_report_date": 20,
                "max_attempts_per_day": 100,
                "max_bytes_per_day": 5 * 1024**3,
                "concurrency": 2,
                "minimum_free_bytes": 300 * 1024**3,
                "order": "newest_first",
                "live_reports_have_priority": True,
            },
            "summary": research_report_backfill.summary(),
            "days": research_report_backfill.list(limit=limit),
        }

    @app.post("/api/rdagent/assets/acquisitions/automatic", status_code=202)
    def acquire_automatic_research_assets(
        payload: ResearchAssetAutomaticRequest,
        request: Request,
    ) -> dict[str, Any]:
        research_day = (
            payload.as_of or datetime.now(UTC).astimezone(ZoneInfo("Asia/Shanghai")).date()
        )
        if research_day > datetime.now(UTC).astimezone(ZoneInfo("Asia/Shanghai")).date():
            raise HTTPException(422, "research asset as_of cannot be in the future")
        try:
            snapshot_name = (
                latest_verified_research_asset_snapshot(settings.data_root, as_of=research_day)
                if payload.include_tushare
                else "arxiv-only"
            )
            actor = authenticated_actor(request, payload.actor)
            idempotency_key = research_asset_acquisition_idempotency_key(
                research_day=research_day.isoformat(),
                snapshot_name=snapshot_name,
                include_tushare=payload.include_tushare,
                include_arxiv=payload.include_arxiv,
            )
            job = jobs.create(
                "research_asset_acquire",
                {
                    "mode": "automatic",
                    "snapshot_name": snapshot_name,
                    "as_of": research_day.isoformat(),
                    "include_tushare": payload.include_tushare,
                    "include_arxiv": payload.include_arxiv,
                    "requested_by": actor,
                },
                platform_root / "logs" / f"research-assets-{research_day.isoformat()}.log",
                dedupe_active_kind=False,
                idempotency_key=idempotency_key,
                max_attempts=3,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return _public_research_asset_job(job)

    @app.post("/api/rdagent/assets/acquisitions/manual-https", status_code=202)
    def acquire_manual_research_asset(
        payload: ResearchAssetManualHttpsRequest,
        request: Request,
    ) -> dict[str, Any]:
        actor = authenticated_actor(request, payload.actor)
        source_identity = hashlib.sha256(payload.url.encode("utf-8")).hexdigest()
        try:
            job = jobs.create(
                "research_asset_acquire",
                {
                    "mode": "manual_https",
                    "url": payload.url,
                    "title": payload.title,
                    "document_kind": payload.document_kind,
                    "published_at": (
                        payload.published_at.isoformat() if payload.published_at else None
                    ),
                    "requested_by": actor,
                },
                platform_root / "logs" / f"research-asset-manual-{source_identity}.log",
                dedupe_active_kind=False,
                idempotency_key=f"research-assets:manual:{source_identity}",
                max_attempts=3,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return _public_research_asset_job(job)

    @app.get("/api/rdagent/admitted-signals")
    def rdagent_admitted_signals(
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        return rdagent_candidates.list_admitted_strategy_signals(limit=limit)

    @app.post("/api/rdagent/health-check")
    def rdagent_health_check() -> dict[str, Any]:
        try:
            if settings.rdagent_worker_url:
                response = requests.post(
                    f"{settings.rdagent_worker_url}/rdagent/health-check",
                    timeout=190,
                )
                response.raise_for_status()
                result = response.json()
            else:
                llm = runtime_secrets.get("llm")
                runtime_env = None
                if llm:
                    runtime_env = {
                        settings.rdagent_llm_key_env: llm["api_key"],
                        "OPENAI_API_BASE": llm.get("api_base", ""),
                        "CHAT_MODEL": llm.get("chat_model", "gpt-4.1-mini"),
                    }
                result = run_official_rdagent_health_check(
                    settings, project_root, runtime_env=runtime_env
                )
        except (requests.RequestException, TypeError, ValueError):
            result = {
                "status": "failed",
                "diagnostic_only": True,
                "platform_readiness_unchanged": True,
                "output": "official RD-Agent diagnostic is unavailable",
            }
        if not isinstance(result, dict):
            raise HTTPException(502, "official RD-Agent diagnostic returned invalid data")
        return _sanitize_public_value(result)

    @app.get("/api/rdagent/runs")
    def list_research_runs(limit: int = Query(50, ge=1, le=200)) -> list[dict]:
        return [_public_rdagent_run(item) for item in research.list_runs(limit)]

    @app.get("/api/rdagent/runs/{run_id}")
    def get_research_run(run_id: str) -> dict:
        try:
            run = research.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(404, "research run not found") from exc
        run["candidates"] = research.list_candidates(run_id=run_id)
        run["events"] = research.list_events(run_id)
        audit = rdagent_candidates.run_audit_summary(run_id)
        run.update(audit)
        run["trace_view"] = _rdagent_trace_view(
            rdagent_candidates,
            research_run_id=run_id,
            scenario_id=scenario_from_research_run(run),
            run_artifacts=list(audit.get("run_artifacts") or []),
        )
        run["asset_consumptions"] = research_assets.list_consumptions(research_run_id=run_id)
        return _public_rdagent_run(run)

    @app.post("/api/rdagent/runs/{run_id}/model-validation", status_code=202)
    def validate_general_model_implementation(
        run_id: str,
        payload: GeneralModelValidationRequest,
        request: Request,
    ) -> dict[str, Any]:
        """Send one paper-derived implementation through the fin_model gate.

        This endpoint deliberately creates only a research candidate.  It does
        not create a StrategyVersion or ModelArtifact and it never opens the
        sealed final OOS window.
        """

        try:
            run = research.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(404, "research run not found") from exc
        if scenario_from_research_run(run) != "general_model":
            raise HTTPException(409, "only general_model implementations may use this gate")
        if run.get("status") != "succeeded":
            raise HTTPException(409, "general_model must finish before validation")
        try:
            artifact = rdagent_candidates.get_run_artifact(payload.artifact_id, verify=True)
        except KeyError as exc:
            raise HTTPException(404, "implementation artifact not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if (
            artifact.get("research_run_id") != run_id
            or artifact.get("artifact_type") != "general_model_implementation_code"
            or artifact.get("status") != "recorded"
        ):
            raise HTTPException(
                409,
                "artifact is not a recorded implementation from this general_model run",
            )

        dataset = require_qlib_dataset(
            payload.dataset,
            purpose="general_model independent validation",
            frequency="day",
        )
        periods, resolution = resolve_dataset_research_periods(
            dataset,
            periods=None,
            period_policy=payload.period_policy,
        )
        run_config = dict(run.get("config") or {})
        try:
            resolve_rdagent_assets(
                settings,
                get_rdagent_scenario("general_model"),
                list(run_config.get("asset_ids") or []),
                expected_manifest_sha256=dict(run_config.get("asset_manifest_sha256") or {}),
                pre_final_end=date.fromisoformat(periods["valid_end"]),
            )
        except ValueError as exc:
            raise HTTPException(
                409,
                "paper asset is not point-in-time eligible for this validation window: " + str(exc),
            ) from exc
        feature_set = get_feature_set(payload.feature_set_id)
        metadata = dict((artifact.get("manifest_json") or {}).get("metadata") or {})
        actor = authenticated_actor(request, payload.requested_by)
        try:
            candidate = rdagent_candidates.create_model_candidate(
                research_run_id=run_id,
                name=str(payload.name or metadata.get("name") or "paper-model"),
                description=str(
                    payload.description
                    or metadata.get("description")
                    or "Paper-derived model awaiting independent Qlib validation"
                ),
                model_type=payload.model_type,
                code_artifact_id=payload.artifact_id,
                architecture=payload.architecture,
                model_hyperparameters=payload.model_hyperparameters,
                training_hyperparameters=payload.training_hyperparameters,
                feature_set_id=feature_set["id"],
                dataset=payload.dataset,
                dataset_identity_sha256=str(dataset["provenance"]["dataset_identity_sha256"]),
                dataset_lineage_id=(
                    str(dataset["lineage_id"]) if dataset.get("lineage_id") else None
                ),
                pre_final_end=date.fromisoformat(periods["valid_end"]),
                final_oos_start=date.fromisoformat(periods["test_start"]),
                final_oos_end=date.fromisoformat(periods["test_end"]),
                source_iteration=artifact.get("source_iteration"),
                rdagent_decision=None,
                rdagent_feedback="general_model implementation submitted to fin_model gate",
            )
            for asset_id in run.get("config", {}).get("asset_ids") or []:
                rdagent_candidates.link_asset(
                    asset_id=str(asset_id),
                    candidate_kind="model",
                    candidate_id=str(candidate["id"]),
                    relationship="implemented_from_paper",
                    actor=actor,
                )
            evaluation_job = jobs.create(
                "model_evaluate",
                {
                    "research_run_id": run_id,
                    "dataset": payload.dataset,
                    "dataset_path": dataset["path"],
                    "dataset_identity_sha256": dataset["provenance"]["dataset_identity_sha256"],
                    "evaluation_profiles": resolution["evaluation_profiles"],
                    "feature_set_id": feature_set["id"],
                    "feature_set_definition_sha256": feature_set["definition_sha256"],
                    "candidates": [
                        {
                            "id": candidate["id"],
                            "code_path": artifact["storage_path"],
                            "code_sha256": artifact["content_sha256"],
                            "model_type": payload.model_type,
                            "training_hyperparameters": payload.training_hyperparameters,
                        }
                    ],
                    "universe": "cn_all",
                    "benchmark": "SH000300",
                },
                platform_root / "logs" / f"model-evaluate-{run_id}-{candidate['id']}.log",
                idempotency_key=(
                    f"general-model-evaluate:{run_id}:{payload.artifact_id}:"
                    f"{dataset['provenance']['dataset_identity_sha256']}:"
                    f"{feature_set['definition_sha256']}"
                ),
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        research.attach_job(run_id, evaluation_job["id"])
        research.mark_run(run_id, "evaluating", actor=actor)
        worker.notify()
        return {
            "status": "queued_for_independent_model_validation",
            "research_run_id": run_id,
            "model_candidate_id": candidate["id"],
            "job_id": evaluation_job["id"],
            "final_oos_opened": False,
            "capital_eligible": False,
        }

    @app.post("/api/rdagent/runs", status_code=202)
    def create_research_run(payload: RDAgentRunRequest, request: Request) -> dict:
        runtime = probe_rdagent(settings, project_root)
        try:
            scenario = require_ready_scenario(runtime, settings, payload.scenario)
        except ValueError as exc:
            raise HTTPException(
                409,
                {"message": f"RD-Agent {payload.scenario} is not ready", "blockers": [str(exc)]},
            ) from exc
        if payload.loop_n > settings.rdagent_max_loops:
            raise HTTPException(
                422, f"loop_n exceeds configured limit {settings.rdagent_max_loops}"
            )
        try:
            duration = validate_duration_limit(payload.duration, settings.rdagent_max_duration)
            expected_runtime_identity = expected_rdagent_runtime_identity(runtime, scenario.id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        research_horizon_profile = (
            {
                "short": SHORT_1_5D,
                "swing": SWING_1_6M,
                "long": LONG_1_3Y,
            }[str(payload.horizon)]
            if scenario.id in HORIZON_RESEARCH_SCENARIOS
            else None
        )
        strategy_horizon_profile = (
            research_horizon_profile if scenario.id == "fin_strategy" else None
        )
        incumbent_binding: dict[str, Any] | None = None
        if payload.incumbent_strategy_version_id is not None:
            try:
                incumbent = strategies.get_version(payload.incumbent_strategy_version_id)
            except KeyError as exc:
                raise HTTPException(409, "incumbent strategy version was not found") from exc
            if incumbent.get("horizon_profile") != research_horizon_profile:
                raise HTTPException(
                    409,
                    "incumbent strategy version belongs to a different horizon",
                )
            if incumbent.get("status") != "approved" or incumbent.get(
                "promotion_stage"
            ) not in {"paper", "recommendation_enabled"}:
                raise HTTPException(
                    409,
                    "incumbent strategy version is not an active governed strategy",
                )
            incumbent_binding = {
                "id": str(incumbent["id"]),
                "strategy_id": str(incumbent["strategy_id"]),
                "version": int(incumbent["version"]),
                "status": str(incumbent["status"]),
                "promotion_stage": incumbent.get("promotion_stage"),
                "horizon_profile": str(incumbent["horizon_profile"]),
                "horizon_contract_sha256": str(
                    incumbent["horizon_contract_sha256"]
                ),
                "strategy_rules_sha256": incumbent.get("strategy_rules_sha256"),
            }
        dataset: dict[str, Any] | None = None
        periods: dict[str, str] | None = None
        period_resolution: dict[str, Any] | None = None
        auto_selected_assets = not payload.asset_ids and scenario.auto_select_assets
        try:
            if scenario.requires_dataset:
                dataset = require_qlib_dataset(
                    str(payload.dataset), purpose=f"RD-Agent {scenario.id}", frequency="day"
                )
                periods, period_resolution = resolve_dataset_research_periods(
                    dataset,
                    periods=None,
                    period_policy=payload.period_policy,
                    horizon_profile=research_horizon_profile,
                    feature_set=(
                        get_feature_set(str(payload.feature_set_id))
                        if scenario.requires_feature_set
                        else None
                    ),
                )
            assets = resolve_rdagent_assets(
                settings,
                scenario,
                payload.asset_ids,
                excluded_auto_asset_ids=(
                    research_assets.unavailable_asset_ids() if auto_selected_assets else frozenset()
                ),
                pre_final_end=(
                    date.fromisoformat(periods["valid_end"]) if periods is not None else None
                ),
                selection_limit=(payload.loop_n if scenario.id == "fin_factor_report" else None),
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        resolved_asset_ids = list(assets["manifest_sha256"])
        actor = authenticated_actor(request, payload.requested_by)
        if auto_selected_assets:
            try:
                for asset_id in resolved_asset_ids:
                    rdagent_candidates.import_manifest(
                        settings.data_root
                        / "artifacts"
                        / "research-assets"
                        / asset_id
                        / "manifest.json",
                        actor=actor,
                    )
            except (OSError, ValueError) as exc:
                raise HTTPException(409, str(exc)) from exc
        feature_set = (
            get_feature_set(str(payload.feature_set_id)) if scenario.requires_feature_set else None
        )
        strategy_research_signal_binding: dict[str, Any] | None = None
        if scenario.id == "fin_strategy":
            if dataset is None or feature_set is None or research_horizon_profile is None:
                raise HTTPException(409, "fin_strategy governed inputs are incomplete")
            try:
                champion_selection = (
                    autopilot.champion_selector.select_champion(
                        dataset=str(dataset["name"]),
                        dataset_identity_sha256=str(
                            dataset["provenance"]["dataset_identity_sha256"]
                        ),
                        horizon_profile=research_horizon_profile,
                    )
                )
            except ValueError as exc:
                if str(exc) != (
                    "no independently admitted signal matches this dataset identity"
                ):
                    raise HTTPException(
                        409,
                        f"governed fin_strategy champion selection failed: {exc}",
                    ) from exc
                champion_selection = None
            try:
                feature_set = research_feature_set_for_champion_selection(
                    feature_set,
                    champion_selection,
                )
                # The selected factor champion owns a different immutable
                # expression grid than the public recipe. Re-resolve the
                # research contract before launching RD-Agent so its YAML,
                # rule allowlist and later score materialization all bind the
                # same effective features.
                periods, period_resolution = resolve_dataset_research_periods(
                    dataset,
                    periods=None,
                    period_policy=payload.period_policy,
                    horizon_profile=research_horizon_profile,
                    feature_set=feature_set,
                )
                strategy_research_signal_binding = (
                    build_strategy_research_signal_binding(
                        horizon_profile=research_horizon_profile,
                        dataset=str(dataset["name"]),
                        dataset_identity_sha256=str(
                            dataset["provenance"]["dataset_identity_sha256"]
                        ),
                        research_feature_set=feature_set,
                        champion_selection=champion_selection,
                    )
                )
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
        artifact = settings.data_root / "artifacts" / "rdagent"
        config: dict[str, Any] = {
            "scenario": scenario.id,
            "asset_ids": resolved_asset_ids,
            "asset_manifest_sha256": assets["manifest_sha256"],
            "asset_selection_mode": "automatic" if auto_selected_assets else "explicit",
            "feature_set": feature_set,
            "expected_rdagent_runtime": expected_runtime_identity,
            "strategy_horizon_profile": strategy_horizon_profile,
            "horizon_profile": research_horizon_profile,
            "incumbent_strategy": incumbent_binding,
            "strategy_research_signal_binding": strategy_research_signal_binding,
            **(
                {
                    "primary_label_policy": primary_label_policy_contract(),
                    "primary_label_policy_sha256": primary_label_policy_contract()[
                        "policy_sha256"
                    ],
                }
                if scenario.id in HORIZON_RESEARCH_SCENARIOS
                else {}
            ),
        }
        if dataset is not None and periods is not None and period_resolution is not None:
            config.update(
                {
                    "periods": periods,
                    "evaluation_profiles": period_resolution["evaluation_profiles"],
                    "period_resolution": period_resolution,
                    "research_window_contract": period_resolution.get(
                        "research_window_contract"
                    ),
                    "research_window_contract_sha256": period_resolution.get(
                        "research_window_contract_sha256"
                    ),
                    "label_horizon_sessions": (
                        primary_label_horizon_sessions(research_horizon_profile)
                        if scenario.id in HORIZON_RESEARCH_SCENARIOS
                        else None
                    ),
                    "dataset_path": dataset["path"],
                }
            )
        try:
            run = research.create_run(
                kind=scenario.research_kind,
                objective=payload.objective,
                dataset=str(payload.dataset or f"lab:{scenario.id}"),
                requested_by=actor,
                budget={"loop_n": payload.loop_n, "duration": duration},
                config=config,
                artifact_path=artifact,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if auto_selected_assets:
            try:
                research_assets.reserve_automatic(
                    research_run_id=run["id"],
                    scenario=scenario.id,
                    asset_manifest_sha256=assets["manifest_sha256"],
                    actor=actor,
                )
            except ValueError as exc:
                research.mark_run(run["id"], "failed", actor="api", error=str(exc))
                raise HTTPException(409, str(exc)) from exc
        log_path = platform_root / "logs" / f"rdagent-{scenario.id}-{run['id']}.log"
        job_payload: dict[str, Any] = {
            "scenario": scenario.id,
            "research_run_id": run["id"],
            "dataset": payload.dataset,
            "dataset_path": dataset["path"] if dataset else None,
            "dataset_identity_sha256": (
                dataset["provenance"]["dataset_identity_sha256"] if dataset else None
            ),
            "dataset_lineage_id": (dataset.get("lineage_id") if dataset else None),
            "objective": payload.objective,
            "loop_n": payload.loop_n,
            "duration": duration,
            "periods": periods,
            "evaluation_profiles": (
                period_resolution["evaluation_profiles"] if period_resolution else []
            ),
            "period_resolution": period_resolution,
            "research_window_contract": (
                period_resolution.get("research_window_contract")
                if period_resolution
                else None
            ),
            "research_window_contract_sha256": (
                period_resolution.get("research_window_contract_sha256")
                if period_resolution
                else None
            ),
            "asset_ids": resolved_asset_ids,
            "asset_manifest_sha256": assets["manifest_sha256"],
            "feature_set": feature_set,
            "expected_rdagent_runtime": expected_runtime_identity,
            "strategy_horizon_profile": strategy_horizon_profile,
            "horizon_profile": research_horizon_profile,
            "label_horizon_sessions": (
                primary_label_horizon_sessions(research_horizon_profile)
                if period_resolution
                and scenario.id in HORIZON_RESEARCH_SCENARIOS
                else None
            ),
            "incumbent_strategy": incumbent_binding,
            "strategy_research_signal_binding": strategy_research_signal_binding,
            **(
                {
                    "primary_label_policy": primary_label_policy_contract(),
                    "primary_label_policy_sha256": primary_label_policy_contract()[
                        "policy_sha256"
                    ],
                }
                if scenario.id in HORIZON_RESEARCH_SCENARIOS
                else {}
            ),
        }
        try:
            job = jobs.create(
                scenario.job_kind,
                job_payload,
                log_path,
                dedupe_active_kind=False,
            )
        except ValueError as exc:
            research.mark_run(run["id"], "failed", actor="api", error=str(exc))
            raise HTTPException(409, str(exc)) from exc
        research.attach_job(run["id"], job["id"])
        worker.notify()
        return _public_rdagent_run(research.get_run(run["id"]))

    @app.get("/api/research-programs")
    def list_research_programs(
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        return _sanitize_public_value(legacy_research_programs.list(limit=limit))

    @app.get("/api/research-programs/{program_id}")
    def get_research_program(program_id: str) -> dict[str, Any]:
        try:
            return _sanitize_public_value(legacy_research_programs.get(program_id))
        except KeyError as exc:
            raise HTTPException(404, "research program not found") from exc

    @app.post("/api/research-programs")
    def create_research_program(request: Request) -> dict[str, Any]:
        del request
        raise HTTPException(
            410,
            "legacy research programs are retired; use /api/autopilot",
        )

    @app.post("/api/research-programs/{program_id}/status")
    def set_research_program_status(
        program_id: str,
        request: Request,
    ) -> dict[str, Any]:
        del program_id, request
        raise HTTPException(410, "legacy research programs are read-only")

    @app.post("/api/research-programs/{program_id}/check-now")
    def check_research_program_now(program_id: str, request: Request) -> dict[str, Any]:
        del program_id, request
        raise HTTPException(410, "legacy research programs are read-only")

    @app.get("/api/research-campaigns")
    def list_research_campaigns(
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        return _sanitize_public_value(legacy_research_campaigns.list(limit=limit))

    @app.get("/api/research-campaigns/{campaign_id}")
    def get_research_campaign(campaign_id: str) -> dict[str, Any]:
        try:
            return _sanitize_public_value(legacy_research_campaigns.get(campaign_id))
        except KeyError as exc:
            raise HTTPException(404, "research campaign not found") from exc

    @app.post("/api/research-campaigns")
    def create_research_campaign(request: Request) -> dict[str, Any]:
        del request
        raise HTTPException(
            410,
            "legacy research campaigns are retired; use /api/autopilot",
        )

    @app.post("/api/research-campaigns/{campaign_id}/status")
    def set_research_campaign_status(
        campaign_id: str,
        request: Request,
    ) -> dict[str, Any]:
        del campaign_id, request
        raise HTTPException(410, "legacy research campaigns are read-only")

    @app.post("/api/research-campaigns/{campaign_id}/retry")
    def retry_research_campaign(campaign_id: str, request: Request) -> dict[str, Any]:
        del campaign_id, request
        raise HTTPException(410, "legacy research campaigns are read-only")

    @app.get("/api/factors")
    def list_factors(
        status: str | None = None,
        run_id: str | None = None,
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict]:
        candidates = research.list_candidates(run_id=run_id, status=status, limit=limit)
        sota_members: dict[str, dict[str, Any]] = {}
        for version in factor_library.list_sota(limit=100):
            if version["status"] != "active":
                continue
            detail = factor_library.get_sota(str(version["id"]))
            for member in detail["members"]:
                sota_members[str(member["factor_candidate_id"])] = {
                    "version_id": version["id"],
                    "status": version["status"],
                    "member_rank": member["member_rank"],
                    "incremental_evidence": member["incremental_evidence"],
                }
        for candidate in candidates:
            candidate["research_sota"] = sota_members.get(str(candidate["id"]))
        return candidates

    @app.get("/api/factors/gate-policy")
    def factor_gate_policy() -> dict:
        return research.policy_summary()

    @app.post("/api/jobs/external-factor-evaluate", status_code=202)
    def create_external_factor_evaluation(
        payload: ExternalFactorEvaluationRequest,
    ) -> dict:
        if jobs.count(
            statuses=("queued", "running"),
            kinds=("external_factor_evaluate", "information_factor_evaluate"),
        ):
            raise HTTPException(409, "an external factor evaluation is already active")
        dataset = require_qlib_dataset(
            payload.dataset, purpose="external factor evaluation", frequency="day"
        )
        periods = payload.periods.model_dump(mode="json")
        require_research_calendar(dataset, periods)
        if dataset.get("start_date") and periods["train_start"] < dataset["start_date"]:
            raise HTTPException(409, "training window starts before the selected dataset")
        if dataset.get("end_date") and periods["test_end"] > dataset["end_date"]:
            raise HTTPException(409, "test window ends after the selected dataset")

        candidates: list[dict[str, Any]] = []
        for candidate_id in payload.candidate_ids:
            try:
                candidate = research.get_candidate(candidate_id)
            except KeyError as exc:
                raise HTTPException(404, f"factor candidate not found: {candidate_id}") from exc
            if candidate.get("status") in {"promoted", "retired"}:
                raise HTTPException(
                    409,
                    f"candidate {candidate_id} cannot be evaluated in "
                    f"{candidate.get('status')} state",
                )
            variables = candidate.get("variables") or {}
            source = variables.get("source") if isinstance(variables, dict) else None
            if not isinstance(source, dict) or not str(source.get("dataset") or "").strip():
                raise HTTPException(
                    409, f"candidate {candidate_id} is not a governed external factor"
                )
            required = ("code_path", "values_path", "code_sha256", "values_sha256")
            if any(not candidate.get(key) for key in required):
                raise HTTPException(
                    409, f"candidate {candidate_id} is missing immutable factor artifacts"
                )
            if (
                not Path(str(candidate["code_path"])).is_file()
                or not Path(str(candidate["values_path"])).is_file()
            ):
                raise HTTPException(
                    409, f"candidate {candidate_id} factor artifacts are unavailable"
                )
            label_horizon_days = int(candidate.get("label_horizon_days") or 1)
            embargo_days = max(5, label_horizon_days)
            if (payload.periods.test_start - payload.periods.valid_end).days <= embargo_days:
                raise HTTPException(
                    409,
                    f"candidate {candidate_id} requires a purge/embargo gap greater "
                    f"than {embargo_days} days",
                )
            candidates.append(
                {
                    "id": candidate_id,
                    "values_path": candidate["values_path"],
                    "code_sha256": candidate["code_sha256"],
                    "values_sha256": candidate["values_sha256"],
                    "experiment_family_id": candidate.get("experiment_family_id"),
                    "experiment_count": int(candidate.get("experiment_count") or 1),
                    "label_horizon_days": label_horizon_days,
                }
            )
        serialized = {
            "dataset": payload.dataset,
            "dataset_path": dataset["path"],
            "dataset_identity_sha256": dataset["provenance"]["dataset_identity_sha256"],
            "periods": periods,
            "universe": payload.universe,
            "benchmark": payload.benchmark,
            "candidates": candidates,
        }
        identity = json.dumps(serialized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        log_path = platform_root / "logs" / "external-factor-evaluate.log"
        try:
            job = jobs.create(
                "external_factor_evaluate",
                serialized,
                log_path,
                idempotency_key=(
                    f"external-factor-evaluate:{uuid.uuid5(uuid.NAMESPACE_URL, identity)}"
                ),
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/factors/{candidate_id}/evaluations", status_code=201)
    def record_factor_evaluation(candidate_id: str, payload: FactorEvaluationRequest) -> dict:
        dataset = require_qlib_dataset(
            payload.dataset, purpose="factor evaluation", frequency="day"
        )
        try:
            return research.record_evaluation(
                candidate_id,
                dataset=payload.dataset,
                dataset_identity_sha256=dataset["provenance"]["dataset_identity_sha256"],
                **payload.periods.model_dump(),
                metrics=payload.metrics,
                artifact_path=payload.artifact_path,
                recomputed_values_path=payload.recomputed_values_path,
                recomputed_values_sha256=payload.recomputed_values_sha256,
                recompute_evidence=payload.recompute_evidence,
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
                economic_hypothesis_group=payload.economic_hypothesis_group,
                hypothesis_group_cap=payload.hypothesis_group_cap,
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
        del payload, request
        raise HTTPException(
            410,
            "pair strategy writes are retired; Autopilot is long-only",
        )

    @app.get("/api/pair-strategies")
    def list_pair_strategies(
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict]:
        return strategies.list_pairs(limit)

    @app.post("/api/pair-strategies/{strategy_id}/versions", status_code=201)
    def create_pair_strategy_version(
        strategy_id: str, payload: PairStrategyVersionCreateRequest, request: Request
    ) -> dict:
        del strategy_id, payload, request
        raise HTTPException(
            410,
            "pair strategy writes are retired; Autopilot is long-only",
        )

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

    @app.post("/api/strategy-versions/{version_id}/parameter-experiments", status_code=202)
    def create_parameter_experiment(
        version_id: str, payload: ParameterExperimentRequest, request: Request
    ) -> dict:
        dataset = require_qlib_dataset(
            payload.dataset, purpose="parameter experiment", frequency="day"
        )
        require_native_execution_controls(
            dataset, start=payload.start, purpose="parameter experiment"
        )
        if dataset.get("start_date") and payload.start.isoformat() < dataset["start_date"]:
            raise HTTPException(409, "experiment starts before the selected dataset")
        if dataset.get("end_date") and payload.end.isoformat() > dataset["end_date"]:
            raise HTTPException(409, "experiment ends after the selected dataset")
        try:
            version = strategies.get_version(version_id)
            if version.get("strategy_type") != "multifactor":
                raise ValueError("parameter experiments require a multifactor strategy")
            execution_method = str(version.get("config", {}).get("execution_method", "open"))
            execution_dataset: dict[str, Any] | None = None
            if execution_method in {"twap", "vwap", "next_bar"}:
                if not payload.execution_dataset:
                    raise ValueError(
                        "minute parameter experiments require a minute execution dataset"
                    )
                execution_dataset = require_qlib_dataset(
                    payload.execution_dataset,
                    purpose="parameter experiment minute execution",
                )
                if execution_dataset.get("frequency") != version["config"].get(
                    "execution_frequency"
                ):
                    raise ValueError(
                        "parameter experiment execution dataset frequency does not match "
                        "the immutable strategy contract"
                    )
                if execution_dataset.get("frequency") not in NATIVE_MINUTE_FREQUENCIES:
                    raise ValueError(
                        "parameter experiment execution requires native 1/5-minute data"
                    )
            elif payload.execution_dataset:
                raise ValueError(
                    "daily-open parameter experiments must not specify minute execution data"
                )
            parameter_grid, trial_parameters = normalize_parameter_grid(
                payload.parameter_grid, max_trials=payload.max_trials
            )
            trial_configs = []
            for parameters in trial_parameters:
                config = _rebind_strategy_execution_contract(
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
            job_payload = bind_transparent_baseline_job_identity(
                config=dict(version.get("config") or {}),
                job_payload={
                    "parameter_experiment_id": experiment["id"],
                    "strategy_version_id": version_id,
                    "dataset": payload.dataset,
                    "dataset_path": dataset["path"],
                    "execution_dataset": execution_dataset,
                },
            )
            job = jobs.create(
                "parameter_experiment",
                job_payload,
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
        require_native_execution_controls(dataset, start=payload.start, purpose="strategy backtest")
        if dataset.get("start_date") and payload.start.isoformat() < dataset["start_date"]:
            raise HTTPException(409, "backtest starts before the selected dataset")
        if dataset.get("end_date") and payload.end.isoformat() > dataset["end_date"]:
            raise HTTPException(409, "backtest ends after the selected dataset")
        artifact_root = settings.data_root / "artifacts" / "backtests"
        execution_dataset: dict | None = None
        try:
            version = strategies.get_version(version_id)
            if version.get("strategy_type") != "multifactor":
                raise ValueError("pair strategy versions require the pair-backtests endpoint")
            execution_method = str(version.get("config", {}).get("execution_method", "open"))
            if execution_method in {"twap", "vwap", "next_bar"}:
                if not payload.execution_dataset:
                    raise ValueError("minute strategy backtests require a minute execution dataset")
                execution_dataset = require_qlib_dataset(
                    payload.execution_dataset, purpose="strategy minute execution"
                )
                frequency = str(execution_dataset.get("frequency") or "")
                if frequency not in NATIVE_MINUTE_FREQUENCIES:
                    raise ValueError("strategy execution requires a native 1/5-minute Qlib dataset")
                execution_start = str(execution_dataset.get("start_date") or "")[:10]
                execution_end = str(execution_dataset.get("end_date") or "")[:10]
                if execution_start and payload.start.isoformat() < execution_start:
                    raise ValueError("backtest starts before the minute execution dataset")
                if execution_end and payload.end.isoformat() > execution_end:
                    raise ValueError("backtest ends after the minute execution dataset")
                bar_minutes = int(frequency.removesuffix("min"))
                slice_minutes = int(version.get("config", {}).get("execution_slice_minutes", 20))
                if bar_minutes > slice_minutes or slice_minutes % bar_minutes:
                    raise ValueError(
                        "execution_slice_minutes must be an integer multiple of the "
                        "minute dataset bar"
                    )
            elif payload.execution_dataset:
                raise ValueError(
                    "open execution backtests must not specify a minute execution dataset"
                )
            backtest_periods = {
                "start": payload.start.isoformat(),
                "end": payload.end.isoformat(),
            }
            trading_dates = load_calendar_days(dataset["path"])
            if (
                version["config"].get("factor_source_mode") == FACTOR_SOURCE_QLIB_BASELINE
                and not version["factors"]
            ):
                dataset_start = str(dataset.get("start_date") or "")[:10]
                if not dataset_start:
                    raise ValueError("baseline final tests require a dated daily dataset")
                embargo_days = int(version["config"].get("outer_embargo_days", 5))
                dates_before_final = sorted(
                    value for value in trading_dates if value < payload.start
                )
                if len(dates_before_final) <= embargo_days:
                    raise ValueError(
                        "baseline dataset has too little history before the final-test embargo"
                    )
                backtest_periods.update(
                    {
                        "historical_start": dataset_start,
                        "historical_end": dates_before_final[-(embargo_days + 1)].isoformat(),
                    }
                )
            backtest = strategies.create_backtest(
                version_id=version_id,
                dataset=payload.dataset,
                execution_dataset=payload.execution_dataset,
                periods=backtest_periods,
                artifact_path=artifact_root,
                trading_dates=trading_dates,
                dataset_lineage_id=dataset.get("lineage_id"),
                dataset_identity_sha256=(dataset.get("provenance") or {}).get(
                    "dataset_identity_sha256"
                ),
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy version not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        log_path = platform_root / "logs" / f"strategy-backtest-{backtest['id']}.log"
        try:
            job = jobs.create(
                "strategy_backtest",
                {
                    "backtest_id": backtest["id"],
                    "strategy_version_id": version_id,
                    "dataset": payload.dataset,
                    "dataset_path": dataset["path"],
                    "execution_dataset": execution_dataset,
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

    @app.get("/api/strategy-versions/{version_id}/model-artifacts")
    def list_model_artifacts(version_id: str) -> list[dict[str, Any]]:
        try:
            return model_artifacts.list_for_strategy(version_id)
        except KeyError as exc:
            raise HTTPException(404, "strategy version not found") from exc

    @app.post(
        "/api/strategy-versions/{version_id}/model-artifacts",
        status_code=201,
    )
    def create_model_artifact(
        version_id: str,
        payload: ModelArtifactCreateRequest,
        request: Request,
    ) -> dict[str, Any]:
        try:
            return model_artifacts.create_from_formal_backtest(
                strategy_version_id=version_id,
                source_backtest_id=payload.source_backtest_id,
                valid_until=payload.valid_until,
                actor=authenticated_actor(request, payload.actor),
                backtests_root=settings.data_root / "artifacts" / "backtests",
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy version not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/model-artifacts/{artifact_id}/activate")
    def activate_model_artifact(
        artifact_id: str,
        payload: ModelArtifactActivateRequest,
        request: Request,
    ) -> dict[str, Any]:
        try:
            return model_artifacts.activate(
                artifact_id,
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "model artifact not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/strategy-versions/{version_id}/model-refits", status_code=202)
    def create_model_refit(
        version_id: str,
        payload: ModelRefitRequest,
        request: Request,
    ) -> dict[str, Any]:
        try:
            version = strategies.get_version(version_id)
            model_signal = version.get("model_signal")
            if (
                version.get("status") != "approved"
                or version.get("is_legacy")
                or not isinstance(model_signal, dict)
            ):
                raise ValueError(
                    "model refit requires an approved governed model-prediction StrategySpec"
                )
            lineage_id = str(model_signal.get("dataset_lineage_id") or "")
            if len(lineage_id) != 64:
                raise ValueError("model refit requires a verified governed dataset lineage")
            source = model_artifacts.select_for_inference(version_id)
            if source.get("selection_status") != "active":
                raise ValueError("model refit requires an active source ModelArtifact")
            dataset = select_qlib_dataset(
                settings.data_root,
                anchor_name=str(source["dataset"]),
                roll_policy="latest_compatible",
                lineage_id=lineage_id,
                required_date=payload.signal_date,
            )
            actor = authenticated_actor(request, payload.actor)
        except KeyError as exc:
            raise HTTPException(404, "strategy version not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        job = jobs.create(
            "model_refit",
            {
                "strategy_version_id": version_id,
                "source_model_artifact_id": source["id"],
                "dataset": dataset["name"],
                "dataset_path": dataset["path"],
                "dataset_identity_sha256": dataset["provenance"]["dataset_identity_sha256"],
                "dataset_lineage_id": lineage_id,
                "signal_date": payload.signal_date.isoformat(),
                "operation": payload.operation,
                "retrain_reason": payload.retrain_reason or "",
                "retrain_evidence": payload.retrain_evidence,
                "retrain_evidence_sha256": (
                    canonical_sha256(payload.retrain_evidence)
                    if payload.retrain_evidence is not None
                    else ""
                ),
                "valid_for_days": payload.valid_for_days,
                "actor": actor,
            },
            platform_root / "logs" / f"model-refit-{version_id}-{payload.signal_date}.log",
            dedupe_active_kind=False,
            idempotency_key=f"model-refit:{version_id}:{payload.signal_date}",
        )
        worker.notify()
        return job

    @app.post("/api/strategy-versions/{version_id}/pair-backtests", status_code=202)
    def create_pair_strategy_backtest(
        version_id: str, payload: PairStrategyBacktestRequest
    ) -> dict:
        del version_id, payload
        raise HTTPException(
            410,
            "pair backtests are retired; Autopilot is long-only",
        )

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

    @app.get("/api/strategy-versions/{version_id}/promotion")
    def get_strategy_promotion(version_id: str) -> dict[str, Any]:
        try:
            strategies.get_version(version_id)
            return {
                "stage": promotions.current_stage(version_id),
                "forward_gate": promotions.evaluate_forward_gate(version_id),
            }
        except KeyError as exc:
            raise HTTPException(404, "strategy version not found") from exc

    @app.post("/api/strategy-versions/{version_id}/paper-stage")
    def open_strategy_paper_stage(
        version_id: str,
        payload: PaperStageOpenRequest,
        request: Request,
    ) -> dict[str, Any]:
        """Recover the safe post-approval transition without changing its gate."""

        try:
            return promotions.prepare_paper_stage(
                version_id,
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy version not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/strategy-versions/{version_id}/promote")
    def promote_strategy_recommendations(
        version_id: str,
        payload: StrategyPromotionRequest,
        request: Request,
    ) -> dict[str, Any]:
        """Recovery action using the same immutable gate as auto-promotion."""

        try:
            return promotions.promote(
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
            item["automation"] = schedules.get_recommendation_allocation_group_optional(
                str(item["id"])
            )
        return items

    @app.post("/api/strategy-allocations", status_code=201)
    def create_strategy_allocation(
        payload: StrategyAllocationCreateRequest,
        request: Request,
    ) -> dict:
        require_qlib_dataset(payload.dataset, purpose="strategy allocation", frequency="day")
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
                member_specs=[item.model_dump() for item in payload.members],
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy version not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/strategy-allocations/{allocation_id}")
    def get_strategy_allocation(allocation_id: str) -> dict:
        try:
            result = allocations.get(allocation_id)
            result["automation"] = schedules.get_recommendation_allocation_group_optional(
                allocation_id
            )
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
            return schedules.create_recommendation_allocation_group(
                allocation_id,
                timezone=payload.timezone,
                run_time=payload.run_time,
                trading_days_only=payload.trading_days_only,
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
            return schedules.set_recommendation_allocation_group_status(
                allocation_id, payload.status
            )
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
            return schedules.set_recommendation_allocation_group_status(allocation_id, "retired")
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
    def refresh_strategy_allocation(allocation_id: str, request: Request) -> dict:
        try:
            return allocations.refresh(
                allocation_id,
                actor=authenticated_actor(request),
            )
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
            result = allocations.set_status(
                allocation_id,
                payload.status,
                actor=authenticated_actor(request, payload.actor),
            )
            automation = schedules.get_recommendation_allocation_group_optional(allocation_id)
            if automation and automation["status"] != "retired":
                schedules.set_recommendation_allocation_group_status(allocation_id, payload.status)
            return result
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

    @app.get("/api/investor-profile")
    def get_investor_profile() -> dict[str, Any]:
        active = investor_profiles.get_active("primary")
        return {
            "configured": active is not None,
            "profile": active,
            "required_before_simulation": active is None,
        }

    @app.put("/api/investor-profile")
    def put_investor_profile(
        payload: InvestorSimulationProfileRequest, request: Request
    ) -> dict[str, Any]:
        try:
            profile = investor_profiles.create_version(
                profile_key="primary",
                initial_capital=payload.initial_capital,
                risk_profile=payload.risk_profile,
                min_cash_weight=payload.min_cash_weight,
                max_gross_exposure=payload.max_gross_exposure,
                market_permissions=payload.market_permissions.model_dump(),
                actor=authenticated_actor(request, payload.actor),
                activate=True,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"configured": True, "profile": profile}

    @app.get("/api/advice/today")
    def get_today_advice() -> dict[str, Any]:
        return advice.today(
            investor_profile=investor_profiles.get_active("primary"),
            platform_safe_mode=safe_mode.status(),
        )

    @app.api_route("/api/portfolios", methods=["GET", "POST"], status_code=410)
    @app.api_route("/api/portfolios/{legacy_path:path}", methods=["GET", "POST"], status_code=410)
    def legacy_portfolios_retired(legacy_path: str = "") -> dict[str, str]:
        return {"status": "retired", "replacement": "/api/recommendation-portfolios"}

    @app.get("/api/recommendation-portfolios")
    def list_recommendation_portfolios(
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict]:
        return recommendations.list(limit)

    @app.post("/api/recommendation-portfolios", status_code=201)
    def create_recommendation_portfolio(
        payload: RecommendationPortfolioCreateRequest, request: Request
    ) -> dict:
        dataset = require_qlib_dataset(
            payload.dataset, purpose="recommendation portfolio", frequency="day"
        )
        lineage_id = str(dataset.get("lineage_id") or "")
        if payload.dataset_roll_policy == "latest_compatible" and len(lineage_id) != 64:
            raise HTTPException(
                409,
                "latest-compatible recommendation requires a verified dataset lineage",
            )
        try:
            return recommendations.create(
                name=payload.name,
                strategy_version_id=payload.strategy_version_id,
                dataset=payload.dataset,
                hypothetical_initial_value=payload.construction_notional,
                actor=authenticated_actor(request, payload.actor),
                dataset_roll_policy=payload.dataset_roll_policy,
                dataset_lineage_id=lineage_id or None,
            )
        except KeyError as exc:
            raise HTTPException(404, "strategy version not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/recommendation-portfolios/{portfolio_id}")
    def get_recommendation_portfolio(portfolio_id: str) -> dict:
        try:
            return recommendations.get(portfolio_id)
        except KeyError as exc:
            raise HTTPException(404, "recommendation portfolio not found") from exc

    @app.get("/api/recommendation-accounts/active")
    def get_active_recommendation_account(
        recommendation_portfolio_id: str = Query(min_length=1, max_length=200),
    ) -> dict:
        try:
            recommendations.get(recommendation_portfolio_id)
            return recommendation_accounts.resolve(recommendation_portfolio_id)
        except KeyError as exc:
            raise HTTPException(404, "recommendation portfolio not found") from exc

    @app.put("/api/recommendation-accounts/active")
    def select_active_recommendation_account(
        payload: ActiveRecommendationAccountRequest,
        request: Request,
    ) -> dict:
        try:
            recommendations.get(payload.recommendation_portfolio_id)
            return recommendation_accounts.select(
                recommendation_portfolio_id=payload.recommendation_portfolio_id,
                account_type=payload.account_type,
                account_id=payload.account_id,
                actor=authenticated_actor(request, "local-admin"),
                reason=payload.reason,
            )
        except KeyError as exc:
            raise HTTPException(404, "recommendation portfolio or account not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/recommendation-portfolios/{portfolio_id}/snapshots/{snapshot_id}")
    def get_recommendation_snapshot(portfolio_id: str, snapshot_id: str) -> dict:
        try:
            snapshot = recommendations.get_snapshot(snapshot_id)
        except KeyError as exc:
            raise HTTPException(404, "recommendation snapshot not found") from exc
        if snapshot["portfolio_id"] != portfolio_id:
            raise HTTPException(404, "recommendation snapshot not found")
        return snapshot

    @app.get("/api/recommendation-portfolios/{portfolio_id}/holdings")
    def get_recommendation_holdings(portfolio_id: str) -> list[dict]:
        try:
            portfolio = recommendations.get(portfolio_id)
        except KeyError as exc:
            raise HTTPException(404, "recommendation portfolio not found") from exc
        latest = portfolio.get("latest_snapshot") or {}
        return list(latest.get("holdings") or [])

    @app.get("/api/recommendation-portfolios/{portfolio_id}/hypothetical-performance")
    def get_recommendation_hypothetical_performance(portfolio_id: str) -> list[dict]:
        try:
            recommendations.get(portfolio_id)
        except KeyError as exc:
            raise HTTPException(404, "recommendation portfolio not found") from exc
        raise HTTPException(
            410,
            "hypothetical recommendation performance is retired; use simulation NAV",
        )

    @app.post("/api/recommendation-portfolios/{portfolio_id}/refresh", status_code=202)
    def refresh_recommendation_portfolio(
        portfolio_id: str, payload: RecommendationRefreshRequest
    ) -> dict:
        try:
            portfolio = recommendations.get(portfolio_id)
            version = strategies.get_version(
                str(portfolio["strategy_version_id"])
            )
            dataset = select_qlib_dataset(
                settings.data_root,
                anchor_name=portfolio["dataset"],
                roll_policy=str(portfolio.get("dataset_roll_policy") or "pinned"),
                lineage_id=portfolio.get("dataset_lineage_id"),
                required_date=payload.as_of_date,
            )
            local_today = datetime.now(UTC).astimezone(ZoneInfo("Asia/Shanghai")).date()
            current_available_date = qlib_trading_date_on_or_before(dataset, local_today)
            if payload.as_of_date != current_available_date:
                raise ValueError(
                    "recommendation signal must use the latest currently available "
                    f"governed trading day {current_available_date.isoformat()}"
                )
            promotions.require_recommendation_signal(
                str(portfolio["strategy_version_id"]),
                signal_date=payload.as_of_date,
            )
            gate = evaluate_recommendation_gate(
                simulations,
                portfolio,
                payload.as_of_date,
                load_calendar_days(dataset["path"]),
            )
            if not gate["passed"]:
                raise ValueError(
                    "recommendation reconciliation gate blocked: " + "; ".join(gate["reasons"])
                )
            dataset_identity_sha256 = str(
                dict(dataset.get("provenance") or {}).get(
                    "dataset_identity_sha256"
                )
                or ""
            )
            model_artifact_binding, factor_materialization_binding = (
                current_live_signal_bindings(
                    version,
                    dataset_identity_sha256=dataset_identity_sha256,
                    signal_date=payload.as_of_date,
                )
            )
            snapshot, created = recommendations.create_snapshot(
                portfolio_id=portfolio_id,
                as_of_date=payload.as_of_date,
                dataset=dataset["name"],
                dataset_identity_sha256=dataset_identity_sha256,
                dataset_lineage_id=dataset.get("lineage_id"),
            )
        except KeyError as exc:
            raise HTTPException(404, "recommendation portfolio not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if not created:
            return snapshot
        job = jobs.create(
            "recommendation_refresh",
            recommendation_refresh_job_payload(
                snapshot,
                dataset,
                model_artifact_binding=model_artifact_binding,
                factor_materialization_binding=factor_materialization_binding,
            ),
            platform_root / "logs" / f"recommendation-refresh-{snapshot['id']}.log",
            dedupe_active_kind=False,
            idempotency_key=recommendation_refresh_job_idempotency_key(
                str(snapshot["id"])
            ),
        )
        recommendations.attach_job(snapshot["id"], job["id"])
        worker.notify()
        return recommendations.get_snapshot(snapshot["id"])

    @app.get("/api/simulation-portfolios")
    def list_simulation_portfolios(
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict]:
        return simulations.list(limit)

    @app.post("/api/simulation-portfolios", status_code=201)
    def create_simulation_portfolio(
        payload: SimulationPortfolioCreateRequest, request: Request
    ) -> dict:
        if payload.execution_adapter == "pair":
            raise HTTPException(
                410,
                "pair simulation writes are retired; historical pair ledgers are read-only",
            )
        source_type = payload.source_type or "recommendation"
        source_id = payload.source_id or payload.recommendation_portfolio_id or ""
        try:
            if source_type == "recommendation":
                source = recommendations.get(source_id)
                daily_dataset_name = str(source["dataset"])
            elif source_type == "strategy_version":
                version = strategies.get_version(source_id)
                if version["status"] != "approved" or version.get("is_legacy"):
                    raise ValueError("simulation requires an approved non-legacy strategy version")
                formal = next(
                    (
                        item
                        for item in strategies.list_backtests(source_id)
                        if item["status"] == "succeeded" and not item.get("is_legacy")
                    ),
                    None,
                )
                if formal is None:
                    raise ValueError(
                        "strategy simulation requires a successful formal Qlib backtest"
                    )
                daily_dataset_name = str(formal["dataset"])
            else:
                source = allocations.get(source_id)
                daily_dataset_name = str(source["dataset"])
        except KeyError as exc:
            raise HTTPException(404, "simulation source not found") from exc
        except ValueError as exc:
            if "pair simulation writes are retired" in str(exc):
                raise HTTPException(410, str(exc)) from exc
            raise HTTPException(409, str(exc)) from exc
        daily = require_qlib_dataset(
            daily_dataset_name, purpose="simulation daily data", frequency="day"
        )
        execution = require_qlib_dataset(
            payload.execution_dataset or daily_dataset_name,
            purpose="simulation execution data",
            frequency=payload.execution_frequency,
        )
        try:
            return simulations.create(
                name=payload.name,
                recommendation_portfolio_id=payload.recommendation_portfolio_id,
                source_type=source_type,
                source_id=source_id,
                daily_dataset=daily,
                execution_dataset=execution,
                initial_cash=payload.initial_cash,
                execution_policy={
                    key: value
                    for key, value in {
                        "execution_algorithm": payload.execution_algorithm,
                        "execution_frequency": payload.execution_frequency,
                        "slice_minutes": payload.slice_minutes,
                        "max_slices": payload.max_slices,
                        "max_participation": payload.max_participation,
                    }.items()
                    if value is not None
                },
                cost_schedule_version=payload.cost_schedule_version,
                actor=authenticated_actor(request, payload.actor),
                execution_adapter=payload.execution_adapter,
                execution_contract_hash=payload.execution_contract_hash,
                daily_roll_policy=payload.daily_roll_policy,
                execution_roll_policy=payload.execution_roll_policy,
            )
        except KeyError as exc:
            raise HTTPException(404, "simulation source not found") from exc
        except ValueError as exc:
            if "pair simulation writes are retired" in str(exc):
                raise HTTPException(410, str(exc)) from exc
            raise HTTPException(409, str(exc)) from exc

    @app.post(
        "/api/simulation-portfolios/{portfolio_id}/order-plans",
        status_code=202,
    )
    def generate_simulation_order_plan(
        portfolio_id: str,
        payload: SimulationOrderPlanGenerationRequest,
        request: Request,
    ) -> dict:
        try:
            portfolio = simulations.get(portfolio_id)
            if portfolio["status"] != "active":
                raise ValueError("simulation portfolio is not active")
            if (
                portfolio["source_type"] != "strategy_version"
                or portfolio["execution_adapter"] != "long_only"
            ):
                raise ValueError(
                    "Qlib order-plan generation requires an active long-only "
                    "strategy-version simulation"
                )
            version = strategies.get_version(portfolio["source_id"])
            if str(version.get("signal_frequency") or "day") != "day" and payload.signal_at is None:
                raise ValueError("minute strategy order-plan generation requires signal_at")
            local_today = datetime.now(UTC).astimezone(ZoneInfo("Asia/Shanghai")).date()
            datasets = {item["name"]: item for item in list_qlib_datasets(settings.data_root)}
            anchor = datasets.get(str(portfolio["daily_dataset"]))
            if anchor is None:
                raise ValueError("paper simulation anchor dataset is unavailable")
            anchor_date = qlib_trading_date_on_or_before(anchor, local_today)
            current_dataset = select_qlib_dataset(
                settings.data_root,
                anchor_name=str(portfolio["daily_dataset"]),
                roll_policy=str(portfolio.get("daily_roll_policy") or "pinned"),
                lineage_id=portfolio.get("daily_dataset_lineage_id"),
                required_date=anchor_date,
            )
            current_available_date = qlib_trading_date_on_or_before(current_dataset, local_today)
            if payload.signal_date != current_available_date:
                raise ValueError(
                    "paper signal must use the latest currently available governed "
                    f"trading day {current_available_date.isoformat()}"
                )
            current_provenance = dict(current_dataset.get("provenance") or {})
            dataset_identity_sha256 = str(
                current_provenance.get("dataset_identity_sha256") or ""
            )
            model_artifact_binding, factor_materialization_binding = (
                current_live_signal_bindings(
                    version,
                    dataset_identity_sha256=dataset_identity_sha256,
                    signal_date=payload.signal_date,
                )
            )
            promotion_stage = promotions.require_paper_signal(
                str(version["id"]),
                portfolio_id=portfolio_id,
                signal_date=payload.signal_date,
            )
            simulations.require_order_plan_predecessor_settled(
                portfolio_id,
                signal_date=payload.signal_date,
            )
        except KeyError as exc:
            raise HTTPException(404, "simulation portfolio or strategy source not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        actor = authenticated_actor(request, payload.actor)
        signal_identity = (
            payload.signal_at.isoformat()
            if payload.signal_at is not None
            else payload.signal_date.isoformat()
        )
        job = jobs.create(
            "simulation_order_plan",
            {
                "simulation_portfolio_id": portfolio_id,
                "signal_date": payload.signal_date.isoformat(),
                "signal_at": (
                    payload.signal_at.isoformat() if payload.signal_at is not None else None
                ),
                "promotion_stage_id": promotion_stage["id"],
                "promotion_stage_opened_at": promotion_stage["opened_at"],
                "dataset_identity_sha256": dataset_identity_sha256,
                "model_artifact_binding": model_artifact_binding,
                "factor_materialization_binding": factor_materialization_binding,
                "actor": actor,
            },
            platform_root / "logs" / f"simulation-order-plan-{portfolio_id}.log",
            dedupe_active_kind=False,
            idempotency_key=(f"simulation-order-plan:{portfolio_id}:{signal_identity}"),
        )
        worker.notify()
        return job

    @app.post("/api/simulation-portfolios/{portfolio_id}/batches", status_code=202)
    def create_simulation_target_batch(
        portfolio_id: str,
        payload: SimulationOrderPlanBatchRequest,
        request: Request,
    ) -> dict:
        try:
            batch, created = simulations.create_batch_from_order_plan(
                portfolio_id,
                order_plan_manifest_sha256=payload.order_plan_manifest_sha256,
                data_root=settings.data_root,
                actor=authenticated_actor(request, payload.actor),
            )
            portfolio = simulations.get(portfolio_id)
        except KeyError as exc:
            raise HTTPException(404, "simulation portfolio not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if created and portfolio["execution_adapter"] == "long_only":
            jobs.create(
                "simulation_replay",
                {"simulation_batch_id": batch["id"]},
                platform_root / "logs" / f"simulation-replay-{batch['id']}.log",
                dedupe_active_kind=False,
                idempotency_key=f"simulation-replay:{batch['id']}",
            )
            worker.notify()
        return batch

    @app.post(
        "/api/simulation-portfolios/{portfolio_id}/pair-replays",
        status_code=202,
    )
    def create_pair_simulation_replay(
        portfolio_id: str, payload: PairSimulationReplayRequest, request: Request
    ) -> dict:
        del portfolio_id, payload, request
        raise HTTPException(
            410,
            "pair simulation writes are retired; Autopilot is long-only",
        )

    @app.get("/api/simulation-portfolios/{portfolio_id}")
    def get_simulation_portfolio(portfolio_id: str) -> dict:
        try:
            return simulations.get(portfolio_id)
        except KeyError as exc:
            raise HTTPException(404, "simulation portfolio not found") from exc

    @app.post("/api/simulation-portfolios/{portfolio_id}/activate")
    def activate_simulation_portfolio(portfolio_id: str) -> dict:
        try:
            portfolio = simulations.get(portfolio_id)
            if portfolio.get("execution_adapter") != "long_only":
                raise HTTPException(
                    410,
                    "pair simulation writes are retired; historical pair ledgers are read-only",
                )
            return simulations.set_status(portfolio_id, "active")
        except KeyError as exc:
            raise HTTPException(404, "simulation portfolio not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/simulation-portfolios/{portfolio_id}/pause")
    def pause_simulation_portfolio(portfolio_id: str) -> dict:
        try:
            return simulations.set_status(portfolio_id, "paused")
        except KeyError as exc:
            raise HTTPException(404, "simulation portfolio not found") from exc

    @app.post("/api/simulation-portfolios/{portfolio_id}/nav/{trade_date}/review")
    def review_simulation_nav(
        portfolio_id: str,
        trade_date: date,
        payload: SimulationNavReviewRequest,
        request: Request,
    ) -> dict:
        try:
            return simulations.review_nav(
                portfolio_id,
                trade_date,
                actor=authenticated_actor(request, payload.actor),
                evidence_sha256=payload.evidence_sha256,
                note=payload.note,
            )
        except KeyError as exc:
            raise HTTPException(404, "simulation portfolio or NAV row not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post(
        "/api/simulation-portfolios/{portfolio_id}/fills/{fill_id}/final-fee",
        status_code=201,
    )
    def record_simulation_final_fee(
        portfolio_id: str,
        fill_id: str,
        payload: SimulationFinalFeeRequest,
        request: Request,
    ) -> dict:
        try:
            return simulations.record_final_fee(
                portfolio_id,
                fill_id=fill_id,
                final_fee=payload.final_fee,
                evidence_sha256=payload.evidence_sha256,
                source=payload.source,
                adjustment_key=payload.adjustment_key,
                actor=authenticated_actor(request, payload.actor),
            )
        except KeyError as exc:
            raise HTTPException(404, "simulation portfolio or fill not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/simulation-portfolios/{portfolio_id}/performance")
    def get_simulation_performance(portfolio_id: str) -> dict:
        try:
            return simulations.performance_summary(portfolio_id)
        except KeyError as exc:
            raise HTTPException(404, "simulation portfolio not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/simulation-portfolios/{portfolio_id}/{resource}")
    def get_simulation_resource(
        portfolio_id: str,
        resource: Literal[
            "batches",
            "orders",
            "fills",
            "positions",
            "nav",
            "events",
            "external_flows",
            "cash_flows",
            "corporate_events",
            "dividend_actions",
            "dividend_entitlements",
            "fee_adjustments",
            "cash_lots",
            "cash_events",
            "cash_reservations",
            "position_reservations",
            "security_events",
            "day_attributions",
        ],
        limit: int = Query(500, ge=1, le=5000),
    ) -> list[dict]:
        try:
            simulations.get(portfolio_id)
            return simulations.rows(portfolio_id, resource, limit=limit)
        except KeyError as exc:
            raise HTTPException(404, "simulation portfolio not found") from exc

    @app.get("/api/schedules")
    def list_schedules(limit: int = Query(200, ge=1, le=500)) -> list[dict]:
        return _sanitize_public_value(schedules.list(limit))

    @app.post("/api/schedules", status_code=201)
    def create_schedule(payload: ScheduleCreateRequest, request: Request) -> dict:
        schedule_actor = authenticated_actor(request, payload.actor)
        schedule_payload = dict(payload.payload)
        if payload.kind == "rdagent_research":
            try:
                research_payload = normalize_research_schedule_payload(
                    schedule_payload,
                    max_loops=settings.rdagent_max_loops,
                    max_duration=settings.rdagent_max_duration,
                )
                runtime = probe_rdagent(settings, project_root)
                scenario = require_ready_scenario(runtime, settings, research_payload["scenario"])
                scheduled_periods: dict[str, str] | None = None
                if scenario.requires_dataset:
                    dataset = require_qlib_dataset(
                        research_payload["dataset"],
                        purpose="scheduled RD-Agent research",
                        frequency="day",
                    )
                    scheduled_periods, _ = resolve_dataset_research_periods(
                        dataset,
                        periods=research_payload.get("periods"),
                        period_policy=research_payload.get("period_policy"),
                        horizon_profile=research_payload.get("horizon_profile"),
                        feature_set=research_payload.get("feature_set"),
                    )
                if research_payload["asset_ids"] or not scenario.auto_select_assets:
                    resolve_rdagent_assets(
                        settings,
                        scenario,
                        research_payload["asset_ids"],
                        pre_final_end=(
                            date.fromisoformat(scheduled_periods["valid_end"])
                            if scheduled_periods is not None
                            else None
                        ),
                        selection_limit=(
                            research_payload["loop_n"]
                            if scenario.id == "fin_factor_report"
                            else None
                        ),
                    )
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            if not scenario.requires_dataset and payload.trading_days_only:
                raise HTTPException(
                    409, "RD-Agent lab scenario schedules must disable trading_days_only"
                )
            research_payload["requested_by"] = schedule_actor
            schedule_payload = research_payload
        elif payload.kind == "recommendation_refresh":
            try:
                portfolio = recommendations.get(
                    str(schedule_payload["recommendation_portfolio_id"])
                )
            except KeyError as exc:
                raise HTTPException(404, "recommendation portfolio not found") from exc
            if portfolio["status"] != "active":
                raise HTTPException(409, "only active recommendation portfolios can be scheduled")
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
            return _sanitize_public_value(response.json())
        except (requests.RequestException, ValueError):
            return {"status": "unavailable"}

    @app.get("/api/operations/health")
    def operational_health(limit: int = Query(48, ge=1, le=500)) -> dict:
        return _sanitize_public_value(
            {
                "latest": health_history.latest(),
                "history": health_history.list(limit),
            }
        )

    @app.get("/api/operations/readiness")
    def operational_readiness() -> dict:
        return _sanitize_public_value(deployment_readiness.assess())

    @app.get("/api/platform/safe-mode")
    def get_safe_mode() -> dict:
        return safe_mode.status()

    @app.post("/api/platform/safe-mode/engage")
    def engage_safe_mode(payload: SafeModeEngageRequest, request: Request) -> dict:
        actor = authenticated_actor(request, payload.actor or "local-operator")
        return safe_mode.activate(
            reason=payload.reason,
            source="manual",
            actor=actor,
        )

    @app.post("/api/platform/safe-mode/release")
    def release_safe_mode(payload: SafeModeReleaseRequest, request: Request) -> dict:
        actor = authenticated_actor(request, payload.actor or "local-operator")
        safe_mode_state = safe_mode.status()
        latest = health_history.latest()
        health_status = safe_mode_recovery_health_status(
            latest,
            triggered_at=safe_mode_state.get("triggered_at"),
        )
        business_loop = deployment_readiness.business_loop_readiness()
        # Recovery must prove that the sealed daily input is current, but it
        # deliberately cannot require three-horizon paper lanes: safe mode
        # itself prevents creating the simulation accounts behind those lanes.
        # ``/api/readyz`` continues to fail closed until all lanes are active.
        daily_data = (business_loop.get("checks") or {}).get("daily_qlib_data") or {}
        if daily_data.get("status") != "ok":
            health_status = "degraded"
        try:
            return safe_mode.deactivate(
                actor=actor,
                reason=payload.reason,
                require_health_ok=payload.require_health_ok,
                health_status=health_status,
            )
        except ValueError as exc:
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
        response.headers["X-Total-Count"] = str(jobs.count(statuses=statuses, kinds=kinds))
        response.headers["Access-Control-Expose-Headers"] = "X-Total-Count"
        return [
            _public_job(item)
            for item in jobs.list(
                limit,
                offset=offset,
                statuses=statuses,
                kinds=kinds,
                payload_keys=tuple(sorted(_PUBLIC_JOB_PAYLOAD_KEYS)),
                progress_keys=tuple(
                    sorted(_PUBLIC_JOB_PROGRESS_KEYS | {"resource_blocked_count"})
                ),
                progress_evaluation_statuses=_MODEL_OUTCOME_EVALUATION_STATUSES,
                progress_evaluation_kind="model_evaluate",
            )
        ]

    @app.get("/api/data-tasks")
    def list_data_tasks() -> list[dict]:
        return _sanitize_public_value(current_data_tasks())

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        try:
            return _public_job(
                jobs.get(
                    job_id,
                    payload_keys=tuple(sorted(_PUBLIC_JOB_PAYLOAD_KEYS)),
                    progress_keys=tuple(
                        sorted(_PUBLIC_JOB_PROGRESS_KEYS | {"resource_blocked_count"})
                    ),
                    progress_evaluation_statuses=_MODEL_OUTCOME_EVALUATION_STATUSES,
                    progress_evaluation_kind="model_evaluate",
                )
            )
        except KeyError as exc:
            raise HTTPException(404, "job not found") from exc

    @app.get("/api/jobs/{job_id}/log")
    def get_job_log(job_id: str, tail: int = Query(200, ge=1, le=2000)) -> dict:
        try:
            jobs.get(job_id, payload_keys=(), progress_keys=())
        except KeyError as exc:
            raise HTTPException(404, "job not found") from exc
        # Raw subprocess logs may contain host paths, signed URLs, or upstream
        # diagnostics with secrets. They are intentionally server-log only.
        return {"job_id": job_id, "lines": [], "restricted": True}

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
            f"cn-{payload.snapshot_start:%Y%m%d}-{requested_end:%Y%m%d}-"
            f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        )
        serialized = {
            "profile": payload.profile,
            "start": payload.start.isoformat(),
            # Freeze "latest" at request time.  Re-resolving it inside a worker
            # in the Shanghai timezone can cross midnight and plan an
            # unfinished trading day outside the immutable snapshot contract.
            "end": requested_end.isoformat(),
            "build_qlib": False,
            "finalize_after_download": payload.build_qlib,
            "pipeline_id": uuid.uuid4().hex,
            "snapshot_start": payload.snapshot_start.isoformat(),
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

    @app.post("/api/jobs/baostock-overlap-validation", status_code=202)
    def create_baostock_overlap_validation(payload: BaoStockOverlapRequest) -> dict:
        if jobs.count(
            statuses=("queued", "running"),
            kinds=("bootstrap", "legacy_market_backfill"),
        ):
            raise HTTPException(409, "market-data download is already active")
        output = settings.data_root / "artifacts" / "data-quality"
        result_path = output / "baostock-overlap-2016.json"
        serialized = {
            "start": payload.start.isoformat(),
            "end": payload.end.isoformat(),
            "symbols": payload.symbols,
            "result_path": str(result_path),
        }
        log_path = platform_root / "logs" / "baostock-overlap-validation.log"
        try:
            job = jobs.create(
                "baostock_overlap_validation",
                serialized,
                log_path,
                idempotency_key=(
                    f"baostock-overlap:{BAOSTOCK_OVERLAP_POLICY_VERSION}:"
                    f"{payload.start}:{payload.end}:"
                    f"{','.join(sorted(payload.symbols)) or 'audited-default'}"
                ),
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/legacy-market-backfill", status_code=202)
    def create_legacy_market_backfill(payload: LegacyMarketBackfillRequest) -> dict:
        if jobs.count(
            statuses=("queued", "running"),
            kinds=("bootstrap", "baostock_overlap_validation"),
        ):
            raise HTTPException(409, "market-data download or validation is already active")
        validation_path = (
            settings.data_root / "artifacts" / "data-quality" / "baostock-overlap-2016.json"
        )
        try:
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(409, "BaoStock overlap validation is unavailable") from exc
        if (
            not isinstance(validation, dict)
            or validation.get("ok") is not True
            or validation.get("source") != "baostock-0.9.3"
            or validation.get("reference_source") != PRIMARY_OVERLAP_PROVIDER
            or validation.get("policy_version") != BAOSTOCK_OVERLAP_POLICY_VERSION
            or str(validation.get("start_date") or "") > "2016-01-01"
            or str(validation.get("end_date") or "") < "2016-12-31"
        ):
            raise HTTPException(409, "BaoStock overlap validation did not pass")
        try:
            require_current_primary_overlap_evidence(validation, checkpoint)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        result_path = (
            settings.data_root
            / "artifacts"
            / "execution-data"
            / f"legacy-market-{payload.start:%Y%m%d}-{payload.end:%Y%m%d}"
            / "result.json"
        )
        serialized = {
            "start": payload.start.isoformat(),
            "end": payload.end.isoformat(),
            "validation_report": str(validation_path),
            "result_path": str(result_path),
        }
        log_path = (
            platform_root
            / "logs"
            / (f"legacy-market-{payload.start:%Y%m%d}-{payload.end:%Y%m%d}.log")
        )
        try:
            job = jobs.create(
                "legacy_market_backfill",
                serialized,
                log_path,
                idempotency_key=f"legacy-market:{payload.start}:{payload.end}:baostock-0.9.3",
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/announcement-nlp", status_code=202)
    def create_announcement_nlp(payload: AnnouncementNlpRequest) -> dict:
        if jobs.count(
            statuses=("queued", "running"),
            kinds=("cninfo_announcements_download", "announcement_nlp"),
        ):
            raise HTTPException(409, "announcement download or NLP processing is already active")
        end_date = payload.end if isinstance(payload.end, date) else date.today()
        serialized = {
            "start": payload.start.isoformat(),
            "end": end_date.isoformat(),
            "ts_codes": sorted(set(payload.ts_codes)),
            "categories": sorted(set(payload.categories)),
            "limit": payload.limit,
            "batch_size": payload.batch_size,
            "workers": payload.workers,
            "prompt_version": ANNOUNCEMENT_PROMPT_VERSION,
        }
        log_path = (
            platform_root
            / "logs"
            / (f"announcement-nlp-{payload.start:%Y%m%d}-{end_date:%Y%m%d}.log")
        )
        try:
            job = jobs.create(
                "announcement_nlp",
                serialized,
                log_path,
                idempotency_key=(
                    f"announcement-nlp:{ANNOUNCEMENT_PROMPT_VERSION}:{payload.start}:"
                    f"{end_date}:{','.join(serialized['categories'])}:"
                    f"{','.join(serialized['ts_codes']) or 'all'}:{payload.limit}:"
                    f"batch-{payload.batch_size}:workers-{payload.workers}"
                ),
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/corpus-nlp", status_code=202)
    def create_corpus_nlp(payload: CorpusNlpRequest) -> dict:
        if jobs.count(statuses=("queued", "running"), kinds=("corpus_nlp",)):
            raise HTTPException(409, "corpus NLP processing is already active")
        end_date = payload.end if isinstance(payload.end, date) else date.today()
        serialized = {
            "start": payload.start.isoformat(),
            "end": end_date.isoformat(),
            "datasets": sorted(set(payload.datasets)),
            "ts_codes": sorted(set(payload.ts_codes)),
            "limit": payload.limit,
            "batch_size": payload.batch_size,
            "workers": payload.workers,
            "major_news_per_day": payload.major_news_per_day,
            "irm_per_instrument_day": payload.irm_per_instrument_day,
            "prompt_version": CORPUS_PROMPT_VERSION,
        }
        log_path = (
            platform_root / "logs" / (f"corpus-nlp-{payload.start:%Y%m%d}-{end_date:%Y%m%d}.log")
        )
        try:
            job = jobs.create(
                "corpus_nlp",
                serialized,
                log_path,
                idempotency_key=(
                    f"corpus-nlp:{CORPUS_PROMPT_VERSION}:{payload.start}:{end_date}:"
                    f"{','.join(serialized['datasets']) or 'all'}:"
                    f"{','.join(serialized['ts_codes']) or 'all'}:{payload.limit}:"
                    f"batch-{payload.batch_size}:major-{payload.major_news_per_day}:"
                    f"irm-{payload.irm_per_instrument_day}:workers-{payload.workers}"
                ),
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/event-market-response", status_code=202)
    def create_event_market_response(payload: EventMarketResponseRequest) -> dict:
        if jobs.count(
            statuses=("queued", "running"),
            kinds=("announcement_nlp", "event_market_response"),
        ):
            raise HTTPException(
                409, "announcement NLP or event market-response labeling is already active"
            )
        snapshot = settings.data_root / "snapshots" / payload.snapshot_name
        verification_path = snapshot / "verification.json"
        nlp_root = settings.data_root / ANNOUNCEMENTS_DIR / NLP_SUBDIR
        fields_path = nlp_root / "fields.parquet"
        logic_manifest_path = nlp_root / "factors" / f"{LOGIC_FACTOR_NAME}.json"
        try:
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(409, "verified snapshot evidence is unavailable") from exc
        if verification.get("ok") is not True or verification.get("errors"):
            raise HTTPException(409, "snapshot did not pass the blocking quality gate")
        if not fields_path.is_file():
            raise HTTPException(409, "announcement NLP fields are unavailable")
        if not logic_manifest_path.is_file():
            raise HTTPException(409, "governed logic factor manifest is unavailable")
        horizons = sorted(set(payload.horizons))
        serialized = {
            "snapshot_name": payload.snapshot_name,
            "horizons": horizons,
            "benchmark_code": payload.benchmark_code.upper(),
            "schema_version": LABEL_SCHEMA_VERSION,
        }
        log_path = platform_root / "logs" / (f"event-market-response-{payload.snapshot_name}.log")
        try:
            job = jobs.create(
                "event_market_response",
                serialized,
                log_path,
                idempotency_key=(
                    f"event-market-response:{LABEL_SCHEMA_VERSION}:{payload.snapshot_name}:"
                    f"{','.join(str(value) for value in horizons)}:"
                    f"{serialized['benchmark_code']}"
                ),
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/multiface-audit", status_code=202)
    def create_multiface_audit(payload: MultifaceAuditRequest) -> dict:
        dataset = require_qlib_dataset(payload.dataset, purpose="multi-face readiness audit")
        if dataset.get("frequency") != "day":
            raise HTTPException(409, "multi-face readiness audit requires a daily Qlib dataset")
        if jobs.count(statuses=("queued", "running"), kinds=("multiface_audit",)):
            raise HTTPException(409, "a multi-face readiness audit is already active")
        serialized = {
            "dataset": payload.dataset,
            "dataset_identity_sha256": dataset["provenance"]["dataset_identity_sha256"],
            "snapshot_name": payload.snapshot_name,
            "require_ready": payload.require_ready,
        }
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        log_path = platform_root / "logs" / f"multiface-audit-{stamp}.log"
        try:
            job = jobs.create(
                "multiface_audit",
                serialized,
                log_path,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/finalize-data", status_code=202)
    def finalize_data_pipeline(payload: DataFinalizeRequest) -> dict:
        datasets = set(checkpoint.datasets())
        if not datasets:
            raise HTTPException(409, "no downloaded work units are available to finalize")
        end_date = payload.end if isinstance(payload.end, date) else date.today()
        snapshot_name = payload.snapshot_name or (
            f"{'research-assets' if payload.profile == 'research-assets' else 'cn'}-"
            f"{payload.start:%Y%m%d}-{end_date:%Y%m%d}-"
            f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
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
        daily_dataset, source_lineage_id = require_bound_daily_source(
            requested_name=payload.daily_dataset,
            start=payload.start,
            end=end_date,
            purpose="core intraday download",
        )
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
            "daily_dataset": daily_dataset["name"],
            "source_lineage_id": source_lineage_id,
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
        daily_dataset, source_lineage_id = require_bound_daily_source(
            requested_name=payload.daily_dataset,
            start=payload.start,
            end=end_date,
            purpose="A-share five-minute download",
        )
        snapshot_name = payload.snapshot_name or (
            f"ashare-5m-{payload.start:%Y%m%d}-{end_date:%Y%m%d}"
        )
        serialized = {
            "start": payload.start.isoformat(),
            "end": end_date.isoformat(),
            "snapshot_name": snapshot_name,
            "daily_dataset": daily_dataset["name"],
            "source_lineage_id": source_lineage_id,
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
        if payload.publish_research_assets:
            snapshot_name = payload.snapshot_name or (
                f"research-assets-{payload.start:%Y%m%d}-{end_date:%Y%m%d}-"
                f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
            )
            serialized.update(
                {
                    "pipeline_id": uuid.uuid4().hex,
                    "profile": "research-assets",
                    "snapshot_name": snapshot_name,
                    "pipeline_steps": [
                        {"kind": "data_verify", "payload": {}},
                        {"kind": "data_snapshot", "payload": {}},
                    ],
                    "pipeline_next_index": 0,
                }
            )
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
            existing = jobs.get(job_id)
            if existing["kind"] == "strategy_backtest":
                raise ValueError("formal final-test jobs cannot be retried")
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
        dataset = require_qlib_dataset(payload.dataset, purpose="Qlib baseline", frequency="day")
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
        source_frequency = str(manifest.get("frequency") or "")
        if source_frequency not in NATIVE_MINUTE_FREQUENCIES:
            raise HTTPException(409, "minute Qlib requires a supported minute snapshot")
        supported = set(MINUTE_DATASETS)
        if not supported.intersection(manifest.get("datasets", {})):
            raise HTTPException(409, "execution snapshot has no supported minute datasets")
        target_frequency = payload.target_frequency or source_frequency
        if target_frequency in NATIVE_MINUTE_FREQUENCIES and target_frequency != source_frequency:
            raise HTTPException(409, "native minute Qlib output must match the snapshot frequency")
        output_name = payload.output_name or f"{payload.snapshot_name}-{target_frequency}"
        serialized = {
            "snapshot_name": payload.snapshot_name,
            "snapshot_manifest_sha256": snapshot["manifest_sha256"],
            "output_name": output_name,
            "source_frequency": source_frequency,
            "target_frequency": target_frequency,
            "frequency": target_frequency,
        }
        log_path = platform_root / "logs" / f"minute-qlib-{output_name}.log"
        try:
            job = jobs.create(
                "minute_qlib",
                serialized,
                log_path,
                idempotency_key=(
                    f"minute-qlib:{payload.snapshot_name}:{output_name}:{target_frequency}"
                ),
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return job

    @app.post("/api/jobs/minute-research", status_code=202)
    def create_minute_research(payload: MinuteResearchRequest) -> dict:
        dataset = require_qlib_dataset(payload.dataset, purpose="minute factor research")
        frequency = str(dataset.get("frequency") or "")
        if frequency not in MINUTE_FREQUENCIES:
            raise HTTPException(409, "minute research requires a minute Qlib dataset")
        try:
            require_minute_signal_contract(dataset["provenance"], frequency=frequency)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        bar_minutes = int(frequency.removesuffix("min"))
        if any(horizon % bar_minutes for horizon in payload.horizons):
            raise HTTPException(409, "research horizons must be multiples of the dataset frequency")
        serialized = {
            "dataset": payload.dataset,
            "dataset_path": dataset["path"],
            "dataset_identity_sha256": dataset["provenance"]["dataset_identity_sha256"],
            "frequency": frequency,
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

    @app.get("/api/capabilities")
    def platform_capabilities() -> dict[str, Any]:
        return {
            # A real broker is intentionally not a deployable capability.  Do
            # not let an environment variable silently widen this boundary.
            "broker_qmt": False,
            "recommendation_tracking": True,
            "research_pipeline_version": "research-pipeline-v2",
        }

    return app


app = create_app()
