from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    Time,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

metadata = MetaData(schema="quantlab")
json_type = JSON().with_variant(JSONB, "postgresql")

work_units = Table(
    "work_units",
    metadata,
    *[
        Column("unit_key", String, primary_key=True),
        Column("dataset", String, nullable=False),
        Column("api_name", String, nullable=False),
        Column("scope_json", json_type, nullable=False),
        Column("params_json", json_type, nullable=False),
        Column("fields_json", json_type, nullable=False),
        Column("allow_empty", Boolean, nullable=False),
        Column("status", String, nullable=False, default="pending"),
        Column("attempts", Integer, nullable=False, default=0),
        Column("max_attempts", Integer, nullable=False),
        Column("next_retry_at", DateTime(timezone=True)),
        Column("lease_until", DateTime(timezone=True)),
        Column("output_path", Text),
        Column("row_count", Integer),
        Column("sha256", String),
        Column("last_error", Text),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    ],
)
Index(
    "idx_work_units_claim",
    work_units.c.status,
    work_units.c.next_retry_at,
    work_units.c.lease_until,
    work_units.c.dataset,
)

jobs = Table(
    "jobs",
    metadata,
    *[
        Column("id", String, primary_key=True),
        Column("kind", String, nullable=False),
        Column("idempotency_key", String, unique=True),
        Column("status", String, nullable=False),
        Column("payload_json", json_type, nullable=False),
        Column("progress_json", json_type),
        Column("log_path", Text),
        Column("exit_code", Integer),
        Column("error", Text),
        Column("attempts", Integer, nullable=False, default=0),
        Column("max_attempts", Integer, nullable=False, default=1),
        Column("next_attempt_at", DateTime(timezone=True)),
        Column("cancel_requested_at", DateTime(timezone=True)),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("started_at", DateTime(timezone=True)),
        Column("finished_at", DateTime(timezone=True)),
    ],
)
Index("idx_jobs_status_created", jobs.c.status, jobs.c.next_attempt_at, jobs.c.created_at.desc())

data_tasks = Table(
    "data_tasks",
    metadata,
    *[
        Column("task_key", String, primary_key=True),
        Column("phase", Integer, nullable=False),
        Column("sort_order", Integer, nullable=False),
        Column("title", String, nullable=False),
        Column("description", Text, nullable=False),
        Column("category", String, nullable=False),
        Column("source", String, nullable=False),
        Column("status", String, nullable=False),
        Column("implementation_status", String, nullable=False),
        Column("depends_on_json", json_type, nullable=False),
        Column("config_json", json_type, nullable=False),
        Column("estimated_storage_gb", Integer),
        Column("job_id", String, ForeignKey("quantlab.jobs.id", ondelete="SET NULL")),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    ],
)
Index("idx_data_tasks_phase_order", data_tasks.c.phase, data_tasks.c.sort_order)

research_runs = Table(
    "research_runs",
    metadata,
    *[
        Column("id", String, primary_key=True),
        Column("job_id", String, ForeignKey("quantlab.jobs.id", ondelete="SET NULL")),
        Column("kind", String, nullable=False),
        Column("objective", Text, nullable=False),
        Column("dataset", String, nullable=False),
        Column("status", String, nullable=False),
        Column("requested_by", String, nullable=False),
        Column("budget_json", json_type, nullable=False),
        Column("config_json", json_type, nullable=False),
        Column("runtime_json", json_type),
        Column("artifact_path", Text),
        Column("error", Text),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("started_at", DateTime(timezone=True)),
        Column("finished_at", DateTime(timezone=True)),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    ],
)
Index("idx_research_runs_status_created", research_runs.c.status, research_runs.c.created_at.desc())
Index(
    "uq_research_runs_active_kind",
    research_runs.c.kind,
    unique=True,
    postgresql_where=research_runs.c.status.in_(("queued", "running", "evaluating")),
)

factor_candidates = Table(
    "factor_candidates",
    metadata,
    *[
        Column("id", String, primary_key=True),
        Column(
            "research_run_id",
            String,
            ForeignKey("quantlab.research_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("name", String, nullable=False),
        Column("description", Text, nullable=False),
        Column("formulation", Text),
        Column("variables_json", json_type, nullable=False),
        Column("status", String, nullable=False),
        Column("source_iteration", Integer),
        Column("experiment_family_id", String),
        Column("label_horizon_days", Integer),
        Column("experiment_count", Integer),
        Column("code_path", Text),
        Column("values_path", Text),
        Column("code_sha256", String),
        Column("values_sha256", String),
        Column("rdagent_decision", Boolean),
        Column("rdagent_feedback", Text),
        Column("promoted_evaluation_id", String),
        Column("promotion_evidence_sha256", String),
        Column("promoted_by", String),
        Column("promoted_at", DateTime(timezone=True)),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    ],
)
Index(
    "uq_factor_candidate_run_name",
    factor_candidates.c.research_run_id,
    factor_candidates.c.name,
    unique=True,
)
Index(
    "idx_factor_candidates_status_updated",
    factor_candidates.c.status,
    factor_candidates.c.updated_at.desc(),
)

factor_evaluations = Table(
    "factor_evaluations",
    metadata,
    *[
        Column("id", String, primary_key=True),
        Column("is_legacy", Boolean, nullable=False, server_default="false"),
        Column(
            "factor_candidate_id",
            String,
            ForeignKey("quantlab.factor_candidates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("dataset", String, nullable=False),
        Column("dataset_identity_sha256", String),
        Column("train_start", Date, nullable=False),
        Column("train_end", Date, nullable=False),
        Column("valid_start", Date, nullable=False),
        Column("valid_end", Date, nullable=False),
        Column("test_start", Date, nullable=False),
        Column("test_end", Date, nullable=False),
        Column("ic", Float),
        Column("icir", Float),
        Column("rank_ic", Float),
        Column("rank_icir", Float),
        Column("turnover", Float),
        Column("max_correlation", Float),
        Column("cost_adjusted_return", Float),
        Column("metrics_json", json_type, nullable=False),
        Column("gate_status", String, nullable=False),
        Column("gate_reasons_json", json_type, nullable=False),
        Column("evaluator_version", String, nullable=False),
        Column("artifact_path", Text),
        Column("artifact_sha256", String),
        Column("candidate_code_sha256", String),
        Column("candidate_values_sha256", String),
        Column("submitted_values_sha256", String),
        Column("recomputed_values_sha256", String),
        Column("recompute_evidence_json", json_type),
        Column("hac_p_value", Float),
        Column("bh_q_value", Float),
        Column("statistical_contract_version", String),
        Column("signal_frequency", String, nullable=False, server_default="day"),
        Column("signal_horizon", String, nullable=False, server_default="1d"),
        Column("execution_frequency", String, nullable=False, server_default="5min"),
        Column(
            "execution_contract_hash",
            String,
            nullable=False,
            server_default="legacy-unversioned",
        ),
        Column("qlib_version", String),
        Column("qlib_commit", String),
        Column("rdagent_version", String),
        Column("rdagent_commit", String),
        Column("final_test_key", String),
        Column("final_test_consumed_at", DateTime(timezone=True)),
        Column("metrics_sha256", String),
        Column("policy_json", json_type),
        Column("policy_sha256", String),
        Column("evidence_sha256", String),
        Column("created_at", DateTime(timezone=True), nullable=False),
    ],
)
Index(
    "idx_factor_evaluations_candidate_created",
    factor_evaluations.c.factor_candidate_id,
    factor_evaluations.c.created_at.desc(),
)

research_events = Table(
    "research_events",
    metadata,
    *[
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column(
            "research_run_id",
            String,
            ForeignKey("quantlab.research_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column(
            "factor_candidate_id",
            String,
            ForeignKey("quantlab.factor_candidates.id", ondelete="CASCADE"),
        ),
        Column("event_type", String, nullable=False),
        Column("actor", String, nullable=False),
        Column("payload_json", json_type, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
    ],
)
Index(
    "idx_research_events_run_created",
    research_events.c.research_run_id,
    research_events.c.created_at,
)

strategies = Table(
    "strategies",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False, unique=True),
    Column("description", Text, nullable=False),
    Column("status", String, nullable=False),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

strategy_versions = Table(
    "strategy_versions",
    metadata,
    Column("id", String, primary_key=True),
    Column("is_legacy", Boolean, nullable=False, server_default="false"),
    Column(
        "strategy_id",
        String,
        ForeignKey("quantlab.strategies.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("version", Integer, nullable=False),
    Column("status", String, nullable=False),
    Column("strategy_type", String, nullable=False, server_default="multifactor"),
    Column("signal_frequency", String, nullable=False, server_default="day"),
    Column("signal_horizon", String, nullable=False, server_default="1d"),
    Column("execution_frequency", String, nullable=False, server_default="5min"),
    Column(
        "execution_contract_hash",
        String,
        nullable=False,
        server_default="legacy-unversioned",
    ),
    Column("qlib_version", String),
    Column("qlib_commit", String),
    Column("rdagent_version", String),
    Column("rdagent_commit", String),
    Column("benchmark", String, nullable=False),
    Column("universe", String, nullable=False),
    Column("config_json", json_type, nullable=False),
    Column("created_by", String, nullable=False),
    Column("approved_by", String),
    Column("approval_reason", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("approved_at", DateTime(timezone=True)),
)
Index(
    "uq_strategy_versions_number",
    strategy_versions.c.strategy_id,
    strategy_versions.c.version,
    unique=True,
)
Index(
    "uq_strategy_versions_approved",
    strategy_versions.c.strategy_id,
    unique=True,
    postgresql_where=strategy_versions.c.status == "approved",
)

strategy_factors = Table(
    "strategy_factors",
    metadata,
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "factor_candidate_id",
        String,
        ForeignKey("quantlab.factor_candidates.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column(
        "factor_evaluation_id",
        String,
        ForeignKey("quantlab.factor_evaluations.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("weight", Float, nullable=False),
    Column("direction", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

strategy_pairs = Table(
    "strategy_pairs",
    metadata,
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("leg_y", String, nullable=False),
    Column("leg_x", String, nullable=False),
    Column("asset_class", String, nullable=False),
    Column("shorting_mode", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

backtest_runs = Table(
    "backtest_runs",
    metadata,
    Column("id", String, primary_key=True),
    Column("is_legacy", Boolean, nullable=False, server_default="false"),
    Column("job_id", String, ForeignKey("quantlab.jobs.id", ondelete="SET NULL")),
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("dataset", String, nullable=False),
    Column("execution_dataset", String),
    Column("signal_frequency", String, nullable=False, server_default="day"),
    Column("execution_frequency", String, nullable=False, server_default="5min"),
    Column(
        "execution_contract_hash",
        String,
        nullable=False,
        server_default="legacy-unversioned",
    ),
    Column("qlib_version", String),
    Column("qlib_commit", String),
    Column("rdagent_version", String),
    Column("rdagent_commit", String),
    Column("status", String, nullable=False),
    Column("periods_json", json_type, nullable=False),
    Column("metrics_json", json_type),
    Column("artifact_path", Text),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
)
Index(
    "idx_backtest_runs_status_created",
    backtest_runs.c.status,
    backtest_runs.c.created_at.desc(),
)

parameter_experiments = Table(
    "parameter_experiments",
    metadata,
    Column("id", String, primary_key=True),
    Column("job_id", String, ForeignKey("quantlab.jobs.id", ondelete="SET NULL")),
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("dataset", String, nullable=False),
    Column("status", String, nullable=False),
    Column("periods_json", json_type, nullable=False),
    Column("parameter_grid_json", json_type, nullable=False),
    Column("baseline_config_json", json_type, nullable=False),
    Column("summary_json", json_type),
    Column("artifact_path", Text, nullable=False),
    Column("error", Text),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
)
Index(
    "idx_parameter_experiments_status_created",
    parameter_experiments.c.status,
    parameter_experiments.c.created_at.desc(),
)

parameter_experiment_trials = Table(
    "parameter_experiment_trials",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "experiment_id",
        String,
        ForeignKey("quantlab.parameter_experiments.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("trial_index", Integer, nullable=False),
    Column("parameters_json", json_type, nullable=False),
    Column("config_json", json_type, nullable=False),
    Column("status", String, nullable=False),
    Column("score", Float),
    Column("metrics_json", json_type),
    Column("warnings_json", json_type),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    UniqueConstraint("experiment_id", "trial_index", name="uq_parameter_experiment_trial"),
)
Index(
    "idx_parameter_experiment_trials_status",
    parameter_experiment_trials.c.experiment_id,
    parameter_experiment_trials.c.status,
    parameter_experiment_trials.c.trial_index,
)

research_programs = Table(
    "research_programs",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False, unique=True),
    Column("status", String, nullable=False),
    Column("recipe_id", String, nullable=False),
    Column("objective", Text, nullable=False),
    Column("benchmark", String, nullable=False),
    Column("universe", String, nullable=False),
    Column("dataset_lineage_id", String, nullable=False),
    Column("config_json", json_type, nullable=False),
    Column("min_new_trading_days", Integer, nullable=False),
    Column("max_active_campaigns", Integer, nullable=False),
    Column("last_dataset_name", String),
    Column("last_dataset_identity_sha256", String),
    Column("last_dataset_end_date", String),
    Column("last_message", Text),
    Column("last_checked_at", DateTime(timezone=True)),
    Column("last_triggered_at", DateTime(timezone=True)),
    Column("last_evaluated_campaign_id", String),
    Column("champion_campaign_id", String),
    Column("champion_strategy_version_id", String),
    Column("champion_score", Float),
    Column("champion_selected_at", DateTime(timezone=True)),
    Column("decay_status", String, nullable=False, server_default="unavailable"),
    Column("decay_message", Text),
    Column("next_check_at", DateTime(timezone=True), nullable=False),
    Column("lease_until", DateTime(timezone=True)),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_research_programs_claim",
    research_programs.c.status,
    research_programs.c.next_check_at,
    research_programs.c.lease_until,
)

research_campaigns = Table(
    "research_campaigns",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False, unique=True),
    Column("status", String, nullable=False),
    Column("stage", String, nullable=False),
    Column("objective", Text, nullable=False),
    Column("dataset", String, nullable=False),
    Column("benchmark", String, nullable=False),
    Column("universe", String, nullable=False),
    Column("recipe_id", String, nullable=False),
    Column(
        "research_program_id",
        String,
        ForeignKey("quantlab.research_programs.id", ondelete="SET NULL"),
    ),
    Column("dataset_identity_sha256", String),
    Column("config_json", json_type, nullable=False),
    Column("state_json", json_type, nullable=False),
    Column("research_run_id", String, ForeignKey("quantlab.research_runs.id", ondelete="SET NULL")),
    Column("strategy_id", String, ForeignKey("quantlab.strategies.id", ondelete="SET NULL")),
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="SET NULL"),
    ),
    Column(
        "parameter_experiment_id",
        String,
        ForeignKey("quantlab.parameter_experiments.id", ondelete="SET NULL"),
    ),
    Column("backtest_id", String, ForeignKey("quantlab.backtest_runs.id", ondelete="SET NULL")),
    Column(
        "paper_portfolio_id",
        String,
        ForeignKey("quantlab.paper_portfolios.id", ondelete="SET NULL"),
    ),
    Column("paper_schedule_id", String, ForeignKey("quantlab.schedules.id", ondelete="SET NULL")),
    Column("error", Text),
    Column("attempts", Integer, nullable=False, server_default="0"),
    Column("next_action_at", DateTime(timezone=True), nullable=False),
    Column("lease_until", DateTime(timezone=True)),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    UniqueConstraint(
        "research_program_id",
        "dataset_identity_sha256",
        name="uq_research_campaign_program_dataset",
    ),
)
Index(
    "idx_research_campaigns_claim",
    research_campaigns.c.status,
    research_campaigns.c.next_action_at,
    research_campaigns.c.lease_until,
)

research_campaign_events = Table(
    "research_campaign_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "campaign_id",
        String,
        ForeignKey("quantlab.research_campaigns.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("event_type", String, nullable=False),
    Column("actor", String, nullable=False),
    Column("payload_json", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_research_campaign_events_created",
    research_campaign_events.c.campaign_id,
    research_campaign_events.c.created_at,
)

research_program_events = Table(
    "research_program_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "program_id",
        String,
        ForeignKey("quantlab.research_programs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("event_type", String, nullable=False),
    Column("actor", String, nullable=False),
    Column("payload_json", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_research_program_events_created",
    research_program_events.c.program_id,
    research_program_events.c.created_at,
)

strategy_events = Table(
    "strategy_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "strategy_id",
        String,
        ForeignKey("quantlab.strategies.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="CASCADE"),
    ),
    Column("event_type", String, nullable=False),
    Column("actor", String, nullable=False),
    Column("payload_json", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_strategy_events_strategy_created",
    strategy_events.c.strategy_id,
    strategy_events.c.created_at,
)

recommendation_portfolios = Table(
    "recommendation_portfolios",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False, unique=True),
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("dataset", String, nullable=False),
    Column("status", String, nullable=False),
    Column("base_currency", String, nullable=False),
    Column("hypothetical_initial_value", Numeric(20, 6), nullable=False),
    Column("risk_exposure_override", Float, nullable=False, server_default="1"),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_recommendation_portfolios_status_updated",
    recommendation_portfolios.c.status,
    recommendation_portfolios.c.updated_at.desc(),
)

recommendation_snapshots = Table(
    "recommendation_snapshots",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.recommendation_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("job_id", String, ForeignKey("quantlab.jobs.id", ondelete="SET NULL")),
    Column("as_of_date", Date, nullable=False),
    Column("effective_date", Date),
    Column("status", String, nullable=False),
    Column("snapshot_json", json_type),
    Column("cost_model_json", json_type, nullable=False),
    Column("policy_version", String, nullable=False),
    Column("backtest_engine_version", String, nullable=False),
    Column("dataset", String, nullable=False),
    Column("dataset_identity_sha256", String, nullable=False),
    Column("strategy_version_id", String, nullable=False),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    UniqueConstraint(
        "portfolio_id", "as_of_date", name="uq_recommendation_snapshots_as_of"
    ),
)
Index(
    "idx_recommendation_snapshots_portfolio_created",
    recommendation_snapshots.c.portfolio_id,
    recommendation_snapshots.c.created_at.desc(),
)

recommendation_holdings = Table(
    "recommendation_holdings",
    metadata,
    Column(
        "snapshot_id",
        String,
        ForeignKey("quantlab.recommendation_snapshots.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("instrument", String, primary_key=True),
    Column("weight", Float, nullable=False),
    Column("previous_weight", Float, nullable=False),
    Column("weight_change", Float, nullable=False),
    Column("action", String, nullable=False),
    Column("reason", Text, nullable=False),
    Column("average_cost", Float),
    Column("take_profit_stage", Integer, nullable=False, server_default="0"),
)

recommendation_nav = Table(
    "recommendation_nav",
    metadata,
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.recommendation_portfolios.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("trade_date", Date, primary_key=True),
    Column("hypothetical_value", Numeric(20, 6), nullable=False),
    Column("daily_return", Float, nullable=False),
    Column("benchmark_return", Float),
    Column("drawdown", Float, nullable=False),
    Column("turnover", Float, nullable=False),
    Column("estimated_cost", Numeric(20, 6), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index("idx_recommendation_nav_trade_date", recommendation_nav.c.trade_date.desc())

simulation_portfolios = Table(
    "simulation_portfolios",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False, unique=True),
    Column(
        "recommendation_portfolio_id",
        String,
        ForeignKey("quantlab.recommendation_portfolios.id", ondelete="RESTRICT"),
        nullable=True,
    ),
    Column("source_type", String, nullable=False, server_default="recommendation"),
    Column("source_id", String, nullable=False),
    Column("status", String, nullable=False),
    Column("base_currency", String, nullable=False),
    Column("initial_cash", Numeric(20, 6), nullable=False),
    Column("cash", Numeric(20, 6), nullable=False),
    Column("nav", Numeric(20, 6), nullable=False),
    Column("high_water_mark", Numeric(20, 6), nullable=False),
    Column("execution_algorithm", String, nullable=False),
    Column("execution_adapter", String, nullable=False, server_default="long_only"),
    Column("execution_frequency", String, nullable=False, server_default="5min"),
    Column("execution_contract_hash", String, nullable=False),
    Column("execution_dataset", String, nullable=False),
    Column("daily_dataset", String, nullable=False),
    Column("daily_dataset_identity_sha256", String, nullable=False),
    Column("daily_dataset_lineage_id", String, nullable=False),
    Column("daily_field_contract_version", String, nullable=False),
    Column("execution_dataset_identity_sha256", String, nullable=False),
    Column("execution_dataset_lineage_id", String, nullable=False),
    Column("execution_field_contract_version", String, nullable=False),
    Column("execution_engine_version", String, nullable=False),
    Column("cost_schedule_version", String, nullable=False),
    Column("execution_policy_json", json_type, nullable=False),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_simulation_portfolios_status_updated",
    simulation_portfolios.c.status,
    simulation_portfolios.c.updated_at.desc(),
)
Index(
    "uq_simulation_portfolios_source_execution",
    simulation_portfolios.c.source_type,
    simulation_portfolios.c.source_id,
    simulation_portfolios.c.execution_dataset,
    unique=True,
)

simulation_batches = Table(
    "simulation_batches",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "recommendation_snapshot_id",
        String,
        ForeignKey("quantlab.recommendation_snapshots.id", ondelete="RESTRICT"),
        nullable=True,
    ),
    Column("source_snapshot_id", String),
    Column("target_payload_json", json_type),
    Column("execution_adapter", String, nullable=False, server_default="long_only"),
    Column("execution_contract_hash", String, nullable=False),
    Column("signal_date", Date, nullable=False),
    Column("trade_date", Date, nullable=False),
    Column("signal_at", DateTime(timezone=True)),
    Column("execution_not_before", DateTime(timezone=True)),
    Column("status", String, nullable=False),
    Column("idempotency_key", String, nullable=False, unique=True),
    Column("created_by", String, nullable=False, server_default="legacy-system"),
    Column("summary_json", json_type),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    CheckConstraint(
        "(signal_at IS NULL AND execution_not_before IS NULL) OR "
        "(signal_at IS NOT NULL AND execution_not_before IS NOT NULL "
        "AND execution_not_before > signal_at "
        "AND (signal_at AT TIME ZONE 'Asia/Shanghai')::date = signal_date "
        "AND (execution_not_before AT TIME ZONE 'Asia/Shanghai')::date = trade_date)",
        name="ck_simulation_batches_next_bar_time",
    ),
)
Index(
    "idx_simulation_batches_portfolio_date",
    simulation_batches.c.portfolio_id,
    simulation_batches.c.trade_date.desc(),
)
Index(
    "uq_simulation_batches_portfolio_recommendation",
    simulation_batches.c.portfolio_id,
    simulation_batches.c.recommendation_snapshot_id,
    unique=True,
    postgresql_where=simulation_batches.c.recommendation_snapshot_id.is_not(None),
)

simulation_orders = Table(
    "simulation_orders",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "batch_id",
        String,
        ForeignKey("quantlab.simulation_batches.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("instrument", String, nullable=False),
    Column("side", String, nullable=False),
    Column("atomic_group_id", String),
    Column("leg_no", Integer),
    Column("position_side", String, nullable=False, server_default="long"),
    Column("borrow_cost", Numeric(20, 6), nullable=False, server_default="0"),
    Column("target_weight", Float, nullable=False),
    Column("requested_quantity", Integer, nullable=False),
    Column("filled_quantity", Integer, nullable=False),
    Column("status", String, nullable=False),
    Column("reject_reason", String),
    Column("requested_value", Numeric(20, 6), nullable=False),
    Column("filled_value", Numeric(20, 6), nullable=False),
    Column("capacity_fill_ratio", Float, nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index("idx_simulation_orders_batch", simulation_orders.c.batch_id, simulation_orders.c.instrument)

simulation_fills = Table(
    "simulation_fills",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "order_id",
        String,
        ForeignKey("quantlab.simulation_orders.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "batch_id",
        String,
        ForeignKey("quantlab.simulation_batches.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("instrument", String, nullable=False),
    Column("side", String, nullable=False),
    Column("atomic_group_id", String),
    Column("leg_no", Integer),
    Column("position_side", String, nullable=False, server_default="long"),
    Column("borrow_cost", Numeric(20, 6), nullable=False, server_default="0"),
    Column("executed_at", DateTime(timezone=True), nullable=False),
    Column("quantity", Integer, nullable=False),
    Column("price", Numeric(20, 8), nullable=False),
    Column("gross_value", Numeric(20, 6), nullable=False),
    Column("fee", Numeric(20, 6), nullable=False),
    Column("cost_breakdown_json", json_type, nullable=False),
    Column("minute_volume", Integer, nullable=False),
    Column("capacity_quantity", Integer, nullable=False),
)
Index("idx_simulation_fills_batch", simulation_fills.c.batch_id, simulation_fills.c.executed_at)

simulation_positions = Table(
    "simulation_positions",
    metadata,
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("instrument", String, primary_key=True),
    Column("atomic_group_id", String),
    Column("leg_no", Integer),
    Column("position_side", String, nullable=False, server_default="long"),
    Column("borrow_cost", Numeric(20, 6), nullable=False, server_default="0"),
    Column("quantity", Integer, nullable=False),
    Column("available_quantity", Integer, nullable=False),
    Column("average_cost", Numeric(20, 8), nullable=False),
    Column("last_trade_date", Date),
    Column("market_price", Numeric(20, 8)),
    Column("market_date", Date),
    Column("stale", Boolean, nullable=False),
    Column("market_value", Numeric(20, 6), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

simulation_cash_flows = Table(
    "simulation_cash_flows",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "batch_id",
        String,
        ForeignKey("quantlab.simulation_batches.id", ondelete="CASCADE"),
    ),
    Column("trade_date", Date, nullable=False),
    Column("flow_type", String, nullable=False),
    Column("amount", Numeric(20, 6), nullable=False),
    Column("balance_after", Numeric(20, 6), nullable=False),
    Column("reference_id", String),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_simulation_cash_flows_portfolio_date",
    simulation_cash_flows.c.portfolio_id,
    simulation_cash_flows.c.trade_date,
)

simulation_nav = Table(
    "simulation_nav",
    metadata,
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("trade_date", Date, primary_key=True),
    Column("cash", Numeric(20, 6), nullable=False),
    Column("market_value", Numeric(20, 6), nullable=False),
    Column("nav", Numeric(20, 6), nullable=False),
    Column("daily_return", Float, nullable=False),
    Column("drawdown", Float, nullable=False),
    Column("market_date", Date),
    Column("has_stale_prices", Boolean, nullable=False),
    Column("status", String, nullable=False),
    Column("performance_certified", Boolean, nullable=False),
    Column("nav_scope", String, nullable=False, server_default="member_ledger"),
    Column("produced_by", String, nullable=False, server_default="legacy-system"),
    Column("reviewed_by", String),
    Column("reviewed_at", DateTime(timezone=True)),
    Column("review_evidence_sha256", String),
    Column("review_note", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index("idx_simulation_nav_trade_date", simulation_nav.c.trade_date.desc())

simulation_events = Table(
    "simulation_events",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "batch_id",
        String,
        ForeignKey("quantlab.simulation_batches.id", ondelete="CASCADE"),
    ),
    Column("trade_date", Date, nullable=False),
    Column("severity", String, nullable=False),
    Column("event_type", String, nullable=False),
    Column("instrument", String),
    Column("reason", String, nullable=False),
    Column("details_json", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_simulation_events_portfolio_date",
    simulation_events.c.portfolio_id,
    simulation_events.c.trade_date.desc(),
)

paper_portfolios = Table(
    "paper_portfolios",
    metadata,
    Column("id", String, primary_key=True),
    Column("is_legacy", Boolean, nullable=False, server_default="true"),
    Column("name", String, nullable=False, unique=True),
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("dataset", String, nullable=False),
    Column("dataset_roll_policy", String, nullable=False, server_default="pinned"),
    Column("dataset_lineage_id", String),
    Column("status", String, nullable=False),
    Column("base_currency", String, nullable=False),
    Column("initial_cash", Numeric(20, 6), nullable=False),
    Column("cash", Numeric(20, 6), nullable=False),
    Column("nav", Numeric(20, 6), nullable=False),
    Column("high_water_mark", Numeric(20, 6), nullable=False),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_paper_portfolios_status_updated",
    paper_portfolios.c.status,
    paper_portfolios.c.updated_at.desc(),
)

portfolio_batches = Table(
    "portfolio_batches",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.paper_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("job_id", String, ForeignKey("quantlab.jobs.id", ondelete="SET NULL")),
    Column("as_of_date", Date, nullable=False),
    Column("trade_date", Date),
    Column("status", String, nullable=False),
    Column("idempotency_key", String, nullable=False, unique=True),
    Column("artifact_path", Text),
    Column("dataset", String),
    Column("dataset_identity_sha256", String),
    Column("dataset_lineage_id", String),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    UniqueConstraint("portfolio_id", "as_of_date", name="uq_portfolio_batches_as_of"),
)
Index(
    "idx_portfolio_batches_status_created",
    portfolio_batches.c.status,
    portfolio_batches.c.created_at.desc(),
)

paper_orders = Table(
    "paper_orders",
    metadata,
    Column("id", String, primary_key=True),
    Column("is_legacy", Boolean, nullable=False, server_default="true"),
    Column(
        "batch_id",
        String,
        ForeignKey("quantlab.portfolio_batches.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.paper_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("instrument", String, nullable=False),
    Column("side", String, nullable=False),
    Column("order_type", String, nullable=False),
    Column("target_weight", Float, nullable=False),
    Column("requested_quantity", Numeric(20, 6), nullable=False),
    Column("status", String, nullable=False),
    Column("reason", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("batch_id", "instrument", name="uq_paper_orders_batch_instrument"),
)
Index("idx_paper_orders_portfolio_created", paper_orders.c.portfolio_id, paper_orders.c.created_at)

paper_fills = Table(
    "paper_fills",
    metadata,
    Column("id", String, primary_key=True),
    Column("is_legacy", Boolean, nullable=False, server_default="true"),
    Column(
        "order_id",
        String,
        ForeignKey("quantlab.paper_orders.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    ),
    Column("fill_time", DateTime(timezone=True), nullable=False),
    Column("quantity", Numeric(20, 6), nullable=False),
    Column("price", Numeric(20, 6), nullable=False),
    Column("gross_value", Numeric(20, 6), nullable=False),
    Column("fee", Numeric(20, 6), nullable=False),
    Column("slippage", Float, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

paper_positions = Table(
    "paper_positions",
    metadata,
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.paper_portfolios.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("instrument", String, primary_key=True),
    Column("industry", String),
    Column("take_profit_stage", Integer, nullable=False, server_default="0"),
    Column("quantity", Numeric(20, 6), nullable=False),
    Column("avg_cost", Numeric(20, 6), nullable=False),
    Column("market_price", Numeric(20, 6), nullable=False),
    Column("market_value", Numeric(20, 6), nullable=False),
    Column("weight", Float, nullable=False),
    Column("realized_pnl", Numeric(20, 6), nullable=False),
    Column("unrealized_pnl", Numeric(20, 6), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

portfolio_nav = Table(
    "portfolio_nav",
    metadata,
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.paper_portfolios.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("trade_date", Date, primary_key=True),
    Column("cash", Numeric(20, 6), nullable=False),
    Column("market_value", Numeric(20, 6), nullable=False),
    Column("nav", Numeric(20, 6), nullable=False),
    Column("daily_return", Float, nullable=False),
    Column("benchmark_return", Float),
    Column("drawdown", Float, nullable=False),
    Column("exposure", Float, nullable=False),
    Column("turnover", Float, nullable=False),
    Column("fees", Numeric(20, 6), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index("idx_portfolio_nav_trade_date", portfolio_nav.c.trade_date.desc())

risk_events = Table(
    "risk_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.paper_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "batch_id",
        String,
        ForeignKey("quantlab.portfolio_batches.id", ondelete="CASCADE"),
    ),
    Column("severity", String, nullable=False),
    Column("event_type", String, nullable=False),
    Column("rule", String, nullable=False),
    Column("observed", Float),
    Column("limit_value", Float),
    Column("status", String, nullable=False),
    Column("details_json", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("acknowledged_by", String),
    Column("acknowledged_at", DateTime(timezone=True)),
    Column("resolved_by", String),
    Column("resolved_at", DateTime(timezone=True)),
    Column("resolution_reason", Text),
)
Index(
    "idx_risk_events_portfolio_created", risk_events.c.portfolio_id, risk_events.c.created_at.desc()
)

portfolio_reviews = Table(
    "portfolio_reviews",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.paper_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "batch_id",
        String,
        ForeignKey("quantlab.portfolio_batches.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    ),
    Column("trade_date", Date, nullable=False),
    Column("status", String, nullable=False),
    Column("summary_json", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_portfolio_reviews_portfolio_date",
    portfolio_reviews.c.portfolio_id,
    portfolio_reviews.c.trade_date.desc(),
)

pair_paper_portfolios = Table(
    "pair_paper_portfolios",
    metadata,
    Column("id", String, primary_key=True),
    Column("is_legacy", Boolean, nullable=False, server_default="true"),
    Column("name", String, nullable=False, unique=True),
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("dataset", String, nullable=False),
    Column("execution_snapshot", String, nullable=False),
    Column("minute_dataset", String, nullable=False),
    Column("shortability_dataset", String, nullable=False),
    Column("dataset_roll_policy", String, nullable=False, server_default="pinned"),
    Column("dataset_lineage_id", String),
    Column("execution_roll_policy", String, nullable=False, server_default="pinned"),
    Column("execution_lineage_id", String),
    Column("status", String, nullable=False),
    Column("base_currency", String, nullable=False),
    Column("initial_cash", Numeric(20, 6), nullable=False),
    Column("cash", Numeric(20, 6), nullable=False),
    Column("nav", Numeric(20, 6), nullable=False),
    Column("high_water_mark", Numeric(20, 6), nullable=False),
    Column("position_direction", Integer, nullable=False),
    Column("quantity_y", BigInteger, nullable=False),
    Column("quantity_x", BigInteger, nullable=False),
    Column("entry_nav", Numeric(20, 6)),
    Column("holding_days", Integer, nullable=False),
    Column("last_signal_date", Date),
    Column("last_trade_date", Date),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_pair_paper_portfolios_status_updated",
    pair_paper_portfolios.c.status,
    pair_paper_portfolios.c.updated_at.desc(),
)

pair_portfolio_batches = Table(
    "pair_portfolio_batches",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.pair_paper_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("job_id", String, ForeignKey("quantlab.jobs.id", ondelete="SET NULL")),
    Column("as_of_date", Date, nullable=False),
    Column("trade_date", Date),
    Column("status", String, nullable=False),
    Column("idempotency_key", String, nullable=False, unique=True),
    Column("starting_state_sha256", String, nullable=False),
    Column("dataset", String),
    Column("dataset_identity_sha256", String),
    Column("dataset_lineage_id", String),
    Column("execution_snapshot", String),
    Column("execution_manifest_sha256", String),
    Column("execution_lineage_id", String),
    Column("artifact_path", Text),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    UniqueConstraint("portfolio_id", "as_of_date", name="uq_pair_batches_as_of"),
)
Index(
    "idx_pair_batches_status_created",
    pair_portfolio_batches.c.status,
    pair_portfolio_batches.c.created_at.desc(),
)

pair_paper_orders = Table(
    "pair_paper_orders",
    metadata,
    Column("id", String, primary_key=True),
    Column("is_legacy", Boolean, nullable=False, server_default="true"),
    Column(
        "batch_id",
        String,
        ForeignKey("quantlab.pair_portfolio_batches.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.pair_paper_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("leg", String, nullable=False),
    Column("instrument", String, nullable=False),
    Column("side", String, nullable=False),
    Column("requested_quantity", BigInteger, nullable=False),
    Column("target_quantity", BigInteger, nullable=False),
    Column("status", String, nullable=False),
    Column("reason", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("batch_id", "leg", name="uq_pair_orders_batch_leg"),
)
Index(
    "idx_pair_orders_portfolio_created",
    pair_paper_orders.c.portfolio_id,
    pair_paper_orders.c.created_at.desc(),
)

pair_paper_fills = Table(
    "pair_paper_fills",
    metadata,
    Column("id", String, primary_key=True),
    Column("is_legacy", Boolean, nullable=False, server_default="true"),
    Column(
        "order_id",
        String,
        ForeignKey("quantlab.pair_paper_orders.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    ),
    Column("fill_time", DateTime(timezone=True), nullable=False),
    Column("quantity", BigInteger, nullable=False),
    Column("price", Numeric(20, 6), nullable=False),
    Column("gross_value", Numeric(20, 6), nullable=False),
    Column("fee", Numeric(20, 6), nullable=False),
    Column("slippage", Float, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

pair_portfolio_nav = Table(
    "pair_portfolio_nav",
    metadata,
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.pair_paper_portfolios.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("trade_date", Date, primary_key=True),
    Column("cash", Numeric(20, 6), nullable=False),
    Column("long_value", Numeric(20, 6), nullable=False),
    Column("short_value", Numeric(20, 6), nullable=False),
    Column("nav", Numeric(20, 6), nullable=False),
    Column("daily_return", Float, nullable=False),
    Column("drawdown", Float, nullable=False),
    Column("gross_exposure", Float, nullable=False),
    Column("net_exposure", Float, nullable=False),
    Column("turnover", Float, nullable=False),
    Column("fees", Numeric(20, 6), nullable=False),
    Column("borrow_cost", Numeric(20, 6), nullable=False),
    Column("zscore", Float, nullable=False),
    Column("correlation", Float, nullable=False),
    Column("cointegration_pvalue", Float, nullable=False),
    Column("position_direction", Integer, nullable=False),
    Column("quantity_y", BigInteger, nullable=False),
    Column("quantity_x", BigInteger, nullable=False),
    Column("price_y", Numeric(20, 6), nullable=False),
    Column("price_x", Numeric(20, 6), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index("idx_pair_nav_trade_date", pair_portfolio_nav.c.trade_date.desc())

pair_portfolio_risk_events = Table(
    "pair_portfolio_risk_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.pair_paper_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "batch_id",
        String,
        ForeignKey("quantlab.pair_portfolio_batches.id", ondelete="CASCADE"),
    ),
    Column("severity", String, nullable=False),
    Column("event_type", String, nullable=False),
    Column("rule", String, nullable=False),
    Column("observed", Float),
    Column("limit_value", Float),
    Column("status", String, nullable=False),
    Column("details_json", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("acknowledged_by", String),
    Column("acknowledged_at", DateTime(timezone=True)),
    Column("resolved_by", String),
    Column("resolved_at", DateTime(timezone=True)),
    Column("resolution_reason", Text),
)
Index(
    "idx_pair_risk_events_portfolio_created",
    pair_portfolio_risk_events.c.portfolio_id,
    pair_portfolio_risk_events.c.created_at.desc(),
)

pair_portfolio_reviews = Table(
    "pair_portfolio_reviews",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.pair_paper_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "batch_id",
        String,
        ForeignKey("quantlab.pair_portfolio_batches.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    ),
    Column("trade_date", Date, nullable=False),
    Column("status", String, nullable=False),
    Column("summary_json", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_pair_reviews_portfolio_date",
    pair_portfolio_reviews.c.portfolio_id,
    pair_portfolio_reviews.c.trade_date.desc(),
)

strategy_allocations = Table(
    "strategy_allocations",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False, unique=True),
    Column("dataset", String, nullable=False),
    Column("status", String, nullable=False),
    Column("is_legacy", Boolean, nullable=False, server_default="false"),
    Column("allocation_method", String, nullable=False),
    Column("lookback_days", Integer, nullable=False),
    Column("target_volatility", Float, nullable=False),
    Column("max_pairwise_correlation", Float, nullable=False),
    Column("max_strategy_weight", Float, nullable=False),
    Column("max_member_drawdown", Float, nullable=False),
    Column("max_drawdown_reduce", Float, nullable=False),
    Column("max_drawdown_liquidate", Float, nullable=False),
    Column("total_capital", Numeric(20, 6), nullable=False),
    Column("cash_reserve", Numeric(20, 6), nullable=False),
    Column("nav", Numeric(20, 6), nullable=False),
    Column("high_water_mark", Numeric(20, 6), nullable=False),
    Column("analysis_json", json_type, nullable=False),
    Column("created_by", String, nullable=False),
    Column("approved_by", String),
    Column("approval_reason", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("approved_at", DateTime(timezone=True)),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_strategy_allocations_status_updated",
    strategy_allocations.c.status,
    strategy_allocations.c.updated_at.desc(),
)

strategy_allocation_members = Table(
    "strategy_allocation_members",
    metadata,
    Column(
        "allocation_id",
        String,
        ForeignKey("quantlab.strategy_allocations.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column(
        "backtest_id",
        String,
        ForeignKey("quantlab.backtest_runs.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.paper_portfolios.id", ondelete="RESTRICT"),
        unique=True,
    ),
    Column(
        "recommendation_portfolio_id",
        String,
        ForeignKey("quantlab.recommendation_portfolios.id", ondelete="RESTRICT"),
        unique=True,
    ),
    Column("target_weight", Float, nullable=False),
    Column("role", String, nullable=False, server_default="core"),
    Column("risk_budget", Float, nullable=False, server_default="1"),
    Column("member_cap", Float, nullable=False, server_default="0.70"),
    Column("annualized_volatility", Float, nullable=False),
    Column("risk_contribution", Float, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_strategy_allocation_members_portfolio",
    strategy_allocation_members.c.portfolio_id,
)

strategy_allocation_nav = Table(
    "strategy_allocation_nav",
    metadata,
    Column(
        "allocation_id",
        String,
        ForeignKey("quantlab.strategy_allocations.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("trade_date", Date, primary_key=True),
    Column("nav", Numeric(20, 6), nullable=False),
    Column("daily_return", Float, nullable=False),
    Column("annualized_volatility", Float, nullable=False),
    Column("drawdown", Float, nullable=False),
    Column("member_nav_json", json_type, nullable=False),
    Column("member_weights_json", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index("idx_strategy_allocation_nav_date", strategy_allocation_nav.c.trade_date.desc())

strategy_allocation_events = Table(
    "strategy_allocation_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "allocation_id",
        String,
        ForeignKey("quantlab.strategy_allocations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.paper_portfolios.id", ondelete="SET NULL"),
    ),
    Column(
        "recommendation_portfolio_id",
        String,
        ForeignKey("quantlab.recommendation_portfolios.id", ondelete="SET NULL"),
    ),
    Column("severity", String, nullable=False),
    Column("event_type", String, nullable=False),
    Column("rule", String, nullable=False),
    Column("observed", Float),
    Column("limit_value", Float),
    Column("status", String, nullable=False),
    Column("details_json", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("acknowledged_by", String),
    Column("acknowledged_at", DateTime(timezone=True)),
    Column("resolved_by", String),
    Column("resolved_at", DateTime(timezone=True)),
    Column("resolution_reason", Text),
)
Index(
    "idx_strategy_allocation_events_created",
    strategy_allocation_events.c.allocation_id,
    strategy_allocation_events.c.created_at.desc(),
)

system_health_snapshots = Table(
    "system_health_snapshots",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("status", String, nullable=False),
    Column("components_json", json_type, nullable=False),
    Column("summary_json", json_type, nullable=False),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
)
Index("idx_system_health_recorded", system_health_snapshots.c.recorded_at.desc())

broker_destinations = Table(
    "broker_destinations",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False, unique=True),
    Column("adapter", String, nullable=False),
    Column("environment", String, nullable=False),
    Column("account_ref", String, nullable=False),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.paper_portfolios.id", ondelete="RESTRICT"),
    ),
    Column("status", String, nullable=False),
    Column("config_json", json_type, nullable=False),
    Column("activation_requested_by", String),
    Column("activation_requested_at", DateTime(timezone=True)),
    Column("activated_by", String),
    Column("activated_at", DateTime(timezone=True)),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_broker_destinations_status_updated",
    broker_destinations.c.status,
    broker_destinations.c.updated_at.desc(),
)
Index(
    "uq_broker_destinations_portfolio",
    broker_destinations.c.portfolio_id,
    unique=True,
    postgresql_where=broker_destinations.c.portfolio_id.is_not(None),
)

broker_order_outbox = Table(
    "broker_order_outbox",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "destination_id",
        String,
        ForeignKey("quantlab.broker_destinations.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.paper_portfolios.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "batch_id",
        String,
        ForeignKey("quantlab.portfolio_batches.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "source_order_id",
        String,
        ForeignKey("quantlab.paper_orders.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("idempotency_key", String, nullable=False, unique=True),
    Column("payload_json", json_type, nullable=False),
    Column("payload_sha256", String, nullable=False),
    Column("status", String, nullable=False),
    Column("attempts", Integer, nullable=False),
    Column("broker_order_id", String),
    Column("created_by", String, nullable=False),
    Column("approved_by", String),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("approved_at", DateTime(timezone=True)),
    Column("submitted_at", DateTime(timezone=True)),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("last_error", Text),
    UniqueConstraint(
        "destination_id",
        "source_order_id",
        name="uq_broker_outbox_destination_source_order",
    ),
)
Index(
    "idx_broker_outbox_status_updated",
    broker_order_outbox.c.status,
    broker_order_outbox.c.updated_at.desc(),
)

broker_events = Table(
    "broker_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "destination_id",
        String,
        ForeignKey("quantlab.broker_destinations.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("outbox_id", String, ForeignKey("quantlab.broker_order_outbox.id", ondelete="SET NULL")),
    Column("event_type", String, nullable=False),
    Column("actor", String, nullable=False),
    Column("details_json", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_broker_events_destination_created",
    broker_events.c.destination_id,
    broker_events.c.created_at.desc(),
)

broker_reconciliations = Table(
    "broker_reconciliations",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "destination_id",
        String,
        ForeignKey("quantlab.broker_destinations.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("status", String, nullable=False),
    Column("broker_as_of", DateTime(timezone=True)),
    Column("expected_json", json_type, nullable=False),
    Column("observed_json", json_type, nullable=False),
    Column("differences_json", json_type, nullable=False),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_broker_reconciliations_destination_created",
    broker_reconciliations.c.destination_id,
    broker_reconciliations.c.created_at.desc(),
)

broker_gateway_parents = Table(
    "broker_gateway_parents",
    metadata,
    Column("id", String, primary_key=True),
    Column("client_order_id", String, nullable=False, unique=True),
    Column("account_ref", String, nullable=False),
    Column("environment", String, nullable=False),
    Column("provider", String, nullable=False),
    Column("payload_json", json_type, nullable=False),
    Column("payload_sha256", String, nullable=False),
    Column("status", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("last_error", Text),
)
Index(
    "idx_broker_gateway_parents_status_updated",
    broker_gateway_parents.c.status,
    broker_gateway_parents.c.updated_at.desc(),
)

broker_gateway_children = Table(
    "broker_gateway_children",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "parent_id",
        String,
        ForeignKey("quantlab.broker_gateway_parents.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("slice_index", Integer, nullable=False),
    Column("scheduled_for", DateTime(timezone=True), nullable=False),
    Column("quantity", Numeric(20, 6), nullable=False),
    Column("limit_price", Numeric(20, 6), nullable=False),
    Column("client_tag", String, nullable=False, unique=True),
    Column("provider_order_id", String),
    Column("status", String, nullable=False),
    Column("submitted_at", DateTime(timezone=True)),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("last_error", Text),
    Column("filled_quantity", Numeric(20, 6), nullable=False, server_default="0"),
    Column("replacement_count", Integer, nullable=False, server_default="0"),
    Column("market_evidence_json", json_type, nullable=False, server_default="{}"),
    Column("cancel_requested_at", DateTime(timezone=True)),
    UniqueConstraint("parent_id", "slice_index", name="uq_broker_gateway_parent_slice"),
)
Index(
    "idx_broker_gateway_children_due",
    broker_gateway_children.c.status,
    broker_gateway_children.c.scheduled_for,
)

broker_gateway_nonces = Table(
    "broker_gateway_nonces",
    metadata,
    Column("nonce", String, primary_key=True),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index("idx_broker_gateway_nonces_expiry", broker_gateway_nonces.c.expires_at)

broker_gateway_attempts = Table(
    "broker_gateway_attempts",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "child_id",
        String,
        ForeignKey("quantlab.broker_gateway_children.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("attempt_no", Integer, nullable=False),
    Column("client_tag", String, nullable=False, unique=True),
    Column("provider_order_id", String),
    Column("quantity", Numeric(20, 6), nullable=False),
    Column("limit_price", Numeric(20, 6), nullable=False),
    Column("traded_quantity", Numeric(20, 6), nullable=False, server_default="0"),
    Column("status", String, nullable=False),
    Column("market_evidence_json", json_type, nullable=False),
    Column("submitted_at", DateTime(timezone=True)),
    Column("cancel_requested_at", DateTime(timezone=True)),
    Column("completed_at", DateTime(timezone=True)),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("last_error", Text),
    UniqueConstraint("child_id", "attempt_no", name="uq_broker_gateway_child_attempt"),
)
Index(
    "idx_broker_gateway_attempts_status_updated",
    broker_gateway_attempts.c.status,
    broker_gateway_attempts.c.updated_at,
)

broker_gateway_events = Table(
    "broker_gateway_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("event_type", String, nullable=False),
    Column("provider_order_id", String),
    Column("client_tag", String),
    Column("payload_json", json_type, nullable=False),
    Column("received_at", DateTime(timezone=True), nullable=False),
)
Index("idx_broker_gateway_events_received", broker_gateway_events.c.received_at.desc())

schedules = Table(
    "schedules",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False, unique=True),
    Column("kind", String, nullable=False),
    Column("status", String, nullable=False),
    Column("desired_status", String, nullable=False),
    Column("suspension_reason", Text),
    Column("timezone", String, nullable=False),
    Column("run_time", Time, nullable=False),
    Column("trading_days_only", Boolean, nullable=False),
    Column("payload_json", json_type, nullable=False),
    Column("misfire_grace_seconds", Integer, nullable=False),
    Column("next_run_at", DateTime(timezone=True), nullable=False),
    Column("last_run_at", DateTime(timezone=True)),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index("idx_schedules_due", schedules.c.status, schedules.c.next_run_at)

allocation_schedule_groups = Table(
    "allocation_schedule_groups",
    metadata,
    Column(
        "allocation_id",
        String,
        ForeignKey("quantlab.strategy_allocations.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("status", String, nullable=False),
    Column("timezone", String, nullable=False),
    Column("run_time", Time, nullable=False),
    Column("trading_days_only", Boolean, nullable=False),
    Column("slippage", Float, nullable=False),
    Column("misfire_grace_seconds", Integer, nullable=False),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index("idx_allocation_schedule_groups_status", allocation_schedule_groups.c.status)

allocation_schedule_members = Table(
    "allocation_schedule_members",
    metadata,
    Column(
        "allocation_id",
        String,
        ForeignKey("quantlab.allocation_schedule_groups.allocation_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.paper_portfolios.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column(
        "schedule_id",
        String,
        ForeignKey("quantlab.schedules.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    ),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_allocation_schedule_members_schedule",
    allocation_schedule_members.c.schedule_id,
)

schedule_runs = Table(
    "schedule_runs",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "schedule_id",
        String,
        ForeignKey("quantlab.schedules.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("job_id", String, ForeignKey("quantlab.jobs.id", ondelete="SET NULL")),
    Column("scheduled_for", DateTime(timezone=True), nullable=False),
    Column("status", String, nullable=False),
    Column("attempts", Integer, nullable=False),
    Column("lease_until", DateTime(timezone=True)),
    Column("dedupe_key", String, nullable=False, unique=True),
    Column("message", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    UniqueConstraint("schedule_id", "scheduled_for", name="uq_schedule_runs_slot"),
)
Index("idx_schedule_runs_created", schedule_runs.c.created_at.desc())

alerts = Table(
    "alerts",
    metadata,
    Column("id", String, primary_key=True),
    Column("source_type", String, nullable=False),
    Column("source_id", String, nullable=False),
    Column("severity", String, nullable=False),
    Column("category", String, nullable=False),
    Column("title", String, nullable=False),
    Column("message", Text, nullable=False),
    Column("status", String, nullable=False),
    Column("dedupe_key", String, nullable=False, unique=True),
    Column("details_json", json_type, nullable=False),
    Column("delivery_status", String, nullable=False),
    Column("delivery_attempts", Integer, nullable=False),
    Column("delivered_at", DateTime(timezone=True)),
    Column("last_delivery_error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("acknowledged_by", String),
    Column("acknowledged_at", DateTime(timezone=True)),
    Column("resolved_by", String),
    Column("resolved_at", DateTime(timezone=True)),
)
Index("idx_alerts_status_created", alerts.c.status, alerts.c.created_at.desc())

users = Table(
    "users",
    metadata,
    Column("id", String, primary_key=True),
    Column("username", String, nullable=False, unique=True),
    Column("display_name", String, nullable=False),
    Column("role", String, nullable=False),
    Column("password_hash", Text, nullable=False),
    Column("active", Boolean, nullable=False),
    Column("failed_login_attempts", Integer, nullable=False),
    Column("locked_until", DateTime(timezone=True)),
    Column("last_login_at", DateTime(timezone=True)),
    Column("password_changed_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index("idx_users_role_active", users.c.role, users.c.active)

auth_sessions = Table(
    "auth_sessions",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "user_id",
        String,
        ForeignKey("quantlab.users.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("token_hash", String, nullable=False, unique=True),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("last_seen_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("revoked_at", DateTime(timezone=True)),
    Column("ip_hash", String),
    Column("user_agent", String),
)
Index("idx_auth_sessions_user_expires", auth_sessions.c.user_id, auth_sessions.c.expires_at)

audit_events = Table(
    "audit_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("user_id", String, ForeignKey("quantlab.users.id", ondelete="SET NULL")),
    Column("username", String, nullable=False),
    Column("action", String, nullable=False),
    Column("method", String, nullable=False),
    Column("path", String, nullable=False),
    Column("status_code", Integer, nullable=False),
    Column("ip_hash", String),
    Column("user_agent", String),
    Column("details_json", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index("idx_audit_events_created", audit_events.c.created_at.desc())
Index("idx_audit_events_user_created", audit_events.c.user_id, audit_events.c.created_at.desc())

runtime_secrets = Table(
    "runtime_secrets",
    metadata,
    Column("name", String, primary_key=True),
    Column("ciphertext", Text, nullable=False),
    Column("metadata_json", json_type, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("updated_by", String, ForeignKey("quantlab.users.id", ondelete="SET NULL")),
)

platform_configs = Table(
    "platform_configs",
    metadata,
    Column("key", String, primary_key=True),
    Column("revision", Integer, nullable=False),
    Column("value_json", json_type, nullable=False),
    Column("updated_by", String, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

platform_config_revisions = Table(
    "platform_config_revisions",
    metadata,
    Column("key", String, primary_key=True),
    Column("revision", Integer, primary_key=True),
    Column("value_json", json_type, nullable=False),
    Column("reason", Text, nullable=False),
    Column("updated_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_platform_config_revisions_created",
    platform_config_revisions.c.created_at.desc(),
)


def open_database(database_url: str) -> Engine:
    if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise ValueError("DATABASE_URL must point to PostgreSQL")
    engine = create_engine(database_url, pool_pre_ping=True, pool_size=5, max_overflow=10)
    return engine


def row_dict(row: Any) -> dict[str, Any]:
    result = dict(row._mapping if hasattr(row, "_mapping") else row)
    for key, value in tuple(result.items()):
        if isinstance(value, datetime):
            result[key] = value.isoformat(timespec="seconds")
        elif isinstance(value, Decimal):
            result[key] = float(value)
    return result
