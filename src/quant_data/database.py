# One governed recovery constraint below embeds an exact byte-for-byte JSON receipt.
# ruff: noqa: E501

from __future__ import annotations

import os
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
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

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
Index(
    "idx_work_units_dataset_page_group",
    work_units.c.dataset,
    work_units.c.scope_json["page_group"].as_string(),
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

factor_definitions = Table(
    "factor_definitions",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("expression", Text, nullable=False),
    Column("expression_sha256", String, nullable=False, unique=True),
    Column("definition_sha256", String, nullable=False, unique=True),
    Column("required_fields_json", json_type, nullable=False),
    Column("max_lookback_days", Integer, nullable=False),
    Column("economic_family", String, nullable=False),
    Column("family_tags_json", json_type, nullable=False),
    Column("aliases_json", json_type, nullable=False),
    Column("source_refs_json", json_type, nullable=False),
    Column("availability_policy", String, nullable=False),
    Column("qlib_commit", String, nullable=False),
    Column("status", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_factor_definitions_family_name",
    factor_definitions.c.economic_family,
    factor_definitions.c.name,
)

factor_library_versions = Table(
    "factor_library_versions",
    metadata,
    Column("id", String, primary_key=True),
    Column("contract_version", String, nullable=False),
    Column("definition_sha256", String, nullable=False, unique=True),
    Column("member_count", Integer, nullable=False),
    Column("source_alias_counts_json", json_type, nullable=False),
    Column("qlib_commit", String, nullable=False),
    Column("status", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("retired_at", DateTime(timezone=True)),
)
Index(
    "uq_factor_library_versions_active",
    factor_library_versions.c.status,
    unique=True,
    postgresql_where=factor_library_versions.c.status == "active",
)

factor_library_members = Table(
    "factor_library_members",
    metadata,
    Column(
        "library_version_id",
        String,
        ForeignKey("quantlab.factor_library_versions.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column(
        "factor_definition_id",
        String,
        ForeignKey("quantlab.factor_definitions.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column("ordinal", Integer, nullable=False),
    UniqueConstraint("library_version_id", "ordinal", name="uq_factor_library_member_ordinal"),
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
        Column(
            "factor_definition_id",
            String,
            ForeignKey("quantlab.factor_definitions.id", ondelete="RESTRICT"),
        ),
        Column("economic_family", String),
        Column("family_tags_json", json_type),
        Column("similarity_cluster_id", String),
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
        Column("profile_consensus_json", json_type),
        Column("profile_consensus_sha256", String),
        Column("promoted_evaluation_id", String),
        Column("promotion_evidence_sha256", String),
        Column("admission_path", String),
        Column("incremental_evidence_json", json_type),
        Column("incremental_evidence_sha256", String),
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
Index("idx_factor_candidates_definition", factor_candidates.c.factor_definition_id)
Index(
    "idx_factor_candidates_family_status",
    factor_candidates.c.economic_family,
    factor_candidates.c.status,
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

factor_similarity_edges = Table(
    "factor_similarity_edges",
    metadata,
    Column("id", String, primary_key=True),
    Column("dataset_identity_sha256", String, nullable=False),
    Column("profile_id", String, nullable=False),
    Column(
        "left_factor_candidate_id",
        String,
        ForeignKey("quantlab.factor_candidates.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "right_factor_candidate_id",
        String,
        ForeignKey("quantlab.factor_candidates.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("mean_abs_spearman", Float, nullable=False),
    Column("relationship", String, nullable=False),
    Column("cluster_id", String),
    Column("evidence_json", json_type, nullable=False),
    Column("evidence_sha256", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "dataset_identity_sha256",
        "profile_id",
        "left_factor_candidate_id",
        "right_factor_candidate_id",
        name="uq_factor_similarity_edge",
    ),
)

factor_definition_similarity_edges = Table(
    "factor_definition_similarity_edges",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "library_version_id",
        String,
        ForeignKey("quantlab.factor_library_versions.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("dataset_identity_sha256", String, nullable=False),
    Column(
        "left_factor_definition_id",
        String,
        ForeignKey("quantlab.factor_definitions.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "right_factor_definition_id",
        String,
        ForeignKey("quantlab.factor_definitions.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("mean_abs_spearman", Float, nullable=False),
    Column("relationship", String, nullable=False),
    Column("cluster_id", String, nullable=False),
    Column("evidence_json", json_type, nullable=False),
    Column("evidence_sha256", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "library_version_id",
        "dataset_identity_sha256",
        "left_factor_definition_id",
        "right_factor_definition_id",
        name="uq_factor_definition_similarity_edge",
    ),
)

research_sota_versions = Table(
    "research_sota_versions",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "library_version_id",
        String,
        ForeignKey("quantlab.factor_library_versions.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "predecessor_id",
        String,
        ForeignKey("quantlab.research_sota_versions.id", ondelete="RESTRICT"),
    ),
    Column("dataset", String, nullable=False),
    Column("dataset_identity_sha256", String, nullable=False),
    Column("universe", String, nullable=False),
    Column("label_horizon_days", Integer, nullable=False),
    Column("periods_json", json_type, nullable=False),
    Column("policy_json", json_type, nullable=False),
    Column("policy_sha256", String, nullable=False),
    Column("evidence_json", json_type, nullable=False),
    Column("evidence_sha256", String, nullable=False),
    Column("status", String, nullable=False),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("activated_at", DateTime(timezone=True), nullable=False),
    Column("superseded_at", DateTime(timezone=True)),
)
Index(
    "uq_research_sota_active_scope",
    research_sota_versions.c.dataset,
    research_sota_versions.c.universe,
    research_sota_versions.c.label_horizon_days,
    unique=True,
    postgresql_where=research_sota_versions.c.status == "active",
)

research_sota_members = Table(
    "research_sota_members",
    metadata,
    Column(
        "sota_version_id",
        String,
        ForeignKey("quantlab.research_sota_versions.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column(
        "factor_candidate_id",
        String,
        ForeignKey("quantlab.factor_candidates.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column(
        "factor_definition_id",
        String,
        ForeignKey("quantlab.factor_definitions.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "factor_evaluation_id",
        String,
        ForeignKey("quantlab.factor_evaluations.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("economic_family", String, nullable=False),
    Column("family_tags_json", json_type, nullable=False),
    Column("similarity_cluster_id", String, nullable=False),
    Column("member_rank", Integer, nullable=False),
    Column("weight", Float),
    Column("action", String, nullable=False),
    Column("replaced_factor_candidate_id", String),
    Column("incremental_evidence_json", json_type, nullable=False),
    Column("incremental_evidence_sha256", String, nullable=False),
    UniqueConstraint("sota_version_id", "member_rank", name="uq_research_sota_member_rank"),
)

# Sealed one-shot consumption ledger for reserved final out-of-sample windows.
# Scope is stable across snapshot identities: a capital alpha family (whose
# immutable mandate contains its label horizon), research program, verified
# dataset lineage, or the fail-closed global standalone scope. Dataset identity
# remains immutable audit evidence, but it never grants a fresh OOS window.
oos_vintages = Table(
    "oos_vintages",
    metadata,
    Column("id", String, primary_key=True),
    Column("scope", String, nullable=False),
    Column("dataset_identity", String, nullable=False),
    Column("dataset_lineage_id", String),
    Column("test_start", Date, nullable=False),
    Column("test_end", Date, nullable=False),
    Column("sealed_at", DateTime(timezone=True), nullable=False),
    Column("first_opened_at", DateTime(timezone=True), nullable=False),
    Column("consumed_at", DateTime(timezone=True)),
    Column(
        "capital_oos_alpha_batch_id",
        String,
        ForeignKey("quantlab.capital_oos_alpha_batches.id", ondelete="RESTRICT"),
        unique=True,
    ),
    Column("sealed_candidate_set_json", json_type, nullable=False),
    Column("sealed_candidate_set_sha256", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "scope",
        "dataset_identity",
        "test_start",
        "test_end",
        name="uq_oos_vintage_window",
    ),
)
Index(
    "idx_oos_vintage_scope_window",
    oos_vintages.c.scope,
    oos_vintages.c.test_start,
    oos_vintages.c.test_end,
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

# Immutable inputs made available to one or more research runs.  These are
# deliberately separate from generated run artifacts: a PDF, paper, dataset
# bundle, or benchmark definition is source evidence, never a capital-bearing
# candidate by itself.
research_assets = Table(
    "research_assets",
    metadata,
    Column("id", String, primary_key=True),
    Column("asset_key", String, nullable=False, unique=True),
    Column("asset_type", String, nullable=False),
    Column("media_type", String, nullable=False),
    Column("source_uri", Text),
    Column("publisher", String),
    Column("published_at", DateTime(timezone=True)),
    Column("retrieved_at", DateTime(timezone=True), nullable=False),
    Column("license_json", json_type, nullable=False),
    Column("storage_path", Text, nullable=False),
    Column("content_sha256", String, nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    Column("manifest_json", json_type, nullable=False),
    Column("manifest_sha256", String, nullable=False),
    Column("status", String, nullable=False),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("quarantined_at", DateTime(timezone=True)),
    Column("quarantine_reason", Text),
    UniqueConstraint(
        "content_sha256",
        "manifest_sha256",
        name="uq_research_assets_content_manifest",
    ),
    CheckConstraint(
        "status IN ('registered', 'quarantined', 'retired')",
        name="ck_research_assets_status",
    ),
    CheckConstraint("size_bytes >= 0", name="ck_research_assets_size"),
)
Index(
    "idx_research_assets_type_created",
    research_assets.c.asset_type,
    research_assets.c.created_at.desc(),
)

# Auto-selected documents are single-use research inputs.  This durable
# reservation ledger prevents two API/scheduler requests from silently using
# the same report or paper while keeping explicit operator-selected reuse a
# separate, auditable decision.
research_asset_consumptions = Table(
    "research_asset_consumptions",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "asset_id",
        String,
        ForeignKey("quantlab.research_assets.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "research_run_id",
        String,
        ForeignKey("quantlab.research_runs.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("scenario", String, nullable=False),
    Column("selection_mode", String, nullable=False),
    Column("asset_manifest_sha256", String, nullable=False),
    Column("status", String, nullable=False),
    Column("reserved_by", String, nullable=False),
    Column("reserved_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    Column("details_json", json_type, nullable=False),
    UniqueConstraint("asset_id", name="uq_research_asset_consumptions_asset"),
    UniqueConstraint(
        "research_run_id",
        "asset_id",
        name="uq_research_asset_consumptions_run_asset",
    ),
    CheckConstraint(
        "selection_mode IN ('automatic')",
        name="ck_research_asset_consumptions_mode",
    ),
    CheckConstraint(
        "status IN ('reserved', 'consumed', 'failed')",
        name="ck_research_asset_consumptions_status",
    ),
)
Index(
    "idx_research_asset_consumptions_run",
    research_asset_consumptions.c.research_run_id,
    research_asset_consumptions.c.reserved_at.desc(),
)

# Generated files are registered before a candidate or an evaluation may
# reference them.  Their file and canonical-manifest hashes are immutable;
# invalidation changes status only and never rewrites the evidence.
research_run_artifacts = Table(
    "research_run_artifacts",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "research_run_id",
        String,
        ForeignKey("quantlab.research_runs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("artifact_type", String, nullable=False),
    Column("contract_version", String, nullable=False),
    Column("status", String, nullable=False),
    Column("storage_path", Text, nullable=False),
    Column("content_sha256", String, nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    Column("manifest_json", json_type, nullable=False),
    Column("manifest_sha256", String, nullable=False),
    Column("producer", String, nullable=False),
    Column("source_iteration", Integer),
    Column("capital_eligible", Boolean, nullable=False, server_default="false"),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("invalidated_by", String),
    Column("invalidated_at", DateTime(timezone=True)),
    Column("invalidation_reason", Text),
    UniqueConstraint(
        "research_run_id",
        "artifact_type",
        "content_sha256",
        name="uq_research_run_artifacts_content",
    ),
    CheckConstraint(
        "status IN ('recorded', 'invalidated')",
        name="ck_research_run_artifacts_status",
    ),
    CheckConstraint("size_bytes >= 0", name="ck_research_run_artifacts_size"),
    CheckConstraint(
        "capital_eligible = false",
        name="ck_research_run_artifacts_non_capital",
    ),
)
Index(
    "idx_research_run_artifacts_run_created",
    research_run_artifacts.c.research_run_id,
    research_run_artifacts.c.created_at.desc(),
)

# RD-Agent model output is a research proposal, not a fitted ModelArtifact.
# Admission here means "eligible for governed downstream research" only.
model_candidates = Table(
    "model_candidates",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "research_run_id",
        String,
        ForeignKey("quantlab.research_runs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("name", String, nullable=False),
    Column("description", Text, nullable=False),
    Column("status", String, nullable=False),
    Column("source_iteration", Integer),
    Column("model_type", String, nullable=False),
    Column(
        "code_artifact_id",
        String,
        ForeignKey("quantlab.research_run_artifacts.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("code_sha256", String, nullable=False),
    Column("architecture_json", json_type, nullable=False),
    Column("model_hyperparameters_json", json_type, nullable=False),
    Column("training_hyperparameters_json", json_type, nullable=False),
    Column("base_features_manifest_json", json_type, nullable=False),
    Column("base_features_manifest_sha256", String, nullable=False),
    Column("feature_set_definition_sha256", String, nullable=False),
    Column("dataset", String, nullable=False),
    Column("dataset_identity_sha256", String, nullable=False),
    Column("pre_final_end", Date, nullable=False),
    Column("final_oos_start", Date, nullable=False),
    Column("final_oos_end", Date, nullable=False),
    Column("manifest_json", json_type, nullable=False),
    Column("manifest_sha256", String, nullable=False),
    Column("rdagent_decision", Boolean),
    Column("rdagent_feedback", Text),
    Column("admission_evidence_json", json_type),
    Column("admission_evidence_sha256", String),
    Column("capital_eligible", Boolean, nullable=False, server_default="false"),
    Column("admitted_by", String),
    Column("admitted_at", DateTime(timezone=True)),
    Column("rejection_reason", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "research_run_id",
        "name",
        name="uq_model_candidates_run_name",
    ),
    CheckConstraint(
        "status IN ('awaiting_independent_evaluation', 'evaluating', "
        "'evaluated', 'research_admitted', 'rejected', 'invalidated')",
        name="ck_model_candidates_status",
    ),
    CheckConstraint(
        "capital_eligible = false",
        name="ck_model_candidates_non_capital",
    ),
    CheckConstraint(
        "pre_final_end < final_oos_start AND final_oos_start <= final_oos_end",
        name="ck_model_candidates_oos_boundary",
    ),
)
Index(
    "idx_model_candidates_status_updated",
    model_candidates.c.status,
    model_candidates.c.updated_at.desc(),
)

model_evaluations = Table(
    "model_evaluations",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "model_candidate_id",
        String,
        ForeignKey("quantlab.model_candidates.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "run_artifact_id",
        String,
        ForeignKey("quantlab.research_run_artifacts.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "oos_vintage_id",
        String,
        ForeignKey("quantlab.oos_vintages.id", ondelete="RESTRICT"),
    ),
    Column("evidence_role", String, nullable=False),
    Column("profile_id", String, nullable=False),
    Column("seed", Integer, nullable=False),
    Column("dataset", String, nullable=False),
    Column("dataset_identity_sha256", String, nullable=False),
    Column("train_start", Date, nullable=False),
    Column("train_end", Date, nullable=False),
    Column("valid_start", Date, nullable=False),
    Column("valid_end", Date, nullable=False),
    Column("final_oos_start", Date, nullable=False),
    Column("final_oos_end", Date, nullable=False),
    Column("metrics_json", json_type, nullable=False),
    Column("metrics_sha256", String, nullable=False),
    Column("gate_status", String, nullable=False),
    Column("gate_reasons_json", json_type, nullable=False),
    Column("evaluator_version", String, nullable=False),
    Column("candidate_manifest_sha256", String, nullable=False),
    Column("evidence_json", json_type, nullable=False),
    Column("evidence_sha256", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "model_candidate_id",
        "evidence_role",
        "profile_id",
        "seed",
        name="uq_model_evaluations_grid",
    ),
    CheckConstraint(
        "evidence_role IN ('official_feedback', 'independent_gate')",
        name="ck_model_evaluations_role",
    ),
    CheckConstraint(
        "gate_status IN ('informational', 'passed', 'failed')",
        name="ck_model_evaluations_gate",
    ),
    CheckConstraint(
        "(evidence_role = 'official_feedback' AND gate_status = 'informational') OR "
        "(evidence_role = 'independent_gate' AND gate_status IN ('passed', 'failed'))",
        name="ck_model_evaluations_role_gate",
    ),
    CheckConstraint(
        "oos_vintage_id IS NULL",
        name="ck_model_evaluations_pre_final",
    ),
)
Index(
    "idx_model_evaluations_candidate_created",
    model_evaluations.c.model_candidate_id,
    model_evaluations.c.created_at.desc(),
)

# A quant bundle is the exact accepted factor set, active model, base features,
# and their immutable hashes.  It cannot exist as a model-only or factor-only
# shell; those forms are evaluation ablations, not publishable bundles.
quant_bundle_candidates = Table(
    "quant_bundle_candidates",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "research_run_id",
        String,
        ForeignKey("quantlab.research_runs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("name", String, nullable=False),
    Column("description", Text, nullable=False),
    Column("status", String, nullable=False),
    Column("source_iteration", Integer),
    Column(
        "model_candidate_id",
        String,
        ForeignKey("quantlab.model_candidates.id", ondelete="RESTRICT"),
    ),
    Column(
        "model_ensemble_candidate_id",
        String,
        ForeignKey("quantlab.model_ensemble_candidates.id", ondelete="RESTRICT"),
    ),
    Column("factor_candidate_ids_json", json_type, nullable=False),
    Column(
        "bundle_artifact_id",
        String,
        ForeignKey("quantlab.research_run_artifacts.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("bundle_artifact_sha256", String, nullable=False),
    Column("base_features_manifest_json", json_type, nullable=False),
    Column("base_features_manifest_sha256", String, nullable=False),
    Column("feature_set_definition_sha256", String, nullable=False),
    Column("dataset", String, nullable=False),
    Column("dataset_identity_sha256", String, nullable=False),
    Column("pre_final_end", Date, nullable=False),
    Column("final_oos_start", Date, nullable=False),
    Column("final_oos_end", Date, nullable=False),
    Column("bundle_manifest_json", json_type, nullable=False),
    Column("bundle_manifest_sha256", String, nullable=False),
    Column("rdagent_decision", Boolean),
    Column("rdagent_feedback", Text),
    Column("ablation_evidence_json", json_type),
    Column("ablation_evidence_sha256", String),
    Column("admission_evidence_json", json_type),
    Column("admission_evidence_sha256", String),
    Column("capital_eligible", Boolean, nullable=False, server_default="false"),
    Column("admitted_by", String),
    Column("admitted_at", DateTime(timezone=True)),
    Column("rejection_reason", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "research_run_id",
        "name",
        name="uq_quant_bundle_candidates_run_name",
    ),
    CheckConstraint(
        "status IN ('awaiting_independent_evaluation', 'evaluating', "
        "'evaluated', 'research_admitted', 'rejected', 'invalidated')",
        name="ck_quant_bundle_candidates_status",
    ),
    CheckConstraint(
        "capital_eligible = false",
        name="ck_quant_bundle_candidates_non_capital",
    ),
    CheckConstraint(
        "pre_final_end < final_oos_start AND final_oos_start <= final_oos_end",
        name="ck_quant_bundle_candidates_oos_boundary",
    ),
    CheckConstraint(
        "(model_candidate_id IS NOT NULL AND model_ensemble_candidate_id IS NULL) OR "
        "(model_candidate_id IS NULL AND model_ensemble_candidate_id IS NOT NULL)",
        name="ck_quant_bundle_candidates_prediction_xor",
    ),
)
Index(
    "idx_quant_bundle_candidates_status_updated",
    quant_bundle_candidates.c.status,
    quant_bundle_candidates.c.updated_at.desc(),
)

quant_bundle_evaluations = Table(
    "quant_bundle_evaluations",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "quant_bundle_candidate_id",
        String,
        ForeignKey("quantlab.quant_bundle_candidates.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "run_artifact_id",
        String,
        ForeignKey("quantlab.research_run_artifacts.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "oos_vintage_id",
        String,
        ForeignKey("quantlab.oos_vintages.id", ondelete="RESTRICT"),
    ),
    Column("evidence_role", String, nullable=False),
    Column("ablation", String, nullable=False),
    Column("profile_id", String, nullable=False),
    Column("seed", Integer, nullable=False),
    Column("dataset", String, nullable=False),
    Column("dataset_identity_sha256", String, nullable=False),
    Column("train_start", Date, nullable=False),
    Column("train_end", Date, nullable=False),
    Column("valid_start", Date, nullable=False),
    Column("valid_end", Date, nullable=False),
    Column("final_oos_start", Date, nullable=False),
    Column("final_oos_end", Date, nullable=False),
    Column("metrics_json", json_type, nullable=False),
    Column("metrics_sha256", String, nullable=False),
    Column("gate_status", String, nullable=False),
    Column("gate_reasons_json", json_type, nullable=False),
    Column("evaluator_version", String, nullable=False),
    Column("bundle_manifest_sha256", String, nullable=False),
    Column("evidence_json", json_type, nullable=False),
    Column("evidence_sha256", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "quant_bundle_candidate_id",
        "evidence_role",
        "ablation",
        "profile_id",
        "seed",
        name="uq_quant_bundle_evaluations_grid",
    ),
    CheckConstraint(
        "evidence_role IN ('official_feedback', 'independent_gate')",
        name="ck_quant_bundle_evaluations_role",
    ),
    CheckConstraint(
        "ablation IN ('factor_only', 'model_only', 'joint')",
        name="ck_quant_bundle_evaluations_ablation",
    ),
    CheckConstraint(
        "gate_status IN ('informational', 'passed', 'failed')",
        name="ck_quant_bundle_evaluations_gate",
    ),
    CheckConstraint(
        "(evidence_role = 'official_feedback' AND gate_status = 'informational') OR "
        "(evidence_role = 'independent_gate' AND gate_status IN ('passed', 'failed'))",
        name="ck_quant_bundle_evaluations_role_gate",
    ),
    CheckConstraint(
        "oos_vintage_id IS NULL",
        name="ck_quant_bundle_evaluations_pre_final",
    ),
)
Index(
    "idx_quant_bundle_evaluations_candidate_created",
    quant_bundle_evaluations.c.quant_bundle_candidate_id,
    quant_bundle_evaluations.c.created_at.desc(),
)

# Polymorphic links retain database-enforced candidate identity without a weak
# free-form candidate_id. Exactly one candidate foreign key must be populated.
candidate_asset_links = Table(
    "candidate_asset_links",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "asset_id",
        String,
        ForeignKey("quantlab.research_assets.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "factor_candidate_id",
        String,
        ForeignKey("quantlab.factor_candidates.id", ondelete="CASCADE"),
    ),
    Column(
        "model_candidate_id",
        String,
        ForeignKey("quantlab.model_candidates.id", ondelete="CASCADE"),
    ),
    Column(
        "quant_bundle_candidate_id",
        String,
        ForeignKey("quantlab.quant_bundle_candidates.id", ondelete="CASCADE"),
    ),
    Column("relationship", String, nullable=False),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "(factor_candidate_id IS NOT NULL AND model_candidate_id IS NULL AND "
        "quant_bundle_candidate_id IS NULL) OR "
        "(factor_candidate_id IS NULL AND model_candidate_id IS NOT NULL AND "
        "quant_bundle_candidate_id IS NULL) OR "
        "(factor_candidate_id IS NULL AND model_candidate_id IS NULL AND "
        "quant_bundle_candidate_id IS NOT NULL)",
        name="ck_candidate_asset_links_one_candidate",
    ),
)
Index(
    "uq_candidate_asset_links_factor",
    candidate_asset_links.c.factor_candidate_id,
    candidate_asset_links.c.asset_id,
    candidate_asset_links.c.relationship,
    unique=True,
    postgresql_where=candidate_asset_links.c.factor_candidate_id.is_not(None),
)
Index(
    "uq_candidate_asset_links_model",
    candidate_asset_links.c.model_candidate_id,
    candidate_asset_links.c.asset_id,
    candidate_asset_links.c.relationship,
    unique=True,
    postgresql_where=candidate_asset_links.c.model_candidate_id.is_not(None),
)
Index(
    "uq_candidate_asset_links_quant",
    candidate_asset_links.c.quant_bundle_candidate_id,
    candidate_asset_links.c.asset_id,
    candidate_asset_links.c.relationship,
    unique=True,
    postgresql_where=candidate_asset_links.c.quant_bundle_candidate_id.is_not(None),
)

strategies = Table(
    "strategies",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False, unique=True),
    Column("description", Text, nullable=False),
    Column("status", String, nullable=False),
    # Immutable economic-hypothesis family. Renaming a strategy or creating a
    # model/frequency wrapper must not reset the family-wide trial count or
    # obtain another independent capital budget (design 6.8/6.10).
    Column(
        "economic_hypothesis_group",
        String,
        nullable=False,
        server_default="legacy-unclassified",
    ),
    Column(
        "hypothesis_group_cap",
        Float,
        nullable=False,
        server_default="0.70",
    ),
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
    # Evidence authority is immutable for the lifetime of a StrategyVersion.
    # Rows created before migration 0088 remain explicitly ambiguous; future
    # final tests and the narrow consumed-history rehabilitation path must opt
    # into their authority instead of inheriting it from a default.
    Column(
        "evidence_mode",
        String,
        nullable=False,
        server_default="legacy_ambiguous",
    ),
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
    # Explicit product/research horizon. Older versions are migrated to the
    # canonical legacy_ambiguous contract; signal_period is never treated as
    # evidence of a prediction or holding horizon.
    Column("horizon_profile", String, nullable=False),
    Column("label_horizons_json", json_type, nullable=False),
    Column("decision_interval_sessions", Integer),
    Column("review_interval_sessions", Integer),
    Column("holding_min_sessions", Integer),
    Column("holding_target_sessions", Integer),
    Column("holding_max_sessions", Integer),
    Column("execution_lag_sessions", Integer),
    Column("purge_sessions", Integer),
    Column("embargo_sessions", Integer),
    Column("sealed_oos_required", Boolean, nullable=False),
    Column("sealed_oos_sessions", Integer),
    Column("horizon_contract_json", json_type, nullable=False),
    Column("horizon_contract_sha256", String, nullable=False),
    Column(
        "source_research_artifact_id",
        String,
        ForeignKey("quantlab.research_run_artifacts.id", ondelete="RESTRICT"),
    ),
    Column("strategy_rules_sha256", String),
    Column("qlib_version", String),
    Column("qlib_commit", String),
    Column("rdagent_version", String),
    Column("rdagent_commit", String),
    Column("benchmark", String, nullable=False),
    Column("universe", String, nullable=False),
    Column("config_json", json_type, nullable=False),
    # Design 6.11 promotion stage: NULL = candidate (pre-gate) or legacy
    # ungated; "paper" is set automatically when the formal hard gate approves
    # the version; "recommendation_enabled" requires the forward evidence gate
    # and the atomic auto-promotion transaction. Every other marker blocks
    # standalone recommendations.
    Column("promotion_stage", String),
    Column("created_by", String, nullable=False),
    Column("approved_by", String),
    Column("approval_reason", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("approved_at", DateTime(timezone=True)),
    CheckConstraint(
        "evidence_mode IN "
        "('legacy_ambiguous', 'sealed_final_oos', 'consumed_historical_replay')",
        name="ck_strategy_versions_evidence_mode",
    ),
    CheckConstraint(
        "horizon_profile IN "
        "('short_1_5d', 'swing_1_6m', 'long_1_3y', 'legacy_ambiguous')",
        name="ck_strategy_versions_horizon_profile",
    ),
    CheckConstraint(
        "(jsonb_typeof(label_horizons_json) = 'array' "
        "AND jsonb_typeof(horizon_contract_json) = 'object' "
        "AND horizon_contract_json ?& ARRAY["
        "'horizon_profile','label_horizons_sessions','decision_interval_sessions',"
        "'review_interval_sessions','holding_min_sessions','holding_target_sessions',"
        "'holding_max_sessions','execution_lag_sessions','purge_sessions',"
        "'embargo_sessions','sealed_oos_required','sealed_oos_sessions',"
        "'contract_version'] "
        "AND (horizon_contract_json - ARRAY["
        "'horizon_profile','label_horizons_sessions','decision_interval_sessions',"
        "'review_interval_sessions','holding_min_sessions','holding_target_sessions',"
        "'holding_max_sessions','execution_lag_sessions','purge_sessions',"
        "'embargo_sessions','sealed_oos_required','sealed_oos_sessions',"
        "'contract_version']) = '{}'::jsonb "
        "AND horizon_contract_json ->> 'horizon_profile' = horizon_profile "
        "AND horizon_contract_json ->> 'contract_version' = 'research-horizon-v1' "
        "AND horizon_contract_json -> 'label_horizons_sessions' = label_horizons_json "
        "AND (horizon_contract_json ->> 'decision_interval_sessions')::integer "
        "IS NOT DISTINCT FROM decision_interval_sessions "
        "AND (horizon_contract_json ->> 'review_interval_sessions')::integer "
        "IS NOT DISTINCT FROM review_interval_sessions "
        "AND (horizon_contract_json ->> 'holding_min_sessions')::integer "
        "IS NOT DISTINCT FROM holding_min_sessions "
        "AND (horizon_contract_json ->> 'holding_target_sessions')::integer "
        "IS NOT DISTINCT FROM holding_target_sessions "
        "AND (horizon_contract_json ->> 'holding_max_sessions')::integer "
        "IS NOT DISTINCT FROM holding_max_sessions "
        "AND (horizon_contract_json ->> 'execution_lag_sessions')::integer "
        "IS NOT DISTINCT FROM execution_lag_sessions "
        "AND (horizon_contract_json ->> 'purge_sessions')::integer "
        "IS NOT DISTINCT FROM purge_sessions "
        "AND (horizon_contract_json ->> 'embargo_sessions')::integer "
        "IS NOT DISTINCT FROM embargo_sessions "
        "AND (horizon_contract_json ->> 'sealed_oos_required')::boolean "
        "IS NOT DISTINCT FROM sealed_oos_required "
        "AND (horizon_contract_json ->> 'sealed_oos_sessions')::integer "
        "IS NOT DISTINCT FROM sealed_oos_sessions "
        "AND CASE horizon_profile "
        "WHEN 'short_1_5d' THEN horizon_contract_sha256 = "
        "'a895312d55f19ffaf35e90c4bd7003af337c94e18b61e27b4ee62a584860e1e9' "
        "WHEN 'swing_1_6m' THEN horizon_contract_sha256 = "
        "'cfc3d9ea05de009af3b4e283f83e9e840c45a0d3913f3cb938c34882b76b16f0' "
        "WHEN 'long_1_3y' THEN horizon_contract_sha256 = "
        "'0e9eae6438b44f8a4234a53f55999d4880dd0917772fb74351bd02f770f11879' "
        "WHEN 'legacy_ambiguous' THEN horizon_contract_sha256 = "
        "'6fb5cd40086f2e46850ba2d6b15e5ca7ec95191f4e42436639c7155cdecf3324' "
        "ELSE false END) IS TRUE",
        name="ck_strategy_versions_horizon_identity",
    ),
    CheckConstraint(
        "((horizon_profile = 'legacy_ambiguous' "
        "AND label_horizons_json = '[]'::jsonb "
        "AND decision_interval_sessions IS NULL "
        "AND review_interval_sessions IS NULL "
        "AND holding_min_sessions IS NULL "
        "AND holding_target_sessions IS NULL "
        "AND holding_max_sessions IS NULL "
        "AND execution_lag_sessions IS NULL "
        "AND purge_sessions IS NULL AND embargo_sessions IS NULL "
        "AND sealed_oos_required = false AND sealed_oos_sessions IS NULL) OR "
        "(horizon_profile = 'short_1_5d' "
        "AND label_horizons_json = '[1,2,3,5]'::jsonb "
        "AND decision_interval_sessions = 1 AND review_interval_sessions = 1 "
        "AND holding_min_sessions = 1 AND holding_target_sessions = 3 "
        "AND holding_max_sessions = 5 AND execution_lag_sessions = 1 "
        "AND purge_sessions = 6 AND embargo_sessions = 6 "
        "AND sealed_oos_required = true AND sealed_oos_sessions = 252) OR "
        "(horizon_profile = 'swing_1_6m' "
        "AND label_horizons_json = '[21,63,126]'::jsonb "
        "AND decision_interval_sessions = 5 AND review_interval_sessions = 5 "
        "AND holding_min_sessions = 21 AND holding_target_sessions = 63 "
        "AND holding_max_sessions = 126 AND execution_lag_sessions = 1 "
        "AND purge_sessions = 127 AND embargo_sessions = 127 "
        "AND sealed_oos_required = true AND sealed_oos_sessions = 504) OR "
        "(horizon_profile = 'long_1_3y' "
        "AND label_horizons_json = '[63,126,252]'::jsonb "
        "AND decision_interval_sessions = 21 AND review_interval_sessions = 21 "
        "AND holding_min_sessions = 252 AND holding_target_sessions = 504 "
        "AND holding_max_sessions = 756 AND execution_lag_sessions = 1 "
        "AND purge_sessions = 253 AND embargo_sessions = 253 "
        "AND sealed_oos_required = true AND sealed_oos_sessions = 756)) IS TRUE",
        name="ck_strategy_versions_horizon_values",
    ),
    CheckConstraint(
        "((horizon_profile = 'legacy_ambiguous') OR "
        "strategy_rules_sha256 ~ '^[0-9a-f]{64}$') IS TRUE",
        name="ck_strategy_versions_rule_identity",
    ),
    CheckConstraint(
        "(horizon_profile <> 'legacy_ambiguous' OR "
        "promotion_stage IS DISTINCT FROM 'recommendation_enabled') IS TRUE",
        name="ck_strategy_versions_legacy_authority",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-08-30-v12' AND "
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "THEN ("
        "config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'6366bd2b77c1c60ea4069afde43d5335c1e362c18acf2955f13902e7ba9ccbc6' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ "
        "'^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v12_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-08-30-v13' AND "
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "THEN ("
        "config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'3000d84d183fe589da13d01902f88b1da1d402f47e18e4aafe91831cc59bdd8f' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ "
        "'^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v13_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-08-30-v14' AND "
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "THEN ("
        "config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'04f9dce110aaf44d32972c68db3edefafbd08cfdbf1f7dfb98aac9a15409bfe5' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ "
        "'^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v14_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-08-30-v15' AND "
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "THEN ("
        "config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'e361d85d69d77cb6e0de7072db6f8aaff5d83f1e7902fe16ef06c2e28fce1867' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ "
        "'^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v15_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-08-30-v16' AND "
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "THEN ("
        "config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'ac1c2020996efa4e3c3e9609dd4c736d8c65893dccd7ba9c90e3464402d2a329' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ "
        "'^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v16_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-08-31-v17' AND "
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "THEN ("
        "config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'c3d6aeb00a8f5286ee117f885231b43ad49e461c5183f5cbf32c4601824bb9fc' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ "
        "'^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v17_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-08-31-v18' THEN ("
        "COALESCE(config_json ->> 'recipe_id', '') = "
        "'short_relative_strength' "
        "AND evidence_mode = 'consumed_historical_replay' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'c45bd901f25ba0cf289e3ec1a71015865d190afb4510c6bc207d53c9cc2ab24d' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'3f0c60adbe3b50ff26771e11ce75f6a48d13772ac5bff549f716469748b92874' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ '^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v18_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-09-01-v19' THEN ("
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "AND evidence_mode = 'sealed_final_oos' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'c45bd901f25ba0cf289e3ec1a71015865d190afb4510c6bc207d53c9cc2ab24d' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'d6ded80dbe88c6bee0403428132e6def80e14bc7ff540ac41e4d83f8343bf1c3' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ '^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v19_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-09-01-v20' THEN ("
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "AND evidence_mode = 'sealed_final_oos' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'c45bd901f25ba0cf289e3ec1a71015865d190afb4510c6bc207d53c9cc2ab24d' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'a0fe9491b27836526af9d7644c2c396973a95933fcfa4bf9365046cf634be53d' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ '^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v20_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-09-01-v21' THEN ("
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "AND evidence_mode = 'sealed_final_oos' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'c45bd901f25ba0cf289e3ec1a71015865d190afb4510c6bc207d53c9cc2ab24d' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'5859384d80aeee8302e1474f08b405eaaccec0e2b056b9f1da89e72454e42f48' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ '^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v21_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-09-01-v22' THEN ("
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "AND evidence_mode = 'sealed_final_oos' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'c45bd901f25ba0cf289e3ec1a71015865d190afb4510c6bc207d53c9cc2ab24d' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'6709b7e2a4abcc2ae9dcb63db2d50f4bf8ba5507ee47c62ee95d70109fe7d7e0' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ '^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v22_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-09-01-v23' THEN ("
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "AND evidence_mode = 'sealed_final_oos' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'c45bd901f25ba0cf289e3ec1a71015865d190afb4510c6bc207d53c9cc2ab24d' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'18c2964083d39c7abc761637f173b94eb72c6c5fff4be762177ca39048792558' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ '^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v23_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-09-03-v24' THEN ("
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "AND evidence_mode = 'sealed_final_oos' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'79ff5cee1f046dea95bba13358005d975f0a83d7324c8f0fe18b07e5962b9615' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'82b151261f4fff98a7848f093fb0ea69e47fadf10b41df83288279e7a076287e' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ '^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v24_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-09-03-v25' THEN ("
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "AND evidence_mode = 'sealed_final_oos' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'79ff5cee1f046dea95bba13358005d975f0a83d7324c8f0fe18b07e5962b9615' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'e9f26125b1675695b708ee48b8bacf65bea8c83abb56ac7741d799a407f48db1' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ '^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v25_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-09-03-v26' THEN ("
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "AND evidence_mode = 'sealed_final_oos' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'79ff5cee1f046dea95bba13358005d975f0a83d7324c8f0fe18b07e5962b9615' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'6a584023edd2b8589ac0bb2dcadf5dc7e06f6999b606aded11459ebc401aa9b4' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ '^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v26_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-09-04-v27' THEN ("
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "AND evidence_mode = 'sealed_final_oos' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'79ff5cee1f046dea95bba13358005d975f0a83d7324c8f0fe18b07e5962b9615' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'c987c4fb6f1f5ac6f4d8cf21f3f901ef1f8c4b8ee33bb7153b8afa3c46162697' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ '^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v27_runtime_identity",
    ),
    CheckConstraint(
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        "'qlib-rdagent-single-mainline-2026-09-04-v28' THEN ("
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "AND evidence_mode = 'sealed_final_oos' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runner_sha256' = "
        "'79ff5cee1f046dea95bba13358005d975f0a83d7324c8f0fe18b07e5962b9615' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        "'d8bb1e2ba21413e4cfb04ab819a270e4699f7d5e9964939ae727aa8055876b85' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ '^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        name="ck_strategy_versions_v28_runtime_identity",
    ),
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
    # Historical approval only starts an isolated paper-validation version.
    # The incumbent stays live until the forward gate atomically promotes the
    # challenger, so uniqueness applies only to recommendation authority.
    postgresql_where=text(
        "status = 'approved' AND promotion_stage = 'recommendation_enabled'"
    ),
)
Index(
    "idx_strategy_versions_horizon_status",
    strategy_versions.c.horizon_profile,
    strategy_versions.c.status,
)

# An exceptional transparent-baseline repair may reuse the same sealed
# calendar window only when it was preregistered before any performance result
# existed.  The source OOS rows stay consumed and immutable; this registry
# binds the one allowed replacement lockbox to the original audit receipt.
transparent_baseline_pre_result_repairs = Table(
    "transparent_baseline_pre_result_repairs",
    metadata,
    Column("receipt_sha256", String, primary_key=True),
    Column(
        "source_audit_event_id",
        BigInteger,
        ForeignKey("quantlab.audit_events.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    ),
    Column("source_batch_sha256", String, nullable=False, unique=True),
    Column("target_batch_sha256", String, nullable=False, unique=True),
    Column("source_dataset_lineage_id", String, nullable=False),
    Column("target_dataset_lineage_id", String, nullable=False),
    Column("target_recipe_version", String, nullable=False),
    Column("source_backtest_ids_json", json_type, nullable=False),
    Column("target_strategy_version_ids_json", json_type, nullable=False),
    Column("verification_json", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_transparent_baseline_pre_result_repairs_created",
    transparent_baseline_pre_result_repairs.c.created_at.desc(),
)
Index(
    "uq_strategy_versions_active_horizon",
    strategy_versions.c.horizon_profile,
    unique=True,
    postgresql_where=text(
        "status = 'approved' AND promotion_stage = 'recommendation_enabled' "
        "AND horizon_profile <> 'legacy_ambiguous'"
    ),
)
Index(
    "uq_strategy_versions_source_research_artifact",
    strategy_versions.c.source_research_artifact_id,
    unique=True,
    postgresql_where=strategy_versions.c.source_research_artifact_id.is_not(None),
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

model_artifacts = Table(
    "model_artifacts",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("artifact_key", String, nullable=False),
    Column("status", String, nullable=False),
    Column("strategy_spec_sha256", String, nullable=False),
    Column("model_recipe_sha256", String, nullable=False),
    Column("model_recipe_json", json_type, nullable=False),
    Column("dataset", String, nullable=False),
    Column("dataset_identity_sha256", String, nullable=False),
    # Nullable only for artifacts created before the governed model-runtime
    # contract. New model candidates and every activation fail closed when the
    # independent/formal execution environment identity is absent.
    Column("execution_environment_sha256", String),
    Column("training_start", Date, nullable=False),
    Column("training_end", Date, nullable=False),
    Column("data_cutoff_at", DateTime(timezone=True), nullable=False),
    Column("scheduled_refit_at", DateTime(timezone=True)),
    Column("valid_until", DateTime(timezone=True), nullable=False),
    Column("artifact_path", Text, nullable=False),
    Column("artifact_sha256", String, nullable=False),
    Column("predictions_sha256", String, nullable=False),
    # Prediction tables and fitted checkpoints are separate immutable
    # artifacts.  Rows created before migration 0069 remain nullable and are
    # deliberately ineligible for live inference until rebuilt.
    Column("checkpoint_path", Text),
    Column("checkpoint_sha256", String),
    Column("checkpoint_format", String),
    Column("model_data_contract_sha256", String),
    Column("training_kind", String),
    Column("training_evidence_json", json_type),
    Column("training_evidence_sha256", String),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("activated_by", String),
    Column("activated_at", DateTime(timezone=True)),
    Column("retired_at", DateTime(timezone=True)),
    Column("failure_reason", Text),
    UniqueConstraint(
        "strategy_version_id",
        "artifact_key",
        name="uq_model_artifacts_key",
    ),
    CheckConstraint(
        "status IN ('candidate', 'active', 'retired', 'failed', 'expired')",
        name="ck_model_artifacts_status",
    ),
    CheckConstraint(
        "checkpoint_format IS NULL OR checkpoint_format IN "
        "('lightgbm_text', 'pytorch_state_dict', 'ridge_numeric_json')",
        name="ck_model_artifacts_checkpoint_format",
    ),
    CheckConstraint(
        "training_kind IS NULL OR training_kind IN "
        "('formal_oos', 'monthly_retrain', 'early_retrain', 'daily_inference')",
        name="ck_model_artifacts_training_kind",
    ),
)
Index(
    "idx_model_artifacts_strategy_created",
    model_artifacts.c.strategy_version_id,
    model_artifacts.c.created_at.desc(),
)
Index(
    "uq_model_artifacts_active",
    model_artifacts.c.strategy_version_id,
    unique=True,
    postgresql_where=model_artifacts.c.status == "active",
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
    Column(
        "evidence_mode",
        String,
        nullable=False,
        server_default="legacy_ambiguous",
    ),
    Column("periods_json", json_type, nullable=False),
    Column("metrics_json", json_type),
    Column("artifact_path", Text),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    CheckConstraint(
        "evidence_mode IN "
        "('legacy_ambiguous', 'sealed_final_oos', 'consumed_historical_replay')",
        name="ck_backtest_runs_evidence_mode",
    ),
)
Index(
    "idx_backtest_runs_status_created",
    backtest_runs.c.status,
    backtest_runs.c.created_at.desc(),
)

# A consumed historical replay can never acquire recommendation authority by
# itself.  This append-only receipt records the one narrow admission that may
# move an exact transparent public baseline into the existing paper stage.
# The existing forward gate and simulation ledger remain the only lifecycle
# and evidence state machines.
strategy_forward_only_rehabilitations = Table(
    "strategy_forward_only_rehabilitations",
    metadata,
    Column("receipt_sha256", String, primary_key=True),
    Column(
        "source_audit_event_id",
        BigInteger,
        ForeignKey("quantlab.audit_events.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    ),
    Column(
        "source_strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "source_backtest_id",
        String,
        ForeignKey("quantlab.backtest_runs.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "source_job_id",
        String,
        ForeignKey("quantlab.jobs.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "source_interruption_recovery_receipt_sha256",
        String,
        ForeignKey(
            "quantlab.formal_backtest_interruption_recoveries.receipt_sha256",
            ondelete="RESTRICT",
        ),
        nullable=False,
    ),
    Column("source_interruption_receipt_authority", String, nullable=False),
    Column("source_lockbox_contract_version", String, nullable=False),
    Column("source_lockbox_batch_sha256", String, nullable=False),
    Column("source_lockbox_member_sha256", String, nullable=False),
    Column("source_history_selection_sha256", String, nullable=False),
    Column("source_unavailable_horizons_sha256", String, nullable=False),
    Column("source_unavailable_evidence_sha256s_json", json_type, nullable=False),
    Column("source_cash_only_scope", String, nullable=False),
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    ),
    Column(
        "backtest_id",
        String,
        ForeignKey("quantlab.backtest_runs.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    ),
    Column(
        "consumed_oos_vintage_id",
        String,
        ForeignKey("quantlab.oos_vintages.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    ),
    Column("contract_version", String, nullable=False),
    Column("evidence_mode", String, nullable=False),
    Column("authority", String, nullable=False),
    Column("recipe_id", String, nullable=False),
    Column("horizon_profile", String, nullable=False),
    Column("dataset", String, nullable=False),
    Column("dataset_identity_sha256", String, nullable=False),
    Column("dataset_lineage_id", String, nullable=False),
    Column("strategy_rules_sha256", String, nullable=False),
    Column("execution_contract_hash", String, nullable=False),
    Column("runner_sha256", String, nullable=False),
    Column("runtime_bundle_sha256", String, nullable=False),
    Column("worker_runtime_image_digest", String, nullable=False),
    Column("replay_periods_json", json_type, nullable=False),
    Column("replay_manifest_sha256", String, nullable=False),
    Column("replay_result_sha256", String, nullable=False),
    Column("replay_artifact_manifest_sha256", String, nullable=False),
    Column("replay_daily_returns_sha256", String, nullable=False),
    Column("strategy_trial_count", Integer, nullable=False),
    Column("trial_count_audit_sha256", String, nullable=False),
    Column("incomplete_family_eligibility_sha256", String, nullable=False),
    Column("forward_criteria_json", json_type, nullable=False),
    Column("forward_criteria_sha256", String, nullable=False),
    Column("qualification_json", json_type, nullable=False),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "contract_version = 'forward-only-rehabilitation-v1' "
        "AND evidence_mode = 'consumed_historical_replay' "
        "AND authority = 'historical_description_only' "
        "AND source_interruption_receipt_authority = "
        "'interruption_identity_only_not_pre_result' "
        "AND source_lockbox_contract_version = "
        "'transparent-baseline-available-horizons-lockbox-v3' "
        "AND source_cash_only_scope = 'cash_only_projection_only' "
        "AND strategy_version_id <> source_strategy_version_id "
        "AND backtest_id <> source_backtest_id "
        "AND strategy_trial_count > 1",
        name="ck_forward_only_rehabilitation_authority",
    ),
    CheckConstraint(
        "receipt_sha256 ~ '^[0-9a-f]{64}$' "
        "AND source_interruption_recovery_receipt_sha256 = "
        "'6345c455862d8bb587e12f8ce0be7c1da291f2dde82fbdbb31e954bb88b9d3df' "
        "AND source_lockbox_batch_sha256 ~ '^[0-9a-f]{64}$' "
        "AND source_lockbox_member_sha256 ~ '^[0-9a-f]{64}$' "
        "AND source_history_selection_sha256 ~ '^[0-9a-f]{64}$' "
        "AND source_unavailable_horizons_sha256 ~ '^[0-9a-f]{64}$' "
        "AND dataset_identity_sha256 ~ '^[0-9a-f]{64}$' "
        "AND dataset_lineage_id ~ '^[0-9a-f]{64}$' "
        "AND strategy_rules_sha256 ~ '^[0-9a-f]{64}$' "
        "AND execution_contract_hash ~ '^[0-9a-f]{64}$' "
        "AND runner_sha256 ~ '^[0-9a-f]{64}$' "
        "AND runtime_bundle_sha256 ~ '^[0-9a-f]{64}$' "
        "AND worker_runtime_image_digest ~ '^sha256:[0-9a-f]{64}$' "
        "AND replay_manifest_sha256 ~ '^[0-9a-f]{64}$' "
        "AND replay_result_sha256 ~ '^[0-9a-f]{64}$' "
        "AND replay_artifact_manifest_sha256 ~ '^[0-9a-f]{64}$' "
        "AND replay_daily_returns_sha256 ~ '^[0-9a-f]{64}$' "
        "AND trial_count_audit_sha256 ~ '^[0-9a-f]{64}$' "
        "AND incomplete_family_eligibility_sha256 ~ '^[0-9a-f]{64}$' "
        "AND forward_criteria_sha256 ~ '^[0-9a-f]{64}$'",
        name="ck_forward_only_rehabilitation_hashes",
    ),
    CheckConstraint(
        "(jsonb_typeof(replay_periods_json) = 'object' "
        "AND replay_periods_json ?& ARRAY['start','end','historical_start','historical_end'] "
        "AND jsonb_typeof(forward_criteria_json) = 'object' "
        "AND (forward_criteria_json -> 'thresholds' ->> "
        "'min_forward_calendar_days')::integer >= 365 "
        "AND (forward_criteria_json -> 'thresholds' ->> "
        "'min_forward_trading_days')::integer >= 252 "
        "AND jsonb_typeof(qualification_json) = 'object' "
        "AND jsonb_typeof(source_unavailable_evidence_sha256s_json) = 'object' "
        "AND source_unavailable_evidence_sha256s_json ?& "
        "ARRAY['swing_1_6m','long_1_3y'] "
        "AND (source_unavailable_evidence_sha256s_json - "
        "ARRAY['swing_1_6m','long_1_3y']) = '{}'::jsonb "
        "AND qualification_json -> 'historical_replay_opened' = 'true'::jsonb "
        "AND qualification_json -> 'final_oos_opened' = 'true'::jsonb "
        "AND qualification_json -> 'capital_eligible' = 'false'::jsonb "
        "AND qualification_json -> 'consumed_oos_replayed' = 'true'::jsonb "
        "AND qualification_json -> 'sealed_final_oos' = 'false'::jsonb "
        "AND qualification_json -> 'unseen_oos' = 'false'::jsonb "
        "AND qualification_json ->> 'authority' = 'historical_description_only' "
        "AND qualification_json ->> 'receipt_sha256' = receipt_sha256) IS TRUE",
        name="ck_forward_only_rehabilitation_evidence",
    ),
)
Index(
    "idx_forward_only_rehabilitations_created",
    strategy_forward_only_rehabilitations.c.created_at.desc(),
)

# This pre-run receipt is intentionally narrower than the rehabilitation
# admission above. It only authorizes the conservative Bonferroni calculation
# for one frozen, historically incomplete factor family. It grants no strategy
# or capital status and cannot be reused by another StrategyVersion.
strategy_incomplete_family_eligibilities = Table(
    "strategy_incomplete_family_eligibilities",
    metadata,
    Column("receipt_sha256", String, primary_key=True),
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    ),
    Column("contract_version", String, nullable=False),
    Column("evidence_mode", String, nullable=False),
    Column("authority", String, nullable=False),
    Column(
        "source_strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "source_backtest_id",
        String,
        ForeignKey("quantlab.backtest_runs.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "source_job_id",
        String,
        ForeignKey("quantlab.jobs.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("economic_hypothesis_group", String, nullable=False),
    Column("eligible_strategy_version_ids_json", json_type, nullable=False),
    Column("strategy_trial_count", Integer, nullable=False),
    Column("trial_count_audit_json", json_type, nullable=False),
    Column("trial_count_audit_sha256", String, nullable=False),
    Column("missing_artifacts_json", json_type, nullable=False),
    Column("cutoff_at", DateTime(timezone=True), nullable=False),
    Column("qualification_json", json_type, nullable=False),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "(contract_version = 'incomplete-factor-family-eligibility-v1' "
        "AND evidence_mode = 'consumed_historical_replay' "
        "AND authority = 'conservative_bonferroni_only' "
        "AND strategy_trial_count > 1 "
        "AND receipt_sha256 ~ '^[0-9a-f]{64}$' "
        "AND trial_count_audit_sha256 ~ '^[0-9a-f]{64}$' "
        "AND jsonb_typeof(eligible_strategy_version_ids_json) = 'array' "
        "AND jsonb_array_length(eligible_strategy_version_ids_json) > 1 "
        "AND jsonb_typeof(trial_count_audit_json) = 'object' "
        "AND jsonb_typeof(missing_artifacts_json) = 'array' "
        "AND jsonb_array_length(missing_artifacts_json) >= 3 "
        "AND qualification_json ->> 'receipt_sha256' = receipt_sha256) IS TRUE",
        name="ck_incomplete_family_eligibility_authority",
    ),
)
Index(
    "idx_incomplete_family_eligibilities_created",
    strategy_incomplete_family_eligibilities.c.created_at.desc(),
)

# A formal final-OOS backtest normally has exactly one process execution.  The
# sole exception is an operator-authorized continuation after an externally
# caused service interruption, before any performance result existed.  The
# mutable job/backtest rows are requeued in place; this append-only registry
# retains the complete pre-requeue evidence and the exact authorization.
_FORMAL_BACKTEST_INTERRUPTION_JOURNAL_EXCERPT_JSON = (
    '{"contract_version":"quantlab-systemd-docker-journal-excerpt-v1","records":['
    '{"message":"Starting quantlab-backup.service - QuantLab bounded control-plane '
    'backup...","observed_at":"2026-08-30T19:29:57.618523+00:00","source":"systemd",'
    '"unit":"quantlab-backup.service"},'
    '{"message":"Container quantlab-platform-evaluation-worker-1 Stopping",'
    '"observed_at":"2026-08-30T19:30:09.189629+00:00","source":"systemd",'
    '"unit":"quantlab-backup.service"},'
    '{"container_id":"710730b1ce29bdd43e081f967860e7f49a791e9e62e62aeeaa5e2b98e3f03218",'
    '"daemon_shutting_down":false,"exit_status":0,"has_been_manually_stopped":true,'
    '"observed_at":"2026-08-30T19:30:14.800407712+00:00","source":"dockerd"},'
    '{"message":"Container quantlab-platform-evaluation-worker-1 Stopped",'
    '"observed_at":"2026-08-30T19:30:14.869740+00:00","source":"systemd",'
    '"unit":"quantlab-backup.service"},'
    '{"message":"Container quantlab-platform-evaluation-worker-1 Started",'
    '"observed_at":"2026-08-30T19:34:33.195752+00:00","source":"systemd",'
    '"unit":"quantlab-backup.service"},'
    '{"message":"Finished quantlab-backup.service - QuantLab bounded control-plane backup.",'
    '"observed_at":"2026-08-30T19:35:56.019550+00:00","source":"systemd",'
    '"unit":"quantlab-backup.service"}]}'
)
_FORMAL_BACKTEST_INTERRUPTION_JOURNAL_EXCERPT_SQL = (
    _FORMAL_BACKTEST_INTERRUPTION_JOURNAL_EXCERPT_JSON.replace(":", "\\:")
)
_FORMAL_BACKTEST_INTERRUPTION_EXPECTED_RECEIPT_JSON = r"""{"contract_version":"transparent-baseline-service-interruption-recovery-v1","execution_controller":{"application_name":"quantlab-v17-recovery-b41f78f9","authorization_application_name":"quantlab-v17-recovery-authorizer","canonical_output_path":"/data/artifacts/backtests/0a113fe28ca741b6be9c09ab046c9d02","contract_version":"quantlab-v17-sealed-one-shot-controller-v1","controller_sha256":"1329aaf46bc9db8a3172abf6ecbce2837a12f1b555aad7c33c4ff0a9e70a2c0d","data_mount_mode":"volumes-from-read-only-with-persistent-target-samefile-bind","sealed_worker_image_digest":"sha256:b41f78f9c99dd9853998d85a52593bcab247907ac60f9d721d863e20904e8bb7","target_artifact_path":"/data/artifacts/formal-backtest-recoveries/0a113fe28ca741b6be9c09ab046c9d02/attempt-2","target_execution_log_mount_mode":"single-file-read-write-bind","target_execution_log_path":"/data/artifacts/formal-backtest-recoveries/0a113fe28ca741b6be9c09ab046c9d02/attempt-2.log"},"external_interruption":{"backtest_id":"0a113fe28ca741b6be9c09ab046c9d02","contract_version":"quantlab-external-service-interruption-evidence-v1","has_been_manually_stopped":true,"job_id":"858a75a6f1994c359fa9c3567ed09f57","journal_excerpt":{"contract_version":"quantlab-systemd-docker-journal-excerpt-v1","records":[{"message":"Starting quantlab-backup.service - QuantLab bounded control-plane backup...","observed_at":"2026-08-30T19:29:57.618523+00:00","source":"systemd","unit":"quantlab-backup.service"},{"message":"Container quantlab-platform-evaluation-worker-1 Stopping","observed_at":"2026-08-30T19:30:09.189629+00:00","source":"systemd","unit":"quantlab-backup.service"},{"container_id":"710730b1ce29bdd43e081f967860e7f49a791e9e62e62aeeaa5e2b98e3f03218","daemon_shutting_down":false,"exit_status":0,"has_been_manually_stopped":true,"observed_at":"2026-08-30T19:30:14.800407712+00:00","source":"dockerd"},{"message":"Container quantlab-platform-evaluation-worker-1 Stopped","observed_at":"2026-08-30T19:30:14.869740+00:00","source":"systemd","unit":"quantlab-backup.service"},{"message":"Container quantlab-platform-evaluation-worker-1 Started","observed_at":"2026-08-30T19:34:33.195752+00:00","source":"systemd","unit":"quantlab-backup.service"},{"message":"Finished quantlab-backup.service - QuantLab bounded control-plane backup.","observed_at":"2026-08-30T19:35:56.019550+00:00","source":"systemd","unit":"quantlab-backup.service"}]},"journal_sha256":"c7a93f1fe86a57ed6fcd51f32fe74be55ef7115cb4a79454e12ccbee338efcc7","observed_at":"2026-08-30T19:30:09.189629+00:00","oom_killed":false,"service":"evaluation-worker","signal":"SIGTERM","source":"systemd-docker-journal","stop_owner":"quantlab-backup.service"},"immutable_binding":{"dataset":"cn-20080101-20260828-v7-failclosed-ed5c8b3","dataset_identity_sha256":"eab69dff43abcc77e60f47ad51d2182c7bfd5d90f5d36c479e3de61cebf768f2","dataset_lineage_id":"1b97efcc3956b4be2dfebff7567efd4707c73d806f9d0cb2b63540d9dedd852e","execution_contract_hash":"0ffa8939a70f0c46499e3ae472b877f68cce1b9fb95dfe81886ab8430ddcce61","periods":{"end":"2019-11-20","historical_end":"2018-10-10","historical_start":"2008-01-02","start":"2018-11-08"},"recipe_id":"short_relative_strength","recipe_sha256":"dee3551a73f2ebb3fbbbddfafdf99f4618e3dfdd98981fdb4b4d8849723d5fd8","recipe_version":"qlib-rdagent-single-mainline-2026-08-31-v17","runner_sha256":"31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3","runtime_bundle_sha256":"c3d6aeb00a8f5286ee117f885231b43ad49e461c5183f5cbf32c4601824bb9fc","source_repair_receipt_sha256":"980ea643755d261cc7ee39ffec5e23af3f8e27ebdc1e2f2a647e0802b9dc636d","strategy_rules_sha256":"644d9ee73ea4c167c7d8f58b2e8b1707289cb569c48ae73ce131cce139f7e756","strategy_version_id":"4414d202dbb641608975e5305bc18da4","worker_runtime_image_digest":"sha256:b41f78f9c99dd9853998d85a52593bcab247907ac60f9d721d863e20904e8bb7"},"performance_information_used":false,"pre_result_evidence":{"artifact_inventory":[{"bytes":57532420,"path":"baseline/composite.parquet","sha256":"3e682431fe9844927d30ba039409b12d733a9adc39c6a1192c4a74d6a222028d"},{"bytes":57206461,"path":"baseline/normalized/amount_expansion_5d.parquet","sha256":"0c7a7fec840be9d26153c2725f40345022cb531a2b918ff3568133a4799c68f3"},{"bytes":53130406,"path":"baseline/normalized/close_location_5d.parquet","sha256":"12ecd49faa1e6a5fb3926f827472c9ea161634f35ac94f6ab1504bb3c42129af"},{"bytes":57217048,"path":"baseline/normalized/extension_penalty_5d.parquet","sha256":"f70f2a59471187e5e54506210828c893e0c5d6076d2cad69c583c376573534c1"},{"bytes":55912111,"path":"baseline/normalized/relative_strength_5d.parquet","sha256":"50c49cbc1223554ba3b0b1a3cd6667cded6407edb89b257b61cc81a26ad3ac12"},{"bytes":33356095,"path":"baseline/raw/amount_expansion_5d.parquet","sha256":"d166122759b5ea42ed582fe217867de2937f62d03142d6d8630b2bd10d24fe2d"},{"bytes":30035531,"path":"baseline/raw/close_location_5d.parquet","sha256":"3be7552ff64404260908b1cc196524fe06dd2f7c6dbc3b2478edd956521f48f2"},{"bytes":32735138,"path":"baseline/raw/extension_penalty_5d.parquet","sha256":"a7697b9a94e333881c2921071e12e28a8adf09a33a2411afe97fb352593e0d8c"},{"bytes":29600893,"path":"baseline/raw/relative_strength_5d.parquet","sha256":"55fddb6562cd756f82f18c268e82a2719fccfd46986996de83826e8f9ef11836"},{"bytes":69327,"path":"manifest.json","sha256":"a4b72701a88247a7bf783b3bfe2950131c936d6d3dcea7d42601ab5bb8d115c7"}],"artifact_inventory_sha256":"c0272f59ca8878d26e95db2b4328cfc9551c4b3dff4382a237c38cd87f00f63e","artifact_path":"/data/artifacts/backtests/0a113fe28ca741b6be9c09ab046c9d02","backtest_metrics_absent":true,"job_progress_absent":true,"log_prefix":{"bytes":25665,"path":"/data/platform/logs/strategy-backtest-0a113fe28ca741b6be9c09ab046c9d02.log","sha256":"c287af99368bf95d16330513f155f795b5fe77bc1e595c47d2fed7a1a447addb"},"manifest_sha256":"a4b72701a88247a7bf783b3bfe2950131c936d6d3dcea7d42601ab5bb8d115c7","result_absent":true,"terminal_artifacts_absent":["artifact_manifest.json","daily_returns.parquet","result.json"]},"reason_code":"external_service_sigterm_before_formal_result","receipt_sha256":"6345c455862d8bb587e12f8ce0be7c1da291f2dde82fbdbb31e954bb88b9d3df","recovery_generation":"v17-control-plane-backup-sigterm-20260831","source_backtest":{"artifact_path":"/data/artifacts/backtests/0a113fe28ca741b6be9c09ab046c9d02","created_at":"2026-08-30T18:56:25.223043+00:00","dataset":"cn-20080101-20260828-v7-failclosed-ed5c8b3","error_absent":true,"execution_contract_hash":"0ffa8939a70f0c46499e3ae472b877f68cce1b9fb95dfe81886ab8430ddcce61","execution_dataset":null,"finished_at_absent":true,"id":"0a113fe28ca741b6be9c09ab046c9d02","job_id":"858a75a6f1994c359fa9c3567ed09f57","metrics_absent":true,"periods":{"end":"2019-11-20","historical_end":"2018-10-10","historical_start":"2008-01-02","start":"2018-11-08"},"qlib_commit":"d5379c520f66a39953bad76234a7019a72796fd0","qlib_version":"0.0.dev0+gd5379c520f66a39953bad76234a7019a72796fd0","rdagent_commit":"4f9ecb005881cddc08df0124a2e894c018007679","rdagent_version":"0.0.dev0+g4f9ecb005881cddc08df0124a2e894c018007679","row_sha256":"cfd64d7ff3006e1d3c5cbf8ed9f80f1efe9067ae8f876f8c1c21732573f95bc4","started_at":"2026-08-30T18:56:26.196535+00:00","status":"running","strategy_version_id":"4414d202dbb641608975e5305bc18da4"},"source_job":{"attempts":1,"created_at":"2026-08-30T18:56:25.236023+00:00","error":"Worker restarted after the bounded attempt limit; operator review is required","exit_code":143,"finished_at":"2026-08-30T19:34:35.198471+00:00","id":"858a75a6f1994c359fa9c3567ed09f57","idempotency_key":"transparent-baseline:4414d202dbb641608975e5305bc18da4:0a113fe28ca741b6be9c09ab046c9d02","kind":"strategy_backtest","log_path":"/data/platform/logs/strategy-backtest-0a113fe28ca741b6be9c09ab046c9d02.log","max_attempts":1,"payload_sha256":"f99fa6c8f2a1364d56a0d0bff9d7401b1f504f3b020c321d996f05a70a05df42","progress_absent":true,"row_sha256":"8a99cddcafb2554414c897e7bf11f6ba822490b76d77c78cb9a7d425422105e9","started_at":"2026-08-30T18:56:26.193585+00:00","status":"failed"},"target":{"artifact_path":"/data/artifacts/formal-backtest-recoveries/0a113fe28ca741b6be9c09ab046c9d02/attempt-2","authorized_attempt":2,"backtest_id":"0a113fe28ca741b6be9c09ab046c9d02","job_id":"858a75a6f1994c359fa9c3567ed09f57","max_attempts":2,"same_formal_oos_identity":true,"strategy_version_id":"4414d202dbb641608975e5305bc18da4"}}"""
_FORMAL_BACKTEST_INTERRUPTION_EXPECTED_RECEIPT_SQL = (
    _FORMAL_BACKTEST_INTERRUPTION_EXPECTED_RECEIPT_JSON.replace(":", "\\:")
)
formal_backtest_interruption_recoveries = Table(
    "formal_backtest_interruption_recoveries",
    metadata,
    Column("receipt_sha256", String, primary_key=True),
    Column(
        "source_audit_event_id",
        BigInteger,
        ForeignKey("quantlab.audit_events.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    ),
    Column(
        "backtest_id",
        String,
        ForeignKey("quantlab.backtest_runs.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    ),
    Column(
        "job_id",
        String,
        ForeignKey("quantlab.jobs.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    ),
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    ),
    Column("contract_version", String, nullable=False),
    Column("recovery_generation", String, nullable=False, unique=True),
    Column("reason_code", String, nullable=False),
    Column("source_job_row_sha256", String, nullable=False, unique=True),
    Column("source_backtest_row_sha256", String, nullable=False, unique=True),
    Column("source_payload_sha256", String, nullable=False),
    Column("source_log_prefix_sha256", String, nullable=False),
    Column("source_log_prefix_bytes", BigInteger, nullable=False),
    Column("source_artifact_inventory_sha256", String, nullable=False),
    Column("target_artifact_path", Text, nullable=False),
    Column("verification_json", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "receipt_sha256 ~ '^[0-9a-f]{64}$' "
        "AND source_job_row_sha256 ~ '^[0-9a-f]{64}$' "
        "AND source_backtest_row_sha256 ~ '^[0-9a-f]{64}$' "
        "AND source_payload_sha256 ~ '^[0-9a-f]{64}$' "
        "AND source_log_prefix_sha256 ~ '^[0-9a-f]{64}$' "
        "AND source_artifact_inventory_sha256 ~ '^[0-9a-f]{64}$' "
        "AND source_log_prefix_bytes > 0",
        name="ck_formal_backtest_interruption_recovery_sha256",
    ),
    CheckConstraint(
        "(contract_version = 'transparent-baseline-service-interruption-recovery-v1' "
        "AND recovery_generation = "
        "'v17-control-plane-backup-sigterm-20260831' "
        "AND reason_code = 'external_service_sigterm_before_formal_result' "
        "AND backtest_id = '0a113fe28ca741b6be9c09ab046c9d02' "
        "AND job_id = '858a75a6f1994c359fa9c3567ed09f57' "
        "AND strategy_version_id = '4414d202dbb641608975e5305bc18da4' "
        "AND receipt_sha256 = "
        "'6345c455862d8bb587e12f8ce0be7c1da291f2dde82fbdbb31e954bb88b9d3df' "
        "AND source_job_row_sha256 = "
        "'8a99cddcafb2554414c897e7bf11f6ba822490b76d77c78cb9a7d425422105e9' "
        "AND source_backtest_row_sha256 = "
        "'cfd64d7ff3006e1d3c5cbf8ed9f80f1efe9067ae8f876f8c1c21732573f95bc4' "
        "AND source_payload_sha256 = "
        "'f99fa6c8f2a1364d56a0d0bff9d7401b1f504f3b020c321d996f05a70a05df42' "
        "AND source_log_prefix_sha256 = "
        "'c287af99368bf95d16330513f155f795b5fe77bc1e595c47d2fed7a1a447addb' "
        "AND source_log_prefix_bytes = 25665 "
        "AND source_artifact_inventory_sha256 = "
        "'c0272f59ca8878d26e95db2b4328cfc9551c4b3dff4382a237c38cd87f00f63e' "
        "AND target_artifact_path = "
        "'/data/artifacts/formal-backtest-recoveries/"
        "0a113fe28ca741b6be9c09ab046c9d02/attempt-2') "
        "IS TRUE",
        name="ck_formal_backtest_interruption_recovery_v17_source",
    ),
    CheckConstraint(
        "(jsonb_typeof(verification_json) = 'object' "
        "AND verification_json ->> 'receipt_sha256' = receipt_sha256 "
        "AND verification_json ->> 'contract_version' = contract_version "
        "AND verification_json ->> 'recovery_generation' = recovery_generation "
        "AND verification_json ->> 'reason_code' = reason_code "
        "AND (verification_json ->> 'performance_information_used')::boolean = false "
        "AND verification_json -> 'source_job' ->> 'id' = job_id "
        "AND verification_json -> 'source_job' ->> 'row_sha256' = "
        "source_job_row_sha256 "
        "AND verification_json -> 'source_backtest' ->> 'id' = backtest_id "
        "AND verification_json -> 'source_backtest' ->> 'row_sha256' = "
        "source_backtest_row_sha256 "
        "AND verification_json -> 'target' ->> 'job_id' = job_id "
        "AND verification_json -> 'target' ->> 'backtest_id' = backtest_id "
        "AND verification_json -> 'target' ->> 'strategy_version_id' = "
        "strategy_version_id "
        "AND verification_json -> 'target' ->> 'artifact_path' = "
        "target_artifact_path "
        "AND (verification_json -> 'target' ->> 'authorized_attempt')::integer = 2 "
        "AND (verification_json -> 'target' ->> 'max_attempts')::integer = 2 "
        "AND verification_json -> 'external_interruption' ->> 'service' = "
        "'evaluation-worker' "
        "AND verification_json -> 'external_interruption' ->> 'stop_owner' = "
        "'quantlab-backup.service' "
        "AND verification_json -> 'external_interruption' ->> 'signal' = 'SIGTERM' "
        "AND (verification_json -> 'external_interruption' ->> "
        "'has_been_manually_stopped')::boolean = true "
        "AND (verification_json -> 'external_interruption' ->> 'oom_killed')::boolean = false "
        "AND verification_json -> 'external_interruption' ->> 'observed_at' = "
        "'2026-08-30T19:30:09.189629+00:00' "
        "AND verification_json -> 'external_interruption' ->> 'journal_sha256' = "
        "'c7a93f1fe86a57ed6fcd51f32fe74be55ef7115cb4a79454e12ccbee338efcc7' "
        "AND verification_json -> 'external_interruption' -> 'journal_excerpt' = '"
        + _FORMAL_BACKTEST_INTERRUPTION_JOURNAL_EXCERPT_SQL
        + "'::jsonb) "
        "IS TRUE",
        name="ck_formal_backtest_interruption_recovery_receipt",
    ),
    CheckConstraint(
        "(verification_json = '"
        + _FORMAL_BACKTEST_INTERRUPTION_EXPECTED_RECEIPT_SQL
        + "'::jsonb) IS TRUE",
        name="ck_formal_backtest_interruption_recovery_exact_receipt",
    ),
)
Index(
    "idx_formal_backtest_interruption_recoveries_created",
    formal_backtest_interruption_recoveries.c.created_at.desc(),
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
    Column("dataset_roll_policy", String, nullable=False, server_default="pinned"),
    Column("dataset_lineage_id", String),
    Column("status", String, nullable=False),
    Column("base_currency", String, nullable=False),
    Column("hypothetical_initial_value", Numeric(20, 6), nullable=False),
    Column("risk_exposure_override", Float, nullable=False, server_default="1"),
    # standalone = sender candidate created via RecommendationStore;
    # allocation_member = structural sub-account owned by an allocation and
    # never eligible as the unique recommendation sender.
    Column("recommendation_scope", String, nullable=False, server_default="standalone"),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_recommendation_portfolios_status_updated",
    recommendation_portfolios.c.status,
    recommendation_portfolios.c.updated_at.desc(),
)
# Design 8.1/9.1: a single active recommendation sender at any moment.
Index(
    "uq_recommendation_portfolios_single_active_sender",
    recommendation_portfolios.c.status,
    unique=True,
    postgresql_where=text("status = 'active' AND recommendation_scope = 'standalone'"),
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
    # Two-dimension account action plan (design 8.4): per-instrument
    # action x execution_state with projected_position and the
    # keep/cancel/replace/new order plan. Holdings keep their legacy
    # increase/decrease export; this JSON carries the richer model.
    Column("account_actions_json", json_type),
    Column("cost_model_json", json_type, nullable=False),
    Column("policy_version", String, nullable=False),
    Column("backtest_engine_version", String, nullable=False),
    Column("dataset", String, nullable=False),
    Column("dataset_identity_sha256", String, nullable=False),
    Column("dataset_lineage_id", String),
    Column("strategy_version_id", String, nullable=False),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    UniqueConstraint("portfolio_id", "as_of_date", name="uq_recommendation_snapshots_as_of"),
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
    # New promotion-chain accounts bind to the exact isolated stage.  NULL is
    # retained for recommendation/allocation accounts and legacy rows created
    # before migration 0064.
    Column(
        "promotion_stage_id",
        String,
        ForeignKey("quantlab.strategy_promotion_stages.id", ondelete="RESTRICT"),
        nullable=True,
    ),
    Column("status", String, nullable=False),
    Column("base_currency", String, nullable=False),
    Column("benchmark", String),
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
    Column("daily_roll_policy", String, nullable=False, server_default="pinned"),
    Column("execution_roll_policy", String, nullable=False, server_default="pinned"),
    Column("daily_dataset_identity_sha256", String, nullable=False),
    Column("daily_dataset_lineage_id", String, nullable=False),
    Column("daily_field_contract_version", String, nullable=False),
    Column("execution_dataset_identity_sha256", String, nullable=False),
    Column("execution_dataset_lineage_id", String, nullable=False),
    Column("execution_field_contract_version", String, nullable=False),
    Column("execution_engine_version", String, nullable=False),
    Column("cost_schedule_version", String, nullable=False),
    Column("execution_policy_json", json_type, nullable=False),
    # 单位化 TWR 链状态（设计 4.4）：与人民币 NAV 口径并存，NULL 表示链断裂。
    Column("investment_wealth", Float, nullable=True),
    Column("twr_high_water_mark", Float, nullable=True),
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
    postgresql_where=simulation_portfolios.c.promotion_stage_id.is_(None),
)

# One recoverable umbrella workflow per immutable daily Qlib publication.
autopilot_cycles = Table(
    "autopilot_cycles",
    metadata,
    Column("id", String, primary_key=True),
    Column("dataset", String, nullable=False),
    Column("dataset_identity_sha256", String, nullable=False),
    Column("dataset_lineage_id", String, nullable=False),
    Column("horizon_profile", String, nullable=False),
    Column("primary_label_policy_sha256", String, nullable=False),
    Column("status", String, nullable=False),
    Column("stage", String, nullable=False),
    Column("config_revision", Integer, nullable=False),
    Column("state_json", json_type, nullable=False),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    CheckConstraint(
        "status IN ('active', 'blocked', 'succeeded', 'paused')",
        name="ck_autopilot_cycles_status",
    ),
    CheckConstraint(
        "horizon_profile IN "
        "('short_1_5d', 'swing_1_6m', 'long_1_3y', 'legacy_ambiguous')",
        name="ck_autopilot_cycles_horizon",
    ),
    UniqueConstraint(
        "dataset_identity_sha256",
        "horizon_profile",
        name="uq_autopilot_cycle_dataset_horizon",
    ),
)
Index(
    "idx_autopilot_cycles_status_updated",
    autopilot_cycles.c.status,
    autopilot_cycles.c.updated_at.desc(),
)

autopilot_branches = Table(
    "autopilot_branches",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "cycle_id",
        String,
        ForeignKey("quantlab.autopilot_cycles.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("scenario", String, nullable=False),
    Column("scope_key", String, nullable=False),
    Column("status", String, nullable=False),
    Column("research_run_id", String, ForeignKey("quantlab.research_runs.id", ondelete="SET NULL")),
    Column("job_id", String, ForeignKey("quantlab.jobs.id", ondelete="SET NULL")),
    Column("details_json", json_type, nullable=False),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    UniqueConstraint("cycle_id", "scenario", "scope_key", name="uq_autopilot_branch_scope"),
    CheckConstraint(
        "status IN ('queued', 'running', 'evaluating', 'succeeded', "
        "'failed', 'blocked', 'skipped')",
        name="ck_autopilot_branches_status",
    ),
)
Index(
    "idx_autopilot_branches_cycle_status",
    autopilot_branches.c.cycle_id,
    autopilot_branches.c.status,
)

# One immutable pre-registered tournament for each decision stage in an
# Autopilot cycle.  This is the shared trial ledger: failed and rejected trials
# remain part of the family and therefore cannot disappear from the final
# multiple-testing count.
research_tournaments = Table(
    "research_tournaments",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "cycle_id",
        String,
        ForeignKey("quantlab.autopilot_cycles.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("stage", String, nullable=False),
    Column("dataset_identity_sha256", String, nullable=False),
    Column("status", String, nullable=False),
    Column("manifest_json", json_type, nullable=False),
    Column("manifest_sha256", String, nullable=False),
    Column("max_trials", Integer, nullable=False),
    Column("selected_trial_ids_json", json_type, nullable=False),
    Column("multiple_testing_json", json_type),
    Column("multiple_testing_sha256", String),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    UniqueConstraint("cycle_id", "stage", name="uq_research_tournament_cycle_stage"),
    CheckConstraint(
        "stage IN ('feature_screen', 'model_screen', 'model_full', "
        "'ensemble', 'quant', 'portfolio')",
        name="ck_research_tournament_stage",
    ),
    CheckConstraint(
        "status IN ('planned', 'running', 'succeeded', 'failed', 'blocked')",
        name="ck_research_tournament_status",
    ),
    CheckConstraint("max_trials > 0", name="ck_research_tournament_trial_limit"),
)
Index(
    "idx_research_tournaments_cycle_status",
    research_tournaments.c.cycle_id,
    research_tournaments.c.status,
)

research_tournament_trials = Table(
    "research_tournament_trials",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "tournament_id",
        String,
        ForeignKey("quantlab.research_tournaments.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "branch_id",
        String,
        ForeignKey("quantlab.autopilot_branches.id", ondelete="SET NULL"),
    ),
    Column("trial_kind", String, nullable=False),
    Column("name", String, nullable=False),
    Column("feature_set_id", String),
    Column("feature_set_definition_sha256", String),
    Column("model_family", String),
    Column("candidate_id", String),
    Column("status", String, nullable=False),
    Column("spec_json", json_type, nullable=False),
    Column("spec_sha256", String, nullable=False),
    Column("metrics_json", json_type),
    Column("evidence_json", json_type),
    Column("evidence_sha256", String),
    Column("resource_json", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("tournament_id", "name", name="uq_research_tournament_trial_name"),
    CheckConstraint(
        "trial_kind IN ('feature_set', 'model', 'model_ensemble', "
        "'quant_bundle', 'portfolio')",
        name="ck_research_tournament_trial_kind",
    ),
    CheckConstraint(
        "status IN ('preregistered', 'queued', 'running', 'passed', "
        "'failed', 'rejected', 'selected')",
        name="ck_research_tournament_trial_status",
    ),
)
Index(
    "idx_research_tournament_trials_status",
    research_tournament_trials.c.tournament_id,
    research_tournament_trials.c.status,
)

model_ensemble_candidates = Table(
    "model_ensemble_candidates",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "tournament_id",
        String,
        ForeignKey("quantlab.research_tournaments.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("name", String, nullable=False),
    Column("status", String, nullable=False),
    Column("dataset", String, nullable=False),
    Column("dataset_identity_sha256", String, nullable=False),
    Column("components_json", json_type, nullable=False),
    Column("combiner", String, nullable=False),
    Column("manifest_json", json_type, nullable=False),
    Column("manifest_sha256", String, nullable=False, unique=True),
    Column("admission_evidence_json", json_type),
    Column("admission_evidence_sha256", String),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("tournament_id", "name", name="uq_model_ensemble_tournament_name"),
    CheckConstraint(
        "status IN ('awaiting_evaluation', 'evaluating', 'research_admitted', "
        "'rejected', 'invalidated')",
        name="ck_model_ensemble_status",
    ),
    CheckConstraint("combiner = 'equal_rank'", name="ck_model_ensemble_combiner"),
)
Index(
    "idx_model_ensemble_candidates_status",
    model_ensemble_candidates.c.status,
    model_ensemble_candidates.c.updated_at.desc(),
)

model_ensemble_evaluations = Table(
    "model_ensemble_evaluations",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "model_ensemble_candidate_id",
        String,
        ForeignKey("quantlab.model_ensemble_candidates.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("profile_id", String, nullable=False),
    Column("seed", Integer, nullable=False),
    Column("metrics_json", json_type, nullable=False),
    Column("gate_status", String, nullable=False),
    Column("evidence_json", json_type, nullable=False),
    Column("evidence_sha256", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "model_ensemble_candidate_id",
        "profile_id",
        "seed",
        name="uq_model_ensemble_evaluation_grid",
    ),
    CheckConstraint(
        "gate_status IN ('passed', 'failed')",
        name="ck_model_ensemble_evaluation_gate",
    ),
)

# Cross-cycle type-I error control is reserved for capital-facing, one-shot
# final OOS tests. Research tournaments retain their own within-cycle Holm/PBO
# ledgers and do not spend or claim this capital alpha.
capital_oos_alpha_families = Table(
    "capital_oos_alpha_families",
    metadata,
    Column("id", String, primary_key=True),
    Column("capital_oos_family_sha256", String, nullable=False),
    Column("mandate_json", json_type, nullable=False),
    Column("total_alpha", Numeric(38, 28), nullable=False),
    Column("policy_json", json_type, nullable=False),
    Column("policy_sha256", String, nullable=False),
    Column("next_ordinal", Integer, nullable=False),
    Column("reserved_alpha", Numeric(38, 28), nullable=False),
    Column("settled_alpha", Numeric(38, 28), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "capital_oos_family_sha256", name="uq_capital_oos_alpha_family"
    ),
    CheckConstraint("total_alpha = 0.05", name="ck_capital_oos_alpha_total"),
    CheckConstraint(
        "capital_oos_family_sha256 ~ '^[0-9a-f]{64}$' "
        "AND policy_sha256 ~ '^[0-9a-f]{64}$' "
        "AND jsonb_typeof(mandate_json) = 'object' "
        "AND jsonb_typeof(policy_json) = 'object'",
        name="ck_capital_oos_alpha_family_identity",
    ),
    CheckConstraint(
        "next_ordinal > 0 AND reserved_alpha >= 0 AND settled_alpha >= 0 "
        "AND settled_alpha <= reserved_alpha AND reserved_alpha <= total_alpha",
        name="ck_capital_oos_alpha_family_counters",
    ),
)
Index(
    "idx_capital_oos_alpha_families_created",
    capital_oos_alpha_families.c.created_at,
)

capital_oos_legacy_attempts = Table(
    "capital_oos_legacy_attempts",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "oos_vintage_id",
        String,
        ForeignKey("quantlab.oos_vintages.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    ),
    Column("status", String, nullable=False),
    Column("scope", String, nullable=False),
    Column("dataset_identity", String, nullable=False),
    Column("dataset_lineage_id", String),
    Column("final_oos_start", Date, nullable=False),
    Column("final_oos_end", Date, nullable=False),
    Column("first_opened_at", DateTime(timezone=True), nullable=False),
    Column("consumed_at", DateTime(timezone=True)),
    Column("sealed_candidate_set_sha256", String, nullable=False),
    Column("raw_p_value", Float, nullable=False),
    Column("passed", Boolean, nullable=False),
    Column("failure_recorded", Boolean, nullable=False),
    Column("legacy_evidence_json", json_type, nullable=False),
    Column("legacy_evidence_sha256", String, nullable=False),
    Column(
        "reconciled_family_id",
        String,
        ForeignKey("quantlab.capital_oos_alpha_families.id", ondelete="RESTRICT"),
    ),
    Column("ordinal", Integer),
    Column("spent_alpha", Numeric(38, 28)),
    Column("reconciliation_json", json_type),
    Column("reconciliation_sha256", String),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("reconciled_at", DateTime(timezone=True)),
    UniqueConstraint(
        "reconciled_family_id",
        "ordinal",
        name="uq_capital_oos_legacy_family_ordinal",
    ),
    CheckConstraint(
        "status IN ('unreconciled', 'reconciled')",
        name="ck_capital_oos_legacy_status",
    ),
    CheckConstraint(
        "length(btrim(scope)) > 0 AND length(btrim(dataset_identity)) > 0 "
        "AND final_oos_end >= final_oos_start "
        "AND sealed_candidate_set_sha256 ~ '^[0-9a-f]{64}$' "
        "AND legacy_evidence_sha256 ~ '^[0-9a-f]{64}$' "
        "AND jsonb_typeof(legacy_evidence_json) = 'object' "
        "AND raw_p_value = 1 AND passed IS FALSE AND failure_recorded IS TRUE",
        name="ck_capital_oos_legacy_evidence",
    ),
    CheckConstraint(
        "(status = 'unreconciled' AND reconciled_family_id IS NULL "
        "AND ordinal IS NULL AND spent_alpha IS NULL "
        "AND reconciliation_json IS NULL AND reconciliation_sha256 IS NULL "
        "AND reconciled_at IS NULL) OR "
        "(status = 'reconciled' AND reconciled_family_id IS NOT NULL "
        "AND ordinal > 0 AND spent_alpha > 0 "
        "AND jsonb_typeof(reconciliation_json) = 'object' "
        "AND reconciliation_sha256 ~ '^[0-9a-f]{64}$' "
        "AND reconciled_at IS NOT NULL)",
        name="ck_capital_oos_legacy_reconciliation",
    ),
)
Index(
    "idx_capital_oos_legacy_status",
    capital_oos_legacy_attempts.c.status,
    capital_oos_legacy_attempts.c.first_opened_at,
)
Index(
    "idx_capital_oos_legacy_family_window",
    capital_oos_legacy_attempts.c.reconciled_family_id,
    capital_oos_legacy_attempts.c.final_oos_start,
    capital_oos_legacy_attempts.c.final_oos_end,
)

capital_oos_alpha_batches = Table(
    "capital_oos_alpha_batches",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "family_id",
        String,
        ForeignKey("quantlab.capital_oos_alpha_families.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("batch_key", String, nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("status", String, nullable=False),
    Column("frozen_bundle_manifest_sha256", String, nullable=False),
    Column("frozen_baseline_manifest_sha256", String, nullable=False),
    Column("dataset_lineage_id", String, nullable=False),
    Column("dataset_identity_sha256", String, nullable=False),
    Column("hypothesis_count", Integer, nullable=False),
    Column("research_data_end", Date, nullable=False),
    Column("final_oos_start", Date, nullable=False),
    Column("final_oos_end", Date, nullable=False),
    Column("trading_day_count", Integer, nullable=False),
    Column("trading_dates_json", json_type, nullable=False),
    Column("trading_dates_sha256", String, nullable=False),
    Column("embargo_trading_day_count", Integer, nullable=False),
    Column("embargo_trading_dates_json", json_type, nullable=False),
    Column("embargo_trading_dates_sha256", String, nullable=False),
    Column("preregistration_json", json_type, nullable=False),
    Column("preregistration_sha256", String, nullable=False),
    Column("batch_alpha", Numeric(38, 28), nullable=False),
    Column("raw_p_value", Float),
    Column("passed", Boolean),
    Column("failure_recorded", Boolean),
    Column("settlement_evidence_json", json_type),
    Column("settlement_evidence_sha256", String),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("settled_at", DateTime(timezone=True)),
    UniqueConstraint("family_id", "batch_key", name="uq_capital_oos_alpha_batch_key"),
    UniqueConstraint(
        "family_id", "ordinal", name="uq_capital_oos_alpha_batch_ordinal"
    ),
    CheckConstraint(
        "ordinal > 0 AND hypothesis_count = 1 AND batch_alpha > 0",
        name="ck_capital_oos_alpha_batch_values",
    ),
    CheckConstraint(
        "status IN ('reserved', 'settled')",
        name="ck_capital_oos_alpha_batch_status",
    ),
    CheckConstraint(
        "research_data_end < final_oos_start "
        "AND final_oos_end >= final_oos_start AND trading_day_count >= 252 "
        "AND embargo_trading_day_count >= 20 "
        "AND (embargo_trading_dates_json ->> 0)::date > research_data_end "
        "AND (embargo_trading_dates_json ->> "
        "(embargo_trading_day_count - 1))::date < final_oos_start",
        name="ck_capital_oos_alpha_batch_window",
    ),
    CheckConstraint(
        "frozen_bundle_manifest_sha256 ~ '^[0-9a-f]{64}$' "
        "AND frozen_baseline_manifest_sha256 ~ '^[0-9a-f]{64}$' "
        "AND dataset_lineage_id ~ '^[0-9a-f]{64}$' "
        "AND dataset_identity_sha256 ~ '^[0-9a-f]{64}$' "
        "AND trading_dates_sha256 ~ '^[0-9a-f]{64}$' "
        "AND embargo_trading_dates_sha256 ~ '^[0-9a-f]{64}$' "
        "AND preregistration_sha256 ~ '^[0-9a-f]{64}$' "
        "AND jsonb_typeof(trading_dates_json) = 'array' "
        "AND jsonb_array_length(trading_dates_json) = trading_day_count "
        "AND jsonb_typeof(embargo_trading_dates_json) = 'array' "
        "AND jsonb_array_length(embargo_trading_dates_json) = embargo_trading_day_count "
        "AND jsonb_typeof(preregistration_json) = 'object' "
        "AND (settlement_evidence_sha256 IS NULL OR "
        "settlement_evidence_sha256 ~ '^[0-9a-f]{64}$')",
        name="ck_capital_oos_alpha_batch_evidence",
    ),
    CheckConstraint(
        "(status = 'reserved' AND raw_p_value IS NULL AND passed IS NULL "
        "AND failure_recorded IS NULL AND settlement_evidence_json IS NULL "
        "AND settlement_evidence_sha256 IS NULL AND settled_at IS NULL) OR "
        "(status = 'settled' AND raw_p_value IS NOT NULL AND passed IS NOT NULL "
        "AND failure_recorded IS NOT NULL AND settlement_evidence_json IS NOT NULL "
        "AND settlement_evidence_sha256 IS NOT NULL AND settled_at IS NOT NULL)",
        name="ck_capital_oos_alpha_batch_settlement",
    ),
    CheckConstraint(
        "status = 'reserved' OR (raw_p_value > 0 AND raw_p_value <= 1 "
        "AND (failure_recorded IS FALSE OR "
        "(failure_recorded IS TRUE AND raw_p_value = 1 AND passed IS FALSE)))",
        name="ck_capital_oos_alpha_batch_result",
    ),
)
Index(
    "idx_capital_oos_alpha_batches_family_window",
    capital_oos_alpha_batches.c.family_id,
    capital_oos_alpha_batches.c.final_oos_start,
    capital_oos_alpha_batches.c.final_oos_end,
)
Index(
    "uq_capital_oos_alpha_one_reserved_family",
    capital_oos_alpha_batches.c.family_id,
    unique=True,
    postgresql_where=capital_oos_alpha_batches.c.status == "reserved",
)

# A date-level cursor makes the three-year PDF backfill resumable and auditable.
research_report_backfill_days = Table(
    "research_report_backfill_days",
    metadata,
    Column("report_date", Date, primary_key=True),
    Column("snapshot_name", String, nullable=False),
    Column("status", String, nullable=False),
    Column("attempts", Integer, nullable=False),
    Column("selected_count", Integer, nullable=False),
    Column("published_count", Integer, nullable=False),
    Column("blocked_count", Integer, nullable=False),
    Column("bytes_downloaded", BigInteger, nullable=False),
    Column("job_id", String, ForeignKey("quantlab.jobs.id", ondelete="SET NULL")),
    Column("last_error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    CheckConstraint(
        "status IN ('pending', 'queued', 'running', 'succeeded', 'blocked')",
        name="ck_research_report_backfill_status",
    ),
    CheckConstraint(
        "attempts >= 0 AND selected_count >= 0 AND published_count >= 0 "
        "AND blocked_count >= 0 AND bytes_downloaded >= 0",
        name="ck_research_report_backfill_counts",
    ),
)
Index(
    "idx_research_report_backfill_status_date",
    research_report_backfill_days.c.status,
    research_report_backfill_days.c.report_date.desc(),
)
Index(
    "uq_simulation_portfolios_promotion_stage",
    simulation_portfolios.c.promotion_stage_id,
    unique=True,
    postgresql_where=simulation_portfolios.c.promotion_stage_id.is_not(None),
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
    Column("daily_dataset", String, nullable=False),
    Column("daily_dataset_identity_sha256", String, nullable=False),
    Column("daily_dataset_lineage_id", String, nullable=False),
    Column("execution_dataset", String, nullable=False),
    Column("execution_dataset_identity_sha256", String, nullable=False),
    Column("execution_dataset_lineage_id", String, nullable=False),
    Column("simulation_semantics_sha256", String, nullable=False),
    Column("signal_date", Date, nullable=False),
    Column("trade_date", Date, nullable=False),
    Column("signal_at", DateTime(timezone=True)),
    Column("execution_not_before", DateTime(timezone=True)),
    Column("status", String, nullable=False),
    Column("idempotency_key", String, nullable=False, unique=True),
    Column(
        "account_netting_plan_id",
        String,
        ForeignKey("quantlab.account_netting_plans.id", ondelete="SET NULL"),
    ),
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
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="CASCADE"),
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
    Column("limit_price", Numeric(20, 8)),
    Column("not_before", DateTime(timezone=True)),
    Column("not_after", DateTime(timezone=True)),
    Column("target_version", String),
    Column(
        "account_netting_plan_id",
        String,
        ForeignKey("quantlab.account_netting_plans.id", ondelete="SET NULL"),
    ),
    Column("strategy_contributions_json", json_type),
    Column("plan_op", String),
    Column("cancel_reason", String),
    Column("updated_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "status IN ('planned', 'open', 'filled', 'partial_filled_expired', "
        "'rejected', 'expired', 'cancelled')",
        name="ck_simulation_orders_status",
    ),
)
Index("idx_simulation_orders_batch", simulation_orders.c.batch_id, simulation_orders.c.instrument)
Index(
    "idx_simulation_orders_portfolio_status",
    simulation_orders.c.portfolio_id,
    simulation_orders.c.status,
)

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

simulation_fee_adjustments = Table(
    "simulation_fee_adjustments",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "fill_id",
        String,
        ForeignKey("quantlab.simulation_fills.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "batch_id",
        String,
        ForeignKey("quantlab.simulation_batches.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("adjustment_key", String, nullable=False),
    Column("trade_date", Date, nullable=False),
    Column("source", String, nullable=False),
    Column("previously_confirmed_fee", Numeric(20, 6), nullable=False),
    Column("final_fee", Numeric(20, 6), nullable=False),
    Column("adjustment_amount", Numeric(20, 6), nullable=False),
    Column("evidence_sha256", String, nullable=False),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "portfolio_id",
        "adjustment_key",
        name="uq_simulation_fee_adjustments_key",
    ),
    CheckConstraint(
        "source IN ('end_of_day', 'user_import')",
        name="ck_simulation_fee_adjustments_source",
    ),
    CheckConstraint(
        "final_fee >= 0",
        name="ck_simulation_fee_adjustments_final_fee",
    ),
)
Index(
    "idx_simulation_fee_adjustments_fill",
    simulation_fee_adjustments.c.fill_id,
    simulation_fee_adjustments.c.created_at,
)

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
    Column("frozen_quantity", Integer, nullable=False, server_default="0"),
    Column("average_cost", Numeric(20, 8), nullable=False),
    Column("last_trade_date", Date),
    Column("market_price", Numeric(20, 8)),
    Column("market_date", Date),
    Column("stale", Boolean, nullable=False),
    Column("market_value", Numeric(20, 6), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "frozen_quantity >= 0 AND frozen_quantity <= available_quantity",
        name="ck_simulation_positions_frozen_quantity",
    ),
)

simulation_position_reservations = Table(
    "simulation_position_reservations",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "order_id",
        String,
        ForeignKey("quantlab.simulation_orders.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    ),
    Column("instrument", String, nullable=False),
    Column("reserved_quantity", Integer, nullable=False),
    Column("remaining_quantity", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "reserved_quantity > 0 AND remaining_quantity >= 0 "
        "AND remaining_quantity <= reserved_quantity",
        name="ck_simulation_position_reservations_quantity",
    ),
)
Index(
    "idx_simulation_position_reservations_portfolio_instrument",
    simulation_position_reservations.c.portfolio_id,
    simulation_position_reservations.c.instrument,
    simulation_position_reservations.c.remaining_quantity,
)

simulation_security_events = Table(
    "simulation_security_events",
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
    Column(
        "order_id",
        String,
        ForeignKey("quantlab.simulation_orders.id", ondelete="CASCADE"),
    ),
    Column("event_key", String, nullable=False),
    Column("event_type", String, nullable=False),
    Column("instrument", String, nullable=False),
    Column("quantity", Integer, nullable=False),
    Column("details_json", json_type, nullable=False),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "portfolio_id",
        "event_key",
        name="uq_simulation_security_events_key",
    ),
    CheckConstraint(
        "event_type IN ('freeze', 'consume', 'release', 'reclassify')",
        name="ck_simulation_security_events_type",
    ),
    CheckConstraint(
        "quantity > 0",
        name="ck_simulation_security_events_quantity",
    ),
)
Index(
    "idx_simulation_security_events_portfolio_time",
    simulation_security_events.c.portfolio_id,
    simulation_security_events.c.occurred_at,
)

simulation_day_attributions = Table(
    "simulation_day_attributions",
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
        nullable=False,
    ),
    Column("trade_date", Date, nullable=False),
    Column("strategy_json", json_type, nullable=False),
    Column("industry_json", json_type, nullable=False),
    Column("asset_json", json_type, nullable=False),
    Column("cost_json", json_type, nullable=False),
    Column("execution_json", json_type, nullable=False),
    Column("coverage_status", String, nullable=False),
    Column("blocker_reasons_json", json_type, nullable=False),
    Column("input_sha256", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("batch_id", name="uq_simulation_day_attributions_batch"),
    CheckConstraint(
        "coverage_status IN ('complete', 'partial')",
        name="ck_simulation_day_attributions_coverage",
    ),
)
Index(
    "idx_simulation_day_attributions_portfolio_date",
    simulation_day_attributions.c.portfolio_id,
    simulation_day_attributions.c.trade_date,
)

simulation_position_lots = Table(
    "simulation_position_lots",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("instrument", String, nullable=False),
    Column("lot_key", String, nullable=False),
    # NULL = acquired date unknown (legacy lots); dividend tax uses the top rate.
    Column("acquired_at", Date),
    Column("sellable_from", Date, nullable=False),
    Column("quantity", Integer, nullable=False),
    Column("cost_basis_total", Numeric(20, 6), nullable=False),
    Column("origin", String, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "portfolio_id", "instrument", "lot_key", name="uq_simulation_position_lots_key"
    ),
)
Index(
    "idx_simulation_position_lots_portfolio",
    simulation_position_lots.c.portfolio_id,
    simulation_position_lots.c.instrument,
)

simulation_dividend_entitlements = Table(
    "simulation_dividend_entitlements",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("instrument", String, nullable=False),
    Column("lot_key", String, nullable=False),
    Column("record_date", Date, nullable=False),
    # cash = pretax cash dividend per share; bonus_par = bonus ratio x par value.
    Column("kind", String, nullable=False),
    Column("income_per_share", Numeric(20, 8), nullable=False),
    Column("untaxed_quantity", Integer, nullable=False),
    # 除权日按除权时点持有期档位（逐批次保守上界）计提的每股应付税负债；
    # 随 untaxed_quantity 消耗同比例释放。旧行缺省 0 = 未计提。
    Column("liability_per_share", Numeric(20, 8), nullable=False, server_default="0"),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "portfolio_id",
        "instrument",
        "lot_key",
        "record_date",
        "kind",
        name="uq_simulation_dividend_entitlements_key",
    ),
)
Index(
    "idx_simulation_dividend_entitlements_portfolio",
    simulation_dividend_entitlements.c.portfolio_id,
    simulation_dividend_entitlements.c.instrument,
)

simulation_dividend_actions = Table(
    "simulation_dividend_actions",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("instrument", String, nullable=False),
    Column("ex_date", Date, nullable=False),
    Column("record_date", Date),
    Column("pay_date", Date),
    Column("eligible_quantity", Integer, nullable=False),
    Column("cash_per_share", Numeric(20, 8), nullable=False, server_default="0"),
    Column("receivable_amount", Numeric(20, 6), nullable=False, server_default="0"),
    # 除权确认时按保守档位计提的应付股息税负债（含现金分红与送股面值两类）。
    Column("tax_liability_amount", Numeric(20, 6), nullable=False, server_default="0"),
    Column("bonus_share_ratio", Float, nullable=False, server_default="0"),
    Column("conversion_ratio", Float, nullable=False, server_default="0"),
    Column("new_shares", Integer, nullable=False, server_default="0"),
    Column("div_listdate", Date),
    # accrued = confirmed at ex-date; paid = reclassified to cash at pay-date.
    Column("status", String, nullable=False, server_default="accrued"),
    Column("tax_rule_version", String, nullable=False),
    Column("valuation_uncertain", Boolean, nullable=False, server_default="false"),
    Column("payload_sha256", String, nullable=False),
    Column("batch_id", String),
    Column("paid_batch_id", String),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "portfolio_id", "instrument", "ex_date", name="uq_simulation_dividend_actions_key"
    ),
)
Index(
    "idx_simulation_dividend_actions_status",
    simulation_dividend_actions.c.portfolio_id,
    simulation_dividend_actions.c.status,
)

# 非分红类公司行动台账（设计 §5.6 类型扩展）：公告/名称变更/拆并股/代码
# 变更/持有人选择/unsupported 事件共用一张 append-only 表；event_key 是
# 唯一幂等键，任何事件在账户内只应用一次。
simulation_corporate_events = Table(
    "simulation_corporate_events",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("event_key", String, nullable=False),
    Column("event_type", String, nullable=False),
    Column("instrument", String, nullable=False),
    Column("effective_date", Date, nullable=False),
    Column("payload_sha256", String, nullable=False),
    Column("details_json", json_type, nullable=False),
    Column("batch_id", String),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("portfolio_id", "event_key", name="uq_simulation_corporate_events_key"),
)
Index(
    "idx_simulation_corporate_events_portfolio_date",
    simulation_corporate_events.c.portfolio_id,
    simulation_corporate_events.c.effective_date,
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

simulation_cash_lots = Table(
    "simulation_cash_lots",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("lot_key", String, nullable=False),
    Column("source_type", String, nullable=False),
    Column("source_reference_id", String),
    # Free and frozen are mutually exclusive classifications of one economic
    # cash asset. Availability timestamps are permissions on free cash, not
    # additional assets.
    Column("free_amount", Numeric(20, 6), nullable=False),
    Column("frozen_amount", Numeric(20, 6), nullable=False),
    Column("tradable_at", DateTime(timezone=True), nullable=False),
    Column("withdrawable_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "portfolio_id",
        "lot_key",
        name="uq_simulation_cash_lots_key",
    ),
    CheckConstraint(
        "free_amount >= 0 AND frozen_amount >= 0",
        name="ck_simulation_cash_lots_nonnegative",
    ),
)
Index(
    "idx_simulation_cash_lots_portfolio_availability",
    simulation_cash_lots.c.portfolio_id,
    simulation_cash_lots.c.tradable_at,
    simulation_cash_lots.c.withdrawable_at,
)

simulation_cash_events = Table(
    "simulation_cash_events",
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
    Column(
        "order_id",
        String,
        ForeignKey("quantlab.simulation_orders.id", ondelete="CASCADE"),
    ),
    Column("event_key", String, nullable=False),
    Column("event_type", String, nullable=False),
    Column("amount", Numeric(20, 6), nullable=False),
    Column("details_json", json_type, nullable=False),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "portfolio_id",
        "event_key",
        name="uq_simulation_cash_events_key",
    ),
    CheckConstraint(
        "event_type IN ('create', 'freeze', 'consume_free', "
        "'consume_frozen', 'release', 'reclassify')",
        name="ck_simulation_cash_events_type",
    ),
    CheckConstraint(
        "amount >= 0",
        name="ck_simulation_cash_events_amount",
    ),
)
Index(
    "idx_simulation_cash_events_portfolio_time",
    simulation_cash_events.c.portfolio_id,
    simulation_cash_events.c.occurred_at,
)

simulation_cash_event_allocations = Table(
    "simulation_cash_event_allocations",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "event_id",
        String,
        ForeignKey("quantlab.simulation_cash_events.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "cash_lot_id",
        String,
        ForeignKey("quantlab.simulation_cash_lots.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("action", String, nullable=False),
    Column("amount", Numeric(20, 6), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "event_id",
        "cash_lot_id",
        "action",
        name="uq_simulation_cash_event_allocations",
    ),
    CheckConstraint(
        "action IN ('create', 'freeze', 'consume_free', 'consume_frozen', 'release', 'reclassify')",
        name="ck_simulation_cash_event_allocations_action",
    ),
    CheckConstraint(
        "amount > 0",
        name="ck_simulation_cash_event_allocations_amount",
    ),
)

simulation_cash_reservations = Table(
    "simulation_cash_reservations",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "order_id",
        String,
        ForeignKey("quantlab.simulation_orders.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "cash_lot_id",
        String,
        ForeignKey("quantlab.simulation_cash_lots.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("reserved_amount", Numeric(20, 6), nullable=False),
    Column("remaining_amount", Numeric(20, 6), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "order_id",
        "cash_lot_id",
        name="uq_simulation_cash_reservations_order_lot",
    ),
    CheckConstraint(
        "reserved_amount > 0 AND remaining_amount >= 0 AND remaining_amount <= reserved_amount",
        name="ck_simulation_cash_reservations_amounts",
    ),
)
Index(
    "idx_simulation_cash_reservations_order",
    simulation_cash_reservations.c.order_id,
    simulation_cash_reservations.c.remaining_amount,
)

simulation_external_flows = Table(
    "simulation_external_flows",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="CASCADE"),
        nullable=False,
    ),
    # 调用方稳定幂等键；缺省由载荷哈希派生，重放同载荷不产生第二行。
    Column("flow_key", String, nullable=False),
    Column("trade_date", Date, nullable=False),
    # open=开盘前确认可用于当日决策；close=盘后确认，次日才进入可投资现金。
    Column("timing", String, nullable=False),
    # 入金为正、出金为负（设计 4.4）；外部现金流不计损益。
    Column("amount", Numeric(20, 6), nullable=False),
    Column("note", Text, nullable=True),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "timing IN ('open', 'close')",
        name="ck_simulation_external_flows_timing",
    ),
)
Index(
    "uq_simulation_external_flows_key",
    simulation_external_flows.c.portfolio_id,
    simulation_external_flows.c.flow_key,
    unique=True,
)
Index(
    "idx_simulation_external_flows_portfolio_date",
    simulation_external_flows.c.portfolio_id,
    simulation_external_flows.c.trade_date,
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
    Column("corporate_receivables", Numeric(20, 6), nullable=False, server_default="0"),
    # 除权日按保守档位计提的未结算股息税负债（NAV 减项，随卖出结算释放）。
    Column("corporate_tax_liabilities", Numeric(20, 6), nullable=False, server_default="0"),
    Column("nav", Numeric(20, 6), nullable=False),
    Column("daily_return", Float, nullable=False),
    Column("benchmark_close", Numeric(20, 8)),
    Column("benchmark_return", Float),
    Column("benchmark_wealth", Float),
    Column("drawdown", Float, nullable=False),
    # 当日外部现金流（开盘前/盘后确认）；单位化 TWR 链状态与回撤（设计 4.4）。
    Column("external_flow_open", Numeric(20, 6), nullable=False, server_default="0"),
    Column("external_flow_close", Numeric(20, 6), nullable=False, server_default="0"),
    Column("twr_daily_return", Float, nullable=True),
    Column("investment_wealth", Float, nullable=True),
    Column("twr_drawdown", Float, nullable=True),
    Column("twr_status", String, nullable=False, server_default="unavailable_legacy"),
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
    # Frozen decision calendar for AllocationArtifacts: member budgets are
    # only re-solved on decision days (weekly/monthly); every other task
    # reuses the still-valid artifact (design 6.10).
    Column("decision_frequency", String, nullable=False, server_default="monthly"),
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
# Design 6.10: a single user-selected active allocation policy at any moment.
Index(
    "uq_strategy_allocations_single_active",
    strategy_allocations.c.status,
    unique=True,
    postgresql_where=text("status = 'active'"),
)

strategy_allocation_artifacts = Table(
    "strategy_allocation_artifacts",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "allocation_id",
        String,
        ForeignKey("quantlab.strategy_allocations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("decision_date", Date, nullable=False),
    Column("inputs_as_of", Date, nullable=False),
    Column("valid_until", Date, nullable=False),
    Column("member_weights_json", json_type, nullable=False),
    Column("analysis_json", json_type, nullable=False),
    Column("artifact_hash", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_strategy_allocation_artifacts_allocation",
    strategy_allocation_artifacts.c.allocation_id,
    strategy_allocation_artifacts.c.decision_date.desc(),
)

# Design 6.10/8.1/9.2: account-level netted target plans. One row per
# idempotent plan key (account + artifact + decision date + inputs as_of +
# policy version + tranche index); the full plan (net targets, signed net
# trades, strategy_contributions, cash remainder, execution policy reference)
# lives in plan_json for the execution layer to consume.
account_netting_plans = Table(
    "account_netting_plans",
    metadata,
    Column("id", String, primary_key=True),
    Column("plan_key", String, nullable=False, unique=True),
    Column("account_id", String, nullable=False),
    Column(
        "allocation_artifact_id",
        String,
        ForeignKey("quantlab.strategy_allocation_artifacts.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("decision_date", Date, nullable=False),
    Column("inputs_as_of", Date, nullable=False),
    Column("policy_version", String, nullable=False),
    Column("execution_policy", String, nullable=False),
    Column("tranche_index", Integer, nullable=False, server_default="0"),
    Column("plan_hash", String, nullable=False),
    Column("plan_json", json_type, nullable=False),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_account_netting_plans_account",
    account_netting_plans.c.account_id,
    account_netting_plans.c.decision_date.desc(),
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
    Column(
        "economic_hypothesis_group",
        String,
        nullable=False,
        server_default="legacy-unclassified",
    ),
    Column("hypothesis_group_cap", Float, nullable=False, server_default="0.70"),
    Column("shared_experiment_count", Integer, nullable=False, server_default="1"),
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

platform_safe_mode_state = Table(
    "platform_safe_mode_state",
    metadata,
    Column("id", String, primary_key=True),
    Column("active", Boolean, nullable=False, server_default="false"),
    Column("reason", Text, nullable=False, server_default=""),
    Column("source", String, nullable=False, server_default=""),
    Column("triggered_by", String, nullable=False, server_default=""),
    Column("triggered_at", DateTime(timezone=True)),
    Column("details_json", json_type),
    Column("cleared_by", String),
    Column("cleared_at", DateTime(timezone=True)),
    Column("clear_reason", Text),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

# Versioned personal market permissions (design draft 8.7). Each row is one
# immutable policy version for a scope (exchange/board/risk_warning/
# etf_subtype); the effective permission is the latest version with
# as_of <= today. New versions may only tighten (buy_sell → sell_only →
# disabled → unknown) unless the relaxation is explicitly confirmed.
market_permission_versions = Table(
    "market_permission_versions",
    metadata,
    Column("id", String, primary_key=True),
    Column("scope_type", String, nullable=False),
    Column("scope_key", String, nullable=False),
    Column("permission", String, nullable=False),
    Column("confirmation_source", Text, nullable=False),
    Column("as_of", Date, nullable=False),
    Column("valid_until", Date),
    Column("supersedes_id", String),
    Column("relaxation_confirmed", Boolean, nullable=False, server_default="false"),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_market_permission_scope",
    market_permission_versions.c.scope_type,
    market_permission_versions.c.scope_key,
    market_permission_versions.c.as_of.desc(),
)

# Manual/CSV shadow account snapshots (design draft 8.6): user-imported real
# holdings, cash, sellable quantities and open orders. Snapshots are immutable
# full-state imports; freshness is judged from imported_at and stale state
# degrades recommendations to simulation-only instead of silently falling
# back to the simulation ledger.
shadow_account_snapshots = Table(
    "shadow_account_snapshots",
    metadata,
    Column("id", String, primary_key=True),
    Column("account_id", String, nullable=False),
    Column("import_source", String, nullable=False),
    Column("cash", Numeric(20, 6), nullable=False),
    Column("holdings_json", json_type, nullable=False),
    Column("open_orders_json", json_type, nullable=False),
    Column("content_sha256", String, nullable=False),
    Column("notes", Text),
    Column("imported_by", String, nullable=False),
    Column("imported_at", DateTime(timezone=True), nullable=False),
)
Index(
    "idx_shadow_account_snapshots_account",
    shadow_account_snapshots.c.account_id,
    shadow_account_snapshots.c.imported_at.desc(),
)

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

# Versioned investor-owned paper-account input. No capital default is allowed:
# the first active profile must record an explicit user choice and explicit
# market permissions before account simulation can be enabled.
investor_simulation_profiles = Table(
    "investor_simulation_profiles",
    metadata,
    Column("id", String, primary_key=True),
    Column("profile_key", String, nullable=False),
    Column("version", Integer, nullable=False),
    Column("status", String, nullable=False),
    Column("supersedes_id", String, ForeignKey("quantlab.investor_simulation_profiles.id")),
    Column("initial_capital", Numeric(20, 6), nullable=False),
    Column("risk_profile", String, nullable=False),
    Column("min_cash_weight", Float, nullable=False),
    Column("max_gross_exposure", Float, nullable=False),
    Column("market_permissions_json", json_type, nullable=False),
    Column("content_sha256", String, nullable=False),
    Column("created_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_by", String, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "profile_key", "version", name="uq_investor_simulation_profile_version"
    ),
    CheckConstraint(
        "status IN ('draft', 'active', 'retired')",
        name="ck_investor_simulation_profile_status",
    ),
    CheckConstraint(
        "initial_capital > 0 AND min_cash_weight >= 0 AND min_cash_weight < 1 "
        "AND max_gross_exposure > 0 AND max_gross_exposure <= 1 "
        "AND min_cash_weight + max_gross_exposure <= 1",
        name="ck_investor_simulation_profile_risk_values",
    ),
    CheckConstraint(
        "length(trim(profile_key)) > 0 AND length(trim(risk_profile)) > 0 "
        "AND jsonb_typeof(market_permissions_json) = 'object' "
        "AND content_sha256 ~ '^[0-9a-f]{64}$'",
        name="ck_investor_simulation_profile_identity",
    ),
)
Index(
    "uq_investor_simulation_profile_active",
    investor_simulation_profiles.c.profile_key,
    unique=True,
    postgresql_where=investor_simulation_profiles.c.status == "active",
)
Index(
    "idx_investor_simulation_profiles_updated",
    investor_simulation_profiles.c.updated_at.desc(),
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
    if os.getenv("QUANTLAB_DATABASE_DISABLE_POOL", "").strip() == "1":
        # One-shot maintenance commands and the integration-test harness create
        # many short-lived Store instances. NullPool makes each context-managed
        # connection close at the server boundary and avoids retaining hundreds
        # of idle sessions. Long-running services keep the bounded pool below.
        return create_engine(database_url, pool_pre_ping=True, poolclass=NullPool)
    engine = create_engine(database_url, pool_pre_ping=True, pool_size=5, max_overflow=10)
    return engine


# Design 4.5/6.11: per-version pre-registered forward evidence gate. The gate
# may only be registered or changed before the version enters paper.
strategy_forward_gates = Table(
    "strategy_forward_gates",
    metadata,
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("min_forward_calendar_days", Integer, nullable=False),
    Column("min_forward_trading_days", Integer, nullable=False),
    Column("min_decision_batches", Integer, nullable=False),
    Column("min_completed_cycles", Integer, nullable=False),
    Column("min_closed_round_trips", Integer, nullable=False),
    Column("min_review_events", Integer, nullable=False),
    Column("min_financial_report_reviews", Integer, nullable=False),
    Column("min_data_completeness", Float, nullable=False),
    Column("min_reconciliation_rate", Float, nullable=False),
    Column("max_cost_deviation", Float, nullable=False),
    Column("criteria_json", json_type, nullable=False),
    Column("criteria_sha256", String, nullable=False),
    Column("registered_by", String, nullable=False),
    Column("registered_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "(jsonb_typeof(criteria_json) = 'object' "
        "AND criteria_json ?& ARRAY['contract_version','horizon_profile',"
        "'horizon_contract_sha256','thresholds'] "
        "AND (criteria_json - ARRAY['contract_version','horizon_profile',"
        "'horizon_contract_sha256','thresholds']) = '{}'::jsonb "
        "AND criteria_json ->> 'contract_version' = 'strategy-forward-gate-v2' "
        "AND criteria_json ->> 'horizon_profile' IN "
        "('short_1_5d','swing_1_6m','long_1_3y','legacy_ambiguous') "
        "AND criteria_json ->> 'horizon_contract_sha256' ~ '^[0-9a-f]{64}$' "
        "AND jsonb_typeof(criteria_json -> 'thresholds') = 'object' "
        "AND (criteria_json -> 'thresholds') ?& ARRAY["
        "'min_forward_calendar_days','min_forward_trading_days',"
        "'min_decision_batches','min_completed_cycles','min_closed_round_trips',"
        "'min_review_events','min_financial_report_reviews',"
        "'min_data_completeness','min_reconciliation_rate','max_cost_deviation'] "
        "AND ((criteria_json -> 'thresholds') - ARRAY["
        "'min_forward_calendar_days','min_forward_trading_days',"
        "'min_decision_batches','min_completed_cycles','min_closed_round_trips',"
        "'min_review_events','min_financial_report_reviews',"
        "'min_data_completeness','min_reconciliation_rate','max_cost_deviation']) "
        "= '{}'::jsonb "
        "AND (criteria_json -> 'thresholds' ->> 'min_forward_calendar_days')::integer "
        "IS NOT DISTINCT FROM min_forward_calendar_days "
        "AND (criteria_json -> 'thresholds' ->> 'min_forward_trading_days')::integer "
        "IS NOT DISTINCT FROM min_forward_trading_days "
        "AND (criteria_json -> 'thresholds' ->> 'min_decision_batches')::integer "
        "IS NOT DISTINCT FROM min_decision_batches "
        "AND (criteria_json -> 'thresholds' ->> 'min_completed_cycles')::integer "
        "IS NOT DISTINCT FROM min_completed_cycles "
        "AND (criteria_json -> 'thresholds' ->> 'min_closed_round_trips')::integer "
        "IS NOT DISTINCT FROM min_closed_round_trips "
        "AND (criteria_json -> 'thresholds' ->> 'min_review_events')::integer "
        "IS NOT DISTINCT FROM min_review_events "
        "AND (criteria_json -> 'thresholds' ->> 'min_financial_report_reviews')::integer "
        "IS NOT DISTINCT FROM min_financial_report_reviews "
        "AND (criteria_json -> 'thresholds' ->> 'min_data_completeness')::double precision "
        "IS NOT DISTINCT FROM min_data_completeness "
        "AND (criteria_json -> 'thresholds' ->> 'min_reconciliation_rate')::double precision "
        "IS NOT DISTINCT FROM min_reconciliation_rate "
        "AND (criteria_json -> 'thresholds' ->> 'max_cost_deviation')::double precision "
        "IS NOT DISTINCT FROM max_cost_deviation "
        "AND min_forward_calendar_days >= 0 "
        "AND min_forward_trading_days >= 0 AND min_closed_round_trips >= 0 "
        "AND min_decision_batches >= 0 AND min_completed_cycles >= 0 "
        "AND min_review_events >= 0 AND min_financial_report_reviews >= 0 "
        "AND min_data_completeness >= 0 AND min_data_completeness <= 1 "
        "AND min_reconciliation_rate >= 0 AND min_reconciliation_rate <= 1 "
        "AND max_cost_deviation >= 0 "
        "AND criteria_sha256 ~ '^[0-9a-f]{64}$') IS TRUE",
        name="ck_strategy_forward_gate_criteria",
    ),
)

# Design 9.5: each isolated forward paper stage owns its simulation account,
# contract hash and evidence scope. A substantive contract drift freezes the
# old stage read-only and opens a new one; evidence is never concatenated.
strategy_promotion_stages = Table(
    "strategy_promotion_stages",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("stage_index", Integer, nullable=False),
    Column(
        "simulation_portfolio_id",
        String,
        ForeignKey("quantlab.simulation_portfolios.id", ondelete="RESTRICT"),
        nullable=True,
    ),
    Column("status", String, nullable=False),
    Column("source_contract_hash", String),
    Column("initial_cash", Numeric(20, 6)),
    Column("opened_at", DateTime(timezone=True), nullable=False),
    Column("frozen_at", DateTime(timezone=True)),
    Column("freeze_reason", Text),
    Column("promoted_at", DateTime(timezone=True)),
    Column("created_by", String, nullable=False),
    UniqueConstraint(
        "strategy_version_id", "stage_index", name="uq_strategy_promotion_stages_index"
    ),
)
Index(
    "idx_strategy_promotion_stages_version",
    strategy_promotion_stages.c.strategy_version_id,
    strategy_promotion_stages.c.status,
)

# Point-in-time health evidence is append-only. Migration 0072 installs the
# database UPDATE/DELETE guard; projections may select the newest row but must
# never rewrite history.
strategy_health_snapshots = Table(
    "strategy_health_snapshots",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "strategy_version_id",
        String,
        ForeignKey("quantlab.strategy_versions.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("horizon_profile", String, nullable=False),
    Column("as_of", DateTime(timezone=True), nullable=False),
    Column("health_status", String, nullable=False),
    Column("criteria_json", json_type, nullable=False),
    Column("criteria_sha256", String, nullable=False),
    Column("evidence_json", json_type, nullable=False),
    Column("evidence_sha256", String, nullable=False),
    Column("snapshot_sha256", String, nullable=False),
    Column("recorded_by", String, nullable=False),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "strategy_version_id",
        "as_of",
        "snapshot_sha256",
        name="uq_strategy_health_snapshot_identity",
    ),
    CheckConstraint(
        "horizon_profile IN "
        "('short_1_5d', 'swing_1_6m', 'long_1_3y', 'legacy_ambiguous')",
        name="ck_strategy_health_horizon_profile",
    ),
    CheckConstraint(
        "health_status IN "
        "('healthy', 'watch', 'restricted', 'suspended', 'retired')",
        name="ck_strategy_health_status",
    ),
    CheckConstraint(
        "jsonb_typeof(criteria_json) = 'object' "
        "AND jsonb_typeof(evidence_json) = 'object' "
        "AND criteria_sha256 ~ '^[0-9a-f]{64}$' "
        "AND evidence_sha256 ~ '^[0-9a-f]{64}$' "
        "AND snapshot_sha256 ~ '^[0-9a-f]{64}$'",
        name="ck_strategy_health_seals",
    ),
)
Index(
    "idx_strategy_health_version_as_of",
    strategy_health_snapshots.c.strategy_version_id,
    strategy_health_snapshots.c.as_of.desc(),
)


def row_dict(row: Any) -> dict[str, Any]:
    result = dict(row._mapping if hasattr(row, "_mapping") else row)
    for key, value in tuple(result.items()):
        if isinstance(value, datetime):
            result[key] = value.isoformat(timespec="seconds")
        elif isinstance(value, Decimal):
            result[key] = float(value)
    return result
