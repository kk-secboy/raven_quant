"""Add immutable RD-Agent research assets and governed candidates.

Revision ID: 0061_rdagent_governance
Revises: 0060_oos_stable_scope
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0061_rdagent_governance"
down_revision: str | None = "0060_oos_stable_scope"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON = sa.JSON().with_variant(JSONB(), "postgresql")
SCHEMA = "quantlab"


def upgrade() -> None:
    op.create_table(
        "research_assets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("asset_key", sa.String(), nullable=False, unique=True),
        sa.Column("asset_type", sa.String(), nullable=False),
        sa.Column("media_type", sa.String(), nullable=False),
        sa.Column("source_uri", sa.Text()),
        sa.Column("publisher", sa.String()),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("license_json", JSON, nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("manifest_json", JSON, nullable=False),
        sa.Column("manifest_sha256", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quarantined_at", sa.DateTime(timezone=True)),
        sa.Column("quarantine_reason", sa.Text()),
        sa.UniqueConstraint(
            "content_sha256", "manifest_sha256", name="uq_research_assets_content_manifest"
        ),
        sa.CheckConstraint(
            "status IN ('registered', 'quarantined', 'retired')",
            name="ck_research_assets_status",
        ),
        sa.CheckConstraint("size_bytes >= 0", name="ck_research_assets_size"),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_research_assets_type_created",
        "research_assets",
        ["asset_type", sa.text("created_at DESC")],
        schema=SCHEMA,
    )

    op.create_table(
        "research_run_artifacts",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "research_run_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.research_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("artifact_type", sa.String(), nullable=False),
        sa.Column("contract_version", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("manifest_json", JSON, nullable=False),
        sa.Column("manifest_sha256", sa.String(), nullable=False),
        sa.Column("producer", sa.String(), nullable=False),
        sa.Column("source_iteration", sa.Integer()),
        sa.Column(
            "capital_eligible",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("invalidated_by", sa.String()),
        sa.Column("invalidated_at", sa.DateTime(timezone=True)),
        sa.Column("invalidation_reason", sa.Text()),
        sa.UniqueConstraint(
            "research_run_id",
            "artifact_type",
            "content_sha256",
            name="uq_research_run_artifacts_content",
        ),
        sa.CheckConstraint(
            "status IN ('recorded', 'invalidated')",
            name="ck_research_run_artifacts_status",
        ),
        sa.CheckConstraint("size_bytes >= 0", name="ck_research_run_artifacts_size"),
        sa.CheckConstraint(
            "capital_eligible = false", name="ck_research_run_artifacts_non_capital"
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_research_run_artifacts_run_created",
        "research_run_artifacts",
        ["research_run_id", sa.text("created_at DESC")],
        schema=SCHEMA,
    )

    op.create_table(
        "model_candidates",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "research_run_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.research_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("source_iteration", sa.Integer()),
        sa.Column("model_type", sa.String(), nullable=False),
        sa.Column(
            "code_artifact_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.research_run_artifacts.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("code_sha256", sa.String(), nullable=False),
        sa.Column("architecture_json", JSON, nullable=False),
        sa.Column("model_hyperparameters_json", JSON, nullable=False),
        sa.Column("training_hyperparameters_json", JSON, nullable=False),
        sa.Column("base_features_manifest_json", JSON, nullable=False),
        sa.Column("base_features_manifest_sha256", sa.String(), nullable=False),
        sa.Column("feature_set_definition_sha256", sa.String(), nullable=False),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("dataset_identity_sha256", sa.String(), nullable=False),
        sa.Column("pre_final_end", sa.Date(), nullable=False),
        sa.Column("final_oos_start", sa.Date(), nullable=False),
        sa.Column("final_oos_end", sa.Date(), nullable=False),
        sa.Column("manifest_json", JSON, nullable=False),
        sa.Column("manifest_sha256", sa.String(), nullable=False),
        sa.Column("rdagent_decision", sa.Boolean()),
        sa.Column("rdagent_feedback", sa.Text()),
        sa.Column("admission_evidence_json", JSON),
        sa.Column("admission_evidence_sha256", sa.String()),
        sa.Column(
            "capital_eligible",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("admitted_by", sa.String()),
        sa.Column("admitted_at", sa.DateTime(timezone=True)),
        sa.Column("rejection_reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("research_run_id", "name", name="uq_model_candidates_run_name"),
        sa.CheckConstraint(
            "status IN ('awaiting_independent_evaluation', 'evaluating', 'evaluated', "
            "'research_admitted', 'rejected', 'invalidated')",
            name="ck_model_candidates_status",
        ),
        sa.CheckConstraint("capital_eligible = false", name="ck_model_candidates_non_capital"),
        sa.CheckConstraint(
            "pre_final_end < final_oos_start AND final_oos_start <= final_oos_end",
            name="ck_model_candidates_oos_boundary",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_model_candidates_status_updated",
        "model_candidates",
        ["status", sa.text("updated_at DESC")],
        schema=SCHEMA,
    )

    op.create_table(
        "model_evaluations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "model_candidate_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.model_candidates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "run_artifact_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.research_run_artifacts.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "oos_vintage_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.oos_vintages.id", ondelete="RESTRICT"),
        ),
        sa.Column("evidence_role", sa.String(), nullable=False),
        sa.Column("profile_id", sa.String(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("dataset_identity_sha256", sa.String(), nullable=False),
        sa.Column("train_start", sa.Date(), nullable=False),
        sa.Column("train_end", sa.Date(), nullable=False),
        sa.Column("valid_start", sa.Date(), nullable=False),
        sa.Column("valid_end", sa.Date(), nullable=False),
        sa.Column("final_oos_start", sa.Date(), nullable=False),
        sa.Column("final_oos_end", sa.Date(), nullable=False),
        sa.Column("metrics_json", JSON, nullable=False),
        sa.Column("metrics_sha256", sa.String(), nullable=False),
        sa.Column("gate_status", sa.String(), nullable=False),
        sa.Column("gate_reasons_json", JSON, nullable=False),
        sa.Column("evaluator_version", sa.String(), nullable=False),
        sa.Column("candidate_manifest_sha256", sa.String(), nullable=False),
        sa.Column("evidence_json", JSON, nullable=False),
        sa.Column("evidence_sha256", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "model_candidate_id",
            "evidence_role",
            "profile_id",
            "seed",
            name="uq_model_evaluations_grid",
        ),
        sa.CheckConstraint(
            "evidence_role IN ('official_feedback', 'independent_gate')",
            name="ck_model_evaluations_role",
        ),
        sa.CheckConstraint(
            "gate_status IN ('informational', 'passed', 'failed')",
            name="ck_model_evaluations_gate",
        ),
        sa.CheckConstraint(
            "(evidence_role = 'official_feedback' AND gate_status = 'informational') OR "
            "(evidence_role = 'independent_gate' AND gate_status IN ('passed', 'failed'))",
            name="ck_model_evaluations_role_gate",
        ),
        sa.CheckConstraint(
            "oos_vintage_id IS NULL",
            name="ck_model_evaluations_pre_final",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_model_evaluations_candidate_created",
        "model_evaluations",
        ["model_candidate_id", sa.text("created_at DESC")],
        schema=SCHEMA,
    )

    op.create_table(
        "quant_bundle_candidates",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "research_run_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.research_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("source_iteration", sa.Integer()),
        sa.Column(
            "model_candidate_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.model_candidates.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("factor_candidate_ids_json", JSON, nullable=False),
        sa.Column(
            "bundle_artifact_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.research_run_artifacts.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("bundle_artifact_sha256", sa.String(), nullable=False),
        sa.Column("base_features_manifest_json", JSON, nullable=False),
        sa.Column("base_features_manifest_sha256", sa.String(), nullable=False),
        sa.Column("feature_set_definition_sha256", sa.String(), nullable=False),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("dataset_identity_sha256", sa.String(), nullable=False),
        sa.Column("pre_final_end", sa.Date(), nullable=False),
        sa.Column("final_oos_start", sa.Date(), nullable=False),
        sa.Column("final_oos_end", sa.Date(), nullable=False),
        sa.Column("bundle_manifest_json", JSON, nullable=False),
        sa.Column("bundle_manifest_sha256", sa.String(), nullable=False),
        sa.Column("rdagent_decision", sa.Boolean()),
        sa.Column("rdagent_feedback", sa.Text()),
        sa.Column("ablation_evidence_json", JSON),
        sa.Column("ablation_evidence_sha256", sa.String()),
        sa.Column("admission_evidence_json", JSON),
        sa.Column("admission_evidence_sha256", sa.String()),
        sa.Column(
            "capital_eligible",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("admitted_by", sa.String()),
        sa.Column("admitted_at", sa.DateTime(timezone=True)),
        sa.Column("rejection_reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "research_run_id", "name", name="uq_quant_bundle_candidates_run_name"
        ),
        sa.CheckConstraint(
            "status IN ('awaiting_independent_evaluation', 'evaluating', 'evaluated', "
            "'research_admitted', 'rejected', 'invalidated')",
            name="ck_quant_bundle_candidates_status",
        ),
        sa.CheckConstraint(
            "capital_eligible = false", name="ck_quant_bundle_candidates_non_capital"
        ),
        sa.CheckConstraint(
            "pre_final_end < final_oos_start AND final_oos_start <= final_oos_end",
            name="ck_quant_bundle_candidates_oos_boundary",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_quant_bundle_candidates_status_updated",
        "quant_bundle_candidates",
        ["status", sa.text("updated_at DESC")],
        schema=SCHEMA,
    )

    op.create_table(
        "quant_bundle_evaluations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "quant_bundle_candidate_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.quant_bundle_candidates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "run_artifact_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.research_run_artifacts.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "oos_vintage_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.oos_vintages.id", ondelete="RESTRICT"),
        ),
        sa.Column("evidence_role", sa.String(), nullable=False),
        sa.Column("ablation", sa.String(), nullable=False),
        sa.Column("profile_id", sa.String(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("dataset_identity_sha256", sa.String(), nullable=False),
        sa.Column("train_start", sa.Date(), nullable=False),
        sa.Column("train_end", sa.Date(), nullable=False),
        sa.Column("valid_start", sa.Date(), nullable=False),
        sa.Column("valid_end", sa.Date(), nullable=False),
        sa.Column("final_oos_start", sa.Date(), nullable=False),
        sa.Column("final_oos_end", sa.Date(), nullable=False),
        sa.Column("metrics_json", JSON, nullable=False),
        sa.Column("metrics_sha256", sa.String(), nullable=False),
        sa.Column("gate_status", sa.String(), nullable=False),
        sa.Column("gate_reasons_json", JSON, nullable=False),
        sa.Column("evaluator_version", sa.String(), nullable=False),
        sa.Column("bundle_manifest_sha256", sa.String(), nullable=False),
        sa.Column("evidence_json", JSON, nullable=False),
        sa.Column("evidence_sha256", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "quant_bundle_candidate_id",
            "evidence_role",
            "ablation",
            "profile_id",
            "seed",
            name="uq_quant_bundle_evaluations_grid",
        ),
        sa.CheckConstraint(
            "evidence_role IN ('official_feedback', 'independent_gate')",
            name="ck_quant_bundle_evaluations_role",
        ),
        sa.CheckConstraint(
            "ablation IN ('factor_only', 'model_only', 'joint')",
            name="ck_quant_bundle_evaluations_ablation",
        ),
        sa.CheckConstraint(
            "gate_status IN ('informational', 'passed', 'failed')",
            name="ck_quant_bundle_evaluations_gate",
        ),
        sa.CheckConstraint(
            "(evidence_role = 'official_feedback' AND gate_status = 'informational') OR "
            "(evidence_role = 'independent_gate' AND gate_status IN ('passed', 'failed'))",
            name="ck_quant_bundle_evaluations_role_gate",
        ),
        sa.CheckConstraint(
            "oos_vintage_id IS NULL",
            name="ck_quant_bundle_evaluations_pre_final",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_quant_bundle_evaluations_candidate_created",
        "quant_bundle_evaluations",
        ["quant_bundle_candidate_id", sa.text("created_at DESC")],
        schema=SCHEMA,
    )

    op.create_table(
        "candidate_asset_links",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "asset_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.research_assets.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "factor_candidate_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.factor_candidates.id", ondelete="CASCADE"),
        ),
        sa.Column(
            "model_candidate_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.model_candidates.id", ondelete="CASCADE"),
        ),
        sa.Column(
            "quant_bundle_candidate_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.quant_bundle_candidates.id", ondelete="CASCADE"),
        ),
        sa.Column("relationship", sa.String(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "(factor_candidate_id IS NOT NULL AND model_candidate_id IS NULL AND "
            "quant_bundle_candidate_id IS NULL) OR "
            "(factor_candidate_id IS NULL AND model_candidate_id IS NOT NULL AND "
            "quant_bundle_candidate_id IS NULL) OR "
            "(factor_candidate_id IS NULL AND model_candidate_id IS NULL AND "
            "quant_bundle_candidate_id IS NOT NULL)",
            name="ck_candidate_asset_links_one_candidate",
        ),
        schema=SCHEMA,
    )
    for suffix, column in (
        ("factor", "factor_candidate_id"),
        ("model", "model_candidate_id"),
        ("quant", "quant_bundle_candidate_id"),
    ):
        op.create_index(
            f"uq_candidate_asset_links_{suffix}",
            "candidate_asset_links",
            [column, "asset_id", "relationship"],
            unique=True,
            schema=SCHEMA,
            postgresql_where=sa.text(f"{column} IS NOT NULL"),
        )


def downgrade() -> None:
    for table in (
        "candidate_asset_links",
        "quant_bundle_evaluations",
        "quant_bundle_candidates",
        "model_evaluations",
        "model_candidates",
        "research_run_artifacts",
        "research_assets",
    ):
        op.drop_table(table, schema=SCHEMA)
