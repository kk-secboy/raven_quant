"""Add the immutable factor library and research SOTA ledger.

Revision ID: 0066_factor_library_sota
Revises: 0065_work_unit_page_group
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0066_factor_library_sota"
down_revision: str | None = "0065_work_unit_page_group"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON = sa.JSON().with_variant(JSONB(), "postgresql")
SCHEMA = "quantlab"


def upgrade() -> None:
    op.create_table(
        "factor_definitions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("expression", sa.Text(), nullable=False),
        sa.Column("expression_sha256", sa.String(), nullable=False, unique=True),
        sa.Column("definition_sha256", sa.String(), nullable=False, unique=True),
        sa.Column("required_fields_json", JSON, nullable=False),
        sa.Column("max_lookback_days", sa.Integer(), nullable=False),
        sa.Column("economic_family", sa.String(), nullable=False),
        sa.Column("family_tags_json", JSON, nullable=False),
        sa.Column("aliases_json", JSON, nullable=False),
        sa.Column("source_refs_json", JSON, nullable=False),
        sa.Column("availability_policy", sa.String(), nullable=False),
        sa.Column("qlib_commit", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("max_lookback_days >= 0", name="ck_factor_definition_lookback"),
        sa.CheckConstraint(
            "status IN ('registered', 'blocked', 'retired')",
            name="ck_factor_definition_status",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_factor_definitions_family_name",
        "factor_definitions",
        ["economic_family", "name"],
        schema=SCHEMA,
    )

    op.create_table(
        "factor_library_versions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("contract_version", sa.String(), nullable=False),
        sa.Column("definition_sha256", sa.String(), nullable=False, unique=True),
        sa.Column("member_count", sa.Integer(), nullable=False),
        sa.Column("source_alias_counts_json", JSON, nullable=False),
        sa.Column("qlib_commit", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retired_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("member_count > 0", name="ck_factor_library_member_count"),
        sa.CheckConstraint(
            "status IN ('active', 'retired')", name="ck_factor_library_status"
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "uq_factor_library_versions_active",
        "factor_library_versions",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
        schema=SCHEMA,
    )
    op.create_table(
        "factor_library_members",
        sa.Column(
            "library_version_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.factor_library_versions.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column(
            "factor_definition_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.factor_definitions.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.UniqueConstraint(
            "library_version_id", "ordinal", name="uq_factor_library_member_ordinal"
        ),
        schema=SCHEMA,
    )

    op.add_column(
        "factor_candidates",
        sa.Column("factor_definition_id", sa.String()),
        schema=SCHEMA,
    )
    op.add_column(
        "factor_candidates", sa.Column("economic_family", sa.String()), schema=SCHEMA
    )
    op.add_column(
        "factor_candidates", sa.Column("family_tags_json", JSON), schema=SCHEMA
    )
    op.add_column(
        "factor_candidates", sa.Column("similarity_cluster_id", sa.String()), schema=SCHEMA
    )
    op.create_foreign_key(
        "fk_factor_candidates_definition",
        "factor_candidates",
        "factor_definitions",
        ["factor_definition_id"],
        ["id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
        ondelete="RESTRICT",
    )
    op.create_index(
        "idx_factor_candidates_definition",
        "factor_candidates",
        ["factor_definition_id"],
        schema=SCHEMA,
    )
    op.create_index(
        "idx_factor_candidates_family_status",
        "factor_candidates",
        ["economic_family", "status"],
        schema=SCHEMA,
    )

    op.create_table(
        "factor_similarity_edges",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("dataset_identity_sha256", sa.String(), nullable=False),
        sa.Column("profile_id", sa.String(), nullable=False),
        sa.Column(
            "left_factor_candidate_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.factor_candidates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "right_factor_candidate_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.factor_candidates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("mean_abs_spearman", sa.Float(), nullable=False),
        sa.Column("relationship", sa.String(), nullable=False),
        sa.Column("cluster_id", sa.String()),
        sa.Column("evidence_json", JSON, nullable=False),
        sa.Column("evidence_sha256", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "dataset_identity_sha256",
            "profile_id",
            "left_factor_candidate_id",
            "right_factor_candidate_id",
            name="uq_factor_similarity_edge",
        ),
        sa.CheckConstraint(
            "left_factor_candidate_id <> right_factor_candidate_id",
            name="ck_factor_similarity_distinct",
        ),
        sa.CheckConstraint(
            "mean_abs_spearman >= 0 AND mean_abs_spearman <= 1",
            name="ck_factor_similarity_range",
        ),
        sa.CheckConstraint(
            "relationship IN ('independent', 'clustered', 'near_duplicate')",
            name="ck_factor_similarity_relationship",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "factor_definition_similarity_edges",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "library_version_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.factor_library_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("dataset_identity_sha256", sa.String(), nullable=False),
        sa.Column(
            "left_factor_definition_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.factor_definitions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "right_factor_definition_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.factor_definitions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("mean_abs_spearman", sa.Float(), nullable=False),
        sa.Column("relationship", sa.String(), nullable=False),
        sa.Column("cluster_id", sa.String(), nullable=False),
        sa.Column("evidence_json", JSON, nullable=False),
        sa.Column("evidence_sha256", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "library_version_id",
            "dataset_identity_sha256",
            "left_factor_definition_id",
            "right_factor_definition_id",
            name="uq_factor_definition_similarity_edge",
        ),
        sa.CheckConstraint(
            "left_factor_definition_id <> right_factor_definition_id",
            name="ck_factor_definition_similarity_distinct",
        ),
        sa.CheckConstraint(
            "mean_abs_spearman >= 0 AND mean_abs_spearman <= 1",
            name="ck_factor_definition_similarity_range",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "research_sota_versions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "library_version_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.factor_library_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("predecessor_id", sa.String()),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("dataset_identity_sha256", sa.String(), nullable=False),
        sa.Column("universe", sa.String(), nullable=False),
        sa.Column("label_horizon_days", sa.Integer(), nullable=False),
        sa.Column("periods_json", JSON, nullable=False),
        sa.Column("policy_json", JSON, nullable=False),
        sa.Column("policy_sha256", sa.String(), nullable=False),
        sa.Column("evidence_json", JSON, nullable=False),
        sa.Column("evidence_sha256", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("superseded_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(
            ["predecessor_id"],
            [f"{SCHEMA}.research_sota_versions.id"],
            ondelete="RESTRICT",
            name="fk_research_sota_predecessor",
        ),
        sa.CheckConstraint("label_horizon_days > 0", name="ck_research_sota_horizon"),
        sa.CheckConstraint(
            "status IN ('active', 'superseded')", name="ck_research_sota_status"
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "uq_research_sota_active_scope",
        "research_sota_versions",
        ["dataset", "universe", "label_horizon_days"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
        schema=SCHEMA,
    )
    op.create_table(
        "research_sota_members",
        sa.Column(
            "sota_version_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.research_sota_versions.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column(
            "factor_candidate_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.factor_candidates.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column(
            "factor_definition_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.factor_definitions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "factor_evaluation_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.factor_evaluations.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("economic_family", sa.String(), nullable=False),
        sa.Column("family_tags_json", JSON, nullable=False),
        sa.Column("similarity_cluster_id", sa.String(), nullable=False),
        sa.Column("member_rank", sa.Integer(), nullable=False),
        sa.Column("weight", sa.Float()),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("replaced_factor_candidate_id", sa.String()),
        sa.Column("incremental_evidence_json", JSON, nullable=False),
        sa.Column("incremental_evidence_sha256", sa.String(), nullable=False),
        sa.UniqueConstraint(
            "sota_version_id", "member_rank", name="uq_research_sota_member_rank"
        ),
        sa.CheckConstraint("member_rank >= 0", name="ck_research_sota_member_rank"),
        sa.CheckConstraint(
            "weight IS NULL OR (weight >= 0 AND weight <= 0.25)",
            name="ck_research_sota_member_weight",
        ),
        sa.CheckConstraint(
            "action IN ('retained', 'added', 'replaced')",
            name="ck_research_sota_member_action",
        ),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("research_sota_members", schema=SCHEMA)
    op.drop_index(
        "uq_research_sota_active_scope",
        table_name="research_sota_versions",
        schema=SCHEMA,
    )
    op.drop_table("research_sota_versions", schema=SCHEMA)
    op.drop_table("factor_definition_similarity_edges", schema=SCHEMA)
    op.drop_table("factor_similarity_edges", schema=SCHEMA)
    op.drop_index(
        "idx_factor_candidates_family_status",
        table_name="factor_candidates",
        schema=SCHEMA,
    )
    op.drop_index(
        "idx_factor_candidates_definition",
        table_name="factor_candidates",
        schema=SCHEMA,
    )
    op.drop_constraint(
        "fk_factor_candidates_definition",
        "factor_candidates",
        schema=SCHEMA,
        type_="foreignkey",
    )
    op.drop_column("factor_candidates", "similarity_cluster_id", schema=SCHEMA)
    op.drop_column("factor_candidates", "family_tags_json", schema=SCHEMA)
    op.drop_column("factor_candidates", "economic_family", schema=SCHEMA)
    op.drop_column("factor_candidates", "factor_definition_id", schema=SCHEMA)
    op.drop_table("factor_library_members", schema=SCHEMA)
    op.drop_index(
        "uq_factor_library_versions_active",
        table_name="factor_library_versions",
        schema=SCHEMA,
    )
    op.drop_table("factor_library_versions", schema=SCHEMA)
    op.drop_index(
        "idx_factor_definitions_family_name",
        table_name="factor_definitions",
        schema=SCHEMA,
    )
    op.drop_table("factor_definitions", schema=SCHEMA)
