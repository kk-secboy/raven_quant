"""Add single-autopilot tournament and model-ensemble governance.

Revision ID: 0068_autopilot_tournaments
Revises: 0067_autopilot_report_backfill
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0068_autopilot_tournaments"
down_revision: str | None = "0067_autopilot_report_backfill"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON = sa.JSON().with_variant(JSONB(), "postgresql")
SCHEMA = "quantlab"


def upgrade() -> None:
    op.add_column("factor_candidates", sa.Column("admission_path", sa.String()), schema=SCHEMA)
    op.add_column(
        "factor_candidates", sa.Column("incremental_evidence_json", JSON), schema=SCHEMA
    )
    op.add_column(
        "factor_candidates",
        sa.Column("incremental_evidence_sha256", sa.String()),
        schema=SCHEMA,
    )

    op.create_table(
        "research_tournaments",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "cycle_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.autopilot_cycles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("dataset_identity_sha256", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("manifest_json", JSON, nullable=False),
        sa.Column("manifest_sha256", sa.String(), nullable=False),
        sa.Column("max_trials", sa.Integer(), nullable=False),
        sa.Column("selected_trial_ids_json", JSON, nullable=False),
        sa.Column("multiple_testing_json", JSON),
        sa.Column("multiple_testing_sha256", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("cycle_id", "stage", name="uq_research_tournament_cycle_stage"),
        sa.CheckConstraint(
            "stage IN ('feature_screen', 'model_screen', 'model_full', "
            "'ensemble', 'quant', 'portfolio')",
            name="ck_research_tournament_stage",
        ),
        sa.CheckConstraint(
            "status IN ('planned', 'running', 'succeeded', 'failed', 'blocked')",
            name="ck_research_tournament_status",
        ),
        sa.CheckConstraint("max_trials > 0", name="ck_research_tournament_trial_limit"),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_research_tournaments_cycle_status",
        "research_tournaments",
        ["cycle_id", "status"],
        schema=SCHEMA,
    )

    op.create_table(
        "research_tournament_trials",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "tournament_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.research_tournaments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "branch_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.autopilot_branches.id", ondelete="SET NULL"),
        ),
        sa.Column("trial_kind", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("feature_set_id", sa.String()),
        sa.Column("feature_set_definition_sha256", sa.String()),
        sa.Column("model_family", sa.String()),
        sa.Column("candidate_id", sa.String()),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("spec_json", JSON, nullable=False),
        sa.Column("spec_sha256", sa.String(), nullable=False),
        sa.Column("metrics_json", JSON),
        sa.Column("evidence_json", JSON),
        sa.Column("evidence_sha256", sa.String()),
        sa.Column("resource_json", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "tournament_id", "name", name="uq_research_tournament_trial_name"
        ),
        sa.CheckConstraint(
            "trial_kind IN ('feature_set', 'model', 'model_ensemble', "
            "'quant_bundle', 'portfolio')",
            name="ck_research_tournament_trial_kind",
        ),
        sa.CheckConstraint(
            "status IN ('preregistered', 'queued', 'running', 'passed', "
            "'failed', 'rejected', 'selected')",
            name="ck_research_tournament_trial_status",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_research_tournament_trials_status",
        "research_tournament_trials",
        ["tournament_id", "status"],
        schema=SCHEMA,
    )

    op.create_table(
        "model_ensemble_candidates",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "tournament_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.research_tournaments.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("dataset_identity_sha256", sa.String(), nullable=False),
        sa.Column("components_json", JSON, nullable=False),
        sa.Column("combiner", sa.String(), nullable=False),
        sa.Column("manifest_json", JSON, nullable=False),
        sa.Column("manifest_sha256", sa.String(), nullable=False, unique=True),
        sa.Column("admission_evidence_json", JSON),
        sa.Column("admission_evidence_sha256", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("tournament_id", "name", name="uq_model_ensemble_tournament_name"),
        sa.CheckConstraint(
            "status IN ('awaiting_evaluation', 'evaluating', 'research_admitted', "
            "'rejected', 'invalidated')",
            name="ck_model_ensemble_status",
        ),
        sa.CheckConstraint("combiner = 'equal_rank'", name="ck_model_ensemble_combiner"),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_model_ensemble_candidates_status",
        "model_ensemble_candidates",
        ["status", sa.text("updated_at DESC")],
        schema=SCHEMA,
    )
    # A frozen fin_quant bundle owns exactly one prediction component.  It can
    # be a single admitted model or an admitted equal-rank model ensemble, but
    # never both and never neither.
    op.alter_column(
        "quant_bundle_candidates",
        "model_candidate_id",
        existing_type=sa.String(),
        nullable=True,
        schema=SCHEMA,
    )
    op.add_column(
        "quant_bundle_candidates",
        sa.Column("model_ensemble_candidate_id", sa.String()),
        schema=SCHEMA,
    )
    op.create_foreign_key(
        "fk_quant_bundle_candidates_model_ensemble",
        "quant_bundle_candidates",
        "model_ensemble_candidates",
        ["model_ensemble_candidate_id"],
        ["id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
        ondelete="RESTRICT",
    )
    op.create_check_constraint(
        "ck_quant_bundle_candidates_prediction_xor",
        "quant_bundle_candidates",
        "(model_candidate_id IS NOT NULL AND model_ensemble_candidate_id IS NULL) OR "
        "(model_candidate_id IS NULL AND model_ensemble_candidate_id IS NOT NULL)",
        schema=SCHEMA,
    )
    op.create_table(
        "model_ensemble_evaluations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "model_ensemble_candidate_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.model_ensemble_candidates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("profile_id", sa.String(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("metrics_json", JSON, nullable=False),
        sa.Column("gate_status", sa.String(), nullable=False),
        sa.Column("evidence_json", JSON, nullable=False),
        sa.Column("evidence_sha256", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "model_ensemble_candidate_id",
            "profile_id",
            "seed",
            name="uq_model_ensemble_evaluation_grid",
        ),
        sa.CheckConstraint(
            "gate_status IN ('passed', 'failed')",
            name="ck_model_ensemble_evaluation_gate",
        ),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("model_ensemble_evaluations", schema=SCHEMA)
    op.drop_constraint(
        "ck_quant_bundle_candidates_prediction_xor",
        "quant_bundle_candidates",
        type_="check",
        schema=SCHEMA,
    )
    op.drop_constraint(
        "fk_quant_bundle_candidates_model_ensemble",
        "quant_bundle_candidates",
        type_="foreignkey",
        schema=SCHEMA,
    )
    op.drop_column(
        "quant_bundle_candidates", "model_ensemble_candidate_id", schema=SCHEMA
    )
    op.alter_column(
        "quant_bundle_candidates",
        "model_candidate_id",
        existing_type=sa.String(),
        nullable=False,
        schema=SCHEMA,
    )
    op.drop_index(
        "idx_model_ensemble_candidates_status",
        table_name="model_ensemble_candidates",
        schema=SCHEMA,
    )
    op.drop_table("model_ensemble_candidates", schema=SCHEMA)
    op.drop_index(
        "idx_research_tournament_trials_status",
        table_name="research_tournament_trials",
        schema=SCHEMA,
    )
    op.drop_table("research_tournament_trials", schema=SCHEMA)
    op.drop_index(
        "idx_research_tournaments_cycle_status",
        table_name="research_tournaments",
        schema=SCHEMA,
    )
    op.drop_table("research_tournaments", schema=SCHEMA)
    op.drop_column("factor_candidates", "incremental_evidence_sha256", schema=SCHEMA)
    op.drop_column("factor_candidates", "incremental_evidence_json", schema=SCHEMA)
    op.drop_column("factor_candidates", "admission_path", schema=SCHEMA)
