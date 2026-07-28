"""Versioned ModelArtifact lifecycle and safe routine refits (design 6.1/6.7).

Every fitted model is bound to one immutable StrategySpec, dataset identity,
training window, recipe, prediction hash and artifact hash.  A partial unique
index permits only one active artifact per StrategySpec version.  Routine
refits may rotate artifacts but cannot mutate the frozen model recipe.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0052_model_artifacts"
down_revision: str | None = "0051_simulation_fee_adjustments"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_artifacts",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("strategy_version_id", sa.String(), nullable=False),
        sa.Column("artifact_key", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("strategy_spec_sha256", sa.String(), nullable=False),
        sa.Column("model_recipe_sha256", sa.String(), nullable=False),
        sa.Column("model_recipe_json", sa.JSON(), nullable=False),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("dataset_identity_sha256", sa.String(), nullable=False),
        sa.Column("training_start", sa.Date(), nullable=False),
        sa.Column("training_end", sa.Date(), nullable=False),
        sa.Column("data_cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("scheduled_refit_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("artifact_sha256", sa.String(), nullable=False),
        sa.Column("predictions_sha256", sa.String(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("activated_by", sa.String(), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["strategy_version_id"],
            ["quantlab.strategy_versions.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "strategy_version_id",
            "artifact_key",
            name="uq_model_artifacts_key",
        ),
        sa.CheckConstraint(
            "status IN ('candidate', 'active', 'retired', 'failed', 'expired')",
            name="ck_model_artifacts_status",
        ),
        schema="quantlab",
    )
    op.create_index(
        "idx_model_artifacts_strategy_created",
        "model_artifacts",
        ["strategy_version_id", sa.text("created_at DESC")],
        schema="quantlab",
    )
    op.create_index(
        "uq_model_artifacts_active",
        "model_artifacts",
        ["strategy_version_id"],
        unique=True,
        schema="quantlab",
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_model_artifacts_active",
        table_name="model_artifacts",
        schema="quantlab",
    )
    op.drop_index(
        "idx_model_artifacts_strategy_created",
        table_name="model_artifacts",
        schema="quantlab",
    )
    op.drop_table("model_artifacts", schema="quantlab")
