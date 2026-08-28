"""Persist safe fitted checkpoints and training provenance on ModelArtifact.

Revision ID: 0069_model_artifact_checkpoints
Revises: 0068_autopilot_tournaments
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0069_model_artifact_checkpoints"
down_revision: str | None = "0068_autopilot_tournaments"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    # Nullable is intentional for pre-0069 rows.  Runtime admission fails
    # closed for those legacy rows; inventing a checkpoint hash in migration
    # would destroy the immutable provenance boundary.
    op.add_column("model_artifacts", sa.Column("checkpoint_path", sa.Text()), schema=SCHEMA)
    op.add_column(
        "model_artifacts", sa.Column("checkpoint_sha256", sa.String()), schema=SCHEMA
    )
    op.add_column(
        "model_artifacts", sa.Column("checkpoint_format", sa.String()), schema=SCHEMA
    )
    op.add_column(
        "model_artifacts",
        sa.Column("model_data_contract_sha256", sa.String()),
        schema=SCHEMA,
    )
    op.add_column(
        "model_artifacts", sa.Column("training_kind", sa.String()), schema=SCHEMA
    )
    op.add_column(
        "model_artifacts", sa.Column("training_evidence_json", JSON), schema=SCHEMA
    )
    op.add_column(
        "model_artifacts",
        sa.Column("training_evidence_sha256", sa.String()),
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_model_artifacts_checkpoint_format",
        "model_artifacts",
        "checkpoint_format IS NULL OR checkpoint_format IN "
        "('lightgbm_text', 'pytorch_state_dict', 'ridge_numeric_json')",
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_model_artifacts_training_kind",
        "model_artifacts",
        "training_kind IS NULL OR training_kind IN "
        "('formal_oos', 'monthly_retrain', 'early_retrain', 'daily_inference')",
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_model_artifacts_training_kind",
        "model_artifacts",
        schema=SCHEMA,
        type_="check",
    )
    op.drop_constraint(
        "ck_model_artifacts_checkpoint_format",
        "model_artifacts",
        schema=SCHEMA,
        type_="check",
    )
    for column in (
        "training_evidence_sha256",
        "training_evidence_json",
        "training_kind",
        "model_data_contract_sha256",
        "checkpoint_format",
        "checkpoint_sha256",
        "checkpoint_path",
    ):
        op.drop_column("model_artifacts", column, schema=SCHEMA)
