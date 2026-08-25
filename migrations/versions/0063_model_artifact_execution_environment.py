"""Bind fitted model artifacts to the governed execution environment."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0063_model_artifact_env"
down_revision: str | None = "0062_research_asset_consumptions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Existing records remain readable for audit but cannot be activated as
    # governed model artifacts until rebuilt from formal evidence.
    op.add_column(
        "model_artifacts",
        sa.Column("execution_environment_sha256", sa.String()),
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_column(
        "model_artifacts",
        "execution_environment_sha256",
        schema="quantlab",
    )
