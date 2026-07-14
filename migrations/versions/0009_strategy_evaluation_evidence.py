"""Pin every immutable strategy factor to its exact evaluation evidence."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_strategy_factor_evidence"
down_revision: str | None = "0008_auth_rbac_audit"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "strategy_factors",
        sa.Column("factor_evaluation_id", sa.String(), nullable=True),
        schema="quantlab",
    )
    op.execute(
        """
        UPDATE quantlab.strategy_factors AS sf
        SET factor_evaluation_id = (
            SELECT fe.id
            FROM quantlab.factor_evaluations AS fe
            WHERE fe.factor_candidate_id = sf.factor_candidate_id
            ORDER BY fe.created_at DESC
            LIMIT 1
        )
        """
    )
    op.alter_column(
        "strategy_factors",
        "factor_evaluation_id",
        nullable=False,
        schema="quantlab",
    )
    op.create_foreign_key(
        "fk_strategy_factors_evaluation",
        "strategy_factors",
        "factor_evaluations",
        ["factor_evaluation_id"],
        ["id"],
        source_schema="quantlab",
        referent_schema="quantlab",
        ondelete="RESTRICT",
    )
    op.create_index(
        "idx_strategy_factors_evaluation",
        "strategy_factors",
        ["factor_evaluation_id"],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_strategy_factors_evaluation",
        table_name="strategy_factors",
        schema="quantlab",
    )
    op.drop_constraint(
        "fk_strategy_factors_evaluation",
        "strategy_factors",
        schema="quantlab",
        type_="foreignkey",
    )
    op.drop_column("strategy_factors", "factor_evaluation_id", schema="quantlab")
