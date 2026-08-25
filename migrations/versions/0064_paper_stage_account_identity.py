"""Bind each isolated paper simulation account to its promotion stage.

Legacy/non-promotion accounts retain ``promotion_stage_id = NULL`` and their
existing source/execution uniqueness.  New paper stages may therefore open a
fresh account after contract drift without colliding with the frozen stage's
account, while a partial unique index still permits only one account per
promotion stage.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0064_paper_stage_account"
down_revision: str | None = "0063_model_artifact_env"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "simulation_portfolios",
        sa.Column("promotion_stage_id", sa.String(), nullable=True),
        schema="quantlab",
    )
    op.create_foreign_key(
        "fk_simulation_portfolios_promotion_stage",
        "simulation_portfolios",
        "strategy_promotion_stages",
        ["promotion_stage_id"],
        ["id"],
        source_schema="quantlab",
        referent_schema="quantlab",
        ondelete="RESTRICT",
    )
    # Backfill only unambiguous legacy ownership.  An already-inconsistent
    # database remains readable/auditable with NULL instead of making the
    # migration destructive or choosing a stage arbitrarily.
    op.execute(
        """
        UPDATE quantlab.simulation_portfolios AS portfolio
           SET promotion_stage_id = stage.id
          FROM quantlab.strategy_promotion_stages AS stage
         WHERE stage.simulation_portfolio_id = portfolio.id
           AND NOT EXISTS (
               SELECT 1
                 FROM quantlab.strategy_promotion_stages AS other
                WHERE other.simulation_portfolio_id = portfolio.id
                  AND other.id <> stage.id
           )
        """
    )
    op.drop_index(
        "uq_simulation_portfolios_source_execution",
        table_name="simulation_portfolios",
        schema="quantlab",
    )
    op.create_index(
        "uq_simulation_portfolios_source_execution",
        "simulation_portfolios",
        ["source_type", "source_id", "execution_dataset"],
        unique=True,
        schema="quantlab",
        postgresql_where=sa.text("promotion_stage_id IS NULL"),
    )
    op.create_index(
        "uq_simulation_portfolios_promotion_stage",
        "simulation_portfolios",
        ["promotion_stage_id"],
        unique=True,
        schema="quantlab",
        postgresql_where=sa.text("promotion_stage_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_simulation_portfolios_promotion_stage",
        table_name="simulation_portfolios",
        schema="quantlab",
    )
    op.drop_index(
        "uq_simulation_portfolios_source_execution",
        table_name="simulation_portfolios",
        schema="quantlab",
    )
    op.create_index(
        "uq_simulation_portfolios_source_execution",
        "simulation_portfolios",
        ["source_type", "source_id", "execution_dataset"],
        unique=True,
        schema="quantlab",
    )
    op.drop_constraint(
        "fk_simulation_portfolios_promotion_stage",
        "simulation_portfolios",
        type_="foreignkey",
        schema="quantlab",
    )
    op.drop_column(
        "simulation_portfolios",
        "promotion_stage_id",
        schema="quantlab",
    )
