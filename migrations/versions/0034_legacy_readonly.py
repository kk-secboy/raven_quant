"""Mark pre-convergence research and execution records as legacy."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0034_legacy_readonly"
down_revision: str | None = "0033_recommendation_pipeline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for table_name in ("factor_evaluations", "strategy_versions", "backtest_runs"):
        op.add_column(
            table_name,
            sa.Column("is_legacy", sa.Boolean(), nullable=False, server_default=sa.true()),
            schema="quantlab",
        )
        op.alter_column(
            table_name,
            "is_legacy",
            server_default=sa.false(),
            schema="quantlab",
        )
    for table_name in (
        "paper_portfolios",
        "paper_orders",
        "paper_fills",
        "pair_paper_portfolios",
        "pair_paper_orders",
        "pair_paper_fills",
    ):
        op.add_column(
            table_name,
            sa.Column("is_legacy", sa.Boolean(), nullable=False, server_default=sa.true()),
            schema="quantlab",
        )


def downgrade() -> None:
    for table_name in (
        "pair_paper_fills",
        "pair_paper_orders",
        "pair_paper_portfolios",
        "paper_fills",
        "paper_orders",
        "paper_portfolios",
        "backtest_runs",
        "strategy_versions",
        "factor_evaluations",
    ):
        op.drop_column(table_name, "is_legacy", schema="quantlab")
