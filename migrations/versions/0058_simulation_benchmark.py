"""Persist the governed simulation benchmark and its reconciled return chain."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0058_simulation_benchmark"
down_revision: str | None = "0057_forward_rollover"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "simulation_portfolios",
        sa.Column("benchmark", sa.String()),
        schema="quantlab",
    )
    for name, column_type in (
        ("benchmark_close", sa.Numeric(20, 8)),
        ("benchmark_return", sa.Float()),
        ("benchmark_wealth", sa.Float()),
    ):
        op.add_column(
            "simulation_nav",
            sa.Column(name, column_type),
            schema="quantlab",
        )


def downgrade() -> None:
    for name in ("benchmark_wealth", "benchmark_return", "benchmark_close"):
        op.drop_column("simulation_nav", name, schema="quantlab")
    op.drop_column("simulation_portfolios", "benchmark", schema="quantlab")
