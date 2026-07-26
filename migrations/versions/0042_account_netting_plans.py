"""Account-level security netting plans (design draft 6.10/8.1/9.2).

Persists the netted account target produced after applying one frozen
AllocationArtifact: budgets applied once, per-member security targets merged
algebraically (opposite demands offset internally, only the net is traded),
with strategy_contributions attribution and an execution-policy reference.
The plan_key carries the design 9.2 stable-key semantics — account, final
target version (artifact), decision date, inputs as_of, policy version and
tranche index; strategy_id is never part of the key, so task retries replay
the stored plan instead of creating duplicate account orders.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0042_account_netting_plans"
down_revision: str | None = "0041_dividend_tax_liability"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "account_netting_plans",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("plan_key", sa.String(), nullable=False, unique=True),
        sa.Column("account_id", sa.String(), nullable=False),
        sa.Column(
            "allocation_artifact_id",
            sa.String(),
            sa.ForeignKey(
                "quantlab.strategy_allocation_artifacts.id", ondelete="RESTRICT"
            ),
            nullable=False,
        ),
        sa.Column("decision_date", sa.Date(), nullable=False),
        sa.Column("inputs_as_of", sa.Date(), nullable=False),
        sa.Column("policy_version", sa.String(), nullable=False),
        sa.Column("execution_policy", sa.String(), nullable=False),
        sa.Column("tranche_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("plan_hash", sa.String(), nullable=False),
        sa.Column("plan_json", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema="quantlab",
    )
    op.create_index(
        "idx_account_netting_plans_account",
        "account_netting_plans",
        ["account_id", sa.text("decision_date DESC")],
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_account_netting_plans_account",
        table_name="account_netting_plans",
        schema="quantlab",
    )
    op.drop_table("account_netting_plans", schema="quantlab")
