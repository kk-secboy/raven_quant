"""Add the append-only transparent-baseline pre-result repair registry.

Revision ID: 0073_baseline_pre_result_repair
Revises: 0072_strategy_horizons
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0073_baseline_pre_result_repair"
down_revision: str | None = "0072_strategy_horizons"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "transparent_baseline_pre_result_repairs",
        sa.Column("receipt_sha256", sa.String(), primary_key=True),
        sa.Column(
            "source_audit_event_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{SCHEMA}.audit_events.id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column("source_batch_sha256", sa.String(), nullable=False, unique=True),
        sa.Column("target_batch_sha256", sa.String(), nullable=False, unique=True),
        sa.Column("source_dataset_lineage_id", sa.String(), nullable=False),
        sa.Column("target_dataset_lineage_id", sa.String(), nullable=False),
        sa.Column("target_recipe_version", sa.String(), nullable=False),
        sa.Column("source_backtest_ids_json", JSON, nullable=False),
        sa.Column("target_strategy_version_ids_json", JSON, nullable=False),
        sa.Column("verification_json", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "receipt_sha256 ~ '^[0-9a-f]{64}$' "
            "AND source_batch_sha256 ~ '^[0-9a-f]{64}$' "
            "AND target_batch_sha256 ~ '^[0-9a-f]{64}$' "
            "AND source_dataset_lineage_id ~ '^[0-9a-f]{64}$' "
            "AND target_dataset_lineage_id ~ '^[0-9a-f]{64}$'",
            name="ck_transparent_baseline_repair_sha256",
        ),
        sa.CheckConstraint(
            "source_batch_sha256 <> target_batch_sha256 "
            "AND source_dataset_lineage_id <> target_dataset_lineage_id",
            name="ck_transparent_baseline_repair_distinct_target",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(source_backtest_ids_json) = 'array' "
            "AND jsonb_array_length(source_backtest_ids_json) = 3 "
            "AND jsonb_typeof(target_strategy_version_ids_json) = 'array' "
            "AND jsonb_array_length(target_strategy_version_ids_json) = 3 "
            "AND jsonb_typeof(verification_json) = 'object'",
            name="ck_transparent_baseline_repair_evidence",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_transparent_baseline_pre_result_repairs_created",
        "transparent_baseline_pre_result_repairs",
        [sa.text("created_at DESC")],
        schema=SCHEMA,
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION quantlab.guard_transparent_baseline_pre_result_repair()
        RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'transparent baseline pre-result repair receipts are append-only';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_transparent_baseline_pre_result_repair
        BEFORE UPDATE OR DELETE ON quantlab.transparent_baseline_pre_result_repairs
        FOR EACH ROW EXECUTE FUNCTION quantlab.guard_transparent_baseline_pre_result_repair();
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_transparent_baseline_pre_result_repair "
        "ON quantlab.transparent_baseline_pre_result_repairs"
    )
    op.execute(
        "DROP FUNCTION IF EXISTS quantlab.guard_transparent_baseline_pre_result_repair()"
    )
    op.drop_index(
        "idx_transparent_baseline_pre_result_repairs_created",
        table_name="transparent_baseline_pre_result_repairs",
        schema=SCHEMA,
    )
    op.drop_table("transparent_baseline_pre_result_repairs", schema=SCHEMA)
