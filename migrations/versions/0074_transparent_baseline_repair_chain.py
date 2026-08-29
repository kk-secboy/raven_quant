"""Allow a governed same-lineage pre-result repair generation.

Revision ID: 0074_baseline_repair_chain
Revises: 0073_baseline_pre_result_repair
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0074_baseline_repair_chain"
down_revision: str | None = "0073_baseline_pre_result_repair"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"


def upgrade() -> None:
    # V1 required rematerialized data and therefore a fresh lineage. V2 fixes
    # runner applicability only and is required to keep the exact v7 dataset
    # and lineage. Distinct immutable batch hashes remain mandatory.
    op.drop_constraint(
        "ck_transparent_baseline_repair_distinct_target",
        "transparent_baseline_pre_result_repairs",
        schema=SCHEMA,
        type_="check",
    )
    op.create_check_constraint(
        "ck_transparent_baseline_repair_distinct_target",
        "transparent_baseline_pre_result_repairs",
        "source_batch_sha256 <> target_batch_sha256",
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_transparent_baseline_repair_distinct_target",
        "transparent_baseline_pre_result_repairs",
        schema=SCHEMA,
        type_="check",
    )
    op.create_check_constraint(
        "ck_transparent_baseline_repair_distinct_target",
        "transparent_baseline_pre_result_repairs",
        "source_batch_sha256 <> target_batch_sha256 "
        "AND source_dataset_lineage_id <> target_dataset_lineage_id",
        schema=SCHEMA,
    )
