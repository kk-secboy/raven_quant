"""Allow a governed same-lineage pre-result repair generation.

Revision ID: 0074_baseline_repair_chain
Revises: 0073_baseline_pre_result_repair
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0074_baseline_repair_chain"
down_revision: str | None = "0073_baseline_pre_result_repair"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
_V2_CONTRACT = "transparent-baseline-pre-result-repair-v2"
_V2_GENERATION = "v7-to-v8-optimizer-applicability"
_V2_TARGET_RECIPE = "qlib-rdagent-single-mainline-2026-08-30-v8"
_V2_TARGET_RUNNER_SHA256 = (
    "fa1090deaa66ca77a045c1a872f7b6451043e116e717954533217908135c584e"
)
_V2_SOURCE_BACKTEST_IDS = (
    "8090c21aa11546bd9d59f732975afc25",
    "9c8a75ac646f452e8a5666bacd708936",
    "f51d7fa2f4fd463e97fd5f6990b3721c",
)


def _distinct_target_constraint() -> str:
    source_ids = "[" + ",".join(f'"{value}"' for value in _V2_SOURCE_BACKTEST_IDS) + "]"
    return (
        "source_batch_sha256 <> target_batch_sha256 AND ("
        "source_dataset_lineage_id <> target_dataset_lineage_id OR ("
        f"verification_json ->> 'receipt_contract_version' = '{_V2_CONTRACT}' AND "
        f"verification_json ->> 'repair_generation' = '{_V2_GENERATION}' AND "
        f"verification_json ->> 'target_runner_sha256' = '{_V2_TARGET_RUNNER_SHA256}' AND "
        f"target_recipe_version = '{_V2_TARGET_RECIPE}' AND "
        f"source_backtest_ids_json = '{source_ids}'::jsonb))"
    )


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
        _distinct_target_constraint(),
        schema=SCHEMA,
    )


def downgrade() -> None:
    same_lineage_repairs = op.get_bind().scalar(
        sa.text(
            "SELECT count(*) FROM quantlab.transparent_baseline_pre_result_repairs "
            "WHERE source_dataset_lineage_id = target_dataset_lineage_id"
        )
    )
    if int(same_lineage_repairs or 0) > 0:
        raise RuntimeError(
            "cannot downgrade 0074_baseline_repair_chain: append-only same-lineage "
            "v2 repair evidence exists"
        )
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
