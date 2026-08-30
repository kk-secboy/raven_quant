"""Allow the one sealed v15 single-member repair evidence row.

Revision ID: 0083_baseline_v15_evidence
Revises: 0082_baseline_v15_repair

The original repair registry admitted only three-member baseline batches.  The
v15 holding-age repair is deliberately a single short-horizon member, so this
revision keeps the historical three-member rule and opens one exact 1-to-1
exception bound to the immutable v6 receipt and source backtest.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0083_baseline_v15_evidence"
down_revision: str | None = "0082_baseline_v15_repair"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
_EVIDENCE_CONSTRAINT = "ck_transparent_baseline_repair_evidence"
_V6_RECEIPT_SHA256 = (
    "4cbd548de4cc38df791e4abeed59423c333769ffbc55f055291d457d4d332aeb"
)
_V6_CONTRACT = "transparent-baseline-pre-result-repair-v6"
_V6_GENERATION = "v13-to-v15-fill-aware-holding-age"
_V6_TARGET_RECIPE = "qlib-rdagent-single-mainline-2026-08-30-v15"
_V6_SOURCE_BACKTEST_ID = "51959d99d199410fa13a718658d03773"


def _three_member_evidence() -> str:
    return (
        "jsonb_array_length(source_backtest_ids_json) = 3 AND "
        "jsonb_array_length(target_strategy_version_ids_json) = 3"
    )


def _v6_single_member_evidence() -> str:
    source_ids = f"'[\"{_V6_SOURCE_BACKTEST_ID}\"]'::jsonb"
    return (
        "jsonb_array_length(source_backtest_ids_json) = 1 AND "
        "jsonb_array_length(target_strategy_version_ids_json) = 1 AND "
        f"receipt_sha256 = '{_V6_RECEIPT_SHA256}' AND "
        f"target_recipe_version = '{_V6_TARGET_RECIPE}' AND "
        f"source_backtest_ids_json = {source_ids} AND "
        "verification_json -> 'source_backtest_ids' = source_backtest_ids_json AND "
        "verification_json -> 'target_strategy_version_ids' = "
        "target_strategy_version_ids_json AND "
        "verification_json ->> 'receipt_sha256' = receipt_sha256 AND "
        "verification_json ->> 'receipt_contract_version' = "
        f"'{_V6_CONTRACT}' AND "
        "verification_json ->> 'repair_generation' = "
        f"'{_V6_GENERATION}'"
    )


def _evidence_constraint() -> str:
    return (
        "jsonb_typeof(source_backtest_ids_json) = 'array' AND "
        "jsonb_typeof(target_strategy_version_ids_json) = 'array' AND "
        "jsonb_typeof(verification_json) = 'object' AND (("
        + _three_member_evidence()
        + ") OR ("
        + _v6_single_member_evidence()
        + "))"
    )


def _legacy_evidence_constraint() -> str:
    return (
        "jsonb_typeof(source_backtest_ids_json) = 'array' AND "
        "jsonb_array_length(source_backtest_ids_json) = 3 AND "
        "jsonb_typeof(target_strategy_version_ids_json) = 'array' AND "
        "jsonb_array_length(target_strategy_version_ids_json) = 3 AND "
        "jsonb_typeof(verification_json) = 'object'"
    )


def upgrade() -> None:
    op.drop_constraint(
        _EVIDENCE_CONSTRAINT,
        "transparent_baseline_pre_result_repairs",
        schema=SCHEMA,
        type_="check",
    )
    op.create_check_constraint(
        _EVIDENCE_CONSTRAINT,
        "transparent_baseline_pre_result_repairs",
        _evidence_constraint(),
        schema=SCHEMA,
    )


def downgrade() -> None:
    v6_rows = op.get_bind().scalar(
        sa.text(
            "SELECT count(*) FROM quantlab.transparent_baseline_pre_result_repairs "
            "WHERE receipt_sha256 = :receipt"
        ),
        {"receipt": _V6_RECEIPT_SHA256},
    )
    if int(v6_rows or 0) > 0:
        raise RuntimeError(
            "cannot downgrade 0083_baseline_v15_evidence: immutable v6 "
            "single-member repair evidence exists"
        )
    op.drop_constraint(
        _EVIDENCE_CONSTRAINT,
        "transparent_baseline_pre_result_repairs",
        schema=SCHEMA,
        type_="check",
    )
    op.create_check_constraint(
        _EVIDENCE_CONSTRAINT,
        "transparent_baseline_pre_result_repairs",
        _legacy_evidence_constraint(),
        schema=SCHEMA,
    )
