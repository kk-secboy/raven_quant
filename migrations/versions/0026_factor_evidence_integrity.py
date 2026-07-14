"""Bind factor promotion to immutable Qlib evaluation evidence."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0026_factor_evidence_integrity"
down_revision: str | None = "0025_data_lineage_rollover"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for name, column_type in (
        ("values_sha256", sa.String()),
        ("promoted_evaluation_id", sa.String()),
        ("promotion_evidence_sha256", sa.String()),
        ("promoted_by", sa.String()),
        ("promoted_at", sa.DateTime(timezone=True)),
    ):
        op.add_column("factor_candidates", sa.Column(name, column_type), schema="quantlab")

    for name, column_type in (
        ("artifact_sha256", sa.String()),
        ("candidate_code_sha256", sa.String()),
        ("candidate_values_sha256", sa.String()),
        ("metrics_sha256", sa.String()),
        ("policy_json", postgresql.JSONB()),
        ("policy_sha256", sa.String()),
        ("evidence_sha256", sa.String()),
    ):
        op.add_column("factor_evaluations", sa.Column(name, column_type), schema="quantlab")


def downgrade() -> None:
    for name in (
        "evidence_sha256",
        "policy_sha256",
        "policy_json",
        "metrics_sha256",
        "candidate_values_sha256",
        "candidate_code_sha256",
        "artifact_sha256",
    ):
        op.drop_column("factor_evaluations", name, schema="quantlab")
    for name in (
        "promoted_at",
        "promoted_by",
        "promotion_evidence_sha256",
        "promoted_evaluation_id",
        "values_sha256",
    ):
        op.drop_column("factor_candidates", name, schema="quantlab")
