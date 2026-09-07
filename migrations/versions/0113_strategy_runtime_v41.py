"""Add the v41 complete research mainline strategy runtime identity.

Revision ID: 0113_strategy_runtime_v41
Revises: 0112_strategy_runtime_v40
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0113_strategy_runtime_v41"
down_revision: str | None = "0112_strategy_runtime_v40"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
CONSTRAINT = "ck_strategy_versions_v41_runtime_identity"
RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-09-07-v41"
RUNNER_SHA256 = "bc0cfad1188eb3103295f33f959e3d4bfed5f8b17527610873796d0c4cacaf4c"
RUNTIME_BUNDLE_SHA256 = (
    "7d6b07b833e71f9309ac0211c084b49a3fbbe0c93b6611e4c426016bd930b5f6"
)


def _constraint() -> str:
    return (
        "(CASE WHEN COALESCE(config_json ->> 'recipe_version', '') = "
        f"'{RECIPE_VERSION}' THEN ("
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "AND evidence_mode = 'sealed_final_oos' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        f"'target_runner_sha256' = '{RUNNER_SHA256}' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        f"'target_runtime_bundle_sha256' = '{RUNTIME_BUNDLE_SHA256}' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ '^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE"
    )


def upgrade() -> None:
    op.create_check_constraint(
        CONSTRAINT,
        "strategy_versions",
        _constraint(),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_constraint(
        CONSTRAINT,
        "strategy_versions",
        schema=SCHEMA,
        type_="check",
    )
