"""Add the v32 YoY-mature style-gate strategy runtime identity.

Revision ID: 0103_strategy_runtime_v32
Revises: 0102_strategy_runtime_v31
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0103_strategy_runtime_v32"
down_revision: str | None = "0102_strategy_runtime_v31"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
CONSTRAINT = "ck_strategy_versions_v32_runtime_identity"
RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-09-04-v32"
RUNNER_SHA256 = "aa6a106935705f341608922f3ba7ef8cb7d8333412b19203917cf2b28613c9e8"
RUNTIME_BUNDLE_SHA256 = (
    "4e35515b1778b7c6e85cd99211581102f17873f6d59f4a710ed6bc603c945630"
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
