"""Add the v25 flag-less-audit-revision strategy runtime identity.

Revision ID: 0096_strategy_runtime_v25
Revises: 0095_strategy_runtime_v24
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0096_strategy_runtime_v25"
down_revision: str | None = "0095_strategy_runtime_v24"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
CONSTRAINT = "ck_strategy_versions_v25_runtime_identity"
RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-09-03-v25"
RUNNER_SHA256 = "79ff5cee1f046dea95bba13358005d975f0a83d7324c8f0fe18b07e5962b9615"
RUNTIME_BUNDLE_SHA256 = (
    "e9f26125b1675695b708ee48b8bacf65bea8c83abb56ac7741d799a407f48db1"
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
